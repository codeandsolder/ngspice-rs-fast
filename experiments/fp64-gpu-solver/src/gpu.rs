use std::{
    borrow::Cow,
    mem,
    sync::mpsc,
    time::{Duration, Instant},
};

use wgpu::util::DeviceExt;

use crate::solver::{self, N};

pub struct Gpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pub name: String,
}

pub struct Measurement {
    pub resident: Duration,
    pub round_trip: Duration,
    pub residual: f64,
    pub deviation: f64,
}

impl Gpu {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        if !adapter.features().contains(wgpu::Features::SHADER_F64) {
            return Err(format!(
                "{} does not support Vulkan SHADER_F64",
                adapter.get_info().name
            )
            .into());
        }
        let name = format!(
            "{} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SHADER_F64,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            }))?;
        let shader = include_bytes!(concat!(env!("OUT_DIR"), "/tridiagonal.spv"));
        assert_eq!(shader.len() % 4, 0);
        let spirv = shader
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        // Native SPIR-V passthrough is required because WGSL cannot represent f64.
        // The source is committed, reproducibly compiled by build.rs and verified
        // with spirv-val; backend validation is bypassed only for this experiment.
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("FP64 tridiagonal"),
                spirv: Some(Cow::Owned(spirv)),
                ..Default::default()
            })
        };
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FP64 matrices and output"),
            entries: &[storage(0, true), storage(1, false)],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FP64 tridiagonal layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FP64 tridiagonal"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device,
            queue,
            layout,
            pipeline,
            name,
        })
    }

    pub fn run(
        &self,
        input: &[f64],
        truth: &[f64],
        repeats: usize,
    ) -> Result<Measurement, Box<dyn std::error::Error>> {
        let count = truth.len() / N;
        if count == 0
            || count > u32::MAX as usize
            || input.len() != truth.len() * 4
            || repeats == 0
            || repeats > u32::MAX as usize
        {
            return Err("invalid batch size or repeat count".into());
        }
        let start = Instant::now();
        let matrices = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resident FP64 matrices"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
        let bytes = (truth.len() * mem::size_of::<f64>()) as u64;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident FP64 solution"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FP64 solution readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let binding = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FP64 tridiagonal batch"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        let dispatch = |label: &'static str,
                        times: usize,
                        copy: bool|
         -> Result<(), Box<dyn std::error::Error>> {
            let mut cmd = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            for _ in 0..times {
                let mut pass = cmd.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &binding, &[]);
                pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
            }
            if copy {
                cmd.copy_buffer_to_buffer(&output, 0, &readback, 0, bytes);
            }
            let index = self.queue.submit([cmd.finish()]);
            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(index),
            })?;
            Ok(())
        };

        dispatch("FP64 cold+copy", 1, true)?;
        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
        })?;
        receiver.recv()??;
        let mapped = slice.get_mapped_range();
        let result = mapped
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().expect("eight bytes")))
            .collect::<Vec<_>>();
        let (residual, deviation) = solver::maximum_residual(input, &result, truth);
        drop(mapped);
        readback.unmap();
        let round_trip = start.elapsed();

        // Warm resident dispatch: these buffers stay on the GPU; no readback,
        // allocation or host upload inside the measured region.
        dispatch("FP64 warm-up", 3, false)?;
        let begin = Instant::now();
        dispatch("FP64 resident dispatch", repeats, false)?;
        let resident = begin.elapsed() / repeats as u32;

        Ok(Measurement {
            resident,
            round_trip,
            residual,
            deviation,
        })
    }
}
