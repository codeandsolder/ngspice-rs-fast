use std::{
    borrow::Cow,
    mem,
    sync::mpsc,
    time::{Duration, Instant},
};

use wgpu::util::DeviceExt;

use crate::{gpu::Gpu, sparse::Plan};

pub struct SparseGpu {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct Measurement {
    pub resident: Duration,
    pub round_trip: Duration,
    pub residual: f64,
    pub deviation: f64,
    pub failures: usize,
}

impl SparseGpu {
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        let shader = include_bytes!(concat!(env!("OUT_DIR"), "/sparse_fixed_pivot.spv"));
        let words = shader
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .collect::<Vec<_>>();
        // Safe for this in-repo, spirv-val verified module and descriptor layout.
        // The native f64 shader cannot pass through WGSL or the Naga frontend.
        let module = unsafe {
            gpu.device
                .create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some("FP64 symbolic-plan sparse LU"),
                    spirv: Some(Cow::Owned(words)),
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
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("FP64 numeric + symbolic plan"),
                entries: &[
                    storage(0, true),
                    storage(1, false),
                    storage(2, true),
                    storage(3, false),
                ],
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("FP64 sparse elimination layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
        let pipeline = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("FP64 sparse elimination"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        Self { pipeline, layout }
    }

    pub fn run(
        &self,
        gpu: &Gpu,
        plan: &Plan,
        input: &[f64],
        truth: &[f64],
        repeats: usize,
    ) -> Result<Measurement, Box<dyn std::error::Error>> {
        let count = truth.len() / plan.n;
        if count == 0
            || count > u32::MAX as usize
            || input.len() != count * plan.input_stride()
            || truth.len() != count * plan.n
            || repeats == 0
            || repeats > u32::MAX as usize
        {
            return Err("invalid sparse numerical batch".into());
        }
        let start = Instant::now();
        let words = plan.words();
        let matrices = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FP64 sparse numeric input"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let shared_plan = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shared immutable sparse symbolic plan"),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let scratch = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("independent candidate LU scratch"),
            size: input.len() as u64 * mem::size_of::<f64>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let output_bytes = count as u64 * plan.output_stride() as u64 * 8;
        let output = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FP64 sparse solution + status"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FP64 sparse readback"),
            size: output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FP64 sparse batch"),
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: shared_plan.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scratch.as_entire_binding(),
                },
            ],
        });

        let dispatch = |label: &'static str,
                        times: usize,
                        copy: bool|
         -> Result<(), Box<dyn std::error::Error>> {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            for _ in 0..times {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
            }
            if copy {
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
            }
            let index = gpu.queue.submit([encoder.finish()]);
            gpu.device.poll(wgpu::PollType::Wait {
                submission_index: Some(index),
            })?;
            Ok(())
        };

        dispatch("FP64 sparse cold and download", 1, true)?;
        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        gpu.device.poll(wgpu::PollType::Wait {
            submission_index: None,
        })?;
        receiver.recv()??;
        let mapped = slice.get_mapped_range();
        let output = mapped
            .chunks_exact(8)
            .map(|bytes| f64::from_le_bytes(bytes.try_into().expect("eight bytes")))
            .collect::<Vec<_>>();
        let (residual, deviation, failures) = plan.validate(input, &output, truth);
        drop(mapped);
        readback.unmap();
        let round_trip = start.elapsed();

        dispatch("FP64 sparse warm-up", 2, false)?;
        let begin = Instant::now();
        dispatch("FP64 sparse GPU resident", repeats, false)?;
        let resident = begin.elapsed() / repeats as u32;
        Ok(Measurement {
            resident,
            round_trip,
            residual,
            deviation,
            failures,
        })
    }
}
