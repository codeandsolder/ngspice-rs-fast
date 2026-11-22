//! Shared measurement and FP64 compute dispatch for standalone experiments.
#![forbid(unsafe_code)]

use anyhow::{Context, Result, ensure};
use std::{borrow::Cow, hint::black_box, sync::mpsc, time::Instant};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

/// Median wall time after one warmup. No allocations or validation in callers'
/// timed closures, except allocations explicitly included in the operation.
pub fn median_ms(mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    let mut times = [0.; 7];
    for t in &mut times {
        let start = Instant::now();
        f()?;
        *t = start.elapsed().as_secs_f64() * 1000.;
    }
    times.sort_by(f64::total_cmp);
    Ok(times[3])
}

pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl Gpu {
    pub async fn new(name: &str, shader: &str) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await?;
        let info = adapter.get_info();
        println!(
            "adapter: {} ({:?}, {})",
            info.name, info.backend, info.driver
        );
        ensure!(
            adapter.features().contains(wgpu::Features::SHADER_F64),
            "adapter lacks FP64"
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SHADER_F64,
                ..Default::default()
            })
            .await?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module: &module,
            entry_point: Some("solve"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    /// Bind immutable input buffers, mutable workspaces (last is result), then
    /// optional shared symbolic plan. Measure single resident dispatch, eight
    /// resident dispatches per submission (per-dispatch average), and full
    /// roundtrip. The returned host output is from the final timed roundtrip.
    pub fn trial(
        &self,
        inputs: &[&[f64]],
        work_sizes: &[usize],
        plan: Option<&[[u32; 4]]>,
        output: &mut [f64],
        groups: u32,
    ) -> Result<(f64, f64, f64)> {
        let limits = self.device.limits();
        let bytes = |n: usize| -> Result<u64> {
            let size = n
                .checked_mul(size_of::<f64>())
                .context("buffer size overflow")? as u64;
            ensure!(
                size > 0 && size <= u64::from(limits.max_storage_buffer_binding_size),
                "buffer past storage binding limit"
            );
            Ok(size)
        };
        ensure!(
            work_sizes.last() == Some(&output.len()),
            "result shape mismatch"
        );
        ensure!(
            groups > 0 && groups <= limits.max_compute_workgroups_per_dimension,
            "invalid dispatch count"
        );
        let mut buffers = Vec::new();
        for input in inputs {
            bytes(input.len())?;
            buffers.push(self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("immutable numeric input"),
                contents: bytemuck::cast_slice(*input),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
        }
        for (i, &n) in work_sizes.iter().enumerate() {
            let mut usage = wgpu::BufferUsages::STORAGE;
            if i + 1 == work_sizes.len() {
                usage |= wgpu::BufferUsages::COPY_SRC;
            }
            buffers.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("numeric workspace"),
                size: bytes(n)?,
                usage,
                mapped_at_creation: false,
            }));
        }
        let result = &buffers[inputs.len() + work_sizes.len() - 1];
        let plan_buffer = plan.map(|plan| {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("shared symbolic plan"),
                contents: bytemuck::cast_slice(plan),
                usage: wgpu::BufferUsages::STORAGE,
            })
        });
        let entries: Vec<_> = buffers
            .iter()
            .chain(plan_buffer.iter())
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch numeric buffers"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let size = bytes(output.len())?;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let dispatch = |count: u32, readback: bool| -> Result<()> {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            // Separate passes ensure storage hazards are tracked between repeats.
            for _ in 0..count {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            if readback {
                encoder.copy_buffer_to_buffer(result, 0, &staging, 0, size);
            }
            let index = self.queue.submit([encoder.finish()]);
            self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(index),
                timeout: Some(std::time::Duration::from_secs(30)),
            })?;
            Ok(())
        };
        let resident = median_ms(|| dispatch(1, false))?;
        let amortized = median_ms(|| dispatch(8, false))? / 8.;
        let transfer = median_ms(|| {
            for (buffer, input) in buffers.iter().zip(inputs) {
                self.queue
                    .write_buffer(buffer, 0, bytemuck::cast_slice(*input));
            }
            dispatch(1, true)?;
            let slice = staging.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(30)),
            })?;
            rx.recv_timeout(std::time::Duration::from_secs(30))
                .context("readback callback")??;
            {
                let mapped = slice.get_mapped_range()?;
                output.copy_from_slice(bytemuck::cast_slice(&mapped));
                black_box(&output);
            }
            staging.unmap();
            Ok(())
        })?;
        Ok((resident, amortized, transfer))
    }
}
