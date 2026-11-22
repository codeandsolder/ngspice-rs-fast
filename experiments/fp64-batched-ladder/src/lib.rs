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
    pub async fn new(name: &str, shader: &str, shared_bytes: u32) -> Result<Self> {
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
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_compute_workgroup_storage_size = required_limits
            .max_compute_workgroup_storage_size
            .max(shared_bytes);
        ensure!(
            required_limits.max_compute_workgroup_storage_size
                <= adapter.limits().max_compute_workgroup_storage_size,
            "adapter lacks requested workgroup storage ({shared_bytes} bytes)"
        );
        let features =
            wgpu::Features::SHADER_F64 | (adapter.features() & wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: features,
                required_limits,
                ..Default::default()
            })
            .await?;
        let pipeline_start = Instant::now();
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
        println!(
            "pipeline {name}: {:.3} ms (creation API latency)",
            pipeline_start.elapsed().as_secs_f64() * 1000.
        );
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
    ) -> Result<(f64, f64, f64, f64)> {
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
        // Device timestamps bracket the actual compute pass, independently
        // of wall-clock synchronization and the timing of map/readback.
        let kernel_ms = if self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            let queries = self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("compute elapsed time"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolved = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp resolve"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp host read"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut readings = [0.; 7];
            for time in &mut readings {
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("timed solve"),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &queries,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }),
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &bindings, &[]);
                    pass.dispatch_workgroups(groups, 1, 1);
                }
                encoder.resolve_query_set(&queries, 0..2, &resolved, 0);
                encoder.copy_buffer_to_buffer(&resolved, 0, &read, 0, 16);
                let submission = self.queue.submit([encoder.finish()]);
                let slice = read.slice(..);
                let (tx, rx) = mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
                self.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(std::time::Duration::from_secs(30)),
                })?;
                rx.recv_timeout(std::time::Duration::from_secs(30))
                    .context("timestamp callback")??;
                {
                    let mapped = slice.get_mapped_range()?;
                    let ticks: &[u64] = bytemuck::cast_slice(&mapped);
                    let elapsed = ticks[1]
                        .checked_sub(ticks[0])
                        .context("GPU timestamp wrapped")?;
                    *time = elapsed as f64 * f64::from(self.queue.get_timestamp_period()) / 1e6;
                }
                read.unmap();
            }
            readings.sort_by(f64::total_cmp);
            readings[3]
        } else {
            f64::NAN
        };
        Ok((resident, amortized, transfer, kernel_ms))
    }
}
