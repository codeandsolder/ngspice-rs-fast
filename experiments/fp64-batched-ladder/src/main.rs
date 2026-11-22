//! Isolated FP64 feasibility probe. This is not a general SPARSE solver.
#![feature(portable_simd)]
#![forbid(unsafe_code)]

use std::{borrow::Cow, hint::black_box, simd::Simd, sync::mpsc, time::Instant};

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

const N: usize = 64;
const TILE: usize = 64;
const CELLS: usize = N * TILE;
const REPEATS: usize = 7;

struct Input {
    diagonal: Vec<f64>,
    rhs: Vec<f64>,
    expected: Vec<f64>,
}

impl Input {
    fn new(tiles: usize) -> Self {
        let len = tiles * CELLS;
        let mut diagonal = vec![0.; len];
        let mut rhs = vec![0.; len];
        let mut expected = vec![0.; len];
        for t in 0..tiles {
            let base = t * CELLS;
            for r in 0..N {
                for lane in 0..TILE {
                    let i = base + r * TILE + lane;
                    let seed = ((t * 23 + r * 13 + lane * 7) % 1021) as u32;
                    diagonal[i] = 2.125 + f64::from(seed) / 1021.;
                    let v = ((t * 19 + r * 11 + lane * 17) % 97) as u32;
                    expected[i] = (f64::from(v) - 48.) / 23.;
                }
            }
            for r in 0..N {
                for lane in 0..TILE {
                    let i = base + r * TILE + lane;
                    let left = if r == 0 { 0. } else { expected[i - TILE] };
                    let right = if r + 1 == N { 0. } else { expected[i + TILE] };
                    rhs[i] = diagonal[i] * expected[i] - left - right;
                }
            }
        }
        Self {
            diagonal,
            rhs,
            expected,
        }
    }

    fn check(&self, x: &[f64]) -> Result<f64> {
        ensure!(x.len() == self.expected.len(), "wrong output length");
        let mut worst = 0.0_f64;
        let mut backward = 0.0_f64;
        for (t, tile) in x.chunks_exact(CELLS).enumerate() {
            for r in 0..N {
                for lane in 0..TILE {
                    let j = r * TILE + lane;
                    let i = t * CELLS + j;
                    let v = tile[j];
                    ensure!(v.is_finite(), "nonfinite solution at {i}");
                    worst = worst.max((v - self.expected[i]).abs());
                    let left = if r == 0 { 0. } else { tile[j - TILE] };
                    let right = if r + 1 == N { 0. } else { tile[j + TILE] };
                    let ax = self.diagonal[i] * v - left - right;
                    let norm = self.diagonal[i].abs() * v.abs()
                        + left.abs()
                        + right.abs()
                        + self.rhs[i].abs();
                    backward = backward.max((ax - self.rhs[i]).abs() / norm.max(f64::MIN_POSITIVE));
                }
            }
        }
        ensure!(
            worst <= 2e-12 && backward <= 2e-13,
            "error={worst:e}, backward={backward:e}"
        );
        Ok(worst.max(backward))
    }
}

fn scalar(d: &[f64], b: &[f64], p: &mut [f64], x: &mut [f64]) {
    for base in (0..d.len()).step_by(CELLS) {
        for lane in 0..TILE {
            let mut prev_p = 0.;
            let mut prev_y = 0.;
            for r in 0..N {
                let i = base + r * TILE + lane;
                prev_p = 1. / (d[i] - prev_p);
                prev_y = (b[i] + prev_y) * prev_p;
                p[i] = prev_p;
                x[i] = prev_y;
            }
            let mut next = 0.;
            for r in (0..N).rev() {
                let i = base + r * TILE + lane;
                next = x[i] + p[i] * next;
                x[i] = next;
            }
        }
    }
}

// Row-major traversal keeps cache lines hot and allows independent vector
// divisions to overlap instead of completing one four-system chain at a time.
fn row_simd(d: &[f64], b: &[f64], p: &mut [f64], x: &mut [f64]) {
    type V = Simd<f64, 4>;
    for (((d, b), p), x) in d
        .chunks_exact(CELLS)
        .zip(b.chunks_exact(CELLS))
        .zip(p.chunks_exact_mut(CELLS))
        .zip(x.chunks_exact_mut(CELLS))
    {
        let mut prev_p = [V::splat(0.); TILE / 4];
        let mut prev_y = [V::splat(0.); TILE / 4];
        for r in 0..N {
            let offset = r * TILE;
            let dr = &d[offset..offset + TILE];
            let br = &b[offset..offset + TILE];
            let pr = &mut p[offset..offset + TILE];
            let xr = &mut x[offset..offset + TILE];
            for j in 0..TILE / 4 {
                let i = j * 4;
                prev_p[j] = V::splat(1.) / (V::from_slice(&dr[i..i + 4]) - prev_p[j]);
                prev_y[j] = (V::from_slice(&br[i..i + 4]) + prev_y[j]) * prev_p[j];
                prev_p[j].copy_to_slice(&mut pr[i..i + 4]);
                prev_y[j].copy_to_slice(&mut xr[i..i + 4]);
            }
        }
        let mut next = [V::splat(0.); TILE / 4];
        for r in (0..N).rev() {
            let offset = r * TILE;
            let pr = &p[offset..offset + TILE];
            let xr = &mut x[offset..offset + TILE];
            for (j, next) in next.iter_mut().enumerate() {
                let i = j * 4;
                *next = V::from_slice(&xr[i..i + 4]) + V::from_slice(&pr[i..i + 4]) * *next;
                next.copy_to_slice(&mut xr[i..i + 4]);
            }
        }
    }
}

// The persistent Rayon pool is initialized before median_ms measures it.
fn parallel_simd(d: &[f64], b: &[f64], p: &mut [f64], x: &mut [f64]) {
    d.par_chunks_exact(CELLS)
        .zip(b.par_chunks_exact(CELLS))
        .zip(p.par_chunks_exact_mut(CELLS))
        .zip(x.par_chunks_exact_mut(CELLS))
        .for_each(|(((d, b), p), x)| row_simd(d, b, p, x));
}

fn median_ms(mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?;
    let mut times = [0.; REPEATS];
    for t in &mut times {
        let start = Instant::now();
        f()?;
        *t = start.elapsed().as_secs_f64() * 1000.;
    }
    times.sort_by(f64::total_cmp);
    Ok(times[REPEATS / 2])
}

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl Gpu {
    async fn new() -> Result<Self> {
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
            label: Some("FP64 ladder solver"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("ladder.wgsl"))),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FP64 ladder solver"),
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

    fn trial(&self, input: &Input, tiles: u32) -> Result<(f64, f64, f64)> {
        let size = (input.diagonal.len() * size_of::<f64>()) as u64;
        ensure!(size <= u64::from(self.device.limits().max_storage_buffer_binding_size));
        let init = |name, data: &[f64]| {
            self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        let allocate = |name, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let diagonal = init("diagonal", &input.diagonal);
        let rhs = init("RHS", &input.rhs);
        let p = allocate("reciprocal", wgpu::BufferUsages::STORAGE);
        let x = allocate(
            "solution",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = allocate(
            "readback",
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ladder buffers"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: diagonal.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x.as_entire_binding(),
                },
            ],
        });
        let dispatch = |readback: bool| -> Result<()> {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bindings, &[]);
                pass.dispatch_workgroups(tiles, 1, 1);
            }
            if readback {
                encoder.copy_buffer_to_buffer(&x, 0, &staging, 0, size);
            }
            self.queue.submit([encoder.finish()]);
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
            Ok(())
        };
        let resident = median_ms(|| dispatch(false))?;
        let mut host_result = vec![0.; input.expected.len()];
        let transfer = median_ms(|| {
            self.queue
                .write_buffer(&diagonal, 0, bytemuck::cast_slice(&input.diagonal));
            self.queue
                .write_buffer(&rhs, 0, bytemuck::cast_slice(&input.rhs));
            dispatch(true)?;
            let slice = staging.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            self.device.poll(wgpu::PollType::wait_indefinitely())?;
            rx.recv().context("readback callback dropped")??;
            {
                let mapped = slice.get_mapped_range()?;
                host_result.copy_from_slice(bytemuck::cast_slice(&mapped));
                black_box(&host_result);
            }
            staging.unmap();
            Ok(())
        })?;
        let measured_error = input.check(&host_result)?;
        Ok((resident, transfer, measured_error))
    }
}

fn main() -> Result<()> {
    let gpu = pollster::block_on(Gpu::new())?;
    println!("persistent CPU workers: {}", rayon::current_num_threads());
    println!("nodes={N}, tile={TILE}, repeats={REPEATS}");
    println!(
        "systems,scalar_ms,row_simd_ms,parallel_row_simd_ms,gpu_resident_ms,gpu_roundtrip_ms,max_error"
    );
    for tiles in [1_u32, 4, 16, 64, 256, 1024] {
        let input = Input::new(tiles as usize);
        let mut p = vec![0.; input.diagonal.len()];
        let mut x = vec![0.; input.diagonal.len()];
        let mut error = 0.0_f64;
        let mut timings = Vec::new();
        for solve in [
            scalar as fn(&[f64], &[f64], &mut [f64], &mut [f64]),
            row_simd,
            parallel_simd,
        ] {
            timings.push(median_ms(|| {
                solve(
                    black_box(&input.diagonal),
                    black_box(&input.rhs),
                    black_box(&mut p),
                    black_box(&mut x),
                );
                Ok(())
            })?);
            error = error.max(input.check(&x)?);
        }
        let (resident, transfer, gpu_error) = gpu.trial(&input, tiles)?;
        println!(
            "{},{:.4},{:.4},{:.4},{resident:.4},{transfer:.4},{:.3e}",
            tiles * TILE as u32,
            timings[0],
            timings[1],
            timings[2],
            error.max(gpu_error)
        );
    }
    Ok(())
}
