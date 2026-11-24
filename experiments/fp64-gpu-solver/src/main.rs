#![feature(portable_simd)]

mod gpu;
mod gpu_sparse;
mod solver;
mod sparse;

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn bench_cpu(input: &[f64], output: &mut [f64], solve: fn(&[f64], &mut [f64])) -> Duration {
    let repeats = (30_000_000usize / output.len()).clamp(3, 200);
    let start = Instant::now();
    for _ in 0..repeats {
        solve(black_box(input), black_box(output));
    }
    start.elapsed() / repeats as u32
}

fn bench_sparse_cpu(plan: &sparse::Plan, input: &[f64], output: &mut [f64]) -> Duration {
    let repeats = (50_000usize / (input.len() / plan.input_stride())).clamp(3, 30);
    let start = Instant::now();
    for _ in 0..repeats {
        let failures = plan.solve_cpu(black_box(input), black_box(output));
        assert_eq!(failures, 0);
    }
    start.elapsed() / repeats as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = gpu::Gpu::new()?;
    println!(
        "GPU: {} / Vulkan shaderFloat64 / one FP64 system per invocation",
        gpu.name
    );
    println!(
        "tri_order={}, matrix=[system][row][lower,diag,upper,rhs] (binary64)",
        solver::N
    );
    println!(
        "tri_systems,scalar_us,simd4_us,gpu_resident_us,gpu_round_trip_us,max_abs_residual,max_abs_error"
    );
    for count in [64, 256, 1024, 4096, 16384, 65536] {
        let (input, truth) = solver::generate(count);
        let mut output = vec![0.0; truth.len()];
        let scalar = bench_cpu(&input, &mut output, solver::scalar);
        let (rs, es) = solver::maximum_residual(&input, &output, &truth);
        let simd = bench_cpu(&input, &mut output, solver::simd4);
        let (rv, ev) = solver::maximum_residual(&input, &output, &truth);
        let samples = gpu.run(&input, &truth, 25)?;
        let residual = rs.max(rv).max(samples.residual);
        let error = es.max(ev).max(samples.deviation);
        if residual > 1e-10 || error > 1e-10 {
            return Err(format!("tri batch {count}: residual {residual:e} error {error:e}").into());
        }
        println!(
            "{count},{:.3},{:.3},{:.3},{:.3},{residual:e},{error:e}",
            scalar.as_secs_f64() * 1e6,
            simd.as_secs_f64() * 1e6,
            samples.resident.as_secs_f64() * 1e6,
            samples.round_trip.as_secs_f64() * 1e6
        );
    }

    let plan = sparse::Plan::new(32);
    let sparse_gpu = gpu_sparse::SparseGpu::new(&gpu);
    println!(
        "sparse_order={},original_nnz={},filled_lu_nnz={},shared_plan_words={}",
        plan.n,
        plan.initial_nonzeros(),
        plan.nnz,
        plan.words().len()
    );
    println!(
        "sparse_systems,cpu_fixed_us,gpu_resident_us,gpu_round_trip_us,max_abs_residual,max_abs_error,gpu_pivot_failures"
    );
    for count in [64, 256, 1024, 4096] {
        let (input, truth) = plan.generate(count);
        let mut output = vec![0.0; count * plan.output_stride()];
        let cpu = bench_sparse_cpu(&plan, &input, &mut output);
        let (rs, es, failures) = plan.validate(&input, &output, &truth);
        assert_eq!(failures, 0);
        let samples = sparse_gpu.run(&gpu, &plan, &input, &truth, 8)?;
        let residual = rs.max(samples.residual);
        let error = es.max(samples.deviation);
        if residual > 1e-10 || error > 1e-10 || samples.failures != 0 {
            return Err(format!(
                "sparse batch {count}: residual {residual:e} error {error:e} GPU failures {}",
                samples.failures
            )
            .into());
        }
        println!(
            "{count},{:.3},{:.3},{:.3},{residual:e},{error:e},{}",
            cpu.as_secs_f64() * 1e6,
            samples.resident.as_secs_f64() * 1e6,
            samples.round_trip.as_secs_f64() * 1e6,
            samples.failures
        );
    }
    // Only the first lane is singular; all other lanes remain independent.
    let (mut bad, truth) = plan.generate(64);
    bad[plan.diagonal[0] as usize] = 0.0;
    let mut output = vec![0.0; 64 * plan.output_stride()];
    assert_eq!(plan.solve_cpu(&bad, &mut output), 1);
    let failed_gpu = sparse_gpu.run(&gpu, &plan, &bad, &truth, 1)?;
    if failed_gpu.failures != 1 {
        return Err(format!(
            "GPU missed one deliberately singular fixed-pivot candidate: {}",
            failed_gpu.failures
        )
        .into());
    }
    println!(
        "sparse_zero_pivot_status=correct isolated GPU failures={}",
        failed_gpu.failures
    );
    Ok(())
}
