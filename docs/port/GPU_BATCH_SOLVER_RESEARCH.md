# FP64 GPU batch research (#101)

**Decision:** retain the GPU prototype as an isolated research experiment.
Do not introduce a GPU dependency or change the ordinary AC solver until an
end-to-end, representative ensemble demonstrates a useful crossover.

## Existing GPU evidence

The reproducible implementation and raw benchmark tables are in
[the standalone experiment at c7b731a](https://github.com/codeandsolder/ngspice-rs-fast/tree/c7b731a/experiments/fp64-batched-ladder)
and its [results](https://github.com/codeandsolder/ngspice-rs-fast/blob/c7b731a/experiments/fp64-batched-ladder/RESULTS.md).
These implement native FP64 WGSL through wgpu/Vulkan, with row-major
tiles across independent systems. They compare a contiguous portable SIMD
CPU path (single thread and persistent worker pool), grouped symbolic GPU
elimination, mixed FP32 factor/FP64 residual correction, cooperative shared
memory and topology-specialized kernels.

On the tested RADV Renoir GPU with FP64 support, seven-trial measurements
for 4096 independent complex 64-node RC meshes were:

| Method | Time (ms) |
| --- | ---: |
| Contiguous pooled CPU SIMD | 6.5240 |
| Grouped GPU, resident amortized | 20.9228 |
| Specialized GPU, resident amortized | 6.6934 |
| Specialized GPU, including transfer | 33.9202 |

The specialized shader was 122,323 bytes and pipeline construction took
14.44 s. Worst solution and normalized backward errors across variants
were 4.990e-15 and 6.200e-16 respectively. The mixed precision variant
separately demonstrated a worst normalized backward error of 1.768e-16
after FP64 residual correction. Results were collected under concurrent
load and do not imply a production SPICE speedup.

These synthetic matrices have benign pivots. They do not establish safety
on indefinite or nonsymmetric MNA, or a general GPU sparse solver. A later
real-only packed fixed-pivot LU experiment also demonstrated per-lane
zero-pivot failure reporting, but used strided system-major input and an
unoptimized CPU build. Those timings are not used as evidence of a
throughput crossover.

## Production numerical baseline

[The AC batching design](AC_BATCHING.md) specifies the production
independent complex solver. It retains a validated sparse elimination
plan, handles incompatible topology and reorders failed pivots in scalar
lanes. Existing numerical regressions exercise singular-lane isolation,
zero-pivot recovery, range extremes, transposed solves, and a parsed RC
ladder against the ordinary scalar AC analysis.

The non-published `experiments/fp64-gpu-solver` workspace member supplements those tests
with a CPU baseline for parsed RLC ladders and meshes, including inductive
and ideal voltage-source branch-current constraints and nonsymmetric
controlled-source stamps. It uses the actual DC bias and small-signal
operators, not a synthetic diagonally dominant matrix. Its dependency
points toward the simulator only; production packages do not
depend on the experiment. It shares the workspace lockfile and is covered
by workspace CI, but is excluded from default-member builds. There are no
additional registry dependencies.

The benchmark assembles matrices outside the timed region, sweeps independent
frequencies from 100 Hz to below 10 MHz, and compares independent scalar
factor/solve (reusing symbolic ordering) with the production four-lane
packed factor/solve. It reports the median of seven trials after warmup,
including output allocations. It verifies every point against the scalar
reference and independently accumulates the componentwise normalized
backward residual from the original static and dynamic circuit stamps.
This measures numerical execution, not end-to-end deck evaluation.

Run with the pinned nightly and the existing managed compiler cache,
using an external target directory:

```sh
cargo +nightly test --locked -p ngspice-ac-ensemble-experiment
cargo +nightly run --release --locked -p ngspice-ac-ensemble-experiment
cargo +nightly test --locked --lib independent_
```

## GPU integration gate

A future prototype for these physical circuits must retain resident matrix
and RHS data, use immutable symbolic topology and independent numeric
states, preserve input-point order, and explicitly report lane failures.
All nonfinite results, unacceptable residuals and failed/unsafe pivots
require checked CPU fallback rather than silent approximate results.
A full comparison needs matrix assembly, device physics, factor/solve,
objective extraction, pipeline amortization and transfer cost in its
end-to-end accounting. The default analysis remains scalar.

## Physically assembled CPU measurements (2026-11-25)

Pinned nightly, release profile, native Renoir host CPU, seven-trial
medians under concurrent build load. The values compare API-level
factor/solve only, not necessarily SIMD execution of every lane.

| Circuit unknowns | Frequencies | Scalar (ms) | Independent batch (ms) |
| ---: | ---: | ---: | ---: |
| 24 | 64 | 0.665 | 0.268 |
| 24 | 256 | 1.261 | 1.108 |
| 24 | 1024 | 5.029 | 7.611 |
| 24 | 4096 | 22.929 | 23.338 |
| 178 | 64 | 2.836 | 2.964 |
| 178 | 256 | 18.370 | 14.235 |
| 178 | 1024 | 50.138 | 101.338 |
| 178 | 4096 | 310.394 | 289.691 |

Both APIs produced identical coordinates in these runs, with worst
componentwise physical-stamp backward error 6.057e-12. The separate
closed-form series-RL/parallel-RC transfer test passed as well. No
consistent CPU batch crossover was established; this is an appropriate
reference for future physical GPU experiments, not a GPU measurement.
