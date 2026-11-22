# Experimental measurements

## 2026-11-22 — first FP64 ladder run

Code `e73a3f0` (methodology `50dc7bb`). Acer, Ryzen 7 4700U (8 Zen 2 cores) with
RADV Renoir integrated Radeon graphics, Mesa 26.3.1, Vulkan 1.4.363,
`shaderFloat64=true`, subgroup size 64. Pinned Rust nightly-2026-11-20,
native release binary; all backends binary64. Concurrent build activity was
present, so these are feasibility measurements, not an isolated hardware
performance characterization.

64-node independently varying grounded tridiagonal systems, one warm-up plus
seven measurements per backend. CPU and GPU use same tiled inputs, reusable
workspaces. GPU resident includes a synchronized command submission, roundtrip
includes input writes, result readback and copy. CPU baseline at this revision
was strided by system (or four systems) and recreated scoped threads per solve.
Later revisions correct those limitations.

| Systems | Scalar ms | SIMD ms | Threaded SIMD ms | Resident GPU ms | Roundtrip GPU ms | Max absolute error |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.0323 | 0.0083 | 2.7787 | 1.0246 | 1.1088 | 1.776e-15 |
| 256 | 0.1792 | 0.0430 | 1.9614 | 1.1109 | 1.4638 | 2.665e-15 |
| 1024 | 2.8343 | 0.3345 | 12.8887 | 1.0355 | 1.7983 | 2.665e-15 |
| 4096 | 4.0702 | 1.8856 | 11.5215 | 2.6960 | 3.8111 | 2.665e-15 |
| 16384 | 17.3715 | 8.3043 | 18.1845 | 4.3345 | 11.2067 | 2.887e-15 |
| 65536 | 148.1165 | 42.4572 | 41.4106 | 15.6921 | 44.7260 | 2.887e-15 |

GPU resident performance at 65,536 systems was 2.64 times faster than the
original threaded SIMD baseline. Including transfers erased that advantage.
The synchronized submission overhead dominated small cases. Do not infer
whole-simulator speedup; the CPU baseline needs a contiguous traversal and
persistent workers. Both issues addressed in `baa9476`.

## Second experiment (implementation)

`a266929` shares a resident GPU runner between experiments and adds an eight-pass
per-submission measurement (amortized dispatch+wait time). `a955a06` adds a
complex-valued 8x8 grounded RC mesh with varying imaginary shunt admittance.
The lower-triangle symbolic elimination plan is built once, including fill-in,
then a packed instruction tape drives the numeric factorization and
forward/back-substitution. CPU and GPU execute the same schedule; GPU uses
one invocation per system, real/imaginary components each contiguous across
64-system tiles. Both solution error against an independently specified vector
and normalized backward error against the original matrix are checked.

The mesh intentionally relies on strict real diagonal dominance, no row/column
pivoting and moderate coefficient magnitudes. It is not yet suitable for
arbitrary MNA systems. The production solver needs scale-aware residuals and
refinement (#104), a validated reusable symbolic plan, and lane-specific pivot
failure/reorder handling (#34, #91, #97, #98). Measurements of this second
experiment will need to be evaluated with those caveats.

## 2026-11-22 — contiguous SIMD, persistent workers and complex mesh

Implementation `a955a06`, same Acer/RADV, pinned nightly. Eight persistent CPU
workers. Concurrent builds remained active. GPU times are synchronized single
submission, eight dispatches averaged, and full transfer respectively.

| Ladder systems | Contiguous SIMD ms | Pool SIMD ms | GPU resident ms | GPU amortized ms | GPU roundtrip ms |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.0029 | 0.0030 | 0.9934 | 0.9776 | 2.9181 |
| 256 | 0.0140 | 0.0242 | 3.4913 | 1.9213 | 3.1700 |
| 1024 | 0.0580 | 0.0802 | 1.3675 | 0.7706 | 5.9899 |
| 4096 | 1.0543 | 1.1926 | 1.4668 | 1.1086 | 3.6546 |
| 16384 | 5.3098 | 3.9844 | 4.1314 | 4.1992 | 12.1300 |
| 65536 | 16.7248 | 43.5561 | 14.8076 | 10.1180 | 49.9679 |

Worst solution error 2.887e-15. Original comparison materially understated CPU
throughput. At the largest size contiguous single-thread SIMD was 2.54x faster
than previous strided SIMD. Thread-pool results are noisy under fleet load.

First complex 8x8 mesh (64 nodes, 176 original lower-triangle entries,
519 after fill, 1967 updates, 3396 tape instructions):

| Systems | Scalar ms | SIMD ms | Pool SIMD ms | GPU resident ms | GPU amortized ms | GPU roundtrip ms |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.9526 | 0.1896 | 0.1987 | 21.6080 | 6.1527 | 6.5528 |
| 256 | 4.6316 | 1.0044 | 1.1338 | 20.8274 | 5.9407 | 11.7666 |
| 1024 | 26.4275 | 7.2404 | 13.3453 | 21.5627 | 7.3564 | 9.1494 |
| 4096 | 67.0461 | 16.7723 | 19.3857 | 37.7704 | 24.0431 | 34.2570 |

Worst solution error 4.990e-15, normalized backward error 6.200e-16.
GPU tape interpretation and global factor access are not competitive.

## 2026-11-22 — grouped elimination, mixed precision and cooperative LDS

Implemented `febff2e`, `162e621`, `6f0011b`, `f5db209` (actual last source
revision of this run). Reusable LDL^T factors separate factorization and
triangular solves, with CPU SIMD and GPU grouped variants using the same
symbolic range plan. CPU factor-reuse regression passed. Mixed GPU stores
FP32 factors and correction RHS; original operator, residual evaluation and
solution accumulation remain FP64. Componentwise 1e-14 scaled L1 residual
criterion, at most five corrections; host independently checks every solution
and normalized residual. Cooperative GPU FP64 stores four systems in 37,824 B
workgroup memory, 16 workers per system, four barriers per factor column.

| Systems | Tape CPU SIMD ms | Grouped pool SIMD ms | Tape GPU amortized ms | Grouped GPU amortized ms | Mixed GPU amortized ms | LDS GPU amortized ms |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.1652 | 0.1539 | 4.3016 | 2.2303 | 4.0803 | 2.7371 |
| 256 | 0.6561 | 0.4370 | 4.7849 | 4.6597 | 6.4505 | 9.2853 |
| 1024 | 4.1864 | 2.6359 | 5.8301 | 4.8439 | 6.7792 | 22.2007 |
| 4096 | 15.8317 | 5.1687 | 21.2228 | 20.7059 | 17.4955 | 63.3574 |

Mixed maximum absolute error by batch: 1.490e-15, 1.573e-15, 1.831e-15,
1.897e-15; normalized backward error <= 1.768e-16. Across all variants:
max solution error 4.990e-15, max backward error 6.200e-16. Mixed GPU
approximately 15.5% less amortized time than grouped FP64 at 4096 systems,
but 3.38x slower than grouped pool CPU. At smaller batches refinement
costs more than FP32 saves. Cooperative LDS is substantially slower than
both grouped and tape at large batch, likely because reduced independent
systems per workgroup and repeated synchronization outweigh saved traffic.
This is not proof of a general limit on GPU factorization or mixed precision.

Next implementation `58eda7b`, `8c2b792`, `27d4880`: compile the symbolic
schedule into a WGSL straight-line numeric kernel, retaining active values
in private variables and only saving eliminated factors for backsolve.
Measure actual GPU compute pass timestamps separately from synchronized
wall clock and pipeline creation API latency. The explicit --adapter selector
allows comparative measurement on Asus's RTX 2060 Max-Q as well as Renoir.
