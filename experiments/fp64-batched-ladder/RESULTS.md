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
