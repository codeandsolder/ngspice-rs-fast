# Checked contiguous real dense kernels (#145)

Real dense row-major storage is separate from the sparse MNA solver and complex RF algebra (#102). Checked shape/allocation contracts (#42) remain the boundary: zero-dimensional rectangular shapes are valid, dimensions and output lengths must match, and no kernel operates on an unvalidated buffer. In particular, a zero-column product overwrites its output with positive zero; a zero-element transpose performs no traversal even for a huge logical dimension.

## Operations and arithmetic

- Matrix-vector multiplication visits four independent rows together. LLVM can vectorize the products and additions **across rows**, rather than reassociate a single dot product. Within each row the original sequence of separately rounded products and additions from positive zero is unchanged; the remaining one to three rows use the same scalar reduction. The allocating operation and `mul_vector_into` share that kernel. The latter checks both vector lengths before changing any element.
- A transpose uses a tiny-input direct path, contiguous columns for one row tile and a 32-by-32 coefficient tile otherwise (16 KiB of source plus destination coefficients). The traversal writes contiguous row intervals of the destination, controls cache conflicts and handles partial tiles; it copies single-row/column inputs directly. The allocating operation and `transpose_into` share that kernel. The latter checks the **shape**, not only element count, before writing.
- The Jacobi symmetric-eigenvalue fallback for mutual-inductor validation still performs the same two ordered rotations, angle calculation, stopping rule and sweep budget. The first rotation walks row slices; the second rotates two disjoint contiguous rows using zipped element operations. LLVM can perform independent coefficient operations in packed lanes without FMA contraction. Do not combine the two rotations by assuming exact floating-point symmetry.
- `Vector::dot` deliberately retains the scalar ordered `Iterator::sum`: its zero identity differs from the matrix row accumulator. Four- and eight-product array unrolling and portable SIMD multiplication followed by ordered additions did not materially improve the serial dependency chain on the measured generic target. Reassociation or fast-math would change cancellation and overflow behavior and is not justified here.

## Reproducible measurements

The ignored tests in `tests/dense_real_bench.rs` exercise square, narrow, wide and partial-tile matrices, dot products, allocated and reusable output, and analytic equicorrelation eigenvalues. The latter is a normalized inductance matrix with diagonal 1 and off-diagonal 0.25. For order at least 8, the ordinary and weighted Gershgorin certificates cannot accept it, exercising the dense Jacobi fallback used by mutual validation. For accuracy, the non-ignored tests check bit-preserving transpose including nonfinite payloads, scalar-order matvec including cancellation, extreme/product overflow and signed zero, transactional dimension mismatch, zero shapes, and an independent Hadamard spectrum at widely separated scales.

Run with the campaign's pinned nightly and managed sccache:

```sh
cargo +nightly test --release --locked --test dense_real_bench
cargo +nightly test --release --locked --test dense_real_bench -- --ignored --nocapture --test-threads=1
```

Hold compiler target, profile and input constant between baseline and candidate. Time five repeated batches per shape with both input and output passed through `black_box`, use medians, alternate baseline/candidate ordering, and pin to the same CPU for noisy shared hosts. Use the `dense_mutual_eigenvalues` case under `perf stat` / `perf record` and the separate `dense_allocations` or `dense_reused_allocations` cases under DHAT; a full ignored benchmark under DHAT needlessly includes the timing loops.

Baseline `4b8a1cf0`; Acer Ryzen 7 4700U, generic x86-64 release build, pinned nightly; CPU 2. Baseline production profile of the dense mutual workload attributes 98% of samples to `jacobi_eigenvalues`. Three alternating paired runs of the production Jacobi workload gave median 524.99 million vs 503.61 million user cycles (-4.1%) and 1,778.23 million vs 1,451.15 million instructions (-18.4%). Independent differential harnesses extracted both implementation bodies, confirmed 2,304 bitwise-identical convergence/results for varied finite symmetric matrices, and repeated this on generic x86-64 and explicit Zen 2 targets. Isolated nonuniform Jacobi workload speed improvements were approximately 5–17% depending on order and target.

DHAT on 128 iterations of 64x64 matvec + transpose and 16x16 Jacobi (including test runtime and setup): baseline allocating operations 4,606,956 bytes / 733 blocks; candidate allocating 4,606,957 bytes / 733 blocks; candidate reused matvec and transpose 380,446 bytes / 479 blocks. Reusable output allocates once during setup, so the two kernels allocate no memory per invocation; Jacobi retains its checked copy and output allocations. Peak live heap 73,184 B baseline and 75,901 B reused (both output buffers deliberately coexist). These are kernel workload results, not end-to-end simulation speed claims.

### Final paired measurements

Three alternating build pairs, five samples per pair (15 samples per shape), durations in microseconds per operation. Reuse is the final checked preallocated API. Dot, unchanged, serves as a host-noise control.

| Kernel / shape | Baseline | Final allocating | Final reuse | Allocating speedup |
|---|---:|---:|---:|---:|
| Matvec 4x4 | 0.0281 | 0.0318 | 0.0118 | 0.88x |
| Matvec 16x16 | 0.108 | 0.088 | 0.063 | 1.23x |
| Matvec 64x64 | 2.136 | 0.924 | 0.884 | 2.31x |
| Matvec 256x256 | 46.234 | 15.881 | 15.300 | 2.91x |
| Matvec 1024x1024 | 904.177 | 586.760 | 578.432 | 1.54x |
| Matvec 4096x7 | 10.871 | 7.830 | 7.609 | 1.39x |
| Matvec 7x4096 | 22.219 | 13.378 | 13.611 | 1.66x |
| Matvec 33x65 | 1.132 | 0.513 | 0.511 | 2.21x |
| Transpose 4x4 | 0.030 | 0.035 | 0.015 | 0.87x |
| Transpose 16x16 | 0.235 | 0.168 | 0.100 | 1.40x |
| Transpose 64x64 | 2.936 | 2.288 | 1.544 | 1.28x |
| Transpose 256x256 | 358.828 | 97.469 | 86.078 | 3.68x |
| Transpose 1024x1024 | 11166.061 | 4961.945 | 3895.884 | 2.25x |
| Transpose 8192x64 | 5722.592 | 1672.035 | 1310.367 | 3.42x |
| Transpose 64x8192 | 3476.548 | 2183.390 | 1618.580 | 1.59x |
| Transpose 4096x7 | 20.282 | 15.956 | 11.116 | 1.27x |
| Transpose 7x4096 | 25.948 | 19.304 | 14.486 | 1.34x |
| Transpose 511x1025 | 1303.376 | 1940.564 | 1471.570 | 0.67x |
| Jacobi 4x4 | 0.272 | 0.297 | — | 0.92x |
| Jacobi 8x8 | 1.247 | 1.137 | — | 1.10x |
| Jacobi 16x16 | 6.812 | 6.224 | — | 1.09x |
| Jacobi 32x32 | 39.503 | 36.302 | — | 1.09x |
| Jacobi 64x64 | 518.016 | 446.601 | — | 1.16x |
| Jacobi 128x128 | 4981.582 | 5020.448 | — | 0.99x |

Unchanged dot ratios were 0.94–1.01x, indicating some host variation. The allocating tiny operations pay a few nanoseconds of additional dispatch overhead; 511x1025 transpose is a real shape-dependent tradeoff in this run. A staged scratch tile and several tiled traversal variants were profiled but did not consistently improve across shapes; no special-case dimensions or architecture-specific intrinsics are introduced.

## Numerical follow-ups

- [#148](https://github.com/codeandsolder/ngspice-rs-fast/issues/148): the existing Jacobi two-sided rotations accumulate tiny asymmetry, and the squared-epsilon stopping threshold can fall below that floor. An exact 64x64 unit-diagonal positive-definite Hadamard-spectrum matrix (min eigenvalue 0.5078125) fails to converge in both baseline and optimized kernels. The symmetry-preserving numerical repair is independent of the performance change; no acceptance tolerance is changed here.
- [#149](https://github.com/codeandsolder/ngspice-rs-fast/issues/149): the existing DC ideal-short assertion compares a 5.551115123125783e-17 V solution residual to literal zero. Both baseline and candidate fail identically in dev and release; the physical error is negligible but the test requires an unjustified exact zero. Use scale-justified physical residual policy rather than forcing a different sparse solver rounding path.
