# BSIM3 arithmetic audit (#88)

Baseline: cd826d9383445c83bc48fedf9d6038fbb33a8f13; candidate: 0035ca9758de9c2b18af11a7802637f78392a075.

Acer Ryzen 7 4700U, rustc 1.102.0-nightly (6905905bb), LLVM 23.1.6, managed sccache. Release test arithmetic_workload: three BSIM3 cards, 2,048 gate/drain/body biases each, 29 output fields, 65,536 evaluations per timing trial, three trials. Baseline has the same workload harness, with original equations. Generic and target-cpu=native builds. Concurrent campaign and CI jobs make timings descriptive, not controlled estimates of speedup.

## Numerical comparisons

Maximum absolute difference / peak is normalized by the largest absolute value in either version for that field across the complete grid. Relative difference excludes baseline values smaller than 1e-12 of that peak. The large relative ratios for almost-zero charges/capacitances should be interpreted with the corresponding absolute errors. No tolerance was changed.

### generic

| Field | Max abs difference | Max abs / peak | Max relative (above floor) | Zero to nonzero |
|---|---:|---:|---:|---:|
| ids | 2.385e-18 | 1.838e-15 | 7.139e-12 | 0 |
| gm | 2.385e-18 | 3.931e-15 | 2.358e-11 | 0 |
| gds | 3.469e-18 | 1.675e-15 | 9.420e-12 | 0 |
| gmbs | 4.811e-19 | 3.354e-15 | 3.244e-11 | 0 |
| isub | 8.216e-20 | 4.364e-14 | 1.612e-11 | 0 |
| vgsteff | 1.332e-15 | 5.401e-16 | 1.722e-11 | 0 |
| vdseff | 1.843e-14 | 1.382e-14 | 2.315e-13 | 0 |
| vdsat | 2.909e-14 | 2.058e-14 | 2.477e-13 | 0 |
| cbs | 1.010e-28 | 1.346e-16 | 1.346e-16 | 0 |
| cbd | 4.039e-28 | 1.795e-16 | 2.126e-16 | 0 |
| qgate | 1.420e-29 | 1.554e-15 | 1.933e-15 | 0 |
| qbulk | 1.104e-29 | 3.067e-15 | 4.263e-15 | 0 |
| qdrn | 4.339e-30 | 1.259e-15 | 1.040e-04 | 0 |
| qsrc | 3.944e-30 | 8.491e-16 | 1.068e-06 | 0 |
| cggb | 1.302e-29 | 4.468e-15 | 6.207e-15 | 0 |
| cgdb | 3.747e-30 | 2.668e-15 | 2.251e-05 | 0 |
| cgsb | 8.283e-30 | 3.776e-15 | 2.727e-12 | 0 |
| cdgb | 5.325e-30 | 3.658e-15 | 2.655e-10 | 0 |
| cddb | 9.269e-30 | 2.074e-15 | 2.058e-05 | 0 |
| cdsb | 4.536e-30 | 1.684e-15 | 6.013e-05 | 0 |
| cbgb | 1.267e-29 | 1.267e-14 | 5.737e-12 | 0 |
| cbdb | 3.550e-30 | 2.307e-15 | 2.241e-05 | 0 |
| cbsb | 3.353e-30 | 5.784e-15 | 1.357e-10 | 0 |
| capbs | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0 |
| capbd | 1.972e-31 | 1.878e-16 | 2.079e-16 | 0 |
| qbs | 1.972e-31 | 1.821e-16 | 5.127e-16 | 0 |
| qbd | 7.889e-31 | 4.206e-16 | 9.693e-16 | 0 |
| channel | 5.963e-18 | 3.045e-15 | 1.561e-06 | 0 |
| flicker_numerator | 1.425e-29 | 9.939e-15 | 1.128e-11 | 0 |

Timing (ms):

| Card | Baseline trials | Candidate trials | Median ratio |
|---|---|---|---:|
| 0 | 56.804, 86.536, 105.207 | 84.629, 79.112, 71.153 | 0.914 |
| 1 | 77.836, 399.771, 315.788 | 82.634, 80.206, 82.772 | 0.262 |
| 2 | 59.730, 56.307, 57.940 | 71.525, 71.781, 72.738 | 1.239 |

### native

| Field | Max abs difference | Max abs / peak | Max relative (above floor) | Zero to nonzero |
|---|---:|---:|---:|---:|
| ids | 2.385e-18 | 1.838e-15 | 7.139e-12 | 0 |
| gm | 2.385e-18 | 3.931e-15 | 2.358e-11 | 0 |
| gds | 3.469e-18 | 1.675e-15 | 9.420e-12 | 0 |
| gmbs | 4.811e-19 | 3.354e-15 | 3.244e-11 | 0 |
| isub | 8.216e-20 | 4.364e-14 | 1.612e-11 | 0 |
| vgsteff | 1.332e-15 | 5.401e-16 | 1.722e-11 | 0 |
| vdseff | 1.843e-14 | 1.382e-14 | 2.315e-13 | 0 |
| vdsat | 2.909e-14 | 2.058e-14 | 2.477e-13 | 0 |
| cbs | 1.010e-28 | 1.346e-16 | 1.346e-16 | 0 |
| cbd | 4.039e-28 | 1.795e-16 | 2.126e-16 | 0 |
| qgate | 1.420e-29 | 1.554e-15 | 1.933e-15 | 0 |
| qbulk | 1.104e-29 | 3.067e-15 | 4.263e-15 | 0 |
| qdrn | 4.339e-30 | 1.259e-15 | 1.040e-04 | 0 |
| qsrc | 3.944e-30 | 8.491e-16 | 1.068e-06 | 0 |
| cggb | 1.302e-29 | 4.468e-15 | 6.207e-15 | 0 |
| cgdb | 3.747e-30 | 2.668e-15 | 2.251e-05 | 0 |
| cgsb | 8.283e-30 | 3.776e-15 | 2.727e-12 | 0 |
| cdgb | 5.325e-30 | 3.658e-15 | 2.655e-10 | 0 |
| cddb | 9.269e-30 | 2.074e-15 | 2.058e-05 | 0 |
| cdsb | 4.536e-30 | 1.684e-15 | 6.013e-05 | 0 |
| cbgb | 1.267e-29 | 1.267e-14 | 5.737e-12 | 0 |
| cbdb | 3.550e-30 | 2.307e-15 | 2.241e-05 | 0 |
| cbsb | 3.353e-30 | 5.784e-15 | 1.357e-10 | 0 |
| capbs | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0 |
| capbd | 1.972e-31 | 1.878e-16 | 2.079e-16 | 0 |
| qbs | 1.972e-31 | 1.821e-16 | 5.127e-16 | 0 |
| qbd | 7.889e-31 | 4.206e-16 | 9.693e-16 | 0 |
| channel | 5.963e-18 | 3.045e-15 | 1.561e-06 | 0 |
| flicker_numerator | 1.425e-29 | 9.939e-15 | 1.128e-11 | 0 |

Timing (ms):

| Card | Baseline trials | Candidate trials | Median ratio |
|---|---|---|---:|
| 0 | 654.591, 91.061, 74.480 | 47.567, 47.111, 46.526 | 0.517 |
| 1 | 58.078, 57.973, 74.275 | 50.617, 50.707, 48.032 | 0.872 |
| 2 | 76.639, 58.812, 97.771 | 44.306, 44.781, 45.576 | 0.584 |

## Code generation and validation

An isolated f64::mul_add release assembly probe with the campaign compiler emits a jump through fma@GOTPCREL for generic x86-64 and vfmadd213sd for native. Generic fma may dispatch to hardware in the system libm but incurs call overhead. Accuracy improvements are retained; native should be preferred for the benchmark, and controlled CPU-time comparisons are needed before claiming a speedup.

The analytic tiny-bias, continuous saturation-root limit and conservation/finite-difference regressions pass. Existing BSIM3 suite (57 tests), live noise (11 tests), golden rawfile (14 tests), and global golden verification (219 verified, 1 unsupported, 0 failed) pass without loosening tolerances. The BSIM3 transient golden has worst normalized error 0.693 of its existing bound. Strict library audit reports no lint diagnostics.

Raw logs: campaign logs/arch-1122-a7c9-88-{base,candidate}-{generic,native}.log; focused.log, analytic-golden.log, golden-verify.log, clippy.jsonl, bench-meta.log under the same prefix.
