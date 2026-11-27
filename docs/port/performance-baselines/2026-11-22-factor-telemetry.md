# Sparse factor telemetry baseline — 2026-11-22

Issue [#105](https://github.com/codeandsolder/ngspice-rs-fast/issues/105), source revision `6020b8a86d2b48df84454eae77a60a9789be3eef`.

- Acer, Ryzen 7 4700U (8 logical CPUs), x86-64 generic release, pinned rustc 1.102.0-nightly (6905905bb), LLVM 23.1.6, managed sccache.
- Nine full workloads passed. Host one-minute load 28.06 at start and 34.25 at end of the validation sequence. Timings are descriptive and not a controlled performance comparison.
- Raw logs: campaign `logs/arch-1122-a7c9-105-release-final.log`, `logs/arch-1122-a7c9-105-final-meta.log`, `logs/arch-1122-a7c9-105-tests-final.log`. Earlier live C checks and full library checks: `logs/arch-1122-a7c9-105-final-tests.log`, `logs/arch-1122-a7c9-105-focused.log`.

## Measured factor storage

Original entries include structural zero slots; fill counts are actual handle creations. Numeric bytes exclude spare capacity, topology, metadata and traversal plans.

| Workload | External order | Translated order | Original entries | Factor entries | Fill entries | Numeric bytes | Fill ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse_ladder | 1024 | 1024 | 3070 | 3070 | 0 | 24560 | 1.000000 |
| sparse_mesh | 1024 | 1024 | 4992 | 24292 | 19300 | 194336 | 4.866186 |
| complex_frequency_ensemble | 576 | 576 | 2784 | 11600 | 8816 | 185600 | 4.166667 |

## Actual matrix work

Real rows are the final cumulative cache snapshot across three NI trials and exclude independent fresh-order benchmark factors. Complex rows are per 64-frequency trial, with solves counted by the harness on successful independent snapshots. All failure counters are zero.

| Workload / mode | Order attempts | Pivot searches | Fixed refactors | Successful factors | Exact hits | Successful RHS solves |
|---|---:|---:|---:|---:|---:|---:|
| sparse_ladder | 3 | 1 | 96 | 99 | 96 | 195 |
| sparse_mesh | 3 | 1 | 72 | 75 | 72 | 147 |
| complex / fresh_order | 64 | 64 | 0 | 64 | — | 64 |
| complex / reuse_order | 1 | 1 | 63 | 64 | — | 64 |

## Full deck final cache epochs

Per retained trial; streamed trials reported the same counters. These are not whole-analysis totals if a pattern was replaced, and do not count independent snapshot RHS solves (including RF).

| Workload | Order | Original | Factor | Order attempts | Searches | Refactors | Factors | Exact hits | Cache RHS | Real → complex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rc_transient | 66 | 195 | 195 | 4 | 1 | 31 | 35 | 2016 | 2051 | 0 |
| nonlinear_mixed | 100 | 392 | 392 | 2 | 1 | 150 | 152 | 1 | 153 | 0 |
| compact_model_array | 388 | 1734 | 1734 | 1 | 1 | 66 | 67 | 1 | 68 | 0 |
| rf_multiport | 25 | 65 | 65 | 3 | 1 | 64 | 67 | 0 | 3 | 1 |
| ltra_history | 6 | 16 | 16 | 1 | 1 | 414 | 415 | 3102 | 3517 | 0 |
| parameter_ensemble | 3 | 6 | 6 | 16 | 1 | 128 | 144 | 120 | 264 | 0 |

Independent parameters: 128 measured final epochs per trial, 256 order attempts, 128 searches, 128 refactors, 384 successful factors, 0 exact hits and 384 cache RHS solves. Largest factor: 6 numeric slots (48 bytes).
- rc_transient: 1022 accepted, 2 rejected, 3 breakpoints.
- ltra_history: 1759 accepted, 0 rejected, 74 breakpoints.

## Numerical signatures

All matrix forward-error and residual signatures exactly match the [#34 baseline](2026-11-22-sparse-lifecycle.md). This statement concerns the reported maxima, not every solution entry.

| Workload | Max absolute forward error | Max absolute residual |
|---|---:|---:|
| sparse_ladder | 6.661338147750939e-16 | 4.440892098500626e-16 |
| sparse_mesh | 1.5543122344752192e-15 | 3.1086244689504383e-15 |
| complex_frequency_ensemble | 2.5989654674417885e-15 | 5.617333549722722e-15 |

## Timing (descriptive only)

Median and individual trials in microseconds per operation; no timing regression or acceleration claim on this busy host. Prior process-wide DHAT measurements are separate.

| Workload | Phase | Median | Individual trials |
|---|---|---:|---|
| compact_model_array | analysis_retained | 21061.457 | 21061.457, 221202.919, 19412.823 |
| compact_model_array | analysis_streamed | 19182.907 | 19182.907, 35780.987, 17722.891 |
| compact_model_array | binary_decode | 430.846 | 430.846, 803.843, 388.456 |
| compact_model_array | binary_encode | 169.852 | 192.094, 142.560, 169.852 |
| compact_model_array | elaborate_setup | 5253.394 | 5253.394 |
| compact_model_array | parse | 1725.269 | 1725.269 |
| compact_model_device_load | bias_initialization | 2450.332 | 2450.332 |
| compact_model_device_load | device_load | 174.994 | 189.986, 166.415, 174.994 |
| compact_model_device_load | elaborate_setup | 2331.748 | 2331.748 |
| compact_model_device_load | parse | 1105.424 | 1105.424 |
| complex_frequency_ensemble | fresh_order_assembly | 146.447 | 57.331, 279.025, 146.447 |
| complex_frequency_ensemble | fresh_order_factor | 22994.659 | 22994.659, 31870.811, 7906.011 |
| complex_frequency_ensemble | fresh_order_solve | 93.097 | 93.097, 259.641, 70.756 |
| complex_frequency_ensemble | reuse_order_assembly | 53.682 | 53.682, 66.277, 48.081 |
| complex_frequency_ensemble | reuse_order_factor | 460.725 | 460.725, 375.940, 2590.085 |
| complex_frequency_ensemble | reuse_order_solve | 34.478 | 33.980, 34.478, 34.867 |
| independent_parameters | elaborate_setup | 31.500 | 31.500 |
| independent_parameters | parse | 37.892 | 37.892 |
| independent_parameters | prepare | 43.688 | 43.688, 40.192, 132.785 |
| independent_parameters | solve | 9.044 | 227.063, 7.828, 9.044 |
| ltra_history | analysis_retained | 435509.090 | 2986748.265, 411469.521, 435509.090 |
| ltra_history | analysis_streamed | 676287.711 | 676287.711, 1448989.884, 243343.591 |
| ltra_history | binary_decode | 161.195 | 212.452, 158.651, 161.195 |
| ltra_history | binary_encode | 45.937 | 40.758, 51.978, 45.937 |
| ltra_history | elaborate_setup | 99.999 | 130.397, 69.602 |
| ltra_history | parse | 80.703 | 88.187, 73.219 |
| ltra_history | transient_with_stats | 244643.623 | 244643.623 |
| nonlinear_device_load | bias_initialization | 817.880 | 817.880 |
| nonlinear_device_load | device_load | 48.242 | 46.843, 51.976, 48.242 |
| nonlinear_device_load | elaborate_setup | 1480.946 | 1480.946 |
| nonlinear_device_load | parse | 967.653 | 967.653 |
| nonlinear_mixed | analysis_retained | 10769.996 | 174010.671, 10769.996, 10474.186 |
| nonlinear_mixed | analysis_streamed | 10564.707 | 181681.094, 10564.707, 10486.941 |
| nonlinear_mixed | binary_decode | 157.639 | 157.639, 241.838, 154.583 |
| nonlinear_mixed | binary_encode | 52.620 | 52.620, 78.208, 48.512 |
| nonlinear_mixed | elaborate_setup | 32713.885 | 32713.885 |
| nonlinear_mixed | parse | 1009.693 | 1009.693 |
| parameter_ensemble | analysis_retained | 2078.077 | 528.221, 6126.408, 2078.077 |
| parameter_ensemble | analysis_streamed | 482.274 | 440.965, 571.132, 482.274 |
| parameter_ensemble | binary_decode | 18.285 | 18.094, 19.817, 18.285 |
| parameter_ensemble | binary_encode | 6.612 | 6.161, 6.993, 6.612 |
| parameter_ensemble | elaborate_setup | 99.879 | 99.879 |
| parameter_ensemble | parse | 59.012 | 59.012 |
| rc_transient | analysis_retained | 480769.768 | 492992.177, 480769.768, 95200.914 |
| rc_transient | analysis_streamed | 329148.921 | 657254.898, 329148.921, 39773.652 |
| rc_transient | binary_decode | 106284.265 | 1357.993, 106284.265, 682594.671 |
| rc_transient | binary_encode | 46700.617 | 73023.031, 46700.617, 3621.542 |
| rc_transient | elaborate_setup | 491.656 | 505.217, 478.096 |
| rc_transient | parse | 402.262 | 389.087, 415.437 |
| rc_transient | transient_with_stats | 42736.155 | 42736.155 |
| rf_multiport | analysis_retained | 1726.532 | 12152.717, 1659.855, 1726.532 |
| rf_multiport | analysis_streamed | 16846.761 | 19282.145, 1769.372, 16846.761 |
| rf_multiport | binary_decode | 212.933 | 204.217, 216.099, 212.933 |
| rf_multiport | binary_encode | 77.106 | 80.673, 77.106, 74.652 |
| rf_multiport | elaborate_setup | 213.976 | 213.976 |
| rf_multiport | parse | 191.764 | 191.764 |
| sparse_ladder | bound_assembly | 258.241 | 255.777, 258.241, 513.937 |
| sparse_ladder | changed_coefficients_ni_solve | 143.933 | 143.933, 41.516, 217.346 |
| sparse_ladder | fresh_order_factor | 290.290 | 290.290, 399.882, 154.346 |
| sparse_ladder | identical_coefficients_ni_solve | 16.633 | 16.633, 29.245, 16.612 |
| sparse_ladder | pattern_setup | 405.408 | 405.408 |
| sparse_ladder | retained_solve | 15.154 | 15.840, 15.154, 15.119 |
| sparse_ladder | triplet_assembly | 481.608 | 481.608, 286.445, 581.031 |
| sparse_mesh | bound_assembly | 251.856 | 316.274, 251.856, 245.460 |
| sparse_mesh | changed_coefficients_ni_solve | 691.353 | 691.353, 487.212, 1323.071 |
| sparse_mesh | fresh_order_factor | 33817.928 | 92601.629, 33817.928, 21835.273 |
| sparse_mesh | identical_coefficients_ni_solve | 38.263 | 38.221, 38.263, 39.123 |
| sparse_mesh | pattern_setup | 734.983 | 734.983 |
| sparse_mesh | retained_solve | 38.594 | 38.594, 37.728, 82.310 |
| sparse_mesh | triplet_assembly | 387.682 | 873.997, 387.682, 380.226 |
