# Native, PGO, aggressive LLVM, Polly, FMA, algebraic arithmetic and BOLT — 2026-11-24

All builds are release O3, fat LTO, codegen-units=1, pinned nightly-2026-11-20, managed sccache. Compilation time excluded. Acer Ryzen 7 4700U pinned CPU 7; cold-storage EPYC 9655 Zen 5 VM pinned CPU 3. Explicit znver2 and znver5 targets. Portable builds use static glibc because Acer glibc 2.44 exceeds cold-storage 2.41. Dynamic BOLT comparison uses a separate identical pre/post-link base. PGO profile trained on target architecture, all five embedded circuits and all three modes; merge via LLVM profdata. Each initial benchmark has five rounds after warmup; final cold-storage matrix interleaves eight variants, five rounds, generic and baked modes, 400 measurements. Solver times exclude circuit setup/output hash.

PGO consistently wins; no major additional advantage to circuit-specific specialization. Aggressive MIR/inlining/unroll/loop transforms, final-LTO Polly, global FMA and sparse algebraic updates offer no consistent large gain. Polly plugin is restricted to final LTO because sccache distribution does not transfer the shared object. Probe confirmed the pass is recognized with the plugin and rejected without it. BOLT instrumentation failed with static PIE and unsupported JITLink relocation; sampled instruction-address profiling succeeded for dynamically linked base. No LBR support on Acer. BOLT profile has 9560 samples, 392 profiled functions; BOLT warns of insufficient sample density. All strict variants preserve exact fixture outputs. BOLT regresses timings, so do not adopt. FMA increases static fused instruction count from 936 to 7716 and changes rounding; all point counts unchanged. Selective algebraic sparse mul_sub feature on experiment/circuit-lto-relaxed (62acf54) changes rounding but not point counts. Full dumps compared to strict baseline, no interpolation. Maximum absolute differences across all scalars: FMA 4.434e-12, selective algebraic 3.809e-12; max time-axis deviations 2.12e-22 s and 6.98e-25 s respectively. These five cases do not establish general numerical fidelity. No global finite-only assumptions introduced.

## Acer native
| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |
|---|---:|---:|---:|---:|---:|
| rc_transient | 0.120 | 0.118 | 0.119 | 1.011x | 1.009x |
| m4_diode_tran | 2.052 | 2.035 | 2.062 | 1.008x | 0.995x |
| m11_bsim4_tran | 4.853 | 4.969 | 4.880 | 0.977x | 0.994x |
| m11_bsim4_dc | 0.418 | 0.406 | 0.410 | 1.030x | 1.019x |
| diode100_op | 0.439 | 0.495 | 0.509 | 0.888x | 0.862x |

## Acer native + PGO
| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |
|---|---:|---:|---:|---:|---:|
| rc_transient | 0.085 | 0.080 | 0.081 | 1.058x | 1.048x |
| m4_diode_tran | 1.425 | 1.372 | 1.347 | 1.038x | 1.058x |
| m11_bsim4_tran | 3.922 | 3.929 | 3.755 | 0.998x | 1.044x |
| m11_bsim4_dc | 0.318 | 0.322 | 0.318 | 0.986x | 0.999x |
| diode100_op | 0.445 | 0.525 | 0.536 | 0.848x | 0.831x |

## Acer aggressive
| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |
|---|---:|---:|---:|---:|---:|
| rc_transient | 0.119 | 0.118 | 0.119 | 1.010x | 1.000x |
| m4_diode_tran | 2.912 | 2.305 | 2.295 | 1.263x | 1.269x |
| m11_bsim4_tran | 6.054 | 6.160 | 5.714 | 0.983x | 1.059x |
| m11_bsim4_dc | 0.483 | 0.504 | 0.501 | 0.958x | 0.964x |
| diode100_op | 0.559 | 0.613 | 0.644 | 0.912x | 0.869x |

## Acer final-LTO Polly
| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |
|---|---:|---:|---:|---:|---:|
| rc_transient | 0.156 | 0.156 | 0.160 | 0.996x | 0.973x |
| m4_diode_tran | 2.669 | 2.698 | 2.731 | 0.989x | 0.977x |
| m11_bsim4_tran | 5.839 | 5.864 | 5.854 | 0.996x | 0.998x |
| m11_bsim4_dc | 0.445 | 0.451 | 0.435 | 0.986x | 1.021x |
| diode100_op | 0.499 | 0.547 | 0.584 | 0.913x | 0.855x |

## Cold-storage znver2 native
| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |
|---|---:|---:|---:|---:|---:|
| rc_transient | 0.080 | 0.081 | 0.081 | 0.984x | 0.988x |
| m4_diode_tran | 1.366 | 1.400 | 1.411 | 0.976x | 0.968x |
| m11_bsim4_tran | 3.000 | 3.043 | 2.875 | 0.986x | 1.043x |
| m11_bsim4_dc | 0.234 | 0.232 | 0.233 | 1.009x | 1.002x |
| diode100_op | 0.238 | 0.260 | 0.270 | 0.913x | 0.882x |

## Cold-storage znver2 pgo
| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |
|---|---:|---:|---:|---:|---:|
| rc_transient | 0.059 | 0.056 | 0.057 | 1.052x | 1.031x |
| m4_diode_tran | 0.986 | 0.949 | 0.968 | 1.040x | 1.019x |
| m11_bsim4_tran | 2.814 | 2.868 | 2.809 | 0.981x | 1.002x |
| m11_bsim4_dc | 0.210 | 0.210 | 0.212 | 0.997x | 0.987x |
| diode100_op | 0.213 | 0.213 | 0.223 | 0.999x | 0.954x |

## Cold-storage znver5 native
| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |
|---|---:|---:|---:|---:|---:|
| rc_transient | 0.080 | 0.080 | 0.079 | 0.997x | 1.012x |
| m4_diode_tran | 1.340 | 1.371 | 1.375 | 0.978x | 0.975x |
| m11_bsim4_tran | 2.870 | 2.845 | 2.947 | 1.009x | 0.974x |
| m11_bsim4_dc | 0.234 | 0.232 | 0.233 | 1.011x | 1.008x |
| diode100_op | 0.249 | 0.267 | 0.262 | 0.934x | 0.951x |

## Cold-storage znver5 pgo
| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |
|---|---:|---:|---:|---:|---:|
| rc_transient | 0.066 | 0.063 | 0.060 | 1.055x | 1.107x |
| m4_diode_tran | 0.973 | 0.939 | 0.937 | 1.036x | 1.038x |
| m11_bsim4_tran | 2.667 | 2.724 | 2.778 | 0.979x | 0.960x |
| m11_bsim4_dc | 0.202 | 0.204 | 0.201 | 0.989x | 1.005x |
| diode100_op | 0.200 | 0.205 | 0.209 | 0.979x | 0.958x |

## Cold-storage interleaved matrix (preferred for cross-variant comparison)
| Case | Variant | Generic ms | Baked ms | Baked speedup | Generic relative to znver2 native |
|---|---|---:|---:|---:|---:|
| rc_transient | znver2/native | 0.0812 | 0.0809 | 1.004x | 1.000x |
| rc_transient | znver2/pgo | 0.0607 | 0.0571 | 1.062x | 1.339x |
| rc_transient | znver2/aggressive | 0.0773 | 0.0785 | 0.984x | 1.051x |
| rc_transient | znver2/polly | 0.0789 | 0.0805 | 0.979x | 1.029x |
| rc_transient | znver2/fma | 0.0793 | 0.0788 | 1.006x | 1.024x |
| rc_transient | znver2/relaxed | 0.0790 | 0.0798 | 0.989x | 1.028x |
| rc_transient | znver5/native | 0.0776 | 0.0767 | 1.011x | 1.047x |
| rc_transient | znver5/pgo | 0.0618 | 0.0594 | 1.040x | 1.314x |
| m4_diode_tran | znver2/native | 1.3835 | 1.4116 | 0.980x | 1.000x |
| m4_diode_tran | znver2/pgo | 0.9989 | 0.9812 | 1.018x | 1.385x |
| m4_diode_tran | znver2/aggressive | 1.3527 | 1.3684 | 0.989x | 1.023x |
| m4_diode_tran | znver2/polly | 1.3493 | 1.4438 | 0.935x | 1.025x |
| m4_diode_tran | znver2/fma | 1.4221 | 1.4137 | 1.006x | 0.973x |
| m4_diode_tran | znver2/relaxed | 1.3736 | 1.4289 | 0.961x | 1.007x |
| m4_diode_tran | znver5/native | 1.3637 | 1.3968 | 0.976x | 1.015x |
| m4_diode_tran | znver5/pgo | 0.9785 | 0.9584 | 1.021x | 1.414x |
| m11_bsim4_tran | znver2/native | 3.0169 | 2.8753 | 1.049x | 1.000x |
| m11_bsim4_tran | znver2/pgo | 2.8773 | 2.7691 | 1.039x | 1.049x |
| m11_bsim4_tran | znver2/aggressive | 2.9588 | 2.8291 | 1.046x | 1.020x |
| m11_bsim4_tran | znver2/polly | 2.9740 | 2.8172 | 1.056x | 1.014x |
| m11_bsim4_tran | znver2/fma | 2.9177 | 2.8567 | 1.021x | 1.034x |
| m11_bsim4_tran | znver2/relaxed | 3.0104 | 2.9274 | 1.028x | 1.002x |
| m11_bsim4_tran | znver5/native | 2.8545 | 3.0634 | 0.932x | 1.057x |
| m11_bsim4_tran | znver5/pgo | 2.7561 | 2.7477 | 1.003x | 1.095x |
| m11_bsim4_dc | znver2/native | 0.2421 | 0.2405 | 1.007x | 1.000x |
| m11_bsim4_dc | znver2/pgo | 0.2186 | 0.2152 | 1.016x | 1.108x |
| m11_bsim4_dc | znver2/aggressive | 0.2328 | 0.2330 | 0.999x | 1.040x |
| m11_bsim4_dc | znver2/polly | 0.2497 | 0.2374 | 1.052x | 0.970x |
| m11_bsim4_dc | znver2/fma | 0.2480 | 0.2385 | 1.040x | 0.976x |
| m11_bsim4_dc | znver2/relaxed | 0.2687 | 0.2573 | 1.044x | 0.901x |
| m11_bsim4_dc | znver5/native | 0.2314 | 0.2364 | 0.979x | 1.047x |
| m11_bsim4_dc | znver5/pgo | 0.2048 | 0.2124 | 0.964x | 1.182x |
| diode100_op | znver2/native | 0.2420 | 0.2755 | 0.879x | 1.000x |
| diode100_op | znver2/pgo | 0.2094 | 0.2243 | 0.934x | 1.156x |
| diode100_op | znver2/aggressive | 0.2293 | 0.2572 | 0.891x | 1.056x |
| diode100_op | znver2/polly | 0.2459 | 0.2666 | 0.922x | 0.984x |
| diode100_op | znver2/fma | 0.2491 | 0.2916 | 0.854x | 0.972x |
| diode100_op | znver2/relaxed | 0.2348 | 0.2590 | 0.906x | 1.031x |
| diode100_op | znver5/native | 0.2504 | 0.2621 | 0.955x | 0.967x |
| diode100_op | znver5/pgo | 0.2023 | 0.2107 | 0.960x | 1.196x |

## Acer dynamic BOLT base
| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |
|---|---:|---:|---:|---:|---:|
| rc_transient | 0.141 | 0.141 | 0.141 | 1.002x | 0.999x |
| m4_diode_tran | 2.209 | 2.279 | 2.275 | 0.970x | 0.971x |
| m11_bsim4_tran | 4.874 | 4.791 | 4.671 | 1.017x | 1.043x |
| m11_bsim4_dc | 0.381 | 0.381 | 0.380 | 1.000x | 1.003x |
| diode100_op | 0.410 | 0.439 | 0.437 | 0.936x | 0.938x |

## Acer BOLT basic-sample optimized
| Case | Generic ms/run | Dispatch ms/run | Baked ms/run | Dispatch speedup | Baked speedup |
|---|---:|---:|---:|---:|---:|
| rc_transient | 0.187 | 0.189 | 0.186 | 0.989x | 1.005x |
| m4_diode_tran | 3.019 | 3.004 | 2.912 | 1.005x | 1.037x |
| m11_bsim4_tran | 5.448 | 5.392 | 5.164 | 1.011x | 1.055x |
| m11_bsim4_dc | 0.441 | 0.443 | 0.441 | 0.994x | 0.999x |
| diode100_op | 0.485 | 0.521 | 0.515 | 0.932x | 0.942x |

## Full plot comparisons
~~~
case=0 mode=generic relaxed {"cells":560,"changed":38,"maxAbsolute":4.336808689942018e-19,"maxScaled":1.0648114570472153e-14,"axisMax":0}
case=0 mode=baked relaxed {"cells":560,"changed":38,"maxAbsolute":4.336808689942018e-19,"maxScaled":1.0648114570472153e-14,"axisMax":0}
case=1 mode=generic relaxed {"cells":10200,"changed":3022,"maxAbsolute":7.66053886991358e-15,"maxScaled":1.0320551281844422e-11,"axisMax":0}
case=1 mode=baked relaxed {"cells":10200,"changed":3022,"maxAbsolute":7.66053886991358e-15,"maxScaled":1.0320551281844422e-11,"axisMax":0}
case=2 mode=generic relaxed {"cells":7392,"changed":3056,"maxAbsolute":7.638334409421077e-14,"maxScaled":0.00005019386195572662,"axisMax":6.979336418416171e-25}
case=2 mode=baked relaxed {"cells":7392,"changed":3056,"maxAbsolute":7.638334409421077e-14,"maxScaled":0.00005019386195572662,"axisMax":6.979336418416171e-25}
case=3 mode=generic relaxed {"cells":1170,"changed":197,"maxAbsolute":2.220446049250313e-16,"maxScaled":0.0001887379141862766,"axisMax":0}
case=3 mode=baked relaxed {"cells":1170,"changed":197,"maxAbsolute":2.220446049250313e-16,"maxScaled":0.0001887379141862766,"axisMax":0}
case=4 mode=generic relaxed {"cells":404,"changed":201,"maxAbsolute":3.808509063674137e-12,"maxScaled":1.9444525039795738e-13,"axisMax":0}
case=4 mode=baked relaxed {"cells":404,"changed":201,"maxAbsolute":3.808509063674137e-12,"maxScaled":1.9444525039795738e-13,"axisMax":0}
case=0 mode=generic FMA {"cells":560,"changed":135,"maxAbsolute":6.217248937900877e-15,"maxScaled":8.301950245599521e-14,"axisMax":2.117582368135751e-22}
case=0 mode=baked FMA {"cells":560,"changed":135,"maxAbsolute":6.217248937900877e-15,"maxScaled":8.301950245599521e-14,"axisMax":2.117582368135751e-22}
case=1 mode=generic FMA {"cells":10200,"changed":3033,"maxAbsolute":7.327471962526033e-15,"maxScaled":8.947351861227067e-12,"axisMax":0}
case=1 mode=baked FMA {"cells":10200,"changed":3033,"maxAbsolute":7.327471962526033e-15,"maxScaled":8.947351861227067e-12,"axisMax":0}
case=2 mode=generic FMA {"cells":7392,"changed":3100,"maxAbsolute":1.8984813721090177e-13,"maxScaled":0.000031415815780929404,"axisMax":2.895132143935597e-24}
case=2 mode=baked FMA {"cells":7392,"changed":3118,"maxAbsolute":7.260858581048524e-14,"maxScaled":0.00005140384762442898,"axisMax":6.203854594147708e-25}
case=3 mode=generic FMA {"cells":1170,"changed":241,"maxAbsolute":2.220446049250313e-16,"maxScaled":0.0001887379141862766,"axisMax":0}
case=3 mode=baked FMA {"cells":1170,"changed":241,"maxAbsolute":2.220446049250313e-16,"maxScaled":0.0001887379141862766,"axisMax":0}
case=4 mode=generic FMA {"cells":404,"changed":201,"maxAbsolute":4.433786671143025e-12,"maxScaled":6.070298215710513e-13,"axisMax":0}
case=4 mode=baked FMA {"cells":404,"changed":201,"maxAbsolute":4.433786671143025e-12,"maxScaled":6.070298215710513e-13,"axisMax":0}
~~~

Raw logs, profiles, flags, build logs: /home/jan/build/ngspice-rs-fast/circuit-lto/optimized. Cold-storage raw matrix SHA256: c08843da7132695f8550c27e3e1d1212abf3b7b832309cb44a0873e7720ff852.
