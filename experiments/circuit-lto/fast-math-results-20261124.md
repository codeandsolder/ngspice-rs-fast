# Blanket LLVM fast math — 2026-11-24

Research-only experiment, on experiment/circuit-lto. C++ LLVM plugin uses the full-link-time optimization early extension point to set all seven fast-math flags on every FPMathOperator in the linked Rust LLVM module. This includes finite-only assumptions, not just algebraic or contraction flags. No rewriting of Rust expressions. Dependencies underwent their ordinary strict prelink optimization, then blanket flags were applied to the combined module before full LTO; external native libc routines are not rewritten. LLVM 23.1.5 system headers; rustc nightly-2026-11-20 LLVM 23.1.6. Managed C++ and Rust sccache wrappers retained. Plugin probe confirms fmul fast in output IR. Native builds changed 78828 (znver2) / 78850 (znver5) eligible instructions in 1100 functions, 814 device functions. PGO builds changed 80588 / 80595 instructions in 1003 functions, 710 device functions. Same benchmark source for strict and fast controls; static glibc, release O3/fat LTO/CGU=1. Existing PGO profiles reused (same two benchmark-harness-only hash-mismatch warnings for both builds). Five rotated rounds with generic and baked modes; solve time excludes setup and digest. Cold-storage pinned CPU 3, Acer CPU 7.

All five decks, both execution modes, complete with same point counts. Maximum absolute difference 5.4001247917767614e-12; maximum time-axis deviation 2.117582368135751e-21 s. All values checked externally for finiteness. Numerical equivalence on these decks does not establish behavior on invalid or exceptional inputs. Cold-storage native generic speedups 1.03–1.21x, baked 1.00–1.15x. On cold-storage with Zen 2 target generic 1.04–1.13x, baked 1.04–1.17x. Acer first run during high concurrent build load is confounded (see logs); use cold-storage for reliable comparison.

## Cold-storage, Zen 5 target, native
| Case | Variant | Generic ms | Baked ms | Generic speedup vs strict | Baked speedup vs strict | Baked vs generic |
|---|---|---:|---:|---:|---:|---:|
| rc_transient | native-strict | 0.0834 | 0.0825 | 1.000x | 1.000x | 1.012x |
| rc_transient | native-fast | 0.0741 | 0.0756 | 1.126x | 1.091x | 0.980x |
| m4_diode_tran | native-strict | 1.4893 | 1.3871 | 1.000x | 1.000x | 1.074x |
| m4_diode_tran | native-fast | 1.2322 | 1.2106 | 1.209x | 1.146x | 1.018x |
| m11_bsim4_tran | native-strict | 2.9322 | 2.9388 | 1.000x | 1.000x | 0.998x |
| m11_bsim4_tran | native-fast | 2.7485 | 2.6943 | 1.067x | 1.091x | 1.020x |
| m11_bsim4_dc | native-strict | 0.2267 | 0.2282 | 1.000x | 1.000x | 0.993x |
| m11_bsim4_dc | native-fast | 0.2101 | 0.2098 | 1.079x | 1.088x | 1.001x |
| diode100_op | native-strict | 0.2506 | 0.2539 | 1.000x | 1.000x | 0.987x |
| diode100_op | native-fast | 0.2426 | 0.2550 | 1.033x | 0.996x | 0.951x |

## Cold-storage, Zen 2 target, native
| Case | Variant | Generic ms | Baked ms | Generic speedup vs strict | Baked speedup vs strict | Baked vs generic |
|---|---|---:|---:|---:|---:|---:|
| rc_transient | native-strict | 0.0841 | 0.0831 | 1.000x | 1.000x | 1.012x |
| rc_transient | native-fast | 0.0758 | 0.0800 | 1.109x | 1.039x | 0.948x |
| m4_diode_tran | native-strict | 1.4658 | 1.4444 | 1.000x | 1.000x | 1.015x |
| m4_diode_tran | native-fast | 1.2982 | 1.2372 | 1.129x | 1.167x | 1.049x |
| m11_bsim4_tran | native-strict | 2.9663 | 2.8918 | 1.000x | 1.000x | 1.026x |
| m11_bsim4_tran | native-fast | 2.8026 | 2.5462 | 1.058x | 1.136x | 1.101x |
| m11_bsim4_dc | native-strict | 0.2269 | 0.2276 | 1.000x | 1.000x | 0.997x |
| m11_bsim4_dc | native-fast | 0.2140 | 0.2108 | 1.061x | 1.079x | 1.015x |
| diode100_op | native-strict | 0.2353 | 0.2607 | 1.000x | 1.000x | 0.902x |
| diode100_op | native-fast | 0.2252 | 0.2489 | 1.045x | 1.048x | 0.905x |

## Acer, Zen 2 target, native (high concurrent load; indicative only)
| Case | Variant | Generic ms | Baked ms | Generic speedup vs strict | Baked speedup vs strict | Baked vs generic |
|---|---|---:|---:|---:|---:|---:|
| rc_transient | native-strict | 0.3544 | 0.2378 | 1.000x | 1.000x | 1.490x |
| rc_transient | native-fast | 0.1943 | 0.2392 | 1.824x | 0.994x | 0.812x |
| m4_diode_tran | native-strict | 7.6842 | 6.1628 | 1.000x | 1.000x | 1.247x |
| m4_diode_tran | native-fast | 4.7044 | 4.7529 | 1.633x | 1.297x | 0.990x |
| m11_bsim4_tran | native-strict | 10.3611 | 8.4766 | 1.000x | 1.000x | 1.222x |
| m11_bsim4_tran | native-fast | 6.7436 | 6.0930 | 1.536x | 1.391x | 1.107x |
| m11_bsim4_dc | native-strict | 0.5987 | 0.5591 | 1.000x | 1.000x | 1.071x |
| m11_bsim4_dc | native-fast | 0.5364 | 0.5796 | 1.116x | 0.965x | 0.925x |
| diode100_op | native-strict | 0.6724 | 0.7389 | 1.000x | 1.000x | 0.910x |
| diode100_op | native-fast | 0.5495 | 0.6587 | 1.224x | 1.122x | 0.834x |

## Cold-storage numerical results
~~~
case=0 mode=generic {"cells":560,"changed":200,"maxAbsolute":1.509903313490213e-14,"maxScaled":4.543663758453844e-13,"axisMax":2.117582368135751e-21}
case=0 mode=baked {"cells":560,"changed":200,"maxAbsolute":1.509903313490213e-14,"maxScaled":4.543663758453844e-13,"axisMax":2.117582368135751e-21}
case=1 mode=generic {"cells":10200,"changed":4242,"maxAbsolute":8.881784197001252e-15,"maxScaled":8.947351861227067e-12,"axisMax":1.3234889800848443e-22}
case=1 mode=baked {"cells":10200,"changed":4242,"maxAbsolute":8.881784197001252e-15,"maxScaled":8.947351861227067e-12,"axisMax":1.3234889800848443e-22}
case=2 mode=generic {"cells":7392,"changed":3199,"maxAbsolute":1.0902390101819037e-13,"maxScaled":0.00006316294610213258,"axisMax":1.8870057723865944e-24}
case=2 mode=baked {"cells":7392,"changed":3187,"maxAbsolute":9.43689570931383e-14,"maxScaled":0.00004804516012196008,"axisMax":1.6543612251060553e-24}
case=3 mode=generic {"cells":1170,"changed":243,"maxAbsolute":2.220446049250313e-16,"maxScaled":0.00021094237467877977,"axisMax":0}
case=3 mode=baked {"cells":1170,"changed":243,"maxAbsolute":2.220446049250313e-16,"maxScaled":0.00021094237467877977,"axisMax":0}
case=4 mode=generic {"cells":404,"changed":199,"maxAbsolute":5.4001247917767614e-12,"maxScaled":1.1298561342665179e-12,"axisMax":0}
case=4 mode=baked {"cells":404,"changed":199,"maxAbsolute":5.4001247917767614e-12,"maxScaled":1.1298561342665179e-12,"axisMax":0}
case=0 mode=generic {"cells":560,"changed":190,"maxAbsolute":1.509903313490213e-14,"maxScaled":4.543663758453844e-13,"axisMax":2.117582368135751e-21}
case=0 mode=baked {"cells":560,"changed":190,"maxAbsolute":1.509903313490213e-14,"maxScaled":4.543663758453844e-13,"axisMax":2.117582368135751e-21}
case=1 mode=generic {"cells":10200,"changed":4242,"maxAbsolute":8.881784197001252e-15,"maxScaled":8.947351861227067e-12,"axisMax":1.3234889800848443e-22}
case=1 mode=baked {"cells":10200,"changed":4242,"maxAbsolute":8.881784197001252e-15,"maxScaled":8.947351861227067e-12,"axisMax":1.3234889800848443e-22}
case=2 mode=generic {"cells":7392,"changed":3209,"maxAbsolute":1.0835776720341528e-13,"maxScaled":0.000048186565277485386,"axisMax":1.6543612251060553e-24}
case=2 mode=baked {"cells":7392,"changed":3214,"maxAbsolute":1.5210055437364645e-13,"maxScaled":0.00004405750317282293,"axisMax":2.481541837659083e-24}
case=3 mode=generic {"cells":1170,"changed":242,"maxAbsolute":2.220446049250313e-16,"maxScaled":0.00021094237467877977,"axisMax":0}
case=3 mode=baked {"cells":1170,"changed":242,"maxAbsolute":2.220446049250313e-16,"maxScaled":0.00021094237467877977,"axisMax":0}
case=4 mode=generic {"cells":404,"changed":199,"maxAbsolute":5.4001247917767614e-12,"maxScaled":1.1298561342665179e-12,"axisMax":0}
case=4 mode=baked {"cells":404,"changed":199,"maxAbsolute":5.4001247917767614e-12,"maxScaled":1.1298561342665179e-12,"axisMax":0}
~~~

## Reproduction
Source: abe04e2; scripts build-fast-math-plugin.sh, build-fast-math.sh, fast-matrix.sh, summarize-fast-matrix.mjs and compare-dumps.mjs. Raw data in /home/jan/build/ngspice-rs-fast/circuit-lto/optimized; cold-storage logs under cold-storage. SHA256 of native matrix logs:
75fef7274a8876fd73aed19f3ee7e4db59fd80f381c78fc320fca3a7a13713fa  /home/jan/build/ngspice-rs-fast/circuit-lto/optimized/cold-storage/znver2/native-fast/matrix.log
7d408c8e88a3a33d7d7ca34a76253a5910ce60e8298faa838a5a0c38ed5154f9  /home/jan/build/ngspice-rs-fast/circuit-lto/optimized/cold-storage/znver5/native-fast/matrix.log

## PGO matched strict and fast controls, cold-storage
Same previous PGO profiles; two benchmark harness hash warnings on both sides, matched source and flags apart from plugin.

### znver5
| Case | Variant | Generic ms | Baked ms | Generic speedup vs strict | Baked speedup vs strict | Baked vs generic |
|---|---|---:|---:|---:|---:|---:|
| rc_transient | pgo-strict | 0.0663 | 0.0622 | 1.000x | 1.000x | 1.066x |
| rc_transient | pgo-fast | 0.0650 | 0.0630 | 1.019x | 0.987x | 1.032x |
| m4_diode_tran | pgo-strict | 0.9867 | 0.9992 | 1.000x | 1.000x | 0.988x |
| m4_diode_tran | pgo-fast | 0.9403 | 0.9148 | 1.049x | 1.092x | 1.028x |
| m11_bsim4_tran | pgo-strict | 2.6539 | 2.7664 | 1.000x | 1.000x | 0.959x |
| m11_bsim4_tran | pgo-fast | 2.5861 | 2.6788 | 1.026x | 1.033x | 0.965x |
| m11_bsim4_dc | pgo-strict | 0.2048 | 0.2030 | 1.000x | 1.000x | 1.009x |
| m11_bsim4_dc | pgo-fast | 0.1915 | 0.1927 | 1.070x | 1.054x | 0.994x |
| diode100_op | pgo-strict | 0.2034 | 0.2129 | 1.000x | 1.000x | 0.955x |
| diode100_op | pgo-fast | 0.1998 | 0.2023 | 1.018x | 1.052x | 0.988x |

### znver2
| Case | Variant | Generic ms | Baked ms | Generic speedup vs strict | Baked speedup vs strict | Baked vs generic |
|---|---|---:|---:|---:|---:|---:|
| rc_transient | pgo-strict | 0.0637 | 0.0611 | 1.000x | 1.000x | 1.043x |
| rc_transient | pgo-fast | 0.0614 | 0.0571 | 1.039x | 1.071x | 1.075x |
| m4_diode_tran | pgo-strict | 1.0232 | 0.9772 | 1.000x | 1.000x | 1.047x |
| m4_diode_tran | pgo-fast | 0.9120 | 0.8849 | 1.122x | 1.104x | 1.031x |
| m11_bsim4_tran | pgo-strict | 2.9886 | 2.6501 | 1.000x | 1.000x | 1.128x |
| m11_bsim4_tran | pgo-fast | 2.7717 | 2.5748 | 1.078x | 1.029x | 1.076x |
| m11_bsim4_dc | pgo-strict | 0.2096 | 0.2110 | 1.000x | 1.000x | 0.993x |
| m11_bsim4_dc | pgo-fast | 0.2034 | 0.2046 | 1.030x | 1.031x | 0.994x |
| diode100_op | pgo-strict | 0.2088 | 0.2094 | 1.000x | 1.000x | 0.998x |
| diode100_op | pgo-fast | 0.1907 | 0.2010 | 1.095x | 1.042x | 0.949x |

PGO absorbs much of the benefit. Blanket fast improves native non-PGO results but adds only about 2–7% generic and -1–9% baked on Zen 5 PGO; Zen 2-targeted PGO on the same host gives 3–12% generic and 3–10% baked. Keep as an experiment, investigate flags individually only if worth the engineering cost. No production change.
