# SPARSE lifecycle comparison — 2026-11-22

Issue #34; benchmark suite introduced by #96.

- Baseline d3ca880a5502906fb7c294f198ea9889abbeb3f6; #34 candidate daa12d586eed1a078b3d2c61a1ee4d14526b42c9 (merged baseline into issue branch).
- Same Ryzen 7 4700U, x86-64 generic release/debug=1, pinned rustc 1.102.0-nightly (6905905bb), managed sccache. Independent binary hashes f0d59083cfe191247e92a20d65044877a1b00c7f458f12936ec46d03f16bb936 (base), d244bc997b1ef379061b9bd8c7946c8d0fdece2225674c1b30119242a1343315 (candidate).
- Three alternating pairs; 18 isolated workload runs and six independent DHAT profiles passed. One-minute host load 27.73–30.46 (8 logical CPUs).
- Timing is preliminary: mesh assembly, which #34 does not change, also varied. No demonstrated speedup or regression from this noisy wall-clock comparison. All reported error/residual/workload signatures are identical, and all allocation block counts unchanged.

## Timing

Microseconds per operation; aggregate median [min–max]; three per-run medians. The unchanged assembly phases are negative controls.

| Workload | Phase | Base | Candidate |
|---|---|---:|---:|
| complex_frequency_ensemble | fresh_order_assembly | 41.844 [31.429–76.093]; 32.087, 40.504, 52.158 | 42.885 [30.917–48.783]; 44.090, 30.920, 46.643 |
| complex_frequency_ensemble | fresh_order_factor | 5263.310 [3962.097–12148.823]; 3989.680, 5263.310, 10355.537 | 5433.304 [3987.232–7071.310]; 6752.950, 4019.283, 5433.304 |
| complex_frequency_ensemble | fresh_order_solve | 47.874 [24.124–210.809]; 25.024, 47.874, 86.470 | 54.473 [23.201–84.733]; 54.473, 25.043, 67.836 |
| complex_frequency_ensemble | reuse_order_assembly | 41.872 [30.371–45.894]; 30.622, 41.872, 44.958 | 35.349 [30.805–46.800]; 35.349, 30.909, 39.052 |
| complex_frequency_ensemble | reuse_order_factor | 313.514 [235.921–362.885]; 243.982, 313.514, 359.011 | 275.109 [231.924–328.375]; 275.109, 239.690, 327.354 |
| complex_frequency_ensemble | reuse_order_solve | 26.549 [19.100–41.195]; 19.240, 26.549, 28.872 | 21.514 [18.777–26.132]; 22.511, 19.387, 24.509 |
| sparse_ladder | bound_assembly | 119.238 [87.237–249.408]; 120.195, 88.588, 110.029 | 110.380 [91.138–213.258]; 98.077, 105.318, 165.764 |
| sparse_ladder | changed_coefficients_ni_solve | 29.214 [24.108–33.317]; 31.221, 24.312, 29.214 | 29.200 [24.720–47.351]; 26.794, 29.200, 40.669 |
| sparse_ladder | fresh_order_factor | 112.004 [87.285–152.131]; 131.862, 89.086, 107.339 | 112.658 [95.023–197.539]; 96.803, 110.739, 166.013 |
| sparse_ladder | identical_coefficients_ni_solve | 13.363 [10.989–24.572]; 14.287, 11.060, 13.363 | 13.411 [11.292–38.833]; 11.978, 13.411, 16.681 |
| sparse_ladder | pattern_setup | 706.889 [400.048–58937.397]; 58937.397, 706.889, 400.048 | 494.306 [428.622–730.654]; 730.654, 428.622, 494.306 |
| sparse_ladder | retained_solve | 11.999 [10.194–18.323]; 13.090, 10.250, 11.999 | 12.134 [10.728–14.746]; 11.097, 12.134, 14.698 |
| sparse_ladder | triplet_assembly | 152.434 [120.695–314.503]; 183.748, 122.386, 149.691 | 152.083 [130.629–409.426]; 138.079, 147.874, 208.388 |
| sparse_mesh | bound_assembly | 174.757 [166.736–233.238]; 174.757, 166.860, 208.971 | 208.385 [171.661–475.293]; 174.272, 208.385, 239.384 |
| sparse_mesh | changed_coefficients_ni_solve | 340.897 [326.076–562.524]; 330.546, 329.534, 516.558 | 411.968 [362.064–696.661]; 362.845, 411.968, 615.614 |
| sparse_mesh | fresh_order_factor | 11970.030 [11425.969–28084.270]; 11617.016, 11726.500, 22185.658 | 13514.456 [11516.625–29868.844]; 12541.238, 13514.456, 24187.593 |
| sparse_mesh | identical_coefficients_ni_solve | 27.380 [25.808–38.876]; 26.509, 25.969, 37.767 | 30.520 [28.590–57.172]; 28.689, 29.548, 37.372 |
| sparse_mesh | pattern_setup | 955.972 [667.986–1256.610]; 667.986, 955.972, 1256.610 | 772.343 [753.627–1515.832]; 1515.832, 753.627, 772.343 |
| sparse_mesh | retained_solve | 25.488 [23.969–79.446]; 24.122, 25.488, 40.102 | 31.385 [23.377–38.468]; 27.988, 31.385, 37.415 |
| sparse_mesh | triplet_assembly | 270.149 [250.062–338.968]; 263.635, 264.310, 322.409 | 309.398 [259.961–780.520]; 281.024, 309.398, 375.422 |

## Heap profiles

DHAT process-wide totals: includes workload construction, measurement and verification. Read/write bytes are instrumented heap accesses, not physical memory bandwidth. Only counts/bytes are compared, not profiled elapsed times.

| Workload | Revision | Allocated bytes | Blocks | Peak bytes | Peak blocks | Read bytes | Write bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| sparse_ladder | base | 66,404,031 | 5,377 | 1,193,389 | 94 | 833,263,212 | 355,185,030 |
| sparse_mesh | base | 323,421,071 | 5,146 | 5,298,133 | 94 | 8,239,757,699 | 1,502,603,814 |
| complex_frequency_ensemble | base | 589,874,863 | 10,999 | 2,147,509 | 71 | 7,345,146,821 | 1,780,374,820 |
| sparse_ladder | candidate | 66,403,241 | 5,377 | 1,193,378 | 94 | 811,071,305 | 355,194,653 |
| sparse_mesh | candidate | 323,420,475 | 5,146 | 5,298,122 | 94 | 7,918,254,529 | 1,501,144,384 |
| complex_frequency_ensemble | candidate | 589,871,801 | 10,999 | 2,147,498 | 71 | 7,316,729,115 | 1,780,371,597 |

## Numerical checks

Exact matching reported workload and max-error signatures (not a claim of bitwise equality of every solution entry; the dedicated solver tests cover the pivot and snapshot invariants):

~~~text
workload=sparse_ladder,forward_abs=6.661338147750939e-16,residual_abs=4.440892098500626e-16
workload=sparse_ladder,n=1024,stamps=5116,pattern_nnz=3070,coefficient_nnz=3070
workload=sparse_mesh,forward_abs=1.5543122344752192e-15,residual_abs=3.1086244689504383e-15
workload=sparse_mesh,n=1024,stamps=8960,pattern_nnz=4992,coefficient_nnz=4992
workload=complex_frequency_ensemble,mode=fresh_order,orders_requested=64,reuse_requested=0,forward_abs=2.5989654674417885e-15,residual_abs=5.617333549722722e-15
workload=complex_frequency_ensemble,mode=reuse_order,orders_requested=1,reuse_requested=63,forward_abs=2.5989654674417885e-15,residual_abs=5.617333549722722e-15
workload=complex_frequency_ensemble,n=576,g_nnz=2784,c_nnz=576,frequencies=64
~~~

Earlier isolated #34 ring microbenchmark (1,024 nodes, 1,000 reload/refactors) reported median 86.51 ms before and 49.55 ms after, checksum 3878787878787026 on both. Different load/build conditions mean it is supporting evidence, not a controlled speedup claim.

Raw records: Acer campaign logs with prefix arch-1122-a7c9-34-lifecycle-baseline-; includes source/compiler/binary metadata, individual run records, and DHAT logs. Scoped benchmark worktree and build directories removed.
