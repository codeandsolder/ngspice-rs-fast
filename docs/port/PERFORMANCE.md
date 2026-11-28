# Representative performance workloads

Tracking issue [#96](https://github.com/codeandsolder/ngspice-rs-fast/issues/96).
This suite establishes the scalar implementation's numerical and performance
reference before changes to arithmetic, threading, SIMD or GPU execution.
Timing is descriptive, never a pass/fail threshold.

## Fixed work

The ordinary integration test runs small instances of every workload. The
nine ignored tests exercise larger deterministic cases. No external circuit
files, randomness, network access or C executable is required.

| Test | Full workload |
| --- | --- |
| sparse_ladder | 1,024 nodes, tridiagonal resistive Laplacian, 32 iterations per phase |
| sparse_mesh | 32 × 32 grounded resistive mesh, 24 iterations per phase |
| complex_frequency_ensemble | 24 × 24 mesh, shunt capacitance, 64 angular frequencies; fresh vs reused pivot order |
| rc_transient | 64 RC sections, finite rise/fall PULSE, 100 us transient, 100 ns maximum step |
| nonlinear_mixed | 48 heterogeneous CMOS inverter/diode branches (240 devices plus sources), 65 DC points; separate mid-transfer operating-point load |
| compact_models | 96 repeated BSIM3/BSIM4 instances on shared model cards plus individual resistors, 33 DC points; separate settled-bias load |
| rf_multiport | 8 equal-impedance ports on a reciprocal grounded RC network, 64 frequencies |
| ltra_history | lossy RLC line, periodic excitation, 300 ns transient, 200 ps maximum step; actual accepted/rejected/breakpoint counts |
| parameter_ensemble | 16 resistor values × 8 input voltages, sequential DC analysis and 128 independent DC solves |

Real matrices use edge stamps, strictly positive diagonal shunts and
binary-exact nonuniform conductances. A predetermined solution and a separate
matrix-vector product generate the RHS and check both forward error and absolute
residual. Complex RHS generation and residual checks use the real G and C
operators independently of complex assembly. Nonfinite values and zero pivots
fail; 1e-10 absolute is specific to these bounded-scale workloads.

Full decks check physical invariants: RC bounds, inverter endpoint polarities,
compact-model drain bounds, reciprocal/passive transmission, delayed response,
and the exact resistor-divider solution at every parameter point. DC load
kernels measure the real device stamp path at an established operating point
and check the loaded residual. Retained output is encoded and decoded
losslessly; non-retaining output checks plot count, row count and row width.
Existing live C conformance remains the detailed independent physics oracle.

## Direct nonlinear diode loading (#99, #164)

The finalized topology preserves the C device load order. Each diode samples
its model and accepted-state inputs, evaluates its complete junction physics,
and commits the MNA/RHS, charge and limiting state before the next device.
There is **no diode tile plan, buffering or separate SIMD load path**.
The explicit diode sampling/evaluation/commit decomposition remains useful
for independent numerical verification without changing scalar execution.

The previous homogeneous 64-lane load planner (#99) was retired after release
measurements showed that scalar gather/queue/commit was consistently slower
than direct scalar loading. An experimental four-lane portable-SIMD finish
for complete diode physics (#164) preserved all tested numerical bits but
also slowed complete device loading. On native Ryzen 7 4700U and generic
x86-64, and independently on Intel Celeron N4100, the overhead remained
material across 4–4,096-device networks; exact full-physics and simpler
diode model variants were both measured. No approximate vector exponential
or reduced model coverage was accepted.

The experimental branch commits, all three host/compiler results, individual
load/factor+solve medians and the reproduction harness are recorded in the
[issue #164 engineering log](https://app.notion.com/p/421f74be9020817eaefcd4ac4e1fb933).
Future vectorization must show **whole-load** gains with C-reference fidelity
before reinstating a batch planner. The regular diode physics tests, full C
golden verification and ordered scalar stamps remain the regression gates.

## Real Newton workspace ownership (#293)

The Newton driver retains current, next, correction RHS and factor scratch in
its execution workspace. Each successful iteration swaps the two solution
vectors; convergence transfers both vectors with the trial that was loaded at
the accepted iterate. Transient retries and continuation stages recycle consumed
vectors only after the last reader. A DC sweep owns one workspace across points,
including predicted-point retries and geometry-driven topology changes. Repeated
frequency-dependent bias and mixed-signal operating-point solves recycle the
same slots; a one-shot small-signal bias drops its DC scratch after use.
The retained primal/adjoint solve and
immutable LU snapshot interfaces accept caller-owned scratch and output, so
repeated right-hand sides need not allocate solution vectors.

A circuit's bound pattern still owns its mutable real/complex factor-frame
transaction: sequential analyses intentionally inherit that frame and its pivot
history. Independent optimizer sessions construct independent circuits and
caches. Immutable snapshots remain independent of subsequent refactors;
separating the topology from this transaction further requires evidence of
shared-pattern contention or copy-on-write costs, not just removal of a lock.

### Matched release allocation evidence

The measured production changes through `7ed5b664` were compared against
`f53dfcc7` on the same Ryzen 7 4700U host with the pinned 2026-11-23 nightly
and mandatory managed sccache. Both builds used the same deterministic
workloads, lockfile, release profile and CPU affinity. Allocation counting
used a separate benchmark-only allocator; native timing used the normal
allocator. All compared output point counts and bitwise plot checksums match.

| Workload | Allocations before | Allocations after | Allocated bytes before | Allocated bytes after |
| --- | ---: | ---: | ---: | ---: |
| 2,000 retained RHS, 32 unknowns | 2,000 | 0 | 512,000 | 0 |
| 2,000 retained RHS, 256 unknowns | 2,000 | 0 | 4,096,000 | 0 |
| 2,000 retained RHS, 1,024 unknowns | 2,000 | 0 | 16,384,000 | 0 |
| 2,000 coefficient refactors at each size | 2,000 | 0 | 16,000 × order | 0 |
| RC transient, 16 sections | 2,877 | 1,300 | 1,044,586 | 817,106 |
| RC transient, 64 sections | 3,246 | 1,669 | 3,747,720 | 2,913,520 |
| RC transient, 256 sections | 4,826 | 3,249 | 14,555,902 | 11,294,822 |
| BSIM3 nested DC | 3,025 | 2,344 | 394,282 | 338,250 |
| BSIM3 transient | 2,965 | 1,127 | 552,732 | 419,780 |
| BSIM4 transient | 1,315 | 614 | 505,970 | 392,362 |

The first three rows and the refactor row exclude initial storage growth.
Whole-circuit allocation counts include the complete analysis after circuit
construction; output collection, device and state preparation still allocate.

A separate rejected-step RC run on `428828a` exercised 100 analyses per
trial, with 30 accepted steps, two rejected steps and two breakpoints in every
analysis. Allocations dropped from 308 to 207 and allocated bytes from 19,074
to 16,074 per analysis. All six trial checksums were `3cf7c32dc7f82248`.

Five alternating native runs under `perf stat` gave median instruction counts
of 5.596 billion versus 5.621 billion for the combined linear/nonlinear/RC
workload, and 2.075 billion versus 2.094 billion for the candidate ensemble.
Median cycles were 1.738 versus 1.726 billion and 1.051 versus 1.024 billion,
respectively. Native wall times vary under concurrent host load; six reversed
pairs on both four-core groups did not reproduce an initial apparent large
ensemble slowdown. This is a demonstrated allocation reduction, not a claim
of a general throughput gain. The factor traversal, pivot policy and
floating-point equations are unchanged.

The harness, binary checksums, raw allocation and hardware-counter logs and
paired timing summaries are recorded in the
[issue #293 engineering log](https://app.notion.com/p/426f74be90208135b584d45aab4e8389).
The existing factor, Newton, DC continuation, transient rejection and
independent-candidate regression tests exercise cache isolation and acceptance
semantics.

## Instantaneous rank-reduction storage (#299)

The rank reducer keeps a reusable sparse accumulator with a column-to-slot
index and a compact filled-coefficient vector. Removing an entry swaps in the
last live coefficient and updates its index. The dense index is linear in the
column count; coefficient storage is linear in actual fill, not in the square
of the matrix dimension. Immutable pivot rows are stored in one contiguous
arena, sorted by column, and referenced by ranges. A reusable min-heap with
membership flags visits each pending pivot once in ascending ordinal order;
previously queued but cancelled pivots still incur the original work charge.

Higher Taylor degree, coefficient magnitude and greatest-column tie breaking
are unchanged. Each individual wide arithmetic operation, propagated error,
normalization, ascending-column back substitution and uncertainty bound is
unchanged. The existing 67,108,864-operation work budget, physical row
priority, gauge selection and outer mandatory equation checks remain intact.
No alternate dense solver, approximation or tolerance change is introduced.
The differential regression compares the exact pivot sequence, independent
rows, remaining budget, binary64 solution and uncertainty on 544 generated
systems with the original tree representation (digest `cefcb985a5d8faee`).

### Reproduction and measurements

Base `b7e7b6a296675fa2e9ee84604340bbfb483480fa`; identical benchmark test
sources in both builds. AMD Ryzen 7 4700U, x86-64 Linux, pinned nightly
2026-11-23, managed sccache, isolated generic release builds. All 18 rank
unit tests/benchmarks passed in each build. Use the complete test names:

~~~sh
cargo +nightly test --release --locked --lib analysis::initial::system::rank:: -- --include-ignored --nocapture
~~~

- `storage_tests::benchmark_rank_reduction_workloads`: cyclic 32/256-column
  sparse, 128-column intermediate fill and 64-column dense systems with
  redundant equations and exact solution one.
- `projected_tests::benchmark_high_index_rank_reduction`: derivative chains of
  depths 8, 32 and 60 (99, 1,155 and 3,843 columns), requiring only the initial
  coordinate prefix rather than inventing constraints on free high-order jets.
- `storage_tests::benchmark_near_dependent_rank_reduction`: 256 columns,
  binary-exact 2^-30 separations, row scales 2^-200 through 2^200, redundant
  equations and nonzero propagated uncertainty.

Run each release test executable under `perf stat -e instructions:u,cycles:u,task-clock`
with its exact test name, `--exact --ignored --nocapture`, and separately
under `valgrind --tool=dhat`. Three interleaved runs per variant, medians:

| Workload | Instructions before | Instructions after | CPU ms before | CPU ms after |
| --- | ---: | ---: | ---: | ---: |
| Sparse-to-dense | 28,578,516,344 | 24,652,212,104 | 4,281.53 | 3,133.32 |
| High-index | 412,398,360 | 259,175,173 | 81.72 | 41.94 |
| Near-dependent | 277,355,319 | 179,275,015 | 43.84 | 27.45 |

Process-wide DHAT measurements (including construction and test harness):

| Workload | Allocations before | Allocations after | Total bytes before | Total bytes after | Peak live bytes before | Peak live bytes after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sparse-to-dense | 309,211 | 14,231 | 94,210,184 | 24,521,065 | 450,886 | 399,623 |
| High-index | 252,304 | 6,354 | 76,917,424 | 34,343,105 | 2,442,082 | 883,342 |
| Near-dependent | 182,631 | 5,431 | 54,373,252 | 14,091,653 | 188,522 | 76,763 |

The complete reduction workload executes 13.7% fewer instructions in the
sparse-to-dense suite, 37.1% fewer in the high-index suite and 35.4% fewer in
the near-dependent suite. CPU task-clock speedups are 1.37x, 1.95x and 1.60x.
Sampling profiles show that B-tree operations disappear from the hot path;
wide arithmetic dominates remaining work. Host wall time was substantially
influenced by concurrent load, so the hardware counters and CPU time are the
comparison, not the DHAT elapsed times. These are rank-reduction workloads,
not entire transient analysis speedups.

## Measurements

Records contain workload, phase, operation count and elapsed nanoseconds.
Repeat steady-state phases three times; use medians per operation.

- Pattern creation, triplet assembly and bound element assembly are separate
  phases. Coefficient nonzero counts describe matrix sparsity (divide by n²).
- Fresh order/factor includes factor-frame construction; retained solve uses a
  factor snapshot with caller-owned scratch and output. The two NI solve phases exercise
  the same bound matrix with changed and then identical coefficients.
- FactorStorage records structural original handle creations, total stored LU
  entries, fill-in handle creations, translated/external order and logical
  numeric bytes. The fill ratio is factor_entries / original_entries
  (including structurally allocated zeroes). It is not heap allocation
  capacity, a count of nonzero values, or a process-wide memory measurement.
- FactorWork records actual order calls, searches, fixed-plan refactors,
  outcomes, and immediate zero-pivot retries. Retaining every existing pivot
  during threshold ordering counts an order attempt but no search. The
  complex phase sums producing-operation snapshots across frequencies and
  reports the observed storage range. Each successful independent snapshot
  solve is counted by the harness after the solve succeeds.
- Real NI measurements read the shared pattern's cumulative work before and
  after the timed pair of changed/identical coefficient solves. They assert
  exactly one numerical refactor and one exact-input hit per pair, plus two
  successful RHS solves and no additional order/search. The warmup and
  post-repeat records are cumulative within the pattern's cache epoch, so their
  difference isolates the measured pair work.
- Full-deck retained, streamed, bias and transient measurements report the
  final pattern cache epoch; an extended/replaced pattern starts a new epoch.
  Those records do not count discarded predecessor epochs, or independent
  snapshot RHS solves (including RF). They must not be interpreted as totals
  of all internal operations in an entire analysis.
- All telemetry reads occur outside the timed kernels. Counter definitions
  and snapshot/reset semantics are documented in [NATIVE_SOLVER.md](NATIVE_SOLVER.md).
  Workload timings and process-wide DHAT measures remain separate from
  numeric storage and work counters.
- Device load excludes solver time and uses the finalized circuit matrix and
  state history. Parse, elaboration, retained analysis, streamed analysis, and
  rawfile codec have separate measurements. Every retained and streaming trial
  gets a fresh circuit. Transient-specific runs report actual accepted/rejected
  work and do not reuse the prior trial's history.
- The independent parameter ensemble carries no earlier Newton seed or factor
  cache across jobs. The final cache epochs of independent jobs are summed,
  and the largest observed factor storage is reported separately. The sequential
  sweep is measured separately.

## Reproduction

Use pinned nightly and managed sccache. Compare generic and host-specific
release builds from the same revision, differing only in target CPU.
Run exclusively, with quiet and thermally stable hardware when possible,
alternating configurations. Record the revision, rustc -Vv, CPU model,
OS, target flags, compiler profile, host load, all trials and medians.

~~~sh
cargo +nightly test --locked --release --test solver_workloads_bench -- \
  --ignored --test-threads=1 --nocapture
cargo +nightly --config 'build.rustflags=["-C","target-cpu=native"]' \
  test --locked --release --test solver_workloads_bench -- \
  --ignored --test-threads=1 --nocapture
~~~

Use a dedicated scoped target root if artifacts must be retained. Never
disable sccache. Use an exact test name to isolate one workload for heap
or CPU profiling. Valgrind DHAT measures process allocation totals and peak
live bytes externally, without adding an allocator or unsafe Rust. On
x86-64 Linux, for example, with an existing dedicated scratch directory:

~~~sh
cargo +nightly \
  --config 'target.x86_64-unknown-linux-gnu.runner=["valgrind","--tool=dhat","--dhat-out-file=/absolute/scratch/mesh-%p.json"]' \
  test --locked --release --test solver_workloads_bench -- \
  sparse_mesh --exact --ignored --test-threads=1 --nocapture
~~~

DHAT totals include harness startup, construction, verification and all three
trials. Label them process totals, not per-kernel allocation counts. For fine
attribution, profile allocation stacks rather than subtract incomparable
executions. DHAT timings are not performance timings. Use generic binaries
under Valgrind if native instructions are unsupported. Remove scratch artifacts
after recording the results.

## Baseline record

Keep raw logs, compiler/host details and actual measurements with a revision.
Do not declare a measured baseline from source dimensions or estimated
complexity alone. The workload contract lives here; dated measurement records
belong in the issue/worklog.

- [2026-11-22 scalar reference](performance-baselines/2026-11-22.md):
  generic/native release timing, nine allocation profiles and numerical records.
- [2026-11-22 sparse factor telemetry](performance-baselines/2026-11-22-factor-telemetry.md): actual storage and work counters, release workload timings and numerical signatures.
