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

## Measurements

Records contain workload, phase, operation count and elapsed nanoseconds.
Repeat steady-state phases three times; use medians per operation.

- Pattern creation, triplet assembly and bound element assembly are separate
  phases. Coefficient nonzero counts describe matrix sparsity (divide by n²).
- Fresh order/factor includes factor-frame construction; retained solve uses a
  factor snapshot with caller-owned scratch. The two NI solve phases exercise
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
