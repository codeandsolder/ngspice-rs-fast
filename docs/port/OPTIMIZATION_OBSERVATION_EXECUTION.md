# Selective objective observation execution (#120)

## Decision and ownership

This extends the established #114 objective-only `ScenarioPlan` /
`PlotObjectivePlan` and #54 physical `ObservationPlan`. It does not add
another optimizer, alter the circuit's observation plan or decide which
analysis state is numerically relevant. The solver always loads and solves
the **complete** physical system and retains the same accepted histories.

`PlotObjectivePlan::required_columns()` already computes the sorted,
deduplicated full-header indices of every necessary raw vector: time or
frequency axis, direct objectives, signed/differential operands, online
measurement events, deferred evaluation/Fourier operands and retained history.
The producer must not try to infer these dependencies by parsing names.

A `PointSink` may opt into a compact point layout by returning
`selected_columns()` after it has accepted the complete original header.
Ordinary sinks return None and receive full rows unchanged. The objective
sink verifies the complete source schema, name and plotname at begin;
the immutable objective plan has **already** bound the same requests
against a compact header comprised of exactly the selected raw variables.
Both the original full-header plan and its compact child are prepared when
`PlotObjectivePlan::bind` executes; thousands of candidate sessions share
them instead of rebuilding column/measurement mappings at each plot begin.
This initial bind uses the same column-expression, event, axis, Fourier
and numeric-limit policies, not a second interpretation engine.
`ScenarioSink` remembers whether the producer negotiated selection:
compact producers must send compact rows, while already materialized
`PointSink::complete` and direct full-row callers still submit complete
rows, validated against the original header and copied through one
reusable compact scratch vector.

An entirely ground-only objective may legitimately select **zero** raw
columns. Its empty compact row still counts as an accepted point, with
the full original plot checked at begin. It is not a fake single-zero
vector or a rootless pole-zero plot.

## Producer-side binding

`SelectedOutput::bind` validates ascending, distinct indices in the full
schema once per plot and partitions the required ordinals into scale,
solved unknowns and independently bound device observations. It owns a
single reusable compact output buffer. AC projects complex solved values;
DC sweep and transient project real solved values as Complex::real.
No full temporary plot row, placeholder zeros, per-point vector-name
resolution or per-point buffer allocation is necessary on this path.

`PreparedRun::observe_real_selected` and
`PreparedRun::observe_parameters_selected` enumerate only requested
indices from the circuit's immutable bound `ObservationPlan`. The existing
`PointLoads` point-local lazy cache guarantees exactly one isolated device
stamp for multiple selected current/power asks on that device. An empty
selected set performs no isolated device load and no spectral
`observation_parameter` call. Unselected AC current/power asks that the
spectral device contract does not support are not spuriously evaluated.
Each selected scalar still validates finiteness and the existing physical
context. The ordinary all-queries methods retain full observation semantics.

## Driver lifecycle / C numerical fidelity

- **AC:** the complete source schema is announced, then required columns
  are negotiated once. Every frequency still performs the same small-signal
  operating-point update/linearization and complex solve. Only requested
  spectral asks and compact output values are produced. Frequency points,
  phase and branch-current ordering remain unchanged.
- **DC sweep:** the entire nonlinear solve and predicted/previous state
  history run unchanged. Selected isolated real observations are obtained
  using the same effective model context and source-override list from the
  solved trial *before* `accept_point`. The same `firstTime` history
  rotation, sweep order and failure-before-acceptance contract remain.
- **Transient:** `observe_step` is evaluated against the converged trial,
  original integration forcing, breakpoint side, and unchanged
  pre-commit state before the ordinary commit. The initial t=0 sample
  keeps its operating-point observation contract; UIC and samples
  before the requested start remain omitted. Adaptive accept/reject logic,
  discontinuities, delay lines, matrix shape and state advance remain
  independent of whether outputs are requested.
- **Materialized drivers / retained bias:** `PointSink::complete` still
  replays a complete existing plot into the compact objective sink. That
  avoids retaining unused cells in objective state, but **does not**
  claim producer-side savings for calculations that already built
  the full plot.

No accepted point is filtered because the consumer did not request a
specific vector. A selected output cannot weaken model validation,
solver convergence or point sequence. Failure on a requested device
quantity is still propagated; ignoring an unrequested one is the
intended selective execution behavior.

## Regression and cost measurement

`tests/optimization_observation_execution.rs` compares compact objective
traces against independently materialized full plots on identical
accepted DC and adaptive transient grids, including source override
parameter asks and a pulse discontinuity. An AC case binds an unrequested
current/power spectral ask that would fail in ordinary full output and
demonstrates successful selected-voltage output, phase and frequency.
The existing `tests/optimization_objectives.rs` exercises full-header
schema errors, multiple ordered plots, last/ground/trace, event crossing
and Fourier equality. Device-local observation tests count isolated
stamps: duplicate current/power queries on the same selected device
stamp once, unselected devices never stamp, and an empty query never
loads a device.

The native performance comparison must use a representative device
circuit with several unrequested extra quantities and many accepted
points. Report source dimension, number of selected full-plot columns,
number of isolated device loads and point count, elapsed full vs
selected execution and retained objective cell count separately.
Selected trace limits deliberately retain caller-requested values;
full-row Plot cells are not retained by objective sinks.

A machine-profiling tool such as Valgrind DHAT may measure allocations
on the resulting test executable, but it must not change the production
Rust allocator or introduce unsafe code. Acer's current Valgrind requires
the platform's glibc debug symbols to start; install the matching
`libc6-dbg` package as root before using it. Distinguish instrumented
allocation metrics from uninstrumented throughput.

## Boundaries

#120 owns the output handshake, one-time checked projection and selective
device asks through AC, DC sweep and transient. #114 continues to own
objective and online-measure reduction semantics. #54 continues to own
physical device-query resolution and point-local isolation. Circuit
topology, nonlinear algorithms, accepted history and existing output
selection for non-objective consumers are unchanged.

## Acer reference measurements — 2026-11-23

Run the explicitly ignored `benchmark_real_producer_selection` in
`tests/optimization_observation_execution.rs`, with pinned nightly, managed
sccache and the per-worker target directory. Three complete repeated DC
sweeps of a 24-resistor example, each at 1001 accepted points: the ordinary
plot had 51 raw columns and retained 51,051 raw cells; the selected producer
negotiated three columns (scale, node voltage, R1 power) and retained no
raw history cells (while returning both requested output traces). The
ordinary observation producer isolated 24 distinct device loads per point;
the selected producer isolated only the requested R1 per point. Total measured
execution: 72.532 ms full, 33.897 ms selected, 2.140x speedup for this test
profile, with the same numerical grid and source values. This does not imply
universal 2x solver acceleration: savings are from device observation stamps,
row materialization and full plot retention rather than LU convergence.

This native benchmark does not count allocations directly: the project forbids
unsafe global allocator instrumentation, and Acer's Valgrind requires the
matching loader debuginfo package (`libc6-dbg`, installed as root) before DHAT
can collect allocation totals. Keep allocation counting separate from
uninstrumented throughput and the C-compatible solve timing.
