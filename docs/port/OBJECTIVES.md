# Objective-only measurement plans (#114)

A fitting or optimization candidate does not generally need the complete
accepted-point matrix. Bind the requirements once per scenario with
`analysis::optimization::objectives::PlotObjectivePlan::bind`. Supply the
expected plot header (which may be taken from a representative run), its
analysis kind, and ordered `ObjectiveRequests`. The bound plan retains a
zero-row header and resolved column expressions, not the representative
data. `ScenarioPlan::new` receives the expected plot plans in the exact
emission order, including any unrequested bias or summary plots. Construct
a separate plan for each scenario.

For each candidate, prepare its circuit, analysis plans, context and
parameter scope. Reuse the scenario plan by calling `ScenarioPlan::run`
or obtain its sink and drive it via `AnalysisPlan::run_into`. A new
sink has independent accumulators and measurement-name bindings. It
rejects a changed schema, unexpected output, incorrect row width or
incomplete lifecycle. Execution/solver errors and output errors are
distinct. Measurements return typed records or an error; one failed
observation does not turn a different valid observation into a NaN.

## Online versus history-dependent work

- Direct observations: last value, interpolated scalar/complex value at
  a physical coordinate, or an explicitly bounded selected trace.
  Queries accept both increasing and decreasing monotonic physical
  coordinates, using the adjacent samples in traversal order and
  exact-coordinate group agreement, not an invented value at a jump.
  Direction changes (e.g. nested DC sweep resets), nonfinite coordinates,
  and queries outside the interval are explicit errors. Trace/last
  observations need not have an ordered axis. Last values capture one
  compact row of shared dependencies and evaluate each requested expression
  only once at finish; invalid earlier rows cannot poison the last value.
  A sorted coordinate
  schedule dispatches only the targets bracketed by each adjacent sample,
  retaining two rows of only the query operand columns. Results stay in
  request order, including duplicate names/targets. Dispatch work is
  linear in accepted rows and targets (plus validation of exact duplicate
  groups), not their Cartesian product. When the last target in
  the traversal direction is passed, the query cursor stops copying rows;
  axis validation still covers every accepted coordinate. Deck measurements retain the
  separate nondecreasing-axis policy of the retained evaluator.
- Static measurements: extrema (including tie and time policy), AVG,
  INTEG, RMS, FIND, WHEN, TRIG/TARG, derivatives and event-triggered
  value queries, and AC/SP phase/gain margins. Retain bounded per-card
  state, even for LAST crossings: an earlier candidate crossing can
  be replaced without storing intervening rows. Derivative stencils
  preserve the first inclusive bracket and duplicate-coordinate policy.
- Scalar PARAM/EXPR records share the existing vector-first two-pass
  resolver and parameter scope; result names shadow earlier bindings
  exactly as for retained measurements.
- A deferred setter depending on earlier results may select an arbitrary
  window or point after the run. Fourier selects the final period and
  resamples the waveform. These use a bounded projection of raw operand
  columns plus scale; no unrelated columns are retained. Set
  `retained_cell_limit` explicitly for such requests. Fourier uses the
  ordinary default resolution policy; configured CLI Fourier rendering
  is not part of this objective API.

`required_columns()` exposes the complete sorted raw dependencies,
including scale whenever used. `retained_columns()` exposes only the
history projection. Zero retained cells means no full accepted-row
history in this sink, not zero solver memory or zero allocation.
Explicitly requested traces have their own point limit. Driver-side
selective observation scheduling is a separate producer-side task
(#120); analyses that currently materialize outputs internally may
still do so.

## Validation and benchmark

`cargo +nightly test --test optimization_objectives` covers retained
parity, duplicate times, out-of-range queries, malformed lifecycles,
scope ordering, limited history, and an analytic complex RC response.
The ignored `objective_only_batch` and `full_plot_batch` benchmarks
compare identical measurement requests over 20 candidates with 50,001
accepted points and 65 columns, in the same test executable. The
baseline constructs a packed Plot and evaluates the retained
measurement API. The streaming case binds once and reuses the plan.
The benchmark isolates output reduction, not numerical solver cost.

Acer, 2026-11-22, initial streaming implementation (prior to #92
accuracy reconciliation), pinned nightly and managed sccache:

| Metric | Retained | Streamed |
|---|---:|---:|
| Accepted cells held at once | 3,250,065 | 0 |
| Median native batch wall time | 2.57 s | 0.81 s |
| Median process RSS peak | 56,108 KiB | 4,688 KiB |
| DHAT total allocated bytes, 20 candidates | 2,747,365,191 | 278,806 |
| DHAT peak live heap bytes | 68,722,894 | 44,435 |

Checksums match (61.54700538418677). DHAT includes fixed harness/plan
costs. Host load is shared; these timings isolate the output stage,
not the circuit solver. Run each ignored benchmark separately with
`--exact --ignored --nocapture` on the test executable and profile
heap allocations with `valgrind --tool=dhat`.

### Indexed direct-coordinate dispatch

Acer, 2026-11-22: native optimized dev profile, pinned nightly and
managed sccache. Compare commit `54e7845` (direction-independent
individual query scans) with `db39d74` (coordinate cursor). Each plot
has 50,001 accepted rows and 65 raw columns; the requested operands span
1–64 distinct columns. Five alternating process CPU measurements after
warmup, medians below. Candidate repetition counts vary across cases,
not between variants; includes constructing rows and reducing/finishing
the scenario, excludes actual circuit solving.

| Queries | Raw operand columns | Candidates | Before (ms) | Cursor (ms) |
|---:|---:|---:|---:|---:|
| 1 | 1 | 20 | 34.505 | 35.180 |
| 2 | 1 | 20 | 47.834 | 37.424 |
| 16 | 4 | 10 | 122.666 | 24.490 |
| 64 | 64 | 4 | 194.140 | 42.723 |
| 256 | 1 | 3 | 517.413 | 7.564 |
| 2,048 | 8 | 1 | 1,430.632 | 6.473 |

All checksum strings match for both variants and all repetitions. The
single-query case requests the initial sample and is effectively
unchanged; the largest case is 221x faster. The cursor retains two
compact operand rows, plus per-target result states. It has no
accepted-row history and does not skip subsequent physical-axis
validation after all targets are dispatched. Exact-coordinate groups
still require all their accepted samples to agree.

Probe source `query_batch.rs`, link script, timing script and raw
logs are in the campaign evidence folder
`logs/worker-arch-1122-b6d9-issue114/probe/`.
