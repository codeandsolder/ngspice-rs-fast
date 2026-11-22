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
  Queries use exact-coordinate group agreement, not an invented value
  at a jump. Nonfinite data and queries outside the range are explicit
  errors. Trace/last observations need not have an ordered axis.
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
