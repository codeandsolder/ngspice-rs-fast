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

### Numerical reconciliation

Retained and streaming integration now use the same compensated panel
accumulator, including weighted RMS evaluated in binary exponent units.
RMS normalizes using the covered axis width only after all panels have
been consumed. No squared ordinate or span-weighted square must be
representable in binary64. Positive normalized quadrature is bounded
by the maximum sampled magnitude. The shared crossing and divided-
difference geometry preserves endpoint and extreme-scale behavior.

The exponent-range regression tests constant panels at all 2,098
binary64 binade exponents and five span scales, including subnormals
and the largest normal binade (10,490 bit-exact cases). An isolated
matched-checksum weighted-square kernel probe on Acer (20 repetitions
of 100,000 four-point panels) measured median CPU time of 84.87 ms
versus 91.07 ms for the intermediate square-root-product formulation.
These kernel timings are not the end-to-end output benchmark above.

Final reconciliation benchmark (commit `494a069`, before this
documentation-only update), same release executable and workload,
three alternating retained/streamed trials:

| Metric | Retained | Streamed |
|---|---:|---:|
| Median userspace task-clock | 714.41 ms | 204.94 ms |
| Median peak RSS | 55,752 KiB | 4,704 KiB |
| DHAT cumulative allocated bytes | 2,747,364,185 | 277,800 |
| DHAT peak live heap bytes | 68,722,112 | 43,653 |
| DHAT cumulative allocated blocks | 2,177 | 1,337 |

All six runs and both DHAT runs produced checksum 61.5470053841868.
The retained baseline uses the same conditioned arithmetic as the
streamed reducer. Output-only CPU speedup is 3.49x. The updated
checksum differs from the historical pre-reconciliation result only
in its last printed digits.

Run with the pinned nightly in release mode and interleave
`full_plot_batch` and `objective_only_batch` under
`perf stat -e task-clock,cycles,instructions` and
`/usr/bin/time -f '%M'`. Separate DHAT profiles reproduce the heap
measurements. Raw logs: `arch-1122-a7c9-114-final-bench-perf.log`
and `arch-1122-a7c9-114-final-{full_plot_batch,objective_only_batch}-dhat.log`
in campaign logs.
