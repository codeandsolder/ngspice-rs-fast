# Repeated evaluations and optimization workload

The simulator must treat repeated circuit evaluations as a first-class workload:
design optimization, model calibration/curve fitting, Monte Carlo and yield
analysis can execute the same topology thousands or millions of times. This is
not equivalent to a sequential DC sweep. Optimizers provide arbitrary candidate
vectors and may need several stimulus, load, temperature or analysis cases per
candidate. An independent evaluation does not inherit another candidate's
Newton state, transient history or error. Within a candidate, intentional
continuation remains a separate policy.

The desired common execution shape is immutable validated topology and parameter
dependency bindings; reusable prepared model/analysis data; independent owned
numeric workspaces; bounded batches with candidate IDs and per-candidate status;
ordered result publication and explicit requested observations/objectives. Use
portable SIMD across independent lanes where profitable, CPU workers across
chunks, and scalar fallback for nonconforming lanes. GPU remains a future backend
for sufficiently large resident batches. Do not require GPU libraries in default
simulation or optimize the CPU layout around GPU transfers. Parameter changes
that alter node, branch, state or stamp layout require validated regrouping or
re-elaboration, not stale reuse.

Related issues: #111 (this workload), #112 (parameter dependency/variation plan),
#113 (independent execution sessions), #114 (objective-only output), #97/#98
(batched sparse arithmetic), #99 (homogeneous device physics), #104 (residuals
and refinement). Do not introduce an optimization algorithm or fitness convention
into core numerical execution.

## Reproducible baseline

`tests/optimization_workloads_bench.rs` exercises arbitrary deterministic
six-parameter candidates and four independent scenarios: a two-pole RC filter
with two different load resistances (25 AC frequencies each), first-order RC
step (eight fixed-time objective samples obtained by streaming interpolation,
not an adaptive-grid average), and diode I-V fitting (ten DC points per
candidate). The expected responses are derived independently from nodal
equations, the ramp/step solution of a first-order ODE and the nominal Shockley
equation. The target parameter vector is fixed; objectives are sums of
normalized squared errors divided by sample count. This is a fixture objective,
not a simulator fitness API. Explicit candidate-domain checks, A/B/A replay,
retained-versus-streaming equality and reversed candidate order test isolation.
The cold baseline parses and elaborates per scenario. The separate repeated DC
benchmark uses existing resistor overrides and independent initial conditions
to compare fresh topology construction with one reused finalized circuit.

Run the normal small correctness test and opt-in larger workload:

~~~sh
cargo test --locked --test optimization_workloads_bench
cargo test --locked --release --test optimization_workloads_bench -- --ignored --nocapture --test-threads=1
~~~

Use managed sccache, pinned nightly and a disposable external build directory.
The opt-in workload contains 128 candidates in each mode, 512 scenarios per
mode, and 16,384 independent divider solves per topology strategy. Report parsing,
preparation and simulation/objective elapsed time separately; don't interpret
cold baseline as optimized performance or treat elapsed time as a correctness
gate. The streaming path currently still produces the full simulator row; it
is a baseline for #114, not evidence that unrequested observations are skipped.
Later add genuine worker/lanewise statuses, topology-change and invalid-model
fixtures, allocation profiles and lifecycle counters as relevant production APIs
land. Keep actual measurements with a dated revision and host-load context.

## Initial measurement — 2026-11-22

Revision `2716b19f714383fd6847b0de6ed8bd5e9535c811`, Acer Ryzen 7 4700U,
pinned nightly, generic release, managed sccache, concurrent fleet activity.
One initial measurement per mode; not a controlled timing comparison.

| Mode (128 candidates, 512 scenarios) | Parse ms | Prepare ms | Evaluate ms | Total ms |
|---|---:|---:|---:|---:|
| Retained plots | 27.122 | 32.296 | 317.218 | 376.636 |
| Streaming objectives | 20.878 | 31.465 | 341.388 | 393.731 |

Both modes validated every sample against independent physical equations
and equivalent objectives in reversed candidate order. The small A/B/A
and invalid-domain tests also passed. Full scenario evaluation still
constructs topology per case. Streaming lowers the required result retention
but does not yet avoid full row/observation generation; this run does not
establish a runtime benefit.

| Independent divider operating points (16,384) | Time ms |
|---|---:|
| Fresh topology per candidate | 740.463 |
| Reuse finalized circuit and existing numeric overrides | 65.795 |

The latter is 11.25x faster in this initial run. Both paths use independent
initial conditions and verify every solution against the analytic divider
formula. This is strong evidence for prioritizing reusable topology and
parameter slots, not a general speedup claim for nonlinear/transient
optimization. Test execution time 1.59s; build times excluded.

The existing reference includes two topology strategies but the simulator
does not yet support arbitrary prepared numeric parameter evaluation or
general independent multi-analysis sessions. #112/#113 must add those
without introducing state leakage. #114 tracks selective objective
execution; #115 tracks reuse of sensitivities/adjoints to avoid repeated
primal solves for smooth objectives.
