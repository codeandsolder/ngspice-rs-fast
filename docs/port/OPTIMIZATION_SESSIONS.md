# Independent candidate execution (#113)

## Architecture

- Shared read-only CircuitParameterPlan (#112/#117) supplies typed candidate binding. Each CandidateWorker owns an independently prepared CircuitCandidate, its circuit, model override context and solver state.
- Each CandidateJob carries its caller-controlled u64 identity and owned full numeric coordinate vector. Errors remain typed and per-candidate: parameter preparation, candidate-specific plan resolution or a scenario execution failure with the ordinal that failed. No partially successful scenario vector escapes as a successful candidate.
- ScenarioAnalyses::Prepared permits reusing already compiled numeric AnalysisPlans. ScenarioAnalyses::Candidate re-resolves explicit dynamic option and hint inputs using the exact candidate RunConfig. ScenarioAnalyses::Deck literalizes the original AnalysisCard expressions (for example .tran stop={param}) against that candidate scope, ensuring full dynamic time/frequency argument fidelity. ScenarioPlan (#114) is the single objective-only sink; trace retention remains bounded by its requested limits. No new plot collector or output evaluator.
- The first scenario starts from independent conditions. Later scenarios reset by default with ScenarioStart::Independent, or intentionally continue previous accepted state with ScenarioStart::Continue. Circuit::clear_candidate_analysis_state invalidates transient history, completed revisions, cached stamps and matrix-coupling state; resetting also occurs on an identical A/A candidate, which otherwise takes the no-op parameter update path.
- CandidateWorker::run_bounded keeps worker setup persistent across a streamed series, emits synchronously and enforces max_candidates. run_parallel builds one worker per requested lane, constructs and retains each non-Send device circuit within its own scoped worker thread, and schedules candidates as workers finish. Only the caller thread emits results, in input ordinal order. The live input+completed reorder window is at most 2 × workers; an unusually slow candidate cannot induce unbounded out-of-order result retention. No unsafe Send/Sync, environment-dependent globals, or GPU concepts.
- This is *independent candidate* scheduling, not serial SPICE .dc continuation. Candidate concurrency and execution are distinct from #135's solver-internal Newton load/RHS workspace work and #116's SIMD/GPU throughput.

## Limit semantics

Known-length inputs whose length exceeds max_candidates are rejected before producing outputs. For a length-unknown stream, the already emitted bounded prefix remains delivered, followed by an explicit limit error. Callback invocation never happens on worker threads; callers must not assume unique candidate IDs, since the stable original input ordinal disambiguates repeated IDs. Each candidate evaluation receives a fresh EvalBudget with the declared EvalLimits.

## Correctness and performance

Run tests/optimization_sessions.rs under the campaign-pinned nightly and managed sccache. Cases: A/B/A and A/A, repeated IDs in deterministic input order, failure followed by success, dynamic input/deck expression resolution, transient UIC seed restart, multi-scenario reset/continuation, parallel numeric equality with serial results, topology rebuild, short tail and work caps. The ignored 10k candidate benchmark reports independent throughput for one and four workers.

The objective-only path returns scalar or limited trace results, rather than plots for every candidate. Further memory reductions inside a single Newton iteration belong to #135 and are deliberately not duplicated.

## Initial native CPU measurements (Acer, 2026-11-23)

Pinned nightly, managed sccache, objective-only operating-point simulation of
10,000 independent candidates with one tuned resistor and a three-element
resistive divider: **289.9 ms** for one worker, **316.5 ms** for four workers.
The tiny workload is communication/setup-bound, so four threads are slower;
this is an architectural correctness baseline, not a demonstrated throughput
speedup. #116 owns chunk scheduling, lane/worker utilization and realistic
multi-device numerical workload optimization. Explicit worker setup, solver
execution and channel costs should be profiled separately on #96 workloads
before choosing a dispatch policy.
