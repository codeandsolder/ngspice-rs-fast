# #140 — Reusable numeric netlist value sites

## Decision

Retain the single immutable CircuitParameterPlan from #112 and the shared ParameterPlan from #117. The sites retain typed baseline values, original SiteKind, SourceLoc provenance and transitive independent coordinate indexes. No second evaluator or netlist reconstruction path is introduced.

The binder uses the existing bounded RefScan to capture direct lexical names, including free variables through nested .func calls, and resolves them to stable defining slots once. CandidateParameters::evaluate_bound uses the original evaluate_root evaluator, so error locations, arithmetic domain, finite-value and operation-budget semantics remain unchanged.

Each independent candidate session owns committed raw and numparam-decimal-rounded site values, plus a reusable vector of staged changes. A site is evaluated only when its directly referenced resolved slot actually changes, rather than when any transitive input changes. This avoids work when derived parameter values cancel. Decimal formatting/read-back is skipped if the raw f64 result is bit-identical. Every effective non-baseline numeric override is still replayed into the fresh candidate context, even when unchanged from the previous candidate.

Updates commit the parameter vector, typed site caches, device context and validated circuit together. A failed parameter evaluation, tight budget or invalid model/device context leaves prior committed values unchanged. Structural and unsupported changes continue through the validated #112 rebuild path; no device physics, topology model or C-reference tolerances are changed.

## API

- CircuitParameterPlan::inputs() returns ordered independent candidate coordinates.
- CircuitParameterPlan::sites() returns stable typed numeric site metadata (kind, location, rounded baseline and independent input indexes).
- CircuitCandidate::site_value(index) returns each committed typed site value.
- CircuitCandidate::last_site_evaluations() and last_site_roundings() report work in the most recent successful candidate update.
- CandidateParameters::evaluate_bound() evaluates against the shared slot map and the normal scalar evaluator, without per-candidate lexical traversal.

## Correctness and performance checks

Run with the pinned campaign nightly and managed sccache. The regression suite is tests/optimization_value_sites.rs, tests/optimization_parameters.rs and tests/optimization_parameter_plan.rs. It covers source categories, nested shadowing/function capture, direct-versus-transitive dirty suppression, raw-equal formatting suppression, candidate rollback, A/B/A consistency and independent parallel sessions.

For a representative 128-site/10,000-candidate batch, run:
    cargo +nightly test --locked --test optimization_value_sites benchmark_ten_thousand_candidates_and_one_hundred_twenty_eight_sites -- --ignored --nocapture

The test reports elapsed wall time, actual site evaluations and decimal round-trips versus 1,280,000 naive site visits. To inspect allocation counts without adding unsafe code or a global allocator to the project, run that same focused executable under Valgrind DHAT using --tool=dhat and inspect total allocation events/bytes. DHAT under Valgrind should not be used for throughput timing; run the benchmark natively for that. The current implementation still clones candidate parameter DAG state and constructs validated analysis scope per update; those costs should be profiled independently before broadening the scope of this issue.

## Measured native candidate update

Acer, pinned nightly and managed sccache, 10,000 prepared independent updates against 128 source sites: 74.383 ms wall time, 10,000 actual site evaluations and decimal roundings versus 1,280,000 naive per-site visits (128-fold reduction in site evaluation count). This measurement is for prepared candidate updates, not SPICE simulation or the entire optimization algorithm. Host contention and debug build settings affect timing; use DHAT separately for allocation counts.
