# #140 — Reusable numeric netlist value sites

## Decision

Retain the single immutable CircuitParameterPlan from #112 and the shared ParameterPlan from #117. The sites retain typed baseline values, original SiteKind, SourceLoc provenance and transitive independent coordinate indexes. No second evaluator or netlist reconstruction path is introduced.

The binder uses the existing bounded RefScan to capture direct lexical names, including free variables through nested .func calls, and resolves them to stable defining slots once. CandidateParameters::evaluate_bound uses the original evaluate_root evaluator, so error locations, arithmetic domain, finite-value and operation-budget semantics remain unchanged.

Each independent candidate session owns committed raw and numparam-decimal-rounded site values, plus a reusable vector of staged changes. A site is evaluated only when its directly referenced resolved slot actually changes, rather than when any transitive input changes. This avoids work when derived parameter values cancel. Decimal rounding is skipped if the raw f64 result is bit-identical. Changed numeric-only sites and conditioned gradient samples use the same allocation-free numeric conversion (#300): the canonical 16-significant-digit scientific representation is formatted into a bounded stack buffer and parsed directly as binary64, without heap-backed source serialization or SPICE suffix parsing. Actual text boundaries retain the normal canonical literal formatter. This preserves the numparam rounding boundary, including signed zero and finite-input overflow. Every effective non-baseline numeric override is still replayed into the fresh candidate context, even when unchanged from the previous candidate.

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

## Numeric-only rounding (#300)

The numeric rounding regression `numparam_numeric_rounding_matches_literal_and_c_oracle` checks 100,000 deterministic binary64 patterns (99,949 finite), adjacent values around decimal powers, exact halfway cases in both directions, signed zero, and integer/overflow boundaries. The expected digest is independently computed by C `snprintf("%.15e", x)` followed by `strtod`. The normal source formatter remains available for source boundaries; the numeric update and gradient paths no longer construct heap-backed literals.

Compare the same base revision with and without the numeric rounding change, with identical benchmark tests, pinned nightly, managed cache, and isolated release builds:

```sh
cargo +nightly test --release --locked --test optimization_value_sites benchmark_changed_numeric_sites -- --exact --ignored --nocapture
cargo +nightly test --release --locked --test optimization_derivatives benchmark_repeated_candidate_gradients -- --exact --ignored --nocapture
```

Run the resulting test executables directly under `perf stat -e instructions:u,cycles:u,task-clock` and separately under `valgrind --tool=dhat`. Three interleaved release runs per version on the same host yielded the following medians:

| Workload | Baseline instructions | Numeric-only instructions | Baseline user CPU | Numeric-only user CPU |
|---|---:|---:|---:|---:|
| 111,000 updates with 32 changed sites (3,552,000 conversions) | 24,205,225,154 | 14,911,651,417 | 3.622 s | 2.379 s |
| 1,000 repeated candidate gradient evaluations | 1,386,272,328 | 1,279,621,381 | 0.286 s | 0.261 s |

The scalar update workload executes 38.4% fewer user-mode instructions and the gradient workload 7.7% fewer. Elapsed wall time was substantially affected by concurrent host load; the instruction counts were stable. The conversion-only DHAT workload (one million numeric conversions) eliminated five million allocation events and 110,000,000 allocated bytes, leaving 13 common process/harness allocations. Both variants produced the same numeric digest. Sampling profiles show that heap-backed formatting disappears from the hot rounding path; exact float formatting, candidate evaluation and device replacement remain visible. These figures describe candidate operations, not the complete solver or optimization algorithm.

The complete 111,000-update scalar benchmark under DHAT allocated 28,424,327 blocks / 1,571,306,743 bytes before and 10,664,327 blocks / 1,180,586,744 bytes after: 17,760,000 allocation events (exactly five per changed site) and 390,719,999 cumulative bytes eliminated. Both benchmark runs passed.
