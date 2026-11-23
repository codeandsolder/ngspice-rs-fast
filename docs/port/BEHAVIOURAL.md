# Behavioural source runtime — compiled gradients and per-device scratch

## Execution contract

A behavioural source owns an immutable compiled Program and mutable per-device
evaluation scratch. The program holds an indexed instruction plan and symbolic
derivative roots in first-seen expression-variable order. Value and derivative
instructions share the same numerical kernels. The plan evaluates every
reachable derivative root for each Newton stamp, AC/PZ Jacobian and convergence
check, including when only the value is ultimately consumed. Lazy ternary
execution never visits inactive branches; error diagnostics keep their source
and derivative evaluation order.

The public Program::evaluate returns an owned Evaluation to independent callers.
Internal prepared-device evaluation returns EvaluationView borrowing the
derivative slice from the caller's Workspace. The workspace holds reusable
instruction values, demand stack, and derivative vector. Without ternaries,
evaluation uses topologically ordered instructions directly; ternaries use the
bounded demand evaluator with finite-result caching.

Behavioural::with_linearization holds one short-lived mutable scratch borrow
through the entire evaluation and its stamping consumer. Per-device scratch:
- circuit-variable values in expression-variable order;
- matrix columns (None for ground; branch controls in reference order);
- program instruction values, pending stack and derivative vector.

These buffers keep capacity across repeated loads. Stamps borrow input values,
columns and gradient only within a closure, so another evaluation cannot reuse
the mutable workspace while a current stamp reads its result. Device cloning
starts empty scratch, never copying a candidate's prior partial evaluation.
The compiled Program and immutable literal tables remain shared.

## Matrix bindings and topology identity

Compiled Variable descriptors already resolve node/branch semantics. Actual
MNA rows are resolved into the reused column vector for each evaluation. The
current MnaUnknowns interface does not expose a topology-generation token to
StampContext, ConvergenceContext or LinearContext. Finalizing a circuit can
rebuild unknown numbering. Caching columns by borrowed object address, clone,
or total row count could stamp stale rows after topology changes. The current
solution resolves the columns without allocating after initial warmup, at cost
proportional to the number of expression variables. Any future persistent
row cache must receive a real, validated topology identity/invalidation API,
not assume pointer stability or the old source shape.

## Physical and numerical semantics

No changed physics or expression grammar: derivatives, domain checks,
nonfinite diagnostics, signed-zero behavior, PWL corner selection, branch
currents, ternary laziness and ordered matrix stamps remain C-compatible.
In particular convergence still evaluates and validates all derivatives
before checking ASRCprev_value; skipping them for a value-only consumer
could silently change error behavior.

A failed evaluation leaves only scratch cache state; the next invocation
clears its gradient and pending stack and overwrites each nonconditional
instruction in source order. Accepted device state is not modified on error.
Independent candidates and new prepared circuits retain separate scratch.

## Validation and benchmarking

- Raw symbolic-plan unit tests compare value, derivative and error-order bits
  against an independent tree evaluator, including conditional branches and
  tables, domain failures, depth limits, and recovery after error.
- devices_behavioural_sources, analysis_behavioural_sources and
  c_behavioural_reference exercise physical Newton residuals, AC Jacobians,
  branch controls, signed corner cases and live pinned C derivatives.
- tests/behavioural_workspace_bench.rs runs repeated prepared nonlinear
  loads and AC assemblies of a multi-B-source network. The ignored workload
  supports native CPU timings and Valgrind DHAT heap allocation attribution.
- Exact conformance against the 220-fixture golden corpus and full strict
  rust-skills2 gate are required before campaign publication.

Current test results, performance measurements and deployment details live
in the companion Notion implementation page; this page describes only
the current runtime contract.
