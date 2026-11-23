# Compiled reusable numeric source sites (#140)

## Architecture and scope

This extends the accepted #112 `CircuitParameterPlan` / `CircuitCandidate`
front-end and #117 lexical `ParameterPlan`; it does **not** replace them.
All preparation starts with the fully validated scalar baseline. The original
netlist AST remains immutable and shared. Topology, model validation and
device state remain the responsibility of the existing circuit binder and
ordinary scalar re-elaboration fallback.

The parameter plan's existing `RefScan` binds each source expression's
free parameter names to their defining stable numeric slots, including
`.func` free variables, lexical shadowing and transitive `.param` aliases.
`CandidateParameters::evaluate_bound` invokes the existing
source-expression evaluator and evaluation budget against these slots,
rather than performing lexical scope lookup for each identifier.

The site plan contains only source expressions depending on selected
independent inputs. Each immutable `CandidateSite` retains the original
`SiteKind` (device/model, analysis argument, `.ic`, `.nodeset`), source
location, candidate input indexes and the fully numparam-rounded baseline
numeric value. `CircuitParameterPlan::sites()` exposes typed records in
the scalar elaborator's category/index order; original source indexes
are authoritative, not a second synthetic deck ordering.

## Candidate update and atomicity

The candidate session owns a typed read-back value for each prepared site.
The dependency test compares directly referenced parameter slots against
the previous successfully committed candidate. It does not merely compare
selected input coordinates against the initial baseline. Thus an
independent coordinate can change while an intermediate `.param` expression
such as `flat={x-x}` preserves its value: downstream sites are not
reevaluated. Unrelated changes never touch independent sites.

Changed sites are evaluated through the original function/error/budget
engine. Their raw evaluated bit pattern is compared against the last
committed raw value; a recomputed result such as `x-x` can be unchanged
even when its input slot changed. Only a new raw result undergoes the
ordinary numparam decimal format and read-back. Unchanged sites reuse
their previous binary64 values without formatting, reparsing or
materializing a replacement deck. The complete
effective set of non-baseline numeric setters is still installed in each
candidate, including unchanged values inherited from the last candidate:
the cache is not a delta-only model context.

Evaluation, read-back and device model checks are staged until every
dependent value and the resulting circuit shape pass validation. Only
then are numeric slots, site values, run config and circuit published.
An invalid expression, non-finite input, budget exhaustion or device
validation failure leaves the previous candidate unchanged.
Rebuilt candidates still use the ordinary scalar elaborator, preserving
nested subcircuit/formal/model/bin geometry and source ordering; old
topology handles cannot be reused after a real rebuild.

`CircuitCandidate::site_value(index)` returns the currently committed typed
numeric value; `last_site_evaluations()` and `last_site_roundings()` report
the exact number of site evaluations and decimal read-back operations
in the last successful candidate transition (zero for an identical
vector). These inspection surfaces are not a second optimizer.

## Verification and measurement

`tests/optimization_value_sites.rs` covers compiled versus ordinary
expression evaluation for nested scopes with function free names and
shadowing, A/B/A candidates, direct-reference dirty suppression,
device/model/analysis/initial-condition/nodeset provenance, full scalar
matrix equivalence, invalid candidate/budget rollback, and parallel
independent candidate sessions.

The existing #112 and #117 test targets must continue passing: they
assert typed setter and topology/fallback equivalence against
independently elaborated scalar circuits. No C golden tolerances, model
algorithms or analysis runners are modified.

A manually invoked, CI-ignored benchmark compares 10,000 prepared
candidate updates over 128 dependent numeric sites against 500 separately
parsed and validated scalar candidates. It reports preparation time,
actual site evaluations, topology reuse/rebuild counts, and scalar
baseline time. Time includes candidate validation but not a SPICE
simulation. The source avoids unsafe allocation hooks: real heap
allocation counts may be measured externally with Valgrind on the
benchmark test binary, not guessed from logical dirty-site counts.

## Ownership and subsequent work

- #117 owns lexical slots, parameter dependency scheduling, `.func`
  evaluation and the candidate's transactional numeric graph.
- #112 owns device/model/topology binding, context overlays, validation
  and scalar re-elaboration fallback.
- #140 owns only pre-bound numeric source site lookup, per-session typed
  site caching, dirty-site work measurement, and API metadata.
