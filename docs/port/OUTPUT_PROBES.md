# Shared transfer/sensitivity output probes (#170)

## Current contract

`.tf` and `.sens` accept the same output prefix, `v(n)`,
`v(n,m)` or `i(source)`, with case-folded names and the circuit's
configured automatic `gnd` alias. The common analysis-internal
`output_probe` module parses only this prefix and leaves the input
source, filter tokens and sweep grammar to the containing driver.
The parser retains card-specific errors and stops exactly before the
first post-output token.

After `Circuit::finalize`, `bind` converts the expression into
a copyable two-row voltage difference or one branch-current row,
plus canonical output node labels and, for a current probe, the device
ordinal. It validates an existing node (a missing row is valid **only**
for ground), an existing device with a `CKTfndBranch`-admissible
current, and its actual branch row. Both drivers use the identical
lookup and return errors with their original C reference contexts:
`TFanal in tfanal.c` or `sens_sens in cktsens.c`.

## Deliberate separation

- `.tf` alone resolves its independent V/I input source and performs
  the two resistance excitations. It retains the branch-current sign
  and C's same-input/output-source shortcut. Its one resolved input
  ordinal is also reused for the shortcut, avoiding a second device
  search.
- `.sens` alone owns parameter selection, records, adjoint projection,
  complex transpose operations, tolerance/roundoff estimates and
  finite-difference policy. The shared `Probe` is the exact linear
  observation functional; there is no conjugation.
- Neither driver uses the device-parameter observation framework in
  `devices::observe`: its semantics and index space differ from a
  circuit-row output probe.
- Syntax is checked once per analysis card. Binding is completed once
  per finalized circuit; repeated numerical evaluation uses only the
  fixed row indices. No additional runtime row scans or allocations
  occur in projection.

## Regression coverage

`analysis::output_probe::tests` validates same rows across the two
card contexts, default/explicit/auto-ground and mixed case, distinct
device enumeration orders, missing nodes, established pre-finalization
row maps, non-findable device currents,
canonical names, exact errors and parser cursor boundaries.

`tests/tf_sens_observations.rs` compares production physical
transfer/sensitivity outputs for reordered dividers, positive/
negative/differential voltage probes, current sign, same-source
resistance and card-specific forbidden forms.

The original `tests/analysis_tf.rs`, `tests/sensitivity_analysis.rs`,
`tests/c_tf_reference.rs` and `tests/c_sens_reference.rs` remain the
cross-check for established C naming, nonlinear model behavior and
actual C-oracle numerical fidelity.
