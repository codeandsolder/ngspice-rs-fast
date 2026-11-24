# Native prepared-model kernel contract

**Owner:** ngspice-rs-fast circuit engine, [architecture issue #189](https://github.com/codeandsolder/ngspice-rs-fast/issues/189).
**Producer:** spice-redox model compiler, [artifact ABI issue #183](https://github.com/codeandsolder/spice-redox/issues/183).

This document describes the native Rust engine-side contract currently in
`src/devices/prepared_model.rs` and `batch.rs`. The model compiler owns
translation, parameter semantics, model-specific current, charge/flux and
noise equations. The circuit engine owns node and branch numbering, topology,
matrix commit, Newton and sparse factorization, transient integration,
accepted/rejected history, analysis orchestration and adjoint noise transfer.

## Model and instance preparation

- Descriptor: native ABI v2, family and 128-bit content fingerprint,
  unique lowercase typed equation slots (electrical/thermal/internal),
  resolved numeric candidate parameter slots, structural invalidation and
  explicit dynamic-storage and noise capabilities. The compiler owns
  source model semantics; the engine binds equation rows once before Newton.
- `Capability::Computed` requires a complete physical equation implementation;
  `Capability::Zero` is a **declared physical zero**, never an implicit
  approximation; `Capability::Unavailable` means the corresponding analysis
  must fail explicitly. Every model supplies a complete generalized static residual and analytic
  Jacobian; these are never optional.
- `Artifact<K>` validates the description once. `Kernel::prepare` derives
  immutable effective constants from typed parameter values and temperature
  **outside** Newton. `PreparedDevice<K>` owns those constants and its reusable
  evaluation buffers. `with_parameter(slot,value)` or
  `with_parameter_named(name,value)` constructs a different prepared instance
  for numeric candidate perturbations; structural changes fail instead of
  silently reusing stale rows.
- The engine passes ambient Celsius to the artifact instance binder (the
  same convention as ModelContext). This boundary converts to absolute
  Kelvin before Kernel::prepare; compiled model equations never receive
  a silently unit-ambiguous temperature. A different ambient temperature
  requires a new prepared instance; a physically modeled self-heating
  coordinate remains an explicit dynamic equation and is not that ambient
  preparation temperature. Stale context use is rejected before solving.
- There is no SPICE deck or protected-library parsing in this API and no
  special-case dispatch keyed by vendor model names.

## Equation and state contract (ABI v2)

The native kernel supplies a generalized DAE: F(x) + dQ(x)/dt = 0.
Electrical equation rows represent current in amperes and stored charge
in coulombs; thermal rows represent watts and joules; internal rows
represent their declared relaxation or dynamic-state equation.
Electrical Kirchhoff current, charge, and all their Jacobian column sums
conserve across electrical rows only. Every physical equation is invariant
under a common-mode shift of electrical node potentials. Both conditions
are checked without heap allocation during Newton evaluation.

DC stamps G=dF/dx and affine RHS -F+G*x; AC stamps G+jw*C
where C=dQ/dx; transient uses the engine owned accepted/trial histories
to integrate Q and stamp its companion derivatives. No thermal power is
misinterpreted as electrical charge. The independently compiled VDMOS
model uses thermal and recovery state equations and requires this layout.

## Homogeneous borrowed tiles

`BatchInput` borrows a contiguous slice of already-prepared instances and
the lane-major generalized physical coordinates. `BatchOutput` owns reusable row-major current,
conductance, charge and capacitance buffers. `Kernel::evaluate_batch`
defaults to independent scalar lanes and may be overridden by compiled
vectorized/SIMD family kernels without changing the output ABI. Every lane
is validated before consumption; invalid or short outputs fail the batch.

A scalar engine device currently has its own reusable evaluation buffers;
there is no heap allocation, string dispatch, parameter lookup or source
translation in its Newton stamping path. The static homogeneous batched
entrypoint is available for the grouped device evaluator; grouping across
separate `Device` entries and final native producer crate linkage are separate
integration steps and must not be counted as completed by this ABI alone.

## Validation and reproduction

The regression `tests/prepared_model_integration.rs` supplies a physically
explicit analytic polynomial two-terminal model with

```text
I = g*V + a*V^3                 G = g + 3*a*V^2
Q = c*V + b*V^3                 C = c + 3*b*V^2
```

and the equal/opposite terminal contributions. It verifies exact term
conservation and Jacobian shapes, version/slot validation, independent
candidate preparation and parameter failures, DC Newton and full operating
point, AC capacitance and the full AC driver, charge history acceptance and
rejection including the full transient driver, and thermal noise generators
and the full noise-spectrum driver.

```sh
cargo +nightly test --locked --test prepared_model_integration
```

An additional analytic four-row electrothermal/recovery DAE exercises
physical cross-derivatives, heat/relaxation balance, DC sparse stamping,
AC and transient integration of heat and recovery storage, atomic trial
rollback/acceptance, electrical-only conservation and gauge invariance,
and invalid structural schemas. It proves representability, not actual VDMOS linkage.
The independent VDMOS solver branch owns that model-specific integration:
https://github.com/codeandsolder/ngspice-rs-fast/pull/191 .

This fixture is an independently implemented **analytic stand-in**, not
a compiled spice-redox artifact. The acceptance milestone for the linked
upstream #183 is an actual artifact produced by spice-redox with its
provenance, source-law verification, noise and dynamic-state policy. Once
that compiler produces a native crate/artifact, adapt the shared ABI at
its producer boundary rather than adding a parallel engine-side parser
or compatibility shim. Existing in-tree models stay in production until
equivalence and native-kernel throughput have both been established.
