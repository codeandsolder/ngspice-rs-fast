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

- `Descriptor`: ABI version (`ABI_VERSION`), family name, 128-bit content
  fingerprint, terminal count (including explicitly modeled hidden nodes),
  ordered named parameter slots with finite defaults and flags for
  topology-changing setters, and separate charge and noise capabilities.
- `Capability::Computed` requires a complete physical equation implementation;
  `Capability::Zero` is a **declared physical zero**, never an implicit
  approximation; `Capability::Unavailable` means the corresponding analysis
  must fail explicitly. Every model supplies terminal current and analytic
  conductance; these are never optional.
- `Artifact<K>` validates the description once. `Kernel::prepare` derives
  immutable effective constants from typed parameter values and temperature
  **outside** Newton. `PreparedDevice<K>` owns those constants and its reusable
  evaluation buffers. `with_parameter(slot,value)` or
  `with_parameter_named(name,value)` constructs a different prepared instance
  for numeric candidate perturbations; structural changes fail instead of
  silently reusing stale rows.
- A different temperature also requires a new prepared instance. Current
  preparation deliberately refuses stale thermal contexts rather than
  pretending effective values are unchanged.
- There is no SPICE deck or protected-library parsing in this API and no
  special-case dispatch keyed by vendor model names.

## Equation and state contract

For each terminal `i`, an evaluation returns the *outgoing* current
`I[i]`, current Jacobian `G[i,j] = ∂I[i]/∂V[j]`, stored terminal charge
`Q[i]`, and charge Jacobian `C[i,j] = ∂Q[i]/∂V[j]`, all in the declared
terminal order. Multi-terminal and hidden-terminal models have the same
shape. All values must be finite, arrays complete, and the sum of terminal
currents, charges and every Jacobian column must vanish to a rounding-scale
tolerance (KCL and charge conservation).

A DC/Newton device stamp is:

```text
A[i,j] += G[i,j]
b[i]   += -I[i] + sum_j G[i,j] * V[j]
```

Ground rows are suppressed by the engine. AC at a converged bias adds
`G` to the static pencil `A` and `C` to the dynamic pencil `E`, giving
`A + jωE`. For a transient trial the engine integrates each `Q[i]` with
its chosen trapezoidal/Gear history and stamps the companion current and
`ag0 * C[i,j]`; derivative and charge state slots belong to engine
`DeviceState`, not to mutable model objects. Dropping a trial cannot
commit model history; `accept_point` remains the only publication path.
Unsupported charge equations reject AC/transient loads. Error propagation
does not replace model equations with weak approximations.

A computed-noise family supplies explicitly labeled `NoiseSource`
generators through `Kernel::evaluate_noise` at the solved bias point.
The existing analysis owns adjoint gains and frequency integration.
Native models use `NoiseFamily::Prepared`, after C's registered families
in noise traversal. A physically noiseless model may declare `Zero`;
`Unavailable` does not silently masquerade as noiseless.

## Homogeneous borrowed tiles

`BatchInput` borrows a contiguous slice of already-prepared instances and
the lane-major voltages. `BatchOutput` owns reusable row-major current,
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

This fixture is an independently implemented **analytic stand-in**, not
a compiled spice-redox artifact. The acceptance milestone for the linked
upstream #183 is an actual artifact produced by spice-redox with its
provenance, source-law verification, noise and dynamic-state policy. Once
that compiler produces a native crate/artifact, adapt the shared ABI at
its producer boundary rather than adding a parallel engine-side parser
or compatibility shim. Existing in-tree models stay in production until
equivalence and native-kernel throughput have both been established.
