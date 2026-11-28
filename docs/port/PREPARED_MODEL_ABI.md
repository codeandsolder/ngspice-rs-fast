# Prepared compact models

The engine consumes the public `model-kernel-api` contract directly. The
dependency is pinned to an accepted producer revision, rather than duplicating
its schema or defining a second kernel trait. The engine MSRV follows the
producer (1.99). Publishing to crates.io requires publishing the shared
contract first; the current Git dependency is suitable for repository builds.

## Engine responsibilities

- `PreparedDevice<K>` binds descriptor terminals and hidden equations to
  circuit nodes, with transactional allocation and structural validation.
- Numeric setters bind to typed descriptor ordinals; topology-changing
  parameters cannot be applied as numeric overrides.
- Circuit temperature is a Celsius analysis context; kernel temperatures
  are absolute Kelvin. Thermal MNA coordinates are rises above circuit
  ambient, independently of the model's effective instance temperature.
- The engine stamps analytic current Jacobians and integrates charge/flux
  using its existing accepted/trial history, retaining the affine AC pencil.
- Noise generators carry arbitrary coherent complex row injections and
  spectral density. Frequency-dependent noise refresh does not force
  reassembly of frequency-independent conductance or charge matrices.
- Exact live artifact identity groups instances into bounded tiles of at
  most 64 lanes, using the producer's structured output buffers. Scatter
  order remains the original circuit load order. Singletons use the scalar
  kernel. Workspaces are local to a prepared circuit run.

## Source-selected models

The native grammar and writer share the model-family registry. Explicit
`simetrix_diode_level1` declarations use the producer's ordered scalar model
compiler and the existing lexical model resolver. A plain native `D` model
still selects the native diode. Source expressions are resolved before
compilation; immutable declaration artifacts are cached by series-node and
instance temperature-policy shape. Numeric instance geometry, multiplicity,
temperature and gmin are prepared at a context boundary, never during a
Newton evaluation. A candidate that changes the descriptor's topology must
re-elaborate instead of reusing numerical rows or accepted state.

Public terminal observations exclude hidden equations, while numerical
coordinate bindings include them. ABI 4 independently declares ownership of
cold-start initialization, limiting, and additional static convergence.
The engine stores actual evaluated local coordinates and exactness in
solver-owned trial slots separate from conservative charge pairs. The
previous Newton load and accepted predictor reference are distinct, and
discarded trials cannot mutate accepted history. Prepared diode families
supply their own junction policy via the artifact; the engine does not
duplicate model-specific junction equations or limiting scales.

Scalar and tile evaluation receive the same local coordinates. The affine
Newton source uses the actual evaluated coordinates, including the thermal
rise obtained by subtracting circuit ambient from the returned absolute
Kelvin coordinate. Integrated charge and its Jacobian use the same local
point. Explicit initialization constraints, exact loads and small-signal
assembly do not invoke limiting. An initialized or limited load marks the
Newton pass nonconvergent; exact candidate evaluation and the declared
family criterion supplement the global convergence check. Electrical rows
use the engine current absolute tolerance; other row kinds use relative
comparison until dimensioned absolute tolerances are available.

## Verification

Integration tests exercise the analytic nonlinear and electrothermal
fixtures, charge Jacobians, accepted/rejected history, candidate overrides,
and the public VDMOS artifact with hidden series nodes, DC, AC and noise.
The live reference test independently compares three VDMOS operating-point
currents at different multipliers against ngspice 47.

## Preparation ownership

The producer's typed tiles own the contiguous prepared data of grouped
instances. Initial standalone preparations move into the tile without a
second physical preparation. Scalar, observation, AC, noise and family
trial evaluation read the same typed lane, while genuine singleton
instances own standalone physical data.

A prepared run holds an exclusive lease on its candidate preparation.
Sequential analysis scopes reuse one idle preparation; concurrent live
runs never share mutable physical tiles. Candidate validation and
publication use the same staged physical plan. Numeric setters are
combined per device before preparing its final recipe.

Every affected tile stages only changed physical lanes. All tile
transactions, singleton preparations and row-offset updates must be
validated before committing the new candidate; a rejected candidate
does not alter the previous prepared lane data or accepted solver state.
Exact artifact identity or topology changes rebuild membership and
coordinate bindings. A temperature change with constant effective
ambient updates thermal row offsets without repeating physical
preparation. Frequency-only and source-only transitions preserve
unchanged physical data.

Unchanged Newton loads reuse their prepared recipes, physical data,
coordinate-to-row maps and output workspaces. Domain errors during
evaluation become numerical trial failures rather than structural circuit
errors.

On Acer (Ryzen 7 4700U, CPU 5), the interleaved release benchmark over 128
VDMOS instances, prebound reusable MNA matrices and 11 measured 300-load
rounds measured median 351.9 us/tiled load and 368.4 us/scalar load.
The complete bound matrix and RHS were bit-identical. This is a host-local
measurement, not a performance assertion or evidence of SIMD acceleration.

The accepted public producer contracts are
[preprocess#38](https://github.com/codeandsolder/ngspice-rs-preprocess/issues/38)
and [preprocess#39](https://github.com/codeandsolder/ngspice-rs-preprocess/issues/39).
The solver consumes ABI 4 trial capabilities directly, including the
prepared SIMetrix diode. Families declaring engine-owned numerical policy
retain exact prepared loads without extra per-instance trial storage.

### Independent VDMOS whole-circuit reference suite (#259)

`tests/vdmos_conformance.rs` reuses the 31 standalone native ngspice 47
reference decks in `tests/fixtures/vdmos`. The test boundary consumes the
producer's evaluated `ModelLibrary::instantiate` topology: model variants,
instance multipliers and ambient laws, structural heating selection, and
ordered descriptor terminals. It removes the producer-owned single-line
cards by source positions before parsing the remaining flat fixture with the
ordinary electrical parser. This is a fixture adapter, not production MOS
source dispatch. Shared artifacts bind through `PreparedDevice` and the
standard analysis plans. Thermal voltages are projected from rise above
circuit ambient to absolute Celsius only for DC operating points and
transient comparisons, not AC perturbations. Noise generator identities
`channel` and `flicker` correspond to C's `id` and `1overf`.

The independent C47 measurement with ±1 mV gate perturbations at the
`thermal_dc_n/p` biases and 1 mHz unit AC gate excitation gives:

| Circuit | Quantity | Central DC derivative | AC real derivative |
| --- | --- | ---: | ---: |
| N | Junction K/V | +9.918498 | -10.802951 |
| N | Case K/V | +8.265415 | -9.002460 |
| N | Supply A/V | -5.510277 | -6.001640 |
| P | Junction K/V | -5.166647 | +5.400206 |
| P | Case K/V | -4.305539 | +4.500171 |
| P | Supply A/V | -2.870359 | -3.000114 |

C's thermal AC response reverses the sign of its own static derivative and
has inconsistent drain-current feedback. The physical low-frequency
consistency test compares the engine to its own central DC derivative and
independent C DC slope, rather than copying the inconsistent C AC signs.
At higher frequencies the thermal phase must lag the DC response. No
model-specific solver branch or relaxed physical tolerance is required by
this coordinate convention.
