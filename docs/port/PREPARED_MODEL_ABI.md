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

Legacy Berkeley BSIM1 and BSIM2 are selected by model levels 4 and 5. Their
separate typed constitutive models, series-node resolution and numerical trial
policies come from the shared producer; source OFF and initial differential
voltages are applied at the circuit boundary.

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

The public producer now supports transactional selective lane replacement
and lane-local scalar evaluation. Solver ownership of scalar/tile
preparation and selective lane replacement is addressed separately in
[#250](https://github.com/codeandsolder/ngspice-rs-fast/issues/250).

Prepared runs keep immutable tile recipes across candidate transitions.
Frequency-only and source-only transitions with unchanged recipes reuse
all tiles; numeric changes replace only affected tiles, after all
replacement preparations succeed. Stable coordinate-to-row mappings are
bound during tile preparation. Domain errors during evaluation become
numerical trial failures rather than structural circuit errors.

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
