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
coordinate bindings include them. The diode's intrinsic junction voltage
has a solver-owned trial slot separate from the conservative charge pairs.
Forward and avalanche limiting use the effective model/instance thermal
scales and existing Newton history. Scalar and tile evaluation receive the
same limited coordinates; the affine Newton source uses those coordinates,
not the original iterate. Linear or electrothermal kernels that do not
request a limiter keep their exact evaluation. Small-signal assembly always
uses the exact converged bias, and no limited load is accepted as converged.

## Verification

Integration tests exercise the analytic nonlinear and electrothermal
fixtures, charge Jacobians, accepted/rejected history, candidate overrides,
and the public VDMOS artifact with hidden series nodes, DC, AC and noise.
The live reference test independently compares three VDMOS operating-point
currents at different multipliers against ngspice 47.

The public ABI currently owns a tile's prepared lanes as a unit. Selective
lane replacement and native-family SIMD implementations are separate
performance improvements; the generic engine does not assume that grouped
evaluation alone improves throughput.

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

The public producer follow-ups for selective lane replacement and optional
family-specific trial limiting are
[preprocess#38](https://github.com/codeandsolder/ngspice-rs-preprocess/issues/38)
and [preprocess#39](https://github.com/codeandsolder/ngspice-rs-preprocess/issues/39).
The source-selected diode now supplies an engine-side junction policy. Other
prepared model families continue to use generic Newton policy until a
family-specific policy is available.
