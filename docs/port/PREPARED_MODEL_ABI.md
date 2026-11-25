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
Until the latter is implemented, the engine uses its generic Newton
policy for prepared kernels, not family-specific limiters.
