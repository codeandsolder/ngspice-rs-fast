# Touchstone sampled N-port networks (#174)

The codec in `src/analysis/touchstone.rs` and the typed physical network in
`src/devices/nport.rs` are separate from SPICE rawfiles. The `Touchstone`
model retains frequency samples in physical hertz, each full complex N×N
matrix in response-row/stimulus-column order, real-positive reference
impedances, optional mixed-mode basis and independent two-port noise data.
Source and destination text use the IBIS Touchstone 2.1 specification
(https://www.ibis.org/touchstone_ver2.1/touchstone_ver2_1.pdf), including the
legacy v1 `.sNp` extension, and v2 `[Version]` and `[Number of Ports]` headers.

## File representation and validation

`Touchstone::parse(text, path, limits)` rejects malformed/non-ASCII source,
nonfinite values, nonmonotone grids, invalid reference impedances, incomplete
records, inconsistent matrix triangularity, malformed mixed-mode order, and
excessive source/port/point/numeric-cell budgets. The parser does not retain
tokens or the complete source; it uses a bounded single-record buffer and
retained numeric matrices. Number of matrix samples, both noise and network,
are explicit resource limits. `Touchstone::write(version, encoding)` performs
validation again and does not invent a finite decibel representation for an
exactly zero complex value.

- V1 two-port data are stored **S11, S21, S12, S22** on disk; internal
  matrices are row-major. V2 `[Two-Port Data Order]` chooses `21_12` or `12_21`.
- Higher-port matrices are row-major; v2 `Lower` or `Upper` triangular inputs
  are expanded symmetrically. RI, MA (degrees), DB (20 log10 magnitude) and
  Hz/kHz/MHz/GHz are accepted. No phase-angle interpolation is performed.
- In v1.x, Y/Z/H/G matrix coefficients and two-port noise resistance are
  normalized by the option-line port reference impedances. For heterogeneous
  R1/R2, physical H (row-major) multipliers are
  `R1, sqrt(R1/R2), sqrt(R1/R2), 1/R2`; physical G multipliers are
  `1/R1, sqrt(R2/R1), sqrt(R2/R1), R2`. In v2.x Y/Z/H/G
  network matrices and effective noise resistance have physical units.
- `[Reference]` is an explicit per-port positive-real impedance list.
  `[Mixed-Mode Order]` preserves differential/common/single-ended axis
  identity; matching D/C pairs must refer to the same ordered physical ports.
  Mixed mode cannot be substituted into ordinary nodal voltage equations
  without the correct basis transformation.
- `Touchstone::from_sp_plot` extracts a complete S, Y, or Z matrix from
  `SP Analysis` columns `S_i_j`, `Y_i_j`, or `Z_i_j` without changing the
  original plot. The provided port-order reference list must agree with
  C's first-port `v(Rbase)` column if that column is present. It creates
  a version-2.1 typed network for subsequent export.

## Measured physical N-port

`MeasuredNPort::new(name, Arc<Touchstone>, terminals, extrapolation)` validates
terminal attachments and holds one immutable model shared across frequencies.
Frequency selection uses binary search; intermediate samples are linearly
interpolated in rectangular complex coefficients. `Extrapolation::Reject` or
explicit `Hold` is required, not implicit extrapolation, and interpolated
admittance is checked for finiteness and singular constraints. There is no
assumption that all imported active networks are passive or reciprocal.

For real-positive power-wave reference impedances, with
`D = diag(sqrt(R_i))`, nodal admittance in physical siemens is

`Y = D^-1 (I - S) (I + S)^-1 D^-1`.

For physical Z data, `Y = Z^-1`; physical Y is used directly. H/G two-port
hybrid data are converted by their defining terminal-current equations.
Singular matrices such as an ideal zero-impedance short or lossless ideal
through are **not** replaced with zero-admittance (which would silently turn
a short into an open): the sampled nodal model rejects them until a branch
constraint representation is explicitly implemented.

Mixed-mode S uses the orthonormal power-wave basis; Y/Z use their appropriate
voltage/current dual basis. `nodal_stamps_at` expands `I = Y V` over each
positive and negative terminal as four signed KCL contributions for every
Y_ij, omitting the ground row. A `MeasuredNPort` implements the simulator's
`Device` trait for frequency-dependent AC small-signal equations and the
same physical DC bias equation when a real-valued zero-frequency sample is
available under the explicit extrapolation policy. The DC path rejects
imaginary bias admittance. The simulator may not treat an arbitrary sampled
frequency response as a stable causal transient macromodel: transient
stamping rejects it explicitly.

## Optional physical audit

`MeasuredNPort::properties_at(f)` reports separate `Verified`, `Violated`, or
`Undetermined` verdicts for reciprocity and passivity, rather than assuming
measured networks are passive or reciprocal. Reciprocity is compared in the
single-ended physical Y basis with a finite backward-error envelope; a tiny
nonzero mismatch is *undetermined*, not silently discarded. Passivity uses the
smallest eigenvalue of the 2N×2N real-symmetric embedding of the Hermitian
conductance `(Y + Yᴴ)/2`, which is equivalent to checking the sign of supplied
real power for every port-voltage phasor. The existing bounded Jacobi solver
and an explicit numerical error band distinguish proven passive, active, and
indeterminate results. A purely reactive network has zero dissipative part and
is passive.

`MeasuredNPort::require_properties_at(f, PhysicalRequirements)` selectively
enforces these findings, rejecting both known violations and uncertain cases.
The check is per frequency and does not promise a global continuous-frequency
result. Causality always reports *undetermined* for an arbitrary sampled file;
a strict causality requirement fails until an independently validated stable
rational macromodel exists. No positivity/reciprocity assumptions are silently
inserted into AC computations.

## Testing and status

`tests/touchstone.rs`: independent v1/v2 row-order/metadata/triangular/mixed-
mode/noise and normalization roundtrips, malformed and budget cases.
`tests/nport_device.rs`: analytic resistor/open/short/ideal through,
heterogeneous 50/75 ohm normalization, physical mixed-mode Y/S transforms,
complex interpolation and signed nodal stamp conservation. These tests
establish the intended direct-network semantics. The architecture campaign
serial validator performs full Rust strict lint and pinned golden verification
before integration. See the separate Notion issue log for acceptance dates
and unresolved engineering questions.

The export path also supports `Touchstone::write_with_limits`, which uses a
bounded formatting sink and rejects a byte-cap breach during streaming
formatting, not after constructing an oversized output string. Every formatted
numeric field, matrix row, metadata line, and noise record shares this cap.
