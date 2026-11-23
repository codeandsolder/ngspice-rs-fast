# Coupled multiconductor transmission lines (P/CPL, #183)

## Compatibility and present implementation boundary

Upstream ngspice's P (CPL/KSPICE) device accepts 1–8 coupled conductors,
with a near/far return terminal, and Maxwell per-unit-length symmetric
R, L, G and C matrices. Unlike the T and O devices, upstream CPLinfo
has no AC or pole-zero loading callback (DEVacLoad = DEVpzLoad = NULL).
It is therefore not correct to claim an upstream CPL AC/SP/PZ numeric
reference. Native DC and transient behavior is the parity target.

Current native substrate: src/devices/cpl/plan.rs and
src/devices/cpl/model.rs, with scoped vector literalization in
src/netlist/elaborate.rs. The internal src/devices/cpl/dc.rs device now
implements real MNA branch rows for C-compatible OP/DC and linear DC
assembly. The public P-device registry remains intentionally disabled
until the full lossy transient propagation kernel is ready; parsing a
model must not silently execute an incomplete or uncoupled substitute.

## Physical model and validated geometry

The physical telegrapher equations for the conductor-voltage and
current vectors relative to their local return references are

    dV/dx = -(R + s L) I
    dI/dx = -(G + s C) V

with all four matrices symmetric. A matrix coefficient is not
independently a line: the off-diagonal elements determine cross-talk
and mode conversion. Negative Maxwell off-diagonal capacitance is
physically meaningful.

CplPlan::prepare validates length > 0, conductor order 1..=8,
complete packed upper-triangular L/C, optional absent/diagonal/packed
R/G, finite coefficients, and passive R/G positive-semidefiniteness.
L/C must be positive definite with resolvable propagation modes. All
entries are retained, without clamping or stripping coupling. The
passivity check uses an existing symmetric Jacobi eigensolver on a
diagonally congruence-scaled matrix; the scale-free comparisons avoid
treating farads as if their raw numerical magnitudes were determinant
conditioning measures.

For the high-frequency wavefront of each mode, if C = S S^T is its
Cholesky factorization, the eigenvalues of S^T L S are those of the
generally nonsymmetric product LC. All are real and positive for
physical positive-definite L and C. The ordered wavefront times are

    tau_k = length * sqrt(lambda_k(LC))

and are derived from the FULL coupled matrices, including unequal,
noncommuting L and C. These delays do not substitute for the
frequency-dependent lossy R/G mode fitting or its transient
convolution; those still require the CPL device solver.

The shared bounded Jacobi implementation also exposes a sorted,
orthonormal eigenbasis via Matrix::symmetric_eigendecomposition.
It accumulates rotations only when requested, so existing
eigenvalue-only physical validations retain their cheaper path.
The immutable CplPlan now stores a single CplModes with mode delays,
power-reciprocal current/voltage bases, and the full positive-definite
characteristic admittance Y0. With C/c_scale = S S^T, the symmetric
basis Q diagonalizes S^T (L/l_scale) S. The physical bases are
B_current = S Q and B_voltage = S^-T Q; they satisfy
B_voltage^T B_current = identity, and the constructed characteristic
admittance satisfies Y0 L Y0 = C. This includes off-diagonal
matched-load coupling without assuming L and C commute.

The allocation-free modal port interface launches the sum of modal
voltage and modal current divided by modal admittance. At the other
port, causally delayed received modal launches determine current as
I = Y0 V - B_current diag(y_mode) received_launch. These identities
describe only the lossless wavefront. General frequency-dependent
R/G attenuation, convolution and accepted-step delay history remain
separate requirements, never scalar per-conductor approximations.
The read-only causal history sampler now evaluates each mode against
accepted WaveformHistory samples using bounded binary search and stable
affine interpolation. At the initial t=0 sample it supplies constant
prehistory; after pruning, an absent required sample is explicitly
rejected instead of being extrapolated. No trial mutates accepted
history or partially writes the received output buffers. Full lossy
Padé and CPL transient callback integration remain pending.

## Lossy frequency-domain and rational-kernel substrate

CplSpectralSample evaluates the full coupled telegrapher response at
positive real Laplace rate s by factoring the scaled matrices L+R/s and
C+G/s with the shared modal Cholesky/Jacobi eigensystem. If the resulting
mode propagation exponent is u, it constructs the exact reciprocal
physical two-port [A,-B;-B,A] using modal y*coth(u) and y*csch(u).
Evaluation uses exp(-u) and -expm1(-2u) to avoid overflow and retain
resolution for short cables. Mode attenuation and complete near/cross
admittance blocks remain available without an AC P-device registration.

For numerical parity, cplsetup.c floors EVERY upper-triangular R
coefficient, including the mutual terms, at 1e-4 before transient
fitting. This is done in the spectral calculation, not in the stored
physical model, and differs from DC loading's use of diagonal R only.
Upstream C setup also requires explicitly given model R/G/L/C/length,
even though the standalone plan can represent physically absent R/G
as zero; this requirement must be enforced at executable registration.

The bounded CPL three-pole Padé primitive shares the already-native
TXL 3x3 solve, cubic roots and real/conjugate pole residues rather than
duplicating the numerical algorithms. It recognizes identically
constant and zero kernels without inventing poles, rejects unstable
fits and accepts C's eight inverse-rate samples through a bounded
Neville/match polynomial expansion. These are *fitting primitives*:
the matrix h1/h2/h3 channel assembly and causal recursive convolution
are not yet integrated into the transient solver. Per-element unconstrained
Padé h1 fitting can produce unstable poles for physically passive coupled
cables; a complete backend must use stable coupled-matrix fitting rather
than excluding those valid models or exposing growing impulse responses.

## Front-end representation

The independent parser reads P card terminal groups in exact near/far
order. For n conductors it requires exactly n near signal terminals,
one near return, n far signal terminals, one far return and a final
named model. An optional length override is a structured scalar or
parameter expression. It does not silently substitute a shared ground
for an explicitly different return terminal.

CPL model R/L/G/C setters carry an ordered, source-positioned numeric
vector in ParameterKind::NumericVector; scalar literals and braced
expressions remain in their original written order and the complete
model retains repeated setters. Vectors are bounded by the maximum
eight-conductor packed upper triangle of 36 entries. Their exact
physical dimension is checked when the instance's conductor count
is known, not guessed during tokenization. The normalized writer,
AST dump and semantic comparison preserve vector components, including
per-component source positions and expressions.

## Remaining execution stages

1. Completed: resolve scoped model parameters, including vector
   expressions, .func calls and instance length overrides, into immutable
   per-instance physical plans. Position-preserving component substitution
   records ModelVectorComponent sites for diagnostics. Refuse absent/invalid
   matrices and unsupported setters instead of defaulting coupling to zero.
2. Completed (internal): C-compatible DC MNA branch currents (2n rows),
   real Circuit setup and Newton/linear paths, series max(Rii,1e-4)*length
   KVL (C setup's 100 micro-ohm per-unit resistance floor), one
   near/far current continuity constraint per conductor, and all four
   +0.1*gmin C matrix cells. The reference C DC loader ignores mutual
   R and shunt G, and does not stamp explicit near/far returns. These C
   limitations are documented, not propagated to the transient model.
3. In progress: spectral samples, Padé primitives, lossless port-wave
   launching and causal read-only modal delay sampling exist. Complete
   the coupled h1/h2/h3 matrix-channel frequency fitting, accepted-step
   recursive convolution and physical transient branch stamps in the
   shared delay-history and transient-input architecture. Every rejected
   Newton or truncation attempt must leave accepted history intact.
4. Differentially test C-backed CPL DC/transient examples,
   differential/common-mode and three-plus-conductor crosstalk,
   near-zero coupling, passivity, sample/hold and rollback. Measure
   memory and CPU by conductor count and accepted-history length.

The fixtures tests/cpl_lines.rs, tests/cpl_model.rs,
tests/cpl_elaborate.rs and tests/cpl_dc.rs cover immutable geometry,
declaration binding, scoped model-vector evaluation and actual assembled
DC and Newton equations, respectively. The one-wire 1V/1ohm-through-1ohm
fixture matches the installed C ngspice operating point to 1e-12 V.
The installed C binary aborts with double-free on even uncoupled two-wire
CPL cards, so reference comparisons for multiple conductors currently
use the upstream cplload.c source-level stamp order plus native matrix
regression; there is no validated multiwire C numerical reference yet. It is NOT a claim of complete P
device support, and no inconsistent partial simulation is exposed.

References: upstream ngspice cplsetup.c/cplload.c/cplinit.c and
src/spicelib/parser/inp2p.c; the ngspice CPL user manual. Thevenin's
public cpl.rs port was inspected as an implementation reference but
is not a linked dependency or copied implementation.
