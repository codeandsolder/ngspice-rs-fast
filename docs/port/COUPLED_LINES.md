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
src/netlist/elaborate.rs. The public P-device
registration remains intentionally pending; parsing a model must not
make an incomplete or uncoupled simulation silently executable.

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
2. Build near/far MNA branch-current unknowns and DC equations with
   the reference binary's actual coupled-line DC behavior and proper
   terminal/return identities.
3. Implement upstream CPL modal frequency fitting, Pade poles/residues,
   convolution and bidirectional delayed waveforms in the existing
   device-owned delay-history and transient-input architecture. Every
   rejected Newton or truncation attempt must leave accepted history
   intact. Model validation must reject unstable poles.
4. Differentially test C-backed CPL DC/transient examples,
   differential/common-mode and three-plus-conductor crosstalk,
   near-zero coupling, passivity, sample/hold and rollback. Measure
   memory and CPU by conductor count and accepted-history length.

The fixtures tests/cpl_lines.rs, tests/cpl_model.rs and
tests/cpl_elaborate.rs cover immutable geometry, declaration binding and
scoped model-vector evaluation, respectively. It is NOT a claim of complete P
device support, and no inconsistent partial simulation is exposed.

References: upstream ngspice cplsetup.c/cplload.c/cplinit.c and
src/spicelib/parser/inp2p.c; the ngspice CPL user manual. Thevenin's
public cpl.rs port was inspected as an implementation reference but
is not a linked dependency or copied implementation.
