# Coupled multiconductor transmission lines (P/CPL)

Issue #183 adds physically coupled R/L/G/C distributed signal conductors and
independent input/output return terminals. A cable contains N signal
conductors (N bounded by an explicit resource cap), with four N×N symmetric
per-unit-length matrices R (ohm/length), L (henry/length), G (siemens/length),
C (farad/length). C ngspice's CPL model cards specify each matrix as its
upper triangle in row-major upper order; e.g. for N=2, M11, M12, M22.
The instance has terminals input1...inputN, return_input,
output1...outputN, return_output. Independent return nodes are real circuit
nodes, not silently grounded.

## Physical model and frequency-domain transfer

The first-order multiconductor telegrapher system is

  d/dx [V; I] = [[0, -Z],[-Y, 0]] [V; I]
  Z(omega) = R + j*omega*L
  Y(omega) = G + j*omega*C

with all I oriented along increasing x. The transfer for cable length ell is

  [V_out; I_out_along_x] = exp(ell*M) [V_in; I_in_along_x].

A similarity transform I' = Zscale*I balances impedance and admittance
without changing electrical physics, where the positive real scalar
Zscale = sqrt(max_row_abs_sum(Z)/max_row_abs_sum(Y)) when both blocks are
nonzero. An order-13 Padé matrix exponential and power-of-two
scaling/squaring handle all noncommuting matrices (no assumption of one
independent scalar transmission line per conductor). The Padé
denominator is inverted by the existing checked dense complex solver;
nonfinite or over-budget propagation is a numerical error.

Partition the scaled transfer as [A B; C D], in V,I' coordinates and
define B_i = inverse(B). Both terminal currents are defined entering
the line; the 2N×2N physical nodal admittance is

  [[-B_i A,        B_i],
   [D B_i A - C,  -D B_i]] / Zscale.

This handles DC R/G conduction and arbitrary-frequency AC/SP through the
same coupled equations, including real losses and mutual inductance and
capacitance. The input and output return terminals participate in signed
KCL stamps for every 2N×2N matrix entry. A singular B block cannot produce an ordinary two-ended nodal admittance.
The diagnostic `CoupledLinePlan::admittance_at` rejects this conversion, but
**the native P Device does not need to invert B**. Instead it allocates N
branch-current unknowns and stamps the exact transfer equations in MNA:

  Vout - A Vin - B Iin' = 0
  Iinput_entering = Iin' / Zscale
  Ioutput_entering = -(C Vin + D Iin') / Zscale.

At DC a lossless line has A=D=I, B=C=0, so the branch rows enforce
Vout=Vin with KCL currents entering and leaving the cable. Exact shorts and
resonant constraints are therefore preserved as actual branch equations,
not replaced with zeros or an arbitrary large conductance. The same
augmented formulation handles dissipative cable segments and complex AC.
The full sparse MNA setup records the required node-to-branch and
branch-to-node positions once; the frequency-dependent transfer coefficients
are then stamped directly at each solve frequency.

### Stable passive very-long-line DC/AC

Even a physically passive line may have a **nonrepresentable ABCD transfer**:
for scalar R=G=1 (per metre), 1000 metres creates eigenvalues ±1000 in
the transfer exponent, but the physical terminal admittance remains
approximately `[[1,0],[0,1]]` siemens. It is incorrect to refuse this
physical circuit because an intermediate `exp(1000)` overflows.

For a lossy cable whose balanced telegrapher operator has infinity norm
above 4, the frequency solver divides *physical length* by successive
powers of two until a short segment can be exponentiated and its physical
2N-port Norton admittance obtained. Each doubling combines identical
segments by eliminating the internal interface with the complex Schur
complement `K = Y22_left + Y11_right`. This keeps the terminal admittance
finite and reciprocal as end-to-end coupling tends to zero, without
building the full-line ABCD. The native Device stamps the resulting full
N-conductor Norton model with the same N branch-current unknowns, and
separate input/output return KCL. Small lines and lossless ideal/resonant
constraints retain their direct augmented branch equations.

The bounded number of doublings is governed by the plan's explicit
`max_squarings`; nonfinite material, singular interface and resource
limits remain errors, never zero admittances. Independent long-line tests
cover scalar unit characteristic admittance at exp(1000), and a reciprocal
noncommuting two-conductor Riccati identity `Yc R Yc = G` at DC, alongside
native `.op` and `.ac` solves.

## Model validity and resource ownership

Material matrices must be finite and symmetric. R,G are checked positive
semidefinite and L,C are checked positive semidefinite via the existing
symmetric eigenvalue solver. This admits lossless and coupled passive
materials while rejecting active or numerically degenerate material.
Length is finite and strictly positive. The immutable CoupledLinePlan
retains material coefficients and limits. No dynamic sparse MNA state is
cached globally; line instances attach the plan to arbitrary node pairs.
Default bounds are 32 conductors, 32 MiB dense workspace and at most
128 exponential squarings. All dimensions and allocation products use
checked arithmetic.

## Causal time-domain finite-volume cable

Native `.tran` does **not** use a sampled AC matrix. It discretizes the full
symmetric coupled RLGC cable into `sections` passive pi cells. A section of
width `dx = length / sections` has two half-shunts `(G + s C)*dx/2`
and a coupled series impedance `(R + s L)*dx`. The default is 64 sections;
`.model cable cpl ... sections=128` chooses an integer in `1..=256`.
Accuracy improves as the spatial grid resolves the fastest coupled mode and
the driving waveform's rise/fall; it is not the continuous line's exact
delay/convolution, nor is it expected to have perfect front causality on a
finite grid. The line's `state_count` is `6 * conductors * sections`: for
each section and conductor, (left shunt charge/current), (series magnetic
flux/voltage), (right shunt charge/current).

For a time integration coefficient `a0`, each cell's physical companion
admittances are `S = inverse(dx*(R + a0*L))` and
`H = dx*(G + a0*C)/2`. Accepted charge/flux histories are integrated
using the same `Coefficients` as all other SPICE capacitors and inductors;
this constructs known historical affine current/voltage forcing terms.
The symmetric cell Norton matrix in end-to-end *entering* currents is

  [Ileft; Iright] = [[S+H,-S],[-S,S+H]] [Vleft;Vright] + [Jleft;Jright].

The interface of two adjacent cells obeys a KCL equation whose coefficient
is `K = Y22_left + Y11_right`. Eliminating its voltage produces a Schur
complement. Right-to-left suffix composition eliminates all interior nodes
without ever multiplying numerically explosive ABCD matrices. The global MNA
receives N real branch-current rows (physical input currents during transient),
plus output-port KCL from the final condensed Norton admittance and source.
Only accepted charges and fluxes feed subsequent loads: Newton iterations
write *trial* local charge, derivative and flux states, and the existing
`StateHistory` transaction publishes them on acceptance. A rejected step
cannot mutate accepted history. The same Schur factors reconstruct the local
interface voltages and series currents to update all trial states.

At a transient `t=0` DC seed, the continuous DC telegrapher transfer at each
half-section reconstructs initial voltages/currents and stores initial
charges/fluxes with zero derivatives. This matches steady linear DC exactly
for the lossless shunt case, and otherwise converges to the spatial DC model
as sections increase. Ordinary `.op` without state history does not create
or require spatial state slots. The time-domain Norton kernel currently
requires nonsingular `R + a0*L` at the selected positive integration
coefficient. Degenerate exactly zero series impedance needs dedicated ideal
constraint equations, not arbitrary numerical regularization.

This is a causal, stable approximation to the continuous distributed cable,
not a bitwise port of C's CPL approximation. Spatial error, integration error,
and differences from C's own impulse-response approximation must be assessed
separately; neither C's waveform nor an untested grid size should be treated
as proof of mathematical accuracy.

## C reference and test contract

Upstream C: `src/spicelib/parser/inp2p.c` and
`src/spicelib/devices/cpl/{cplsetup.c,cplload.c}`. Models:
`tests/transmission/cpl3_4_line.cir`, `cpl_ibm2.cir` and
`examples/TransmissionLines/cpl2_2_line.sp`. C CPL has no `DEVacLoad`
hook, so the AC/SP reference is independent analytic cable theory, not C.

`tests/cpl_lines.rs` checks lossless trigonometric AC, even/odd coupled
modes, noncommuting R/L/G/C against independent RK4 integration, RC diffusion
AC, DC matrix inverse, lossless branch constraint, physical return-node KCL,
malformed materials, full `.op`/`.ac` and writer model-vector roundtrip, plus
native transient initialized steady state and causal pulse propagation.
The four-conductor transient corpus is also derived from the upstream
`cpl3_4_line.cir` coefficients, run against local ngspice 47. Retain sampled
comparisons and note both numerical method and grid resolution.

## Integration and remaining verification

Complete native P/CPL parsing, model resolution, exact distributed DC/AC,
finite-volume causal transient with trial/accept states and physical N-port
KCL have implementation. The full strict lint and campaign validator are
responsible for integration. Remaining checks include the upstream C
four-conductor temporal coupling, transient grid convergence, variable-step
Gear, performance, energy/passivity and exact degenerate-series handling.
