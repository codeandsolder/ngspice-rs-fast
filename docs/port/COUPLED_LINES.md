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
**the native P Device does not need to invert B**. For nonmodal AC it allocates N
physical input-current unknowns and stamps the exact transfer equations in MNA:

  Vout - A Vin - B Iin' = 0
  Iinput_entering = Iin' / Zscale
  Ioutput_entering = -(C Vin + D Iin') / Zscale.

At zero frequency a lossless line has A=D=I, B=C=0, so the branch rows enforce
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
`squarings`; nonfinite material, singular interface and resource
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

## Exact LC modes and flight time

The additional pure `CoupledLinePlan::exact_lc_modes()` prepares the full
reciprocal LC eigenbasis for positive-definite L and C without assuming that
they commute. Factor C = B B^T, diagonalize the *symmetric* metric
H = B^T L B = Q diag(lambda) Q^T, then the physical voltage and current
bases are V = B^(-T) Q and I = B Q. They satisfy C V = I, L I = V diag(lambda)
and V^T I = identity. Each exact flight time is
`length * sqrt(lambda)`, and the modal admittance scalar is
`1/sqrt(lambda)` **in metric-normalized modal coordinates**. The native travelling-wave model power-normalizes each voltage basis column by
`1/sqrt(y)` and each current basis column by `sqrt(y)`, giving reciprocal
physical power-wave transforms. Extremely small/large material magnitudes
are scaled during the Cholesky/Jacobi solve and the physical LC products are
reconstructed using split binary exponents. Pure RC/RL and other singular
material regimes remain in the general diffuse/descriptor solver, rather
than inventing a positive phase delay for modes with no finite wave speed.

For zero R and G with positive-definite L and C, the native device uses
these modes directly in DC, AC and transient. There are 2N physical
inward-current branch unknowns, and for each mode the receiving wave
`v_m - i_m` equals the opposite launching wave `v_m + i_m` delayed by
its physical flight time. The DC factor is unity and the AC factor is
`exp(-j omega tau_m)`, preserving exact short and resonant constraints.

Transient stores power-normalized incident waves in the existing accepted
waveform history, seeded by the DC operating point. Retarded values are
linearly interpolated between accepted samples; within a trial interval
the interpolation coefficient of the new incident wave is stamped into
MNA, rather than extrapolating past the accepted solution. The shortest
modal flight time bounds the step and slope changes schedule arrival
breakpoints. Pruning retains the bracket at the longest delay. Rejected
trials do not mutate the accepted history. No spatial charge/flux slots
or finite-volume approximation are needed for this lossless regime.

The scalar 50-ohm matched regression (1 ns flight time) now checks the
entire pre-arrival window to 1e-10 V, including the former finite-volume
pre-arrival artifact at 2.25 ns. A matched two-conductor regression checks
both physical voltages against independent even/odd waves with 0.6 ns
and 1.4 ns delays, and a mismatched DC-biased line checks steady-state
initialization.

General lossy lines with positive-definite L/C still require the passive
coupled delayed-scattering transient implementation. The finite-volume
approximation below does not guarantee an exact physical front and must
not be presented as complete distributed transient parity.

## Passive finite-volume cable

For lossy or singular LC material the current `.tran` approximation does
**not** use a sampled AC matrix. It discretizes the full
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

For integration coefficient `a0`, a section has series impedance
`Z = dx*(R + a0*L)` and half-shunt admittance `H = dx*(G + a0*C)/2`.
Accepted energy histories supply a series voltage `hseries` and half-shunt
currents `hleft`, `hright`. Instead of inverting Z, use the hybrid port
operator (both port currents enter the section):

  Vleft = A Ileft + B Vright + f
  Iright = -B^T Ileft + D Vright + g

where in common balanced current units

  B = inverse(I + Z H), A = B Z, D = H + H B
  f = B (hseries - Z hleft), g = hleft + hright + H f.

A and D are positive semidefinite. The identity term in the solve
preserves exact zero-impedance modes, even with partially singular,
noncommuting material matrices. The two adjacent sections share interface
voltage `Vm` satisfying

  (I + A_right D_left) Vm = A_right B_left^T Ileft
      + B_right Vright + f_right - A_right g_left.

Composition eliminates the interface with this equation without inverting
series impedance or multiplying growing ABCD transfers. The condensed MNA
has N physical input-current unknowns, and the output KCL is stamped from
the hybrid operator. The homogeneous cell coefficients are computed once
per load. Retained affine interface operators reconstruct each cell's
voltages and series currents using the same elimination as the terminal
stamp. Accepted energy histories are written only through the existing
trial/accept transaction.

For exact continuous DC, convert a bounded short-segment transfer to a
hybrid operator by solving its A block, then repeatedly double with the
same positive-real interface elimination. This avoids both exp(+gamma L)
overflow and inversion of a singular series B block. DC seed states are
reconstructed from hybrid half-sections with zero derivatives. The spatial
transient model still approximates the continuous lossy line; it does not
satisfy the exact finite-speed front. Ideal zero-series modes remain exact
constraints for any section count, including partial-rank R and L.

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
exact lossless modal transient and approximate lossy/singular finite-volume
transient with trial/accept states and physical N-port KCL have implementation. The full strict lint and campaign validator are
responsible for integration. Remaining work includes general lossy exact-front transient, singular
long-line AC constraints, and broad performance, energy/passivity and
variable-step verification.
