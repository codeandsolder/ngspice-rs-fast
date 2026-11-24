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

## C reference and test contract

Upstream C: src/spicelib/parser/inp2p.c,
src/spicelib/devices/cpl/cplsetup.c and cplload.c. Example models:
tests/transmission/cpl3_4_line.cir, cpl_ibm2.cir. Crucially the C
CPL device's DEV acLoad hook is NULL: C's implementation is transient
only. Rust AC/SP validation uses independent analytic eigenmode and
transmission-matrix calculations rather than inventing a C reference.

The independent regressions in tests/cpl_lines.rs currently cover
single-wire lossless cotangent/cosecant, 2-wire even/odd coupled
orthonormal modes, nonsingular zero-frequency R-matrix inverse,
uncoupled three-conductor limits, separate return-node KCL, and
negative/singular/oversized matrices.

## Integration boundary / follow-on

The typed distributed DC/AC transfer matrices, exact branch-constraint MNA
and Device interface are in place.
Native P/CPL netlist parsing, including separately positioned real-vector
coefficients and independent return terminals, and model binding are active.
The complete passive rational transient propagation is a separate step. Arbitrary AC frequency responses cannot
be sampled into transient forcing: correct transient must derive from the
time-domain RLGC system, using an accepted/trial history or validated
passive rational spatial approximation. No nonlinear transient dynamics
are pretended to be handled by a frequency-only matrix.
