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
nonzero; pure series or shunt material instead uses the scale of its
length-integrated immittance. An order-13 Padé matrix exponential and power-of-two
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
**the native P device does not need to invert B**. All analyses use 2N
physical inward-current unknowns, one for every conductor at each end.
The MNA sparsity pattern binds each branch to its own signed port KCL
and allows all terminal currents and voltages in the 2N constitutive rows.

### Stable passive very-long-line AC

A passive cable may have a nonrepresentable full-length transfer: for
R=G=1 and length=1000 the growing eigenvalue is exp(1000), even though
the terminal behavior is finite. More generally, partially singular
series matrices can impose exact voltage constraints in parallel with
heavily attenuating modes; a finite nodal admittance does not then exist.

For nonmodal AC, balance the telegrapher generator and divide physical
length by powers of two until the generator infinity norm is at most 0.5.
Compute its bounded short-segment exponential by the even/odd series
of ZY, retaining both floating-point words through the material products.
Convert the transfer to reciprocal scattering in waves a=V+Zref I,
b=V-Zref I, with both currents entering. For a uniform symmetric cable
the two-ended scattering has reflection R_s and transmission T_s:

  S = [[R_s, T_s], [T_s, R_s]].

Doubling the segment eliminates its internal incident waves through the
Redheffer star product:

  X = solve(I - R_s R_s, T_s)
  T_double = T_s X
  R_double = R_s + T_s R_s X.

This composes bounded terminal relations rather than exponentially
growing ABCD transfers or a potentially nonexistent Norton matrix.
All operations in the short-segment and cascade preparation retain
two-word precision, shared with the LC eigenbasis calculation.
Form the small complementary coefficients I±S before narrowing to
binary64, then stamp (I-S)V - Zref(I+S)I=0, with ordinary physical
branch KCL. No inverse of B or I+S is required in native AC. DC uses
the real passive hybrid elimination described below.

The diagnostic finite-admittance query still uses the separate bounded
Norton elimination when it needs a finite Y. It is not the native device
representation and rejects ideal constraints. Both paths have explicit
work and memory budgets. Tests cover exp(1000) attenuation, a
noncommuting reciprocal characteristic Riccati identity, complex
rank-deficient lines (including noncommuting series/shunt material),
and the analytic pure-series and pure-shunt limits.

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
reconstructed using split binary exponents. The immutable Cholesky,
symmetric Jacobi and basis reconstruction use two-word floating-point
arithmetic before rounding the finished modal basis to binary64. This
prevents nearly coincident eigenvalues from degrading componentwise
physical reciprocity without increasing runtime stamp precision. Pure
RC/RL and other singular
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
breakpoints. Each accepted knot stores the two directions' modal waves
and their absolute evaluation-error bounds (4N binary64 values in total).
The slope-change detector propagates these bounds through extrapolation:
a matched port's almost-cancelled reflected wave is not treated as a real
corner merely because its relative roundoff is large. This preserves weak
physical reflections without scheduling exponential numbers of numerical
noise events. Pruning retains the bracket at the longest delay. Rejected
trials do not mutate the accepted history. No spatial charge/flux slots
or finite-volume approximation are needed for this lossless regime.

The scalar 50-ohm matched regression (1 ns flight time) now checks the
entire pre-arrival window to 1e-10 V, including the former finite-volume
pre-arrival artifact at 2.25 ns. A matched two-conductor regression checks
both physical voltages against independent even/odd waves with 0.6 ns
and 1.4 ns delays, and a mismatched DC-biased line checks steady-state
initialization.

## Passive delayed scattering for lossy positive-definite L/C

In the same power-normalized modal basis, split each spatial section into
a half-length exact lossless modal flight, an exact static distributed R/G
scattering relation, and a second half-flight. This symmetric Strang
splitting approximates the continuous noncommuting RLGC generator while
preserving reciprocity, passivity and physical arrival times. It is exact
when the loss and propagation generators commute. Every mode conversion
and internal reflection follows the sum of its actual modal path delays.

For mutually converting modes, the time-domain solver uses a continuous
method of steps. Each stage merges the knots of its physically delayed
incident piecewise-linear waves and applies its static scattering matrix.
The internal substep is at most the shortest interstage flight. Launch,
interstage and exit delays are continuous, not rounded to an integer
lattice. Ordinary steps cannot traverse a complete interstage flight: terminal
loads sample only the accepted history and the necessary terminal
scattering rows, while acceptance propagates the realized scalar waveform.
Sorted scattering times are sampled by monotonic cursors, using the same
interpolation and error checks as random access. Ordinary acceptance
constructs each next delay window directly from the old history and one
stage's newly scattered waveforms. It keeps the old history immutable
and does not retain a full duplicate trial. If the newly scattered curve
already contains the delay-window predecessor, the expired history is
not copied. Accepted waveforms are stored in immutable blocks of
binary64 knots; complete live blocks are shared by the old and successor
versions. Only partially retained blocks and the appendable suffix are
copied. Oversized trials use waveforms affine in all new boundary
endpoint increments, so the terminal relation includes arrivals and internal
feedback during the trial. Rejected trials cannot advance the accepted
wave histories.

Histories retain only the delay window needed by the next stage or
terminal and the source-corner times needed for front protection.
All accepted boundary samples remain in the input PWL waveform and
participate in scattering. Source-corner markers, however, are independent
of solver sampling density: resolved slope changes use the existing
breakpoint significance and roundoff criterion, and exact transitions
into or out of flat source segments are marked even below that threshold.
The physically delayed first and straight modal arrivals of these markers
are protected; a pending affine endpoint is conservatively treated as a
potential corner in each input column. In the scattered waveform, each
affine component's exact zero-support boundaries and terminal load
samples are also protected. Other PWL knots, including unmarked sampled
slope changes and secondary arrivals, remain subject to the same numerical
error and energy tests, rather than being forced to propagate as exact
corner times. To prevent combinatorial growth of reflection knots, the
solver may replace a curve by an endpoint-preserving chord or two-segment
tent whose peak is at an existing knot, avoiding synthetic arrival times.
It bounds the normalized
interpolation error independently in every affine component (the known
wave uses its current causal history-window amplitude as a scale), and checks
the wave-energy Gram inequality over every prefix of the replaced
interval. For a scattering matrix S, a certified upper bound q on its
squared operator norm allows the inequality q E_reduced <= E_exact:
summing this bound over the output channels yields total outgoing
energy no greater than the incident energy. The bound is prepared with
outward-rounded Gram entries and either Gershgorin or interval LDL
positive pivots. Half of any certified strict dissipation is reserved;
when no margin is proven, q=1 retains the original per-wave contraction.
A cubic Bernstein certificate enforces every prefix for arbitrary
combinations of endpoint increments; inconclusive certificates trigger
subdivision rather than unverified reduction. Scalar chords with a
certified strictly dissipative scattering bound can pass the same error
and prefix tests directly over the original knots without allocating
normalized affine columns. Other candidates use the general reducer.
Protected arrival times are merged into the sorted curve, borrowing the
original knots when no interpolation is needed. The waveform-error
tolerance is unchanged. As with other numerical
integration tolerances, local waveform error is not a global pointwise
error bound. Reduce `reltol` to refine the history approximation.

For separable modes or a single scattering stage, the solver instead
compiles the inexpensive exact impulse response. The ordered arrival map
keys events by their rounded time and node; one lookup either coalesces
the contribution in its original arithmetic order, retaining the first
two-word arrival time, or inserts a new event after pruning and resource
checks. Finite boundary ramps are integrated against its cumulative response. When a verified
nonnegative absolute-path majorant exists, numerical roundoff-scale
contributions may be discarded using a single accumulated allowance per
input column; first arrivals are retained. Finite-support response
histories fold expired segments into the steady baseline.

DC initializes both representations consistently with the static scattering
network. The spatial splitting approximates noncommuting R/G and L/C;
refine `sections` for spatial accuracy. Explicit workspace and work
limits account for retained scalar histories and affine trial scratch.
Output-row batches and overlapping time chunks bound temporary scattering
curves without changing their exact endpoint relations. The ordinary
acceptance workspace accounts for both accepted versions and one stage's
scratch under the total configured limit. A shared immutable block is
charged once, including its reference-count allocation, while each
version's block descriptors and all newly allocated blocks are charged
separately. Thus the continuous solver need not reserve half the budget
for a duplicate of the unchanged history. The exact impulse solver
retains its separate half-budget policy. Exhaustion
is reported rather than silently discarding significant arrivals.

## Passive finite-volume cable

For singular LC material the current `.tran` approximation does
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
has 2N physical terminal-current unknowns, with the output-current
constitutive rows stamped directly from the hybrid operator. The homogeneous cell coefficients are computed once
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
exact lossless modal transient, passive split-generator lossy modal transient,
and singular-LC finite-volume transient with trial/accept states and physical
N-port KCL have implementation. The full strict lint and campaign validator
are responsible for integration. The lossy spatial split and dense modal
path scaling require ongoing numerical and performance verification.
