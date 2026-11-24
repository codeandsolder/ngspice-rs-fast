# Native forced harmonic balance — development design (#172)

This is a separate frequency-domain solver, not periodic shooting, startup
transient integration or small-signal distortion. It solves the projected
physical residual F(x, phase) + dQ(x)/dt = 0, with MNA branch constraints
included in F and signed flux contributions included in Q.

## Harmonic representation

For independent fundamentals f_i and orders h_i, the retained rectangular
integer lattice has product(2 h_i + 1) real degrees of freedom per physical
unknown. DC is stored once; every positive physical frequency has a real
cosine and sine coefficient. Negative-frequency modes are implied by
conjugate symmetry. Output phasors use peak amplitudes (a - j b) for
a cos(wt) + b sin(wt). Frequencies are sorted numerically, not by lattice
ordinal. Mixed-sign tone combinations are retained even when their frequency
is much smaller than either fundamental.

Commensurate sources should share an axis with the common fundamental and
adequate retained order. The lattice rejects duplicate physical frequencies
or nontrivial zero-frequency combinations rather than double counting them.
Close tones whose retained combinations are distinct do not require a
long physical common period.

## Alternating frequency/time Newton method

Evaluate F, Q and their exact derivatives G, C at points of an oversampled
independent phase torus. Project back with the orthogonal real Fourier basis:
DC weight 1/N, other coordinates 2/N. The derivative rotates each cosine/sine
pair (a,b) to (w b,-w a). The expanded Jacobian is assembled from

    J_kl = sum_s [P_ks G_s + (D P)_ks C_s] B_sl

where P is the projection, B the synthesis and D the spectral derivative.
Only physical sparse MNA sites expand to harmonic blocks. Bound matrix
patterns retain structural zero entries and permit symbolic pivot reuse.
Row tolerances normalize the residual, with a decreasing-residual line
search and explicit failure on nonconvergence.

Oversampling is at least 2 per axis. A finite grid does not eliminate aliasing
for arbitrary nonlinear devices; verify convergence as both retained order
and oversampling increase.

Resource admission independently bounds axes, coordinates, phase samples,
basis entries, expanded unknowns, Jacobian entries, worst-case factor fill,
cumulative assembly work, model evaluations, Newton iterations and line
search trials. There is no transient fallback.

## Circuit integration

The circuit adapter uses prepared exact-state MNA loads without history
carryover. The static residual includes independently synthesized source
forcing; linear charge and signed flux use the existing descriptor matrix.
Diodes, level-1 JFETs and BJTs expose physical nonlinear charge and its
derivatives, including BJT cross-charge coupling, device polarity and
multiplicity. Memoryless behavioural sources contribute static equations.
History-dependent devices, switches with discrete state, explicit time or
frequency-dependent behavioural expressions, unsupported charge models
and nonperiodic forcing are rejected explicitly.

## Netlist and API

```
.hb 1meg 3 1.013meg 3 oversample=3 hbiter=40 abstol=1p vntol=1u
```

Specify one frequency/order pair per independent axis. For related tones
use a single common fundamental with sufficient order. Independent sources
may be constant or undamped SIN with an explicit retained frequency.
The output is a complex frequency plot containing DC and peak phasors
ordered by positive physical frequency. Library callers can construct
a HarmonicLattice, solve a circuit with solve_circuit, and optionally
supply a previous coefficient vector as a warm start. Explicit equation
tolerances apply to the projected physical residual.

## Validation

Focused regressions exercise analytic RC and RLC response, Fourier and
derivative conventions, two-axis projection, nonlinear manufactured
solutions, diode rectification against long-run transient, two-tone diode
mixing, behavioural amplifier compression, BJT/JFET charge integration,
typed deck analysis, and invalid inputs and resource limits.
