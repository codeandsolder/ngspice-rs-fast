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

## Device integration plan

Use the existing prepared circuit run and exact state-independent/nonlimited
device load path. Source forcing must be expressed on the phase lattice,
not by picking a single physical time for each tuple. Static residual and
charge Jacobians must be evaluated at the same state. Reuse the actual
device charge/flux laws rather than using C's AC capacitance as a surrogate
for nonlinear charge. Start with linear MNA, diode physical charge and
explicit periodic SIN forcing; extend nonlinear charge coverage through
device-specific physical charge hooks. Discrete state, nonperiodic and
explicit-time device equations need an explicit support policy.

## Development checkpoint

The multidimensional lattice, direct transforms and generic sparse AFT
Newton kernel are implemented as library building blocks. Circuit binding,
typed analysis-plan/CLI integration, device coverage and independent
simulator comparisons remain in development. Do not interpret this document
as claiming full HB support.
