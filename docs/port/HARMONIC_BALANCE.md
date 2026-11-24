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
The implementation obtains the coefficients of G and C independently on the
doubled integer lattice using separable projections. It constructs each block
from their sum/difference modes by harmonic convolution, avoiding an outer
product for every phase sample. DC charge offsets are subtracted before
projection; a constant derivative field uses an exact DC-only fast path.
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
multiplicity. BSIM3 uses the physical terminal charge evaluation and shares
its charge derivatives with AC, including the inverse operating mode.
Memoryless behavioural sources contribute static equations. Exact device
trials and sparse matrix-vector outputs reuse their allocated storage.
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
typed deck analysis, BSIM3 forward/inverse NMOS and PMOS AC limits, and
invalid inputs and resource limits. Independent tests compare the convolution
blocks against direct projection for one, two and three axes, and verify
current and power conservation for related sources on a common axis.

An independent QucsatorRF 1.0.7 reference (Qucs-S 26.1.1 portable Linux
release, AppImage SHA256
a86ab951118bfdffc8acfda8893cde0e2aeedc7129cc73978ae7d16761ec5cb5)
converges in 11 Newton iterations on this circuit:

```
Vac:V1 in gnd U="0.8" f="100e3" Phase="0"
R:R1 in out R="1000" Temp="27" Tnom="27"
C:C1 out gnd C="20e-12"
Diode:D1 gnd out Is="1e-12" N="1" M="0.5" Cj0="0" Vj="0.7" Tt="0" Temp="27" Tnom="27"
.HB:HB1 n="63" f="100e3" iabstol="1e-9" vabstol="1e-8" reltol="1e-6" MaxIter="150"
```

The Rust regression uses a cosine (90-degree SIN) and 31 harmonic orders
with fourfold oversampling. DC and harmonics 1 through 6 match QucsatorRF
within 2 microvolts in complex absolute error. The QucsatorRF values also
agree with numerical Fourier integration of an independent ngspice-47 C
transient (last period 90–100 us, maximum step 5 ns, reltol=1e-7,
abstol=1e-13, vntol=1e-9) within 0.35 microvolts.

The 81-coordinate two-tone diode case (1 and 1.013 MHz, order four,
729 phase points) required 15,309 physical evaluations and ten corrections
from zero. A converged seed required only 729 evaluations and no
corrections. Timings on Acer varied across concurrent builds (cold
512–848 ms, warm 25–26 ms). These are individual solver timings, not
parallel throughput measurements. Solver statistics expose charged
assembly work, peak sparse factor storage and final scaled residual.
