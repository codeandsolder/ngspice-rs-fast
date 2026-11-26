# Native behavioural passive devices

Behavioural primary expressions on R, C and L, and explicit capacitor Q
expressions, are compiled with the behavioural expression program.
The current orientation is from the first terminal to the second.

## Constitutive residuals

| Device | Algebraic residual | Storage residual |
| --- | --- | --- |
| R | v - R(x)i | 0 |
| C (differential) | i - C(x)u; u | 0; -v |
| C (explicit charge) | z - Q(x); i | 0; -z |
| L (differential) | v - L(x)u; u | 0; -i |

The system is F(x) + dQ(x)/dt = 0. The second equation on differential
devices defines the rate u, rather than treating the product of a
voltage-dependent capacitance and voltage as physical charge. All direct
and expression derivatives contribute to the same Jacobian, including
when control variables and terminal or branch variables coincide.

The first branch represents physical current. A differential C/L has a
second branch for the derivative, while an explicit charge law has a
private charge coordinate z. Plot coordinates are i(device), dvdt(device),
didt(device) and q(device), with the respective physical units. KCL uses
only the physical current. Explicit
charge and differential storage use the solver's existing trial/accepted
state integration. DC initializes the stored quantity and sets its
derivative to zero; small-signal assembly separates the algebraic and
storage Jacobians. Expressions are bound once and reuse the existing
behavioural evaluation workspace.

Resistance uses the implicit law rather than dividing by a possibly
zero resistance. Multiplicity and temperature factors follow the
corresponding C behavioural passive transformation. In particular C
and Q apply reciprocal temperature scaling, R and L reciprocal
multiplicity. Signed finite coefficients are preserved, including active
negative resistance and differential capacitance or inductance. This is
not a passivity guarantee. Singular constitutive systems and nonfinite
values produce solver diagnostics. Unsupported instance setters are rejected.

## Verification

Independent ngspice 47 measurements on synthetic OP/AC/transient cases
agree with the analytical differential C, explicit Q and L laws.
The transient test distinguishes i=C(v) dv/dt from d(C(v)v)/dt.
The reference run contains 1008 points; away from ramp endpoints,
maximum errors relative to the analytical laws are 6.5e-14 A for C,
5.5e-7 A for Q, and 1.2e-12 V for L.

## Lexical resolution and dynamic operators

Passive primary expressions are classified after lexical macro expansion.
Scalar expressions retain the numeric parameter evaluator and its precision.
Runtime expressions are resolved in the instance's scope and compiled once.
Resolved current references use existing branch rows where possible; other
targets share one series voltage measurement with collision-checked names.
Model-backed inductors expose their existing physical current coordinate.

`ddt(f)` is expanded once into a solver-owned signal y and private input
storage z. The algebraic equation is z=f; the evolution equation has
F=y and Q=-z. It consumes one internal signal node, one private storage
branch, and two accepted-state slots, not an artificial electrical load. The
same integration and Jacobian assembly serve transient, small-signal and
physical DAE analyses. Nested operators get independent state and share
ordinary expression bindings. In DC y=0; the AC response is j omega times
the exact bias derivative. This intentionally differs from ngspice 47's
PTddt, which returns zero outside transient and uses a sampled derivative
with its own time history. Independent C transient measurements on a
quadratic ramp showed a maximum interior error of 3.000000003 V/s at a
1 us step; the native derivative tests check timestep refinement under
both trapezoidal and Gear integration. Generated dynamic coordinates
evolve continuously even when the surrounding expression does not select
their output; their inputs must therefore be defined throughout the run.

The C47 reference does not accept an idt() expression. Integrators are
instead representable as rational transfer blocks.

## Rational transfer models

The native XSPICE `s_xfer` model realizes H(s)=g N(s/w)/D(s/w), with
coefficients supplied in descending polynomial order and w in rad/s.
The normalized denominator is monic. For n storage coordinates, with
tau=1/w, the residuals implement tau*x_i'=x_(i+1), and
tau*x_(n-1)'=g*(u+offset)-sum(a_i*x_i). The output is the sum of
the numerator-derived coordinate terms and direct feedthrough.
The same residuals supply DC, AC, transient and pole-zero assembly.
There is no sampling interval, per-step polynomial parsing, or separate
AC approximation. Improper rational functions are rejected explicitly;
origin poles and unstable proper rational systems are not clamped.

The four analog port forms support differential voltage inputs, zero-voltage
current-sensing inputs, differential voltage outputs and oriented current
outputs. The compiled realization is shared by instances of a model;
every instance owns its accepted/trial integration state. Transfer plots
identify the output current as i(device), the sensing input current as
i(device:input), and the normalized states as state(device,index). The native
model grammar accepts scoped expressions within coefficient vectors.

Native `int_ic` values are listed in descending physical integrator order;
the normalized coordinate x_i is D_leading*physical_i/w^i. Missing native
initial values default to zero, matching the C implementation. The typed
engine API also allows equilibrium initialization for non-origin poles
with zero unconstrained origin-pole coordinates. Gain acts on input forcing,
not the free response from explicit initial conditions. Physical-coordinate
normalization uses extended binary exponents when an intermediate
frequency power or quotient is unrepresentable. Only the final
normalized nonzero coordinate must fit binary64. UIC fixes all
internal coordinates to their specified values (zero for unspecified
coordinates) before seeding accepted history.

Eight native tests cover orders zero through five, nonzero feedthrough,
multiple integrators, explicit initial states, all four analog port laws,
transient integration and netlist round trips. The live ngspice 47 test
has maximum AC absolute error 2.24e-16. For a ramp with a 1 us rise time,
the maximum absolute errors against the independent analytical solution
were 1.86e-7 V (Rust, 100 ns), 4.39e-4 V (C, 100 ns), and 4.88e-5 V
(C, 10 ns). The C error decreases on timestep refinement; the test checks
analytical accuracy and this convergence rather than requiring agreement
with the coarser sampled C realization. The Rust zero-order gain also
works where the tested C47 zero-order model crashed.

## Initial-condition constraints

The UIC initializer binds prescribed storage coordinates independently
of their evolution equations. For a differential inductor, the fixed
current is local branch 0 and its released equation is branch 1;
the voltage/rate constitutive equation remains intact. For a differential
capacitor, the signed stored coordinate is -v, with explicit instance
`ic=` taking precedence over node hints. Its existing rate equation is
replaced by the initial voltage relation; an additional current unknown
is not required. Scalar capacitors use the same voltage-constraint
interface with an appended current, and scalar inductors publish their
ordinary fixed current. Rational transfer coordinates use the same
initialization contract for arbitrary order.

Multiport devices publish ideal voltage relations through a device-owned
interface: transfer voltage outputs are controlled, current inputs
impose zero differential voltage, and analog/DAC and ordinary sources
declare their corresponding relations. The consistency checker keeps
the established independent/controlled constraint graph and rejects
inconsistent constraints before history acceptance. The nonlinear
instantaneous check reloads the actual static device residual and
Jacobian at each iterate of the reduced constrained system. It verifies
the physical residual after reloading, refines the consistent point
toward roundoff, and uses that point as the transient starting guess.
Redundant voltage constraints within the IC tolerance are canonicalized
to the solved voltage before committing history. The ordinary device UIC
load still owns semiconductor initial charge semantics.

Arbitrary charge laws declare their own signed storage constraints.
During UIC, the physical kernel evaluates Q at the seeded point;
an instance ic= supplies the local terminal voltage (taking precedence
over the corresponding node hint), while other voltage and current
controls retain their seeded values. The resulting signed charge, not
the terminal voltage, is fixed during the instantaneous solve. The
physical charge Jacobian replaces the evolution row and every converged
candidate must satisfy the original charge equality in charge units
(reltol and chgtol). The history is canonicalized to the verified
solution within these tolerances. For example, if Q=1u*(v(out)+2*v(ctrl)),
ic=1 and the control is unhinted but its right-limit source is 1 V,
the initial charge is 1 uC and the consistent output is -1 V.

Redundant charge constraints can leave the instantaneous current
distribution undetermined. The normal sparse solve remains the fast
path; a singular or rectangular reduced system uses a sparse rank basis
with power-of-two row/column scaling and coefficient-local rounding
uncertainty. Only declared storage-current and derivative-signal
coordinates can complete the basis as zero gauges. The
original algebraic equations and actual nonlinear charge values are
all checked on reload; no constraint is relaxed by dropping it from
the basis. The instantaneous reduction alone does not establish derivative
consistency of a high-index DAE or resolve unrelated floating coordinates.

The stored input of ddt is not automatically an independent initial
condition. The rank basis considers its seeded value only after the
algebraic equations and mandatory physical storage constraints. If those
already determine the input, the history is initialized at the actual
consistent input rather than an arbitrary zero seed. If the derivative
defines an independent state (for example v=1-tau*ddt(v)), its seed is
retained, including a supplied .ic. For circuits with analytic physical
Taylor kernels, the derivative-array solver determines consistent
right-limit values and rates instead of
assigning zero to undetermined derivative outputs.

### Physical derivative arrays

Use local coefficients X[k] = h^k x^(k)/k!, where h is a positive time
normalization scale. Each degree k of the original physical equations
contributes F[k] + (k+1) Q[k+1]/h = 0. Conductance and storage Jacobian
coefficients act by convolution on the corresponding perturbations of
X. Independent sources supply analytic one-sided waveform coefficients;
behavioural expressions propagate local coefficients through the compiled
instruction DAG. Reverse differentiation of the active value graph
provides the Jacobian of the same branch, including clipped exponential
and regularized division kernels. Typed source power laws retain their
absolute-base, real-domain caret and principal complex-real distinctions.
The physical Jacobian differentiates the local value law, not legacy
regularized scalar Newton slopes; singular or nonanalytic power germs
without a classical expansion are rejected. No finite differences or
artificial derivative prehistory are used to construct the array.

Sparse elimination projects out the free higher coefficients before
selecting the necessary state and first-rate coordinates. Mandatory
physical storage constraints precede optional derived inputs.
Rank and residuals are rechecked at the nonlinear solution; the retained
order hint is only a performance optimization. Intermediate pivot products
and coefficient-local uncertainty use extended binary exponents so that
changing the time normalization does not itself erase independent
constraints through underflow. The array has explicit order, variable and
stamp budgets.

During a smooth transient step, the physical array is completed with
independent companion constraints, in physical-before-derived order.
Redundant numerical derivative constraints are not imposed: they would
differentiate the integration error of an already determined quantity.
Accepted history stores the independently evaluated storage rates.
Mandatory initial conditions are checked in the units of the stored
quantity (including differential voltage, not common-mode voltage);
only numerical companion residuals receive an arithmetic backward-error
floor.

At an independent-source breakpoint the solver integrates to the left
limit and solves the right limit while preserving physical storage.
Derivative input discontinuities that require impulses are rejected
rather than approximated by large finite rates. A successful event
starts fresh integration history on the right; the final endpoint retains
its left limit. Non-UIC initialization also checks continuity from the
DC prehistory (the prepared source contract supplies its own right-hand
startup bias, rather than the native source left-limit convention). UIC
does not invent a prehistory before the requested initial time. Prepared
PWL and signed damped-sine laws provide the same exact one-sided analytic
coefficients as the physical source evaluator.

The high-order path currently covers linear memoryless and rational
models and analytic behavioural constitutive kernels. Other nonlinear
device families continue to use their existing ordinary solver; general
physical Taylor kernels for those models are tracked in
[#242](https://github.com/codeandsolder/ngspice-rs-fast/issues/242).
Independent source waveform boundaries are scheduled explicitly;
localization of arbitrary behavioural expression branch changes is
tracked in [#241](https://github.com/codeandsolder/ngspice-rs-fast/issues/241).
The derivative-order bound and projected rank tests are not a proof of
solvability for every singular or nonanalytic nonlinear DAE. Delayed rank
emergence from time-dependent coefficients is covered at startup and exact
interior events, including coefficients vanishing through dozens of Taylor
orders. Recovery from a singular provisional Newton seed and genuinely
singular nonlinear germs is tracked in
[#251](https://github.com/codeandsolder/ngspice-rs-fast/issues/251).

Nonlinear C/L `ic=` is intentionally retained under UIC, rather than
discarded by the frontend as observed in ngspice 47 (instance and node
IC cases both remained at zero in independent discharge tests).
Analytical nonlinear RC/RL discharges and a lossless nonlinear LC
oscillator pass; the LC verifies KCL at every point and energy drift
below 0.1%. Independent C47 s_xfer UIC test (2011 points, 1 us maximum
step) agrees with the initial free-response law, with maximum analytic
error 1.84e-4 V. Native transfer tests check the same law against a
2e-5 V tolerance, second-order free response with zero gain, and
conflicts at voltage outputs and current-sensing inputs.

## Convergence and noise

Newton distinguishes physical voltage/current coordinates from internal
constitutive rates. The latter are checked through the corresponding
physical residual, including the companion derivative at the new
iterate, rather than comparing a V/s or A/s coordinate against an
absolute current tolerance. The storage evolution row is linear, with exactly zero affine offset.
For explicit storage it is power-of-two normalized when the integration
coefficient dominates. After the sparse solve the private coordinate is
reconstructed from that row, without changing electrical unknowns.
The physical derivative is independently recorded in accepted history.
Its divided differences estimate the leading truncation error without
differencing nearly equal absolute charges. This uses the same error
constants and tolerance policy as the ordinary charge estimator.

Behavioural resistors are noiseless by default. Explicit noisy=1 (also
noise=1) injects a series thermal-voltage PSD 4 k T R into the native
constitutive branch equation, using the effective operating resistance
and instance temperature. This preserves current-controlled feedback,
unlike adding a parallel Norton source outside the expression's current
measurement. Negative or nonfinite thermal resistance and invalid
absolute temperature are rejected. Generalized noise sources use
device-local branch or node coordinates, resolved during noise layout.
For a self-current-controlled resistance of 2 kohm at the bias point,
3 kohm differential resistance and 1 kohm load, the native and live
ngspice 47 output spectral amplitudes both equal
1.439447175 nV/sqrt(Hz) at the default bias tolerance.

## Coupling and candidate preparation

A K card naming an expression-controlled L has no defined native constant
coupling inductance. ngspice 47 independently rejects such a card because
its behavioural transformation removes the named inductor. The native
diagnostic requires an explicit coupled flux law instead of inventing
a constant mutual coefficient from a voltage- or current-controlled
expression. Ordinary constant-inductance K models retain their existing
coupled-flux implementation.

Numeric model-vector entries are recorded as individual literalization
sites, with their original expressions and locations. Candidate updates
therefore invalidate the prepared realization when a coefficient or
integrator initial value changes. The A/B/A regression includes failed
zero-denominator candidates and verifies independent DC and AC responses.

Candidate plans also bind the actual macro-expanded runtime law's static
reads, including captured names, embedded numparam expressions and
coefficients introduced by controlled-source lowering. The scalar
literalization record retains the actual expression when a syntactically
behavioral passive resolves to a numeric value. Unchanged effective
parameter reads preserve the validated program and circuit topology,
including when independent input changes cancel in a derived parameter.
Root option expressions use the same dependency comparison; literal
options and seeds do not invalidate otherwise reusable candidates.
Changing a compiled coefficient triggers validated structural
re-elaboration; unsupported physical coefficient gradients are reported
as such rather than returning a synthetic zero.

An opt-in preparation benchmark
(`benchmark_behavioural_candidate_preparation` in
`tests/optimization_parameters.rs`) exercised 256 source-amplitude
candidates for 64 expression-controlled resistors and one independent
source, with a fixed lexical law and literal convergence/seed options.
All candidates reused the original topology. On the development host,
prepared updates took 1.42 ms total, versus 250.6 ms for independently
parsed and elaborated decks, after a one-time 2.1 ms plan preparation.
This measures preparation, not nonlinear analysis runtime.

## External charge-network verification

An extracted EPC23101 power-FET gate-source, gate-drain and source-drain
charge network from the publicly available GeecPE2526 simulation deck
was tested with simultaneous terminal ramps (0.5 to 3.5 V gate and
10 to 2 V drain over 1 us). The six original capacitance/charge laws
were retained; the LTspice Q=(...) spelling was normalized to Q={...}.
With a 1 ns maximum step and tightened reltol=1e-8, abstol=1e-12,
vntol=1e-9, the native and C47 runs both produced 1008 points.
Over the interior 10 to 990 ns, the maximum terminal current
difference between engines was 7.36e-13 A, the native error versus
analytically differentiated charge was 1.45e-7 A, and terminal KCL
closed within 5.34e-15 A. This is a charge-network reference, not a
claim that the full vendor switching model has been ported.

## Stiff controlled-charge validation

The analytical stress case uses a 1 kOhm shunt and
Q=1 uF * (u+0.5*u^3), where u=v(out)-v(ctrl). The control ramps
from 0 to 1 V in 1 ns after 0.5 ms, while the nominal discharge
time constant is 1 ms. The tolerances are reltol=1e-8,
abstol=1e-12 A and vntol=1e-9 V.

Two numerical issues were identified: cancellation in the original
combined constitutive/integration row near the ramp endpoint, and
ngspice's erroneous symbolic derivative for odd powers of negative
bases. The latter incorrectly computes 1-1.5*u^2 rather than
1+1.5*u^2 for the dimensionless slope in this circuit. The corrected
power Jacobian follows the actual value law. Ngspice 47 still aborts
during discharge with both trapezoidal and Gear integration.

The native regression completes under both methods. The post-ramp
analytical invariant is

    2.5*ln(v)-3*v+0.75*v^2 + (t-t_end)/1 ms = constant.

| Method | Points | Max KCL residual (A) | Charge balance (C) | Decay error (V) |
| --- | ---: | ---: | ---: | ---: |
| Trapezoidal | 2052 | 1.08e-19 | 1.01e-14 | 1.10e-7 |
| Gear | 2053 | 1.08e-19 | 2.03e-13 | 4.39e-7 |

The reference numbers are measured with the split-storage formulation.
No increase in the timestep work limit or constitutive tolerance floor
is used.

## Remaining work

Source-dialect E/G LAPLACE normalization is tracked in the preprocessing
repository (#44), using the native rational transfer model. The
physical derivative-array implementation initializes cascaded derivatives
of linear and analytic behavioural networks at their consistent right
limits, including under UIC. Remaining higher-order model kernels and
behavioural branch-event localization are tracked in #242 and #241.
Broader reference coverage and numerical golden validation remain ongoing. Static statistical realization is documented in
[PARAM_EXPRESSIONS.md](PARAM_EXPRESSIONS.md).
