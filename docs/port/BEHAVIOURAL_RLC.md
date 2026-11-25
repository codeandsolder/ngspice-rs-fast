# Native behavioural passive devices

Behavioural primary expressions on R, C and L, and explicit capacitor Q
expressions, are compiled with the behavioural expression program.
The current orientation is from the first terminal to the second.

## Constitutive residuals

| Device | Algebraic residual | Storage residual |
| --- | --- | --- |
| R | v - R(x)i | 0 |
| C (differential) | i - C(x)u; u | 0; -v |
| C (explicit charge) | i | -Q(x) |
| L (differential) | v - L(x)u; u | 0; -i |

The system is F(x) + dQ(x)/dt = 0. The second equation on differential
devices defines the rate u, rather than treating the product of a
voltage-dependent capacitance and voltage as physical charge. All direct
and expression derivatives contribute to the same Jacobian, including
when control variables and terminal or branch variables coincide.

The first branch represents physical current; the second, when present,
represents the derivative. KCL uses the physical current only. Explicit
charge and differential storage use the solver's existing trial/accepted
state integration. DC initializes the stored quantity and sets its
derivative to zero; small-signal assembly separates the algebraic and
storage Jacobians. Expressions are bound once and reuse the existing
behavioural evaluation workspace.

Resistance uses the implicit law rather than dividing by a possibly
zero resistance. Multiplicity and temperature factors follow the
corresponding C behavioural passive transformation. In particular C
and Q apply reciprocal temperature scaling, R and L reciprocal
multiplicity. Explicit noise and unsupported instance setters are rejected.

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

`ddt(f)` is expanded once into a solver-owned coordinate y, with residual
F=y and storage Q=-f. It consumes one internal node and two accepted-state
slots, not a voltage-source branch or an artificial electrical load. The
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
every instance owns its accepted/trial integration state. The native
model grammar accepts scoped expressions within coefficient vectors.

Native `int_ic` values are listed in descending physical integrator order;
the normalized coordinate x_i is D_leading*physical_i/w^i. Missing native
initial values default to zero, matching the C implementation. The typed
engine API also allows equilibrium initialization for non-origin poles
with zero unconstrained origin-pole coordinates. Gain acts on input forcing,
not the free response from explicit initial conditions. UIC fixes all
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
inconsistent constraints before history acceptance. Its nonlinear
instantaneous check still uses the static linearization supplied by the
transient driver, not a general nonlinear constrained solve.

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

## Remaining work

Source-dialect E/G LAPLACE normalization is tracked in the preprocessing
repository (#44), using the native rational transfer model. Statistical
functions, arbitrary charge-law initial-condition constraints, a
fully nonlinear instantaneous consistency solve, noise, mutual coupling,
broader reference coverage, and nonlinear convergence/candidate testing
remain within issue #184.
