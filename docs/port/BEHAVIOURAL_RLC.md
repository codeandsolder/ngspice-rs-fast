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

The C47 reference does not accept an idt() expression; integral and rational
transfer blocks need explicit realization and initial-state semantics.

## Remaining work

Laplace/transfer blocks, statistical functions, explicit nonlinear
initial-condition setters, noise, mutual coupling, broader reference
coverage, and nonlinear convergence/candidate testing remain within
issue #184.
