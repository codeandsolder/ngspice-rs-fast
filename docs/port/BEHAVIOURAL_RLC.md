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

## Remaining work

Lexical runtime detection through nested macro calls and shadowed probe
names needs further testing. Dynamic expression operators, explicit
nonlinear initial-condition setters, noise, mutual coupling, and broader
reference and convergence cases remain separate work within issue #184.
