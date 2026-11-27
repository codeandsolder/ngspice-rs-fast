# LTspice idealized diode

The explicit model family `ltspice_ideal_diode` is selected by the
LTspice dialect preprocessor. Native SPICE `D` model cards continue
to select the ordinary semiconductor diode. This device implements a
memoryless two-terminal conduction law, not a semiconductor charge
or recovery model.

## Constitutive law

Public model documentation:
https://ltwiki.org/files/LTspiceHelp.chm/html/D-device.htm

For either conducting direction let T be its threshold voltage, h
the off conductance, g the on conductance, w the smoothing width and
L an optional current limit. Use the magnitude of voltage on that
side and reflect the reverse current's sign.

Below T, I=hV. Above T define d=V-T and the baseline I0=hT.
For w>0 the transition adds a quadratic increment to I0:
d times the average of h and the interpolated slope.
Without a finite current limit the slope at the outer join is g;
beyond the join its linear continuation is used.

For a finite limit, define u as the normalized current increment at
the outer join. Matching the quadratic to the tangent of the
current-limited branch gives

```text
u = (w/(2L)) * (h + g*(1-u*u))
G_join = g*(1-u*u)
```

The positive quadratic root is evaluated without subtracting nearly
equal roots. Beyond the join, z=(d-w)*g/L and t=tanh(z).
Hyperbolic addition yields

```text
I = I0 + I_join + L*t*(G_join/g)/(1+u*t)
G = G_join*sech(z)^2/(1+u*t)^2
```

With w=0, u and I_join are zero. Computing sech squared from the
decaying exponential retains representable derivatives after tanh
rounds to one. Model inputs for which no tangent-matched transition
exists are rejected. Forward and reverse parameters are independent.

## Solver binding

Ron defaults to one ohm; Roff defaults to the reciprocal of the
effective Gmin (zero conductance for zero Gmin); Rrev defaults to Ron.
No explicit Vrev means no reverse conducting branch. M scales
terminal current and N scales terminal voltage; differential
conductance scales by M/N. Model values are resolved before
elaboration. Explicit parameter edits are validated.

The DC Jacobian is the derivative of the same current used in the
residual. AC and pole-zero use this differential conductance.
Transient loading adds no charge. Newton trials for finite-limit
branches avoid crossing directly into numerically flat saturation
tails; the previous trial or accepted predictor reference is owned
by the engine and is never used to alter the physical current law.
Observation reloads use the exact supplied bias.

## Qualification

Synthetic LTspice 24.0.12 DC witness points are recorded in
`tests/ltspice_ideal_diode.rs` (issue #226). They include both
directions, smoothing alone, current limiting alone and their
combination. The unmodified ngspice XSPICE sidiode witness is used
only for common piecewise-linear regions, not as an oracle for the
different smooth or finite-limit regions. The witness circuit and
regeneration command are under `tests/fixtures/ltspice_ideal_diode`.

A small default leakage discrepancy remains: a measured 1.001 pS
with a documented 1 pS Gmin. The implementation follows the
documented default rather than inventing an additional conductance.
Area and temperature are observable instance metadata; a
temperature-dependent constitutive law has not been established.
Noise and unsupported model/instance fields are explicit
diagnostics rather than approximated semiconductor physics.
