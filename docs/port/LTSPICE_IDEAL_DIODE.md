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
L an optional current limit. Use signed anode-to-cathode voltage for the forward branch and
negated voltage for the reverse branch. Select the forward branch if
V >= Vfwd, otherwise the reverse branch if -V >= Vrev and Vrev was
specified, otherwise off conduction. Reflect the reverse current's sign.
Signed thresholds and resistances are supported.

Below T, I=hV. At an unsmoothed threshold the conducting-side slope
applies (at zero, a zero forward threshold takes precedence over a zero
reverse threshold). Above T define d=V-T and the baseline I0=hT.
For w>0 the transition adds a quadratic increment to I0:
d times the average of h and the interpolated slope.
Without a finite current limit the slope at the outer join is g;
beyond the join its linear continuation is used.

For a finite limit, let g be the magnitude of on conductance and
reflect h into the conducting slope's sign. Define u as the normalized
current increment at the outer join. Matching the quadratic to the tangent of the
current-limited branch gives

```text
u = (w/(2L)) * (h + g*(1-u*u))
G_join = g*(1-u*u)
```

The positive quadratic root is evaluated without subtracting nearly
equal roots. The complementary quadratic identity computes the join
slope independently of the rounded normalized current, and fused scaled
subtraction retains a small remaining current budget. Beyond the join, z=(d-w)*g/L and t=tanh(z).
Hyperbolic addition yields

```text
I = I0 + I_join + L*t*(G_join/g)/(1+u*t)
G = G_join*sech(z)^2/(1+u*t)^2
```

With w=0, u and I_join are zero. Computing sech squared from the
decaying exponential retains representable derivatives after tanh
rounds to one. In the extreme tail, factor the exponential before
multiplication so a large conductance can recover a representable
derivative even after the standalone exponential underflows. When h >= g, h < -g, or w*h/(2L) >= 1, the measured vendor law
instead offsets the outer tanh by half the smoothing width. Its join
uses u=tanh(w*g/(2L)) and the corresponding tangent, but the outer
current is L*u rather than the quadratic join increment. Consequently
the current can be discontinuous at the outer join; the quadratic
includes the join itself. This behavior is intentional, not a smoothed
approximation. Restore the original on-conductance sign for the limited
branch. Forward and reverse parameters are independent.

## Solver binding

Ron defaults to 1 milliohm, including an explicit zero Ron. Roff
defaults to the reciprocal of effective Gmin (zero conductance for zero
Gmin), including explicit zero Roff. Zero or omitted Rrev defaults to
Ron. Nonpositive smoothing widths and current limits disable the
corresponding feature. Negative nonzero resistance values are preserved.
No explicit Vrev means no reverse conducting branch. Independent
measurements establish an additional parallel conductance of Gmin/1000
per series element, unaffected by M. For local voltage v=V/N,
terminal current is M*I(v)+(Gmin/1000)*v, and differential conductance
is (M*G(v)+Gmin/1000)/N. This floor is outside the smoothing and
current-limiting law. Model values are resolved before elaboration.
The last rounded LEVEL setter must be 0 or 1 for the ideal family;
other levels require a distinct semiconductor model. Explicit
parameter edits are validated.

The DC Jacobian uses the derivative of the same current branch used in
the residual, with the selected one-sided slope at a discontinuity. AC and pole-zero use this differential conductance.
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

The independent LTspice 24.0.12 witnesses include 85 DC floor samples,
11 AC bias/level configurations, 135 noise-frequency samples and 3509
DC edge samples across 59 signed, defaulted and fallback model cards.
The latter include two independently measured operating-point diode
currents at exact zero bias where vendor DC sweep reporting disagrees
with the actual constitutive current.
Thermal current noise is 4*k*T times the complete terminal differential
conductance. Shot noise is 2*q*abs(I)/N, and flicker noise is
KF*abs(I)^AF/f (no extra multiplier or series-count factor). The common
noise machinery sums independent generator powers; instance temperature
overrides circuit temperature. The two simulators use slightly different
physical constants, accounted for by the measured comparison tolerance.

Area is observable metadata but electrically inert in the ideal family.
The tested semiconductor-only setters (including RS, IS, junction charge
and temperature coefficients) and the OFF flag do not modify this law.
Unrecognized fields and nonrepresentable noise remain explicit errors.
