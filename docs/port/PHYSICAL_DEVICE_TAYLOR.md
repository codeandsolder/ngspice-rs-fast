# Device-local physical Taylor equations

A physical Taylor kernel assembles the coefficients of the original
(F(x,t)+dQ(x,t)/dt=0) equations and the local state Jacobian series
of both (F) and (Q), in the finalized MNA coordinates. This is not
the Taylor expansion of a discretized companion, a limited Newton
iterate, or a small-signal AC approximation.

For a scalar junction law (y=f(v)), the value and gradient jets
are related by the analytic identity
(dy/d\tau=f'(v)dv/d\tau). Anchor the constant coefficient with the
stable scalar constitutive law and integrate the coefficient
convolution. Use the same analytic derivative for the Jacobian.
Scaled exponentials may be seeded from an already representable
scaled scalar value rather than computing an overflowing intermediate
exponential. The normalized time coordinate is chosen by the analysis;
device kernels do not assume a particular timestep.

Two-terminal current and charge use the same incidence signs and
gradient column signs. Aliased terminals cancel before stamping.
For a one-sided branch, select the branch from the voltage jet's
first nonzero coefficient at the boundary, not from an arbitrary
voltage epsilon. Explicit time/event localization remains a separate
analysis concern.

## Local storage versus global MNA charge

A global MNA charge row is the **sum** of signed charges contributed
by devices at that row. It cannot identify the charge retained in
one device's own integration history. In particular, parallel
junctions, capacitors, and multi-terminal transistor charge partitions
must not each be assigned that sum.

Devices declare their local charge/charge-rate history slot pairs.
The prepared circuit binds them to distinct absolute slot pairs.
A Taylor evaluation records the device-local charge jet and its
sparse state-gradient jets in addition to its global MNA stamps.
Each declared pair must be present exactly once; invalid ranges,
overlaps, nonfinite coefficients and excessive coefficient counts
are errors. Left-limit reversal applies equally to local charge
and gradient jets.

At UIC, junction charges from the initial load are optional seeds
rather than prescribed capacitor ICs; a source may determine their
values instead. In ordinary initialization and at source events,
the actual accepted local charge is a physical continuity constraint.
During a transient trial, each local storage pair supplies its own
numerical companion constraint, selected by the same projected-rank
logic as independent capacitor/inductor storage. Redundant numerical
constraints are not imposed on an already determined trajectory.
The verified right-limit charge and rate replace the corresponding
candidate history slots, so the trial's temporary companion
linearization does not define accepted physical history.

## Native diode

The diode physical kernel covers the temperature-adjusted bottom
and sidewall characteristics, cubic reverse continuation, breakdown,
recombination, tunnelling and high-injection knees; depletion and
equilibrium diffusion charges; the actual series-resistance topology;
and the separately retained conservative diffusion-recovery coordinate
when VP activates it. The recombination gradient follows the physical
Newton law rather than the distinct C-compatible small-signal
conductance.

The charge slot retains depletion plus diffusion charge when recovery
is inactive. With recovery active it retains depletion alone; the
additional slot retains physical recovery charge (q=x/qpscale).
The separate state equation and terminal recovery-current coefficient
are identical to the ordinary conservative charge-state contract.

This architecture does not by itself port every transistor family, arbitrary
history-dependent or discrete device models. Other device families
must provide their own exact kernels and local storage contracts.

## Level-1 JFET

The normal and inverse Sydney channel polynomials and their analytic gate
and drain partials are expanded on the active one-sided germ. Gate
junction current and depletion charge reuse the native diode Taylor
laws, with the JFET's temperature-adjusted emission and grading.
Source/drain series conductances retain their physical prime-node
coordinates. The two local charge/rate pairs retain the normalized
per-unit-multiplicity charge expected by the ordinary JFET load; global
MNA charge includes polarity and multiplicity. This distinction matters
for P-channel devices and parallel instances. No finite differences or
AC-to-transient approximations are used.

## Gummel-Poon level-1 BJT

The multivariate local differential jet carries sparse partials with respect
to normalized BE, BC, BX, substrate, and external-base/prime-base voltages.
The same Taylor arithmetic supplies each current or charge and all of its
partial series; only absolute sparse MNA derivatives are stamped. BE
diffusion includes the Early/high-injection base charge and its BC-dependent
transit-time multiplier. The four physical charge-history pairs stay
separate. Base spreading resistance is differentiated through its
bias dependence, not frozen as in the ordinary C Newton load. On the
substrate exponential clamp the physical derivative is the derivative of
the clamped current, not the C load's retained exponential slope.
