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

Devices declare each local charge/charge-rate history pair together with
its electrical node incidence and signed scale. The prepared circuit
binds them to distinct absolute slot pairs.
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

For instantaneous physical terminal observations, the isolated static
device load contains no integration companion. Reconstruct displacement
current by applying each declared electrical incidence and scale to the
solved local charge rate, not to the temporary observation scratch state.
Only external rows receive these corrections: the static current at an
external series-resistance terminal already includes the internal
displacement current through solved KCL. Ground is the negative sum of
distinct external rows. This works independently of device multiplicity,
polarity and recovery-coordinate normalization. Behavioural branch-event
values remain the solved candidate values rather than a scalar
re-evaluation at an isolated branch boundary.

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

## Boundary of memoryless storage

A device's AC capacitance is not sufficient to specify its physical charge.
In the classic MOS Meyer model, gate-source capacitance depends on both
junction voltages while the branch charge increment uses only the
source-junction voltage change. A voltage-only gate-source charge would
therefore require incompatible mixed partials in the general case. MESFET
and JFET2 partition a Statz gate charge increment using both the current
and accepted junction voltages; JFET2 additionally retains delayed gate
voltages and averaged power. HFET incremental charge updates likewise
retain accepted charge and voltage. These models cannot in general be
converted to memoryless F/Q kernels by integrating the AC capacitance at
one bias or freezing a previous state.

History-dependent models require their own physical state coordinates or
rate equations and independent validation of their continuous limits
([#253](https://github.com/codeandsolder/ngspice-rs-fast/issues/253)). Keep
an explicit unsupported diagnostic in the memoryless path until such an
implementation is available. Model variants that are genuinely memoryless
may be qualified on their own exact equations, not an approximate removal
of their history terms.

Prepared compact models must expose the same exact Taylor equations at the
producer-owned kernel boundary. Scalar residual and Jacobian values do
not determine higher derivatives. The ABI work is tracked in
[preprocess #79](https://github.com/codeandsolder/ngspice-rs-preprocess/issues/79).

Conservative BSIM charge modes and their exact multivariate kernels are
tracked in [#256](https://github.com/codeandsolder/ngspice-rs-fast/issues/256).
Continuous switch and prepared PWL-resistor branch laws are tracked in
[#257](https://github.com/codeandsolder/ngspice-rs-fast/issues/257).
These are separate implementation and qualification obligations, not
claims that their current scalar models support high-index derivatives.
