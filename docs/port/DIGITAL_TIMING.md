# Directional digital propagation

The event queue selects propagation delay from the actual old and requested
`Drive`, not only its logic value. `Delays::paths` takes six finite,
nonnegative times (seconds), in the order LH, HL, LZ, HZ, ZL, ZH. Zero is
an explicit instantaneous transition. Inertial scheduling selects from the
committed drive and transport scheduling from the last requested drive.
The ordinary two-state XSPICE rise/fall rule is unchanged. For explicit
high-impedance paths with an unknown logic endpoint the generic deterministic
policy selects the later of the two possible known transitions. This is not
a min/max timing-window implementation or a dialect-specific timing law.

The `preprocess_logic_event` model accepts either `delay` or `delay_paths`,
never both. The source-bound tri-state kernel and the event engine use all six paths. A voltage DAC accepts
only rise/fall slew times, not impedance propagation delays. This is a deterministic contract, not an interval timing-corner simulation or physical analog drive/load/power model.

## Source-level drive and sequential state

The input vector accepts analog receivers and explicit digital terminals
(`%d`) in source order. A digital input uses the resolved bus directly,
without an analog threshold or an invented voltage. Digital terminal
inversion applies after resolution. Analog receiver hints and thresholds
apply only to the analog inputs.

Operations 17–20 select dual-edge SR/JK/D/T storage; 21–22 select active
high/low tri-state buffers (data then enable); 23–26 are XOR, XNOR,
buffer and inverter. `strength=1` is resistive, `strength=2` is strong.
Disabled tri-state output disconnects; an unknown enable is represented
by an uncertain connected drive, since the current three-valued engine
cannot represent a conditional drive. `ic=2` initializes storage to X.
The optional `async_policy=[preset_active clear_active both_result]`
adds preset and clear after the clock port. Active levels are 0 or 1,
simultaneous result 0, 1 or 2 (unknown). Async controls dominate capture;
clock history advances while they are asserted. Ambiguous clock transitions
and uncertain set/reset/enable inputs retain known results when every
possible binary outcome agrees.

These are explicit generic primitives. The ideal voltage bridge remains a
logic observation/conversion (X or a completely undriven net maps to the
midpoint). It is not a physical tri-state analog output; a finite-resistance,
power-referenced drive bridge requires its own electrical binding. Source-dependent path delay windows remain separate work.


## Sequential timing constraints

Optional vectors, in seconds, specify a deterministic contract over resolved
input transitions. `timing_setup` and `timing_hold` each contain one interval
per data port. `timing_clock_min=[low high]` specifies minimum observed clock
or gate widths. `timing_async_min=[preset clear]` specifies minimum asserted
widths, and `timing_recovery=[preset clear]` specifies release-to-capture
intervals. A transparent latch checks setup/hold at its closing edge. Initial
logic levels establish state but do not claim a transition prior to time zero.
Exact deadline equality passes, including explicitly disabled zero intervals.
Known asserted asynchronous controls supersede synchronous captures.

`timing_violation=0` (default when a constraint is declared) prevents
acceptance of the offending candidate, not its speculative evaluation; `1` emits a diagnostic and invalidates affected storage to X;
`2` reports a diagnostic without changing the ideal logical result. Diagnostic
records carry the flattened instance name, physical time, delta, input port,
observed interval and required interval. Only accepted event transactions
publish them; analog Newton retries cannot duplicate warnings. Successful
mixed analyses expose them through `Circuit::last_digital_diagnostics` and
batch/control output. These policies are generic: a source dialect may select
one only after its own timing semantics have been independently established.

## Physical output conductance

`a1 q %gd(out supply) drive` with
`.model drive preprocess_logic_conductance(conductance=[...])` selects
one of seven nonnegative conductances in siemens from the resolved drive:
strong L/H/X, resistive L/H/X, then Z. Inversion exchanges L and H within
each strength; the X and Z entries remain unchanged. Zero is exactly open
circuit. The branch stamps conductance between the specified physical
terminals, not an ideal source or a ground-referenced voltage. Two such
branches to real supply rails implement a powered push-pull stage with
independent resistances. Ordinary R/C devices supply input loading,
leakage and storage; supply currents follow Kirchhoff's laws. Accepted
strength-only transitions are electrical breakpoints when their conductance
differs. No DAC approximation or implicit pull resistor is used.

## Causal propagation paths

`delay_paths=[LH HL]` is the binary two-path form; the six-entry form also
binds the four impedance paths. Clocked storage interprets the primary
pair as clock-to-output; transparent storage interprets it as data-to-output.
`delay_gate=[LH HL]` selects gate-to-output, and `delay_preset` and
`delay_clear` select the respective asynchronous control paths. Omitted
causal paths inherit the primary pair; explicit zero does not inherit.
Transparent latches also accept `async_policy` and the trailing preset/clear
ports. The native model selects the cause of the requested output change
before applying inertial/transport scheduling, so unrelated input evaluations
cannot replace an outstanding causal deadline. Simultaneous ambiguous causes
select the latest participating path in this deterministic contract, with
known active asynchronous controls dominating a normal capture. It does not
assert a particular source vendor's simultaneous-event policy.

## Ordinal drive and digital fanout loading

`strength_paths=[L H X]` selects explicit 0–63 drive ranks by output logic
state. Rank 0 disconnects without erasing the model's requested logical
state, enabling open-drain outputs. `override_margin` (1–63, default 1)
is the minimum ordinal difference needed to override a conflicting
connected driver; declarations on one output net must agree. Resolution
compares the maximum rank of each possible state against the strongest,
independently of iteration order. `strength=1` and `strength=2` designate
the lowest connected and forcing ranks; they cannot be combined with
`strength_paths`. The seven-value conductance bridge explicitly groups
ranks 1–62 in the resistive class and rank 63 in the forcing class.
The engine does not infer a resistance from an ordinal or choose a
source dialect's quantization/overdrive conversion.

`input_capacitance=[...]` declares one digital input capacitance per
terminal; `output_capacitance` declares output capacitance. The
flattened topology sums every attached digital terminal occurrence once,
including multiple pins on one net and inactive output drivers. The
optional `load_resistance=[R_L R_H R_Z]` requests additive delay
`R_target * C_total * ln(2)` on each nonzero propagation path, including
causal paths. All values are finite and nonnegative. An explicitly
zero propagation path remains zero. Analog receiver capacitance belongs
in physical circuit elements, not the digital timing approximation.
This is a fixed-topology fanout loading model; it does not replace
analog RC dynamics or infer source-specific converter switching offsets.

## Causal timing-window algebra

The independent `analysis::digital::timing` layer represents propagation
intervals as persistent arrival expressions, with explicitly identified
uncertainty origins. Fanout shares ancestry. Reconvergent relative-time
comparisons cancel shared delays before bounding independent uncertainty;
earliest/latest joins preserve all input expressions. Affine comparisons
are exact in the mathematical model; joins provide enclosures, not
necessarily tight bounds. Comparison work is explicitly bounded, and
deep expressions are released iteratively.

For example, a common arrival in [15,30] followed by data delay [10,30]
and clock delay [40,60] gives data arrival [25,60] and clock arrival
[55,90]. Independent absolute comparisons would permit a negative
setup interval; the shared-origin comparison gives [10,50].

The source-bound causal scheduler transports this provenance through
bounded and exact transport paths. Uncertain rising/falling transitions
retain their original arrival expression, rather than becoming an
ordinary unknown logic drive. Optional hazards and revisions of an
already announced conditional waveform require the guarded representation
described below.

## Pulse qualification and directional ambiguity

The deterministic event kernel distinguishes stable L/H, X, and the two
directional uncertainty phases R/F. Gates propagate the direction when the
known pre-phase and post-phase Boolean results permit it; a controlling
stable input still suppresses ambiguity. Clocked storage retains possible
leading/trailing capture alternatives, rather than treating an R/F marker
as a definite clock edge. A pulse that may have toggled a dual-edge T
flip-flop twice can restore its original known value when it ends.

An optional `pulse_reject` time selects a transactional pulse policy
instead of the explicit `inertial` selector. If a desired output changes
again before its propagation deadline, an excursion shorter than the
rejection threshold is suppressed. A nonzero excursion at or above the
threshold emits R or F at its original propagation deadline when its
connected binary endpoints establish a direction, otherwise X. A
transition persisting through the deadline is definite (including exact
deadline equality). A later earlier-due transition supersedes pending
deliveries under the usual queue ordering law. No event time is moved
backward and trial rollback includes every pending pulse. This implements
the deterministic short-pulse distinction illustrated by the PSpice user
guide, chapter 7, but does not claim complete worst-case timing semantics
for arbitrary correlated overlapping windows.

Ideal voltage observations and the four-state HDL interface explicitly
project R/F to unknown at their representation boundary. A physical
conductance bridge does not silently do so: extend the existing
seven-entry conductance table with four additional entries (strong R,
strong F, resistive R, resistive F) to bind directional states. Missing
R/F entries diagnose a directional phase at runtime. Inverted terminals
exchange R and F as well as L and H.

## Ordered differential-voltage receivers

A source-bound `preprocess_logic_range` instance connects one analog
voltage (single-ended or differential) to one digital net. Its `ranges`
vector contains ordered groups of four finite numbers: inclusive minimum
voltage, inclusive maximum voltage, logic code (0=L, 1=H, 2=X, 3=R,
4=F) and explicit ordinal strength. A zero strength is disconnected.
Range order is retained, including overlapping intervals and duplicate
logic labels. `initial_range` supplies the bias scan origin (default 0).
A sample contained in the current range does not switch; otherwise the
search proceeds cyclically from that range.

A gap uses one explicit policy: `gap_policy=0` rejects a final candidate
(default); `gap_policy=1` selects the nearest interval (ties use the cyclic
scan order); `gap_drive=[logic rank]` selects an explicit fallback and
retains the preceding scan origin. Gap entry or nearest-index changes
generate typed accepted diagnostics, with voltage and selected range,
rather than warning at every analog sample. During mixed bias settlement
gap diagnostics are provisional until the final bias is accepted. Range
identity changes trigger crossing localization even if the output drive
does not change. The index is held per occurrence and committed together
with the converged analog/event point. Rejected trials cannot change it.

The generic ordered-range rule corresponds to the cyclic search described
in the PSpice reference guide (digital/analog interface devices). Initial
index and nearest tie behavior are explicit generic choices pending
source-specific qualification. Converter input resistance and capacitance
are ordinary physical elements; this receiver does not replace them with
a digital approximation.

## Cumulative arrival overlap

A causal transport driver retains every active arrival and the most
recent guaranteed delivery. Overlapping envelopes may expose an X phase;
when the earlier transition completes the remaining envelope can recover
its directional state. A later completion supersedes an earlier pending
completion, preventing stale events from reviving an obsolete level.
The latest causal transition remains available at fanout even while
multiple arrival envelopes overlap. Accepted diagnostics report both
absolute intervals and the relative separation after common-origin
cancellation. Absolute overlap alone does not prove that the physical
pulse order can reverse. The current causal transport contract requires
consistent announced endpoint order; reversed earliest-boundary order
and independent multi-driver waveform convergence need further
qualification and are diagnosed instead of treated as exact transitions.

## Physical electrical realization

A conductance bridge can use `conductance_by_rank` instead of the
explicit grouped `conductance` table. Supply Z first, followed by
L/H/X/R/F conductances in siemens for each connected rank in ascending
order. Rank-specific tables are shared by occurrences, with inversion
applied at each port. A strength rank is not itself a resistance law.

An optional `resistance_duration=[L H X R F Z]` vector in seconds and
`resistance_shape` (positive, dimensionless duration/time-constant
ratio) describe a finite exponential transition in resistance, not
conductance. Both are required when the law is selected. For duration
D and shape s, at normalized time u=(t-t0)/D the resistance is

```text
w = (1 - exp(-s*u)) / (1 - exp(-s))
R(t) = (1-w)*R(t0) + w*Rtarget
G(t) = 1/R(t)
```

The transition begins with the actual resistance at the accepted event,
including an interrupted ramp, and reaches the target exactly at t0+D.
A zero duration selects an immediate change. All table entries must
represent finite positive resistances if any nonzero resistance duration
is requested; an ideal open is not a finite resistance. The plain
conductance table continues to support exact open branches. This is a
generic explicit law, not an inferred normalization for any source
dialect's switching-time field.

To represent charge storage, elaborate each driver onto its own digital
net and connect the corresponding physical rail branches to the common
analog node. Place the actual capacitances and leakage branches on that
node, then use an ordered voltage receiver to obtain the digital value
where needed. Resolving ordinal strengths first would discard parallel
drive paths, and storing the last logical value would discard leakage.
The physical RC network retains charge independently of the digital
event history.

For bounded transport, causally ordered input transitions are evaluated
in their provable physical order even when their absolute timing windows
overlap. Equal arrival expressions are evaluated together. The output
may therefore contain several arrivals; preserving each expression
retains common-origin correlation in downstream timing checks. This
applies to non-monotone gates and tri-state enables as well as binary
gates. Where no total order can be proven, monotone earliest/latest
joins remain available; conditional waveforms and revision of previously
announced arrivals are not silently replaced by scalar timing.

For clocked storage, `delay_preset_windows` and `delay_clear_windows`
are independent bounded propagation tables. As with `delay_windows`,
each takes four directional bounds (rise min/max, fall min/max) or
twelve bounds for all six paths. An omitted primary bounded table uses
the explicitly declared exact primary delay. Exact secondary path tables
accept two or six values; each secondary path has exactly one declaration.
When both asynchronous controls assert at the same causal instant, their
two propagation expressions are joined by latest arrival. Resistive
loading applies independently to each declared path.

An ordinary transport path downstream of bounded timing carries the
incoming causal expression through its exact delay. Zero-delay inertial
paths also have no nonzero rejection interval and can carry that
expression. Reachability is computed over the flattened digital topology,
independently of device declaration order, so unrelated scalar components
keep their existing event path. Exact propagation preserves the scalar
law's actual X and impedance-transition selections, rather than inventing
an additional local interval. Nonzero inertial and explicit pulse-reject
policies require their own causal cancellation semantics and are not
silently converted into transport.


## Level-sensitive storage and conditional ordering

Enabled storage and SR latches use the same logical-state transition
kernel as deterministic storage, with bounded primary, gate, preset and
clear propagation paths. Transparent storage captures its closing edge
for setup, hold and recovery checks; independent gate pulse-width and
asynchronous pulse-width checks share the correlation-aware sequential
constraint implementation. Rank-only changes update the electrical
endpoint identity without discarding the preceding logical edge.

An order partition retains each physically distinct ordering, including
simultaneous groups, as a conjunction of exact arrival comparisons.
Relative enclosures and strict-order-cycle detection eliminate simple
contradictions; exact integer affine feasibility then establishes a
jointly feasible origin assignment for each retained case. For a concrete joint assignment,
exact weighted-product sums evaluate min/max and shared affine paths,
so one and only one order case is selected without sub-ULP time-order
loss. The same native reducer evaluates each case's logical output
request sequence. A later-posted arrival suppresses older events that
would otherwise occur after it; same-time announcement delta groups
retain the explicit native ordering convention.

An unconstrained latch retains the unresolved input suffix and its
published logical requests. If every guarded case produces the same
state and request sequence, the common trace can be emitted or extended
without an artificial uncertainty state. A settled prefix can be
compacted even when its absolute envelopes overlap the remaining suffix,
provided relative comparisons prove the physical order. If branches
differ or an already-published request requires revision, conditional
delivery is required. The branch representation preserves optional
pulses; the event engine still needs guarded delivery and conditional sequential
timing checks. These are not silently replaced by deterministic paths.

### Exact joint feasibility and observation projection

Order cases are pruned by exact affine feasibility, rather than only
pairwise relative enclosures. Binary64 constants become integers in
units of the smallest subnormal. Shared origins are real variables
bounded by [0,1]. Integer Fourier–Motzkin elimination preserves strict
and non-strict inequalities without epsilon tests. A min/max expression
uses one selected child and its defining inequalities; shared nodes
reuse that selector. A work budget bounds enumeration and integer
arithmetic, with explicit failure on exhaustion.

For an observation at an unspecified time, retain time as one variable
and eliminate every delay origin. This projects each guarded branch
onto the representable physical-time axis, including interior endpoints
created by correlated constraints (not just unconditional arrival
bounds). Strict rational endpoints are converted using exact integer
comparisons to the first or last satisfying binary64 observation.
Intervals are merged only when adjacent on that lattice. Feasibility
over real origins and exact projection onto representable observation
times are distinct operations.

Guarded transport traces retain posting order separately from physical
arrival order. At a concrete observation, the newest posted request
that has arrived determines the drive; an older late request cannot
resurrect an obsolete value. The guard partition composes through exact
conjunction, so two reconverging copies of an uncertainty do not
generate independent alternatives. Projection of a trace considers the
selected request delivered and every later-posted request undelivered;
the initial drive requires all requests undelivered. An interval sweep
produces the exact feasible drive set at every representable observation.
The symbolic branches remain available to downstream model evaluation;
the feasible drive set is an observation, not a substitute for them.
