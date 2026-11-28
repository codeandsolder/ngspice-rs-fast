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
