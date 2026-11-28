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
