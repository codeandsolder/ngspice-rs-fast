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
never both. Its present binary strong-output kernel uses the LH and HL
entries; the event engine itself supports all six paths. A voltage DAC accepts
only rise/fall slew times, not impedance propagation delays. This is the
foundation for disconnectable output bindings, not a complete implementation
of source-level three-state gates, sequential timing checks, timing corners,
or analog drive/load/power behavior.

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
power-referenced drive bridge requires its own electrical binding. Timing
checks and source-dependent path delay windows remain separate work.
