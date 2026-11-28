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
