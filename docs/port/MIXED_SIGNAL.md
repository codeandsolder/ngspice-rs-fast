# Native event-driven mixed-signal execution

The native event network is separate from the analog MNA unknown space.
A model implements the typed `CodeModel` interface with cloneable per-instance
state. Models receive resolved logic and an event context, and return output
drives and propagation policies. The network resolves all simultaneous drivers
before evaluating affected models; insertion sequence disambiguates same-driver
updates. Physical time, delta cycle, and insertion sequence order the queue.
Strong drives dominate resistive drives; equal-strength disagreement resolves to
unknown. Undriven nets are high impedance.

An `Engine` transaction owns its event queue, cancellation index, resolved nets,
model states, and work counters. Dropping the transaction discards all trial
changes. Transport output scheduling supersedes later queued transitions on
the same output; inertial scheduling additionally cancels pending pulses.
Event and delta-cycle limits report an error instead of silently truncating
oscillation. Initial event evaluation is at time zero without propagation delay.
External event identifiers cannot alias events created after rollback.

`mixed_signal_transient` binds differential ADC inputs and differential ideal
voltage DAC outputs to the ordinary companion transient driver. Initialization
settles analog bias and digital outputs before committing history. Candidate ADC
crossings shorten the proposed step and force a fresh analog solve. A DAC ramp
is read-only during Newton iteration; event state and new ramps commit only
after the analog point is accepted. A rejected trial cannot consume events,
change digital state or advance the output ramp. Digital work counters are
separate from analog transient statistics. DAC slew times describe a full-scale
transition, including when a new target interrupts a partial transition.

The implementation is a native typed execution interface, not a C plugin
loader. Behavioral references: upstream XSPICE `evt/evtiter.c`,
`evt/evtqueue.c`, `evt/evtbackup.c`, and the code models
`d_and`, `d_dff`, `adc_bridge`, and `dac_bridge`. The API currently
provides combinational gates, a rising-edge D flip-flop, comparator/Schmitt
ADC sampling and voltage DAC ramps. Arbitrary analog code model kernels,
A-instance parsing, additional sequential models, mixed-signal UIC,
periodic shooting, AC and noise analysis are not implemented by this interface.
ADC crossing detection is based on accepted and trial endpoint samples;
nonmonotone crossings wholly within one analog step require a suitable analog
maximum timestep.
