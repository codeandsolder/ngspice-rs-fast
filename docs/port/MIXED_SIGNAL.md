# Native event-driven mixed-signal execution

The native event network is separate from the analog MNA unknown space.
A model implements the typed `CodeModel` interface with cloneable per-instance
state and a port schema separating differential analog voltage and digital
event domains. Models receive resolved logic and an event context, and return output
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
separate from analog transient statistics. Before Newton, the driver previews
queued events transactionally until the first settled DAC target change, or
the proposed step endpoint. Unrelated digital events do not force analog
steps. An accepted preview is reused rather than evaluated again. DAC slew
times describe a full-scale
transition, including when a new target interrupts a partial transition.

The implementation is a native typed execution interface, not a C plugin
loader. Legacy C code models rely on C runtime callbacks, mutable state
storage, and host integration conventions, not just a pure numerical function
signature. Loading them directly would require an additional unsafe ABI and
runtime isolation layer with its own state and rollback semantics. Keep that
outside the native execution interface; prefer translating supported model
equations into typed Rust kernels. Behavioral references: upstream XSPICE `evt/evtiter.c`,
`evt/evtqueue.c`, `evt/evtbackup.c`, and the code models
`d_and`, `d_dff`, `adc_bridge`, and `dac_bridge`. The API currently
provides combinational gates, a rising-edge D flip-flop (D and clock, Q and complement; no asynchronous
set/reset), comparator/Schmitt
ADC sampling and voltage DAC ramps. Native analog voltage equations provide
gain, one-pole low-pass and unsaturated integrator kernels. They reuse the
companion charge-history and truncation machinery, expose the affine AC pencil,
and explicitly declare no intrinsic noise. Integrators initialize their
output at the supplied bias value; their AC pencil uses the dynamic equation.
Arbitrary nonlinear analog model kernels, A-instance parsing, additional
sequential models, mixed-signal UIC, periodic shooting, and mixed-signal AC and
noise are not implemented by this interface.
ADC crossing detection is based on accepted and trial endpoint samples;
nonmonotone crossings wholly within one analog step require a suitable analog
maximum timestep.

## Numerical verification

The native tests exercise simultaneous drive resolution, transactional rollback
and stale cancellation handles, inertial and transport delays, bounded delta
cycles, ADC threshold boundaries, DAC slew, a closed-loop RC Schmitt oscillator,
and an ADC/DFF/DAC sequential circuit with unknown clock and data states.
A constant-DAC circuit with 10,000 unrelated digital events has exactly the
same accepted time grid and analog work counters as the event-free circuit.
Gain, filter and integrator tests check both transient history and complex AC
response against their analytical equations.

The opt-in XSPICE comparison tests in `tests/digital_events.rs` run an
independent ngspice binary. On ngspice 47, a 100 µs RC Schmitt oscillator with
1 µs combined propagation delay and 1 µs full-scale DAC slew had seven
transition midpoints agreeing within 128 ns (native maximum step 500 ns,
reference maximum step 50 ns). A sequential ADC/DFF/DAC circuit agreed on
all tested settled levels and transition midpoints within 15 ns (reference
maximum step 5 ns). These tolerances include the reference's analog-step
threshold sampling error; they do not require identical internal time grids.

Run the native tests with `cargo +nightly test --locked --test digital_events`.
For the live reference, set `NGSPICE_BIN` to an XSPICE-enabled ngspice binary
and run the same command with `-- --ignored`.
