# Independent timed-switch references

Generated against ngspice-47 and the public producer's XSPICE runtime
(`ngspice-rs-preprocess` pinned by the solver at
`fd55b8255bf2a49f3f2fe784b718e512c66340fa`; runtime source unchanged
from `46cb5924387c5ed92cb03376453bd6140103f282`). The simulator and
module are independent of this native Rust kernel. Set up the matching
ngspice binary and load the producer's `ngspice_preprocess_logic.cm`
in the working directory's `.spiceinit`. The native engine never loads
that module.

* `pulse-matrix.cir`: 16 switches with 1 V branch bias relative to a
  nonzero reference, 8 ohm independent input loads, and asymmetric ramp
  and propagation times. Run `tran 1n 6u 0 0.25n`. Export time and
  `i(vb<node>)` for each case with 17-digit precision. Negate the
  source currents and linearly interpolate at the times in
  `pulse-samples.csv` (149 samples). Covers TIMETOL from 1 fs to 100 ns,
  a rejected pulse followed by a longer pulse, widths 499–650 ns
  around exact deadline and ramp-end equality, and reversal during fall.
* `oscillator.cir`: run `tran 1n 10u 0 1n`, export `v(c)` and
  time, retaining all points. The 20 strict local extrema after
  100 ns are in `oscillator-extrema.csv`.
* `bias-ac.cir`: run `op` and `ac lin 1 1k 1k`. Record negative
  branch- and input-source currents, both real and imaginary AC
  components, for five control voltages and both initial hints. The
  separate input load is 0.125 S, and the branch has no incremental
  control gain. See `bias-ac.csv`.

The pulse and oscillator decks use the same physical cases and numerical
tolerances as the producer's independent SIMetrix vendor qualification.
Only compact numerical witnesses are committed, not the C runtime or
the full raw trace.

## Abrupt inertial reference

`instant-inertial.cir` is an independently authored ngspice-47 XSPICE
`d_source` / `d_buffer` circuit using `inertial_delay=true`, with rise and
fall delays exactly representable in binary (2^-20 and 2^-21 seconds).
Run `ngspice -b instant-inertial.cir` in this directory; `eprint` gives
the digital event sequence. `instant-inertial.csv` records the resulting
time, input, and output levels, including the suppressed short pulse, the
exactly equal-width pulse, and the longer pulses. The equal-width pulse
commits at the coincident deadline and reverses after the fall delay.
The native tests independently reconstruct analog threshold crossings at
these timestamps. No custom code model or external dialect simulator is
required for this reference.
