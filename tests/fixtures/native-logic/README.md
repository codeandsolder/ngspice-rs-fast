# Independent native logic reference

Synthetic public circuits evaluated with ngspice 47 and the public producer's
`runtime/logic-xspice` C code model module at producer revision
`46cb5924387c5ed92cb03376453bd6140103f282`. The Rust implementation does
not load or link this module. Reference traces use a maximum step of 1 ns.

- `matrix.cir`: operations 0–6, both initial latch states and inversion
  settings, common-mode differential inputs, independent/inverted rails,
  10 ohm series resistance and 100 ohm load. 308 settled values and 83
  physical transitions. Crossing threshold is the midpoint of the two
  loaded rail voltages.
- `pulse.cir`: short and long pulses through inertial and transport models;
  independent reset-dominant latch with 1 us RC feedback. The sampled
  capacitor voltage tests the continuous analog state, not just output bits.

To regenerate, build the public producer's module using its documented
`runtime/logic-xspice/build.sh` against a configured XSPICE-enabled ngspice
source tree, using the managed compiler wrapper. In a separate scratch
working directory, add `codemodel /absolute/path/to/module.cm` to
`.spiceinit`. Copy the fixture deck, remove the final `.end`, append a
control block containing `set numdgt=17`, `set wr_singlescale`,
`tran 1n 9u 0 1n`, and `wrdata matrix.dat` followed by all 28 output vectors
`v(g000)` through `v(g611)` in operation, initial state, inversion order.
Terminate the block with `quit`, `.endc`, `.end` and run `ngspice -b` on
that deck. For `pulse.cir`, use `tran 1n 8u 0 1n` and
`wrdata pulse.dat v(o1) v(o2) v(fb) v(cap)` instead.
The CSV level rows are actual reference samples nearest the nominal times
0, 0.5, 1.5, 2.5, ..., 8.5, 9 us (matrix) and 0, 0.5, 1.3, 1.6, 2,
2.3, 2.7, 3.5, 4.2, 4.6, 5.5, 6.2, 7, 8 us (pulse). The transition
rows are the first reference samples on the opposite side of the midpoint
(2.113636363636364 V for matrix, 2.5 V for pulse). No interpolation or
smoothing of reference transitions. The native tests compare the same
settled values, direction, number and timing of edges.


The sequential matrix adds 22 configurations spanning operations 7–16 with
zero delay and transport delay, plus positive inertial delay for transparent
latches. It uses the same ngspice-47 reference and 1 ns maximum timestep,
with a 7 us stop time and 2.5 V transition midpoint. The sampled columns are
named in the CSV headers/rows; nominal times are 0, 0.5, 1.3, 1.95, 2.4,
3.3, 3.95, 4.4, 5.3, 5.95, 6.4 and 7 us. 264 sampled levels and 53
transitions. The producer's positive-inertial-delay clocked models currently
lose captured events (ngspice-rs-preprocess #78); native positive-inertial
clocked timing is tested independently against the specified edge and
propagation time, not against that known defective oracle.
