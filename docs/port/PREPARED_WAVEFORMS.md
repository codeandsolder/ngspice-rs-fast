# Prepared independent-source waveforms

Prepared artifacts can carry typed waveform declarations alongside the emitted
electrical deck. Parse the artifact (including its companion files) and call
`RunConfig::prepared_circuit` with the declaration slice. The circuit uses the
shared producer contract, not a second parser for source-dialect syntax.

The declaration is identified by artifact filename, subcircuit definition and
source name. During lexical instance expansion, resolve its expressions using
that occurrence's parameter scope and the common evaluation budget. Unused
definitions need not be instantiated, but every declaration must identify a
unique independent source in the parsed deck. Duplicate and unmatched
declarations are errors. A rebuilt circuit resolves the declarations anew;
`Circuit::from_prepared_netlist_with_limits` also accepts an explicit root
parameter scope. The electrical source's identity, terminals, multiplier and
AC phasor remain unchanged.

The resolved law stores unique knots with distinct external left and right
ordinates. It interpolates the right value of one knot to the left value of the
next, including across repeating-cycle boundaries. Signed cycle origins and
optional startup holds are preserved. Analytic sine forcing uses its signed
delay and damping without transient-plan defaults. The initial operating point
uses the prepared right-hand startup value; native sources retain their existing
left-limit transient startup convention.

Exact prepared-source events are enumerated lazily. The companion driver
previews them from the last accepted time, outside the native breakpoint
coalescing table. An event-ending step uses the left limit; the next step
restarts at first integration order. Rejected trials cannot consume events.
An unrepresentable periodic index or knot spacing is an error, not an
approximation or an implicit merge.

Native PULSE/PWL and native source parsing remain unchanged. The currently
pinned producer contract defines DC as the right-hand startup value. Periodic
operating-point metadata is retained in the resolved forcing; dispatch of the
source-specific periodic analysis policy is separate from ordinary transient
execution.
