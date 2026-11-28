# Prepared independent-source waveforms

Prepared artifacts can carry typed waveform declarations alongside the emitted
electrical deck. Parse the artifact (including its companion files) and call
`RunConfig::prepared_circuit` with the complete `PreparedDeclarations` value. The circuit uses the
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
optional startup holds are preserved. A finite repeating train has an exact
absolute cutoff, with independently specified held value after termination.
Analytic sine forcing uses signed frequency, amplitude, delay, damping and phase
without transient-plan defaults. A finite sine train holds its capped endpoint;
a negative elapsed cap becomes effective when the startup hold releases.
Exponential forcing evaluates both signed transition delays independently, even
when the fall precedes the rise. The DC bias is independent of the transient
startup. Prepared transient initialization uses the physical right-hand limit;
native sources retain their existing left-limit startup convention.

Exact prepared-source events are enumerated lazily. The companion driver
previews them from the last accepted time, outside the native breakpoint
coalescing table. An event-ending step uses the left limit; the next step
restarts at first integration order. Rejected trials cannot consume events.
An unrepresentable periodic index or knot spacing is an error, not an
approximation or an implicit merge.

For the physical derivative-array solver, the same sided interpolation
interval supplies the analytic, time-normalized Taylor coefficients: the
zeroth coefficient is the external limit, the first is the scaled slope,
and higher coefficients vanish on each linear segment. Damped sine
coefficients follow the analytic envelope and phase; a held sine has a
constant left germ at its activation boundary. These coefficients do not
differentiate rounded samples. The derivative solver uses its merged exact
event iterator rather than a second prepared-source scheduler. At an
interior event it establishes the right-limit state while retaining physical
storage and rejects derivative-input jumps that require impulses. The
requested final endpoint retains the left limit.

Native PULSE/PWL and native source parsing retain their own dialect semantics.
The producer binds DC and transient startup independently. Periodic
operating-point metadata is retained in the resolved forcing; dispatch of the
source-specific periodic analysis policy is separate from ordinary transient
execution.
