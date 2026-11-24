# Autonomous periodic steady state

The autonomous PSS analysis finds a closed periodic orbit rather than returning the
end of an arbitrarily long transient. It shares the companion integrator, device
loading, accepted/trial history, error control and Newton machinery with transient
analysis.

## Input and output

```spice
.pss fguess stabtime oscnode [points [harmonics [sc_iter [steady_coeff]]]] [uic]
```

The positional order follows ngspice's PSS card parser. Frequency must be positive,
stabilization time nonnegative, and the phase node must be a non-ground voltage
unknown. Defaults are 256 time intervals, zero harmonic rows, 20 Newton iterations
and a dimensionless closure tolerance of 0.005. The time plot contains both endpoints
of a uniform grid; intermediate accepted integration points need not coincide with
that grid. Harmonics, if requested, include the signed DC mean and subsequent
one-sided magnitudes from the half-open periodic grid, following the C convention.
The frequency axis is in hertz. `uic` uses ordinary transient initialization for
startup. Unlike C's experimental transient-based period correction, this method
solves an autonomous shooting residual.

The Rust `AnalysisPlan::run_periodic` API returns `PeriodicOrbit` in addition to
streaming the requested plots. It exposes period, frequency, initial MNA vector,
normalized state and charge residuals, transverse contraction estimate, map count
and accepted/rejected step counts. A previous orbit can seed another solve on the
same topology and phase node; the period must remain within an eightfold ratio of
the guess. Closure and stability are checked again, rather than assuming the old
solution remains valid.

## Numerical design

Define F(x,T) as the adaptive transient endpoint after duration T from x.
The residual is (F(x,T)-x)/s, where each row scale s is derived from the
exploratory orbit and the appropriate voltage/current absolute tolerance.
A phase-node coordinate is fixed to an actual accepted state near a central
crossing. The other scaled MNA coordinates and log(T) are the unknowns of the
square system. Centered differences approximate the monodromy columns minus
identity, with the phase column replaced by the log-period sensitivity. Dense
pivoted LU in scaled coordinates solves the Newton correction; bounded backtracking
requires a decreasing residual. The phase derivative must be nonzero.

For stability, eliminate phase motion from the monodromy by subtracting the
period-derivative column times the ratio of the phase-row derivative to the
phase speed. An induced infinity norm of powers of the resulting transverse
matrix provides a sufficient contraction estimate for the *numerical* map.
Unstable or unresolved contraction is rejected. The selected node must also
exhibit actual oscillation, so a DC equilibrium is not a valid orbit.

Every map reconstructs initial charges/fluxes by an exact device load, followed
by the normal first backward-Euler step; no intermediate shooting run publishes
device history. A finer integration verifies the converged boundary independently.
Charge/flux mismatches are scaled by the excursion of the corresponding device
state across the orbit and the absolute charge tolerance. Uniform output times
are accepted integration points, not interpolation and not artificial source
breakpoints. Earlier same-direction phase crossings are checked against the
complete scaled state to avoid unnecessarily returning multiple periods.

Phase-zero device observations are not DC-bias asks. The verified map retains
the final pre-acceptance trial, accepted history, integration coefficients and
forcing; the emission pass replays that exact transient observation at the
periodic boundary. Only selected quantities are requested, and probes never
evaluate optional observations. This preserves nonzero capacitor currents and
other derivative-dependent quantities at both endpoints without changing the
shooting residual or retaining an extra waveform.

## Scope and resource limits

Explicit time dependence, delay-history devices, and hysteretic switches are
rejected: their full state cannot be reconstructed from an instantaneous MNA
vector. This is a bounded dense single-shooting implementation (2–64 MNA
unknowns, 16–4096 output intervals, at most 100 Newton iterations per resolution).
It is not a general RF periodic small-signal or noise solver. The aggregate
transient step budget covers startup, shooting probes, refinement and output.
Intermediate maps do not retain waveforms or evaluate unused observations.
Only a bounded exploratory or verification trace is retained. A time-only
result streams; harmonic calculation bounds its retained grid and DFT work.

## Regression references

`tests/periodic_steady_state.rs` checks a radial oscillator of known amplitude
and frequency, a Van der Pol relaxation oscillator, a nonlinear RF LC oscillator,
warm starts, multiple-cycle period guesses, phase-zero capacitor currents and
their harmonics, selective observation streaming, and invalid forcing/absent
cycles.
An opt-in external C reference compares the relaxation and LC results against
long C transient runs using the same decks and physical tolerances:

```sh
NGSPICE_BIN=/absolute/path/to/ngspice cargo test --locked --test periodic_steady_state -- --ignored
```
