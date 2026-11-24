# Reproducible transient stochastic sources

Independent voltage/current sources support `TRNOISE` and `TRRANDOM` in
transient analyses. Their noise is a **time-dependent physical excitation**;
the independent devices remain noiseless generators for the linearized
spectral `.noise` analysis, which is a different simulation.

This implementation emphasizes reproducible physical-time processes rather
than dependence on the number/order of Newton solves or rejected timesteps.

## Supported waveforms and units

| Input syntax | Physical parameters |
| --- | --- |
| `TRNOISE(NA TS [NALPHA [NAMP [RTSAM [RTSCAPT [RTSEMT]]]]])` | Gaussian white-knot standard deviation `NA` in V or A; spacing `TS` in s; colored fractional exponent `NALPHA` in (0,2); Gaussian innovation amplitude `NAMP` in V or A; telegraph pulse amplitude `RTSAM` in V or A; mean capture and emission dwell times `RTSCAPT`, `RTSEMT` in s |
| `TRRANDOM(TYPE TS [TD [PARAM1 [PARAM2]]])` | Distribution 1=uniform, 2=Gaussian, 3=exponential, 4=Poisson; fixed sample spacing `TS` in s, first-draw delay `TD` in s, distribution `PARAM1` (width/sigma/mean/lambda), `PARAM2` (offset/mean) |

The trailing optional parameters take upstream defaults. As in
`VSRCparam`/`ISRCparam`, supplying `NAMP` with zero `NALPHA` does not
activate the colored generator; the original setter remains serialized.
For `TRRANDOM`,
`PARAM1` defaults to 1, `PARAM2` to 0, and `TD` to 0. Its
type-1 distribution is *centered*:
`PARAM2 + PARAM1 * U(-1,1)`, exactly as ngspice's `drand()` behaves.
Types 2, 3, and 4 are Gaussian `N(PARAM2, PARAM1²)`, exponential with
mean `PARAM1` plus `PARAM2`, and Poisson with mean `PARAM1` plus
`PARAM2`. A nonzero explicit `DC` adds a constant only in transient
excitation, while OP/DC uses the independent-source DC semantics.

The nominal Gaussian `TRNOISE` sample at time zero is zero (as in C).
Subsequent independent Gaussian knots are indexed by the *absolute* integer
physical grid coordinate. Between adjacent knots values interpolate linearly.
The resulting process is bandwidth-limited: for white noise with knot
sigma `NA` the **two-sided** physical power spectral density is

\[
S(f)=NA^2\,TS\,\operatorname{sinc}^4(\pi f TS),
\qquad E[X(t)^2]_{\text{phase}}=\tfrac23 NA^2.
\]

`TRRANDOM` uses constant held values between updates, not interpolation.
The first drawn value is held from `TD` until the next grid event. A
one-sided limit at an update selects the preceding or following sample
without consuming any random generator state.

Colored noise is a bounded-memory stochastic approximation to ngspice's
Kasdin fractional-difference filter. It derives h[k] recursively by
`h[k] = h[k-1] (NALPHA/2+k-1)/k` and groups the **squared coefficients**
for k=1..65535 into 16 octave bands. The k=0 innovation stays separate,
with NAMP as the input Gaussian drive standard deviation, **not** a fixed
output RMS. Those octave bands reproduce the target power per logarithmic
frequency range with low-frequency cutoff around
`1/(65536 TS)`. The runtime never allocates the C filter's entire
time-domain FFT data set. Exact bitstreams and finite-window end effects
are deliberately different, so compare C trajectories statistically.

The RTS component alternates between zero and `RTSAM`. Capture and
emission dwell times are independent exponential distributions with
means `RTSCAPT` and `RTSEMT`; stationary high-state probability is
`RTSEMT/(RTSCAPT+RTSEMT)`. The physically timed RTS transitions are
independent of the sample grid (a pure RTS is valid with `TS=0`).
`src/devices/noise/transient.rs` contains the reference white PSD,
integrated power and RTS covariance formulas.

## Determinism and memory

`src/devices/stochastic.rs` computes random draws using an explicit
integer mixing function of the seed, stable case-insensitive elaborated
source identity, physical sample index and independent stream type.
No global/thread-local RNG is read. Re-evaluating the same absolute
physical time is bitwise deterministic even after speculative Newton
trials, rejected/truncated timesteps, backtracking or independent source
evaluations. The default seed is fixed; callers may select a different
stream with `StochasticWaveform::with_seed`.

Optimizer `CandidateWorker` automatically incorporates the immutable caller
`CandidateJob.id` into the source's stream identity **after** validated
candidate preparation and before any scenario runs. The exact same candidate
ID repeats a bitwise-identical physical process across workers and A/B/A
execution; different IDs obtain independent random trajectories even when
all numerical parameter coordinates are identical. The finalized topology
stores only the ordinals of stochastic independent sources, so deterministic
candidate workloads do not scan device lists or clone any source waveforms.
Candidate replay is state-independent, and changing a source's candidate ID
replaces its bound waveform without invalidating the MNA topology or the
numeric parameter plan.

Only a 32-event **derived** RTS timestamp cache is kept for fast sequential
access; discarding/cloning it leaves every physical event unchanged. The
sample grids and merged breakpoint stream are lazy; no array of the
complete run's noise or event history is generated. The accept-phase
scheduler asks for the next source event, allowing timestep placement
at sample knots and exact RTS jumps.

RMS parameters must be finite and nonnegative, sample spacing positive
whenever white/colored noise is enabled, and active telegraph means
strictly positive. Invalid or unresolvable binary64 physical grid
coordinates are errors, not wrapped indices.

## Verification

The opt-in fixed-seed regression suite `tests/transient_noise.rs`
compares Gaussian knot/midpoint variance, all four TRRANDOM
mean/variance laws, RTS dwell times, seed/source isolation,
one-sided event behavior, physical optimizer candidate A/B/A and
one-/multi-worker dispatch equivalence, complete deck parse/write/simulate
results, current-source held-random voltage against an independent resistor
law, and exact RTS event landing under adaptive transient timesteps. The unit tests in `src/devices/stochastic.rs` exercise
candidate-free replay and rejected-step lookup order; the tests in
`src/devices/noise/transient.rs` check the analytic power and
autocovariance formulas.

Live C reference decks and measurements are recorded in the
[issue #182 implementation log](https://app.notion.com/p/421f74be902081d6b189e7bc30a9f79a).
A C ngspice 47 run with `TRRANDOM(1 1u 0 2 -1)` returned mean about
-1.017 and variance 1.349 over 8 ms, matching centered uniform
theory (-1, 4/3); a RTS reference with mean capture 10 µs and mean
emission 20 µs spent 67.3% of the sampled time in the high state,
matching 2/3. Stochastic stream bytes are intentionally independent
from C's global RNG and FFT implementation.

The normal no-noise constant/PULSE/PWL/SIN waveform path remains
unchanged and does not construct stochastic state.
