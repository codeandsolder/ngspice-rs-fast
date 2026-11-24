# Native CPL #183: passive shared-pole matrix research

2026-11-24. Standalone numerical experiments, not an integrated P/CPL
device. The campaign issue is actively held by worker arch-1122-1944-b6e9.
Earlier validated native CPL work has been preserved on the rebased GitHub
branch lint-campaign/c04a4381fab6/gpt6-acr-20261123-1723-q8-1795478803721843000
at commit 3189aaabee46cd24851b323671bfb1089d1be68b.

## Guaranteed positive-real h1 response

For every positive real relaxation rate a_j and symmetric PSD matrices
A_j and B_j, approximate the LOCAL characteristic admittance by

    H(s) = Σ_j [ s/(s+a_j) A_j + a_j/(s+a_j) B_j ]

Both rational functions have nonnegative real parts for Re(s)>=0, so H
is reciprocal and positive-real for the entire right half plane, not
only on a sampled frequency grid. Fit nonnegative coefficients of
rank-one PSD atoms using one NNLS problem for all matrix entries and
both real and imaginary reference axes; group atoms sharing a_j into
matrix A_j/B_j. No unstable per-entry poles. Reference samples solve
Yc(s) Z(s) Yc(s)=Y(s), Z=R+sL, Y=G+sC, with the principal branch.
The complex reference Riccati residual is under 5.4e-15 relative for
scalar, 3-wire and 8-wire fixtures.

## Tested numerical accuracy (independent frequency holdouts)

| Model | Poles | Real-axis maximum | Imaginary-axis maximum | NNLS setup |
| --- | ---: | ---: | ---: | ---: |
| Single wire, R>0, G=0 | 64 | 0.0099% | 0.060% | <0.1 s |
| Noncommuting coupled three-wire | 24 | 0.00015% | 0.028% | ~0.9 s |
| Noncommuting coupled eight-wire | 16 | 0.020% | 0.81% | ~78 s |

The eight-wire fit has 591 active rank-one weights, but only 16 shared
rates. A matrix-pole realization therefore needs 16*8=128 filter scalars
rather than 591. The costly model fit can be shared across equivalent
line instances and should be optimized further before production use.
Fixed 12 poles for G=0 previously caused up to 59% error; adaptive
pole range and holdout accuracy gates are mandatory. The finite-band
approximation does not claim global uniform accuracy, particularly
for singular zero-conductance branch points.

## Exact accepted/trial step convolution

For the vector-valued normalized filter

    z_j(t) = a_j ∫ exp[-a_j(t-u)] V(u) du

and a linearly interpolated candidate step of width Δ, write
h=a_j Δ, e=exp(-h), α=1-(1-e)/h and β=1-e-α. Then

    z_trial = e*z_accepted + β*V_accepted + α*V_trial
    I_trial = Σ [ A_j V_trial + (B_j-A_j) z_trial ]
    J_trial = Σ [ (1-α) A_j + α B_j ]

For every Δ>0, 0<=α<=1. Therefore J_trial is PSD, an exact
convex combination of the positive-real coefficient matrices for
each pole. Step preparation computes J_trial and the accepted-history
right-hand-side once. Repeated Newton iterations need only a matrix-
vector product; rejecting a trial never alters the accepted filters.
Stable expm1/Taylor weights handle tiny steps and effectively infinite
rate*time products. Explicit revision checks prevent stale acceptance.

The standalone Rust 2024 crate in rust/ demonstrates both independent
rank-one and compact matrix-pole realizations and compares them across
accepted/rejected Newton trials. Eleven unit tests passed on stable Rust,
managed sccache, with deterministic stale and cross-port transaction rejection.
Strict library and integration-test Clippy pass with pedantic/nursery and no
unwrap/expect/panic/todo/unimplemented allowances.

## Reproducing

    uv run --no-project --offline --with numpy --with scipy python shared_pole_core.py
    uv run --no-project --offline --with numpy --with scipy python matrix_benchmarks.py
    uv run --no-project --offline --with numpy --with scipy python refinement_benchmarks.py
    uv run --no-project --offline --with numpy --with scipy python validate_reference.py
    uv run --no-project --offline --with numpy --with scipy python transfer_stability.py
    uv run --no-project --offline --with numpy --with scipy python spatial_causality.py
    uv run --no-project --offline --with numpy --with scipy python spatial_rise_time.py

    cargo +stable --offline --config 'build.rustc-wrapper="/usr/local/bin/sccache"' \
      test --manifest-path rust/Cargo.toml \
      --target-dir /home/jan/build/ngspice-rs-fast/quality/issue-campaign-20261120/build/cpl-passive-research

Run from the research/cpl-passive-matrix directory (or use absolute script paths); target build
products only go under the campaign build directory. The Rust crate
has no third-party dependencies. Python dependencies are uv-managed.

## Remaining design scope

Only the local admittance kernel h1 is fitted. Physical delayed h2/h3
propagation, mode continuity across eigenvalue crossings, line MNA
stamps, accepted periodic convolution and public P registration are not
provided by this research. The active owner has a separate transfer-
matrix implementation, which requires attention to long-lossy-cable
exponential overflow and ideal lossless DC branch constraints.

Notion details: https://app.notion.com/p/421f74be902081499061d20be194cc86
Coordination: https://github.com/codeandsolder/ngspice-rs-fast/issues/183#issuecomment-7120144807

## Strict finite-front causality is not reproduced by finite pi ladders

The active #183 finite-volume spatial transient backend uses passive
capacitor/inductor pi cells. Its behavior can converge toward a continuous
distributed line for sufficiently low-frequency excitation, but no finite
number of those cells can reproduce a strict positive propagation delay
for a broadband input: its rational finite-dimensional transfer function
is analytic after t=0, so any nonzero received waveform generically has
a nonzero tail before the actual distributed-line wavefront.

An independent 1V source, matched 50-ohm driver/load, L=250nH/m,
C=100pF/m, length=0.2m test has exact physical arrival tau=1ns.
Using exact sparse-matrix-exponential integration of each pi ladder
(eliminating timestep discretization error), the far-end voltage at
0.99tau is 118.6 mV for 64 sections, 93.9 mV for 128 sections, and
63.7 mV for 256 sections, whereas the continuous distributed cable
must remain at 0 V until tau. With a 100ps linear input rise,
the 64-cell model still has 26.6mV and the 256-cell model 5.19mV
at 0.99tau. These prearrival responses are entirely spatial artifacts.
For meaningful strict delay semantics, propagation must retain an
explicit pure-delay factor with a causal post-delay response, not
just increasingly fine pi cells.

These results are in spatial_causality.json and spatial_rise_time.json.
The corresponding scripts can be rerun without a SPICE solver or
time integration assumptions.

## Alternative complete-line substrate: exact LC delays plus passive R/G

A more direct way to avoid both finite-pi-ladder pre-wavefront leakage and
unstable full-length ABCD exponentials is a power-normalized two-port
scattering cascade using physical operator splitting.

Write the telegrapher first-order generator as A(s)=A_LC(s)+A_RG,
where the LC part contains sL/sC and the R/G part contains physical
static R/G. One symmetric Strang cell of width dx is

    exp[A_LC(s) dx/2]  exp[A_RG dx]  exp[A_LC(s) dx/2].

Its lossless factors are **exact per-mode pure delays**, computed by a
single SPD modal eigensystem C=S S^T, eig(S^T L S)=Q diag(lambda).
The voltage/current bases S^-T Q and S Q are power reciprocal, and
each mode propagates with tau=dx sqrt(lambda). All conductors and
mutual coefficients are retained, with no frequency-dependent mode
tracking necessary for LC. The static middle stage is a passive,
memoryless reciprocal R/G two-port (which can have zero R or G).

A cascade of passive scattering blocks stays passive, while the
earliest possible transmission must traverse the sum of the exact
LC-stage mode delays: at least length*min(sqrt(lambda)).
Thus the model has an **exact front lower bound at every finite cell
count**, even though attenuation/dispersion are approximated by the
Strang split. The error in the full generator is second order in dx
for smooth finite-band excitations; adaptive refinement should be
driven by frequency-domain and time-domain error checks.

Using power-normalized modal waves makes the scattering matrices
contractive in ordinary Euclidean norm. The static block is converted
from its small physical transfer by solving a matrix equation that
does not require inverses of singular series resistance R, then
composed with exact cross-port modal delay scattering blocks through
Redheffer star products. No exponentially growing full-length
transfer is formed, and no artificial numerical regularization is
needed for zero R or zero G.

Validated in split_delay_modal.py / split_delay_modal.json over 42
log-spaced angular frequencies 1e3..1e10 rad/s:

| Cable | Cells | Max relative physical port Y error |
| --- | ---: | ---: |
| Lossless scalar | any | approximately 5e-12 near resonances |
| Noncommuting coupled 3-wire | 32 | 0.141% |
| Noncommuting coupled 3-wire | 64 | 0.0293% |
| Noncommuting coupled 3-wire | 128 | 0.00720% |
| Noncommuting coupled 8-wire | 16 | 0.0727% |
| Noncommuting coupled 8-wire | 32 | 0.0172% |
| Noncommuting coupled 8-wire | 64 | 0.00423% |

The exact modal lossless transfer also matched an independent 2N
matrix exponential for scalar, three-wire and eight-wire SPD LC
cases to relative error below 1e-11.

In scattering_split.py / scattering_split.json, 1/2/4/8/16-section
Redheffer cascades matched the independently constructed split
ABCD transfer to <2e-14 relative error for lossless, zero-G,
three-wire and eight-wire cases. All checked reciprocal scattering
matrices had largest eigenvalue of S^H S <=1+7e-16.
A physically passive DC line with gamma*length=1000, whose full
transfer would overflow float64, retained **finite scattering**:
expected input reflection -9/11, computed -0.8181818181818182,
transmission underflowing to zero as physically expected.

The scripts are research-quality Python and the full exact-delay
cascade is **not yet a Rust transient backend**: it still needs
per-port accepted launch histories, mode-wise delay interpolation,
timestep constraints, static scattering solves and internal-node
elimination, with MNA rollback transactions. Positive-definite
L/C is assumed for strictly positive modal delays; singular LC
limits (pure RC/RL or ideal constraints) require separate treatment.
Do not label this as complete P/CPL production support.

Additional reproductions:

    uv run --no-project --offline --with numpy --with scipy python split_delay_modal.py
    uv run --no-project --offline --with numpy --with scipy python scattering_split.py

## Adaptive section count without violating passivity

The exact-delay Strang splitting stage count must be chosen for the
driving waveform's relevant frequency bandwidth and material losses,
not fixed at 64 or any other unexplained number. An error-gated
prototype in adaptive_split.py compares the physically passive
power-normalized scattering of N and 2N stages at 21 log-spaced training
frequencies. It selects the *2N-stage physical network itself* once
their maximum relative difference falls below 0.1% (or returns a
nonconverged result when reaching the configured cap); it never
extrapolates admittance/scattering coefficients, which could break
passivity. Tests on a separate 42-frequency holdout grid compare
selected S against exact continuous RLGC S:

| Validated model | Selected stages | Maximum relative holdout S error |
| --- | ---: | ---: |
| Pure lossless scalar | 1 (exact analytical) | <1e-12 |
| R>0, G=0 high-Q scalar | 512 | 0.0141% |
| R>0, G>0 scalar | 256 | 0.0566% |
| Coupled noncommuting 3-wire | 64 | 0.0376% |
| Coupled noncommuting 8-wire | 32 | 0.0169% |

All converged with a 0.1% coarse-versus-fine training tolerance,
and the final passive networks remained contractive on holdouts.
The high-Q G=0 case required 512 sections, illustrating that error
tolerance alone cannot justify a fixed low iteration cap. The
coarse/fine criterion is *a heuristic finite-band error estimator*:
it does not certify all frequencies or time waveforms. For production,
derive checked frequency coverage from the actual transient source
bandwidth, identify poles/breakpoints, and ensure absolute as well as
relative tolerances are meaningful near resonances. Preserve the exact
physical pure delay at all selected resolutions.

## Causal accepted-history time-domain modal scattering prototype

The pure Python reference time_scattering.py now implements an actual
time-domain network of static, passive full-coupling R/G scattering cells
connected by exact per-mode LC delay lines (including half-length
boundary stages). It does not need any complex frequency-domain
fit to run a timestep: each static block has a constant real
power-normalized scattering matrix, and each LC link reads the
previously accepted mode wave at t minus its exact modal delay.
A candidate step computes every cell's output from accepted history
only, then commits the entire network's wave arrays on acceptance.
A rejected/stale trial cannot change accepted state.

**An important interpolation caveat was found experimentally**:
ordinary linear interpolation between samples of a delayed
signal can leak a positive sample backwards through the physical
front. Even a network made exclusively of exact delay factors can
exhibit a numerically premature wavefront if the history reader
interpolates across an arrival discontinuity. We fixed the
interpolation model, rather than clipping simulated output:
compute the **first physically reachable arrival of each modal
output** by shortest-path traversal of the passive scattering graph.
Each edge is an exact mode-dependent propagation delay, and each
static matrix coefficient is an instantaneous directed connection.
WaveHistory sampling returns its physical zero prehistory for query
times strictly before that reachable front; interpolation begins
at or after the front. This automatically accommodates actual
cross-mode scattering and reflection paths, arbitrary cell counts,
zero shunt/series loss, and unequal modal delays.

test_time_scattering.py verifies lossless matched step/ramp for
1/4/16/32 cells (no reflection, **strict zero prearrival**, complete
transmission after the physical delay), lossy scalar R-only/RG
and three-wire noncommuting R/L/G/C (coupled crosstalk but strictly
zero before earliest allowed far-end arrival), energy inequality
and rejection of outdated candidate trials. Numerical results
are captured in time_scattering.json.

test_frequency_time.py provides an independent Fourier-domain
consistency test: drive a matched cable with a sine wave in the
time-domain accepted-history simulator, then recover its modal
transmission phasor from the final four cycles via sine/cosine
least squares. Compare with independently star-composed
**complex AC scattering S21**. With 8 static cells and omega=2e9
rad/s, the maximum relative phasor discrepancies are:

- scalar lossless: 5.91e-6
- scalar lossy RLGC: 5.91e-6
- noncommuting three-wire, full modal coupling: 2.92e-6

All have physical early-front bounds and input energy exceeding
emitted energy for the sampled tests. Per-mode delay interpolation
and finite timestep introduce those tiny remaining discrepancies;
they are not unexplained complex transfer coefficients.

This Python simulator is a verified standalone research reference,
not yet linked into native Rust DeviceState/MNA. The retained wave
arrays can be pruned to bounded delay horizons, with careful retention
of interpolation brackets and physical-front metadata. The Rust h1
prototype separately demonstrates accepted-state owner/revision
transactions; the same semantics should be used when implementing
the full network in Rust.

## Independent Rust time-domain scattering state and differential reference

The standalone Rust crate now also contains a full causal power-wave
network engine at rust/src/wave_delay.rs, with a separately tested,
bounded sample-history reader, **exact** mode-dependent LC delays
between passive R/G scattering sections, and automatic earliest-front
tracking using a Dijkstra shortest-path search over the full
scattering graph (including mode conversion and reflected paths).
Every static matrix is validated for reciprocity and full power
contraction by checking the eigenvalues of its transpose-product;
an arbitrary active or nonreciprocal model is rejected at construction.

The engine applies full coupled scattering matrices in real time, not
a diagonal conductor approximation, and the earliest transmitted
front always traverses the full physical line. Crucially, the
accepted wave interpolation reader **never interpolates across the
first physically reachable front**. Wave launch histories are
retained as logical VecDeque ring sequences, preserving the
bracketing accepted sample at/before the longest required delay
when pruning; no copy of the history is needed for binary search.

A trial time step reads *only previously accepted* delayed waves
and cannot exceed the shortest half-cell modal delay. All static
outputs are computed without modifying accepted state. Accept
preflights the entire trial (including all shapes, finite values,
full energy, unique owner and accepted-state revision) and reserves
capacity for every trace before any accepted samples or time are
mutated. Foreign, malformed, stale, future or nonfinite trials
are refused atomically. The mutable state deliberately cannot
be cloned into two independent devices with the same identity.

The Rust crate's **19 unit tests** pass, including lossless strict
wavefronts for 1/4/16/32 sections, two-mode cross-coupling and energy
contraction, zero-shunt-loss limits, full static passivity rejection,
transaction isolation, and bounded-history equivalence. The full
all-target Clippy ruleset with -Dwarnings, pedantic, nursery, and
no unwrap/expect/panic/todo/unimplemented/debug debt is green
under managed sccache. The independent Python pruning tests compare
each transmitted/reflected sample for 801 accepted steps: exact
equality with no pruning, while retaining 100× fewer scalar history
samples and 62× fewer three-wire history samples.

Physical source matrices are tested as **cross-language differential
fixtures**, not just invented contractive matrices. The generator
generate_rust_fixtures.py evaluates actual coupled scalar and
noncommuting three-wire R/L/G/C materials using independent
Python/SciPy matrix exponentials and accepted-wave histories,
then emits small checked Rust source fixtures using exact IEEE-754
bit patterns (with grouped hex digits) to avoid parser rounding.
The Rust engine matches the Python reflected/transmitted wave
arrays at seven sampled checkpoints per cable across 192 accepted
steps, with component absolute discrepancy below 5e-10. Fixture
JSON records original physical materials, scattering matrices,
modal delays, expected outputs and energies. The generator is
deterministic and can be rerun with uv.

The independent time-to-frequency holdout compares the Python
accepted-history wave response against the full coupled AC S21
scattering at omega=2e9 rad/s and finds maximum relative errors
5.91e-6 (scalar lossless), 5.91e-6 (scalar RLGC), and 2.92e-6
(noncommuting three-wire). This is a check on complex transmission,
not merely temporal wavefront support.

The Rust code is a substantial complete **research stage network**
but is not wired into the actual simulator's MNA or accepted
StateHistory API. A timestep larger than the smallest half-cell
modal delay currently returns a truncation request. Removing
that restriction without violating exact wavefront causality
would require an in-step propagation solution for trial
input waveforms and its corresponding Newton Jacobian; it
must not be worked around by extrapolating unaccepted histories.
The research code does not claim full native P/CPL support.
