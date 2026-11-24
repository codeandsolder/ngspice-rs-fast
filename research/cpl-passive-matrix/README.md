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
