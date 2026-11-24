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

## Prepared large global timesteps: causal inner propagation with terminal Jacobians

The Rust standalone research engine now has a reusable
PreparedWaveStep from WaveNetwork::prepare_step(global_end_time).
It eliminates the **global** timestep requirement that every
circuit/Newton step be shorter than half of the shortest LC
subsegment delay. Instead, it resolves as many internal causal
substeps as the physical delay topology needs, while symbolically
keeping all internal waves affine in the 2N unknown *final* terminal
incident-wave amplitudes. A single prepared response may be
re-evaluated for arbitrarily many proposed terminal voltages/currents
without rerunning its O(internal-substeps*stages) propagation.

The endpoint reflected/transmitted-wave Jacobians are explicit
row-major matrices, including near-to-far and far-to-near
cross-coupling. WaveNetwork::accept_prepared commits **all**
interior delay samples in one atomic accepted-state operation; no
internal knot is committed when an outer Newton trial is rejected.
The unique owner/revision marker is still enforced, and all
wave arrays, times, energy and history capacities are preflighted
before any accepted state change. Histories retain all necessary
within-step causal knots and then prune safely at the macro end.

A newly found numerical detail matters: simply taking uniform
internal substeps can smear sharp source corners when their delayed
arrivals fall between the uniform knots. For example, a one-stage
lossless cable with 1ns propagation, a linear input ramp 0..2ns,
and then a held input must transmit exactly 1V at t=3ns, but an
unaugmented 3-point substep grid gave 0.958333V due interpolation
across the delayed corner. The event scheduler in
wave_delay/prepared/events.rs propagates accepted trace timestamps
and actual physical source front times through the exact modal
delay / nonzero static scattering graph, inserting delayed corner
times into the internal grid. It preserves *both* front causality
and piecewise-linear corner timing without output clipping.

Rust test coverage increased from 19 to **25 passing tests**.
New tests include one global timestep traversing entire lossless
1/4/16/32-cell cables, precise delayed input corners, reuse of
the same affine response across different Newton incident waves,
analytic vs finite-variation terminal Jacobians, future declared
source front, and atomic stale/foreign/malformed prepared-batch
rejection. All-target strict Clippy under managed sccache remains
zero-warning. The local internal work and materialized history
still scale with the number of necessary physical propagation
events; an event-driven sparse implementation and native solver
state integration are potential further performance optimizations.
This is a **prepared response substrate**, not yet a native P
device registered in the simulator.

## Physically coupled passive R/G stage and terminal MNA branch equations

The Rust research now eliminates another major native backend gap:
the static R/G network can be generated directly from the full
physical per-length R and G Maxwell matrices and the reciprocal LC
modal basis, without fitting any unstable Padé scalar element or
computing/exponentiating a full 2N transfer matrix.

The routine coupled_static_rg_pi in rust/src/wave_delay/static_pi.rs
realizes one *symmetric passive* resistive shunt/series/shunt stage:
both ends have G*dx/2 shunts and the center has series R*dx.
Instead of inverting singular R or G, it solves a scaled, pivoted
3N-by-3N linear system for the two physical traveling-wave outputs
and a vector of physical series branch currents. It solves all
2N input basis columns together, yielding a real power-normalized
2N-by-2N scattering matrix. It checks reciprocal power normalization
of the full LC modal voltage/current bases and verifies that the
result is reciprocal and contractive, rejecting active static
segments. Exact R=G=0 returns the identity mode-through
connection. Valid PSD singular R or G needs no artificial floor.

The static symmetric pi approximation retains second-order spatial
accuracy of the overall Strang LC/RG network. Independent SciPy
calculations across scalar, zero-G, noncommuting three- and eight-wire
examples give a static-stage relative scattering error below
3.3e-8 for a 1/32 length segment versus the exact R/G generator
matrix exponential. With 64 whole-line physical segments, the
noncommuting 3-wire and 8-wire maximum relative AC scattering errors
over 42 angular frequencies 1e3..1e10 are approximately
0.0376% and 0.00417%, respectively. The higher-Q single-wire
cases require bandwidth-dependent section counts; a 64-stage
constant count is not an accuracy guarantee.

The same Rust code passes exact-bit cross-language static R/G
fixtures generated by generate_rust_static_pi.py from independent
SciPy linear solves, including n=1,3,8 and all mutual couplings.
These fixtures also exercise zero series/shunt conductance, negative
resistance rejection, invalid geometry and nonreciprocal bases.

The physical terminal bridge in rust/src/wave_delay/prepared/terminal.rs
converts the prepared endpoint map (outgoing wave b = b0 + S*a)
into **inversion-free MNA branch rows**. In power-normalized
modal voltage alpha and inward line current beta, with
a=(alpha+beta)/2 and b=(alpha-beta)/2, the terminal relation is
(I+S)*beta + (S-I)*alpha = -2*b0. Keeping physical conductor
branch-current unknowns avoids the unsafe inversion of I+S in
ideal short/constraint limits.

The projection uses **full n-by-n** LC modal transforms at both
ends; it does not assume diagonal characteristic admittances or
uncoupled conductor modes. It emits sparse branch-equation Jacobian
triplets, signed conductor/return-node KCL branch stamps, and
branch-row RHS terms. Grounded return nodes are represented
explicitly as absent indices. After the circuit converges,
incident_from_physical reconstructs the normalized input waves
from the solved conductor differential voltages and branch currents,
allowing the exact prepared batch to be accepted atomically.
Full two-mode tests verify arbitrary near/far Newton candidates,
physical basis cross-coupling, grounded and non-grounded returns,
KCL signs, physical RHS residual and incident-wave recovery.

This remains a **standalone validated physical preparation and
stamping substrate**: the simulator's actual port node identities,
matrix storage handles, StateHistory lifetime, per-model LC
basis and P factory registration still need an intentional native
integration. Another issue worker owns the native #183 paths;
research work does not overwrite or bypass that campaign claim.

## Native CplModes-to-physical coupled-transient bridge

The standalone Rust research now provides an explicit native-model
interface in rust/src/wave_delay/native_modal.rs:
NativeModalCoefficients groups the existing modal eigensolver's
*reciprocal* voltage and current basis columns, characteristic modal
admittances, per-mode total line delays, full physical R/G Maxwell
matrices and line length. WaveDiscretization groups desired section
count, exterior source-front bounds and accepted-history pruning.

PhysicalModalLine::from_native_modes turns the existing native
modal decomposition into the power-normalized fixed LC bases for
causal passive scattering. For each mode k it scales the native
voltage basis column by 1/sqrt(admittance[k]) and current basis by
sqrt(admittance[k]), preserving instantaneous power. Its two inverse
physical-to-modal transforms are merely the transposes of the
opposite power-reciprocal bases, not separately inverted matrices.
The code uses the full 3N-by-3N coupled static R/G segment solver,
exact half-segment modal delays, and the existing causal accepted
history engine. It does not rerun LC eigenvalue decomposition.

The physical preparation-to-acceptance interface is now cohesive:
- prepare(time): compute all causal interior events for one global
  step and project its full reflected-wave Jacobian into physical
  conductor voltage and branch current coefficients.
- stamp(signal_nodes,return_nodes,branch_unknowns): emit complete
  sparse MNA KCL+branch row entries and RHS, with grounded returns.
- trial_physical(solved_differential_voltage,solved_branch_current):
  reconstruct normalized modal launched waves from the physical
  circuit solution and evaluate the complete non-mutating step.
- accept(batch): atomically commit the accepted interior modal waves
  only after the parent Newton/circuit solver accepts the trial.

Independent exact-bit physical LC/RG basis fixtures generated by
generate_native_modal_fixtures.py cover scalar lossless, zero-G and
fully lossy, plus fully noncommuting 3- and 8-conductor models. They
verify complete power-normalized bases and static scattering against
independent Python linear algebra, and round-trip arbitrary physical
conductor voltages/currents through the prepared MNA and accepted
modal wave conversion on **all five physical examples**.

The current complete research crate passes **38 Rust tests** and
strict all-target Clippy without warnings. A new dense 3-mode
8-stage breakpoint benchmark also identified nontrivial intrinsic
event fan-out: a single global step 0..0.9ns generates 101 internal
events, 1.5ns generates 426, 2.1ns 1302 and 3ns 5321. These are
physically causal propagated source/coupling echoes, not a cache
leak. Exact enumeration is expensive for long windows and many
modes; future optimization must preserve true first-arrival time
and provide a quantitative approximation/error bound for omitted
small-amplitude echoes, rather than silently capping path count.

The native MNA and StateHistory source paths of issue #183 are
currently claimed by another campaign worker. This research branch
has not modified those native files or claimed production P-device
registration; the validated bridge is ready for deliberate native
integration and harmonization with that worker's existing code.

## Canonical multimodal breakpoint identities and allocation reduction

Exact-delay passive networks generate distinct waveform corners from
sums of positive per-mode propagation times. The old graph event
scheduler identified them only by their accumulated floating-point
arrival timestamp. Two paths using the *same multiset* of modal
delays, but in a different order, could differ by rounding in the
last bit and be propagated redundantly. This becomes especially
expensive after repeated cross-mode conversion and reflection.

The new event scheduler in rust/src/wave_delay/prepared/events.rs
encodes each causal corner as:
- an original accepted source timestamp (exact f64 bits);
- the static-output graph node;
- a checked integer count of per-mode half-delay traversals.

Because addition of physical propagation lengths commutes, all
reorderings of the same modal-delay multiset have an identical
canonical identity and are expanded only once. This does **not**
quantize or discard physically distinct arrival times. Event times
are reconstructed deterministically from root timestamp and sorted
per-mode integer counts; overflow is rejected instead of wrapping.
A two-mode permutation-invariance test and checked count-overflow
test protect this mathematical invariant.

The 8-section 3-mode stress suite with irrational modal delay ratios
reports the following individual global-step event count reduction
relative to the previous floating-arrival scheduler:

| Global interval | Previously | Canonical |
| --- | ---: | ---: |
| 0..0.9 ns | 101 | 83 |
| 0..1.5 ns | 426 | 277 |
| 0..2.1 ns | 1302 | 716 |
| 0..3.0 ns | 5321 | 2081 |

This is **fewer events**, not a claim of guaranteed proportional
runtime reduction: the canonical hash keys are larger than a
floating arrival timestamp, and the physical solver still
recomputes nonlinear internal wave interactions at each event.

The affine Newton propagation coefficients additionally use
smallvec 1.15.1 (a long-established pinned version) for their
typically 2N<=16 modal Jacobian variables. This avoids a
separate heap allocation for each intermediate affine wave in
the 1..8-conductor native domain, while preserving a heap-backed
overflow path for wider research-only modal dimensions. No
physically supported case is silently truncated.

A physical, dense 8-conductor, 8-stage reference now has a stress
test for macro-steps 20%, 35% and 50% of the minimum physical line
delay: 85, 427 and 2529 physical event times respectively. After
inline Jacobian storage, the largest debug-build preparation took
8.36s versus 13.63s in the previous test run; an optimized
release build took approximately 1.63s. The remaining physical
event count is substantial, so eventual long-window optimization
needs a cached future-event frontier or quantified approximation
of low-amplitude reflected corners (while retaining strict
first-arrival boundaries).

A new complete **physical n=8 transient transmission test** also
simulates 200 short accepted steps across twice the fastest
continuous-line propagation delay. It checks exact zero far-end
prearrival, real nonzero transmitted modal crosstalk after arrival,
substantial direct transmission, and emitted energy no greater than
incident energy. This test is against the fully coupled native
modal/static-RG adapter, not an uncoupled synthetic line.

Total standalone Rust suite: **42/42 tests passing**, strict
all-target pedantic/nursery Clippy with no warnings and no
unwrap/expect/panic/todo/unimplemented usage. Source, bit-exact
fixture generators, Cargo.lock and the performance regression are
version controlled in the research branch; all Cargo build output
remains under the designated campaign build directory.
