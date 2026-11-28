# Periodic conversion and noise around harmonic-balance orbits

Issue #175. This is a native typed analysis interface, distinct from C's
DC-biased AC, SP and noise analyses. No new netlist dialect is needed to
use it. It builds on the physical DAE and harmonic lattice of #172;
shooting-orbit support would need to provide the same physical samples and
convergence contract, not a substitution of DC bias.

## Binding and numerical contract

1. Construct a `HarmonicLattice` of independent pump axes and retained
   orders. Solve the physical current/charge equations with
   `harmonic_balance::solve_circuit` (or `solve` for an external physical
   model). Related tones use one common fundamental.
2. `PeriodicLinearization::circuit` binds the finalized circuit,
   candidate context and orbit; `prepare` binds an external
   `HarmonicModel`. Both reevaluate the actual physical Fourier residual
   against the supplied positive row tolerances. An unverified convergence
   flag or a stale coefficient vector cannot authorize a periodic analysis.
3. Project analytic current and charge Jacobians at the phase samples
   through HB's separable Fourier transform onto the complete **difference**
   lattice. With signed sideband offsets k,l and probe frequency fp,

       f_k = fp + k dot f_pump
       A_(row,k; col,l) = G_(row,col)[k-l] + j 2 pi f_k C_(row,col)[k-l].

   The derivative acts at the **output** frequency. Using the input or
   probe frequency for C would give incorrect charge mixing.
4. `factor(fp)` assembles an immutable sparse complex operator with this
   pattern. Exactly zero spectral derivatives omit the corresponding
   structural block, without thresholding small mixing coefficients.
   Constant Jacobians therefore scale linearly with sideband count.
   The real Newton matrix is not allocated during periodic preparation.
   Independent complex factors share the structural pattern and a
   one-entry exact-probe cache; previously returned responses own their
   immutable factor snapshots. Circuit-specific pivot tolerances are retained.
5. `PeriodicResponse::transfer` accepts sparse input equation terms and
   sparse linear observations, returning selected complex responses in
   input-major order. Multiple RHS reuse the native tiled complex solve.
   The optional `ResponseLimits::max_backward_error` uses the existing
   exact original-operator componentwise residual evaluator. This checks
   backward error, not a guarantee of well-conditioned forward accuracy.

The physical DAE refuses unsupported history-dependent, discrete-state,
frequency-local or nonperiodic device equations. The native analytic
model derivatives and physical charge law are the source of truth, not
finite differences of separate DC operating points. Independent source
forcing supports constants, undamped SIN and resolved RF PORT cosine
series, including the C load's accumulated predecessor semantics.

## Signed frequency conventions

`SidebandPlan` keeps integer offsets and signed physical frequencies in
ascending frequency order. A positive-frequency pump harmonic has the
same integer offset as HB; its negative counterpart has negated offsets.
Different integer combinations that collapse numerically to one
frequency are refused. No positive-frequency folding is performed in
the complex linear solve.

For a real excitation Re(A exp(j 2 pi fp t)), the opposite sector is its
complex conjugate. A physical positive-frequency peak phasor is the sum
of the complex-sector amplitude at +f and the conjugate of its amplitude
at -f; DC is the real part of the zero-frequency amplitude. The
`SidebandPlan::real_phasor` helper implements this rule for a scalar
observation. Overlapping conjugate sectors are explicitly flagged; complex
sector scattering results are not automatically interpreted as independent
real power channels in that case. Noise accumulation refuses overlapping
conjugate sectors rather than ignoring their phase-sensitive augmented
covariance. Zero frequency is not treated as a finite positive-frequency
flicker-noise sample.

## Native RF conversion

The circuit's native Thevenin ports (`portnum`, `z0`) retain their actual
series reference conductance and branch current in the HB and perturbation
matrices. Their impedance may differ by port. Port numbers must be contiguous.
For a real positive reference Z0 and current I into the network,

    a = (V + Z0 I)/(2 sqrt(Z0))
    b = (V - Z0 I)/(2 sqrt(Z0)).

The native source's branch current has the opposite direction to I.
Driving its branch equation with 2 sqrt(Z0) gives one incident wave;
the outgoing wave is a sparse observation of terminal voltage and branch
current. `incident_wave`, `scattered_wave` and `scattering` expose these
bindings for selected input/output sideband channels. No dense S/Y/Z
conversion or all-channel output allocation is required.

## Noise conversion

`periodic_noise::NoiseAccumulator` accepts selected sparse observations
and covariance pairs. It uses one **transposed** complex solve per
observation (not conjugate-transposed), so every generator can reuse
the same adjoints and factors. An externally prepared independent
stationary process is represented by one sparse complex RHS factor
whose amplitudes already include the square root of its spectral density.
The terms of this one factor are coherent across rows and sidebands.
Different factors represent statistically independent processes.

For a circuit-bound reference, `add_circuit_noise` samples the actual
physical orbit at its phase points. A fresh independent trial load
and small-signal slot evaluation produce each state snapshot; no accepted
or carried circuit state is published. The ordinary `NoisePlan` binds
stable named generator identities and refreshes their laws and transfer
rows at each phase and each signed incoming sideband's absolute
frequency. Thermal, shot, flicker and device-specific gain densities
use their existing scalar laws; coherent thermal generators retain
both transfer pairs and the relative phase (conjugated at negative
frequency). Nonfinite/negative PSDs and changed generator identities
fail the query. A failed native accumulation restores the previous
covariance and resource counters.

For a stationary generator at f_l with phase-local complex amplitude
b(theta, f_l), the translated injection into f_k is its Fourier
coefficient at k-l. Instead of materializing generator-by-sideband
matrices, synthesize each adjoint's phase envelope

    lambda(row,theta) = sum_k lambda(row,k) exp(-j k dot theta)

once. Multiply by b(theta,f_l) exp(+j l dot theta) and average over
the independent phase grid. Each independent source and input frequency
contributes a coherent observation amplitude vector h. The selected
covariance is

    N_ij = sum_source,sideband h_i conj(h_j).

The implementation accumulates coherent Fourier sums with compensation
and covariance products in exact bounded accumulators. This preserves
cross terms and phase without subtractive covariance construction.

The native amplitude model is the periodic modulation of a stationary
independent noise process by its instantaneous physical spectral law.
This describes white thermal/shot noise and separable colored noise
under the specified instantaneous-amplitude model. General noise with
additional temporal correlations is **not** inferred from a local DC
PSD; an external prepared factor model must supply that behavior.
Sideband truncation and phase oversampling require convergence checks
for both response and folded noise.

Native RF source-reference resistances have the same noiseless
device semantics as ordinary SP. The caller explicitly adds external
matched thermal termination factors with `add_thermal_input` (one
kT W/Hz incident-wave density per chosen input sideband).
`NoiseFigure::from_added_noise` reports |S|² conversion power gain,
input-referred added density, F=1+N_added/(kT|S|²), and 10 log10 F.
Added output noise excludes the chosen input's own reference kT
contribution, but includes other termination/sideband noise when supplied.
This defines a selected-channel noise figure rather than a hidden
single- versus double-sideband convention.

## Limits and verification

Existing `BalanceLimits` admit physical samples, spectral coordinates,
conservative expanded Jacobian work/storage and worst-case factor fill.
`ResponseLimits` bound selected transfer coefficients, terms and RHS
tiles. `NoiseLimits` bound selected covariance pairs, observation
adjoints, source factors, phase/model evaluations, aggregate phase
workspace and propagation work. Native noise may only be accumulated once
per result to avoid counting the same physical generators twice.

Regression suite: `tests/periodic_small_signal.rs`. It includes signed
two-pump modulation of both static and charge derivatives; comparison
against native ordinary AC/SP/noise at zero pump; independent analytic
Fourier noise-folding and coherent covariance; native pumped diode shot
noise compared to numerical Fourier integration of the junction law;
native correlated thermal phase at positive/negative frequencies and
rollback of a late generator-layout failure; exact-residual and factor
reuse checks; RF PORT pump forcing; a smoothly switched nonlinear mixer
compared to independent weak-signal transient; and zero-pump sparse
factor scaling. Existing HB, noise and SP regressions remain in the
focused suite.
