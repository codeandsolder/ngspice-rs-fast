# Independent small-signal point factorization (#97)

A prepared fixed-bias small-signal circuit defines real conductance and dynamic
operators `A` and `E`. The value of `A + j omega E` changes per frequency,
but the structural matrix pattern and its validated symbolic elimination plan
need not be rebuilt for every point. This is separate from the SIMD
**multiple-RHS solve of one matrix** (#98): here each lane has a different
numeric matrix and its own pivots, factors, status and solution.

## Execution eligibility and production policy

The ordinary `.ac` driver retains C's sequential `SmallSignal::at_frequency`
loop and the circuit-owned SPARSE ordering cache. Empirical test- and
release-profile 1D/2D factor-and-solve measurements show no stable useful
throughput crossover for the packed SIMD implementation. Thus automatic
`.ac` SIMD dispatch is deliberately not enabled: this refactor must not
regress common SPICE analysis throughput.

The explicit `IndependentComplexBatch::solve` API operates on already
assembled, independent `ComplexMatrix` values and one RHS per point.
Callers that can establish independence may use it in bounded four-point
tiles; it returns ordered per-point `SpiceResult<Vec<Complex>>`. The
`factorize` variant retains independent factors for repeated RHS or
transpose solves. A frequency-dependent operating point
(`depends_on_frequency`) and frequency-dependent stamp reassembly
(`small_signal_depends_on_frequency`) remain unsuitable for independent
tiling: C's warm-start bias, accepted state, and stamp history semantics
must not change. No transient, nonlinear continuation, noise or multiport
execution is implicitly parallelized.

## Symbolic and numeric ownership

`ComplexMatrix::factorize_independent` can factor the initial matrix with
fresh ordering, then fork a compatible immutable SPARSE topology/packed
elimination plan for a separate numeric refactor. Incompatibility starts a
fresh order. `IndependentComplexBatch` groups up to four such compatible
matrices and retains only the most recent successful independent ordering.

The packed working coefficients and scratch use one element/row per
`LaneParts` (real and imaginary `f64x4`), with an active-lane mask and
independent reciprocal/pivot status. SIMD covers ordinary complex
multiplication and Schur updates; exponent-range, underflow and severe
cancellation use the same range-preserving scalar recovery kernel as the
ordinary SPARSE implementation. A fixed-order zero pivot retries only its
own lane using the standard threshold reorder. Scalar factors retain valid
normal and transposed solves; no C bit-for-bit pivot sequence is promised.

The logical tile width is four, independent of the potentially much larger
frequency grid or optimization candidate batch. `solve` packs the numeric
refactor and forward/back substitution without materializing a scalar LU for
each normal lane; its reusable tile scratch is not retained per point. Both bound circuit matrices and compatible
standalone sparse matrices are supported.

## Validation and crossover

- `cargo +nightly test --locked --lib independent_` checks scalar-equivalent
  normal/transposed solves, an explicitly batched real AC circuit against the
  scalar plot, observation ordering, and singular/zero-pivot isolation.
- `cargo +nightly test --locked --test solver_workloads_bench independent_complex_ -- --ignored --nocapture`
  compares scalar and packed factorization separately from matrix assembly
  on 1D and 2D grounded complex workloads. All solutions and failures are
  checked; timing is informational, not a numerical acceptance threshold.
- End-to-end and live-C analysis conformance remain authoritative for
  physical equations, event/error semantics and observation accuracy.

Performance data: seven serial test-profile and nine serial release-profile
repetitions of representative 1D and 2D matrices compared factorization and
direct factor+solve separately. The packed path is predominantly slower
(including 1024-node meshes at about 0.26–0.28x scalar solve throughput in
release) despite matching reference solutions. Some isolated noisy cases
appear faster but do not establish a reliable crossover. The durable raw
measurements, proposed next experiments and decisions live in the separate
Notion architecture #97 worklog. Any later production dispatch must
demonstrate useful end-to-end speed on comparable, representative workloads,
not just correctness or synthetic peak width.
