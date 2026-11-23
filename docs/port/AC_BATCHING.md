# Independent small-signal point factorization (#97)

A prepared fixed-bias small-signal circuit defines real conductance and dynamic
operators `A` and `E`. The value of `A + j omega E` changes per frequency,
but the structural matrix pattern and its validated symbolic elimination plan
need not be rebuilt for every point. This is separate from the SIMD
**multiple-RHS solve of one matrix** (#98): here each lane has a different
numeric matrix and its own pivots, factors, status and solution.

## Execution eligibility

The `.ac` driver uses at most four frequency points per tile only when no
device requires a frequency-dependent operating point
(`depends_on_frequency`) or frequency-dependent small-signal reassembly
(`small_signal_depends_on_frequency`). These cases continue to call
`SmallSignal::at_frequency` sequentially, including the prior bias as the
warm start and the accepted device-state recording. No transient, nonlinear
continuation, noise or multiport execution is implicitly parallelized.

Every point retains its original frequency index, its own assembled
`A + j omega E`, and the parameter observation context at that frequency.
The sink sees checked points strictly in grid order and stops on its first
error. The independent numeric API can report subsequent lanes' errors and
solutions without one failing lane poisoning its peers.

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
frequency grid or optimization candidate batch. Tile scratch is reused,
not retained once per point. Both bound circuit matrices and compatible
standalone sparse matrices are supported.

## Validation and crossover

- `cargo +nightly test --locked --lib independent_` checks scalar-equivalent
  normal/transposed solves, physical multi-point AC vs singleton AC, frequency
  and observation ordering, and singular/zero-pivot isolation.
- `cargo +nightly test --locked --test solver_workloads_bench independent_complex_ -- --ignored --nocapture`
  compares scalar and packed factorization separately from matrix assembly
  on 1D and 2D grounded complex workloads. All solutions and failures are
  checked; timing is informational, not a numerical acceptance threshold.
- End-to-end and live-C analysis conformance remain authoritative for
  physical equations, event/error semantics and observation accuracy.

On the initially measured narrow-band matrix workloads, the packed path was
mostly slower than scalar before the vectorized-mask and reusable-workspace
optimizations. The implementation must demonstrate a useful crossover (or
use scalar dispatch where it does not) before being treated as a general
throughput improvement; matrix assembly and sink overhead are distinct from
the factorization microbenchmark.
