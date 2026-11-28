# BSIM4 temperature and prepared-instance numerical conditioning

The temperature stage follows BSIM4 4.8.3's model/instance preparation
sequence, given-state, parameter checks, and branch conditions. The
conditioned expressions below implement the same physical equations;
the original floating-point evaluation is not an oracle for severe
cancellation or intermediate range loss.

## Conditioned kernels

- For Eg(T) = Eg0 - alpha T²/(T + beta), evaluate the difference of
  Eg(T)/kT at nominal and operating temperatures by factoring T-Tnom.
  The reciprocal difference is (T-Tnom)/T/Tnom, and the material
  contribution is alpha beta (T-Tnom)/((T+beta)(Tnom+beta)).
  Near nominal, use the relative difference and log1p.
- Short-channel exponentials use expm1 for small arguments before
  squaring. The C exponential threshold and minimum exponential remain.
- The body-effect denominator 2(sqrt(phi(phi-vbm))-phi)+vbm
  equals -(sqrt(phi-vbm)-sqrt(phi))². Rationalize the square-root
  differences. Preserve the original evaluation for singular and
  nonphysical inputs.
- Forward diode limiting solves for z = exp(Vjm/Nvtm)-1 using a
  cancellation-safe quadratic root, hypot, and log1p. Use the defining
  current and the scaled discriminant for the corresponding slope.
  This avoids reconstructing a small forward current by subtracting
  exponentials close to one.
- Length/width/cross bin interpolation shares one typed weighted-sum
  kernel for all 160 fields. Normal, sufficiently well-conditioned
  sums retain the ordered expression; zero corrections retain the
  ordered IEEE zero/nonfinite behavior. When |sum| <= 1/8 sum(abs(terms))
  or the result is nonnormal, recover finite products with the existing
  exact accumulator. Cross weights scale the effective dimensions
  only when their direct product is outside the normal range.
- Temperature density and body resistance scaling share the normal
  exponential per exponent. When exp would overflow or become
  subnormal, multiply by exp(exponent/4) four times, avoiding an
  out-of-range intermediate while the result remains representable.
- For well proximity, use x = rate*distance/reference and
  z = rate*width/reference. The exponential integral is
  scale*exp(-x)*(x*(-expm1(-z)/z) + (1-(1+z)*exp(-z))/z).
  Evaluate the second quotient by its convergent series for z <= 1/2.
  The narrow-width limit is scale*x*exp(-x). The rational WPE term
  and parallel body resistance use scaled arithmetic when needed.
- EOT poly depletion uses the same physical helper as the load stage;
  overlap-capacitance, oxide-thickness and body coefficient updates
  use fused operations where subtraction exposes rounding error.

## Ordered source/drain junction preparation

The source and drain junction limiter calculations share one pure
side-input preparation kernel. Side-specific saturation-current densities,
effective area/perimeter, gate-edge width, finger count, emission voltage,
breakdown voltage/coefficient and forward/reverse limits are passed
explicitly. A side with zero effective area **and** perimeter has zero
saturation current, even when its gate-edge term alone would be nonzero.
A side with nonpositive or nonfinite saturation does not overwrite any
previous limiter fields.

The C diode modes deliberately write different subsets of fields:
mode 0 sets only the breakdown exponential; mode 1 sets only forward
voltage/current; mode 2 sets breakdown, forward voltage/current/slope,
and reverse voltage/current/slope. An unsupported mode emits its
original warning only when saturation is positive. Mode 2 clamps a
reverse threshold ratio strictly below 1 to 10 and produces the exact
source or drain warning text. The model and instance are never
mutated by the shared evaluator: its results are committed in C order,
**source first, drain second**, including all warnings. Physical side
asymmetry and prior conditioned diode-limit numerics are retained.
Old C-style scratch assignments without subsequent reads are removed,
not reintroduced as adapter state.

Bitwise differential tests against the pre-extraction source/drain
calculation exercise all diode modes, zero/negative/NaN geometry,
one-ULP exponent thresholds, reverse threshold crossings, different
finger counts and source/drain density asymmetry. The public
prepared-model API tests warning order, side inactivity, polarities
and temperature bins; existing C-reference setup and device suites
continue to enforce the physical model outcomes.

## IEEE and C parity

No change to cache keys, parameter-set ordering, version dispatch,
model-bin selection, or warning comparison thresholds. The former
module-wide FMA/midpoint suppressions are narrowed to legacy
nonphysical-input paths and the ordered preparation functions.
Remaining ordinary sums and C diagnostic/clamp arithmetic do not
justify mechanically introducing software FMA on generic targets.
In device.rs, zero-coefficient integration intentionally retains
separate IEEE multiplication, addition and subtraction: 0*infinity
is NaN and signed zero can be observable. Regression tests explicitly
protect those identities.

## Evidence and performance

Independent Decimal references cover adjacent temperatures, near-zero
body bias, short-channel factors, exponential scaling and narrow-width
integrals. Tests also exercise exact cancellation including product
rounding errors and a prepared instance with cancelling threshold
bins. Existing C-reference tests are not relaxed.

The first conditioning stage passed 44 BSIM4 unit tests, 25 offline
goldens, and five live C-reference tests against ngspice 47+ (059066b5).
The additional range-conditioning stage is checked by the campaign
regression command and strict lint gate.

Acer scalar microbenchmark (nightly, opt-level=3, native and x86-64):
16,384 coefficient/weight tuples, repeated 6000 times (98.304M)
for sparse/dense or 600 times (9.8304M) for 10% severe cancellation.
Three trials per case; median wall time in milliseconds:

| Target | Distribution | Ordered C arithmetic | Conditioned |
|---|---|---:|---:|
| native | sparse (90% zero corrections) | 413 | 981 |
| native | dense | 138 | 1038 |
| native | 10% severe cancellation | 26 | 309 |
| x86-64 | sparse | 558 | 1005 |
| x86-64 | dense | 209 | 1876 |
| x86-64 | 10% severe cancellation | 47 | 356 |

The host was shared with other builds; timings are noisy and not
whole-circuit throughput. Ordinary checksums agree. The severe
cancellation checksum changes from 8986525.38 to 9969925.38,
recovering 983400 units lost by the ordered sum (1639 cases per
repetition). This extra work occurs in model preparation, not Newton
load iterations. The cold exact accumulator is intentionally reserved
for poorly conditioned sums rather than used for every field.

### Whole-preparation interpretation (#194)

The extra conditioning is justified by recovered physical results, not by
isolated scalar-tuple throughput. Two algebraically equivalent shortcuts
for normal/zero bin corrections were tested in alternating A/B runs with
hardware counters pinned to separate CPU cores. Neither produced a
repeatable whole-model preparation improvement; the zero-coefficient
shortcut actually increased standalone interpolation CPU cycles by about
5.7% across both cores, while whole-preparation differences remained
within shared-host variability. The accepted exact-product recovery and
its sparse/dense fast paths are therefore retained unchanged.

The [#194 investigation](https://app.notion.com/p/422f74be902081a796f9c809566b8905)
contains the pinned binaries, physical-output checksums, counter traces and
reproducible experimental benchmark branch. A future optimization should
profile complete model preparation, not optimize only this tiny subkernel.
