# BSIM4 nonlinear channel and charge numerics

Issue #156. The C equations remain the physical reference, not a requirement
to reproduce an intermediate binary64 rounding artifact. The load stages,
prepared size/model parameters, source/drain orientation and host-call
ordering remain intact.

## Constitutive-law changes

- Use log1p for the softplus lower tail and small logarithmic voltage
  corrections. At the original exp(-34) cutoff, log(1 + x) has approximately
  3.64% relative error. Keep the model's exponential clamp and branch rules.
- The smooth drain voltage is the smaller root of
  y² - (s + d + delta)y + s*d = 0. Evaluate it as
  2*s*d/(s + d + delta + sqrt((s-d-delta)² + 4*delta*s)).
  Evaluate both complementary gaps directly using rationalized positive
  half sums; the derivatives with respect to s and d are the corresponding
  gaps divided by the discriminant root. At s=0.5, delta=0.01, the original
  subtraction has about 9.91 ppm relative error at d=1e-12 V, 1.92% at
  d=1e-15 V and returns zero at d=1e-18 V.
- The channel saturation root of a*y²/2 - b*y + c = 0 uses
  2*c/(b + sqrt(b² - 2*a*c)), including the regular linear limit. Implicit
  differentiation uses (y²*da/2 - y*db + dc)/sqrt(b² - 2*a*c).
- For positive depletion increment x and positive k1, rationalize
  k1*(sqrt((k1/2)²+x)-k1/2) as k1*x/(sqrt((k1/2)²+x)+k1/2).
  Its slope is (k1/2)/sqrt((k1/2)²+x). Retain the existing zero-coefficient
  branch and linear negative continuation. Poly depletion and CV centroid
  use related rationalizations.
- Predict bias as previous + ratio*(previous - older), rather than
  (1+ratio)*previous - ratio*older. The difference form preserves a
  constant history and small increments at large step ratios.
- Smooth positive part p=(x+sqrt(x²+4*q))/2 uses p=2*q/(root-x) on the
  negative branch. Its slope is p/root; the unsmoothed origin uses the
  symmetric slope 1/2. Share this between series-resistance and flat-band
  charge laws. Accumulation charge consumes the shift directly instead
  of subtracting the rounded effective flat-band voltage.
- Effective-overdrive quotient derivatives use (N' - (N/D)*D')/D,
  avoiding D² and fusing the remaining subtraction. The intrinsic
  series-resistance conductance derivative uses 1/D/D instead of
  (1 - I*R)/D. Transient NQS partition derivatives use the same fused
  subtraction already used by the PZ path.

## Independent checks

The direct constitutive fixture drives the prepared intrinsic stages without
Newton iteration, voltage limiting, companion integration or subtraction of
terminal currents. The independent Richardson reference samples at ±h and
±h/2. Check both polarities, forward and reverse orientation, three charge
partitions and cap/CV modes (180 operating points), current derivatives,
the complete four-terminal charge Jacobian, charge conservation, both
Jacobian sum identities, positive-capacitance sign and low-field limits.
Use a global matrix-scale rounding bound for conservation because the
source row is reconstructed from other rows. Additional coverage includes
vanishing series resistance, mobility modes at 250/400 K, predictor
constant-history limits and the rationalized positive-part tail.

Branch discontinuities are not a reason to widen unrelated goldens. A
central derivative is meaningful away from the model's explicit branch
cuts; endpoint/zero-limit tests exercise the relevant one-sided laws.

## Audit of remaining FMA expectations

The force-warn inventory before the final tail/quotient changes contained
275 diagnostics in channel.rs and 211 in charge.rs (including overlapping
expression spans, not that many distinct operations). Inspect cancellation
sites separately from ordinary product-rule additions and ordered charge
assembly. The parent load module's broad suboptimal_flops and
manual_midpoint expectations have been removed, and the bias predictor
no longer inherits them. Channel and charge retain their translated
expression expectations for the remaining sums/branch equations, not as
a C bitwise-rounding requirement. Do not blindly fuse all suggested
expressions: the portable target may implement FMA in software, and
a changed parenthesization can alter charge-balance or branch semantics.
The independently tested sensitive expressions above take precedence.
Other load stages (junction, leakage, stamp) are outside this issue.

## Reproducible validation and performance

Before the changes the campaign base, incorporating the #154 normalized
back-substitution recovery, passed 9 device and 3 sensitivity tests without
changing tolerances. At 691853d all 25 device/schema/sensitivity goldens
passed, and both live tests against pinned ngspice 47+ (059066b5) passed:
the analysis/mode matrix and nested width/temperature sweeps. The prior
smoothing stage passed the 180-point independent derivative and
conservation test and the BSIM4 unit suite (44 tests). No golden reference
values were regenerated.

On Acer (Ryzen 7 4700U), an isolated drain-root/gap/derivative microbenchmark
using 16,384 deterministic input tuples, repeated 6000 times, with
saturation in [0.01,1.01], drain in [0,1.5] and delta=0.01, measured median
user CPU times of 0.90 s original / 0.75 s rationalized at target-cpu=native
and 1.00 s / 0.97 s at target-cpu=x86-64, three trials each. These measure
98,304,000 evaluations per trial, not full simulator throughput. The
benchmark used the original positive-branch arithmetic midpoint in the
rationalized kernel. Build with the campaign-pinned compiler and managed
sccache; opt-level=3. The original and rationalized paths return effective
voltage, drain gap and both derivatives. Use black-boxed inputs and
accumulate all four outputs to prevent elimination.

A second run used the actual overflow-safe midpoint in the rationalized
kernel, and compared otherwise identical fused and unfused discriminant
expressions. Median user CPU seconds (three trials):

| Target | Original | Rationalized + FMA | Rationalized, unfused |
| --- | ---: | ---: | ---: |
| native | 1.15 | 1.05 | 1.07 |
| x86-64 | 1.21 | 1.41 | 1.16 |

The portable fused path costs approximately 22% more than the same
unfused path, while the native difference is about 2%. Concurrent
campaign load changed CPU frequency between sessions, so compare
within a session. FMA remains appropriate where it directly preserves
a cancellation-sensitive derivative; these measurements do not
justify globally substituting software FMA for ordinary additions.

Focused reproduction:
```sh
cargo +nightly test --locked --lib devices::bsim4
cargo +nightly test --locked --test bsim4_device --test bsim4_schema --test bsim4_sens
# With NGSPICE_BIN pointing at the pinned C executable:
cargo +nightly test --locked --test c_bsim4_device_reference -- --ignored
```
