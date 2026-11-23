# Distributed-line and waveform numerical audit (#93)

This audit separates two invariants: **C's physical model and event semantics** (ordered accepted/trial history, delay causality, exact knots and one-sided discontinuities) and **intermediate binary64 rounding**, which can be changed when it destroys a finite physical result. Ordinary reference operations remain in C order unless doing so causes cancellation or avoidable overflow. The [issue](https://github.com/codeandsolder/ngspice-rs-fast/issues/93) follows #76/#78/#44; waveform/passive FMA changes from #85 are not recreated here. No relaxed acceptance gates or global fast math.

## Periodic source phases and EXP

- Written phases are exact finite binary64 degrees. `phase_radians` reduces degrees modulo 360 before multiplying by `PI` and dividing by 180. Thus the finite input `123456789012345°` has the same phase as `105°`; `f64::MAX` also has a finite remainder. Ordinary intra-cycle phases retain the reference multiply/divide order. Source construction, model sweeps, sensitivity replay, and transient values share this conversion. A signed zero remains signed.
- C itself converts huge degrees without periodic reduction and may lose the phase or overflow an intermediate. Pinned-C comparisons of those values therefore use their *analytically equivalent reduced phases* in the C deck; moderate inputs remain direct C comparisons. This is a documented deliberate numerical improvement, not a claim of matching C's inaccurate large-angle bits.
- `EXP` formerly evaluated `1-exp(-x)`, which becomes zero for tiny positive `x`. Rise now uses `-expm1(-x)`. Once both edges are active, combine the exponentials into `exp(fall)-exp(rise)`, using the larger exponential and `expm1` of the exponent difference. Equal time constants have an analytic difference `(fall_delay-rise_delay)/tau` independent of large absolute time; this retains the late tail. Reconstruct the value through the shared finite affine evaluator. No new step limits or event scheduling.
- PWL linear interpolation retains C's arithmetic within the segment but returns the **stored ordinate bits at an exact knot**; recomputing `v0 + 1*(v1-v0)` can incorrectly turn a perfectly represented `1` into `0` when `v0=1e16`. For opposite-sign extreme finite ordinates where `v1-v0` overflows, evaluate the convex combination without that intermediate. The same evaluator is used for TXL's exact accepted-history endpoints. TXL still chooses C's strict preceding bracket; only its arithmetic at a fraction of exactly one differs. Repeated PWL event landing and discontinuity-side logic is untouched.

## URC geometry

For a geometric ratio `K > 0, K != 1` and `n` sections, let the emitted section weights be `w_i=K^i`, `0 <= i < n`. The original C code finds

```
r1 = r0 (K-1) / (2 K^n - 2)
c1 = c0 (K-1) / (K^(n-1)(K+1) - 2)
```

These subtract nearly equal quantities for `K` close to 1 and can overflow `K^n` even when all **emitted** weights are finite. The model instead forms the physically equivalent positive normalizers `S_R = 2 sum(w_i)` and `S_C = 2 sum(w_i, i<n-1) + w_(n-1)`. Normalize first by the largest emitted weight (1 if `K<1`, `K^(n-1)` otherwise), sum with compensation, then divide totals `r0/c0/i0` by the scaled normalizers. `n=1` and the terminal capacitor's single multiplicity are explicit. This improves conditioning near `K=1`, avoids unrepresentable `K^n` and preserves total line R/C/IS. C's section number, elaboration order and positive-device checks remain.

The physical section topology regression compares individual coefficients with a 4-ulp relative bound against C's algebraically equivalent formula, not exact bit identity with its cancellation-prone division. Additional tests check aggregate R/C/IS conservation for adjacent-to-one ratios, 1/2, 3/2, many sections and finite ratios of order `1e308`.

## LTRA damped Bessel kernels

The RLC self endpoint contains `lag*(exp(-x)*(I0(x)+I1(x))-1)` for `x=beta*lag`. For tiny `x`, subtracting one erases the first-order `-x/2` response. Evaluate `expm1(-x) + exp(-x)*((I0(x)-1)+I1(x))`, constructing `I0-1` directly from the already used A&S polynomial's non-constant coefficients. For large `x`, use the existing A&S large-argument polynomials as `exp(-x)*Iν(x) = Pν(3.75/x)/sqrt(x)`; this avoids `exp(x)` overflow followed by multiplication by zero. Separately finite input factors are decomposed into significands and binary exponents, and the combined attenuation exponent is applied before final scaling; this rescues results when `alpha² * delay`, the Bessel growth or the attenuation alone exceed binary64 even though the complete response fits. The common finite midrange retains C's polynomial/operation sequence unless its intermediate scale destroys a representable result.

The RLC delayed propagation kernels use `sqrt(lag^2-delay^2)`. Only when squared times overflow or underflow does the implementation switch to `sqrt(lag-delay)*sqrt((lag+delay)/2)*sqrt(2)`. For Bessel arguments above 700, factor attenuation into `exp(arg-beta*lag)` and the existing scaled asymptotic polynomial. This allows finite, strongly damped responses to survive despite separate intermediate overflow. When a squared leading coefficient would overflow, use `alpha*(alpha*delay)` rather than `(alpha*alpha)*delay` where it remains finite. The packed four-lag history kernel keeps the ordinary path and falls back to stable scalar evaluation for rare ill-conditioned argument/near-zero lanes; the ring/history and reduction order are unchanged.

These are conditioning and range improvements to the *existing A&S approximation*, not a higher-order replacement for the Bessel series. Accuracy outside the physical finite regime is not manufactured; genuinely unrepresentable outputs still fail downstream validation.

## TXL recurrence

The accepted-state Padé update contains a term `(exp(pole*dt)-1)*right_hand_side`. The increment uses `expm1(pole*dt)`, preserving first-order changes for small steps; the separately rounded full exponential still drives the other term. No change to stored integer-picosecond ticks, age/prune logic, accepted/trial state, or ordering of recurrence sums. TXL sampled interpolation uses exact stored endpoints instead of reproducing the cancellation error from the earlier value at fraction one.

## Verification and performance

- Focused unit regressions cover waveform exact knots, modulo-period phase, small EXP edges/late tail, URC sum-conservation, TXL exact delayed tick, and LTRA small/large Bessel response with independent leading asymptotic estimates and scalar/batch agreement.
- Compare with the pinned ngspice executable `/home/jan/build/ngspice-rs-fast/parallel-analysis/ngspice-059066b5`, not system ngspice 47; the latter differs in supported cards and output.
- Existing `tests/{ltra,txl,urc_lines,source_functions,source_waveforms}` and live-C `c_{ltra,source_functions,urc}_reference` exercise time-domain and AC behavior. The binary C waveform sample test uses the true waveform at the rawfile timestamp and the immediately adjacent representable times to account for one-step accepted-time ambiguity (especially at discontinuities); smooth samples may differ by four ulps, while exact stored-knot arithmetic is separately bitwise verified. Include `cargo xtask golden verify` if individual focused tests are green. For direct RLC history cost use the ignored `tests/ltra_simd_bench.rs` coefficient and transient workloads, 8 to 16,384 accepted samples, contiguous and wrapped storage, on the same host and toolchain.
- Baseline LTRA benchmark and raw logs are under campaign `STATE/logs/worker-arch-1122-2155-b6f2/numerics-93`; production source uses managed sccache and the pinned nightly compiler. Integrate via the serial campaign validator only, never by manually merging the branch.

## Focused validation checkpoint — 2026-11-23

The first full focused development-profile integration run passed 11/11 LTRA, 15/15 TXL, 19/19 URC and 5/5 independent-source-function tests, in addition to the source-waveform suite and LTRA full crate 8/8. The pinned ngspice C reference binary passed all 4/4 live LTRA cases (DC, AC, S-parameters, transient including nonlinear diode), 8/8 live source-function cases and 2/2 live URC cases. Across 391 C timepoints, 8,602 individual source function values were compared. For 1,084 binary C waveform samples, the local time-envelope test found ordinary nonrepeating PWL differences of at most two ulps, 29 adjacent-time C landings for repeated PWL and 22 for PULSE. No accepted-time or stored-knot relaxation was introduced in production.

The isolated paired coefficient harness compiled **both** baseline and candidate LTRA numerical modules with the same pinned nightly, managed sccache, optimized flags and CPU affinity. Every RC/RLC coefficient checksum matched bit-for-bit for 8, 128, 2,048 and 16,384 history entries in both contiguous and wrapped storage. Throughput varied substantially with concurrent host-wide compiler activity, even for the untouched RC implementation; do not interpret those timings as a statistically supported speedup or slowdown. Files: `numerics-93/ltra-isolated-paired.log`, `ltra-isolated-interleaved.log` and `baseline-release-bench.log` under the campaign's worker logs.

Separate [TXL setup range issue #152](https://github.com/codeandsolder/ngspice-rs-fast/issues/152) records finite characteristic admittance and delay rejected because `(c/l).sqrt()` underflows or `(l*c).sqrt()` overflows. That geometric setup change is intentionally separated from the present waveform and propagation-kernel work.
