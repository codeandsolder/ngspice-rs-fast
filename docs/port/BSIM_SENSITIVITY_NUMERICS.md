# BSIM4 convergence and sensitivity rounding

BSIM4 convergence is a numerical decision procedure, not merely a prediction
of device current. The predictor arithmetic and the **branch boundary** have
different rounding contracts.

## Convergence decisions

The **cvtest.rs** convergence routine follows original C b4cvtest.c
forward/reverse formulas. The first six drain, source resistance and gate
leakage checks reject when the magnitude of a predicted current difference
is **greater than or equal to** the reference current tolerance. The final
bulk-current check rejects only when it is **strictly greater**. An off
instance bypasses the check only under MODEINITFIX. Preserve all junction
selection, model signs, and NaN-sensitive C-style MAX arguments.

Predictor sums use hardware FMA for cancellation accuracy only when the
requested compilation target enables the FMA target feature. On baseline
x86-64/x86-64-v2, they use ordinary multiply/add rather than costly software
FMA. This is a compile-time choice with no runtime dispatch or intermediate
allocation. The convergence tolerance is *never* fused: it first rounds the relative
product in binary64, then adds absolute tolerance in binary64. The local
**convergence_tolerance** function preserves this behavior.

A one-ULP witness uses epsilon=2^-27, reltol=1+epsilon,
magnitude=1-epsilon, and abstol=3*2^-54. The separately rounded tolerance
is 1+2^-52 whereas a fused multiply-add yields exactly 1. An increment
of precisely one ampere is accepted by the C branch and would be rejected
by the fused alternative. Tests cover both sides of the >= and > decisions,
forward and reverse mode, signed zero, and nonfinite input values.

## Sensitivity and rejected geometry

BSIM4 sensitivity retains the C parameter set, setup, unsetup, temperature
and bias-load sequence. C sensitivity output is not automatically the same
as an independent physical central-difference derivative: a rejected
geometry may leave a partly prepared instance and model record.

**sens.rs** uses a local **c_effective_span** function to preserve the
original separately rounded effective length/width subtractions before
the rejection checks. Doubling a very large side-loss value may overflow
even when the fused difference would be finite. C's overflow, zero
classification, and stored members are intentional here.

## Verification

- **src/devices/bsim4/cvtest.rs**: 1-ULP convergence threshold tests,
  C signed-zero/NaN comparison behavior, mode reversal and off/INITFIX.
- **src/devices/bsim4/sens.rs**: effective geometry below/at/above zero
  and overflow behavior.
- **tests/bsim4_convergence_boundaries.rs**: an independently converged
  operating-point central/Richardson model-parameter derivative compared
  to the simulator's physical sensitivity result.
- **tests/bsim4_sens.rs** and committed C-reference suites: historical
  C sensitivity results and their known NaNs/platform differences.
- The full committed C golden corpus and strict campaign inventory must
  remain green without relaxing any acceptance tolerances.

See [verification policy](VERIFICATION.md). The #158 Notion implementation
log records detailed experiments and performance data separately from this
permanent numerical contract.

## Predictor decomposition

Bias deltas and current predictions are represented separately from the
discrete convergence decisions. Fixed-size ordered gain/delta reductions
preserve the reference accumulation order and target-specific fusion policy.
The bulk predictor is evaluated only after the first six checks pass.

A comparison against the preceding nested predictor implementation covered
200,000 randomized decisions on each of native and generic x86-64, including
aliased nodes, both polarities, nonfinite values and signed zero, with no
mismatches. Three-run median timings for 1,638,400 decisions were 234.72 to
210.44 ms (native) and 167.50 to 164.81 ms (generic). These are isolated
predictor measurements on a shared machine, not whole-solver speedups.
