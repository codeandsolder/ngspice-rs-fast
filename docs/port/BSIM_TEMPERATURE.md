# BSIM4 temperature and prepared-instance numerical contract

`src/devices/bsim4/temp.rs` implements the ordered BSIM4 4.8.3 model and
instance temperature passes (`b4temp.c`) and the subsequent model validation
(`b4check.c`). Model-selector branches, binning and geometry preparation,
"given" flags, temperature cache identity and the original source/instance
warning order remain owned by their existing stages. The temperature code
returns physical parameters, not accepted solver state.

## Short-channel screening

Seven prepared-instance pathways use the same size-scaled screening function:

\[
\Theta(x)=\frac{\exp(x)}{(\exp(x)-1)^2+2\,\exp(x)\,\text{MIN\_EXP}},
\qquad \text{MIN\_EXP}=1.713908431\times10^{-15}.
\]

These are the `dsub` and `drout` factors, two width/length threshold factors,
`mobMod=3`'s effective subthreshold coefficient, and two EOT/material-model
threshold factors. All call `short_channel_screening` rather than duplicating
locally rounded `exp(x)-1` expressions.

Close to zero, `exp(x)-1` loses meaningful bits before the result is squared.
For small arguments the helper obtains this difference using `expm1(x)` and
constructs the corresponding exponential; outside that cancellation region
it keeps the original well-conditioned arithmetic. The original `x < 34`
branch, ceiling value, NaN/infinity behavior and the finite-input branch
decisions are unchanged. There is no approximate polynomial, model subset or
branch-threshold adjustment.

The regression unit test in `src/devices/bsim4/temp.rs` independently compares
the factor with 90-digit decimal evaluations at exact binary64 inputs, across
very small positive/negative arguments, larger values, the clipping boundary
and exceptional values. Device-level temperature and physical C fixture tests
cover the surrounding geometry, model and solver semantics.

## Verification boundaries

`tests/bsim4_temperature_extremes.rs` exercises full model/instance
preparation at different temperatures, zero and tiny screening coefficients,
and model binning; it checks finite geometry and deterministic replay. The
existing `tests/c_bsim4_temp_reference.rs` compares temperature-stage fields
against the pinned native ngspice C executable when explicitly enabled.
Full 220-fixture C golden checks also validate operating-point and transient
behavior. Historical C intermediate rounding is not a physical accuracy
requirement, while deliberate selector, state-lifecycle and model-threshold
semantics still are.

Implementation reasoning, high-precision references, performance experiments
and campaign validation logs are tracked separately in the
[issue #155 engineering journal](https://app.notion.com/p/422f74be90208158994ed7ac6e86a0a1).
