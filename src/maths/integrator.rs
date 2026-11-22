//! SPICE companion-model integration of charge-storage elements.
//!
//! When a capacitor or inductor is stamped into the MNA matrix, its current
//! depends on the history of its charge (or flux). ngspice turns that
//! dependence into an equivalent conductance plus a known current source — the
//! "companion model" — using a numerical integration rule:
//!
//! - **Trapezoidal** (order 2, the default), with backward-Euler order 1 used
//!   at startup and after breakpoints;
//! - **Gear** (variable-step backward differentiation), selected by
//!   `.option method=gear`.
//!
//! The port follows `src/maths/ni/nicomcof.c` (coefficients `CKTag`/`CKTagp`),
//! `src/maths/ni/niinteg.c` (`NIintegrate`), `src/maths/ni/nipred.c`
//! (`NIpred`) and `src/spicelib/analysis/cktterr.c` (`CKTterr`): trapezoidal
//! orders 1–2 and variable-step Gear orders 1–6 (#98). Which order a run
//! actually uses is the transient driver's policy; `dctran.c` (and therefore
//! [`crate::analysis::companion`]) only ever toggles between orders 1 and 2,
//! whatever `maxord` says, so Gear orders 3–6 are reachable through this API
//! but not from an ngspice-compatible `.tran`.
//!
//! # Trial versus accepted state
//!
//! [`StepHistory`] holds the step sizes of *accepted* time points, most recent
//! first (C `CKTdeltaOld[1..]`). [`StepHistory::trial`] builds immutable
//! [`Coefficients`] for a proposed step without touching the history, so any
//! number of rejected or repeated trials leave it intact. Only
//! [`StepHistory::accept`] advances it. The same rule applies to the charge and
//! derivative histories, which are owned by the caller (devices) and passed in
//! as slices.
//!

use crate::primitives::{Real, SpiceError, SpiceResult};

/// The present charge followed by up to [`IntegrationMethod::MAX_GEAR_ORDER`]
/// accepted charges, on the stack: device loads build one per charge per
/// Newton iteration, so it must not allocate.
#[derive(Debug, Clone, Copy)]
pub struct ChargeHistory {
    values: [Real; IntegrationMethod::MAX_GEAR_ORDER as usize + 1],
    len: usize,
}

impl ChargeHistory {
    /// A history holding only the present charge.
    #[must_use]
    pub const fn new(present: Real) -> Self {
        let mut values = [0.0; IntegrationMethod::MAX_GEAR_ORDER as usize + 1];
        values[0] = present;
        Self { values, len: 1 }
    }

    /// Appends the next older accepted charge.
    ///
    /// # Errors
    /// More points than the highest integration order can use.
    #[inline]
    pub fn push(&mut self, value: Real) -> SpiceResult<()> {
        let slot = self
            .values
            .get_mut(self.len)
            .ok_or_else(|| numerical("charge history exceeds the highest integration order"))?;
        *slot = value;
        self.len += 1;
        Ok(())
    }
}

impl std::ops::Deref for ChargeHistory {
    type Target = [Real];
    fn deref(&self) -> &[Real] {
        &self.values[..self.len]
    }
}

/// Which integration rule to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegrationMethod {
    /// Second-order trapezoidal rule. ngspice's default.
    #[default]
    Trapezoidal,
    /// Gear's backward differentiation formula, representable up to order 6.
    Gear {
        /// Maximum integration order (`.option maxord`).
        order: u8,
    },
}

impl IntegrationMethod {
    /// The highest Gear order ngspice supports.
    pub const MAX_GEAR_ORDER: u8 = 6;

    /// The highest order with implemented coefficient/history operations:
    /// every representable Gear order since #98 (trapezoidal stops at 2).
    pub const MAX_RUNTIME_ORDER: u8 = Self::MAX_GEAR_ORDER;

    /// True when this method/order is representable, not proof that its
    /// coefficient/history operations are implemented. See
    /// [`Self::validate_runtime`].
    #[must_use]
    pub const fn is_valid(self) -> bool {
        match self {
            Self::Trapezoidal => true,
            Self::Gear { order } => order >= 1 && order <= Self::MAX_GEAR_ORDER,
        }
    }

    /// The highest order this method may run at: 2 for trapezoidal, the
    /// configured order for Gear.
    #[must_use]
    pub const fn max_order(self) -> u8 {
        match self {
            Self::Trapezoidal => 2,
            Self::Gear { order } => order,
        }
    }

    /// Checks that this method's maximum order is implemented.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Unsupported`] for Gear orders 0 and above 6 (ngspice's
    /// `cktsopt.c` clamps `maxord` to `1..=6` with a warning; the port refuses
    /// instead of silently changing the request).
    pub fn validate_runtime(self) -> SpiceResult<()> {
        if self.is_valid() {
            Ok(())
        } else {
            Err(SpiceError::Unsupported {
                feature: format!(
                    "{} integration order {}; orders 1 to {} are implemented",
                    self.as_str(),
                    self.max_order(),
                    Self::MAX_GEAR_ORDER
                ),
                location: None,
            })
        }
    }

    /// The name used by `.option method=`.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Trapezoidal => "trap",
            Self::Gear { .. } => "gear",
        }
    }

    /// Parses `.option method=trap|gear` together with `maxord`.
    #[must_use]
    pub const fn parse(name: &str, gear_order: u8) -> Option<Self> {
        if name.eq_ignore_ascii_case("trap") || name.eq_ignore_ascii_case("trapezoidal") {
            Some(Self::Trapezoidal)
        } else if name.eq_ignore_ascii_case("gear") {
            Some(Self::Gear { order: gear_order })
        } else {
            None
        }
    }
}

/// The ngspice default trapezoidal weighting, `.option xmu=0.5`
/// (`cktntask.c`). `0` degenerates to backward Euler.
pub const DEFAULT_XMU: Real = 0.5;

/// Number of accepted step sizes retained (C `CKTdeltaOld[1..=6]`): enough
/// for order-6 Gear prediction and truncation-error estimation, which need
/// `order` accepted steps.
const HISTORY_LEN: usize = IntegrationMethod::MAX_GEAR_ORDER as usize;

/// `cktterr.c` `gearCoeff`: the error constants of Gear orders 1–6.
const GEAR_ERROR_CONSTANTS: [Real; 6] = [
    0.5,
    0.222_222_222_2,
    0.136_363_636_4,
    0.096,
    0.072_992_700_73,
    0.058_309_037_90,
];
/// `cktterr.c` `trapCoeff`: the error constants of trapezoidal orders 1–2.
const TRAP_ERROR_CONSTANTS: [Real; 2] = [0.5, 0.083_333_333_33];

fn numerical(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "integrator".to_owned(),
        message: message.into(),
    }
}

#[inline]
fn finite(value: Real, what: &str) -> SpiceResult<Real> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(not_finite(what))
    }
}

#[cold]
#[inline(never)]
fn not_finite(what: &str) -> SpiceError {
    numerical(format!("{what} is not finite"))
}

#[cold]
#[inline(never)]
fn cold_numerical(message: &str) -> SpiceError {
    numerical(message)
}

/// Accepted step sizes, most recent first — C `CKTdeltaOld[1..]`.
///
/// This is the only integration state that advances between time points, and
/// it advances only through [`Self::accept`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepHistory {
    accepted: [Real; HISTORY_LEN],
    len: usize,
    /// Placeholder for steps older than the first accepted one.
    fill: Option<Real>,
}

impl StepHistory {
    /// A history with no accepted steps.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            accepted: [0.0; HISTORY_LEN],
            len: 0,
            fill: None,
        }
    }

    /// A history whose not-yet-accepted older steps read as `fill`, as C
    /// initializes `CKTdeltaOld[0..7]` to `CKTmaxStep` before the first step.
    ///
    /// The placeholders only feed [`Coefficients::truncation_timestep`] (which
    /// may then run at an order above the number of accepted steps, exactly as
    /// `dctran.c` probes order 2 on its second step); coefficients and
    /// prediction still require real accepted steps.
    #[must_use]
    pub const fn with_fill(fill: Real) -> Self {
        Self {
            accepted: [0.0; HISTORY_LEN],
            len: 0,
            fill: Some(fill),
        }
    }

    /// Accepted step sizes, most recent first.
    #[must_use]
    pub fn accepted(&self) -> &[Real] {
        &self.accepted[..self.len]
    }

    /// The number of retained accepted steps.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// True before any step was accepted.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The highest order `method` can use for a trial right now. Gear order
    /// `k` needs `k - 1` previous steps; trapezoidal order 2 needs a previous
    /// derivative, which exists once one step was accepted.
    #[must_use]
    pub fn available_order(&self, method: IntegrationMethod) -> u8 {
        let by_history = u8::try_from(self.len + 1).unwrap_or(u8::MAX);
        method
            .max_order()
            .min(by_history)
            .min(IntegrationMethod::MAX_GEAR_ORDER)
    }

    /// Builds the coefficients for a trial step of size `dt`, without
    /// changing this history. `xmu` only affects trapezoidal order 2.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] for a nonpositive/nonfinite step, `xmu`
    /// outside `[0, 0.5]`, nonfinite coefficients or a history too short for
    /// the order; [`SpiceError::Unsupported`] for order 0, an invalid method
    /// (Gear above order 6) or an order above the method's maximum (2 for
    /// trapezoidal).
    pub fn trial(
        &self,
        method: IntegrationMethod,
        order: u8,
        dt: Real,
        xmu: Real,
    ) -> SpiceResult<Coefficients> {
        if !(dt.is_finite() && dt > 0.0) {
            return Err(numerical(format!(
                "timestep must be positive and finite, got {dt}"
            )));
        }
        if !(xmu.is_finite() && (0.0..=0.5).contains(&xmu)) {
            return Err(numerical(format!("xmu must be in [0, 0.5], got {xmu}")));
        }
        if !method.is_valid() || order == 0 || order > method.max_order() {
            return Err(SpiceError::Unsupported {
                feature: format!(
                    "{} integration at order {order} (maximum {}; implemented: trap 1..=2, \
                     gear 1..={})",
                    method.as_str(),
                    method.max_order(),
                    IntegrationMethod::MAX_GEAR_ORDER
                ),
                location: None,
            });
        }
        if order > self.available_order(method) {
            return Err(numerical(format!(
                "order {order} needs {} accepted step(s), have {}",
                order - 1,
                self.len
            )));
        }
        let mut deltas = [0.0; HISTORY_LEN + 1];
        deltas[0] = dt;
        deltas[1..=self.len].copy_from_slice(self.accepted());
        if let Some(fill) = self.fill {
            deltas[self.len + 1..].fill(fill);
        }
        // nicomcof.c
        let ag = match (method, order) {
            (IntegrationMethod::Trapezoidal, 1) => {
                let mut ag = [0.0; HISTORY_LEN + 1];
                ag[..2].copy_from_slice(&[1.0 / dt, -1.0 / dt]);
                ag
            }
            (IntegrationMethod::Trapezoidal, _) => {
                let mut ag = [0.0; HISTORY_LEN + 1];
                ag[..2].copy_from_slice(&[1.0 / dt / (1.0 - xmu), xmu / (1.0 - xmu)]);
                ag
            }
            // nicomcof.c: Gear order 1 falls through to the general case.
            (IntegrationMethod::Gear { .. }, _) => gear_corrector(&deltas, usize::from(order)),
        };
        for value in ag {
            finite(value, "integration coefficient")?;
        }
        Ok(Coefficients {
            method,
            order,
            deltas,
            history: self.len,
            filled: self.fill.is_some(),
            ag,
            truncation: TruncationSpans::new(&deltas, usize::from(order)),
        })
    }

    /// Commits the step of accepted `coefficients`, shifting older steps
    /// (C keeps `CKTdeltaOld[0..=6]`; the oldest step drops out).
    /// This is the only operation that advances the history.
    pub fn accept(&mut self, coefficients: &Coefficients) {
        let retained = self.len.min(HISTORY_LEN - 1);
        self.accepted.copy_within(0..retained, 1);
        self.accepted[0] = coefficients.dt();
        self.len = (self.len + 1).min(HISTORY_LEN);
    }

    /// Forgets every accepted step, e.g. when a run restarts.
    pub const fn clear(&mut self) {
        *self = Self {
            fill: self.fill,
            ..Self::new()
        };
    }
}

/// An exact binary64 power of two, including subnormals. The caller supplies
/// an exponent within the representable range.
fn span_power(exponent: i32) -> Real {
    if exponent >= -1022 {
        Real::from_bits(u64::from((exponent + 1023).cast_unsigned()) << 52)
    } else {
        Real::from_bits(1_u64 << (exponent + 1074).cast_unsigned())
    }
}

/// Positive step interval as a sum scaled by the largest binary exponent.
/// Summing separately avoids losing the interval between adjacent old
/// timestamps when the trial step is much larger.
fn scaled_span(steps: &[Real]) -> (Real, i32) {
    let largest = steps.iter().copied().fold(0., Real::max);
    let exponent = crate::primitives::numeric::binary_exponent(largest).unwrap_or(0);
    let scale = span_power(exponent);
    (steps.iter().map(|&step| step / scale).sum(), exponent)
}

/// Finish a scaled coefficient with one final rounding into the subnormal
/// range. The intermediate significands of at most six span ratios stay
/// bounded independently of the original step sizes.
fn finish_weight(value: Real, exponent: i32) -> Real {
    let Some(local) = crate::primitives::numeric::binary_exponent(value) else {
        return value;
    };
    let normalized = value / span_power(local);
    match exponent + local {
        ..=-1076 => 0_f64.copysign(value),
        -1075 => (normalized * 0.5) * span_power(-1074),
        e @ -1074..=1023 => normalized * span_power(e),
        _ => Real::INFINITY.copysign(value),
    }
}

/// The variable-step Gear (BDF) corrector on points at distances
/// `s_i = deltas[0] + ... + deltas[i-1]` from the trial point.
///
/// Differentiate the Lagrange basis at the trial point directly:
/// `ag_0 = sum_i 1/s_i`, `ag_i = -(1/s_i) product_{j!=i} s_j/(s_j-s_i)`.
/// Unlike the unpivoted Vandermonde solve in nicomcof.c, this does not raise
/// long/short step ratios to high powers or subtract nearly equal nodes.
/// Form differences from their original positive intervals and retain the
/// binary exponent separately until the final coefficient.
fn gear_corrector(deltas: &[Real; HISTORY_LEN + 1], order: usize) -> [Real; HISTORY_LEN + 1] {
    let mut ag = [0.; HISTORY_LEN + 1];
    let mut spans = [(0., 0); HISTORY_LEN + 1];
    for i in 1..=order {
        spans[i] = scaled_span(&deltas[..i]);
        let (mantissa, exponent) = spans[i];
        ag[0] += finish_weight(1. / mantissa, -exponent);
    }
    for i in 1..=order {
        let (span, power) = spans[i];
        let mut value = -1. / span;
        let mut exponent = -power;
        for (j, &(numerator, numerator_power)) in spans.iter().enumerate().take(order + 1).skip(1) {
            if j != i {
                let (interval, interval_power) = scaled_span(&deltas[i.min(j)..i.max(j)]);
                let denominator = if j < i { -interval } else { interval };
                value *= numerator / denominator;
                exponent += numerator_power - interval_power;
            }
        }
        ag[i] = finish_weight(value, exponent);
    }
    ag
}

/// Packed triangular table of `CKTterr`'s divided-difference denominators.
const SPAN_COUNT: usize = (HISTORY_LEN + 1) * (HISTORY_LEN + 2) / 2;

#[derive(Debug, Clone, PartialEq)]
struct TruncationSpans {
    /// Widest row first, then progressively shorter rows. The same trial
    /// coefficients are used by every charge-storage device.
    values: [Real; SPAN_COUNT],
    /// Keep unavailable history or overflowing spans from rejecting a
    /// legitimate integration trial that never requests truncation.
    error: Option<&'static str>,
}

impl TruncationSpans {
    fn new(deltas: &[Real; HISTORY_LEN + 1], order: usize) -> Self {
        let mut values = [0.0; SPAN_COUNT];
        let mut span = [0.0; HISTORY_LEN + 1];
        span[..=order].copy_from_slice(&deltas[..=order]);
        if span[..=order]
            .iter()
            .any(|dt| !dt.is_finite() || *dt <= 0.0)
        {
            return Self {
                values,
                error: Some("truncation-error steps must be positive and finite"),
            };
        }
        let mut offset = 0;
        for width in (1..=order + 1).rev() {
            values[offset..offset + width].copy_from_slice(&span[..width]);
            offset += width;
            // Preserve C's exact order: span[i + 1] + delta[i], not a
            // prefix-sum subtraction or an algebraically reassociated sum.
            for i in 0..width - 1 {
                span[i] = span[i + 1] + deltas[i];
                if !span[i].is_finite() {
                    return Self {
                        values,
                        error: Some("truncation-error span is not finite"),
                    };
                }
            }
        }
        Self {
            values,
            error: None,
        }
    }

    fn divided_difference(&self, charge: &[Real], order: usize) -> SpiceResult<Real> {
        if let Some(error) = self.error {
            return Err(cold_numerical(error));
        }
        let mut diff = [0.0; HISTORY_LEN + 2];
        diff[..order + 2].copy_from_slice(&charge[..order + 2]);
        let mut offset = 0;
        for width in (1..=order + 1).rev() {
            for (i, &span) in self.values[offset..offset + width].iter().enumerate() {
                diff[i] = finite((diff[i] - diff[i + 1]) / span, "charge divided difference")?;
            }
            offset += width;
        }
        Ok(diff[0])
    }
}

/// Immutable coefficients for one trial step.
///
/// `ag()[0]` multiplies the present charge; the rest follow C `CKTag`. They
/// are only valid together with the charge/derivative histories that were
/// accepted when the [`StepHistory`] produced them.
#[derive(Debug, Clone, PartialEq)]
pub struct Coefficients {
    method: IntegrationMethod,
    order: u8,
    /// C `CKTdeltaOld`: the trial step, then accepted steps.
    deltas: [Real; HISTORY_LEN + 1],
    history: usize,
    /// Older steps are placeholders (see [`StepHistory::with_fill`]).
    filled: bool,
    ag: [Real; HISTORY_LEN + 1],
    truncation: TruncationSpans,
}

/// One integrated charge-storage element in companion form, as
/// `NIintegrate` produces: `derivative = conductance * v + current` with
/// `conductance = ag0 * C`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Companion {
    /// The derivative of the integrated quantity at the trial point (C
    /// `CKTstate0[ccap]`): capacitor current or inductor voltage.
    pub derivative: Real,
    /// Equivalent conductance (or resistance for flux), `ag0 * capacitance`.
    pub conductance: Real,
    /// History source, `derivative - ag0 * q0`.
    pub current: Real,
}

/// Tolerances for [`Coefficients::truncation_timestep`], named as in C.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruncationTolerances {
    /// `.option reltol`.
    pub reltol: Real,
    /// `.option abstol`, applied to the derivative (current).
    pub abstol: Real,
    /// `.option chgtol`, the charge floor.
    pub chgtol: Real,
    /// `.option trtol`, the truncation-error overestimation factor.
    pub trtol: Real,
}

impl Default for TruncationTolerances {
    /// ngspice defaults (`cktntask.c`).
    fn default() -> Self {
        Self {
            reltol: 1e-3,
            abstol: 1e-12,
            chgtol: 1e-14,
            trtol: 7.0,
        }
    }
}

/// Multiply a charge difference and add in one rounding step. Opposite
/// near-maximum charges can overflow the subtraction even when the weighted
/// difference is representable. Halving both operands in that exceptional case
/// is exact; doubling the weight restores the original scale.
#[inline]
fn weighted_difference(weight: Real, left: Real, right: Real, addend: Real) -> Real {
    let difference = left - right;
    if difference.is_finite() {
        weight.mul_add(difference, addend)
    } else {
        (weight + weight).mul_add(left * 0.5 - right * 0.5, addend)
    }
}

impl Coefficients {
    /// The integration rule.
    #[must_use]
    pub const fn method(&self) -> IntegrationMethod {
        self.method
    }

    /// The order of this trial.
    #[must_use]
    pub const fn order(&self) -> u8 {
        self.order
    }

    /// The trial step size.
    #[must_use]
    pub const fn dt(&self) -> Real {
        self.deltas[0]
    }

    /// C `CKTdeltaOld[index]`: index 0 is the trial step (`CKTdelta`), 1 the
    /// last accepted step, and so on. Unfilled older slots hold zero,
    /// or the configured placeholder; `None` beyond the retained capacity.
    #[must_use]
    pub fn delta_old(&self, index: usize) -> Option<Real> {
        self.deltas.get(index).copied()
    }

    /// The corrector coefficients `ag[0..=order]` (C `CKTag`).
    #[must_use]
    pub fn ag(&self) -> &[Real] {
        &self.ag[..=usize::from(self.order)]
    }

    /// Number of accepted charge values [`Self::integrate`] reads after the
    /// trial one: the order for Gear and trapezoidal order 1, one for
    /// trapezoidal order 2.
    #[must_use]
    #[inline]
    pub fn charge_history_len(&self) -> usize {
        if self.needs_previous_derivative() {
            1
        } else {
            usize::from(self.order)
        }
    }

    /// True when [`Self::integrate`] needs the previous accepted derivative.
    #[must_use]
    #[inline]
    pub const fn needs_previous_derivative(&self) -> bool {
        matches!(self.method, IntegrationMethod::Trapezoidal) && self.order == 2
    }

    /// Integrates one element, as `NIintegrate` does.
    ///
    /// `charge` is the trial charge followed by accepted charges, most recent
    /// first; at least [`Self::charge_history_len`] accepted values are
    /// needed. `previous_derivative` is the accepted derivative at the last
    /// time point, required by trapezoidal order 2.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] for a short history, a missing derivative,
    /// nonfinite inputs/capacitance or nonfinite results.
    #[inline]
    pub fn integrate(
        &self,
        charge: &[Real],
        previous_derivative: Option<Real>,
        capacitance: Real,
    ) -> SpiceResult<Companion> {
        let needed = self.charge_history_len() + 1;
        let Some(charge) = charge.get(..needed) else {
            return Err(self.short_history(needed, charge.len()));
        };
        if charge.iter().any(|q| !q.is_finite()) || !capacitance.is_finite() {
            return Err(cold_numerical("nonfinite charge history or capacitance"));
        }
        let (derivative, current) = if self.needs_previous_derivative() {
            let Some(previous) = previous_derivative else {
                return Err(cold_numerical(
                    "trapezoidal order 2 needs the previous derivative",
                ));
            };
            finite(previous, "previous derivative")?;
            let previous = -previous * self.ag[1];
            (
                weighted_difference(self.ag[0], charge[0], charge[1], previous),
                (-self.ag[0]).mul_add(charge[1], previous),
            )
        } else if self.order == 1 {
            // The derivative of constant charge is exactly zero, regardless
            // of the magnitude of the common charge or rounded coefficient.
            (
                weighted_difference(self.ag[0], charge[0], charge[1], 0.),
                -self.ag[0] * charge[1],
            )
        } else {
            // Gear weights annihilate constants. Shift charges by the trial
            // value before the dot product to avoid catastrophic cancellation
            // of large, nearly equal absolute charges. Form the history source
            // independently, not by subtracting the rounded present term.
            let mut derivative = 0.;
            let mut current = 0.;
            for k in (1..needed).rev() {
                derivative = weighted_difference(self.ag[k], charge[k], charge[0], derivative);
                current = self.ag[k].mul_add(charge[k], current);
            }
            (derivative, current)
        };
        Ok(Companion {
            derivative: finite(derivative, "integrated derivative")?,
            conductance: finite(self.ag[0] * capacitance, "companion conductance")?,
            current: finite(current, "companion current")?,
        })
    }

    /// [`Self::integrate`]'s error for a history shorter than `needed`.
    #[cold]
    #[inline(never)]
    fn short_history(&self, needed: usize, got: usize) -> SpiceError {
        numerical(format!(
            "order {} {} integration needs {needed} charge values, got {got}",
            self.order,
            self.method.as_str(),
        ))
    }

    /// `CKTdelta / CKTdeltaOld[1]`, the ratio `DEVpred` extrapolates with.
    #[must_use]
    pub fn predictor_ratio(&self) -> Real {
        self.deltas[0] / self.deltas[1]
    }

    /// Predicts one unknown at the trial time from accepted solutions, most
    /// recent first, as `NIpred` does (C's optional `PREDICTOR` build).
    ///
    /// Trapezoidal uses the `NIpred` divided-difference formulas; Gear uses
    /// polynomial extrapolation of degree `order` (`CKTagp`). Order `k` needs
    /// `k` accepted steps and `k + 1` solutions.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] for short/nonfinite history or results.
    pub fn predict(&self, solutions: &[Real]) -> SpiceResult<Real> {
        let order = usize::from(self.order);
        if self.history < order || solutions.len() < order + 1 {
            return Err(numerical(format!(
                "order {order} prediction needs {order} accepted steps and {} solutions",
                order + 1
            )));
        }
        let x = &solutions[..=order];
        if x.iter().any(|v| !v.is_finite()) {
            return Err(numerical("nonfinite solution history"));
        }
        let d = &self.deltas;
        let predicted = match self.method {
            IntegrationMethod::Trapezoidal => {
                let dd0 = (x[0] - x[1]) / d[1];
                if order == 1 {
                    d[0].mul_add(dd0, x[0])
                } else {
                    let b = -d[0] / (2.0 * d[1]);
                    let dd1 = (x[1] - x[2]) / d[2];
                    d[0].mul_add(b.mul_add(dd1, (1.0 - b) * dd0), x[0])
                }
            }
            IntegrationMethod::Gear { .. } => {
                // Lagrange extrapolation to the trial time from nodes at
                // distance t_{n+1} - t_{n-i}.
                let mut nodes = [0.0; HISTORY_LEN + 1];
                let mut sum = 0.0;
                for (i, node) in nodes.iter_mut().take(order + 1).enumerate() {
                    sum += d[i];
                    *node = sum;
                }
                (0..=order)
                    .map(|i| {
                        let weight: Real = (0..=order)
                            .filter(|&j| j != i)
                            .map(|j| nodes[j] / (nodes[j] - nodes[i]))
                            .product();
                        weight * x[i]
                    })
                    .sum()
            }
        };
        finite(predicted, "prediction")
    }

    /// Local-truncation-error timestep bound for one element, as `CKTterr`
    /// computes it from divided differences of the charge.
    ///
    /// `charge` is the trial charge followed by at least `order + 1` accepted
    /// charges; `derivative` is `[trial, previous accepted]`. Requires `order`
    /// accepted steps. The bound is `(trtol tol / max(abstol, c_k |dd|))^(1/k)`
    /// with the order-`k` error constant `c_k` of `cktterr.c` (`gearCoeff`,
    /// `trapCoeff`) and the `(k + 1)`-th divided difference `dd` of the charge.
    /// The caller takes the minimum over all elements.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] for short/nonfinite inputs, invalid
    /// tolerances or a nonfinite result.
    pub fn truncation_timestep(
        &self,
        charge: &[Real],
        derivative: [Real; 2],
        tolerances: &TruncationTolerances,
    ) -> SpiceResult<Real> {
        let order = usize::from(self.order);
        let tol = tolerances;
        if [tol.reltol, tol.abstol, tol.chgtol, tol.trtol]
            .iter()
            .any(|v| !(v.is_finite() && *v > 0.0))
        {
            return Err(numerical(
                "truncation tolerances must be positive and finite",
            ));
        }
        if (self.history < order && !self.filled) || charge.len() < order + 2 {
            return Err(numerical(format!(
                "order {order} truncation estimate needs {order} accepted steps and {} charges",
                order + 2
            )));
        }
        if charge.iter().chain(&derivative).any(|v| !v.is_finite()) {
            return Err(numerical("nonfinite truncation-error history"));
        }
        let volttol = tol
            .reltol
            .mul_add(derivative[0].abs().max(derivative[1].abs()), tol.abstol);
        let chargetol =
            tol.reltol * charge[0].abs().max(charge[1].abs()).max(tol.chgtol) / self.dt();
        let tolerance = finite(volttol.max(chargetol), "truncation tolerance")?;
        let difference = self.truncation.divided_difference(charge, order)?;
        // cktterr.c gearCoeff/trapCoeff (trial() guarantees the index).
        let factor = match self.method {
            IntegrationMethod::Trapezoidal => TRAP_ERROR_CONSTANTS[order - 1],
            IntegrationMethod::Gear { .. } => GEAR_ERROR_CONSTANTS[order - 1],
        };
        let del = tol.trtol * tolerance / tol.abstol.max(factor * difference.abs());
        let del = match order {
            1 => del,
            2 => del.sqrt(),
            3 => del.cbrt(),
            _ => (del.ln() / Real::from(self.order)).exp(),
        };
        finite(del, "truncation timestep")
    }
}

#[cfg(test)]
mod tests {
    use super::{Coefficients, DEFAULT_XMU, IntegrationMethod, StepHistory, TruncationTolerances};

    const TRAP: IntegrationMethod = IntegrationMethod::Trapezoidal;
    const GEAR2: IntegrationMethod = IntegrationMethod::Gear { order: 2 };
    const GEAR6: IntegrationMethod = IntegrationMethod::Gear { order: 6 };

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
    }

    /// History after accepting `steps` (oldest first).
    fn history(steps: &[f64]) -> StepHistory {
        let mut history = StepHistory::new();
        for &dt in steps {
            let coefficients = history.trial(TRAP, 1, dt, DEFAULT_XMU).unwrap();
            history.accept(&coefficients);
        }
        history
    }

    /// Prepared spans reproduce the recursive divided differences.
    #[test]
    fn prepared_truncation_spans_preserve_c_order_for_every_supported_order() {
        let mut history = StepHistory::new();
        for dt in [0.071, 0.143, 0.211, 0.137, 0.293, 0.019] {
            history.accept(&history.trial(TRAP, 1, dt, DEFAULT_XMU).unwrap());
        }
        for order in 1..=6 {
            let c = history
                .trial(
                    IntegrationMethod::Gear { order: 6 },
                    order,
                    0.079,
                    DEFAULT_XMU,
                )
                .unwrap();
            let order = usize::from(order);
            for charge in [
                [0.31, 0.24, 0.97, -0.35, 0.19, 0.463, 0.291, -0.44],
                [1e18, -1e18, 0.01, 0.04, 0.06, -0.03, 0.07, 0.1],
            ] {
                let mut diff = charge;
                let mut span = c.deltas;
                for j in (0..=order).rev() {
                    for i in 0..=j {
                        diff[i] = (diff[i] - diff[i + 1]) / span[i];
                    }
                    for i in 0..j {
                        span[i] = span[i + 1] + c.deltas[i];
                    }
                }
                let actual = c.truncation.divided_difference(&charge, order).unwrap();
                assert_eq!(actual.to_bits(), diff[0].to_bits(), "order {order}");
            }
        }
    }

    #[test]
    fn gear_corrector_matches_exact_rational_weights() {
        let deltas = [0.125, 0.25, 0.1875, 0.375, 0.125, 0.3125, 0.0625];
        let expected = [
            [
                0x4020_0000_0000_0000,
                0xc020_0000_0000_0000,
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
            ],
            [
                0x4025_5555_5555_5555,
                0xc028_0000_0000_0000,
                0x3ff5_5555_5555_5555,
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
            ],
            [
                0x4028_e38e_38e3_8e39,
                0xc02e_db6d_b6db_6db7,
                0x4010_0000_0000_0000,
                0xbff0_4104_1041_0410,
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
            ],
            [
                0x402b_05b0_5b05_b05b,
                0xc031_cd5c_d5cd_5cd6,
                0x401a_aaaa_aaaa_aaab,
                0xc004_5145_1451_4514,
                0x3fc5_0150_1501_5015,
                0x0000_0000_0000_0000,
                0x0000_0000_0000_0000,
            ],
            [
                0x402c_e792_3ce7_923d,
                0xc034_2d02_d02d_02d0,
                0x4024_9b26_c9b2_6c9b,
                0xc015_9659_6596_5966,
                0x3ff6_5165_1651_6516,
                0xbfe2_7b35_63ef_927b,
                0x0000_0000_0000_0000,
            ],
            [
                0x402e_5bef_542d_63b1,
                0xc036_3183_1831_8318,
                0x402c_5555_5555_5555,
                0xc022_4424_4244_2442,
                0x4011_8918_9189_1892,
                0xc004_5454_5454_5454,
                0x3fc1_9ba5_5d3e_8e17,
            ],
        ];
        for (order, expected) in (1..=6).zip(expected) {
            let ag = super::gear_corrector(&deltas, order);
            // Independent exact rational Lagrange derivatives, rounded only
            // once to binary64. The direct span formula need not match
            // the exact rounding bitwise.
            for (actual, reference) in ag.map(f64::to_bits).into_iter().zip(expected) {
                assert!(
                    actual.abs_diff(reference) <= 4,
                    "order {order}: {actual:x} versus {reference:x}"
                );
            }
        }
    }

    #[test]
    fn gear_weights_remain_accurate_for_uneven_and_extreme_steps() {
        // Independent exact rational reference for the two-point historical
        // interval: order 2 on s1=h and s2=h+k.
        for k in [1e-15, 1e-12, 1e-6, 1e6, 1e12, 1e15] {
            let d = [1., k, 1., 1., 1., 1., 1.];
            let a = super::gear_corrector(&d, 2);
            assert!((a[0] / (1. + 1. / (1. + k)) - 1.).abs() <= 4e-16);
            assert!((a[1] / (-(1. + k) / k) - 1.).abs() <= 4e-16);
            assert!((a[2] / (1. / (k * (1. + k))) - 1.).abs() <= 4e-16);
        }
        let ordinary = super::gear_corrector(&[1.; 7], 6);
        for scale in [1e-300, 1e300, 1e308] {
            let huge = super::gear_corrector(&[scale; 7], 6);
            for (actual, expected) in huge.iter().zip(ordinary) {
                assert!(actual.is_finite());
                assert!((actual * scale - expected).abs() <= 2e-13 * (1. + expected.abs()));
            }
        }
    }

    #[test]
    fn trapezoidal_is_the_default() {
        assert_eq!(IntegrationMethod::default(), TRAP);
        assert!(TRAP.is_valid());
        assert_eq!(TRAP.max_order(), 2);
        assert!(TRAP.validate_runtime().is_ok());
    }

    #[test]
    fn gear_orders_one_to_six_run_and_others_are_rejected() {
        assert!(IntegrationMethod::Gear { order: 6 }.is_valid());
        assert!(!IntegrationMethod::Gear { order: 0 }.is_valid());
        assert!(!IntegrationMethod::Gear { order: 7 }.is_valid());
        for order in 1..=6 {
            let method = IntegrationMethod::Gear { order };
            assert!(method.validate_runtime().is_ok(), "{order}");
            // Order k needs k - 1 accepted steps.
            let warm = history(&vec![1.0; usize::from(order) - 1]);
            assert_eq!(warm.available_order(method), order);
            assert!(warm.trial(method, order, 1.0, DEFAULT_XMU).is_ok());
            if order > 1 {
                let short = history(&vec![1.0; usize::from(order) - 2]);
                assert!(short.trial(method, order, 1.0, DEFAULT_XMU).is_err());
            }
        }
        for order in [0, 7, u8::MAX] {
            let method = IntegrationMethod::Gear { order };
            let error = method.validate_runtime().unwrap_err();
            assert!(error.to_string().contains("orders 1 to 6"), "{error}");
            // An invalid method never indexes past the coefficient storage.
            assert!(
                history(&[1.0; 6])
                    .trial(method, 7, 1.0, DEFAULT_XMU)
                    .is_err()
            );
        }
        // A method's own maximum is honored, and trapezoidal stops at 2.
        assert!(
            history(&[1.0])
                .trial(IntegrationMethod::Gear { order: 1 }, 2, 1.0, DEFAULT_XMU)
                .is_err()
        );
        assert!(
            history(&[1.0; 4])
                .trial(IntegrationMethod::Gear { order: 3 }, 4, 1.0, DEFAULT_XMU)
                .is_err()
        );
        let error = history(&[1.0; 4])
            .trial(TRAP, 3, 1.0, DEFAULT_XMU)
            .unwrap_err();
        assert!(error.to_string().contains("trap 1..=2"), "{error}");
        assert_eq!(history(&[1.0; 6]).available_order(TRAP), 2);
        assert_eq!(history(&[1.0; 6]).available_order(GEAR6), 6);
        // The step history keeps exactly six accepted steps (C CKTdeltaOld[1..=6]).
        let long = history(&[7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(long.accepted(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn method_parsing_uses_the_option_spelling() {
        assert_eq!(IntegrationMethod::parse("TRAP", 2), Some(TRAP));
        assert_eq!(
            IntegrationMethod::parse("gear", 4),
            Some(IntegrationMethod::Gear { order: 4 })
        );
        assert_eq!(IntegrationMethod::parse("euler", 1), None);
        assert_eq!(GEAR2.as_str(), "gear");
        assert_eq!(TRAP.as_str(), "trap");
    }

    #[test]
    fn known_fixed_step_coefficients() {
        let h = 1e-6;
        let fresh = StepHistory::new();
        // Backward Euler startup for both methods.
        for method in [TRAP, GEAR2] {
            let c = fresh.trial(method, 1, h, DEFAULT_XMU).unwrap();
            assert!(close(c.ag()[0], 1.0 / h) && close(c.ag()[1], -1.0 / h));
        }
        let warm = history(&[h]);
        let trap = warm.trial(TRAP, 2, h, DEFAULT_XMU).unwrap();
        assert!(close(trap.ag()[0], 2.0 / h) && close(trap.ag()[1], 1.0));
        let damped = warm.trial(TRAP, 2, h, 0.25).unwrap();
        assert!(close(damped.ag()[0], 1.0 / h / 0.75) && close(damped.ag()[1], 1.0 / 3.0));
        let gear = warm.trial(GEAR2, 2, h, DEFAULT_XMU).unwrap();
        let expected = [1.5 / h, -2.0 / h, 0.5 / h];
        assert!(gear.ag().iter().zip(expected).all(|(a, b)| close(*a, b)));
    }

    /// Gear corrector and predictor reproduce polynomial derivatives/values
    /// exactly up to their order, on nonuniform steps.
    #[test]
    fn gear_is_exact_for_polynomials_on_nonuniform_steps() {
        type Poly = (fn(f64) -> f64, fn(f64) -> f64, u8);
        let polys: [Poly; 3] = [
            (|_| 3.0, |_| 0.0, 1),
            (|t| 2.0 * t - 1.0, |_| 2.0, 1),
            (|t| t * t - 3.0 * t + 0.5, |t| 2.0 * t - 3.0, 2),
        ];
        let (h1, h) = (0.3, 0.7);
        let warm = history(&[0.2, h1]);
        let times = [1.0 + h, 1.0, 1.0 - h1, 1.0 - h1 - 0.2];
        for (f, df, degree) in polys {
            for order in degree..=2 {
                let c = warm.trial(GEAR2, order, h, DEFAULT_XMU).unwrap();
                let q: Vec<f64> = times.iter().map(|t| f(*t)).collect();
                let companion = c.integrate(&q, None, 1.0).unwrap();
                assert!(close(companion.derivative, df(times[0])), "order {order}");
                assert!(close(c.predict(&q[1..]).unwrap(), q[0]), "order {order}");
            }
        }
    }

    /// Trapezoidal order 2 integrates the mean of the endpoint derivatives,
    /// exact for quadratics; order 1 is backward Euler.
    #[test]
    fn trapezoidal_integration_matches_its_rule() {
        let (h1, h) = (0.4, 0.25);
        let warm = history(&[h1]);
        let f = |t: f64| t * t + t;
        let df = |t: f64| 2.0 * t + 1.0;
        let c = warm.trial(TRAP, 2, h, DEFAULT_XMU).unwrap();
        assert!(c.needs_previous_derivative());
        assert_eq!(c.charge_history_len(), 1);
        let q = [f(1.0 + h), f(1.0)];
        let companion = c.integrate(&q, Some(df(1.0)), 2.0).unwrap();
        assert!(close(companion.derivative, df(1.0 + h)));
        assert!(close(companion.conductance, 2.0 * 2.0 / h));
        assert!(close(
            companion.current,
            companion.derivative - c.ag()[0] * q[0]
        ));
        let be = warm.trial(TRAP, 1, h, DEFAULT_XMU).unwrap();
        let companion = be.integrate(&q, None, 1.0).unwrap();
        assert!(close(companion.derivative, (q[0] - q[1]) / h));
        // NIpred's trapezoidal predictors extrapolate lines exactly on
        // nonuniform steps.
        let warm = history(&[0.3, h1]);
        let times = [1.0 + h, 1.0, 1.0 - h1, 1.0 - h1 - 0.3];
        let line: Vec<f64> = times.iter().map(|t| 4.0 * t - 1.0).collect();
        for order in [1, 2] {
            let c = warm.trial(TRAP, order, h, DEFAULT_XMU).unwrap();
            assert!(close(c.predict(&line[1..]).unwrap(), line[0]));
        }
    }

    #[test]
    fn charge_derivative_is_invariant_under_a_common_offset() {
        let warm = history(&[0.125; 6]);
        for method in [TRAP, GEAR6] {
            for order in 1..=method.max_order() {
                let c = warm.trial(method, order, 0.125, DEFAULT_XMU).unwrap();
                let constant = [1e16; 7];
                let zero = c.integrate(&constant, Some(0.), 1.).unwrap();
                assert_eq!(zero.derivative, 0., "{method:?} order {order}");
                // The unit-spaced charges remain exactly representable after
                // adding this offset. The affine difference is invariant.
                let charge = [12., 10., 8., 6., 4., 2., 0.];
                let shifted = charge.map(|q| q + 1e15);
                let a = c.integrate(&charge, Some(16.), 1.).unwrap();
                let b = c.integrate(&shifted, Some(16.), 1.).unwrap();
                assert_eq!(a.derivative.to_bits(), b.derivative.to_bits());
                assert!(close(a.derivative, 16.), "{method:?} order {order}");
            }
        }
    }

    #[test]
    fn representable_derivatives_survive_overflowing_charge_differences() {
        let warm = history(&[4.; 6]);
        let charge = [f64::MAX, -f64::MAX];
        for method in [TRAP, GEAR6] {
            let c = warm.trial(method, 1, 4., DEFAULT_XMU).unwrap();
            let result = c.integrate(&charge, None, 1.).unwrap();
            assert_eq!(result.derivative, f64::MAX * 0.5);
            assert_eq!(result.current, f64::MAX * 0.25);
        }
        let c = warm.trial(TRAP, 2, 4., DEFAULT_XMU).unwrap();
        let result = c.integrate(&charge, Some(0.), 1.).unwrap();
        assert_eq!(result.derivative, f64::MAX);
        assert_eq!(result.current, f64::MAX * 0.5);
    }

    #[test]
    fn startup_and_order_changes_follow_the_accepted_history() {
        let mut history = StepHistory::new();
        assert_eq!(history.available_order(GEAR2), 1);
        assert_eq!(history.available_order(TRAP), 1);
        assert!(history.trial(GEAR2, 2, 1e-6, DEFAULT_XMU).is_err());
        let first = history.trial(GEAR2, 1, 1e-6, DEFAULT_XMU).unwrap();
        history.accept(&first);
        assert_eq!(history.available_order(GEAR2), 2);
        assert_eq!(
            history.available_order(IntegrationMethod::Gear { order: 1 }),
            1
        );
        let second = history.trial(GEAR2, 2, 2e-6, DEFAULT_XMU).unwrap();
        // Drop back to order 1 (as after a breakpoint) with history intact.
        let back = history.trial(GEAR2, 1, 5e-7, DEFAULT_XMU).unwrap();
        assert_eq!(back.order(), 1);
        history.accept(&second);
        history.accept(&back);
        history.accept(&back);
        assert_eq!(history.accepted(), &[5e-7, 5e-7, 2e-6, 1e-6]);
        assert_eq!(history.len(), 4);
        history.clear();
        assert!(history.is_empty());
    }

    #[test]
    fn rejected_trials_never_advance_the_history() {
        let accepted = history(&[1e-6, 2e-6]);
        let before = accepted.clone();
        for dt in [1e-6, 1e-9, 5e-6] {
            let c = accepted.trial(GEAR2, 2, dt, DEFAULT_XMU).unwrap();
            let _ = c.integrate(&[1.0, 0.5, 0.2], None, 1e-9).unwrap();
        }
        assert!(accepted.trial(TRAP, 2, -1.0, DEFAULT_XMU).is_err());
        assert_eq!(accepted, before);
    }

    #[test]
    fn invalid_inputs_are_rejected_without_panic() {
        let warm = history(&[1e-6]);
        for dt in [0.0, -1e-6, f64::NAN, f64::INFINITY] {
            let error = warm.trial(TRAP, 1, dt, DEFAULT_XMU).unwrap_err();
            assert!(error.to_string().contains("must be positive and finite"));
        }
        assert!(warm.trial(TRAP, 0, 1e-6, DEFAULT_XMU).is_err());
        for xmu in [-0.1, 0.6, f64::NAN] {
            assert!(warm.trial(TRAP, 2, 1e-6, xmu).is_err());
        }
        // 1/dt overflows for subnormal steps.
        let error = warm.trial(TRAP, 1, 1e-320, DEFAULT_XMU).unwrap_err();
        assert!(error.to_string().contains("not finite"), "{error}");

        let trap = warm.trial(TRAP, 2, 1e-6, DEFAULT_XMU).unwrap();
        assert!(trap.integrate(&[1.0], Some(0.0), 1.0).is_err());
        assert!(trap.integrate(&[1.0, 0.0], None, 1.0).is_err());
        assert!(trap.integrate(&[f64::NAN, 0.0], Some(0.0), 1.0).is_err());
        assert!(
            trap.integrate(&[1.0, 0.0], Some(0.0), f64::INFINITY)
                .is_err()
        );
        assert!(
            trap.integrate(&[f64::MAX, -f64::MAX], Some(0.0), 1.0)
                .is_err()
        );
        let gear = warm.trial(GEAR2, 2, 1e-6, DEFAULT_XMU).unwrap();
        assert!(gear.integrate(&[1.0, 0.0], None, 1.0).is_err());
        assert!(gear.predict(&[1.0, 2.0, 3.0]).is_err(), "needs 2 steps");
        assert!(trap.predict(&[1.0]).is_err());
        assert!(trap.predict(&[1.0, f64::NAN, 0.0]).is_err());
    }

    /// Direct transcription of the order-1/2 `CKTterr` divided differences.
    fn reference_terr(c: &Coefficients, q: &[f64], d: [f64; 2], t: &TruncationTolerances) -> f64 {
        let h = &c.deltas;
        let volttol = t.abstol + t.reltol * d[0].abs().max(d[1].abs());
        let chargetol = t.reltol * q[0].abs().max(q[1].abs()).max(t.chgtol) / c.dt();
        let tol = volttol.max(chargetol);
        let dd = |a: f64, b: f64, h: f64| (a - b) / h;
        let (d0, d1) = (dd(q[0], q[1], h[0]), dd(q[1], q[2], h[1]));
        let e0 = dd(d0, d1, h[0] + h[1]);
        let (diff, factor) = if c.order() == 1 {
            (e0, 0.5)
        } else {
            let d2 = dd(q[2], q[3], h[2]);
            let e1 = dd(d1, d2, h[1] + h[2]);
            let factor = if c.method() == TRAP {
                0.083_333_333_33
            } else {
                0.222_222_222_2
            };
            (dd(e0, e1, h[0] + h[1] + h[2]), factor)
        };
        let del = t.trtol * tol / t.abstol.max(factor * diff.abs());
        if c.order() == 2 { del.sqrt() } else { del }
    }

    #[test]
    fn truncation_timestep_follows_cktterr() {
        let tol = TruncationTolerances::default();
        let warm = history(&[2e-6, 1e-6]);
        let q = [3.1e-9, 2.0e-9, 1.6e-9, 1.5e-9];
        let d = [1e-3, 8e-4];
        for (method, order) in [(TRAP, 1), (TRAP, 2), (GEAR2, 1), (GEAR2, 2)] {
            let c = warm.trial(method, order, 5e-7, DEFAULT_XMU).unwrap();
            let del = c.truncation_timestep(&q, d, &tol).unwrap();
            assert!(
                close(del, reference_terr(&c, &q, d, &tol)),
                "{method:?} {order}"
            );
            assert!(del > 0.0);
        }
        let c = warm.trial(GEAR2, 2, 5e-7, DEFAULT_XMU).unwrap();
        // Short history/charges and invalid tolerances fail.
        assert!(c.truncation_timestep(&q[..3], d, &tol).is_err());
        let fresh = history(&[1e-6]).trial(TRAP, 2, 1e-6, DEFAULT_XMU).unwrap();
        assert!(fresh.truncation_timestep(&q, d, &tol).is_err());
        let bad = TruncationTolerances { trtol: 0.0, ..tol };
        assert!(c.truncation_timestep(&q, d, &bad).is_err());
        assert!(c.truncation_timestep(&q, [f64::NAN, 0.0], &tol).is_err());
    }

    #[test]
    fn truncation_rejects_invalid_spans_and_hidden_nonfinite_differences() {
        let tolerance = TruncationTolerances::default();
        for fill in [0., -1., f64::NAN, f64::INFINITY] {
            let trial = StepHistory::with_fill(fill)
                .trial(TRAP, 1, 1., DEFAULT_XMU)
                .unwrap();
            assert!(
                trial
                    .truncation_timestep(&[0., 0., 0.], [0., 0.], &tolerance)
                    .is_err()
            );
        }
        let trial = history(&[f64::MAX])
            .trial(TRAP, 1, f64::MAX, DEFAULT_XMU)
            .unwrap();
        assert!(
            trial
                .truncation_timestep(&[1., 1., 1.], [0., 0.], &tolerance)
                .is_err()
        );
        let trial = history(&[0.5]).trial(TRAP, 1, 0.5, DEFAULT_XMU).unwrap();
        // Two overflowing first differences used to produce NaN, which
        // f64::max(abstol, NaN) silently replaced with abstol.
        let relaxed = TruncationTolerances {
            abstol: 1e300,
            ..tolerance
        };
        assert!(
            trial
                .truncation_timestep(&[f64::MAX, 0., -f64::MAX], [0., 0.], &relaxed)
                .is_err()
        );
        assert!(
            trial
                .truncation_timestep(&[f64::MAX, -f64::MAX, -f64::MAX], [0., 0.], &relaxed)
                .is_err()
        );
        assert!(
            trial
                .truncation_timestep(&[1., 0., -1.], [0., 0.], &tolerance)
                .unwrap()
                > 0.
        );
    }

    #[test]
    fn filled_history_supplies_placeholder_steps_for_truncation_only() {
        // dctran.c fills CKTdeltaOld with CKTmaxStep, so the second step can
        // already probe an order-2 truncation estimate.
        let tol = TruncationTolerances::default();
        let q = [3.1e-9, 2.0e-9, 1.6e-9, 1.5e-9];
        let d = [1e-3, 8e-4];
        let mut filled = StepHistory::with_fill(1e-4);
        let mut plain = StepHistory::new();
        for history in [&mut filled, &mut plain] {
            let first = history.trial(TRAP, 1, 1e-6, DEFAULT_XMU).unwrap();
            history.accept(&first);
        }
        let probe = |h: &StepHistory| h.trial(TRAP, 2, 5e-7, DEFAULT_XMU).unwrap();
        assert!(probe(&plain).truncation_timestep(&q, d, &tol).is_err());
        let with_fill = probe(&filled).truncation_timestep(&q, d, &tol).unwrap();
        // The older step is the placeholder (1e-4) after one real step (1e-6).
        let reference = history(&[1e-4, 1e-6])
            .trial(TRAP, 2, 5e-7, DEFAULT_XMU)
            .unwrap();
        assert!(close(
            with_fill,
            reference.truncation_timestep(&q, d, &tol).unwrap()
        ));
        // Prediction still needs real accepted steps, and clear() keeps the fill.
        assert!(probe(&filled).predict(&[1.0, 0.5, 0.25]).is_err());
        filled.clear();
        assert!(filled.is_empty());
        assert!(probe_after_clear(&filled));
    }

    fn probe_after_clear(history: &StepHistory) -> bool {
        // Order 2 is unavailable without a real step, filled or not.
        history.trial(TRAP, 2, 5e-7, DEFAULT_XMU).is_err()
            && history.trial(TRAP, 1, 5e-7, DEFAULT_XMU).is_ok()
    }

    /// Nonuniform accepted steps (oldest first) and the trial step used by the
    /// order-3..6 tests: ratios up to 3.5 between neighbours.
    const OLD_STEPS: [f64; 6] = [0.11, 0.35, 0.1, 0.2, 0.15, 0.3];
    const TRIAL_STEP: f64 = 0.25;

    /// Time points `t_{n+1}, t_n, ..., t_{n-6}` for [`OLD_STEPS`], most recent
    /// first, with `t_n = 1`.
    fn nonuniform_times() -> [f64; 8] {
        let mut times = [0.0; 8];
        times[0] = 1.0 + TRIAL_STEP;
        times[1] = 1.0;
        for (i, dt) in OLD_STEPS.iter().rev().enumerate() {
            times[i + 2] = times[i + 1] - dt;
        }
        times
    }

    /// Textbook fixed-step BDF coefficients (times `h`) for orders 1 to 6.
    #[test]
    fn gear_matches_the_fixed_step_bdf_tables() {
        let tables: [&[f64]; 6] = [
            &[1.0, -1.0],
            &[1.5, -2.0, 0.5],
            &[11.0 / 6.0, -3.0, 1.5, -1.0 / 3.0],
            &[25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 0.25],
            &[137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 1.25, -0.2],
            &[147.0 / 60.0, -6.0, 7.5, -20.0 / 3.0, 3.75, -1.2, 1.0 / 6.0],
        ];
        let h = 2.5e-7;
        let warm = history(&[h; 6]);
        for (order, table) in (1..=6).zip(tables) {
            let c = warm.trial(GEAR6, order, h, DEFAULT_XMU).unwrap();
            assert_eq!(c.ag().len(), usize::from(order) + 1);
            for (a, b) in c.ag().iter().zip(table) {
                assert!((a * h - b).abs() <= 1e-12, "order {order}: {a} vs {b}/h");
            }
        }
    }

    /// An independent solve of the `nicomcof.c` system: `sum_i ag_i
    /// (s_i/h)^j = -[j == 1]/h` by Gaussian elimination with partial pivoting.
    fn vandermonde_ag(deltas: &[f64], order: usize) -> Vec<f64> {
        let n = order + 1;
        let h = deltas[0];
        let mut s = vec![0.0; n];
        for i in 1..n {
            s[i] = s[i - 1] + deltas[i - 1];
        }
        let mut m: Vec<Vec<f64>> = (0..n)
            .map(|j| {
                let mut row: Vec<f64> = s.iter().map(|si| (si / h).powi(j as i32)).collect();
                row.push(if j == 1 { -1.0 / h } else { 0.0 });
                row
            })
            .collect();
        for k in 0..n {
            let pivot = (k..n)
                .max_by(|&a, &b| m[a][k].abs().total_cmp(&m[b][k].abs()))
                .unwrap();
            m.swap(k, pivot);
            let pivot_row = m[k].clone();
            for row in m.iter_mut().skip(k + 1) {
                let f = row[k] / pivot_row[k];
                for (value, p) in row.iter_mut().zip(&pivot_row).skip(k) {
                    *value -= f * p;
                }
            }
        }
        let mut x = vec![0.0; n];
        for k in (0..n).rev() {
            let tail: f64 = (k + 1..n).map(|c| m[k][c] * x[c]).sum();
            x[k] = (m[k][n] - tail) / m[k][k];
        }
        x
    }

    /// Orders 1..6 on nonuniform steps: the corrector reproduces the derivative
    /// and the predictor the value of every polynomial of degree `order`, and
    /// the coefficients agree with the C Vandermonde formulation.
    #[test]
    fn high_order_gear_is_exact_for_polynomials_on_nonuniform_steps() {
        let warm = history(&OLD_STEPS);
        let times = nonuniform_times();
        for order in 1..=6_u8 {
            let k = i32::from(order);
            // Degree `order` with nonzero lower terms, centred to keep the
            // values O(1).
            let f = |t: f64| {
                (0..=k)
                    .map(|p| (t - 0.6).powi(p) / f64::from(p + 1))
                    .sum::<f64>()
            };
            let df = |t: f64| {
                (1..=k)
                    .map(|p| f64::from(p) * (t - 0.6).powi(p - 1) / f64::from(p + 1))
                    .sum::<f64>()
            };
            let c = warm.trial(GEAR6, order, TRIAL_STEP, DEFAULT_XMU).unwrap();
            let q: Vec<f64> = times.iter().map(|t| f(*t)).collect();
            assert_eq!(c.charge_history_len(), usize::from(order));
            let companion = c.integrate(&q, None, 1.0).unwrap();
            let close6 = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + b.abs());
            assert!(close6(companion.derivative, df(times[0])), "order {order}");
            assert!(close6(c.predict(&q[1..]).unwrap(), q[0]), "order {order}");
            let reference = vandermonde_ag(&c.deltas, usize::from(order));
            for (a, b) in c.ag().iter().zip(&reference) {
                assert!(
                    (a - b).abs() <= 1e-10 * b.abs().max(1.0 / TRIAL_STEP),
                    "order {order}"
                );
            }
            // One degree more is not reproduced: the rule really has order k.
            let g = |t: f64| (t - 0.6).powi(k + 1);
            let q: Vec<f64> = times.iter().map(|t| g(*t)).collect();
            let derivative = c.integrate(&q, None, 1.0).unwrap().derivative;
            let exact = f64::from(k + 1) * (times[0] - 0.6).powi(k);
            assert!((derivative - exact).abs() > 1e-6, "order {order}");
        }
        // Short charge or solution histories are errors, not panics.
        let c = warm.trial(GEAR6, 6, TRIAL_STEP, DEFAULT_XMU).unwrap();
        assert!(c.integrate(&[0.0; 6], None, 1.0).is_err());
        assert!(c.predict(&[0.0; 6]).is_err());
        assert!(
            c.truncation_timestep(&[0.0; 7], [0.0; 2], &TruncationTolerances::default())
                .is_err()
        );
    }

    /// `CKTterr` for orders 1..6: on a polynomial of degree `order + 1` with
    /// leading coefficient `a` the `(order + 1)`-th divided difference is `a`,
    /// so the bound is `(trtol tol / max(abstol, c_k |a|))^(1/k)` exactly.
    #[test]
    fn high_order_truncation_timestep_follows_cktterr() {
        let constants = [
            0.5,
            0.222_222_222_2,
            0.136_363_636_4,
            0.096,
            0.072_992_700_73,
            0.058_309_037_90,
        ];
        let tol = TruncationTolerances::default();
        let warm = history(&OLD_STEPS);
        let times = nonuniform_times();
        for order in 1..=6_u8 {
            let k = i32::from(order);
            let a = 3.0e-3;
            let q: Vec<f64> = times.iter().map(|t| a * t.powi(k + 1) + 1e-9 * t).collect();
            let d = [2e-3, 1e-3];
            let c = warm.trial(GEAR6, order, TRIAL_STEP, DEFAULT_XMU).unwrap();
            let del = c.truncation_timestep(&q, d, &tol).unwrap();
            let volttol = tol.abstol + tol.reltol * 2e-3;
            let chargetol = tol.reltol * q[0].abs().max(q[1].abs()).max(tol.chgtol) / TRIAL_STEP;
            let base = tol.trtol * volttol.max(chargetol)
                / tol.abstol.max(constants[usize::from(order) - 1] * a);
            let expected = base.powf(1.0 / f64::from(k));
            assert!(
                (del - expected).abs() <= 1e-9 * expected,
                "order {order}: {del} vs {expected}"
            );
        }
    }

    /// Fixed-step convergence: Gear order `k` integrating `q' = -q` over
    /// `[0, 1]` from exact starting values loses a factor of about `2^k` in
    /// global error per halving of every step, on a nonuniform repeating step
    /// pattern (so the variable-step coefficients are exercised, not just the
    /// fixed-step table).
    #[test]
    fn gear_orders_converge_at_their_order() {
        let pattern = [1.0, 0.7, 1.3];
        let solve = |order: u8, base: f64| -> f64 {
            // Exact history at t = 0 and the order - 1 earlier points (steps
            // from the pattern), most recent first; older steps are accepted
            // first.
            let dts: Vec<f64> = (0..usize::from(order) - 1)
                .map(|i| base * pattern[(i + 1) % 3])
                .collect();
            let mut steps = StepHistory::new();
            for &dt in dts.iter().rev() {
                let c = steps.trial(TRAP, 1, dt, DEFAULT_XMU).unwrap();
                steps.accept(&c);
            }
            let mut t = 0.0;
            let mut q = vec![1.0];
            for dt in &dts {
                t -= dt;
                q.push((-t).exp());
            }
            let mut now = 0.0;
            let mut i = 0;
            while now < 1.0 - 1e-12 {
                let dt = (base * pattern[i % 3]).min(1.0 - now);
                i += 1;
                let c = steps.trial(GEAR6, order, dt, DEFAULT_XMU).unwrap();
                // derivative = ag0 q0 + history; with q0 = 0 the companion
                // current is the history part, so ag0 q0 + current = -q0.
                let mut trial = vec![0.0];
                trial.extend_from_slice(&q);
                let companion = c.integrate(&trial, None, 1.0).unwrap();
                let q0 = -companion.current / (companion.conductance + 1.0);
                q.insert(0, q0);
                q.truncate(8);
                steps.accept(&c);
                now += dt;
            }
            (q[0] - (-1.0_f64).exp()).abs()
        };
        for order in 1..=6_u8 {
            // Coarser for high orders so the finest error stays far above
            // rounding (about 1e-13 at order 6).
            let base = [0.04, 0.04, 0.04, 0.06, 0.08, 0.12][usize::from(order) - 1];
            let errors: Vec<f64> = (0..3)
                .map(|n| solve(order, base / f64::from(1 << n)))
                .collect();
            let expected = f64::from(1_u32 << order);
            for pair in errors.windows(2) {
                let ratio = pair[0] / pair[1];
                assert!(
                    ratio > 0.75 * expected && ratio < 1.35 * expected,
                    "order {order}: errors {errors:?}, ratio {ratio} (expected about {expected})"
                );
            }
        }
    }
}
