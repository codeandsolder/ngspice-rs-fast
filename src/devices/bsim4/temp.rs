//! `b4temp.c` (`BSIM4temp`): the model's temperature pass, the instance's
//! size-dependent parameters and its temperature-dependent instance values,
//! then `b4check.c`'s parameter check.
//!
//! Ported statement by statement from BSIM4 4.8.3 (`b4temp.c`, Copyright
//! (c) 2025 University of California, Educational Community License 2.0);
//! C's member and local names are kept so the port can be read against it.
// Preserve C model branches and ordered setup; cancellation-sensitive
// physical equations use the conditioned laws documented in BSIM_TEMPERATURE.md.
#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols keep their original distinct names for source-level audit"
)]
#![expect(
    clippy::float_cmp,
    reason = "C device branches compare floating-point values exactly, including nonfinite cases"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "physical-model equation symbols are kept consistent with the ngspice C reference"
)]
#![expect(
    clippy::while_float,
    reason = "the C iterative floating-point loop must preserve its original termination and rounding order"
)]
#![expect(
    non_snake_case,
    reason = "preserves upstream C identifiers for equation-by-equation comparison"
)]
#![expect(
    non_upper_case_globals,
    reason = "constant names intentionally match the reference device equations"
)]

use super::check::check_model;
use super::fields::{Instance, Model, SizeParam};
use super::geo::{pa_eff_geo, parallel_resistance, rdseff_geo};
use crate::primitives::numeric::{scale_power, split_power};
use crate::primitives::{SpiceError, SpiceResult};

const KboQ: f64 = 8.617_087e-5;
const EPS0: f64 = 8.85418e-12;
const EPSSI: f64 = 1.03594e-10;
// b4temp.c's own rounded PI, not the platform's.
const PI: f64 = f64::from_bits(0x4009_21fb_5452_4550);
const MAX_EXP: f64 = 5.834_617_425e14;
const MIN_EXP: f64 = 1.713_908_431e-15;
const EXP_THRESHOLD: f64 = 34.0;
const Charge_q: f64 = 1.60219e-19;
const DELTA: f64 = 1.0E-9;
/// `const.h` `CHARGE`.
const CHARGE: f64 = 1.602_176_620_8e-19;
/// `main.c`: `CONSTroot2 = CONSTsqrt2`.
const CONSTroot2: f64 = std::f64::consts::SQRT_2;
/// `main.c`: `CONSTvt0 = CONSTboltz * REFTEMP / CHARGE`.
const CONSTvt0: f64 = 1.380_648_52e-23 * (27.0 + 273.15) / CHARGE;
/// `bsim4def.h` `NMOS`.
const NMOS: i32 = 1;
/// `bsim4def.h` `BSIM4v48V480`.
const BSIM4v48V480: i32 = 480;

// C's libm calls, under C's names so the port reads as the C does.
fn exp(x: f64) -> f64 {
    x.exp()
}
fn log(x: f64) -> f64 {
    x.ln()
}
fn sqrt(x: f64) -> f64 {
    x.sqrt()
}
fn pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}
const fn fabs(x: f64) -> f64 {
    x.abs()
}
fn atan(x: f64) -> f64 {
    x.atan()
}
/// `ABS` from `ngspice.h`.
#[expect(
    clippy::suboptimal_flops,
    reason = "C ABS retains negative zero and negates a NaN instead of clearing its sign bit"
)]
fn ABS(x: f64) -> f64 {
    if x >= 0.0 { x } else { -x }
}

/// b4temp.c's two-argument `DEXP(A, B)`: `B = dexp(A)`.
fn dexp(a: f64) -> f64 {
    if a > EXP_THRESHOLD {
        MAX_EXP * (1.0 + a - EXP_THRESHOLD)
    } else if a < -EXP_THRESHOLD {
        MIN_EXP
    } else {
        exp(a)
    }
}

/// Difference Eg(Tnom)/(k*Tnom) - Eg(T)/(k*T), for
/// Eg(T) = e0 - alpha*T²/(T+beta). Factor the temperature increment
/// before rounding instead of subtracting two nearly equal ratios.
fn thermal_bandgap_difference(
    nominal: f64,
    temperature: f64,
    e0: f64,
    alpha: f64,
    beta: f64,
) -> f64 {
    let delta = temperature - nominal;
    let inverse = delta / nominal.max(temperature) / nominal.min(temperature);
    let material = (beta / (temperature + beta)) * (delta / (nominal + beta));
    e0.mul_add(inverse, alpha * material) / KboQ
}

fn relative_temperature_change(temperature: f64, nominal: f64) -> f64 {
    if temperature > 0. && nominal > 0. && temperature.is_finite() && nominal.is_finite() {
        (temperature - nominal) / nominal
    } else {
        temperature / nominal - 1.
    }
}

fn log_temperature_ratio(temperature: f64, nominal: f64) -> f64 {
    let relative = relative_temperature_change(temperature, nominal);
    if relative.abs() < 0.5 {
        relative.ln_1p()
    } else {
        let ratio = temperature / nominal;
        if (ratio == 0. || ratio.is_infinite())
            && temperature > 0.
            && nominal > 0.
            && temperature.is_finite()
            && nominal.is_finite()
        {
            temperature.ln() - nominal.ln()
        } else {
            ratio.ln()
        }
    }
}

/// Multiply by exp(exponent) without materializing an overflowing or
/// subnormal exponential. Only the exceptional path uses four monotone
/// products by exp(exponent/4); ordinary density scaling still shares one
/// exponential per source/drain side.
struct ExpScale {
    factor: f64,
    quarter: Option<f64>,
    finite_exponent: bool,
}

impl ExpScale {
    fn new(exponent: f64) -> Self {
        let factor = exponent.exp();
        let finite_exponent = exponent.is_finite();
        let quarter = if finite_exponent && !factor.is_normal() {
            let candidate = (0.25 * exponent).exp();
            candidate.is_normal().then_some(candidate)
        } else {
            None
        };
        Self {
            factor,
            quarter,
            finite_exponent,
        }
    }

    fn apply(&self, value: f64) -> f64 {
        if !value.is_finite() || !self.finite_exponent {
            return value * self.factor;
        }
        if value == 0. {
            return value;
        }
        self.quarter.map_or(value * self.factor, |quarter| {
            ((value * quarter) * quarter) * quarter * quarter
        })
    }
}

/// Rational well-proximity term, scaled only if a product is out of range.
fn well_proximity_rational(distance: f64, width: f64, reference: f64) -> f64 {
    let numerator = reference * reference;
    let denominator = distance * (distance + width);
    if (numerator.is_normal() && denominator.is_normal())
        || !(distance > 0.
            && width > 0.
            && reference > 0.
            && distance.is_finite()
            && width.is_finite()
            && reference.is_finite())
    {
        return numerator / denominator;
    }
    let largest = distance.max(width);
    let (r, re) = split_power(reference);
    let (d, de) = split_power(distance);
    let (m, me) = split_power(largest);
    let sum = m * (1. + distance.min(width) / largest);
    let numerator = r * r;
    let denominator = d * sum;
    scale_power(numerator / denominator, 2 * re - de - me)
}

/// Mean of x exp(-x) over a dimensionless interval, scaled by the WPE
/// coefficient. Integrate analytically, then use expm1 and a convergent
/// small-width series for the second integral instead of subtracting
/// nearly identical endpoint antiderivatives.
#[expect(
    clippy::suboptimal_flops,
    reason = "the nonphysical-input endpoint fallback preserves C IEEE arithmetic"
)]
fn well_proximity_exp(distance: f64, width: f64, reference: f64, rate: f64, scale: f64) -> f64 {
    if !(distance > 0.
        && width > 0.
        && reference > 0.
        && distance.is_finite()
        && width.is_finite()
        && reference.is_finite())
    {
        let far = distance + width;
        let inverse = 1. / reference;
        let intercept = scale / rate * reference;
        return ((scale * distance + intercept) * (-rate * distance * inverse).exp()
            - (scale * far + intercept) * (-rate * far * inverse).exp())
            / width;
    }
    let x = rate * (distance / reference);
    let z = rate * (width / reference);
    if x.is_infinite() {
        return 0.;
    }
    let (first, second) = if z == 0. {
        (1., 0.)
    } else if z.is_infinite() {
        let reciprocal = reference / width / rate;
        (reciprocal, reciprocal)
    } else {
        let one_minus = -(-z).exp_m1();
        let second = if z <= 0.5 {
            let mut term = 0.5 * z;
            let mut total = term;
            for order in 1..=16 {
                let n = f64::from(order);
                term *= -z * (n + 1.) / (n * (n + 2.));
                let next = total + term;
                if next == total {
                    break;
                }
                total = next;
            }
            total
        } else {
            (-z).mul_add((-z).exp(), one_minus) / z
        };
        (one_minus / z, second)
    };
    ExpScale::new(-x).apply(scale * x.mul_add(first, second))
}

/// The expm1 form preserves the small argument of the short-channel
/// factor; its large-argument clamp and nonfinite branch are unchanged.
fn short_channel_factor(argument: f64) -> f64 {
    if argument < EXP_THRESHOLD {
        let (exponential, difference) = if argument.abs() < 0.5 {
            let difference = argument.exp_m1();
            (1. + difference, difference)
        } else {
            let exponential = argument.exp();
            (exponential, exponential - 1.)
        };
        exponential / difference.mul_add(difference, 2. * exponential * MIN_EXP)
    } else {
        1. / (MAX_EXP - 2.)
    }
}

/// The denominator is identically -(sqrt(phi-vbm)-sqrt(phi))².
/// Rationalize both differences and retain the singular/invalid input path.
#[expect(
    clippy::suboptimal_flops,
    reason = "the invalid/singular body-effect fallback preserves C IEEE propagation"
)]
fn body_coefficient(phi: f64, vbx: f64, vbm: f64, gamma_difference: f64) -> f64 {
    let root = phi.sqrt();
    if phi > 0. && phi.is_finite() && vbx.is_finite() && vbm.is_finite() && vbx <= 0. && vbm < 0. {
        let numerator = -vbx / ((phi - vbx).sqrt() + root);
        let denominator_root = -vbm / ((phi - vbm).sqrt() + root);
        -(gamma_difference * (numerator / denominator_root)) / denominator_root
    } else {
        gamma_difference * ((phi - vbx).sqrt() - root)
            / (2. * ((phi * (phi - vbm)).sqrt() - phi) + vbm)
    }
}

fn sqrt_one_plus_difference(x: f64) -> f64 {
    if x.abs() < 0.5 {
        x / ((1. + x).sqrt() + 1.)
    } else {
        (1. + x).sqrt() - 1.
    }
}

fn physical_diode_limit(nvtm: f64, threshold: f64, saturation: f64, breakdown: f64) -> bool {
    threshold >= 0.
        && threshold.is_finite()
        && saturation > 0.
        && saturation.is_finite()
        && breakdown >= 0.
        && breakdown.is_finite()
        && nvtm > 0.
        && nvtm.is_finite()
}

/// `BSIM4DioIjthVjmEval`: solve in z=exp(Vjm/Nvtm)-1.
/// z²+(1+XExpBV-Ijth/Isb)z-Ijth/Isb=0. Rationalize the positive
/// root and use hypot to avoid squaring a large coefficient.
#[expect(
    clippy::manual_midpoint,
    reason = "the invalid diode input path preserves C intermediate overflow"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "the invalid diode input path preserves C IEEE propagation"
)]
fn dio_ijth_vjm_eval(nvtm: f64, threshold: f64, saturation: f64, breakdown: f64) -> f64 {
    let ratio = threshold / saturation;
    if physical_diode_limit(nvtm, threshold, saturation, breakdown) {
        if threshold == 0. {
            return 0.;
        }
        if ratio.is_infinite() {
            // Solve for z/(threshold/saturation). The FMA computes the
            // scaled difference even if breakdown*saturation would overflow.
            let reciprocal = saturation / threshold;
            let x = breakdown.mul_add(saturation, -threshold) / threshold + reciprocal;
            let root = x.hypot(2. * reciprocal.sqrt());
            let scaled = if x >= 0. {
                reciprocal / root.midpoint(x)
            } else {
                root.midpoint(-x)
            };
            return nvtm * (threshold.ln() - saturation.ln() + scaled.ln());
        }
        if breakdown == 0. {
            return nvtm * ratio.ln_1p();
        }
        let x = 1. + (breakdown - ratio);
        let root = x.hypot(2. * ratio.sqrt());
        let z = if x >= 0. {
            ratio / root.midpoint(x)
        } else {
            root.midpoint(-x)
        };
        return nvtm * z.ln_1p();
    }
    let tb = 1. + ratio - breakdown;
    let exponential = 0.5 * (tb + (tb * tb + 4. * breakdown).sqrt());
    nvtm * exponential.ln()
}

struct ForwardLimit {
    voltage: f64,
    current: f64,
    slope: f64,
}

/// Mode 1 stores the exponential current, mode 2 the physical diode
/// current and its slope. Evaluate the latter from the quadratic identity,
/// rather than exponentiating the rounded voltage and subtracting one.
fn forward_limit(
    nvtm: f64,
    threshold: f64,
    saturation: f64,
    breakdown: Option<f64>,
) -> ForwardLimit {
    let coefficient = breakdown.unwrap_or(0.);
    let voltage = dio_ijth_vjm_eval(nvtm, threshold, saturation, coefficient);
    if physical_diode_limit(nvtm, threshold, saturation, coefficient) {
        let current = if breakdown.is_some() {
            threshold
        } else {
            saturation + threshold
        };
        let difference = (-coefficient).mul_add(saturation, threshold) + saturation;
        let cross = 2. * coefficient.sqrt() * saturation;
        let slope = difference.hypot(cross) / nvtm;
        ForwardLimit {
            voltage,
            current,
            slope,
        }
    } else {
        let exponential = (voltage / nvtm).exp();
        let current = if breakdown.is_some() {
            saturation * (exponential - coefficient / exponential + coefficient - 1.)
        } else {
            saturation * exponential
        };
        let slope = saturation * (exponential + coefficient / exponential) / nvtm;
        ForwardLimit {
            voltage,
            current,
            slope,
        }
    }
}

/// One side of BSIM4's junction-temperature preparation. All side-dependent
/// inputs are explicit so the shared physics never writes model/instance state.
#[derive(Clone, Copy)]
struct JunctionTemperatureInput {
    area: f64,
    perimeter: f64,
    width: f64,
    fingers: f64,
    area_density: f64,
    sidewall_density: f64,
    gate_sidewall_density: f64,
    emission_voltage: f64,
    breakdown_voltage: f64,
    breakdown_factor: f64,
    forward_threshold: f64,
    reverse_threshold: f64,
    diode_mode: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JunctionTemperatureWarning {
    ReverseThresholdTooSmall,
    UnknownDiodeMode,
}

struct ReverseLimit {
    voltage: f64,
    current: f64,
    slope: f64,
}

#[derive(Default)]
struct JunctionTemperaturePrepared {
    breakdown_exponential: Option<f64>,
    forward: Option<ForwardLimit>,
    reverse: Option<ReverseLimit>,
    warning: Option<JunctionTemperatureWarning>,
}

/// Follow the original source/drain C expressions in the same operation
/// order, including the zero-Isat fast path and the mode-specific fields.
/// Return only fields C actually writes: untouched members stay untouched.
#[expect(
    clippy::suboptimal_flops,
    reason = "three-term C saturation and reverse-limit bounds require ordered binary64 rounding"
)]
fn prepare_junction_temperature(input: JunctionTemperatureInput) -> JunctionTemperaturePrepared {
    let JunctionTemperatureInput {
        area,
        perimeter,
        width,
        fingers,
        area_density,
        sidewall_density,
        gate_sidewall_density,
        emission_voltage: nvtm,
        breakdown_voltage,
        breakdown_factor,
        forward_threshold,
        reverse_threshold,
        diode_mode,
    } = input;
    let saturation = if (area <= 0.0) && (perimeter <= 0.0) {
        0.0
    } else {
        area * area_density + perimeter * sidewall_density + width * fingers * gate_sidewall_density
    };
    let mut result = JunctionTemperaturePrepared::default();
    if saturation > 0.0 {
        match diode_mode {
            0 => {
                result.breakdown_exponential = Some(if breakdown_voltage / nvtm > EXP_THRESHOLD {
                    breakdown_factor * MIN_EXP
                } else {
                    breakdown_factor * exp(-breakdown_voltage / nvtm)
                });
            }
            1 => {
                result.forward = Some(forward_limit(nvtm, forward_threshold, saturation, None));
            }
            2 => {
                let breakdown_exponential = if breakdown_voltage / nvtm > EXP_THRESHOLD {
                    breakdown_factor * MIN_EXP
                } else {
                    let exponential = exp(-breakdown_voltage / nvtm);
                    exponential * breakdown_factor
                };
                result.breakdown_exponential = Some(breakdown_exponential);
                result.forward = Some(forward_limit(
                    nvtm,
                    forward_threshold,
                    saturation,
                    Some(breakdown_exponential),
                ));
                let mut reverse_ratio = reverse_threshold / saturation;
                if reverse_ratio < 1.0 {
                    reverse_ratio = 10.0;
                    result.warning = Some(JunctionTemperatureWarning::ReverseThresholdTooSmall);
                }
                let voltage =
                    -breakdown_voltage - nvtm * log((reverse_ratio - 1.0) / breakdown_factor);
                let exponential = breakdown_factor * exp(-(breakdown_voltage + voltage) / nvtm);
                result.reverse = Some(ReverseLimit {
                    voltage,
                    current: saturation * (1.0 + exponential),
                    slope: -saturation * exponential / nvtm,
                });
            }
            _ => result.warning = Some(JunctionTemperatureWarning::UnknownDiodeMode),
        }
    }
    result
}

/// Product of effective dimensions can overflow or underflow even when
/// the reciprocal binning weight is representable. Keep the normal path
/// unchanged and only scale the dimensions for an exceptional product.
fn cross_bin_weight(unit: f64, length: f64, width: f64) -> f64 {
    let product = length * width;
    if product.is_normal()
        || !(length > 0. && width > 0. && length.is_finite() && width.is_finite())
    {
        return unit / product;
    }
    let (numerator, exponent) = split_power(unit);
    let (left, left_exp) = split_power(length);
    let (right, right_exp) = split_power(width);
    scale_power(numerator / (left * right), exponent - left_exp - right_exp)
}

#[derive(Clone, Copy)]
struct ModelWeights([f64; 4]);

impl ModelWeights {
    /// Retain the ordinary fast path; recover severe cancellation or
    /// intermediate overflow using the existing exact finite-product sum.
    #[inline]
    fn interpolate(self, coefficients: [f64; 4]) -> f64 {
        let [base, l, w, cross] = coefficients;
        let [_, inverse_l, inverse_w, inverse_lw] = self.0;
        // Unset bin corrections are the overwhelmingly common case. When
        // the inverse weights are finite, their zero products cannot alter
        // any finite nonzero base, even with negative signed-zero inputs.
        // Zero bases and nonfinite weights still use the full ordered IEEE
        // path, preserving signed zero and 0 * infinity = NaN exactly.
        if l == 0.
            && w == 0.
            && cross == 0.
            && base != 0.
            && base.is_finite()
            && inverse_l.is_finite()
            && inverse_w.is_finite()
            && inverse_lw.is_finite()
        {
            return base;
        }
        let lt = l * inverse_l;
        let wt = w * inverse_w;
        let lwt = cross * inverse_lw;
        let sum = base + lt + wt + lwt;
        // Most cards leave the length/width/cross terms unset. Returning the
        // ordered sum also retains signed-zero and nonfinite propagation.
        if l == 0. && w == 0. && cross == 0. {
            return sum;
        }
        let magnitude = base.abs() + lt.abs() + wt.abs() + lwt.abs();
        if sum.is_normal() && sum.abs() > 0.125 * magnitude {
            sum
        } else {
            self.recover(coefficients, sum)
        }
    }

    #[cold]
    fn recover(self, coefficients: [f64; 4], ordinary: f64) -> f64 {
        if !coefficients
            .iter()
            .chain(&self.0)
            .all(|value| value.is_finite())
        {
            return ordinary;
        }
        let mut exact = crate::primitives::ExactSum::new();
        for (coefficient, weight) in coefficients.into_iter().zip(self.0) {
            exact.add_product(coefficient, weight);
        }
        let result = exact.rounded_sum();
        if result == 0. && ordinary == 0. {
            ordinary
        } else {
            result
        }
    }
}

/// A C `printf("%g")` spelling for the ported messages.
pub(super) fn c_g(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    if !value.is_finite() {
        return if value.is_nan() {
            if value.is_sign_negative() {
                "-nan"
            } else {
                "nan"
            }
            .into()
        } else if value > 0.0 {
            "inf".into()
        } else {
            "-inf".into()
        };
    }
    let scientific = format!("{value:.5e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    if (-4..6).contains(&exponent) {
        let decimals = usize::try_from(5 - exponent).unwrap_or(0);
        let fixed = format!("{value:.decimals$}");
        // Do not strip meaningful integer digits when precision is zero.
        if decimals == 0 {
            fixed
        } else {
            fixed.trim_end_matches('0').trim_end_matches('.').to_owned()
        }
    } else {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!(
            "{mantissa}e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.unsigned_abs()
        )
    }
}

/// C's `E_BADPARM` with its `IFerrorf(ERR_FATAL, ...)` message.
fn bad_parameter(message: String) -> SpiceError {
    SpiceError::circuit(message)
}

/// The locals `BSIM4temp`'s model pass computes and its instance loop
/// reads (C keeps them as function locals across the loop).
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ModelTemp {
    /// C local `Temp`.
    pub(crate) Temp: f64,
    /// C local `Tnom`.
    pub(crate) Tnom: f64,
    /// C local `TRatio`.
    pub(crate) TRatio: f64,
    /// C local `Vtm0`.
    pub(crate) Vtm0: f64,
    /// C local `Eg0`.
    pub(crate) Eg0: f64,
    /// C local `Eg`.
    pub(crate) Eg: f64,
    /// C local `ni`.
    pub(crate) ni: f64,
    /// C local `epssub`.
    pub(crate) epssub: f64,
    /// C local `epsrox`.
    pub(crate) epsrox: f64,
    /// C local `toxe`.
    pub(crate) toxe: f64,
    /// C local `delTemp`.
    pub(crate) delTemp: f64,
}

/// `BSIM4temp`'s model pass (everything before its instance loop) at
/// `temperature` (`CKTtemp`, K): updates `m` as C does and returns the
/// locals the instance loop reads. C runs it once per model per call.
#[expect(
    unused_assignments,
    reason = "C state-machine locals retain the reference assignment sequence"
)]
#[expect(
    unused_mut,
    reason = "local mutability reflects the source C intermediate state declarations"
)]
#[expect(
    unused_variables,
    reason = "C-translated intermediate variables preserve the reference layout"
)]
#[expect(
    clippy::too_many_lines,
    reason = "keeps the reference numerical routine in the original statement and rounding order"
)]
#[expect(
    clippy::if_same_then_else,
    reason = "preserves the C branch evaluation and state-update sequence"
)]
#[expect(
    clippy::assign_op_pattern,
    reason = "retains the reference calculation grouping and floating-point operation order"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "reference clamp/diagnostic comparisons and remaining ordinary product sums retain the ordered C model stage; sensitive exponent and overlap laws are conditioned"
)]
pub(super) fn temperature_model(
    m: &mut Model,
    temperature: f64,
    warnings: &mut Vec<String>,
) -> ModelTemp {
    let (mut tmp, mut tmp1, mut tmp2, mut tmp3, mut Eg, mut Eg0, mut ni, mut epssub) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut T0, mut T1, mut T2, mut T3, mut T4, mut T5, mut T6, mut T7, mut T8, mut T9) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut Lnew, mut Wnew) = (0.0_f64, 0.0_f64);
    let (mut delTemp, mut Temp, mut TRatio, mut Inv_L, mut Inv_W, mut Inv_LW, mut Vtm0, mut Tnom) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut dumPs, mut dumPd, mut dumAs, mut dumAd, mut PowWeffWr) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut DMCGeff, mut DMCIeff, mut DMDGeff) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut Nvtms, mut Nvtmd, mut SourceSatCurrent, mut DrainSatCurrent) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut T10, mut T11) = (0.0_f64, 0.0_f64);
    let (mut Inv_saref, mut Inv_sbref, mut Inv_sa, mut Inv_sb, mut rho, mut Ldrn, mut dvth0_lod) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut W_tmp, mut Inv_ODeff, mut OD_offset, mut dk2_lod, mut deta0_lod) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut lnl, mut lnw, mut lnnf, mut rbpbx, mut rbpby, mut rbsbx, mut rbsby) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut rbdbx, mut rbdby, mut bodymode) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut kvsat, mut wlod, mut sceff, mut Wdrn) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut V0, mut lt1, mut ltw, mut Theta0, mut Delt_vth, mut Vth_NarrowW, mut Lpe_Vb, mut Vth) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut n, mut n0, mut Vgsteff, mut Vgs_eff, mut niter, mut toxpf, mut toxpi, mut Tcen) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut toxe, mut epsrox, mut vddeot) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut vtfbphi2eot, mut phieot, mut TempRatioeot, mut Vtm0eot, mut Vtmeot, mut vbieot) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut Size_Not_Found: i32 = 0;
    let mut i: i32 = 0;
    let mut Fatal_Flag: i32 = 0;
    {
        Temp = temperature;
        if m.SbulkJctPotential < 0.1 {
            m.SbulkJctPotential = 0.1;
            warnings.push("Given pbs is less than 0.1. Pbs is set to 0.1.".to_string());
        }
        if m.SsidewallJctPotential < 0.1 {
            m.SsidewallJctPotential = 0.1;
            warnings.push("Given pbsws is less than 0.1. Pbsws is set to 0.1.".to_string());
        }
        if m.SGatesidewallJctPotential < 0.1 {
            m.SGatesidewallJctPotential = 0.1;
            warnings.push("Given pbswgs is less than 0.1. Pbswgs is set to 0.1.".to_string());
        }
        if m.DbulkJctPotential < 0.1 {
            m.DbulkJctPotential = 0.1;
            warnings.push("Given pbd is less than 0.1. Pbd is set to 0.1.".to_string());
        }
        if m.DsidewallJctPotential < 0.1 {
            m.DsidewallJctPotential = 0.1;
            warnings.push("Given pbswd is less than 0.1. Pbswd is set to 0.1.".to_string());
        }
        if m.DGatesidewallJctPotential < 0.1 {
            m.DGatesidewallJctPotential = 0.1;
            warnings.push("Given pbswgd is less than 0.1. Pbswgd is set to 0.1.".to_string());
        }
        if m.mtrlMod == 0 {
            let mut chktol = fabs(m.toxe);
            if fabs(m.toxp) > chktol {
                chktol = fabs(m.toxp);
            }
            if fabs(m.dtox) > chktol {
                chktol = fabs(m.dtox);
            }
            chktol = chktol * 1e-14;
            if (m.toxeGiven)
                && (m.toxpGiven)
                && (m.dtoxGiven)
                && (fabs(m.toxe - (m.toxp + m.dtox)) > chktol)
            {
                warnings.push(
                    "Warning: toxe, toxp and dtox all given and toxe != toxp + dtox; dtox ignored."
                        .to_string(),
                );
            } else if (m.toxeGiven) && (!m.toxpGiven) {
                m.toxp = m.toxe - m.dtox;
            } else if (!m.toxeGiven) && (m.toxpGiven) {
                m.toxe = m.toxp + m.dtox;
                if !m.toxmGiven {
                    m.toxm = m.toxe;
                }
            }
        } else if m.mtrlCompatMod != 0 {
            T0 = m.epsrox / 3.9;
            if (m.eotGiven)
                && (m.toxpGiven)
                && (m.dtoxGiven)
                && (ABS(m.eot * T0 - (m.toxp + m.dtox)) > 1.0e-20)
            {
                warnings.push("Warning: eot, toxp and dtox all given and eot * EPSROX / 3.9 != toxp + dtox; dtox ignored.".to_string());
            } else if (m.eotGiven) && (!m.toxpGiven) {
                m.toxp = T0.mul_add(m.eot, -m.dtox);
            } else if (!m.eotGiven) && (m.toxpGiven) {
                m.eot = (m.toxp + m.dtox) / T0;
                if !m.toxmGiven {
                    m.toxm = m.eot;
                }
            }
        }
        if m.mtrlMod != 0 {
            epsrox = 3.9;
            toxe = m.eot;
            epssub = EPS0 * m.epsrsub;
        } else {
            epsrox = m.epsrox;
            toxe = m.toxe;
            epssub = EPSSI;
        }
        if !m.cfGiven {
            m.cf = 2.0 * epsrox * EPS0 / PI * (0.4e-6 / toxe).ln_1p();
        }
        m.coxe = epsrox * EPS0 / toxe;
        if m.mtrlMod == 0 || m.mtrlCompatMod != 0 {
            m.coxp = m.epsrox * EPS0 / m.toxp;
        }
        if !m.cgdoGiven {
            if m.dlcGiven && (m.dlc > 0.0) {
                m.cgdo = m.dlc.mul_add(m.coxe, -m.cgdl);
            } else {
                m.cgdo = 0.6 * m.xj * m.coxe;
            }
        }
        if !m.cgsoGiven {
            if m.dlcGiven && (m.dlc > 0.0) {
                m.cgso = m.dlc.mul_add(m.coxe, -m.cgsl);
            } else {
                m.cgso = 0.6 * m.xj * m.coxe;
            }
        }
        if !m.cgboGiven {
            m.cgbo = 2.0 * m.dwc * m.coxe;
        }
        // C resets the model's size-dependent parameter cache here.
        Tnom = m.tnom;
        TRatio = Temp / Tnom;
        m.vcrit = CONSTvt0 * log(CONSTvt0 / (CONSTroot2 * 1.0e-14));
        m.factor1 = sqrt(epssub / (epsrox * EPS0) * toxe);
        m.vtm0 = KboQ * Tnom;
        Vtm0 = m.vtm0;
        if m.mtrlMod == 0 {
            Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0);
            ni = 1.45e10
                * (Tnom / 300.15)
                * sqrt(Tnom / 300.15)
                * exp(21.556_598_1 - Eg0 / (2.0 * Vtm0));
        } else {
            Eg0 = m.bg0sub - m.tbgasub * Tnom * Tnom / (Tnom + m.tbgbsub);
            T0 = m.bg0sub - m.tbgasub * 90090.0225 / (300.15 + m.tbgbsub);
            ni = m.ni0sub * (Tnom / 300.15) * sqrt(Tnom / 300.15) * exp((T0 - Eg0) / (2.0 * Vtm0));
        }
        m.Eg0 = Eg0;
        m.vtm = KboQ * Temp;
        if m.mtrlMod == 0 {
            Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0);
        } else {
            Eg = m.bg0sub - m.tbgasub * Temp * Temp / (Temp + m.tbgbsub);
        }
        if Temp == Tnom {
            m.SjctTempSatCurDensity = m.SjctSatCurDensity;
            m.SjctSidewallTempSatCurDensity = m.SjctSidewallSatCurDensity;
            m.SjctGateSidewallTempSatCurDensity = m.SjctGateSidewallSatCurDensity;
            m.DjctTempSatCurDensity = m.DjctSatCurDensity;
            m.DjctSidewallTempSatCurDensity = m.DjctSidewallSatCurDensity;
            m.DjctGateSidewallTempSatCurDensity = m.DjctGateSidewallSatCurDensity;
        } else {
            let (e0, alpha, beta) = if m.mtrlMod == 0 {
                (1.16, 7.02e-4, 1108.)
            } else {
                (m.bg0sub, m.tbgasub, m.tbgbsub)
            };
            T0 = if Temp > 0.
                && Tnom > 0.
                && Temp.is_finite()
                && Tnom.is_finite()
                && e0.is_finite()
                && alpha.is_finite()
                && beta.is_finite()
                && Temp + beta > 0.
                && Tnom + beta > 0.
            {
                thermal_bandgap_difference(Tnom, Temp, e0, alpha, beta)
            } else {
                Eg0 / Vtm0 - Eg / m.vtm
            };
            T1 = log_temperature_ratio(Temp, Tnom);
            let scale = ExpScale::new(m.SjctTempExponent.mul_add(T1, T0) / m.SjctEmissionCoeff);
            m.SjctTempSatCurDensity = scale.apply(m.SjctSatCurDensity);
            m.SjctSidewallTempSatCurDensity = scale.apply(m.SjctSidewallSatCurDensity);
            m.SjctGateSidewallTempSatCurDensity = scale.apply(m.SjctGateSidewallSatCurDensity);
            let scale = ExpScale::new(m.DjctTempExponent.mul_add(T1, T0) / m.DjctEmissionCoeff);
            m.DjctTempSatCurDensity = scale.apply(m.DjctSatCurDensity);
            m.DjctSidewallTempSatCurDensity = scale.apply(m.DjctSidewallSatCurDensity);
            m.DjctGateSidewallTempSatCurDensity = scale.apply(m.DjctGateSidewallSatCurDensity);
        }
        if m.SjctTempSatCurDensity < 0.0 {
            m.SjctTempSatCurDensity = 0.0;
        }
        if m.SjctSidewallTempSatCurDensity < 0.0 {
            m.SjctSidewallTempSatCurDensity = 0.0;
        }
        if m.SjctGateSidewallTempSatCurDensity < 0.0 {
            m.SjctGateSidewallTempSatCurDensity = 0.0;
        }
        if m.DjctTempSatCurDensity < 0.0 {
            m.DjctTempSatCurDensity = 0.0;
        }
        if m.DjctSidewallTempSatCurDensity < 0.0 {
            m.DjctSidewallTempSatCurDensity = 0.0;
        }
        if m.DjctGateSidewallTempSatCurDensity < 0.0 {
            m.DjctGateSidewallTempSatCurDensity = 0.0;
        }
        delTemp = temperature - m.tnom;
        T0 = m.tcj * delTemp;
        if T0 >= -1.0 {
            m.SunitAreaTempJctCap = m.SunitAreaJctCap * (1.0 + T0);
            m.DunitAreaTempJctCap = m.DunitAreaJctCap * (1.0 + T0);
        } else {
            if m.SunitAreaJctCap > 0.0 {
                m.SunitAreaTempJctCap = 0.0;
                warnings.push(
                    "Temperature effect has caused cjs to be negative. Cjs is clamped to zero."
                        .to_string(),
                );
            }
            if m.DunitAreaJctCap > 0.0 {
                m.DunitAreaTempJctCap = 0.0;
                warnings.push(
                    "Temperature effect has caused cjd to be negative. Cjd is clamped to zero."
                        .to_string(),
                );
            }
        }
        T0 = m.tcjsw * delTemp;
        if m.SunitLengthSidewallJctCap < 0.0 {
            m.SunitLengthSidewallJctCap = 0.0;
            warnings.push("CJSWS is negative. Cjsws is clamped to zero.".to_string());
        }
        if m.DunitLengthSidewallJctCap < 0.0 {
            m.DunitLengthSidewallJctCap = 0.0;
            warnings.push("CJSWD is negative. Cjswd is clamped to zero.".to_string());
        }
        if T0 >= -1.0 {
            m.SunitLengthSidewallTempJctCap = m.SunitLengthSidewallJctCap * (1.0 + T0);
            m.DunitLengthSidewallTempJctCap = m.DunitLengthSidewallJctCap * (1.0 + T0);
        } else {
            if m.SunitLengthSidewallJctCap > 0.0 {
                m.SunitLengthSidewallTempJctCap = 0.0;
                warnings.push(
                    "Temperature effect has caused cjsws to be negative. Cjsws is clamped to zero."
                        .to_string(),
                );
            }
            if m.DunitLengthSidewallJctCap > 0.0 {
                m.DunitLengthSidewallTempJctCap = 0.0;
                warnings.push(
                    "Temperature effect has caused cjswd to be negative. Cjswd is clamped to zero."
                        .to_string(),
                );
            }
        }
        T0 = m.tcjswg * delTemp;
        if T0 >= -1.0 {
            m.SunitLengthGateSidewallTempJctCap = m.SunitLengthGateSidewallJctCap * (1.0 + T0);
            m.DunitLengthGateSidewallTempJctCap = m.DunitLengthGateSidewallJctCap * (1.0 + T0);
        } else {
            if m.SunitLengthGateSidewallJctCap > 0.0 {
                m.SunitLengthGateSidewallTempJctCap = 0.0;
                warnings.push("Temperature effect has caused cjswgs to be negative. Cjswgs is clamped to zero.".to_string());
            }
            if m.DunitLengthGateSidewallJctCap > 0.0 {
                m.DunitLengthGateSidewallTempJctCap = 0.0;
                warnings.push("Temperature effect has caused cjswgd to be negative. Cjswgd is clamped to zero.".to_string());
            }
        }
        m.PhiBS = m.SbulkJctPotential - m.tpb * delTemp;
        if m.PhiBS < 0.01 {
            m.PhiBS = 0.01;
            warnings.push(
                "Temperature effect has caused pbs to be less than 0.01. Pbs is clamped to 0.01."
                    .to_string(),
            );
        }
        m.PhiBD = m.DbulkJctPotential - m.tpb * delTemp;
        if m.PhiBD < 0.01 {
            m.PhiBD = 0.01;
            warnings.push(
                "Temperature effect has caused pbd to be less than 0.01. Pbd is clamped to 0.01."
                    .to_string(),
            );
        }
        m.PhiBSWS = m.SsidewallJctPotential - m.tpbsw * delTemp;
        if m.PhiBSWS < 0.01 {
            m.PhiBSWS = 0.01;
            warnings.push("Temperature effect has caused pbsws to be less than 0.01. Pbsws is clamped to 0.01.".to_string());
        }
        m.PhiBSWD = m.DsidewallJctPotential - m.tpbsw * delTemp;
        if m.PhiBSWD < 0.01 {
            m.PhiBSWD = 0.01;
            warnings.push("Temperature effect has caused pbswd to be less than 0.01. Pbswd is clamped to 0.01.".to_string());
        }
        m.PhiBSWGS = m.SGatesidewallJctPotential - m.tpbswg * delTemp;
        if m.PhiBSWGS < 0.01 {
            m.PhiBSWGS = 0.01;
            warnings.push("Temperature effect has caused pbswgs to be less than 0.01. Pbswgs is clamped to 0.01.".to_string());
        }
        m.PhiBSWGD = m.DGatesidewallJctPotential - m.tpbswg * delTemp;
        if m.PhiBSWGD < 0.01 {
            m.PhiBSWGD = 0.01;
            warnings.push("Temperature effect has caused pbswgd to be less than 0.01. Pbswgd is clamped to 0.01.".to_string());
        }
        if m.ijthdfwd <= 0.0 {
            m.ijthdfwd = 0.0;
            warnings.push(format!("Ijthdfwd reset to {}.", c_g(m.ijthdfwd)));
        }
        if m.ijthsfwd <= 0.0 {
            m.ijthsfwd = 0.0;
            warnings.push(format!("Ijthsfwd reset to {}.", c_g(m.ijthsfwd)));
        }
        if m.ijthdrev <= 0.0 {
            m.ijthdrev = 0.0;
            warnings.push(format!("Ijthdrev reset to {}.", c_g(m.ijthdrev)));
        }
        if m.ijthsrev <= 0.0 {
            m.ijthsrev = 0.0;
            warnings.push(format!("Ijthsrev reset to {}.", c_g(m.ijthsrev)));
        }
        if (m.xjbvd <= 0.0) && (m.dioMod == 2) {
            m.xjbvd = 0.0;
            warnings.push(format!("Xjbvd reset to {}.", c_g(m.xjbvd)));
        } else if (m.xjbvd < 0.0) && (m.dioMod == 0) {
            m.xjbvd = 0.0;
            warnings.push(format!("Xjbvd reset to {}.", c_g(m.xjbvd)));
        }
        if m.bvd <= 0.0 {
            m.bvd = 0.0;
            warnings.push(format!("BVD reset to {}.", c_g(m.bvd)));
        }
        if (m.xjbvs <= 0.0) && (m.dioMod == 2) {
            m.xjbvs = 0.0;
            warnings.push(format!("Xjbvs reset to {}.", c_g(m.xjbvs)));
        } else if (m.xjbvs < 0.0) && (m.dioMod == 0) {
            m.xjbvs = 0.0;
            warnings.push(format!("Xjbvs reset to {}.", c_g(m.xjbvs)));
        }
        if m.bvs <= 0.0 {
            m.bvs = 0.0;
            warnings.push(format!("BVS reset to {}.", c_g(m.bvs)));
        }
    }
    ModelTemp {
        Temp,
        Tnom,
        TRatio,
        Vtm0,
        Eg0,
        Eg,
        ni,
        epssub,
        epsrox,
        toxe,
        delTemp,
    }
}

/// `BSIM4temp`'s instance loop body for `h`: its size-dependent parameters
/// (returned), its temperature-dependent instance values (into `h`) and,
/// when `check` is set, `BSIM4checkModel` (see [`check_model`] for which
/// instance C checks). `m` is the model after [`temperature_model`] and
/// after every instance C processed earlier in this pass: C's instance loop
/// updates the model (`njts*temp`, `dlcig`, `jtweff`, and the checked
/// instance's `b4check.c` resets), and later instances read those updates.
/// `model_name`/`instance_name` are C's `BSIM4modName`/`BSIM4name`, which
/// its messages print. Messages C prints to stdout/stderr are appended to
/// `warnings` in C order, followed by the check's nonfatal messages.
///
/// C quirks the per-instance port reproduces or changes:
/// - C ignores the instance `dtemp` here: everything uses `CKTtemp`.
/// - `here->BSIM4grgeltd` uses `Lnew`, which C only assigns while it
///   computes a new size-dependent parameter set; an instance that reuses
///   an earlier instance's set sees the `Lnew` of the last set C computed
///   in this `BSIM4temp` call (possibly another geometry). Here every
///   instance computes its own set, so it always sees its own `Lnew`.
/// - `Phi`/`phieot` "Fatal" messages set C's local `Fatal_Flag`, which
///   `BSIM4temp` never reads: they are warnings, as in C.
///
/// # Errors
/// C's fatal `BSIM4checkModel` result and nonpositive effective sizes
/// (`E_BADPARM`), with C's messages.
#[expect(
    unused_assignments,
    reason = "upstream C temperature routine assigns intermediates in original order"
)]
#[expect(
    unused_parens,
    reason = "preserves C parenthesization in temperature equations"
)]
#[expect(
    unused_mut,
    reason = "mutable locals retain the reference C state flow"
)]
#[expect(
    unused_variables,
    reason = "reference C temperature locals are kept for statement-level comparison"
)]
#[expect(
    clippy::too_many_lines,
    reason = "temperature model follows the upstream C routine statement by statement"
)]
#[expect(
    clippy::assign_op_pattern,
    reason = "C temperature arithmetic preserves its original evaluation order"
)]
#[expect(
    clippy::manual_clamp,
    reason = "explicit clamp branches preserve the C handling of nonfinite values"
)]
#[expect(
    clippy::collapsible_else_if,
    reason = "keeps ngspice C temperature branch topology unchanged"
)] // A statement-by-statement port of C.
#[expect(
    clippy::suboptimal_flops,
    reason = "reference branch thresholds and ordinary prepared-instance products retain their order; sensitive bin, well-proximity, diode and thermal laws are conditioned; see BSIM_TEMPERATURE.md"
)]
pub(super) fn temperature_instance(
    m: &mut Model,
    h: &mut Instance,
    locals: &ModelTemp,
    check: bool,
    model_name: &str,
    instance_name: &str,
    warnings: &mut Vec<String>,
) -> SpiceResult<SizeParam> {
    let mut p = SizeParam::default();
    let (mut tmp, mut tmp1, mut tmp2, mut tmp3, mut Eg, mut Eg0, mut ni, mut epssub) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut T0, mut T1, mut T2, mut T3, mut T4, mut T5, mut T6, mut T7, mut T8, mut T9) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut Lnew, mut Wnew) = (0.0_f64, 0.0_f64);
    let (mut delTemp, mut Temp, mut TRatio, mut Inv_L, mut Inv_W, mut Inv_LW, mut Vtm0, mut Tnom) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut dumPs, mut dumPd, mut dumAs, mut dumAd, mut PowWeffWr) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut DMCGeff, mut DMCIeff, mut DMDGeff) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut T10, mut T11) = (0.0_f64, 0.0_f64);
    let (mut Inv_saref, mut Inv_sbref, mut Inv_sa, mut Inv_sb, mut rho, mut Ldrn, mut dvth0_lod) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut W_tmp, mut Inv_ODeff, mut OD_offset, mut dk2_lod, mut deta0_lod) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut lnl, mut lnw, mut lnnf, mut rbpbx, mut rbpby, mut rbsbx, mut rbsby) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut rbdbx, mut rbdby, mut bodymode) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut kvsat, mut wlod, mut sceff, mut Wdrn) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut V0, mut lt1, mut ltw, mut Theta0, mut Delt_vth, mut Vth_NarrowW, mut Lpe_Vb, mut Vth) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut n, mut n0, mut Vgsteff, mut Vgs_eff, mut niter, mut toxpf, mut toxpi, mut Tcen) = (
        0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64,
    );
    let (mut toxe, mut epsrox, mut vddeot) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut vtfbphi2eot, mut phieot, mut TempRatioeot, mut Vtm0eot, mut Vtmeot, mut vbieot) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut Size_Not_Found: i32 = 0;
    let mut i: i32 = 0;
    let mut Fatal_Flag: i32 = 0;
    let ModelTemp {
        Temp,
        Tnom,
        TRatio,
        Vtm0,
        Eg0,
        Eg,
        ni,
        epssub,
        epsrox,
        toxe,
        delTemp,
    } = *locals;
    // The instance loop, for this instance. C searches the model's
    // `pSizeDependParamKnot` list for an earlier instance with equal
    // (l, w, nf) and shares its parameters; recomputing them per instance
    // gives the same values, so the search always "fails" here.
    {
        Size_Not_Found = 1;
        Ldrn = h.l;
        Wdrn = h.w / h.nf;
        if Size_Not_Found != 0 {
            p.Length = h.l;
            p.Width = h.w;
            p.NFinger = h.nf;
            Lnew = h.l + m.xl;
            Wnew = h.w / h.nf + m.xw;
            T0 = pow(Lnew, m.Lln);
            T1 = pow(Wnew, m.Lwn);
            tmp1 = m.Ll / T0 + m.Lw / T1 + m.Lwl / (T0 * T1);
            p.dl = m.Lint + tmp1;
            tmp2 = m.Llc / T0 + m.Lwc / T1 + m.Lwlc / (T0 * T1);
            p.dlc = m.dlc + tmp2;
            T2 = pow(Lnew, m.Wln);
            T3 = pow(Wnew, m.Wwn);
            tmp1 = m.Wl / T2 + m.Ww / T3 + m.Wwl / (T2 * T3);
            p.dw = m.Wint + tmp1;
            tmp2 = m.Wlc / T2 + m.Wwc / T3 + m.Wwlc / (T2 * T3);
            p.dwc = m.dwc + tmp2;
            p.dwj = m.dwj + tmp2;
            p.leff = Lnew - 2.0 * p.dl;
            if p.leff <= 0.0 {
                return Err(bad_parameter(format!(
                    "BSIM4: mosfet {model_name}, model {instance_name}: Effective channel length <= 0"
                )));
            }
            p.weff = Wnew - 2.0 * p.dw;
            if p.weff <= 0.0 {
                return Err(bad_parameter(format!(
                    "BSIM4: mosfet {model_name}, model {instance_name}: Effective channel width <= 0"
                )));
            }
            p.leffCV = Lnew - 2.0 * p.dlc;
            if p.leffCV <= 0.0 {
                return Err(bad_parameter(format!(
                    "BSIM4: mosfet {model_name}, model {instance_name}: Effective channel length for C-V <= 0"
                )));
            }
            p.weffCV = Wnew - 2.0 * p.dwc;
            if p.weffCV <= 0.0 {
                return Err(bad_parameter(format!(
                    "BSIM4: mosfet {model_name}, model {instance_name}: Effective channel width for C-V <= 0"
                )));
            }
            p.weffCJ = Wnew - 2.0 * p.dwj;
            if p.weffCJ <= 0.0 {
                return Err(bad_parameter(format!(
                    "BSIM4: mosfet {model_name}, model {instance_name}: Effective channel width for S/D junctions <= 0"
                )));
            }
            if m.binUnit == 1 {
                Inv_L = 1.0e-6 / p.leff;
                Inv_W = 1.0e-6 / p.weff;
                Inv_LW = cross_bin_weight(1.0e-12, p.leff, p.weff);
            } else {
                Inv_L = 1.0 / p.leff;
                Inv_W = 1.0 / p.weff;
                Inv_LW = cross_bin_weight(1.0, p.leff, p.weff);
            }
            let weights = ModelWeights([1., Inv_L, Inv_W, Inv_LW]);
            p.cdsc = weights.interpolate([m.cdsc, m.lcdsc, m.wcdsc, m.pcdsc]);
            p.cdscb = weights.interpolate([m.cdscb, m.lcdscb, m.wcdscb, m.pcdscb]);
            p.cdscd = weights.interpolate([m.cdscd, m.lcdscd, m.wcdscd, m.pcdscd]);
            p.cit = weights.interpolate([m.cit, m.lcit, m.wcit, m.pcit]);
            p.nfactor = weights.interpolate([m.nfactor, m.lnfactor, m.wnfactor, m.pnfactor]);
            p.tnfactor = weights.interpolate([m.tnfactor, m.ltnfactor, m.wtnfactor, m.ptnfactor]);
            p.xj = weights.interpolate([m.xj, m.lxj, m.wxj, m.pxj]);
            p.vsat = weights.interpolate([m.vsat, m.lvsat, m.wvsat, m.pvsat]);
            p.at = weights.interpolate([m.at, m.lat, m.wat, m.pat]);
            p.a0 = weights.interpolate([m.a0, m.la0, m.wa0, m.pa0]);
            p.ags = weights.interpolate([m.ags, m.lags, m.wags, m.pags]);
            p.a1 = weights.interpolate([m.a1, m.la1, m.wa1, m.pa1]);
            p.a2 = weights.interpolate([m.a2, m.la2, m.wa2, m.pa2]);
            p.keta = weights.interpolate([m.keta, m.lketa, m.wketa, m.pketa]);
            p.ketac = weights.interpolate([m.ketac, m.lketac, m.wketac, m.pketac]);
            p.nsub = weights.interpolate([m.nsub, m.lnsub, m.wnsub, m.pnsub]);
            p.ndep = weights.interpolate([m.ndep, m.lndep, m.wndep, m.pndep]);
            p.nsd = weights.interpolate([m.nsd, m.lnsd, m.wnsd, m.pnsd]);
            p.phin = weights.interpolate([m.phin, m.lphin, m.wphin, m.pphin]);
            p.ngate = weights.interpolate([m.ngate, m.lngate, m.wngate, m.pngate]);
            p.gamma1 = weights.interpolate([m.gamma1, m.lgamma1, m.wgamma1, m.pgamma1]);
            p.gamma2 = weights.interpolate([m.gamma2, m.lgamma2, m.wgamma2, m.pgamma2]);
            p.vbx = weights.interpolate([m.vbx, m.lvbx, m.wvbx, m.pvbx]);
            p.vbm = weights.interpolate([m.vbm, m.lvbm, m.wvbm, m.pvbm]);
            p.xt = weights.interpolate([m.xt, m.lxt, m.wxt, m.pxt]);
            p.vfb = weights.interpolate([m.vfb, m.lvfb, m.wvfb, m.pvfb]);
            p.k1 = weights.interpolate([m.k1, m.lk1, m.wk1, m.pk1]);
            p.kt1 = weights.interpolate([m.kt1, m.lkt1, m.wkt1, m.pkt1]);
            p.kt1l = weights.interpolate([m.kt1l, m.lkt1l, m.wkt1l, m.pkt1l]);
            p.k2 = weights.interpolate([m.k2, m.lk2, m.wk2, m.pk2]);
            p.kt2 = weights.interpolate([m.kt2, m.lkt2, m.wkt2, m.pkt2]);
            p.k3 = weights.interpolate([m.k3, m.lk3, m.wk3, m.pk3]);
            p.k3b = weights.interpolate([m.k3b, m.lk3b, m.wk3b, m.pk3b]);
            p.w0 = weights.interpolate([m.w0, m.lw0, m.ww0, m.pw0]);
            p.lpe0 = weights.interpolate([m.lpe0, m.llpe0, m.wlpe0, m.plpe0]);
            p.lpeb = weights.interpolate([m.lpeb, m.llpeb, m.wlpeb, m.plpeb]);
            p.dvtp0 = weights.interpolate([m.dvtp0, m.ldvtp0, m.wdvtp0, m.pdvtp0]);
            p.dvtp1 = weights.interpolate([m.dvtp1, m.ldvtp1, m.wdvtp1, m.pdvtp1]);
            p.dvtp2 = weights.interpolate([m.dvtp2, m.ldvtp2, m.wdvtp2, m.pdvtp2]);
            p.dvtp3 = weights.interpolate([m.dvtp3, m.ldvtp3, m.wdvtp3, m.pdvtp3]);
            p.dvtp4 = weights.interpolate([m.dvtp4, m.ldvtp4, m.wdvtp4, m.pdvtp4]);
            p.dvtp5 = weights.interpolate([m.dvtp5, m.ldvtp5, m.wdvtp5, m.pdvtp5]);
            p.dvt0 = weights.interpolate([m.dvt0, m.ldvt0, m.wdvt0, m.pdvt0]);
            p.dvt1 = weights.interpolate([m.dvt1, m.ldvt1, m.wdvt1, m.pdvt1]);
            p.dvt2 = weights.interpolate([m.dvt2, m.ldvt2, m.wdvt2, m.pdvt2]);
            p.dvt0w = weights.interpolate([m.dvt0w, m.ldvt0w, m.wdvt0w, m.pdvt0w]);
            p.dvt1w = weights.interpolate([m.dvt1w, m.ldvt1w, m.wdvt1w, m.pdvt1w]);
            p.dvt2w = weights.interpolate([m.dvt2w, m.ldvt2w, m.wdvt2w, m.pdvt2w]);
            p.drout = weights.interpolate([m.drout, m.ldrout, m.wdrout, m.pdrout]);
            p.dsub = weights.interpolate([m.dsub, m.ldsub, m.wdsub, m.pdsub]);
            p.vth0 = weights.interpolate([m.vth0, m.lvth0, m.wvth0, m.pvth0]);
            p.ua = weights.interpolate([m.ua, m.lua, m.wua, m.pua]);
            p.ua1 = weights.interpolate([m.ua1, m.lua1, m.wua1, m.pua1]);
            p.ub = weights.interpolate([m.ub, m.lub, m.wub, m.r#pub]);
            p.ub1 = weights.interpolate([m.ub1, m.lub1, m.wub1, m.pub1]);
            p.uc = weights.interpolate([m.uc, m.luc, m.wuc, m.puc]);
            p.uc1 = weights.interpolate([m.uc1, m.luc1, m.wuc1, m.puc1]);
            p.ud = weights.interpolate([m.ud, m.lud, m.wud, m.pud]);
            p.ud1 = weights.interpolate([m.ud1, m.lud1, m.wud1, m.pud1]);
            p.up = weights.interpolate([m.up, m.lup, m.wup, m.pup]);
            p.lp = weights.interpolate([m.lp, m.llp, m.wlp, m.plp]);
            p.eu = weights.interpolate([m.eu, m.leu, m.weu, m.peu]);
            p.u0 = weights.interpolate([m.u0, m.lu0, m.wu0, m.pu0]);
            p.ute = weights.interpolate([m.ute, m.lute, m.wute, m.pute]);
            p.ucs = weights.interpolate([m.ucs, m.lucs, m.wucs, m.pucs]);
            p.ucste = weights.interpolate([m.ucste, m.lucste, m.wucste, m.pucste]);
            p.voff = weights.interpolate([m.voff, m.lvoff, m.wvoff, m.pvoff]);
            p.tvoff = weights.interpolate([m.tvoff, m.ltvoff, m.wtvoff, m.ptvoff]);
            p.minv = weights.interpolate([m.minv, m.lminv, m.wminv, m.pminv]);
            p.minvcv = weights.interpolate([m.minvcv, m.lminvcv, m.wminvcv, m.pminvcv]);
            p.fprout = weights.interpolate([m.fprout, m.lfprout, m.wfprout, m.pfprout]);
            p.pdits = weights.interpolate([m.pdits, m.lpdits, m.wpdits, m.ppdits]);
            p.pditsd = weights.interpolate([m.pditsd, m.lpditsd, m.wpditsd, m.ppditsd]);
            p.delta = weights.interpolate([m.delta, m.ldelta, m.wdelta, m.pdelta]);
            p.rdsw = weights.interpolate([m.rdsw, m.lrdsw, m.wrdsw, m.prdsw]);
            p.rdw = weights.interpolate([m.rdw, m.lrdw, m.wrdw, m.prdw]);
            p.rsw = weights.interpolate([m.rsw, m.lrsw, m.wrsw, m.prsw]);
            p.prwg = weights.interpolate([m.prwg, m.lprwg, m.wprwg, m.pprwg]);
            p.prwb = weights.interpolate([m.prwb, m.lprwb, m.wprwb, m.pprwb]);
            p.prt = weights.interpolate([m.prt, m.lprt, m.wprt, m.pprt]);
            p.eta0 = weights.interpolate([m.eta0, m.leta0, m.weta0, m.peta0]);
            p.teta0 = weights.interpolate([m.teta0, m.lteta0, m.wteta0, m.pteta0]);
            p.tvoffcv = weights.interpolate([m.tvoffcv, m.ltvoffcv, m.wtvoffcv, m.ptvoffcv]);
            p.etab = weights.interpolate([m.etab, m.letab, m.wetab, m.petab]);
            p.pclm = weights.interpolate([m.pclm, m.lpclm, m.wpclm, m.ppclm]);
            p.pdibl1 = weights.interpolate([m.pdibl1, m.lpdibl1, m.wpdibl1, m.ppdibl1]);
            p.pdibl2 = weights.interpolate([m.pdibl2, m.lpdibl2, m.wpdibl2, m.ppdibl2]);
            p.pdiblb = weights.interpolate([m.pdiblb, m.lpdiblb, m.wpdiblb, m.ppdiblb]);
            p.pscbe1 = weights.interpolate([m.pscbe1, m.lpscbe1, m.wpscbe1, m.ppscbe1]);
            p.pscbe2 = weights.interpolate([m.pscbe2, m.lpscbe2, m.wpscbe2, m.ppscbe2]);
            p.pvag = weights.interpolate([m.pvag, m.lpvag, m.wpvag, m.ppvag]);
            p.wr = weights.interpolate([m.wr, m.lwr, m.wwr, m.pwr]);
            p.dwg = weights.interpolate([m.dwg, m.ldwg, m.wdwg, m.pdwg]);
            p.dwb = weights.interpolate([m.dwb, m.ldwb, m.wdwb, m.pdwb]);
            p.b0 = weights.interpolate([m.b0, m.lb0, m.wb0, m.pb0]);
            p.b1 = weights.interpolate([m.b1, m.lb1, m.wb1, m.pb1]);
            p.alpha0 = weights.interpolate([m.alpha0, m.lalpha0, m.walpha0, m.palpha0]);
            p.alpha1 = weights.interpolate([m.alpha1, m.lalpha1, m.walpha1, m.palpha1]);
            p.beta0 = weights.interpolate([m.beta0, m.lbeta0, m.wbeta0, m.pbeta0]);
            p.agidl = weights.interpolate([m.agidl, m.lagidl, m.wagidl, m.pagidl]);
            p.bgidl = weights.interpolate([m.bgidl, m.lbgidl, m.wbgidl, m.pbgidl]);
            p.cgidl = weights.interpolate([m.cgidl, m.lcgidl, m.wcgidl, m.pcgidl]);
            p.egidl = weights.interpolate([m.egidl, m.legidl, m.wegidl, m.pegidl]);
            p.rgidl = weights.interpolate([m.rgidl, m.lrgidl, m.wrgidl, m.prgidl]);
            p.kgidl = weights.interpolate([m.kgidl, m.lkgidl, m.wkgidl, m.pkgidl]);
            p.fgidl = weights.interpolate([m.fgidl, m.lfgidl, m.wfgidl, m.pfgidl]);
            p.agisl = weights.interpolate([m.agisl, m.lagisl, m.wagisl, m.pagisl]);
            p.bgisl = weights.interpolate([m.bgisl, m.lbgisl, m.wbgisl, m.pbgisl]);
            p.cgisl = weights.interpolate([m.cgisl, m.lcgisl, m.wcgisl, m.pcgisl]);
            p.egisl = weights.interpolate([m.egisl, m.legisl, m.wegisl, m.pegisl]);
            p.rgisl = weights.interpolate([m.rgisl, m.lrgisl, m.wrgisl, m.prgisl]);
            p.kgisl = weights.interpolate([m.kgisl, m.lkgisl, m.wkgisl, m.pkgisl]);
            p.fgisl = weights.interpolate([m.fgisl, m.lfgisl, m.wfgisl, m.pfgisl]);
            p.aigc = weights.interpolate([m.aigc, m.laigc, m.waigc, m.paigc]);
            p.bigc = weights.interpolate([m.bigc, m.lbigc, m.wbigc, m.pbigc]);
            p.cigc = weights.interpolate([m.cigc, m.lcigc, m.wcigc, m.pcigc]);
            p.aigsd = weights.interpolate([m.aigsd, m.laigsd, m.waigsd, m.paigsd]);
            p.bigsd = weights.interpolate([m.bigsd, m.lbigsd, m.wbigsd, m.pbigsd]);
            p.cigsd = weights.interpolate([m.cigsd, m.lcigsd, m.wcigsd, m.pcigsd]);
            p.aigs = weights.interpolate([m.aigs, m.laigs, m.waigs, m.paigs]);
            p.bigs = weights.interpolate([m.bigs, m.lbigs, m.wbigs, m.pbigs]);
            p.cigs = weights.interpolate([m.cigs, m.lcigs, m.wcigs, m.pcigs]);
            p.aigd = weights.interpolate([m.aigd, m.laigd, m.waigd, m.paigd]);
            p.bigd = weights.interpolate([m.bigd, m.lbigd, m.wbigd, m.pbigd]);
            p.cigd = weights.interpolate([m.cigd, m.lcigd, m.wcigd, m.pcigd]);
            p.aigbacc = weights.interpolate([m.aigbacc, m.laigbacc, m.waigbacc, m.paigbacc]);
            p.bigbacc = weights.interpolate([m.bigbacc, m.lbigbacc, m.wbigbacc, m.pbigbacc]);
            p.cigbacc = weights.interpolate([m.cigbacc, m.lcigbacc, m.wcigbacc, m.pcigbacc]);
            p.aigbinv = weights.interpolate([m.aigbinv, m.laigbinv, m.waigbinv, m.paigbinv]);
            p.bigbinv = weights.interpolate([m.bigbinv, m.lbigbinv, m.wbigbinv, m.pbigbinv]);
            p.cigbinv = weights.interpolate([m.cigbinv, m.lcigbinv, m.wcigbinv, m.pcigbinv]);
            p.nigc = weights.interpolate([m.nigc, m.lnigc, m.wnigc, m.pnigc]);
            p.nigbacc = weights.interpolate([m.nigbacc, m.lnigbacc, m.wnigbacc, m.pnigbacc]);
            p.nigbinv = weights.interpolate([m.nigbinv, m.lnigbinv, m.wnigbinv, m.pnigbinv]);
            p.ntox = weights.interpolate([m.ntox, m.lntox, m.wntox, m.pntox]);
            p.eigbinv = weights.interpolate([m.eigbinv, m.leigbinv, m.weigbinv, m.peigbinv]);
            p.pigcd = weights.interpolate([m.pigcd, m.lpigcd, m.wpigcd, m.ppigcd]);
            p.poxedge = weights.interpolate([m.poxedge, m.lpoxedge, m.wpoxedge, m.ppoxedge]);
            p.xrcrg1 = weights.interpolate([m.xrcrg1, m.lxrcrg1, m.wxrcrg1, m.pxrcrg1]);
            p.xrcrg2 = weights.interpolate([m.xrcrg2, m.lxrcrg2, m.wxrcrg2, m.pxrcrg2]);
            p.lambda = weights.interpolate([m.lambda, m.llambda, m.wlambda, m.plambda]);
            p.vtl = weights.interpolate([m.vtl, m.lvtl, m.wvtl, m.pvtl]);
            p.xn = weights.interpolate([m.xn, m.lxn, m.wxn, m.pxn]);
            p.vfbsdoff = weights.interpolate([m.vfbsdoff, m.lvfbsdoff, m.wvfbsdoff, m.pvfbsdoff]);
            p.tvfbsdoff =
                weights.interpolate([m.tvfbsdoff, m.ltvfbsdoff, m.wtvfbsdoff, m.ptvfbsdoff]);
            p.cgsl = weights.interpolate([m.cgsl, m.lcgsl, m.wcgsl, m.pcgsl]);
            p.cgdl = weights.interpolate([m.cgdl, m.lcgdl, m.wcgdl, m.pcgdl]);
            p.ckappas = weights.interpolate([m.ckappas, m.lckappas, m.wckappas, m.pckappas]);
            p.ckappad = weights.interpolate([m.ckappad, m.lckappad, m.wckappad, m.pckappad]);
            p.cf = weights.interpolate([m.cf, m.lcf, m.wcf, m.pcf]);
            p.clc = weights.interpolate([m.clc, m.lclc, m.wclc, m.pclc]);
            p.cle = weights.interpolate([m.cle, m.lcle, m.wcle, m.pcle]);
            p.vfbcv = weights.interpolate([m.vfbcv, m.lvfbcv, m.wvfbcv, m.pvfbcv]);
            p.acde = weights.interpolate([m.acde, m.lacde, m.wacde, m.pacde]);
            p.moin = weights.interpolate([m.moin, m.lmoin, m.wmoin, m.pmoin]);
            p.noff = weights.interpolate([m.noff, m.lnoff, m.wnoff, m.pnoff]);
            p.voffcv = weights.interpolate([m.voffcv, m.lvoffcv, m.wvoffcv, m.pvoffcv]);
            p.kvth0we = weights.interpolate([m.kvth0we, m.lkvth0we, m.wkvth0we, m.pkvth0we]);
            p.k2we = weights.interpolate([m.k2we, m.lk2we, m.wk2we, m.pk2we]);
            p.ku0we = weights.interpolate([m.ku0we, m.lku0we, m.wku0we, m.pku0we]);
            p.abulkCVfactor = 1.0 + pow((p.clc / p.leffCV), p.cle);
            T0 = (relative_temperature_change(Temp, Tnom));
            PowWeffWr = pow(p.weffCJ * 1.0e6, p.wr) * h.nf;
            T4 = 0.0;
            T3 = T4;
            T2 = T4;
            T1 = T4;
            p.ucs = p.ucs * pow(TRatio, p.ucste);
            if m.tempMod == 0 {
                p.ua = p.ua + p.ua1 * T0;
                p.ub = p.ub + p.ub1 * T0;
                p.uc = p.uc + p.uc1 * T0;
                p.ud = p.ud + p.ud1 * T0;
                p.vsattemp = p.vsat - p.at * T0;
                T10 = p.prt * T0;
                if m.rdsMod != 0 {
                    T1 = p.rdw + T10;
                    T2 = m.rdwmin + T10;
                    T3 = p.rsw + T10;
                    T4 = m.rswmin + T10;
                }
                p.rds0 = (p.rdsw + T10) * h.nf / PowWeffWr;
                p.rdswmin = (m.rdswmin + T10) * h.nf / PowWeffWr;
            } else {
                if m.tempMod == 3 {
                    p.ua = p.ua * pow(TRatio, p.ua1);
                    p.ub = p.ub * pow(TRatio, p.ub1);
                    p.uc = p.uc * pow(TRatio, p.uc1);
                    p.ud = p.ud * pow(TRatio, p.ud1);
                } else {
                    p.ua = p.ua * (1.0 + p.ua1 * delTemp);
                    p.ub = p.ub * (1.0 + p.ub1 * delTemp);
                    p.uc = p.uc * (1.0 + p.uc1 * delTemp);
                    p.ud = p.ud * (1.0 + p.ud1 * delTemp);
                }
                p.vsattemp = p.vsat * (1.0 - p.at * delTemp);
                T10 = 1.0 + p.prt * delTemp;
                if m.rdsMod != 0 {
                    T1 = p.rdw * T10;
                    T2 = m.rdwmin * T10;
                    T3 = p.rsw * T10;
                    T4 = m.rswmin * T10;
                }
                p.rds0 = p.rdsw * T10 * h.nf / PowWeffWr;
                p.rdswmin = m.rdswmin * T10 * h.nf / PowWeffWr;
            }
            if T1 < 0.0 {
                T1 = 0.0;
                warnings
                    .push("Warning: Rdw at current temperature is negative; set to 0.".to_string());
            }
            if T2 < 0.0 {
                T2 = 0.0;
                warnings.push(
                    "Warning: Rdwmin at current temperature is negative; set to 0.".to_string(),
                );
            }
            p.rd0 = T1 / PowWeffWr;
            p.rdwmin = T2 / PowWeffWr;
            if T3 < 0.0 {
                T3 = 0.0;
                warnings
                    .push("Warning: Rsw at current temperature is negative; set to 0.".to_string());
            }
            if T4 < 0.0 {
                T4 = 0.0;
                warnings.push(
                    "Warning: Rswmin at current temperature is negative; set to 0.".to_string(),
                );
            }
            p.rs0 = T3 / PowWeffWr;
            p.rswmin = T4 / PowWeffWr;
            if p.u0 > 1.0 {
                p.u0 = p.u0 / 1.0e4;
            }
            let mobility_argument = -p.leff / p.lp;
            T5 = if mobility_argument.abs() < 0.5 {
                (-p.up).mul_add(mobility_argument.exp_m1(), 1. - p.up)
            } else {
                1. - p.up * exp(mobility_argument)
            };
            p.u0temp = p.u0 * T5 * pow(TRatio, p.ute);
            if p.eu < 0.0 {
                p.eu = 0.0;
                warnings.push("Warning: eu has been negative; reset to 0.0.".to_string());
            }
            if p.ucs < 0.0 {
                p.ucs = 0.0;
                warnings.push("Warning: ucs has been negative; reset to 0.0.".to_string());
            }
            p.vfbsdoff = p.vfbsdoff * (1.0 + p.tvfbsdoff * delTemp);
            p.voff = p.voff * (1.0 + p.tvoff * delTemp);
            p.nfactor = p.nfactor + p.tnfactor * delTemp / Tnom;
            p.voffcv = p.voffcv * (1.0 + p.tvoffcv * delTemp);
            p.eta0 = p.eta0 + p.teta0 * delTemp / Tnom;
            if (m.vtlGiven) && (m.vtl > 0.0) {
                if m.lc < 0.0 {
                    p.lc = 0.0;
                } else {
                    p.lc = m.lc;
                }
                T0 = p.leff / (p.xn * p.leff + p.lc);
                p.tfactor = (1.0 - T0) / (1.0 + T0);
            }
            p.cgdo = (m.cgdo + p.cf) * p.weffCV;
            p.cgso = (m.cgso + p.cf) * p.weffCV;
            p.cgbo = m.cgbo * p.leffCV * h.nf;
            if !m.ndepGiven && m.gamma1Given {
                T0 = p.gamma1 * m.coxe;
                p.ndep = 3.01248e22 * T0 * T0;
            }
            p.phi = Vtm0 * log(p.ndep / ni) + p.phin + 0.4;
            if p.phi <= 0.0 {
                warnings.push(format!(
                    "Fatal: Phi = {} is not positive. Please check Phin and Ndep",
                    c_g(p.phi)
                ));
                warnings.push(format!(
                    "       Phin = {}  Ndep = {} ",
                    c_g(p.phin),
                    c_g(p.ndep)
                ));
                Fatal_Flag = 1;
            }
            p.sqrtPhi = sqrt(p.phi);
            p.phis3 = p.sqrtPhi * p.phi;
            p.Xdep0 = sqrt(2.0 * epssub / (Charge_q * p.ndep * 1.0e6)) * p.sqrtPhi;
            p.sqrtXdep0 = sqrt(p.Xdep0);
            if m.mtrlMod == 0 {
                p.litl = sqrt(3.0 * 3.9 / epsrox * p.xj * toxe);
            } else {
                p.litl = sqrt(m.epsrsub / epsrox * p.xj * toxe);
            }
            p.vbi = Vtm0 * log(p.nsd * p.ndep / (ni * ni));
            if m.mtrlMod == 0 {
                if p.ngate > 0.0 {
                    p.vfbsd = Vtm0 * log(p.ngate / p.nsd);
                } else {
                    p.vfbsd = 0.0;
                }
            } else {
                T0 = Vtm0 * log(p.nsd / ni);
                T1 = 0.5 * Eg0;
                if T0 > T1 {
                    T0 = T1;
                }
                T2 = m.easub + T1 - f64::from(m.r#type) * T0;
                p.vfbsd = m.phig - T2;
            }
            p.cdep0 = sqrt(Charge_q * epssub * p.ndep * 1.0e6 / 2.0 / p.phi);
            p.ToxRatio = exp(p.ntox * log(m.toxref / toxe)) / toxe / toxe;
            p.ToxRatioEdge = exp(p.ntox * log(m.toxref / (toxe * p.poxedge)))
                / toxe
                / toxe
                / p.poxedge
                / p.poxedge;
            p.Aechvb = if m.r#type == NMOS {
                4.97232e-7
            } else {
                3.42537e-7
            };
            p.Bechvb = if m.r#type == NMOS {
                7.45669e11
            } else {
                1.16645e12
            };
            if m.v48intVersion <= BSIM4v48V480 {
                p.AechvbEdgeS = p.Aechvb * p.weff * m.dlcig * p.ToxRatioEdge;
            } else {
                if m.dlcig < 0.0 {
                    warnings.push(format!(
                        "Warning: dlcig = {} is negative. Set to zero.",
                        c_g(m.dlcig)
                    ));
                    m.dlcig = 0.0;
                }
                p.AechvbEdgeS = p.Aechvb * p.weff * m.dlcig * p.ToxRatioEdge;
                if m.dlcigd < 0.0 {
                    warnings.push(format!(
                        "Warning: dlcigd = {} is negative. Set to zero.",
                        c_g(m.dlcigd)
                    ));
                    m.dlcigd = 0.0;
                }
            }
            p.AechvbEdgeD = p.Aechvb * p.weff * m.dlcigd * p.ToxRatioEdge;
            p.BechvbEdge = -p.Bechvb * toxe * p.poxedge;
            p.Aechvb *= p.weff * p.leff * p.ToxRatio;
            p.Bechvb *= -toxe;
            p.mstar = 0.5 + atan(p.minv) / PI;
            p.mstarcv = 0.5 + atan(p.minvcv) / PI;
            p.voffcbn = p.voff + m.voffl / p.leff;
            p.voffcbncv = p.voffcv + m.voffcvl / p.leff;
            p.ldeb = sqrt(epssub * Vtm0 / (Charge_q * p.ndep * 1.0e6)) / 3.0;
            p.acde *= pow((p.ndep / 2.0e16), -0.25);
            if m.k1Given || m.k2Given {
                if !m.k1Given {
                    warnings.push("Warning: k1 should be specified with k2.".to_string());
                    p.k1 = 0.53;
                }
                if !m.k2Given {
                    warnings.push("Warning: k2 should be specified with k1.".to_string());
                    p.k2 = -0.0186;
                }
                if m.nsubGiven {
                    warnings
                        .push("Warning: nsub is ignored because k1 or k2 is given.".to_string());
                }
                if m.xtGiven {
                    warnings.push("Warning: xt is ignored because k1 or k2 is given.".to_string());
                }
                if m.vbxGiven {
                    warnings.push("Warning: vbx is ignored because k1 or k2 is given.".to_string());
                }
                if m.gamma1Given {
                    warnings
                        .push("Warning: gamma1 is ignored because k1 or k2 is given.".to_string());
                }
                if m.gamma2Given {
                    warnings
                        .push("Warning: gamma2 is ignored because k1 or k2 is given.".to_string());
                }
            } else {
                if !m.vbxGiven {
                    p.vbx = p.phi - 7.7348e-4 * p.ndep * p.xt * p.xt;
                }
                if p.vbx > 0.0 {
                    p.vbx = -p.vbx;
                }
                if p.vbm > 0.0 {
                    p.vbm = -p.vbm;
                }
                if !m.gamma1Given {
                    p.gamma1 = 5.753e-12 * sqrt(p.ndep) / m.coxe;
                }
                if !m.gamma2Given {
                    p.gamma2 = 5.753e-12 * sqrt(p.nsub) / m.coxe;
                }
                T0 = p.gamma1 - p.gamma2;
                p.k2 = body_coefficient(p.phi, p.vbx, p.vbm, T0);
                p.k1 = (-2.0 * p.k2).mul_add(sqrt(p.phi - p.vbm), p.gamma2);
            }
            if !m.vfbGiven {
                if m.vth0Given {
                    p.vfb = f64::from(m.r#type) * p.vth0 - p.phi - p.k1 * p.sqrtPhi;
                } else {
                    if (m.mtrlMod != 0) && (m.phigGiven) && (m.nsubGiven) {
                        T0 = Vtm0 * log(p.nsub / ni);
                        T1 = 0.5 * Eg0;
                        if T0 > T1 {
                            T0 = T1;
                        }
                        T2 = m.easub + T1 + f64::from(m.r#type) * T0;
                        p.vfb = m.phig - T2;
                    } else {
                        p.vfb = -1.0;
                    }
                }
            }
            if !m.vth0Given {
                p.vth0 = f64::from(m.r#type) * (p.vfb + p.phi + p.k1 * p.sqrtPhi);
            }
            p.k1ox = p.k1 * toxe / m.toxm;
            tmp = sqrt(epssub / (epsrox * EPS0) * toxe * p.Xdep0);
            T0 = p.dsub * p.leff / tmp;
            p.theta0vb0 = short_channel_factor(T0);
            T0 = p.drout * p.leff / tmp;
            T5 = short_channel_factor(T0);
            p.thetaRout = p.pdibl1 * T5 + p.pdibl2;
            tmp = sqrt(p.Xdep0);
            tmp1 = p.vbi - p.phi;
            tmp2 = m.factor1 * tmp;
            T0 = p.dvt1w * p.weff * p.leff / tmp2;
            T8 = short_channel_factor(T0);
            T0 = p.dvt0w * T8;
            T8 = T0 * tmp1;
            T0 = p.dvt1 * p.leff / tmp2;
            T9 = short_channel_factor(T0);
            T9 = p.dvt0 * T9 * tmp1;
            T4 = toxe * p.phi / (p.weff + p.w0);
            T0 = sqrt_one_plus_difference(p.lpe0 / p.leff);
            if (m.tempMod == 1) || (m.tempMod == 0) {
                T3 = (p.kt1 + p.kt1l / p.leff) * (relative_temperature_change(Temp, Tnom));
            }
            if (m.tempMod == 2) || (m.tempMod == 3) {
                T3 = -p.kt1 * (relative_temperature_change(Temp, Tnom));
            }
            T5 = p.k1ox * T0 * p.sqrtPhi + T3;
            p.vfbzbfactor = -T8 - T9 + p.k3 * T4 + T5 - p.phi - p.k1 * p.sqrtPhi;
            wlod = m.wlod;
            if m.wlod < 0.0 {
                warnings.push(format!(
                    "Warning: WLOD = {} is less than 0. 0.0 is used",
                    c_g(m.wlod)
                ));
                wlod = 0.0;
            }
            T0 = pow(Lnew, m.llodku0);
            W_tmp = Wnew + wlod;
            T1 = pow(W_tmp, m.wlodku0);
            tmp1 = m.lku0 / T0 + m.wku0 / T1 + m.pku0 / (T0 * T1);
            p.ku0 = 1.0 + tmp1;
            T0 = pow(Lnew, m.llodvth);
            T1 = pow(W_tmp, m.wlodvth);
            tmp1 = m.lkvth0 / T0 + m.wkvth0 / T1 + m.pkvth0 / (T0 * T1);
            p.kvth0 = 1.0 + tmp1;
            p.kvth0 = sqrt(p.kvth0 * p.kvth0 + DELTA);
            T0 = (relative_temperature_change(Temp, Tnom));
            p.ku0temp = p.ku0 * (1.0 + m.tku0 * T0) + DELTA;
            Inv_saref = 1.0 / (m.saref + 0.5 * Ldrn);
            Inv_sbref = 1.0 / (m.sbref + 0.5 * Ldrn);
            p.inv_od_ref = Inv_saref + Inv_sbref;
            p.rho_ref = m.ku0 / p.ku0temp * p.inv_od_ref;
            if m.mobMod == 3 {
                lt1 = m.factor1 * p.sqrtXdep0;
                T0 = p.dvt1 * p.leff / lt1;
                Theta0 = short_channel_factor(T0);
                tmp1 = epssub / p.Xdep0;
                tmp2 = p.nfactor * tmp1;
                tmp3 = (tmp2 + p.cdsc * Theta0 + p.cit) / m.coxe;
                if tmp3 >= -0.5 {
                    n0 = 1.0 + tmp3;
                } else {
                    T0 = 1.0 / (3.0 + 8.0 * tmp3);
                    n0 = (1.0 + 3.0 * tmp3) * T0;
                }
                T0 = n0 * m.vtm;
                T1 = p.voffcbn;
                T2 = T1 / T0;
                if T2 < -EXP_THRESHOLD {
                    T3 = m.coxe * MIN_EXP / p.cdep0;
                } else if T2 > EXP_THRESHOLD {
                    T3 = m.coxe * MAX_EXP / p.cdep0;
                } else {
                    T3 = exp(T2) * m.coxe / p.cdep0;
                }
                T4 = p.mstar + T3 * n0;
                p.VgsteffVth = T0 * log(2.0) / T4;
            }
            T0 = -p.dvtp3 * log(p.leff);
            T1 = dexp(T0);
            p.dvtp2factor = p.dvtp5 + p.dvtp2 * T1;
        }
        if (h.sa > 0.0) && (h.sb > 0.0) && ((h.nf == 1.0) || ((h.nf > 1.0) && (h.sd > 0.0))) {
            Inv_sa = 0.0;
            Inv_sb = 0.0;
            kvsat = m.kvsat;
            if m.kvsat < -1.0 {
                warnings.push(format!(
                    "Warning: KVSAT = {} is too small; -1.0 is used.",
                    c_g(m.kvsat)
                ));
                kvsat = -1.0;
            }
            if m.kvsat > 1.0 {
                warnings.push(format!(
                    "Warning: KVSAT = {} is too big; 1.0 is used.",
                    c_g(m.kvsat)
                ));
                kvsat = 1.0;
            }
            i = 0;
            while f64::from(i) < h.nf {
                T0 = 1.0 / h.nf / (h.sa + 0.5 * Ldrn + f64::from(i) * (h.sd + Ldrn));
                T1 = 1.0 / h.nf / (h.sb + 0.5 * Ldrn + f64::from(i) * (h.sd + Ldrn));
                Inv_sa += T0;
                Inv_sb += T1;
                i += 1;
            }
            Inv_ODeff = Inv_sa + Inv_sb;
            rho = m.ku0 / p.ku0temp * Inv_ODeff;
            T0 = (1.0 + rho) / (1.0 + p.rho_ref);
            h.u0temp = p.u0temp * T0;
            T1 = (1.0 + kvsat * rho) / (1.0 + kvsat * p.rho_ref);
            h.vsattemp = p.vsattemp * T1;
            OD_offset = Inv_ODeff - p.inv_od_ref;
            dvth0_lod = m.kvth0 / p.kvth0 * OD_offset;
            dk2_lod = m.stk2 / pow(p.kvth0, m.lodk2) * OD_offset;
            deta0_lod = m.steta0 / pow(p.kvth0, m.lodeta0) * OD_offset;
            h.vth0 = p.vth0 + dvth0_lod;
            h.eta0 = p.eta0 + deta0_lod;
            h.k2 = p.k2 + dk2_lod;
        } else {
            h.u0temp = p.u0temp;
            h.vth0 = p.vth0;
            h.vsattemp = p.vsattemp;
            h.eta0 = p.eta0;
            h.k2 = p.k2;
        }
        if m.wpemod != 0.0 {
            if (!h.scaGiven) && (!h.scbGiven) && (!h.sccGiven) {
                if (h.scGiven) && (h.sc > 0.0) {
                    h.sca = well_proximity_rational(h.sc, Wdrn, m.scref);
                    h.scb = well_proximity_exp(h.sc, Wdrn, m.scref, 10., 0.1);
                    h.scc = well_proximity_exp(h.sc, Wdrn, m.scref, 20., 0.05);
                } else {
                    warnings.push("Warning: No WPE as none of SCA, SCB, SCC, SC is given and/or SC not positive.".to_string());
                }
            }
            if h.sca < 0.0 {
                warnings.push(format!(
                    "Warning: SCA = {} is negative. Set to 0.0.",
                    c_g(h.sca)
                ));
                h.sca = 0.0;
            }
            if h.scb < 0.0 {
                warnings.push(format!(
                    "Warning: SCB = {} is negative. Set to 0.0.",
                    c_g(h.scb)
                ));
                h.scb = 0.0;
            }
            if h.scc < 0.0 {
                warnings.push(format!(
                    "Warning: SCC = {} is negative. Set to 0.0.",
                    c_g(h.scc)
                ));
                h.scc = 0.0;
            }
            if h.sc < 0.0 {
                warnings.push(format!(
                    "Warning: SC = {} is negative. Set to 0.0.",
                    c_g(h.sc)
                ));
                h.sc = 0.0;
            }
            sceff = h.sca + m.web * h.scb + m.wec * h.scc;
            h.vth0 += p.kvth0we * sceff;
            h.k2 += p.k2we * sceff;
            T3 = 1.0 + p.ku0we * sceff;
            if T3 <= 0.0 {
                T3 = 0.0;
                warnings.push(format!(
                    "Warning: ku0we = {} is negatively too high. Negative mobility! ",
                    c_g(p.ku0we)
                ));
            }
            h.u0temp *= T3;
        }
        h.vth0 += h.delvto;
        h.vfb = p.vfb + f64::from(m.r#type) * h.delvto;
        h.u0temp = p.u0temp * h.mulu0;
        T3 = f64::from(m.r#type) * h.vth0 - h.vfb - p.phi;
        T4 = T3 + T3;
        T5 = 2.5 * T3;
        h.vtfbphi1 = if m.r#type == NMOS { T4 } else { T5 };
        if h.vtfbphi1 < 0.0 {
            h.vtfbphi1 = 0.0;
        }
        h.vtfbphi2 = 4.0 * T3;
        if h.vtfbphi2 < 0.0 {
            h.vtfbphi2 = 0.0;
        }
        if h.k2 < 0.0 {
            T0 = 0.5 * p.k1 / h.k2;
            h.vbsc = 0.9 * (p.phi - T0 * T0);
            if h.vbsc > -3.0 {
                h.vbsc = -3.0;
            } else if h.vbsc < -30.0 {
                h.vbsc = -30.0;
            }
        } else {
            h.vbsc = -30.0;
        }
        if h.vbsc > p.vbm {
            h.vbsc = p.vbm;
        }
        h.k2ox = h.k2 * toxe / m.toxm;
        h.vfbzb = p.vfbzbfactor + f64::from(m.r#type) * h.vth0;
        h.cgso = p.cgso;
        h.cgdo = p.cgdo;
        lnl = log(p.leff * 1.0e6);
        lnw = log(p.weff * 1.0e6);
        lnnf = log(h.nf);
        bodymode = 5.0;
        if (!m.rbps0Given) || (!m.rbpd0Given) {
            bodymode = 1.0;
        } else if (!m.rbsbx0Given && !m.rbsby0Given) || (!m.rbdbx0Given && !m.rbdby0Given) {
            bodymode = 3.0;
        }
        if h.rbodyMod == 2 {
            let body_weights = ModelWeights([1., lnl, lnw, lnnf]);
            if bodymode == 5.0 {
                rbsbx =
                    ExpScale::new(body_weights.interpolate([0., m.rbsdbxl, m.rbsdbxw, m.rbsdbxnf]))
                        .apply(m.rbsbx0);
                rbsby =
                    ExpScale::new(body_weights.interpolate([0., m.rbsdbyl, m.rbsdbyw, m.rbsdbynf]))
                        .apply(m.rbsby0);
                h.rbsb = parallel_resistance(rbsbx, rbsby);
                rbdbx =
                    ExpScale::new(body_weights.interpolate([0., m.rbsdbxl, m.rbsdbxw, m.rbsdbxnf]))
                        .apply(m.rbdbx0);
                rbdby =
                    ExpScale::new(body_weights.interpolate([0., m.rbsdbyl, m.rbsdbyw, m.rbsdbynf]))
                        .apply(m.rbdby0);
                h.rbdb = parallel_resistance(rbdbx, rbdby);
            }
            if (bodymode == 3.0) || (bodymode == 5.0) {
                h.rbps = ExpScale::new(body_weights.interpolate([0., m.rbpsl, m.rbpsw, m.rbpsnf]))
                    .apply(m.rbps0);
                h.rbpd = ExpScale::new(body_weights.interpolate([0., m.rbpdl, m.rbpdw, m.rbpdnf]))
                    .apply(m.rbpd0);
            }
            rbpbx = ExpScale::new(body_weights.interpolate([0., m.rbpbxl, m.rbpbxw, m.rbpbxnf]))
                .apply(m.rbpbx0);
            rbpby = ExpScale::new(body_weights.interpolate([0., m.rbpbyl, m.rbpbyw, m.rbpbynf]))
                .apply(m.rbpby0);
            h.rbpb = parallel_resistance(rbpbx, rbpby);
        }
        if (h.rbodyMod == 1) || ((h.rbodyMod == 2) && (bodymode == 5.0)) {
            if h.rbdb < 1.0e-3 {
                h.grbdb = 1.0e3;
            } else {
                h.grbdb = m.gbmin + 1.0 / h.rbdb;
            }
            if h.rbpb < 1.0e-3 {
                h.grbpb = 1.0e3;
            } else {
                h.grbpb = m.gbmin + 1.0 / h.rbpb;
            }
            if h.rbps < 1.0e-3 {
                h.grbps = 1.0e3;
            } else {
                h.grbps = m.gbmin + 1.0 / h.rbps;
            }
            if h.rbsb < 1.0e-3 {
                h.grbsb = 1.0e3;
            } else {
                h.grbsb = m.gbmin + 1.0 / h.rbsb;
            }
            if h.rbpd < 1.0e-3 {
                h.grbpd = 1.0e3;
            } else {
                h.grbpd = m.gbmin + 1.0 / h.rbpd;
            }
        }
        if (h.rbodyMod == 2) && (bodymode == 3.0) {
            h.grbsb = m.gbmin;
            h.grbdb = h.grbsb;
            if h.rbpb < 1.0e-3 {
                h.grbpb = 1.0e3;
            } else {
                h.grbpb = m.gbmin + 1.0 / h.rbpb;
            }
            if h.rbps < 1.0e-3 {
                h.grbps = 1.0e3;
            } else {
                h.grbps = m.gbmin + 1.0 / h.rbps;
            }
            if h.rbpd < 1.0e-3 {
                h.grbpd = 1.0e3;
            } else {
                h.grbpd = m.gbmin + 1.0 / h.rbpd;
            }
        }
        if (h.rbodyMod == 2) && (bodymode == 1.0) {
            h.grbsb = m.gbmin;
            h.grbdb = h.grbsb;
            h.grbpd = 1.0e3;
            h.grbps = h.grbpd;
            if h.rbpb < 1.0e-3 {
                h.grbpb = 1.0e3;
            } else {
                h.grbpb = m.gbmin + 1.0 / h.rbpb;
            }
        }
        h.grgeltd = m.rshg * (h.xgw + p.weffCJ / 3.0 / h.ngcon) / (h.ngcon * h.nf * (Lnew - m.xgl));
        if h.grgeltd > 0.0 {
            h.grgeltd = 1.0 / h.grgeltd;
        } else {
            h.grgeltd = 1.0e3;
            if h.rgateMod != 0 {
                warnings.push("Warning: The gate conductance reset to 1.0e3 mho.".to_string());
            }
        }
        DMCGeff = m.dmcg - m.dmcgt;
        DMCIeff = m.dmci;
        DMDGeff = m.dmdg - m.dmcgt;
        if h.sourcePerimeterGiven {
            if h.sourcePerimeter == 0.0 {
                h.Pseff = 0.0;
            } else if h.sourcePerimeter < 0.0 {
                warnings.push(
                    "Warning: Source Perimeter is specified as negative, it is set to zero."
                        .to_string(),
                );
                h.Pseff = 0.0;
            } else {
                if m.perMod == 0 {
                    h.Pseff = h.sourcePerimeter;
                } else {
                    h.Pseff = h.sourcePerimeter - p.weffCJ * h.nf;
                }
            }
        } else {
            (h.Pseff, dumPd, dumAs, dumAd) = pa_eff_geo(
                h.nf,
                h.geoMod,
                h.min,
                p.weffCJ,
                DMCGeff,
                DMCIeff,
                DMDGeff,
                (h.Pseff, dumPd, dumAs, dumAd),
                warnings,
            );
        }
        if h.Pseff < 0.0 {
            h.Pseff = 0.0;
            warnings.push("Warning: Pseff is negative, it is set to zero.".to_string());
        }
        if h.drainPerimeterGiven {
            if h.drainPerimeter == 0.0 {
                h.Pdeff = 0.0;
            } else if h.drainPerimeter < 0.0 {
                warnings.push(
                    "Warning: Drain Perimeter is specified as negative, it is set to zero."
                        .to_string(),
                );
                h.Pdeff = 0.0;
            } else {
                if m.perMod == 0 {
                    h.Pdeff = h.drainPerimeter;
                } else {
                    h.Pdeff = h.drainPerimeter - p.weffCJ * h.nf;
                }
            }
        } else {
            (dumPs, h.Pdeff, dumAs, dumAd) = pa_eff_geo(
                h.nf,
                h.geoMod,
                h.min,
                p.weffCJ,
                DMCGeff,
                DMCIeff,
                DMDGeff,
                (dumPs, h.Pdeff, dumAs, dumAd),
                warnings,
            );
        }
        if h.Pdeff < 0.0 {
            h.Pdeff = 0.0;
            warnings.push("Warning: Pdeff is negative, it is set to zero.".to_string());
        }
        if h.sourceAreaGiven {
            h.Aseff = h.sourceArea;
        } else {
            (dumPs, dumPd, h.Aseff, dumAd) = pa_eff_geo(
                h.nf,
                h.geoMod,
                h.min,
                p.weffCJ,
                DMCGeff,
                DMCIeff,
                DMDGeff,
                (dumPs, dumPd, h.Aseff, dumAd),
                warnings,
            );
        }
        if h.Aseff < 0.0 {
            h.Aseff = 0.0;
            warnings.push("Warning: Aseff is negative, it is set to zero.".to_string());
        }
        if h.drainAreaGiven {
            h.Adeff = h.drainArea;
        } else {
            (dumPs, dumPd, dumAs, h.Adeff) = pa_eff_geo(
                h.nf,
                h.geoMod,
                h.min,
                p.weffCJ,
                DMCGeff,
                DMCIeff,
                DMDGeff,
                (dumPs, dumPd, dumAs, h.Adeff),
                warnings,
            );
        }
        if h.Adeff < 0.0 {
            h.Adeff = 0.0;
            warnings.push("Warning: Adeff is negative, it is set to zero.".to_string());
        }
        if h.sNodePrime == h.sNode {
            h.sourceConductance = 0.0;
        } else {
            h.sourceConductance = 0.0;
            if h.sourceSquaresGiven {
                h.sourceConductance = m.sheetResistance * h.sourceSquares;
            } else if h.rgeoMod > 0 {
                h.sourceConductance = rdseff_geo(
                    h.nf,
                    h.geoMod,
                    h.rgeoMod,
                    h.min,
                    p.weffCJ,
                    m.sheetResistance,
                    DMCGeff,
                    DMCIeff,
                    DMDGeff,
                    1,
                    warnings,
                );
            } else {
                h.sourceConductance = 0.0;
            }
            if h.sourceConductance > 0.0 {
                h.sourceConductance = 1.0 / h.sourceConductance;
            } else {
                h.sourceConductance = 1.0e3;
                warnings.push("Warning: Source conductance reset to 1.0e3 mho.".to_string());
            }
        }
        if h.dNodePrime == h.dNode {
            h.drainConductance = 0.0;
        } else {
            h.drainConductance = 0.0;
            if h.drainSquaresGiven {
                h.drainConductance = m.sheetResistance * h.drainSquares;
            } else if h.rgeoMod > 0 {
                h.drainConductance = rdseff_geo(
                    h.nf,
                    h.geoMod,
                    h.rgeoMod,
                    h.min,
                    p.weffCJ,
                    m.sheetResistance,
                    DMCGeff,
                    DMCIeff,
                    DMDGeff,
                    0,
                    warnings,
                );
            } else {
                h.drainConductance = 0.0;
            }
            if h.drainConductance > 0.0 {
                h.drainConductance = 1.0 / h.drainConductance;
            } else {
                h.drainConductance = 1.0e3;
                warnings.push("Warning: Drain conductance reset to 1.0e3 mho.".to_string());
            }
        }
        // Preserve source-before-drain C field updates and warning order.
        // The shared evaluator never mutates the instance or model.
        let source = prepare_junction_temperature(JunctionTemperatureInput {
            area: h.Aseff,
            perimeter: h.Pseff,
            width: p.weffCJ,
            fingers: h.nf,
            area_density: m.SjctTempSatCurDensity,
            sidewall_density: m.SjctSidewallTempSatCurDensity,
            gate_sidewall_density: m.SjctGateSidewallTempSatCurDensity,
            emission_voltage: m.vtm * m.SjctEmissionCoeff,
            breakdown_voltage: m.bvs,
            breakdown_factor: m.xjbvs,
            forward_threshold: m.ijthsfwd,
            reverse_threshold: m.ijthsrev,
            diode_mode: m.dioMod,
        });
        if let Some(value) = source.breakdown_exponential {
            h.XExpBVS = value;
        }
        if let Some(limit) = source.forward {
            h.vjsmFwd = limit.voltage;
            h.IVjsmFwd = limit.current;
            if m.dioMod == 2 {
                h.SslpFwd = limit.slope;
            }
        }
        if let Some(limit) = source.reverse {
            h.vjsmRev = limit.voltage;
            h.IVjsmRev = limit.current;
            h.SslpRev = limit.slope;
        }
        match source.warning {
            Some(JunctionTemperatureWarning::ReverseThresholdTooSmall) => {
                warnings
                    .push("Warning: ijthsrev too small and set to 10 times IsbSat.".to_string());
            }
            Some(JunctionTemperatureWarning::UnknownDiodeMode) => {
                warnings.push(format!("Specified dioMod = {} not matched", m.dioMod));
            }
            None => {}
        }
        let drain = prepare_junction_temperature(JunctionTemperatureInput {
            area: h.Adeff,
            perimeter: h.Pdeff,
            width: p.weffCJ,
            fingers: h.nf,
            area_density: m.DjctTempSatCurDensity,
            sidewall_density: m.DjctSidewallTempSatCurDensity,
            gate_sidewall_density: m.DjctGateSidewallTempSatCurDensity,
            emission_voltage: m.vtm * m.DjctEmissionCoeff,
            breakdown_voltage: m.bvd,
            breakdown_factor: m.xjbvd,
            forward_threshold: m.ijthdfwd,
            reverse_threshold: m.ijthdrev,
            diode_mode: m.dioMod,
        });
        if let Some(value) = drain.breakdown_exponential {
            h.XExpBVD = value;
        }
        if let Some(limit) = drain.forward {
            h.vjdmFwd = limit.voltage;
            h.IVjdmFwd = limit.current;
            if m.dioMod == 2 {
                h.DslpFwd = limit.slope;
            }
        }
        if let Some(limit) = drain.reverse {
            h.vjdmRev = limit.voltage;
            h.IVjdmRev = limit.current;
            h.DslpRev = limit.slope;
        }
        match drain.warning {
            Some(JunctionTemperatureWarning::ReverseThresholdTooSmall) => {
                warnings
                    .push("Warning: ijthdrev too small and set to 10 times IdbSat.".to_string());
            }
            Some(JunctionTemperatureWarning::UnknownDiodeMode) => {
                warnings.push(format!("Specified dioMod = {} not matched", m.dioMod));
            }
            None => {}
        }
        T0 = (relative_temperature_change(Temp, Tnom));
        m.njtsstemp = m.njts * (1.0 + m.tnjts * T0);
        m.njtsswstemp = m.njtssw * (1.0 + m.tnjtssw * T0);
        m.njtsswgstemp = m.njtsswg * (1.0 + m.tnjtsswg * T0);
        m.njtsdtemp = m.njtsd * (1.0 + m.tnjtsd * T0);
        m.njtsswdtemp = m.njtsswd * (1.0 + m.tnjtsswd * T0);
        m.njtsswgdtemp = m.njtsswgd * (1.0 + m.tnjtsswgd * T0);
        T7 = Eg0 / m.vtm * T0;
        T9 = m.xtss * T7;
        T1 = dexp(T9);
        T9 = m.xtsd * T7;
        T2 = dexp(T9);
        T9 = m.xtssws * T7;
        T3 = dexp(T9);
        T9 = m.xtsswd * T7;
        T4 = dexp(T9);
        T9 = m.xtsswgs * T7;
        T5 = dexp(T9);
        T9 = m.xtsswgd * T7;
        T6 = dexp(T9);
        if m.jtweff < 0.0 {
            m.jtweff = 0.0;
            warnings.push(
                "TAT width dependence effect is negative. Jtweff is clamped to zero.".to_string(),
            );
        }
        T11 = sqrt(m.jtweff / p.weffCJ) + 1.0;
        T10 = p.weffCJ * h.nf;
        h.SjctTempRevSatCur = T1 * h.Aseff * m.jtss;
        h.DjctTempRevSatCur = T2 * h.Adeff * m.jtsd;
        h.SswTempRevSatCur = T3 * h.Pseff * m.jtssws;
        h.DswTempRevSatCur = T4 * h.Pdeff * m.jtsswd;
        h.SswgTempRevSatCur = T5 * T10 * T11 * m.jtsswgs;
        h.DswgTempRevSatCur = T6 * T10 * T11 * m.jtsswgd;
        if m.mtrlMod != 0 && m.mtrlCompatMod == 0 {
            Vtm0eot = KboQ * m.tempeot;
            Vtmeot = Vtm0eot;
            vbieot = Vtm0eot * log(p.nsd * p.ndep / (ni * ni));
            phieot = Vtm0eot * log(p.ndep / ni) + p.phin + 0.4;
            if phieot <= 0.0 {
                warnings.push(format!(
                    "Fatal: phieot = {} is not positive. Please Check Phin and Ndep",
                    c_g(p.phi)
                ));
                warnings.push(format!("phieot = {} Ndep = {} ", c_g(p.phin), c_g(p.ndep)));
                Fatal_Flag = 1;
            }
            tmp2 = h.vfb + phieot;
            vddeot = f64::from(m.r#type) * m.vddeot;
            T0 = m.epsrgate * EPS0;
            Vgs_eff = super::load::BSIM4polyDepletion(tmp2, p.ngate, T0, m.coxe, vddeot).0;
            V0 = vbieot - phieot;
            lt1 = m.factor1 * p.sqrtXdep0;
            ltw = lt1;
            T0 = p.dvt1 * m.leffeot / lt1;
            Theta0 = short_channel_factor(T0);
            Delt_vth = p.dvt0 * Theta0 * V0;
            T0 = p.dvt1w * m.weffeot * m.leffeot / ltw;
            T5 = short_channel_factor(T0);
            T2 = p.dvt0w * T5 * V0;
            TempRatioeot = relative_temperature_change(m.tempeot, m.tnom);
            T0 = sqrt_one_plus_difference(p.lpe0 / m.leffeot);
            T1 = p.k1ox * T0 * sqrt(phieot) + (p.kt1 + p.kt1l / m.leffeot) * TempRatioeot;
            let width_with_offset = m.weffeot + p.w0;
            Vth_NarrowW = toxe * phieot / width_with_offset;
            Lpe_Vb = sqrt(1.0 + p.lpeb / m.leffeot);
            Vth = f64::from(m.r#type) * h.vth0 + (p.k1ox - p.k1) * sqrt(phieot) * Lpe_Vb
                - Delt_vth
                - T2
                + p.k3 * Vth_NarrowW
                + T1;
            tmp1 = epssub / p.Xdep0;
            tmp2 = p.nfactor * tmp1;
            tmp3 = (tmp2 + p.cdsc * Theta0 + p.cit) / m.coxe;
            if tmp3 >= -0.5 {
                n = 1.0 + tmp3;
            } else {
                T0 = 1.0 / (3.0 + 8.0 * tmp3);
                n = (1.0 + 3.0 * tmp3) * T0;
            }
            if p.dvtp0 > 0.0 {
                T3 = m.leffeot + p.dvtp0 * 2.0;
                if m.tempMod < 2 {
                    T4 = -Vtmeot * (p.dvtp0 * 2.0 / m.leffeot).ln_1p();
                } else {
                    T4 = -Vtm0eot * (p.dvtp0 * 2.0 / m.leffeot).ln_1p();
                }
                Vth -= n * T4;
            }
            Vgsteff = Vgs_eff - Vth;
            T3 = f64::from(m.r#type) * h.vth0 - h.vfb - phieot;
            T4 = T3 + T3;
            T5 = 2.5 * T3;
            vtfbphi2eot = 4.0 * T3;
            if vtfbphi2eot < 0.0 {
                vtfbphi2eot = 0.0;
            }
            niter = 0.0;
            toxpf = toxe;
            loop {
                toxpi = toxpf;
                tmp2 = 2.0e8 * toxpf;
                T0 = (Vgsteff + vtfbphi2eot) / tmp2;
                T1 = 1.0 + exp(m.bdos * 0.7 * log(T0));
                Tcen = m.ados * 1.9e-9 / T1;
                toxpf = toxe - epsrox / m.epsrsub * Tcen;
                niter += 1.0;
                if !((niter <= 4.0) && (ABS(toxpf - toxpi) > 1e-12)) {
                    break;
                }
            }
            h.toxp = toxpf;
            h.coxp = epsrox * EPS0 / h.toxp;
        } else {
            h.toxp = m.toxp;
            h.coxp = m.coxp;
        }
        let mut messages = Vec::new();
        if check && check_model(m, h, &mut p, Temp, model_name, &mut messages) {
            messages.push(format!(
                    "detected during BSIM4.8.3 parameter checking for \n    model {model_name} of device instance {instance_name}\n"
                ));
            return Err(bad_parameter(messages.join("\n")));
        }
        warnings.append(&mut messages);
    }
    Ok(p)
}

/// [`temperature_model`] then [`temperature_instance`] for a single
/// instance whose `BSIM4checkModel` C runs.
///
/// # Errors
/// As [`temperature_instance`].
pub(super) fn temperature(
    m: &mut Model,
    h: &mut Instance,
    temperature: f64,
    model_name: &str,
    instance_name: &str,
    warnings: &mut Vec<String>,
) -> SpiceResult<SizeParam> {
    let locals = temperature_model(m, temperature, warnings);
    temperature_instance(m, h, &locals, true, model_name, instance_name, warnings)
}

/// `(model members, instance asks, messages)` of
/// [`super::ModelParameters::temperature_values`].
type TemperatureValues = (
    Vec<(&'static str, f64)>,
    Vec<(&'static str, f64)>,
    Vec<String>,
);

impl super::ModelParameters {
    /// A verification hook for the temperature stage: one instance set up
    /// by `b4set.c` and processed by `b4temp.c` (with its `b4check.c`
    /// check) at `temperature_celsius`, nominal `nominal_celsius`. Returns
    /// every numeric model member C's `@model[keyword]` ask reads (after
    /// `b4temp.c`), the instance asks `b4ask.c` answers from values
    /// `b4temp.c` leaves and `b4ld.c` does not overwrite (`weff`, `leff`,
    /// `cgbo`, the body resistances, the WPE distances), and the messages
    /// in C order. Not a simulation result.
    ///
    /// # Errors
    /// Invalid input or temperatures, or a fatal setup/temperature check.
    pub fn temperature_values(
        &self,
        instance: &crate::netlist::ast::DeviceInstance,
        temperature_celsius: f64,
        nominal_celsius: f64,
    ) -> SpiceResult<TemperatureValues> {
        use super::host::node;
        use crate::devices::schema::temperature_kelvin;
        let tnom = temperature_kelvin(nominal_celsius, &self.location)?;
        let temp = temperature_kelvin(temperature_celsius, &instance.location)?;
        let prepared = super::set::prepare_model(super::apply::model(self)?, tnom);
        let mut h = super::apply::instance(&self.instance_parameters(instance, 1.)?)?;
        (h.dNode, h.gNodeExt, h.sNode, h.bNode) = (node::D, node::G_EXT, node::S, node::B);
        let super::set::PreparedSetup {
            model: mut m,
            instance: mut h,
            mut warnings,
            ..
        } = prepared.prepare_instance(h, false);
        let model_name = instance.model.as_deref().unwrap_or_default();
        let p = temperature(
            &mut m,
            &mut h,
            temp,
            model_name,
            &instance.name,
            &mut warnings,
        )?;
        let mut model = Vec::new();
        for parameter in super::parameters::MODEL.iter().filter(|p| !p.redundant) {
            if !matches!(
                parameter.input_type,
                super::InputType::Real | super::InputType::Integer
            ) {
                continue;
            }
            let Some(field) = parameter.c_field else {
                continue;
            };
            let value = if let Some(v) = m.real_mut(field) {
                *v
            } else if let Some(v) = m.int_mut(field) {
                f64::from(*v)
            } else {
                return Err(SpiceError::circuit(format!(
                    "BSIM4 model member {field} is missing"
                )));
            };
            model.push((parameter.name, value));
        }
        let asks = vec![
            ("weff", p.weff),
            ("leff", p.leff),
            ("cgbo", p.cgbo * h.mult_q * h.m),
            ("rbdb", h.rbdb),
            ("rbsb", h.rbsb),
            ("rbpb", h.rbpb),
            ("rbps", h.rbps),
            ("rbpd", h.rbpd),
            ("sca", h.sca),
            ("scb", h.scb),
            ("scc", h.scc),
            ("sc", h.sc),
        ];
        Ok((model, asks, warnings))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ijth_vjm_inverts_the_limited_diode_current() {
        // With XExpBV = 0 the limit is where Isb * (exp(V/Nvtm) - 1) = Ijth.
        let v = dio_ijth_vjm_eval(0.025, 0.1, 1e-14, 0.0);
        let current = 1e-14 * ((v / 0.025).exp() - 1.0);
        assert!((current - 0.1).abs() < 1e-12, "{current}");
    }

    #[test]
    fn thermal_exponent_preserves_adjacent_temperature() {
        let nominal = 300.15_f64;
        let temperature = nominal.next_up();
        let relative = relative_temperature_change(temperature, nominal);
        assert!((relative - (temperature - nominal) / nominal).abs() <= f64::EPSILON * relative);
        // 80-digit Decimal evaluation of the exact binary64 inputs.
        let expected = 8.752_540_934_682_474e-15;
        let actual = thermal_bandgap_difference(nominal, temperature, 1.16, 7.02e-4, 1108.);
        assert!((actual - expected).abs() < 4. * f64::EPSILON * expected);
        for temperature in [100., 250., 300.15, 400., 1000.] {
            let actual = thermal_bandgap_difference(nominal, temperature, 1.16, 7.02e-4, 1108.);
            let reverse = thermal_bandgap_difference(temperature, nominal, 1.16, 7.02e-4, 1108.);
            assert!((actual + reverse).abs() < 8. * f64::EPSILON * actual.abs() + 1e-30);
        }
        assert!(log_temperature_ratio(1e-300, 1e300).is_finite());
        assert!(log_temperature_ratio(1e300, 1e-300).is_finite());
    }

    #[test]
    fn body_effect_keeps_the_negative_denominator_at_small_bias() {
        for (vbm, expected) in [
            (-1e-3, -32_372.714_843_375_565),
            (-1e-6, -32_349_635_207.732_25),
            (-1e-9, -3.234_961_212_397_745e16),
            (-1e-12, -3.234_961_210_089_369_6e22),
        ] {
            let actual = body_coefficient(0.7, -0.1, vbm, 0.2);
            assert!((actual - expected).abs() < 8. * f64::EPSILON * expected.abs());
        }
        assert_eq!(sqrt_one_plus_difference(1e-20), 5e-21);
    }

    #[test]
    fn short_channel_factor_matches_high_precision_near_zero_and_clamp() {
        for (argument, expected) in [
            (1e-12, 291_730_871_270_878.6),
            (1e-8, 283_461_426_462_222.06),
            (1e-7, 74_472_269_787_201.64),
            (0.1, 99.916_708_316_770_5),
        ] {
            let actual = short_channel_factor(argument);
            assert!((actual - expected).abs() < 8. * f64::EPSILON * expected);
        }
        assert_eq!(short_channel_factor(EXP_THRESHOLD), 1. / (MAX_EXP - 2.));
        assert_eq!(short_channel_factor(f64::NAN), 1. / (MAX_EXP - 2.));
    }

    #[test]
    fn diode_limit_inverts_current_even_when_exp_rounds_to_one() {
        for (ratio, coefficient) in [
            (1e-18, 0.),
            (1e-18, 1.),
            (1e-12, 1e8),
            (1e-4, 1e12),
            (1e-6, 0.01),
            (1e100, 0.1),
            (1e300, 1.),
        ] {
            let nvtm = 0.025;
            let limit = forward_limit(nvtm, ratio, 1., Some(coefficient));
            let z = (limit.voltage / nvtm).exp_m1();
            let reconstructed = z + coefficient * (z / (1. + z));
            assert!(
                (reconstructed / ratio - 1.).abs() < 2e-13,
                "ratio={ratio:e}, breakdown={coefficient:e}"
            );
            assert_eq!(limit.current, ratio);
            let y = 1. + z;
            let slope = (y + coefficient / y) / nvtm;
            assert!((limit.slope / slope - 1.).abs() < 2e-13);
        }
        let limit = forward_limit(0.025, 1e100, 1e-300, Some(0.1));
        assert!(limit.voltage.is_finite());
        assert!((limit.voltage / 0.025 - (1e100_f64.ln() - 1e-300_f64.ln())).abs() < 1e-12);
        assert_eq!(limit.current, 1e100);
        assert!((limit.slope / 4e101 - 1.).abs() < 1e-14);
        assert_eq!(forward_limit(0.025, 0., 1., Some(1e100)).voltage, 0.);
    }

    #[test]
    #[ignore = "opt-in representative normal/cancellation BSIM4 model-bin throughput"]
    fn model_bin_interpolation_benchmark() {
        use std::{hint::black_box, time::Instant};
        const ROWS: usize = 16_384;
        const REPETITIONS: usize = 159;
        let weights = ModelWeights([1., 1., 1., 1.]);
        for mode in ["sparse", "dense", "severe"] {
            let rows: Vec<_> = (0..ROWS)
                .map(|index| {
                    let sign = if index % 2 == 0 { 1. } else { -1. };
                    let x = f64::from(u32::try_from(index % 173).unwrap()) / 173.;
                    let coefficients = match mode {
                        "sparse" if index % 10 != 0 => [1. + x, 0., 0., 0.],
                        "sparse" => [1. + x, 0.01 * sign, 0.02 * x, -0.005 * sign],
                        "dense" => [1. + x, 0.01 * sign, 0.02 * x, -0.005 * sign],
                        "severe" if index % 10 == 0 => [1e16, 1., -1e16, 0.],
                        "severe" => [1. + x, 0.01 * sign, 0.02 * x, -0.005 * sign],
                        _ => unreachable!(),
                    };
                    (weights, coefficients)
                })
                .collect();
            let mut times = Vec::new();
            let mut expected = None;
            for pass in 0..6 {
                let start = Instant::now();
                let mut checksum = 0_u64;
                for _ in 0..REPETITIONS {
                    for &(weights, coefficients) in black_box(&rows) {
                        checksum = checksum.wrapping_add(
                            black_box(weights.interpolate(black_box(coefficients)))
                                .to_bits()
                                .rotate_left(17),
                        );
                    }
                }
                let duration = start.elapsed();
                if let Some(previous) = expected {
                    assert_eq!(previous, checksum, "unstable {mode} checksum");
                }
                expected = Some(checksum);
                if pass > 0 {
                    times.push(duration.as_nanos());
                }
            }
            times.sort_unstable();
            println!(
                "model_bin_{mode}: median_ns_per_field={:.3} samples={} checksum={:#x}",
                times[times.len() / 2] as f64 / (ROWS * REPETITIONS) as f64,
                ROWS * REPETITIONS,
                expected.unwrap(),
            );
        }
    }

    #[test]
    fn zero_correction_binning_preserves_all_ieee_cases_and_signed_zeros() {
        let weights = [
            [1.; 4],
            [1., 1e308, 1e-300, 0.],
            [1., -1., 1., 2.],
            [1., -0., 0., 0.],
            [1., f64::INFINITY, 1., 1.],
            [1., f64::NAN, 1., 1.],
            [1., f64::NEG_INFINITY, 1., 1.],
        ];
        let bases = [
            0.,
            -0.,
            1.,
            -1.,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::MIN_POSITIVE.next_down(),
            f64::MAX,
            -f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for inputs in weights {
            let weight = ModelWeights(inputs);
            for base in bases {
                for signs in 0..8_u8 {
                    let correction = |bit: u32| if signs & (1_u8 << bit) == 0 { 0. } else { -0. };
                    let (l, w, cross) = (correction(0), correction(1), correction(2));
                    let direct = base + l * inputs[1] + w * inputs[2] + cross * inputs[3];
                    let result = weight.interpolate([base, l, w, cross]);
                    if direct.is_nan() {
                        assert!(result.is_nan(), "{inputs:?}, {base:?}, signs={signs}");
                    } else {
                        assert_eq!(
                            result.to_bits(),
                            direct.to_bits(),
                            "{inputs:?}, {base:?}, signs={signs}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn model_bin_recovery_rounds_cancelled_products_once() {
        let unit = ModelWeights([1.; 4]);
        assert_eq!(unit.interpolate([1e16, 1., -1e16, 0.]), 1.);
        let eps = f64::EPSILON;
        let product = ModelWeights([1., 1. - eps, 1., 1.]);
        assert_eq!(product.interpolate([-1., 1. + eps, 0., 0.]), -eps * eps);
        let wide = ModelWeights([1., 2., 2., 1.]);
        assert_eq!(wide.interpolate([0., 1e308, -1e308, 1.]), 1.);
        assert!(unit.interpolate([f64::INFINITY, 0., 0., 0.]).is_infinite());
        assert!(
            unit.interpolate([f64::INFINITY, f64::NEG_INFINITY, 0., 0.])
                .is_nan()
        );
        assert_eq!(unit.interpolate([-0.; 4]).to_bits(), (-0_f64).to_bits());
    }

    #[test]
    fn exponential_density_scaling_recovers_finite_products() {
        // Independently evaluated with 100-digit Decimal arithmetic.
        for (density, exponent, expected) in [
            (1e-300, 1000., 1.970_071_114_017_047e134),
            (1e300, -1000., 5.075_958_897_549_457e-135),
            (1e308, -745., 2.822_350_730_471_937e-16),
            (1e-308, 710., 2.233_994_766_161_711),
        ] {
            let actual = ExpScale::new(exponent).apply(density);
            assert!((actual / expected - 1.).abs() < 8. * f64::EPSILON);
        }
        assert_eq!(
            ExpScale::new(1000.).apply(-0.).to_bits(),
            (-0_f64).to_bits()
        );
        assert!(ExpScale::new(f64::NAN).apply(0.).is_nan());
        assert!(ExpScale::new(f64::INFINITY).apply(0.).is_nan());
        assert_eq!(ExpScale::new(3000.).apply(1e-300), f64::INFINITY);
        assert_eq!(ExpScale::new(-3000.).apply(1e300), 0.);
    }

    #[test]
    fn well_proximity_integrals_retain_the_narrow_width_limit() {
        // 100-digit Decimal evaluation of the original endpoint integrals.
        for (distance, width, rate, scale, expected) in [
            (1e-6, 1e-24, 10., 0.1, 4.539_992_976_248_485e-5),
            (1e-30, 1e-30, 10., 0.1, 1.500_000_000_000_000_3e-24),
            (1e-6, 1e-6, 10., 0.1, 4.993_559_431_612_621_5e-6),
            (1e-6, 1e-6, 20., 0.05, 1.082_105_647_425_679_8e-10),
            (1e-7, 5e-8, 10., 0.1, 0.035_586_696_394_362_016),
            (
                1e-7,
                5.000_000_000_000_001e-8,
                10.,
                0.1,
                0.035_586_696_394_362_016,
            ),
        ] {
            let actual = well_proximity_exp(distance, width, 1e-6, rate, scale);
            assert!(
                (actual / expected - 1.).abs() < 16. * f64::EPSILON,
                "{actual:e}"
            );
        }
        assert_eq!(well_proximity_rational(1e200, 1e200, 1e200), 0.5);
        assert_eq!(well_proximity_rational(1e-200, 1e-200, 1e-200), 0.5);
        assert_eq!(parallel_resistance(1e308, 1e308), 5e307);
        assert_eq!(parallel_resistance(1e-300, 1e-300), 5e-301);
    }

    #[test]
    fn bin_weight_survives_intermediate_dimension_product_range() {
        let small = cross_bin_weight(1e-12, 1e-160, 1e-160);
        assert!((small / 1e308 - 1.).abs() < 4. * f64::EPSILON);
        let large = cross_bin_weight(1., 1e160, 1e150);
        assert!(large > 0.);
        assert!((large / 1e-310 - 1.).abs() < 1e-12);
    }

    #[test]
    fn prepared_instance_consumes_recovered_binned_threshold() {
        use super::super::set;
        let model = Model {
            vth0: 1e16,
            vth0Given: true,
            lvth0: 0.5,
            lvth0Given: true,
            wvth0: -1e16,
            wvth0Given: true,
            k1: 0.4,
            k1Given: true,
            k2: 0.,
            k2Given: true,
            binUnit: 1,
            binUnitGiven: true,
            ..Model::default()
        };
        let instance = Instance {
            l: 1e-6,
            lGiven: true,
            w: 1e-6,
            wGiven: true,
            ..Instance::default()
        };
        let set::PreparedSetup {
            mut model,
            mut instance,
            mut warnings,
            ..
        } = set::prepare_model(model, 300.15).prepare_instance(instance, false);
        let size = temperature(
            &mut model,
            &mut instance,
            300.15,
            "nch",
            "m1",
            &mut warnings,
        )
        .unwrap();
        assert_eq!(size.leff, 1e-6);
        assert_eq!(size.weff, 1e-6);
        assert_eq!(size.vth0, 0.5);
        assert_eq!(instance.vth0, 0.5);
        assert!(model.vth0Given && model.lvth0Given && model.wvth0Given);
        assert_eq!(model.vth0, 1e16);
    }

    #[test]
    fn c_g_spells_like_printf() {
        for (value, text) in [
            (0.0, "0"),
            (0.1, "0.1"),
            (-1e-9, "-1e-09"),
            (123_456.0, "123456"),
            (100_000.0, "100000"),
            (-120_000.0, "-120000"),
            (999_999.4, "999999"),
            (999_999.6, "1e+06"),
            (-0.0, "-0"),
            (f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (1_234_567.0, "1.23457e+06"),
            (1e-4, "0.0001"),
            (-2.64404e10, "-2.64404e+10"),
        ] {
            assert_eq!(c_g(value), text);
        }
    }

    #[test]
    fn dexp_follows_the_c_macro_branches() {
        assert_eq!(dexp(35.0), MAX_EXP * 2.0);
        assert_eq!(dexp(-35.0), MIN_EXP);
        assert_eq!(dexp(1.0), 1f64.exp());
    }
}

#[cfg(test)]
mod junction_preparation_tests {
    use super::{
        EXP_THRESHOLD, ForwardLimit, JunctionTemperatureInput as Input,
        JunctionTemperaturePrepared as Prepared, JunctionTemperatureWarning as Warning, MIN_EXP,
        ReverseLimit, exp, forward_limit, log, prepare_junction_temperature,
    };

    /// Independent spelling of the pre-extraction C source/drain blocks,
    /// including their distinct diode-mode field-write sets.
    fn original_c_reference(input: Input) -> Prepared {
        let mut result = Prepared::default();
        let nvtm = input.emission_voltage;
        let saturation = if (input.area <= 0.0) && (input.perimeter <= 0.0) {
            0.0
        } else {
            input.area * input.area_density
                + input.perimeter * input.sidewall_density
                + input.width * input.fingers * input.gate_sidewall_density
        };
        if saturation > 0.0 {
            match input.diode_mode {
                0 => {
                    if (input.breakdown_voltage / nvtm) > EXP_THRESHOLD {
                        result.breakdown_exponential = Some(input.breakdown_factor * MIN_EXP);
                    } else {
                        result.breakdown_exponential =
                            Some(input.breakdown_factor * exp(-input.breakdown_voltage / nvtm));
                    }
                }
                1 => {
                    result.forward = Some(forward_limit(
                        nvtm,
                        input.forward_threshold,
                        saturation,
                        None,
                    ));
                }
                2 => {
                    let xexp = if (input.breakdown_voltage / nvtm) > EXP_THRESHOLD {
                        input.breakdown_factor * MIN_EXP
                    } else {
                        let unweighted = exp(-input.breakdown_voltage / nvtm);
                        unweighted * input.breakdown_factor
                    };
                    result.breakdown_exponential = Some(xexp);
                    result.forward = Some(forward_limit(
                        nvtm,
                        input.forward_threshold,
                        saturation,
                        Some(xexp),
                    ));
                    let mut ratio = input.reverse_threshold / saturation;
                    if ratio < 1.0 {
                        ratio = 10.0;
                        result.warning = Some(Warning::ReverseThresholdTooSmall);
                    }
                    let voltage = -input.breakdown_voltage
                        - nvtm * log((ratio - 1.0) / input.breakdown_factor);
                    let t1 =
                        input.breakdown_factor * exp(-(input.breakdown_voltage + voltage) / nvtm);
                    result.reverse = Some(ReverseLimit {
                        voltage,
                        current: saturation * (1.0 + t1),
                        slope: -saturation * t1 / nvtm,
                    });
                }
                _ => result.warning = Some(Warning::UnknownDiodeMode),
            }
        }
        result
    }

    fn compare_float(actual: f64, original: f64, label: &str) {
        assert_eq!(
            actual.to_bits(),
            original.to_bits(),
            "{label}: actual {actual:.17e} != C {original:.17e}"
        );
    }

    fn compare_opt(actual: Option<f64>, original: Option<f64>, label: &str) {
        match (actual, original) {
            (Some(a), Some(b)) => compare_float(a, b, label),
            (None, None) => {}
            _ => panic!("{label}: C and common kernel write different field sets"),
        }
    }

    fn compare_prepared(actual: Prepared, original: Prepared) {
        assert_eq!(actual.warning, original.warning);
        compare_opt(
            actual.breakdown_exponential,
            original.breakdown_exponential,
            "breakdown exponential",
        );
        let compare_forward =
            |actual: Option<ForwardLimit>, original: Option<ForwardLimit>| match (actual, original)
            {
                (Some(a), Some(b)) => {
                    compare_float(a.voltage, b.voltage, "forward voltage");
                    compare_float(a.current, b.current, "forward current");
                    compare_float(a.slope, b.slope, "forward slope");
                }
                (None, None) => {}
                _ => panic!("different forward member write set"),
            };
        compare_forward(actual.forward, original.forward);
        match (actual.reverse, original.reverse) {
            (Some(a), Some(b)) => {
                compare_float(a.voltage, b.voltage, "reverse voltage");
                compare_float(a.current, b.current, "reverse current");
                compare_float(a.slope, b.slope, "reverse slope");
            }
            (None, None) => {}
            _ => panic!("different reverse member write set"),
        }
    }

    #[test]
    fn shared_kernel_matches_original_source_and_drain_across_boundaries() {
        let mut compared = 0;
        for mode in [0, 1, 2, 4] {
            for area in [-0.0, 0.0, 1e-12, f64::NAN] {
                for perimeter in [0.0, 2e-6] {
                    for breakdown in [
                        0.0,
                        EXP_THRESHOLD,
                        f64::from_bits(EXP_THRESHOLD.to_bits() + 1),
                        f64::INFINITY,
                    ] {
                        for ratio in [0.5, 1.0, 10.0, f64::NAN] {
                            for side in [0, 1] {
                                let density = if side == 0 { 1e-4 } else { 3e-5 };
                                let input = Input {
                                    area,
                                    perimeter,
                                    width: if side == 0 { 1e-6 } else { 2e-6 },
                                    fingers: if side == 0 { 1.0 } else { 4.0 },
                                    area_density: density,
                                    sidewall_density: 1e-10,
                                    gate_sidewall_density: 1e-10,
                                    emission_voltage: 1.0,
                                    breakdown_voltage: breakdown,
                                    breakdown_factor: if side == 0 { 1.5 } else { 0.5 },
                                    forward_threshold: 2e-12,
                                    reverse_threshold: ratio
                                        * (area * density
                                            + perimeter * 1e-10
                                            + (if side == 0 { 1e-6 } else { 2e-6 })
                                                * (if side == 0 { 1.0 } else { 4.0 })
                                                * 1e-10),
                                    diode_mode: mode,
                                };
                                compare_prepared(
                                    prepare_junction_temperature(input),
                                    original_c_reference(input),
                                );
                                compared += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(compared, 1024);
    }

    #[test]
    fn no_saturation_never_writes_any_limiter_or_emits_warnings() {
        let input = Input {
            area: 0.,
            perimeter: 0.,
            width: 1e-6,
            fingers: 3.,
            area_density: 1e-4,
            sidewall_density: 1e-10,
            gate_sidewall_density: 1e-4,
            emission_voltage: 0.025,
            breakdown_voltage: 10.,
            breakdown_factor: 2.,
            forward_threshold: 1e-12,
            reverse_threshold: 1e-13,
            diode_mode: 2,
        };
        // Gate sidewall current is not admitted when both area/perimeter
        // have no effective junction surface, even if width*fingers is set.
        let result = prepare_junction_temperature(input);
        assert!(result.breakdown_exponential.is_none());
        assert!(result.forward.is_none());
        assert!(result.reverse.is_none());
        assert!(result.warning.is_none());
    }
}
