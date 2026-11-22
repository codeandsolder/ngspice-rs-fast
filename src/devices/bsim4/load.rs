//! `b4ld.c` (`BSIM4load`): the DC/transient load of one instance, and
//! `BSIM4polyDepletion`.
//!
//! Physics stages retain C equation dependencies and host-call ordering.
//! Bias selection, junction currents, channel electrostatics and transport,
//! leakage, intrinsic charges, junction charges, state commit, transient
//! companions, and ordered stamping are distinct operations. C line1000
//! (small-signal initialization) is represented by an absent companion.
//! The port uses the reference NOBYPASS / NEWCONV configuration.
#![expect(
    non_snake_case,
    reason = "preserves upstream C identifiers for equation-by-equation comparison"
)] // C's member and local names.
#![expect(
    non_upper_case_globals,
    reason = "constant names intentionally match the reference device equations"
)] // C's `Charge_q`.
#![expect(
    clippy::collapsible_if,
    reason = "preserves the reference conditional order and side-effect boundaries"
)] // C's if nesting.
use super::fields::{Instance, Model, SizeParam};
use super::host::{LoadCkt, mode, state};
use crate::primitives::SpiceResult;

mod bias;
mod channel;
mod charge;
mod junction;
mod leakage;
mod stamp;

const MAX_EXPL: f64 = 2.688_117_142e+43;
const MIN_EXPL: f64 = 3.720_075_976e-44;
const EXPL_THRESHOLD: f64 = 100.0;
const MAX_EXP: f64 = 5.834_617_425e14;
const MIN_EXP: f64 = 1.713_908_431e-15;
const EXP_THRESHOLD: f64 = 34.0;
const EPS0: f64 = 8.85418e-12;
const EPSSI: f64 = 1.03594e-10;
const Charge_q: f64 = 1.60219e-19;
const DELTA_1: f64 = 0.02;
const DELTA_3: f64 = 0.02;
const DELTA_4: f64 = 0.02;
/// Smoothing coefficient.
const MM: i32 = 3;
/// `const.h`'s `CHARGE`.
const CHARGE: f64 = 1.602_176_620_8e-19;
/// `CONSTvt0 = CONSTboltz * REFTEMP / CHARGE` (`main.c`), REFTEMP = 27 C.
const CONST_VT0: f64 = 1.380_648_52e-23 * (27.0 + 273.15) / CHARGE;

fn exp(x: f64) -> f64 {
    x.exp()
}
fn log(x: f64) -> f64 {
    x.ln()
}
fn sqrt(x: f64) -> f64 {
    x.sqrt()
}
const fn fabs(x: f64) -> f64 {
    x.abs()
}
/// `b4ld.c`'s `DEXP(A, B, C)` macro: returns `(B, C)`.
fn DEXP(A: f64) -> (f64, f64) {
    if A > EXP_THRESHOLD {
        (MAX_EXP * (1.0 + (A) - EXP_THRESHOLD), MAX_EXP)
    } else if A < -EXP_THRESHOLD {
        (MIN_EXP, 0.0)
    } else {
        let B = exp(A);
        (B, B)
    }
}

/// The positive gap between a voltage and the smoothed minimum.
fn smoothed_gap(upper: f64, shift: f64, root: f64, delta: f64) -> f64 {
    if shift >= 0. {
        shift.midpoint(root)
    } else {
        upper * (delta / (-shift).midpoint(root))
    }
}

/// Scale only the root and denominator, not the smaller physical voltage.
/// This retains a representable effective voltage even when the sum or
/// discriminant exceeds the floating-point range (or the normalized small
/// voltage would underflow).
fn smooth_vds_scaled(saturation: f64, drain: f64, delta: f64) -> (f64, f64, f64) {
    let scale = saturation.max(drain).max(delta);
    let scaled_saturation = saturation / scale;
    let scaled_drain = drain / scale;
    let scaled_delta = delta / scale;
    let shift = scaled_saturation - scaled_drain - scaled_delta;
    let root = shift.hypot(2. * scaled_saturation.sqrt() * scaled_delta.sqrt());
    let denominator = scaled_saturation.midpoint(scaled_drain) + scaled_delta.midpoint(root);
    let value = (scaled_saturation.max(scaled_drain) / denominator) * saturation.min(drain);
    let saturation_gap = smoothed_gap(scaled_saturation, shift, root, scaled_delta);
    let drain_gap = smoothed_gap(
        scaled_drain,
        scaled_drain - scaled_saturation - scaled_delta,
        root,
        scaled_delta,
    );
    (value, drain_gap / root, saturation_gap / root)
}

/// The smoothed minimum of positive saturation and drain voltages and its
/// two independent derivatives. The smaller root satisfies
/// v² - (saturation + drain + delta) v + saturation * drain = 0.
/// Rationalize the root and both gaps so a tiny terminal voltage or
/// derivative survives a large difference in scales.
fn smooth_vds(saturation: f64, drain: f64, delta: f64) -> (f64, f64, f64) {
    let shift = saturation - drain - delta;
    let discriminant = (4. * delta).mul_add(saturation, shift * shift);
    let root = if discriminant.is_normal() && discriminant > 0. {
        discriminant.sqrt()
    } else if saturation >= 0. && delta >= 0. {
        shift.hypot(2. * saturation.sqrt() * delta.sqrt())
    } else {
        discriminant.sqrt()
    };
    let saturation_gap = smoothed_gap(saturation, shift, root, delta);
    if saturation >= 0. && drain >= 0. && delta > 0. {
        let denominator = saturation.midpoint(drain) + delta.midpoint(root);
        if (!root.is_normal() || !denominator.is_normal())
            && saturation.is_finite()
            && drain.is_finite()
            && delta.is_finite()
        {
            return smooth_vds_scaled(saturation, drain, delta);
        }
        let (large, small) = if saturation >= drain {
            (saturation, drain)
        } else {
            (drain, saturation)
        };
        let value = if small == 0. {
            0.
        } else {
            (large / denominator) * small
        };
        let drain_gap = smoothed_gap(drain, drain - saturation - delta, root, delta);
        (value, drain_gap / root, saturation_gap / root)
    } else {
        let value = saturation - saturation_gap;
        (value, (drain - value) / root, saturation_gap / root)
    }
}

/// Accurate difference of products, including exact cancellation when both
/// products are equal. Correct the rounding error in the second product
/// rather than fusing one product against the rounded value of the other.
fn difference_of_products(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let product = c * d;
    if product.is_finite() {
        a.mul_add(b, -product) - c.mul_add(d, -product)
    } else if [a, b, c, d].iter().all(|value| value.is_finite()) {
        // The second product overflowed before it could be subtracted.
        // Scale the larger factor of each product by the same exact power
        // of two. At most two such reductions suffice for finite f64 inputs.
        const SCALE: f64 = f64::from_bits((1023 - 512) << 52);
        let scale_product = |left: f64, right: f64| {
            if left.abs() >= right.abs() {
                (left * SCALE, right)
            } else {
                (left, right * SCALE)
            }
        };
        let (a, b) = scale_product(a, b);
        let (c, d) = scale_product(c, d);
        difference_of_products(a, b, c, d) / SCALE
    } else {
        a.mul_add(b, -product)
    }
}

/// `BSIM4polyDepletion`: the poly-gate depletion effective gate voltage and
/// its gate derivative, `(Vgs_eff, dVgs_eff_dVg)`.
pub(super) fn BSIM4polyDepletion(
    phi: f64,
    ngate: f64,
    epsgate: f64,
    coxe: f64,
    Vgs: f64,
) -> (f64, f64) {
    // Poly gate Si depletion effect.
    if (ngate > 1.0e18) && (ngate < 1.0e25) && (Vgs > phi) && (epsgate != 0.0) {
        let T1 = 1.0e6 * CHARGE * epsgate * ngate / (coxe * coxe);
        let T8 = Vgs - phi;
        let T4 = sqrt(1.0 + 2.0 * T8 / T1);
        let T2 = 2.0 * T8 / (T4 + 1.0);
        let T3 = 0.5 * T2 * T2 / T1; // Vpoly
        let T7 = 1.12 - T3 - 0.05;
        let T6 = sqrt(T7.mul_add(T7, 0.224));
        let T5 = 1.12 - T7.midpoint(T6);
        (Vgs - T5, (0.5 - 0.5 / T4).mul_add(-(1.0 + T7 / T6), 1.0))
    } else {
        (Vgs, 1.0)
    }
}

/// `BSIM4load` for one instance: evaluates the device at the voltages C's
/// mode selects, writes `CKTstate0`, integrates charges and adds the
/// matrix and right-hand-side terms, as C does.
///
/// # Errors
/// `NIintegrate` failures (C returns its error).
pub(super) fn load(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    ckt: &mut dyn LoadCkt,
) -> SpiceResult<()> {
    let bias = bias::select(model, here, ckt);
    let junction_bias = junction::JunctionBias {
        source: bias.vbs_jct,
        drain: bias.vbd_jct,
    };
    junction::current(model, here, pParam, ckt, junction_bias);
    let electrostatics = channel::electrostatics(model, here, pParam, ckt, &bias);
    let transport = channel::transport(model, here, pParam, &bias, &electrostatics);
    let cdrain = leakage::evaluate(model, here, pParam, &bias, &electrostatics, &transport);
    let charges = charge::intrinsic(model, here, pParam, ckt, &bias, &electrostatics, &transport)?;
    if bias.ChargeComputationNeeded != 0 {
        junction::charge(model, here, pParam, ckt, junction_bias);
    }
    bias::commit(here, ckt, &bias);
    if let Some(companion) = charge::companion(model, here, pParam, ckt, &bias, &charges)? {
        stamp::apply(model, here, ckt, &bias, cdrain, &companion);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BSIM4polyDepletion, difference_of_products, smooth_vds};

    #[test]
    fn poly_depletion_matches_hand_evaluation() {
        // NGATE = 2e19 cm^-3 gives a 58 mV poly drop at Vgs - phi = 0.6 V;
        // the values are b4ld.c's expressions evaluated in IEEE doubles.
        let (vgs_eff, d) = BSIM4polyDepletion(0.9, 2e19, 1.03594e-10, 0.0115, 1.5);
        assert_eq!(vgs_eff, 1.444_195_874_463_227_7);
        assert_eq!(d, 0.830_890_287_421_742_1);
    }

    #[test]
    fn smoothed_drain_voltage_keeps_tiny_root_and_derivatives() {
        let (value, ds, dd) = smooth_vds(1., 1e-30, 0.02);
        assert!((value / (1e-30 / 1.02) - 1.).abs() < 1e-14);
        assert!((ds / (1e-30 * 0.02 / (1.02 * 1.02)) - 1.).abs() < 1e-13);
        assert!((dd - 1. / 1.02).abs() < 1e-14);
        let (reversed, rds, rdd) = smooth_vds(1e-30, 1., 0.02);
        assert!((reversed / value - 1.).abs() < 1e-14);
        assert!((rds - dd).abs() < 1e-14);
        assert!((rdd / ds - 1.).abs() < 1e-13);
        for saturation in [0.1, 1., 5.] {
            for drain in [0.1, 1., 5.] {
                let (value, ds, dd) = smooth_vds(saturation, drain, 0.02);
                let step = 1e-6;
                let ds_numeric = (smooth_vds(saturation + step, drain, 0.02).0
                    - smooth_vds(saturation - step, drain, 0.02).0)
                    / (2. * step);
                let dd_numeric = (smooth_vds(saturation, drain + step, 0.02).0
                    - smooth_vds(saturation, drain - step, 0.02).0)
                    / (2. * step);
                assert!(value >= 0. && value <= saturation.min(drain));
                assert!((ds - ds_numeric).abs() < 1e-8);
                assert!((dd - dd_numeric).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn smoothed_voltage_retains_finite_values_after_intermediate_overflow() {
        let expected = smooth_vds(1., 1., 1.);
        for scale in [1e155, 1e300, 1e308, f64::MAX] {
            let (value, ds, dd) = smooth_vds(scale, scale, scale);
            assert!((value / scale - expected.0).abs() < 1e-14);
            assert!((ds - expected.1).abs() < 1e-14);
            assert!((dd - expected.2).abs() < 1e-14);
        }
        // Normalizing the smaller voltage itself would underflow to zero.
        let (value, ds, dd) = smooth_vds(1e-200, 1e308, 1e308);
        assert!((value / 5e-201 - 1.).abs() < 1e-14);
        assert!((ds - 0.5).abs() < 1e-14);
        assert_eq!(dd, 0.);
    }

    #[test]
    fn smoothed_voltage_derivatives_remain_scale_invariant_near_underflow() {
        let expected = smooth_vds(1., 1., 1.);
        for scale in [1e-160, 1e-250, 1e-308, 1e-320, f64::from_bits(1)] {
            let (value, ds, dd) = smooth_vds(scale, scale, scale);
            let expected_value = expected.0 * scale;
            assert!(value.to_bits().abs_diff(expected_value.to_bits()) <= 1);
            assert!((ds - expected.1).abs() < 1e-14);
            assert!((dd - expected.2).abs() < 1e-14);
        }
    }

    #[test]
    fn product_difference_preserves_zero_and_small_discriminant() {
        let epsilon = 2_f64.powi(-27);
        assert_eq!(
            difference_of_products(1. + epsilon, 1. - epsilon, 1., 1.),
            -epsilon * epsilon
        );
        let x = 1. + epsilon;
        let y = 1. - epsilon;
        assert_eq!(difference_of_products(x, y, x, y), 0.);
    }

    #[test]
    fn product_difference_handles_overflow_before_cancellation() {
        let max = f64::MAX;
        assert_eq!(difference_of_products(max, max, max, max), 0.);
        assert_eq!(difference_of_products(max, 2., max, 2.), 0.);
        assert_eq!(difference_of_products(max, 2., max, 1.5), max * 0.5);
        assert_eq!(difference_of_products(max, 1.5, max, 2.), -max * 0.5);
    }

    #[test]
    fn poly_depletion_inactive_outside_its_ranges() {
        // Below the flat-band phi, NGATE outside (1e18, 1e25), or EPSGATE = 0.
        for (phi, ngate, eps, vgs) in [
            (0.9, 2e19, 1.03594e-10, 0.5),
            (0.9, 1e18, 1.03594e-10, 1.5),
            (0.9, 1e25, 1.03594e-10, 1.5),
            (0.9, 2e19, 0.0, 1.5),
        ] {
            assert_eq!(BSIM4polyDepletion(phi, ngate, eps, 0.0115, vgs), (vgs, 1.0));
        }
    }
}

#[cfg(test)]
#[path = "load/tests.rs"]
mod trace_tests;
