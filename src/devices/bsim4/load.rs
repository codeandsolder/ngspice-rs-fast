//! `b4ld.c` (`BSIM4load`): the DC/transient load of one instance, and
//! `BSIM4polyDepletion`.
//!
//! Physics stages retain the original C expression and host-call ordering.
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

/// Smooth the transition from linear drain voltage to saturation using
/// the smaller root of y² - (sat + drain + delta)y + sat*drain = 0.
/// Keep the complementary gaps as independent values: reconstructing them
/// by subtraction would lose precisely the low-field terms needed by the
/// channel-length and capacitance derivatives.
struct DrainSmoothing {
    effective: f64,
    drain_excess: f64,
    saturation_partial: f64,
    drain_partial: f64,
}

fn positive_half_sum(x: f64, root: f64, product: f64) -> f64 {
    if x >= 0. {
        0.5 * (root + x)
    } else {
        2. * product / (root - x)
    }
}

fn smooth_drain(saturation: f64, drain: f64, delta: f64) -> DrainSmoothing {
    let x = saturation - drain - delta;
    let root = x.mul_add(x, 4. * delta * saturation).sqrt();
    let effective = (2. * saturation / (saturation + drain + delta + root)) * drain;
    let drain_excess = positive_half_sum(drain - saturation - delta, root, delta * drain);
    let saturation_excess = positive_half_sum(x, root, delta * saturation);
    DrainSmoothing {
        effective,
        drain_excess,
        saturation_partial: drain_excess / root,
        drain_partial: saturation_excess / root,
    }
}

/// Smaller physical saturation root of a*y²/2 - b*y + c = 0.
struct SaturationRoot {
    value: f64,
    discriminant_root: f64,
}

impl SaturationRoot {
    fn new(a: f64, b: f64, c: f64) -> Self {
        let discriminant_root = (-2. * a).mul_add(c, b * b).sqrt();
        Self {
            value: 2. * c / (b + discriminant_root),
            discriminant_root,
        }
    }

    fn derivative(&self, da: f64, db: f64, dc: f64) -> f64 {
        let y = self.value;
        (0.5 * y * y).mul_add(da, (-y).mul_add(db, dc)) / self.discriminant_root
    }
}

/// Normalized depletion charge and its differential capacitance.
/// The negative-potential continuation is linear. Rationalization makes
/// the positive branch continuous down to the representable zero limit.
fn depletion_charge(k1: f64, increment: f64) -> (f64, f64) {
    if k1 == 0. {
        return (0., 0.);
    }
    if increment < 0. {
        return (increment, 1.);
    }
    let base = 0.5 * k1;
    let root = base.mul_add(base, increment).sqrt();
    let difference = if base > 0. {
        increment / (root + base)
    } else {
        root - base
    };
    (k1 * difference, base / root)
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
        let T4 = (2.0_f64.mul_add(T8 / T1, 1.0)).sqrt();
        let T2 = 2.0 * T8 / (T4 + 1.0);
        let T3 = 0.5 * T2 * T2 / T1; // Vpoly
        let T7 = 1.07 - T3;
        let T6 = T7.mul_add(T7, 0.224).sqrt();
        // Rationalized band-bending drop. The original difference
        // 1.12 - (T7+T6)/2 loses the entire drop near the onset of depletion.
        let T5 = 2.24 * T3 / (2.24 - T7 + T6);
        let slope = (-(0.5 - 0.5 / T4)).mul_add(1.0 + T7 / T6, 1.0);
        (Vgs - T5, slope)
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
    use super::BSIM4polyDepletion;

    #[test]
    fn poly_depletion_matches_hand_evaluation() {
        // Independent 80-digit evaluation of the physical equations:
        // a 55.80412553677218 mV drop, not a particular C rounding order.
        let (vgs_eff, d) = BSIM4polyDepletion(0.9, 2e19, 1.03594e-10, 0.0115, 1.5);
        assert!((vgs_eff - 1.444_195_874_463_227_8).abs() <= 2. * f64::EPSILON);
        assert!((d - 0.830_890_287_421_742_1).abs() <= 2. * f64::EPSILON);
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
