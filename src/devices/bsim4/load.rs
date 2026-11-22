//! `b4ld.c` (`BSIM4load`): the DC/transient load of one instance, and
//! `BSIM4polyDepletion`.
//!
//! Physics stages retain the original C expression and host-call ordering.
//! Bias selection, junction currents, channel electrostatics and transport,
//! leakage, intrinsic charges, junction charges, state commit, transient
//! companions, and ordered stamping are distinct operations. C line1000
//! (small-signal initialization) is represented by an absent companion.
//! The port uses the reference NOBYPASS / NEWCONV configuration.
// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::manual_midpoint,
    reason = "C source midpoint arithmetic retains its original evaluation and rounding order"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]
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
        let T6 = sqrt(T7 * T7 + 0.224);
        let T5 = 1.12 - 0.5 * (T7 + T6);
        (Vgs - T5, 1.0 - (0.5 - 0.5 / T4) * (1.0 + T7 / T6))
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
        // NGATE = 2e19 cm^-3 gives a 58 mV poly drop at Vgs - phi = 0.6 V;
        // the values are b4ld.c's expressions evaluated in IEEE doubles.
        let (vgs_eff, d) = BSIM4polyDepletion(0.9, 2e19, 1.03594e-10, 0.0115, 1.5);
        assert_eq!(vgs_eff, 1.444_195_874_463_227_7);
        assert_eq!(d, 0.830_890_287_421_742_1);
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
