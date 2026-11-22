//! `b4temp.c` (`BSIM4temp`): the model's temperature pass, the instance's
//! size-dependent parameters and its temperature-dependent instance values,
//! then `b4check.c`'s parameter check.
//!
//! Ported statement by statement from BSIM4 4.8.3 (`b4temp.c`, Copyright
//! (c) 2025 University of California, Educational Community License 2.0);
//! C's member and local names are kept so the port can be read against it.
// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
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
use super::geo::{pa_eff_geo, rdseff_geo};
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

/// `BSIM4DioIjthVjmEval`: the diode limiting voltage `Vjm`.
fn dio_ijth_vjm_eval(Nvtm: f64, Ijth: f64, Isb: f64, XExpBV: f64) -> f64 {
    let ratio = Ijth / Isb;
    if ratio.is_finite() && ratio >= 0. && XExpBV.is_finite() && XExpBV >= 0. {
        // Solve for exp(V/Nvtm) - 1, rather than subtracting one from a
        // rounded root near unity. Rationalize the negative-root branch.
        let b = (ratio - XExpBV) - 1.;
        let discriminant = b.hypot(2. * ratio.sqrt());
        let delta = if b >= 0. {
            b.midpoint(discriminant)
        } else {
            ratio / (-b).midpoint(discriminant)
        };
        return Nvtm * delta.ln_1p();
    }
    let Tc = XExpBV;
    let Tb = 1.0 + ratio - Tc;
    let EVjmovNv = f64::midpoint(Tb, sqrt(4.0f64.mul_add(Tc, Tb * Tb)));
    Nvtm * log(EVjmovNv)
}

/// The BSIM inverse-length/width binning polynomial. Keep the coefficients
/// in their physical order while fusing each product into the accumulation.
fn bin_parameter(base: f64, coefficients: [f64; 3], inverse: [f64; 3]) -> f64 {
    coefficients
        .into_iter()
        .zip(inverse)
        .fold(base, |sum, (coefficient, dimension)| {
            coefficient.mul_add(dimension, sum)
        })
}

/// Averaged well-proximity integral. Direct subtraction of the exponential
/// antiderivative loses the entire effect when width is much smaller than
/// the reference spacing.
fn well_proximity_average(spacing: f64, width: f64, reference: f64, order: f64) -> f64 {
    let x = order * (spacing / reference);
    let u = order * (width / reference);
    let scale = order.recip();
    if x.is_finite() && x >= 0. && u.is_finite() && u >= 0. && width > 0. {
        let mean = if u == 0. { 1. } else { -(-u).exp_m1() / u };
        let tail = if u < 1e-3 {
            // (1-exp(-u))/u - exp(-u), evaluated without cancellation.
            u * u.mul_add(
                u.mul_add(
                    u.mul_add(u.mul_add(1. / 144., -1. / 30.), 1. / 8.),
                    -1. / 3.,
                ),
                0.5,
            )
        } else {
            mean - (-u).exp()
        };
        return scale * (-x).exp() * x.mul_add(mean, tail);
    }
    // Retain the original expression for nonphysical/nonfinite parameters.
    let end = spacing + width;
    let offset = scale * scale * reference;
    scale.mul_add(end, offset).mul_add(
        -(-order * (end / reference)).exp(),
        scale.mul_add(spacing, offset) * (-x).exp(),
    ) / width
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
                && (m.eot.mul_add(T0, -(m.toxp + m.dtox)).abs() > 1.0e-20)
            {
                warnings.push("Warning: eot, toxp and dtox all given and eot * EPSROX / 3.9 != toxp + dtox; dtox ignored.".to_string());
            } else if (m.eotGiven) && (!m.toxpGiven) {
                m.toxp = f64::mul_add(T0, m.eot, -m.dtox);
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
            m.cf = 2.0 * epsrox * EPS0 / PI * log(1.0 + 0.4e-6 / toxe);
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
            T0 = Eg0 / Vtm0 - Eg / m.vtm;
            T1 = log(Temp / Tnom);
            T2 = m.SjctTempExponent.mul_add(T1, T0);
            T3 = exp(T2 / m.SjctEmissionCoeff);
            m.SjctTempSatCurDensity = m.SjctSatCurDensity * T3;
            m.SjctSidewallTempSatCurDensity = m.SjctSidewallSatCurDensity * T3;
            m.SjctGateSidewallTempSatCurDensity = m.SjctGateSidewallSatCurDensity * T3;
            T2 = m.DjctTempExponent.mul_add(T1, T0);
            T3 = exp(T2 / m.DjctEmissionCoeff);
            m.DjctTempSatCurDensity = m.DjctSatCurDensity * T3;
            m.DjctSidewallTempSatCurDensity = m.DjctSidewallSatCurDensity * T3;
            m.DjctGateSidewallTempSatCurDensity = m.DjctGateSidewallSatCurDensity * T3;
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
        m.PhiBS = m.tpb.mul_add(-delTemp, m.SbulkJctPotential);
        if m.PhiBS < 0.01 {
            m.PhiBS = 0.01;
            warnings.push(
                "Temperature effect has caused pbs to be less than 0.01. Pbs is clamped to 0.01."
                    .to_string(),
            );
        }
        m.PhiBD = m.tpb.mul_add(-delTemp, m.DbulkJctPotential);
        if m.PhiBD < 0.01 {
            m.PhiBD = 0.01;
            warnings.push(
                "Temperature effect has caused pbd to be less than 0.01. Pbd is clamped to 0.01."
                    .to_string(),
            );
        }
        m.PhiBSWS = m.tpbsw.mul_add(-delTemp, m.SsidewallJctPotential);
        if m.PhiBSWS < 0.01 {
            m.PhiBSWS = 0.01;
            warnings.push("Temperature effect has caused pbsws to be less than 0.01. Pbsws is clamped to 0.01.".to_string());
        }
        m.PhiBSWD = m.tpbsw.mul_add(-delTemp, m.DsidewallJctPotential);
        if m.PhiBSWD < 0.01 {
            m.PhiBSWD = 0.01;
            warnings.push("Temperature effect has caused pbswd to be less than 0.01. Pbswd is clamped to 0.01.".to_string());
        }
        m.PhiBSWGS = m.tpbswg.mul_add(-delTemp, m.SGatesidewallJctPotential);
        if m.PhiBSWGS < 0.01 {
            m.PhiBSWGS = 0.01;
            warnings.push("Temperature effect has caused pbswgs to be less than 0.01. Pbswgs is clamped to 0.01.".to_string());
        }
        m.PhiBSWGD = m.tpbswg.mul_add(-delTemp, m.DGatesidewallJctPotential);
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
            p.leff = 2.0f64.mul_add(-p.dl, Lnew);
            if p.leff <= 0.0 {
                return Err(bad_parameter(format!(
                    "BSIM4: mosfet {model_name}, model {instance_name}: Effective channel length <= 0"
                )));
            }
            p.weff = 2.0f64.mul_add(-p.dw, Wnew);
            if p.weff <= 0.0 {
                return Err(bad_parameter(format!(
                    "BSIM4: mosfet {model_name}, model {instance_name}: Effective channel width <= 0"
                )));
            }
            p.leffCV = 2.0f64.mul_add(-p.dlc, Lnew);
            if p.leffCV <= 0.0 {
                return Err(bad_parameter(format!(
                    "BSIM4: mosfet {model_name}, model {instance_name}: Effective channel length for C-V <= 0"
                )));
            }
            p.weffCV = 2.0f64.mul_add(-p.dwc, Wnew);
            if p.weffCV <= 0.0 {
                return Err(bad_parameter(format!(
                    "BSIM4: mosfet {model_name}, model {instance_name}: Effective channel width for C-V <= 0"
                )));
            }
            p.weffCJ = 2.0f64.mul_add(-p.dwj, Wnew);
            if p.weffCJ <= 0.0 {
                return Err(bad_parameter(format!(
                    "BSIM4: mosfet {model_name}, model {instance_name}: Effective channel width for S/D junctions <= 0"
                )));
            }
            if m.binUnit == 1 {
                Inv_L = 1.0e-6 / p.leff;
                Inv_W = 1.0e-6 / p.weff;
                Inv_LW = 1.0e-12 / (p.leff * p.weff);
            } else {
                Inv_L = 1.0 / p.leff;
                Inv_W = 1.0 / p.weff;
                Inv_LW = 1.0 / (p.leff * p.weff);
            }
            let bin = |base, length, width, cross| {
                bin_parameter(base, [length, width, cross], [Inv_L, Inv_W, Inv_LW])
            };
            p.cdsc = bin(m.cdsc, m.lcdsc, m.wcdsc, m.pcdsc);
            p.cdscb = bin(m.cdscb, m.lcdscb, m.wcdscb, m.pcdscb);
            p.cdscd = bin(m.cdscd, m.lcdscd, m.wcdscd, m.pcdscd);
            p.cit = bin(m.cit, m.lcit, m.wcit, m.pcit);
            p.nfactor = bin(m.nfactor, m.lnfactor, m.wnfactor, m.pnfactor);
            p.tnfactor = bin(m.tnfactor, m.ltnfactor, m.wtnfactor, m.ptnfactor);
            p.xj = bin(m.xj, m.lxj, m.wxj, m.pxj);
            p.vsat = bin(m.vsat, m.lvsat, m.wvsat, m.pvsat);
            p.at = bin(m.at, m.lat, m.wat, m.pat);
            p.a0 = bin(m.a0, m.la0, m.wa0, m.pa0);
            p.ags = bin(m.ags, m.lags, m.wags, m.pags);
            p.a1 = bin(m.a1, m.la1, m.wa1, m.pa1);
            p.a2 = bin(m.a2, m.la2, m.wa2, m.pa2);
            p.keta = bin(m.keta, m.lketa, m.wketa, m.pketa);
            p.ketac = bin(m.ketac, m.lketac, m.wketac, m.pketac);
            p.nsub = bin(m.nsub, m.lnsub, m.wnsub, m.pnsub);
            p.ndep = bin(m.ndep, m.lndep, m.wndep, m.pndep);
            p.nsd = bin(m.nsd, m.lnsd, m.wnsd, m.pnsd);
            p.phin = bin(m.phin, m.lphin, m.wphin, m.pphin);
            p.ngate = bin(m.ngate, m.lngate, m.wngate, m.pngate);
            p.gamma1 = bin(m.gamma1, m.lgamma1, m.wgamma1, m.pgamma1);
            p.gamma2 = bin(m.gamma2, m.lgamma2, m.wgamma2, m.pgamma2);
            p.vbx = bin(m.vbx, m.lvbx, m.wvbx, m.pvbx);
            p.vbm = bin(m.vbm, m.lvbm, m.wvbm, m.pvbm);
            p.xt = bin(m.xt, m.lxt, m.wxt, m.pxt);
            p.vfb = bin(m.vfb, m.lvfb, m.wvfb, m.pvfb);
            p.k1 = bin(m.k1, m.lk1, m.wk1, m.pk1);
            p.kt1 = bin(m.kt1, m.lkt1, m.wkt1, m.pkt1);
            p.kt1l = bin(m.kt1l, m.lkt1l, m.wkt1l, m.pkt1l);
            p.k2 = bin(m.k2, m.lk2, m.wk2, m.pk2);
            p.kt2 = bin(m.kt2, m.lkt2, m.wkt2, m.pkt2);
            p.k3 = bin(m.k3, m.lk3, m.wk3, m.pk3);
            p.k3b = bin(m.k3b, m.lk3b, m.wk3b, m.pk3b);
            p.w0 = bin(m.w0, m.lw0, m.ww0, m.pw0);
            p.lpe0 = bin(m.lpe0, m.llpe0, m.wlpe0, m.plpe0);
            p.lpeb = bin(m.lpeb, m.llpeb, m.wlpeb, m.plpeb);
            p.dvtp0 = bin(m.dvtp0, m.ldvtp0, m.wdvtp0, m.pdvtp0);
            p.dvtp1 = bin(m.dvtp1, m.ldvtp1, m.wdvtp1, m.pdvtp1);
            p.dvtp2 = bin(m.dvtp2, m.ldvtp2, m.wdvtp2, m.pdvtp2);
            p.dvtp3 = bin(m.dvtp3, m.ldvtp3, m.wdvtp3, m.pdvtp3);
            p.dvtp4 = bin(m.dvtp4, m.ldvtp4, m.wdvtp4, m.pdvtp4);
            p.dvtp5 = bin(m.dvtp5, m.ldvtp5, m.wdvtp5, m.pdvtp5);
            p.dvt0 = bin(m.dvt0, m.ldvt0, m.wdvt0, m.pdvt0);
            p.dvt1 = bin(m.dvt1, m.ldvt1, m.wdvt1, m.pdvt1);
            p.dvt2 = bin(m.dvt2, m.ldvt2, m.wdvt2, m.pdvt2);
            p.dvt0w = bin(m.dvt0w, m.ldvt0w, m.wdvt0w, m.pdvt0w);
            p.dvt1w = bin(m.dvt1w, m.ldvt1w, m.wdvt1w, m.pdvt1w);
            p.dvt2w = bin(m.dvt2w, m.ldvt2w, m.wdvt2w, m.pdvt2w);
            p.drout = bin(m.drout, m.ldrout, m.wdrout, m.pdrout);
            p.dsub = bin(m.dsub, m.ldsub, m.wdsub, m.pdsub);
            p.vth0 = bin(m.vth0, m.lvth0, m.wvth0, m.pvth0);
            p.ua = bin(m.ua, m.lua, m.wua, m.pua);
            p.ua1 = bin(m.ua1, m.lua1, m.wua1, m.pua1);
            p.ub = bin(m.ub, m.lub, m.wub, m.r#pub);
            p.ub1 = bin(m.ub1, m.lub1, m.wub1, m.pub1);
            p.uc = bin(m.uc, m.luc, m.wuc, m.puc);
            p.uc1 = bin(m.uc1, m.luc1, m.wuc1, m.puc1);
            p.ud = bin(m.ud, m.lud, m.wud, m.pud);
            p.ud1 = bin(m.ud1, m.lud1, m.wud1, m.pud1);
            p.up = bin(m.up, m.lup, m.wup, m.pup);
            p.lp = bin(m.lp, m.llp, m.wlp, m.plp);
            p.eu = bin(m.eu, m.leu, m.weu, m.peu);
            p.u0 = bin(m.u0, m.lu0, m.wu0, m.pu0);
            p.ute = bin(m.ute, m.lute, m.wute, m.pute);
            p.ucs = bin(m.ucs, m.lucs, m.wucs, m.pucs);
            p.ucste = bin(m.ucste, m.lucste, m.wucste, m.pucste);
            p.voff = bin(m.voff, m.lvoff, m.wvoff, m.pvoff);
            p.tvoff = bin(m.tvoff, m.ltvoff, m.wtvoff, m.ptvoff);
            p.minv = bin(m.minv, m.lminv, m.wminv, m.pminv);
            p.minvcv = bin(m.minvcv, m.lminvcv, m.wminvcv, m.pminvcv);
            p.fprout = bin(m.fprout, m.lfprout, m.wfprout, m.pfprout);
            p.pdits = bin(m.pdits, m.lpdits, m.wpdits, m.ppdits);
            p.pditsd = bin(m.pditsd, m.lpditsd, m.wpditsd, m.ppditsd);
            p.delta = bin(m.delta, m.ldelta, m.wdelta, m.pdelta);
            p.rdsw = bin(m.rdsw, m.lrdsw, m.wrdsw, m.prdsw);
            p.rdw = bin(m.rdw, m.lrdw, m.wrdw, m.prdw);
            p.rsw = bin(m.rsw, m.lrsw, m.wrsw, m.prsw);
            p.prwg = bin(m.prwg, m.lprwg, m.wprwg, m.pprwg);
            p.prwb = bin(m.prwb, m.lprwb, m.wprwb, m.pprwb);
            p.prt = bin(m.prt, m.lprt, m.wprt, m.pprt);
            p.eta0 = bin(m.eta0, m.leta0, m.weta0, m.peta0);
            p.teta0 = bin(m.teta0, m.lteta0, m.wteta0, m.pteta0);
            p.tvoffcv = bin(m.tvoffcv, m.ltvoffcv, m.wtvoffcv, m.ptvoffcv);
            p.etab = bin(m.etab, m.letab, m.wetab, m.petab);
            p.pclm = bin(m.pclm, m.lpclm, m.wpclm, m.ppclm);
            p.pdibl1 = bin(m.pdibl1, m.lpdibl1, m.wpdibl1, m.ppdibl1);
            p.pdibl2 = bin(m.pdibl2, m.lpdibl2, m.wpdibl2, m.ppdibl2);
            p.pdiblb = bin(m.pdiblb, m.lpdiblb, m.wpdiblb, m.ppdiblb);
            p.pscbe1 = bin(m.pscbe1, m.lpscbe1, m.wpscbe1, m.ppscbe1);
            p.pscbe2 = bin(m.pscbe2, m.lpscbe2, m.wpscbe2, m.ppscbe2);
            p.pvag = bin(m.pvag, m.lpvag, m.wpvag, m.ppvag);
            p.wr = bin(m.wr, m.lwr, m.wwr, m.pwr);
            p.dwg = bin(m.dwg, m.ldwg, m.wdwg, m.pdwg);
            p.dwb = bin(m.dwb, m.ldwb, m.wdwb, m.pdwb);
            p.b0 = bin(m.b0, m.lb0, m.wb0, m.pb0);
            p.b1 = bin(m.b1, m.lb1, m.wb1, m.pb1);
            p.alpha0 = bin(m.alpha0, m.lalpha0, m.walpha0, m.palpha0);
            p.alpha1 = bin(m.alpha1, m.lalpha1, m.walpha1, m.palpha1);
            p.beta0 = bin(m.beta0, m.lbeta0, m.wbeta0, m.pbeta0);
            p.agidl = bin(m.agidl, m.lagidl, m.wagidl, m.pagidl);
            p.bgidl = bin(m.bgidl, m.lbgidl, m.wbgidl, m.pbgidl);
            p.cgidl = bin(m.cgidl, m.lcgidl, m.wcgidl, m.pcgidl);
            p.egidl = bin(m.egidl, m.legidl, m.wegidl, m.pegidl);
            p.rgidl = bin(m.rgidl, m.lrgidl, m.wrgidl, m.prgidl);
            p.kgidl = bin(m.kgidl, m.lkgidl, m.wkgidl, m.pkgidl);
            p.fgidl = bin(m.fgidl, m.lfgidl, m.wfgidl, m.pfgidl);
            p.agisl = bin(m.agisl, m.lagisl, m.wagisl, m.pagisl);
            p.bgisl = bin(m.bgisl, m.lbgisl, m.wbgisl, m.pbgisl);
            p.cgisl = bin(m.cgisl, m.lcgisl, m.wcgisl, m.pcgisl);
            p.egisl = bin(m.egisl, m.legisl, m.wegisl, m.pegisl);
            p.rgisl = bin(m.rgisl, m.lrgisl, m.wrgisl, m.prgisl);
            p.kgisl = bin(m.kgisl, m.lkgisl, m.wkgisl, m.pkgisl);
            p.fgisl = bin(m.fgisl, m.lfgisl, m.wfgisl, m.pfgisl);
            p.aigc = bin(m.aigc, m.laigc, m.waigc, m.paigc);
            p.bigc = bin(m.bigc, m.lbigc, m.wbigc, m.pbigc);
            p.cigc = bin(m.cigc, m.lcigc, m.wcigc, m.pcigc);
            p.aigsd = bin(m.aigsd, m.laigsd, m.waigsd, m.paigsd);
            p.bigsd = bin(m.bigsd, m.lbigsd, m.wbigsd, m.pbigsd);
            p.cigsd = bin(m.cigsd, m.lcigsd, m.wcigsd, m.pcigsd);
            p.aigs = bin(m.aigs, m.laigs, m.waigs, m.paigs);
            p.bigs = bin(m.bigs, m.lbigs, m.wbigs, m.pbigs);
            p.cigs = bin(m.cigs, m.lcigs, m.wcigs, m.pcigs);
            p.aigd = bin(m.aigd, m.laigd, m.waigd, m.paigd);
            p.bigd = bin(m.bigd, m.lbigd, m.wbigd, m.pbigd);
            p.cigd = bin(m.cigd, m.lcigd, m.wcigd, m.pcigd);
            p.aigbacc = bin(m.aigbacc, m.laigbacc, m.waigbacc, m.paigbacc);
            p.bigbacc = bin(m.bigbacc, m.lbigbacc, m.wbigbacc, m.pbigbacc);
            p.cigbacc = bin(m.cigbacc, m.lcigbacc, m.wcigbacc, m.pcigbacc);
            p.aigbinv = bin(m.aigbinv, m.laigbinv, m.waigbinv, m.paigbinv);
            p.bigbinv = bin(m.bigbinv, m.lbigbinv, m.wbigbinv, m.pbigbinv);
            p.cigbinv = bin(m.cigbinv, m.lcigbinv, m.wcigbinv, m.pcigbinv);
            p.nigc = bin(m.nigc, m.lnigc, m.wnigc, m.pnigc);
            p.nigbacc = bin(m.nigbacc, m.lnigbacc, m.wnigbacc, m.pnigbacc);
            p.nigbinv = bin(m.nigbinv, m.lnigbinv, m.wnigbinv, m.pnigbinv);
            p.ntox = bin(m.ntox, m.lntox, m.wntox, m.pntox);
            p.eigbinv = bin(m.eigbinv, m.leigbinv, m.weigbinv, m.peigbinv);
            p.pigcd = bin(m.pigcd, m.lpigcd, m.wpigcd, m.ppigcd);
            p.poxedge = bin(m.poxedge, m.lpoxedge, m.wpoxedge, m.ppoxedge);
            p.xrcrg1 = bin(m.xrcrg1, m.lxrcrg1, m.wxrcrg1, m.pxrcrg1);
            p.xrcrg2 = bin(m.xrcrg2, m.lxrcrg2, m.wxrcrg2, m.pxrcrg2);
            p.lambda = bin(m.lambda, m.llambda, m.wlambda, m.plambda);
            p.vtl = bin(m.vtl, m.lvtl, m.wvtl, m.pvtl);
            p.xn = bin(m.xn, m.lxn, m.wxn, m.pxn);
            p.vfbsdoff = bin(m.vfbsdoff, m.lvfbsdoff, m.wvfbsdoff, m.pvfbsdoff);
            p.tvfbsdoff = bin(m.tvfbsdoff, m.ltvfbsdoff, m.wtvfbsdoff, m.ptvfbsdoff);
            p.cgsl = bin(m.cgsl, m.lcgsl, m.wcgsl, m.pcgsl);
            p.cgdl = bin(m.cgdl, m.lcgdl, m.wcgdl, m.pcgdl);
            p.ckappas = bin(m.ckappas, m.lckappas, m.wckappas, m.pckappas);
            p.ckappad = bin(m.ckappad, m.lckappad, m.wckappad, m.pckappad);
            p.cf = bin(m.cf, m.lcf, m.wcf, m.pcf);
            p.clc = bin(m.clc, m.lclc, m.wclc, m.pclc);
            p.cle = bin(m.cle, m.lcle, m.wcle, m.pcle);
            p.vfbcv = bin(m.vfbcv, m.lvfbcv, m.wvfbcv, m.pvfbcv);
            p.acde = bin(m.acde, m.lacde, m.wacde, m.pacde);
            p.moin = bin(m.moin, m.lmoin, m.wmoin, m.pmoin);
            p.noff = bin(m.noff, m.lnoff, m.wnoff, m.pnoff);
            p.voffcv = bin(m.voffcv, m.lvoffcv, m.wvoffcv, m.pvoffcv);
            p.kvth0we = bin(m.kvth0we, m.lkvth0we, m.wkvth0we, m.pkvth0we);
            p.k2we = bin(m.k2we, m.lk2we, m.wk2we, m.pk2we);
            p.ku0we = bin(m.ku0we, m.lku0we, m.wku0we, m.pku0we);
            p.abulkCVfactor = 1.0 + pow((p.clc / p.leffCV), p.cle);
            T0 = (TRatio - 1.0);
            PowWeffWr = pow(p.weffCJ * 1.0e6, p.wr) * h.nf;
            T4 = 0.0;
            T3 = T4;
            T2 = T4;
            T1 = T4;
            p.ucs = p.ucs * pow(TRatio, p.ucste);
            if m.tempMod == 0 {
                p.ua = f64::mul_add(p.ua1, T0, p.ua);
                p.ub = f64::mul_add(p.ub1, T0, p.ub);
                p.uc = f64::mul_add(p.uc1, T0, p.uc);
                p.ud = f64::mul_add(p.ud1, T0, p.ud);
                p.vsattemp = f64::mul_add(p.at, -T0, p.vsat);
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
                    p.ua = p.ua * f64::mul_add(p.ua1, delTemp, 1.0);
                    p.ub = p.ub * f64::mul_add(p.ub1, delTemp, 1.0);
                    p.uc = p.uc * f64::mul_add(p.uc1, delTemp, 1.0);
                    p.ud = p.ud * f64::mul_add(p.ud1, delTemp, 1.0);
                }
                p.vsattemp = p.vsat * f64::mul_add(p.at, -delTemp, 1.0);
                T10 = f64::mul_add(p.prt, delTemp, 1.0);
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
            T5 = f64::mul_add(p.up, -exp(-p.leff / p.lp), 1.0);
            p.u0temp = p.u0 * T5 * pow(TRatio, p.ute);
            if p.eu < 0.0 {
                p.eu = 0.0;
                warnings.push("Warning: eu has been negative; reset to 0.0.".to_string());
            }
            if p.ucs < 0.0 {
                p.ucs = 0.0;
                warnings.push("Warning: ucs has been negative; reset to 0.0.".to_string());
            }
            p.vfbsdoff = p.vfbsdoff * f64::mul_add(p.tvfbsdoff, delTemp, 1.0);
            p.voff = p.voff * f64::mul_add(p.tvoff, delTemp, 1.0);
            p.nfactor = p.nfactor + p.tnfactor * delTemp / Tnom;
            p.voffcv = p.voffcv * f64::mul_add(p.tvoffcv, delTemp, 1.0);
            p.eta0 = p.eta0 + p.teta0 * delTemp / Tnom;
            if (m.vtlGiven) && (m.vtl > 0.0) {
                if m.lc < 0.0 {
                    p.lc = 0.0;
                } else {
                    p.lc = m.lc;
                }
                T0 = p.leff / f64::mul_add(p.xn, p.leff, p.lc);
                p.tfactor = (1.0 - T0) / (1.0 + T0);
            }
            p.cgdo = (m.cgdo + p.cf) * p.weffCV;
            p.cgso = (m.cgso + p.cf) * p.weffCV;
            p.cgbo = m.cgbo * p.leffCV * h.nf;
            if !m.ndepGiven && m.gamma1Given {
                T0 = p.gamma1 * m.coxe;
                p.ndep = 3.01248e22 * T0 * T0;
            }
            p.phi = f64::mul_add(Vtm0, log(p.ndep / ni), p.phin) + 0.4;
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
                T2 = f64::mul_add(f64::from(m.r#type), -T0, m.easub + T1);
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
                    p.vbx = f64::mul_add((7.7348e-4 * p.ndep * p.xt), -p.xt, p.phi);
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
                T1 = sqrt(p.phi - p.vbx) - p.sqrtPhi;
                T2 = sqrt(p.phi * (p.phi - p.vbm)) - p.phi;
                p.k2 = T0 * T1 / 2.0f64.mul_add(T2, p.vbm);
                p.k1 = f64::mul_add((2.0 * p.k2), -sqrt(p.phi - p.vbm), p.gamma2);
            }
            if !m.vfbGiven {
                if m.vth0Given {
                    p.vfb = f64::mul_add(
                        p.k1,
                        -p.sqrtPhi,
                        f64::mul_add(f64::from(m.r#type), p.vth0, -p.phi),
                    );
                } else {
                    if (m.mtrlMod != 0) && (m.phigGiven) && (m.nsubGiven) {
                        T0 = Vtm0 * log(p.nsub / ni);
                        T1 = 0.5 * Eg0;
                        if T0 > T1 {
                            T0 = T1;
                        }
                        T2 = f64::mul_add(f64::from(m.r#type), T0, m.easub + T1);
                        p.vfb = m.phig - T2;
                    } else {
                        p.vfb = -1.0;
                    }
                }
            }
            if !m.vth0Given {
                p.vth0 = f64::from(m.r#type) * f64::mul_add(p.k1, p.sqrtPhi, p.vfb + p.phi);
            }
            p.k1ox = p.k1 * toxe / m.toxm;
            tmp = sqrt(epssub / (epsrox * EPS0) * toxe * p.Xdep0);
            T0 = p.dsub * p.leff / tmp;
            if T0 < EXP_THRESHOLD {
                T1 = exp(T0);
                T2 = T1 - 1.0;
                T3 = T2 * T2;
                T4 = f64::mul_add((2.0 * T1), MIN_EXP, T3);
                p.theta0vb0 = T1 / T4;
            } else {
                p.theta0vb0 = 1.0 / (MAX_EXP - 2.0);
            }
            T0 = p.drout * p.leff / tmp;
            if T0 < EXP_THRESHOLD {
                T1 = exp(T0);
                T2 = T1 - 1.0;
                T3 = T2 * T2;
                T4 = f64::mul_add((2.0 * T1), MIN_EXP, T3);
                T5 = T1 / T4;
            } else {
                T5 = 1.0 / (MAX_EXP - 2.0);
            }
            p.thetaRout = f64::mul_add(p.pdibl1, T5, p.pdibl2);
            tmp = sqrt(p.Xdep0);
            tmp1 = p.vbi - p.phi;
            tmp2 = m.factor1 * tmp;
            T0 = p.dvt1w * p.weff * p.leff / tmp2;
            if T0 < EXP_THRESHOLD {
                T1 = exp(T0);
                T2 = T1 - 1.0;
                T3 = T2 * T2;
                T4 = f64::mul_add((2.0 * T1), MIN_EXP, T3);
                T8 = T1 / T4;
            } else {
                T8 = 1.0 / (MAX_EXP - 2.0);
            }
            T0 = p.dvt0w * T8;
            T8 = T0 * tmp1;
            T0 = p.dvt1 * p.leff / tmp2;
            if T0 < EXP_THRESHOLD {
                T1 = exp(T0);
                T2 = T1 - 1.0;
                T3 = T2 * T2;
                T4 = f64::mul_add((2.0 * T1), MIN_EXP, T3);
                T9 = T1 / T4;
            } else {
                T9 = 1.0 / (MAX_EXP - 2.0);
            }
            T9 = p.dvt0 * T9 * tmp1;
            T4 = toxe * p.phi / (p.weff + p.w0);
            T0 = sqrt(1.0 + p.lpe0 / p.leff);
            if (m.tempMod == 1) || (m.tempMod == 0) {
                T3 = (p.kt1 + p.kt1l / p.leff) * (TRatio - 1.0);
            }
            if (m.tempMod == 2) || (m.tempMod == 3) {
                T3 = -p.kt1 * (TRatio - 1.0);
            }
            T5 = f64::mul_add((p.k1ox * (T0 - 1.0)), p.sqrtPhi, T3);
            p.vfbzbfactor = f64::mul_add(
                p.k1,
                -p.sqrtPhi,
                f64::mul_add(p.k3, T4, -T8 - T9) + T5 - p.phi,
            );
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
            p.kvth0 = sqrt(f64::mul_add(p.kvth0, p.kvth0, DELTA));
            T0 = (TRatio - 1.0);
            p.ku0temp = p.ku0.mul_add(m.tku0.mul_add(T0, 1.), DELTA);
            Inv_saref = 1.0 / 0.5f64.mul_add(Ldrn, m.saref);
            Inv_sbref = 1.0 / 0.5f64.mul_add(Ldrn, m.sbref);
            p.inv_od_ref = Inv_saref + Inv_sbref;
            p.rho_ref = m.ku0 / p.ku0temp * p.inv_od_ref;
            if m.mobMod == 3 {
                lt1 = m.factor1 * p.sqrtXdep0;
                T0 = p.dvt1 * p.leff / lt1;
                if T0 < EXP_THRESHOLD {
                    T1 = exp(T0);
                    T2 = T1 - 1.0;
                    T3 = T2 * T2;
                    T4 = f64::mul_add((2.0 * T1), MIN_EXP, T3);
                    Theta0 = T1 / T4;
                } else {
                    Theta0 = 1.0 / (MAX_EXP - 2.0);
                }
                tmp1 = epssub / p.Xdep0;
                tmp2 = p.nfactor * tmp1;
                tmp3 = (f64::mul_add(p.cdsc, Theta0, tmp2) + p.cit) / m.coxe;
                if tmp3 >= -0.5 {
                    n0 = 1.0 + tmp3;
                } else {
                    T0 = 1.0 / 8.0f64.mul_add(tmp3, 3.0);
                    n0 = 3.0f64.mul_add(tmp3, 1.0) * T0;
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
                T4 = f64::mul_add(T3, n0, p.mstar);
                p.VgsteffVth = T0 * log(2.0) / T4;
            }
            T0 = -p.dvtp3 * log(p.leff);
            T1 = dexp(T0);
            p.dvtp2factor = f64::mul_add(p.dvtp2, T1, p.dvtp5);
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
                T0 = 1.0
                    / h.nf
                    / f64::mul_add(f64::from(i), h.sd + Ldrn, 0.5f64.mul_add(Ldrn, h.sa));
                T1 = 1.0
                    / h.nf
                    / f64::mul_add(f64::from(i), h.sd + Ldrn, 0.5f64.mul_add(Ldrn, h.sb));
                Inv_sa += T0;
                Inv_sb += T1;
                i += 1;
            }
            Inv_ODeff = Inv_sa + Inv_sb;
            rho = m.ku0 / p.ku0temp * Inv_ODeff;
            T0 = (1.0 + rho) / (1.0 + p.rho_ref);
            h.u0temp = p.u0temp * T0;
            T1 = f64::mul_add(kvsat, rho, 1.0) / f64::mul_add(kvsat, p.rho_ref, 1.0);
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
                    T1 = h.sc + Wdrn;
                    h.sca = m.scref * m.scref / (h.sc * T1);
                    h.scb = well_proximity_average(h.sc, Wdrn, m.scref, 10.);
                    h.scc = well_proximity_average(h.sc, Wdrn, m.scref, 20.);
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
            sceff = m.wec.mul_add(h.scc, m.web.mul_add(h.scb, h.sca));
            h.vth0 = f64::mul_add(p.kvth0we, sceff, h.vth0);
            h.k2 = f64::mul_add(p.k2we, sceff, h.k2);
            T3 = f64::mul_add(p.ku0we, sceff, 1.0);
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
        h.vfb = f64::mul_add(f64::from(m.r#type), h.delvto, p.vfb);
        h.u0temp = p.u0temp * h.mulu0;
        T3 = f64::mul_add(f64::from(m.r#type), h.vth0, -h.vfb) - p.phi;
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
            h.vbsc = 0.9 * f64::mul_add(T0, -T0, p.phi);
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
        h.vfbzb = f64::mul_add(f64::from(m.r#type), h.vth0, p.vfbzbfactor);
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
            if bodymode == 5.0 {
                rbsbx = m.rbsbx0
                    * exp(m
                        .rbsdbxnf
                        .mul_add(lnnf, m.rbsdbxw.mul_add(lnw, m.rbsdbxl * lnl)));
                rbsby = m.rbsby0
                    * exp(m
                        .rbsdbynf
                        .mul_add(lnnf, m.rbsdbyw.mul_add(lnw, m.rbsdbyl * lnl)));
                h.rbsb = rbsbx * rbsby / (rbsbx + rbsby);
                rbdbx = m.rbdbx0
                    * exp(m
                        .rbsdbxnf
                        .mul_add(lnnf, m.rbsdbxw.mul_add(lnw, m.rbsdbxl * lnl)));
                rbdby = m.rbdby0
                    * exp(m
                        .rbsdbynf
                        .mul_add(lnnf, m.rbsdbyw.mul_add(lnw, m.rbsdbyl * lnl)));
                h.rbdb = rbdbx * rbdby / (rbdbx + rbdby);
            }
            if (bodymode == 3.0) || (bodymode == 5.0) {
                h.rbps = m.rbps0 * exp(m.rbpsnf.mul_add(lnnf, m.rbpsw.mul_add(lnw, m.rbpsl * lnl)));
                h.rbpd = m.rbpd0 * exp(m.rbpdnf.mul_add(lnnf, m.rbpdw.mul_add(lnw, m.rbpdl * lnl)));
            }
            rbpbx = m.rbpbx0
                * exp(m
                    .rbpbxnf
                    .mul_add(lnnf, m.rbpbxw.mul_add(lnw, m.rbpbxl * lnl)));
            rbpby = m.rbpby0
                * exp(m
                    .rbpbynf
                    .mul_add(lnnf, m.rbpbyw.mul_add(lnw, m.rbpbyl * lnl)));
            h.rbpb = rbpbx * rbpby / (rbpbx + rbpby);
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
                    h.Pseff = f64::mul_add(p.weffCJ, -h.nf, h.sourcePerimeter);
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
                    h.Pdeff = f64::mul_add(p.weffCJ, -h.nf, h.drainPerimeter);
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
        Nvtms = m.vtm * m.SjctEmissionCoeff;
        if (h.Aseff <= 0.0) && (h.Pseff <= 0.0) {
            SourceSatCurrent = 0.0;
        } else {
            SourceSatCurrent = (p.weffCJ * h.nf).mul_add(
                m.SjctGateSidewallTempSatCurDensity,
                h.Pseff.mul_add(
                    m.SjctSidewallTempSatCurDensity,
                    h.Aseff * m.SjctTempSatCurDensity,
                ),
            );
        }
        if SourceSatCurrent > 0.0 {
            match m.dioMod {
                0 => {
                    if (m.bvs / Nvtms) > EXP_THRESHOLD {
                        h.XExpBVS = m.xjbvs * MIN_EXP;
                    } else {
                        h.XExpBVS = m.xjbvs * exp(-m.bvs / Nvtms);
                    }
                }
                1 => {
                    h.vjsmFwd = dio_ijth_vjm_eval(Nvtms, m.ijthsfwd, SourceSatCurrent, 0.0);
                    h.IVjsmFwd = SourceSatCurrent * exp(h.vjsmFwd / Nvtms);
                }
                2 => {
                    if (m.bvs / Nvtms) > EXP_THRESHOLD {
                        h.XExpBVS = m.xjbvs * MIN_EXP;
                        tmp = MIN_EXP;
                    } else {
                        h.XExpBVS = exp(-m.bvs / Nvtms);
                        tmp = h.XExpBVS;
                        h.XExpBVS *= m.xjbvs;
                    }
                    h.vjsmFwd = dio_ijth_vjm_eval(Nvtms, m.ijthsfwd, SourceSatCurrent, h.XExpBVS);
                    T0 = exp(h.vjsmFwd / Nvtms);
                    h.IVjsmFwd = SourceSatCurrent * (T0 - h.XExpBVS / T0 + h.XExpBVS - 1.0);
                    h.SslpFwd = SourceSatCurrent * (T0 + h.XExpBVS / T0) / Nvtms;
                    T2 = m.ijthsrev / SourceSatCurrent;
                    if T2 < 1.0 {
                        T2 = 10.0;
                        warnings.push(
                            "Warning: ijthsrev too small and set to 10 times IsbSat.".to_string(),
                        );
                    }
                    h.vjsmRev = f64::mul_add(Nvtms, -log((T2 - 1.0) / m.xjbvs), -m.bvs);
                    T1 = m.xjbvs * exp(-(m.bvs + h.vjsmRev) / Nvtms);
                    h.IVjsmRev = SourceSatCurrent * (1.0 + T1);
                    h.SslpRev = -SourceSatCurrent * T1 / Nvtms;
                }
                _ => {
                    warnings.push(format!("Specified dioMod = {} not matched", m.dioMod));
                }
            }
        }
        Nvtmd = m.vtm * m.DjctEmissionCoeff;
        if (h.Adeff <= 0.0) && (h.Pdeff <= 0.0) {
            DrainSatCurrent = 0.0;
        } else {
            DrainSatCurrent = (p.weffCJ * h.nf).mul_add(
                m.DjctGateSidewallTempSatCurDensity,
                h.Pdeff.mul_add(
                    m.DjctSidewallTempSatCurDensity,
                    h.Adeff * m.DjctTempSatCurDensity,
                ),
            );
        }
        if DrainSatCurrent > 0.0 {
            match m.dioMod {
                0 => {
                    if (m.bvd / Nvtmd) > EXP_THRESHOLD {
                        h.XExpBVD = m.xjbvd * MIN_EXP;
                    } else {
                        h.XExpBVD = m.xjbvd * exp(-m.bvd / Nvtmd);
                    }
                }
                1 => {
                    h.vjdmFwd = dio_ijth_vjm_eval(Nvtmd, m.ijthdfwd, DrainSatCurrent, 0.0);
                    h.IVjdmFwd = DrainSatCurrent * exp(h.vjdmFwd / Nvtmd);
                }
                2 => {
                    if (m.bvd / Nvtmd) > EXP_THRESHOLD {
                        h.XExpBVD = m.xjbvd * MIN_EXP;
                        tmp = MIN_EXP;
                    } else {
                        h.XExpBVD = exp(-m.bvd / Nvtmd);
                        tmp = h.XExpBVD;
                        h.XExpBVD *= m.xjbvd;
                    }
                    h.vjdmFwd = dio_ijth_vjm_eval(Nvtmd, m.ijthdfwd, DrainSatCurrent, h.XExpBVD);
                    T0 = exp(h.vjdmFwd / Nvtmd);
                    h.IVjdmFwd = DrainSatCurrent * (T0 - h.XExpBVD / T0 + h.XExpBVD - 1.0);
                    h.DslpFwd = DrainSatCurrent * (T0 + h.XExpBVD / T0) / Nvtmd;
                    T2 = m.ijthdrev / DrainSatCurrent;
                    if T2 < 1.0 {
                        T2 = 10.0;
                        warnings.push(
                            "Warning: ijthdrev too small and set to 10 times IdbSat.".to_string(),
                        );
                    }
                    h.vjdmRev = f64::mul_add(Nvtmd, -log((T2 - 1.0) / m.xjbvd), -m.bvd);
                    T1 = m.xjbvd * exp(-(m.bvd + h.vjdmRev) / Nvtmd);
                    h.IVjdmRev = DrainSatCurrent * (1.0 + T1);
                    h.DslpRev = -DrainSatCurrent * T1 / Nvtmd;
                }
                _ => {
                    warnings.push(format!("Specified dioMod = {} not matched", m.dioMod));
                }
            }
        }
        T0 = (TRatio - 1.0);
        m.njtsstemp = m.njts * m.tnjts.mul_add(T0, 1.0);
        m.njtsswstemp = m.njtssw * m.tnjtssw.mul_add(T0, 1.0);
        m.njtsswgstemp = m.njtsswg * m.tnjtsswg.mul_add(T0, 1.0);
        m.njtsdtemp = m.njtsd * m.tnjtsd.mul_add(T0, 1.0);
        m.njtsswdtemp = m.njtsswd * m.tnjtsswd.mul_add(T0, 1.0);
        m.njtsswgdtemp = m.njtsswgd * m.tnjtsswgd.mul_add(T0, 1.0);
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
            phieot = f64::mul_add(Vtm0eot, log(p.ndep / ni), p.phin) + 0.4;
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
            if (p.ngate > 1.0e18) && (p.ngate < 1.0e25) && (vddeot > tmp2) && (T0 != 0.0) {
                T1 = 1.0e6 * CHARGE * T0 * p.ngate / (m.coxe * m.coxe);
                T8 = vddeot - tmp2;
                T4 = sqrt(1.0 + 2.0 * T8 / T1);
                T2 = 2.0 * T8 / (T4 + 1.0);
                T3 = 0.5 * T2 * T2 / T1;
                T7 = 1.12 - T3 - 0.05;
                T6 = sqrt(f64::mul_add(T7, T7, 0.224));
                T5 = 1.12 - T7.midpoint(T6);
                Vgs_eff = vddeot - T5;
            } else {
                Vgs_eff = vddeot;
            }
            V0 = vbieot - phieot;
            lt1 = m.factor1 * p.sqrtXdep0;
            ltw = lt1;
            T0 = p.dvt1 * m.leffeot / lt1;
            if T0 < EXP_THRESHOLD {
                T1 = exp(T0);
                T2 = T1 - 1.0;
                T3 = T2 * T2;
                T4 = f64::mul_add((2.0 * T1), MIN_EXP, T3);
                Theta0 = T1 / T4;
            } else {
                Theta0 = 1.0 / (MAX_EXP - 2.0);
            }
            Delt_vth = p.dvt0 * Theta0 * V0;
            T0 = p.dvt1w * m.weffeot * m.leffeot / ltw;
            if T0 < EXP_THRESHOLD {
                T1 = exp(T0);
                T2 = T1 - 1.0;
                T3 = T2 * T2;
                T4 = f64::mul_add((2.0 * T1), MIN_EXP, T3);
                T5 = T1 / T4;
            } else {
                T5 = 1.0 / (MAX_EXP - 2.0);
            }
            T2 = p.dvt0w * T5 * V0;
            TempRatioeot = m.tempeot / m.tnom - 1.0;
            T0 = sqrt(1.0 + p.lpe0 / m.leffeot);
            T1 = (p.kt1 + p.kt1l / m.leffeot)
                .mul_add(TempRatioeot, p.k1ox * (T0 - 1.0) * sqrt(phieot));
            let width_with_offset = m.weffeot + p.w0;
            Vth_NarrowW = toxe * phieot / width_with_offset;
            Lpe_Vb = sqrt(1.0 + p.lpeb / m.leffeot);
            Vth = f64::mul_add(
                p.k3,
                Vth_NarrowW,
                ((p.k1ox - p.k1) * sqrt(phieot)).mul_add(Lpe_Vb, f64::from(m.r#type) * h.vth0)
                    - Delt_vth
                    - T2,
            ) + T1;
            tmp1 = epssub / p.Xdep0;
            tmp2 = p.nfactor * tmp1;
            tmp3 = (f64::mul_add(p.cdsc, Theta0, tmp2) + p.cit) / m.coxe;
            if tmp3 >= -0.5 {
                n = 1.0 + tmp3;
            } else {
                T0 = 1.0 / 8.0f64.mul_add(tmp3, 3.0);
                n = 3.0f64.mul_add(tmp3, 1.0) * T0;
            }
            if p.dvtp0 > 0.0 {
                T3 = f64::mul_add(p.dvtp0, 2.0, m.leffeot);
                if m.tempMod < 2 {
                    T4 = Vtmeot * log(m.leffeot / T3);
                } else {
                    T4 = Vtm0eot * log(m.leffeot / T3);
                }
                Vth = f64::mul_add(n, -T4, Vth);
            }
            Vgsteff = Vgs_eff - Vth;
            T3 = f64::mul_add(f64::from(m.r#type), h.vth0, -h.vfb) - phieot;
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
                toxpf = (epsrox / m.epsrsub).mul_add(-Tcen, toxe);
                niter += 1.0;
                if !((niter <= 4.0) && ((toxpf - toxpi).abs() > 1e-12)) {
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
    fn small_diode_current_and_narrow_well_integrals_do_not_round_to_zero() {
        let voltage = dio_ijth_vjm_eval(0.025, 1e-20, 1., 3.);
        assert!((voltage / (0.025 * 2.5e-21) - 1.).abs() < 1e-14);
        assert_eq!(dio_ijth_vjm_eval(0.025, 0., 1., 3.), 0.);
        for order in [10., 20.] {
            let average = well_proximity_average(0.2, 1e-20, 1., order);
            assert!((average / (0.2 * (-0.2 * order).exp()) - 1.).abs() < 1e-14);
            let ordinary = well_proximity_average(0.2, 0.03, 1., order);
            let coeff = order.recip();
            let f = |x: f64| coeff * (x + coeff) * (-order * x).exp();
            assert!((ordinary / ((f(0.2) - f(0.23)) / 0.03) - 1.).abs() < 1e-13);
        }
    }

    #[test]
    fn binning_retains_product_residual() {
        let epsilon = 2_f64.powi(-27);
        assert_eq!(
            bin_parameter(-1., [1. + epsilon, 0., 0.], [1. - epsilon, 1., 1.]),
            -epsilon * epsilon
        );
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
