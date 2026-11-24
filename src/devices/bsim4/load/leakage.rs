//! Leakage and final current scaling in original C statement order.
#![expect(
    non_snake_case,
    reason = "C-translated equation symbols identify their upstream terms"
)]
#![expect(
    clippy::similar_names,
    reason = "C equation names preserve the reference distinctions"
)]
#![expect(
    unused_mut,
    reason = "reference scratch variables may be assigned on different branches"
)]
#![expect(
    clippy::assign_op_pattern,
    reason = "retain C expression grouping and operand order"
)]
#![expect(
    clippy::float_cmp,
    reason = "C branch predicates compare floating point exactly"
)]
#![expect(clippy::manual_range_contains, reason = "retain C comparison ordering")]
#![expect(
    clippy::too_many_lines,
    reason = "preserve cohesive C physics equation sequence"
)]
#![expect(
    clippy::cognitive_complexity,
    reason = "retain version-specific reference physics branches"
)]
use super::{
    DELTA_3, DELTA_4, EXP_THRESHOLD, EXPL_THRESHOLD, Instance, MAX_EXP, MAX_EXPL, MIN_EXP,
    MIN_EXPL, Model, SizeParam, exp,
};
use super::{
    bias::LoadBias,
    channel::{Electrostatics, Transport},
};

/// Source/drain channel-tunneling partition and derivative with respect
/// to its exponential argument. These are the C regularized fractions,
/// not renormalized weights: their sum need not be exactly one away from
/// equilibrium. The terminal stamps conserve the two physical currents.
struct GatePartition {
    source: f64,
    drain: f64,
    source_slope: f64,
    drain_slope: f64,
}

impl GatePartition {
    fn new(argument: f64) -> Self {
        const REGULARIZATION: f64 = 1e-4;
        if argument.is_finite() && argument.abs() > EXP_THRESHOLD {
            let exponential = if argument > 0. { MAX_EXP } else { MIN_EXP };
            let difference = exponential - 1.;
            let inverse = argument.recip();
            let normalized = (2. * REGULARIZATION * inverse).mul_add(inverse, 1.);
            let source_scaled = (difference + REGULARIZATION).mul_add(inverse, -1.) / normalized;
            let drain_scaled =
                (REGULARIZATION - difference).mul_add(inverse, exponential) / normalized;
            return Self {
                source: source_scaled * inverse,
                drain: drain_scaled * inverse,
                source_slope: (-2_f64).mul_add(source_scaled, -1.) * inverse * inverse / normalized,
                drain_slope: (-2_f64).mul_add(drain_scaled, exponential) * inverse * inverse
                    / normalized,
            };
        }
        let denominator = argument.mul_add(argument, 2. * REGULARIZATION);
        if argument.abs() <= 0.5 {
            // exp(x)-1-x-x²/2 = x³*(1/3! + x/4! + ...).
            // Terms through 15! bound the remainder below binary64
            // rounding error for |x| <= 1/2.
            let mut term = argument * argument * argument / 6.;
            let mut third = term;
            for order in 4..=15 {
                term *= argument / f64::from(order);
                let next = third + term;
                if next == third {
                    break;
                }
                third = next;
            }
            let second = 0.5 * argument * argument + third;
            let source_residual = third / denominator;
            let drain_residual = argument.mul_add(second, -third) / denominator;
            Self {
                source: 0.5 + source_residual,
                drain: 0.5 + drain_residual,
                source_slope: (-2. * argument).mul_add(source_residual, second) / denominator,
                drain_slope: (-2. * argument).mul_add(drain_residual, argument * argument.exp_m1())
                    / denominator,
            }
        } else {
            let (exponential, derivative) = if argument > EXP_THRESHOLD {
                (MAX_EXP, 0.)
            } else if argument < -EXP_THRESHOLD {
                (MIN_EXP, 0.)
            } else {
                let exponential = argument.exp();
                (exponential, exponential)
            };
            let difference = exponential - 1.;
            let source = (difference - argument + REGULARIZATION) / denominator;
            let drain = argument.mul_add(exponential, REGULARIZATION - difference) / denominator;
            let source_slope = (-2. * argument).mul_add(source, derivative - 1.) / denominator;
            let drain_slope = (-2. * argument)
                .mul_add(drain, (argument - 1.).mul_add(derivative, exponential))
                / denominator;
            Self {
                source,
                drain,
                source_slope,
                drain_slope,
            }
        }
    }
}

/// Gate-induced leakage, tunneling, finger scaling and thermal-noise auxiliaries.
#[expect(
    clippy::suboptimal_flops,
    reason = "retain ordered C version-specific leakage branches; conditioned smoothing, softplus and partition laws use their dedicated kernels"
)]
pub(super) fn evaluate(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    bias: &LoadBias,
    electrostatics: &Electrostatics,
    transport: &Transport,
) -> f64 {
    let &LoadBias {
        vbd,
        vbs,
        vds,
        vgd,
        vgs,
        ..
    } = bias;
    let &Electrostatics {
        Vbs,
        Vbseff,
        Vds,
        Vgs,
        Vgs_eff,
        Vgsteff,
        Vtm,
        Vtm0,
        dVbseff_dVb,
        dVgs_eff_dVg,
        dVgsteff_dVb,
        dVgsteff_dVd,
        dVgsteff_dVg,
        dVth_dVb,
        dVth_dVd,
        epsrox,
        toxe,
        ..
    } = electrostatics;
    let &Transport {
        Abulk0_Q,
        Coxeff,
        Rds,
        mut Vdseff,
        beta,
        unscaled_drain_current: mut cdrain,
        dVdseff_dVb,
        dVdseff_dVd,
        dVdseff_dVg,
        gche,
        ..
    } = transport;
    let mut vgs_eff: f64;
    let mut vgd_eff: f64;
    let mut dvgs_eff_dvg: f64;
    let mut dvgd_eff_dvg: f64;
    let mut dVfbeff_dVg: f64;
    let mut dVfbeff_dVb: f64;
    let mut V3: f64;
    let mut Igidl: f64;
    let mut Ggidld: f64;
    let mut Ggidlg: f64;
    let mut Ggidlb: f64;
    let mut Voxacc: f64 = 0.0;
    let mut dVoxacc_dVg: f64 = 0.0;
    let mut dVoxacc_dVb: f64 = 0.0;
    let mut Voxdepinv: f64 = 0.0;
    let mut dVoxdepinv_dVg: f64 = 0.0;
    let mut dVoxdepinv_dVd: f64 = 0.0;
    let mut dVoxdepinv_dVb: f64 = 0.0;
    let mut VxNVt: f64 = 0.0;
    let mut ExpVxNVt: f64;
    let mut Vaux: f64 = 0.0;
    let mut dVaux_dVg: f64 = 0.0;
    let mut dVaux_dVd: f64 = 0.0;
    let mut dVaux_dVb: f64 = 0.0;
    let mut Igc: f64;
    let mut dIgc_dVg: f64;
    let mut dIgc_dVd: f64;
    let mut dIgc_dVb: f64;
    let mut Igcs: f64;
    let mut dIgcs_dVg: f64;
    let mut dIgcs_dVd: f64;
    let mut dIgcs_dVb: f64;
    let mut Igcd: f64;
    let mut dIgcd_dVg: f64;
    let mut dIgcd_dVd: f64;
    let mut dIgcd_dVb: f64;
    let mut Igs: f64;
    let mut dIgs_dVg: f64;
    let mut dIgs_dVs: f64;
    let mut Igd: f64;
    let mut dIgd_dVg: f64;
    let mut dIgd_dVd: f64;
    let mut Igbacc: f64;
    let mut dIgbacc_dVg: f64;
    let mut dIgbacc_dVb: f64;
    let mut Igbinv: f64;
    let mut dIgbinv_dVg: f64;
    let mut dIgbinv_dVd: f64;
    let mut dIgbinv_dVb: f64;
    let mut Pigcd: f64;
    let mut dPigcd_dVg: f64;
    let mut dPigcd_dVd: f64;
    let mut dPigcd_dVb: f64;
    let mut Vfb: f64 = 0.0;
    let mut Vdsat: f64;
    let mut T0: f64;
    let mut T1: f64;
    let mut dT1_dVg: f64;
    let mut dT1_dVd: f64;
    let mut T2: f64;
    let mut dT2_dVg: f64;
    let mut dT2_dVd: f64;
    let mut dT2_dVb: f64;
    let mut T3: f64;
    let mut T4: f64;
    let mut T5: f64;
    let mut T6: f64;
    let mut dT6_dVg: f64;
    let mut dT6_dVd: f64;
    let mut dT6_dVb: f64;
    let mut T7: f64;
    let mut dT7_dVg: f64;
    let mut dT7_dVd: f64;
    let mut dT7_dVb: f64;
    let mut T8: f64;
    let mut T11: f64;
    let mut T12: f64;
    let mut T13: f64;
    let mut T14: f64;
    let mut tmp: f64;
    let mut Abulk: f64;
    let mut Igisl: f64;
    let mut Ggislg: f64;
    let mut Ggislb: f64;
    let mut Ggisls: f64;
    if model.mtrlMod == 0 {
        T0 = 3.0 * toxe;
    } else {
        T0 = model.epsrsub * toxe / epsrox;
    }
    vgs_eff = here.vgs_eff;
    dvgs_eff_dvg = here.dvgs_eff_dvg;
    vgd_eff = here.vgd_eff;
    dvgd_eff_dvg = here.dvgd_eff_dvg;
    if model.gidlMod == 0 {
        if model.mtrlMod == 0 {
            T1 = (vds - vgs_eff - pParam.egidl) / T0;
        } else {
            T1 = (vds - vgs_eff - pParam.egidl + pParam.vfbsd) / T0;
        }
        if (pParam.agidl <= 0.0)
            || (pParam.bgidl <= 0.0)
            || (T1 <= 0.0)
            || (pParam.cgidl <= 0.0)
            || (vbd > 0.0)
        {
            Ggidlb = 0.0;
            Ggidlg = Ggidlb;
            Ggidld = Ggidlg;
            Igidl = Ggidld;
        } else {
            dT1_dVd = 1.0 / T0;
            dT1_dVg = -dvgs_eff_dvg * dT1_dVd;
            T2 = pParam.bgidl / T1;
            if T2 < 100.0 {
                Igidl = pParam.agidl * pParam.weffCJ * T1 * exp(-T2);
                T3 = Igidl * (1.0 + T2) / T1;
                Ggidld = T3 * dT1_dVd;
                Ggidlg = T3 * dT1_dVg;
            } else {
                Igidl = pParam.agidl * pParam.weffCJ * 3.720_075_976e-44;
                Ggidld = Igidl * dT1_dVd;
                Ggidlg = Igidl * dT1_dVg;
                Igidl *= T1;
            }
            T4 = vbd * vbd;
            T5 = -vbd * T4;
            T6 = pParam.cgidl + T5;
            T7 = T5 / T6;
            T8 = 3.0 * pParam.cgidl * T4 / T6 / T6;
            Ggidld = Ggidld * T7 + Igidl * T8;
            Ggidlg = Ggidlg * T7;
            Ggidlb = -Igidl * T8;
            Igidl *= T7;
        }
        here.Igidl = Igidl;
        here.ggidld = Ggidld;
        here.ggidlg = Ggidlg;
        here.ggidlb = Ggidlb;
        if model.mtrlMod == 0 {
            T1 = (-vds - vgd_eff - pParam.egisl) / T0;
        } else {
            T1 = (-vds - vgd_eff - pParam.egisl + pParam.vfbsd) / T0;
        }
        if (pParam.agisl <= 0.0)
            || (pParam.bgisl <= 0.0)
            || (T1 <= 0.0)
            || (pParam.cgisl <= 0.0)
            || (vbs > 0.0)
        {
            Ggislb = 0.0;
            Ggislg = Ggislb;
            Ggisls = Ggislg;
            Igisl = Ggisls;
        } else {
            dT1_dVd = 1.0 / T0;
            dT1_dVg = -dvgd_eff_dvg * dT1_dVd;
            T2 = pParam.bgisl / T1;
            if T2 < 100.0 {
                Igisl = pParam.agisl * pParam.weffCJ * T1 * exp(-T2);
                T3 = Igisl * (1.0 + T2) / T1;
                Ggisls = T3 * dT1_dVd;
                Ggislg = T3 * dT1_dVg;
            } else {
                Igisl = pParam.agisl * pParam.weffCJ * 3.720_075_976e-44;
                Ggisls = Igisl * dT1_dVd;
                Ggislg = Igisl * dT1_dVg;
                Igisl *= T1;
            }
            T4 = vbs * vbs;
            T5 = -vbs * T4;
            T6 = pParam.cgisl + T5;
            T7 = T5 / T6;
            T8 = 3.0 * pParam.cgisl * T4 / T6 / T6;
            Ggisls = Ggisls * T7 + Igisl * T8;
            Ggislg = Ggislg * T7;
            Ggislb = -Igisl * T8;
            Igisl *= T7;
        }
        here.Igisl = Igisl;
        here.ggisls = Ggisls;
        here.ggislg = Ggislg;
        here.ggislb = Ggislb;
    } else {
        if model.mtrlMod == 0 {
            T1 = (-vds - pParam.rgisl * vgd_eff - pParam.egisl) / T0;
        } else {
            T1 = (-vds - pParam.rgisl * vgd_eff - pParam.egisl + pParam.vfbsd) / T0;
        }
        if (pParam.agisl <= 0.0) || (pParam.bgisl <= 0.0) || (T1 <= 0.0) || (pParam.cgisl < 0.0) {
            Ggislb = 0.0;
            Ggislg = Ggislb;
            Ggisls = Ggislg;
            Igisl = Ggisls;
        } else {
            dT1_dVd = 1.0 / T0;
            dT1_dVg = -pParam.rgisl * dT1_dVd * dvgd_eff_dvg;
            T2 = pParam.bgisl / T1;
            if T2 < EXPL_THRESHOLD {
                Igisl = pParam.weffCJ * pParam.agisl * T1 * exp(-T2);
                T3 = Igisl / T1 * (T2 + 1.0);
            } else {
                T3 = pParam.weffCJ * pParam.agisl * MIN_EXPL;
                Igisl = T3 * T1;
            }
            Ggisls = T3 * dT1_dVd;
            Ggislg = T3 * dT1_dVg;
            T4 = vbs - pParam.fgisl;
            if T4 > model.gidlclamp {
                T4 = model.gidlclamp;
            }
            if T4 == 0.0 {
                T5 = EXPL_THRESHOLD;
            } else {
                T5 = pParam.kgisl / T4;
            }
            if T5 < EXPL_THRESHOLD {
                T6 = exp(T5);
                Ggislb = -Igisl * T6 * T5 / T4;
            } else {
                T6 = MAX_EXPL;
                Ggislb = 0.0;
            }
            Ggisls *= T6;
            Ggislg *= T6;
            Igisl *= T6;
        }
        here.Igisl = Igisl;
        here.ggisls = Ggisls;
        here.ggislg = Ggislg;
        here.ggislb = Ggislb;
        if model.mtrlMod == 0 {
            T1 = (vds - pParam.rgidl * vgs_eff - pParam.egidl) / T0;
        } else {
            T1 = (vds - pParam.rgidl * vgs_eff - pParam.egidl + pParam.vfbsd) / T0;
        }
        if (pParam.agidl <= 0.0) || (pParam.bgidl <= 0.0) || (T1 <= 0.0) || (pParam.cgidl < 0.0) {
            Ggidlb = 0.0;
            Ggidlg = Ggidlb;
            Ggidld = Ggidlg;
            Igidl = Ggidld;
        } else {
            dT1_dVd = 1.0 / T0;
            dT1_dVg = -pParam.rgidl * dT1_dVd * dvgs_eff_dvg;
            T2 = pParam.bgidl / T1;
            if T2 < EXPL_THRESHOLD {
                Igidl = pParam.weffCJ * pParam.agidl * T1 * exp(-T2);
                T3 = Igidl / T1 * (T2 + 1.0);
            } else {
                T3 = pParam.weffCJ * pParam.agidl * MIN_EXPL;
                Igidl = T3 * T1;
            }
            Ggidld = T3 * dT1_dVd;
            Ggidlg = T3 * dT1_dVg;
            T4 = vbd - pParam.fgidl;
            if T4 > model.gidlclamp {
                T4 = model.gidlclamp;
            }
            if T4 == 0.0 {
                T5 = EXPL_THRESHOLD;
            } else {
                T5 = pParam.kgidl / T4;
            }
            if T5 < EXPL_THRESHOLD {
                T6 = exp(T5);
                Ggidlb = -Igidl * T6 * T5 / T4;
            } else {
                T6 = MAX_EXPL;
                Ggidlb = 0.0;
            }
            Ggidld *= T6;
            Ggidlg *= T6;
            Igidl *= T6;
        }
        here.Igidl = Igidl;
        here.ggidld = Ggidld;
        here.ggidlg = Ggidlg;
        here.ggidlb = Ggidlb;
    }
    if (model.igcMod != 0) || (model.igbMod != 0) {
        Vfb = here.vfbzb;
        V3 = Vfb - Vgs_eff + Vbseff - DELTA_3;
        (Voxacc, T1) = super::smooth_positive(V3, DELTA_3 * Vfb.abs());
        dVfbeff_dVg = T1 * dVgs_eff_dVg;
        dVfbeff_dVb = -T1;
        dVoxacc_dVg = -dVfbeff_dVg;
        dVoxacc_dVb = -dVfbeff_dVb;
        if Voxacc < 0.0 {
            dVoxacc_dVb = 0.0;
            dVoxacc_dVg = dVoxacc_dVb;
            Voxacc = dVoxacc_dVg;
        }
        T3 = (Vgs_eff - Vfb - Vbseff - Vgsteff) + Voxacc;
        if pParam.k1ox == 0.0 {
            dVoxdepinv_dVb = 0.0;
            dVoxdepinv_dVd = dVoxdepinv_dVb;
            dVoxdepinv_dVg = dVoxdepinv_dVd;
            Voxdepinv = dVoxdepinv_dVg;
        } else if T3 < 0.0 {
            Voxdepinv = -T3;
            dVoxdepinv_dVg = -dVgs_eff_dVg + dVfbeff_dVg + dVgsteff_dVg;
            dVoxdepinv_dVd = dVgsteff_dVd;
            dVoxdepinv_dVb = dVfbeff_dVb + 1.0 + dVgsteff_dVb;
        } else {
            (Voxdepinv, T2) = super::depletion_charge(pParam.k1ox, T3);
            dVoxdepinv_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg);
            dVoxdepinv_dVd = -T2 * dVgsteff_dVd;
            dVoxdepinv_dVb = -T2 * (dVfbeff_dVb + 1.0 + dVgsteff_dVb);
        }
        Voxdepinv += Vgsteff;
        dVoxdepinv_dVg += dVgsteff_dVg;
        dVoxdepinv_dVd += dVgsteff_dVd;
        dVoxdepinv_dVb += dVgsteff_dVb;
    }
    if model.tempMod < 2 {
        tmp = Vtm;
    } else {
        tmp = Vtm0;
    }
    if model.igcMod != 0 {
        T0 = tmp * pParam.nigc;
        if model.igcMod == 1 {
            VxNVt = (Vgs_eff - f64::from(model.r#type) * here.vth0) / T0;
            if VxNVt > EXP_THRESHOLD {
                Vaux = Vgs_eff - f64::from(model.r#type) * here.vth0;
                dVaux_dVg = dVgs_eff_dVg;
                dVaux_dVd = 0.0;
                dVaux_dVb = 0.0;
            }
        } else if model.igcMod == 2 {
            VxNVt = (Vgs_eff - here.von) / T0;
            if VxNVt > EXP_THRESHOLD {
                Vaux = Vgs_eff - here.von;
                dVaux_dVg = dVgs_eff_dVg;
                dVaux_dVd = -dVth_dVd;
                dVaux_dVb = -dVth_dVb;
            }
        }
        if VxNVt < -EXP_THRESHOLD {
            Vaux = T0 * MIN_EXP.ln_1p();
            dVaux_dVb = 0.0;
            dVaux_dVd = dVaux_dVb;
            dVaux_dVg = dVaux_dVd;
        } else if (VxNVt >= -EXP_THRESHOLD) && (VxNVt <= EXP_THRESHOLD) {
            ExpVxNVt = exp(VxNVt);
            Vaux = T0 * ExpVxNVt.ln_1p();
            dVaux_dVg = ExpVxNVt / (1.0 + ExpVxNVt);
            if model.igcMod == 1 {
                dVaux_dVd = 0.0;
                dVaux_dVb = 0.0;
            } else if model.igcMod == 2 {
                dVaux_dVd = -dVaux_dVg * dVth_dVd;
                dVaux_dVb = -dVaux_dVg * dVth_dVb;
            }
            dVaux_dVg *= dVgs_eff_dVg;
        }
        T2 = Vgs * Vaux;
        dT2_dVg = Vaux + Vgs * dVaux_dVg;
        dT2_dVd = Vgs * dVaux_dVd;
        dT2_dVb = Vgs * dVaux_dVb;
        T11 = pParam.Aechvb;
        T12 = pParam.Bechvb;
        T3 = pParam.aigc * pParam.cigc - pParam.bigc;
        T4 = pParam.bigc * pParam.cigc;
        T5 = T12 * (pParam.aigc + T3 * Voxdepinv - T4 * Voxdepinv * Voxdepinv);
        if T5 > EXP_THRESHOLD {
            T6 = MAX_EXP;
            dT6_dVb = 0.0;
            dT6_dVd = dT6_dVb;
            dT6_dVg = dT6_dVd;
        } else if T5 < -EXP_THRESHOLD {
            T6 = MIN_EXP;
            dT6_dVb = 0.0;
            dT6_dVd = dT6_dVb;
            dT6_dVg = dT6_dVd;
        } else {
            T6 = exp(T5);
            dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxdepinv);
            dT6_dVd = dT6_dVg * dVoxdepinv_dVd;
            dT6_dVb = dT6_dVg * dVoxdepinv_dVb;
            dT6_dVg *= dVoxdepinv_dVg;
        }
        Igc = T11 * T2 * T6;
        dIgc_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
        dIgc_dVd = T11 * (T2 * dT6_dVd + T6 * dT2_dVd);
        dIgc_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb);
        if model.pigcdGiven {
            Pigcd = pParam.pigcd;
            dPigcd_dVb = 0.0;
            dPigcd_dVd = dPigcd_dVb;
            dPigcd_dVg = dPigcd_dVd;
        } else {
            T11 = -pParam.Bechvb;
            T12 = Vgsteff + 1.0e-20;
            T13 = T11 / T12 / T12;
            T14 = -T13 / T12;
            Pigcd = T13 * (1.0 - 0.5 * Vdseff / T12);
            dPigcd_dVg = T14 * (2.0 + 0.5 * (dVdseff_dVg - 3.0 * Vdseff / T12));
            dPigcd_dVd = 0.5 * T14 * dVdseff_dVd;
            dPigcd_dVb = 0.5 * T14 * dVdseff_dVb;
        }
        T7 = -Pigcd * Vdseff;
        dT7_dVg = -Vdseff * dPigcd_dVg - Pigcd * dVdseff_dVg;
        dT7_dVd = -Vdseff * dPigcd_dVd - Pigcd * dVdseff_dVd + dT7_dVg * dVgsteff_dVd;
        dT7_dVb = -Vdseff * dPigcd_dVb - Pigcd * dVdseff_dVb + dT7_dVg * dVgsteff_dVb;
        dT7_dVg *= dVgsteff_dVg;
        let partition = GatePartition::new(T7);
        let source_slope = partition.source_slope;
        let drain_slope = partition.drain_slope;
        Igcs = Igc * partition.source;
        dIgcs_dVg = dIgc_dVg.mul_add(partition.source, Igc * source_slope * dT7_dVg);
        dIgcs_dVd = dIgc_dVd.mul_add(partition.source, Igc * source_slope * dT7_dVd);
        dIgcs_dVb = dIgc_dVb.mul_add(partition.source, Igc * source_slope * dT7_dVb);
        Igcd = Igc * partition.drain;
        dIgcd_dVg = dIgc_dVg.mul_add(partition.drain, Igc * drain_slope * dT7_dVg);
        dIgcd_dVd = dIgc_dVd.mul_add(partition.drain, Igc * drain_slope * dT7_dVd);
        dIgcd_dVb = dIgc_dVb.mul_add(partition.drain, Igc * drain_slope * dT7_dVb);
        here.Igcs = Igcs;
        here.gIgcsg = dIgcs_dVg;
        here.gIgcsd = dIgcs_dVd;
        here.gIgcsb = dIgcs_dVb * dVbseff_dVb;
        here.Igcd = Igcd;
        here.gIgcdg = dIgcd_dVg;
        here.gIgcdd = dIgcd_dVd;
        here.gIgcdb = dIgcd_dVb * dVbseff_dVb;
        T0 = vgs - (pParam.vfbsd + pParam.vfbsdoff);
        vgs_eff = T0.hypot(0.01);
        dvgs_eff_dvg = T0 / vgs_eff;
        T2 = vgs * vgs_eff;
        dT2_dVg = vgs * dvgs_eff_dvg + vgs_eff;
        T11 = pParam.AechvbEdgeS;
        T12 = pParam.BechvbEdge;
        T3 = pParam.aigs * pParam.cigs - pParam.bigs;
        T4 = pParam.bigs * pParam.cigs;
        T5 = T12 * (pParam.aigs + T3 * vgs_eff - T4 * vgs_eff * vgs_eff);
        if T5 > EXP_THRESHOLD {
            T6 = MAX_EXP;
            dT6_dVg = 0.0;
        } else if T5 < -EXP_THRESHOLD {
            T6 = MIN_EXP;
            dT6_dVg = 0.0;
        } else {
            T6 = exp(T5);
            dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * vgs_eff) * dvgs_eff_dvg;
        }
        Igs = T11 * T2 * T6;
        dIgs_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
        dIgs_dVs = -dIgs_dVg;
        T0 = vgd - (pParam.vfbsd + pParam.vfbsdoff);
        vgd_eff = T0.hypot(0.01);
        dvgd_eff_dvg = T0 / vgd_eff;
        T2 = vgd * vgd_eff;
        dT2_dVg = vgd * dvgd_eff_dvg + vgd_eff;
        T11 = pParam.AechvbEdgeD;
        T3 = pParam.aigd * pParam.cigd - pParam.bigd;
        T4 = pParam.bigd * pParam.cigd;
        T5 = T12 * (pParam.aigd + T3 * vgd_eff - T4 * vgd_eff * vgd_eff);
        if T5 > EXP_THRESHOLD {
            T6 = MAX_EXP;
            dT6_dVg = 0.0;
        } else if T5 < -EXP_THRESHOLD {
            T6 = MIN_EXP;
            dT6_dVg = 0.0;
        } else {
            T6 = exp(T5);
            dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * vgd_eff) * dvgd_eff_dvg;
        }
        Igd = T11 * T2 * T6;
        dIgd_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
        dIgd_dVd = -dIgd_dVg;
        here.Igs = Igs;
        here.gIgsg = dIgs_dVg;
        here.gIgss = dIgs_dVs;
        here.Igd = Igd;
        here.gIgdg = dIgd_dVg;
        here.gIgdd = dIgd_dVd;
    } else {
        here.gIgcsb = 0.0;
        here.gIgcsd = here.gIgcsb;
        here.gIgcsg = here.gIgcsd;
        here.Igcs = here.gIgcsg;
        here.gIgcdb = 0.0;
        here.gIgcdd = here.gIgcdb;
        here.gIgcdg = here.gIgcdd;
        here.Igcd = here.gIgcdg;
        here.gIgss = 0.0;
        here.gIgsg = here.gIgss;
        here.Igs = here.gIgsg;
        here.gIgdd = 0.0;
        here.gIgdg = here.gIgdd;
        here.Igd = here.gIgdg;
    }
    if model.igbMod != 0 {
        T0 = tmp * pParam.nigbacc;
        T1 = -Vgs_eff + Vbseff + Vfb;
        VxNVt = T1 / T0;
        if VxNVt > EXP_THRESHOLD {
            Vaux = T1;
            dVaux_dVg = -dVgs_eff_dVg;
            dVaux_dVb = 1.0;
        } else if VxNVt < -EXP_THRESHOLD {
            Vaux = T0 * MIN_EXP.ln_1p();
            dVaux_dVb = 0.0;
            dVaux_dVg = dVaux_dVb;
        } else {
            ExpVxNVt = exp(VxNVt);
            Vaux = T0 * ExpVxNVt.ln_1p();
            dVaux_dVb = ExpVxNVt / (1.0 + ExpVxNVt);
            dVaux_dVg = -dVaux_dVb * dVgs_eff_dVg;
        }
        T2 = (Vgs - Vbs) * Vaux;
        dT2_dVg = Vaux + (Vgs - Vbs) * dVaux_dVg;
        dT2_dVb = -Vaux + (Vgs - Vbs) * dVaux_dVb;
        T11 = 4.97232e-7 * pParam.weff * pParam.leff * pParam.ToxRatio;
        T12 = -7.45669e11 * toxe;
        T3 = pParam.aigbacc * pParam.cigbacc - pParam.bigbacc;
        T4 = pParam.bigbacc * pParam.cigbacc;
        T5 = T12 * (pParam.aigbacc + T3 * Voxacc - T4 * Voxacc * Voxacc);
        if T5 > EXP_THRESHOLD {
            T6 = MAX_EXP;
            dT6_dVb = 0.0;
            dT6_dVg = dT6_dVb;
        } else if T5 < -EXP_THRESHOLD {
            T6 = MIN_EXP;
            dT6_dVb = 0.0;
            dT6_dVg = dT6_dVb;
        } else {
            T6 = exp(T5);
            dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxacc);
            dT6_dVb = dT6_dVg * dVoxacc_dVb;
            dT6_dVg *= dVoxacc_dVg;
        }
        Igbacc = T11 * T2 * T6;
        dIgbacc_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
        dIgbacc_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb);
        T0 = tmp * pParam.nigbinv;
        T1 = Voxdepinv - pParam.eigbinv;
        VxNVt = T1 / T0;
        if VxNVt > EXP_THRESHOLD {
            Vaux = T1;
            dVaux_dVg = dVoxdepinv_dVg;
            dVaux_dVd = dVoxdepinv_dVd;
            dVaux_dVb = dVoxdepinv_dVb;
        } else if VxNVt < -EXP_THRESHOLD {
            Vaux = T0 * MIN_EXP.ln_1p();
            dVaux_dVb = 0.0;
            dVaux_dVd = dVaux_dVb;
            dVaux_dVg = dVaux_dVd;
        } else {
            ExpVxNVt = exp(VxNVt);
            Vaux = T0 * ExpVxNVt.ln_1p();
            dVaux_dVg = ExpVxNVt / (1.0 + ExpVxNVt);
            dVaux_dVd = dVaux_dVg * dVoxdepinv_dVd;
            dVaux_dVb = dVaux_dVg * dVoxdepinv_dVb;
            dVaux_dVg *= dVoxdepinv_dVg;
        }
        T2 = (Vgs - Vbs) * Vaux;
        dT2_dVg = Vaux + (Vgs - Vbs) * dVaux_dVg;
        dT2_dVd = (Vgs - Vbs) * dVaux_dVd;
        dT2_dVb = -Vaux + (Vgs - Vbs) * dVaux_dVb;
        T11 *= 0.75610;
        T12 *= 1.31724;
        T3 = pParam.aigbinv * pParam.cigbinv - pParam.bigbinv;
        T4 = pParam.bigbinv * pParam.cigbinv;
        T5 = T12 * (pParam.aigbinv + T3 * Voxdepinv - T4 * Voxdepinv * Voxdepinv);
        if T5 > EXP_THRESHOLD {
            T6 = MAX_EXP;
            dT6_dVb = 0.0;
            dT6_dVd = dT6_dVb;
            dT6_dVg = dT6_dVd;
        } else if T5 < -EXP_THRESHOLD {
            T6 = MIN_EXP;
            dT6_dVb = 0.0;
            dT6_dVd = dT6_dVb;
            dT6_dVg = dT6_dVd;
        } else {
            T6 = exp(T5);
            dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxdepinv);
            dT6_dVd = dT6_dVg * dVoxdepinv_dVd;
            dT6_dVb = dT6_dVg * dVoxdepinv_dVb;
            dT6_dVg *= dVoxdepinv_dVg;
        }
        Igbinv = T11 * T2 * T6;
        dIgbinv_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
        dIgbinv_dVd = T11 * (T2 * dT6_dVd + T6 * dT2_dVd);
        dIgbinv_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb);
        here.Igb = Igbinv + Igbacc;
        here.gIgbg = dIgbinv_dVg + dIgbacc_dVg;
        here.gIgbd = dIgbinv_dVd;
        here.gIgbb = (dIgbinv_dVb + dIgbacc_dVb) * dVbseff_dVb;
    } else {
        here.gIgbb = 0.0;
        here.gIgbs = here.gIgbb;
        here.gIgbd = here.gIgbs;
        here.gIgbg = here.gIgbd;
        here.Igb = here.gIgbg;
    }
    if here.nf != 1.0 {
        cdrain *= here.nf;
        here.gds *= here.nf;
        here.gm *= here.nf;
        here.gmbs *= here.nf;
        here.IdovVds *= here.nf;
        here.gbbs *= here.nf;
        here.gbgs *= here.nf;
        here.gbds *= here.nf;
        here.csub *= here.nf;
        here.Igidl *= here.nf;
        here.ggidld *= here.nf;
        here.ggidlg *= here.nf;
        here.ggidlb *= here.nf;
        here.Igisl *= here.nf;
        here.ggisls *= here.nf;
        here.ggislg *= here.nf;
        here.ggislb *= here.nf;
        here.Igcs *= here.nf;
        here.gIgcsg *= here.nf;
        here.gIgcsd *= here.nf;
        here.gIgcsb *= here.nf;
        here.Igcd *= here.nf;
        here.gIgcdg *= here.nf;
        here.gIgcdd *= here.nf;
        here.gIgcdb *= here.nf;
        here.Igs *= here.nf;
        here.gIgsg *= here.nf;
        here.gIgss *= here.nf;
        here.Igd *= here.nf;
        here.gIgdg *= here.nf;
        here.gIgdd *= here.nf;
        here.Igb *= here.nf;
        here.gIgbg *= here.nf;
        here.gIgbd *= here.nf;
        here.gIgbb *= here.nf;
    }
    here.ggidls = -(here.ggidld + here.ggidlg + here.ggidlb);
    here.ggisld = -(here.ggisls + here.ggislg + here.ggislb);
    here.gIgbs = -(here.gIgbg + here.gIgbd + here.gIgbb);
    here.gIgcss = -(here.gIgcsg + here.gIgcsd + here.gIgcsb);
    here.gIgcds = -(here.gIgcdg + here.gIgcdd + here.gIgcdb);
    here.cd = cdrain;
    if model.tnoiMod == 0 {
        Abulk = Abulk0_Q * pParam.abulkCVfactor;
        Vdsat = Vgsteff / Abulk;
        Vdseff = if Vds == 0. {
            0.
        } else {
            super::smooth_drain(Vdsat, Vds, DELTA_4).effective
        };
        T0 = Abulk * Vdseff;
        T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1.0e-20);
        T2 = Vdseff / T1;
        T3 = T0 * T2;
        here.qinv =
            Coxeff * pParam.weffCV * here.nf * pParam.leffCV * (Vgsteff - 0.5 * T0 + Abulk * T3);
    } else if model.tnoiMod == 2 {
        here.noiGd0 = here.nf * beta * Vgsteff / (1.0 + gche * Rds);
    }
    cdrain
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_retains_the_tiny_derivative_at_half_partition() {
        for argument in [-1e-18, 1e-18] {
            let p = GatePartition::new(argument);
            assert_eq!(p.source, 0.5);
            assert_eq!(p.drain, 0.5);
            assert!((p.source_slope / 2.5e-33 - 1.).abs() < 8. * f64::EPSILON);
            assert!((p.drain_slope / 5e-33 - 1.).abs() < 8. * f64::EPSILON);
        }
    }

    #[test]
    fn partition_matches_independent_derivatives_across_clamp_regions() {
        for argument in [
            -100., -35., -33., -0.51, -0.49, -0.1, 0., 0.1, 0.49, 0.51, 33., 35., 100.,
        ] {
            let p = GatePartition::new(argument);
            let h = 1e-5;
            let lo = GatePartition::new(argument - h);
            let hi = GatePartition::new(argument + h);
            for (slope, difference) in [
                (p.source_slope, (hi.source - lo.source) / (2. * h)),
                (p.drain_slope, (hi.drain - lo.drain) / (2. * h)),
            ] {
                assert!(
                    (slope - difference).abs() < 1e-6 * slope.abs() + 1e-8,
                    "{argument}: {slope:e}, {difference:e}"
                );
            }
        }
        let extreme = GatePartition::new(1e200);
        assert!(extreme.source.is_finite() && extreme.source < 0.);
        assert!(extreme.drain.is_finite() && extreme.drain > 0.);
    }
}
