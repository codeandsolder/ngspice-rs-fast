//! Intrinsic electrostatics and channel transport from b4ld.c.
#![expect(
    non_snake_case,
    reason = "C-translated equation symbols identify their upstream terms"
)]
#![expect(
    clippy::similar_names,
    reason = "C equation names preserve the reference distinctions"
)]
#![expect(clippy::manual_midpoint, reason = "retain C midpoint rounding order")]
#![expect(
    clippy::suboptimal_flops,
    reason = "Remaining translated product-rule sums and branch expressions have no demonstrated cancellation defect; indiscriminate software FMA would add portable-target cost. See BSIM_CHANNEL_NUMERICS.md."
)]
#![expect(
    unused_assignments,
    reason = "reference intermediate assignments remain in their equation sequence"
)]
#![expect(
    unused_variables,
    reason = "reference intermediate symbols retain their calculation context"
)]
#![expect(
    unused_mut,
    reason = "reference scratch variables may be assigned on different branches"
)]
#![expect(
    clippy::float_cmp,
    reason = "C branch predicates compare floating point exactly"
)]
#![expect(
    clippy::too_many_lines,
    reason = "preserve cohesive C physics equation sequence"
)]
#![expect(
    clippy::cognitive_complexity,
    reason = "retain version-specific reference physics branches"
)]
use super::bias::LoadBias;
use super::{
    CHARGE, Charge_q, DEXP, EPS0, EPSSI, EXP_THRESHOLD, Instance, LoadCkt, MAX_EXP, MIN_EXP, MM,
    Model, SizeParam, exp, log, smooth_drain, smooth_positive, sqrt,
};

use crate::devices::bsim_physics::{
    SlopeParameters, ThresholdParameters, ThresholdRollOff, bsim4_body_limit, channel_conductance,
    channel_rolloff, dibl_early_voltage, effective_overdrive, impact_ionization, poly_depletion,
    rational_floor, reciprocal_floor, saturation_early_voltage, saturation_shape,
    saturation_voltage, series_conductance, substrate_early_voltage, surface_potential,
};
use crate::primitives::SpiceResult;

/// Body potential, threshold, effective gate bias and their independent derivatives.
pub(super) struct Electrostatics {
    pub(super) Leff: f64,
    pub(super) Lpe_Vb: f64,
    pub(super) Phis: f64,
    pub(super) Vbs: f64,
    pub(super) Vbseff: f64,
    pub(super) Vds: f64,
    pub(super) Vgs: f64,
    pub(super) Vgs_eff: f64,
    pub(super) Vgst: f64,
    pub(super) Vgsteff: f64,
    pub(super) Vth: f64,
    pub(super) Vth_NarrowW: f64,
    pub(super) Vtm: f64,
    pub(super) Vtm0: f64,
    pub(super) Xdep: f64,
    pub(super) dPhis_dVb: f64,
    pub(super) dVbseff_dVb: f64,
    pub(super) dVgs_eff_dVg: f64,
    pub(super) dVgsteff_dVb: f64,
    pub(super) dVgsteff_dVd: f64,
    pub(super) dVgsteff_dVg: f64,
    pub(super) dVth_dVb: f64,
    pub(super) dVth_dVd: f64,
    pub(super) dXdep_dVb: f64,
    pub(super) dn_dVb: f64,
    pub(super) dn_dVd: f64,
    pub(super) dsqrtPhis_dVb: f64,
    pub(super) epsrox: f64,
    pub(super) epssub: f64,
    pub(super) n: f64,
    pub(super) sqrtPhis: f64,
    pub(super) toxe: f64,
}

/// Unscaled channel current, effective drain voltage, bulk charge factor and transport quantities used by leakage/C-V.
pub(super) struct Transport {
    pub(super) Abulk0_Q: f64,
    pub(super) Coxeff: f64,
    pub(super) Rds: f64,
    pub(super) Vdseff: f64,
    pub(super) beta: f64,
    pub(super) unscaled_drain_current: f64,
    pub(super) dAbulk0_Q_dVb: f64,
    pub(super) dVdseff_dVb: f64,
    pub(super) dVdseff_dVd: f64,
    pub(super) dVdseff_dVg: f64,
    pub(super) gche: f64,
}

/// Threshold and effective gate-voltage equations, in the reference orientation.
pub(super) fn electrostatics(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    ckt: &dyn LoadCkt,
    bias: &LoadBias,
) -> SpiceResult<Electrostatics> {
    let &LoadBias {
        vbd,
        vbs,
        vds,
        vgd,
        vgs,
        ..
    } = bias;
    let mut vgs_eff: f64;
    let mut vgd_eff: f64;
    let mut dvgs_eff_dvg: f64;
    let mut dvgd_eff_dvg: f64;
    let mut Vdb: f64;
    let mut Vds: f64;
    let mut Vgs: f64;
    let mut Vbs: f64;
    let mut Vgs_eff: f64;
    let mut Vth_NarrowW: f64;
    let mut Phis: f64;
    let mut dPhis_dVb: f64;
    let mut sqrtPhis: f64;
    let mut dsqrtPhis_dVb: f64;
    let mut Vth: f64;
    let mut dVth_dVb: f64;
    let mut dVth_dVd: f64;
    let mut Vgst: f64;
    let mut dVgs_eff_dVg: f64;
    let mut Vtm: f64;
    let mut Vtm0: f64;
    let mut n: f64;
    let mut dn_dVb: f64;
    let mut dn_dVd: f64;
    let mut V0: f64;
    let mut Vbseff: f64;
    let mut dVbseff_dVb: f64;
    let mut T0: f64;
    let mut T1: f64;
    let mut T2: f64;
    let mut dT2_dVd: f64;
    let mut T3: f64;
    let mut dT3_dVd: f64;
    let mut T4: f64;
    let mut dT4_dVd: f64;
    let mut T10: f64;
    let mut Lpe_Vb: f64;
    let mut dDITS_Sft_dVb: f64;
    let mut dDITS_Sft_dVd: f64;
    let mut DITS_Sft2: f64;
    let mut dDITS_Sft2_dVd: f64;
    let mut Xdep: f64;
    let mut dXdep_dVb: f64;
    let mut Theta0: f64;
    let mut TempRatio: f64;
    let mut Vgsteff: f64;
    let mut dVgsteff_dVg: f64;
    let mut dVgsteff_dVd: f64;
    let mut dVgsteff_dVb: f64;
    let mut Leff: f64;
    let mut epssub: f64;
    let mut toxe: f64;
    let mut epsrox: f64;
    if vds >= 0.0 {
        here.mode = 1;
        Vds = vds;
        Vgs = vgs;
        Vbs = vbs;
        Vdb = vds - vbs;
    } else {
        here.mode = -1;
        Vds = -vds;
        Vgs = vgd;
        Vbs = vbd;
        Vdb = -vbs;
    }
    if model.mtrlMod != 0 {
        epsrox = 3.9;
        toxe = model.eot;
        epssub = EPS0 * model.epsrsub;
    } else {
        epsrox = model.epsrox;
        toxe = model.toxe;
        epssub = EPSSI;
    }
    (Vbseff, dVbseff_dVb) = bsim4_body_limit(&Vbs, here.vbsc, pParam.phi)?;
    let surface = surface_potential(&Vbseff, pParam.phi, None)?;
    Phis = surface.potential;
    sqrtPhis = surface.root;
    dPhis_dVb = surface.potential_partial;
    dsqrtPhis_dVb = surface.root_partial;
    Xdep = pParam.Xdep0 * sqrtPhis / pParam.sqrtPhi;
    dXdep_dVb = (pParam.Xdep0 / pParam.sqrtPhi) * dsqrtPhis_dVb;
    Leff = pParam.leff;
    Vtm = model.vtm;
    Vtm0 = model.vtm0;
    V0 = pParam.vbi - pParam.phi;
    let short = channel_rolloff(
        &Vbseff,
        &Xdep,
        pParam.dvt2,
        model.factor1,
        pParam.dvt1 * Leff,
        ThresholdRollOff::Bsim4,
    )?;
    let narrow = channel_rolloff(
        &Vbseff,
        &Xdep,
        pParam.dvt2w,
        model.factor1,
        pParam.dvt1w * pParam.weff * Leff,
        ThresholdRollOff::Bsim4,
    )?;
    let body_depth = [[0., 0., 1.], [0., 0., dXdep_dVb]];
    let short_slopes = short.project(&body_depth);
    let narrow_slopes = narrow.project(&body_depth);
    Theta0 = short.value;
    here.thetavth = pParam.dvt0 * Theta0;
    TempRatio = ckt.temp() / model.tnom - 1.0;
    let length_ratio = pParam.lpe0 / Leff;
    let root_offset = length_ratio / ((1. + length_ratio).sqrt() + 1.);
    let temperature_offset = (pParam.kt1 + pParam.kt1l / Leff)
        .mul_add(TempRatio, pParam.k1ox * root_offset * pParam.sqrtPhi);
    Vth_NarrowW = toxe * pParam.phi / (pParam.weff + pParam.w0);
    Lpe_Vb = sqrt(1.0 + pParam.lpeb / Leff);
    let threshold = ThresholdParameters {
        fixed: pParam.k3.mul_add(
            Vth_NarrowW,
            (pParam.k1 * Lpe_Vb).mul_add(-pParam.sqrtPhi, f64::from(model.r#type) * here.vth0)
                + temperature_offset,
        ),
        surface: pParam.k1ox * Lpe_Vb,
        body: pParam
            .kt2
            .mul_add(TempRatio, pParam.k3b.mul_add(Vth_NarrowW, -here.k2ox)),
        short: -pParam.dvt0 * V0,
        narrow: -pParam.dvt0w * V0,
        eta0: here.eta0,
        etab: pParam.etab,
        theta0vb0: pParam.theta0vb0,
    }
    .evaluate(&Vbseff, &Vds, &sqrtPhis, &Theta0, &narrow.value)?;
    Vth = threshold.value;
    [_, dVth_dVd, dVth_dVb] = threshold.project(&[
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 0., dsqrtPhis_dVb],
        short_slopes,
        narrow_slopes,
    ]);
    here.nstar = model.vtm / Charge_q * (model.coxe + epssub / Xdep + pParam.cit);
    let slope = SlopeParameters {
        depletion: pParam.nfactor * epssub,
        coupling: [pParam.cdsc, pParam.cdscb, pParam.cdscd],
        interface: pParam.cit,
        oxide: model.coxe,
    }
    .evaluate(&Vbseff, &Vds, &Xdep, &Theta0)?;
    n = slope.value;
    [_, dn_dVd, dn_dVb] = slope.project(&[
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 0., dXdep_dVb],
        short_slopes,
    ]);
    if pParam.dvtp0 > 0.0 {
        T0 = -pParam.dvtp1 * Vds;
        if T0 < -EXP_THRESHOLD {
            T2 = MIN_EXP;
            dT2_dVd = 0.0;
        } else {
            T2 = exp(T0);
            dT2_dVd = -pParam.dvtp1 * T2;
        }
        T3 = Leff + pParam.dvtp0 * (1.0 + T2);
        dT3_dVd = pParam.dvtp0 * dT2_dVd;
        if model.tempMod < 2 {
            T4 = -Vtm * (pParam.dvtp0 * (1.0 + T2) / Leff).ln_1p();
            dT4_dVd = -Vtm * dT3_dVd / T3;
        } else {
            T4 = -model.vtm0 * (pParam.dvtp0 * (1.0 + T2) / Leff).ln_1p();
            dT4_dVd = -model.vtm0 * dT3_dVd / T3;
        }
        dDITS_Sft_dVd = dn_dVd * T4 + n * dT4_dVd;
        dDITS_Sft_dVb = T4 * dn_dVb;
        Vth -= n * T4;
        dVth_dVd -= dDITS_Sft_dVd;
        dVth_dVb -= dDITS_Sft_dVb;
    }
    if (pParam.dvtp4 == 0.0) || (pParam.dvtp2factor == 0.0) {
        T0 = 0.0;
        DITS_Sft2 = 0.0;
    } else {
        T1 = 2.0 * pParam.dvtp4 * Vds;
        (T0, T10) = DEXP(T1);
        DITS_Sft2 = pParam.dvtp2factor * (T0 - 1.0) / (T0 + 1.0);
        dDITS_Sft2_dVd = pParam.dvtp2factor * pParam.dvtp4 * 4.0 * T10 / ((T0 + 1.0) * (T0 + 1.0));
        Vth -= DITS_Sft2;
        dVth_dVd -= dDITS_Sft2_dVd;
    }
    here.von = Vth;
    T0 = here.vfb + pParam.phi;
    if model.mtrlMod == 0 {
        T1 = EPSSI;
    } else {
        T1 = model.epsrgate * EPS0;
    }
    (vgs_eff, dvgs_eff_dvg) = poly_depletion(vgs, T0, pParam.ngate, T1, model.coxe, CHARGE)?;
    (vgd_eff, dvgd_eff_dvg) = poly_depletion(vgd, T0, pParam.ngate, T1, model.coxe, CHARGE)?;
    if here.mode > 0 {
        Vgs_eff = vgs_eff;
        dVgs_eff_dVg = dvgs_eff_dvg;
    } else {
        Vgs_eff = vgd_eff;
        dVgs_eff_dVg = dvgd_eff_dvg;
    }
    here.vgs_eff = vgs_eff;
    here.vgd_eff = vgd_eff;
    here.dvgs_eff_dvg = dvgs_eff_dvg;
    here.dvgd_eff_dvg = dvgd_eff_dvg;
    Vgst = Vgs_eff - Vth;
    let (value, voltage_partial, factor_partial) = effective_overdrive(
        &Vgst,
        &n,
        Vtm,
        pParam.mstar,
        pParam.voffcbn,
        model.coxe / pParam.cdep0,
    )?;
    Vgsteff = value;
    here.Vgsteff = Vgsteff;
    dVgsteff_dVg = voltage_partial * dVgs_eff_dVg;
    dVgsteff_dVd = factor_partial.mul_add(dn_dVd, -voltage_partial * dVth_dVd);
    dVgsteff_dVb = factor_partial.mul_add(dn_dVb, -voltage_partial * dVth_dVb);
    Ok(Electrostatics {
        Leff,
        Lpe_Vb,
        Phis,
        Vbs,
        Vbseff,
        Vds,
        Vgs,
        Vgs_eff,
        Vgst,
        Vgsteff,
        Vth,
        Vth_NarrowW,
        Vtm,
        Vtm0,
        Xdep,
        dPhis_dVb,
        dVbseff_dVb,
        dVgs_eff_dVg,
        dVgsteff_dVb,
        dVgsteff_dVd,
        dVgsteff_dVg,
        dVth_dVb,
        dVth_dVd,
        dXdep_dVb,
        dn_dVb,
        dn_dVd,
        dsqrtPhis_dVb,
        epsrox,
        epssub,
        n,
        sqrtPhis,
        toxe,
    })
}

/// Channel mobility, saturation, current, substrate current and series resistances.
pub(super) fn transport(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    bias: &LoadBias,
    electrostatics: &Electrostatics,
) -> SpiceResult<Transport> {
    let &LoadBias {
        vbd,
        vbs,
        vdes,
        vds,
        vgd,
        vgs,
        vses,
        ..
    } = bias;
    let &Electrostatics {
        Leff,
        Lpe_Vb,
        Vbseff,
        Vds,
        Vgsteff,
        Vth,
        Vth_NarrowW,
        Vtm,
        Xdep,
        dVbseff_dVb,
        dVgsteff_dVb,
        dVgsteff_dVd,
        dVgsteff_dVg,
        dVth_dVb,
        dVth_dVd,
        dXdep_dVb,
        dsqrtPhis_dVb,
        epssub,
        sqrtPhis,
        toxe,
        ..
    } = electrostatics;
    let mut dgstot_dvd: f64;
    let mut dgstot_dvg: f64;
    let mut dgstot_dvs: f64;
    let mut dgstot_dvb: f64;
    let mut dgdtot_dvd: f64;
    let mut dgdtot_dvg: f64;
    let mut dgdtot_dvs: f64;
    let mut dgdtot_dvb: f64;
    let mut Rs: f64;
    let mut Rd: f64;
    let mut vgs_eff: f64;
    let mut vgd_eff: f64;
    let mut dvgs_eff_dvg: f64;
    let mut dvgd_eff_dvg: f64;
    let mut dRs_dvg: f64;
    let mut dRd_dvg: f64;
    let mut dRs_dvb: f64;
    let mut dRd_dvb: f64;
    let mut dT0_dvg: f64;
    let mut dT1_dvb: f64;
    let mut dT3_dvg: f64;
    let mut dT3_dvb: f64;
    let mut cdrain: f64;
    let mut Tcen: f64;
    let mut dTcen_dVg: f64;
    let mut Coxeff: f64;
    let mut dCoxeff_dVg: f64;
    let mut Denomi: f64;
    let mut dDenomi_dVg: f64;
    let mut dDenomi_dVd: f64;
    let mut dDenomi_dVb: f64;
    let mut ueff: f64;
    let mut dueff_dVg: f64;
    let mut dueff_dVd: f64;
    let mut dueff_dVb: f64;
    let mut Esat: f64;
    let mut Vdsat: f64;
    let mut EsatL: f64;
    let mut dEsatL_dVg: f64;
    let mut dEsatL_dVd: f64;
    let mut dEsatL_dVb: f64;
    let mut dVdsat_dVg: f64;
    let mut dVdsat_dVb: f64;
    let mut dVdsat_dVd: f64;
    let mut Vasat: f64;
    let mut dVasat_dVg: f64;
    let mut dVasat_dVb: f64;
    let mut dVasat_dVd: f64;
    let mut Va: f64;
    let mut dVa_dVd: f64;
    let mut dVa_dVg: f64;
    let mut dVa_dVb: f64;
    let mut VgsteffVth: f64;
    let mut dT11_dVg: f64;
    let mut T0: f64;
    let mut dT0_dVg: f64;
    let mut dT0_dVd: f64;
    let mut dT0_dVb: f64;
    let mut T1: f64;
    let mut dT1_dVg: f64;
    let mut dT1_dVd: f64;
    let mut dT1_dVb: f64;
    let mut T2: f64;
    let mut dT2_dVg: f64;
    let mut dT2_dVd: f64;
    let mut dT2_dVb: f64;
    let mut T3: f64;
    let mut dT3_dVg: f64;
    let mut dT3_dVb: f64;
    let mut T4: f64;
    let mut T5: f64;
    let mut dT5_dVg: f64;
    let mut dT5_dVd: f64;
    let mut dT5_dVb: f64;
    let mut T6: f64;
    let mut dT6_dVg: f64;
    let mut dT6_dVd: f64;
    let mut dT6_dVb: f64;
    let mut T7: f64;
    let mut T8: f64;
    let mut dT8_dVg: f64;
    let mut dT8_dVd: f64;
    let mut dT8_dVb: f64;
    let mut T9: f64;
    let mut dT9_dVg: f64;
    let mut dT9_dVd: f64;
    let mut dT9_dVb: f64;
    let mut T10: f64;
    let mut dT10_dVg: f64;
    let mut dT10_dVb: f64;
    let mut dT10_dVd: f64;
    let mut T11: f64;
    let mut T12: f64;
    let mut T13: f64;
    let mut T14: f64;
    let mut tmp: f64;
    let mut Abulk: f64;
    let mut dAbulk_dVb: f64;
    let mut Abulk0: f64;
    let mut Abulk0_Q: f64;
    let mut dAbulk0_dVb: f64;
    let mut dAbulk0_Q_dVb: f64;
    let mut Cclm: f64;
    let mut dCclm_dVg: f64;
    let mut dCclm_dVd: f64;
    let mut dCclm_dVb: f64;
    let mut FP: f64;
    let mut dFP_dVg: f64;
    let mut PvagTerm: f64;
    let mut dPvagTerm_dVg: f64;
    let mut dPvagTerm_dVd: f64;
    let mut dPvagTerm_dVb: f64;
    let mut VADITS: f64;
    let mut dVADITS_dVg: f64;
    let mut dVADITS_dVd: f64;
    let mut VACLM: f64;
    let mut dVACLM_dVg: f64;
    let mut dVACLM_dVd: f64;
    let mut dVACLM_dVb: f64;
    let mut VADIBL: f64;
    let mut dVADIBL_dVg: f64;
    let mut dVADIBL_dVd: f64;
    let mut dVADIBL_dVb: f64;
    let mut tmp1: f64;
    let mut tmp2: f64;
    let mut tmp3: f64;
    let mut tmp4: f64;
    let mut Lambda: f64;
    let mut dLambda_dVg: f64;
    let mut a1: f64;
    let mut Vdseff: f64;
    let mut dVdseff_dVg: f64;
    let mut dVdseff_dVd: f64;
    let mut dVdseff_dVb: f64;
    let mut diffVds: f64;
    let mut dAbulk_dVg: f64;
    let mut beta: f64;
    let mut dbeta_dVg: f64;
    let mut dbeta_dVd: f64;
    let mut dbeta_dVb: f64;
    let mut gche: f64;
    let mut Idl: f64;
    let mut dIdl_dVg: f64;
    let mut dIdl_dVd: f64;
    let mut dIdl_dVb: f64;
    let mut Idsa: f64;
    let mut dIdsa_dVg: f64;
    let mut dIdsa_dVd: f64;
    let mut dIdsa_dVb: f64;
    let mut Ids: f64;
    let mut Gm: f64;
    let mut Gds: f64;
    let mut Gmb: f64;
    let mut Isub: f64;
    let mut Gbd: f64;
    let mut Gbg: f64;
    let mut Gbb: f64;
    let mut VASCBE: f64;
    let mut dVASCBE_dVg: f64;
    let mut dVASCBE_dVd: f64;
    let mut dVASCBE_dVb: f64;
    let mut CoxeffWovL: f64;
    let mut Rds: f64;
    let mut dRds_dVg: f64;
    let mut dRds_dVb: f64;
    let mut WVCox: f64;
    let mut WVCoxRds: f64;
    let mut Vgst2Vtm: f64;
    let mut Weff: f64;
    let mut dWeff_dVg: f64;
    let mut dWeff_dVb: f64;
    let mut vs: f64;
    let mut Fsevl: f64;
    let mut dvs_dVg: f64;
    let mut dvs_dVd: f64;
    let mut dvs_dVb: f64;
    let mut dFsevl_dVg: f64;
    let mut dFsevl_dVd: f64;
    let mut dFsevl_dVb: f64;
    T9 = sqrtPhis - pParam.sqrtPhi;
    Weff = pParam.weff - 2.0 * (pParam.dwg * Vgsteff + pParam.dwb * T9);
    dWeff_dVg = -2.0 * pParam.dwg;
    dWeff_dVb = -2.0 * pParam.dwb * dsqrtPhis_dVb;
    (Weff, T0) = rational_floor(Weff, 2e-8, 0., 1e8)?;
    dWeff_dVg *= T0;
    dWeff_dVb *= T0;
    if model.rdsMod == 1 {
        dRds_dVb = 0.0;
        dRds_dVg = dRds_dVb;
        Rds = dRds_dVg;
    } else {
        T0 = 1.0 + pParam.prwg * Vgsteff;
        dT0_dVg = -pParam.prwg / T0 / T0;
        T1 = pParam.prwb * T9;
        dT1_dVb = pParam.prwb * dsqrtPhis_dVb;
        T2 = 1.0 / T0 + T1;
        let (positive, slope) = smooth_positive(T2, 0.0025)?;
        T3 = 2.0 * positive;
        dT3_dVg = 2.0 * slope;
        dT3_dVb = dT3_dVg * dT1_dVb;
        dT3_dVg *= dT0_dVg;
        T4 = pParam.rds0 * 0.5;
        Rds = pParam.rdswmin + T3 * T4;
        dRds_dVg = T4 * dT3_dVg;
        dRds_dVb = T4 * dT3_dVb;
        if Rds > 0.0 {
            here.grdsw = 1.0 / Rds * here.nf;
        } else {
            here.grdsw = 0.0;
        }
    }
    T9 = 0.5 * pParam.k1ox * Lpe_Vb / sqrtPhis;
    T1 = T9 + here.k2ox - pParam.k3b * Vth_NarrowW;
    dT1_dVb = -T9 / sqrtPhis * dsqrtPhis_dVb;
    T9 = sqrt(pParam.xj * Xdep);
    tmp1 = Leff + 2.0 * T9;
    T5 = Leff / tmp1;
    tmp2 = pParam.a0 * T5;
    tmp3 = pParam.weff + pParam.b1;
    tmp4 = pParam.b0 / tmp3;
    T2 = tmp2 + tmp4;
    dT2_dVb = -T9 / tmp1 / Xdep * dXdep_dVb;
    T6 = T5 * T5;
    T7 = T5 * T6;
    Abulk0 = 1.0 + T1 * T2;
    dAbulk0_dVb = T1 * tmp2 * dT2_dVb + T2 * dT1_dVb;
    T8 = pParam.ags * pParam.a0 * T7;
    dAbulk_dVg = -T1 * T8;
    Abulk = Abulk0 + dAbulk_dVg * Vgsteff;
    dAbulk_dVb = dAbulk0_dVb - T8 * Vgsteff * (dT1_dVb + 3.0 * T1 * dT2_dVb);
    (Abulk0, T9) = rational_floor(Abulk0, 0.1, 0., 20.)?;
    dAbulk0_dVb *= T9;
    (Abulk, T10) = rational_floor(Abulk, 0.1, 0., 20.)?;
    dAbulk_dVb *= T10;
    dAbulk_dVg *= T10;
    here.Abulk = Abulk;
    (T0, dT0_dVb) = reciprocal_floor(pParam.keta * Vbseff, -0.9, 20.)?;
    dT0_dVb *= pParam.keta;
    dAbulk_dVg *= T0;
    dAbulk_dVb = dAbulk_dVb * T0 + Abulk * dT0_dVb;
    dAbulk0_Q_dVb = dAbulk0_dVb;
    dAbulk0_dVb = dAbulk0_dVb * T0 + Abulk0 * dT0_dVb;
    Abulk *= T0;
    Abulk0_Q = Abulk0;
    Abulk0 *= T0;
    if pParam.ketac == pParam.keta {
        dAbulk0_Q_dVb = dAbulk0_dVb;
        Abulk0_Q = Abulk0;
    } else {
        (T0, dT0_dVb) = reciprocal_floor(pParam.ketac * Vbseff, -0.9, 20.)?;
        dT0_dVb *= pParam.ketac;
        dAbulk0_Q_dVb = dAbulk0_Q_dVb * T0 + Abulk0_Q * dT0_dVb;
        Abulk0_Q *= T0;
    }
    if (model.mtrlMod != 0) && model.mtrlCompatMod == 0 {
        T14 = 2.0 * f64::from(model.r#type) * (model.phig - model.easub - 0.5 * model.Eg0 + 0.45);
    } else {
        T14 = 0.0;
    }
    if model.mobMod == 0 {
        T0 = Vgsteff + Vth + Vth - T14;
        T2 = pParam.ua + pParam.uc * Vbseff;
        T3 = T0 / toxe;
        T12 = sqrt(Vth * Vth + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * Vth;
        T6 = T8 * Vth;
        T5 = T3 * (T2 + pParam.ub * T3) + T6;
        T7 = -2.0 * T6 * T9;
        T11 = T7 * Vth / T12;
        dDenomi_dVg = (T2 + 2.0 * pParam.ub * T3) / toxe;
        T13 = 2.0 * (dDenomi_dVg + T11 + T8);
        dDenomi_dVd = T13 * dVth_dVd;
        dDenomi_dVb = T13 * dVth_dVb + pParam.uc * T3;
        dDenomi_dVg += T7;
    } else if model.mobMod == 1 {
        T0 = Vgsteff + Vth + Vth - T14;
        T2 = 1.0 + pParam.uc * Vbseff;
        T3 = T0 / toxe;
        T4 = T3 * (pParam.ua + pParam.ub * T3);
        T12 = sqrt(Vth * Vth + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * Vth;
        T6 = T8 * Vth;
        T5 = T4 * T2 + T6;
        T7 = -2.0 * T6 * T9;
        T11 = T7 * Vth / T12;
        dDenomi_dVg = (pParam.ua + 2.0 * pParam.ub * T3) * T2 / toxe;
        T13 = 2.0 * (dDenomi_dVg + T11 + T8);
        dDenomi_dVd = T13 * dVth_dVd;
        dDenomi_dVb = T13 * dVth_dVb + pParam.uc * T4;
        dDenomi_dVg += T7;
    } else if model.mobMod == 2 {
        T0 = (Vgsteff + here.vtfbphi1) / toxe;
        T1 = exp(pParam.eu * log(T0));
        dT1_dVg = T1 * pParam.eu / T0 / toxe;
        T2 = pParam.ua + pParam.uc * Vbseff;
        T12 = sqrt(Vth * Vth + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * Vth;
        T6 = T8 * Vth;
        T5 = T1 * T2 + T6;
        T7 = -2.0 * T6 * T9;
        T11 = T7 * Vth / T12;
        dDenomi_dVg = T2 * dT1_dVg + T7;
        T13 = 2.0 * (T11 + T8);
        dDenomi_dVd = T13 * dVth_dVd;
        dDenomi_dVb = T13 * dVth_dVb + T1 * pParam.uc;
    } else if model.mobMod == 4 {
        T0 = Vgsteff + here.vtfbphi1 - T14;
        T2 = pParam.ua + pParam.uc * Vbseff;
        T3 = T0 / toxe;
        T12 = sqrt(here.vtfbphi1 * here.vtfbphi1 + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * here.vtfbphi1;
        T6 = T8 * here.vtfbphi1;
        T5 = T3 * (T2 + pParam.ub * T3) + T6;
        T7 = -2.0 * T6 * T9;
        dDenomi_dVg = (T2 + 2.0 * pParam.ub * T3) / toxe;
        dDenomi_dVd = 0.0;
        dDenomi_dVb = pParam.uc * T3;
        dDenomi_dVg += T7;
    } else if model.mobMod == 5 {
        T0 = Vgsteff + here.vtfbphi1 - T14;
        T2 = 1.0 + pParam.uc * Vbseff;
        T3 = T0 / toxe;
        T4 = T3 * (pParam.ua + pParam.ub * T3);
        T12 = sqrt(here.vtfbphi1 * here.vtfbphi1 + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * here.vtfbphi1;
        T6 = T8 * here.vtfbphi1;
        T5 = T4 * T2 + T6;
        T7 = -2.0 * T6 * T9;
        dDenomi_dVg = (pParam.ua + 2.0 * pParam.ub * T3) * T2 / toxe;
        dDenomi_dVd = 0.0;
        dDenomi_dVb = pParam.uc * T4;
        dDenomi_dVg += T7;
    } else if model.mobMod == 6 {
        T0 = (Vgsteff + here.vtfbphi1) / toxe;
        T1 = exp(pParam.eu * log(T0));
        dT1_dVg = T1 * pParam.eu / T0 / toxe;
        T2 = pParam.ua + pParam.uc * Vbseff;
        T12 = sqrt(here.vtfbphi1 * here.vtfbphi1 + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * here.vtfbphi1;
        T6 = T8 * here.vtfbphi1;
        T5 = T1 * T2 + T6;
        T7 = -2.0 * T6 * T9;
        dDenomi_dVg = T2 * dT1_dVg + T7;
        dDenomi_dVd = 0.0;
        dDenomi_dVb = T1 * pParam.uc;
    } else {
        T0 = (Vgsteff + here.vtfbphi1) * 1.0e-8 / toxe / 6.0;
        T1 = exp(pParam.eu * log(T0));
        dT1_dVg = T1 * pParam.eu * 1.0e-8 / T0 / toxe / 6.0;
        T2 = pParam.ua + pParam.uc * Vbseff;
        VgsteffVth = pParam.VgsteffVth;
        T10 = exp(pParam.ucs * log(0.5 + 0.5 * Vgsteff / VgsteffVth));
        T11 = pParam.ud / T10;
        dT11_dVg = -0.5 * pParam.ucs * T11 / (0.5 + 0.5 * Vgsteff / VgsteffVth) / VgsteffVth;
        dDenomi_dVg = T2 * dT1_dVg + dT11_dVg;
        dDenomi_dVd = 0.0;
        dDenomi_dVb = T1 * pParam.uc;
        T5 = T1 * T2 + T11;
    }
    (Denomi, T9) = rational_floor(T5, -0.8, 1., 10.)?;
    dDenomi_dVg *= T9;
    dDenomi_dVd *= T9;
    dDenomi_dVb *= T9;
    ueff = here.u0temp / Denomi;
    here.ueff = ueff;
    T9 = -ueff / Denomi;
    dueff_dVg = T9 * dDenomi_dVg;
    dueff_dVd = T9 * dDenomi_dVd;
    dueff_dVb = T9 * dDenomi_dVb;
    WVCox = Weff * here.vsattemp * model.coxe;
    WVCoxRds = WVCox * Rds;
    Esat = 2.0 * here.vsattemp / ueff;
    EsatL = Esat * Leff;
    here.EsatL = EsatL;
    T0 = -EsatL / ueff;
    dEsatL_dVg = T0 * dueff_dVg;
    dEsatL_dVd = T0 * dueff_dVd;
    dEsatL_dVb = T0 * dueff_dVb;
    (Lambda, dLambda_dVg) = saturation_shape(&Vgsteff, pParam.a1, pParam.a2)?;
    Vgst2Vtm = Vgsteff + 2.0 * Vtm;
    let resistance_slopes = [
        WVCox.mul_add(dRds_dVg, WVCoxRds * dWeff_dVg / Weff),
        0.,
        WVCox.mul_add(dRds_dVb, WVCoxRds * dWeff_dVb / Weff),
    ];
    let bulk_slopes = [dAbulk_dVg, 0., dAbulk_dVb];
    let shape_slopes = [dLambda_dVg, 0., 0.];
    let saturation = saturation_voltage(&Vgst2Vtm, &Abulk, &EsatL, &WVCoxRds, &Lambda)?;
    let saturation_slopes = saturation.project(&[
        [1., 0., 0.],
        bulk_slopes,
        [dEsatL_dVg, dEsatL_dVd, dEsatL_dVb],
        resistance_slopes,
        shape_slopes,
    ]);
    [dVdsat_dVg, dVdsat_dVd, dVdsat_dVb] = saturation_slopes;
    Vdsat = saturation.value;
    here.vdsat = Vdsat;
    let smoothing = smooth_drain(Vdsat, Vds, pParam.delta)?;
    Vdseff = smoothing.effective;
    dVdseff_dVg = smoothing.saturation_partial * dVdsat_dVg;
    dVdseff_dVd = smoothing
        .saturation_partial
        .mul_add(dVdsat_dVd, smoothing.drain_partial);
    dVdseff_dVb = smoothing.saturation_partial * dVdsat_dVb;
    diffVds = smoothing.drain_excess;
    here.Vdseff = Vdseff;
    if (model.lambdaGiven) && (model.lambda > 0.0) {
        T1 = Leff * ueff;
        T2 = pParam.lambda / T1;
        T3 = -T2 / T1 * Leff;
        dT2_dVd = T3 * dueff_dVd;
        dT2_dVg = T3 * dueff_dVg;
        dT2_dVb = T3 * dueff_dVb;
        T5 = 1.0 / (Esat * pParam.litl);
        T4 = -T5 / EsatL;
        dT5_dVg = dEsatL_dVg * T4;
        dT5_dVd = dEsatL_dVd * T4;
        dT5_dVb = dEsatL_dVb * T4;
        T6 = 1.0 + diffVds * T5;
        dT6_dVg = dT5_dVg * diffVds - dVdseff_dVg * T5;
        dT6_dVd = dT5_dVd * diffVds + (1.0 - dVdseff_dVd) * T5;
        dT6_dVb = dT5_dVb * diffVds - dVdseff_dVb * T5;
        T7 = 2.0 / (T6 * T6 + 1.0);
        T8 = 1.0 - T7;
        T9 = T6 * T7 * T7;
        dT8_dVg = T9 * dT6_dVg;
        dT8_dVd = T9 * dT6_dVd;
        dT8_dVb = T9 * dT6_dVb;
        T10 = 1.0 + T2 * T8;
        dT10_dVg = dT2_dVg * T8 + T2 * dT8_dVg;
        dT10_dVd = dT2_dVd * T8 + T2 * dT8_dVd;
        dT10_dVb = dT2_dVb * T8 + T2 * dT8_dVb;
        if T10 == 1.0 {
            dT10_dVb = 0.0;
            dT10_dVd = dT10_dVb;
            dT10_dVg = dT10_dVd;
        }
        dEsatL_dVg *= T10;
        dEsatL_dVg += EsatL * dT10_dVg;
        dEsatL_dVd *= T10;
        dEsatL_dVd += EsatL * dT10_dVd;
        dEsatL_dVb *= T10;
        dEsatL_dVb += EsatL * dT10_dVb;
        EsatL *= T10;
        Esat = EsatL / Leff;
        here.EsatL = EsatL;
    }
    let early =
        saturation_early_voltage(&Vgsteff, &Abulk, &Vdsat, &EsatL, &WVCoxRds, &Lambda, Vtm)?;
    [dVasat_dVg, dVasat_dVd, dVasat_dVb] = early.project(&[
        [1., 0., 0.],
        bulk_slopes,
        saturation_slopes,
        [dEsatL_dVg, dEsatL_dVd, dEsatL_dVb],
        resistance_slopes,
        shape_slopes,
    ]);
    Vasat = early.value;
    tmp1 = here.vtfbphi2;
    tmp2 = 2.0e8 * here.toxp;
    dT0_dVg = 1.0 / tmp2;
    T0 = (Vgsteff + tmp1) * dT0_dVg;
    tmp3 = exp(model.bdos * 0.7 * log(T0));
    T1 = 1.0 + tmp3;
    T2 = model.bdos * 0.7 * tmp3 / T0;
    Tcen = model.ados * 1.9e-9 / T1;
    dTcen_dVg = -Tcen * T2 * dT0_dVg / T1;
    Coxeff = epssub * here.coxp / (epssub + here.coxp * Tcen);
    here.Coxeff = Coxeff;
    dCoxeff_dVg = -Coxeff * Coxeff * dTcen_dVg / epssub;
    CoxeffWovL = Coxeff * Weff / Leff;
    beta = ueff * CoxeffWovL;
    T3 = ueff / Leff;
    dbeta_dVg = CoxeffWovL * dueff_dVg + T3 * (Weff * dCoxeff_dVg + Coxeff * dWeff_dVg);
    dbeta_dVd = CoxeffWovL * dueff_dVd;
    dbeta_dVb = CoxeffWovL * dueff_dVb + T3 * Coxeff * dWeff_dVb;
    here.AbovVgst2Vtm = Abulk / Vgst2Vtm;
    let channel = channel_conductance(&Vgsteff, &Abulk, &Vdseff, &EsatL, &beta, Vtm)?;
    let [dg, dd, db] = channel.project(&[
        [1., 0., 0.],
        [dAbulk_dVg, 0., dAbulk_dVb],
        [dVdseff_dVg, dVdseff_dVd, dVdseff_dVb],
        [dEsatL_dVg, dEsatL_dVd, dEsatL_dVb],
        [dbeta_dVg, dbeta_dVd, dbeta_dVb],
    ]);
    gche = channel.value;
    let series = series_conductance(&gche, &Rds)?;
    [dIdl_dVg, dIdl_dVd, dIdl_dVb] = series.project(&[[dg, dd, db], [dRds_dVg, 0., dRds_dVb]]);
    Idl = series.value;
    if pParam.fprout <= 0.0 {
        FP = 1.0;
        dFP_dVg = 0.0;
    } else {
        T9 = pParam.fprout * sqrt(Leff) / Vgst2Vtm;
        FP = 1.0 / (1.0 + T9);
        dFP_dVg = FP * FP * T9 / Vgst2Vtm;
    }
    T8 = pParam.pvag / EsatL;
    T9 = T8 * Vgsteff;
    (PvagTerm, T4) = rational_floor(T9, -0.9, 1., 20.)?;
    dPvagTerm_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL) * T4;
    dPvagTerm_dVb = -T9 * dEsatL_dVb / EsatL * T4;
    dPvagTerm_dVd = -T9 * dEsatL_dVd / EsatL * T4;
    if (pParam.pclm > MIN_EXP) && (diffVds > 1.0e-10) {
        T0 = 1.0 + Rds * Idl;
        dT0_dVg = dRds_dVg * Idl + Rds * dIdl_dVg;
        dT0_dVd = Rds * dIdl_dVd;
        dT0_dVb = dRds_dVb * Idl + Rds * dIdl_dVb;
        T2 = Vdsat / Esat;
        T1 = Leff + T2;
        dT1_dVg = (dVdsat_dVg - T2 * dEsatL_dVg / Leff) / Esat;
        dT1_dVd = (dVdsat_dVd - T2 * dEsatL_dVd / Leff) / Esat;
        dT1_dVb = (dVdsat_dVb - T2 * dEsatL_dVb / Leff) / Esat;
        Cclm = FP * PvagTerm * T0 * T1 / (pParam.pclm * pParam.litl);
        dCclm_dVg = Cclm * (dFP_dVg / FP + dPvagTerm_dVg / PvagTerm + dT0_dVg / T0 + dT1_dVg / T1);
        dCclm_dVb = Cclm * (dPvagTerm_dVb / PvagTerm + dT0_dVb / T0 + dT1_dVb / T1);
        dCclm_dVd = Cclm * (dPvagTerm_dVd / PvagTerm + dT0_dVd / T0 + dT1_dVd / T1);
        VACLM = Cclm * diffVds;
        dVACLM_dVg = dCclm_dVg * diffVds - dVdseff_dVg * Cclm;
        dVACLM_dVb = dCclm_dVb * diffVds - dVdseff_dVb * Cclm;
        dVACLM_dVd = dCclm_dVd * diffVds + (1.0 - dVdseff_dVd) * Cclm;
    } else {
        Cclm = MAX_EXP;
        VACLM = Cclm;
        dVACLM_dVb = 0.0;
        dVACLM_dVg = dVACLM_dVb;
        dVACLM_dVd = dVACLM_dVg;
        dCclm_dVb = 0.0;
        dCclm_dVg = dCclm_dVb;
        dCclm_dVd = dCclm_dVg;
    }
    if pParam.thetaRout > MIN_EXP {
        let early = dibl_early_voltage(
            &Vgst2Vtm,
            &Abulk,
            &Vdsat,
            &Vbseff,
            pParam.thetaRout,
            pParam.pdiblb,
        )?;
        [dVADIBL_dVg, dVADIBL_dVd, dVADIBL_dVb] =
            early.project(&[[1., 0., 0.], bulk_slopes, saturation_slopes, [0., 0., 1.]]);
        VADIBL = early.value;
        dVADIBL_dVg = dVADIBL_dVg * PvagTerm + VADIBL * dPvagTerm_dVg;
        dVADIBL_dVb = dVADIBL_dVb * PvagTerm + VADIBL * dPvagTerm_dVb;
        dVADIBL_dVd = dVADIBL_dVd * PvagTerm + VADIBL * dPvagTerm_dVd;
        VADIBL *= PvagTerm;
    } else {
        VADIBL = MAX_EXP;
        dVADIBL_dVb = 0.0;
        dVADIBL_dVg = dVADIBL_dVb;
        dVADIBL_dVd = dVADIBL_dVg;
    }
    Va = Vasat + VACLM;
    dVa_dVg = dVasat_dVg + dVACLM_dVg;
    dVa_dVb = dVasat_dVb + dVACLM_dVb;
    dVa_dVd = dVasat_dVd + dVACLM_dVd;
    T0 = pParam.pditsd * Vds;
    if T0 > EXP_THRESHOLD {
        T1 = MAX_EXP;
        dT1_dVd = 0.0;
    } else {
        T1 = exp(T0);
        dT1_dVd = T1 * pParam.pditsd;
    }
    if pParam.pdits > MIN_EXP {
        T2 = 1.0 + model.pditsl * Leff;
        VADITS = (1.0 + T2 * T1) / pParam.pdits;
        dVADITS_dVg = VADITS * dFP_dVg;
        dVADITS_dVd = FP * T2 * dT1_dVd / pParam.pdits;
        VADITS *= FP;
    } else {
        VADITS = MAX_EXP;
        dVADITS_dVd = 0.0;
        dVADITS_dVg = dVADITS_dVd;
    }
    let (value, slope) = substrate_early_voltage(
        &diffVds,
        Leff,
        pParam.pscbe1 * pParam.litl,
        pParam.pscbe2,
        pParam.pscbe2 > 0. && pParam.pscbe1 >= 0.,
    )?;
    VASCBE = value;
    dVASCBE_dVg = -slope * dVdseff_dVg;
    dVASCBE_dVd = slope * (1. - dVdseff_dVd);
    dVASCBE_dVb = -slope * dVdseff_dVb;
    T9 = diffVds / VADIBL;
    T0 = 1.0 + T9;
    Idsa = Idl * T0;
    dIdsa_dVg = T0 * dIdl_dVg - Idl * (dVdseff_dVg + T9 * dVADIBL_dVg) / VADIBL;
    dIdsa_dVd = T0 * dIdl_dVd + Idl * (1.0 - dVdseff_dVd - T9 * dVADIBL_dVd) / VADIBL;
    dIdsa_dVb = T0 * dIdl_dVb - Idl * (dVdseff_dVb + T9 * dVADIBL_dVb) / VADIBL;
    T9 = diffVds / VADITS;
    T0 = 1.0 + T9;
    dIdsa_dVg = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVADITS_dVg) / VADITS;
    dIdsa_dVd = T0 * dIdsa_dVd + Idsa * (1.0 - dVdseff_dVd - T9 * dVADITS_dVd) / VADITS;
    dIdsa_dVb = T0 * dIdsa_dVb - Idsa * dVdseff_dVb / VADITS;
    Idsa *= T0;
    T0 = log(Va / Vasat);
    dT0_dVg = dVa_dVg / Va - dVasat_dVg / Vasat;
    dT0_dVb = dVa_dVb / Va - dVasat_dVb / Vasat;
    dT0_dVd = dVa_dVd / Va - dVasat_dVd / Vasat;
    T1 = T0 / Cclm;
    T9 = 1.0 + T1;
    dT9_dVg = (dT0_dVg - T1 * dCclm_dVg) / Cclm;
    dT9_dVb = (dT0_dVb - T1 * dCclm_dVb) / Cclm;
    dT9_dVd = (dT0_dVd - T1 * dCclm_dVd) / Cclm;
    dIdsa_dVg = dIdsa_dVg * T9 + Idsa * dT9_dVg;
    dIdsa_dVb = dIdsa_dVb * T9 + Idsa * dT9_dVb;
    dIdsa_dVd = dIdsa_dVd * T9 + Idsa * dT9_dVd;
    Idsa *= T9;
    tmp = pParam.alpha0 + pParam.alpha1 * Leff;
    if (tmp <= 0.0) || (pParam.beta0 <= 0.0) {
        Gbg = 0.0;
        Gbb = Gbg;
        Gbd = Gbb;
        Isub = Gbd;
    } else {
        let (value, slope) = impact_ionization(&diffVds, tmp / Leff, pParam.beta0)?;
        T1 = value;
        dT1_dVg = -slope * dVdseff_dVg;
        dT1_dVd = slope * (1. - dVdseff_dVd);
        dT1_dVb = -slope * dVdseff_dVb;
        T4 = Idsa * Vdseff;
        Isub = T1 * T4;
        Gbg = T1 * (dIdsa_dVg * Vdseff + Idsa * dVdseff_dVg) + T4 * dT1_dVg;
        Gbd = T1 * (dIdsa_dVd * Vdseff + Idsa * dVdseff_dVd) + T4 * dT1_dVd;
        Gbb = T1 * (dIdsa_dVb * Vdseff + Idsa * dVdseff_dVb) + T4 * dT1_dVb;
        Gbd += Gbg * dVgsteff_dVd;
        Gbb += Gbg * dVgsteff_dVb;
        Gbg *= dVgsteff_dVg;
        Gbb *= dVbseff_dVb;
    }
    here.csub = Isub;
    here.gbbs = Gbb;
    here.gbgs = Gbg;
    here.gbds = Gbd;
    T9 = diffVds / VASCBE;
    T0 = 1.0 + T9;
    Ids = Idsa * T0;
    Gm = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVASCBE_dVg) / VASCBE;
    Gds = T0 * dIdsa_dVd + Idsa * (1.0 - dVdseff_dVd - T9 * dVASCBE_dVd) / VASCBE;
    Gmb = T0 * dIdsa_dVb - Idsa * (dVdseff_dVb + T9 * dVASCBE_dVb) / VASCBE;
    tmp1 = Gds + Gm * dVgsteff_dVd;
    tmp2 = Gmb + Gm * dVgsteff_dVb;
    tmp3 = Gm;
    Gm = (Ids * dVdseff_dVg + Vdseff * tmp3) * dVgsteff_dVg;
    Gds = Ids * (dVdseff_dVd + dVdseff_dVg * dVgsteff_dVd) + Vdseff * tmp1;
    Gmb = (Ids * (dVdseff_dVb + dVdseff_dVg * dVgsteff_dVb) + Vdseff * tmp2) * dVbseff_dVb;
    cdrain = Ids * Vdseff;
    if (model.vtlGiven) && (model.vtl > 0.0) {
        T12 = 1.0 / Leff / CoxeffWovL;
        T11 = T12 / Vgsteff;
        T10 = -T11 / Vgsteff;
        vs = cdrain * T11;
        dvs_dVg = Gm * T11 + cdrain * T10 * dVgsteff_dVg;
        dvs_dVd = Gds * T11 + cdrain * T10 * dVgsteff_dVd;
        dvs_dVb = Gmb * T11 + cdrain * T10 * dVgsteff_dVb;
        T0 = f64::from(2 * MM);
        T1 = vs / (pParam.vtl * pParam.tfactor);
        if T1 > 0.0 {
            T2 = 1.0 + exp(T0 * log(T1));
            T3 = (T2 - 1.0) * T0 / vs;
            Fsevl = 1.0 / exp(log(T2) / T0);
            dT2_dVg = T3 * dvs_dVg;
            dT2_dVd = T3 * dvs_dVd;
            dT2_dVb = T3 * dvs_dVb;
            T4 = -1.0 / T0 * Fsevl / T2;
            dFsevl_dVg = T4 * dT2_dVg;
            dFsevl_dVd = T4 * dT2_dVd;
            dFsevl_dVb = T4 * dT2_dVb;
        } else {
            Fsevl = 1.0;
            dFsevl_dVg = 0.0;
            dFsevl_dVd = 0.0;
            dFsevl_dVb = 0.0;
        }
        Gm *= Fsevl;
        Gm += cdrain * dFsevl_dVg;
        Gmb *= Fsevl;
        Gmb += cdrain * dFsevl_dVb;
        Gds *= Fsevl;
        Gds += cdrain * dFsevl_dVd;
        cdrain *= Fsevl;
    }
    here.gds = Gds;
    here.gm = Gm;
    here.gmbs = Gmb;
    here.IdovVds = Ids;
    if here.IdovVds <= model.idovvdsc {
        here.IdovVds = model.idovvdsc;
    }
    if (here.rgateMod > 1) || (here.trnqsMod != 0) || (here.acnqsMod != 0) {
        T9 = pParam.xrcrg2 * model.vtm;
        T0 = T9 * beta;
        dT0_dVd = (dbeta_dVd + dbeta_dVg * dVgsteff_dVd) * T9;
        dT0_dVb = (dbeta_dVb + dbeta_dVg * dVgsteff_dVb) * T9;
        dT0_dVg = dbeta_dVg * T9;
        here.gcrg = pParam.xrcrg1 * (T0 + Ids);
        here.gcrgd = pParam.xrcrg1 * (dT0_dVd + tmp1);
        here.gcrgb = pParam.xrcrg1 * (dT0_dVb + tmp2) * dVbseff_dVb;
        here.gcrgg = pParam.xrcrg1 * (dT0_dVg + tmp3) * dVgsteff_dVg;
        if here.nf != 1.0 {
            here.gcrg *= here.nf;
            here.gcrgg *= here.nf;
            here.gcrgd *= here.nf;
            here.gcrgb *= here.nf;
        }
        if here.rgateMod == 2 {
            T10 = here.grgeltd * here.grgeltd;
            T11 = here.grgeltd + here.gcrg;
            here.gcrg = here.grgeltd * here.gcrg / T11;
            T12 = T10 / T11 / T11;
            here.gcrgg *= T12;
            here.gcrgd *= T12;
            here.gcrgb *= T12;
        }
        here.gcrgs = -(here.gcrgg + here.gcrgd + here.gcrgb);
    }
    if model.rdsMod != 0 {
        T0 = vgs - pParam.vfbsd;
        (vgs_eff, dvgs_eff_dvg) = smooth_positive(T0, 2.5e-5)?;
        T0 = 1.0 + pParam.prwg * vgs_eff;
        dT0_dvg = -pParam.prwg / T0 / T0 * dvgs_eff_dvg;
        T1 = -pParam.prwb * vbs;
        dT1_dvb = -pParam.prwb;
        T2 = 1.0 / T0 + T1;
        let (positive, slope) = smooth_positive(T2, 0.0025)?;
        T3 = 2.0 * positive;
        dT3_dvg = 2.0 * slope;
        dT3_dvb = dT3_dvg * dT1_dvb;
        dT3_dvg *= dT0_dvg;
        T4 = pParam.rs0 * 0.5;
        Rs = pParam.rswmin + T3 * T4;
        dRs_dvg = T4 * dT3_dvg;
        dRs_dvb = T4 * dT3_dvb;
        T0 = 1.0 + here.sourceConductance * Rs;
        here.gstot = here.sourceConductance / T0;
        T0 = -here.gstot * here.gstot;
        dgstot_dvd = 0.0;
        dgstot_dvg = T0 * dRs_dvg;
        dgstot_dvb = T0 * dRs_dvb;
        dgstot_dvs = -(dgstot_dvg + dgstot_dvb + dgstot_dvd);
        T0 = vgd - pParam.vfbsd;
        (vgd_eff, dvgd_eff_dvg) = smooth_positive(T0, 2.5e-5)?;
        T0 = 1.0 + pParam.prwg * vgd_eff;
        dT0_dvg = -pParam.prwg / T0 / T0 * dvgd_eff_dvg;
        T1 = -pParam.prwb * vbd;
        dT1_dvb = -pParam.prwb;
        T2 = 1.0 / T0 + T1;
        let (positive, slope) = smooth_positive(T2, 0.0025)?;
        T3 = 2.0 * positive;
        dT3_dvg = 2.0 * slope;
        dT3_dvb = dT3_dvg * dT1_dvb;
        dT3_dvg *= dT0_dvg;
        T4 = pParam.rd0 * 0.5;
        Rd = pParam.rdwmin + T3 * T4;
        dRd_dvg = T4 * dT3_dvg;
        dRd_dvb = T4 * dT3_dvb;
        T0 = 1.0 + here.drainConductance * Rd;
        here.gdtot = here.drainConductance / T0;
        T0 = -here.gdtot * here.gdtot;
        dgdtot_dvs = 0.0;
        dgdtot_dvg = T0 * dRd_dvg;
        dgdtot_dvb = T0 * dRd_dvb;
        dgdtot_dvd = -(dgdtot_dvg + dgdtot_dvb + dgdtot_dvs);
        here.gstotd = vses * dgstot_dvd;
        here.gstotg = vses * dgstot_dvg;
        here.gstots = vses * dgstot_dvs;
        here.gstotb = vses * dgstot_dvb;
        T2 = vdes - vds;
        here.gdtotd = T2 * dgdtot_dvd;
        here.gdtotg = T2 * dgdtot_dvg;
        here.gdtots = T2 * dgdtot_dvs;
        here.gdtotb = T2 * dgdtot_dvb;
    } else {
        here.gstotg = 0.0;
        here.gstotd = here.gstotg;
        here.gstot = here.gstotd;
        here.gstotb = 0.0;
        here.gstots = here.gstotb;
        here.gdtotg = 0.0;
        here.gdtotd = here.gdtotg;
        here.gdtot = here.gdtotd;
        here.gdtotb = 0.0;
        here.gdtots = here.gdtotb;
    }
    Ok(Transport {
        Abulk0_Q,
        Coxeff,
        Rds,
        Vdseff,
        beta,
        unscaled_drain_current: cdrain,
        dAbulk0_Q_dVb,
        dVdseff_dVb,
        dVdseff_dVd,
        dVdseff_dVg,
        gche,
    })
}
