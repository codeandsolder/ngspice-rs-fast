//! Junction diode conduction/reverse tunneling and depletion charge stages.
//! Ownership: BSIM4 load inventory fields cbs/cbd/gbs/gbd and capbs/capbd;
//! charge also writes QBS/QBD (no integration here).
#![expect(non_snake_case, reason = "use upstream C equation symbols")]
#![expect(
    clippy::similar_names,
    reason = "source/drain terms retain reference naming"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "non-fused C floating-point operation order"
)]
#![expect(
    clippy::too_many_lines,
    reason = "preserve C diode branches and depletion equation sequence"
)]
#![expect(
    clippy::cognitive_complexity,
    reason = "preserve C diode and charge branch structure"
)]
#![expect(
    clippy::float_cmp,
    reason = "preserve exact C junction branch predicates"
)]
#![expect(
    unused_assignments,
    reason = "preserve C intermediate assignment order"
)]
use super::{
    DEXP, EXP_THRESHOLD, Instance, LoadCkt, MIN_EXP, Model, SizeParam, exp, log, sqrt, state,
};

/// Body-source and body-drain voltage, selecting bulk-resistance internal
/// nodes when present. Both are already polarity-normalized and limited.
#[derive(Clone, Copy)]
pub(super) struct JunctionBias {
    pub(super) source: f64,
    pub(super) drain: f64,
}

/// Evaluate the four load-owned junction conduction members, in C order.
pub(super) fn current(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    ckt: &dyn LoadCkt,
    bias: JunctionBias,
) {
    let JunctionBias {
        source: vbs_jct,
        drain: vbd_jct,
    } = bias;
    let mut SourceSatCurrent: f64 = 0.0;
    let mut DrainSatCurrent: f64 = 0.0;
    let mut evbd: f64 = 0.0;
    let mut evbs: f64 = 0.0;
    let mut Nvtms: f64 = 0.0;
    let mut Nvtmd: f64 = 0.0;
    let mut T0: f64 = 0.0;
    let mut dT0_dVb: f64 = 0.0;
    let mut T1: f64 = 0.0;
    let mut dT1_dVb: f64 = 0.0;
    let mut T2: f64 = 0.0;
    let mut dT2_dVb: f64 = 0.0;
    let mut T3: f64 = 0.0;
    let mut dT3_dVb: f64 = 0.0;
    let mut T4: f64 = 0.0;
    let mut dT4_dVb: f64 = 0.0;
    let mut T5: f64 = 0.0;
    let mut dT5_dVb: f64 = 0.0;
    let mut T6: f64 = 0.0;
    let mut dT6_dVb: f64 = 0.0;
    let mut T9: f64 = 0.0;
    let mut T10: f64 = 0.0;
    let mut devbs_dvb: f64 = 0.0;
    let mut devbd_dvb: f64 = 0.0;
    let mut Nvtmrss: f64 = 0.0;
    let mut Nvtmrssws: f64 = 0.0;
    let mut Nvtmrsswgs: f64 = 0.0;
    let mut Nvtmrsd: f64 = 0.0;
    let mut Nvtmrsswd: f64 = 0.0;
    let mut Nvtmrsswgd: f64 = 0.0;
    Nvtms = model.vtm * model.SjctEmissionCoeff;
    if (here.Aseff <= 0.0) && (here.Pseff <= 0.0) {
        SourceSatCurrent = 0.0;
    } else {
        SourceSatCurrent = here.Aseff * model.SjctTempSatCurDensity
            + here.Pseff * model.SjctSidewallTempSatCurDensity
            + pParam.weffCJ * here.nf * model.SjctGateSidewallTempSatCurDensity;
    }
    if SourceSatCurrent <= 0.0 {
        here.gbs = ckt.gmin();
        here.cbs = here.gbs * vbs_jct;
    } else {
        match model.dioMod {
            0 => {
                evbs = exp(vbs_jct / Nvtms);
                T1 = model.xjbvs * exp(-(model.bvs + vbs_jct) / Nvtms);
                here.gbs = SourceSatCurrent * (evbs + T1) / Nvtms + ckt.gmin();
                here.cbs =
                    SourceSatCurrent * (evbs + here.XExpBVS - T1 - 1.0) + ckt.gmin() * vbs_jct;
            }
            1 => {
                T2 = vbs_jct / Nvtms;
                if T2 < -EXP_THRESHOLD {
                    here.gbs = ckt.gmin();
                    here.cbs = SourceSatCurrent * (MIN_EXP - 1.0) + ckt.gmin() * vbs_jct;
                } else if vbs_jct <= here.vjsmFwd {
                    evbs = exp(T2);
                    here.gbs = SourceSatCurrent * evbs / Nvtms + ckt.gmin();
                    here.cbs = SourceSatCurrent * (evbs - 1.0) + ckt.gmin() * vbs_jct;
                } else {
                    T0 = here.IVjsmFwd / Nvtms;
                    here.gbs = T0 + ckt.gmin();
                    here.cbs = here.IVjsmFwd - SourceSatCurrent
                        + T0 * (vbs_jct - here.vjsmFwd)
                        + ckt.gmin() * vbs_jct;
                }
            }
            2 => {
                if vbs_jct < here.vjsmRev {
                    T0 = vbs_jct / Nvtms;
                    if T0 < -EXP_THRESHOLD {
                        evbs = MIN_EXP;
                        devbs_dvb = 0.0;
                    } else {
                        evbs = exp(T0);
                        devbs_dvb = evbs / Nvtms;
                    }
                    T1 = evbs - 1.0;
                    T2 = here.IVjsmRev + here.SslpRev * (vbs_jct - here.vjsmRev);
                    here.gbs = devbs_dvb * T2 + T1 * here.SslpRev + ckt.gmin();
                    here.cbs = T1 * T2 + ckt.gmin() * vbs_jct;
                } else if vbs_jct <= here.vjsmFwd {
                    T0 = vbs_jct / Nvtms;
                    if T0 < -EXP_THRESHOLD {
                        evbs = MIN_EXP;
                        devbs_dvb = 0.0;
                    } else {
                        evbs = exp(T0);
                        devbs_dvb = evbs / Nvtms;
                    }
                    T1 = (model.bvs + vbs_jct) / Nvtms;
                    if T1 > EXP_THRESHOLD {
                        T2 = MIN_EXP;
                        T3 = 0.0;
                    } else {
                        T2 = exp(-T1);
                        T3 = -T2 / Nvtms;
                    }
                    here.gbs = SourceSatCurrent * (devbs_dvb - model.xjbvs * T3) + ckt.gmin();
                    here.cbs = SourceSatCurrent * (evbs + here.XExpBVS - 1.0 - model.xjbvs * T2)
                        + ckt.gmin() * vbs_jct;
                } else {
                    here.gbs = here.SslpFwd + ckt.gmin();
                    here.cbs = here.IVjsmFwd
                        + here.SslpFwd * (vbs_jct - here.vjsmFwd)
                        + ckt.gmin() * vbs_jct;
                }
            }
            _ => {}
        }
    }
    Nvtmd = model.vtm * model.DjctEmissionCoeff;
    if (here.Adeff <= 0.0) && (here.Pdeff <= 0.0) {
        DrainSatCurrent = 0.0;
    } else {
        DrainSatCurrent = here.Adeff * model.DjctTempSatCurDensity
            + here.Pdeff * model.DjctSidewallTempSatCurDensity
            + pParam.weffCJ * here.nf * model.DjctGateSidewallTempSatCurDensity;
    }
    if DrainSatCurrent <= 0.0 {
        here.gbd = ckt.gmin();
        here.cbd = here.gbd * vbd_jct;
    } else {
        match model.dioMod {
            0 => {
                evbd = exp(vbd_jct / Nvtmd);
                T1 = model.xjbvd * exp(-(model.bvd + vbd_jct) / Nvtmd);
                here.gbd = DrainSatCurrent * (evbd + T1) / Nvtmd + ckt.gmin();
                here.cbd =
                    DrainSatCurrent * (evbd + here.XExpBVD - T1 - 1.0) + ckt.gmin() * vbd_jct;
            }
            1 => {
                T2 = vbd_jct / Nvtmd;
                if T2 < -EXP_THRESHOLD {
                    here.gbd = ckt.gmin();
                    here.cbd = DrainSatCurrent * (MIN_EXP - 1.0) + ckt.gmin() * vbd_jct;
                } else if vbd_jct <= here.vjdmFwd {
                    evbd = exp(T2);
                    here.gbd = DrainSatCurrent * evbd / Nvtmd + ckt.gmin();
                    here.cbd = DrainSatCurrent * (evbd - 1.0) + ckt.gmin() * vbd_jct;
                } else {
                    T0 = here.IVjdmFwd / Nvtmd;
                    here.gbd = T0 + ckt.gmin();
                    here.cbd = here.IVjdmFwd - DrainSatCurrent
                        + T0 * (vbd_jct - here.vjdmFwd)
                        + ckt.gmin() * vbd_jct;
                }
            }
            2 => {
                if vbd_jct < here.vjdmRev {
                    T0 = vbd_jct / Nvtmd;
                    if T0 < -EXP_THRESHOLD {
                        evbd = MIN_EXP;
                        devbd_dvb = 0.0;
                    } else {
                        evbd = exp(T0);
                        devbd_dvb = evbd / Nvtmd;
                    }
                    T1 = evbd - 1.0;
                    T2 = here.IVjdmRev + here.DslpRev * (vbd_jct - here.vjdmRev);
                    here.gbd = devbd_dvb * T2 + T1 * here.DslpRev + ckt.gmin();
                    here.cbd = T1 * T2 + ckt.gmin() * vbd_jct;
                } else if vbd_jct <= here.vjdmFwd {
                    T0 = vbd_jct / Nvtmd;
                    if T0 < -EXP_THRESHOLD {
                        evbd = MIN_EXP;
                        devbd_dvb = 0.0;
                    } else {
                        evbd = exp(T0);
                        devbd_dvb = evbd / Nvtmd;
                    }
                    T1 = (model.bvd + vbd_jct) / Nvtmd;
                    if T1 > EXP_THRESHOLD {
                        T2 = MIN_EXP;
                        T3 = 0.0;
                    } else {
                        T2 = exp(-T1);
                        T3 = -T2 / Nvtmd;
                    }
                    here.gbd = DrainSatCurrent * (devbd_dvb - model.xjbvd * T3) + ckt.gmin();
                    here.cbd = DrainSatCurrent * (evbd + here.XExpBVD - 1.0 - model.xjbvd * T2)
                        + ckt.gmin() * vbd_jct;
                } else {
                    here.gbd = here.DslpFwd + ckt.gmin();
                    here.cbd = here.IVjdmFwd
                        + here.DslpFwd * (vbd_jct - here.vjdmFwd)
                        + ckt.gmin() * vbd_jct;
                }
            }
            _ => {}
        }
    }
    Nvtmrssws = model.vtm0 * model.njtsswstemp;
    Nvtmrsswgs = model.vtm0 * model.njtsswgstemp;
    Nvtmrss = model.vtm0 * model.njtsstemp;
    Nvtmrsswd = model.vtm0 * model.njtsswdtemp;
    Nvtmrsswgd = model.vtm0 * model.njtsswgdtemp;
    Nvtmrsd = model.vtm0 * model.njtsdtemp;
    if (model.vtss - vbs_jct) < (model.vtss * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbs_jct / Nvtmrss * T9;
        (T1, T10) = DEXP(T0);
        dT1_dVb = T10 / Nvtmrss * T9;
    } else {
        T9 = 1.0 / (model.vtss - vbs_jct);
        T0 = -vbs_jct / Nvtmrss * model.vtss * T9;
        dT0_dVb = model.vtss / Nvtmrss * (T9 + vbs_jct * T9 * T9);
        (T1, T10) = DEXP(T0);
        dT1_dVb = T10 * dT0_dVb;
    }
    if (model.vtsd - vbd_jct) < (model.vtsd * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbd_jct / Nvtmrsd * T9;
        (T2, T10) = DEXP(T0);
        dT2_dVb = T10 / Nvtmrsd * T9;
    } else {
        T9 = 1.0 / (model.vtsd - vbd_jct);
        T0 = -vbd_jct / Nvtmrsd * model.vtsd * T9;
        dT0_dVb = model.vtsd / Nvtmrsd * (T9 + vbd_jct * T9 * T9);
        (T2, T10) = DEXP(T0);
        dT2_dVb = T10 * dT0_dVb;
    }
    if (model.vtssws - vbs_jct) < (model.vtssws * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbs_jct / Nvtmrssws * T9;
        (T3, T10) = DEXP(T0);
        dT3_dVb = T10 / Nvtmrssws * T9;
    } else {
        T9 = 1.0 / (model.vtssws - vbs_jct);
        T0 = -vbs_jct / Nvtmrssws * model.vtssws * T9;
        dT0_dVb = model.vtssws / Nvtmrssws * (T9 + vbs_jct * T9 * T9);
        (T3, T10) = DEXP(T0);
        dT3_dVb = T10 * dT0_dVb;
    }
    if (model.vtsswd - vbd_jct) < (model.vtsswd * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbd_jct / Nvtmrsswd * T9;
        (T4, T10) = DEXP(T0);
        dT4_dVb = T10 / Nvtmrsswd * T9;
    } else {
        T9 = 1.0 / (model.vtsswd - vbd_jct);
        T0 = -vbd_jct / Nvtmrsswd * model.vtsswd * T9;
        dT0_dVb = model.vtsswd / Nvtmrsswd * (T9 + vbd_jct * T9 * T9);
        (T4, T10) = DEXP(T0);
        dT4_dVb = T10 * dT0_dVb;
    }
    if (model.vtsswgs - vbs_jct) < (model.vtsswgs * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbs_jct / Nvtmrsswgs * T9;
        (T5, T10) = DEXP(T0);
        dT5_dVb = T10 / Nvtmrsswgs * T9;
    } else {
        T9 = 1.0 / (model.vtsswgs - vbs_jct);
        T0 = -vbs_jct / Nvtmrsswgs * model.vtsswgs * T9;
        dT0_dVb = model.vtsswgs / Nvtmrsswgs * (T9 + vbs_jct * T9 * T9);
        (T5, T10) = DEXP(T0);
        dT5_dVb = T10 * dT0_dVb;
    }
    if (model.vtsswgd - vbd_jct) < (model.vtsswgd * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbd_jct / Nvtmrsswgd * T9;
        (T6, T10) = DEXP(T0);
        dT6_dVb = T10 / Nvtmrsswgd * T9;
    } else {
        T9 = 1.0 / (model.vtsswgd - vbd_jct);
        T0 = -vbd_jct / Nvtmrsswgd * model.vtsswgd * T9;
        dT0_dVb = model.vtsswgd / Nvtmrsswgd * (T9 + vbd_jct * T9 * T9);
        (T6, T10) = DEXP(T0);
        dT6_dVb = T10 * dT0_dVb;
    }
    here.gbs += here.SjctTempRevSatCur * dT1_dVb
        + here.SswTempRevSatCur * dT3_dVb
        + here.SswgTempRevSatCur * dT5_dVb;
    here.cbs -= here.SjctTempRevSatCur * (T1 - 1.0)
        + here.SswTempRevSatCur * (T3 - 1.0)
        + here.SswgTempRevSatCur * (T5 - 1.0);
    here.gbd += here.DjctTempRevSatCur * dT2_dVb
        + here.DswTempRevSatCur * dT4_dVb
        + here.DswgTempRevSatCur * dT6_dVb;
    here.cbd -= here.DjctTempRevSatCur * (T2 - 1.0)
        + here.DswTempRevSatCur * (T4 - 1.0)
        + here.DswgTempRevSatCur * (T6 - 1.0);
}

/// Evaluate depletion charge and capacitance (C's finished label).
pub(super) fn charge(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    ckt: &mut dyn LoadCkt,
    bias: JunctionBias,
) {
    let JunctionBias {
        source: vbs_jct,
        drain: vbd_jct,
    } = bias;
    let mut czbd: f64 = 0.0;
    let mut czbdsw: f64 = 0.0;
    let mut czbdswg: f64 = 0.0;
    let mut czbs: f64 = 0.0;
    let mut czbssw: f64 = 0.0;
    let mut czbsswg: f64 = 0.0;
    let mut arg: f64 = 0.0;
    let mut sarg: f64 = 0.0;
    let mut MJD: f64 = 0.0;
    let mut MJSWD: f64 = 0.0;
    let mut MJSWGD: f64 = 0.0;
    let mut MJS: f64 = 0.0;
    let mut MJSWS: f64 = 0.0;
    let mut MJSWGS: f64 = 0.0;
    let mut T0: f64 = 0.0;
    let mut T1: f64 = 0.0;
    czbd = model.DunitAreaTempJctCap * here.Adeff;
    czbs = model.SunitAreaTempJctCap * here.Aseff;
    czbdsw = model.DunitLengthSidewallTempJctCap * here.Pdeff;
    czbdswg = model.DunitLengthGateSidewallTempJctCap * pParam.weffCJ * here.nf;
    czbssw = model.SunitLengthSidewallTempJctCap * here.Pseff;
    czbsswg = model.SunitLengthGateSidewallTempJctCap * pParam.weffCJ * here.nf;
    MJS = model.SbulkJctBotGradingCoeff;
    MJSWS = model.SbulkJctSideGradingCoeff;
    MJSWGS = model.SbulkJctGateSideGradingCoeff;
    MJD = model.DbulkJctBotGradingCoeff;
    MJSWD = model.DbulkJctSideGradingCoeff;
    MJSWGD = model.DbulkJctGateSideGradingCoeff;
    if vbs_jct == 0.0 {
        ckt.set_state0(state::QBS, 0.0);
        here.capbs = czbs + czbssw + czbsswg;
    } else if vbs_jct < 0.0 {
        if czbs > 0.0 {
            arg = 1.0 - vbs_jct / model.PhiBS;
            if MJS == 0.5 {
                sarg = 1.0 / sqrt(arg);
            } else {
                sarg = exp(-MJS * log(arg));
            }
            ckt.set_state0(
                state::QBS,
                model.PhiBS * czbs * (1.0 - arg * sarg) / (1.0 - MJS),
            );
            here.capbs = czbs * sarg;
        } else {
            ckt.set_state0(state::QBS, 0.0);
            here.capbs = 0.0;
        }
        if czbssw > 0.0 {
            arg = 1.0 - vbs_jct / model.PhiBSWS;
            if MJSWS == 0.5 {
                sarg = 1.0 / sqrt(arg);
            } else {
                sarg = exp(-MJSWS * log(arg));
            }
            ckt.set_state0(
                state::QBS,
                ckt.state0(state::QBS)
                    + (model.PhiBSWS * czbssw * (1.0 - arg * sarg) / (1.0 - MJSWS)),
            );
            here.capbs += czbssw * sarg;
        }
        if czbsswg > 0.0 {
            arg = 1.0 - vbs_jct / model.PhiBSWGS;
            if MJSWGS == 0.5 {
                sarg = 1.0 / sqrt(arg);
            } else {
                sarg = exp(-MJSWGS * log(arg));
            }
            ckt.set_state0(
                state::QBS,
                ckt.state0(state::QBS)
                    + (model.PhiBSWGS * czbsswg * (1.0 - arg * sarg) / (1.0 - MJSWGS)),
            );
            here.capbs += czbsswg * sarg;
        }
    } else {
        T0 = czbs + czbssw + czbsswg;
        T1 = vbs_jct
            * (czbs * MJS / model.PhiBS
                + czbssw * MJSWS / model.PhiBSWS
                + czbsswg * MJSWGS / model.PhiBSWGS);
        ckt.set_state0(state::QBS, vbs_jct * (T0 + 0.5 * T1));
        here.capbs = T0 + T1;
    }
    if vbd_jct == 0.0 {
        ckt.set_state0(state::QBD, 0.0);
        here.capbd = czbd + czbdsw + czbdswg;
    } else if vbd_jct < 0.0 {
        if czbd > 0.0 {
            arg = 1.0 - vbd_jct / model.PhiBD;
            if MJD == 0.5 {
                sarg = 1.0 / sqrt(arg);
            } else {
                sarg = exp(-MJD * log(arg));
            }
            ckt.set_state0(
                state::QBD,
                model.PhiBD * czbd * (1.0 - arg * sarg) / (1.0 - MJD),
            );
            here.capbd = czbd * sarg;
        } else {
            ckt.set_state0(state::QBD, 0.0);
            here.capbd = 0.0;
        }
        if czbdsw > 0.0 {
            arg = 1.0 - vbd_jct / model.PhiBSWD;
            if MJSWD == 0.5 {
                sarg = 1.0 / sqrt(arg);
            } else {
                sarg = exp(-MJSWD * log(arg));
            }
            ckt.set_state0(
                state::QBD,
                ckt.state0(state::QBD)
                    + (model.PhiBSWD * czbdsw * (1.0 - arg * sarg) / (1.0 - MJSWD)),
            );
            here.capbd += czbdsw * sarg;
        }
        if czbdswg > 0.0 {
            arg = 1.0 - vbd_jct / model.PhiBSWGD;
            if MJSWGD == 0.5 {
                sarg = 1.0 / sqrt(arg);
            } else {
                sarg = exp(-MJSWGD * log(arg));
            }
            ckt.set_state0(
                state::QBD,
                ckt.state0(state::QBD)
                    + (model.PhiBSWGD * czbdswg * (1.0 - arg * sarg) / (1.0 - MJSWGD)),
            );
            here.capbd += czbdswg * sarg;
        }
    } else {
        T0 = czbd + czbdsw + czbdswg;
        T1 = vbd_jct
            * (czbd * MJD / model.PhiBD
                + czbdsw * MJSWD / model.PhiBSWD
                + czbdswg * MJSWGD / model.PhiBSWGD);
        ckt.set_state0(state::QBD, vbd_jct * (T0 + 0.5 * T1));
        here.capbd = T0 + T1;
    }
}
