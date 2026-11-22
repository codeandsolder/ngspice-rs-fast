//! Junction diode conduction/reverse tunneling and depletion charge stages.
//! Ownership: BSIM4 load inventory fields cbs/cbd/gbs/gbd and capbs/capbd;
//! charge also writes QBS/QBD (no integration here).
#![expect(non_snake_case, reason = "use upstream C equation symbols")]
#![expect(
    clippy::similar_names,
    reason = "source/drain terms retain reference naming"
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

/// Keep tiny forward junction current instead of subtracting rounded
/// exponentials. Outside the cancellation region reuse the exponential.
fn forward_excess(exponent: f64, exponential: f64) -> f64 {
    if exponent.abs() <= 0.5 {
        exponent.exp_m1()
    } else {
        exponential - 1.
    }
}

/// Preserve independent setup and load clamping. Factor the exponential
/// only when both values are unclamped and the voltage is close to zero.
fn reverse_excess(
    exponent: f64,
    equilibrium: f64,
    reverse: f64,
    equilibrium_unclamped: bool,
    reverse_unclamped: bool,
) -> f64 {
    match (equilibrium_unclamped, reverse_unclamped) {
        (false, false) if equilibrium.is_finite() && reverse.is_finite() => 0.,
        (true, true) if exponent.abs() <= 0.5 => -equilibrium * (-exponent).exp_m1(),
        _ => equilibrium - reverse,
    }
}

/// DEXP's unchanged clamp branches with its value shifted by one.
fn tunneling_excess(exponent: f64) -> (f64, f64) {
    let (value, slope) = DEXP(exponent);
    (forward_excess(exponent, value), slope)
}

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
        SourceSatCurrent = (pParam.weffCJ * here.nf).mul_add(
            model.SjctGateSidewallTempSatCurDensity,
            here.Pseff.mul_add(
                model.SjctSidewallTempSatCurDensity,
                here.Aseff * model.SjctTempSatCurDensity,
            ),
        );
    }
    if SourceSatCurrent <= 0.0 {
        here.gbs = ckt.gmin();
        here.cbs = here.gbs * vbs_jct;
    } else {
        match model.dioMod {
            0 => {
                T0 = vbs_jct / Nvtms;
                evbs = exp(T0);
                T1 = model.xjbvs * exp(-(model.bvs + vbs_jct) / Nvtms);
                here.gbs = SourceSatCurrent * (evbs + T1) / Nvtms + ckt.gmin();
                let excess = forward_excess(T0, evbs)
                    + reverse_excess(
                        T0,
                        here.XExpBVS,
                        T1,
                        model.bvs / Nvtms <= EXP_THRESHOLD,
                        true,
                    );
                here.cbs = ckt.gmin().mul_add(vbs_jct, SourceSatCurrent * excess);
            }
            1 => {
                T2 = vbs_jct / Nvtms;
                if T2 < -EXP_THRESHOLD {
                    here.gbs = ckt.gmin();
                    here.cbs = ckt
                        .gmin()
                        .mul_add(vbs_jct, SourceSatCurrent * (MIN_EXP - 1.0));
                } else if vbs_jct <= here.vjsmFwd {
                    evbs = exp(T2);
                    here.gbs = SourceSatCurrent * evbs / Nvtms + ckt.gmin();
                    here.cbs = ckt
                        .gmin()
                        .mul_add(vbs_jct, SourceSatCurrent * forward_excess(T2, evbs));
                } else {
                    T0 = here.IVjsmFwd / Nvtms;
                    here.gbs = T0 + ckt.gmin();
                    here.cbs = ckt.gmin().mul_add(
                        vbs_jct,
                        T0.mul_add(vbs_jct - here.vjsmFwd, here.IVjsmFwd - SourceSatCurrent),
                    );
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
                    T1 = forward_excess(T0, evbs);
                    T2 = here.SslpRev.mul_add(vbs_jct - here.vjsmRev, here.IVjsmRev);
                    here.gbs = T1.mul_add(here.SslpRev, devbs_dvb * T2) + ckt.gmin();
                    here.cbs = ckt.gmin().mul_add(vbs_jct, T1 * T2);
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
                    here.gbs =
                        SourceSatCurrent.mul_add(model.xjbvs.mul_add(-T3, devbs_dvb), ckt.gmin());
                    let excess = forward_excess(T0, evbs)
                        + reverse_excess(
                            T0,
                            here.XExpBVS,
                            model.xjbvs * T2,
                            model.bvs / Nvtms <= EXP_THRESHOLD,
                            T1 <= EXP_THRESHOLD,
                        );
                    here.cbs = ckt.gmin().mul_add(vbs_jct, SourceSatCurrent * excess);
                } else {
                    here.gbs = here.SslpFwd + ckt.gmin();
                    here.cbs = ckt.gmin().mul_add(
                        vbs_jct,
                        here.SslpFwd.mul_add(vbs_jct - here.vjsmFwd, here.IVjsmFwd),
                    );
                }
            }
            _ => {}
        }
    }
    Nvtmd = model.vtm * model.DjctEmissionCoeff;
    if (here.Adeff <= 0.0) && (here.Pdeff <= 0.0) {
        DrainSatCurrent = 0.0;
    } else {
        DrainSatCurrent = (pParam.weffCJ * here.nf).mul_add(
            model.DjctGateSidewallTempSatCurDensity,
            here.Pdeff.mul_add(
                model.DjctSidewallTempSatCurDensity,
                here.Adeff * model.DjctTempSatCurDensity,
            ),
        );
    }
    if DrainSatCurrent <= 0.0 {
        here.gbd = ckt.gmin();
        here.cbd = here.gbd * vbd_jct;
    } else {
        match model.dioMod {
            0 => {
                T0 = vbd_jct / Nvtmd;
                evbd = exp(T0);
                T1 = model.xjbvd * exp(-(model.bvd + vbd_jct) / Nvtmd);
                here.gbd = DrainSatCurrent * (evbd + T1) / Nvtmd + ckt.gmin();
                let excess = forward_excess(T0, evbd)
                    + reverse_excess(
                        T0,
                        here.XExpBVD,
                        T1,
                        model.bvd / Nvtmd <= EXP_THRESHOLD,
                        true,
                    );
                here.cbd = ckt.gmin().mul_add(vbd_jct, DrainSatCurrent * excess);
            }
            1 => {
                T2 = vbd_jct / Nvtmd;
                if T2 < -EXP_THRESHOLD {
                    here.gbd = ckt.gmin();
                    here.cbd = ckt
                        .gmin()
                        .mul_add(vbd_jct, DrainSatCurrent * (MIN_EXP - 1.0));
                } else if vbd_jct <= here.vjdmFwd {
                    evbd = exp(T2);
                    here.gbd = DrainSatCurrent * evbd / Nvtmd + ckt.gmin();
                    here.cbd = ckt
                        .gmin()
                        .mul_add(vbd_jct, DrainSatCurrent * forward_excess(T2, evbd));
                } else {
                    T0 = here.IVjdmFwd / Nvtmd;
                    here.gbd = T0 + ckt.gmin();
                    here.cbd = ckt.gmin().mul_add(
                        vbd_jct,
                        T0.mul_add(vbd_jct - here.vjdmFwd, here.IVjdmFwd - DrainSatCurrent),
                    );
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
                    T1 = forward_excess(T0, evbd);
                    T2 = here.DslpRev.mul_add(vbd_jct - here.vjdmRev, here.IVjdmRev);
                    here.gbd = T1.mul_add(here.DslpRev, devbd_dvb * T2) + ckt.gmin();
                    here.cbd = ckt.gmin().mul_add(vbd_jct, T1 * T2);
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
                    here.gbd =
                        DrainSatCurrent.mul_add(model.xjbvd.mul_add(-T3, devbd_dvb), ckt.gmin());
                    let excess = forward_excess(T0, evbd)
                        + reverse_excess(
                            T0,
                            here.XExpBVD,
                            model.xjbvd * T2,
                            model.bvd / Nvtmd <= EXP_THRESHOLD,
                            T1 <= EXP_THRESHOLD,
                        );
                    here.cbd = ckt.gmin().mul_add(vbd_jct, DrainSatCurrent * excess);
                } else {
                    here.gbd = here.DslpFwd + ckt.gmin();
                    here.cbd = ckt.gmin().mul_add(
                        vbd_jct,
                        here.DslpFwd.mul_add(vbd_jct - here.vjdmFwd, here.IVjdmFwd),
                    );
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
        (T1, T10) = tunneling_excess(T0);
        dT1_dVb = T10 / Nvtmrss * T9;
    } else {
        T9 = 1.0 / (model.vtss - vbs_jct);
        T0 = -vbs_jct / Nvtmrss * model.vtss * T9;
        dT0_dVb = model.vtss / Nvtmrss * (vbs_jct * T9).mul_add(T9, T9);
        (T1, T10) = tunneling_excess(T0);
        dT1_dVb = T10 * dT0_dVb;
    }
    if (model.vtsd - vbd_jct) < (model.vtsd * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbd_jct / Nvtmrsd * T9;
        (T2, T10) = tunneling_excess(T0);
        dT2_dVb = T10 / Nvtmrsd * T9;
    } else {
        T9 = 1.0 / (model.vtsd - vbd_jct);
        T0 = -vbd_jct / Nvtmrsd * model.vtsd * T9;
        dT0_dVb = model.vtsd / Nvtmrsd * (vbd_jct * T9).mul_add(T9, T9);
        (T2, T10) = tunneling_excess(T0);
        dT2_dVb = T10 * dT0_dVb;
    }
    if (model.vtssws - vbs_jct) < (model.vtssws * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbs_jct / Nvtmrssws * T9;
        (T3, T10) = tunneling_excess(T0);
        dT3_dVb = T10 / Nvtmrssws * T9;
    } else {
        T9 = 1.0 / (model.vtssws - vbs_jct);
        T0 = -vbs_jct / Nvtmrssws * model.vtssws * T9;
        dT0_dVb = model.vtssws / Nvtmrssws * (vbs_jct * T9).mul_add(T9, T9);
        (T3, T10) = tunneling_excess(T0);
        dT3_dVb = T10 * dT0_dVb;
    }
    if (model.vtsswd - vbd_jct) < (model.vtsswd * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbd_jct / Nvtmrsswd * T9;
        (T4, T10) = tunneling_excess(T0);
        dT4_dVb = T10 / Nvtmrsswd * T9;
    } else {
        T9 = 1.0 / (model.vtsswd - vbd_jct);
        T0 = -vbd_jct / Nvtmrsswd * model.vtsswd * T9;
        dT0_dVb = model.vtsswd / Nvtmrsswd * (vbd_jct * T9).mul_add(T9, T9);
        (T4, T10) = tunneling_excess(T0);
        dT4_dVb = T10 * dT0_dVb;
    }
    if (model.vtsswgs - vbs_jct) < (model.vtsswgs * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbs_jct / Nvtmrsswgs * T9;
        (T5, T10) = tunneling_excess(T0);
        dT5_dVb = T10 / Nvtmrsswgs * T9;
    } else {
        T9 = 1.0 / (model.vtsswgs - vbs_jct);
        T0 = -vbs_jct / Nvtmrsswgs * model.vtsswgs * T9;
        dT0_dVb = model.vtsswgs / Nvtmrsswgs * (vbs_jct * T9).mul_add(T9, T9);
        (T5, T10) = tunneling_excess(T0);
        dT5_dVb = T10 * dT0_dVb;
    }
    if (model.vtsswgd - vbd_jct) < (model.vtsswgd * 1e-3) {
        T9 = 1.0e3;
        T0 = -vbd_jct / Nvtmrsswgd * T9;
        (T6, T10) = tunneling_excess(T0);
        dT6_dVb = T10 / Nvtmrsswgd * T9;
    } else {
        T9 = 1.0 / (model.vtsswgd - vbd_jct);
        T0 = -vbd_jct / Nvtmrsswgd * model.vtsswgd * T9;
        dT0_dVb = model.vtsswgd / Nvtmrsswgd * (vbd_jct * T9).mul_add(T9, T9);
        (T6, T10) = tunneling_excess(T0);
        dT6_dVb = T10 * dT0_dVb;
    }
    here.gbs += here.SswgTempRevSatCur.mul_add(
        dT5_dVb,
        here.SswTempRevSatCur
            .mul_add(dT3_dVb, here.SjctTempRevSatCur * dT1_dVb),
    );
    here.cbs -= here.SswgTempRevSatCur.mul_add(
        T5,
        here.SswTempRevSatCur
            .mul_add(T3, here.SjctTempRevSatCur * (T1)),
    );
    here.gbd += here.DswgTempRevSatCur.mul_add(
        dT6_dVb,
        here.DswTempRevSatCur
            .mul_add(dT4_dVb, here.DjctTempRevSatCur * dT2_dVb),
    );
    here.cbd -= here.DswgTempRevSatCur.mul_add(
        T6,
        here.DswTempRevSatCur
            .mul_add(T4, here.DjctTempRevSatCur * (T2)),
    );
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
                model.PhiBS * czbs * arg.mul_add(-sarg, 1.0) / (1.0 - MJS),
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
                    + (model.PhiBSWS * czbssw * arg.mul_add(-sarg, 1.0) / (1.0 - MJSWS)),
            );
            here.capbs = czbssw.mul_add(sarg, here.capbs);
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
                    + (model.PhiBSWGS * czbsswg * arg.mul_add(-sarg, 1.0) / (1.0 - MJSWGS)),
            );
            here.capbs = czbsswg.mul_add(sarg, here.capbs);
        }
    } else {
        T0 = czbs + czbssw + czbsswg;
        T1 = vbs_jct
            * (czbs * MJS / model.PhiBS
                + czbssw * MJSWS / model.PhiBSWS
                + czbsswg * MJSWGS / model.PhiBSWGS);
        ckt.set_state0(state::QBS, vbs_jct * 0.5f64.mul_add(T1, T0));
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
                model.PhiBD * czbd * arg.mul_add(-sarg, 1.0) / (1.0 - MJD),
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
                    + (model.PhiBSWD * czbdsw * arg.mul_add(-sarg, 1.0) / (1.0 - MJSWD)),
            );
            here.capbd = czbdsw.mul_add(sarg, here.capbd);
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
                    + (model.PhiBSWGD * czbdswg * arg.mul_add(-sarg, 1.0) / (1.0 - MJSWGD)),
            );
            here.capbd = czbdswg.mul_add(sarg, here.capbd);
        }
    } else {
        T0 = czbd + czbdsw + czbdswg;
        T1 = vbd_jct
            * (czbd * MJD / model.PhiBD
                + czbdsw * MJSWD / model.PhiBSWD
                + czbdswg * MJSWGD / model.PhiBSWGD);
        ckt.set_state0(state::QBD, vbd_jct * 0.5f64.mul_add(T1, T0));
        here.capbd = T0 + T1;
    }
}

#[cfg(test)]
mod tests {
    use super::{forward_excess, reverse_excess, tunneling_excess};

    #[test]
    fn near_zero_diode_currents_keep_their_sign_and_derivative() {
        for exponent in [1e-30_f64, -1e-30] {
            assert_eq!(forward_excess(exponent, exponent.exp()), exponent);
            assert_eq!(
                reverse_excess(exponent, 0.3, 0.3, true, true),
                0.3 * exponent
            );
            assert_eq!(tunneling_excess(exponent), (exponent, 1.));
        }
        assert_eq!(reverse_excess(0., 0.3, 0.3, false, false), 0.);
        assert_eq!(reverse_excess(0., 0.3, 0.1, false, true), 0.3_f64 - 0.1);
    }
}
