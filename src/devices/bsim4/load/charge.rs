//! Intrinsic charge and transient companion stages of b4ld.c.
#![expect(
    non_snake_case,
    reason = "C-translated equation symbols identify their upstream terms"
)]
#![expect(
    clippy::similar_names,
    reason = "C equation names preserve the reference distinctions"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "Remaining translated product-rule sums and companion stamps have no demonstrated cancellation defect; preserve shared charge identities and avoid indiscriminate software FMA. See BSIM_CHANNEL_NUMERICS.md."
)]
#![expect(
    unused_assignments,
    reason = "reference intermediate assignments remain in their equation sequence"
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
#![expect(
    unused_variables,
    reason = "reference intermediate symbols retain their calculation context"
)]
#![expect(unused_parens, reason = "preserve C expression grouping")]
use super::{
    DELTA_3, DELTA_4, EXP_THRESHOLD, Instance, LoadCkt, MAX_EXP, MIN_EXP, Model, SizeParam,
    depletion_charge, exp, fabs, log, mode, smooth_drain, smooth_positive, state,
};
use super::{
    bias::LoadBias,
    channel::{Electrostatics, Transport},
};
use crate::devices::bsim_physics::{
    CapZeroBias, CapZeroParameters, ChargeChain, cap_zero, channel_inversion, charge_overdrive,
    effective_overdrive, thermal_softplus,
};
use crate::primitives::SpiceResult;

/// Intrinsic terminal charges before extrinsic overlap and junction assembly.
pub(super) struct IntrinsicCharges {
    pub(super) qbulk: f64,
    pub(super) qdrn: f64,
    pub(super) qgate: f64,
}

/// Evaluate the intrinsic C-V model and charge-deficit equilibrium.
pub(super) fn intrinsic(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    ckt: &mut dyn LoadCkt,
    bias: &LoadBias,
    electrostatics: &Electrostatics,
    transport: &Transport,
) -> SpiceResult<IntrinsicCharges> {
    let &LoadBias {
        ChargeComputationNeeded,
        ..
    } = bias;
    let &Electrostatics {
        Phis,
        Vbs,
        Vbseff,
        Vds,
        Vgs_eff,
        Vgst,
        Vtm,
        dPhis_dVb,
        dVbseff_dVb,
        dVgs_eff_dVg,
        dVth_dVb,
        dVth_dVd,
        dn_dVb,
        dn_dVd,
        dsqrtPhis_dVb,
        epssub,
        n,
        sqrtPhis,
        ..
    } = electrostatics;
    let &Transport {
        Abulk0_Q,
        dAbulk0_Q_dVb,
        ..
    } = transport;
    let mut Coxeff: f64;
    let mut Vgsteff: f64;
    let mut dVgsteff_dVb: f64;
    let mut dVgsteff_dVd: f64;
    let mut dVgsteff_dVg: f64;
    let mut ceq: f64;
    let mut geq: f64;
    let mut Vfbeff: f64;
    let mut dVfbeff_dVg: f64;
    let mut dVfbeff_dVb: f64;
    let mut V3: f64;
    let mut qgate: f64 = 0.0;
    let mut qbulk: f64 = 0.0;
    let mut qdrn: f64 = 0.0;
    let mut qsrc: f64;
    let mut Vfb: f64;
    let mut voffcv: f64;
    let mut noff: f64;
    let mut CoxWLcen: f64;
    let mut QovCox: f64;
    let mut LINK: f64;
    let mut VgDP: f64;
    let mut dVgDP_dVg: f64;
    let mut Cox: f64;
    let mut Tox: f64;
    let mut Tcen: f64;
    let mut dTcen_dVg: f64;
    let mut dTcen_dVd: f64;
    let mut dTcen_dVb: f64;
    let mut Ccen: f64;
    let mut dCoxeff_dVd: f64;
    let mut dCoxeff_dVg: f64;
    let mut dCoxeff_dVb: f64;
    let mut VbseffCV: f64;
    let mut dVbseffCV_dVb: f64;
    let mut CoxWL: f64;
    let mut T0: f64;
    let mut dT0_dVg: f64;
    let mut dT0_dVb: f64;
    let mut T1: f64;
    let mut T2: f64;
    let mut T3: f64;
    let mut tmp: f64;
    let mut VdseffCV: f64;
    let mut dVdseffCV_dVg: f64;
    let mut dVdseffCV_dVd: f64;
    let mut dVdseffCV_dVb: f64;
    let mut VdsatCV: f64;
    let mut AbulkCV: f64;
    let mut dAbulkCV_dVb: f64;
    let mut qcheq: f64;
    let mut Cgg: f64;
    let mut Cgd: f64;
    let mut Cgb: f64;
    let mut Csg: f64;
    let mut Csd: f64;
    let mut Csb: f64;
    let mut Cbg: f64;
    let mut Cbd: f64;
    let mut Cbb: f64;
    let mut Cgg1: f64;
    let mut Cgb1: f64;
    let mut Cgd1: f64;
    let mut Cbg1: f64;
    let mut Cbb1: f64;
    let mut Cbd1: f64;
    let mut Qac0: f64;
    let mut Qsub0: f64;
    let mut dQac0_dVg: f64;
    let mut dQac0_dVb: f64;
    let mut dQsub0_dVg: f64;
    let mut dQsub0_dVd: f64;
    let mut dQsub0_dVb: f64;
    let ScalingFactor = 1.0e-9;
    // C `goto finished` (only from the first branch below) skips the
    // block between this if/else chain and the `finished:` label.
    let mut goto_finished = false;
    if (model.xpart < 0.0) || (ChargeComputationNeeded == 0) {
        qbulk = 0.0;
        qsrc = qbulk;
        qdrn = qsrc;
        qgate = qdrn;
        here.cgdb = 0.0;
        here.cgsb = here.cgdb;
        here.cggb = here.cgsb;
        here.cddb = 0.0;
        here.cdsb = here.cddb;
        here.cdgb = here.cdsb;
        here.cbdb = 0.0;
        here.cbsb = here.cbdb;
        here.cbgb = here.cbsb;
        here.csdb = 0.0;
        here.cssb = here.csdb;
        here.csgb = here.cssb;
        here.cbbb = 0.0;
        here.cdbb = here.cbbb;
        here.csbb = here.cdbb;
        here.cgbb = here.csbb;
        here.cqbb = 0.0;
        here.cqgb = here.cqbb;
        here.cqsb = here.cqgb;
        here.cqdb = here.cqsb;
        here.gtau = 0.0;
        goto_finished = true;
    } else if model.capMod == 0 {
        if Vbseff < 0.0 {
            VbseffCV = Vbs;
            dVbseffCV_dVb = 1.0;
        } else {
            VbseffCV = pParam.phi - Phis;
            dVbseffCV_dVb = -dPhis_dVb * dVbseff_dVb;
        }
        let [gate, bulk, drain] = cap_zero(
            &CapZeroBias {
                gate: Vgs_eff,
                body_cv: VbseffCV,
                surface: pParam.k1ox * sqrtPhis,
                drain: Vds,
                bulk_factor: Abulk0_Q * pParam.abulkCVfactor,
            },
            CapZeroParameters {
                flatband: pParam.vfbcv,
                phi: pParam.phi,
                body_effect: pParam.k1ox,
                oxide_area: model.coxe * pParam.weffCV * pParam.leffCV * here.nf,
                partition: model.xpart,
            },
        )?;
        let coordinates = [
            [dVgs_eff_dVg, 0., 0.],
            [0., 0., dVbseffCV_dVb],
            [0., 0., pParam.k1ox * dsqrtPhis_dVb * dVbseff_dVb],
            [0., 1., 0.],
            [0., 0., pParam.abulkCVfactor * dAbulk0_Q_dVb * dVbseff_dVb],
        ];
        let [gg, gd, gb] = gate.project(&coordinates);
        let [bg, bd, bb] = bulk.project(&coordinates);
        let [dg, dd, db] = drain.project(&coordinates);
        qgate = gate.value;
        qbulk = bulk.value;
        qdrn = drain.value;
        here.cggb = gg;
        here.cgdb = gd;
        here.cgsb = -(gg + gd + gb);
        here.cbgb = bg;
        here.cbdb = bd;
        here.cbsb = -(bg + bd + bb);
        here.cdgb = dg;
        here.cddb = dd;
        here.cdsb = -(dg + dd + db);
    } else {
        if Vbseff < 0.0 {
            VbseffCV = Vbseff;
            dVbseffCV_dVb = 1.0;
        } else {
            VbseffCV = pParam.phi - Phis;
            dVbseffCV_dVb = -dPhis_dVb;
        }
        CoxWL = model.coxe * pParam.weffCV * pParam.leffCV * here.nf;
        let (value, voltage_partial, factor_partial) = if model.cvchargeMod == 0 {
            let (value, voltage_partial, factor_partial) =
                thermal_softplus(&(Vgst - pParam.voffcv), &(n * pParam.noff), Vtm)?;
            (value, voltage_partial, factor_partial * pParam.noff)
        } else {
            effective_overdrive(
                &Vgst,
                &n,
                Vtm,
                pParam.mstarcv,
                pParam.voffcbncv,
                model.coxe / pParam.cdep0,
            )?
        };
        Vgsteff = value;
        dVgsteff_dVg = voltage_partial * dVgs_eff_dVg;
        dVgsteff_dVd = factor_partial.mul_add(dn_dVd, -voltage_partial * dVth_dVd);
        dVgsteff_dVb = factor_partial.mul_add(dn_dVb, -voltage_partial * dVth_dVb);
        if model.capMod == 1 {
            Vfb = here.vfbzb;
            V3 = Vfb - Vgs_eff + VbseffCV - DELTA_3;
            let (flatband_shift, flatband_slope) = smooth_positive(V3, DELTA_3 * Vfb.abs())?;
            T1 = flatband_slope;
            Vfbeff = Vfb - flatband_shift;
            dVfbeff_dVg = T1 * dVgs_eff_dVg;
            dVfbeff_dVb = -T1 * dVbseffCV_dVb;
            Qac0 = -CoxWL * flatband_shift;
            dQac0_dVg = CoxWL * dVfbeff_dVg;
            dQac0_dVb = CoxWL * dVfbeff_dVb;
            T3 = Vgs_eff - Vfbeff - VbseffCV - Vgsteff;
            let (charge, slope) = depletion_charge(pParam.k1ox, T3)?;
            Qsub0 = CoxWL * charge;
            T2 = CoxWL * slope;
            dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg);
            dQsub0_dVd = -T2 * dVgsteff_dVd;
            dQsub0_dVb = -T2 * (dVfbeff_dVb + dVbseffCV_dVb + dVgsteff_dVb);
            AbulkCV = Abulk0_Q * pParam.abulkCVfactor;
            dAbulkCV_dVb = pParam.abulkCVfactor * dAbulk0_Q_dVb;
            VdsatCV = Vgsteff / AbulkCV;
            let smoothing = smooth_drain(VdsatCV, Vds, DELTA_4)?;
            VdseffCV = smoothing.effective;
            dVdseffCV_dVg = smoothing.saturation_partial * (1.0 / AbulkCV);
            dVdseffCV_dVd = smoothing.drain_partial;
            dVdseffCV_dVb = -smoothing.saturation_partial * VdsatCV * dAbulkCV_dVb / AbulkCV;
            let [inversion, bulk, source] =
                channel_inversion(&Vgsteff, &VdseffCV, &AbulkCV, &CoxWL, model.xpart, 1e-20)?;
            let chain = ChargeChain {
                overdrive: [dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb],
                drain: [
                    dVdseffCV_dVg * dVgsteff_dVg,
                    dVdseffCV_dVg.mul_add(dVgsteff_dVd, dVdseffCV_dVd),
                    dVdseffCV_dVg.mul_add(dVgsteff_dVb, dVdseffCV_dVb),
                ],
                bulk: [0., 0., dAbulkCV_dVb],
                relative_oxide: [0.; 3],
            };
            let [ig, id, ib] = chain.apply(&inversion);
            [Cbg1, Cbd1, Cbb1] = chain.apply(&bulk);
            [Csg, Csd, Csb] = chain.apply(&source);
            Cgg1 = ig - Cbg1;
            Cgd1 = id - Cbd1;
            Cgb1 = ib - Cbb1;
            qgate = inversion.value - bulk.value;
            qbulk = bulk.value;
            qsrc = source.value;
            qgate += Qac0 + Qsub0;
            qbulk -= (Qac0 + Qsub0);
            qdrn = -(qgate + qbulk + qsrc);
            Cgg = dQac0_dVg + dQsub0_dVg + Cgg1;
            Cgd = dQsub0_dVd + Cgd1;
            Cgb = dQac0_dVb + dQsub0_dVb + Cgb1;
            Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
            Cbd = Cbd1 - dQsub0_dVd;
            Cbb = Cbb1 - dQac0_dVb - dQsub0_dVb;
            Cgb *= dVbseff_dVb;
            Cbb *= dVbseff_dVb;
            Csb *= dVbseff_dVb;
            here.cggb = Cgg;
            here.cgsb = -(Cgg + Cgd + Cgb);
            here.cgdb = Cgd;
            here.cdgb = -(Cgg + Cbg + Csg);
            here.cdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb + Csg + Csd + Csb);
            here.cddb = -(Cgd + Cbd + Csd);
            here.cbgb = Cbg;
            here.cbsb = -(Cbg + Cbd + Cbb);
            here.cbdb = Cbd;
        } else if model.capMod == 2 {
            V3 = here.vfbzb - Vgs_eff + VbseffCV - DELTA_3;
            let (flatband_shift, flatband_slope) = smooth_positive(V3, DELTA_3 * here.vfbzb.abs())?;
            T1 = flatband_slope;
            Vfbeff = here.vfbzb - flatband_shift;
            dVfbeff_dVg = T1 * dVgs_eff_dVg;
            dVfbeff_dVb = -T1 * dVbseffCV_dVb;
            Cox = here.coxp;
            Tox = 1.0e8 * here.toxp;
            T0 = (Vgs_eff - VbseffCV - here.vfbzb) / Tox;
            dT0_dVg = dVgs_eff_dVg / Tox;
            dT0_dVb = -dVbseffCV_dVb / Tox;
            tmp = T0 * pParam.acde;
            if (-EXP_THRESHOLD < tmp) && (tmp < EXP_THRESHOLD) {
                Tcen = pParam.ldeb * exp(tmp);
                dTcen_dVg = pParam.acde * Tcen;
                dTcen_dVb = dTcen_dVg * dT0_dVb;
                dTcen_dVg *= dT0_dVg;
            } else if tmp <= -EXP_THRESHOLD {
                Tcen = pParam.ldeb * MIN_EXP;
                dTcen_dVb = 0.0;
                dTcen_dVg = dTcen_dVb;
            } else {
                Tcen = pParam.ldeb * MAX_EXP;
                dTcen_dVb = 0.0;
                dTcen_dVg = dTcen_dVb;
            }
            LINK = 1.0e-3 * here.toxp;
            let centroid = smooth_drain(pParam.ldeb, Tcen, LINK)?;
            Tcen = centroid.effective;
            T1 = centroid.drain_partial;
            dTcen_dVg *= T1;
            dTcen_dVb *= T1;
            Ccen = epssub / Tcen;
            T2 = Cox / (Cox + Ccen);
            Coxeff = T2 * Ccen;
            T3 = -Ccen / Tcen;
            dCoxeff_dVg = T2 * T2 * T3;
            dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb;
            dCoxeff_dVg *= dTcen_dVg;
            CoxWLcen = CoxWL * Coxeff / model.coxe;
            Qac0 = -CoxWLcen * flatband_shift;
            QovCox = Qac0 / Coxeff;
            dQac0_dVg = CoxWLcen * dVfbeff_dVg + QovCox * dCoxeff_dVg;
            dQac0_dVb = CoxWLcen * dVfbeff_dVb + QovCox * dCoxeff_dVb;
            T3 = Vgs_eff - Vfbeff - VbseffCV - Vgsteff;
            let (charge, slope) = depletion_charge(pParam.k1ox, T3)?;
            Qsub0 = CoxWLcen * charge;
            T2 = CoxWLcen * slope;
            QovCox = Qsub0 / Coxeff;
            dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg) + QovCox * dCoxeff_dVg;
            dQsub0_dVd = -T2 * dVgsteff_dVd;
            dQsub0_dVb = -T2 * (dVfbeff_dVb + dVbseffCV_dVb + dVgsteff_dVb) + QovCox * dCoxeff_dVb;
            (VgDP, dVgDP_dVg) =
                charge_overdrive(Vgsteff, pParam.k1ox, pParam.moin, Vtm, pParam.sqrtPhi)?;
            Tox += Tox;
            T0 = (Vgsteff + here.vtfbphi2) / Tox;
            tmp = exp(model.bdos * 0.7 * log(T0));
            T1 = 1.0 + tmp;
            T2 = model.bdos * 0.7 * tmp / (T0 * Tox);
            Tcen = model.ados * 1.9e-9 / T1;
            dTcen_dVg = -Tcen * T2 / T1;
            dTcen_dVd = dTcen_dVg * dVgsteff_dVd;
            dTcen_dVb = dTcen_dVg * dVgsteff_dVb;
            dTcen_dVg *= dVgsteff_dVg;
            Ccen = epssub / Tcen;
            T0 = Cox / (Cox + Ccen);
            Coxeff = T0 * Ccen;
            T1 = -Ccen / Tcen;
            dCoxeff_dVg = T0 * T0 * T1;
            dCoxeff_dVd = dCoxeff_dVg * dTcen_dVd;
            dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb;
            dCoxeff_dVg *= dTcen_dVg;
            CoxWLcen = CoxWL * Coxeff / model.coxe;
            AbulkCV = Abulk0_Q * pParam.abulkCVfactor;
            dAbulkCV_dVb = pParam.abulkCVfactor * dAbulk0_Q_dVb;
            VdsatCV = VgDP / AbulkCV;
            let smoothing = smooth_drain(VdsatCV, Vds, DELTA_4)?;
            VdseffCV = smoothing.effective;
            dVdseffCV_dVg = smoothing.saturation_partial * (dVgDP_dVg / AbulkCV);
            dVdseffCV_dVd = smoothing.drain_partial;
            dVdseffCV_dVb = -smoothing.saturation_partial * VdsatCV * dAbulkCV_dVb / AbulkCV;
            let [inversion, bulk, source] =
                channel_inversion(&VgDP, &VdseffCV, &AbulkCV, &CoxWLcen, model.xpart, 1e-20)?;
            let chain = ChargeChain {
                overdrive: [dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb]
                    .map(|slope| slope * dVgDP_dVg),
                drain: [
                    dVdseffCV_dVg * dVgsteff_dVg,
                    dVdseffCV_dVg.mul_add(dVgsteff_dVd, dVdseffCV_dVd),
                    dVdseffCV_dVg.mul_add(dVgsteff_dVb, dVdseffCV_dVb),
                ],
                bulk: [0., 0., dAbulkCV_dVb],
                relative_oxide: [dCoxeff_dVg, dCoxeff_dVd, dCoxeff_dVb].map(|slope| slope / Coxeff),
            };
            [Cgg1, Cgd1, Cgb1] = chain.apply(&inversion);
            [Cbg1, Cbd1, Cbb1] = chain.apply(&bulk);
            [Csg, Csd, Csb] = chain.apply(&source);
            qgate = inversion.value;
            qbulk = bulk.value;
            qsrc = source.value;
            qgate += Qac0 + Qsub0 - qbulk;
            qbulk -= (Qac0 + Qsub0);
            qdrn = -(qgate + qbulk + qsrc);
            Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
            Cbd = Cbd1 - dQsub0_dVd;
            Cbb = Cbb1 - dQac0_dVb - dQsub0_dVb;
            Cgg = Cgg1 - Cbg;
            Cgd = Cgd1 - Cbd;
            Cgb = Cgb1 - Cbb;
            Cgb *= dVbseff_dVb;
            Cbb *= dVbseff_dVb;
            Csb *= dVbseff_dVb;
            here.cggb = Cgg;
            here.cgsb = -(Cgg + Cgd + Cgb);
            here.cgdb = Cgd;
            here.cdgb = -(Cgg + Cbg + Csg);
            here.cdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb + Csg + Csd + Csb);
            here.cddb = -(Cgd + Cbd + Csd);
            here.cbgb = Cbg;
            here.cbsb = -(Cbg + Cbd + Cbb);
            here.cbdb = Cbd;
        }
    }
    if !goto_finished {
        here.csgb = -here.cggb - here.cdgb - here.cbgb;
        here.csdb = -here.cgdb - here.cddb - here.cbdb;
        here.cssb = -here.cgsb - here.cdsb - here.cbsb;
        here.cgbb = -here.cgdb - here.cggb - here.cgsb;
        here.cdbb = -here.cddb - here.cdgb - here.cdsb;
        here.cbbb = -here.cbgb - here.cbdb - here.cbsb;
        here.csbb = -here.cgbb - here.cdbb - here.cbbb;
        here.qgate = qgate;
        here.qbulk = qbulk;
        here.qdrn = qdrn;
        here.qsrc = -(qgate + qbulk + qdrn);
        if ((here.trnqsMod) != 0) || ((here.acnqsMod) != 0) {
            qcheq = -(qbulk + qgate);
            here.qchqs = qcheq;
            here.cqgb = -(here.cggb + here.cbgb);
            here.cqdb = -(here.cgdb + here.cbdb);
            here.cqsb = -(here.cgsb + here.cbsb);
            here.cqbb = -(here.cqgb + here.cqdb + here.cqsb);
            CoxWL = model.coxe * pParam.weffCV * here.nf * pParam.leffCV;
            T1 = here.gcrg / CoxWL;
            here.gtau = T1 * ScalingFactor;
            if here.acnqsMod != 0 {
                here.taunet = 1.0 / T1;
            }
            ckt.set_state0(state::QCHEQ, qcheq);
            if (ckt.mode() & mode::MODEINITTRAN) != 0 {
                ckt.set_state1(state::QCHEQ, ckt.state0(state::QCHEQ));
            }
            if here.trnqsMod != 0 {
                (geq, ceq) = ckt.integrate(0.0, state::QCHEQ)?;
            }
        }
    }
    Ok(IntrinsicCharges { qbulk, qdrn, qgate })
}

/// Charge companion RHS current contributions.
pub(super) struct ChargeCurrents {
    pub(super) ceqqb: f64,
    pub(super) ceqqd: f64,
    pub(super) ceqqg: f64,
    pub(super) ceqqgmid: f64,
    pub(super) ceqqjd: f64,
    pub(super) ceqqjs: f64,
}

/// Terminal and internal-node companion conductances.
pub(super) struct CapacitanceStamp {
    pub(super) gcbbb: f64,
    pub(super) gcbdb: f64,
    pub(super) gcbgb: f64,
    pub(super) gcbgmb: f64,
    pub(super) gcbsb: f64,
    pub(super) gcdbb: f64,
    pub(super) gcdbdb: f64,
    pub(super) gcddb: f64,
    pub(super) gcdgb: f64,
    pub(super) gcdgmb: f64,
    pub(super) gcdsb: f64,
    pub(super) gcgbb: f64,
    pub(super) gcgdb: f64,
    pub(super) gcggb: f64,
    pub(super) gcgmbb: f64,
    pub(super) gcgmdb: f64,
    pub(super) gcgmgmb: f64,
    pub(super) gcgmsb: f64,
    pub(super) gcgsb: f64,
    pub(super) gcsbb: f64,
    pub(super) gcsbsb: f64,
    pub(super) gcsdb: f64,
    pub(super) gcsgb: f64,
    pub(super) gcsgmb: f64,
    pub(super) gcssb: f64,
}

/// Non-quasi-static charge-deficit conductances and partition derivatives.
pub(super) struct NqsStamp {
    pub(super) cqcheq: f64,
    pub(super) cqdef: f64,
    pub(super) ddxpart_dVb: f64,
    pub(super) ddxpart_dVd: f64,
    pub(super) ddxpart_dVg: f64,
    pub(super) ddxpart_dVs: f64,
    pub(super) dsxpart_dVb: f64,
    pub(super) dsxpart_dVd: f64,
    pub(super) dsxpart_dVg: f64,
    pub(super) dsxpart_dVs: f64,
    pub(super) dxpart: f64,
    pub(super) gcqbb: f64,
    pub(super) gcqdb: f64,
    pub(super) gcqgb: f64,
    pub(super) gcqsb: f64,
    pub(super) ggtb: f64,
    pub(super) ggtd: f64,
    pub(super) ggtg: f64,
    pub(super) ggts: f64,
    pub(super) gqdef: f64,
    pub(super) sxpart: f64,
}

/// Companion terms required for the final ordered stamp.
pub(super) struct Companion {
    pub(super) currents: ChargeCurrents,
    pub(super) capacitance: CapacitanceStamp,
    pub(super) nqs: NqsStamp,
}

/// Assemble overlap, junction and NQS companions; small-signal initialization returns no stamp.
pub(super) fn companion(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    ckt: &mut dyn LoadCkt,
    bias: &LoadBias,
    charges: &IntrinsicCharges,
) -> SpiceResult<Option<Companion>> {
    let &LoadBias {
        ChargeComputationNeeded,
        qdef,
        vbd,
        vbd_jct,
        vbs,
        vbs_jct,
        vgb,
        vgd,
        vgmb,
        vgmd,
        vgms,
        vgs,
        ..
    } = bias;
    let &IntrinsicCharges {
        mut qbulk,
        mut qdrn,
        mut qgate,
        ..
    } = charges;
    let mut gcgmgmb: f64 = 0.0;
    let mut gcgmdb: f64 = 0.0;
    let mut gcgmsb: f64 = 0.0;
    let mut gcdgmb: f64 = 0.0;
    let mut gcsgmb: f64 = 0.0;
    let mut gcgmbb: f64 = 0.0;
    let mut gcbgmb: f64 = 0.0;
    let mut qgmb: f64 = 0.0;
    let mut qgmid: f64 = 0.0;
    let mut ceqqgmid: f64 = 0.0;
    let mut ag0: f64 = 0.0;
    let mut qgb: f64 = 0.0;
    let mut ceqqb: f64 = 0.0;
    let mut ceqqd: f64 = 0.0;
    let mut ceqqg: f64 = 0.0;
    let mut ceqqjd: f64 = 0.0;
    let mut ceqqjs: f64 = 0.0;
    let mut ceq: f64 = 0.0;
    let mut geq: f64 = 0.0;
    let mut gcbdb: f64 = 0.0;
    let mut gcbgb: f64 = 0.0;
    let mut gcbsb: f64 = 0.0;
    let mut gcddb: f64 = 0.0;
    let mut gcdgb: f64 = 0.0;
    let mut gcdsb: f64 = 0.0;
    let mut gcgdb: f64 = 0.0;
    let mut gcggb: f64 = 0.0;
    let mut gcgsb: f64 = 0.0;
    let mut gcsdb: f64 = 0.0;
    let mut gcgbb: f64 = 0.0;
    let mut gcdbb: f64 = 0.0;
    let mut gcsbb: f64 = 0.0;
    let mut gcbbb: f64 = 0.0;
    let mut gcdbdb: f64 = 0.0;
    let mut gcsbsb: f64 = 0.0;
    let mut gcsgb: f64 = 0.0;
    let mut gcssb: f64 = 0.0;
    let mut qsrc: f64 = 0.0;
    let mut cqgate: f64 = 0.0;
    let mut cqbody: f64 = 0.0;
    let mut cqdrn: f64 = 0.0;
    let mut CoxWL: f64 = 0.0;
    let mut T0: f64 = 0.0;
    let mut T1: f64 = 0.0;
    let mut T2: f64 = 0.0;
    let mut T3: f64 = 0.0;
    let mut T4: f64 = 0.0;
    let mut qcheq: f64 = 0.0;
    let mut gqdef: f64 = 0.0;
    let mut cqdef: f64 = 0.0;
    let mut cqcheq: f64 = 0.0;
    let mut gcqdb: f64 = 0.0;
    let mut gcqsb: f64 = 0.0;
    let mut gcqgb: f64 = 0.0;
    let mut gcqbb: f64 = 0.0;
    let mut dxpart: f64 = 0.0;
    let mut sxpart: f64 = 0.0;
    let mut ggtg: f64 = 0.0;
    let mut ggtd: f64 = 0.0;
    let mut ggts: f64 = 0.0;
    let mut ggtb: f64 = 0.0;
    let mut ddxpart_dVd: f64 = 0.0;
    let mut ddxpart_dVg: f64 = 0.0;
    let mut ddxpart_dVb: f64 = 0.0;
    let mut ddxpart_dVs: f64 = 0.0;
    let mut dsxpart_dVd: f64 = 0.0;
    let mut dsxpart_dVg: f64 = 0.0;
    let mut dsxpart_dVb: f64 = 0.0;
    let mut dsxpart_dVs: f64 = 0.0;
    let mut qgdo: f64 = 0.0;
    let mut qgso: f64 = 0.0;
    let mut cgdo: f64 = 0.0;
    let mut cgso: f64 = 0.0;
    let mut Cdg: f64 = 0.0;
    let mut Cdd: f64 = 0.0;
    let mut Cds: f64 = 0.0;
    let mut Csg: f64 = 0.0;
    let mut Csd: f64 = 0.0;
    let mut Css: f64 = 0.0;
    let mut vgdx: f64 = 0.0;
    let mut vgsx: f64 = 0.0;
    let ScalingFactor = 1.0e-9;
    // C's `goto line850` and `goto line1000` out of the charge block are
    // breaks out of 'charges with these flags; leaving the block normally
    // is C's `goto line860`. (C's bypass `goto line755`/`line860` cannot
    // happen: the port never bypasses, CKTbypass = 0.)
    let mut goto_line850 = false;
    'charges: {
        if ChargeComputationNeeded == 0 {
            goto_line850 = true;
            break 'charges;
        }
        if here.rgateMod == 3 {
            vgdx = vgmd;
            vgsx = vgms;
        } else {
            vgdx = vgd;
            vgsx = vgs;
        }
        if model.capMod == 0 {
            cgdo = pParam.cgdo;
            qgdo = pParam.cgdo * vgdx;
            cgso = pParam.cgso;
            qgso = pParam.cgso * vgsx;
        } else {
            (cgdo, qgdo) = crate::devices::bsim_physics::overlap(
                vgdx,
                pParam.cgdo,
                pParam.weffCV * pParam.cgdl,
                pParam.ckappad,
            )?;
            (cgso, qgso) = crate::devices::bsim_physics::overlap(
                vgsx,
                pParam.cgso,
                pParam.weffCV * pParam.cgsl,
                pParam.ckappas,
            )?;
        }
        if here.nf != 1.0 {
            cgdo *= here.nf;
            cgso *= here.nf;
            qgdo *= here.nf;
            qgso *= here.nf;
        }
        here.cgdo = cgdo;
        here.qgdo = qgdo;
        here.cgso = cgso;
        here.qgso = qgso;
        ag0 = ckt.ag(0);
        if here.mode > 0 {
            if here.trnqsMod == 0 {
                qdrn -= qgdo;
                if here.rgateMod == 3 {
                    gcgmgmb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgmdb = -cgdo * ag0;
                    gcgmsb = -cgso * ag0;
                    gcgmbb = -pParam.cgbo * ag0;
                    gcdgmb = gcgmdb;
                    gcsgmb = gcgmsb;
                    gcbgmb = gcgmbb;
                    gcggb = here.cggb * ag0;
                    gcgdb = here.cgdb * ag0;
                    gcgsb = here.cgsb * ag0;
                    gcgbb = -(gcggb + gcgdb + gcgsb);
                    gcdgb = here.cdgb * ag0;
                    gcsgb = -(here.cggb + here.cbgb + here.cdgb) * ag0;
                    gcbgb = here.cbgb * ag0;
                    qgmb = pParam.cgbo * vgmb;
                    qgmid = qgdo + qgso + qgmb;
                    qbulk -= qgmb;
                    qsrc = -(qgate + qgmid + qbulk + qdrn);
                } else {
                    gcggb = (here.cggb + cgdo + cgso + pParam.cgbo) * ag0;
                    gcgdb = (here.cgdb - cgdo) * ag0;
                    gcgsb = (here.cgsb - cgso) * ag0;
                    gcgbb = -(gcggb + gcgdb + gcgsb);
                    gcdgb = (here.cdgb - cgdo) * ag0;
                    gcsgb = -(here.cggb + here.cbgb + here.cdgb + cgso) * ag0;
                    gcbgb = (here.cbgb - pParam.cgbo) * ag0;
                    gcbgmb = 0.0;
                    gcsgmb = gcbgmb;
                    gcdgmb = gcsgmb;
                    qgb = pParam.cgbo * vgb;
                    qgate += qgdo + qgso + qgb;
                    qbulk -= qgb;
                    qsrc = -(qgate + qbulk + qdrn);
                }
                gcddb = (here.cddb + here.capbd + cgdo) * ag0;
                gcdsb = here.cdsb * ag0;
                gcsdb = -(here.cgdb + here.cbdb + here.cddb) * ag0;
                gcssb = (here.capbs + cgso - (here.cgsb + here.cbsb + here.cdsb)) * ag0;
                if here.rbodyMod == 0 {
                    gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb);
                    gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb);
                    gcbdb = (here.cbdb - here.capbd) * ag0;
                    gcbsb = (here.cbsb - here.capbs) * ag0;
                    gcdbdb = 0.0;
                    gcsbsb = 0.0;
                } else {
                    gcdbb = -(here.cddb + here.cdgb + here.cdsb) * ag0;
                    gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb) + here.capbs * ag0;
                    gcbdb = here.cbdb * ag0;
                    gcbsb = here.cbsb * ag0;
                    gcdbdb = -here.capbd * ag0;
                    gcsbsb = -here.capbs * ag0;
                }
                gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb);
                ggts = 0.0;
                ggtb = ggts;
                ggtd = ggtb;
                ggtg = ggtd;
                sxpart = 0.6;
                dxpart = 0.4;
                ddxpart_dVs = 0.0;
                ddxpart_dVb = ddxpart_dVs;
                ddxpart_dVg = ddxpart_dVb;
                ddxpart_dVd = ddxpart_dVg;
                dsxpart_dVs = 0.0;
                dsxpart_dVb = dsxpart_dVs;
                dsxpart_dVg = dsxpart_dVb;
                dsxpart_dVd = dsxpart_dVg;
            } else {
                qcheq = here.qchqs;
                CoxWL = model.coxe * pParam.weffCV * here.nf * pParam.leffCV;
                T0 = qdef * ScalingFactor / CoxWL;
                here.gtg = T0 * here.gcrgg;
                ggtg = here.gtg;
                here.gtd = T0 * here.gcrgd;
                ggtd = here.gtd;
                here.gts = T0 * here.gcrgs;
                ggts = here.gts;
                here.gtb = T0 * here.gcrgb;
                ggtb = here.gtb;
                gqdef = ScalingFactor * ag0;
                gcqgb = here.cqgb * ag0;
                gcqdb = here.cqdb * ag0;
                gcqsb = here.cqsb * ag0;
                gcqbb = here.cqbb * ag0;
                if fabs(qcheq) <= 1.0e-5 * CoxWL {
                    if model.xpart < 0.5 {
                        dxpart = 0.4;
                    } else if model.xpart > 0.5 {
                        dxpart = 0.0;
                    } else {
                        dxpart = 0.5;
                    }
                    ddxpart_dVs = 0.0;
                    ddxpart_dVb = ddxpart_dVs;
                    ddxpart_dVg = ddxpart_dVb;
                    ddxpart_dVd = ddxpart_dVg;
                } else {
                    dxpart = qdrn / qcheq;
                    Cdd = here.cddb;
                    Csd = -(here.cgdb + here.cddb + here.cbdb);
                    ddxpart_dVd = dxpart.mul_add(-(Cdd + Csd), Cdd) / qcheq;
                    Cdg = here.cdgb;
                    Csg = -(here.cggb + here.cdgb + here.cbgb);
                    ddxpart_dVg = dxpart.mul_add(-(Cdg + Csg), Cdg) / qcheq;
                    Cds = here.cdsb;
                    Css = -(here.cgsb + here.cdsb + here.cbsb);
                    ddxpart_dVs = dxpart.mul_add(-(Cds + Css), Cds) / qcheq;
                    ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs);
                }
                sxpart = 1.0 - dxpart;
                dsxpart_dVd = -ddxpart_dVd;
                dsxpart_dVg = -ddxpart_dVg;
                dsxpart_dVs = -ddxpart_dVs;
                dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs);
                if here.rgateMod == 3 {
                    gcgmgmb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgmdb = -cgdo * ag0;
                    gcgmsb = -cgso * ag0;
                    gcgmbb = -pParam.cgbo * ag0;
                    gcdgmb = gcgmdb;
                    gcsgmb = gcgmsb;
                    gcbgmb = gcgmbb;
                    gcbgb = 0.0;
                    gcsgb = gcbgb;
                    gcdgb = gcsgb;
                    gcgbb = 0.0;
                    gcgsb = gcgbb;
                    gcgdb = gcgsb;
                    gcggb = gcgdb;
                    qgmb = pParam.cgbo * vgmb;
                    qgmid = qgdo + qgso + qgmb;
                    qgate = 0.0;
                    qbulk = -qgmb;
                    qdrn = -qgdo;
                    qsrc = -(qgmid + qbulk + qdrn);
                } else {
                    gcggb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgdb = -cgdo * ag0;
                    gcgsb = -cgso * ag0;
                    gcgbb = -pParam.cgbo * ag0;
                    gcdgb = gcgdb;
                    gcsgb = gcgsb;
                    gcbgb = gcgbb;
                    gcbgmb = 0.0;
                    gcsgmb = gcbgmb;
                    gcdgmb = gcsgmb;
                    qgb = pParam.cgbo * vgb;
                    qgate = qgdo + qgso + qgb;
                    qbulk = -qgb;
                    qdrn = -qgdo;
                    qsrc = -(qgate + qbulk + qdrn);
                }
                gcddb = (here.capbd + cgdo) * ag0;
                gcsdb = 0.0;
                gcdsb = gcsdb;
                gcssb = (here.capbs + cgso) * ag0;
                if here.rbodyMod == 0 {
                    gcdbb = -(gcdgb + gcddb + gcdgmb);
                    gcsbb = -(gcsgb + gcssb + gcsgmb);
                    gcbdb = -here.capbd * ag0;
                    gcbsb = -here.capbs * ag0;
                    gcdbdb = 0.0;
                    gcsbsb = 0.0;
                } else {
                    gcbsb = 0.0;
                    gcbdb = gcbsb;
                    gcsbb = gcbdb;
                    gcdbb = gcsbb;
                    gcdbdb = -here.capbd * ag0;
                    gcsbsb = -here.capbs * ag0;
                }
                gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb);
            }
        } else {
            if here.trnqsMod == 0 {
                qsrc = qdrn - qgso;
                if here.rgateMod == 3 {
                    gcgmgmb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgmdb = -cgdo * ag0;
                    gcgmsb = -cgso * ag0;
                    gcgmbb = -pParam.cgbo * ag0;
                    gcdgmb = gcgmdb;
                    gcsgmb = gcgmsb;
                    gcbgmb = gcgmbb;
                    gcggb = here.cggb * ag0;
                    gcgdb = here.cgsb * ag0;
                    gcgsb = here.cgdb * ag0;
                    gcgbb = -(gcggb + gcgdb + gcgsb);
                    gcdgb = -(here.cggb + here.cbgb + here.cdgb) * ag0;
                    gcsgb = here.cdgb * ag0;
                    gcbgb = here.cbgb * ag0;
                    qgmb = pParam.cgbo * vgmb;
                    qgmid = qgdo + qgso + qgmb;
                    qbulk -= qgmb;
                    qdrn = -(qgate + qgmid + qbulk + qsrc);
                } else {
                    gcggb = (here.cggb + cgdo + cgso + pParam.cgbo) * ag0;
                    gcgdb = (here.cgsb - cgdo) * ag0;
                    gcgsb = (here.cgdb - cgso) * ag0;
                    gcgbb = -(gcggb + gcgdb + gcgsb);
                    gcdgb = -(here.cggb + here.cbgb + here.cdgb + cgdo) * ag0;
                    gcsgb = (here.cdgb - cgso) * ag0;
                    gcbgb = (here.cbgb - pParam.cgbo) * ag0;
                    gcbgmb = 0.0;
                    gcsgmb = gcbgmb;
                    gcdgmb = gcsgmb;
                    qgb = pParam.cgbo * vgb;
                    qgate += qgdo + qgso + qgb;
                    qbulk -= qgb;
                    qdrn = -(qgate + qbulk + qsrc);
                }
                gcddb = (here.capbd + cgdo - (here.cgsb + here.cbsb + here.cdsb)) * ag0;
                gcdsb = -(here.cgdb + here.cbdb + here.cddb) * ag0;
                gcsdb = here.cdsb * ag0;
                gcssb = (here.cddb + here.capbs + cgso) * ag0;
                if here.rbodyMod == 0 {
                    gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb);
                    gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb);
                    gcbdb = (here.cbsb - here.capbd) * ag0;
                    gcbsb = (here.cbdb - here.capbs) * ag0;
                    gcdbdb = 0.0;
                    gcsbsb = 0.0;
                } else {
                    gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb) + here.capbd * ag0;
                    gcsbb = -(here.cddb + here.cdgb + here.cdsb) * ag0;
                    gcbdb = here.cbsb * ag0;
                    gcbsb = here.cbdb * ag0;
                    gcdbdb = -here.capbd * ag0;
                    gcsbsb = -here.capbs * ag0;
                }
                gcbbb = -(gcbgb + gcbdb + gcbsb + gcbgmb);
                ggts = 0.0;
                ggtb = ggts;
                ggtd = ggtb;
                ggtg = ggtd;
                sxpart = 0.4;
                dxpart = 0.6;
                ddxpart_dVs = 0.0;
                ddxpart_dVb = ddxpart_dVs;
                ddxpart_dVg = ddxpart_dVb;
                ddxpart_dVd = ddxpart_dVg;
                dsxpart_dVs = 0.0;
                dsxpart_dVb = dsxpart_dVs;
                dsxpart_dVg = dsxpart_dVb;
                dsxpart_dVd = dsxpart_dVg;
            } else {
                qcheq = here.qchqs;
                CoxWL = model.coxe * pParam.weffCV * here.nf * pParam.leffCV;
                T0 = qdef * ScalingFactor / CoxWL;
                here.gtg = T0 * here.gcrgg;
                ggtg = here.gtg;
                here.gts = T0 * here.gcrgd;
                ggts = here.gts;
                here.gtd = T0 * here.gcrgs;
                ggtd = here.gtd;
                here.gtb = T0 * here.gcrgb;
                ggtb = here.gtb;
                gqdef = ScalingFactor * ag0;
                gcqgb = here.cqgb * ag0;
                gcqdb = here.cqsb * ag0;
                gcqsb = here.cqdb * ag0;
                gcqbb = here.cqbb * ag0;
                if fabs(qcheq) <= 1.0e-5 * CoxWL {
                    if model.xpart < 0.5 {
                        sxpart = 0.4;
                    } else if model.xpart > 0.5 {
                        sxpart = 0.0;
                    } else {
                        sxpart = 0.5;
                    }
                    dsxpart_dVs = 0.0;
                    dsxpart_dVb = dsxpart_dVs;
                    dsxpart_dVg = dsxpart_dVb;
                    dsxpart_dVd = dsxpart_dVg;
                } else {
                    sxpart = qdrn / qcheq;
                    Css = here.cddb;
                    Cds = -(here.cgdb + here.cddb + here.cbdb);
                    dsxpart_dVs = sxpart.mul_add(-(Css + Cds), Css) / qcheq;
                    Csg = here.cdgb;
                    Cdg = -(here.cggb + here.cdgb + here.cbgb);
                    dsxpart_dVg = sxpart.mul_add(-(Csg + Cdg), Csg) / qcheq;
                    Csd = here.cdsb;
                    Cdd = -(here.cgsb + here.cdsb + here.cbsb);
                    dsxpart_dVd = sxpart.mul_add(-(Csd + Cdd), Csd) / qcheq;
                    dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs);
                }
                dxpart = 1.0 - sxpart;
                ddxpart_dVd = -dsxpart_dVd;
                ddxpart_dVg = -dsxpart_dVg;
                ddxpart_dVs = -dsxpart_dVs;
                ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs);
                if here.rgateMod == 3 {
                    gcgmgmb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgmdb = -cgdo * ag0;
                    gcgmsb = -cgso * ag0;
                    gcgmbb = -pParam.cgbo * ag0;
                    gcdgmb = gcgmdb;
                    gcsgmb = gcgmsb;
                    gcbgmb = gcgmbb;
                    gcbgb = 0.0;
                    gcsgb = gcbgb;
                    gcdgb = gcsgb;
                    gcgbb = 0.0;
                    gcgsb = gcgbb;
                    gcgdb = gcgsb;
                    gcggb = gcgdb;
                    qgmb = pParam.cgbo * vgmb;
                    qgmid = qgdo + qgso + qgmb;
                    qgate = 0.0;
                    qbulk = -qgmb;
                } else {
                    gcggb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgdb = -cgdo * ag0;
                    gcgsb = -cgso * ag0;
                    gcgbb = -pParam.cgbo * ag0;
                    gcdgb = gcgdb;
                    gcsgb = gcgsb;
                    gcbgb = gcgbb;
                    gcbgmb = 0.0;
                    gcsgmb = gcbgmb;
                    gcdgmb = gcsgmb;
                    qgb = pParam.cgbo * vgb;
                    qgate = qgdo + qgso + qgb;
                    qbulk = -qgb;
                }
                qdrn = -qgdo;
                qsrc = -qgso;
                gcddb = (here.capbd + cgdo) * ag0;
                gcsdb = 0.0;
                gcdsb = gcsdb;
                gcssb = (here.capbs + cgso) * ag0;
                if here.rbodyMod == 0 {
                    gcdbb = -(gcdgb + gcddb + gcdgmb);
                    gcsbb = -(gcsgb + gcssb + gcsgmb);
                    gcbdb = -here.capbd * ag0;
                    gcbsb = -here.capbs * ag0;
                    gcdbdb = 0.0;
                    gcsbsb = 0.0;
                } else {
                    gcbsb = 0.0;
                    gcbdb = gcbsb;
                    gcsbb = gcbdb;
                    gcdbb = gcsbb;
                    gcdbdb = -here.capbd * ag0;
                    gcsbsb = -here.capbs * ag0;
                }
                gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb);
            }
        }
        if here.trnqsMod != 0 {
            ckt.set_state0(state::QCDUMP, qdef * ScalingFactor);
            if (ckt.mode() & mode::MODEINITTRAN) != 0 {
                ckt.set_state1(state::QCDUMP, ckt.state0(state::QCDUMP));
            }
            (geq, ceq) = ckt.integrate(0.0, state::QCDUMP)?;
        }
        ckt.set_state0(state::QG, qgate);
        ckt.set_state0(state::QD, qdrn - ckt.state0(state::QBD));
        ckt.set_state0(state::QS, qsrc - ckt.state0(state::QBS));
        if here.rgateMod == 3 {
            ckt.set_state0(state::QGMID, qgmid);
        }
        if here.rbodyMod == 0 {
            ckt.set_state0(
                state::QB,
                qbulk + ckt.state0(state::QBD) + ckt.state0(state::QBS),
            );
        } else {
            ckt.set_state0(state::QB, qbulk);
        }
        if (ckt.mode() & mode::MODEINITSMSIG) != 0 {
            return Ok(None);
        }
        if ChargeComputationNeeded == 0 {
            goto_line850 = true;
            break 'charges;
        }
        if (ckt.mode() & mode::MODEDCTRANCURVE) != 0 {
            goto_line850 = true;
            break 'charges;
        }
        if (ckt.mode() & mode::MODEINITTRAN) != 0 {
            ckt.set_state1(state::QB, ckt.state0(state::QB));
            ckt.set_state1(state::QG, ckt.state0(state::QG));
            ckt.set_state1(state::QD, ckt.state0(state::QD));
            if here.rgateMod == 3 {
                ckt.set_state1(state::QGMID, ckt.state0(state::QGMID));
            }
            if here.rbodyMod != 0 {
                ckt.set_state1(state::QBS, ckt.state0(state::QBS));
                ckt.set_state1(state::QBD, ckt.state0(state::QBD));
            }
        }
        (geq, ceq) = ckt.integrate(0.0, state::QB)?;
        (geq, ceq) = ckt.integrate(0.0, state::QG)?;
        (geq, ceq) = ckt.integrate(0.0, state::QD)?;
        if here.rgateMod == 3 {
            (geq, ceq) = ckt.integrate(0.0, state::QGMID)?;
        }
        if here.rbodyMod != 0 {
            (geq, ceq) = ckt.integrate(0.0, state::QBS)?;
            (geq, ceq) = ckt.integrate(0.0, state::QBD)?;
        }
    }
    if goto_line850 {
        // C label `line850:`.
        ceqqd = 0.0;
        ceqqb = ceqqd;
        ceqqg = ceqqb;
        ceqqjs = 0.0;
        ceqqjd = ceqqjs;
        cqdef = 0.0;
        cqcheq = cqdef;
        gcdbb = 0.0;
        gcdsb = gcdbb;
        gcddb = gcdsb;
        gcdgb = gcddb;
        gcsbb = 0.0;
        gcssb = gcsbb;
        gcsdb = gcssb;
        gcsgb = gcsdb;
        gcgbb = 0.0;
        gcgsb = gcgbb;
        gcgdb = gcgsb;
        gcggb = gcgdb;
        gcbbb = 0.0;
        gcbsb = gcbbb;
        gcbgb = gcbsb;
        gcbdb = gcbgb;
        gcgmbb = 0.0;
        gcgmsb = gcgmbb;
        gcgmdb = gcgmsb;
        gcgmgmb = gcgmdb;
        ceqqgmid = 0.0;
        gcbgmb = ceqqgmid;
        gcsgmb = gcbgmb;
        gcdgmb = gcsgmb;
        gcsbsb = 0.0;
        gcdbdb = gcsbsb;
        gcqbb = 0.0;
        gcqsb = gcqbb;
        gcqdb = gcqsb;
        gcqgb = gcqdb;
        gqdef = gcqgb;
        ggts = 0.0;
        ggtb = ggts;
        ggtd = ggtb;
        ggtg = ggtd;
        dxpart = (if (here.mode > 0) { 0.4 } else { 0.6 });
        sxpart = (1.0 - (dxpart));
        ddxpart_dVs = 0.0;
        ddxpart_dVb = ddxpart_dVs;
        ddxpart_dVg = ddxpart_dVb;
        ddxpart_dVd = ddxpart_dVg;
        dsxpart_dVs = 0.0;
        dsxpart_dVb = dsxpart_dVs;
        dsxpart_dVg = dsxpart_dVb;
        dsxpart_dVd = dsxpart_dVg;
        if here.trnqsMod != 0 {
            CoxWL = model.coxe * pParam.weffCV * here.nf * pParam.leffCV;
            T1 = here.gcrg / CoxWL;
            here.gtau = T1 * ScalingFactor;
        } else {
            here.gtau = 0.0;
        }
    } else {
        // C label `line860:`.
        cqgate = ckt.state0(state::CQG);
        cqbody = ckt.state0(state::CQB);
        cqdrn = ckt.state0(state::CQD);
        ceqqg = cqgate - gcggb * vgb + gcgdb * vbd + gcgsb * vbs;
        ceqqd = cqdrn - gcdgb * vgb - gcdgmb * vgmb + (gcddb + gcdbdb) * vbd - gcdbdb * vbd_jct
            + gcdsb * vbs;
        ceqqb = cqbody - gcbgb * vgb - gcbgmb * vgmb + gcbdb * vbd + gcbsb * vbs;
        if here.rgateMod == 3 {
            ceqqgmid = ckt.state0(state::CQGMID) + gcgmdb * vbd + gcgmsb * vbs - gcgmgmb * vgmb;
        } else {
            ceqqgmid = 0.0;
        }
        if here.rbodyMod != 0 {
            ceqqjs = ckt.state0(state::CQBS) + gcsbsb * vbs_jct;
            ceqqjd = ckt.state0(state::CQBD) + gcdbdb * vbd_jct;
        }
        if here.trnqsMod != 0 {
            T0 = ggtg * vgb - ggtd * vbd - ggts * vbs;
            ceqqg += T0;
            T1 = qdef * here.gtau;
            ceqqd -= dxpart * T0 + T1 * (ddxpart_dVg * vgb - ddxpart_dVd * vbd - ddxpart_dVs * vbs);
            cqdef = ckt.state0(state::CQCDUMP) - gqdef * qdef;
            cqcheq = ckt.state0(state::CQCHEQ) - (gcqgb * vgb - gcqdb * vbd - gcqsb * vbs) + T0;
        }
        if (ckt.mode() & mode::MODEINITTRAN) != 0 {
            ckt.set_state1(state::CQB, ckt.state0(state::CQB));
            ckt.set_state1(state::CQG, ckt.state0(state::CQG));
            ckt.set_state1(state::CQD, ckt.state0(state::CQD));
            if here.rgateMod == 3 {
                ckt.set_state1(state::CQGMID, ckt.state0(state::CQGMID));
            }
            if here.rbodyMod != 0 {
                ckt.set_state1(state::CQBS, ckt.state0(state::CQBS));
                ckt.set_state1(state::CQBD, ckt.state0(state::CQBD));
            }
        }
    }
    Ok(Some(Companion {
        currents: ChargeCurrents {
            ceqqb,
            ceqqd,
            ceqqg,
            ceqqgmid,
            ceqqjd,
            ceqqjs,
        },
        capacitance: CapacitanceStamp {
            gcbbb,
            gcbdb,
            gcbgb,
            gcbgmb,
            gcbsb,
            gcdbb,
            gcdbdb,
            gcddb,
            gcdgb,
            gcdgmb,
            gcdsb,
            gcgbb,
            gcgdb,
            gcggb,
            gcgmbb,
            gcgmdb,
            gcgmgmb,
            gcgmsb,
            gcgsb,
            gcsbb,
            gcsbsb,
            gcsdb,
            gcsgb,
            gcsgmb,
            gcssb,
        },
        nqs: NqsStamp {
            cqcheq,
            cqdef,
            ddxpart_dVb,
            ddxpart_dVd,
            ddxpart_dVg,
            ddxpart_dVs,
            dsxpart_dVb,
            dsxpart_dVd,
            dsxpart_dVg,
            dsxpart_dVs,
            dxpart,
            gcqbb,
            gcqdb,
            gcqgb,
            gcqsb,
            ggtb,
            ggtd,
            ggtg,
            ggts,
            gqdef,
            sxpart,
        },
    }))
}
