//! C's initial mode selection, predictor and junction voltage limiting.
//! No prepared or live instance fields are modified by this stage.
//! The host history writes remain in their original order.
#![expect(
    non_snake_case,
    reason = "C voltage names identify the reference state slots"
)]
#![expect(
    unused_assignments,
    reason = "retain original intermediate assignment sequence during mode selection"
)]
#![expect(
    clippy::similar_names,
    reason = "C source and drain voltage symbols retain their distinctions"
)]
#![expect(
    clippy::too_many_lines,
    reason = "initial selection, history writes and limiting retain C's order"
)]
#![expect(unused_parens, reason = "retain C's grouping during translation")]
use super::{CONST_VT0, Instance, LoadCkt, Model, mode, state};
use crate::devices::limiting::{fetlim, limvds, pnjlim};

/// Extrapolate the difference between accepted states. A constant history
/// remains the same value even for a very large step-size ratio.
fn predict(ratio: f64, recent: f64, older: f64) -> f64 {
    ratio.mul_add(recent - older, recent)
}

/// The selected and limited terminal voltages (polarity normalized), plus
/// the reference charge-computation and convergence decisions.
pub(super) struct LoadBias {
    pub(super) ChargeComputationNeeded: i32,
    pub(super) Check: i32,
    pub(super) qdef: f64,
    pub(super) vbd: f64,
    pub(super) vbd_jct: f64,
    pub(super) vbs: f64,
    pub(super) vbs_jct: f64,
    pub(super) vdbd: f64,
    pub(super) vdbs: f64,
    pub(super) vdes: f64,
    pub(super) vds: f64,
    pub(super) vgb: f64,
    pub(super) vgd: f64,
    pub(super) vges: f64,
    pub(super) vgmb: f64,
    pub(super) vgmd: f64,
    pub(super) vgms: f64,
    pub(super) vgs: f64,
    pub(super) vsbs: f64,
    pub(super) vses: f64,
}

/// Select C's prior state or trial voltage, then limit junctions where enabled.
pub(super) fn select(model: &Model, here: &Instance, ckt: &mut dyn LoadCkt) -> LoadBias {
    let mut vses: f64 = 0.0;
    let mut vdes: f64 = 0.0;
    let mut vges: f64 = 0.0;
    let mut vgms: f64 = 0.0;
    let mut vgedo: f64 = 0.0;
    let mut vgmdo: f64 = 0.0;
    let mut vged: f64 = 0.0;
    let mut vgmd: f64 = 0.0;
    let mut vgmb: f64 = 0.0;
    let mut vbd: f64 = 0.0;
    let mut vbs: f64 = 0.0;
    let mut vds: f64 = 0.0;
    let mut vgb: f64 = 0.0;
    let mut vgd: f64 = 0.0;
    let mut vgs: f64 = 0.0;
    let mut vgdo: f64 = 0.0;
    let mut xfact: f64 = 0.0;
    let mut vdbs: f64 = 0.0;
    let mut vdbd: f64 = 0.0;
    let mut vsbs: f64 = 0.0;
    let mut vsbdo: f64 = 0.0;
    let mut vsbd: f64 = 0.0;
    let mut vbs_jct: f64 = 0.0;
    let mut vbd_jct: f64 = 0.0;
    let mut von: f64 = 0.0;
    let mut qdef: f64 = 0.0;
    let mut ChargeComputationNeeded: i32 = 0;
    let mut Check: i32 = 0;
    let mut Check1: i32 = 0;
    let mut Check2: i32 = 0;
    // No integration if DC sweep, but keep evaluating capacitances.
    ChargeComputationNeeded = i32::from(
        (ckt.mode()
            & (mode::MODEDCTRANCURVE | mode::MODEAC | mode::MODETRAN | mode::MODEINITSMSIG))
            != 0
            || ((ckt.mode() & mode::MODETRANOP) != 0 && (ckt.mode() & mode::MODEUIC) != 0),
    );

    Check2 = 1;
    Check1 = Check2;
    Check = Check1;
    if (ckt.mode() & mode::MODEINITSMSIG) != 0 {
        vds = ckt.state0(state::VDS);
        vgs = ckt.state0(state::VGS);
        vbs = ckt.state0(state::VBS);
        vges = ckt.state0(state::VGES);
        vgms = ckt.state0(state::VGMS);
        vdbs = ckt.state0(state::VDBS);
        vsbs = ckt.state0(state::VSBS);
        vses = ckt.state0(state::VSES);
        vdes = ckt.state0(state::VDES);
        qdef = ckt.state0(state::QDEF);
    } else if (ckt.mode() & mode::MODEINITTRAN) != 0 {
        vds = ckt.state1(state::VDS);
        vgs = ckt.state1(state::VGS);
        vbs = ckt.state1(state::VBS);
        vges = ckt.state1(state::VGES);
        vgms = ckt.state1(state::VGMS);
        vdbs = ckt.state1(state::VDBS);
        vsbs = ckt.state1(state::VSBS);
        vses = ckt.state1(state::VSES);
        vdes = ckt.state1(state::VDES);
        qdef = ckt.state1(state::QDEF);
    } else if ((ckt.mode() & mode::MODEINITJCT) != 0) && (here.off == 0) {
        vds = f64::from(model.r#type) * here.icVDS;
        vgms = f64::from(model.r#type) * here.icVGS;
        vges = vgms;
        vgs = vges;
        vsbs = f64::from(model.r#type) * here.icVBS;
        vdbs = vsbs;
        vbs = vdbs;
        if vds > 0.0 {
            vdes = vds + 0.01;
            vses = -0.01;
        } else if vds < 0.0 {
            vdes = vds - 0.01;
            vses = 0.01;
        } else {
            vses = 0.0;
            vdes = vses;
        }
        qdef = 0.0;
        if (vds == 0.0)
            && (vgs == 0.0)
            && (vbs == 0.0)
            && (((ckt.mode()
                & (((mode::MODETRAN | mode::MODEAC) | mode::MODEDCOP) | mode::MODEDCTRANCURVE))
                != 0)
                || ((ckt.mode() & mode::MODEUIC) == 0))
        {
            vds = 0.1;
            vdes = 0.11;
            vses = -0.01;
            vgms = f64::mul_add(f64::from(model.r#type), here.vth0, 0.1);
            vges = vgms;
            vgs = vges;
            vsbs = 0.0;
            vdbs = vsbs;
            vbs = vdbs;
        }
    } else if ((ckt.mode() & (mode::MODEINITJCT | mode::MODEINITFIX)) != 0) && ((here.off) != 0) {
        vgms = 0.0;
        vges = vgms;
        vbs = vges;
        vgs = vbs;
        vds = vgs;
        qdef = 0.0;
        vses = qdef;
        vdes = vses;
        vsbs = vdes;
        vdbs = vsbs;
    } else {
        if (ckt.mode() & mode::MODEINITPRED) != 0 {
            xfact = ckt.delta() / ckt.delta_old(1);
            ckt.set_state0(state::VDS, ckt.state1(state::VDS));
            vds = predict(xfact, ckt.state1(state::VDS), ckt.state2(state::VDS));
            ckt.set_state0(state::VGS, ckt.state1(state::VGS));
            vgs = predict(xfact, ckt.state1(state::VGS), ckt.state2(state::VGS));
            ckt.set_state0(state::VGES, ckt.state1(state::VGES));
            vges = predict(xfact, ckt.state1(state::VGES), ckt.state2(state::VGES));
            ckt.set_state0(state::VGMS, ckt.state1(state::VGMS));
            vgms = predict(xfact, ckt.state1(state::VGMS), ckt.state2(state::VGMS));
            ckt.set_state0(state::VBS, ckt.state1(state::VBS));
            vbs = predict(xfact, ckt.state1(state::VBS), ckt.state2(state::VBS));
            ckt.set_state0(state::VBD, ckt.state0(state::VBS) - ckt.state0(state::VDS));
            ckt.set_state0(state::VDBS, ckt.state1(state::VDBS));
            vdbs = predict(xfact, ckt.state1(state::VDBS), ckt.state2(state::VDBS));
            ckt.set_state0(
                state::VDBD,
                ckt.state0(state::VDBS) - ckt.state0(state::VDS),
            );
            ckt.set_state0(state::VSBS, ckt.state1(state::VSBS));
            vsbs = predict(xfact, ckt.state1(state::VSBS), ckt.state2(state::VSBS));
            ckt.set_state0(state::VSES, ckt.state1(state::VSES));
            vses = predict(xfact, ckt.state1(state::VSES), ckt.state2(state::VSES));
            ckt.set_state0(state::VDES, ckt.state1(state::VDES));
            vdes = predict(xfact, ckt.state1(state::VDES), ckt.state2(state::VDES));
            ckt.set_state0(state::QDEF, ckt.state1(state::QDEF));
            qdef = predict(xfact, ckt.state1(state::QDEF), ckt.state2(state::QDEF));
        } else {
            vds = f64::from(model.r#type)
                * (ckt.rhs_old(here.dNodePrime) - ckt.rhs_old(here.sNodePrime));
            vgs = f64::from(model.r#type)
                * (ckt.rhs_old(here.gNodePrime) - ckt.rhs_old(here.sNodePrime));
            vbs = f64::from(model.r#type)
                * (ckt.rhs_old(here.bNodePrime) - ckt.rhs_old(here.sNodePrime));
            vges = f64::from(model.r#type)
                * (ckt.rhs_old(here.gNodeExt) - ckt.rhs_old(here.sNodePrime));
            vgms = f64::from(model.r#type)
                * (ckt.rhs_old(here.gNodeMid) - ckt.rhs_old(here.sNodePrime));
            vdbs =
                f64::from(model.r#type) * (ckt.rhs_old(here.dbNode) - ckt.rhs_old(here.sNodePrime));
            vsbs =
                f64::from(model.r#type) * (ckt.rhs_old(here.sbNode) - ckt.rhs_old(here.sNodePrime));
            vses =
                f64::from(model.r#type) * (ckt.rhs_old(here.sNode) - ckt.rhs_old(here.sNodePrime));
            vdes =
                f64::from(model.r#type) * (ckt.rhs_old(here.dNode) - ckt.rhs_old(here.sNodePrime));
            qdef = f64::from(model.r#type) * (ckt.rhs_old(here.qNode));
        }
        vgdo = ckt.state0(state::VGS) - ckt.state0(state::VDS);
        vgedo = ckt.state0(state::VGES) - ckt.state0(state::VDS);
        vgmdo = ckt.state0(state::VGMS) - ckt.state0(state::VDS);
        vbd = vbs - vds;
        vdbd = vdbs - vds;
        vgd = vgs - vds;
        vged = vges - vds;
        vgmd = vgms - vds;
        // NOBYPASS and NEWCONV: the reference's predicted current
        // hats have no consumer in load; cvtest owns convergence testing.
        // Port-only: an exact (unlimited) load evaluates the CKTrhsOld
        // voltages unchanged and reports no limiting; C always limits here.
        if ckt.limiting() {
            von = here.von;
            if ckt.state0(state::VDS) >= 0.0 {
                vgs = fetlim(vgs, ckt.state0(state::VGS), von);
                vds = vgs - vgd;
                vds = limvds(vds, ckt.state0(state::VDS));
                vgd = vgs - vds;
                if here.rgateMod == 3 {
                    vges = fetlim(vges, ckt.state0(state::VGES), von);
                    vgms = fetlim(vgms, ckt.state0(state::VGMS), von);
                    vged = vges - vds;
                    vgmd = vgms - vds;
                } else if (here.rgateMod == 1) || (here.rgateMod == 2) {
                    vges = fetlim(vges, ckt.state0(state::VGES), von);
                    vged = vges - vds;
                }
                if model.rdsMod != 0 {
                    vdes = limvds(vdes, ckt.state0(state::VDES));
                    vses = -limvds(-vses, -(ckt.state0(state::VSES)));
                }
            } else {
                vgd = fetlim(vgd, vgdo, von);
                vds = vgs - vgd;
                vds = -limvds(-vds, -(ckt.state0(state::VDS)));
                vgs = vgd + vds;
                if here.rgateMod == 3 {
                    vged = fetlim(vged, vgedo, von);
                    vges = vged + vds;
                    vgmd = fetlim(vgmd, vgmdo, von);
                    vgms = vgmd + vds;
                }
                if (here.rgateMod == 1) || (here.rgateMod == 2) {
                    vged = fetlim(vged, vgedo, von);
                    vges = vged + vds;
                }
                if model.rdsMod != 0 {
                    vdes = -limvds(-vdes, -(ckt.state0(state::VDES)));
                    vses = limvds(vses, ckt.state0(state::VSES));
                }
            }
            if vds >= 0.0 {
                let limited = pnjlim(vbs, ckt.state0(state::VBS), CONST_VT0, model.vcrit);
                vbs = limited.value;
                Check = i32::from(limited.limited);
                vbd = vbs - vds;
                if here.rbodyMod != 0 {
                    let limited = pnjlim(vdbs, ckt.state0(state::VDBS), CONST_VT0, model.vcrit);
                    vdbs = limited.value;
                    Check1 = i32::from(limited.limited);
                    vdbd = vdbs - vds;
                    let limited = pnjlim(vsbs, ckt.state0(state::VSBS), CONST_VT0, model.vcrit);
                    vsbs = limited.value;
                    Check2 = i32::from(limited.limited);
                    if (Check1 == 0) && (Check2 == 0) {
                        Check = 0;
                    } else {
                        Check = 1;
                    }
                }
            } else {
                let limited = pnjlim(vbd, ckt.state0(state::VBD), CONST_VT0, model.vcrit);
                vbd = limited.value;
                Check = i32::from(limited.limited);
                vbs = vbd + vds;
                if here.rbodyMod != 0 {
                    let limited = pnjlim(vdbd, ckt.state0(state::VDBD), CONST_VT0, model.vcrit);
                    vdbd = limited.value;
                    Check1 = i32::from(limited.limited);
                    vdbs = vdbd + vds;
                    vsbdo = ckt.state0(state::VSBS) - ckt.state0(state::VDS);
                    vsbd = vsbs - vds;
                    let limited = pnjlim(vsbd, vsbdo, CONST_VT0, model.vcrit);
                    vsbd = limited.value;
                    Check2 = i32::from(limited.limited);
                    vsbs = vsbd + vds;
                    if (Check1 == 0) && (Check2 == 0) {
                        Check = 0;
                    } else {
                        Check = 1;
                    }
                }
            }
        } else {
            // Nothing was limited, so the load does not report
            // nonconvergence (C's `DEVpnjlim` would clear `Check`).
            Check = 0;
        }
    }
    vbd = vbs - vds;
    vgd = vgs - vds;
    vgb = vgs - vbs;
    vged = vges - vds;
    vgmd = vgms - vds;
    vgmb = vgms - vbs;
    vdbd = vdbs - vds;
    vbs_jct = (if (here.rbodyMod == 0) { vbs } else { vsbs });
    vbd_jct = (if (here.rbodyMod == 0) { vbd } else { vdbd });
    LoadBias {
        ChargeComputationNeeded,
        Check,
        qdef,
        vbd,
        vbd_jct,
        vbs,
        vbs_jct,
        vdbd,
        vdbs,
        vdes,
        vds,
        vgb,
        vgd,
        vges,
        vgmb,
        vgmd,
        vgms,
        vgs,
        vsbs,
        vses,
    }
}

/// Store limited terminal voltages and report limiting in C order.
pub(super) fn commit(here: &Instance, ckt: &mut dyn LoadCkt, bias: &LoadBias) {
    let &LoadBias {
        Check,
        qdef,
        vbd,
        vbs,
        vdbd,
        vdbs,
        vdes,
        vds,
        vges,
        vgms,
        vgs,
        vsbs,
        vses,
        ..
    } = bias;
    if (here.off == 0) || ((ckt.mode() & mode::MODEINITFIX) == 0) {
        if Check == 1 {
            ckt.noncon();
        }
    }
    ckt.set_state0(state::VDS, vds);
    ckt.set_state0(state::VGS, vgs);
    ckt.set_state0(state::VBS, vbs);
    ckt.set_state0(state::VBD, vbd);
    ckt.set_state0(state::VGES, vges);
    ckt.set_state0(state::VGMS, vgms);
    ckt.set_state0(state::VDBS, vdbs);
    ckt.set_state0(state::VDBD, vdbd);
    ckt.set_state0(state::VSBS, vsbs);
    ckt.set_state0(state::VSES, vses);
    ckt.set_state0(state::VDES, vdes);
    ckt.set_state0(state::QDEF, qdef);
}

#[cfg(test)]
mod tests {
    use super::predict;

    #[test]
    fn constant_history_remains_constant_at_large_step_ratios() {
        assert_eq!(predict(1e20, 0.123, 0.123), 0.123);
        assert_eq!(predict(2., 3., 2.), 5.);
    }
}
