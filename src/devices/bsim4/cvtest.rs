//! `b4cvtest.c` (`BSIM4convTest`): the device convergence test, followed
//! statement by statement over C's struct members.
// Current predictors select hardware FMA when available; C's two-round
// convergence tolerances and strict comparison operators remain unchanged.
#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols keep their original distinct names for source-level audit"
)]
#![expect(
    non_snake_case,
    reason = "preserves upstream C identifiers for equation-by-equation comparison"
)]
use super::fields::{Instance, Model};
use super::host::{mode, state};

/// C's `MAX` for the tolerance bases.
fn max(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// Compile-time FMA selection: only request the fused instruction when
/// the requested CPU target provides hardware FMA. Generic builds must
/// not invoke the much slower software floating-point FMA implementation.
/// Both alternatives accumulate the terms in their original C order.
#[inline(always)]
fn predictor_term(gain: f64, delta: f64, sum: f64) -> f64 {
    if cfg!(target_feature = "fma") {
        gain.mul_add(delta, sum)
    } else {
        let product = gain * delta;
        sum + product
    }
}

/// Preserve the two C rounding steps before the discontinuous convergence
/// comparisons. Fusing the multiply-add can move a one-ULP boundary.
#[expect(
    clippy::suboptimal_flops,
    reason = "BSIM4convTest compares separately rounded reltol*magnitude+abstol"
)]
fn convergence_tolerance(predicted: f64, original: f64, reltol: f64, abstol: f64) -> f64 {
    reltol * max(predicted.abs(), original.abs()) + abstol
}

/// `BSIM4convTest` for one instance: `true` when C would increment
/// `CKTnoncon`. `rhs_old` is the new iterate C reads as `CKTrhsOld`,
/// `state0` the last load's state vector, `mode` C's `CKTmode`.
pub(super) fn conv_test(
    m: &Model,
    h: &Instance,
    rhs_old: &dyn Fn(i32) -> f64,
    state0: &dyn Fn(usize) -> f64,
    mode: u32,
    reltol: f64,
    abstol: f64,
) -> bool {
    let ty = f64::from(m.r#type);
    let vds = ty * (rhs_old(h.dNodePrime) - rhs_old(h.sNodePrime));
    let vgs = ty * (rhs_old(h.gNodePrime) - rhs_old(h.sNodePrime));
    let vbs = ty * (rhs_old(h.bNodePrime) - rhs_old(h.sNodePrime));
    let vdbs = ty * (rhs_old(h.dbNode) - rhs_old(h.sNodePrime));
    let vsbs = ty * (rhs_old(h.sbNode) - rhs_old(h.sNodePrime));
    let vses = ty * (rhs_old(h.sNode) - rhs_old(h.sNodePrime));
    let vdes = ty * (rhs_old(h.dNode) - rhs_old(h.sNodePrime));

    let vgdo = state0(state::VGS) - state0(state::VDS);
    let vbd = vbs - vds;
    let vdbd = vdbs - vds;
    let vgd = vgs - vds;

    let delvbd = vbd - state0(state::VBD);
    let delvdbd = vdbd - state0(state::VDBD);
    let delvgd = vgd - vgdo;

    let delvds = vds - state0(state::VDS);
    let delvgs = vgs - state0(state::VGS);
    let delvbs = vbs - state0(state::VBS);
    let delvsbs = vsbs - state0(state::VSBS);

    let delvses = vses - (state0(state::VSES));
    let vdedo = state0(state::VDES) - state0(state::VDS);
    let delvdes = vdes - state0(state::VDES);
    let delvded = vdes - vds - vdedo;
    let _ = delvdes;

    let delvbd_jct = if h.rbodyMod == 0 { delvbd } else { delvdbd };
    let delvbs_jct = if h.rbodyMod == 0 { delvbs } else { delvsbs };

    let (Idtot, cdhat, Igstot, cgshat, Igdtot, cgdhat, Igbtot, cgbhat);
    if h.mode >= 0 {
        Idtot = h.cd + h.csub - h.cbd + h.Igidl;
        cdhat = predictor_term(
            h.gds + h.gbds + h.ggidld,
            delvds,
            predictor_term(
                h.gm + h.gbgs + h.ggidlg,
                delvgs,
                predictor_term(
                    h.gmbs + h.gbbs + h.ggidlb,
                    delvbs,
                    predictor_term(h.gbd, -delvbd_jct, Idtot),
                ),
            ),
        );
        Igstot = h.Igs + h.Igcs;
        cgshat = predictor_term(
            h.gIgcsb,
            delvbs,
            predictor_term(
                h.gIgcsd,
                delvds,
                predictor_term(h.gIgsg + h.gIgcsg, delvgs, Igstot),
            ),
        );
        Igdtot = h.Igd + h.Igcd;
        cgdhat = predictor_term(
            h.gIgcdb,
            delvbs,
            predictor_term(
                h.gIgcdd,
                delvds,
                predictor_term(h.gIgcdg, delvgs, predictor_term(h.gIgdg, delvgd, Igdtot)),
            ),
        );
        Igbtot = h.Igb;
        cgbhat = predictor_term(
            h.gIgbb,
            delvbs,
            predictor_term(h.gIgbd, delvds, predictor_term(h.gIgbg, delvgs, h.Igb)),
        );
    } else {
        // C's own "bugfix" spelling of the reverse-mode drain current.
        Idtot = h.cd + h.cbd - h.Igidl;
        cdhat = predictor_term(
            h.ggidlb,
            -delvbs,
            predictor_term(
                h.ggidlg,
                -delvgs,
                predictor_term(
                    h.gds + h.ggidls,
                    -delvds,
                    predictor_term(
                        h.gm,
                        delvgd,
                        predictor_term(h.gmbs, delvbd, predictor_term(h.gbd, delvbd_jct, Idtot)),
                    ),
                ),
            ),
        );
        Igstot = h.Igs + h.Igcd;
        cgshat = predictor_term(
            h.gIgcdb,
            delvbd,
            predictor_term(
                h.gIgcdd,
                -delvds,
                predictor_term(h.gIgcdg, delvgd, predictor_term(h.gIgsg, delvgs, Igstot)),
            ),
        );
        Igdtot = h.Igd + h.Igcs;
        cgdhat = predictor_term(
            h.gIgcsb,
            delvbd,
            predictor_term(
                h.gIgcsd,
                -delvds,
                predictor_term(h.gIgdg + h.gIgcsg, delvgd, Igdtot),
            ),
        );
        Igbtot = h.Igb;
        cgbhat = predictor_term(
            h.gIgbb,
            delvbd,
            predictor_term(h.gIgbd, -delvds, predictor_term(h.gIgbg, delvgd, h.Igb)),
        );
    }

    let Isestot = h.gstot * (state0(state::VSES));
    let cseshat = predictor_term(
        h.gstotb,
        delvbs,
        predictor_term(
            h.gstotg,
            delvgs,
            predictor_term(h.gstotd, delvds, predictor_term(h.gstot, delvses, Isestot)),
        ),
    );
    let Idedtot = h.gdtot * vdedo;
    let cdedhat = predictor_term(
        h.gdtotb,
        delvbs,
        predictor_term(
            h.gdtotg,
            delvgs,
            predictor_term(h.gdtotd, delvds, predictor_term(h.gdtot, delvded, Idedtot)),
        ),
    );

    // Check convergence
    if (h.off == 0) || ((mode & mode::MODEINITFIX) == 0) {
        let tol0 = convergence_tolerance(cdhat, Idtot, reltol, abstol);
        let tol1 = convergence_tolerance(cseshat, Isestot, reltol, abstol);
        let tol2 = convergence_tolerance(cdedhat, Idedtot, reltol, abstol);
        let tol3 = convergence_tolerance(cgshat, Igstot, reltol, abstol);
        let tol4 = convergence_tolerance(cgdhat, Igdtot, reltol, abstol);
        let tol5 = convergence_tolerance(cgbhat, Igbtot, reltol, abstol);
        if ((cdhat - Idtot).abs() >= tol0)
            || ((cseshat - Isestot).abs() >= tol1)
            || ((cdedhat - Idedtot).abs() >= tol2)
        {
            return true;
        }
        if ((cgshat - Igstot).abs() >= tol3)
            || ((cgdhat - Igdtot).abs() >= tol4)
            || ((cgbhat - Igbtot).abs() >= tol5)
        {
            return true;
        }
        let Ibtot = h.cbs + h.cbd - h.Igidl - h.Igisl - h.csub;
        let cbhat = if h.mode >= 0 {
            predictor_term(
                h.ggisls,
                delvds,
                predictor_term(
                    h.ggislb,
                    -delvbd,
                    predictor_term(
                        h.ggislg,
                        -delvgd,
                        predictor_term(
                            h.gbds + h.ggidld,
                            -delvds,
                            predictor_term(
                                h.gbgs + h.ggidlg,
                                -delvgs,
                                predictor_term(
                                    h.gbbs + h.ggidlb,
                                    -delvbs,
                                    predictor_term(
                                        h.gbs,
                                        delvbs_jct,
                                        predictor_term(h.gbd, delvbd_jct, Ibtot),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        } else {
            predictor_term(
                h.ggidlb,
                -delvbs,
                predictor_term(
                    h.ggidlg,
                    -delvgs,
                    predictor_term(
                        h.gbds + h.ggisld - h.ggidls,
                        delvds,
                        predictor_term(
                            h.gbgs + h.ggislg,
                            -delvgd,
                            predictor_term(
                                h.gbbs + h.ggislb,
                                -delvbd,
                                predictor_term(
                                    h.gbd,
                                    delvbd_jct,
                                    predictor_term(h.gbs, delvbs_jct, Ibtot),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        };
        let tol6 = convergence_tolerance(cbhat, Ibtot, reltol, abstol);
        if (cbhat - Ibtot).abs() > tol6 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::super::acld::tests::shaped;
    use super::super::host::{mode, node, state};
    use super::{Model, conv_test, convergence_tolerance, predictor_term};

    fn run(h: &super::Instance, rhs: &[f64; node::COUNT], s0: &[f64; state::COUNT]) -> bool {
        let m = Model {
            r#type: 1,
            ..Model::default()
        };
        let rhs_old = |n: i32| usize::try_from(n).map_or(0.0, |n| rhs[n]);
        let state0 = |k: usize| s0[k];
        conv_test(&m, h, &rhs_old, &state0, mode::MODEINITFLOAT, 1e-3, 1e-12)
    }

    #[test]
    fn converged_when_the_iterate_matches_the_load() {
        let mut h = shaped(0, 0, 0);
        h.cd = 1e-3;
        h.gm = 1e-3;
        h.gds = 1e-5;
        let mut rhs = [0.0; node::COUNT];
        rhs[node::DP as usize] = 1.0;
        rhs[node::D as usize] = 1.0;
        rhs[node::G_EXT as usize] = 0.8;
        let mut s0 = [0.0; state::COUNT];
        s0[state::VDS] = 1.0;
        s0[state::VGS] = 0.8;
        s0[state::VDES] = 1.0;
        s0[state::VBD] = -1.0;
        s0[state::VDBD] = -1.0;
        assert!(!run(&h, &rhs, &s0));
        // A large gate step predicts a current change well beyond reltol.
        rhs[node::G_EXT as usize] = 1.8;
        assert!(run(&h, &rhs, &s0));
    }

    #[test]
    fn predictor_ge_and_bulk_gt_keep_one_ulp_threshold_boundaries() {
        // C uses >= on the first six predictor checks, but > on bulk.
        // A gate change of 1 V makes each chosen conductance exactly the
        // corresponding predicted current increment, independent of
        // source/device capacitance and Newton step history.
        let mut h = shaped(0, 0, 0);
        let mut rhs = [0.0; node::COUNT];
        rhs[node::G_EXT as usize] = 1.0;
        let s0 = [0.0; state::COUNT];
        let m = Model {
            r#type: 1,
            ..Model::default()
        };
        let rhs_old = |node: i32| usize::try_from(node).map_or(0.0, |i| rhs[i]);
        let state0 = |index: usize| s0[index];
        let threshold = 0.25_f64;
        let just_below = f64::from_bits(threshold.to_bits() - 1);
        let just_above = f64::from_bits(threshold.to_bits() + 1);

        for mode_sign in [1, -1] {
            h.mode = mode_sign;
            for (increment, nonconverged) in
                [(just_below, false), (threshold, true), (just_above, true)]
            {
                h.gm = increment;
                assert_eq!(
                    conv_test(
                        &m,
                        &h,
                        &rhs_old,
                        &state0,
                        mode::MODEINITFLOAT,
                        0.0,
                        threshold
                    ),
                    nonconverged,
                    "drain predictor: mode={mode_sign} current={increment:.17e}"
                );
            }

            h.gm = 0.0;
            // ggislg contributes only to bulk prediction (not an earlier
            // drain/gate/series predictor) when operating in forward mode.
            // Reverse mode uses ggislb on delvbd; test forward only below.
        }

        h.mode = 1;
        for (increment, nonconverged) in
            [(just_below, false), (threshold, false), (just_above, true)]
        {
            h.ggislg = increment;
            assert_eq!(
                conv_test(
                    &m,
                    &h,
                    &rhs_old,
                    &state0,
                    mode::MODEINITFLOAT,
                    0.0,
                    threshold
                ),
                nonconverged,
                "bulk predictor: conductance={increment:.17e}"
            );
        }
    }

    #[test]
    fn predictor_uses_hardware_fma_only_when_the_compilation_target_has_it() {
        let e = 2_f64.powi(-27);
        let result = predictor_term(1. + e, 1. - e, -1.);
        let expected = if cfg!(target_feature = "fma") {
            -2_f64.powi(-54)
        } else {
            0.
        };
        assert_eq!(
            result.to_bits(),
            expected.to_bits(),
            "predictor result for compile-time target feature fma"
        );
    }

    #[test]
    fn two_round_threshold_prevents_false_fma_nonconvergence() {
        // (1+e)(1-e) = 1-e², but C multiplication first rounds to 1,
        // then adds 3*2^-54. The fused value rounds back to 1.
        let e = 2_f64.powi(-27);
        let reltol = 1. + e;
        let abstol = 3. * 2_f64.powi(-54);
        let predicted = 1. - e;
        let old = -e;
        assert_eq!((predicted - old).to_bits(), 1_f64.to_bits());
        assert_eq!(
            convergence_tolerance(predicted, old, reltol, abstol).to_bits(),
            (1. + f64::EPSILON).to_bits()
        );
        assert_eq!(reltol.mul_add(predicted, abstol).to_bits(), 1_f64.to_bits());
        let mut h = shaped(0, 0, 0);
        h.cd = old;
        h.gm = 1.;
        let mut rhs = [0.0; node::COUNT];
        rhs[node::G_EXT as usize] = 1.;
        let state = [0.0; state::COUNT];
        let model = Model {
            r#type: 1,
            ..Model::default()
        };
        let rhs_old = |node: i32| usize::try_from(node).map_or(0.0, |i| rhs[i]);
        let state0 = |index: usize| state[index];
        assert!(
            !conv_test(
                &model,
                &h,
                &rhs_old,
                &state0,
                mode::MODEINITFLOAT,
                reltol,
                abstol
            ),
            "two-round C convergence bound should accept this current change"
        );
    }

    #[test]
    fn c_nonfinite_and_signed_zero_comparisons_are_deliberate() {
        let mut h = shaped(0, 0, 0);
        let mut rhs = [0.0; node::COUNT];
        rhs[node::G_EXT as usize] = 1.0;
        let s0 = [0.0; state::COUNT];
        let m = Model {
            r#type: 1,
            ..Model::default()
        };
        let rhs_old = |node: i32| usize::try_from(node).map_or(0.0, |i| rhs[i]);
        let state0 = |index: usize| s0[index];
        for current in [-0.0, 0.0, f64::NAN] {
            h.gm = current;
            assert!(!conv_test(
                &m,
                &h,
                &rhs_old,
                &state0,
                mode::MODEINITFLOAT,
                0.0,
                1e-12
            ));
        }
    }

    #[test]
    fn an_off_instance_is_not_tested_in_initfix() {
        let mut h = shaped(0, 0, 0);
        h.off = 1;
        h.gm = 1.0;
        let mut rhs = [0.0; node::COUNT];
        rhs[node::G_EXT as usize] = 5.0;
        let s0 = [0.0; state::COUNT];
        let m = Model {
            r#type: 1,
            ..Model::default()
        };
        let rhs_old = |n: i32| usize::try_from(n).map_or(0.0, |n| rhs[n]);
        let state0 = |k: usize| s0[k];
        let fix = mode::MODEDCOP | mode::MODEINITFIX;
        assert!(!conv_test(&m, &h, &rhs_old, &state0, fix, 1e-3, 1e-12));
        assert!(conv_test(
            &m,
            &h,
            &rhs_old,
            &state0,
            mode::MODEINITFLOAT,
            1e-3,
            1e-12
        ));
    }
}
