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
#[inline]
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

/// Apply smooth predictor terms in exactly the reference accumulation order.
#[inline]
fn predictor<const N: usize>(original: f64, terms: [(f64, f64); N]) -> f64 {
    terms.into_iter().fold(original, |sum, (gain, delta)| {
        predictor_term(gain, delta, sum)
    })
}

#[derive(Clone, Copy)]
struct Prediction {
    original: f64,
    estimated: f64,
}

impl Prediction {
    const fn new(original: f64, estimated: f64) -> Self {
        Self {
            original,
            estimated,
        }
    }

    fn rejects(self, reltol: f64, abstol: f64, inclusive: bool) -> bool {
        let tolerance = convergence_tolerance(self.estimated, self.original, reltol, abstol);
        let error = (self.estimated - self.original).abs();
        if inclusive {
            error >= tolerance
        } else {
            error > tolerance
        }
    }
}

#[derive(Clone, Copy)]
struct BiasDelta {
    delvbd: f64,
    delvgd: f64,
    delvds: f64,
    delvgs: f64,
    delvbs: f64,
    delvses: f64,
    delvded: f64,
    delvbd_jct: f64,
    delvbs_jct: f64,
    vdedo: f64,
}

impl BiasDelta {
    fn read(
        m: &Model,
        h: &Instance,
        rhs_old: &dyn Fn(i32) -> f64,
        state0: &dyn Fn(usize) -> f64,
    ) -> Self {
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
        let delvded = vdes - vds - vdedo;

        let delvbd_jct = if h.rbodyMod == 0 { delvbd } else { delvdbd };
        let delvbs_jct = if h.rbodyMod == 0 { delvbs } else { delvsbs };

        Self {
            delvbd,
            delvgd,
            delvds,
            delvgs,
            delvbs,
            delvses,
            delvded,
            delvbd_jct,
            delvbs_jct,
            vdedo,
        }
    }
}

/// Drain, gate-source, gate-drain and gate-bulk predictions in the
/// reference forward/reverse orientation.
fn drain_and_gate(h: &Instance, d: BiasDelta) -> [Prediction; 4] {
    let BiasDelta {
        delvbd,
        delvgd,
        delvds,
        delvgs,
        delvbs,
        delvbd_jct,
        ..
    } = d;
    let (Idtot, cdhat, Igstot, cgshat, Igdtot, cgdhat, cgbhat);
    if h.mode >= 0 {
        Idtot = h.cd + h.csub - h.cbd + h.Igidl;
        cdhat = predictor(
            Idtot,
            [
                (h.gbd, -delvbd_jct),
                (h.gmbs + h.gbbs + h.ggidlb, delvbs),
                (h.gm + h.gbgs + h.ggidlg, delvgs),
                (h.gds + h.gbds + h.ggidld, delvds),
            ],
        );
        Igstot = h.Igs + h.Igcs;
        cgshat = predictor(
            Igstot,
            [
                (h.gIgsg + h.gIgcsg, delvgs),
                (h.gIgcsd, delvds),
                (h.gIgcsb, delvbs),
            ],
        );
        Igdtot = h.Igd + h.Igcd;
        cgdhat = predictor(
            Igdtot,
            [
                (h.gIgdg, delvgd),
                (h.gIgcdg, delvgs),
                (h.gIgcdd, delvds),
                (h.gIgcdb, delvbs),
            ],
        );
        cgbhat = predictor(
            h.Igb,
            [(h.gIgbg, delvgs), (h.gIgbd, delvds), (h.gIgbb, delvbs)],
        );
    } else {
        // C's own "bugfix" spelling of the reverse-mode drain current.
        Idtot = h.cd + h.cbd - h.Igidl;
        cdhat = predictor(
            Idtot,
            [
                (h.gbd, delvbd_jct),
                (h.gmbs, delvbd),
                (h.gm, delvgd),
                (h.gds + h.ggidls, -delvds),
                (h.ggidlg, -delvgs),
                (h.ggidlb, -delvbs),
            ],
        );
        Igstot = h.Igs + h.Igcd;
        cgshat = predictor(
            Igstot,
            [
                (h.gIgsg, delvgs),
                (h.gIgcdg, delvgd),
                (h.gIgcdd, -delvds),
                (h.gIgcdb, delvbd),
            ],
        );
        Igdtot = h.Igd + h.Igcs;
        cgdhat = predictor(
            Igdtot,
            [
                (h.gIgdg + h.gIgcsg, delvgd),
                (h.gIgcsd, -delvds),
                (h.gIgcsb, delvbd),
            ],
        );
        cgbhat = predictor(
            h.Igb,
            [(h.gIgbg, delvgd), (h.gIgbd, -delvds), (h.gIgbb, delvbd)],
        );
    }

    [
        Prediction::new(Idtot, cdhat),
        Prediction::new(Igstot, cgshat),
        Prediction::new(Igdtot, cgdhat),
        Prediction::new(h.Igb, cgbhat),
    ]
}

fn resistance(h: &Instance, d: BiasDelta, state0: &dyn Fn(usize) -> f64) -> [Prediction; 2] {
    let BiasDelta {
        delvbs,
        delvgs,
        delvds,
        delvses,
        delvded,
        vdedo,
        ..
    } = d;
    let Isestot = h.gstot * (state0(state::VSES));
    let cseshat = predictor(
        Isestot,
        [
            (h.gstot, delvses),
            (h.gstotd, delvds),
            (h.gstotg, delvgs),
            (h.gstotb, delvbs),
        ],
    );
    let Idedtot = h.gdtot * vdedo;
    let cdedhat = predictor(
        Idedtot,
        [
            (h.gdtot, delvded),
            (h.gdtotd, delvds),
            (h.gdtotg, delvgs),
            (h.gdtotb, delvbs),
        ],
    );

    [
        Prediction::new(Isestot, cseshat),
        Prediction::new(Idedtot, cdedhat),
    ]
}

fn bulk(h: &Instance, d: BiasDelta) -> Prediction {
    let BiasDelta {
        delvds,
        delvbd,
        delvgd,
        delvgs,
        delvbs,
        delvbs_jct,
        delvbd_jct,
        ..
    } = d;
    let Ibtot = h.cbs + h.cbd - h.Igidl - h.Igisl - h.csub;
    let cbhat = if h.mode >= 0 {
        predictor(
            Ibtot,
            [
                (h.gbd, delvbd_jct),
                (h.gbs, delvbs_jct),
                (h.gbbs + h.ggidlb, -delvbs),
                (h.gbgs + h.ggidlg, -delvgs),
                (h.gbds + h.ggidld, -delvds),
                (h.ggislg, -delvgd),
                (h.ggislb, -delvbd),
                (h.ggisls, delvds),
            ],
        )
    } else {
        predictor(
            Ibtot,
            [
                (h.gbs, delvbs_jct),
                (h.gbd, delvbd_jct),
                (h.gbbs + h.ggislb, -delvbd),
                (h.gbgs + h.ggislg, -delvgd),
                (h.gbds + h.ggisld - h.ggidls, delvds),
                (h.ggidlg, -delvgs),
                (h.ggidlb, -delvbs),
            ],
        )
    };
    Prediction::new(Ibtot, cbhat)
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
    let delta = BiasDelta::read(m, h, rhs_old, state0);
    let [drain, gate_source, gate_drain, gate_bulk] = drain_and_gate(h, delta);
    let [source_resistance, drain_resistance] = resistance(h, delta, state0);
    if h.off != 0 && mode & mode::MODEINITFIX != 0 {
        return false;
    }
    // The reference first checks drain and terminal resistance, then gate
    // currents, and evaluates bulk only if all six inclusive checks pass.
    [
        drain,
        source_resistance,
        drain_resistance,
        gate_source,
        gate_drain,
        gate_bulk,
    ]
    .into_iter()
    .any(|prediction| prediction.rejects(reltol, abstol, true))
        || bulk(h, delta).rejects(reltol, abstol, false)
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
