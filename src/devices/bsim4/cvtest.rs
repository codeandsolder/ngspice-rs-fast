//! `b4cvtest.c` (`BSIM4convTest`): the device convergence test, followed
//! statement by statement over C's struct members.
#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols keep their original distinct names for source-level audit"
)]
use super::fields::{Instance, Model};
use super::host::{mode, state};

/// C's `MAX` for the tolerance bases.
fn max(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// Calculate the linearized current change independently of the previous
/// current. Forming the predicted total and subtracting it again loses a
/// small correction when the two totals round to the same value.
fn prediction<const N: usize>(
    previous: f64,
    terms: [(f64, f64); N],
    reltol: f64,
    abstol: f64,
) -> (f64, f64) {
    let delta = terms
        .into_iter()
        .fold(0., |sum, (slope, voltage)| slope.mul_add(voltage, sum));
    let predicted = previous + delta;
    let error = if previous.is_finite() && predicted.is_finite() {
        delta.abs()
    } else {
        // Preserve the reference comparison's NaN/Inf behavior.
        (predicted - previous).abs()
    };
    let tolerance = reltol.mul_add(max(predicted.abs(), previous.abs()), abstol);
    (error, tolerance)
}

/// `BSIM4convTest` for one instance: `true` when C would increment
/// `CKTnoncon`. `rhs_old` is the new iterate C reads as `CKTrhsOld`,
/// `state0` the last load's state vector, `mode` C's `CKTmode`.
#[expect(
    clippy::too_many_lines,
    reason = "keep the forward/reverse terminal-current prediction families together"
)] // One C function, kept in C's shape.
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

    // The fixed-off instance does not participate in convergence testing.
    if h.off != 0 && (mode & mode::MODEINITFIX) != 0 {
        return false;
    }

    let compare = |(error, tolerance): (f64, f64)| error >= tolerance;
    let (drain, gate_source, gate_drain, gate_bulk) = if h.mode >= 0 {
        (
            prediction(
                h.cd + h.csub - h.cbd + h.Igidl,
                [
                    (h.gbd, -delvbd_jct),
                    (h.gmbs + h.gbbs + h.ggidlb, delvbs),
                    (h.gm + h.gbgs + h.ggidlg, delvgs),
                    (h.gds + h.gbds + h.ggidld, delvds),
                ],
                reltol,
                abstol,
            ),
            prediction(
                h.Igs + h.Igcs,
                [
                    (h.gIgsg + h.gIgcsg, delvgs),
                    (h.gIgcsd, delvds),
                    (h.gIgcsb, delvbs),
                ],
                reltol,
                abstol,
            ),
            prediction(
                h.Igd + h.Igcd,
                [
                    (h.gIgdg, delvgd),
                    (h.gIgcdg, delvgs),
                    (h.gIgcdd, delvds),
                    (h.gIgcdb, delvbs),
                ],
                reltol,
                abstol,
            ),
            prediction(
                h.Igb,
                [(h.gIgbg, delvgs), (h.gIgbd, delvds), (h.gIgbb, delvbs)],
                reltol,
                abstol,
            ),
        )
    } else {
        (
            // C's bugfix spelling of the reverse-mode drain current.
            prediction(
                h.cd + h.cbd - h.Igidl,
                [
                    (h.gbd, delvbd_jct),
                    (h.gmbs, delvbd),
                    (h.gm, delvgd),
                    (h.gds + h.ggidls, -delvds),
                    (h.ggidlg, -delvgs),
                    (h.ggidlb, -delvbs),
                ],
                reltol,
                abstol,
            ),
            prediction(
                h.Igs + h.Igcd,
                [
                    (h.gIgsg, delvgs),
                    (h.gIgcdg, delvgd),
                    (h.gIgcdd, -delvds),
                    (h.gIgcdb, delvbd),
                ],
                reltol,
                abstol,
            ),
            prediction(
                h.Igd + h.Igcs,
                [
                    (h.gIgdg + h.gIgcsg, delvgd),
                    (h.gIgcsd, -delvds),
                    (h.gIgcsb, delvbd),
                ],
                reltol,
                abstol,
            ),
            prediction(
                h.Igb,
                [(h.gIgbg, delvgd), (h.gIgbd, -delvds), (h.gIgbb, delvbd)],
                reltol,
                abstol,
            ),
        )
    };
    let source_series = prediction(
        h.gstot * state0(state::VSES),
        [
            (h.gstot, delvses),
            (h.gstotd, delvds),
            (h.gstotg, delvgs),
            (h.gstotb, delvbs),
        ],
        reltol,
        abstol,
    );
    let drain_series = prediction(
        h.gdtot * vdedo,
        [
            (h.gdtot, delvded),
            (h.gdtotd, delvds),
            (h.gdtotg, delvgs),
            (h.gdtotb, delvbs),
        ],
        reltol,
        abstol,
    );
    if [
        drain,
        source_series,
        drain_series,
        gate_source,
        gate_drain,
        gate_bulk,
    ]
    .into_iter()
    .any(compare)
    {
        return true;
    }

    let bulk_current = h.cbs + h.cbd - h.Igidl - h.Igisl - h.csub;
    let (error, tolerance) = if h.mode >= 0 {
        prediction(
            bulk_current,
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
            reltol,
            abstol,
        )
    } else {
        prediction(
            bulk_current,
            [
                (h.gbs, delvbs_jct),
                (h.gbd, delvbd_jct),
                (h.gbbs + h.ggislb, -delvbd),
                (h.gbgs + h.ggislg, -delvgd),
                (h.gbds + h.ggisld - h.ggidls, delvds),
                (h.ggidlg, -delvgs),
                (h.ggidlb, -delvbs),
            ],
            reltol,
            abstol,
        )
    };
    // C uses a strict comparison for bulk, inclusive for other currents.
    error > tolerance
}

#[cfg(test)]
mod tests {
    use super::super::acld::tests::shaped;
    use super::super::host::{mode, node, state};
    use super::{Model, conv_test, prediction};

    #[test]
    fn small_current_corrections_are_not_rounded_out_of_the_residual() {
        let (error, tolerance) = prediction(1e20, [(1., 1.)], 0., 0.5);
        assert_eq!((error, tolerance), (1., 0.5));
        assert_eq!(1e20 + 1., 1e20);
        assert!(error > tolerance);
        let (error, _) = prediction(f64::INFINITY, [(1., 1.)], 1e-3, 1e-12);
        assert!(error.is_nan());
    }

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
