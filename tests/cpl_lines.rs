//! #183: bounded multiconductor Maxwell model preparation and modal delay geometry.
use ngspice_rs::devices::cpl::plan::{CplMatrixKind, CplPlan, MAX_CONDUCTORS};

fn close(actual: f64, expected: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= relative * expected.abs(),
        "actual {actual:.17e}, expected {expected:.17e}"
    );
}

#[test]
fn one_conductor_recovers_the_exact_lc_delay_and_zero_loss() {
    let plan = CplPlan::prepare(1, 2., &[], &[1e-6], &[], &[1e-12]).unwrap();
    assert_eq!(plan.conductors(), 1);
    assert_eq!(plan.length(), 2.);
    assert_eq!(plan.matrix(CplMatrixKind::Resistance).get(0, 0), Some(0.));
    assert_eq!(plan.matrix(CplMatrixKind::Conductance).get(0, 0), Some(0.));
    close(plan.modal_delays()[0], 2e-9, 1e-14);
}

#[test]
fn differential_and_common_mode_delays_preserve_the_full_coupled_matrices() {
    // L has positive mutual terms; Maxwell C has negative mutual terms.
    // Symmetric 2x2 eigenvalues are (Lself +/- Lmutual)*(Cself +/- Cmutual).
    let plan = CplPlan::prepare(
        2,
        3.,
        &[1., 0., 2.],
        &[4e-7, 0.8e-7, 4e-7],
        &[0., 0.],
        &[1e-10, -1e-11, 1e-10],
    )
    .unwrap();
    assert_eq!(plan.matrix(CplMatrixKind::Resistance).get(0, 1), Some(0.));
    assert_eq!(
        plan.matrix(CplMatrixKind::Capacitance).get(1, 0),
        Some(-1e-11)
    );
    assert_eq!(
        plan.matrix(CplMatrixKind::Inductance).get(1, 0),
        Some(0.8e-7)
    );
    let differential = 3. * ((4e-7_f64 - 0.8e-7) * (1e-10 + 1e-11)).sqrt();
    let common = 3. * ((4e-7_f64 + 0.8e-7) * (1e-10 - 1e-11)).sqrt();
    let expected = if differential < common {
        [differential, common]
    } else {
        [common, differential]
    };
    for (actual, expected) in plan.modal_delays().iter().zip(expected) {
        close(*actual, expected, 1e-12);
    }
}

#[test]
fn noncommuting_lc_matrices_follow_the_general_telegrapher_eigenvalues() {
    let l = [4e-7, 0.5e-7, 3e-7];
    let c = [1e-10, -2e-11, 2e-10];
    let plan = CplPlan::prepare(2, 1.5, &[0., 0.], &l, &[], &c).unwrap();
    let trace = l[2].mul_add(c[2], (2. * l[1]).mul_add(c[1], l[0] * c[0]));
    let determinant = l[1].mul_add(-l[1], l[0] * l[2]) * c[1].mul_add(-c[1], c[0] * c[2]);
    let root = (-4_f64).mul_add(determinant, trace * trace).sqrt();
    let expected = [
        1.5 * f64::midpoint(trace, -root).sqrt(),
        1.5 * f64::midpoint(trace, root).sqrt(),
    ];
    for (actual, expected) in plan.modal_delays().iter().zip(expected) {
        close(*actual, expected, 1e-12);
    }
}

#[test]
fn three_conductor_cable_includes_crosstalk_and_lossless_modes() {
    let plan = CplPlan::prepare(
        3,
        0.5,
        &[0.02, 0.03, 0.04],
        &[4e-7, 0.6e-7, 0.3e-7, 4e-7, 0.6e-7, 4e-7],
        &[0.],
        &[1e-10, -1e-11, -0.4e-11, 1e-10, -1e-11, 1e-10],
    );
    assert!(plan.is_err()); // G must be empty, diagonal 3, or triangle 6.
    let plan = CplPlan::prepare(
        3,
        0.5,
        &[0.02, 0.03, 0.04],
        &[4e-7, 0.6e-7, 0.3e-7, 4e-7, 0.6e-7, 4e-7],
        &[],
        &[1e-10, -1e-11, -0.4e-11, 1e-10, -1e-11, 1e-10],
    )
    .unwrap();
    assert_eq!(plan.modal_delays().len(), 3);
    assert!(plan.modal_delays().windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(
        plan.matrix(CplMatrixKind::Inductance).get(2, 0),
        Some(0.3e-7)
    );
    assert_eq!(
        plan.matrix(CplMatrixKind::Capacitance).get(2, 1),
        Some(-1e-11)
    );
}

#[test]
fn eight_conductor_input_is_bounded_and_retains_all_coefficients() {
    let n = MAX_CONDUCTORS;
    let mut l = Vec::new();
    let mut c = Vec::new();
    for row in 0..n {
        for col in row..n {
            l.push(if row == col { 4e-7 } else { 0. });
            c.push(if row == col { 1e-10 } else { 0. });
        }
    }
    let plan = CplPlan::prepare(n, 1., &[], &l, &[], &c).unwrap();
    assert_eq!(plan.modal_delays().len(), 8);
    assert_eq!(plan.matrix(CplMatrixKind::Capacitance).data().len(), 64);
    for delay in plan.modal_delays() {
        close(*delay, (4e-7_f64 * 1e-10).sqrt(), 1e-12);
    }
    assert!(CplPlan::prepare(n + 1, 1., &[], &l, &[], &c).is_err());
}

#[test]
fn singular_nonpassive_and_badly_conditioned_models_are_rejected() {
    let l = [4e-7, 1e-7, 4e-7];
    let c = [1e-10, -1e-11, 1e-10];
    assert!(CplPlan::prepare(2, 1., &[1., 2., 1.], &l, &[], &c).is_err());
    assert!(CplPlan::prepare(2, 1., &[0., 1., 1.], &l, &[], &c).is_err());
    assert!(CplPlan::prepare(2, 1., &[-1., 0.], &l, &[], &c).is_err());
    assert!(CplPlan::prepare(2, 1., &[], &[1., 1., 1.], &[], &c).is_err());
    assert!(CplPlan::prepare(2, 1., &[], &l, &[], &[1., 1., 1.]).is_err());
    assert!(CplPlan::prepare(2, 1., &[], &l, &[], &[1., 1. - 1e-15, 1.]).is_err());
}

#[test]
fn exactly_semidefinite_loss_and_negative_maxwell_mutual_are_valid() {
    let l = [4e-7, 0.5e-7, 4e-7];
    let c = [1e-10, -2e-11, 1e-10];
    let plan = CplPlan::prepare(2, 1., &[2., -2., 2.], &l, &[0., 0., 0.], &c).unwrap();
    assert_eq!(plan.matrix(CplMatrixKind::Resistance).get(0, 1), Some(-2.));
    assert_eq!(
        plan.matrix(CplMatrixKind::Capacitance).get(0, 1),
        Some(-2e-11)
    );
    assert_eq!(plan.modal_delays().len(), 2);
}

#[test]
fn incomplete_nonfinite_and_out_of_range_inputs_never_truncate_or_default() {
    let l = [1e-6];
    let c = [1e-12];
    for length in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(CplPlan::prepare(1, length, &[], &l, &[], &c).is_err());
    }
    assert!(CplPlan::prepare(0, 1., &[], &l, &[], &c).is_err());
    assert!(CplPlan::prepare(1, 1., &[f64::NAN], &l, &[], &c).is_err());
    assert!(CplPlan::prepare(1, 1., &[], &[f64::INFINITY], &[], &c).is_err());
    assert!(CplPlan::prepare(1, 1., &[], &[], &[], &c).is_err());
    assert!(CplPlan::prepare(1, 1., &[], &l, &[], &[]).is_err());
    assert!(CplPlan::prepare(2, 1., &[], &l, &[], &c).is_err());
}
