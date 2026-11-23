//! Physical coupled modal geometry: verify against the telegrapher equations,
//! not merely the Jacobi algorithm's self-reported eigenvalues.
use ngspice_rs::devices::cpl::plan::{CplMatrixKind, CplPlan, MAX_CONDUCTORS};
use ngspice_rs::maths::Matrix;

fn rel_close(got: f64, want: f64, tolerance: f64) {
    assert!(
        (got - want).abs() <= tolerance * got.abs().max(want.abs()).max(1e-300),
        "got {got:.17e} vs expected {want:.17e}"
    );
}

fn matrix_close(a: &Matrix, b: &Matrix, rel: f64) {
    assert_eq!(a.rows(), b.rows());
    assert_eq!(a.cols(), b.cols());
    for row in 0..a.rows() {
        for col in 0..a.cols() {
            rel_close(a.get(row, col).unwrap(), b.get(row, col).unwrap(), rel);
        }
    }
}

fn product(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols(), b.rows());
    let mut out = Matrix::zeros(a.rows(), b.cols()).unwrap();
    for row in 0..a.rows() {
        for col in 0..b.cols() {
            let mut entry = 0.;
            for k in 0..a.cols() {
                entry += a.get(row, k).unwrap() * b.get(k, col).unwrap();
            }
            out.set(row, col, entry).unwrap();
        }
    }
    out
}

fn max_abs_difference(a: &Matrix, b: &Matrix) -> f64 {
    a.data()
        .iter()
        .zip(b.data())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn verify_lossless_modal_physics(plan: &CplPlan) {
    let n = plan.conductors();
    let modes = plan.modes();
    let bv = modes.voltage_basis();
    let bi = modes.current_basis();
    let y = modes.characteristic_admittance();
    let l = plan.matrix(CplMatrixKind::Inductance);
    let c = plan.matrix(CplMatrixKind::Capacitance);

    // The two distinct transforms are reciprocal transposes. A single
    // orthogonal basis is *not* sufficient when L and C do not commute.
    let power = product(&bv.transpose().unwrap(), bi);
    for row in 0..n {
        for col in 0..n {
            let expected = if row == col { 1. } else { 0. };
            assert!((power.get(row, col).unwrap() - expected).abs() < 4e-11);
        }
    }
    let yt = y.transpose().unwrap();
    matrix_close(y, &yt, 1e-12);
    // Characteristic admittance is the UNIQUE positive-definite solution
    // of Y0 * L * Y0 = C for the high-frequency lossless telegrapher system.
    let reconstructed_c = product(&product(y, l), y);
    let discrepancy = max_abs_difference(&reconstructed_c, c);
    let c_max = c.data().iter().fold(0.0_f64, |v, e| v.max(e.abs()));
    assert!(
        discrepancy < c_max * 2e-10,
        "YLY=C error {:.4e} / {:.4e}",
        discrepancy,
        c_max
    );
    for mode in 0..n {
        let admittance = modes.modal_admittances()[mode];
        let time = modes.delays()[mode];
        assert!(time.is_finite() && time > 0.);
        assert!(admittance.is_finite() && admittance > 0.);
        let mode_scale = (0..n)
            .map(|row| admittance * bi.get(row, mode).unwrap().abs())
            .fold(0.0_f64, f64::max)
            .max(1e-12);
        for row in 0..n {
            // Physical matched currents are NOT each conductor's own
            // scalar admittance: Y0 * Bv[:,k] = yk * Bi[:,k].
            let actual = (0..n)
                .map(|col| y.get(row, col).unwrap() * bv.get(col, mode).unwrap())
                .sum::<f64>();
            let expected = admittance * bi.get(row, mode).unwrap();
            assert!((actual - expected).abs() <= 1e-10 * mode_scale);
        }
    }
    // A positive definite Y0 is necessary for strictly passive wave
    // launching, even with negative physical mutual capacitances.
    assert!(
        y.symmetric_eigenvalues()
            .unwrap()
            .iter()
            .all(|value| *value > 0.)
    );
}

#[test]
fn noncommuting_two_conductor_lc_has_physical_power_normalized_modes() {
    let plan = CplPlan::prepare(
        2,
        3.,
        &[2., 0.2, 3.],
        &[4e-7, 0.6e-7, 6e-7],
        &[1e-7, 0., 2e-7],
        &[2e-10, -2e-11, 1e-10],
    )
    .unwrap();
    // Verify that the product LC is not symmetric: the entire point is
    // to avoid applying a symmetric eigensolver to LC directly.
    let l = plan.matrix(CplMatrixKind::Inductance);
    let c = plan.matrix(CplMatrixKind::Capacitance);
    let lc = product(l, c);
    let cl = product(c, l);
    assert!(max_abs_difference(&lc, &cl) > 1e-19);
    verify_lossless_modal_physics(&plan);
}

#[test]
fn three_conductor_cross_coupling_produces_offdiagonal_characteristic_loading() {
    let plan = CplPlan::prepare(
        3,
        0.5,
        &[],
        &[4e-7, 0.6e-7, 0.3e-7, 5e-7, 0.4e-7, 6e-7],
        &[],
        &[1e-10, -1e-11, -4e-12, 2e-10, -2e-11, 1.5e-10],
    )
    .unwrap();
    verify_lossless_modal_physics(&plan);
    let y = plan.modes().characteristic_admittance();
    assert!(y.get(0, 1).unwrap().abs() > 1e-6);
    assert!(y.get(1, 2).unwrap().abs() > 1e-6);
}

#[test]
fn single_wire_modal_admittance_recovers_exact_sqrt_c_over_l() {
    let plan = CplPlan::prepare(1, 2., &[], &[2e-6], &[], &[8e-12]).unwrap();
    let modes = plan.modes();
    let expected_y = (8e-12_f64 / 2e-6).sqrt();
    let expected_delay = 2. * (2e-6_f64 * 8e-12).sqrt();
    rel_close(modes.modal_admittances()[0], expected_y, 1e-12);
    rel_close(modes.delays()[0], expected_delay, 1e-12);
    rel_close(
        modes.characteristic_admittance().get(0, 0).unwrap(),
        expected_y,
        1e-12,
    );
    verify_lossless_modal_physics(&plan);
}

#[test]
fn maximum_eight_conductors_have_valid_cross_modes_without_truncation() {
    let mut l = Vec::new();
    let mut c = Vec::new();
    for row in 0..MAX_CONDUCTORS {
        for col in row..MAX_CONDUCTORS {
            l.push(if row == col { 4e-7 } else { 0.5e-8 });
            c.push(if row == col { 1e-10 } else { -1e-12 });
        }
    }
    let plan = CplPlan::prepare(MAX_CONDUCTORS, 1., &[], &l, &[], &c).unwrap();
    assert_eq!(plan.modes().delays().len(), MAX_CONDUCTORS);
    verify_lossless_modal_physics(&plan);
}
