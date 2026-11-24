//! Independent analytic distributed multiconductor transmission-line checks.
//!
//! Includes differential/common eigenmodes, actual cross-conductor coupling,
//! zero-frequency resistive coupling, and explicit invalid model rejection.
use std::sync::Arc;

use ngspice_rs::devices::MnaUnknowns;
use ngspice_rs::devices::cpl::CoupledLine;
use ngspice_rs::devices::cpl::plan::{CoupledLineLimits, CoupledLinePlan};
use ngspice_rs::maths::Matrix;
use ngspice_rs::primitives::{Complex, NodeId, NodeTable};

fn model(n: usize, length: f64, r: &[f64], l: &[f64], g: &[f64], c: &[f64]) -> CoupledLinePlan {
    CoupledLinePlan::from_packed(n, length, r, l, g, c, CoupledLineLimits::default()).unwrap()
}
#[track_caller]
fn near(actual: Complex, expected: Complex, relative: f64) {
    let scale = expected.magnitude().max(1e-3);
    let error = (actual - expected).magnitude();
    assert!(
        error <= relative * scale,
        "got {actual:?} instead of {expected:?}, absolute error={error}, relative={}",
        error / scale
    );
}
fn analytic_lossless(l: f64, c: f64, length: f64, hz: f64) -> (Complex, Complex) {
    let phase = std::f64::consts::TAU * hz * length * (l * c).sqrt();
    let y0 = (c / l).sqrt();
    (
        Complex::imaginary(-phase.cos() / phase.sin() * y0),
        Complex::imaginary(phase.sin().recip() * y0),
    )
}

#[test]
fn scalar_lossless_line_matches_exact_trigonometric_solution() {
    let line = model(1, 1., &[0.], &[8e-9], &[0.], &[3e-12]);
    for hz in [100e6, 300e6, 1e9, 1.5e9] {
        let y = line.admittance_at(hz).unwrap();
        let (diagonal, far) = analytic_lossless(8e-9, 3e-12, 1., hz);
        near(y.get(0, 0).unwrap(), diagonal, 2e-11);
        near(y.get(0, 1).unwrap(), far, 2e-11);
        near(y.get(1, 0).unwrap(), far, 2e-11);
        near(y.get(1, 1).unwrap(), diagonal, 2e-11);
    }
}

#[test]
fn dc_series_resistance_couples_both_ends_correctly() {
    // A positive definite R and zero G give Ydc = [[K,-K],[-K,K]],
    // K = inverse(R*length). This must hold even if L,C are coupled.
    let length = 2.;
    let plan = model(
        2,
        length,
        &[3., 1., 4.],
        &[9e-9, 2e-9, 5e-9],
        &[0., 0., 0.],
        &[4e-12, -0.5e-12, 3e-12],
    );
    let y = plan.admittance_at(0.).unwrap();
    let inv_det = 1. / (3. * 4. - 1.);
    let expected = [[4. * inv_det, -inv_det], [-inv_det, 3. * inv_det]];
    for (i, row) in expected.iter().enumerate() {
        for (j, &entry) in row.iter().enumerate() {
            let physical = entry / length;
            near(y.get(i, j).unwrap(), Complex::real(physical), 2e-12);
            near(y.get(i, j + 2).unwrap(), Complex::real(-physical), 2e-12);
            near(y.get(i + 2, j).unwrap(), Complex::real(-physical), 2e-12);
            near(y.get(i + 2, j + 2).unwrap(), Complex::real(physical), 2e-12);
        }
    }
}

#[test]
fn coupled_2line_even_odd_modes_match_independent_orthonormal_transform() {
    // U[:,0]=(1,1)/sqrt(2); U[:,1]=(1,-1)/sqrt(2).
    // L_even=8nH, L_odd=4nH, C_even=2.5pF, C_odd=3.5pF.
    let line = model(
        2,
        1.,
        &[0., 0., 0.],
        &[6e-9, 2e-9, 6e-9],
        &[0., 0., 0.],
        &[3e-12, -0.5e-12, 3e-12],
    );
    for hz in [50e6, 300e6, 1e9, 1.5e9] {
        let matrix = line.admittance_at(hz).unwrap();
        let (even_d, even_o) = analytic_lossless(8e-9, 2.5e-12, 1., hz);
        let (odd_d, odd_o) = analytic_lossless(4e-9, 3.5e-12, 1., hz);
        let expected_diag = (even_d + odd_d) * Complex::real(0.5);
        let expected_cross = (even_d - odd_d) * Complex::real(0.5);
        let expected_far_diag = (even_o + odd_o) * Complex::real(0.5);
        let expected_far_cross = (even_o - odd_o) * Complex::real(0.5);
        for (a, b, target) in [
            (0, 0, expected_diag),
            (1, 1, expected_diag),
            (0, 1, expected_cross),
            (1, 0, expected_cross),
            (2, 2, expected_diag),
            (3, 3, expected_diag),
            (2, 3, expected_cross),
            (3, 2, expected_cross),
            (0, 2, expected_far_diag),
            (1, 3, expected_far_diag),
            (0, 3, expected_far_cross),
            (1, 2, expected_far_cross),
            (2, 0, expected_far_diag),
            (3, 1, expected_far_diag),
            (2, 1, expected_far_cross),
            (3, 0, expected_far_cross),
        ] {
            near(matrix.get(a, b).unwrap(), target, 3e-10);
        }
    }
}

#[test]
fn uncoupled_multiple_conductors_do_not_generate_false_crosstalk() {
    let plan = model(
        3,
        0.7,
        &[0.25, 0., 0., 0.5, 0., 0.75],
        &[4e-9, 0., 0., 6e-9, 0., 8e-9],
        &[0., 0., 0., 0., 0., 0.],
        &[2e-12, 0., 0., 3e-12, 0., 4e-12],
    );
    let y = plan.admittance_at(1e9).unwrap();
    for i in 0..6 {
        for j in 0..6 {
            if i % 3 != j % 3 {
                near(y.get(i, j).unwrap(), Complex::ZERO, 2e-10);
            }
        }
    }
}

#[test]
fn model_rejects_nonphysical_singular_and_oversized_materials() {
    let limit = CoupledLineLimits::default();
    assert!(CoupledLinePlan::from_packed(0, 1., &[], &[], &[], &[], limit).is_err());
    assert!(
        CoupledLinePlan::from_packed(
            2,
            1.,
            &[1., 0.],
            &[1., 0., 1.],
            &[0., 0., 0.],
            &[1., 0., 1.],
            limit
        )
        .is_err()
    );
    assert!(CoupledLinePlan::from_packed(1, 0., &[1.], &[1.], &[0.], &[1.], limit).is_err());
    assert!(CoupledLinePlan::from_packed(1, f64::NAN, &[1.], &[1.], &[0.], &[1.], limit).is_err());
    assert!(CoupledLinePlan::from_packed(1, 1., &[-1.], &[1.], &[0.], &[1.], limit).is_err());
    assert!(CoupledLinePlan::from_packed(1, 1., &[1.], &[0.], &[0.], &[1.], limit).is_err());
    assert!(CoupledLinePlan::from_packed(1, 1., &[1.], &[1.], &[-1.], &[1.], limit).is_err());
    assert!(CoupledLinePlan::from_packed(1, 1., &[1.], &[1.], &[0.], &[0.], limit).is_err());
    assert!(
        CoupledLinePlan::from_packed(
            1,
            1.,
            &[1.],
            &[1.],
            &[0.],
            &[1.],
            CoupledLineLimits {
                max_conductors: 0,
                ..limit
            }
        )
        .is_err()
    );
    assert!(
        CoupledLinePlan::from_packed(
            1,
            1.,
            &[1.],
            &[1.],
            &[0.],
            &[1.],
            CoupledLineLimits {
                max_workspace_bytes: 0,
                ..limit
            }
        )
        .is_err()
    );
    assert!(
        CoupledLinePlan::from_packed(1, 1., &[1.], &[1.], &[0.], &[1.], limit)
            .unwrap()
            .admittance_at(f64::MAX)
            .is_err()
    );
    let mut asymmetric = Matrix::zeros(2, 2).unwrap();
    asymmetric.set(0, 0, 3.).unwrap();
    asymmetric.set(1, 1, 3.).unwrap();
    asymmetric.set(0, 1, 1.).unwrap();
    asymmetric.set(1, 0, 0.).unwrap();
    let l = CoupledLinePlan::unpack_symmetric(2, &[8e-9, 0., 8e-9], limit).unwrap();
    let g = CoupledLinePlan::unpack_symmetric(2, &[0., 0., 0.], limit).unwrap();
    let c = CoupledLinePlan::unpack_symmetric(2, &[3e-12, 0., 3e-12], limit).unwrap();
    assert!(CoupledLinePlan::new(1., asymmetric, l, g, c, limit).is_err());
}

#[test]
fn singular_ideal_dc_line_is_not_silently_replaced_with_an_open() {
    let line = model(1, 0.5, &[0.], &[1e-6], &[0.], &[1e-9]);
    assert!(line.admittance_at(0.).is_err());
    assert!(line.admittance_at(1e6).is_ok());
}

#[test]
fn distinct_input_and_output_return_nodes_stamp_exact_kcl_signs() {
    let line = Arc::new(model(
        2,
        2.,
        &[3., 0., 4.],
        &[9e-9, 0., 5e-9],
        &[0., 0., 0.],
        &[4e-12, 0., 3e-12],
    ));
    let device = CoupledLine::new(
        "p1",
        line,
        vec![NodeId::new(1), NodeId::new(2)],
        NodeId::new(3),
        vec![NodeId::new(4), NodeId::new(5)],
        NodeId::new(6),
    )
    .unwrap();
    let mut nodes = NodeTable::new();
    for i in 1..=6 {
        nodes.intern(&format!("node{i}")).unwrap();
    }
    let mut mna = MnaUnknowns::new();
    mna.rebuild(&nodes);
    let stamps = device.nodal_stamps_at(0., &mna).unwrap();
    let mut dense = [[0.; 6]; 6];
    for (row, col, value) in stamps {
        assert_eq!(value.im, 0.);
        dense[row][col] += value.re;
    }
    // Every row and column sums to zero under a global voltage shift:
    // returns are not silently replaced by ground.
    for row in &dense {
        near(Complex::real(row.iter().sum()), Complex::ZERO, 2e-10);
    }
    for col in 0..6 {
        let sum = dense.iter().map(|row| row[col]).sum::<f64>();
        near(Complex::real(sum), Complex::ZERO, 2e-10);
    }
    near(Complex::real(dense[0][0]), Complex::real(1. / 6.), 2e-12);
    near(Complex::real(dense[0][2]), Complex::real(-1. / 6.), 2e-12);
    near(Complex::real(dense[0][3]), Complex::real(-1. / 6.), 2e-12);
    near(Complex::real(dense[0][5]), Complex::real(1. / 6.), 2e-12);
}
