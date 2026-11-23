//! Exact positive-real Laplace samples of the coupled telegrapher equation.
//! These tests validate the lossy fitting substrate independently of Padé.

use ngspice_rs::devices::cpl::plan::{CplPlan, MAX_CONDUCTORS};
use ngspice_rs::maths::Matrix;

fn close(actual: f64, expected: f64, relative: f64) {
    assert!(
        (actual - expected).abs() <= expected.abs().max(1e-18) * relative,
        "actual={actual:.17e}, expected={expected:.17e}, relative tolerance={relative:e}"
    );
}

#[test]
fn single_wire_matches_exact_distributed_rlgc_two_port() {
    let (r, l, g, c, length, rate) = (2., 3e-6, 5e-4, 9e-10, 0.8, 1e6);
    let plan = CplPlan::prepare(1, length, &[r], &[l], &[g], &[c]).unwrap();
    let sample = plan.laplace_sample(rate).unwrap();
    let z = r + rate * l;
    let y = g + rate * c;
    let propagation = f64::sqrt(z * y) * length;
    let admittance = f64::sqrt(y / z);
    let decay = f64::exp(-propagation);
    let denominator = -f64::exp_m1(-2. * propagation);
    close(sample.modes().modal_admittances()[0], admittance, 1e-12);
    close(sample.attenuation()[0], decay, 1e-12);
    close(
        sample.self_admittance().get(0, 0).unwrap(),
        admittance * (1. + decay * decay) / denominator,
        1e-12,
    );
    close(
        sample.cross_admittance().get(0, 0).unwrap(),
        admittance * 2. * decay / denominator,
        1e-12,
    );
    assert_eq!(sample.rate(), rate);
    // Loss-induced modal delay depends on the Laplace rate, unlike the
    // fixed high-frequency lossless propagation wavefront.
    assert!((sample.modes().delays()[0] - plan.modal_delays()[0]).abs() > 1e-10);
}

#[test]
fn kspice_clamps_every_zero_resistance_coefficient_before_laplace_fitting() {
    let l = [4e-7, 1e-7, 6e-7];
    let c = [1e-10, -2e-11, 2e-10];
    let zero = CplPlan::prepare(2, 0.5, &[], &l, &[], &c).unwrap();
    let c_effective = CplPlan::prepare(2, 0.5, &[1e-4, 1e-4, 1e-4], &l, &[], &c).unwrap();
    for rate in [1., 1e3, 1e6, 1e9, 1e12] {
        let actual = zero.laplace_sample(rate).unwrap();
        let expected = c_effective.laplace_sample(rate).unwrap();
        for row in 0..2 {
            for col in 0..2 {
                close(
                    actual.self_admittance().get(row, col).unwrap(),
                    expected.self_admittance().get(row, col).unwrap(),
                    2e-11,
                );
                close(
                    actual.cross_admittance().get(row, col).unwrap(),
                    expected.cross_admittance().get(row, col).unwrap(),
                    2e-11,
                );
            }
        }
    }
    assert_eq!(
        zero.matrix(ngspice_rs::devices::cpl::plan::CplMatrixKind::Resistance)
            .get(0, 1),
        Some(0.),
    );
}

#[test]
fn negative_mutual_series_resistance_is_clamped_for_c_reference_only() {
    use ngspice_rs::devices::cpl::plan::CplMatrixKind;
    let inductance = [4e-7, 0.5e-7, 6e-7];
    let capacitance = [1e-10, -1e-11, 2e-10];
    let raw = CplPlan::prepare(2, 1., &[1., -0.1, 1.], &inductance, &[], &capacitance).unwrap();
    let reference =
        CplPlan::prepare(2, 1., &[1., 1e-4, 1.], &inductance, &[], &capacitance).unwrap();
    for rate in [1e3, 1e6, 1e9] {
        let a = raw.laplace_sample(rate).unwrap();
        let b = reference.laplace_sample(rate).unwrap();
        for row in 0..2 {
            for col in 0..2 {
                close(
                    a.self_admittance().get(row, col).unwrap(),
                    b.self_admittance().get(row, col).unwrap(),
                    1e-11,
                );
                close(
                    a.cross_admittance().get(row, col).unwrap(),
                    b.cross_admittance().get(row, col).unwrap(),
                    1e-11,
                );
            }
        }
    }
    assert_eq!(raw.matrix(CplMatrixKind::Resistance).get(0, 1), Some(-0.1));
}

#[test]
fn noncommuting_lossy_coupling_has_symmetric_positive_two_port() {
    let plan = CplPlan::prepare(
        3,
        0.6,
        &[2., 0.4, 0.2, 3., 0.15, 4.],
        &[4e-7, 0.6e-7, 0.2e-7, 7e-7, 0.4e-7, 9e-7],
        &[0.01, -0.001, 0., 0.025, -0.002, 0.03],
        &[1e-10, -2e-11, -1e-11, 2e-10, -2e-11, 1.5e-10],
    )
    .unwrap();
    let sample = plan.laplace_sample(2e8).unwrap();
    let a = sample.self_admittance();
    let b = sample.cross_admittance();
    assert_eq!(a.rows(), 3);
    assert_eq!(b.rows(), 3);
    assert!(a.get(0, 1).unwrap().abs() > 1e-7);
    assert!(b.get(0, 1).unwrap().abs() > 1e-7);
    for row in 0..3 {
        for col in 0..3 {
            close(a.get(row, col).unwrap(), a.get(col, row).unwrap(), 1e-12);
            close(b.get(row, col).unwrap(), b.get(col, row).unwrap(), 1e-12);
        }
    }
    // The 2n two-port [A,-B;-B,A] is SPD precisely when A+B and
    // A-B are both SPD. This asserts positive-real port passivity.
    for polarity in [-1., 1.] {
        let mut block = Matrix::zeros(3, 3).unwrap();
        for row in 0..3 {
            for col in 0..3 {
                block
                    .set(
                        row,
                        col,
                        a.get(row, col).unwrap() + polarity * b.get(row, col).unwrap(),
                    )
                    .unwrap();
            }
        }
        assert!(
            block
                .symmetric_eigenvalues()
                .unwrap()
                .iter()
                .all(|&eigen| eigen > 0.)
        );
    }
}

#[test]
fn high_laplace_rate_approaches_coupled_lossless_wavefront_and_y0() {
    let plan = CplPlan::prepare(
        2,
        0.5,
        &[2., 0.25, 3.],
        &[4e-7, 0.6e-7, 6e-7],
        &[0.01, -0.002, 0.02],
        &[1e-10, -1e-11, 2e-10],
    )
    .unwrap();
    let sample = plan.laplace_sample(1e14).unwrap();
    for (actual, &expected) in sample.modes().delays().iter().zip(plan.modal_delays()) {
        close(*actual, expected, 1e-5);
    }
    for row in 0..2 {
        for col in 0..2 {
            close(
                sample.self_admittance().get(row, col).unwrap(),
                plan.modes()
                    .characteristic_admittance()
                    .get(row, col)
                    .unwrap(),
                2e-5,
            );
            assert_eq!(sample.cross_admittance().get(row, col), Some(0.));
        }
    }
}

#[test]
fn weakly_damped_low_rate_approaches_dc_series_resistance() {
    let resistance = 2.5;
    let length = 2.;
    let plan = CplPlan::prepare(1, length, &[resistance], &[1e-6], &[], &[1e-9]).unwrap();
    let sample = plan.laplace_sample(1e-3).unwrap();
    close(
        sample.self_admittance().get(0, 0).unwrap(),
        1. / (resistance * length),
        1e-8,
    );
    close(
        sample.cross_admittance().get(0, 0).unwrap(),
        1. / (resistance * length),
        1e-8,
    );
    assert!(sample.cross_admittance().get(0, 0).unwrap() > 0.);
}

#[test]
fn all_eight_conductors_keep_modal_and_cross_coupling_without_truncation() {
    let mut l = Vec::new();
    let mut c = Vec::new();
    let mut r = Vec::new();
    let mut g = Vec::new();
    for row in 0..MAX_CONDUCTORS {
        for col in row..MAX_CONDUCTORS {
            l.push(if row == col { 4e-7 } else { 0.5e-8 });
            c.push(if row == col { 1e-10 } else { -1e-12 });
            r.push(if row == col { 2. } else { 0.02 });
            g.push(if row == col { 0.002 } else { -0.00001 });
        }
    }
    let plan = CplPlan::prepare(MAX_CONDUCTORS, 1., &r, &l, &g, &c).unwrap();
    let sample = plan.laplace_sample(1e8).unwrap();
    assert_eq!(sample.attenuation().len(), MAX_CONDUCTORS);
    assert!(
        sample
            .attenuation()
            .iter()
            .all(|&value| (0. ..=1.).contains(&value))
    );
    assert!(sample.self_admittance().get(0, 7).unwrap().abs() > 1e-8);
}

#[test]
fn rejects_nonphysical_sampling_rates_and_overflowing_coefficient_scales() {
    let plan = CplPlan::prepare(1, 1., &[3.], &[1e-6], &[0.], &[1e-9]).unwrap();
    for rate in [0., -1., Real::INFINITY, Real::NAN] {
        assert!(plan.laplace_sample(rate).is_err());
    }
    assert!(plan.laplace_sample(1e-320).is_err());
}

use ngspice_rs::primitives::Real;
