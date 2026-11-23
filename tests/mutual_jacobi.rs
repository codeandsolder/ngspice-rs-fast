//! Analytic symmetric spectra: the baseline rejected dense physically valid
//! coupling despite strict positive-definiteness and moderate conditioning.
use ngspice_rs::maths::Matrix;

fn hadamard_spectrum(order: usize, eigenvalues: &[f64], scale: f64) -> Matrix {
    assert!(order.is_power_of_two());
    assert_eq!(order, eigenvalues.len());
    let mut matrix = Matrix::zeros(order, order).unwrap();
    for row in 0..order {
        for col in 0..=row {
            let total: f64 = eigenvalues
                .iter()
                .enumerate()
                .map(|(index, eigenvalue)| {
                    if ((row ^ col) & index).count_ones().is_multiple_of(2) {
                        *eigenvalue
                    } else {
                        -*eigenvalue
                    }
                })
                .sum();
            let value = total / f64::from(u32::try_from(order).unwrap()) * scale;
            matrix.set(row, col, value).unwrap();
            matrix.set(col, row, value).unwrap();
        }
    }
    matrix
}

#[test]
fn orthogonal_hadamard_spectra_are_recovered_for_spd_psd_and_indefinite() {
    for order in [32_usize, 64, 128] {
        let spectrum: Vec<f64> = (0..order)
            .map(|index| {
                let permutation = (13 * index) % order;
                1. + 2.0_f64.mul_add(
                    f64::from(u32::try_from(permutation).unwrap()),
                    -f64::from(u32::try_from(order - 1).unwrap()),
                ) / (2. * f64::from(u32::try_from(order).unwrap()))
            })
            .collect();
        for mode in 0..3 {
            let mut eigenvalues = spectrum.clone();
            match mode {
                1 => eigenvalues[0] = 0.,
                2 => eigenvalues[0] = -0.5,
                _ => {}
            }
            for scale in [1., 2.0_f64.powi(400), 2.0_f64.powi(-400)] {
                let matrix = hadamard_spectrum(order, &eigenvalues, scale);
                let actual = matrix.symmetric_eigenvalues().unwrap();
                let mut expected = eigenvalues.clone();
                expected.sort_unstable_by(f64::total_cmp);
                let bound = 32. * f64::EPSILON * f64::from(u32::try_from(order).unwrap());
                for (&found, want) in actual.iter().zip(expected) {
                    assert!(
                        (found / scale - want).abs() <= bound,
                        "order {order}, mode {mode}, scale {scale}, expected {want}, got {found}"
                    );
                }
            }
        }
    }
}

#[test]
fn analytic_normalized_coupling_is_strictly_positive_definite() {
    let order = 64_usize;
    let expected: Vec<f64> = (0..order)
        .map(|k| {
            1. + 2.0_f64.mul_add(f64::from(u32::try_from((13 * k) % order).unwrap()), -63.) / 128.
        })
        .collect();
    let matrix = hadamard_spectrum(order, &expected, 1.);
    for index in 0..order {
        assert_eq!(matrix.get(index, index), Some(1.));
    }
    let found = matrix.symmetric_eigenvalues().unwrap();
    assert!((found[0] - 0.507_812_5).abs() <= 1e-13);
    assert!((found[order - 1] - 1.492_187_5).abs() <= 1e-13);
}
