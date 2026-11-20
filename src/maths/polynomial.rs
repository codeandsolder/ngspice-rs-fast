//! Small, bounded power-basis polynomial fits for front-end resampling.
//! This is independent of the MNA factorization and its rank/residual policy.
use crate::primitives::Real;

const MAX_COEFFICIENTS: usize = 17;

/// Fits the unique polynomial through distinct finite samples in absolute coordinates.
///
/// Uses partial-pivot Gaussian elimination; returns `None` on zero pivots,
/// nonfinite arithmetic, mismatched inputs or more than 17 coefficients.
#[must_use]
pub fn fit_power_basis(abscissas: &[Real], ordinates: &[Real]) -> Option<Vec<Real>> {
    let count = abscissas.len();
    if !(2..=MAX_COEFFICIENTS).contains(&count)
        || count != ordinates.len()
        || abscissas.iter().chain(ordinates).any(|v| !v.is_finite())
    {
        return None;
    }
    if count == 2 {
        let dx = abscissas[1] - abscissas[0];
        if dx == 0.0 || !dx.is_finite() {
            return None;
        }
        let result = vec![
            abscissas[1].mul_add(ordinates[0], -abscissas[0] * ordinates[1]) / dx,
            (ordinates[1] - ordinates[0]) / dx,
        ];
        return result.iter().all(|v| v.is_finite()).then_some(result);
    }
    let mut storage = [[0.; MAX_COEFFICIENTS]; MAX_COEFFICIENTS];
    let matrix = &mut storage[..count];
    let mut rhs = [0.; MAX_COEFFICIENTS];
    let values = &mut rhs[..count];
    values.copy_from_slice(ordinates);
    for (row, abscissa) in matrix.iter_mut().zip(abscissas) {
        let mut power = 1.;
        for value in &mut row[..count] {
            *value = power;
            power *= abscissa;
        }
    }
    for col in 0..count {
        let mut pivot = col;
        for row in col..count {
            if matrix[row][col].abs() > matrix[pivot][col].abs() {
                pivot = row;
            }
        }
        matrix.swap(col, pivot);
        values.swap(col, pivot);
        if matrix[col][col] == 0. {
            return None;
        }
        let (head, tail) = matrix.split_at_mut(col + 1);
        let pivot = &head[col][..count];
        let (before, after) = values.split_at_mut(col + 1);
        for (row, rhs) in tail.iter_mut().zip(after) {
            eliminate(&mut row[..count], pivot, col, rhs, before[col]);
        }
    }
    for col in (1..count).rev() {
        let (head, tail) = matrix.split_at_mut(col);
        let pivot = &tail[0][..count];
        let (before, after) = values.split_at_mut(col);
        for (row, rhs) in head.iter_mut().zip(before).rev() {
            eliminate(&mut row[..count], pivot, col, rhs, after[0]);
        }
    }
    let result: Vec<_> = (0..count).map(|i| values[i] / matrix[i][i]).collect();
    result.iter().all(|v| v.is_finite()).then_some(result)
}

fn eliminate(row: &mut [Real], pivot: &[Real], col: usize, rhs: &mut Real, pivot_rhs: Real) {
    let factor = row[col] / pivot[col];
    for (value, &pivot_value) in row.iter_mut().zip(pivot) {
        *value = (-factor).mul_add(pivot_value, *value);
    }
    *rhs = (-factor).mul_add(pivot_rhs, *rhs);
}

/// Evaluates coefficients ordered from constant to highest power using Horner's rule.
#[must_use]
pub fn evaluate(coefficients: &[Real], x: Real) -> Real {
    coefficients.iter().rev().fold(0., |sum, c| sum * x + c)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Frozen coefficient bits from the original elimination, independently
    // reproduced with IEEE binary64 fused multiply-add. Includes pivot ties,
    // unsorted abscissas, and the largest supported fit.
    #[test]
    fn fit_preserves_elimination_rounding() {
        let coefficients = fit_power_basis(
            &[0.125, -0.75, 1.5, 0.875, -1.25],
            &[1.25, -2.125, 0.75, 3.5, -0.625],
        )
        .unwrap();
        assert_eq!(
            coefficients.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [
                0x3fe3_dd2a_786d_089d,
                0x4014_4d60_aeb7_29af,
                0xbf9f_1828_84a5_e891,
                0xc003_7ccc_c534_ddb3,
                0x3fc4_5ddd_b30b_5a7d
            ]
        );
        let coefficients = fit_power_basis(
            &[
                -1.0, -0.875, -0.75, -0.625, -0.5, -0.375, -0.25, -0.125, 0.0, 0.125, 0.25, 0.375,
                0.5, 0.625, 0.75, 0.875, 1.0,
            ],
            &[
                -0.625, -0.5, -0.125, 0.5, 0.0, -0.25, -0.25, 0.0, 0.5, -0.125, -0.5, -0.625, -0.5,
                -0.125, 0.5, 0.0, -0.25,
            ],
        )
        .unwrap();
        assert_eq!(
            coefficients.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            [
                0x3fdf_ffff_fffc_82f4,
                0xbfe0_7ce7_ce76_9f06,
                0xc049_2875_ca68_f0c4,
                0x3ff7_669b_ac86_ec72,
                0x4090_c1d4_76e5_8022,
                0xc042_31fe_765b_e8ee,
                0xc0c5_31a9_1d92_a401,
                0x4076_8617_4018_e4a3,
                0x40eb_dc88_d3f9_b6fa,
                0xc099_e72b_67e2_4ece,
                0xc103_e057_56d1_bfd1,
                0x40ac_d835_ef84_ac78,
                0x410e_d680_0d9d_264d,
                0xc0ad_8ece_2df0_0a78,
                0xc108_4d8d_2790_6a46,
                0x4096_415b_fe74_317c,
                0x40ee_4d90_b03a_67f2
            ]
        );
    }

    #[test]
    fn fits_polynomials_and_rejects_degenerate_samples() {
        let x = [-2., -0.5, 1., 3.];
        let y = x.map(|x| 2. - 3. * x + 4. * x * x - x * x * x);
        let c = fit_power_basis(&x, &y).unwrap();
        for (a, b) in c.iter().zip([2., -3., 4., -1.]) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(fit_power_basis(&[1., 1.], &[1., 2.]).is_none());
        assert!(fit_power_basis(&[-f64::MAX, f64::MAX], &[0., 1e-300]).is_none());
        assert!(fit_power_basis(&[0., 1.], &[1.]).is_none());
    }
}
