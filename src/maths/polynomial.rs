//! Small, bounded power-basis polynomial fits for front-end resampling.
//! This is independent of the MNA factorization and its rank/residual policy.
use crate::primitives::Real;

const MAX_COEFFICIENTS: usize = 17;

/// Fits the unique polynomial through distinct finite samples in absolute coordinates.
///
/// Uses partial-pivot Gaussian elimination; returns `None` on zero pivots,
/// nonfinite arithmetic, mismatched inputs or more than 17 coefficients.
#[must_use]
pub fn fit_power_basis(x: &[Real], y: &[Real]) -> Option<Vec<Real>> {
    let n = x.len();
    if !(2..=MAX_COEFFICIENTS).contains(&n)
        || n != y.len()
        || x.iter().chain(y).any(|v| !v.is_finite())
    {
        return None;
    }
    if n == 2 {
        let dx = x[1] - x[0];
        let result = vec![x[1].mul_add(y[0], -x[0] * y[1]) / dx, (y[1] - y[0]) / dx];
        return result.iter().all(|v| v.is_finite()).then_some(result);
    }
    let mut storage = [[0.; MAX_COEFFICIENTS]; MAX_COEFFICIENTS];
    let a = &mut storage[..n];
    let mut rhs = [0.; MAX_COEFFICIENTS];
    let b = &mut rhs[..n];
    b.copy_from_slice(y);
    for (row, t) in a.iter_mut().zip(x) {
        let mut power = 1.;
        for value in &mut row[..n] {
            *value = power;
            power *= t;
        }
    }
    for col in 0..n {
        let mut pivot = col;
        for row in col..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        if a[col][col] == 0. {
            return None;
        }
        let (head, tail) = a.split_at_mut(col + 1);
        let pivot = &head[col][..n];
        let (before, after) = b.split_at_mut(col + 1);
        for (row, rhs) in tail.iter_mut().zip(after) {
            eliminate(&mut row[..n], pivot, col, rhs, before[col]);
        }
    }
    for col in (1..n).rev() {
        let (head, tail) = a.split_at_mut(col);
        let pivot = &tail[0][..n];
        let (before, after) = b.split_at_mut(col);
        for (row, rhs) in head.iter_mut().zip(before).rev() {
            eliminate(&mut row[..n], pivot, col, rhs, after[0]);
        }
    }
    let result: Vec<_> = (0..n).map(|i| b[i] / a[i][i]).collect();
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
        assert!(fit_power_basis(&[0., 1.], &[1.]).is_none());
    }
}
