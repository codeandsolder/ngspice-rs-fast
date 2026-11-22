//! Small dense complex matrices for port-level algebra.
//!
//! Mirrors the `CMat` helpers of `src/maths/dense/dense.c` that the `.sp`
//! analysis uses to turn incident/scattered power waves into S, Y and Z
//! matrices (`cmultiply`, `csum`, `cminus`, `ceye`, `cinverse`). These are
//! `N x N` matrices for `N` RF ports, never the MNA system, so a plain
//! row-major store with Gauss-Jordan inversion (partial pivoting, as C's
//! `cinv_gj`) is adequate; the MNA solves stay with the SPARSE LU of
//! [`crate::maths::complex`].

use super::dense::DenseShape;
use crate::primitives::{Complex, SpiceError, SpiceResult, filled_vec};

fn numerical(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "dense complex matrix".to_owned(),
        message: message.into(),
    }
}

/// A dense, row-major, square complex matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseComplex {
    shape: DenseShape,
    data: Vec<Complex>,
}

impl DenseComplex {
    /// The zero matrix.
    /// # Errors
    /// Dimensions overflow or storage cannot be allocated.
    pub fn zeros(n: usize) -> SpiceResult<Self> {
        let shape = DenseShape::checked::<Complex>(n, n, "dense complex matrix")?;
        Ok(Self {
            shape,
            data: filled_vec(shape.len(), Complex::ZERO, "dense complex matrix")?,
        })
    }

    /// The identity matrix (`ceye`).
    /// # Errors
    /// Dimensions overflow or storage cannot be allocated.
    pub fn identity(n: usize) -> SpiceResult<Self> {
        let mut matrix = Self::zeros(n)?;
        for i in 0..n {
            matrix.data[i * n + i] = Complex::real(1.);
        }
        Ok(matrix)
    }

    /// A diagonal matrix.
    /// # Errors
    /// Dimensions overflow or storage cannot be allocated.
    pub fn diagonal(values: &[Complex]) -> SpiceResult<Self> {
        let mut matrix = Self::zeros(values.len())?;
        for (i, value) in values.iter().enumerate() {
            matrix.data[i * values.len() + i] = *value;
        }
        Ok(matrix)
    }

    /// The dimension `n`.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.shape.rows()
    }

    /// The entry at `(row, col)`, or `None` out of range.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<Complex> {
        self.shape.offset(row, col).map(|offset| self.data[offset])
    }

    /// Sets the entry at `(row, col)`.
    /// # Errors
    /// An index out of range.
    pub fn set(&mut self, row: usize, col: usize, value: Complex) -> SpiceResult<()> {
        let offset = self.shape.offset(row, col).ok_or_else(|| {
            numerical(format!(
                "index ({row}, {col}) is out of range for a {0}x{0} matrix",
                self.dim()
            ))
        })?;
        self.data[offset] = value;
        Ok(())
    }

    /// The entries in row-major order.
    #[must_use]
    pub fn data(&self) -> &[Complex] {
        &self.data
    }

    fn same_dimension(&self, other: &Self) -> SpiceResult<()> {
        if self.shape == other.shape {
            Ok(())
        } else {
            Err(numerical(format!(
                "dimension mismatch {0}x{0} against {1}x{1}",
                self.dim(),
                other.dim()
            )))
        }
    }

    fn check_finite(&self) -> SpiceResult<()> {
        if self.data.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err(numerical("non-finite result"))
        }
    }

    pub(crate) fn clear(&mut self) {
        self.data.fill(Complex::ZERO);
    }

    /// `self + other` (`csum`).
    /// # Errors
    /// Mismatched dimensions, a non-finite result, or unavailable storage.
    pub fn add(&self, other: &Self) -> SpiceResult<Self> {
        let mut out = Self::zeros(self.dim())?;
        self.add_into(other, &mut out)?;
        Ok(out)
    }

    pub(crate) fn add_into(&self, other: &Self, out: &mut Self) -> SpiceResult<()> {
        self.same_dimension(other)?;
        self.same_dimension(out)?;
        for ((value, a), b) in out.data.iter_mut().zip(&self.data).zip(&other.data) {
            *value = *a + *b;
        }
        out.check_finite()
    }

    /// `self - other` (`cminus`).
    /// # Errors
    /// Mismatched dimensions, a non-finite result, or unavailable storage.
    pub fn sub(&self, other: &Self) -> SpiceResult<Self> {
        let mut out = Self::zeros(self.dim())?;
        self.sub_into(other, &mut out)?;
        Ok(out)
    }

    pub(crate) fn sub_into(&self, other: &Self, out: &mut Self) -> SpiceResult<()> {
        self.same_dimension(other)?;
        self.same_dimension(out)?;
        for ((value, a), b) in out.data.iter_mut().zip(&self.data).zip(&other.data) {
            *value = *a - *b;
        }
        out.check_finite()
    }

    /// `self * other` (`cmultiply`).
    /// # Errors
    /// Mismatched dimensions, a non-finite result, or unavailable storage.
    pub fn mul(&self, other: &Self) -> SpiceResult<Self> {
        let mut out = Self::zeros(self.dim())?;
        self.mul_into(other, &mut out)?;
        Ok(out)
    }

    pub(crate) fn mul_into(&self, other: &Self, out: &mut Self) -> SpiceResult<()> {
        self.same_dimension(other)?;
        self.same_dimension(out)?;
        // Skipping a zero multiplier is valid only for finite operands.
        if self.data.iter().chain(&other.data).any(|v| !v.is_finite()) {
            return Err(numerical("non-finite entry"));
        }
        out.clear();
        let n = self.dim();
        if n != 0 {
            for (a_row, out_row) in self.data.chunks_exact(n).zip(out.data.chunks_exact_mut(n)) {
                for (&a, b_row) in a_row.iter().zip(other.data.chunks_exact(n)) {
                    if a == Complex::ZERO {
                        continue;
                    }
                    for (out, &b) in out_row.iter_mut().zip(b_row) {
                        *out = *out + a * b;
                    }
                }
            }
        }
        out.check_finite()
    }

    /// Right diagonal product in O(n²), equivalent to the finite generic
    /// product's one nonzero term. Accumulate from positive zero even
    /// for underflowed/signed-zero products.
    pub(crate) fn right_diagonal_into(
        &self,
        diagonal: &[Complex],
        out: &mut Self,
    ) -> SpiceResult<()> {
        self.check_diagonal(diagonal, out)?;
        let n = self.dim().max(1);
        for (input, output) in self.data.chunks_exact(n).zip(out.data.chunks_exact_mut(n)) {
            for ((&a, &d), value) in input.iter().zip(diagonal).zip(output) {
                *value = if a == Complex::ZERO {
                    Complex::ZERO
                } else {
                    Complex::ZERO + a * d
                };
            }
        }
        out.check_finite()
    }

    /// Left diagonal product with the original operand order.
    pub(crate) fn left_diagonal_into(
        &self,
        diagonal: &[Complex],
        out: &mut Self,
    ) -> SpiceResult<()> {
        self.check_diagonal(diagonal, out)?;
        let n = self.dim().max(1);
        for ((input, output), &d) in self
            .data
            .chunks_exact(n)
            .zip(out.data.chunks_exact_mut(n))
            .zip(diagonal)
        {
            for (&a, value) in input.iter().zip(output) {
                *value = if d == Complex::ZERO {
                    Complex::ZERO
                } else {
                    Complex::ZERO + d * a
                };
            }
        }
        out.check_finite()
    }

    fn check_diagonal(&self, diagonal: &[Complex], out: &Self) -> SpiceResult<()> {
        self.same_dimension(out)?;
        if diagonal.len() != self.dim() {
            return Err(numerical("diagonal dimension mismatch"));
        }
        if self.data.iter().chain(diagonal).any(|v| !v.is_finite()) {
            return Err(numerical("non-finite entry"));
        }
        Ok(())
    }

    /// The inverse by Gauss-Jordan elimination with partial pivoting, or
    /// `None` when the matrix is singular to working precision: a pivot no
    /// larger than `128 n eps` times the largest entry magnitude (the
    /// threshold of the port's complex LU rank guard). C's `cinv_gj` only
    /// stops at an exactly zero pivot, so rounding-level pivots of a
    /// mathematically singular matrix give it huge, rounding-dependent
    /// entries instead.
    /// # Errors
    /// A non-finite entry or result, or unavailable workspace storage.
    pub fn inverse(&self) -> SpiceResult<Option<Self>> {
        let mut out = Self::zeros(self.dim())?;
        let mut scratch = Self::zeros(self.dim())?;
        if self.inverse_into(&mut out, &mut scratch)? {
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }

    /// Invert into two shape-checked reusable buffers. False denotes
    /// a singular input; the output contents are unspecified in that case.
    pub(crate) fn inverse_into(&self, out: &mut Self, scratch: &mut Self) -> SpiceResult<bool> {
        self.same_dimension(out)?;
        self.same_dimension(scratch)?;
        let n = self.dim();
        if self.data.iter().any(|value| !value.is_finite()) {
            return Err(numerical("non-finite entry"));
        }
        let a = &mut scratch.data;
        a.copy_from_slice(&self.data);
        let max_component = a
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.re.abs()).max(v.im.abs()));
        let normalization = super::dense::extreme_norm_scale(max_component);
        if let Some(scale) = normalization {
            for value in &mut *a {
                value.re /= scale;
                value.im /= scale;
            }
        }
        let scale = a.iter().fold(0_f64, |m, value| m.max(value.magnitude()));
        if scale == 0. {
            return Ok(n == 0);
        }
        // A real allocation of n*n complex elements bounds n well below
        // u32::MAX on the supported address spaces. Check before converting.
        let dimension =
            u32::try_from(n).map_err(|_| numerical("matrix dimension exceeds rank guard range"))?;
        let threshold = 128. * f64::EPSILON * f64::from(dimension) * scale;
        out.clear();
        for i in 0..n {
            out.data[i * n + i] = Complex::real(1.);
        }
        let b = &mut out.data;
        for c in 0..n {
            let pivot_row = (c..n)
                .max_by(|&x, &y| {
                    a[x * n + c]
                        .magnitude()
                        .total_cmp(&a[y * n + c].magnitude())
                })
                .unwrap_or(c);
            if a[pivot_row * n + c].magnitude() <= threshold {
                return Ok(false);
            }
            if pivot_row != c {
                swap_rows(a, n, c, pivot_row);
                swap_rows(b, n, c, pivot_row);
            }
            let (a_before, a_pivot, a_after) = split_row(a, n, c);
            let (b_before, b_pivot, b_after) = split_row(b, n, c);
            let inverse_pivot = Complex::real(1.) / a_pivot[c];
            for (a_value, b_value) in a_pivot.iter_mut().zip(b_pivot.iter_mut()) {
                *a_value = *a_value * inverse_pivot;
                *b_value = *b_value * inverse_pivot;
            }
            let a_rows = a_before
                .chunks_exact_mut(n)
                .chain(a_after.chunks_exact_mut(n));
            let b_rows = b_before
                .chunks_exact_mut(n)
                .chain(b_after.chunks_exact_mut(n));
            for (a_row, b_row) in a_rows.zip(b_rows) {
                let factor = a_row[c];
                if factor == Complex::ZERO {
                    continue;
                }
                for ((a_value, b_value), (&a_p, &b_p)) in a_row
                    .iter_mut()
                    .zip(b_row)
                    .zip(a_pivot.iter().zip(b_pivot.iter()))
                {
                    *a_value = *a_value - factor * a_p;
                    *b_value = *b_value - factor * b_p;
                }
            }
        }
        // If A was scaled down by s, A'^{-1} = s A^{-1}.
        // Undo that scale in the returned inverse, not the pivot policy.
        if let Some(scale) = normalization {
            for value in &mut *b {
                value.re /= scale;
                value.im /= scale;
            }
        }
        out.check_finite()?;
        Ok(true)
    }
}

/// Disjoint rows around the pivot, preserving row-major traversal order.
const fn split_row<T>(data: &mut [T], n: usize, row: usize) -> (&mut [T], &mut [T], &mut [T]) {
    let (before, tail) = data.split_at_mut(row * n);
    let (pivot, after) = tail.split_at_mut(n);
    (before, pivot, after)
}

fn swap_rows<T>(data: &mut [T], n: usize, first: usize, second: usize) {
    debug_assert!(first < second);
    let (before, after) = data.split_at_mut(second * n);
    before[first * n..(first + 1) * n].swap_with_slice(&mut after[..n]);
}

#[cfg(test)]
mod tests {
    use super::DenseComplex;
    use crate::primitives::Complex;

    fn matrix(n: usize, values: &[(f64, f64)]) -> DenseComplex {
        let mut m = DenseComplex::zeros(n).unwrap();
        for (index, (re, im)) in values.iter().enumerate() {
            m.set(index / n, index % n, Complex::new(*re, *im)).unwrap();
        }
        m
    }

    #[test]
    fn zero_multiplier_does_not_hide_nonfinite_input() {
        let a = DenseComplex::zeros(1).unwrap();
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut b = DenseComplex::zeros(1).unwrap();
            b.set(0, 0, Complex::real(invalid)).unwrap();
            assert!(a.mul(&b).is_err());
            assert!(b.mul(&a).is_err());
        }
    }

    #[test]
    fn zero_order_product_and_row_accumulation() {
        assert_eq!(
            DenseComplex::zeros(0)
                .unwrap()
                .mul(&DenseComplex::zeros(0).unwrap())
                .unwrap(),
            DenseComplex::zeros(0).unwrap()
        );
        // Preserve k order across cancellation, even with a sparse row.
        let a = matrix(
            3,
            &[
                (1e16, 0.),
                (-1e16, 0.),
                (1., 0.),
                (0., 0.),
                (0., 0.),
                (0., 0.),
                (0., 0.),
                (0., 0.),
                (0., 0.),
            ],
        );
        let b = matrix(3, &[(1., 0.); 9]);
        let out = a.mul(&b).unwrap();
        assert_eq!(&out.data()[..3], &[Complex::real(1.); 3]);
    }

    #[test]
    fn dimensions_cannot_wrap() {
        assert!(DenseComplex::zeros(usize::MAX).is_err());
        assert!(DenseComplex::zeros(usize::MAX.isqrt()).is_err());
    }

    #[test]
    fn inverse_times_matrix_is_identity_with_pivoting() {
        // A zero leading entry needs a row exchange.
        let a = matrix(
            3,
            &[
                (0., 0.),
                (2., 1.),
                (1., 0.),
                (1., -1.),
                (0., 0.),
                (3., 0.),
                (4., 0.),
                (1., 1.),
                (0., 2.),
            ],
        );
        let inverse = a.inverse().unwrap().expect("nonsingular");
        let product = a.mul(&inverse).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1. } else { 0. };
                let value = product.get(i, j).unwrap();
                assert!((value.re - expected).abs() < 1e-14, "{i},{j}: {value}");
                assert!(value.im.abs() < 1e-14, "{i},{j}: {value}");
            }
        }
    }

    #[test]
    fn inverse_normalizes_extreme_complex_pivots() {
        for scale in [1e-300, 1e300, 1e308] {
            let a =
                DenseComplex::diagonal(&[Complex::new(scale, scale), Complex::new(scale, -scale)])
                    .unwrap();
            let inverse = a
                .inverse()
                .unwrap()
                .expect("well-conditioned even at extreme scale");
            let identity = a.mul(&inverse).unwrap();
            for i in 0..2 {
                for j in 0..2 {
                    let expected = if i == j { 1. } else { 0. };
                    let v = identity.get(i, j).unwrap();
                    assert!((v.re - expected).abs() < 1e-14 && v.im.abs() < 1e-14);
                }
            }
        }
        assert!(
            DenseComplex::diagonal(&[Complex::real(f64::from_bits(1))])
                .unwrap()
                .inverse()
                .is_err()
        );
    }

    #[test]
    fn singular_matrices_have_no_inverse() {
        let exact = matrix(2, &[(1., 0.), (-1., 0.), (-1., 0.), (1., 0.)]);
        assert_eq!(exact.inverse().unwrap(), None);
        // Singular up to rounding, as E - S of a series resistor.
        let third = 1. / 3.;
        let rounded = matrix(
            2,
            &[
                (1. - third, 0.),
                (-(1. - third) + 1e-17, 0.),
                (-(1. - third), 0.),
                (1. - third, 0.),
            ],
        );
        assert_eq!(rounded.inverse().unwrap(), None);
        assert_eq!(DenseComplex::zeros(2).unwrap().inverse().unwrap(), None);
        assert_eq!(
            DenseComplex::zeros(0).unwrap().inverse().unwrap(),
            Some(DenseComplex::zeros(0).unwrap())
        );
    }

    #[test]
    fn arithmetic_checks_dimensions_and_finiteness() {
        let two = DenseComplex::identity(2).unwrap();
        let three = DenseComplex::identity(3).unwrap();
        assert!(two.add(&three).is_err());
        assert!(two.sub(&three).is_err());
        assert!(two.mul(&three).is_err());
        assert!(two.clone().set(2, 0, Complex::ZERO).is_err());
        assert_eq!(two.get(2, 0), None);
        let huge = DenseComplex::diagonal(&[Complex::real(f64::MAX); 2]).unwrap();
        assert!(huge.add(&huge).is_err());
        let nan = DenseComplex::diagonal(&[Complex::real(f64::NAN); 2]).unwrap();
        assert!(nan.inverse().is_err());
        let sum = two.add(&two).unwrap().sub(&two).unwrap();
        assert_eq!(sum, two);
        assert_eq!(sum.dim(), 2);
        assert_eq!(sum.data().len(), 4);
    }
    #[test]
    fn diagonal_kernels_preserve_generic_product_bits_and_errors() {
        let values = [
            (0., -0.),
            (-0., 0.),
            (-0., -0.),
            (f64::from_bits(1), -f64::from_bits(2)),
            (-2., 3.),
            (1e150, -1e-150),
            (1e-150, 1e150),
            (-1., -1.),
            (4., 0.),
        ];
        let a = matrix(3, &values);
        let bits = |m: &DenseComplex| {
            m.data()
                .iter()
                .map(|v| [v.re.to_bits(), v.im.to_bits()])
                .collect::<Vec<_>>()
        };
        let mut out = DenseComplex::zeros(3).unwrap();
        for diagonal in [
            [
                Complex::new(-0., -0.),
                Complex::real(1.),
                Complex::new(-2., 3.),
            ],
            [
                Complex::real(f64::from_bits(1)),
                Complex::new(1e150, -1e150),
                Complex::real(-0.5),
            ],
            [
                Complex::real(1e-150),
                Complex::real(1e-150),
                Complex::real(1e-150),
            ],
        ] {
            let d = DenseComplex::diagonal(&diagonal).unwrap();
            a.right_diagonal_into(&diagonal, &mut out).unwrap();
            assert_eq!(bits(&out), bits(&a.mul(&d).unwrap()));
            a.left_diagonal_into(&diagonal, &mut out).unwrap();
            assert_eq!(bits(&out), bits(&d.mul(&a).unwrap()));
        }
        for invalid in [f64::INFINITY, f64::NAN] {
            let diag = [Complex::real(invalid); 3];
            assert!(a.right_diagonal_into(&diag, &mut out).is_err());
            assert!(a.left_diagonal_into(&diag, &mut out).is_err());
        }
        assert!(
            a.right_diagonal_into(&[Complex::ZERO; 2], &mut out)
                .is_err()
        );
        let huge = DenseComplex::diagonal(&[Complex::real(f64::MAX); 3]).unwrap();
        assert!(
            huge.right_diagonal_into(&[Complex::real(2.); 3], &mut out)
                .is_err()
        );
        assert!(
            huge.left_diagonal_into(&[Complex::real(2.); 3], &mut out)
                .is_err()
        );
        let mut empty = DenseComplex::zeros(0).unwrap();
        DenseComplex::zeros(0)
            .unwrap()
            .left_diagonal_into(&[], &mut empty)
            .unwrap();
        DenseComplex::zeros(0)
            .unwrap()
            .right_diagonal_into(&[], &mut empty)
            .unwrap();
    }

    #[test]
    fn reusable_inverse_and_product_overwrite_stale_scratch() {
        let a = matrix(2, &[(2., 1.), (3., 4.), (5., -1.), (1., 2.)]);
        let mut scratch = DenseComplex::diagonal(&[Complex::real(f64::NAN); 2]).unwrap();
        let mut out = scratch.clone();
        let expected = a.inverse().unwrap().unwrap();
        assert!(a.inverse_into(&mut out, &mut scratch).unwrap());
        assert_eq!(out, expected);
        assert!(
            !DenseComplex::zeros(2)
                .unwrap()
                .inverse_into(&mut out, &mut scratch)
                .unwrap()
        );
        assert!(a.inverse_into(&mut out, &mut scratch).unwrap());
        assert_eq!(out, expected);
        a.mul_into(&expected, &mut scratch).unwrap();
        assert_eq!(scratch, a.mul(&expected).unwrap());
        assert!(
            a.inverse_into(&mut DenseComplex::zeros(1).unwrap(), &mut scratch)
                .is_err()
        );
    }
}
