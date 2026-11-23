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
use std::simd::{
    Mask,
    cmp::{SimdPartialEq, SimdPartialOrd},
    f64x4,
    num::SimdFloat,
};

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
                    accumulate_row::<false>(out_row, a, b_row);
                }
            }
        }
        out.check_finite()
    }

    /// Accumulate an outer product using the same strict complex product and
    /// addition order as the independent row-major scalar covariance loop.
    /// The caller supplies already-conjugated right entries when needed.
    pub(crate) fn add_outer(&mut self, left: &[Complex], right: &[Complex]) -> SpiceResult<()> {
        let n = self.dim();
        if left.len() != n || right.len() != n {
            return Err(numerical("outer product dimension mismatch"));
        }
        for (&factor, out) in left.iter().zip(self.data.chunks_exact_mut(n.max(1))) {
            accumulate_row::<false>(out, factor, right);
        }
        Ok(())
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
            multiply_row(a_pivot, inverse_pivot);
            multiply_row(b_pivot, inverse_pivot);
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
                accumulate_row::<true>(a_row, factor, a_pivot);
                accumulate_row::<true>(b_row, factor, b_pivot);
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

/// Compute four independent complex products in parallel without changing
/// the C-compatible scalar product's exceptional/cancellation recovery.
/// A lane takes the SIMD result only if *both* product components would take
/// the scalar fast branch; otherwise the original Complex multiplication is
/// evaluated for that lane. No FMA or reassociation is introduced.
#[inline]
fn row_products(factor: Complex, source: &[Complex]) -> ([f64; 4], [f64; 4], Mask<i64, 4>) {
    let a_re = f64x4::splat(factor.re);
    let a_im = f64x4::splat(factor.im);
    let b_re = f64x4::from_array(std::array::from_fn(|i| source[i].re));
    let b_im = f64x4::from_array(std::array::from_fn(|i| source[i].im));
    let ac = a_re * b_re;
    let bd = f64x4::splat(-factor.im) * b_im;
    let ad = a_re * b_im;
    let bc = a_im * b_re;
    let real = ac + bd;
    let imag = ad + bc;
    let axial = Mask::splat(factor.re == 0. || factor.im == 0.)
        | b_re.simd_eq(f64x4::splat(0.))
        | b_im.simd_eq(f64x4::splat(0.));
    let normal = real.is_normal() & imag.is_normal();
    let well_separated = real
        .abs()
        .simd_gt(ac.abs().simd_max(bd.abs()) * f64x4::splat(0.5))
        & imag
            .abs()
            .simd_gt(ad.abs().simd_max(bc.abs()) * f64x4::splat(0.5));
    (
        real.to_array(),
        imag.to_array(),
        axial | (normal & well_separated),
    )
}

/// Add or subtract a product of one complex scalar and a dense row.
/// Row order and the individual complex accumulation order remain unchanged.
#[inline]
fn accumulate_row<const SUBTRACT: bool>(
    destination: &mut [Complex],
    factor: Complex,
    source: &[Complex],
) {
    debug_assert_eq!(destination.len(), source.len());
    // Short port matrices are measurably faster with scalar code: packing
    // and four-lane classification cost more than the few saved products.
    if destination.len() < 32 {
        for (out, &input) in destination.iter_mut().zip(source) {
            *out = if SUBTRACT {
                *out - factor * input
            } else {
                *out + factor * input
            };
        }
        return;
    }
    let mut out_chunks = destination.chunks_exact_mut(4);
    let mut in_chunks = source.chunks_exact(4);
    for (out, input) in out_chunks.by_ref().zip(in_chunks.by_ref()) {
        let (re, im, fast) = row_products(factor, input);
        if fast.all() {
            let old_re = f64x4::from_array(std::array::from_fn(|i| out[i].re));
            let old_im = f64x4::from_array(std::array::from_fn(|i| out[i].im));
            let new_re = if SUBTRACT {
                old_re - f64x4::from_array(re)
            } else {
                old_re + f64x4::from_array(re)
            };
            let new_im = if SUBTRACT {
                old_im - f64x4::from_array(im)
            } else {
                old_im + f64x4::from_array(im)
            };
            for lane in 0..4 {
                out[lane] = Complex::new(new_re[lane], new_im[lane]);
            }
        } else {
            for lane in 0..4 {
                let product = if fast.test(lane) {
                    Complex::new(re[lane], im[lane])
                } else {
                    factor * input[lane]
                };
                out[lane] = if SUBTRACT {
                    out[lane] - product
                } else {
                    out[lane] + product
                };
            }
        }
    }
    for (out, &input) in out_chunks
        .into_remainder()
        .iter_mut()
        .zip(in_chunks.remainder())
    {
        *out = if SUBTRACT {
            *out - factor * input
        } else {
            *out + factor * input
        };
    }
}

/// Multiply a pivot row in place, preserving the scalar product for every
/// lane requiring exponent-range or cancellation recovery.
#[inline]
fn multiply_row(row: &mut [Complex], factor: Complex) {
    if row.len() < 32 {
        for value in row {
            *value = *value * factor;
        }
        return;
    }
    let mut chunks = row.chunks_exact_mut(4);
    for chunk in chunks.by_ref() {
        let (re, im, fast) = row_products(factor, chunk);
        for lane in 0..4 {
            chunk[lane] = if fast.test(lane) {
                Complex::new(re[lane], im[lane])
            } else {
                chunk[lane] * factor
            };
        }
    }
    for value in chunks.into_remainder() {
        *value = *value * factor;
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

    fn bits(values: &[Complex]) -> Vec<[u64; 2]> {
        values
            .iter()
            .map(|value| [value.re.to_bits(), value.im.to_bits()])
            .collect()
    }

    /// Independent scalar reference to the *same* complex product and
    /// accumulation operations, including IEEE zero signs.
    fn scalar_mul_into(
        left: &DenseComplex,
        right: &DenseComplex,
        out: &mut DenseComplex,
    ) -> Result<(), String> {
        let n = left.dim();
        if left
            .data()
            .iter()
            .chain(right.data())
            .any(|value| !value.is_finite())
        {
            return Err("non-finite entry".to_owned());
        }
        out.clear();
        for (a_row, out_row) in left
            .data
            .chunks_exact(n.max(1))
            .zip(out.data.chunks_exact_mut(n.max(1)))
        {
            for (&a, row) in a_row.iter().zip(right.data.chunks_exact(n.max(1))) {
                if a == Complex::ZERO {
                    continue;
                }
                for (out, &b) in out_row.iter_mut().zip(row) {
                    *out = *out + a * b;
                }
            }
        }
        if out.data.iter().any(|v| !v.is_finite()) {
            Err("non-finite result".to_owned())
        } else {
            Ok(())
        }
    }

    fn scalar_mul(left: &DenseComplex, right: &DenseComplex) -> Result<DenseComplex, String> {
        let mut out = DenseComplex::zeros(left.dim()).unwrap();
        scalar_mul_into(left, right, &mut out)?;
        Ok(out)
    }

    #[test]
    fn vectorized_complex_rows_preserve_every_scalar_product_bit() {
        let smallest = f64::from_bits(1);
        let corpus = [
            Complex::new(0., -0.),
            Complex::new(-0., 0.),
            Complex::new(-0., -0.),
            Complex::new(1., -1.),
            Complex::new(1e16, 1.),
            Complex::new(-1e16, 1e-16),
            Complex::new(smallest, -smallest),
            Complex::new(1e-300, -1e300),
            Complex::new(1e300, 1e-300),
            Complex::new(f64::MAX, f64::MAX),
            Complex::new(0.125, 0.25),
            Complex::new(-0.125, -0.25),
            Complex::new(3., -7.),
            Complex::new(-2., 5.),
        ];
        for n in 0..=35 {
            let source: Vec<_> = (0..n).map(|i| corpus[(7 * i + 3) % corpus.len()]).collect();
            let initial: Vec<_> = (0..n).map(|i| corpus[(3 * i + 5) % corpus.len()]).collect();
            for &factor in &corpus {
                let mut optimized = initial.clone();
                let mut reference = initial.clone();
                super::accumulate_row::<false>(&mut optimized, factor, &source);
                for (out, &value) in reference.iter_mut().zip(&source) {
                    *out = *out + factor * value;
                }
                assert_eq!(
                    bits(&optimized),
                    bits(&reference),
                    "add len={n} factor={factor:?}"
                );

                let mut optimized = initial.clone();
                let mut reference = initial.clone();
                super::accumulate_row::<true>(&mut optimized, factor, &source);
                for (out, &value) in reference.iter_mut().zip(&source) {
                    *out = *out - factor * value;
                }
                assert_eq!(
                    bits(&optimized),
                    bits(&reference),
                    "subtract len={n} factor={factor:?}"
                );

                let mut optimized = source.clone();
                let mut reference = source.clone();
                super::multiply_row(&mut optimized, factor);
                for value in &mut reference {
                    *value = *value * factor;
                }
                assert_eq!(
                    bits(&optimized),
                    bits(&reference),
                    "scale len={n} factor={factor:?}"
                );
            }
        }
    }

    #[test]
    fn dense_product_and_outer_covariance_match_scalar_order() {
        let corpus = [
            Complex::new(0., -0.),
            Complex::new(-0., 0.),
            Complex::new(1e16, -1.),
            Complex::new(-1e16, 1.),
            Complex::new(1., 1e-300),
            Complex::new(1e-300, -1.),
            Complex::new(-0.75, 0.5),
            Complex::new(2., -1.),
            Complex::new(1e150, 1e-150),
        ];
        for n in [0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 32] {
            let mut a = DenseComplex::zeros(n).unwrap();
            let mut b = DenseComplex::zeros(n).unwrap();
            for i in 0..n * n {
                a.data[i] = corpus[(i * 5 + 1) % corpus.len()];
                b.data[i] = corpus[(i * 7 + 3) % corpus.len()];
            }
            let scalar = scalar_mul(&a, &b);
            let vectorized = a.mul(&b);
            match (scalar, vectorized) {
                (Ok(reference), Ok(actual)) => {
                    assert_eq!(bits(reference.data()), bits(actual.data()), "product n={n}")
                }
                (Err(expected), Err(actual)) => assert!(
                    actual.to_string().contains(&expected),
                    "product n={n}: {actual}"
                ),
                (reference, actual) => {
                    panic!("product n={n}: scalar {reference:?}, SIMD {actual:?}")
                }
            }
            let lhs: Vec<_> = (0..n).map(|i| corpus[(i * 3 + 2) % corpus.len()]).collect();
            let rhs: Vec<_> = (0..n).map(|i| corpus[(i * 7 + 1) % corpus.len()]).collect();
            let mut optimized = DenseComplex::zeros(n).unwrap();
            let mut reference = optimized.clone();
            for _ in 0..3 {
                optimized.add_outer(&lhs, &rhs).unwrap();
                for (i, &factor) in lhs.iter().enumerate() {
                    for (j, &value) in rhs.iter().enumerate() {
                        let dest = &mut reference.data[i * n + j];
                        *dest = *dest + factor * value;
                    }
                }
                assert_eq!(
                    bits(optimized.data()),
                    bits(reference.data()),
                    "outer n={n}"
                );
            }
            if n > 0 {
                assert!(optimized.add_outer(&lhs[..n - 1], &rhs).is_err());
                assert!(optimized.add_outer(&lhs, &rhs[..n - 1]).is_err());
            }
        }
    }

    #[test]
    #[ignore = "opt-in dense complex matrix microbenchmark"]
    fn dense_complex_row_kernel_benchmark() {
        use std::hint::black_box;
        use std::time::Instant;
        for n in [2, 8, 16, 32, 64] {
            let mut a = DenseComplex::zeros(n).unwrap();
            let mut b = DenseComplex::zeros(n).unwrap();
            for (i, value) in a.data.iter_mut().enumerate() {
                *value = Complex::new(
                    f64::from(u8::try_from(i % 31).unwrap()) / 31. + 0.1,
                    f64::from(u8::try_from(i % 19).unwrap()) / 19. + 0.1,
                );
            }
            for (i, value) in b.data.iter_mut().enumerate() {
                *value = Complex::new(
                    f64::from(u8::try_from(i % 23).unwrap()) / 23. - 0.5,
                    f64::from(u8::try_from(i % 29).unwrap()) / 29. - 0.5,
                );
            }
            let mut out = DenseComplex::zeros(n).unwrap();
            let mut reference = out.clone();
            let repeats = (2048 / n).max(2);
            let mut baseline = Vec::new();
            let mut vectorized = Vec::new();
            for _ in 0..7 {
                let start = Instant::now();
                for _ in 0..repeats {
                    scalar_mul_into(black_box(&a), black_box(&b), black_box(&mut reference))
                        .unwrap();
                }
                black_box(&reference);
                baseline.push(start.elapsed());
                let start = Instant::now();
                for _ in 0..repeats {
                    a.mul_into(black_box(&b), black_box(&mut out)).unwrap();
                }
                black_box(&out);
                vectorized.push(start.elapsed());
            }
            baseline.sort();
            vectorized.sort();
            assert_eq!(bits(reference.data()), bits(out.data()));
            eprintln!(
                "dense-complex n={n} reps={repeats} scalar_ns={} simd_ns={}",
                baseline[3].as_nanos(),
                vectorized[3].as_nanos()
            );
            let currents = &a.data[..n];
            let conjugates = currents
                .iter()
                .map(|value| value.conj())
                .collect::<Vec<_>>();
            let mut original_cov = DenseComplex::zeros(n).unwrap();
            let mut simd_cov = original_cov.clone();
            baseline.clear();
            vectorized.clear();
            for _ in 0..7 {
                let start = Instant::now();
                for _ in 0..repeats {
                    original_cov.clear();
                    for (row, &factor) in currents.iter().enumerate() {
                        for (col, &value) in conjugates.iter().enumerate() {
                            let cell = &mut original_cov.data[row * n + col];
                            *cell = *cell + factor * value;
                        }
                    }
                    black_box(&original_cov);
                }
                baseline.push(start.elapsed());
                let start = Instant::now();
                for _ in 0..repeats {
                    simd_cov.clear();
                    simd_cov
                        .add_outer(black_box(currents), black_box(&conjugates))
                        .unwrap();
                    black_box(&simd_cov);
                }
                vectorized.push(start.elapsed());
            }
            baseline.sort();
            vectorized.sort();
            assert_eq!(bits(original_cov.data()), bits(simd_cov.data()));
            eprintln!(
                "complex-outer n={n} reps={repeats} scalar_ns={} simd_ns={}",
                baseline[3].as_nanos(),
                vectorized[3].as_nanos()
            );
        }
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
