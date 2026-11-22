//! Dense row-major real matrix and vector storage.
//!
//! Mirrors `src/maths/dense/` (`smat.c`), which ngspice uses for the small
//! systems in `.sens`, `.tf` and `.pz`. The MNA matrix itself is sparse — see
//! [`crate::maths::sparse`]. This is a general storage/LU API, not an implementation
//! of `.sens` or `.tf`; `.pz` uses [`crate::maths::pencil`] on dense copies.
//!
//! Factorization returns an owned pivoted LU; public storage stays row-major.

use std::{alloc::Layout, ops::Range};

use crate::primitives::{Real, SpiceError, SpiceResult, filled_vec};

use crate::maths::linear::DenseLu;

fn index_error(rows: usize, cols: usize, row: usize, col: usize) -> SpiceError {
    SpiceError::Numerical {
        context: "dense matrix".to_owned(),
        message: format!("index ({row}, {col}) is out of range for a {rows}x{cols} matrix"),
    }
}

fn length_error(expected: usize, actual: usize) -> SpiceError {
    SpiceError::Numerical {
        context: "dense vector".to_owned(),
        message: format!("expected length {expected}, got {actual}"),
    }
}

/// Validated row-major dimensions for dense storage. Every in-range row
/// offset fits because the entire element count and allocation layout fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DenseShape {
    rows: usize,
    cols: usize,
}

impl DenseShape {
    pub(super) fn checked<T>(rows: usize, cols: usize, context: &str) -> SpiceResult<Self> {
        let count = rows
            .checked_mul(cols)
            .ok_or_else(|| SpiceError::Numerical {
                context: context.to_owned(),
                message: format!("matrix dimensions {rows}x{cols} overflow"),
            })?;
        Layout::array::<T>(count).map_err(|_| SpiceError::Numerical {
            context: context.to_owned(),
            message: format!("matrix allocation layout {rows}x{cols} overflows"),
        })?;
        Ok(Self { rows, cols })
    }

    pub(super) const fn rows(self) -> usize {
        self.rows
    }

    pub(super) const fn len(self) -> usize {
        self.rows * self.cols
    }

    pub(super) const fn offset(self, row: usize, col: usize) -> Option<usize> {
        if row < self.rows && col < self.cols {
            Some(row * self.cols + col)
        } else {
            None
        }
    }

    const fn row_range(self, row: usize) -> Option<Range<usize>> {
        if row < self.rows {
            Some(row * self.cols..(row + 1) * self.cols)
        } else {
            None
        }
    }

    const fn transpose(self) -> Self {
        Self {
            rows: self.cols,
            cols: self.rows,
        }
    }
}

/// Allocate once from an exact-size dense iterator (including snapshots).
/// The iterator is consumed in its existing order.
pub(super) fn collect_dense<T>(
    values: impl ExactSizeIterator<Item = T>,
    context: &str,
) -> SpiceResult<Vec<T>> {
    let count = values.len();
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|error| SpiceError::Numerical {
            context: context.to_owned(),
            message: format!("cannot allocate {count} entries: {error}"),
        })?;
    data.extend(values);
    Ok(data)
}

/// A dense, row-major real matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    shape: DenseShape,
    data: Vec<Real>,
}

impl Matrix {
    /// A zeroed matrix.
    /// # Errors
    /// Dimensions overflow or storage cannot be allocated.
    pub fn zeros(rows: usize, cols: usize) -> SpiceResult<Self> {
        let shape = DenseShape::checked::<Real>(rows, cols, "dense matrix")?;
        Ok(Self {
            shape,
            data: filled_vec(shape.len(), 0.0, "dense matrix")?,
        })
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.shape.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.shape.cols
    }

    /// Total number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the matrix has no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// True when the matrix is square.
    #[must_use]
    pub const fn is_square(&self) -> bool {
        self.shape.rows == self.shape.cols
    }

    /// Reads an element, or `None` when out of range.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<Real> {
        self.shape.offset(row, col).map(|offset| self.data[offset])
    }

    /// Writes an element.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when the index is out of range.
    pub fn set(&mut self, row: usize, col: usize, value: Real) -> SpiceResult<()> {
        let offset = self
            .shape
            .offset(row, col)
            .ok_or_else(|| index_error(self.shape.rows, self.shape.cols, row, col))?;
        self.data[offset] = value;
        Ok(())
    }

    /// Adds to an element, which is what MNA stamping does.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when the index is out of range.
    pub fn add_to(&mut self, row: usize, col: usize, value: Real) -> SpiceResult<()> {
        let offset = self
            .shape
            .offset(row, col)
            .ok_or_else(|| index_error(self.shape.rows, self.shape.cols, row, col))?;
        self.data[offset] += value;
        Ok(())
    }

    /// One row, as a slice.
    #[must_use]
    pub fn row(&self, row: usize) -> Option<&[Real]> {
        self.shape.row_range(row).map(|range| &self.data[range])
    }

    /// The elements, row-major.
    #[must_use]
    pub fn data(&self) -> &[Real] {
        &self.data
    }

    /// The elements, row-major, mutable.
    pub fn data_mut(&mut self) -> &mut [Real] {
        &mut self.data
    }

    /// Sets every element to `value`.
    pub fn fill(&mut self, value: Real) {
        self.data.fill(value);
    }

    /// Sets every element to zero.
    pub fn clear(&mut self) {
        self.fill(0.0);
    }

    /// The transpose.
    /// # Errors
    /// The transposed storage cannot be allocated.
    pub fn transpose(&self) -> SpiceResult<Self> {
        let mut result = Self {
            shape: self.shape.transpose(),
            data: filled_vec(self.len(), 0.0, "dense matrix transpose")?,
        };
        if self.is_empty() {
            return Ok(result);
        }
        for row in 0..self.shape.rows {
            for col in 0..self.shape.cols {
                result.data[col * self.shape.rows + row] = self.data[row * self.shape.cols + col];
            }
        }
        Ok(result)
    }

    /// Matrix–vector product.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when the dimensions do not agree or output storage cannot be allocated.
    pub fn mul_vector(&self, vector: &Vector) -> SpiceResult<Vector> {
        if vector.len() != self.shape.cols {
            return Err(length_error(self.shape.cols, vector.len()));
        }
        let mut result = Vector::from(filled_vec(self.shape.rows, 0.0, "dense matrix product")?);
        if self.shape.cols != 0 {
            for (row, output) in self
                .data
                .chunks_exact(self.shape.cols)
                .zip(result.as_mut_slice())
            {
                let mut sum = 0.0;
                for (entry, value) in row.iter().zip(vector.as_slice()) {
                    sum += entry * value;
                }
                *output = sum;
            }
        }
        Ok(result)
    }

    /// Factors a snapshot without changing row-major storage.
    ///
    /// # Errors
    ///
    /// Invalid dimensions/coefficients or singular pivots. Empty systems are rejected.
    pub fn lu_decompose(&self) -> SpiceResult<DenseLu> {
        DenseLu::new(self)
    }

    /// Convenience solve, factoring the current storage. No prior LU is required.
    ///
    /// # Errors
    ///
    /// Invalid matrix/RHS or failed backward residual.
    pub fn solve(&self, rhs: &Vector) -> SpiceResult<Vector> {
        self.lu_decompose()?.solve(rhs)
    }

    /// Eigenvalues of a real symmetric matrix, in nondecreasing order, by
    /// cyclic Jacobi rotations (accurate to a few ulps of the largest
    /// eigenvalue). Used for definiteness checks of small coupled-inductance
    /// matrices; it is not a general eigensolver API.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] for an empty or non-square matrix, nonfinite
    /// entries, an asymmetric matrix (entries must agree exactly), or a
    /// failed decomposition, or unavailable workspace storage.
    pub fn symmetric_eigenvalues(&self) -> SpiceResult<Vec<Real>> {
        let error = |message: String| SpiceError::Numerical {
            context: "dense symmetric eigenvalues".to_owned(),
            message,
        };
        if self.shape.rows != self.shape.cols || self.shape.rows == 0 {
            return Err(error(format!(
                "needs a nonempty square matrix, got {}x{}",
                self.shape.rows, self.shape.cols
            )));
        }
        if self.data.iter().any(|value| !value.is_finite()) {
            return Err(error("nonfinite entry".to_owned()));
        }
        let n = self.shape.rows;
        for row in 0..n {
            for col in 0..row {
                if self.data[row * n + col].partial_cmp(&self.data[col * n + row])
                    != Some(std::cmp::Ordering::Equal)
                {
                    return Err(error(format!(
                        "entries ({row}, {col}) and ({col}, {row}) differ"
                    )));
                }
            }
        }
        let mut data = collect_dense(self.data.iter().copied(), "dense symmetric eigenvalues")?;
        // Ordinary physical matrices keep their original arithmetic. At
        // extreme scales the sum of squared entries (or its epsilon-squared
        // threshold) can overflow or underflow, falsely reporting convergence.
        // Normalize by a power of two only outside the safe norm range.
        let max = data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let scale = extreme_norm_scale(max);
        if let Some(scale) = scale {
            for value in &mut data {
                *value /= scale;
            }
        }
        let mut values = jacobi_eigenvalues(data, n)?
            .ok_or_else(|| error("Jacobi rotations did not converge".to_owned()))?;
        if let Some(scale) = scale {
            for value in &mut values {
                *value *= scale;
            }
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(error("nonfinite eigenvalue".to_owned()));
        }
        Ok(values)
    }
}

/// A power-of-two normalization for finite nonnegative maximum components,
/// only when squaring unscaled values risks overflow or underflow. Keep
/// ordinary matrices on their reference-compatible arithmetic path.
pub(super) fn extreme_norm_scale(max: Real) -> Option<Real> {
    if max == 0.0 || (2.0_f64.powi(-450)..=2.0_f64.powi(450)).contains(&max) {
        return None;
    }
    // The exponent bits yield the leading power of two. Subnormal inputs
    // use MIN_POSITIVE, so the smallest ratio remains 2^-52.
    let exponent = f64::from_bits(max.to_bits() & 0x7ff0_0000_0000_0000);
    Some(exponent.max(f64::MIN_POSITIVE))
}

/// Cyclic Jacobi eigenvalues of the symmetric row-major `a` (n x n), sorted.
/// `None` if the off-diagonal mass fails to vanish within the sweep budget.
fn jacobi_eigenvalues(mut data: Vec<Real>, order: usize) -> SpiceResult<Option<Vec<Real>>> {
    let off_diagonal_norm = |data: &[Real]| -> Real {
        let mut sum = 0.0;
        for row in 0..order {
            for col in 0..order {
                if row != col {
                    let entry = data[row * order + col];
                    let squared = entry * entry;
                    sum += squared;
                }
            }
        }
        sum
    };
    let total: Real = data.iter().map(|value| value * value).sum();
    for _sweep in 0..100 {
        if off_diagonal_norm(&data) <= Real::EPSILON * Real::EPSILON * total {
            let mut values = collect_dense(
                (0..order).map(|index| data[index * order + index]),
                "dense symmetric eigenvalues",
            )?;
            values.sort_unstable_by(Real::total_cmp);
            return Ok(Some(values));
        }
        for first in 0..order {
            for second in first + 1..order {
                let coupling = data[first * order + second];
                if coupling == 0.0 {
                    continue;
                }
                let theta = (data[second * order + second] - data[first * order + first])
                    / (2.0 * coupling);
                let theta_squared = theta * theta;
                let tangent = theta.signum() / (theta.abs() + (theta_squared + 1.0).sqrt());
                let tangent = if theta == 0.0 { 1.0 } else { tangent };
                let tangent_squared = tangent * tangent;
                let cosine = 1.0 / (tangent_squared + 1.0).sqrt();
                let sine = tangent * cosine;
                for index in 0..order {
                    let first_slot = index * order + first;
                    let second_slot = index * order + second;
                    let (left, right) =
                        rotate_pair(data[first_slot], data[second_slot], cosine, sine);
                    data[first_slot] = left;
                    data[second_slot] = right;
                }
                for index in 0..order {
                    let first_slot = first * order + index;
                    let second_slot = second * order + index;
                    let (left, right) =
                        rotate_pair(data[first_slot], data[second_slot], cosine, sine);
                    data[first_slot] = left;
                    data[second_slot] = right;
                }
            }
        }
    }
    Ok(None)
}

// Round the products separately, preserving the original rotation arithmetic.
const fn rotate_pair(first: Real, second: Real, cosine: Real, sine: Real) -> (Real, Real) {
    let direct = (cosine * first, cosine * second);
    let crossed = (sine * second, sine * first);
    (direct.0 - crossed.0, crossed.1 + direct.1)
}

/// A dense real vector, used for the MNA right-hand side and the solution.
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    data: Vec<Real>,
}

impl From<Vec<Real>> for Vector {
    fn from(data: Vec<Real>) -> Self {
        Self { data }
    }
}

impl Vector {
    /// A zeroed vector of `len` elements.
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    /// A vector with the given elements.
    #[must_use]
    pub fn from_slice(values: &[Real]) -> Self {
        Self {
            data: values.to_vec(),
        }
    }

    /// Number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the vector has no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Reads an element, or `None` when out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<Real> {
        self.data.get(index).copied()
    }

    /// Writes an element.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when the index is out of range.
    pub fn set(&mut self, index: usize, value: Real) -> SpiceResult<()> {
        match self.data.get_mut(index) {
            Some(slot) => {
                *slot = value;
                Ok(())
            }
            None => Err(SpiceError::Numerical {
                context: "dense vector".to_owned(),
                message: format!(
                    "index {index} is out of range for length {}",
                    self.data.len()
                ),
            }),
        }
    }

    /// Adds to an element.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when the index is out of range.
    #[inline]
    pub fn add_to(&mut self, index: usize, value: Real) -> SpiceResult<()> {
        match self.data.get_mut(index) {
            Some(slot) => {
                *slot += value;
                Ok(())
            }
            None => Err(out_of_range(index, self.data.len())),
        }
    }

    /// The elements.
    #[must_use]
    pub fn as_slice(&self) -> &[Real] {
        &self.data
    }

    /// The elements, mutable.
    pub fn as_mut_slice(&mut self) -> &mut [Real] {
        &mut self.data
    }

    /// Sets every element to zero.
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Dot product.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when the lengths differ.
    pub fn dot(&self, other: &Self) -> SpiceResult<Real> {
        if self.len() != other.len() {
            return Err(length_error(self.len(), other.len()));
        }
        Ok(self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum())
    }

    /// Largest absolute element.
    #[must_use]
    pub fn max_abs(&self) -> Real {
        self.data
            .iter()
            .fold(0.0, |acc, value| acc.max(value.abs()))
    }

    /// True when every element is finite.
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|value| value.is_finite())
    }
}

/// [`Vector::add_to`]'s error, out of its hot path.
#[cold]
#[inline(never)]
fn out_of_range(index: usize, len: usize) -> SpiceError {
    SpiceError::Numerical {
        context: "dense vector".to_owned(),
        message: format!("index {index} is out of range for length {len}"),
    }
}

#[cfg(test)]
mod tests {
    use super::{Matrix, Vector};

    #[test]
    fn symmetric_eigenvalues_do_not_converge_on_overflowed_or_underflowed_norms() {
        for scale in [
            2.0_f64.powi(-1000),
            2.0_f64.powi(-600),
            2.0_f64.powi(600),
            2.0_f64.powi(1000),
        ] {
            let mut matrix = Matrix::zeros(2, 2).unwrap();
            matrix
                .data_mut()
                .copy_from_slice(&[2. * scale, scale, scale, 2. * scale]);
            let values = matrix.symmetric_eigenvalues().unwrap();
            assert!((values[0] / scale - 1.).abs() < 1e-14);
            assert!((values[1] / scale - 3.).abs() < 1e-14);
            matrix
                .data_mut()
                .copy_from_slice(&[scale, 2. * scale, 2. * scale, scale]);
            let values = matrix.symmetric_eigenvalues().unwrap();
            assert!((values[0] / scale + 1.).abs() < 1e-14);
            assert!((values[1] / scale - 3.).abs() < 1e-14);
        }
    }

    #[test]
    fn dimensions_cannot_wrap() {
        assert!(Matrix::zeros(usize::MAX, 2).is_err());
        assert!(Matrix::zeros(usize::MAX / 8, 1).is_err());
    }

    #[test]
    fn zero_column_matrix_multiplies_to_zero_vector() {
        let a = Matrix::zeros(3, 0).unwrap();
        assert_eq!(a.mul_vector(&Vector::zeros(0)).unwrap(), Vector::zeros(3));
        assert_eq!(a.transpose().unwrap(), Matrix::zeros(0, 3).unwrap());
    }

    #[test]
    fn empty_extreme_shapes_do_not_allocate_or_iterate_over_rows() {
        let tall = Matrix::zeros(usize::MAX, 0).unwrap();
        assert_eq!(tall.row(usize::MAX - 1), Some(&[][..]));
        assert!(tall.row(usize::MAX).is_none());
        assert!(tall.mul_vector(&Vector::zeros(0)).is_err());
        let wide = tall.transpose().unwrap();
        assert_eq!(wide.rows(), 0);
        assert_eq!(wide.cols(), usize::MAX);
        assert_eq!(wide.transpose().unwrap(), tall);
        assert!(wide.get(0, 0).is_none());
    }

    #[test]
    fn symmetric_eigenvalues_are_sorted_and_validated() {
        let mut matrix = Matrix::zeros(2, 2).unwrap();
        for (row, col, value) in [(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 2.0)] {
            matrix.set(row, col, value).unwrap();
        }
        let values = matrix.symmetric_eigenvalues().unwrap();
        assert!((values[0] - 1.0).abs() < 1e-14 && (values[1] - 3.0).abs() < 1e-14);
        // An indefinite matrix has a negative eigenvalue.
        matrix.set(0, 1, 3.0).unwrap();
        matrix.set(1, 0, 3.0).unwrap();
        assert!(matrix.symmetric_eigenvalues().unwrap()[0] < 0.0);
        matrix.set(1, 0, 2.0).unwrap();
        assert!(matrix.symmetric_eigenvalues().is_err());
        let mut signed_zero = Matrix::zeros(2, 2).unwrap();
        signed_zero.set(0, 1, -0.0).unwrap();
        assert!(signed_zero.symmetric_eigenvalues().is_ok());
        assert!(
            Matrix::zeros(2, 3)
                .unwrap()
                .symmetric_eigenvalues()
                .is_err()
        );
        assert!(
            Matrix::zeros(0, 0)
                .unwrap()
                .symmetric_eigenvalues()
                .is_err()
        );
        let mut nonfinite = Matrix::zeros(1, 1).unwrap();
        nonfinite.data_mut()[0] = f64::NAN;
        assert!(nonfinite.symmetric_eigenvalues().is_err());
    }

    #[test]
    fn matrix_indexing_is_row_major() {
        let mut matrix = Matrix::zeros(2, 3).unwrap();
        matrix.set(1, 2, 7.0).unwrap();
        assert_eq!(matrix.get(1, 2), Some(7.0));
        assert_eq!(matrix.get(2, 2), None);
        assert_eq!(matrix.data(), &[0.0, 0.0, 0.0, 0.0, 0.0, 7.0]);
        assert_eq!(matrix.row(1), Some(&[0.0, 0.0, 7.0][..]));
        assert_eq!(matrix.len(), 6);
        assert!(!matrix.is_square());
    }

    #[test]
    fn stamping_accumulates() {
        let mut matrix = Matrix::zeros(1, 1).unwrap();
        matrix.add_to(0, 0, 0.5).unwrap();
        matrix.add_to(0, 0, 0.25).unwrap();
        assert_eq!(matrix.get(0, 0), Some(0.75));
    }

    #[test]
    fn out_of_range_writes_are_errors() {
        let mut matrix = Matrix::zeros(1, 1).unwrap();
        assert!(matrix.set(0, 1, 1.0).is_err());
        assert!(matrix.add_to(1, 0, 1.0).is_err());
        let mut vector = Vector::zeros(1);
        assert!(vector.set(1, 1.0).is_err());
    }

    #[test]
    fn transpose_moves_elements() {
        let mut matrix = Matrix::zeros(2, 3).unwrap();
        for row in 0..2 {
            for col in 0..3 {
                matrix.set(row, col, (row * 3 + col) as f64).unwrap();
            }
        }
        let transposed = matrix.transpose().unwrap();
        assert_eq!((transposed.rows(), transposed.cols()), (3, 2));
        assert_eq!(transposed.get(2, 1), Some(5.0));
    }

    #[test]
    fn multiply_and_dot_agree_on_an_identity() {
        let mut identity = Matrix::zeros(3, 3).unwrap();
        for index in 0..3 {
            identity.set(index, index, 1.0).unwrap();
        }
        let vector = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let product = identity.mul_vector(&vector).unwrap();
        assert_eq!(product, vector);
        assert_eq!(product.dot(&vector).unwrap(), 14.0);
        assert_eq!(product.max_abs(), 3.0);
        assert!(product.is_finite());
    }

    #[test]
    fn dimension_mismatches_are_errors() {
        let matrix = Matrix::zeros(2, 3).unwrap();
        assert!(matrix.mul_vector(&Vector::zeros(2)).is_err());
        assert!(matrix.solve(&Vector::zeros(3)).is_err());
        assert!(Vector::zeros(2).dot(&Vector::zeros(3)).is_err());
    }

    #[test]
    fn solve_without_prior_factorization() {
        let mut matrix = Matrix::zeros(1, 1).unwrap();
        matrix.set(0, 0, 2.0).unwrap();
        assert_eq!(
            matrix
                .solve(&Vector::from_slice(&[4.0]))
                .unwrap()
                .as_slice(),
            &[2.0]
        );
    }
}
