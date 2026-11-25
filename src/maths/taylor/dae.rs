//! Linear derivative-array construction and rank projection.
//!
//! For E x' + A x = b(t), with x = sum(x_k u^k), u=t/scale,
//! coefficient k imposes A x_k + E (k+1)/scale x_(k+1) = b_k.
//! The last derivative block generally remains partially free. Project
//! onto the requested lower coefficients rather than silently gauging it.

use super::{MAX_ORDER, index};
use crate::maths::constraint_rank::{self, Projection};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Sparse linear DAE whose forcing has been expanded in normalized time.
pub struct LinearDae<'a> {
    /// Static residual Jacobian.
    pub static_matrix: &'a SparseMatrix,
    /// Storage Jacobian, with the same row and column coordinates.
    pub storage_matrix: &'a SparseMatrix,
    /// RHS coefficients in degree order; one more unknown degree is added.
    pub forcing: &'a [Vector],
    /// Physical time represented by one unit of the normalized coordinate.
    pub time_scale: Real,
}

/// Physical initial constraints on the degree-zero solution.
pub struct InitialRows<'a> {
    /// Any linear relation among original state coordinates.
    pub matrix: &'a SparseMatrix,
    /// Prescribed values in constraint-row order.
    pub rhs: &'a Vector,
    /// Optional constraints retained only when independently needed.
    pub optional: &'a [usize],
}

/// Explicit bounds on derivative-array storage and sparse elimination.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Maximum combined row and column dimension.
    pub dimension: usize,
    /// Maximum number of input stamps (before duplicate folding).
    pub stamps: usize,
    /// Maximum sparse coefficient eliminations, including projection.
    pub rank_work: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            dimension: 262_144,
            stamps: 2_000_000,
            rank_work: 8_000_000,
        }
    }
}

/// Residual verification in the original equation units.
pub struct Tolerances<'a> {
    /// Relative backward-error tolerance.
    pub relative: Real,
    /// Absolute tolerance for each DAE equation, repeated across degrees.
    pub dae_absolute: &'a [Real],
    /// Absolute tolerance for each initial constraint.
    pub initial_absolute: &'a [Real],
}

/// Determination of the requested leading coefficients, independent of any
/// arbitrary choice of the free higher coefficients.
pub struct Solution {
    /// Degree-major coordinates, with None for genuinely unresolved values.
    pub coefficients: Vec<Vec<Option<Real>>>,
    /// Initial rows that were selected as independent constraints.
    pub retained_initial: Vec<bool>,
}

fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "initial derivative array".to_owned(),
        message: message.into(),
    }
}

struct Array {
    matrix: SparseMatrix,
    rhs: Vec<Real>,
    optional: Vec<usize>,
    dae_rows: usize,
    width: usize,
    requested: usize,
}

impl LinearDae<'_> {
    fn assemble(
        &self,
        initial: &InitialRows<'_>,
        requested_degrees: usize,
        limits: Limits,
    ) -> SpiceResult<Array> {
        let width = self.static_matrix.cols();
        let equations = self.static_matrix.rows();
        let degree_count = self.forcing.len();
        if width == 0
            || equations == 0
            || degree_count == 0
            || degree_count > MAX_ORDER + 1
            || !(1..=degree_count + 1).contains(&requested_degrees)
            || !self.time_scale.is_finite()
            || self.time_scale <= 0.
            || self.storage_matrix.cols() != width
            || self.storage_matrix.rows() != equations
            || initial.matrix.cols() != width
            || initial.matrix.rows() != initial.rhs.len()
            || initial.optional.iter().any(|&row| row >= initial.rhs.len())
            || self.forcing.iter().any(|rhs| rhs.len() != equations)
        {
            return Err(failure(
                "invalid derivative-array input shape or time scale",
            ));
        }
        let columns = width
            .checked_mul(degree_count + 1)
            .ok_or_else(|| failure("derivative-array dimension overflow"))?;
        let dae_rows = equations
            .checked_mul(degree_count)
            .ok_or_else(|| failure("derivative-array dimension overflow"))?;
        let rows = dae_rows
            .checked_add(initial.rhs.len())
            .ok_or_else(|| failure("derivative-array dimension overflow"))?;
        let stamps = self
            .static_matrix
            .triplets()
            .len()
            .checked_add(self.storage_matrix.triplets().len())
            .and_then(|count| count.checked_mul(degree_count))
            .and_then(|count| count.checked_add(initial.matrix.triplets().len()))
            .ok_or_else(|| failure("derivative-array stamp count overflow"))?;
        if rows.saturating_add(columns) > limits.dimension || stamps > limits.stamps {
            return Err(failure("derivative-array storage limit exceeded"));
        }
        let mut matrix = SparseMatrix::new(rows, columns);
        let mut rhs = Vec::with_capacity(rows);
        for (degree, forcing) in self.forcing.iter().enumerate() {
            let row_offset = degree * equations;
            let column_offset = degree * width;
            rhs.extend_from_slice(forcing.as_slice());
            for term in self.static_matrix.triplets() {
                matrix.add(row_offset + term.row, column_offset + term.col, term.value)?;
            }
            for term in self.storage_matrix.triplets() {
                if !term.value.is_finite() {
                    return Err(failure("nonfinite storage derivative coefficient"));
                }
                if term.value == 0. {
                    continue;
                }
                let multiplier = index(degree + 1);
                let mut value = term.value * (multiplier / self.time_scale);
                if !value.is_finite() || value == 0. {
                    value = crate::primitives::rounded_product_ratio(
                        [(term.value, multiplier, false)],
                        [(self.time_scale, false)],
                    );
                }
                if !value.is_finite() || value == 0. {
                    return Err(failure("nonrepresentable storage derivative coefficient"));
                }
                matrix.add(
                    row_offset + term.row,
                    column_offset + width + term.col,
                    value,
                )?;
            }
        }
        rhs.extend_from_slice(initial.rhs.as_slice());
        for term in initial.matrix.triplets() {
            matrix.add(dae_rows + term.row, term.col, term.value)?;
        }
        matrix.fold_duplicates();
        Ok(Array {
            matrix,
            rhs,
            optional: initial.optional.iter().map(|&row| dae_rows + row).collect(),
            dae_rows,
            width,
            requested: requested_degrees * width,
        })
    }

    /// Solve a truncated linear derivative array and identify its unique
    /// leading coordinates. Mandatory inconsistent rows fail. Optional
    /// initial values can only choose real remaining degrees of freedom.
    ///
    /// # Errors
    /// Shape, domain, budget or consistency failure.
    pub fn solve(
        &self,
        initial: &InitialRows<'_>,
        requested_degrees: usize,
        tolerance: &Tolerances<'_>,
        limits: Limits,
    ) -> SpiceResult<Solution> {
        let array = self.assemble(initial, requested_degrees, limits)?;
        let projection = constraint_rank::project(
            &array.matrix,
            &array.rhs,
            &array.optional,
            array.requested,
            limits.rank_work,
        )?;
        array.verify(&projection, tolerance, self.static_matrix.rows())?;
        Ok(Solution {
            coefficients: projection.representative.as_slice()[..array.requested]
                .chunks_exact(array.width)
                .zip(projection.determined.chunks_exact(array.width))
                .map(|(values, determined)| {
                    values
                        .iter()
                        .zip(determined)
                        .map(|(&value, &known)| known.then_some(value))
                        .collect()
                })
                .collect(),
            retained_initial: projection.independent[array.dae_rows..].to_vec(),
        })
    }
}

impl Array {
    fn verify(
        &self,
        projection: &Projection,
        tolerance: &Tolerances<'_>,
        equations: usize,
    ) -> SpiceResult<()> {
        if !tolerance.relative.is_finite()
            || tolerance.relative < 0.
            || tolerance.dae_absolute.len() != equations
            || tolerance.initial_absolute.len() != self.rhs.len() - self.dae_rows
            || tolerance
                .dae_absolute
                .iter()
                .chain(tolerance.initial_absolute)
                .any(|&value| !value.is_finite() || value < 0.)
        {
            return Err(failure("invalid derivative-array residual tolerance"));
        }
        let mut residuals: Vec<_> = self
            .rhs
            .iter()
            .map(|&value| (-value, value.abs()))
            .collect();
        for term in self.matrix.triplets() {
            let x = projection.representative.as_slice()[term.col];
            let (residual, scale) = &mut residuals[term.row];
            *residual = term.value.mul_add(x, *residual);
            *scale += (term.value * x).abs();
        }
        let mut optional = vec![false; self.rhs.len()];
        for &row in &self.optional {
            optional[row] = true;
        }
        for (row, (residual, scale)) in residuals.into_iter().enumerate() {
            if optional[row] && !projection.independent[row] {
                continue;
            }
            let absolute = if row < self.dae_rows {
                tolerance.dae_absolute[row % equations]
            } else {
                tolerance.initial_absolute[row - self.dae_rows]
            };
            if !residual.is_finite()
                || !scale.is_finite()
                || residual.abs() > tolerance.relative.mul_add(scale, absolute)
            {
                return Err(failure(format!(
                    "inconsistent derivative-array equation {row}: residual {residual}"
                )));
            }
        }
        Ok(())
    }
}
