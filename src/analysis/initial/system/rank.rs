//! Rank-aware completion of the instantaneous constraint system.
//! Only physically free storage currents may be used as gauges. All original
//! equations are retained by the caller for residual verification; the basis
//! is merely a way to obtain a candidate from a redundant linearization.
use std::collections::{BTreeMap, BTreeSet};

use crate::maths::{SparseMatrix, Vector};
use crate::primitives::numeric::{binary_exponent, scale_power};
use crate::primitives::{Real, SpiceResult};

use super::failure;

type Row = BTreeMap<usize, Real>;

struct Pivot {
    column: usize,
    terms: Row,
    rhs: Real,
    uncertainty: Real,
}

struct Basis {
    pivots: Vec<Pivot>,
    by_column: Vec<Option<usize>>,
}

impl Basis {
    fn new(columns: usize) -> Self {
        Self {
            pivots: Vec::new(),
            by_column: vec![None; columns],
        }
    }

    fn insert(&mut self, mut terms: Row, mut rhs: Real) -> SpiceResult<()> {
        let mut pending: BTreeSet<_> = terms
            .keys()
            .filter_map(|&column| self.by_column[column])
            .collect();
        let mut uncertainty: Real = 0.;
        while let Some(index) = pending.pop_first() {
            let pivot = &self.pivots[index];
            let Some(factor) = terms.remove(&pivot.column) else {
                continue;
            };
            let mut rounding: Real = 0.;
            for (&column, &value) in &pivot.terms {
                let old = terms.get(&column).copied().unwrap_or(0.);
                let next = (-factor).mul_add(value, old);
                // Fused elimination has one rounding per coefficient. Use
                // the largest possible accumulated error, including the
                // uncertainty of the previously normalized pivot row.
                rounding = rounding.max(Real::EPSILON * next.abs());
                if next == 0. {
                    terms.remove(&column);
                } else {
                    terms.insert(column, next);
                    if let Some(other) = self.by_column[column] {
                        pending.insert(other);
                    }
                }
            }
            uncertainty += factor.abs() * pivot.uncertainty + rounding;
            rhs = (-factor).mul_add(pivot.rhs, rhs);
        }
        if !rhs.is_finite() || !uncertainty.is_finite() || terms.values().any(|v| !v.is_finite()) {
            return Err(failure(
                "nonfinite rank reduction of instantaneous constraints",
            ));
        }
        let Some((&column, &value)) = terms
            .iter()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
        else {
            return Ok(());
        };
        if value.abs() <= uncertainty {
            return Ok(());
        }
        terms.remove(&column);
        for coefficient in terms.values_mut() {
            *coefficient /= value;
        }
        rhs /= value;
        uncertainty = uncertainty / value.abs() + Real::EPSILON;
        if !rhs.is_finite() || !uncertainty.is_finite() {
            return Err(failure("nonfinite normalized instantaneous constraint"));
        }
        self.by_column[column] = Some(self.pivots.len());
        self.pivots.push(Pivot {
            column,
            terms,
            rhs,
            uncertainty,
        });
        Ok(())
    }

    fn solution(&self, column_exponents: &[i32]) -> SpiceResult<Vector> {
        if self.pivots.len() != self.by_column.len() {
            return Err(failure(
                "uic initial conditions cannot be shown consistent: singular instantaneous system has freedoms not resolved by storage-current gauges",
            ));
        }
        let mut solution = Vector::zeros(self.by_column.len());
        for pivot in self.pivots.iter().rev() {
            let value = pivot
                .terms
                .iter()
                .fold(pivot.rhs, |rhs, (&column, &coefficient)| {
                    (-coefficient).mul_add(solution.as_slice()[column], rhs)
                });
            solution.as_mut_slice()[pivot.column] = value;
        }
        for (value, &exponent) in solution.as_mut_slice().iter_mut().zip(column_exponents) {
            *value = scale_power(*value, -exponent);
        }
        if !solution.is_finite() {
            return Err(failure("nonfinite rank-completed instantaneous solution"));
        }
        Ok(solution)
    }
}

/// A rectangular or singular instantaneous linearization can contain dependent
/// constraints. Select an independent basis, then add only independent
/// zero-current gauge equations. Scaling by exact powers of two keeps charge,
/// voltage and current rows on comparable numerical scales. No least-squares
/// relaxation of the original constraints is performed.
pub(super) fn complete(
    matrix: &SparseMatrix,
    rhs: &[Real],
    gauges: &[usize],
    late_rows: &[usize],
) -> SpiceResult<Vector> {
    if rhs.len() != matrix.rows()
        || late_rows.iter().any(|&row| row >= matrix.rows())
        || gauges.iter().any(|&column| column >= matrix.cols())
        || rhs.iter().any(|value| !value.is_finite())
        || matrix
            .triplets()
            .iter()
            .any(|entry| !entry.value.is_finite())
    {
        return Err(failure("invalid rank-reduction input"));
    }
    let mut row_exponents = vec![i32::MIN; matrix.rows()];
    for entry in matrix.triplets() {
        if let Some(exponent) = binary_exponent(entry.value) {
            row_exponents[entry.row] = row_exponents[entry.row].max(exponent);
        }
    }
    let mut column_exponents = vec![i32::MIN; matrix.cols()];
    for entry in matrix.triplets() {
        if let Some(exponent) = binary_exponent(entry.value) {
            column_exponents[entry.col] =
                column_exponents[entry.col].max(exponent - row_exponents[entry.row]);
        }
    }
    // An empty column can only be determined by an explicit gauge.
    for exponent in &mut column_exponents {
        if *exponent == i32::MIN {
            *exponent = 0;
        }
    }
    let mut basis = Basis::new(matrix.cols());
    let entries = matrix.triplets();
    let mut ranges = Vec::with_capacity(matrix.rows());
    let mut cursor = 0;
    for row in 0..matrix.rows() {
        let start = cursor;
        while entries.get(cursor).is_some_and(|entry| entry.row == row) {
            cursor += 1;
        }
        ranges.push(start..cursor);
    }
    let mut late = vec![false; matrix.rows()];
    for &row in late_rows {
        late[row] = true;
    }
    for row in (0..matrix.rows())
        .filter(|&row| !late[row])
        .chain((0..matrix.rows()).filter(|&row| late[row]))
    {
        let exponent = row_exponents[row];
        if exponent == i32::MIN {
            continue;
        }
        let mut terms = Row::new();
        for entry in &entries[ranges[row].clone()] {
            if entry.value != 0. {
                let value = scale_power(entry.value, -exponent - column_exponents[entry.col]);
                if value != 0. {
                    terms.insert(entry.col, value);
                }
            }
        }
        basis.insert(terms, scale_power(rhs[row], -exponent))?;
    }
    for &column in gauges {
        if basis.pivots.len() == matrix.cols() {
            break;
        }
        basis.insert(Row::from([(column, 1.)]), 0.)?;
    }
    basis.solution(&column_exponents)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectangular_basis_only_completes_declared_free_coordinates() {
        let mut matrix = SparseMatrix::new(2, 3);
        matrix.add(0, 0, 1e-300).unwrap();
        matrix.add(1, 0, 1e300).unwrap();
        matrix.fold_duplicates();
        let rhs = [2e-300, 2e300];
        assert!(complete(&matrix, &rhs, &[1], &[]).is_err());
        let result = complete(&matrix, &rhs, &[1, 2], &[]).unwrap();
        assert!((result.as_slice()[0] - 2.).abs() < 1e-14);
        assert_eq!(&result.as_slice()[1..], &[0., 0.]);
    }

    #[test]
    fn algebraic_equations_precede_dependent_charge_tangents() {
        let mut matrix = SparseMatrix::new(2, 1);
        matrix.add(0, 0, 1e-6).unwrap();
        matrix.add(1, 0, 1.).unwrap();
        matrix.fold_duplicates();
        // Inconsistent linearization: the algebraic source must win over
        // the dependent charge tangent. The outer physical residual check
        // decides whether the nonlinear charge constraint is satisfied.
        let result = complete(&matrix, &[1e-6, 2.], &[], &[0]).unwrap();
        assert_eq!(result.as_slice(), &[2.]);
    }

    #[test]
    fn redundant_physical_rows_are_completed_only_by_storage_gauges() {
        let mut matrix = SparseMatrix::new(3, 3);
        for (row, col, value) in [(0, 0, 1e-15), (1, 0, 2e6), (2, 1, 1.), (2, 2, 1.)] {
            matrix.add(row, col, value).unwrap();
        }
        matrix.fold_duplicates();
        let rhs = [3e-15, 6e6, 2.];
        assert!(complete(&matrix, &rhs, &[], &[]).is_err());
        let solution = complete(&matrix, &rhs, &[1, 2], &[]).unwrap();
        assert!((solution.as_slice()[0] - 3.).abs() < 1e-14);
        assert_eq!(solution.as_slice()[1] + solution.as_slice()[2], 2.);
        assert_eq!(solution.as_slice()[1], 0.);
    }
}
