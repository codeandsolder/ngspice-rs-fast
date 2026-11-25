//! Rank-aware sparse constraint reduction.
//! Only declared storage currents and derivative signals may be gauges.
//! Mandatory original equations are retained for physical verification.
//! Derived storage constraints are considered only after mandatory rows.
use std::collections::{BTreeMap, BTreeSet};

use crate::maths::{SparseMatrix, Vector};
use crate::primitives::numeric::{binary_exponent, scale_power};
use crate::primitives::{Real, SpiceResult};

fn failure(message: impl Into<String>) -> crate::primitives::SpiceError {
    crate::primitives::SpiceError::Numerical {
        context: "initial conditions".to_owned(),
        message: message.into(),
    }
}

/// A coefficient carries its own propagated rounding uncertainty. Errors
/// in unrelated columns must not suppress a small but independent pivot.
#[derive(Clone, Copy, Debug, Default)]
struct Coefficient {
    value: Real,
    error: Real,
}

impl Coefficient {
    const fn exact(value: Real) -> Self {
        Self { value, error: 0. }
    }

    const fn finite(self) -> bool {
        self.value.is_finite() && self.error.is_finite()
    }

    fn eliminate(self, factor: Self, pivot: Self) -> Self {
        let value = (-factor.value).mul_add(pivot.value, self.value);
        let error = Real::EPSILON.mul_add(
            value.abs(),
            factor.error.mul_add(
                pivot.value.abs() + pivot.error,
                factor.value.abs().mul_add(pivot.error, self.error),
            ),
        );
        Self { value, error }
    }

    fn normalized(self, pivot: Self) -> Self {
        let value = self.value / pivot.value;
        let error = Real::EPSILON.mul_add(
            value.abs(),
            value.abs().mul_add(pivot.error, self.error) / (pivot.value.abs() - pivot.error),
        );
        Self { value, error }
    }

    fn significant(self) -> bool {
        self.value.abs() > self.error
    }
}

type Row = BTreeMap<usize, Coefficient>;

struct Pivot {
    column: usize,
    terms: Row,
    rhs: Real,
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

    fn reduce(
        &self,
        mut terms: Row,
        mut rhs: Real,
        budget: &mut usize,
    ) -> SpiceResult<(Row, Real)> {
        let mut pending: BTreeSet<_> = terms
            .keys()
            .filter_map(|&column| self.by_column[column])
            .collect();
        while let Some(index) = pending.pop_first() {
            let pivot = &self.pivots[index];
            let Some(factor) = terms.remove(&pivot.column) else {
                continue;
            };
            *budget = budget
                .checked_sub(pivot.terms.len() + 1)
                .ok_or_else(|| failure("derivative-array rank work limit exceeded"))?;
            for (&column, &value) in &pivot.terms {
                let old = terms.get(&column).copied().unwrap_or_default();
                let next = old.eliminate(factor, value);
                if next.value == 0. && next.error == 0. {
                    terms.remove(&column);
                } else {
                    terms.insert(column, next);
                    if let Some(other) = self.by_column[column] {
                        pending.insert(other);
                    }
                }
            }
            rhs = (-factor.value).mul_add(pivot.rhs, rhs);
        }
        if !rhs.is_finite() || terms.values().any(|v| !v.finite()) {
            return Err(failure(
                "nonfinite rank reduction of instantaneous constraints",
            ));
        }
        Ok((terms, rhs))
    }

    fn insert_bounded(&mut self, terms: Row, rhs: Real, budget: &mut usize) -> SpiceResult<bool> {
        let (mut terms, mut rhs) = self.reduce(terms, rhs, budget)?;
        let Some((&column, &pivot)) = terms
            .iter()
            .filter(|(_, coefficient)| coefficient.significant())
            .max_by(|(_, a), (_, b)| a.value.abs().total_cmp(&b.value.abs()))
        else {
            return Ok(false);
        };
        terms.remove(&column);
        for coefficient in terms.values_mut() {
            *coefficient = coefficient.normalized(pivot);
        }
        rhs /= pivot.value;
        if !rhs.is_finite() || terms.values().any(|v| !v.finite()) {
            return Err(failure("nonfinite normalized instantaneous constraint"));
        }
        self.by_column[column] = Some(self.pivots.len());
        self.pivots.push(Pivot { column, terms, rhs });
        Ok(true)
    }

    fn insert(&mut self, terms: Row, rhs: Real) -> SpiceResult<bool> {
        let mut unlimited = usize::MAX;
        self.insert_bounded(terms, rhs, &mut unlimited)
    }

    fn solution(&self, column_exponents: &[i32]) -> SpiceResult<Vector> {
        if self.pivots.len() != self.by_column.len() {
            return Err(failure(
                "uic initial conditions cannot be shown consistent: singular instantaneous system has freedoms not resolved by storage-coordinate gauges",
            ));
        }
        self.representative(column_exponents)
    }

    fn representative(&self, column_exponents: &[i32]) -> SpiceResult<Vector> {
        let mut solution = Vector::zeros(self.by_column.len());
        for pivot in self.pivots.iter().rev() {
            let value = pivot
                .terms
                .iter()
                .fold(pivot.rhs, |rhs, (&column, &coefficient)| {
                    (-coefficient.value).mul_add(solution.as_slice()[column], rhs)
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

/// Result of rank selection. Independent original rows are identified
/// separately from synthetic gauge rows so optional initial expressions
/// can be checked against the actual selected physical state.
pub(crate) struct Completed {
    pub values: Vector,
    pub independent: Vec<bool>,
}

/// A rectangular or singular instantaneous linearization can contain dependent
/// constraints. Select an independent basis, then add only independent
/// zero-current gauge equations. Scaling by exact powers of two keeps charge,
/// voltage and current rows on comparable numerical scales. No least-squares
/// relaxation of the original constraints is performed.
pub(crate) fn complete(
    matrix: &SparseMatrix,
    rhs: &[Real],
    gauges: &[usize],
    late_rows: &[usize],
    optional_rows: &[usize],
) -> SpiceResult<Completed> {
    if gauges.iter().any(|&column| column >= matrix.cols()) {
        return Err(failure("invalid rank gauge"));
    }
    let mut unlimited = usize::MAX;
    let (mut basis, column_exponents, independent) =
        reduce_rows(matrix, rhs, late_rows, optional_rows, &mut unlimited)?;
    for &column in gauges {
        if basis.pivots.len() == matrix.cols() {
            break;
        }
        basis.insert(Row::from([(column, Coefficient::exact(1.))]), 0.)?;
    }
    Ok(Completed {
        values: basis.solution(&column_exponents)?,
        independent,
    })
}

fn reduce_rows(
    matrix: &SparseMatrix,
    rhs: &[Real],
    late_rows: &[usize],
    optional_rows: &[usize],
    budget: &mut usize,
) -> SpiceResult<(Basis, Vec<i32>, Vec<bool>)> {
    if rhs.len() != matrix.rows()
        || late_rows
            .iter()
            .chain(optional_rows)
            .any(|&row| row >= matrix.rows())
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
    let mut priority = vec![0_u8; matrix.rows()];
    for &row in late_rows {
        priority[row] = 1;
    }
    for &row in optional_rows {
        priority[row] = 2;
    }
    let mut independent = vec![false; matrix.rows()];
    let priority = &priority;
    for row in
        (0..=2).flat_map(|level| (0..matrix.rows()).filter(move |&row| priority[row] == level))
    {
        let exponent = row_exponents[row];
        if exponent == i32::MIN {
            continue;
        }
        let mut terms = Row::new();
        for entry in &entries[ranges[row].clone()] {
            if entry.value != 0. {
                let value = scale_power(entry.value, -exponent - column_exponents[entry.col]);
                if value == 0. {
                    return Err(failure(
                        "instantaneous rank scaling would lose a nonzero coefficient",
                    ));
                }
                terms.insert(entry.col, Coefficient::exact(value));
            }
        }
        *budget = budget
            .checked_sub(terms.len() + 1)
            .ok_or_else(|| failure("derivative-array rank work limit exceeded"))?;
        independent[row] = basis.insert_bounded(terms, scale_power(rhs[row], -exponent), budget)?;
    }
    Ok((basis, column_exponents, independent))
}

/// A particular point, the row-space determination of requested columns,
/// and the selection of independent rows. Unresolved free coefficients
/// are zero only in the representative; they are not physical conditions.
pub(super) struct Projection {
    pub(super) representative: Vector,
    pub(super) determined: Vec<bool>,
    pub(super) independent: Vec<bool>,
}

/// Project a rectangular constraint system onto the requested coordinates.
/// Row-space membership is checked independently of any arbitrary gauge.
pub(super) fn project(
    matrix: &SparseMatrix,
    rhs: &[Real],
    optional_rows: &[usize],
    requested: usize,
    mut budget: usize,
) -> SpiceResult<Projection> {
    if requested > matrix.cols() || !matrix.is_folded() {
        return Err(failure("invalid derivative-array projection shape"));
    }
    let (basis, exponents, independent) =
        reduce_rows(matrix, rhs, &[], optional_rows, &mut budget)?;
    let representative = basis.representative(&exponents)?;
    let mut determined = Vec::with_capacity(requested);
    for column in 0..requested {
        // A coordinate is unique exactly when its unit row is in the row
        // span. Eliminate that row with the very same numerical rank basis.
        // No arbitrary zero of a free higher derivative can make it unique.
        let (free, _) = basis.reduce(
            Row::from([(column, Coefficient::exact(1.))]),
            0.,
            &mut budget,
        )?;
        determined.push(!free.values().any(|value| value.significant()));
    }
    Ok(Projection {
        representative,
        determined,
        independent,
    })
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
        assert!(complete(&matrix, &rhs, &[1], &[], &[]).is_err());
        let result = complete(&matrix, &rhs, &[1, 2], &[], &[]).unwrap();
        assert!((result.values.as_slice()[0] - 2.).abs() < 1e-14);
        assert_eq!(&result.values.as_slice()[1..], &[0., 0.]);
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
        let result = complete(&matrix, &[1e-6, 2.], &[], &[0], &[]).unwrap();
        assert_eq!(result.values.as_slice(), &[2.]);
    }

    #[test]
    fn redundant_physical_rows_are_completed_only_by_storage_gauges() {
        let mut matrix = SparseMatrix::new(3, 3);
        for (row, col, value) in [(0, 0, 1e-15), (1, 0, 2e6), (2, 1, 1.), (2, 2, 1.)] {
            matrix.add(row, col, value).unwrap();
        }
        matrix.fold_duplicates();
        let rhs = [3e-15, 6e6, 2.];
        assert!(complete(&matrix, &rhs, &[], &[], &[]).is_err());
        let solution = complete(&matrix, &rhs, &[1, 2], &[], &[]).unwrap();
        assert!((solution.values.as_slice()[0] - 3.).abs() < 1e-14);
        assert_eq!(
            solution.values.as_slice()[1] + solution.values.as_slice()[2],
            2.
        );
        assert_eq!(solution.values.as_slice()[1], 0.);
    }
}

#[cfg(test)]
mod priority_tests {
    use super::*;

    #[test]
    fn independent_derived_rows_follow_mandatory_storage() {
        let mut matrix = SparseMatrix::new(4, 3);
        for (row, col, value) in [(0, 0, 1.), (1, 0, 2.), (2, 1, 1.), (3, 1, 3.)] {
            matrix.add(row, col, value).unwrap();
        }
        matrix.fold_duplicates();
        let result = complete(&matrix, &[1., 7., 2., 8.], &[2], &[2], &[1, 3]).unwrap();
        assert_eq!(result.values.as_slice(), &[1., 2., 0.]);
        assert_eq!(result.independent, [true, false, true, false]);
    }

    #[test]
    fn unrelated_elimination_rounding_does_not_hide_a_small_pivot() {
        // First pivot mixes column 0 and 1, but the independent column 2
        // is never touched by it. A scalar row error can lose that column.
        let mut basis = Basis::new(3);
        assert!(
            basis
                .insert(
                    Row::from([(0, Coefficient::exact(2.)), (1, Coefficient::exact(1.))]),
                    4.
                )
                .unwrap()
        );
        assert!(
            basis
                .insert(Row::from([(1, Coefficient::exact(1.))]), 2.)
                .unwrap()
        );
        assert!(
            basis
                .insert(
                    Row::from([
                        (0, Coefficient::exact(2.)),
                        (1, Coefficient::exact(1.)),
                        (2, Coefficient::exact(1e-30))
                    ]),
                    4.
                )
                .unwrap()
        );
        let result = basis.solution(&[0, 0, 0]).unwrap();
        assert_eq!(result.as_slice(), &[1., 2., 0.]);
    }
}
