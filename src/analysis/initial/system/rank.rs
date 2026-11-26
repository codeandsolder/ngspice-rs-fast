//! Rank-aware completion of the instantaneous constraint system.
//! Only declared storage currents and derivative signals may be gauges.
//! Mandatory original equations are retained for physical verification.
//! Derived storage constraints are considered only after mandatory rows.
use std::collections::{BTreeMap, BTreeSet};

use crate::maths::{SparseMatrix, Vector};
use crate::primitives::numeric::{binary_exponent, scale_power};
use crate::primitives::{Real, SpiceResult};

use super::failure;

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
    required_prefix: usize,
    remaining_work: usize,
}

impl Basis {
    fn new(columns: usize, required_prefix: usize) -> Self {
        Self {
            pivots: Vec::new(),
            by_column: vec![None; columns],
            required_prefix,
            remaining_work: 67_108_864,
        }
    }

    fn charge(&mut self, work: usize) -> SpiceResult<()> {
        self.remaining_work = self.remaining_work.checked_sub(work).ok_or_else(|| {
            failure("instantaneous rank reduction exceeded its elimination work limit")
        })?;
        Ok(())
    }

    fn insert(&mut self, mut terms: Row, mut rhs: Real) -> SpiceResult<bool> {
        self.charge(terms.len())?;
        let mut pending: BTreeSet<_> = terms
            .keys()
            .filter_map(|&column| self.by_column[column])
            .collect();
        while let Some(index) = pending.pop_first() {
            self.charge(self.pivots[index].terms.len().saturating_add(1))?;
            let pivot = &self.pivots[index];
            let Some(factor) = terms.remove(&pivot.column) else {
                continue;
            };
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
        let Some((&column, &pivot)) = terms
            .iter()
            .filter(|(_, coefficient)| coefficient.significant())
            // Eliminate higher Taylor degrees first. A prefix pivot can
            // appear only when no significant tail coefficient survives;
            // it is then a constraint on the required initial coordinates.
            .max_by(|(ca, a), (cb, b)| {
                (**ca >= self.required_prefix)
                    .cmp(&(**cb >= self.required_prefix))
                    .then_with(|| a.value.abs().total_cmp(&b.value.abs()))
            })
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

    fn solution(&self, column_exponents: &[i32]) -> SpiceResult<Vector> {
        if self.by_column[..self.required_prefix]
            .iter()
            .any(Option::is_none)
        {
            return Err(failure(if self.required_prefix == self.by_column.len() {
                "uic initial conditions cannot be shown consistent: singular instantaneous system has freedoms not resolved by storage-coordinate gauges"
            } else {
                "derivative array leaves required initial coordinates undetermined at this order"
            }));
        }
        // Free tail coefficients are not initial gauges. The projected
        // equations proved that none changes the required prefix, so zero
        // is only a choice of particular solution for the unused tail.
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
pub(super) struct Completed {
    pub values: Vector,
    pub independent: Vec<bool>,
}

/// A rectangular or singular instantaneous linearization can contain dependent
/// constraints. Select an independent basis, then add only independent
/// zero-current gauge equations. Scaling by exact powers of two keeps charge,
/// voltage and current rows on comparable numerical scales. No least-squares
/// relaxation of the original constraints is performed.
///
/// When only an initial coordinate prefix is required, higher-order
/// columns are eliminated preferentially. Every required coordinate must
/// have a pivot independent of the free tail; unused jet coefficients
/// may remain arbitrary. The ordinary instantaneous solve requires all
/// columns and therefore retains its original pivot selection.
pub(super) fn complete(
    matrix: &SparseMatrix,
    rhs: &[Real],
    gauges: &[usize],
    late_rows: &[usize],
    optional_rows: &[usize],
    required_prefix: usize,
) -> SpiceResult<Completed> {
    let reduction = reduce(
        matrix,
        rhs,
        gauges,
        late_rows,
        optional_rows,
        required_prefix,
    )?;
    Ok(Completed {
        values: reduction.basis.solution(&reduction.column_exponents)?,
        independent: reduction.independent,
    })
}

/// The number of independent constraints on the initial-coordinate prefix
/// after projecting away all higher derivative coefficients.
pub(super) fn projected_rank(matrix: &SparseMatrix, required_prefix: usize) -> SpiceResult<usize> {
    let rhs = crate::primitives::filled_vec(matrix.rows(), 0., "derivative array rank")?;
    let reduction = reduce(matrix, &rhs, &[], &[], &[], required_prefix)?;
    Ok(reduction.basis.by_column[..required_prefix]
        .iter()
        .filter(|p| p.is_some())
        .count())
}

struct Reduction {
    basis: Basis,
    column_exponents: Vec<i32>,
    independent: Vec<bool>,
}

fn reduce(
    matrix: &SparseMatrix,
    rhs: &[Real],
    gauges: &[usize],
    late_rows: &[usize],
    optional_rows: &[usize],
    required_prefix: usize,
) -> SpiceResult<Reduction> {
    if required_prefix > matrix.cols()
        || rhs.len() != matrix.rows()
        || late_rows
            .iter()
            .chain(optional_rows)
            .any(|&row| row >= matrix.rows())
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
    let mut basis = Basis::new(matrix.cols(), required_prefix);
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
        independent[row] = basis.insert(terms, scale_power(rhs[row], -exponent))?;
    }
    for &column in gauges {
        if basis.pivots.len() == matrix.cols() {
            break;
        }
        basis.insert(Row::from([(column, Coefficient::exact(1.))]), 0.)?;
    }
    Ok(Reduction {
        basis,
        column_exponents,
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
        assert!(complete(&matrix, &rhs, &[1], &[], &[], matrix.cols()).is_err());
        let result = complete(&matrix, &rhs, &[1, 2], &[], &[], matrix.cols()).unwrap();
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
        let result = complete(&matrix, &[1e-6, 2.], &[], &[0], &[], matrix.cols()).unwrap();
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
        assert!(complete(&matrix, &rhs, &[], &[], &[], matrix.cols()).is_err());
        let solution = complete(&matrix, &rhs, &[1, 2], &[], &[], matrix.cols()).unwrap();
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
        let result = complete(
            &matrix,
            &[1., 7., 2., 8.],
            &[2],
            &[2],
            &[1, 3],
            matrix.cols(),
        )
        .unwrap();
        assert_eq!(result.values.as_slice(), &[1., 2., 0.]);
        assert_eq!(result.independent, [true, false, true, false]);
    }

    #[test]
    fn unrelated_elimination_rounding_does_not_hide_a_small_pivot() {
        // First pivot mixes column 0 and 1, but the independent column 2
        // is never touched by it. A scalar row error can lose that column.
        let mut basis = Basis::new(3, 3);
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

#[cfg(test)]
mod projected_tests {
    use super::*;

    // Exact derivative-array structure for an arbitrary linear chain:
    // x_0(t) = p(t), x_i(t) = d x_(i-1) / dt. Highest-order
    // Taylor coefficients are intentionally not assigned boundary values.
    fn derivative_chain(depth: usize, order: usize) -> (SparseMatrix, Vec<Real>) {
        let width = depth + 1;
        let mut matrix = SparseMatrix::new(order * width, (order + 1) * width);
        let mut rhs = vec![0.; order * width];
        let mut degree = 1.;
        for k in 0..order {
            let offset = k * width;
            matrix.add(offset, offset, 1.).unwrap();
            rhs[offset] = [1., 2., 3., 4.].get(k).copied().unwrap_or(0.);
            for i in 1..width {
                matrix.add(offset + i, offset + i, 1.).unwrap();
                matrix
                    .add(offset + i, offset + width + i - 1, -degree)
                    .unwrap();
            }
            degree += 1.;
        }
        matrix.fold_duplicates();
        (matrix, rhs)
    }

    #[test]
    fn derivative_depth_resolves_prefix_without_gauging_highest_jets() {
        for depth in 1..=32 {
            let width = depth + 1;
            let (short, rhs) = derivative_chain(depth, depth);
            assert!(complete(&short, &rhs, &[], &[], &[], width).is_err());
            let (matrix, rhs) = derivative_chain(depth, depth + 1);
            let result = complete(&matrix, &rhs, &[], &[], &[], width).unwrap();
            let expected = [1., 2., 6., 24.];
            for (i, &value) in result.values.as_slice()[..width].iter().enumerate() {
                assert!((value - expected.get(i).copied().unwrap_or(0.)).abs() < 1e-12);
            }
            // The full system must still be rank deficient: projected
            // uniqueness is not the same as inventing a full-rank system.
            assert!(complete(&matrix, &rhs, &[], &[], &[], matrix.cols()).is_err());
        }
    }

    #[test]
    fn genuine_feedback_storage_needs_an_independent_initial_constraint() {
        // x_1 = dx_0/dt, x_1 + x_0 = 0. The first derivative and
        // subsequent jets are fixed by the physical initial storage x_0.
        let mut matrix = SparseMatrix::new(8, 8);
        let mut degree = 1.;
        for k in 0..3 {
            let row = 2 * k;
            matrix.add(row, row + 1, 1.).unwrap();
            matrix.add(row, row + 2, -degree).unwrap();
            matrix.add(row + 1, row, 1.).unwrap();
            matrix.add(row + 1, row + 1, 1.).unwrap();
            degree += 1.;
        }
        matrix.fold_duplicates();
        assert!(complete(&matrix, &[0.; 8], &[], &[], &[], 2).is_err());
        matrix.add(6, 0, 1.).unwrap();
        // The derived condition x_1 = 0 is dependent and must not
        // displace the physical storage x_0 = 2.
        matrix.add(7, 1, 1.).unwrap();
        matrix.fold_duplicates();
        let result = complete(
            &matrix,
            &[0., 0., 0., 0., 0., 0., 2., 0.],
            &[],
            &[6],
            &[7],
            2,
        )
        .unwrap();
        assert_eq!(&result.values.as_slice()[..2], &[2., -2.]);
        assert!(result.independent[6]);
        assert!(!result.independent[7]);
    }
}

#[cfg(test)]
mod projected_oracle {
    use super::*;

    // Fraction-free integer elimination, independent of sparse floating
    // pivot selection and scaling. Tiny test matrices keep all products
    // well within i128 even without fraction reduction.
    fn exact_rank(mut rows: Vec<Vec<i128>>, columns: usize) -> usize {
        let mut rank = 0;
        for col in 0..columns {
            let Some(pivot) = (rank..rows.len()).find(|&row| rows[row][col] != 0) else {
                continue;
            };
            rows.swap(rank, pivot);
            let reference = rows[rank].clone();
            let scale = reference[col];
            for row in &mut rows[rank + 1..] {
                let factor = row[col];
                if factor == 0 {
                    continue;
                }
                for j in col + 1..columns {
                    row[j] = row[j] * scale - reference[j] * factor;
                }
                row[col] = 0;
            }
            rank += 1;
        }
        rank
    }

    #[test]
    fn projected_rank_matches_exact_integer_nullspace_dimension() {
        let mut random = 0x243f_6a88_85a3_08d3_u64;
        let mut next = || {
            random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
            (random >> 32).cast_signed()
        };
        for case in 0..512 {
            let columns = 6;
            let rows = 1 + case % 7;
            let prefix = 1 + case % columns;
            let mut entries: Vec<Vec<i128>> = (0..rows)
                .map(|_| (0..columns).map(|_| i128::from(next() % 7 - 3)).collect())
                .collect();
            // Include dependencies spanning the projected/free boundary.
            if case % 3 == 0 {
                for row in &mut entries {
                    row[columns - 1] = row[0] + row[1];
                }
            }
            if case % 5 == 0 && rows > 1 {
                entries[rows - 1] = entries[0].clone();
            }
            let tail: Vec<_> = entries.iter().map(|row| row[prefix..].to_vec()).collect();
            let determined =
                exact_rank(entries.clone(), columns) == prefix + exact_rank(tail, columns - prefix);
            let known = [2., -3., 5., 7., -1., 4.];
            let mut matrix = SparseMatrix::new(rows, columns);
            let mut rhs = vec![0.; rows];
            for (r, row) in entries.iter().enumerate() {
                for (c, &value) in row.iter().enumerate() {
                    // Every oracle coefficient is a small integer.
                    let value = Real::from(i32::try_from(value).unwrap());
                    matrix.add(r, c, value).unwrap();
                    rhs[r] += value * known[c];
                }
            }
            matrix.fold_duplicates();
            let result = complete(&matrix, &rhs, &[], &[], &[], prefix);
            assert_eq!(result.is_ok(), determined, "case {case}: {entries:?}");
            if let Ok(result) = result {
                for (&actual, &expected) in result.values.as_slice()[..prefix]
                    .iter()
                    .zip(&known[..prefix])
                {
                    assert!(
                        (actual - expected).abs() < 1e-9,
                        "case {case}: {actual} != {expected}"
                    );
                }
                let residual = matrix.mul_vector(&result.values).unwrap();
                for (&actual, &expected) in residual.as_slice().iter().zip(&rhs) {
                    assert!(
                        (actual - expected).abs() < 1e-9,
                        "case {case}: {actual} != {expected}"
                    );
                }
            }
        }
    }
}
