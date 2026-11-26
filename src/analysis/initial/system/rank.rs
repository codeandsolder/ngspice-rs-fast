//! Rank-aware completion of the instantaneous constraint system.
//! Only declared storage currents and derivative signals may be gauges.
//! Mandatory original equations are retained for physical verification.
//! Derived storage constraints are considered only after mandatory rows.
use std::collections::{BTreeMap, BTreeSet};

use crate::maths::{SparseMatrix, Vector};
use crate::primitives::numeric::binary_exponent;

mod wide;
use crate::primitives::{Real, SpiceResult};
use wide::Wide;

use super::failure;

/// A coefficient carries its own propagated rounding uncertainty. Errors
/// in unrelated columns must not suppress a small but independent pivot.
#[derive(Clone, Copy, Debug, Default)]
struct Coefficient {
    value: Wide,
    error: Wide,
}

impl Coefficient {
    fn exact(value: Real) -> Self {
        Self {
            value: Wide::from_real(value),
            error: Wide::ZERO,
        }
    }

    fn scaled(value: Real, exponent: i32) -> Self {
        Self {
            value: Wide::from_real(value).shifted(i64::from(exponent)),
            error: Wide::ZERO,
        }
    }

    const fn finite(self) -> bool {
        self.value.finite() && self.error.finite()
    }

    fn eliminate(self, factor: Self, pivot: Self) -> Self {
        let (value, roundoff) = factor
            .value
            .negate()
            .mul_add_with_roundoff(pivot.value, self.value);
        let error = factor
            .error
            .upper_mul_add(
                pivot.value.abs().upper_add(pivot.error),
                factor.value.abs().upper_mul_add(pivot.error, self.error),
            )
            .upper_add(roundoff);
        Self { value, error }
    }

    fn normalized(self, pivot: Self) -> Self {
        let (value, residual) = self.value.divide_with_residual(pivot.value);
        // For exact numerator n+dn and denominator d+dd,
        // (n+dn)-q*(d+dd) = (n-q*d)+dn-q*dd. Bound the residual
        // before dividing by the lower bound on the pivot magnitude.
        let numerator = value
            .abs()
            .upper_mul_add(pivot.error, self.error)
            .upper_add(residual);
        let error = numerator.upper_divide(pivot.value.abs().lower_subtract(pivot.error));
        Self { value, error }
    }

    fn significant(self) -> bool {
        self.value.magnitude_cmp(self.error).is_gt()
    }
}

type Row = BTreeMap<usize, Coefficient>;

struct Pivot {
    column: usize,
    terms: Row,
    rhs: Wide,
    rhs_bound: Option<Wide>,
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

    fn insert(&mut self, terms: Row, rhs: Wide) -> SpiceResult<bool> {
        self.insert_bounded(terms, rhs, None)
    }

    fn insert_bounded(
        &mut self,
        mut terms: Row,
        mut rhs: Wide,
        mut rhs_bound: Option<Wide>,
    ) -> SpiceResult<bool> {
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
                if next.value.is_zero() && next.error.is_zero() {
                    terms.remove(&column);
                } else {
                    terms.insert(column, next);
                    if let Some(other) = self.by_column[column] {
                        pending.insert(other);
                    }
                }
            }
            rhs = factor.value.negate().mul_add(pivot.rhs, rhs);
            if let (Some(bound), Some(pivot_bound)) = (&mut rhs_bound, pivot.rhs_bound) {
                *bound = factor
                    .value
                    .abs()
                    .upper_add(factor.error)
                    .upper_mul_add(pivot_bound, *bound);
            }
        }
        if !rhs.finite() || terms.values().any(|v| !v.finite()) {
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
                    .then_with(|| a.value.magnitude_cmp(b.value))
            })
        else {
            return Ok(false);
        };
        terms.remove(&column);
        for coefficient in terms.values_mut() {
            *coefficient = coefficient.normalized(pivot);
        }
        rhs = rhs.divide(pivot.value);
        if let Some(bound) = &mut rhs_bound {
            *bound = bound.upper_divide(pivot.value.abs().lower_subtract(pivot.error));
        }
        if !rhs.finite() || terms.values().any(|v| !v.finite()) {
            return Err(failure("nonfinite normalized instantaneous constraint"));
        }
        self.by_column[column] = Some(self.pivots.len());
        self.pivots.push(Pivot {
            column,
            terms,
            rhs,
            rhs_bound,
        });
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
        self.particular_solution(column_exponents)
    }

    fn prefix_rank(&self) -> usize {
        self.by_column[..self.required_prefix]
            .iter()
            .filter(|pivot| pivot.is_some())
            .count()
    }

    /// Particular Newton correction with zero corrections in unresolved
    /// coordinates. This does not prove consistency of the initial prefix.
    fn particular_solution(&self, column_exponents: &[i32]) -> SpiceResult<Vector> {
        let mut solution = vec![Wide::ZERO; self.by_column.len()];
        self.back_substitute(&mut solution, false);
        Self::unscale(solution, column_exponents)
    }

    /// A homogeneous direction along a free prefix coordinate. Its tail
    /// follows the same selected linearized equations, not artificial
    /// independent constraints on the unresolved Taylor coefficients.
    fn direction(&self, column_exponents: &[i32], column: usize) -> SpiceResult<Vector> {
        let mut solution = vec![Wide::ZERO; self.by_column.len()];
        solution[column] = Wide::from_real(1.).shifted(i64::from(column_exponents[column]));
        self.back_substitute(&mut solution, true);
        Self::unscale(solution, column_exponents)
    }

    fn back_substitute(&self, solution: &mut [Wide], homogeneous: bool) {
        for pivot in self.pivots.iter().rev() {
            let rhs = if homogeneous { Wide::ZERO } else { pivot.rhs };
            let value = pivot
                .terms
                .iter()
                .fold(rhs, |rhs, (&column, &coefficient)| {
                    coefficient.value.negate().mul_add(solution[column], rhs)
                });
            solution[pivot.column] = value;
        }
    }

    /// Forward error envelope for perturbations of the selected physical
    /// rows. This is a local linearized uncertainty, not a global root
    /// isolation certificate. Free tail perturbations are projected out.
    fn prefix_uncertainty(&self, column_exponents: &[i32]) -> SpiceResult<Vector> {
        if self.prefix_rank() != self.required_prefix {
            return Err(failure("cannot bound an undetermined derivative prefix"));
        }
        let mut bounds = vec![Wide::ZERO; self.by_column.len()];
        for pivot in self.pivots.iter().rev() {
            let rhs = pivot
                .rhs_bound
                .ok_or_else(|| failure("missing rank uncertainty input"))?;
            bounds[pivot.column] = pivot.terms.iter().fold(rhs, |sum, (&column, term)| {
                term.value
                    .abs()
                    .upper_add(term.error)
                    .upper_mul_add(bounds[column], sum)
            });
        }
        bounds.truncate(self.required_prefix);
        Self::unscale(bounds, column_exponents)
    }

    fn unscale(solution: Vec<Wide>, column_exponents: &[i32]) -> SpiceResult<Vector> {
        let solution = Vector::from(
            solution
                .into_iter()
                .zip(column_exponents)
                .map(|(value, &exponent)| value.shifted(-i64::from(exponent)).finish())
                .collect::<Vec<_>>(),
        );
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
    pub prefix_rank: usize,
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
        None,
    )?;
    Ok(Completed {
        values: reduction.basis.solution(&reduction.column_exponents)?,
        independent: reduction.independent,
        prefix_rank: reduction.basis.prefix_rank(),
    })
}

/// A provisional Newton correction to a possibly rank-deficient array.
/// Free coordinates retain their current iterate; the caller must recheck
/// the projected rank and may not accept an incomplete correction.
pub(super) fn projected_step(
    matrix: &SparseMatrix,
    rhs: &[Real],
    late_rows: &[usize],
    optional_rows: &[usize],
    required_prefix: usize,
) -> SpiceResult<Completed> {
    let reduction = reduce(
        matrix,
        rhs,
        &[],
        late_rows,
        optional_rows,
        required_prefix,
        None,
    )?;
    Ok(Completed {
        values: reduction
            .basis
            .particular_solution(&reduction.column_exponents)?,
        independent: reduction.independent,
        prefix_rank: reduction.basis.prefix_rank(),
    })
}

/// The number of independent constraints on the initial-coordinate prefix
/// after projecting away all higher derivative coefficients.
pub(super) fn projected_rank(matrix: &SparseMatrix, required_prefix: usize) -> SpiceResult<usize> {
    let rhs = crate::primitives::filled_vec(matrix.rows(), 0., "derivative array rank")?;
    let reduction = reduce(matrix, &rhs, &[], &[], &[], required_prefix, None)?;
    Ok(reduction.basis.prefix_rank())
}

/// Projected physical rank and disposable nullspace directions, with
/// optional numerical storage constraints excluded. Such constraints
/// must not prevent a search for a physically determined nonlinear root.
pub(super) struct ProbeSpace {
    pub rank: usize,
    pub directions: Vec<Vector>,
    pub independent: Vec<bool>,
}

pub(super) fn probe_space(
    matrix: &SparseMatrix,
    late: &[usize],
    optional: &[usize],
    prefix: usize,
    max_directions: usize,
) -> SpiceResult<ProbeSpace> {
    let mut mandatory = SparseMatrix::new(matrix.rows(), matrix.cols());
    let mut excluded = vec![false; matrix.rows()];
    for &row in optional {
        excluded[row] = true;
    }
    for term in matrix.triplets() {
        if !excluded[term.row] {
            mandatory.add(term.row, term.col, term.value)?;
        }
    }
    mandatory.fold_duplicates();
    let rhs = vec![0.; matrix.rows()];
    let reduction = reduce(&mandatory, &rhs, &[], late, &[], prefix, None)?;
    let rank = reduction.basis.prefix_rank();
    let count = prefix - rank;
    if count > max_directions && max_directions != 0 {
        return Err(failure(
            "nonlinear germ recovery exceeds its nullspace dimension budget",
        ));
    }
    let directions = reduction.basis.by_column[..prefix]
        .iter()
        .enumerate()
        .filter(|(_, pivot)| pivot.is_none())
        .take(max_directions)
        .map(|(column, _)| {
            reduction
                .basis
                .direction(&reduction.column_exponents, column)
        })
        .collect::<SpiceResult<Vec<_>>>()?;
    Ok(ProbeSpace {
        rank,
        directions,
        independent: reduction.independent,
    })
}

/// Propagate a residual-space error envelope through the same selected
/// basis used by the Newton correction. Bounds are in normalized jet units.
pub(super) fn projected_uncertainty(
    matrix: &SparseMatrix,
    bounds: &[Real],
    late: &[usize],
    optional: &[usize],
    prefix: usize,
) -> SpiceResult<Vector> {
    let rhs = crate::primitives::filled_vec(matrix.rows(), 0., "derivative uncertainty")?;
    let reduction = reduce(matrix, &rhs, &[], late, optional, prefix, Some(bounds))?;
    reduction
        .basis
        .prefix_uncertainty(&reduction.column_exponents)
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
    row_bounds: Option<&[Real]>,
) -> SpiceResult<Reduction> {
    if required_prefix > matrix.cols()
        || rhs.len() != matrix.rows()
        || row_bounds.is_some_and(|bounds| {
            bounds.len() != matrix.rows() || bounds.iter().any(|v| !v.is_finite() || *v < 0.)
        })
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
    // Process sparse physical rows before denser rows. In particular,
    // direct source constraints must not be buried behind a long chain
    // of equivalent normalized pivots with correlated rounding errors.
    // Preserve the order of late and optional constraints: their priority
    // is physical policy, not a matrix fill-reduction heuristic.
    let mut order: Vec<_> = (0..matrix.rows()).collect();
    order.sort_unstable_by_key(|&row| {
        let level = priority[row];
        (level, if level == 0 { ranges[row].len() } else { 0 }, row)
    });
    for row in order {
        let exponent = row_exponents[row];
        if exponent == i32::MIN {
            continue;
        }
        let mut terms = Row::new();
        for entry in &entries[ranges[row].clone()] {
            if entry.value != 0. {
                terms.insert(
                    entry.col,
                    Coefficient::scaled(entry.value, -exponent - column_exponents[entry.col]),
                );
            }
        }
        independent[row] = basis.insert_bounded(
            terms,
            Wide::from_real(rhs[row]).shifted(-i64::from(exponent)),
            row_bounds.map(|bounds| Wide::from_real(bounds[row]).shifted(-i64::from(exponent))),
        )?;
    }
    for &column in gauges {
        if basis.pivots.len() == matrix.cols() {
            break;
        }
        basis.insert(Row::from([(column, Coefficient::exact(1.))]), Wide::ZERO)?;
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
    fn projected_uncertainty_tracks_coupled_conditioning() {
        let mut matrix = SparseMatrix::new(3, 3);
        // The first two physical rows constrain x and y. An unrelated
        // free tail coordinate must not inflate their uncertainty.
        matrix.add(0, 0, 1e-9).unwrap();
        matrix.add(1, 0, 2.).unwrap();
        matrix.add(1, 1, 1.).unwrap();
        matrix.fold_duplicates();
        let bound = projected_uncertainty(&matrix, &[1e-15, 1e-12, 1e10], &[], &[], 2).unwrap();
        assert!(bound.as_slice()[0] >= 1e-6 && bound.as_slice()[0] < 1.00001e-6);
        assert!(bound.as_slice()[1] >= 2e-6 && bound.as_slice()[1] < 2.00001e-6);
    }

    #[test]
    fn nonlinear_probe_directions_preserve_mandatory_constraints() {
        let mut matrix = SparseMatrix::new(2, 3);
        matrix.add(0, 0, 2.).unwrap();
        matrix.add(0, 1, 1.).unwrap();
        matrix.add(1, 1, 1.).unwrap();
        matrix.fold_duplicates();
        let space = probe_space(&matrix, &[], &[1], 2, 2).unwrap();
        assert_eq!(space.rank, 1);
        assert_eq!(space.directions.len(), 1);
        let direction = space.directions[0].as_slice();
        assert!(direction[0].abs().max(direction[1].abs()) >= 1.);
        assert!(2_f64.mul_add(direction[0], direction[1]).abs() < 1e-14);
        assert_eq!(space.independent, [true, false]);
    }

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
                    Wide::from_real(4.)
                )
                .unwrap()
        );
        assert!(
            basis
                .insert(
                    Row::from([(1, Coefficient::exact(1.))]),
                    Wide::from_real(2.)
                )
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
                    Wide::from_real(4.)
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
    fn derivative_chain(depth: usize, order: usize, scale: Real) -> (SparseMatrix, Vec<Real>) {
        let width = depth + 1;
        let mut matrix = SparseMatrix::new(order * width, (order + 1) * width);
        let mut rhs = vec![0.; order * width];
        let mut degree = 1.;
        for k in 0..order {
            let offset = k * width;
            matrix.add(offset, offset, 1.).unwrap();
            rhs[offset] = [1., 2. * scale, 3. * scale.powi(2), 4. * scale.powi(3)]
                .get(k)
                .copied()
                .unwrap_or(0.);
            for i in 1..width {
                matrix.add(offset + i, offset + i, 1.).unwrap();
                matrix
                    .add(offset + i, offset + width + i - 1, -degree / scale)
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
            let (short, rhs) = derivative_chain(depth, depth, 1.);
            assert!(complete(&short, &rhs, &[], &[], &[], width).is_err());
            let (matrix, rhs) = derivative_chain(depth, depth + 1, 1.);
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
    fn rank_and_solution_do_not_depend_on_taylor_time_normalization() {
        for depth in [8, 32, 60] {
            let width = depth + 1;
            for scale in [1e-12, 1e-6, 1., 1e6, 1e12] {
                let (matrix, rhs) = derivative_chain(depth, depth + 2, scale);
                assert_eq!(
                    projected_rank(&matrix, 2 * width).unwrap(),
                    2 * width,
                    "depth={depth}, scale={scale:e}"
                );
                let result = complete(&matrix, &rhs, &[], &[], &[], 2 * width).unwrap();
                for (index, &value) in result.values.as_slice()[..width].iter().enumerate() {
                    let expected = [1., 2., 6., 24.].get(index).copied().unwrap_or(0.);
                    assert!(
                        (value - expected).abs() < 1e-11,
                        "depth={depth}, scale={scale:e}, x[{index}]={value:e}"
                    );
                }
            }
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
