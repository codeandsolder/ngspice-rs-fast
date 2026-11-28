//! Explicit, bounded row/column equilibration around the existing checked LU APIs.
//!
//! Scaling does not supply a uniqueness theorem or forward-error guarantee.
//!
//! For `M x = b`, factors solve `(R M C) y = R b` and return `x = C y`.
//! Original assembled snapshots check residuals in physical units. This is a
//! maths-library opt-in with no production caller: the circuit solves follow
//! ngspice, which neither scales nor checks residuals (`spconfig.h` builds
//! SPARSE without `SCALING`), so Newton's former equilibrated fallback was
//! removed with the native solver (M10 wave 0.5,
//! `docs/port/SPARSE_RANK_DIAGNOSTICS.md`).
//! `src/maths/sparse/sputils.c::spScale` is the read-only behavioral reference
//! for row/RHS and column/solution transforms; the bounded max-based and
//! Curtis-Reid selections here are independent policies, not a copied solver
//! or C-option parity claim.

use crate::maths::complex::{ComplexLu, ComplexMatrix};
use crate::maths::linear::{assembled_triplets, numerical, square};
use crate::maths::{DenseLu, Matrix, SparseLu, SparseMatrix, SparseSymbolic, Triplet, Vector};
use crate::primitives::{Complex, SpiceResult};

/// Largest absolute exponent of any row or column factor (`2^±512`).
pub const MAX_SCALING_EXPONENT: i32 = 512;

/// Extra conjugate-gradient steps, beyond `4 n`, of the Curtis-Reid balancing
/// behind [`EquilibratedSparseLu::new_balanced`].
pub const MAX_BALANCING_ITERATIONS: usize = 64;

type Entry<T> = (usize, usize, T);

/// Scaling and residual arithmetic is generic over the original numeric
/// domain. Real systems need neither complex snapshots nor promoted RHSs.
trait ScaleValue:
    Copy + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self> + std::ops::Mul<Output = Self>
{
    const ZERO: Self;
    fn component_max(self) -> f64;
    fn magnitude(self) -> f64;
    fn finite(self) -> bool;
    fn scaled(self, factor: f64) -> SpiceResult<Self>;
}

impl ScaleValue for f64 {
    const ZERO: Self = 0.0;
    fn component_max(self) -> f64 {
        self.abs()
    }
    fn magnitude(self) -> f64 {
        self.abs()
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn scaled(self, factor: f64) -> SpiceResult<Self> {
        scale(self, factor)
    }
}

impl ScaleValue for Complex {
    const ZERO: Self = Self::ZERO;
    fn component_max(self) -> f64 {
        self.re.abs().max(self.im.abs())
    }
    fn magnitude(self) -> f64 {
        Self::magnitude(self)
    }
    fn finite(self) -> bool {
        self.is_finite()
    }
    fn scaled(self, factor: f64) -> SpiceResult<Self> {
        Ok(Self::new(scale(self.re, factor)?, scale(self.im, factor)?))
    }
}

/// Immutable positive power-of-two factors for `R M C`.
///
/// The default selection is a single row-max pass followed by a column-max pass
/// over the largest absolute real/imaginary component, not a condition-number
/// estimate; [`EquilibratedSparseLu::new_balanced`] instead uses Curtis-Reid
/// least-squares balancing. Factors are clamped independently to
/// `2^-512 ..= 2^512`. No permutation or sparsity change occurs.
#[derive(Debug, Clone, PartialEq)]
pub struct Equilibration {
    rows: Vec<f64>,
    columns: Vec<f64>,
}

impl Equilibration {
    /// Row factors multiplying the matrix and RHS.
    #[must_use = "the factors are needed to scale matrix rows and the right-hand side consistently"]
    pub fn row_factors(&self) -> &[f64] {
        &self.rows
    }

    /// Column factors multiplying the matrix and the scaled solution.
    #[must_use = "the factors are needed to scale matrix columns and recover the original solution"]
    pub fn column_factors(&self) -> &[f64] {
        &self.columns
    }

    fn new<T: ScaleValue>(n: usize, entries: &[Entry<T>]) -> SpiceResult<Self> {
        let mut rows = vec![0.0_f64; n];
        for &(r, _, v) in entries {
            rows[r] = rows[r].max(v.component_max());
        }
        let rows = factors(&rows, "row")?;
        let mut columns = vec![0.0_f64; n];
        for &(r, c, v) in entries {
            columns[c] = columns[c].max(v.scaled(rows[r])?.component_max());
        }
        Ok(Self {
            rows,
            columns: factors(&columns, "column")?,
        })
    }

    /// Curtis-Reid power-of-two balancing: row and column exponents that
    /// minimise `sum (log2|a_rc| + r_r + c_c)^2` over the stored entries.
    ///
    /// The least-squares normal equations form a bipartite graph Laplacian,
    /// solved here by conjugate gradients (at most `4 n +`
    /// [`MAX_BALANCING_ITERATIONS`] steps) and rounded to integer exponents.
    /// Unlike the single row-then-column max pass of [`Equilibration::new`]
    /// (or an infinity-norm Ruiz iteration, whose fixed points can keep it),
    /// this balances a row whose large coefficient is coupled through a cycle
    /// to a column dominated by a small one: a behavioural-source row
    /// `v(out) - J v(in)` with `|J| ~ 1e32` next to a 1 mS conductance needs
    /// both that row and the `v(out)` column scaled by about `J`. The method
    /// is A. R. Curtis and J. K. Reid, "On the automatic scaling of matrices
    /// for Gaussian elimination", J. Inst. Maths Applics 10 (1972); it is a
    /// heuristic, so callers keep every rank and residual guard.
    fn balanced(n: usize, entries: &[Entry<f64>]) -> SpiceResult<Self> {
        let logs: Vec<(usize, usize, f64)> = entries
            .iter()
            .filter(|(_, _, v)| v.abs() != 0.0)
            .map(|&(r, c, v)| {
                Ok((
                    r,
                    c,
                    f64::from(floor_log2(v.component_max(), || "entry".to_owned())?),
                ))
            })
            .collect::<SpiceResult<_>>()?;
        let mut counts = vec![0_usize; 2 * n];
        for &(r, c, _) in &logs {
            counts[r] += 1;
            counts[n + c] += 1;
        }
        if let Some(empty) = counts.iter().position(|&count| count == 0) {
            let (namespace, index) = if empty < n {
                ("row", empty)
            } else {
                ("column", empty - n)
            };
            return Err(numerical(
                "equilibration",
                format!("all-zero/non-finite {namespace} {index}"),
            ));
        }
        // Unknowns: row exponents 0..n, then column exponents n..2n.
        let laplacian = |x: &[f64], y: &mut [f64]| {
            y.fill(0.0);
            for &(r, c, _) in &logs {
                let sum = x[r] + x[n + c];
                y[r] += sum;
                y[n + c] += sum;
            }
        };
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
        let mut rhs = vec![0.0; 2 * n];
        for &(r, c, l) in &logs {
            rhs[r] -= l;
            rhs[n + c] -= l;
        }
        let mut x = vec![0.0; 2 * n];
        let mut residual = rhs;
        let mut direction = residual.clone();
        let mut image = vec![0.0; 2 * n];
        let mut norm = dot(&residual, &residual);
        // Rounding to integer exponents needs far less than this accuracy.
        let tolerance = 1e-12 * norm.max(1.0);
        for _ in 0..4 * n + MAX_BALANCING_ITERATIONS {
            if norm <= tolerance {
                break;
            }
            laplacian(&direction, &mut image);
            let curvature = dot(&direction, &image);
            if !(curvature.is_finite() && curvature > 0.0) {
                break;
            }
            let step = norm / curvature;
            for ((x, r), (d, a)) in x
                .iter_mut()
                .zip(residual.iter_mut())
                .zip(direction.iter().zip(&image))
            {
                *x += step * d;
                *r -= step * a;
            }
            let next = dot(&residual, &residual);
            let beta = next / norm;
            for (d, r) in direction.iter_mut().zip(&residual) {
                *d = r + beta * *d;
            }
            norm = next;
        }
        Ok(Self {
            rows: x[..n]
                .iter()
                .copied()
                .map(balanced_factor)
                .collect::<SpiceResult<_>>()?,
            columns: x[n..]
                .iter()
                .copied()
                .map(balanced_factor)
                .collect::<SpiceResult<_>>()?,
        })
    }

    fn scale_entries<'a, T: ScaleValue + 'a>(
        &'a self,
        entries: &'a [Entry<T>],
    ) -> impl Iterator<Item = SpiceResult<Entry<T>>> + 'a {
        entries
            .iter()
            .map(|&(r, c, v)| Ok((r, c, v.scaled(self.rows[r])?.scaled(self.columns[c])?)))
    }

    fn transform_real(values: &Vector, factors: &[f64]) -> SpiceResult<Vector> {
        transform(values.as_slice(), factors).map(Vector::from)
    }

    fn transform_complex(values: &[Complex], factors: &[f64]) -> SpiceResult<Vec<Complex>> {
        transform(values, factors)
    }
}

fn transform<T: ScaleValue>(values: &[T], factors: &[f64]) -> SpiceResult<Vec<T>> {
    if values.len() != factors.len() {
        return Err(numerical(
            "equilibration transform",
            "vector dimension mismatch",
        ));
    }
    values
        .iter()
        .zip(factors)
        .map(|(&v, &s)| v.scaled(s))
        .collect()
}

fn floor_log2(v: f64, what: impl FnOnce() -> String) -> SpiceResult<i32> {
    if v <= 0.0 || !v.is_finite() {
        return Err(numerical(
            "equilibration",
            format!("nonpositive/non-finite {}", what()),
        ));
    }
    crate::primitives::numeric::binary_exponent(v)
        .ok_or_else(|| numerical("equilibration", "invalid binary exponent"))
}

/// Quantize a least-squares exponent without changing the supported scale range.
fn balanced_factor(exponent: f64) -> SpiceResult<f64> {
    crate::primitives::numeric::bounded_rounded_exponent(
        exponent,
        -MAX_SCALING_EXPONENT..=MAX_SCALING_EXPONENT,
    )
    .map(|exponent| 2.0_f64.powi(exponent))
    .ok_or_else(|| numerical("equilibration", "non-finite balancing exponent"))
}

fn factors(maxima: &[f64], namespace: &str) -> SpiceResult<Vec<f64>> {
    maxima
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let exponent = floor_log2(v, || format!("{namespace} {i}"))?;
            Ok(2.0_f64.powi((-exponent).clamp(-MAX_SCALING_EXPONENT, MAX_SCALING_EXPONENT)))
        })
        .collect()
}

fn scale(v: f64, s: f64) -> SpiceResult<f64> {
    let out = v * s;
    if !v.is_finite() || !out.is_finite() || (v != 0.0 && (out == 0.0 || out.is_subnormal())) {
        return Err(numerical(
            "equilibration transform",
            "non-finite value, overflow or destructive underflow",
        ));
    }
    Ok(out)
}

fn real_entries(matrix: &SparseMatrix) -> Vec<Entry<f64>> {
    matrix
        .triplets()
        .iter()
        .map(|t| (t.row, t.col, t.value))
        .collect()
}

fn dense_entries(matrix: &Matrix) -> SpiceResult<Vec<Entry<f64>>> {
    square(matrix.rows(), matrix.cols())?;
    if matrix.data().iter().any(|v| !v.is_finite()) {
        return Err(numerical(
            "equilibration assembly",
            "non-finite coefficient",
        ));
    }
    Ok(matrix
        .data()
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0.)
        .map(|(i, &v)| (i / matrix.cols(), i % matrix.cols(), v))
        .collect())
}

fn real_operator(
    n: usize,
    entries: impl Iterator<Item = SpiceResult<Entry<f64>>>,
) -> SpiceResult<SparseMatrix> {
    let triplets = entries
        .map(|entry| {
            let (row, col, value) = entry?;
            Ok(Triplet { row, col, value })
        })
        .collect::<SpiceResult<Vec<_>>>()?;
    Ok(SparseMatrix::from_assembled(n, n, triplets))
}

/// Merge two folded row-major operators without a coordinate tree or
/// any change in the original single-element scaling operation.
fn complex_entries(a: &[Triplet], e: &[Triplet], omega: f64) -> SpiceResult<Vec<Entry<Complex>>> {
    let mut a = a.iter().peekable();
    let mut e = e.iter().peekable();
    let key = |t: &Triplet| (t.row, t.col);
    let mut entries = Vec::new();
    while a.peek().is_some() || e.peek().is_some() {
        let (row, col) = match (a.peek(), e.peek()) {
            (Some(a), Some(e)) => key(a).min(key(e)),
            (Some(a), None) => key(a),
            (None, Some(e)) => key(e),
            (None, None) => break,
        };
        let re = a.next_if(|t| key(t) == (row, col)).map_or(0., |t| t.value);
        let im = match e.next_if(|t| key(t) == (row, col)) {
            Some(t) if omega != 0. => scale(t.value, omega)?,
            _ => 0.,
        };
        let v = Complex::new(re, im);
        if v != Complex::ZERO {
            entries.push((row, col, v));
        }
    }
    Ok(entries)
}

/// Dimensionless per-equation backward errors against the original operator.
///
/// This is evidence, not an automatic circuit rejection or a forward-error
/// guarantee. All terms are accumulated exactly before one normalized ratio.
#[derive(Debug, Clone, PartialEq)]
pub struct ComponentwiseBackwardError {
    /// Per-row residual divided by the original-row absolute product sum
    /// and RHS magnitude. Complex values use the sum of their components
    /// for a finite, conservative bound at arbitrarily large input scales.
    pub rows: Vec<f64>,
    /// Largest componentwise backward error.
    pub maximum: f64,
}

/// Borrowed, original-unit real or complex sparse operator.
///
/// Uses one reusable equation traversal. Preparation allocates only an index, not coefficient
/// snapshots. No diagnostics run on the ordinary factor/solve fast path.
#[derive(Debug)]
pub struct OriginalResidual<'a, T> {
    order: usize,
    entries: &'a [(usize, usize, T)],
    by_equation: Vec<usize>,
    transposed: bool,
}

trait ResidualScalar: ScaleValue {
    fn correction(
        real: &crate::primitives::ExactSum,
        imaginary: &crate::primitives::ExactSum,
    ) -> Self;
    fn product<const MEASURE: bool>(
        left: Self,
        right: Self,
        real: &mut crate::primitives::ExactSum,
        imaginary: &mut crate::primitives::ExactSum,
        denominator: &mut crate::primitives::ExactSum,
    );
    fn rhs<const MEASURE: bool>(
        rhs: Self,
        real: &mut crate::primitives::ExactSum,
        imaginary: &mut crate::primitives::ExactSum,
        denominator: &mut crate::primitives::ExactSum,
    );
}

impl ResidualScalar for f64 {
    fn correction(
        real: &crate::primitives::ExactSum,
        _imaginary: &crate::primitives::ExactSum,
    ) -> Self {
        -real.rounded_sum()
    }

    fn product<const MEASURE: bool>(
        left: Self,
        right: Self,
        real: &mut crate::primitives::ExactSum,
        _imaginary: &mut crate::primitives::ExactSum,
        denominator: &mut crate::primitives::ExactSum,
    ) {
        real.add_product(left, right);
        if MEASURE {
            denominator.add_product(left.abs(), right.abs());
        }
    }

    fn rhs<const MEASURE: bool>(
        rhs: Self,
        real: &mut crate::primitives::ExactSum,
        _imaginary: &mut crate::primitives::ExactSum,
        denominator: &mut crate::primitives::ExactSum,
    ) {
        real.add_product(-rhs, 1.);
        if MEASURE {
            denominator.add_product(rhs.abs(), 1.);
        }
    }
}

impl ResidualScalar for Complex {
    fn correction(
        real: &crate::primitives::ExactSum,
        imaginary: &crate::primitives::ExactSum,
    ) -> Self {
        Self::new(-real.rounded_sum(), -imaginary.rounded_sum())
    }

    fn product<const MEASURE: bool>(
        left: Self,
        right: Self,
        real: &mut crate::primitives::ExactSum,
        imaginary: &mut crate::primitives::ExactSum,
        denominator: &mut crate::primitives::ExactSum,
    ) {
        real.add_product(left.re, right.re);
        real.add_product(-left.im, right.im);
        imaginary.add_product(left.re, right.im);
        imaginary.add_product(left.im, right.re);
        if MEASURE {
            for (a, b) in [
                (left.re, right.re),
                (left.im, right.im),
                (left.re, right.im),
                (left.im, right.re),
            ] {
                denominator.add_product(a.abs(), b.abs());
            }
        }
    }

    fn rhs<const MEASURE: bool>(
        rhs: Self,
        real: &mut crate::primitives::ExactSum,
        imaginary: &mut crate::primitives::ExactSum,
        denominator: &mut crate::primitives::ExactSum,
    ) {
        real.add_product(-rhs.re, 1.);
        imaginary.add_product(-rhs.im, 1.);
        if MEASURE {
            denominator.add_product(rhs.re.abs(), 1.);
            denominator.add_product(rhs.im.abs(), 1.);
        }
    }
}

fn original_residual_prepare<T: ResidualScalar>(
    order: usize,
    entries: &[Entry<T>],
    transposed: bool,
) -> SpiceResult<OriginalResidual<'_, T>> {
    if order == 0
        || entries
            .iter()
            .any(|&(row, col, value)| row >= order || col >= order || !value.finite())
    {
        return Err(numerical(
            "original operator residual",
            "nonpositive dimension, out-of-bounds entry or non-finite coefficient",
        ));
    }
    let mut by_equation = Vec::new();
    by_equation.try_reserve_exact(entries.len()).map_err(|e| {
        numerical(
            "original operator residual",
            format!("cannot allocate index: {e}"),
        )
    })?;
    by_equation.extend(0..entries.len());
    by_equation.sort_unstable_by_key(|&idx| {
        let (row, col, _) = entries[idx];
        if transposed { col } else { row }
    });
    Ok(OriginalResidual {
        order,
        entries,
        by_equation,
        transposed,
    })
}

fn original_residual_measure<T: ResidualScalar, const MEASURE: bool>(
    view: &OriginalResidual<'_, T>,
    rhs: &[T],
    solution: &[T],
    mut correction: Option<&mut [T]>,
) -> SpiceResult<ComponentwiseBackwardError> {
    if rhs.len() != view.order
        || solution.len() != view.order
        || rhs.iter().chain(solution).any(|&value| !value.finite())
        || correction
            .as_ref()
            .is_some_and(|values| values.len() != view.order)
    {
        return Err(numerical(
            "original operator residual",
            "invalid RHS/solution dimensions or non-finite values",
        ));
    }
    let mut rows = Vec::new();
    if MEASURE {
        rows.try_reserve_exact(view.order).map_err(|e| {
            numerical(
                "original operator residual",
                format!("cannot allocate rows: {e}"),
            )
        })?;
    }
    let mut cursor = 0;
    let mut maximum = 0.0_f64;
    for (row, &rhs_value) in rhs.iter().enumerate() {
        let mut real = crate::primitives::ExactSum::new();
        let mut imaginary = crate::primitives::ExactSum::new();
        let mut denominator = crate::primitives::ExactSum::new();
        while let Some(&index) = view.by_equation.get(cursor) {
            let (source_row, source_col, value) = view.entries[index];
            let (actual_row, actual_col) = if view.transposed {
                (source_col, source_row)
            } else {
                (source_row, source_col)
            };
            if actual_row != row {
                break;
            }
            T::product::<MEASURE>(
                value,
                solution[actual_col],
                &mut real,
                &mut imaginary,
                &mut denominator,
            );
            cursor += 1;
        }
        T::rhs::<MEASURE>(rhs_value, &mut real, &mut imaginary, &mut denominator);
        if let Some(values) = correction.as_deref_mut() {
            values[row] = T::correction(&real, &imaginary);
        }
        if MEASURE {
            let relative = real
                .magnitude_ratio(&denominator)
                .max(imaginary.magnitude_ratio(&denominator));
            maximum = maximum.max(relative);
            rows.push(relative);
        }
    }
    Ok(ComponentwiseBackwardError { rows, maximum })
}

fn original_residual_many<T: ResidualScalar>(
    view: &OriginalResidual<'_, T>,
    systems: &[(&[T], &[T])],
) -> SpiceResult<Vec<ComponentwiseBackwardError>> {
    let mut reports = Vec::new();
    reports.try_reserve_exact(systems.len()).map_err(|e| {
        numerical(
            "original operator residual",
            format!("cannot allocate reports: {e}"),
        )
    })?;
    for &(rhs, solution) in systems {
        reports.push(original_residual_measure::<T, true>(
            view, rhs, solution, None,
        )?);
    }
    Ok(reports)
}

impl<'a> OriginalResidual<'a, f64> {
    /// Prepare original real operator diagnostics. Duplicate entries are
    /// accumulated exactly. A transposed view uses the same borrowed data.
    /// # Errors
    /// Invalid dimensions, coefficients or insufficient index storage.
    pub fn real(order: usize, entries: &'a [Entry<f64>], transposed: bool) -> SpiceResult<Self> {
        original_residual_prepare(order, entries, transposed)
    }

    /// Inspect one real solution against the original physical equations.
    /// # Errors
    /// Invalid RHS or solution, or insufficient report storage.
    pub fn evaluate(
        &self,
        rhs: &[f64],
        solution: &[f64],
    ) -> SpiceResult<ComponentwiseBackwardError> {
        original_residual_measure::<_, true>(self, rhs, solution, None)
    }

    /// Inspect one solution while rounding each exact physical correction
    /// RHS only after accumulating its complete signed product row. A
    /// nonrepresentable correction becomes infinite and must not be solved.
    /// # Errors
    /// Invalid dimensions/finite inputs or insufficient report storage.
    pub fn evaluate_with_correction(
        &self,
        rhs: &[f64],
        solution: &[f64],
        correction: &mut [f64],
    ) -> SpiceResult<ComponentwiseBackwardError> {
        original_residual_measure::<_, true>(self, rhs, solution, Some(correction))
    }

    /// Form only the exact residual correction, avoiding normalization
    /// products and diagnostic report allocation when the caller uses
    /// coordinate updates rather than backward-error ratios.
    /// # Errors
    /// Invalid dimensions or non-finite inputs.
    pub(crate) fn correction_into(
        &self,
        rhs: &[f64],
        solution: &[f64],
        correction: &mut [f64],
    ) -> SpiceResult<()> {
        original_residual_measure::<_, false>(self, rhs, solution, Some(correction)).map(drop)
    }

    /// Form the exact residual of both solution words against the unfolded
    /// stamps. Retain both words when rounding each correction RHS.
    pub(super) fn wide_correction(
        &self,
        rhs: &[f64],
        solution: &[crate::primitives::wide::Wide],
        out: &mut [crate::primitives::wide::Wide],
    ) -> SpiceResult<()> {
        use crate::primitives::{ExactSum, wide::Wide};
        if rhs.len() != self.order
            || solution.len() != self.order
            || out.len() != self.order
            || rhs.iter().any(|value| !value.is_finite())
            || solution.iter().any(|value| !value.finite())
        {
            return Err(numerical(
                "original operator residual",
                "invalid promoted RHS/solution",
            ));
        }
        let mut cursor = 0;
        for (row, (&rhs_value, correction)) in rhs.iter().zip(out).enumerate() {
            let mut sum = ExactSum::new();
            sum.add_product(rhs_value, 1.);
            while let Some(&index) = self.by_equation.get(cursor) {
                let (r, c, value) = self.entries[index];
                let (r, c) = if self.transposed { (c, r) } else { (r, c) };
                if r != row {
                    break;
                }
                solution[c].add_product(-value, &mut sum);
                cursor += 1;
            }
            *correction = Wide::from_sum(sum);
        }
        Ok(())
    }

    /// Inspect many real RHS and solution pairs, reusing the same operator.
    /// # Errors
    /// Invalid input or insufficient report storage.
    pub fn evaluate_many(
        &self,
        systems: &[(&[f64], &[f64])],
    ) -> SpiceResult<Vec<ComponentwiseBackwardError>> {
        original_residual_many(self, systems)
    }
}

impl<'a> OriginalResidual<'a, Complex> {
    /// Prepare original complex operator diagnostics in normal or transposed
    /// orientation, without projecting away imaginary contributions.
    /// # Errors
    /// Invalid dimensions, coefficients or insufficient index storage.
    pub fn complex(
        order: usize,
        entries: &'a [Entry<Complex>],
        transposed: bool,
    ) -> SpiceResult<Self> {
        original_residual_prepare(order, entries, transposed)
    }

    /// Inspect one complex solution and both residual components.
    /// # Errors
    /// Invalid RHS or solution, or insufficient report storage.
    pub fn evaluate(
        &self,
        rhs: &[Complex],
        solution: &[Complex],
    ) -> SpiceResult<ComponentwiseBackwardError> {
        original_residual_measure::<_, true>(self, rhs, solution, None)
    }

    /// Report exact complex-product residual quality while rounding both
    /// physical correction components once into caller-owned storage.
    /// # Errors
    /// Invalid dimensions/finite inputs or insufficient report storage.
    pub fn evaluate_with_correction(
        &self,
        rhs: &[Complex],
        solution: &[Complex],
        correction: &mut [Complex],
    ) -> SpiceResult<ComponentwiseBackwardError> {
        original_residual_measure::<_, true>(self, rhs, solution, Some(correction))
    }

    /// Inspect many complex RHS and solution pairs with a shared traversal.
    /// # Errors
    /// Invalid input or insufficient report storage.
    pub fn evaluate_many(
        &self,
        systems: &[(&[Complex], &[Complex])],
    ) -> SpiceResult<Vec<ComponentwiseBackwardError>> {
        original_residual_many(self, systems)
    }
}

/// Original-unit row residual magnitudes and unchanged LU backward-error bounds.
///
/// A successful check guarantees each residual is at most its bound, not forward
/// accuracy: `bound[r] = 128 ε n (sum_c |M[r,c]| ||x||inf + |b[r]|)`.
#[derive(Debug, Clone, PartialEq)]
pub struct BackwardError {
    /// `|M x - b|` per row (physical equation units).
    pub row_residuals: Vec<f64>,
    /// Permitted normwise backward error per row, in the same units.
    pub row_bounds: Vec<f64>,
}

/// The opt-in LU residual policy uses the dimension as a floating-point scale.
#[expect(
    clippy::cast_precision_loss,
    reason = "the error budget is a floating-point scale rather than an integer index"
)]
fn residual_tolerance(n: usize) -> f64 {
    128.0 * f64::EPSILON * n as f64
}

fn check_residual<T: ScaleValue>(
    n: usize,
    entries: &[Entry<T>],
    rhs: &[T],
    x: &[T],
) -> SpiceResult<BackwardError> {
    if rhs.len() != n || x.len() != n || rhs.iter().chain(x).any(|v| !v.finite()) {
        return Err(numerical(
            "equilibration original residual",
            "invalid RHS/solution dimensions or values",
        ));
    }
    let max_x = x.iter().fold(0.0_f64, |m, v| m.max(v.magnitude()));
    let mut ax = vec![T::ZERO; n];
    let mut bounds: Vec<_> = rhs.iter().map(|v| v.magnitude()).collect();
    for &(r, c, v) in entries {
        ax[r] = ax[r] + v * x[c];
        bounds[r] = v.magnitude().mul_add(max_x, bounds[r]);
    }
    let tol = residual_tolerance(n);
    let mut residuals = Vec::with_capacity(n);
    for r in 0..n {
        let error = (ax[r] - rhs[r]).magnitude();
        if !error.is_finite()
            || !bounds[r].is_finite()
            || !max_x.is_finite()
            || error > tol * bounds[r]
        {
            return Err(numerical(
                "equilibration original residual",
                format!("backward residual failed at row {r}"),
            ));
        }
        residuals.push(error);
        bounds[r] *= tol;
    }
    Ok(BackwardError {
        row_residuals: residuals,
        row_bounds: bounds,
    })
}

/// Owned opt-in dense LU of `R M C` plus original-unit diagnostics.
#[derive(Debug)]
pub struct EquilibratedDenseLu {
    original: Vec<Entry<f64>>,
    scaling: Equilibration,
    factor: DenseLu,
}

impl EquilibratedDenseLu {
    /// Equilibrates a finite nonempty square snapshot, retaining dense LU guards.
    /// # Errors
    /// Invalid assembly, zero rows/columns, failed transform or unresolved rank.
    pub fn new(matrix: &Matrix) -> SpiceResult<Self> {
        let original = dense_entries(matrix)?;
        let scaling = Equilibration::new(matrix.rows(), &original)?;
        let mut scaled = Matrix::zeros(matrix.rows(), matrix.cols())?;
        for entry in scaling.scale_entries(&original) {
            let (r, c, v) = entry?;
            scaled.set(r, c, v)?;
        }
        Ok(Self {
            original,
            scaling,
            factor: DenseLu::new(&scaled)?,
        })
    }

    /// Immutable scaling metadata belonging to this factor snapshot.
    #[must_use = "the scaling metadata belongs to this real factor snapshot and must match its solves"]
    pub const fn scaling(&self) -> &Equilibration {
        &self.scaling
    }

    /// Solves a new physical-unit RHS; no caller storage is mutated.
    /// # Errors
    /// Invalid RHS, failed transforms, checked solve or original-unit residual.
    pub fn solve(&self, rhs: &Vector) -> SpiceResult<Vector> {
        let scaled_rhs = Equilibration::transform_real(rhs, &self.scaling.rows)?;
        let y = self.factor.solve(&scaled_rhs)?;
        let x = Equilibration::transform_real(&y, &self.scaling.columns)?;
        self.check_residual(rhs, &x)?;
        Ok(x)
    }

    /// Checks a physical-unit candidate against the original assembled snapshot.
    /// # Errors
    /// Invalid dimensions/values, residual arithmetic overflow or excessive error.
    pub fn check_residual(&self, rhs: &Vector, x: &Vector) -> SpiceResult<BackwardError> {
        check_residual(
            self.scaling.rows.len(),
            &self.original,
            rhs.as_slice(),
            x.as_slice(),
        )
    }
}

/// Hard upper bound for explicit post-factorization refinement rounds.
/// A caller must request fewer rounds instead of silently running forever.
pub const MAX_REFINEMENT_STEPS: usize = 8;

/// Explicit policy for an optional original-equation iterative refinement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefinementPolicy {
    /// Maximum attempted correction rounds, at most `MAX_REFINEMENT_STEPS`.
    pub max_steps: usize,
    /// Desired maximum componentwise backward error, finite and in [0, 1].
    pub target_backward_error: f64,
}

/// Why bounded refinement stopped. A stopped solve still returns its best
/// known candidate with original-unit measured quality for caller decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefinementStatus {
    /// The requested componentwise backward-error target was met.
    TargetMet,
    /// Every allowed round was spent without reaching the target.
    Exhausted,
    /// A correction failed to reduce the exact backward-error ratio.
    Stalled,
    /// The correction residual or corrected solution was not representable
    /// as finite binary64 data despite a representable normalized error.
    UnrepresentableCorrection,
    /// The existing factors could not solve a correction; includes the
    /// numerical error for explicit fallback handling.
    CorrectionSolveFailed(String),
}

/// Best solution and original-operator evidence after bounded refinement.
#[derive(Debug)]
pub struct RefinedSolution {
    /// Best finite physical-unit solution.
    pub solution: Vector,
    /// Scale-aware per-row errors measured at the final solution.
    pub quality: ComponentwiseBackwardError,
    /// Number of accepted corrections (each strictly improved the metric).
    pub corrections: usize,
    /// Convergence or precise nonconvergence/fallback reason.
    pub status: RefinementStatus,
}

/// Owned opt-in sparse LU with exact assembled-pattern symbolic reuse.
#[derive(Debug)]
pub struct EquilibratedSparseLu {
    original: Vec<Entry<f64>>,
    scaling: Equilibration,
    factor: SparseLu,
}

impl EquilibratedSparseLu {
    /// Folds/validates original duplicates before scaling and factoring a snapshot.
    /// Reuse is allowed only for the identical assembled pattern, independent of
    /// previous factors' scaling values. Nonzero scaling cannot change the pattern.
    /// # Errors
    /// Invalid assembly, transforms, zero rows/columns, changed pattern or rank.
    pub fn new(matrix: &SparseMatrix, reuse: Option<&SparseSymbolic>) -> SpiceResult<Self> {
        let snapshot = assembled_triplets(matrix, "equilibration assembly")?;
        let original = real_entries(&snapshot);
        let scaling = Equilibration::new(snapshot.rows(), &original)?;
        let scaled = real_operator(snapshot.rows(), scaling.scale_entries(&original))?;
        Ok(Self {
            original,
            scaling,
            factor: SparseLu::new(&scaled, reuse)?,
        })
    }

    /// Like [`EquilibratedSparseLu::new`], but with power-of-two
    /// row/column balancing (Curtis-Reid least squares in exponent space) instead of the single
    /// row-then-column pass. The factor of the balanced `R M C` keeps every
    /// existing guard: the numerical rank/conditioning check applies to the
    /// scaled system, and every solve is checked against the original snapshot
    /// in physical units.
    /// # Errors
    /// Invalid assembly, transforms, zero rows/columns, changed pattern or rank.
    pub fn new_balanced(
        matrix: &SparseMatrix,
        reuse: Option<&SparseSymbolic>,
    ) -> SpiceResult<Self> {
        let snapshot = assembled_triplets(matrix, "equilibration assembly")?;
        let original = real_entries(&snapshot);
        let scaling = Equilibration::balanced(snapshot.rows(), &original)?;
        let scaled = real_operator(snapshot.rows(), scaling.scale_entries(&original))?;
        Ok(Self {
            original,
            scaling,
            factor: SparseLu::new(&scaled, reuse)?,
        })
    }

    /// Immutable scaling metadata belonging to this factor snapshot.
    #[must_use = "the scaling metadata belongs to this factor snapshot and must match its solves"]
    pub const fn scaling(&self) -> &Equilibration {
        &self.scaling
    }

    /// Refine the physical-unit solution against the original sparse
    /// operator, using the existing factors for at most the requested
    /// number of correction solves.
    ///
    /// Row residuals, normalization, and correction RHSs are exact binary64
    /// product sums, each rounded once only when physically representable.
    /// The best finite candidate is retained only on strict improvement in
    /// maximum componentwise backward error. An unattainable target returns
    /// a structured nonconvergence status, never a silently relaxed bound.
    /// The ordinary solve path and its C compatibility are unaffected.
    ///
    /// # Errors
    /// An invalid policy, failed initial factor solve or invalid original
    /// operator/RHS data. Later correction failures are reported as status.
    pub fn solve_refined(
        &self,
        rhs: &Vector,
        policy: RefinementPolicy,
    ) -> SpiceResult<RefinedSolution> {
        if policy.max_steps > MAX_REFINEMENT_STEPS
            || !policy.target_backward_error.is_finite()
            || !(0. ..=1.).contains(&policy.target_backward_error)
        {
            return Err(numerical(
                "equilibration refinement",
                "invalid maximum corrections or target backward error",
            ));
        }
        let original = OriginalResidual::real(self.scaling.rows.len(), &self.original, false)?;
        let mut workspace = Vec::new();
        let mut x = self.solve_factors_in(rhs, &mut workspace)?;
        let mut residual = Vector::zeros(rhs.len());
        let mut quality = original.evaluate_with_correction(
            rhs.as_slice(),
            x.as_slice(),
            residual.as_mut_slice(),
        )?;
        let mut corrections = 0;
        let status = loop {
            if quality.maximum <= policy.target_backward_error {
                break RefinementStatus::TargetMet;
            }
            if corrections >= policy.max_steps {
                break RefinementStatus::Exhausted;
            }
            if !residual.is_finite() {
                break RefinementStatus::UnrepresentableCorrection;
            }
            let mut candidate = match self.solve_factors_in(&residual, &mut workspace) {
                Ok(solution) => solution,
                Err(error) => break RefinementStatus::CorrectionSolveFailed(error.to_string()),
            };
            for (delta, value) in candidate.as_mut_slice().iter_mut().zip(x.as_slice()) {
                *delta += *value;
            }
            if !candidate.is_finite() {
                break RefinementStatus::UnrepresentableCorrection;
            }
            let candidate_quality = original.evaluate_with_correction(
                rhs.as_slice(),
                candidate.as_slice(),
                residual.as_mut_slice(),
            )?;
            if candidate_quality.maximum >= quality.maximum {
                break RefinementStatus::Stalled;
            }
            x = candidate;
            quality = candidate_quality;
            corrections += 1;
        };
        Ok(RefinedSolution {
            solution: x,
            quality,
            corrections,
            status,
        })
    }

    /// Reusable symbolic factors for an identical assembled sparse pattern.
    #[must_use = "retaining the symbolic factors enables reuse only for an identical sparse pattern"]
    pub const fn symbolic(&self) -> &SparseSymbolic {
        self.factor.symbolic()
    }

    /// Solves a new physical-unit RHS against the owned snapshot.
    /// # Errors
    /// Invalid RHS, failed transforms, checked solve or original-unit residual.
    pub fn solve(&self, rhs: &Vector) -> SpiceResult<Vector> {
        self.solve_in(rhs, &mut Vec::new())
    }

    fn solve_in(&self, rhs: &Vector, workspace: &mut Vec<f64>) -> SpiceResult<Vector> {
        let x = self.solve_factors_in(rhs, workspace)?;
        self.check_residual(rhs, &x)?;
        Ok(x)
    }

    fn solve_factors_in(&self, rhs: &Vector, workspace: &mut Vec<f64>) -> SpiceResult<Vector> {
        let scaled_rhs = Equilibration::transform_real(rhs, &self.scaling.rows)?;
        let y = self.factor.solve_with_workspace(&scaled_rhs, workspace)?;
        Equilibration::transform_real(&y, &self.scaling.columns)
    }

    /// Checks a physical-unit candidate against the original assembled snapshot.
    /// # Errors
    /// Invalid dimensions/values, residual arithmetic overflow or excessive error.
    pub fn check_residual(&self, rhs: &Vector, x: &Vector) -> SpiceResult<BackwardError> {
        check_residual(
            self.scaling.rows.len(),
            &self.original,
            rhs.as_slice(),
            x.as_slice(),
        )
    }
}

/// Owned opt-in complex LU of the assembled `A + j omega E` snapshot.
#[derive(Debug)]
pub struct EquilibratedComplexLu {
    original: Vec<Entry<Complex>>,
    scaling: Equilibration,
    factor: ComplexLu,
}

impl EquilibratedComplexLu {
    /// Folds original A/E duplicates, assembles frequency, then scales and factors.
    /// Frequency is finite and nonnegative. Assembly underflow is an error too.
    /// # Errors
    /// Invalid dimensions/frequency/coefficients, transforms, zero rows/columns or rank.
    pub fn from_operators(a: &SparseMatrix, e: &SparseMatrix, omega: f64) -> SpiceResult<Self> {
        let a = assembled_triplets(a, "equilibration assembly")?;
        let e = assembled_triplets(e, "equilibration assembly")?;
        if a.rows() != e.rows() || !omega.is_finite() || omega < 0.0 {
            return Err(numerical(
                "equilibration complex assembly",
                "operator dimensions/frequency invalid",
            ));
        }
        let original = complex_entries(a.triplets(), e.triplets(), omega)?;
        let scaling = Equilibration::new(a.rows(), &original)?;
        let scaled = scaling
            .scale_entries(&original)
            .collect::<SpiceResult<Vec<_>>>()?;
        let factor = ComplexMatrix::from_assembled(a.rows(), scaled).factorize()?;
        Ok(Self {
            original,
            scaling,
            factor,
        })
    }

    /// Immutable scaling metadata belonging to this factor snapshot.
    #[must_use = "the scaling metadata belongs to this factor snapshot and must match its solves"]
    pub const fn scaling(&self) -> &Equilibration {
        &self.scaling
    }

    /// Solves a complex physical-unit RHS, preserving voltage/current phases.
    /// # Errors
    /// Invalid RHS, failed transforms, checked solve or original-unit residual.
    pub fn solve(&self, rhs: &[Complex]) -> SpiceResult<Vec<Complex>> {
        let scaled_rhs = Equilibration::transform_complex(rhs, &self.scaling.rows)?;
        let y = self.factor.solve(&scaled_rhs)?;
        let x = Equilibration::transform_complex(&y, &self.scaling.columns)?;
        self.check_residual(rhs, &x)?;
        Ok(x)
    }

    /// Checks a physical-unit candidate against the original complex assembly.
    /// # Errors
    /// Invalid dimensions/values, residual arithmetic overflow or excessive error.
    pub fn check_residual(&self, rhs: &[Complex], x: &[Complex]) -> SpiceResult<BackwardError> {
        check_residual(self.scaling.rows.len(), &self.original, rhs, x)
    }
}

#[cfg(test)]
mod floor_log2_tests {
    use super::floor_log2;

    #[test]
    fn assembled_scaled_operator_preserves_zero_elision() {
        use super::*;
        let scaled =
            real_operator(2, [(0, 0, 2.), (0, 1, -0.), (1, 1, 3.)].into_iter().map(Ok)).unwrap();
        assert!(scaled.is_folded());
        assert_eq!(scaled.nnz(), 2);
        assert_eq!(scaled.get(0, 1), 0.);
    }

    #[test]
    fn real_residual_matches_complex_promotion_and_merge_preserves_order() {
        use super::*;
        let real = [(0, 0, 2.), (0, 1, -1.), (1, 0, 1.), (1, 1, 4.)];
        let complex = real.map(|(r, c, v)| (r, c, Complex::real(v)));
        let x = [1.25, -0.5];
        let b = [3., -0.75];
        assert_eq!(
            check_residual(2, &real, &b, &x).unwrap(),
            check_residual(2, &complex, &b.map(Complex::real), &x.map(Complex::real)).unwrap()
        );
        assert_eq!(
            Equilibration::new(2, &real).unwrap(),
            Equilibration::new(2, &complex).unwrap()
        );
        let a = [
            Triplet {
                row: 0,
                col: 0,
                value: 2.,
            },
            Triplet {
                row: 1,
                col: 0,
                value: -1.,
            },
        ];
        let e = [
            Triplet {
                row: 0,
                col: 0,
                value: 0.5,
            },
            Triplet {
                row: 0,
                col: 1,
                value: -2.,
            },
        ];
        assert_eq!(
            complex_entries(&a, &e, 2.).unwrap(),
            [
                (0, 0, Complex::new(2., 1.)),
                (0, 1, Complex::imaginary(-4.)),
                (1, 0, Complex::real(-1.))
            ]
        );
        assert_eq!(
            complex_entries(&a, &e, 0.).unwrap(),
            [(0, 0, Complex::real(2.)), (1, 0, Complex::real(-1.))]
        );
    }

    #[test]
    fn correction_only_matches_diagnostic_in_both_orientations() {
        use super::*;
        let stamps = [
            (0, 0, 1e12),
            (0, 0, 1e-3),
            (0, 1, -1e12),
            (1, 0, -2e12),
            (1, 1, 2e12),
            (1, 1, 1e-3),
        ];
        let x = [1., 1.];
        let b = [0.002, 0.003];
        for transpose in [false, true] {
            let operator = OriginalResidual::real(2, &stamps, transpose).unwrap();
            let mut diagnostic = [0.; 2];
            let mut correction = [0.; 2];
            operator
                .evaluate_with_correction(&b, &x, &mut diagnostic)
                .unwrap();
            operator.correction_into(&b, &x, &mut correction).unwrap();
            assert_eq!(correction, diagnostic);
            assert!(
                operator
                    .correction_into(&b, &x, &mut correction[..1])
                    .is_err()
            );
        }
    }

    #[test]
    fn exact_exponent_boundaries_and_invalid_signs() {
        for (value, exponent) in [
            (f64::from_bits(1), -1074),
            (f64::from_bits(2), -1073),
            (f64::from_bits(f64::MIN_POSITIVE.to_bits() - 1), -1023),
            (f64::MIN_POSITIVE, -1022),
            (0.5, -1),
            (1., 0),
            (f64::from_bits(1f64.to_bits() - 1), -1),
            (2., 1),
            (f64::MAX, 1023),
        ] {
            assert_eq!(
                floor_log2(value, || "test".to_owned()).unwrap(),
                exponent,
                "{value:?}"
            );
        }
        for value in [
            0.,
            -0.,
            -1.,
            -f64::MIN_POSITIVE,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert!(
                floor_log2(value, || "test".to_owned()).is_err(),
                "{value:?}"
            );
        }
    }
}
