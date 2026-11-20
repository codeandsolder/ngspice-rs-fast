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
    #[must_use]
    pub fn row_factors(&self) -> &[f64] {
        &self.rows
    }

    /// Column factors multiplying the matrix and the scaled solution.
    #[must_use]
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
        if x.iter().any(|v| !v.is_finite()) {
            return Err(numerical("equilibration", "non-finite balancing exponent"));
        }
        Ok(Self {
            rows: x[..n].iter().copied().map(balanced_factor).collect(),
            columns: x[n..].iter().copied().map(balanced_factor).collect(),
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
    crate::primitives::value::binary_exponent(v)
        .ok_or_else(|| numerical("equilibration", "invalid binary exponent"))
}

/// Quantize a finite least-squares exponent to the supported power-of-two range.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the rounded exponent is finite and clamped to the exact i32 interval -512..=512"
)]
fn balanced_factor(exponent: f64) -> f64 {
    let limit = f64::from(MAX_SCALING_EXPONENT);
    2.0_f64.powi(exponent.round().clamp(-limit, limit) as i32)
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

#[expect(
    clippy::suboptimal_flops,
    reason = "retain separately rounded residual-bound products to preserve existing acceptance thresholds"
)]
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
        bounds[r] += v.magnitude() * max_x;
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
        let mut scaled = Matrix::zeros(matrix.rows(), matrix.cols());
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
    #[must_use]
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
    #[must_use]
    pub const fn scaling(&self) -> &Equilibration {
        &self.scaling
    }

    /// [`EquilibratedSparseLu::solve`] followed by at most `steps` rounds of
    /// iterative refinement in physical units. A correction `M dx = b - M x`
    /// is kept only while it lowers the componentwise backward error
    /// `max_r |b - M x|_r / (sum_c |M[r,c]| |x_c| + |b_r|)`.
    ///
    /// The normwise residual check of [`EquilibratedSparseLu::solve`] accepts
    /// errors relative to the largest solution component; when the solution
    /// spans tens of decades (a `1e99` unknown beside a `2 V` source), a small
    /// component can be lost entirely within that bound. Refinement recovers
    /// it without changing the factor or any guard.
    /// # Errors
    /// As [`EquilibratedSparseLu::solve`], for the first solve; failing
    /// corrections end refinement and keep the best solution so far.
    pub fn solve_refined(&self, rhs: &Vector, steps: usize) -> SpiceResult<Vector> {
        let mut workspace = Vec::new();
        let mut x = self.solve_in(rhs, &mut workspace)?;
        let mut residual = rhs.clone();
        let mut scale = vec![0.0; rhs.len()];
        let mut error = self.componentwise_error(rhs, &x, &mut residual, &mut scale);
        for _ in 0..steps {
            if error == 0.0 {
                break;
            }
            // The error calculation leaves the physical residual in the
            // workspace for the correction solve.
            if !residual.is_finite() {
                break;
            }
            let Ok(mut candidate) = self.solve_in(&residual, &mut workspace) else {
                break;
            };
            for (delta, value) in candidate.as_mut_slice().iter_mut().zip(x.as_slice()) {
                // Both operands are finite here; binary64 addition is
                // commutative, including signed zeros.
                *delta += *value;
            }
            let candidate_error =
                self.componentwise_error(rhs, &candidate, &mut residual, &mut scale);
            let improves = candidate_error.partial_cmp(&error) == Some(std::cmp::Ordering::Less);
            if !improves || self.check_residual(rhs, &candidate).is_err() {
                break;
            }
            x = candidate;
            error = candidate_error;
        }
        Ok(x)
    }

    fn componentwise_error(
        &self,
        rhs: &Vector,
        x: &Vector,
        residual: &mut Vector,
        scale: &mut [f64],
    ) -> f64 {
        let b = rhs.as_slice();
        let residual = residual.as_mut_slice();
        residual.copy_from_slice(b);
        for (s, v) in scale.iter_mut().zip(b) {
            *s = v.abs();
        }
        for &(r, c, v) in &self.original {
            let term = v * x.as_slice()[c];
            residual[r] -= term;
            scale[r] += term.abs();
        }
        residual
            .iter()
            .zip(scale.iter())
            .map(|(r, s)| match (r.abs(), *s) {
                (0.0, _) => 0.0,
                (r, s) if s > 0.0 => r / s,
                _ => f64::INFINITY,
            })
            .fold(0.0, f64::max)
    }

    /// Reusable symbolic factors for an identical assembled sparse pattern.
    #[must_use]
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
        let scaled_rhs = Equilibration::transform_real(rhs, &self.scaling.rows)?;
        let y = self.factor.solve_with_workspace(&scaled_rhs, workspace)?;
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
    #[must_use]
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
