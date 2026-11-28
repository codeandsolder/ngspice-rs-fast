//! Owned LU factors from the SPARSE 1.3 port (`maths::sparse13`).
//! Matrices remain plain, cloneable stamping storage.
//!
//! This follows ngspice's `spsmp.c` use of SPARSE: a fresh pattern is
//! MNA-preordered (`SMPpreOrder`) and ordered with threshold pivoting
//! (`SMPreorder`, `spOrderAndFactor` with the matrix's [`PivotTolerances`],
//! C's `CKTpivotRelTol`/`CKTpivotAbsTol`); a reused pattern is refactored with the
//! existing pivot order (`SMPluFac`, `spFactor`) and, on a zero pivot,
//! reordered as `NIiter` does after `E_SINGULAR`. Like C there is no
//! condition estimate or residual check: only structural/numeric singularity
//! (`spSINGULAR`/`spZERO_DIAG`) and non-finite solutions are errors, and small
//! pivots are accepted (`spSMALL_PIVOT` is `OK` in ngspice, which says
//! nothing; the port prints a warning, see [`solver_warning`]). A singular
//! circuit matrix names its unknowns as `NIiter`'s warning does
//! (`singular matrix:  check node X`).
//!
//! An unknown with no allocated matrix entry is structurally singular.
//! SPARSE leaves such indices untranslated and copies their RHS to the
//! solution, but the circuit/linear factorization boundary rejects that
//! behavior: an injected current is not a node voltage.
//!
//! Elements are created in the folded triplet order (row-major), which sets
//! SPARSE's internal numbering. ngspice instead creates them in device setup
//! order, so pivot choices can differ from C where Markowitz ties occur.

use std::borrow::Cow;
use std::sync::Arc;

mod recovery;

use recovery::RetainedOriginal;

use crate::maths::sparse13::{Scalar, Sparse, Status};
use crate::maths::{FactorStorage, FactorWork, Matrix, Pattern, PatternWork, SparseMatrix, Vector};
use crate::primitives::{Complex, SpiceError, SpiceResult};

/// Number of independent right-hand sides in a sparse solve tile.
pub const RHS_BATCH_WIDTH: usize = 4;

/// Reusable, scalar-typed sparse multiple-RHS solve scratch.
///
/// Internal storage is tiled by matrix row, with one lane per RHS.
/// Regardless of the total RHS count, scratch holds only one tile and one scalar tail.
#[derive(Debug)]
pub struct SolveBatchWorkspace<T> {
    pub(super) real: Vec<std::simd::f64x4>,
    pub(super) imag: Vec<std::simd::f64x4>,
    pub(super) scalar: Vec<T>,
}

impl<T> Default for SolveBatchWorkspace<T> {
    fn default() -> Self {
        Self {
            real: Vec::new(),
            imag: Vec::new(),
            scalar: Vec::new(),
        }
    }
}

/// A batched solve failure, retaining the first affected RHS for callers
/// that attach independent circuit excitation or parameter identities.
#[derive(Debug)]
pub struct BatchSolveFailure {
    /// Zero-based RHS index, or none for a shape, factor or resource error.
    pub rhs_index: Option<usize>,
    /// The underlying numerical/storage diagnostic.
    pub cause: SpiceError,
}

impl From<SpiceError> for BatchSolveFailure {
    fn from(cause: SpiceError) -> Self {
        Self {
            rhs_index: None,
            cause,
        }
    }
}

impl From<BatchSolveFailure> for SpiceError {
    fn from(error: BatchSolveFailure) -> Self {
        error.cause
    }
}

impl std::fmt::Display for BatchSolveFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(index) = self.rhs_index {
            write!(f, "RHS {index}: ")?;
        }
        self.cause.fmt(f)
    }
}

impl std::error::Error for BatchSolveFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}

/// Sparse pivot tolerances used when ordering a circuit matrix.
///
/// SPARSE threshold-pivoting limits for ordering a circuit matrix: C's
/// `CKTpivotAbsTol` (`.options pivtol`) and `CKTpivotRelTol` (`.options
/// pivrel`), passed to every `SMPreorder`/`SMPcReorder` (`niiter.c`,
/// `niaciter.c`). A refactor with an existing order (`spFactor`) does not
/// read them.
///
/// `spOrderAndFactor` silently falls back to its stored threshold for a
/// relative value outside `(0, 1]` or a negative absolute value; here such
/// values are rejected by [`Self::validate`] instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PivotTolerances {
    /// Smallest acceptable pivot magnitude (`pivtol`, default `1e-13`).
    pub absolute: f64,
    /// Smallest acceptable ratio of a pivot to the largest magnitude in its
    /// column (`pivrel`, default `1e-3`).
    pub relative: f64,
}

impl PivotTolerances {
    /// ngspice's defaults (`cktntask.c`): `pivtol = 1e-13`, `pivrel = 1e-3`.
    pub const DEFAULT: Self = Self {
        absolute: 1.0e-13,
        relative: 1.0e-3,
    };

    /// Checks `0 <= absolute` and `0 < relative <= 1`, both finite.
    ///
    /// # Errors
    /// A message naming the offending option.
    pub fn validate(&self) -> Result<(), String> {
        if !self.absolute.is_finite() || self.absolute < 0. {
            return Err(format!(
                "pivtol must be finite and >= 0, not {}",
                self.absolute
            ));
        }
        if !self.relative.is_finite() || self.relative <= 0. || self.relative > 1. {
            return Err(format!("pivrel must be in (0, 1], not {}", self.relative));
        }
        Ok(())
    }
}

impl Default for PivotTolerances {
    fn default() -> Self {
        Self::DEFAULT
    }
}

pub(crate) fn numerical(context: &str, message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: context.into(),
        message: message.into(),
    }
}

pub(crate) fn square(rows: usize, cols: usize) -> SpiceResult<()> {
    if rows != cols || rows == 0 {
        return Err(numerical(
            "LU",
            format!("expected a nonempty square system, got {rows}x{cols}"),
        ));
    }
    Ok(())
}

/// Names of a circuit's unknowns by matrix row (C `NODENAME`/`CKTnodName`):
/// node names and `<device>#branch` currents.
pub type UnknownNames = Arc<[String]>;

/// The name of 1-based external index `ext`, or `unknown <row>` (0-based
/// row) without names.
fn unknown_label(names: Option<&UnknownNames>, ext: usize) -> String {
    let row = ext.saturating_sub(1);
    names
        .and_then(|names| names.get(row))
        .cloned()
        .unwrap_or_else(|| format!("unknown {row}"))
}

/// At most this many solver warnings are printed per process, as `NIiter`
/// prints at most six "singular matrix" warnings unless `ngdebug` is set
/// (its static `msgcount`).
const MAX_SOLVER_WARNINGS: usize = 6;

/// Prints a solver warning on standard error (ngspice's
/// `IFerrorf(ERR_WARNING, ...)`), each distinct message once and at most
/// [`MAX_SOLVER_WARNINGS`] per process. Warnings never change a result.
pub(crate) fn solver_warning(message: String) {
    static PRINTED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
    let Ok(mut printed) = PRINTED.lock() else {
        return;
    };
    if printed.len() >= MAX_SOLVER_WARNINGS || printed.contains(&message) {
        return;
    }
    eprintln!("warning: {message}");
    printed.push(message);
}

/// Reject unknowns omitted by SPARSE's translation before any factor is
/// published. This does not inspect numeric values or prune circuit nodes:
/// an allocated zero coefficient may become nonzero in another analysis.
pub(in crate::maths) fn require_translated<T: Scalar>(
    context: &str,
    sparse: &Sparse<T>,
    names: Option<&UnknownNames>,
) -> SpiceResult<()> {
    if sparse.fully_translated() {
        return Ok(());
    }
    let labels: Vec<_> = sparse
        .untranslated()
        .map(|ext| unknown_label(names, ext))
        .collect();
    let unknowns = if labels.len() == 1 {
        format!("node {}", labels[0])
    } else {
        format!("nodes {}", labels.join(", "))
    };
    Err(numerical(
        context,
        format!("singular matrix:  check {unknowns}; no matrix entries constrain these unknowns"),
    ))
}

fn rhs_valid(rhs: &Vector, n: usize) -> SpiceResult<()> {
    if rhs.len() != n || !rhs.is_finite() {
        return Err(numerical(
            "LU solve",
            "RHS must have the system dimension and finite values",
        ));
    }
    Ok(())
}

/// Maps a SPARSE status to the port's error, naming the singular unknown as
/// `NIiter`'s warning does (`SMPgetError` gives the column first, then the
/// row: `singular matrix:  check nodes <col> and <row>`, or `check node
/// <name>` when both name one unknown). `spSMALL_PIVOT`, which ngspice maps
/// to `OK` without a message, is accepted with a [`solver_warning`].
pub(in crate::maths) fn check_status<T: Scalar>(
    context: &str,
    status: Status,
    sparse: &Sparse<T>,
    names: Option<&UnknownNames>,
) -> SpiceResult<()> {
    match status {
        Status::Okay => Ok(()),
        Status::Resource => Err(numerical(
            context,
            "sparse factor storage capacity exceeded",
        )),
        Status::NeedsReload => Err(numerical(
            context,
            "matrix coefficients must be reloaded before factorization",
        )),
        Status::InvalidPlan => Err(numerical(
            context,
            "ordered sparse topology lacks a valid traversal plan",
        )),
        Status::NonFinite => Err(numerical(
            context,
            "non-finite or unrepresentable LU factor",
        )),
        Status::SmallPivot => {
            if let Some((row, col)) = sparse.small_pivot() {
                let (col, row) = (unknown_label(names, col), unknown_label(names, row));
                let at = if row == col {
                    format!("node {row}")
                } else {
                    format!("nodes {col} and {row}")
                };
                solver_warning(format!(
                    "small pivot in the {context}: no pivot passed the pivtol/pivrel \
                     thresholds, so the largest remaining element (at {at}) was used; \
                     the result may be inaccurate (ngspice accepts this silently)"
                ));
            }
            Ok(())
        }
        Status::ZeroDiag | Status::Singular => {
            let (row, col) = sparse.where_singular();
            let (col, row) = (unknown_label(names, col), unknown_label(names, row));
            let unknowns = if row == col {
                format!("node {row}")
            } else {
                format!("nodes {col} and {row}")
            };
            Err(numerical(
                context,
                format!("singular matrix:  check {unknowns}"),
            ))
        }
    }
}

/// Orders and factors a freshly built matrix (`SMPpreOrder` + `SMPreorder`).
pub(in crate::maths) fn order_fresh<T: Scalar>(
    context: &str,
    sparse: &mut Sparse<T>,
    pivot: PivotTolerances,
    names: Option<&UnknownNames>,
) -> SpiceResult<()> {
    require_translated(context, sparse, names)?;
    sparse.mna_preorder();
    let status = sparse.order_and_factor(pivot.relative, pivot.absolute);
    check_status(context, status, sparse, names)
}

/// What a [`SparseSymbolic`] was built from: a folded triplet pattern, or a
/// circuit [`Pattern`] (identified by its `Arc`).
#[derive(Debug, Clone)]
enum Key {
    Triplets(Arc<Vec<(usize, usize)>>),
    /// Weak, because the pattern caches this symbolic (no reference cycle).
    Pattern(std::sync::Weak<Pattern>),
}

/// Reusable pivot order and fill-in structure for one exact assembled
/// pattern (SPARSE's ordered matrix).
#[derive(Debug, Clone)]
pub struct SparseSymbolic {
    n: usize,
    key: Key,
    sparse: Arc<Sparse<f64>>,
    /// What `sparse`'s factors were computed from, when they came from a
    /// clean `NIiter` refactorization ([`solve_ni`]); `None` after anything
    /// else wrote the frame (an AC factorization, a fresh order, a zero or
    /// small pivot), which forces the next step to factor.
    factored: Option<Arc<FactoredInputs>>,
    /// Original coefficient contributions of the actual retained numeric
    /// load. Even identical rounded inputs may have different contributions.
    original: Option<RetainedOriginal>,
}

/// The circuit cache retains the scalar type actually factored. Repeated
/// AC sweeps therefore reuse the complex frame rather than round-tripping
/// through zeroed real storage at every frequency. Real factor snapshots
/// and complex factor snapshots are independently owned.
#[derive(Debug, Clone)]
pub(super) struct ComplexFactors {
    pub(super) sparse: Arc<Sparse<Complex>>,
    pub(super) work: FactorWork,
}

#[derive(Debug)]
pub(super) enum CachedFrame {
    Real(SparseSymbolic),
    Complex(ComplexFactors),
}

impl CachedFrame {
    fn into_real(self, pattern: &Arc<Pattern>, work: &mut PatternWork) -> SparseSymbolic {
        match self {
            Self::Real(symbolic) => symbolic,
            Self::Complex(ComplexFactors { sparse, .. }) => {
                work.complex_to_real = work.complex_to_real.saturating_add(1);
                SparseSymbolic {
                    n: pattern.order(),
                    key: Key::Pattern(Arc::downgrade(pattern)),
                    sparse: Arc::new(Sparse::retype_shared(sparse)),
                    factored: None,
                    original: None,
                }
            }
        }
    }

    pub(super) fn storage(&self) -> Option<FactorStorage> {
        match self {
            Self::Real(symbolic) => symbolic
                .sparse
                .is_factored()
                .then(|| symbolic.sparse.storage()),
            Self::Complex(factors) => factors
                .sparse
                .is_factored()
                .then(|| factors.sparse.storage()),
        }
    }
}

/// The inputs of one clean `spFactor` in [`solve_ni`]: the loaded element
/// values (as bits, in handle order), `LoadGmin`'s `diag_gmin` and the pivot
/// thresholds. Refactoring identical inputs with the same pivot order
/// reproduces the same factors bit for bit, so [`solve_ni`] reuses them.
#[derive(Debug, PartialEq, Eq)]
struct FactoredInputs {
    values: Vec<u64>,
    diag_gmin: u64,
    pivot: (u64, u64),
}

impl FactoredInputs {
    fn new(values: &[f64], diag_gmin: f64, pivot: PivotTolerances) -> Self {
        Self {
            values: values.iter().map(|v| v.to_bits()).collect(),
            diag_gmin: diag_gmin.to_bits(),
            pivot: (pivot.relative.to_bits(), pivot.absolute.to_bits()),
        }
    }

    fn matches(&self, values: &[f64], diag_gmin: f64, pivot: PivotTolerances) -> bool {
        self.diag_gmin == diag_gmin.to_bits()
            && self.pivot == (pivot.relative.to_bits(), pivot.absolute.to_bits())
            && self.values.len() == values.len()
            && self
                .values
                .iter()
                .zip(values)
                .all(|(a, b)| *a == b.to_bits())
    }
}

fn solve_real(
    sparse: &Sparse<f64>,
    n: usize,
    rhs: &Vector,
    workspace: &mut Vec<f64>,
    transposed: bool,
) -> SpiceResult<Vector> {
    rhs_valid(rhs, n)?;
    let factors = sparse.factors()?;
    let x = Vector::from(if transposed {
        factors.solve_transposed(rhs.as_slice(), workspace)?
    } else {
        factors.solve(rhs.as_slice(), workspace)?
    });
    if !x.is_finite() {
        return Err(numerical(
            "LU solve",
            "non-finite solution (singular system)",
        ));
    }
    Ok(x)
}

impl SparseSymbolic {
    fn solve(&self, rhs: &Vector) -> SpiceResult<Vector> {
        self.solve_with_workspace(rhs, &mut Vec::new())
    }

    fn solve_with_workspace(&self, rhs: &Vector, workspace: &mut Vec<f64>) -> SpiceResult<Vector> {
        solve_real(&self.sparse, self.n, rhs, workspace, false)
    }

    fn solve_transposed_with_workspace(
        &self,
        rhs: &Vector,
        workspace: &mut Vec<f64>,
    ) -> SpiceResult<Vector> {
        solve_real(&self.sparse, self.n, rhs, workspace, true)
    }

    /// Whether a matrix has exactly this assembled pattern.
    pub fn matches(&self, matrix: &SparseMatrix) -> bool {
        if matrix.rows() != self.n || matrix.cols() != self.n {
            return false;
        }
        match (&self.key, matrix.bound_values()) {
            (Key::Pattern(pattern), Some((bound, _))) => {
                std::ptr::eq(pattern.as_ptr(), Arc::as_ptr(bound))
            }
            (Key::Pattern(_), None) | (Key::Triplets(_), Some(_)) => false,
            (Key::Triplets(pattern), None) => {
                let mut folded = std::borrow::Cow::Borrowed(matrix);
                if !matrix.is_folded() {
                    folded.to_mut().fold_duplicates();
                }
                folded
                    .triplets()
                    .iter()
                    .map(|t| (t.row, t.col))
                    .eq(pattern.iter().copied())
            }
        }
    }
}

/// Solve a sparse matrix, reusing a compatible pivot ordering.
///
/// Solves `matrix x = rhs`, refactoring with `ordering`'s pivots when the
/// pattern is unchanged (`SMPluFac`) and ordering afresh otherwise
/// (`SMPreorder`), then keeps the new factors' order for the next call. The
/// previous factors are refactored in place when nothing else holds them.
///
/// # Errors
/// As [`SparseLu::new`] and [`SparseLu::solve`].
pub fn solve_with_ordering(
    matrix: &SparseMatrix,
    rhs: &Vector,
    ordering: &mut Option<SparseSymbolic>,
) -> SpiceResult<Vector> {
    if let Some((pattern, _)) = matrix.bound_values() {
        // A bound matrix uses its pattern's single persistent order instead.
        return pattern.with_ordering(|state| {
            let mut cached = state
                .frame
                .take()
                .map(|frame| frame.into_real(pattern, &mut state.work));
            let x = solve_reusing(
                matrix,
                rhs,
                &mut cached,
                &mut state.solve_workspace,
                &mut state.work,
            );
            state.frame = cached.map(CachedFrame::Real);
            x
        });
    }
    solve_reusing(
        matrix,
        rhs,
        ordering,
        &mut Vec::new(),
        &mut PatternWork::default(),
    )
}

fn solve_reusing(
    matrix: &SparseMatrix,
    rhs: &Vector,
    ordering: &mut Option<SparseSymbolic>,
    workspace: &mut Vec<f64>,
    work: &mut PatternWork,
) -> SpiceResult<Vector> {
    let previous = ordering.take();
    let matrix = work.factor.assembly(prepare_matrix(matrix))?;
    let factor = match previous {
        Some(previous) if previous.matches(&matrix) => {
            SparseLu::refactor(previous, &matrix, workspace, &mut work.factor)?
        }
        _ => SparseLu::factor_prepared(&matrix, &mut work.factor)?,
    };
    let result = factor.symbolic.solve_with_workspace(rhs, workspace);
    work.record_rhs(&result);
    // An invalid RHS does not invalidate the matrix's newly computed LU.
    *ordering = Some(factor.symbolic);
    result
}

/// `NIacIter`'s factorization (`niaciter.c`) of a complex system loaded into
/// a circuit pattern's elements (`CKTacLoad`), in the same SPARSE frame as
/// the circuit's real solves: the frame is switched to complex
/// ([`Sparse::into_retyped`]), `SMPcReorder` runs when the pattern's
/// `NIACSHOULDREORDER` flag is set (once per circuit), `SMPcLUfac` otherwise,
/// and a zero pivot sets the flag and reorders. The frame, with whatever
/// reordering happened, is stored back for the circuit's next solve.
///
/// `values` holds one value per pattern element, in element order; `names`
/// label the unknowns in diagnostics.
///
/// # Errors
/// A singular matrix.
pub(in crate::maths) fn factor_shared_complex(
    pattern: &Arc<Pattern>,
    values: &[Complex],
    pivot: PivotTolerances,
    names: Option<&UnknownNames>,
) -> SpiceResult<ComplexFactors> {
    pattern.with_ordering(|state| {
        let (mut sparse, fresh) = match state.frame.take() {
            Some(CachedFrame::Real(symbolic))
                if matches!(&symbolic.key, Key::Pattern(weak) if std::ptr::eq(weak.as_ptr(), Arc::as_ptr(pattern))) =>
            {
                state.work.real_to_complex = state.work.real_to_complex.saturating_add(1);
                (Arc::new(Sparse::retype_shared_loaded(symbolic.sparse, values.iter().copied())), false)
            }
            Some(CachedFrame::Complex(ComplexFactors { mut sparse, .. })) => {
                Sparse::reload_shared(&mut sparse, values.iter().copied());
                (sparse, false)
            }
            _ => {
                let sparse = state.work.factor.assembly(Sparse::from_pattern(pattern, values))?;
                (Arc::new(sparse), true)
            }
        };
        let numeric = Arc::make_mut(&mut sparse);
        // A failed fresh factor must never be cached as a reusable frame.
        require_translated("complex sparse LU", numeric, names)?;
        let mut work = FactorWork::default();
        let result = if fresh {
            state.ac_reorder = false;
            let result = order_fresh("complex sparse LU", numeric, pivot, names);
            work += numeric.last_attempt();
            result
        } else {
            let mut status = if state.ac_reorder {
                state.ac_reorder = false;
                numeric.order_and_factor(pivot.relative, pivot.absolute)
            } else {
                pattern.with_complex_workspace(|workspace| numeric.factor_with_workspace(workspace))
            };
            work += numeric.last_attempt();
            if status == Status::ZeroDiag {
                // NIacIter: E_SINGULAR sets NIACSHOULDREORDER and retries.
                work.zero_pivot_retries += 1;
                numeric.reload_originals(values);
                status = numeric.order_and_factor(pivot.relative, pivot.absolute);
                work += numeric.last_attempt();
            }
            check_status("complex sparse LU", status, numeric, names)
        };
        state.work.factor += work;
        let factors = ComplexFactors { sparse, work };
        state.frame = Some(CachedFrame::Complex(factors.clone()));
        result.map(|()| factors)
    })
}

/// The result of one `NIiter` factor-and-solve step ([`solve_ni`]).
#[derive(Debug)]
pub enum NiStep {
    /// The new iterate.
    Solved(Vector),
    /// `SMPluFac` met a zero pivot with the existing order (`E_SINGULAR`):
    /// `NIiter` sets `NISHOULDREORDER` and reloads before trying again.
    ZeroPivot,
}

/// A SPARSE frame for `matrix` with elements created in its order (a bound
/// matrix's pattern order, else folded row-major), values loaded, not yet
/// ordered.
struct LoadedFrame {
    sparse: Arc<Sparse<f64>>,
    key: Key,
    previous_inputs: Option<Arc<FactoredInputs>>,
}

impl LoadedFrame {
    /// Recover an owned mutable frame, copying only if another factor
    /// snapshot still owns the same SPARSE storage.
    fn from_symbolic(previous: SparseSymbolic) -> Self {
        Self {
            sparse: previous.sparse,
            key: previous.key,
            previous_inputs: previous.factored,
        }
    }

    /// Reload coefficients in handle order, keeping fill-ins and pivot order.
    fn reload(&mut self, matrix: &SparseMatrix) {
        if let Some((_, values)) = matrix.bound_values() {
            Sparse::reload_shared(&mut self.sparse, values.iter().copied());
        } else {
            Sparse::reload_shared(&mut self.sparse, matrix.triplets().iter().map(|t| t.value));
        }
    }

    fn capture_inputs(
        &mut self,
        values: &[f64],
        diag_gmin: f64,
        pivot: PivotTolerances,
    ) -> Arc<FactoredInputs> {
        if let Some(mut previous) = self.previous_inputs.take()
            && let Some(inputs) = Arc::get_mut(&mut previous)
        {
            inputs.values.clear();
            inputs.values.extend(values.iter().map(|v| v.to_bits()));
            inputs.diag_gmin = diag_gmin.to_bits();
            inputs.pivot = (pivot.relative.to_bits(), pivot.absolute.to_bits());
            return previous;
        }
        Arc::new(FactoredInputs::new(values, diag_gmin, pivot))
    }

    fn into_symbolic(self, n: usize, factored: Option<Arc<FactoredInputs>>) -> SparseSymbolic {
        SparseSymbolic {
            n,
            key: self.key,
            sparse: self.sparse,
            factored,
            original: None,
        }
    }
}

/// Build the initial frame from a validated, assembled matrix.
fn fresh_frame(matrix: &SparseMatrix) -> SpiceResult<LoadedFrame> {
    if let Some((pattern, values)) = matrix.bound_values() {
        let sparse = Sparse::from_pattern(pattern, values)?;
        return Ok(LoadedFrame {
            sparse: Arc::new(sparse),
            key: Key::Pattern(Arc::downgrade(pattern)),
            previous_inputs: None,
        });
    }
    let n = matrix.rows();
    let pattern: Vec<_> = matrix.triplets().iter().map(|t| (t.row, t.col)).collect();
    let sparse = Sparse::from_originals(
        n,
        matrix
            .triplets()
            .iter()
            .map(|t| (t.row + 1, t.col + 1, t.value)),
    )?;
    Ok(LoadedFrame {
        sparse: Arc::new(sparse),
        key: Key::Triplets(Arc::new(pattern)),
        previous_inputs: None,
    })
}

/// One `NIiter` step on `matrix`: the factorization ngspice performs after a
/// load, then `SMPsolve`.
///
/// A matrix never ordered (no matching `ordering`; a bound matrix uses its
/// pattern's persistent frame instead) is MNA-preordered once
/// (`SMPpreOrder`). Then `LoadGmin(diag_gmin)` and, with `reorder`
/// (`NISHOULDREORDER`), `spOrderAndFactor`, which keeps every existing pivot
/// that still passes the thresholds and reorders from the first that does
/// not; without it `spFactor` with the existing order, which reports a zero
/// pivot as [`NiStep::ZeroPivot`] for the caller to reload and reorder.
///
/// # Errors
/// Invalid assembly or RHS, a singular reordered matrix, or a non-finite
/// solution.
pub fn solve_ni(
    matrix: &SparseMatrix,
    rhs: &Vector,
    ordering: &mut Option<SparseSymbolic>,
    reorder: bool,
    diag_gmin: f64,
) -> SpiceResult<NiStep> {
    solve_ni_impl(matrix, rhs, ordering, reorder, diag_gmin, None)
}

/// Factor and solve a load that recorded its unfolded physical operator.
/// Transfer the ledger within the same numeric frame transaction, without
/// re-stamping or retaining coefficients on the ordinary path.
/// # Errors
/// Missing original contributions, invalid assembly or failed solve.
pub(crate) fn solve_ni_recorded(
    matrix: &mut SparseMatrix,
    rhs: &Vector,
    ordering: &mut Option<SparseSymbolic>,
    reorder: bool,
) -> SpiceResult<NiStep> {
    let mut original = matrix.take_original().ok_or_else(|| {
        numerical(
            "retained sparse LU",
            "missing original matrix contributions",
        )
    })?;
    let result = solve_ni_impl(matrix, rhs, ordering, reorder, 0., Some(&mut original));
    matrix.recycle_original(original);
    result
}

fn solve_ni_impl(
    matrix: &SparseMatrix,
    rhs: &Vector,
    ordering: &mut Option<SparseSymbolic>,
    reorder: bool,
    diag_gmin: f64,
    original: Option<&mut Vec<crate::maths::Triplet>>,
) -> SpiceResult<NiStep> {
    if let Some((pattern, _)) = matrix.bound_values() {
        return pattern.with_ordering(|state| {
            let mut cached = state
                .frame
                .take()
                .map(|frame| frame.into_real(pattern, &mut state.work));
            let previous = cached.as_mut().and_then(|frame| frame.original.take());
            let step = ni_step(
                matrix,
                rhs,
                &mut cached,
                reorder,
                diag_gmin,
                &mut state.solve_workspace,
                &mut state.work,
            );
            if matches!(&step, Ok(NiStep::Solved(_)))
                && let Some(original) = original
                && let Some(frame) = &mut cached
            {
                frame.original = Some(RetainedOriginal::new(
                    std::mem::take(original),
                    matrix.pivot_tolerances(),
                ));
                *original = previous
                    .and_then(|original| Arc::try_unwrap(original.stamps).ok())
                    .unwrap_or_default();
            }
            state.frame = cached.map(CachedFrame::Real);
            step
        });
    }
    ni_step(
        matrix,
        rhs,
        ordering,
        reorder,
        diag_gmin,
        &mut Vec::new(),
        &mut PatternWork::default(),
    )
}

fn ni_step(
    matrix: &SparseMatrix,
    rhs: &Vector,
    ordering: &mut Option<SparseSymbolic>,
    reorder: bool,
    diag_gmin: f64,
    workspace: &mut Vec<f64>,
    work: &mut PatternWork,
) -> SpiceResult<NiStep> {
    let pivot = matrix.pivot_tolerances();
    let previous = ordering.take();
    // spFactor of the very inputs the frame's factors came from, with the
    // same pivot order, recomputes them bit for bit: solve with them as they
    // are (a linear circuit's later Newton iterations and sweep points, a
    // fixed-step linear transient).
    let reuse = !reorder
        && matrix.bound_values().is_some_and(|(_, values)| {
            previous.as_ref().is_some_and(|previous| {
                previous.matches(matrix)
                    && previous
                        .factored
                        .as_ref()
                        .is_some_and(|inputs| inputs.matches(values, diag_gmin, pivot))
            })
        });
    if reuse && let Some(previous) = previous {
        work.exact_input_hits = work.exact_input_hits.saturating_add(1);
        let result = previous
            .solve_with_workspace(rhs, workspace)
            .map(NiStep::Solved);
        work.record_rhs(&result);
        *ordering = Some(previous);
        return result;
    }
    // The bit-identical fast path above needs no additional coefficient
    // scan. All remaining paths validate and assemble exactly once.
    let matrix = work.factor.assembly(prepare_matrix(matrix))?;
    let previous = previous.filter(|previous| previous.matches(&matrix));
    let (mut frame, fresh) = if let Some(previous) = previous {
        let mut frame = LoadedFrame::from_symbolic(previous);
        frame.reload(&matrix);
        (frame, false)
    } else {
        let mut frame = work.factor.assembly(fresh_frame(&matrix))?;
        require_translated("sparse LU", &frame.sparse, matrix.unknown_names())?;
        Arc::make_mut(&mut frame.sparse).mna_preorder();
        (frame, true)
    };
    let sparse = Arc::make_mut(&mut frame.sparse);
    sparse.load_gmin(diag_gmin);
    let clean_refactor = !fresh && !reorder && sparse.can_refactor();
    let status = if fresh || reorder {
        sparse.order_and_factor(pivot.relative, pivot.absolute)
    } else {
        sparse.factor_with_workspace(workspace)
    };
    work.factor += sparse.last_attempt();
    // spZERO_DIAG and spSINGULAR are both E_SINGULAR to NIiter.
    let zero_pivot = !fresh && !reorder && matches!(status, Status::ZeroDiag | Status::Singular);
    let checked = if zero_pivot {
        Ok(())
    } else {
        check_status("sparse LU", status, sparse, matrix.unknown_names())
    };
    // Only a clean refactorization of a bound matrix is reusable: a fresh
    // order or a reorder may change the pivots, and a small or zero pivot
    // keeps its diagnostics.
    let inputs = match matrix.bound_values() {
        Some((_, values)) if clean_refactor && status == Status::Okay => {
            Some(frame.capture_inputs(values, diag_gmin, pivot))
        }
        _ => None,
    };
    let factored = frame.into_symbolic(matrix.rows(), inputs);
    let result = checked.and_then(|()| {
        if zero_pivot {
            return Ok(NiStep::ZeroPivot);
        }
        let result = factored.solve_with_workspace(rhs, workspace);
        work.record_rhs(&result);
        result.map(NiStep::Solved)
    });
    *ordering = Some(factored);
    result
}

/// Solve with a circuit's retained sparse matrix factors.
///
/// `SMPsolve` with the factors a circuit pattern's frame holds now: those of
/// the last factorization of the circuit matrix (after `CKTop`, the last
/// `NIiter` pass's load), as `.tf` (`tfanal.c`) solves its unit excitations.
/// `None` when the frame holds no factors.
///
/// # Errors
/// An invalid RHS or a non-finite solution.
pub fn solve_last_factors(pattern: &Arc<Pattern>, rhs: &Vector) -> SpiceResult<Option<Vector>> {
    solve_last_factors_direction(pattern, rhs, false)
}

/// Solve `A^T λ = g` from the circuit's retained primal factorization.
///
/// This is an ordinary *transpose*, not an inverse of the transposed input
/// stamps and not a separately factorized Jacobian. Objective derivatives
/// use one such solve for all parameter residuals at the same bias.
///
/// # Errors
/// Invalid RHS or non-finite solution. Returns `None` if the circuit's
/// previously factored matrix has been invalidated.
pub(crate) fn solve_last_factors_transposed(
    pattern: &Arc<Pattern>,
    rhs: &Vector,
) -> SpiceResult<Option<Vector>> {
    solve_last_factors_direction(pattern, rhs, true)
}

fn solve_last_factors_direction(
    pattern: &Arc<Pattern>,
    rhs: &Vector,
    transpose: bool,
) -> SpiceResult<Option<Vector>> {
    pattern.with_ordering(|state| {
        match &mut state.frame {
            Some(CachedFrame::Real(symbolic))
                if matches!(&symbolic.key, Key::Pattern(weak) if std::ptr::eq(weak.as_ptr(), Arc::as_ptr(pattern)))
                    && symbolic.sparse.is_factored() =>
            {
                Some(solve_retained_rhs(
                    symbolic, rhs, transpose, &mut state.solve_workspace, &mut state.work,
                ))
            }
            _ => None,
        }.transpose()
    })
}

/// Measure the largest relative correction of any represented unknown.
/// Unlike a normwise residual this does not hide a small branch current
/// behind a much larger voltage or conductance.
fn update_coordinates(next: &mut Vector, current: &Vector) -> f64 {
    let mut relative = 0.0_f64;
    for (delta, &value) in next.as_mut_slice().iter_mut().zip(current.as_slice()) {
        let change = *delta;
        *delta += value;
        let scale = value.abs().max(delta.abs());
        if scale != 0. {
            relative = relative.max(change.abs() / scale);
        }
    }
    relative
}

// A poorly conditioned recovered operator can contract much more slowly than
// an equilibrated ordinary solve. Bound the total work, and stop early if
// consecutive corrections cannot improve the measured solution change.
const MAX_RECOVERED_RHS_CORRECTIONS: usize = 64;
const RECOVERED_RHS_STALL_STEPS: usize = 8;

/// Use the existing factors as a preconditioner for the unfolded physical
/// operator, in either orientation. The original coefficient contributions
/// can contain terms lost by folding; refinement of only the folded matrix
/// would miss those terms. A stalled, exhausted or non-finite correction
/// requests precision escalation instead of returning an unreliable iterate.
fn refine_retained_rhs(
    order: usize,
    stamps: &[crate::maths::Triplet],
    rhs: &Vector,
    initial: Vector,
    transpose: bool,
    mut solve: impl FnMut(&Vector) -> SpiceResult<Vector>,
) -> SpiceResult<Option<Vector>> {
    use crate::maths::equilibration::OriginalResidual;

    let entries = recovery::operator_entries(stamps)?;
    let original = OriginalResidual::real(order, &entries, transpose)?;
    let mut current = initial;
    let mut best_change = f64::INFINITY;
    let mut residual = Vector::zeros(order);
    let mut stalled = 0;
    for _ in 0..MAX_RECOVERED_RHS_CORRECTIONS {
        original.correction_into(rhs.as_slice(), current.as_slice(), residual.as_mut_slice())?;
        let mut next = solve(&residual)?;
        let relative = update_coordinates(&mut next, &current);
        if !next.is_finite() || !relative.is_finite() {
            return Ok(None);
        }
        if relative <= f64::EPSILON.sqrt() {
            return Ok(Some(next));
        }
        if relative < best_change {
            best_change = relative;
            stalled = 0;
        } else {
            stalled += 1;
            if stalled == RECOVERED_RHS_STALL_STEPS {
                break;
            }
        }
        current = next;
    }
    Ok(None)
}

fn solve_retained_rhs(
    symbolic: &mut SparseSymbolic,
    rhs: &Vector,
    transpose: bool,
    workspace: &mut Vec<f64>,
    work: &mut PatternWork,
) -> SpiceResult<Vector> {
    if rhs.len() != symbolic.n || !rhs.is_finite() {
        return Err(numerical("retained sparse solve", "invalid RHS"));
    }
    if let Some(original) = &mut symbolic.original
        && original.promoted()
    {
        return original.solve_wide(symbolic.n, rhs, transpose, work);
    }
    let mut solve = |rhs: &Vector| {
        let result = if transpose {
            symbolic.solve_transposed_with_workspace(rhs, workspace)
        } else {
            symbolic.solve_with_workspace(rhs, workspace)
        };
        work.record_rhs(&result);
        result
    };
    if let Some(original) = &symbolic.original {
        if let Ok(Some(solution)) = solve(rhs).and_then(|initial| {
            refine_retained_rhs(symbolic.n, &original.stamps, rhs, initial, transpose, solve)
        }) {
            return Ok(solution);
        }
    } else {
        return solve(rhs);
    }
    symbolic
        .original
        .as_mut()
        .ok_or_else(|| numerical("retained sparse recovery", "missing unfolded operator"))?
        .solve_wide(symbolic.n, rhs, transpose, work)
}

/// Owned numeric sparse LU factors.
#[derive(Debug)]
pub struct SparseLu {
    symbolic: SparseSymbolic,
    work: FactorWork,
}

pub(super) fn assembled_triplets<'a>(
    matrix: &'a SparseMatrix,
    context: &str,
) -> SpiceResult<Cow<'a, SparseMatrix>> {
    square(matrix.rows(), matrix.cols())?;
    if matrix.triplets().iter().any(|t| !t.value.is_finite()) {
        return Err(numerical(context, "non-finite coefficient"));
    }
    if matrix.is_folded() {
        return Ok(Cow::Borrowed(matrix));
    }
    let mut matrix = matrix.clone();
    matrix.fold_duplicates();
    if matrix.triplets().iter().any(|t| !t.value.is_finite()) {
        return Err(numerical(context, "duplicate summation overflow"));
    }
    Ok(Cow::Owned(matrix))
}

fn finite_bound(values: &[f64]) -> SpiceResult<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(numerical("sparse LU assembly", "non-finite coefficient"));
    }
    Ok(())
}

/// Prepare once for symbolic matching, fresh ordering and numeric refactoring.
fn prepare_matrix(matrix: &SparseMatrix) -> SpiceResult<Cow<'_, SparseMatrix>> {
    if let Some((_, values)) = matrix.bound_values() {
        square(matrix.rows(), matrix.cols())?;
        finite_bound(values)?;
        Ok(Cow::Borrowed(matrix))
    } else {
        assembled_triplets(matrix, "sparse LU assembly")
    }
}

impl SparseLu {
    /// Factors a snapshot, optionally refactoring with the pivot order of an
    /// identical earlier pattern.
    ///
    /// # Errors
    /// Invalid dimensions, coefficients, changed pattern, or a singular matrix.
    pub fn new(matrix: &SparseMatrix, reuse: Option<&SparseSymbolic>) -> SpiceResult<Self> {
        let matrix = prepare_matrix(matrix)?;
        if let Some(previous) = reuse {
            if !previous.matches(&matrix) {
                return Err(numerical(
                    "sparse symbolic LU",
                    "assembled CSC pattern changed",
                ));
            }
            return Self::refactor(
                previous.clone(),
                &matrix,
                &mut Vec::new(),
                &mut FactorWork::default(),
            );
        }
        Self::factor_prepared(&matrix, &mut FactorWork::default())
    }

    fn factor_prepared(matrix: &SparseMatrix, total: &mut FactorWork) -> SpiceResult<Self> {
        let mut frame = total.assembly(fresh_frame(matrix))?;
        let numeric = Arc::make_mut(&mut frame.sparse);
        let result = order_fresh(
            "sparse LU",
            numeric,
            matrix.pivot_tolerances(),
            matrix.unknown_names(),
        );
        let work = numeric.last_attempt();
        *total += work;
        result?;
        Ok(Self {
            symbolic: frame.into_symbolic(matrix.rows(), None),
            work,
        })
    }

    /// `SMPluFac` with `previous`'s pivots on a matching matrix, reordering
    /// after a zero pivot as `NIiter` does after `E_SINGULAR`.
    fn refactor(
        previous: SparseSymbolic,
        matrix: &SparseMatrix,
        workspace: &mut Vec<f64>,
        total: &mut FactorWork,
    ) -> SpiceResult<Self> {
        let mut frame = LoadedFrame::from_symbolic(previous);
        frame.reload(matrix);
        let numeric = Arc::make_mut(&mut frame.sparse);
        let mut status = numeric.factor_with_workspace(workspace);
        let mut work = numeric.last_attempt();
        if status == Status::ZeroDiag {
            work.zero_pivot_retries += 1;
            frame.reload(matrix);
            let pivot = matrix.pivot_tolerances();
            let numeric = Arc::make_mut(&mut frame.sparse);
            status = numeric.order_and_factor(pivot.relative, pivot.absolute);
            work += numeric.last_attempt();
        }
        *total += work;
        check_status("sparse LU", status, &frame.sparse, matrix.unknown_names())?;
        Ok(Self {
            symbolic: frame.into_symbolic(matrix.rows(), None),
            work,
        })
    }

    /// Actual structural and numeric storage in this immutable factor snapshot.
    #[must_use = "the returned storage counts quantify the immutable real factor snapshot"]
    pub fn factor_storage(&self) -> FactorStorage {
        self.symbolic.sparse.storage()
    }

    /// Actual factor work that produced this snapshot, including any immediate retry.
    #[must_use = "the returned work includes the factorization and any immediate retry"]
    pub const fn factor_work(&self) -> FactorWork {
        self.work
    }

    /// Pivot order for subsequent identical-pattern assemblies.
    #[must_use = "retaining the pivot order enables reuse for subsequent identical-pattern assemblies"]
    pub const fn symbolic(&self) -> &SparseSymbolic {
        &self.symbolic
    }

    /// Solves with this snapshot; later mutations to storage cannot stale it.
    ///
    /// # Errors
    /// Invalid RHS or non-finite solution.
    pub fn solve(&self, rhs: &Vector) -> SpiceResult<Vector> {
        self.symbolic.solve(rhs)
    }

    /// Solves with this immutable snapshot and caller-owned numeric scratch.
    /// # Errors
    /// Invalid RHS or non-finite solution.
    pub fn solve_with_workspace(
        &self,
        rhs: &Vector,
        workspace: &mut Vec<f64>,
    ) -> SpiceResult<Vector> {
        self.symbolic.solve_with_workspace(rhs, workspace)
    }

    /// Solves the transposed system `A^T x = b` with the same factors.
    /// # Errors
    /// Invalid RHS or non-finite solution.
    pub fn solve_transposed(&self, rhs: &Vector) -> SpiceResult<Vector> {
        self.solve_transposed_with_workspace(rhs, &mut Vec::new())
    }

    /// Transposed solve with caller-owned scratch.
    /// # Errors
    /// Invalid RHS or non-finite solution.
    pub fn solve_transposed_with_workspace(
        &self,
        rhs: &Vector,
        workspace: &mut Vec<f64>,
    ) -> SpiceResult<Vector> {
        solve_real(&self.symbolic.sparse, self.symbolic.n, rhs, workspace, true)
    }

    /// Solve consecutive RHS-major vectors using one factor traversal per tile.
    ///
    /// The output retains RHS order and external matrix ordering. A transpose
    /// solve uses the same factors. Scratch and output capacities are reusable.
    ///
    /// # Errors
    /// Invalid RHS length, non-finite input or solution, or insufficient storage.
    pub fn solve_batch_into(
        &self,
        rhs: &[f64],
        workspace: &mut SolveBatchWorkspace<f64>,
        out: &mut Vec<f64>,
        transposed: bool,
    ) -> Result<(), BatchSolveFailure> {
        self.symbolic
            .sparse
            .factors()?
            .solve_batch_into(rhs, workspace, out, transposed)
    }

    /// Binary-scaled determinant: `(mantissa, base-2 exponent)`.
    /// # Errors
    /// Non-finite/invalid inverse factors or an out-of-range exponent.
    pub fn determinant(&self) -> SpiceResult<(f64, i64)> {
        self.symbolic.sparse.factors()?.determinant()
    }
}

/// Dense LU through the same SPARSE factorization (ngspice has no separate
/// dense simulator solve); zero entries are not stored, as with SPARSE.
#[derive(Debug)]
pub struct DenseLu {
    sparse: Sparse<f64>,
    n: usize,
}

impl DenseLu {
    /// Factors a nonempty square finite matrix.
    ///
    /// # Errors
    /// Invalid dimensions/coefficients or a singular matrix.
    pub fn new(matrix: &Matrix) -> SpiceResult<Self> {
        square(matrix.rows(), matrix.cols())?;
        if matrix.data().iter().any(|a| !a.is_finite()) {
            return Err(numerical("dense LU", "non-finite coefficient"));
        }
        let n = matrix.rows();
        // Unique nonzero entries in the same folded row-major creation order.
        let entries = matrix
            .data()
            .chunks_exact(n)
            .enumerate()
            .flat_map(|(row, values)| {
                values
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v != 0.)
                    .map(move |(col, &v)| (row + 1, col + 1, v))
            });
        let mut sparse = Sparse::from_originals(n, entries)?;
        order_fresh("dense LU", &mut sparse, PivotTolerances::DEFAULT, None)?;
        Ok(Self { sparse, n })
    }

    /// Solves against the stored factors.
    ///
    /// # Errors
    /// Invalid RHS or non-finite solution.
    pub fn solve(&self, rhs: &Vector) -> SpiceResult<Vector> {
        self.solve_with_workspace(rhs, &mut Vec::new())
    }

    /// Solves with caller-owned numeric scratch.
    /// # Errors
    /// Invalid RHS or non-finite solution.
    pub fn solve_with_workspace(
        &self,
        rhs: &Vector,
        workspace: &mut Vec<f64>,
    ) -> SpiceResult<Vector> {
        solve_real(&self.sparse, self.n, rhs, workspace, false)
    }

    /// Solves the transposed system with the same factors.
    /// # Errors
    /// Invalid RHS or non-finite solution.
    pub fn solve_transposed(&self, rhs: &Vector) -> SpiceResult<Vector> {
        self.solve_transposed_with_workspace(rhs, &mut Vec::new())
    }

    /// Transposed solve with caller-owned numeric scratch.
    /// # Errors
    /// Invalid RHS or non-finite solution.
    pub fn solve_transposed_with_workspace(
        &self,
        rhs: &Vector,
        workspace: &mut Vec<f64>,
    ) -> SpiceResult<Vector> {
        solve_real(&self.sparse, self.n, rhs, workspace, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maths::complex::ComplexMatrix;

    #[test]
    fn dense_and_sparse_transposed_solves_share_workspace_and_preserve_order() {
        let mut dense = Matrix::zeros(2, 2).unwrap();
        let mut sparse = SparseMatrix::new(2, 2);
        for (r, c, v) in [(0, 0, 2.), (0, 1, 1.), (1, 0, 3.), (1, 1, 4.)] {
            dense.set(r, c, v).unwrap();
            sparse.add(r, c, v).unwrap();
        }
        let dense = DenseLu::new(&dense).unwrap();
        let sparse = SparseLu::new(&sparse, None).unwrap();
        let rhs = Vector::from_slice(&[5., 5.]);
        let mut workspace = Vec::new();
        let x = sparse
            .solve_transposed_with_workspace(&rhs, &mut workspace)
            .unwrap();
        assert!((x.get(0).unwrap() - 1.).abs() < 1e-14);
        assert!((x.get(1).unwrap() - 1.).abs() < 1e-14);
        let storage = workspace.as_ptr();
        assert_eq!(
            dense
                .solve_transposed_with_workspace(&rhs, &mut workspace)
                .unwrap(),
            x
        );
        assert_eq!(sparse.solve_transposed(&rhs).unwrap(), x);
        assert_eq!(dense.solve_transposed(&rhs).unwrap(), x);
        assert_eq!(
            sparse.solve_with_workspace(&rhs, &mut workspace).unwrap(),
            sparse.solve(&rhs).unwrap()
        );
        assert_eq!(workspace.as_ptr(), storage);
        assert!(dense.solve_transposed(&Vector::zeros(1)).is_err());
    }

    #[test]
    fn bound_solves_reuse_scratch_and_keep_valid_factors_on_rhs_failure() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        for (r, c, v) in [(0, 0, 3.), (0, 1, 1.), (1, 0, 1.), (1, 1, 4.)] {
            a.set(r, c, v).unwrap();
        }
        let rhs = Vector::from_slice(&[4., 5.]);
        let first = solve_with_ordering(&a, &rhs, &mut None).unwrap();
        let storage = pattern.with_ordering(|state| state.solve_workspace.as_ptr());
        let second = solve_with_ordering(&a, &rhs, &mut None).unwrap();
        assert_eq!(first, second);
        assert!(solve_with_ordering(&a, &Vector::zeros(1), &mut None).is_err());
        let retained = solve_last_factors(&pattern, &rhs).unwrap().unwrap();
        assert_eq!(retained, first);
        assert_eq!(
            storage,
            pattern.with_ordering(|state| state.solve_workspace.as_ptr())
        );
    }

    #[test]
    fn unique_refactors_keep_frame_and_input_allocations() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        for (r, c, v) in [(0, 0, 3.), (0, 1, 1.), (1, 0, 1.), (1, 1, 4.)] {
            a.set(r, c, v).unwrap();
        }
        let rhs = Vector::from_slice(&[4., 5.]);
        let read = || {
            pattern.with_ordering(|state| {
                let Some(CachedFrame::Real(symbolic)) = &state.frame else {
                    panic!("missing real frame")
                };
                let inputs = symbolic.factored.as_ref().unwrap();
                (
                    Arc::as_ptr(&symbolic.sparse),
                    Arc::as_ptr(inputs),
                    inputs.values.as_ptr(),
                )
            })
        };
        for _ in 0..2 {
            solve_ni(&a, &rhs, &mut None, false, 0.).unwrap();
        }
        let before = read();
        a.set(0, 0, 4.).unwrap();
        solve_ni(&a, &rhs, &mut None, false, 0.).unwrap();
        assert_eq!(before, read());
        let snapshot = pattern.with_ordering(|state| {
            let Some(CachedFrame::Real(symbolic)) = &state.frame else {
                panic!("missing real frame")
            };
            symbolic.clone()
        });
        let x = snapshot.solve(&rhs).unwrap();
        a.set(0, 0, 5.).unwrap();
        solve_ni(&a, &rhs, &mut None, false, 0.).unwrap();
        assert_eq!(snapshot.solve(&rhs).unwrap(), x);
        assert_ne!(before.0, read().0);
        assert_ne!(before.1, read().1);
        assert_ne!(before.2, read().2);
    }

    #[test]
    fn unique_complex_refactors_keep_frame_allocation() {
        let pattern = Arc::new(Pattern::new(1, [(0, 0)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        a.set(0, 0, 3.).unwrap();
        let mut e = SparseMatrix::new(1, 1);
        e.add(0, 0, 0.5).unwrap();
        let factor = |omega| {
            ComplexMatrix::from_operators_on(&pattern, &a, &e, omega)
                .unwrap()
                .unwrap()
                .factorize()
                .unwrap()
        };
        let read = || {
            pattern.with_ordering(|state| {
                let Some(CachedFrame::Complex(factors)) = &state.frame else {
                    panic!("missing complex frame")
                };
                Arc::as_ptr(&factors.sparse)
            })
        };
        drop(factor(1.));
        let initial = read();
        drop(factor(2.));
        assert_eq!(read(), initial);
        let snapshot = factor(3.);
        let x = snapshot.solve(&[Complex::real(1.)]).unwrap();
        let next = factor(4.);
        assert_ne!(read(), initial);
        assert_eq!(snapshot.solve(&[Complex::real(1.)]).unwrap(), x);
        assert_ne!(next.solve(&[Complex::real(1.)]).unwrap(), x);
    }

    #[test]
    fn recorded_operator_refines_both_orientations_and_follows_numeric_frame() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap());
        let mut matrix = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        let large = 0.01;
        let small = 1e-15;
        let changed = small - 2e-19;
        let rhs = Vector::from_slice(&[small, 2. * small]);
        let dual = Vector::from_slice(&[2. * small, 2. * small]);
        let load = |matrix: &mut SparseMatrix, physical: f64, record: bool| {
            matrix.clear();
            matrix.record_original(record);
            for (row, col, value) in [
                (0, 0, large),
                (0, 0, physical),
                (0, 1, -large),
                (1, 0, -2. * large),
                (1, 1, 2. * large),
                (1, 1, 2. * physical),
            ] {
                matrix.add(row, col, value).unwrap();
            }
        };
        let check = |expected: f64| {
            let direct = solve_last_factors(&pattern, &rhs).unwrap().unwrap();
            let transpose = solve_last_factors_transposed(&pattern, &dual)
                .unwrap()
                .unwrap();
            for &value in direct.as_slice() {
                assert!((value - expected).abs() < 1e-10, "{value} != {expected}");
            }
            for (&value, want) in transpose.as_slice().iter().zip([2. * expected, expected]) {
                assert!((value - want).abs() < 1e-10, "{value} != {want}");
            }
        };
        // Establish a clean refactorized frame, not just a fresh pivot order.
        for _ in 0..2 {
            load(&mut matrix, small, true);
            let NiStep::Solved(raw) =
                solve_ni_recorded(&mut matrix, &rhs, &mut None, false).unwrap()
            else {
                panic!("expected a valid factor")
            };
            assert!((raw.as_slice()[0] - 1.).abs() > 1e-6);
        }
        let work = pattern.factor_telemetry().work;
        check(1.);
        assert_eq!(pattern.factor_telemetry().work.factor, work.factor);
        let rounded = matrix.bound_values().unwrap().1.to_vec();
        load(&mut matrix, changed, true);
        assert_eq!(matrix.bound_values().unwrap().1, rounded);
        assert!(matches!(
            solve_ni_recorded(&mut matrix, &rhs, &mut None, false).unwrap(),
            NiStep::Solved(_)
        ));
        assert_eq!(
            pattern.factor_telemetry().work.exact_input_hits,
            work.exact_input_hits + 1
        );
        check(small / changed);
        assert_eq!(pattern.factor_telemetry().work.factor, work.factor);
        // An ordinary load clears even a bit-identical retained operator.
        load(&mut matrix, changed, false);
        let NiStep::Solved(plain) = solve_ni(&matrix, &rhs, &mut None, false, 0.).unwrap() else {
            panic!("expected a valid factor")
        };
        assert_eq!(solve_last_factors(&pattern, &rhs).unwrap().unwrap(), plain);
    }

    #[test]
    fn divergent_retained_refinement_promotes_and_invalidates_both_orientations() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (0, 1), (1, 1)]).unwrap());
        let mut matrix = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        let large = 2.0_f64.powi(54);
        let load = |matrix: &mut SparseMatrix, hidden: f64| {
            matrix.clear();
            matrix.record_original(true);
            for (row, col, value) in [
                (0, 0, large),
                (0, 0, hidden),
                (0, 0, -large),
                (0, 0, 0.25),
                (0, 1, 0.5),
                (1, 1, 2.),
            ] {
                matrix.add(row, col, value).unwrap();
            }
        };
        load(&mut matrix, 1.);
        let rhs = Vector::from_slice(&[4., 6.]);
        let NiStep::Solved(raw) = solve_ni_recorded(&mut matrix, &rhs, &mut None, false).unwrap()
        else {
            panic!("expected valid binary64 factors")
        };
        // Rounded A[0,0] is 0.25, unfolded A[0,0] is 1.25.
        // The refinement iteration has eigenvalue -4 and must escalate.
        assert_eq!(raw.as_slice(), &[10., 3.]);
        let before = pattern.factor_telemetry().work.factor;
        let direct = solve_last_factors(&pattern, &rhs).unwrap().unwrap();
        assert_eq!(direct.as_slice(), &[2., 3.]);
        let promoted = pattern.factor_telemetry().work.factor;
        assert_ne!(before, promoted);
        let transpose_rhs = Vector::from_slice(&[2.5, 7.]);
        let transpose = solve_last_factors_transposed(&pattern, &transpose_rhs)
            .unwrap()
            .unwrap();
        assert_eq!(transpose.as_slice(), &[2., 3.]);
        assert_eq!(pattern.factor_telemetry().work.factor, promoted);
        // Same folded binary64 values, different physical operator.
        let rounded = matrix.bound_values().unwrap().1.to_vec();
        load(&mut matrix, 1.5);
        assert_eq!(matrix.bound_values().unwrap().1, rounded);
        let rhs = Vector::from_slice(&[5., 6.]);
        assert!(matches!(
            solve_ni_recorded(&mut matrix, &rhs, &mut None, false).unwrap(),
            NiStep::Solved(_)
        ));
        assert_eq!(
            solve_last_factors(&pattern, &rhs)
                .unwrap()
                .unwrap()
                .as_slice(),
            &[2., 3.]
        );
        assert_ne!(pattern.factor_telemetry().work.factor, promoted);
    }

    #[test]
    fn retained_sparse_transpose_projects_one_objective_without_refactoring() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap());
        let mut matrix = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        for (r, c, value) in [(0, 0, 3.), (0, 1, 1.), (1, 0, 2.), (1, 1, 4.)] {
            matrix.set(r, c, value).unwrap();
        }
        let source = Vector::from_slice(&[5., 7.]);
        assert!(matches!(
            solve_ni(&matrix, &source, &mut None, false, 0.).unwrap(),
            NiStep::Solved(_)
        ));
        let output = Vector::from_slice(&[1., -1.]);
        let adjoint = solve_last_factors_transposed(&pattern, &output)
            .unwrap()
            .unwrap();
        let expected = [0.6, -0.4];
        for (actual, want) in adjoint.as_slice().iter().zip(expected) {
            assert!((*actual - want).abs() < 1e-14);
        }
        let primal = solve_last_factors(&pattern, &source).unwrap().unwrap();
        let projected = adjoint.as_slice()[0] * source.as_slice()[0]
            + adjoint.as_slice()[1] * source.as_slice()[1];
        let directly = primal.as_slice()[0] - primal.as_slice()[1];
        assert!((projected - directly).abs() < 1e-14);
        matrix.set(0, 0, f64::from_bits(1)).unwrap();
        assert!(solve_ni(&matrix, &source, &mut None, false, 0.).is_err());
        assert!(
            solve_last_factors_transposed(&pattern, &output)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn unrepresentable_bound_factor_never_enters_retained_solves() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        for (r, c, v) in [(0, 0, 2.), (0, 1, 1.), (1, 0, 1.), (1, 1, 2.)] {
            a.set(r, c, v).unwrap();
        }
        let rhs = Vector::from_slice(&[3., 3.]);
        for _ in 0..2 {
            assert!(matches!(
                solve_ni(&a, &rhs, &mut None, false, 0.).unwrap(),
                NiStep::Solved(_)
            ));
        }
        assert!(solve_last_factors(&pattern, &rhs).unwrap().is_some());
        a.set(0, 0, f64::from_bits(1)).unwrap();
        assert!(solve_ni(&a, &rhs, &mut None, false, 0.).is_err());
        assert!(solve_last_factors(&pattern, &rhs).unwrap().is_none());
        a.set(0, 0, 2.).unwrap();
        let NiStep::Solved(x) = solve_ni(&a, &rhs, &mut None, false, 0.).unwrap() else {
            panic!("factor recovery failed")
        };
        assert_eq!(x.as_slice(), &[1., 1.]);
    }

    #[test]
    fn dense_direct_frame_matches_folded_sparse_order_bit_for_bit() {
        let mut dense = Matrix::zeros(4, 4).unwrap();
        let mut sparse = SparseMatrix::new(4, 4);
        for (r, c, value) in [
            (0, 1, 3.),
            (1, 0, -2.),
            (1, 2, 1.),
            (2, 1, 0.5),
            (2, 2, 4.),
            (2, 3, -1.),
            (3, 2, 2.),
            (3, 3, 5.),
        ] {
            dense.set(r, c, value).unwrap();
            sparse.add(r, c, value).unwrap();
        }
        let rhs = Vector::from_slice(&[1.25, -2.5, 3.75, -4.25]);
        let direct = DenseLu::new(&dense).unwrap();
        let previous = SparseLu::new(&sparse, None).unwrap();
        let bits = |x: Vector| x.as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(direct.solve(&rhs).unwrap()),
            bits(previous.solve(&rhs).unwrap())
        );
        assert_eq!(
            direct
                .sparse
                .factors()
                .unwrap()
                .determinant()
                .unwrap()
                .0
                .to_bits(),
            previous.determinant().unwrap().0.to_bits()
        );
        assert_eq!(
            direct.sparse.factors().unwrap().determinant().unwrap().1,
            previous.determinant().unwrap().1
        );
    }

    #[test]
    fn complex_cache_keeps_snapshots_and_changes_scalar_only_at_boundary() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        for (r, c, value) in [(0, 0, 3.), (0, 1, 1.), (1, 0, 2.), (1, 1, 4.)] {
            a.add(r, c, value).unwrap();
        }
        let rhs = Vector::from_slice(&[5., 10.]);
        let NiStep::Solved(real) = solve_ni(&a, &rhs, &mut None, false, 0.).unwrap() else {
            panic!("initial real solve failed");
        };
        let mut e = SparseMatrix::new(2, 2);
        e.add(0, 0, 0.5).unwrap();
        e.add(1, 1, 2.).unwrap();
        let rhs_complex = [Complex::real(5.), Complex::real(10.)];
        let ac = |omega| {
            ComplexMatrix::from_operators_on(&pattern, &a, &e, omega)
                .unwrap()
                .unwrap()
                .factorize()
                .unwrap()
        };
        let first = ac(1.);
        let solution = first.solve(&rhs_complex).unwrap();
        let first_frame = pattern.with_ordering(|state| state.frame.take()).unwrap();
        let CachedFrame::Complex(ComplexFactors { sparse, .. }) = &first_frame else {
            panic!("AC must retain a complex frame");
        };
        assert!(sparse.is_factored());
        assert_eq!(Arc::strong_count(sparse), 2);
        pattern.with_ordering(|state| state.frame = Some(first_frame));
        let second = ac(2.);
        assert_ne!(second.solve(&rhs_complex).unwrap(), solution);
        assert_eq!(first.solve(&rhs_complex).unwrap(), solution);
        drop(second);
        let frame = pattern.with_ordering(|state| state.frame.take()).unwrap();
        let CachedFrame::Complex(ComplexFactors { sparse, .. }) = &frame else {
            panic!("second AC factorization lost the complex frame");
        };
        assert_eq!(Arc::strong_count(sparse), 1);
        pattern.with_ordering(|state| state.frame = Some(frame));
        assert_eq!(solve_last_factors(&pattern, &rhs).unwrap(), None);
        let NiStep::Solved(after) = solve_ni(&a, &rhs, &mut None, false, 0.).unwrap() else {
            panic!("real refactor failed");
        };
        assert_eq!(real, after);
        assert_eq!(first.solve(&rhs_complex).unwrap(), solution);
        assert!(matches!(
            pattern.with_ordering(|state| state.frame.take()),
            Some(CachedFrame::Real(_))
        ));
    }

    #[test]
    fn unfolded_symbolic_reuse_keeps_pattern_and_snapshots() {
        let mut a = SparseMatrix::new(2, 2);
        a.add(1, 1, 3.).unwrap();
        a.add(0, 0, 2.).unwrap();
        a.add(0, 1, 1.).unwrap();
        let rhs = Vector::from_slice(&[6., 6.]);
        let original = SparseLu::new(&a, None).unwrap();
        a.add(0, 0, 2.).unwrap();
        a.add(1, 0, 9.).unwrap();
        a.add(1, 0, -9.).unwrap();
        let reused = SparseLu::new(&a, Some(original.symbolic())).unwrap();
        assert!(reused.symbolic.matches(&a));
        assert_eq!(original.solve(&rhs).unwrap().as_slice(), &[2., 2.]);
        assert_eq!(reused.solve(&rhs).unwrap().as_slice(), &[1., 2.]);

        let mut ordering = Some(original.symbolic().clone());
        assert_eq!(
            solve_with_ordering(&a, &rhs, &mut ordering)
                .unwrap()
                .as_slice(),
            &[1., 2.]
        );
        let NiStep::Solved(x) = solve_ni(&a, &rhs, &mut ordering, false, 0.).unwrap() else {
            panic!("unexpected zero pivot");
        };
        assert_eq!(x.as_slice(), &[1., 2.]);
        a.add(0, 0, f64::MAX).unwrap();
        a.add(0, 0, f64::MAX).unwrap();
        assert!(solve_ni(&a, &rhs, &mut ordering, false, 0.).is_err());
    }

    #[test]
    fn complex_frame_retains_order_but_not_real_factors() {
        let pattern = Arc::new(Pattern::new(2, [(0, 1), (1, 0), (1, 1), (0, 0)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        a.add(0, 1, 2.).unwrap();
        a.add(1, 0, 3.).unwrap();
        a.add(1, 1, 4.).unwrap();
        let rhs = Vector::from_slice(&[4., 11.]);
        let NiStep::Solved(before) = solve_ni(&a, &rhs, &mut None, false, 0.).unwrap() else {
            panic!("first factorization failed");
        };
        assert_eq!(
            solve_last_factors(&pattern, &rhs).unwrap(),
            Some(before.clone())
        );
        let mut e = SparseMatrix::new(2, 2);
        e.add(0, 0, 0.5).unwrap();
        let ac = ComplexMatrix::from_operators_on(&pattern, &a, &e, 2.)
            .unwrap()
            .unwrap()
            .factorize()
            .unwrap();
        assert!(
            ac.solve(&[
                crate::primitives::Complex::real(4.),
                crate::primitives::Complex::real(11.)
            ])
            .is_ok()
        );
        // The retained frame is retyped with zero coefficients. Its order
        // can be reused, but it cannot answer a real solve until reloaded.
        assert_eq!(solve_last_factors(&pattern, &rhs).unwrap(), None);
        let NiStep::Solved(after) = solve_ni(&a, &rhs, &mut None, false, 0.).unwrap() else {
            panic!("refactorization failed");
        };
        assert_eq!(before, after);
        assert_eq!(solve_last_factors(&pattern, &rhs).unwrap(), Some(after));
    }

    #[test]
    fn assembled_view_borrows_folded_storage_and_checks_overflow() {
        let mut a = SparseMatrix::new(1, 1);
        a.add(0, 0, 2.).unwrap();
        assert!(matches!(
            assembled_triplets(&a, "sparse LU assembly").unwrap(),
            Cow::Owned(_)
        ));
        a.fold_duplicates();
        assert!(matches!(
            assembled_triplets(&a, "sparse LU assembly").unwrap(),
            Cow::Borrowed(_)
        ));
        a.add(0, 0, f64::MAX).unwrap();
        a.add(0, 0, f64::MAX).unwrap();
        assert!(assembled_triplets(&a, "sparse LU assembly").is_err());
        a.fold_duplicates();
        assert!(assembled_triplets(&a, "sparse LU assembly").is_err());
    }
}
