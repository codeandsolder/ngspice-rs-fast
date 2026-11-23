//! Complex sparse LU for AC through the SPARSE 1.3 port. Scalar conversion
//! happens only at this boundary.
//!
//! C: `SMPcLUfac`/`SMPcReorder`/`SMPcSolve`/`SMPcaSolve` (`spsmp.c`). As in
//! [`crate::maths::linear`], there is no condition estimate or residual check;
//! singular pivots and non-finite solutions are errors.
use std::sync::Arc;

use crate::maths::linear::{
    BatchSolveFailure, ComplexFactors, PivotTolerances, SolveBatchWorkspace, check_status,
    numerical, order_fresh, square,
};
use crate::maths::sparse13::{Sparse, Status};
use crate::maths::{FactorStorage, FactorWork, Pattern, SparseMatrix};
use crate::primitives::{Complex, SpiceResult};

/// Capacitive contributions grouped at a single matrix element.
///
/// Contributions of `E` (a range of its unfolded triplets, all at one
/// position) that one device adds to an element as `scale * (x1 + x2 + ...)`,
/// each `x` being `cap * CKTomega`: `mos1acld.c` adds `xgd + xgs + xgb`,
/// `bjtacld.c` adds `m * (xcpi + xcmu + xcmcb)`.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicGroup {
    /// The contributions it replaces, as indices into `E`'s triplets (which
    /// hold `scale * term`, zero terms dropped).
    pub range: std::ops::Range<usize>,
    /// The unscaled terms (capacitances), in C's order.
    pub terms: Vec<f64>,
    /// The factor C applies to the sum (the instance multiplier `m`).
    pub scale: f64,
    /// The divisor C applies after scaling (`indacld.c`'s
    /// `CKTomega * INDinduct / m`); `1` for none.
    pub divisor: f64,
    /// The terms are imaginary parts that already include `CKTomega`
    /// (`jfet2acld.c`'s `PSacload` outputs), so they are summed unscaled.
    /// Only a load reassembled at every frequency may use them
    /// ([`crate::devices::Device::small_signal_depends_on_frequency`]).
    pub prescaled: bool,
}

/// `omega` times each of `e`'s contributions in stamp order, except that
/// the contributions of one of `groups` are scaled one by one, summed and
/// multiplied by the group's scale into a single value first
/// ([`DynamicGroup`]). The ordinary reference stamping path retains the
/// individual product rounding and contribution order; it is not a fused
/// dot product.
pub(crate) fn iter_dynamic_terms<'a>(
    e: &'a SparseMatrix,
    groups: &'a [DynamicGroup],
    omega: f64,
) -> impl Iterator<Item = (usize, usize, f64)> + 'a {
    let triplets = e.triplets();
    let mut groups = groups.iter().peekable();
    let mut i = 0;
    std::iter::from_fn(move || {
        let t = triplets.get(i)?;
        while groups
            .next_if(|g| g.range.end <= i || g.range.is_empty())
            .is_some()
        {}
        let value = match groups.peek() {
            Some(g) if g.range.start == i && g.range.end <= triplets.len() => {
                let mut terms = g
                    .terms
                    .iter()
                    .map(|term| if g.prescaled { *term } else { omega * term });
                let mut sum = terms.next().unwrap_or(0.);
                for term in terms {
                    sum += term;
                }
                i = g.range.end;
                g.scale * sum / g.divisor
            }
            _ => {
                i += 1;
                omega * t.value
            }
        };
        Some((t.row, t.col, value))
    })
}

/// An assembled circuit matrix stores only coefficients in element-ID
/// order; standalone matrices retain their sorted coordinate triples.
#[derive(Debug, Clone)]
enum Assembly {
    Bound {
        pattern: Arc<Pattern>,
        values: Vec<Complex>,
    },
    Standalone(Vec<(usize, usize, Complex)>),
}

/// The coordinate order of a factor snapshot, shared without duplicating a
/// circuit pattern's positions. Standalone reuse can still compare against
/// either kind of previous snapshot.
#[derive(Debug, Clone)]
enum ComplexOrder {
    Bound(Arc<Pattern>),
    Standalone(Arc<Vec<(usize, usize)>>),
}

impl ComplexOrder {
    fn matches(&self, entries: &[(usize, usize, Complex)]) -> bool {
        let positions = entries.iter().map(|&(r, c, _)| (r, c));
        match self {
            Self::Bound(pattern) => pattern.positions().eq(positions),
            Self::Standalone(previous) => previous.iter().copied().eq(positions),
        }
    }
}

/// Complex assembled sparse matrix, independent of real stamping storage.
#[derive(Debug, Clone)]
pub struct ComplexMatrix {
    n: usize,
    assembly: Assembly,
    pivot: PivotTolerances,
    /// The circuit unknowns' names (from the real operator), for diagnostics.
    names: Option<crate::maths::UnknownNames>,
}
impl ComplexMatrix {
    /// A trusted sorted, unique assembly pass already produced complex
    /// values. Do not split them back into two real operators and reassemble.
    pub(super) fn from_assembled(n: usize, mut entries: Vec<(usize, usize, Complex)>) -> Self {
        debug_assert!(
            entries
                .iter()
                .all(|&(r, c, v)| r < n && c < n && v.is_finite())
        );
        debug_assert!(
            entries
                .windows(2)
                .all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1))
        );
        entries.retain(|&(_, _, v)| v != Complex::ZERO);
        for (_, _, v) in &mut entries {
            // Ordinary assembly drops zero real or imaginary stamps.
            if v.re == 0. {
                v.re = 0.;
            }
            if v.im == 0. {
                v.im = 0.;
            }
        }
        Self {
            n,
            assembly: Assembly::Standalone(entries),
            pivot: PivotTolerances::DEFAULT,
            names: None,
        }
    }

    /// Assembles `A + j omega E`, preserving cancellations and checking
    /// overflow. The pivot tolerances are `a`'s.
    /// # Errors
    /// Invalid dimensions, frequency or coefficients.
    pub fn from_operators(a: &SparseMatrix, e: &SparseMatrix, omega: f64) -> SpiceResult<Self> {
        Self::from_operators_grouped(a, e, &[], omega)
    }

    /// [`Self::from_operators`] with `groups` of `e`'s stamp-order
    /// contributions (see [`dynamic_terms`]) each added as one sum.
    /// # Errors
    /// As [`Self::from_operators`].
    pub fn from_operators_grouped(
        a: &SparseMatrix,
        e: &SparseMatrix,
        groups: &[DynamicGroup],
        omega: f64,
    ) -> SpiceResult<Self> {
        square(a.rows(), a.cols())?;
        if e.rows() != a.rows() || e.cols() != a.cols() || !omega.is_finite() || omega < 0. {
            return Err(numerical(
                "complex assembly",
                "operator dimensions/frequency invalid",
            ));
        }
        let real = a
            .triplets()
            .iter()
            .map(|t| (t.row, t.col, Complex::real(t.value)));
        let imaginary = iter_dynamic_terms(e, groups, omega)
            .map(|(row, col, v)| (row, col, Complex::imaginary(v)));
        let mut assembled =
            Vec::with_capacity(a.triplets().len().saturating_add(e.triplets().len()));
        for entry in real.chain(imaginary) {
            if !entry.2.is_finite() {
                return Err(numerical(
                    "complex assembly",
                    "non-finite coefficient/overflow",
                ));
            }
            assembled.push(entry);
        }
        // Equal coordinates must retain the original stamp order, including
        // intermediate cancellations and the zero-initialized complex sum.
        assembled.sort_by_key(|&(row, col, _)| (row, col));
        let mut len = 0;
        for read in 0..assembled.len() {
            let (row, col, value) = assembled[read];
            if len > 0 {
                let last = &mut assembled[len - 1];
                if (last.0, last.1) == (row, col) {
                    last.2 = last.2 + value;
                    if !last.2.is_finite() {
                        return Err(numerical("complex assembly", "duplicate sum overflow"));
                    }
                    continue;
                }
            }
            assembled[len] = (row, col, Complex::ZERO + value);
            len += 1;
        }
        assembled.truncate(len);
        assembled.retain(|&(_, _, v)| v != Complex::ZERO);
        Ok(Self {
            n: a.rows(),
            assembly: Assembly::Standalone(assembled),
            pivot: a.pivot_tolerances(),
            names: a.unknown_names().cloned(),
        })
    }

    /// `CKTacLoad` into a circuit pattern: every element of `pattern` (zeros
    /// included, as ngspice allocates them at setup) holds the sum of `a`'s
    /// contributions as its real part and of `omega` times `e`'s as its
    /// imaginary part, each summed in stamp order (`omega` may be negative, as
    /// `CKTomega` is for `.disto`'s `f1 - f2`). `None` when a stamp lies
    /// outside the pattern.
    ///
    /// Factoring it ([`Self::factorize`]) shares the circuit's SPARSE frame
    /// and pivot order with its real solves, as `NIacIter` does.
    /// # Errors
    /// Invalid dimensions, frequency or coefficients.
    pub fn from_operators_on(
        pattern: &Arc<crate::maths::Pattern>,
        a: &SparseMatrix,
        e: &SparseMatrix,
        omega: f64,
    ) -> SpiceResult<Option<Self>> {
        Self::from_operators_grouped_on(pattern, a, e, &[], omega)
    }

    /// [`Self::from_operators_on`] with `groups` of `e`'s stamp-order
    /// contributions (see [`dynamic_terms`]) each added as one sum.
    /// # Errors
    /// As [`Self::from_operators_on`].
    pub fn from_operators_grouped_on(
        pattern: &Arc<crate::maths::Pattern>,
        a: &SparseMatrix,
        e: &SparseMatrix,
        groups: &[DynamicGroup],
        omega: f64,
    ) -> SpiceResult<Option<Self>> {
        let n = pattern.order();
        if a.rows() != n || a.cols() != n || e.rows() != n || e.cols() != n {
            return Err(numerical("complex assembly", "operator dimensions invalid"));
        }
        // `CKTomega` may be negative: `.disto` loads at `f1 - f2`.
        if !omega.is_finite() {
            return Err(numerical("complex assembly", "frequency invalid"));
        }
        let mut values = vec![Complex::ZERO; pattern.len()];
        // The bound real operator already has one accumulated value for
        // each element. Skip the row-major view and coordinate lookups if
        // it is bound to this exact pattern. Zero (including -0) was
        // omitted by the old triplet view, so keep the zero initialization.
        if let Some((source, real)) = a
            .bound_values()
            .filter(|(source, _)| Arc::ptr_eq(source, pattern))
        {
            debug_assert_eq!(source.len(), values.len());
            for (value, &re) in values.iter_mut().zip(real) {
                if !re.is_finite() {
                    return Err(numerical(
                        "complex assembly",
                        "non-finite coefficient/overflow",
                    ));
                }
                if re != 0.0 {
                    value.re = re;
                }
            }
        } else {
            for t in a.triplets() {
                let Some(id) = pattern.find(t.row, t.col) else {
                    return Ok(None);
                };
                let value = &mut values[id.index()];
                value.re += t.value;
                if !value.is_finite() {
                    return Err(numerical(
                        "complex assembly",
                        "non-finite coefficient/overflow",
                    ));
                }
            }
        }
        if groups.is_empty()
            && let Some((_, dynamic)) = e
                .bound_values()
                .filter(|(source, _)| Arc::ptr_eq(source, pattern))
        {
            // An already accumulated E has only one coefficient per
            // element. Preserve the previous zero-initialized addition,
            // including the sign of an underflowed imaginary term.
            for (value, &coefficient) in values.iter_mut().zip(dynamic) {
                if coefficient != 0.0 {
                    let contribution = omega * coefficient;
                    value.im += contribution;
                    if !value.is_finite() {
                        return Err(numerical(
                            "complex assembly",
                            "non-finite coefficient/overflow",
                        ));
                    }
                }
            }
        } else {
            for (row, col, v) in iter_dynamic_terms(e, groups, omega) {
                let Some(id) = pattern.find(row, col) else {
                    return Ok(None);
                };
                let value = &mut values[id.index()];
                value.im += v;
                if !value.is_finite() {
                    return Err(numerical(
                        "complex assembly",
                        "non-finite coefficient/overflow",
                    ));
                }
            }
        }
        Ok(Some(Self {
            n,
            assembly: Assembly::Bound {
                pattern: Arc::clone(pattern),
                values,
            },
            pivot: a.pivot_tolerances(),
            names: a.unknown_names().cloned(),
        }))
    }

    /// Orders and factors a brand-new SPARSE matrix whose elements are
    /// created in `entries` order (for a bound matrix, the circuit's element
    /// order), with `SMPcReorder` alone: no `spMNA_Preorder` and no pivot
    /// order inherited from a real factorization. This is the matrix
    /// `sens_sens()` (`cktsens.c`) factors at each frequency, after
    /// `CKTunsetup`/`CKTsetup` rebuilt the circuit matrix and `NIacIter`
    /// reordered it.
    /// # Errors
    /// Structurally or numerically singular matrix.
    pub fn factorize_new_matrix(&self) -> SpiceResult<ComplexLu> {
        let mut sparse = self.fresh_frame()?;
        if !sparse.fully_translated() {
            // A circuit's matrix follows C (see `maths::linear`).
            if self.names.is_none() && matches!(self.assembly, Assembly::Standalone(_)) {
                return Err(numerical(
                    "complex sparse LU",
                    "singular matrix: an unknown has no matrix entries",
                ));
            }
            crate::maths::linear::warn_untranslated(&sparse, self.names.as_ref());
        }
        let status = sparse.order_and_factor(self.pivot.relative, self.pivot.absolute);
        check_status("complex sparse LU", status, &sparse, self.names.as_ref())?;
        Ok(ComplexLu {
            n: self.n,
            order: self.order_key(),
            factors: ComplexFactors {
                work: sparse.last_attempt(),
                sparse: Arc::new(sparse),
            },
        })
    }

    /// Orders and factors (`SMPpreOrder` + `SMPcReorder`).
    /// # Errors
    /// Structurally or numerically singular matrix.
    pub fn factorize(&self) -> SpiceResult<ComplexLu> {
        if let Assembly::Bound { pattern, values } = &self.assembly {
            let factors = crate::maths::linear::factor_shared_complex(
                pattern,
                values,
                self.pivot,
                self.names.as_ref(),
            )?;
            return Ok(ComplexLu {
                n: self.n,
                order: self.order_key(),
                factors,
            });
        }
        let mut sparse = self.fresh_frame()?;
        order_fresh(
            "complex sparse LU",
            &mut sparse,
            self.pivot,
            self.names.as_ref(),
            self.names.is_some(),
        )?;
        Ok(ComplexLu {
            n: self.n,
            order: self.order_key(),
            factors: ComplexFactors {
                work: sparse.last_attempt(),
                sparse: Arc::new(sparse),
            },
        })
    }

    fn fresh_frame(&self) -> SpiceResult<Sparse<Complex>> {
        match &self.assembly {
            Assembly::Bound { pattern, values } => Sparse::from_pattern(pattern, values),
            Assembly::Standalone(entries) => {
                Sparse::from_originals(self.n, entries.iter().map(|&(r, c, v)| (r + 1, c + 1, v)))
            }
        }
    }

    fn order_key(&self) -> ComplexOrder {
        match &self.assembly {
            Assembly::Bound { pattern, .. } => ComplexOrder::Bound(Arc::clone(pattern)),
            Assembly::Standalone(entries) => ComplexOrder::Standalone(Arc::new(
                entries.iter().map(|&(r, c, _)| (r, c)).collect(),
            )),
        }
    }

    /// Factor an independent complex operator with another point's symbolic
    /// topology, without mutating the circuit's cached pivot order.
    ///
    /// Numeric factors and per-point pivot status remain owned independently.
    /// A zero pivot retries with a fresh threshold order for this point only.
    /// Callers must not use this for a warm-started, frequency-dependent bias.
    ///
    /// # Errors
    /// Invalid or singular numeric factorization.
    pub fn factorize_independent(&self, previous: Option<&ComplexLu>) -> SpiceResult<ComplexLu> {
        let Some(previous) = previous else {
            return self.factorize_new_matrix();
        };
        let matching = match (&self.assembly, &previous.order) {
            (Assembly::Bound { pattern, .. }, ComplexOrder::Bound(old)) => {
                Arc::ptr_eq(pattern, old)
            }
            (Assembly::Standalone(entries), order) => order.matches(entries),
            _ => false,
        };
        if previous.n != self.n || !matching {
            return self.factorize_new_matrix();
        }
        let mut sparse = match &self.assembly {
            Assembly::Bound { values, .. } => {
                previous.factors.sparse.fork_loaded(values.iter().copied())
            }
            Assembly::Standalone(entries) => previous
                .factors
                .sparse
                .fork_loaded(entries.iter().map(|&(_, _, value)| value)),
        };
        let mut status = sparse.factor_with_workspace(&mut Vec::new());
        let mut work = sparse.last_attempt();
        if status == Status::ZeroDiag {
            work.zero_pivot_retries += 1;
            match &self.assembly {
                Assembly::Bound { values, .. } => {
                    sparse.reload_originals_iter(values.iter().copied());
                }
                Assembly::Standalone(entries) => {
                    sparse.reload_originals_iter(entries.iter().map(|&(_, _, value)| value));
                }
            }
            status = sparse.order_and_factor(self.pivot.relative, self.pivot.absolute);
            work += sparse.last_attempt();
        }
        check_status("complex sparse LU", status, &sparse, self.names.as_ref())?;
        Ok(ComplexLu {
            n: self.n,
            order: self.order_key(),
            factors: ComplexFactors {
                work,
                sparse: Arc::new(sparse),
            },
        })
    }

    /// Refactors with the pivot order of `previous` when the pattern is
    /// identical (`SMPcLUfac`, `spFactor`), reordering after a zero pivot
    /// (`SMPcReorder`) as `NIacIter` does; otherwise orders afresh.
    /// # Errors
    /// Structurally or numerically singular matrix.
    pub fn factorize_reusing(&self, previous: &ComplexLu) -> SpiceResult<ComplexLu> {
        let Assembly::Standalone(entries) = &self.assembly else {
            // The circuit pattern carries the pivot order itself.
            return self.factorize();
        };
        if previous.n != self.n || !previous.order.matches(entries) {
            return self.factorize();
        }
        let mut sparse = previous
            .factors
            .sparse
            .fork_loaded(entries.iter().map(|&(_, _, v)| v));
        let load = |sparse: &mut Sparse<Complex>| {
            sparse.reload_originals_iter(entries.iter().map(|&(_, _, v)| v));
        };
        let mut status = sparse.factor_with_workspace(&mut Vec::new());
        let mut work = sparse.last_attempt();
        if status == Status::ZeroDiag {
            work.zero_pivot_retries += 1;
            load(&mut sparse);
            status = sparse.order_and_factor(self.pivot.relative, self.pivot.absolute);
            work += sparse.last_attempt();
        }
        check_status("complex sparse LU", status, &sparse, self.names.as_ref())?;
        Ok(ComplexLu {
            n: self.n,
            order: previous.order.clone(),
            factors: ComplexFactors {
                work,
                sparse: Arc::new(sparse),
            },
        })
    }
}

/// Independent small-signal points share an immutable sparse elimination plan,
/// never their numeric pivots or the circuit's mutable ordering cache.
/// A tile contains at most four points and is allowed to fail per lane.
#[derive(Debug, Default)]
pub struct IndependentComplexBatch {
    previous: Option<ComplexLu>,
}

impl IndependentComplexBatch {
    /// Factor independent matrices in input order. A failing point terminates
    /// the tile; a zero pivot retries that lane with a new scalar pivot search.
    /// Tail tiles are masked by their actual length, not by dummy operators.
    pub fn factorize(&mut self, matrices: &[ComplexMatrix]) -> Vec<SpiceResult<ComplexLu>> {
        let mut results = Vec::with_capacity(matrices.len());
        if matrices.is_empty() {
            return results;
        }
        let mut offset = 0;
        if self.previous.is_none() {
            let first = matrices[0].factorize_independent(None);
            match first {
                Ok(factor) => {
                    self.previous = Some(factor.clone());
                    results.push(Ok(factor));
                    offset = 1;
                }
                Err(error) => {
                    results.push(Err(error));
                    return results;
                }
            }
        }
        let previous = self
            .previous
            .as_ref()
            .expect("first point produced ordering");
        let same_order = matrices[offset..].iter().all(|matrix| {
            matrix.n == previous.n
                && match (&matrix.assembly, &previous.order) {
                    (Assembly::Bound { pattern, .. }, ComplexOrder::Bound(old)) => {
                        Arc::ptr_eq(pattern, old)
                    }
                    (Assembly::Standalone(entries), order) => order.matches(entries),
                    _ => false,
                }
        });
        if same_order && matrices.len() - offset >= 2 {
            let mut frames: Vec<_> = matrices[offset..]
                .iter()
                .map(|matrix| match &matrix.assembly {
                    Assembly::Bound { values, .. } => {
                        previous.factors.sparse.fork_loaded(values.iter().copied())
                    }
                    Assembly::Standalone(entries) => previous
                        .factors
                        .sparse
                        .fork_loaded(entries.iter().map(|&(_, _, value)| value)),
                })
                .collect();
            if let Some(statuses) = Sparse::factor_independent_tile(&mut frames) {
                for ((matrix, mut sparse), mut status) in
                    matrices[offset..].iter().zip(frames).zip(statuses)
                {
                    let mut work = sparse.last_attempt();
                    if status == Status::ZeroDiag {
                        work.zero_pivot_retries += 1;
                        status =
                            sparse.order_and_factor(matrix.pivot.relative, matrix.pivot.absolute);
                        work += sparse.last_attempt();
                    }
                    if let Err(error) =
                        check_status("complex sparse LU", status, &sparse, matrix.names.as_ref())
                    {
                        results.push(Err(error));
                        return results;
                    }
                    let factor = ComplexLu {
                        n: matrix.n,
                        order: matrix.order_key(),
                        factors: ComplexFactors {
                            sparse: Arc::new(sparse),
                            work,
                        },
                    };
                    self.previous = Some(factor.clone());
                    results.push(Ok(factor));
                }
                return results;
            }
        }
        for matrix in &matrices[offset..] {
            match matrix.factorize_independent(self.previous.as_ref()) {
                Ok(factor) => {
                    self.previous = Some(factor.clone());
                    results.push(Ok(factor));
                }
                Err(error) => {
                    results.push(Err(error));
                    break;
                }
            }
        }
        results
    }
}

/// Owned complex factors; subsequent real operator mutations cannot stale them.
#[derive(Debug, Clone)]
pub struct ComplexLu {
    n: usize,
    order: ComplexOrder,
    factors: ComplexFactors,
}
impl ComplexLu {
    /// Actual structural and numeric storage in this immutable factor snapshot.
    #[must_use]
    pub fn factor_storage(&self) -> FactorStorage {
        self.factors.sparse.storage()
    }

    /// Actual work of the operation producing this snapshot.
    #[must_use]
    pub const fn factor_work(&self) -> FactorWork {
        self.factors.work
    }

    /// Solves `A x = b` (`SMPcSolve`).
    /// # Errors
    /// Wrong RHS size, non-finite RHS or solution.
    pub fn solve(&self, rhs: &[Complex]) -> SpiceResult<Vec<Complex>> {
        self.solve_with(rhs, false)
    }

    /// Solves the transposed (not conjugated) system `A^T x = b` with the same
    /// factors. This is the adjoint solve of `.noise` (C `SMPcaSolve` in
    /// `NInzIter`).
    /// # Errors
    /// Wrong RHS size, non-finite RHS or solution.
    pub fn solve_transposed(&self, rhs: &[Complex]) -> SpiceResult<Vec<Complex>> {
        self.solve_with(rhs, true)
    }

    fn solve_with(&self, rhs: &[Complex], transposed: bool) -> SpiceResult<Vec<Complex>> {
        if let ComplexOrder::Bound(pattern) = &self.order {
            pattern.with_complex_workspace(|workspace| self.solve_in(rhs, transposed, workspace))
        } else {
            self.solve_in(rhs, transposed, &mut Vec::new())
        }
    }

    /// Solves using caller-owned scratch, for repeated RHS or concurrent
    /// independent solves. The scratch contents are unspecified afterward.
    /// # Errors
    /// Wrong RHS size, non-finite RHS or solution.
    pub fn solve_with_workspace(
        &self,
        rhs: &[Complex],
        workspace: &mut Vec<Complex>,
    ) -> SpiceResult<Vec<Complex>> {
        self.solve_in(rhs, false, workspace)
    }

    /// Transposed counterpart of [`Self::solve_with_workspace`].
    /// # Errors
    /// Wrong RHS size, non-finite RHS or solution.
    pub fn solve_transposed_with_workspace(
        &self,
        rhs: &[Complex],
        workspace: &mut Vec<Complex>,
    ) -> SpiceResult<Vec<Complex>> {
        self.solve_in(rhs, true, workspace)
    }

    fn solve_in(
        &self,
        rhs: &[Complex],
        transposed: bool,
        workspace: &mut Vec<Complex>,
    ) -> SpiceResult<Vec<Complex>> {
        let mut out = Vec::new();
        self.solve_into(rhs, workspace, &mut out, transposed)?;
        Ok(out)
    }

    /// Solve into caller-owned buffers, retaining both internal scratch and
    /// external-order result capacity between port excitations. The result
    /// and scratch contents are unspecified on error.
    pub(crate) fn solve_into(
        &self,
        rhs: &[Complex],
        workspace: &mut Vec<Complex>,
        out: &mut Vec<Complex>,
        transposed: bool,
    ) -> SpiceResult<()> {
        if rhs.len() != self.n || rhs.iter().any(|v| !v.is_finite()) {
            return Err(numerical("complex solve", "invalid RHS"));
        }
        self.factors
            .sparse
            .factors()?
            .solve_into(rhs, workspace, out, transposed)?;
        if out.iter().any(|v| !v.is_finite()) {
            return Err(numerical(
                "complex solve",
                "non-finite solution/singular system",
            ));
        }
        Ok(())
    }

    /// Solve consecutive RHS-major vectors using one factor traversal per tile.
    ///
    /// Output preserves RHS order and external matrix ordering. Transposition
    /// does not conjugate the factors. Scratch and output capacities are reusable.
    ///
    /// # Errors
    /// Invalid RHS length, non-finite input or solution, or insufficient storage.
    pub fn solve_batch_into(
        &self,
        rhs: &[Complex],
        workspace: &mut SolveBatchWorkspace<Complex>,
        out: &mut Vec<Complex>,
        transposed: bool,
    ) -> Result<(), BatchSolveFailure> {
        self.factors
            .sparse
            .factors()?
            .solve_batch_into(rhs, workspace, out, transposed)
    }

    /// Binary-scaled determinant: `(mantissa, base-2 exponent)` with
    /// `1 <= max(|re|, |im|) < 2` normalization.
    /// # Errors
    /// Non-finite/invalid inverse factors or an out-of-range exponent.
    pub fn determinant(&self) -> SpiceResult<(Complex, i64)> {
        self.factors.sparse.factors()?.determinant()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_frequency_tiles_match_scalar_solve_and_transpose() {
        let positions = (0..3).flat_map(|r| (0..3).map(move |c| (r, c)));
        let pattern = Arc::new(Pattern::new(3, positions).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        let mut e = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        for (r, c, conductance, capacitance) in [
            (0, 0, 5., 0.125),
            (0, 1, -0.5, -0.05),
            (1, 0, 0.75, 0.025),
            (1, 1, 7., 0.3),
            (1, 2, -0.25, 0.2),
            (2, 1, -0.125, 0.08),
            (2, 2, 9., 0.4),
        ] {
            a.add(r, c, conductance).unwrap();
            e.add(r, c, capacitance).unwrap();
        }
        let frequencies = [0., 0.01, 0.1, 1., 3., 10., 20.];
        let matrices: Vec<_> = frequencies
            .map(|omega| {
                ComplexMatrix::from_operators_on(&pattern, &a, &e, omega)
                    .unwrap()
                    .unwrap()
            })
            .into();
        let rhs = [
            Complex::new(1., 2.),
            Complex::new(-2., 3.),
            Complex::new(4., -1.),
        ];
        let mut batch = IndependentComplexBatch::default();
        let mut refactored = 0;
        for tile in matrices.chunks(4) {
            let factors = batch.factorize(tile);
            assert_eq!(factors.len(), tile.len());
            for (matrix, factor) in tile.iter().zip(factors) {
                let factor = factor.unwrap();
                refactored += factor.factor_work().fixed_refactors;
                let scalar = matrix.factorize_new_matrix().unwrap();
                for transpose in [false, true] {
                    let actual = if transpose {
                        factor.solve_transposed(&rhs).unwrap()
                    } else {
                        factor.solve(&rhs).unwrap()
                    };
                    let expected = if transpose {
                        scalar.solve_transposed(&rhs).unwrap()
                    } else {
                        scalar.solve(&rhs).unwrap()
                    };
                    for (actual, expected) in actual.into_iter().zip(expected) {
                        let scale = expected.re.abs().max(expected.im.abs()).max(1.);
                        assert!((actual.re - expected.re).abs() <= 1e-12 * scale);
                        assert!((actual.im - expected.im).abs() <= 1e-12 * scale);
                    }
                }
            }
        }
        assert!(
            refactored >= 5,
            "the independent SIMD refactor path was not used"
        );
    }

    #[test]
    fn independent_tile_isolates_a_singular_frequency_lane() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (1, 1)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        let mut e = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        a.add(0, 0, 2.).unwrap();
        e.add(1, 1, 1.).unwrap();
        let matrices: Vec<_> = [1., 2., 0., 4.]
            .map(|omega| {
                ComplexMatrix::from_operators_on(&pattern, &a, &e, omega)
                    .unwrap()
                    .unwrap()
            })
            .into();
        let mut batch = IndependentComplexBatch::default();
        let factors = batch.factorize(&matrices);
        assert_eq!(factors.len(), 3);
        assert!(factors[0].is_ok());
        assert!(factors[1].is_ok());
        assert!(factors[2].is_err());
        let rhs = [Complex::real(2.), Complex::real(1.)];
        let reference = matrices[1].factorize_new_matrix().unwrap();
        let actual = factors[1].as_ref().unwrap().solve(&rhs).unwrap();
        assert_eq!(actual, reference.solve(&rhs).unwrap());
    }

    #[test]
    fn standalone_assembly_preserves_grouped_stamp_order_and_zero_bits() {
        let mut a = SparseMatrix::new(3, 3);
        let mut e = SparseMatrix::new(3, 3);
        let mut seed = 7_u64;
        // Mixed coordinates and magnitudes plus exact cancellations.
        for i in 0..300 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let row = ((seed >> 32) % 3) as usize;
            let col = ((seed >> 40) % 3) as usize;
            let v = [1e16, 1., -1e16, 0.125, -0.125][i % 5];
            a.add(row, col, v).unwrap();
            e.add(col, row, v * 0.5).unwrap();
        }
        let groups = [DynamicGroup {
            range: 0..1,
            terms: vec![1., -2., 0.25],
            scale: 3.,
            divisor: 2.,
            prescaled: false,
        }];
        let omega = 0.25;
        let mut expected = std::collections::BTreeMap::new();
        let real = a
            .triplets()
            .iter()
            .map(|t| (t.row, t.col, Complex::real(t.value)));
        let imag =
            iter_dynamic_terms(&e, &groups, omega).map(|(r, c, v)| (r, c, Complex::imaginary(v)));
        for (r, c, v) in real.chain(imag) {
            let entry = expected.entry((r, c)).or_insert(Complex::ZERO);
            *entry = *entry + v;
        }
        let matrix = ComplexMatrix::from_operators_grouped(&a, &e, &groups, omega).unwrap();
        let Assembly::Standalone(entries) = matrix.assembly else {
            panic!("standalone matrix");
        };
        let actual = entries
            .iter()
            .map(|&(r, c, v)| ((r, c), v.re.to_bits(), v.im.to_bits()));
        let expected = expected
            .into_iter()
            .filter(|(_, v)| *v != Complex::ZERO)
            .map(|(rc, v)| (rc, v.re.to_bits(), v.im.to_bits()));
        assert!(actual.eq(expected));

        // Intermediate overflow is an error even if later terms cancel it.
        let mut overflow = SparseMatrix::new(1, 1);
        for v in [f64::MAX, f64::MAX, -f64::MAX] {
            overflow.add(0, 0, v).unwrap();
        }
        assert!(ComplexMatrix::from_operators(&overflow, &SparseMatrix::new(1, 1), 1.).is_err());
    }

    #[test]
    fn complex_workspace_is_reused_across_snapshots_and_contended_solves() {
        let pattern = Arc::new(Pattern::new(2, [(1, 0), (0, 1), (0, 0), (1, 1)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        let mut e = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        for (r, c, v) in [(0, 0, 3.), (0, 1, 1.), (1, 0, 2.), (1, 1, 4.)] {
            a.add(r, c, v).unwrap();
            e.add(r, c, v * 0.125).unwrap();
        }
        let factor = |omega| {
            ComplexMatrix::from_operators_on(&pattern, &a, &e, omega)
                .unwrap()
                .unwrap()
                .factorize()
                .unwrap()
        };
        let rhs = [Complex::new(1., 2.), Complex::new(-3., 4.)];
        let first = factor(1.);
        let expected = first.solve(&rhs).unwrap();
        let addr = pattern.with_complex_workspace(|w| w.as_ptr());
        let second = factor(2.);
        assert_ne!(second.solve(&rhs).unwrap(), expected);
        assert_eq!(pattern.with_complex_workspace(|w| w.as_ptr()), addr);
        let transposed = first.solve_transposed(&rhs).unwrap();
        let mut workspace = Vec::new();
        assert_eq!(
            first.solve_with_workspace(&rhs, &mut workspace).unwrap(),
            expected
        );
        let addr2 = workspace.as_ptr();
        assert_eq!(
            first
                .solve_transposed_with_workspace(&rhs, &mut workspace)
                .unwrap(),
            transposed
        );
        assert_eq!(workspace.as_ptr(), addr2);
        // Held slot forces the local fallback, including when the caller
        // holds an older immutable factor snapshot.
        pattern.with_complex_workspace(|_| {
            assert_eq!(first.solve(&rhs).unwrap(), expected);
            assert_eq!(first.solve_transposed(&rhs).unwrap(), transposed);
        });
        assert_eq!(pattern.with_complex_workspace(|w| w.as_ptr()), addr);
        assert!(
            first
                .solve_with_workspace(&rhs[..1], &mut workspace)
                .is_err()
        );
    }

    #[test]
    fn bound_dynamic_coefficients_match_triplet_path_bit_for_bit() {
        let pattern = Arc::new(Pattern::new(2, [(1, 1), (0, 0), (0, 1), (1, 0)]).unwrap());
        let a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        let mut e = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        e.set(1, 1, 0.5).unwrap();
        e.set(0, 0, 1e-300).unwrap();
        e.set(0, 1, -0.).unwrap();
        e.set(1, 0, -1e-200).unwrap();
        let mut plain = SparseMatrix::new(2, 2);
        for t in e.triplets() {
            plain.add(t.row, t.col, t.value).unwrap();
        }
        let bits = |e: &SparseMatrix, omega| {
            let matrix = ComplexMatrix::from_operators_on(&pattern, &a, e, omega)
                .unwrap()
                .unwrap();
            let Assembly::Bound { values, .. } = matrix.assembly else {
                panic!("expected bound matrix");
            };
            values
                .iter()
                .map(|v| (v.re.to_bits(), v.im.to_bits()))
                .collect::<Vec<_>>()
        };
        for omega in [-2., 0., 1e-30, 1e200] {
            assert_eq!(bits(&e, omega), bits(&plain, omega));
        }
    }

    #[test]
    fn bound_and_standalone_assembly_follow_stamp_order() {
        let pattern = Arc::new(Pattern::new(2, [(1, 1), (0, 1), (0, 0), (1, 0)]).unwrap());
        let mut bound = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        let mut triplets = SparseMatrix::new(2, 2);
        for (r, c, v) in [(0, 0, 3.), (1, 1, 4.), (0, 1, -0.), (1, 0, 2.), (0, 0, 1.)] {
            bound.add(r, c, v).unwrap();
            triplets.add(r, c, v).unwrap();
        }
        let mut e = SparseMatrix::new(2, 2);
        for v in [1e16, -1e16, 1.] {
            e.add(0, 1, v).unwrap();
        }
        e.add(1, 1, 0.5).unwrap();
        let groups = [
            DynamicGroup {
                range: 0..3,
                terms: vec![1e16, -1e16, 1.],
                scale: 3.,
                divisor: 2.,
                prescaled: true,
            },
            DynamicGroup {
                range: 3..4,
                terms: vec![0.5],
                scale: 1.,
                divisor: 1.,
                prescaled: false,
            },
        ];
        let values = |a: &SparseMatrix| {
            let matrix = ComplexMatrix::from_operators_grouped_on(&pattern, a, &e, &groups, -2.)
                .unwrap()
                .unwrap();
            let Assembly::Bound { values, .. } = matrix.assembly else {
                panic!("expected bound values")
            };
            values
        };
        assert_eq!(values(&bound), values(&triplets));
        assert_eq!(
            values(&bound),
            [
                Complex::new(4., -1.),
                Complex::imaginary(1.5),
                Complex::real(4.),
                Complex::real(2.)
            ]
        );
        let mut overflow = SparseMatrix::new(2, 2);
        overflow.add(0, 0, f64::MAX).unwrap();
        overflow.add(0, 0, f64::MAX).unwrap();
        assert!(ComplexMatrix::from_operators_on(&pattern, &overflow, &e, 1.).is_err());
    }
}
