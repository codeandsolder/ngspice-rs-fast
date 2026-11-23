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
use crate::primitives::{Complex, SpiceError, SpiceResult};

/// Arithmetic of one grouped dynamic element, including physical units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DynamicMode {
    /// Angular-frequency-scaled capacitance: scale * sum(omega*term)/divisor.
    Scaled {
        /// Multiplier applied after summing all frequency-scaled terms.
        scale: f64,
        /// Divider applied after multiplying the sum.
        divisor: f64,
    },
    /// Terms already include angular frequency, so scale * sum(term).
    Prescaled {
        /// Multiplier applied after summing the already-scaled terms.
        scale: f64,
    },
}
impl DynamicMode {
    const fn stamp(self, term: f64) -> f64 {
        match self {
            Self::Scaled { scale, divisor } => scale * term / divisor,
            Self::Prescaled { scale } => scale * term,
        }
    }

    fn coefficient(self, terms: &[f64], omega: f64) -> f64 {
        let mut iter = terms.iter().map(|&term| match self {
            Self::Scaled { .. } => omega * term,
            Self::Prescaled { .. } => term,
        });
        let mut sum = iter.next().unwrap_or(0.);
        for value in iter {
            sum += value;
        }
        match self {
            Self::Scaled { scale, divisor } => scale * sum / divisor,
            Self::Prescaled { scale } => scale * sum,
        }
    }

    fn pencil(self, terms: &[f64]) -> SpiceResult<f64> {
        let Self::Scaled { scale, .. } = self else {
            return Err(dynamic_error(
                "prescaled AC terms cannot form a pole-zero pencil",
            ));
        };
        let mut iter = terms.iter().copied();
        let mut sum = iter.next().unwrap_or(0.);
        for term in iter {
            sum += term;
        }
        // Preserve the C pencil's original scale * sum coefficient.
        Ok(scale * sum)
    }
}

/// Verified ranges in one unfolded E operator; fields remain private.
#[derive(Debug, Clone, PartialEq)]
struct Group {
    stamps: std::ops::Range<usize>,
    terms: std::ops::Range<usize>,
    position: (usize, usize),
    mode: DynamicMode,
}

/// Read-only group view; all coefficients borrow one contiguous arena.
pub struct DynamicGroup<'a> {
    /// Contiguous original nonzero E stamps replaced by this statement.
    pub range: std::ops::Range<usize>,
    /// Source terms in original C addition order, including zeros.
    pub terms: &'a [f64],
    /// Whether terms already include frequency scaling.
    pub mode: DynamicMode,
}

/// One reusable set of terms and ordered replacement ranges.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DynamicGroups {
    owner: Option<Arc<()>>,
    groups: Vec<Group>,
    terms: Vec<f64>,
    sealed_generation: Option<u64>,
}

fn dynamic_error(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "dynamic stamp groups".into(),
        message: message.into(),
    }
}

impl DynamicGroups {
    /// Create a reusable, initially empty group and term arena.
    pub const fn new() -> Self {
        Self {
            owner: None,
            groups: Vec::new(),
            terms: Vec::new(),
            sealed_generation: None,
        }
    }

    /// Inspect ordered group metadata without allocating or transferring ownership.
    pub fn iter(&self) -> impl Iterator<Item = DynamicGroup<'_>> {
        self.groups.iter().map(|group| DynamicGroup {
            range: group.stamps.clone(),
            terms: &self.terms[group.terms.clone()],
            mode: group.mode,
        })
    }

    /// Number of C grouped-element statements.
    pub const fn len(&self) -> usize {
        self.groups.len()
    }
    /// Whether the arena contains no grouped statements.
    pub const fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }
    /// Retained capacities of the statement and raw term arenas.
    pub fn capacities(&self) -> (usize, usize) {
        (self.groups.capacity(), self.terms.capacity())
    }

    pub(crate) fn clear(&mut self) {
        self.groups.clear();
        self.terms.clear();
        self.owner = None;
        self.sealed_generation = None;
    }

    pub(crate) fn rebind(&mut self, e: &SparseMatrix) {
        self.owner = (!self.groups.is_empty()).then(|| e.stamp_owner());
        self.sealed_generation = self.owner.as_ref().map(|_| e.stamp_generation());
    }

    /// Register a group only after its ordered nonzero stamps were added.
    /// Stored zero source terms remain visible during the complex sum.
    pub(crate) fn register(
        &mut self,
        e: &SparseMatrix,
        stamps: std::ops::Range<usize>,
        position: (usize, usize),
        terms: &[f64],
        mode: DynamicMode,
    ) -> SpiceResult<()> {
        if self
            .owner
            .as_ref()
            .is_some_and(|owner| !e.has_stamp_owner(owner))
            || self
                .groups
                .last()
                .is_some_and(|last| last.stamps.end > stamps.start)
            || stamps.is_empty()
            || terms.is_empty()
        {
            return Err(dynamic_error("overlapping, empty or stale dynamic group"));
        }
        let candidate = Group {
            stamps: stamps.clone(),
            terms: 0..terms.len(),
            position,
            mode,
        };
        Self::check_group(e, &candidate, terms)?;
        self.groups
            .try_reserve(1)
            .map_err(|error| dynamic_error(format!("group arena allocation failed: {error}")))?;
        self.terms
            .try_reserve(terms.len())
            .map_err(|error| dynamic_error(format!("term arena allocation failed: {error}")))?;
        if self.owner.is_none() {
            self.owner = Some(e.stamp_owner());
        }
        let begin = self.terms.len();
        self.terms.extend_from_slice(terms);
        self.groups.push(Group {
            stamps,
            terms: begin..self.terms.len(),
            position,
            mode,
        });
        self.sealed_generation = None;
        Ok(())
    }

    fn check_group(e: &SparseMatrix, group: &Group, terms: &[f64]) -> SpiceResult<()> {
        if e.is_folded() || group.stamps.is_empty() || group.terms.is_empty() {
            return Err(dynamic_error(
                "group requires unfolded, nonempty E contributions",
            ));
        }
        let stamps = e
            .triplets()
            .get(group.stamps.clone())
            .ok_or_else(|| dynamic_error("group range exceeds its E operator"))?;
        let mut index = 0;
        for &term in terms {
            let stamped = group.mode.stamp(term);
            if stamped == 0. {
                continue;
            }
            let Some(t) = stamps.get(index) else {
                return Err(dynamic_error("group has fewer stamps than nonzero terms"));
            };
            if (t.row, t.col) != group.position || t.value.to_bits() != stamped.to_bits() {
                return Err(dynamic_error("group differs from ordered E stamps"));
            }
            index += 1;
        }
        if index != stamps.len() {
            return Err(dynamic_error("group contains unrelated E stamps"));
        }
        Ok(())
    }

    /// Verify one completed operator assembly and seal the metadata.
    pub(crate) fn seal(&mut self, e: &SparseMatrix) -> SpiceResult<()> {
        self.validate_unsealed(e)?;
        self.sealed_generation = (!self.groups.is_empty()).then(|| e.stamp_generation());
        Ok(())
    }

    fn validate_unsealed(&self, e: &SparseMatrix) -> SpiceResult<()> {
        if self.groups.is_empty() {
            if !self.terms.is_empty() || self.owner.is_some() {
                return Err(dynamic_error("orphan dynamic term arena"));
            }
            return Ok(());
        }
        if !self
            .owner
            .as_ref()
            .is_some_and(|owner| e.has_stamp_owner(owner))
        {
            return Err(dynamic_error("group belongs to a different E operator"));
        }
        let (mut next_stamp, mut next_term) = (0, 0);
        for group in &self.groups {
            if group.stamps.start < next_stamp
                || group.terms.start != next_term
                || group.terms.end > self.terms.len()
            {
                return Err(dynamic_error("unordered or overlapping group ranges"));
            }
            Self::check_group(e, group, &self.terms[group.terms.clone()])?;
            next_stamp = group.stamps.end;
            next_term = group.terms.end;
        }
        if next_term != self.terms.len() {
            return Err(dynamic_error("unowned dynamic term suffix"));
        }
        Ok(())
    }

    fn validate(&self, e: &SparseMatrix) -> SpiceResult<()> {
        if let Some(generation) = self.sealed_generation {
            if self
                .owner
                .as_ref()
                .is_some_and(|owner| e.has_stamp_owner(owner))
                && e.stamp_generation() == generation
            {
                return Ok(());
            }
            return Err(dynamic_error("sealed dynamic group has a stale E operator"));
        }
        self.validate_unsealed(e)
    }

    /// Verify that every group still belongs to this exact, unchanged E stream.
    /// # Errors
    /// Invalid or stale group metadata.
    pub fn validate_operator(&self, e: &SparseMatrix) -> SpiceResult<()> {
        self.validate(e)
    }

    /// Convert a checked capacitance statement to the C pole-zero pencil.
    /// # Errors
    /// A frequency-prescaled term cannot contribute as an unscaled capacitance.
    pub fn pencil_value(&self, group: &DynamicGroup<'_>) -> SpiceResult<f64> {
        group.mode.pencil(group.terms)
    }
}

/// Expand E's stamp stream in original order, replacing each group exactly
/// once. Invalid metadata is a structural error, never silently skipped.
pub(crate) fn iter_dynamic_terms<'a>(
    e: &'a SparseMatrix,
    groups: &'a DynamicGroups,
    omega: f64,
) -> SpiceResult<impl Iterator<Item = (usize, usize, f64)> + 'a> {
    groups.validate(e)?;
    let triplets = e.triplets();
    let mut entries = groups.groups.iter().peekable();
    let mut i = 0;
    Ok(std::iter::from_fn(move || {
        let stamp = triplets.get(i)?;
        let value = match entries.peek() {
            Some(group) if group.stamps.start == i => {
                let value = group
                    .mode
                    .coefficient(&groups.terms[group.terms.clone()], omega);
                i = group.stamps.end;
                entries.next();
                value
            }
            _ => {
                i += 1;
                omega * stamp.value
            }
        };
        Some((stamp.row, stamp.col, value))
    }))
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
        Self::from_operators_grouped(a, e, &DynamicGroups::default(), omega)
    }

    /// [`Self::from_operators`] with `groups` of `e`'s stamp-order
    /// contributions (see [`dynamic_terms`]) each added as one sum.
    /// # Errors
    /// As [`Self::from_operators`].
    pub fn from_operators_grouped(
        a: &SparseMatrix,
        e: &SparseMatrix,
        groups: &DynamicGroups,
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
        let imaginary = iter_dynamic_terms(e, groups, omega)?
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
        Self::from_operators_grouped_on(pattern, a, e, &DynamicGroups::default(), omega)
    }

    /// [`Self::from_operators_on`] with `groups` of `e`'s stamp-order
    /// contributions (see [`dynamic_terms`]) each added as one sum.
    /// # Errors
    /// As [`Self::from_operators_on`].
    pub fn from_operators_grouped_on(
        pattern: &Arc<crate::maths::Pattern>,
        a: &SparseMatrix,
        e: &SparseMatrix,
        groups: &DynamicGroups,
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
            for (row, col, v) in iter_dynamic_terms(e, groups, omega)? {
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

/// Owned complex factors; subsequent real operator mutations cannot stale them.
#[derive(Debug)]
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
    fn dynamic_arena_enforces_stamp_identity_ranges_and_reuses_storage() {
        let mut e = SparseMatrix::new(2, 2);
        let mut groups = DynamicGroups::new();
        let mode = DynamicMode::Scaled {
            scale: 3.,
            divisor: 2.,
        };
        let original = [1., 0., -2., 0.25];
        for &term in &original {
            e.add(0, 0, mode.stamp(term)).unwrap();
        }
        assert_eq!(e.triplets().len(), 3);
        groups.register(&e, 0..3, (0, 0), &original, mode).unwrap();
        assert_eq!(groups.iter().next().unwrap().terms, original);
        assert!(groups.register(&e, 2..3, (0, 0), &original, mode).is_err());
        assert!(groups.register(&e, 3..4, (0, 0), &[1.], mode).is_err());
        assert!(groups.register(&e, 0..3, (1, 0), &original, mode).is_err());

        groups.seal(&e).unwrap();
        let expected = mode.coefficient(&original, 0.25);
        let entries = iter_dynamic_terms(&e, &groups, 0.25)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(entries, vec![(0, 0, expected)]);
        assert_eq!(entries[0].2.to_bits(), expected.to_bits());
        assert!(groups.validate_operator(&e.clone()).is_err());

        let mut copy = groups.clone();
        let cloned = e.clone();
        copy.rebind(&cloned);
        assert_eq!(
            iter_dynamic_terms(&cloned, &copy, 0.25)
                .unwrap()
                .collect::<Vec<_>>(),
            entries
        );

        e.add(1, 1, 1.).unwrap();
        assert!(groups.validate_operator(&e).is_err());
        assert!(iter_dynamic_terms(&e, &groups, 0.25).is_err());
        groups.seal(&e).unwrap();
        assert_eq!(iter_dynamic_terms(&e, &groups, 0.25).unwrap().count(), 2);

        let capacity = groups.capacities();
        let ptr = groups.terms.as_ptr();
        groups.clear();
        e.clear();
        assert_eq!(groups.capacities(), capacity);
        for &term in &original {
            e.add(0, 0, mode.stamp(term)).unwrap();
        }
        groups.register(&e, 0..3, (0, 0), &original, mode).unwrap();
        groups.seal(&e).unwrap();
        assert_eq!(ptr, groups.terms.as_ptr());

        let mut orphan = groups.clone();
        orphan.groups[0].stamps = 1..3;
        orphan.sealed_generation = None;
        assert!(orphan.validate_operator(&e).is_err());
        let mut orphan = groups.clone();
        orphan.groups[0].terms = 1..4;
        orphan.sealed_generation = None;
        assert!(orphan.validate_operator(&e).is_err());

        let mut prescaled = DynamicGroups::new();
        let mut other = SparseMatrix::new(1, 1);
        for v in [1., 0., 2.] {
            other.add(0, 0, v).unwrap();
        }
        prescaled
            .register(
                &other,
                0..2,
                (0, 0),
                &[1., 0., 2.],
                DynamicMode::Prescaled { scale: 1. },
            )
            .unwrap();
        prescaled.seal(&other).unwrap();
        let group = prescaled.iter().next().unwrap();
        assert!(prescaled.pencil_value(&group).is_err());
        assert_eq!(
            iter_dynamic_terms(&other, &prescaled, 17.)
                .unwrap()
                .collect::<Vec<_>>(),
            vec![(0, 0, 3.)]
        );
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
        let mut groups = DynamicGroups::new();
        let terms = [1., -2., 0.25];
        let start = e.triplets().len();
        for term in terms {
            e.add(0, 0, 3. * term / 2.).unwrap();
        }
        groups
            .register(
                &e,
                start..e.triplets().len(),
                (0, 0),
                &terms,
                DynamicMode::Scaled {
                    scale: 3.,
                    divisor: 2.,
                },
            )
            .unwrap();
        groups.seal(&e).unwrap();
        let omega = 0.25;
        let mut expected = std::collections::BTreeMap::new();
        let real = a
            .triplets()
            .iter()
            .map(|t| (t.row, t.col, Complex::real(t.value)));
        let imag = iter_dynamic_terms(&e, &groups, omega)
            .unwrap()
            .map(|(r, c, v)| (r, c, Complex::imaginary(v)));
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
        let mut groups = DynamicGroups::new();
        // These inputs are already omega-scaled; their C-ordered sum,
        // including cancellation, is one matrix element.
        groups
            .register(
                &e,
                0..3,
                (0, 1),
                &[1e16, -1e16, 1.],
                DynamicMode::Prescaled { scale: 1. },
            )
            .unwrap();
        groups
            .register(
                &e,
                3..4,
                (1, 1),
                &[0.5],
                DynamicMode::Scaled {
                    scale: 1.,
                    divisor: 1.,
                },
            )
            .unwrap();
        groups.seal(&e).unwrap();
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
