//! Triplet storage for the sparse MNA matrix.
//!
//! ngspice's MNA matrix is built by every device adding small contributions at
//! scattered `(row, column)` positions — see `cktload.c` and the `*Load`
//! functions throughout `src/spicelib/devices/`. SPARSE 1.3
//! (`src/maths/sparse/spbuild.c`) and KLU each have their own way of accepting
//! those contributions; a coordinate list is the format both can be fed from,
//! and the format the port uses to accumulate a stamp before assembling.
//!
//! Factorization returns owned SPARSE 1.3 factors, independent of stamping storage.
//!
//! A matrix can also be *bound* to a [`Pattern`]: the fixed set of element
//! positions a circuit stamps, each with an [`ElementId`] assigned in the order
//! the devices first stamped it (ngspice's `TSTALLOC`/`spGetElement` order).
//! A bound matrix accumulates `values[id] += v` instead of appending triplets,
//! so a Newton iteration neither allocates nor sorts (ngspice: `spClear` then
//! `*ptr += v`). Contributions outside the pattern go to an overflow list, from
//! which the owner extends the pattern (SPARSE: a new element forces a reorder).

use std::borrow::Cow;
use std::cell::OnceCell;
use std::sync::Arc;

use crate::primitives::{Complex, Real, SpiceError, SpiceResult};
use petgraph::graphmap::UnGraphMap;

use crate::maths::dense::Vector;
use crate::maths::linear::SparseLu;

/// One `(row, column, value)` entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triplet {
    /// Row index.
    pub row: usize,
    /// Column index.
    pub col: usize,
    /// The contribution.
    pub value: Real,
}

/// Handle of one matrix element in a [`Pattern`] (SPARSE's element pointer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ElementId(u32);

impl ElementId {
    /// Position of the element in its pattern's creation order.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Immutable row index and stable creation-order positions.
#[derive(Debug, Clone)]
struct PatternIndex {
    n: usize,
    positions: Vec<(u32, u32)>,
    row_start: Vec<u32>,
    row_cols: Vec<u32>,
    row_ids: Vec<u32>,
}

/// The element positions of a square matrix, numbered in creation order, with
/// a per-row sorted index for `(row, col) -> ElementId` lookup.
///
/// The pattern also keeps the SPARSE pivot order last used for it, so every
/// solve of a circuit refactors one ordered matrix as ngspice does, across
/// Newton solves and timepoints.
#[derive(Debug)]
pub struct Pattern {
    index: PatternIndex,
    ordering: std::sync::Mutex<OrderingState>,
    /// The `(row, col, id)` sequence of the last load, replayed by bound
    /// matrices to skip lookups (ngspice's per-device element pointers).
    trace: std::sync::Mutex<Arc<Trace>>,
    /// Numeric factor/solve scratch shared across successive AC snapshots, independently of
    /// pivot-order transactions. Concurrent solves may use local scratch.
    complex_workspace: std::sync::Mutex<Vec<Complex>>,
}

/// The retained SPARSE frame and its AC reorder flag form one transaction.
/// A shared pattern cannot lose its pivot history to a concurrent solve.
#[derive(Debug)]
pub(super) struct OrderingState {
    pub(super) frame: Option<crate::maths::linear::CachedFrame>,
    pub(super) ac_reorder: bool,
    /// Reused real factor/triangular-solve scratch, owned by the circuit
    /// transaction rather than by an immutable factor snapshot.
    pub(super) solve_workspace: Vec<Real>,
}

impl Default for OrderingState {
    fn default() -> Self {
        Self {
            frame: None,
            ac_reorder: true,
            solve_workspace: Vec::new(),
        }
    }
}

/// One recorded stamp: `(row, col, element)`.
type TraceEntry = (u32, u32, u32);

/// A load's stamp sequence, split per device: device `k` stamped
/// `entries[starts[k]..starts[k + 1]]`.
#[derive(Debug, Clone, Default, PartialEq)]
struct Trace {
    entries: Vec<TraceEntry>,
    starts: Vec<usize>,
}

/// Verified fixed-stamp batches for a circuit's sparse matrix.
///
/// One circuit's verified fixed stamp batches ([`SparseMatrix::add_fixed`]):
/// the exact trace they were verified against and the ordered batches within
/// each device. A device may interleave several fixed-size stamp groups with
/// ordinary stamps. No dynamic matrix lookup is needed for verified groups.
/// The circuit keeps this metadata between loads, without changing device
/// order, matrix-entry accumulation order, or trial-state transactions.
#[derive(Debug, Clone, Default)]
pub struct FixedStamps {
    trace: Option<Arc<Trace>>,
    batches: Vec<Vec<Option<VerifiedBatch>>>,
}

impl FixedStamps {
    /// Drops every verification, for a circuit whose devices or numbering
    /// changed.
    pub fn forget(&mut self) {
        self.trace = None;
        self.batches.clear();
    }
}

impl PartialEq for FixedStamps {
    /// A cache of how a matrix may be stamped, not part of its value.
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// One segment of the immutable per-device trace, with entries omitted at
/// ground represented by zero mask bits. `start..end` may cover a strict
/// subset of the device: preceding or following nonfixed stamps stay ordered.
#[derive(Debug, Clone, Copy)]
struct VerifiedBatch {
    start: usize,
    end: usize,
    mask: u32,
    len: u8,
}

const FIXED_MAX: usize = u32::BITS as usize;

impl Clone for Pattern {
    /// A copy without the cached pivot order.
    fn clone(&self) -> Self {
        Self::with_index(self.index.clone())
    }
}

impl PartialEq for Pattern {
    fn eq(&self, other: &Self) -> bool {
        self.index.n == other.index.n && self.index.positions == other.index.positions
    }
}

impl Pattern {
    /// A pattern in creation order; repeated positions keep their first id.
    /// # Errors
    /// Unsupported dimensions, coordinates, compact handle count or allocation.
    pub fn new(n: usize, positions: impl IntoIterator<Item = (usize, usize)>) -> SpiceResult<Self> {
        pattern_rows(n)?;
        let mut seen = std::collections::HashSet::new();
        let mut unique = Vec::new();
        for (row, col) in positions {
            let position = compact_position(n, row, col)?;
            if insert_position(&mut seen, position)? {
                push_compact(&mut unique, position)?;
            }
        }
        Self::index(n, unique)
    }

    fn index(n: usize, positions: Vec<(u32, u32)>) -> SpiceResult<Self> {
        let len = compact_count(positions.len())?;
        let rows = pattern_rows(n)?;
        let mut row_start = crate::primitives::filled_vec(rows, 0_u32, "sparse pattern")?;
        for &(row, _) in &positions {
            row_start[row as usize + 1] += 1;
        }
        for row in 0..n {
            row_start[row + 1] += row_start[row];
        }
        let row_ids = if positions.windows(2).all(|pair| pair[0] < pair[1]) {
            // Already row-major and unique: element handles are the index.
            let mut order = Vec::new();
            reserve(&mut order, positions.len())?;
            order.extend(0..len);
            order
        } else {
            // Bucket handles by row in linear time, then sort only within
            // rows. The original positions and handle numbering never move.
            let mut order =
                crate::primitives::filled_vec(positions.len(), 0_u32, "sparse pattern")?;
            for (id, &(row, _)) in (0..len).zip(&positions) {
                let cursor = &mut row_start[row as usize];
                order[*cursor as usize] = id;
                *cursor += 1;
            }
            // Each write cursor is now the following row's start. Recover
            // the prefix offsets in place rather than allocate another n-vector.
            row_start.copy_within(..n, 1);
            row_start[0] = 0;
            for bounds in row_start.windows(2) {
                order[bounds[0] as usize..bounds[1] as usize]
                    .sort_unstable_by_key(|&id| positions[id as usize].1);
            }
            order
        };
        let mut row_cols = Vec::new();
        reserve(&mut row_cols, row_ids.len())?;
        row_cols.extend(row_ids.iter().map(|&id| positions[id as usize].1));
        Ok(Self::with_index(PatternIndex {
            n,
            positions,
            row_start,
            row_cols,
            row_ids,
        }))
    }

    fn with_index(index: PatternIndex) -> Self {
        Self {
            index,
            ordering: std::sync::Mutex::new(OrderingState::default()),
            trace: std::sync::Mutex::new(Arc::new(Trace::default())),
            complex_workspace: std::sync::Mutex::new(Vec::new()),
        }
    }

    fn trace(&self) -> Arc<Trace> {
        self.trace.lock().map_or_default(|trace| Arc::clone(&trace))
    }

    fn store_trace(&self, trace: Arc<Trace>) {
        if let Ok(mut slot) = self.trace.lock() {
            *slot = trace;
        }
    }

    /// Reuse the circuit's complex factor/solve scratch without serializing
    /// independent factor snapshots. A busy or poisoned slot is optional:
    /// the solve still runs with its own scratch in exactly the same order.
    pub(super) fn with_complex_workspace<R>(&self, f: impl FnOnce(&mut Vec<Complex>) -> R) -> R {
        if let Ok(mut workspace) = self.complex_workspace.try_lock() {
            let result = f(&mut workspace);
            drop(workspace);
            result
        } else {
            f(&mut Vec::new())
        }
    }

    /// Serializes factor-cache updates, including scalar changes and AC
    /// reordering. A poisoned cache is discarded rather than reused.
    pub(super) fn with_ordering<R>(&self, f: impl FnOnce(&mut OrderingState) -> R) -> R {
        let mut state = match self.ordering.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                *state = OrderingState::default();
                self.ordering.clear_poison();
                state
            }
        };
        let result = f(&mut state);
        drop(state);
        result
    }

    /// Extend in creation order while retaining existing element ids.
    /// # Errors
    /// Invalid coordinates, unsupported compact handle count or allocation.
    pub fn extended(&self, extra: impl IntoIterator<Item = (usize, usize)>) -> SpiceResult<Self> {
        let mut positions = copy_entries(&self.index.positions)?;
        let mut appended = std::collections::HashSet::new();
        for (row, col) in extra {
            let position = compact_position(self.index.n, row, col)?;
            if self.find(row, col).is_none() && insert_position(&mut appended, position)? {
                push_compact(&mut positions, position)?;
            }
        }
        Self::index(self.index.n, positions)
    }

    /// Matrix positions in stable creation order.
    pub(crate) fn positions(&self) -> impl ExactSizeIterator<Item = (usize, usize)> + '_ {
        self.index
            .positions
            .iter()
            .map(|&(row, col)| (row as usize, col as usize))
    }

    /// Order of the matrix.
    #[must_use]
    pub const fn order(&self) -> usize {
        self.index.n
    }

    /// Number of elements.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.index.positions.len()
    }

    /// True when the pattern has no elements.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.index.positions.is_empty()
    }

    /// `(row, col)` of an element.
    #[must_use]
    pub fn position(&self, id: ElementId) -> Option<(usize, usize)> {
        self.index
            .positions
            .get(id.index())
            .map(|&(row, col)| (row as usize, col as usize))
    }

    /// The element at `(row, col)`, if the pattern has one.
    #[must_use]
    pub fn find(&self, row: usize, col: usize) -> Option<ElementId> {
        if row >= self.index.n {
            return None;
        }
        let (start, end) = self.row_bounds(row);
        let col = u32::try_from(col).ok()?;
        self.index.row_cols[start..end]
            .binary_search(&col)
            .ok()
            .map(|offset| ElementId(self.index.row_ids[start + offset]))
    }

    /// Index interval of one row within the row-major handle array.
    fn row_bounds(&self, row: usize) -> (usize, usize) {
        (
            self.index.row_start[row] as usize,
            self.index.row_start[row + 1] as usize,
        )
    }

    /// Validated element handles and coordinates in row-major order.
    fn sorted(&self) -> impl Iterator<Item = (ElementId, (usize, usize))> + '_ {
        self.index.row_ids.iter().map(|&id| {
            let (row, col) = self.index.positions[id as usize];
            (ElementId(id), (row as usize, col as usize))
        })
    }
}

fn pattern_error(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "sparse pattern".to_owned(),
        message: message.into(),
    }
}

fn pattern_rows(n: usize) -> SpiceResult<usize> {
    compact_count(n)?;
    let rows = n
        .checked_add(1)
        .ok_or_else(|| pattern_error("row offsets overflow"))?;
    std::alloc::Layout::array::<u32>(rows)
        .map_err(|error| pattern_error(format!("row offset capacity: {error}")))?;
    Ok(rows)
}

fn insert_position(
    seen: &mut std::collections::HashSet<(u32, u32)>,
    position: (u32, u32),
) -> SpiceResult<bool> {
    // A duplicate does not require storage; on the usual path there is just
    // one hash lookup, as in the original infallible builder.
    if seen.len() == seen.capacity() {
        if seen.contains(&position) {
            return Ok(false);
        }
        seen.try_reserve(1).map_err(|error| {
            pattern_error(format!("cannot reserve sparse position index: {error}"))
        })?;
    }
    Ok(seen.insert(position))
}

fn reserve<T>(values: &mut Vec<T>, additional: usize) -> SpiceResult<()> {
    values.try_reserve(additional).map_err(|error| {
        pattern_error(format!(
            "cannot reserve {additional} sparse entries: {error}"
        ))
    })
}

fn push_compact<T>(values: &mut Vec<T>, value: T) -> SpiceResult<()> {
    let next = values
        .len()
        .checked_add(1)
        .ok_or_else(|| pattern_error("compact entry count overflows"))?;
    compact_count(next)?;
    if values.len() == values.capacity() {
        reserve(values, 1)?;
    }
    values.push(value);
    Ok(())
}

fn copy_entries<T: Copy>(entries: &[T]) -> SpiceResult<Vec<T>> {
    let mut result = Vec::new();
    reserve(&mut result, entries.len())?;
    result.extend_from_slice(entries);
    Ok(result)
}

fn offsets_len(devices: usize) -> SpiceResult<usize> {
    let count = devices
        .checked_add(1)
        .ok_or_else(|| pattern_error("trace device offset count overflows"))?;
    compact_count(count)?;
    Ok(count)
}

fn compact_count(value: usize) -> SpiceResult<u32> {
    u32::try_from(value).map_err(|_| pattern_error("compact pattern count exceeds u32"))
}

fn compact_position(n: usize, row: usize, col: usize) -> SpiceResult<(u32, u32)> {
    if row >= n || col >= n {
        return Err(pattern_error(format!(
            "position ({row}, {col}) outside {n}x{n} pattern"
        )));
    }
    Ok((compact_count(row)?, compact_count(col)?))
}

/// Values of a matrix bound to a [`Pattern`].
#[derive(Debug)]
struct Bound {
    pattern: Arc<Pattern>,
    values: Vec<Real>,
    /// The folded triplet view, built on demand for triplet consumers.
    view: OnceCell<Vec<Triplet>>,
    /// The pattern's recorded stamp sequence, the device now stamping and the
    /// part of its segment still to match.
    trace: Arc<Trace>,
    device: usize,
    active: bool,
    /// The next fixed-group call within this device in stamping order.
    batch_index: usize,
    /// Out-of-order device positions cannot form one contiguous trace.
    traceable: bool,
    /// Where the present device's segment of `trace` starts.
    start: usize,
    cursor: usize,
    end: usize,
    /// This load's own sequence, once any device diverged from `trace`.
    recording: Option<Trace>,
    /// The circuit's verified fixed batches, while it loads this matrix.
    fixed: FixedStamps,
}

impl Clone for Bound {
    fn clone(&self) -> Self {
        Self {
            pattern: Arc::clone(&self.pattern),
            values: self.values.clone(),
            // A read-through projection is not part of the matrix state.
            view: OnceCell::new(),
            trace: Arc::clone(&self.trace),
            device: self.device,
            active: self.active,
            batch_index: self.batch_index,
            traceable: self.traceable,
            start: self.start,
            cursor: self.cursor,
            end: self.end,
            recording: self.recording.clone(),
            fixed: self.fixed.clone(),
        }
    }
}

impl PartialEq for Bound {
    fn eq(&self, other: &Self) -> bool {
        self.pattern == other.pattern && self.values == other.values
    }
}

impl Bound {
    /// Start recording only when the previously matched prefix differs.
    /// Completed segments before this device remain exact; a shortened
    /// segment contributes only its matched prefix, not its old suffix.
    fn start_recording(&mut self) -> SpiceResult<()> {
        if self.recording.is_some() {
            return Ok(());
        }
        let offsets = offsets_len(self.device)?;
        let mut starts = Vec::new();
        reserve(&mut starts, offsets)?;
        starts.extend(self.trace.starts.iter().copied().take(offsets));
        starts.resize(offsets, self.cursor);
        self.recording = Some(Trace {
            entries: copy_entries(&self.trace.entries[..self.cursor])?,
            starts,
        });
        Ok(())
    }

    /// Starts device `index`'s stamps: replay resumes at its own segment.
    #[inline]
    fn begin_device(&mut self, index: usize) -> SpiceResult<()> {
        // Sequential, completely replayed devices are overwhelmingly the
        // common load path. The previous end is the next start, so read
        // only the new end, and leave all recording transitions cold.
        // The slow path checked room for both sentinel offsets when this
        // device was installed; the adjacent index arithmetic cannot overflow.
        if self.active
            && self.cursor == self.end
            && index == self.device + 1
            && self.recording.is_none()
            && self.traceable
            && let Some(&end) = self.trace.starts.get(index + 1)
        {
            self.device = index;
            self.batch_index = 0;
            self.start = self.end;
            self.end = end;
            return Ok(());
        }
        self.begin_device_slow(index)
    }

    #[cold]
    fn begin_device_slow(&mut self, index: usize) -> SpiceResult<()> {
        let next = offsets_len(index)?;
        offsets_len(next)?;
        if !self.traceable {
            return Ok(());
        }
        if self.active {
            if index <= self.device {
                // Numerical stamping still works; this load cannot publish
                // a device-indexed trace or use fixed-batch verification.
                self.traceable = false;
                self.cursor = 0;
                self.end = 0;
                self.recording = None;
                return Ok(());
            }
            if self.recording.is_none() && (self.cursor != self.end || index != self.device + 1) {
                self.start_recording()?;
            }
        } else if index != 0 {
            self.recording = Some(Trace::default());
        }
        self.device = index;
        self.batch_index = 0;
        self.active = true;
        if let Some(recording) = &mut self.recording {
            let additional = next.saturating_sub(recording.starts.len());
            reserve(&mut recording.starts, additional)?;
            recording.starts.resize(next, recording.entries.len());
        }
        let past = self.trace.entries.len();
        let start = self.trace.starts.get(index).copied().unwrap_or(past);
        let end = self.trace.starts.get(index + 1).copied().unwrap_or(past);
        self.start = start;
        self.cursor = start;
        self.end = end;
        Ok(())
    }

    fn reset_trace(&mut self) {
        let trace = self.pattern.trace();
        if !Arc::ptr_eq(&trace, &self.trace) {
            self.fixed.forget();
        }
        self.trace = trace;
        self.device = 0;
        self.batch_index = 0;
        self.active = false;
        self.traceable = true;
        self.start = 0;
        self.cursor = 0;
        self.end = 0;
        self.recording = None;
    }

    fn finish_trace(&mut self) -> SpiceResult<()> {
        if !self.active || !self.traceable {
            return Ok(());
        }
        if self.recording.is_none()
            && (self.cursor != self.end || self.trace.starts.len() != self.device + 2)
        {
            self.start_recording()?;
        }
        if let Some(mut recording) = self.recording.take() {
            push_compact(&mut recording.starts, recording.entries.len())?;
            let trace = Arc::new(recording);
            self.pattern.store_trace(Arc::clone(&trace));
            self.trace = trace;
            self.fixed.forget();
        }
        self.active = false;
        self.start = 0;
        self.cursor = 0;
        self.end = 0;
        Ok(())
    }

    /// The device's next traced element when it is at `(row, col)` and this
    /// load is not recording its own trace (otherwise [`Self::element`]).
    #[inline]
    fn replay(&mut self, row: usize, col: usize) -> Option<ElementId> {
        if self.cursor >= self.end || self.recording.is_some() {
            return None;
        }
        let (r, c, id) = *self.trace.entries.get(self.cursor)?;
        if r as usize == row && c as usize == col {
            self.cursor += 1;
            Some(ElementId(id))
        } else {
            None
        }
    }

    /// The element at `(row, col)`: the device's next traced stamp when it
    /// matches (two comparisons), otherwise a pattern lookup that switches
    /// this load to recording its own sequence.
    #[inline]
    fn element(&mut self, row: usize, col: usize) -> SpiceResult<Option<ElementId>> {
        if !self.traceable {
            return Ok(self.pattern.find(row, col));
        }
        if !self.active {
            self.begin_device(0)?;
        }
        let (r, c) = compact_position(self.pattern.order(), row, col)?;
        if self.cursor < self.end {
            let (tr, tc, id) = self.trace.entries[self.cursor];
            if tr == r && tc == c {
                self.cursor += 1;
                if let Some(recording) = &mut self.recording {
                    push_compact(&mut recording.entries, (r, c, id))?;
                }
                return Ok(Some(ElementId(id)));
            }
        }
        self.start_recording()?;
        // The rest of this device is looked up.
        self.end = 0;
        let Some(id) = self.pattern.find(row, col) else {
            return Ok(None);
        };
        if let Some(recording) = &mut self.recording {
            push_compact(&mut recording.entries, (r, c, id.0))?;
        }
        Ok(Some(id))
    }
}

/// Stable in-place folding over an arbitrary stamp sequence, including
/// cancellation and accumulation in original equal-coordinate order.
fn fold_triplets(triplets: &mut Vec<Triplet>) {
    triplets.sort_by_key(|triplet| (triplet.row, triplet.col));
    // Stable sorting preserves the order of equal stamps. Compact in
    // place, retaining the first value and adding subsequent values in
    // exactly the same order, even through intermediate cancellations.
    let mut len = 0;
    for read in 0..triplets.len() {
        let triplet = triplets[read];
        if len > 0 {
            let last = &mut triplets[len - 1];
            if last.row == triplet.row && last.col == triplet.col {
                last.value += triplet.value;
                continue;
            }
        }
        triplets[len] = triplet;
        len += 1;
    }
    triplets.truncate(len);
    triplets.retain(|triplet| triplet.value != 0.0);
}

/// A sparse matrix as a list of contributions.
///
/// Duplicate positions are allowed and are summed by [`SparseMatrix::get`] and by
/// [`SparseMatrix::fold_duplicates`]. Devices rely on this: a resistor between
/// two non-ground nodes stamps four positions, and several devices may stamp the
/// same position.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseMatrix {
    rows: usize,
    cols: usize,
    /// Every contribution (unbound), or the overflow outside the pattern.
    triplets: Vec<Triplet>,
    folded: bool,
    bound: Option<Bound>,
    /// Thresholds used when this matrix is (re)ordered.
    pivot: crate::maths::PivotTolerances,
    /// Per row, the circuit unknown's name (C `NODENAME`), used by solver
    /// diagnostics; `None` for a standalone matrix.
    names: Option<crate::maths::UnknownNames>,
}

impl SparseMatrix {
    /// An empty `rows` by `cols` matrix.
    #[must_use]
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            triplets: Vec::new(),
            folded: false,
            bound: None,
            pivot: crate::maths::PivotTolerances::DEFAULT,
            names: None,
        }
    }

    /// Construct an already folded stream from a trusted assembly pass.
    /// Its coordinates are sorted and unique, as with `fold_duplicates`.
    pub(super) fn from_assembled(rows: usize, cols: usize, mut triplets: Vec<Triplet>) -> Self {
        debug_assert!(triplets.iter().all(|t| t.row < rows && t.col < cols));
        debug_assert!(
            triplets
                .windows(2)
                .all(|w| (w[0].row, w[0].col) < (w[1].row, w[1].col))
        );
        // The ordinary stamp path drops zeros even when they were present
        // in a previous folded matrix whose row was cleared.
        triplets.retain(|t| t.value != 0.);
        Self {
            triplets,
            folded: true,
            ..Self::new(rows, cols)
        }
    }

    /// An all-zero matrix bound to `pattern`, accumulating by element.
    ///
    /// # Errors
    /// Coefficient storage cannot be allocated.
    pub fn bound(pattern: Arc<Pattern>) -> SpiceResult<Self> {
        let n = pattern.order();
        Ok(Self {
            rows: n,
            cols: n,
            triplets: Vec::new(),
            folded: true,
            bound: Some(Bound {
                values: crate::primitives::filled_vec(pattern.len(), 0.0, "bound sparse matrix")?,
                trace: pattern.trace(),
                device: 0,
                active: false,
                batch_index: 0,
                traceable: true,
                start: 0,
                cursor: 0,
                end: 0,
                recording: None,
                fixed: FixedStamps::default(),
                pattern,
                view: OnceCell::new(),
            }),
            pivot: crate::maths::PivotTolerances::DEFAULT,
            names: None,
        })
    }

    /// Pivot thresholds used when this matrix is ordered for factorization
    /// (default [`crate::maths::PivotTolerances::DEFAULT`]).
    #[must_use]
    pub const fn pivot_tolerances(&self) -> crate::maths::PivotTolerances {
        self.pivot
    }

    /// Replaces the pivot thresholds (`.options pivtol`/`pivrel`); validated
    /// by the caller (see [`crate::maths::PivotTolerances::validate`]).
    pub const fn set_pivot_tolerances(&mut self, pivot: crate::maths::PivotTolerances) {
        self.pivot = pivot;
    }

    /// The circuit unknowns' names by row, when set: solver warnings and
    /// singular-matrix errors name nodes as ngspice's `NIiter` does
    /// (`singular matrix:  check node ...`) instead of row numbers.
    #[must_use]
    pub const fn unknown_names(&self) -> Option<&crate::maths::UnknownNames> {
        self.names.as_ref()
    }

    /// Sets (or clears) the per-row unknown names used in diagnostics.
    pub fn set_unknown_names(&mut self, names: Option<crate::maths::UnknownNames>) {
        self.names = names;
    }

    /// The pattern and per-element values when bound with no overflow: the
    /// fast path for factorization.
    #[must_use]
    pub fn bound_values(&self) -> Option<(&Arc<Pattern>, &[Real])> {
        match &self.bound {
            Some(bound) if self.triplets.is_empty() => Some((&bound.pattern, &bound.values)),
            _ => None,
        }
    }

    /// The bound pattern and, in stamp order, the positions outside it.
    #[must_use]
    pub fn overflow(&self) -> Option<(&Arc<Pattern>, impl Iterator<Item = (usize, usize)> + '_)> {
        self.bound
            .as_ref()
            .map(|bound| (&bound.pattern, self.triplets.iter().map(|t| (t.row, t.col))))
    }

    /// Marks the start of device `index`'s stamps within a load, so a
    /// diverging device only affects its own trace segment.
    ///
    /// # Errors
    /// The trace device offset or allocation exceeds the supported range.
    #[inline]
    pub fn begin_device(&mut self, index: usize) -> SpiceResult<()> {
        if let Some(bound) = &mut self.bound {
            bound.begin_device(index)?;
        }
        Ok(())
    }

    /// Lends a circuit's verified fixed batches to this bound matrix for one
    /// load of `devices` devices; they are reset unless they were verified
    /// against this matrix's trace. An unbound matrix hands them back
    /// untouched.
    ///
    /// # Errors
    /// The fixed batch storage or device count exceeds the supported range.
    pub fn lend_fixed(
        &mut self,
        mut fixed: FixedStamps,
        devices: usize,
    ) -> SpiceResult<Option<FixedStamps>> {
        let Some(bound) = &mut self.bound else {
            return Ok(Some(fixed));
        };
        offsets_len(devices)?;
        let same = fixed
            .trace
            .as_ref()
            .is_some_and(|trace| Arc::ptr_eq(trace, &bound.trace));
        if !same || fixed.batches.len() != devices {
            fixed.trace = Some(Arc::clone(&bound.trace));
            fixed.batches.clear();
            reserve(&mut fixed.batches, devices)?;
            fixed.batches.resize_with(devices, Vec::new);
        }
        bound.fixed = fixed;
        Ok(None)
    }

    /// Returns the batches lent by [`Self::lend_fixed`], with those this load
    /// verified.
    pub fn return_fixed(&mut self) -> FixedStamps {
        self.bound
            .as_mut()
            .map_or_default(|bound| std::mem::take(&mut bound.fixed))
    }

    /// Adds a device's fixed batch by element when an earlier load verified
    /// it against this trace: `values[k]` for each entry `k` stamped then, in
    /// order, as [`Self::add`] would. Returns `false` (adding nothing) when
    /// the batch must be stamped entry by entry instead.
    ///
    /// The caller guarantees that the batch's positions are the same on every
    /// load (C's `TSTALLOC` pointers); only its values change.
    #[inline]
    pub fn add_fixed<const N: usize>(&mut self, values: &[Real; N]) -> bool {
        let Some(bound) = &mut self.bound else {
            return false;
        };
        let Some(batch) = bound
            .fixed
            .batches
            .get(bound.device)
            .and_then(|groups| groups.get(bound.batch_index))
            .copied()
            .flatten()
        else {
            return false;
        };
        if N != usize::from(batch.len)
            || !bound.traceable
            || !bound.active
            || bound.recording.is_some()
            || bound.cursor != batch.start
            || batch.start < bound.start
            || batch.end > bound.end
        {
            return false;
        }
        let Some(traced) = bound.trace.entries.get(batch.start..batch.end) else {
            return false;
        };
        // A verified group's mask is tied to the immutable trace. Other
        // groups and ordinary stamps may be interleaved in one device load.
        let mut mask = batch.mask;
        debug_assert_eq!(traced.len(), mask.count_ones() as usize);
        bound.view.take();
        if traced.len() == N {
            for (&(_, _, id), &value) in traced.iter().zip(values) {
                bound.values[id as usize] += value;
            }
        } else {
            for &(_, _, id) in traced {
                let k = mask.trailing_zeros() as usize;
                bound.values[id as usize] += values[k];
                mask &= mask - 1;
            }
        }
        bound.cursor = batch.end;
        bound.batch_index += 1;
        true
    }

    /// The trace position a fixed batch starts at, for
    /// [`Self::verify_fixed`].
    #[must_use]
    pub fn fixed_start(&self) -> Option<usize> {
        self.bound
            .as_ref()
            .filter(|bound| bound.active && bound.traceable)
            .map(|bound| bound.cursor)
    }

    /// After a fixed group stamped entry by entry from `start`, verify only
    /// that group's positions against the immutable trace. Other groups and
    /// ordinary stamps may precede or follow it in the same device. Invalid
    /// lengths, masks and changing traces can never become verified batches.
    ///
    /// # Errors
    /// A batch ordinal cannot be stored within the bounded preparation cache.
    pub fn verify_fixed(&mut self, start: Option<usize>, len: usize, mask: u64) -> SpiceResult<()> {
        let Some(bound) = &mut self.bound else {
            return Ok(());
        };
        let batch_index = bound.batch_index;
        bound.batch_index += 1;
        let Some(groups) = bound.fixed.batches.get_mut(bound.device) else {
            return Ok(());
        };
        let verified = if let Some(start) = start
            && len <= FIXED_MAX
            && mask >> len == 0
            && bound.traceable
            && bound.active
            && bound.recording.is_none()
            && start >= bound.start
            && bound.cursor <= bound.end
            && bound.cursor.checked_sub(start) == Some(mask.count_ones() as usize)
            && let (Ok(len), Ok(mask)) = (u8::try_from(len), u32::try_from(mask))
        {
            Some(VerifiedBatch {
                start,
                end: bound.cursor,
                mask,
                len,
            })
        } else {
            None
        };
        if groups.len() == batch_index {
            reserve(groups, 1)?;
            groups.push(verified);
        } else if let Some(slot) = groups.get_mut(batch_index) {
            *slot = verified;
        }
        Ok(())
    }

    /// Saves this load's stamp sequence on the pattern when it differed from
    /// the recorded one (call once the load is complete). The recording's
    /// owned buffers move into the published trace without a second copy.
    ///
    /// # Errors
    /// Trace storage cannot be extended or exceeds the compact range.
    pub fn record_trace(&mut self) -> SpiceResult<()> {
        if !self.triplets.is_empty() {
            return Ok(());
        }
        if let Some(bound) = &mut self.bound {
            bound.finish_trace()?;
        }
        Ok(())
    }

    /// Adds to an element by handle (`*ptr += value`).
    ///
    /// # Errors
    /// No bound pattern, or the handle lies outside it.
    pub fn add_element(&mut self, element: ElementId, value: Real) -> SpiceResult<()> {
        let bound = self
            .bound
            .as_mut()
            .ok_or_else(|| pattern_error("element requires a bound matrix"))?;
        let slot = bound
            .values
            .get_mut(element.index())
            .ok_or_else(|| pattern_error("element handle outside bound pattern"))?;
        *slot += value;
        bound.view.take();
        Ok(())
    }

    /// Zeroes every element of `row` whose column `keep` rejects, keeping it
    /// in the structure (C `ZeroNoncurRow` in `cktload.c` stores `0.0`
    /// through `SMPfindElt`). Returns whether a kept column holds an element.
    /// The predicate is a column selection, independent of traversal order.
    pub fn zero_row(&mut self, row: usize, keep: impl Fn(usize) -> bool) -> bool {
        let mut kept = false;
        if let Some(bound) = &mut self.bound
            && row < bound.pattern.order()
        {
            bound.view.take();
            let (start, end) = bound.pattern.row_bounds(row);
            for (&id, &col) in bound.pattern.index.row_ids[start..end]
                .iter()
                .zip(&bound.pattern.index.row_cols[start..end])
            {
                if keep(col as usize) {
                    kept = true;
                } else {
                    bound.values[id as usize] = 0.0;
                }
            }
        }
        let triplets = if self.folded {
            let start = self.triplets.partition_point(|t| t.row < row);
            let end = start + self.triplets[start..].partition_point(|t| t.row == row);
            &mut self.triplets[start..end]
        } else {
            &mut self.triplets[..]
        };
        for triplet in triplets.iter_mut().filter(|t| t.row == row) {
            if keep(triplet.col) {
                kept = true;
            } else {
                triplet.value = 0.0;
            }
        }
        kept
    }

    /// Overwrites the value at a position (C `*ptr = value` on an element),
    /// creating the position if it is not stored.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when the position is out of range.
    pub fn set(&mut self, row: usize, col: usize, value: Real) -> SpiceResult<()> {
        if let Some(bound) = &mut self.bound
            && let Some(id) = bound.pattern.find(row, col)
        {
            // Overflow coordinates are disjoint from the pattern.
            bound.view.take();
            bound.values[id.index()] = value;
            return Ok(());
        }
        if self.folded {
            if let Ok(i) = self
                .triplets
                .binary_search_by_key(&(row, col), |t| (t.row, t.col))
            {
                self.triplets.remove(i);
            }
        } else {
            self.triplets.retain(|t| t.row != row || t.col != col);
        }
        // Preserve the unbound stamp sequence: replacement is appended,
        // rather than moved into the folded row-major run.
        self.add(row, col, value)
    }

    /// Number of rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Number of stored contributions, including duplicates.
    #[must_use]
    pub fn nnz(&self) -> usize {
        match &self.bound {
            Some(bound) if self.triplets.is_empty() => {
                bound.values.iter().filter(|&&value| value != 0.0).count()
            }
            Some(_) => self.triplets().len(),
            None => self.triplets.len(),
        }
    }

    /// True when nothing has been stamped.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match &self.bound {
            Some(bound) if self.triplets.is_empty() => {
                bound.values.iter().all(|&value| value == 0.0)
            }
            Some(_) => self.triplets().is_empty(),
            None => self.triplets.is_empty(),
        }
    }

    /// True when duplicate positions have been folded away.
    #[must_use]
    pub const fn is_folded(&self) -> bool {
        self.folded || self.bound.is_some()
    }

    /// Adds a contribution.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when the position is out of range.
    #[inline]
    pub fn add(&mut self, row: usize, col: usize, value: Real) -> SpiceResult<()> {
        // The common case: the device's next traced stamp, which lies in the
        // pattern and hence in range.
        if let Some(bound) = &mut self.bound
            && let Some(id) = bound.replay(row, col)
        {
            bound.view.take();
            bound.values[id.index()] += value;
            return Ok(());
        }
        self.add_untraced(row, col, value)
    }

    /// [`Self::add`] off the replayed trace: range check, pattern lookup
    /// (which may start this load's own trace) or a triplet.
    #[inline(never)]
    fn add_untraced(&mut self, row: usize, col: usize, value: Real) -> SpiceResult<()> {
        if row >= self.rows || col >= self.cols {
            return Err(SpiceError::Numerical {
                context: "sparse matrix".to_owned(),
                message: format!(
                    "stamp at ({row}, {col}) is out of range for a {}x{} matrix",
                    self.rows, self.cols
                ),
            });
        }
        if let Some(bound) = &mut self.bound {
            bound.view.take();
            if let Some(id) = bound.element(row, col)? {
                bound.values[id.index()] += value;
                return Ok(());
            }
        }
        if value != 0.0 {
            if self.triplets.len() == self.triplets.capacity() {
                reserve(&mut self.triplets, 1)?;
            }
            self.triplets.push(Triplet { row, col, value });
            self.folded = false;
        }
        Ok(())
    }

    /// The contribution at a position, summed over duplicates.
    ///
    /// Binary search after folding, otherwise a scan in stamp order.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Real {
        let bound = self.bound.as_ref().and_then(|bound| {
            bound
                .pattern
                .find(row, col)
                .map(|id| bound.values[id.index()])
        });
        let overflow = if self.folded {
            // Match the floating-point sum's negative-zero identity.
            self.triplets
                .binary_search_by_key(&(row, col), |t| (t.row, t.col))
                .map_or(-0.0, |i| -0.0 + self.triplets[i].value)
        } else {
            self.triplets
                .iter()
                .filter(|triplet| triplet.row == row && triplet.col == col)
                .map(|triplet| triplet.value)
                .sum::<Real>()
        };
        bound.unwrap_or(0.0) + overflow
    }

    /// Sorts and sums duplicate positions.
    pub fn fold_duplicates(&mut self) {
        if self.folded {
            return;
        }
        fold_triplets(&mut self.triplets);
        self.folded = true;
    }

    /// Drops every contribution, keeping the dimensions.
    pub fn clear(&mut self) {
        self.triplets.clear();
        self.folded = true;
        if let Some(bound) = &mut self.bound {
            bound.values.fill(0.0);
            bound.view.take();
            bound.reset_trace();
        }
    }

    /// The stored contributions. For a bound matrix: the folded nonzero
    /// elements in row-major order, merged with any overflow.
    #[must_use]
    pub fn triplets(&self) -> &[Triplet] {
        let Some(bound) = &self.bound else {
            return &self.triplets;
        };
        bound.view.get_or_init(|| {
            let bound_entries = bound.pattern.sorted().filter_map(|(id, (row, col))| {
                let value = bound.values[id.index()];
                if value == 0.0 {
                    return None;
                }
                Some(Triplet { row, col, value })
            });
            if self.triplets.is_empty() {
                return bound_entries.collect();
            }
            // The overflow has no positions in the pattern. Fold only the
            // overflow, and merge the two sorted runs without another
            // whole-matrix sort or an intermediate SparseMatrix.
            let overflow: Cow<'_, [Triplet]> = if self.folded {
                Cow::Borrowed(&self.triplets)
            } else {
                let mut entries = self.triplets.clone();
                fold_triplets(&mut entries);
                Cow::Owned(entries)
            };
            let mut additional = overflow
                .iter()
                .filter(|t| t.value != 0.0)
                .copied()
                .peekable();
            let mut bound = bound_entries.peekable();
            std::iter::from_fn(|| match (bound.peek(), additional.peek()) {
                (Some(a), Some(b)) if (b.row, b.col) < (a.row, a.col) => additional.next(),
                (Some(_), _) => bound.next(),
                (None, _) => additional.next(),
            })
            .collect()
        })
    }

    /// Builds a petgraph graph of the assembled matrix's structural couplings.
    ///
    /// Vertices are matrix row indices, including diagonal-only/empty rows.
    /// An undirected edge exists when either off-diagonal entry is nonzero
    /// after summing duplicate stamps (as in `cktload.c` assembly). Exact zero
    /// cancellations disappear; asymmetric entries are not summed together.
    /// The input matrix is left unchanged. Use petgraph algorithms for block
    /// decomposition instead of maintaining a separate adjacency structure.
    ///
    /// MNA eliminates ground, so row 0 is an ordinary unknown, **not** ground.
    /// This graph has no physical ground-path information. Disconnected blocks
    /// can all be nonsingular; even a connected matrix can be singular. No
    /// solvability or finite-value checks are performed by this structural API.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] if the matrix is not square: rows and columns
    /// must refer to the same unknown set for this coupling interpretation.
    pub fn coupling_graph(&self) -> SpiceResult<UnGraphMap<usize, ()>> {
        if self.rows != self.cols {
            return Err(SpiceError::Numerical {
                context: "matrix coupling graph".to_owned(),
                message: format!("expected a square matrix, got {}x{}", self.rows, self.cols),
            });
        }
        let mut graph = UnGraphMap::new();
        for row in 0..self.rows {
            graph.add_node(row);
        }
        if let Some((pattern, values)) = self.bound_values() {
            for (id, (row, col)) in pattern.sorted() {
                if row != col && values[id.index()] != 0.0 {
                    graph.add_edge(row, col, ());
                }
            }
            return Ok(graph);
        }
        let mut assembled = Cow::Borrowed(self);
        if !self.is_folded() {
            assembled.to_mut().fold_duplicates();
        }
        for triplet in assembled.triplets() {
            if triplet.row != triplet.col && triplet.value != 0.0 {
                graph.add_edge(triplet.row, triplet.col, ());
            }
        }
        Ok(graph)
    }

    /// Multiplies by a dense vector, for testing the assembled system.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when the dimensions do not agree.
    pub fn mul_vector(&self, vector: &Vector) -> SpiceResult<Vector> {
        if vector.len() != self.cols {
            return Err(SpiceError::Numerical {
                context: "sparse matrix".to_owned(),
                message: format!(
                    "vector length {} does not match {} columns",
                    vector.len(),
                    self.cols
                ),
            });
        }
        let mut result = Vector::from(crate::primitives::filled_vec(
            self.rows,
            0.,
            "sparse matrix-vector output",
        )?);
        let y = result.as_mut_slice();
        let x = vector.as_slice();
        if let Some(bound) = &self.bound
            && self.triplets.is_empty()
        {
            // Use the same row-major accumulation as the triplet view,
            // without materializing coordinates for the bound coefficients.
            for (id, (row, col)) in bound.pattern.sorted() {
                let value = bound.values[id.index()];
                if value != 0.0 {
                    let product = value * x[col];
                    y[row] += product;
                }
            }
        } else {
            for triplet in self.triplets() {
                let product = triplet.value * x[triplet.col];
                y[triplet.row] += product;
            }
        }
        Ok(result)
    }

    /// Factors a snapshot. Empty systems are explicitly rejected.
    ///
    /// # Errors
    ///
    /// Invalid assembly or a structurally/numerically singular matrix.
    pub fn factorize(&self) -> SpiceResult<SparseLu> {
        SparseLu::new(self, None)
    }

    /// Convenience solve, factoring the current storage each time.
    ///
    /// For repeated RHS solves, retain the owned result of `factorize`.
    /// No factorization is required beforehand and there is no backend cache.
    ///
    /// # Errors
    ///
    /// Invalid assembly/RHS, singular matrix, or failed residual.
    pub fn solve(&self, rhs: &Vector) -> SpiceResult<Vector> {
        self.factorize()?.solve(rhs)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    #[ignore = "manual sparse construction and replay throughput benchmark"]
    fn sparse_construction_and_replay_bench() {
        use std::{hint::black_box, time::Instant};
        const N: usize = 32_768;
        const LOADS: usize = 100;
        let entries: Vec<_> = (0..N)
            .flat_map(|row| [2, 0, 1, N - 1, 2].map(move |offset| (row, (row + offset) % N)))
            .collect();
        for _ in 0..5 {
            let start = Instant::now();
            let pattern = Pattern::new(N, black_box(entries.iter().copied())).unwrap();
            let setup = start.elapsed();
            let start = Instant::now();
            let extended = pattern
                .extended((0..N).map(|row| (row, (row + 7) % N)))
                .unwrap();
            let extension = start.elapsed();
            assert_eq!(pattern.len(), 4 * N);
            assert_eq!(extended.len(), 5 * N);
            for position in [entries[0], entries[N], entries[entries.len() - 1]] {
                assert_eq!(
                    pattern.find(position.0, position.1),
                    extended.find(position.0, position.1)
                );
            }
            let pattern = Arc::new(pattern);
            let start = Instant::now();
            let mut matrix = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
            let bound = start.elapsed();
            let load = |matrix: &mut SparseMatrix| {
                for (row, stamps) in entries.as_chunks::<5>().0.iter().enumerate() {
                    matrix.begin_device(row).unwrap();
                    for &(r, c) in stamps {
                        matrix.add(r, c, black_box(0.5)).unwrap();
                    }
                }
                matrix.record_trace().unwrap();
            };
            let start = Instant::now();
            load(&mut matrix);
            let record = start.elapsed();
            let start = Instant::now();
            for _ in 0..LOADS {
                matrix.clear();
                load(&mut matrix);
                black_box(matrix.bound_values());
            }
            let replay = start.elapsed();
            assert_eq!(matrix.get(0, 2).to_bits(), 1.0_f64.to_bits());
            assert_eq!(matrix.nnz(), 4 * N);
            eprintln!(
                "setup_ms={:.3} extension_ms={:.3} bound_ms={:.3} trace_ms={:.3} replay_ms={:.3} stamps={}",
                setup.as_secs_f64() * 1e3,
                extension.as_secs_f64() * 1e3,
                bound.as_secs_f64() * 1e3,
                record.as_secs_f64() * 1e3,
                replay.as_secs_f64() * 1e3,
                LOADS * entries.len()
            );
        }
    }

    #[test]
    fn compact_capacity_and_trace_offsets_are_checked_before_allocation() {
        assert_eq!(compact_count(0).unwrap(), 0);
        assert_eq!(compact_count(u32::MAX as usize).unwrap(), u32::MAX);
        if let Some(beyond) = (u32::MAX as usize).checked_add(1) {
            assert!(compact_count(beyond).is_err());
            assert!(pattern_rows(beyond).is_err());
        }
        assert!(offsets_len(usize::MAX).is_err());
        assert!(offsets_len(u32::MAX as usize).is_err());
        assert!(reserve::<u64>(&mut Vec::new(), usize::MAX).is_err());

        let pattern = Arc::new(Pattern::new(2, [(1, 0), (0, 1)]).unwrap());
        let mut matrix = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        assert!(matrix.begin_device(usize::MAX).is_err());
        assert!(
            matrix
                .lend_fixed(FixedStamps::default(), usize::MAX)
                .is_err()
        );
        let foreign = Pattern::new(2, [(0, 0), (0, 1), (1, 1)]).unwrap();
        let foreign_id = foreign.find(1, 1).unwrap();
        assert_eq!(pattern.position(foreign_id), None);
        assert!(matrix.add_element(foreign_id, 1.).is_err());
        matrix.begin_device(0).unwrap();
        matrix.add(1, 0, 3.).unwrap();
        matrix.record_trace().unwrap();
        assert_eq!(matrix.get(1, 0), 3.);
        assert_eq!(pattern.trace().starts, [0, 1]);
    }

    #[test]
    fn extension_preserves_first_appearance_and_rejects_bad_coordinates() {
        let pattern = Pattern::new(4, [(3, 2), (0, 0), (3, 2)]).unwrap();
        assert!(pattern.extended([(2, 1), (4, 0)]).is_err());
        assert_eq!(pattern.len(), 2);
        let extended = pattern.extended([(2, 1), (0, 0), (3, 1), (2, 1)]).unwrap();
        assert_eq!(
            extended.positions().collect::<Vec<_>>(),
            [(3, 2), (0, 0), (2, 1), (3, 1)]
        );
        for (position, index) in extended.positions().zip(0..) {
            assert_eq!(
                extended.find(position.0, position.1).unwrap().index(),
                index
            );
        }
    }

    #[test]
    fn poisoned_ordering_discards_cache_and_unlocks_after_recovery() {
        let pattern = Pattern::new(1, [(0, 0)]).unwrap();
        std::thread::scope(|scope| {
            let result = scope
                .spawn(|| {
                    pattern.with_ordering(|state| {
                        state.ac_reorder = false;
                        state.solve_workspace.push(42.);
                        panic!("interrupted ordering transaction");
                    })
                })
                .join();
            assert!(result.is_err());
        });
        pattern.with_ordering(|state| {
            assert!(state.ac_reorder);
            assert!(state.solve_workspace.is_empty());
            state.ac_reorder = false;
        });
        assert!(!pattern.ordering.is_poisoned());
        assert!(!pattern.with_ordering(|state| state.ac_reorder));
    }

    #[test]
    fn bucketed_pattern_index_matches_global_sort_and_clone_resets_cache() {
        let mut seed = 17_u64;
        let mut positions = Vec::new();
        for _ in 0..1000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let row = ((seed >> 32) % 101) as usize;
            let col = ((seed >> 40) % 101) as usize;
            if !row.is_multiple_of(3) {
                positions.push((row, col));
            }
        }
        let pattern = Pattern::new(101, positions).unwrap();
        let mut expected: Vec<_> = (0..pattern.len())
            .map(|index| ElementId(u32::try_from(index).unwrap()))
            .collect();
        expected.sort_unstable_by_key(|&id| pattern.position(id));
        assert!(
            pattern
                .sorted()
                .map(|(id, _)| id)
                .eq(expected.iter().copied())
        );
        for id in expected {
            let (r, c) = pattern.position(id).unwrap();
            assert_eq!(pattern.find(r, c), Some(id));
        }
        assert!(pattern.find(0, 1).is_none());
        assert!(pattern.find(99, 1).is_none());
        pattern.with_ordering(|state| state.ac_reorder = false);
        let copy = pattern.clone();
        assert_eq!(copy, pattern);
        assert!(copy.with_ordering(|state| state.ac_reorder));
        assert!(!pattern.with_ordering(|state| state.ac_reorder));
        assert_eq!(copy.index.row_start, pattern.index.row_start);
        assert_eq!(copy.index.row_ids, pattern.index.row_ids);
        assert_eq!(copy.index.row_cols, pattern.index.row_cols);

        let empty = Pattern::new(0, []).unwrap();
        assert_eq!(empty.index.row_start, [0]);
        let monotonic = Pattern::new(5, [(1, 0), (1, 3), (3, 3)]).unwrap();
        assert_eq!(monotonic.index.row_start, [0, 0, 2, 2, 3, 3]);
        assert_eq!(monotonic.index.row_ids, [0, 1, 2]);
    }

    #[test]
    fn indexed_edits_preserve_overflow_and_standalone_stamp_order() {
        let pattern = Arc::new(Pattern::new(4, [(0, 0), (1, 1), (2, 2), (3, 3)]).unwrap());
        let mut a = SparseMatrix::bound(pattern).unwrap();
        a.add(1, 3, 2.).unwrap();
        a.add(1, 3, 1.).unwrap();
        a.set(1, 1, 4.).unwrap();
        assert_eq!(
            a.triplets()
                .iter()
                .map(|t| (t.row, t.col, t.value))
                .collect::<Vec<_>>(),
            [(1, 1, 4.), (1, 3, 3.)]
        );
        a.set(1, 1, 5.).unwrap();
        assert_eq!(a.get(1, 3), 3.);
        assert_eq!(a.triplets()[0].value, 5.);
        a.fold_duplicates();
        a.zero_row(1, |c| c == 1);
        assert_eq!(a.get(1, 3), 0.);
        assert_eq!(a.get(1, 1), 5.);

        let mut plain = SparseMatrix::new(4, 4);
        for (r, c, v) in [(3, 3, 9.), (1, 3, 2.), (1, 1, 4.), (0, 0, 1.)] {
            plain.add(r, c, v).unwrap();
        }
        plain.fold_duplicates();
        assert!(plain.zero_row(1, |c| c == 1));
        assert_eq!(plain.triplets()[2].value, 0.);
        plain.set(1, 1, 7.).unwrap();
        let positions = plain
            .triplets()
            .iter()
            .map(|t| (t.row, t.col, t.value))
            .collect::<Vec<_>>();
        assert_eq!(positions, [(0, 0, 1.), (1, 3, 0.), (3, 3, 9.), (1, 1, 7.)]);
        assert_eq!(plain.get(1, 1), 7.);
    }

    #[test]
    fn folding_compacts_in_place_in_stable_stamp_order() {
        let mut matrix = SparseMatrix::new(3, 3);
        for (row, col, value) in [
            (1, 2, 1e16),
            (0, 0, 2.),
            (1, 2, -1e16),
            (2, 0, -1.),
            (0, 0, -2.),
            (1, 2, 1.),
            (2, 0, 1.),
        ] {
            matrix.add(row, col, value).unwrap();
        }
        let allocation = matrix.triplets.as_ptr();
        let capacity = matrix.triplets.capacity();
        matrix.fold_duplicates();
        assert_eq!(matrix.triplets.as_ptr(), allocation);
        assert_eq!(matrix.triplets.capacity(), capacity);
        assert_eq!(
            matrix.triplets(),
            &[Triplet {
                row: 1,
                col: 2,
                value: 1.
            }]
        );
        matrix.fold_duplicates();
        assert_eq!(
            matrix.triplets(),
            &[Triplet {
                row: 1,
                col: 2,
                value: 1.
            }]
        );
    }

    #[test]
    fn bound_read_paths_do_not_materialize_triplets() {
        let pattern = Arc::new(Pattern::new(3, [(0, 2), (0, 0), (0, 1), (2, 1)]).unwrap());
        let mut matrix = SparseMatrix::bound(pattern).unwrap();
        assert!(matrix.is_empty());
        for (r, c, v) in [(0, 2, 1.0), (0, 0, 1e16), (0, 1, -1e16), (2, 1, -2.0)] {
            matrix.add(r, c, v).unwrap();
        }
        assert_eq!(matrix.nnz(), 4);
        assert!(!matrix.is_empty());
        assert_eq!(matrix.get(0, 1), -1e16);
        let x = Vector::from_slice(&[1., 1., 1.]);
        // Creation order would lose the 1; row-major order preserves it.
        assert_eq!(matrix.mul_vector(&x).unwrap().as_slice(), &[1., 0., -2.]);
        assert!(matrix.bound.as_ref().unwrap().view.get().is_none());
        matrix.add(1, 2, 3.).unwrap();
        matrix.add(1, 2, -3.).unwrap();
        assert_eq!(matrix.nnz(), 4);
        assert_eq!(matrix.mul_vector(&x).unwrap().as_slice(), &[1., 0., -2.]);
    }

    #[test]
    fn folded_lookup_preserves_ordinary_and_overflow_values() {
        let pattern = Arc::new(Pattern::new(4, [(1, 1)]).unwrap());
        let mut matrix = SparseMatrix::bound(pattern).unwrap();
        matrix.add(1, 1, 4.).unwrap();
        matrix.add(2, 0, 1e16).unwrap();
        matrix.add(2, 0, -1e16).unwrap();
        matrix.add(2, 0, 1.).unwrap();
        matrix.add(0, 2, -2.).unwrap();
        let values = [(1, 1), (2, 0), (0, 2), (3, 3)].map(|(r, c)| matrix.get(r, c).to_bits());
        matrix.fold_duplicates();
        assert_eq!(
            values,
            [(1, 1), (2, 0), (0, 2), (3, 3)].map(|(r, c)| matrix.get(r, c).to_bits())
        );
    }

    #[test]
    fn pattern_extension_preserves_old_handles_and_new_creation_order() {
        let old = Pattern::new(4, [(3, 1), (0, 0)]).unwrap();
        let new = old
            .extended([(0, 0), (2, 3), (1, 2), (2, 3), (0, 1)])
            .unwrap();
        assert_eq!(new.len(), 5);
        for (i, (r, c)) in [(3, 1), (0, 0), (2, 3), (1, 2), (0, 1)]
            .into_iter()
            .enumerate()
        {
            assert_eq!(new.find(r, c).unwrap().index(), i);
        }
    }

    #[test]
    fn extension_rejects_out_of_range_columns() {
        assert!(
            Pattern::new(1, [(0, 0)])
                .unwrap()
                .extended([(0, 1)])
                .is_err()
        );
        assert!(Pattern::new(1, [(1, 0)]).is_err());
        assert!(Pattern::new(usize::MAX, []).is_err());
    }

    #[test]
    fn row_clearing_and_graph_use_current_nonzero_values() {
        let mut plain = SparseMatrix::new(3, 3);
        let pattern = Arc::new(Pattern::new(3, [(2, 0), (1, 2), (1, 0), (1, 1)]).unwrap());
        let mut bound = SparseMatrix::bound(pattern).unwrap();
        for (r, c, value) in [(2, 0, 7.), (1, 2, 4.), (1, 0, 3.), (1, 1, 2.)] {
            plain.add(r, c, value).unwrap();
            bound.add(r, c, value).unwrap();
        }
        plain.fold_duplicates();
        for matrix in [&mut plain, &mut bound] {
            assert!(matrix.zero_row(1, |col| col == 1));
            assert!(!matrix.zero_row(9, |_| true));
            assert_eq!(matrix.get(1, 1), 2.);
            assert_eq!(matrix.get(1, 0), 0.);
            assert_eq!(matrix.get(1, 2), 0.);
            assert_eq!(matrix.get(2, 0), 7.);
            let graph = matrix.coupling_graph().unwrap();
            assert_eq!(graph.node_count(), 3);
            assert_eq!(graph.edge_count(), 1);
            assert!(graph.contains_edge(2, 0));
        }
        assert!(bound.bound.as_ref().unwrap().view.get().is_none());
    }

    use std::sync::Arc;

    use super::{
        ElementId, FixedStamps, Pattern, SparseMatrix, Triplet, compact_count, offsets_len,
        pattern_rows, reserve,
    };
    use crate::maths::dense::Vector;

    /// One device's load of `stamps` into a fresh matrix bound to `pattern`,
    /// as a fixed batch (`mask` marks the stamped, non-ground entries of a
    /// batch of `len`). Returns the matrix, whether the batch went by
    /// element, and the verifications to lend to the next load.
    fn fixed_load(
        pattern: &Arc<Pattern>,
        fixed: FixedStamps,
        stamps: &[(usize, usize, f64)],
        mask: u64,
        values: &[f64; 3],
    ) -> (SparseMatrix, bool, FixedStamps) {
        let mut matrix = SparseMatrix::bound(Arc::clone(pattern)).unwrap();
        assert!(matrix.lend_fixed(fixed, 1).unwrap().is_none());
        matrix.begin_device(0).unwrap();
        let by_element = matrix.add_fixed(values);
        if !by_element {
            let start = matrix.fixed_start();
            for &(row, col, value) in stamps {
                matrix.add(row, col, value).unwrap();
            }
            matrix.verify_fixed(start, values.len(), mask).unwrap();
        }
        let fixed = matrix.return_fixed();
        matrix.record_trace().unwrap();
        (matrix, by_element, fixed)
    }

    #[test]
    fn bound_overflow_projection_merges_sorted_runs_and_preserves_bits() {
        let pattern = Arc::new(Pattern::new(3, [(2, 2), (0, 1), (1, 1)]).unwrap());
        let mut matrix = SparseMatrix::bound(pattern).unwrap();
        let mut plain = SparseMatrix::new(3, 3);
        for (r, c, v) in [
            (2, 2, 4.),
            (0, 1, 3.),
            (1, 1, 9.),
            (2, 0, 5.),
            (0, 0, 1e16),
            (0, 0, -1e16),
            (0, 0, 1.),
            (1, 2, 2.),
            (1, 2, -2.),
            (0, 2, -5.),
        ] {
            matrix.add(r, c, v).unwrap();
            plain.add(r, c, v).unwrap();
        }
        plain.fold_duplicates();
        assert_eq!(matrix.triplets(), plain.triplets());
        matrix.fold_duplicates();
        assert_eq!(matrix.triplets(), plain.triplets());
        matrix.zero_row(0, |c| c == 1);
        plain.zero_row(0, |c| c == 1);
        let expected: Vec<_> = plain
            .triplets()
            .iter()
            .filter(|t| t.value != 0.)
            .copied()
            .collect();
        assert_eq!(matrix.triplets(), expected);
        matrix.add(0, 2, 7.).unwrap();
        assert_eq!(matrix.get(0, 2), 7.);
        assert!(
            matrix
                .triplets()
                .iter()
                .any(|t| t.row == 0 && t.col == 2 && t.value == 7.)
        );
    }

    #[test]
    fn trace_replaces_shortened_and_skipped_segments_and_reuses_its_storage() {
        let pattern = Arc::new(Pattern::new(3, [(0, 0), (0, 1), (1, 1), (2, 2)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        a.begin_device(0).unwrap();
        a.add(0, 0, 1.).unwrap();
        a.add(0, 1, 2.).unwrap();
        a.begin_device(1).unwrap();
        a.add(1, 1, 3.).unwrap();
        a.begin_device(2).unwrap();
        a.add(2, 2, 4.).unwrap();
        a.record_trace().unwrap();
        assert_eq!(pattern.trace().starts, [0, 2, 3, 4]);

        a.clear();
        a.begin_device(0).unwrap();
        a.add(0, 0, 10.).unwrap();
        a.begin_device(2).unwrap();
        a.add(2, 2, 20.).unwrap();
        a.record_trace().unwrap();
        let trace = pattern.trace();
        assert_eq!(trace.starts, [0, 1, 1, 2]);
        assert_eq!(trace.entries, [(0, 0, 0), (2, 2, 3)]);
        assert_eq!(a.get(0, 1), 0.);
        assert_eq!(a.get(1, 1), 0.);
        assert_eq!(a.get(2, 2), 20.);

        a.clear();
        a.begin_device(0).unwrap();
        a.add(0, 0, 1.).unwrap();
        a.begin_device(1).unwrap();
        a.begin_device(2).unwrap();
        a.add(2, 2, 2.).unwrap();
        a.record_trace().unwrap();
        assert!(Arc::ptr_eq(&trace, &pattern.trace()));

        // Omit a trailing device entirely: only actual segments remain.
        a.clear();
        a.begin_device(0).unwrap();
        a.add(0, 0, 3.).unwrap();
        a.record_trace().unwrap();
        assert_eq!(pattern.trace().starts, [0, 1]);
        assert_eq!(pattern.trace().entries, [(0, 0, 0)]);

        // Invalid traversal disables learning, never numerical stamping.
        let before = pattern.trace();
        a.clear();
        a.begin_device(0).unwrap();
        a.add(0, 0, 1.).unwrap();
        a.begin_device(0).unwrap();
        a.add(0, 1, 9.).unwrap();
        a.record_trace().unwrap();
        assert_eq!(a.get(0, 1), 9.);
        assert!(Arc::ptr_eq(&before, &pattern.trace()));
    }

    #[test]
    fn stale_fixed_batches_are_invalidated_without_a_lend_cycle() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (1, 1)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        a.begin_device(0).unwrap();
        a.add(0, 0, 1.).unwrap();
        a.record_trace().unwrap();
        a.clear();
        assert!(a.lend_fixed(FixedStamps::default(), 1).unwrap().is_none());
        a.begin_device(0).unwrap();
        let start = a.fixed_start();
        a.add(0, 0, 1.).unwrap();
        a.verify_fixed(start, 2, 0b01).unwrap();
        a.record_trace().unwrap();

        // Another load changes the same-length trace: blindly keeping
        // the old mask would apply the wrong batch value to element 1.
        let mut b = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        b.begin_device(0).unwrap();
        b.add(1, 1, 3.).unwrap();
        b.record_trace().unwrap();
        a.clear();
        a.begin_device(0).unwrap();
        assert!(!a.add_fixed(&[2., 4.]));
        a.add(1, 1, 4.).unwrap();
        assert_eq!(a.get(1, 1), 4.);
    }

    #[test]
    fn invalid_fixed_mask_cannot_drop_traced_stamps() {
        let pattern = Arc::new(Pattern::new(1, [(0, 0)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        a.begin_device(0).unwrap();
        a.add(0, 0, 1.).unwrap();
        a.record_trace().unwrap();
        a.clear();
        assert!(a.lend_fixed(FixedStamps::default(), 1).unwrap().is_none());
        a.begin_device(0).unwrap();
        let start = a.fixed_start();
        a.add(0, 0, 1.).unwrap();
        a.verify_fixed(start, 1, 0b10).unwrap();
        let fixed = a.return_fixed();
        a.record_trace().unwrap();
        a.clear();
        assert!(a.lend_fixed(fixed, 1).unwrap().is_none());
        a.begin_device(0).unwrap();
        assert!(!a.add_fixed(&[2.]));
    }

    #[test]
    fn bound_equality_and_clone_ignore_lazy_projection_and_trace_state() {
        let pattern = Arc::new(Pattern::new(1, [(0, 0)]).unwrap());
        let mut a = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        a.add(0, 0, 2.).unwrap();
        let b = a.clone();
        let _ = a.triplets();
        assert_eq!(a, b);
        a.record_trace().unwrap();
        assert_eq!(a, b);
        let clone = a.clone();
        assert!(clone.bound.as_ref().unwrap().view.get().is_none());
        assert_eq!(clone, a);
    }

    #[test]
    fn interleaved_fixed_groups_replay_exactly_in_device_stamp_order() {
        // Two independently verifiable groups separated by an ordinary
        // stamp, each with positions omitted at ground. A device's one trace
        // may contain any number of such groups; no group may bypass a
        // neighboring stamp or fold its floating-point accumulation order.
        let pattern = Arc::new(Pattern::new(3, [(0, 0), (1, 1), (2, 2), (2, 0), (0, 2)]).unwrap());
        let mut matrix = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        let mut fixed = FixedStamps::default();
        let mut used_fast = 0;
        for cycle in 0..4 {
            matrix.clear();
            assert!(matrix.lend_fixed(fixed, 1).unwrap().is_none());
            matrix.begin_device(0).unwrap();
            let first = [1.0 + f64::from(cycle), 0.0, 2.0];
            if matrix.add_fixed(&first) {
                used_fast += 1;
            } else {
                let start = matrix.fixed_start();
                matrix.add(0, 0, first[0]).unwrap();
                matrix.add(1, 1, first[2]).unwrap();
                matrix.verify_fixed(start, first.len(), 0b101).unwrap();
            }
            // The middle stamp cannot be moved across the fixed groups.
            matrix.add(2, 2, 3.0).unwrap();
            let second = [4.0, 5.0, 0.0];
            if matrix.add_fixed(&second) {
                used_fast += 1;
            } else {
                let start = matrix.fixed_start();
                matrix.add(2, 0, second[0]).unwrap();
                matrix.add(0, 2, second[1]).unwrap();
                matrix.verify_fixed(start, second.len(), 0b011).unwrap();
            }
            fixed = matrix.return_fixed();
            matrix.record_trace().unwrap();
            assert_eq!(matrix.get(0, 0), first[0]);
            assert_eq!(matrix.get(1, 1), 2.0);
            assert_eq!(matrix.get(2, 2), 3.0);
            assert_eq!(matrix.get(2, 0), 4.0);
            assert_eq!(matrix.get(0, 2), 5.0);
        }
        assert_eq!(used_fast, 4, "both groups should replay on cycles 3 and 4");
    }

    #[test]
    fn a_verified_fixed_batch_is_added_by_element() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (1, 1), (0, 1)]).unwrap());
        // Entry 1 of the batch is at ground and never stamped.
        let stamps = [(1, 1, 1.0), (0, 1, 2.0)];
        let values = [1.0, 7.0, 2.0];
        // The first load records the trace, the second verifies the batch
        // against it, the third adds by element.
        let (_, by_element, fixed) =
            fixed_load(&pattern, FixedStamps::default(), &stamps, 0b101, &values);
        assert!(!by_element);
        let (_, by_element, fixed) = fixed_load(&pattern, fixed, &stamps, 0b101, &values);
        assert!(!by_element);
        let (matrix, by_element, fixed) = fixed_load(&pattern, fixed, &stamps, 0b101, &values);
        assert!(by_element);
        assert_eq!(matrix.get(1, 1), 1.0);
        assert_eq!(matrix.get(0, 1), 2.0);
        assert_eq!(matrix.get(0, 0), 0.0);
        // A batch of another length is not the verified one.
        let mut other = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        assert!(other.lend_fixed(fixed, 1).unwrap().is_none());
        other.begin_device(0).unwrap();
        assert!(!other.add_fixed(&[1.0, 2.0]));
    }

    #[test]
    fn a_new_trace_or_a_partial_batch_is_not_verified() {
        let pattern = Arc::new(Pattern::new(2, [(0, 0), (1, 1), (0, 1)]).unwrap());
        let stamps = [(1, 1, 1.0), (0, 1, 2.0)];
        let values = [1.0, 7.0, 2.0];
        let (_, _, fixed) = fixed_load(&pattern, FixedStamps::default(), &stamps, 0b101, &values);
        let (_, _, fixed) = fixed_load(&pattern, fixed, &stamps, 0b101, &values);
        // A load that diverges records a new trace: the verification was
        // made against the old one and is dropped.
        let mut diverging = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        assert!(diverging.lend_fixed(fixed, 1).unwrap().is_none());
        diverging.begin_device(0).unwrap();
        diverging.add(0, 0, 5.0).unwrap();
        let fixed = diverging.return_fixed();
        diverging.record_trace().unwrap();
        let (_, by_element, fixed) = fixed_load(&pattern, fixed, &[(0, 0, 5.0)], 0b1, &values);
        assert!(!by_element, "verified against a replaced trace");
        // Stamps after the batch: the batch is not the device's whole
        // segment, so it is never verified.
        let mut partial = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        assert!(partial.lend_fixed(fixed, 1).unwrap().is_none());
        partial.begin_device(0).unwrap();
        let start = partial.fixed_start();
        partial.add(0, 0, 5.0).unwrap();
        partial.verify_fixed(start, 1, 0b1).unwrap();
        let fixed = partial.return_fixed();
        let mut next = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        assert!(next.lend_fixed(fixed, 1).unwrap().is_none());
        next.begin_device(0).unwrap();
        assert!(next.add_fixed(&[5.0]), "the whole segment verifies");
        // The same verification after forget() does not.
        let mut fixed = next.return_fixed();
        fixed.forget();
        let mut forgotten = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        assert!(forgotten.lend_fixed(fixed, 1).unwrap().is_none());
        forgotten.begin_device(0).unwrap();
        assert!(!forgotten.add_fixed(&[5.0]));
        // Unbound matrices never add by element.
        let mut unbound = SparseMatrix::new(2, 2);
        assert!(
            unbound
                .lend_fixed(FixedStamps::default(), 1)
                .unwrap()
                .is_some()
        );
        unbound.begin_device(0).unwrap();
        assert!(!unbound.add_fixed(&[5.0]));
    }

    #[test]
    fn zero_contributions_are_not_stored() {
        let mut matrix = SparseMatrix::new(2, 2);
        matrix.add(0, 0, 0.0).unwrap();
        assert!(matrix.is_empty());
        assert_eq!(matrix.nnz(), 0);
    }

    #[test]
    fn duplicates_sum_on_read() {
        let mut matrix = SparseMatrix::new(2, 2);
        matrix.add(0, 0, 0.001).unwrap();
        matrix.add(0, 0, 0.002).unwrap();
        matrix.add(1, 1, 5.0).unwrap();
        assert_eq!(matrix.nnz(), 3);
        assert!((matrix.get(0, 0) - 0.003).abs() < 1e-18);
        assert_eq!(matrix.get(0, 1), 0.0);
    }

    #[test]
    fn folding_collapses_duplicates_and_cancels_zeros() {
        let mut matrix = SparseMatrix::new(2, 2);
        matrix.add(1, 1, 1.0).unwrap();
        matrix.add(0, 0, 2.0).unwrap();
        matrix.add(0, 0, -2.0).unwrap();
        matrix.fold_duplicates();
        assert!(matrix.is_folded());
        assert_eq!(matrix.nnz(), 1);
        assert_eq!(matrix.triplets()[0].row, 1);
        assert_eq!(matrix.get(0, 0), 0.0);
    }

    #[test]
    fn out_of_range_stamps_are_errors() {
        let mut matrix = SparseMatrix::new(2, 2);
        assert!(matrix.add(0, 2, 1.0).is_err());
        assert!(matrix.add(2, 0, 1.0).is_err());
    }

    #[test]
    fn multiply_walks_the_contributions() {
        let mut matrix = SparseMatrix::new(2, 2);
        matrix.add(0, 0, 2.0).unwrap();
        matrix.add(0, 1, 1.0).unwrap();
        matrix.add(1, 1, 3.0).unwrap();
        let product = matrix.mul_vector(&Vector::from_slice(&[1.0, 1.0])).unwrap();
        assert_eq!(product.as_slice(), &[3.0, 3.0]);
        assert!(matrix.mul_vector(&Vector::zeros(3)).is_err());
    }

    #[test]
    fn singular_system_is_rejected() {
        let matrix = SparseMatrix::new(1, 1);
        assert!(matrix.solve(&Vector::zeros(1)).is_err());
        assert!(!matrix.factorize().unwrap_err().is_not_yet_ported());
    }
}
