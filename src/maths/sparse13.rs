//! Rust port of Kenneth Kundert's SPARSE 1.3, ngspice's native sparse LU.
//!
//! C: `src/maths/sparse/` (`spbuild.c`, `spfactor.c`, `spsolve.c`,
//! `sputils.c`), driven through `spsmp.c`. SPARSE (author Kenneth S.
//! Kundert, advising professor Alberto Sangiovanni-Vincentelli, UC Berkeley)
//! is distributed under the permissive Berkeley notice in its headers, so
//! unlike KLU it may be ported here. The full notice is preserved verbatim
//! in the source comments immediately below and in `NOTICE`.
//
// Verbatim upstream license notice:
// Copyright (c) 1985,86,87,88,89,90
// by Kenneth S. Kundert and the University of California.
//
// Permission to use, copy, modify, and distribute this software and
// its documentation for any purpose and without fee is hereby granted,
// provided that the copyright notices appear in all copies and
// supporting documentation and that the authors and the University of
// California are properly credited.  The authors and the University of
// California make no representations as to the suitability of this
// software for any purpose.  It is provided `as is', without express
// or implied warranty.
//!
//! The configuration matches ngspice's `spconfig.h`: `TRANSLATE`,
//! `EXPANDABLE` (storage is allocated for the external size up front, and
//! the internal size grows as `Translate` numbers new rows, as in C),
//! `DIAGONAL_PIVOTING`,
//! `MODIFIED_NODAL`, `TRANSPOSE`, `DETERMINANT`; no `MODIFIED_MARKOWITZ`,
//! scaling or condition estimation. The default relative threshold is
//! `1e-3` and `TIES_MULTIPLIER` 5 (`MAX_MARKOWITZ_TIES` only applies to
//! `MODIFIED_MARKOWITZ`).
//!
//! The orthogonal linked lists of C pointers become an element arena with
//! index links; a C `ElementPtr *` "address of a link" is a [`Slot`]. The
//! control flow of every routine follows the C function of the same name,
//! including the order of pivot candidates, tie breaking and fill-in creation,
//! so pivot sequences match SPARSE for the same element creation order.
//! Divergences: the factor loop always uses the direct (scatter/gather)
//! variant of `spFactor`, which performs the same arithmetic as the indirect
//! one; and `SearchForSingleton` stops on a structurally missing diagonal,
//! which is where ngspice's version either breaks or dereferences `NULL`.

// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::float_cmp,
    reason = "C device branches compare floating-point values exactly, including nonfinite cases"
)]
#![expect(
    clippy::while_float,
    reason = "the C iterative floating-point loop must preserve its original termination and rounding order"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]

use super::linear::numerical;
use crate::primitives::{Complex, SpiceResult};

const NONE: usize = usize::MAX;
const DEFAULT_THRESHOLD: f64 = 1.0e-3;
const TIES_MULTIPLIER: i64 = 5;

/// Stable original element handle obtained only from a validated frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SolverElement(usize);

fn sparse_reserve<T>(entries: &mut Vec<T>, additional: usize) -> SpiceResult<()> {
    entries.try_reserve(additional).map_err(|error| {
        numerical(
            "SPARSE storage",
            format!("cannot reserve {additional} entries: {error}"),
        )
    })
}

fn sparse_rows(size: usize) -> SpiceResult<usize> {
    // The external indexing uses a zero sentinel and the pivot workspace
    // also needs a trailing sentinel.
    let count = size
        .checked_add(2)
        .ok_or_else(|| numerical("SPARSE storage", "row sentinel count overflows"))?;
    std::alloc::Layout::array::<usize>(count)
        .map_err(|error| numerical("SPARSE storage", format!("row index capacity: {error}")))?;
    Ok(count - 1)
}

fn sparse_indices(size: usize) -> SpiceResult<Vec<usize>> {
    let mut indices = Vec::new();
    sparse_reserve(&mut indices, sparse_rows(size)?)?;
    indices.extend(0..=size);
    Ok(indices)
}

/// Matrix entry type: `f64` for real systems, [`Complex`] for AC.
pub(super) trait Scalar: Copy + PartialEq + std::fmt::Debug {
    /// Additive identity.
    const ZERO: Self;
    /// C `ELEMENT_MAG`: `|re| + |im|`.
    fn mag(self) -> f64;
    /// Real part, used by `CountTwins`' `ABS(Real) == 1.0` test.
    fn real(self) -> f64;
    /// C `1.0 / x` or `CMPLX_RECIPROCAL`.
    fn recip(self) -> Self;
    /// C `x * y` or `CMPLX_MULT`.
    fn mul(self, other: Self) -> Self;
    /// C `to -= a * b` or `CMPLX_MULT_SUBT_ASSIGN(to, a, b)`.
    fn mul_sub(self, a: Self, b: Self) -> Self;
    /// Whether the value is exactly zero (`== 0.0` on every part).
    fn is_zero(self) -> bool;
    /// Whether every scalar component is finite.
    fn is_finite(self) -> bool;
    /// `x.Real += r`.
    fn add_real(self, r: f64) -> Self;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    fn mag(self) -> f64 {
        self.abs()
    }
    fn real(self) -> f64 {
        self
    }
    fn recip(self) -> Self {
        1.0 / self
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn mul_sub(self, a: Self, b: Self) -> Self {
        self - a * b
    }
    fn is_zero(self) -> bool {
        self == 0.0
    }
    fn is_finite(self) -> bool {
        Self::is_finite(self)
    }
    fn add_real(self, r: f64) -> Self {
        self + r
    }
}

impl Scalar for Complex {
    const ZERO: Self = Self::ZERO;
    fn mag(self) -> f64 {
        self.re.abs() + self.im.abs()
    }
    fn real(self) -> f64 {
        self.re
    }
    fn recip(self) -> Self {
        // CMPLX_RECIPROCAL (spdefs.h).
        let den = self;
        let result =
            if (den.re >= den.im && den.re > -den.im) || (den.re < den.im && den.re <= -den.im) {
                let r = den.im / den.re;
                let re = 1.0 / (den.re + r * den.im);
                Self::new(re, -r * re)
            } else {
                let r = den.re / den.im;
                let im = -1.0 / (den.im + r * den.re);
                Self::new(-r * im, im)
            };
        if !result.is_finite() || result == Self::ZERO {
            // The C ratio formula can overflow its denominator even for
            // a finite pivot with a representable reciprocal.
            Self::real(1.0) / den
        } else {
            result
        }
    }
    fn mul(self, b: Self) -> Self {
        Self::new(
            self.re * b.re - self.im * b.im,
            self.re * b.im + self.im * b.re,
        )
    }
    fn mul_sub(self, a: Self, b: Self) -> Self {
        Self::new(
            self.re - (a.re * b.re - a.im * b.im),
            self.im - (a.re * b.im + a.im * b.re),
        )
    }
    fn is_zero(self) -> bool {
        self.re == 0.0 && self.im == 0.0
    }
    fn is_finite(self) -> bool {
        Self::is_finite(self)
    }
    fn add_real(self, r: f64) -> Self {
        Self::new(self.re + r, self.im)
    }
}

/// SPARSE's error state (`spOKAY`, `spSMALL_PIVOT`, `spZERO_DIAG`,
/// `spSINGULAR`). ngspice maps `spSMALL_PIVOT` to `OK`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Status {
    Okay,
    SmallPivot,
    ZeroDiag,
    Singular,
    /// The factorization produced an unrepresentable numeric value.
    NonFinite,
    /// A traversal plan cannot be derived from the ordered topology.
    InvalidPlan,
    /// Coefficients must be reloaded before another factorization.
    NeedsReload,
    /// A required factor allocation exceeds storage capacity.
    Resource,
}

#[derive(Debug, Clone)]
struct Element {
    row: usize,
    col: usize,
    next_in_row: usize,
    next_in_col: usize,
}

/// Index width of an ordered traversal plan. Most circuits fit in u32;
/// retain usize support for larger standalone matrices, with the dispatch
/// outside all numeric loops.
trait PlanIndex: Copy {
    fn pack(index: usize) -> Option<Self>;
    fn unpack(self) -> usize;
}

impl PlanIndex for usize {
    fn pack(index: usize) -> Option<Self> {
        Some(index)
    }
    fn unpack(self) -> usize {
        self
    }
}

impl PlanIndex for u32 {
    fn pack(index: usize) -> Option<Self> {
        Self::try_from(index).ok()
    }
    fn unpack(self) -> usize {
        // The simulator supports 32- and 64-bit targets.
        self as usize
    }
}

/// A numeric traversal entry: stable element handle and internal row
/// (column runs) or internal column (upper-row runs).
#[derive(Debug, Clone, Copy)]
struct RunEntry<I> {
    handle: I,
    coordinate: I,
}

impl<I: PlanIndex> RunEntry<I> {
    fn new(handle: usize, coordinate: usize) -> Option<Self> {
        Some(Self {
            handle: I::pack(handle)?,
            coordinate: I::pack(coordinate)?,
        })
    }

    fn indices(self) -> (usize, usize) {
        (self.handle.unpack(), self.coordinate.unpack())
    }
}

#[derive(Debug, Clone, Copy)]
struct ColumnRun<I> {
    start: I,
    diagonal: I,
    end: I,
}

impl<I: PlanIndex> ColumnRun<I> {
    fn new(start: usize, diagonal: usize, end: usize) -> Option<Self> {
        if !(start <= diagonal && diagonal < end) {
            return None;
        }
        Some(Self {
            start: I::pack(start)?,
            diagonal: I::pack(diagonal)?,
            end: I::pack(end)?,
        })
    }
}

/// Immutable packed traversal, derived from linked topology after ordering.
#[derive(Debug)]
struct FactorRuns<I> {
    columns: Vec<ColumnRun<I>>,
    column_entries: Vec<RunEntry<I>>,
    upper_row_start: Vec<I>,
    upper_row_entries: Vec<RunEntry<I>>,
}

#[derive(Debug)]
enum FactorPlan {
    Compact(FactorRuns<u32>),
    Wide(FactorRuns<usize>),
}

enum RefactorIssue {
    ZeroPivot(usize),
    NonFinite,
}

impl<I: PlanIndex> FactorRuns<I> {
    fn column(&self, col: usize) -> &[RunEntry<I>] {
        let run = self.columns[col - 1];
        &self.column_entries[run.start.unpack()..run.end.unpack()]
    }

    fn upper_column(&self, col: usize) -> &[RunEntry<I>] {
        let run = self.columns[col - 1];
        &self.column_entries[run.start.unpack()..run.diagonal.unpack()]
    }

    fn lower_column(&self, col: usize) -> &[RunEntry<I>] {
        let run = self.columns[col - 1];
        &self.column_entries[run.diagonal.unpack() + 1..run.end.unpack()]
    }

    fn upper_row(&self, row: usize) -> &[RunEntry<I>] {
        &self.upper_row_entries
            [self.upper_row_start[row - 1].unpack()..self.upper_row_start[row].unpack()]
    }

    /// The fixed-order SPARSE scatter/gather refactor, after column one.
    fn refactor<T: Scalar>(
        &self,
        values: &mut [T],
        diag: &[usize],
        dest: &mut [T],
    ) -> Result<(), RefactorIssue> {
        for step in 2..=self.columns.len() {
            for entry in self.column(step) {
                let (handle, row) = entry.indices();
                dest[row] = values[handle];
            }
            for entry in self.upper_column(step) {
                let (handle, row) = entry.indices();
                let mult = dest[row].mul(values[diag[row]]);
                values[handle] = mult;
                for entry in self.lower_column(row) {
                    let (handle, r) = entry.indices();
                    dest[r] = dest[r].mul_sub(mult, values[handle]);
                }
            }
            for entry in self.lower_column(step) {
                let (handle, row) = entry.indices();
                values[handle] = dest[row];
            }
            if dest[step].is_zero() {
                return Err(RefactorIssue::ZeroPivot(step));
            }
            let inverse = dest[step].recip();
            if !inverse.is_finite() || inverse.is_zero() {
                return Err(RefactorIssue::NonFinite);
            }
            values[diag[step]] = inverse;
        }
        Ok(())
    }

    fn solve<T: Scalar>(&self, values: &[T], diag: &[usize], x: &mut [T]) {
        let size = self.columns.len();
        for i in 1..=size {
            let mut temp = x[i];
            if !temp.is_zero() {
                temp = temp.mul(values[diag[i]]);
                x[i] = temp;
                for entry in self.lower_column(i) {
                    let (handle, r) = entry.indices();
                    x[r] = x[r].mul_sub(temp, values[handle]);
                }
            }
        }
        for i in (1..=size).rev() {
            let mut temp = x[i];
            for entry in self.upper_row(i) {
                let (handle, c) = entry.indices();
                temp = temp.mul_sub(values[handle], x[c]);
            }
            x[i] = temp;
        }
    }

    fn solve_transposed<T: Scalar>(&self, values: &[T], diag: &[usize], x: &mut [T]) {
        let size = self.columns.len();
        for i in 1..=size {
            let temp = x[i];
            if !temp.is_zero() {
                for entry in self.upper_row(i) {
                    let (handle, c) = entry.indices();
                    x[c] = x[c].mul_sub(temp, values[handle]);
                }
            }
        }
        for i in (1..=size).rev() {
            let mut temp = x[i];
            for entry in self.lower_column(i) {
                let (handle, r) = entry.indices();
                temp = temp.mul_sub(x[r], values[handle]);
            }
            x[i] = temp.mul(values[diag[i]]);
        }
    }
}

/// The address of a column (or row) link: a list head or an element's link.
#[derive(Debug, Clone, Copy)]
enum Slot {
    Head(usize),
    After(usize),
}

/// Linked topology and maps are immutable between structural edits.
/// Numeric snapshots and scalar transitions share these allocations;
/// ordering that changes their structure detaches only the writer.
#[derive(Debug, Clone)]
struct Topology {
    elements: Vec<Element>,
    diag: Vec<usize>,
    first_in_col: Vec<usize>,
    first_in_row: Vec<usize>,
    ext_to_int_row: Vec<usize>,
    ext_to_int_col: Vec<usize>,
    int_to_ext_row: Vec<usize>,
    int_to_ext_col: Vec<usize>,
}

impl Topology {
    fn translate(
        &mut self,
        current_size: &mut usize,
        ext_row: usize,
        ext_col: usize,
    ) -> (usize, usize) {
        let mut int_row = self.ext_to_int_row[ext_row];
        if int_row == NONE {
            *current_size += 1;
            int_row = *current_size;
            self.ext_to_int_row[ext_row] = int_row;
            self.ext_to_int_col[ext_row] = int_row;
            self.int_to_ext_row[int_row] = ext_row;
            self.int_to_ext_col[int_row] = ext_row;
        }
        let mut int_col = self.ext_to_int_col[ext_col];
        if int_col == NONE {
            *current_size += 1;
            int_col = *current_size;
            self.ext_to_int_row[ext_col] = int_col;
            self.ext_to_int_col[ext_col] = int_col;
            self.int_to_ext_row[int_col] = ext_col;
            self.int_to_ext_col[int_col] = ext_col;
        }
        (int_row, int_col)
    }

    fn locate_in_col(&self, mut last: Slot, row: usize) -> (Slot, usize) {
        let mut e = self.col_get(last);
        while e != NONE {
            let r = self.elements[e].row;
            match r.cmp(&row) {
                std::cmp::Ordering::Less => {
                    last = Slot::After(e);
                    e = self.elements[e].next_in_col;
                }
                std::cmp::Ordering::Equal => return (last, e),
                std::cmp::Ordering::Greater => break,
            }
        }
        (last, NONE)
    }

    fn create_element(&mut self, row: usize, col: usize, last: Slot, rows_linked: bool) -> usize {
        let e = self.elements.len();
        self.elements.push(Element {
            row,
            col,
            next_in_row: NONE,
            next_in_col: self.col_get(last),
        });
        if row == col {
            self.diag[row] = e;
        }
        self.col_set(last, e);
        if rows_linked {
            let mut previous = NONE;
            let mut p = self.first_in_row[row];
            while p != NONE && self.elements[p].col < col {
                previous = p;
                p = self.elements[p].next_in_row;
            }
            if previous == NONE {
                self.elements[e].next_in_row = self.first_in_row[row];
                self.first_in_row[row] = e;
            } else {
                self.elements[e].next_in_row = self.elements[previous].next_in_row;
                self.elements[previous].next_in_row = e;
            }
        }
        e
    }

    /// Append to a sorted column in constant time, or mark it for deferred
    /// sorting if rows arrive in an arbitrary order. A missing tail with a
    /// nonempty head denotes an unordered column.
    fn append_original(&mut self, row: usize, col: usize, tails: &mut [usize]) -> usize {
        let tail = tails[col];
        let head = self.first_in_col[col];
        if head == NONE {
            let handle = self.create_element(row, col, Slot::Head(col), false);
            tails[col] = handle;
            handle
        } else if tail != NONE && row > self.elements[tail].row {
            let handle = self.create_element(row, col, Slot::After(tail), false);
            tails[col] = handle;
            handle
        } else {
            if row >= self.elements[head].row {
                tails[col] = NONE;
            }
            self.create_element(row, col, Slot::Head(col), false)
        }
    }

    /// Finalize only unordered columns. Stable handles remain in original
    /// creation order; the linked traversal is sorted by internal row.
    fn finish_original_columns(&mut self, tails: &[usize]) -> SpiceResult<()> {
        let mut run = Vec::new();
        for (col, &tail) in tails.iter().enumerate().skip(1) {
            if tail != NONE || self.first_in_col[col] == NONE {
                continue;
            }
            run.clear();
            let mut e = self.first_in_col[col];
            while e != NONE {
                if run.len() == run.capacity() {
                    sparse_reserve(&mut run, 1)?;
                }
                run.push(e);
                e = self.elements[e].next_in_col;
            }
            run.sort_unstable_by_key(|&handle| self.elements[handle].row);
            self.first_in_col[col] = run[0];
            for pair in run.windows(2) {
                if self.elements[pair[0]].row == self.elements[pair[1]].row {
                    return Err(numerical(
                        "SPARSE construction",
                        "duplicate original sparse element",
                    ));
                }
                self.elements[pair[0]].next_in_col = pair[1];
            }
            self.elements[run[run.len() - 1]].next_in_col = NONE;
        }
        Ok(())
    }

    fn col_get(&self, slot: Slot) -> usize {
        match slot {
            Slot::Head(col) => self.first_in_col[col],
            Slot::After(e) => self.elements[e].next_in_col,
        }
    }

    fn col_set(&mut self, slot: Slot, value: usize) {
        match slot {
            Slot::Head(col) => self.first_in_col[col] = value,
            Slot::After(e) => self.elements[e].next_in_col = value,
        }
    }

    fn row_get(&self, slot: Slot) -> usize {
        match slot {
            Slot::Head(row) => self.first_in_row[row],
            Slot::After(e) => self.elements[e].next_in_row,
        }
    }

    fn row_set(&mut self, slot: Slot, value: usize) {
        match slot {
            Slot::Head(row) => self.first_in_row[row] = value,
            Slot::After(e) => self.elements[e].next_in_row = value,
        }
    }

    /// Exchange elements in one column, preserving its row order.
    fn exchange_col_elements(
        &mut self,
        row1: usize,
        e1: usize,
        row2: usize,
        e2: usize,
        column: usize,
    ) {
        let mut above1 = Slot::Head(column);
        let mut p = self.col_get(above1);
        while self.elements[p].row < row1 {
            above1 = Slot::After(p);
            p = self.col_get(above1);
        }
        if e1 == NONE {
            let below1 = p;
            if self.elements[below1].row != row2 {
                let mut above2;
                loop {
                    above2 = Slot::After(p);
                    p = self.col_get(above2);
                    if self.elements[p].row >= row2 {
                        break;
                    }
                }
                let below2 = self.elements[e2].next_in_col;
                self.col_set(above2, below2);
                self.col_set(above1, e2);
                self.elements[e2].next_in_col = below1;
            }
            self.elements[e2].row = row1;
        } else {
            let below1 = self.elements[e1].next_in_col;
            if e2 == NONE {
                if below1 != NONE && self.elements[below1].row < row2 {
                    self.col_set(above1, below1);
                    let mut p = below1;
                    let mut above2;
                    loop {
                        above2 = Slot::After(p);
                        p = self.col_get(above2);
                        if !(p != NONE && self.elements[p].row < row2) {
                            break;
                        }
                    }
                    self.col_set(above2, e1);
                    self.elements[e1].next_in_col = p;
                    self.col_set(above1, below1);
                }
                self.elements[e1].row = row2;
            } else {
                if self.elements[below1].row == row2 {
                    self.elements[e1].next_in_col = self.elements[e2].next_in_col;
                    self.elements[e2].next_in_col = e1;
                    self.col_set(above1, e2);
                } else {
                    let mut p = below1;
                    let mut above2;
                    loop {
                        above2 = Slot::After(p);
                        p = self.col_get(above2);
                        if self.elements[p].row >= row2 {
                            break;
                        }
                    }
                    let below2 = self.elements[e2].next_in_col;
                    self.col_set(above1, e2);
                    self.elements[e2].next_in_col = below1;
                    self.col_set(above2, e1);
                    self.elements[e1].next_in_col = below2;
                }
                self.elements[e1].row = row2;
                self.elements[e2].row = row1;
            }
        }
    }

    /// Row-list counterpart of the column exchange.
    fn exchange_row_elements(
        &mut self,
        col1: usize,
        e1: usize,
        col2: usize,
        e2: usize,
        row: usize,
    ) {
        let mut left1 = Slot::Head(row);
        let mut p = self.row_get(left1);
        while self.elements[p].col < col1 {
            left1 = Slot::After(p);
            p = self.row_get(left1);
        }
        if e1 == NONE {
            let right1 = p;
            if self.elements[right1].col != col2 {
                let mut left2;
                loop {
                    left2 = Slot::After(p);
                    p = self.row_get(left2);
                    if self.elements[p].col >= col2 {
                        break;
                    }
                }
                let right2 = self.elements[e2].next_in_row;
                self.row_set(left2, right2);
                self.row_set(left1, e2);
                self.elements[e2].next_in_row = right1;
            }
            self.elements[e2].col = col1;
        } else {
            let right1 = self.elements[e1].next_in_row;
            if e2 == NONE {
                if right1 != NONE && self.elements[right1].col < col2 {
                    self.row_set(left1, right1);
                    let mut p = right1;
                    let mut left2;
                    loop {
                        left2 = Slot::After(p);
                        p = self.row_get(left2);
                        if !(p != NONE && self.elements[p].col < col2) {
                            break;
                        }
                    }
                    self.row_set(left2, e1);
                    self.elements[e1].next_in_row = p;
                    self.row_set(left1, right1);
                }
                self.elements[e1].col = col2;
            } else {
                if self.elements[right1].col == col2 {
                    self.elements[e1].next_in_row = self.elements[e2].next_in_row;
                    self.elements[e2].next_in_row = e1;
                    self.row_set(left1, e2);
                } else {
                    let mut p = right1;
                    let mut left2;
                    loop {
                        left2 = Slot::After(p);
                        p = self.row_get(left2);
                        if self.elements[p].col >= col2 {
                            break;
                        }
                    }
                    let right2 = self.elements[e2].next_in_row;
                    self.row_set(left1, e2);
                    self.elements[e2].next_in_row = right1;
                    self.row_set(left2, e1);
                    self.elements[e1].next_in_row = right2;
                }
                self.elements[e1].col = col2;
                self.elements[e2].col = col1;
            }
        }
    }

    /// `spcRowExchange` (topology only).
    fn row_exchange(&mut self, mut row1: usize, mut row2: usize) {
        if row1 > row2 {
            std::mem::swap(&mut row1, &mut row2);
        }
        let mut p1 = self.first_in_row[row1];
        let mut p2 = self.first_in_row[row2];
        while p1 != NONE || p2 != NONE {
            let (column, e1, e2);
            if p1 == NONE {
                column = self.elements[p2].col;
                (e1, e2) = (NONE, p2);
                p2 = self.elements[p2].next_in_row;
            } else if p2 == NONE || self.elements[p1].col < self.elements[p2].col {
                column = self.elements[p1].col;
                (e1, e2) = (p1, NONE);
                p1 = self.elements[p1].next_in_row;
            } else if self.elements[p1].col > self.elements[p2].col {
                column = self.elements[p2].col;
                (e1, e2) = (NONE, p2);
                p2 = self.elements[p2].next_in_row;
            } else {
                column = self.elements[p1].col;
                (e1, e2) = (p1, p2);
                p1 = self.elements[p1].next_in_row;
                p2 = self.elements[p2].next_in_row;
            }
            self.exchange_col_elements(row1, e1, row2, e2, column);
        }
        self.first_in_row.swap(row1, row2);
        self.int_to_ext_row.swap(row1, row2);
        self.ext_to_int_row[self.int_to_ext_row[row1]] = row1;
        self.ext_to_int_row[self.int_to_ext_row[row2]] = row2;
    }

    /// `spcColExchange` (topology only).
    fn col_exchange(&mut self, mut col1: usize, mut col2: usize) {
        if col1 > col2 {
            std::mem::swap(&mut col1, &mut col2);
        }
        let mut p1 = self.first_in_col[col1];
        let mut p2 = self.first_in_col[col2];
        while p1 != NONE || p2 != NONE {
            let (row, e1, e2);
            if p1 == NONE {
                row = self.elements[p2].row;
                (e1, e2) = (NONE, p2);
                p2 = self.elements[p2].next_in_col;
            } else if p2 == NONE || self.elements[p1].row < self.elements[p2].row {
                row = self.elements[p1].row;
                (e1, e2) = (p1, NONE);
                p1 = self.elements[p1].next_in_col;
            } else if self.elements[p1].row > self.elements[p2].row {
                row = self.elements[p2].row;
                (e1, e2) = (NONE, p2);
                p2 = self.elements[p2].next_in_col;
            } else {
                row = self.elements[p1].row;
                (e1, e2) = (p1, p2);
                p1 = self.elements[p1].next_in_col;
                p2 = self.elements[p2].next_in_col;
            }
            self.exchange_row_elements(col1, e1, col2, e2, row);
        }
        self.first_in_col.swap(col1, col2);
        self.int_to_ext_col.swap(col1, col2);
        self.ext_to_int_col[self.int_to_ext_col[col1]] = col1;
        self.ext_to_int_col[self.int_to_ext_col[col2]] = col2;
    }
}

/// Parity of the row/column permutation accumulated during ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PermutationParity {
    Even,
    Odd,
}

impl PermutationParity {
    const fn toggle(&mut self) {
        *self = match self {
            Self::Even => Self::Odd,
            Self::Odd => Self::Even,
        };
    }
}

/// Cached summaries of contiguous Markowitz-product blocks. They permit
/// skipping ranges that cannot contain the next reference-order candidate.
/// Dirty blocks are rebuilt on demand, never changing the candidate order.
#[derive(Debug, Clone, Copy)]
struct ProductSummary {
    min: i64,
    zeros: u64,
}

#[derive(Debug, Clone, Default)]
struct PivotProducts {
    values: Vec<i64>,
    summaries: Vec<Option<ProductSummary>>,
}

impl PivotProducts {
    const BLOCK: usize = u64::BITS as usize;
    // For a short search, walking the contiguous products is cheaper
    // than maintaining summaries. Amortize them over longer scans.
    const MIN_INDEXED_PRODUCTS: usize = Self::BLOCK * 8 + 2;

    fn new(len: usize) -> SpiceResult<Self> {
        Ok(Self {
            values: crate::primitives::filled_vec(len, 0, "SPARSE Markowitz products")?,
            summaries: if len > Self::MIN_INDEXED_PRODUCTS {
                crate::primitives::filled_vec(
                    len.div_ceil(Self::BLOCK),
                    None,
                    "SPARSE product summaries",
                )?
            } else {
                Vec::new()
            },
        })
    }

    fn set(&mut self, i: usize, value: i64) {
        if self.summaries.is_empty() {
            self.values[i] = value;
        } else if self.values[i] != value {
            self.values[i] = value;
            self.summaries[i / Self::BLOCK] = None;
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        let (x, y) = (self.values[a], self.values[b]);
        self.set(a, y);
        self.set(b, x);
    }

    fn summary(&mut self, block: usize) -> ProductSummary {
        *self.summaries[block].get_or_insert_with(|| {
            let start = block * Self::BLOCK;
            let values = &self.values[start..(start + Self::BLOCK).min(self.values.len())];
            let mut min = i64::MAX;
            let mut zeros = 0;
            for (i, &v) in values.iter().enumerate() {
                min = min.min(v);
                if v == 0 {
                    zeros |= 1_u64 << i;
                }
            }
            ProductSummary { min, zeros }
        })
    }

    /// Greatest index strictly before the cursor with product below limit.
    /// Sentinels ensure the search has a match in the C pivot routine.
    fn previous_below(&mut self, mut before: usize, limit: i64) -> Option<usize> {
        if self.summaries.is_empty() {
            return self.values[..before].iter().rposition(|&v| v < limit);
        }
        while before != 0 {
            let block = (before - 1) / Self::BLOCK;
            let start = block * Self::BLOCK;
            if (before - start != Self::BLOCK || self.summary(block).min < limit)
                && let Some(pos) = self.values[start..before].iter().rposition(|&v| v < limit)
            {
                return Some(start + pos);
            }
            before = start;
        }
        None
    }

    /// Greatest zero-product index strictly before the cursor.
    fn previous_zero(&mut self, mut before: usize) -> Option<usize> {
        if self.summaries.is_empty() {
            return self.values[..before].iter().rposition(|&v| v == 0);
        }
        while before != 0 {
            let block = (before - 1) / Self::BLOCK;
            let start = block * Self::BLOCK;
            let len = before - start;
            if len == Self::BLOCK {
                let mask = self.summary(block).zeros;
                if mask != 0 {
                    return Some(start + (u64::BITS - 1 - mask.leading_zeros()) as usize);
                }
            } else if let Some(pos) = self.values[start..before].iter().rposition(|&v| v == 0) {
                return Some(start + pos);
            }
            before = start;
        }
        None
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl std::ops::Index<usize> for PivotProducts {
    type Output = i64;
    fn index(&self, index: usize) -> &i64 {
        &self.values[index]
    }
}

/// The legal combinations of topology, symbolic ordering and numeric factors.
#[derive(Debug, Clone)]
enum Lifecycle {
    ColumnsOnly,
    Unordered,
    /// Reuse the numbering of a partially changed order after singularity.
    PartialOrder,
    Symbolic(std::sync::Arc<FactorPlan>),
    Factored(std::sync::Arc<FactorPlan>),
    /// Full ordering required, but the old traversal still describes the topology.
    ReorderRequired(std::sync::Arc<FactorPlan>),
}

impl Lifecycle {
    const fn rows_linked(&self) -> bool {
        !matches!(self, Self::ColumnsOnly)
    }

    const fn needs_ordering(&self) -> bool {
        matches!(
            self,
            Self::ColumnsOnly | Self::Unordered | Self::ReorderRequired(_)
        )
    }

    const fn plan(&self) -> Option<&std::sync::Arc<FactorPlan>> {
        match self {
            Self::Symbolic(plan) | Self::Factored(plan) | Self::ReorderRequired(plan) => Some(plan),
            Self::ColumnsOnly | Self::Unordered | Self::PartialOrder => None,
        }
    }

    fn invalidate_numeric(&mut self) {
        if matches!(self, Self::Factored(_))
            && let Self::Factored(plan) = std::mem::replace(self, Self::Unordered)
        {
            *self = Self::Symbolic(plan);
        }
    }

    fn into_loaded(mut self) -> Self {
        self.invalidate_numeric();
        self
    }

    fn invalidate_structure(&mut self, original: bool) {
        *self = match self {
            Self::ColumnsOnly => Self::ColumnsOnly,
            Self::Unordered | Self::ReorderRequired(_) => Self::Unordered,
            _ if original => Self::Unordered,
            _ => Self::PartialOrder,
        };
    }

    fn require_ordering(&mut self) {
        *self = match std::mem::replace(self, Self::Unordered) {
            Self::ColumnsOnly => Self::ColumnsOnly,
            Self::Unordered | Self::PartialOrder => Self::Unordered,
            Self::Symbolic(plan) | Self::Factored(plan) | Self::ReorderRequired(plan) => {
                Self::ReorderRequired(plan)
            }
        };
    }

    fn finish(&mut self, plan: Option<FactorPlan>) -> Result<(), Status> {
        let plan = if let Some(plan) = plan {
            std::sync::Arc::new(plan)
        } else {
            match std::mem::replace(self, Self::Unordered) {
                Self::Symbolic(plan) | Self::Factored(plan) | Self::ReorderRequired(plan) => plan,
                _ => return Err(Status::InvalidPlan),
            }
        };
        *self = Self::Factored(plan);
        Ok(())
    }
}

/// Diagnostic payload of the latest operation.
#[derive(Debug, Clone, Copy)]
enum Diagnostic {
    Okay,
    SmallPivot(usize, usize),
    ZeroDiag(usize, usize),
    Singular(usize, usize),
    NonFinite,
    InvalidPlan,
    NeedsReload,
    Resource,
}

impl Diagnostic {
    const fn status(self) -> Status {
        match self {
            Self::Okay => Status::Okay,
            Self::SmallPivot(..) => Status::SmallPivot,
            Self::ZeroDiag(..) => Status::ZeroDiag,
            Self::Singular(..) => Status::Singular,
            Self::NonFinite => Status::NonFinite,
            Self::InvalidPlan => Status::InvalidPlan,
            Self::NeedsReload => Status::NeedsReload,
            Self::Resource => Status::Resource,
        }
    }

    const fn singular(self) -> (usize, usize) {
        match self {
            Self::ZeroDiag(row, col) | Self::Singular(row, col) => (row, col),
            _ => (0, 0),
        }
    }

    const fn small_pivot(self) -> Option<(usize, usize)> {
        match self {
            Self::SmallPivot(row, col) => Some((row, col)),
            _ => None,
        }
    }
}

/// Scratch for an ordering attempt, returned to the frame for allocation reuse.
/// A snapshot never needs the previous operation's counts or search sentinels.
#[derive(Debug, Default)]
struct Markowitz {
    row: Vec<i64>,
    col: Vec<i64>,
    products: PivotProducts,
    singletons: i64,
}

impl Markowitz {
    fn ensure_size(&mut self, size: usize) -> SpiceResult<()> {
        if self.row.len() <= size {
            let rows = sparse_rows(size)?;
            let row = crate::primitives::filled_vec(rows, 0, "SPARSE Markowitz row counts")?;
            let col = crate::primitives::filled_vec(rows, 0, "SPARSE Markowitz column counts")?;
            let products = PivotProducts::new(rows + 1)?;
            self.row = row;
            self.col = col;
            self.products = products;
        }
        Ok(())
    }

    fn product_of(&self, i: usize) -> i64 {
        self.row[i].saturating_mul(self.col[i])
    }

    const fn track_singleton(&mut self, new: i64, old: i64) {
        if (new == 0) != (old == 0) {
            if old == 0 {
                self.singletons -= 1;
            } else {
                self.singletons += 1;
            }
        }
    }
}

/// A SPARSE matrix frame (`struct MatrixFrame`) with 1-based internal and
/// external indices; external index 0 (ground) is never used here.
#[derive(Debug)]
pub(super) struct Sparse<T> {
    /// Internal order (C `Size`): the rows translated so far. An external
    /// index never referenced by an element has no internal row; as in
    /// ngspice, its solution entry is left holding the right-hand side.
    size: usize,
    /// External order (C `ExtSize`, here fixed by [`Self::new`]).
    ext_size: usize,
    /// Stable handles index topology and numeric storage independently.
    /// Reordering changes links, never moves element handles.
    topology: std::sync::Arc<Topology>,
    values: Vec<T>,
    lifecycle: Lifecycle,
    ordering_scratch: Markowitz,
    rel_threshold: f64,
    abs_threshold: f64,
    parity: PermutationParity,
    diagnostic: Diagnostic,
}

impl<T: Clone> Clone for Sparse<T> {
    fn clone(&self) -> Self {
        Self {
            size: self.size,
            ext_size: self.ext_size,
            topology: self.topology.clone(),
            values: self.values.clone(),
            lifecycle: self.lifecycle.clone(),
            ordering_scratch: Markowitz::default(),
            rel_threshold: self.rel_threshold,
            abs_threshold: self.abs_threshold,
            parity: self.parity,
            diagnostic: self.diagnostic,
        }
    }
}

impl<T: Scalar> Sparse<T> {
    /// `spCreate(size)`.
    pub(crate) fn new(size: usize) -> SpiceResult<Self> {
        let rows = sparse_rows(size)?;
        let empty = || crate::primitives::filled_vec(rows, NONE, "SPARSE row index");
        let external = || {
            let mut indices = empty()?;
            indices[0] = 0;
            Ok::<_, crate::primitives::SpiceError>(indices)
        };
        Ok(Self {
            size: 0,
            ext_size: size,
            topology: std::sync::Arc::new(Topology {
                elements: Vec::new(),
                diag: empty()?,
                first_in_col: empty()?,
                first_in_row: empty()?,
                ext_to_int_row: external()?,
                ext_to_int_col: external()?,
                int_to_ext_row: sparse_indices(size)?,
                int_to_ext_col: sparse_indices(size)?,
            }),
            values: Vec::new(),
            lifecycle: Lifecycle::ColumnsOnly,
            ordering_scratch: Markowitz::default(),
            rel_threshold: DEFAULT_THRESHOLD,
            abs_threshold: 0.0,
            parity: PermutationParity::Even,
            diagnostic: Diagnostic::Okay,
        })
    }

    /// Whether every external index has been translated (no empty unknown).
    pub(crate) const fn fully_translated(&self) -> bool {
        self.size == self.ext_size
    }

    /// External indices (1-based) that no element references, so have no
    /// internal row (C leaves them out of the matrix; see [`Self::solve_with_workspace`]).
    pub(crate) fn untranslated(&self) -> impl Iterator<Item = usize> + '_ {
        (1..=self.ext_size).filter(|&ext| self.topology.ext_to_int_row[ext] == NONE)
    }

    /// External row and column of the last singular/zero pivot.
    pub(crate) const fn where_singular(&self) -> (usize, usize) {
        self.diagnostic.singular()
    }

    /// External row and column of the first small pivot of the last
    /// ordering (status [`Status::SmallPivot`]), if any.
    pub(crate) const fn small_pivot(&self) -> Option<(usize, usize)> {
        self.diagnostic.small_pivot()
    }

    fn mag(&self, e: usize) -> f64 {
        self.values[e].mag()
    }

    // --- spbuild.c ------------------------------------------------------

    /// `spGetElement(ext_row, ext_col)`: the element handle, created if
    /// missing. Indices are external and 1-based.
    pub(super) fn get_element(&mut self, row: usize, col: usize) -> SpiceResult<SolverElement> {
        self.check_coordinate(row, col)?;
        if self.topology.ext_to_int_row[row] == NONE || self.topology.ext_to_int_col[col] == NONE {
            // Reserve the original's storage before publishing new translations.
            self.reserve_element_storage()?;
        }
        let (row, col) = self.translate(row, col);
        if row == col && self.topology.diag[row] != NONE {
            return Ok(SolverElement(self.topology.diag[row]));
        }
        let (last, found) = self.topology.locate_in_col(Slot::Head(col), row);
        if found != NONE {
            return Ok(SolverElement(found));
        }
        self.create_element(row, col, last, false)
            .map(SolverElement)
    }

    fn check_coordinate(&self, row: usize, col: usize) -> SpiceResult<()> {
        if row == 0 || col == 0 || row > self.ext_size || col > self.ext_size {
            return Err(numerical(
                "SPARSE construction",
                format!(
                    "external coordinate ({row}, {col}) outside 1..={}",
                    self.ext_size
                ),
            ));
        }
        Ok(())
    }

    /// Adds using the stable handle; no coordinate conversion in the load.
    pub(super) fn add(&mut self, element: SolverElement, value: T)
    where
        T: std::ops::Add<Output = T>,
    {
        self.note_numeric_write();
        let v = &mut self.values[element.0];
        *v = *v + value;
    }

    /// `Translate`: assigns internal indices in first-reference order.
    fn translate(&mut self, ext_row: usize, ext_col: usize) -> (usize, usize) {
        let row = self.topology.ext_to_int_row[ext_row];
        let col = self.topology.ext_to_int_col[ext_col];
        if row != NONE && col != NONE {
            return (row, col);
        }
        std::sync::Arc::make_mut(&mut self.topology).translate(&mut self.size, ext_row, ext_col)
    }

    /// `spcFindElementInCol`.
    fn find_element_in_col(&self, last: Slot, row: usize) -> usize {
        self.topology.locate_in_col(last, row).1
    }

    fn reserve_element_storage(&mut self) -> SpiceResult<()> {
        let topology = std::sync::Arc::make_mut(&mut self.topology);
        if topology.elements.len() == topology.elements.capacity() {
            sparse_reserve(&mut topology.elements, 1)?;
        }
        if self.values.len() == self.values.capacity() {
            sparse_reserve(&mut self.values, 1)?;
        }
        Ok(())
    }

    /// `spcCreateElement`. Reserve both arenas before mutating links.
    fn create_element(
        &mut self,
        row: usize,
        col: usize,
        last: Slot,
        fillin: bool,
    ) -> SpiceResult<usize> {
        self.reserve_element_storage()?;
        let topology = std::sync::Arc::make_mut(&mut self.topology);
        let e = topology.create_element(row, col, last, self.lifecycle.rows_linked());
        if !fillin {
            self.note_numeric_write();
        }
        self.lifecycle.invalidate_structure(!fillin);
        self.values.push(T::ZERO);
        Ok(e)
    }

    /// `spcLinkRows`.
    fn link_rows(&mut self) {
        let topology = std::sync::Arc::make_mut(&mut self.topology);
        for col in (1..=self.size).rev() {
            let mut e = topology.first_in_col[col];
            while e != NONE {
                let row = topology.elements[e].row;
                topology.elements[e].col = col;
                topology.elements[e].next_in_row = topology.first_in_row[row];
                topology.first_in_row[row] = e;
                e = topology.elements[e].next_in_col;
            }
        }
        self.lifecycle = Lifecycle::Unordered;
    }

    /// `spClear`: zero every value, keeping structure and pivot order.
    pub(crate) fn clear(&mut self) {
        self.values.fill(T::ZERO);
        self.reset_numeric_status();
    }

    fn reset_numeric_status(&mut self) {
        self.lifecycle.invalidate_numeric();
        self.diagnostic = Diagnostic::Okay;
    }

    /// A write into existing LU storage revokes its factor view but does
    /// not restore the original coefficients. Likewise, stamping into a
    /// failed partial LU cannot repair it; clear/reload must happen first.
    fn note_numeric_write(&mut self) {
        if self.is_factored() {
            self.lifecycle.invalidate_numeric();
            self.diagnostic = Diagnostic::NeedsReload;
        }
    }

    const fn coefficients_loaded(&self) -> bool {
        !self.is_factored() && matches!(self.diagnostic, Diagnostic::Okay)
    }

    /// Build a fresh frame from unique, already assembled original
    /// elements in reference creation order (external indices are 1-based).
    /// Borrow the exclusively owned topology once and reserve both arenas.
    pub(super) fn from_originals(
        size: usize,
        entries: impl IntoIterator<Item = (usize, usize, T)>,
    ) -> SpiceResult<Self> {
        let mut sparse = Self::new(size)?;
        let entries = entries.into_iter();
        let capacity = entries.size_hint().0;
        let topology = std::sync::Arc::make_mut(&mut sparse.topology);
        sparse_reserve(&mut topology.elements, capacity)?;
        sparse_reserve(&mut sparse.values, capacity)?;
        let mut tails =
            crate::primitives::filled_vec(sparse_rows(size)?, NONE, "SPARSE column tails")?;
        for (row, col, value) in entries {
            if row == 0 || col == 0 || row > size || col > size {
                return Err(numerical(
                    "SPARSE construction",
                    format!("external coordinate ({row}, {col}) outside 1..={size}"),
                ));
            }
            if topology.elements.len() == topology.elements.capacity() {
                sparse_reserve(&mut topology.elements, 1)?;
            }
            if sparse.values.len() == sparse.values.capacity() {
                sparse_reserve(&mut sparse.values, 1)?;
            }
            let (r, c) = topology.translate(&mut sparse.size, row, col);
            let handle = topology.append_original(r, c, &mut tails);
            debug_assert_eq!(handle, sparse.values.len());
            sparse.values.push(value);
        }
        topology.finish_original_columns(&tails[..=sparse.size])?;
        Ok(sparse)
    }

    /// A pattern already contains its unique positions in creation order.
    pub(super) fn from_pattern(pattern: &crate::maths::Pattern, values: &[T]) -> SpiceResult<Self> {
        if pattern.len() != values.len() {
            return Err(numerical(
                "SPARSE construction",
                "pattern and coefficient count differ",
            ));
        }
        Self::from_originals(
            pattern.order(),
            pattern
                .positions()
                .zip(values)
                .map(|((r, c), &value)| (r + 1, c + 1, value)),
        )
    }

    /// Reload consecutive originals, clearing only the fill-in suffix.
    pub(super) fn reload_originals(&mut self, values: &[T]) {
        let (originals, fillins) = self.values.split_at_mut(values.len());
        originals.copy_from_slice(values);
        fillins.fill(T::ZERO);
        self.reset_numeric_status();
    }

    /// Reload from a folded stream without a temporary numeric vector.
    pub(super) fn reload_originals_iter(&mut self, values: impl ExactSizeIterator<Item = T>) {
        let (originals, fillins) = self.values.split_at_mut(values.len());
        for (out, value) in originals.iter_mut().zip(values) {
            *out = value;
        }
        fillins.fill(T::ZERO);
        self.reset_numeric_status();
    }

    /// Transfer one scalar frame's ordering; numeric coefficients are
    /// supplied by the caller, never copied from the old LU.
    fn into_retyped_values<U: Scalar>(self, values: Vec<U>) -> Sparse<U> {
        Sparse {
            size: self.size,
            ext_size: self.ext_size,
            values,
            topology: self.topology,
            lifecycle: self.lifecycle.into_loaded(),
            ordering_scratch: self.ordering_scratch,
            rel_threshold: self.rel_threshold,
            abs_threshold: self.abs_threshold,
            parity: self.parity,
            diagnostic: Diagnostic::Okay,
        }
    }

    /// Fork the structural state, discarding scratch and old factors.
    fn fork_values<U: Scalar>(&self, values: Vec<U>) -> Sparse<U> {
        Sparse {
            size: self.size,
            ext_size: self.ext_size,
            topology: self.topology.clone(),
            values,
            lifecycle: self.lifecycle.clone().into_loaded(),
            ordering_scratch: Markowitz::default(),
            rel_threshold: self.rel_threshold,
            abs_threshold: self.abs_threshold,
            parity: self.parity,
            diagnostic: Diagnostic::Okay,
        }
    }

    /// Construct the complete numeric buffer directly. The original
    /// handles are consecutive; only the fill-in suffix needs zeroing.
    fn loaded_values<U: Scalar>(&self, originals: impl ExactSizeIterator<Item = U>) -> Vec<U> {
        let total = self.values.len();
        assert!(
            originals.len() <= total,
            "more originals than sparse elements"
        );
        let mut values = Vec::with_capacity(total);
        values.extend(originals);
        values.resize(total, U::ZERO);
        values
    }

    pub(super) fn fork_loaded<U: Scalar>(
        &self,
        originals: impl ExactSizeIterator<Item = U>,
    ) -> Sparse<U> {
        let mut next = self.fork_values(self.loaded_values(originals));
        next.reset_numeric_status();
        next
    }

    /// Reload in place if the factor buffer is uniquely held, otherwise
    /// fork directly from the newly assembled values. No old LU or blanket
    /// zero pass is needed on the retained-snapshot path.
    pub(super) fn reload_shared(
        shared: &mut std::sync::Arc<Self>,
        originals: impl ExactSizeIterator<Item = T>,
    ) {
        if let Some(owned) = std::sync::Arc::get_mut(shared) {
            owned.reload_originals_iter(originals);
        } else {
            *shared = std::sync::Arc::new(shared.fork_loaded(originals));
        }
    }

    /// Switch scalar type, moving unique ordering storage or sharing it
    /// with a retained snapshot. The blank form is used before matching a
    /// real matrix against the cached order.
    pub(super) fn retype_shared<U: Scalar>(shared: std::sync::Arc<Self>) -> Sparse<U> {
        let values = vec![U::ZERO; shared.values.len()];
        match std::sync::Arc::try_unwrap(shared) {
            Ok(owned) => owned.into_retyped_values(values),
            Err(retained) => retained.fork_values(values),
        }
    }

    /// Retype and load in a single pass when the new assembly is known.
    pub(super) fn retype_shared_loaded<U: Scalar>(
        shared: std::sync::Arc<Self>,
        originals: impl ExactSizeIterator<Item = U>,
    ) -> Sparse<U> {
        let values = shared.loaded_values(originals);
        let mut next = match std::sync::Arc::try_unwrap(shared) {
            Ok(owned) => owned.into_retyped_values(values),
            Err(retained) => retained.fork_values(values),
        };
        next.reset_numeric_status();
        next
    }

    #[cfg(test)]
    fn into_retyped<U: Scalar>(self) -> Sparse<U> {
        let values = vec![U::ZERO; self.values.len()];
        self.into_retyped_values(values)
    }

    /// `LoadGmin` (`spsmp.c`): adds `gmin` to the real part of every
    /// internal diagonal element (after the MNA preorder's column swaps, so a
    /// voltage source's twin can be one), from the last row to the first.
    pub(crate) fn load_gmin(&mut self, gmin: f64) {
        if gmin == 0.0 {
            return;
        }
        self.note_numeric_write();
        for i in (1..=self.size).rev() {
            let diag = self.topology.diag[i];
            if diag != NONE {
                self.values[diag] = self.values[diag].add_real(gmin);
            }
        }
    }

    pub(super) fn values_are_finite(&self) -> bool {
        self.values.iter().all(|value| value.is_finite())
    }

    /// True only when a direct numerical refactor can use the established
    /// pivot order, rather than implicitly running the ordering algorithm.
    pub(super) const fn can_refactor(&self) -> bool {
        matches!(
            self.lifecycle,
            Lifecycle::Symbolic(_) | Lifecycle::Factored(_)
        )
    }

    /// Whether the frame holds LU factors (`Matrix->Factored`).
    pub(crate) const fn is_factored(&self) -> bool {
        matches!(self.lifecycle, Lifecycle::Factored(_))
    }

    /// Overwrites one element's value (`*element = value`).
    pub(super) fn set(&mut self, element: SolverElement, value: T) {
        self.note_numeric_write();
        self.values[element.0] = value;
    }

    /// One element's value.
    pub(super) fn value(&self, element: SolverElement) -> T {
        self.values[element.0]
    }

    /// `SMPcAddCol` (`spsmp.c`): adds every element of external column
    /// `addend` (fill-ins included) to the same row of column `accum`,
    /// creating the missing elements there as originals (which, once rows
    /// are linked, forces the next factorization to reorder). Untranslated
    /// columns are left alone.
    pub(crate) fn add_col(&mut self, accum: usize, addend: usize) -> SpiceResult<()>
    where
        T: std::ops::Add<Output = T>,
    {
        self.check_coordinate(accum, addend)?;
        let (accum, addend) = (
            self.topology.ext_to_int_col[accum],
            self.topology.ext_to_int_col[addend],
        );
        if accum == NONE || addend == NONE {
            return Ok(());
        }
        let mut next = self.topology.first_in_col[addend];
        let mut prev = Slot::Head(accum);
        let mut acc = self.topology.col_get(prev);
        while next != NONE {
            let row = self.topology.elements[next].row;
            while acc != NONE && self.topology.elements[acc].row < row {
                prev = Slot::After(acc);
                acc = self.topology.elements[acc].next_in_col;
            }
            if acc == NONE || self.topology.elements[acc].row > row {
                acc = self.create_element(row, accum, prev, false)?;
            }
            let value = self.values[next];
            self.add(SolverElement(acc), value);
            next = self.topology.elements[next].next_in_col;
        }
        Ok(())
    }

    /// `SMPcZeroCol` (`spsmp.c`): zeroes every element of external column
    /// `col`, keeping them in the structure.
    pub(crate) fn zero_col(&mut self, col: usize) -> SpiceResult<()> {
        self.check_coordinate(col, col)?;
        let col = self.topology.ext_to_int_col[col];
        if col == NONE {
            return Ok(());
        }
        self.note_numeric_write();
        let mut e = self.topology.first_in_col[col];
        while e != NONE {
            self.values[e] = T::ZERO;
            e = self.topology.elements[e].next_in_col;
        }
        Ok(())
    }

    /// `spMNA_Preorder`: swap columns so structural `±1` twins from voltage
    /// sources land on the diagonal before ordering.
    pub(crate) fn mna_preorder(&mut self) {
        if self.lifecycle.rows_linked() {
            return;
        }
        let size = self.size;
        let mut start_at = 1;
        loop {
            let mut another_pass = false;
            let mut swapped = false;
            // C's `StartAt` changes only take effect on the next pass.
            // Snapshot the range bound so changes within the pass cannot
            // affect its current iteration range.
            let pass_start = start_at;
            for j in pass_start..=size {
                if self.topology.diag[j] == NONE {
                    let (twins, t1, t2) = self.count_twins(j);
                    if twins == 1 {
                        self.swap_cols(t1, t2);
                        swapped = true;
                    } else if twins > 1 && !another_pass {
                        another_pass = true;
                        start_at = j;
                    }
                }
            }
            if another_pass {
                let mut j = start_at;
                while !swapped && j <= size {
                    if self.topology.diag[j] == NONE {
                        let (_, t1, t2) = self.count_twins(j);
                        self.swap_cols(t1, t2);
                        swapped = true;
                    }
                    j += 1;
                }
            }
            if !another_pass {
                break;
            }
        }
    }

    /// `CountTwins`: the twin pair found is the last one before a second.
    fn count_twins(&mut self, col: usize) -> (usize, usize, usize) {
        let topology = std::sync::Arc::make_mut(&mut self.topology);
        let mut twins = 0;
        let (mut out1, mut out2) = (NONE, NONE);
        let mut t1 = topology.first_in_col[col];
        while t1 != NONE {
            if self.values[t1].real().abs() == 1.0 {
                let row = topology.elements[t1].row;
                let mut t2 = topology.first_in_col[row];
                while t2 != NONE && topology.elements[t2].row != col {
                    t2 = topology.elements[t2].next_in_col;
                }
                if t2 != NONE && self.values[t2].real().abs() == 1.0 {
                    twins += 1;
                    if twins >= 2 {
                        return (twins, out1, out2);
                    }
                    topology.elements[t1].col = col;
                    topology.elements[t2].col = row;
                    out1 = t1;
                    out2 = t2;
                }
            }
            t1 = topology.elements[t1].next_in_col;
        }
        (twins, out1, out2)
    }

    /// `SwapCols`.
    fn swap_cols(&mut self, t1: usize, t2: usize) {
        let topology = std::sync::Arc::make_mut(&mut self.topology);
        let (c1, c2) = (topology.elements[t1].col, topology.elements[t2].col);
        topology.first_in_col.swap(c1, c2);
        topology.int_to_ext_col.swap(c1, c2);
        topology.ext_to_int_col[topology.int_to_ext_col[c2]] = c2;
        topology.ext_to_int_col[topology.int_to_ext_col[c1]] = c1;
        topology.diag[c1] = t2;
        topology.diag[c2] = t1;
        self.parity.toggle();
    }

    // --- spfactor.c -----------------------------------------------------

    /// `spOrderAndFactor(Matrix, NULL, rel, abs, DiagPivoting = YES)`.
    pub(crate) fn order_and_factor(&mut self, rel_threshold: f64, abs_threshold: f64) -> Status {
        if !self.coefficients_loaded() {
            return Status::NeedsReload;
        }
        let mut scratch = std::mem::take(&mut self.ordering_scratch);
        let status = self.order_with_scratch(rel_threshold, abs_threshold, &mut scratch);
        self.ordering_scratch = scratch;
        status
    }

    fn order_with_scratch(
        &mut self,
        rel_threshold: f64,
        abs_threshold: f64,
        scratch: &mut Markowitz,
    ) -> Status {
        if !self.values_are_finite() {
            return self.nonfinite_factor();
        }
        let size = self.size;
        if rel_threshold > 0.0 && rel_threshold <= 1.0 {
            self.rel_threshold = rel_threshold;
        }
        if abs_threshold >= 0.0 {
            self.abs_threshold = abs_threshold;
        }
        let mut step = 1;
        let mut reorder = false;
        if self.lifecycle.needs_ordering() {
            if !self.lifecycle.rows_linked() {
                self.link_rows();
            }
        } else {
            // A failed partial reorder may have invalidated the plan.
            // Retain its numbering like C. The accepted prefix needs
            // no Markowitz counts; compute them only if a search begins.
            while step <= size {
                let pivot = self.topology.diag[step];
                if pivot == NONE {
                    reorder = true;
                    break;
                }
                let largest = self.find_largest_in_col(self.topology.elements[pivot].next_in_col);
                if largest * self.rel_threshold < self.mag(pivot) {
                    self.row_col_elimination(None, pivot);
                    if matches!(
                        self.diagnostic.status(),
                        Status::Singular | Status::NonFinite | Status::Resource
                    ) {
                        return self.diagnostic.status();
                    }
                } else {
                    reorder = true;
                    break;
                }
                step += 1;
            }
            if !reorder {
                return self.done();
            }
        }
        // Reordering always rebuilds all active counts. A snapshot fork
        // needs no copy of the previous ordering's scratch arrays.
        if scratch.ensure_size(size).is_err() {
            return self.resource_failure();
        }
        self.count_markowitz(scratch, step);
        self.markowitz_products(scratch, step);
        while step <= size {
            let Some(pivot) = self.search_for_pivot(scratch, step) else {
                return self.matrix_is_singular(step);
            };
            self.exchange_rows_and_cols(scratch, pivot, step);
            self.row_col_elimination(Some(scratch), pivot);
            if matches!(
                self.diagnostic.status(),
                Status::Singular | Status::NonFinite | Status::Resource
            ) {
                return self.diagnostic.status();
            }
            self.update_markowitz_numbers(scratch, pivot);
            step += 1;
        }
        self.done()
    }

    fn done(&mut self) -> Status {
        let new = if self.lifecycle.plan().is_none() {
            match self.compile_plan() {
                Ok(plan) => Some(plan),
                Err(Status::Resource) => return self.resource_failure(),
                Err(_) => return self.invalid_plan(),
            }
        } else {
            None
        };
        if self.lifecycle.finish(new).is_err() {
            return self.invalid_plan();
        }
        self.diagnostic.status()
    }

    fn invalid_plan(&mut self) -> Status {
        self.lifecycle.require_ordering();
        self.diagnostic = Diagnostic::InvalidPlan;
        Status::InvalidPlan
    }

    fn compile_plan(&self) -> Result<FactorPlan, Status> {
        // Offsets include the exclusive end of the entry array, so both
        // counts (not just the largest existing handles) must fit.
        if u32::try_from(self.size).is_ok() && u32::try_from(self.topology.elements.len()).is_ok() {
            self.compile_runs().map(FactorPlan::Compact)
        } else {
            self.compile_runs().map(FactorPlan::Wide)
        }
    }

    fn compile_runs<I: PlanIndex>(&self) -> Result<FactorRuns<I>, Status> {
        let elements = &self.topology.elements;
        if self.values.len() != elements.len() || !self.lifecycle.rows_linked() {
            return Err(Status::InvalidPlan);
        }
        let mut columns = Vec::new();
        columns
            .try_reserve_exact(self.size)
            .map_err(|_| Status::Resource)?;
        let mut column_entries = Vec::new();
        column_entries
            .try_reserve_exact(self.topology.elements.len())
            .map_err(|_| Status::Resource)?;
        let mut upper_row_start = Vec::new();
        upper_row_start
            .try_reserve_exact(self.size.checked_add(1).ok_or(Status::Resource)?)
            .map_err(|_| Status::Resource)?;
        let mut upper_row_entries = Vec::new();

        let mut column_upper_count = 0;
        upper_row_start.push(I::pack(0).ok_or(Status::InvalidPlan)?);
        for i in 1..=self.size {
            let start = column_entries.len();
            let diagonal = *self.topology.diag.get(i).ok_or(Status::InvalidPlan)?;
            let mut diagonal_offset = None;
            let mut e = *self
                .topology
                .first_in_col
                .get(i)
                .ok_or(Status::InvalidPlan)?;
            let mut previous = 0;
            while e != NONE {
                let element = elements.get(e).ok_or(Status::InvalidPlan)?;
                // Strict coordinate order also rules out cyclic links and
                // duplicate handles, before constructing packed offsets.
                if element.col != i || element.row <= previous || element.row > self.size {
                    return Err(Status::InvalidPlan);
                }
                previous = element.row;
                if element.row == i {
                    if e != diagonal {
                        return Err(Status::InvalidPlan);
                    }
                    diagonal_offset = Some(column_entries.len());
                } else if element.row < i {
                    column_upper_count += 1;
                }
                if column_entries.len() == elements.len() {
                    return Err(Status::InvalidPlan);
                }
                column_entries.push(RunEntry::new(e, element.row).ok_or(Status::InvalidPlan)?);
                e = element.next_in_col;
            }
            columns.push(
                ColumnRun::new(
                    start,
                    diagonal_offset.ok_or(Status::InvalidPlan)?,
                    column_entries.len(),
                )
                .ok_or(Status::InvalidPlan)?,
            );
            let mut e = elements
                .get(diagonal)
                .ok_or(Status::InvalidPlan)?
                .next_in_row;
            let mut previous = i;
            while e != NONE {
                let element = elements.get(e).ok_or(Status::InvalidPlan)?;
                if element.row != i || element.col <= previous || element.col > self.size {
                    return Err(Status::InvalidPlan);
                }
                previous = element.col;
                if upper_row_entries.len() == upper_row_entries.capacity() {
                    upper_row_entries
                        .try_reserve(1)
                        .map_err(|_| Status::Resource)?;
                }
                upper_row_entries.push(RunEntry::new(e, element.col).ok_or(Status::InvalidPlan)?);
                e = element.next_in_row;
            }
            upper_row_start.push(I::pack(upper_row_entries.len()).ok_or(Status::InvalidPlan)?);
        }
        if column_entries.len() != elements.len() || upper_row_entries.len() != column_upper_count {
            return Err(Status::InvalidPlan);
        }
        Ok(FactorRuns {
            columns,
            column_entries,
            upper_row_start,
            upper_row_entries,
        })
    }

    #[cfg(test)]
    fn factor(&mut self) -> Status {
        self.factor_with_workspace(&mut Vec::new())
    }

    /// `spFactor`: refactor with the existing pivot order (no threshold
    /// test). Scratch belongs to the operation, not to factor snapshots.
    pub(super) fn factor_with_workspace(&mut self, workspace: &mut Vec<T>) -> Status {
        // C retains a partial order after a singular reordering. If
        // topology changed, no complete numeric traversal is available;
        // retry the threshold path, preserving C's existing pivot prefix.
        if !self.coefficients_loaded() {
            return Status::NeedsReload;
        }
        let Lifecycle::Symbolic(plan) = &self.lifecycle else {
            return self.order_and_factor(0.0, 0.0);
        };
        let size = self.size;
        if size == 0 {
            return self.done();
        }
        let first = self.topology.diag[1];
        if self.values[first].is_zero() {
            return self.refactor_zero_pivot(1);
        }
        let inverse = self.values[first].recip();
        if !inverse.is_finite() || inverse.is_zero() {
            return self.nonfinite_factor();
        }
        self.values[first] = inverse;
        // Each column is scattered before use. Successful ordering
        // inserted every structural elimination target as a fill-in, so
        // the lower-column updates cannot read an unscattered row.
        // Existing scratch may hold the previous solve's internal RHS;
        // no blanket initialization is required.
        let additional = (size + 1).saturating_sub(workspace.len());
        if sparse_reserve(workspace, additional).is_err() {
            return self.resource_failure();
        }
        workspace.resize(size + 1, T::ZERO);
        let dest = workspace;
        let result = match plan.as_ref() {
            FactorPlan::Compact(runs) => runs.refactor(&mut self.values, &self.topology.diag, dest),
            FactorPlan::Wide(runs) => runs.refactor(&mut self.values, &self.topology.diag, dest),
        };
        match result {
            Ok(()) => {}
            Err(RefactorIssue::ZeroPivot(step)) => return self.refactor_zero_pivot(step),
            Err(RefactorIssue::NonFinite) => return self.nonfinite_factor(),
        }
        // Fixed-order factorization does not compare coefficient magnitudes
        // to select pivots. Validate the stored factors in one contiguous
        // pass, including the untouched subdiagonal entries of column one.
        // Every computed entry is stored; a nonfinite pivot is rejected above.
        if !self.values_are_finite() {
            return self.nonfinite_factor();
        }
        self.done()
    }

    fn refactor_zero_pivot(&mut self, step: usize) -> Status {
        // Preserve the distinction between singular and nonfinite input
        // even when an early zero pivot stops the factorization.
        if !self.values_are_finite() {
            return self.nonfinite_factor();
        }
        self.zero_pivot(step)
    }

    /// `CountMarkowitz` (no RHS).
    fn count_markowitz(&self, scratch: &mut Markowitz, step: usize) {
        for i in step..=self.size {
            let mut count = -1;
            let mut e = self.topology.first_in_row[i];
            while e != NONE && self.topology.elements[e].col < step {
                e = self.topology.elements[e].next_in_row;
            }
            while e != NONE {
                count += 1;
                e = self.topology.elements[e].next_in_row;
            }
            scratch.row[i] = count;
        }
        for i in step..=self.size {
            let mut count = -1;
            let mut e = self.topology.first_in_col[i];
            while e != NONE && self.topology.elements[e].row < step {
                e = self.topology.elements[e].next_in_col;
            }
            while e != NONE {
                count += 1;
                e = self.topology.elements[e].next_in_col;
            }
            scratch.col[i] = count;
        }
    }

    /// `MarkowitzProducts`.
    fn markowitz_products(&self, scratch: &mut Markowitz, step: usize) {
        scratch.singletons = 0;
        for i in step..=self.size {
            let product = scratch.row[i].saturating_mul(scratch.col[i]);
            scratch.products.set(i, product);
            if product == 0 {
                scratch.singletons += 1;
            }
        }
    }

    /// `SearchForPivot` with diagonal pivoting.
    fn search_for_pivot(&mut self, scratch: &mut Markowitz, step: usize) -> Option<usize> {
        if scratch.singletons != 0
            && let Some(pivot) = self.search_for_singleton(scratch, step)
        {
            return Some(pivot);
        }
        if let Some(pivot) = self.quickly_search_diagonal(scratch, step) {
            return Some(pivot);
        }
        if let Some(pivot) = self.search_diagonal(scratch, step) {
            return Some(pivot);
        }
        self.search_entire_matrix(scratch, step)
    }

    fn acceptable(&self, pivot: usize, step: usize) -> bool {
        let mag = self.mag(pivot);
        mag > self.abs_threshold
            && mag > self.rel_threshold * self.find_biggest_in_col_exclude(pivot, step)
    }

    /// `SearchForSingleton`.
    fn search_for_singleton(&self, scratch: &mut Markowitz, step: usize) -> Option<usize> {
        let size = self.size;
        scratch.products.set(size + 1, scratch.products[step]);
        let mut singletons = scratch.singletons;
        scratch.singletons -= 1;
        scratch.products.set(step - 1, 0);
        // C advances a pointer *after* testing each product. Track the
        // next index plus one instead: the first read is size + 1 and each
        // subsequent read moves left, preserving C's sentinel and pivot
        // priority without an isize that can become -1 at step == 1.
        let mut cursor = size + 2;
        while singletons > 0 {
            singletons -= 1;
            let Some(next) = scratch.products.previous_zero(cursor) else {
                break;
            };
            cursor = next;
            let mut i = cursor;
            if i < step {
                break;
            }
            if i > size {
                i = step;
            }
            let diag = self.topology.diag[i];
            if diag == NONE {
                // ngspice's off-diagonal singleton branches `break` (or would
                // dereference NULL) when no diagonal exists.
                break;
            }
            if self.acceptable(diag, step) {
                return Some(diag);
            }
        }
        scratch.singletons += 1;
        None
    }

    /// `QuicklySearchDiagonal` (without `MODIFIED_MARKOWITZ`).
    fn quickly_search_diagonal(&self, scratch: &mut Markowitz, step: usize) -> Option<usize> {
        let size = self.size;
        let mut chosen = NONE;
        let mut min = i64::MAX;
        scratch.products.set(size + 1, scratch.products[step]);
        scratch.products.set(step - 1, -1);
        let mut p = size + 2;
        while let Some(next) = scratch.products.previous_below(p, min) {
            p = next;
            let mut i = p;
            if i < step {
                break;
            }
            if i > size {
                i = step;
            }
            let diag = self.topology.diag[i];
            if diag == NONE {
                continue;
            }
            let magnitude = self.mag(diag);
            if magnitude <= self.abs_threshold {
                continue;
            }
            if scratch.products[p] == 1 {
                let mut other_row = self.topology.elements[diag].next_in_row;
                let mut other_col = self.topology.elements[diag].next_in_col;
                if other_row == NONE && other_col == NONE {
                    other_row = self.topology.first_in_row[i];
                    while other_row != NONE {
                        let c = self.topology.elements[other_row].col;
                        if c >= step && c != i {
                            break;
                        }
                        other_row = self.topology.elements[other_row].next_in_row;
                    }
                    other_col = self.topology.first_in_col[i];
                    while other_col != NONE {
                        let r = self.topology.elements[other_col].row;
                        if r >= step && r != i {
                            break;
                        }
                        other_col = self.topology.elements[other_col].next_in_col;
                    }
                }
                if other_row != NONE
                    && other_col != NONE
                    && self.topology.elements[other_row].col
                        == self.topology.elements[other_col].row
                {
                    let largest_off = self.mag(other_row).max(self.mag(other_col));
                    if magnitude >= largest_off {
                        return Some(diag);
                    }
                }
            }
            min = scratch.products[p];
            chosen = diag;
        }
        if chosen != NONE {
            let largest = self.find_biggest_in_col_exclude(chosen, step);
            if self.mag(chosen) <= self.rel_threshold * largest {
                return None;
            }
            return Some(chosen);
        }
        None
    }

    /// `SearchDiagonal`.
    fn search_diagonal(&self, scratch: &mut Markowitz, step: usize) -> Option<usize> {
        let size = self.size;
        let mut chosen = NONE;
        let mut min = i64::MAX;
        let mut ties = 0;
        let mut accepted_ratio = 0.0;
        scratch.products.set(size + 1, scratch.products[step]);
        for j in (step + 1..=size + 1).rev() {
            let product = scratch.products[j];
            if product > min {
                continue;
            }
            let i = if j > size { step } else { j };
            let diag = self.topology.diag[i];
            if diag == NONE {
                continue;
            }
            let magnitude = self.mag(diag);
            if magnitude <= self.abs_threshold {
                continue;
            }
            let largest = self.find_biggest_in_col_exclude(diag, step);
            if magnitude <= self.rel_threshold * largest {
                continue;
            }
            if product < min {
                chosen = diag;
                min = product;
                accepted_ratio = largest / magnitude;
                ties = 0;
            } else {
                ties += 1;
                let ratio = largest / magnitude;
                if ratio < accepted_ratio {
                    chosen = diag;
                    accepted_ratio = ratio;
                }
                if ties >= min.saturating_mul(TIES_MULTIPLIER) {
                    return Some(chosen);
                }
            }
        }
        (chosen != NONE).then_some(chosen)
    }

    /// `SearchEntireMatrix`. Sets `SmallPivot`/`Singular` like C.
    fn search_entire_matrix(&mut self, scratch: &Markowitz, step: usize) -> Option<usize> {
        let mut chosen = NONE;
        let mut largest_element = NONE;
        let mut largest_mag = 0.0;
        let mut min = i64::MAX;
        let mut ties = 0;
        let mut accepted_ratio = 0.0;
        for i in step..=self.size {
            let mut e = self.topology.first_in_col[i];
            while e != NONE && self.topology.elements[e].row < step {
                e = self.topology.elements[e].next_in_col;
            }
            let largest_in_col = self.find_largest_in_col(e);
            if largest_in_col == 0.0 {
                continue;
            }
            while e != NONE {
                let magnitude = self.mag(e);
                if magnitude > largest_mag {
                    largest_mag = magnitude;
                    largest_element = e;
                }
                let element = &self.topology.elements[e];
                let product = scratch.row[element.row].saturating_mul(scratch.col[element.col]);
                if product <= min
                    && magnitude > self.rel_threshold * largest_in_col
                    && magnitude > self.abs_threshold
                {
                    if product < min {
                        chosen = e;
                        min = product;
                        accepted_ratio = largest_in_col / magnitude;
                        ties = 0;
                    } else {
                        ties += 1;
                        let ratio = largest_in_col / magnitude;
                        if ratio < accepted_ratio {
                            chosen = e;
                            accepted_ratio = ratio;
                        }
                        if ties >= min.saturating_mul(TIES_MULTIPLIER) {
                            return Some(chosen);
                        }
                    }
                }
                e = self.topology.elements[e].next_in_col;
            }
        }
        if chosen != NONE {
            return Some(chosen);
        }
        if largest_mag == 0.0 {
            return None;
        }
        if self.diagnostic.small_pivot().is_none() {
            let element = &self.topology.elements[largest_element];
            self.diagnostic = Diagnostic::SmallPivot(
                self.topology.int_to_ext_row[element.row],
                self.topology.int_to_ext_col[element.col],
            );
        }
        Some(largest_element)
    }

    /// `FindLargestInCol` from `e` downwards.
    fn find_largest_in_col(&self, mut e: usize) -> f64 {
        let mut largest = 0.0;
        while e != NONE {
            let magnitude = self.mag(e);
            if magnitude > largest {
                largest = magnitude;
            }
            e = self.topology.elements[e].next_in_col;
        }
        largest
    }

    /// `FindBiggestInColExclude`.
    fn find_biggest_in_col_exclude(&self, pivot: usize, step: usize) -> f64 {
        let row = self.topology.elements[pivot].row;
        let col = self.topology.elements[pivot].col;
        let mut e = self.topology.first_in_col[col];
        while e != NONE && self.topology.elements[e].row < step {
            e = self.topology.elements[e].next_in_col;
        }
        let mut largest = if self.topology.elements[e].row == row {
            0.0
        } else {
            self.mag(e)
        };
        e = self.topology.elements[e].next_in_col;
        while e != NONE {
            let magnitude = self.mag(e);
            if magnitude > largest && self.topology.elements[e].row != row {
                largest = magnitude;
            }
            e = self.topology.elements[e].next_in_col;
        }
        largest
    }

    /// `ExchangeRowsAndCols`.
    fn exchange_rows_and_cols(&mut self, scratch: &mut Markowitz, pivot: usize, step: usize) {
        let row = self.topology.elements[pivot].row;
        let col = self.topology.elements[pivot].col;
        if row == step && col == step {
            return;
        }
        self.lifecycle.invalidate_structure(false);
        if row == col {
            self.row_exchange(scratch, step, row);
            self.col_exchange(scratch, step, col);
            scratch.products.swap(step, row);
            std::sync::Arc::make_mut(&mut self.topology)
                .diag
                .swap(row, step);
        } else {
            let old_step = scratch.products[step];
            let old_row = scratch.products[row];
            let old_col = scratch.products[col];
            if row != step {
                self.row_exchange(scratch, step, row);
                self.parity.toggle();
                scratch.products.set(row, scratch.product_of(row));
                scratch.track_singleton(scratch.products[row], old_row);
            }
            if col != step {
                self.col_exchange(scratch, step, col);
                self.parity.toggle();
                scratch.products.set(col, scratch.product_of(col));
                scratch.track_singleton(scratch.products[col], old_col);
                std::sync::Arc::make_mut(&mut self.topology).diag[col] =
                    self.find_element_in_col(Slot::Head(col), col);
            }
            if row != step {
                std::sync::Arc::make_mut(&mut self.topology).diag[row] =
                    self.find_element_in_col(Slot::Head(row), row);
            }
            std::sync::Arc::make_mut(&mut self.topology).diag[step] =
                self.find_element_in_col(Slot::Head(step), step);
            scratch.products.set(step, scratch.product_of(step));
            scratch.track_singleton(scratch.products[step], old_step);
        }
    }

    /// Exchange rows in the linked topology and Markowitz counts.
    fn row_exchange(&mut self, scratch: &mut Markowitz, a: usize, b: usize) {
        std::sync::Arc::make_mut(&mut self.topology).row_exchange(a, b);
        scratch.row.swap(a, b);
    }

    fn col_exchange(&mut self, scratch: &mut Markowitz, a: usize, b: usize) {
        std::sync::Arc::make_mut(&mut self.topology).col_exchange(a, b);
        scratch.col.swap(a, b);
    }

    /// `RealRowColElimination` / `ComplexRowColElimination`.
    fn row_col_elimination(&mut self, mut scratch: Option<&mut Markowitz>, pivot: usize) {
        if self.values[pivot].mag() == 0.0 {
            let row = self.topology.elements[pivot].row;
            self.matrix_is_singular(row);
            return;
        }
        let inverse = self.values[pivot].recip();
        if !inverse.is_finite() || inverse.is_zero() {
            self.nonfinite_factor();
            return;
        }
        self.values[pivot] = inverse;
        let mut upper = self.topology.elements[pivot].next_in_row;
        while upper != NONE {
            let u = self.values[upper].mul(inverse);
            if !u.is_finite() {
                self.nonfinite_factor();
                return;
            }
            self.values[upper] = u;
            let upper_col = self.topology.elements[upper].col;
            let mut above = upper;
            let mut sub = self.topology.elements[upper].next_in_col;
            let mut lower = self.topology.elements[pivot].next_in_col;
            while lower != NONE {
                let row = self.topology.elements[lower].row;
                while sub != NONE && self.topology.elements[sub].row < row {
                    above = sub;
                    sub = self.topology.elements[sub].next_in_col;
                }
                if sub == NONE || self.topology.elements[sub].row > row {
                    let Ok(element) = self.create_fillin(
                        scratch.as_deref_mut(),
                        row,
                        upper_col,
                        Slot::After(above),
                    ) else {
                        self.resource_failure();
                        return;
                    };
                    sub = element;
                }
                let l = self.values[lower];
                let value = self.values[sub].mul_sub(u, l);
                if !value.is_finite() {
                    self.nonfinite_factor();
                    return;
                }
                self.values[sub] = value;
                above = sub;
                sub = self.topology.elements[sub].next_in_col;
                lower = self.topology.elements[lower].next_in_col;
            }
            upper = self.topology.elements[upper].next_in_row;
        }
    }

    /// `UpdateMarkowitzNumbers`.
    fn update_markowitz_numbers(&self, scratch: &mut Markowitz, pivot: usize) {
        let mut e = self.topology.elements[pivot].next_in_col;
        while e != NONE {
            let row = self.topology.elements[e].row;
            scratch.row[row] -= 1;
            scratch.products.set(row, scratch.product_of(row));
            if scratch.row[row] == 0 {
                scratch.singletons += 1;
            }
            e = self.topology.elements[e].next_in_col;
        }
        let mut e = self.topology.elements[pivot].next_in_row;
        while e != NONE {
            let col = self.topology.elements[e].col;
            scratch.col[col] -= 1;
            scratch.products.set(col, scratch.product_of(col));
            if scratch.col[col] == 0 && scratch.row[col] != 0 {
                scratch.singletons += 1;
            }
            e = self.topology.elements[e].next_in_row;
        }
    }

    /// `CreateFillin`: the elimination merge has already located the
    /// exact predecessor in its sorted column. Do not rescan its prefix.
    fn create_fillin(
        &mut self,
        scratch: Option<&mut Markowitz>,
        row: usize,
        col: usize,
        above: Slot,
    ) -> SpiceResult<usize> {
        debug_assert!(matches!(above, Slot::After(e) if self.topology.elements[e].row < row));
        debug_assert!({
            let next = self.topology.col_get(above);
            next == NONE || self.topology.elements[next].row > row
        });
        let element = self.create_element(row, col, above, true)?;
        if let Some(scratch) = scratch {
            scratch.row[row] += 1;
            scratch.products.set(row, scratch.product_of(row));
            if scratch.row[row] == 1 && scratch.col[row] != 0 {
                scratch.singletons -= 1;
            }
            scratch.col[col] += 1;
            scratch.products.set(col, scratch.product_of(col));
            if scratch.row[col] != 0 && scratch.col[col] == 1 {
                scratch.singletons -= 1;
            }
        }
        Ok(element)
    }

    fn resource_failure(&mut self) -> Status {
        self.lifecycle.require_ordering();
        self.diagnostic = Diagnostic::Resource;
        Status::Resource
    }

    fn nonfinite_factor(&mut self) -> Status {
        self.lifecycle.require_ordering();
        self.diagnostic = Diagnostic::NonFinite;
        Status::NonFinite
    }

    fn matrix_is_singular(&mut self, step: usize) -> Status {
        self.diagnostic = Diagnostic::Singular(
            self.topology.int_to_ext_row[step],
            self.topology.int_to_ext_col[step],
        );
        Status::Singular
    }

    fn zero_pivot(&mut self, step: usize) -> Status {
        self.diagnostic = Diagnostic::ZeroDiag(
            self.topology.int_to_ext_row[step],
            self.topology.int_to_ext_col[step],
        );
        Status::ZeroDiag
    }

    // --- spsolve.c ------------------------------------------------------

    /// Borrow proof that both the traversal and numeric factors are valid.
    pub(super) fn factors(&self) -> SpiceResult<FactorView<'_, T>> {
        match &self.lifecycle {
            Lifecycle::Factored(plan) => Ok(FactorView { frame: self, plan }),
            _ => Err(numerical(
                "SPARSE factors",
                "matrix has no valid numeric factors",
            )),
        }
    }

    #[cfg(test)]
    fn solve(&self, rhs: &[T]) -> Vec<T> {
        self.factors().unwrap().solve(rhs, &mut Vec::new()).unwrap()
    }

    #[cfg(test)]
    fn solve_transposed(&self, rhs: &[T]) -> Vec<T> {
        self.factors()
            .unwrap()
            .solve_transposed(rhs, &mut Vec::new())
            .unwrap()
    }
    /// Gather directly into the internal 1-based work vector without
    /// first zero-initializing every translated entry.
    fn gather_rhs(&self, rhs: &[T], x: &mut Vec<T>, transposed: bool) -> SpiceResult<()> {
        if rhs.len() != self.ext_size {
            return Err(numerical(
                "SPARSE solve",
                "RHS length differs from external matrix order",
            ));
        }
        let indices = if transposed {
            &self.topology.int_to_ext_col
        } else {
            &self.topology.int_to_ext_row
        };
        x.clear();
        sparse_reserve(x, self.size + 1)?;
        x.push(T::ZERO);
        x.extend(indices[1..=self.size].iter().map(|&ext| rhs[ext - 1]));
        Ok(())
    }

    /// Emit external-order values sequentially. Untranslated unknowns keep
    /// exactly their original RHS bits, including signed zeros.
    fn scatter_solution(
        &self,
        rhs: &[T],
        x: &[T],
        transposed: bool,
        result: &mut Vec<T>,
    ) -> SpiceResult<()> {
        let indices = if transposed {
            &self.topology.ext_to_int_row
        } else {
            &self.topology.ext_to_int_col
        };
        result.clear();
        sparse_reserve(result, rhs.len())?;
        result.extend(rhs.iter().zip(&indices[1..]).map(|(&value, &internal)| {
            if internal == NONE { value } else { x[internal] }
        }));
        Ok(())
    }
}

/// A view of immutable numeric factors, obtained only from Factored state.
/// The shared borrow forbids topology or coefficient mutation for its lifetime.
pub(super) struct FactorView<'a, T> {
    frame: &'a Sparse<T>,
    plan: &'a FactorPlan,
}

impl<T: Scalar> FactorView<'_, T> {
    /// Stored numeric entries, including fill-ins, of valid factors.
    pub(super) const fn stored_entries(&self) -> usize {
        self.frame.values.len()
    }

    pub(super) fn solve(&self, rhs: &[T], workspace: &mut Vec<T>) -> SpiceResult<Vec<T>> {
        let mut out = Vec::new();
        self.solve_into(rhs, workspace, &mut out, false)?;
        Ok(out)
    }

    pub(super) fn solve_transposed(
        &self,
        rhs: &[T],
        workspace: &mut Vec<T>,
    ) -> SpiceResult<Vec<T>> {
        let mut out = Vec::new();
        self.solve_into(rhs, workspace, &mut out, true)?;
        Ok(out)
    }

    pub(super) fn solve_into(
        &self,
        rhs: &[T],
        workspace: &mut Vec<T>,
        out: &mut Vec<T>,
        transposed: bool,
    ) -> SpiceResult<()> {
        self.frame.gather_rhs(rhs, workspace, transposed)?;
        match (self.plan, transposed) {
            (FactorPlan::Compact(runs), false) => {
                runs.solve(&self.frame.values, &self.frame.topology.diag, workspace);
            }
            (FactorPlan::Wide(runs), false) => {
                runs.solve(&self.frame.values, &self.frame.topology.diag, workspace);
            }
            (FactorPlan::Compact(runs), true) => {
                runs.solve_transposed(&self.frame.values, &self.frame.topology.diag, workspace);
            }
            (FactorPlan::Wide(runs), true) => {
                runs.solve_transposed(&self.frame.values, &self.frame.topology.diag, workspace);
            }
        }
        self.frame.scatter_solution(rhs, workspace, transposed, out)
    }

    /// `spDeterminant`: normalized decimal mantissa and exponent.
    /// Preserve C's multiplication/division and
    /// decimal scaling on every representable intermediate. If an individual
    /// step would overflow or underflow, normalize its inverse pivot first.
    pub(super) fn determinant(&self) -> SpiceResult<(T, i32)>
    where
        T: DeterminantScale,
    {
        use crate::maths::linear::numerical;
        let mut exponent = 0_i64;
        let mut det = T::one();
        for i in 1..=self.frame.size {
            let inverse = self.frame.values[self.frame.topology.diag[i]];
            if !inverse.is_finite() || inverse.is_zero() {
                return Err(numerical(
                    "SPARSE determinant",
                    "non-finite or zero inverse pivot",
                ));
            }
            let mut next = T::det_step(det, inverse);
            if !next.is_finite() || next.is_zero() || next.norm().is_subnormal() {
                let mut shift = 0;
                let normalized = normalize_decimal(inverse, &mut shift, DecimalScale::Coarse);
                next = T::det_step(det, normalized);
                exponent -= shift;
                if !next.is_finite() || next.is_zero() {
                    return Err(numerical(
                        "SPARSE determinant",
                        "unrepresentable determinant step",
                    ));
                }
            }
            det = normalize_decimal(next, &mut exponent, DecimalScale::Coarse);
        }
        det = normalize_decimal(det, &mut exponent, DecimalScale::Final);
        if self.frame.parity == PermutationParity::Odd {
            det = det.scale(-1.0);
        }
        let exponent = i32::try_from(exponent)
            .map_err(|_| numerical("SPARSE determinant", "decimal exponent out of range"))?;
        Ok((det, exponent))
    }
}

/// Decimal normalization stages in C's spDeterminant. The constants and
/// order of multiplications are intentional: replacing them with logarithms,
/// powi, or a reciprocal changes rounding on ordinary matrices.
#[derive(Clone, Copy)]
enum DecimalScale {
    Coarse,
    Final,
}

fn normalize_decimal<T: DeterminantScale>(
    mut value: T,
    exponent: &mut i64,
    scale: DecimalScale,
) -> T {
    debug_assert!(value.is_finite());
    let (low, high, up, down, step) = match scale {
        DecimalScale::Coarse => (1.0e-12, 1.0e12, 1.0e12, 1.0e-12, 12),
        DecimalScale::Final => (1.0, 10.0, 10.0, 0.1, 1),
    };
    let mut norm = value.norm();
    if norm != 0.0 {
        while norm >= high {
            value = value.scale(down);
            *exponent += step;
            norm = value.norm();
        }
        while norm < low {
            value = value.scale(up);
            *exponent -= step;
            norm = value.norm();
        }
    }
    value
}

/// Scalar operations used only by [`Sparse::determinant`].
pub(super) trait DeterminantScale: Scalar {
    /// One.
    fn one() -> Self;
    /// C's per-pivot update: real `det /= diag` (diag holds `1/pivot`), complex
    /// `det *= CMPLX_RECIPROCAL(diag)`.
    fn det_step(det: Self, inverse_pivot: Self) -> Self;
    /// `NORM`: `max(|re|, |im|)` (real: `|x|`).
    fn norm(self) -> f64;
    /// Multiply by a real factor.
    fn scale(self, factor: f64) -> Self;
}

impl DeterminantScale for f64 {
    fn one() -> Self {
        1.0
    }
    fn det_step(det: Self, inverse_pivot: Self) -> Self {
        det / inverse_pivot
    }
    fn norm(self) -> f64 {
        self.abs()
    }
    fn scale(self, factor: f64) -> Self {
        self * factor
    }
}

impl DeterminantScale for Complex {
    fn one() -> Self {
        Self::real(1.0)
    }
    fn det_step(det: Self, inverse_pivot: Self) -> Self {
        // CMPLX_MULT_ASSIGN(det, reciprocal).
        det.mul(inverse_pivot.recip())
    }
    fn norm(self) -> f64 {
        self.re.abs().max(self.im.abs())
    }
    fn scale(self, factor: f64) -> Self {
        Self::new(self.re * factor, self.im * factor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(n: usize, entries: &[(usize, usize, f64)]) -> Sparse<f64> {
        let mut m = Sparse::new(n).unwrap();
        for &(r, c, v) in entries {
            let e = m.get_element(r, c).unwrap();
            m.add(e, v);
        }
        m
    }

    fn residual(entries: &[(usize, usize, f64)], x: &[f64], b: &[f64]) -> f64 {
        let mut ax = vec![0.0; b.len()];
        for &(r, c, v) in entries {
            ax[r - 1] += v * x[c - 1];
        }
        ax.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn sparse_construction_rejects_invalid_dimensions_and_external_coordinates() {
        assert!(Sparse::<f64>::new(usize::MAX).is_err());
        assert!(Sparse::<f64>::new(usize::MAX / size_of::<usize>()).is_err());
        assert!(Sparse::<f64>::from_originals(2, [(0, 1, 1.)]).is_err());
        assert!(Sparse::<f64>::from_originals(2, [(2, 3, 1.)]).is_err());
        let pattern = crate::maths::Pattern::new(2, [(0, 0), (1, 1)]).unwrap();
        assert!(Sparse::<f64>::from_pattern(&pattern, &[1.]).is_err());
        let mut sparse = Sparse::<f64>::new(2).unwrap();
        assert!(sparse.get_element(1, usize::MAX).is_err());
        assert!(sparse.get_element(0, 1).is_err());
        assert!(sparse.add_col(1, usize::MAX).is_err());
        assert!(sparse.zero_col(0).is_err());
        assert_eq!(sparse.size, 0);
        let first = sparse.get_element(2, 1).unwrap();
        sparse.set(first, 2.);
        let second = sparse.get_element(1, 1).unwrap();
        sparse.add(first, 3.);
        assert_eq!(sparse.get_element(2, 1).unwrap(), first);
        assert_ne!(second, first);
        assert_eq!(sparse.value(first), 5.);
    }

    #[test]
    fn unfactored_solve_rejects_without_touching_workspace() {
        let m = Sparse::<f64>::new(1).unwrap();
        let mut workspace = vec![42.0];
        assert!(
            m.factors()
                .and_then(|factors| factors.solve(&[1.0], &mut workspace))
                .is_err()
        );
        assert_eq!(workspace, vec![42.0]);
    }

    #[test]
    fn lifecycle_rejects_stale_factors_after_every_numeric_edit() {
        let entries = [(1, 1, 3.), (1, 2, 1.), (2, 1, 2.), (2, 2, 4.)];
        let input = entries.map(|(_, _, v)| v);
        let mut m = build(2, &entries);
        assert!(matches!(m.lifecycle, Lifecycle::ColumnsOnly));
        assert!(m.factors().is_err());
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        assert!(matches!(m.lifecycle, Lifecycle::Factored(_)));
        let plan = m.lifecycle.plan().unwrap().clone();
        let initial = m.solve(&[4., 6.]);
        assert_eq!(m.factor(), Status::NeedsReload);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::NeedsReload);
        assert_eq!(m.solve(&[4., 6.]), initial);
        let handle = m.get_element(1, 1).unwrap();
        assert!(m.factors().is_ok(), "existing-handle lookup is read-only");
        let edits: [fn(&mut Sparse<f64>, SolverElement); 6] = [
            |frame: &mut Sparse<f64>, h| frame.set(h, 1.),
            |frame: &mut Sparse<f64>, h| frame.add(h, 1.),
            |frame: &mut Sparse<f64>, _| frame.load_gmin(1.),
            |frame: &mut Sparse<f64>, _| frame.zero_col(1).unwrap(),
            |frame: &mut Sparse<f64>, _| frame.add_col(1, 2).unwrap(),
            |frame: &mut Sparse<f64>, _| frame.clear(),
        ];
        for edit in edits {
            let saved = m.clone();
            let rhs = [4., 6.];
            let expected = saved.solve(&rhs);
            edit(&mut m, handle);
            assert!(m.factors().is_err());
            if !m.coefficients_loaded() {
                assert_eq!(m.factor(), Status::NeedsReload);
                assert_eq!(m.order_and_factor(1e-3, 0.), Status::NeedsReload);
            }
            assert!(matches!(m.lifecycle, Lifecycle::Symbolic(_)));
            assert!(std::sync::Arc::ptr_eq(m.lifecycle.plan().unwrap(), &plan));
            assert_eq!(saved.solve(&rhs), expected);
            m.reload_originals(&input);
            assert_eq!(m.factor(), Status::Okay);
            assert_eq!(m.solve(&rhs), expected);
        }
        let retyped: Sparse<Complex> = m.into_retyped();
        assert!(matches!(retyped.lifecycle, Lifecycle::Symbolic(_)));
        assert!(retyped.factors().is_err());
    }

    #[test]
    fn plan_construction_rejects_inconsistent_links_and_offsets() {
        let entries = [(1, 1, 3.), (1, 2, 1.), (2, 1, 2.), (2, 2, 4.)];
        let mut original = build(2, &entries);
        assert_eq!(original.order_and_factor(1e-3, 0.), Status::Okay);
        let corruptions: [fn(&mut Topology); 6] = [
            |topology: &mut Topology| topology.diag[1] = NONE,
            |topology: &mut Topology| topology.first_in_col[1] = topology.elements.len(),
            |topology: &mut Topology| {
                let head = topology.first_in_col[1];
                topology.elements[head].next_in_col = head;
            },
            |topology: &mut Topology| {
                let head = topology.first_in_col[1];
                topology.elements[head].row = 0;
            },
            |topology: &mut Topology| {
                let h = topology.diag[1];
                topology.elements[h].next_in_row = NONE;
            },
            |topology: &mut Topology| {
                let h = topology.diag[1];
                topology.elements[h].next_in_row = h;
            },
        ];
        for corrupt in corruptions {
            let mut frame = original.clone();
            corrupt(std::sync::Arc::make_mut(&mut frame.topology));
            frame.lifecycle = Lifecycle::Unordered;
            assert!(matches!(
                frame.compile_runs::<u32>(),
                Err(Status::InvalidPlan)
            ));
            assert!(matches!(
                frame.compile_runs::<usize>(),
                Err(Status::InvalidPlan)
            ));
        }
        assert!(ColumnRun::<u32>::new(0, 0, 0).is_none());
        assert!(ColumnRun::<usize>::new(3, 2, 4).is_none());
        if let Ok(max) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(ColumnRun::<u32>::new(0, 0, max).is_none());
            assert!(ColumnRun::<usize>::new(0, 0, max).is_some());
        }
    }

    #[test]
    fn zero_diagonal_and_nonfinite_failures_have_distinct_retry_states() {
        let entries = [(1, 1, 2.), (2, 2, 3.)];
        let input = entries.map(|(_, _, v)| v);
        let mut m = build(2, &entries);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let plan = m.lifecycle.plan().unwrap().clone();
        m.reload_originals(&[0., 3.]);
        assert_eq!(m.factor(), Status::ZeroDiag);
        assert_eq!(m.factor(), Status::NeedsReload);
        assert!(matches!(m.lifecycle, Lifecycle::Symbolic(_)));
        assert!(m.factors().is_err());
        assert_ne!(m.where_singular(), (0, 0));
        m.reload_originals(&[f64::NAN, 3.]);
        assert_eq!(m.factor(), Status::NonFinite);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::NeedsReload);
        assert!(matches!(m.lifecycle, Lifecycle::ReorderRequired(_)));
        assert!(m.factors().is_err());
        assert_eq!(m.where_singular(), (0, 0));
        m.reload_originals(&input);
        assert!(!m.can_refactor());
        assert_eq!(m.factor(), Status::Okay);
        assert!(std::sync::Arc::ptr_eq(m.lifecycle.plan().unwrap(), &plan));
        assert!(m.factors().is_ok());
    }

    #[test]
    fn pivot_product_summaries_match_reference_reverse_searches() {
        for size in [1, 63, 64, 323, 514, 515, 1027] {
            let mut products = PivotProducts::new(size).unwrap();
            let mut seed = 71_u64;
            for i in 0..6000 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let at = ((seed >> 32) as usize) % size;
                let v = ((seed >> 48) % 17) as i64 - 2;
                products.set(at, v);
                if i % 11 == 0 {
                    products.swap(at, (at + 99) % size);
                }
                let before = ((seed >> 16) as usize) % (size + 1);
                let limit = ((seed >> 40) % 20) as i64 - 3;
                let expect = products.values[..before].iter().rposition(|&v| v < limit);
                assert_eq!(products.previous_below(before, limit), expect);
                let expect = products.values[..before].iter().rposition(|&v| v == 0);
                assert_eq!(products.previous_zero(before), expect);
            }
        }
    }

    #[test]
    fn solve_workspace_reuses_storage_and_keeps_external_permutation() {
        // Nodes 2 and 4 are deliberately absent, and the remaining two
        // have a zero diagonal forcing MNA/pivot exchange.
        let mut m = build(4, &[(3, 1, 2.), (1, 3, 4.), (3, 3, 1.)]);
        m.mna_preorder();
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let rhs = [8., -0., 6., -17.];
        let mut workspace = Vec::new();
        let first = m.factors().unwrap().solve(&rhs, &mut workspace).unwrap();
        let storage = workspace.as_ptr();
        assert_eq!(first[0], 2.);
        assert_eq!(first[2], 2.);
        assert_eq!(first[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(first[3], -17.);
        let second = m
            .factors()
            .unwrap()
            .solve(&[4., 3., 4., 7.], &mut workspace)
            .unwrap();
        assert_eq!(storage, workspace.as_ptr());
        assert_eq!(second, [1.5, 3., 1., 7.]);
        let transposed = m.solve_transposed(&rhs);
        assert_eq!(transposed[0], 0.5);
        assert_eq!(transposed[2], 4.);
        assert_eq!(transposed[1].to_bits(), (-0.0_f64).to_bits());
        assert_eq!(transposed[3], -17.);
    }

    #[test]
    fn extreme_complex_reciprocals_and_invalid_factor_recovery() {
        let mut c = Sparse::<Complex>::new(1).unwrap();
        let handle = c.get_element(1, 1).unwrap();
        c.set(handle, Complex::new(f64::MAX, f64::MAX));
        assert!(matches!(
            c.order_and_factor(1e-3, 0.),
            Status::Okay | Status::SmallPivot
        ));
        let x = c.solve(&[Complex::real(1.)])[0];
        assert!(x.re > 0. && x.im < 0.);
        assert!((x.re * f64::MAX - 0.5).abs() < 1e-14);
        let mut m = build(2, &[(1, 1, 2.), (1, 2, 1.), (2, 1, 1.), (2, 2, 2.)]);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        m.clear();
        for (r, c, v) in [(1, 1, 1.), (1, 2, f64::MAX), (2, 1, f64::MAX), (2, 2, 1.)] {
            let h = m.get_element(r, c).unwrap();
            m.set(h, v);
        }
        assert_eq!(m.factor(), Status::NonFinite);
        assert!(!m.is_factored());
        assert!(m.factors().is_err());
        m.clear();
        for (r, c, v) in [(1, 1, 2.), (1, 2, 1.), (2, 1, 1.), (2, 2, 2.)] {
            let h = m.get_element(r, c).unwrap();
            m.set(h, v);
        }
        assert_eq!(m.factor(), Status::Okay);
        assert_eq!(m.solve(&[3., 3.]), [1., 1.]);
    }

    #[test]
    fn fixed_order_checks_every_stored_factor_and_early_zero_pivot() {
        let entries = [(1, 1, 2.), (1, 2, 1.), (2, 1, 1.), (2, 2, 2.)];
        let mut m = build(2, &entries);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        for (bad_row, bad_col, bad_value) in [
            (2, 1, f64::INFINITY),
            (1, 2, f64::NAN),
            (2, 2, f64::NEG_INFINITY),
        ] {
            m.clear();
            for (r, c, v) in entries {
                let h = m.get_element(r, c).unwrap();
                m.set(
                    h,
                    if (r, c) == (bad_row, bad_col) {
                        bad_value
                    } else {
                        v
                    },
                );
            }
            assert_eq!(m.factor(), Status::NonFinite);
            assert!(!m.is_factored());
            m.clear();
            for (r, c, v) in entries {
                let h = m.get_element(r, c).unwrap();
                m.set(h, v);
            }
            assert_eq!(m.factor(), Status::Okay);
        }
        m.clear();
        let h = m.get_element(2, 2).unwrap();
        m.set(h, f64::NAN);
        assert_eq!(m.factor(), Status::NonFinite);
    }

    #[test]
    fn determinant_rescales_extreme_pivots_without_nonfinite_loops() {
        for (a, b, expected) in [(1e307, 1e308, 615), (1e-307, 1e-308, -615)] {
            let mut m = build(2, &[(1, 1, a), (2, 2, b)]);
            assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
            let (mantissa, exponent) = m.factors().unwrap().determinant().unwrap();
            assert!((mantissa * 10_f64.powi(exponent - expected) - 1.).abs() < 1e-12);
        }
        let mut m = build(1, &[(1, 1, f64::from_bits(1))]);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::NonFinite);
        assert!(!m.is_factored());
        assert!(
            m.factors().is_err(),
            "inverse pivot overflow must be explicit"
        );

        let mut m = Sparse::<Complex>::new(2).unwrap();
        for i in 1..=2 {
            let h = m.get_element(i, i).unwrap();
            m.set(h, Complex::new(1e307, 1e307));
        }
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let (mantissa, exponent) = m.factors().unwrap().determinant().unwrap();
        let imag = mantissa.im * 10_f64.powi(exponent - 614);
        assert!((imag - 2.).abs() < 1e-12);
        assert!((mantissa.re * 10_f64.powi(exponent - 614)).abs() < 1e-12);
    }

    #[test]
    fn bulk_column_finalization_preserves_nonmonotonic_links() {
        // Translate all unknowns first, then insert a densely populated
        // column in deliberately unsorted internal row order.
        let mut entries: Vec<_> = (1..=8).map(|i| (i, i, 20.)).collect();
        entries.extend([7, 3, 8, 2, 6, 4, 5].map(|r| (r, 1, r as f64)));
        entries.extend([8, 6, 5, 4, 3, 1, 7].map(|r| (r, 2, -1.)));
        let a = Sparse::from_originals(8, entries.iter().copied()).unwrap();
        let b = build(8, &entries);
        assert_eq!(a.topology.first_in_col, b.topology.first_in_col);
        assert_eq!(a.topology.diag, b.topology.diag);
        for (a, b) in a.topology.elements.iter().zip(&b.topology.elements) {
            assert_eq!(
                (a.row, a.col, a.next_in_col, a.next_in_row),
                (b.row, b.col, b.next_in_col, b.next_in_row)
            );
        }
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn bulk_column_builder_rejects_duplicate_originals() {
        let error = Sparse::from_originals(2, [(1, 2, 1.), (2, 2, 2.), (1, 2, 3.)]).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("duplicate original sparse element")
        );
    }

    #[test]
    fn batched_original_builder_preserves_translation_and_creation_order() {
        let entries = [
            (4, 2, 2.),
            (2, 1, 1.),
            (1, 4, 3.),
            (2, 2, 4.),
            (1, 1, 5.),
            (4, 4, 6.),
        ];
        let mut a = Sparse::from_originals(5, entries).unwrap();
        let mut b = build(5, &entries);
        assert_eq!(a.size, b.size);
        assert_eq!(a.topology.int_to_ext_row, b.topology.int_to_ext_row);
        assert_eq!(a.topology.int_to_ext_col, b.topology.int_to_ext_col);
        assert_eq!(a.topology.first_in_col, b.topology.first_in_col);
        assert_eq!(a.topology.diag, b.topology.diag);
        for (a, b) in a.topology.elements.iter().zip(&b.topology.elements) {
            assert_eq!(
                (a.row, a.col, a.next_in_col, a.next_in_row),
                (b.row, b.col, b.next_in_col, b.next_in_row)
            );
        }
        assert_eq!(a.values, b.values);
        a.mna_preorder();
        b.mna_preorder();
        assert_eq!(a.order_and_factor(1e-3, 0.), Status::Okay);
        assert_eq!(b.order_and_factor(1e-3, 0.), Status::Okay);
        assert_eq!(a.topology.int_to_ext_row, b.topology.int_to_ext_row);
        assert_eq!(a.topology.int_to_ext_col, b.topology.int_to_ext_col);
        assert!(
            a.values
                .iter()
                .map(|v| v.to_bits())
                .eq(b.values.iter().map(|v| v.to_bits()))
        );
        assert_eq!(
            a.solve(&[1., -0., 2., 3., 4.]),
            b.solve(&[1., -0., 2., 3., 4.])
        );
    }

    #[test]
    fn traversal_plan_matches_links_and_is_shared_until_topology_changes() {
        let entries = [
            (1, 2, 3.),
            (2, 1, 2.),
            (2, 2, 4.),
            (2, 3, 1.),
            (3, 3, 4.),
            (3, 4, 1.),
            (4, 4, 4.),
            (4, 1, 1.),
        ];
        let mut m = Sparse::from_originals(4, entries).unwrap();
        m.mna_preorder();
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let plan = std::sync::Arc::clone(m.lifecycle.plan().unwrap());
        let FactorPlan::Compact(runs) = plan.as_ref() else {
            panic!("small matrix uses the compact plan");
        };
        for i in 1..=m.size {
            let mut col = Vec::new();
            let mut e = m.topology.first_in_col[i];
            while e != NONE {
                col.push((e, m.topology.elements[e].row));
                e = m.topology.elements[e].next_in_col;
            }
            assert!(runs.column(i).iter().map(|entry| entry.indices()).eq(col));
            let mut row = Vec::new();
            let mut e = m.topology.elements[m.topology.diag[i]].next_in_row;
            while e != NONE {
                row.push((e, m.topology.elements[e].col));
                e = m.topology.elements[e].next_in_row;
            }
            assert!(
                runs.upper_row(i)
                    .iter()
                    .map(|entry| entry.indices())
                    .eq(row)
            );
        }
        let rhs = [5., 6., 7., 8.];
        let x = m.solve(&rhs);
        let xt = m.solve_transposed(&rhs);
        m.reload_originals(&entries.map(|(_, _, v)| v));
        assert_eq!(m.factor(), Status::Okay);
        assert_eq!(m.solve(&rhs), x);
        assert_eq!(m.solve_transposed(&rhs), xt);
        assert!(std::sync::Arc::ptr_eq(m.lifecycle.plan().unwrap(), &plan));
        let mut c = m.into_retyped::<Complex>();
        assert!(std::sync::Arc::ptr_eq(c.lifecycle.plan().unwrap(), &plan));
        // Insert a missing structural entry after a completed ordering.
        let h = c.get_element(1, 4).unwrap();
        c.set(h, Complex::real(2.));
        assert!(c.lifecycle.plan().is_none());
    }

    #[test]
    fn compact_and_wide_runs_preserve_factor_and_solve_bits() {
        assert_eq!(std::mem::size_of::<RunEntry<u32>>(), 8);
        assert_eq!(std::mem::size_of::<ColumnRun<u32>>(), 12);
        let entries = [
            (1, 1, 4.),
            (1, 2, 1.),
            (2, 2, 5.),
            (2, 3, 2.),
            (3, 1, -1.),
            (3, 3, 6.),
        ];
        let input = entries.map(|(_, _, v)| v);
        let mut narrow = build(3, &entries);
        assert_eq!(narrow.order_and_factor(1e-3, 0.), Status::Okay);
        let mut wide = narrow.clone();
        wide.lifecycle = Lifecycle::Factored(std::sync::Arc::new(FactorPlan::Wide(
            wide.compile_runs().unwrap(),
        )));
        narrow.reload_originals(&input);
        wide.reload_originals(&input);
        assert_eq!(narrow.factor(), Status::Okay);
        assert_eq!(wide.factor(), Status::Okay);
        assert!(
            narrow
                .values
                .iter()
                .map(|v| v.to_bits())
                .eq(wide.values.iter().map(|v| v.to_bits()))
        );
        let rhs = [2., -0., 1.];
        for (a, b) in [
            (narrow.solve(&rhs), wide.solve(&rhs)),
            (narrow.solve_transposed(&rhs), wide.solve_transposed(&rhs)),
        ] {
            assert!(
                a.iter()
                    .map(|v| v.to_bits())
                    .eq(b.iter().map(|v| v.to_bits()))
            );
        }
        let input = input.map(|v| Complex::new(v, v * 0.125));
        let mut narrow = narrow.into_retyped::<Complex>();
        let mut wide = wide.into_retyped::<Complex>();
        narrow.reload_originals(&input);
        wide.reload_originals(&input);
        assert_eq!(narrow.factor(), Status::Okay);
        assert_eq!(wide.factor(), Status::Okay);
        let bits = |z: &Complex| (z.re.to_bits(), z.im.to_bits());
        assert!(
            narrow
                .values
                .iter()
                .map(bits)
                .eq(wide.values.iter().map(bits))
        );
        let rhs = rhs.map(Complex::real);
        for (a, b) in [
            (narrow.solve(&rhs), wide.solve(&rhs)),
            (narrow.solve_transposed(&rhs), wide.solve_transposed(&rhs)),
        ] {
            assert!(a.iter().map(bits).eq(b.iter().map(bits)));
        }
    }

    #[test]
    fn failed_reorder_cannot_publish_a_partial_traversal() {
        let entries = [
            (1, 1, 4.),
            (1, 2, 1.),
            (1, 3, 1.),
            (2, 1, 1.),
            (2, 2, 5.),
            (2, 3, 1.),
            (3, 1, 1.),
            (3, 2, 1.),
            (3, 3, 6.),
        ];
        let input = entries.map(|(_, _, v)| v);
        let mut a = build(3, &entries);
        assert_eq!(a.order_and_factor(1e-3, 0.), Status::Okay);
        let saved = a.clone();
        let rhs = [6., 7., 8.];
        let expected = saved.solve(&rhs);
        a.clear();
        // Force a threshold failure at the first old pivot, and allow
        // a different diagonal pivot before encountering singularity.
        a.values[a.topology.diag[2]] = 2.;
        assert_eq!(a.order_and_factor(1e-3, 0.), Status::Singular);
        assert!(!a.is_factored());
        assert!(!a.can_refactor());
        assert!(
            matches!(a.lifecycle, Lifecycle::PartialOrder),
            "C retains its partial ordering after singular failure"
        );
        assert!(a.lifecycle.plan().is_none());
        let mut threshold = a.clone();
        a.reload_originals(&input);
        threshold.reload_originals(&input);
        assert_eq!(threshold.order_and_factor(0., 0.), Status::Okay);
        assert_eq!(a.factor(), Status::Okay);
        assert_eq!(a.topology.int_to_ext_row, threshold.topology.int_to_ext_row);
        assert_eq!(a.topology.int_to_ext_col, threshold.topology.int_to_ext_col);
        assert_eq!(a.values, threshold.values);
        let x = a.solve(&rhs);
        for v in x {
            assert!((v - 1.).abs() < 1e-14);
        }
        assert_eq!(saved.solve(&rhs), expected);
    }

    #[test]
    fn loaded_scalar_transition_preserves_original_bits_and_zero_fillins() {
        let entries = [
            (1, 1, 4.),
            (1, 2, 1.),
            (2, 2, 5.),
            (2, 3, 1.),
            (3, 3, 6.),
            (3, 1, 1.),
        ];
        let mut real = Sparse::from_originals(3, entries).unwrap();
        assert_eq!(real.order_and_factor(1e-3, 0.), Status::Okay);
        assert!(real.values.len() > entries.len());
        let real = std::sync::Arc::new(real);
        let before = real.solve(&[1.; 3]);
        let loaded = entries.map(|(_, _, v)| Complex::new(v, -0.));
        let complex =
            Sparse::retype_shared_loaded(std::sync::Arc::clone(&real), loaded.into_iter());
        assert!(!complex.is_factored());
        assert!(std::sync::Arc::ptr_eq(&real.topology, &complex.topology));
        assert!(
            complex.values[..loaded.len()]
                .iter()
                .zip(loaded)
                .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits())
        );
        assert!(
            complex.values[loaded.len()..]
                .iter()
                .all(|v| v.re.to_bits() == 0 && v.im.to_bits() == 0)
        );
        assert_eq!(real.solve(&[1.; 3]), before);
    }

    #[test]
    fn reload_fork_preserves_snapshots_without_copying_numeric_scratch() {
        let entries = [(1, 1, 4.), (1, 2, 1.), (2, 1, 2.), (2, 2, 5.)];
        let rhs = [5., 7.];
        let mut a = build(2, &entries);
        assert_eq!(a.order_and_factor(1e-3, 0.), Status::Okay);
        a.reload_originals(&entries.map(|(_, _, v)| v));
        assert_eq!(a.factor(), Status::Okay);
        let saved = std::sync::Arc::new(a);
        let expected = saved.solve(&rhs);
        let mut changed = std::sync::Arc::clone(&saved);
        Sparse::reload_shared(&mut changed, [8., 1., 2., 5.].into_iter());
        let next = std::sync::Arc::get_mut(&mut changed).unwrap();
        assert_eq!(&next.values[..4], &[8., 1., 2., 5.]);
        assert!(next.values[4..].iter().all(|v| *v == 0.));
        assert!(next.ordering_scratch.row.is_empty());
        assert!(next.ordering_scratch.col.is_empty());
        assert!(next.ordering_scratch.products.is_empty());
        assert_eq!(next.factor(), Status::Okay);
        assert_ne!(next.solve(&rhs), expected);
        assert!(std::sync::Arc::ptr_eq(
            next.lifecycle.plan().unwrap(),
            saved.lifecycle.plan().unwrap()
        ));
        assert_eq!(saved.solve(&rhs), expected);
        // Force threshold failure and a new pivot order: deferred counts
        // must be allocated before searching and creating fill-ins.
        next.reload_originals(&[0., 1., 2., 5.]);
        assert_eq!(next.order_and_factor(1e-3, 0.), Status::Okay);
        assert!(!next.ordering_scratch.row.is_empty());
        assert_eq!(saved.solve(&rhs), expected);
        let c: Sparse<Complex> = Sparse::retype_shared(std::sync::Arc::clone(&saved));
        assert!(c.values.iter().all(|v| *v == Complex::ZERO));
        assert!(c.ordering_scratch.row.is_empty());
        assert_eq!(saved.solve(&rhs), expected);
    }

    #[test]
    fn snapshot_topology_is_shared_until_structural_mutation() {
        let mut a = build(3, &[(1, 1, 4.), (2, 2, 5.), (1, 2, 1.)]);
        assert_eq!(a.order_and_factor(1e-3, 0.), Status::Okay);
        let saved = a.clone();
        assert!(std::sync::Arc::ptr_eq(&a.topology, &saved.topology));
        let rhs = [5., 5., -0.];
        let answer = saved.solve(&rhs);
        a.reload_originals(&[6., 5., 1.]);
        assert_eq!(a.factor(), Status::Okay);
        assert!(std::sync::Arc::ptr_eq(&a.topology, &saved.topology));
        assert_ne!(a.solve(&rhs), answer);
        let mut complex = a.into_retyped::<Complex>();
        assert!(std::sync::Arc::ptr_eq(&complex.topology, &saved.topology));
        // Lookup of an already allocated element does not detach.
        complex.get_element(1, 1).unwrap();
        assert!(std::sync::Arc::ptr_eq(&complex.topology, &saved.topology));
        // Add a previously untranslated unknown after successful ordering.
        let h = complex.get_element(3, 3).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&complex.topology, &saved.topology));
        complex.clear();
        for i in 1..=3 {
            let h = complex.get_element(i, i).unwrap();
            complex.set(h, Complex::real(i as f64));
        }
        complex.set(h, Complex::real(3.));
        assert_eq!(complex.order_and_factor(1e-3, 0.), Status::Okay);
        assert_eq!(
            complex.solve(&[Complex::real(1.), Complex::real(2.), Complex::real(3.)]),
            [Complex::real(1.); 3]
        );
        assert_eq!(saved.solve(&rhs), answer);
        assert_eq!(saved.untranslated().collect::<Vec<_>>(), [3]);
    }

    #[test]
    fn contiguous_original_reload_clears_fillins_and_preserves_handles() {
        let entries = [
            (1, 1, 4.),
            (1, 2, 1.),
            (2, 2, 4.),
            (2, 3, 1.),
            (3, 3, 4.),
            (3, 4, 1.),
            (4, 4, 4.),
            (4, 1, 1.),
        ];
        let mut m = Sparse::from_originals(4, entries).unwrap();
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        assert!(m.values.len() > entries.len());
        let rhs = [5.; 4];
        let before = m.solve(&rhs);
        let ptr = m.values.as_ptr();
        let topology = m.topology.elements.as_ptr();
        let input = entries.map(|(_, _, v)| v);
        m.reload_originals(&input);
        assert!(!m.is_factored());
        assert_eq!(&m.values[..entries.len()], &input);
        assert!(m.values[entries.len()..].iter().all(|v| *v == 0.));
        assert_eq!(m.factor(), Status::Okay);
        assert_eq!(m.solve(&rhs), before);
        assert_eq!(m.values.as_ptr(), ptr);
        assert_eq!(m.topology.elements.as_ptr(), topology);
        m.reload_originals_iter(input.into_iter());
        assert_eq!(m.factor(), Status::Okay);
        assert_eq!(m.solve(&rhs), before);
    }

    #[test]
    fn scalar_transition_moves_structural_allocations() {
        let mut real = build(2, &[(1, 1, 2.), (1, 2, 1.), (2, 1, 3.), (2, 2, 4.)]);
        assert_eq!(real.order_and_factor(1e-3, 0.), Status::Okay);
        let elements = real.topology.elements.as_ptr();
        let diagonal = real.topology.diag.as_ptr();
        let rows = real.topology.int_to_ext_row.as_ptr();
        let columns = real.topology.int_to_ext_col.as_ptr();
        let complex = real.into_retyped::<Complex>();
        assert_eq!(complex.topology.elements.as_ptr(), elements);
        assert_eq!(complex.topology.diag.as_ptr(), diagonal);
        assert_eq!(complex.topology.int_to_ext_row.as_ptr(), rows);
        assert_eq!(complex.topology.int_to_ext_col.as_ptr(), columns);
        assert!(!complex.is_factored());
        assert!(complex.values.iter().all(|v| v.is_zero()));
        let real = complex.into_retyped::<f64>();
        assert_eq!(real.topology.elements.as_ptr(), elements);
        assert_eq!(real.topology.diag.as_ptr(), diagonal);
        assert_eq!(real.topology.int_to_ext_row.as_ptr(), rows);
        assert_eq!(real.topology.int_to_ext_col.as_ptr(), columns);
    }

    #[test]
    fn singleton_scan_preserves_c_sentinel_and_reverse_order() {
        // C places the current step's product at size+1. A singleton at
        // step must be returned from that sentinel before scanning other
        // positions. Otherwise the scan proceeds from the highest index.
        let m = build(4, &[(1, 1, 5.), (2, 2, 2.), (3, 3, 3.), (4, 4, 4.)]);
        let mut scratch = Markowitz::default();
        scratch.ensure_size(m.size).unwrap();
        for i in 1..=4 {
            scratch.products.set(i, 0);
        }
        scratch.singletons = 4;
        let pivot = m.search_for_singleton(&mut scratch, 1).unwrap();
        assert_eq!(m.topology.elements[pivot].row, 1);

        // The sentinel is not a singleton: the search selects the highest
        // remaining zero-product diagonal, never reading a negative index.
        for (i, v) in [1, 1, 0, 0].into_iter().enumerate() {
            scratch.products.set(i + 1, v);
        }
        scratch.singletons = 2;
        let pivot = m.search_for_singleton(&mut scratch, 1).unwrap();
        assert_eq!(m.topology.elements[pivot].row, 4);
    }

    #[test]
    fn solves_with_pivoting_and_mna_twins() {
        // Voltage source branch (row/col 3) with zero diagonal.
        let entries = [
            (1, 1, 1e-3),
            (1, 2, -1e-3),
            (2, 1, -1e-3),
            (2, 2, 3e-3),
            (1, 3, 1.0),
            (3, 1, 1.0),
        ];
        let mut m = build(3, &entries);
        m.mna_preorder();
        assert_eq!(m.order_and_factor(1e-3, 1e-13), Status::Okay);
        let b = [0.0, 0.0, 5.0];
        let x = m.solve(&b);
        assert!(residual(&entries, &x, &b) < 1e-12, "{x:?}");
        let xt = m.solve_transposed(&b);
        let transposed: Vec<_> = entries.iter().map(|&(r, c, v)| (c, r, v)).collect();
        assert!(residual(&transposed, &xt, &b) < 1e-12, "{xt:?}");
    }

    #[test]
    fn refactor_reuses_pivots_and_detects_zero_diagonal() {
        let entries = [(1, 1, 2.0), (1, 2, 1.0), (2, 1, 1.0), (2, 2, 3.0)];
        let mut m = build(2, &entries);
        assert_eq!(m.order_and_factor(1e-3, 0.0), Status::Okay);
        let handles: Vec<_> = entries
            .iter()
            .map(|&(r, c, _)| m.get_element(r, c).unwrap())
            .collect();
        m.clear();
        for (h, v) in handles.iter().zip([4.0, 1.0, 1.0, 5.0]) {
            m.set(*h, v);
        }
        assert_eq!(m.factor(), Status::Okay);
        let x = m.solve(&[5.0, 6.0]);
        assert!((4.0 * x[0] + x[1] - 5.0).abs() < 1e-14);
        m.clear();
        assert_eq!(m.factor(), Status::ZeroDiag);
    }

    #[test]
    fn fixed_order_refactor_does_not_read_previous_workspace_contents() {
        // A nonmonotonic, fill-producing original pattern, with exact
        // structural zeros deliberately retained in its assembly.
        let n = 30;
        let mut seed = 19_u64;
        let mut entries = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for i in (1..=n).rev() {
            entries.push((i, i, 20.));
            seen.insert((i, i));
            for j in 0..5 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let c = (seed >> 32) as usize % n + 1;
                if seen.insert((i, c)) {
                    entries.push((i, c, if j == 0 { -0. } else { 0.125 }));
                }
            }
        }
        let mut real = Sparse::from_originals(n, entries.iter().copied()).unwrap();
        assert_eq!(real.order_and_factor(1e-3, 0.), Status::Okay);
        assert!(real.values.len() > entries.len());
        let mut clean = real.clone();
        let mut poisoned = real.clone();
        let values = entries.iter().map(|&(_, _, v)| v * 1.5);
        clean.reload_originals_iter(values.clone());
        poisoned.reload_originals_iter(values);
        assert_eq!(clean.factor_with_workspace(&mut Vec::new()), Status::Okay);
        assert_eq!(
            poisoned.factor_with_workspace(&mut vec![f64::NAN; n + 1]),
            Status::Okay
        );
        assert!(
            clean
                .values
                .iter()
                .map(|v| v.to_bits())
                .eq(poisoned.values.iter().map(|v| v.to_bits()))
        );
        let mut complex = real.into_retyped::<Complex>();
        complex.reload_originals_iter(entries.iter().map(|&(_, _, v)| Complex::new(v, v * 0.25)));
        let mut poisoned = complex.clone();
        assert_eq!(complex.factor_with_workspace(&mut Vec::new()), Status::Okay);
        assert_eq!(
            poisoned.factor_with_workspace(&mut vec![Complex::new(f64::NAN, f64::NAN); n + 1]),
            Status::Okay
        );
        let bits = |v: &Complex| (v.re.to_bits(), v.im.to_bits());
        assert!(
            complex
                .values
                .iter()
                .map(bits)
                .eq(poisoned.values.iter().map(bits))
        );
    }

    #[test]
    fn refactor_workspace_survives_success_and_singularity() {
        let entries = [(1, 1, 3.), (1, 2, 1.), (2, 1, 1.), (2, 2, 4.)];
        let mut m = build(2, &entries);
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let handles: Vec<_> = entries
            .iter()
            .map(|&(r, c, _)| m.get_element(r, c).unwrap())
            .collect();
        let load = |m: &mut Sparse<f64>, values: &[f64]| {
            m.clear();
            for (&h, &v) in handles.iter().zip(values) {
                m.set(h, v);
            }
        };
        let mut workspace = Vec::new();
        load(&mut m, &[3., 1., 1., 4.]);
        assert_eq!(m.factor_with_workspace(&mut workspace), Status::Okay);
        let initial = m.solve(&[4., 5.]);
        let address = workspace.as_ptr();
        let capacity = workspace.capacity();
        load(&mut m, &[1., 1., 1., 1.]);
        assert_eq!(m.factor_with_workspace(&mut workspace), Status::ZeroDiag);
        load(&mut m, &[3., 1., 1., 4.]);
        assert_eq!(m.factor_with_workspace(&mut workspace), Status::Okay);
        assert_eq!(workspace.as_ptr(), address);
        assert_eq!(workspace.capacity(), capacity);
        assert_eq!(
            m.factors()
                .unwrap()
                .solve(&[4., 5.], &mut workspace)
                .unwrap(),
            initial
        );
        assert_eq!(workspace.as_ptr(), address);
    }

    /// External row 2 has no element: as with ngspice's `Translate`, the
    /// internal order is 2 and `spSolve` (in place in `SMPsolve`) leaves the
    /// right-hand side in the untranslated entry.
    #[test]
    fn untranslated_rows_keep_their_right_hand_side() {
        let entries = [(1, 1, 2.0), (1, 3, 1.0), (3, 1, 1.0), (3, 3, 4.0)];
        let mut m = build(3, &entries);
        assert!(!m.fully_translated());
        assert_eq!(m.untranslated().collect::<Vec<_>>(), [2]);
        m.mna_preorder();
        assert_eq!(m.order_and_factor(1e-3, 1e-13), Status::Okay);
        let x = m.solve(&[3.0, 7.5, 5.0]);
        assert_eq!(x[1], 7.5);
        assert!((2.0 * x[0] + x[2] - 3.0).abs() < 1e-15 && (x[0] + 4.0 * x[2] - 5.0).abs() < 1e-15);
        assert_eq!(m.solve_transposed(&[3.0, -1.0, 5.0])[1], -1.0);
        let (mantissa, exponent) = m.factors().unwrap().determinant().unwrap();
        assert!((mantissa * 10f64.powi(exponent) - 7.0).abs() < 1e-12);
    }

    /// No element passes the absolute threshold, so `SearchEntireMatrix`
    /// takes the largest and reports `spSMALL_PIVOT` with its position.
    #[test]
    fn small_pivot_is_located() {
        let mut m = build(2, &[(1, 1, 1e-14), (2, 2, 2e-14)]);
        assert_eq!(m.order_and_factor(1e-3, 1e-13), Status::SmallPivot);
        assert_eq!(m.small_pivot(), Some((2, 2)));
        assert_eq!(m.solve(&[1e-14, 2e-14]), [1.0, 1.0]);
        let mut m = build(2, &[(1, 1, 1e-14), (2, 2, 2e-14)]);
        assert_eq!(m.order_and_factor(1e-3, 1e-16), Status::Okay);
        assert_eq!(m.small_pivot(), None);
    }

    #[test]
    fn singular_and_determinant() {
        let mut m = build(2, &[(1, 1, 1.0), (1, 2, 2.0), (2, 1, 2.0), (2, 2, 4.0)]);
        assert_eq!(m.order_and_factor(1e-3, 0.0), Status::Singular);
        let mut m = build(2, &[(1, 1, 3.0), (1, 2, 1.0), (2, 1, 2.0), (2, 2, 4.0)]);
        assert_eq!(m.order_and_factor(1e-3, 0.0), Status::Okay);
        let (mantissa, exponent) = m.factors().unwrap().determinant().unwrap();
        assert!((mantissa * 10f64.powi(exponent) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn complex_solve_matches_product() {
        let entries = [
            (1, 1, Complex::new(1.0, 2.0)),
            (1, 2, Complex::new(0.0, -1.0)),
            (2, 1, Complex::new(0.0, -1.0)),
            (2, 2, Complex::new(2.0, 0.5)),
        ];
        let mut m: Sparse<Complex> = Sparse::new(2).unwrap();
        for &(r, c, v) in &entries {
            let e = m.get_element(r, c).unwrap();
            m.add(e, v);
        }
        assert_eq!(m.order_and_factor(1e-3, 0.0), Status::Okay);
        let b = [Complex::real(1.0), Complex::new(0.0, 1.0)];
        let x = m.solve(&b);
        for row in 1..=2 {
            let mut sum = Complex::ZERO;
            for &(r, c, v) in &entries {
                if r == row {
                    sum = sum + v * x[c - 1];
                }
            }
            assert!((sum - b[row - 1]).magnitude() < 1e-14);
        }
    }

    #[test]
    fn larger_ladder_fills_and_solves() {
        let n = 200;
        let mut entries = Vec::new();
        for i in 1..=n {
            entries.push((i, i, 2.0 + 1e-3 * i as f64));
            if i > 1 {
                entries.push((i, i - 1, -1.0));
                entries.push((i - 1, i, -1.0));
            }
        }
        entries.push((1, n, 0.5));
        entries.push((n, 1, 0.25));
        let mut m = build(n, &entries);
        assert_eq!(m.order_and_factor(1e-3, 1e-13), Status::Okay);
        let b: Vec<_> = (0..n).map(|i| (i as f64).sin()).collect();
        let x = m.solve(&b);
        assert!(residual(&entries, &x, &b) < 1e-10);
    }
    #[test]
    #[ignore = "manual repeated reload/refactor benchmark"]
    fn sparse_reload_refactor_benchmark() {
        use std::{hint::black_box, time::Instant};
        const N: usize = 1024;
        const RUNS: u32 = 1000;
        let entries: Vec<_> = (1..=N)
            .flat_map(|i| [(i, i, 5.), (i, i % N + 1, -0.25), (i % N + 1, i, -0.5)])
            .collect();
        let originals: Vec<_> = entries.iter().map(|&(_, _, v)| v).collect();
        let mut m = Sparse::from_originals(N, entries).unwrap();
        assert_eq!(m.order_and_factor(1e-3, 0.), Status::Okay);
        let mut scratch = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            for _ in 0..RUNS {
                m.reload_originals(black_box(&originals));
                assert_eq!(m.factor_with_workspace(&mut scratch), Status::Okay);
                black_box(&m);
            }
            let duration = start.elapsed();
            let rhs = vec![1.; N];
            let x = m.solve(&rhs);
            let checksum = x.iter().fold(0_u64, |sum, v| sum.wrapping_add(v.to_bits()));
            eprintln!(
                "reload/refactor n={N} runs={RUNS} elapsed={duration:?} checksum={checksum:016x}"
            );
        }
    }
}
