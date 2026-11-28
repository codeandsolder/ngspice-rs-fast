//! Streaming symbolic construction with creation-order handles and sorted rows.
//!
//! The ordinary sparse row uses a bounded sorted chain. A high-degree row
//! promotes its coordinates to compact open-addressed slots, avoiding
//! quadratic scans without charging the ordinary rows for a hash table.
//! Extension builds only the additional coordinates and merges sorted rows.

use std::hash::{BuildHasher, RandomState};

use super::{
    ElementId, PatternIndex, compact_count, compact_position, copy_entries, pattern_error,
    pattern_rows, push_compact, reserve,
};
use crate::primitives::{SpiceResult, filled_vec};

/// At most one cache-line's worth of handle links is searched per sparse row.
/// Dense rows switch to indexed membership at this bound.
const ROW_LINEAR_LIMIT: u8 = 16;

impl PatternIndex {
    pub(super) fn row_bounds(&self, row: usize) -> (usize, usize) {
        (
            self.row_start[row] as usize,
            self.row_start[row + 1] as usize,
        )
    }

    pub(super) fn find(&self, row: usize, col: u32) -> Option<ElementId> {
        let (start, end) = self.row_bounds(row);
        self.row_cols[start..end]
            .binary_search(&col)
            .ok()
            .map(|offset| ElementId(self.row_ids[start + offset]))
    }

    fn row(&self, row: usize) -> (&[u32], &[u32]) {
        let (start, end) = self.row_bounds(row);
        (&self.row_cols[start..end], &self.row_ids[start..end])
    }

    fn try_copy(&self) -> SpiceResult<Self> {
        Ok(Self {
            n: self.n,
            positions: copy_entries(&self.positions)?,
            row_start: copy_entries(&self.row_start)?,
            row_cols: copy_entries(&self.row_cols)?,
            row_ids: copy_entries(&self.row_ids)?,
        })
    }
}

/// A slot contains a one-based creation-order handle, never a copied key.
/// Only coordinates in promoted rows occupy this table.
#[derive(Default)]
struct DenseIndex {
    slots: Vec<u32>,
    used: usize,
    seed: u64,
}

impl DenseIndex {
    fn slot(&self, position: (u32, u32), positions: &[(u32, u32)]) -> usize {
        let packed = (u64::from(position.0) << 32) | u64::from(position.1);
        // SplitMix64's avalanche finalizer on a per-builder randomized key.
        let mut hash = packed.wrapping_add(self.seed);
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        hash ^= hash >> 31;
        let mask = self.slots.len() - 1;
        let mut slot = (hash & mask as u64) as usize;
        loop {
            let handle = self.slots[slot];
            if handle == 0 || positions[handle_index(handle)] == position {
                return slot;
            }
            slot = (slot + 1) & mask;
        }
    }

    fn ensure_room(&mut self, additional: usize, positions: &[(u32, u32)]) -> SpiceResult<()> {
        let needed = self
            .used
            .checked_add(additional)
            .ok_or_else(|| pattern_error("dense coordinate count overflows"))?;
        let mut capacity = self.slots.len().max(32);
        while needed > capacity - capacity / 4 {
            capacity = capacity
                .checked_mul(2)
                .ok_or_else(|| pattern_error("dense coordinate index capacity overflows"))?;
        }
        if capacity != self.slots.len() {
            if self.slots.is_empty() {
                self.seed = RandomState::new().hash_one(0_u64);
            }
            let slots = filled_vec(capacity, 0, "sparse dense-coordinate index")?;
            let old = std::mem::replace(&mut self.slots, slots);
            for handle in old.into_iter().filter(|&handle| handle != 0) {
                let slot = self.slot(positions[handle_index(handle)], positions);
                self.slots[slot] = handle;
            }
        }
        Ok(())
    }

    fn vacant(
        &mut self,
        position: (u32, u32),
        positions: &[(u32, u32)],
    ) -> SpiceResult<Option<usize>> {
        // A duplicate remains allocation-free, even exactly at the load limit.
        let slot = self.slot(position, positions);
        if self.slots[slot] != 0 {
            return Ok(None);
        }
        let capacity = self.slots.len();
        self.ensure_room(1, positions)?;
        Ok(Some(if self.slots.len() == capacity {
            slot
        } else {
            self.slot(position, positions)
        }))
    }

    fn record(&mut self, slot: usize, handle: u32) {
        self.slots[slot] = handle;
        self.used += 1;
    }
}

fn handle_index(handle: u32) -> usize {
    (handle - 1) as usize
}

/// The row heads are eventually reused as the finished prefix offsets.
/// Links and the saturated degree are temporary; only promoted rows are
/// present in the dense index. Zero is the link sentinel, not an element ID.
struct Builder {
    positions: Vec<(u32, u32)>,
    heads: Vec<u32>,
    links: Vec<u32>,
    degrees: Vec<u8>,
    dense: DenseIndex,
    remaining: u32,
}

impl Builder {
    fn new(rows: usize, remaining: u32) -> SpiceResult<Self> {
        Ok(Self {
            positions: Vec::new(),
            heads: filled_vec(rows, 0, "sparse row heads")?,
            links: Vec::new(),
            degrees: filled_vec(rows - 1, 0, "sparse row degrees")?,
            dense: DenseIndex::default(),
            remaining,
        })
    }

    fn append(&mut self, position: (u32, u32), next: u32) -> SpiceResult<u32> {
        let handle = compact_count(self.positions.len())?
            .checked_add(1)
            .filter(|&handle| handle <= self.remaining)
            .ok_or_else(|| pattern_error("compact pattern count exceeds u32"))?;
        reserve(&mut self.positions, 1)?;
        reserve(&mut self.links, 1)?;
        self.positions.push(position);
        self.links.push(next);
        Ok(handle)
    }

    fn insert(&mut self, position: (u32, u32)) -> SpiceResult<()> {
        let row = position.0 as usize;
        if self.degrees[row] == ROW_LINEAR_LIMIT {
            self.insert_dense(row, position)
        } else {
            self.insert_sparse(row, position)
        }
    }

    fn insert_sparse(&mut self, row: usize, position: (u32, u32)) -> SpiceResult<()> {
        let mut previous = 0;
        let mut next = self.heads[row];
        while next != 0 {
            let index = handle_index(next);
            let col = self.positions[index].1;
            if col >= position.1 {
                if col == position.1 {
                    return Ok(());
                }
                break;
            }
            previous = next;
            next = self.links[index];
        }
        let handle = self.append(position, next)?;
        if previous == 0 {
            self.heads[row] = handle;
        } else {
            self.links[handle_index(previous)] = handle;
        }
        self.degrees[row] += 1;
        if self.degrees[row] == ROW_LINEAR_LIMIT {
            self.promote(row)?;
        }
        Ok(())
    }

    fn promote(&mut self, row: usize) -> SpiceResult<()> {
        self.dense
            .ensure_room(usize::from(ROW_LINEAR_LIMIT), &self.positions)?;
        for handle in RowChain::new(self.heads[row], &self.links) {
            let slot = self
                .dense
                .slot(self.positions[handle_index(handle)], &self.positions);
            self.dense.record(slot, handle);
        }
        Ok(())
    }

    fn insert_dense(&mut self, row: usize, position: (u32, u32)) -> SpiceResult<()> {
        if let Some(slot) = self.dense.vacant(position, &self.positions)? {
            let handle = self.append(position, self.heads[row])?;
            self.heads[row] = handle;
            self.dense.record(slot, handle);
        }
        Ok(())
    }

    fn finish(self, n: usize, previous: Option<&PatternIndex>) -> SpiceResult<PatternIndex> {
        let Self {
            positions,
            mut heads,
            links,
            degrees,
            dense,
            remaining: _,
        } = self;
        // The membership table is no longer needed while building the index.
        drop(dense);
        let base = previous.map_or(0, |index| index.positions.len());
        let total = base
            .checked_add(positions.len())
            .ok_or_else(|| pattern_error("compact entry count overflows"))?;
        let count = compact_count(total)?;
        let base_handle = compact_count(base)?;
        let mut output = RowOutput::new(total)?;
        let mut dense_order = Vec::new();
        for (row, (head, degree)) in heads.iter_mut().zip(&degrees).enumerate() {
            let chain = RowChain::new(*head, &links);
            *head = compact_count(output.ids.len())?;
            let old = previous.map_or((&[][..], &[][..]), |index| index.row(row));
            if *degree == ROW_LINEAR_LIMIT {
                dense_order.clear();
                for handle in chain {
                    push_compact(&mut dense_order, handle)?;
                }
                dense_order.sort_unstable_by_key(|&handle| positions[handle_index(handle)].1);
                output.append_row(old, dense_order.iter().copied(), &positions, base_handle);
            } else {
                output.append_row(old, chain, &positions, base_handle);
            }
        }
        heads[n] = count;
        drop(links);
        drop(degrees);
        drop(dense_order);
        let positions = match previous {
            Some(index) => {
                let mut combined = Vec::new();
                reserve(&mut combined, total)?;
                combined.extend_from_slice(&index.positions);
                combined.extend(positions);
                combined
            }
            None => positions,
        };
        Ok(PatternIndex {
            n,
            positions,
            row_start: heads,
            row_cols: output.cols,
            row_ids: output.ids,
        })
    }
}

struct RowChain<'a> {
    next: u32,
    links: &'a [u32],
}

impl<'a> RowChain<'a> {
    fn new(next: u32, links: &'a [u32]) -> Self {
        Self { next, links }
    }
}

impl Iterator for RowChain<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let handle = self.next;
        if handle == 0 {
            return None;
        }
        self.next = self.links[handle_index(handle)];
        Some(handle)
    }
}

/// Output capacity is checked once from the combined unique count.
struct RowOutput {
    cols: Vec<u32>,
    ids: Vec<u32>,
}

impl RowOutput {
    fn new(total: usize) -> SpiceResult<Self> {
        let mut output = Self {
            cols: Vec::new(),
            ids: Vec::new(),
        };
        reserve(&mut output.cols, total)?;
        reserve(&mut output.ids, total)?;
        Ok(output)
    }

    fn push(&mut self, col: u32, id: u32) {
        self.cols.push(col);
        self.ids.push(id);
    }

    fn append_row(
        &mut self,
        old: (&[u32], &[u32]),
        handles: impl Iterator<Item = u32>,
        positions: &[(u32, u32)],
        base: u32,
    ) {
        let mut old = old.0.iter().copied().zip(old.1.iter().copied()).peekable();
        for handle in handles {
            let index = handle_index(handle);
            let col = positions[index].1;
            while let Some(&(old_col, old_id)) = old.peek() {
                if old_col >= col {
                    break;
                }
                self.push(old_col, old_id);
                old.next();
            }
            // Builder::finish checked the combined count against u32::MAX.
            self.push(col, base + (handle - 1));
        }
        for (col, id) in old {
            self.push(col, id);
        }
    }
}

pub(super) fn build(
    n: usize,
    coordinates: impl IntoIterator<Item = (usize, usize)>,
    previous: Option<&PatternIndex>,
) -> SpiceResult<PatternIndex> {
    let rows = pattern_rows(n)?;
    let remaining = u32::MAX - compact_count(previous.map_or(0, |index| index.positions.len()))?;
    let mut builder = None;
    for (row, col) in coordinates {
        let position = compact_position(n, row, col)?;
        if previous.is_some_and(|index| index.find(row, position.1).is_some()) {
            continue;
        }
        let scratch = match &mut builder {
            Some(scratch) => scratch,
            None => builder.insert(Builder::new(rows, remaining)?),
        };
        scratch.insert(position)?;
    }
    match (builder, previous) {
        (Some(scratch), _) => scratch.finish(n, previous),
        (None, Some(index)) => index.try_copy(),
        (None, None) => Ok(PatternIndex {
            n,
            positions: Vec::new(),
            row_start: filled_vec(rows, 0, "sparse pattern")?,
            row_cols: Vec::new(),
            row_ids: Vec::new(),
        }),
    }
}
