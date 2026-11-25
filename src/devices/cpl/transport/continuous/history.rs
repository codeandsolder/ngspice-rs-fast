//! Immutable accepted PWL storage. The old and tentative accepted
//! versions share complete live blocks; only the pruned prefix and
//! appendable suffix need copying. Times and values remain binary64.
use std::{mem::size_of, sync::Arc};

use super::{Knot, Point, Real, SpiceResult, Wave, WaveValue, append, invalid, prune};

const BLOCK: usize = 128;

pub(super) trait WaveRead {
    type Value: WaveValue;

    fn len(&self) -> usize;
    fn get_hint(&self, index: usize, hint: &mut usize) -> Option<&Point<Self::Value>>;
    fn partition_point(&self, predicate: impl FnMut(&Point<Self::Value>) -> bool) -> usize;
}

impl<Value: WaveValue> WaveRead for [Point<Value>] {
    type Value = Value;

    fn len(&self) -> usize {
        <[Point<Value>]>::len(self)
    }

    fn get_hint(&self, index: usize, _hint: &mut usize) -> Option<&Point<Value>> {
        self.get(index)
    }

    fn partition_point(&self, predicate: impl FnMut(&Point<Value>) -> bool) -> usize {
        <[Point<Value>]>::partition_point(self, predicate)
    }
}

impl<Value: WaveValue> WaveRead for Vec<Point<Value>> {
    type Value = Value;

    fn len(&self) -> usize {
        Self::len(self)
    }

    fn get_hint(&self, index: usize, _hint: &mut usize) -> Option<&Point<Value>> {
        self.get(index)
    }

    fn partition_point(&self, predicate: impl FnMut(&Point<Value>) -> bool) -> usize {
        <[Point<Value>]>::partition_point(self, predicate)
    }
}

impl<Value: WaveValue, const N: usize> WaveRead for [Point<Value>; N] {
    type Value = Value;

    fn len(&self) -> usize {
        N
    }

    fn get_hint(&self, index: usize, _hint: &mut usize) -> Option<&Point<Value>> {
        self.get(index)
    }

    fn partition_point(&self, predicate: impl FnMut(&Point<Value>) -> bool) -> usize {
        <[Point<Value>]>::partition_point(self, predicate)
    }
}

#[derive(Debug, Clone)]
struct Block {
    points: Arc<[Point<Real>]>,
    end: usize,
}

impl Block {
    fn start(&self) -> usize {
        self.end - self.points.len()
    }

    fn contains(&self, index: usize) -> bool {
        index >= self.start() && index < self.end
    }

    fn bytes(&self) -> usize {
        size_of::<usize>() * 2 + self.points.len() * size_of::<Point<Real>>()
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct KnownWave {
    blocks: Vec<Block>,
}

impl PartialEq for KnownWave {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl KnownWave {
    pub(super) fn single(point: Point<Real>) -> Self {
        let mut wave = Self::default();
        wave.push_block(Arc::from([point]));
        wave
    }

    pub(super) fn len(&self) -> usize {
        self.blocks.last().map_or(0, |block| block.end)
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &Point<Real>> {
        self.blocks.iter().flat_map(|block| block.points.iter())
    }

    pub(super) fn clear(&mut self) {
        self.blocks.clear();
    }

    fn push_block(&mut self, points: Arc<[Point<Real>]>) {
        if !points.is_empty() {
            self.blocks.push(Block {
                end: self.len() + points.len(),
                points,
            });
        }
    }

    fn append_slice(&mut self, points: &[Point<Real>]) {
        for points in points.chunks(BLOCK) {
            self.push_block(Arc::from(points));
        }
    }

    fn append_range(&mut self, old: &Self, first: usize, end: usize) {
        let start = old.blocks.partition_point(|block| block.end <= first);
        for block in &old.blocks[start..] {
            if block.start() >= end {
                break;
            }
            let left = first.saturating_sub(block.start());
            let right = end.min(block.end) - block.start();
            if left == 0 && right == block.points.len() {
                self.push_block(Arc::clone(&block.points));
            } else if left < right {
                self.append_slice(&block.points[left..right]);
            }
        }
    }

    // Copy the last block so flat-run coalescing has the last two
    // samples available, including across a one-point block boundary.
    fn tail_start(&self, first: usize) -> usize {
        let Some(last) = self.blocks.last() else {
            return first;
        };
        let start = if last.points.len() == 1 && self.blocks.len() > 1 {
            self.blocks[self.blocks.len() - 2].start()
        } else {
            last.start()
        };
        first.max(start)
    }

    pub(super) fn workspace(&self) -> usize {
        self.blocks.capacity() * size_of::<Block>()
            + self.blocks.iter().map(Block::bytes).sum::<usize>()
    }

    // Blocks can only be shared from this wave's immediately preceding
    // accepted version. Their sorted times identify the candidate block
    // without an allocation or a global pointer set.
    pub(super) fn unique_workspace(&self, old: &Self) -> usize {
        let mut previous = 0;
        let mut bytes = self.blocks.capacity() * size_of::<Block>();
        for block in &self.blocks {
            let time = block.points[0].time;
            while old
                .blocks
                .get(previous)
                .is_some_and(|entry| entry.points[0].time < time)
            {
                previous += 1;
            }
            if !old
                .blocks
                .get(previous)
                .is_some_and(|entry| Arc::ptr_eq(&entry.points, &block.points))
            {
                bytes = bytes.saturating_add(block.bytes());
            }
        }
        bytes
    }
}

impl WaveRead for KnownWave {
    type Value = Real;

    fn len(&self) -> usize {
        self.len()
    }

    fn get_hint(&self, index: usize, hint: &mut usize) -> Option<&Point<Real>> {
        if !self
            .blocks
            .get(*hint)
            .is_some_and(|block| block.contains(index))
        {
            let following = hint.saturating_add(1);
            if self
                .blocks
                .get(following)
                .is_some_and(|block| block.contains(index))
            {
                *hint = following;
            } else {
                *hint = self.blocks.partition_point(|block| block.end <= index);
            }
        }
        let block = self.blocks.get(*hint)?;
        block.points.get(index - block.start())
    }

    fn partition_point(&self, mut predicate: impl FnMut(&Point<Real>) -> bool) -> usize {
        let index = self
            .blocks
            .partition_point(|block| block.points.last().is_some_and(&mut predicate));
        self.blocks.get(index).map_or(self.len(), |block| {
            block.start() + block.points.partition_point(predicate)
        })
    }
}

pub(super) struct RetainedWindow {
    first: usize,
    incoming: usize,
}

impl RetainedWindow {
    pub(super) fn new<Value>(
        old: &KnownWave,
        new: &[Knot<Value>],
        start: Real,
        cutoff: Real,
    ) -> Self {
        let incoming = new.partition_point(|knot| knot.time <= start);
        let expired = new.partition_point(|knot| knot.time <= cutoff);
        if expired > incoming {
            Self {
                first: old.len(),
                incoming: expired - 1,
            }
        } else {
            Self {
                first: old
                    .partition_point(|knot| knot.time <= cutoff)
                    .saturating_sub(1),
                incoming,
            }
        }
    }

    // Bound both the mutable suffix and its immutable block copy, the
    // partial prefix block, and all successor block descriptors.
    pub(super) fn allocation_bound(&self, old: &KnownWave, new: usize) -> usize {
        let tail = old.len() - old.tail_start(self.first);
        let copied = (new - self.incoming)
            .saturating_add(tail)
            .saturating_add(BLOCK);
        let blocks = old
            .blocks
            .len()
            .saturating_add(copied.div_ceil(BLOCK))
            .saturating_add(2);
        copied
            .saturating_mul(2 * size_of::<Point<Real>>())
            .saturating_add(blocks.saturating_mul(size_of::<Block>() + 2 * size_of::<usize>()))
    }

    pub(super) fn merge<Value: WaveValue>(
        &self,
        old: &KnownWave,
        new: Wave<Value>,
        cutoff: Real,
        delta: &[Real],
    ) -> SpiceResult<KnownWave> {
        let tail_start = old.tail_start(self.first);
        let count = old.len() - tail_start + new.len() - self.incoming;
        let mut result = KnownWave {
            blocks: Vec::with_capacity(old.blocks.len() + count.div_ceil(BLOCK) + 1),
        };
        result.append_range(old, self.first, tail_start);
        let mut tail = Vec::with_capacity(count);
        let mut hint = 0;
        for index in tail_start..old.len() {
            let point = old
                .get_hint(index, &mut hint)
                .ok_or_else(|| invalid("invalid CPL retained history range"))?;
            tail.push(point.clone());
        }
        for knot in new.into_iter().skip(self.incoming) {
            let value = knot.wave.evaluate(delta);
            if !value.is_finite() {
                return Err(invalid("nonfinite accepted CPL internal wave"));
            }
            append(
                &mut tail,
                Point {
                    time: knot.time,
                    wave: value,
                },
            );
        }
        prune(&mut tail, cutoff);
        result.append_slice(&tail);
        result.blocks.shrink_to_fit();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_window_preserves_cutoff_and_flat_run_semantics() {
        let old_points: Vec<_> = (0..260)
            .map(|index| Point {
                time: Real::from(index),
                wave: if index >= 258 {
                    257.
                } else {
                    Real::from(index)
                },
            })
            .collect();
        let mut old = KnownWave::default();
        old.append_slice(&old_points);
        let new = vec![
            Point {
                time: 259.,
                wave: 257.,
            },
            Point {
                time: 260.,
                wave: 257.,
            },
            Point {
                time: 261.,
                wave: 261.,
            },
            Point {
                time: 262.,
                wave: 262.,
            },
        ];
        for cutoff in [0.5, 127.5, 128., 256., 258., 259., 260., 260.5, 262.] {
            let mut reference = old_points.clone();
            for point in new.iter().skip(1) {
                append(&mut reference, point.clone());
            }
            prune(&mut reference, cutoff);
            let window = RetainedWindow::new(&old, &new, 259., cutoff);
            let next = window.merge(&old, new.clone(), cutoff, &[]).unwrap();
            // A coalesced flat run may retain a different (equivalent)
            // predecessor, but the entire live interval is unchanged.
            for time in [cutoff, cutoff.midpoint(262.), 262.] {
                assert_eq!(
                    super::super::sample(&next, time, 0.).unwrap(),
                    super::super::sample(&reference, time, 0.).unwrap()
                );
            }
            assert!(next.unique_workspace(&old) <= window.allocation_bound(&old, new.len()));
        }
    }

    #[test]
    fn shared_blocks_retain_exact_samples_and_account_unique_allocations() {
        let old_points: Vec<_> = (0..512)
            .map(|index| Point {
                time: Real::from(index),
                wave: Real::from(index % 7),
            })
            .collect();
        let mut old = KnownWave::default();
        old.append_slice(&old_points);
        let new = vec![
            Point {
                time: 512.,
                wave: 3.,
            },
            Point {
                time: 513.,
                wave: 4.,
            },
        ];
        let window = RetainedWindow::new(&old, &new, 511., 64.5);
        let next = window.merge(&old, new, 64.5, &[]).unwrap();
        let mut expected = old_points[64..].to_vec();
        expected.extend([
            Point {
                time: 512.,
                wave: 3.,
            },
            Point {
                time: 513.,
                wave: 4.,
            },
        ]);
        assert!(next.iter().eq(expected.iter()));
        assert!(old.iter().eq(old_points.iter()));
        assert!(next.unique_workspace(&old) < next.workspace());
        assert!(next.unique_workspace(&old) <= window.allocation_bound(&old, 2));
        let mut hint = 0;
        for (index, expected) in expected.iter().enumerate() {
            assert_eq!(next.get_hint(index, &mut hint), Some(expected));
        }
        assert!(next.get_hint(expected.len(), &mut hint).is_none());
        for time in [0., 64., 64.5, 127., 128., 256., 511., 513., 514.] {
            assert_eq!(
                next.partition_point(|p| p.time <= time),
                expected.partition_point(|p| p.time <= time)
            );
        }
    }
}
