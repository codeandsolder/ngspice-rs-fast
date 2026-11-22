//! Coordinate-indexed direct objectives over a monotonic accepted stream.

use super::{BoundObservation, ObservationReduction, ObservationState};
use crate::analysis::ordered_samples::same_coordinate;
use crate::primitives::{Complex, Real, SpiceResult};
use std::ops::Range;

/// Direct point queries are scheduled by coordinate, independent of result
/// order. Their operands are bound to the union of their raw dependencies;
/// last/trace observations still use the producer row directly.
pub(super) struct QueryPlan {
    targets: Vec<(Real, usize)>,
    columns: Vec<usize>,
    pub(super) ordinary: Vec<usize>,
}

impl QueryPlan {
    pub(super) fn bind(observations: &mut [BoundObservation]) -> SpiceResult<Self> {
        let mut targets = Vec::new();
        let mut columns = Vec::new();
        let mut ordinary = Vec::new();
        for (index, observation) in observations.iter().enumerate() {
            if let ObservationReduction::At(at) = observation.reduction {
                targets.push((at, index));
                columns.extend(observation.column.input_columns());
            } else {
                ordinary.push(index);
            }
        }
        targets.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        columns.sort_unstable();
        columns.dedup();
        for &(_, index) in &targets {
            observations[index].column.remap_inputs(&columns)?;
        }
        Ok(Self {
            targets,
            columns,
            ordinary,
        })
    }
}

/// Two compact raw rows and the sorted query range at the previous coordinate.
/// On a monotonic stream each query is dispatched once, plus validation of
/// every sample of an exact coordinate group. No scan of unrelated targets.
pub(super) struct QueryRows {
    previous_x: Option<Real>,
    exhausted: bool,
    group: Range<usize>,
    previous: Vec<Complex>,
    current: Vec<Complex>,
}

impl QueryRows {
    pub(super) fn new(plan: &QueryPlan) -> Self {
        Self {
            previous_x: None,
            exhausted: plan.targets.is_empty(),
            group: 0..0,
            previous: Vec::with_capacity(plan.columns.len()),
            current: Vec::with_capacity(plan.columns.len()),
        }
    }

    fn intervals(&self, targets: &[(Real, usize)], x: Real) -> (Range<usize>, Range<usize>) {
        match self.previous_x {
            None => {
                let lo = targets.partition_point(|target| target.0 < x);
                let hi = lo + targets[lo..].partition_point(|target| target.0 <= x);
                (lo..hi, lo..hi)
            }
            Some(previous) if x > previous => {
                let start = self.group.end;
                let mut lo = start;
                while targets.get(lo).is_some_and(|target| target.0 < x) {
                    lo += 1;
                }
                let mut hi = lo;
                while targets
                    .get(hi)
                    .is_some_and(|target| same_coordinate(target.0, x))
                {
                    hi += 1;
                }
                (start..hi, lo..hi)
            }
            Some(previous) if x < previous => {
                let end = self.group.start;
                let mut hi = end;
                while hi > 0 && targets[hi - 1].0 > x {
                    hi -= 1;
                }
                let mut lo = hi;
                while lo > 0 && same_coordinate(targets[lo - 1].0, x) {
                    lo -= 1;
                }
                (lo..end, lo..hi)
            }
            Some(_) => (self.group.clone(), self.group.clone()),
        }
    }

    pub(super) fn observe(
        &mut self,
        plan: &QueryPlan,
        bound: &[BoundObservation],
        states: &mut [ObservationState],
        x: Real,
        row: &[Complex],
    ) {
        if self.exhausted {
            return;
        }
        let (visit, group) = self.intervals(&plan.targets, x);
        // Once traversal has moved beyond every target in its direction,
        // neither operand reads nor row copies are useful. The outer axis
        // tracker still validates every coordinate, including later errors.
        // A nonempty exact group must remain live for subsequent duplicates.
        self.exhausted = group.is_empty()
            && self.previous_x.is_some_and(|previous| {
                (x > previous && group.end == plan.targets.len())
                    || (x < previous && group.start == 0)
            });
        if self.exhausted && visit.is_empty() {
            return;
        }
        self.current.clear();
        self.current
            .extend(plan.columns.iter().map(|&index| row[index]));
        for &(at, index) in &plan.targets[visit] {
            let state = &mut states[index];
            let observation = &bound[index];
            if !same_coordinate(at, x)
                && let Some(previous_x) = self.previous_x
            {
                state.observe(observation, &self.previous, Some(previous_x), None);
            }
            state.observe(observation, &self.current, Some(x), None);
        }
        std::mem::swap(&mut self.previous, &mut self.current);
        self.previous_x = Some(x);
        self.group = group;
    }
}
