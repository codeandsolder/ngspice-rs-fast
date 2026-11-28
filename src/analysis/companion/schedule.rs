//! Resolved transient timing: output sampling and adaptive integration
//! ceilings are independent partitions of one continuous physical run.

use crate::analysis::linear::unsupported;
use crate::primitives::{Real, SpiceResult};
use std::sync::Arc;

/// One interval of a piecewise-constant timing law. Boundaries are absolute
/// simulation times; the interval starts at the preceding boundary (or zero).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimingSegment {
    /// Exclusive upper time boundary, except the final stop.
    pub until: Real,
    /// Positive output interval or adaptive step ceiling.
    pub step: Real,
}

/// Validated, source-independent timing for a continuous transient.
/// Output times form a lazy grid starting at the output gate, including each
/// subsequent output-segment boundary and final stop exactly once.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientSchedule {
    start: Real,
    output: Arc<[TimingSegment]>,
    ceiling: Arc<[TimingSegment]>,
    minimum_ceiling: Real,
}

fn validate(segments: &[TimingSegment]) -> SpiceResult<Real> {
    let mut previous = 0.;
    for segment in segments {
        if !segment.until.is_finite()
            || segment.until <= previous
            || !segment.step.is_finite()
            || segment.step <= 0.
            || previous + segment.step <= previous
        {
            return Err(unsupported("invalid transient timing segment"));
        }
        previous = segment.until;
    }
    if segments.is_empty() {
        return Err(unsupported("empty transient timing schedule"));
    }
    Ok(previous)
}

impl TransientSchedule {
    /// Build independent output and integration schedules. Both must cover
    /// the same complete simulation duration starting from zero.
    ///
    /// # Errors
    /// Empty schedules, nonfinite or nonpositive intervals, nonincreasing
    /// boundaries, mismatched endpoints, or an invalid output gate.
    pub fn new(
        start: Real,
        output: Vec<TimingSegment>,
        ceiling: Vec<TimingSegment>,
    ) -> SpiceResult<Self> {
        let stop = validate(&output)?;
        let ceiling_stop = validate(&ceiling)?;
        if stop.to_bits() != ceiling_stop.to_bits()
            || !start.is_finite()
            || start < 0.
            || start >= stop
        {
            return Err(unsupported("inconsistent transient timing schedule"));
        }
        let minimum_ceiling = ceiling
            .iter()
            .fold(Real::INFINITY, |min, s| min.min(s.step));
        Ok(Self {
            start: if matches!(start, 0.) { 0. } else { start },
            minimum_ceiling,
            output: output.into(),
            ceiling: ceiling.into(),
        })
    }

    pub(super) fn bounds(&self) -> [Real; 4] {
        let output = self.output[0];
        let stop = self.output[self.output.len() - 1].until;
        [output.step, stop, self.start, self.ceiling[0].step]
    }

    pub(super) const fn minimum_ceiling(&self) -> Real {
        self.minimum_ceiling
    }

    pub(super) fn ceiling_at(&self, time: Real) -> TimingSegment {
        let index = self.ceiling.partition_point(|s| s.until <= time);
        self.ceiling[index.min(self.ceiling.len() - 1)]
    }

    pub(super) fn grid(&self) -> ScheduledGrid {
        ScheduledGrid {
            schedule: self.clone(),
            segment: self.output.partition_point(|s| s.until <= self.start),
            anchor: self.start,
            index: 0,
            time: self.start,
        }
    }
}

/// A lazy cursor; neither a fine output interval nor a large stop time
/// allocates a dense time vector. The driver owns the aggregate work limit.
pub(super) struct ScheduledGrid {
    schedule: TransientSchedule,
    segment: usize,
    anchor: Real,
    index: u32,
    time: Real,
}

impl ScheduledGrid {
    pub(super) const fn time(&self) -> Real {
        self.time
    }

    pub(super) fn advance(&mut self, on_sample: bool) -> SpiceResult<Real> {
        if !on_sample {
            return Ok(self.time);
        }
        if self.time >= self.schedule.output[self.segment].until {
            self.segment += 1;
            self.anchor = self.time;
            self.index = 0;
        }
        let segment = self
            .schedule
            .output
            .get(self.segment)
            .ok_or_else(|| unsupported("output schedule exhausted before transient stop"))?;
        self.index = self
            .index
            .checked_add(1)
            .ok_or_else(|| unsupported("transient output grid exceeds index range"))?;
        let next = Real::from(self.index)
            .mul_add(segment.step, self.anchor)
            .min(segment.until);
        if next <= self.time {
            return Err(unsupported(
                "transient output grid cannot advance at this time precision",
            ));
        }
        self.time = next;
        Ok(next)
    }
}
