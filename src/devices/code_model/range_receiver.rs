//! An ordered, hysteretic differential-voltage receiver. Range identity,
//! not the output logic label, is the state of this receiver.

use super::{Drive, Logic, crossing_time};
use crate::primitives::{ExactSum, Real, SpiceError, SpiceResult};
use std::sync::Arc;

/// One inclusive voltage interval and the drive selected by that interval.
#[derive(Clone, Copy, Debug)]
pub struct VoltageRange {
    minimum: Real,
    maximum: Real,
    drive: Drive,
}

impl VoltageRange {
    /// Construct a finite ordered interval.
    ///
    /// # Errors
    /// Invalid voltage limits.
    pub fn new(minimum: Real, maximum: Real, drive: Drive) -> SpiceResult<Self> {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err(SpiceError::circuit("invalid receiver voltage range"));
        }
        Ok(Self {
            minimum,
            maximum,
            drive,
        })
    }

    fn contains(self, voltage: Real) -> bool {
        self.minimum <= voltage && voltage <= self.maximum
    }
}

/// Policy for a voltage outside all declared ranges.
#[derive(Clone, Copy, Debug)]
pub enum GapPolicy {
    /// Reject a converged candidate.
    Error,
    /// Use the nearest interval; ties use the cyclic scan order.
    Nearest,
    /// Use an explicit drive, retaining the previous cyclic scan index.
    Drive(Drive),
}

/// The accepted state of one receiver occurrence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeState {
    index: usize,
    gap: bool,
}

impl RangeState {
    /// Most recently selected range (also the cyclic scan origin).
    #[must_use = "range identity determines the subsequent scan order"]
    pub const fn index(self) -> usize {
        self.index
    }

    /// Whether the sample was outside every range.
    #[must_use = "gap status distinguishes a fallback from a matching range"]
    pub const fn in_gap(self) -> bool {
        self.gap
    }
}

/// A candidate receiver state and output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RangeSample {
    /// Candidate range identity and gap status.
    pub state: RangeState,
    /// Selected digital drive.
    pub drive: Drive,
}

/// Immutable receiver material, shared by elaborated occurrences. The
/// mutable range index is kept by the accepted analog/event transaction.
#[derive(Clone, Debug)]
pub struct RangeReceiver {
    ranges: Arc<[VoltageRange]>,
    boundaries: Arc<[Real]>,
    initial: usize,
    gap: GapPolicy,
}

impl RangeReceiver {
    /// Construct an ordered receiver. No sorting or deduplication of
    /// ranges is performed; duplicated logic labels and overlapping
    /// ranges are intentional. The initial scan index is explicit.
    ///
    /// # Errors
    /// Empty range set or invalid initial index.
    pub fn new(ranges: Vec<VoltageRange>, initial: usize, gap: GapPolicy) -> SpiceResult<Self> {
        if initial >= ranges.len() {
            return Err(SpiceError::circuit("invalid receiver initial range index"));
        }
        let mut boundaries: Vec<_> = ranges.iter().flat_map(|r| [r.minimum, r.maximum]).collect();
        boundaries.sort_by(Real::total_cmp);
        boundaries.dedup_by(|a, b| *a == *b);
        if matches!(gap, GapPolicy::Nearest) {
            // In an uncovered interval, only the neighboring endpoints
            // can be nearest. Include their midpoint as a possible
            // nearest-selection boundary; extra covered midpoints are
            // harmless and are filtered by the state comparison.
            let midpoints: Vec<_> = boundaries.windows(2).map(|p| p[0].midpoint(p[1])).collect();
            boundaries.extend(midpoints);
            boundaries.sort_by(Real::total_cmp);
            boundaries.dedup_by(|a, b| *a == *b);
        }
        Ok(Self {
            ranges: ranges.into(),
            boundaries: boundaries.into(),
            initial,
            gap,
        })
    }

    /// Bias scan starts at the configured range, without inherited
    /// state from a different analysis.
    #[must_use = "the state belongs to an individual receiver occurrence"]
    pub const fn initial_state(&self) -> RangeState {
        RangeState {
            index: self.initial,
            gap: false,
        }
    }

    fn scan(&self, index: usize) -> impl Iterator<Item = usize> + '_ {
        (index..self.ranges.len()).chain(0..index)
    }

    pub(crate) fn select(&self, voltage: Real, previous: RangeState) -> SpiceResult<RangeSample> {
        if !voltage.is_finite() || previous.index >= self.ranges.len() {
            return Err(SpiceError::circuit("invalid receiver voltage or state"));
        }
        if let Some(index) = self
            .scan(previous.index)
            .find(|&index| self.ranges[index].contains(voltage))
        {
            return Ok(RangeSample {
                state: RangeState { index, gap: false },
                drive: self.ranges[index].drive,
            });
        }
        let (index, drive) = match self.gap {
            GapPolicy::Error => (previous.index, Drive::strong(Logic::Unknown)),
            GapPolicy::Drive(drive) => (previous.index, drive),
            GapPolicy::Nearest => {
                let mut below = None::<(usize, Real)>;
                let mut above = None::<(usize, Real)>;
                for index in self.scan(previous.index) {
                    let range = self.ranges[index];
                    if range.maximum < voltage && below.is_none_or(|(_, v)| range.maximum > v) {
                        below = Some((index, range.maximum));
                    }
                    if range.minimum > voltage && above.is_none_or(|(_, v)| range.minimum < v) {
                        above = Some((index, range.minimum));
                    }
                }
                let index = match (below, above) {
                    (Some((lo, a)), Some((hi, b))) => {
                        let mut difference = ExactSum::new();
                        difference.add_product(voltage, 2.);
                        difference.add_product(a, -1.);
                        difference.add_product(b, -1.);
                        let sign = difference.rounded_sum();
                        if sign < 0. {
                            lo
                        } else if sign > 0. {
                            hi
                        } else {
                            self.scan(previous.index)
                                .find(|&i| i == lo || i == hi)
                                .ok_or_else(|| SpiceError::circuit("invalid nearest range scan"))?
                        }
                    }
                    (Some((index, _)), None) | (None, Some((index, _))) => index,
                    (None, None) => return Err(SpiceError::circuit("no receiver range")),
                };
                (index, self.ranges[index].drive)
            }
        };
        Ok(RangeSample {
            state: RangeState { index, gap: true },
            drive,
        })
    }

    pub(crate) const fn rejects_gaps(&self) -> bool {
        matches!(self.gap, GapPolicy::Error)
    }

    /// Evaluate a converged analog candidate without modifying the
    /// prior state. Explicit error policy applies only to candidates,
    /// not to interpolation probes.
    ///
    /// # Errors
    /// Invalid voltage/state or a gap under error policy.
    pub fn sample(&self, voltage: Real, previous: RangeState) -> SpiceResult<RangeSample> {
        let sample = self.select(voltage, previous)?;
        if sample.state.gap && self.rejects_gaps() {
            return Err(SpiceError::circuit(format!(
                "receiver voltage {voltage:e} outside all ranges"
            )));
        }
        Ok(sample)
    }

    /// Earliest range identity or gap-status change along a linearly
    /// interpolated trial. This includes hidden intermediate changes
    /// even when the endpoint has the same output logic label.
    ///
    /// # Errors
    /// Invalid interval or receiver state.
    pub fn crossing(
        &self,
        start: (Real, Real),
        end: (Real, Real),
        previous: RangeState,
    ) -> SpiceResult<Option<Real>> {
        let (t0, v0) = start;
        let (t1, v1) = end;
        if ![t0, v0, t1, v1].into_iter().all(Real::is_finite) || t0 < 0. || t1 <= t0 {
            return Err(SpiceError::circuit(
                "invalid ordered receiver crossing interval",
            ));
        }
        if v0 == v1 {
            return Ok(None);
        }
        let rising = v1 > v0;
        let step = |v: Real| {
            if rising {
                v.next_up().min(v1)
            } else {
                v.next_down().max(v1)
            }
        };
        let within = |v: Real| {
            if rising {
                v0 <= v && v <= v1
            } else {
                v1 <= v && v <= v0
            }
        };
        let crossing = |boundary: Real| -> SpiceResult<Option<Real>> {
            if !within(boundary) {
                return Ok(None);
            }
            let at = if boundary == v0 {
                t0
            } else if boundary == v1 {
                t1
            } else if let Some(at) = crossing_time(start, end, boundary) {
                at
            } else {
                return Ok(None);
            };
            if self.select(boundary, previous)?.state != previous {
                return Ok(Some(at));
            }
            if self.select(step(boundary), previous)?.state != previous {
                return Ok(Some(at.next_up().min(t1)));
            }
            Ok(None)
        };
        if self.select(step(v0), previous)?.state != previous {
            return Ok(Some(t0.next_up().min(t1)));
        }
        if rising {
            for &boundary in self.boundaries.iter() {
                if let Some(at) = crossing(boundary)? {
                    return Ok(Some(at));
                }
            }
        } else {
            for &boundary in self.boundaries.iter().rev() {
                if let Some(at) = crossing(boundary)? {
                    return Ok(Some(at));
                }
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receiver(gap: GapPolicy) -> SpiceResult<RangeReceiver> {
        RangeReceiver::new(
            vec![
                VoltageRange::new(0., 2., Drive::strong(Logic::Zero))?,
                VoltageRange::new(3., 5., Drive::strong(Logic::One))?,
                VoltageRange::new(1., 4., Drive::strong(Logic::Zero))?,
                VoltageRange::new(4., 6., Drive::strong(Logic::Rising))?,
            ],
            0,
            gap,
        )
    }

    #[test]
    fn overlapping_repeated_labels_retain_distinct_range_identity() -> SpiceResult<()> {
        let r = receiver(GapPolicy::Nearest)?;
        let a = r.sample(1.5, r.initial_state())?;
        let b = r.sample(2.5, a.state)?;
        assert_eq!(a.drive, b.drive);
        assert_eq!((a.state.index(), b.state.index()), (0, 2));
        let c = r.sample(1.5, b.state)?;
        assert_eq!(c.state.index(), 2);
        // From index 2 the next matching entry at 4.5 is index 3,
        // not index 1. Wrap to index 1 only when index 3 exits.
        let d = r.sample(4.5, c.state)?;
        assert_eq!(d.drive.logic, Logic::Rising);
        let e = r.sample(3.5, d.state)?;
        assert_eq!(e.state.index(), 1);
        Ok(())
    }

    #[test]
    fn hidden_crossing_and_inclusive_boundaries() -> SpiceResult<()> {
        let r = receiver(GapPolicy::Nearest)?;
        let initial = r.sample(1., r.initial_state())?.state;
        assert_eq!(
            r.crossing((10., 1.), (14., 5.), initial)?,
            Some(11_f64.next_up())
        );
        assert_eq!(r.sample(2., initial)?.state, initial);
        let later = r.sample(2_f64.next_up(), initial)?.state;
        assert_eq!(later.index(), 2);
        assert_eq!(
            r.crossing((12., 3.), (15., 0.), later)?,
            Some(14_f64.next_up())
        );
        Ok(())
    }

    #[test]
    fn gap_nearest_ties_follow_cyclic_order_and_error_is_explicit() -> SpiceResult<()> {
        let ranges = vec![
            VoltageRange::new(-2., -1., Drive::strong(Logic::Zero))?,
            VoltageRange::new(1., 2., Drive::strong(Logic::One))?,
        ];
        let r = RangeReceiver::new(ranges.clone(), 0, GapPolicy::Nearest)?;
        let a = r.sample(0., r.initial_state())?;
        assert!(a.state.in_gap());
        assert_eq!(a.state.index(), 0);
        let b = r.sample(0.1, a.state)?;
        assert_eq!(b.state.index(), 1);
        assert_eq!(r.sample(0., b.state)?.state.index(), 1);
        assert_eq!(
            r.crossing((1., -0.5), (2., 0.5), a.state)?,
            Some(1.5_f64.next_up())
        );
        let r = RangeReceiver::new(ranges, 0, GapPolicy::Error)?;
        assert!(r.sample(0., r.initial_state()).is_err());
        assert!(
            r.crossing((1., -1.), (2., 0.), r.initial_state())?
                .is_some()
        );
        Ok(())
    }
}
