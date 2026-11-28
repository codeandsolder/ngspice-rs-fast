//! Guarded driver waveforms and their exact observation projection.
//! A branch is a sequence of posted transport requests, not a guessed
//! chronological sequence. Actual delivery uses the last posted request
//! whose arrival has occurred; later posting supersedes an older
//! arrival when its physical delivery precedes that older event.

use super::conditional::exact_at;
use super::{Arrival, EventDelta, Guarded, Origin, TimingRelation, TimingWindow, charge};
use crate::devices::code_model::{Drive, Logic, Strength};
use crate::primitives::{ExactSum, Real, SpiceError, SpiceResult};
use std::cmp::Ordering;
use std::collections::BTreeMap;

/// One posted transition, retaining its physical-time expression and
/// conditional same-time delta ordering.
#[derive(Clone, Debug)]
pub struct WaveformRequest {
    arrival: Arrival,
    delta: EventDelta,
}

impl WaveformRequest {
    /// Bind an arrival to its explicit or propagated delta semantics.
    #[must_use = "the request retains arrival and delta provenance"]
    pub fn new(arrival: Arrival, delta: EventDelta) -> Self {
        Self { arrival, delta }
    }
    /// Stable requested endpoints and physical arrival expression.
    #[must_use = "the request retains physical arrival provenance"]
    pub const fn arrival(&self) -> &Arrival {
        &self.arrival
    }
    /// Same-time ordering specification.
    #[must_use = "the delta determines causal equality order"]
    pub const fn delta(&self) -> &EventDelta {
        &self.delta
    }
}

/// A complete transport waveform within one guard.
#[derive(Clone, Debug)]
pub struct DriveTrace {
    initial: Drive,
    requests: Vec<WaveformRequest>,
}

impl DriveTrace {
    /// Construct a logical request sequence. Physical delivery may
    /// differ in order because independently delayed paths can overtake
    /// one another; the requested endpoints must remain consistent.
    ///
    /// # Errors
    /// A directional initial phase, or inconsistent requested endpoints.
    pub fn new(initial: Drive, requests: Vec<WaveformRequest>) -> SpiceResult<Self> {
        if matches!(initial.logic, Logic::Rising | Logic::Falling) {
            return Err(SpiceError::circuit(
                "guarded waveform initial drive is not stable",
            ));
        }
        let mut prior = initial;
        for request in &requests {
            if prior != request.arrival.from() {
                return Err(SpiceError::circuit(
                    "inconsistent guarded waveform endpoints",
                ));
            }
            prior = request.arrival.to();
        }
        Ok(Self { initial, requests })
    }

    /// Construct a source waveform whose requests use delta zero.
    ///
    /// # Errors
    /// Invalid stable drive or inconsistent requested endpoints.
    pub fn source(initial: Drive, arrivals: Vec<Arrival>) -> SpiceResult<Self> {
        Self::new(
            initial,
            arrivals
                .into_iter()
                .map(|arrival| WaveformRequest::new(arrival, EventDelta::Fixed(0)))
                .collect(),
        )
    }

    /// Drive before any request has physically arrived.
    #[must_use = "the initial drive is part of the guarded trace"]
    pub const fn initial(&self) -> Drive {
        self.initial
    }

    /// Transport requests in posting order.
    #[must_use = "posting order determines physical supersession"]
    pub fn requests(&self) -> &[WaveformRequest] {
        &self.requests
    }

    /// The final requested drive after all events complete.
    #[must_use = "the final drive follows the last posted request"]
    pub fn final_drive(&self) -> Drive {
        self.requests
            .last()
            .map_or(self.initial, |request| request.arrival.to())
    }

    fn realize_at(
        &self,
        origins: &BTreeMap<Origin, Real>,
        time: Real,
        budget: &mut usize,
    ) -> SpiceResult<Drive> {
        let mut observation = ExactSum::new();
        observation.add_product(time, 1.);
        for request in self.requests.iter().rev() {
            if exact_at(request.arrival.window(), origins, budget)?.exact_cmp(&observation)
                != Ordering::Greater
            {
                return Ok(request.arrival.to());
            }
        }
        Ok(self.initial)
    }
}

/// A complete partition of transport waveforms on shared delay origins.
/// Its guards and arrival expressions are kept intact across fanout.
#[derive(Clone, Debug)]
pub struct GuardedWaveform {
    traces: Guarded<DriveTrace>,
}

impl GuardedWaveform {
    /// Bind an exhaustive, mutually exclusive guarded trace partition.
    #[must_use = "the waveform retains the complete guarded trace partition"]
    pub fn new(traces: Guarded<DriveTrace>) -> Self {
        Self { traces }
    }

    /// The complete symbolic transport traces.
    #[must_use = "the branches preserve cross-signal delay correlation"]
    pub const fn traces(&self) -> &Guarded<DriveTrace> {
        &self.traces
    }

    /// Actual drive at one observation in a concrete origin assignment.
    ///
    /// # Errors
    /// Invalid observation, missing origin or exhausted work budget.
    pub fn realize_at(
        &self,
        origins: &BTreeMap<Origin, Real>,
        time: Real,
        mut budget: usize,
    ) -> SpiceResult<Drive> {
        TimingWindow::exact(time)?;
        self.traces
            .realize_with_budget(origins, &mut budget)?
            .realize_at(origins, time, &mut budget)
    }

    /// Project all guarded branches to exact possible drive sets at
    /// representable observation times. This is a union of feasible
    /// states, not a union of unconditional arrival bounds.
    ///
    /// # Errors
    /// Exhausted exact projection budget, or inconsistent coverage.
    pub fn project(&self, mut budget: usize) -> SpiceResult<WaveformProjection> {
        let mut boundaries = BTreeMap::<u64, Vec<(Drive, bool)>>::new();
        for (guard, trace) in self.traces.branches() {
            let mut conditions = Vec::new();
            for (drive, request) in trace
                .requests
                .iter()
                .rev()
                .map(|request| (request.arrival.to(), Some(request)))
                .chain(std::iter::once((trace.initial, None)))
            {
                charge(&mut budget, conditions.len().saturating_add(1))?;
                if let Some(request) = request {
                    conditions.push((request.arrival.window().clone(), TimingRelation::NoLater));
                }
                let intervals =
                    guard.observation_intervals_with_budget(&conditions, &mut budget)?;
                for interval in intervals {
                    charge(&mut budget, 2)?;
                    boundaries
                        .entry(interval.first().to_bits())
                        .or_default()
                        .push((drive, true));
                    if interval.last() < Real::MAX {
                        boundaries
                            .entry(interval.last().next_up().to_bits())
                            .or_default()
                            .push((drive, false));
                    }
                }
                // All requests later in posting order must be absent
                // for this older drive to be visible.
                if let Some((_, relation)) = conditions.last_mut() {
                    *relation = TimingRelation::After;
                }
            }
        }
        let mut active = BTreeMap::<(Strength, Logic), usize>::new();
        let mut changes = Vec::<ProjectedDriveChange>::new();
        for (time, events) in boundaries {
            charge(
                &mut budget,
                events.len().saturating_add(active.len()).saturating_add(1),
            )?;
            // Apply all additions/removals before observing the drive
            // set. Adjacent projected intervals must not create an
            // artificial zero-width event.
            for (drive, add) in events {
                let key = (drive.strength, drive.logic);
                if add {
                    *active.entry(key).or_default() += 1;
                } else {
                    let count = active.get_mut(&key).ok_or_else(|| {
                        SpiceError::circuit("unbalanced guarded waveform projection")
                    })?;
                    *count -= 1;
                    if *count == 0 {
                        active.remove(&key);
                    }
                }
            }
            if active.is_empty() {
                return Err(SpiceError::circuit(
                    "guarded waveform has no feasible drive at observation",
                ));
            }
            let possible: Vec<_> = active
                .keys()
                .map(|&(strength, logic)| Drive { strength, logic })
                .collect();
            if changes
                .last()
                .is_none_or(|change| change.possible != possible)
            {
                changes.push(ProjectedDriveChange {
                    time: Real::from_bits(time),
                    possible,
                });
            }
        }
        if changes.first().is_none_or(|change| change.time != 0.) {
            return Err(SpiceError::circuit(
                "guarded waveform does not cover initial observation",
            ));
        }
        Ok(WaveformProjection { changes })
    }
}

/// Change in the feasible drive set at one representable physical time.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedDriveChange {
    time: Real,
    possible: Vec<Drive>,
}

impl ProjectedDriveChange {
    /// Inclusive physical time at which this drive set begins.
    #[must_use = "the time is a guarded waveform breakpoint"]
    pub const fn time(&self) -> Real {
        self.time
    }

    /// All and only drives with a feasible joint origin realization.
    #[must_use = "the set describes the exact feasible drives"]
    pub fn possible(&self) -> &[Drive] {
        &self.possible
    }
}

/// Sorted observation boundaries of a guarded transport waveform.
#[derive(Clone, Debug)]
pub struct WaveformProjection {
    changes: Vec<ProjectedDriveChange>,
}

impl WaveformProjection {
    /// Boundaries of changes to the exact feasible drive set.
    #[must_use = "each change is a possible physical breakpoint"]
    pub fn changes(&self) -> &[ProjectedDriveChange] {
        &self.changes
    }

    /// Exact set of possible drives at a representable observation time.
    ///
    /// # Errors
    /// Invalid physical observation time.
    pub fn possible_at(&self, time: Real) -> SpiceResult<&[Drive]> {
        TimingWindow::exact(time)?;
        let index = self.changes.partition_point(|change| change.time <= time);
        self.changes
            .get(index.saturating_sub(1))
            .map(ProjectedDriveChange::possible)
            .ok_or_else(|| SpiceError::circuit("empty guarded waveform projection"))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{DelayWindow, OrderPartition};
    use super::*;

    #[test]
    fn optional_pulse_projection_uses_guarded_intermediate_boundary() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let a = TimingWindow::exact(10.)?.after(DelayWindow::new(0., 20.)?, u)?;
        let b = TimingWindow::exact(15.)?.after(DelayWindow::new(0., 10.)?, u)?;
        let partition = OrderPartition::new(vec![a.clone(), b.clone()], 100_000)?;
        let traces = partition.evaluate(100_000, |order| {
            let requests = if order.groups() == [vec![0], vec![1]] {
                vec![
                    Arrival::new(low, high, a.clone())?,
                    Arrival::new(high, low, b.clone())?,
                ]
            } else {
                Vec::new()
            };
            DriveTrace::source(low, requests)
        })?;
        let waveform = GuardedWaveform::new(Guarded::from_conditional(traces));
        let projection = waveform.project(1_000_000)?;
        assert_eq!(
            projection
                .changes()
                .iter()
                .map(ProjectedDriveChange::time)
                .collect::<Vec<_>>(),
            [0., 10., 20.]
        );
        assert_eq!(projection.possible_at(10.)?, [low, high]);
        assert_eq!(projection.possible_at(20.)?, [low]);
        for fraction in [0., 0.25, 0.5, 0.75, 1.] {
            for time in [0., 10., 12.5, 15., 17.5, 20., 25.] {
                let expected = if fraction < 0.5
                    && time >= 10. + 20. * fraction
                    && time < 15. + 10. * fraction
                {
                    high
                } else {
                    low
                };
                let actual =
                    waveform.realize_at(&BTreeMap::from([(u, fraction)]), time, 100_000)?;
                assert_eq!(actual, expected);
                assert!(projection.possible_at(time)?.contains(&actual));
            }
        }
        Ok(())
    }

    #[test]
    fn reconvergent_guards_do_not_create_independent_cross_products() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let v = Origin::shared(1);
        let a = TimingWindow::exact(0.)?.after(DelayWindow::new(0., 10.)?, u)?;
        let b = TimingWindow::exact(0.)?.after(DelayWindow::new(0., 10.)?, v)?;
        let order = || OrderPartition::new(vec![a.clone(), b.clone()], 100_000);
        let left = Guarded::from_conditional(
            order()?.evaluate(100_000, |case| Ok(case.groups().to_vec()))?,
        );
        let right = Guarded::from_conditional(
            order()?.evaluate(100_000, |case| Ok(case.groups().to_vec()))?,
        );
        let pair = left.zip(&right, 1_000_000, |a, b| Ok((a.clone(), b.clone())))?;
        assert_eq!(pair.branches().count(), 3);
        assert!(pair.branches().all(|(_, (a, b))| a == b));
        Ok(())
    }

    #[test]
    fn transport_projection_respects_later_posted_supersession() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let rise = TimingWindow::exact(10.)?.after(DelayWindow::new(0., 10.)?, u)?;
        let fall = TimingWindow::exact(15.)?;
        let waveform = GuardedWaveform::new(Guarded::unconditional(DriveTrace::source(
            low,
            vec![
                Arrival::new(low, high, rise)?,
                Arrival::new(high, low, fall)?,
            ],
        )?));
        let projected = waveform.project(1_000_000)?;
        assert_eq!(projected.possible_at(10.)?, [low, high]);
        assert_eq!(projected.possible_at(15.)?, [low]);
        assert_eq!(
            waveform.realize_at(&BTreeMap::from([(u, 1.)]), 25., 100_000)?,
            low
        );
        Ok(())
    }
}
