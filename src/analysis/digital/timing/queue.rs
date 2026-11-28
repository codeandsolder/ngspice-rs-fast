//! Symbolic event queue. A branch selects its next physical event
//! group without choosing a scalar delay realization. Events tied at
//! one physical time are processed by delta group, then posting order.
//! Only feasible guards survive, so subsequent cancellation and state
//! changes take place in the same joint timing domain.

use super::{GuardStatus, TimingGuard, TimingRelation, TimingWindow, charge};
use crate::primitives::{SpiceError, SpiceResult};

/// Delta ordering for a physical request. A propagated request starts
/// at delta zero if its physical delay is positive; if that delay is
/// exactly zero, it follows the causative group. The latter decision
/// is made under a timing guard, not from an interval endpoint.
#[derive(Clone, Debug)]
pub enum EventDelta {
    /// Explicit same-time delta group.
    Fixed(usize),
    /// Delta depends on whether the physical arrival equals its cause.
    Propagated {
        /// Physical event that requests this delivery.
        cause: TimingWindow,
        /// Delta group of the causative event.
        parent_delta: usize,
    },
}

impl EventDelta {
    fn fixed(&self) -> SpiceResult<usize> {
        match self {
            Self::Fixed(delta) => Ok(*delta),
            Self::Propagated { .. } => Err(SpiceError::circuit("unresolved symbolic delta")),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Scheduled<T> {
    pub(super) time: TimingWindow,
    pub(super) delta: EventDelta,
    pub(super) sequence: u64,
    pub(super) payload: T,
}

#[derive(Clone, Debug)]
pub(super) struct SymbolicQueue<T> {
    events: Vec<Scheduled<T>>,
    sequence: u64,
}

impl<T> Default for SymbolicQueue<T> {
    fn default() -> Self {
        Self {
            events: Vec::new(),
            sequence: 0,
        }
    }
}

impl<T> SymbolicQueue<T> {
    pub(super) fn post(
        &mut self,
        time: TimingWindow,
        delta: EventDelta,
        payload: T,
    ) -> SpiceResult<u64> {
        let sequence = self.sequence;
        self.sequence = sequence
            .checked_add(1)
            .ok_or_else(|| SpiceError::circuit("symbolic event sequence overflow"))?;
        self.events.push(Scheduled {
            time,
            delta,
            sequence,
            payload,
        });
        Ok(sequence)
    }

    pub(super) fn cancel(&mut self, mut matches: impl FnMut(&T) -> bool) {
        self.events.retain(|event| !matches(&event.payload));
    }

    pub(super) fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[derive(Clone, Debug)]
pub(super) struct DueGroup<T> {
    pub(super) time: TimingWindow,
    pub(super) delta: usize,
    pub(super) events: Vec<Scheduled<T>>,
    pub(super) remaining: SymbolicQueue<T>,
}

impl<T: Clone> SymbolicQueue<T> {
    /// Select only the next feasible event group, not every full
    /// permutation of the queue. The pivot's unique (delta, sequence)
    /// tie-break chooses one representative for each physical group.
    ///
    /// The returned alternatives partition the supplied guard's domain.
    /// No child allocates an independent copy of a shared delay origin.
    pub(super) fn split_next(
        &self,
        condition: &TimingGuard,
        budget: &mut usize,
    ) -> SpiceResult<Vec<(TimingGuard, DueGroup<T>)>> {
        let mut pending = vec![(condition.clone(), self.clone())];
        let mut result = Vec::new();
        while let Some((guard, queue)) = pending.pop() {
            let Some((index, event)) = queue
                .events
                .iter()
                .enumerate()
                .find(|(_, event)| matches!(event.delta, EventDelta::Propagated { .. }))
            else {
                result.extend(queue.split_fixed(&guard, budget)?);
                continue;
            };
            let EventDelta::Propagated {
                cause,
                parent_delta,
            } = &event.delta
            else {
                return Err(SpiceError::circuit("missing symbolic delta cause"));
            };
            let invalid = guard.clone().and(TimingGuard::compare(
                event.time.clone(),
                cause.clone(),
                TimingRelation::Before,
            ));
            if invalid.status_with_budget(budget)? != GuardStatus::Impossible {
                return Err(SpiceError::circuit("symbolic event precedes its cause"));
            }
            for relation in [TimingRelation::Coincident, TimingRelation::After] {
                charge(budget, queue.events.len())?;
                let branch = guard.clone().and(TimingGuard::compare(
                    event.time.clone(),
                    cause.clone(),
                    relation,
                ));
                if branch.status_with_budget(budget)? != GuardStatus::Impossible {
                    let delta = if relation == TimingRelation::Coincident {
                        parent_delta
                            .checked_add(1)
                            .ok_or_else(|| SpiceError::circuit("symbolic delta overflow"))?
                    } else {
                        0
                    };
                    let mut queue = queue.clone();
                    queue.events[index].delta = EventDelta::Fixed(delta);
                    pending.push((branch, queue));
                }
            }
        }
        Ok(result)
    }

    fn split_fixed(
        &self,
        condition: &TimingGuard,
        budget: &mut usize,
    ) -> SpiceResult<Vec<(TimingGuard, DueGroup<T>)>> {
        let mut branches = Vec::new();
        for (pivot_index, pivot) in self.events.iter().enumerate() {
            charge(budget, self.events.len())?;
            let pivot_delta = pivot.delta.fixed()?;
            let mut first = condition.clone();
            for (index, event) in self.events.iter().enumerate() {
                if index == pivot_index {
                    continue;
                }
                let precedes =
                    (event.delta.fixed()?, event.sequence) < (pivot_delta, pivot.sequence);
                first = first.and(TimingGuard::compare(
                    pivot.time.clone(),
                    event.time.clone(),
                    if precedes {
                        TimingRelation::Before
                    } else {
                        TimingRelation::NoLater
                    },
                ));
            }
            if first.status_with_budget(budget)? == GuardStatus::Impossible {
                continue;
            }
            let mut groups = vec![(first, vec![pivot_index])];
            for (index, event) in self.events.iter().enumerate() {
                if index == pivot_index
                    || event.delta.fixed()? != pivot_delta
                    || event.sequence < pivot.sequence
                {
                    continue;
                }
                let mut next = Vec::new();
                for (guard, members) in groups {
                    for (relation, join) in [
                        (TimingRelation::Coincident, true),
                        (TimingRelation::Before, false),
                    ] {
                        charge(budget, members.len().saturating_add(1))?;
                        let guard = guard.clone().and(TimingGuard::compare(
                            pivot.time.clone(),
                            event.time.clone(),
                            relation,
                        ));
                        if guard.status_with_budget(budget)? != GuardStatus::Impossible {
                            let mut members = members.clone();
                            if join {
                                members.push(index);
                            }
                            next.push((guard, members));
                        }
                    }
                }
                groups = next;
            }
            for (guard, mut members) in groups {
                charge(budget, self.events.len().saturating_add(members.len()))?;
                members.sort_unstable();
                let mut events = Vec::with_capacity(members.len());
                let mut remaining = Self {
                    events: Vec::new(),
                    sequence: self.sequence,
                };
                let mut selected = members.into_iter().peekable();
                for (index, event) in self.events.iter().enumerate() {
                    if selected.peek() == Some(&index) {
                        selected.next();
                        events.push(event.clone());
                    } else {
                        remaining.events.push(event.clone());
                    }
                }
                // The vector is in posting sequence order by construction.
                branches.push((
                    guard,
                    DueGroup {
                        time: pivot.time.clone(),
                        delta: pivot_delta,
                        events,
                        remaining,
                    },
                ));
            }
        }
        if branches.is_empty() && !self.events.is_empty() {
            return Err(SpiceError::circuit(
                "symbolic queue has no feasible next group",
            ));
        }
        Ok(branches)
    }
}

/// Process the queue to completion with a transactional state reducer.
/// A reducer may post new symbolic events or cancel pending events.
/// Each branch receives its own state and queue; the exact shared guard
/// accompanies it through every subsequent event group.
pub(super) fn execute<T: Clone, S: Clone>(
    condition: TimingGuard,
    queue: SymbolicQueue<T>,
    initial: S,
    budget: &mut usize,
    mut evaluate: impl FnMut(
        &TimingGuard,
        &TimingWindow,
        usize,
        &[Scheduled<T>],
        &mut SymbolicQueue<T>,
        &mut S,
        &mut usize,
    ) -> SpiceResult<()>,
) -> SpiceResult<Vec<(TimingGuard, S)>> {
    let mut pending = vec![(condition, queue, initial)];
    let mut finished = Vec::new();
    while let Some((guard, queue, state)) = pending.pop() {
        charge(budget, 1)?;
        if queue.is_empty() {
            finished.push((guard, state));
            continue;
        }
        for (guard, group) in queue.split_next(&guard, budget)? {
            let mut state = state.clone();
            let mut remaining = group.remaining;
            evaluate(
                &guard,
                &group.time,
                group.delta,
                &group.events,
                &mut remaining,
                &mut state,
                budget,
            )?;
            pending.push((guard, remaining, state));
        }
    }
    Ok(finished)
}

#[cfg(test)]
mod tests {
    use super::super::{DelayWindow, Guarded, Origin};
    use super::*;
    use crate::primitives::Real;
    use std::collections::BTreeMap;

    #[test]
    fn exact_next_groups_preserve_equal_time_delta_order() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let a = TimingWindow::exact(0.)?.after(DelayWindow::new(1., 3.)?, u)?;
        let b = TimingWindow::exact(2.)?;
        let mut q = SymbolicQueue::default();
        q.post(a, EventDelta::Fixed(0), 'a')?;
        q.post(b.clone(), EventDelta::Fixed(0), 'b')?;
        q.post(b, EventDelta::Fixed(1), 'c')?;
        let result = Guarded::from_partitioned(execute(
            TimingGuard::default(),
            q,
            Vec::<Vec<char>>::new(),
            &mut 1_000_000,
            |_, _, _, events, _, state, _| {
                state.push(events.iter().map(|event| event.payload).collect());
                Ok(())
            },
        )?)?;
        for (value, expected) in [
            (0., vec![vec!['a'], vec!['b'], vec!['c']]),
            (0.5, vec![vec!['a', 'b'], vec!['c']]),
            (1., vec![vec!['b'], vec!['c'], vec!['a']]),
        ] {
            assert_eq!(
                result.realize(&BTreeMap::from([(u, value)]), 100_000)?,
                &expected
            );
        }
        Ok(())
    }

    #[test]
    fn cancellation_occurs_only_in_the_branch_where_input_precedes_deadline() -> SpiceResult<()> {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        enum Event {
            Input,
            Deadline,
        }
        let u = Origin::shared(0);
        let input = TimingWindow::exact(0.)?.after(DelayWindow::new(1., 3.)?, u)?;
        let deadline = TimingWindow::exact(2.)?;
        let mut q = SymbolicQueue::default();
        q.post(deadline, EventDelta::Fixed(0), Event::Deadline)?;
        q.post(input, EventDelta::Fixed(1), Event::Input)?;
        let result = Guarded::from_partitioned(execute(
            TimingGuard::default(),
            q,
            false,
            &mut 1_000_000,
            |_, _, _, events, queue, state, _| {
                for event in events {
                    match event.payload {
                        Event::Deadline => *state = true,
                        Event::Input => queue.cancel(|event| *event == Event::Deadline),
                    }
                }
                Ok(())
            },
        )?)?;
        for (fraction, expected) in [(0., false), (0.5, true), (1., true)] {
            assert_eq!(
                *result.realize(&BTreeMap::<Origin, Real>::from([(u, fraction)]), 100_000)?,
                expected
            );
        }
        Ok(())
    }
    #[test]
    fn zero_delay_realization_uses_causal_delta_positive_delay_uses_zero() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let cause = TimingWindow::exact(10.)?;
        let arrival = cause.after(DelayWindow::new(0., 2.)?, u)?;
        let mut q = SymbolicQueue::default();
        q.post(cause.clone(), EventDelta::Fixed(2), 'a')?;
        q.post(
            arrival,
            EventDelta::Propagated {
                cause,
                parent_delta: 2,
            },
            'b',
        )?;
        q.post(TimingWindow::exact(12.)?, EventDelta::Fixed(0), 'c')?;
        let result = Guarded::from_partitioned(execute(
            TimingGuard::default(),
            q,
            Vec::new(),
            &mut 1_000_000,
            |_, _, delta, events, _, state, _| {
                state.push((
                    delta,
                    events.iter().map(|event| event.payload).collect::<Vec<_>>(),
                ));
                Ok(())
            },
        )?)?;
        assert_eq!(
            result.realize(&BTreeMap::from([(u, 0.)]), 100_000)?,
            &[(2, vec!['a']), (3, vec!['b']), (0, vec!['c'])]
        );
        assert_eq!(
            result.realize(&BTreeMap::from([(u, 1.)]), 100_000)?,
            &[(2, vec!['a']), (0, vec!['b', 'c'])]
        );
        Ok(())
    }
    #[test]
    fn propagated_delta_rejects_feasible_noncausal_arrival() -> SpiceResult<()> {
        let cause = TimingWindow::exact(2.)?;
        let arrival =
            TimingWindow::exact(0.)?.after(DelayWindow::new(1., 3.)?, Origin::shared(0))?;
        let mut queue = SymbolicQueue::default();
        queue.post(
            arrival,
            EventDelta::Propagated {
                cause,
                parent_delta: 0,
            },
            (),
        )?;
        let error = queue
            .split_next(&TimingGuard::default(), &mut 1_000_000)
            .expect_err("a feasible noncausal branch must not silently disappear");
        assert!(error.to_string().contains("precedes its cause"));
        Ok(())
    }

    #[test]
    fn positive_delay_does_not_increment_parent_delta() -> SpiceResult<()> {
        let mut queue = SymbolicQueue::default();
        queue.post(
            TimingWindow::exact(2.)?,
            EventDelta::Propagated {
                cause: TimingWindow::exact(1.)?,
                parent_delta: usize::MAX,
            },
            (),
        )?;
        let groups = queue.split_next(&TimingGuard::default(), &mut 100_000)?;
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].1.delta, 0);
        Ok(())
    }
}
