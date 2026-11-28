//! Symbolic event queue. A branch selects its next physical event
//! group without choosing a scalar delay realization. Events tied at
//! one physical time are processed by delta group, then posting order.
//! Only feasible guards survive, so subsequent cancellation and state
//! changes take place in the same joint timing domain.

use super::{GuardStatus, Guarded, TimingGuard, TimingRelation, TimingWindow, charge};
use crate::primitives::{Real, SpiceError, SpiceResult};

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

#[derive(Clone, Debug)]
/// A pending or delivered symbolic event.
pub struct Scheduled<T> {
    /// Physical arrival expression.
    pub time: TimingWindow,
    /// Physical or propagated delta ordering.
    pub delta: EventDelta,
    /// Posting order within the branch.
    pub sequence: u64,
    /// User-defined event payload.
    pub payload: T,
}

#[derive(Clone, Debug)]
/// Pending events of one feasible symbolic branch.
pub struct SymbolicQueue<T> {
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
    /// Schedule a branch-local request. The executor checks causal
    /// ordering against its last settled group before delivery.
    ///
    /// # Errors
    /// Posting sequence exhaustion.
    pub fn post(&mut self, time: TimingWindow, delta: EventDelta, payload: T) -> SpiceResult<u64> {
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

    /// Remove requests matching the predicate from this branch.
    pub fn cancel(&mut self, mut matches: impl FnMut(&T) -> bool) {
        self.events.retain(|event| !matches(&event.payload));
    }

    /// Whether this branch has no pending events.
    #[must_use = "the result describes pending event work"]
    pub fn is_empty(&self) -> bool {
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
    /// Select only the next feasible event group. Partition first by
    /// physical time, then by delta among events tied at that time.
    /// An event beyond an observation horizon does not require resolving
    /// its possibly ambiguous delta group yet.
    #[cfg(test)]
    fn split_next(
        &self,
        condition: &TimingGuard,
        budget: &mut usize,
    ) -> SpiceResult<Vec<(TimingGuard, DueGroup<T>)>> {
        self.split_ready(condition, None, budget)
    }

    fn split_ready(
        &self,
        condition: &TimingGuard,
        horizon: Option<&TimingWindow>,
        budget: &mut usize,
    ) -> SpiceResult<Vec<(TimingGuard, DueGroup<T>)>> {
        let mut branches = Vec::new();
        for (pivot_index, pivot) in self.events.iter().enumerate() {
            charge(budget, self.events.len())?;
            let mut first = condition.clone();
            if let Some(horizon) = horizon {
                first = first.and(TimingGuard::compare(
                    pivot.time.clone(),
                    horizon.clone(),
                    TimingRelation::NoLater,
                ));
            }
            // Posting order chooses a unique representative of the
            // earliest physical time, independently of delta.
            for (index, event) in self.events.iter().enumerate() {
                if index == pivot_index {
                    continue;
                }
                first = first.and(TimingGuard::compare(
                    pivot.time.clone(),
                    event.time.clone(),
                    if index < pivot_index {
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
            for index in pivot_index + 1..self.events.len() {
                let event = &self.events[index];
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
            for (guard, members) in groups {
                self.resolve_tied_deltas(guard, &members, &mut branches, budget)?;
            }
        }
        if branches.is_empty() && horizon.is_none() && !self.events.is_empty() {
            return Err(SpiceError::circuit(
                "symbolic queue has no feasible next group",
            ));
        }
        Ok(branches)
    }

    fn quiet_guard(
        &self,
        condition: &TimingGuard,
        horizon: &TimingWindow,
        budget: &mut usize,
    ) -> SpiceResult<Option<TimingGuard>> {
        let mut quiet = condition.clone();
        charge(budget, self.events.len())?;
        for event in &self.events {
            quiet = quiet.and(TimingGuard::compare(
                event.time.clone(),
                horizon.clone(),
                TimingRelation::After,
            ));
        }
        Ok((quiet.status_with_budget(budget)? != GuardStatus::Impossible).then_some(quiet))
    }

    fn resolve_tied_deltas(
        &self,
        guard: TimingGuard,
        members: &[usize],
        output: &mut Vec<(TimingGuard, DueGroup<T>)>,
        budget: &mut usize,
    ) -> SpiceResult<()> {
        // Only tied events need their deltas fixed. Keep their resolved
        // delta on the remaining queue if another event wins this group.
        let mut partial = vec![(guard, Vec::<usize>::new())];
        for &index in members {
            let event = &self.events[index];
            let mut next = Vec::new();
            for (guard, mut deltas) in partial {
                charge(budget, deltas.len().saturating_add(1))?;
                match &event.delta {
                    EventDelta::Fixed(delta) => {
                        deltas.push(*delta);
                        next.push((guard, deltas));
                    }
                    EventDelta::Propagated {
                        cause,
                        parent_delta,
                    } => {
                        let invalid = guard.clone().and(TimingGuard::compare(
                            event.time.clone(),
                            cause.clone(),
                            TimingRelation::Before,
                        ));
                        if invalid.status_with_budget(budget)? != GuardStatus::Impossible {
                            return Err(SpiceError::circuit("symbolic event precedes its cause"));
                        }
                        for relation in [TimingRelation::Coincident, TimingRelation::After] {
                            let branch = guard.clone().and(TimingGuard::compare(
                                event.time.clone(),
                                cause.clone(),
                                relation,
                            ));
                            if branch.status_with_budget(budget)? != GuardStatus::Impossible {
                                let delta = if relation == TimingRelation::Coincident {
                                    parent_delta.checked_add(1).ok_or_else(|| {
                                        SpiceError::circuit("symbolic delta overflow")
                                    })?
                                } else {
                                    0
                                };
                                let mut deltas = deltas.clone();
                                deltas.push(delta);
                                next.push((branch, deltas));
                            }
                        }
                    }
                }
            }
            partial = next;
        }
        for (guard, deltas) in partial {
            charge(budget, self.events.len().saturating_add(members.len()))?;
            let delta = *deltas
                .iter()
                .min()
                .ok_or_else(|| SpiceError::circuit("empty symbolic physical event group"))?;
            let mut events = Vec::new();
            let mut remaining = Self {
                events: Vec::new(),
                sequence: self.sequence,
            };
            let mut tied = members.iter().zip(deltas).peekable();
            for (index, event) in self.events.iter().enumerate() {
                let mut event = event.clone();
                if let Some((member, fixed)) = tied.peek().copied()
                    && *member == index
                {
                    tied.next();
                    event.delta = EventDelta::Fixed(fixed);
                    if fixed == delta {
                        events.push(event);
                        continue;
                    }
                }
                remaining.events.push(event);
            }
            let time = events
                .first()
                .ok_or_else(|| SpiceError::circuit("empty symbolic delta group"))?
                .time
                .clone();
            output.push((
                guard,
                DueGroup {
                    time,
                    delta,
                    events,
                    remaining,
                },
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct FrontierBranch<T, S> {
    queue: SymbolicQueue<T>,
    state: S,
    cursor: Option<(TimingWindow, usize)>,
}

/// Incremental symbolic execution at representable observation times.
/// A frontier is a complete partition of state and pending events under
/// shared timing guards, not independent distributions of each net.
/// Clone a frontier for a speculative transient step; a failed advance
/// does not publish partially evaluated branches.
#[derive(Clone, Debug)]
pub struct GuardedFrontier<T, S> {
    time: Real,
    branches: Guarded<FrontierBranch<T, S>>,
}

impl<T: Clone, S: Clone> GuardedFrontier<T, S> {
    /// Start with an unconditional state and an empty pending queue.
    #[must_use = "the frontier owns transactional symbolic state"]
    pub fn new(state: S) -> Self {
        Self {
            time: 0.,
            branches: Guarded::unconditional(FrontierBranch {
                queue: SymbolicQueue::default(),
                state,
                cursor: None,
            }),
        }
    }

    /// The last accepted observation time.
    #[must_use = "the observation is the scheduling lower bound"]
    pub const fn time(&self) -> Real {
        self.time
    }

    /// Read the feasible guarded states at the accepted observation.
    pub fn states(&self) -> impl Iterator<Item = (&TimingGuard, &S)> {
        self.branches
            .branches()
            .map(|(guard, branch)| (guard, &branch.state))
    }

    /// Announce an external event. A request at the current observation
    /// follows any already settled equal-time group; a future request
    /// starts at delta zero. The same payload and timing expression are
    /// shared by every branch.
    ///
    /// # Errors
    /// A feasible request precedes the accepted observation, sequence
    /// space is exhausted, or the exact work budget is exceeded.
    pub fn announce(
        &mut self,
        time: TimingWindow,
        payload: T,
        mut budget: usize,
    ) -> SpiceResult<()> {
        let observation = TimingWindow::exact(self.time)?;
        let mut branches = Vec::new();
        for (guard, branch) in self.branches.branches() {
            charge(&mut budget, 1)?;
            let invalid = guard.clone().and(TimingGuard::compare(
                time.clone(),
                observation.clone(),
                TimingRelation::Before,
            ));
            if invalid.status_with_budget(&mut budget)? != GuardStatus::Impossible {
                return Err(SpiceError::circuit(
                    "symbolic announcement precedes observation",
                ));
            }
            let mut branch = branch.clone();
            let delta =
                branch
                    .cursor
                    .as_ref()
                    .map_or(EventDelta::Fixed(0), |(cause, parent_delta)| {
                        EventDelta::Propagated {
                            cause: cause.clone(),
                            parent_delta: *parent_delta,
                        }
                    });
            branch.queue.post(time.clone(), delta, payload.clone())?;
            branches.push((guard.clone(), branch));
        }
        self.branches = Guarded::from_partitioned(branches)?;
        Ok(())
    }

    /// Settle all events up to and including the observation, preserving
    /// events strictly later than it. The reducer must keep all mutable
    /// simulation state in its branch payload and queue.
    ///
    /// # Errors
    /// Backward/invalid observation, noncausal reducer, exhausted work
    /// or delta-cycle limit, or reducer failure. State and observation
    /// time remain unchanged on failure.
    pub fn advance_to(
        &mut self,
        time: Real,
        mut budget: usize,
        delta_limit: usize,
        mut evaluate: impl FnMut(
            &TimingGuard,
            &TimingWindow,
            usize,
            &[Scheduled<T>],
            &mut SymbolicQueue<T>,
            &mut S,
            &mut usize,
        ) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        if time < self.time {
            return Err(SpiceError::circuit("symbolic time cannot run backwards"));
        }
        let horizon = TimingWindow::exact(time)?;
        let pending = self
            .branches
            .branches()
            .map(|(guard, branch)| (guard.clone(), branch.clone()))
            .collect();
        let finished = run(
            pending,
            Some(&horizon),
            delta_limit,
            &mut budget,
            &mut evaluate,
        )?;
        self.branches = Guarded::from_partitioned(finished)?;
        self.time = time;
        Ok(())
    }
}

fn run<T: Clone, S: Clone>(
    mut pending: Vec<(TimingGuard, FrontierBranch<T, S>)>,
    horizon: Option<&TimingWindow>,
    delta_limit: usize,
    budget: &mut usize,
    evaluate: &mut impl FnMut(
        &TimingGuard,
        &TimingWindow,
        usize,
        &[Scheduled<T>],
        &mut SymbolicQueue<T>,
        &mut S,
        &mut usize,
    ) -> SpiceResult<()>,
) -> SpiceResult<Vec<(TimingGuard, FrontierBranch<T, S>)>> {
    let mut finished = Vec::new();
    while let Some((guard, branch)) = pending.pop() {
        charge(budget, 1)?;
        if branch.queue.is_empty() {
            finished.push((guard, branch));
            continue;
        }
        if let Some(horizon) = horizon
            && let Some(quiet) = branch.queue.quiet_guard(&guard, horizon, budget)?
        {
            finished.push((quiet, branch.clone()));
        }
        for (guard, group) in branch.queue.split_ready(&guard, horizon, budget)? {
            if group.delta >= delta_limit {
                return Err(SpiceError::circuit("symbolic delta-cycle limit exceeded"));
            }
            if let Some((prior, delta)) = &branch.cursor {
                let invalid = guard.clone().and(TimingGuard::compare(
                    group.time.clone(),
                    prior.clone(),
                    if group.delta <= *delta {
                        TimingRelation::NoLater
                    } else {
                        TimingRelation::Before
                    },
                ));
                if invalid.status_with_budget(budget)? != GuardStatus::Impossible {
                    return Err(SpiceError::circuit("symbolic event group is noncausal"));
                }
            }
            let mut state = branch.state.clone();
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
            pending.push((
                guard,
                FrontierBranch {
                    queue: remaining,
                    state,
                    cursor: Some((group.time, group.delta)),
                },
            ));
        }
    }
    Ok(finished)
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
    Ok(run(
        vec![(
            condition,
            FrontierBranch {
                queue,
                state: initial,
                cursor: None,
            },
        )],
        None,
        usize::MAX,
        budget,
        &mut evaluate,
    )?
    .into_iter()
    .map(|(guard, branch)| (guard, branch.state))
    .collect())
}

#[cfg(test)]
mod tests {
    use super::super::{DelayWindow, Origin};
    use super::*;
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
    #[test]
    fn frontier_preserves_pending_events_across_observations_and_late_announcements()
    -> SpiceResult<()> {
        let u = Origin::shared(0);
        let mut frontier = GuardedFrontier::new(Vec::<(char, usize)>::new());
        frontier.announce(
            TimingWindow::exact(1.)?.after(DelayWindow::new(0., 2.)?, u)?,
            'a',
            100_000,
        )?;
        let reduce = |_: &TimingGuard,
                      _: &TimingWindow,
                      delta: usize,
                      events: &[Scheduled<char>],
                      _: &mut SymbolicQueue<char>,
                      state: &mut Vec<(char, usize)>,
                      _: &mut usize| {
            state.extend(events.iter().map(|event| (event.payload, delta)));
            Ok(())
        };
        frontier.advance_to(2., 1_000_000, 32, reduce)?;
        assert_eq!(frontier.time(), 2.);
        let at_two = Guarded::from_partitioned(
            frontier
                .states()
                .map(|(guard, state)| (guard.clone(), state.clone()))
                .collect(),
        )?;
        for (value, expected) in [(0., vec![('a', 0)]), (0.5, vec![('a', 0)]), (1., vec![])] {
            assert_eq!(
                at_two.realize(&BTreeMap::from([(u, value)]), 100_000)?,
                &expected
            );
        }
        // A newly announced external event at a settled observation
        // belongs to the next delta only where a prior group occurred
        // at that exact observation, rather than at an earlier time.
        frontier.announce(TimingWindow::exact(2.)?, 'b', 100_000)?;
        frontier.advance_to(4., 1_000_000, 32, reduce)?;
        let finished = Guarded::from_partitioned(
            frontier
                .states()
                .map(|(guard, state)| (guard.clone(), state.clone()))
                .collect(),
        )?;
        for (value, expected) in [
            (0., vec![('a', 0), ('b', 0)]),
            (0.5, vec![('a', 0), ('b', 1)]),
            (1., vec![('b', 0), ('a', 0)]),
        ] {
            assert_eq!(
                finished.realize(&BTreeMap::from([(u, value)]), 100_000)?,
                &expected
            );
        }
        Ok(())
    }

    #[test]
    fn frontier_rejects_noncausal_reducer_and_rolls_back() -> SpiceResult<()> {
        let mut frontier = GuardedFrontier::new(0);
        frontier.announce(TimingWindow::exact(1.)?, (), 100_000)?;
        let error = frontier
            .advance_to(2., 100_000, 32, |_, time, _, _, queue, state, _| {
                *state += 1;
                queue.post(time.clone(), EventDelta::Fixed(0), ())?;
                Ok(())
            })
            .expect_err("a reducer may not repeat an already settled group");
        assert!(error.to_string().contains("noncausal"));
        assert_eq!(frontier.time(), 0.);
        assert!(frontier.states().all(|(_, state)| *state == 0));
        frontier.advance_to(2., 100_000, 32, |_, _, _, _, _, state, _| {
            *state += 1;
            Ok(())
        })?;
        assert!(frontier.states().all(|(_, state)| *state == 1));
        assert!(
            frontier
                .announce(TimingWindow::exact(1.)?, (), 100_000)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn quiet_horizon_does_not_partition_future_delta_uncertainty() -> SpiceResult<()> {
        let mut frontier = GuardedFrontier::new(());
        frontier.announce(TimingWindow::exact(1.)?, (), 100_000)?;
        let u = Origin::shared(0);
        frontier.advance_to(1., 100_000, 32, |_, time, delta, _, queue, _, _| {
            let future = time.after(DelayWindow::new(10., 11.)?, u)?;
            queue.post(
                future,
                EventDelta::Propagated {
                    cause: TimingWindow::exact(11.)?,
                    parent_delta: delta,
                },
                (),
            )?;
            Ok(())
        })?;
        frontier.advance_to(5., 100_000, 32, |_, _, _, _, _, _, _| {
            Err(SpiceError::circuit("future event must not execute"))
        })?;
        assert_eq!(frontier.states().count(), 1);
        Ok(())
    }
}
