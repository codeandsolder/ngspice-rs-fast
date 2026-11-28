//! Input waveform history for causal storage evaluation. Announcing the
//! beginning of a bounded edge records the edge exactly once; its
//! completion only settles the envelope and is not another edge.

use super::{Drive, EventContext, Logic};
use crate::analysis::digital::timing::{
    Arrival, Bounds, Guarded, TimedInput, TimingRelation, TimingWindow,
};
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::collections::VecDeque;

pub(super) const COMPARISON_BUDGET: usize = 100_000;

#[derive(Clone, Debug)]
pub(super) struct InputEdge {
    pub(super) from: Drive,
    pub(super) to: Drive,
    pub(super) window: TimingWindow,
    // Same physical time can contain distinct ordered delta groups.
    pub(super) announced: EventContext,
}

impl InputEdge {
    pub(super) fn logic_changed(&self) -> bool {
        self.from.logic != self.to.logic
    }

    pub(super) fn relative_to(&self, earlier: &Self) -> SpiceResult<Bounds> {
        self.window.relative_to(&earlier.window, COMPARISON_BUDGET)
    }

    pub(super) fn precedes_at_equality(&self, other: &Self) -> bool {
        // If both edges occur at the same physical time, a data edge in
        // the capture's group is sampled before the capture; one in a
        // later delta group is not. Nonexact coincident expressions use
        // the same ordering of their announced boundaries.
        (self.announced.time, self.announced.delta) <= (other.announced.time, other.announced.delta)
    }

    pub(super) fn occurs_before(&self, capture: &Self) -> SpiceResult<Relation> {
        let bounds = capture.relative_to(self)?;
        if bounds.minimum() > 0. || (bounds.minimum() == 0. && self.precedes_at_equality(capture)) {
            Ok(Relation::Before)
        } else if bounds.maximum() < 0.
            || (bounds.maximum() == 0. && !self.precedes_at_equality(capture))
        {
            Ok(Relation::After)
        } else {
            Ok(Relation::Either)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Relation {
    Before,
    After,
    Either,
}

#[derive(Clone, Debug)]
pub(super) struct PortHistory {
    initial: Drive,
    latest: Drive,
    last_arrival: Option<Arrival>,
    edges: VecDeque<InputEdge>,
}

impl PortHistory {
    pub(super) fn new(input: &TimedInput) -> Self {
        Self {
            initial: input.drive,
            latest: input.drive,
            last_arrival: input.arrival.clone(),
            edges: VecDeque::new(),
        }
    }

    pub(super) fn edges(&self) -> &VecDeque<InputEdge> {
        &self.edges
    }

    pub(super) fn preview(
        &self,
        context: EventContext,
        input: &TimedInput,
    ) -> SpiceResult<Option<InputEdge>> {
        let (from, to, window) = if let Some(arrival) = &input.arrival {
            if self
                .last_arrival
                .as_ref()
                .is_some_and(|last| last.same_transition(arrival))
            {
                return Ok(None);
            }
            if self.latest != arrival.from() {
                return Err(SpiceError::circuit(
                    "causal input history disagrees with announced endpoint",
                ));
            }
            (arrival.from(), arrival.to(), arrival.window().clone())
        } else {
            if self.latest == input.drive {
                return Ok(None);
            }
            (self.latest, input.drive, TimingWindow::exact(context.time)?)
        };
        Ok(Some(InputEdge {
            from,
            to,
            window,
            announced: context,
        }))
    }

    pub(super) fn observe(
        &mut self,
        context: EventContext,
        input: &TimedInput,
    ) -> SpiceResult<Option<InputEdge>> {
        let Some(edge) = self.preview(context, input)? else {
            return Ok(None);
        };
        self.latest = edge.to;
        self.last_arrival = input.arrival.clone();
        // These histories are logical, not electrical. A strength-only
        // change still updates the endpoint identity but must not reset
        // the logical pulse-width origin or consume history space.
        if !edge.logic_changed() {
            return Ok(None);
        }
        self.edges.push_back(edge.clone());
        Ok(Some(edge))
    }

    /// The exact feasible logical values at the capture expression.
    /// A visible edge must have arrived and every later-posted edge
    /// must still be undelivered. Their conjunction is solved over
    /// shared origins, not independent pairwise possibility tests.
    fn sample_cases(&self, capture: &InputEdge, budget: &mut usize) -> SpiceResult<Guarded<Drive>> {
        Guarded::first_match(
            self.edges.iter().rev().map(|edge| {
                (
                    edge.window.clone(),
                    if edge.precedes_at_equality(capture) {
                        TimingRelation::NoLater
                    } else {
                        TimingRelation::Before
                    },
                    capture.window.clone(),
                    Drive::strong(edge.to.logic.instantaneous()),
                )
            }),
            Drive::strong(self.initial.logic.instantaneous()),
            budget,
        )
    }

    /// Keep the newest completed logical predecessor and all later
    /// logical events. Rank-only deliveries never enter this history.
    pub(super) fn prune(&mut self, before: Real) {
        let Some(index) = self
            .edges
            .iter()
            .rposition(|edge| edge.window.bounds().maximum() < before)
        else {
            return;
        };
        for edge in self.edges.drain(..index) {
            self.initial = edge.to;
        }
    }
}

/// Sample all ports in the same feasible origin assignment. Combining
/// marginal three-valued samples would invent impossible combinations
/// of data and asynchronous control states at a common capture.
pub(super) fn sample_joint(
    ports: &[PortHistory],
    capture: &InputEdge,
) -> SpiceResult<Guarded<Vec<Drive>>> {
    let mut budget = COMPARISON_BUDGET;
    let mut values = Guarded::unconditional(Vec::with_capacity(ports.len()));
    for port in ports {
        let cases = port.sample_cases(capture, &mut budget)?;
        values = values.zip_with_budget(&cases, &mut budget, |prior, value| {
            let mut next = prior.clone();
            next.push(*value);
            Ok(next)
        })?;
    }
    Ok(values)
}

pub(super) fn qualify_inputs(
    inputs: &[TimedInput],
    inversion: &[bool],
) -> SpiceResult<Vec<TimedInput>> {
    inputs
        .iter()
        .zip(inversion)
        .map(|(input, &invert)| {
            if !invert {
                return Ok(input.clone());
            }
            let invert = |drive: Drive| Drive {
                logic: drive.logic.invert(),
                ..drive
            };
            Ok(TimedInput {
                drive: invert(input.drive),
                arrival: input
                    .arrival
                    .as_ref()
                    .map(|a| Arrival::new(invert(a.from()), invert(a.to()), a.window().clone()))
                    .transpose()?,
            })
        })
        .collect()
}

/// A group of newly announced transitions at the same physical time.
/// For distinct groups, the earlier group's arrival is proven to
/// precede the later group's for every realization.
pub(super) fn ordered_groups<'a>(
    windows: impl IntoIterator<Item = (usize, &'a TimingWindow)>,
) -> SpiceResult<Option<Vec<Vec<usize>>>> {
    let mut budget = COMPARISON_BUDGET;
    let order =
        crate::analysis::digital::timing::proven_order(windows.into_iter().collect(), &mut budget)?;
    Ok(order.map(|case| case.groups().to_vec()))
}

#[cfg(test)]
mod tests {
    use super::super::Strength;
    use super::*;

    #[test]
    fn pruning_does_not_discard_logic_edge_on_strength_change() -> SpiceResult<()> {
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let weak_high = Drive {
            logic: Logic::One,
            strength: Strength::RESISTIVE,
        };
        let input = |drive| TimedInput {
            drive,
            arrival: None,
        };
        let context = |time| EventContext {
            time,
            delta: 0,
            initial: false,
        };
        let mut history = PortHistory::new(&input(low));
        history.observe(context(1.), &input(high))?;
        history.observe(context(2.), &input(weak_high))?;
        history.prune(5.);
        history.observe(context(6.), &input(low))?;
        history.prune(6.);
        let edges: Vec<_> = history
            .edges()
            .iter()
            .filter(|edge| edge.logic_changed())
            .collect();
        assert_eq!(edges.len(), 2);
        assert_eq!(edges[1].relative_to(edges[0])?, Bounds::new(5., 5.)?);
        Ok(())
    }
}
