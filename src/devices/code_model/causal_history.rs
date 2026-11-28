//! Input waveform history for causal storage evaluation. Announcing the
//! beginning of a bounded edge records the edge exactly once; its
//! completion only settles the envelope and is not another edge.

use super::{Drive, EventContext, Logic};
use crate::analysis::digital::timing::{Arrival, Bounds, TimedInput, TimingWindow};
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

    pub(super) fn observe(
        &mut self,
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
            // An unqualified R/F marker is an observed uncertainty
            // phase, not a guaranteed binary transition.
            (self.latest, input.drive, TimingWindow::exact(context.time)?)
        };
        let edge = InputEdge {
            from,
            to,
            window,
            announced: context,
        };
        self.latest = to;
        self.last_arrival = input.arrival.clone();
        self.edges.push_back(edge.clone());
        Ok(Some(edge))
    }

    /// The possible binary value at the capture expression, factoring
    /// common causal origins before deciding which input edge has
    /// already occurred. Ordered transport supersedes earlier arrivals.
    pub(super) fn sample(&self, capture: &InputEdge) -> SpiceResult<Drive> {
        let mut value = None;
        let mut later = Vec::<&InputEdge>::new();
        let mut include = |drive: Drive| {
            let logic = drive.logic.instantaneous();
            value = Some(value.map_or(logic, |prior| {
                if prior == logic {
                    logic
                } else {
                    Logic::Unknown
                }
            }));
        };
        for edge in self.edges.iter().rev() {
            let relation = edge.occurs_before(capture)?;
            if relation == Relation::After {
                continue;
            }
            // If a later posted edge must arrive no later than this
            // edge, the older value cannot be visible at this capture.
            let mut superseded = false;
            for candidate in &later {
                if candidate.occurs_before(edge)? == Relation::Before {
                    superseded = true;
                    break;
                }
            }
            if !superseded {
                include(edge.to);
            }
            if relation == Relation::Before {
                return Ok(Drive::strong(value.unwrap_or(Logic::Unknown)));
            }
            later.push(edge);
        }
        include(self.initial);
        Ok(Drive::strong(value.unwrap_or(Logic::Unknown)))
    }

    /// Retain the newest completed predecessor and every potentially
    /// relevant later edge. A newer definitely completed transport
    /// event makes older pending events irrelevant.
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
