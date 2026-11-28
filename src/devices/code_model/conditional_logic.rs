//! Conditional logical traces. Evaluate the same native state-transition
//! kernel under each guarded physical ordering. Delay origins are added
//! only after choosing or publishing a trace, so comparing alternatives
//! cannot accidentally assign independent uncertainty to the same event.

use super::Drive;
use super::causal_history::{COMPARISON_BUDGET, InputEdge};
use super::source_logic::{Causes, LogicKernel, LogicMemory};
use crate::analysis::digital::timing::{Conditional, OrderCase, OrderPartition, TimingWindow};
use crate::primitives::{SpiceError, SpiceResult};

#[derive(Clone, Debug)]
pub(super) struct PortEvent {
    pub(super) port: usize,
    pub(super) edge: InputEdge,
}

#[derive(Clone, Debug)]
pub(super) struct LogicalTransition {
    pub(super) cause: TimingWindow,
    pub(super) causes: Causes,
    pub(super) before: Drive,
    pub(super) after: Drive,
}

impl LogicalTransition {
    pub(super) fn equivalent(&self, other: &Self) -> SpiceResult<bool> {
        if self.before != other.before || self.after != other.after || self.causes != other.causes {
            return Ok(false);
        }
        let b = self.cause.relative_to(&other.cause, COMPARISON_BUDGET)?;
        Ok(b.is_exact() && b.minimum() == 0.)
    }
}

#[derive(Clone, Debug)]
pub(super) struct LogicTrace {
    pub(super) memory: LogicMemory,
    pub(super) inputs: Vec<Drive>,
    pub(super) transitions: Vec<LogicalTransition>,
}

impl LogicTrace {
    fn equivalent(&self, other: &Self) -> SpiceResult<bool> {
        if self.memory != other.memory
            || self.inputs != other.inputs
            || self.transitions.len() != other.transitions.len()
        {
            return Ok(false);
        }
        for (lhs, rhs) in self.transitions.iter().zip(&other.transitions) {
            if !lhs.equivalent(rhs)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// For a given physical ordering, execute delta groups atomically.
/// Transport supersession is per port: a more recently posted arrival
/// which has already occurred prevents an obsolete earlier event from
/// resurrecting an old state.
fn evaluate_case(
    kernel: &LogicKernel,
    initial: &LogicMemory,
    inputs: &[Drive],
    events: &[PortEvent],
    order: &OrderCase,
) -> SpiceResult<LogicTrace> {
    let mut trace = LogicTrace {
        memory: initial.clone(),
        inputs: inputs.to_vec(),
        transitions: Vec::new(),
    };
    let mut delivered = vec![None; inputs.len()];
    for group in order.groups() {
        let mut members = group.clone();
        members.sort_by(|&a, &b| {
            let a = events[a].edge.announced;
            let b = events[b].edge.announced;
            a.time
                .total_cmp(&b.time)
                .then_with(|| a.delta.cmp(&b.delta))
        });
        // Distinct announcement delta groups at a coincident physical
        // time retain their established digital ordering.
        let mut remaining = members.as_slice();
        while let Some((&first, _)) = remaining.split_first() {
            let context = events[first].edge.announced;
            let count = remaining.partition_point(|&i| {
                let t = events[i].edge.announced;
                t.time == context.time && t.delta == context.delta
            });
            let (delta, rest) = remaining.split_at(count);
            remaining = rest;
            let mut changed = false;
            for &index in delta {
                let event = &events[index];
                let prior = &mut delivered[event.port];
                if prior.is_some_and(|earlier| earlier > index) {
                    continue;
                }
                *prior = Some(index);
                changed |= trace.inputs[event.port] != event.edge.to;
                trace.inputs[event.port] = event.edge.to;
            }
            if !changed {
                continue;
            }
            let before = trace.memory.requested;
            let (after, causes) = kernel.operation.desired(
                &trace.inputs,
                &mut trace.memory,
                false,
                kernel.strength,
            )?;
            trace.memory.requested = after;
            if before != after {
                trace.transitions.push(LogicalTransition {
                    cause: events[first].edge.window.clone(),
                    causes,
                    before,
                    after,
                });
            }
        }
    }
    Ok(trace)
}

pub(super) fn conditional_trace(
    kernel: &LogicKernel,
    initial: &LogicMemory,
    inputs: &[Drive],
    events: &[PortEvent],
) -> SpiceResult<Conditional<LogicTrace>> {
    let partition = OrderPartition::new(
        events.iter().map(|e| e.edge.window.clone()).collect(),
        COMPARISON_BUDGET,
    )?;
    partition.evaluate(COMPARISON_BUDGET, |order| {
        evaluate_case(kernel, initial, inputs, events, order)
    })
}

/// Largest settled announcement prefix that is guaranteed to precede
/// every remaining event (including same-time delta order). A continuous
/// chain of overlapping absolute envelopes can still be compacted when
/// common origins prove the physical order.
pub(super) fn settled_prefix(
    events: &[PortEvent],
    now: crate::primitives::Real,
) -> SpiceResult<usize> {
    let settled = events
        .iter()
        .take_while(|event| event.edge.window.bounds().maximum() <= now)
        .count();
    if settled == 0 || settled == events.len() {
        return Ok(settled);
    }
    let mut blocked_until = 0;
    let mut prefix = 0;
    for i in 0..settled {
        let earlier = &events[i].edge;
        for (j, event) in events.iter().enumerate().skip(i + 1) {
            let later = &event.edge;
            if earlier.window.bounds().maximum() < later.window.bounds().minimum() {
                continue;
            }
            let bounds = later.relative_to(earlier)?;
            let ordered_delta = (earlier.announced.time, earlier.announced.delta)
                < (later.announced.time, later.announced.delta);
            if bounds.minimum() < 0. || (bounds.minimum() == 0. && !ordered_delta) {
                blocked_until = blocked_until.max(j);
            }
        }
        if blocked_until <= i {
            prefix = i + 1;
        }
    }
    Ok(prefix)
}

/// Reduction is valid only when every guarded branch agrees on both the
/// complete output request sequence and the final native logical state.
/// An infeasible but not yet excluded guard can prevent reduction; it
/// can never make a differing feasible branch disappear.
pub(super) fn common_trace(
    conditional: &Conditional<LogicTrace>,
) -> SpiceResult<Option<LogicTrace>> {
    let mut branches = conditional.alternatives();
    let (_, first) = branches
        .next()
        .ok_or_else(|| SpiceError::circuit("empty conditional logic trace"))?;
    for (_, trace) in branches {
        if !first.equivalent(trace)? {
            return Ok(None);
        }
    }
    Ok(Some(first.clone()))
}

#[cfg(test)]
mod tests {
    use super::super::source_logic::{CausalPaths, DriveStrengths, Operation};
    use super::super::{DelayMode, Delays, EventContext, GateKind, Logic, Strength};
    use super::*;
    use crate::analysis::digital::timing::{DelayWindow, Origin};
    use crate::primitives::Real;
    use std::collections::BTreeMap;

    fn gate(kind: GateKind, inputs: usize) -> LogicKernel {
        LogicKernel {
            operation: Operation::Gate(kind),
            inputs,
            initial: Logic::Zero,
            delays: Delays::ZERO,
            causal: CausalPaths::default(),
            mode: DelayMode::Transport,
            pulse: None,
            strength: DriveStrengths::uniform(Strength::STRONG),
            invert: vec![false; inputs],
            timing: None,
        }
    }

    fn edge(
        port: usize,
        from: Logic,
        to: Logic,
        window: TimingWindow,
        announced: Real,
    ) -> PortEvent {
        PortEvent {
            port,
            edge: InputEdge {
                from: Drive::strong(from),
                to: Drive::strong(to),
                window,
                announced: EventContext {
                    time: announced,
                    delta: 0,
                    initial: false,
                },
            },
        }
    }

    #[test]
    fn guarded_and_pulse_is_optional_and_correlated() -> SpiceResult<()> {
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let u = Origin::shared(0);
        let v = Origin::shared(1);
        let a = TimingWindow::exact(0.)?.after(DelayWindow::new(10., 20.)?, u)?;
        let b = TimingWindow::exact(0.)?.after(DelayWindow::new(10., 20.)?, v)?;
        let trace = conditional_trace(
            &gate(GateKind::And, 2),
            &LogicMemory::new(low),
            &[low, high],
            &[
                edge(0, Logic::Zero, Logic::One, a, 10.),
                edge(1, Logic::One, Logic::Zero, b, 10.),
            ],
        )?;
        assert!(common_trace(&trace)?.is_none());
        for (x, y) in [(0., 1.), (1., 0.), (0.5, 0.5)] {
            let origins = BTreeMap::from([(u, x), (v, y)]);
            let selected = trace.realize(&origins, 100_000)?;
            let sequence: Vec<_> = selected
                .transitions
                .iter()
                .map(|event| event.after)
                .collect();
            assert_eq!(sequence, if x < y { vec![high, low] } else { vec![] });
            let mut matching = 0;
            for (guard, payload) in trace.guarded() {
                if guard.holds(&origins, 100_000)? {
                    matching += 1;
                    assert!(payload.equivalent(selected)?);
                }
            }
            assert_eq!(matching, 1);
        }
        Ok(())
    }

    #[test]
    fn later_posted_event_suppresses_earlier_event_in_reversed_realization() -> SpiceResult<()> {
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let u = Origin::shared(0);
        let v = Origin::shared(1);
        let a = TimingWindow::exact(0.)?.after(DelayWindow::new(10., 20.)?, u)?;
        let b = TimingWindow::exact(0.)?.after(DelayWindow::new(12., 22.)?, v)?;
        let trace = conditional_trace(
            &gate(GateKind::Buffer, 1),
            &LogicMemory::new(low),
            &[low],
            &[
                edge(0, Logic::Zero, Logic::One, a, 10.),
                edge(0, Logic::One, Logic::Zero, b, 12.),
            ],
        )?;
        let ordinary = trace.realize(&BTreeMap::from([(u, 0.), (v, 1.)]), 100_000)?;
        assert_eq!(
            ordinary
                .transitions
                .iter()
                .map(|t| t.after)
                .collect::<Vec<_>>(),
            [high, low]
        );
        let reversed = trace.realize(&BTreeMap::from([(u, 1.), (v, 0.)]), 100_000)?;
        assert!(reversed.transitions.is_empty());
        assert_eq!(reversed.memory.requested, low);
        Ok(())
    }
    #[test]
    fn correlated_settled_prefix_compacts_overlapping_absolute_windows() -> SpiceResult<()> {
        let root =
            TimingWindow::exact(0.)?.after(DelayWindow::new(10., 100.)?, Origin::shared(0))?;
        let mut events = Vec::new();
        for index in 0..10_u32 {
            let time = f64::from(index);
            let window = root.after(DelayWindow::new(time, time)?, Origin::shared(1))?;
            events.push(edge(0, Logic::Zero, Logic::One, window, 10. + time));
        }
        assert_eq!(settled_prefix(&events, 104.)?, 5);
        assert_eq!(settled_prefix(&events, 99.)?, 0);
        // No cut through a simultaneous group even if both endpoints
        // are at the accepted physical time.
        let mut same = events[..2].to_vec();
        same[1].edge.window = same[0].edge.window.clone();
        same[1].edge.announced = same[0].edge.announced;
        assert_eq!(settled_prefix(&same, 100.)?, 2);
        Ok(())
    }
}
