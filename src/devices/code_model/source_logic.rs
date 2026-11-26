//! Qualified receiver logic. Internal timer state is committed only through
//! the event engine; inertial timers wake the model, not an analog load.
use super::{
    CodeModel, DelayMode, Delays, Drive, EventContext, GateKind, Logic, ModelSchema, Output,
};
use crate::primitives::{Real, SpiceError, SpiceResult};

#[derive(Clone, Copy, Debug)]
pub(super) enum Operation {
    Gate(GateKind),
    ResetDominant,
    SetDominant,
    HoldOnEqual,
}

impl Operation {
    pub(super) fn from_value(value: Real, inputs: usize) -> SpiceResult<Self> {
        if inputs == 0 {
            return Err(SpiceError::circuit("empty logic input vector"));
        }
        let operation = match value {
            0. => Self::Gate(GateKind::And),
            1. => Self::Gate(GateKind::Or),
            2. => Self::Gate(GateKind::Nand),
            3. => Self::Gate(GateKind::Nor),
            4. => Self::ResetDominant,
            5. => Self::SetDominant,
            6. => Self::HoldOnEqual,
            _ => return Err(SpiceError::circuit("unsupported native logic operation")),
        };
        if !matches!(operation, Self::Gate(_)) && inputs != 2 {
            return Err(SpiceError::circuit("latch requires set and reset inputs"));
        }
        Ok(operation)
    }

    fn desired(self, inputs: &[Drive], committed: Logic) -> SpiceResult<Logic> {
        if inputs.iter().any(|drive| drive.logic == Logic::Unknown) {
            return Err(SpiceError::circuit(
                "qualified receiver supplied unknown logic",
            ));
        }
        if let Self::Gate(kind) = self {
            return Ok(kind.evaluate(inputs));
        }
        let [set, reset] = inputs else {
            return Err(SpiceError::circuit("invalid latch input dimensions"));
        };
        Ok(match (set.logic, reset.logic) {
            (Logic::One, Logic::Zero) => Logic::One,
            (Logic::Zero, Logic::One) => Logic::Zero,
            (Logic::One, Logic::One) => match self {
                Self::ResetDominant => Logic::Zero,
                Self::SetDominant => Logic::One,
                _ => committed,
            },
            _ => committed,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct LogicKernel {
    pub(super) operation: Operation,
    pub(super) inputs: usize,
    pub(super) initial: Logic,
    pub(super) delay: Real,
    pub(super) mode: DelayMode,
}

#[derive(Clone, Debug)]
pub(super) struct LogicState {
    committed: Logic,
    requested: Logic,
    deadline: Option<Real>,
}

impl CodeModel for LogicKernel {
    type State = LogicState;

    fn schema(&self) -> ModelSchema {
        ModelSchema::digital(self.inputs, 1)
    }

    fn initial_state(&self) -> Self::State {
        LogicState {
            committed: self.initial,
            requested: self.initial,
            deadline: None,
        }
    }

    fn next_wakeup(&self, state: &Self::State) -> Option<Real> {
        state.deadline
    }

    fn evaluate(
        &self,
        context: EventContext,
        inputs: &[Drive],
        state: &mut Self::State,
    ) -> SpiceResult<Vec<Output>> {
        if inputs.len() != self.inputs {
            return Err(SpiceError::circuit(
                "qualified logic input dimension mismatch",
            ));
        }
        if context.initial {
            state.committed = self.initial;
            state.deadline = None;
        }
        let desired = self.operation.desired(inputs, state.committed)?;
        let delay = if context.initial {
            state.committed = desired;
            state.requested = desired;
            0.
        } else if self.mode == DelayMode::Transport {
            // The requested logical state changes immediately; the queue
            // delivers its output at the propagation time.
            state.committed = desired;
            state.requested = desired;
            self.delay
        } else {
            if state
                .deadline
                .is_some_and(|deadline| context.time >= deadline)
                && desired == state.requested
            {
                state.committed = state.requested;
                state.deadline = None;
            }
            if desired == state.committed {
                state.requested = desired;
                state.deadline = None;
            } else if state.deadline.is_none() || desired != state.requested {
                state.requested = desired;
                let deadline = context.time + self.delay;
                if self.delay == 0. {
                    state.committed = desired;
                    state.deadline = None;
                } else if !deadline.is_finite() || deadline <= context.time {
                    return Err(SpiceError::circuit(
                        "logic deadline outside representable time range",
                    ));
                } else {
                    state.deadline = Some(deadline);
                }
            }
            0.
        };
        Ok(vec![Output {
            drive: Drive::strong(state.committed),
            delays: Delays::new(delay, delay)?,
            mode: DelayMode::Transport,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::digital::{Engine, Limits, Network};

    #[test]
    fn deadline_cancellation_and_rejected_trial_preserve_state() -> SpiceResult<()> {
        let mut network = Network::default();
        let input = network.add_net();
        let output = network.add_net();
        let source = network.add_source(input)?;
        network.add_model(
            LogicKernel {
                operation: Operation::Gate(GateKind::And),
                inputs: 1,
                initial: Logic::Zero,
                delay: 2.,
                mode: DelayMode::Inertial,
            },
            &[input],
            &[output],
        )?;
        let mut engine = Engine::new(network, Limits::default())?;
        let mut initial = engine.trial();
        initial.schedule(source, 0., Drive::strong(Logic::Zero))?;
        initial.schedule(source, 1., Drive::strong(Logic::One))?;
        initial.advance_to(2.)?;
        initial.commit()?;
        let before = engine.stats();
        {
            let mut rejected = engine.trial();
            rejected.advance_to(3.)?;
            assert_eq!(rejected.value(output), Some(Drive::strong(Logic::One)));
        }
        assert_eq!(engine.stats(), before);
        let mut equal_width = engine.trial();
        equal_width.schedule(source, 3., Drive::strong(Logic::Zero))?;
        equal_width.advance_to(5.)?;
        assert_eq!(equal_width.value(output), Some(Drive::strong(Logic::Zero)));
        assert!(
            equal_width
                .changes()
                .iter()
                .all(|change| change.net != output)
        );
        equal_width.commit()?;
        Ok(())
    }
}
