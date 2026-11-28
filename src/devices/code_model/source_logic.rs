//! Qualified receiver logic. Internal timer state is committed only through
//! the event engine; inertial timers wake the model, not an analog load.
use super::{
    CodeModel, DelayMode, Delays, Drive, EventContext, GateKind, Logic, ModelSchema, Output,
};
use crate::primitives::{Real, SpiceError, SpiceResult};

#[derive(Clone, Copy, Debug)]
pub(super) enum SrPolicy {
    ResetDominant,
    SetDominant,
    HoldOnEqual,
    ToggleOnBoth,
}

impl SrPolicy {
    const fn next(self, set: Logic, reset: Logic, previous: Logic) -> Logic {
        match (set, reset) {
            (Logic::One, Logic::Zero) => Logic::One,
            (Logic::Zero, Logic::One) => Logic::Zero,
            (Logic::One, Logic::One) => match self {
                Self::ResetDominant => Logic::Zero,
                Self::SetDominant => Logic::One,
                Self::HoldOnEqual => previous,
                Self::ToggleOnBoth => previous.invert(),
            },
            _ => previous,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum ClockedKind {
    Sr,
    Jk,
    Data,
    Toggle,
}

impl ClockedKind {
    const fn arity(self) -> usize {
        match self {
            Self::Sr | Self::Jk => 3,
            Self::Data | Self::Toggle => 2,
        }
    }

    fn capture(self, inputs: &[Drive], previous: Logic) -> SpiceResult<Logic> {
        match (self, inputs) {
            (Self::Sr, [set, reset, _]) => {
                Ok(SrPolicy::HoldOnEqual.next(set.logic, reset.logic, previous))
            }
            (Self::Jk, [set, reset, _]) => {
                Ok(SrPolicy::ToggleOnBoth.next(set.logic, reset.logic, previous))
            }
            (Self::Data, [data, _]) => Ok(data.logic),
            (Self::Toggle, [toggle, _]) => Ok(if toggle.logic == Logic::One {
                previous.invert()
            } else {
                previous
            }),
            _ => Err(SpiceError::circuit(
                "invalid clocked logic input dimensions",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Operation {
    Gate(GateKind),
    Sr(SrPolicy),
    Clocked { kind: ClockedKind, trigger: Logic },
    Enabled(Logic),
}

impl Operation {
    pub(super) fn from_value(value: Real, inputs: usize) -> SpiceResult<Self> {
        let clocked = |kind, trigger| Self::Clocked { kind, trigger };
        let operation = match value {
            0. => Self::Gate(GateKind::And),
            1. => Self::Gate(GateKind::Or),
            2. => Self::Gate(GateKind::Nand),
            3. => Self::Gate(GateKind::Nor),
            4. => Self::Sr(SrPolicy::ResetDominant),
            5. => Self::Sr(SrPolicy::SetDominant),
            6. => Self::Sr(SrPolicy::HoldOnEqual),
            7. => clocked(ClockedKind::Sr, Logic::One),
            8. => clocked(ClockedKind::Sr, Logic::Zero),
            9. => clocked(ClockedKind::Jk, Logic::One),
            10. => clocked(ClockedKind::Jk, Logic::Zero),
            11. => clocked(ClockedKind::Data, Logic::One),
            12. => clocked(ClockedKind::Data, Logic::Zero),
            13. => clocked(ClockedKind::Toggle, Logic::One),
            14. => clocked(ClockedKind::Toggle, Logic::Zero),
            15. => Self::Enabled(Logic::One),
            16. => Self::Enabled(Logic::Zero),
            _ => return Err(SpiceError::circuit("unsupported native logic operation")),
        };
        let valid = match operation {
            Self::Gate(_) => inputs > 0,
            Self::Sr(_) | Self::Enabled(_) => inputs == 2,
            Self::Clocked { kind, .. } => inputs == kind.arity(),
        };
        if !valid {
            return Err(SpiceError::circuit("invalid native logic input arity"));
        }
        Ok(operation)
    }

    fn desired(
        self,
        inputs: &[Drive],
        state: &mut LogicState,
        initial: bool,
    ) -> SpiceResult<Logic> {
        if inputs.iter().any(|drive| drive.logic == Logic::Unknown) {
            return Err(SpiceError::circuit(
                "qualified receiver supplied unknown logic",
            ));
        }
        match (self, inputs) {
            (Self::Gate(kind), _) => Ok(kind.evaluate(inputs)),
            (Self::Sr(policy), [set, reset]) => {
                Ok(policy.next(set.logic, reset.logic, state.committed))
            }
            (Self::Enabled(active), [data, enable]) => Ok(if enable.logic == active {
                data.logic
            } else {
                state.committed
            }),
            (Self::Clocked { kind, trigger }, [.., clock]) => {
                let previous_clock = std::mem::replace(&mut state.clock, clock.logic);
                // Bias initialization establishes receiver history, not an edge.
                // The most recent captured bit is independent of output delay.
                // A timer wakeup or a nontriggering input must not erase it.
                if !initial && previous_clock != clock.logic && clock.logic == trigger {
                    kind.capture(inputs, state.requested)
                } else {
                    Ok(state.requested)
                }
            }
            _ => Err(SpiceError::circuit("invalid logic input dimensions")),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct LogicKernel {
    pub(super) operation: Operation,
    pub(super) inputs: usize,
    pub(super) initial: Logic,
    pub(super) delays: Delays,
    pub(super) mode: DelayMode,
}

#[derive(Clone, Debug)]
pub(super) struct LogicState {
    committed: Logic,
    requested: Logic,
    deadline: Option<Real>,
    clock: Logic,
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
            clock: Logic::Unknown,
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
            state.requested = self.initial;
            state.deadline = None;
        }
        let desired = self.operation.desired(inputs, state, context.initial)?;
        let transition = |from: Logic, to: Logic| {
            self.delays
                .transition(Drive::strong(from), Drive::strong(to))
        };
        let delay = if context.initial {
            state.committed = desired;
            state.requested = desired;
            0.
        } else if self.mode == DelayMode::Transport {
            // The requested logical state changes immediately; the queue
            // delivers its output at the propagation time.
            let delay = transition(state.requested, desired);
            state.committed = desired;
            state.requested = desired;
            delay
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
                let delay = transition(state.committed, desired);
                let deadline = context.time + delay;
                if delay == 0. {
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
    fn independent_rise_fall_paths_and_explicit_zero() -> SpiceResult<()> {
        let zero = Drive::strong(Logic::Zero);
        let one = Drive::strong(Logic::One);
        assert_eq!(
            Delays::paths([0., 3., 4., 5., 6., 7.])?.transition(zero, one),
            0.
        );
        for mode in [DelayMode::Inertial, DelayMode::Transport] {
            let mut network = Network::default();
            let input = network.add_net();
            let output = network.add_net();
            let source = network.add_source(input)?;
            network.add_model(
                LogicKernel {
                    operation: Operation::Gate(GateKind::Buffer),
                    inputs: 1,
                    initial: Logic::Zero,
                    delays: Delays::paths([2., 3., 4., 5., 6., 7.])?,
                    mode,
                },
                &[input],
                &[output],
            )?;
            let mut engine = Engine::new(network, Limits::default())?;
            let mut trial = engine.trial();
            for (at, bit) in [
                (0., Logic::Zero),
                (1., Logic::One),
                (5., Logic::Zero),
                (10., Logic::One),
                (11., Logic::Zero),
            ] {
                trial.schedule(source, at, Drive::strong(bit))?;
            }
            trial.advance_to(20.)?;
            let actual: Vec<_> = trial
                .changes()
                .iter()
                .filter(|change| change.net == output)
                .map(|change| (change.time, change.drive.logic))
                .collect();
            let mut expected = vec![(0., Logic::Zero), (3., Logic::One), (8., Logic::Zero)];
            if mode == DelayMode::Transport {
                expected.extend([(12., Logic::One), (14., Logic::Zero)]);
            }
            assert_eq!(actual, expected);
        }
        Ok(())
    }

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
                delays: Delays::new(2., 2.)?,
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

    #[test]
    fn clocked_pending_capture_and_toggle_rollback() -> SpiceResult<()> {
        let mut network = Network::default();
        let data = network.add_net();
        let clock = network.add_net();
        let output = network.add_net();
        let d = network.add_source(data)?;
        let c = network.add_source(clock)?;
        network.add_model(
            LogicKernel {
                operation: Operation::from_value(13., 2)?,
                inputs: 2,
                initial: Logic::Zero,
                delays: Delays::new(2., 2.)?,
                mode: DelayMode::Inertial,
            },
            &[data, clock],
            &[output],
        )?;
        let mut engine = Engine::new(network, Limits::default())?;
        let mut init = engine.trial();
        // Initial high clock must not toggle. First real rising edge toggles.
        init.schedule(d, 0., Drive::strong(Logic::One))?;
        init.schedule(c, 0., Drive::strong(Logic::One))?;
        init.schedule(c, 0.5, Drive::strong(Logic::Zero))?;
        init.schedule(c, 1., Drive::strong(Logic::One))?;
        init.advance_to(1.5)?;
        assert_eq!(init.value(output), Some(Drive::strong(Logic::Zero)));
        init.commit()?;
        let before = engine.stats();
        {
            let mut rejected = engine.trial();
            rejected.schedule(c, 2., Drive::strong(Logic::Zero))?;
            // Second trigger toggles the captured one back to zero before
            // the first delayed output becomes visible.
            rejected.schedule(c, 2.5, Drive::strong(Logic::One))?;
            rejected.advance_to(5.)?;
            assert_eq!(rejected.value(output), Some(Drive::strong(Logic::Zero)));
            assert!(rejected.changes().iter().all(|change| change.net != output));
        }
        assert_eq!(engine.stats(), before);
        let mut retry = engine.trial();
        retry.schedule(c, 2., Drive::strong(Logic::Zero))?;
        retry.advance_to(3.)?;
        assert_eq!(retry.value(output), Some(Drive::strong(Logic::One)));
        retry.schedule(c, 4., Drive::strong(Logic::One))?;
        retry.advance_to(6.)?;
        assert_eq!(retry.value(output), Some(Drive::strong(Logic::Zero)));
        assert_eq!(
            retry
                .changes()
                .iter()
                .filter(|change| change.net == output)
                .count(),
            2
        );
        retry.commit()?;
        Ok(())
    }
}
