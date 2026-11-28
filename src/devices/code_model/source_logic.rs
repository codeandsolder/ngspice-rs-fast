//! Qualified receiver logic. Internal timer state is committed only through
//! the event engine; inertial timers wake the model, not an analog load.
use super::ModelDiagnostic;
use super::sequential_timing::{TimingContract, TimingState};
use super::{
    CodeModel, DelayMode, Delays, Drive, EventContext, GateKind, Logic, ModelSchema, Output,
    Strength,
};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Conservative three-valued consensus: retain a known result if every
/// physically possible binary interpretation gives that result.
fn possible(value: Logic) -> &'static [Logic] {
    match value {
        Logic::Zero => &[Logic::Zero],
        Logic::One => &[Logic::One],
        Logic::Unknown => &[Logic::Zero, Logic::One],
    }
}

fn consensus(values: impl IntoIterator<Item = Logic>) -> Logic {
    let mut values = values.into_iter();
    let first = values.next().unwrap_or(Logic::Unknown);
    if values.all(|value| value == first) {
        first
    } else {
        Logic::Unknown
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum SrPolicy {
    ResetDominant,
    SetDominant,
    HoldOnEqual,
    ToggleOnBoth,
}

impl SrPolicy {
    fn next(self, set: Logic, reset: Logic, previous: Logic) -> Logic {
        consensus(possible(set).iter().flat_map(|&set| {
            possible(reset)
                .iter()
                .map(move |&reset| self.binary(set, reset, previous))
        }))
    }

    const fn binary(self, set: Logic, reset: Logic, previous: Logic) -> Logic {
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
    pub(super) const fn arity(self) -> usize {
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
            (Self::Toggle, [toggle, _]) => {
                Ok(consensus(possible(toggle.logic).iter().map(|&bit| {
                    if bit == Logic::One {
                        previous.invert()
                    } else {
                        previous
                    }
                })))
            }
            _ => Err(SpiceError::circuit(
                "invalid clocked logic input dimensions",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Edge {
    Rising,
    Falling,
    Both,
}

impl Edge {
    fn capture(self, before: Logic, after: Logic, previous: Logic, captured: Logic) -> Logic {
        // No observed change is not an edge, including a timer wakeup
        // while a receiver remains unknown.
        if before == after {
            return previous;
        }
        consensus(possible(before).iter().flat_map(|&before| {
            possible(after).iter().map(move |&after| {
                let trigger = before != after
                    && match self {
                        Self::Rising => after == Logic::One,
                        Self::Falling => after == Logic::Zero,
                        Self::Both => true,
                    };
                if trigger { captured } else { previous }
            })
        }))
    }
}

/// Explicit active levels and simultaneous preset/clear result for a
/// single-output clocked primitive. Different source dialects can choose
/// different conflict policies without changing the engine.
#[derive(Clone, Copy, Debug)]
pub(super) struct AsyncPolicy {
    pub(super) preset: Logic,
    pub(super) clear: Logic,
    pub(super) both: Logic,
}

impl AsyncPolicy {
    fn resolve(self, preset: Logic, clear: Logic, normal: Logic) -> Logic {
        consensus(possible(preset).iter().flat_map(|&preset| {
            possible(clear).iter().map(move |&clear| {
                match (preset == self.preset, clear == self.clear) {
                    (true, true) => self.both,
                    (true, false) => Logic::One,
                    (false, true) => Logic::Zero,
                    (false, false) => normal,
                }
            })
        }))
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Operation {
    Gate(GateKind),
    Sr(SrPolicy),
    Clocked {
        kind: ClockedKind,
        trigger: Edge,
        asynchronous: Option<AsyncPolicy>,
    },
    Enabled(Logic),
    TriState(Logic),
}

impl Operation {
    pub(super) fn from_value(
        value: Real,
        inputs: usize,
        asynchronous: Option<AsyncPolicy>,
    ) -> SpiceResult<Self> {
        let clocked = |kind, trigger| Self::Clocked {
            kind,
            trigger,
            asynchronous,
        };
        let operation = match value {
            0. => Self::Gate(GateKind::And),
            1. => Self::Gate(GateKind::Or),
            2. => Self::Gate(GateKind::Nand),
            3. => Self::Gate(GateKind::Nor),
            4. => Self::Sr(SrPolicy::ResetDominant),
            5. => Self::Sr(SrPolicy::SetDominant),
            6. => Self::Sr(SrPolicy::HoldOnEqual),
            7. => clocked(ClockedKind::Sr, Edge::Rising),
            8. => clocked(ClockedKind::Sr, Edge::Falling),
            9. => clocked(ClockedKind::Jk, Edge::Rising),
            10. => clocked(ClockedKind::Jk, Edge::Falling),
            11. => clocked(ClockedKind::Data, Edge::Rising),
            12. => clocked(ClockedKind::Data, Edge::Falling),
            13. => clocked(ClockedKind::Toggle, Edge::Rising),
            14. => clocked(ClockedKind::Toggle, Edge::Falling),
            15. => Self::Enabled(Logic::One),
            16. => Self::Enabled(Logic::Zero),
            17. => clocked(ClockedKind::Sr, Edge::Both),
            18. => clocked(ClockedKind::Jk, Edge::Both),
            19. => clocked(ClockedKind::Data, Edge::Both),
            20. => clocked(ClockedKind::Toggle, Edge::Both),
            21. => Self::TriState(Logic::One),
            22. => Self::TriState(Logic::Zero),
            23. => Self::Gate(GateKind::Xor),
            24. => Self::Gate(GateKind::Xnor),
            25. => Self::Gate(GateKind::Buffer),
            26. => Self::Gate(GateKind::Inverter),
            _ => return Err(SpiceError::circuit("unsupported native logic operation")),
        };
        let valid = match operation {
            Self::Gate(GateKind::Buffer | GateKind::Inverter) => inputs == 1,
            Self::Gate(_) => inputs > 0,
            Self::Sr(_) | Self::Enabled(_) | Self::TriState(_) => inputs == 2,
            Self::Clocked { kind, .. } => {
                inputs == kind.arity() + if asynchronous.is_some() { 2 } else { 0 }
            }
        };
        if !valid || (asynchronous.is_some() && !matches!(operation, Self::Clocked { .. })) {
            return Err(SpiceError::circuit(
                "invalid native logic input arity or asynchronous policy",
            ));
        }
        Ok(operation)
    }

    fn desired(
        self,
        inputs: &[Drive],
        state: &mut LogicState,
        initial: bool,
        strength: Strength,
    ) -> SpiceResult<Drive> {
        let drive = |logic| Drive { logic, strength };
        let prior = state.requested.logic;
        let logic = match (self, inputs) {
            (Self::Gate(kind), _) => kind.evaluate(inputs),
            (Self::Sr(policy), [set, reset]) => policy.next(set.logic, reset.logic, prior),
            (Self::Enabled(active), [data, enable]) => consensus(
                possible(enable.logic)
                    .iter()
                    .map(|&bit| if bit == active { data.logic } else { prior }),
            ),
            (Self::TriState(active), [data, enable]) => {
                if enable.logic == Logic::Unknown {
                    // A conditional connection cannot be expressed as a
                    // single exact Drive; retain uncertainty, not a
                    // falsely disconnected output.
                    Logic::Unknown
                } else if enable.logic == active {
                    data.logic
                } else {
                    return Ok(Drive::default());
                }
            }
            (
                Self::Clocked {
                    kind,
                    trigger,
                    asynchronous,
                },
                _,
            ) => {
                let (synchronous, controls) = inputs.split_at(kind.arity());
                let clock = synchronous
                    .last()
                    .ok_or_else(|| SpiceError::circuit("missing clock input"))?
                    .logic;
                let previous_clock = std::mem::replace(&mut state.clock, clock);
                let normal = if initial {
                    prior
                } else {
                    trigger.capture(
                        previous_clock,
                        clock,
                        prior,
                        kind.capture(synchronous, prior)?,
                    )
                };
                match (asynchronous, controls) {
                    (Some(policy), [preset, clear]) => {
                        policy.resolve(preset.logic, clear.logic, normal)
                    }
                    (None, []) => normal,
                    _ => return Err(SpiceError::circuit("invalid asynchronous logic inputs")),
                }
            }
            _ => return Err(SpiceError::circuit("invalid logic input dimensions")),
        };
        Ok(drive(logic))
    }
}

#[derive(Clone, Debug)]
pub(super) struct LogicKernel {
    pub(super) operation: Operation,
    pub(super) inputs: usize,
    pub(super) initial: Logic,
    pub(super) delays: Delays,
    pub(super) mode: DelayMode,
    pub(super) strength: Strength,
    /// Input inversion at digital terminals, after net resolution.
    pub(super) invert: Vec<bool>,
    pub(super) timing: Option<TimingContract>,
}

#[derive(Clone, Debug)]
pub(super) struct LogicState {
    committed: Drive,
    requested: Drive,
    deadline: Option<Real>,
    clock: Logic,
    timing: Option<TimingState>,
    diagnostics: Vec<ModelDiagnostic>,
}

impl CodeModel for LogicKernel {
    type State = LogicState;

    fn schema(&self) -> ModelSchema {
        ModelSchema::digital(self.inputs, 1)
    }

    fn initial_state(&self) -> Self::State {
        LogicState {
            committed: Drive {
                logic: self.initial,
                strength: self.strength,
            },
            requested: Drive {
                logic: self.initial,
                strength: self.strength,
            },
            deadline: None,
            clock: Logic::Unknown,
            timing: self.timing.as_ref().map(TimingContract::initial_state),
            diagnostics: Vec::new(),
        }
    }

    fn take_diagnostics(&self, state: &mut Self::State) -> Vec<ModelDiagnostic> {
        std::mem::take(&mut state.diagnostics)
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
            state.committed = Drive {
                logic: self.initial,
                strength: self.strength,
            };
            state.requested = state.committed;
            state.deadline = None;
        }
        let transformed;
        let inputs = if self.invert.iter().any(|&value| value) {
            transformed = inputs
                .iter()
                .zip(&self.invert)
                .map(|(&drive, &invert)| {
                    if invert {
                        Drive {
                            logic: drive.logic.invert(),
                            ..drive
                        }
                    } else {
                        drive
                    }
                })
                .collect::<Vec<_>>();
            transformed.as_slice()
        } else {
            inputs
        };
        let unknown = match (&self.timing, &mut state.timing) {
            (Some(contract), Some(timing)) => {
                contract.observe(context, inputs, timing, &mut state.diagnostics)?
            }
            (None, None) => false,
            _ => return Err(SpiceError::circuit("sequential timing state mismatch")),
        };
        let mut desired = self
            .operation
            .desired(inputs, state, context.initial, self.strength)?;
        if unknown {
            desired.logic = Logic::Unknown;
        }
        let transition = |from, to| self.delays.transition(from, to);
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
            drive: state.committed,
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
    fn resolved_tristate_matches_independent_ngspice_bus_events() -> SpiceResult<()> {
        let mut network = Network::default();
        let data = network.add_net();
        let enable = network.add_net();
        let bus = network.add_net();
        let d = network.add_source(data)?;
        let e = network.add_source(enable)?;
        let pull = network.add_source(bus)?;
        network.add_model(
            LogicKernel {
                operation: Operation::TriState(Logic::One),
                inputs: 2,
                initial: Logic::Zero,
                delays: Delays::new(1e-6, 1e-6)?,
                mode: DelayMode::Transport,
                strength: Strength::Strong,
                invert: Vec::new(),
                timing: None,
            },
            &[data, enable],
            &[bus],
        )?;
        let mut engine = Engine::new(network, Limits::default())?;
        let mut trial = engine.trial();
        let weak = Drive {
            logic: Logic::Zero,
            strength: Strength::Resistive,
        };
        trial.schedule(pull, 0., weak)?;
        // Reference values use exact source and output timestamps from the
        // independent C event trace, including strength-only transitions.
        let mut previous = None;
        let mut previous_inputs = None;
        for row in include_str!("../../../tests/fixtures/native-logic/tristate.csv")
            .lines()
            .skip(1)
        {
            let fields: Vec<_> = row.split(',').collect();
            let time = fields[0].parse::<Real>().unwrap();
            let bit = |field: &str| {
                if field.starts_with('1') {
                    Logic::One
                } else {
                    Logic::Zero
                }
            };
            let inputs = (bit(fields[1]), bit(fields[2]));
            if previous_inputs != Some(inputs) {
                trial.schedule(d, time, Drive::strong(inputs.0))?;
                trial.schedule(e, time, Drive::strong(inputs.1))?;
                previous_inputs = Some(inputs);
            }
            let drive = Drive {
                logic: bit(fields[3]),
                strength: if fields[3].ends_with('s') {
                    Strength::Strong
                } else {
                    Strength::Resistive
                },
            };
            // Decimal C trace roundtrips to the same binary64 timestamp.
            trial.advance_to(time)?;
            assert_eq!(trial.value(bus), Some(drive), "{row}");
            previous = Some(drive);
        }
        assert_eq!(previous, Some(weak));
        trial.commit()
    }

    #[test]
    fn dual_edge_and_async_unknown_consensus() -> SpiceResult<()> {
        let mut network = Network::default();
        let inputs: Vec<_> = (0..4).map(|_| network.add_net()).collect();
        let out = network.add_net();
        let sources = inputs
            .iter()
            .map(|&net| network.add_source(net))
            .collect::<SpiceResult<Vec<_>>>()?;
        network.add_model(
            LogicKernel {
                operation: Operation::from_value(
                    19.,
                    4,
                    Some(AsyncPolicy {
                        preset: Logic::One,
                        clear: Logic::One,
                        both: Logic::Unknown,
                    }),
                )?,
                inputs: 4,
                initial: Logic::Zero,
                delays: Delays::ZERO,
                mode: DelayMode::Inertial,
                strength: Strength::Strong,
                invert: Vec::new(),
                timing: None,
            },
            &inputs,
            &[out],
        )?;
        let mut engine = Engine::new(network, Limits::default())?;
        let mut trial = engine.trial();
        use Logic::{One, Unknown, Zero};
        // data, clock, preset, clear. Known dominating controls and
        // ambiguous edges are resolved by possible-outcome consensus.
        let witness = [
            (0., [One, One, Zero, Zero], Zero),
            (1., [One, Zero, Zero, Zero], One),
            (2., [Zero, One, Zero, Zero], Zero),
            (3., [Zero, One, One, Zero], One),
            (4., [Zero, Zero, One, One], Unknown),
            (5., [Zero, Zero, Zero, One], Zero),
            (6., [Zero, Zero, Zero, Zero], Zero),
            (7., [Zero, Unknown, Zero, Zero], Zero),
            (8., [One, One, Zero, Zero], Unknown),
            (9., [One, Zero, Zero, Zero], One),
            (10., [Zero, Zero, Unknown, One], Unknown),
        ];
        for (time, bits, expected) in witness {
            for (&source, bit) in sources.iter().zip(bits) {
                trial.schedule(source, time, Drive::strong(bit))?;
            }
            trial.advance_to(time)?;
            assert_eq!(trial.value(out), Some(Drive::strong(expected)), "at {time}");
        }
        trial.commit()
    }

    #[test]
    fn timing_warning_is_committed_once_without_poisoning_output() -> SpiceResult<()> {
        use super::super::sequential_timing::TimingSpec;
        let mut network = Network::default();
        let data = network.add_net();
        let clock = network.add_net();
        let output = network.add_net();
        let d = network.add_source(data)?;
        let c = network.add_source(clock)?;
        let operation = Operation::from_value(11., 2, None)?;
        let mut spec = TimingSpec::default();
        spec.set_vector("timing_setup", vec![2.])?;
        spec.set_policy(2.)?;
        network.add_named_model(
            "xstage.a1",
            LogicKernel {
                operation,
                inputs: 2,
                initial: Logic::Zero,
                delays: Delays::ZERO,
                mode: DelayMode::Inertial,
                strength: Strength::Strong,
                invert: Vec::new(),
                timing: spec.bind(operation, 2)?,
            },
            &[data, clock],
            &[output],
        )?;
        let mut engine = Engine::new(network, Limits::default())?;
        let mut initial = engine.trial();
        initial.schedule(d, 0., Drive::strong(Logic::Zero))?;
        initial.schedule(c, 0., Drive::strong(Logic::Zero))?;
        initial.schedule(d, 1., Drive::strong(Logic::One))?;
        initial.advance_to(1.)?;
        initial.commit()?;
        let mut rejected = engine.trial();
        rejected.schedule(c, 2., Drive::strong(Logic::One))?;
        rejected.advance_to(2.)?;
        assert_eq!(rejected.diagnostics().len(), 1);
        assert_eq!(rejected.value(output), Some(Drive::strong(Logic::One)));
        drop(rejected);
        assert!(engine.diagnostics().is_empty());
        assert_eq!(engine.value(output), Some(Drive::strong(Logic::Zero)));
        let mut accepted = engine.trial();
        accepted.schedule(c, 2., Drive::strong(Logic::One))?;
        accepted.advance_to(4.)?;
        accepted.commit()?;
        let records = engine.diagnostics();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].model.as_ref(), "xstage.a1");
        assert_eq!(records[0].detail.kind, "setup");
        assert_eq!(engine.stats().diagnostics, 1);
        assert_eq!(engine.value(output), Some(Drive::strong(Logic::One)));
        Ok(())
    }

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
                    strength: Strength::Strong,
                    invert: Vec::new(),
                    timing: None,
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
                strength: Strength::Strong,
                invert: Vec::new(),
                timing: None,
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
                operation: Operation::from_value(13., 2, None)?,
                inputs: 2,
                initial: Logic::Zero,
                delays: Delays::new(2., 2.)?,
                mode: DelayMode::Inertial,
                strength: Strength::Strong,
                invert: Vec::new(),
                timing: None,
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
