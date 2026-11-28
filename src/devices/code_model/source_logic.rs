//! Qualified receiver logic. Internal timer state is committed only through
//! the event engine; inertial timers wake the model, not an analog load.
use super::ModelDiagnostic;
use super::pulse::{PulseFilter, PulseState};
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
        Logic::Unknown | Logic::Rising | Logic::Falling => &[Logic::Zero, Logic::One],
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

    pub(super) fn capture(self, inputs: &[Drive], previous: Logic) -> SpiceResult<Logic> {
        match (self, inputs) {
            (Self::Sr, [set, reset, _]) => {
                Ok(SrPolicy::HoldOnEqual.next(set.logic, reset.logic, previous))
            }
            (Self::Jk, [set, reset, _]) => {
                Ok(SrPolicy::ToggleOnBoth.next(set.logic, reset.logic, previous))
            }
            (Self::Data, [data, _]) => Ok(data.logic.instantaneous()),
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
    pub(super) fn capture(
        self,
        before: Logic,
        after: Logic,
        previous: Logic,
        captured: Logic,
    ) -> Logic {
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
/// A clock phase denotes a single possible excursion from its known
/// starting level. Retain the no-edge, leading-edge and completed-pulse
/// alternatives until the ending level is known. In particular, two
/// correlated toggles of a dual-edge T flip-flop can restore a known bit.
#[derive(Clone, Copy, Debug)]
pub(super) struct ClockAmbiguity {
    phase: Logic,
    before: Logic,
    leading: Logic,
    both: Logic,
}

impl Edge {
    const fn on(self, rising: bool) -> bool {
        matches!(self, Self::Both)
            || (rising && matches!(self, Self::Rising))
            || (!rising && matches!(self, Self::Falling))
    }

    pub(super) fn timed_capture(
        self,
        previous: Logic,
        current: Logic,
        prior: Logic,
        uncertainty: &mut Option<ClockAmbiguity>,
        capture: impl Fn(Logic) -> SpiceResult<Logic>,
    ) -> SpiceResult<Logic> {
        if let Some(mut phase) = *uncertainty {
            if current == phase.phase || current.is_binary() {
                let rising = phase.phase == Logic::Rising;
                let leading = if self.on(rising) {
                    capture(phase.before)?
                } else {
                    phase.before
                };
                phase.leading = consensus([phase.leading, leading]);
                let both = if self.on(!rising) {
                    capture(phase.leading)?
                } else {
                    phase.leading
                };
                phase.both = consensus([phase.both, both]);
                if current == phase.phase {
                    *uncertainty = Some(phase);
                    return Ok(consensus([phase.before, phase.leading, phase.both]));
                }
                *uncertainty = None;
                return Ok(if current == phase.phase.phase_end() {
                    phase.leading
                } else {
                    consensus([phase.before, phase.both])
                });
            }
            // An unqualified change of ambiguity phase is not proof of
            // a particular edge sequence.
            *uncertainty = None;
            return Ok(Logic::Unknown);
        }
        if matches!(current, Logic::Rising | Logic::Falling) {
            let rising = current == Logic::Rising;
            let leading = if self.on(rising) {
                capture(prior)?
            } else {
                prior
            };
            let both = if self.on(!rising) {
                capture(leading)?
            } else {
                leading
            };
            *uncertainty = Some(ClockAmbiguity {
                phase: current,
                before: prior,
                leading,
                both,
            });
            return Ok(consensus([prior, leading, both]));
        }
        Ok(self.capture(previous, current, prior, capture(prior)?))
    }
}

impl ClockAmbiguity {
    fn force(&mut self, logic: Logic) {
        self.before = logic;
        self.leading = logic;
        self.both = logic;
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct AsyncPolicy {
    pub(super) preset: Logic,
    pub(super) clear: Logic,
    pub(super) both: Logic,
}

impl AsyncPolicy {
    pub(super) fn resolve(self, preset: Logic, clear: Logic, normal: Logic) -> Logic {
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
    Enabled {
        active: Logic,
        asynchronous: Option<AsyncPolicy>,
    },
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
            15. => Self::Enabled {
                active: Logic::One,
                asynchronous,
            },
            16. => Self::Enabled {
                active: Logic::Zero,
                asynchronous,
            },
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
            Self::Sr(_) | Self::TriState(_) => inputs == 2,
            Self::Enabled { .. } => inputs == 2 + if asynchronous.is_some() { 2 } else { 0 },
            Self::Clocked { kind, .. } => {
                inputs == kind.arity() + if asynchronous.is_some() { 2 } else { 0 }
            }
        };
        if !valid
            || (asynchronous.is_some()
                && !matches!(operation, Self::Clocked { .. } | Self::Enabled { .. }))
        {
            return Err(SpiceError::circuit(
                "invalid native logic input arity or asynchronous policy",
            ));
        }
        Ok(operation)
    }

    pub(super) const fn has_gate_path(self) -> bool {
        matches!(self, Self::Enabled { .. })
    }

    pub(super) const fn has_async_paths(self) -> bool {
        matches!(
            self,
            Self::Clocked {
                asynchronous: Some(_),
                ..
            } | Self::Enabled {
                asynchronous: Some(_),
                ..
            }
        )
    }

    pub(super) fn desired(
        self,
        inputs: &[Drive],
        state: &mut LogicMemory,
        initial: bool,
        strength: DriveStrengths,
    ) -> SpiceResult<(Drive, Causes)> {
        let prior = state.requested.logic;
        let mut causes = Causes::PRIMARY;
        let logic = match (self, inputs) {
            (Self::Gate(kind), _) => kind.evaluate(inputs),
            (Self::Sr(policy), [set, reset]) => policy.next(set.logic, reset.logic, prior),
            (
                Self::Enabled {
                    active,
                    asynchronous,
                },
                [data, enable, controls @ ..],
            ) => {
                let previous_gate = std::mem::replace(&mut state.clock, enable.logic);
                let previous_data = std::mem::replace(&mut state.data, data.logic);
                let transparent = |data: Logic, gate: Logic| {
                    consensus(possible(gate).iter().map(|&bit| {
                        if bit == active {
                            data
                        } else {
                            prior.instantaneous()
                        }
                    }))
                };
                let normal = transparent(data.logic, enable.logic);
                if previous_gate != enable.logic {
                    // Evaluate each change in isolation. If either is
                    // sufficient, retain its cause; if both are required,
                    // retain both. This also handles ambiguous enable edges.
                    let gate_alone = transparent(previous_data, enable.logic);
                    let data_alone = transparent(data.logic, previous_gate);
                    causes.gate = gate_alone != prior;
                    causes.primary = data_alone != prior;
                    if normal != prior && !causes.gate && !causes.primary {
                        causes.gate = true;
                        causes.primary = previous_data != data.logic;
                    }
                    if !causes.gate && !causes.primary {
                        causes.gate = true;
                    }
                }
                apply_async(asynchronous, controls, normal, prior, state, &mut causes)?
            }
            (Self::TriState(active), [data, enable]) => {
                if !enable.logic.is_binary() {
                    // A conditional connection cannot be expressed as a
                    // single exact Drive; retain uncertainty.
                    Logic::Unknown
                } else if enable.logic == active {
                    data.logic
                } else {
                    return Ok((Drive::default(), causes));
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
                    state.clock_ambiguity = None;
                    prior
                } else {
                    trigger.timed_capture(
                        previous_clock,
                        clock,
                        prior,
                        &mut state.clock_ambiguity,
                        |prior| kind.capture(synchronous, prior),
                    )?
                };
                let result =
                    apply_async(asynchronous, controls, normal, prior, state, &mut causes)?;
                if let (Some(policy), [preset, clear], Some(phase)) =
                    (asynchronous, controls, &mut state.clock_ambiguity)
                    && (preset.logic == policy.preset
                        || clear.logic == policy.clear
                        || !preset.logic.is_binary()
                        || !clear.logic.is_binary())
                {
                    phase.force(result);
                }
                result
            }
            _ => return Err(SpiceError::circuit("invalid logic input dimensions")),
        };
        Ok((strength.drive(logic), causes))
    }
}

/// A simultaneous group can have more than one causally relevant path.
/// For the deterministic contract, the latest selected path wins.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Causes {
    pub(super) primary: bool,
    pub(super) gate: bool,
    pub(super) preset: bool,
    pub(super) clear: bool,
}

impl Causes {
    const PRIMARY: Self = Self {
        primary: true,
        gate: false,
        preset: false,
        clear: false,
    };
}

fn apply_async(
    policy: Option<AsyncPolicy>,
    controls: &[Drive],
    normal: Logic,
    prior: Logic,
    state: &mut LogicMemory,
    causes: &mut Causes,
) -> SpiceResult<Logic> {
    match (policy, controls) {
        (None, []) => Ok(normal),
        (Some(policy), [preset, clear]) => {
            let new = [preset.logic, clear.logic];
            let old = std::mem::replace(&mut state.asynchronous, new);
            let result = policy.resolve(new[0], new[1], normal);
            if result != prior {
                let no_async_change = policy.resolve(old[0], old[1], normal);
                let no_normal_change = policy.resolve(new[0], new[1], prior);
                let preset_alone = policy.resolve(new[0], old[1], normal);
                let clear_alone = policy.resolve(old[0], new[1], normal);
                let changed = [new[0] != old[0], new[1] != old[1]];
                let known_asserted = new[0] == policy.preset || new[1] == policy.clear;
                let normal_caused =
                    normal != prior && !known_asserted && no_normal_change != result;
                causes.primary &= normal_caused;
                causes.gate &= normal_caused;
                causes.preset =
                    changed[0] && (preset_alone != no_async_change || no_async_change != result);
                causes.clear =
                    changed[1] && (clear_alone != no_async_change || no_async_change != result);
                // Simultaneous controls may reach the same result as a
                // normal capture. A known asserted control dominates that
                // capture, even if its counterfactual happens to agree.
                if known_asserted && !causes.preset && !causes.clear {
                    causes.preset = new[0] == policy.preset;
                    causes.clear = new[1] == policy.clear;
                }
                if !causes.primary && !causes.gate && !causes.preset && !causes.clear {
                    causes.primary = true;
                }
            }
            Ok(result)
        }
        _ => Err(SpiceError::circuit("invalid asynchronous logic inputs")),
    }
}

/// Complete directional propagation tables for additional causal inputs.
/// Missing tables inherit the explicit primary path.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CausalPaths {
    pub(super) gate: Option<Delays>,
    pub(super) preset: Option<Delays>,
    pub(super) clear: Option<Delays>,
}

impl CausalPaths {
    pub(super) fn validate(self, operation: Operation) -> SpiceResult<()> {
        if self.gate.is_some() && !operation.has_gate_path() {
            return Err(SpiceError::circuit(
                "gate propagation path requires transparent storage",
            ));
        }
        if (self.preset.is_some() || self.clear.is_some()) && !operation.has_async_paths() {
            return Err(SpiceError::circuit(
                "asynchronous propagation paths require preset/clear ports",
            ));
        }
        Ok(())
    }

    pub(super) fn with_load(self, load: [Real; 3]) -> SpiceResult<Self> {
        let apply = |path: Option<Delays>| path.map(|delays| delays.with_load(load)).transpose();
        Ok(Self {
            gate: apply(self.gate)?,
            preset: apply(self.preset)?,
            clear: apply(self.clear)?,
        })
    }

    fn delay(self, primary: Delays, causes: Causes, from: Drive, to: Drive) -> Real {
        [
            (causes.primary, primary),
            (causes.gate, self.gate.unwrap_or(primary)),
            (causes.preset, self.preset.unwrap_or(primary)),
            (causes.clear, self.clear.unwrap_or(primary)),
        ]
        .into_iter()
        .filter(|(active, _)| *active)
        .map(|(_, path)| path.transition(from, to))
        .fold(0., Real::max)
    }
}

/// Output strength may depend on the logical state (open-drain is the
/// important zero/high-impedance example). It does not erase stored logic.
#[derive(Clone, Copy, Debug)]
pub(super) struct DriveStrengths(pub(super) [Strength; 3]);

impl DriveStrengths {
    pub(super) const fn uniform(strength: Strength) -> Self {
        Self([strength; 3])
    }

    pub(super) const fn drive(self, logic: Logic) -> Drive {
        let index = match logic {
            Logic::Zero => 0,
            Logic::One => 1,
            Logic::Unknown | Logic::Rising | Logic::Falling => 2,
        };
        Drive {
            logic,
            strength: self.0[index],
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct LogicKernel {
    pub(super) operation: Operation,
    pub(super) inputs: usize,
    pub(super) initial: Logic,
    pub(super) delays: Delays,
    pub(super) causal: CausalPaths,
    pub(super) mode: DelayMode,
    pub(super) pulse: Option<PulseFilter>,
    pub(super) strength: DriveStrengths,
    /// Input inversion at digital terminals, after net resolution.
    pub(super) invert: Vec<bool>,
    pub(super) timing: Option<TimingContract>,
}

impl LogicKernel {
    pub(super) fn supports_causal_transport(&self) -> bool {
        // Zero-delay inertial scheduling has no pulse to reject. A
        // nonzero inertial path or a separate pulse filter needs causal
        // cancellation semantics, not an implicit transport conversion.
        let zero = self.delays.is_zero()
            && [self.causal.gate, self.causal.preset, self.causal.clear]
                .into_iter()
                .all(|d| d.is_none_or(Delays::is_zero));
        self.pulse.is_none() && (self.mode == DelayMode::Transport || zero)
    }
}

#[derive(Clone, Debug)]
pub(super) struct LogicMemory {
    pub(super) requested: Drive,
    clock: Logic,
    clock_ambiguity: Option<ClockAmbiguity>,
    data: Logic,
    asynchronous: [Logic; 2],
}

impl LogicMemory {
    pub(super) fn new(requested: Drive) -> Self {
        Self {
            requested,
            clock: Logic::Unknown,
            clock_ambiguity: None,
            data: Logic::Unknown,
            asynchronous: [Logic::Unknown; 2],
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct LogicState {
    committed: Drive,
    memory: LogicMemory,
    deadline: Option<Real>,
    pulse: PulseState,
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
            committed: self.strength.drive(self.initial),
            memory: LogicMemory::new(self.strength.drive(self.initial)),
            deadline: None,
            pulse: PulseState::default(),
            timing: self.timing.as_ref().map(TimingContract::initial_state),
            diagnostics: Vec::new(),
        }
    }

    fn take_diagnostics(&self, state: &mut Self::State) -> Vec<ModelDiagnostic> {
        std::mem::take(&mut state.diagnostics)
    }

    fn next_wakeup(&self, state: &Self::State) -> Option<Real> {
        state.pulse.next_wakeup().or(state.deadline)
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
            state.committed = self.strength.drive(self.initial);
            state.memory = LogicMemory::new(state.committed);
            state.deadline = None;
            state.pulse = PulseState::default();
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
        let (mut desired, causes) =
            self.operation
                .desired(inputs, &mut state.memory, context.initial, self.strength)?;
        if unknown {
            desired = self.strength.drive(Logic::Unknown);
            if let Some(phase) = &mut state.memory.clock_ambiguity {
                phase.force(Logic::Unknown);
            }
        }
        let transition = |from, to| self.causal.delay(self.delays, causes, from, to);
        let delay = if context.initial {
            state.committed = desired;
            state.memory.requested = desired;
            0.
        } else if let Some(filter) = self.pulse {
            filter.evaluate(
                context.time,
                desired,
                transition,
                &mut state.memory.requested,
                &mut state.committed,
                &mut state.pulse,
            )?;
            0.
        } else if self.mode == DelayMode::Transport {
            // The requested logical state changes immediately; the queue
            // delivers its output at the propagation time.
            let delay = transition(state.memory.requested, desired);
            state.committed = desired;
            state.memory.requested = desired;
            delay
        } else {
            if state
                .deadline
                .is_some_and(|deadline| context.time >= deadline)
                && desired == state.memory.requested
            {
                state.committed = state.memory.requested;
                state.deadline = None;
            }
            if desired == state.committed {
                state.memory.requested = desired;
                state.deadline = None;
            } else if state.deadline.is_none() || desired != state.memory.requested {
                state.memory.requested = desired;
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
                strength: DriveStrengths::uniform(Strength::STRONG),
                pulse: None,
                invert: Vec::new(),
                causal: CausalPaths::default(),
                timing: None,
            },
            &[data, enable],
            &[bus],
        )?;
        let mut engine = Engine::new(network, Limits::default())?;
        let mut trial = engine.trial();
        let weak = Drive {
            logic: Logic::Zero,
            strength: Strength::RESISTIVE,
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
                    Strength::STRONG
                } else {
                    Strength::RESISTIVE
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
                strength: DriveStrengths::uniform(Strength::STRONG),
                pulse: None,
                invert: Vec::new(),
                causal: CausalPaths::default(),
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
    fn causal_paths_match_independent_ngspice_clock_and_latch() -> SpiceResult<()> {
        let unit = 2_f64.powi(-20);
        for mode in [DelayMode::Transport, DelayMode::Inertial] {
            let mut network = Network::default();
            let inputs: Vec<_> = (0..4).map(|_| network.add_net()).collect();
            let outputs = [network.add_net(), network.add_net()];
            let sources = inputs
                .iter()
                .map(|&net| network.add_source(net))
                .collect::<SpiceResult<Vec<_>>>()?;
            for (index, operation, primary) in [(0, 11., [3., 4.]), (1, 15., [7., 8.])] {
                let operation = Operation::from_value(
                    operation,
                    4,
                    Some(AsyncPolicy {
                        preset: Logic::One,
                        clear: Logic::One,
                        both: Logic::Unknown,
                    }),
                )?;
                network.add_model(
                    LogicKernel {
                        operation,
                        inputs: 4,
                        initial: Logic::Zero,
                        delays: Delays::new(primary[0] * unit, primary[1] * unit)?,
                        causal: CausalPaths {
                            gate: if index == 1 {
                                Some(Delays::new(5. * unit, 6. * unit)?)
                            } else {
                                None
                            },
                            preset: Some(Delays::new(10. * unit, 11. * unit)?),
                            clear: Some(Delays::new(9. * unit, 10. * unit)?),
                        },
                        mode,
                        strength: DriveStrengths::uniform(Strength::STRONG),
                        pulse: None,
                        invert: Vec::new(),
                        timing: None,
                    },
                    &inputs,
                    &outputs[index..=index],
                )?;
            }
            let mut engine = Engine::new(network, Limits::default())?;
            let mut trial = engine.trial();
            let bit = |field: &str| {
                if field == "1s" {
                    Logic::One
                } else {
                    Logic::Zero
                }
            };
            for row in include_str!("../../../tests/fixtures/native-logic/causal.txt").lines() {
                let fields: Vec<_> = row.split_whitespace().collect();
                let time = fields[0].parse::<Real>().unwrap();
                for (&source, field) in sources.iter().zip(&fields[1..]) {
                    trial.schedule(source, time, Drive::strong(bit(field)))?;
                }
            }
            let mut reference = [Vec::new(), Vec::new()];
            for row in include_str!("../../../tests/fixtures/native-logic/causal.csv")
                .lines()
                .skip(1)
            {
                let fields: Vec<_> = row.split(',').collect();
                let time = fields[0].parse::<Real>().unwrap();
                trial.advance_to(time)?;
                for (index, &output) in outputs.iter().enumerate() {
                    let expected = bit(fields[5 + index]);
                    assert_eq!(
                        trial.value(output),
                        Some(Drive::strong(expected)),
                        "{row}, output {index}, {mode:?}"
                    );
                    if reference[index]
                        .last()
                        .is_none_or(|&(_, prior)| prior != expected)
                    {
                        reference[index].push((time, expected));
                    }
                }
            }
            for (index, &output) in outputs.iter().enumerate() {
                let actual: Vec<_> = trial
                    .changes()
                    .iter()
                    .filter(|change| change.net == output)
                    .map(|change| (change.time, change.drive.logic))
                    .collect();
                assert_eq!(actual, reference[index], "output {index}, {mode:?}");
            }
            trial.commit()?;
        }
        Ok(())
    }

    #[test]
    fn native_pulse_policy_preserves_direction_and_correlated_clock_excursions() -> SpiceResult<()>
    {
        use Logic::{Falling, One, Rising, Unknown, Zero};
        let mut network = Network::default();
        let data = network.add_net();
        let enable = network.add_net();
        let pulse = network.add_net();
        let inverted = network.add_net();
        let toggle = network.add_net();
        let d = network.add_source(data)?;
        let e = network.add_source(enable)?;
        let kernel = |operation, inputs, initial, delays, pulse| LogicKernel {
            operation,
            inputs,
            initial,
            delays,
            pulse,
            causal: CausalPaths::default(),
            mode: DelayMode::Transport,
            strength: DriveStrengths::uniform(Strength::STRONG),
            invert: Vec::new(),
            timing: None,
        };
        network.add_model(
            kernel(
                Operation::Gate(GateKind::Buffer),
                1,
                Zero,
                Delays::new(10., 10.)?,
                Some(PulseFilter::new(2.)?),
            ),
            &[data],
            &[pulse],
        )?;
        network.add_model(
            kernel(
                Operation::Gate(GateKind::Inverter),
                1,
                One,
                Delays::ZERO,
                None,
            ),
            &[pulse],
            &[inverted],
        )?;
        network.add_model(
            kernel(
                Operation::from_value(20., 2, None)?,
                2,
                Zero,
                Delays::ZERO,
                None,
            ),
            &[enable, pulse],
            &[toggle],
        )?;
        let mut engine = Engine::new(network, Limits::default())?;
        let mut init = engine.trial();
        init.schedule(d, 0., Drive::strong(Zero))?;
        init.schedule(e, 0., Drive::strong(One))?;
        init.schedule(d, 20., Drive::strong(One))?;
        init.advance_to(21.)?;
        init.commit()?;
        // A shorter pulse and the speculative state changes disappear.
        let mut rejected = engine.trial();
        rejected.schedule(d, 21.5, Drive::strong(Zero))?;
        rejected.advance_to(40.)?;
        assert_eq!(rejected.value(pulse), Some(Drive::strong(Zero)));
        assert_eq!(rejected.value(toggle), Some(Drive::strong(Zero)));
        drop(rejected);
        let mut trial = engine.trial();
        trial.schedule(d, 22., Drive::strong(Zero))?;
        trial.advance_to(29.)?;
        assert_eq!(trial.value(pulse), Some(Drive::strong(Zero)));
        trial.advance_to(30.)?;
        assert_eq!(trial.value(pulse), Some(Drive::strong(Rising)));
        assert_eq!(trial.value(inverted), Some(Drive::strong(Falling)));
        assert_eq!(trial.value(toggle), Some(Drive::strong(Unknown)));
        trial.advance_to(32.)?;
        assert_eq!(trial.value(pulse), Some(Drive::strong(Zero)));
        assert_eq!(trial.value(toggle), Some(Drive::strong(Zero)));
        trial.commit()?;
        Ok(())
    }

    #[test]
    fn directional_gate_dominance_and_convergence_are_distinct_from_unknown() {
        use Logic::{Falling, One, Rising, Unknown, Zero};
        let evaluate = |kind: GateKind, a, b| kind.evaluate(&[Drive::strong(a), Drive::strong(b)]);
        assert_eq!(evaluate(GateKind::And, Zero, Rising), Zero);
        assert_eq!(evaluate(GateKind::And, Rising, One), Rising);
        assert_eq!(evaluate(GateKind::And, Rising, Falling), Rising);
        assert_eq!(evaluate(GateKind::Or, Rising, Falling), Falling);
        assert_eq!(evaluate(GateKind::And, Rising, Unknown), Unknown);
        assert_eq!(evaluate(GateKind::Xor, Rising, Falling), Falling);
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
                strength: DriveStrengths::uniform(Strength::STRONG),
                pulse: None,
                invert: Vec::new(),
                causal: CausalPaths::default(),
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
        assert_eq!(records[0].timing().unwrap().kind, "setup");
        assert_eq!(engine.stats().diagnostics, 1);
        assert_eq!(engine.value(output), Some(Drive::strong(Logic::One)));
        Ok(())
    }

    #[test]
    fn fatal_timing_violation_is_deferred_to_commit() -> SpiceResult<()> {
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
        network.add_model(
            LogicKernel {
                operation,
                inputs: 2,
                initial: Logic::Zero,
                delays: Delays::ZERO,
                mode: DelayMode::Inertial,
                strength: DriveStrengths::uniform(Strength::STRONG),
                pulse: None,
                invert: Vec::new(),
                causal: CausalPaths::default(),
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
        assert!(rejected.diagnostics()[0].timing().unwrap().fatal);
        assert!(
            rejected
                .commit()
                .unwrap_err()
                .to_string()
                .contains("setup violation")
        );
        assert!(engine.diagnostics().is_empty());
        let mut accepted = engine.trial();
        accepted.schedule(c, 3., Drive::strong(Logic::One))?;
        accepted.advance_to(3.)?;
        accepted.commit()?;
        assert!(engine.diagnostics().is_empty());
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
                    strength: DriveStrengths::uniform(Strength::STRONG),
                    pulse: None,
                    invert: Vec::new(),
                    causal: CausalPaths::default(),
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
                strength: DriveStrengths::uniform(Strength::STRONG),
                pulse: None,
                invert: Vec::new(),
                causal: CausalPaths::default(),
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
                strength: DriveStrengths::uniform(Strength::STRONG),
                pulse: None,
                invert: Vec::new(),
                causal: CausalPaths::default(),
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
