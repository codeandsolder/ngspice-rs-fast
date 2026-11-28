//! Clocked storage samples the causal waveform, not its displayed phase.
//! Each announced edge is processed once; completion only settles its
//! envelope. History belongs to the disposable digital trial.
use super::causal_checks::{aperture, check_capture, check_widths, controls_may_be_inactive};
use super::causal_history::{InputEdge, PortHistory, Relation, qualify_inputs, sample_joint};
use super::sequential_timing::may_trigger;
use super::source_logic::{AsyncPolicy, ClockedKind, Edge, LogicKernel, Operation, consensus};
use super::window_gate::WindowPaths;
use super::{
    CodeModel, DelayMode, Delays, Drive, EventContext, Logic, ModelDiagnostic, ModelSchema, Output,
    WindowDelays,
};
use crate::analysis::digital::timing::{Arrival, Origin, TimedInput, TimedOutput, TimingWindow};
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::collections::VecDeque;

#[derive(Debug)]
pub(super) struct WindowClocked {
    kernel: LogicKernel,
    kind: ClockedKind,
    trigger: Edge,
    asynchronous: Option<AsyncPolicy>,
    delays: WindowDelays,
    preset: WindowDelays,
    clear: WindowDelays,
}

#[derive(Clone, Copy, Debug)]
enum SelectedPaths {
    One(WindowDelays),
    Both(WindowDelays, WindowDelays),
}

impl From<WindowDelays> for SelectedPaths {
    fn from(delays: WindowDelays) -> Self {
        Self::One(delays)
    }
}

impl SelectedPaths {
    fn arrival(self, cause: &TimingWindow, from: Drive, to: Drive) -> SpiceResult<TimingWindow> {
        let propagate =
            |delay: WindowDelays| cause.after(delay.transition(from, to)?, Origin::independent()?);
        match self {
            Self::One(delay) => propagate(delay),
            Self::Both(a, b) => TimingWindow::latest(vec![propagate(a)?, propagate(b)?]),
        }
    }
}

#[derive(Clone, Debug)]
struct Capture {
    edge: InputEdge,
    before: Logic,
    result: Logic,
    poisoned: bool,
}

#[derive(Clone, Debug)]
pub(super) struct WindowClockedState {
    requested: Drive,
    ports: Vec<PortHistory>,
    captures: VecDeque<Capture>,
    // Asynchronous assertions supersede the clocked state. Retain the
    // last causal control edge to establish its order against capture.
    control: Option<InputEdge>,
    diagnostics: Vec<ModelDiagnostic>,
}

impl WindowClocked {
    pub(super) fn new(
        kernel: LogicKernel,
        delays: WindowDelays,
        paths: WindowPaths,
    ) -> SpiceResult<Self> {
        let Operation::Clocked {
            kind,
            trigger,
            asynchronous,
        } = kernel.operation
        else {
            return Err(SpiceError::circuit("causal clocked binding mismatch"));
        };
        let preset = match paths.preset {
            Some(path) => path,
            None => kernel
                .causal
                .preset
                .map(WindowDelays::exact)
                .unwrap_or(delays),
        };
        let clear = match paths.clear {
            Some(path) => path,
            None => kernel
                .causal
                .clear
                .map(WindowDelays::exact)
                .unwrap_or(delays),
        };
        if paths.gate.is_some() {
            return Err(SpiceError::circuit(
                "gate path requires transparent storage",
            ));
        }
        Ok(Self {
            kernel,
            kind,
            trigger,
            asynchronous,
            delays,
            preset,
            clear,
        })
    }

    fn clock(&self) -> usize {
        self.kind.arity() - 1
    }

    fn sample(&self, ports: &[PortHistory], edge: &InputEdge, prior: Logic) -> SpiceResult<Logic> {
        let cases = sample_joint(ports, edge)?;
        let results = cases
            .branches()
            .map(|(_, inputs)| {
                let captured = self.kind.capture(&inputs[..self.kind.arity()], prior)?;
                let clocked = self
                    .trigger
                    .capture(edge.from.logic, edge.to.logic, prior, captured);
                Ok(self.resolve_controls(inputs, clocked))
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        Ok(consensus(results))
    }

    fn resolve_controls(&self, inputs: &[Drive], normal: Logic) -> Logic {
        self.asynchronous.map_or(normal, |policy| {
            policy.resolve(
                inputs[self.kind.arity()].logic,
                inputs[self.kind.arity() + 1].logic,
                normal,
            )
        })
    }

    fn enabled_at(&self, ports: &[PortHistory], edge: &InputEdge) -> SpiceResult<bool> {
        let Some(policy) = self.asynchronous else {
            return Ok(true);
        };
        controls_may_be_inactive(policy, &ports[self.kind.arity()..], edge)
    }

    fn initial(&self, inputs: &[TimedInput], state: &mut WindowClockedState) -> Vec<TimedOutput> {
        state.ports = inputs.iter().map(PortHistory::new).collect();
        state.captures.clear();
        state.control = None;
        let drives: Vec<_> = inputs.iter().map(|input| input.drive).collect();
        let initial = self.resolve_controls(&drives, self.kernel.initial);
        state.requested = self.kernel.strength.drive(initial);
        vec![TimedOutput::Scalar(Output {
            drive: state.requested,
            delays: Delays::ZERO,
            mode: DelayMode::Transport,
        })]
    }

    fn transition(
        &self,
        state: &mut WindowClockedState,
        value: Logic,
        cause: &TimingWindow,
        delays: SelectedPaths,
        waveform: &mut Vec<Arrival>,
    ) -> SpiceResult<()> {
        let target = self.kernel.strength.drive(value);
        if state.requested != target {
            let window = delays.arrival(cause, state.requested, target)?;
            waveform.push(Arrival::new(state.requested, target, window)?);
            state.requested = target;
        }
        Ok(())
    }

    fn revise_captures(
        &self,
        state: &mut WindowClockedState,
        changes: &[Option<InputEdge>],
    ) -> SpiceResult<Option<TimingWindow>> {
        let data = &changes[..self.clock()];
        if data.iter().all(Option::is_none) {
            return Ok(None);
        }
        let mut revised = Vec::new();
        let mut prior = state.captures.front().map(|capture| capture.before);
        for capture in &mut state.captures {
            let before = prior.unwrap_or(capture.before);
            let old = capture.result;
            let mut causes = Vec::new();
            for (index, edge) in data.iter().enumerate() {
                let Some(edge) = edge.as_ref().filter(|edge| edge.logic_changed()) else {
                    continue;
                };
                if edge.occurs_before(&capture.edge)? != Relation::After {
                    causes.push(edge.window.clone());
                }
                if let Some(timing) = &self.kernel.timing
                    && self.enabled_at(&state.ports, &capture.edge)?
                    && aperture(timing, &capture.edge, edge, index, &mut state.diagnostics)?
                {
                    capture.poisoned = true;
                    causes.push(edge.window.clone());
                }
            }
            capture.before = before;
            capture.result = if capture.poisoned {
                Logic::Unknown
            } else {
                self.sample(&state.ports, &capture.edge, before)?
            };
            prior = Some(capture.result);
            if capture.result != old {
                revised.extend(causes);
            }
        }
        if state
            .captures
            .back()
            .is_some_and(|capture| capture.result != state.requested.logic)
        {
            // A later data announcement cannot retroactively change an
            // already delivered output. The revision is causal from the
            // newly announced data (and its capture), not an invented
            // earlier scalar timestamp.
            if revised.is_empty() {
                revised.extend(data.iter().flatten().map(|edge| edge.window.clone()));
            }
            let cause = TimingWindow::earliest(revised)?;
            let capture = state
                .captures
                .back()
                .ok_or_else(|| SpiceError::circuit("missing revised capture"))?;
            return Ok(Some(TimingWindow::latest(vec![
                cause,
                capture.edge.window.clone(),
            ])?));
        }
        Ok(None)
    }

    fn prune(&self, state: &mut WindowClockedState, now: Real) {
        let margin = self.kernel.timing.as_ref().map_or(0., |timing| {
            timing
                .setup
                .iter()
                .chain(&timing.hold)
                .chain(&timing.recovery)
                .copied()
                .fold(0., Real::max)
        });
        let cutoff = now - margin;
        // A completed capture outside all relevant apertures will no
        // longer be revised by a subsequently announced input edge.
        while state
            .captures
            .front()
            .is_some_and(|c| c.edge.window.bounds().maximum() < cutoff)
        {
            state.captures.pop_front();
        }
        let earliest = state
            .captures
            .front()
            .map_or(cutoff, |c| c.edge.window.bounds().minimum() - margin);
        for port in &mut state.ports {
            port.prune(earliest);
        }
    }

    fn control_edge(
        &self,
        changes: &[Option<InputEdge>],
    ) -> SpiceResult<Option<(InputEdge, SelectedPaths)>> {
        let Some(_) = self.asynchronous else {
            return Ok(None);
        };
        let preset = changes[self.kind.arity()]
            .as_ref()
            .filter(|edge| edge.logic_changed());
        let clear = changes[self.kind.arity() + 1]
            .as_ref()
            .filter(|edge| edge.logic_changed());
        match (preset, clear) {
            (None, None) => Ok(None),
            (Some(p), None) => Ok(Some((p.clone(), self.preset.into()))),
            (None, Some(c)) => Ok(Some((c.clone(), self.clear.into()))),
            (Some(p), Some(c)) => {
                let separation = p.relative_to(c)?;
                if !separation.is_exact() || separation.minimum() != 0. {
                    return Err(SpiceError::circuit(
                        "independently timed simultaneous asynchronous controls require conditional storage",
                    ));
                }
                // Shared transition time; conflict resolution determines
                // the output, and its active controlling path is selected.
                let policy = self
                    .asynchronous
                    .ok_or_else(|| SpiceError::circuit("missing asynchronous policy"))?;
                Ok(Some(
                    if p.to.logic == policy.preset && c.to.logic == policy.clear {
                        (c.clone(), SelectedPaths::Both(self.preset, self.clear))
                    } else if c.to.logic == policy.clear {
                        (c.clone(), self.clear.into())
                    } else {
                        (p.clone(), self.preset.into())
                    },
                ))
            }
        }
    }
}

impl CodeModel for WindowClocked {
    type State = WindowClockedState;

    fn schema(&self) -> ModelSchema {
        ModelSchema::digital(self.kernel.inputs, 1)
    }
    fn initial_state(&self) -> Self::State {
        WindowClockedState {
            requested: self.kernel.strength.drive(self.kernel.initial),
            ports: Vec::new(),
            captures: VecDeque::new(),
            control: None,
            diagnostics: Vec::new(),
        }
    }
    fn tracks_arrivals(&self) -> bool {
        true
    }
    fn take_diagnostics(&self, state: &mut Self::State) -> Vec<ModelDiagnostic> {
        std::mem::take(&mut state.diagnostics)
    }
    fn evaluate(
        &self,
        _: EventContext,
        _: &[Drive],
        _: &mut Self::State,
    ) -> SpiceResult<Vec<Output>> {
        Err(SpiceError::circuit(
            "causal clocked model requires arrival-aware event engine",
        ))
    }
    fn evaluate_timed(
        &self,
        context: EventContext,
        inputs: &[TimedInput],
        state: &mut Self::State,
    ) -> SpiceResult<Vec<TimedOutput>> {
        if inputs.len() != self.kernel.inputs {
            return Err(SpiceError::circuit(
                "causal clocked input dimension mismatch",
            ));
        }
        let inputs = qualify_inputs(inputs, &self.kernel.invert)?;
        if context.initial {
            return Ok(self.initial(&inputs, state));
        }
        let changes = state
            .ports
            .iter_mut()
            .zip(&inputs)
            .map(|(port, input)| port.observe(context, input))
            .collect::<SpiceResult<Vec<_>>>()?;
        if changes.iter().all(Option::is_none) {
            return Ok(vec![TimedOutput::Retain]);
        }
        self.prune(state, context.time);
        let clock = changes[self.clock()]
            .as_ref()
            .filter(|edge| edge.logic_changed());
        if let Some(edge) = clock {
            if !edge.from.logic.is_binary() || !edge.to.logic.is_binary() {
                return Err(SpiceError::circuit(
                    "unqualified clock ambiguity requires conditional storage",
                ));
            }
            if let Some(previous) = state.ports[self.clock()]
                .edges()
                .iter()
                .rev()
                .filter(|edge| edge.logic_changed())
                .nth(1)
                && previous.occurs_before(edge)? != Relation::Before
            {
                return Err(SpiceError::circuit("clock arrival order is not guaranteed"));
            }
        }
        let width_poison = if let Some(timing) = &self.kernel.timing {
            check_widths(timing, &state.ports, &mut state.diagnostics, &changes)?
        } else {
            false
        };
        let mut waveform = Vec::new();
        if let Some(cause) = self.revise_captures(state, &changes)? {
            let value = state
                .captures
                .back()
                .ok_or_else(|| SpiceError::circuit("missing revised capture"))?
                .result;
            self.transition(state, value, &cause, self.delays.into(), &mut waveform)?;
        }
        if let Some((edge, path)) = self.control_edge(&changes)? {
            if let Some(capture) = state.captures.back()
                && capture.edge.occurs_before(&edge)? != Relation::Before
            {
                return Err(SpiceError::circuit(
                    "overlapping asynchronous and clock arrivals require conditional storage",
                ));
            }
            if let Some(previous) = &state.control
                && previous.occurs_before(&edge)? != Relation::Before
            {
                return Err(SpiceError::circuit(
                    "asynchronous control arrival order is not guaranteed",
                ));
            }
            let cases = sample_joint(&state.ports, &edge)?;
            let value = consensus(
                cases
                    .branches()
                    .map(|(_, drives)| self.resolve_controls(drives, state.requested.logic)),
            );
            // Even an assertion that leaves Q unchanged supersedes earlier
            // captures. It must not be revised by a later data edge.
            if !self.enabled_at(&state.ports, &edge)? || value != state.requested.logic {
                state.captures.clear();
            }
            self.transition(state, value, &edge.window, path, &mut waveform)?;
            state.control = Some(edge);
        }
        if let Some(edge) =
            clock.filter(|edge| may_trigger(self.trigger, edge.from.logic, edge.to.logic))
        {
            if let Some(control) = &state.control
                && control.occurs_before(edge)? != Relation::Before
            {
                return Err(SpiceError::circuit(
                    "clock capture overlaps asynchronous control order",
                ));
            }
            if self.enabled_at(&state.ports, edge)? {
                let before = state.requested.logic;
                let poisoned = match &self.kernel.timing {
                    Some(timing) => {
                        check_capture(timing, &state.ports, &mut state.diagnostics, edge)?
                    }
                    None => false,
                } || width_poison;
                let value = if poisoned {
                    Logic::Unknown
                } else {
                    self.sample(&state.ports, edge, before)?
                };
                state.captures.push_back(Capture {
                    edge: edge.clone(),
                    before,
                    result: value,
                    poisoned,
                });
                self.transition(
                    state,
                    value,
                    &edge.window,
                    self.delays.into(),
                    &mut waveform,
                )?;
            }
        } else if width_poison {
            if let Some(capture) = state.captures.back_mut() {
                capture.result = Logic::Unknown;
                capture.poisoned = true;
            }
            let cause = TimingWindow::latest(
                changes
                    .iter()
                    .flatten()
                    .map(|edge| edge.window.clone())
                    .collect(),
            )?;
            self.transition(
                state,
                Logic::Unknown,
                &cause,
                self.delays.into(),
                &mut waveform,
            )?;
        }
        Ok(vec![if waveform.is_empty() {
            TimedOutput::Retain
        } else {
            TimedOutput::Waveform(waveform)
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::super::Strength;
    use super::super::sequential_timing::TimingSpec;
    use super::super::source_logic::{CausalPaths, DriveStrengths};
    use super::*;
    use crate::analysis::digital::timing::{Bounds, Certainty, DelayWindow};
    use crate::analysis::digital::{Engine, Limits, Network};

    fn model(setup: Real, hold: Real, policy: Real, op: Real) -> SpiceResult<WindowClocked> {
        let operation = Operation::from_value(op, 2, None)?;
        let mut timing = TimingSpec::default();
        timing.set_vector("timing_setup", vec![setup])?;
        timing.set_vector("timing_hold", vec![hold])?;
        timing.set_policy(policy)?;
        let kernel = LogicKernel {
            operation,
            inputs: 2,
            initial: Logic::Zero,
            delays: Delays::ZERO,
            causal: CausalPaths::default(),
            mode: DelayMode::Transport,
            pulse: None,
            strength: DriveStrengths::uniform(Strength::STRONG),
            invert: vec![false; 2],
            timing: timing.bind(operation, 2)?,
        };
        WindowClocked::new(
            kernel,
            WindowDelays::binary(DelayWindow::new(1., 2.)?, DelayWindow::new(1., 2.)?),
            WindowPaths::default(),
        )
    }

    #[test]
    fn correlated_setup_classification_and_capture_are_transactional() -> SpiceResult<()> {
        for (setup, policy, expected, certainty) in [
            (10., 2., Logic::One, None),
            (11., 2., Logic::One, Some(Certainty::PossibleViolation)),
            (11., 1., Logic::Unknown, Some(Certainty::PossibleViolation)),
            (51., 2., Logic::One, Some(Certainty::DefiniteViolation)),
        ] {
            let mut network = Network::default();
            let d = network.add_net();
            let c = network.add_net();
            let q = network.add_net();
            let data = network.add_source(d)?;
            let clock = network.add_source(c)?;
            network.add_model(model(setup, 0., policy, 11.)?, &[d, c], &[q])?;
            let mut engine = Engine::new(network, Limits::default())?;
            let low = Drive::strong(Logic::Zero);
            let high = Drive::strong(Logic::One);
            let root =
                TimingWindow::exact(0.)?.after(DelayWindow::new(15., 30.)?, Origin::shared(0))?;
            let data_at = root.after(DelayWindow::new(10., 30.)?, Origin::shared(1))?;
            let clock_at = root.after(DelayWindow::new(40., 60.)?, Origin::shared(2))?;
            let mut init = engine.trial();
            init.schedule(data, 0., low)?;
            init.schedule(clock, 0., low)?;
            init.schedule_arrival(data, Arrival::new(low, high, data_at)?)?;
            init.schedule_arrival(clock, Arrival::new(low, high, clock_at)?)?;
            init.advance_to(20.)?;
            init.commit()?;
            for commit in [false, true] {
                let mut trial = engine.trial();
                trial.advance_to(100.)?;
                assert_eq!(trial.value(q), Some(Drive::strong(expected)));
                let diagnostics = trial.diagnostics();
                if let Some(certainty) = certainty {
                    assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
                    let timing = diagnostics[0].timing().expect("timing diagnostic");
                    assert_eq!(timing.kind, "setup");
                    assert_eq!(timing.certainty, certainty);
                    assert_eq!((timing.elapsed, timing.maximum_elapsed), (10., 50.));
                } else {
                    assert!(diagnostics.is_empty());
                }
                let arrival = trial.arrival(q).expect("Q capture");
                assert_eq!(arrival.window().bounds(), Bounds::new(56., 92.)?);
                if commit {
                    trial.commit()?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn later_data_edge_inside_hold_aperture_revises_requested_capture() -> SpiceResult<()> {
        let mut network = Network::default();
        let d = network.add_net();
        let c = network.add_net();
        let q = network.add_net();
        let data = network.add_source(d)?;
        let clock = network.add_source(c)?;
        network.add_model(model(0., 3., 1., 11.)?, &[d, c], &[q])?;
        let mut engine = Engine::new(network, Limits::default())?;
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let mut trial = engine.trial();
        trial.schedule(data, 0., high)?;
        trial.schedule(clock, 0., low)?;
        trial.schedule(clock, 10., high)?;
        trial.schedule(data, 12., low)?;
        trial.advance_to(20.)?;
        assert_eq!(trial.value(q), Some(Drive::strong(Logic::Unknown)));
        let diagnostics = trial.diagnostics();
        assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
        let timing = diagnostics[0].timing().expect("timing diagnostic");
        assert_eq!(
            (timing.kind, timing.certainty),
            ("hold", Certainty::DefiniteViolation)
        );
        assert_eq!((timing.elapsed, timing.maximum_elapsed), (2., 2.));
        trial.commit()
    }

    #[test]
    fn bounded_clock_completion_is_not_an_additional_dual_edge_toggle() -> SpiceResult<()> {
        let mut network = Network::default();
        let t = network.add_net();
        let c = network.add_net();
        let q = network.add_net();
        let toggle = network.add_source(t)?;
        let clock = network.add_source(c)?;
        network.add_model(model(0., 0., 2., 20.)?, &[t, c], &[q])?;
        let mut engine = Engine::new(network, Limits::default())?;
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let mut trial = engine.trial();
        trial.schedule(toggle, 0., high)?;
        trial.schedule(clock, 0., low)?;
        let root =
            TimingWindow::exact(0.)?.after(DelayWindow::new(10., 20.)?, Origin::shared(0))?;
        let fall = root.after(DelayWindow::new(2., 2.)?, Origin::shared(1))?;
        trial.schedule_arrival(clock, Arrival::new(low, high, root)?)?;
        trial.schedule_arrival(clock, Arrival::new(high, low, fall)?)?;
        trial.advance_to(40.)?;
        assert_eq!(trial.value(q), Some(low));
        assert_eq!(trial.arrival(q).expect("falling Q").to(), low);
        trial.commit()
    }
    #[test]
    fn capture_excludes_jointly_impossible_visible_data_pulse() -> SpiceResult<()> {
        let mut network = Network::default();
        let d = network.add_net();
        let c = network.add_net();
        let q = network.add_net();
        let data = network.add_source(d)?;
        let clock = network.add_source(c)?;
        network.add_model(model(0., 0., 2., 11.)?, &[d, c], &[q])?;
        let mut engine = Engine::new(network, Limits::default())?;
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        // The pulse is visible only when A <= C < B. At C=20,
        // A=10+20u and B=15+10u make this conjunction impossible.
        // Marginal timing bounds nevertheless admit both endpoints.
        let u = Origin::shared(0);
        let root = TimingWindow::exact(0.)?;
        let rise = root.after(DelayWindow::new(10., 30.)?, u)?;
        let fall = root.after(DelayWindow::new(15., 25.)?, u)?;
        let mut trial = engine.trial();
        trial.schedule(data, 0., low)?;
        trial.schedule(clock, 0., low)?;
        trial.schedule_arrival(data, Arrival::new(low, high, rise)?)?;
        trial.schedule_arrival(data, Arrival::new(high, low, fall)?)?;
        trial.schedule(clock, 20., high)?;
        trial.advance_to(40.)?;
        assert_eq!(trial.value(q), Some(low));
        trial.commit()
    }

    #[test]
    fn capture_combines_data_ports_under_one_origin_assignment() -> SpiceResult<()> {
        let mut network = Network::default();
        let s = network.add_net();
        let r = network.add_net();
        let c = network.add_net();
        let q = network.add_net();
        let set = network.add_source(s)?;
        let reset = network.add_source(r)?;
        let clock = network.add_source(c)?;
        let kernel = LogicKernel {
            operation: Operation::from_value(7., 3, None)?,
            inputs: 3,
            initial: Logic::Zero,
            delays: Delays::ZERO,
            causal: CausalPaths::default(),
            mode: DelayMode::Transport,
            pulse: None,
            strength: DriveStrengths::uniform(Strength::STRONG),
            invert: vec![false; 3],
            timing: None,
        };
        let model = WindowClocked::new(
            kernel,
            WindowDelays::exact(Delays::ZERO),
            WindowPaths::default(),
        )?;
        network.add_model(model, &[s, r, c], &[q])?;
        let mut engine = Engine::new(network, Limits::default())?;
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let both =
            TimingWindow::exact(0.)?.after(DelayWindow::new(10., 30.)?, Origin::shared(0))?;
        let mut trial = engine.trial();
        for source in [set, reset, clock] {
            trial.schedule(source, 0., low)?;
        }
        trial.schedule_arrival(set, Arrival::new(low, high, both.clone())?)?;
        trial.schedule_arrival(reset, Arrival::new(low, high, both)?)?;
        trial.schedule(clock, 20., high)?;
        trial.advance_to(40.)?;
        // SR hold-on-equal sees either (0,0) or (1,1), never (1,0).
        assert_eq!(trial.value(q), Some(low));
        trial.commit()
    }
}
