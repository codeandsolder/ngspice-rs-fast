//! Arrival-aware level-sensitive storage. The scalar state transition and
//! controlling-path selection are shared with the deterministic kernel;
//! only the time algebra and accepted timing history differ.

use super::causal_checks::{aperture, capture_enabled, check_capture, check_widths};
use super::causal_history::{InputEdge, PortHistory, Relation, ordered_groups, qualify_inputs};
use super::sequential_timing::may_trigger;
use super::source_logic::{Causes, LogicKernel, LogicMemory, Operation};
use super::window_gate::WindowPaths;
use super::{
    CodeModel, DelayMode, Delays, Drive, EventContext, ModelDiagnostic, ModelSchema, Output,
    WindowDelays,
};
use crate::analysis::digital::timing::{Arrival, Origin, TimedInput, TimedOutput, TimingWindow};
use crate::primitives::{Real, SpiceError, SpiceResult};

#[derive(Debug)]
pub(super) struct WindowLatch {
    kernel: LogicKernel,
    paths: [WindowDelays; 4],
}

#[derive(Clone, Debug)]
pub(super) struct WindowLatchState {
    logic: LogicMemory,
    inputs: Vec<Drive>,
    ports: Vec<PortHistory>,
    last: Option<InputEdge>,
    capture: Option<InputEdge>,
    diagnostics: Vec<ModelDiagnostic>,
}

impl WindowLatch {
    pub(super) fn new(
        kernel: LogicKernel,
        primary: WindowDelays,
        paths: WindowPaths,
    ) -> SpiceResult<Self> {
        if !matches!(
            kernel.operation,
            Operation::Enabled { .. } | Operation::Sr(_)
        ) {
            return Err(SpiceError::circuit("causal latch binding mismatch"));
        }
        let path = |bounded: Option<WindowDelays>, exact: Option<Delays>| {
            bounded
                .or_else(|| exact.map(WindowDelays::exact))
                .unwrap_or(primary)
        };
        let paths = [
            primary,
            path(paths.gate, kernel.causal.gate),
            path(paths.preset, kernel.causal.preset),
            path(paths.clear, kernel.causal.clear),
        ];
        Ok(Self { kernel, paths })
    }

    fn emit(
        &self,
        cause: &TimingWindow,
        causes: Causes,
        before: Drive,
        after: Drive,
        waveform: &mut Vec<Arrival>,
    ) -> SpiceResult<()> {
        if before == after {
            return Ok(());
        }
        let arrivals = [causes.primary, causes.gate, causes.preset, causes.clear]
            .into_iter()
            .zip(self.paths)
            .filter(|(active, _)| *active)
            .map(|(_, path)| cause.after(path.transition(before, after)?, Origin::independent()?))
            .collect::<SpiceResult<Vec<_>>>()?;
        let arrival = TimingWindow::latest(arrivals)?;
        waveform.push(Arrival::new(before, after, arrival)?);
        Ok(())
    }

    fn prune(&self, state: &mut WindowLatchState, now: Real) {
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
        if state
            .capture
            .as_ref()
            .is_some_and(|edge| edge.window.bounds().maximum() < cutoff)
        {
            state.capture = None;
        }
        for port in &mut state.ports {
            port.prune(cutoff);
        }
    }

    fn group(
        &self,
        context: EventContext,
        inputs: &[TimedInput],
        indices: &[usize],
        state: &mut WindowLatchState,
        waveform: &mut Vec<Arrival>,
    ) -> SpiceResult<()> {
        let mut changes = vec![None; inputs.len()];
        for &index in indices {
            let edge = state.ports[index].observe(context, &inputs[index])?;
            if let Some(edge) = &edge {
                state.inputs[index] = edge.to;
            }
            changes[index] = edge;
        }
        let representative = changes
            .iter()
            .flatten()
            .next()
            .ok_or_else(|| SpiceError::circuit("missing causal latch edge"))?;
        if let Some(last) = &state.last
            && last.occurs_before(representative)? != Relation::Before
        {
            return Err(SpiceError::circuit(
                "latch event ordering requires a conditional waveform",
            ));
        }

        self.prune(state, context.time);
        let before = state.logic.requested;
        let (mut target, causes) = self.kernel.operation.desired(
            &state.inputs,
            &mut state.logic,
            false,
            self.kernel.strength,
        )?;
        if let Some(timing) = &self.kernel.timing {
            let mut poisoned =
                check_widths(timing, &state.ports, &mut state.diagnostics, &changes)?;
            if capture_enabled(timing, &state.ports, representative)? {
                // Check the previous closing edge's hold aperture before
                // installing the new capture. All groups are processed in
                // proven physical order, including delta-group equality.
                if let Some(capture) = &state.capture {
                    for (index, edge) in changes.iter().enumerate().take(timing.clock) {
                        if let Some(edge) = edge {
                            poisoned |=
                                aperture(timing, capture, edge, index, &mut state.diagnostics)?;
                        }
                    }
                }
                if let Some(edge) = &changes[timing.clock]
                    && may_trigger(timing.edge, edge.from.logic, edge.to.logic)
                {
                    poisoned |= check_capture(timing, &state.ports, &mut state.diagnostics, edge)?;
                    state.capture = Some(edge.clone());
                }
            } else {
                state.capture = None;
            }
            if poisoned {
                target = self.kernel.strength.drive(super::Logic::Unknown);
            }
        }
        state.logic.requested = target;
        // A timing violation may change the result even if the ordinary
        // truth table leaves it unchanged. The shared desired() always
        // provides at least its primary or gate causal path in that case.
        self.emit(&representative.window, causes, before, target, waveform)?;
        state.last = Some(representative.clone());
        Ok(())
    }
}

impl CodeModel for WindowLatch {
    type State = WindowLatchState;

    fn schema(&self) -> ModelSchema {
        ModelSchema::digital(self.kernel.inputs, 1)
    }
    fn initial_state(&self) -> Self::State {
        WindowLatchState {
            logic: LogicMemory::new(self.kernel.strength.drive(self.kernel.initial)),
            inputs: Vec::new(),
            ports: Vec::new(),
            last: None,
            capture: None,
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
            "causal latch requires arrival-aware event engine",
        ))
    }
    fn evaluate_timed(
        &self,
        context: EventContext,
        inputs: &[TimedInput],
        state: &mut Self::State,
    ) -> SpiceResult<Vec<TimedOutput>> {
        if inputs.len() != self.kernel.inputs {
            return Err(SpiceError::circuit("causal latch input dimension mismatch"));
        }
        let inputs = qualify_inputs(inputs, &self.kernel.invert)?;
        if context.initial {
            state.logic = LogicMemory::new(self.kernel.strength.drive(self.kernel.initial));
            state.ports = inputs.iter().map(PortHistory::new).collect();
            state.inputs = inputs.iter().map(|input| input.drive).collect();
            state.last = None;
            state.capture = None;
            let (drive, _) = self.kernel.operation.desired(
                &state.inputs,
                &mut state.logic,
                true,
                self.kernel.strength,
            )?;
            state.logic.requested = drive;
            return Ok(vec![TimedOutput::Scalar(Output {
                drive,
                delays: Delays::ZERO,
                mode: DelayMode::Transport,
            })]);
        }

        let preview = state
            .ports
            .iter()
            .zip(&inputs)
            .map(|(port, input)| port.preview(context, input))
            .collect::<SpiceResult<Vec<_>>>()?;
        let groups = ordered_groups(preview.iter().enumerate().filter_map(|(i, e)| {
            e.as_ref()
                .filter(|e| e.logic_changed())
                .map(|e| (i, &e.window))
        }))?
        .ok_or_else(|| {
            SpiceError::circuit("latch event ordering requires a conditional waveform")
        })?;
        // Rank-only changes update endpoint identity but are not logic
        // transitions and do not alter the storage or aperture history.
        for (index, edge) in preview.iter().enumerate() {
            if edge.as_ref().is_some_and(|edge| !edge.logic_changed()) {
                state.ports[index].observe(context, &inputs[index])?;
            }
        }
        let mut waveform = Vec::new();
        for group in groups {
            self.group(context, &inputs, &group, state, &mut waveform)?;
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
    use super::super::sequential_timing::TimingSpec;
    use super::super::source_logic::{CausalPaths, DriveStrengths};
    use super::super::{Logic, Strength};
    use super::*;
    use crate::analysis::digital::timing::{Bounds, DelayWindow};
    use crate::analysis::digital::{Engine, Limits, Network};

    fn model(op: Real, setup: Real, hold: Real, policy: Real) -> SpiceResult<WindowLatch> {
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
        WindowLatch::new(
            kernel,
            WindowDelays::binary(DelayWindow::new(1., 2.)?, DelayWindow::new(1., 2.)?),
            WindowPaths {
                gate: Some(WindowDelays::binary(
                    DelayWindow::new(3., 4.)?,
                    DelayWindow::new(3., 4.)?,
                )),
                ..WindowPaths::default()
            },
        )
    }

    #[test]
    fn correlated_transparent_latch_has_distinct_opening_and_data_paths() -> SpiceResult<()> {
        for (op, inactive, active) in [
            (15., Logic::Zero, Logic::One),
            (16., Logic::One, Logic::Zero),
        ] {
            let mut network = Network::default();
            let d = network.add_net();
            let g = network.add_net();
            let q = network.add_net();
            let data = network.add_source(d)?;
            let gate = network.add_source(g)?;
            network.add_model(model(op, 0., 0., 2.)?, &[d, g], &[q])?;
            let mut engine = Engine::new(network, Limits::default())?;
            let low = Drive::strong(Logic::Zero);
            let high = Drive::strong(Logic::One);
            let inactive = Drive::strong(inactive);
            let active = Drive::strong(active);
            let root =
                TimingWindow::exact(0.)?.after(DelayWindow::new(10., 30.)?, Origin::shared(0))?;
            let opening = root.after(DelayWindow::new(5., 5.)?, Origin::shared(1))?;
            let falling = root.after(DelayWindow::new(10., 10.)?, Origin::shared(2))?;
            let closing = root.after(DelayWindow::new(14., 14.)?, Origin::shared(3))?;
            let rising = root.after(DelayWindow::new(17., 17.)?, Origin::shared(4))?;
            let mut trial = engine.trial();
            trial.schedule(data, 0., high)?;
            trial.schedule(gate, 0., inactive)?;
            trial.schedule_arrival(gate, Arrival::new(inactive, active, opening)?)?;
            trial.schedule_arrival(data, Arrival::new(high, low, falling)?)?;
            trial.schedule_arrival(gate, Arrival::new(active, inactive, closing)?)?;
            trial.schedule_arrival(data, Arrival::new(low, high, rising)?)?;
            trial.advance_to(19.)?;
            assert_eq!(trial.value(q), Some(Drive::strong(Logic::Rising)));
            assert_eq!(
                trial.arrival(q).expect("opening").window().bounds(),
                Bounds::new(18., 39.)?
            );
            trial.advance_to(25.)?;
            assert_eq!(trial.value(q), Some(Drive::strong(Logic::Unknown)));
            assert_eq!(
                trial.arrival(q).expect("data").window().bounds(),
                Bounds::new(21., 42.)?
            );
            trial.advance_to(40.)?;
            assert_eq!(trial.value(q), Some(Drive::strong(Logic::Falling)));
            trial.advance_to(50.)?;
            assert_eq!(trial.value(q), Some(low));
            trial.commit()?;
        }
        Ok(())
    }

    #[test]
    fn correlated_closing_aperture_is_checked_without_absolute_interval_flattening()
    -> SpiceResult<()> {
        use crate::analysis::digital::timing::Certainty;
        for (policy, expected) in [(1., Logic::Unknown), (2., Logic::One)] {
            let mut network = Network::default();
            let d = network.add_net();
            let g = network.add_net();
            let q = network.add_net();
            let data = network.add_source(d)?;
            let gate = network.add_source(g)?;
            network.add_model(model(15., 4., 4., policy)?, &[d, g], &[q])?;
            let mut engine = Engine::new(network, Limits::default())?;
            let low = Drive::strong(Logic::Zero);
            let high = Drive::strong(Logic::One);
            let root =
                TimingWindow::exact(0.)?.after(DelayWindow::new(10., 30.)?, Origin::shared(0))?;
            let data_at = root.clone();
            let close = root.after(DelayWindow::new(3., 3.)?, Origin::shared(1))?;
            let after = root.after(DelayWindow::new(6., 6.)?, Origin::shared(2))?;
            let mut trial = engine.trial();
            trial.schedule(data, 0., low)?;
            trial.schedule(gate, 0., high)?;
            trial.schedule_arrival(data, Arrival::new(low, high, data_at)?)?;
            trial.schedule_arrival(gate, Arrival::new(high, low, close)?)?;
            trial.schedule_arrival(data, Arrival::new(high, low, after)?)?;
            trial.advance_to(50.)?;
            assert_eq!(trial.value(q), Some(Drive::strong(expected)));
            let diagnostics: Vec<_> = trial
                .diagnostics()
                .iter()
                .filter_map(|d| d.timing().cloned())
                .collect();
            assert_eq!(diagnostics.len(), 2, "{diagnostics:?}");
            for (diagnostic, kind) in diagnostics.iter().zip(["setup", "hold"]) {
                assert_eq!(
                    (diagnostic.kind, diagnostic.certainty),
                    (kind, Certainty::DefiniteViolation)
                );
                assert_eq!((diagnostic.elapsed, diagnostic.maximum_elapsed), (3., 3.));
            }
            trial.commit()?;
        }
        Ok(())
    }

    #[test]
    fn strength_only_events_preserve_clock_width_origin() -> SpiceResult<()> {
        let operation = Operation::from_value(15., 2, None)?;
        let mut spec = TimingSpec::default();
        spec.set_vector("timing_clock_min", vec![0., 6.])?;
        spec.set_policy(2.)?;
        let mut latch = model(15., 0., 0., 2.)?;
        latch.kernel.timing = spec.bind(operation, 2)?;
        let mut network = Network::default();
        let d = network.add_net();
        let g = network.add_net();
        let q = network.add_net();
        let data = network.add_source(d)?;
        let gate = network.add_source(g)?;
        network.add_model(latch, &[d, g], &[q])?;
        let mut engine = Engine::new(network, Limits::default())?;
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let mut trial = engine.trial();
        trial.schedule(data, 0., low)?;
        trial.schedule(gate, 0., low)?;
        trial.schedule(gate, 10., high)?;
        trial.schedule(
            gate,
            12.,
            Drive {
                logic: Logic::One,
                strength: Strength::RESISTIVE,
            },
        )?;
        trial.schedule(data, 14., high)?;
        trial.schedule(gate, 15., low)?;
        trial.advance_to(30.)?;
        let diagnostics = trial.diagnostics();
        assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
        let timing = diagnostics[0].timing().expect("width");
        assert_eq!(
            (timing.kind, timing.elapsed, timing.maximum_elapsed),
            ("clock/gate pulse width", 5., 5.)
        );
        trial.commit()
    }
}
