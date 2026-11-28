//! Transport propagation of guaranteed causal transitions through gates.
//! Keep phase boundaries attached to the same transition, not two
//! independently delayed scalar events.

use super::causal_history::{COMPARISON_BUDGET, ordered_groups};
use super::source_logic::DriveStrengths;
use super::{
    CodeModel, DelayMode, Delays, Drive, EventContext, GateKind, Logic, ModelSchema, Output,
    Strength,
};
use crate::analysis::digital::timing::{
    Arrival, DelayWindow, Origin, TimedInput, TimedOutput, TimingWindow,
};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Directional inclusive propagation intervals.
#[derive(Clone, Copy, Debug)]
pub struct WindowDelays(WindowDelayLaw);

#[derive(Clone, Copy, Debug)]
enum WindowDelayLaw {
    Exact(Delays),
    Bounded {
        rise: DelayWindow,
        fall: DelayWindow,
        impedance: Option<[DelayWindow; 4]>,
    },
}

impl WindowDelays {
    /// Binary rise and fall intervals.
    #[must_use = "the intervals define the propagation law"]
    pub const fn binary(rise: DelayWindow, fall: DelayWindow) -> Self {
        Self(WindowDelayLaw::Bounded {
            rise,
            fall,
            impedance: None,
        })
    }

    /// LH, HL, LZ, HZ, ZL, ZH intervals.
    #[must_use = "the intervals define the directional propagation law"]
    pub const fn paths(paths: [DelayWindow; 6]) -> Self {
        let [rise, fall, lz, hz, zl, zh] = paths;
        Self(WindowDelayLaw::Bounded {
            rise,
            fall,
            impedance: Some([lz, hz, zl, zh]),
        })
    }

    /// Transport an upstream arrival using the declared scalar propagation
    /// law, including its exact unknown-state and impedance selection.
    pub(super) const fn exact(delays: Delays) -> Self {
        Self(WindowDelayLaw::Exact(delays))
    }

    fn envelope(a: DelayWindow, b: DelayWindow) -> SpiceResult<DelayWindow> {
        DelayWindow::new(
            a.bounds().minimum().min(b.bounds().minimum()),
            a.bounds().maximum().max(b.bounds().maximum()),
        )
    }

    pub(super) fn transition(self, from: Drive, to: Drive) -> SpiceResult<DelayWindow> {
        let (rise, fall, impedance) = match self.0 {
            WindowDelayLaw::Exact(delays) => {
                let d = delays.transition(from, to);
                return DelayWindow::new(d, d);
            }
            WindowDelayLaw::Bounded {
                rise,
                fall,
                impedance,
            } => (rise, fall, impedance),
        };
        if let Some([lz, hz, zl, zh]) = impedance {
            match (
                from.strength == Strength::HIGH_IMPEDANCE,
                to.strength == Strength::HIGH_IMPEDANCE,
            ) {
                (true, true) => return DelayWindow::new(0., 0.),
                (false, true) => {
                    return match from.logic {
                        Logic::Zero => Ok(lz),
                        Logic::One => Ok(hz),
                        _ => Self::envelope(lz, hz),
                    };
                }
                (true, false) => {
                    return match to.logic {
                        Logic::Zero => Ok(zl),
                        Logic::One => Ok(zh),
                        _ => Self::envelope(zl, zh),
                    };
                }
                _ => {}
            }
        }
        match to.logic {
            Logic::One => Ok(rise),
            Logic::Zero => Ok(fall),
            _ => match from.logic {
                Logic::Zero => Ok(rise),
                Logic::One => Ok(fall),
                _ => Self::envelope(rise, fall),
            },
        }
    }

    pub(super) fn with_load(self, load: [Real; 3]) -> SpiceResult<Self> {
        let (rise, fall, impedance) = match self.0 {
            WindowDelayLaw::Exact(delays) => return Ok(Self::exact(delays.with_load(load)?)),
            WindowDelayLaw::Bounded {
                rise,
                fall,
                impedance,
            } => (rise, fall, impedance),
        };
        let add = |window: DelayWindow, extra: Real| {
            let b = window.bounds();
            let endpoint = |value| if value == 0. { 0. } else { value + extra };
            DelayWindow::new(endpoint(b.minimum()), endpoint(b.maximum()))
        };
        let rise = add(rise, load[1])?;
        let fall = add(fall, load[0])?;
        let impedance = impedance
            .map(|[lz, hz, zl, zh]| -> SpiceResult<_> {
                Ok([
                    add(lz, load[2])?,
                    add(hz, load[2])?,
                    add(zl, load[0])?,
                    add(zh, load[1])?,
                ])
            })
            .transpose()?;
        Ok(Self(WindowDelayLaw::Bounded {
            rise,
            fall,
            impedance,
        }))
    }
}

/// Per-cause propagation intervals. Omitted paths inherit the primary
/// table or the corresponding explicitly declared exact path.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct WindowPaths {
    pub(super) gate: Option<WindowDelays>,
    pub(super) preset: Option<WindowDelays>,
    pub(super) clear: Option<WindowDelays>,
}

impl WindowPaths {
    pub(super) const fn any(self) -> bool {
        self.gate.is_some() || self.preset.is_some() || self.clear.is_some()
    }

    pub(super) fn with_load(self, load: [Real; 3]) -> SpiceResult<Self> {
        let apply = |path: Option<WindowDelays>| path.map(|d| d.with_load(load)).transpose();
        Ok(Self {
            gate: apply(self.gate)?,
            preset: apply(self.preset)?,
            clear: apply(self.clear)?,
        })
    }
}
/// A combinational gate transporting causal arrival expressions. A
/// simultaneous group of monotone independent input transitions uses
/// the appropriate earliest/latest join. Shared-origin simultaneous
/// changes are evaluated as one correlated transition.
#[derive(Debug)]
pub struct WindowGate {
    function: GateFunction,
    inputs: usize,
    delays: WindowDelays,
    strength: DriveStrengths,
    invert: Vec<bool>,
}

#[derive(Clone, Copy, Debug)]
enum GateFunction {
    Boolean(GateKind),
    TriState { active: Logic },
}

/// Disposable state of a causal gate.
#[derive(Clone, Debug)]
pub struct WindowGateState {
    requested: Option<Drive>,
    inputs: Vec<Drive>,
    arrivals: Vec<Option<Arrival>>,
    last_output: Option<Arrival>,
}

impl WindowGate {
    /// Construct a strongly driven transport gate with causal intervals.
    ///
    /// # Errors
    /// Invalid gate arity.
    pub fn new(kind: GateKind, inputs: usize, delays: WindowDelays) -> SpiceResult<Self> {
        if inputs == 0 || (matches!(kind, GateKind::Buffer | GateKind::Inverter) && inputs != 1) {
            return Err(SpiceError::circuit("invalid causal gate arity"));
        }
        Ok(Self {
            function: GateFunction::Boolean(kind),
            inputs,
            delays,
            strength: DriveStrengths::uniform(Strength::STRONG),
            invert: vec![false; inputs],
        })
    }

    pub(super) fn tristate(active: Logic, delays: WindowDelays) -> SpiceResult<Self> {
        if !active.is_binary() {
            return Err(SpiceError::circuit("invalid causal tri-state polarity"));
        }
        Ok(Self {
            function: GateFunction::TriState { active },
            inputs: 2,
            delays,
            strength: DriveStrengths::uniform(Strength::STRONG),
            invert: vec![false; 2],
        })
    }

    pub(super) fn with_binding(mut self, strength: DriveStrengths, invert: Vec<bool>) -> Self {
        self.strength = strength;
        self.invert = invert;
        self
    }

    fn drive(&self, inputs: &[Drive]) -> Drive {
        let inputs: Vec<_> = inputs
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
            .collect();
        match self.function {
            GateFunction::Boolean(kind) => self.strength.drive(kind.evaluate(&inputs)),
            GateFunction::TriState { active } => {
                if inputs[1].logic == active {
                    self.strength.drive(inputs[0].logic)
                } else if inputs[1].logic.is_binary() {
                    Drive::default()
                } else {
                    self.strength.drive(Logic::Unknown)
                }
            }
        }
    }

    fn initial(
        &self,
        state: &mut WindowGateState,
        inputs: &[TimedInput],
    ) -> SpiceResult<Vec<TimedOutput>> {
        state.inputs = inputs.iter().map(|i| i.drive).collect();
        state.arrivals = inputs.iter().map(|i| i.arrival.clone()).collect();
        let drive = self.drive(&state.inputs);
        state.requested = Some(drive);
        state.last_output = None;
        Ok(vec![TimedOutput::Scalar(Output {
            drive,
            delays: Delays::new(0., 0.)?,
            mode: DelayMode::Transport,
        })])
    }

    fn causal_window(
        &self,
        context: EventContext,
        transitions: &[(usize, Arrival)],
        target: Drive,
        state: &WindowGateState,
    ) -> SpiceResult<TimingWindow> {
        let Some((_, first)) = transitions.first() else {
            return TimingWindow::exact(context.time);
        };
        let same = transitions.iter().all(|(_, a)| {
            a.window().same_expression(first.window())
                || (a.window().bounds().is_exact()
                    && first.window().bounds().is_exact()
                    && a.window().bounds() == first.window().bounds())
        });
        if same {
            return Ok(first.window().clone());
        }
        let GateFunction::Boolean(kind) = self.function else {
            return Err(SpiceError::circuit(
                "independently ordered tri-state transitions require a conditional waveform",
            ));
        };
        let base = match kind {
            GateKind::And | GateKind::Nand => GateKind::And,
            GateKind::Or | GateKind::Nor => GateKind::Or,
            _ => {
                return Err(SpiceError::circuit(
                    "independent overlapping nonmonotone gate arrivals require a hazard model",
                ));
            }
        };
        let raw_target = if matches!(kind, GateKind::Nand | GateKind::Nor) {
            target.logic.invert()
        } else {
            target.logic
        };
        let arrival_direction = |index: usize, logic: Logic| {
            if self.invert[index] {
                logic.invert()
            } else {
                logic
            }
        };
        // Include only changing inputs that affect the monotone output;
        // a transition in the opposite direction may introduce a hazard.
        if !raw_target.is_binary()
            || transitions.iter().any(|(index, a)| {
                let from = arrival_direction(*index, a.from().logic);
                let to = arrival_direction(*index, a.to().logic);
                !from.is_binary() || to != raw_target || from == to
            })
            || state.requested.is_none_or(|d| !d.logic.is_binary())
        {
            return Err(SpiceError::circuit(
                "mixed-direction causal gate arrivals require a hazard model",
            ));
        }
        let arrivals = transitions
            .iter()
            .map(|(_, a)| a.window().clone())
            .collect();
        match (base, raw_target) {
            (GateKind::And, Logic::One) | (GateKind::Or, Logic::Zero) => {
                TimingWindow::latest(arrivals)
            }
            _ => TimingWindow::earliest(arrivals),
        }
    }
}

fn strictly_after(new: &[(usize, Arrival)], old: &[(usize, Arrival)]) -> SpiceResult<bool> {
    for (_, next) in new {
        for (_, prior) in old {
            if next
                .window()
                .relative_to(prior.window(), COMPARISON_BUDGET)?
                .minimum()
                <= 0.
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

impl WindowGate {
    fn ordered_waveform(
        &self,
        state: &WindowGateState,
        transitions: &[(usize, Arrival)],
        previous: Drive,
        groups: Vec<Vec<usize>>,
    ) -> SpiceResult<Vec<Arrival>> {
        let mut drives = state.inputs.clone();
        let mut prior = previous;
        let mut waveform = Vec::new();
        for group in groups {
            let cause = transitions[group[0]].1.window();
            for index in group {
                let (port, arrival) = &transitions[index];
                drives[*port] = arrival.to();
            }
            let target = self.drive(&drives);
            if target != prior {
                let delay = self.delays.transition(prior, target)?;
                let window = cause.after(delay, Origin::independent()?)?;
                waveform.push(Arrival::new(prior, target, window)?);
                prior = target;
            }
        }
        Ok(waveform)
    }

    fn remember(
        state: &mut WindowGateState,
        inputs: &[TimedInput],
        next: Vec<Drive>,
        target: Drive,
    ) {
        state.inputs = next;
        state.arrivals = inputs.iter().map(|i| i.arrival.clone()).collect();
        state.requested = Some(target);
    }
}

impl CodeModel for WindowGate {
    type State = WindowGateState;

    fn schema(&self) -> ModelSchema {
        ModelSchema::digital(self.inputs, 1)
    }

    fn initial_state(&self) -> Self::State {
        WindowGateState {
            requested: None,
            inputs: vec![Drive::default(); self.inputs],
            arrivals: vec![None; self.inputs],
            last_output: None,
        }
    }

    fn tracks_arrivals(&self) -> bool {
        true
    }

    fn evaluate(
        &self,
        _: EventContext,
        _: &[Drive],
        _: &mut Self::State,
    ) -> SpiceResult<Vec<Output>> {
        Err(SpiceError::circuit(
            "causal gate requires arrival-aware event engine",
        ))
    }

    fn evaluate_timed(
        &self,
        context: EventContext,
        inputs: &[TimedInput],
        state: &mut Self::State,
    ) -> SpiceResult<Vec<TimedOutput>> {
        if inputs.len() != self.inputs {
            return Err(SpiceError::circuit("causal gate input dimension mismatch"));
        }
        if context.initial {
            return self.initial(state, inputs);
        }
        let mut next = state.inputs.clone();
        let mut transitions = Vec::new();
        let mut old_active = Vec::new();
        for (index, input) in inputs.iter().enumerate() {
            if let Some(arrival) = &input.arrival {
                if state.arrivals[index]
                    .as_ref()
                    .is_some_and(|old| old.same_transition(arrival))
                {
                    if arrival.active_at(context.time) {
                        old_active.push((index, arrival.clone()));
                    }
                    continue;
                }
                if arrival.from() != state.inputs[index] {
                    return Err(SpiceError::circuit(
                        "causal gate input endpoints disagree with preceding transition",
                    ));
                }
                next[index] = arrival.to();
                transitions.push((index, arrival.clone()));
            } else if input.drive != state.inputs[index] {
                return Err(SpiceError::circuit(
                    "causal gate input lacks qualified transition provenance",
                ));
            }
        }
        if transitions.is_empty() {
            return Ok(vec![TimedOutput::Retain]);
        }
        let target = self.drive(&next);
        let previous = state
            .requested
            .ok_or_else(|| SpiceError::circuit("causal gate has no initial output"))?;
        // A physically later announcement can follow a still-active
        // earlier envelope. Keep both transitions instead of refusing
        // overlap or treating the old boundary as another edge.
        if (old_active.is_empty() || strictly_after(&transitions, &old_active)?)
            && let Some(groups) = ordered_groups(
                transitions
                    .iter()
                    .enumerate()
                    .map(|(i, (_, a))| (i, a.window())),
            )?
        {
            let waveform = self.ordered_waveform(state, &transitions, previous, groups)?;
            if let Some(last) = waveform.last() {
                state.last_output = Some(last.clone());
            }
            Self::remember(state, inputs, next, target);
            return Ok(vec![if waveform.is_empty() {
                TimedOutput::Retain
            } else {
                TimedOutput::Waveform(waveform)
            }]);
        }
        let instantaneous: Vec<_> = inputs.iter().map(|i| i.drive).collect();
        let current = self.drive(&instantaneous);
        if !old_active.is_empty() {
            // A previously masked transition can become relevant when
            // another input changes. Join its original arrival rather
            // than substituting the current envelope boundary.
            let output_pending = state
                .last_output
                .as_ref()
                .is_some_and(|a| a.window().bounds().maximum() > context.time);
            if output_pending {
                return Err(SpiceError::circuit(
                    "overlapping causal gate evaluations require waveform refinement",
                ));
            }
            if target != previous {
                let mut before = state.inputs.clone();
                for (index, arrival) in &old_active {
                    before[*index] = arrival.from();
                }
                if self.drive(&before) != previous {
                    return Err(SpiceError::circuit(
                        "causal gate has an unresolved preceding hazard",
                    ));
                }
                transitions.extend(old_active);
            } else if current != target || !target.logic.is_binary() {
                return Err(SpiceError::circuit(
                    "overlapping causal gate evaluations require a hazard model",
                ));
            }
        }
        let result = if target == previous {
            if current != target
                && transitions
                    .iter()
                    .any(|(_, a)| !a.window().bounds().is_exact())
            {
                let first = &transitions[0].1;
                if transitions
                    .iter()
                    .any(|(_, a)| !a.window().same_expression(first.window()))
                {
                    return Err(SpiceError::circuit(
                        "causal gate may produce an overlapping-input hazard",
                    ));
                }
            }
            TimedOutput::Retain
        } else {
            let cause = self.causal_window(context, &transitions, target, state)?;
            let delay = self.delays.transition(previous, target)?;
            let window = cause.after(delay, Origin::independent()?)?;
            let arrival = Arrival::new(previous, target, window)?;
            state.last_output = Some(arrival.clone());
            TimedOutput::Arrival(arrival)
        };
        Self::remember(state, inputs, next, target);
        Ok(vec![result])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_window_paths_preserve_scalar_selection_for_every_drive_pair() -> SpiceResult<()> {
        // In particular, actual X->Z and Z->X select the later scalar
        // impedance path, not a new bounded interval.
        let drives = [
            Drive::strong(Logic::Zero),
            Drive::strong(Logic::One),
            Drive::strong(Logic::Unknown),
            Drive::strong(Logic::Rising),
            Drive::strong(Logic::Falling),
            Drive::default(),
        ];
        for scalar in [
            Delays::new(1., 2.)?,
            Delays::paths([1., 2., 3., 4., 5., 6.])?,
        ] {
            let loaded = scalar.with_load([0.25, 0.5, 0.75])?;
            let windows = WindowDelays::exact(scalar).with_load([0.25, 0.5, 0.75])?;
            for from in drives {
                for to in drives {
                    let bounds = windows.transition(from, to)?.bounds();
                    let expected = loaded.transition(from, to);
                    assert_eq!((bounds.minimum(), bounds.maximum()), (expected, expected));
                }
            }
        }
        Ok(())
    }
}
