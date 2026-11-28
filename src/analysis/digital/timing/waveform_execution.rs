//! Execute guarded waveforms in physical event order. This layer shares
//! the generic symbolic queue between combinational resolution and
//! inertial deadline cancellation, rather than approximating either
//! with a union of possible levels.

use super::queue::{Scheduled, SymbolicQueue, execute};
use super::{
    Arrival, DelayWindow, DriveTrace, EventDelta, Guarded, GuardedWaveform, Origin, TimingWindow,
    WaveformRequest, charge,
};
use crate::devices::code_model::Drive;
use crate::primitives::{SpiceError, SpiceResult};

#[derive(Clone, Copy, Debug)]
struct PortRequest {
    port: usize,
    target: Drive,
    posting: usize,
}

#[derive(Clone, Debug)]
struct InputState {
    drives: Vec<Drive>,
    latest: Vec<Option<usize>>,
}

impl InputState {
    fn new(drives: Vec<Drive>) -> Self {
        Self {
            latest: vec![None; drives.len()],
            drives,
        }
    }

    fn deliver(&mut self, request: PortRequest) -> bool {
        let prior = &mut self.latest[request.port];
        if prior.is_some_and(|index| index > request.posting) {
            return false;
        }
        *prior = Some(request.posting);
        let old = std::mem::replace(&mut self.drives[request.port], request.target);
        old != request.target
    }
}

fn requests(traces: &[DriveTrace]) -> SpiceResult<SymbolicQueue<PortRequest>> {
    let mut queue = SymbolicQueue::default();
    for (port, trace) in traces.iter().enumerate() {
        for (posting, arrival) in trace.requests().iter().enumerate() {
            queue.post(
                arrival.arrival().window().clone(),
                arrival.delta().clone(),
                PortRequest {
                    port,
                    posting,
                    target: arrival.arrival().to(),
                },
            )?;
        }
    }
    Ok(queue)
}

#[derive(Clone, Debug)]
struct Combined {
    inputs: InputState,
    initial: Drive,
    output: Drive,
    transitions: Vec<WaveformRequest>,
}

impl Combined {
    fn new(traces: &[DriveTrace], function: &impl Fn(&[Drive]) -> Drive) -> Self {
        let inputs = InputState::new(traces.iter().map(DriveTrace::initial).collect());
        let initial = function(&inputs.drives);
        Self {
            inputs,
            initial,
            output: initial,
            transitions: Vec::new(),
        }
    }

    fn group(
        &mut self,
        time: &TimingWindow,
        delta: usize,
        events: &[Scheduled<PortRequest>],
        function: &impl Fn(&[Drive]) -> Drive,
    ) -> SpiceResult<()> {
        let mut changed = false;
        for event in events {
            changed |= self.inputs.deliver(event.payload);
        }
        if changed {
            let target = function(&self.inputs.drives);
            if target != self.output {
                self.transitions.push(WaveformRequest::new(
                    Arrival::new(self.output, target, time.clone())?,
                    EventDelta::Fixed(delta),
                ));
                self.output = target;
            }
        }
        Ok(())
    }

    fn trace(self) -> SpiceResult<DriveTrace> {
        DriveTrace::new(self.initial, self.transitions)
    }
}

/// Explicit deadline equality for an inertial propagation law.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeadlineEquality {
    /// A pending request commits if its input is removed exactly at
    /// the delivery deadline. A subsequent change starts a new timer.
    Commit,
    /// Removal at the exact deadline cancels the pending request.
    Cancel,
}

#[derive(Clone, Copy, Debug)]
enum InertialEvent {
    Input(PortRequest),
    Deadline(Drive),
}

#[derive(Clone, Debug)]
struct Inertial {
    input: InputState,
    initial: Drive,
    output: Drive,
    pending: Option<Drive>,
    transitions: Vec<WaveformRequest>,
}

impl Inertial {
    fn new(initial: Drive) -> Self {
        Self {
            input: InputState::new(vec![initial]),
            initial,
            output: initial,
            pending: None,
            transitions: Vec::new(),
        }
    }

    fn group(
        &mut self,
        time: &TimingWindow,
        delta: usize,
        events: &[Scheduled<InertialEvent>],
        queue: &mut SymbolicQueue<InertialEvent>,
        equality: DeadlineEquality,
        delay: &impl Fn(Drive, Drive) -> SpiceResult<DelayWindow>,
    ) -> SpiceResult<()> {
        let mut deadline = None;
        let mut changed = false;
        for event in events {
            match event.payload {
                InertialEvent::Input(request) => changed |= self.input.deliver(request),
                InertialEvent::Deadline(target) => {
                    if deadline.replace(target).is_some() {
                        return Err(SpiceError::circuit(
                            "multiple inertial deadlines in one branch",
                        ));
                    }
                }
            }
        }
        let desired = self.input.drives[0];
        if let Some(target) = deadline {
            if self.pending.take() != Some(target) {
                return Err(SpiceError::circuit(
                    "inertial deadline disagrees with pending target",
                ));
            }
            if (equality == DeadlineEquality::Commit || desired == target) && target != self.output
            {
                self.transitions.push(WaveformRequest::new(
                    Arrival::new(self.output, target, time.clone())?,
                    EventDelta::Fixed(delta),
                ));
                self.output = target;
            }
        }
        if !changed {
            return Ok(());
        }
        if self.pending == Some(desired) {
            return Ok(());
        }
        queue.cancel(|event| matches!(event, InertialEvent::Deadline(_)));
        self.pending = None;
        if desired != self.output {
            let arrival = time.after(delay(self.output, desired)?, Origin::independent()?)?;
            // A timer posted during this group cannot execute before
            // this group even if its delay is zero. At a later equal
            // physical time the chosen equality policy, not a guessed
            // minimum or maximum delay, determines cancellation.
            queue.post(
                arrival,
                EventDelta::Propagated {
                    cause: time.clone(),
                    parent_delta: delta,
                },
                InertialEvent::Deadline(desired),
            )?;
            self.pending = Some(desired);
        }
        Ok(())
    }

    fn trace(self) -> SpiceResult<DriveTrace> {
        if self.pending.is_some() {
            return Err(SpiceError::circuit(
                "unresolved inertial deadline after complete execution",
            ));
        }
        DriveTrace::new(self.initial, self.transitions)
    }
}

impl GuardedWaveform {
    /// Compose any number of guarded input waveforms through a pure
    /// combinational function, including strength-aware net resolution.
    /// The function receives an actual simultaneous group of stable
    /// drives in each feasible realization, not independent R/F or X
    /// approximations. The resulting transitions have zero added
    /// propagation delay; apply the selected transport/inertial law
    /// separately.
    ///
    /// # Errors
    /// Infeasible or invalid traces, exhausted exact execution budget,
    /// or invalid stable output from the function.
    pub fn combine(
        inputs: &[Self],
        mut budget: usize,
        function: impl Fn(&[Drive]) -> Drive,
    ) -> SpiceResult<Self> {
        Self::combine_with_budget(inputs, &mut budget, function)
    }

    fn combine_with_budget(
        inputs: &[Self],
        budget: &mut usize,
        function: impl Fn(&[Drive]) -> Drive,
    ) -> SpiceResult<Self> {
        let mut product = Guarded::unconditional(Vec::<DriveTrace>::new());
        for input in inputs {
            product = product.zip_with_budget(input.traces(), budget, |traces, trace| {
                let mut traces = traces.clone();
                traces.push(trace.clone());
                Ok(traces)
            })?;
        }
        let mut branches = Vec::new();
        for (guard, traces) in product.branches() {
            let queue = requests(traces)?;
            let initial = Combined::new(traces, &function);
            for (guard, state) in execute(
                guard.clone(),
                queue,
                initial,
                budget,
                |_, time, delta, events, _, state, _| state.group(time, delta, events, &function),
            )? {
                branches.push((guard, state.trace()?));
            }
        }
        Ok(Self::new(Guarded::from_partitioned(branches)?))
    }

    /// Apply a transport delay law to each actual desired transition.
    /// Each occurrence has its own delay origin; the input guard and
    /// causal expression are retained. Output requests remain in posting
    /// order so later delivery supersession is preserved.
    ///
    /// # Errors
    /// Invalid delay, arrival time or exhausted execution budget.
    pub fn transport(
        &self,
        mut budget: usize,
        delay: impl Fn(Drive, Drive) -> SpiceResult<DelayWindow>,
    ) -> SpiceResult<Self> {
        let actual =
            Self::combine_with_budget(std::slice::from_ref(self), &mut budget, |drives| drives[0])?;
        let traces = actual.traces().try_map(|trace| {
            let requests = trace
                .requests()
                .iter()
                .map(|request| {
                    charge(&mut budget, 1)?;
                    let arrival = request.arrival();
                    let EventDelta::Fixed(parent_delta) = request.delta() else {
                        return Err(SpiceError::circuit("transport cause has unresolved delta"));
                    };
                    let window = arrival
                        .window()
                        .after(delay(arrival.from(), arrival.to())?, Origin::independent()?)?;
                    Ok(WaveformRequest::new(
                        Arrival::new(arrival.from(), arrival.to(), window)?,
                        EventDelta::Propagated {
                            cause: arrival.window().clone(),
                            parent_delta: *parent_delta,
                        },
                    ))
                })
                .collect::<SpiceResult<Vec<_>>>()?;
            DriveTrace::new(trace.initial(), requests)
        })?;
        Ok(Self::new(traces))
    }

    /// Apply an inertial delay law by scheduling and cancelling actual
    /// symbolic deadlines within each feasible input branch. It does
    /// not decide pulse qualification using independent interval
    /// endpoints. Equality at the deadline is an explicit policy.
    ///
    /// # Errors
    /// Invalid delay, arrival time, or exhausted execution budget.
    pub fn inertial(
        &self,
        equality: DeadlineEquality,
        mut budget: usize,
        delay: impl Fn(Drive, Drive) -> SpiceResult<DelayWindow>,
    ) -> SpiceResult<Self> {
        let mut branches = Vec::new();
        for (guard, trace) in self.traces().branches() {
            let mut queue = SymbolicQueue::default();
            for (posting, request) in trace.requests().iter().enumerate() {
                charge(&mut budget, 1)?;
                queue.post(
                    request.arrival().window().clone(),
                    request.delta().clone(),
                    InertialEvent::Input(PortRequest {
                        port: 0,
                        target: request.arrival().to(),
                        posting,
                    }),
                )?;
            }
            for (guard, state) in execute(
                guard.clone(),
                queue,
                Inertial::new(trace.initial()),
                &mut budget,
                |_, time, delta, events, queue, state, _| {
                    state.group(time, delta, events, queue, equality, &delay)
                },
            )? {
                branches.push((guard, state.trace()?));
            }
        }
        Ok(Self::new(Guarded::from_partitioned(branches)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::digital::timing::TimingRelation;
    use crate::devices::code_model::{Logic, Strength};
    use crate::primitives::Real;
    use std::collections::BTreeMap;

    #[test]
    fn exact_reconvergence_cancels_optional_pulse_through_xor() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let start = TimingWindow::exact(10.)?.after(DelayWindow::new(0., 10.)?, u)?;
        let end = TimingWindow::exact(15.)?;
        let input = GuardedWaveform::new(Guarded::unconditional(DriveTrace::source(
            low,
            vec![
                Arrival::new(low, high, start)?,
                Arrival::new(high, low, end)?,
            ],
        )?));
        let result = GuardedWaveform::combine(&[input.clone(), input], 1_000_000, |drives| {
            Drive::strong(if drives[0].logic == drives[1].logic {
                Logic::Zero
            } else {
                Logic::One
            })
        })?;
        assert_eq!(result.project(1_000_000)?.changes().len(), 1);
        assert!(
            result
                .traces()
                .branches()
                .all(|(_, trace)| trace.requests().is_empty())
        );
        Ok(())
    }

    #[test]
    fn symbolic_resolution_retains_actual_rank_not_independent_unknowns() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let weak = Drive {
            logic: Logic::Zero,
            strength: Strength::RESISTIVE,
        };
        let strong = Drive::strong(Logic::One);
        let disconnected = Drive::default();
        let time = TimingWindow::exact(10.)?.after(DelayWindow::new(0., 10.)?, u)?;
        let a = GuardedWaveform::new(Guarded::unconditional(DriveTrace::source(
            strong,
            vec![Arrival::new(strong, disconnected, time.clone())?],
        )?));
        let b = GuardedWaveform::new(Guarded::unconditional(DriveTrace::source(
            disconnected,
            vec![Arrival::new(disconnected, strong, time)?],
        )?));
        let pull = GuardedWaveform::new(Guarded::unconditional(DriveTrace::source(
            weak,
            Vec::new(),
        )?));
        let result = GuardedWaveform::combine(&[a, b, pull], 1_000_000, |drives| {
            Drive::resolve(drives.iter().copied())
        })?;
        assert_eq!(result.project(1_000_000)?.possible_at(15.)?, [strong]);
        Ok(())
    }

    #[test]
    fn guarded_inertial_pulse_uses_actual_relative_width_and_equality() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let v = Origin::shared(1);
        let low = Drive::strong(Logic::Zero);
        let high = Drive::strong(Logic::One);
        let start = TimingWindow::exact(10.)?.after(DelayWindow::new(0., 10.)?, u)?;
        let finish = start.after(DelayWindow::new(1., 3.)?, v)?;
        let waveform = GuardedWaveform::new(Guarded::unconditional(DriveTrace::source(
            low,
            vec![
                Arrival::new(low, high, start)?,
                Arrival::new(high, low, finish)?,
            ],
        )?));
        for equality in [DeadlineEquality::Commit, DeadlineEquality::Cancel] {
            let output = waveform.inertial(equality, 1_000_000, |_, _| DelayWindow::new(2., 2.))?;
            for fraction in [0., 0.5, 1.] {
                let origins = BTreeMap::<Origin, Real>::from([(u, 0.5), (v, fraction)]);
                let high_at_deadline =
                    fraction > 0.5 || (fraction == 0.5 && equality == DeadlineEquality::Commit);
                assert_eq!(
                    output.realize_at(&origins, 17., 100_000)?,
                    if high_at_deadline { high } else { low }
                );
                assert_eq!(output.realize_at(&origins, 30., 100_000)?, low);
            }
            let projected = output.project(1_000_000)?;
            assert!(projected.possible_at(18.)?.contains(&high));
            assert_eq!(projected.possible_at(25.)?, [low]);
        }
        // The exact deadline equality is an actual branch, not a
        // rounded interval comparison.
        let root = TimingWindow::exact(0.)?;
        assert_ne!(
            super::super::TimingGuard::compare(root.clone(), root, TimingRelation::Coincident)
                .status(100_000)?,
            super::super::GuardStatus::Impossible
        );
        Ok(())
    }
}
