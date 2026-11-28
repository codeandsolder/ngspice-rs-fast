//! Deterministic digital event transactions, independent of analog Newton
//! iterations. See upstream XSPICE evtqueue.c, evtiter.c and evtbackup.c.

use std::collections::{BTreeMap, BTreeSet};
mod diagnostics;
pub mod timing;
use diagnostics::DiagnosticLog;
pub use diagnostics::{DiagnosticDetail, EventDiagnostic};
use timing::{Arrival, TimedInput, TimedOutput, TimingWindow};

use crate::devices::code_model::{
    AutonomousAdvance, CodeModel, DelayMode, Drive, EventContext, ModelDiagnostic, Output,
};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Digital net identifier, distinct from an analog MNA node.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct NetId(usize);

/// External event source identifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceId(usize);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Time(u64);

impl Time {
    fn new(time: Real) -> SpiceResult<Self> {
        if !time.is_finite() || time < 0. {
            return Err(SpiceError::circuit("invalid digital event time"));
        }
        Ok(Self(if time == 0. { 0 } else { time.to_bits() }))
    }

    const fn value(self) -> Real {
        Real::from_bits(self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Key {
    time: Time,
    delta: usize,
    sequence: u64,
}

/// Handle for cancelling a queued external event.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventId {
    key: Key,
    completion: Option<Key>,
    identity: u64,
}

fn external_identity() -> SpiceResult<u64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| SpiceError::circuit("external event identity space exhausted"))
}

#[derive(Clone, Debug)]
enum Event {
    Drive {
        driver: usize,
        drive: Drive,
        external: Option<u64>,
        arrival: Option<(Arrival, bool)>,
    },
    Wake(usize),
}

#[derive(Debug, Default)]
struct Net {
    margin: Option<std::num::NonZeroU8>,
    drivers: Vec<usize>,
    subscribers: Vec<usize>,
}

trait ModelState: std::fmt::Debug + std::any::Any {
    fn clone_state(&self) -> Box<dyn ModelState>;
}

impl<S: Clone + std::fmt::Debug + 'static> ModelState for S {
    fn clone_state(&self) -> Box<dyn ModelState> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn ModelState> {
    fn clone(&self) -> Self {
        self.as_ref().clone_state()
    }
}

trait ErasedModel: std::fmt::Debug {
    fn initial_state(&self) -> Box<dyn ModelState>;
    fn next_wakeup(&self, state: &dyn ModelState) -> SpiceResult<Option<Real>>;
    fn advance_autonomous(
        &self,
        now: Real,
        horizon: Real,
        state: &mut dyn ModelState,
    ) -> SpiceResult<AutonomousAdvance>;
    fn evaluate(
        &self,
        context: EventContext,
        inputs: &[Drive],
        state: &mut dyn ModelState,
    ) -> SpiceResult<(Vec<Output>, Vec<ModelDiagnostic>)>;
    fn evaluate_timed(
        &self,
        context: EventContext,
        inputs: &[TimedInput],
        state: &mut dyn ModelState,
    ) -> SpiceResult<(Vec<TimedOutput>, Vec<ModelDiagnostic>)>;
}

impl<M: CodeModel> ErasedModel for M {
    fn initial_state(&self) -> Box<dyn ModelState> {
        Box::new(CodeModel::initial_state(self))
    }

    fn next_wakeup(&self, state: &dyn ModelState) -> SpiceResult<Option<Real>> {
        let state = (state as &dyn std::any::Any)
            .downcast_ref::<M::State>()
            .ok_or_else(|| SpiceError::circuit("digital model state identity mismatch"))?;
        Ok(CodeModel::next_wakeup(self, state))
    }

    fn advance_autonomous(
        &self,
        now: Real,
        horizon: Real,
        state: &mut dyn ModelState,
    ) -> SpiceResult<AutonomousAdvance> {
        let state = (state as &mut dyn std::any::Any)
            .downcast_mut::<M::State>()
            .ok_or_else(|| SpiceError::circuit("digital model state identity mismatch"))?;
        CodeModel::advance_autonomous(self, now, horizon, state)
    }

    fn evaluate(
        &self,
        context: EventContext,
        inputs: &[Drive],
        state: &mut dyn ModelState,
    ) -> SpiceResult<(Vec<Output>, Vec<ModelDiagnostic>)> {
        let state = (state as &mut dyn std::any::Any)
            .downcast_mut::<M::State>()
            .ok_or_else(|| SpiceError::circuit("digital model state identity mismatch"))?;
        let outputs = CodeModel::evaluate(self, context, inputs, state)?;
        Ok((outputs, CodeModel::take_diagnostics(self, state)))
    }
    fn evaluate_timed(
        &self,
        context: EventContext,
        inputs: &[TimedInput],
        state: &mut dyn ModelState,
    ) -> SpiceResult<(Vec<TimedOutput>, Vec<ModelDiagnostic>)> {
        let state = (state as &mut dyn std::any::Any)
            .downcast_mut::<M::State>()
            .ok_or_else(|| SpiceError::circuit("digital model state identity mismatch"))?;
        let outputs = CodeModel::evaluate_timed(self, context, inputs, state)?;
        Ok((outputs, CodeModel::take_diagnostics(self, state)))
    }
}

#[derive(Debug)]
struct Model {
    name: std::sync::Arc<str>,
    kernel: Box<dyn ErasedModel>,
    inputs: Vec<NetId>,
    outputs: Vec<usize>,
}

/// Build a fixed digital network before beginning simulation.
#[derive(Debug, Default)]
pub struct Network {
    nets: Vec<Net>,
    drivers: Vec<NetId>,
    sources: Vec<usize>,
    models: Vec<Model>,
    lookahead: Vec<usize>,
    track_arrivals: bool,
}

impl Network {
    /// Add an initially undriven net.
    pub fn add_net(&mut self) -> NetId {
        let id = NetId(self.nets.len());
        self.nets.push(Net::default());
        id
    }

    pub(crate) const fn net_count(&self) -> usize {
        self.nets.len()
    }

    fn check_net(&self, id: NetId) -> SpiceResult<()> {
        if id.0 >= self.nets.len() {
            return Err(SpiceError::circuit("unknown digital net"));
        }
        Ok(())
    }

    /// Declare the minimum rank difference required to override a
    /// conflicting connected driver. Repeated declarations must agree.
    ///
    /// # Errors
    /// Unknown net, inconsistent declaration, or rank margin above 63.
    pub fn set_override_margin(
        &mut self,
        net: NetId,
        margin: std::num::NonZeroU8,
    ) -> SpiceResult<()> {
        self.check_net(net)?;
        let slot = &mut self.nets[net.0].margin;
        if margin.get() > 63 || slot.is_some_and(|prior| prior != margin) {
            return Err(SpiceError::circuit(
                "inconsistent or invalid digital net strength margin",
            ));
        }
        *slot = Some(margin);
        Ok(())
    }

    fn add_driver(&mut self, net: NetId) -> usize {
        let driver = self.drivers.len();
        self.drivers.push(net);
        self.nets[net.0].drivers.push(driver);
        driver
    }

    /// Add an externally scheduled driver (e.g. an ADC output).
    ///
    /// # Errors
    /// Unknown net.
    pub fn add_source(&mut self, net: NetId) -> SpiceResult<SourceId> {
        self.check_net(net)?;
        let source = SourceId(self.sources.len());
        let driver = self.add_driver(net);
        self.sources.push(driver);
        Ok(source)
    }

    /// Bind a native model's ordered ports. Multiple outputs may drive the
    /// same net, and a model may subscribe to its own output.
    ///
    /// # Errors
    /// Unknown nets or port-count mismatch.
    pub fn add_model<M: CodeModel>(
        &mut self,
        model: M,
        inputs: &[NetId],
        outputs: &[NetId],
    ) -> SpiceResult<()> {
        self.add_named_model(
            &format!("model#{}", self.models.len()),
            model,
            inputs,
            outputs,
        )
    }

    /// Bind a named instance for timing diagnostics, using the same port
    /// and resource checks as an ordinary event model.
    ///
    /// # Errors
    /// Unknown nets or port-count mismatch.
    pub fn add_named_model<M: CodeModel>(
        &mut self,
        name: &str,
        model: M,
        inputs: &[NetId],
        outputs: &[NetId],
    ) -> SpiceResult<()> {
        if model.schema()
            != crate::devices::code_model::ModelSchema::digital(inputs.len(), outputs.len())
        {
            return Err(SpiceError::circuit("digital model port count mismatch"));
        }
        for &net in inputs.iter().chain(outputs) {
            self.check_net(net)?;
        }
        let index = self.models.len();
        self.track_arrivals |= model.tracks_arrivals();
        if model.requires_lookahead() {
            self.lookahead.push(index);
        }
        for &net in inputs {
            self.nets[net.0].subscribers.push(index);
        }
        let drivers = outputs.iter().map(|&net| self.add_driver(net)).collect();
        self.models.push(Model {
            name: name.into(),
            kernel: Box::new(model),
            inputs: inputs.to_vec(),
            outputs: drivers,
        });
        Ok(())
    }
}

/// Bounds on event work, independent of analog step/Newton limits.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Maximum delta cycles at a single physical time.
    pub delta_cycles: usize,
    /// Maximum delivered events plus model evaluations in one trial advance.
    pub work: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            delta_cycles: 1024,
            work: 1_000_000,
        }
    }
}

/// Event work counters; discarded trial counters are not accepted.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Stats {
    /// Delivered output events.
    pub events: usize,
    /// Invocations of event models.
    pub evaluations: usize,
    /// Net-resolution rounds.
    pub rounds: usize,
    /// Autonomous horizon probes of external simulators.
    pub probes: usize,
    /// Timing diagnostics committed during model evaluation.
    pub diagnostics: usize,
}

/// Resolved net change after all output events at a time/delta are applied.
#[derive(Clone, Debug)]
pub struct Change {
    /// Physical time.
    pub time: Real,
    /// Delta cycle.
    pub delta: usize,
    /// Net that changed.
    pub net: NetId,
    /// Resolved value.
    pub drive: Drive,
    /// Causal transition behind this boundary, if tracked.
    pub arrival: Option<Arrival>,
}

#[derive(Clone, Debug)]
struct Prediction {
    time: Time,
    state: Box<dyn ModelState>,
    outputs: Option<Vec<Output>>,
}

struct Group {
    time: Time,
    delta: usize,
}

#[derive(Clone, Debug)]
struct Arrivals {
    drivers: Vec<Option<Arrival>>,
    nets: Vec<Option<Arrival>>,
}

impl Arrivals {
    fn new(drivers: usize, nets: usize) -> Self {
        Self {
            drivers: vec![None; drivers],
            nets: vec![None; nets],
        }
    }
}

fn same_arrival(a: Option<&Arrival>, b: Option<&Arrival>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.same_transition(b),
        (None, None) => true,
        _ => false,
    }
}

fn exact_arrival(from: Drive, to: Drive, time: Real) -> SpiceResult<Option<Arrival>> {
    use crate::devices::code_model::Logic;
    if from == to
        || [from, to]
            .iter()
            .any(|d| matches!(d.logic, Logic::Rising | Logic::Falling))
    {
        return Ok(None);
    }
    Ok(Some(Arrival::new(from, to, TimingWindow::exact(time)?)?))
}

#[derive(Clone, Debug)]
struct State {
    time: Time,
    delta: usize,
    initialized: bool,
    sequence: u64,
    queue: BTreeMap<Key, Event>,
    drivers: Vec<Drive>,
    desired: Vec<Option<Drive>>,
    pending: Vec<BTreeSet<Key>>,
    wakeups: Vec<Option<Key>>,
    nets: Vec<Drive>,
    arrivals: Option<Arrivals>,
    models: Vec<Box<dyn ModelState>>,
    predictions: Vec<Option<Prediction>>,
    stats: Stats,
    diagnostics: DiagnosticLog,
    first_fatal: Option<EventDiagnostic>,
}

impl State {
    fn new(network: &Network) -> Self {
        Self {
            time: Time(0),
            delta: 0,
            initialized: false,
            sequence: 0,
            queue: BTreeMap::new(),
            drivers: vec![Drive::default(); network.drivers.len()],
            desired: vec![None; network.drivers.len()],
            pending: vec![BTreeSet::new(); network.drivers.len()],
            wakeups: vec![None; network.models.len()],
            nets: vec![Drive::default(); network.nets.len()],
            arrivals: network
                .track_arrivals
                .then(|| Arrivals::new(network.drivers.len(), network.nets.len())),
            models: network
                .models
                .iter()
                .map(|m| m.kernel.initial_state())
                .collect(),
            predictions: vec![None; network.models.len()],
            stats: Stats::default(),
            diagnostics: DiagnosticLog::default(),
            first_fatal: None,
        }
    }

    fn schedule(
        &mut self,
        time: Time,
        driver: usize,
        drive: Drive,
        external: Option<u64>,
    ) -> SpiceResult<Key> {
        self.schedule_event(
            time,
            Event::Drive {
                driver,
                drive,
                external,
                arrival: None,
            },
        )
    }

    fn enable_arrivals(&mut self) {
        if self.arrivals.is_none() {
            self.arrivals = Some(Arrivals::new(self.drivers.len(), self.nets.len()));
        }
    }

    fn schedule_arrival(
        &mut self,
        driver: usize,
        arrival: Arrival,
        external: Option<u64>,
    ) -> SpiceResult<(Key, Option<Key>)> {
        let bounds = arrival.window().bounds();
        let start = Time::new(bounds.minimum())?;
        let end = Time::new(bounds.maximum())?;
        if start == Time(0) && start != end {
            return Err(SpiceError::circuit(
                "causal uncertainty starts before initial settlement",
            ));
        }
        if start < self.time
            || self
                .sequence
                .checked_add(if start == end { 1 } else { 2 })
                .is_none()
            || (start == self.time && self.initialized && self.delta == usize::MAX)
        {
            return Err(SpiceError::circuit(
                "noncausal or unrepresentable arrival boundaries",
            ));
        }
        self.enable_arrivals();
        let key = self.schedule_event(
            start,
            Event::Drive {
                driver,
                drive: if start == end {
                    arrival.to()
                } else {
                    arrival.phase()
                },
                external,
                arrival: Some((arrival.clone(), start == end)),
            },
        )?;
        let completion = if start == end {
            None
        } else {
            Some(self.schedule_event(
                end,
                Event::Drive {
                    driver,
                    drive: arrival.to(),
                    external,
                    arrival: Some((arrival, true)),
                },
            )?)
        };
        Ok((key, completion))
    }

    fn deliver_drive(
        &mut self,
        driver: usize,
        drive: Drive,
        delivery: Option<(Arrival, bool)>,
    ) -> SpiceResult<bool> {
        let previous = self.drivers[driver];
        let mut changed = previous != drive;
        if let Some(records) = &mut self.arrivals {
            let prior = &mut records.drivers[driver];
            let now = self.time.value();
            let next = if let Some((arrival, complete)) = delivery {
                let exact = arrival.window().bounds().is_exact();
                if (complete
                    && !exact
                    && !prior.as_ref().is_some_and(|p| p.same_transition(&arrival)))
                    || ((!complete || exact)
                        && (previous != arrival.from()
                            || prior.as_ref().is_some_and(|p| p.active_at(now))))
                {
                    return Err(SpiceError::circuit(
                        "overlapping or inconsistent causal driver transitions",
                    ));
                }
                Some(arrival)
            } else {
                if prior.as_ref().is_some_and(|p| p.active_at(now)) {
                    return Err(SpiceError::circuit(
                        "scalar event interrupts causal driver transition",
                    ));
                }
                if !changed {
                    return Ok(false);
                }
                exact_arrival(previous, drive, now)?
            };
            changed |= !same_arrival(prior.as_ref(), next.as_ref());
            *prior = next;
        }
        self.drivers[driver] = drive;
        Ok(changed)
    }

    fn net_arrival(
        &self,
        binding: &Net,
        net: NetId,
        resolved: Drive,
    ) -> SpiceResult<Option<Arrival>> {
        let Some(records) = &self.arrivals else {
            return Ok(None);
        };
        let now = self.time.value();
        if let [driver] = binding.drivers.as_slice() {
            if let Some(arrival) = &records.drivers[*driver] {
                let from = Drive::resolve([arrival.from()]);
                let to = Drive::resolve([arrival.to()]);
                if from != to {
                    return Ok(Some(Arrival::new(from, to, arrival.window().clone())?));
                }
                if arrival.active_at(now) {
                    return Err(SpiceError::circuit(
                        "causal drive has indistinguishable resolved endpoints",
                    ));
                }
            }
            return if self.nets[net.0] == resolved {
                Ok(records.nets[net.0].clone())
            } else {
                exact_arrival(self.nets[net.0], resolved, now)
            };
        }
        let margin = binding.margin.unwrap_or(std::num::NonZeroU8::MIN);
        let mut active = binding
            .drivers
            .iter()
            .filter_map(|&id| records.drivers[id].as_ref().filter(|a| a.active_at(now)));
        if let Some(first) = active.next() {
            // A stable driver which dominates every possible endpoint
            // masks uncertainty independently of the transition order.
            let masked =
                binding.drivers.iter().any(|&id| {
                    self.drivers[id] == resolved
                        && records.drivers[id]
                            .as_ref()
                            .is_none_or(|a| !a.active_at(now))
                        && binding.drivers.iter().all(|&other| {
                            records.drivers[other]
                                .as_ref()
                                .filter(|a| a.active_at(now))
                                .is_none_or(|a| {
                                    resolved.strength.rank().saturating_sub(
                                        a.from().strength.max(a.to().strength).rank(),
                                    ) >= margin.get()
                                })
                        })
                });
            if masked {
                return if self.nets[net.0] == resolved {
                    Ok(records.nets[net.0].clone())
                } else {
                    exact_arrival(self.nets[net.0], resolved, now)
                };
            }
            if active.any(|a| !a.window().same_expression(first.window())) {
                return Err(SpiceError::circuit(
                    "independent overlapping causal arrivals on resolved bus",
                ));
            }
            let endpoint = |after: bool| {
                Drive::resolve_with_margin(
                    binding.drivers.iter().map(|&id| {
                        records.drivers[id]
                            .as_ref()
                            .filter(|a| a.active_at(now))
                            .map_or(self.drivers[id], |a| if after { a.to() } else { a.from() })
                    }),
                    margin,
                )
            };
            let from = endpoint(false);
            let to = endpoint(true);
            if from == to && from == resolved {
                return if self.nets[net.0] == resolved {
                    Ok(records.nets[net.0].clone())
                } else {
                    exact_arrival(self.nets[net.0], resolved, now)
                };
            }
            let arrival = Arrival::new(from, to, first.window().clone())?;
            if arrival.phase() != resolved
                || (self.nets[net.0] != from
                    && !records.nets[net.0]
                        .as_ref()
                        .is_some_and(|old| old.same_transition(&arrival)))
            {
                return Err(SpiceError::circuit(
                    "bus resolution changes within causal arrival",
                ));
            }
            return Ok(Some(arrival));
        }
        let prior = &records.nets[net.0];
        if prior
            .as_ref()
            .is_some_and(|a| a.window().bounds().maximum() == now && a.to() == resolved)
        {
            return Ok(prior.clone());
        }
        if self.nets[net.0] == resolved {
            return Ok(prior.clone());
        }
        exact_arrival(self.nets[net.0], resolved, now)
    }

    fn model_arrival(&mut self, driver: usize, arrival: Arrival) -> SpiceResult<()> {
        let earliest = Time::new(arrival.window().bounds().minimum())?;
        // A complete transport transition is indivisible. Do not retain a
        // stale end event when a later waveform overlaps its uncertainty.
        if self.pending[driver]
            .last()
            .is_some_and(|key| key.time > earliest)
            || self
                .arrivals
                .as_ref()
                .and_then(|a| a.drivers[driver].as_ref())
                .is_some_and(|a| a.window().bounds().maximum() > earliest.value())
        {
            return Err(SpiceError::circuit(
                "overlapping causal output transitions require a hazard model",
            ));
        }
        self.desired[driver] = Some(arrival.to());
        let (start, end) = self.schedule_arrival(driver, arrival, None)?;
        self.pending[driver].insert(start);
        if let Some(end) = end {
            self.pending[driver].insert(end);
        }
        Ok(())
    }

    fn schedule_event(&mut self, time: Time, event: Event) -> SpiceResult<Key> {
        if time < self.time {
            return Err(SpiceError::circuit(
                "cannot schedule event before current time",
            ));
        }
        let delta = if time == self.time && self.initialized {
            self.delta
                .checked_add(1)
                .ok_or_else(|| SpiceError::circuit("digital delta overflow"))?
        } else {
            0
        };
        let key = Key {
            time,
            delta,
            sequence: self.sequence,
        };
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| SpiceError::circuit("digital event sequence overflow"))?;
        self.queue.insert(key, event);
        Ok(key)
    }

    fn model_output(&mut self, driver: usize, output: Output) -> SpiceResult<()> {
        self.model_output_at(driver, output, self.time)
    }

    fn model_output_at(&mut self, driver: usize, output: Output, at: Time) -> SpiceResult<()> {
        if self.arrivals.is_some()
            && self.pending[driver].iter().any(|key| {
                matches!(
                    self.queue.get(key),
                    Some(Event::Drive {
                        arrival: Some(_),
                        ..
                    })
                )
            })
        {
            return Err(SpiceError::circuit(
                "scalar output supersedes an unfinished causal transition",
            ));
        }
        let previous = self.desired[driver];
        if previous == Some(output.drive) {
            return Ok(());
        }
        let old = if output.mode == DelayMode::Inertial {
            self.drivers[driver]
        } else {
            previous.unwrap_or(self.drivers[driver])
        };
        self.desired[driver] = Some(output.drive);
        if output.mode == DelayMode::Inertial {
            for key in std::mem::take(&mut self.pending[driver]) {
                self.queue.remove(&key);
            }
            if self.drivers[driver] == output.drive {
                return Ok(());
            }
        }
        let delay = if at == Time(0) {
            0.
        } else {
            output.delays.transition(old, output.drive)
        };
        let time = Time::new(at.value() + delay)?;
        if delay > 0. && time == at {
            return Err(SpiceError::circuit(
                "digital propagation delay is below time resolution",
            ));
        }
        let key = self.schedule(time, driver, output.drive, None)?;
        // EVTqueue_output supersedes already-posted events on this output
        // whose delivery time is at or after the newly posted event.
        let start = Key { sequence: 0, ..key };
        for superseded in self.pending[driver].split_off(&start) {
            self.queue.remove(&superseded);
        }
        self.pending[driver].insert(key);
        Ok(())
    }
}

impl State {
    fn record_diagnostic(&mut self, diagnostic: EventDiagnostic) -> SpiceResult<()> {
        self.stats.diagnostics = self
            .stats
            .diagnostics
            .checked_add(1)
            .ok_or_else(|| SpiceError::circuit("digital diagnostic count overflow"))?;
        if diagnostic.is_fatal() && self.first_fatal.is_none() {
            self.first_fatal = Some(diagnostic.clone());
        }
        self.diagnostics.push(diagnostic);
        Ok(())
    }

    fn check_acceptance(&self) -> SpiceResult<()> {
        if let Some(diagnostic) = &self.first_fatal {
            return Err(SpiceError::circuit(diagnostic.to_string()));
        }
        Ok(())
    }

    fn charge_work(&self, before: Stats, limit: usize) -> SpiceResult<()> {
        let events = self.stats.events - before.events;
        let evaluations = self.stats.evaluations - before.evaluations;
        let probes = self.stats.probes - before.probes;
        if events
            .saturating_add(evaluations)
            .saturating_add(probes)
            .saturating_add(self.stats.diagnostics - before.diagnostics)
            > limit
        {
            return Err(SpiceError::circuit("digital event work limit exceeded"));
        }
        Ok(())
    }

    fn evaluate_models(
        &mut self,
        network: &Network,
        dirty: BTreeSet<usize>,
        before: Stats,
        limit: usize,
    ) -> SpiceResult<()> {
        for index in dirty {
            self.stats.evaluations += 1;
            self.charge_work(before, limit)?;
            let instance = &network.models[index];
            let context = EventContext {
                time: self.time.value(),
                delta: self.delta,
                initial: self.time == Time(0),
            };
            enum Evaluated {
                Scalar(Vec<Output>),
                Timed(Vec<TimedOutput>),
            }
            let (outputs, diagnostics) = if let Some(records) = &self.arrivals {
                let inputs: Vec<_> = instance
                    .inputs
                    .iter()
                    .map(|id| TimedInput {
                        drive: self.nets[id.0],
                        arrival: records.nets[id.0].clone(),
                    })
                    .collect();
                let (outputs, diagnostics) = instance.kernel.evaluate_timed(
                    context,
                    &inputs,
                    self.models[index].as_mut(),
                )?;
                (Evaluated::Timed(outputs), diagnostics)
            } else {
                let inputs: Vec<_> = instance.inputs.iter().map(|id| self.nets[id.0]).collect();
                let (outputs, diagnostics) =
                    instance
                        .kernel
                        .evaluate(context, &inputs, self.models[index].as_mut())?;
                (Evaluated::Scalar(outputs), diagnostics)
            };
            for detail in diagnostics {
                self.record_diagnostic(EventDiagnostic {
                    model: instance.name.clone(),
                    time: self.time.value(),
                    delta: self.delta,
                    detail: DiagnosticDetail::Timing(detail),
                })?;
            }
            self.charge_work(before, limit)?;
            match outputs {
                Evaluated::Scalar(outputs) => self.accept_model_outputs(network, index, outputs)?,
                Evaluated::Timed(outputs) => self.accept_timed_outputs(network, index, outputs)?,
            }
        }
        Ok(())
    }

    fn accept_model_outputs(
        &mut self,
        network: &Network,
        index: usize,
        outputs: Vec<Output>,
    ) -> SpiceResult<()> {
        let instance = &network.models[index];
        if outputs.len() != instance.outputs.len() {
            return Err(SpiceError::circuit(
                "digital model returned incorrect output count",
            ));
        }
        for (&driver, output) in instance.outputs.iter().zip(outputs) {
            self.model_output(driver, output)?;
        }
        self.update_wakeup(network, index)
    }

    fn accept_timed_outputs(
        &mut self,
        network: &Network,
        index: usize,
        outputs: Vec<TimedOutput>,
    ) -> SpiceResult<()> {
        let instance = &network.models[index];
        if outputs.len() != instance.outputs.len() {
            return Err(SpiceError::circuit(
                "digital model returned incorrect output count",
            ));
        }
        for (&driver, output) in instance.outputs.iter().zip(outputs) {
            match output {
                TimedOutput::Retain => {}
                TimedOutput::Scalar(output) => self.model_output(driver, output)?,
                TimedOutput::Arrival(arrival) => self.model_arrival(driver, arrival)?,
            }
        }
        self.update_wakeup(network, index)
    }

    fn update_wakeup(&mut self, network: &Network, index: usize) -> SpiceResult<()> {
        let instance = &network.models[index];
        if let Some(key) = self.wakeups[index].take() {
            self.queue.remove(&key);
        }
        if let Some(time) = instance.kernel.next_wakeup(self.models[index].as_ref())? {
            let key = self.schedule_event(Time::new(time)?, Event::Wake(index))?;
            self.wakeups[index] = Some(key);
        }
        Ok(())
    }

    // A prediction is valid until the model's input changes. Retaining
    // predictions past unrelated events avoids rewinding independent HDL
    // simulators simply because another device has an earlier event.
    fn predict(
        &mut self,
        network: &Network,
        index: usize,
        horizon: Time,
        before: Stats,
        limit: usize,
    ) -> SpiceResult<Option<Time>> {
        if let Some(prediction) = &self.predictions[index]
            && (prediction.outputs.is_some() || prediction.time >= horizon)
        {
            return Ok(prediction.outputs.as_ref().map(|_| prediction.time));
        }
        self.stats.probes += 1;
        self.charge_work(before, limit)?;
        let mut state = self.predictions[index]
            .take()
            .map_or_else(|| self.models[index].clone(), |prediction| prediction.state);
        let result = network.models[index].kernel.advance_autonomous(
            self.time.value(),
            horizon.value(),
            state.as_mut(),
        )?;
        let (time, outputs) = match result {
            AutonomousAdvance::Quiet => (horizon, None),
            AutonomousAdvance::Event { time, outputs } => {
                let time = Time::new(time)?;
                if time <= self.time || time > horizon {
                    return Err(SpiceError::circuit(
                        "autonomous simulator returned a noncausal event",
                    ));
                }
                (time, Some(outputs))
            }
        };
        let event = outputs.as_ref().map(|_| time);
        self.predictions[index] = Some(Prediction {
            time,
            state,
            outputs,
        });
        Ok(event)
    }

    fn next_group(
        &mut self,
        network: &Network,
        end: Time,
        before: Stats,
        limit: usize,
    ) -> SpiceResult<Option<Group>> {
        if !self.initialized {
            return Ok(Some(Group {
                time: Time(0),
                delta: 0,
            }));
        }
        let next = self
            .queue
            .first_key_value()
            .map(|(key, _)| *key)
            .filter(|key| key.time <= end);
        let horizon = next.map_or(end, |key| key.time);
        let mut boundary = next.map(|key| (key.time, key.delta));
        if horizon > self.time {
            for &index in &network.lookahead {
                if let Some(time) = self.predict(network, index, horizon, before, limit)?
                    && time <= horizon
                    && boundary.is_none_or(|(old, _)| time < old)
                {
                    boundary = Some((time, 0));
                }
            }
        }
        if let Some((time, delta)) = boundary {
            Ok(Some(Group { time, delta }))
        } else {
            self.accept_predictions(network, horizon, &BTreeSet::new())?;
            Ok(None)
        }
    }

    fn accept_predictions(
        &mut self,
        network: &Network,
        time: Time,
        dirty: &BTreeSet<usize>,
    ) -> SpiceResult<()> {
        for &index in &network.lookahead {
            if dirty.contains(&index) {
                // An incoming drive at the same physical time takes
                // precedence over autonomous evolution with old inputs.
                self.predictions[index] = None;
                continue;
            }
            if self.predictions[index]
                .as_ref()
                .is_some_and(|prediction| prediction.time <= time)
                && let Some(prediction) = self.predictions[index].take()
            {
                self.models[index] = prediction.state;
                if let Some(outputs) = prediction.outputs {
                    if prediction.time != time || time != self.time {
                        return Err(SpiceError::circuit("missed autonomous event"));
                    }
                    self.accept_model_outputs(network, index, outputs)?;
                }
            }
        }
        Ok(())
    }

    fn advance(
        &mut self,
        network: &Network,
        end: Time,
        limits: Limits,
        changes: &mut Vec<Change>,
    ) -> SpiceResult<()> {
        self.advance_until(network, end, limits, changes, |_| Ok(false))?;
        Ok(())
    }

    // Stop only after all delta cycles at a physical time settle. This
    // allows a mixed run to preview digital work independently of MNA.
    fn advance_until(
        &mut self,
        network: &Network,
        end: Time,
        limits: Limits,
        changes: &mut Vec<Change>,
        mut stop: impl FnMut(&Self) -> SpiceResult<bool>,
    ) -> SpiceResult<bool> {
        if end < self.time {
            return Err(SpiceError::circuit("digital time cannot run backwards"));
        }
        let before = self.stats;
        while let Some(group) = self.next_group(network, end, before, limits.work)? {
            if group.delta >= limits.delta_cycles {
                return Err(SpiceError::circuit(
                    "digital delta-cycle limit exceeded (zero-delay feedback)",
                ));
            }
            (self.time, self.delta) = (group.time, group.delta);
            let mut dirty = BTreeSet::new();
            if !self.initialized {
                dirty.extend(0..network.models.len());
                self.initialized = true;
            }
            let mut affected = BTreeSet::new();
            while let Some((&key, _)) = self.queue.first_key_value() {
                if (key.time, key.delta) != (group.time, group.delta) {
                    break;
                }
                let Some((_, event)) = self.queue.pop_first() else {
                    return Err(SpiceError::circuit("missing queued digital event"));
                };
                self.stats.events += 1;
                self.charge_work(before, limits.work)?;
                match event {
                    Event::Drive {
                        driver,
                        drive,
                        arrival,
                        ..
                    } => {
                        self.pending[driver].remove(&key);
                        if self.deliver_drive(driver, drive, arrival)? {
                            affected.insert(network.drivers[driver]);
                        }
                    }
                    Event::Wake(index) => {
                        self.wakeups[index] = None;
                        dirty.insert(index);
                    }
                }
            }
            for net in affected {
                let binding = &network.nets[net.0];
                let resolved = Drive::resolve_with_margin(
                    binding.drivers.iter().map(|&id| self.drivers[id]),
                    binding.margin.unwrap_or(std::num::NonZeroU8::MIN),
                );
                let arrival = self.net_arrival(binding, net, resolved)?;
                let provenance_changed = if let Some(records) = &mut self.arrivals {
                    let changed = !same_arrival(records.nets[net.0].as_ref(), arrival.as_ref());
                    records.nets[net.0] = arrival.clone();
                    changed
                } else {
                    false
                };
                if self.nets[net.0] != resolved || provenance_changed {
                    self.nets[net.0] = resolved;
                    changes.push(Change {
                        time: self.time.value(),
                        delta: self.delta,
                        net,
                        drive: resolved,
                        arrival,
                    });
                    dirty.extend(binding.subscribers.iter().copied());
                }
            }
            self.stats.rounds += 1;
            self.accept_predictions(network, group.time, &dirty)?;
            self.evaluate_models(network, dirty, before, limits.work)?;
            if self
                .queue
                .first_key_value()
                .is_none_or(|(key, _)| key.time != self.time)
                && stop(self)?
            {
                return Ok(true);
            }
        }
        if self.time < end {
            self.time = end;
            self.delta = 0;
        }
        Ok(false)
    }
}

/// Fixed network with an accepted digital state. Analog-only runs need not
/// allocate an engine or enter any digital scheduling code.
#[derive(Debug)]
pub struct Engine {
    network: std::rc::Rc<Network>,
    accepted: State,
    limits: Limits,
}

impl Engine {
    /// Bind a network with explicit event work limits.
    ///
    /// # Errors
    /// Zero work or delta-cycle bound.
    pub fn new(network: Network, limits: Limits) -> SpiceResult<Self> {
        Self::from_shared(std::rc::Rc::new(network), limits)
    }

    fn from_shared(network: std::rc::Rc<Network>, limits: Limits) -> SpiceResult<Self> {
        if limits.delta_cycles == 0 || limits.work == 0 {
            return Err(SpiceError::circuit("zero digital work limit"));
        }
        let accepted = State::new(&network);
        Ok(Self {
            network,
            accepted,
            limits,
        })
    }

    /// Accepted simulation time.
    #[must_use = "the returned accepted time identifies the committed digital simulation horizon"]
    pub const fn time(&self) -> Real {
        self.accepted.time.value()
    }

    /// Next known queued physical event time, including initialization.
    /// Lookahead-driven models may discover earlier autonomous events.
    #[must_use = "the returned option identifies the next queued physical event for analog timestep scheduling"]
    pub fn next_event_time(&self) -> Option<Real> {
        if self.accepted.initialized {
            self.accepted
                .queue
                .first_key_value()
                .map(|(key, _)| key.time.value())
        } else {
            Some(0.)
        }
    }

    /// True if a model can discover events only by advancing through a horizon.
    #[must_use = "the result determines whether advancement can reveal unqueued autonomous events"]
    pub fn has_lookahead(&self) -> bool {
        !self.network.lookahead.is_empty()
    }

    /// Accepted net value.
    #[must_use = "the returned option distinguishes the accepted drive of a valid net from an invalid net ID"]
    pub fn value(&self, net: NetId) -> Option<Drive> {
        self.accepted.nets.get(net.0).copied()
    }

    /// Latest accepted causal arrival for a valid net, if tracked.
    #[must_use = "the accepted expression retains correlated event timing"]
    pub fn arrival(&self, net: NetId) -> Option<&Arrival> {
        self.accepted.arrivals.as_ref()?.nets.get(net.0)?.as_ref()
    }

    /// Accepted counters.
    #[must_use = "the returned counters quantify committed digital engine work"]
    pub const fn stats(&self) -> Stats {
        self.accepted.stats
    }

    /// Chronologically ordered diagnostics from committed event evaluations.
    #[must_use = "the returned records report accepted timing violations"]
    pub fn diagnostics(&self) -> Vec<EventDiagnostic> {
        self.accepted.diagnostics.records()
    }

    /// Start a disposable trial. Dropping it leaves all accepted state and
    /// queued events untouched (including model state and cancellation).
    pub fn trial(&mut self) -> Trial<'_> {
        let state = self.accepted.clone();
        Trial {
            engine: self,
            state,
            changes: Vec::new(),
            failed: false,
        }
    }
}

/// Trial event state for one or more candidate analog time points.
pub struct Trial<'a> {
    engine: &'a mut Engine,
    state: State,
    changes: Vec<Change>,
    failed: bool,
}

impl Trial<'_> {
    /// Schedule an external output at the specified absolute time.
    ///
    /// # Errors
    /// Unknown source, invalid or past time, exhausted event sequence.
    pub fn schedule(&mut self, source: SourceId, time: Real, drive: Drive) -> SpiceResult<EventId> {
        let driver = *self
            .engine
            .network
            .sources
            .get(source.0)
            .ok_or_else(|| SpiceError::circuit("unknown digital source"))?;
        let identity = external_identity()?;
        let key = self
            .state
            .schedule(Time::new(time)?, driver, drive, Some(identity))?;
        Ok(EventId {
            key,
            completion: None,
            identity,
        })
    }

    /// Queue both boundaries of one causal transition atomically.
    ///
    /// # Errors
    /// Invalid source, past arrival or exhausted event identifiers.
    /// Overlapping or inconsistent source waveforms are rejected at delivery.
    pub fn schedule_arrival(&mut self, source: SourceId, arrival: Arrival) -> SpiceResult<EventId> {
        let driver = *self
            .engine
            .network
            .sources
            .get(source.0)
            .ok_or_else(|| SpiceError::circuit("unknown digital source"))?;
        let identity = external_identity()?;
        let (key, completion) = self
            .state
            .schedule_arrival(driver, arrival, Some(identity))?;
        Ok(EventId {
            key,
            completion,
            identity,
        })
    }

    /// Cancel an external event before its first boundary has been delivered.
    pub fn cancel(&mut self, event: EventId) -> bool {
        let matches = |key| {
            self.state.queue.get(&key).is_some_and(
            |queued| matches!(queued, Event::Drive { external: Some(id), .. } if *id == event.identity))
        };
        if !matches(event.key) || event.completion.is_some_and(|key| !matches(key)) {
            return false;
        }
        self.state.queue.remove(&event.key);
        if let Some(key) = event.completion {
            self.state.queue.remove(&key);
        }
        true
    }

    /// Evaluate all events up to and including the specified physical time.
    /// Failed evaluation poisons this trial: discard it rather than committing
    /// partially processed events or model state.
    ///
    /// # Errors
    /// Invalid time, model failure or exceeded event limits.
    pub fn advance_to(&mut self, time: Real) -> SpiceResult<()> {
        if self.failed {
            return Err(SpiceError::circuit("digital trial already failed"));
        }
        let result = Time::new(time).and_then(|end| {
            self.state.advance(
                &self.engine.network,
                end,
                self.engine.limits,
                &mut self.changes,
            )
        });
        self.failed = result.is_err();
        result
    }

    /// Current trial net value.
    #[must_use = "the returned option identifies the provisional net drive without committing trial state"]
    pub fn value(&self, net: NetId) -> Option<Drive> {
        self.state.nets.get(net.0).copied()
    }

    /// Latest provisional causal arrival for a valid net, if tracked.
    #[must_use = "the provisional expression retains correlated event timing"]
    pub fn arrival(&self, net: NetId) -> Option<&Arrival> {
        self.state.arrivals.as_ref()?.nets.get(net.0)?.as_ref()
    }

    /// Diagnostics in the disposable event state, including prior committed
    /// diagnostics. Dropping the trial discards newly generated records.
    #[must_use = "the returned records describe trial timing violations"]
    pub fn diagnostics(&self) -> Vec<EventDiagnostic> {
        self.state.diagnostics.records()
    }

    /// Resolved net transitions in this trial, in deterministic order.
    #[must_use = "the returned ordered transitions supply the trial changes for mixed-signal synchronization"]
    pub fn changes(&self) -> &[Change] {
        &self.changes
    }

    /// Next known queued event in the trial; autonomous horizon probes may
    /// discover earlier events.
    #[must_use = "the returned option identifies the next queued event of the provisional trial state"]
    pub fn next_event_time(&self) -> Option<Real> {
        if self.state.initialized {
            self.state
                .queue
                .first_key_value()
                .map(|(key, _)| key.time.value())
        } else {
            Some(0.)
        }
    }

    /// Commit the trial at the analog acceptance boundary.
    ///
    /// # Errors
    /// A previous advance failed.
    pub fn commit(self) -> SpiceResult<()> {
        if self.failed {
            return Err(SpiceError::circuit("cannot commit failed digital trial"));
        }
        self.state.check_acceptance()?;
        self.engine.accepted = self.state;
        Ok(())
    }
}

use crate::devices::Circuit;
use crate::devices::code_model::{
    Adc, AnalogOutputConfig, AnalogOutputDevice, DacConfig, Delays, OutputPort, OutputRamp,
    RangeReceiver, RangeState,
};
use crate::maths::Vector;
use crate::primitives::NodeId;

/// Differential analog input coupled to an exclusively owned digital source.
#[derive(Clone, Copy, Debug)]
pub struct AdcBinding {
    /// Positive and negative analog terminals.
    pub terminals: [NodeId; 2],
    /// Digital source dedicated to this ADC.
    pub source: SourceId,
    /// Comparator or Schmitt thresholds.
    pub thresholds: Adc,
    /// Output propagation delays.
    pub delays: Delays,
    /// Output cancellation policy.
    pub mode: DelayMode,
}

/// Ordered analog receiver coupled to an exclusively owned digital source.
#[derive(Clone, Debug)]
pub struct RangeBinding {
    /// Instance name used by accepted gap diagnostics.
    pub name: std::sync::Arc<str>,
    /// Positive and negative analog terminals.
    pub terminals: [NodeId; 2],
    /// Digital source dedicated to this receiver.
    pub source: SourceId,
    /// Ordered voltage intervals and explicit fallback policy.
    pub receiver: RangeReceiver,
    /// Propagation delays.
    pub delays: Delays,
    /// Output cancellation policy.
    pub mode: DelayMode,
}

#[derive(Clone, Debug)]
enum ReceiverKind {
    Binary(Adc),
    Ordered {
        name: std::sync::Arc<str>,
        material: RangeReceiver,
    },
}

#[derive(Clone, Debug)]
pub(crate) struct ReceiverBinding {
    terminals: [NodeId; 2],
    source: SourceId,
    kind: ReceiverKind,
    delays: Delays,
    mode: DelayMode,
}

impl From<AdcBinding> for ReceiverBinding {
    fn from(binding: AdcBinding) -> Self {
        Self {
            terminals: binding.terminals,
            source: binding.source,
            kind: ReceiverKind::Binary(binding.thresholds),
            delays: binding.delays,
            mode: binding.mode,
        }
    }
}

impl From<RangeBinding> for ReceiverBinding {
    fn from(binding: RangeBinding) -> Self {
        Self {
            terminals: binding.terminals,
            source: binding.source,
            kind: ReceiverKind::Ordered {
                name: binding.name,
                material: binding.receiver,
            },
            delays: binding.delays,
            mode: binding.mode,
        }
    }
}

impl ReceiverBinding {
    fn initial_state(&self) -> Option<RangeState> {
        match &self.kind {
            ReceiverKind::Binary(_) => None,
            ReceiverKind::Ordered { material, .. } => Some(material.initial_state()),
        }
    }

    fn crossing(
        &self,
        start: (Real, Real),
        end: (Real, Real),
        previous: Drive,
        state: Option<RangeState>,
    ) -> SpiceResult<Option<Real>> {
        match (&self.kind, state) {
            (ReceiverKind::Binary(adc), None) => adc.crossing(start, end, previous.logic),
            (ReceiverKind::Ordered { material, .. }, Some(state)) => {
                material.crossing(start, end, state)
            }
            _ => Err(SpiceError::circuit("receiver state does not match binding")),
        }
    }

    fn sample(
        &self,
        time: Time,
        voltage: Real,
        previous: Drive,
        state: Option<&mut RangeState>,
        diagnostics: &mut Vec<EventDiagnostic>,
    ) -> SpiceResult<(Drive, bool)> {
        match (&self.kind, state) {
            (ReceiverKind::Binary(adc), None) => Ok((
                Drive::strong(adc.sample_at(voltage, previous.logic, time == Time(0))?),
                adc.is_qualified(),
            )),
            (ReceiverKind::Ordered { name, material }, Some(prior)) => {
                let sample = material.select(voltage, *prior)?;
                if sample.state.in_gap()
                    && (time == Time(0) || sample.state != *prior || material.rejects_gaps())
                {
                    diagnostics.push(EventDiagnostic {
                        model: name.clone(),
                        time: time.value(),
                        delta: 0,
                        detail: DiagnosticDetail::RangeGap {
                            fatal: material.rejects_gaps(),
                            voltage,
                            drive: sample.drive,
                            range_index: sample.state.index(),
                        },
                    });
                }
                *prior = sample.state;
                Ok((sample.drive, true))
            }
            _ => Err(SpiceError::circuit("receiver state does not match binding")),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AnalogOutputBinding {
    pub(crate) name: String,
    pub(crate) net: NetId,
    pub(crate) config: AnalogOutputConfig,
    pub(crate) port: OutputPort,
}

/// Immutable source-bound topology. Each analysis receives a new event
/// state and accepted history, while its analog device ports stay bound.
#[derive(Debug)]
pub(crate) struct NativeTopology {
    network: std::rc::Rc<Network>,
    receivers: Vec<ReceiverBinding>,
    analog_outputs: Vec<AnalogOutputBinding>,
}

impl NativeTopology {
    pub(crate) fn new(
        network: Network,
        receivers: Vec<ReceiverBinding>,
        analog_outputs: Vec<AnalogOutputBinding>,
    ) -> Self {
        Self {
            network: std::rc::Rc::new(network),
            receivers,
            analog_outputs,
        }
    }

    pub(crate) fn start(&self) -> SpiceResult<MixedSignal> {
        let mut mixed = MixedSignal::new(Engine::from_shared(
            std::rc::Rc::clone(&self.network),
            Limits::default(),
        )?);
        mixed.receivers.clone_from(&self.receivers);
        mixed.range_states = self
            .receivers
            .iter()
            .filter_map(ReceiverBinding::initial_state)
            .collect();
        mixed.analog_outputs.clone_from(&self.analog_outputs);
        Ok(mixed)
    }
}

/// Mixed-signal transient configuration and accepted event history.
/// The analog driver is the only consumer of the bridge forcing signals.
#[derive(Debug)]
pub struct MixedSignal {
    engine: Engine,
    receivers: Vec<ReceiverBinding>,
    analog_outputs: Vec<AnalogOutputBinding>,
    range_states: Vec<RangeState>,
    initial_gap_diagnostics: Vec<EventDiagnostic>,
    previous: Option<(Time, Vec<Real>)>,
    changes: Vec<Change>,
    initial_backup: Option<State>,
    boundary: bool,
}

/// Disposable event work up to the requested analog horizon, or the first
/// settled digital transition that changes a DAC target. Nothing is accepted.
pub(super) struct EventPreview {
    state: State,
    changes: Vec<Change>,
    boundary: Option<Real>,
}

impl EventPreview {
    pub(super) const fn boundary(&self) -> Option<Real> {
        self.boundary
    }
}

/// Disposable mixed-signal result at one converged analog candidate.
pub(super) struct MixedPoint {
    time: Time,
    voltages: Vec<Real>,
    range_states: Vec<RangeState>,
    initial_gap_diagnostics: Vec<EventDiagnostic>,
    state: Option<State>,
    ramps: Vec<OutputRamp>,
    changes: Vec<Change>,
    boundary: bool,
}

impl MixedSignal {
    /// Couple a digital engine to an analog transient. External stimuli may
    /// already be queued, but physical time must not have advanced.
    #[must_use = "the constructed coupling must be retained to synchronize analog advancement with the digital engine"]
    pub const fn new(engine: Engine) -> Self {
        Self {
            engine,
            receivers: Vec::new(),
            range_states: Vec::new(),
            initial_gap_diagnostics: Vec::new(),
            analog_outputs: Vec::new(),
            previous: None,
            changes: Vec::new(),
            initial_backup: None,
            boundary: false,
        }
    }

    /// Accepted event engine, including work counters.
    #[must_use = "the returned engine exposes accepted digital state and work counters"]
    pub const fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Digital transitions accepted by the coupled run.
    #[must_use = "the returned transitions are the committed digital events of the coupled run"]
    pub fn changes(&self) -> &[Change] {
        &self.changes
    }

    /// Bind an ADC to a source not also driven by external stimuli.
    ///
    /// # Errors
    /// Unknown or already assigned source.
    pub fn add_adc(&mut self, binding: AdcBinding) -> SpiceResult<()> {
        self.add_receiver(binding.into())
    }

    /// Bind an ordered differential voltage receiver to an exclusive source.
    ///
    /// # Errors
    /// Unknown or already assigned source.
    pub fn add_range_receiver(&mut self, binding: RangeBinding) -> SpiceResult<()> {
        self.add_receiver(binding.into())
    }

    fn add_receiver(&mut self, binding: ReceiverBinding) -> SpiceResult<()> {
        let driver = *self
            .engine
            .network
            .sources
            .get(binding.source.0)
            .ok_or_else(|| SpiceError::circuit("unknown ADC source"))?;
        if self
            .receivers
            .iter()
            .any(|receiver| receiver.source == binding.source)
            || self
                .engine
                .accepted
                .queue
                .values()
                .any(|event| matches!(event, Event::Drive { driver: id, .. } if *id == driver))
            || self.engine.accepted.drivers[driver] != Drive::default()
        {
            return Err(SpiceError::circuit("ADC source must be exclusively owned"));
        }
        self.range_states.extend(binding.initial_state());
        self.receivers.push(binding);
        Ok(())
    }

    /// Add an ideal differential voltage DAC to the analog circuit, bound
    /// to a resolved digital net. The DAC port is inactive outside this run.
    ///
    /// # Errors
    /// Unknown digital net, duplicate device name or invalid analog terminal.
    pub fn add_dac(
        &mut self,
        circuit: &mut Circuit,
        name: impl Into<String>,
        terminals: [NodeId; 2],
        net: NetId,
        config: DacConfig,
    ) -> SpiceResult<()> {
        self.engine.network.check_net(net)?;
        if terminals[0] == terminals[1] {
            return Err(SpiceError::circuit("DAC output terminals must differ"));
        }
        let name = name.into();
        let port = OutputPort::default();
        circuit.add_device(Box::new(AnalogOutputDevice {
            name: name.clone(),
            terminals,
            port: port.clone(),
            config: AnalogOutputConfig::Voltage(config),
            native: None,
        }))?;
        self.analog_outputs.push(AnalogOutputBinding {
            name,
            net,
            config: AnalogOutputConfig::Voltage(config),
            port,
        });
        Ok(())
    }

    pub(super) fn activate(&mut self, circuit: &Circuit) -> SpiceResult<()> {
        if self.engine.accepted.time != Time(0)
            || self.previous.is_some()
            || self.initial_backup.is_some()
        {
            return Err(SpiceError::circuit(
                "mixed-signal transient requires an unused time-zero engine",
            ));
        }
        for receiver in &self.receivers {
            if receiver
                .terminals
                .iter()
                .any(|&node| circuit.nodes().node(node).is_none())
            {
                return Err(SpiceError::circuit("ADC terminal is not in the circuit"));
            }
        }
        for binding in &self.analog_outputs {
            let device = circuit
                .device(&binding.name)
                .and_then(|device| {
                    (device as &dyn std::any::Any).downcast_ref::<AnalogOutputDevice>()
                })
                .ok_or_else(|| {
                    SpiceError::circuit("analog output device is missing from the circuit")
                })?;
            if !device.port.same(&binding.port) {
                return Err(SpiceError::circuit(
                    "analog output binding belongs to a different circuit",
                ));
            }
        }
        let ramps = self
            .analog_outputs
            .iter()
            .map(|binding| {
                binding
                    .config
                    .target(self.engine.accepted.nets[binding.net.0])
                    .map(OutputRamp::initial)
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        self.initial_backup = Some(self.engine.accepted.clone());
        for (binding, ramp) in self.analog_outputs.iter().zip(ramps) {
            binding.port.set(Some(ramp));
        }
        Ok(())
    }

    pub(super) fn deactivate(&mut self) {
        for binding in &self.analog_outputs {
            binding.port.set(None);
        }
        if let Some(previous) = self.initial_backup.take() {
            self.engine.accepted = previous;
            self.previous = None;
            self.range_states = self
                .receivers
                .iter()
                .filter_map(ReceiverBinding::initial_state)
                .collect();
            self.initial_gap_diagnostics.clear();
            self.changes.clear();
            self.boundary = false;
        }
    }

    pub(super) fn accept_initial(&mut self) -> SpiceResult<()> {
        let before = self.engine.accepted.stats;
        for diagnostic in std::mem::take(&mut self.initial_gap_diagnostics) {
            self.engine.accepted.record_diagnostic(diagnostic)?;
        }
        self.engine
            .accepted
            .charge_work(before, self.engine.limits.work)?;
        self.engine.accepted.check_acceptance()?;
        self.initial_backup = None;
        Ok(())
    }

    fn voltages(&self, circuit: &Circuit, solution: &Vector) -> SpiceResult<Vec<Real>> {
        let voltage = |node| {
            if node == NodeId::GROUND {
                Ok(0.)
            } else {
                circuit
                    .unknowns()
                    .node_row(node)
                    .and_then(|row| solution.get(row))
                    .ok_or_else(|| SpiceError::circuit("missing ADC analog row"))
            }
        };
        self.receivers
            .iter()
            .map(|receiver| {
                let v = voltage(receiver.terminals[0])? - voltage(receiver.terminals[1])?;
                if !v.is_finite() {
                    return Err(SpiceError::circuit("nonfinite differential ADC input"));
                }
                Ok(v)
            })
            .collect()
    }

    fn receiver_previous(&self, receiver: &ReceiverBinding) -> Drive {
        let driver = self.engine.network.sources[receiver.source.0];
        self.engine.accepted.desired[driver].unwrap_or(self.engine.accepted.drivers[driver])
    }

    pub(super) fn first_crossing(
        &self,
        circuit: &Circuit,
        solution: &Vector,
        time: Real,
    ) -> SpiceResult<Option<Real>> {
        let Some((start, previous)) = &self.previous else {
            return Ok(None);
        };
        let voltages = self.voltages(circuit, solution)?;
        let mut earliest = None;
        let mut states = self.range_states.iter().copied();
        for ((receiver, &v0), v1) in self.receivers.iter().zip(previous).zip(voltages) {
            let state = match receiver.kind {
                ReceiverKind::Binary(_) => None,
                ReceiverKind::Ordered { .. } => states.next(),
            };
            if let Some(crossing) = receiver.crossing(
                (start.value(), v0),
                (time, v1),
                self.receiver_previous(receiver),
                state,
            )? {
                earliest = Some(earliest.map_or(crossing, |old: Real| old.min(crossing)));
            }
        }
        if states.next().is_some() {
            return Err(SpiceError::circuit("surplus ordered receiver state"));
        }
        Ok(earliest)
    }

    /// Earliest end of an accepted DAC ramp.
    pub(super) fn next_ramp_boundary(&self, now: Real) -> SpiceResult<Option<Real>> {
        let mut next: Option<Real> = None;
        for binding in &self.analog_outputs {
            if let Some(end) = binding.port.ramp()?.next_break(now) {
                next = Some(next.map_or(end, |old| old.min(end)));
            }
        }
        Ok(next.filter(|time| *time > now))
    }

    /// Process queued events only as far as the first actual DAC target
    /// change, rather than forcing an analog solve for every digital event.
    /// Reuse this state on acceptance if the analog endpoint is unchanged.
    pub(super) fn preview(&self, horizon: Real) -> SpiceResult<Option<EventPreview>> {
        if self.analog_outputs.is_empty()
            || (!self.engine.has_lookahead()
                && self
                    .engine
                    .next_event_time()
                    .is_none_or(|next| next > horizon))
        {
            return Ok(None);
        }
        let end = Time::new(horizon)?;
        let mut state = self.engine.accepted.clone();
        let mut changes = Vec::new();
        let landed = state.advance_until(
            &self.engine.network,
            end,
            self.engine.limits,
            &mut changes,
            |trial| {
                for binding in &self.analog_outputs {
                    let old = binding
                        .config
                        .target(self.engine.accepted.nets[binding.net.0])?;
                    let new = binding.config.target(trial.nets[binding.net.0])?;
                    if old.to_bits() != new.to_bits() {
                        return Ok(true);
                    }
                }
                Ok(false)
            },
        )?;
        let boundary = landed.then_some(state.time.value());
        Ok(Some(EventPreview {
            state,
            changes,
            boundary,
        }))
    }

    pub(super) const fn boundary(&self) -> bool {
        self.boundary
    }

    pub(super) fn prepare_point(
        &self,
        circuit: &Circuit,
        solution: &Vector,
        time: Real,
        preview: Option<EventPreview>,
    ) -> SpiceResult<MixedPoint> {
        let end = Time::new(time)?;
        if end < self.engine.accepted.time {
            return Err(SpiceError::circuit(
                "analog candidate precedes accepted event time",
            ));
        }
        let voltages = self.voltages(circuit, solution)?;
        let mut updates = Vec::new();
        let mut qualified_updates = Vec::new();
        let mut range_states = self.range_states.clone();
        let mut range_diagnostics = Vec::new();
        let mut states = range_states.iter_mut();
        for (receiver, &voltage) in self.receivers.iter().zip(&voltages) {
            let range_state = match receiver.kind {
                ReceiverKind::Binary(_) => None,
                ReceiverKind::Ordered { .. } => states.next(),
            };
            let previous = self.receiver_previous(receiver);
            let (drive, qualified) =
                receiver.sample(end, voltage, previous, range_state, &mut range_diagnostics)?;
            if drive != previous {
                let destination = if qualified {
                    &mut qualified_updates
                } else {
                    &mut updates
                };
                destination.push((
                    self.engine.network.sources[receiver.source.0],
                    Output {
                        drive,
                        delays: receiver.delays,
                        mode: receiver.mode,
                    },
                ));
            }
        }
        if states.next().is_some() {
            return Err(SpiceError::circuit("surplus ordered receiver state"));
        }
        let events_due = (self.engine.has_lookahead() && end > self.engine.accepted.time)
            || self
                .engine
                .next_event_time()
                .is_some_and(|next| next <= time);
        let ramp_boundary = self
            .analog_outputs
            .iter()
            .try_fold(false, |found, binding| {
                let ramp = binding.port.ramp()?;
                let previous = self.engine.time();
                Ok::<_, SpiceError>(
                    found || ramp.next_break(previous).is_some_and(|end| end <= time),
                )
            })?;
        let mut point = MixedPoint {
            time: end,
            voltages,
            range_states,
            initial_gap_diagnostics: Vec::new(),
            state: None,
            ramps: Vec::new(),
            changes: Vec::new(),
            boundary: ramp_boundary,
        };
        if end == Time(0) {
            point.initial_gap_diagnostics = std::mem::take(&mut range_diagnostics);
        }
        if !events_due
            && updates.is_empty()
            && qualified_updates.is_empty()
            && range_diagnostics.is_empty()
        {
            return Ok(point);
        }
        let mut state =
            self.prepare_event_state(end, preview, qualified_updates, updates, &mut point.changes)?;
        for diagnostic in range_diagnostics {
            state.record_diagnostic(diagnostic)?;
        }
        state.charge_work(self.engine.accepted.stats, self.engine.limits.work)?;
        // Previews can discover tentative timing failures. Only the replay
        // with converged receiver values may prevent analog acceptance.
        state.check_acceptance()?;
        for binding in &self.analog_outputs {
            let drive = state.nets[binding.net.0];
            let previous_ramp = binding.port.ramp()?;
            let ramp = if end == Time(0) {
                OutputRamp::initial(binding.config.target(drive)?)
            } else {
                binding.config.retarget(previous_ramp, time, drive)?
            };
            point.boundary |= !ramp.same_target(previous_ramp);
            point.ramps.push(ramp);
        }
        point.state = Some(state);
        Ok(point)
    }

    /// Qualified receivers are sampled before any timer at the same endpoint.
    /// The pre-end preview is disposable; replay it if a receiver changes.
    fn prepare_event_state(
        &self,
        end: Time,
        preview: Option<EventPreview>,
        early: Vec<(usize, Output)>,
        late: Vec<(usize, Output)>,
        changes: &mut Vec<Change>,
    ) -> SpiceResult<State> {
        let mut state = if early.is_empty() {
            if let Some(preview) = preview.filter(|p| p.state.time == end) {
                *changes = preview.changes;
                preview.state
            } else {
                let mut state = self.engine.accepted.clone();
                if state.initialized {
                    state.advance(&self.engine.network, end, self.engine.limits, changes)?;
                }
                state
            }
        } else {
            let mut state = self.engine.accepted.clone();
            if state.initialized && end > state.time {
                let before = Time::new(end.value().next_down())?;
                state.advance(&self.engine.network, before, self.engine.limits, changes)?;
            }
            for (driver, output) in early {
                state.model_output_at(driver, output, end)?;
            }
            if state.initialized {
                state.advance(&self.engine.network, end, self.engine.limits, changes)?;
            }
            state
        };
        for (driver, output) in late {
            state.model_output(driver, output)?;
        }
        state.advance(&self.engine.network, end, self.engine.limits, changes)?;
        Ok(state)
    }

    pub(super) fn initial_changed(&self, point: &MixedPoint) -> SpiceResult<bool> {
        for (binding, ramp) in self.analog_outputs.iter().zip(&point.ramps) {
            if binding.port.ramp()?.value(0.).to_bits() != ramp.value(0.).to_bits() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Update the initial linear source RHS to match the current mixed bias
    /// candidate; the companion system's topology and A/E remain immutable.
    pub(super) fn initial_rhs(&self, circuit: &Circuit, rhs: &mut Vector) -> SpiceResult<()> {
        for binding in &self.analog_outputs {
            if !binding.config.is_voltage() {
                continue;
            }
            let index = circuit
                .devices()
                .iter()
                .position(|device| device.name().eq_ignore_ascii_case(&binding.name))
                .ok_or_else(|| SpiceError::circuit("missing DAC device"))?;
            let row = circuit
                .branch_rows(index)
                .and_then(|mut rows| rows.next())
                .ok_or_else(|| SpiceError::circuit("missing DAC branch row"))?;
            let value = binding.port.ramp()?.value(0.);
            let slot = rhs
                .as_mut_slice()
                .get_mut(row)
                .ok_or_else(|| SpiceError::circuit("missing DAC RHS row"))?;
            *slot = value;
        }
        Ok(())
    }

    pub(super) fn commit_point(&mut self, point: MixedPoint) {
        if let Some(state) = point.state {
            self.engine.accepted = state;
            for (binding, ramp) in self.analog_outputs.iter().zip(point.ramps) {
                binding.port.set(Some(ramp));
            }
        } else if self.engine.accepted.time < point.time {
            self.engine.accepted.time = point.time;
            self.engine.accepted.delta = 0;
        }
        self.range_states = point.range_states;
        if point.time == Time(0) {
            self.initial_gap_diagnostics = point.initial_gap_diagnostics;
        }
        self.previous = Some((point.time, point.voltages));
        self.changes.extend(point.changes);
        self.boundary = point.boundary;
    }
}
