//! Deterministic digital event transactions, independent of analog Newton
//! iterations. See upstream XSPICE evtqueue.c, evtiter.c and evtbackup.c.

use std::collections::{BTreeMap, BTreeSet};

use crate::devices::code_model::{CodeModel, DelayMode, Drive, EventContext, Output};
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
    identity: u64,
}

fn external_identity() -> SpiceResult<u64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| SpiceError::circuit("external event identity space exhausted"))
}

#[derive(Clone, Copy, Debug)]
struct Event {
    driver: usize,
    drive: Drive,
    external: Option<u64>,
}

#[derive(Debug, Default)]
struct Net {
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
    fn evaluate(
        &self,
        context: EventContext,
        inputs: &[Drive],
        state: &mut dyn ModelState,
    ) -> SpiceResult<Vec<Output>>;
}

impl<M: CodeModel> ErasedModel for M {
    fn initial_state(&self) -> Box<dyn ModelState> {
        Box::new(CodeModel::initial_state(self))
    }

    fn evaluate(
        &self,
        context: EventContext,
        inputs: &[Drive],
        state: &mut dyn ModelState,
    ) -> SpiceResult<Vec<Output>> {
        let state = (state as &mut dyn std::any::Any)
            .downcast_mut::<M::State>()
            .ok_or_else(|| SpiceError::circuit("digital model state identity mismatch"))?;
        CodeModel::evaluate(self, context, inputs, state)
    }
}

#[derive(Debug)]
struct Model {
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
}

impl Network {
    /// Add an initially undriven net.
    pub fn add_net(&mut self) -> NetId {
        let id = NetId(self.nets.len());
        self.nets.push(Net::default());
        id
    }

    fn check_net(&self, id: NetId) -> SpiceResult<()> {
        if id.0 >= self.nets.len() {
            return Err(SpiceError::circuit("unknown digital net"));
        }
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
        if model.schema()
            != crate::devices::code_model::ModelSchema::digital(inputs.len(), outputs.len())
        {
            return Err(SpiceError::circuit("digital model port count mismatch"));
        }
        for &net in inputs.iter().chain(outputs) {
            self.check_net(net)?;
        }
        let index = self.models.len();
        for &net in inputs {
            self.nets[net.0].subscribers.push(index);
        }
        let drivers = outputs.iter().map(|&net| self.add_driver(net)).collect();
        self.models.push(Model {
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
}

/// Resolved net change after all output events at a time/delta are applied.
#[derive(Clone, Copy, Debug)]
pub struct Change {
    /// Physical time.
    pub time: Real,
    /// Delta cycle.
    pub delta: usize,
    /// Net that changed.
    pub net: NetId,
    /// Resolved value.
    pub drive: Drive,
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
    nets: Vec<Drive>,
    models: Vec<Box<dyn ModelState>>,
    stats: Stats,
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
            nets: vec![Drive::default(); network.nets.len()],
            models: network
                .models
                .iter()
                .map(|m| m.kernel.initial_state())
                .collect(),
            stats: Stats::default(),
        }
    }

    fn schedule(
        &mut self,
        time: Time,
        driver: usize,
        drive: Drive,
        external: Option<u64>,
    ) -> SpiceResult<Key> {
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
        self.queue.insert(
            key,
            Event {
                driver,
                drive,
                external,
            },
        );
        Ok(key)
    }

    fn model_output(&mut self, driver: usize, output: Output) -> SpiceResult<()> {
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
        let delay = if self.time == Time(0) {
            0.
        } else {
            output.delays.transition(old.logic, output.drive.logic)
        };
        let time = Time::new(self.time.value() + delay)?;
        if delay > 0. && time == self.time {
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
    fn charge_work(&self, before: Stats, limit: usize) -> SpiceResult<()> {
        let events = self.stats.events - before.events;
        let evaluations = self.stats.evaluations - before.evaluations;
        if events.saturating_add(evaluations) > limit {
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
            let inputs: Vec<_> = instance.inputs.iter().map(|id| self.nets[id.0]).collect();
            let outputs = instance.kernel.evaluate(
                EventContext {
                    time: self.time.value(),
                    delta: self.delta,
                    initial: self.time == Time(0),
                },
                &inputs,
                self.models[index].as_mut(),
            )?;
            if outputs.len() != instance.outputs.len() {
                return Err(SpiceError::circuit(
                    "digital model returned incorrect output count",
                ));
            }
            for (&driver, output) in instance.outputs.iter().zip(outputs) {
                self.model_output(driver, output)?;
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
        loop {
            let next = self.queue.first_key_value().map(|(key, _)| *key);
            let group = if self.initialized {
                let Some(key) = next.filter(|key| key.time <= end) else {
                    break;
                };
                (key.time, key.delta)
            } else {
                (Time(0), 0)
            };
            if group.1 >= limits.delta_cycles {
                return Err(SpiceError::circuit(
                    "digital delta-cycle limit exceeded (zero-delay feedback)",
                ));
            }
            (self.time, self.delta) = group;
            let mut dirty = BTreeSet::new();
            if !self.initialized {
                dirty.extend(0..network.models.len());
                self.initialized = true;
            }
            let mut affected = BTreeSet::new();
            while let Some((&key, _)) = self.queue.first_key_value() {
                if (key.time, key.delta) != group {
                    break;
                }
                let Some((_, event)) = self.queue.pop_first() else {
                    return Err(SpiceError::circuit("missing queued digital event"));
                };
                self.stats.events += 1;
                self.charge_work(before, limits.work)?;
                self.pending[event.driver].remove(&key);
                if self.drivers[event.driver] != event.drive {
                    self.drivers[event.driver] = event.drive;
                    affected.insert(network.drivers[event.driver]);
                }
            }
            for net in affected {
                let binding = &network.nets[net.0];
                let resolved = Drive::resolve(binding.drivers.iter().map(|&id| self.drivers[id]));
                if self.nets[net.0] != resolved {
                    self.nets[net.0] = resolved;
                    changes.push(Change {
                        time: self.time.value(),
                        delta: self.delta,
                        net,
                        drive: resolved,
                    });
                    dirty.extend(binding.subscribers.iter().copied());
                }
            }
            self.stats.rounds += 1;
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
    network: Network,
    accepted: State,
    limits: Limits,
}

impl Engine {
    /// Bind a network with explicit event work limits.
    ///
    /// # Errors
    /// Zero work or delta-cycle bound.
    pub fn new(network: Network, limits: Limits) -> SpiceResult<Self> {
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
    #[must_use]
    pub const fn time(&self) -> Real {
        self.accepted.time.value()
    }

    /// Next pending physical event time, including initial model evaluation.
    #[must_use]
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

    /// Accepted net value.
    #[must_use]
    pub fn value(&self, net: NetId) -> Option<Drive> {
        self.accepted.nets.get(net.0).copied()
    }

    /// Accepted counters.
    #[must_use]
    pub const fn stats(&self) -> Stats {
        self.accepted.stats
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
        Ok(EventId { key, identity })
    }

    /// Cancel an external event if it has not yet been delivered.
    pub fn cancel(&mut self, event: EventId) -> bool {
        if self
            .state
            .queue
            .get(&event.key)
            .is_some_and(|queued| queued.external == Some(event.identity))
        {
            self.state.queue.remove(&event.key);
            true
        } else {
            false
        }
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
    #[must_use]
    pub fn value(&self, net: NetId) -> Option<Drive> {
        self.state.nets.get(net.0).copied()
    }

    /// Resolved net transitions in this trial, in deterministic order.
    #[must_use]
    pub fn changes(&self) -> &[Change] {
        &self.changes
    }

    /// Next pending event in the trial.
    #[must_use]
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
        self.engine.accepted = self.state;
        Ok(())
    }
}

use crate::devices::Circuit;
use crate::devices::code_model::{Adc, DacConfig, DacPort, DacRamp, DacSource, Delays};
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

#[derive(Debug)]
struct DacBinding {
    name: String,
    net: NetId,
    config: DacConfig,
    port: DacPort,
}

/// Mixed-signal transient configuration and accepted event history.
/// The analog driver is the only consumer of the bridge forcing signals.
#[derive(Debug)]
pub struct MixedSignal {
    engine: Engine,
    adcs: Vec<AdcBinding>,
    dacs: Vec<DacBinding>,
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
    state: Option<State>,
    ramps: Vec<DacRamp>,
    changes: Vec<Change>,
    boundary: bool,
}

impl MixedSignal {
    /// Couple a digital engine to an analog transient. External stimuli may
    /// already be queued, but physical time must not have advanced.
    #[must_use]
    pub const fn new(engine: Engine) -> Self {
        Self {
            engine,
            adcs: Vec::new(),
            dacs: Vec::new(),
            previous: None,
            changes: Vec::new(),
            initial_backup: None,
            boundary: false,
        }
    }

    /// Accepted event engine, including work counters.
    #[must_use]
    pub const fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Digital transitions accepted by the coupled run.
    #[must_use]
    pub fn changes(&self) -> &[Change] {
        &self.changes
    }

    /// Bind an ADC to a source not also driven by external stimuli.
    ///
    /// # Errors
    /// Unknown or already assigned source.
    pub fn add_adc(&mut self, binding: AdcBinding) -> SpiceResult<()> {
        let driver = *self
            .engine
            .network
            .sources
            .get(binding.source.0)
            .ok_or_else(|| SpiceError::circuit("unknown ADC source"))?;
        if self.adcs.iter().any(|adc| adc.source == binding.source)
            || self
                .engine
                .accepted
                .queue
                .values()
                .any(|event| event.driver == driver)
            || self.engine.accepted.drivers[driver] != Drive::default()
        {
            return Err(SpiceError::circuit("ADC source must be exclusively owned"));
        }
        self.adcs.push(binding);
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
        let port = DacPort::default();
        circuit.add_device(Box::new(DacSource {
            name: name.clone(),
            terminals,
            port: port.clone(),
        }))?;
        self.dacs.push(DacBinding {
            name,
            net,
            config,
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
        for adc in &self.adcs {
            if adc
                .terminals
                .iter()
                .any(|&node| circuit.nodes().node(node).is_none())
            {
                return Err(SpiceError::circuit("ADC terminal is not in the circuit"));
            }
        }
        for dac in &self.dacs {
            let device = circuit
                .device(&dac.name)
                .and_then(|device| (device as &dyn std::any::Any).downcast_ref::<DacSource>())
                .ok_or_else(|| SpiceError::circuit("DAC device is missing from the circuit"))?;
            if !device.port.same(&dac.port) {
                return Err(SpiceError::circuit(
                    "DAC binding belongs to a different circuit",
                ));
            }
        }
        let ramps = self
            .dacs
            .iter()
            .map(|dac| {
                dac.config
                    .target(self.engine.accepted.nets[dac.net.0].logic)
                    .map(DacRamp::initial)
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        self.initial_backup = Some(self.engine.accepted.clone());
        for (dac, ramp) in self.dacs.iter().zip(ramps) {
            dac.port.set(Some(ramp));
        }
        Ok(())
    }

    pub(super) fn deactivate(&mut self) {
        for dac in &self.dacs {
            dac.port.set(None);
        }
        if let Some(previous) = self.initial_backup.take() {
            self.engine.accepted = previous;
            self.previous = None;
            self.changes.clear();
            self.boundary = false;
        }
    }

    pub(super) fn accept_initial(&mut self) {
        self.initial_backup = None;
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
        self.adcs
            .iter()
            .map(|adc| {
                let v = voltage(adc.terminals[0])? - voltage(adc.terminals[1])?;
                if !v.is_finite() {
                    return Err(SpiceError::circuit("nonfinite differential ADC input"));
                }
                Ok(v)
            })
            .collect()
    }

    fn adc_previous(&self, adc: &AdcBinding) -> Drive {
        let driver = self.engine.network.sources[adc.source.0];
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
        for ((adc, &v0), v1) in self.adcs.iter().zip(previous).zip(voltages) {
            if let Some(crossing) = adc.thresholds.crossing(
                (start.value(), v0),
                (time, v1),
                self.adc_previous(adc).logic,
            )? {
                earliest = Some(earliest.map_or(crossing, |old: Real| old.min(crossing)));
            }
        }
        Ok(earliest)
    }

    /// Earliest end of an accepted DAC ramp.
    pub(super) fn next_ramp_boundary(&self, now: Real) -> SpiceResult<Option<Real>> {
        let mut next: Option<Real> = None;
        for dac in &self.dacs {
            if let Some(end) = dac.port.ramp()?.next_break(now) {
                next = Some(next.map_or(end, |old| old.min(end)));
            }
        }
        Ok(next.filter(|time| *time > now))
    }

    /// Process queued events only as far as the first actual DAC target
    /// change, rather than forcing an analog solve for every digital event.
    /// Reuse this state on acceptance if the analog endpoint is unchanged.
    pub(super) fn preview(&self, horizon: Real) -> SpiceResult<Option<EventPreview>> {
        if self.dacs.is_empty()
            || self
                .engine
                .next_event_time()
                .is_none_or(|next| next > horizon)
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
                for dac in &self.dacs {
                    let old = dac
                        .config
                        .target(self.engine.accepted.nets[dac.net.0].logic)?;
                    let new = dac.config.target(trial.nets[dac.net.0].logic)?;
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
        for (adc, &voltage) in self.adcs.iter().zip(&voltages) {
            let previous = self.adc_previous(adc);
            let drive = Drive::strong(adc.thresholds.sample(voltage, previous.logic)?);
            if drive != previous {
                updates.push((
                    self.engine.network.sources[adc.source.0],
                    Output {
                        drive,
                        delays: adc.delays,
                        mode: adc.mode,
                    },
                ));
            }
        }
        let events_due = self
            .engine
            .next_event_time()
            .is_some_and(|next| next <= time);
        let ramp_boundary = self.dacs.iter().try_fold(false, |found, dac| {
            let ramp = dac.port.ramp()?;
            let previous = self.engine.time();
            Ok::<_, SpiceError>(found || ramp.next_break(previous).is_some_and(|end| end <= time))
        })?;
        let mut point = MixedPoint {
            time: end,
            voltages,
            state: None,
            ramps: Vec::new(),
            changes: Vec::new(),
            boundary: ramp_boundary,
        };
        if !events_due && updates.is_empty() {
            return Ok(point);
        }
        let mut state = if let Some(preview) = preview.filter(|p| p.state.time == end) {
            point.changes = preview.changes;
            preview.state
        } else {
            let mut state = self.engine.accepted.clone();
            // Before the first event round, seed ADC drivers at t=0. During
            // transient, queued events precede the hybrid callback.
            if state.initialized {
                state.advance(
                    &self.engine.network,
                    end,
                    self.engine.limits,
                    &mut point.changes,
                )?;
            }
            state
        };
        for (driver, output) in updates {
            state.model_output(driver, output)?;
        }
        state.advance(
            &self.engine.network,
            end,
            self.engine.limits,
            &mut point.changes,
        )?;
        state.charge_work(self.engine.accepted.stats, self.engine.limits.work)?;
        for dac in &self.dacs {
            let logic = state.nets[dac.net.0].logic;
            let previous_ramp = dac.port.ramp()?;
            let ramp = if end == Time(0) {
                DacRamp::initial(dac.config.target(logic)?)
            } else {
                dac.config.retarget(previous_ramp, time, logic)?
            };
            point.boundary |= !ramp.same_target(previous_ramp);
            point.ramps.push(ramp);
        }
        point.state = Some(state);
        Ok(point)
    }

    pub(super) fn initial_changed(&self, point: &MixedPoint) -> SpiceResult<bool> {
        for (dac, ramp) in self.dacs.iter().zip(&point.ramps) {
            if dac.port.ramp()?.value(0.).to_bits() != ramp.value(0.).to_bits() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Update the initial linear source RHS to match the current mixed bias
    /// candidate; the companion system's topology and A/E remain immutable.
    pub(super) fn initial_rhs(&self, circuit: &Circuit, rhs: &mut Vector) -> SpiceResult<()> {
        for dac in &self.dacs {
            let index = circuit
                .devices()
                .iter()
                .position(|device| device.name().eq_ignore_ascii_case(&dac.name))
                .ok_or_else(|| SpiceError::circuit("missing DAC device"))?;
            let row = circuit
                .branch_rows(index)
                .and_then(|mut rows| rows.next())
                .ok_or_else(|| SpiceError::circuit("missing DAC branch row"))?;
            let value = dac.port.ramp()?.value(0.);
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
            for (dac, ramp) in self.dacs.iter().zip(point.ramps) {
                dac.port.set(Some(ramp));
            }
        } else if self.engine.accepted.time < point.time {
            self.engine.accepted.time = point.time;
            self.engine.accepted.delta = 0;
        }
        self.previous = Some((point.time, point.voltages));
        self.changes.extend(point.changes);
        self.boundary = point.boundary;
    }
}
