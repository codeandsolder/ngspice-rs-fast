//! Native digital code-model semantics, independent of the event scheduler.
//! Truth tables follow upstream XSPICE `d_and`, `d_or`, `d_xor` and related gates.

use crate::primitives::{Real, SpiceError, SpiceResult};
pub mod analog;
pub mod rational;
mod sequential_timing;
pub mod source;
mod source_logic;
mod timed_switch;
mod timed_switch_device;

/// Three-valued digital logic.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Logic {
    /// Logical zero.
    Zero,
    /// Logical one.
    One,
    /// Unknown or conflicting state.
    #[default]
    Unknown,
}

impl Logic {
    /// Three-valued inversion.
    #[must_use = "the returned logic value is inverted without changing the original"]
    pub const fn invert(self) -> Self {
        match self {
            Self::Zero => Self::One,
            Self::One => Self::Zero,
            Self::Unknown => Self::Unknown,
        }
    }
}

/// Drive strength, in increasing resolution priority.
#[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd)]
pub enum Strength {
    /// Disconnected output.
    #[default]
    HighImpedance,
    /// Resistive output.
    Resistive,
    /// Strong output.
    Strong,
}

/// State and strength of a digital output.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Drive {
    /// Logic state.
    pub logic: Logic,
    /// Drive strength.
    pub strength: Strength,
}

impl Drive {
    /// Strong output.
    #[must_use = "the returned drive participates in strength-aware net resolution"]
    pub const fn strong(logic: Logic) -> Self {
        Self {
            logic,
            strength: Strength::Strong,
        }
    }

    /// Resolve concurrent drives, ignoring disconnected outputs.
    #[must_use = "the returned drive is the resolved result of all concurrent outputs"]
    pub fn resolve(drives: impl IntoIterator<Item = Self>) -> Self {
        drives
            .into_iter()
            .filter(|d| d.strength != Strength::HighImpedance)
            .fold(Self::default(), |a, b| match a.strength.cmp(&b.strength) {
                std::cmp::Ordering::Less => b,
                std::cmp::Ordering::Greater => a,
                std::cmp::Ordering::Equal if a.logic == b.logic => a,
                std::cmp::Ordering::Equal => Self {
                    logic: Logic::Unknown,
                    strength: a.strength,
                },
            })
    }
}

/// Delayed output scheduling policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DelayMode {
    /// Preserve transitions in delivery order; an earlier newly posted
    /// output supersedes later pending transitions (upstream `EVTqueue_output`).
    Transport,
    /// Replace an outstanding transition when the desired value changes.
    Inertial,
}

/// Validated propagation delays. Two-state models use rise/fall; a
/// disconnectable driver can specify four additional high-impedance paths.
#[derive(Clone, Copy, Debug)]
pub struct Delays {
    rise: Real,
    fall: Real,
    impedance: Option<ImpedanceDelays>,
}

#[derive(Clone, Copy, Debug)]
struct ImpedanceDelays {
    low_to_z: Real,
    high_to_z: Real,
    z_to_low: Real,
    z_to_high: Real,
}

impl Delays {
    pub(super) const ZERO: Self = Self {
        rise: 0.,
        fall: 0.,
        impedance: None,
    };

    /// Construct finite, nonnegative delays.
    ///
    /// # Errors
    /// Invalid delay.
    pub fn new(rise: Real, fall: Real) -> SpiceResult<Self> {
        if !rise.is_finite() || !fall.is_finite() || rise < 0. || fall < 0. {
            return Err(SpiceError::circuit("invalid digital propagation delay"));
        }
        Ok(Self {
            rise,
            fall,
            impedance: None,
        })
    }

    /// Construct six independent physical transition paths in the order
    /// low→high, high→low, low→Z, high→Z, Z→low, Z→high.
    ///
    /// # Errors
    /// Any delay is negative or nonfinite.
    pub fn paths(values: [Real; 6]) -> SpiceResult<Self> {
        let [rise, fall, low_to_z, high_to_z, z_to_low, z_to_high] = values;
        let mut delays = Self::new(rise, fall)?;
        if [low_to_z, high_to_z, z_to_low, z_to_high]
            .into_iter()
            .any(|value| !value.is_finite() || value < 0.)
        {
            return Err(SpiceError::circuit(
                "invalid high-impedance propagation delay",
            ));
        }
        delays.impedance = Some(ImpedanceDelays {
            low_to_z,
            high_to_z,
            z_to_low,
            z_to_high,
        });
        Ok(delays)
    }

    /// Select the physical transition using both logic and drive strength.
    /// An unknown endpoint of an explicitly specified Z path uses the later
    /// of its two possible known transitions. Ordinary XSPICE rise/fall
    /// timing retains the previous-state rule for unknown transitions.
    #[must_use = "the returned delay determines the scheduled transition time"]
    pub fn transition(self, previous: Drive, next: Drive) -> Real {
        if let Some(paths) = self.impedance {
            let old_z = previous.strength == Strength::HighImpedance;
            let new_z = next.strength == Strength::HighImpedance;
            match (old_z, new_z) {
                (true, true) => return 0.,
                (false, true) => {
                    return match previous.logic {
                        Logic::Zero => paths.low_to_z,
                        Logic::One => paths.high_to_z,
                        Logic::Unknown => paths.low_to_z.max(paths.high_to_z),
                    };
                }
                (true, false) => {
                    return match next.logic {
                        Logic::Zero => paths.z_to_low,
                        Logic::One => paths.z_to_high,
                        Logic::Unknown => paths.z_to_low.max(paths.z_to_high),
                    };
                }
                (false, false) => {}
            }
        }
        match next.logic {
            Logic::One => self.rise,
            Logic::Unknown if previous.logic == Logic::Zero => self.rise,
            Logic::Zero | Logic::Unknown => self.fall,
        }
    }
}

/// Desired output and its propagation policy.
#[derive(Clone, Copy, Debug)]
pub struct Output {
    /// New drive.
    pub drive: Drive,
    /// Propagation delays.
    pub delays: Delays,
    /// Scheduling mode.
    pub mode: DelayMode,
}

/// Event execution context. At time zero models settle the initial bias;
/// propagation delays apply only after initialization.
#[derive(Clone, Copy, Debug)]
pub struct EventContext {
    /// Physical event time.
    pub time: Real,
    /// Same-time delta cycle.
    pub delta: usize,
    /// True during initial operating-point settlement.
    pub initial: bool,
}

/// A diagnostic emitted by a successfully evaluated digital model. The
/// event engine assigns its model identity and physical timestamp at acceptance.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDiagnostic {
    /// Whether this diagnostic prevents accepting the trial.
    pub fatal: bool,
    /// Type of timing constraint (setup, hold, pulse width, recovery).
    pub kind: &'static str,
    /// Zero-based input port involved in the constraint.
    pub input: usize,
    /// Observed interval in seconds.
    pub elapsed: Real,
    /// Required minimum interval in seconds.
    pub minimum: Real,
}

/// Number of ordered input and output ports in one signal domain.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PortCount {
    /// Input ports.
    pub inputs: usize,
    /// Output ports.
    pub outputs: usize,
}

/// Native code-model terminal schema. Each analog voltage port is a
/// differential terminal pair; each digital port is one resolved event net.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ModelSchema {
    /// Analog voltage ports.
    pub analog_voltage: PortCount,
    /// Digital event ports.
    pub digital: PortCount,
}

impl ModelSchema {
    /// A model with digital ports only.
    #[must_use = "the returned schema defines the digital port counts"]
    pub const fn digital(inputs: usize, outputs: usize) -> Self {
        Self {
            analog_voltage: PortCount {
                inputs: 0,
                outputs: 0,
            },
            digital: PortCount { inputs, outputs },
        }
    }
}

/// Result of advancing an externally clocked simulator with unchanged inputs.
///
/// An output event stops the advance at its physical time; otherwise the
/// simulator has reached the requested horizon without changing its outputs.
#[derive(Debug)]
pub enum AutonomousAdvance {
    /// No externally visible event through the horizon.
    Quiet,
    /// First settled output change.
    Event {
        /// Physical time of the output event.
        time: Real,
        /// Complete output values at this time.
        outputs: Vec<Output>,
    },
}

/// Immutable native digital model with typed, cloneable state. State must
/// not share mutable data between clones; the scheduler owns trial copies.
pub trait CodeModel: std::fmt::Debug + Send + Sync + 'static {
    /// Model-defined state, including arbitrary counters and numerical data.
    type State: Clone + std::fmt::Debug + Send + Sync + 'static;
    /// Ordered ports and their signal domains.
    fn schema(&self) -> ModelSchema;
    /// Initial model-defined state.
    fn initial_state(&self) -> Self::State;
    /// Absolute time of the next autonomous evaluation, if any. A time equal
    /// to the current time schedules the next delta cycle. The scheduler
    /// replaces the previous wakeup after each evaluation.
    fn next_wakeup(&self, _state: &Self::State) -> Option<Real> {
        None
    }
    /// Whether this model must be probed through a time horizon because its
    /// simulator cannot directly query the next autonomous event.
    fn requires_lookahead(&self) -> bool {
        false
    }

    /// Advance a disposable model state with its inputs held constant.
    /// A returned event must be strictly after `now` and at or before
    /// `horizon`. A quiet result must reach the horizon.
    ///
    /// # Errors
    /// Simulator failure, or unsupported autonomous lookahead.
    fn advance_autonomous(
        &self,
        _now: Real,
        _horizon: Real,
        _state: &mut Self::State,
    ) -> SpiceResult<AutonomousAdvance> {
        Err(SpiceError::circuit(
            "model does not support autonomous lookahead",
        ))
    }

    /// Drain diagnostics produced by the most recent input evaluation. The
    /// engine records these in the same disposable transaction as outputs.
    fn take_diagnostics(&self, _state: &mut Self::State) -> Vec<ModelDiagnostic> {
        Vec::new()
    }

    /// Evaluate the model against resolved inputs.
    ///
    /// # Errors
    /// Invalid inputs or state.
    fn evaluate(
        &self,
        context: EventContext,
        inputs: &[Drive],
        state: &mut Self::State,
    ) -> SpiceResult<Vec<Output>>;
}

/// Basic variadic gate kinds.
#[derive(Clone, Copy, Debug)]
pub enum GateKind {
    /// AND.
    And,
    /// OR.
    Or,
    /// XOR.
    Xor,
    /// NAND.
    Nand,
    /// NOR.
    Nor,
    /// XNOR.
    Xnor,
    /// Buffer.
    Buffer,
    /// Inverter.
    Inverter,
}

impl GateKind {
    fn evaluate(self, inputs: &[Drive]) -> Logic {
        let has = |value| inputs.iter().any(|drive| drive.logic == value);
        let and = || {
            if has(Logic::Zero) {
                Logic::Zero
            } else if has(Logic::Unknown) {
                Logic::Unknown
            } else {
                Logic::One
            }
        };
        let or = || {
            if has(Logic::One) {
                Logic::One
            } else if has(Logic::Unknown) {
                Logic::Unknown
            } else {
                Logic::Zero
            }
        };
        let xor = || {
            if has(Logic::Unknown) {
                Logic::Unknown
            } else if inputs.iter().filter(|d| d.logic == Logic::One).count() % 2 == 1 {
                Logic::One
            } else {
                Logic::Zero
            }
        };
        match self {
            Self::And => and(),
            Self::Or => or(),
            Self::Xor => xor(),
            Self::Nand => and().invert(),
            Self::Nor => or().invert(),
            Self::Xnor => xor().invert(),
            Self::Buffer => inputs[0].logic,
            Self::Inverter => inputs[0].logic.invert(),
        }
    }
}

/// Variadic gate with a single strong output.
#[derive(Debug)]
pub struct Gate {
    kind: GateKind,
    inputs: usize,
    delays: Delays,
    mode: DelayMode,
}

impl Gate {
    /// Construct a gate.
    ///
    /// # Errors
    /// Invalid arity.
    pub fn new(
        kind: GateKind,
        inputs: usize,
        delays: Delays,
        mode: DelayMode,
    ) -> SpiceResult<Self> {
        if inputs == 0 || (matches!(kind, GateKind::Buffer | GateKind::Inverter) && inputs != 1) {
            return Err(SpiceError::circuit("invalid digital gate arity"));
        }
        Ok(Self {
            kind,
            inputs,
            delays,
            mode,
        })
    }
}

impl CodeModel for Gate {
    type State = ();
    fn schema(&self) -> ModelSchema {
        ModelSchema::digital(self.inputs, 1)
    }
    fn initial_state(&self) -> Self::State {}
    fn evaluate(
        &self,
        _context: EventContext,
        inputs: &[Drive],
        _state: &mut Self::State,
    ) -> SpiceResult<Vec<Output>> {
        if inputs.len() != self.inputs {
            return Err(SpiceError::circuit("digital gate dimension mismatch"));
        }
        Ok(vec![Output {
            drive: Drive::strong(self.kind.evaluate(inputs)),
            delays: self.delays,
            mode: self.mode,
        }])
    }
}

/// Positive-edge D flip-flop, with D and clock inputs and Q/complement outputs.
/// As in upstream `d_dff`, a change to ONE captures D; changes to ZERO
/// or UNKNOWN retain the previous state.
#[derive(Debug)]
pub struct DFlipFlop {
    /// Initial Q.
    pub initial: Logic,
    /// Propagation delays.
    pub delays: Delays,
    /// Propagation mode.
    pub mode: DelayMode,
}

impl CodeModel for DFlipFlop {
    type State = [Logic; 2];
    fn schema(&self) -> ModelSchema {
        ModelSchema::digital(2, 2)
    }
    fn initial_state(&self) -> Self::State {
        [Logic::Unknown, self.initial]
    }
    fn evaluate(
        &self,
        context: EventContext,
        inputs: &[Drive],
        state: &mut Self::State,
    ) -> SpiceResult<Vec<Output>> {
        let [data, clock] = inputs else {
            return Err(SpiceError::circuit("flip-flop input dimension mismatch"));
        };
        let [previous_clock, q] = state;
        if context.initial {
            *q = self.initial;
        } else if *previous_clock != Logic::One && clock.logic == Logic::One {
            *q = data.logic;
        }
        *previous_clock = clock.logic;
        Ok([*q, q.invert()]
            .map(|logic| Output {
                drive: Drive::strong(logic),
                delays: self.delays,
                mode: self.mode,
            })
            .to_vec())
    }
}

/// Single-channel ADC thresholds; low > high selects Schmitt hysteresis
/// (upstream `adc_bridge/cfunc.mod`).
#[derive(Clone, Copy, Debug)]
pub struct Adc {
    low: Real,
    high: Real,
    qualification: Option<ReceiverQualification>,
}

#[derive(Clone, Copy, Debug)]
struct ReceiverQualification {
    initial: Logic,
    inverted: bool,
}

impl Adc {
    /// One differential voltage input and one digital output.
    #[must_use = "the returned schema declares the ADC voltage input and digital output"]
    pub const fn schema() -> ModelSchema {
        ModelSchema {
            analog_voltage: PortCount {
                inputs: 1,
                outputs: 0,
            },
            digital: PortCount {
                inputs: 0,
                outputs: 1,
            },
        }
    }

    /// Construct finite thresholds.
    ///
    /// # Errors
    /// Nonfinite threshold.
    pub fn new(low: Real, high: Real) -> SpiceResult<Self> {
        if !low.is_finite() || !high.is_finite() {
            return Err(SpiceError::circuit("nonfinite ADC threshold"));
        }
        Ok(Self {
            low,
            high,
            qualification: None,
        })
    }

    /// Construct a qualified binary receiver. Equality at either threshold
    /// retains history; in the initial operating point the declared hint
    /// replaces history. The inversion is applied only on a decisive sample.
    ///
    /// # Errors
    /// Invalid threshold order or nonbinary initial state.
    pub fn qualified(low: Real, high: Real, initial: Logic, inverted: bool) -> SpiceResult<Self> {
        if !low.is_finite() || !high.is_finite() || low > high || initial == Logic::Unknown {
            return Err(SpiceError::circuit("invalid qualified receiver"));
        }
        Ok(Self {
            low,
            high,
            qualification: Some(ReceiverQualification { initial, inverted }),
        })
    }

    pub(crate) const fn is_qualified(self) -> bool {
        self.qualification.is_some()
    }

    pub(crate) fn sample_at(
        self,
        voltage: Real,
        previous: Logic,
        initial: bool,
    ) -> SpiceResult<Logic> {
        if let Some(qualification) = self.qualification {
            if !voltage.is_finite() {
                return Err(SpiceError::circuit("nonfinite qualified receiver input"));
            }
            let decisive = if voltage > self.high {
                Some(Logic::One)
            } else if voltage < self.low {
                Some(Logic::Zero)
            } else {
                None
            };
            return Ok(decisive.map_or_else(
                || {
                    if initial || previous == Logic::Unknown {
                        qualification.initial
                    } else {
                        previous
                    }
                },
                |bit| {
                    if qualification.inverted {
                        bit.invert()
                    } else {
                        bit
                    }
                },
            ));
        }
        self.sample(voltage, previous)
    }

    /// Evaluate a voltage against the previous accepted state.
    ///
    /// # Errors
    /// Nonfinite voltage.
    pub fn sample(self, voltage: Real, previous: Logic) -> SpiceResult<Logic> {
        if self.is_qualified() {
            return self.sample_at(voltage, previous, false);
        }
        if !voltage.is_finite() {
            return Err(SpiceError::circuit("nonfinite ADC input"));
        }
        Ok(if voltage >= self.low.max(self.high) {
            Logic::One
        } else if voltage <= self.low.min(self.high) {
            Logic::Zero
        } else if self.low > self.high {
            previous
        } else {
            Logic::Unknown
        })
    }

    /// Candidate threshold inside a linearly interpolated analog trial.
    /// A driver must solve again at the proposed time, not accept an
    /// interpolated voltage as a solved circuit state.
    ///
    /// # Errors
    /// Invalid time interval or nonfinite samples.
    pub fn crossing(
        self,
        start: (Real, Real),
        end: (Real, Real),
        previous: Logic,
    ) -> SpiceResult<Option<Real>> {
        let (t0, v0) = start;
        let (t1, v1) = end;
        if ![t0, v0, t1, v1].into_iter().all(Real::is_finite) || t0 < 0. || t1 <= t0 {
            return Err(SpiceError::circuit("invalid ADC crossing interval"));
        }
        if let Some(qualification) = self.qualification {
            return self.qualified_crossing(start, end, previous, qualification);
        }
        // A transition into an open voltage interval may start at its
        // threshold (e.g. zero to unknown on a rising ordinary ADC input).
        // In that case the driver must solve again just beyond the accepted
        // point, rather than delay the transition until the next large step.
        let adjacent = if v1 > v0 {
            v0.next_up().min(v1)
        } else if v1 < v0 {
            v0.next_down().max(v1)
        } else {
            return Ok(None);
        };
        if self.sample(v0, previous)? == previous && self.sample(adjacent, previous)? != previous {
            return Ok(Some(t0));
        }
        let thresholds = if self.low > self.high {
            if v1 > v0 {
                if previous == Logic::One {
                    return Ok(None);
                }
                [self.low; 2]
            } else {
                if previous == Logic::Zero {
                    return Ok(None);
                }
                [self.high; 2]
            }
        } else {
            [self.low, self.high]
        };
        let threshold = if v1 > v0 {
            thresholds
                .into_iter()
                .filter(|&v| v > v0 && v < v1)
                .min_by(Real::total_cmp)
        } else {
            thresholds
                .into_iter()
                .filter(|&v| v < v0 && v > v1)
                .max_by(Real::total_cmp)
        };
        Ok(threshold.and_then(|v| crossing_time(start, end, v)))
    }

    fn qualified_crossing(
        self,
        start: (Real, Real),
        end: (Real, Real),
        previous: Logic,
        qualification: ReceiverQualification,
    ) -> SpiceResult<Option<Real>> {
        let (t0, v0) = start;
        let (t1, v1) = end;
        if self.sample(v1, previous)? == previous {
            return Ok(None);
        }
        let physical = if qualification.inverted {
            previous.invert()
        } else {
            previous
        };
        let threshold = match physical {
            Logic::Zero if v0 <= self.high && v1 > self.high => self.high,
            Logic::One if v0 >= self.low && v1 < self.low => self.low,
            _ => return Ok(None),
        };
        // At equality no transition has happened. Land just past the
        // interpolated threshold; the analog solver still verifies the bit.
        Ok(crossing_time(start, end, threshold).map(|time| time.next_up().max(t0).min(t1)))
    }
}

fn crossing_time(start: (Real, Real), end: (Real, Real), threshold: Real) -> Option<Real> {
    let (t0, v0) = start;
    let (t1, v1) = end;
    let scale = v0.abs().max(v1.abs());
    // Prefer unscaled differences when representable, to retain precision
    // for small excursions around large biases.
    let span = v1 - v0;
    let fraction = if span.is_finite() {
        (threshold - v0) / span
    } else {
        (threshold / scale - v0 / scale) / (v1 / scale - v0 / scale)
    };
    let time = fraction.mul_add(t1 - t0, t0);
    (time >= t0 && time < t1).then_some(time)
}

/// DAC voltage targets, including the unknown state.
#[derive(Clone, Copy, Debug)]
pub struct Dac {
    /// Logic zero target.
    pub low: Real,
    /// Logic one target.
    pub high: Real,
    /// Unknown target.
    pub unknown: Real,
}

impl Dac {
    /// One digital input and one differential voltage output.
    #[must_use = "the returned schema declares the DAC digital input and voltage output"]
    pub const fn schema() -> ModelSchema {
        ModelSchema {
            analog_voltage: PortCount {
                inputs: 0,
                outputs: 1,
            },
            digital: PortCount {
                inputs: 1,
                outputs: 0,
            },
        }
    }

    /// Convert a digital state to an analog target.
    ///
    /// # Errors
    /// Nonfinite target.
    pub fn target(self, logic: Logic) -> SpiceResult<Real> {
        let value = match logic {
            Logic::Zero => self.low,
            Logic::One => self.high,
            Logic::Unknown => self.unknown,
        };
        if !value.is_finite() {
            return Err(SpiceError::circuit("nonfinite DAC output"));
        }
        Ok(value)
    }
}

/// Voltage DAC configuration. Rise and fall are full-scale transition times
/// (`dac_bridge/cfunc.mod`); a partially completed transition slews from the
/// present output, not from the previous target.
#[derive(Clone, Copy, Debug)]
pub struct DacConfig {
    levels: Dac,
    slew: Option<(Delays, Real)>,
}

impl DacConfig {
    /// Validate levels and full-scale transition times.
    ///
    /// # Errors
    /// Invalid voltage range, unknown-state level, or impedance timing on
    /// a voltage-only bridge.
    pub fn new(levels: Dac, slew: Delays) -> SpiceResult<Self> {
        if slew.impedance.is_some() {
            return Err(SpiceError::circuit(
                "voltage DAC requires rise/fall slew times, not impedance delays",
            ));
        }
        let span = levels.high - levels.low;
        if !span.is_finite()
            || span <= 0.
            || !levels.unknown.is_finite()
            || levels.unknown < levels.low
            || levels.unknown > levels.high
        {
            return Err(SpiceError::circuit("invalid DAC voltage range"));
        }
        Ok(Self {
            levels,
            slew: Some((slew, span)),
        })
    }

    /// Ideal voltage bridge with independently specified rails. Equal and
    /// inverted rails are valid; unknown logic has the explicit midpoint law.
    ///
    /// # Errors
    /// Nonfinite voltage.
    pub fn ideal(low: Real, high: Real) -> SpiceResult<Self> {
        if !low.is_finite() || !high.is_finite() {
            return Err(SpiceError::circuit("nonfinite ideal logic voltage rail"));
        }
        Ok(Self {
            levels: Dac {
                low,
                high,
                unknown: low.midpoint(high),
            },
            slew: None,
        })
    }

    pub(crate) fn target(self, logic: Logic) -> SpiceResult<Real> {
        self.levels.target(logic)
    }

    pub(crate) fn retarget(self, ramp: DacRamp, time: Real, logic: Logic) -> SpiceResult<DacRamp> {
        let target = self.target(logic)?;
        if ramp.target.to_bits() == target.to_bits() {
            return Ok(ramp);
        }
        let Some((slew, span)) = self.slew else {
            return Ok(DacRamp {
                start: time,
                end: time,
                from: target,
                target,
            });
        };
        let from = ramp.value(time);
        let duration =
            ((target - from).abs() / span) * if target > from { slew.rise } else { slew.fall };
        let end = time + duration;
        if !end.is_finite() || (duration > 0. && end <= time) {
            return Err(SpiceError::circuit(
                "DAC transition outside representable time range",
            ));
        }
        Ok(DacRamp {
            start: time,
            end,
            from,
            target,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DacRamp {
    start: Real,
    end: Real,
    from: Real,
    target: Real,
}

impl DacRamp {
    pub(crate) const fn initial(target: Real) -> Self {
        Self {
            start: 0.,
            end: 0.,
            from: target,
            target,
        }
    }

    pub(crate) fn value(self, time: Real) -> Real {
        if time >= self.end {
            self.target
        } else if time <= self.start {
            self.from
        } else {
            ((time - self.start) / (self.end - self.start))
                .mul_add(self.target - self.from, self.from)
        }
    }

    pub(crate) const fn same_target(self, other: Self) -> bool {
        self.target.to_bits() == other.target.to_bits()
    }

    pub(crate) const fn next_break(self, time: Real) -> Option<Real> {
        if self.end > time {
            Some(self.end)
        } else {
            None
        }
    }
}

/// The mixed-signal analysis owns this accepted forcing signal; stamps only
/// read it. An unbound port cannot accidentally run as an ordinary source.
#[derive(Clone, Debug, Default)]
pub(crate) struct DacPort(std::rc::Rc<std::cell::Cell<Option<DacRamp>>>);

impl DacPort {
    pub(crate) fn set(&self, ramp: Option<DacRamp>) {
        self.0.set(ramp);
    }

    pub(crate) fn ramp(&self) -> SpiceResult<DacRamp> {
        self.0
            .get()
            .ok_or_else(|| SpiceError::circuit("DAC requires an active mixed-signal transient"))
    }

    pub(crate) fn same(&self, other: &Self) -> bool {
        std::rc::Rc::ptr_eq(&self.0, &other.0)
    }
}

/// Ideal differential voltage output driven by the accepted DAC ramp.
/// Constructed and bound by the analysis layer, not by the netlist parser.
#[derive(Debug)]
pub(crate) struct DacSource {
    pub(crate) name: String,
    pub(crate) terminals: [crate::primitives::NodeId; 2],
    pub(crate) port: DacPort,
    pub(crate) native: Option<source::NativeVoltageBinding>,
}

impl crate::devices::Device for DacSource {
    fn model_name(&self) -> Option<&str> {
        self.native.as_ref().map(|binding| binding.model.as_str())
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'a'
    }
    fn terminals(&self) -> &[crate::primitives::NodeId] {
        &self.terminals
    }
    fn branch_currents(&self) -> usize {
        1
    }
    fn findable_branch(&self) -> Option<usize> {
        Some(0)
    }
    fn depends_on_time(&self) -> bool {
        true
    }

    fn initial_voltage_relations(
        &self,
        _inductors_short: bool,
    ) -> Vec<crate::devices::InitialVoltageRelation> {
        vec![crate::devices::InitialVoltageRelation {
            terminals: self.terminals,
            law: crate::devices::InitialVoltageLaw::Source(0),
        }]
    }

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let branch = setup.branch(0)?;
        setup.voltage_branch(self.terminals[0], self.terminals[1], branch);
        Ok(true)
    }

    fn stamp(&self, context: &mut crate::devices::StampContext<'_>) -> SpiceResult<()> {
        use crate::devices::traits::AnalysisMode;
        let ramp = self.port.ramp()?;
        let value = match context.mode {
            AnalysisMode::OperatingPoint => ramp.value(0.) * context.source_factor,
            AnalysisMode::Transient { time, .. } => ramp.value(time),
            _ => {
                return Err(SpiceError::not_yet_ported(
                    "DAC outside mixed-signal transient",
                    "src/xspice/icm/digital/dac_bridge/cfunc.mod",
                ));
            }
        };
        let branch = context.branch(0)?;
        crate::devices::linear::branch_stamp(
            context.matrix,
            context.unknowns,
            self.terminals,
            branch,
        )?;
        context.rhs.add_to(branch, value)
    }

    fn assemble_linear(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
    ) -> SpiceResult<()> {
        use crate::devices::linear::{LinearSource, SourceRows, Waveform};
        use crate::primitives::Complex;
        let value = self.port.ramp()?.value(0.);
        let mut rows = SourceRows::new();
        rows.push(context.branch(self.terminals)?, 1.)?;
        context.system.sources.push(LinearSource {
            name: std::sync::Arc::new(self.name.clone()),
            kind: crate::devices::SourceKind::Voltage,
            rows,
            dc: value,
            ac: Complex::ZERO,
            waveform: std::sync::Arc::new(Waveform::Constant(value)),
        });
        Ok(())
    }
}
