//! Source-level native code-model schemas. Port defaults, parameter shapes and
//! vector dimensions are shared by parsing, normalized emission and binding.
use crate::netlist::ast::{
    CodePortKind, DeviceInstance, ModelCard, ParameterAssignment, ParameterKind,
};
use crate::primitives::{Real, SpiceError, SpiceResult, parse_spice_number};

/// Registered source-level A-instance model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceModel {
    /// Rational voltage/current transfer.
    RationalTransfer,
    /// Tabulated complex AC transfer.
    FrequencyTransfer,
    /// Qualified analog receivers with event logic output.
    LogicEvent,
    /// Ordered differential-voltage ranges with independent occurrence history.
    LogicRange,
    /// Ideal digital-to-voltage bridge.
    LogicVoltage,
    /// Digital drive-selected passive conductance.
    LogicConductance,
    /// Timed conductance switch.
    TimedSwitch,
    /// Cubic logarithmic or linear PSpice-style switch.
    PSwitch,
    /// Exponential or linear analog switch.
    ASwitch,
}

/// Static parameter representation accepted by a source model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterShape {
    /// Finite real or scoped scalar expression.
    Real,
    /// Exact integer or scoped scalar expression.
    Integer,
    /// Boolean literal or scoped scalar expression producing 0 or 1.
    Boolean,
    /// Vector of finite numeric coefficients.
    RealVector,
    /// Vector of exact integer values.
    IntegerVector,
    /// Vector of boolean values.
    BooleanVector,
}

impl ParameterShape {
    /// Whether a bracketed array is required.
    #[must_use = "the shape determines whether a parameter requires bracketed grouping"]
    pub const fn vector(self) -> bool {
        matches!(
            self,
            Self::RealVector | Self::IntegerVector | Self::BooleanVector
        )
    }

    /// Whether literal boolean spellings are accepted.
    #[must_use = "the result determines whether boolean spellings are valid"]
    pub const fn boolean(self) -> bool {
        matches!(self, Self::Boolean | Self::BooleanVector)
    }
}

/// One ordered scalar/vector port declaration.
#[derive(Clone, Copy, Debug)]
pub struct PortSpec {
    /// Kind for unmarked terminals.
    pub default: CodePortKind,
    /// Explicitly permitted terminal types.
    pub allowed: &'static [CodePortKind],
    /// Required bracket grouping.
    pub vector: bool,
    /// Whether inversion is meaningful on this port.
    pub inversion: bool,
}

use CodePortKind::{
    BranchCurrent, Conductance, ConductanceDifferential, Current, CurrentDifferential, Digital,
    Voltage, VoltageDifferential,
};
const ANALOG: &[CodePortKind] = &[Voltage, VoltageDifferential];
const TRANSFER: &[CodePortKind] = &[Voltage, VoltageDifferential, Current, CurrentDifferential];
const EVENT: &[CodePortKind] = &[Digital];
const LOGIC_INPUT: &[CodePortKind] = &[Voltage, VoltageDifferential, Digital];
const BRANCH: &[CodePortKind] = &[ConductanceDifferential];

impl PortSpec {
    const fn new(
        default: CodePortKind,
        allowed: &'static [CodePortKind],
        vector: bool,
        inversion: bool,
    ) -> Self {
        Self {
            default,
            allowed,
            vector,
            inversion,
        }
    }
}

impl SourceModel {
    /// Registered model type, without applying node aliasing.
    #[must_use = "the lookup result identifies the registered source model or an unsupported name"]
    pub const fn from_base(base: &str) -> Option<Self> {
        if base.eq_ignore_ascii_case("s_xfer") {
            Some(Self::RationalTransfer)
        } else if base.eq_ignore_ascii_case("xfer") {
            Some(Self::FrequencyTransfer)
        } else if base.eq_ignore_ascii_case("preprocess_logic_event") {
            Some(Self::LogicEvent)
        } else if base.eq_ignore_ascii_case("preprocess_logic_range") {
            Some(Self::LogicRange)
        } else if base.eq_ignore_ascii_case("preprocess_logic_voltage") {
            Some(Self::LogicVoltage)
        } else if base.eq_ignore_ascii_case("preprocess_logic_conductance") {
            Some(Self::LogicConductance)
        } else if base.eq_ignore_ascii_case("preprocess_timed_switch") {
            Some(Self::TimedSwitch)
        } else if base.eq_ignore_ascii_case("pswitch") {
            Some(Self::PSwitch)
        } else if base.eq_ignore_ascii_case("aswitch") {
            Some(Self::ASwitch)
        } else {
            None
        }
    }

    /// Ordered terminal groups and their default domains.
    #[must_use = "the port specifications are required to bind terminal groups in source order"]
    pub const fn ports(self) -> &'static [PortSpec] {
        const TRANSFER_PORTS: &[PortSpec] = &[
            PortSpec::new(Voltage, TRANSFER, false, false),
            PortSpec::new(Voltage, TRANSFER, false, false),
        ];
        const LOGIC_PORTS: &[PortSpec] = &[
            PortSpec::new(Voltage, LOGIC_INPUT, true, true),
            PortSpec::new(Digital, EVENT, false, false),
        ];
        const RANGE_PORTS: &[PortSpec] = &[
            PortSpec::new(Voltage, ANALOG, false, false),
            PortSpec::new(Digital, EVENT, false, false),
        ];
        const VOLTAGE_PORTS: &[PortSpec] = &[
            PortSpec::new(Digital, EVENT, false, true),
            PortSpec::new(Voltage, ANALOG, false, false),
        ];
        const CONDUCTANCE_PORTS: &[PortSpec] = &[
            PortSpec::new(Digital, EVENT, false, true),
            PortSpec::new(ConductanceDifferential, BRANCH, false, false),
        ];
        const SWITCH_PORTS: &[PortSpec] = &[
            PortSpec::new(VoltageDifferential, ANALOG, false, false),
            PortSpec::new(ConductanceDifferential, BRANCH, false, false),
        ];
        const PSWITCH_PORTS: &[PortSpec] = &[
            PortSpec::new(
                ConductanceDifferential,
                &[Conductance, ConductanceDifferential],
                false,
                false,
            ),
            PortSpec::new(ConductanceDifferential, BRANCH, false, false),
        ];
        const ASWITCH_PORTS: &[PortSpec] = &[
            PortSpec::new(
                Voltage,
                &[
                    Voltage,
                    VoltageDifferential,
                    Current,
                    CurrentDifferential,
                    BranchCurrent,
                ],
                false,
                false,
            ),
            PortSpec::new(ConductanceDifferential, BRANCH, false, false),
        ];
        match self {
            Self::PSwitch => PSWITCH_PORTS,
            Self::ASwitch => ASWITCH_PORTS,
            Self::RationalTransfer | Self::FrequencyTransfer => TRANSFER_PORTS,
            Self::LogicEvent => LOGIC_PORTS,
            Self::LogicRange => RANGE_PORTS,
            Self::LogicVoltage => VOLTAGE_PORTS,
            Self::LogicConductance => CONDUCTANCE_PORTS,
            Self::TimedSwitch => SWITCH_PORTS,
        }
    }

    /// Shape of a declared static setter. Unknown setters are rejected during
    /// model binding, after source expressions have been evaluated.
    #[must_use = "the shape determines how the static setter must be interpreted"]
    pub fn parameter(self, name: &str) -> Option<ParameterShape> {
        use ParameterShape::{Boolean, BooleanVector, Integer, IntegerVector, Real, RealVector};
        match self {
            Self::RationalTransfer => match name {
                "num_coeff" | "den_coeff" | "int_ic" => Some(RealVector),
                "gain" | "denormalized_freq" | "in_offset" => Some(Real),
                _ => None,
            },
            Self::FrequencyTransfer => match name {
                "table" => Some(RealVector),
                "r_i" | "db" | "rad" => Some(Boolean),
                "span" | "offset" => Some(Integer),
                _ => None,
            },
            Self::LogicEvent => match name {
                "operation" | "ic" | "strength" | "override_margin" | "timing_violation" => {
                    Some(Integer)
                }
                "delay" | "low" | "high" | "output_capacitance" | "pulse_reject" => Some(Real),
                "input_capacitance" | "load_resistance" | "delay_paths" | "delay_windows"
                | "delay_gate" | "delay_preset" | "delay_clear" | "timing_setup"
                | "timing_hold" | "timing_clock_min" | "timing_async_min" | "timing_recovery" => {
                    Some(RealVector)
                }
                "inertial" => Some(Boolean),
                "inverted" => Some(BooleanVector),
                "rx_ic" | "async_policy" | "strength_paths" => Some(IntegerVector),
                _ => None,
            },
            Self::LogicRange => match name {
                "ranges" => Some(RealVector),
                "initial_range" | "gap_policy" => Some(Integer),
                "gap_drive" => Some(IntegerVector),
                _ => None,
            },
            Self::LogicVoltage => match name {
                "out_low" | "out_high" => Some(Real),
                _ => None,
            },
            Self::PSwitch | Self::ASwitch => match name {
                "cntl_on" | "cntl_off" | "r_on" | "r_off" => Some(Real),
                "log" => Some(Boolean),
                "r_cntl_in" if self == Self::PSwitch => Some(Real),
                "limit" if self == Self::ASwitch => Some(Boolean),
                _ => None,
            },
            Self::LogicConductance => match name {
                "conductance" | "conductance_by_rank" | "resistance_duration" => Some(RealVector),
                "resistance_shape" => Some(Real),
                _ => None,
            },
            Self::TimedSwitch => match name {
                "ron" | "roff" | "vt" | "vh" | "ton" | "toff" | "td_on" | "td_off" | "timetol" => {
                    Some(Real)
                }
                "ic" => Some(Integer),
                _ => None,
            },
        }
    }

    /// Validate the complete typed connection list.
    ///
    /// # Errors
    /// Unknown domain, invalid grouping, inversion, missing or extra ports.
    pub fn validate_ports(self, instance: &DeviceInstance) -> SpiceResult<()> {
        let invalid = |message| SpiceError::parse(instance.location.clone(), message);
        let schema_ports = self.ports();
        if !instance.nodes.is_empty()
            || !instance.parameters.is_empty()
            || instance.code_ports.len() != schema_ports.len()
        {
            return Err(invalid(
                "invalid A-instance port count or instance parameters",
            ));
        }
        for (port, spec) in instance.code_ports.iter().zip(schema_ports) {
            if port.vector != spec.vector
                || port.terminals.is_empty()
                || (!spec.vector && port.terminals.len() != 1)
            {
                return Err(invalid("A-instance port grouping does not match model"));
            }
            for terminal in &port.terminals {
                let kind = terminal.kind;
                if !spec.allowed.contains(&kind)
                    || terminal.negative.is_some() != kind.differential()
                    || (terminal.inverted && (!spec.inversion || kind != Digital))
                    || terminal.positive.eq_ignore_ascii_case("null")
                    || terminal
                        .negative
                        .as_ref()
                        .is_some_and(|node| node.eq_ignore_ascii_case("null"))
                {
                    return Err(invalid("invalid A-instance terminal shape"));
                }
            }
        }
        Ok(())
    }

    /// Validate static parameter names and shapes, including all duplicate
    /// setters before the model applies their values in source order.
    ///
    /// # Errors
    /// Unknown, unresolved, nonfinite or incompatible parameter.
    pub fn validate_parameters(self, card: &ModelCard) -> SpiceResult<()> {
        for parameter in &card.parameters {
            let shape = self.parameter(&parameter.name).ok_or_else(|| {
                SpiceError::parse(
                    parameter.location.clone(),
                    format!("unknown {} parameter {}", card.base, parameter.name),
                )
            })?;
            if shape.vector() {
                for value in numeric_vector(parameter)? {
                    validate_value(parameter, shape, value)?;
                }
            } else {
                validate_value(parameter, shape, numeric_scalar(parameter)?)?;
            }
        }
        Ok(())
    }
}

pub(super) fn numeric_scalar(parameter: &ParameterAssignment) -> SpiceResult<Real> {
    parse_spice_number(&parameter.value)
        .filter(|value| value.is_finite() && parameter.kind == ParameterKind::Scalar)
        .ok_or_else(|| {
            SpiceError::parse(
                parameter.location.clone(),
                "expected finite resolved scalar",
            )
        })
}

pub(super) fn numeric_vector(parameter: &ParameterAssignment) -> SpiceResult<Vec<Real>> {
    let ParameterKind::NumericVector(entries) = &parameter.kind else {
        return Err(SpiceError::parse(
            parameter.location.clone(),
            "expected resolved numeric vector",
        ));
    };
    entries
        .iter()
        .map(|entry| {
            parse_spice_number(&entry.source.text)
                .filter(|value| value.is_finite() && entry.expression.is_none())
                .ok_or_else(|| {
                    SpiceError::parse(
                        entry.source.location.clone(),
                        "expected finite resolved vector entry",
                    )
                })
        })
        .collect()
}

fn validate_value(
    parameter: &ParameterAssignment,
    shape: ParameterShape,
    value: Real,
) -> SpiceResult<()> {
    let valid = match shape {
        ParameterShape::Integer | ParameterShape::IntegerVector => value.fract() == 0.,
        ParameterShape::Boolean | ParameterShape::BooleanVector => matches!(value, 0. | 1.),
        ParameterShape::Real | ParameterShape::RealVector => true,
    };
    if !valid {
        return Err(SpiceError::parse(
            parameter.location.clone(),
            "invalid integer or boolean code-model parameter",
        ));
    }
    Ok(())
}

use super::pulse::PulseFilter;
use super::sequential_timing::TimingSpec;
use super::source_logic::{AsyncPolicy, CausalPaths, DriveStrengths, LogicKernel, Operation};
use super::{
    Adc, AnalogOutputConfig, AnalogOutputDevice, ConductanceConfig, DacConfig, DelayMode, Delays,
    Drive, GapPolicy, Logic, OutputPort, RangeReceiver, Strength, VoltageRange,
};
use crate::analysis::digital::{
    AdcBinding, AnalogOutputBinding, NativeTopology, NetId, Network, RangeBinding, ReceiverBinding,
};
use crate::devices::circuit::ElaboratedDevice;
use crate::devices::models::{PreparationKey, ResolvedModel};
use crate::devices::{Device, ElaborationLimits};
use crate::netlist::ast::CodeTerminal;
use crate::primitives::{NodeAppend, NodeId};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct LogicMaterial {
    operation: Real,
    delays: Delays,
    windows: Option<super::WindowDelays>,
    causal: CausalPaths,
    inertial: bool,
    pulse: Option<PulseFilter>,
    initial: Logic,
    strength: DriveStrengths,
    margin: Option<std::num::NonZeroU8>,
    input_capacitance: Option<Vec<Real>>,
    output_capacitance: Real,
    load_resistance: Option<[Real; 3]>,
    asynchronous: Option<AsyncPolicy>,
    timing: TimingSpec,
    low: Real,
    high: Real,
    inverted: Option<Vec<bool>>,
    receiver_initial: Option<Vec<Logic>>,
}

impl Default for LogicMaterial {
    fn default() -> Self {
        Self {
            operation: 0.,
            delays: Delays::ZERO,
            windows: None,
            causal: CausalPaths::default(),
            inertial: true,
            pulse: None,
            initial: Logic::Zero,
            strength: DriveStrengths::uniform(Strength::STRONG),
            margin: None,
            input_capacitance: None,
            output_capacitance: 0.,
            load_resistance: None,
            asynchronous: None,
            timing: TimingSpec::default(),
            low: 2.,
            high: 3.,
            inverted: None,
            receiver_initial: None,
        }
    }
}

fn binary(value: Real) -> SpiceResult<bool> {
    match value {
        0. => Ok(false),
        1. => Ok(true),
        _ => Err(SpiceError::circuit("expected binary code-model parameter")),
    }
}

fn logic(value: Real) -> SpiceResult<Logic> {
    Ok(if binary(value)? {
        Logic::One
    } else {
        Logic::Zero
    })
}

fn three_logic(value: Real) -> SpiceResult<Logic> {
    if matches!(value, 2.) {
        Ok(Logic::Unknown)
    } else {
        logic(value)
    }
}

fn rank(value: Real) -> SpiceResult<Strength> {
    Strength::from_rank(
        crate::primitives::exact_nonnegative_integer(value)
            .ok_or_else(|| SpiceError::circuit("invalid digital strength rank"))?,
    )
}

fn nonnegative(value: Real) -> SpiceResult<Real> {
    if value.is_finite() && value >= 0. {
        Ok(value)
    } else {
        Err(SpiceError::circuit(
            "invalid digital electrical loading parameter",
        ))
    }
}

impl LogicMaterial {
    fn compile(card: &ModelCard) -> SpiceResult<Self> {
        SourceModel::LogicEvent.validate_parameters(card)?;
        let mut material = Self::default();
        let mut timing_given = false;
        let mut strength_given = false;
        let mut mode_given = false;
        for parameter in &card.parameters {
            match parameter.name.as_str() {
                "operation" => material.operation = numeric_scalar(parameter)?,
                "delay" | "delay_paths" => {
                    if timing_given {
                        return Err(SpiceError::circuit(
                            "specify exactly one logic timing declaration",
                        ));
                    }
                    timing_given = true;
                    material.delays = if parameter.name == "delay" {
                        let delay = numeric_scalar(parameter)?;
                        Delays::new(delay, delay)?
                    } else {
                        match numeric_vector(parameter)?.as_slice() {
                            [rise, fall] => Delays::new(*rise, *fall)?,
                            [rise, fall, lz, hz, zl, zh] => {
                                Delays::paths([*rise, *fall, *lz, *hz, *zl, *zh])?
                            }
                            _ => {
                                return Err(SpiceError::circuit(
                                    "delay_paths requires two or six directional values",
                                ));
                            }
                        }
                    };
                }
                "delay_windows" => {
                    if timing_given {
                        return Err(SpiceError::circuit(
                            "specify exactly one logic timing declaration",
                        ));
                    }
                    timing_given = true;
                    use crate::analysis::digital::timing::DelayWindow;
                    let values = numeric_vector(parameter)?;
                    material.windows = Some(match values.as_slice() {
                        [rmin, rmax, fmin, fmax] => super::WindowDelays::binary(
                            DelayWindow::new(*rmin, *rmax)?,
                            DelayWindow::new(*fmin, *fmax)?,
                        ),
                        [a, b, c, d, e, f, g, h, i, j, k, l] => super::WindowDelays::paths([
                            DelayWindow::new(*a, *b)?,
                            DelayWindow::new(*c, *d)?,
                            DelayWindow::new(*e, *f)?,
                            DelayWindow::new(*g, *h)?,
                            DelayWindow::new(*i, *j)?,
                            DelayWindow::new(*k, *l)?,
                        ]),
                        _ => {
                            return Err(SpiceError::circuit(
                                "delay_windows requires four or twelve directional bounds",
                            ));
                        }
                    });
                }
                "delay_gate" | "delay_preset" | "delay_clear" => {
                    let [rise, fall]: [Real; 2] =
                        numeric_vector(parameter)?.try_into().map_err(|_| {
                            SpiceError::circuit(
                                "causal propagation path requires two directional values",
                            )
                        })?;
                    let target = match parameter.name.as_str() {
                        "delay_gate" => &mut material.causal.gate,
                        "delay_preset" => &mut material.causal.preset,
                        _ => &mut material.causal.clear,
                    };
                    *target = Some(Delays::new(rise, fall)?);
                }
                "inertial" | "pulse_reject" => {
                    if mode_given {
                        return Err(SpiceError::circuit(
                            "specify exactly one logic pulse policy",
                        ));
                    }
                    mode_given = true;
                    if parameter.name == "inertial" {
                        material.inertial = binary(numeric_scalar(parameter)?)?;
                    } else {
                        material.pulse = Some(PulseFilter::new(numeric_scalar(parameter)?)?);
                    }
                }
                "ic" => material.initial = three_logic(numeric_scalar(parameter)?)?,
                "strength" | "strength_paths" => {
                    if strength_given {
                        return Err(SpiceError::circuit(
                            "specify exactly one logic strength declaration",
                        ));
                    }
                    strength_given = true;
                    material.strength = if parameter.name == "strength" {
                        DriveStrengths::uniform(match numeric_scalar(parameter)? {
                            1. => Strength::RESISTIVE,
                            2. => Strength::STRONG,
                            _ => {
                                return Err(SpiceError::circuit(
                                    "logic strength must be resistive (1) or strong (2)",
                                ));
                            }
                        })
                    } else {
                        let [low, high, unknown]: [Real; 3] =
                            numeric_vector(parameter)?.try_into().map_err(|_| {
                                SpiceError::circuit("strength_paths requires L/H/X ranks")
                            })?;
                        DriveStrengths([rank(low)?, rank(high)?, rank(unknown)?])
                    };
                }
                "override_margin" => {
                    material.margin = Some(
                        std::num::NonZeroU8::new(rank(numeric_scalar(parameter)?)?.rank())
                            .ok_or_else(|| {
                                SpiceError::circuit("digital net strength margin must be nonzero")
                            })?,
                    );
                }
                "input_capacitance" => {
                    material.input_capacitance = Some(
                        numeric_vector(parameter)?
                            .into_iter()
                            .map(nonnegative)
                            .collect::<SpiceResult<_>>()?,
                    );
                }
                "output_capacitance" => {
                    material.output_capacitance = nonnegative(numeric_scalar(parameter)?)?
                }
                "load_resistance" => {
                    let values: [Real; 3] = numeric_vector(parameter)?
                        .into_iter()
                        .map(nonnegative)
                        .collect::<SpiceResult<Vec<_>>>()?
                        .try_into()
                        .map_err(|_| {
                            SpiceError::circuit("load_resistance requires L/H/Z resistances")
                        })?;
                    material.load_resistance = Some(values);
                }
                "async_policy" => {
                    let [preset, clear, both]: [Real; 3] = numeric_vector(parameter)?.try_into()
                        .map_err(|_| SpiceError::circuit("async_policy requires preset level, clear level and simultaneous result"))?;
                    material.asynchronous = Some(AsyncPolicy {
                        preset: logic(preset)?,
                        clear: logic(clear)?,
                        both: three_logic(both)?,
                    });
                }
                "timing_setup" | "timing_hold" | "timing_clock_min" | "timing_async_min"
                | "timing_recovery" => {
                    material
                        .timing
                        .set_vector(&parameter.name, numeric_vector(parameter)?)?;
                }
                "timing_violation" => material.timing.set_policy(numeric_scalar(parameter)?)?,
                "low" => material.low = numeric_scalar(parameter)?,
                "high" => material.high = numeric_scalar(parameter)?,
                "inverted" => {
                    material.inverted = Some(
                        numeric_vector(parameter)?
                            .into_iter()
                            .map(binary)
                            .collect::<SpiceResult<_>>()?,
                    );
                }
                "rx_ic" => {
                    material.receiver_initial = Some(
                        numeric_vector(parameter)?
                            .into_iter()
                            .map(logic)
                            .collect::<SpiceResult<_>>()?,
                    );
                }
                _ => return Err(SpiceError::circuit("unregistered logic parameter")),
            }
        }
        if material.windows.is_some() {
            if material.pulse.is_some() || (mode_given && material.inertial) {
                return Err(SpiceError::circuit(
                    "causal arrival windows require transport mode",
                ));
            }
            material.inertial = false;
        }
        Adc::qualified(material.low, material.high, Logic::Zero, false)?;
        Ok(material)
    }

    fn bind(
        &self,
        terminals: &[CodeTerminal],
        nodes: &mut NodeAppend<'_>,
    ) -> SpiceResult<(LogicKernel, Vec<LogicInput>, Vec<NodeId>)> {
        let inputs = terminals.len();
        if self
            .inverted
            .as_ref()
            .is_some_and(|values| values.len() != inputs)
            || self
                .receiver_initial
                .as_ref()
                .is_some_and(|values| values.len() != inputs)
            || self
                .input_capacitance
                .as_ref()
                .is_some_and(|values| values.len() != inputs)
        {
            return Err(SpiceError::circuit(
                "logic parameter vector length must match input port",
            ));
        }
        let mut invert = Vec::with_capacity(inputs);
        let mut bindings = Vec::with_capacity(inputs);
        let mut analog = Vec::with_capacity(inputs.saturating_mul(2));
        for (index, terminal) in terminals.iter().enumerate() {
            let inverted = self.inverted.as_ref().is_some_and(|values| values[index]);
            let capacitance = self
                .input_capacitance
                .as_ref()
                .map_or(0., |values| values[index]);
            if terminal.kind == Digital {
                invert.push(terminal.inverted ^ inverted);
                bindings.push(LogicInput::Digital {
                    name: terminal.positive.clone(),
                    capacitance,
                });
            } else {
                if capacitance != 0. {
                    return Err(SpiceError::circuit(
                        "analog receiver capacitance must be a physical circuit element",
                    ));
                }
                invert.push(false);
                let terminals = analog_pair(terminal, nodes)?;
                analog.extend(terminals);
                bindings.push(LogicInput::Analog {
                    terminals,
                    thresholds: Adc::qualified(
                        self.low,
                        self.high,
                        self.receiver_initial
                            .as_ref()
                            .map_or(Logic::Zero, |values| values[index]),
                        inverted,
                    )?,
                });
            }
        }
        let operation = Operation::from_value(self.operation, inputs, self.asynchronous)?;
        self.causal.validate(operation)?;
        let timing = self.timing.bind(operation, inputs)?;
        if self.windows.is_some()
            && !matches!(operation, Operation::Gate(_) | Operation::Clocked { .. })
        {
            return Err(SpiceError::circuit(
                "causal arrival windows require a combinational gate or clocked storage",
            ));
        }
        let kernel = LogicKernel {
            operation,
            timing,
            inputs,
            initial: self.initial,
            delays: self.delays,
            causal: self.causal,
            pulse: self.pulse,
            mode: if self.inertial {
                DelayMode::Inertial
            } else {
                DelayMode::Transport
            },
            strength: self.strength,
            invert,
        };
        Ok((kernel, bindings, analog))
    }
}

#[derive(Debug)]
enum LogicInput {
    Analog {
        terminals: [NodeId; 2],
        thresholds: Adc,
    },
    Digital {
        name: String,
        capacitance: Real,
    },
}

#[derive(Debug)]
struct LogicReceiverDevice {
    name: String,
    model: String,
    terminals: Vec<NodeId>,
    output: String,
    kernel: LogicKernel,
    windows: Option<super::WindowDelays>,
    margin: Option<std::num::NonZeroU8>,
    output_capacitance: Real,
    load_resistance: Option<[Real; 3]>,
    inputs: Vec<LogicInput>,
}

impl Device for LogicReceiverDevice {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'a'
    }
    fn model_name(&self) -> Option<&str> {
        Some(&self.model)
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn setup_matrix(
        &self,
        _setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        Ok(true)
    }
    fn stamp(&self, _context: &mut crate::devices::StampContext<'_>) -> SpiceResult<()> {
        Ok(())
    }
    fn assemble_linear(
        &self,
        _context: &mut crate::devices::linear::LinearContext<'_>,
    ) -> SpiceResult<()> {
        Ok(())
    }
}

#[derive(Debug)]
struct RangeReceiverDevice {
    name: String,
    model: String,
    terminals: [NodeId; 2],
    output: String,
    receiver: RangeReceiver,
}

impl Device for RangeReceiverDevice {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'a'
    }
    fn model_name(&self) -> Option<&str> {
        Some(&self.model)
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn setup_matrix(&self, _: &mut crate::devices::setup::MatrixSetup<'_>) -> SpiceResult<bool> {
        Ok(true)
    }
    fn stamp(&self, _: &mut crate::devices::StampContext<'_>) -> SpiceResult<()> {
        Ok(())
    }
    fn assemble_linear(
        &self,
        _: &mut crate::devices::linear::LinearContext<'_>,
    ) -> SpiceResult<()> {
        Ok(())
    }
}

fn range_drive(value: Real, strength: Real) -> SpiceResult<Drive> {
    let logic = match value {
        0. => Logic::Zero,
        1. => Logic::One,
        2. => Logic::Unknown,
        3. => Logic::Rising,
        4. => Logic::Falling,
        _ => return Err(SpiceError::circuit("invalid receiver range logic code")),
    };
    Ok(Drive {
        logic,
        strength: rank(strength)?,
    })
}

fn range_material(card: &ModelCard) -> SpiceResult<RangeReceiver> {
    SourceModel::LogicRange.validate_parameters(card)?;
    let mut ranges = Vec::new();
    let mut initial = 0;
    let mut gap = None;
    for parameter in &card.parameters {
        match parameter.name.as_str() {
            "ranges" => {
                let values = numeric_vector(parameter)?;
                let (entries, remainder) = values.as_chunks::<4>();
                if entries.is_empty() || !remainder.is_empty() {
                    return Err(SpiceError::circuit(
                        "ranges requires voltage-low, voltage-high, logic, strength tuples",
                    ));
                }
                ranges = entries
                    .iter()
                    .map(|&[lo, hi, logic, strength]| {
                        VoltageRange::new(lo, hi, range_drive(logic, strength)?)
                    })
                    .collect::<SpiceResult<_>>()?;
            }
            "initial_range" => {
                initial = crate::primitives::exact_nonnegative_integer(numeric_scalar(parameter)?)
                    .ok_or_else(|| {
                    SpiceError::circuit("invalid receiver initial range index")
                })?
            }
            "gap_policy" | "gap_drive" => {
                if gap.is_some() {
                    return Err(SpiceError::circuit("specify one receiver gap policy"));
                }
                gap = Some(if parameter.name == "gap_policy" {
                    match numeric_scalar(parameter)? {
                        0. => GapPolicy::Error,
                        1. => GapPolicy::Nearest,
                        _ => return Err(SpiceError::circuit("invalid receiver gap policy")),
                    }
                } else {
                    let [logic, strength]: [Real; 2] =
                        numeric_vector(parameter)?.try_into().map_err(|_| {
                            SpiceError::circuit("gap_drive requires logic and strength")
                        })?;
                    GapPolicy::Drive(range_drive(logic, strength)?)
                });
            }
            _ => {
                return Err(SpiceError::circuit(
                    "unregistered ordered receiver parameter",
                ));
            }
        }
    }
    RangeReceiver::new(ranges, initial, gap.unwrap_or(GapPolicy::Error))
}

#[derive(Clone, Debug)]
pub(crate) struct NativeOutputBinding {
    pub(crate) model: String,
    net: String,
}

fn analog_pair(terminal: &CodeTerminal, nodes: &mut NodeAppend<'_>) -> SpiceResult<[NodeId; 2]> {
    Ok([
        nodes.intern(&terminal.positive)?,
        nodes.intern(terminal.negative.as_deref().unwrap_or("0"))?,
    ])
}

fn voltage_material(card: &ModelCard) -> SpiceResult<[Real; 2]> {
    SourceModel::LogicVoltage.validate_parameters(card)?;
    let mut rails = [0., 1.];
    for parameter in &card.parameters {
        match parameter.name.as_str() {
            "out_low" => rails[0] = numeric_scalar(parameter)?,
            "out_high" => rails[1] = numeric_scalar(parameter)?,
            _ => return Err(SpiceError::circuit("unregistered voltage parameter")),
        }
    }
    Ok(rails)
}

fn conductance_material(card: &ModelCard) -> SpiceResult<ConductanceConfig> {
    use super::conductance::ResistanceSlew;
    SourceModel::LogicConductance.validate_parameters(card)?;
    let mut table = None;
    let mut duration = None;
    let mut shape = None;
    for parameter in &card.parameters {
        match parameter.name.as_str() {
            "conductance" | "conductance_by_rank" => {
                if table.is_some() {
                    return Err(SpiceError::circuit("specify one electrical drive table"));
                }
                table = Some((
                    numeric_vector(parameter)?,
                    parameter.name == "conductance_by_rank",
                ));
            }
            "resistance_duration" => {
                if duration.is_some() {
                    return Err(SpiceError::circuit("duplicate resistance duration"));
                }
                let values: [Real; 6] = numeric_vector(parameter)?.try_into().map_err(|_| {
                    SpiceError::circuit("resistance_duration requires L H X R F Z durations")
                })?;
                duration = Some(values);
            }
            "resistance_shape" => {
                if shape.is_some() {
                    return Err(SpiceError::circuit("duplicate resistance shape"));
                }
                shape = Some(numeric_scalar(parameter)?);
            }
            _ => return Err(SpiceError::circuit("unregistered conductance parameter")),
        }
    }
    let (values, ranked) = table.ok_or_else(|| {
        SpiceError::circuit("logic conductance requires an electrical drive table")
    })?;
    let slew = match (duration, shape) {
        (None, None) => None,
        (Some(duration), Some(shape)) => Some(ResistanceSlew::new(duration, shape)?),
        _ => {
            return Err(SpiceError::circuit(
                "resistance slew requires both duration and shape",
            ));
        }
    };
    ConductanceConfig::new(&values, ranked, slew)
}

pub(in crate::devices) fn instantiate(
    schema: SourceModel,
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
) -> SpiceResult<Box<dyn Device>> {
    schema.validate_ports(instance)?;
    match schema {
        SourceModel::LogicEvent => {
            let material = model.prepare(PreparationKey::Inputs, || {
                LogicMaterial::compile(model.card())
            })?;
            let [input, output] = instance.code_ports.as_slice() else {
                return Err(SpiceError::circuit("logic port count mismatch"));
            };
            let (kernel, inputs, terminals) = material.bind(&input.terminals, nodes)?;
            Ok(Box::new(LogicReceiverDevice {
                name: instance.name.clone(),
                model: model.card().name.clone(),
                terminals,
                output: output.terminals[0].positive.clone(),
                kernel,
                windows: material.windows,
                margin: material.margin,
                output_capacitance: material.output_capacitance,
                load_resistance: material.load_resistance,
                inputs,
            }))
        }
        SourceModel::LogicRange => {
            let receiver =
                model.prepare(PreparationKey::Inputs, || range_material(model.card()))?;
            let [input, output] = instance.code_ports.as_slice() else {
                return Err(SpiceError::circuit("ordered receiver port count mismatch"));
            };
            let terminals = analog_pair(&input.terminals[0], nodes)?;
            Ok(Box::new(RangeReceiverDevice {
                name: instance.name.clone(),
                model: model.card().name.clone(),
                terminals,
                output: output.terminals[0].positive.clone(),
                receiver: (*receiver).clone(),
            }))
        }
        SourceModel::LogicVoltage | SourceModel::LogicConductance => {
            let [input, output] = instance.code_ports.as_slice() else {
                return Err(SpiceError::circuit(
                    "logic analog-output port count mismatch",
                ));
            };
            let input = &input.terminals[0];
            let terminals = analog_pair(&output.terminals[0], nodes)?;
            if terminals[0] == terminals[1] {
                return Err(SpiceError::circuit("shorted logic analog output"));
            }
            let config = if schema == SourceModel::LogicVoltage {
                let rails =
                    model.prepare(PreparationKey::Inputs, || voltage_material(model.card()))?;
                let [low, high] = *rails;
                AnalogOutputConfig::Voltage(if input.inverted {
                    DacConfig::ideal(high, low)?
                } else {
                    DacConfig::ideal(low, high)?
                })
            } else {
                let values = model.prepare(PreparationKey::Inputs, || {
                    conductance_material(model.card())
                })?;
                AnalogOutputConfig::Conductance((*values).clone().with_inversion(input.inverted))
            };
            Ok(Box::new(AnalogOutputDevice {
                name: instance.name.clone(),
                terminals,
                port: OutputPort::default(),
                config,
                native: Some(NativeOutputBinding {
                    model: model.card().name.clone(),
                    net: input.positive.clone(),
                }),
            }))
        }
        SourceModel::RationalTransfer => super::rational::instantiate(instance, nodes, model),
        SourceModel::FrequencyTransfer => super::frequency::instantiate(instance, nodes, model),
        SourceModel::PSwitch | SourceModel::ASwitch => {
            super::continuous_switch_device::instantiate(schema, instance, nodes, model)
        }
        SourceModel::TimedSwitch => {
            let material = model.prepare(PreparationKey::Inputs, || {
                super::timed_switch_device::compile(model.card())
            })?;
            let [control, branch] = instance.code_ports.as_slice() else {
                return Err(SpiceError::circuit("timed switch port count mismatch"));
            };
            let [cp, cn] = analog_pair(&control.terminals[0], nodes)?;
            let [p, n] = analog_pair(&branch.terminals[0], nodes)?;
            Ok(Box::new(super::timed_switch_device::TimedSwitch {
                name: instance.name.clone(),
                model_name: model.card().name.clone(),
                terminals: [cp, cn, p, n],
                material: *material,
            }))
        }
    }
}

fn digital_net(
    network: &mut Network,
    names: &mut BTreeMap<String, NetId>,
    name: &str,
    limit: usize,
) -> SpiceResult<NetId> {
    if let Some(&id) = names.get(name) {
        return Ok(id);
    }
    if network.net_count() >= limit {
        return Err(SpiceError::circuit("digital net resource limit exceeded"));
    }
    let id = network.add_net();
    names.insert(name.to_owned(), id);
    Ok(id)
}

fn bind_receiver(
    device: &LogicReceiverDevice,
    network: &mut Network,
    names: &mut BTreeMap<String, NetId>,
    adcs: &mut Vec<ReceiverBinding>,
    capacitances: &BTreeMap<&str, Real>,
    limits: ElaborationLimits,
) -> SpiceResult<()> {
    let output = digital_net(network, names, &device.output, limits.nodes)?;
    if let Some(margin) = device.margin {
        network.set_override_margin(output, margin)?;
    }
    let mut inputs = Vec::with_capacity(device.inputs.len());
    let zero = Delays::ZERO;
    for input in &device.inputs {
        let net = match input {
            LogicInput::Digital { name, .. } => digital_net(network, names, name, limits.nodes)?,
            LogicInput::Analog {
                terminals,
                thresholds,
            } => {
                if network.net_count() >= limits.nodes || adcs.len() >= limits.state_slots {
                    return Err(SpiceError::circuit(
                        "qualified receiver resource limit exceeded",
                    ));
                }
                // Independent receiver history even when two terminals
                // have identical analog node spelling.
                let net = network.add_net();
                let source = network.add_source(net)?;
                adcs.push(
                    AdcBinding {
                        terminals: *terminals,
                        source,
                        thresholds: *thresholds,
                        delays: zero,
                        mode: DelayMode::Transport,
                    }
                    .into(),
                );
                net
            }
        };
        inputs.push(net);
    }
    let mut kernel = device.kernel.clone();
    let mut windows = device.windows;
    if let Some(resistances) = device.load_resistance {
        let total = capacitances
            .get(device.output.as_str())
            .copied()
            .unwrap_or(0.);
        let load = resistances.map(|resistance| resistance * (total * std::f64::consts::LN_2));
        if load.iter().any(|value| !value.is_finite()) {
            return Err(SpiceError::circuit("digital loading delay overflow"));
        }
        kernel.delays = kernel.delays.with_load(load)?;
        kernel.causal = kernel.causal.with_load(load)?;
        windows = windows.map(|w| w.with_load(load)).transpose()?;
    }
    if let Some(windows) = windows {
        match kernel.operation {
            Operation::Gate(kind) => {
                let model = super::WindowGate::new(kind, inputs.len(), windows)?
                    .with_binding(kernel.strength, kernel.invert);
                network.add_named_model(&device.name, model, &inputs, &[output])
            }
            Operation::Clocked { .. } => network.add_named_model(
                &device.name,
                super::window_clocked::WindowClocked::new(kernel, windows)?,
                &inputs,
                &[output],
            ),
            _ => Err(SpiceError::circuit("causal window logic binding mismatch")),
        }
    } else {
        network.add_named_model(&device.name, kernel, &inputs, &[output])
    }
}

/// Sum terminal capacitance once after flattening, including every
/// occurrence even when the same net is attached to multiple input ports.
fn digital_capacitances(devices: &[ElaboratedDevice]) -> SpiceResult<BTreeMap<&str, Real>> {
    let mut totals = BTreeMap::<&str, Real>::new();
    for device in devices {
        if device.designator() != 'a' {
            continue;
        }
        let Some(receiver) =
            (device.as_ref() as &dyn std::any::Any).downcast_ref::<LogicReceiverDevice>()
        else {
            continue;
        };
        let mut add = |name, value| -> SpiceResult<()> {
            let total = totals.entry(name).or_default();
            *total += value;
            if !total.is_finite() {
                return Err(SpiceError::circuit("digital net capacitance overflow"));
            }
            Ok(())
        };
        add(receiver.output.as_str(), receiver.output_capacitance)?;
        for input in &receiver.inputs {
            if let LogicInput::Digital { name, capacitance } = input {
                add(name.as_str(), *capacitance)?;
            }
        }
    }
    Ok(totals)
}

/// Build the fixed mixed-domain topology once after native device binding.
/// The analog-only path never constructs a network or event state.
pub(in crate::devices) fn mixed_topology(
    devices: &[ElaboratedDevice],
    limits: ElaborationLimits,
) -> SpiceResult<Option<NativeTopology>> {
    let capacitances = digital_capacitances(devices)?;
    let mut topology = None;
    for device in devices {
        if device.designator() != 'a' {
            continue;
        }
        let any = device.as_ref() as &dyn std::any::Any;
        let receiver = any.downcast_ref::<LogicReceiverDevice>();
        let range = any.downcast_ref::<RangeReceiverDevice>();
        let voltage = any
            .downcast_ref::<AnalogOutputDevice>()
            .and_then(|device| device.native.as_ref().map(|binding| (device, binding)));
        if receiver.is_none() && range.is_none() && voltage.is_none() {
            continue;
        }
        let (network, names, adcs, analog_outputs): &mut (
            Network,
            BTreeMap<String, NetId>,
            Vec<ReceiverBinding>,
            Vec<AnalogOutputBinding>,
        ) = topology.get_or_insert_with(Default::default);
        if let Some(receiver) = receiver {
            bind_receiver(receiver, network, names, adcs, &capacitances, limits)?;
        }
        if let Some(device) = range {
            if adcs.len() >= limits.state_slots {
                return Err(SpiceError::circuit(
                    "ordered receiver resource limit exceeded",
                ));
            }
            let net = digital_net(network, names, &device.output, limits.nodes)?;
            let source = network.add_source(net)?;
            adcs.push(
                RangeBinding {
                    name: device.name.as_str().into(),
                    terminals: device.terminals,
                    source,
                    receiver: device.receiver.clone(),
                    delays: Delays::ZERO,
                    mode: DelayMode::Transport,
                }
                .into(),
            );
        }
        if let Some((device, binding)) = voltage {
            let net = digital_net(network, names, &binding.net, limits.nodes)?;
            analog_outputs.push(AnalogOutputBinding {
                name: device.name.clone(),
                net,
                config: device.config.clone(),
                port: device.port.clone(),
            });
        }
    }
    Ok(topology.map(|(network, _, adcs, analog_outputs)| {
        NativeTopology::new(network, adcs, analog_outputs)
    }))
}
