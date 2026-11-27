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
    /// Ideal digital-to-voltage bridge.
    LogicVoltage,
    /// Timed conductance switch.
    TimedSwitch,
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
    #[must_use]
    pub const fn vector(self) -> bool {
        matches!(
            self,
            Self::RealVector | Self::IntegerVector | Self::BooleanVector
        )
    }

    /// Whether literal boolean spellings are accepted.
    #[must_use]
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
    ConductanceDifferential, Current, CurrentDifferential, Digital, Voltage, VoltageDifferential,
};
const ANALOG: &[CodePortKind] = &[Voltage, VoltageDifferential];
const TRANSFER: &[CodePortKind] = &[Voltage, VoltageDifferential, Current, CurrentDifferential];
const EVENT: &[CodePortKind] = &[Digital];
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
    #[must_use]
    pub const fn from_base(base: &str) -> Option<Self> {
        if base.eq_ignore_ascii_case("s_xfer") {
            Some(Self::RationalTransfer)
        } else if base.eq_ignore_ascii_case("xfer") {
            Some(Self::FrequencyTransfer)
        } else if base.eq_ignore_ascii_case("preprocess_logic_event") {
            Some(Self::LogicEvent)
        } else if base.eq_ignore_ascii_case("preprocess_logic_voltage") {
            Some(Self::LogicVoltage)
        } else if base.eq_ignore_ascii_case("preprocess_timed_switch") {
            Some(Self::TimedSwitch)
        } else {
            None
        }
    }

    /// Ordered terminal groups and their default domains.
    #[must_use]
    pub const fn ports(self) -> &'static [PortSpec] {
        const TRANSFER_PORTS: &[PortSpec] = &[
            PortSpec::new(Voltage, TRANSFER, false, false),
            PortSpec::new(Voltage, TRANSFER, false, false),
        ];
        const LOGIC_PORTS: &[PortSpec] = &[
            PortSpec::new(Voltage, ANALOG, true, false),
            PortSpec::new(Digital, EVENT, false, false),
        ];
        const VOLTAGE_PORTS: &[PortSpec] = &[
            PortSpec::new(Digital, EVENT, false, true),
            PortSpec::new(Voltage, ANALOG, false, false),
        ];
        const SWITCH_PORTS: &[PortSpec] = &[
            PortSpec::new(VoltageDifferential, ANALOG, false, false),
            PortSpec::new(ConductanceDifferential, BRANCH, false, false),
        ];
        match self {
            Self::RationalTransfer | Self::FrequencyTransfer => TRANSFER_PORTS,
            Self::LogicEvent => LOGIC_PORTS,
            Self::LogicVoltage => VOLTAGE_PORTS,
            Self::TimedSwitch => SWITCH_PORTS,
        }
    }

    /// Shape of a declared static setter. Unknown setters are rejected during
    /// model binding, after source expressions have been evaluated.
    #[must_use]
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
                "operation" | "ic" => Some(Integer),
                "delay" | "low" | "high" => Some(Real),
                "inertial" => Some(Boolean),
                "inverted" => Some(BooleanVector),
                "rx_ic" => Some(IntegerVector),
                _ => None,
            },
            Self::LogicVoltage => match name {
                "out_low" | "out_high" => Some(Real),
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
                    || (terminal.inverted && !spec.inversion)
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

use super::source_logic::{LogicKernel, Operation};
use super::{Adc, DacConfig, DacPort, DacSource, DelayMode, Delays, Logic};
use crate::analysis::digital::{AdcBinding, DacBinding, NativeTopology, NetId, Network};
use crate::devices::circuit::ElaboratedDevice;
use crate::devices::models::{PreparationKey, ResolvedModel};
use crate::devices::{Device, ElaborationLimits};
use crate::netlist::ast::CodeTerminal;
use crate::primitives::{NodeAppend, NodeId};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
struct LogicMaterial {
    operation: Real,
    delay: Real,
    inertial: bool,
    initial: Logic,
    low: Real,
    high: Real,
    inverted: Option<Vec<bool>>,
    receiver_initial: Option<Vec<Logic>>,
}

impl Default for LogicMaterial {
    fn default() -> Self {
        Self {
            operation: 0.,
            delay: 0.,
            inertial: true,
            initial: Logic::Zero,
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

impl LogicMaterial {
    fn compile(card: &ModelCard) -> SpiceResult<Self> {
        SourceModel::LogicEvent.validate_parameters(card)?;
        let mut material = Self::default();
        for parameter in &card.parameters {
            match parameter.name.as_str() {
                "operation" => material.operation = numeric_scalar(parameter)?,
                "delay" => material.delay = numeric_scalar(parameter)?,
                "inertial" => material.inertial = binary(numeric_scalar(parameter)?)?,
                "ic" => material.initial = logic(numeric_scalar(parameter)?)?,
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
        Delays::new(material.delay, material.delay)?;
        Adc::qualified(material.low, material.high, material.initial, false)?;
        Ok(material)
    }

    fn bind(&self, inputs: usize) -> SpiceResult<(LogicKernel, Vec<Adc>)> {
        if self
            .inverted
            .as_ref()
            .is_some_and(|values| values.len() != inputs)
            || self
                .receiver_initial
                .as_ref()
                .is_some_and(|values| values.len() != inputs)
        {
            return Err(SpiceError::circuit(
                "logic parameter vector length must match input port",
            ));
        }
        let kernel = LogicKernel {
            operation: Operation::from_value(self.operation, inputs)?,
            inputs,
            initial: self.initial,
            delay: self.delay,
            mode: if self.inertial {
                DelayMode::Inertial
            } else {
                DelayMode::Transport
            },
        };
        let receivers = (0..inputs)
            .map(|index| {
                Adc::qualified(
                    self.low,
                    self.high,
                    self.receiver_initial
                        .as_ref()
                        .map_or(Logic::Zero, |values| values[index]),
                    self.inverted.as_ref().is_some_and(|values| values[index]),
                )
            })
            .collect::<SpiceResult<_>>()?;
        Ok((kernel, receivers))
    }
}

#[derive(Debug)]
struct LogicReceiverDevice {
    name: String,
    model: String,
    terminals: Vec<NodeId>,
    output: String,
    kernel: LogicKernel,
    receivers: Vec<Adc>,
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

#[derive(Clone, Debug)]
pub(crate) struct NativeVoltageBinding {
    pub(crate) model: String,
    net: String,
    config: DacConfig,
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
            let (kernel, receivers) = material.bind(input.terminals.len())?;
            let mut terminals = Vec::with_capacity(input.terminals.len().saturating_mul(2));
            for terminal in &input.terminals {
                terminals.extend(analog_pair(terminal, nodes)?);
            }
            Ok(Box::new(LogicReceiverDevice {
                name: instance.name.clone(),
                model: model.card().name.clone(),
                terminals,
                output: output.terminals[0].positive.clone(),
                kernel,
                receivers,
            }))
        }
        SourceModel::LogicVoltage => {
            let rails = model.prepare(PreparationKey::Inputs, || voltage_material(model.card()))?;
            let [input, output] = instance.code_ports.as_slice() else {
                return Err(SpiceError::circuit("logic voltage port count mismatch"));
            };
            let input = &input.terminals[0];
            let terminals = analog_pair(&output.terminals[0], nodes)?;
            if terminals[0] == terminals[1] {
                return Err(SpiceError::circuit("shorted logic voltage output"));
            }
            let [low, high] = *rails;
            let config = if input.inverted {
                DacConfig::ideal(high, low)?
            } else {
                DacConfig::ideal(low, high)?
            };
            Ok(Box::new(DacSource {
                name: instance.name.clone(),
                terminals,
                port: DacPort::default(),
                native: Some(NativeVoltageBinding {
                    model: model.card().name.clone(),
                    net: input.positive.clone(),
                    config,
                }),
            }))
        }
        SourceModel::RationalTransfer => super::rational::instantiate(instance, nodes, model),
        SourceModel::FrequencyTransfer => super::frequency::instantiate(instance, nodes, model),
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
    adcs: &mut Vec<AdcBinding>,
    limits: ElaborationLimits,
) -> SpiceResult<()> {
    let output = digital_net(network, names, &device.output, limits.nodes)?;
    let mut inputs = Vec::with_capacity(device.receivers.len());
    let zero = Delays::new(0., 0.)?;
    for (terminals, &thresholds) in device
        .terminals
        .as_chunks::<2>()
        .0
        .iter()
        .zip(&device.receivers)
    {
        if network.net_count() >= limits.nodes || adcs.len() >= limits.state_slots {
            return Err(SpiceError::circuit(
                "qualified receiver resource limit exceeded",
            ));
        }
        // Every receiver has its own private event net. This is intentionally
        // not interned by its analog terminal spelling or threshold value.
        let net = network.add_net();
        let source = network.add_source(net)?;
        adcs.push(AdcBinding {
            terminals: *terminals,
            source,
            thresholds,
            delays: zero,
            mode: DelayMode::Transport,
        });
        inputs.push(net);
    }
    network.add_model(device.kernel.clone(), &inputs, &[output])
}

/// Build the fixed mixed-domain topology once after native device binding.
/// The analog-only path never constructs a network or event state.
pub(in crate::devices) fn mixed_topology(
    devices: &[ElaboratedDevice],
    limits: ElaborationLimits,
) -> SpiceResult<Option<NativeTopology>> {
    let mut topology = None;
    for device in devices {
        if device.designator() != 'a' {
            continue;
        }
        let any = device.as_ref() as &dyn std::any::Any;
        let receiver = any.downcast_ref::<LogicReceiverDevice>();
        let voltage = any
            .downcast_ref::<DacSource>()
            .and_then(|device| device.native.as_ref().map(|binding| (device, binding)));
        if receiver.is_none() && voltage.is_none() {
            continue;
        }
        let (network, names, adcs, dacs): &mut (
            Network,
            BTreeMap<String, NetId>,
            Vec<AdcBinding>,
            Vec<DacBinding>,
        ) = topology.get_or_insert_with(Default::default);
        if let Some(receiver) = receiver {
            bind_receiver(receiver, network, names, adcs, limits)?;
        }
        if let Some((device, binding)) = voltage {
            let net = digital_net(network, names, &binding.net, limits.nodes)?;
            dacs.push(DacBinding {
                name: device.name.clone(),
                net,
                config: binding.config,
                port: device.port.clone(),
            });
        }
    }
    Ok(topology.map(|(network, _, adcs, dacs)| NativeTopology::new(network, adcs, dacs)))
}
