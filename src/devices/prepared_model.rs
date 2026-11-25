//! Circuit-side binding for the shared native compact-model contract.
//! The engine owns numerical rows, integration and accepted/trial history.

use std::{cell::RefCell, sync::Arc};

use model_kernel_api::noise::{Sink, Source};
use model_kernel_api::{
    Artifact, Binding, Capability, EquationKind, Kernel, Point, PointBuffer, Prepared,
};

use crate::devices::linear::LinearContext;
use crate::devices::noise::{DeviceNoise, NoiseContext, NoiseCoordinate, NoiseFamily, NoiseSource};
use crate::devices::traits::{AnalysisMode, ChargeContext, Device, StampContext};
use crate::devices::{Circuit, elaboration::ElaborationSize};

pub mod batch;
use crate::maths::Vector;
use crate::primitives::{Complex, NodeAppend, NodeId, NodeKind, Real, SpiceError, SpiceResult};

const CELSIUS_TO_KELVIN: Real = 273.15;

fn kernel_error(error: model_kernel_api::Error) -> SpiceError {
    match error {
        model_kernel_api::Error::Domain(message) => {
            SpiceError::circuit(format!("kernel domain: {message}"))
        }
        other => SpiceError::circuit(other.to_string()),
    }
}

fn evaluation_error(error: model_kernel_api::Error) -> SpiceError {
    match error {
        model_kernel_api::Error::Domain(message) => SpiceError::Numerical {
            context: "prepared model evaluation".into(),
            message,
        },
        other => kernel_error(other),
    }
}

struct CircuitNoiseSink<'a> {
    terminals: &'a [NodeId],
    sources: &'a mut Vec<NoiseSource>,
}

impl Sink for CircuitNoiseSink<'_> {
    fn push(&mut self, source: Source<'_>) -> model_kernel_api::Result<()> {
        let injections = source
            .injections
            .iter()
            .map(|term| {
                (
                    NoiseCoordinate::Node(self.terminals[term.row]),
                    Complex::new(term.re, term.im),
                )
            })
            .collect();
        self.sources.push(NoiseSource::generalized(
            source.name,
            source.density,
            injections,
        ));
        Ok(())
    }
}

/// Instance temperature law, independent of a candidate's circuit temperature.
#[derive(Debug, Clone, Copy)]
pub enum AmbientTemperature {
    /// The circuit ambient itself.
    Circuit,
    /// Fixed absolute instance temperature.
    AbsoluteKelvin(Real),
    /// Offset from circuit ambient.
    DeltaKelvin(Real),
}

/// Immutable temperature context at the preparation boundary.
///
/// Circuit nodes carrying thermal quantities use a rise above the common circuit ambient;
/// the kernel sees absolute kelvin. The model's instance ambient may differ.
#[derive(Debug, Clone, Copy)]
pub struct InstanceTemperature {
    /// Circuit temperature in degrees Celsius.
    pub circuit_celsius: Real,
    /// Instance temperature law.
    pub ambient: AmbientTemperature,
}

impl InstanceTemperature {
    /// Prepare at the circuit temperature without instance overrides.
    #[must_use = "the returned temperature policy carries the circuit ambient temperature"]
    pub const fn circuit(celsius: Real) -> Self {
        Self {
            circuit_celsius: celsius,
            ambient: AmbientTemperature::Circuit,
        }
    }

    /// Effective instance ambient in absolute kelvin.
    #[must_use = "the ambient temperature in kelvin is required for physical kernel evaluation"]
    pub fn ambient_kelvin(self) -> Real {
        match self.ambient {
            AmbientTemperature::Circuit => self.circuit_celsius + CELSIUS_TO_KELVIN,
            AmbientTemperature::AbsoluteKelvin(value) => value,
            AmbientTemperature::DeltaKelvin(delta) => {
                self.circuit_celsius + CELSIUS_TO_KELVIN + delta
            }
        }
    }

    const fn at(self, celsius: Real) -> Self {
        Self {
            circuit_celsius: celsius,
            ..self
        }
    }
}

/// Source-side scalar name and unit conversion for one prepared slot.
#[derive(Debug)]
pub(super) struct SourceParameter {
    pub name: &'static str,
    pub slot: usize,
    pub offset: Real,
}

/// Source identity and contextual bindings, independent of equation stamping.
#[derive(Debug)]
pub(super) struct SourceBinding {
    pub model_name: String,
    pub designator: char,
    pub voltage_step_limiting: bool,
    pub parameters: &'static [SourceParameter],
    pub gmin_slot: Option<usize>,
    pub model_ambient: AmbientTemperature,
    pub relative_temperature_slot: usize,
    pub absolute_temperature_slot: Option<usize>,
}

/// Native prepared model integrated through the normal device trial lifecycle.
#[derive(Debug)]
pub struct PreparedDevice<K: Kernel> {
    name: String,
    terminals: Box<[NodeId]>,
    artifact: Arc<Artifact<K>>,
    values: Box<[Real]>,
    temperature: InstanceTemperature,
    prepared: Prepared<K>,
    source: Option<Arc<SourceBinding>>,
    voltage: RefCell<Vec<Real>>,
    output: RefCell<PointBuffer>,
    dynamic: RefCell<Vec<Real>>,
}

impl<K: Kernel> PreparedDevice<K> {
    /// Bind an artifact whose coordinates are all external terminals.
    ///
    /// # Errors
    /// Invalid coordinate shape, temperature, or model parameters.
    pub fn new(
        artifact: &Arc<Artifact<K>>,
        name: impl Into<String>,
        terminals: Vec<NodeId>,
        values: &[Real],
        temperature: InstanceTemperature,
    ) -> SpiceResult<Self> {
        if artifact
            .descriptor()
            .equations
            .iter()
            .any(|e| e.binding == Binding::Hidden)
        {
            return Err(SpiceError::circuit(
                "hidden coordinates require circuit binding",
            ));
        }
        Self::from_rows(artifact, name.into(), terminals, values, temperature)
    }

    fn from_rows(
        artifact: &Arc<Artifact<K>>,
        name: String,
        terminals: Vec<NodeId>,
        values: &[Real],
        temperature: InstanceTemperature,
    ) -> SpiceResult<Self> {
        if terminals.len() != artifact.descriptor().equations.len() {
            return Err(SpiceError::circuit("prepared coordinate count mismatch"));
        }
        if !temperature.circuit_celsius.is_finite()
            || temperature.circuit_celsius <= -CELSIUS_TO_KELVIN
        {
            return Err(SpiceError::circuit("invalid circuit temperature"));
        }
        let prepared = artifact
            .prepare(values, temperature.ambient_kelvin())
            .map_err(kernel_error)?;
        let width = terminals.len();
        Ok(Self {
            name,
            terminals: terminals.into(),
            artifact: Arc::clone(artifact),
            values: values.into(),
            temperature,
            prepared,
            source: None,
            voltage: RefCell::new(vec![0.; width]),
            output: RefCell::new(PointBuffer::new(width).map_err(kernel_error)?),
            dynamic: RefCell::new(vec![0.; width]),
        })
    }

    /// Add a prepared instance and allocate its hidden equation rows atomically.
    /// Public terminals are supplied in descriptor order with hidden slots omitted.
    ///
    /// # Errors
    /// Invalid terminals, parameter domains, duplicate names or resource limits.
    pub fn bind(
        circuit: &mut Circuit,
        artifact: &Arc<Artifact<K>>,
        name: &str,
        terminals: &[NodeId],
        values: &[Real],
        temperature: InstanceTemperature,
    ) -> SpiceResult<()> {
        let descriptor = artifact.descriptor();
        let hidden = descriptor
            .equations
            .iter()
            .filter(|e| e.binding == Binding::Hidden)
            .count();
        if terminals.len() != descriptor.equations.len() - hidden {
            return Err(SpiceError::circuit(
                "prepared public terminal count mismatch",
            ));
        }
        let states = if descriptor.charge == Capability::Computed {
            2 * descriptor.equations.len()
        } else {
            0
        };
        circuit.add_prepared_devices(ElaborationSize::new(1, hidden, states), |nodes| {
            Ok(vec![Box::new(Self::bind_nodes(
                nodes,
                artifact,
                name,
                terminals,
                values,
                temperature,
            )?)])
        })
    }

    // Shared staged binding for directly added artifacts and ordinary netlist
    // factories. Node allocation is transactional, including preparation errors.
    pub(super) fn bind_nodes(
        nodes: &mut NodeAppend<'_>,
        artifact: &Arc<Artifact<K>>,
        name: &str,
        terminals: &[NodeId],
        values: &[Real],
        temperature: InstanceTemperature,
    ) -> SpiceResult<Self> {
        let descriptor = artifact.descriptor();
        let public_count = descriptor
            .equations
            .iter()
            .filter(|equation| equation.binding == Binding::Terminal)
            .count();
        if terminals.len() != public_count {
            return Err(SpiceError::circuit(
                "prepared public terminal count mismatch",
            ));
        }
        nodes.transaction(|nodes| {
            let mut public = terminals.iter().copied();
            let mut rows = Vec::with_capacity(descriptor.equations.len());
            for equation in &descriptor.equations {
                let node = match equation.binding {
                    Binding::Terminal => public
                        .next()
                        .ok_or_else(|| SpiceError::circuit("missing public terminal"))?,
                    Binding::Hidden => {
                        let label = format!("{name}#prepared_{}", equation.name);
                        if nodes.get(&label).is_some() {
                            return Err(SpiceError::circuit(format!(
                                "hidden coordinate collision: {label}"
                            )));
                        }
                        let node = nodes.intern(&label)?;
                        nodes.set_kind(node, NodeKind::Internal)?;
                        node
                    }
                };
                rows.push(node);
            }
            Self::from_rows(artifact, name.to_owned(), rows, values, temperature)
        })
    }

    pub(super) fn with_source(mut self, source: SourceBinding) -> Self {
        self.source = Some(Arc::new(source));
        self
    }

    fn rebound(&self, values: &[Real], temperature: InstanceTemperature) -> SpiceResult<Self> {
        let mut device = Self::from_rows(
            &self.artifact,
            self.name.clone(),
            self.terminals.to_vec(),
            values,
            temperature,
        )?;
        device.source = self.source.clone();
        Ok(device)
    }

    fn source_parameter(&self, name: &str) -> Option<&SourceParameter> {
        self.source
            .as_ref()?
            .parameters
            .iter()
            .find(|parameter| parameter.name.eq_ignore_ascii_case(name))
    }

    fn context_values(&self, context: &crate::devices::ModelContext) -> Vec<Real> {
        let mut values = self.values.to_vec();
        if let Some(slot) = self.source.as_ref().and_then(|source| source.gmin_slot) {
            values[slot] = context.gmin;
        }
        values
    }

    /// Make a new immutable prepared parameter context for a candidate.
    /// A failed preparation leaves the source instance unchanged.
    ///
    /// # Errors
    /// Unknown, structural or nonphysical parameter updates.
    pub fn with_parameter(&self, slot: usize, value: Real) -> SpiceResult<Self> {
        let parameter = self
            .artifact
            .descriptor()
            .parameters
            .get(slot)
            .ok_or_else(|| SpiceError::circuit("prepared model parameter ordinal out of range"))?;
        if parameter.changes_topology || !value.is_finite() {
            return Err(SpiceError::circuit(
                "structural or nonfinite model perturbation",
            ));
        }
        let mut values = self.values.to_vec();
        values[slot] = value;
        self.rebound(&values, self.temperature)
    }

    /// Reprepare a named scalar slot at a candidate boundary, never in Newton.
    ///
    /// # Errors
    /// Unknown, structural or nonphysical parameter updates.
    pub fn with_parameter_named(&self, name: &str, value: Real) -> SpiceResult<Self> {
        let slot = self
            .artifact
            .descriptor()
            .parameters
            .iter()
            .position(|p| p.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                SpiceError::circuit(format!(
                    "{}: unknown prepared model parameter {name}",
                    self.name
                ))
            })?;
        self.with_parameter(slot, value)
    }

    fn check_context(&self, temperature: Real, gmin: Real) -> SpiceResult<()> {
        if self.temperature.circuit_celsius.to_bits() != temperature.to_bits() {
            return Err(SpiceError::circuit(format!(
                "{}: prepared model temperature changed; reprepare before solving",
                self.name
            )));
        }
        if let Some(slot) = self.source.as_ref().and_then(|source| source.gmin_slot)
            && self.values[slot].to_bits() != gmin.to_bits()
        {
            return Err(SpiceError::circuit(format!(
                "{}: prepared model gmin changed; reprepare before solving",
                self.name
            )));
        }
        Ok(())
    }

    fn read_coordinates(&self, output: &mut [Real], voltage: impl Fn(NodeId) -> Real) {
        for ((value, &node), equation) in output
            .iter_mut()
            .zip(&self.terminals)
            .zip(&self.artifact.descriptor().equations)
        {
            *value = voltage(node);
            if equation.kind == EquationKind::Thermal {
                *value += self.temperature.circuit_celsius + CELSIUS_TO_KELVIN;
            }
        }
    }

    fn check_charge(&self) -> SpiceResult<()> {
        if self.artifact.descriptor().charge == Capability::Unavailable {
            return Err(SpiceError::circuit(format!(
                "{}: prepared model does not supply charge/flux equations",
                self.name
            )));
        }
        Ok(())
    }

    fn stamp_point(&self, context: &mut StampContext<'_>, output: &Point<'_>) -> SpiceResult<()> {
        let width = self.terminals.len();
        if context.mode.is_transient() {
            self.check_charge()?;
        }
        let mut dynamic = self.dynamic.borrow_mut();
        dynamic.fill(0.);
        if let AnalysisMode::Transient { dt, .. } = context.mode {
            let coeff = context.integration.ok_or_else(|| {
                SpiceError::circuit("prepared model transient load lacks integration coefficients")
            })?;
            if coeff.dt().to_bits() != dt.to_bits() {
                return Err(SpiceError::circuit(
                    "prepared model integration timestep mismatch",
                ));
            }
            for (index, &charge) in output.charge.iter().enumerate() {
                if self.artifact.descriptor().charge == Capability::Computed {
                    let mut history = crate::maths::integrator::ChargeHistory::new(charge);
                    for age in 1..=coeff.charge_history_len() {
                        history.push(context.states.accepted(age, 2 * index).ok_or_else(
                            || {
                                SpiceError::circuit(
                                    "prepared model missing accepted charge history",
                                )
                            },
                        )?)?;
                    }
                    let previous = if coeff.needs_previous_derivative() {
                        context.states.accepted(1, 2 * index + 1)
                    } else {
                        None
                    };
                    let companion = coeff.integrate(&history, previous, 1.)?;
                    dynamic[index] = companion.derivative;
                    context.states.set(2 * index, charge)?;
                    context.states.set(2 * index + 1, companion.derivative)?;
                }
            }
        } else if self.artifact.descriptor().charge == Capability::Computed {
            for (index, &charge) in output.charge.iter().enumerate() {
                context.states.set(2 * index, charge)?;
                context.states.set(2 * index + 1, 0.)?;
            }
        }
        for (i, &row) in self.terminals.iter().enumerate() {
            let mut linearized = -output.current[i] - dynamic[i];
            for (j, &col) in self.terminals.iter().enumerate() {
                let derivative = context.integration.map_or_else(
                    || output.conductance[i * width + j],
                    |coeff| {
                        coeff.ag()[0].mul_add(
                            output.capacitance[i * width + j],
                            output.conductance[i * width + j],
                        )
                    },
                );
                // Matrix is the Jacobian of total current (including
                // integrated charge), and RHS is its affine Newton source.
                context.stamp(row, col, derivative)?;
                linearized = derivative.mul_add(context.node_voltage(col), linearized);
            }
            context.stamp_rhs(row, linearized)?;
        }
        Ok(())
    }
}

impl<K: Kernel> PreparedDevice<K> {
    fn stamp_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        output: &Point<'_>,
    ) -> SpiceResult<()> {
        self.check_charge()?;
        let width = self.terminals.len();
        for (i, &row) in self.terminals.iter().enumerate() {
            for (j, &col) in self.terminals.iter().enumerate() {
                if let (Some(r), Some(c)) = (
                    context.unknowns.node_row(row),
                    context.unknowns.node_row(col),
                ) {
                    context
                        .system
                        .a
                        .add(r, c, output.conductance[i * width + j])?;
                    context
                        .system
                        .e
                        .add(r, c, output.capacitance[i * width + j])?;
                }
            }
        }
        Ok(())
    }
}

impl<K: Kernel> Device for PreparedDevice<K> {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        self.source.as_ref().map_or('x', |source| source.designator)
    }
    fn model_name(&self) -> Option<&str> {
        self.source
            .as_ref()
            .map(|source| source.model_name.as_str())
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn noise_depends_on_frequency(&self) -> bool {
        self.artifact.descriptor().noise == Capability::Computed
    }
    fn limits_voltage_steps(&self) -> bool {
        self.source
            .as_ref()
            .is_some_and(|source| source.voltage_step_limiting)
    }
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        for &row in &self.terminals {
            for &col in &self.terminals {
                setup.nodes(row, col);
            }
        }
        Ok(true)
    }
    fn state_count(&self) -> usize {
        if self.artifact.descriptor().charge == Capability::Computed {
            self.terminals.len() * 2
        } else {
            0
        }
    }

    fn truncation_slots(&self) -> Vec<usize> {
        if self.artifact.descriptor().charge == Capability::Computed {
            (0..self.terminals.len()).map(|index| 2 * index).collect()
        } else {
            Vec::new()
        }
    }

    fn validate_dc_inputs(&self, context: &crate::devices::ModelContext) -> SpiceResult<()> {
        self.check_context(context.temperature, context.gmin)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        self.check_context(context.temperature, context.gmin)?;
        if context.mode.is_ac() {
            return Err(SpiceError::circuit(
                "prepared model AC requires small-signal assembly",
            ));
        }
        let mut voltage = self.voltage.borrow_mut();
        self.read_coordinates(&mut voltage, |node| context.node_voltage(node));
        let mut buffer = self.output.borrow_mut();
        let mut output = buffer.point();
        self.artifact
            .evaluate(&self.prepared, &voltage, &mut output)
            .map_err(evaluation_error)?;
        self.stamp_point(context, &output)
    }

    fn bind_numeric_parameter(&self, name: &str) -> Option<crate::devices::sweep::ParameterKey> {
        if self.source.is_some() {
            return self
                .source_parameter(name)
                .map(|parameter| crate::devices::sweep::ParameterKey::Native(parameter.name));
        }
        self.artifact
            .descriptor()
            .parameters
            .iter()
            .position(|p| !p.changes_topology && p.name.eq_ignore_ascii_case(name))
            .map(crate::devices::sweep::ParameterKey::Prepared)
    }

    fn numeric_parameter_name(&self, key: crate::devices::sweep::ParameterKey) -> Option<&str> {
        let slot = match key {
            crate::devices::sweep::ParameterKey::Native(name) => {
                return self.source_parameter(name).map(|parameter| parameter.name);
            }
            crate::devices::sweep::ParameterKey::Prepared(slot) if self.source.is_none() => slot,
            _ => return None,
        };
        self.artifact
            .descriptor()
            .parameters
            .get(slot)
            .map(|p| p.name.as_str())
    }

    fn observation_parameter(
        &self,
        name: &str,
        _context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<Real>> {
        if let Some(source) = &self.source {
            if name.eq_ignore_ascii_case("temp") {
                let base = InstanceTemperature {
                    circuit_celsius: self.temperature.circuit_celsius,
                    ambient: source.model_ambient,
                }
                .ambient_kelvin();
                let effective = source.absolute_temperature_slot.map_or_else(
                    || base + self.values[source.relative_temperature_slot],
                    |slot| self.values[slot],
                );
                return Ok(Some(effective - CELSIUS_TO_KELVIN));
            }
            return Ok(self
                .source_parameter(name)
                .map(|parameter| self.values[parameter.slot] - parameter.offset));
        }
        Ok(self
            .artifact
            .descriptor()
            .parameters
            .iter()
            .position(|p| p.name.eq_ignore_ascii_case(name))
            .map(|slot| self.values[slot]))
    }

    fn with_bound_parameter(
        &self,
        key: crate::devices::sweep::ParameterKey,
        value: Real,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let (slot, value) = match key {
            crate::devices::sweep::ParameterKey::Native(name) => {
                let parameter = self
                    .source_parameter(name)
                    .ok_or_else(|| SpiceError::circuit("foreign source prepared setter"))?;
                (parameter.slot, value + parameter.offset)
            }
            crate::devices::sweep::ParameterKey::Prepared(slot) if self.source.is_none() => {
                (slot, value)
            }
            _ => return Err(SpiceError::circuit("foreign prepared setter")),
        };
        let parameter = self
            .artifact
            .descriptor()
            .parameters
            .get(slot)
            .ok_or_else(|| SpiceError::circuit("invalid prepared parameter slot"))?;
        if parameter.changes_topology || !value.is_finite() {
            return Err(SpiceError::circuit(
                "structural or nonfinite model perturbation",
            ));
        }
        let mut values = self.context_values(context);
        *values
            .get_mut(slot)
            .ok_or_else(|| SpiceError::circuit("invalid prepared parameter slot"))? = value;
        Ok(Box::new(self.rebound(
            &values,
            self.temperature.at(context.temperature),
        )?))
    }

    fn reprepare_context(
        &self,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<Box<dyn Device>>> {
        if self
            .check_context(context.temperature, context.gmin)
            .is_ok()
        {
            return Ok(None);
        }
        Ok(Some(Box::new(self.rebound(
            &self.context_values(context),
            self.temperature.at(context.temperature),
        )?)))
    }

    fn prepared_batch_key(&self) -> Option<batch::Key> {
        Some(batch::Key::new(&self.artifact))
    }

    fn prepared_recipe(&self) -> Option<batch::Recipe<'_>> {
        Some(batch::Recipe {
            key: batch::Key::new(&self.artifact),
            values: &self.values,
            ambient: self.temperature.ambient_kelvin(),
            circuit_temperature: self.temperature.circuit_celsius,
        })
    }

    fn prepare_batch(
        &self,
        instances: &[&dyn Device],
        unknowns: &crate::devices::MnaUnknowns,
    ) -> SpiceResult<Box<dyn batch::Tile>> {
        Ok(Box::new(batch::BoundTile::new(
            &self.artifact,
            instances,
            unknowns,
        )?))
    }

    fn stamp_prepared(&self, context: &mut StampContext<'_>, point: &Point<'_>) -> SpiceResult<()> {
        self.check_context(context.temperature, context.gmin)?;
        self.stamp_point(context, point)
    }

    fn assemble_prepared(
        &self,
        context: &mut LinearContext<'_>,
        point: &Point<'_>,
    ) -> SpiceResult<()> {
        self.check_context(
            context.model_context.temperature,
            context.model_context.gmin,
        )?;
        self.stamp_small_signal(context, point)
    }

    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        self.check_context(
            context.model_context.temperature,
            context.model_context.gmin,
        )?;
        match self.artifact.descriptor().noise {
            Capability::Zero => Ok(DeviceNoise::Noiseless),
            Capability::Unavailable => Err(SpiceError::Unsupported {
                feature: format!("{}: missing prepared-model noise equations", self.name),
                location: None,
            }),
            Capability::Computed => {
                let mut voltage = self.voltage.borrow_mut();
                self.read_coordinates(&mut voltage, |node| context.voltage(node));
                let start = sources.len();
                let mut sink = CircuitNoiseSink {
                    terminals: &self.terminals,
                    sources,
                };
                self.artifact
                    .evaluate_noise(
                        &self.prepared,
                        &voltage,
                        context.model_context.frequency,
                        &mut sink,
                    )
                    .map_err(evaluation_error)?;
                if sources.len() == start {
                    return Err(SpiceError::circuit(
                        "computed model noise has no generators",
                    ));
                }
                Ok(DeviceNoise::Sources {
                    family: NoiseFamily::Prepared,
                    model: Some(&self.artifact.descriptor().family),
                    total: sources.len() - start > 1,
                })
            }
        }
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.check_context(
            context.model_context.temperature,
            context.model_context.gmin,
        )?;
        self.check_charge()?;
        let mut voltage = self.voltage.borrow_mut();
        self.read_coordinates(&mut voltage, |node| {
            context
                .unknowns
                .node_row(node)
                .and_then(|row| bias.get(row))
                .unwrap_or(0.)
        });
        let mut buffer = self.output.borrow_mut();
        let mut output = buffer.point();
        self.artifact
            .evaluate(&self.prepared, &voltage, &mut output)
            .map_err(evaluation_error)?;
        self.stamp_small_signal(context, &output)
    }

    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }

    fn assemble_nonlinear_charge(&self, context: &mut ChargeContext<'_>) -> SpiceResult<()> {
        self.check_context(context.model.temperature, context.model.gmin)?;
        self.check_charge()?;
        let mut coordinates = self.voltage.borrow_mut();
        self.read_coordinates(&mut coordinates, |node| context.voltage(node));
        let mut buffer = self.output.borrow_mut();
        let mut point = buffer.point();
        self.artifact
            .evaluate(&self.prepared, &coordinates, &mut point)
            .map_err(evaluation_error)?;
        let width = self.terminals.len();
        for (i, &node) in self.terminals.iter().enumerate() {
            if let Some(row) = context.unknowns.node_row(node) {
                context.charge.as_mut_slice()[row] += point.charge[i];
                for (j, &node) in self.terminals.iter().enumerate() {
                    if let Some(col) = context.unknowns.node_row(node) {
                        context
                            .jacobian
                            .add(row, col, point.capacitance[i * width + j])?;
                    }
                }
            }
        }
        Ok(())
    }
}
