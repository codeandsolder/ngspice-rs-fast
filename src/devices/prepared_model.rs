//! Native prepared-model kernels supplied by an independent model compiler.
//!
//! The device engine owns MNA rows, accepted/trial history and integration.
//! The model compiler owns physical I/Q laws and their analytic Jacobians.
//! This ABI is native Rust: no runtime deck parsing, plugin FFI or callbacks
//! per scalar component. A family can implement homogeneous SIMD tiles.

use std::cell::RefCell;
use std::sync::Arc;

use model_kernel_api::{
    Artifact, Binding, Capability, EquationKind, Kernel, PointBuffer, Prepared,
    noise::{Sink, Source},
};

use crate::devices::linear::LinearContext;
use crate::devices::noise::{DeviceNoise, NoiseContext, NoiseFamily, NoiseSource};
use crate::devices::nonlinear::charge::ChargeBranch;
use crate::devices::traits::{AnalysisMode, Device, StampContext};
use crate::maths::Vector;
use crate::primitives::{Complex, NodeAppend, NodeId, NodeKind, Real, SpiceError, SpiceResult};

const CELSIUS_TO_KELVIN: Real = 273.15;

fn physical_error(error: model_kernel_api::Error) -> SpiceError {
    SpiceError::circuit(error.to_string())
}

/// Instance ambient temperature policy, in Celsius.
#[derive(Debug, Clone, Copy)]
pub enum AmbientTemperature {
    /// Track the circuit temperature with an instance offset.
    CircuitOffset(Real),
    /// Independent absolute Celsius temperature.
    Fixed(Real),
}

impl AmbientTemperature {
    fn at(self, circuit: Real) -> Real {
        match self {
            Self::CircuitOffset(offset) => circuit + offset,
            Self::Fixed(value) => value,
        }
    }
}

/// Native prepared model integrated through the normal device trial lifecycle.
#[derive(Debug)]
pub struct PreparedDevice<K: Kernel> {
    name: String,
    terminals: Box<[NodeId]>,
    artifact: Arc<Artifact<K>>,
    values: Box<[Real]>,
    temperature: Real,
    ambient: AmbientTemperature,
    model_name: Option<String>,
    designator: char,
    c_device_type: crate::devices::setup::CDeviceType,
    noise_family: NoiseFamily,
    prepared: Prepared<K>,
    voltage: RefCell<Vec<Real>>,
    limited: bool,
    previous: RefCell<Vec<Real>>,
    target: RefCell<Vec<Real>>,
    setters: Box<[(&'static str, usize)]>,
    output: RefCell<PointBuffer>,
    dynamic: RefCell<Vec<Real>>,
}

impl<K: Kernel> PreparedDevice<K> {
    /// Bind all descriptor-ordered coordinates to a finalized topology.
    /// Thermal circuit voltages are Celsius; the physical artifact sees Kelvin.
    ///
    /// # Errors
    /// Incorrect binding shape, parameter domain, or allocation failure.
    pub fn new(
        artifact: Arc<Artifact<K>>,
        name: impl Into<String>,
        terminals: Vec<NodeId>,
        values: &[Real],
        temperature: Real,
    ) -> SpiceResult<Self> {
        Self::new_with_ambient(
            artifact,
            name.into(),
            terminals,
            values,
            temperature,
            AmbientTemperature::CircuitOffset(0.),
        )
    }

    fn new_with_ambient(
        artifact: Arc<Artifact<K>>,
        name: String,
        terminals: Vec<NodeId>,
        values: &[Real],
        temperature: Real,
        ambient: AmbientTemperature,
    ) -> SpiceResult<Self> {
        let width = artifact.descriptor().equations.len();
        if terminals.len() != width {
            return Err(SpiceError::circuit("prepared model binding shape"));
        }
        let prepared = artifact
            .prepare(values, ambient.at(temperature) + CELSIUS_TO_KELVIN)
            .map_err(physical_error)?;
        Ok(Self {
            name,
            terminals: terminals.into(),
            values: values.into(),
            temperature,
            ambient,
            model_name: None,
            designator: 'x',
            c_device_type: crate::devices::setup::CDeviceType::CodeModel,
            noise_family: NoiseFamily::Prepared,
            prepared,
            artifact,
            voltage: RefCell::new(vec![0.; width]),
            limited: false,
            previous: RefCell::new(vec![0.; width]),
            target: RefCell::new(vec![0.; width]),
            setters: Box::new([]),
            output: RefCell::new(PointBuffer::new(width).map_err(physical_error)?),
            dynamic: RefCell::new(vec![0.; width]),
        })
    }

    /// Bind exposed terminals in descriptor order and allocate private
    /// coordinates transactionally. Hidden names cannot alias existing nodes.
    ///
    /// # Errors
    /// Incorrect terminal count, node collision, or invalid preparation.
    pub fn bind(
        artifact: Arc<Artifact<K>>,
        name: &str,
        terminal_names: &[String],
        nodes: &mut NodeAppend<'_>,
        values: &[Real],
        temperature: Real,
        ambient: AmbientTemperature,
    ) -> SpiceResult<Self> {
        nodes.transaction(|nodes| {
            let mut exposed = terminal_names.iter();
            let mut rows = Vec::with_capacity(artifact.descriptor().equations.len());
            for equation in &artifact.descriptor().equations {
                let row =
                    match equation.binding {
                        Binding::Terminal => nodes.intern(exposed.next().ok_or_else(|| {
                            SpiceError::circuit("missing prepared model terminal")
                        })?)?,
                        Binding::Hidden => {
                            let hidden = format!("{name}#{}", equation.name);
                            if nodes.get(&hidden).is_some() {
                                return Err(SpiceError::circuit(format!(
                                    "prepared model hidden node collision: {hidden}"
                                )));
                            }
                            let row = nodes.intern(&hidden)?;
                            nodes.set_kind(row, NodeKind::Internal)?;
                            row
                        }
                    };
                rows.push(row);
            }
            if exposed.next().is_some() {
                return Err(SpiceError::circuit("extra prepared model terminal"));
            }
            Self::new_with_ambient(
                artifact,
                name.to_owned(),
                rows,
                values,
                temperature,
                ambient,
            )
        })
    }

    /// Identify a source-level device without changing its bound equations.
    #[must_use]
    pub fn with_identity(
        mut self,
        model_name: String,
        designator: char,
        c_device_type: crate::devices::setup::CDeviceType,
        noise_family: NoiseFamily,
    ) -> Self {
        self.model_name = Some(model_name);
        self.designator = designator;
        self.c_device_type = c_device_type;
        self.noise_family = noise_family;
        self
    }

    /// Enable coordinate-domain Newton step limiting for strongly nonlinear kernels.
    #[must_use]
    pub fn with_limiting(mut self) -> Self {
        self.limited = true;
        self
    }

    /// Map a source-level instance keyword to an immutable artifact slot.
    ///
    /// # Errors
    /// Unknown or topology-changing parameter slot.
    pub fn with_setter(mut self, keyword: &'static str, slot: usize) -> SpiceResult<Self> {
        if self
            .artifact
            .descriptor()
            .parameters
            .get(slot)
            .is_none_or(|p| p.changes_topology)
        {
            return Err(SpiceError::circuit("invalid prepared model setter slot"));
        }
        let mut setters = self.setters.into_vec();
        setters.push((keyword, slot));
        self.setters = setters.into_boxed_slice();
        Ok(self)
    }

    fn coordinates(&self, raw: impl Fn(NodeId) -> Real, voltage: &mut [Real]) {
        for ((value, &node), equation) in voltage
            .iter_mut()
            .zip(&self.terminals)
            .zip(&self.artifact.descriptor().equations)
        {
            *value = raw(node);
            if equation.kind == EquationKind::Thermal {
                *value += CELSIUS_TO_KELVIN;
            }
        }
    }

    fn circuit_coordinate(&self, index: usize, value: Real) -> Real {
        if self.artifact.descriptor().equations[index].kind == EquationKind::Thermal {
            value - CELSIUS_TO_KELVIN
        } else {
            value
        }
    }

    fn evaluate(&self, voltage: &[Real], output: &mut PointBuffer) -> SpiceResult<()> {
        self.artifact
            .evaluate(&self.prepared, voltage, &mut output.point())
            .map_err(physical_error)
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
        self.reprepared(&values, self.temperature)
    }

    fn reprepared(&self, values: &[Real], temperature: Real) -> SpiceResult<Self> {
        let mut candidate = Self::new_with_ambient(
            Arc::clone(&self.artifact),
            self.name.clone(),
            self.terminals.to_vec(),
            values,
            temperature,
            self.ambient,
        )?;
        candidate.model_name.clone_from(&self.model_name);
        candidate.designator = self.designator;
        candidate.c_device_type = self.c_device_type;
        candidate.noise_family = self.noise_family;
        candidate.limited = self.limited;
        candidate.setters = self.setters.clone();
        Ok(candidate)
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
            .position(|slot| slot.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                SpiceError::circuit(format!(
                    "{}: unknown prepared model parameter {name}",
                    self.name
                ))
            })?;
        self.with_parameter(slot, value)
    }

    fn check_temperature(&self, context: Real) -> SpiceResult<()> {
        if self.temperature.to_bits() == context.to_bits() {
            Ok(())
        } else {
            Err(SpiceError::circuit(format!(
                "{}: prepared model temperature changed; reprepare before solving",
                self.name
            )))
        }
    }

    fn storage_slots(&self) -> usize {
        if self.artifact.descriptor().charge == Capability::Computed {
            self.terminals.len() * 2
        } else {
            0
        }
    }

    // Limit the full physical coordinate vector, preserving its gauge.
    // A failed physical evaluation may retry only along this segment;
    // structural/contract failures remain errors.
    fn evaluate_trial(
        &self,
        context: &mut StampContext<'_>,
        voltage: &mut [Real],
        output: &mut PointBuffer,
    ) -> SpiceResult<()> {
        if !self.limited {
            return self.evaluate(voltage, output);
        }
        let offset = self.storage_slots();
        let mut previous = self.previous.borrow_mut();
        let mut target = self.target.borrow_mut();
        target.copy_from_slice(voltage);
        let has_previous = context.states.device_limiting()
            && previous.iter_mut().enumerate().all(|(i, value)| {
                if let Some(old) = super::limiting::previous_finite(&context.states, offset + i) {
                    *value = old;
                    true
                } else {
                    false
                }
            });
        let mut factor: Real = 1.;
        if has_previous {
            let equations = &self.artifact.descriptor().equations;
            let reference = equations
                .iter()
                .position(|e| e.kind == EquationKind::Electrical);
            let (old_ref, new_ref) = reference.map_or((0., 0.), |i| (previous[i], target[i]));
            for (i, equation) in equations.iter().enumerate() {
                let (delta, bound) = match equation.kind {
                    EquationKind::Electrical => {
                        let old = previous[i] - old_ref;
                        ((target[i] - new_ref - old).abs(), 1. + 0.5 * old.abs())
                    }
                    EquationKind::Thermal => (
                        (target[i] - previous[i]).abs(),
                        10. + 0.2 * (previous[i] - CELSIUS_TO_KELVIN).abs(),
                    ),
                    _ => continue,
                };
                if delta > bound {
                    factor = factor.min(bound / delta);
                }
            }
        }
        for attempt in 0..=32 {
            if factor < 1. {
                for (i, value) in voltage.iter_mut().enumerate() {
                    *value = factor.mul_add(target[i] - previous[i], previous[i]);
                }
            }
            match self
                .artifact
                .evaluate(&self.prepared, voltage, &mut output.point())
            {
                Ok(()) => break,
                Err(model_kernel_api::Error::Domain(_)) if has_previous && attempt < 32 => {
                    factor *= 0.5;
                }
                Err(error) => return Err(physical_error(error)),
            }
        }
        if factor < 1. {
            context.states.report_nonconvergence()?;
        }
        for (i, &value) in voltage.iter().enumerate() {
            context.states.set(offset + i, value)?;
        }
        Ok(())
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
}

impl<K: Kernel> Device for PreparedDevice<K> {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        self.designator
    }
    fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }
    fn c_device_type(&self) -> crate::devices::setup::CDeviceType {
        self.c_device_type
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn limits_voltage_steps(&self) -> bool {
        self.limited
    }
    fn state_count(&self) -> usize {
        self.storage_slots()
            + if self.limited {
                self.terminals.len()
            } else {
                0
            }
    }

    fn truncation_slots(&self) -> Vec<usize> {
        (0..self.storage_slots()).step_by(2).collect()
    }

    fn has_context_preparation(&self) -> bool {
        true
    }

    fn prepare_context(
        &self,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<Box<dyn Device>>> {
        if context.temperature.to_bits() == self.temperature.to_bits() {
            Ok(None)
        } else {
            Ok(Some(Box::new(
                self.reprepared(&self.values, context.temperature)?,
            )))
        }
    }

    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        self.setters
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(keyword))
            .map(|(name, _)| *name)
    }

    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let slot = self
            .setters
            .iter()
            .find(|(name, _)| *name == parameter)
            .map(|(_, slot)| *slot)
            .ok_or_else(|| SpiceError::circuit("unknown prepared model instance setter"))?;
        let mut candidate = self.with_parameter(slot, value)?;
        if candidate.temperature.to_bits() != context.temperature.to_bits() {
            candidate = candidate.reprepared(&candidate.values, context.temperature)?;
        }
        Ok(Box::new(candidate))
    }

    fn validate_dc_inputs(&self, context: &crate::devices::ModelContext) -> SpiceResult<()> {
        self.check_temperature(context.temperature)
    }

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        for &row in &self.terminals {
            for &column in &self.terminals {
                setup.nodes(row, column);
            }
        }
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        self.check_temperature(context.temperature)?;
        if context.mode.is_ac() {
            return Err(SpiceError::circuit(
                "prepared model AC requires small-signal assembly",
            ));
        }
        let width = self.terminals.len();
        let mut voltage = self.voltage.borrow_mut();
        self.coordinates(|node| context.node_voltage(node), &mut voltage);
        let mut output = self.output.borrow_mut();
        self.evaluate_trial(context, &mut voltage, &mut output)?;
        let output = output.point();
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
                    let branch = ChargeBranch::new(2 * index, charge, 1.);
                    if context.states.initial_transient() {
                        branch.seed(&mut context.states)?;
                    }
                    let companion = branch.integrate(&mut context.states, coeff)?;
                    if context.states.initial_transient() {
                        branch.seed_derivative(&mut context.states, companion.derivative)?;
                    }
                    dynamic[index] = companion.derivative;
                    branch.store(&mut context.states)?;
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
                let derivative = if let Some(coeff) = context.integration {
                    coeff.ag()[0].mul_add(
                        output.capacitance[i * width + j],
                        output.conductance[i * width + j],
                    )
                } else {
                    output.conductance[i * width + j]
                };
                // Matrix is the Jacobian of total current (including
                // integrated charge), and RHS is its affine Newton source.
                context.stamp(row, col, derivative)?;
                linearized = derivative.mul_add(self.circuit_coordinate(j, voltage[j]), linearized);
            }
            context.stamp_rhs(row, linearized)?;
        }
        Ok(())
    }

    fn noise_depends_on_frequency(&self) -> bool {
        self.artifact.descriptor().noise == Capability::Computed
    }

    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        self.check_temperature(context.model_context.temperature)?;
        if self.artifact.descriptor().noise == Capability::Zero {
            return Ok(DeviceNoise::Noiseless);
        }
        let mut voltage = self.voltage.borrow_mut();
        self.coordinates(|node| context.voltage(node), &mut voltage);
        let start = sources.len();
        let mut sink = BoundSink {
            terminals: &self.terminals,
            start,
            sources,
        };
        let result = self.artifact.evaluate_noise(
            &self.prepared,
            &voltage,
            context.model_context.frequency,
            &mut sink,
        );
        if let Err(error) = result {
            sources.truncate(start);
            return Err(physical_error(error));
        }
        if sources.len() == start {
            return Ok(DeviceNoise::Noiseless);
        }
        Ok(DeviceNoise::Sources {
            family: self.noise_family,
            model: self
                .model_name
                .as_deref()
                .or(Some(&self.artifact.descriptor().family)),
            total: sources.len() - start > 1,
        })
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.check_temperature(context.model_context.temperature)?;
        self.check_charge()?;
        let width = self.terminals.len();
        let mut voltage = self.voltage.borrow_mut();
        self.coordinates(
            |node| {
                context
                    .unknowns
                    .node_row(node)
                    .and_then(|row| bias.get(row))
                    .unwrap_or(0.)
            },
            &mut voltage,
        );
        let mut output = self.output.borrow_mut();
        self.evaluate(&voltage, &mut output)?;
        let output = output.point();
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

struct BoundSink<'a> {
    terminals: &'a [NodeId],
    start: usize,
    sources: &'a mut Vec<NoiseSource>,
}

impl Sink for BoundSink<'_> {
    fn push(&mut self, source: Source<'_>) -> model_kernel_api::Result<()> {
        let suffix = format!("_{}", source.name);
        if self.sources[self.start..]
            .iter()
            .any(|item| item.suffix() == suffix)
        {
            return Err(model_kernel_api::Error::Domain(
                "duplicate prepared noise source name".into(),
            ));
        }
        let injections = source
            .injections
            .iter()
            .map(|term| {
                let node = self
                    .terminals
                    .get(term.row)
                    .ok_or(model_kernel_api::Error::Shape("noise row binding"))?;
                Ok((*node, Complex::new(term.re, term.im)))
            })
            .collect::<model_kernel_api::Result<Vec<_>>>()?;
        self.sources.push(NoiseSource::Generalized {
            suffix,
            density: source.density,
            injections,
        });
        Ok(())
    }
}
