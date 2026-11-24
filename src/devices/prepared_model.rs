//! Native prepared-model kernels supplied by an independent model compiler.
//!
//! The device engine owns MNA rows, accepted/trial history and integration.
//! The model compiler owns physical I/Q laws and their analytic Jacobians.
//! This ABI is native Rust: no runtime deck parsing, plugin FFI or callbacks
//! per scalar component. A family can implement homogeneous SIMD tiles.

pub mod batch;

use std::cell::RefCell;
use std::fmt::Debug;
use std::sync::Arc;

use crate::devices::linear::LinearContext;
use crate::devices::noise::{DeviceNoise, NoiseContext, NoiseFamily, NoiseSource};
use crate::devices::traits::{AnalysisMode, Device, StampContext};
use crate::maths::Vector;
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

/// The distinguished version of the native prepared-model contract.
pub const ABI_VERSION: u32 = 2;

const CELSIUS_TO_KELVIN: Real = 273.15;

/// Physical support must be explicit: unavailable is different from known-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capability {
    /// The model declares that this contribution is physically zero.
    Zero,
    /// The model computes and supplies the contribution and its derivatives.
    Computed,
    /// The artifact cannot represent this physical contribution.
    Unavailable,
}

/// A parameter slot is resolved before model evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct Parameter {
    /// Stable, lower-case parameter identifier.
    pub name: String,
    /// Default physical value as encoded in the source artifact.
    pub default: Real,
    /// Whether changing this slot can alter the number of states or terminals.
    pub changes_topology: bool,
}

/// Physical equation identity; not every MNA row carries electrical current.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EquationKind {
    /// Electrical KCL in amperes; dynamic storage is charge in coulombs.
    Electrical,
    /// A heat balance in watts; dynamic storage is energy in joules.
    Thermal,
    /// A generalized internal equation (e.g. a recovery-state relaxation).
    Internal,
}

/// One engine-bound MNA coordinate and its equation's physical unit domain.
/// All binding ordinals are frozen before any Newton evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Equation {
    /// Stable, lowercase unique name in this artifact.
    pub name: String,
    /// Whether the equation conserves electrical charge or evolves another state.
    pub kind: EquationKind,
}

/// Self-describing native model artifact metadata. Values are not SPICE cards.
#[derive(Clone, Debug, PartialEq)]
pub struct Descriptor {
    /// Model family identifier.
    pub family: String,
    /// Compiler-controlled content fingerprint.
    pub fingerprint: [u8; 16],
    /// Native interface version.
    pub abi: u32,
    /// Ordered electrical, thermal and hidden-state MNA coordinates.
    pub equations: Vec<Equation>,
    /// Ordered input parameter slots, without textual lookups in Newton loads.
    pub parameters: Vec<Parameter>,
    /// Available stored charge/flux physics.
    pub charge: Capability,
    /// Available noise physics (a separate noise evaluation is required).
    pub noise: Capability,
}

impl Descriptor {
    /// Validate the schema before a model or instance is accepted.
    ///
    /// # Errors
    /// Unsupported ABI, invalid or ambiguous schema and missing current laws.
    pub fn validate(&self) -> SpiceResult<()> {
        if self.abi != ABI_VERSION
            || self.family.is_empty()
            || self.equations.len() < 2
            || self.fingerprint == [0; 16]
            || self
                .equations
                .len()
                .checked_mul(self.equations.len())
                .is_none()
            || self
                .equations
                .iter()
                .filter(|slot| slot.kind == EquationKind::Electrical)
                .count()
                < 2
        {
            return Err(SpiceError::circuit(
                "invalid prepared model ABI, family or equation slots",
            ));
        }
        for (index, equation) in self.equations.iter().enumerate() {
            if equation.name.is_empty()
                || equation.name != equation.name.to_ascii_lowercase()
                || self.equations[..index]
                    .iter()
                    .any(|previous| previous.name == equation.name)
            {
                return Err(SpiceError::circuit(format!(
                    "{}: invalid or duplicate equation slot '{}'",
                    self.family, equation.name
                )));
            }
        }
        for (index, slot) in self.parameters.iter().enumerate() {
            if slot.name.is_empty()
                || slot.name != slot.name.to_ascii_lowercase()
                || !slot.default.is_finite()
                || self.parameters[..index]
                    .iter()
                    .any(|previous| previous.name == slot.name)
            {
                return Err(SpiceError::circuit(format!(
                    "{}: invalid or duplicate parameter slot '{}'",
                    self.family, slot.name
                )));
            }
        }
        Ok(())
    }

    fn parameter_index(&self, name: &str) -> Option<usize> {
        self.parameters
            .iter()
            .position(|slot| slot.name.eq_ignore_ascii_case(name))
    }
}

/// A family-specific, statically dispatched, pure physical equation kernel.
///
/// Values and derivatives must be complete for every declared channel.
/// Preparation is called on parameter or temperature changes, never Newton
/// iteration; the prepared value must contain no mutable accepted history.
pub trait Kernel: Debug + Send + Sync + 'static {
    /// Immutable model/instance-specific derived data.
    type Prepared: Debug + Send + Sync + 'static;

    /// Derive immutable geometry and physical constants at absolute Kelvin.
    ///
    /// # Errors
    /// Invalid physical parameter domains or unsupported model features.
    fn prepare(&self, values: &[Real], temperature_kelvin: Real) -> SpiceResult<Self::Prepared>;

    /// Evaluate a point using only borrowed buffers.
    ///
    /// # Errors
    /// Nonfinite biases or unsupported physical regions.
    fn evaluate(
        &self,
        prepared: &Self::Prepared,
        voltages: &[Real],
        output: &mut Point<'_>,
    ) -> SpiceResult<()>;

    /// Evaluate a homogeneous prepared tile. A family may override this
    /// default scalar path with portable SIMD or its generated vector kernel.
    /// The batch wrapper validates every output and its conservation.
    ///
    /// # Errors
    /// Failed family physics for any lane.
    fn evaluate_batch(
        &self,
        prepared: &[Self::Prepared],
        voltages: &[Real],
        output: &mut batch::BatchOutput,
    ) -> SpiceResult<()> {
        batch::evaluate_scalar_tile(self, prepared, voltages, output)
    }

    /// Describe physical noise generators at a solved bias point. The engine
    /// owns adjoint transfer gains and frequency integration; generators
    /// belong to the external model family. Called outside Newton.
    ///
    /// # Errors
    /// Missing family noise support or invalid model physics.
    fn evaluate_noise(
        &self,
        _prepared: &Self::Prepared,
        _voltages: &[Real],
        _terminals: &[NodeId],
        _context: &NoiseContext<'_>,
        _sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<()> {
        Err(SpiceError::Unsupported {
            feature: "native model family has no noise kernel".into(),
            location: None,
        })
    }
}

/// Mutable view of a single prepared-kernel result; all arrays are row-major.
pub struct Point<'a> {
    /// Static residual in the physical units of the corresponding equation row.
    pub current: &'a mut [Real],
    /// Static residual Jacobian dF[row]/dx[col].
    pub conductance: &'a mut [Real],
    /// Generalized dynamic storage: electrical charge, thermal energy or state quantity.
    pub charge: &'a mut [Real],
    /// Dynamic storage Jacobian dQ[row]/dx[col].
    pub capacitance: &'a mut [Real],
}

impl Point<'_> {
    /// Validate the physical output and its exact declared capability.
    ///
    /// # Errors
    /// Nonfinite, incomplete, nonconservative electrical or gauge-variant outputs.
    pub fn validate(&self, descriptor: &Descriptor) -> SpiceResult<()> {
        let width = descriptor.equations.len();
        let capability = descriptor.charge;
        let size = width
            .checked_mul(width)
            .ok_or_else(|| SpiceError::circuit("prepared kernel Jacobian shape overflow"))?;
        if self.current.len() != width
            || self.conductance.len() != size
            || self.charge.len() != width
            || self.capacitance.len() != size
        {
            return Err(SpiceError::circuit("prepared kernel output shape mismatch"));
        }
        if self
            .current
            .iter()
            .chain(self.conductance.iter())
            .any(|v| !v.is_finite())
        {
            return Err(SpiceError::circuit(
                "missing or nonfinite prepared static residual/Jacobian",
            ));
        }
        if capability != Capability::Unavailable
            && self
                .charge
                .iter()
                .chain(self.capacitance.iter())
                .any(|v| !v.is_finite())
        {
            return Err(SpiceError::circuit(
                "missing or nonfinite prepared dynamic storage/Jacobian",
            ));
        }
        if capability == Capability::Zero
            && self
                .charge
                .iter()
                .chain(self.capacitance.iter())
                .any(|v| *v != 0.)
        {
            return Err(SpiceError::circuit(
                "known-zero storage model supplied nonzero state",
            ));
        }
        // A Kirchhoff group conserves only ELECTRICAL rows. A thermal
        // balance or recovery state cannot be summed with amperes.
        // Iterating the prevalidated schema must not allocate on Newton.
        let conservative = |values: &[Real], stride: usize, column: usize| {
            let mut sum = 0.;
            let mut scale = 0.;
            for (row, equation) in descriptor.equations.iter().enumerate() {
                if equation.kind == EquationKind::Electrical {
                    let value = values[row * stride + column];
                    sum += value;
                    scale += value.abs();
                }
            }
            sum.abs() <= 1e-12 * scale
        };
        // All equations, including thermal power and recovery, must be
        // unchanged by a global shift of electrical reference potential.
        let gauge_invariant = |jacobian: &[Real]| {
            (0..width).all(|row| {
                let mut sum = 0.;
                let mut scale = 0.;
                for (column, equation) in descriptor.equations.iter().enumerate() {
                    if equation.kind == EquationKind::Electrical {
                        let value = jacobian[row * width + column];
                        sum += value;
                        scale += value.abs();
                    }
                }
                sum.abs() <= 1e-12 * scale
            })
        };
        if !conservative(self.current, 1, 0)
            || (0..width).any(|column| !conservative(self.conductance, width, column))
            || !gauge_invariant(self.conductance)
            || (capability != Capability::Unavailable
                && (!conservative(self.charge, 1, 0)
                    || (0..width).any(|column| !conservative(self.capacitance, width, column))
                    || !gauge_invariant(self.capacitance)))
        {
            return Err(SpiceError::circuit(
                "prepared kernel violates electrical conservation or gauge invariance",
            ));
        }
        Ok(())
    }
}

/// Reusable point buffers, created at instance preparation, never per trial.
#[derive(Debug)]
pub struct PointBuffer {
    /// Static residual for every bound equation.
    pub current: Vec<Real>,
    /// Static Jacobian dF/dx for every pair of equation and coordinate.
    pub conductance: Vec<Real>,
    /// Generalized stored quantity per equation row.
    pub charge: Vec<Real>,
    /// Storage derivative matrix dQ/dx.
    pub capacitance: Vec<Real>,
}

impl PointBuffer {
    /// Allocate the exact output shape once per prepared instance/tile.
    ///
    /// # Errors
    /// The Jacobian shape overflows or allocation fails.
    pub fn new(width: usize) -> SpiceResult<Self> {
        let entries = width
            .checked_mul(width)
            .ok_or_else(|| SpiceError::circuit("prepared model Jacobian dimensions overflow"))?;
        let allocate = |count: usize| -> SpiceResult<Vec<Real>> {
            let mut values = Vec::new();
            values.try_reserve_exact(count).map_err(|error| {
                SpiceError::circuit(format!("cannot allocate prepared model workspace: {error}"))
            })?;
            values.resize(count, 0.);
            Ok(values)
        };
        Ok(Self {
            current: allocate(width)?,
            conductance: allocate(entries)?,
            charge: allocate(width)?,
            capacitance: allocate(entries)?,
        })
    }

    /// Reset all required outputs to missing sentinels before evaluation.
    pub fn reset(&mut self, capability: Capability) {
        self.current.fill(Real::NAN);
        self.conductance.fill(Real::NAN);
        let initial = if capability == Capability::Zero {
            0.
        } else {
            Real::NAN
        };
        self.charge.fill(initial);
        self.capacitance.fill(initial);
    }

    /// Create a borrowed, non-allocating writer for kernel evaluation.
    pub fn point(&mut self) -> Point<'_> {
        Point {
            current: &mut self.current,
            conductance: &mut self.conductance,
            charge: &mut self.charge,
            capacitance: &mut self.capacitance,
        }
    }
}

/// Shared prepared model artifact and immutable family implementation.
#[derive(Debug)]
pub struct Artifact<K: Kernel> {
    /// Validated artifact metadata.
    pub descriptor: Descriptor,
    kernel: K,
}

impl<K: Kernel> Artifact<K> {
    /// Construct and verify a model artifact outside any Newton iteration.
    ///
    /// # Errors
    /// Invalid or unsupported versioned artifact schema.
    pub fn new(descriptor: Descriptor, kernel: K) -> SpiceResult<Arc<Self>> {
        descriptor.validate()?;
        Ok(Arc::new(Self { descriptor, kernel }))
    }

    /// Build a prepared, topology-stable native device using the engine's
    /// Celsius temperature, converted to Kelvin for the physical kernel.
    ///
    /// # Errors
    /// Invalid equation shape, scalar parameters or absolute temperature.
    pub fn instance(
        self: &Arc<Self>,
        name: impl Into<String>,
        terminals: Vec<NodeId>,
        values: &[Real],
        temperature_celsius: Real,
    ) -> SpiceResult<PreparedDevice<K>> {
        if terminals.len() != self.descriptor.equations.len()
            || values.len() != self.descriptor.parameters.len()
            || !temperature_celsius.is_finite()
            || temperature_celsius <= -CELSIUS_TO_KELVIN
            || values.iter().any(|value| !value.is_finite())
        {
            return Err(SpiceError::circuit(
                "prepared model instance shape or values mismatch",
            ));
        }
        let temperature_kelvin = temperature_celsius + CELSIUS_TO_KELVIN;
        if !temperature_kelvin.is_finite() || temperature_kelvin <= 0. {
            return Err(SpiceError::circuit("invalid absolute prepared temperature"));
        }
        let prepared = self.kernel.prepare(values, temperature_kelvin)?;
        let width = terminals.len();
        Ok(PreparedDevice {
            name: name.into(),
            terminals: terminals.into(),
            artifact: Arc::clone(self),
            values: values.into(),
            temperature: temperature_celsius,
            prepared,
            voltage: RefCell::new(vec![0.; width]),
            output: RefCell::new(PointBuffer::new(width)?),
            dynamic: RefCell::new(vec![0.; width]),
        })
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
    prepared: K::Prepared,
    voltage: RefCell<Vec<Real>>,
    output: RefCell<PointBuffer>,
    dynamic: RefCell<Vec<Real>>,
}

impl<K: Kernel> PreparedDevice<K> {
    fn evaluate(&self, voltage: &[Real], output: &mut PointBuffer) -> SpiceResult<()> {
        output.reset(self.artifact.descriptor.charge);
        self.artifact
            .kernel
            .evaluate(&self.prepared, voltage, &mut output.point())?;
        output.point().validate(&self.artifact.descriptor)
    }

    /// Make a new immutable prepared parameter context for a candidate.
    /// A failed preparation leaves the source instance unchanged.
    ///
    /// # Errors
    /// Unknown, structural or nonphysical parameter updates.
    pub fn with_parameter(&self, slot: usize, value: Real) -> SpiceResult<Self> {
        let parameter = self
            .artifact
            .descriptor
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
        self.artifact.instance(
            self.name.clone(),
            self.terminals.to_vec(),
            &values,
            self.temperature,
        )
    }

    /// Reprepare a named scalar slot at a candidate boundary, never in Newton.
    ///
    /// # Errors
    /// Unknown, structural or nonphysical parameter updates.
    pub fn with_parameter_named(&self, name: &str, value: Real) -> SpiceResult<Self> {
        let slot = self
            .artifact
            .descriptor
            .parameter_index(name)
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

    fn check_charge(&self) -> SpiceResult<()> {
        if self.artifact.descriptor.charge == Capability::Unavailable {
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
        'x'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn limits_voltage_steps(&self) -> bool {
        false
    }
    fn state_count(&self) -> usize {
        if self.artifact.descriptor.charge == Capability::Computed {
            self.terminals.len() * 2
        } else {
            0
        }
    }

    fn validate_dc_inputs(&self, context: &crate::devices::ModelContext) -> SpiceResult<()> {
        self.check_temperature(context.temperature)
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
        for (value, &node) in voltage.iter_mut().zip(&self.terminals) {
            *value = context.node_voltage(node);
        }
        let mut output = self.output.borrow_mut();
        self.evaluate(&voltage, &mut output)?;
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
                if self.artifact.descriptor.charge == Capability::Computed {
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
        } else if self.artifact.descriptor.charge == Capability::Computed {
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
                linearized = derivative.mul_add(voltage[j], linearized);
            }
            context.stamp_rhs(row, linearized)?;
        }
        Ok(())
    }

    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        self.check_temperature(context.model_context.temperature)?;
        match self.artifact.descriptor.noise {
            Capability::Zero => Ok(DeviceNoise::Noiseless),
            Capability::Unavailable => Err(SpiceError::Unsupported {
                feature: format!("{}: missing prepared-model noise equations", self.name),
                location: None,
            }),
            Capability::Computed => {
                let mut voltage = self.voltage.borrow_mut();
                for (value, &node) in voltage.iter_mut().zip(&self.terminals) {
                    *value = context.voltage(node);
                }
                let start = sources.len();
                self.artifact.kernel.evaluate_noise(
                    &self.prepared,
                    &voltage,
                    &self.terminals,
                    context,
                    sources,
                )?;
                if sources.len() == start {
                    return Err(SpiceError::circuit(
                        "computed model noise has no generators",
                    ));
                }
                Ok(DeviceNoise::Sources {
                    family: NoiseFamily::Prepared,
                    model: Some(&self.artifact.descriptor.family),
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
        self.check_temperature(context.model_context.temperature)?;
        self.check_charge()?;
        let width = self.terminals.len();
        let mut voltage = self.voltage.borrow_mut();
        for (value, &node) in voltage.iter_mut().zip(&self.terminals) {
            *value = context
                .unknowns
                .node_row(node)
                .and_then(|row| bias.get(row))
                .unwrap_or(0.);
        }
        let mut output = self.output.borrow_mut();
        self.evaluate(&voltage, &mut output)?;
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
