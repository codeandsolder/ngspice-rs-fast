//! Arbitrary-order proper rational transfer functions as native MNA equations.
//!
//! The coefficients describe H(s)=gain N(s/omega)/D(s/omega).
//! A controllable canonical realization uses normalized integrator coordinates
//! with tau=1/omega. It never evaluates a frequency polynomial in transient
//! analysis or constructs a separate AC-only response. Every state is part
//! of the ordinary accepted/trial history and the A+sE pencil.

use crate::devices::{
    Device, LinearContext, SourceKind, StampContext, Waveform,
    linear::{LinearSource, SourceRows},
    noise::{DeviceNoise, NoiseContext, NoiseSource},
    nonlinear::charge::ChargeBranch,
    setup::MatrixSetup,
};
use crate::maths::Vector;
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};
use std::sync::Arc;

/// Electrical port law. Current inputs insert an ideal zero-voltage
/// current-sensing branch; voltage inputs are unloaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortLaw {
    /// Differential voltage port.
    Voltage,
    /// Current from positive to negative.
    Current,
}

/// Electrical bindings of the two-port transfer.
#[derive(Debug, Clone, Copy)]
pub struct TransferPorts {
    /// Positive/negative output nodes.
    pub output: [NodeId; 2],
    /// Positive/negative input nodes.
    pub input: [NodeId; 2],
    /// Output quantity.
    pub output_law: PortLaw,
    /// Input quantity.
    pub input_law: PortLaw,
}

/// Cold transfer specification. Polynomial coefficients are in descending
/// powers, following the XSPICE s_xfer convention.
#[derive(Debug, Clone, Copy)]
pub struct RationalSpec<'a> {
    /// Numerator coefficients, highest degree first.
    pub numerator: &'a [Real],
    /// Denominator coefficients, highest degree first.
    pub denominator: &'a [Real],
    /// Positive reference angular frequency in rad/s. Use 1 for coefficients
    /// expressed directly in powers of s; normalization avoids needless
    /// powers of large/small physical frequencies in the realization.
    pub frequency_scale: Real,
    /// Overall transfer gain.
    pub gain: Real,
    /// Offset added to the sensed input before filtering.
    pub input_offset: Real,
    /// Optional explicit initial coordinates [x0, x1, ...] in normalized
    /// controllable canonical order (after gain, before output polynomial). Without these, nonzero poles start
    /// at equilibrium and origin-pole coordinates start at zero.
    pub initial: Option<&'a [Real]>,
}

/// An immutable, normalized state-space realization of a proper rational
/// function. No root finding, pole cancellation, or transient allocations.
#[derive(Debug, Clone)]
struct Realization {
    // D(z)=z^n+a[n-1]z^(n-1)+...+a[0], z=s/omega.
    denominator: Vec<Real>,
    // C_i=b_i-d*a_i, so y=C*x+d*u.
    output: Vec<Real>,
    feedthrough: Real,
    input_gain: Real,
    tau: Real,
    // A prefix of coordinates fixed in the initial DC solve.
    initial: Vec<Real>,
}

fn invalid(message: &str) -> SpiceError {
    SpiceError::circuit(format!("rational transfer: {message}"))
}

fn coefficients(values: &[Real]) -> SpiceResult<&[Real]> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(invalid("expected nonempty finite polynomial coefficients"));
    }
    // Retain one zero for a zero numerator, but not a zero denominator.
    let first = values
        .iter()
        .position(|value| *value != 0.)
        .unwrap_or(values.len() - 1);
    Ok(&values[first..])
}

fn normalized(value: Real, leading: Real, gain: Real) -> SpiceResult<Real> {
    let result = (value / leading) * gain;
    if !result.is_finite() || (value != 0. && gain != 0. && result == 0.) {
        return Err(invalid(
            "coefficient normalization is outside finite nonzero range",
        ));
    }
    Ok(result)
}

impl Realization {
    fn compile(spec: RationalSpec<'_>) -> SpiceResult<Self> {
        let numerator = coefficients(spec.numerator)?;
        let denominator = coefficients(spec.denominator)?;
        let leading = denominator[0];
        if leading == 0. {
            return Err(invalid("zero denominator"));
        }
        if numerator.len() > denominator.len() {
            return Err(invalid(
                "improper transfer requires explicit derivative operators",
            ));
        }
        let tau = 1. / spec.frequency_scale;
        if !spec.gain.is_finite() || !spec.input_offset.is_finite() || !tau.is_finite() || tau <= 0.
        {
            return Err(invalid(
                "gain/offset must be finite and frequency scale positive and representable",
            ));
        }
        let order = denominator.len() - 1;
        // The device reserves one branch row and two history slots per state.
        order
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("state dimension overflow"))?;
        let feedthrough = if numerator.len() == denominator.len() {
            normalized(numerator[0], leading, 1.)?
        } else {
            0.
        };
        let denominator = denominator[1..]
            .iter()
            .rev()
            .map(|&value| normalized(value, leading, 1.))
            .collect::<SpiceResult<Vec<_>>>()?;
        let output = denominator
            .iter()
            .enumerate()
            .map(|(index, &a)| {
                let b = numerator.iter().rev().nth(index).copied().unwrap_or(0.);
                let b = normalized(b, leading, 1.)?;
                let value = (-feedthrough).mul_add(a, b);
                if !value.is_finite() {
                    return Err(invalid("nonfinite output realization coefficient"));
                }
                Ok(value)
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        let origin_poles = denominator.iter().take_while(|&&value| value == 0.).count();
        let initial = if let Some(values) = spec.initial {
            if values.len() != order || values.iter().any(|value| !value.is_finite()) {
                return Err(invalid(
                    "explicit initial coordinates must match the finite state dimension",
                ));
            }
            values.to_vec()
        } else {
            vec![0.; origin_poles]
        };
        let feedthrough = feedthrough * spec.gain;
        let input_offset = spec.input_offset * spec.gain;
        if !feedthrough.is_finite()
            || !input_offset.is_finite()
            || !(feedthrough * spec.input_offset).is_finite()
        {
            return Err(invalid("nonfinite feedthrough offset"));
        }
        Ok(Self {
            denominator,
            output,
            feedthrough,
            input_gain: spec.gain,
            tau,
            initial,
        })
    }

    const fn order(&self) -> usize {
        self.denominator.len()
    }
}

/// Native rational transfer element with voltage/current ports. One physical branch
/// current followed by normalized state coordinates is allocated in MNA.
/// Stable, unstable, and marginal causal systems use the same equations;
/// no coefficient or pole is silently clamped.
#[derive(Debug, Clone)]
pub struct RationalTransfer {
    name: String,
    // Output positive/negative, input positive/negative.
    terminals: [NodeId; 4],
    output: PortLaw,
    input: PortLaw,
    model: Arc<Realization>,
    input_offset: Real,
}

impl RationalTransfer {
    /// Compile a proper rational function and bind its electrical ports.
    /// # Errors
    /// Invalid or unrepresentable coefficients, improper transfer,
    /// mismatched initial coordinates, or a shorted voltage output.
    pub fn new(
        name: impl Into<String>,
        ports: TransferPorts,
        spec: RationalSpec<'_>,
    ) -> SpiceResult<Self> {
        let TransferPorts {
            output,
            input,
            output_law,
            input_law,
        } = ports;
        if output_law == PortLaw::Voltage && output[0] == output[1] {
            return Err(invalid("shorted voltage output"));
        }
        if input_law == PortLaw::Current && input[0] == input[1] {
            return Err(invalid("shorted current input"));
        }
        let model = Arc::new(Realization::compile(spec)?);
        Ok(Self {
            name: name.into(),
            terminals: [output[0], output[1], input[0], input[1]],
            output: output_law,
            input: input_law,
            model,
            input_offset: spec.input_offset,
        })
    }

    /// Number of dynamic states in the retained realization.
    #[must_use]
    pub fn order(&self) -> usize {
        self.model.order()
    }

    /// Enumerate the exact common sparsity and coefficient stream. The DC
    /// initial constraints replace only the requested rows, never the AC
    /// or transient dynamics. The same stream builds setup, load and pencil.
    fn entries(
        &self,
        branch: usize,
        nodes: [Option<usize>; 4],
        dc: bool,
        mut emit: impl FnMut(usize, usize, Real, bool) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        let [op, on, ip, inn] = nodes;
        let mut term = |row: usize, column: Option<usize>, value: Real| -> SpiceResult<()> {
            if let Some(column) = column
                && value != 0.
            {
                emit(row, column, value, false)?;
            }
            Ok(())
        };
        for (node, sign) in [(op, 1.), (on, -1.)] {
            if let Some(row) = node {
                term(row, Some(branch), sign)?;
                if self.output == PortLaw::Voltage {
                    term(branch, Some(row), sign)?;
                }
            }
        }
        if self.output == PortLaw::Current {
            term(branch, Some(branch), 1.)?;
        }
        let sensor = branch + 1 + self.order();
        let input = if self.input == PortLaw::Current {
            for (node, sign) in [(ip, 1.), (inn, -1.)] {
                if let Some(row) = node {
                    term(row, Some(sensor), sign)?;
                    term(sensor, Some(row), sign)?;
                }
            }
            [(Some(sensor), -1.), (None, 0.)]
        } else {
            [(ip, -1.), (inn, 1.)]
        };
        for (node, sign) in input {
            term(branch, node, sign * self.model.feedthrough)?;
        }
        for (index, &coefficient) in self.model.output.iter().enumerate() {
            term(branch, Some(branch + 1 + index), -coefficient)?;
        }
        for index in 0..self.order() {
            let row = branch + 1 + index;
            if dc && index < self.model.initial.len() {
                term(row, Some(row), 1.)?;
                continue;
            }
            if index + 1 == self.order() {
                for (column, &coefficient) in self.model.denominator.iter().enumerate() {
                    term(row, Some(branch + 1 + column), coefficient)?;
                }
                for (node, sign) in input {
                    term(row, node, sign * self.model.input_gain)?;
                }
            } else {
                term(row, Some(row + 1), -1.)?;
            }
        }
        if !dc {
            for index in 0..self.order() {
                let row = branch + 1 + index;
                emit(row, row, self.model.tau, true)?;
            }
        }
        Ok(())
    }

    fn forcing(
        &self,
        branch: usize,
        dc: bool,
        mut emit: impl FnMut(usize, Real) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        let direct = self.model.feedthrough * self.input_offset;
        if direct != 0. {
            emit(branch, direct)?;
        }
        if dc {
            for (index, &value) in self.model.initial.iter().enumerate() {
                emit(branch + 1 + index, value)?;
            }
        }
        if self.order() > 0 && (!dc || self.model.initial.len() < self.order()) {
            emit(
                branch + self.order(),
                self.model.input_gain * self.input_offset,
            )?;
        }
        Ok(())
    }
}

impl Device for RationalTransfer {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'a'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn branch_currents(&self) -> usize {
        1 + self.order() + usize::from(self.input == PortLaw::Current)
    }
    fn findable_branch(&self) -> Option<usize> {
        Some(0)
    }
    fn state_count(&self) -> usize {
        2 * self.order()
    }
    fn truncation_slots(&self) -> Vec<usize> {
        (0..self.state_count()).step_by(2).collect()
    }

    fn setup_matrix(&self, setup: &mut MatrixSetup<'_>) -> SpiceResult<bool> {
        let branch = setup.branch(0)?;
        let nodes = self.terminals.map(|node| setup.node(node));
        self.entries(branch, nodes, false, |row, column, _, _| {
            setup.element(Some(row), Some(column));
            Ok(())
        })?;
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if context.mode.is_ac() {
            return Err(invalid("AC requires the complex pencil"));
        }
        if self.order() > 0 && context.states.initial_conditions() {
            return Err(invalid(
                "UIC needs an explicit transfer-coordinate initialization solve",
            ));
        }
        let branch = context.branch(0)?;
        let nodes = self.terminals.map(|node| context.unknowns.node_row(node));
        let dc = context.mode.is_dc();
        let coefficients = if dc {
            None
        } else {
            Some(
                context
                    .integration
                    .ok_or_else(|| invalid("missing integration coefficients"))?,
            )
        };
        let multiplier = coefficients.map_or(0., |coefficients| coefficients.ag()[0]);
        self.entries(branch, nodes, dc, |row, column, value, dynamic| {
            context.matrix.add(
                row,
                column,
                if dynamic { value * multiplier } else { value },
            )
        })?;
        self.forcing(branch, dc, |row, value| context.rhs.add_to(row, value))?;
        for index in 0..self.order() {
            let row = branch + 1 + index;
            let value = context
                .solution
                .get(row)
                .ok_or_else(|| invalid("missing state coordinate"))?;
            let storage = ChargeBranch::new(2 * index, value, 1.);
            if let Some(coefficients) = coefficients {
                let companion = storage.integrate(&mut context.states, coefficients)?;
                context
                    .rhs
                    .add_to(row, -self.model.tau * companion.current)?;
            } else {
                storage.zero_derivative(&mut context.states)?;
            }
            storage.store(&mut context.states)?;
        }
        Ok(())
    }

    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        let branch = context
            .branch
            .ok_or_else(|| invalid("missing transfer branch"))?;
        let nodes = self.terminals.map(|node| context.unknowns.node_row(node));
        self.entries(branch, nodes, false, |row, column, value, dynamic| {
            if dynamic {
                context.system.e.add(row, column, value)
            } else {
                context.system.a.add(row, column, value)
            }
        })?;
        if !self.model.initial.is_empty() {
            context.system.has_initial_conditions = true;
        }
        if self.input_offset != 0. {
            let mut rows = SourceRows::new();
            if self.model.feedthrough != 0. {
                rows.push(branch, self.model.feedthrough)?;
            }
            if self.order() > 0 {
                rows.push(branch + self.order(), self.model.input_gain)?;
            }
            context.system.sources.push(LinearSource {
                name: Arc::new(self.name.clone()),
                kind: SourceKind::Voltage,
                rows,
                dc: self.input_offset,
                ac: Complex::ZERO,
                waveform: Arc::new(Waveform::Constant(self.input_offset)),
            });
        }
        Ok(())
    }

    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble_linear(context)
    }

    fn noise(
        &self,
        _context: &NoiseContext<'_>,
        _sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        Ok(DeviceNoise::Noiseless)
    }
}

use crate::devices::models::{PreparationKey, ResolvedModel};
use crate::netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind};
use crate::primitives::{NodeAppend, parse_spice_number};

fn literal(parameter: &ParameterAssignment) -> SpiceResult<Real> {
    parse_spice_number(&parameter.value)
        .filter(|value| value.is_finite() && parameter.kind == ParameterKind::Scalar)
        .ok_or_else(|| {
            SpiceError::parse(
                parameter.location.clone(),
                "expected a literal transfer coefficient",
            )
        })
}

fn vector(parameter: &ParameterAssignment) -> SpiceResult<Vec<Real>> {
    let ParameterKind::NumericVector(entries) = &parameter.kind else {
        return Err(SpiceError::parse(
            parameter.location.clone(),
            "expected transfer coefficient vector",
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
                        "unresolved transfer coefficient",
                    )
                })
        })
        .collect()
}

#[derive(Debug, Clone)]
struct Model {
    realization: Arc<Realization>,
    input_offset: Real,
}

impl Model {
    fn compile(model: &ResolvedModel<'_>) -> SpiceResult<Self> {
        let mut numerator = None;
        let mut denominator = None;
        let mut initial = None;
        let mut gain = 1.;
        let mut scale = 1.;
        let mut offset = 0.;
        for parameter in &model.card().parameters {
            match parameter.name.as_str() {
                "num_coeff" => numerator = Some(vector(parameter)?),
                "den_coeff" => denominator = Some(vector(parameter)?),
                "int_ic" => initial = Some(vector(parameter)?),
                "gain" => gain = literal(parameter)?,
                "denormalized_freq" => scale = literal(parameter)?,
                "in_offset" => offset = literal(parameter)?,
                _ => {
                    return Err(SpiceError::parse(
                        parameter.location.clone(),
                        format!("unknown s_xfer parameter {}", parameter.name),
                    ));
                }
            }
        }
        let numerator = numerator.ok_or_else(|| invalid("s_xfer needs num_coeff"))?;
        let denominator = denominator.ok_or_else(|| invalid("s_xfer needs den_coeff"))?;
        let effective = coefficients(&denominator)?;
        let order = effective.len() - 1;
        let mut coordinates = initial.unwrap_or_else(|| vec![0.; order]);
        if coordinates.len() != order {
            return Err(invalid(
                "int_ic must contain one entry per integrator stage",
            ));
        }
        // XSPICE orders initial integrators highest derivative first, in
        // physical s-domain units. Normalize those coordinates without
        // changing the free response, including gain=0.
        coordinates.reverse();
        for (index, coordinate) in coordinates.iter_mut().enumerate() {
            if *coordinate != 0. {
                let power = scale.powi(
                    i32::try_from(index)
                        .map_err(|_| invalid("initial coordinate order overflow"))?,
                );
                *coordinate = normalized(*coordinate, power, effective[0])?;
            }
        }
        let realization = Realization::compile(RationalSpec {
            numerator: &numerator,
            denominator: &denominator,
            frequency_scale: scale,
            gain,
            input_offset: offset,
            initial: Some(&coordinates),
        })?;
        Ok(Self {
            realization: Arc::new(realization),
            input_offset: offset,
        })
    }
}

/// Bind a parsed XSPICE s_xfer model. Its immutable realization is shared
/// across instances; accepted histories remain entirely device-local.
pub(in crate::devices) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
) -> SpiceResult<Box<dyn Device>> {
    let [op, on, ip, inn] = instance.nodes.as_slice() else {
        return Err(invalid("expected two differential ports"));
    };
    let [input, output] = instance.parameters.as_slice() else {
        return Err(invalid("expected typed transfer port modes"));
    };
    let port = |parameter: &ParameterAssignment, name: &str| match (
        parameter.name.as_str(),
        &parameter.kind,
        parameter.value.as_str(),
    ) {
        (actual, ParameterKind::Textual, "voltage") if actual == name => Ok(PortLaw::Voltage),
        (actual, ParameterKind::Textual, "current") if actual == name => Ok(PortLaw::Current),
        _ => Err(invalid("invalid transfer port mode")),
    };
    let input = port(input, "input")?;
    let output = port(output, "output")?;
    let prepared = model.prepare(PreparationKey::Inputs, || Model::compile(model))?;
    nodes.transaction(|nodes| {
        let terminals = [
            nodes.intern(op)?,
            nodes.intern(on)?,
            nodes.intern(ip)?,
            nodes.intern(inn)?,
        ];
        if (output == PortLaw::Voltage && terminals[0] == terminals[1])
            || (input == PortLaw::Current && terminals[2] == terminals[3])
        {
            return Err(invalid("shorted ideal transfer port"));
        }
        Ok(Box::new(RationalTransfer {
            name: instance.name.clone(),
            terminals,
            output,
            input,
            model: Arc::clone(&prepared.realization),
            input_offset: prepared.input_offset,
        }) as Box<dyn Device>)
    })
}
