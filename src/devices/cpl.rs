//! Multiconductor distributed transmission lines with coupled RLGC physics.
//!
//! The line's physical node connection is N input conductors and one input
//! return, then N output conductors and one output return. Currents at both
//! ends are defined entering the line. General noncommuting matrices are
//! handled by one 2N-dimensional telegrapher state-space propagator.
pub mod plan;

use std::sync::Arc;

use crate::devices::linear::LinearContext;
use crate::devices::models::ResolvedModel;
use crate::devices::traits::{AnalysisMode, StampContext};
use crate::devices::{Device, MnaUnknowns};
use crate::maths::{Vector, dense_complex::DenseComplex};
use crate::netlist::ast::{DeviceInstance, ParameterKind};
use crate::primitives::{
    Complex, NodeAppend, NodeId, Real, SpiceError, SpiceResult, parse_spice_number,
};
use plan::{CoupledLineLimits, CoupledLinePlan};

fn invalid(message: impl std::fmt::Display) -> SpiceError {
    SpiceError::circuit(format!("coupled transmission line: {message}"))
}

/// A physically connected N-conductor distributed transmission line.
///
/// The material plan is immutable. Instantiations may share one plan and
/// use distinct node attachments or length-adjusted material plans.
#[derive(Debug, Clone)]
pub struct CoupledLine {
    name: String,
    /// Shared, validated R/L/G/C and length.
    pub plan: Arc<CoupledLinePlan>,
    input: Vec<NodeId>,
    output: Vec<NodeId>,
    input_return: NodeId,
    output_return: NodeId,
    terminals: Vec<NodeId>,
}
type CplMaterialVectors = std::collections::BTreeMap<String, Vec<Real>>;

impl CoupledLine {
    /// Bind one P instance to an evaluated CPL model and physical node table.
    ///
    /// Model coefficients are prepared before any node insertion. The
    /// caller owns a node-append transaction, so failures never publish a
    /// partially elaborated cable. Length on the instance overrides model.
    /// # Errors
    /// Unknown model setters, malformed numeric vector, topology or physics.
    pub fn instantiate(
        instance: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        model: &ResolvedModel<'_>,
    ) -> SpiceResult<Self> {
        let count = instance.nodes.len();
        if count < 4 || count % 2 != 0 || model.family() != crate::devices::ModelFamily::Cpl {
            return Err(invalid(
                "P instance requires N input/output conductors and two returns",
            ));
        }
        let conductors = (count - 2) / 2;
        let expected = conductors
            .checked_add(1)
            .and_then(|v| v.checked_mul(conductors))
            .map(|v| v / 2)
            .ok_or_else(|| invalid("CPL matrix dimensions overflow"))?;
        let (mut length, arrays) = Self::prepare_model(model, expected)?;
        for parameter in &instance.parameters {
            if parameter.name != "length" {
                return Err(SpiceError::parse(
                    parameter.location.clone(),
                    format!("unsupported P instance parameter {}", parameter.name),
                ));
            }
            length = Some(Self::scalar_parameter(parameter)?);
        }
        let len = length.ok_or_else(|| invalid("CPL needs a model or instance length"))?;
        let zero = vec![0.; expected];
        let inductance = arrays
            .get("l")
            .ok_or_else(|| invalid("CPL missing inductance matrix"))?;
        let capacitance = arrays
            .get("c")
            .ok_or_else(|| invalid("CPL missing capacitance matrix"))?;
        let resistance = arrays.get("r").unwrap_or(&zero);
        let conductance = arrays.get("g").unwrap_or(&zero);
        let plan = Arc::new(CoupledLinePlan::from_packed(
            conductors,
            len,
            resistance,
            inductance,
            conductance,
            capacitance,
            CoupledLineLimits::default(),
        )?);
        nodes.transaction(|nodes| {
            let terminals = instance
                .nodes
                .iter()
                .map(|name| nodes.intern(name))
                .collect::<SpiceResult<Vec<_>>>()?;
            Self::new(
                instance.name.clone(),
                Arc::clone(&plan),
                terminals[..conductors].to_vec(),
                terminals[conductors],
                terminals[conductors + 1..conductors + 1 + conductors].to_vec(),
                terminals[count - 1],
            )
        })
    }

    fn prepare_model(
        model: &ResolvedModel<'_>,
        expected: usize,
    ) -> SpiceResult<(Option<Real>, CplMaterialVectors)> {
        let mut length = None;
        let mut arrays = std::collections::BTreeMap::<String, Vec<Real>>::new();
        for parameter in &model.card().parameters {
            if parameter.name == "length" {
                length = Some(Self::scalar_parameter(parameter)?);
            } else if ["r", "l", "g", "c"].contains(&parameter.name.as_str()) {
                let ParameterKind::NumericVector(entries) = &parameter.kind else {
                    return Err(SpiceError::parse(
                        parameter.location.clone(),
                        "CPL material must be a complete numeric upper-triangle vector",
                    ));
                };
                if entries.len() != expected {
                    return Err(SpiceError::parse(
                        parameter.location.clone(),
                        format!(
                            "CPL {} requires {expected} upper-triangle coefficients, got {}",
                            parameter.name,
                            entries.len()
                        ),
                    ));
                }
                let mut values = Vec::new();
                values
                    .try_reserve(expected)
                    .map_err(|_| invalid("CPL coefficient storage allocation"))?;
                for entry in entries {
                    if entry.expression.is_some() {
                        return Err(SpiceError::parse(
                            entry.source.location.clone(),
                            "unresolved CPL vector expression",
                        ));
                    }
                    values.push(
                        parse_spice_number(&entry.source.text)
                            .filter(|v| v.is_finite())
                            .ok_or_else(|| {
                                SpiceError::parse(
                                    entry.source.location.clone(),
                                    format!("nonfinite CPL coefficient {}", entry.source.text),
                                )
                            })?,
                    );
                }
                arrays.insert(parameter.name.clone(), values);
            } else {
                return Err(SpiceError::parse(
                    parameter.location.clone(),
                    format!("unknown CPL material setter {}", parameter.name),
                ));
            }
        }
        Ok((length, arrays))
    }

    fn scalar_parameter(parameter: &crate::netlist::ast::ParameterAssignment) -> SpiceResult<Real> {
        if parameter.kind != ParameterKind::Scalar {
            return Err(SpiceError::parse(
                parameter.location.clone(),
                "CPL length requires a literal scalar",
            ));
        }
        parse_spice_number(&parameter.value)
            .filter(|v| v.is_finite())
            .ok_or_else(|| SpiceError::parse(parameter.location.clone(), "nonfinite CPL length"))
    }

    /// Construct one line with unambiguous conductor and return ordering.
    ///
    /// # Errors
    /// Invalid physical conductor bindings or missing model name.
    pub fn new(
        name: impl Into<String>,
        plan: Arc<CoupledLinePlan>,
        input: Vec<NodeId>,
        input_return: NodeId,
        output: Vec<NodeId>,
        output_return: NodeId,
    ) -> SpiceResult<Self> {
        let name = name.into();
        if name.is_empty()
            || input.len() != plan.conductors()
            || output.len() != plan.conductors()
            || input.contains(&input_return)
            || output.contains(&output_return)
        {
            return Err(invalid("invalid conductor/return terminal layout"));
        }
        let capacity = input
            .len()
            .checked_mul(2)
            .and_then(|count| count.checked_add(2))
            .ok_or_else(|| invalid("port terminal count overflow"))?;
        let mut terminals = Vec::new();
        terminals
            .try_reserve(capacity)
            .map_err(|_| invalid("terminal allocation failed"))?;
        terminals.extend_from_slice(&input);
        terminals.push(input_return);
        terminals.extend_from_slice(&output);
        terminals.push(output_return);
        Ok(Self {
            name,
            plan,
            input,
            output,
            input_return,
            output_return,
            terminals,
        })
    }

    /// Total physical conductor count, excluding the two return terminals.
    #[must_use]
    pub fn conductors(&self) -> usize {
        self.plan.conductors()
    }

    /// Borrow the model without copying its per-length matrices.
    #[must_use]
    pub fn plan(&self) -> &CoupledLinePlan {
        &self.plan
    }

    fn physical_pair(&self, port: usize) -> (NodeId, NodeId) {
        let n = self.conductors();
        if port < n {
            (self.input[port], self.input_return)
        } else {
            (self.output[port - n], self.output_return)
        }
    }

    /// Physical port current vs voltage matrix at a frequency.
    ///
    /// # Errors
    /// Invalid frequency or singular distributed transfer.
    pub fn admittance_at(&self, hz: Real) -> SpiceResult<DenseComplex> {
        self.plan.admittance_at(hz)
    }

    /// The exact KCL stamp order, omitting ground rows and columns.
    ///
    /// # Errors
    /// Failed propagation, invalid MNA bindings or work limits.
    pub fn nodal_stamps_at(
        &self,
        hz: Real,
        unknowns: &MnaUnknowns,
    ) -> SpiceResult<Vec<(usize, usize, Complex)>> {
        let y = self.admittance_at(hz)?;
        let ports = self.conductors() * 2;
        let mut positions = Vec::new();
        positions
            .try_reserve(ports)
            .map_err(|_| invalid("nodal mapping allocation failed"))?;
        for p in 0..ports {
            let (positive, negative) = self.physical_pair(p);
            positions.push((
                unknowns.checked_node_row(positive)?,
                unknowns.checked_node_row(negative)?,
            ));
        }
        let total = ports
            .checked_mul(ports)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| invalid("stamp count overflow"))?;
        let mut entries = Vec::new();
        entries
            .try_reserve(total)
            .map_err(|_| invalid("stamp allocation failed"))?;
        for (r, &(rp, rn)) in positions.iter().enumerate() {
            for (c, &(cp, cn)) in positions.iter().enumerate() {
                let conductance = y.get(r, c).ok_or_else(|| invalid("matrix port mismatch"))?;
                for (row, sign_row) in [(rp, 1.), (rn, -1.)] {
                    for (col, sign_col) in [(cp, 1.), (cn, -1.)] {
                        if let (Some(row), Some(col)) = (row, col) {
                            let signed = conductance * Complex::real(sign_row * sign_col);
                            if signed != Complex::ZERO {
                                entries.push((row, col, signed));
                            }
                        }
                    }
                }
            }
        }
        Ok(entries)
    }
    /// Augmented telegrapher constraints avoid ever inverting the B block.
    /// Input current unknowns are the physical currents multiplied by the
    /// positive wave-impedance balancing scalar. Their branch equations are
    /// Vout - A Vin - B Iin'=0, and output KCL includes -C Vin - D Iin'.
    /// This representation handles ideal shorts and lossless DC *exactly*.
    fn transfer_stamps_at(
        &self,
        hz: Real,
        unknowns: &MnaUnknowns,
        first_branch: usize,
    ) -> SpiceResult<Vec<(usize, usize, Complex)>> {
        let conductors = self.conductors();
        let (matrix, scale) = self.plan.transfer_at(hz)?;
        let inverse_scale = 1. / scale;
        if !inverse_scale.is_finite() {
            return Err(invalid("nonfinite cable current scaling"));
        }
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        inputs
            .try_reserve(conductors)
            .map_err(|_| invalid("input MNA mapping capacity"))?;
        outputs
            .try_reserve(conductors)
            .map_err(|_| invalid("output MNA mapping capacity"))?;
        for i in 0..conductors {
            let (ip, ir) = self.physical_pair(i);
            let (op, or) = self.physical_pair(conductors + i);
            inputs.push((
                unknowns.checked_node_row(ip)?,
                unknowns.checked_node_row(ir)?,
            ));
            outputs.push((
                unknowns.checked_node_row(op)?,
                unknowns.checked_node_row(or)?,
            ));
        }
        let size = conductors
            .checked_mul(conductors)
            .and_then(|v| v.checked_mul(13))
            .ok_or_else(|| invalid("augmented transmission-stamp count overflow"))?;
        let mut entries = Vec::new();
        entries
            .try_reserve(size)
            .map_err(|_| invalid("augmented stamp capacity"))?;
        let mut insert = |row: Option<usize>, col: Option<usize>, value: Complex| {
            if let (Some(row), Some(col)) = (row, col) {
                if value != Complex::ZERO {
                    entries.push((row, col, value));
                }
            }
        };
        for i in 0..conductors {
            let branch_i = first_branch
                .checked_add(i)
                .ok_or_else(|| invalid("branch row overflow"))?;
            let (input_positive, input_return) = inputs[i];
            let (output_positive, output_return) = outputs[i];
            insert(input_positive, Some(branch_i), Complex::real(inverse_scale));
            insert(input_return, Some(branch_i), Complex::real(-inverse_scale));
            insert(Some(branch_i), output_positive, Complex::real(1.));
            insert(Some(branch_i), output_return, Complex::real(-1.));
            for j in 0..conductors {
                let branch_j = first_branch
                    .checked_add(j)
                    .ok_or_else(|| invalid("branch row overflow"))?;
                let (source_positive, source_return) = inputs[j];
                let a = matrix
                    .get(i, j)
                    .ok_or_else(|| invalid("missing transmission A"))?;
                let b = matrix
                    .get(i, conductors + j)
                    .ok_or_else(|| invalid("missing transmission B"))?;
                let c = matrix
                    .get(conductors + i, j)
                    .ok_or_else(|| invalid("missing transmission C"))?;
                let d = matrix
                    .get(conductors + i, conductors + j)
                    .ok_or_else(|| invalid("missing transmission D"))?;
                // Positive/negative output terminal KCL: -C Vsource / Zscale.
                let out_current = -c * Complex::real(inverse_scale);
                insert(output_positive, source_positive, out_current);
                insert(output_positive, source_return, -out_current);
                insert(output_return, source_positive, -out_current);
                insert(output_return, source_return, out_current);
                // The same KCL equation's -D I' coupling.
                let out_branch = -d * Complex::real(inverse_scale);
                insert(output_positive, Some(branch_j), out_branch);
                insert(output_return, Some(branch_j), -out_branch);
                // One voltage constraint row per conductor: Vout-A Vin-B I'=0.
                insert(Some(branch_i), source_positive, -a);
                insert(Some(branch_i), source_return, a);
                insert(Some(branch_i), Some(branch_j), -b);
            }
        }
        if entries.iter().any(|(_, _, value)| !value.is_finite()) {
            return Err(invalid("nonfinite augmented transmission-line stamps"));
        }
        Ok(entries)
    }
}

impl Device for CoupledLine {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'p'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn small_signal_depends_on_frequency(&self) -> bool {
        true
    }

    fn branch_currents(&self) -> usize {
        self.conductors()
    }

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let conductors = self.conductors();
        for i in 0..conductors {
            let branch_i = setup.branch(i)?;
            let (input_positive, input_return) = self.physical_pair(i);
            let (output_positive, output_return) = self.physical_pair(conductors + i);
            for row in [input_positive, input_return] {
                setup.element(setup.node(row), Some(branch_i));
            }
            for col in [output_positive, output_return] {
                setup.element(Some(branch_i), setup.node(col));
            }
            for j in 0..conductors {
                let branch_j = setup.branch(j)?;
                let (source_positive, source_return) = self.physical_pair(j);
                for row in [output_positive, output_return] {
                    setup.element(setup.node(row), Some(branch_j));
                    for col in [source_positive, source_return] {
                        setup.nodes(row, col);
                    }
                }
                for col in [source_positive, source_return] {
                    setup.element(Some(branch_i), setup.node(col));
                }
                setup.element(Some(branch_i), Some(branch_j));
            }
        }
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        match context.mode {
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => {
                let first = context.branch(0)?;
                for (row, col, value) in self.transfer_stamps_at(0., context.unknowns, first)? {
                    if value.im != 0. {
                        return Err(invalid("DC line constraint requires a real MNA stamp"));
                    }
                    context.matrix.add(row, col, value.re)?;
                }
                Ok(())
            }
            AnalysisMode::Ac { .. } => Err(invalid("AC transfer belongs to assemble_small_signal")),
            AnalysisMode::Transient { .. } => Err(invalid(
                "transient coupled convolution requires an accepted-history propagator",
            )),
        }
    }

    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        let first = context
            .branch
            .ok_or_else(|| invalid("missing line branch rows"))?;
        for (row, col, value) in self.transfer_stamps_at(0., context.unknowns, first)? {
            if value.im != 0. {
                return Err(invalid("nonreal DC line bias operator"));
            }
            if value.re != 0. {
                context.system.a.add(row, col, value.re)?;
            }
        }
        Ok(())
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        let hz = context.model_context.frequency;
        let first = context
            .branch
            .ok_or_else(|| invalid("missing line branch rows"))?;
        for (row, col, value) in self.transfer_stamps_at(hz, context.unknowns, first)? {
            if value.re != 0. {
                context.system.a.add(row, col, value.re)?;
            }
            if value.im != 0. {
                context.dynamic_prescaled(row, col, value.im)?;
            }
        }
        Ok(())
    }
}
