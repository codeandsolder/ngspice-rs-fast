//! Multiconductor distributed transmission lines with coupled RLGC physics.
//!
//! The line's physical node connection is N input conductors and one input
//! return, then N output conductors and one output return. Currents at both
//! ends are defined entering the line. General noncommuting matrices are
//! handled by one 2N-dimensional telegrapher state-space propagator.
pub mod plan;

use std::sync::Arc;

use crate::devices::linear::LinearContext;
use crate::devices::traits::{AnalysisMode, StampContext};
use crate::devices::{Device, MnaUnknowns};
use crate::maths::{Vector, dense_complex::DenseComplex};
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};
use plan::CoupledLinePlan;

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
impl CoupledLine {
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
            || input.iter().any(|&node| node == input_return)
            || output.iter().any(|&node| node == output_return)
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

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let ports = self.conductors() * 2;
        for output in 0..ports {
            for input in 0..ports {
                let (rp, rn) = self.physical_pair(output);
                let (cp, cn) = self.physical_pair(input);
                for row in [rp, rn] {
                    for col in [cp, cn] {
                        setup.nodes(row, col);
                    }
                }
            }
        }
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        match context.mode {
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => {
                for (row, col, value) in self.nodal_stamps_at(0., context.unknowns)? {
                    if value.im != 0. {
                        return Err(invalid("DC line admits no complex-valued real MNA stamp"));
                    }
                    context.matrix.add(row, col, value.re)?;
                }
                Ok(())
            }
            AnalysisMode::Ac { .. } => Err(invalid(
                "complex frequency load belongs to assemble_small_signal",
            )),
            AnalysisMode::Transient { .. } => Err(invalid(
                "transient coupled convolution requires an accepted-history propagator, not an AC matrix",
            )),
        }
    }

    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        for (row, col, value) in self.nodal_stamps_at(0., context.unknowns)? {
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
        for (row, col, value) in self.nodal_stamps_at(hz, context.unknowns)? {
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
