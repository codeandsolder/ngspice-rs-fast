//! Topology-bound prepared RF line. Stamps voltage/current relations
//! directly, including exact ideal through constraints.
use std::sync::Arc;
mod realization;
pub use realization::bind_realization;

use crate::analysis::rf_network::{PreparedLine, invalid};
use crate::devices::linear::LinearContext;
use crate::devices::nport::PortTerminals;
use crate::devices::traits::{AnalysisMode, StampContext};
use crate::devices::{Device, MnaUnknowns};
use crate::maths::Vector;
use crate::primitives::{Complex, NodeId, SpiceResult};

/// A prepared reciprocal transmission line attached to physical port pairs.
/// Current unknowns point into each terminal pair.
#[derive(Debug, Clone)]
pub struct RfLine {
    name: String,
    line: Arc<PreparedLine>,
    ports: Box<[PortTerminals]>,
    nodes: Box<[NodeId]>,
}

impl RfLine {
    /// Bind a shared prepared line; near-end conductor ports first.
    /// # Errors
    /// Invalid instance name, port count or degenerate terminal pair.
    pub fn new(
        name: impl Into<String>,
        line: Arc<PreparedLine>,
        ports: Vec<PortTerminals>,
    ) -> SpiceResult<Self> {
        let name = name.into();
        if name.is_empty()
            || ports.len() != 2 * line.conductors()
            || ports.iter().any(|p| p.positive == p.negative)
        {
            return Err(invalid("invalid prepared line terminals"));
        }
        let nodes = ports
            .iter()
            .flat_map(|p| [p.positive, p.negative])
            .collect();
        Ok(Self {
            name,
            line,
            ports: ports.into_boxed_slice(),
            nodes,
        })
    }

    fn entries(
        &self,
        unknowns: &MnaUnknowns,
        branch: usize,
        hz: f64,
        temperature: f64,
        mut add: impl FnMut(usize, usize, Complex) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        let n = self.line.conductors();
        let relations = self.line.relations(hz, temperature)?;
        // KCL is independent of frequency and the modal basis.
        for (k, p) in self.ports.iter().enumerate() {
            for (node, sign) in [(p.positive, 1.), (p.negative, -1.)] {
                if let Some(row) = unknowns.node_row(node) {
                    add(row, branch + k, Complex::real(sign))?;
                }
            }
        }
        for (mode, (basis, r)) in self
            .line
            .basis()
            .chunks_exact(n)
            .zip(relations.iter().copied())
            .enumerate()
        {
            for (k, &weight) in basis.iter().enumerate() {
                for (port, sign) in [(k, 1.), (k + n, -1.)] {
                    let p = self.ports[port];
                    // One common and one differential equation per mode.
                    for (row, v, i) in [
                        (
                            branch + mode,
                            Complex::real(-weight) * r.common_voltage,
                            Complex::real(weight) * r.common_current,
                        ),
                        (
                            branch + n + mode,
                            Complex::real(sign * weight) * r.differential_voltage,
                            Complex::real(-sign * weight) * r.differential_current,
                        ),
                    ] {
                        for (node, polarity) in [(p.positive, 1.), (p.negative, -1.)] {
                            if let Some(col) = unknowns.node_row(node) {
                                add(row, col, Complex::real(polarity) * v)?;
                            }
                        }
                        add(row, branch + port, i)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn assemble(&self, ctx: &mut LinearContext<'_>, hz: f64) -> SpiceResult<()> {
        let branch = ctx
            .branch
            .ok_or_else(|| invalid("missing RF line branch binding"))?;
        self.entries(
            ctx.unknowns,
            branch,
            hz,
            ctx.model_context.temperature,
            |r, c, v| {
                ctx.system.a.add(r, c, v.re)?;
                if v.im != 0. {
                    ctx.dynamic_prescaled(r, c, v.im)?;
                }
                Ok(())
            },
        )
    }
}

impl Device for RfLine {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'n'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.nodes
    }
    fn branch_currents(&self) -> usize {
        self.ports.len()
    }
    fn small_signal_depends_on_frequency(&self) -> bool {
        true
    }

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        for (k, p) in self.ports.iter().enumerate() {
            let current = setup.branch(k)?;
            for node in [p.positive, p.negative] {
                setup.element(setup.node(node), Some(current));
            }
        }
        for k in 0..self.ports.len() {
            let row = Some(setup.branch(k)?);
            for (j, p) in self.ports.iter().enumerate() {
                setup.element(row, setup.node(p.positive));
                setup.element(row, setup.node(p.negative));
                setup.element(row, Some(setup.branch(j)?));
            }
        }
        Ok(true)
    }

    fn stamp(&self, ctx: &mut StampContext<'_>) -> SpiceResult<()> {
        match ctx.mode {
            AnalysisMode::OperatingPoint
            | AnalysisMode::DcSweep
            | AnalysisMode::Instantaneous { .. } => {
                let branch = ctx.branch(0)?;
                self.entries(ctx.unknowns, branch, 0., ctx.temperature, |r, c, v| {
                    ctx.matrix.add(r, c, v.re)
                })
            }
            AnalysisMode::Ac { .. } => Err(invalid("use complex small-signal assembly")),
            AnalysisMode::Transient { .. } => Err(invalid(
                "distributed line requires a causal transient realization",
            )),
        }
    }

    fn assemble_linear(&self, ctx: &mut LinearContext<'_>) -> SpiceResult<()> {
        self.assemble(ctx, 0.)
    }

    fn assemble_small_signal(
        &self,
        ctx: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble(ctx, ctx.model_context.frequency)
    }
}
