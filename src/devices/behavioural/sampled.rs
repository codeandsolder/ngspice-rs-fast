//! Phase-selected iteration sampling of a physical branch coordinate.
//!
//! The private signal has no electrical incidence or independent history.
//! Transient loads sample the current iterate with a frozen control tangent.

use crate::devices::{
    Device, LinearContext,
    setup::MatrixSetup,
    traits::{AnalysisMode, ControlReference, ConvergenceContext, MnaUnknowns, StampContext},
};
use crate::maths::Vector;
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{NodeAppend, NodeId, NodeKind, Real, SpiceError, SpiceResult};

#[derive(Debug)]
struct SampledCurrent {
    name: String,
    signal: [NodeId; 1],
    control: [ControlReference; 1],
}

impl SampledCurrent {
    fn row(&self, unknowns: &MnaUnknowns) -> SpiceResult<usize> {
        unknowns
            .node_row(self.signal[0])
            .ok_or_else(|| SpiceError::circuit("missing sampled-current signal row"))
    }

    fn control_row(controls: &[usize]) -> SpiceResult<usize> {
        controls
            .first()
            .copied()
            .ok_or_else(|| SpiceError::circuit("unbound sampled current"))
    }

    fn current(solution: &Vector, controls: &[usize]) -> SpiceResult<Real> {
        solution
            .get(Self::control_row(controls)?)
            .ok_or_else(|| SpiceError::circuit("missing sampled-current control value"))
    }
}

impl Device for SampledCurrent {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'b'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.signal
    }
    fn external_terminals(&self) -> &[NodeId] {
        &[]
    }
    fn constitutive_nodes(&self) -> &[NodeId] {
        &self.signal
    }
    fn controlling_sources(&self) -> &[ControlReference] {
        &self.control
    }
    fn control_branch(&self, target: &dyn Device) -> Option<usize> {
        target.expression_current_branch()
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn limits_voltage_steps(&self) -> bool {
        false
    }
    fn setup_matrix(&self, setup: &mut MatrixSetup<'_>) -> SpiceResult<bool> {
        let row = setup.node(self.signal[0]);
        setup.element(row, row);
        setup.element(row, Some(Self::control_row(setup.controls)?));
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let row = self.row(context.unknowns)?;
        let control = Self::control_row(context.controls)?;
        context.matrix.add(row, row, 1.)?;
        match context.mode {
            AnalysisMode::OperatingPoint
            | AnalysisMode::DcSweep
            | AnalysisMode::Instantaneous { .. } => {
                // The converged static/physical law is exact equality. Its
                // Jacobian is not the source-selected small-signal tangent.
                context.matrix.add(row, control, -1.)
            }
            AnalysisMode::Transient { .. } => {
                // Picard sampling: re-evaluate from each Newton load's
                // input iterate, never persist a rejected trial observation.
                context
                    .rhs
                    .add_to(row, Self::current(context.solution, context.controls)?)
            }
            AnalysisMode::Ac { .. } => Err(SpiceError::circuit(
                "sampled current uses small-signal assembly for AC",
            )),
        }
    }

    fn assemble_nonlinear_charge(
        &self,
        _context: &mut crate::devices::traits::ChargeContext<'_>,
    ) -> SpiceResult<()> {
        Ok(())
    }

    fn assemble_nonlinear_taylor(
        &self,
        context: &mut crate::devices::physical_dae::TaylorContext<'_>,
    ) -> SpiceResult<()> {
        let row = Some(self.row(context.unknowns)?);
        let control = Some(Self::control_row(context.controls)?);
        context.linear_static(row, row, 1.)?;
        context.linear_static(row, control, -1.)
    }

    fn convergence_test(&self, context: &ConvergenceContext<'_>) -> SpiceResult<bool> {
        let actual = context
            .solution
            .get(self.row(context.unknowns)?)
            .ok_or_else(|| SpiceError::circuit("missing sampled-current signal value"))?;
        let expected = Self::current(context.solution, context.controls)?;
        Ok((actual - expected).abs()
            > context
                .reltol
                .mul_add(actual.abs().max(expected.abs()), context.abstol))
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        let row = self.row(context.unknowns)?;
        // The DC value remains in the expression's bias point, but there is
        // no infinitesimal signal through this sampled observation.
        context.system.a.add(row, row, 1.)
    }

    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }

    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        Ok(crate::devices::noise::DeviceNoise::Noiseless)
    }
}

pub(in crate::devices) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
) -> SpiceResult<Box<dyn Device>> {
    let [parameter] = instance.parameters.as_slice() else {
        return Err(SpiceError::circuit("invalid sampled-current operator"));
    };
    if instance.model.is_some() {
        return Err(SpiceError::circuit("invalid sampled-current operator"));
    }
    let signal = nodes.intern_array::<1>(&instance.nodes)?;
    nodes.set_kind(signal[0], NodeKind::Internal)?;
    Ok(Box::new(SampledCurrent {
        name: instance.name.clone(),
        signal,
        control: [ControlReference {
            name: parameter.value.clone(),
            location: Some(parameter.location.clone()),
        }],
    }))
}
