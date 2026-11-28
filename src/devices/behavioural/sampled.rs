//! Accepted-point sampling of a physical branch coordinate.
//!
//! The private signal equation has no electrical incidence. Its stored value
//! is published from the solved control coordinate, never from a Newton guess.

use crate::devices::{
    Device, LinearContext,
    setup::MatrixSetup,
    traits::{AnalysisMode, ControlReference, ConvergenceContext, MnaUnknowns, StampContext},
};
use crate::maths::Vector;
use crate::netlist::ast::{DeviceInstance, ParameterKind};
use crate::primitives::{NodeAppend, NodeId, NodeKind, Real, SpiceError, SpiceResult};

pub(super) const MARKER: &str = "__dynamic_sampled_current";

pub(in crate::devices) fn is_generated(instance: &DeviceInstance) -> bool {
    instance.designator == 'b'
        && matches!(instance.parameters.as_slice(), [parameter]
            if parameter.name == MARKER && parameter.kind == ParameterKind::Instance)
}

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

    fn held(sample: Option<Real>) -> SpiceResult<Real> {
        sample.ok_or_else(|| {
            SpiceError::circuit("sampled current requires an accepted initial point")
        })
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
    fn state_count(&self) -> usize {
        1
    }
    fn accepted_control_samples(&self) -> &[(usize, usize)] {
        &[(0, 0)]
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
        let current = Self::current(context.solution, context.controls)?;
        // Satisfy the trial-state contract. Acceptance overwrites this slot
        // with the final solved current, which may differ from this load.
        context.states.set(0, current)?;
        context.matrix.add(row, row, 1.)?;
        match context.mode {
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => {
                context.matrix.add(row, control, -1.)
            }
            AnalysisMode::Transient { .. } => context
                .rhs
                .add_to(row, Self::held(context.states.accepted(1, 0))?),
            _ => Err(SpiceError::circuit(
                "sampled current requires bias, companion transient, or small-signal assembly",
            )),
        }
    }

    fn convergence_test(&self, context: &ConvergenceContext<'_>) -> SpiceResult<bool> {
        let actual = context
            .solution
            .get(self.row(context.unknowns)?)
            .ok_or_else(|| SpiceError::circuit("missing sampled-current signal value"))?;
        let expected = if context.integration.is_some() {
            Self::held(context.accepted(1, 0))?
        } else {
            Self::current(context.solution, context.controls)?
        };
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
        // no infinitesimal signal through an accepted-point observation.
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
    if !is_generated(instance) || instance.model.is_some() {
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
