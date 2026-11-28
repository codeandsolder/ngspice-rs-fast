//! Independent terminal bias equations. They own no charge or flux history.

use std::sync::Arc;

use crate::devices::{
    BranchQuantity, Device, LinearContext, StampContext,
    distortion::{DeviceDistortion, DistortionContext},
    passive_bias::PassiveInitialization,
    setup::MatrixSetup,
};
use crate::primitives::{AnalysisKind, NodeId, SpiceResult};

#[derive(Debug)]
pub(super) struct TerminalBias {
    pub(super) name: String,
    pub(super) terminals: [NodeId; 2],
    pub(super) policy: Arc<PassiveInitialization>,
}

impl Device for TerminalBias {
    fn name(&self) -> &str {
        &self.name
    }

    fn designator(&self) -> char {
        // This is a physical bias equation, not an independent excitation
        // or a reactive state owner.
        'b'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    fn has_passive_bias(&self) -> bool {
        true
    }

    fn has_active_passive_bias(&self, analysis: AnalysisKind) -> bool {
        self.policy.constrains(analysis)
    }

    fn branch_currents(&self) -> usize {
        usize::from(self.policy.needs_branch())
    }

    fn branch_quantity(&self, _branch: usize) -> BranchQuantity {
        BranchQuantity::BiasCurrent
    }

    fn setup_matrix(&self, setup: &mut MatrixSetup<'_>) -> SpiceResult<bool> {
        let [p, n] = self.terminals;
        setup.conductance(p, n);
        if self.policy.needs_branch() {
            setup.inductor(p, n, setup.branch(0)?);
        }
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        self.policy.stamp_terminal(self.terminals, context)
    }

    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        self.policy.release_capacitor(context)
    }

    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &crate::maths::Vector,
    ) -> SpiceResult<()> {
        self.assemble_linear(context)
    }

    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        Ok(crate::devices::noise::DeviceNoise::Noiseless)
    }

    fn distortion(&self, _context: &DistortionContext<'_>) -> SpiceResult<DeviceDistortion> {
        Ok(DeviceDistortion::Linear)
    }
}
