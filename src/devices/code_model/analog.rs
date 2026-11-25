//! Native linear analog code models using the ordinary MNA and integration
//! machinery. The voltage interface is differential on both ports.
//!
//! The gain and integrator follow the unsaturated equations of upstream
//! XSPICE `gain` and `int`; the low-pass is the one-pole `s_xfer` equation.

use super::super::{
    Device, LinearContext, SourceKind, StampContext, Waveform,
    linear::{LinearSource, SourceRows},
    noise::{DeviceNoise, NoiseContext, NoiseSource},
    setup::MatrixSetup,
};
use crate::maths::{SparseMatrix, integrator::ChargeHistory};
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};
use std::sync::Arc;

/// Dynamical equation for a native analog voltage code model.
#[derive(Clone, Copy, Debug)]
pub enum Dynamics {
    /// Algebraic gain: output = gain * input + offset.
    Gain,
    /// First-order low-pass: tau * output' + output = gain * input + offset.
    LowPass {
        /// Positive finite time constant, in seconds.
        time_constant: Real,
    },
    /// Unsaturated integrator: output' = gain * input + offset.
    Integrator {
        /// Output at the DC / initial transient point.
        initial_output: Real,
    },
}

impl Dynamics {
    const fn storage_scale(self) -> Real {
        match self {
            Self::Gain => 0.,
            Self::LowPass { time_constant } => time_constant,
            Self::Integrator { .. } => 1.,
        }
    }

    const fn static_coefficient(self) -> Real {
        match self {
            Self::Gain | Self::LowPass { .. } => 1.,
            Self::Integrator { .. } => 0.,
        }
    }
}

/// Compiled, typed analog voltage model.
///
/// The model's two differential ports
/// belong to the analog MNA domain; unlike digital models, its state slots
/// belong to the companion solver's accepted/trial history.
#[derive(Debug)]
pub struct AnalogCodeModel {
    name: String,
    // Output positive, output negative, input positive, input negative.
    terminals: [NodeId; 4],
    dynamics: Dynamics,
    gain: Real,
    offset: Real,
}

impl AnalogCodeModel {
    /// One differential voltage input and one differential voltage output.
    #[must_use = "the returned schema declares the analog input and output voltage ports"]
    pub const fn schema() -> super::ModelSchema {
        super::ModelSchema {
            analog_voltage: super::PortCount {
                inputs: 1,
                outputs: 1,
            },
            digital: super::PortCount {
                inputs: 0,
                outputs: 0,
            },
        }
    }

    /// Construct a gain, low-pass, or unsaturated integrator kernel.
    ///
    /// `offset` is the additive right-hand-side term, in volts for gain and
    /// low-pass, or volts per second for an integrator. Input offsets can be
    /// compiled into it without adding an independent signal.
    ///
    /// # Errors
    /// Shorted output, nonfinite coefficients, or invalid time constant.
    pub fn new(
        name: impl Into<String>,
        output: [NodeId; 2],
        input: [NodeId; 2],
        dynamics: Dynamics,
        gain: Real,
        offset: Real,
    ) -> SpiceResult<Self> {
        let valid = match dynamics {
            Dynamics::Gain => true,
            Dynamics::LowPass { time_constant } => time_constant.is_finite() && time_constant > 0.,
            Dynamics::Integrator { initial_output } => initial_output.is_finite(),
        };
        if !valid || !gain.is_finite() || !offset.is_finite() || output[0] == output[1] {
            return Err(SpiceError::circuit("invalid analog code-model equation"));
        }
        Ok(Self {
            name: name.into(),
            terminals: [output[0], output[1], input[0], input[1]],
            dynamics,
            gain,
            offset,
        })
    }

    pub(crate) const fn initial_constraint(&self) -> Option<Real> {
        match self.dynamics {
            Dynamics::Integrator { initial_output } => Some(initial_output),
            _ => None,
        }
    }

    const fn dynamic(&self) -> bool {
        !matches!(self.dynamics, Dynamics::Gain)
    }

    fn output_voltage(&self, context: &StampContext<'_>) -> Real {
        context.node_voltage(self.terminals[0]) - context.node_voltage(self.terminals[1])
    }

    /// Stamp the output branch's KCL and differential equation separately.
    /// The branch current is independent of the differential equation's
    /// coefficient, including a zero static coefficient for an integrator.
    fn stamp_equation(
        &self,
        matrix: &mut SparseMatrix,
        unknowns: &super::super::MnaUnknowns,
        branch: usize,
        output: Real,
        input: Real,
    ) -> SpiceResult<()> {
        let [op, on, ip, inn] = self.terminals;
        for (node, sign) in [(op, 1.), (on, -1.)] {
            if let Some(row) = unknowns.node_row(node) {
                matrix.add(row, branch, sign)?;
                matrix.add(branch, row, sign * output)?;
            }
        }
        for (node, sign) in [(ip, 1.), (inn, -1.)] {
            if let Some(row) = unknowns.node_row(node) {
                matrix.add(branch, row, sign * input)?;
            }
        }
        Ok(())
    }

    fn transient_equation(&self, context: &mut StampContext<'_>) -> SpiceResult<(Real, Real)> {
        let coefficients = context.integration.ok_or_else(|| {
            SpiceError::circuit("analog code-model transient requires companion integration")
        })?;
        if context.states.uic() && context.states.initial_transient() {
            return Err(SpiceError::circuit(
                "analog code-model storage requires operating-point initialization",
            ));
        }
        let y = self.output_voltage(context);
        let mut history = ChargeHistory::new(y);
        for age in 1..=coefficients.charge_history_len() {
            history.push(context.states.accepted(age, 0).ok_or_else(|| {
                SpiceError::circuit("analog code-model output history is incomplete")
            })?)?;
        }
        let previous = context.states.accepted(1, 1);
        let companion = coefficients.integrate(&history, previous, 1.)?;
        context.states.set(0, y)?;
        context.states.set(1, companion.derivative)?;
        let scale = self.dynamics.storage_scale();
        Ok((
            scale.mul_add(companion.conductance, self.dynamics.static_coefficient()),
            (-scale).mul_add(companion.current, self.offset),
        ))
    }
}

impl Device for AnalogCodeModel {
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
        1
    }

    fn findable_branch(&self) -> Option<usize> {
        Some(0)
    }

    fn state_count(&self) -> usize {
        if self.dynamic() { 2 } else { 0 }
    }

    fn truncation_slot(&self) -> Option<usize> {
        self.dynamic().then_some(0)
    }

    fn setup_matrix(&self, setup: &mut MatrixSetup<'_>) -> SpiceResult<bool> {
        let branch = setup.branch(0)?;
        let [op, on, ip, inn] = self.terminals;
        setup.voltage_branch(op, on, branch);
        setup.element(Some(branch), setup.node(ip));
        setup.element(Some(branch), setup.node(inn));
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        if context.mode.is_ac() {
            return Err(SpiceError::circuit(
                "use complex analog code-model assembly for AC",
            ));
        }
        let (output, input, rhs) = if context.mode.is_dc() {
            if self.dynamic() && !context.states.is_empty() {
                context.states.set(0, self.output_voltage(context))?;
                context.states.set(1, 0.)?;
            }
            self.initial_constraint()
                .map_or((1., -self.gain, self.offset), |initial| (1., 0., initial))
        } else if self.dynamic() {
            let (output, rhs) = self.transient_equation(context)?;
            (output, -self.gain, rhs)
        } else {
            (1., -self.gain, self.offset)
        };
        let branch = context.branch(0)?;
        self.stamp_equation(context.matrix, context.unknowns, branch, output, input)?;
        context.rhs.add_to(branch, rhs)
    }

    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        let branch = context
            .branch
            .ok_or_else(|| SpiceError::circuit("missing analog code-model branch row"))?;
        self.stamp_equation(
            &mut context.system.a,
            context.unknowns,
            branch,
            self.dynamics.static_coefficient(),
            -self.gain,
        )?;
        let scale = self.dynamics.storage_scale();
        if self.dynamic() {
            for (node, sign) in [(self.terminals[0], 1.), (self.terminals[1], -1.)] {
                if let Some(row) = context.unknowns.node_row(node) {
                    context.system.e.add(branch, row, sign * scale)?;
                }
            }
        }
        if self.initial_constraint().is_some() {
            context.system.has_initial_conditions = true;
        }
        if self.offset != 0. {
            let mut rows = SourceRows::new();
            rows.push(branch, 1.)?;
            context.system.sources.push(LinearSource {
                name: Arc::new(self.name.clone()),
                kind: SourceKind::Voltage,
                rows,
                dc: self.offset,
                ac: Complex::ZERO,
                waveform: Arc::new(Waveform::Constant(self.offset)),
            });
        }
        Ok(())
    }

    fn noise(
        &self,
        _context: &NoiseContext<'_>,
        _sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        // These ideal deterministic equations have no intrinsic noise.
        Ok(DeviceNoise::Noiseless)
    }
}
