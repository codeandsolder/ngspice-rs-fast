//! Analysis-bound physical conductance, using the native nodal kernels.
use crate::devices::passive_bias::PassiveInitialization;
use crate::devices::{
    AnalysisMode, Device, LinearContext, ModelContext, ResistorNoise, StampContext,
    distortion::{DeviceDistortion, DistortionContext},
    noise::{DeviceNoise, NoiseContext, NoiseSource},
    rlc::{assemble_conductance, resistor_noise, stamp_conductance},
    setup::MatrixSetup,
};
use crate::maths::Vector;
use crate::primitives::{AnalysisKind, NodeId, Real, SpiceError, SpiceResult};
use std::sync::Arc;

/// Affine physical conductance in the configured junction minimum conductance.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct ConductanceLaw {
    /// Constant physical conductance in siemens.
    pub constant: Real,
    /// Coefficient multiplying the analysis's configured gmin.
    pub gmin_scale: Real,
}

impl ConductanceLaw {
    fn scaled(self, m: Real) -> SpiceResult<Self> {
        let scaled = Self {
            constant: self.constant * m,
            gmin_scale: self.gmin_scale * m,
        };
        if [
            self.constant,
            self.gmin_scale,
            scaled.constant,
            scaled.gmin_scale,
        ]
        .into_iter()
        .any(|g| !g.is_finite() || g < 0.)
            || (self.constant != 0. && scaled.constant == 0.)
            || (self.gmin_scale != 0. && scaled.gmin_scale == 0.)
        {
            return Err(SpiceError::circuit("invalid scaled damping conductance"));
        }
        Ok(scaled)
    }

    fn evaluate(self, gmin: Real) -> SpiceResult<Real> {
        let conductance = self.gmin_scale.mul_add(gmin, self.constant);
        if !conductance.is_finite() || conductance < 0. {
            return Err(SpiceError::circuit(
                "invalid contextual damping conductance",
            ));
        }
        Ok(conductance)
    }
}

/// Physical damping policies for equilibrium, frequency and time-domain loads.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct AnalysisDamping {
    /// Standalone DC and OP conductance.
    pub steady: ConductanceLaw,
    /// Small-signal conductance, including the corresponding bias.
    pub small_signal: ConductanceLaw,
    /// Transient conductance, including startup.
    pub transient: ConductanceLaw,
    /// Whether this is a physical thermal generator rather than noiseless
    /// numerical damping.
    pub noisy: bool,
}

impl AnalysisDamping {
    pub(super) fn scaled(self, m: Real) -> SpiceResult<Self> {
        Ok(Self {
            steady: self.steady.scaled(m)?,
            small_signal: self.small_signal.scaled(m)?,
            transient: self.transient.scaled(m)?,
            noisy: self.noisy,
        })
    }

    const fn law(self, analysis: Option<AnalysisKind>) -> ConductanceLaw {
        match analysis {
            Some(
                AnalysisKind::Transient | AnalysisKind::PeriodicSteadyState | AnalysisKind::Fourier,
            ) => self.transient,
            Some(
                AnalysisKind::Ac
                | AnalysisKind::Noise
                | AnalysisKind::Distortion
                | AnalysisKind::PoleZero
                | AnalysisKind::SParameter
                | AnalysisKind::HarmonicBalance,
            ) => self.small_signal,
            _ => self.steady,
        }
    }

    const fn stamp_law(self, mode: AnalysisMode, analysis: Option<AnalysisKind>) -> ConductanceLaw {
        match mode {
            AnalysisMode::Instantaneous { .. } | AnalysisMode::Transient { .. } => self.transient,
            AnalysisMode::Ac { .. } => self.small_signal,
            _ => self.law(analysis),
        }
    }
}

#[derive(Debug)]
pub(super) struct DampingShunt {
    pub(super) name: String,
    pub(super) terminals: [NodeId; 2],
    pub(super) damping: AnalysisDamping,
    pub(super) temperature: Option<Real>,
    pub(super) excluded_bias: Option<Arc<PassiveInitialization>>,
}

impl DampingShunt {
    pub(super) fn resistive(
        name: String,
        terminals: [NodeId; 2],
        resistance: Real,
        temperature: Option<Real>,
        excluded_bias: Option<Arc<PassiveInitialization>>,
    ) -> Self {
        let law = ConductanceLaw {
            constant: 1. / resistance,
            gmin_scale: 0.,
        };
        Self {
            name,
            terminals,
            temperature,
            excluded_bias,
            damping: AnalysisDamping {
                steady: law,
                small_signal: law,
                transient: law,
                noisy: true,
            },
        }
    }
}

impl Device for DampingShunt {
    fn name(&self) -> &str {
        &self.name
    }

    fn designator(&self) -> char {
        'r'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }

    fn requires_analysis_context(&self) -> bool {
        true
    }

    fn setup_matrix(&self, setup: &mut MatrixSetup<'_>) -> SpiceResult<bool> {
        setup.conductance(self.terminals[0], self.terminals[1]);
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let g = if self
            .excluded_bias
            .as_ref()
            .is_some_and(|policy| policy.active(context).is_some())
        {
            0.
        } else {
            self.damping
                .stamp_law(context.mode, context.analysis_kind)
                .evaluate(context.configured_gmin)?
        };
        stamp_conductance(self.terminals, g, context)
    }

    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        let g = self
            .damping
            .law(context.model_context.analysis_kind)
            .evaluate(context.model_context.configured_gmin)?;
        assemble_conductance(self.terminals, g, context)
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        let g = self
            .damping
            .small_signal
            .evaluate(context.model_context.configured_gmin)?;
        assemble_conductance(self.terminals, g, context)
    }

    fn assemble_pole_zero(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble_small_signal(context, bias)
    }

    fn observation_parameter(
        &self,
        keyword: &str,
        context: &ModelContext,
    ) -> SpiceResult<Option<Real>> {
        match keyword {
            "g" | "conductance" => self
                .damping
                .law(context.analysis_kind)
                .evaluate(context.configured_gmin)
                .map(Some),
            _ => Ok(None),
        }
    }

    fn distortion(&self, _context: &DistortionContext<'_>) -> SpiceResult<DeviceDistortion> {
        Ok(DeviceDistortion::Linear)
    }

    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        let g = self
            .damping
            .small_signal
            .evaluate(context.model_context.configured_gmin)?;
        Ok(resistor_noise(
            self.terminals,
            g,
            &ResistorNoise {
                noisy: self.damping.noisy,
                temperature: self.temperature,
                ..Default::default()
            },
            None,
            context,
            None,
            sources,
        ))
    }
}
