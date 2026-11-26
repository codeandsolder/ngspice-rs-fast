//! Native physical binding for the prepared timed-switch schema.
use super::source::{SourceModel, numeric_scalar};
use super::timed_switch::{Material, Parameters, State};
use crate::devices::traits::{ConvergenceContext, DeviceTransientEvents, TransientEventContext};
use crate::devices::{AnalysisMode, Device, LinearContext, StampContext};
use crate::maths::Vector;
use crate::netlist::ast::ModelCard;
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

pub(super) fn compile(card: &ModelCard) -> SpiceResult<Material> {
    SourceModel::TimedSwitch.validate_parameters(card)?;
    let mut p = Parameters::default();
    for parameter in &card.parameters {
        let value = numeric_scalar(parameter)?;
        match parameter.name.as_str() {
            "roff" => p.resistance[0] = value,
            "ron" => p.resistance[1] = value,
            "toff" => p.duration[0] = value,
            "ton" => p.duration[1] = value,
            "td_off" => p.delay[0] = value,
            "td_on" => p.delay[1] = value,
            "vt" => p.center = value,
            "vh" => p.hysteresis = value,
            "timetol" => p.tolerance = value,
            "ic" if matches!(value, 0. | 1.) => p.initial = matches!(value, 1.),
            _ => {
                return Err(SpiceError::parse(
                    parameter.location.clone(),
                    "invalid timed switch parameter",
                ));
            }
        }
    }
    Material::compile(p)
}

#[derive(Debug)]
pub(super) struct TimedSwitch {
    pub(super) name: String,
    pub(super) model_name: String,
    /// Source port order: unloaded control, conducting branch.
    pub(super) terminals: [NodeId; 4],
    pub(super) material: Material,
}

impl TimedSwitch {
    const fn branch(&self) -> [NodeId; 2] {
        [self.terminals[2], self.terminals[3]]
    }

    fn control(&self, voltage: impl Fn(NodeId) -> Real) -> Real {
        voltage(self.terminals[0]) - voltage(self.terminals[1])
    }

    fn state(&self, time: Real, control: Real, accepted: Option<&[Real]>) -> SpiceResult<State> {
        if time == 0. {
            self.material.equilibrium(control)
        } else {
            let accepted = accepted
                .ok_or_else(|| SpiceError::circuit("timed switch has no accepted history"))?;
            self.material
                .advance(State::from_values(accepted)?, time, control)
        }
    }
}

impl Device for TimedSwitch {
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'a'
    }
    fn model_name(&self) -> Option<&str> {
        Some(&self.model_name)
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn has_discrete_state(&self) -> bool {
        true
    }
    fn depends_on_time(&self) -> bool {
        true
    }
    fn state_count(&self) -> usize {
        State::SLOTS
    }
    fn small_signal_uses_trial_state(&self) -> bool {
        true
    }

    fn transient_events(&self) -> Option<&dyn DeviceTransientEvents> {
        Some(self)
    }

    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        let [p, n] = self.branch();
        setup.conductance(p, n);
        Ok(true)
    }

    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let time = match context.mode {
            AnalysisMode::OperatingPoint | AnalysisMode::DcSweep => 0.,
            AnalysisMode::Transient { time, .. } | AnalysisMode::Instantaneous { time } => time,
            AnalysisMode::Ac { .. } => {
                return Err(SpiceError::circuit(
                    "timed switch AC requires a frozen operating point",
                ));
            }
        };
        let control = self.control(|node| context.node_voltage(node));
        let mut accepted = [0.; State::SLOTS];
        let history = if time > 0. {
            for (slot, value) in accepted.iter_mut().enumerate() {
                *value = context
                    .states
                    .accepted(1, slot)
                    .ok_or_else(|| SpiceError::circuit("missing timed switch history"))?;
            }
            Some(accepted.as_slice())
        } else {
            None
        };
        let state = self.state(time, control, history)?;
        let conductance = state.at_conductance(self.material)?;
        if !context.states.is_empty() {
            for (slot, value) in state.values().into_iter().enumerate() {
                context.states.set(slot, value)?;
            }
        }
        crate::devices::linear::nodal_stamp(
            context.matrix,
            context.unknowns,
            self.branch(),
            conductance,
        )
    }

    fn convergence_test(&self, context: &ConvergenceContext<'_>) -> SpiceResult<bool> {
        let time = context.time;
        let control = self.control(|node| context.voltage(node));
        let accepted = context
            .history
            .accepted(1)
            .and_then(|values| values.get(context.state_range.clone()));
        let actual = self.state(time, control, accepted)?;
        let stamped = State::from_values(context.states)?;
        if actual.discrete() != stamped.discrete() {
            return Ok(true);
        }
        let [p, n] = self.branch();
        let voltage = context.voltage(p) - context.voltage(n);
        let expected = actual.at_conductance(self.material)? * voltage;
        let observed = stamped.at_conductance(self.material)? * voltage;
        if !expected.is_finite() || !observed.is_finite() {
            return Err(SpiceError::circuit("nonfinite timed switch branch current"));
        }
        let tolerance = context
            .reltol
            .mul_add(expected.abs().max(observed.abs()), context.abstol);
        Ok((expected - observed).abs() > tolerance)
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let state = if let Some(values) = context.states {
            State::from_values(values)?
        } else {
            let control = self.control(|node| {
                context
                    .unknowns
                    .node_row(node)
                    .and_then(|row| bias.get(row))
                    .unwrap_or(0.)
            });
            self.material.equilibrium(control)?
        };
        context.nodal(self.branch(), state.at_conductance(self.material)?, false)
    }

    fn assemble_nonlinear_charge(
        &self,
        _context: &mut crate::devices::traits::ChargeContext<'_>,
    ) -> SpiceResult<()> {
        // This memory is logical history, not electric charge or flux.
        Ok(())
    }
}

impl DeviceTransientEvents for TimedSwitch {
    fn next_boundary(&self, accepted: &[Real]) -> SpiceResult<Option<Real>> {
        State::from_values(accepted)?.next_boundary(self.material)
    }

    fn refine_trial(
        &self,
        context: &TransientEventContext<'_>,
        trial: &mut [Real],
    ) -> SpiceResult<Option<Real>> {
        let old = State::from_values(context.accepted)?;
        let control = self.control(|node| context.voltage(node));
        let candidate = self.material.advance(old, context.time, control)?;
        // The last Newton stamp may be one iterate behind. Publish the exact
        // converged receiver sample, once its stamp has passed convergence.
        if trial.len() != State::SLOTS {
            return Err(SpiceError::circuit("invalid timed switch trial dimension"));
        }
        trial.copy_from_slice(&candidate.values());
        Ok(self.material.retry_at(old, candidate))
    }
}
