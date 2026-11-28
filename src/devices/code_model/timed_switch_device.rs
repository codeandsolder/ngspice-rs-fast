//! Native physical binding for the prepared timed-switch schema.
use super::source::{SourceModel, numeric_scalar};
use super::switch_control::{SwitchControl, read_pair};
use super::timed_switch::{Material, Parameters, RisingBoundary, State, Transition};
use crate::devices::noise::{DeviceNoise, NoiseContext, NoiseFamily, NoiseKind, NoiseSource};
use crate::devices::traits::{
    ControlReference, ConvergenceContext, DeviceTransientEvents, MnaUnknowns, TransientEventContext,
};
use crate::devices::{AnalysisMode, Device, LinearContext, StampContext};
use crate::maths::Vector;
use crate::netlist::ast::ModelCard;
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

pub(super) fn compile(card: &ModelCard) -> SpiceResult<Material> {
    SourceModel::TimedSwitch.validate_parameters(card)?;
    let mut p = Parameters::default();
    let Transition::Smooth(mut duration) = p.transition else {
        return Err(SpiceError::circuit(
            "default timed switch transition must be smooth",
        ));
    };
    let mut explicit_duration = false;
    let mut instantaneous = false;
    for parameter in &card.parameters {
        let value = numeric_scalar(parameter)?;
        match parameter.name.as_str() {
            "roff" => p.resistance[0] = value,
            "ron" => p.resistance[1] = value,
            "toff" => {
                duration[0] = value;
                explicit_duration = true;
            }
            "ton" => {
                duration[1] = value;
                explicit_duration = true;
            }
            "instantaneous" => instantaneous = matches!(value, 1.),
            "inclusive_rise" => {
                p.rising_boundary = if matches!(value, 1.) {
                    RisingBoundary::Inclusive
                } else {
                    RisingBoundary::Strict
                }
            }
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
    p.transition = if instantaneous {
        if explicit_duration {
            return Err(SpiceError::circuit(
                "instantaneous timed switch must not specify TON or TOFF",
            ));
        }
        Transition::InstantInertial
    } else {
        Transition::Smooth(duration)
    };
    Material::compile(p)
}

#[derive(Debug)]
pub(super) struct TimedSwitch {
    pub(super) name: String,
    pub(super) model_name: String,
    /// Source port order: control, conducting branch.
    pub(super) terminals: [NodeId; 4],
    /// Differential voltage, local ammeter, or existing named branch.
    pub(super) control: SwitchControl,
    pub(super) material: Material,
}

impl TimedSwitch {
    const fn branch(&self) -> [NodeId; 2] {
        [self.terminals[2], self.terminals[3]]
    }

    const fn sense_terminals(&self) -> [NodeId; 2] {
        [self.terminals[0], self.terminals[1]]
    }

    fn control(
        &self,
        unknowns: &MnaUnknowns,
        branch: Option<usize>,
        controls: &[usize],
        solution: &Vector,
    ) -> SpiceResult<Real> {
        let rows = self.control.rows(
            self.sense_terminals().map(|node| unknowns.node_row(node)),
            branch,
            controls,
        )?;
        read_pair(rows.coordinates, solution)
    }

    fn frozen_state(
        &self,
        states: Option<&[Real]>,
        control: impl FnOnce() -> SpiceResult<Real>,
    ) -> SpiceResult<State> {
        if let Some(values) = states {
            State::from_values(values)
        } else {
            self.material.equilibrium(control()?)
        }
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
        if self.control.is_named() {
            &self.terminals[2..]
        } else {
            &self.terminals
        }
    }
    fn controlling_sources(&self) -> &[ControlReference] {
        self.control.references()
    }
    fn branch_currents(&self) -> usize {
        usize::from(self.control.is_sensor())
    }
    fn branch_quantity(&self, _branch: usize) -> crate::devices::BranchQuantity {
        crate::devices::BranchQuantity::InputCurrent
    }
    fn findable_branch(&self) -> Option<usize> {
        self.control.is_sensor().then_some(0)
    }
    fn initial_voltage_relations(
        &self,
        _inductors_short: bool,
    ) -> Vec<crate::devices::InitialVoltageRelation> {
        use crate::devices::{InitialVoltageLaw, InitialVoltageRelation};
        if self.control.is_sensor() {
            vec![InitialVoltageRelation {
                terminals: self.sense_terminals(),
                law: InitialVoltageLaw::Fixed(0.),
            }]
        } else {
            Vec::new()
        }
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
        if self.control.is_sensor() {
            let [cp, cn] = self.sense_terminals();
            setup.voltage_branch(cp, cn, setup.branch(0)?);
        }
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
        let branch = if self.control.is_sensor() {
            Some(context.branch(0)?)
        } else {
            None
        };
        let control = self.control(context.unknowns, branch, context.controls, context.solution)?;
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
        let conductance = if context.integration.is_some() {
            state.step_conductance(self.material, history.map(State::from_values).transpose()?)?
        } else {
            state.at_conductance(self.material)?
        };
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
        )?;
        if let Some(branch) = branch {
            crate::devices::linear::branch_stamp(
                context.matrix,
                context.unknowns,
                self.sense_terminals(),
                branch,
            )?;
        }
        Ok(())
    }

    fn convergence_hook(&self) -> Option<&dyn crate::devices::DeviceConvergence> {
        Some(self)
    }

    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let state = self.frozen_state(context.states, || {
            self.control(context.unknowns, context.branch, context.controls, bias)
        })?;
        context.nodal(self.branch(), state.at_conductance(self.material)?, false)?;
        if self.control.is_sensor() {
            context.branch(self.sense_terminals())?;
        }
        Ok(())
    }

    fn noise(
        &self,
        context: &NoiseContext<'_>,
        sources: &mut Vec<NoiseSource>,
    ) -> SpiceResult<DeviceNoise<'_>> {
        let state = self.frozen_state(context.states, || {
            // Analysis supplies the solved device-local bias state. The
            // unloaded voltage receiver also supports a memoryless fallback.
            self.control(context.unknowns, None, context.controls, context.bias)
        })?;
        sources.push(NoiseSource::new(
            "",
            self.branch(),
            NoiseKind::Thermal {
                conductance: state.at_conductance(self.material)?,
                temperature: context.circuit_kelvin(),
            },
        ));
        Ok(DeviceNoise::Sources {
            family: if self.control.is_current() {
                NoiseFamily::CurrentSwitch
            } else {
                NoiseFamily::VoltageSwitch
            },
            model: Some(&self.model_name),
            total: false,
        })
    }

    fn assemble_nonlinear_charge(
        &self,
        _context: &mut crate::devices::traits::ChargeContext<'_>,
    ) -> SpiceResult<()> {
        // This memory is logical history, not electric charge or flux.
        Ok(())
    }
}

impl crate::devices::DeviceConvergence for TimedSwitch {
    fn convergence_test(&self, context: &ConvergenceContext<'_>) -> SpiceResult<bool> {
        let time = context.time;
        let control = self.control(
            context.unknowns,
            context.branch,
            context.controls,
            context.solution,
        )?;
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
        let old = if context.integration.is_some() {
            accepted.map(State::from_values).transpose()?
        } else {
            None
        };
        let expected = actual.step_conductance(self.material, old)? * voltage;
        let observed = stamped.step_conductance(self.material, old)? * voltage;
        if !expected.is_finite() || !observed.is_finite() {
            return Err(SpiceError::circuit("nonfinite timed switch branch current"));
        }
        let tolerance = context
            .reltol
            .mul_add(expected.abs().max(observed.abs()), context.abstol);
        Ok((expected - observed).abs() > tolerance)
    }
}

impl DeviceTransientEvents for TimedSwitch {
    fn next_boundary(&self, accepted: &[Real]) -> SpiceResult<Option<Real>> {
        State::from_values(accepted)?.next_boundary(self.material)
    }

    fn discontinuity(&self, accepted: &[Real], candidate: &[Real]) -> SpiceResult<Option<Real>> {
        Ok(State::from_values(candidate)?
            .discontinuity(self.material, State::from_values(accepted)?))
    }

    fn refine_trial(
        &self,
        context: &TransientEventContext<'_>,
        trial: &mut [Real],
    ) -> SpiceResult<Option<Real>> {
        let old = State::from_values(context.accepted)?;
        let control = self.control(
            context.unknowns,
            context.branch,
            context.controls,
            context.solution,
        )?;
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
