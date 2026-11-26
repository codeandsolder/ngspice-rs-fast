//! Independent conservative diode diffusion-charge state (`dioload.c`).
//!
//! Let `q` be the equilibrium diffusion charge filtered by a first-order
//! recovery law. For temperature-adjusted transit time `tt`,
//! `q + vp*tt*dq/dt = tt*i(v)` and the terminal current contribution is
//! `(1-vp)*dq/dt`. The internal MNA coordinate is `x = qpscale*q`, while
//! accepted integration history stores physical charge in coulombs. Scaling
//! the state equation by `qpscale` avoids dividing by small VP values.
use super::{Diode, DiodeModelParameters, JunctionPoint, Thermal, charge::ChargeBranch};
use crate::devices::{
    LinearContext, StampContext,
    traits::{ChargeContext, ConvergenceContext},
};
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

pub(super) const CHARGE_SLOT: usize = super::DIODE_STATE_COUNT;

/// Immutable setup coordinates and row normalization. Sensitivity
/// replays must not change their matrix basis when perturbing model VP
/// or the purely numerical coordinate scale.
#[derive(Debug, Clone, Copy)]
pub(super) struct RecoveryBinding {
    pub(super) node: NodeId,
    scale: Real,
    normalizer: Real,
}

impl RecoveryBinding {
    pub(super) const fn new(node: NodeId, model: &DiodeModelParameters) -> Self {
        Self {
            node,
            scale: model.qpscale,
            normalizer: model.vp,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Recovery {
    node: NodeId,
    scale: Real,
    vp: Real,
    tt: Real,
    row_weight: Real,
}

impl Diode {
    pub(super) fn recovery(&self, thermal: &Thermal) -> Option<Recovery> {
        self.recovery_node.map(|binding| Recovery {
            node: binding.node,
            scale: binding.scale,
            vp: self.parameters.model.vp,
            tt: thermal.tt,
            row_weight: binding.normalizer / self.parameters.model.vp,
        })
    }
}

impl Recovery {
    fn time_constant(self) -> Real {
        self.vp * self.tt
    }

    fn terminal_gain(self) -> Real {
        1. - self.vp
    }

    pub(super) fn stamp(
        self,
        context: &mut StampContext<'_>,
        [pos, neg]: [NodeId; 2],
        junction: JunctionPoint,
        voltage: Real,
    ) -> SpiceResult<()> {
        let coordinate = context.node_voltage(self.node);
        let charge = ChargeBranch::new(CHARGE_SLOT, coordinate / self.scale, self.scale.recip());
        let (history, conductance) = if let Some(coefficients) = context.integration {
            let companion = charge.integrate(&mut context.states, coefficients)?;
            (companion.current, companion.conductance)
        } else {
            charge.zero_derivative(&mut context.states)?;
            (0., 0.)
        };
        charge.store(&mut context.states)?;
        let equilibrium_gain = self.scale * self.tt;
        let slope = equilibrium_gain * junction.conductance;
        let dynamic_gain = self.time_constant() * self.scale;
        let dynamic_conductance = dynamic_gain * conductance;
        let terminal_conductance = self.terminal_gain() * conductance;
        let linearized_current = junction.conductance.mul_add(-voltage, junction.current);
        context.stamp_rhs(
            self.node,
            self.row_weight
                * (-dynamic_gain).mul_add(history, equilibrium_gain * linearized_current),
        )?;
        let terminal_history = self.terminal_gain() * history;
        context.stamp_rhs(pos, -terminal_history)?;
        context.stamp_rhs(neg, terminal_history)?;
        context.stamp_fixed([
            (
                self.node,
                self.node,
                self.row_weight * (1. + dynamic_conductance),
            ),
            (self.node, pos, -self.row_weight * slope),
            (self.node, neg, self.row_weight * slope),
            (pos, self.node, terminal_conductance),
            (neg, self.node, -terminal_conductance),
        ])
    }

    /// The numerical coordinate is not a voltage. Test its constitutive
    /// residual in amperes, independent of coordinate scaling.
    pub(super) fn not_converged(
        self,
        context: &ConvergenceContext<'_>,
        current: Real,
    ) -> SpiceResult<bool> {
        let q = context.voltage(self.node) / self.scale;
        let (derivative, rounding) = if let Some(integration) = context.integration {
            let companion = ChargeBranch::new(CHARGE_SLOT, q, 1.)
                .companion(integration, |age, slot| context.accepted(age, slot))?;
            let rounding = (4. * Real::EPSILON * companion.conductance.abs())
                .mul_add(q.abs(), 4. * Real::EPSILON * companion.current.abs());
            (companion.derivative, rounding)
        } else {
            (0., 0.)
        };
        let disagree = |actual: Real, expected: Real, floor: Real| {
            !actual.is_finite()
                || !expected.is_finite()
                || (actual - expected).abs()
                    > context
                        .reltol
                        .mul_add(actual.abs().max(expected.abs()), context.abstol + floor)
        };
        let last = |slot| {
            context
                .states
                .get(slot)
                .copied()
                .ok_or_else(|| SpiceError::circuit("missing recovery trial state"))
        };
        let charge_current = q / self.tt;
        let balance = self.vp.mul_add(derivative, charge_current);
        // A Newton trial stores the last load's state, not the newly solved
        // coordinate. Check both the physical evolution residual and the
        // stored state/current, otherwise a clamped junction could accept
        // a changed auxiliary coordinate with stale integration history.
        Ok(disagree(balance, current, self.vp * rounding)
            || disagree(charge_current, last(CHARGE_SLOT)? / self.tt, 0.)
            || disagree(
                self.terminal_gain() * derivative,
                self.terminal_gain() * last(CHARGE_SLOT + 1)?,
                self.terminal_gain().abs() * rounding,
            ))
    }

    /// The same conservative charge-state equation, before any integration.
    pub(super) fn stamp_taylor(
        self,
        context: &mut crate::devices::physical_dae::TaylorContext<'_>,
        junction: [Option<usize>; 2],
        current: &super::taylor::Law,
    ) -> SpiceResult<()> {
        let node = context.unknowns.node_row(self.node);
        let gain = -self.row_weight * self.scale * self.tt;
        context.branch_taylor(
            [node, None],
            junction,
            &current.value.clone().scale(gain),
            &current.slope.clone().scale(gain),
            false,
        )?;
        context.linear_static(node, node, self.row_weight)?;
        context.linear_charge(node, node, self.row_weight * self.time_constant())?;
        for (row, sign) in junction.into_iter().zip([1., -1.]) {
            context.linear_charge(row, node, sign * self.terminal_gain() / self.scale)?;
        }
        Ok(())
    }

    pub(super) fn assemble_charge(
        self,
        context: &mut ChargeContext<'_>,
        junction: [NodeId; 2],
    ) -> SpiceResult<()> {
        let coordinate = context.voltage(self.node);
        let control = [self.node, NodeId::GROUND];
        let terminal_slope = self.terminal_gain() / self.scale;
        let capacitance = self.row_weight * self.time_constant();
        context.branch(control, capacitance * coordinate, capacitance)?;
        // Terminal charge depends on the recovery coordinate, not on junction voltage.
        for (node, sign) in [(junction[0], 1.), (junction[1], -1.)] {
            if let Some(row) = context.unknowns.node_row(node) {
                context
                    .charge
                    .add_to(row, sign * terminal_slope * coordinate)?;
            }
        }
        context.coupling(junction, control, terminal_slope)
    }

    pub(super) fn assemble_linear(
        self,
        context: &mut LinearContext<'_>,
        [pos, neg]: [NodeId; 2],
        conductance: Real,
    ) -> SpiceResult<()> {
        let slope = self.row_weight * self.scale * self.tt * conductance;
        for (row, col, value) in [
            (self.node, self.node, self.row_weight),
            (self.node, pos, -slope),
            (self.node, neg, slope),
        ] {
            if let (Some(row), Some(col)) = (
                context.unknowns.node_row(row),
                context.unknowns.node_row(col),
            ) {
                context.system.a.add(row, col, value)?;
            }
        }
        let terminal_slope = self.terminal_gain() / self.scale;
        for (row, col, value) in [
            (self.node, self.node, self.row_weight * self.time_constant()),
            (pos, self.node, terminal_slope),
            (neg, self.node, -terminal_slope),
        ] {
            context.dynamic_sum(row, col, &[value])?;
        }
        Ok(())
    }
}
