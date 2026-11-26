//! Independent conservative diode diffusion-charge state (`dioload.c`).
//!
//! Let `q` be the equilibrium diffusion charge filtered by a first-order
//! recovery law. For temperature-adjusted transit time `tt`,
//! `q + vp*tt*dq/dt = tt*i(v)` and the terminal current contribution is
//! `(1-vp)*dq/dt`. The internal MNA coordinate is `x = qpscale*q`, while
//! accepted integration history stores physical charge in coulombs. Scaling
//! the state equation by `qpscale` avoids dividing by small VP values.
use super::{Diode, JunctionPoint, Thermal, charge::ChargeBranch};
use crate::devices::{LinearContext, StampContext, traits::ChargeContext};
use crate::primitives::{NodeId, Real, SpiceResult};

pub(super) const CHARGE_SLOT: usize = super::DIODE_STATE_COUNT;

#[derive(Clone, Copy)]
pub(super) struct Recovery {
    node: NodeId,
    scale: Real,
    vp: Real,
    tt: Real,
}

impl Diode {
    pub(super) fn recovery(&self, thermal: &Thermal) -> Option<Recovery> {
        self.recovery_node.map(|node| Recovery {
            node,
            scale: self.parameters.model.qpscale,
            vp: self.parameters.model.vp,
            tt: thermal.tt,
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
            (-dynamic_gain).mul_add(history, equilibrium_gain * linearized_current),
        )?;
        let terminal_history = self.terminal_gain() * history;
        context.stamp_rhs(pos, -terminal_history)?;
        context.stamp_rhs(neg, terminal_history)?;
        context.stamp_fixed([
            (self.node, self.node, 1. + dynamic_conductance),
            (self.node, pos, -slope),
            (self.node, neg, slope),
            (pos, self.node, terminal_conductance),
            (neg, self.node, -terminal_conductance),
        ])
    }

    pub(super) fn assemble_charge(
        self,
        context: &mut ChargeContext<'_>,
        junction: [NodeId; 2],
    ) -> SpiceResult<()> {
        let coordinate = context.voltage(self.node);
        let control = [self.node, NodeId::GROUND];
        let terminal_slope = self.terminal_gain() / self.scale;
        context.branch(
            control,
            self.time_constant() * coordinate,
            self.time_constant(),
        )?;
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
        let slope = self.scale * self.tt * conductance;
        for (row, col, value) in [
            (self.node, self.node, 1.),
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
            (self.node, self.node, self.time_constant()),
            (pos, self.node, terminal_slope),
            (neg, self.node, -terminal_slope),
        ] {
            context.dynamic_sum(row, col, &[value])?;
        }
        Ok(())
    }
}
