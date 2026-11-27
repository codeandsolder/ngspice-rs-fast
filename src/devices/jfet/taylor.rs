//! Analytic level-one channel and gate-junction Taylor equations.
use super::{Jfet, Model, Operating, slot};
use crate::devices::nonlinear::taylor::{characteristic, depletion};
use crate::devices::physical_dae::TaylorContext;
use crate::maths::taylor::Jet;
use crate::primitives::{Real, SpiceResult};
use std::cmp::Ordering;

struct ChannelJet {
    current: Jet,
    gate: Jet,
    drain: Jet,
}

impl Model {
    /// Channel and partials with respect to the active gate and the
    /// nonnegative drain voltage. Both polarities share this polynomial.
    fn channel_germ(&self, op: &Operating, gate: Jet, drain: &Jet) -> ChannelJet {
        let overdrive = gate.offset(-op.vto);
        if overdrive.sign() != Ordering::Greater {
            let zero = Jet::constant(overdrive.order(), 0.);
            return ChannelJet {
                current: zero.clone(),
                gate: zero.clone(),
                drain: zero,
            };
        }
        let beta = drain.clone().scale(self.lambda).offset(1.).scale(op.beta);
        let remaining = overdrive.clone().add(&drain.clone().scale(-1.));
        let (part, gate_partial, drain_partial) = if remaining.sign() != Ordering::Less {
            let apart = remaining.clone().scale(3. * op.bfac).offset(2. * self.b);
            let part = drain.multiply(
                &drain
                    .multiply(&drain.clone().scale(op.bfac).offset(-self.b))
                    .add(&overdrive.multiply(&apart)),
            );
            let gate_partial = drain.multiply(&overdrive.clone().scale(3. * op.bfac).add(&apart));
            (part, gate_partial, remaining.multiply(&apart))
        } else {
            let shape = overdrive.clone().scale(op.bfac);
            let part = overdrive
                .multiply(&overdrive)
                .multiply(&shape.clone().offset(self.b));
            let gate_partial = overdrive.multiply(&shape.scale(3.).offset(2. * self.b));
            (part, gate_partial, Jet::constant(overdrive.order(), 0.))
        };
        ChannelJet {
            current: beta.multiply(&part),
            gate: beta.multiply(&gate_partial),
            drain: beta
                .multiply(&drain_partial)
                .add(&part.scale(op.beta * self.lambda)),
        }
    }
}

impl Jfet {
    fn stamp_gate_germ(
        &self,
        context: &mut TaylorContext<'_>,
        op: &Operating,
        rows: [Option<usize>; 2],
        voltage: &Jet,
        cap: Real,
        charge_slot: usize,
    ) -> SpiceResult<()> {
        let pol = self.model.pol;
        let m = self.instance.m;
        let current = characteristic(voltage, op.vt_n, op.csat, None)?;
        let slope = current.slope.offset(context.model.gmin);
        let current = current
            .value
            .add(&voltage.clone().scale(context.model.gmin));
        context.branch_taylor(rows, rows, &current.scale(pol * m), &slope.scale(m), false)?;
        let charge = depletion(voltage, op.gate_law, cap)?;
        context.branch_taylor(
            rows,
            rows,
            &charge.value.clone().scale(pol * m),
            &charge.slope.clone().scale(m),
            true,
        )?;
        // C-compatible gate history is normalized and per unit M, unlike
        // physical KCL charge. Preserve that coordinate, including PJF.
        let gradients = if rows[0] == rows[1] {
            Vec::new()
        } else {
            rows.into_iter()
                .zip([pol, -pol])
                .filter_map(|(row, sign)| row.map(|row| (row, charge.slope.clone().scale(sign))))
                .collect()
        };
        context.record_charge(charge_slot, &charge.value, gradients)
    }

    pub(super) fn stamp_taylor(&self, context: &mut TaylorContext<'_>) -> SpiceResult<()> {
        let op = self.operating(context.model)?;
        let [drain, gate, source] = self.inner.map(|node| context.unknowns.node_row(node));
        let pol = self.model.pol;
        let m = self.instance.m;
        let vg = context.variable(gate)?;
        let vgs = vg
            .clone()
            .add(&context.variable(source)?.scale(-1.))
            .scale(pol);
        let vgd = vg.add(&context.variable(drain)?.scale(-1.)).scale(pol);
        self.stamp_gate_germ(context, &op, [gate, source], &vgs, op.czgs, slot::QGS)?;
        self.stamp_gate_germ(context, &op, [gate, drain], &vgd, op.czgd, slot::QGD)?;
        let vds = vgs.clone().add(&vgd.clone().scale(-1.));
        let (active, high, low, delta) = if vds.sign() == Ordering::Less {
            (vgd, source, drain, vds.scale(-1.))
        } else {
            (vgs, drain, source, vds)
        };
        let channel = self.model.channel_germ(&op, active, &delta);
        if high != low {
            for (row, sign) in [(high, 1.), (low, -1.)] {
                context.static_value(row, &channel.current, sign * pol * m)?;
                context.static_gradient(row, gate, &channel.gate, sign * m)?;
                context.static_gradient(row, high, &channel.drain, sign * m)?;
                context.static_gradient(
                    row,
                    low,
                    &channel.gate.clone().add(&channel.drain),
                    -sign * m,
                )?;
            }
        }
        for (outer, inner, conductance) in [
            (self.terminals[0], drain, op.gdpr),
            (self.terminals[2], source, op.gspr),
        ] {
            let outer = context.unknowns.node_row(outer);
            if outer != inner {
                for (row, sign) in [(outer, 1.), (inner, -1.)] {
                    context.linear_static(row, outer, sign * m * conductance)?;
                    context.linear_static(row, inner, -sign * m * conductance)?;
                }
            }
        }
        Ok(())
    }
}
