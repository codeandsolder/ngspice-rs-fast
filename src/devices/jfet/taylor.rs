//! Analytic level-one channel and gate-junction Taylor equations.
use super::{Jfet, Model, Operating, slot};
use crate::devices::nonlinear::taylor::{depletion_differential, junction_differential};
use crate::devices::physical_dae::{TaylorContext, TaylorControl};
use crate::maths::taylor::differential::Differential;
use crate::primitives::SpiceResult;
use std::cmp::Ordering;

type Law = Differential<2>;

impl Model {
    /// The Sydney channel polynomial, evaluated in the active direction.
    /// The same local differential arithmetic propagates its partials.
    fn channel_germ(&self, op: &Operating, vgs: &Law, vgd: &Law) -> Law {
        let vds = vgs.clone().add(&vgd.clone().scale(-1.));
        let (gate, drain, direction) = if vds.value.sign() == Ordering::Less {
            (vgd, vds.scale(-1.), -1.)
        } else {
            (vgs, vds, 1.)
        };
        let overdrive = gate.clone().offset(-op.vto);
        if overdrive.value.sign() != Ordering::Greater {
            return Law::constant(overdrive.value.order(), 0.);
        }
        let beta = drain.clone().scale(self.lambda).offset(1.).scale(op.beta);
        let remaining = overdrive.clone().add(&drain.clone().scale(-1.));
        let part = if remaining.value.sign() != Ordering::Less {
            let apart = remaining.scale(3. * op.bfac).offset(2. * self.b);
            drain.clone().multiply(
                &drain
                    .clone()
                    .multiply(&drain.clone().scale(op.bfac).offset(-self.b))
                    .add(&overdrive.multiply(&apart)),
            )
        } else {
            let shape = overdrive.clone().scale(op.bfac).offset(self.b);
            overdrive.clone().multiply(&overdrive).multiply(&shape)
        };
        beta.multiply(&part).scale(direction)
    }
}

impl Jfet {
    pub(super) fn stamp_taylor(&self, context: &mut TaylorContext<'_>) -> SpiceResult<()> {
        let op = self.operating(context.model)?;
        let [drain, gate, source] = self.inner.map(|node| context.unknowns.node_row(node));
        let pol = self.model.pol;
        let m = self.instance.m;
        let controls = [
            TaylorControl {
                rows: [gate, source],
                polarity: pol,
            },
            TaylorControl {
                rows: [gate, drain],
                polarity: pol,
            },
        ];
        let voltage = [
            controls[0].input(0, context)?,
            controls[1].input(1, context)?,
        ];
        for (index, cap, charge_slot) in [(0, op.czgs, slot::QGS), (1, op.czgd, slot::QGD)] {
            let v = &voltage[index];
            let current = junction_differential(v, op.vt_n, op.csat)?
                .add(&v.clone().scale(context.model.gmin));
            context.local_law(controls[index].rows, &current, &controls, pol * m, None)?;
            let charge = depletion_differential(v, op.gate_law, cap)?;
            // History is normalized and per-unit M, unlike physical KCL.
            context.local_law(
                controls[index].rows,
                &charge,
                &controls,
                pol * m,
                Some(charge_slot),
            )?;
        }
        let channel = self.model.channel_germ(&op, &voltage[0], &voltage[1]);
        context.local_law([drain, source], &channel, &controls, pol * m, None)?;
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
