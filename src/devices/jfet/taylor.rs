//! Analytic level-one channel and gate-junction Taylor equations.
use super::{Jfet, Model, Operating, slot};
use crate::devices::nonlinear::taylor::{depletion_differential, junction_differential};
use crate::devices::physical_dae::{ConstitutiveGuards, TaylorContext, TaylorControl};
use crate::maths::taylor::differential::Differential;
use crate::primitives::SpiceResult;

type Law = Differential<2>;
const CHANNEL_DIRECTION: usize = 4;
const CHANNEL_CUTOFF: usize = 5;
const CHANNEL_SATURATION: usize = 6;

impl Model {
    /// The Sydney channel polynomial, evaluated in the active direction.
    /// The same local differential arithmetic propagates its partials.
    fn channel_germ(
        &self,
        op: &Operating,
        vgs: &Law,
        vgd: &Law,
        guards: &mut ConstitutiveGuards<'_>,
    ) -> SpiceResult<Law> {
        let vds = vgs.clone().add(&vgd.clone().scale(-1.));
        let active = op.beta != 0.;
        let (gate, drain, direction) = if guards.below(CHANNEL_DIRECTION, &vds.value, 0., active)? {
            (vgd, vds.scale(-1.), -1.)
        } else {
            (vgs, vds, 1.)
        };
        let overdrive = gate.clone().offset(-op.vto);
        if !guards.above(CHANNEL_CUTOFF, &overdrive.value, 0., active)? {
            return Ok(Law::constant(overdrive.value.order(), 0.));
        }
        let beta = drain.clone().scale(self.lambda).offset(1.).scale(op.beta);
        let remaining = overdrive.clone().add(&drain.clone().scale(-1.));
        let part = if guards.below(CHANNEL_SATURATION, &remaining.value, 0., active)? {
            let shape = overdrive.clone().scale(op.bfac).offset(self.b);
            overdrive.clone().multiply(&overdrive).multiply(&shape)
        } else {
            let apart = remaining.scale(3. * op.bfac).offset(2. * self.b);
            let shape = drain.clone().scale(op.bfac).offset(-self.b);
            drain
                .clone()
                .multiply(&drain.multiply(&shape).add(&overdrive.multiply(&apart)))
        };
        Ok(beta.multiply(&part).scale(direction))
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
        let mut gate = [(0., 0.); 2];
        for (index, cap, charge_slot) in [(0, op.czgs, slot::QGS), (1, op.czgd, slot::QGD)] {
            let v = &voltage[index];
            let current = junction_differential(
                v,
                op.vt_n,
                op.csat,
                &mut context.constitutive_guards(),
                index * 2,
            )?
            .add(&v.clone().scale(context.model.gmin));
            gate[index] = (
                current.value.value(),
                current.gradient[index]
                    .as_ref()
                    .map_or(0., crate::maths::taylor::Jet::value),
            );
            context.local_law(controls[index].rows, &current, &controls, pol * m, None)?;
            let charge = depletion_differential(
                v,
                op.gate_law,
                cap,
                &mut context.constitutive_guards(),
                index * 2 + 1,
            )?;
            // History is normalized and per-unit M, unlike physical KCL.
            context.local_law(
                controls[index].rows,
                &charge,
                &controls,
                pol * m,
                Some(charge_slot),
            )?;
        }
        let channel = self.model.channel_germ(
            &op,
            &voltage[0],
            &voltage[1],
            &mut context.constitutive_guards(),
        )?;
        context.local_law([drain, source], &channel, &controls, pol * m, None)?;
        let [gs, gd] = channel
            .gradient
            .each_ref()
            .map(|g| g.as_ref().map_or(0., crate::maths::taylor::Jet::value));
        let [(cgs, ggs), (cgd, ggd)] = gate;
        for (slot, value) in [
            (slot::VGS, voltage[0].value.value()),
            (slot::VGD, voltage[1].value.value()),
            (slot::CG, cgs + cgd),
            (slot::CD, channel.value.value() - cgd),
            (slot::CGD, cgd),
            (slot::GM, gs + gd),
            (slot::GDS, -gd),
            (slot::GGS, ggs),
            (slot::GGD, ggd),
        ] {
            context.candidate_state(slot, value)?;
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
