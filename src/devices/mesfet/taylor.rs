//! Continuous Statz partition, independently integrated in each gate branch.
//! The incremental C companion uses the average of two rectangular paths;
//! shrinking the rectangle gives `dqgs = Q_gs dvgs`, `dqgd = Q_gd dvgd`.
use super::{DEL, Mesfet, Model, Operating, VMAX, VT0, jfet, slot};
use crate::devices::nonlinear::taylor::junction_differential;
use crate::devices::physical_dae::{TaylorContext, TaylorControl};
use crate::maths::taylor::differential::Differential;
use crate::primitives::{Real, SpiceResult};
use std::cmp::Ordering;

type Law = Differential<2>;

/// Stable smoothed maximum and weights, retaining the small weight even when
/// subtracting the large weight from one would round to zero.
fn smooth_high(a: &Law, b: &Law, width: Real) -> SpiceResult<(Law, [Law; 2])> {
    let difference = a.clone().add(&b.clone().scale(-1.));
    let (large, distance, inverse) = if difference.value.sign() == Ordering::Less {
        (b, difference.scale(-1.), true)
    } else {
        (a, difference, false)
    };
    // Scale before squaring; neither a large voltage nor a small smoothing
    // width alone should overflow the norm calculation.
    let norm = distance.value.value().abs().max(width);
    let scaled = distance.scale(norm.recip());
    let span = scaled
        .clone()
        .multiply(&scaled)
        .offset((width / norm).powi(2))
        .sqrt()?;
    let ratio = span.clone().reciprocal()?.scale(width / norm);
    let denominator = scaled.divide(&span)?.offset(1.).scale(2.);
    let small = ratio.clone().multiply(&ratio).divide(&denominator)?;
    let high = large.clone().add(&ratio.scale(width).divide(&denominator)?);
    let big = small.clone().scale(-1.).offset(1.);
    Ok((high, if inverse { [small, big] } else { [big, small] }))
}

impl Model {
    fn channel_germ(
        &self,
        op: &Operating,
        vgs: &Law,
        vgd: &Law,
        context: &mut TaylorContext<'_>,
    ) -> SpiceResult<Law> {
        let vds = vgs.clone().add(&vgd.clone().scale(-1.));
        context.guard(0, &vds.value, 0.)?;
        let (gate, drain, direction) = if vds.value.sign() == Ordering::Less {
            (vgd, vds.scale(-1.), -1.)
        } else {
            (vgs, vds, 1.)
        };
        let overdrive = gate.clone().offset(-self.vto);
        context.guard(1, &overdrive.value, 0.)?;
        if overdrive.value.sign() != Ordering::Greater {
            return Ok(Law::constant(overdrive.value.order(), 0.));
        }
        let fraction = drain.clone().scale(self.alpha / 3.);
        context.guard(2, &fraction.value, 1.)?;
        let shape = if fraction.value.compare(1.) == Some(Ordering::Less) {
            let complement = fraction.clone().scale(-1.).offset(1.);
            fraction.multiply(
                &complement
                    .clone()
                    .multiply(&complement)
                    .add(&complement)
                    .offset(1.),
            )
        } else {
            Law::constant(overdrive.value.order(), 1.)
        };
        let denominator = overdrive.clone().scale(self.b).offset(1.);
        Ok(drain
            .scale(self.lambda)
            .offset(1.)
            .multiply(
                &overdrive
                    .clone()
                    .multiply(&overdrive)
                    .divide(&denominator)?,
            )
            .multiply(&shape)
            .scale(direction * op.beta))
    }

    fn capacitance_germs(
        &self,
        op: &Operating,
        vgs: &Law,
        vgd: &Law,
        context: &mut TaylorContext<'_>,
    ) -> SpiceResult<[Law; 2]> {
        let order = vgs.value.order();
        if op.czgs == 0. && op.czgd == 0. {
            return Ok(std::array::from_fn(|_| Law::constant(order, 0.)));
        }
        let (gate, [plus, minus]) = smooth_high(vgs, vgd, 1. / self.alpha)?;
        let (bounded, [weight, _]) = smooth_high(&gate, &Law::constant(order, self.vto), DEL)?;
        context.guard(3, &bounded.value, VMAX)?;
        let voltage = if bounded.value.compare(VMAX) == Some(Ordering::Less) {
            bounded
        } else {
            Law::constant(order, VMAX)
        };
        let cap = voltage
            .scale(-1. / self.pb)
            .offset(1.)
            .sqrt()?
            .reciprocal()?
            .scale(op.czgs)
            .multiply(&weight);
        Ok([
            cap.clone()
                .multiply(&plus)
                .add(&minus.clone().scale(op.czgd)),
            cap.multiply(&minus).add(&plus.scale(op.czgd)),
        ])
    }
}

impl Mesfet {
    pub(super) fn stamp_taylor(&self, context: &mut TaylorContext<'_>) -> SpiceResult<()> {
        let op = self.operating()?;
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
        let caps = self
            .model
            .capacitance_germs(&op, &voltage[0], &voltage[1], context)?;
        for (index, charge_slot) in [(0, slot::QGS), (1, slot::QGD)] {
            let v = &voltage[index];
            context.guard(4 + index, &v.value, -3. * VT0)?;
            let current =
                junction_differential(v, VT0, op.csat)?.add(&v.clone().scale(context.model.gmin));
            context.local_law(controls[index].rows, &current, &controls, pol * m, None)?;
            if self.model.cgs != 0. || self.model.cgd != 0. {
                context.path_charge(
                    charge_slot,
                    pol * m,
                    controls[index],
                    &caps[index],
                    &controls,
                )?;
            }
        }
        let channel = self
            .model
            .channel_germ(&op, &voltage[0], &voltage[1], context)?;
        context.local_law([drain, source], &channel, &controls, pol * m, None)?;
        let [gd, gs] = jfet::series_conductances(self.instance.area, self.model.rd, self.model.rs);
        for (outer, inner, conductance) in [
            (self.terminals[0], drain, gd),
            (self.terminals[2], source, gs),
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
