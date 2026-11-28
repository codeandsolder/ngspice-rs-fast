//! Analytic Gummel-Poon current and charge germs in normalized local voltages.
use super::physics::{RESISTANCE_REDUCTION_SERIES, RESISTANCE_SERIES_LIMIT, early_denominator};
use super::{Bjt, Thermal, slot};
use crate::devices::nonlinear::taylor::{
    depletion_differential as depletion_law, junction_differential as junction,
};
use crate::devices::nonlinear::{DepletionLaw, exponential_junction_at_argument};
use crate::devices::physical_dae::{TaylorContext, TaylorControl};
use crate::maths::taylor::differential::Differential;
use crate::primitives::{Real, SpiceResult};

const BE: usize = 0;
const BC: usize = 1;
const BX: usize = 2;
const SUB: usize = 3;
const BB: usize = 4;
type Law = Differential<5>;
fn base_charge(t: &Thermal, v: &[Law; 5], be: &Law, bc: &Law) -> SpiceResult<Law> {
    let mut denominator = v[BE]
        .clone()
        .scale(-t.inv_var)
        .add(&v[BC].clone().scale(-t.inv_vaf))
        .offset(1.);
    denominator.value = denominator.value.with_value(early_denominator(
        t,
        v[BE].value.value(),
        v[BC].value.value(),
    ));
    let q1 = denominator.reciprocal()?;
    if t.inv_ikf == 0. && t.inv_ikr == 0. {
        return Ok(q1);
    }
    let argument = be
        .clone()
        .scale(t.inv_ikf)
        .add(&bc.clone().scale(t.inv_ikr))
        .scale(4.)
        .offset(1.);
    let root = if argument.value.sign().is_gt() {
        argument.power(t.nkf.unwrap_or(0.5))?
    } else {
        // C uses a unit root on the clamped nonpositive branch.
        Law::constant(q1.value.order(), 1.)
    };
    Ok(q1.multiply(&root.offset(1.)).scale(0.5))
}

fn resistance_reduction(ratio: Law) -> SpiceResult<Law> {
    let inv_root = ratio.sqrt()?.reciprocal()?;
    let denominator = inv_root
        .clone()
        .multiply(&inv_root)
        .offset(14.59025)
        .sqrt()?
        .add(&inv_root);
    let angle = denominator.reciprocal()?.scale(14.59025 / 2.4317);
    if angle.value.value().abs() < RESISTANCE_SERIES_LIMIT {
        let squared = angle.clone().multiply(&angle);
        let reduction = RESISTANCE_REDUCTION_SERIES
            .iter()
            .rev()
            .fold(Law::constant(angle.value.order(), 0.), |value, &c| {
                value.multiply(&squared).offset(c)
            });
        Ok(reduction)
    } else {
        let tangent = angle.clone().tan()?;
        tangent
            .clone()
            .add(&angle.clone().scale(-1.))
            .scale(3.)
            .divide(&angle)?
            .divide(&tangent)?
            .divide(&tangent)
    }
}

fn base_conductance(t: &Thermal, base: &Law, qb: &Law) -> SpiceResult<Law> {
    let spread = t.rb - t.rbm;
    if spread == 0. {
        return Ok(Law::constant(
            base.value.order(),
            if t.rbm == 0. { 0. } else { t.rbm.recip() },
        ));
    }
    let resistance = if t.irb == 0. {
        qb.clone().reciprocal()?.scale(spread).offset(t.rbm)
    } else {
        let ratio = base.clone().scale(t.irb.recip());
        let ratio = if ratio
            .value
            .compare(1e-9)
            .is_some_and(std::cmp::Ordering::is_gt)
        {
            ratio
        } else {
            Law::constant(ratio.value.order(), 1e-9)
        };
        resistance_reduction(ratio)?.scale(spread).offset(t.rbm)
    };
    if resistance.value.value() == 0. {
        Ok(Law::constant(resistance.value.order(), 0.))
    } else {
        resistance.reciprocal()
    }
}

fn forward_diffusion(t: &Thermal, v: &[Law; 5], be: &Law, qb: &Law) -> SpiceResult<Law> {
    if t.tf == 0. || !v[BE].value.sign().is_gt() {
        return Ok(be.clone().scale(t.tf));
    }
    let mut factor = if t.xtf == 0. {
        Law::constant(be.value.order(), 0.)
    } else {
        v[BC].clone().scale(t.vtf_factor).exp().scale(t.xtf)
    };
    if t.xtf != 0. && t.itf != 0. {
        let ratio = be.clone().divide(&be.clone().offset(t.itf))?;
        factor = factor.multiply(&ratio).multiply(&ratio);
    }
    be.clone()
        .multiply(&factor.offset(1.))
        .divide(qb)
        .map(|charge| charge.scale(t.tf))
}

fn substrate(t: &Thermal, v: &Law, gmin: Real) -> SpiceResult<Law> {
    let mut current = v.clone().scale(gmin);
    if let Some(sat) = t.iss {
        let vt = t.ns * t.vt;
        let intrinsic = if v
            .value
            .compare(709. * vt)
            .is_some_and(std::cmp::Ordering::is_gt)
        {
            // The physical current is clamped, unlike C's retained Newton slope.
            let (value, _) = exponential_junction_at_argument(709., vt, sat);
            Law::constant(v.value.order(), value)
        } else {
            junction(v, vt, sat)?
        };
        current = current.add(&intrinsic);
    }
    Ok(current)
}

struct Point {
    collector: Law,
    base: Law,
    spread: Law,
    substrate: Law,
    charges: [Law; 4],
}

impl Thermal {
    fn taylor_point(&self, v: &[Law; 5], gmin: Real) -> SpiceResult<Point> {
        let be = junction(&v[BE], self.nf * self.vt, self.is_be)?;
        let bc = junction(&v[BC], self.nr * self.vt, self.is_bc)?;
        let be_leak =
            junction(&v[BE], self.ne * self.vt, self.ise)?.add(&v[BE].clone().scale(gmin));
        let bc_leak =
            junction(&v[BC], self.nc * self.vt, self.isc)?.add(&v[BC].clone().scale(gmin));
        let qb = base_charge(self, v, &be, &bc)?;
        let reverse = bc.clone().scale(self.br.recip()).add(&bc_leak);
        let base = be
            .clone()
            .scale(self.bf.recip())
            .add(&be_leak)
            .add(&reverse);
        let collector = be
            .clone()
            .add(&bc.clone().scale(-1.))
            .divide(&qb)?
            .add(&reverse.scale(-1.));
        let spread = base_conductance(self, &base, &qb)?.multiply(&v[BB]);
        let substrate = substrate(self, &v[SUB], gmin)?;
        let cap_bc = self.cjc * self.xcjc;
        let charges = [
            depletion_law(&v[BE], self.be, self.cje)?.add(&forward_diffusion(self, v, &be, &qb)?),
            depletion_law(&v[BC], self.bc, cap_bc)?.add(&bc.scale(self.tr)),
            depletion_law(&v[BX], self.bc, self.cjc - cap_bc)?,
            depletion_law(&v[SUB], DepletionLaw::new(self.vjs, self.mjs, 0.), self.cjs)?,
        ];
        Ok(Point {
            collector,
            base,
            spread,
            substrate,
            charges,
        })
    }
}

impl Bjt {
    pub(super) fn stamp_taylor(&self, context: &mut TaylorContext<'_>) -> SpiceResult<()> {
        let thermal = self.thermal(context.model)?;
        let nodes = self.nodes;
        let row = |node| context.unknowns.node_row(node);
        let [
            collector,
            base,
            emitter,
            substrate,
            collector_prime,
            base_prime,
            emitter_prime,
            substrate_connection,
        ] = [
            nodes.c, nodes.b, nodes.e, nodes.s, nodes.cp, nodes.bp, nodes.ep, nodes.sc,
        ]
        .map(row);
        let pol = self.pol;
        let multiplier = self.instance.multiplier;
        let controls = [
            TaylorControl {
                rows: [base_prime, emitter_prime],
                polarity: pol,
            },
            TaylorControl {
                rows: [base_prime, collector_prime],
                polarity: pol,
            },
            TaylorControl {
                rows: [base, collector_prime],
                polarity: pol,
            },
            TaylorControl {
                rows: [substrate, substrate_connection],
                polarity: pol * self.subs,
            },
            TaylorControl {
                rows: [base, base_prime],
                polarity: pol,
            },
        ];
        let voltages = [
            controls[BE].input(BE, context)?,
            controls[BC].input(BC, context)?,
            controls[BX].input(BX, context)?,
            controls[SUB].input(SUB, context)?,
            controls[BB].input(BB, context)?,
        ];
        let point = thermal.taylor_point(&voltages, context.model.gmin)?;
        for (rows, law, scale) in [
            (
                [collector_prime, emitter_prime],
                &point.collector,
                pol * multiplier,
            ),
            ([base_prime, emitter_prime], &point.base, pol * multiplier),
            ([base, base_prime], &point.spread, pol * multiplier),
            (
                [substrate, substrate_connection],
                &point.substrate,
                pol * self.subs * multiplier,
            ),
        ] {
            context.local_law(rows, law, &controls, scale, None)?;
        }
        for (index, slot) in [
            (BE, slot::QBE),
            (BC, slot::QBC),
            (BX, slot::QBX),
            (SUB, slot::QSUB),
        ] {
            context.local_law(
                controls[index].rows,
                &point.charges[index],
                &controls,
                controls[index].polarity * multiplier,
                Some(slot),
            )?;
        }
        for (outer, inner, conductance) in [
            (collector, collector_prime, thermal.gc),
            (emitter, emitter_prime, thermal.ge),
        ] {
            if outer != inner {
                for (row, sign) in [(outer, 1.), (inner, -1.)] {
                    context.linear_static(row, outer, sign * multiplier * conductance)?;
                    context.linear_static(row, inner, -sign * multiplier * conductance)?;
                }
            }
        }
        Ok(())
    }
}
