//! Physical diode jets. The scalar laws supply the stable constant terms;
//! Taylor coefficients follow their analytic derivatives, with no
//! differencing of sampled currents or integrated companion loads.
use super::{
    DepletionLaw, Diode, Thermal, cubic_reverse_junction, exponential_junction_at_argument,
    high_injection,
};
use crate::devices::physical_dae::{ConstitutiveGuards, TaylorContext};
use crate::maths::taylor::Jet;
use crate::primitives::{Real, SpiceResult};

/// A constitutive value and its local partial derivative with respect to
/// junction voltage, both expanded along the same state germ.
pub(super) struct Law {
    pub(super) value: Jet,
    pub(super) slope: Jet,
}

impl Law {
    fn zero(v: &Jet) -> Self {
        Self {
            value: Jet::constant(v.order(), 0.),
            slope: Jet::constant(v.order(), 0.),
        }
    }

    fn add(self, rhs: &Self) -> Self {
        Self {
            value: self.value.add(&rhs.value),
            slope: self.slope.add(&rhs.slope),
        }
    }

    fn scale(self, scale: Real) -> Self {
        Self {
            value: self.value.scale(scale),
            slope: self.slope.scale(scale),
        }
    }

    fn multiply(self, rhs: &Self) -> Self {
        Self {
            value: self.value.multiply(&rhs.value),
            slope: self
                .slope
                .multiply(&rhs.value)
                .add(&self.value.multiply(&rhs.slope)),
        }
    }

    fn lift(v: &Jet, value: Real, slope: Jet) -> Self {
        Self {
            value: v.compose_derivative(value, &slope),
            slope,
        }
    }

    fn checked(self) -> SpiceResult<Self> {
        self.value.check()?;
        self.slope.check()?;
        Ok(self)
    }
}

// Stable device-local constitutive sites; not state-history offsets.
const BOTTOM_REGION: usize = 0;
const BOTTOM_BREAKDOWN: usize = 1;
const BOTTOM_FORWARD_KNEE: usize = 2;
const BOTTOM_REVERSE_KNEE: usize = 3;
const SIDEWALL_REGION: usize = 4;
const SIDEWALL_BREAKDOWN: usize = 5;
const SIDEWALL_KNEE: usize = 6;
const BOTTOM_DEPLETION: usize = 7;
const SIDEWALL_DEPLETION: usize = 8;

/// Lift the exponential using the scalar kernel's cancellation-safe value
/// and scaled derivative, including representable results beyond exp overflow.
pub(super) fn exponential(v: &Jet, vt: Real, sat: Real) -> Law {
    let argument = v.clone().scale(vt.recip()).with_value(v.value() / vt);
    let (value, derivative) = exponential_junction_at_argument(argument.value(), vt, sat);
    Law::lift(v, value, argument.exp_with_value(derivative))
}

fn characteristic(
    v: &Jet,
    vt: Real,
    sat: Real,
    breakdown: Option<(Real, Real)>,
    forward: bool,
    guards: &mut ConstitutiveGuards<'_>,
    breakdown_site: usize,
) -> SpiceResult<Law> {
    if let Some((bv, vbr)) = breakdown
        && !forward
        && guards.below(breakdown_site, v, -bv, sat != 0. && -bv < -3. * vt)?
    {
        let arg = v.clone().offset(bv).scale(-vbr.recip());
        let value = -sat * (-(bv + v.value()) / vbr).exp();
        return Law::lift(v, value, arg.exp_with_value(-value / vbr)).checked();
    }
    if sat == 0. {
        return Ok(Law::zero(v));
    }
    if !forward {
        let (value, derivative) = cubic_reverse_junction(v.value(), vt, sat);
        // The slope varies as v^-4. Normalize around the expansion point,
        // avoiding powers of a small dimensional voltage or of saturation current.
        let relative = v.clone().scale(v.value().recip()).with_value(1.);
        let slope = relative.integer_power(-4)?.scale(derivative);
        Law::lift(v, value, slope).checked()
    } else {
        exponential(v, vt, sat).checked()
    }
}

fn depletion(
    v: &Jet,
    law: DepletionLaw,
    cz: Real,
    guards: &mut ConstitutiveGuards<'_>,
    site: usize,
) -> SpiceResult<Law> {
    let (value, derivative) = law.charge(v.value(), cz, None);
    let slope = if cz == 0. {
        Jet::constant(v.order(), 0.)
    } else if guards.below(site, v, law.threshold, law.grading != 0.)? {
        let argument = v.clone().scale(-law.potential.recip()).offset(1.);
        let relative = argument
            .clone()
            .scale(argument.value().recip())
            .with_value(1.);
        relative
            .power(&Jet::constant(v.order(), -law.grading))?
            .scale(derivative)
    } else {
        v.clone()
            .offset(-v.value())
            .scale((cz / law.f2) * (law.grading / law.potential))
            .with_value(derivative)
    };
    Law::lift(v, value, slope).checked()
}

/// Lift the shared normalized junction law into any local gradient basis.
pub(in crate::devices) fn junction_differential<const N: usize>(
    v: &crate::maths::taylor::differential::Differential<N>,
    vt: Real,
    sat: Real,
    guards: &mut ConstitutiveGuards<'_>,
    site: usize,
) -> SpiceResult<crate::maths::taylor::differential::Differential<N>> {
    let forward = !guards.below(site, &v.value, -3. * vt, sat != 0.)?;
    let law = characteristic(&v.value, vt, sat, None, forward, guards, site)?;
    Ok(v.clone().lift(law.value, &law.slope))
}

pub(in crate::devices) fn depletion_differential<const N: usize>(
    v: &crate::maths::taylor::differential::Differential<N>,
    material: DepletionLaw,
    cap: Real,
    guards: &mut ConstitutiveGuards<'_>,
    site: usize,
) -> SpiceResult<crate::maths::taylor::differential::Differential<N>> {
    let law = depletion(&v.value, material, cap, guards, site)?;
    Ok(v.clone().lift(law.value, &law.slope))
}

fn generation(v: &Jet, law: DepletionLaw) -> SpiceResult<Law> {
    let arg = v.clone().scale(-law.potential.recip()).offset(1.);
    let base = arg.multiply(&arg).offset(0.005);
    let exponent = law.grading / 2.;
    let value = base.power(&Jet::constant(v.order(), exponent))?;
    let slope = base
        .power(&Jet::constant(v.order(), exponent - 1.))?
        .multiply(&arg)
        .scale(-law.grading / law.potential);
    Ok(Law { value, slope })
}

fn recombination(t: &Thermal, v: &Jet, sat: Real, nr: Real, forward: bool) -> SpiceResult<Law> {
    let vt = nr * t.vt;
    if forward {
        Ok(exponential(v, vt, sat).multiply(&generation(v, t.bottom)?))
    } else {
        let edge = Jet::constant(v.order(), -3. * (t.n * t.vt));
        let constant = exponential(&edge, vt, sat)
            .value
            .multiply(&generation(&edge, t.bottom)?.value);
        Ok(Law {
            value: constant,
            slope: Jet::constant(v.order(), 0.),
        })
    }
}

fn tunnel(v: &Jet, vt: Real, sat: Real) -> Law {
    let mirrored = v.clone().scale(-1.);
    let mut law = exponential(&mirrored, vt, sat).scale(-1.);
    law.slope = law.slope.scale(-1.);
    law
}

fn injection(law: &Law, knee: Real, forward: bool) -> SpiceResult<Law> {
    let (value, slope) = high_injection(law.value.value(), law.slope.value(), knee, forward);
    let root = law
        .value
        .clone()
        .scale(if forward { knee.recip() } else { -knee.recip() })
        .sqrt()?;
    let reciprocal = root.clone().offset(1.).reciprocal()?;
    let derivative = root
        .scale(0.5)
        .offset(1.)
        .multiply(&reciprocal)
        .multiply(&reciprocal);
    let current = law.value.compose_derivative(value, &derivative);
    let conductance = law.slope.multiply(&derivative).with_value(slope);
    Law {
        value: current,
        slope: conductance,
    }
    .checked()
}

impl Thermal {
    fn bottom_taylor(
        &self,
        v: &Jet,
        forward: bool,
        breakdown: Option<(Real, Real)>,
        guards: &mut ConstitutiveGuards<'_>,
    ) -> SpiceResult<Law> {
        let mut bottom = characteristic(
            v,
            self.n * self.vt,
            self.csat,
            breakdown,
            forward,
            guards,
            BOTTOM_BREAKDOWN,
        )?;
        if let Some((sat, nr)) = self.recombination {
            bottom = bottom.add(&recombination(self, v, sat, nr, forward)?);
        }
        if let Some(sat) = self.tunnel {
            bottom = bottom.add(&tunnel(v, self.ntun * self.vt, sat));
        }
        if forward {
            if let Some(knee) = self.ikf
                && guards.above(BOTTOM_FORWARD_KNEE, &bottom.value, 1e-18, true)?
            {
                bottom = injection(&bottom, knee, true)?;
            }
        } else if let Some(knee) = self.ikr
            && guards.below(BOTTOM_REVERSE_KNEE, &bottom.value, -1e-18, true)?
        {
            bottom = injection(&bottom, knee, false)?;
        }
        bottom.checked()
    }

    fn sidewall_taylor(
        &self,
        v: &Jet,
        forward: bool,
        breakdown: Option<(Real, Real)>,
        guards: &mut ConstitutiveGuards<'_>,
    ) -> SpiceResult<Law> {
        let mut sidewall = if let Some(sat) = self.csatsw {
            if let Some(ns) = self.ns {
                let vt = ns * self.vt;
                let forward = !guards.below(SIDEWALL_REGION, v, -3. * vt, sat != 0.)?;
                characteristic(v, vt, sat, breakdown, forward, guards, SIDEWALL_BREAKDOWN)?
            } else {
                let vt = self.n * self.vt;
                let blocked = if let Some((bv, _)) = breakdown
                    && !forward
                {
                    guards.below(SIDEWALL_BREAKDOWN, v, -bv, sat != 0. && -bv < -3. * vt)?
                } else {
                    false
                };
                if blocked {
                    Law::zero(v)
                } else {
                    characteristic(v, vt, sat, None, forward, guards, SIDEWALL_BREAKDOWN)?
                }
            }
        } else {
            Law::zero(v)
        };
        if let Some(sat) = self.tunnel_sidewall {
            sidewall = sidewall.add(&tunnel(v, self.ntun * self.vt, sat));
        }
        if let Some(knee) = self.ikp
            && guards.above(SIDEWALL_KNEE, &sidewall.value, 1e-18, true)?
        {
            sidewall = injection(&sidewall, knee, true)?;
        }
        sidewall.checked()
    }

    pub(super) fn taylor(
        &self,
        v: &Jet,
        gmin: Real,
        recovery: bool,
        guards: &mut ConstitutiveGuards<'_>,
    ) -> SpiceResult<(Law, Law)> {
        let active = self.csat != 0.
            || self.recombination.is_some_and(|(sat, _)| sat != 0.)
            || (self.ns.is_none() && self.csatsw.is_some_and(|sat| sat != 0.))
            || ((self.ikf.is_some() || self.ikr.is_some())
                && self.tunnel.is_some_and(|sat| sat != 0.));
        let forward = !guards.below(BOTTOM_REGION, v, -3. * (self.n * self.vt), active)?;
        let breakdown = self.breakdown.map(|bv| (bv, self.nbv * self.vt));
        let bottom = self.bottom_taylor(v, forward, breakdown, guards)?;
        let sidewall = self.sidewall_taylor(v, forward, breakdown, guards)?;
        let current = bottom.add(&sidewall).add(&Law {
            value: v.clone().scale(gmin),
            slope: Jet::constant(v.order(), gmin),
        });
        let diffusion = if recovery {
            Law::zero(v)
        } else {
            Law {
                value: current.value.clone(),
                slope: current.slope.clone(),
            }
            .scale(self.tt)
        };
        let charge = diffusion
            .add(&depletion(
                v,
                self.bottom,
                self.cjo,
                guards,
                BOTTOM_DEPLETION,
            )?)
            .add(&depletion(
                v,
                self.sidewall,
                self.cjsw,
                guards,
                SIDEWALL_DEPLETION,
            )?);
        Ok((current.checked()?, charge.checked()?))
    }
}

impl Diode {
    pub(super) fn stamp_taylor(&self, context: &mut TaylorContext<'_>) -> SpiceResult<()> {
        let thermal = self.thermal(context.model)?;
        let rows = self.junction.map(|node| context.unknowns.node_row(node));
        let v = context
            .variable(rows[0])?
            .add(&context.variable(rows[1])?.scale(-1.));
        let recovery = self.recovery(&thermal);
        let (current, charge) = thermal.taylor(
            &v,
            context.model.gmin,
            recovery.is_some(),
            &mut context.constitutive_guards(),
        )?;
        context.branch_taylor(rows, rows, &current.value, &current.slope, false)?;
        context.branch_taylor(rows, rows, &charge.value, &charge.slope, true)?;
        let gradients = if rows[0] == rows[1] {
            Vec::new()
        } else {
            rows.into_iter()
                .zip([1., -1.])
                .filter_map(|(column, sign)| {
                    column.map(|column| (column, charge.slope.clone().scale(sign)))
                })
                .collect()
        };
        context.record_charge(super::DIODE_CHARGE_SLOT, &charge.value, gradients)?;
        context.candidate_state(super::DIODE_VOLTAGE_SLOT, v.value())?;
        let external = context.unknowns.node_row(self.terminals[0]);
        if external != rows[0] {
            for (row, sign) in [external, rows[0]].into_iter().zip([1., -1.]) {
                context.linear_static(row, external, sign * thermal.conductance)?;
                context.linear_static(row, rows[0], -sign * thermal.conductance)?;
            }
        }
        if let Some(recovery) = recovery {
            recovery.stamp_taylor(context, rows, &current)?;
        }
        Ok(())
    }
}
