//! Physical diode jets. The scalar laws supply the stable constant terms;
//! Taylor coefficients follow their analytic derivatives, with no
//! differencing of sampled currents or integrated companion loads.
use super::{
    DepletionLaw, Diode, Thermal, cubic_reverse_junction, exponential_junction_at_argument,
    high_injection,
};
use crate::devices::physical_dae::TaylorContext;
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

    fn add(self, rhs: Self) -> Self {
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

    fn multiply(self, rhs: Self) -> Self {
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

fn below(v: &Jet, edge: Real) -> bool {
    v.compare(edge).is_some_and(std::cmp::Ordering::is_lt)
}

fn above(v: &Jet, edge: Real) -> bool {
    v.compare(edge).is_some_and(std::cmp::Ordering::is_gt)
}

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
) -> SpiceResult<Law> {
    if let Some((bv, vbr)) = breakdown
        && below(v, -3. * vt)
        && below(v, -bv)
    {
        let arg = v.clone().offset(bv).scale(-vbr.recip());
        let value = -sat * (-(bv + v.value()) / vbr).exp();
        return Law::lift(v, value, arg.exp_with_value(-value / vbr)).checked();
    }
    if sat == 0. {
        return Ok(Law::zero(v));
    }
    if below(v, -3. * vt) {
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

fn depletion(v: &Jet, law: DepletionLaw, cz: Real) -> SpiceResult<Law> {
    let (value, derivative) = law.charge(v.value(), cz, None);
    let slope = if cz == 0. {
        Jet::constant(v.order(), 0.)
    } else if below(v, law.threshold) {
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
) -> SpiceResult<crate::maths::taylor::differential::Differential<N>> {
    let law = characteristic(&v.value, vt, sat, None)?;
    Ok(v.clone().lift(law.value, &law.slope))
}

pub(in crate::devices) fn depletion_differential<const N: usize>(
    v: &crate::maths::taylor::differential::Differential<N>,
    material: DepletionLaw,
    cap: Real,
) -> SpiceResult<crate::maths::taylor::differential::Differential<N>> {
    let law = depletion(&v.value, material, cap)?;
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
        Ok(exponential(v, vt, sat).multiply(generation(v, t.bottom)?))
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

fn injection(law: Law, knee: Real, forward: bool) -> SpiceResult<Law> {
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
    ) -> SpiceResult<Law> {
        let mut bottom = characteristic(v, self.n * self.vt, self.csat, breakdown)?;
        if let Some((sat, nr)) = self.recombination {
            bottom = bottom.add(recombination(self, v, sat, nr, forward)?);
        }
        if let Some(sat) = self.tunnel {
            bottom = bottom.add(tunnel(v, self.ntun * self.vt, sat));
        }
        if forward {
            if let Some(knee) = self.ikf.filter(|_| above(&bottom.value, 1e-18)) {
                bottom = injection(bottom, knee, true)?;
            }
        } else if let Some(knee) = self.ikr.filter(|_| below(&bottom.value, -1e-18)) {
            bottom = injection(bottom, knee, false)?;
        }
        bottom.checked()
    }

    fn sidewall_taylor(
        &self,
        v: &Jet,
        forward: bool,
        breakdown: Option<(Real, Real)>,
    ) -> SpiceResult<Law> {
        let mut sidewall = if let Some(sat) = self.csatsw {
            match self.ns {
                Some(ns) => characteristic(v, ns * self.vt, sat, breakdown)?,
                None if forward || breakdown.is_none_or(|(bv, _)| !below(v, -bv)) => {
                    characteristic(v, self.n * self.vt, sat, None)?
                }
                None => Law::zero(v),
            }
        } else {
            Law::zero(v)
        };
        if let Some(sat) = self.tunnel_sidewall {
            sidewall = sidewall.add(tunnel(v, self.ntun * self.vt, sat));
        }
        if let Some(knee) = self.ikp.filter(|_| above(&sidewall.value, 1e-18)) {
            sidewall = injection(sidewall, knee, true)?;
        }
        sidewall.checked()
    }

    pub(super) fn taylor(&self, v: &Jet, gmin: Real, recovery: bool) -> SpiceResult<(Law, Law)> {
        let forward = !below(v, -3. * (self.n * self.vt));
        let breakdown = self.breakdown.map(|bv| (bv, self.nbv * self.vt));
        let bottom = self.bottom_taylor(v, forward, breakdown)?;
        let sidewall = self.sidewall_taylor(v, forward, breakdown)?;
        let current = bottom.add(sidewall).add(Law {
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
            .add(depletion(v, self.bottom, self.cjo)?)
            .add(depletion(v, self.sidewall, self.cjsw)?);
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
        let (current, charge) = thermal.taylor(&v, context.model.gmin, recovery.is_some())?;
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
