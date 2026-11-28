//! Constitutive laws shared by the conservative BSIM charge models.
//! Numerical rationalizations select by the constant term; physical
//! continuations select by the entire one-sided germ.
use std::cmp::Ordering;

use crate::maths::taylor::Constitutive;
use crate::primitives::{Real, SpiceResult};

const EXP_LIMIT: Real = 34.;
const MIN_EXP: Real = 1.713_908_431e-15;
const MAX_EXP: Real = 5.834_617_425e14;

/// The clipped thermal softplus and its independent voltage and slope-factor
/// partials. The upper asymptote is linear; the lower one is proportional
/// to the thermal slope factor. Preserve the reference clipping constants.
pub(super) fn thermal_softplus<S: Constitutive>(
    voltage: &S,
    factor: &S,
    thermal: Real,
) -> SpiceResult<(S, S, S)> {
    let argument = voltage.clone().divide(&factor.clone().scale(thermal))?;
    if argument.compare(EXP_LIMIT) == Some(Ordering::Greater) {
        return Ok((voltage.clone(), voltage.constant(1.), voltage.constant(0.)));
    }
    if argument.compare(-EXP_LIMIT) == Some(Ordering::Less) {
        let slope = thermal * MIN_EXP.ln_1p();
        return Ok((
            factor.clone().scale(slope),
            voltage.constant(0.),
            voltage.constant(slope),
        ));
    }
    thermal_softplus_interior(voltage, factor, thermal, argument)
}

/// Unclipped softplus. BSIM3's DC lower tail is a different law and may
/// not use the C-V exponential floor, even when its argument is below -34.
fn thermal_softplus_interior<S: Constitutive>(
    voltage: &S,
    factor: &S,
    thermal: Real,
    argument: S,
) -> SpiceResult<(S, S, S)> {
    // For z > 0, evaluate the linear asymptote plus its small tail.
    // This also avoids cancellation of the n-partial in automatic differentiation.
    if argument.primal() > 0. {
        let inverse = argument.clone().scale(-1.).exp();
        let divisor = inverse.clone().offset(1.);
        let tail = inverse.clone().divide(&divisor)?;
        let sigmoid = voltage.constant(1.).divide(&divisor)?;
        let logarithm = inverse.ln_1p()?;
        let entropy = argument.mul_add(&tail, &logarithm);
        let value = factor
            .clone()
            .multiply(&logarithm.scale(thermal))
            .add(voltage);
        let factor_partial = entropy.scale(thermal);
        let value = value.with_partials([voltage, factor], [&sigmoid, &factor_partial]);
        Ok((value, sigmoid, factor_partial))
    } else {
        let exponential = argument.clone().exp();
        let sigmoid = exponential
            .clone()
            .divide(&exponential.clone().offset(1.))?;
        let logarithm = exponential.ln_1p()?;
        let entropy = argument.scale(-1.).mul_add(&sigmoid, &logarithm);
        let value = factor.clone().multiply(&logarithm.scale(thermal));
        let factor_partial = entropy.scale(thermal);
        let value = value.with_partials([voltage, factor], [&sigmoid, &factor_partial]);
        Ok((value, sigmoid, factor_partial))
    }
}

/// BSIM3 DC effective overdrive. Its weak-inversion asymptote differs
/// from the BSIM4/C-V clipped softplus and is retained explicitly.
pub(super) fn bsim3_overdrive<S: Constitutive>(
    voltage: &S,
    factor: &S,
    thermal: Real,
    offset: Real,
    oxide_ratio: Real,
) -> SpiceResult<(S, S, S)> {
    let thermal_factor = factor.clone().scale(thermal);
    let double_thermal = thermal_factor.clone().scale(2.);
    let argument = voltage.clone().divide(&double_thermal)?;
    if argument.compare(EXP_LIMIT) == Some(Ordering::Greater) {
        return Ok((voltage.clone(), voltage.constant(1.), voltage.constant(0.)));
    }
    let denominator_argument = voltage
        .clone()
        .scale(-1.)
        .offset(2. * offset)
        .divide(&double_thermal)?;
    if denominator_argument.compare(EXP_LIMIT) == Some(Ordering::Greater) {
        let exponent = voltage.clone().offset(-offset).divide(&thermal_factor)?;
        let value = exponent.clone().exp().scale(thermal / oxide_ratio);
        let voltage_partial = value.clone().divide(&thermal_factor)?;
        let factor_partial = exponent.scale(-thermal).multiply(&voltage_partial);
        let value = value.with_partials([voltage, factor], [&voltage_partial, &factor_partial]);
        return Ok((value, voltage_partial, factor_partial));
    }
    let (numerator, numerator_voltage, numerator_factor) =
        thermal_softplus_interior(voltage, factor, 2. * thermal, argument)?;
    let exponential = denominator_argument.clone().exp().scale(2. * oxide_ratio);
    let denominator = factor.clone().mul_add(&exponential, &voltage.constant(1.));
    let denominator_voltage = exponential.clone().scale(-0.5 / thermal);
    let denominator_factor = denominator_argument
        .scale(-1.)
        .offset(1.)
        .multiply(&exponential);
    let value = numerator.divide(&denominator)?;
    let voltage_partial = value
        .clone()
        .scale(-1.)
        .mul_add(&denominator_voltage, &numerator_voltage)
        .divide(&denominator)?;
    let factor_partial = value
        .clone()
        .scale(-1.)
        .mul_add(&denominator_factor, &numerator_factor)
        .divide(&denominator)?;
    let value = value.with_partials([voltage, factor], [&voltage_partial, &factor_partial]);
    Ok((value, voltage_partial, factor_partial))
}

/// BSIM4 effective overdrive, shared between intrinsic DC and the
/// cvchargeMod=1 charge law. Returns value, partial in Vgst and partial in n.
pub(super) fn effective_overdrive<S: Constitutive>(
    voltage: &S,
    factor: &S,
    thermal: Real,
    mstar: Real,
    offset: Real,
    oxide_ratio: Real,
) -> SpiceResult<(S, S, S)> {
    let (numerator, numerator_voltage, numerator_factor) =
        thermal_softplus(&voltage.clone().scale(mstar), factor, thermal)?;
    let argument = voltage
        .clone()
        .scale(mstar - 1.)
        .offset(offset)
        .divide(&factor.clone().scale(thermal))?;
    let (exponential, exp_slope) = if argument.compare(-EXP_LIMIT) == Some(Ordering::Less) {
        (voltage.constant(MIN_EXP), voltage.constant(0.))
    } else if argument.compare(EXP_LIMIT) == Some(Ordering::Greater) {
        (voltage.constant(MAX_EXP), voltage.constant(0.))
    } else {
        let value = argument.clone().exp();
        (value.clone(), value)
    };
    let denominator_voltage = exp_slope
        .clone()
        .scale(oxide_ratio * (mstar - 1.) / thermal);
    let denominator_factor = argument
        .scale(-1.)
        .mul_add(&exp_slope, &exponential)
        .scale(oxide_ratio);
    let denominator = factor
        .clone()
        .multiply(&exponential.scale(oxide_ratio))
        .offset(mstar);
    let value = numerator.divide(&denominator)?;
    let voltage_partial = value
        .clone()
        .scale(-1.)
        .mul_add(&denominator_voltage, &numerator_voltage.scale(mstar))
        .divide(&denominator)?;
    let factor_partial = value
        .clone()
        .scale(-1.)
        .mul_add(&denominator_factor, &numerator_factor)
        .divide(&denominator)?;
    let value = value.with_partials([voltage, factor], [&voltage_partial, &factor_partial]);
    Ok((value, voltage_partial, factor_partial))
}

/// Bias-dependent overlap charge and capacitance. The two rationalized
/// square-root offsets are essential in the small-charge/large-bias tails.
pub(super) fn overlap<S: Constitutive>(
    voltage: S,
    base: Real,
    extra: Real,
    kappa: Real,
) -> SpiceResult<(S, S)> {
    const DELTA: Real = 0.02;
    let offset = voltage.clone().offset(DELTA);
    let root = offset
        .clone()
        .mul_add(&offset, &voltage.constant(4. * DELTA))
        .sqrt()?;
    // These are algebraically equivalent expressions, not separate
    // physical branches. Do not emit a constitutive event here.
    let half = if offset.primal() > 0. {
        voltage.constant(-2. * DELTA).divide(&offset.add(&root))?
    } else {
        offset.add(&root.clone().scale(-1.)).scale(0.5)
    };
    let ratio = half.clone().scale(-4.).divide(&voltage.constant(kappa))?;
    let secondary = ratio.clone().offset(1.).sqrt()?;
    let secondary_offset = ratio.divide(&secondary.clone().offset(1.))?;
    let correction = secondary_offset
        .clone()
        .divide(&secondary)?
        .multiply(&half.clone().scale(-1.).divide(&root)?);
    let capacitance = voltage
        .constant(extra)
        .mul_add(&correction.scale(-1.), &voltage.constant(base + extra));
    let correction = half
        .scale(-1.)
        .multiply(&secondary_offset.divide(&secondary.offset(1.))?);
    let charge = voltage
        .constant(extra)
        .mul_add(&correction, &voltage.scale(base + extra));
    Ok((capacitance, charge))
}

#[derive(Clone, Copy)]
pub(super) struct JunctionCapacitance {
    pub(super) zero: Real,
    pub(super) potential: Real,
    pub(super) grading: Real,
}

/// Three physical junction regions (bottom, sidewall, gate sidewall). The
/// reverse integral and quadratic forward continuation meet at zero bias.
/// Returns (charge, differential capacitance), before multiplicity. The
/// observer receives the running charge in reference region order, so the
/// BSIM4 host retains the state-write sequence while other consumers can
/// use the final constitutive pair directly.
pub(super) fn depletion<S: Constitutive>(
    voltage: &S,
    parts: [JunctionCapacitance; 3],
    mut observe_charge: impl FnMut(&S),
) -> SpiceResult<(S, S)> {
    let [bottom, side, gate] = parts;
    let zero = bottom.zero + side.zero + gate.zero;
    if voltage.compare(0.) == Some(Ordering::Less) {
        let mut charge = voltage.constant(0.);
        let mut cap = voltage.constant(0.);
        for (index, part) in parts.into_iter().enumerate() {
            if part.zero > 0. {
                let relative = voltage
                    .clone()
                    .scale(-1.)
                    .divide(&voltage.constant(part.potential))?;
                let (factor, integral) = if part.grading.to_bits() == 0.5_f64.to_bits() {
                    let root = relative.clone().offset(1.).sqrt()?;
                    (
                        voltage.constant(1.).divide(&root)?,
                        relative.divide(&root.offset(1.))?.scale(-2.),
                    )
                } else {
                    let log = relative.ln_1p()?;
                    let exponent = 1. - part.grading;
                    let factor = log.clone().scale(-part.grading).exp();
                    let integral = if exponent == 0. {
                        log.scale(-1.)
                    } else {
                        log.scale(exponent)
                            .exp_m1()
                            .scale(-1.)
                            .divide(&voltage.constant(exponent))?
                    };
                    (factor, integral)
                };
                charge = voltage
                    .constant(part.potential * part.zero)
                    .mul_add(&integral, &charge);
                cap = voltage.constant(part.zero).mul_add(&factor, &cap);
                observe_charge(&charge);
            } else if index == 0 {
                observe_charge(&charge);
            }
        }
        Ok((charge, cap))
    } else {
        let increment = voltage.clone().scale(
            bottom.zero * bottom.grading / bottom.potential
                + side.zero * side.grading / side.potential
                + gate.zero * gate.grading / gate.potential,
        );
        let charge = voltage.clone().multiply(
            &voltage
                .constant(0.5)
                .mul_add(&increment, &voltage.constant(zero)),
        );
        observe_charge(&charge);
        Ok((charge, increment.offset(zero)))
    }
}

/// Tangent-matched rational continuation of a linear constitutive factor.
/// Below the join, the displacement is compressed toward -1/stiffness;
/// both value and first partial match the linear law at the join.
#[inline]
pub(super) fn rational_floor<S: Constitutive>(
    x: S,
    join: Real,
    offset: Real,
    stiffness: Real,
) -> SpiceResult<(S, S)> {
    if x.compare(join) == Some(Ordering::Less) {
        let displacement = x.offset(-join);
        let one = displacement.constant(1.);
        let inverse = one
            .clone()
            .divide(&displacement.scale(-stiffness).add(&one))?;
        let limit = join + offset - stiffness.recip();
        let value = inverse
            .constant(stiffness.recip())
            .mul_add(&inverse, &inverse.constant(limit));
        let slope = inverse.clone().multiply(&inverse);
        Ok((value, slope))
    } else {
        let one = x.constant(1.);
        Ok((x.offset(offset), one))
    }
}

/// Reciprocal of the same continued resistance/body-bias factor.
#[inline]
pub(super) fn reciprocal_floor<S: Constitutive>(
    x: S,
    join: Real,
    stiffness: Real,
) -> SpiceResult<(S, S)> {
    let (value, slope) = rational_floor(x, join, 1., stiffness)?;
    let reciprocal = value.constant(1.).divide(&value)?;
    let derivative = slope
        .scale(-1.)
        .multiply(&reciprocal.clone().multiply(&reciprocal));
    Ok((reciprocal, derivative))
}

/// Smooth positive half-root, evaluated using its conjugate on the small
/// negative tail. The selection is numerical, not a physical branch.
pub(super) fn positive_half_sum<S: Constitutive>(x: S, root: S, product: S) -> SpiceResult<S> {
    if x.primal() >= 0. {
        Ok(root.midpoint(&x))
    } else {
        product.scale(2.).divide(&root.add(&x.scale(-1.)))
    }
}

/// Regularized positive part and its partial in the input coordinate.
#[inline]
pub(super) fn smooth_positive<S: Constitutive>(
    x: S,
    quarter_width_squared: Real,
) -> SpiceResult<(S, S)> {
    if quarter_width_squared == 0.
        && let Some(direction) = x.compare(0.)
    {
        return Ok(match direction {
            Ordering::Less => (x.constant(0.), x.constant(0.)),
            Ordering::Greater => {
                let slope = x.constant(1.);
                (x, slope)
            }
            Ordering::Equal => (x.clone().scale(0.5), x.constant(0.5)),
        });
    }
    let root = x
        .clone()
        .mul_add(&x, &x.constant(4. * quarter_width_squared))
        .sqrt()?;
    let value = positive_half_sum(x.clone(), root.clone(), x.constant(quarter_width_squared))?;
    let slope_value = value.clone().divide(&root)?.primal();
    let curvature = x
        .constant(2. * quarter_width_squared)
        .divide(&root)?
        .divide(&root)?
        .divide(&root)?;
    let slope = x.clone().compose_derivative(slope_value, &curvature);
    let value = x.compose_derivative(value.primal(), &slope);
    Ok((value, slope))
}

/// Smaller root of y² - (sat + drain + delta)y + sat*drain = 0.
/// Retain the complementary drain gap and both independent partials.
pub(super) struct DrainSmoothing<S> {
    pub(super) effective: S,
    pub(super) drain_excess: S,
    pub(super) saturation_excess: S,
    pub(super) saturation_partial: S,
    pub(super) drain_partial: S,
}

#[inline]
pub(super) fn smooth_drain<S: Constitutive>(
    saturation: S,
    drain: S,
    delta: Real,
) -> SpiceResult<DrainSmoothing<S>> {
    let x = saturation
        .clone()
        .add(&drain.clone().scale(-1.))
        .offset(-delta);
    let root = x
        .clone()
        .mul_add(&x, &saturation.clone().scale(4. * delta))
        .sqrt()?;
    let denominator = saturation.clone().add(&drain).offset(delta).add(&root);
    // Use the conjugate where its denominator is nonzero. Outside that
    // numerical region the direct quadratic root is still the same law.
    let effective = if denominator.primal() > 0. && delta >= 0. {
        saturation
            .clone()
            .divide(&denominator)?
            .scale(2.)
            .multiply(&drain)
    } else {
        saturation
            .clone()
            .add(&x.clone().midpoint(&root).scale(-1.))
    };
    let drain_excess = positive_half_sum(
        drain
            .clone()
            .add(&saturation.clone().scale(-1.))
            .offset(-delta),
        root.clone(),
        drain.scale(delta),
    )?;
    let saturation_excess = positive_half_sum(x, root.clone(), saturation.scale(delta))?;
    Ok(DrainSmoothing {
        effective,
        saturation_partial: drain_excess.clone().divide(&root)?,
        drain_partial: saturation_excess.clone().divide(&root)?,
        drain_excess,
        saturation_excess,
    })
}

/// Smooth lower body-bias limit and its exact differential. The conjugate
/// expression preserves the zero and tiny-bias limit without subtracting
/// nearly equal threshold voltages.
pub(super) fn body_lower_limit<S: Constitutive>(bias: &S, cutoff: Real) -> SpiceResult<(S, S)> {
    const DELTA: Real = 0.001;
    let argument = bias.clone().offset(-cutoff).offset(-DELTA);
    let width = -DELTA * cutoff;
    let root = argument
        .clone()
        .mul_add(&argument, &bias.constant(4. * width))
        .sqrt()?;
    let half = positive_half_sum(argument.clone(), root.clone(), bias.constant(width))?;
    let slope_value = half.clone().divide(&root)?.primal();
    let curvature = bias
        .constant(2. * width)
        .divide(&root)?
        .divide(&root)?
        .divide(&root)?;
    let derivative = argument.compose_derivative(slope_value, &curvature);
    let sum = bias.clone().offset(cutoff - DELTA);
    let value = if sum.primal() <= 0. && cutoff < 0. {
        bias.clone()
            .scale(2. * cutoff)
            .divide(&sum.add(&root.scale(-1.)))?
    } else {
        half.offset(cutoff)
    };
    Ok((
        bias.clone().compose_derivative(value.primal(), &derivative),
        derivative,
    ))
}

/// BSIM3 reference body limiter: the charge/current value is clamped to
/// the applied bias when the smoothing lies below it. The original C
/// stamp retains the smoothing partial even on that clamp. Differential
/// arithmetic of the returned value follows the actual branch instead.
pub(super) fn bsim3_body_limit<S: Constitutive>(bias: &S, cutoff: Real) -> SpiceResult<(S, S)> {
    let (raw, reference_partial) = body_lower_limit(bias, cutoff)?;
    let clamp = if cutoff < 0. {
        bias.compare(0.) == Some(Ordering::Greater)
    } else {
        raw.clone().add(&bias.clone().scale(-1.)).compare(0.) == Some(Ordering::Less)
    };
    let value = if clamp { bias.clone() } else { raw };
    Ok((value, reference_partial))
}

/// BSIM4 upper body-bias limiter at 0.95 phi, in addition to the common
/// lower threshold. Both are smooth physical laws.
pub(super) fn bsim4_body_limit<S: Constitutive>(
    bias: &S,
    cutoff: Real,
    phi: Real,
) -> SpiceResult<(S, S)> {
    let (lower, lower_partial) = body_lower_limit(bias, cutoff)?;
    let upper_limit = 0.95 * phi;
    let argument = lower.clone().scale(-1.).offset(upper_limit - 0.001);
    let (_, slope) = smooth_positive(argument, 0.001 * upper_limit)?;
    // The conjugate smaller root retains the value at near-zero body bias.
    let upper = smooth_drain(bias.constant(upper_limit), lower.clone(), 0.001)?;
    let value = lower.compose_derivative(upper.effective.primal(), &slope);
    Ok((value, slope.multiply(&lower_partial)))
}

/// Surface potential and the model's square-root factor and local partials.
/// BSIM3 uses a rational approximation in forward body bias; BSIM4 uses
/// the actual square root of the upper-limited potential.
pub(super) struct SurfacePotential<S> {
    pub(super) potential: S,
    pub(super) root: S,
    pub(super) potential_partial: S,
    pub(super) root_partial: S,
}

pub(super) fn surface_potential<S: Constitutive>(
    body: &S,
    phi: Real,
    forward_approximation: Option<Real>,
) -> SpiceResult<SurfacePotential<S>> {
    if let Some(phis3) = forward_approximation
        && body.compare(0.) == Some(Ordering::Greater)
    {
        let ratio = body.constant(phi).divide(&body.clone().offset(phi))?;
        let root = body
            .constant(phis3)
            .divide(&body.clone().scale(0.5).offset(phi))?;
        let root_partial = root.clone().scale(-0.5 / phis3).multiply(&root);
        Ok(SurfacePotential {
            potential: ratio.clone().scale(phi),
            potential_partial: ratio.clone().scale(-1.).multiply(&ratio),
            root,
            root_partial,
        })
    } else {
        let potential = body.clone().scale(-1.).offset(phi);
        let root = potential.clone().sqrt()?;
        let root_partial = body.constant(-0.5).divide(&root)?;
        Ok(SurfacePotential {
            potential,
            root,
            potential_partial: body.constant(-1.),
            root_partial,
        })
    }
}

/// Short/narrow-channel threshold factor and its exact exponent partial.
/// BSIM4 uses expm1 near zero instead of losing the small difference from
/// unity; the original `MIN_EXP` regularizer is retained.
#[derive(Clone, Copy)]
pub(super) enum ThresholdRollOff {
    Bsim3,
    Bsim4,
}

impl ThresholdRollOff {
    pub(super) fn evaluate<S: Constitutive>(self, argument: &S) -> SpiceResult<(S, S)> {
        match self {
            Self::Bsim3 => {
                if argument.compare(-EXP_LIMIT) != Some(Ordering::Greater) {
                    return Ok((
                        argument.constant(MIN_EXP * 2.0_f64.mul_add(MIN_EXP, 1.)),
                        argument.constant(0.),
                    ));
                }
                let exponential = argument.clone().exp();
                let value = exponential
                    .clone()
                    .scale(2.)
                    .offset(1.)
                    .multiply(&exponential);
                let slope = exponential
                    .clone()
                    .scale(4.)
                    .offset(1.)
                    .multiply(&exponential);
                Ok((value, slope))
            }
            Self::Bsim4 => {
                if argument.compare(EXP_LIMIT) != Some(Ordering::Less) {
                    return Ok((
                        argument.constant(1. / (MAX_EXP - 2.)),
                        argument.constant(0.),
                    ));
                }
                // Divide numerator and denominator by exp(x). Evaluate the
                // resulting even law on the negative exponential tail.
                // This retains a zero derivative at the exact extremum.
                let sign = if argument.primal() < 0. { -1. } else { 1. };
                let z = argument.clone().scale(-sign);
                let exponential = z.clone().exp();
                let offset = z.exp_m1().scale(-1.);
                let denominator = offset
                    .clone()
                    .mul_add(&offset, &exponential.clone().scale(2. * MIN_EXP));
                let value = exponential.clone().divide(&denominator)?;
                let slope = offset
                    .scale(-sign)
                    .multiply(&exponential.offset(1.))
                    .divide(&denominator)?
                    .multiply(&value);
                let value = argument.clone().compose_derivative(value.primal(), &slope);
                Ok((value, slope))
            }
        }
    }
}

/// Short/narrow-channel factor and independent body/depletion partials.
pub(super) fn channel_rolloff<S: Constitutive>(
    body: &S,
    depletion: &S,
    body_coefficient: Real,
    length_factor: Real,
    argument_scale: Real,
    generation: ThresholdRollOff,
) -> SpiceResult<LocalPart<S, 2>> {
    let (body_factor, slope) = rational_floor(body.clone().scale(body_coefficient), -0.5, 1., 8.)?;
    let length = depletion
        .clone()
        .sqrt()?
        .multiply(&body_factor)
        .scale(length_factor);
    let argument = body.constant(argument_scale).divide(&length)?;
    let (value, exponent_partial) = generation.evaluate(&argument)?;
    let common = exponent_partial.multiply(&argument).scale(-1.);
    let body_partial = common
        .clone()
        .multiply(&slope)
        .scale(body_coefficient)
        .divide(&body_factor)?;
    let depletion_partial = common.scale(0.5).divide(depletion)?;
    Ok(LocalPart {
        value,
        partials: [body_partial, depletion_partial],
    })
}

/// Bias-independent threshold coefficients shared by both generations.
/// The fixed term includes nominal, reference surface, temperature and
/// width corrections; the linear body term includes their body partial.
pub(super) struct ThresholdParameters {
    pub(super) fixed: Real,
    pub(super) surface: Real,
    pub(super) body: Real,
    pub(super) short: Real,
    pub(super) narrow: Real,
    pub(super) eta0: Real,
    pub(super) etab: Real,
    pub(super) theta0vb0: Real,
}

impl ThresholdParameters {
    /// Value and independent partials in body, drain, surface-root,
    /// short-channel factor, narrow-channel factor.
    pub(super) fn evaluate<S: Constitutive>(
        &self,
        body: &S,
        drain: &S,
        surface: &S,
        short: &S,
        narrow: &S,
    ) -> SpiceResult<LocalPart<S, 5>> {
        let (eta, slope) = rational_floor(
            body.clone().scale(self.etab).offset(self.eta0),
            1e-4,
            0.,
            2e4,
        )?;
        let dibl = eta.scale(-self.theta0vb0);
        let value = body
            .clone()
            .scale(self.body)
            .offset(self.fixed)
            .add(&surface.clone().scale(self.surface))
            .add(&short.clone().scale(self.short))
            .add(&narrow.clone().scale(self.narrow));
        let value = dibl.clone().mul_add(drain, &value);
        let body_partial = slope
            .scale(-self.etab * self.theta0vb0)
            .mul_add(drain, &body.constant(self.body));
        Ok(LocalPart {
            value,
            partials: [
                body_partial,
                dibl,
                body.constant(self.surface),
                body.constant(self.short),
                body.constant(self.narrow),
            ],
        })
    }
}

/// Subthreshold slope-factor parameters, including the bias-dependent
/// charge sharing coefficients and interface/oxide capacitances.
pub(super) struct SlopeParameters {
    pub(super) depletion: Real,
    pub(super) coupling: [Real; 3],
    pub(super) interface: Real,
    pub(super) oxide: Real,
}

impl SlopeParameters {
    /// Value and independent partials in body, drain, depletion depth
    /// and the short-channel threshold factor.
    pub(super) fn evaluate<S: Constitutive>(
        &self,
        body: &S,
        drain: &S,
        depletion: &S,
        theta: &S,
    ) -> SpiceResult<LocalPart<S, 4>> {
        let [constant, body_coefficient, drain_coefficient] = self.coupling;
        let coupling = body.constant(drain_coefficient).mul_add(
            drain,
            &body
                .constant(body_coefficient)
                .mul_add(body, &body.constant(constant)),
        );
        let capacitance = body.constant(self.depletion).divide(depletion)?;
        let raw = coupling
            .clone()
            .mul_add(theta, &capacitance.clone().offset(self.interface))
            .divide(&body.constant(self.oxide))?;
        let (value, slope) = rational_floor(raw, -0.5, 1., 8.)?;
        let scale = slope.scale(self.oxide.recip());
        let body_partial = theta.clone().scale(body_coefficient).multiply(&scale);
        let drain_partial = theta.clone().scale(drain_coefficient).multiply(&scale);
        let depletion_partial = capacitance.scale(-1.).divide(depletion)?.multiply(&scale);
        let theta_partial = coupling.multiply(&scale);
        Ok(LocalPart {
            value,
            partials: [
                body_partial,
                drain_partial,
                depletion_partial,
                theta_partial,
            ],
        })
    }
}

/// Saturation-shape coefficient, with the exact overdrive partial.
/// The sign of a1 selects a parameterization, not a trajectory event.
#[inline]
pub(super) fn saturation_shape<S: Constitutive>(
    overdrive: &S,
    a1: Real,
    a2: Real,
) -> SpiceResult<(S, S)> {
    if a1 == 0. {
        return Ok((overdrive.constant(a2), overdrive.constant(0.)));
    }
    let (argument, width) = if a1 > 0. {
        (
            overdrive
                .constant(-a1)
                .mul_add(overdrive, &overdrive.constant(1. - a2 - 0.0001)),
            0.0001 * (1. - a2),
        )
    } else {
        (
            overdrive
                .constant(a1)
                .mul_add(overdrive, &overdrive.constant(a2 - 0.0001)),
            0.0001 * a2,
        )
    };
    let (root, partial) = smooth_positive(argument, width)?;
    let value = if a1 > 0. {
        root.scale(-1.).offset(1.)
    } else {
        root
    };
    Ok((value, partial.scale(a1)))
}

/// Substrate-current multiplier before the channel-current product.
/// The exponential is clipped exactly where the compact model clips it.
#[inline]
pub(super) fn impact_ionization<S: Constitutive>(
    excess: &S,
    alpha: Real,
    beta: Real,
) -> SpiceResult<(S, S)> {
    if alpha <= 0. || beta <= 0. {
        return Ok((excess.constant(0.), excess.constant(0.)));
    }
    if excess.compare(beta / EXP_LIMIT) == Some(Ordering::Greater) {
        let argument = excess.constant(-beta).divide(excess)?;
        let coefficient = argument.clone().exp().scale(alpha);
        let slope = coefficient
            .clone()
            .multiply(&argument.scale(-1.).offset(1.));
        Ok((coefficient.multiply(excess), slope))
    } else {
        let slope = alpha * MIN_EXP;
        Ok((excess.clone().scale(slope), excess.constant(slope)))
    }
}

/// Early voltage due to substrate-current-induced body effect. The
/// caller supplies the version-specific enable predicate.
#[inline]
pub(super) fn substrate_early_voltage<S: Constitutive>(
    excess: &S,
    length: Real,
    threshold: Real,
    scale: Real,
    enabled: bool,
) -> SpiceResult<(S, S)> {
    if !enabled {
        return Ok((excess.constant(MAX_EXP), excess.constant(0.)));
    }
    if excess.compare(threshold / EXP_LIMIT) == Some(Ordering::Greater) {
        let argument = excess.constant(threshold).divide(excess)?;
        let value = argument.clone().exp().scale(length / scale);
        let slope = argument.scale(-1.).divide(excess)?.multiply(&value);
        Ok((value, slope))
    } else {
        Ok((
            excess.constant(MAX_EXP * length / scale),
            excess.constant(0.),
        ))
    }
}

/// Channel conductance before the series-resistance correction. Independent
/// coordinates are overdrive, bulk factor, effective drain, saturation field
/// times length, and mobility/oxide/geometry coefficient.
#[inline]
pub(super) fn channel_conductance<S: Constitutive>(
    overdrive: &S,
    bulk: &S,
    drain: &S,
    field: &S,
    beta: &S,
    thermal: Real,
) -> SpiceResult<LocalPart<S, 5>> {
    let thermal_overdrive = overdrive.clone().offset(2. * thermal);
    let half_drain = drain.clone().scale(0.5).divide(&thermal_overdrive)?;
    let edge = bulk
        .clone()
        .scale(-1.)
        .mul_add(&half_drain, &overdrive.constant(1.));
    let field_ratio = drain.clone().divide(field)?;
    let denominator = field_ratio.clone().offset(1.);
    let normalized = overdrive.clone().divide(&denominator)?;
    let gain = beta.clone().multiply(&normalized);
    let value = gain.clone().multiply(&edge);
    let gate_partial = beta.clone().divide(&denominator)?.multiply(
        &bulk
            .clone()
            .multiply(&half_drain)
            .multiply(&overdrive.clone().divide(&thermal_overdrive)?)
            .add(&edge),
    );
    let bulk_partial = gain.clone().scale(-1.).multiply(&half_drain);
    let drain_partial = gain
        .scale(-0.5)
        .multiply(bulk)
        .divide(&thermal_overdrive)?
        .add(
            &value
                .clone()
                .scale(-1.)
                .divide(field)?
                .divide(&denominator)?,
        );
    let field_partial = value
        .clone()
        .multiply(&field_ratio)
        .divide(field)?
        .divide(&denominator)?;
    let beta_partial = normalized.multiply(&edge);
    Ok(LocalPart {
        value,
        partials: [
            gate_partial,
            bulk_partial,
            drain_partial,
            field_partial,
            beta_partial,
        ],
    })
}

/// Conductance loaded by the intrinsic source/drain series resistance.
#[inline]
pub(super) fn series_conductance<S: Constitutive>(
    conductance: &S,
    resistance: &S,
) -> SpiceResult<LocalPart<S, 2>> {
    let denominator = conductance
        .clone()
        .mul_add(resistance, &conductance.constant(1.));
    let inverse = conductance.constant(1.).divide(&denominator)?;
    let value = conductance.clone().multiply(&inverse);
    let conductance_partial = inverse.clone().multiply(&inverse);
    let resistance_partial = value.clone().scale(-1.).multiply(&value);
    Ok(LocalPart {
        value,
        partials: [conductance_partial, resistance_partial],
    })
}

/// Smaller root of a*y²/2 - b*y + c = 0, regular in the linear limit.
pub(super) struct SaturationRoot<S> {
    pub(super) value: S,
    discriminant: S,
}

impl<S: Constitutive> SaturationRoot<S> {
    pub(super) fn new(a: &S, b: S, c: S) -> SpiceResult<Self> {
        let discriminant = a
            .clone()
            .scale(-2.)
            .mul_add(&c, &b.clone().multiply(&b))
            .sqrt()?;
        let value = if b.primal() > 0. {
            c.divide(&b.add(&discriminant))?.scale(2.)
        } else {
            b.add(&discriminant.clone().scale(-1.)).divide(a)?
        };
        Ok(Self {
            value,
            discriminant,
        })
    }

    #[cfg(test)]
    pub(super) fn derivative(&self, da: &S, db: &S, dc: &S) -> SpiceResult<S> {
        let y = &self.value;
        y.clone()
            .scale(0.5)
            .multiply(y)
            .mul_add(da, &y.clone().scale(-1.).mul_add(db, dc))
            .divide(&self.discriminant)
    }
}

/// Saturation voltage and implicit local partials. Inputs are the thermal
/// overdrive, bulk factor, saturation field times length, resistance-width
/// product and saturation shape. The rationalized root includes the
/// zero-resistance/unit-shape linear limit without a separate model branch.
#[inline]
pub(super) fn saturation_voltage<S: Constitutive>(
    drive: &S,
    bulk: &S,
    field: &S,
    resistance: &S,
    shape: &S,
) -> SpiceResult<LocalPart<S, 5>> {
    let one = drive.constant(1.);
    let shape_delta = shape.clone().scale(-1.).offset(1.).divide(shape)?;
    let bulk_resistance = bulk.clone().multiply(resistance);
    let inverse_shape = one.divide(shape)?;
    let shape_term = inverse_shape.clone().scale(2.).offset(-1.);
    let mixed = bulk_resistance.clone().scale(3.).add(&shape_term);
    let quadratic = bulk
        .clone()
        .scale(2.)
        .multiply(&bulk_resistance.clone().add(&shape_delta));
    let linear = drive.clone().mul_add(&mixed, &bulk.clone().multiply(field));
    let drive_resistance = drive.clone().multiply(resistance);
    let constant = drive
        .clone()
        .multiply(&drive_resistance.clone().scale(2.).add(field));
    let root = SaturationRoot::new(&quadratic, linear, constant)?;
    let y = &root.value;
    let gap = bulk.clone().scale(-1.).mul_add(y, drive);
    let double_gap = gap.clone().add(drive);
    let drive_partial = y
        .clone()
        .scale(-1.)
        .mul_add(&mixed, &drive_resistance.clone().scale(4.).add(field));
    let bulk_partial = y.clone().multiply(&y.clone().mul_add(
        &bulk_resistance.scale(2.).add(&shape_delta),
        &drive_resistance.scale(-3.).add(&field.clone().scale(-1.)),
    ));
    let resistance_partial = gap.clone().multiply(&double_gap);
    let shape_partial = y
        .clone()
        .multiply(&double_gap)
        .multiply(&inverse_shape.clone().multiply(&inverse_shape));
    Ok(LocalPart {
        partials: [
            drive_partial.divide(&root.discriminant)?,
            bulk_partial.divide(&root.discriminant)?,
            gap.divide(&root.discriminant)?,
            resistance_partial.divide(&root.discriminant)?,
            shape_partial.divide(&root.discriminant)?,
        ],
        value: root.value,
    })
}

/// Saturation Early voltage with local partials in overdrive, bulk factor,
/// saturation voltage, field times length, resistance-width product and
/// saturation shape.
#[inline]
pub(super) fn saturation_early_voltage<S: Constitutive>(
    overdrive: &S,
    bulk: &S,
    saturation: &S,
    field: &S,
    resistance: &S,
    shape: &S,
    thermal: Real,
) -> SpiceResult<LocalPart<S, 6>> {
    let drive = overdrive.clone().offset(2. * thermal);
    let half_saturation = saturation.clone().scale(0.5).divide(&drive)?;
    let edge = bulk
        .clone()
        .scale(-1.)
        .mul_add(&half_saturation, &overdrive.constant(1.));
    let inverse_shape = shape.constant(1.).divide(shape)?;
    let denominator = bulk
        .clone()
        .mul_add(resistance, &inverse_shape.clone().scale(2.).offset(-1.));
    let product = resistance.clone().multiply(overdrive);
    let numerator = product
        .clone()
        .scale(2.)
        .mul_add(&edge, &field.clone().add(saturation));
    let value = numerator.divide(&denominator)?;
    let drive_partial = bulk
        .clone()
        .multiply(&half_saturation)
        .multiply(&overdrive.clone().divide(&drive)?)
        .add(&edge)
        .multiply(&resistance.clone().scale(2.));
    let bulk_partial = product.clone().scale(-2.).mul_add(
        &half_saturation,
        &value.clone().scale(-1.).multiply(resistance),
    );
    let saturation_partial = product.scale(-1.).multiply(bulk).divide(&drive)?.offset(1.);
    let field_partial = overdrive.constant(1.);
    let resistance_partial = overdrive
        .clone()
        .scale(2.)
        .mul_add(&edge, &value.clone().scale(-1.).multiply(bulk));
    let shape_partial = value
        .clone()
        .scale(2.)
        .multiply(&inverse_shape.clone().multiply(&inverse_shape));
    Ok(LocalPart {
        value,
        partials: [
            drive_partial.divide(&denominator)?,
            bulk_partial.divide(&denominator)?,
            saturation_partial.divide(&denominator)?,
            field_partial.divide(&denominator)?,
            resistance_partial.divide(&denominator)?,
            shape_partial.divide(&denominator)?,
        ],
    })
}

/// Drain-induced barrier-lowering Early voltage with exact partials in the
/// thermal overdrive, bulk factor, saturation voltage and effective body
/// bias. The caller retains the model generation's enable predicate.
#[inline]
pub(super) fn dibl_early_voltage<S: Constitutive>(
    drive: &S,
    bulk: &S,
    saturation: &S,
    body: &S,
    theta: Real,
    body_coefficient: Real,
) -> SpiceResult<LocalPart<S, 4>> {
    let denominator = bulk.clone().mul_add(saturation, drive);
    let ratio = drive.clone().divide(&denominator)?;
    let base = ratio.clone().multiply(drive).scale(theta.recip());
    let (body_factor, slope) = reciprocal_floor(body.clone().scale(body_coefficient), -0.9, 20.)?;
    let scaled = body_factor.clone().scale(theta.recip());
    let derivative_factor = ratio.clone().multiply(&scaled);
    let drain_factor = derivative_factor.clone().scale(-1.).multiply(&ratio);
    let drive_partial = ratio.scale(-1.).offset(2.).multiply(&derivative_factor);
    let bulk_partial = drain_factor.clone().multiply(saturation);
    let saturation_partial = drain_factor.multiply(bulk);
    let body_partial = base.clone().multiply(&slope).scale(body_coefficient);
    let value = base.multiply(&body_factor);
    Ok(LocalPart {
        value,
        partials: [
            drive_partial,
            bulk_partial,
            saturation_partial,
            body_partial,
        ],
    })
}

/// Charge-thickness effective overdrive and its gate partial. The smooth
/// positive root also retains the dependence of its width on overdrive.
pub(super) fn charge_overdrive<S: Constitutive>(
    overdrive: S,
    k1ox: Real,
    moin: Real,
    thermal: Real,
    sqrt_phi: Real,
) -> SpiceResult<(S, S)> {
    let (denominator, surface) = if k1ox <= 0. {
        (0.25 * moin * thermal, 0.5 * sqrt_phi)
    } else {
        (moin * thermal * k1ox * k1ox, k1ox * sqrt_phi)
    };
    let product = overdrive.clone().offset(2. * surface).multiply(&overdrive);
    let delta_phi = product
        .clone()
        .divide(&overdrive.constant(denominator))?
        .ln_1p()?
        .scale(thermal);
    let delta_phi_slope = overdrive
        .clone()
        .offset(surface)
        .scale(2. * thermal)
        .divide(&product.offset(denominator))?;
    let argument = overdrive.clone().add(&delta_phi.scale(-1.)).offset(-0.001);
    let root = argument
        .clone()
        .mul_add(&argument, &overdrive.clone().scale(0.004))
        .sqrt()?;
    let value = positive_half_sum(argument, root.clone(), overdrive.scale(0.001))?;
    let slope = delta_phi_slope
        .scale(-1.)
        .offset(1.)
        .mul_add(&value, &value.constant(0.001))
        .divide(&root)?;
    Ok((value, slope))
}

/// Normalized bulk depletion charge, with its differential capacitance.
/// The negative-potential continuation is linear and the positive root
/// is rationalized to retain very small nonzero charges.
#[inline]
pub(super) fn depletion_charge<S: Constitutive>(k1: Real, increment: S) -> SpiceResult<(S, S)> {
    if k1 == 0. {
        return Ok((increment.constant(0.), increment.constant(0.)));
    }
    if increment.compare(0.) == Some(Ordering::Less) {
        let slope = increment.constant(1.);
        return Ok((increment, slope));
    }
    let base = 0.5 * k1;
    // Use the fused primal through the common arithmetic, not a separately rounded base square.
    let root = increment
        .constant(base)
        .mul_add(&increment.constant(base), &increment)
        .sqrt()?;
    let slope = increment.constant(base).divide(&root)?;
    let difference = if base > 0. {
        increment.divide(&root.offset(base))?
    } else {
        root.offset(-base)
    };
    Ok((difference.scale(k1), slope))
}

/// Constitutive value and analytic partials in the local input coordinates.
pub(super) struct LocalPart<S, const N: usize = 3> {
    pub(super) value: S,
    partials: [S; N],
}

/// Conservative channel inversion, bulk and source charges in reference
/// orientation, before accumulation/depletion, overlap or junction terms.
/// The 50/50 source fraction is half the inversion charge, not half the
/// gate charge. The normalized ratios retain the regularizing 1e-20 in both
/// the values and the exact derivatives.
pub(super) fn channel_inversion<S: Constitutive>(
    overdrive: &S,
    drain: &S,
    bulk: &S,
    oxide_area: &S,
    partition: Real,
    regularization: Real,
) -> SpiceResult<[LocalPart<S, 4>; 3]> {
    let one = overdrive.constant(1.);
    let weighted_drain = bulk.clone().multiply(drain);
    let width = weighted_drain
        .clone()
        .scale(-0.5)
        .add(overdrive)
        .offset(regularization);
    let ratio = weighted_drain.clone().divide(&width)?;
    let normalized_sum = overdrive.clone().offset(regularization).divide(&width)?;
    let fraction = ratio.clone().scale(1. / 12.);
    let edge = fraction.clone().scale(-1.).offset(0.5);
    // weighted_drain * d(fraction)/d(weighted_drain).
    let weighted_fraction_partial = fraction.clone().multiply(&normalized_sum);
    let inversion_gate_partial = ratio.clone().scale(-1.).mul_add(&fraction, &one);
    let inversion_weighted_partial = weighted_fraction_partial
        .clone()
        .add(&edge.clone().scale(-1.));
    let inversion_value = weighted_drain.clone().scale(-1.).mul_add(&edge, overdrive);
    let part = |value: S, du: S, dd: S, da: S| LocalPart {
        value: value.clone().multiply(oxide_area),
        partials: [
            du.multiply(oxide_area),
            dd.multiply(oxide_area),
            da.multiply(oxide_area),
            value,
        ],
    };
    let inversion = part(
        inversion_value.clone(),
        inversion_gate_partial.clone(),
        inversion_weighted_partial.clone().multiply(bulk),
        inversion_weighted_partial.clone().multiply(drain),
    );
    let complement = bulk.clone().scale(-1.).offset(1.);
    let normalized_drain = drain.clone().divide(&width)?;
    let bulk_gate_partial = complement
        .clone()
        .multiply(&normalized_drain)
        .multiply(&fraction);
    let bulk_drain_partial = complement
        .clone()
        .multiply(&inversion_weighted_partial.clone().scale(-1.));
    let bulk_factor_partial = complement
        .clone()
        .multiply(&normalized_drain)
        .multiply(&normalized_sum)
        .scale(1. / 12.)
        .add(&edge)
        .multiply(drain)
        .scale(-1.);
    let bulk_charge = part(
        complement.multiply(drain).multiply(&edge),
        bulk_gate_partial,
        bulk_drain_partial,
        bulk_factor_partial,
    );
    let (source_value, source_gate_partial, source_weighted_partial) = if partition > 0.5 {
        (
            weighted_drain.mul_add(
                &fraction.clone().scale(0.5).offset(-0.25),
                &overdrive.clone().scale(-0.5),
            ),
            ratio
                .scale(-0.5)
                .mul_add(&fraction, &overdrive.constant(-0.5)),
            fraction
                .clone()
                .add(&weighted_fraction_partial)
                .scale(0.5)
                .offset(-0.25),
        )
    } else if partition < 0.5 {
        partition_40_60(overdrive, &weighted_drain, &ratio, &width)?
    } else {
        (
            inversion_value.scale(-0.5),
            inversion_gate_partial.scale(-0.5),
            inversion_weighted_partial.scale(-0.5),
        )
    };
    let source = part(
        source_value,
        source_gate_partial,
        source_weighted_partial.clone().multiply(bulk),
        source_weighted_partial.multiply(drain),
    );
    Ok([inversion, bulk_charge, source])
}

/// Independent local coordinates of the abrupt charge model, in the
/// reference source-oriented frame. The surface term is `k1ox*sqrt(phi_s)`.
pub(super) struct CapZeroBias<S> {
    pub(super) gate: S,
    pub(super) body_cv: S,
    pub(super) surface: S,
    pub(super) drain: S,
    pub(super) bulk_factor: S,
}

/// Prepared, bias-independent parameters of the abrupt C-V model.
#[derive(Clone, Copy)]
pub(super) struct CapZeroParameters {
    pub(super) flatband: Real,
    pub(super) phi: Real,
    pub(super) body_effect: Real,
    pub(super) oxide_area: Real,
    pub(super) partition: Real,
}

impl<S: Constitutive, const N: usize> LocalPart<S, N> {
    fn scaled(&self, factor: Real) -> Self {
        Self {
            value: self.value.clone().scale(factor),
            partials: self.partials.clone().map(|partial| partial.scale(factor)),
        }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            value: self.value.clone().add(&other.value),
            partials: std::array::from_fn(|i| self.partials[i].clone().add(&other.partials[i])),
        }
    }
}

impl<S: Constitutive, const N: usize> LocalPart<S, N> {
    /// Project the independent constitutive partials onto terminal biases.
    pub(super) fn project<const M: usize>(&self, coordinates: &[[Real; M]; N]) -> [Real; M] {
        std::array::from_fn(|column| {
            self.partials
                .iter()
                .zip(coordinates)
                .fold(0., |sum, (partial, direction)| {
                    partial.primal().mul_add(direction[column], sum)
                })
        })
    }
}

impl<S: Constitutive, const N: usize> LocalPart<S, N> {
    /// Compose reference-local first derivatives with an outer coordinate
    /// basis while retaining the shared constitutive value unchanged.
    #[inline]
    pub(super) fn into_local<const M: usize>(
        self,
        inputs: [&crate::maths::taylor::Local<S, M>; N],
    ) -> crate::maths::taylor::Local<S, M> {
        let partials = self.project(&inputs.map(|input| input.partials));
        crate::maths::taylor::Local::new(self.value, partials)
    }
}

/// BSIM4 capMod=0: accumulation, depletion, inversion in the linear
/// region, and saturated inversion. The latter is the exact endpoint of
/// the same Ward-Dutton law, with no numerical 1e-20 regularization.
/// Returns gate, bulk and drain charges and all five local partials.
pub(super) fn cap_zero<S: Constitutive>(
    bias: &CapZeroBias<S>,
    parameters: CapZeroParameters,
) -> SpiceResult<[LocalPart<S, 5>; 3]> {
    let CapZeroParameters {
        flatband,
        phi,
        body_effect,
        oxide_area,
        partition,
    } = parameters;
    let zero = bias.gate.constant(0.);
    let accumulation = bias
        .gate
        .clone()
        .add(&bias.body_cv.clone().scale(-1.))
        .offset(-flatband);
    let overdrive = bias
        .gate
        .clone()
        .add(&bias.surface.clone().offset(phi + flatband).scale(-1.));
    let accumulating = accumulation
        .compare(0.)
        .is_some_and(|order| order != Ordering::Greater);
    if accumulating
        || overdrive
            .compare(0.)
            .is_some_and(|order| order != Ordering::Greater)
    {
        let (charge, slope) = if accumulating {
            (accumulation, bias.gate.constant(1.))
        } else {
            depletion_charge(body_effect, accumulation)?
        };
        let gate = LocalPart {
            value: charge.scale(oxide_area),
            partials: [
                slope.clone().scale(oxide_area),
                slope.scale(-oxide_area),
                zero.clone(),
                zero.clone(),
                zero.clone(),
            ],
        };
        let bulk = gate.scaled(-1.);
        let drain = LocalPart {
            value: zero.clone(),
            partials: std::array::from_fn(|_| zero.clone()),
        };
        return Ok([gate, bulk, drain]);
    }

    let saturation = overdrive.clone().divide(&bias.bulk_factor)?;
    let gap = bias
        .bulk_factor
        .clone()
        .scale(-1.)
        .mul_add(&bias.drain, &overdrive);
    if gap.compare(0.) != Some(Ordering::Greater) {
        return cap_zero_saturation(bias, parameters, &overdrive, &saturation);
    }

    let [inversion, bulk, source] = channel_inversion(
        &overdrive,
        &bias.drain,
        &bias.bulk_factor,
        &bias.gate.constant(oxide_area),
        partition,
        0.,
    )?;
    let project = |part: LocalPart<S, 4>| {
        let [gate, drain, factor, _oxide] = part.partials;
        let surface = gate.clone().scale(-1.);
        LocalPart {
            value: part.value,
            partials: [gate, zero.clone(), surface, drain, factor],
        }
    };
    let inversion = project(inversion);
    let bulk = project(bulk);
    let source = project(source);
    let surface = LocalPart {
        value: bias.surface.clone().scale(oxide_area),
        partials: [
            zero.clone(),
            zero.clone(),
            bias.gate.constant(oxide_area),
            zero.clone(),
            zero,
        ],
    };
    let bulk = bulk.add(&surface.scaled(-1.));
    let gate = inversion.add(&bulk.scaled(-1.));
    let drain = if partition > 0.5 {
        // For 0/100 partition the drain charge vanishes quadratically
        // at saturation. Compute that small charge directly instead of
        // subtracting the almost equal source and inversion charges.
        cap_zero_drain_tail(bias, &overdrive, gap, oxide_area)?
    } else {
        inversion.add(&source).scaled(-1.)
    };
    Ok([gate, bulk, drain])
}

/// Rationalized 0/100 drain charge in the linear region.
fn cap_zero_drain_tail<S: Constitutive>(
    bias: &CapZeroBias<S>,
    overdrive: &S,
    gap: S,
    oxide_area: Real,
) -> SpiceResult<LocalPart<S, 5>> {
    let width = gap.clone().midpoint(overdrive);
    let ratio = gap.clone().divide(&width)?;
    let value = gap.scale(-0.5 * oxide_area).multiply(&ratio);
    let gate_partial = ratio
        .clone()
        .scale(0.5)
        .offset(-1.)
        .multiply(&ratio)
        .scale(oxide_area);
    let weighted_partial = ratio
        .clone()
        .scale(-0.25)
        .offset(1.)
        .multiply(&ratio)
        .scale(oxide_area);
    Ok(LocalPart {
        value,
        partials: [
            gate_partial.clone(),
            overdrive.constant(0.),
            gate_partial.scale(-1.),
            weighted_partial.clone().multiply(&bias.bulk_factor),
            weighted_partial.multiply(&bias.drain),
        ],
    })
}

/// The saturation endpoint is evaluated directly: the 0/100 drain
/// charge and its derivatives are identically zero, not a subtraction
/// between nearly equal inversion and source charges.
fn cap_zero_saturation<S: Constitutive>(
    bias: &CapZeroBias<S>,
    parameters: CapZeroParameters,
    overdrive: &S,
    saturation: &S,
) -> SpiceResult<[LocalPart<S, 5>; 3]> {
    let CapZeroParameters {
        flatband,
        phi,
        oxide_area,
        partition,
        ..
    } = parameters;
    let zero = bias.gate.constant(0.);
    let third = oxide_area / 3.;
    let two_thirds = 2. * third;
    let saturation_slope = bias.gate.constant(third).divide(&bias.bulk_factor)?;
    let gate = LocalPart {
        value: saturation.clone().scale(-third).add(
            &bias
                .gate
                .clone()
                .offset(-flatband)
                .offset(-phi)
                .scale(oxide_area),
        ),
        partials: [
            saturation_slope.clone().scale(-1.).offset(oxide_area),
            zero.clone(),
            saturation_slope.clone(),
            zero.clone(),
            saturation_slope.multiply(saturation),
        ],
    };
    let inversion = LocalPart {
        value: overdrive.clone().scale(two_thirds),
        partials: [
            bias.gate.constant(two_thirds),
            zero.clone(),
            bias.gate.constant(-two_thirds),
            zero.clone(),
            zero,
        ],
    };
    let bulk = inversion.add(&gate.scaled(-1.));
    let drain_fraction = if partition > 0.5 {
        0.
    } else if partition < 0.5 {
        0.4
    } else {
        0.5
    };
    let drain = inversion.scaled(-drain_fraction);
    Ok([gate, bulk, drain])
}

/// Evaluate the Ward-Dutton polynomial and its overdrive/weighted-drain partials.
fn partition_40_60<S: Constitutive>(
    overdrive: &S,
    weighted_drain: &S,
    ratio: &S,
    width: &S,
) -> SpiceResult<(S, S, S)> {
    // Evaluate the Ward-Dutton cubic divided by width² without forming
    // powers of a small dimensional voltage in numerator/denominator.
    let normalized_gate = overdrive.clone().divide(width)?;
    let ratio_squared = ratio.clone().multiply(ratio);
    let gate_squared = normalized_gate.clone().multiply(&normalized_gate);
    let mixed = normalized_gate.multiply(ratio);
    let polynomial = ratio_squared
        .clone()
        .scale(2. / 3.)
        .add(&gate_squared)
        .add(&mixed.clone().scale(-4. / 3.));
    let value = overdrive
        .clone()
        .mul_add(
            &polynomial,
            &weighted_drain
                .clone()
                .multiply(&ratio_squared)
                .scale(-2. / 15.),
        )
        .scale(-0.5);
    let relative = value.clone().divide(width)?;
    let du = gate_squared
        .clone()
        .scale(3.)
        .add(&mixed.clone().scale(-8. / 3.))
        .add(&ratio_squared.clone().scale(2. / 3.))
        .scale(-0.5)
        .add(&relative.clone().scale(-2.));
    let dt = gate_squared
        .scale(-4. / 3.)
        .add(&mixed.scale(4. / 3.))
        .add(&ratio_squared.scale(-0.4))
        .scale(-0.5)
        .add(&relative);
    Ok((value, du, dt))
}

/// Effective gate voltage and local gate derivative after poly-Si depletion.
/// BSIM3 and BSIM4 use the same law, but their reference electron-charge
/// constants differ. The prepared scale preserves that distinction.
pub(super) fn poly_depletion<S: Constitutive>(
    voltage: S,
    threshold: Real,
    doping: Real,
    permittivity: Real,
    oxide_capacitance: Real,
    electron_charge: Real,
) -> SpiceResult<(S, S)> {
    if !(1e18 < doping && doping < 1e25 && permittivity != 0.)
        || voltage.compare(threshold) != Some(Ordering::Greater)
    {
        let slope = voltage.constant(1.);
        return Ok((voltage, slope));
    }
    let scale =
        1e6 * electron_charge * permittivity * doping / (oxide_capacitance * oxide_capacitance);
    let overdrive = voltage.clone().offset(-threshold);
    let root = overdrive
        .clone()
        .scale(2.)
        .divide(&voltage.constant(scale))?
        .offset(1.)
        .sqrt()?;
    // Rationalize the two small square-root differences; the gate slope
    // must not subtract nearly equal halves near the depletion threshold.
    let gate_drop = overdrive.scale(2.).divide(&root.clone().offset(1.))?;
    let poly = gate_drop
        .clone()
        .scale(0.5)
        .multiply(&gate_drop)
        .divide(&voltage.constant(scale))?;
    let barrier = poly.clone().scale(-1.).offset(1.07);
    let band_root = barrier
        .clone()
        .mul_add(&barrier, &voltage.constant(0.224))
        .sqrt()?;
    let band_drop = poly
        .scale(2.24)
        .divide(&barrier.clone().scale(-1.).offset(2.24).add(&band_root))?;
    let slope = gate_drop.scale(-0.5).divide(&root.scale(scale))?.mul_add(
        &barrier.divide(&band_root)?.offset(1.),
        &voltage.constant(1.),
    );
    Ok((voltage.add(&band_drop.scale(-1.)), slope))
}

/// BSIM3's prepared diode saturation and optional linear forward tail.
/// Returns the actual static residual and its differential conductance.
pub(super) fn bsim3_junction<S: Constitutive>(
    voltage: S,
    saturation: Real,
    thermal: Real,
    continuation: Option<(Real, Real)>,
    gmin: Real,
) -> SpiceResult<(S, S)> {
    if saturation <= 0. {
        let conductance = voltage.constant(gmin);
        return Ok((voltage.scale(gmin), conductance));
    }
    if let Some((edge, current)) = continuation
        && voltage.compare(edge) != Some(Ordering::Less)
    {
        let slope = current / thermal;
        let intrinsic = voltage.constant(slope).mul_add(
            &voltage.clone().offset(-edge),
            &voltage.constant(current - saturation),
        );
        let conductance = voltage.constant(slope + gmin);
        return Ok((
            voltage.constant(gmin).mul_add(&voltage, &intrinsic),
            conductance,
        ));
    }
    let argument = voltage.clone().divide(&voltage.constant(thermal))?;
    // This is a cancellation-avoidance choice within one exponential law.
    let (ev, excess) = if argument.primal().abs() <= std::f64::consts::LN_2 {
        let excess = argument.exp_m1();
        (excess.clone().offset(1.), excess)
    } else {
        let ev = argument.exp();
        (ev.clone(), ev.offset(-1.))
    };
    let conductance = ev
        .scale(saturation)
        .divide(&voltage.constant(thermal))?
        .offset(gmin);
    let current = voltage
        .constant(gmin)
        .mul_add(&voltage, &excess.scale(saturation));
    Ok((current, conductance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::maths::taylor::{Jet, differential::Differential};

    const ORDER: usize = 8;
    type Law = Differential<1>;

    fn close(actual: Real, expected: Real, absolute: Real) {
        assert!(
            (actual - expected).abs() <= absolute + 4e-12 * expected.abs(),
            "{actual:.17e} != {expected:.17e}"
        );
    }

    fn variable(bias: Real, rate: Real) -> SpiceResult<Law> {
        Law::variable(0, Jet::affine(ORDER, bias, rate))
    }

    fn value_at(jet: &Jet, time: Real) -> Real {
        jet.coefficients()
            .iter()
            .rev()
            .fold(0., |value, coefficient| value.mul_add(time, *coefficient))
    }

    fn check_charge_gradient(charge: &Law, cap: &Law) {
        let gradient = charge.gradient[0].as_ref().unwrap();
        for (&actual, &expected) in gradient.coefficients().iter().zip(cap.value.coefficients()) {
            close(actual, expected, 2e-26);
        }
    }

    #[test]
    fn body_limits_and_surface_potential_match_reference_and_directional_derivatives()
    -> SpiceResult<()> {
        let cutoff: Real = -3.;
        let phi: Real = 0.75;
        let phi_three_halves = phi * f64::sqrt(phi);
        for bias in [
            -30.0_f64, -3., -2.999, -0.5, -1e-16, 0., 1e-16, 0.1, 0.7125, 1.5,
        ] {
            for direction in [-1., 1.] {
                let rate = if bias.abs() > 0. && bias.abs() < 1e-12 {
                    0.1 * bias.abs() * direction
                } else {
                    0.0001 * direction
                };
                let input = variable(bias, rate)?;
                let (lower, lower_partial) = body_lower_limit(&input, cutoff)?;
                check_charge_gradient(&lower, &lower_partial);
                let (bsim3, reference_partial) = bsim3_body_limit(&input, cutoff)?;
                let (bsim4, exact_partial) = bsim4_body_limit(&input, cutoff, phi)?;
                check_charge_gradient(&bsim4, &exact_partial);
                if input.value.sign() == Ordering::Greater {
                    let actual = bsim3.gradient[0].as_ref().unwrap();
                    close(actual.value(), 1., 1e-15);
                } else {
                    check_charge_gradient(&bsim3, &reference_partial);
                }
                for (limited, approximation) in [(&bsim3, Some(phi_three_halves)), (&bsim4, None)] {
                    let body = Law::variable(0, limited.value.clone())?;
                    let surface = surface_potential(&body, phi, approximation)?;
                    check_charge_gradient(&surface.potential, &surface.potential_partial);
                    check_charge_gradient(&surface.root, &surface.root_partial);
                    for time in [0., 0.001, 0.02] {
                        let b = value_at(&body.value, time);
                        let (potential, root) = if approximation.is_some() && b > 0. {
                            (
                                phi * phi / (phi + b),
                                phi_three_halves / 0.5_f64.mul_add(b, phi),
                            )
                        } else {
                            (phi - b, (phi - b).sqrt())
                        };
                        close(value_at(&surface.potential.value, time), potential, 1e-14);
                        close(value_at(&surface.root.value, time), root, 1e-14);
                    }
                }
                for time in [0., 0.001, 0.02] {
                    let b = rate.mul_add(time, bias);
                    let x = b - cutoff - 0.001;
                    let raw = cutoff + x.midpoint(x.mul_add(x, -0.004 * cutoff).sqrt());
                    let upper = 0.95 * phi;
                    let x = upper - raw - 0.001;
                    let limited = upper - x.midpoint(x.mul_add(x, 0.004 * upper).sqrt());
                    close(value_at(&lower.value, time), raw, 3e-15);
                    close(value_at(&bsim3.value, time), raw.max(b), 3e-15);
                    close(value_at(&bsim4.value, time), limited, 3e-15);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn threshold_rolloff_and_bsim3_overdrive_match_independent_equations() -> SpiceResult<()> {
        let cases: [(ThresholdRollOff, [Real; 7]); 2] = [
            (
                ThresholdRollOff::Bsim3,
                [-45., -34., -10., -1., 0., 1., 10.],
            ),
            (
                ThresholdRollOff::Bsim4,
                [-10., -1., -1e-8, 0., 1e-8, 1., 34.],
            ),
        ];
        for (generation, arguments) in cases {
            for bias in arguments {
                let rate = if matches!(generation, ThresholdRollOff::Bsim4) && bias.abs() < 1e-6 {
                    1e-11
                } else {
                    0.001
                };
                for direction in [-1., 1.] {
                    let slope = rate * direction;
                    let input = variable(bias, slope)?;
                    let (value, partial) = generation.evaluate(&input)?;
                    for (degree, &expected) in partial.value.coefficients().iter().enumerate() {
                        let actual = value.gradient[0]
                            .as_ref()
                            .map_or(0., |g| g.coefficients()[degree]);
                        close(actual, expected, 1e-10 * value.value.value().abs().max(1.));
                    }
                    for time in [0.001, 0.02] {
                        let z = slope.mul_add(time, bias);
                        let reference = match generation {
                            ThresholdRollOff::Bsim3 => {
                                let exponential = if z <= -34. { MIN_EXP } else { z.exp() };
                                exponential * 2.0_f64.mul_add(exponential, 1.)
                            }
                            ThresholdRollOff::Bsim4 => {
                                if z >= 34. {
                                    1. / (MAX_EXP - 2.)
                                } else {
                                    let e = z.exp();
                                    z.exp_m1().mul_add(z.exp_m1(), 2. * MIN_EXP * e).recip() * e
                                }
                            }
                        };
                        close(value_at(&value.value, time), reference, 1e-20);
                    }
                }
            }
        }
        type Local = Differential<2>;
        let thermal = 0.025;
        let oxide_ratio = 1.8;
        for offset in [-2., -0.1, 0.2] {
            for bias in [-5., -2., -0.5, 0., 0.5, 2., 4.] {
                let factor = 1.4;
                let rates = [0.001, 0.0001];
                let v = Local::variable(0, Jet::affine(ORDER, bias, rates[0]))?;
                let n = Local::variable(1, Jet::affine(ORDER, factor, rates[1]))?;
                let (value, voltage_partial, factor_partial) =
                    bsim3_overdrive(&v, &n, thermal, offset, oxide_ratio)?;
                for (coordinate, partial) in [voltage_partial, factor_partial].iter().enumerate() {
                    for (degree, &expected) in partial.value.coefficients().iter().enumerate() {
                        let actual = value.gradient[coordinate]
                            .as_ref()
                            .map_or(0., |g| g.coefficients()[degree]);
                        close(actual, expected, 1e-20);
                    }
                }
                for time in [0., 0.001, 0.02] {
                    let v = rates[0].mul_add(time, bias);
                    let n = rates[1].mul_add(time, factor);
                    let width = 2. * n * thermal;
                    let numerator_argument = v / width;
                    let denominator_argument = 2.0_f64.mul_add(offset, -v) / width;
                    let reference = if numerator_argument > 34. {
                        v
                    } else if denominator_argument > 34. {
                        thermal / oxide_ratio * ((v - offset) / (n * thermal)).exp()
                    } else {
                        width * numerator_argument.exp().ln_1p()
                            / (2. * n * oxide_ratio).mul_add(denominator_argument.exp(), 1.)
                    };
                    close(value_at(&value.value, time), reference, 1e-30);
                }
            }
        }
        Ok(())
    }

    fn verify_local<const N: usize>(part: &LocalPart<Differential<N>, N>) {
        for (coordinate, partial) in part.partials.iter().enumerate() {
            for (degree, &expected) in partial.value.coefficients().iter().enumerate() {
                let actual = part.value.gradient[coordinate]
                    .as_ref()
                    .map_or(0., |g| g.coefficients()[degree]);
                close(actual, expected, 1e-18);
            }
        }
    }

    #[test]
    fn shared_threshold_and_slope_factors_match_reference_trajectories() -> SpiceResult<()> {
        for body in [-3.1, -0.3, 0.1] {
            let depth = 1e-7;
            let b = Differential::<2>::variable(0, Jet::affine(ORDER, body, 0.003))?;
            let d = Differential::<2>::variable(1, Jet::affine(ORDER, depth, 1e-10))?;
            for (generation, scale) in [
                (ThresholdRollOff::Bsim3, -1e-6),
                (ThresholdRollOff::Bsim4, 1e-6),
            ] {
                let rolloff = channel_rolloff(&b, &d, 0.2, 0.003, scale, generation)?;
                verify_local(&rolloff);
                for t in [0., 0.01, 0.02] {
                    let x = 0.2 * 0.003_f64.mul_add(t, body);
                    let factor = if x < -0.5 {
                        (1. + 3. * x) / (3. + 8. * x)
                    } else {
                        1. + x
                    };
                    let arg = scale / (0.003 * 1e-10_f64.mul_add(t, depth).sqrt() * factor);
                    let e = arg.exp();
                    let expected = match generation {
                        ThresholdRollOff::Bsim3 => e * (1. + 2. * e),
                        ThresholdRollOff::Bsim4 => e / ((e - 1.).powi(2) + 2. * e * MIN_EXP),
                    };
                    close(value_at(&rolloff.value.value, t), expected, 1e-15);
                }
            }
        }
        for eta0 in [-0.03, 0.07] {
            let parameters = ThresholdParameters {
                fixed: 0.4,
                surface: 0.2,
                body: -0.1,
                short: -0.3,
                narrow: -0.2,
                eta0,
                etab: 0.02,
                theta0vb0: 0.5,
            };
            let point = [-0.3, 0.4, 0.8, 0.2, 0.1];
            let rates = [0.002, 0.003, -0.004, 0.001, -0.002];
            let local =
                |i: usize| Differential::<5>::variable(i, Jet::affine(ORDER, point[i], rates[i]));
            let law =
                parameters.evaluate(&local(0)?, &local(1)?, &local(2)?, &local(3)?, &local(4)?)?;
            verify_local(&law);
            for t in [0., 0.01, 0.02] {
                let [b, drain, root, short, narrow] =
                    std::array::from_fn(|i| rates[i].mul_add(t, point[i]));
                let x = 0.02_f64.mul_add(b, eta0);
                let eta = if x < 1e-4 {
                    (2e-4 - x) / (3. - 2e4 * x)
                } else {
                    x
                };
                let expected =
                    0.4 + 0.2 * root - 0.1 * b - 0.3 * short - 0.2 * narrow - 0.5 * eta * drain;
                close(value_at(&law.value.value, t), expected, 1e-15);
            }
        }
        for interface in [-0.01, 0.001] {
            let parameters = SlopeParameters {
                depletion: 1e-10,
                coupling: [0.001, -0.0002, 0.0003],
                interface,
                oxide: 0.01,
            };
            let point = [-0.3, 0.4, 1e-7, 0.2];
            let rates = [0.002, 0.003, 1e-10, 0.001];
            let local =
                |i: usize| Differential::<4>::variable(i, Jet::affine(ORDER, point[i], rates[i]));
            let law = parameters.evaluate(&local(0)?, &local(1)?, &local(2)?, &local(3)?)?;
            verify_local(&law);
            for t in [0., 0.01, 0.02] {
                let [b, drain, depth, theta] =
                    std::array::from_fn(|i| rates[i].mul_add(t, point[i]));
                let raw =
                    (1e-10 / depth + (0.001 - 0.0002 * b + 0.0003 * drain) * theta + interface)
                        / 0.01;
                let expected = if raw < -0.5 {
                    (1. + 3. * raw) / (3. + 8. * raw)
                } else {
                    1. + raw
                };
                close(value_at(&law.value.value, t), expected, 1e-15);
            }
        }
        Ok(())
    }

    #[test]
    fn saturation_and_early_voltages_share_exact_local_partials() -> SpiceResult<()> {
        type Local = Differential<6>;
        for point in [
            [0.8, 1.2, 0.5, 0.2, 0.8, -0.5],
            [0.1, 0.8, 0.1, 0., 1., -1.1],
            [1e-12, 1., 0.8, 1e-13, 0.999999999999, 0.1],
        ] {
            let thermal = 0.026;
            let rates = [point[0] * 0.02, 0.003, 0.004, 0.001, -0.002, 0.003];
            let local = |i: usize| Local::variable(i, Jet::affine(ORDER, point[i], rates[i]));
            let overdrive = local(0)?;
            let bulk = local(1)?;
            let field = local(2)?;
            let resistance = local(3)?;
            let shape = local(4)?;
            let body = local(5)?;
            let drive = overdrive.clone().offset(2. * thermal);
            let saturation = saturation_voltage(&drive, &bulk, &field, &resistance, &shape)?;
            for (coordinate, partial) in saturation.partials.iter().enumerate() {
                for (degree, &expected) in partial.value.coefficients().iter().enumerate() {
                    let actual = saturation.value.gradient[coordinate]
                        .as_ref()
                        .map_or(0., |g| g.coefficients()[degree]);
                    close(actual, expected, 3e-15);
                }
            }
            // Independent coordinates for the local Early-voltage laws.
            let independent_sat = Local::variable(2, saturation.value.value.clone())?;
            let early = saturation_early_voltage(
                &local(0)?,
                &local(1)?,
                &independent_sat,
                &Local::variable(3, field.value.clone())?,
                &Local::variable(4, resistance.value.clone())?,
                &Local::variable(5, shape.value.clone())?,
                thermal,
            )?;
            let dibl = dibl_early_voltage(
                &drive,
                &bulk,
                &independent_sat,
                &Local::variable(3, body.value.clone())?,
                0.2,
                1.,
            )?;
            for (value, partials) in [
                (&early.value, early.partials.as_slice()),
                (&dibl.value, dibl.partials.as_slice()),
            ] {
                for (coordinate, partial) in partials.iter().enumerate() {
                    for (degree, &expected) in partial.value.coefficients().iter().enumerate() {
                        let actual = value.gradient[coordinate]
                            .as_ref()
                            .map_or(0., |g| g.coefficients()[degree]);
                        close(actual, expected, 3e-15);
                    }
                }
            }
            for time in [0., 0.001, 0.02] {
                let [u, a, e, r, l, b]: [Real; 6] =
                    std::array::from_fn(|i| rates[i].mul_add(time, point[i]));
                let drive = u + 2. * thermal;
                let quadratic = 2. * a * (a * r + (1. - l) / l);
                let linear = drive.mul_add((3. * a).mul_add(r, 2. / l - 1.), a * e);
                let constant = drive * (2. * drive).mul_add(r, e);
                let y = value_at(&saturation.value.value, time);
                // Check the polynomial rather than a second copy of the root.
                let residual = (0.5 * quadratic * y).mul_add(y, (-linear).mul_add(y, constant));
                close(residual, 0., 3e-14 * constant.abs());
                let expected_early = (2. * r * u).mul_add((-0.5 * a).mul_add(y / drive, 1.), e + y)
                    / a.mul_add(r, 2. / l - 1.);
                let body_factor = if b < -0.9 {
                    20.0_f64.mul_add(b, 17.) / (0.8 + b)
                } else {
                    1. / (1. + b)
                };
                let expected_dibl = drive * drive / a.mul_add(y, drive) / 0.2 * body_factor;
                close(value_at(&early.value.value, time), expected_early, 1e-14);
                close(value_at(&dibl.value.value, time), expected_dibl, 1e-14);
            }
        }
        Ok(())
    }

    #[test]
    fn channel_and_series_conductance_match_reference_and_partial_germs() -> SpiceResult<()> {
        type Local = Differential<6>;
        for point in [
            [0.8, 1.2, 0.2, 0.5, 0.003, 150.],
            [0.1, 0.8, 0., 0.1, 0.002, 0.],
            [1e-12, 1., 1e-13, 0.8, 0.001, 50.],
        ] {
            let rates = [point[0] * 0.02, 0.003, point[0] * 0.01, 0.004, 1e-5, 0.5];
            let local = |i: usize| Local::variable(i, Jet::affine(ORDER, point[i], rates[i]));
            let thermal = 0.026;
            let channel = channel_conductance(
                &local(0)?,
                &local(1)?,
                &local(2)?,
                &local(3)?,
                &local(4)?,
                thermal,
            )?;
            let series = series_conductance(&channel.value, &local(5)?)?;
            for (coordinate, partial) in channel.partials.iter().enumerate() {
                for (degree, &expected) in partial.value.coefficients().iter().enumerate() {
                    let actual = channel.value.gradient[coordinate]
                        .as_ref()
                        .map_or(0., |g| g.coefficients()[degree]);
                    close(actual, expected, 1e-22);
                }
            }
            for coordinate in 0..6 {
                let upstream = channel.value.gradient[coordinate]
                    .as_ref()
                    .map_or_else(|| Jet::constant(ORDER, 0.), Clone::clone);
                let mut expected = series.partials[0].value.multiply(&upstream);
                if coordinate == 5 {
                    expected = expected.add(&series.partials[1].value);
                }
                for (degree, &coefficient) in expected.coefficients().iter().enumerate() {
                    let actual = series.value.gradient[coordinate]
                        .as_ref()
                        .map_or(0., |g| g.coefficients()[degree]);
                    close(actual, coefficient, 1e-22);
                }
            }
            for time in [0., 0.001, 0.02] {
                let [u, a, d, e, b, r]: [Real; 6] =
                    std::array::from_fn(|i| rates[i].mul_add(time, point[i]));
                let reference =
                    b * u * (-0.5_f64).mul_add(a * d / (u + 2. * thermal), 1.) / (1. + d / e);
                close(value_at(&channel.value.value, time), reference, 1e-24);
                close(
                    value_at(&series.value.value, time),
                    reference / reference.mul_add(r, 1.),
                    1e-24,
                );
            }
        }
        Ok(())
    }

    #[test]
    fn transport_continuations_match_reference_and_one_sided_partials() -> SpiceResult<()> {
        for a1 in [-0.4, 0., 0.4] {
            for bias in [0., 0.1, 1., 3., 20.] {
                let a2 = 0.8;
                let (value, slope) = saturation_shape(&variable(bias, 0.01)?, a1, a2)?;
                if a1 != 0. {
                    check_charge_gradient(&value, &slope);
                }
                for time in [0., 0.02] {
                    let u = 0.01_f64.mul_add(time, bias);
                    let reference = if a1 > 0. {
                        let x = (-a1).mul_add(u, 1. - a2 - 0.0001);
                        1. - x.midpoint(x.mul_add(x, 0.0004 * (1. - a2)).sqrt())
                    } else if a1 < 0. {
                        let x = a1.mul_add(u, a2 - 0.0001);
                        x.midpoint(x.mul_add(x, 0.0004 * a2).sqrt())
                    } else {
                        a2
                    };
                    close(value_at(&value.value, time), reference, 1e-15);
                }
            }
        }
        let threshold = 0.34;
        for bias in [0., 1e-18, threshold / 34., 0.04, 0.4] {
            for rate in [-0.0001, 0.0001] {
                let input = variable(bias, rate)?;
                let (impact, slope) = impact_ionization(&input, 2., threshold)?;
                check_charge_gradient(&impact, &slope);
                let (early, slope) = substrate_early_voltage(&input, 1., threshold, 2., true)?;
                for (degree, &expected) in slope.value.coefficients().iter().enumerate() {
                    let actual = early.gradient[0]
                        .as_ref()
                        .map_or(0., |g| g.coefficients()[degree]);
                    close(actual, expected, 1e-25);
                }
                for time in [0.001, 0.02] {
                    let x = rate.mul_add(time, bias);
                    let (i, v) = if x > threshold / 34. {
                        (2. * x * (-threshold / x).exp(), 0.5 * (threshold / x).exp())
                    } else {
                        (2. * MIN_EXP * x, 0.5 * MAX_EXP)
                    };
                    close(value_at(&impact.value, time), i, 1e-25);
                    close(value_at(&early.value, time), v, 1e-25);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn thermal_overdrive_matches_reference_and_local_partials() -> SpiceResult<()> {
        type Local = Differential<2>;
        const THERMAL: Real = 0.03125;
        const FACTOR: Real = 1.5;
        let scale = THERMAL * FACTOR;
        for argument in [-60., -34., -25., -1., 0., 1., 25., 34., 60.] {
            for direction in [-1., 1.] {
                let voltage = scale * argument;
                let rates = [direction * 0.015, 0.001];
                let input = Local::variable(0, Jet::affine(ORDER, voltage, rates[0]))?;
                let factor = Local::variable(1, Jet::affine(ORDER, FACTOR, rates[1]))?;
                for (value, voltage_partial, factor_partial) in [
                    thermal_softplus(&input, &factor, THERMAL)?,
                    effective_overdrive(&input, &factor, THERMAL, 0.6, -0.02, 1.7)?,
                ] {
                    for (coordinate, partial) in
                        [voltage_partial, factor_partial].iter().enumerate()
                    {
                        for (degree, &expected) in partial.value.coefficients().iter().enumerate() {
                            let actual = value.gradient[coordinate]
                                .as_ref()
                                .map_or(0., |g| g.coefficients()[degree]);
                            close(actual, expected, 3e-15);
                        }
                    }
                }
                let (soft, _, _) = thermal_softplus(&input, &factor, THERMAL)?;
                let (effective, _, _) =
                    effective_overdrive(&input, &factor, THERMAL, 0.6, -0.02, 1.7)?;
                let reference = |v: Real, n: Real| {
                    let z = v / (THERMAL * n);
                    if z > 34. {
                        v
                    } else if z < -34. {
                        THERMAL * n * MIN_EXP.ln_1p()
                    } else {
                        THERMAL * n * z.exp().ln_1p()
                    }
                };
                for time in [0.001, 0.025] {
                    let v = rates[0].mul_add(time, voltage);
                    let n = rates[1].mul_add(time, FACTOR);
                    close(value_at(&soft.value, time), reference(v, n), 1e-27);
                    let z = (-0.4_f64).mul_add(v, -0.02) / (n * THERMAL);
                    let exponential = if z < -34. {
                        MIN_EXP
                    } else if z > 34. {
                        MAX_EXP
                    } else {
                        z.exp()
                    };
                    let expected = reference(0.6 * v, n) / (1.7 * n).mul_add(exponential, 0.6);
                    close(value_at(&effective.value, time), expected, 1e-27);
                }
            }
        }
        Ok(())
    }

    fn cap_zero_reference(point: [Real; 5], p: CapZeroParameters) -> [Real; 3] {
        let [gate, body, surface, drain, bulk] = point;
        let argument = gate - body - p.flatband;
        let overdrive = gate - (p.flatband + p.phi + surface);
        let area = p.oxide_area;
        if argument <= 0. {
            let charge = area * argument;
            return [charge, -charge, 0.];
        }
        if overdrive <= 0. {
            let half = p.body_effect * 0.5;
            let charge = area * p.body_effect * (half.mul_add(half, argument).sqrt() - half);
            return [charge, -charge, 0.];
        }
        let saturation = overdrive / bulk;
        if drain >= saturation {
            let gate_charge = area * (gate - p.flatband - p.phi - saturation / 3.);
            let inversion = 2. * area / 3. * overdrive;
            let fraction = if p.partition > 0.5 {
                0.
            } else if p.partition < 0.5 {
                0.4
            } else {
                0.5
            };
            return [gate_charge, inversion - gate_charge, -inversion * fraction];
        }
        // Direct C capMod=0 linear-region formula; it computes the drain
        // partition rather than the complementary source partition.
        let sum = 2.0f64.mul_add(saturation, -drain);
        let ratio = drain / (3. * sum);
        let correction = ratio * drain;
        let factor = 0.25 * area * bulk;
        let gate_charge = area * (-0.5_f64).mul_add(drain - correction, gate - p.flatband - p.phi);
        let drain_charge = if p.partition > 0.5 {
            factor * (-3.0_f64).mul_add(correction, 2.0f64.mul_add(drain, -sum))
        } else if p.partition < 0.5 {
            let polynomial = (8. * saturation).mul_add(
                saturation,
                (-6. * saturation).mul_add(drain, 1.2 * drain * drain),
            );
            factor * (ratio / sum).mul_add(-polynomial, drain - sum)
        } else {
            -factor * (sum + correction)
        };
        let inversion = 2. * factor * (sum + correction);
        [gate_charge, inversion - gate_charge, drain_charge]
    }

    #[test]
    fn abrupt_zero_hundred_drain_charge_retains_quadratic_saturation_tail() -> SpiceResult<()> {
        type Local = Differential<2>;
        let area = 2e-12;
        let gate = 0.875;
        let surface = 0.375;
        let drive = gate - surface;
        let gap = 2.0_f64.powi(-40);
        let drain = drive - gap;
        let rate = gap * 0.25;
        let gate_jet = Local::variable(0, Jet::constant(ORDER, gate))?;
        let drain_jet = Local::variable(1, Jet::affine(ORDER, drain, rate))?;
        let [_, _, charge] = cap_zero(
            &CapZeroBias {
                gate: gate_jet,
                body_cv: Local::constant(ORDER, 0.),
                surface: Local::constant(ORDER, surface),
                drain: drain_jet,
                bulk_factor: Local::constant(ORDER, 1.),
            },
            CapZeroParameters {
                flatband: -0.5,
                phi: 0.5,
                body_effect: 0.75,
                oxide_area: area,
                partition: 1.,
            },
        )?;
        // Use the independently known gap rather than subtracting two
        // nearly equal sampled voltages in the reference calculation.
        for time in [0., 0.01, 0.025] {
            let g = (-rate).mul_add(time, gap);
            let width = 0.5 * (drive + g);
            let reference = -0.5 * area * g * g / width;
            close(value_at(&charge.value.value, time), reference, 0.);
        }
        for (coordinate, partial) in [(0, &charge.partials[0]), (1, &charge.partials[3])] {
            for (&actual, &expected) in charge.value.gradient[coordinate]
                .as_ref()
                .unwrap()
                .coefficients()
                .iter()
                .zip(partial.value.coefficients())
            {
                close(actual, expected, 1e-300);
            }
        }
        Ok(())
    }

    #[test]
    fn abrupt_cv_matches_reference_regions_and_all_local_jacobian_germs() -> SpiceResult<()> {
        type Local = Differential<5>;
        for partition in [0., 0.5, 1.] {
            let parameters = CapZeroParameters {
                flatband: -0.5,
                phi: 0.5,
                body_effect: 0.75,
                oxide_area: 2e-12,
                partition,
            };
            for point in [
                [-0.5, 0.25, 0.375, 0.1, 1.],
                [-0.25, 0.25, 0.375, 0.1, 1.],
                [-0.125, 0.25, 0.375, 0.1, 1.],
                [0.375, 0.25, 0.375, 0.1, 1.],
                [0.875, 0.25, 0.375, 0., 1.],
                [0.875, 0.25, 0.375, 0.2, 1.25],
                [0.875, 0.25, 0.375, 0.5, 1.],
                [0.875, 0.25, 0.375, 0.7, 1.25],
            ] {
                for direction in [-1., 1.] {
                    let rates = [0.02, 0.003, -0.002, 0.04, 0.004].map(|v| v * direction);
                    let law = |index: usize| {
                        Local::variable(index, Jet::affine(ORDER, point[index], rates[index]))
                    };
                    let charges = cap_zero(
                        &CapZeroBias {
                            gate: law(0)?,
                            body_cv: law(1)?,
                            surface: law(2)?,
                            drain: law(3)?,
                            bulk_factor: law(4)?,
                        },
                        parameters,
                    )?;
                    for part in &charges {
                        for (coordinate, partial) in part.partials.iter().enumerate() {
                            for (degree, &expected) in
                                partial.value.coefficients().iter().enumerate()
                            {
                                let actual = part.value.gradient[coordinate]
                                    .as_ref()
                                    .map_or(0., |g| g.coefficients()[degree]);
                                close(actual, expected, 3e-25);
                            }
                        }
                    }
                    for time in [0., 1e-3, 0.025] {
                        let sample = std::array::from_fn(|i| rates[i].mul_add(time, point[i]));
                        let expected = cap_zero_reference(sample, parameters);
                        for (charge, reference) in charges.iter().zip(expected) {
                            close(value_at(&charge.value.value, time), reference, 3e-25);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn rational_continuations_match_reference_and_one_sided_partials() -> SpiceResult<()> {
        // Original C rational factors, parameterized here only to share
        // the independent verification of their linear and continued sides.
        for (join, offset, stiffness, numerator, denominator) in [
            (-0.5, 1., 8., [1., 3.], [3., 8.]),
            (1e-4, 0., 2e4, [2e-4, -1.], [3., -2e4]),
            (2e-8, 0., 1e8, [8e-16, -2e-8], [6e-8, -2.]),
            (0.1, 0., 20., [0.2, -1.], [3., -20.]),
            (-0.9, 1., 20., [0.8, 1.], [17., 20.]),
            (-0.8, 1., 10., [0.6, 1.], [7., 10.]),
        ] {
            let reference = |x: Real| {
                if x < join {
                    f64::mul_add(numerator[1], x, numerator[0])
                        / f64::mul_add(denominator[1], x, denominator[0])
                } else {
                    x + offset
                }
            };
            for (shift, direction) in [(-0.2, -1.), (0., -1.), (0., 1.), (0.2, 1.)] {
                let bias = join + shift / stiffness;
                let rate = direction / (8. * stiffness);
                let (value, slope) =
                    rational_floor(variable(bias, rate)?, join, offset, stiffness)?;
                check_charge_gradient(&value, &slope);
                for time in [0., 1e-3, 0.02] {
                    let x = rate.mul_add(time, bias);
                    close(value_at(&value.value, time), reference(x), 2e-26);
                }
            }
        }
        for (bias, rate) in [(-1.2, 0.01), (-0.9, -0.01), (-0.9, 0.01), (0., -0.01)] {
            let (value, slope) = reciprocal_floor(variable(bias, rate)?, -0.9, 20.)?;
            check_charge_gradient(&value, &slope);
            for time in [0., 1e-3, 0.02] {
                let x = rate.mul_add(time, bias);
                let reference = if x < -0.9 {
                    20.0f64.mul_add(x, 17.) / (0.8 + x)
                } else {
                    1. / (1. + x)
                };
                close(value_at(&value.value, time), reference, 2e-26);
            }
        }
        Ok(())
    }

    #[test]
    fn mixed_junction_depletion_has_exact_one_sided_charge_and_partial_series() -> SpiceResult<()> {
        let parts = [
            JunctionCapacitance {
                zero: 2e-12,
                potential: 0.6,
                grading: 0.5,
            },
            JunctionCapacitance {
                zero: 3e-13,
                potential: 0.9,
                grading: 1.,
            },
            JunctionCapacitance {
                zero: 4e-13,
                potential: 0.8,
                grading: 0.33,
            },
        ];
        for (bias, rate) in [
            (-0.9, 0.04),
            (-1e-14, -0.02),
            (0., -0.04),
            (0., 0.04),
            (0.25, -0.03),
        ] {
            let (charge, cap) = depletion(&variable(bias, rate)?, parts, |_| {})?;
            let (scalar_charge, scalar_cap) = depletion(&bias, parts, |_| {})?;
            close(charge.value.value(), scalar_charge, 2e-28);
            close(cap.value.value(), scalar_cap, 2e-28);
            check_charge_gradient(&charge, &cap);
            let mut expected = [0.; ORDER + 1];
            let reverse = bias < 0. || (bias == 0. && rate < 0.);
            for part in parts {
                if reverse {
                    let mut coefficient =
                        part.zero * (1. - bias / part.potential).powf(-part.grading);
                    for (n, term) in expected.iter_mut().enumerate() {
                        *term += coefficient;
                        coefficient *= (part.grading + n as Real) / (n as Real + 1.) * rate
                            / (part.potential - bias);
                    }
                } else {
                    expected[0] += part.zero * (1. + bias * part.grading / part.potential);
                    expected[1] += rate * part.zero * part.grading / part.potential;
                }
            }
            for (n, (&actual, &analytic)) in
                cap.value.coefficients().iter().zip(&expected).enumerate()
            {
                close(actual, analytic, 2e-28);
                if n < ORDER {
                    close(
                        charge.value.coefficients()[n + 1],
                        rate * analytic / (n as Real + 1.),
                        2e-28,
                    );
                }
            }
            let sample = 0.02;
            let (reference, _) = depletion(&rate.mul_add(sample, bias), parts, |_| {})?;
            close(value_at(&charge.value, sample), reference, 2e-25);
        }
        Ok(())
    }

    #[test]
    fn overlap_uses_one_charge_law_and_preserves_analytic_capacitance() -> SpiceResult<()> {
        let base = 1.3e-12;
        let extra = 4.7e-13;
        let kappa = 0.6;
        for bias in [-10., -0.8, -0.02, 0., 0.5, 10., 1e6] {
            let rate = 0.001;
            let (cap, charge) = overlap(variable(bias, rate)?, base, extra, kappa)?;
            check_charge_gradient(&charge, &cap);
            for sample in [-0.03, 0.03] {
                let v = rate.mul_add(sample, bias);
                let offset = v + 0.02;
                let root = (offset * offset + 0.08).sqrt();
                let half = 0.5 * (offset - root);
                let secondary = (1. - 4. * half / kappa).sqrt();
                // Reference's original, unrationalized charge formula.
                let reference =
                    (base + extra) * v - extra * (half + 0.5 * kappa * (secondary - 1.));
                close(value_at(&charge.value, sample), reference, 2e-23);
            }
        }
        Ok(())
    }

    #[test]
    fn saturation_and_charge_overdrive_preserve_higher_order_partials() -> SpiceResult<()> {
        type Local = Differential<3>;
        let rates = [0.01, -0.02, 0.005];
        for a in [-0.1, 0., 0.1] {
            let b = 1.;
            let c = 0.125;
            let root = SaturationRoot::new(
                &Local::variable(0, Jet::affine(ORDER, a, rates[0]))?,
                Local::variable(1, Jet::affine(ORDER, b, rates[1]))?,
                Local::variable(2, Jet::affine(ORDER, c, rates[2]))?,
            )?;
            for i in 0..3 {
                let basis: [Local; 3] =
                    std::array::from_fn(|j| Local::constant(ORDER, if i == j { 1. } else { 0. }));
                let partial = root.derivative(&basis[0], &basis[1], &basis[2])?;
                for (&actual, &expected) in root.value.gradient[i]
                    .as_ref()
                    .unwrap()
                    .coefficients()
                    .iter()
                    .zip(partial.value.coefficients())
                {
                    close(actual, expected, 2e-15);
                }
            }
            for time in [-0.02, 0.02] {
                let x = value_at(&root.value.value, time);
                let aa = rates[0].mul_add(time, a);
                let bb = rates[1].mul_add(time, b);
                let cc = rates[2].mul_add(time, c);
                close(aa * x * x * 0.5 - bb * x + cc, 0., 3e-15);
            }
        }
        for k1 in [0., 0.4] {
            for (bias, rate) in [(1e-20, 1e-21), (0.001, 0.0001), (0.5, 0.01)] {
                let (value, slope) = charge_overdrive(variable(bias, rate)?, k1, 15., 0.026, 0.8)?;
                check_charge_gradient(&value, &slope);
                let time = 0.02;
                let (expected, _) =
                    charge_overdrive(rate.mul_add(time, bias), k1, 15., 0.026, 0.8)?;
                close(value_at(&value.value, time), expected, 1e-30);
            }
        }
        Ok(())
    }

    #[test]
    fn inversion_partitions_match_independent_charge_formulas_and_full_partial_germs()
    -> SpiceResult<()> {
        type Local = Differential<4>;
        let area = 1.7e-12;
        for partition in [0., 0.5, 1.] {
            for (overdrive, drain, bulk) in [
                (0.8, 0., 1.2),
                (0.8, 0.2, 1.2),
                (0.8, 0.6, 1.2),
                (0.2, 0.1, 1.),
                (1e-21, 1e-22, 0.8),
            ] {
                let rates = [overdrive * 0.01, overdrive * 0.01, -0.003, area * 0.02];
                let point = [overdrive, drain, bulk, area];
                let [u, d, a, c] = point;
                let laws = channel_inversion(
                    &Local::variable(0, Jet::affine(ORDER, u, rates[0]))?,
                    &Local::variable(1, Jet::affine(ORDER, d, rates[1]))?,
                    &Local::variable(2, Jet::affine(ORDER, a, rates[2]))?,
                    &Local::variable(3, Jet::affine(ORDER, c, rates[3]))?,
                    partition,
                    1e-20,
                )?;
                for law in &laws {
                    for (coordinate, partial) in law.partials.iter().enumerate() {
                        for (degree, &expected) in partial.value.coefficients().iter().enumerate() {
                            let actual = law.value.gradient[coordinate]
                                .as_ref()
                                .map_or(0., |g| g.coefficients()[degree]);
                            close(actual, expected, 3e-26);
                        }
                    }
                }
                for time in [0., 0.01, 0.025] {
                    let [u, d, a, c]: [Real; 4] =
                        std::array::from_fn(|i| rates[i].mul_add(time, point[i]));
                    // Independently evaluate the original C cubic and charge laws.
                    let weighted = a * d;
                    let width = (-0.5_f64).mul_add(weighted, u) + 1e-20;
                    let denominator = 12. * width;
                    let gate = c * (u - weighted * (0.5 - weighted / denominator));
                    let bulk = c * (1. - a) * (d * 0.5 - weighted * d / denominator);
                    let source = if partition > 0.5 {
                        -c * (u * 0.5 + weighted * 0.25 - weighted * weighted * 0.5 / denominator)
                    } else if partition < 0.5 {
                        -0.5 * c / (width * width)
                            * (u * (2. * weighted * weighted / 3. + u * (u - 4. * weighted / 3.))
                                - 2. * weighted * weighted * weighted / 15.)
                    } else {
                        -gate * 0.5
                    };
                    for (law, expected) in laws.iter().zip([gate, bulk, source]) {
                        close(value_at(&law.value.value, time), expected, 3e-16 * c * u);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn smoothing_retains_full_trajectory_and_local_partials() -> SpiceResult<()> {
        type Pair = Differential<2>;
        let delta = 0.01;
        for (saturation, drain, ds, dd) in [
            (0.5, 0., 0.01, 0.02),
            (0.5, 0.1, 0.01, -0.02),
            (0.5, 0.5, -0.01, 0.02),
            (0.5, 1.2, 0.01, 0.02),
            (1e-8, 1., 1e-9, 0.01),
            (1., 1e-18, 0.01, 1e-19),
        ] {
            let sat = Pair::variable(0, Jet::affine(ORDER, saturation, ds))?;
            let vds = Pair::variable(1, Jet::affine(ORDER, drain, dd))?;
            let law = smooth_drain(sat, vds, delta)?;
            for (index, partial) in [&law.saturation_partial, &law.drain_partial]
                .into_iter()
                .enumerate()
            {
                let gradient = law.effective.gradient[index].as_ref().unwrap();
                for (&actual, &expected) in gradient
                    .coefficients()
                    .iter()
                    .zip(partial.value.coefficients())
                {
                    close(actual, expected, 3e-15);
                }
            }
            let reconstructed = law.effective.value.clone().add(&law.drain_excess.value);
            let reference = Jet::affine(ORDER, drain, dd);
            for (&actual, &expected) in reconstructed
                .coefficients()
                .iter()
                .zip(reference.coefficients())
            {
                close(actual, expected, 3e-15);
            }
            for time in [0.01, 0.02] {
                let reference =
                    smooth_drain(ds.mul_add(time, saturation), dd.mul_add(time, drain), delta)?;
                close(
                    value_at(&law.effective.value, time),
                    reference.effective,
                    2e-24,
                );
                close(
                    value_at(&law.drain_excess.value, time),
                    reference.drain_excess,
                    2e-24,
                );
            }
        }
        Ok(())
    }

    #[test]
    fn smooth_positive_and_depletion_share_local_derivatives() -> SpiceResult<()> {
        for (bias, rate, width) in [
            (-1e12, 1e10, 0.0025),
            (-0.5, 0.01, 0.0025),
            (0., 0.01, 0.0025),
            (0., -0.01, 0.),
            (0., 0.01, 0.),
            (0.5, -0.01, 0.),
        ] {
            let (value, slope) = smooth_positive(variable(bias, rate)?, width)?;
            for (index, &expected) in slope.value.coefficients().iter().enumerate() {
                let actual = value.gradient[0]
                    .as_ref()
                    .map_or(0., |gradient| gradient.coefficients()[index]);
                close(actual, expected, 1e-300);
            }
            let sample = 0.02;
            let (reference, _) = smooth_positive(rate.mul_add(sample, bias), width)?;
            close(value_at(&value.value, sample), reference, 1e-24);
        }
        for (bias, rate) in [
            (-0.2, 0.01),
            (0., -0.01),
            (0., 0.01),
            (1e-30, 1e-31),
            (0.1, -0.01),
        ] {
            let (charge, slope) = depletion_charge(0.4, variable(bias, rate)?)?;
            check_charge_gradient(&charge, &slope);
            let sample = 0.02;
            let (reference, _) = depletion_charge(0.4, rate.mul_add(sample, bias))?;
            close(value_at(&charge.value, sample), reference, 1e-40);
        }
        Ok(())
    }

    #[test]
    fn poly_depletion_preserves_local_curvature_and_one_sided_threshold() -> SpiceResult<()> {
        const PHI: Real = 0.9;
        const DOPING: Real = 2e19;
        const EPS: Real = 1.03594e-10;
        const COX: Real = 0.0115;
        for electron_charge in [1.602_19e-19, 1.602_176_620_8e-19] {
            let evaluate =
                |voltage| poly_depletion(voltage, PHI, DOPING, EPS, COX, electron_charge);
            let scalar = |voltage| poly_depletion(voltage, PHI, DOPING, EPS, COX, electron_charge);
            for (bias, rate) in [
                (0.7, 0.02),
                (PHI, -0.02),
                (PHI, 0.02),
                (1.5, 0.02),
                (2., -0.02),
            ] {
                let (effective, slope) = evaluate(variable(bias, rate)?)?;
                let (value, derivative) = scalar(bias)?;
                close(effective.value.value(), value, 2e-15);
                close(slope.value.value(), derivative, 2e-15);
                let gradient = effective.gradient[0].as_ref().unwrap();
                for (&actual, &expected) in gradient
                    .coefficients()
                    .iter()
                    .zip(slope.value.coefficients())
                {
                    close(actual, expected, 2e-15);
                }
                let time = 0.03;
                let (expected, expected_slope) = scalar(rate.mul_add(time, bias))?;
                close(value_at(&effective.value, time), expected, 2e-15);
                close(value_at(&slope.value, time), expected_slope, 2e-15);
                if bias.to_bits() == PHI.to_bits() {
                    let scale = 1e6 * electron_charge * EPS * DOPING / (COX * COX);
                    let expected = if rate > 0. {
                        -0.5 * rate * rate / scale * (0.5 + 0.5 * 1.07 / 1.17)
                    } else {
                        0.
                    };
                    close(effective.value.coefficients()[2], expected, 1e-18);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn bsim3_diode_continuation_preserves_actual_current_and_partial_series() -> SpiceResult<()> {
        let sat = 2e-14;
        let thermal = 0.031;
        let edge = 0.17;
        let edge_current = sat * (edge / thermal).exp();
        let gmin = 1e-13;
        for (bias, rate, continuation) in [
            (-0.12, 0.01, None),
            (1e-18, 0.01, None),
            (edge, -0.01, Some((edge, edge_current))),
            (edge, 0.01, Some((edge, edge_current))),
            (0.4, -0.01, Some((edge, edge_current))),
        ] {
            let (current, slope) =
                bsim3_junction(variable(bias, rate)?, sat, thermal, continuation, gmin)?;
            check_charge_gradient(&current, &slope);
            let linear = continuation.is_some() && (bias > edge || (bias == edge && rate > 0.));
            let (expected, first) = if linear {
                let conductance = edge_current / thermal;
                (
                    gmin.mul_add(bias, conductance.mul_add(bias - edge, edge_current - sat)),
                    (conductance + gmin) * rate,
                )
            } else {
                (
                    gmin.mul_add(bias, sat * (bias / thermal).exp_m1()),
                    (sat / thermal * (bias / thermal).exp() + gmin) * rate,
                )
            };
            close(current.value.value(), expected, 2e-28);
            close(current.value.coefficients()[1], first, 2e-28);
            let mut coefficient = sat * (bias / thermal).exp() * rate / thermal;
            for (n, &actual) in current.value.coefficients().iter().enumerate().skip(2) {
                coefficient *= rate / thermal / n as Real;
                close(actual, if linear { 0. } else { coefficient }, 2e-28);
            }
        }
        Ok(())
    }
}
