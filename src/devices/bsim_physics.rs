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
        Ok((value, sigmoid, entropy.scale(thermal)))
    } else {
        let exponential = argument.clone().exp();
        let sigmoid = exponential
            .clone()
            .divide(&exponential.clone().offset(1.))?;
        let logarithm = exponential.ln_1p()?;
        let entropy = argument.scale(-1.).mul_add(&sigmoid, &logarithm);
        Ok((
            factor.clone().multiply(&logarithm.scale(thermal)),
            sigmoid,
            entropy.scale(thermal),
        ))
    }
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
    let slope = value.clone().divide(&root)?;
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

    pub(super) fn derivative(&self, da: &S, db: &S, dc: &S) -> SpiceResult<S> {
        let y = &self.value;
        y.clone()
            .scale(0.5)
            .multiply(y)
            .mul_add(da, &y.clone().scale(-1.).mul_add(db, dc))
            .divide(&self.discriminant)
    }
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

/// A charge value and its local partials in the (overdrive, effective
/// drain voltage, bulk factor) coordinates. All include oxide area.
pub(super) struct ChargePart<S, const N: usize = 3> {
    pub(super) value: S,
    partials: [S; N],
}

/// The scalar load's existing local-coordinate derivatives. Oxide area
/// changes in proportion to effective oxide capacitance.
pub(super) struct ChargeChain {
    pub(super) overdrive: [Real; 3],
    pub(super) drain: [Real; 3],
    pub(super) bulk: [Real; 3],
    pub(super) relative_oxide: [Real; 3],
}

impl ChargeChain {
    pub(super) fn apply(&self, part: &ChargePart<Real>) -> [Real; 3] {
        std::array::from_fn(|i| {
            part.partials[0].mul_add(
                self.overdrive[i],
                part.partials[1].mul_add(
                    self.drain[i],
                    part.partials[2].mul_add(self.bulk[i], part.value * self.relative_oxide[i]),
                ),
            )
        })
    }
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
) -> SpiceResult<[ChargePart<S>; 3]> {
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
    let part = |value: S, du: S, dd: S, da: S| ChargePart {
        value: value.multiply(oxide_area),
        partials: [du, dd, da].map(|derivative| derivative.multiply(oxide_area)),
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
/// reference source-oriented frame. The surface term is k1ox*sqrt(phi_s).
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

impl<S: Constitutive, const N: usize> ChargePart<S, N> {
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

impl<const N: usize> ChargePart<Real, N> {
    /// Project the independent constitutive partials onto terminal biases.
    pub(super) fn project(&self, coordinates: &[[Real; 3]; N]) -> [Real; 3] {
        std::array::from_fn(|column| {
            self.partials
                .iter()
                .zip(coordinates)
                .fold(0., |sum, (&partial, direction)| {
                    partial.mul_add(direction[column], sum)
                })
        })
    }
}

/// BSIM4 capMod=0: accumulation, depletion, inversion in the linear
/// region, and saturated inversion. The latter is the exact endpoint of
/// the same Ward-Dutton law, with no numerical 1e-20 regularization.
/// Returns gate, bulk and drain charges and all five local partials.
pub(super) fn cap_zero<S: Constitutive>(
    bias: &CapZeroBias<S>,
    parameters: CapZeroParameters,
) -> SpiceResult<[ChargePart<S, 5>; 3]> {
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
        let gate = ChargePart {
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
        let drain = ChargePart {
            value: zero.clone(),
            partials: std::array::from_fn(|_| zero.clone()),
        };
        return Ok([gate, bulk, drain]);
    }

    let saturation = overdrive.clone().divide(&bias.bulk_factor)?;
    if bias
        .drain
        .clone()
        .add(&saturation.clone().scale(-1.))
        .compare(0.)
        != Some(Ordering::Less)
    {
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
    let project = |part: ChargePart<S>| {
        let [gate, drain, factor] = part.partials;
        let surface = gate.clone().scale(-1.);
        ChargePart {
            value: part.value,
            partials: [gate, zero.clone(), surface, drain, factor],
        }
    };
    let inversion = project(inversion);
    let bulk = project(bulk);
    let source = project(source);
    let surface = ChargePart {
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
    let drain = inversion.add(&source).scaled(-1.);
    Ok([gate, bulk, drain])
}

/// The saturation endpoint is evaluated directly: the 0/100 drain
/// charge and its derivatives are identically zero, not a subtraction
/// between nearly equal inversion and source charges.
fn cap_zero_saturation<S: Constitutive>(
    bias: &CapZeroBias<S>,
    parameters: CapZeroParameters,
    overdrive: &S,
    saturation: &S,
) -> SpiceResult<[ChargePart<S, 5>; 3]> {
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
    let gate = ChargePart {
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
    let inversion = ChargePart {
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
