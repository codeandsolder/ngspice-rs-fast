//! Constitutive laws shared by the conservative BSIM charge models.
//! Numerical rationalizations select by the constant term; physical
//! continuations select by the entire one-sided germ.
use std::cmp::Ordering;

use crate::maths::taylor::arithmetic::Constitutive;
use crate::primitives::{Real, SpiceResult};

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
/// Returns (charge, differential capacitance), before multiplicity.
pub(super) fn depletion<S: Constitutive>(
    voltage: S,
    parts: [JunctionCapacitance; 3],
) -> SpiceResult<(S, S)> {
    let [bottom, side, gate] = parts;
    let zero = bottom.zero + side.zero + gate.zero;
    if voltage.compare(0.) == Some(Ordering::Less) {
        let mut charge = voltage.constant(0.);
        let mut cap = voltage.constant(0.);
        for part in parts {
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
        Ok((charge, increment.offset(zero)))
    }
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
            let (charge, cap) = depletion(variable(bias, rate)?, parts)?;
            let (scalar_charge, scalar_cap) = depletion(bias, parts)?;
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
            let (reference, _) = depletion(rate.mul_add(sample, bias), parts)?;
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
