//! Idealized diode I-V law: tangent-matched quadratic transitions and
//! optional current-limited conducting branches. No solver state is needed.

use crate::primitives::{
    Real, SpiceError, SpiceResult,
    numeric::{scale_power, split_power},
};

#[derive(Clone, Copy, Debug)]
pub(super) struct Point {
    pub current: Real,
    pub conductance: Real,
}

/// Multiply then divide, retaining a representable result when an
/// intermediate product or quotient overflows or underflows.
pub(super) fn ratio_product(a: Real, b: Real, c: Real) -> Real {
    let product = a * b;
    let direct = product / c;
    if product.is_normal() && direct.is_normal() {
        return direct;
    }
    if a == 0. || b == 0. || !a.is_finite() || !b.is_finite() {
        return direct;
    }
    let (ma, ea) = split_power(a);
    let (mb, eb) = split_power(b);
    let (mc, ec) = split_power(c);
    let (mantissa, adjustment) = split_power(ma * mb / mc);
    scale_power(mantissa, ea + eb - ec + adjustment)
}

#[derive(Clone, Copy, Debug)]
struct Limited {
    maximum: Real,
    /// Normalized conducting current at the end of the quadratic.
    normalized_join: Real,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Branch {
    threshold: Real,
    off: Real,
    on: Real,
    width: Real,
    join: Point,
    limited: Option<Limited>,
}

impl Branch {
    pub(super) fn new(
        threshold: Real,
        off: Real,
        on: Real,
        width: Real,
        maximum: Option<Real>,
    ) -> SpiceResult<Self> {
        let (join, limited) = if let Some(maximum) = maximum {
            let (normalized_join, conductance) = if width == 0. {
                (0., on)
            } else {
                limited_join(off, on, width, maximum)?
            };
            (
                Point {
                    current: width * Real::midpoint(off, conductance),
                    conductance,
                },
                Some(Limited {
                    maximum,
                    normalized_join,
                }),
            )
        } else {
            (
                Point {
                    current: width * Real::midpoint(off, on),
                    conductance: on,
                },
                None,
            )
        };
        if !(threshold * off).is_finite()
            || !(threshold + width).is_finite()
            || !join.current.is_finite()
            || !join.conductance.is_finite()
        {
            return Err(SpiceError::circuit(
                "nonrepresentable ideal diode transition",
            ));
        }
        Ok(Self {
            threshold,
            off,
            on,
            width,
            join,
            limited,
        })
    }

    pub(super) fn point(self, voltage: Real) -> Point {
        if voltage <= self.threshold {
            return Point {
                current: voltage * self.off,
                conductance: self.off,
            };
        }
        let distance = voltage - self.threshold;
        let baseline = self.threshold * self.off;
        if distance < self.width {
            let fraction = distance / self.width;
            let conductance = (self.join.conductance - self.off).mul_add(fraction, self.off);
            return Point {
                current: distance.mul_add(Real::midpoint(self.off, conductance), baseline),
                conductance,
            };
        }
        let excess = distance - self.width;
        let outer = self.limited.map_or_else(
            || Point {
                current: self.on.mul_add(excess, self.join.current),
                conductance: self.on,
            },
            |limit| self.limited_point(excess, limit),
        );
        Point {
            current: baseline + outer.current,
            conductance: outer.conductance,
        }
    }

    pub(super) const fn has_limit(self) -> bool {
        self.limited.is_some()
    }

    /// Keep an increasing Newton trial from jumping directly into a
    /// numerically flat current-limit tail. The physical law is unchanged.
    pub(super) fn limit_trial(self, raw: Real, previous: Real) -> Real {
        let Some(limit) = self.limited else {
            return raw;
        };
        let join = self.threshold + self.width;
        let reference = previous.max(join);
        if raw <= reference {
            return raw;
        }
        let normalized = ratio_product(reference - join, self.on, limit.maximum);
        if normalized.tanh() >= 1. {
            // Already on the rounded asymptote: allow an imposed voltage
            // to advance without taking arbitrarily many tiny steps.
            return raw;
        }
        let next = reference + limit.maximum / self.on;
        if next > reference { raw.min(next) } else { raw }
    }

    fn limited_point(self, excess: Real, limit: Limited) -> Point {
        let z = ratio_product(excess, self.on, limit.maximum);
        let t = z.tanh();
        let denominator = limit.normalized_join.mul_add(t, 1.);
        // Hyperbolic addition avoids subtracting the (potentially very
        // large) horizontal offset. Evaluate sech^2 without cancellation
        // when tanh(z) has rounded to one.
        let increment = if z < Real::MIN_POSITIVE {
            // In the subnormal/zero range tanh(z) is z to machine precision.
            // Avoid losing a representable current to normalization by L.
            excess * self.join.conductance
        } else {
            ratio_product(limit.maximum * t, self.join.conductance, self.on) / denominator
        };
        let decay = (-2. * z).exp();
        let conductance = if decay.is_normal() {
            self.join.conductance * (4. * decay / (1. + decay).powi(2)) / denominator.powi(2)
        } else {
            // Four equal exponent factors keep every possible nonzero
            // binary64 result in range, even if exp(-2z) underflows.
            // The first factor includes 4 / denominator^2; all following
            // factors decrease magnitude, and 1 + decay rounds to one.
            let quarter = (-0.5 * z).exp();
            (((self.join.conductance * (4. * quarter / denominator.powi(2))) * quarter) * quarter)
                * quarter
        };
        Point {
            current: (self.join.current + increment).min(limit.maximum),
            conductance,
        }
    }
}

/// Compute 1 - w*h/(2*L) without rounding the product before subtracting.
/// A negative remainder means there is no nonnegative tangent solution.
fn remaining_limit_fraction(width: Real, off: Real, maximum: Real) -> SpiceResult<Real> {
    if off == 0. {
        return Ok(1.);
    }
    let (mw, ew) = split_power(width);
    let (mh, eh) = split_power(off);
    let (ml, el) = split_power(maximum);
    let remainder = (-mw).mul_add(scale_power(mh, ew + eh - el - 1), ml);
    if remainder < 0. {
        return Err(SpiceError::circuit(
            "ideal diode smoothing width and leakage exceed the current limit",
        ));
    }
    Ok(remainder / ml)
}

fn limited_join(off: Real, on: Real, width: Real, maximum: Real) -> SpiceResult<(Real, Real)> {
    // Let u=I_join/I_limit. Continuity of current and slope gives
    // u = width/(2*I_limit) * [G_off + G_on*(1-u^2)].
    // Normalize conductances before solving the quadratic, then use its
    // cancellation-free positive root.
    let remaining = remaining_limit_fraction(width, off, maximum)?;
    let scale = on.max(off);
    let a = on / scale;
    let b = off / scale;
    let sum = a + b;
    let ratio = ratio_product(width, scale, maximum);
    let root = if ratio <= 2. {
        let half = ratio * 0.5;
        2. * half * sum / (1. + (2. * half * (a * sum).sqrt()).hypot(1.))
    } else {
        let inverse = 2. / ratio;
        2. * sum / (inverse + inverse.hypot(2. * (a * sum).sqrt()))
    };
    let u = root.clamp(0., 1.);
    // Use the complementary quadratic identity instead of subtracting
    // a rounded u from one. With X=w*g/(2*L), Y=w*h/(2*L):
    // (1-u)*(1+X*(1+u)) = 1-Y.
    // Both branches preserve a finite slope even when X overflows or u
    // rounds to one. The second form cancels X analytically.
    let on_ratio = ratio_product(width, on, maximum);
    let factor = 1. + u;
    let conductance = if on_ratio <= 2. {
        ratio_product(on, remaining * factor, (0.5 * on_ratio).mul_add(factor, 1.))
    } else {
        ratio_product(
            maximum,
            2. * remaining / (1. + (2. / on_ratio) / factor),
            width,
        )
    };
    Ok((u, conductance))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_joins_preserve_value_and_slope_for_multiple_scales() -> SpiceResult<()> {
        for scale in [1e-150, 1e-50, 1., 1e50, 1e150] {
            for limit in [None, Some(0.25 * scale), Some(1e100 * scale)] {
                let branch = Branch::new(1., 0.01 * scale, 0.5 * scale, 0.4, limit)?;
                for center in [1., 1.2, 1.4, 1.9, 4.] {
                    let delta = 1e-6;
                    let left = branch.point(center - delta);
                    let right = branch.point(center + delta);
                    let exact = branch.point(center);
                    assert!(left.current <= exact.current && exact.current <= right.current);
                    assert!(exact.conductance >= 0.);
                    let finite_difference = (right.current - left.current) / (2. * delta);
                    assert!(
                        (finite_difference - exact.conductance).abs() / scale < 2e-6,
                        "{scale} {limit:?} {center}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn representable_product_ratios_keep_subnormal_factors() {
        let min = Real::from_bits(1);
        assert_eq!(ratio_product(min, 1e-100, 1e-100).to_bits(), min.to_bits());
        assert_eq!(
            ratio_product(Real::MAX, 2., Real::MAX).to_bits(),
            2_f64.to_bits()
        );
        let expected = 1e-200;
        let actual = ratio_product(1e-300, 1e-100, 1e-200);
        assert!((actual / expected - 1.).abs() < 1e-15);
    }

    #[test]
    fn current_limit_retains_representable_derivative_after_tanh_rounds() -> SpiceResult<()> {
        let branch = Branch::new(0., 0., 1., 0., Some(1.))?;
        let point = branch.point(30.);
        assert_eq!(point.current.to_bits(), 1_f64.to_bits());
        let expected = 1. / 30_f64.cosh().powi(2);
        assert!((point.conductance / expected - 1.).abs() < 1e-14);
        Ok(())
    }

    #[test]
    fn limited_tail_preserves_representable_conductance_at_extreme_scales() -> SpiceResult<()> {
        // Independent 100-digit decimal evaluation of 4*g*exp(-2*z)/(1+exp(-2*z))^2.
        for (on, voltage, expected) in [
            (1e300, 400., 1.467_149_833_671_074_9e-47),
            (1e300, 700., 3.888_528_861_902_665e-308),
            (1e308, 720., 1.651_984_6e-317),
            (1e308, 727., Real::from_bits(3)),
        ] {
            let branch = Branch::new(0., 0., on, 0., Some(on))?;
            let actual = branch.point(voltage).conductance;
            assert!(actual > 0.);
            let ulp = Real::from_bits(expected.to_bits() + 1) - expected;
            assert!(
                (actual - expected).abs() <= 2e-13 * expected + ulp,
                "{actual:e} != {expected:e}"
            );
        }
        Ok(())
    }

    #[test]
    fn tangent_matching_retains_small_join_slope() -> SpiceResult<()> {
        // When the normalized join rounds toward one, its complementary
        // distance is not recoverable by subtraction. Compare the implicit
        // tangent equation in terms of physical conductance.
        for on in [1e10, 1e14, 1e16, 1e100, 1e300] {
            let (u, slope) = limited_join(0., on, 1., 1.)?;
            assert!((slope - 2. * u).abs() < 4e-15, "{on:e} {u:e} {slope:e}");
        }
        // (1+2^-52)*(1-2^-52) = 1-2^-104. The rounded product is
        // exactly one, but the positive remaining current must survive.
        let width = 1. + Real::EPSILON;
        let off = 1. - Real::EPSILON;
        let remaining = remaining_limit_fraction(width, off, 0.5)?;
        assert_eq!(remaining.to_bits(), 2_f64.powi(-104).to_bits());
        let (_, slope) = limited_join(off, 1., width, 0.5)?;
        assert!(slope > 0.);
        Ok(())
    }
}
