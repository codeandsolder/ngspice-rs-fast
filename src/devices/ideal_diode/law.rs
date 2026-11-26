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

    fn limited_point(self, excess: Real, limit: Limited) -> Point {
        let z = ratio_product(excess, self.on, limit.maximum);
        let t = z.tanh();
        let denominator = limit.normalized_join.mul_add(t, 1.);
        // Hyperbolic addition avoids subtracting the (potentially very
        // large) horizontal offset. Evaluate sech^2 without cancellation
        // when tanh(z) has rounded to one.
        let scaled = if t == 0. {
            excess * self.on
        } else {
            limit.maximum * t
        };
        let decay = (-2. * z).exp();
        let sech_squared = 4. * decay / (1. + decay).powi(2);
        let increment = ratio_product(scaled, self.join.conductance, self.on) / denominator;
        Point {
            current: (self.join.current + increment).min(limit.maximum),
            conductance: self.join.conductance * sech_squared / denominator.powi(2),
        }
    }
}

fn limited_join(off: Real, on: Real, width: Real, maximum: Real) -> SpiceResult<(Real, Real)> {
    // Let u=I_join/I_limit. Continuity of current and slope gives
    // u = width/(2*I_limit) * [G_off + G_on*(1-u^2)].
    // Normalize conductances before solving the quadratic, then use its
    // cancellation-free positive root. The physically possible root is
    // within [0,1]; a wider transition than the leakage alone permits
    // has no finite tangent match.
    let leakage = ratio_product(width, off, maximum);
    if leakage > 2. {
        return Err(SpiceError::circuit(
            "ideal diode smoothing width and leakage exceed the current limit",
        ));
    }
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
    let conductance = if u.to_bits() == 1_f64.to_bits() {
        // At machine precision the root can round to the asymptote while
        // the tangent slope remains nonzero.
        (ratio_product(maximum, 2. * u, width) - off).max(0.)
    } else {
        on * ((1. - u) * (1. + u))
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
}
