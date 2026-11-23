//! Compensated reductions for cancellation-sensitive analysis quantities.

use crate::primitives::Real;
use crate::primitives::numeric::binary_exponent;

/// Neumaier accumulation: retain the rounding error of each addition even
/// when the new term exceeds the running total in magnitude.
#[derive(Clone, Copy, Default)]
pub(super) struct CompensatedSum {
    sum: Real,
    correction: Real,
}

impl CompensatedSum {
    pub(super) const fn with_initial(value: Real) -> Self {
        Self {
            sum: value,
            correction: 0.,
        }
    }

    pub(super) const fn negate(&mut self) {
        self.sum = -self.sum;
        self.correction = -self.correction;
    }

    pub(super) const fn add(&mut self, value: Real) {
        let next = self.sum + value;
        if next.is_finite() {
            self.correction += if self.sum.abs() >= value.abs() {
                (self.sum - next) + value
            } else {
                (value - next) + self.sum
            };
        } else {
            // Preserve ordinary IEEE propagation for sensitivity's block-local
            // nonfinite handling; compensation cannot recover an overflow.
            self.correction = 0.;
        }
        self.sum = next;
    }

    /// Accumulate the two parts of this sum multiplied by another value,
    /// retaining the rounding residual of each finite product.
    pub(super) fn multiply_into(self, multiplier: Real, target: &mut Self) {
        target.add_product(self.sum, multiplier);
        if self.correction != 0. {
            target.add_product(self.correction, multiplier);
        }
    }

    fn add_product(&mut self, left: Real, right: Real) {
        let product = left * right;
        self.add(product);
        if product.is_finite() {
            self.add(left.mul_add(right, -product));
        }
    }

    pub(super) const fn total(self) -> Real {
        if self.correction == 0. {
            self.sum
        } else {
            self.sum + self.correction
        }
    }
}

/// Weighted square-sum in binary exponent units. Neither a squared
/// ordinate nor its product with a physical span has to fit in binary64.
/// Normalize only at the end, when the actual covered window is known.
pub(super) struct ScaledSquares {
    exponent: Option<i32>,
    squares: CompensatedSum,
}

fn power_of_two(exponent: i32) -> Real {
    match exponent {
        ..=-1075 => 0.,
        -1074..=-1023 => Real::from_bits(1_u64 << (exponent + 1074).cast_unsigned()),
        -1022..=1023 => Real::from_bits(u64::from((exponent + 1023).cast_unsigned()) << 52),
        1024.. => Real::INFINITY,
    }
}

fn split(value: Real) -> Option<(Real, i32)> {
    let exponent = binary_exponent(value)?;
    Some((value / power_of_two(exponent), exponent))
}

impl ScaledSquares {
    pub(super) const fn new() -> Self {
        Self {
            exponent: None,
            squares: CompensatedSum::with_initial(0.),
        }
    }

    /// Add squared ordinates with a finite binary64 positive scale.
    pub(super) fn add_weighted(
        &mut self,
        scale: Real,
        samples: impl Iterator<Item = (Real, Real)>,
    ) {
        if let Some(scale) = split(scale) {
            self.add_scaled(scale, samples);
        }
    }

    /// Add squared ordinates with an interval represented outside binary64's
    /// exponent range. The exact endpoint geometry is converted once to a
    /// normalized significand/exponent; individual products never overflow.
    pub(super) fn add_scaled(
        &mut self,
        (sm, se): (Real, i32),
        samples: impl Iterator<Item = (Real, Real)>,
    ) {
        if sm <= 0. {
            return;
        }
        for (value, weight) in samples {
            let (Some((m, me)), Some((w, we))) = (split(value.abs()), split(weight)) else {
                continue;
            };
            let term = m * m * sm * w;
            let exponent = 2 * me + se + we;
            let scale = match self.exponent {
                Some(old) if exponent > old => {
                    // The old accumulator may have collected many terms.
                    // Split its total as well before rescaling.
                    let total = self.squares.total();
                    let rescaled =
                        split(total).map_or(0., |(m, e)| m * power_of_two(e + old - exponent));
                    self.squares = CompensatedSum::with_initial(rescaled);
                    self.exponent = Some(exponent);
                    exponent
                }
                Some(old) => old,
                None => {
                    self.exponent = Some(exponent);
                    exponent
                }
            };
            self.squares.add(term * power_of_two(exponent - scale));
        }
    }

    /// Square root of the normalized positive sum. Split the remaining
    /// exponent by two before taking the root, so a small weighted term
    /// is not rounded to zero before normalization.
    pub(super) fn root_mean(self, divisor: Real) -> Real {
        split(divisor).map_or(Real::NAN, |scale| self.root_mean_scaled(scale))
    }

    /// Normalize by a positive axis span, even when its exact width cannot be
    /// represented as a finite binary64 value.
    pub(super) fn root_mean_scaled(self, (d, de): (Real, i32)) -> Real {
        let Some(exponent) = self.exponent else {
            return 0.;
        };
        if d <= 0. {
            return Real::NAN;
        }
        let exponent = exponent - de;
        let odd = if exponent.rem_euclid(2) == 0 { 1. } else { 2. };
        let root = (self.squares.total() / d * odd).sqrt();
        let Some((m, e)) = split(root) else {
            return root;
        };
        let exponent = exponent.div_euclid(2) + e;
        if exponent < -1022 {
            (m * power_of_two(exponent + 1022)) * Real::MIN_POSITIVE
        } else {
            m * power_of_two(exponent)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CompensatedSum;

    #[test]
    fn scaled_weighted_constant_panels_cover_every_binary64_exponent() {
        use super::{ScaledSquares, power_of_two};
        for ye in -1074..=1023 {
            let value = power_of_two(ye);
            for se in [-1074, -536, 0, 536, 1023] {
                let span = power_of_two(se);
                let mut norm = ScaledSquares::new();
                norm.add_weighted(
                    span,
                    [0.125, 0.375, 0.375, 0.125]
                        .into_iter()
                        .map(|weight| (value, weight)),
                );
                assert_eq!(norm.root_mean(span).to_bits(), value.to_bits());
            }
        }
        let mut norm = ScaledSquares::new();
        norm.add_weighted(1e-100, [(1e-300, 1.)].into_iter());
        norm.add_weighted(1e100, [(1e300, 1.)].into_iter());
        assert!((norm.root_mean(1e100) - 1e300).abs() <= 4. * f64::EPSILON * 1e300);
    }

    #[test]
    fn retains_small_terms_on_both_sides_of_large_cancellation() {
        for values in [[1e16, 1., -1e16], [1., 1e16, -1e16]] {
            let mut sum = CompensatedSum::default();
            for value in values {
                sum.add(value);
            }
            assert_eq!(sum.total(), 1.);
        }
    }
}
