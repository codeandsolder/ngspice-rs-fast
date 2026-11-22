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

/// Square-sum in binary exponent units. Products can exceed the binary64
/// range or fall below its smallest subnormal without losing their norm
/// after normalization. The final divisor need not be known while adding
/// samples (for example, an adaptive integration window).
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

    /// Add the square of a product without forming the product in its
    /// original units. Both factors are finite and nonnegative.
    pub(super) fn add_product(&mut self, magnitude: Real, factor: Real) {
        let (Some((m, me)), Some((f, fe))) = (split(magnitude), split(factor)) else {
            return;
        };
        let mantissa = m * f;
        let exponent = me + fe;
        let scale = match self.exponent {
            Some(old) if exponent > old => {
                // The old accumulator may have collected many terms. Split
                // its total as well, before applying the exponent difference.
                let total = self.squares.total();
                let rescaled =
                    split(total).map_or(0., |(m, e)| m * power_of_two(e + 2 * (old - exponent)));
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
        self.squares
            .add(mantissa * mantissa * power_of_two(2 * (exponent - scale)));
    }

    /// Euclidean norm divided by a finite positive factor. The final
    /// multiplication performs the only rounding into the subnormal range.
    pub(super) fn norm_over(self, divisor: Real) -> Real {
        let Some(exponent) = self.exponent else {
            return 0.;
        };
        let Some((d, de)) = split(divisor) else {
            return Real::NAN;
        };
        let norm = self.squares.total().sqrt() / d;
        let Some((m, e)) = split(norm) else {
            return norm;
        };
        let exponent = exponent - de + e;
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
    fn scaled_products_survive_exponents_outside_binary64() {
        use super::ScaledSquares;
        for (value, factor) in [(f64::MAX, 2_f64.powi(500)), (f64::from_bits(1), 1e-150)] {
            let mut norm = ScaledSquares::new();
            norm.add_product(value, factor);
            let actual = norm.norm_over(factor);
            assert!((actual - value).abs() <= 4. * f64::EPSILON * value);
        }
        // An initially tiny scale must not overflow when rebased.
        let mut norm = ScaledSquares::new();
        norm.add_product(1e-300, 1e-100);
        norm.add_product(1e300, 1e100);
        assert!((norm.norm_over(1e100) - 1e300).abs() <= 4. * f64::EPSILON * 1e300);
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
