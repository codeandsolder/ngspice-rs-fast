//! Binary64 precision with an extended exponent for rank elimination.
//! A long product of individually representable pivots must not turn a
//! nonzero independent constraint into zero merely through underflow.

use crate::primitives::Real;
use crate::primitives::numeric::{scale_power, split_power};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Wide {
    mantissa: Real,
    exponent: i64,
}

impl Wide {
    pub(super) const ZERO: Self = Self {
        mantissa: 0.,
        exponent: 0,
    };
    const ONE: Self = Self {
        mantissa: 1.,
        exponent: 0,
    };
    const INVALID: Self = Self {
        mantissa: Real::NAN,
        exponent: 0,
    };

    pub(super) fn from_real(value: Real) -> Self {
        Self::normalized(value, 0)
    }

    fn normalized(value: Real, exponent: i64) -> Self {
        if value == 0. || !value.is_finite() {
            return Self {
                mantissa: value,
                exponent: 0,
            };
        }
        let (mantissa, shift) = split_power(value);
        exponent
            .checked_add(i64::from(shift))
            .map_or(Self::INVALID, |exponent| Self { mantissa, exponent })
    }

    pub(super) fn shifted(self, shift: i64) -> Self {
        if self.is_zero() || !self.finite() {
            return self;
        }
        self.exponent
            .checked_add(shift)
            .map_or(Self::INVALID, |exponent| Self { exponent, ..self })
    }

    pub(super) const fn finite(self) -> bool {
        self.mantissa.is_finite()
    }

    pub(super) const fn is_zero(self) -> bool {
        self.mantissa == 0.
    }

    pub(super) const fn abs(self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            ..self
        }
    }

    pub(super) const fn negate(self) -> Self {
        Self {
            mantissa: -self.mantissa,
            ..self
        }
    }

    pub(super) fn magnitude_cmp(self, other: Self) -> Ordering {
        match (self.is_zero(), other.is_zero()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self
                .exponent
                .cmp(&other.exponent)
                .then_with(|| self.mantissa.abs().total_cmp(&other.mantissa.abs())),
        }
    }

    /// Align to a common exponent without first producing a huge or
    /// tiny physical binary64 value.
    fn at(self, exponent: i64) -> Real {
        if self.is_zero() || !self.finite() {
            return self.mantissa;
        }
        let delta = self.exponent.saturating_sub(exponent);
        if delta < -1075 {
            0_f64.copysign(self.mantissa)
        } else if delta > 1023 {
            Real::INFINITY.copysign(self.mantissa)
        } else {
            scale_power(self.mantissa, i32::try_from(delta).unwrap_or_default())
        }
    }

    pub(super) fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        if !self.finite() || !multiplier.finite() || !addend.finite() {
            return Self::INVALID;
        }
        if self.is_zero() || multiplier.is_zero() {
            return addend;
        }
        let Some(product_exponent) = self.exponent.checked_add(multiplier.exponent) else {
            return Self::INVALID;
        };
        if addend.is_zero() {
            return Self::normalized(self.mantissa * multiplier.mantissa, product_exponent);
        }
        let common = product_exponent.max(addend.exponent);
        let factor = Self {
            mantissa: self.mantissa,
            exponent: product_exponent,
        }
        .at(common);
        Self::normalized(
            factor.mul_add(multiplier.mantissa, addend.at(common)),
            common,
        )
    }

    /// Return the computed value and a bound on the actual rounding
    /// residual of a fused update. Do not manufacture error for exact
    /// dyadic eliminations: repeatedly normalizing such rows otherwise
    /// makes a large, exactly full-rank sparse system appear singular.
    pub(super) fn mul_add_with_roundoff(self, multiplier: Self, addend: Self) -> (Self, Self) {
        let value = self.mul_add(multiplier, addend);
        if !value.finite() {
            return (value, Self::INVALID);
        }
        if self.is_zero() || multiplier.is_zero() {
            return (value, Self::ZERO);
        }
        let Some(product_exponent) = self.exponent.checked_add(multiplier.exponent) else {
            return (Self::INVALID, Self::INVALID);
        };
        if addend.is_zero() {
            let product = self.mantissa * multiplier.mantissa;
            let residual = self.mantissa.mul_add(multiplier.mantissa, -product);
            return (value, Self::normalized(residual.abs(), product_exponent));
        }
        let common = product_exponent.max(addend.exponent);
        // At most 900 bits of alignment leaves even the low word of a
        // 106-bit significand product in the normal binary64 range.
        // For wider separation cancellation is impossible; include both
        // the alignment and final rounding errors conservatively.
        if common.saturating_sub(product_exponent.min(addend.exponent)) > 900 {
            let bound = Self::from_real(2. * Real::EPSILON).upper_mul_add(value.abs(), Self::ZERO);
            return (value, bound);
        }
        let factor = Self {
            mantissa: self.mantissa,
            exponent: product_exponent,
        }
        .at(common);
        let addend = addend.at(common);
        let product = factor * multiplier.mantissa;
        let product_error = factor.mul_add(multiplier.mantissa, -product);
        let (sum, sum_error) = two_sum(product, addend);
        let (delta, delta_error) = two_sum(sum, -value.at(common));
        let (low, low_error) = two_sum(product_error, sum_error);
        let (residual, residual_error) = two_sum(delta, low);
        // The four components sum exactly to a*b+c-value in the common
        // exponent. Their absolute sum is a rigorous upper bound, even
        // when the components themselves do not fit in one significand.
        let bound = [residual, residual_error, delta_error, low_error]
            .into_iter()
            .map(|part| Self::normalized(part.abs(), common))
            .fold(Self::ZERO, Self::upper_add);
        (value, bound)
    }

    /// Division together with the residual in numerator units, |n-q*d|.
    /// The two normalized significands and their quotient lie in a small
    /// normal range; a fused residual retains exact dyadic divisions.
    pub(super) fn divide_with_residual(self, divisor: Self) -> (Self, Self) {
        let value = self.divide(divisor);
        if !value.finite() {
            return (value, Self::INVALID);
        }
        if self.is_zero() {
            return (value, Self::ZERO);
        }
        let quotient = self.mantissa / divisor.mantissa;
        let residual = (-quotient).mul_add(divisor.mantissa, self.mantissa);
        (
            value,
            Self::normalized(residual.abs(), self.exponent).upward(),
        )
    }

    // These helpers are used only for nonnegative error magnitudes.
    // Round outward so accumulating an uncertainty cannot shrink it.
    fn upward(self) -> Self {
        if self.is_zero() || !self.finite() {
            self
        } else {
            Self::normalized(self.mantissa.next_up(), self.exponent)
        }
    }

    pub(super) fn upper_add(self, other: Self) -> Self {
        if self.is_zero() {
            other
        } else if other.is_zero() {
            self
        } else {
            self.add(other).upward()
        }
    }

    pub(super) fn upper_mul_add(self, multiplier: Self, addend: Self) -> Self {
        if self.is_zero() || multiplier.is_zero() {
            addend
        } else {
            self.mul_add(multiplier, addend).upward()
        }
    }

    pub(super) fn upper_divide(self, divisor: Self) -> Self {
        self.divide(divisor).upward()
    }

    pub(super) fn lower_subtract(self, other: Self) -> Self {
        if other.is_zero() {
            return self;
        }
        let value = self.subtract(other);
        if value.is_zero() || !value.finite() {
            value
        } else {
            Self::normalized(value.mantissa.next_down(), value.exponent)
        }
    }

    pub(super) fn add(self, other: Self) -> Self {
        self.mul_add(Self::ONE, other)
    }

    pub(super) fn subtract(self, other: Self) -> Self {
        self.add(other.negate())
    }

    pub(super) fn divide(self, other: Self) -> Self {
        if !self.finite() || !other.finite() || other.is_zero() {
            return Self::INVALID;
        }
        if self.is_zero() {
            return Self::ZERO;
        }
        let Some(exponent) = self.exponent.checked_sub(other.exponent) else {
            return Self::INVALID;
        };
        Self::normalized(self.mantissa / other.mantissa, exponent)
    }

    pub(super) fn finish(self) -> Real {
        self.at(0)
    }
}

/// Error-free addition of aligned finite normal significands (and zero).
fn two_sum(a: Real, b: Real) -> (Real, Real) {
    let sum = a + b;
    let recovered = sum - a;
    (sum, (a - (sum - recovered)) + (b - recovered))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extreme_exponents_remain_independent_and_cancel_exactly() {
        let large = Wide::from_real(1.5).shifted(3000);
        let small = Wide::from_real(0.5).shifted(-3000);
        assert_eq!(small.magnitude_cmp(Wide::ZERO), Ordering::Greater);
        assert!(large.finish().is_infinite());
        assert_eq!(small.finish(), 0.);
        assert_eq!(large.mul_add(small, Wide::ZERO).finish(), 0.75);
        assert!(large.subtract(large).is_zero());
        assert_eq!(small.divide(small).finish(), 1.);
        assert_eq!(large.mul_add(Wide::ONE, large.negate()).finish(), 0.);
    }

    #[test]
    fn rounding_residual_preserves_exact_products_sums_and_cancellation() {
        let a = Wide::from_real(1.5).shifted(3000);
        let b = Wide::from_real(2.).shifted(-1000);
        let c = Wide::from_real(3.).shifted(2000);
        let (value, error) = a.mul_add_with_roundoff(b, c.negate());
        assert!(value.is_zero());
        assert!(error.is_zero());

        let a = Wide::from_real(1. + Real::EPSILON);
        let b = Wide::from_real(1. - Real::EPSILON);
        let (value, error) = a.mul_add_with_roundoff(b, Wide::ONE.negate());
        assert_eq!(value.finish(), -Real::EPSILON * Real::EPSILON);
        assert!(error.is_zero());

        let (value, error) =
            Wide::ONE.mul_add_with_roundoff(Wide::ONE, Wide::from_real(0.5 * Real::EPSILON));
        assert_eq!(value.finish(), 1.);
        assert_eq!(error.finish(), 0.5 * Real::EPSILON);

        let (value, error) = Wide::ONE
            .shifted(5000)
            .mul_add_with_roundoff(Wide::ONE, Wide::ONE.shifted(-5000));
        assert_eq!(
            value.magnitude_cmp(Wide::ONE.shifted(5000)),
            Ordering::Equal
        );
        assert!(error.magnitude_cmp(Wide::ONE.shifted(-5000)).is_ge());
    }

    #[test]
    fn division_residual_is_measured_in_numerator_units() {
        let a = Wide::from_real(1.5).shifted(5000);
        let b = Wide::from_real(1.5).shifted(-3000);
        let (value, residual) = a.divide_with_residual(b);
        assert_eq!(
            value.magnitude_cmp(Wide::ONE.shifted(8000)),
            Ordering::Equal
        );
        assert!(residual.is_zero());

        let (value, residual) = Wide::ONE.divide_with_residual(Wide::from_real(3.));
        assert_eq!(value.finish(), 1. / 3.);
        let exact_residual = Wide::ONE.shifted(-54);
        assert!(residual.magnitude_cmp(exact_residual).is_ge());
        assert!(residual.magnitude_cmp(Wide::ONE.shifted(-53)).is_lt());
    }
}
