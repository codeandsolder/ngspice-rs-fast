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
}
