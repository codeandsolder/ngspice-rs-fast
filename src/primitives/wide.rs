//! Portable two-word real arithmetic for cancellation-sensitive calculations.
//! Both words remain binary64; this extends precision, not exponent range.
use super::{
    ExactSum, Real, Wide,
    numeric::{scale_power, split_power},
};

impl Wide {
    pub(crate) const ZERO: Self = Self { hi: 0., lo: 0. };
    pub(crate) const ONE: Self = Self { hi: 1., lo: 0. };

    pub(crate) const fn from_real(hi: Real) -> Self {
        Self { hi, lo: 0. }
    }

    // Normalize every factor first: the physical product can overflow
    // even when the balanced dimensionless generator is bounded.
    pub(crate) fn product_ratio(factors: &[Real], divisor: Real, exponent: i32) -> Self {
        if factors.contains(&0.) {
            return Self::ZERO;
        }
        let (denominator, power) = split_power(divisor);
        let mut result = Self::ONE / Self::from_real(denominator);
        let mut exponent = exponent - power;
        for &factor in factors {
            let (significand, power) = split_power(factor);
            result = result * Self::from_real(significand);
            exponent += power;
        }
        Self {
            hi: scale_power(result.hi, exponent),
            lo: scale_power(result.lo, exponent),
        }
    }

    fn two_sum(a: Real, b: Real) -> Self {
        let hi = a + b;
        let v = hi - a;
        let lo = (a - (hi - v)) + (b - v);
        Self { hi, lo }
    }

    pub(crate) fn abs(self) -> Self {
        if self.hi.is_sign_negative() {
            -self
        } else {
            self
        }
    }

    /// Round an exact sum into two words without losing the low word
    /// during cancellation of widely separated physical stamps.
    pub(crate) fn from_sum(mut sum: ExactSum) -> Self {
        let hi = sum.rounded_sum();
        if !hi.is_finite() {
            return Self::from_real(hi);
        }
        sum.add_product(-hi, 1.);
        Self::two_sum(hi, sum.rounded_sum())
    }

    /// Both binary64 words contribute independently to the exact product.
    pub(crate) fn add_product(self, factor: Real, sum: &mut ExactSum) {
        sum.add_product(self.hi, factor);
        sum.add_product(self.lo, factor);
    }

    pub(crate) const fn finite(self) -> bool {
        self.hi.is_finite() && self.lo.is_finite()
    }

    pub(crate) fn sqrt(self) -> Self {
        if self == Self::ZERO {
            return self;
        }
        let root = Self::from_real(self.hi.sqrt());
        root + (self - root * root) / (root + root)
    }
}

impl std::ops::Neg for Wide {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl std::ops::Add for Wide {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let sum = Self::two_sum(self.hi, other.hi);
        Self::two_sum(sum.hi, sum.lo + self.lo + other.lo)
    }
}

impl std::ops::Sub for Wide {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl std::ops::Mul for Wide {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let hi = self.hi * other.hi;
        let error = self.hi.mul_add(other.hi, -hi);
        Self::two_sum(
            hi,
            self.hi.mul_add(other.lo, self.lo.mul_add(other.hi, error)),
        )
    }
}

impl std::ops::Div for Wide {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let quotient = self.hi / other.hi;
        let remainder = self - Self::from_real(quotient) * other;
        Self::two_sum(quotient, remainder.hi / other.hi)
    }
}
