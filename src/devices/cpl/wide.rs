//! Two-word arithmetic for ill-conditioned immutable cable preparation.
//!
//! Keeping the low word through eigenbasis and scattering composition
//! prevents rounding-induced modal rotation or attenuation in exact null
//! spaces. Only prepared terminal coefficients are rounded to binary64.
use crate::primitives::{
    Complex, Real, SpiceResult,
    numeric::{scale_power, split_power},
};

use super::invalid;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Wide {
    pub(super) hi: Real,
    lo: Real,
}

impl Wide {
    pub(super) const ZERO: Self = Self { hi: 0., lo: 0. };
    pub(super) const ONE: Self = Self { hi: 1., lo: 0. };

    pub(super) const fn from_real(hi: Real) -> Self {
        Self { hi, lo: 0. }
    }

    // Normalize every factor first: the physical product can overflow
    // even when the balanced dimensionless generator is bounded.
    pub(super) fn product_ratio(factors: &[Real], divisor: Real, exponent: i32) -> Self {
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

    pub(super) fn abs(self) -> Self {
        if self.hi.is_sign_negative() {
            -self
        } else {
            self
        }
    }

    pub(super) fn sqrt(self) -> Self {
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

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct WideComplex {
    re: Wide,
    im: Wide,
}

impl WideComplex {
    pub(super) const ZERO: Self = Self::real(Wide::ZERO);
    pub(super) const ONE: Self = Self::real(Wide::ONE);

    pub(super) const fn new(re: Wide, im: Wide) -> Self {
        Self { re, im }
    }

    pub(super) const fn real(re: Wide) -> Self {
        Self::new(re, Wide::ZERO)
    }

    pub(super) const fn narrow(self) -> Complex {
        Complex::new(self.re.hi, self.im.hi)
    }

    fn magnitude(self) -> Real {
        self.re.hi.hypot(self.im.hi)
    }

    const fn finite(self) -> bool {
        self.re.hi.is_finite()
            && self.re.lo.is_finite()
            && self.im.hi.is_finite()
            && self.im.lo.is_finite()
    }
}

impl std::ops::Neg for WideComplex {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl std::ops::Add for WideComplex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(self.re + other.re, self.im + other.im)
    }
}

impl std::ops::Sub for WideComplex {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(self.re - other.re, self.im - other.im)
    }
}

impl std::ops::Mul for WideComplex {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )
    }
}

impl std::ops::Div for WideComplex {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        // Smith division: do not square a nearly singular pivot.
        if other.re.hi.abs() >= other.im.hi.abs() {
            let ratio = other.im / other.re;
            let denominator = other.re + other.im * ratio;
            Self::new(
                (self.re + self.im * ratio) / denominator,
                (self.im - self.re * ratio) / denominator,
            )
        } else {
            let ratio = other.re / other.im;
            let denominator = other.im + other.re * ratio;
            Self::new(
                (self.re * ratio + self.im) / denominator,
                (self.im * ratio - self.re) / denominator,
            )
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct WideMatrix {
    pub(super) n: usize,
    data: Vec<WideComplex>,
}

impl WideMatrix {
    pub(super) fn zero(n: usize) -> Self {
        Self {
            n,
            data: vec![WideComplex::ZERO; n * n],
        }
    }

    pub(super) fn identity(n: usize) -> Self {
        let mut result = Self::zero(n);
        for i in 0..n {
            result.set(i, i, WideComplex::ONE);
        }
        result
    }

    pub(super) fn at(&self, row: usize, col: usize) -> WideComplex {
        self.data[row * self.n + col]
    }

    pub(super) fn set(&mut self, row: usize, col: usize, value: WideComplex) {
        self.data[row * self.n + col] = value;
    }

    pub(super) fn add(&self, other: &Self) -> Self {
        Self {
            n: self.n,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| a + b)
                .collect(),
        }
    }

    pub(super) fn sub(&self, other: &Self) -> Self {
        Self {
            n: self.n,
            data: self
                .data
                .iter()
                .zip(&other.data)
                .map(|(&a, &b)| a - b)
                .collect(),
        }
    }

    pub(super) fn scale(&self, factor: Wide) -> Self {
        let factor = WideComplex::real(factor);
        Self {
            n: self.n,
            data: self.data.iter().map(|&a| a * factor).collect(),
        }
    }

    pub(super) fn transpose(&self) -> Self {
        let mut result = Self::zero(self.n);
        for r in 0..self.n {
            for c in 0..self.n {
                result.set(c, r, self.at(r, c));
            }
        }
        result
    }

    pub(super) fn symmetric(mut self) -> Self {
        let half = WideComplex::real(Wide::from_real(0.5));
        for r in 0..self.n {
            for c in 0..r {
                let average = (self.at(r, c) + self.at(c, r)) * half;
                self.set(r, c, average);
                self.set(c, r, average);
            }
        }
        self
    }

    pub(super) fn mul(&self, other: &Self) -> Self {
        let mut result = Self::zero(self.n);
        for (left, out) in self
            .data
            .chunks_exact(self.n)
            .zip(result.data.chunks_exact_mut(self.n))
        {
            for (&value, right) in left.iter().zip(other.data.chunks_exact(self.n)) {
                if value == WideComplex::ZERO {
                    continue;
                }
                for (target, &entry) in out.iter_mut().zip(right) {
                    *target = *target + value * entry;
                }
            }
        }
        result
    }

    pub(super) fn magnitude(&self) -> Real {
        self.data
            .iter()
            .fold(0., |maximum, v| maximum.max(v.magnitude()))
    }

    // Solve a bounded complex multiple-right-hand-side relation with
    // row pivoting. No inverse of a physical impedance/admittance is used.
    pub(super) fn infinity_norm(&self) -> Real {
        self.data
            .chunks_exact(self.n)
            .map(|row| row.iter().map(|value| value.magnitude()).sum::<Real>())
            .fold(0., Real::max)
    }

    pub(super) fn solve(&self, rhs: &Self) -> SpiceResult<Self> {
        let mut a = self.clone();
        let mut b = rhs.clone();
        let n = a.n;
        for k in 0..n {
            let pivot = (k..n)
                .max_by(|&i, &j| a.at(i, k).magnitude().total_cmp(&a.at(j, k).magnitude()))
                .ok_or_else(|| invalid("empty scattering pivot range"))?;
            if a.at(pivot, k) == WideComplex::ZERO {
                return Err(invalid("singular passive scattering interface"));
            }
            if pivot != k {
                for c in 0..n {
                    a.data.swap(pivot * n + c, k * n + c);
                    b.data.swap(pivot * n + c, k * n + c);
                }
            }
            let denominator = a.at(k, k);
            for c in k..n {
                a.set(k, c, a.at(k, c) / denominator);
            }
            for c in 0..n {
                b.set(k, c, b.at(k, c) / denominator);
            }
            for r in 0..n {
                if r == k {
                    continue;
                }
                let factor = a.at(r, k);
                if factor == WideComplex::ZERO {
                    continue;
                }
                a.set(r, k, WideComplex::ZERO);
                for c in k + 1..n {
                    a.set(r, c, a.at(r, c) - factor * a.at(k, c));
                }
                for c in 0..n {
                    b.set(r, c, b.at(r, c) - factor * b.at(k, c));
                }
            }
        }
        b.check_finite()?;
        Ok(b)
    }

    pub(super) fn check_finite(&self) -> SpiceResult<()> {
        if self.data.iter().any(|v| !v.finite()) {
            return Err(invalid(
                "nonfinite extended-precision scattering coefficient",
            ));
        }
        Ok(())
    }
}
