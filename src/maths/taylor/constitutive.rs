//! Shared constitutive arithmetic for allocation-free scalar Newton loads
//! and exact multivariate Taylor germs. Both modes select the same law;
//! only the comparison at a join uses the right-hand time germ.
use super::{Jet, differential::Differential};
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::cmp::Ordering;

pub(crate) trait Constitutive: Clone {
    fn value(&self) -> Real;
    fn compare(&self, boundary: Real) -> Ordering;
    fn constant_like(&self, value: Real) -> Self;
    fn with_value(self, value: Real) -> Self;
    fn add(self, rhs: &Self) -> Self;
    fn offset(self, value: Real) -> Self;
    fn scale(self, factor: Real) -> Self;
    fn multiply(self, rhs: &Self) -> Self;
    fn reciprocal(self) -> SpiceResult<Self>;
    fn exp(self) -> Self;
    fn divide(self, rhs: &Self) -> SpiceResult<Self>;
    fn divide_constant(self, divisor: Real) -> Self;
    fn tanh(self) -> SpiceResult<Self>;
    fn atan_scaled(self, width: Real) -> SpiceResult<Self>;
    fn check(&self) -> SpiceResult<()>;

    fn affine(self, factor: Real, offset: Real) -> Self {
        let value = self.value().mul_add(factor, offset);
        self.scale(factor).offset(offset).with_value(value)
    }

    fn square(self) -> Self {
        self.clone().multiply(&self)
    }
}

/// Stack-only scalar value and local partial derivatives.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dual<const N: usize> {
    pub(crate) value: Real,
    pub(crate) gradient: [Real; N],
}

impl<const N: usize> Dual<N> {
    pub(crate) fn variable(index: usize, value: Real) -> SpiceResult<Self> {
        let mut gradient = [0.; N];
        *gradient
            .get_mut(index)
            .ok_or_else(|| SpiceError::circuit("invalid scalar constitutive coordinate"))? = 1.;
        Ok(Self { value, gradient })
    }

    fn lift(self, value: Real, slope: Real) -> Self {
        Self {
            value,
            gradient: self.gradient.map(|g| {
                if g == 0. || slope == 0. {
                    0.
                } else {
                    g * slope
                }
            }),
        }
    }
}

impl<const N: usize> Constitutive for Dual<N> {
    fn value(&self) -> Real {
        self.value
    }
    fn compare(&self, boundary: Real) -> Ordering {
        self.value.partial_cmp(&boundary).unwrap_or(Ordering::Equal)
    }
    fn constant_like(&self, value: Real) -> Self {
        Self {
            value,
            gradient: [0.; N],
        }
    }
    fn with_value(mut self, value: Real) -> Self {
        self.value = value;
        self
    }
    fn add(self, rhs: &Self) -> Self {
        Self {
            value: self.value + rhs.value,
            gradient: std::array::from_fn(|i| self.gradient[i] + rhs.gradient[i]),
        }
    }
    fn offset(mut self, value: Real) -> Self {
        self.value += value;
        self
    }
    fn scale(self, factor: Real) -> Self {
        self.lift(self.value * factor, factor)
    }
    fn multiply(self, rhs: &Self) -> Self {
        Self {
            value: self.value * rhs.value,
            gradient: std::array::from_fn(|i| {
                let a = if self.gradient[i] == 0. {
                    0.
                } else {
                    self.gradient[i] * rhs.value
                };
                let b = if rhs.gradient[i] == 0. {
                    0.
                } else {
                    rhs.gradient[i] * self.value
                };
                a + b
            }),
        }
    }
    fn reciprocal(self) -> SpiceResult<Self> {
        let value = self.value.recip();
        let result = Self {
            value,
            gradient: self.gradient.map(|g| -(g * value) * value),
        };
        result.check()?;
        Ok(result)
    }
    fn exp(self) -> Self {
        let value = self.value.exp();
        self.lift(value, value)
    }
    fn divide(self, rhs: &Self) -> SpiceResult<Self> {
        let value = self.value / rhs.value;
        let result = Self {
            value,
            gradient: std::array::from_fn(|i| {
                (-value).mul_add(rhs.gradient[i], self.gradient[i]) / rhs.value
            }),
        };
        result.check()?;
        Ok(result)
    }
    fn divide_constant(self, divisor: Real) -> Self {
        Self {
            value: self.value / divisor,
            gradient: self.gradient.map(|g| g / divisor),
        }
    }
    fn tanh(self) -> SpiceResult<Self> {
        let value = self.value.tanh();
        Ok(self.lift(value, sech_squared(self.value)))
    }
    fn atan_scaled(self, width: Real) -> SpiceResult<Self> {
        let value = self.value.atan2(width);
        let scale = self.value.abs().max(width);
        let slope = if self.value.is_infinite() {
            0.
        } else {
            let w = width / scale;
            let x = self.value / scale;
            (w / x.mul_add(x, w * w)) / scale
        };
        Ok(self.lift(value, slope))
    }
    fn check(&self) -> SpiceResult<()> {
        if self.value.is_finite() && self.gradient.iter().all(|v| v.is_finite()) {
            Ok(())
        } else {
            Err(SpiceError::circuit(
                "nonfinite scalar constitutive value or derivative",
            ))
        }
    }
}

impl<const N: usize> Constitutive for Differential<N> {
    fn value(&self) -> Real {
        self.value.value()
    }
    fn compare(&self, boundary: Real) -> Ordering {
        self.value.compare(boundary).unwrap_or(Ordering::Equal)
    }
    fn constant_like(&self, value: Real) -> Self {
        Self::constant(self.value.order(), value)
    }
    fn with_value(mut self, value: Real) -> Self {
        self.value = self.value.with_value(value);
        self
    }
    fn add(self, rhs: &Self) -> Self {
        self.add(rhs)
    }
    fn offset(self, value: Real) -> Self {
        self.offset(value)
    }
    fn scale(self, factor: Real) -> Self {
        self.scale(factor)
    }
    fn multiply(self, rhs: &Self) -> Self {
        self.multiply(rhs)
    }
    fn reciprocal(self) -> SpiceResult<Self> {
        self.reciprocal()
    }
    fn exp(self) -> Self {
        self.exp()
    }
    fn divide(self, rhs: &Self) -> SpiceResult<Self> {
        self.divide(rhs)
    }
    fn divide_constant(mut self, divisor: Real) -> Self {
        self.value = self.value.divide_constant(divisor);
        self.gradient = self
            .gradient
            .map(|entry| entry.map(|gradient| gradient.divide_constant(divisor)));
        self
    }
    fn tanh(self) -> SpiceResult<Self> {
        let value = self.value.value().tanh();
        if self.value.value().is_infinite() || (-2. * self.value.value().abs()).exp() == 0. {
            return Ok(Self::constant(self.value.order(), value));
        }
        let slope = tanh_slope(&self.value)?;
        let value = self.value.compose_derivative(value, &slope);
        Ok(self.lift(value, &slope))
    }
    fn atan_scaled(self, width: Real) -> SpiceResult<Self> {
        let value = self.value.value().atan2(width);
        if self.value.value().is_infinite() {
            return Ok(Self::constant(self.value.order(), value));
        }
        let scale = self.value.value().abs().max(width);
        let x = self.value.clone().divide_constant(scale);
        let w = width / scale;
        let norm = x.multiply(&x).offset(w * w);
        let slope = norm.reciprocal()?.scale(w).divide_constant(scale);
        let value = self.value.compose_derivative(value, &slope);
        Ok(self.lift(value, &slope))
    }
    fn check(&self) -> SpiceResult<()> {
        self.value.check()?;
        for gradient in self.gradient.iter().flatten() {
            gradient.check()?;
        }
        Ok(())
    }
}

/// Stable slope germ for tanh, including saturated values whose rounded
/// constant tanh is +/-1 while the physical derivative is still nonzero.
fn tanh_slope(input: &Jet) -> SpiceResult<Jet> {
    let sign = if input.value().is_sign_negative() {
        -1.
    } else {
        1.
    };
    let tail = input.clone().scale(-2. * sign).exp();
    let denom = tail.clone().offset(1.);
    tail.scale(4.).divide(&denom.multiply(&denom))
}

/// Stable scalar derivative even when tanh rounds to +/-1.
pub(crate) fn sech_squared(argument: Real) -> Real {
    let tail = (-2. * argument.abs()).exp();
    4. * tail / (1. + tail).powi(2)
}
