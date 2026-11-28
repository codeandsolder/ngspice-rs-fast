//! Shared constitutive arithmetic for the scalar and physical Taylor paths.
//!
//! The scalar instantiation preserves the original floating-point operations.
//! The differential instantiation carries the full time jet and local partial
//! jets; no finite differences or separately maintained value law are needed.
use std::cmp::Ordering;

use super::{Constitutive, differential::Differential};
use crate::primitives::{Real, SpiceResult};

impl Constitutive for Real {
    #[inline]
    fn constant(&self, value: Real) -> Self {
        value
    }

    #[inline]
    fn primal(&self) -> Real {
        *self
    }

    #[inline]
    fn compare(&self, boundary: Real) -> Option<Ordering> {
        self.partial_cmp(&boundary)
    }

    #[inline]
    fn add(self, rhs: &Self) -> Self {
        self + rhs
    }

    #[inline]
    fn midpoint(self, rhs: &Self) -> Self {
        Self::midpoint(self, *rhs)
    }

    #[inline]
    fn multiply(self, rhs: &Self) -> Self {
        self * rhs
    }

    #[inline]
    fn divide(self, rhs: &Self) -> SpiceResult<Self> {
        Ok(self / rhs)
    }

    #[inline]
    fn scale(self, factor: Real) -> Self {
        self * factor
    }

    #[inline]
    fn offset(self, value: Real) -> Self {
        self + value
    }

    #[inline]
    fn mul_add(self, rhs: &Self, add: &Self) -> Self {
        Self::mul_add(self, *rhs, *add)
    }

    #[inline]
    fn sqrt(self) -> SpiceResult<Self> {
        Ok(Self::sqrt(self))
    }

    #[inline]
    fn exp(self) -> Self {
        Self::exp(self)
    }

    #[inline]
    fn exp_with_primal(self, value: Real) -> Self {
        value
    }

    #[inline]
    fn compose_derivative(self, value: Real, _slope: &Self) -> Self {
        value
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Self::exp_m1(self)
    }

    #[inline]
    fn ln_1p(self) -> SpiceResult<Self> {
        Ok(Self::ln_1p(self))
    }
}

impl<const N: usize> Constitutive for Differential<N> {
    fn constant(&self, value: Real) -> Self {
        Self::constant(self.value.order(), value)
    }

    fn primal(&self) -> Real {
        self.value.value()
    }

    fn compare(&self, boundary: Real) -> Option<Ordering> {
        self.value.compare(boundary)
    }

    fn add(self, rhs: &Self) -> Self {
        Self::add(self, rhs)
    }

    fn midpoint(self, rhs: &Self) -> Self {
        let constant = self.primal().midpoint(rhs.primal());
        let mut result = self.scale(0.5).add(&rhs.clone().scale(0.5));
        result.value = result.value.with_value(constant);
        result
    }

    fn multiply(self, rhs: &Self) -> Self {
        Self::multiply(self, rhs)
    }

    fn divide(self, rhs: &Self) -> SpiceResult<Self> {
        Self::divide(self, rhs)
    }

    fn scale(self, factor: Real) -> Self {
        Self::scale(self, factor)
    }

    fn offset(self, value: Real) -> Self {
        Self::offset(self, value)
    }

    fn mul_add(self, rhs: &Self, add: &Self) -> Self {
        let constant = self.primal().mul_add(rhs.primal(), add.primal());
        let mut result = self.multiply(rhs).add(add);
        result.value = result.value.with_value(constant);
        result
    }

    fn sqrt(self) -> SpiceResult<Self> {
        Self::sqrt(self)
    }

    fn exp(self) -> Self {
        Self::exp(self)
    }

    fn exp_with_primal(self, value: Real) -> Self {
        let value = self.value.exp_with_value(value);
        self.lift(value.clone(), &value)
    }

    fn compose_derivative(self, value: Real, slope: &Self) -> Self {
        let germ = self.value.compose_derivative(value, &slope.value);
        self.lift(germ, &slope.value)
    }

    fn exp_m1(self) -> Self {
        let slope = self.value.exp();
        let value = slope.clone().with_value(self.primal().exp_m1());
        self.lift(value, &slope)
    }

    fn ln_1p(self) -> SpiceResult<Self> {
        let slope = self.value.clone().offset(1.).reciprocal()?;
        let value = self.value.compose_derivative(self.primal().ln_1p(), &slope);
        value.check()?;
        Ok(self.lift(value, &slope))
    }
}
