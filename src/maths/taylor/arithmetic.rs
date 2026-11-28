//! Shared constitutive arithmetic for the scalar and physical Taylor paths.
//!
//! The scalar instantiation preserves the original floating-point operations.
//! The differential instantiation carries the full time jet and local partial
//! jets; no finite differences or separately maintained value law are needed.
use std::{cell::RefCell, cmp::Ordering};

use super::{Constitutive, Jet, differential::Differential};
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
    fn with_partials<const N: usize>(self, _inputs: [&Self; N], _partials: [&Self; N]) -> Self {
        self
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Self::exp_m1(self)
    }

    #[inline]
    fn ln_1p(self) -> SpiceResult<Self> {
        Ok(Self::ln_1p(self))
    }

    #[inline]
    fn ln(self) -> SpiceResult<Self> {
        Ok(Self::ln(self))
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

    fn with_partials<const M: usize>(self, inputs: [&Self; M], partials: [&Self; M]) -> Self {
        let mut result = Self::constant(self.value.order(), self.primal());
        for (input, partial) in inputs.into_iter().zip(partials) {
            let contribution = input.value.compose_derivative(0., &partial.value);
            result = result.add(&input.clone().lift(contribution, &partial.value));
        }
        result
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
    fn ln(self) -> SpiceResult<Self> {
        let value = self.value.ln()?;
        let slope = self.value.reciprocal()?;
        Ok(self.lift(value, &slope))
    }
}

/// An actual bias-dependent branch comparison in a constitutive evaluation.
/// The caller publishes these in executed order as device-local guards.
pub(crate) struct BranchComparison {
    pub(crate) argument: Jet,
    pub(crate) boundary: Real,
}

/// Differential arithmetic with a disposable branch observer. The scalar
/// arithmetic and ordinary jet arithmetic retain their existing layouts.
/// Numerical rationalization uses primal(), which does not record an event.
#[derive(Clone)]
pub(crate) struct ObservedDifferential<'a, const N: usize> {
    pub(crate) inner: Differential<N>,
    trace: Option<&'a RefCell<Vec<BranchComparison>>>,
}

impl<'a, const N: usize> ObservedDifferential<'a, N> {
    pub(crate) const fn new(
        inner: Differential<N>,
        trace: Option<&'a RefCell<Vec<BranchComparison>>>,
    ) -> Self {
        Self { inner, trace }
    }
}

impl<const N: usize> Constitutive for ObservedDifferential<'_, N> {
    fn constant(&self, value: Real) -> Self {
        Self::new(Constitutive::constant(&self.inner, value), self.trace)
    }

    fn primal(&self) -> Real {
        self.inner.primal()
    }

    fn compare(&self, boundary: Real) -> Option<Ordering> {
        if let Some(trace) = self.trace {
            trace.borrow_mut().push(BranchComparison {
                argument: self.inner.value.clone(),
                boundary,
            });
        }
        self.inner.compare(boundary)
    }

    fn add(self, rhs: &Self) -> Self {
        Self::new(self.inner.add(&rhs.inner), self.trace)
    }

    fn midpoint(self, rhs: &Self) -> Self {
        Self::new(self.inner.midpoint(&rhs.inner), self.trace)
    }

    fn multiply(self, rhs: &Self) -> Self {
        Self::new(self.inner.multiply(&rhs.inner), self.trace)
    }

    fn divide(self, rhs: &Self) -> SpiceResult<Self> {
        Ok(Self::new(self.inner.divide(&rhs.inner)?, self.trace))
    }

    fn scale(self, factor: Real) -> Self {
        Self::new(self.inner.scale(factor), self.trace)
    }

    fn offset(self, value: Real) -> Self {
        Self::new(self.inner.offset(value), self.trace)
    }

    fn mul_add(self, rhs: &Self, add: &Self) -> Self {
        Self::new(self.inner.mul_add(&rhs.inner, &add.inner), self.trace)
    }

    fn sqrt(self) -> SpiceResult<Self> {
        Ok(Self::new(self.inner.sqrt()?, self.trace))
    }

    fn exp(self) -> Self {
        Self::new(self.inner.exp(), self.trace)
    }

    fn exp_with_primal(self, value: Real) -> Self {
        Self::new(self.inner.exp_with_primal(value), self.trace)
    }

    fn compose_derivative(self, value: Real, slope: &Self) -> Self {
        Self::new(
            self.inner.compose_derivative(value, &slope.inner),
            self.trace,
        )
    }

    fn with_partials<const M: usize>(self, inputs: [&Self; M], partials: [&Self; M]) -> Self {
        Self::new(
            self.inner
                .with_partials(inputs.map(|v| &v.inner), partials.map(|v| &v.inner)),
            self.trace,
        )
    }

    fn exp_m1(self) -> Self {
        Self::new(self.inner.exp_m1(), self.trace)
    }

    fn ln_1p(self) -> SpiceResult<Self> {
        Ok(Self::new(self.inner.ln_1p()?, self.trace))
    }

    fn ln(self) -> SpiceResult<Self> {
        Ok(Self::new(self.inner.ln()?, self.trace))
    }
}
