//! Stack-only first partials in an independently chosen local basis.
//! The value may itself be a physical Taylor germ. The auxiliary partials
//! then retain the reference Newton stamp's local coordinate convention
//! without allocating another series or replacing the physical derivative.

use super::{Constitutive, Local};
use crate::primitives::{Real, SpiceResult};
use std::cmp::Ordering;

impl<S: Constitutive, const N: usize> Local<S, N> {
    pub(crate) const fn new(value: S, partials: [Real; N]) -> Self {
        Self { value, partials }
    }

    fn lifted(self, value: S, slope: Real) -> Self {
        Self::new(
            value,
            self.partials.map(|p| if p == 0. { 0. } else { p * slope }),
        )
    }
}

impl<S: Constitutive, const N: usize> Constitutive for Local<S, N> {
    fn constant(&self, value: Real) -> Self {
        Self::new(self.value.constant(value), [0.; N])
    }

    fn primal(&self) -> Real {
        self.value.primal()
    }

    fn compare(&self, boundary: Real) -> Option<Ordering> {
        self.value.compare(boundary)
    }

    fn add(self, rhs: &Self) -> Self {
        Self::new(
            self.value.add(&rhs.value),
            std::array::from_fn(|i| self.partials[i] + rhs.partials[i]),
        )
    }

    fn midpoint(self, rhs: &Self) -> Self {
        Self::new(
            self.value.midpoint(&rhs.value),
            std::array::from_fn(|i| self.partials[i].midpoint(rhs.partials[i])),
        )
    }

    fn multiply(self, rhs: &Self) -> Self {
        let left = self.primal();
        let right = rhs.primal();
        let partials =
            std::array::from_fn(|i| self.partials[i].mul_add(right, rhs.partials[i] * left));
        Self::new(self.value.multiply(&rhs.value), partials)
    }

    fn divide(self, rhs: &Self) -> SpiceResult<Self> {
        let right = rhs.primal();
        let value = self.value.divide(&rhs.value)?;
        let result = value.primal();
        let partials =
            std::array::from_fn(|i| (-result).mul_add(rhs.partials[i], self.partials[i]) / right);
        Ok(Self::new(value, partials))
    }

    fn scale(self, factor: Real) -> Self {
        Self::new(self.value.scale(factor), self.partials.map(|p| p * factor))
    }

    fn offset(self, value: Real) -> Self {
        Self::new(self.value.offset(value), self.partials)
    }

    fn mul_add(self, rhs: &Self, add: &Self) -> Self {
        let left = self.primal();
        let right = rhs.primal();
        let partials = std::array::from_fn(|i| {
            self.partials[i].mul_add(right, rhs.partials[i].mul_add(left, add.partials[i]))
        });
        Self::new(self.value.mul_add(&rhs.value, &add.value), partials)
    }

    fn sqrt(self) -> SpiceResult<Self> {
        let value = self.value.clone().sqrt()?;
        let slope = 0.5 / value.primal();
        Ok(self.lifted(value, slope))
    }

    fn exp(self) -> Self {
        let value = self.value.clone().exp();
        let slope = value.primal();
        self.lifted(value, slope)
    }

    fn exp_with_primal(self, value: Real) -> Self {
        let law = self.value.clone().exp_with_primal(value);
        self.lifted(law, value)
    }

    fn compose_derivative(self, value: Real, slope: &Self) -> Self {
        let law = self.value.clone().compose_derivative(value, &slope.value);
        self.lifted(law, slope.primal())
    }

    fn with_partials<const M: usize>(self, inputs: [&Self; M], partials: [&Self; M]) -> Self {
        let gradient = std::array::from_fn(|i| {
            inputs
                .iter()
                .zip(partials)
                .fold(0., |sum, (input, partial)| {
                    partial.primal().mul_add(input.partials[i], sum)
                })
        });
        let value = self
            .value
            .with_partials(inputs.map(|v| &v.value), partials.map(|v| &v.value));
        Self::new(value, gradient)
    }

    fn exp_m1(self) -> Self {
        let slope = self.primal().exp();
        let value = self.value.clone().exp_m1();
        self.lifted(value, slope)
    }

    fn ln_1p(self) -> SpiceResult<Self> {
        let slope = (1. + self.primal()).recip();
        let value = self.value.clone().ln_1p()?;
        Ok(self.lifted(value, slope))
    }

    fn ln(self) -> SpiceResult<Self> {
        let slope = self.primal().recip();
        let value = self.value.clone().ln()?;
        Ok(self.lifted(value, slope))
    }
}
