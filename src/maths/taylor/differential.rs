//! Multivariate first partials whose values are univariate time jets.
//! The sparse local gradient retains only active constitutive coordinates.
use super::Jet;
use crate::primitives::{Real, SpiceResult};

#[derive(Clone)]
pub(crate) struct Differential<const N: usize> {
    pub(crate) value: Jet,
    pub(crate) gradient: [Option<Jet>; N],
}

fn sum(a: Option<Jet>, b: Option<Jet>) -> Option<Jet> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.add(&b)),
        (a, None) | (None, a) => a,
    }
}

impl<const N: usize> Differential<N> {
    pub(crate) fn constant(order: usize, value: Real) -> Self {
        Self {
            value: Jet::constant(order, value),
            gradient: std::array::from_fn(|_| None),
        }
    }

    pub(crate) fn variable(index: usize, value: Jet) -> SpiceResult<Self> {
        let mut gradient = std::array::from_fn(|_| None);
        let entry = gradient.get_mut(index).ok_or_else(|| {
            crate::primitives::SpiceError::circuit("invalid local Taylor coordinate")
        })?;
        *entry = Some(Jet::constant(value.order(), 1.));
        Ok(Self { value, gradient })
    }

    /// Apply a unary law with an already computed value and derivative germ.
    pub(crate) fn lift(self, value: Jet, slope: &Jet) -> Self {
        Self {
            value,
            gradient: self.gradient.map(|entry| entry.map(|g| g.multiply(slope))),
        }
    }

    pub(crate) fn add(mut self, rhs: &Self) -> Self {
        self.value = self.value.add(&rhs.value);
        for (lhs, rhs) in self.gradient.iter_mut().zip(&rhs.gradient) {
            *lhs = sum(lhs.take(), rhs.clone());
        }
        self
    }

    pub(crate) fn offset(mut self, value: Real) -> Self {
        self.value = self.value.offset(value);
        self
    }

    pub(crate) fn scale(mut self, factor: Real) -> Self {
        self.value = self.value.scale(factor);
        self.gradient = self
            .gradient
            .map(|entry| entry.filter(|_| factor != 0.).map(|g| g.scale(factor)));
        self
    }

    pub(crate) fn multiply(self, rhs: &Self) -> Self {
        let gradient = std::array::from_fn(|index| {
            sum(
                self.gradient[index]
                    .as_ref()
                    .map(|g| g.multiply(&rhs.value)),
                rhs.gradient[index]
                    .as_ref()
                    .map(|g| g.multiply(&self.value)),
            )
        });
        Self {
            value: self.value.multiply(&rhs.value),
            gradient,
        }
    }

    pub(crate) fn divide(self, rhs: &Self) -> SpiceResult<Self> {
        let value = self.value.divide(&rhs.value)?;
        let mut gradient = self.gradient;
        for (lhs, rhs_gradient) in gradient.iter_mut().zip(&rhs.gradient) {
            let correction = rhs_gradient.as_ref().map(|g| g.multiply(&value).scale(-1.));
            *lhs = sum(lhs.take(), correction)
                .map(|g| g.divide(&rhs.value))
                .transpose()?;
        }
        Ok(Self { value, gradient })
    }

    pub(crate) fn reciprocal(self) -> SpiceResult<Self> {
        Self::constant(self.value.order(), 1.).divide(&self)
    }

    pub(crate) fn sqrt(self) -> SpiceResult<Self> {
        let value = self.value.sqrt()?;
        if self.gradient.iter().all(Option::is_none) {
            return Ok(Self {
                value,
                gradient: self.gradient,
            });
        }
        let slope = value.reciprocal()?.scale(0.5);
        Ok(self.lift(value, &slope))
    }

    pub(crate) fn power(self, exponent: Real) -> SpiceResult<Self> {
        if exponent == 0. {
            return Ok(Self::constant(self.value.order(), 1.));
        }
        if exponent.to_bits() == 1_f64.to_bits() {
            return Ok(self);
        }
        let value = self
            .value
            .power(&Jet::constant(self.value.order(), exponent))?;
        let slope = self
            .value
            .power(&Jet::constant(self.value.order(), exponent - 1.))?
            .scale(exponent);
        Ok(self.lift(value, &slope))
    }

    pub(crate) fn exp(self) -> Self {
        let value = self.value.exp();
        self.lift(value.clone(), &value)
    }

    pub(crate) fn tan(self) -> SpiceResult<Self> {
        let (sin, cos) = self.value.sin_cos();
        let value = sin.divide(&cos)?;
        let slope = cos.multiply(&cos).reciprocal()?;
        Ok(self.lift(value, &slope))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_partial_series_follow_the_same_analytic_germ() -> SpiceResult<()> {
        type D = Differential<2>;
        let x = D::variable(0, Jet::affine(8, 2., 0.3))?;
        let y = D::variable(1, Jet::affine(8, 3., -0.2))?;
        let result = x.clone().multiply(&y).exp().divide(&x)?;
        let at = |jet: &Jet, t: Real| {
            jet.coefficients()
                .iter()
                .rev()
                .fold(0., |sum: Real, &coef| sum.mul_add(t, coef))
        };
        for t in [0., 0.1, -0.1, 0.25] {
            let a = 0.3_f64.mul_add(t, 2.);
            let b = (-0.2_f64).mul_add(t, 3.);
            let value = (a * b).exp() / a;
            let derivatives = [value * (b - a.recip()), value * a];
            assert!((at(&result.value, t) / value - 1.).abs() < 1e-11);
            for (gradient, expected) in result.gradient.iter().zip(derivatives) {
                let gradient = gradient.as_ref().unwrap();
                assert!((at(gradient, t) / expected - 1.).abs() < 1e-11);
            }
        }
        Ok(())
    }
}
