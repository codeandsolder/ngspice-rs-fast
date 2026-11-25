//! Bounded normalized Taylor coefficients for differentiated residuals.
//!
//! Coefficient k is the multiplier of u^k, with u a dimensionless local
//! time coordinate. No factorials or differences of sampled values are
//! needed. Coefficients at different time scales must not be mixed.

use crate::primitives::{Real, SpiceError, SpiceResult};
pub mod dae;

/// Maximum supported expansion order, bounding storage and convolution work.
pub const MAX_ORDER: usize = 32;

/// One finite truncated Taylor series with an explicit expansion order.
#[derive(Clone, Debug, PartialEq)]
pub struct Jet {
    coefficients: Vec<Real>,
}

fn invalid() -> SpiceError {
    SpiceError::circuit("invalid or nonrepresentable Taylor coefficients")
}

impl Jet {
    /// Construct a constant or affine series of the requested order.
    /// # Errors
    /// Excessive order or nonfinite coefficient.
    pub fn affine(order: usize, value: Real, slope: Real) -> SpiceResult<Self> {
        if order > MAX_ORDER || !value.is_finite() || (order != 0 && !slope.is_finite()) {
            return Err(invalid());
        }
        let mut coefficients = vec![0.; order + 1];
        coefficients[0] = value;
        if let Some(first) = coefficients.get_mut(1) {
            *first = slope;
        }
        Ok(Self { coefficients })
    }

    /// Construct a series from normalized coefficients in increasing order.
    /// # Errors
    /// Empty, excessive or nonfinite coefficients.
    pub fn from_coefficients(coefficients: Vec<Real>) -> SpiceResult<Self> {
        if coefficients.is_empty()
            || coefficients.len() > MAX_ORDER + 1
            || coefficients.iter().any(|x| !x.is_finite())
        {
            return Err(invalid());
        }
        Ok(Self { coefficients })
    }

    /// Normalized coefficients, starting with the expansion-point value.
    #[must_use]
    pub fn coefficients(&self) -> &[Real] {
        &self.coefficients
    }

    /// Add another series, requiring equal truncation orders.
    /// # Errors
    /// Mismatched order or nonfinite result.
    pub fn sum(&self, rhs: &Self) -> SpiceResult<Self> {
        self.compatible(rhs)?;
        Self::from_coefficients(
            self.coefficients
                .iter()
                .zip(&rhs.coefficients)
                .map(|(a, b)| a + b)
                .collect(),
        )
    }

    /// Multiply by a scalar.
    /// # Errors
    /// Nonfinite result.
    pub fn scaled(&self, scale: Real) -> SpiceResult<Self> {
        Self::from_coefficients(self.coefficients.iter().map(|x| x * scale).collect())
    }

    /// Add a scalar to the constant coefficient only.
    /// # Errors
    /// Nonfinite result.
    pub fn shifted(mut self, shift: Real) -> SpiceResult<Self> {
        self.coefficients[0] += shift;
        if self.coefficients[0].is_finite() {
            Ok(self)
        } else {
            Err(invalid())
        }
    }

    fn compatible(&self, rhs: &Self) -> SpiceResult<()> {
        if self.coefficients.len() == rhs.coefficients.len() {
            Ok(())
        } else {
            Err(invalid())
        }
    }

    /// Truncated Cauchy product, without subtractive finite differences.
    /// # Errors
    /// Mismatched order or nonfinite coefficient.
    pub fn product(&self, rhs: &Self) -> SpiceResult<Self> {
        self.compatible(rhs)?;
        let out = (0..self.coefficients.len())
            .map(|n| {
                (0..=n).fold(0., |acc, k| {
                    self.coefficients[k].mul_add(rhs.coefficients[n - k], acc)
                })
            })
            .collect();
        Self::from_coefficients(out)
    }

    /// Exponential via y' = x' y.
    /// # Errors
    /// Nonrepresentable result.
    pub fn exp(&self) -> SpiceResult<Self> {
        let mut out = vec![self.coefficients[0].exp()];
        for n in 1..self.coefficients.len() {
            let value = (1..=n).fold(0., |acc, k| {
                (index(k) / index(n) * self.coefficients[k]).mul_add(out[n - k], acc)
            });
            out.push(value);
        }
        Self::from_coefficients(out)
    }

    /// Sine and cosine via the coupled analytic derivative identities.
    /// # Errors
    /// Nonrepresentable result.
    pub fn sin_cos(&self) -> SpiceResult<(Self, Self)> {
        let (sin, cos) = self.coefficients[0].sin_cos();
        let mut s = vec![sin];
        let mut c = vec![cos];
        for n in 1..self.coefficients.len() {
            let (sn, cn) = (1..=n).fold((0., 0.), |(sa, ca), k| {
                let weight = index(k) / index(n) * self.coefficients[k];
                (
                    weight.mul_add(c[n - k], sa),
                    (-weight).mul_add(s[n - k], ca),
                )
            });
            s.push(sn);
            c.push(cn);
        }
        Ok((Self::from_coefficients(s)?, Self::from_coefficients(c)?))
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "Taylor indices are bounded by MAX_ORDER"
)]
const fn index(n: usize) -> Real {
    n as Real
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_of_quadratic_and_trigonometric_identity() {
        let x = Jet::from_coefficients(vec![0., 2., 3., 0., 0., 0.]).unwrap();
        let exp = x.exp().unwrap();
        let expected = [1., 2., 5., 22. / 3., 67. / 6., 199. / 15.];
        for (&value, reference) in exp.coefficients().iter().zip(expected) {
            assert!((value - reference).abs() < 1e-13, "{value} != {reference}");
        }
        let (s, c) = x.sin_cos().unwrap();
        let identity = s.product(&s).unwrap().sum(&c.product(&c).unwrap()).unwrap();
        assert!((identity.coefficients()[0] - 1.).abs() < 1e-14);
        assert!(identity.coefficients()[1..].iter().all(|x| x.abs() < 1e-13));
    }

    #[test]
    fn order_and_nonfinite_arithmetic_are_checked() {
        assert!(Jet::affine(MAX_ORDER + 1, 0., 0.).is_err());
        assert!(Jet::from_coefficients(vec![]).is_err());
        let a = Jet::affine(2, 1e300, 0.).unwrap();
        assert!(a.product(&a).is_err());
        assert!(a.sum(&Jet::affine(1, 0., 0.).unwrap()).is_err());
    }
}
