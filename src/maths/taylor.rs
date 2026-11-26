//! Truncated Taylor coefficients in a dimensionless local time coordinate.
//!
//! Coefficient k is the k-th derivative divided by k!, multiplied by the
//! physical time scale to the k-th power. Products and elementary functions
//! use coefficient recurrences, not differences of nearly equal samples.

use crate::primitives::{Real, SpiceError, SpiceResult};

/// The resource ceiling for a single local Taylor expansion.
pub const MAX_ORDER: usize = 64;

#[derive(Debug, Clone)]
pub(crate) struct Jet {
    coefficients: Vec<Real>,
}

impl Jet {
    pub(crate) fn constant(order: usize, value: Real) -> Self {
        let mut coefficients = vec![0.; order + 1];
        coefficients[0] = value;
        Self { coefficients }
    }

    pub(crate) fn affine(order: usize, value: Real, slope: Real) -> Self {
        let mut jet = Self::constant(order, value);
        if order != 0 {
            jet.coefficients[1] = slope;
        }
        jet
    }

    pub(crate) fn scale(mut self, factor: Real) -> Self {
        for value in &mut self.coefficients {
            *value *= factor;
        }
        self
    }

    pub(crate) fn offset(mut self, value: Real) -> Self {
        self.coefficients[0] += value;
        self
    }

    pub(crate) fn add(mut self, rhs: &Self) -> Self {
        for (a, b) in self.coefficients.iter_mut().zip(&rhs.coefficients) {
            *a += b;
        }
        self
    }

    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        let order = self.coefficients.len() - 1;
        let mut product = Self::constant(order, 0.);
        for (n, coefficient) in product.coefficients.iter_mut().enumerate() {
            *coefficient = self.coefficients[..=n]
                .iter()
                .zip(rhs.coefficients[..=n].iter().rev())
                .fold(0., |sum, (a, b)| a.mul_add(*b, sum));
        }
        product
    }

    /// From `y' = x' y`, `n y_n = sum_(k=1..n) k x_k y_(n-k)`.
    pub(crate) fn exp(&self) -> Self {
        let mut result = Self::constant(self.coefficients.len() - 1, self.coefficients[0].exp());
        for n in 1..result.coefficients.len() {
            let mut weight = 0.;
            let sum = self.coefficients[1..=n]
                .iter()
                .zip(result.coefficients[..n].iter().rev())
                .fold(0., |sum, (x, y)| {
                    weight += 1.;
                    (weight * x).mul_add(*y, sum)
                });
            result.coefficients[n] = sum / weight;
        }
        result
    }

    /// Coupled recurrences from s' = x' c and c' = -x' s.
    pub(crate) fn sin_cos(&self) -> (Self, Self) {
        let (sin, cos) = self.coefficients[0].sin_cos();
        let order = self.coefficients.len() - 1;
        let mut s = Self::constant(order, sin);
        let mut c = Self::constant(order, cos);
        for n in 1..=order {
            let mut sv = 0.;
            let mut cv = 0.;
            let mut weight = 0.;
            for (&x, (&sn, &cn)) in self.coefficients[1..=n].iter().zip(
                s.coefficients[..n]
                    .iter()
                    .rev()
                    .zip(c.coefficients[..n].iter().rev()),
            ) {
                // The iterator covers consecutive Taylor degrees starting at 1.
                weight += 1.;
                let derivative = weight * x;
                sv = derivative.mul_add(cn, sv);
                cv = (-derivative).mul_add(sn, cv);
            }
            s.coefficients[n] = sv / weight;
            c.coefficients[n] = cv / weight;
        }
        (s, c)
    }

    pub(crate) fn with_value(mut self, value: Real) -> Self {
        self.coefficients[0] = value;
        self
    }

    pub(crate) fn finish(self) -> SpiceResult<Vec<Real>> {
        if self.coefficients.iter().any(|value| !value.is_finite()) {
            return Err(SpiceError::circuit("nonfinite local Taylor coefficients"));
        }
        Ok(self.coefficients)
    }
}
