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

    pub(crate) fn from_coefficients(coefficients: Vec<Real>) -> SpiceResult<Self> {
        if coefficients.is_empty() || coefficients.len() > MAX_ORDER + 1 {
            return Err(SpiceError::circuit("invalid local Taylor order"));
        }
        let jet = Self { coefficients };
        jet.check()?;
        Ok(jet)
    }

    pub(crate) fn coefficients(&self) -> &[Real] {
        &self.coefficients
    }

    pub(crate) fn value(&self) -> Real {
        self.coefficients[0]
    }

    pub(crate) const fn order(&self) -> usize {
        self.coefficients.len() - 1
    }

    pub(crate) fn is_constant(&self) -> bool {
        self.coefficients[1..].iter().all(|&value| value == 0.)
    }

    pub(crate) fn is_zero(&self) -> bool {
        self.coefficients.iter().all(|&value| value == 0.)
    }

    /// Compare the local right-hand germ to a scalar boundary without
    /// perturbing the constant value (which might be one ulp from the knot).
    pub(crate) fn compare(&self, boundary: Real) -> Option<std::cmp::Ordering> {
        let constant = self.value().partial_cmp(&boundary)?;
        if constant != std::cmp::Ordering::Equal {
            return Some(constant);
        }
        Some(self.direction())
    }

    pub(crate) fn direction(&self) -> std::cmp::Ordering {
        self.coefficients[1..]
            .iter()
            .find(|&&value| value != 0.)
            .map_or(std::cmp::Ordering::Equal, |value| value.total_cmp(&0.))
    }

    /// Sign on the right-hand germ, including a zero constant coefficient.
    pub(crate) fn sign(&self) -> std::cmp::Ordering {
        self.compare(0.).unwrap_or(std::cmp::Ordering::Equal)
    }

    pub(crate) fn check(&self) -> SpiceResult<()> {
        if self.coefficients.iter().any(|value| !value.is_finite()) {
            Err(SpiceError::circuit("nonfinite local Taylor coefficients"))
        } else {
            Ok(())
        }
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
        if self.is_constant() {
            return rhs.clone().scale(self.value());
        }
        if rhs.is_constant() {
            return self.clone().scale(rhs.value());
        }
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

    /// Formal division, with a nonzero constant denominator.
    pub(crate) fn divide(&self, rhs: &Self) -> SpiceResult<Self> {
        let leading = rhs.value();
        if leading == 0. {
            return Err(SpiceError::circuit("singular Taylor denominator"));
        }
        let mut result = Self::constant(self.order(), 0.);
        for n in 0..=self.order() {
            let numerator = rhs.coefficients[1..=n]
                .iter()
                .zip(result.coefficients[..n].iter().rev())
                .fold(self.coefficients[n], |sum, (a, b)| (-a).mul_add(*b, sum));
            result.coefficients[n] = numerator / leading;
        }
        result.check()?;
        Ok(result)
    }

    pub(crate) fn reciprocal(&self) -> SpiceResult<Self> {
        Self::constant(self.order(), 1.).divide(self)
    }

    /// Integrate the chain rule from a supplied outer value and slope jet.
    pub(crate) fn compose_derivative(&self, value: Real, slope: &Self) -> Self {
        let mut result = Self::constant(self.order(), value);
        for n in 1..=self.order() {
            let mut degree = 0.;
            let integral = self.coefficients[1..=n]
                .iter()
                .zip(slope.coefficients[..n].iter().rev())
                .fold(0., |sum, (x, d)| {
                    degree += 1.;
                    (degree * x).mul_add(*d, sum)
                });
            result.coefficients[n] = integral / degree;
        }
        result
    }

    pub(crate) fn ln(&self) -> SpiceResult<Self> {
        if self.value() <= 0. {
            return Err(SpiceError::circuit(
                "logarithm has no regular Taylor expansion at this value",
            ));
        }
        Ok(self.compose_derivative(self.value().ln(), &self.reciprocal()?))
    }

    /// Integer powers at zero need no logarithm or singular derivative.
    pub(crate) fn integer_power(&self, exponent: i64) -> SpiceResult<Self> {
        let mut remaining = exponent.unsigned_abs();
        let mut base = if exponent < 0 {
            self.reciprocal()?
        } else {
            self.clone()
        };
        let mut result = Self::constant(self.order(), 1.);
        while remaining != 0 {
            if remaining & 1 != 0 {
                result = result.multiply(&base);
            }
            remaining >>= 1;
            if remaining != 0 {
                base = base.multiply(&base);
            }
        }
        result.check()?;
        Ok(result)
    }

    pub(crate) fn power(&self, exponent: &Self) -> SpiceResult<Self> {
        if exponent.is_constant()
            && let Some(integer) =
                crate::primitives::numeric::exact_signed_integer::<i64>(exponent.value())
        {
            return self.integer_power(integer);
        }
        let result = self.ln()?.multiply(exponent).exp();
        result.check()?;
        Ok(result)
    }

    pub(crate) fn sqrt(&self) -> SpiceResult<Self> {
        if self.value() < 0. {
            return Err(SpiceError::circuit("negative Taylor square root"));
        }
        if self.value() == 0. {
            if self.is_constant() {
                return Ok(self.clone());
            }
            // At a zero root the missing higher coefficients of the input
            // can affect even the retained output order. Do not manufacture
            // an analytic jet by assuming uncomputed coefficients vanish.
            return Err(SpiceError::circuit(
                "square root at a singular Taylor expansion point",
            ));
        }
        let root = self.value().sqrt();
        let mut result = Self::constant(self.order(), root);
        for n in 1..=self.order() {
            let remainder = result.coefficients[1..n]
                .iter()
                .zip(result.coefficients[1..n].iter().rev())
                .fold(self.coefficients[n], |sum, (a, b)| (-a).mul_add(*b, sum));
            result.coefficients[n] = (remainder / root) * 0.5;
        }
        result.check()?;
        Ok(result)
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
        let (sin, cos) = self.value().sin_cos();
        self.trig_pair(sin, cos, -1.)
    }

    pub(crate) fn sinh_cosh(&self) -> (Self, Self) {
        self.trig_pair(self.value().sinh(), self.value().cosh(), 1.)
    }

    fn trig_pair(&self, first: Real, second: Real, sign: Real) -> (Self, Self) {
        let order = self.order();
        let mut s = Self::constant(order, first);
        let mut c = Self::constant(order, second);
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
                cv = (sign * derivative).mul_add(sn, cv);
            }
            s.coefficients[n] = sv / weight;
            c.coefficients[n] = cv / weight;
        }
        (s, c)
    }

    /// The hyperbolic tangent recurrence avoids large exp intermediates.
    pub(crate) fn tanh(&self) -> Self {
        let mut result = Self::constant(self.order(), self.value().tanh());
        let mut slope = vec![0.; self.order()];
        for n in 1..=self.order() {
            let degree = n - 1;
            let square = result.coefficients[..=degree]
                .iter()
                .zip(result.coefficients[..=degree].iter().rev())
                .fold(0., |sum, (a, b)| a.mul_add(*b, sum));
            slope[degree] = if degree == 0 { 1. - square } else { -square };
            let mut weight = 0.;
            let derivative = self.coefficients[1..=n]
                .iter()
                .zip(slope[..n].iter().rev())
                .fold(0., |sum, (a, b)| {
                    weight += 1.;
                    (weight * a).mul_add(*b, sum)
                });
            result.coefficients[n] = derivative / weight;
        }
        result
    }

    pub(crate) fn with_value(mut self, value: Real) -> Self {
        self.coefficients[0] = value;
        self
    }

    pub(crate) fn finish(self) -> SpiceResult<Vec<Real>> {
        self.check()?;
        Ok(self.coefficients)
    }
}
