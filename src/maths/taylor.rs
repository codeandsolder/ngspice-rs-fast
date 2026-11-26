//! Truncated ordinary Taylor series in a dimensionless local time.
//!
//! Coefficients are derivatives divided by factorial, with the time
//! scale absorbed into the independent variable. Arithmetic is exact
//! up to floating-point rounding and truncation; no high-order finite
//! difference of absolute circuit coordinates is needed.

use crate::primitives::{Real, SpiceError, SpiceResult};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Taylor(Vec<Real>);

fn failure(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "right-limit Taylor series".into(),
        message: message.into(),
    }
}

impl Taylor {
    pub(crate) fn new(coefficients: Vec<Real>) -> SpiceResult<Self> {
        if coefficients.is_empty() || coefficients.iter().any(|value| !value.is_finite()) {
            return Err(failure("nonfinite or empty coefficients"));
        }
        Ok(Self(coefficients))
    }

    pub(crate) fn constant(value: Real, order: usize) -> Self {
        let mut coefficients = vec![0.; order + 1];
        coefficients[0] = value;
        Self(coefficients)
    }

    pub(crate) fn affine(value: Real, slope: Real, order: usize) -> Self {
        let mut result = Self::constant(value, order);
        if let Some(first) = result.0.get_mut(1) {
            *first = slope;
        }
        result
    }

    pub(crate) fn order(&self) -> usize {
        self.0.len() - 1
    }

    pub(crate) fn coefficients(&self) -> &[Real] {
        &self.0
    }

    pub(crate) fn base(&self) -> Real {
        self.0[0]
    }

    pub(crate) fn with_base(mut self, value: Real) -> SpiceResult<Self> {
        self.0[0] = value;
        Self::new(self.0)
    }

    pub(crate) fn is_constant(&self) -> bool {
        self.0[1..].iter().all(|&value| value == 0.)
    }

    /// Sign in a sufficiently small right-hand open interval, including
    /// the first nonzero derivative when the base is on a switching surface.
    pub(crate) fn right_sign(&self) -> Real {
        self.0
            .iter()
            .find(|&&value| value != 0.)
            .map_or(0., |value| value.signum())
    }

    /// Direction of departure from the constant term.
    pub(crate) fn direction(&self) -> Real {
        self.0[1..]
            .iter()
            .find(|&&value| value != 0.)
            .map_or(0., |value| value.signum())
    }

    fn same_order(&self, other: &Self) -> SpiceResult<()> {
        if self.0.len() != other.0.len() {
            return Err(failure("incompatible series orders"));
        }
        Ok(())
    }

    pub(crate) fn add(&self, other: &Self) -> SpiceResult<Self> {
        self.same_order(other)?;
        Self::new(self.0.iter().zip(&other.0).map(|(&a, &b)| a + b).collect())
    }

    pub(crate) fn sub(&self, other: &Self) -> SpiceResult<Self> {
        self.same_order(other)?;
        Self::new(self.0.iter().zip(&other.0).map(|(&a, &b)| a - b).collect())
    }

    pub(crate) fn scale(&self, scale: Real) -> SpiceResult<Self> {
        Self::new(self.0.iter().map(|value| value * scale).collect())
    }

    pub(crate) fn mul(&self, other: &Self) -> SpiceResult<Self> {
        self.same_order(other)?;
        if self.is_constant() {
            return other.scale(self.base());
        }
        if other.is_constant() {
            return self.scale(other.base());
        }
        Self::new(
            (0..=self.order())
                .map(|n| {
                    self.0[..=n]
                        .iter()
                        .zip(other.0[..=n].iter().rev())
                        .fold(0., |sum, (&a, &b)| a.mul_add(b, sum))
                })
                .collect(),
        )
    }

    pub(crate) fn div(&self, denominator: &Self) -> SpiceResult<Self> {
        self.same_order(denominator)?;
        let base = denominator.base();
        if base == 0. {
            return Err(failure("singular Taylor quotient at the right limit"));
        }
        if denominator.is_constant() {
            // Direct division also works with subnormal denominators for
            // which forming the reciprocal would overflow.
            return Self::new(self.0.iter().map(|value| value / base).collect());
        }
        let mut result = Vec::with_capacity(self.0.len());
        for (n, &a) in self.0.iter().enumerate() {
            let remainder = denominator.0[1..=n]
                .iter()
                .zip(result.iter().rev())
                .fold(a, |sum, (&b, &q)| (-b).mul_add(q, sum));
            result.push(remainder / base);
        }
        Self::new(result)
    }

    /// Integrate y'(s) = g(x(s)) x'(s), using its known constant term.
    pub(crate) fn compose_derivative(&self, slope: &Self, base: Real) -> SpiceResult<Self> {
        self.same_order(slope)?;
        let mut result = Vec::with_capacity(self.0.len());
        result.push(base);
        for (n, degree) in (1..self.0.len()).zip(1_u32..) {
            let coefficient = self.0[1..=n]
                .iter()
                .zip(slope.0[..n].iter().rev())
                .zip(1_u32..)
                .fold(0., |sum, ((&x, &g), k)| (Real::from(k) * x).mul_add(g, sum));
            result.push(coefficient / Real::from(degree));
        }
        Self::new(result)
    }

    pub(crate) fn exp(&self, base: Real) -> SpiceResult<Self> {
        let mut result = Vec::with_capacity(self.0.len());
        result.push(base);
        for (n, degree) in (1..self.0.len()).zip(1_u32..) {
            let coefficient = self.0[1..=n]
                .iter()
                .zip(result.iter().rev())
                .zip(1_u32..)
                .fold(0., |sum, ((&x, &y), k)| (Real::from(k) * x).mul_add(y, sum));
            result.push(coefficient / Real::from(degree));
        }
        Self::new(result)
    }

    pub(crate) fn log(&self, base: Real) -> SpiceResult<Self> {
        self.compose_derivative(&Self::constant(1., self.order()).div(self)?, base)
    }

    pub(crate) fn sqrt(&self, base: Real) -> SpiceResult<Self> {
        if self.is_constant() {
            return Self::new(Self::constant(base, self.order()).0);
        }
        if base <= 0. {
            return Err(failure(
                "square root has no regular Taylor expansion at this point",
            ));
        }
        let mut result = Vec::with_capacity(self.0.len());
        result.push(base);
        for (n, &x) in self.0.iter().enumerate().skip(1) {
            let coefficient = result[1..n]
                .iter()
                .zip(result[1..n].iter().rev())
                .fold(x, |sum, (&a, &b)| (-a).mul_add(b, sum));
            result.push(coefficient / (2. * base));
        }
        Self::new(result)
    }

    /// Sine/cosine, or sinh/cosh, of the same analytic argument. The
    /// caller supplies the exact value-kernel constant terms.
    pub(crate) fn trig_pair(
        &self,
        sine: Real,
        cosine: Real,
        hyperbolic: bool,
    ) -> SpiceResult<(Self, Self)> {
        let mut s = Vec::with_capacity(self.0.len());
        let mut c = Vec::with_capacity(self.0.len());
        s.push(sine);
        c.push(cosine);
        let sign = if hyperbolic { 1. } else { -1. };
        for (n, degree) in (1..self.0.len()).zip(1_u32..) {
            let (a, b) = self.0[1..=n]
                .iter()
                .zip(c.iter().rev().zip(s.iter().rev()))
                .zip(1_u32..)
                .fold((0., 0.), |(a, b), ((&x, (&c, &s)), k)| {
                    let derivative = Real::from(k) * x;
                    (derivative.mul_add(c, a), derivative.mul_add(s, b))
                });
            s.push(a / Real::from(degree));
            c.push(sign * b / Real::from(degree));
        }
        Ok((Self::new(s)?, Self::new(c)?))
    }

    pub(crate) fn powi(&self, exponent: i32) -> SpiceResult<Self> {
        let mut result = Self::constant(1., self.order());
        let mut factor = self.clone();
        let mut power = exponent.unsigned_abs();
        while power != 0 {
            if power & 1 != 0 {
                result = result.mul(&factor)?;
            }
            power >>= 1;
            if power != 0 {
                factor = factor.mul(&factor)?;
            }
        }
        if exponent < 0 {
            Self::constant(1., self.order()).div(&result)
        } else {
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Taylor, expected: &[Real]) {
        assert_eq!(a.coefficients().len(), expected.len());
        for (actual, &expected) in a.coefficients().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-13, "{actual} != {expected}");
        }
    }

    #[test]
    fn formal_products_division_and_integer_powers() {
        let x = Taylor::affine(2., 3., 4);
        let square = x.mul(&x).unwrap();
        close(&square, &[4., 12., 9., 0., 0.]);
        close(&square.div(&x).unwrap(), &[2., 3., 0., 0., 0.]);
        close(&x.powi(3).unwrap(), &[8., 36., 54., 27., 0.]);
        let origin = Taylor::affine(0., 2., 4);
        close(&origin.powi(3).unwrap(), &[0., 0., 0., 8., 0.]);
        assert!(x.div(&origin).is_err());
        assert_eq!(origin.right_sign(), 1.);
    }

    #[test]
    fn exact_analytic_recurrences() {
        let x = Taylor::affine(0., 1., 5);
        let exp = x.exp(1.).unwrap();
        close(&exp, &[1., 1., 0.5, 1. / 6., 1. / 24., 1. / 120.]);
        close(&exp.log(0.).unwrap(), &[0., 1., 0., 0., 0., 0.]);
        let (s, c) = x.trig_pair(0., 1., false).unwrap();
        close(&s, &[0., 1., 0., -1. / 6., 0., 1. / 120.]);
        close(&c, &[1., 0., -0.5, 0., 1. / 24., 0.]);
        let (sh, ch) = x.trig_pair(0., 1., true).unwrap();
        close(&sh, &[0., 1., 0., 1. / 6., 0., 1. / 120.]);
        close(&ch, &[1., 0., 0.5, 0., 1. / 24., 0.]);
        let y = Taylor::affine(4., 2., 5);
        let root = y.sqrt(2.).unwrap();
        close(&root.mul(&root).unwrap(), y.coefficients());
    }

    #[test]
    fn nonlinear_composition_is_not_a_difference_of_absolute_values() {
        let x = Taylor::affine(1e12, 1e-8, 3);
        let y = x.powi(3).unwrap();
        assert!((y.coefficients()[1] / 3e16 - 1.).abs() < 1e-15);
        assert!((y.coefficients()[2] / 3e-4 - 1.).abs() < 1e-15);
        assert!((y.coefficients()[3] / 1e-24 - 1.).abs() < 1e-15);
    }
}
