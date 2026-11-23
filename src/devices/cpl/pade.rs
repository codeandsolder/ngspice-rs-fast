//! CPL's three-pole Padé rational-kernel primitive.
//!
//! The original coupled-line code fits each matrix-polynomial channel to
//! three poles in the inverse Laplace variable. Its Gaussian elimination,
//! cubic root solver and pole residues are algebraically identical to the
//! already native TXL solver. Reuse those tested functions, not a second
//! private numerical implementation.
//!
//! The leading coefficient represents the high-frequency direct term.
//! Coefficients 1..5 are relative to that leading term; the final scalar
//! is the required low-frequency limit. For a real Laplace frequency s:
//!
//!   Rational response = direct term plus the sum of all residue/pole fractions.
//!
//! A conjugate pair contributes twice the real part of one complex term.
//! An unstable fitted pole is a model error, never an exponential that can
//! grow without bound during accepted transient simulation.

use crate::devices::txl::pade::{Propagation, Term, residues, roots, solve_series};
use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

fn error(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "CPL cplsetup.c three-pole Padé".into(),
        message: message.into(),
    }
}

/// Bounded three-pole rational response with C-compatible storage for
/// either three real roots or one real root and one conjugate pair.
#[derive(Debug, Clone)]
pub struct CplPade {
    leading: Real,
    dc_limit: Real,
    poles: Option<Propagation<3, 1>>,
}

/// C's Neville polynomial interpolation at the asymptotic origin.
/// Fixed order eight avoids allocation while building model coefficients.
fn polynomial_at_zero(x: &[Real], y: &[Real]) -> SpiceResult<Real> {
    let order = x.len();
    let mut c = [0.; 8];
    let mut d = [0.; 8];
    c[..order].copy_from_slice(y);
    d[..order].copy_from_slice(y);
    let mut nearest = 0;
    for index in 1..order {
        if x[index].abs() < x[nearest].abs() {
            nearest = index;
        }
    }
    let mut value = y[nearest];
    for degree in 1..order {
        for row in 0..(order - degree) {
            let difference = x[row] - x[row + degree];
            if difference.partial_cmp(&0.) == Some(std::cmp::Ordering::Equal) {
                return Err(error("duplicate CPL Padé interpolation abscissae"));
            }
            let slope = (c[row + 1] - d[row]) / difference;
            d[row] = x[row + degree] * slope;
            c[row] = x[row] * slope;
        }
        if 2 * nearest < order - degree {
            value += c[nearest];
        } else {
            nearest -= 1;
            value += d[nearest];
        }
    }
    if !value.is_finite() {
        return Err(error("nonfinite CPL inverse-Laplace interpolation"));
    }
    Ok(value)
}

/// C's interpolation routine: strip the value at zero, divide the remaining samples by
/// their abscissa and repeat on one fewer sample to find each coefficient.
fn polynomial_coefficients(samples: &[(Real, Real); 8]) -> SpiceResult<[Real; 8]> {
    if samples
        .iter()
        .any(|(abscissa, value)| !abscissa.is_finite() || !value.is_finite())
    {
        return Err(error("nonfinite CPL inverse-Laplace sample"));
    }
    let equal =
        |left: Real, right: Real| left.partial_cmp(&right) == Some(std::cmp::Ordering::Equal);
    if !samples.iter().any(|&(abscissa, _)| equal(abscissa, 0.)) {
        return Err(error(
            "CPL Padé requires an exact high-frequency sample at inverse rate zero",
        ));
    }
    for left in 0..8 {
        for right in left + 1..8 {
            if equal(samples[left].0, samples[right].0) {
                return Err(error("repeated CPL inverse-Laplace frequency sample"));
            }
        }
    }
    let mut abscissae = samples.map(|(abscissa, _)| abscissa);
    let mut ordinates = samples.map(|(_, value)| value);
    let mut coefficients = [0.; 8];
    for (degree, coefficient) in coefficients.iter_mut().enumerate() {
        let count = 8 - degree;
        *coefficient = polynomial_at_zero(&abscissae[..count], &ordinates[..count])?;
        let mut nearest = 0;
        for index in 0..count {
            if abscissae[index].abs() < abscissae[nearest].abs() {
                nearest = index;
            }
            if !equal(abscissae[index], 0.) {
                ordinates[index] = (ordinates[index] - *coefficient) / abscissae[index];
            }
        }
        for index in nearest + 1..count {
            abscissae[index - 1] = abscissae[index];
            ordinates[index - 1] = ordinates[index];
        }
    }
    Ok(coefficients)
}

impl CplPade {
    /// Fit KSPICE's eight inverse-rate samples to the same three-pole
    /// rational form used by the convolution engine. The samples must
    /// include the exact high-frequency point at inverse-rate zero;
    /// finite nonzero abscissae may occur in any order.
    ///
    /// # Errors
    /// Invalid or repeated abscissae, polynomial overflow, unstable Padé
    /// roots or nonfinite coefficients.
    pub fn fit_samples(samples: &[(Real, Real); 8], dc_limit: Real) -> SpiceResult<Self> {
        let polynomial = polynomial_coefficients(samples)?;
        let leading = polynomial[0];
        if !leading.is_finite() || leading.partial_cmp(&0.) == Some(std::cmp::Ordering::Equal) {
            return Err(error("zero-leading CPL inverse-Laplace fit"));
        }
        let coefficients = std::array::from_fn(|i| polynomial[i + 1] / leading);
        Self::fit(leading, coefficients, dc_limit)
    }

    /// Fit the inverse-Laplace Maclaurin expansion to a three-pole rational
    /// with the physical zero-frequency limit. Both the leading coefficient
    /// and target limit are in the same physical units.
    ///
    /// # Errors
    /// Degenerate or nonfinite series, unsolvable cubic, unstable poles or
    /// nonfinite pole residues.
    pub fn fit(leading: Real, coefficients: [Real; 5], dc_limit: Real) -> SpiceResult<Self> {
        if !leading.is_finite()
            || !dc_limit.is_finite()
            || coefficients.iter().any(|value| !value.is_finite())
        {
            return Err(error("nonfinite CPL Padé series"));
        }
        // Identically constant and zero channels have no poles at all.
        // They are frequent for lossless models and uncoupled matrix entries;
        // a rank-deficient three-pole solve must not discard them.
        let equal =
            |left: Real, right: Real| left.partial_cmp(&right) == Some(std::cmp::Ordering::Equal);
        if coefficients
            .iter()
            .all(|&coefficient| equal(coefficient, 0.))
            && equal(dc_limit, leading)
        {
            return Ok(Self {
                leading,
                dc_limit,
                poles: None,
            });
        }
        if equal(leading, 0.) {
            return Err(error(
                "zero-leading CPL Padé channel requires zero coefficients",
            ));
        }
        let normalized_dc = dc_limit / leading;
        if !normalized_dc.is_finite() {
            return Err(error("unrepresentable normalized CPL DC kernel"));
        }
        let (denominator, numerator) = solve_series(coefficients, normalized_dc, 1e-28)?;
        let (roots, complex_pair) = roots(denominator, true)?;
        let relative_residues = residues(denominator, numerator, roots, complex_pair);
        let terms = std::array::from_fn(|index| Term {
            c: relative_residues[index] * leading,
            x: roots[index],
        });
        if terms
            .iter()
            .any(|term| !term.c.is_finite() || !term.x.is_finite())
            || terms[0].x >= 0.
            || terms[1].x >= 0.
            || (!complex_pair && terms[2].x >= 0.)
            || (complex_pair && terms[2].x <= 0.)
        {
            return Err(error("CPL Padé has nonfinite or unstable poles/residues"));
        }
        let poles = Propagation::from_c(terms, complex_pair);
        Ok(Self {
            leading,
            dc_limit,
            poles: Some(poles),
        })
    }

    /// Direct high-frequency coefficient (asymptotic value at infinity).
    #[must_use]
    pub const fn leading(&self) -> Real {
        self.leading
    }

    /// Physically required DC limit passed to the rational approximation.
    #[must_use]
    pub const fn dc_limit(&self) -> Real {
        self.dc_limit
    }

    /// Evaluate the rational kernel at a nonnegative real Laplace rate.
    /// This is used by numerical fitting validation, not an AC P-device.
    ///
    /// # Errors
    /// Invalid evaluation rate or nonfinite value.
    pub fn evaluate(&self, rate: Real) -> SpiceResult<Real> {
        if !rate.is_finite() || rate < 0. {
            return Err(error("CPL Padé requires finite nonnegative Laplace rate"));
        }
        let result = match &self.poles {
            None => self.leading,
            Some(Propagation::Real(terms)) => terms
                .iter()
                .fold(self.leading, |sum, term| sum + term.c / (rate - term.x)),
            Some(Propagation::Conjugate { real, pair }) => {
                let one = real[0].c / (rate - real[0].x);
                let conjugate = pair.residue / (Complex::real(rate) - pair.pole);
                2.0_f64.mul_add(conjugate.re, self.leading + one)
            }
        };
        if !result.is_finite() {
            return Err(error("CPL Padé evaluation overflowed"));
        }
        Ok(result)
    }
}
