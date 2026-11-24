//! Conditioned finite reductions shared by constitutive derivatives and nonlinear stamps.

/// Ordered finite product sum with recovery for severe cancellation or
/// range loss. The Newton residual is the difference between a physical
/// current and its linearization; losing the small remainder changes the
/// RHS even when the individual branch current and slope are accurate.
#[inline]
pub(super) fn residual<const N: usize>(terms: [(f64, f64); N]) -> f64 {
    let mut products = terms.into_iter().map(|(a, b)| a * b);
    let Some(first) = products.next() else {
        return 0.;
    };
    let (ordinary, magnitude) = products.fold((first, first.abs()), |(sum, magnitude), term| {
        (sum + term, magnitude + term.abs())
    });
    if ordinary.is_normal() && ordinary.abs() > 0.125 * magnitude {
        return ordinary;
    }
    if magnitude == 0. && terms.iter().all(|&(a, b)| a == 0. || b == 0.) {
        return ordinary;
    }
    recover_residual(terms, ordinary)
}

#[cold]
fn recover_residual<const N: usize>(terms: [(f64, f64); N], ordinary: f64) -> f64 {
    if !terms.iter().all(|&(a, b)| a.is_finite() && b.is_finite()) {
        return ordinary;
    }
    let mut accumulator = crate::primitives::ExactSum::new();
    for (a, b) in terms {
        accumulator.add_product(a, b);
    }
    let result = accumulator.rounded_sum();
    if result == 0. && ordinary == 0. {
        ordinary
    } else {
        result
    }
}

pub(super) fn sum<const N: usize>(terms: [f64; N]) -> f64 {
    residual(terms.map(|term| (term, 1.)))
}

/// Mix two already-grouped current/charge contributions. In the common
/// unit-multiplier case this is just a correctly rounded addition. If
/// one multiplier is ±1, one FMA rounds the entire two-product sum;
/// otherwise the adaptive residual kernel handles cancellation.
#[inline]
#[expect(
    clippy::float_cmp,
    reason = "only exact unit multipliers admit an exact sign change"
)]
#[expect(
    clippy::suboptimal_flops,
    reason = "unit weights need only an exact sign change and correctly rounded addition; avoid generic software FMA"
)]
pub(super) fn mixed(first: (f64, f64), second: (f64, f64)) -> f64 {
    let (a, x) = first;
    let (b, y) = second;
    if a.abs() == 1. && b.abs() == 1. {
        a * x + b * y
    } else if a.abs() == 1. && a.is_finite() && x.is_finite() && b.is_finite() && y.is_finite() {
        b.mul_add(y, a * x)
    } else if b.abs() == 1. && a.is_finite() && x.is_finite() && b.is_finite() && y.is_finite() {
        a.mul_add(x, b * y)
    } else {
        residual([first, second])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonlinear_residual_recovers_cancellation_and_product_rounding() {
        assert_eq!(sum([1e16, 1., -1e16]), 1.);
        let epsilon = f64::EPSILON;
        assert_eq!(
            residual([(-1., 1.), (1. + epsilon, 1. - epsilon)]),
            -epsilon * epsilon
        );
        assert_eq!(residual([(1e308, 2.), (-1e308, 2.), (1., 1.)]), 1.);
        assert_eq!(sum([-0., -0.]).to_bits(), (-0_f64).to_bits());
        assert!(residual([(f64::INFINITY, 0.), (1., 1.)]).is_nan());
        assert_eq!(residual([(f64::INFINITY, 1.), (1., 1.)]), f64::INFINITY);
    }
}
