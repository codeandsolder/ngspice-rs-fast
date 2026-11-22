//! Distortion loads from second- and third-order Taylor monomials.
//! Evaluate each complex product once, with the mixing coefficients used by
//! ngspice's `dloadfns.c`.

use super::Taylor;
use crate::primitives::{Complex, Real};

/// The three controls' values in one kernel (zero for a missing control).
pub(super) type Values = [Complex; 3];

fn scaled(value: Complex, coefficient: Real) -> Complex {
    Complex::new(coefficient * value.re, coefficient * value.im)
}

fn quadratic(t: &Taylor, product: impl Fn(usize, usize) -> Complex) -> Complex {
    t.second()
        .into_iter()
        .fold(Complex::ZERO, |sum, (i, j, c)| {
            sum + scaled(product(i, j), c)
        })
}

fn cubic(t: &Taylor, product: impl Fn(usize, usize, usize) -> Complex) -> Complex {
    t.third()
        .into_iter()
        .fold(Complex::ZERO, |sum, (i, j, k, c)| {
            sum + scaled(product(i, j, k), c)
        })
}

/// Second harmonic, from the quadratic Taylor terms of the first kernel.
pub(super) fn two_f1(t: &Taylor, h1: Values) -> Complex {
    quadratic(t, |i, j| h1[i] * h1[j])
}

/// Third harmonic: mixing the first two kernels plus pure cubic terms.
pub(super) fn three_f1(t: &Taylor, h1: Values, h2: Values) -> Complex {
    quadratic(t, |i, j| h1[i] * h2[j] + h1[j] * h2[i]) + cubic(t, |i, j, k| h1[i] * h1[j] * h1[k])
}

/// Sum or difference of two first-order frequencies.
pub(super) fn f12(t: &Taylor, h1: Values, h2: Values) -> Complex {
    scaled(quadratic(t, |i, j| h1[i] * h2[j] + h1[j] * h2[i]), 0.5)
}

/// Third-order intermodulation from the distinct first/second-order kernels.
pub(super) fn two_f1_minus_f2(
    t: &Taylor,
    h1: Values,
    h1m2: Values,
    h11: Values,
    h1m2_2: Values,
) -> Complex {
    let second = quadratic(t, |i, j| {
        scaled(h1[i] * h1m2_2[j] + h1[j] * h1m2_2[i], 2.) + h1m2[i] * h11[j] + h1m2[j] * h11[i]
    });
    let third = cubic(t, |i, j, k| {
        h1[i] * h1[j] * h1m2[k] + h1[i] * h1[k] * h1m2[j] + h1[j] * h1[k] * h1m2[i]
    });
    let sum = second + third;
    Complex::new(sum.re / 3., sum.im / 3.)
}
