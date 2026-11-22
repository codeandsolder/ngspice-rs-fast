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
    scaled(product(0, 0), t.xx)
        + scaled(product(1, 1), t.yy)
        + scaled(product(2, 2), t.zz)
        + scaled(product(0, 1), t.xy)
        + scaled(product(1, 2), t.yz)
        + scaled(product(0, 2), t.xz)
}

fn cubic(t: &Taylor, product: impl Fn(usize, usize, usize) -> Complex) -> Complex {
    scaled(product(0, 0, 0), t.xxx)
        + scaled(product(1, 1, 1), t.yyy)
        + scaled(product(2, 2, 2), t.zzz)
        + scaled(product(0, 0, 1), t.xxy)
        + scaled(product(0, 0, 2), t.xxz)
        + scaled(product(0, 1, 1), t.xyy)
        + scaled(product(1, 1, 2), t.yyz)
        + scaled(product(0, 2, 2), t.xzz)
        + scaled(product(1, 2, 2), t.yzz)
        + scaled(product(0, 1, 2), t.xyz)
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
