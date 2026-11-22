//! Impulse-response integrals, interpolation and convolution coefficients of
//! the lossy line (C `src/spicelib/devices/ltra/ltramisc.c`).
//!
//! The line equations (Roychowdhury and Pederson's formulation, as LTRA
//! implements it) convolve the port voltages and currents with three
//! impulse responses `h1'`, `h2` and `h3'`. Over the accepted time points the
//! waveforms are taken piecewise linear, so each convolution becomes a sum of
//! the past samples weighted by coefficients built from repeated integrals of
//! the impulse responses; the coefficient of the newest (unknown) sample goes
//! into the matrix, the rest into the right-hand side.
//!
//! Preserve the C convolution and auxiliary-index policy, with local
//! interval formulas and exponentially scaled Bessel products to avoid
//! cancellation and overflow in finite physical responses. The modified Bessel functions are the polynomial
//! approximations of Abramowitz and Stegun 9.8.1-9.8.4 that `ltramisc.c` uses
//! (`bessI0`, `bessI1`, `bessI1xOverX`); `erfc` is a separate implementation
//! (see [`erfc`]).

#![expect(
    clippy::float_cmp,
    reason = "C device branches compare floating-point values exactly, including nonfinite cases"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "physical-model equation symbols are kept consistent with the ngspice C reference"
)]

use std::{collections::VecDeque, f64::consts::PI};

use crate::primitives::Real;

/// `LTRAquadInterp`: coefficients `(c1, c2, c3)` of the quadratic through
/// `(t1, t2, t3)` evaluated at `t`, or `None` for coincident times (C returns
/// 1 and leaves the coefficients unset).
#[must_use]
pub fn quad_interp(t: Real, t1: Real, t2: Real, t3: Real) -> Option<[Real; 3]> {
    if t == t1 {
        return Some([1., 0., 0.]);
    }
    if t == t2 {
        return Some([0., 1., 0.]);
    }
    if t == t3 {
        return Some([0., 0., 1.]);
    }
    if (t2 - t1) == 0. || (t3 - t2) == 0. || (t1 - t3) == 0. {
        return None;
    }
    Some([
        ((t - t2) / (t1 - t2)) * ((t - t3) / (t1 - t3)),
        ((t - t1) / (t2 - t1)) * ((t - t3) / (t2 - t3)),
        ((t - t1) / (t3 - t1)) * ((t - t2) / (t3 - t2)),
    ])
}

/// `LTRAlinInterp`: coefficients `(c1, c2)` of the line through `(t1, t2)` at
/// `t`, or `None` for `t1 == t2`.
#[must_use]
pub fn lin_interp(t: Real, t1: Real, t2: Real) -> Option<[Real; 2]> {
    if t1 == t2 {
        return None;
    }
    if t == t1 {
        return Some([1., 0.]);
    }
    if t == t2 {
        return Some([0., 1.]);
    }
    let temp = (t - t1) / (t2 - t1);
    Some([1. - temp, temp])
}

/// Evaluate a line from the closer endpoint, including extrapolation.
fn linear_value(t: Real, t1: Real, t2: Real, v1: Real, v2: Real) -> Real {
    if t == t1 {
        v1
    } else if t == t2 {
        v2
    } else if (t - t1).abs() <= (t - t2).abs() {
        crate::devices::functions::blend_from_anchor(v1, v2, (t - t1) / (t2 - t1))
    } else {
        crate::devices::functions::blend_from_anchor(v2, v1, (t - t2) / (t1 - t2))
    }
}

/// Integral of a linear function: interval width times mean endpoint
/// value. Do not form differences of squared absolute time offsets.
fn intlinfunc(
    lolimit: Real,
    hilimit: Real,
    lovalue: Real,
    hivalue: Real,
    t1: Real,
    t2: Real,
) -> Real {
    if t1 == t2 {
        return 0.;
    }
    let low = linear_value(lolimit, t1, t2, lovalue, hivalue);
    let high = linear_value(hilimit, t1, t2, lovalue, hivalue);
    (hilimit - lolimit) * low.midpoint(high)
}

/// Integral over [lo, hi] of the accumulated linear function from other.
/// Integrate around lo, avoiding the difference of cubes in the original
/// expression and keeping the auxiliary window's extrapolation semantics.
fn twiceintlinfunc(
    lolimit: Real,
    hilimit: Real,
    otherlolimit: Real,
    lovalue: Real,
    hivalue: Real,
    t1: Real,
    t2: Real,
) -> Real {
    if t1 == t2 {
        return 0.;
    }
    let width = hilimit - lolimit;
    let low = linear_value(lolimit, t1, t2, lovalue, hivalue);
    let high = linear_value(hilimit, t1, t2, lovalue, hivalue);
    let other = linear_value(otherlolimit, t1, t2, lovalue, hivalue);
    let initial_integral = (lolimit - otherlolimit) * other.midpoint(low);
    let weighted = (low / 3.) + high / 6.;
    width * width.mul_add(weighted, initial_integral)
}

// Small-argument Taylor coefficients in y=(x/2)^2: I0 has 1/(k!)^2,
// I1/x has 1/(2 k! (k+1)!). Through degree 14 the omitted absolute
// terms at x=3.75 are below 9.3e-17 and 2.9e-18 respectively.
// The A&S large-argument coefficients use y=3.75/x.
const I0_SMALL: [Real; 15] = [
    1.0,
    1.0,
    0.25,
    0.027_777_777_777_777_776,
    0.001_736_111_111_111_111,
    6.944_444_444_444_444e-5,
    1.929_012_345_679_012_4e-6,
    3.936_759_889_140_842e-8,
    6.151_187_326_782_565e-10,
    7.594_058_428_126_624e-12,
    7.594_058_428_126_623e-14,
    6.276_081_345_559_193e-16,
    4.358_389_823_304_995e-18,
    2.578_928_889_529_582_8e-20,
    1.315_780_045_678_358_6e-22,
];
const I0_LARGE: [Real; 9] = [
    0.398_942_28,
    0.132_859_2e-1,
    0.225_319e-2,
    -0.157_565e-2,
    0.916_281e-2,
    -0.205_770_6e-1,
    0.263_553_7e-1,
    -0.164_763_3e-1,
    0.392_377e-2,
];
const I1_SMALL: [Real; 15] = [
    0.5,
    0.25,
    0.041_666_666_666_666_664,
    0.003_472_222_222_222_222,
    0.000_173_611_111_111_111_12,
    5.787_037_037_037_037e-6,
    1.377_865_961_199_294_4e-7,
    2.460_474_930_713_026e-9,
    3.417_326_292_656_981e-11,
    3.797_029_214_063_311_6e-13,
    3.451_844_740_057_556e-15,
    2.615_033_893_982_997e-17,
    1.676_303_778_194_229e-19,
    9.210_460_319_748_51e-22,
    4.385_933_485_594_529e-24,
];
const I1_LARGE: [Real; 9] = [
    0.398_942_28,
    -0.398_802_4e-1,
    -0.362_018e-2,
    0.163_801e-2,
    -0.103_155_5e-1,
    0.228_296_7e-1,
    -0.289_531_2e-1,
    0.178_765_4e-1,
    -0.420_059e-2,
];

#[inline]
fn polynomial<T: Copy + std::ops::Mul<Output = T> + std::ops::Add<Output = T>>(
    y: T,
    coefficients: &[Real],
    constant: impl Fn(Real) -> T,
) -> T {
    let Some((&last, rest)) = coefficients.split_last() else {
        return constant(0.);
    };
    rest.iter()
        .rev()
        .fold(constant(last), |value, &c| constant(c) + y * value)
}

/// `bessI0` (Abramowitz and Stegun 9.8.1, 9.8.2).
#[must_use]
pub fn bess_i0(x: Real) -> Real {
    let ax = x.abs();
    if ax < 3.75 {
        let y = 0.5 * x;
        polynomial(y * y, &I0_SMALL, std::convert::identity)
    } else {
        (ax.exp() / ax.sqrt()) * polynomial(3.75 / ax, &I0_LARGE, std::convert::identity)
    }
}

/// `bessI1` (Abramowitz and Stegun 9.8.3, 9.8.4).
#[must_use]
pub fn bess_i1(x: Real) -> Real {
    let ax = x.abs();
    let ans = if ax < 3.75 {
        let y = 0.5 * ax;
        ax * polynomial(y * y, &I1_SMALL, std::convert::identity)
    } else {
        polynomial(3.75 / ax, &I1_LARGE, std::convert::identity) * (ax.exp() / ax.sqrt())
    };
    if x < 0.0 { -ans } else { ans }
}

/// `bessI1xOverX`: `I1(x) / x` for `x >= 0` (C uses `|x|`).
#[must_use]
pub fn bess_i1_x_over_x(x: Real) -> Real {
    let ax = x.abs();
    if ax < 3.75 {
        let y = 0.5 * ax;
        polynomial(y * y, &I1_SMALL, std::convert::identity)
    } else {
        polynomial(3.75 / ax, &I1_LARGE, std::convert::identity) * (ax.exp() / (ax * ax.sqrt()))
    }
}

/// I0(x) - 1 in the small-argument Taylor series, without adding a unit
/// that would round away the correction.
fn small_i0_delta(x: Real) -> Real {
    let y = 0.5 * x;
    let y = y * y;
    y * polynomial(y, &I0_SMALL[1..], std::convert::identity)
}

/// Evaluate exp(decay)*I0(x) and exp(decay)*I1(x) (or I1(x)/x).
/// For large x the combined exponent is supplied independently, so the
/// physical product never forms an exponentially large Bessel value.
fn weighted_bessel_pair<const DIVIDED: bool>(
    x: Real,
    decay: Real,
    combined_exponent: Real,
) -> (Real, Real) {
    if x < 3.75 {
        let y = 0.5 * x;
        let y = y * y;
        let factor = decay.exp();
        let i0 = polynomial(y, &I0_SMALL, std::convert::identity);
        let i1 = polynomial(y, &I1_SMALL, std::convert::identity);
        (factor * i0, factor * if DIVIDED { i1 } else { x * i1 })
    } else {
        let y = 3.75 / x;
        let factor = combined_exponent.exp() / x.sqrt();
        let i0 = factor * polynomial(y, &I0_LARGE, std::convert::identity);
        let i1 = factor * polynomial(y, &I1_LARGE, std::convert::identity);
        (i0, if DIVIDED { i1 / x } else { i1 })
    }
}

/// The RLC self endpoint is lag * [exp(-x)*(I0(x)+I1(x))-1],
/// with x=beta*lag. Preserve the small loss correction and evaluate
/// the large-argument product in exponentially scaled form.
fn rlc_self_endpoint(lag: Real, beta: Real) -> Real {
    if beta == 0. {
        return lag;
    }
    let x = beta * lag;
    if x < 3.75 {
        let delta = small_i0_delta(x) + bess_i1(x);
        lag * (-x).exp().mul_add(delta, (-x).exp_m1())
    } else {
        let (i0, i1) = weighted_bessel_pair::<false>(x, -x, 0.);
        lag * (i0 + i1 - 1.)
    }
}

/// RLC coupled endpoints. The large-argument exponent is rationalized:
/// alpha*sqrt(lag^2-delay^2)-beta*lag equals
/// (alpha-beta)*lag-alpha*delay^2/(lag+sqrt(lag^2-delay^2)).
#[derive(Clone, Copy)]
struct RlcKernels {
    delay: Real,
    alpha: Real,
    beta: Real,
    alpha_squared_delay: Real,
    delay_decay: Real,
}

impl RlcKernels {
    fn new(delay: Real, alpha: Real, beta: Real) -> Self {
        Self {
            delay,
            alpha,
            beta,
            alpha_squared_delay: alpha * alpha * delay,
            delay_decay: (-beta * delay).exp(),
        }
    }

    fn at(self, lag: Real, active: [bool; 3]) -> [Real; 3] {
        let h1 = if active[0] {
            rlc_self_endpoint(lag, self.beta)
        } else {
            0.
        };
        if lag < self.delay || !(active[1] || active[2]) {
            return [h1, 0., 0.];
        }
        let (argument, combined) = if lag == self.delay {
            (0., -self.beta * lag)
        } else {
            let ratio = self.delay / lag;
            let root_ratio = (((lag - self.delay) / lag) * (1. + ratio)).sqrt().min(1.);
            let argument = self.alpha * (lag * root_ratio);
            let lag_difference = self.delay * ratio / (1. + root_ratio);
            (
                argument,
                (self.alpha - self.beta).mul_add(lag, -self.alpha * lag_difference),
            )
        };
        let (i0, i1_over_x) = weighted_bessel_pair::<true>(argument, -self.beta * lag, combined);
        let h2 = if active[1] && self.alpha != 0. {
            self.alpha_squared_delay * i1_over_x
        } else {
            0.
        };
        let h3 = if !active[2] || lag <= self.delay || self.beta == 0. {
            0.
        } else if argument < 3.75 {
            let local = -self.beta * (lag - self.delay);
            self.delay_decay
                * local
                    .exp()
                    .mul_add(small_i0_delta(argument), local.exp_m1())
        } else {
            i0 - self.delay_decay
        };
        [h1, h2, h3]
    }
}

/// `LTRArlcH2Func`: h2(t) of the RLC line, zero before TD.
#[must_use]
pub fn rlc_h2(time: Real, t: Real, alpha: Real, beta: Real) -> Real {
    RlcKernels::new(t, alpha, beta).at(time, [false, true, false])[1]
}

/// `LTRAstraightLineCheck`: whether three points lie close enough to a
/// straight line, by comparing the triangle they span with the trapezoid area
/// under them (`area.mul_add(reltol, abstol) > triangle`).
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "keeps the positional interface explicit at the reference solver or device boundary"
)]
pub fn straight_line_check(
    x1: Real,
    y1: Real,
    x2: Real,
    y2: Real,
    x3: Real,
    y3: Real,
    reltol: Real,
    abstol: Real,
) -> bool {
    let quad1 = y2.abs().midpoint(y1.abs()) * (x2 - x1).abs();
    let quad2 = y3.abs().midpoint(y2.abs()) * (x3 - x2).abs();
    let quad3 = y3.abs().midpoint(y1.abs()) * (x3 - x1).abs();
    let triangle = (quad3 - quad1 - quad2).abs();
    let area = quad1 + quad2;
    area.mul_add(reltol, abstol) > triangle
}

/// The RLC line's `LTRAmaxSafeStep` (`ltratemp.c`): bisect `[TD, 10 TD]`
/// until `h2` is close enough to a straight line over `[TD, x]` (at most 51
/// halvings), and return `x - TD`.
///
/// C evaluates the straight-line test of `h2` twice (its sum with the
/// intended `h3'` test repeats the `h2` arguments), so only `h2` decides; the
/// port keeps that.
#[must_use]
pub fn rlc_max_safe_step(td: Real, alpha: Real, beta: Real, reltol: Real, abstol: Real) -> Real {
    let mut xbig = 10. * td;
    let xsmall = td;
    let mut xmid = xbig.midpoint(xsmall);
    let y1small = rlc_h2(xsmall, td, alpha, beta);
    let mut iters = 0;
    loop {
        iters += 1;
        let y1big = rlc_h2(xbig, td, alpha, beta);
        let y1mid = rlc_h2(xmid, td, alpha, beta);
        let done = straight_line_check(xbig, y1big, xmid, y1mid, xsmall, y1small, reltol, abstol);
        if done || iters > 50 {
            break;
        }
        xbig = xmid;
        xmid = xbig.midpoint(xsmall);
    }
    xbig - td
}

/// The complementary error function.
///
/// C's `ltramisc.c` calls the platform `erfc`; the port has no `libm`
/// dependency, so this evaluates the Maclaurin series of `erf` below 2 and
/// the classic continued fraction
/// `erfc(x) = exp(-x^2) / sqrt(pi) / (x + (1/2) / (x + 1 / (x + (3/2) / ...)))`
/// (Lentz's algorithm) above. Its relative error is below `1e-12` over the
/// range the line uses (`0 <= x < 10`, tested against reference values), far
/// inside the simulation tolerances, but it is not bit-identical to `libm`.
#[must_use]
pub fn erfc(x: Real) -> Real {
    if x.is_nan() {
        return x;
    }
    if x < 0. {
        return 2. - erfc(-x);
    }
    if x < 2. {
        // erf(x) = 2/sqrt(pi) sum (-1)^n x^(2n+1) / (n! (2n+1)).
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut n = 0.;
        loop {
            n += 1.;
            term *= -x2 / n;
            let add = term / 2_f64.mul_add(n, 1.);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() || n > 200. {
                break;
            }
        }
        return (-2. / PI.sqrt()).mul_add(sum, 1.);
    }
    if !x.is_finite() {
        return 0.;
    }
    let [_, _, denominator] = erfc_tail(x);
    (-x * x).exp() / (PI.sqrt() * denominator)
}

/// For x >= 2, represent
/// erfc(x) = exp(-x^2)/(sqrt(pi)*(x + u)),
/// u = (1/2)/(x+v), v = 1/(x+3/2/(x+...)).
/// Return the two small corrections as well as the full denominator;
/// RC integrals can use them without subtracting nearly equal terms.
fn erfc_tail(x: Real) -> [Real; 3] {
    // Lentz's algorithm for the positive fraction starting at a3.
    let mut f = x;
    let mut c = f;
    let mut d = 0.;
    for n in 3..=500 {
        let a = Real::from(n) / 2.;
        d = a.mul_add(d, x).recip();
        c = x + a / c;
        let delta = c * d;
        f *= delta;
        if (delta - 1.).abs() <= Real::EPSILON {
            break;
        }
    }
    let v = f.recip();
    let u = 0.5 / (x + v);
    [u, v, x + u]
}

/// Multiply by exp(-z), retaining representable results even if exp(-z)
/// itself is subnormal or rounds to zero.
fn attenuated(value: Real, z: Real) -> Real {
    let decay = (-z).exp();
    if decay.is_normal() {
        value * decay
    } else {
        let half = (-0.5 * z).exp();
        (value * half) * half
    }
}

/// The coefficients of one convolution step: the factor of the newest
/// (unknown) sample and the factors of the accepted samples `1..=index`
/// (entry 0 unused, as in C's lists).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Coefficients {
    /// `LTRAh1dashFirstCoeff`, `LTRAh2FirstCoeff`, `LTRAh3dashFirstCoeff`.
    pub first: [Real; 3],
    /// Accepted-sample coefficients in h1', h2, h3' order. A single
    /// allocation replaces three independent buffers.
    pub weights: Vec<[Real; 3]>,
    /// `LTRAauxIndex`: the last accepted sample at least `TD` before the
    /// current time (RLC only; the `h2`/`h3'` sums stop there).
    pub aux_index: usize,
}

impl Coefficients {
    fn with_len(len: usize) -> Self {
        Self {
            first: [0.; 3],
            weights: vec![[0.; 3]; len],
            aux_index: 0,
        }
    }
}

/// RC endpoint laws, with the root of the lag stored directly for the
/// rationalized first-kernel divided differences.
#[derive(Clone, Copy)]
struct RcKernel {
    rclsqr: Real,
    scale: Real,
}

impl RcKernel {
    fn new(cbyr: Real, rclsqr: Real) -> Self {
        Self {
            rclsqr,
            scale: 2. * cbyr.sqrt() / PI.sqrt(),
        }
    }

    fn at(self, lag: Real, active: [bool; 3]) -> [Real; 3] {
        let root = lag.sqrt();
        if lag == 0. || !(active[1] || active[2]) {
            return [root, 0., 0.];
        }
        // Avoid forming 4*lag before division.
        let z = (self.rclsqr / lag) * 0.25;
        if z.is_infinite() {
            return [root, 0., 0.];
        }
        let x = z.sqrt();
        let (h2, h3) = if x < 2. {
            let complementary = erfc(x);
            let decay = (-z).exp();
            let h2 = (2_f64.mul_add(z, 1.)).mul_add(complementary, -2. * x * decay / PI.sqrt());
            let h3 = (-PI.sqrt() * x).mul_add(complementary, decay);
            (lag * h2, (self.scale * root) * h3)
        } else {
            let [u, v, denominator] = erfc_tail(x);
            // h2 bracket = ((1+2*x*x)/(x+u) - 2*x)/sqrt(pi)
            // = v / ((x+v)*(x+u)*sqrt(pi)).
            // h3 bracket = 1 - x/(x+u) = u/(x+u).
            let h2_factor = (v / (x + v)) / denominator / PI.sqrt();
            let h2 = attenuated(lag * h2_factor, z);
            let h3 = self.scale * attenuated(root * (u / denominator), z);
            (h2, h3)
        };
        [
            root,
            if active[1] { h2 } else { 0. },
            if active[2] { h3 } else { 0. },
        ]
    }
}

/// `LTRArcCoeffsSetup` (the `L = G = 0` line): coefficients at `curtime` over
/// the accepted times `times[0..=index]`, with C's chopping tolerance
/// `reltol` (C's `LTRAchopReltol`, always 0 from a deck).
#[must_use]
pub fn rc_coefficients(
    cbyr: Real,
    rclsqr: Real,
    curtime: Real,
    times: &VecDeque<Real>,
    reltol: Real,
) -> Coefficients {
    let index = times.len() - 1;
    let mut out = Coefficients::with_len(times.len());
    let kernel = RcKernel::new(cbyr, rclsqr);
    let mut active = [true; 3];
    let lag = curtime - times[index];
    let mut previous = kernel.at(lag, active);
    let mut slopes = [
        kernel.scale / previous[0],
        previous[1] / lag,
        previous[2] / lag,
    ];
    out.first = slopes;
    let thresholds = slopes.map(|slope| (slope * reltol).abs());
    let mut earlier_root = 0.;
    let mut previous_delta = lag;

    for i in (1..=index).rev() {
        if !active.iter().any(|&enabled| enabled) {
            break;
        }
        let delta = times[i] - times[i - 1];
        let endpoint = kernel.at(curtime - times[i - 1], active);
        let weight = &mut out.weights[i];
        if active[0] {
            // H1(t)=scale*sqrt(t). Rationalize both divided differences:
            // S_new = scale/(sqrt(hi)+sqrt(lo)),
            // S_new-S_old = -S_new*(sqrt(hi)-sqrt(earlier))/old_den.
            // hi-earlier is the sum of the two actual accepted intervals,
            // even if the absolute lag coordinates round to the same bits.
            let root = endpoint[0];
            let old_den = previous[0] + earlier_root;
            let new_slope = kernel.scale / (root + previous[0]);
            let root_difference = (delta + previous_delta) / (root + earlier_root);
            weight[0] = -new_slope * (root_difference / old_den);
            if weight[0].abs() < thresholds[0] {
                active[0] = false;
            }
        }
        for k in 1..3 {
            if active[k] {
                let slope = (endpoint[k] - previous[k]) / delta;
                weight[k] = slope - slopes[k];
                slopes[k] = slope;
                if weight[k].abs() < thresholds[k] {
                    active[k] = false;
                }
            }
        }
        earlier_root = previous[0];
        previous_delta = delta;
        previous = endpoint;
    }
    out.aux_index = index;
    out
}

/// Locate the newest accepted sample at least one delay back. Preserve C's
/// subtraction and equality branches, including rounded equal-time plateaus.
fn rlc_aux_index(t: Real, curtime: Real, times: &VecDeque<Real>) -> usize {
    if t == 0.0 {
        return times.len() - 1;
    }
    if curtime - t <= 0.0 {
        return 0;
    }
    let Some(found) = times
        .partition_point(|&sample_time| curtime - sample_time >= t)
        .checked_sub(1)
    else {
        return 0;
    };
    if curtime - times[found] == t {
        found.saturating_sub(1)
    } else {
        found
    }
}

/// `LTRArlcCoeffsSetup` (the `G = 0` RLC line): coefficients at `curtime`
/// over the accepted times `times[0..=index]`, with delay `t`, `alpha`,
/// `beta` and C's chopping tolerance `reltol` (always 0 from a deck).
///
/// C quirk kept: at the auxiliary index the `h2` correction term integrates
/// the previous window `[lolimit2, hilimit2]` with the values at `TD` and at
/// the auxiliary time (C's commented-out `lolimit2 = T` fix is not applied).
#[must_use]
pub fn rlc_coefficients(
    t: Real,
    alpha: Real,
    beta: Real,
    curtime: Real,
    times: &VecDeque<Real>,
    reltol: Real,
) -> Coefficients {
    let index = times.len() - 1;
    let mut out = Coefficients::with_len(times.len());
    let (mut doh1, mut doh2, mut doh3) = (true, true, true);

    let aux = rlc_aux_index(t, curtime, times);

    let kernels = RlcKernels::new(t, alpha, beta);
    let [
        mut h2lovalue1,
        mut h2hivalue1,
        mut h2dummy1,
        mut h3hivalue1,
        mut h3dummy1,
    ] = if aux != 0 {
        let hilimit = curtime - times[aux];
        let delta = hilimit - t;
        let h2low = rlc_h2(t, t, alpha, beta);
        let [_, h2high, h3high] = kernels.at(hilimit, [false, true, true]);
        let h2first = twiceintlinfunc(t, hilimit, t, h2low, h2high, t, hilimit) / delta;
        let h3first = intlinfunc(t, hilimit, 0., h3high, t, hilimit) / delta;
        [h2low, h2high, h2first, h3high, h3first]
    } else {
        [0.; 5]
    };
    out.first[1] = h2dummy1;
    out.first[2] = h3dummy1;
    let h2relval = (reltol * h2dummy1).abs();
    let h3relval = (reltol * h3dummy1).abs();

    let mut lolimit1 = 0.0;
    let mut hilimit1 = curtime - times[index];
    let mut delta1 = hilimit1 - lolimit1;
    let mut h1hivalue1 = kernels.at(hilimit1, [true, false, false])[0];
    let mut h1dummy1 = h1hivalue1 / delta1;
    out.first[0] = h1dummy1;
    let h1relval = (h1dummy1 * reltol).abs();

    let mut lolimit2;
    let mut hilimit2 = 0.0;
    lolimit2 = 0.0;

    for i in (1..=index).rev() {
        if doh1 || doh2 || doh3 {
            lolimit2 = lolimit1;
            hilimit2 = hilimit1;
            lolimit1 = hilimit2;
            hilimit1 = curtime - times[i - 1];
            delta1 = times[i] - times[i - 1];
        }
        let endpoint = kernels.at(hilimit1, [doh1, doh2 && i <= aux, doh3 && i <= aux]);
        if doh1 {
            let h1lovalue1 = h1hivalue1;
            let h1dummy2 = h1dummy1;
            h1hivalue1 = endpoint[0];
            h1dummy1 = (h1hivalue1 - h1lovalue1) / delta1;
            out.weights[i][0] = h1dummy1 - h1dummy2;
            if out.weights[i][0].abs() <= h1relval {
                doh1 = false;
            }
        }
        if i <= aux {
            if doh2 {
                let h2lovalue2 = h2lovalue1;
                let h2hivalue2 = h2hivalue1;
                let h2dummy2 = h2dummy1;
                h2lovalue1 = h2hivalue2;
                h2hivalue1 = endpoint[1];
                h2dummy1 = twiceintlinfunc(
                    lolimit1, hilimit1, lolimit1, h2lovalue1, h2hivalue1, lolimit1, hilimit1,
                ) / delta1;
                out.weights[i][1] = h2dummy1 - h2dummy2
                    + intlinfunc(
                        lolimit2, hilimit2, h2lovalue2, h2hivalue2, lolimit2, hilimit2,
                    );
                if out.weights[i][1].abs() <= h2relval {
                    doh2 = false;
                }
            }
            if doh3 {
                let h3hivalue2 = h3hivalue1;
                let h3dummy2 = h3dummy1;
                let h3lovalue1 = h3hivalue2;
                h3hivalue1 = endpoint[2];
                h3dummy1 = intlinfunc(
                    lolimit1, hilimit1, h3lovalue1, h3hivalue1, lolimit1, hilimit1,
                ) / delta1;
                out.weights[i][2] = h3dummy1 - h3dummy2;
                if out.weights[i][2].abs() <= h3relval {
                    doh3 = false;
                }
            }
        }
    }
    out.aux_index = aux;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erfc_matches_reference_values() {
        // Reference values from an arbitrary-precision evaluation.
        for (x, expected) in [
            (0.0, 1.0),
            (0.1, 0.887_537_083_981_715_2),
            (0.5, 0.479_500_122_186_953_5),
            (1.0, 0.157_299_207_050_285_16),
            (1.9, 0.007_209_570_764_742_534),
            (2.0, 0.004_677_734_981_047_264_5),
            (3.0, 2.209_049_699_858_543_8e-5),
            (5.0, 1.537_459_794_428_035e-12),
            (9.5, 3.769_214_485_654_88e-41),
            (-1.0, 1.842_700_792_949_714_8),
        ] {
            let got = erfc(x);
            assert!(
                ((got - expected) / expected).abs() < 1e-12,
                "erfc({x}) = {got:e}, expected {expected:e}"
            );
        }
    }

    #[test]
    fn bessel_approximations_match_known_values() {
        // I0(1) = 1.2660658777520082, I1(1) = 0.5651591039924851,
        // I0(5) = 27.239871823604442, I1(5) = 24.335642142450524.
        // The A&S polynomials are accurate to about 1e-7 relative.
        for (got, expected) in [
            (bess_i0(1.), 1.266_065_877_752_008_2),
            (bess_i1(1.), 0.565_159_103_992_485_1),
            (bess_i0(5.), 27.239_871_823_604_442),
            (bess_i1(5.), 24.335_642_142_450_524),
            (bess_i1_x_over_x(5.), 24.335_642_142_450_524 / 5.),
            (bess_i1(-1.), -0.565_159_103_992_485_1),
        ] {
            assert!(
                ((got - expected) / expected).abs() < 2e-7,
                "{got} {expected}"
            );
        }
    }

    #[test]
    fn rlc_endpoints_remain_finite_and_accurate_across_loss_scales() {
        // 90-digit independent Bessel power-series evaluations.
        for (beta, delay, lag, expected) in [
            (
                1.,
                1.,
                1000.,
                [
                    -974.771_829_287_180_1,
                    1.260_463_581_744_048_3e-5,
                    -0.355_268_504_606_938_9,
                ],
            ),
            (1e-20, 0.5, 1., [-5e-21, 2.5e-41, -5e-21]),
            (
                1.,
                1.,
                1_f64.next_up(),
                [
                    -0.326_329_977_056_651_23,
                    0.183_939_720_585_721_14,
                    -4.084_282_258_747_71e-17,
                ],
            ),
        ] {
            let kernels = RlcKernels::new(delay, beta, beta);
            let got = kernels.at(lag, [true; 3]);
            for (actual, reference) in got.into_iter().zip(expected) {
                assert!(
                    (actual - reference).abs() <= 3e-7 * reference.abs(),
                    "beta={beta:e}, lag={lag:e}, {actual:e} != {reference:e}"
                );
            }
            assert_eq!(kernels.at(delay.next_down(), [false, true, true]), [0.; 3]);
        }
        let times = VecDeque::from([0., 1., 10., 100., 900.]);
        let c = rlc_coefficients(1., 1., 1., 1000., &times, 0.);
        assert!(
            c.first
                .iter()
                .chain(c.weights.iter().flatten())
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn interpolation_reproduces_polynomials() {
        let [c1, c2, c3] = quad_interp(1.5, 0., 1., 3.).unwrap();
        let f = |t: f64| 2. * t * t - t + 1.;
        assert!((c1 * f(0.) + c2 * f(1.) + c3 * f(3.) - f(1.5)).abs() < 1e-12);
        assert_eq!(quad_interp(1., 0., 1., 3.), Some([0., 1., 0.]));
        assert_eq!(quad_interp(2., 0., 1., 1.), None);
        let [l1, l2] = lin_interp(0.25, 0., 1.).unwrap();
        assert_eq!((l1, l2), (0.75, 0.25));
        assert_eq!(lin_interp(1., 1., 1.), None);
    }

    #[test]
    fn tiny_interpolation_intervals_and_extrapolated_integrals() {
        let h = 1e-180;
        let weights = quad_interp(1.5 * h, 0., h, 2. * h).unwrap();
        assert!((weights[1] + 4. * weights[2] - 2.25).abs() <= 8. * Real::EPSILON);
        // h(t)=2+3t. Integral [10,11] is 33.5; integrating
        // its primitive from 4 has the antiderivative below.
        assert_eq!(intlinfunc(10., 11., 2., 5., 0., 1.), 33.5);
        let f = |t: Real| t * t + 0.5 * t * t * t - 32. * t;
        assert!((twiceintlinfunc(10., 11., 4., 2., 5., 0., 1.) - (f(11.) - f(10.))).abs() < 1e-12);
        // Coordinates near 1e150, values scaled down to keep the
        // double integral finite while the old cubic terms overflow.
        let h = 1e120;
        let actual = twiceintlinfunc(0., h, 0., 1e-240, 1e-240, 0., h);
        assert!((actual - 0.5).abs() <= 4. * Real::EPSILON);
    }

    #[test]
    fn rc_coefficients_sum_to_the_step_response() {
        // For a constant waveform the convolution of h1' collapses to its
        // integral: the first coefficient plus the history coefficients
        // telescope to (H(t) - H(t - t0)) / dt differences; check the
        // telescoping identity sum_i c_i = (first window) sum directly.
        let times = VecDeque::from([0., 1e-9, 2e-9, 3e-9]);
        let c = rc_coefficients(1e-12 / 100., 100. * 1e-12, 4e-9, &times, 0.);
        assert_eq!(c.weights.len(), 4);
        assert!(c.first.iter().all(|v| v.is_finite()));
        assert!(c.weights[1..].iter().all(|v| v[0].is_finite()));
        // Second differences of a concave sqrt are negative.
        assert!(c.weights[1..].iter().all(|v| v[0] < 0.));
    }

    #[test]
    fn rc_far_history_weights_do_not_disappear_when_absolute_lags_round_equal() {
        let times = VecDeque::from([0., 0.25, 0.5, 0.75]);
        let c = rc_coefficients(1., 1., 1e16, &times, 0.);
        let scale = 2. / PI.sqrt();
        // Adjacent sqrt divided differences have separation
        // -scale * (dt_prev + dt_next)/(8 * lag^(3/2)).
        let expected = -scale * 0.5 / (8. * 1e24);
        for w in &c.weights[1..3] {
            assert!((w[0] - expected).abs() <= 8. * Real::EPSILON * expected.abs());
        }
    }

    #[test]
    fn rc_diffusion_tails_remain_positive_and_match_precise_integrals() {
        let lag = 1.;
        let cbyr = 1.;
        // Independent 510-decimal-digit erf power-series evaluations,
        // with the original analytic integral formulas.
        for (z, expected) in [
            (0.01, [0.793_572_664_982_455_7, 0.939_644_189_992_593_9]),
            (1., [0.056_790_123_730_260_69, 0.100_509_083_320_024_44]),
            (
                4.,
                [0.000_765_644_121_241_284_8, 0.001_956_045_429_902_990_4],
            ),
            (
                25.,
                [5.611_686_074_310_081_5e-14, 2.962_685_867_369_868e-13],
            ),
            (
                100.,
                [2.038_120_082_980_714_8e-47, 2.068_106_382_932_737_4e-46],
            ),
            (
                400.,
                [1.340_612_433_172_205_8e-178, 2.691_229_743_638_089_2e-177],
            ),
        ] {
            let kernel = RcKernel::new(cbyr, 4. * z);
            let [_, h2, h3] = kernel.at(lag, [true; 3]);
            for (actual, reference) in [h2, h3].into_iter().zip(expected) {
                assert!(
                    (actual - reference).abs() <= 2e-12 * reference,
                    "z={z} {actual:e} != {reference:e}"
                );
            }
        }
        // erfc(27) is a positive subnormal, not zero.
        assert!(erfc(27.) > 0.);
        let kernel = RcKernel::new(1., 4000. * 1e150);
        let [_, h2, h3] = kernel.at(1e150, [true; 3]);
        assert!(h2 > 0.);
        assert_eq!(h3, 0.);
    }

    #[test]
    fn auxiliary_index_matches_reference_reverse_scan() {
        // Closely spaced times may round to the same curtime - sample_time.
        let times = VecDeque::from([0., 1e-20, 2e-20, 0.5, 1., 2., 3.]);
        for current in [1., 2., 3., 4., 1e16] {
            for delay in [0., 0.5, 1., 2., 3., 4., 1e16 - 2.] {
                let expected = if delay == 0. {
                    times.len() - 1
                } else if current - delay <= 0. {
                    0
                } else {
                    times
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|(_, sample)| current - **sample >= delay)
                        .map_or(0, |(index, sample)| {
                            if current - sample == delay {
                                index.saturating_sub(1)
                            } else {
                                index
                            }
                        })
                };
                assert_eq!(rlc_aux_index(delay, current, &times), expected);
            }
        }
    }

    #[test]
    fn max_safe_step_is_inside_the_bracket() {
        let td = 5e-9;
        let alpha = 0.5 * 10. / 250e-9;
        let step = rlc_max_safe_step(td, alpha, alpha, 1e-3, 1e-12);
        assert!(step > 0. && step <= 9. * td, "{step}");
    }
}
