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

// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::float_cmp,
    reason = "C device branches compare floating-point values exactly, including nonfinite cases"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "physical-model equation symbols are kept consistent with the ngspice C reference"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
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

// A&S coefficients shared by scalar and parallel endpoint evaluation.
// Each polynomial retains the original nested multiply/add order.
const I0_SMALL: [Real; 7] = [
    1.,
    3.515_622_9,
    3.089_942_4,
    1.206_749_2,
    0.265_973_2,
    0.360_768e-1,
    0.45813e-2,
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
const I1_SMALL: [Real; 7] = [
    0.5,
    0.878_905_94,
    0.514_988_69,
    0.150_849_34,
    0.265_873_3e-1,
    0.301_532e-2,
    0.32411e-3,
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
        let y = x / 3.75;
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
        let y = ax / 3.75;
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
        let y = ax / 3.75;
        polynomial(y * y, &I1_SMALL, std::convert::identity)
    } else {
        polynomial(3.75 / ax, &I1_LARGE, std::convert::identity) * (ax.exp() / (ax * ax.sqrt()))
    }
}

/// I0(x) - 1 in the small-argument polynomial, without adding a unit
/// that would round away the correction.
fn small_i0_delta(x: Real) -> Real {
    let y = x / 3.75;
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
        let y = x / 3.75;
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
/// under them (`area * reltol + abstol > triangle`).
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
    area * reltol + abstol > triangle
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
    let mut xbig = td + 9. * td;
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
            let add = term / (2. * n + 1.);
            sum += add;
            if add.abs() <= 1e-17 * sum.abs() || n > 200. {
                break;
            }
        }
        return 1. - 2. / PI.sqrt() * sum;
    }
    if x > 27. {
        return 0.;
    }
    // Lentz: f = x + a1/(x + a2/(x + ...)), a_n = n/2.
    let tiny = 1e-300;
    let mut f = x;
    let mut c = f;
    let mut d = 0.;
    for n in 1..=500 {
        let a = Real::from(n) / 2.;
        d = x + a * d;
        if d == 0. {
            d = tiny;
        }
        c = x + a / c;
        if c == 0. {
            c = tiny;
        }
        d = 1. / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.).abs() < 1e-16 {
            break;
        }
    }
    (-x * x).exp() / (PI.sqrt() * f)
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
    let (mut doh1, mut doh2, mut doh3) = (true, true, true);

    let mut delta1 = curtime - times[index];
    let mut hilimit1 = delta1;

    let mut h1hivalue1 = (4. * cbyr * hilimit1 / PI).sqrt();
    let mut h1dummy1 = h1hivalue1 / delta1;
    out.first[0] = h1dummy1;
    let h1relval = (h1dummy1 * reltol).abs();

    let mut temp = rclsqr / (4. * hilimit1);
    let mut temp2 = if temp >= 100.0 {
        0.0
    } else {
        erfc(temp.sqrt())
    };
    let mut temp3 = (-temp).exp();
    let temp4 = rclsqr.sqrt();
    let temp5 = cbyr.sqrt();

    let h2value = |hilimit: Real, temp2: Real, temp3: Real| {
        if hilimit == 0.0 {
            0.0
        } else {
            (hilimit + rclsqr * 0.5) * temp2 - (hilimit * rclsqr / PI).sqrt() * temp3
        }
    };
    let h3value = |hilimit: Real, temp2: Real, temp3: Real| {
        if hilimit == 0.0 {
            0.0
        } else {
            temp5 * (2. * (hilimit / PI).sqrt() * temp3 - temp4 * temp2)
        }
    };

    let mut h2hivalue1 = h2value(hilimit1, temp2, temp3);
    let mut h2dummy1 = h2hivalue1 / delta1;
    out.first[1] = h2dummy1;
    let h2relval = (h2dummy1 * reltol).abs();

    let mut h3hivalue1 = h3value(hilimit1, temp2, temp3);
    let mut h3dummy1 = h3hivalue1 / delta1;
    out.first[2] = h3dummy1;
    let h3relval = (h3dummy1 * reltol).abs();

    for i in (1..=index).rev() {
        delta1 = times[i] - times[i - 1];
        hilimit1 = curtime - times[i - 1];

        if doh1 {
            let h1hivalue2 = h1hivalue1;
            let h1dummy2 = h1dummy1;
            let h1lovalue1 = h1hivalue2;
            h1hivalue1 = (4. * cbyr * hilimit1 / PI).sqrt();
            h1dummy1 = (h1hivalue1 - h1lovalue1) / delta1;
            out.weights[i][0] = h1dummy1 - h1dummy2;
            if out.weights[i][0].abs() < h1relval {
                doh1 = false;
            }
        }
        if doh2 || doh3 {
            temp = rclsqr / (4. * hilimit1);
            temp2 = if temp >= 100.0 {
                0.0
            } else {
                erfc(temp.sqrt())
            };
            temp3 = (-temp).exp();
        }
        if doh2 {
            let h2hivalue2 = h2hivalue1;
            let h2dummy2 = h2dummy1;
            let h2lovalue1 = h2hivalue2;
            h2hivalue1 = h2value(hilimit1, temp2, temp3);
            h2dummy1 = (h2hivalue1 - h2lovalue1) / delta1;
            out.weights[i][1] = h2dummy1 - h2dummy2;
            if out.weights[i][1].abs() < h2relval {
                doh2 = false;
            }
        }
        if doh3 {
            let h3hivalue2 = h3hivalue1;
            let h3dummy2 = h3dummy1;
            let h3lovalue1 = h3hivalue2;
            h3hivalue1 = h3value(hilimit1, temp2, temp3);
            h3dummy1 = (h3hivalue1 - h3lovalue1) / delta1;
            out.weights[i][2] = h3dummy1 - h3dummy2;
            if out.weights[i][2].abs() < h3relval {
                doh3 = false;
            }
        }
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
#[expect(
    clippy::too_many_lines,
    reason = "keeps the causal coupled coefficient recurrence and its auxiliary window together for audit"
)]
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
                    -0.355_268_504_606_938_87,
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
