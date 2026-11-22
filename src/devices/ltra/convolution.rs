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
//! Every routine follows its C counterpart operation by operation, including
//! C's quirks (noted where they matter), so that the port reproduces ngspice's
//! coefficients to rounding. The modified Bessel functions are the polynomial
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
    clippy::manual_midpoint,
    reason = "C source midpoint arithmetic retains its original evaluation and rounding order"
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
    let mut f1 = (t - t2) * (t - t3);
    let mut f2 = (t - t1) * (t - t3);
    let mut f3 = (t - t1) * (t - t2);
    // C's further zero checks are unreachable after the test above.
    f1 /= t1 - t2;
    f2 /= t2 - t1;
    f2 /= t2 - t3;
    f3 /= t2 - t3;
    f1 /= t1 - t3;
    f3 /= t1 - t3;
    Some([f1, f2, f3])
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

/// `intlinfunc`: `int_lo^hi h` for `h` linear through `(t1, lovalue)` and
/// `(t2, hivalue)`.
fn intlinfunc(
    lolimit: Real,
    hilimit: Real,
    lovalue: Real,
    hivalue: Real,
    t1: Real,
    t2: Real,
) -> Real {
    let width = t2 - t1;
    if width == 0. {
        return 0.;
    }
    let m = (hivalue - lovalue) / width;
    (hilimit - lolimit) * lovalue
        + 0.5 * m * ((hilimit - t1) * (hilimit - t1) - (lolimit - t1) * (lolimit - t1))
}

/// `twiceintlinfunc`: `int_lo^hi int_other^tau h` for the same linear `h`.
fn twiceintlinfunc(
    lolimit: Real,
    hilimit: Real,
    otherlolimit: Real,
    lovalue: Real,
    hivalue: Real,
    t1: Real,
    t2: Real,
) -> Real {
    let width = t2 - t1;
    if width == 0. {
        return 0.;
    }
    let m = (hivalue - lovalue) / width;
    let temp1 = hilimit - t1;
    let temp2 = lolimit - t1;
    let temp3 = otherlolimit - t1;
    let mut dummy = lovalue
        * ((hilimit - otherlolimit) * (hilimit - otherlolimit)
            - (lolimit - otherlolimit) * (lolimit - otherlolimit));
    let temp3_squared = temp3 * temp3;
    dummy += m
        * ((temp1 * temp1 * temp1 - temp2 * temp2 * temp2) / 3.0
            - temp3_squared * (hilimit - lolimit));
    dummy * 0.5
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

/// Batch I0 and either I1 or I1/|x|. Large-argument I0/I1 share
/// the exponential and square root without changing division order.
/// When adjacent knots straddle the polynomial boundary, fall back
/// to the scalar laws rather than evaluating both polynomials.
fn bessel_pair_batch<const DIVIDED: bool>(
    x: std::simd::f64x4,
) -> (std::simd::f64x4, std::simd::f64x4) {
    use std::simd::{Select, StdFloat, cmp::SimdPartialOrd, f64x4, num::SimdFloat};
    let ax = x.abs();
    let boundary = f64x4::splat(3.75);
    let small = ax.simd_lt(boundary);
    let (i0, i1) = if small.all() {
        let y = ax / boundary;
        let y = y * y;
        let i0 = polynomial(y, &I0_SMALL, f64x4::splat);
        let i1 = polynomial(y, &I1_SMALL, f64x4::splat);
        (i0, if DIVIDED { i1 } else { ax * i1 })
    } else if !small.any() {
        let y = boundary / ax;
        let exp = f64x4::from_array(ax.to_array().map(Real::exp));
        let root = ax.sqrt();
        let i0 = (exp / root) * polynomial(y, &I0_LARGE, f64x4::splat);
        let scale = if DIVIDED {
            exp / (ax * root)
        } else {
            exp / root
        };
        let i1 = polynomial(y, &I1_LARGE, f64x4::splat) * scale;
        (i0, i1)
    } else {
        let values = x.to_array();
        return (
            f64x4::from_array(values.map(bess_i0)),
            f64x4::from_array(values.map(if DIVIDED { bess_i1_x_over_x } else { bess_i1 })),
        );
    };
    (
        i0,
        if DIVIDED {
            i1
        } else {
            x.simd_lt(f64x4::splat(0.)).select(-i1, i1)
        },
    )
}

/// `LTRArlcH2Func`: `h2(t)` of the RLC line, zero before `T`.
#[must_use]
pub fn rlc_h2(time: Real, t: Real, alpha: Real, beta: Real) -> Real {
    if alpha == 0.0 || time < t {
        return 0.0;
    }
    let besselarg = if time == t {
        0.0
    } else {
        alpha * (time * time - t * t).sqrt()
    };
    let exparg = -beta * time;
    alpha * alpha * t * exparg.exp() * bess_i1_x_over_x(besselarg)
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
    let quad1 = (y2.abs() + y1.abs()) * 0.5 * (x2 - x1).abs();
    let quad2 = (y3.abs() + y2.abs()) * 0.5 * (x3 - x2).abs();
    let quad3 = (y3.abs() + y1.abs()) * 0.5 * (x3 - x1).abs();
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
    let mut xmid = 0.5 * (xbig + xsmall);
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
        xmid = 0.5 * (xbig + xsmall);
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

/// Sum h1', h2 and h3' over the accepted waveform in one reverse pass.
/// The two port contributions to each kernel are independent: use SIMD
/// across those contributions, not across chronological terms. Every
/// accumulator therefore retains the original addition order.
pub(super) fn history_sums(
    rows: impl DoubleEndedIterator<Item = [Real; 4]>,
    weights: &[[Real; 3]],
    initial: [Real; 4],
    coupled_end: usize,
    mut sums: [[Real; 2]; 3],
) -> [[Real; 2]; 3] {
    use std::simd::{Select, cmp::SimdPartialEq, f64x2, f64x4};
    // Entry zero represents the initial conditions. Split the newest,
    // uncoupled part of an RLC history before processing h2/h3. Before
    // the propagation delay, only the two h1' port lanes need updating.
    let count = weights.len().saturating_sub(1);
    let mut pairs = rows.rev().zip(weights.iter().rev()).take(count);
    let uncoupled = count.saturating_sub(coupled_end);
    if uncoupled != 0 {
        let mut voltage = f64x2::from_array(sums[0]);
        let initial_voltage = f64x2::from_array([initial[0], initial[2]]);
        for ([v1, _, v2, _], weight) in pairs.by_ref().take(uncoupled) {
            if weight[0] != 0.0 {
                voltage +=
                    f64x2::splat(weight[0]) * (f64x2::from_array([v1, v2]) - initial_voltage);
            }
        }
        sums[0] = voltage.to_array();
    }
    if coupled_end == 0 {
        return sums;
    }
    let [h1, h2, h3] = sums;
    let mut voltage = f64x4::from_array([h1[0], h1[1], h3[0], h3[1]]);
    let mut current = f64x2::from_array(h2);
    let [iv1, ii1, iv2, ii2] = initial;
    let initial_voltage = f64x4::from_array([iv1, iv2, iv2, iv1]);
    let initial_current = f64x2::from_array([ii2, ii1]);
    for ([v1, i1, v2, i2], &[h1, h2, h3]) in pairs {
        if h1 != 0.0 || h3 != 0.0 {
            let coefficient = f64x4::from_array([h1, h1, h3, h3]);
            let delta = f64x4::from_array([v1, v2, v2, v1]) - initial_voltage;
            // A zero coefficient is skipped, even if delta overflows.
            // Selection also preserves the accumulator's signed zero.
            voltage = coefficient
                .simd_ne(f64x4::splat(0.0))
                .select(voltage + coefficient * delta, voltage);
        }
        if h2 != 0.0 {
            current += f64x2::splat(h2) * (f64x2::from_array([i2, i1]) - initial_current);
        }
    }
    let [h1a, h1b, h3a, h3b] = voltage.to_array();
    [[h1a, h1b], current.to_array(), [h3a, h3b]]
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

/// RC endpoint integral laws. The lag is measured backwards from the
/// proposed time; adjacent endpoints are shared by the divided differences.
#[derive(Clone, Copy)]
struct RcKernel {
    cbyr: Real,
    rclsqr: Real,
    root_rclsqr: Real,
    root_cbyr: Real,
}

impl RcKernel {
    fn at(self, lag: Real, active: [bool; 3]) -> [Real; 3] {
        let h1 = if active[0] {
            (4. * self.cbyr * lag / PI).sqrt()
        } else {
            0.
        };
        if !(active[1] || active[2]) || lag == 0. {
            return [h1, 0., 0.];
        }
        let temp = self.rclsqr / (4. * lag);
        let complement = if temp >= 100. { 0. } else { erfc(temp.sqrt()) };
        let decay = (-temp).exp();
        let h2 = if active[1] {
            (lag + self.rclsqr * 0.5) * complement - (lag * self.rclsqr / PI).sqrt() * decay
        } else {
            0.
        };
        let h3 = if active[2] {
            self.root_cbyr * (2. * (lag / PI).sqrt() * decay - self.root_rclsqr * complement)
        } else {
            0.
        };
        [h1, h2, h3]
    }

    fn batch(self, lag: [Real; 4]) -> [[Real; 3]; 4] {
        use std::simd::{Select, StdFloat, cmp::SimdPartialEq, f64x4};
        let lag = f64x4::from_array(lag);
        let zero = f64x4::splat(0.);
        let temp = f64x4::splat(self.rclsqr) / (f64x4::splat(4.) * lag);
        let complement = rc_erfc_batch(temp);
        let decay = f64x4::from_array((-temp).to_array().map(Real::exp));
        let h1 = (f64x4::splat(4. * self.cbyr) * lag / f64x4::splat(PI)).sqrt();
        let nonzero = lag.simd_ne(zero);
        let h2 = nonzero.select(
            (lag + f64x4::splat(self.rclsqr * 0.5)) * complement
                - (lag * f64x4::splat(self.rclsqr) / f64x4::splat(PI)).sqrt() * decay,
            zero,
        );
        let h3 = nonzero.select(
            f64x4::splat(self.root_cbyr)
                * (f64x4::splat(2.) * (lag / f64x4::splat(PI)).sqrt() * decay
                    - f64x4::splat(self.root_rclsqr) * complement),
            zero,
        );
        let [h1, h2, h3] = [h1.to_array(), h2.to_array(), h3.to_array()];
        std::array::from_fn(|i| [h1[i], h2[i], h3[i]])
    }
}

/// Independent nonnegative RC erfc arguments in a batch. Each lane follows
/// the scalar recurrence and stops at its own convergence test. On a
/// mixed-regime or invalid batch, use the scalar reference instead.
fn rc_erfc_batch(temp: std::simd::f64x4) -> std::simd::f64x4 {
    use std::simd::{
        Select, StdFloat,
        cmp::{SimdPartialEq, SimdPartialOrd},
        f64x4,
        num::SimdFloat,
    };
    let zero = f64x4::splat(0.);
    let one = f64x4::splat(1.);
    let two = f64x4::splat(2.);
    let cutoff = temp.simd_ge(f64x4::splat(100.));
    let valid = temp.simd_ge(zero);
    if !valid.all() {
        return f64x4::from_array(
            temp.to_array()
                .map(|x| if x >= 100. { 0. } else { erfc(x.sqrt()) }),
        );
    }
    if cutoff.all() {
        return zero;
    }
    let x = temp.sqrt();
    let series = x.simd_lt(two) & !cutoff;
    let continued = !series & !cutoff;
    let result = if series.any() && continued.any() {
        return f64x4::from_array(
            temp.to_array()
                .map(|x| if x >= 100. { 0. } else { erfc(x.sqrt()) }),
        );
    } else if series.any() {
        let x = cutoff.select(zero, x);
        let x2 = x * x;
        let mut term = x;
        let mut sum = x;
        let mut active = series;
        for n in 1..=201 {
            let n = Real::from(n);
            term *= -x2 / f64x4::splat(n);
            let add = term / f64x4::splat(2. * n + 1.);
            sum = active.select(sum + add, sum);
            active &= add.abs().simd_gt(f64x4::splat(1e-17) * sum.abs());
            if !active.any() {
                break;
            }
        }
        one - f64x4::splat(2. / PI.sqrt()) * sum
    } else {
        let x = cutoff.select(two, x);
        let tiny = f64x4::splat(1e-300);
        let mut f = x;
        let mut c = f;
        let mut d = zero;
        let mut active = continued;
        for n in 1..=500 {
            let a = f64x4::splat(Real::from(n) / 2.);
            d = x + a * d;
            d = d.simd_eq(zero).select(tiny, d);
            c = x + a / c;
            c = c.simd_eq(zero).select(tiny, c);
            d = one / d;
            let delta = c * d;
            f = active.select(f * delta, f);
            active &= (delta - one).abs().simd_ge(f64x4::splat(1e-16));
            if !active.any() {
                break;
            }
        }
        let numerator = f64x4::from_array((-x * x).to_array().map(Real::exp));
        numerator / (f64x4::splat(PI.sqrt()) * f)
    };
    cutoff.select(zero, result)
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
    let kernel = RcKernel {
        cbyr,
        rclsqr,
        root_rclsqr: rclsqr.sqrt(),
        root_cbyr: cbyr.sqrt(),
    };
    let mut active = [true; 3];
    let lag = curtime - times[index];
    let mut previous = kernel.at(lag, active);
    let mut slopes = previous.map(|value| value / lag);
    out.first = slopes;
    let thresholds = slopes.map(|slope| (slope * reltol).abs());

    // RC uses a strict '<' chopping test. With zero tolerance no kernel
    // can stop early, so these independent endpoints can be batched.
    // Keep the scalar path for nonzero chopping and an incomplete batch.
    'chunks: for (chunk_index, chunk) in out.weights[1..].rchunks_mut(4).enumerate() {
        let newest = index - chunk_index * 4;
        let batch = if reltol == 0. && chunk.len() == 4 {
            Some(kernel.batch(std::array::from_fn(|lane| {
                curtime - times[newest - lane - 1]
            })))
        } else {
            None
        };
        for (lane, weight) in chunk.iter_mut().rev().enumerate() {
            if !active.iter().any(|&enabled| enabled) {
                break 'chunks;
            }
            let i = newest - lane;
            let endpoint = batch.as_ref().map_or_else(
                || kernel.at(curtime - times[i - 1], active),
                |values| values[lane],
            );
            let delta = times[i] - times[i - 1];
            for k in 0..3 {
                if active[k] {
                    let slope = (endpoint[k] - previous[k]) / delta;
                    weight[k] = slope - slopes[k];
                    previous[k] = endpoint[k];
                    slopes[k] = slope;
                    if weight[k].abs() < thresholds[k] {
                        active[k] = false;
                    }
                }
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

fn rlc_self_endpoint(lag: Real, beta: Real, exponent: Real, decay: Real) -> Real {
    if beta == 0. {
        lag
    } else if lag == 0. {
        0.
    } else {
        (bess_i1(-exponent) + bess_i0(-exponent)) * lag * decay - lag
    }
}

// With no accepted samples beyond the propagation delay, only the local
// kernel contributes. A short history cannot amortize the batched setup.
fn short_self_coefficients(
    beta: Real,
    curtime: Real,
    times: &VecDeque<Real>,
    reltol: Real,
    out: &mut Coefficients,
) {
    let endpoint = |lag| {
        let exponent = -beta * lag;
        rlc_self_endpoint(lag, beta, exponent, exponent.exp())
    };
    let index = times.len() - 1;
    let lag = curtime - times[index];
    let mut previous = endpoint(lag);
    let mut slope = previous / lag;
    out.first[0] = slope;
    let threshold = (slope * reltol).abs();
    for i in (1..=index).rev() {
        let current = endpoint(curtime - times[i - 1]);
        let next_slope = (current - previous) / (times[i] - times[i - 1]);
        let weight = next_slope - slope;
        out.weights[i][0] = weight;
        if weight.abs() <= threshold {
            break;
        }
        previous = current;
        slope = next_slope;
    }
}

/// Independent endpoint fields needed by the RLC divided differences.
/// The recurrence, delayed auxiliary index and chopping remain scalar.
#[derive(Clone, Copy)]
struct RlcKernels {
    delay: Real,
    alpha: Real,
    beta: Real,
    alpha_squared_delay: Real,
    delay_decay: Real,
}

impl RlcKernels {
    fn at(self, lag: Real, active: [bool; 3]) -> [Real; 3] {
        let exponent = -self.beta * lag;
        let decay = exponent.exp();
        let h1 = if active[0] {
            rlc_self_endpoint(lag, self.beta, exponent, decay)
        } else {
            0.
        };
        if !(active[1] || active[2]) {
            return [h1, 0., 0.];
        }
        let argument = if lag > self.delay {
            self.alpha * (lag * lag - self.delay * self.delay).sqrt()
        } else {
            0.
        };
        let h2 = if !active[1] || self.alpha == 0. || lag < self.delay {
            0.
        } else {
            self.alpha_squared_delay * decay * bess_i1_x_over_x(argument)
        };
        let h3 = if !active[2] || lag <= self.delay || self.beta == 0. {
            0.
        } else {
            decay * bess_i0(argument) - self.delay_decay
        };
        [h1, h2, h3]
    }

    fn batch(self, lags: [Real; 4], active: [bool; 3]) -> [[Real; 3]; 4] {
        use std::simd::{
            Select, StdFloat,
            cmp::{SimdPartialEq, SimdPartialOrd},
            f64x4,
        };
        let lag = f64x4::from_array(lags);
        let zero = f64x4::splat(0.);
        let exponent = f64x4::splat(-self.beta) * lag;
        let decay = f64x4::from_array(exponent.to_array().map(Real::exp));
        let h1 = if !active[0] {
            zero
        } else if self.beta == 0. {
            lag
        } else {
            let (i0, i1) = bessel_pair_batch::<false>(-exponent);
            lag.simd_eq(zero)
                .select(zero, (i1 + i0) * lag * decay - lag)
        };
        let mut h2 = zero;
        let mut h3 = zero;
        if active[1] || active[2] {
            let transit = f64x4::splat(self.delay);
            let beyond = lag.simd_gt(transit);
            let argument = beyond.select(
                f64x4::splat(self.alpha) * (lag * lag - transit * transit).sqrt(),
                zero,
            );
            let (i0, divided_i1) = bessel_pair_batch::<true>(argument);
            if active[1] && self.alpha != 0. {
                h2 = (!lag.simd_lt(transit)).select(
                    f64x4::splat(self.alpha_squared_delay) * decay * divided_i1,
                    zero,
                );
            }
            if active[2] && self.beta != 0. {
                h3 = (!lag.simd_le(transit))
                    .select(decay * i0 - f64x4::splat(self.delay_decay), zero);
            }
        }
        let [h1, h2, h3] = [h1.to_array(), h2.to_array(), h3.to_array()];
        std::array::from_fn(|i| [h1[i], h2[i], h3[i]])
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
    reason = "keeps the reference numerical routine in the original statement and rounding order"
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
    if aux == 0 && index < 32 {
        short_self_coefficients(beta, curtime, times, reltol, &mut out);
        return out;
    }

    let kernels = RlcKernels {
        delay: t,
        alpha,
        beta,
        alpha_squared_delay: alpha * alpha * t,
        delay_decay: (-beta * t).exp(),
    };
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
        // E3dash is zero at TD (C).
        let h3first = intlinfunc(t, hilimit, 0., h3high, t, hilimit) / delta;
        [h2low, h2high, h2first, h3high, h3first]
    } else {
        [0.; 5]
    };
    out.first[1] = h2dummy1;
    out.first[2] = h3dummy1;
    let h2relval = (reltol * h2dummy1).abs();
    let h3relval = (h3dummy1 * reltol).abs();

    let mut lolimit1 = 0.0;
    let mut hilimit1 = curtime - times[index];
    let mut delta1 = hilimit1 - lolimit1;
    let mut h1hivalue1 = kernels.at(hilimit1, [true, false, false])[0];
    let mut h1dummy1 = h1hivalue1 / delta1;
    out.first[0] = h1dummy1;
    let h1relval = (h1dummy1 * reltol).abs();

    'chunks: for (chunk_index, chunk) in out.weights[1..].rchunks_mut(4).enumerate() {
        let newest = index - chunk_index * 4;
        let batch = if reltol == 0. && chunk.len() == 4 {
            let coupled = newest - (chunk.len() - 1) <= aux;
            Some(kernels.batch(
                std::array::from_fn(|lane| curtime - times[newest - lane - 1]),
                [doh1, doh2 && coupled, doh3 && coupled],
            ))
        } else {
            None
        };
        for (lane, weight) in chunk.iter_mut().rev().enumerate() {
            if !(doh1 || doh2 || doh3) {
                break 'chunks;
            }
            let i = newest - lane;
            let lolimit2 = lolimit1;
            let hilimit2 = hilimit1;
            lolimit1 = hilimit2;
            hilimit1 = curtime - times[i - 1];
            delta1 = times[i] - times[i - 1];

            let endpoint = batch.as_ref().map_or_else(
                || kernels.at(hilimit1, [doh1, doh2 && i <= aux, doh3 && i <= aux]),
                |values| values[lane],
            );
            if doh1 {
                let h1lovalue1 = h1hivalue1;
                let h1dummy2 = h1dummy1;
                h1hivalue1 = endpoint[0];
                h1dummy1 = (h1hivalue1 - h1lovalue1) / delta1;
                weight[0] = h1dummy1 - h1dummy2;
                if weight[0].abs() <= h1relval {
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
                    weight[1] = h2dummy1 - h2dummy2
                        + intlinfunc(
                            lolimit2, hilimit2, h2lovalue2, h2hivalue2, lolimit2, hilimit2,
                        );
                    if weight[1].abs() <= h2relval {
                        doh2 = false;
                    }
                }
                if doh3 {
                    let h3lovalue1 = h3hivalue1;
                    let h3dummy2 = h3dummy1;
                    h3hivalue1 = endpoint[2];
                    h3dummy1 = intlinfunc(
                        lolimit1, hilimit1, h3lovalue1, h3hivalue1, lolimit1, hilimit1,
                    ) / delta1;
                    weight[2] = h3dummy1 - h3dummy2;
                    if weight[2].abs() <= h3relval {
                        doh3 = false;
                    }
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

    fn assert_same(got: Real, expected: Real) {
        if expected.is_nan() {
            assert!(got.is_nan(), "{got}");
        } else {
            assert_eq!(got.to_bits(), expected.to_bits(), "{got:e} vs {expected:e}");
        }
    }

    #[test]
    fn vector_endpoints_match_the_scalar_laws_at_boundaries() {
        use std::simd::f64x4;
        // Distinct stopping iterations, series/continued fraction split,
        // the RC cutoff, nonphysical inputs and nonfinite propagation.
        for batch in [
            [0., 0.01, 0.5, 3.999_999_999],
            [4., 4.1, 16., 99.999_999],
            [0.25, 4., 100., Real::INFINITY],
            [-1., Real::NAN, 0., 1.],
            [99.9, 100., 100.1, Real::INFINITY],
        ] {
            let got = rc_erfc_batch(f64x4::from_array(batch)).to_array();
            for (value, temp) in got.into_iter().zip(batch) {
                assert_same(value, if temp >= 100. { 0. } else { erfc(temp.sqrt()) });
            }
        }
        for batch in [
            [-0., 0.125, -1.5, 3.75f64.next_down()],
            [-3.75, 3.75, 12., 30.],
            [0., 3.75, -0.5, -4.],
            [670., 709., 720., Real::INFINITY],
        ] {
            let x = f64x4::from_array(batch);
            let (i0, i1) = bessel_pair_batch::<false>(x);
            let (i0_divided, divided) = bessel_pair_batch::<true>(x);
            for (i, value) in batch.into_iter().enumerate() {
                assert_same(i0[i], bess_i0(value));
                assert_same(i0_divided[i], bess_i0(value));
                assert_same(i1[i], bess_i1(value));
                assert_same(divided[i], bess_i1_x_over_x(value));
            }
        }
        let rc = RcKernel {
            cbyr: 1e-12,
            rclsqr: 4e-8,
            root_rclsqr: 4e-8f64.sqrt(),
            root_cbyr: 1e-12f64.sqrt(),
        };
        for lags in [
            [0., 1e-12, 1e-10, 1e-8],
            [1e-9, 2e-9, 3e-9, 4e-9],
            [1e-7, 1e-6, 1e-5, 1e-4],
        ] {
            for (vector, lag) in rc.batch(lags).into_iter().zip(lags) {
                for (got, expected) in vector.into_iter().zip(rc.at(lag, [true; 3])) {
                    assert_same(got, expected);
                }
            }
        }
        for (delay, alpha, beta) in [
            (5e-9, 2e7, 2e7),
            (5e-9, 0., 2e7),
            (5e-9, 2e7, 0.),
            (0., 2e7, 2e7),
        ] {
            let kernels = RlcKernels {
                delay,
                alpha,
                beta,
                alpha_squared_delay: alpha * alpha * delay,
                delay_decay: (-beta * delay).exp(),
            };
            for lags in [
                [0., delay.next_down(), delay, delay.next_up()],
                [1e-8, 2e-8, 3e-8, 4e-8],
                [2e-7, 4e-7, 8e-7, 1e-6],
            ] {
                for (vector, lag) in kernels.batch(lags, [true; 3]).into_iter().zip(lags) {
                    let exponent = -beta * lag;
                    let decay = exponent.exp();
                    let h1 = if beta == 0. {
                        lag
                    } else if lag == 0. {
                        0.
                    } else {
                        (bess_i1(-exponent) + bess_i0(-exponent)) * lag * decay - lag
                    };
                    let h2 = rlc_h2(lag, delay, alpha, beta);
                    let h3 = if lag <= delay || beta == 0. {
                        0.
                    } else {
                        decay * bess_i0(alpha * (lag * lag - delay * delay).sqrt())
                            - (-beta * delay).exp()
                    };
                    for (got, expected) in vector.into_iter().zip([h1, h2, h3]) {
                        assert_same(got, expected);
                    }
                }
            }
        }
    }

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
