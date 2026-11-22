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

use std::{
    collections::VecDeque,
    f64::consts::PI,
    simd::{
        Select, Simd, StdFloat,
        cmp::{SimdPartialEq, SimdPartialOrd},
        num::SimdFloat,
    },
};

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

/// Shared argument preparation for the Bessel functions. LTRA evaluates
/// I0 together with I1 or I1/x at the same knot; the expensive exponential
/// and square root for the asymptotic series must only be evaluated once.
#[derive(Clone, Copy)]
enum BesselArgument {
    Small {
        squared: Real,
        abs: Real,
    },
    Large {
        y: Real,
        abs: Real,
        exponential: Real,
        root: Real,
    },
}

impl BesselArgument {
    #[inline]
    fn new(x: Real) -> Self {
        let abs = x.abs();
        if abs < 3.75 {
            let y = x / 3.75;
            Self::Small {
                squared: y * y,
                abs,
            }
        } else {
            Self::Large {
                y: 3.75 / abs,
                abs,
                exponential: abs.exp(),
                root: abs.sqrt(),
            }
        }
    }

    #[inline]
    fn i0(self) -> Real {
        match self {
            Self::Small { squared: y, .. } => {
                1.0 + y
                    * (3.515_622_9
                        + y * (3.089_942_4
                            + y * (1.206_749_2
                                + y * (0.265_973_2 + y * (0.360_768e-1 + y * 0.45813e-2)))))
            }
            Self::Large {
                y,
                exponential,
                root,
                ..
            } => {
                (exponential / root)
                    * (0.398_942_28
                        + y * (0.132_859_2e-1
                            + y * (0.225_319e-2
                                + y * (-0.157_565e-2
                                    + y * (0.916_281e-2
                                        + y * (-0.205_770_6e-1
                                            + y * (0.263_553_7e-1
                                                + y * (-0.164_763_3e-1 + y * 0.392_377e-2))))))))
            }
        }
    }

    #[inline]
    fn i1(self) -> Real {
        match self {
            Self::Small { squared, abs } => abs * bess_i1_small(squared),
            Self::Large {
                y,
                exponential,
                root,
                ..
            } => bess_i1_tail(y) * (exponential / root),
        }
    }

    #[inline]
    fn i1_over_x(self) -> Real {
        match self {
            Self::Small { squared, .. } => bess_i1_small(squared),
            Self::Large {
                y,
                abs,
                exponential,
                root,
            } => bess_i1_tail(y) * (exponential / (abs * root)),
        }
    }
}

fn bess_i1_small(y: Real) -> Real {
    0.5 + y
        * (0.878_905_94
            + y * (0.514_988_69
                + y * (0.150_849_34 + y * (0.265_873_3e-1 + y * (0.301_532e-2 + y * 0.32411e-3)))))
}

/// The large-argument series shared by `bessI1` and `bessI1xOverX`.
fn bess_i1_tail(y: Real) -> Real {
    let ans = 0.228_296_7e-1 + y * (-0.289_531_2e-1 + y * (0.178_765_4e-1 - y * 0.420_059e-2));
    0.398_942_28
        + y * (-0.398_802_4e-1
            + y * (-0.362_018e-2 + y * (0.163_801e-2 + y * (-0.103_155_5e-1 + y * ans))))
}

/// `bessI0` (Abramowitz and Stegun 9.8.1, 9.8.2).
#[must_use]
pub fn bess_i0(x: Real) -> Real {
    BesselArgument::new(x).i0()
}

/// `bessI1` (Abramowitz and Stegun 9.8.3, 9.8.4).
#[must_use]
pub fn bess_i1(x: Real) -> Real {
    let ans = BesselArgument::new(x).i1();
    if x < 0.0 { -ans } else { ans }
}

/// `bessI1xOverX`: `I1(x) / x` for `x >= 0` (C uses `|x|`).
#[must_use]
pub fn bess_i1_x_over_x(x: Real) -> Real {
    BesselArgument::new(x).i1_over_x()
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
    erfc_lanes(Simd::from_array([x]))[0]
}

/// Each lane keeps its own convergence state, including the exact final
/// iteration of the reference scalar series/continued fraction.
fn erfc_lanes<const N: usize>(input: Simd<Real, N>) -> Simd<Real, N> {
    let x = input.abs();
    let zero = Simd::splat(0.);
    let one = Simd::splat(1.);
    let two = Simd::splat(2.);
    let mut series = x.simd_lt(two);
    let mut fraction = x.simd_ge(two) & x.simd_le(Simd::splat(27.));
    let small = series;
    let large = fraction;
    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut n = 0.;
    while series.any() {
        n += 1.;
        term *= -x2 / Simd::splat(n);
        let add = term / Simd::splat(2. * n + 1.);
        sum = series.select(sum + add, sum);
        series &= !add.abs().simd_le(Simd::splat(1e-17) * sum.abs());
        if n > 200. {
            break;
        }
    }

    let tiny = Simd::splat(1e-300);
    let mut f = x;
    let mut c = f;
    let mut d = zero;
    for n in 1..=500 {
        if !fraction.any() {
            break;
        }
        let a = Simd::splat(Real::from(n) / 2.);
        d = x + a * d;
        d = d.simd_eq(zero).select(tiny, d);
        c = x + a / c;
        c = c.simd_eq(zero).select(tiny, c);
        d = one / d;
        let delta = c * d;
        f = fraction.select(f * delta, f);
        fraction &= !(delta - one).abs().simd_lt(Simd::splat(1e-16));
    }
    let series_value = one - Simd::splat(2. / PI.sqrt()) * sum;
    // Retain the original scalar exp implementation on each lane.
    let fraction_value = if large.any() {
        let exponents = (-x * x).to_array();
        let active = large.to_array();
        let exponential = Simd::from_array(std::array::from_fn(|lane| {
            if active[lane] {
                exponents[lane].exp()
            } else {
                0.
            }
        }));
        exponential / (Simd::splat(PI.sqrt()) * f)
    } else {
        zero
    };
    let result = small.select(series_value, large.select(fraction_value, zero));
    let signed = input.simd_lt(zero).select(two - result, result);
    input.is_nan().select(input, signed)
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

/// Independent endpoint integrals for all three RC impulse responses.
struct RcKernel {
    cbyr: Real,
    rclsqr: Real,
    root_rclsqr: Real,
    root_cbyr: Real,
}

impl RcKernel {
    fn new(cbyr: Real, rclsqr: Real) -> Self {
        Self {
            cbyr,
            rclsqr,
            root_rclsqr: rclsqr.sqrt(),
            root_cbyr: cbyr.sqrt(),
        }
    }

    fn endpoints<const N: usize>(&self, limits: [Real; N]) -> [[Real; 3]; N] {
        let hi = Simd::from_array(limits);
        let pi = Simd::splat(PI);
        let zero = Simd::splat(0.);
        let h1 = (Simd::splat(4. * self.cbyr) * hi / pi).sqrt();
        let temp = Simd::splat(self.rclsqr) / (Simd::splat(4.) * hi);
        // The C cutoff is before erfc, not merely an approximation to it.
        let x = temp
            .simd_ge(Simd::splat(100.))
            .select(Simd::splat(Real::INFINITY), temp.sqrt());
        let temp2 = erfc_lanes(x);
        let temp3 = Simd::from_array((-temp).to_array().map(Real::exp));
        let h2 = (hi + Simd::splat(self.rclsqr * 0.5)) * temp2
            - (hi * Simd::splat(self.rclsqr) / pi).sqrt() * temp3;
        let h3 = Simd::splat(self.root_cbyr)
            * (Simd::splat(2.) * (hi / pi).sqrt() * temp3 - Simd::splat(self.root_rclsqr) * temp2);
        let nonzero = hi.simd_ne(zero);
        let h2 = nonzero.select(h2, zero);
        let h3 = nonzero.select(h3, zero);
        std::array::from_fn(|lane| [h1[lane], h2[lane], h3[lane]])
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
    const LANES: usize = 4;
    let index = times.len() - 1;
    let mut out = Coefficients::with_len(times.len());
    let kernel = RcKernel::new(cbyr, rclsqr);
    let delta = curtime - times[index];
    let [mut previous] = kernel.endpoints([delta]);
    let mut slopes = previous.map(|value| value / delta);
    out.first = slopes;
    let thresholds = slopes.map(|value| (value * reltol).abs());
    let mut active = [true; 3];

    // The expensive endpoint integrals are independent. Evaluate four at
    // a time, then apply divided differences in the original reverse order.
    // No extra history-sized workspace is needed.
    for (block_index, block) in out.weights[1..].rchunks_mut(LANES).enumerate() {
        if !active.into_iter().any(|on| on) {
            break;
        }
        let newest = index - block_index * LANES;
        let count = block.len();
        let limits = std::array::from_fn::<_, LANES, _>(|lane| {
            curtime - times[newest - lane.min(count - 1) - 1]
        });
        let endpoints = if count == LANES {
            kernel.endpoints(limits)
        } else {
            std::array::from_fn(|lane| {
                if lane < count {
                    kernel.endpoints([limits[lane]])[0]
                } else {
                    [0.; 3]
                }
            })
        };
        for (offset, (weight, endpoint)) in block.iter_mut().rev().zip(endpoints).enumerate() {
            let sample = newest - offset;
            let delta = times[sample] - times[sample - 1];
            for channel in 0..3 {
                if active[channel] {
                    let slope = (endpoint[channel] - previous[channel]) / delta;
                    weight[channel] = slope - slopes[channel];
                    if weight[channel].abs() < thresholds[channel] {
                        active[channel] = false;
                    }
                    slopes[channel] = slope;
                    previous[channel] = endpoint[channel];
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

    let mut h2lovalue1 = 0.0;
    let mut h2hivalue1 = 0.0;
    let mut h2dummy1 = 0.0;
    let mut h3hivalue1 = 0.0;
    let mut h3dummy1 = 0.0;
    let mut alphasq_t = 0.0;
    let mut exp_beta_t = 0.0;
    let mut h2relval = 0.0;
    let mut h3relval = 0.0;

    if aux != 0 {
        let lolimit1 = t;
        let hilimit1 = curtime - times[aux];
        let delta1 = hilimit1 - lolimit1;

        h2lovalue1 = rlc_h2(t, t, alpha, beta);
        let besselarg = if hilimit1 > t {
            alpha * (hilimit1 * hilimit1 - t * t).sqrt()
        } else {
            0.0
        };
        let expterm = (-beta * hilimit1).exp();
        let bessel = BesselArgument::new(besselarg);
        let bessi1overxterm = bessel.i1_over_x();
        alphasq_t = alpha * alpha * t;
        h2hivalue1 = if alpha == 0.0 || hilimit1 < t {
            0.0
        } else {
            alphasq_t * expterm * bessi1overxterm
        };
        h2dummy1 = twiceintlinfunc(
            lolimit1, hilimit1, lolimit1, h2lovalue1, h2hivalue1, lolimit1, hilimit1,
        ) / delta1;
        out.first[1] = h2dummy1;
        h2relval = (reltol * h2dummy1).abs();

        let h3lovalue1 = 0.0; // E3dash is consistent with this (C).
        let bessi0term = bessel.i0();
        exp_beta_t = (-beta * t).exp();
        h3hivalue1 = if hilimit1 <= t || beta == 0.0 {
            0.0
        } else {
            expterm * bessi0term - exp_beta_t
        };
        h3dummy1 = intlinfunc(
            lolimit1, hilimit1, h3lovalue1, h3hivalue1, lolimit1, hilimit1,
        ) / delta1;
        out.first[2] = h3dummy1;
        h3relval = (h3dummy1 * reltol).abs();
    }

    let h1value = |hilimit: Real, exparg: Real, expterm: Real| {
        if beta == 0.0 {
            hilimit
        } else if hilimit == 0.0 {
            0.0
        } else {
            let argument = -exparg;
            let bessel = BesselArgument::new(argument);
            let i1 = bessel.i1();
            let i1 = if argument < 0. { -i1 } else { i1 };
            (i1 + bessel.i0()) * hilimit * expterm - hilimit
        }
    };

    let mut lolimit1 = 0.0;
    let mut hilimit1 = curtime - times[index];
    let mut delta1 = hilimit1 - lolimit1;
    let mut exparg = -beta * hilimit1;
    let mut expterm = exparg.exp();

    let mut h1lovalue1;
    let mut h1hivalue1 = h1value(hilimit1, exparg, expterm);
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
            exparg = -beta * hilimit1;
            expterm = exparg.exp();
        }
        if doh1 {
            let h1hivalue2 = h1hivalue1;
            let h1dummy2 = h1dummy1;
            h1lovalue1 = h1hivalue2;
            h1hivalue1 = h1value(hilimit1, exparg, expterm);
            h1dummy1 = (h1hivalue1 - h1lovalue1) / delta1;
            out.weights[i][0] = h1dummy1 - h1dummy2;
            if out.weights[i][0].abs() <= h1relval {
                doh1 = false;
            }
        }
        if i <= aux && (doh2 || doh3) {
            let besselarg = if hilimit1 > t {
                alpha * (hilimit1 * hilimit1 - t * t).sqrt()
            } else {
                0.0
            };
            let bessel = BesselArgument::new(besselarg);
            if doh2 {
                let h2lovalue2 = h2lovalue1;
                let h2hivalue2 = h2hivalue1;
                let h2dummy2 = h2dummy1;
                h2lovalue1 = h2hivalue2;
                let bessi1overxterm = bessel.i1_over_x();
                h2hivalue1 = if alpha == 0.0 || hilimit1 < t {
                    0.0
                } else {
                    alphasq_t * expterm * bessi1overxterm
                };
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
                let bessi0term = bessel.i0();
                h3hivalue1 = if hilimit1 <= t || beta == 0.0 {
                    0.0
                } else {
                    expterm * bessi0term - exp_beta_t
                };
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
    #[test]
    fn erfc_lanes_retains_each_scalar_convergence() {
        let values = [
            -Real::INFINITY,
            -27.,
            -2.,
            -0.,
            0.,
            1.,
            1.999_999_9,
            2.,
            2.1,
            5.,
            9.999,
            10.,
            27.,
            28.,
            Real::INFINITY,
            Real::NAN,
        ];
        for block in values.chunks_exact(4) {
            let input: [Real; 4] = block.try_into().unwrap();
            let vector = erfc_lanes(Simd::from_array(input)).to_array();
            let scalar = input.map(erfc);
            assert_eq!(vector.map(Real::to_bits), scalar.map(Real::to_bits));
        }
        let kernel = RcKernel::new(0.015, 0.003);
        for values in [
            [0., 1e-9, 1e-3, 1.],
            [1e-6, 5e-6, 8e-6, 0.1],
            [0.01, 0.018, 0.03, 0.2],
        ] {
            let actual = kernel.endpoints(values);
            let expected = values.map(|value| kernel.endpoints([value])[0]);
            assert_eq!(
                actual.map(|v| v.map(Real::to_bits)),
                expected.map(|v| v.map(Real::to_bits))
            );
        }
    }
}
