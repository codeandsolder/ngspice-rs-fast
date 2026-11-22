//! SPICE numeric literals, scale factors, and the complex type.
//!
//! # Scale factors
//!
//! Ported from `INPevaluate()` in `src/spicelib/parser/inpeval.c` (the switch on
//! `*here` around line 143) and the scale-factor scanning in
//! `src/frontend/inpcom.c` around line 3098.
//!
//! The table is **not** SI, and the surprises are load-bearing for compatibility:
//!
//! | Suffix | Multiplier | Note |
//! | --- | --- | --- |
//! | `T` | 1e12 | |
//! | `G` | 1e9 | |
//! | `MEG` | 1e6 | |
//! | `K` | 1e3 | |
//! | `MIL` | 25.4e-6 | thousandths of an inch |
//! | `M` | 1e-3 | **milli**, not mega |
//! | `L` | 1e-3 | accepted by `INPevaluate()` |
//! | `U` | 1e-6 | |
//! | `N` | 1e-9 | |
//! | `P` | 1e-12 | |
//! | `F` | 1e-15 | |
//! | `A` | 1e-18 | accepted by `INPevaluate()` |
//!
//! An unrecognised trailing letter does not change the value: `5V` is `5`, and
//! `1kohm` is `1000`. Anything else after the literal — `1k2` — is rejected by
//! [`parse_spice_number`]; see the RKM note on that function.

use super::numeric::{scale_power, split_power};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The floating point type used throughout the port.
///
/// ngspice mixes `double` and `float`; the port uses `f64` everywhere and
/// records the places where the C code downcasts as a divergence risk.
pub type Real = f64;

/// Compares two reals with a relative tolerance, falling back to an absolute one
/// near zero.
#[must_use]
pub fn approx_eq(a: Real, b: Real, relative_tolerance: Real) -> bool {
    if a == b {
        return true;
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    let scale = a.abs().max(b.abs());
    (a - b).abs() <= relative_tolerance * scale.max(1.0)
}

/// ngspice's `AlmostEqualUlps(a, b, max_ulps)` (`src/maths/misc/equality.c`):
/// `a == b`, or the two's-complement-ordered bit patterns of `a` and `b` are at
/// most `max_ulps` apart.
#[must_use]
pub fn almost_equal_ulps(a: Real, b: Real, max_ulps: i64) -> bool {
    // The ordered mapping already makes equal values, including both signed
    // zeros, zero ULPs apart.
    // Lexicographically ordered integer representations (`INT64_MIN - bits`
    // for negative values), compared without overflow.
    let ordered = |x: Real| {
        // Reinterpret the IEEE-754 sign bit, rather than numerically narrowing.
        let bits = x.to_bits().cast_signed();
        i128::from(if bits < 0 { i64::MIN - bits } else { bits })
    };
    (ordered(a) - ordered(b)).unsigned_abs() <= u128::from(max_ulps.unsigned_abs())
}

/// A complex number, mirroring ngspice's `complex` struct
/// (`src/include/ngspice/ngspice.h`), which is a two-element `double` array.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    /// Real part.
    pub re: Real,
    /// Imaginary part.
    pub im: Real,
}

impl Complex {
    /// `0 + 0i`.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Builds a complex number.
    #[must_use]
    pub const fn new(re: Real, im: Real) -> Self {
        Self { re, im }
    }

    /// A purely real value.
    #[must_use]
    pub const fn real(re: Real) -> Self {
        Self { re, im: 0.0 }
    }

    /// A purely imaginary value.
    #[must_use]
    pub const fn imaginary(im: Real) -> Self {
        Self { re: 0.0, im }
    }

    /// `sqrt(re^2 + im^2)`.
    #[must_use]
    pub fn magnitude(self) -> Real {
        self.re.hypot(self.im)
    }

    /// The argument in radians.
    #[must_use]
    pub fn phase(self) -> Real {
        self.im.atan2(self.re)
    }

    /// The complex conjugate.
    #[must_use]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// True when both parts are finite.
    #[must_use]
    pub const fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// True when the imaginary part is zero.
    #[must_use]
    pub fn is_real(self) -> bool {
        self.im == 0.0
    }
}

impl From<Real> for Complex {
    fn from(re: Real) -> Self {
        Self::real(re)
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let ac = self.re * rhs.re;
        let bd = -self.im * rhs.im;
        let ad = self.re * rhs.im;
        let bc = self.im * rhs.re;
        let real = ac + bd;
        let imag = ad + bc;
        // An axial operand has only one potentially nonzero product in
        // each component: no two-product cancellation or range recovery
        // is possible. Preserve the IEEE zero/nonfinite signs of the sum.
        if self.re == 0. || self.im == 0. || rhs.re == 0. || rhs.im == 0. {
            return Self::new(real, imag);
        }
        Self::new(
            refine_product(self.re, rhs.re, -self.im, rhs.im, ac, bd, real),
            refine_product(self.re, rhs.im, self.im, rhs.re, ad, bc, imag),
        )
    }
}

impl Div for Complex {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        let cc = rhs.re * rhs.re;
        let dd = rhs.im * rhs.im;
        let denominator = cc + dd;
        let ac = self.re * rhs.re;
        let bd = self.im * rhs.im;
        let bc = self.im * rhs.re;
        let ad = -self.re * rhs.im;
        let real = compensated_sum(self.re, rhs.re, self.im, rhs.im, ac, bd);
        let imag = compensated_sum(self.im, rhs.re, -self.re, rhs.im, bc, ad);
        let result = Self::new(real / denominator, imag / denominator);
        // Keep numerator and denominator in exponent-scaled form whenever
        // materializing either could erase a representable quotient component.
        if (!denominator.is_normal()
            || !result.is_finite()
            || (!real.is_normal()
                && (lost_product(ac, self.re, rhs.re)
                    || lost_product(bd, self.im, rhs.im)
                    || quotient_numerator_lost_range(real, ac, bd)))
            || (!imag.is_normal()
                && (lost_product(bc, self.im, rhs.re)
                    || lost_product(ad, self.re, rhs.im)
                    || quotient_numerator_lost_range(imag, bc, ad))))
            && self.is_finite()
            && rhs.is_finite()
            && rhs != Self::ZERO
        {
            return scaled_complex_div(self, rhs);
        }
        result
    }
}

/// Recover the rounding residual in a cancellation-prone sum of two
/// products. The IEEE sum is sufficient for well-conditioned sums; avoid
/// software FMA calls on generic targets for these common cases.
#[inline]
fn compensated_sum(a: Real, b: Real, c: Real, d: Real, first: Real, second: Real) -> Real {
    let sum = first + second;
    if sum.abs() <= 0.5 * first.abs().max(second.abs()) && first.is_normal() && second.is_normal() {
        a.mul_add(b, second) + c.mul_add(d, -second)
    } else {
        sum
    }
}

#[inline]
fn refine_product(
    a: Real,
    b: Real,
    c: Real,
    d: Real,
    first: Real,
    second: Real,
    sum: Real,
) -> Real {
    if sum.is_normal() && sum.abs() > 0.5 * first.abs().max(second.abs()) {
        return sum;
    }
    refine_product_sum(a, b, c, d, first, second, sum)
}

#[cold]
fn refine_product_sum(
    a: Real,
    b: Real,
    c: Real,
    d: Real,
    first: Real,
    second: Real,
    sum: Real,
) -> Real {
    if (!sum.is_finite() || lost_product(first, a, b) || lost_product(second, c, d))
        && [a, b, c, d].into_iter().all(Real::is_finite)
    {
        let scaled = ScaledProduct::product(a, b).add(ScaledProduct::product(c, d));
        scale_power(scaled.mantissa, scaled.exponent)
    } else {
        compensated_sum(a, b, c, d, first, second)
    }
}

#[inline]
fn lost_product(product: Real, a: Real, b: Real) -> bool {
    !product.is_normal() && a != 0.0 && b != 0.0
}

/// A cancellation residual below the normal range must not be rounded
/// before division by a potentially tiny denominator.
fn quotient_numerator_lost_range(sum: Real, first: Real, second: Real) -> bool {
    !sum.is_normal()
        && first != 0.0
        && second != 0.0
        && first.abs().max(second.abs()) < Real::from_bits(54_u64 << 52)
}

/// A number represented outside the native binary64 exponent range.
#[derive(Clone, Copy)]
struct ScaledNumber {
    mantissa: Real,
    exponent: i32,
}

impl ScaledNumber {
    fn divide(self, rhs: Self) -> Real {
        scale_power(self.mantissa / rhs.mantissa, self.exponent - rhs.exponent)
    }
}

/// An exponent-scaled product with its exact rounding residual. Keeping
/// products independent preserves each quotient component even with a
/// very large dynamic range between the real and imaginary inputs. The
/// residual also recovers cancellation between two products that round
/// to the same significand.
#[derive(Clone, Copy)]
struct ScaledProduct {
    value: ScaledNumber,
    roundoff: Real,
}

impl ScaledProduct {
    fn product(a: Real, b: Real) -> Self {
        if a == 0.0 || b == 0.0 {
            return Self {
                value: ScaledNumber {
                    mantissa: a * b,
                    exponent: 0,
                },
                roundoff: 0.0,
            };
        }
        let (a, ae) = split_power(a);
        let (b, be) = split_power(b);
        let mantissa = a * b;
        Self {
            value: ScaledNumber {
                mantissa,
                exponent: ae + be,
            },
            // Both scaled factors have magnitude in [1, 2), so this
            // error-free product cannot overflow or underflow.
            roundoff: a.mul_add(b, -mantissa),
        }
    }

    fn add(self, other: Self) -> ScaledNumber {
        let x = self.value;
        let y = other.value;
        if x.mantissa == 0.0 && y.mantissa == 0.0 {
            return ScaledNumber {
                mantissa: x.mantissa + y.mantissa,
                exponent: 0,
            };
        }
        if x.mantissa == 0.0 {
            return y;
        }
        if y.mantissa == 0.0 {
            return x;
        }
        let exponent = x.exponent.max(y.exponent);
        let a = scale_power(x.mantissa, x.exponent - exponent);
        let b = scale_power(y.mantissa, y.exponent - exponent);
        let sum = a + b;
        // Knuth's TwoSum: keep the addition rounding residual too. This
        // is strictly on the exceptional path, never ordinary C arithmetic.
        let virtual_b = sum - a;
        let residual = (a - (sum - virtual_b)) + (b - virtual_b);
        let roundoff = residual
            + scale_power(self.roundoff, x.exponent - exponent)
            + scale_power(other.roundoff, y.exponent - exponent);
        ScaledNumber {
            mantissa: sum + roundoff,
            exponent,
        }
    }
}

#[cold]
fn scaled_complex_div(a: Complex, b: Complex) -> Complex {
    let p = ScaledProduct::product;
    let denominator = p(b.re, b.re).add(p(b.im, b.im));
    Complex::new(
        p(a.re, b.re).add(p(a.im, b.im)).divide(denominator),
        p(a.im, b.re).add(p(-a.re, b.im)).divide(denominator),
    )
}

/// Exact signed binary64 product as an integer significand and exponent.
/// The input pair is finite; its product needs at most 106 bits.
#[derive(Clone, Copy)]
struct ExactTerm {
    significand: i128,
    exponent: i32,
}

impl ExactTerm {
    fn new(a: Real, b: Real) -> Self {
        let unpack = |v: Real| {
            let bits = v.to_bits();
            let biased = i32::try_from((bits >> 52) & 0x7ff).unwrap_or_default();
            let fraction = bits & 0x000f_ffff_ffff_ffff;
            let significand = if biased == 0 {
                fraction
            } else {
                fraction | (1 << 52)
            };
            (significand, biased.max(1) - 1075)
        };
        let (a_int, a_exp) = unpack(a);
        let (b_int, b_exp) = unpack(b);
        let abs = i128::from(a_int) * i128::from(b_int);
        Self {
            significand: if a.is_sign_negative() == b.is_sign_negative() {
                abs
            } else {
                -abs
            },
            exponent: a_exp + b_exp,
        }
    }
}

/// Exact sum of up to four products, before normalization and rounding.
/// With exponents within 19 bits, four 106-bit significands fit in a
/// signed i128. Use the full finite-binary64 superaccumulator only when
/// the products span a wider range.
#[cold]
fn exact_product_sum<const N: usize>(pairs: [(Real, Real); N]) -> ScaledNumber {
    let terms = pairs.map(|(a, b)| ExactTerm::new(a, b));
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    for term in terms.iter().filter(|term| term.significand != 0) {
        min = min.min(term.exponent);
        max = max.max(term.exponent);
    }
    if min > max {
        return ScaledNumber {
            mantissa: 0.,
            exponent: 0,
        };
    }
    if max - min <= 19 {
        let sum = terms
            .iter()
            .filter(|term| term.significand != 0)
            .map(|term| term.significand << (term.exponent - min))
            .sum::<i128>();
        return scaled_integer(sum, min);
    }
    wide_product_sum(&terms)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "round the exact integer sum once to binary64"
)]
fn scaled_integer(sum: i128, exponent: i32) -> ScaledNumber {
    if sum == 0 {
        return ScaledNumber {
            mantissa: 0.,
            exponent: 0,
        };
    }
    let (mantissa, shift) = split_power(sum as Real);
    ScaledNumber {
        mantissa,
        exponent: exponent + shift,
    }
}

/// 66 limbs cover the full finite binary64 product range, including
/// four maximal products. Positive and negative sums stay separate
/// until subtraction, so even cancellation across a huge exponent
/// spread cannot erase a nonzero determinant.
#[cold]
fn wide_product_sum(terms: &[ExactTerm]) -> ScaledNumber {
    const LIMBS: usize = 66;
    const MIN_EXP: i32 = -2148;
    let mut positive = [0_u64; LIMBS];
    let mut negative = [0_u64; LIMBS];
    for term in terms.iter().filter(|term| term.significand != 0) {
        let product = term.significand.unsigned_abs();
        let shift = usize::try_from(term.exponent - MIN_EXP).unwrap_or_default();
        let index = shift / 64;
        let offset = shift % 64;
        let bytes = product.to_le_bytes();
        let lo = u64::from_le_bytes(*bytes.first_chunk::<8>().unwrap_or(&[0; 8]));
        let hi = u64::from_le_bytes(*bytes.last_chunk::<8>().unwrap_or(&[0; 8]));
        let parts = if offset == 0 {
            [lo, hi, 0]
        } else {
            [
                lo << offset,
                (hi << offset) | (lo >> (64 - offset)),
                hi >> (64 - offset),
            ]
        };
        let out = if term.significand > 0 {
            &mut positive
        } else {
            &mut negative
        };
        let mut carry = false;
        for (i, slot) in out[index..].iter_mut().enumerate() {
            let (value, overflow) = slot.overflowing_add(parts.get(i).copied().unwrap_or_default());
            let (value, next) = value.overflowing_add(u64::from(carry));
            *slot = value;
            carry = overflow || next;
            if i >= 2 && !carry {
                break;
            }
        }
    }
    let negative_sign = positive.iter().rev().cmp(negative.iter().rev()).is_lt();
    let (mut magnitude, subtrahend) = if negative_sign {
        (negative, positive)
    } else {
        (positive, negative)
    };
    let mut borrow = false;
    for (value, other) in magnitude.iter_mut().zip(subtrahend) {
        let (diff, underflow) = value.overflowing_sub(other);
        let (diff, next) = diff.overflowing_sub(u64::from(borrow));
        *value = diff;
        borrow = underflow || next;
    }
    let Some(index) = magnitude.iter().rposition(|&word| word != 0) else {
        return ScaledNumber {
            mantissa: 0.,
            exponent: 0,
        };
    };
    let high_bit = index * 64 + usize::try_from(magnitude[index].ilog2()).unwrap_or_default();
    let exponent = i32::try_from(high_bit).unwrap_or_default() + MIN_EXP;
    let shift = high_bit.saturating_sub(52);
    let limb = shift / 64;
    let offset = shift % 64;
    let mut significand = magnitude[limb] >> offset;
    if offset != 0 {
        significand |= magnitude.get(limb + 1).copied().unwrap_or_default() << (64 - offset);
    }
    if high_bit < 52 {
        significand <<= 52 - high_bit;
    } else if shift != 0 {
        let halfway = shift - 1;
        let halfway_limb = halfway / 64;
        let halfway_mask = 1_u64 << (halfway % 64);
        let halfway_set = magnitude[halfway_limb] & halfway_mask != 0;
        let sticky = magnitude[halfway_limb] & (halfway_mask - 1) != 0
            || magnitude[..halfway_limb].iter().any(|&word| word != 0);
        significand += u64::from(halfway_set && (sticky || significand & 1 != 0));
    }
    // The implicit bit at 52 supplies the low exponent bit; adding
    // rather than masking also handles rounding the significand to 2.
    let mantissa = Real::from_bits((1022_u64 << 52) + significand);
    ScaledNumber {
        mantissa: if negative_sign { -mantissa } else { mantissa },
        exponent,
    }
}

/// Accurate complex value minus product. Ordinary inputs use a direct
/// three-term update; cancellation-sensitive components are recovered
/// from the original operands, rather than a separately rounded product.
#[inline]
pub(crate) fn complex_mul_sub(value: Complex, a: Complex, b: Complex) -> Complex {
    let ac = a.re * b.re;
    let bd = a.im * b.im;
    let ad = a.re * b.im;
    let bc = a.im * b.re;
    let real = value.re - (ac - bd);
    let imag = value.im - (ad + bc);
    let result = Complex::new(real, imag);
    let scale = value.re.abs().max(value.im.abs()) * 0.0625;
    // Complex multiplication is normwise well-conditioned. A small
    // individual component does not by itself imply a cancellation of
    // the vector update. Recompute when the entire update nearly cancels,
    // or when its exponent range threatens to erase it.
    if result.is_finite()
        && ((real.abs() > scale || imag.abs() > scale) && (real.is_normal() || imag.is_normal()))
    {
        return result;
    }
    if !value.is_finite()
        || !a.is_finite()
        || !b.is_finite()
        || a == Complex::ZERO
        || b == Complex::ZERO
    {
        return result;
    }
    complex_mul_sub_recover(value, a, b)
}

#[cold]
fn complex_mul_sub_recover(value: Complex, a: Complex, b: Complex) -> Complex {
    let restore = |terms: [(Real, Real); 3]| {
        let exact = exact_product_sum(terms);
        scale_power(exact.mantissa, exact.exponent)
    };
    Complex::new(
        restore([(value.re, 1.), (-a.re, b.re), (a.im, b.im)]),
        restore([(value.im, 1.), (-a.re, b.im), (-a.im, b.re)]),
    )
}

/// Accurate (value * pivot - upper * lower) / pivot. All four inputs
/// are original, unnormalized coefficients of the current Schur step.
#[cold]
pub(crate) fn real_schur(value: Real, pivot: Real, upper: Real, lower: Real) -> Real {
    let product = upper * lower;
    let other = value * pivot;
    // Normal products with enough exponent headroom have an exact FMA
    // rounding residual. A two-product difference needs no wide accumulator.
    if product.is_normal()
        && other.is_normal()
        && product.abs().min(other.abs()) >= Real::from_bits(55_u64 << 52)
        && product.abs().max(other.abs()) < Real::MAX * 0.5
    {
        return (value.mul_add(pivot, -product) - upper.mul_add(lower, -product)) / pivot;
    }
    let numerator = exact_product_sum([(value, pivot), (-upper, lower)]);
    let (mantissa, exponent) = split_power(pivot);
    numerator.divide(ScaledNumber { mantissa, exponent })
}

/// Same Schur update for complex entries. Accumulate the cross products
/// before rounding, then divide in scaled form, preserving independently
/// scaled real and imaginary numerator components.
#[cold]
pub(crate) fn complex_schur(
    value: Complex,
    pivot: Complex,
    upper: Complex,
    lower: Complex,
) -> Complex {
    let real = exact_product_sum([
        (value.re, pivot.re),
        (-value.im, pivot.im),
        (-upper.re, lower.re),
        (upper.im, lower.im),
    ]);
    let imag = exact_product_sum([
        (value.re, pivot.im),
        (value.im, pivot.re),
        (-upper.re, lower.im),
        (-upper.im, lower.re),
    ]);
    let times = |x: ScaledNumber, v: Real| {
        let mut product = ScaledProduct::product(x.mantissa, v);
        product.value.exponent += x.exponent;
        product
    };
    let p = ScaledProduct::product;
    let denominator = p(pivot.re, pivot.re).add(p(pivot.im, pivot.im));
    Complex::new(
        times(real, pivot.re)
            .add(times(imag, pivot.im))
            .divide(denominator),
        times(imag, pivot.re)
            .add(times(real, -pivot.im))
            .divide(denominator),
    )
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl fmt::Display for Complex {
    /// ngspice's ASCII rawfile spelling: `re` when real, `re,im` otherwise.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im == 0.0 {
            write!(f, "{}", format_spice_number(self.re))
        } else {
            write!(
                f,
                "{},{}",
                format_spice_number(self.re),
                format_spice_number(self.im)
            )
        }
    }
}

/// A numeric literal plus how many bytes of the input it used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParsedNumber {
    /// The value, with the scale factor applied.
    pub value: Real,
    /// Bytes of the input consumed, including the scale suffix.
    pub consumed: usize,
}

/// `INPevaluate`'s decimal exponent adjustment and byte width of a scale
/// suffix, plus whether it is `MIL` (which also scales the mantissa by 25.4).
///
/// `letter` is the first letter after the numeric part, `rest` the text that
/// follows it. `MEG` and `MIL` are three bytes wide, everything else one.
/// Other letters (units such as `V`, `F`, `ohm`) do not scale.
fn scale_suffix(letter: char, rest: &str) -> (i32, usize, bool) {
    fn starts_with_ignore_case(text: &str, prefix: &str) -> bool {
        text.len() >= prefix.len()
            && text.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
    }

    match letter.to_ascii_lowercase() {
        't' => (12, 1, false),
        'g' => (9, 1, false),
        'k' => (3, 1, false),
        'u' => (-6, 1, false),
        'n' => (-9, 1, false),
        'p' => (-12, 1, false),
        'f' => (-15, 1, false),
        'a' => (-18, 1, false),
        'm' => {
            if starts_with_ignore_case(rest, "eg") {
                (6, 3, false)
            } else if starts_with_ignore_case(rest, "il") {
                (-6, 3, true)
            } else {
                (-3, 1, false)
            }
        }
        _ => (0, 0, false),
    }
}

/// Consume consecutive decimal digits with the reference mantissa accumulator.
#[expect(
    clippy::suboptimal_flops,
    reason = "INPevaluate rounds multiplication by ten before adding the digit"
)]
fn scan_mantissa_digits(bytes: &[u8], index: &mut usize, mantissa: &mut Real) -> usize {
    let start = *index;
    while bytes.get(*index).is_some_and(u8::is_ascii_digit) {
        *mantissa = 10.0 * *mantissa + Real::from(bytes[*index] - b'0');
        *index += 1;
    }
    *index - start
}

/// Parses a literal from the start of `input`, reporting bytes consumed.
///
/// Does not skip leading whitespace: the literal must begin at byte 0. Returns
/// `None` when no numeric part is present. The numeric part is
/// `[+-]?(digits[.digits?]|.digits)([eEdD][+-]?digits)?`, optionally followed
/// by a scale suffix.
///
/// The value is computed exactly as `INPevaluate()` (`inpeval.c`) does, not
/// as the nearest double to the decimal literal: the digits accumulate into a
/// double mantissa (`mantis = 10 * mantis + digit`, fraction digits counted
/// as negative powers), `MIL` multiplies the mantissa by 25.4, and the result
/// is `sign * mantis * pow(10, expo)`. So `1.2` is `12 * 0.1`,
/// `1.2000000000000002`, as in ngspice.
///
/// # Divergence from `INPevaluate()`
///
/// `INPevaluate()` applies the suffix multiplier but leaves `MEG`/`MIL`
/// unconsumed in its output pointer, because its callers do the skipping.
/// This function consumes the whole recognised suffix, so `consumed` is the
/// full width. An exponent marker without digits (`1e`) is left for the
/// caller; C would consume it with exponent zero, which gives the same value
/// unless a scale letter follows it.
#[must_use]
pub fn parse_spice_number_prefix(input: &str) -> Option<ParsedNumber> {
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut sign = 1.0;
    match bytes.first() {
        Some(b'+') => i += 1,
        Some(b'-') => {
            i += 1;
            sign = -1.0;
        }
        _ => {}
    }

    let mut mantis: Real = 0.0;
    let integer_digits = scan_mantissa_digits(bytes, &mut i, &mut mantis);
    let fractional_digits = if bytes.get(i) == Some(&b'.') {
        i += 1;
        scan_mantissa_digits(bytes, &mut i, &mut mantis)
    } else {
        0
    };
    if integer_digits == 0 && fractional_digits == 0 {
        return None;
    }
    // Match the old saturating decrement once per fractional digit.
    let expo1 = i32::try_from(fractional_digits).map_or(i32::MIN, |count| -count);

    let mut expo2: i32 = 0;
    if bytes
        .get(i)
        .is_some_and(|byte| matches!(byte.to_ascii_lowercase(), b'e' | b'd'))
    {
        let mut exponent_end = i + 1;
        let mut negative = false;
        match bytes.get(exponent_end) {
            Some(b'+') => exponent_end += 1,
            Some(b'-') => {
                exponent_end += 1;
                negative = true;
            }
            _ => {}
        }
        let exponent_digits_start = exponent_end;
        let mut exponent: i32 = 0;
        while exponent_end < bytes.len() && bytes[exponent_end].is_ascii_digit() {
            exponent = exponent
                .saturating_mul(10)
                .saturating_add(i32::from(bytes[exponent_end] - b'0'));
            exponent_end += 1;
        }
        // `1e` with nothing after it is a mantissa of `1` followed by junk, not
        // an exponent; leave it for the caller.
        if exponent_end > exponent_digits_start {
            i = exponent_end;
            expo2 = if negative { -exponent } else { exponent };
        }
    }

    let (scale, suffix_width, mil) = input[i..].chars().next().map_or((0, 0, false), |letter| {
        scale_suffix(letter, &input[i + letter.len_utf8()..])
    });
    if mil {
        mantis *= 25.4;
    }
    let expo = expo1.saturating_add(scale).saturating_add(expo2);

    Some(ParsedNumber {
        value: sign * mantis * Real::from(expo).powf_c(),
        consumed: i + suffix_width,
    })
}

/// Parse numeric literals and unit suffixes accepted by numparam.
///
/// Parses a number inside a `.param`/`{...}` expression as numparam's
/// `fetchnumber()` (`xpressn.c`) does: `sscanf("%lG")` (the correctly rounded
/// decimal value, `e`/`E` exponents only) times `parseunit()`'s multiplier
/// for the next letter (`T G K MEG M U N P F A`; `MIL` is milli there).
/// `consumed` covers the numeric part only; the caller swallows the unit
/// letters, as `fetchnumber()` does.
#[must_use]
pub fn parse_numparam_number_prefix(input: &str) -> Option<ParsedNumber> {
    let bytes = input.as_bytes();
    let mut i = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let mut digits = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        digits += 1;
    }
    if bytes.get(i) == Some(&b'.') {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if bytes
        .get(i)
        .is_some_and(|byte| byte.eq_ignore_ascii_case(&b'e'))
    {
        let mut end = i + 1;
        if matches!(bytes.get(end), Some(b'+' | b'-')) {
            end += 1;
        }
        let start = end;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end > start {
            i = end;
        }
    }
    let value: Real = input[..i].parse().ok()?;
    let rest = &input[i..];
    let unit = match rest.bytes().next().map(|b| b.to_ascii_uppercase()) {
        Some(b'T') => 1e12,
        Some(b'G') => 1e9,
        Some(b'K') => 1e3,
        Some(b'M') if rest.len() >= 3 && rest[..3].eq_ignore_ascii_case("meg") => 1e6,
        Some(b'M') => 1e-3,
        Some(b'U') => 1e-6,
        Some(b'N') => 1e-9,
        Some(b'P') => 1e-12,
        Some(b'F') => 1e-15,
        Some(b'A') => 1e-18,
        _ => 1.0,
    };
    Some(ParsedNumber {
        value: value * unit,
        consumed: i,
    })
}

/// A complete numparam number ([`parse_numparam_number_prefix`]) followed by
/// nothing but unit letters, which `fetchnumber()` swallows.
#[must_use]
pub fn parse_numparam_number(input: &str) -> Option<Real> {
    let trimmed = input.trim();
    let parsed = parse_numparam_number_prefix(trimmed)?;
    trimmed[parsed.consumed..]
        .chars()
        .all(|c| c.is_ascii_alphabetic())
        .then_some(parsed.value)
}

/// `pow(10.0, expo)` through the platform `pow`, as C calls it.
trait PowTen {
    fn powf_c(self) -> Real;
}

impl PowTen for Real {
    fn powf_c(self) -> Real {
        Self::powf(10.0, self)
    }
}

/// Parses a complete SPICE numeric literal, scale factor included.
///
/// Trailing letters are treated as a unit name and ignored, as ngspice does:
/// `5V` is `5`, `1kohm` is `1000`. Leading and trailing whitespace is allowed.
///
/// # Not yet ported
///
/// The RKM-style literals accepted by `inp2r.c`, `inp2c.c` and `inp2l.c` — `4k7`
/// for `4.7k`, `2M2` for `2.2M` — are **not** handled, and this function returns
/// `None` for them. See `docs/port/ARCHITECTURE.md` for the divergence record.
#[must_use]
pub fn parse_spice_number(input: &str) -> Option<Real> {
    let trimmed = input.trim_start();
    let parsed = parse_spice_number_prefix(trimmed)?;
    let rest = trimmed[parsed.consumed..].trim();
    if rest.is_empty() || rest.chars().all(|c| c.is_ascii_alphabetic()) {
        Some(parsed.value)
    } else {
        None
    }
}

/// Formats a real the way ngspice writes ASCII rawfiles: `%-.15e`, always with a
/// signed two-digit exponent.
///
/// `5.0` becomes `5.000000000000000e+00` and `-2.5e-3` becomes
/// `-2.500000000000000e-03`, matching `src/frontend/rawfile.c`.
#[must_use]
pub fn format_spice_number(value: Real) -> String {
    let text = format!("{value:.15e}");
    let Some((mantissa, exponent)) = text.split_once('e') else {
        // `NaN`, `inf` and `-inf` have no exponent part.
        return text;
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return text;
    };
    format!(
        "{mantissa}e{}{:02}",
        if exponent < 0 { '-' } else { '+' },
        exponent.unsigned_abs()
    )
}

#[cfg(test)]
mod tests {
    use super::{
        Complex, almost_equal_ulps, approx_eq, format_spice_number, parse_numparam_number_prefix,
        parse_spice_number, parse_spice_number_prefix,
    };

    #[test]
    fn scaled_division_recovers_product_cancellation_below_one_ulp() {
        let epsilon = f64::EPSILON;
        for scale in [2_f64.powi(-700), 2_f64.powi(700)] {
            let a = Complex::new((1. + epsilon) * scale, -scale);
            let b = Complex::new((1. - epsilon) * scale, scale);
            let quotient = a / b;
            // Exact real numerator is -epsilon^2 * scale^2, even though
            // the two rounded products cancel. It must not become zero.
            let expected = -(epsilon * epsilon) / ((1. - epsilon) * (1. - epsilon) + 1.);
            assert!((quotient.re / expected - 1.).abs() < 1e-14);
            assert!((quotient.im + 1.).abs() < 1e-14);
        }
    }

    #[test]
    fn complex_division_handles_normal_and_extreme_components() {
        let a = Complex::new(1.23456789, -9.87654321);
        let b = Complex::new(-0.371, 2.415);
        // Correct rounding of the exact rational quotient of the binary64
        // input values (not the original cancellation-prone calculation).
        let expected = Complex::new(-4.072_097_521_006_333_5, 0.114_360_368_651_490_67);
        let actual = a / b;
        assert!(almost_equal_ulps(actual.re, expected.re, 2));
        assert!(almost_equal_ulps(actual.im, expected.im, 2));
        for scale in [1e-300, 1e-200, 1e200, 1e300, f64::MAX] {
            let z = Complex::new(scale, scale);
            assert_eq!(z / z, Complex::real(1.));
            let q = Complex::new(scale, -scale) / z;
            assert_eq!(q, Complex::imaginary(-1.));
            let reciprocal = Complex::real(1.) / z;
            assert!(reciprocal.is_finite() && reciprocal.re > 0.);
            assert!((reciprocal.re * scale - 0.5).abs() < 1e-14);
        }
        // Scaling the entire numerator first would erase its small component.
        let q = Complex::new(1e300, 1e-300) / Complex::real(1e-10);
        assert!(q.re.is_infinite());
        assert!((q.im / 1e-290 - 1.).abs() < 1e-14);
        let q = Complex::new(1e150, 1e-200) / Complex::real(1e-150);
        assert!((q.re / 1e300 - 1.).abs() < 1e-14);
        assert!((q.im / 1e-50 - 1.).abs() < 1e-14);
        let smallest = f64::from_bits(1);
        assert_eq!(
            Complex::new(smallest, smallest) / Complex::new(smallest, smallest),
            Complex::real(1.)
        );
    }

    #[test]
    fn ordinary_complex_arithmetic_recovers_product_roundoff() {
        let epsilon = f64::EPSILON;
        let a = Complex::new(1. + epsilon, 1.);
        let b = Complex::new(1. - epsilon, 1.);
        let expected = -(epsilon * epsilon);
        assert_eq!((a * b).re, expected);
        let q = Complex::new(a.re, -a.im) / b;
        assert_eq!(q.re, expected / ((1. - epsilon).powi(2) + 1.));
        assert_eq!((b * a).re, expected);
    }

    #[test]
    fn exact_products_preserve_cancellation_across_the_full_exponent_range() {
        let tiny = Real::from_bits(1);
        let huge = Real::MAX;
        let exact = exact_product_sum([(huge, huge), (-huge, huge), (tiny, tiny)]);
        assert_eq!(exact.mantissa, 1.);
        assert_eq!(exact.exponent, -2148);
        let exact = exact_product_sum([(huge, huge), (-huge, huge), (-tiny, tiny)]);
        assert_eq!(exact.mantissa, -1.);
        assert_eq!(exact.exponent, -2148);
        let power = scale_power(1., 1023);
        let exact = exact_product_sum([(power, power); 4]);
        assert_eq!(exact.mantissa, 1.);
        assert_eq!(exact.exponent, 2048);
        // The wide sum rounds halfway to even, but a sticky bit at the
        // opposite end of the accumulator changes the rounded result.
        let half = Real::EPSILON / 2.;
        for (extra, expected) in [(0., 1.), (tiny, 1.0_f64.next_up())] {
            let exact = exact_product_sum([(1., 1.), (half, 1.), (extra, 1.)]);
            assert_eq!(scale_power(exact.mantissa, exact.exponent), expected);
        }
    }

    #[test]
    fn unnormalized_schur_recovers_tiny_nonzero_determinants() {
        for (pivot_exp, lower_exp) in [(700, 700), (-500, -700), (500, -500)] {
            let pivot = scale_power(1., pivot_exp);
            let lower = scale_power(1., lower_exp);
            let value = lower.next_up();
            assert_eq!(real_schur(lower, pivot, pivot, lower), 0.);
            assert_eq!(real_schur(value, pivot, pivot, lower), value - lower);
        }
        let pivot = Complex::new(scale_power(1., 500), scale_power(1., -500));
        let lower = Complex::new(scale_power(1., 600), scale_power(1., -600));
        let value = Complex::new(lower.re, lower.im.next_up());
        assert_eq!(complex_schur(lower, pivot, pivot, lower), Complex::ZERO);
        let result = complex_schur(value, pivot, pivot, lower);
        let expected = value.im - lower.im;
        assert!((result.im / expected - 1.).abs() < 1e-14);
        assert!(result.re.abs() <= expected.abs() * 1e-14);
    }

    #[test]
    fn quotient_retains_cancellation_before_subnormal_rounding() {
        let e = f64::EPSILON;
        let scale = 2_f64.powi(-500);
        let a = Complex::new((1. + e) * scale, -scale);
        let b = Complex::new((1. - e) * scale, scale);
        let q = a / b;
        let expected = -(e * e) / ((1. - e).powi(2) + 1.);
        assert!((q.re / expected - 1.).abs() < 1e-14);
        assert!((q.im + 1.).abs() < 1e-14);
    }

    #[test]
    fn complex_product_recovers_finite_components_across_exponent_limits() {
        let large = Complex::new(f64::MAX, f64::MAX);
        let z = large * Complex::new(2., 2.);
        assert_eq!(z.re, 0.);
        assert!(z.im.is_infinite() && z.im.is_sign_positive());
        let tiny = f64::from_bits(1);
        let z = Complex::new(tiny, tiny) * Complex::new(0.5, -0.5);
        assert_eq!(z.re, tiny);
        assert_eq!(z.im, 0.);
        let z = Complex::new(1e300, 1e-300) * Complex::real(1e-10);
        assert!((z.re / 1e290 - 1.).abs() < 1e-14);
        assert!((z.im / 1e-310 - 1.).abs() < 1e-13);
    }

    #[test]
    fn complex_division_keeps_signed_zero_on_both_paths() {
        for divisor in [1., 1e300, 1e-300, f64::from_bits(1)] {
            let q = Complex::new(0., -0.) / Complex::real(divisor);
            assert_eq!(q.re.to_bits(), 0_f64.to_bits());
            assert_eq!(q.im.to_bits(), (-0_f64).to_bits());
        }
    }

    #[test]
    fn parses_plain_numbers() {
        assert_eq!(parse_spice_number("1"), Some(1.0));
        assert_eq!(parse_spice_number("1.5"), Some(1.5));
        assert_eq!(parse_spice_number(".5"), Some(0.5));
        assert_eq!(parse_spice_number("-2.5e-3"), Some(-2.5e-3));
        assert_eq!(parse_spice_number("+1E3"), Some(1000.0));
        assert_eq!(parse_spice_number("  42  "), Some(42.0));
    }

    #[test]
    fn scale_factors_match_inpevaluate() {
        assert_eq!(parse_spice_number("1k"), Some(1000.0));
        assert_eq!(parse_spice_number("1K"), Some(1000.0));
        assert_eq!(parse_spice_number("1M"), Some(1e-3), "M is milli, not mega");
        assert_eq!(parse_spice_number("1m"), Some(1e-3));
        assert_eq!(parse_spice_number("1meg"), Some(1e6));
        assert_eq!(parse_spice_number("1MEG"), Some(1e6));
        assert_eq!(parse_spice_number("1MeG"), Some(1e6));
        // `mantis *= 25.4`, then `pow(10, -6)`.
        assert_eq!(parse_spice_number("1mil"), Some(25.4 * 1e-6));
        assert_eq!(parse_spice_number("1MIL"), Some(25.4 * 1e-6));
        assert_eq!(parse_spice_number("1u"), Some(1e-6));
        assert_eq!(parse_spice_number("1n"), Some(1e-9));
        assert_eq!(parse_spice_number("1p"), Some(1e-12));
        assert_eq!(parse_spice_number("1f"), Some(1e-15));
        assert_eq!(parse_spice_number("1a"), Some(1e-18));
        assert_eq!(parse_spice_number("1t"), Some(1e12));
        assert_eq!(parse_spice_number("1g"), Some(1e9));
        // INPevaluate has no `l` suffix: `2l` is 2 (a unit letter).
        assert_eq!(parse_spice_number("2l"), Some(2.0));
    }

    #[test]
    fn values_are_computed_as_inpevaluate_does() {
        // Digits accumulate into a double mantissa scaled by pow(10, expo),
        // so these differ from the nearest double; ngspice gives the same.
        assert_eq!(parse_spice_number("1.2"), Some(1.200_000_000_000_000_2));
        assert_eq!(parse_spice_number("1.2"), Some(12.0 * 10f64.powf(-1.0)));
        // numparam writes `{3}` back as "% 23.15e" text before INPevaluate.
        assert_eq!(
            parse_spice_number("3.000000000000000e+00"),
            Some(3e15 * 10f64.powf(-15.0))
        );
        // `d`/`D` is an exponent marker, as `e`.
        assert_eq!(parse_spice_number("1d3"), Some(1000.0));
        assert_eq!(
            parse_spice_number("-2.5D-3"),
            Some(-25.0 * 10f64.powf(-4.0))
        );
    }

    #[test]
    fn ulp_comparison_uses_a_wide_difference_for_distant_floats() {
        assert!(almost_equal_ulps(1., 1., 0));
        assert!(almost_equal_ulps(-0., 0., 0));
        assert!(almost_equal_ulps(f64::INFINITY, f64::INFINITY, 0));
        assert!(almost_equal_ulps(f64::NAN, f64::NAN, 0));
        assert!(almost_equal_ulps(1., f64::from_bits(1f64.to_bits() + 1), 1));
        assert!(!almost_equal_ulps(
            1.,
            f64::from_bits(1f64.to_bits() + 2),
            1
        ));
        // The old LTRA-local implementation computed this difference as i64
        // and overflowed in debug builds for widely separated values.
        assert!(!almost_equal_ulps(-f64::MAX, f64::MAX, 3));
        assert!(!almost_equal_ulps(f64::NEG_INFINITY, f64::INFINITY, 3));
        assert!(!almost_equal_ulps(-f64::MAX, f64::MAX, i64::MAX));
        // The sign-bit transformation must order adjacent negative values as
        // well as both sides of zero without overflowing at i64::MIN.
        let first_subnormal = f64::from_bits(1);
        let second_subnormal = f64::from_bits(2);
        assert!(almost_equal_ulps(-first_subnormal, -second_subnormal, 1));
        assert!(!almost_equal_ulps(-first_subnormal, second_subnormal, 2));
        assert!(almost_equal_ulps(-first_subnormal, first_subnormal, 2));
    }

    #[test]
    fn numparam_numbers_are_strtod_times_the_unit() {
        let value = |text| parse_numparam_number_prefix(text).map(|p| p.value);
        assert_eq!(value("1.2"), Some(1.2));
        assert_eq!(value("1.2k"), Some(1.2 * 1e3));
        assert_eq!(value("2meg"), Some(2e6));
        // parseunit() has no MIL: `1mil` is milli.
        assert_eq!(value("1mil"), Some(1e-3));
        // %lG has no `d` exponent.
        assert_eq!(
            parse_numparam_number_prefix("1d3").map(|p| p.consumed),
            Some(1)
        );
    }

    #[test]
    fn trailing_unit_names_are_ignored() {
        assert_eq!(parse_spice_number("5V"), Some(5.0));
        assert_eq!(parse_spice_number("1kohm"), Some(1000.0));
        assert_eq!(parse_spice_number("1000F"), Some(1e-15 * 1000.0));
    }

    #[test]
    fn rejects_non_literals() {
        assert_eq!(parse_spice_number(""), None);
        assert_eq!(parse_spice_number("abc"), None);
        assert_eq!(parse_spice_number("-"), None);
        assert_eq!(parse_spice_number("."), None);
        assert_eq!(parse_spice_number("2N2222"), None, "a BJT part number");
    }

    #[test]
    fn rkm_literals_are_not_yet_supported() {
        // Pins the divergence documented in `docs/port/ARCHITECTURE.md`.
        // `INPevaluateRKM_R()` in `src/spicelib/parser/inpeval.c` accepts these.
        assert_eq!(parse_spice_number("4k7"), None);
        assert_eq!(parse_spice_number("2M2"), None);
    }

    #[test]
    fn prefix_reports_bytes_consumed() {
        let parsed = parse_spice_number_prefix("1megohm").unwrap();
        assert_eq!(parsed.consumed, 4);
        assert_eq!(parsed.value, 1e6);

        let parsed = parse_spice_number_prefix("2.5uF").unwrap();
        assert_eq!(parsed.consumed, 4);
        // `INPevaluate()` computes `mantissa * pow(10, expo)`; it does not parse
        // the scaled literal directly, so the result is this product and not
        // necessarily the nearest double to `2.5e-6`.
        assert_eq!(parsed.value, 2.5 * 1e-6);

        let parsed = parse_spice_number_prefix("1e3").unwrap();
        assert_eq!(parsed.consumed, 3);
        assert_eq!(parsed.value, 1000.0);

        let parsed = parse_spice_number_prefix("-2.5e-3").unwrap();
        assert_eq!(parsed.consumed, 7);
        assert_eq!(parsed.value, -2.5e-3);
    }

    #[test]
    fn formats_like_the_rawfile_writer() {
        assert_eq!(format_spice_number(5.0), "5.000000000000000e+00");
        assert_eq!(format_spice_number(2.5), "2.500000000000000e+00");
        assert_eq!(format_spice_number(-2.5e-3), "-2.500000000000000e-03");
        assert_eq!(format_spice_number(0.0), "0.000000000000000e+00");
        assert_eq!(format_spice_number(1e100), "1.000000000000000e+100");
        assert_eq!(format_spice_number(f64::NAN), "NaN");
        assert_eq!(format_spice_number(f64::INFINITY), "inf");
        assert_eq!(format_spice_number(f64::NEG_INFINITY), "-inf");
        assert_eq!(format_spice_number(-0.0), "-0.000000000000000e+00");
    }

    #[test]
    fn formats_and_parses_round_trip() {
        for value in [0.0, 1.0, -1.0, 2.5e-3, 1e12, -3.75e-9] {
            let formatted = format_spice_number(value);
            let parsed = parse_spice_number(&formatted).expect("formatted value must parse");
            assert!(approx_eq(value, parsed, 1e-15), "{formatted}");
        }
    }

    #[test]
    fn complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);
        assert_eq!(a + b, Complex::new(4.0, 2.0));
        assert_eq!(a - b, Complex::new(2.0, 6.0));
        assert_eq!(a * b, Complex::new(11.0, -2.0));
        assert_eq!(a.conj(), Complex::new(3.0, -4.0));
        assert_eq!(a.magnitude(), 5.0);
        assert_eq!(Complex::real(2.0).to_string(), "2.000000000000000e+00");
        assert_eq!(
            b.to_string(),
            "1.000000000000000e+00,-2.000000000000000e+00"
        );
    }
}
