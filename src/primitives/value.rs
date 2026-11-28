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
//! | `U` | 1e-6 | |
//! | `N` | 1e-9 | |
//! | `P` | 1e-12 | |
//! | `F` | 1e-15 | |
//! | `A` | 1e-18 | accepted by `INPevaluate()` |
//!
//! An unrecognised trailing letter does not change the value: `5V` is `5`, and
//! `1kohm` is `1000`. Anything else after the literal — `1k2` — is rejected by
//! [`parse_spice_number`]; see the RKM note on that function.

use super::exact::{ScaledNumber, exact_product_sum};
use super::numeric::{DecimalMultiplier, rounded_decimal, scale_power, split_power};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The floating point type used throughout the port.
///
/// ngspice mixes `double` and `float`; the port uses `f64` everywhere and
/// records the places where the C code downcasts as a divergence risk.
pub type Real = f64;

/// Compares two reals with a relative tolerance, falling back to an absolute one
/// near zero.
#[must_use = "the result determines whether the values agree within the specified tolerance"]
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
#[must_use = "the result determines whether the values satisfy the ULP bound"]
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
    #[must_use = "the constructed value retains both complex components"]
    pub const fn new(re: Real, im: Real) -> Self {
        Self { re, im }
    }

    /// A purely real value.
    #[must_use = "the constructed complex value has a zero imaginary component"]
    pub const fn real(re: Real) -> Self {
        Self { re, im: 0.0 }
    }

    /// A purely imaginary value.
    #[must_use = "the constructed complex value has a zero real component"]
    pub const fn imaginary(im: Real) -> Self {
        Self { re: 0.0, im }
    }

    /// `sqrt(re^2 + im^2)`.
    #[must_use = "the returned scalar is the modulus of this complex value"]
    pub fn magnitude(self) -> Real {
        self.re.hypot(self.im)
    }

    /// The argument in radians.
    #[must_use = "the returned angle is the complex argument in radians"]
    pub fn phase(self) -> Real {
        self.im.atan2(self.re)
    }

    /// The complex conjugate.
    #[must_use = "the returned value reverses the imaginary sign without modifying the original"]
    pub const fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// True when both parts are finite.
    #[must_use = "the result detects nonfinite values in either complex component"]
    pub const fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// True when the imaginary part is zero.
    #[must_use = "the result determines whether the value has zero imaginary component"]
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

/// Literal dialects share decimal syntax and exact scaling, but not the
/// exponent markers, MIL suffix or consumed-letter convention.
#[derive(Clone, Copy)]
enum LiteralDialect {
    Spice,
    Numparam,
}

impl LiteralDialect {
    const fn exponent_marker(self, byte: u8) -> bool {
        match self {
            Self::Spice => matches!(byte, b'e' | b'E' | b'd' | b'D'),
            Self::Numparam => matches!(byte, b'e' | b'E'),
        }
    }

    fn suffix(self, rest: &[u8]) -> (i32, usize, DecimalMultiplier) {
        use DecimalMultiplier::{Mil, One};
        let three = rest.get(..3);
        match rest.first().map(u8::to_ascii_lowercase) {
            Some(b't') => (12, 1, One),
            Some(b'g') => (9, 1, One),
            Some(b'k') => (3, 1, One),
            Some(b'u') => (-6, 1, One),
            Some(b'n') => (-9, 1, One),
            Some(b'p') => (-12, 1, One),
            Some(b'f') => (-15, 1, One),
            Some(b'a') => (-18, 1, One),
            Some(b'm') if three.is_some_and(|s| s.eq_ignore_ascii_case(b"meg")) => (6, 3, One),
            // 25.4e-6 = 254e-7 exactly, not a binary64 multiplication.
            Some(b'm')
                if matches!(self, Self::Spice)
                    && three.is_some_and(|s| s.eq_ignore_ascii_case(b"mil")) =>
            {
                (-7, 3, Mil)
            }
            Some(b'm') => (-3, 1, One),
            _ => (0, 0, One),
        }
    }
}

/// Scan decimal digits and simultaneously compute a candidate short coefficient.
/// Wrapping is harmless for longer literals: the result is used only when
/// there were at most 16 digits (then the exact integer fits in u64).
fn scan_digits(bytes: &[u8], index: &mut usize, coefficient: &mut u64) -> usize {
    let start = *index;
    while let Some(&digit) = bytes.get(*index).filter(|digit| digit.is_ascii_digit()) {
        *coefficient = coefficient
            .wrapping_mul(10)
            .wrapping_add(u64::from(digit - b'0'));
        *index += 1;
    }
    *index - start
}

fn decimal_prefix(input: &str, dialect: LiteralDialect) -> Option<ParsedNumber> {
    let bytes = input.as_bytes();
    let negative = bytes.first() == Some(&b'-');
    let start = usize::from(matches!(bytes.first(), Some(b'+' | b'-')));
    let mut i = start;
    let mut coefficient = 0;
    let integer_digits = scan_digits(bytes, &mut i, &mut coefficient);
    let fractional_digits = if bytes.get(i) == Some(&b'.') {
        i += 1;
        scan_digits(bytes, &mut i, &mut coefficient)
    } else {
        0
    };
    if integer_digits == 0 && fractional_digits == 0 {
        return None;
    }
    let mantissa = &input[start..i];
    let mut exponent = 0_i128;
    if bytes
        .get(i)
        .is_some_and(|&byte| dialect.exponent_marker(byte))
    {
        let mut end = i + 1;
        let negative_exponent = bytes.get(end) == Some(&b'-');
        if matches!(bytes.get(end), Some(b'+' | b'-')) {
            end += 1;
        }
        let exponent_start = end;
        while let Some(&digit) = bytes.get(end).filter(|digit| digit.is_ascii_digit()) {
            exponent = exponent
                .saturating_mul(10)
                .saturating_add(i128::from(digit - b'0'));
            end += 1;
        }
        // Incomplete exponent markers remain for the caller, as before.
        if end > exponent_start {
            i = end;
            if negative_exponent {
                exponent = -exponent;
            }
        }
    }
    let (scale, suffix_width, multiplier) = dialect.suffix(&bytes[i..]);
    // A saturated exponent cannot be cancelled by the length of an actual
    // input slice, so saturating adjustment is exact for every finite result.
    let exponent = exponent
        .saturating_sub(i128::try_from(fractional_digits).ok()?)
        .saturating_add(i128::from(scale));
    let short_coefficient = (integer_digits + fractional_digits <= 16).then_some(coefficient);
    let value = rounded_decimal(mantissa, short_coefficient, negative, exponent, multiplier)?;
    Some(ParsedNumber {
        value,
        consumed: i + if matches!(dialect, LiteralDialect::Spice) {
            suffix_width
        } else {
            0
        },
    })
}

/// Parse a SPICE literal prefix with its scale suffix, without leading space.
///
/// Decimal digits, optional sign, point and e/E/d/D exponent are recognized
/// before the optional scale. The exact scaled decimal is rounded once to
/// nearest binary64. Overflow produces infinity, underflow gradual or signed
/// zero; semantic parsers retain their existing finite-value checks.
///
/// The full MEG/MIL suffix is consumed, unlike C's internal rest pointer.
/// A marker without exponent digits is left for the caller.
#[must_use = "the result contains the scaled value and consumed prefix length"]
pub fn parse_spice_number_prefix(input: &str) -> Option<ParsedNumber> {
    decimal_prefix(input, LiteralDialect::Spice)
}

/// Parse a numparam literal prefix.
///
/// Numparam recognizes only e/E exponents,
/// and MIL means milli, not thousandths of an inch. The scale is applied
/// before correct rounding, but consumed counts just the numeric part;
/// the expression parser swallows trailing unit letters separately.
#[must_use = "the result retains numparam scaling and token boundaries"]
pub fn parse_numparam_number_prefix(input: &str) -> Option<ParsedNumber> {
    decimal_prefix(input, LiteralDialect::Numparam)
}

/// A complete numparam number ([`parse_numparam_number_prefix`]) followed by
/// nothing but unit letters, which `fetchnumber()` swallows.
#[must_use = "the result validates and converts the complete numparam literal"]
pub fn parse_numparam_number(input: &str) -> Option<Real> {
    let trimmed = input.trim();
    let parsed = parse_numparam_number_prefix(trimmed)?;
    trimmed[parsed.consumed..]
        .chars()
        .all(|c| c.is_ascii_alphabetic())
        .then_some(parsed.value)
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
#[must_use = "the result validates and converts the complete SPICE literal"]
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

// Numparam and ASCII rawfiles both emit 16 significant decimal digits.
const SPICE_FRACTION_DIGITS: usize = 15;

/// The numeric result of numparam's decimal write/read boundary, without
/// allocating or normalizing an intermediate literal. The generated decimal
/// has no suffix or other SPICE-specific syntax, so the binary64 parser can
/// consume it directly. Nonfinite inputs are rejected just as literal parsing
/// rejects them; finite inputs may round to infinity at the upper boundary.
#[must_use = "the returned value preserves numparam decimal rounding"]
pub(crate) fn round_numparam_literal(value: Real) -> Option<Real> {
    use std::io::Write as _;

    if !value.is_finite() {
        return None;
    }
    // Sign, leading digit, point, 15 fraction digits, e, exponent sign and
    // at most three exponent digits cover the entire finite binary64 range.
    let mut buffer = [0_u8; 24];
    let capacity = buffer.len();
    let mut remaining = buffer.as_mut_slice();
    write!(remaining, "{value:.SPICE_FRACTION_DIGITS$e}").ok()?;
    let len = capacity - remaining.len();
    std::str::from_utf8(&buffer[..len]).ok()?.parse().ok()
}

/// Formats a real the way ngspice writes ASCII rawfiles: `%-.15e`, always with a
/// signed two-digit exponent.
///
/// `5.0` becomes `5.000000000000000e+00` and `-2.5e-3` becomes
/// `-2.500000000000000e-03`, matching `src/frontend/rawfile.c`.
#[must_use = "the returned text is the reference-compatible decimal representation"]
pub fn format_spice_number(value: Real) -> String {
    let text = format!("{value:.SPICE_FRACTION_DIGITS$e}");
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
        Complex, Real, almost_equal_ulps, approx_eq, complex_schur, exact_product_sum,
        format_spice_number, parse_numparam_number_prefix, parse_spice_number,
        parse_spice_number_prefix, real_schur, round_numparam_literal, scale_power,
    };

    #[test]
    fn numparam_numeric_rounding_matches_literal_and_c_oracle() {
        let check = |value: Real| {
            let expected = parse_spice_number(&format_spice_number(value)).map(Real::to_bits);
            let actual = round_numparam_literal(value).map(Real::to_bits);
            assert_eq!(actual, expected, "input bits {:016x}", value.to_bits());
            actual
        };
        for value in [
            0.0,
            -0.0,
            Real::MAX,
            -Real::MAX,
            Real::MIN_POSITIVE,
            Real::MIN_POSITIVE.next_down(),
            Real::from_bits(1),
            Real::from_bits(2),
            Real::INFINITY,
            Real::NEG_INFINITY,
            Real::NAN,
            2_147_483_647.5,
            -2_147_483_648.5,
            1.0_f64.next_up(),
            1.0_f64.next_down(),
            0.125,
            0.000_030_517_578_125,
        ] {
            check(value);
        }
        // Exact binary fractions that fall halfway between two 16-digit
        // decimals. C rounds down to the even last digit in the first case
        // and up in the second, for both signs.
        for (input, expected) in [
            (0x3ff0_0010_0000_0000, 0x3ff0_000f_ffff_fffe),
            (0x3ff0_0030_0000_0000, 0x3ff0_0030_0000_0002),
            (0xbff0_0010_0000_0000, 0xbff0_000f_ffff_fffe),
            (0xbff0_0030_0000_0000, 0xbff0_0030_0000_0002),
        ] {
            assert_eq!(check(Real::from_bits(input)), Some(expected));
        }
        // Decimal rounding may move a value across a later integer setter's
        // rounding boundary. Preserve the decimal read-back before that step.
        for value in [2_147_483_647.5_f64, -2_147_483_648.5_f64, 0.5_f64, -0.5_f64] {
            for input in [value.next_down(), value, value.next_up()] {
                let expected = parse_spice_number(&format_spice_number(input))
                    .and_then(crate::primitives::c_integer);
                assert_eq!(
                    round_numparam_literal(input).and_then(crate::primitives::c_integer),
                    expected
                );
            }
        }
        for exponent in -323..=308 {
            let power: Real = format!("1e{exponent}").parse().unwrap();
            for value in [power.next_down(), power, power.next_up()] {
                check(value);
                check(-value);
            }
        }
        // Independent C-locale snprintf("%.15e") / strtod oracle, generated
        // for the same xorshift64 sequence (nearest-even rounding).
        let mut bits = 0x1234_5678_9abc_def0_u64;
        let mut digest = 0xcbf2_9ce4_8422_2325_u64;
        let mut count = 0;
        for _ in 0..100_000 {
            bits ^= bits << 13;
            bits ^= bits >> 7;
            bits ^= bits << 17;
            if let Some(rounded) = check(Real::from_bits(bits)) {
                digest = (digest ^ rounded).wrapping_mul(1_099_511_628_211);
                count += 1;
            }
        }
        assert_eq!(count, 99_949);
        assert_eq!(digest, 0x29ef_5948_92db_bd6f_u64);
    }

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
        assert_eq!(parse_spice_number("1mil"), Some(25.4e-6));
        assert_eq!(parse_spice_number("1MIL"), Some(25.4e-6));
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
    fn literals_round_once_after_exact_scaling() {
        assert_eq!(parse_spice_number("1.2"), Some(1.2));
        assert_eq!(parse_spice_number("3.000000000000000e+00"), Some(3.));
        assert_eq!(parse_spice_number("1d3"), Some(1e3));
        assert_eq!(parse_spice_number("-2.5D-3"), Some(-2.5e-3));
        assert_eq!(parse_spice_number("1e312f"), Some(1e297));
        assert_eq!(parse_spice_number("1e-330meg"), Some(0.));
        assert_eq!(parse_spice_number("1e-330t"), Some(1e-318));
        assert_eq!(parse_spice_number("1e309mil"), Some(2.54e304));
        assert_eq!(parse_spice_number("1e-323mil"), Some(0.));
        assert_eq!(parse_spice_number("0e99999999999999999999999"), Some(0.));
        assert_eq!(
            parse_spice_number("-1e-99999999999999999999999")
                .unwrap()
                .to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            parse_spice_number("-0e99999999999999999999999")
                .unwrap()
                .to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            parse_spice_number("1e99999999999999999999999"),
            Some(f64::INFINITY)
        );
    }

    #[test]
    fn decimal_edges_match_independent_exact_rounding() {
        // Expected bits from exact decimal products converted with CPython
        // Decimal -> binary64, including boundaries on both sides of a tie.
        for (text, expected) in [
            ("9007199254740993", 0x4340_0000_0000_0000),
            ("9007199254740995", 0x4340_0000_0000_0002),
            ("9007199254740993e-3k", 0x4340_0000_0000_0000),
            ("1.7976931348623157e311m", f64::MAX.to_bits()),
            ("1.7976931348623159e308", f64::INFINITY.to_bits()),
            ("2.2250738585072011e-308", 0x000f_ffff_ffff_ffff),
            ("2.2250738585072014e-308", f64::MIN_POSITIVE.to_bits()),
            ("2.4703282292062327e-306a", 0),
            ("2.4703282292062328e-306a", 1),
            ("1.234567890123456789mil", 0x3f00_70cd_83b2_078b),
            ("9999999999999999mil", 0x424d_91ca_35ff_ffff),
            ("1.234567890123456789e12a", 0x3eb4_b66d_c01e_c6fb),
            ("-0.0", (-0_f64).to_bits()),
        ] {
            assert_eq!(
                parse_spice_number(text).unwrap().to_bits(),
                expected,
                "{text}"
            );
        }
    }

    #[test]
    fn decimal_sticky_tail_distinguishes_exact_halfway_cases() {
        // Exact midpoint between 1 and the next binary64, and perturbations
        // smaller than the retained 768 significant digits.
        let midpoint = "1.00000000000000011102230246251565404236316680908203125";
        let zeros = "0".repeat(1000);
        let tie = format!("{midpoint}{zeros}");
        let above = format!("{tie}1");
        let below = format!("{}4{}", &midpoint[..midpoint.len() - 1], "9".repeat(1000));
        for (text, bits) in [
            (tie, 1_f64.to_bits()),
            (above, 1_f64.to_bits() + 1),
            (below, 1_f64.to_bits()),
        ] {
            assert_eq!(parse_spice_number(&text).unwrap().to_bits(), bits);
            assert_eq!(
                parse_numparam_number_prefix(&text).unwrap().value.to_bits(),
                bits
            );
        }

        // MIL is the non-power-of-ten case. Construct a decimal from an
        // independent exact rational: midpoint / (254e-7). The odd factor 127
        // gives a repeating decimal, so truncation and incrementing the last
        // digit straddle the exact midpoint. Carries from the discarded
        // decimal tail must be taken into account *before* truncation.
        let denominator = (1_u128 << 53) * 254;
        let numerator = ((1_u128 << 53) + 1) * 10_000_000;
        let mut lower = format!("{}.", numerator / denominator);
        let mut remainder = numerator % denominator;
        while lower.len() < 1000 || lower.ends_with('9') {
            remainder *= 10;
            lower.push(char::from(
                b'0' + u8::try_from(remainder / denominator).unwrap(),
            ));
            remainder %= denominator;
        }
        assert_ne!(remainder, 0);
        let mut upper = lower.clone();
        let last = upper.pop().unwrap();
        upper.push(char::from_u32(u32::from(last) + 1).unwrap());
        assert_eq!(
            parse_spice_number(&format!("{lower}mil"))
                .unwrap()
                .to_bits(),
            1_f64.to_bits()
        );
        assert_eq!(
            parse_spice_number(&format!("{upper}mil"))
                .unwrap()
                .to_bits(),
            1_f64.to_bits() + 1
        );
    }

    #[test]
    fn huge_decimal_coefficients_cancel_exponents_without_intermediate_overflow() {
        let count = 100_000;
        let zeros = "0".repeat(count);
        let integer = format!("1{zeros}e-{count}");
        let fraction = format!("0.{zeros}12345e{}k", count + 1);
        assert_eq!(parse_spice_number(&integer), Some(1.));
        assert_eq!(parse_spice_number(&format!("{integer}mil")), Some(25.4e-6));
        assert_eq!(parse_spice_number(&fraction), Some(1234.5));
        assert_eq!(parse_numparam_number_prefix(&integer).unwrap().value, 1.);
    }

    #[test]
    fn decimal_prefix_boundaries_are_dialect_specific_and_utf8_safe() {
        for (text, spice, numparam, spice_width, numparam_width) in [
            ("1d2k", 100_000., 1., 4, 1),
            ("1e+", 1., 1., 1, 1),
            ("1MeGoHm", 1e6, 1e6, 4, 1),
            ("1mIl", 25.4e-6, 1e-3, 4, 1),
            ("1m²", 1e-3, 1e-3, 2, 1),
            ("1m€", 1e-3, 1e-3, 2, 1),
            ("1Ω", 1., 1., 1, 1),
            ("1l", 1., 1., 1, 1),
            ("1e-330T", 1e-318, 1e-318, 7, 6),
        ] {
            let a = parse_spice_number_prefix(text).unwrap();
            let b = parse_numparam_number_prefix(text).unwrap();
            assert_eq!((a.value, a.consumed), (spice, spice_width), "{text}");
            assert_eq!((b.value, b.consumed), (numparam, numparam_width), "{text}");
        }
        for text in ["1m²", "1m€", "1Ω", "1e+", "4k7", "2N2222"] {
            assert_eq!(parse_spice_number(text), None, "{text}");
        }
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
    fn numparam_retains_its_own_suffix_and_exponent_grammar() {
        let value = |text| parse_numparam_number_prefix(text).map(|p| p.value);
        assert_eq!(value("1.2"), Some(1.2));
        assert_eq!(value("1.2k"), Some(1.2e3));
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
        assert_eq!(parse_spice_number("1000F"), Some(1e-12));
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
        assert_eq!(parsed.value, 2.5e-6);

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
