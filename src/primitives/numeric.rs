//! Checked integral conversion and binary64 exponent/significand kernels.
//!
//! Rounding is a caller-selected operation: `IF_INTEGER` uses floor(x + 0.5),
//! while other C casts truncate toward zero. Never round an already integral
//! field again. Ordinary reference arithmetic must retain its operation order;
//! scaled complex recovery uses the separate exponent/significand helpers.
use super::Real;
use std::ops::RangeInclusive;

/// C's `INPgetValue(IF_INTEGER)` conversion (`inpgval.c`):
/// `(int) floor(0.5 + value)`, or `None` when that is not a finite `int`.
///
/// The signed C range is asymmetric: `i32::MIN` is a valid result, while
/// `-i32::MIN` is not. Check the actual bounds before the narrowing cast.
#[must_use = "the returned checked integer preserves C rounding and range validation"]
pub fn c_integer(value: Real) -> Option<i64> {
    exact_signed_integer::<i32>((0.5 + value).floor()).map(i64::from)
}

/// Convert an already integral, nonnegative `f64` to a checked integer
/// type without silently saturating an out-of-range value.
///
/// Rust's float-to-integer `as` conversion saturates on overflow. First
/// enforce the **exclusive** `2^128` bound, then use `TryFrom<u128>` for the
/// actual target integer type. This works for both `usize` and the full
/// `u32` range, even on targets whose `usize` is narrower than 32 bits.
/// It validates the supplied `f64`, not digits already lost during parsing.
#[must_use = "the returned conversion distinguishes valid integral magnitudes from overflow"]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is checked for integrality and bounded below 2^128 before conversion"
)]
#[expect(
    clippy::cast_sign_loss,
    reason = "the range check rejects every negative value except signed zero"
)]
pub fn exact_nonnegative_integer<T: TryFrom<u128>>(value: Real) -> Option<T> {
    let exclusive_upper_bound = Real::from(128).exp2();
    if !value.is_finite() || value < 0. || value.fract() != 0. || value >= exclusive_upper_bound {
        return None;
    }
    T::try_from(value as u128).ok()
}

/// Convert an integral finite value through the full signed i128 domain.
/// The lower bound is inclusive, but 2^127 is not representable as i128.
#[must_use = "the returned conversion distinguishes valid signed integers from overflow"]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the value is integral and within the asymmetric i128 range"
)]
pub fn exact_signed_integer<T: TryFrom<i128>>(value: Real) -> Option<T> {
    let bound = 2_f64.powi(127);
    if !value.is_finite() || value.fract() != 0. || value < -bound || value >= bound {
        return None;
    }
    T::try_from(value as i128).ok()
}

/// Truncate toward zero before checking the full signed target range (C casts
/// on representable values, not the `IF_INTEGER` parser's half-up rounding).
#[must_use = "the returned conversion preserves truncation and destination range validation"]
pub fn truncating_signed_integer<T: TryFrom<i128>>(value: Real) -> Option<T> {
    exact_signed_integer(value.trunc())
}

/// Quantize a finite exponent, ties away from zero, saturating at the given
/// integer limits. Invalid exponents or reversed bounds are rejected.
#[must_use = "the returned exponent preserves quantization and bounds validation"]
pub(crate) fn bounded_rounded_exponent(value: Real, bounds: RangeInclusive<i32>) -> Option<i32> {
    if !value.is_finite() || bounds.is_empty() {
        return None;
    }
    exact_signed_integer(
        value
            .round()
            .clamp(Real::from(*bounds.start()), Real::from(*bounds.end())),
    )
}

/// Powers of ten whose integer value is exactly representable in binary64:
/// 5^22 fits in 53 bits, while 5^23 does not. If the scaled coefficient is
/// also exact, one IEEE multiplication or division already rounds the
/// mathematical decimal correctly. No approximate reciprocal is used.
const EXACT_DECIMAL_POWERS: [Real; 23] = [
    1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16,
    1e17, 1e18, 1e19, 1e20, 1e21, 1e22,
];

#[expect(
    clippy::cast_precision_loss,
    reason = "the integer coefficient is checked to be at most 2^53 before conversion"
)]
fn short_decimal(coefficient: u64, exponent: i128, multiplier: DecimalMultiplier) -> Option<Real> {
    use std::io::Write as _;

    let coefficient = coefficient.checked_mul(u64::from(multiplier.factor()))?;
    if coefficient == 0 {
        return Some(0.);
    }
    if coefficient <= 1_u64 << 53
        && let Some(power) = usize::try_from(exponent.unsigned_abs())
            .ok()
            .and_then(|index| EXACT_DECIMAL_POWERS.get(index))
    {
        let coefficient = coefficient as Real;
        return Some(if exponent < 0 {
            coefficient / power
        } else {
            coefficient * power
        });
    }
    // Even a u64 cannot rescue a decimal exponent outside this range.
    if exponent > 400 {
        return Some(Real::INFINITY);
    }
    if exponent < -400 {
        return Some(0.);
    }
    // A short exact integer product needs at most twenty digits, so avoid
    // the general long-coefficient buffer even outside the arithmetic fast path.
    let mut text = [0; 32];
    let available = text.len();
    let mut remaining = text.as_mut_slice();
    write!(remaining, "{coefficient}e{exponent}").ok()?;
    let used = available - remaining.len();
    std::str::from_utf8(&text[..used]).ok()?.parse().ok()
}

/// Convert a validated decimal mantissa times an exact integer and power
/// of ten to nearest binary64 (ties to even). The sign is independent of the
/// magnitude so zero and gradual underflow retain negative zero.
///
/// Round once, after scaling. A binary64 rounding boundary is a dyadic
/// rational with a numerator of at most 54 bits and a denominator at most
/// 2^1075 (or an integer of at most 309 decimal digits). Its terminating
/// decimal therefore needs at most ceil(54 log10(2) + 1075 log10(5)) = 768
/// significant digits. Retaining 768 leading digits and a sticky digit is
/// enough: no rounding boundary can lie strictly inside the discarded tail
/// interval. This also prevents enormous mantissas or exponents from
/// requiring a proportional allocation or overwhelming the library parser.
pub(super) fn rounded_decimal(
    mantissa: &str,
    short_coefficient: Option<u64>,
    negative: bool,
    exponent: i128,
    multiplier: DecimalMultiplier,
) -> Option<Real> {
    use std::io::Write as _;

    const BOUNDARY_DIGITS: usize = 768;
    // MIL multiplies by 254, requiring up to three extra leading positions.
    const EXTRA_DIGITS: usize = 3;

    if let Some(magnitude) =
        short_coefficient.and_then(|coefficient| short_decimal(coefficient, exponent, multiplier))
    {
        return Some(if negative { -magnitude } else { magnitude });
    }

    let significant = mantissa.trim_start_matches(['0', '.']);
    if significant.is_empty() {
        return Some(if negative { -0. } else { 0. });
    }
    let source_digits = significant.len() - usize::from(significant.contains('.'));
    let max_digits = source_digits.checked_add(multiplier.extra_digits())?;
    let mut position = max_digits;
    let mut digits = [b'0'; BOUNDARY_DIGITS + EXTRA_DIGITS];
    let mut sticky = false;
    let mut carry = 0_u16;
    // Multiply from the least significant end, even for discarded digits:
    // their carry into the retained prefix is part of the exact product.
    {
        let mut emit = |digit: u16| -> Option<()> {
            position -= 1;
            if let Some(slot) = digits.get_mut(position) {
                *slot = b'0' + u8::try_from(digit).ok()?;
            } else {
                sticky |= digit != 0;
            }
            Some(())
        };
        for digit in significant.bytes().rev().filter(|&byte| byte != b'.') {
            let product = u16::from(digit - b'0') * multiplier.factor() + carry;
            emit(product % 10)?;
            carry = product / 10;
        }
        while carry != 0 {
            emit(carry % 10)?;
            carry /= 10;
        }
    }
    // position contains unused high padding, not leading significant digits.
    let scientific = exponent.saturating_add(i128::try_from(max_digits - position - 1).ok()?);
    let magnitude = if scientific > 308 {
        Real::INFINITY
    } else if scientific < -324 {
        0.
    } else {
        let prefix = std::str::from_utf8(&digits[position..max_digits.min(digits.len())]).ok()?;
        let adjusted_exponent =
            scientific - i128::try_from(prefix.len() - 1 + usize::from(sticky)).ok()?;
        let mut text = [0; BOUNDARY_DIGITS + EXTRA_DIGITS + 16];
        let available = text.len();
        let mut remaining = text.as_mut_slice();
        write!(
            remaining,
            "{prefix}{}e{adjusted_exponent}",
            if sticky { "1" } else { "" }
        )
        .ok()?;
        let used = available - remaining.len();
        std::str::from_utf8(&text[..used])
            .ok()?
            .parse::<Real>()
            .ok()?
    };
    Some(if negative { -magnitude } else { magnitude })
}

/// The only non-power-of-ten multiplier in SPICE's literal grammar.
#[derive(Clone, Copy)]
pub(super) enum DecimalMultiplier {
    One,
    Mil,
}

impl DecimalMultiplier {
    const fn factor(self) -> u16 {
        match self {
            Self::One => 1,
            Self::Mil => 254,
        }
    }

    const fn extra_digits(self) -> usize {
        match self {
            Self::One => 0,
            Self::Mil => 3,
        }
    }
}

/// Extract the binary64 exponent field without a checked narrowing that can panic.
fn biased_exponent(bits: u64) -> i32 {
    let [high, next, ..] = bits.to_be_bytes();
    i32::from((u16::from_be_bytes([high, next]) >> 4) & 0x07ff)
}

/// Exact unbiased exponent of a finite, nonzero binary64 magnitude,
/// including subnormals. Returns None for zero, infinity and NaN.
pub(crate) fn binary_exponent(value: Real) -> Option<i32> {
    if !value.is_finite() || value == 0.0 {
        return None;
    }
    let bits = value.abs().to_bits();
    let biased = biased_exponent(bits);
    if biased == 0 {
        // The count is at most 64, so its signed representation is identical.
        let leading = bits.leading_zeros().cast_signed();
        Some(63 - leading - 1074)
    } else {
        Some(biased - 1023)
    }
}

/// Split a finite nonzero binary64 into a signed significand in [1, 2)
/// in magnitude and an unbiased exponent, including subnormals.
pub(crate) fn split_power(value: Real) -> (Real, i32) {
    let bits = value.to_bits();
    let exponent = biased_exponent(bits);
    if exponent == 0 {
        let (mantissa, exponent) = split_power(value * 4_503_599_627_370_496.0);
        return (mantissa, exponent - 52);
    }
    (
        Real::from_bits((bits & 0x800f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000),
        exponent - 1023,
    )
}

/// Multiply by a power of two without first forming an unrepresentable
/// power. The final multiply is the only one that can round into subnormals
/// for the normalized significands used by the scaled division.
pub(crate) fn scale_power(mut value: Real, mut exponent: i32) -> Real {
    while exponent > 1023 {
        value *= Real::from_bits(0x7fe0_0000_0000_0000);
        exponent -= 1023;
    }
    while exponent < -1022 {
        // Defer the smallest normal factor to the final multiplication.
        if exponent >= -2044 {
            return scale_power(value, exponent + 1022) * Real::MIN_POSITIVE;
        }
        value *= Real::MIN_POSITIVE;
        exponent += 1022;
    }
    value * Real::from_bits(u64::from((exponent + 1023).cast_unsigned()) << 52)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_exponent_covers_every_finite_exponent_and_subnormal_bit() {
        for biased in 1_u64..=2046 {
            let value = f64::from_bits(biased << 52);
            let expected = i32::try_from(biased).unwrap() - 1023;
            assert_eq!(binary_exponent(value), Some(expected));
            assert_eq!(binary_exponent(-value), Some(expected));
        }
        for bit in 0_u32..52 {
            let value = f64::from_bits(1_u64 << bit);
            let expected = i32::try_from(bit).unwrap() - 1074;
            assert_eq!(binary_exponent(value), Some(expected));
            assert_eq!(binary_exponent(-value), Some(expected));
        }
        for value in [0., -0., f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(binary_exponent(value), None);
        }
    }

    #[test]
    fn c_integer_accepts_the_full_signed_c_int_range() {
        assert_eq!(c_integer(f64::from(i32::MIN)), Some(i64::from(i32::MIN)));
        assert_eq!(c_integer(f64::from(i32::MAX)), Some(i64::from(i32::MAX)));
        assert_eq!(c_integer(f64::from(i32::MAX) + 1.0), None);
        assert_eq!(c_integer(f64::from(i32::MIN) - 1.0), None);
        assert_eq!(c_integer(f64::NAN), None);
        assert_eq!(c_integer(f64::INFINITY), None);
        assert_eq!(c_integer(f64::NEG_INFINITY), None);
        assert_eq!(c_integer(1.5), Some(2));
        assert_eq!(c_integer(-1.5), Some(-1));
    }

    #[test]
    fn exact_nonnegative_integer_rejects_invalid_values_and_target_overflow() {
        let as_usize = exact_nonnegative_integer::<usize>;
        let usize_bound = f64::from(usize::BITS).exp2();
        assert_eq!(as_usize(0.0), Some(0));
        assert_eq!(as_usize(-0.0), Some(0));
        assert_eq!(as_usize(31.0), Some(31));
        assert_eq!(as_usize(31.5), None);
        assert_eq!(as_usize(-1.0), None);
        assert_eq!(as_usize(f64::NAN), None);
        assert_eq!(as_usize(f64::INFINITY), None);
        assert_eq!(as_usize(f64::NEG_INFINITY), None);
        assert_eq!(as_usize(usize_bound), None);
        assert_eq!(as_usize(usize_bound * 2.0), None);

        // Validate the actual float just below the usize bound, not an
        // unchecked float-to-usize cast used as a test oracle.
        if usize::BITS >= 53 {
            let prior = f64::from_bits(usize_bound.to_bits() - 1);
            let spacing = 1_usize << usize::BITS.saturating_sub(53);
            assert_eq!(as_usize(prior), Some(usize::MAX - (spacing - 1)));
        }

        // A u32-valued SPICE field must accept all of u32, regardless of
        // whether the target's usize can represent that value.
        let as_u32 = exact_nonnegative_integer::<u32>;
        assert_eq!(as_u32(f64::from(u32::MAX)), Some(u32::MAX));
        assert_eq!(as_u32(f64::from(u32::MAX) + 1.0), None);
        assert_eq!(exact_nonnegative_integer::<u16>(65_535.0), Some(u16::MAX));
        assert_eq!(exact_nonnegative_integer::<u16>(65_536.0), None);

        // At 2^128 the conversion would saturate at u128::MAX without
        // the independent floating-point range check.
        let u128_bound = f64::from(128).exp2();
        assert_eq!(exact_nonnegative_integer::<u128>(u128_bound), None);
        let prior = f64::from_bits(u128_bound.to_bits() - 1);
        let expected = u128::MAX - ((1_u128 << (128 - 53)) - 1);
        assert_eq!(exact_nonnegative_integer::<u128>(prior), Some(expected));
    }

    #[test]
    fn signed_conversion_is_asymmetric_and_never_saturates() {
        let bound = 2_f64.powi(127);
        assert_eq!(exact_signed_integer::<i128>(-bound), Some(i128::MIN));
        assert_eq!(exact_signed_integer::<i128>(bound), None);
        assert_eq!(
            exact_signed_integer::<i128>(f64::from_bits((-bound).to_bits() + 1)),
            None
        );
        assert_eq!(
            exact_signed_integer::<i128>(f64::from_bits(bound.to_bits() - 1)),
            Some(i128::MAX - ((1_i128 << 74) - 1))
        );
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.5, -0.5] {
            assert_eq!(exact_signed_integer::<i128>(value), None);
        }
        assert_eq!(exact_signed_integer::<i32>(-0.), Some(0));
        assert_eq!(
            exact_signed_integer::<i32>(f64::from(i32::MIN)),
            Some(i32::MIN)
        );
        assert_eq!(
            exact_signed_integer::<i32>(f64::from(i32::MAX)),
            Some(i32::MAX)
        );
        assert_eq!(exact_signed_integer::<i32>(f64::from(i32::MAX) + 1.), None);
        assert_eq!(exact_signed_integer::<i32>(f64::from(i32::MIN) - 1.), None);
        assert_eq!(truncating_signed_integer::<i32>(-1.6), Some(-1));
        assert_eq!(c_integer(-1.6), Some(-2));
        assert_eq!(c_integer(-0.5), Some(0));
        assert_eq!(
            truncating_signed_integer::<i32>(f64::from(i32::MIN) - 0.75),
            Some(i32::MIN)
        );
        assert_eq!(
            truncating_signed_integer::<i32>(f64::from(i32::MAX) + 0.75),
            Some(i32::MAX)
        );
    }

    #[test]
    fn quantization_preserves_rounding_and_saturation() {
        for (value, expected) in [
            (1.5, 2),
            (-1.5, -2),
            (0.49, 0),
            (-0.49, 0),
            (f64::MAX, 512),
            (-f64::MAX, -512),
        ] {
            assert_eq!(bounded_rounded_exponent(value, -512..=512), Some(expected));
        }
        assert_eq!(bounded_rounded_exponent(f64::NAN, -512..=512), None);
        assert_eq!(
            bounded_rounded_exponent(0., i32::MAX..=i32::MAX),
            Some(i32::MAX)
        );
    }

    #[test]
    fn significand_split_round_trips_all_exponents_and_subnormal_bits() {
        for biased in 1_u64..=2046 {
            for fraction in [0, 1, 0x000f_ffff_ffff_ffff] {
                for sign in [0, 1_u64 << 63] {
                    let value = f64::from_bits(sign | (biased << 52) | fraction);
                    let (significand, exponent) = split_power(value);
                    assert!((1.0..2.0).contains(&significand.abs()));
                    assert_eq!(binary_exponent(value), Some(exponent));
                    assert_eq!(
                        scale_power(significand, exponent).to_bits(),
                        value.to_bits()
                    );
                }
            }
        }
        for bit in 0..52 {
            for fraction in [1_u64 << bit, (2_u64 << bit) - 1] {
                for sign in [0, 1_u64 << 63] {
                    let value = f64::from_bits(sign | fraction);
                    let (significand, exponent) = split_power(value);
                    assert!((1.0..2.0).contains(&significand.abs()));
                    assert_eq!(binary_exponent(value), Some(exponent));
                    assert_eq!(
                        scale_power(significand, exponent).to_bits(),
                        value.to_bits()
                    );
                }
            }
        }
    }
}
