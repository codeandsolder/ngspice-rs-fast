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
#[must_use]
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
#[must_use]
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
#[must_use]
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
#[must_use]
pub fn truncating_signed_integer<T: TryFrom<i128>>(value: Real) -> Option<T> {
    exact_signed_integer(value.trunc())
}

/// Quantize a finite exponent, ties away from zero, saturating at the given
/// integer limits. Invalid exponents or reversed bounds are rejected.
#[must_use]
pub fn bounded_rounded_exponent(value: Real, bounds: RangeInclusive<i32>) -> Option<i32> {
    if !value.is_finite() || bounds.is_empty() {
        return None;
    }
    exact_signed_integer(
        value
            .round()
            .clamp(Real::from(*bounds.start()), Real::from(*bounds.end())),
    )
}

/// Extract the binary64 exponent field without a checked narrowing that can panic.
fn biased_exponent(bits: u64) -> i32 {
    let [high, next, ..] = bits.to_be_bytes();
    i32::from((u16::from_be_bytes([high, next]) >> 4) & 0x07ff)
}

/// Exact unbiased exponent of a finite, nonzero binary64 magnitude,
/// including subnormals. Returns None for zero, infinity and NaN.
pub fn binary_exponent(value: Real) -> Option<i32> {
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
pub(super) fn split_power(value: Real) -> (Real, i32) {
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
pub(super) fn scale_power(mut value: Real, mut exponent: i32) -> Real {
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
