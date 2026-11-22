//! Exact finite binary64 product sums, with bounded exponent storage.
use super::numeric::{scale_power, split_power};
use super::{ExactSum, Real};

/// A number represented outside the native binary64 exponent range.
#[derive(Clone, Copy)]
pub(super) struct ScaledNumber {
    pub(super) mantissa: Real,
    pub(super) exponent: i32,
}

impl ScaledNumber {
    pub(super) fn divide(self, rhs: Self) -> Real {
        scale_power(self.mantissa / rhs.mantissa, self.exponent - rhs.exponent)
    }
}

/// The exact integer significand (including its sign) and unit exponent.
#[derive(Clone, Copy)]
struct ExactTerm {
    significand: i128,
    exponent: i32,
}

impl ExactTerm {
    fn from_real(value: Real) -> Self {
        let bits = value.to_bits();
        let biased = i32::try_from((bits >> 52) & 0x7ff).unwrap_or_default();
        let fraction = bits & 0x000f_ffff_ffff_ffff;
        let magnitude = i128::from(if biased == 0 {
            fraction
        } else {
            fraction | (1 << 52)
        });
        Self {
            significand: if value.is_sign_negative() {
                -magnitude
            } else {
                magnitude
            },
            exponent: biased.max(1) - 1075,
        }
    }

    const fn product(self, other: Self) -> Self {
        Self {
            significand: self.significand * other.significand,
            exponent: self.exponent + other.exponent,
        }
    }
}

/// Exact sum of up to four products, before normalization and rounding.
/// With exponents within 19 bits, four 106-bit significands fit in a
/// signed i128. Use the full finite-binary64 superaccumulator only when
/// the products span a wider range.
#[cold]
pub(super) fn exact_product_sum<const N: usize>(pairs: [(Real, Real); N]) -> ScaledNumber {
    let terms = pairs.map(|(a, b)| ExactTerm::from_real(a).product(ExactTerm::from_real(b)));
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
    if N <= 4 && max - min <= 19 {
        let sum = terms
            .iter()
            .filter(|term| term.significand != 0)
            .map(|term| term.significand << (term.exponent - min))
            .sum::<i128>();
        return scaled_integer(sum, min);
    }
    let mut sum = ExactSum::new();
    for term in terms {
        sum.add(term);
    }
    sum.magnitude().scaled()
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

// Product unit -2148, maximum product exponent <2048, panel weights
// fit in u8, and at most 4 * usize::MAX terms. 68 limbs cover the entire
// range and sign, including all intermediate partial sums.
pub(super) const LIMBS: usize = 68;
const MIN_EXP: i32 = -2148;

fn halves(value: u128) -> (u64, u64) {
    let bytes = value.to_le_bytes();
    (
        u64::from_le_bytes(*bytes.first_chunk::<8>().unwrap_or(&[0; 8])),
        u64::from_le_bytes(*bytes.last_chunk::<8>().unwrap_or(&[0; 8])),
    )
}

impl ExactSum {
    pub(crate) const fn new() -> Self {
        Self {
            bins: [0; LIMBS],
            terms: 0,
        }
    }

    fn normalize(&mut self) {
        for i in 0..LIMBS - 1 {
            let carry = self.bins[i] >> 64;
            self.bins[i] = i128::from(halves(self.bins[i].cast_unsigned()).0);
            self.bins[i + 1] += carry;
        }
    }

    fn add(&mut self, term: ExactTerm) {
        self.add_unsigned(
            term.significand.unsigned_abs(),
            term.exponent,
            term.significand < 0,
        );
    }

    fn add_unsigned(&mut self, magnitude: u128, exponent: i32, negative: bool) {
        if magnitude == 0 {
            return;
        }
        let shift = usize::try_from(exponent - MIN_EXP).unwrap_or_default();
        let index = shift / 64;
        let offset = shift % 64;
        let (lo, hi) = halves(magnitude);
        let sign = if negative { -1 } else { 1 };
        self.bins[index] += i128::from(lo << offset) * sign;
        if offset == 0 {
            self.bins[index + 1] += i128::from(hi) * sign;
        } else {
            self.bins[index + 1] += i128::from((lo >> (64 - offset)) | (hi << offset)) * sign;
            self.bins[index + 2] += i128::from(hi >> (64 - offset)) * sign;
        }
        // Each term contributes at most one u64 per bin. Normalize before
        // more than 2^32 terms accumulate, far inside signed i128 capacity.
        self.terms = self.terms.wrapping_add(1);
        if self.terms == 0 {
            self.normalize();
        }
    }

    /// Add the weighted sum of samples times a common finite span. Group
    /// only when the aligned numerator times a 53-bit span fits u128;
    /// otherwise insert each exact weighted product in the same bins.
    pub(crate) fn add_weighted<const N: usize>(
        &mut self,
        span: Real,
        samples: [Real; N],
        weights: [u8; N],
    ) {
        let span = ExactTerm::from_real(span);
        let mut terms = samples.map(ExactTerm::from_real);
        let mut min = i32::MAX;
        let mut max = i32::MIN;
        let mut weight_sum = 0_u8;
        for (term, weight) in terms.iter_mut().zip(weights) {
            term.significand *= i128::from(weight);
            if term.significand != 0 {
                min = min.min(term.exponent);
                max = max.max(term.exponent);
                weight_sum = weight_sum.saturating_add(weight);
            }
        }
        if min > max || span.significand == 0 {
            return;
        }
        if max - min <= 17 && weight_sum <= 24 {
            let sum: i128 = terms
                .iter()
                .filter(|term| term.significand != 0)
                .map(|term| term.significand << (term.exponent - min))
                .sum();
            self.add_unsigned(
                sum.unsigned_abs() * span.significand.unsigned_abs(),
                min + span.exponent,
                (sum < 0) != (span.significand < 0),
            );
        } else {
            for term in terms {
                self.add(term.product(span));
            }
        }
    }

    fn magnitude(mut self) -> Magnitude {
        self.normalize();
        let negative = self.bins[LIMBS - 1] < 0;
        let mut words = self.bins.map(|bin| halves(bin.cast_unsigned()).0);
        if negative {
            let mut carry = true;
            for word in &mut words {
                (*word, carry) = (!*word).overflowing_add(u64::from(carry));
            }
        }
        Magnitude { words, negative }
    }

    /// Final integral or normalized mean, rounded once after the exact
    /// weighted sum is divided by 24 times a positive finite divisor.
    pub(crate) fn panel_result(self, divisor: Real) -> Real {
        if !divisor.is_finite() || divisor <= 0. {
            return Real::NAN;
        }
        let divisor = ExactTerm::from_real(divisor);
        let denominator = 24 * u64::try_from(divisor.significand).unwrap_or_default();
        self.magnitude()
            .rounded(denominator, MIN_EXP - divisor.exponent)
    }
}

struct Magnitude {
    words: [u64; LIMBS],
    negative: bool,
}

impl Magnitude {
    fn shifted_word(&self, shift: usize) -> u64 {
        let offset = shift % 64;
        let low = self.words.get(shift / 64).copied().unwrap_or_default() >> offset;
        if offset == 0 {
            low
        } else {
            low | (self.words.get(shift / 64 + 1).copied().unwrap_or_default() << (64 - offset))
        }
    }

    fn sticky_below(&self, shift: usize) -> bool {
        let full = (shift / 64).min(LIMBS);
        let offset = shift % 64;
        self.words[..full].iter().any(|&word| word != 0)
            || (offset != 0
                && self
                    .words
                    .get(full)
                    .is_some_and(|word| word & ((1_u64 << offset) - 1) != 0))
    }

    fn highest_bit(&self) -> Option<usize> {
        self.words.iter().rposition(|&word| word != 0).map(|index| {
            index * 64 + usize::try_from(self.words[index].ilog2()).unwrap_or_default()
        })
    }

    fn rounded(&self, denominator: u64, scale: i32) -> Real {
        let Some(high) = self.highest_bit() else {
            return 0.;
        };
        let high = i32::try_from(high).unwrap_or_default();
        let delta = high - i32::try_from(denominator.ilog2()).unwrap_or_default();
        let quotient_exp = if delta >= 0
            && self.shifted_word(usize::try_from(delta).unwrap_or_default()) < denominator
        {
            delta - 1
        } else {
            delta
        };
        let shift =
            usize::try_from((quotient_exp - 52).max(-1074 - scale).max(0)).unwrap_or_default();
        // At most 53 quotient bits and 58 denominator bits are retained.
        let retained = u128::from(self.shifted_word(shift))
            | (u128::from(self.shifted_word(shift + 64)) << 64);
        let denom = u128::from(denominator);
        let mut mantissa = u64::try_from(retained / denom).unwrap_or_default();
        let remainder = retained % denom;
        let halfway = denom / 2;
        // The denominator is even, so the discarded fractional numerator
        // can influence nearest-even rounding only at an exact midpoint.
        if remainder > halfway
            || (remainder == halfway && (mantissa & 1 != 0 || self.sticky_below(shift)))
        {
            mantissa += 1;
        }
        let sign = u64::from(self.negative) << 63;
        if mantissa == 0 {
            return Real::from_bits(sign);
        }
        let mut exponent = scale + i32::try_from(shift).unwrap_or_default();
        if mantissa == 1_u64 << 53 {
            mantissa >>= 1;
            exponent += 1;
        }
        let top = i32::try_from(mantissa.ilog2()).unwrap_or_default();
        let binade = exponent + top;
        let bits = if binade > 1023 {
            0x7ff_u64 << 52
        } else if binade < -1022 {
            mantissa << (exponent + 1074)
        } else {
            (u64::try_from(binade + 1023).unwrap_or_default() << 52)
                | ((mantissa << (52 - top)) & ((1_u64 << 52) - 1))
        };
        Real::from_bits(sign | bits)
    }

    fn scaled(&self) -> ScaledNumber {
        let Some(high_bit) = self.highest_bit() else {
            return ScaledNumber {
                mantissa: 0.,
                exponent: 0,
            };
        };
        let exponent = i32::try_from(high_bit).unwrap_or_default() + MIN_EXP;
        let shift = high_bit.saturating_sub(52);
        let mut significand = self.shifted_word(shift);
        if high_bit < 52 {
            significand <<= 52 - high_bit;
        } else if shift != 0 {
            let halfway = shift - 1;
            let halfway_set = self.shifted_word(halfway) & 1 != 0;
            significand +=
                u64::from(halfway_set && (self.sticky_below(halfway) || significand & 1 != 0));
        }
        // Adding rather than masking also handles a significand rounded to 2.
        let mantissa = Real::from_bits((1022_u64 << 52) + significand);
        ScaledNumber {
            mantissa: if self.negative { -mantissa } else { mantissa },
            exponent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ExactSum;
    use crate::primitives::Real;

    #[test]
    fn normalized_constants_cover_binary64_exponents_and_span_scales() {
        for exponent in -1074..=1023 {
            let bits = if exponent < -1022 {
                1_u64 << (exponent + 1074)
            } else {
                u64::try_from(exponent + 1023).unwrap() << 52
            };
            let ordinate = Real::from_bits(bits);
            for span in [Real::from_bits(1), 1e-100, 1., 1e100, Real::MAX] {
                for sign in [-1., 1.] {
                    let value = sign * ordinate;
                    let mut sum = ExactSum::new();
                    sum.add_weighted(span, [value; 4], [3, 9, 9, 3]);
                    assert_eq!(sum.panel_result(span).to_bits(), value.to_bits());
                }
            }
        }
    }

    #[test]
    fn weighted_products_round_exact_midpoints_and_sticky_residuals() {
        let tiny = Real::from_bits(1);
        for (value, expected) in [(tiny, 0.), (Real::from_bits(3), Real::from_bits(2))] {
            let mut sum = ExactSum::new();
            sum.add_weighted(1., [value, 0.], [12, 12]);
            assert_eq!(sum.panel_result(1.).to_bits(), expected.to_bits());
        }
        let half_ulp = Real::EPSILON / 2.;
        for (tail, expected) in [(0., 1.), (tiny, 1_f64.next_up()), (-tiny, 1.)] {
            let mut sum = ExactSum::new();
            sum.add_weighted(1., [1.], [24]);
            sum.add_weighted(1., [half_ulp], [24]);
            sum.add_weighted(tiny, [tail], [24]);
            assert_eq!(sum.panel_result(1.).to_bits(), expected.to_bits());
        }
        // Differing exponents require the individual-product path rather
        // than grouping. The two huge products cancel before the residual
        // (3/4 of the minimum subnormal) rounds up to one subnormal.
        let mut sum = ExactSum::new();
        sum.add_weighted(1., [Real::MAX, tiny, tiny, -Real::MAX], [3, 9, 9, 3]);
        assert_eq!(sum.panel_result(1.).to_bits(), tiny.to_bits());
    }
}
