//! Power-law integration between two finite noise samples.
//!
//! The integral of a power law is the logarithmic mean of the two
//! frequency-weighted endpoint densities times the log frequency ratio.
//! Neither an absolute power of frequency nor a dimensional prefactor is
//! needed, even when either would overflow or underflow.
use crate::primitives::Real;
use crate::primitives::numeric::{scale_power, split_power};

/// C's logarithm floor for a sample that does not have a positive logarithm.
const LOG_FLOOR: Real = 1e-38;

/// log(numerator / denominator), with accurate near-unity ratios even when
/// the absolute logarithms coincide in binary64.
fn log_ratio(numerator: Real, denominator: Real) -> Real {
    let ratio = numerator / denominator;
    if (0.5..=2.).contains(&ratio) {
        ((numerator - denominator) / denominator).ln_1p()
    } else if ratio.is_normal() {
        ratio.ln()
    } else {
        numerator.ln() - denominator.ln()
    }
}

/// A finite ascending frequency interval. Its geometry is shared by all
/// generators at this step; the first sample has zero width.
#[derive(Clone, Copy)]
pub(super) struct Interval {
    start: Real,
    end: Real,
    width: Real,
    log_width: Real,
}

impl Interval {
    pub(super) fn new(start: Real, end: Real) -> Self {
        Self {
            start,
            end,
            width: end - start,
            log_width: log_ratio(end, start),
        }
    }

    pub(super) const fn is_empty(self) -> bool {
        self.width == 0.
    }

    /// Integrate a power law through positive endpoint densities. Where
    /// there is no real positive logarithm, retain C's explicit floor
    /// convention, including the signed right rectangle for equal floors.
    pub(super) fn integrate(self, previous: Real, current: Real) -> Area {
        if self.is_empty() || previous.to_bits() == current.to_bits() {
            return Area([current, self.width, 1.]);
        }
        let (previous, current_log_sample) = if previous > 0. && current > 0. {
            (previous, current)
        } else {
            (previous.max(LOG_FLOOR), current.max(LOG_FLOOR))
        };
        if previous.to_bits() == current_log_sample.to_bits() {
            return Area([current, self.width, 1.]);
        }

        // In log-frequency coordinates the integrand is f * density.
        // Its endpoint ratio gives the exponent over the whole interval.
        // Use the weighted products when normal: this also avoids
        // cancellation of large opposite logarithms near a 1/f spectrum.
        let first = self.start * previous;
        let last = self.end * current_log_sample;
        let z = if first.is_normal() && last.is_normal() {
            log_ratio(last, first)
        } else {
            self.log_width + log_ratio(current_log_sample, previous)
        };
        let (frequency, density) = if z < 0. {
            (self.start, previous)
        } else {
            (self.end, current_log_sample)
        };
        let magnitude = z.abs();
        // Anchor at the larger endpoint. This exponential-relative factor
        // is bounded by one and has the continuous limit one at z = 0.
        let relative = if magnitude == 0. {
            1.
        } else {
            -(-magnitude).exp_m1() / magnitude
        };
        Area([frequency, density, self.log_width * relative])
    }
}

/// Keep the factors until the output or input-referred integral is requested.
/// Scaling an already rounded output would lose a recoverable input integral.
pub(super) struct Area([Real; 3]);

impl Area {
    pub(super) fn scaled(&self, gain: Real) -> Real {
        let [a, b, c] = self.0;
        if a == 0. || b == 0. || c == 0. {
            return 0_f64.copysign(a);
        }
        product([a, b, c, gain])
    }
}

/// Normally multiply directly. If any intermediate loses the normal range,
/// split the factors into exact binary significands and exponents, and round
/// the product only after the last scale (including gradual underflow).
fn product(factors: [Real; 4]) -> Real {
    let mut result = 1.;
    for factor in factors {
        result *= factor;
        if !result.is_normal() {
            return scaled_product(factors);
        }
    }
    result
}

#[cold]
fn scaled_product(factors: [Real; 4]) -> Real {
    let negative = factors.iter().filter(|x| x.is_sign_negative()).count() % 2 != 0;
    if factors.contains(&0.) {
        return if negative { -0. } else { 0. };
    }
    let mut significand = 1.;
    let mut exponent = 0;
    for factor in factors {
        let (value, power) = split_power(factor);
        significand *= value;
        exponent += power;
    }
    let (significand, shift) = split_power(significand);
    exponent += shift;
    match exponent {
        ..=-1076 => 0_f64.copysign(significand),
        -1075..=1023 => scale_power(significand, exponent),
        _ => Real::INFINITY.copysign(significand),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: Real, expected: Real, tolerance: Real) {
        assert!(
            (actual / expected - 1.).abs() <= tolerance,
            "{actual:e} != {expected:e}"
        );
    }

    #[test]
    fn analytic_power_laws_across_frequency_scales() {
        for first in [1e-200, 1e-100, 1., 1e100, 1e200] {
            let interval = Interval::new(first, 2. * first);
            for (a, b, area) in [
                (2., 2., 2.),
                (1., 4., 7. / 3.),
                (1., 0.5, 2_f64.ln()),
                (1., 0.25, 0.5),
            ] {
                close(interval.integrate(a, b).scaled(1.), first * area, 8e-14);
            }
        }
    }

    #[test]
    fn adjacent_frequencies_use_the_actual_interval() {
        let first = 1e100_f64;
        let last = first.next_up();
        assert_eq!(first.ln(), last.ln());
        let interval = Interval::new(first, last);
        assert_eq!(interval.integrate(2., 2.).scaled(1.), 2. * (last - first));
        // Over such a short interval f is locally constant; the mean of the
        // log-linearly varying density 1 -> 2 is 1/ln(2).
        close(
            interval.integrate(1., 2.).scaled(1.),
            (last - first) / 2_f64.ln(),
            4e-15,
        );
    }

    #[test]
    fn near_minus_one_and_flat_slopes_have_continuous_limits() {
        let interval = Interval::new(1., 2.);
        for end in [0.5_f64.next_down(), 0.5, 0.5_f64.next_up()] {
            close(interval.integrate(1., end).scaled(1.), 2_f64.ln(), 4e-15);
        }
        for end in [1_f64.next_down(), 1., 1_f64.next_up()] {
            close(interval.integrate(1., end).scaled(1.), 1., 4e-15);
        }
    }

    #[test]
    fn actual_positive_subfloor_samples_are_not_replaced_by_the_log_floor() {
        let interval = Interval::new(1e200, 2e200);
        close(
            interval.integrate(1e-200, 4e-200).scaled(1.),
            7. / 3.,
            4e-15,
        );
    }

    #[test]
    fn gain_is_combined_before_rounding_an_underflowed_output() {
        let interval = Interval::new(1e-200, 2e-200);
        let area = interval.integrate(1e-200, 1e-200);
        assert_eq!(area.scaled(1.), 0.);
        close(area.scaled(1e100), 1e-300, 4e-15);
        // The intermediate f * density overflows, but the short interval
        // shrinks it back into the finite range.
        let first = 1e160_f64;
        let last = first.next_up();
        let area = Interval::new(first, last).integrate(1e160, 1e160_f64.next_up());
        close(area.scaled(1.), (last - first) * 1e160, 4e-15);
    }

    #[test]
    fn zero_and_signed_samples_retain_the_explicit_reference_policy() {
        let interval = Interval::new(1., 2.);
        assert_eq!(interval.integrate(0., 0.).scaled(1.), 0.);
        assert_eq!(interval.integrate(-1., -2.).scaled(1.), -2.);
        assert_eq!(
            interval.integrate(-0., -0.).scaled(1.).to_bits(),
            (-0_f64).to_bits()
        );
        assert_eq!(
            interval.integrate(0., 1.).scaled(1.).to_bits(),
            interval.integrate(LOG_FLOOR, 1.).scaled(1.).to_bits()
        );
        assert_eq!(
            interval.integrate(1., 0.).scaled(1.).to_bits(),
            interval.integrate(1., LOG_FLOOR).scaled(1.).to_bits()
        );
    }

    #[test]
    fn scaled_products_recover_range_and_preserve_final_rounding() {
        close(product([1e300, 1e300, 1e-300, 1e-300]), 1., 4e-15);
        close(product([1e-300, 1e-300, 1e300, 1e300]), 1., 4e-15);
        assert_eq!(
            product([Real::MIN_POSITIVE, 0.5, 1., 1.]),
            Real::MIN_POSITIVE / 2.
        );
        assert_eq!(product([Real::from_bits(1), 0.5, 1., 1.]), 0.);
        assert_eq!(
            product([Real::from_bits(1), 0.75, 1., 1.]),
            Real::from_bits(1)
        );
        assert!(product([Real::MAX, 2., 1., 1.]).is_infinite());
        assert_eq!(
            product([Real::MAX, 2., -0., 1.]).to_bits(),
            (-0_f64).to_bits()
        );
    }
}
