//! Time-domain source-noise statistics and finite-band spectral laws.
//!
//! These are analyses of the physical stochastic excitation, not the
//! small-signal device noise-generating PSD of the ordinary .noise driver.
use crate::primitives::Real;

/// Sampled white-noise linear reconstruction at fixed phase u in [0,1].
/// Two independent knot amplitudes each have variance sigma².
#[must_use = "the returned variance depends on the reconstruction phase"]
pub fn interpolated_white_variance(knot_sigma: Real, phase: Real) -> Real {
    knot_sigma * knot_sigma * ((1. - phase).mul_add(1. - phase, phase * phase))
}

/// Average white power over all phases of one sample interval.
#[must_use = "the returned power accounts for interpolation over the sample interval"]
pub fn mean_white_power(knot_sigma: Real) -> Real {
    (2. / 3.) * knot_sigma * knot_sigma
}

/// Two-sided white PSD for piecewise-linearly interpolated iid knots.
///
/// The zero-frequency limit is sigma² times spacing; fourth-order sinc
/// attenuation makes the integrated two-sided variance 2/3 sigma².
#[must_use = "the returned spectral density accounts for the interpolation frequency response"]
pub fn white_psd(knot_sigma: Real, spacing: Real, frequency: Real) -> Real {
    let x = std::f64::consts::PI * frequency * spacing;
    let sinc = if x == 0. { 1. } else { x.sin() / x };
    (knot_sigma * knot_sigma * spacing) * sinc.powi(4)
}

/// Mean occupation probability of the high RTS state.
#[must_use = "the returned probability determines the stationary high-state occupation"]
pub fn telegraph_occupation(capture_mean: Real, emission_mean: Real) -> Real {
    emission_mean / (capture_mean + emission_mean)
}

/// Stationary two-state telegraph covariance at a physical lag, with exact
/// exponential capture and emission waiting times.
#[must_use = "the returned covariance determines the stationary lag-dependent noise correlation"]
pub fn telegraph_covariance(
    amplitude: Real,
    capture_mean: Real,
    emission_mean: Real,
    lag: Real,
) -> Real {
    let p = telegraph_occupation(capture_mean, emission_mean);
    let rate = capture_mean.recip() + emission_mean.recip();
    amplitude * amplitude * p * (1. - p) * (-rate * lag.abs()).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sample_knot_white_psd_and_power_agree() {
        assert_eq!(interpolated_white_variance(2., 0.), 4.);
        assert_eq!(interpolated_white_variance(2., 0.5), 2.);
        assert_eq!(white_psd(2., 0.01, 0.), 0.04);
        assert!((mean_white_power(2.) - 8. / 3.).abs() < 1e-15);
        assert!(white_psd(2., 0.01, 50.) < white_psd(2., 0.01, 10.));
        assert!(white_psd(2., 0.01, 100.).abs() < 1e-30);
    }
    #[test]
    fn telegraph_covariance_obeys_stationary_occupation_and_decay() {
        let p = telegraph_occupation(2., 8.);
        assert!((p - 0.8).abs() < 1e-14);
        let c0 = telegraph_covariance(3., 2., 8., 0.);
        let c1 = telegraph_covariance(3., 2., 8., 1.);
        assert!((c0 - 9. * 0.8 * 0.2).abs() < 1e-14);
        assert!((c1 / c0 - (-0.625_f64).exp()).abs() < 1e-14);
    }
}
