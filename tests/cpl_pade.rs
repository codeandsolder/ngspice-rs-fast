//! Direct analytical regression for the CPL three-pole rational kernel.
//! The coefficients are independent exact Laurent expansions of known poles.

use ngspice_rs::devices::cpl::pade::CplPade;
use ngspice_rs::primitives::Complex;

fn compare(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() < tolerance * expected.abs().max(1e-12),
        "{actual:.16e} vs {expected:.16e}"
    );
}

fn real_kernel(rate: f64, leading: f64, poles: [f64; 3], residues: [f64; 3]) -> f64 {
    poles
        .iter()
        .copied()
        .zip(residues)
        .fold(leading, |sum, (pole, residue)| {
            sum + residue / (rate - pole)
        })
}

fn real_series(leading: f64, poles: [f64; 3], residues: [f64; 3]) -> [f64; 5] {
    std::array::from_fn(|order| {
        poles
            .iter()
            .copied()
            .zip(residues)
            .map(|(pole, residue)| residue * pole.powi([0, 1, 2, 3, 4][order]))
            .sum::<f64>()
            / leading
    })
}

#[test]
fn reconstructs_three_distinct_real_stable_poles_and_dc_limit() {
    let poles = [-1., -3., -8.];
    let residues = [0.5, -0.35, 0.2];
    let leading = 1.1;
    let series = real_series(leading, poles, residues);
    let dc = real_kernel(0., leading, poles, residues);
    let fitted = CplPade::fit(leading, series, dc).unwrap();
    compare(fitted.leading(), leading, 1e-12);
    compare(fitted.dc_limit(), dc, 1e-12);
    for rate in [0., 0.01, 0.1, 1., 2.5, 11., 1e3, 1e7] {
        compare(
            fitted.evaluate(rate).unwrap(),
            real_kernel(rate, leading, poles, residues),
            1e-8,
        );
    }
}

fn complex_kernel(rate: f64, leading: f64, pole: Complex, residue: Complex) -> f64 {
    leading + 2. * (residue / (Complex::real(rate) - pole)).re
}

#[test]
fn reconstructs_one_real_pole_plus_complex_conjugate_pair() {
    let pole = Complex::new(-3., 2.);
    let residue = Complex::new(0.25, 0.15);
    let real_pole: f64 = -1.;
    let real_residue = 0.35;
    let leading = 1.5;
    let mut power = Complex::real(1.);
    let series = std::array::from_fn(|order| {
        let value = (real_residue * real_pole.powi([0, 1, 2, 3, 4][order])
            + 2. * (residue * power).re)
            / leading;
        power = power * pole;
        value
    });
    let reference =
        |rate| complex_kernel(rate, leading, pole, residue) + real_residue / (rate - real_pole);
    let fitted = CplPade::fit(leading, series, reference(0.)).unwrap();
    for rate in [0., 0.01, 0.1, 1., 2., 3., 9., 1e3, 1e6] {
        compare(fitted.evaluate(rate).unwrap(), reference(rate), 2e-8);
    }
}

#[test]
fn rejects_numerically_degenerate_and_unstable_rational_kernels() {
    assert!(CplPade::fit(0., [1.; 5], 0.).is_err());
    assert!(CplPade::fit(1., [0.; 5], 2.).is_err());
    assert!(CplPade::fit(1., [f64::NAN; 5], 1.).is_err());
    assert!(CplPade::fit(1., [1.; 5], f64::INFINITY).is_err());
    let poles = [1., -2., -4.];
    let residues = [0.2, 0.3, 0.1];
    assert!(
        CplPade::fit(
            1.,
            real_series(1., poles, residues),
            real_kernel(0., 1., poles, residues)
        )
        .is_err()
    );
}

#[test]
fn evaluation_rejects_negative_and_nonfinite_laplace_rates() {
    let poles = [-1., -3., -8.];
    let residues = [0.5, -0.35, 0.2];
    let fitted = CplPade::fit(
        1.1,
        real_series(1.1, poles, residues),
        real_kernel(0., 1.1, poles, residues),
    )
    .unwrap();
    for rate in [-1., f64::NEG_INFINITY, f64::NAN, f64::INFINITY] {
        assert!(fitted.evaluate(rate).is_err());
    }
}

#[test]
fn scalar_characteristic_admittance_matches_coupled_spectral_sampler() {
    // The scalar case has a closed-form high-frequency expansion:
    // H(s)/H(infinity) = sqrt((1 + G/C/s) / (1 + R/L/s)).
    // Derive its inverse-rate coefficients by enforcing
    // H(y)^2(1+b*y) = 1+a*y, without numerical differentiation.
    let (r, l, g, c): (f64, f64, f64, f64) = (3., 1e-6, 0.01, 1e-8);
    let leading = (c / l).sqrt();
    let dc = (g / r).sqrt();
    let (a, b) = (g / c, r / l);
    let mut coefficients = [0.; 6];
    coefficients[0] = 1.;
    for order in 1..=5 {
        let middle: f64 = (1..order)
            .map(|i| coefficients[i] * coefficients[order - i])
            .sum();
        let previous_square: f64 = (0..order)
            .map(|i| coefficients[i] * coefficients[order - i - 1])
            .sum();
        let input = if order == 1 { a } else { 0. };
        coefficients[order] = (input - middle - b * previous_square) / 2.;
    }
    let fit = CplPade::fit(leading, coefficients[1..].try_into().unwrap(), dc).unwrap();
    let physical =
        ngspice_rs::devices::cpl::plan::CplPlan::prepare(1, 1., &[r], &[l], &[g], &[c]).unwrap();
    for rate in [1e7, 2e7, 5e7, 1e8] {
        let true_sample = physical
            .laplace_sample(rate)
            .unwrap()
            .modes()
            .characteristic_admittance()
            .get(0, 0)
            .unwrap();
        compare(fit.evaluate(rate).unwrap(), true_sample, 2e-5);
    }
}

#[test]
fn exact_constant_and_zero_channels_require_no_artificial_poles() {
    for value in [0., 1., -3.] {
        let fitted = CplPade::fit(value, [0.; 5], value).unwrap();
        for rate in [0., 1e-3, 1., 1e9] {
            assert_eq!(fitted.evaluate(rate).unwrap(), value);
        }
    }
}

#[test]
fn eight_sample_polynomial_matching_recovers_the_c_rational_fit() {
    let leading = 1.1;
    let poles: [f64; 3] = [-1., -2., -4.];
    let residues: [f64; 3] = [0.5, -0.3, 0.2];
    let mut coefficients = [0.; 8];
    coefficients[0] = leading;
    for order in 1..8 {
        coefficients[order] = poles
            .iter()
            .copied()
            .zip(residues)
            .map(|(pole, residue)| residue * pole.powi([0, 1, 2, 3, 4, 5, 6][order - 1]))
            .sum();
    }
    // Model setup interpolates inverse-rate samples at 0 and 7 positive
    // abscissae. The sample function here is exactly degree-seven, so a
    // successful match must preserve the first six Laurent coefficients.
    let samples = std::array::from_fn(|i| {
        let x = f64::from([0_u8, 1, 2, 3, 4, 5, 6, 7][i]) * 0.02;
        let value = coefficients
            .iter()
            .rev()
            .fold(0., |value, &coefficient| value * x + coefficient);
        (x, value)
    });
    let dc = real_kernel(0., leading, poles, residues);
    let by_samples = CplPade::fit_samples(&samples, dc).unwrap();
    let direct = CplPade::fit(
        leading,
        std::array::from_fn(|i| coefficients[i + 1] / leading),
        dc,
    )
    .unwrap();
    for rate in [0., 0.01, 0.1, 1., 10., 100.] {
        compare(
            by_samples.evaluate(rate).unwrap(),
            direct.evaluate(rate).unwrap(),
            1e-5,
        );
    }
    // Source ordering must not be an implicit requirement of the algorithm.
    let mut reversed = samples;
    reversed.reverse();
    let reverse_fit = CplPade::fit_samples(&reversed, dc).unwrap();
    compare(
        reverse_fit.evaluate(0.8).unwrap(),
        direct.evaluate(0.8).unwrap(),
        1e-5,
    );
}

#[test]
fn inverse_rate_interpolation_rejects_missing_zero_repeated_and_nan_samples() {
    let mut samples =
        std::array::from_fn(|index| (f64::from([0_u8, 1, 2, 3, 4, 5, 6, 7][index]) * 0.1, 1.));
    let original = samples;
    samples[0].0 = 1.;
    assert!(CplPade::fit_samples(&samples, 1.).is_err());
    samples = original;
    samples[0].0 = 0.1;
    assert!(CplPade::fit_samples(&samples, 1.).is_err());
    samples = original;
    samples[3].1 = f64::NAN;
    assert!(CplPade::fit_samples(&samples, 1.).is_err());
}
