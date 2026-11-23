//! Harmonic lattice regression tests while the shared analysis module
//! integration is reserved by the independent PSS campaign worker.
//! This compiles and exercises the exact production plan source.
pub use ngspice_rs::{maths, primitives};
#[path = "../src/analysis/harmonic_balance.rs"]
mod harmonic_balance;

use harmonic_balance::plan::{HarmonicLimits, HarmonicPlan, Tone};
use harmonic_balance::{HarmonicProblem, NonlinearConstitutive, SolveLimits};
use primitives::{Complex, Real, SpiceResult};
use std::f64::consts::TAU;

fn close(a: f64, b: f64) {
    let scale = a.abs().max(b.abs()).max(1.);
    assert!(
        (a - b).abs() <= scale * 1e-11,
        "{a:.17e} != {b:.17e} (difference {:.17e})",
        a - b
    );
}

#[test]
fn signed_single_tone_synthesis_and_projection_use_correct_amplitude() {
    let plan = HarmonicPlan::new(
        &[Tone::Independent {
            frequency_hz: 1250.,
            order: 3,
        }],
        HarmonicLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.axes().len(), 1);
    assert_eq!(plan.modes().len(), 7);
    assert_eq!(plan.sample_count(), 13);
    let samples: Vec<_> = (0..plan.sample_count())
        .map(|j| {
            let theta = TAU * j as f64 / plan.sample_count() as f64;
            1.7 + 0.8 * theta.cos() - 0.25 * (2. * theta).sin()
        })
        .collect();
    let spectrum = plan.project(&samples).unwrap();
    for (mode, value) in plan.modes().iter().zip(&spectrum) {
        let k = mode.index[0];
        close(
            value.re,
            match k {
                0 => 1.7,
                1 | -1 => 0.4,
                _ => 0.,
            },
        );
        close(
            value.im,
            match k {
                2 => 0.125,
                -2 => -0.125,
                _ => 0.,
            },
        );
        close(mode.frequency_hz, f64::from(k) * 1250.);
    }
    let reconstructed = plan.synthesize(&spectrum).unwrap();
    for (a, b) in reconstructed.iter().zip(samples.iter()) {
        close(*a, *b);
    }
}

#[test]
fn genuinely_multidimensional_two_tone_mixing_is_retained() {
    let plan = HarmonicPlan::new(
        &[
            Tone::Independent {
                frequency_hz: 1_000_000.,
                order: 2,
            },
            Tone::Independent {
                frequency_hz: 1_414_213.562_373,
                order: 2,
            },
        ],
        HarmonicLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.axes().len(), 2);
    assert_eq!(plan.modes().len(), 25);
    assert_eq!(plan.sample_count(), 81);
    let samples: Vec<_> = (0..plan.sample_count())
        .map(|j| {
            let p1 = TAU * (j / 9) as f64 / 9.;
            let p2 = TAU * (j % 9) as f64 / 9.;
            let v = 0.8 * p1.cos() + 0.5 * p2.cos();
            v * v
        })
        .collect();
    let coefficients = plan.project(&samples).unwrap();
    for (mode, value) in plan.modes().iter().zip(coefficients) {
        let (k, l) = (mode.index[0], mode.index[1]);
        close(
            value.re,
            match (k.abs(), l.abs()) {
                (0, 0) => 0.445,
                (2, 0) => 0.16,
                (0, 2) => 0.0625,
                (1, 1) => 0.2,
                _ => 0.,
            },
        );
        close(value.im, 0.);
    }
}

#[test]
fn locked_rational_tones_collapse_to_one_physical_frequency_axis() {
    let plan = HarmonicPlan::new(
        &[
            Tone::Independent {
                frequency_hz: 300.,
                order: 2,
            },
            Tone::Locked {
                to: 0,
                numerator: 3,
                denominator: 2,
                order: 1,
            },
            Tone::Locked {
                to: 1,
                numerator: 2,
                denominator: 3,
                order: 1,
            },
        ],
        HarmonicLimits::default(),
    )
    .unwrap();
    assert_eq!(plan.axes().len(), 1);
    close(plan.axes()[0].frequency_hz, 150.);
    assert_eq!(plan.axes()[0].order, 2 * 2 + 3 + 2);
    assert_eq!(plan.tone_weight(0), Some((0, 2)));
    assert_eq!(plan.tone_weight(1), Some((0, 3)));
    assert_eq!(plan.tone_weight(2), Some((0, 2)));
    for (actual, expected) in plan.tone_frequencies().iter().zip([300., 450., 300.]) {
        close(*actual, expected);
    }
}

#[test]
fn rejected_accidental_ratio_alias_is_not_silently_folded() {
    let error = HarmonicPlan::new(
        &[
            Tone::Independent {
                frequency_hz: 1_000_000.,
                order: 2,
            },
            Tone::Independent {
                frequency_hz: 2_000_000.,
                order: 2,
            },
        ],
        HarmonicLimits::default(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("alias harmonics"), "{error}");
}

#[test]
fn limits_fail_before_any_large_grid_is_created() {
    let tones = [Tone::Independent {
        frequency_hz: 1.,
        order: 150,
    }];
    let error = HarmonicPlan::new(
        &tones,
        HarmonicLimits {
            dimensions: 1,
            coefficients: 10,
            samples: 1000,
            dft_products: 100,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("budget"));
    let failure = HarmonicPlan::new(
        &[Tone::Locked {
            to: 0,
            numerator: 1,
            denominator: 1,
            order: 1,
        }],
        HarmonicLimits::default(),
    )
    .unwrap_err();
    assert!(failure.to_string().contains("earlier reference"));
}

#[test]
fn checked_conjugate_symmetry_rejects_inconsistent_real_waveforms() {
    let plan = HarmonicPlan::new(
        &[Tone::Independent {
            frequency_hz: 500.,
            order: 1,
        }],
        HarmonicLimits::default(),
    )
    .unwrap();
    let values = [
        primitives::Complex::new(1., 0.),
        primitives::Complex::real(0.),
        primitives::Complex::new(2., 3.),
    ];
    assert!(plan.synthesize(&values).is_err());
}

#[test]
fn conjugate_lookup_is_involutive_over_full_multitone_lattice() {
    let plan = HarmonicPlan::new(
        &[
            Tone::Independent {
                frequency_hz: 510_000.,
                order: 2,
            },
            Tone::Independent {
                frequency_hz: 700_001.,
                order: 3,
            },
        ],
        HarmonicLimits::default(),
    )
    .unwrap();
    for (position, mode) in plan.modes().iter().enumerate() {
        let negative = &plan.modes()[mode.conjugate];
        assert_eq!(negative.conjugate, position);
        assert_eq!(
            negative.index,
            mode.index.iter().map(|n| -n).collect::<Vec<_>>()
        );
        close(negative.frequency_hz, -mode.frequency_hz);
    }
}

struct Linear;
impl NonlinearConstitutive for Linear {
    fn evaluate(
        &mut self,
        _state: &[Real],
        _currents: &mut [Real],
        _charges: &mut [Real],
        _conductance: &mut [Real],
        _capacitance: &mut [Real],
    ) -> SpiceResult<()> {
        Ok(())
    }
}

fn mode_at(plan: &HarmonicPlan, wanted: &[i32]) -> usize {
    plan.modes()
        .iter()
        .position(|mode| mode.index == wanted)
        .unwrap()
}

#[test]
fn linear_rc_harmonic_matches_analytic_complex_transfer_function() {
    let plan = HarmonicPlan::new(
        &[Tone::Independent {
            frequency_hz: 1_000.,
            order: 3,
        }],
        HarmonicLimits::default(),
    )
    .unwrap();
    let mut forcing = vec![Complex::ZERO; plan.modes().len()];
    forcing[mode_at(&plan, &[1])] = Complex::real(5e-4);
    forcing[mode_at(&plan, &[-1])] = Complex::real(5e-4);
    let circuit = HarmonicProblem::new(
        plan.clone(),
        1,
        vec![1e-3],
        vec![1e-7],
        forcing,
        SolveLimits::default(),
    )
    .unwrap();
    let result = circuit.solve(&mut Linear, None).unwrap();
    assert_eq!(result.iterations, 1);
    assert!(result.residual_ratio <= 1.);
    let omega = 2. * std::f64::consts::PI * 1_000.;
    let expected = Complex::real(5e-4) / Complex::new(1e-3, omega * 1e-7);
    let actual = result.spectrum[mode_at(&plan, &[1])];
    close(actual.re, expected.re);
    close(actual.im, expected.im);
    let negative = result.spectrum[mode_at(&plan, &[-1])];
    close(negative.re, expected.re);
    close(negative.im, -expected.im);
    close(result.spectrum[mode_at(&plan, &[0])].magnitude(), 0.);
}

struct Junction {
    is: Real,
    thermal: Real,
    c: Real,
}

impl NonlinearConstitutive for Junction {
    fn evaluate(
        &mut self,
        state: &[Real],
        current: &mut [Real],
        charge: &mut [Real],
        conductance: &mut [Real],
        capacitance: &mut [Real],
    ) -> SpiceResult<()> {
        let v = state[0];
        current[0] = self.is * (v / self.thermal).exp_m1();
        conductance[0] = self.is * (v / self.thermal).exp() / self.thermal;
        charge[0] = self.c * (v + 0.3 * v * v);
        capacitance[0] = self.c * (1. + 0.6 * v);
        Ok(())
    }
}

#[test]
fn nonlinear_rectifier_converges_with_charge_and_second_harmonic() {
    let plan = HarmonicPlan::new(
        &[Tone::Independent {
            frequency_hz: 50_000.,
            order: 4,
        }],
        HarmonicLimits::default(),
    )
    .unwrap();
    let mut forcing = vec![Complex::ZERO; plan.modes().len()];
    forcing[mode_at(&plan, &[0])] = Complex::real(1e-3);
    forcing[mode_at(&plan, &[1])] = Complex::real(3e-4);
    forcing[mode_at(&plan, &[-1])] = Complex::real(3e-4);
    let circuit = HarmonicProblem::new(
        plan.clone(),
        1,
        vec![0.02],
        vec![0.],
        forcing,
        SolveLimits {
            absolute: 1e-12,
            relative: 1e-9,
            ..SolveLimits::default()
        },
    )
    .unwrap();
    let result = circuit
        .solve(
            &mut Junction {
                is: 1e-7,
                thermal: 0.026,
                c: 1e-8,
            },
            None,
        )
        .unwrap();
    assert!(result.iterations > 1, "{result:?}");
    assert!(result.residual_ratio <= 1., "{result:?}");
    let dc = result.spectrum[mode_at(&plan, &[0])].re;
    assert!(dc > 0. && dc < 0.06, "unexpected rectified DC {dc:e}");
    let second = result.spectrum[mode_at(&plan, &[2])].magnitude();
    assert!(second > 1e-7, "nonlinear second harmonic missing");
}

struct Cubic;
impl NonlinearConstitutive for Cubic {
    fn evaluate(
        &mut self,
        state: &[Real],
        current: &mut [Real],
        _charge: &mut [Real],
        conductance: &mut [Real],
        _capacitance: &mut [Real],
    ) -> SpiceResult<()> {
        let v = state[0];
        current[0] = 0.5 * v.powi(3);
        conductance[0] = 1.5 * v * v;
        Ok(())
    }
}

#[test]
fn two_incommensurate_tones_generate_third_order_mixer_products() {
    let plan = HarmonicPlan::new(
        &[
            Tone::Independent {
                frequency_hz: 1_000_000.,
                order: 3,
            },
            Tone::Independent {
                frequency_hz: 1_414_213.562_373,
                order: 3,
            },
        ],
        HarmonicLimits::default(),
    )
    .unwrap();
    let mut forcing = vec![Complex::ZERO; plan.modes().len()];
    for tone in [[1, 0], [-1, 0], [0, 1], [0, -1]] {
        let amplitude = if tone[0] != 0 { 0.08 } else { 0.06 };
        forcing[mode_at(&plan, &tone)] = Complex::real(amplitude);
    }
    let problem = HarmonicProblem::new(
        plan.clone(),
        1,
        vec![1.],
        vec![0.],
        forcing,
        SolveLimits::default(),
    )
    .unwrap();
    let solution = problem.solve(&mut Cubic, None).unwrap();
    assert!(solution.residual_ratio <= 1.);
    for index in [[2, -1], [-2, 1], [-1, 2], [1, -2]] {
        let amplitude = solution.spectrum[mode_at(&plan, &index)].magnitude();
        assert!(
            amplitude > 1e-5,
            "missing mixer product {index:?}: {amplitude:e}"
        );
    }
}

#[test]
fn invalid_forcing_and_overbudget_newton_dimensions_are_rejected() {
    let plan = HarmonicPlan::new(
        &[Tone::Independent {
            frequency_hz: 100.,
            order: 2,
        }],
        HarmonicLimits::default(),
    )
    .unwrap();
    let forcing = vec![Complex::new(0., 1.); plan.modes().len()];
    assert!(
        HarmonicProblem::new(
            plan.clone(),
            1,
            vec![1.],
            vec![0.],
            forcing,
            SolveLimits::default(),
        )
        .is_err()
    );
    assert!(
        HarmonicProblem::new(
            plan.clone(),
            1,
            vec![1.],
            vec![0.],
            vec![Complex::ZERO; plan.modes().len()],
            SolveLimits {
                dense_dofs: 4,
                ..SolveLimits::default()
            },
        )
        .is_err()
    );
}
