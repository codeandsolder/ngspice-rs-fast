//! Harmonic lattice regression tests while the shared analysis module
//! integration is reserved by the independent PSS campaign worker.
//! This compiles and exercises the exact production plan source.
pub use ngspice_rs::{maths, primitives};
#[path = "../src/analysis/harmonic_balance.rs"]
pub mod harmonic_balance;

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
    assert_eq!(
        plan.tone_harmonic_ordinal(0, 1),
        plan.tone_harmonic_ordinal(2, 1)
    );
    assert_eq!(plan.tone_harmonic_ordinal(1, 2), plan.mode_ordinal(&[6]));
    assert_eq!(plan.tone_harmonic_ordinal(1, 4), None);
    assert_eq!(plan.mode_ordinal(&[0, 0]), None);
    assert_eq!(plan.mode_ordinal(&[100]), None);
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
    let ordinal = plan.mode_ordinal(wanted).unwrap();
    assert_eq!(plan.modes()[ordinal].index, wanted);
    ordinal
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
    assert_eq!(circuit.plan().modes().len(), plan.modes().len());
    assert_eq!(result.iterations, 1);
    assert_eq!(result.sampled_model_calls, plan.sample_count() * 2);
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

#[test]
fn rlc_mna_branch_unknown_follows_analytic_admittance() {
    // The inductor branch is an algebraic current unknown with a
    // frequency-dependent -L coefficient, not an extra node-voltage-only
    // harmonic. This exercises the generic saddle-point MNA structure.
    let plan = HarmonicPlan::new(
        &[Tone::Independent {
            frequency_hz: 1_000.,
            order: 2,
        }],
        HarmonicLimits::default(),
    )
    .unwrap();
    let r = 1_000.;
    let c = 1e-9;
    let l = 1e-3;
    let mut forcing = vec![Complex::ZERO; plan.modes().len() * 2];
    for tone in [1, -1] {
        forcing[mode_at(&plan, &[tone]) * 2] = Complex::real(0.0005);
    }
    let problem = HarmonicProblem::new(
        plan.clone(),
        2,
        vec![1. / r, 1., 1., 0.],
        vec![c, 0., 0., -l],
        forcing,
        SolveLimits::default(),
    )
    .unwrap();
    let solved = problem.solve(&mut Linear, None).unwrap();
    let omega = TAU * 1_000.;
    let total_admittance =
        Complex::new(1. / r, omega * c) + Complex::real(1.) / Complex::imaginary(omega * l);
    let expected_v = Complex::real(5e-4) / total_admittance;
    let expected_i = expected_v / Complex::imaginary(omega * l);
    let mode = mode_at(&plan, &[1]);
    let actual_v = solved.spectrum[mode * 2];
    let actual_i = solved.spectrum[mode * 2 + 1];
    close(actual_v.re, expected_v.re);
    close(actual_v.im, expected_v.im);
    close(actual_i.re, expected_i.re);
    close(actual_i.im, expected_i.im);
    assert!(solved.residual_ratio <= 1.);
}

struct TwoTerminalJunction;

impl NonlinearConstitutive for TwoTerminalJunction {
    fn evaluate(
        &mut self,
        state: &[Real],
        currents: &mut [Real],
        charges: &mut [Real],
        conductance: &mut [Real],
        capacitance: &mut [Real],
    ) -> SpiceResult<()> {
        let voltage = state[0] - state[1];
        let current = 1e-8 * (voltage / 0.04).exp_m1();
        let g = 1e-8 * (voltage / 0.04).exp() / 0.04;
        let q = 2e-9 * (voltage + 0.1 * voltage * voltage);
        let c = 2e-9 * (1. + 0.2 * voltage);
        currents[0] = current;
        currents[1] = -current;
        charges[0] = q;
        charges[1] = -q;
        conductance.copy_from_slice(&[g, -g, -g, g]);
        capacitance.copy_from_slice(&[c, -c, -c, c]);
        Ok(())
    }
}

#[test]
fn nonlinear_two_terminal_charge_and_current_preserve_kcl() {
    let plan = HarmonicPlan::new(
        &[Tone::Independent {
            frequency_hz: 25_000.,
            order: 3,
        }],
        HarmonicLimits::default(),
    )
    .unwrap();
    let mut forcing = vec![Complex::ZERO; plan.modes().len() * 2];
    for (mode, positive) in [(0, 2e-3), (-1, 3e-4), (1, 3e-4)] {
        forcing[mode_at(&plan, &[mode]) * 2] = Complex::real(positive);
    }
    let problem = HarmonicProblem::new(
        plan.clone(),
        2,
        vec![0.1, 0., 0., 0.2],
        vec![0.; 4],
        forcing,
        SolveLimits::default(),
    )
    .unwrap();
    let solved = problem.solve(&mut TwoTerminalJunction, None).unwrap();
    assert!(solved.residual_ratio <= 1.);
    let dc = mode_at(&plan, &[0]) * 2;
    let positive = solved.spectrum[dc];
    let negative = solved.spectrum[dc + 1];
    assert!(positive.re > negative.re);
    assert!(positive.re > 0.);
    assert!(solved.spectrum.iter().all(|v| v.is_finite()));
    for (mode, harmonic) in plan.modes().iter().enumerate() {
        let conjugate = harmonic.conjugate;
        for row in 0..2 {
            let value = solved.spectrum[mode * 2 + row];
            let opposite = solved.spectrum[conjugate * 2 + row];
            close(value.re, opposite.re);
            close(value.im, -opposite.im);
        }
    }
}

#[test]
fn singular_nonlinear_start_fails_explicitly_without_fake_dc_fallback() {
    let plan = HarmonicPlan::new(
        &[Tone::Independent {
            frequency_hz: 1000.,
            order: 1,
        }],
        HarmonicLimits::default(),
    )
    .unwrap();
    let mut forcing = vec![Complex::ZERO; plan.modes().len()];
    forcing[mode_at(&plan, &[0])] = Complex::real(1.);
    let problem =
        HarmonicProblem::new(plan, 1, vec![0.], vec![0.], forcing, SolveLimits::default()).unwrap();
    // All cubic first derivatives vanish at the zero initial guess.
    let error = problem.solve(&mut Cubic, None).unwrap_err();
    let message = error.to_string();
    assert!(
        message.contains("pivot") || message.contains("singular") || message.contains("zero"),
        "{error}"
    );
}

#[test]
#[ignore = "manual: release-profile the complete coupled Newton workload"]
fn harmonic_balance_full_solve_release_profile() {
    use std::hint::black_box;
    use std::time::Instant;
    for (dimensions, order) in [(1, 1), (1, 4), (1, 8), (2, 1), (2, 3), (2, 4)] {
        let tones: Vec<_> = (0..dimensions)
            .map(|index| Tone::Independent {
                frequency_hz: if index == 0 {
                    1_000_000.
                } else {
                    1_414_213.562_373
                },
                order,
            })
            .collect();
        let plan = HarmonicPlan::new(&tones, HarmonicLimits::default()).unwrap();
        let mut forcing = vec![Complex::ZERO; plan.modes().len()];
        for axis in 0..dimensions {
            for polarity in [-1, 1] {
                let mut index = vec![0_i32; dimensions];
                index[axis] = polarity;
                forcing[mode_at(&plan, &index)] = Complex::real(0.06);
            }
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
        let mut durations = Vec::with_capacity(5);
        let mut iterations = 0;
        let mut calls = 0;
        for _ in 0..6 {
            let mut model = Cubic;
            let start = Instant::now();
            let solved = problem.solve(&mut model, None).unwrap();
            let elapsed = start.elapsed();
            assert!(solved.residual_ratio <= 1.);
            iterations = solved.iterations;
            calls = solved.sampled_model_calls;
            black_box(&solved);
            durations.push(elapsed.as_nanos());
        }
        let warm = durations.remove(0);
        black_box(warm);
        durations.sort_unstable();
        println!(
            "HB_PROFILE axes={dimensions} order={order} modes={} samples={} iterations={iterations} calls={calls} median_full_solve_ns={}",
            plan.modes().len(),
            plan.sample_count(),
            durations[2]
        );
    }
}
