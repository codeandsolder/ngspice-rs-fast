//! Harmonic lattice and nonlinear frequency-domain solver regressions.

use ngspice_rs::analysis::harmonic_balance::{HarmonicLattice, LatticeLimits, ToneAxis};

fn lattice(axes: &[ToneAxis]) -> HarmonicLattice {
    HarmonicLattice::new(axes, 3, LatticeLimits::default()).unwrap()
}

#[test]
fn multidimensional_projection_preserves_every_retained_coordinate() {
    let lattice = lattice(&[
        ToneAxis {
            frequency: 1e9,
            order: 2,
        },
        ToneAxis {
            frequency: 1.013e9,
            order: 2,
        },
    ]);
    assert_eq!(lattice.coordinates(), 25);
    assert_eq!(lattice.harmonics().len(), 12);
    assert!(lattice.locate(13e6).is_some());
    assert!(lattice.locate(2. * 1e9 - 1.013e9).is_some());
    let values: Vec<_> = (0..25).map(|i| (f64::from(i) * 0.17).cos()).collect();
    let reconstructed = lattice
        .project(&lattice.synthesize(&values).unwrap())
        .unwrap();
    for (actual, expected) in reconstructed.iter().zip(values) {
        assert!((actual - expected).abs() < 2e-14);
    }
}

#[test]
fn two_tone_product_projects_sum_and_difference_without_common_period() {
    let lattice = lattice(&[
        ToneAxis {
            frequency: 1e9,
            order: 2,
        },
        ToneAxis {
            frequency: 1.000_003e9,
            order: 2,
        },
    ]);
    let mut a = vec![0.; lattice.coordinates()];
    let mut b = a.clone();
    a[1 + 2 * lattice.locate(1e9).unwrap()] = 2.;
    b[1 + 2 * lattice.locate(1.000_003e9).unwrap()] = 3.;
    let product: Vec<_> = lattice
        .synthesize(&a)
        .unwrap()
        .into_iter()
        .zip(lattice.synthesize(&b).unwrap())
        .map(|(a, b)| a * b)
        .collect();
    let coefficients = lattice.project(&product).unwrap();
    for (index, harmonic) in lattice.harmonics().iter().enumerate() {
        let expected = if harmonic.indices() == [1, 1] || harmonic.indices() == [-1, 1] {
            3.
        } else {
            0.
        };
        assert!((coefficients[1 + 2 * index] - expected).abs() < 1e-12);
        assert!(coefficients[2 + 2 * index].abs() < 1e-12);
    }
    assert!(coefficients[0].abs() < 1e-12);
}

#[test]
fn real_derivative_and_complex_amplitude_sign_convention() {
    let lattice = lattice(&[ToneAxis {
        frequency: 2.,
        order: 2,
    }]);
    let coefficients = [5., 2., 3., 0., 0.];
    let derivative = lattice.derivative(&coefficients).unwrap();
    let omega = 4. * std::f64::consts::PI;
    assert!((derivative[1] - 3. * omega).abs() < 1e-13);
    assert!((derivative[2] + 2. * omega).abs() < 1e-13);
    let amplitude = lattice.phasors(&coefficients).unwrap();
    assert!((amplitude[1].re - 2.).abs() < 1e-15);
    assert!((amplitude[1].im + 3.).abs() < 1e-15);
}

#[test]
fn related_tones_share_one_axis_and_duplicate_lattices_are_rejected() {
    let related = [
        ToneAxis {
            frequency: 1e6,
            order: 3,
        },
        ToneAxis {
            frequency: 2e6,
            order: 2,
        },
    ];
    assert!(HarmonicLattice::new(&related, 3, LatticeLimits::default()).is_err());
    let single = lattice(&[ToneAxis {
        frequency: 1e6,
        order: 7,
    }]);
    assert_eq!(single.locate(2e6), Some(1));
    assert_eq!(single.harmonics().len(), 7);
    let limits = LatticeLimits {
        basis_entries: 100,
        ..LatticeLimits::default()
    };
    assert!(
        HarmonicLattice::new(
            &[ToneAxis {
                frequency: 1e6,
                order: 7
            }],
            3,
            limits
        )
        .is_err()
    );
}

use ngspice_rs::analysis::harmonic_balance::{BalanceLimits, HarmonicModel, Sample, solve};
use ngspice_rs::maths::{Pattern, Vector};
use ngspice_rs::primitives::SpiceResult;
use std::sync::Arc;

struct DrivenNode {
    pattern: Arc<Pattern>,
    forcing: Vec<f64>,
    conductance: f64,
    capacitance: f64,
    cubic_current: f64,
    cubic_charge: f64,
}

impl HarmonicModel for DrivenNode {
    fn pattern(&self) -> &Arc<Pattern> {
        &self.pattern
    }

    fn evaluate(&mut self, phase: usize, state: &Vector, sample: &mut Sample) -> SpiceResult<()> {
        let x = state.as_slice()[0];
        let x2 = x * x;
        sample.current.set(
            0,
            (self.cubic_current * x2 + self.conductance).mul_add(x, -self.forcing[phase]),
        )?;
        sample
            .charge
            .set(0, (self.cubic_charge * x2 + self.capacitance) * x)?;
        sample.conductance.add(
            0,
            0,
            (3. * self.cubic_current).mul_add(x2, self.conductance),
        )?;
        sample
            .capacitance
            .add(0, 0, (3. * self.cubic_charge).mul_add(x2, self.capacitance))
    }
}

fn node(lattice: &HarmonicLattice, force: &[f64]) -> DrivenNode {
    DrivenNode {
        pattern: Arc::new(Pattern::new(1, [(0, 0)]).unwrap()),
        forcing: lattice.synthesize(force).unwrap(),
        conductance: 0.001,
        capacitance: 1e-9,
        cubic_current: 0.,
        cubic_charge: 0.,
    }
}

#[test]
fn linear_rc_matches_analytic_frequency_response() {
    let lattice = lattice(&[ToneAxis {
        frequency: 1e5,
        order: 3,
    }]);
    let mut force = vec![0.; lattice.coordinates()];
    force[0] = 0.001;
    force[1] = 0.002;
    let mut model = node(&lattice, &force);
    let result = solve(
        &lattice,
        &mut model,
        Vector::zeros(lattice.coordinates()),
        &[1e-12],
        BalanceLimits::default(),
    )
    .unwrap();
    let response = lattice.phasors(result.coefficients.as_slice()).unwrap();
    let expected = ngspice_rs::primitives::Complex::real(0.002)
        / ngspice_rs::primitives::Complex::new(
            model.conductance,
            std::f64::consts::TAU * 1e5 * model.capacitance,
        );
    assert!((response[0].re - 1.).abs() < 1e-9);
    assert!((response[1] - expected).magnitude() < 1e-9);
    assert!(response[2..].iter().all(|z| z.magnitude() < 1e-9));
    assert_eq!(result.stats.corrections, 1);
}

#[test]
fn analytic_nonlinear_current_and_charge_recover_manufactured_two_tone_solution() {
    let lattice = lattice(&[
        ToneAxis {
            frequency: 1e6,
            order: 2,
        },
        ToneAxis {
            frequency: 1.013e6,
            order: 2,
        },
    ]);
    let mut expected = vec![0.; lattice.coordinates()];
    expected[0] = 0.2;
    expected[1 + 2 * lattice.locate(1e6).unwrap()] = 0.4;
    expected[2 + 2 * lattice.locate(1.013e6).unwrap()] = -0.3;
    expected[1 + 2 * lattice.locate(2. * 1e6 - 1.013e6).unwrap()] = 0.05;
    let states = lattice.synthesize(&expected).unwrap();
    let mut model = node(&lattice, &vec![0.; lattice.coordinates()]);
    model.cubic_current = 0.003;
    model.cubic_charge = 4e-10;
    let current: Vec<_> = states
        .iter()
        .map(|x| (model.cubic_current * x * x + model.conductance) * x)
        .collect();
    let charge: Vec<_> = states
        .iter()
        .map(|x| (model.cubic_charge * x * x + model.capacitance) * x)
        .collect();
    let mut forcing = lattice.project(&current).unwrap();
    let dynamic = lattice
        .derivative(&lattice.project(&charge).unwrap())
        .unwrap();
    for (f, q) in forcing.iter_mut().zip(dynamic) {
        *f += q;
    }
    model.forcing = lattice.synthesize(&forcing).unwrap();
    let result = solve(
        &lattice,
        &mut model,
        Vector::zeros(lattice.coordinates()),
        &[1e-12],
        BalanceLimits::default(),
    )
    .unwrap();
    for (&actual, &target) in result.coefficients.as_slice().iter().zip(&expected) {
        assert!((actual - target).abs() < 1e-8, "{actual} vs {target}");
    }
    assert!(result.stats.corrections > 1);
    assert!(result.residual.max_abs() <= 1e-12);
}

#[test]
fn nonlinear_solver_reports_budget_and_nonconvergence() {
    let lattice = lattice(&[ToneAxis {
        frequency: 1e5,
        order: 3,
    }]);
    let mut force = vec![0.; lattice.coordinates()];
    force[1] = 1.;
    let mut model = node(&lattice, &force);
    model.cubic_current = 1.;
    for limits in [
        BalanceLimits {
            iterations: 1,
            ..BalanceLimits::default()
        },
        BalanceLimits {
            model_evaluations: 1,
            ..BalanceLimits::default()
        },
        BalanceLimits {
            jacobian_entries: 1,
            ..BalanceLimits::default()
        },
    ] {
        assert!(
            solve(
                &lattice,
                &mut model,
                Vector::zeros(lattice.coordinates()),
                &[1e-12],
                limits
            )
            .is_err()
        );
    }
}
