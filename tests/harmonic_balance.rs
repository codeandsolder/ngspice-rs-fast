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

use ngspice_rs::analysis::harmonic_balance::{BalanceSolution, solve_circuit};
use ngspice_rs::analysis::{AnalysisInput, RunConfig};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, Complex};
use std::path::Path;

fn circuit(body: &str) -> (Circuit, RunConfig) {
    let deck = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("hb.cir"),
            &format!("HB circuit\n{body}\n.end\n"),
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&deck).unwrap();
    (config.circuit(&deck).unwrap(), config)
}

fn physical_run(
    circuit: &mut Circuit,
    config: &RunConfig,
    lattice: &HarmonicLattice,
) -> BalanceSolution {
    circuit.finalize().unwrap();
    let tolerances = vec![1e-11; circuit.unknown_count()];
    solve_circuit(
        circuit,
        config.context().model_context(),
        lattice,
        None,
        &tolerances,
        BalanceLimits::default(),
    )
    .unwrap()
}

fn node_phasors(
    circuit: &Circuit,
    lattice: &HarmonicLattice,
    solution: &BalanceSolution,
    node: &str,
) -> Vec<Complex> {
    let row = circuit
        .unknowns()
        .node_row(circuit.nodes().get(node).unwrap())
        .unwrap();
    let m = lattice.coordinates();
    lattice
        .phasors(&solution.coefficients.as_slice()[row * m..(row + 1) * m])
        .unwrap()
}

#[test]
fn prepared_linear_rc_and_rlc_solve_with_physical_source_and_flux() {
    for (body, inductance) in [
        ("v1 in 0 sin(0 1 100k)\nr1 in out 1k\nc1 out 0 1n", 0.),
        (
            "v1 in 0 sin(0 1 100k)\nr1 in mid 1k\nl1 mid out 1m\nc1 out 0 1n",
            1e-3,
        ),
    ] {
        let lattice = lattice(&[ToneAxis {
            frequency: 100e3,
            order: 3,
        }]);
        let (mut circuit, config) = circuit(body);
        let solution = physical_run(&mut circuit, &config, &lattice);
        let spectrum = node_phasors(&circuit, &lattice, &solution, "out");
        let omega = std::f64::consts::TAU * 100e3;
        let expected = Complex::new(0., -1.)
            / Complex::new(1. - omega * omega * inductance * 1e-9, omega * 1e-6);
        assert!((spectrum[1] - expected).magnitude() < 1e-9, "{spectrum:?}");
        assert!(spectrum[0].magnitude() < 1e-9);
        assert!(spectrum[2..].iter().all(|z| z.magnitude() < 1e-9));
        assert_eq!(solution.stats.corrections, 1);
    }
}

#[test]
fn diode_rectifier_generates_dc_and_harmonics_with_physical_charge() {
    let lattice = lattice(&[ToneAxis {
        frequency: 100e3,
        order: 7,
    }]);
    let (mut circuit, config) = circuit(
        "v1 in 0 sin(0 0.8 100k)\nr1 in out 1k\nd1 out 0 d\n.model d d(is=1e-12 cjo=20p tt=5n)",
    );
    let solution = physical_run(&mut circuit, &config, &lattice);
    let spectrum = node_phasors(&circuit, &lattice, &solution, "out");
    assert!(spectrum[0].re < -0.02, "{spectrum:?}");
    assert!(spectrum[2].magnitude() > 0.02, "{spectrum:?}");
    assert!(solution.residual.max_abs() < 1e-11);
    // Compare to the existing transient solver only as an independent
    // verification path, never as an initialization/fallback of HB.
    let plan = config
        .plan(AnalysisInput::with_arguments(
            AnalysisKind::Transient,
            ["50n", "100u", "90u"],
        ))
        .unwrap();
    let transient = plan.run(&mut circuit, &config.context()).unwrap();
    let times: Vec<_> = transient
        .column("time")
        .unwrap()
        .iter()
        .map(|z| z.re)
        .collect();
    let values: Vec<_> = transient
        .column("v(out)")
        .unwrap()
        .iter()
        .map(|z| z.re)
        .collect();
    let samples = 2000u32;
    let mut coeff = Complex::ZERO;
    let mut mean = 0.;
    let mut cursor = 0usize;
    for index in 0..samples {
        let t = 90e-6 + (f64::from(index) + 0.5) * 10e-6 / f64::from(samples);
        while times[cursor + 1] < t {
            cursor += 1;
        }
        let ratio = (t - times[cursor]) / (times[cursor + 1] - times[cursor]);
        let x = (values[cursor + 1] - values[cursor]).mul_add(ratio, values[cursor]);
        mean += x / f64::from(samples);
        let (sin, cos) = (std::f64::consts::TAU * 100e3 * t).sin_cos();
        coeff = coeff + Complex::new(cos, -sin) * Complex::real(2. * x / f64::from(samples));
    }
    assert!(
        (spectrum[0].re - mean).abs() < 0.003,
        "{} vs {mean}",
        spectrum[0].re
    );
    assert!(
        (spectrum[1] - coeff).magnitude() < 0.003,
        "{:?} vs {coeff:?}",
        spectrum[1]
    );
}

#[test]
fn nonlinear_two_tone_diode_has_mixing_products() {
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
    let (mut circuit, config) = circuit(
        "v1 in 0 sin(0.3 0.18 1meg)\nv2 drive in sin(0 0.15 1.013meg)\nr1 drive out 1k\nd1 out 0 d\n.model d d(is=1e-12 cjo=10p)",
    );
    let solution = physical_run(&mut circuit, &config, &lattice);
    let spectrum = node_phasors(&circuit, &lattice, &solution, "out");
    for frequency in [13e3, 2e6 - 1.013e6, 2. * 1.013e6 - 1e6] {
        let index = lattice.locate(frequency).unwrap() + 1;
        assert!(
            spectrum[index].magnitude() > 1e-4,
            "{frequency}: {:?}",
            spectrum[index]
        );
    }
    assert!(solution.residual.max_abs() < 1e-11);
}

#[test]
fn physical_hb_refuses_unbound_or_nonperiodic_sources() {
    for source in [
        "sin(0 1 200k)",
        "sin(0 1 100k 0 1k)",
        "pulse(0 1 0 1n 1n 1u 2u)",
    ] {
        let lattice = lattice(&[ToneAxis {
            frequency: 100e3,
            order: 1,
        }]);
        let (mut circuit, config) = circuit(&format!("v1 in 0 {source}\nr1 in 0 1k"));
        circuit.finalize().unwrap();
        let tolerances = vec![1e-11; circuit.unknown_count()];
        assert!(
            solve_circuit(
                &mut circuit,
                config.context().model_context(),
                &lattice,
                None,
                &tolerances,
                BalanceLimits::default()
            )
            .is_err()
        );
    }
}

#[test]
fn deck_hb_plan_emits_frequency_indexed_complex_mna_rows() {
    let deck = Parser::new().parse_deck(&parse_deck_text(
        Path::new("hb-deck.cir"),
        "HB deck\nv1 in 0 sin(0 1 100k)\nr1 in out 1k\nc1 out 0 1n\n.hb 100k 3 oversample=4 abstol=1e-11\n.end\n"
    )).unwrap();
    let config = RunConfig::from_netlist(&deck).unwrap();
    let plans = config.plans_for(&deck.analyses).unwrap();
    assert_eq!(plans.len(), 1);
    assert_eq!(plans[0].kind(), AnalysisKind::HarmonicBalance);
    let mut circuit = config.circuit(&deck).unwrap();
    let plot = plans[0].run(&mut circuit, &config.context()).unwrap();
    assert_eq!(plot.point_count(), 4);
    assert!((plot.value("frequency", 1).unwrap().re - 100e3).abs() < 1e-10);
    let expected = Complex::new(0., -1.) / Complex::new(1., std::f64::consts::TAU * 0.1);
    assert!((plot.value("v(out)", 1).unwrap() - expected).magnitude() < 1e-9);
}

#[test]
fn memoryless_behavioural_amplifier_compresses_and_generates_third_harmonic() {
    let lattice = lattice(&[ToneAxis {
        frequency: 100e3,
        order: 5,
    }]);
    let (mut circuit, config) =
        circuit("v1 in 0 sin(0 0.5 100k)\nb1 out 0 v=2*v(in)-v(in)^3\nr1 out 0 1k");
    let solution = physical_run(&mut circuit, &config, &lattice);
    let spectrum = node_phasors(&circuit, &lattice, &solution, "out");
    assert!((spectrum[1].im + 0.90625).abs() < 1e-9, "{spectrum:?}");
    assert!((spectrum[3].im + 0.03125).abs() < 1e-9, "{spectrum:?}");
    assert!(spectrum[0].magnitude() < 1e-9);
}

#[test]
fn bjt_and_jfet_use_signed_physical_charge_at_every_phase() {
    for body in [
        "vcc rail 0 5\nv1 in 0 sin(0.65 0.01 100k)\nr1 rail out 2k\nq1 out in 0 q\n.model q npn(is=1e-15 bf=100 cje=20p cjc=5p tf=1n)",
        "vcc rail 0 5\nv1 in 0 sin(-0.5 0.05 100k)\nr1 rail out 1k\nj1 out in 0 j\n.model j njf(vto=-2 beta=1m cgs=20p cgd=5p)",
    ] {
        let lattice = lattice(&[ToneAxis {
            frequency: 100e3,
            order: 4,
        }]);
        let (mut circuit, config) = circuit(body);
        let solution = physical_run(&mut circuit, &config, &lattice);
        let spectrum = node_phasors(&circuit, &lattice, &solution, "out");
        assert!(spectrum[0].re > 0. && spectrum[0].re < 5., "{spectrum:?}");
        assert!(spectrum[1].magnitude() > 0.01);
        assert!(spectrum[2].magnitude() > 1e-6);
        assert!(solution.residual.max_abs() < 1e-11);
    }
}

#[test]
fn bsim3_quasistatic_charge_and_inverse_mode_match_small_signal_limit() {
    for (polarity, family) in [(1., "nmos"), (-1., "pmos")] {
        for inverse in [false, true] {
            let terminals = if inverse { "0 in out 0" } else { "out in 0 0" };
            let body = format!(
                "vrail rail 0 {}\nv1 in 0 dc {} ac 1m sin({} 1m 100meg)\nr1 rail out 1k\nm1 {terminals} mm w=10u l=1u\n.model mm {family} level=8 capmod=3 vth0={} cgso=1n cgdo=1n",
                polarity * 1.8,
                polarity * 0.9,
                polarity * 0.9,
                polarity * 0.5
            );
            let lattice = lattice(&[ToneAxis {
                frequency: 100e6,
                order: 3,
            }]);
            let (mut circuit, config) = circuit(&body);
            let solution = physical_run(&mut circuit, &config, &lattice);
            let spectrum = node_phasors(&circuit, &lattice, &solution, "out");
            let plan = config
                .plan(AnalysisInput::with_arguments(
                    AnalysisKind::Ac,
                    ["lin", "1", "100meg", "100meg"],
                ))
                .unwrap();
            let plot = plan.run(&mut circuit, &config.context()).unwrap();
            let ac = plot.value("v(out)", 0).unwrap();
            // AC's real 1 mV phasor must rotate by -90 degrees for SIN.
            let expected = ac * Complex::imaginary(-1.);
            assert!(
                (spectrum[1] - expected).magnitude() < 1e-7_f64.max(expected.magnitude() * 0.005),
                "{family} inverse={inverse}: {:?} vs {expected:?}",
                spectrum[1]
            );
        }
    }
}

#[test]
fn higher_order_two_tone_solution_is_stable_and_reuses_a_converged_seed() {
    let (mut circuit, config) = circuit(
        "v1 in 0 sin(0.3 0.18 1meg)\nv2 drive in sin(0 0.15 1.013meg)\nr1 drive out 1k\nd1 out 0 d\n.model d d(is=1e-12 cjo=10p)",
    );
    let axes = [
        ToneAxis {
            frequency: 1e6,
            order: 4,
        },
        ToneAxis {
            frequency: 1.013e6,
            order: 4,
        },
    ];
    let small = lattice(&axes.map(|axis| ToneAxis { order: 3, ..axis }));
    let low = physical_run(&mut circuit, &config, &small);
    let high = lattice(&axes);
    let start = std::time::Instant::now();
    let result = physical_run(&mut circuit, &config, &high);
    let elapsed = start.elapsed();
    let high_spectrum = node_phasors(&circuit, &high, &result, "out");
    let low_spectrum = node_phasors(&circuit, &small, &low, "out");
    for frequency in [1e6, 1.013e6, 13e3, 987e3, 1.026e6] {
        let a = low_spectrum[small.locate(frequency).unwrap() + 1];
        let b = high_spectrum[high.locate(frequency).unwrap() + 1];
        assert!((a - b).magnitude() < 5e-4, "{frequency}: {a:?} {b:?}");
    }
    let tolerances = vec![1e-11; circuit.unknown_count()];
    let start = std::time::Instant::now();
    let repeated = solve_circuit(
        &mut circuit,
        config.context().model_context(),
        &high,
        Some(result.coefficients),
        &tolerances,
        BalanceLimits::default(),
    )
    .unwrap();
    eprintln!(
        "81-coordinate two-tone solve: {elapsed:?}, {:?}; warm start: {:?}, {:?}",
        result.stats,
        start.elapsed(),
        repeated.stats
    );
    assert_eq!(repeated.stats.corrections, 0);
    assert!(repeated.stats.model_evaluations < result.stats.model_evaluations);
    let override_resistance = circuit.resistor_override("r1", 1020.).unwrap();
    let changed_context = config
        .context()
        .model_context()
        .with_resistor_override(override_resistance)
        .unwrap();
    let start = std::time::Instant::now();
    let cold = solve_circuit(
        &mut circuit,
        changed_context,
        &high,
        None,
        &tolerances,
        BalanceLimits::default(),
    )
    .unwrap();
    let cold_elapsed = start.elapsed();
    let start = std::time::Instant::now();
    let warm = solve_circuit(
        &mut circuit,
        changed_context,
        &high,
        Some(repeated.coefficients),
        &tolerances,
        BalanceLimits::default(),
    )
    .unwrap();
    eprintln!(
        "resistance sweep: cold {cold_elapsed:?}, {:?}; warm {:?}, {:?}",
        cold.stats,
        start.elapsed(),
        warm.stats
    );
    assert!(warm.stats.model_evaluations < cold.stats.model_evaluations);
    for (&a, &b) in cold
        .coefficients
        .as_slice()
        .iter()
        .zip(warm.coefficients.as_slice())
    {
        assert!((a - b).abs() < 1e-7);
    }
}

#[test]
fn converged_diode_spectrum_matches_independent_qucsator_hb() {
    // QucsatorRF 1.0.7, Qucs-S 26.1.1 portable release. 65 frequencies,
    // convergence reported after 11 iterations. Qucs uses a cosine source
    // and reverses its diode terminal ordering relative to SPICE.
    let reference = [
        Complex::real(-0.0594477961247413),
        Complex::new(0.6918775968421407, -0.0083330897178689),
        Complex::new(-0.0796899202417851, 0.0024747089337585),
        Complex::new(-0.0448044643296278, 0.0019665426730398),
        Complex::new(-0.0150255416861985, 0.0007072924002563),
        Complex::new(0.0025148320465375, -0.0004129509216921),
        Complex::new(0.0074792858699303, -0.0007828033738540),
    ];
    let (mut circuit, config) = circuit(
        "v1 in 0 sin(0 0.8 100k 0 0 90)\nr1 in out 1k\nc1 out 0 20p\nd1 out 0 d\n.model d d(is=1e-12 cjo=0 tt=0 vj=0.7)\n.options temp=27 tnom=27",
    );
    let lattice = HarmonicLattice::new(
        &[ToneAxis {
            frequency: 100e3,
            order: 31,
        }],
        4,
        LatticeLimits::default(),
    )
    .unwrap();
    let result = physical_run(&mut circuit, &config, &lattice);
    let spectrum = node_phasors(&circuit, &lattice, &result, "out");
    for (k, (&actual, expected)) in spectrum.iter().zip(reference).enumerate() {
        assert!(
            (actual - expected).magnitude() < 2e-6,
            "harmonic {k}: {actual:?} vs {expected:?}"
        );
    }
}

#[test]
fn related_source_tones_on_common_axis_conserve_current_and_power() {
    let (mut circuit, config) =
        circuit("v1 in 0 sin(0 1 100k)\nv2 drive in sin(0 0.5 200k)\nr1 drive out 1k\nc1 out 0 1n");
    let lattice = lattice(&[ToneAxis {
        frequency: 100e3,
        order: 4,
    }]);
    let solution = physical_run(&mut circuit, &config, &lattice);
    let input = node_phasors(&circuit, &lattice, &solution, "drive");
    let output = node_phasors(&circuit, &lattice, &solution, "out");
    let drop: Vec<_> = input.iter().zip(&output).map(|(&x, &y)| x - y).collect();
    let mut input_power = 0.;
    let mut resistor_power = 0.;
    for (k, (dv, v)) in drop.iter().zip(&input).enumerate() {
        let current = *dv * Complex::real(1e-3);
        let load_current = output[k]
            * Complex::imaginary(
                std::f64::consts::TAU * 100e3 * f64::from(u32::try_from(k).unwrap()) * 1e-9,
            );
        assert!((current - load_current).magnitude() < 1e-12);
        let scale = if k == 0 { 1. } else { 0.5 };
        input_power += scale * (*v * current.conj()).re;
        resistor_power += scale * (*dv * current.conj()).re;
    }
    assert!((input_power - resistor_power).abs() < 1e-12);
    assert!(input_power > 0.);
}
