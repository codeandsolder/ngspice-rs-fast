//! Signed sideband and periodic RF conversion regressions.

use ngspice_rs::analysis::{
    harmonic_balance::{HarmonicLattice, LatticeLimits, ToneAxis},
    periodic_small_signal::SidebandPlan,
};

#[test]
fn signed_sidebands_preserve_integer_pump_coordinates() {
    let lattice = HarmonicLattice::new(
        &[
            ToneAxis {
                frequency: 10.,
                order: 1,
            },
            ToneAxis {
                frequency: 23.,
                order: 1,
            },
        ],
        2,
        LatticeLimits::default(),
    )
    .unwrap();
    let plan = SidebandPlan::new(&lattice, 2., 9).unwrap();
    assert_eq!(plan.bands().len(), 9);
    assert_eq!(
        plan.bands()[plan.locate(&[-1, 1]).unwrap()].frequency(),
        15.
    );
    assert_eq!(
        plan.bands()[plan.locate(&[1, -1]).unwrap()].frequency(),
        -11.
    );
    assert_eq!(plan.bands()[plan.locate(&[0, 0]).unwrap()].frequency(), 2.);
    assert!(!plan.conjugate_overlap());
    assert!(
        SidebandPlan::new(&lattice, 0., 9)
            .unwrap()
            .conjugate_overlap()
    );
    assert!(
        SidebandPlan::new(&lattice, 5., 9)
            .unwrap()
            .conjugate_overlap()
    );
    assert!(SidebandPlan::new(&lattice, 1e30, 9).is_err());
    assert!(SidebandPlan::new(&lattice, 2., 8).is_err());
}

use ngspice_rs::analysis::harmonic_balance::{
    BalanceLimits, HarmonicModel, Sample, solve, solve_circuit,
};
use ngspice_rs::analysis::periodic_small_signal::{
    Coordinate, PeriodicLinearization, PortChannel, ResponseLimits, Term,
};
use ngspice_rs::analysis::{AnalysisInput, RunConfig};
use ngspice_rs::maths::{Pattern, Vector};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, Complex, SpiceResult};
use std::{f64::consts::TAU, path::Path, sync::Arc};

fn close(actual: Complex, expected: Complex, tolerance: f64) {
    assert!(
        (actual - expected).magnitude() < tolerance,
        "{actual:?} != {expected:?}"
    );
}

struct Modulator {
    pattern: Arc<Pattern>,
    conductance: Vec<f64>,
    capacitance: Vec<f64>,
}

impl HarmonicModel for Modulator {
    fn pattern(&self) -> &Arc<Pattern> {
        &self.pattern
    }

    fn evaluate(&mut self, phase: usize, state: &Vector, sample: &mut Sample) -> SpiceResult<()> {
        let [drive, output] = [state.as_slice()[0], state.as_slice()[1]];
        let g = self.conductance[phase];
        let c = self.capacitance[phase];
        sample.current.set(0, drive)?;
        sample.current.set(1, (-g).mul_add(drive, output))?;
        sample.charge.set(1, -c * drive)?;
        sample.conductance.add(0, 0, 1.)?;
        sample.conductance.add(1, 0, -g)?;
        sample.conductance.add(1, 1, 1.)?;
        sample.capacitance.add(1, 0, -c)
    }
}

fn unit(row: usize, band: usize) -> Vec<Term> {
    vec![Term {
        at: Coordinate { row, band },
        value: Complex::real(1.),
    }]
}

#[test]
fn analytic_two_pump_conversion_uses_output_frequency_and_signed_phase() {
    let lattice = HarmonicLattice::new(
        &[
            ToneAxis {
                frequency: 10.,
                order: 1,
            },
            ToneAxis {
                frequency: 23.,
                order: 1,
            },
        ],
        3,
        LatticeLimits::default(),
    )
    .unwrap();
    let m = lattice.coordinates();
    let mut g = vec![0.; m];
    let mut c = vec![0.; m];
    g[0] = 2.;
    let first = 1 + 2 * lattice.locate(10.).unwrap();
    let second = 1 + 2 * lattice.locate(23.).unwrap();
    g[first] = 0.6;
    g[second + 1] = 0.4;
    c[first + 1] = 0.003;
    c[second] = 0.002;
    let mut model = Modulator {
        pattern: Arc::new(Pattern::new(2, [(1, 1), (0, 0), (1, 0)]).unwrap()),
        conductance: lattice.synthesize(&g).unwrap(),
        capacitance: lattice.synthesize(&c).unwrap(),
    };
    let tolerance = [1e-12; 2];
    let limits = BalanceLimits::default();
    let mut orbit = solve(
        &lattice,
        &mut model,
        Vector::zeros(2 * m),
        &tolerance,
        limits,
    )
    .unwrap();
    orbit.coefficients.set(0, 1.).unwrap();
    assert!(
        PeriodicLinearization::prepare(&lattice, &mut model, &orbit, &tolerance, limits).is_err()
    );
    orbit.coefficients.set(0, 0.).unwrap();
    let prepared =
        PeriodicLinearization::prepare(&lattice, &mut model, &orbit, &tolerance, limits).unwrap();
    for probe in [2., -2., 17.] {
        let response = prepared.factor(probe).unwrap();
        let plan = response.plan();
        let center = plan.locate(&[0, 0]).unwrap();
        let all = (0..m).map(|band| unit(1, band)).collect::<Vec<_>>();
        let values = response
            .transfer(&[unit(0, center)], &all, ResponseLimits::default())
            .unwrap();
        for (band, value) in plan.bands().iter().zip(values) {
            let (static_part, charge_part) = match band.indices() {
                [0, 0] => (Complex::real(2.), Complex::ZERO),
                [1, 0] => (Complex::real(0.3), Complex::imaginary(-0.0015)),
                [-1, 0] => (Complex::real(0.3), Complex::imaginary(0.0015)),
                [0, 1] => (Complex::imaginary(-0.2), Complex::real(0.001)),
                [0, -1] => (Complex::imaginary(0.2), Complex::real(0.001)),
                _ => (Complex::ZERO, Complex::ZERO),
            };
            close(
                value,
                static_part + Complex::imaginary(TAU * band.frequency()) * charge_part,
                2e-14,
            );
        }
        if !plan.conjugate_overlap() {
            let observed = [
                center,
                plan.locate(&[1, 0]).unwrap(),
                plan.locate(&[0, 1]).unwrap(),
            ];
            let observations = observed.map(|band| unit(1, band));
            let pairs = [
                CovariancePair { left: 0, right: 0 },
                CovariancePair { left: 0, right: 1 },
                CovariancePair { left: 1, right: 2 },
            ];
            let mut accumulator =
                NoiseAccumulator::new(&response, &observations, &pairs, NoiseLimits::default())
                    .unwrap();
            let mut expected = [Complex::ZERO; 3];
            for (incoming, source) in plan.bands().iter().enumerate() {
                accumulator.add_factor(&unit(0, incoming)).unwrap();
                let h = observed.map(|index| {
                    let target = &plan.bands()[index];
                    let delta = [
                        target.indices()[0] - source.indices()[0],
                        target.indices()[1] - source.indices()[1],
                    ];
                    let (g, c) = match delta {
                        [0, 0] => (Complex::real(2.), Complex::ZERO),
                        [1, 0] => (Complex::real(0.3), Complex::imaginary(-0.0015)),
                        [-1, 0] => (Complex::real(0.3), Complex::imaginary(0.0015)),
                        [0, 1] => (Complex::imaginary(-0.2), Complex::real(0.001)),
                        [0, -1] => (Complex::imaginary(0.2), Complex::real(0.001)),
                        _ => (Complex::ZERO, Complex::ZERO),
                    };
                    g + Complex::imaginary(TAU * target.frequency()) * c
                });
                for (sum, pair) in expected.iter_mut().zip(pairs) {
                    *sum = *sum + h[pair.left] * h[pair.right].conj();
                }
            }
            let result = accumulator.finish().unwrap();
            for (actual, reference) in result.densities.into_iter().zip(expected) {
                close(actual, reference, 1e-12);
            }
        }
        let tiny = ResponseLimits {
            tile_entries: 1,
            ..ResponseLimits::default()
        };
        assert!(response.transfer(&[unit(0, center)], &all, tiny).is_err());
    }
}

fn circuit(body: &str) -> (ngspice_rs::devices::Circuit, RunConfig) {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("periodic.cir"),
            &format!("periodic conversion\n{body}\n.end\n"),
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    (config.circuit(&netlist).unwrap(), config)
}

#[test]
fn zero_pump_native_ports_reduce_to_ordinary_sp() {
    let lattice = HarmonicLattice::new(
        &[ToneAxis {
            frequency: 1e6,
            order: 2,
        }],
        3,
        LatticeLimits::default(),
    )
    .unwrap();
    let (mut circuit, config) = circuit(
        "v1 in 0 dc 0 portnum 1 z0 50\nv2 out 0 dc 0 portnum 2 z0 75\nr1 in out 100\nc1 out 0 1n",
    );
    circuit.finalize().unwrap();
    let tolerance = vec![1e-12; circuit.unknown_count()];
    let context = config.context().model_context();
    let orbit = solve_circuit(
        &mut circuit,
        context.clone(),
        &lattice,
        None,
        &tolerance,
        BalanceLimits::default(),
    )
    .unwrap();
    let plan = config
        .plan(AnalysisInput::with_arguments(
            AnalysisKind::SParameter,
            ["lin", "1", "1.3meg", "1.3meg"],
        ))
        .unwrap();
    let ordinary = plan.run(&mut circuit, &config.context()).unwrap();
    let prepared = PeriodicLinearization::circuit(
        &lattice,
        &mut circuit,
        context,
        &orbit,
        &tolerance,
        BalanceLimits::default(),
    )
    .unwrap();
    let f = 1.3e6;
    let response = prepared.factor(f).unwrap();
    let center = response.plan().locate(&[0]).unwrap();
    let inputs = [
        PortChannel {
            port: 1,
            band: center,
        },
        PortChannel {
            port: 2,
            band: center,
        },
    ];
    let outputs = inputs;
    let scattering = response
        .scattering(&inputs, &outputs, ResponseLimits::default())
        .unwrap();
    for (column, port) in inputs.iter().enumerate() {
        for (row, observed) in outputs.iter().enumerate() {
            close(
                scattering[column * 2 + row],
                ordinary
                    .value(&format!("S_{}_{}", observed.port, port.port), 0)
                    .unwrap(),
                1e-12,
            );
        }
    }
    // A unit incident thermal wave contributes kT times the selected
    // conversion power gain, even with unequal reference impedances.
    let observation = response.scattered_wave(outputs[1]).unwrap();
    let mut thermal = NoiseAccumulator::new(
        &response,
        &[observation],
        &[CovariancePair { left: 0, right: 0 }],
        NoiseLimits::default(),
    )
    .unwrap();
    thermal.add_thermal_input(inputs[0], 290.).unwrap();
    let expected = BOLTZMANN * 290. * scattering[1].magnitude().powi(2);
    let actual = thermal.finish().unwrap().densities[0];
    close(actual, Complex::real(expected), expected * 1e-12);
    let other = PortChannel {
        port: 2,
        band: response.plan().locate(&[1]).unwrap(),
    };
    close(
        response
            .scattering(&inputs[..1], &[other], ResponseLimits::default())
            .unwrap()[0],
        Complex::ZERO,
        1e-14,
    );
}

use ngspice_rs::analysis::periodic_noise::{
    CovariancePair, NoiseAccumulator, NoiseFigure, NoiseLimits,
};
use ngspice_rs::devices::noise::{BOLTZMANN, CHARGE};

#[test]
fn coherent_noise_factors_preserve_phase_and_fold_all_input_sidebands() {
    let lattice = HarmonicLattice::new(
        &[ToneAxis {
            frequency: 10.,
            order: 1,
        }],
        3,
        LatticeLimits::default(),
    )
    .unwrap();
    let mut model = Modulator {
        pattern: Arc::new(Pattern::new(2, [(0, 0), (1, 0), (1, 1)]).unwrap()),
        conductance: lattice.synthesize(&[2., 0.6, 0.8]).unwrap(),
        capacitance: vec![0.; lattice.samples()],
    };
    let limits = BalanceLimits::default();
    let tolerance = [1e-12; 2];
    let orbit = solve(&lattice, &mut model, Vector::zeros(6), &tolerance, limits).unwrap();
    let prepared =
        PeriodicLinearization::prepare(&lattice, &mut model, &orbit, &tolerance, limits).unwrap();
    let response = prepared.factor(3.).unwrap();
    assert!(response.shares_factor(&prepared.factor(3.).unwrap()));
    assert!(!response.shares_factor(&prepared.factor(4.).unwrap()));
    let outputs = [unit(1, 1), unit(1, 2)];
    let pairs = [
        CovariancePair { left: 0, right: 0 },
        CovariancePair { left: 1, right: 1 },
        CovariancePair { left: 0, right: 1 },
        CovariancePair { left: 1, right: 0 },
    ];
    let mut noise =
        NoiseAccumulator::new(&response, &outputs, &pairs, NoiseLimits::default()).unwrap();
    // One source at each retained signed frequency. 2, (0.3-j0.4),
    // (0.3+j0.4) are the exact modulation coefficients.
    for band in 0..3 {
        noise.add_factor(&unit(0, band)).unwrap();
    }
    let result = noise.finish().unwrap();
    close(result.densities[0], Complex::real(4.5), 2e-14);
    close(result.densities[1], Complex::real(4.25), 2e-14);
    close(result.densities[2], Complex::new(1.2, 1.6), 2e-14);
    close(result.densities[3], result.densities[2].conj(), 1e-15);
    // A single factor coherently drives two inputs, not two independent
    // noise sources. This also exercises arbitrary complex cross-row terms.
    let mut correlated =
        NoiseAccumulator::new(&response, &outputs, &pairs, NoiseLimits::default()).unwrap();
    let mut factor = unit(0, 1);
    factor.push(Term {
        at: Coordinate { row: 0, band: 2 },
        value: Complex::imaginary(2.),
    });
    let amplitudes = response
        .transfer(
            &[factor.clone()],
            &outputs,
            ResponseLimits {
                max_backward_error: Some(1e-12),
                ..ResponseLimits::default()
            },
        )
        .unwrap();
    correlated.add_factor(&factor).unwrap();
    let density = correlated.finish().unwrap();
    close(
        density.densities[0],
        amplitudes[0] * amplitudes[0].conj(),
        2e-14,
    );
    close(
        density.densities[2],
        amplitudes[0] * amplitudes[1].conj(),
        2e-14,
    );
    let degenerate = prepared.factor(5.).unwrap();
    assert!(NoiseAccumulator::new(&degenerate, &outputs, &pairs, NoiseLimits::default()).is_err());
    let phasor = degenerate
        .plan()
        .real_phasor(
            &[Complex::new(1., 2.), Complex::new(3., 4.), Complex::ZERO],
            5.,
        )
        .unwrap();
    close(phasor, Complex::new(4., 2.), 1e-15);
}

#[test]
fn zero_pump_native_noise_and_ac_match_dc_bias_analyses() {
    let lattice = HarmonicLattice::new(
        &[ToneAxis {
            frequency: 1e6,
            order: 2,
        }],
        3,
        LatticeLimits::default(),
    )
    .unwrap();
    let (mut circuit, config) =
        circuit("v1 in 0 dc 1 ac 1\nr1 in out 1k\nr2 out 0 3k\nc1 out 0 1n");
    let ordinary = config
        .plan(AnalysisInput::with_arguments(
            AnalysisKind::Noise,
            ["v(out)", "v1", "lin", "1", "1.3meg", "1.3meg"],
        ))
        .unwrap()
        .run_plots(&mut circuit, &config.context())
        .unwrap();
    let expected = ordinary[0].value("onoise_spectrum", 0).unwrap().re.powi(2);
    let ac = config
        .plan(AnalysisInput::with_arguments(
            AnalysisKind::Ac,
            ["lin", "1", "1.3meg", "1.3meg"],
        ))
        .unwrap()
        .run(&mut circuit, &config.context())
        .unwrap()
        .value("v(out)", 0)
        .unwrap();
    circuit.finalize().unwrap();
    let node = circuit
        .unknowns()
        .node_row(circuit.nodes().get("out").unwrap())
        .unwrap();
    let branch = circuit.branch_rows(0).unwrap().start;
    let context = config.context().model_context();
    let tolerance = vec![1e-12; circuit.unknown_count()];
    let orbit = solve_circuit(
        &mut circuit,
        context,
        &lattice,
        None,
        &tolerance,
        BalanceLimits::default(),
    )
    .unwrap();
    let prepared = PeriodicLinearization::circuit(
        &lattice,
        &mut circuit,
        context,
        &orbit,
        &tolerance,
        BalanceLimits::default(),
    )
    .unwrap();
    let response = prepared.factor(1.3e6).unwrap();
    let center = response.plan().locate(&[0]).unwrap();
    let outputs = [unit(node, center)];
    let gain = response
        .transfer(
            &[unit(branch, center)],
            &outputs,
            ResponseLimits {
                max_backward_error: Some(1e-12),
                ..ResponseLimits::default()
            },
        )
        .unwrap()[0];
    close(gain, ac, 2e-13);
    let mut noise = NoiseAccumulator::new(
        &response,
        &outputs,
        &[CovariancePair { left: 0, right: 0 }],
        NoiseLimits::default(),
    )
    .unwrap();
    noise.add_circuit_noise().unwrap();
    assert!(noise.add_circuit_noise().is_err());
    let result = noise.finish().unwrap();
    assert!(
        (result.densities[0].re / expected - 1.).abs() < 2e-12,
        "{:?} vs {expected:e}",
        result.densities
    );
    let impedance = Complex::real(1.) / Complex::new(1. / 1000. + 1. / 3000., TAU * 1.3e6 * 1e-9);
    let density =
        4. * BOLTZMANN * 300.15 * (1. / 1000. + 1. / 3000.) * impedance.magnitude().powi(2);
    assert!((result.densities[0].re / density - 1.).abs() < 2e-12);
}

#[test]
fn pumped_diode_shot_noise_has_cyclostationary_sideband_covariance() {
    // Exact forced voltage with a small LO; ideal voltage source branch
    // observes the junction current. The independent reference computes
    // Fourier coefficients of sqrt(2 q I(v)) and finite-lattice covariance.
    let lattice = HarmonicLattice::new(
        &[ToneAxis {
            frequency: 1e6,
            order: 5,
        }],
        4,
        LatticeLimits::default(),
    )
    .unwrap();
    let (mut circuit, config) =
        circuit("v1 in 0 sin(0.45 0.025 1meg)\nd1 in 0 d\n.model d d(is=1e-12 kf=0 rs=0)");
    circuit.finalize().unwrap();
    let branch = circuit.branch_rows(0).unwrap().start;
    let context = config.context().model_context();
    let tolerance = vec![1e-11; circuit.unknown_count()];
    let orbit = solve_circuit(
        &mut circuit,
        context,
        &lattice,
        None,
        &tolerance,
        BalanceLimits::default(),
    )
    .unwrap();
    let prepared = PeriodicLinearization::circuit(
        &lattice,
        &mut circuit,
        context,
        &orbit,
        &tolerance,
        BalanceLimits::default(),
    )
    .unwrap();
    let response = prepared.factor(1.3e6).unwrap();
    let center = response.plan().locate(&[0]).unwrap();
    let up = response.plan().locate(&[1]).unwrap();
    let outputs = [unit(branch, center), unit(branch, up)];
    let pairs = [
        CovariancePair { left: 0, right: 0 },
        CovariancePair { left: 1, right: 1 },
        CovariancePair { left: 0, right: 1 },
    ];
    let mut noise =
        NoiseAccumulator::new(&response, &outputs, &pairs, NoiseLimits::default()).unwrap();
    noise.add_circuit_noise().unwrap();
    let actual = noise.finish().unwrap();
    let coefficients = (-10..=10)
        .map(|k| {
            (0..4096).fold(Complex::ZERO, |sum, i| {
                let phase = TAU * f64::from(i) / 4096.;
                let voltage = 0.025_f64.mul_add(phase.sin(), 0.45);
                let current = 1e-12 * (voltage / (BOLTZMANN * 300.15 / CHARGE)).exp_m1()
                    + context.gmin * voltage;
                let amplitude = (2. * CHARGE * current.abs()).sqrt() / 4096.;
                let (sin, cos) = (-f64::from(k) * phase).sin_cos();
                sum + Complex::new(amplitude * cos, amplitude * sin)
            })
        })
        .collect::<Vec<_>>();
    let mut expected = [Complex::ZERO; 3];
    for k in 0..11 {
        let a = coefficients[15 - k];
        let b = coefficients[16 - k];
        expected[0] = expected[0] + a * a.conj();
        expected[1] = expected[1] + b * b.conj();
        expected[2] = expected[2] + a * b.conj();
    }
    // Offset correction: the signed source coordinate runs -5..=5,
    // so difference for center is 5-index, for +1 is 6-index.
    for (value, want) in actual.densities.iter().zip(expected) {
        assert!(
            (*value - want).magnitude() < want.magnitude() * 3e-6,
            "{value:?} != {want:?}"
        );
    }
}

#[test]
fn conversion_noise_figure_uses_reference_channel_not_total_sideband_gain() {
    let figure =
        NoiseFigure::from_added_noise(Complex::real(0.5), BOLTZMANN * 290. * 0.25, 290.).unwrap();
    assert_eq!(figure.gain, 0.25);
    assert!((figure.factor - 2.).abs() < 1e-15);
    assert!((figure.decibels - 3.010_299_956_639_812).abs() < 1e-14);
}

fn transient_phasor(
    plot: &ngspice_rs::analysis::Plot,
    variable: &str,
    frequency: f64,
    start: f64,
    stop: f64,
) -> Complex {
    let time = plot.column("time").unwrap();
    let data = plot.column(variable).unwrap();
    let mut cursor = 0;
    let samples = 16384;
    let mut sum = Complex::ZERO;
    for index in 0..samples {
        let t = (f64::from(index) + 0.5).mul_add((stop - start) / f64::from(samples), start);
        while time[cursor + 1].re < t {
            cursor += 1;
        }
        let fraction = (t - time[cursor].re) / (time[cursor + 1].re - time[cursor].re);
        let value = (data[cursor + 1].re - data[cursor].re).mul_add(fraction, data[cursor].re);
        let (sin, cos) = (-TAU * frequency * t).sin_cos();
        sum = sum + Complex::new(cos, sin) * Complex::real(2. * value / f64::from(samples));
    }
    sum
}

#[test]
fn smooth_switched_mixer_matches_independent_weak_signal_transient() {
    let lattice = HarmonicLattice::new(
        &[ToneAxis {
            frequency: 100e3,
            order: 15,
        }],
        4,
        LatticeLimits::default(),
    )
    .unwrap();
    let mixer = "v1 lo 0 sin(0 1 100k)\nb1 out 0 v=tanh(5*v(lo))*v(rf)\nr1 out 0 1k";
    let (mut pumped, config) = circuit(&format!("{mixer}\nv2 rf 0 dc 0"));
    pumped.finalize().unwrap();
    let row = pumped
        .unknowns()
        .node_row(pumped.nodes().get("out").unwrap())
        .unwrap();
    let drive = pumped
        .devices()
        .iter()
        .position(|d| d.name() == "v2")
        .unwrap();
    let branch = pumped.branch_rows(drive).unwrap().start;
    let tolerance = vec![1e-10; pumped.unknown_count()];
    let context = config.context().model_context();
    let orbit = solve_circuit(
        &mut pumped,
        context,
        &lattice,
        None,
        &tolerance,
        BalanceLimits::default(),
    )
    .unwrap();
    let prepared = PeriodicLinearization::circuit(
        &lattice,
        &mut pumped,
        context,
        &orbit,
        &tolerance,
        BalanceLimits::default(),
    )
    .unwrap();
    let response = prepared.factor(130e3).unwrap();
    let center = response.plan().locate(&[0]).unwrap();
    let excitation = vec![Term {
        at: Coordinate {
            row: branch,
            band: center,
        },
        value: Complex::imaginary(-1e-3),
    }];
    let outputs = (0..lattice.coordinates())
        .map(|band| unit(row, band))
        .collect::<Vec<_>>();
    let converted = response
        .transfer(
            &[excitation],
            &outputs,
            ResponseLimits {
                max_backward_error: Some(1e-10),
                ..ResponseLimits::default()
            },
        )
        .unwrap();
    let (mut transient, config) = circuit(&format!("{mixer}\nv2 rf 0 sin(0 1m 130k)"));
    let plot = config
        .plan(AnalysisInput::with_arguments(
            AnalysisKind::Transient,
            ["10n", "200u", "90u", "10n"],
        ))
        .unwrap()
        .run(&mut transient, &config.context())
        .unwrap();
    for frequency in [30e3, 170e3, 230e3, 430e3] {
        let expected = response.plan().real_phasor(&converted, frequency).unwrap();
        let actual = transient_phasor(&plot, "v(out)", frequency, 100e-6, 200e-6);
        assert!(expected.magnitude() > 5e-5);
        close(actual, expected, 2e-7);
    }
}

#[test]
fn zero_pump_pattern_storage_grows_linearly_in_sideband_count() {
    let (mut circuit, config) = circuit("v1 in 0 dc 1\nr1 in out 1k\nc1 out 0 1n");
    for order in [1, 9] {
        let lattice = HarmonicLattice::new(
            &[ToneAxis {
                frequency: 1e6,
                order,
            }],
            3,
            LatticeLimits::default(),
        )
        .unwrap();
        circuit.finalize().unwrap();
        let tolerance = vec![1e-12; circuit.unknown_count()];
        let context = config.context().model_context();
        let orbit = solve_circuit(
            &mut circuit,
            context,
            &lattice,
            None,
            &tolerance,
            BalanceLimits::default(),
        )
        .unwrap();
        let prepared = PeriodicLinearization::circuit(
            &lattice,
            &mut circuit,
            context,
            &orbit,
            &tolerance,
            BalanceLimits::default(),
        )
        .unwrap();
        let response = prepared.factor(1.3e6).unwrap();
        let stored = response.factor_storage();
        assert!(
            stored.original_entries <= 8 * lattice.coordinates(),
            "{stored:?}"
        );
        assert!(
            stored.factor_entries <= 16 * lattice.coordinates(),
            "{stored:?}"
        );
    }
}

#[derive(Debug)]
struct CoherentThermal {
    terminals: [ngspice_rs::primitives::NodeId; 2],
    fail: bool,
}
impl ngspice_rs::devices::Device for CoherentThermal {
    fn name(&self) -> &str {
        "ncoherent"
    }
    fn designator(&self) -> char {
        'n'
    }
    fn terminals(&self) -> &[ngspice_rs::primitives::NodeId] {
        &self.terminals
    }
    fn stamp(&self, _: &mut ngspice_rs::devices::StampContext<'_>) -> SpiceResult<()> {
        Ok(())
    }
    fn assemble_linear(&self, _: &mut ngspice_rs::devices::LinearContext<'_>) -> SpiceResult<()> {
        Ok(())
    }
    fn noise(
        &self,
        ctx: &ngspice_rs::devices::noise::NoiseContext<'_>,
        sources: &mut Vec<ngspice_rs::devices::noise::NoiseSource>,
    ) -> SpiceResult<ngspice_rs::devices::noise::DeviceNoise<'_>> {
        use ngspice_rs::devices::noise::{DeviceNoise, NoiseFamily, NoiseKind, NoiseSource};
        use ngspice_rs::primitives::NodeId;
        let [a, b] = self.terminals;
        let suffix = if self.fail && ctx.model_context.frequency.to_bits() == 13_f64.to_bits() {
            "_changed"
        } else {
            "_coherent"
        };
        sources.push(NoiseSource::new(
            suffix,
            [a, NodeId::GROUND],
            NoiseKind::CorrelatedThermal {
                conductance: 2e-3,
                second: [b, NodeId::GROUND],
                second_conductance: 1e-3,
                phase: 0.7,
                temperature: 300.15,
            },
        ));
        Ok(DeviceNoise::Sources {
            family: NoiseFamily::Resistor,
            model: None,
            total: false,
        })
    }
}

#[test]
fn native_correlated_thermal_noise_conjugates_negative_sidebands_and_refresh_is_atomic() {
    let lattice = HarmonicLattice::new(
        &[ToneAxis {
            frequency: 10.,
            order: 1,
        }],
        3,
        LatticeLimits::default(),
    )
    .unwrap();
    for fail in [false, true] {
        let (mut circuit, config) = circuit("r1 a 0 1k\nr2 b 0 2k");
        let nodes = [
            circuit.nodes().get("a").unwrap(),
            circuit.nodes().get("b").unwrap(),
        ];
        circuit
            .add_device(Box::new(CoherentThermal {
                terminals: nodes,
                fail,
            }))
            .unwrap();
        circuit.finalize().unwrap();
        let [a, b] = nodes.map(|node| circuit.unknowns().node_row(node).unwrap());
        let tolerance = vec![1e-12; circuit.unknown_count()];
        let context = config.context().model_context();
        let orbit = solve_circuit(
            &mut circuit,
            context,
            &lattice,
            None,
            &tolerance,
            BalanceLimits::default(),
        )
        .unwrap();
        let prepared = PeriodicLinearization::circuit(
            &lattice,
            &mut circuit,
            context,
            &orbit,
            &tolerance,
            BalanceLimits::default(),
        )
        .unwrap();
        let response = prepared.factor(3.).unwrap();
        let outputs = [unit(a, 1), unit(b, 1), unit(a, 0), unit(b, 0)];
        let pairs = [
            CovariancePair { left: 0, right: 1 },
            CovariancePair { left: 2, right: 3 },
            CovariancePair { left: 0, right: 0 },
        ];
        let mut noise =
            NoiseAccumulator::new(&response, &outputs, &pairs, NoiseLimits::default()).unwrap();
        let source = vec![Term {
            at: Coordinate { row: a, band: 1 },
            value: Complex::real(1e-10),
        }];
        if fail {
            noise.add_factor(&source).unwrap();
            assert!(noise.add_circuit_noise().is_err());
            let result = noise.finish().unwrap();
            assert_eq!(result.factors, 1);
            assert_eq!(result.densities[0], Complex::ZERO);
            assert_eq!(result.densities[1], Complex::ZERO);
            close(result.densities[2], Complex::real(1e-14), 1e-28);
        } else {
            noise.add_circuit_noise().unwrap();
            let result = noise.finish().unwrap();
            let scale = 4. * BOLTZMANN * 300.15 * (2e-3_f64 * 1e-3).sqrt() * 1000. * 2000.;
            let expected = Complex::new(0.7_f64.cos(), -0.7_f64.sin()) * Complex::real(scale);
            close(result.densities[0], expected, scale * 1e-12);
            close(result.densities[1], expected.conj(), scale * 1e-12);
        }
    }
}

#[test]
fn hb_native_port_power_tone_uses_the_resolved_physical_source() {
    let lattice = HarmonicLattice::new(
        &[ToneAxis {
            frequency: 1e6,
            order: 1,
        }],
        3,
        LatticeLimits::default(),
    )
    .unwrap();
    let (mut circuit, config) = circuit("v1 a 0 portnum 1 z0 50 pwr 0.001 freq 1meg\nr1 a 0 50");
    circuit.finalize().unwrap();
    let a = circuit
        .unknowns()
        .node_row(circuit.nodes().get("a").unwrap())
        .unwrap();
    let tolerance = vec![1e-12; circuit.unknown_count()];
    let context = config.context().model_context();
    let orbit = solve_circuit(
        &mut circuit,
        context,
        &lattice,
        None,
        &tolerance,
        BalanceLimits::default(),
    )
    .unwrap();
    let spectrum = lattice
        .phasors(&orbit.coefficients.as_slice()[a * 3..(a + 1) * 3])
        .unwrap();
    close(spectrum[1], Complex::real((0.001_f64 * 50.).sqrt()), 1e-12);
}
