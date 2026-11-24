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
    let plan = config
        .plan(AnalysisInput::with_arguments(
            AnalysisKind::SParameter,
            ["lin", "1", "1.3meg", "1.3meg"],
        ))
        .unwrap();
    let ordinary = plan.run(&mut circuit, &config.context()).unwrap();
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
