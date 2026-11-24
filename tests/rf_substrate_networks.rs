//! Independent line and end-to-end RF port regressions.
use std::{f64::consts::TAU, path::Path, sync::Arc};

use ngspice_rs::analysis::{
    RunConfig,
    rf_network::{FrequencyBand, LineMode, ModeParameters, PreparedLine},
};
use ngspice_rs::devices::{nport::PortTerminals, rf_passive::RfLine};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{Complex, NodeId};

fn mode(z: f64, velocity: f64) -> ModeParameters {
    ModeParameters {
        resistance: 0.,
        inductance: z / velocity,
        conductance: 0.,
        capacitance: 1. / (z * velocity),
        skin: 0.,
        relaxation_capacitance: 0.,
        relaxation_time: 0.,
        reference_temperature: 27.,
        resistance_temperature_coefficient: 0.,
    }
}

fn band() -> FrequencyBand {
    FrequencyBand::new(0., 1e15).unwrap()
}

fn close(actual: Complex, expected: Complex, tolerance: f64) {
    assert!(
        (actual - expected).magnitude() <= tolerance,
        "actual {actual:?} expected {expected:?}"
    );
}

#[test]
fn matched_lossless_line_is_exact_delay_even_at_resonances() {
    let z = 50.;
    let velocity = 2e8;
    let length = 0.1;
    let line =
        PreparedLine::single(LineMode::new(mode(z, velocity)).unwrap(), length, band()).unwrap();
    for hz in [0., 1e-12, 1e6, 5e8, 1e9, 2e9, 1.01e10, 1e12] {
        let phase = -(hz * length / velocity) * TAU;
        let expected = Complex::new(phase.cos(), phase.sin());
        let s = line.scattering(hz, 27., z).unwrap();
        close(s.get(0, 0).unwrap(), Complex::ZERO, 2e-12);
        close(s.get(1, 0).unwrap(), expected, 2e-11);
        close(s.get(0, 1).unwrap(), expected, 2e-11);
    }
}

#[test]
fn dc_and_zero_length_are_constraints_not_artificial_shunts() {
    let mut p = mode(50., 2e8);
    p.resistance = 25.;
    let line = PreparedLine::single(LineMode::new(p).unwrap(), 2., band()).unwrap();
    let s = line.scattering(0., 27., 50.).unwrap();
    close(s.get(0, 0).unwrap(), Complex::real(1. / 3.), 2e-15);
    close(s.get(1, 0).unwrap(), Complex::real(2. / 3.), 2e-15);
    let zero = PreparedLine::single(LineMode::new(p).unwrap(), 0., band()).unwrap();
    for f in [0., 1e6, 1e12] {
        let s = zero.scattering(f, 27., 50.).unwrap();
        assert_eq!(s.get(0, 0), Some(Complex::ZERO));
        assert_eq!(s.get(1, 0), Some(Complex::real(1.)));
    }
}

#[test]
fn unequal_coupled_modes_preserve_power_and_reciprocity() {
    let p = LineMode::new(mode(50., 2e8)).unwrap();
    let q = LineMode::new(mode(70., 1.6e8)).unwrap();
    let angle = 0.37_f64;
    let (sin, cos) = angle.sin_cos();
    let line = PreparedLine::modal(vec![p, q], vec![cos, sin, -sin, cos], 0.1, band()).unwrap();
    let first = PreparedLine::single(p, 0.1, band()).unwrap();
    let second = PreparedLine::single(q, 0.1, band()).unwrap();
    for hz in [0., 1e6, 5e8, 1e9, 1e10] {
        let s = line.scattering(hz, 27., 50.).unwrap();
        let a = first.scattering(hz, 27., 50.).unwrap().get(1, 0).unwrap();
        let b = second.scattering(hz, 27., 50.).unwrap().get(1, 0).unwrap();
        close(
            s.get(2, 1).unwrap(),
            Complex::real(sin * cos) * (a - b),
            2e-14,
        );
        for i in 0..4 {
            for j in 0..4 {
                close(s.get(i, j).unwrap(), s.get(j, i).unwrap(), 1e-15);
                let dot = (0..4).fold(Complex::ZERO, |sum, k| {
                    sum + s.get(k, i).unwrap().conj() * s.get(k, j).unwrap()
                });
                close(dot, Complex::real(if i == j { 1. } else { 0. }), 2e-14);
            }
        }
    }
}

#[test]
fn lossy_material_laws_are_passive_and_temperature_aware() {
    let mut p = mode(50., 2e8);
    p.resistance = 3.;
    p.skin = 1e-4;
    p.conductance = 1e-5;
    p.relaxation_capacitance = 1e-11;
    p.relaxation_time = 1e-10;
    p.resistance_temperature_coefficient = 0.004;
    let m = LineMode::new(p).unwrap();
    let (cold, _) = m.rlgc(1e8, 27.).unwrap();
    let (hot, _) = m.rlgc(1e8, 77.).unwrap();
    assert!(hot.re > cold.re);
    let line = PreparedLine::single(m, 1., band()).unwrap();
    for hz in [0., 1., 1e3, 1e6, 1e9, 1e12, 1e15] {
        let [even, odd] = line.relations(hz, 77.).unwrap()[0]
            .reflections(50.)
            .unwrap();
        assert!(even.magnitude() <= 1. + 2e-15);
        assert!(odd.magnitude() <= 1. + 2e-15);
    }
    assert!(line.relations(1e16, 27.).is_err());
    assert!(line.relations(1e8, -274.).is_err());
    assert!(PreparedLine::modal(vec![m, m], vec![1., 1., 0., 1.], 1., band()).is_err());
}

#[test]
fn prepared_line_runs_through_real_sp_and_ac_ports() {
    let deck = "prepared RF line\n\
      v1 in 0 dc 0 ac 1 portnum 1 z0 50\n\
      v2 out 0 dc 0 portnum 2 z0 50\n\
      .sp lin 3 500meg 1.5g\n\
      .ac lin 3 500meg 1.5g\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("rf.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let line = Arc::new(
        PreparedLine::single(LineMode::new(mode(50., 2e8)).unwrap(), 0.1, band()).unwrap(),
    );
    for analysis in &netlist.analyses {
        let mut circuit = config.circuit(&netlist).unwrap();
        let input = circuit.add_node("in").unwrap();
        let output = circuit.add_node("out").unwrap();
        circuit
            .add_device(Box::new(
                RfLine::new(
                    "nrf",
                    Arc::clone(&line),
                    vec![
                        PortTerminals {
                            positive: input,
                            negative: NodeId::GROUND,
                        },
                        PortTerminals {
                            positive: output,
                            negative: NodeId::GROUND,
                        },
                    ],
                )
                .unwrap(),
            ))
            .unwrap();
        let plot = config
            .plan_for(analysis)
            .unwrap()
            .run(&mut circuit, &config.context())
            .unwrap();
        assert_eq!(plot.point_count(), 3);
        if plot.plotname == "SP Analysis" {
            for (i, hz) in [5e8, 1e9, 1.5e9].into_iter().enumerate() {
                let s = line.scattering(hz, 27., 50.).unwrap();
                close(plot.value("S_1_1", i).unwrap(), s.get(0, 0).unwrap(), 1e-10);
                close(plot.value("S_2_1", i).unwrap(), s.get(1, 0).unwrap(), 1e-10);
            }
        } else {
            assert!(plot.rows().flatten().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn geometry_matches_independent_scikit_rf_reference() {
    use ngspice_rs::analysis::rf_network::geometry::{
        Coax, GroundedCpw, Microstrip, ModeLoss, Substrate,
    };
    // scikit-rf 2.2.0, 1 Hz, frequency-invariant dielectric, no dispersion
    // or losses. CPW is conductor-backed and zero-thickness.
    for (width, height, thickness, er, z, eps) in [
        (
            1e-3,
            0.8e-3,
            0.,
            4.4,
            63.503268067653146,
            3.2186839143754926,
        ),
        (
            1e-3,
            0.8e-3,
            35e-6,
            4.4,
            62.203031339095034,
            3.1646935207408653,
        ),
        (0.2e-3, 1e-3, 0., 9.8, 90.02269432867917, 6.040594353278439),
        (5e-3, 0.5e-3, 0., 2.2, 20.43921552095505, 2.015990046075876),
    ] {
        let cross = Microstrip { width, thickness }
            .quasi_static(Substrate {
                height,
                relative_permittivity: er,
            })
            .unwrap();
        assert!(
            (cross.impedance / z - 1.).abs() < 2e-8,
            "{cross:?} expected {z}"
        );
        assert!((cross.effective_permittivity / eps - 1.).abs() < 2e-8);
        let p = cross.prepare(ModeLoss::default()).unwrap().parameters();
        assert!(((p.inductance / p.capacitance).sqrt() / z - 1.).abs() < 2e-8);
    }
    for (width, gap, height, er, z, eps) in [
        (
            1e-3,
            0.2e-3,
            0.8e-3,
            4.4,
            48.93485254446609,
            2.885529890899531,
        ),
        (
            0.2e-3,
            0.1e-3,
            1e-3,
            9.8,
            51.33996037546831,
            5.430740930684419,
        ),
        (
            2e-3,
            0.1e-3,
            0.2e-3,
            3.66,
            15.401974675626064,
            3.1183735732951217,
        ),
    ] {
        let cross = GroundedCpw { width, gap }
            .quasi_static(Substrate {
                height,
                relative_permittivity: er,
            })
            .unwrap();
        // scikit-rf uses an approximate elliptic-integral ratio; the AGM
        // implementation is more accurate, so use the approximation budget.
        assert!(
            (cross.impedance / z - 1.).abs() < 5e-6,
            "{cross:?} expected {z}"
        );
        assert!((cross.effective_permittivity / eps - 1.).abs() < 5e-6);
    }
    let coax = Coax {
        inner_radius: 1e-3,
        shield_inner_radius: 3.5e-3,
        shield_outer_radius: 4e-3,
        relative_permittivity: 2.25,
    };
    let cross = coax.quasi_static().unwrap();
    assert!(
        (cross.impedance - 376.730313668 / std::f64::consts::TAU * 3.5_f64.ln() / 1.5).abs()
            < 1e-12
    );
    assert!((cross.velocity() - 299792458. / 1.5).abs() < 1e-7);
    let loss = coax.conductor_loss(5.8e7, 20., 0.0039).unwrap();
    assert!(loss.dc_resistance > 0. && loss.skin > 0.);
    let invalid = Coax {
        inner_radius: 4e-3,
        ..coax
    };
    assert!(invalid.quasi_static().is_err());
}

#[test]
fn coupled_microstrip_matches_qucs_kirschning_jansen_reference() {
    use ngspice_rs::analysis::rf_network::geometry::{CoupledMicrostrip, Substrate};
    // Direct C++ extraction from Qucsator e995f9acc71a8c7319286944e4a1692318b9dd80.
    for (width, height, gap, er, expected) in [
        (
            1e-3,
            8e-4,
            2e-4,
            4.4,
            [
                79.975293549611635,
                42.108690278121792,
                3.4329014410086169,
                2.8232237517350418,
            ],
        ),
        (
            2e-4,
            1e-3,
            1e-4,
            9.8,
            [
                133.35851520986853,
                42.258530582179446,
                6.2700271220999886,
                5.425856758714974,
            ],
        ),
        (
            3e-3,
            1e-3,
            1e-3,
            2.2,
            [
                56.406170266792088,
                44.635320755383042,
                1.9603410543742061,
                1.7670952865793828,
            ],
        ),
    ] {
        let [even, odd] = CoupledMicrostrip { width, gap }
            .quasi_static(Substrate {
                height,
                relative_permittivity: er,
            })
            .unwrap();
        for (actual, expected) in [
            even.impedance,
            odd.impedance,
            even.effective_permittivity,
            odd.effective_permittivity,
        ]
        .into_iter()
        .zip(expected)
        {
            assert!(
                (actual / expected - 1.).abs() < 2e-12,
                "{actual} != {expected}"
            );
        }
    }
}

fn ports(input: NodeId, output: NodeId) -> Vec<PortTerminals> {
    [input, output]
        .into_iter()
        .map(|positive| PortTerminals {
            positive,
            negative: NodeId::GROUND,
        })
        .collect()
}

#[test]
fn causal_lossy_ladder_matches_exact_sp_and_enforces_preparation_context() {
    use ngspice_rs::analysis::rf_network::realization::LadderSpec;
    use ngspice_rs::devices::rf_passive::bind_realization;
    let mut p = mode(50., 2e8);
    p.resistance = 2.;
    p.conductance = 1e-6;
    p.skin = 2e-5;
    p.relaxation_capacitance = 1e-11;
    p.relaxation_time = 1e-9;
    let line = PreparedLine::single(LineMode::new(p).unwrap(), 0.02, band()).unwrap();
    let spec = LadderSpec {
        band: FrequencyBand::new(0., 1e9).unwrap(),
        reference_impedance: 50.,
        max_error: 0.005,
        max_sections: 64,
        temperature: 27.,
    };
    let ladder = Arc::new(line.realize(spec).unwrap());
    let report = ladder.report().unwrap();
    assert!(report.samples >= 256 && report.worst_error <= spec.max_error);
    assert!(ladder.validate_context(1e9, 28.).is_err());
    assert!(ladder.validate_context(2e9, 27.).is_err());
    assert!(
        line.realize(LadderSpec {
            max_sections: 1,
            max_error: 1e-10,
            ..spec
        })
        .is_err()
    );
    let deck = "causal RF ladder\nv1 in 0 portnum 1 z0 50\nv2 out 0 portnum 2 z0 50\n.sp lin 9 1meg 1g\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("ladder.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let input = circuit.add_node("in").unwrap();
    let output = circuit.add_node("out").unwrap();
    bind_realization(
        &mut circuit,
        "nladder",
        Arc::clone(&ladder),
        ports(input, output),
    )
    .unwrap();
    let plot = config
        .plan_for(&netlist.analyses[0])
        .unwrap()
        .run(&mut circuit, &config.context())
        .unwrap();
    for point in 0..plot.point_count() {
        let hz = plot.value("frequency", point).unwrap().re;
        let exact = line.scattering(hz, 27., 50.).unwrap();
        close(
            plot.value("S_1_1", point).unwrap(),
            exact.get(0, 0).unwrap(),
            spec.max_error,
        );
        close(
            plot.value("S_2_1", point).unwrap(),
            exact.get(1, 0).unwrap(),
            spec.max_error,
        );
    }
}

#[test]
fn native_ladder_transient_propagates_bandlimited_sine() {
    use ngspice_rs::analysis::rf_network::realization::LadderSpec;
    use ngspice_rs::devices::rf_passive::bind_realization;
    let line = PreparedLine::single(LineMode::new(mode(50., 2e8)).unwrap(), 0.1, band()).unwrap();
    let spec = LadderSpec {
        band: FrequencyBand::new(0., 200e6).unwrap(),
        reference_impedance: 50.,
        max_error: 0.002,
        max_sections: 64,
        temperature: 27.,
    };
    let ladder = Arc::new(line.realize(spec).unwrap());
    let deck = "RF transient\nv1 src 0 sin(0 1 100meg)\nr1 src in 50\nr2 out 0 50\n.tran 20p 40n 0 20p\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("ladder-tran.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let input = circuit.add_node("in").unwrap();
    let output = circuit.add_node("out").unwrap();
    bind_realization(&mut circuit, "nladder", ladder, ports(input, output)).unwrap();
    let plot = config
        .plan_for(&netlist.analyses[0])
        .unwrap()
        .run(&mut circuit, &config.context())
        .unwrap();
    let mut max_error: f64 = 0.;
    let mut count = 0;
    for point in 0..plot.point_count() {
        let time = plot.value("time", point).unwrap().re;
        if time > 30e-9 {
            let expected = 0.5 * ((time - 0.5e-9) * 100e6 * TAU).sin();
            max_error = max_error.max((plot.value("v(out)", point).unwrap().re - expected).abs());
            count += 1;
        }
    }
    assert!(
        count > 100 && max_error < 0.01,
        "samples {count}, error {max_error}"
    );
}

#[test]
fn passive_junction_binding_is_atomic_and_preserves_floating_ports() {
    use ngspice_rs::analysis::rf_network::realization::PassiveRealization;
    use ngspice_rs::devices::{circuit::Circuit, rf_passive::bind_realization};
    let graph =
        Arc::new(PassiveRealization::junction(&[1e-9, 2e-9, 3e-9], 0.2e-12, band()).unwrap());
    let mut circuit = Circuit::default();
    let a = circuit.add_node("a").unwrap();
    let b = circuit.add_node("b").unwrap();
    let c = circuit.add_node("c").unwrap();
    let reference = circuit.add_node("ref").unwrap();
    let ports = [a, b, c]
        .map(|positive| PortTerminals {
            positive,
            negative: reference,
        })
        .to_vec();
    let before = circuit.nodes().len();
    let mut bad = ports.clone();
    bad[1].negative = NodeId::from_index(9999).unwrap();
    assert!(bind_realization(&mut circuit, "ntee", Arc::clone(&graph), bad).is_err());
    assert_eq!(circuit.nodes().len(), before);
    assert!(circuit.devices().is_empty());
    bind_realization(&mut circuit, "ntee", Arc::clone(&graph), ports.clone()).unwrap();
    let after = circuit.nodes().len();
    let count = circuit.devices().len();
    assert!(bind_realization(&mut circuit, "ntee", graph, ports).is_err());
    assert_eq!(circuit.nodes().len(), after);
    assert_eq!(circuit.devices().len(), count);
    circuit.finalize().unwrap();
}
