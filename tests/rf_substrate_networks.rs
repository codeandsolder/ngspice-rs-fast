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
