//! Independent analytic and KCL regressions for sampled physical N-port models.
use std::sync::Arc;

use ngspice_rs::analysis::touchstone::{Touchstone, TouchstoneLimits};
use ngspice_rs::devices::MnaUnknowns;
use ngspice_rs::devices::nport::{
    Extrapolation, MeasuredNPort, NetworkProperties, PhysicalRequirements, PortTerminals,
    PropertyStatus,
};
use ngspice_rs::primitives::{Complex, NodeId, NodeTable};

fn measured(text: &str, nodes: Vec<PortTerminals>, interpolation: Extrapolation) -> MeasuredNPort {
    let path = if text.trim_start().starts_with('#') {
        format!("filter.s{}p", nodes.len())
    } else {
        "filter.ts".to_owned()
    };
    let file = Touchstone::parse(text, &path, TouchstoneLimits::default()).unwrap();
    MeasuredNPort::new("nnetwork", Arc::new(file), nodes, interpolation).unwrap()
}
fn oneport(text: &str) -> MeasuredNPort {
    measured(
        text,
        vec![PortTerminals {
            positive: NodeId::new(1),
            negative: NodeId::GROUND,
        }],
        Extrapolation::Reject,
    )
}
fn value(a: Complex, b: Complex) {
    assert!((a.re - b.re).abs() <= 2e-12 * b.re.abs().max(1.));
    assert!((a.im - b.im).abs() <= 2e-12 * b.im.abs().max(1.));
}

#[test]
fn matched_resistor_open_and_short_are_physical() {
    let resistor = oneport("# Hz S RI R 50\n1 0.3333333333333333 0\n3 0.3333333333333333 0\n");
    value(
        resistor.admittance_at(1.).unwrap().get(0, 0).unwrap(),
        Complex::real(0.01),
    );
    let open = oneport("# Hz S RI R 50\n1 1 0\n");
    assert_eq!(
        open.admittance_at(1.).unwrap().get(0, 0),
        Some(Complex::ZERO)
    );
    let short = oneport("# Hz S RI R 50\n1 -1 0\n");
    assert!(short.admittance_at(1.).is_err());
    assert!(resistor.admittance_at(0.).is_err());
    value(
        resistor.admittance_at(2.).unwrap().get(0, 0).unwrap(),
        Complex::real(0.01),
    );
}

#[test]
fn interpolation_is_rectangular_and_extrapolation_is_explicit() {
    let text = "[Version] 2.1\n# Hz S RI R 50\n[Number of Ports] 1\n[Number of Frequencies] 2\n[Network Data]\n1 0 0\n3 0.5 0\n[End]\n";
    let port = oneport(text);
    value(
        port.admittance_at(2.).unwrap().get(0, 0).unwrap(),
        Complex::real(0.012),
    );
    assert!(port.admittance_at(4.).is_err());
    let held = measured(
        text,
        vec![PortTerminals {
            positive: NodeId::new(1),
            negative: NodeId::GROUND,
        }],
        Extrapolation::Hold,
    );
    value(
        held.admittance_at(4.).unwrap().get(0, 0).unwrap(),
        Complex::real(1. / 150.),
    );
}

#[test]
fn heterogeneous_reference_impedances_preserve_power_wave_normalization() {
    let text = "[Version] 2.1\n# Hz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Reference] 50 75\n[Network Data]\n1 0.2 0 0 0 0 0 0.14285714285714285 0\n[End]\n";
    let net = measured(
        text,
        vec![
            PortTerminals {
                positive: NodeId::new(1),
                negative: NodeId::GROUND,
            },
            PortTerminals {
                positive: NodeId::new(2),
                negative: NodeId::GROUND,
            },
        ],
        Extrapolation::Reject,
    );
    let y = net.admittance_at(1.).unwrap();
    value(y.get(0, 0).unwrap(), Complex::real(1. / 75.));
    value(y.get(1, 1).unwrap(), Complex::real(1. / 100.));
    assert_eq!(y.get(0, 1), Some(Complex::ZERO));
}

#[test]
fn mixed_mode_s_and_y_are_transformed_back_to_single_ended() {
    let s = "[Version] 2.1\n# Hz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Mixed-Mode Order] D1,2 C1,2\n[Network Data]\n1 0.2 0 0 0 0 0 0.6 0\n[End]\n";
    let nodes = vec![
        PortTerminals {
            positive: NodeId::new(1),
            negative: NodeId::GROUND,
        },
        PortTerminals {
            positive: NodeId::new(2),
            negative: NodeId::GROUND,
        },
    ];
    let net = measured(s, nodes.clone(), Extrapolation::Reject);
    let y = net.admittance_at(1.).unwrap();
    value(y.get(0, 0).unwrap(), Complex::real(11. / 1200.));
    value(y.get(0, 1).unwrap(), Complex::real(-5. / 1200.));
    value(y.get(1, 0).unwrap(), Complex::real(-5. / 1200.));
    let yfile = "[Version] 2.1\n# Hz Y RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Mixed-Mode Order] D1,2 C1,2\n[Network Data]\n1 4 0 0 0 0 0 8 0\n[End]\n";
    let yport = measured(yfile, nodes, Extrapolation::Reject);
    let y = yport.admittance_at(1.).unwrap();
    for (r, c, expect) in [(0, 0, 6.), (0, 1, -2.), (1, 0, -2.), (1, 1, 6.)] {
        value(y.get(r, c).unwrap(), Complex::real(expect));
    }
}

#[test]
fn terminal_references_produce_signed_kcl_stamps() {
    let port = measured(
        "# Hz Y RI R 50\n1 0.5 0\n",
        vec![PortTerminals {
            positive: NodeId::new(1),
            negative: NodeId::new(2),
        }],
        Extrapolation::Reject,
    );
    let mut nodes = NodeTable::new();
    nodes.intern("out").unwrap();
    nodes.intern("reference").unwrap();
    let mut mna = MnaUnknowns::new();
    mna.rebuild(&nodes);
    let stamps = port.nodal_stamps_at(1., &mna).unwrap();
    let magnitude = 0.5 / 50.;
    assert_eq!(stamps.len(), 4);
    for ((row, col, actual), (expected_row, expected_col, expected)) in stamps.into_iter().zip([
        (0, 0, magnitude),
        (0, 1, -magnitude),
        (1, 0, -magnitude),
        (1, 1, magnitude),
    ]) {
        assert_eq!((row, col), (expected_row, expected_col));
        value(actual, Complex::real(expected));
    }
    // Nodes not in the finalized map cannot silently become ground.
    let external = measured(
        "# Hz S RI R 50\n1 0 0\n",
        vec![PortTerminals {
            positive: NodeId::new(3),
            negative: NodeId::GROUND,
        }],
        Extrapolation::Reject,
    );
    assert!(external.nodal_stamps_at(1., &mna).is_err());
}

#[test]
fn ideal_through_two_port_is_a_nodal_constraint_not_zero_admittance() {
    let text = "[Version] 2.1\n# Hz S RI R 50\n[Number of Ports] 2\n[Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n[Network Data]\n1 0 0 1 0 1 0 0 0\n[End]\n";
    let network = measured(
        text,
        vec![
            PortTerminals {
                positive: NodeId::new(1),
                negative: NodeId::GROUND,
            },
            PortTerminals {
                positive: NodeId::new(2),
                negative: NodeId::GROUND,
            },
        ],
        Extrapolation::Reject,
    );
    assert!(network.admittance_at(1.).is_err());
}

#[test]
fn attached_measured_nport_participates_in_full_ac_and_sp_solves() {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::Parser;
    use ngspice_rs::netlist::source::parse_deck_text;
    use std::path::Path;

    let deck = "attached 100ohm measured load\n\
        v1 in 0 dc 0 ac 1 portnum 1 z0 50\n\
        .sp lin 1 1k 1k\n\
        .ac lin 1 1k 1k\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("attached.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let file = Touchstone::parse(
        "# Hz S RI R 50\n1e3 0.3333333333333333 0\n",
        "load.s1p",
        TouchstoneLimits::default(),
    )
    .unwrap();
    for analysis in &netlist.analyses {
        let mut circuit = config.circuit(&netlist).unwrap();
        let node = circuit.add_node("in").unwrap();
        circuit
            .add_device(Box::new(
                MeasuredNPort::new(
                    "nmeasured",
                    Arc::new(file.clone()),
                    vec![PortTerminals {
                        positive: node,
                        negative: NodeId::GROUND,
                    }],
                    Extrapolation::Hold,
                )
                .unwrap(),
            ))
            .unwrap();
        let request = config.plan_for(analysis).unwrap();
        let plot = request.run(&mut circuit, &config.context()).unwrap();
        if plot.plotname == "SP Analysis" {
            assert_eq!(plot.point_count(), 1);
            value(plot.value("S_1_1", 0).unwrap(), Complex::real(1. / 3.));
            value(plot.value("Z_1_1", 0).unwrap(), Complex::real(100.));
        } else {
            assert_eq!(plot.plotname, "AC Analysis");
            assert_eq!(plot.point_count(), 1);
            assert!(plot.rows().flatten().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn sampled_network_physical_properties_reject_gain_nonreciprocity_and_unproved_causality() {
    let passive = oneport("# Hz S RI R 50\n1 0.3333333333333333 0\n");
    assert_eq!(
        passive.properties_at(1.).unwrap(),
        NetworkProperties {
            reciprocity: PropertyStatus::Verified,
            passivity: PropertyStatus::Verified,
            causality: PropertyStatus::Undetermined,
        }
    );
    assert!(
        passive
            .require_properties_at(
                1.,
                PhysicalRequirements {
                    reciprocal: true,
                    passive: true,
                    causal: false,
                }
            )
            .is_ok()
    );
    assert!(
        passive
            .require_properties_at(
                1.,
                PhysicalRequirements {
                    reciprocal: false,
                    passive: false,
                    causal: true,
                }
            )
            .is_err()
    );

    let active = oneport("# Hz S RI R 50\n1 2 0\n");
    assert_eq!(
        active.properties_at(1.).unwrap().passivity,
        PropertyStatus::Violated
    );
    assert!(
        active
            .require_properties_at(
                1.,
                PhysicalRequirements {
                    passive: true,
                    ..PhysicalRequirements::default()
                }
            )
            .is_err()
    );

    let nonreciprocal = "[Version] 2.1\n# Hz Y RI R 50\n[Number of Ports] 2\n\
        [Two-Port Data Order] 12_21\n[Number of Frequencies] 1\n\
        [Network Data]\n1 1 0 0.1 0 0.2 0 1 0\n[End]\n";
    let network = measured(
        nonreciprocal,
        vec![
            PortTerminals {
                positive: NodeId::new(1),
                negative: NodeId::GROUND,
            },
            PortTerminals {
                positive: NodeId::new(2),
                negative: NodeId::GROUND,
            },
        ],
        Extrapolation::Reject,
    );
    let report = network.properties_at(1.).unwrap();
    assert_eq!(report.passivity, PropertyStatus::Verified);
    assert_eq!(report.reciprocity, PropertyStatus::Violated);
    assert!(
        network
            .require_properties_at(
                1.,
                PhysicalRequirements {
                    reciprocal: true,
                    ..PhysicalRequirements::default()
                }
            )
            .is_err()
    );
    assert!(
        network
            .require_properties_at(
                1.,
                PhysicalRequirements {
                    passive: true,
                    ..PhysicalRequirements::default()
                }
            )
            .is_ok()
    );
}

#[test]
fn lossless_reactive_networks_have_zero_dissipative_hermitian_part() {
    let reactive = oneport(
        "[Version] 2.1\n# Hz Y RI R 50\n[Number of Ports] 1\n\
        [Number of Frequencies] 1\n[Network Data]\n1 0 0.25\n[End]\n",
    );
    let report = reactive.properties_at(1.).unwrap();
    assert_eq!(report.passivity, PropertyStatus::Verified);
    assert_eq!(report.reciprocity, PropertyStatus::Verified);
}

#[test]
fn four_port_unequal_references_and_two_coupled_pairs_match_exact_spectral_algebra() {
    use std::fmt::Write as _;
    let refs = [25., 100., 50., 75.];
    let mut text = String::from(
        "[Version] 2.1\n# Hz S RI R 50\n[Number of Ports] 4\n[Number of Frequencies] 2\n[Reference] 25 100 50 75\n[Network Data]\n",
    );
    let s = [
        [0., 0.5, 0., 0.],
        [0.5, 0., 0., 0.],
        [0., 0., 0., -0.25],
        [0., 0., -0.25, 0.],
    ];
    for hz in [1000., 2000.] {
        write!(text, "{hz}").unwrap();
        for row in s {
            for value in row {
                write!(text, " {value} 0").unwrap();
            }
        }
        text.push('\n');
    }
    text.push_str("[End]\n");
    let nodes = (1..=4)
        .map(|i| PortTerminals {
            positive: NodeId::new(i),
            negative: NodeId::GROUND,
        })
        .collect();
    let four = measured(&text, nodes, Extrapolation::Reject);
    for hz in [1000., 1500., 2000.] {
        let y = four.admittance_at(hz).unwrap();
        for (base, k) in [(0, 0.5_f64), (2, -0.25)] {
            let first = refs[base];
            let second = refs[base + 1];
            let diagonal = k.mul_add(k, 1.) / k.mul_add(-k, 1.);
            let mutual = -2. * k / k.mul_add(-k, 1.);
            value(y.get(base, base).unwrap(), Complex::real(diagonal / first));
            value(
                y.get(base + 1, base + 1).unwrap(),
                Complex::real(diagonal / second),
            );
            value(
                y.get(base, base + 1).unwrap(),
                Complex::real(mutual / (first * second).sqrt()),
            );
            value(
                y.get(base + 1, base).unwrap(),
                Complex::real(mutual / (first * second).sqrt()),
            );
        }
        for row in 0..4 {
            for col in 0..4 {
                if row / 2 != col / 2 {
                    assert_eq!(y.get(row, col), Some(Complex::ZERO));
                }
            }
        }
        let report = four.properties_at(hz).unwrap();
        assert_eq!(report.passivity, PropertyStatus::Verified);
        assert_eq!(report.reciprocity, PropertyStatus::Verified);
    }
    let file = four
        .network()
        .write(
            ngspice_rs::analysis::touchstone::Version::V2_1,
            ngspice_rs::analysis::touchstone::Encoding::Ri,
        )
        .unwrap();
    let reloaded = Touchstone::parse(&file, "fourport.ts", TouchstoneLimits::default()).unwrap();
    assert_eq!(reloaded.points, four.network().points);
}

#[test]
fn active_nonreciprocal_four_port_is_reported_without_assuming_passivity() {
    use std::fmt::Write as _;
    let mut text = String::from(
        "[Version] 2.1\n# Hz Y RI R 50\n[Number of Ports] 4\n[Number of Frequencies] 1\n[Network Data]\n1",
    );
    for row in 0..4 {
        for col in 0..4 {
            let value = if row == col {
                if row == 0 { -1. } else { 0.01 }
            } else if row == 0 && col == 1 {
                2.
            } else {
                0.
            };
            write!(text, " {value} 0").unwrap();
        }
    }
    text.push_str("\n[End]\n");
    let nodes = (1..=4)
        .map(|i| PortTerminals {
            positive: NodeId::new(i),
            negative: NodeId::GROUND,
        })
        .collect();
    let four = measured(&text, nodes, Extrapolation::Reject);
    let report = four.properties_at(1.).unwrap();
    assert_eq!(report.passivity, PropertyStatus::Violated);
    assert_eq!(report.reciprocity, PropertyStatus::Violated);
    assert_eq!(report.causality, PropertyStatus::Undetermined);
}
