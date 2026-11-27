//! Native physical network tests. The measured admittances were reported
//! with the independent LTspice 24.0.12 experiments in issue #228.

use std::{f64::consts::TAU, path::Path};

use ngspice_rs::{
    analysis::{AnalysisContext, AnalysisInput, Plot},
    devices::{
        Circuit, ElaborationLimits, IndependentSource, Resistor, StorageKind, Waveform,
        passive_network::{PassiveNetworkCoefficients, PreparedPassiveNetwork},
    },
    netlist::{Parser, source::parse_deck_text},
    primitives::{AnalysisKind, Complex, NodeId, SpiceResult},
};

fn coefficients(kind: StorageKind, value: f64) -> PassiveNetworkCoefficients {
    PassiveNetworkCoefficients {
        kind,
        value,
        series_resistance: 0.,
        series_inductance: 0.,
        series_inductance_shunt: None,
        storage_shunt: None,
        parallel_resistance: None,
        parallel_capacitance: 0.,
        multiplicity: 1.,
        temperature: None,
    }
}

fn circuit() -> SpiceResult<(Circuit, NodeId)> {
    let mut circuit = Circuit::new();
    let drive = circuit.add_node("drive")?;
    let port = circuit.add_node("port")?;
    circuit.add_device(Box::new(IndependentSource::new(
        "vdrive",
        [drive, NodeId::GROUND],
        true,
        0.,
        Complex::real(1.),
        Waveform::Constant(0.),
    )?))?;
    circuit.add_device(Box::new(Resistor::new("rdrive", [drive, port], 0.1)?))?;
    Ok((circuit, port))
}

fn run(circuit: &mut Circuit, kind: AnalysisKind, arguments: &[&str]) -> SpiceResult<Plot> {
    circuit.finalize()?;
    AnalysisInput::with_arguments(kind, arguments.iter().copied())
        .resolve()?
        .run(circuit, &AnalysisContext::default())
}

fn close(got: Complex, want: Complex) {
    let error = (got - want).magnitude();
    assert!(
        error <= 1e-12 + 1e-10 * want.magnitude(),
        "{got:?} != {want:?}; error {error:e}"
    );
}

fn admittance(
    input: PassiveNetworkCoefficients,
    name: &str,
    frequency: &str,
) -> SpiceResult<Complex> {
    let (mut circuit, port) = circuit()?;
    let network = PreparedPassiveNetwork::prepare(input, None)?.unwrap();
    let binding = network.bind(&mut circuit, name, [port, NodeId::GROUND])?;
    assert_eq!(binding.primary, name);
    let plot = run(
        &mut circuit,
        AnalysisKind::Ac,
        &["lin", "1", frequency, frequency],
    )?;
    let supply_current = -plot.value("i(vdrive)", 0).unwrap();
    let terminal_current = plot
        .value(&format!("i({})", binding.terminal_probe), 0)
        .unwrap();
    close(terminal_current, supply_current);
    Ok(terminal_current / plot.value("v(port)", 0).unwrap())
}

#[test]
fn capacitor_series_and_parallel_topology_matches_measured_admittance() -> SpiceResult<()> {
    let base = coefficients(StorageKind::Capacitor, 2e-6);
    let series = PassiveNetworkCoefficients {
        series_resistance: 5.,
        ..base
    };
    let parallel = PassiveNetworkCoefficients {
        parallel_resistance: Some(100.),
        parallel_capacitance: 1e-6,
        ..series
    };
    let complete = PassiveNetworkCoefficients {
        series_inductance: 3e-3,
        series_inductance_shunt: Some(20.),
        ..parallel
    };
    for (input, measured) in [
        (base, Complex::new(0., 0.0634601716025138)),
        (series, Complex::new(0.018294118115039, 0.0576554322280287)),
        (
            parallel,
            Complex::new(0.0282941181150396, 0.0893855180292856),
        ),
        (
            complete,
            Complex::new(0.043496142969925, 0.0480014503017573),
        ),
        (
            PassiveNetworkCoefficients {
                multiplicity: 3.,
                ..complete
            },
            Complex::new(0.043496142969925, 0.0480014503017573) * Complex::real(3.),
        ),
    ] {
        close(admittance(input, "c1", "5050")?, measured);
    }
    Ok(())
}

#[test]
fn winding_parasitics_and_parallel_copies_match_measured_admittance() -> SpiceResult<()> {
    let base = coefficients(StorageKind::Inductor, 3e-3);
    let series = PassiveNetworkCoefficients {
        series_resistance: 5.,
        ..base
    };
    let parallel = PassiveNetworkCoefficients {
        parallel_resistance: Some(100.),
        parallel_capacitance: 1e-6,
        ..series
    };
    for (input, measured) in [
        (base, Complex::new(0., -0.530516476972984)),
        (
            PassiveNetworkCoefficients {
                series_resistance: 0.001,
                ..base
            },
            Complex::new(0.000281447653126163, -0.530516327660367),
        ),
        (series, Complex::new(0.175112594692244, -0.0660158929243445)),
        (
            parallel,
            Complex::new(0.185112594692246, -0.0653875743936265),
        ),
        (
            PassiveNetworkCoefficients {
                multiplicity: 3.,
                ..parallel
            },
            Complex::new(0.185112594692246, -0.0653875743936265) * Complex::real(3.),
        ),
    ] {
        close(admittance(input, "l1", "100")?, measured);
    }
    Ok(())
}

#[test]
fn winding_shunt_and_external_shunt_are_physically_distinct() -> SpiceResult<()> {
    let base = PassiveNetworkCoefficients {
        series_resistance: 5.,
        value: 3e-3,
        ..coefficients(StorageKind::Inductor, 3e-3)
    };
    let z = Complex::new(0., TAU * 100. * base.value);
    let g = Complex::real(0.01);
    let r = Complex::real(base.series_resistance);
    let one = Complex::real(1.);
    let winding = PassiveNetworkCoefficients {
        storage_shunt: Some(100.),
        ..base
    };
    let external = PassiveNetworkCoefficients {
        parallel_resistance: Some(100.),
        ..base
    };
    close(
        admittance(winding, "l1", "100")?,
        one / (r + one / (one / z + g)),
    );
    close(admittance(external, "l1", "100")?, one / (r + z) + g);
    Ok(())
}

#[test]
fn zero_series_and_removed_networks_do_not_create_artificial_resistors() -> SpiceResult<()> {
    let (mut circuit, port) = circuit()?;
    let input = coefficients(StorageKind::Capacitor, 2e-6);
    assert!(
        PreparedPassiveNetwork::prepare(
            PassiveNetworkCoefficients {
                multiplicity: 0.,
                ..input
            },
            None,
        )?
        .is_none()
    );
    let network = PreparedPassiveNetwork::prepare(input, None)?.unwrap();
    let before = circuit.devices().len();
    network.bind(&mut circuit, "c1", [port, NodeId::GROUND])?;
    assert_eq!(circuit.devices().len() - before, 2);
    assert!(circuit.device("rc1#series").is_none());
    assert!(circuit.device("lc1#series").is_none());
    Ok(())
}

#[test]
fn duplicate_names_and_budget_errors_rollback_all_nodes_and_devices() -> SpiceResult<()> {
    let (mut circuit, port) = circuit()?;
    let input = PassiveNetworkCoefficients {
        series_resistance: 5.,
        series_inductance: 3e-3,
        ..coefficients(StorageKind::Capacitor, 2e-6)
    };
    let network = PreparedPassiveNetwork::prepare(input, None)?.unwrap();
    circuit.add_node("c1#passive3")?;
    let before = circuit.nodes().nodes().to_vec();
    let devices = circuit.devices().len();
    assert!(
        network
            .bind(&mut circuit, "c1", [port, NodeId::GROUND])
            .is_err()
    );
    assert_eq!(circuit.nodes().nodes(), before);
    assert_eq!(circuit.devices().len(), devices);
    circuit.add_device(Box::new(Resistor::new(
        "rc2#series",
        [port, NodeId::GROUND],
        1.,
    )?))?;
    let devices = circuit.devices().len();
    assert!(
        network
            .bind(&mut circuit, "c2", [port, NodeId::GROUND])
            .is_err()
    );
    assert_eq!(circuit.nodes().nodes(), before);
    assert_eq!(circuit.devices().len(), devices);
    let mut small = Circuit::with_elaboration_limits(ElaborationLimits {
        devices: 2,
        ..Default::default()
    });
    let node = small.add_node("port")?;
    assert!(
        network
            .bind(&mut small, "c1", [node, NodeId::GROUND])
            .is_err()
    );
    assert_eq!(small.nodes().nodes().len(), 2);
    assert!(small.devices().is_empty());
    Ok(())
}

#[test]
fn noise_matches_equivalent_native_network() -> SpiceResult<()> {
    let deck = Parser::new().parse_deck(&parse_deck_text(
        Path::new("passive-noise.cir"),
        "reference\nvdrive drive 0 dc 0 ac 1\nrdrive drive port 0.1\nr1 port mid 5 rm temp=87\nc1 mid 0 2u\nr2 port 0 100 rm temp=87\nc2 port 0 1u\n.model rm r(tc1=0)\n.end\n",
    ))?;
    let mut reference = Circuit::from_netlist(&deck)?;
    let (mut circuit, port) = circuit()?;
    let input = PassiveNetworkCoefficients {
        series_resistance: 5.,
        parallel_resistance: Some(100.),
        parallel_capacitance: 1e-6,
        temperature: Some(87.),
        ..coefficients(StorageKind::Capacitor, 2e-6)
    };
    PreparedPassiveNetwork::prepare(input, None)?
        .unwrap()
        .bind(&mut circuit, "c1", [port, NodeId::GROUND])?;
    let args = ["v(port)", "vdrive", "dec", "5", "10", "100k"];
    let actual = run(&mut circuit, AnalysisKind::Noise, &args)?;
    let expected = run(&mut reference, AnalysisKind::Noise, &args)?;
    for variable in ["onoise_spectrum", "inoise_spectrum"] {
        for index in 0..expected.point_count() {
            let got = actual.value(variable, index).unwrap().re;
            let want = expected.value(variable, index).unwrap().re;
            assert!(
                (got - want).abs() <= 1e-10 * want.abs(),
                "{variable}: {got} != {want}"
            );
        }
    }
    Ok(())
}

#[test]
fn coupling_uses_winding_identity_not_terminal_probe() -> SpiceResult<()> {
    use ngspice_rs::devices::{ControlReference, MutualInductance};

    let (mut circuit, port) = circuit()?;
    let secondary = circuit.add_node("secondary")?;
    circuit.add_device(Box::new(Resistor::new(
        "rload",
        [secondary, NodeId::GROUND],
        100.,
    )?))?;
    let primary = PreparedPassiveNetwork::prepare(
        PassiveNetworkCoefficients {
            series_resistance: 5.,
            parallel_resistance: Some(100.),
            parallel_capacitance: 1e-6,
            ..coefficients(StorageKind::Inductor, 3e-3)
        },
        None,
    )?
    .unwrap()
    .bind(&mut circuit, "l1", [port, NodeId::GROUND])?;
    PreparedPassiveNetwork::prepare(coefficients(StorageKind::Inductor, 12e-3), None)?
        .unwrap()
        .bind(&mut circuit, "l2", [secondary, NodeId::GROUND])?;
    circuit.add_device(Box::new(MutualInductance::new(
        "k1",
        ["l1", "l2"]
            .map(|name| ControlReference {
                name: name.into(),
                location: None,
            })
            .to_vec(),
        0.8,
    )?))?;
    let plot = run(&mut circuit, AnalysisKind::Ac, &["lin", "1", "100", "100"])?;
    let w = TAU * 100.;
    let v = plot.value("v(port)", 0).unwrap();
    let z2 = Complex::new(100., w * 12e-3);
    let mutual = Complex::new(0., w * 0.8 * (3e-3_f64 * 12e-3).sqrt());
    let winding = v / (Complex::new(5., w * 3e-3) - mutual * mutual / z2);
    close(plot.value("i(l1)", 0).unwrap(), winding);
    close(plot.value("i(l2)", 0).unwrap(), -mutual * winding / z2);
    close(
        plot.value(&format!("i({})", primary.terminal_probe), 0)
            .unwrap(),
        winding + v * Complex::new(0.01, w * 1e-6),
    );
    Ok(())
}

#[test]
fn startup_bias_applies_to_storage_and_not_to_parallel_terminal_paths() -> SpiceResult<()> {
    use ngspice_rs::devices::passive_bias::{PassiveBias, PassiveInitialization};

    let mut circuit = Circuit::new();
    let drive = circuit.add_node("drive")?;
    let port = circuit.add_node("port")?;
    circuit.add_device(Box::new(IndependentSource::new(
        "vdrive",
        [drive, NodeId::GROUND],
        true,
        2.,
        Complex::real(1.),
        Waveform::Constant(2.),
    )?))?;
    circuit.add_device(Box::new(Resistor::new("rdrive", [drive, port], 10.)?))?;
    let initialization = PassiveInitialization::new(
        StorageKind::Capacitor,
        Some(0.75),
        [(
            AnalysisKind::OperatingPoint,
            PassiveBias::Voltage {
                voltage: 0.75,
                conductance: None,
            },
        )],
    )?;
    let binding = PreparedPassiveNetwork::prepare(
        PassiveNetworkCoefficients {
            series_resistance: 0.001,
            ..coefficients(StorageKind::Capacitor, 2e-6)
        },
        Some(initialization),
    )?
    .unwrap()
    .bind(&mut circuit, "c1", [port, NodeId::GROUND])?;
    let plot = run(&mut circuit, AnalysisKind::OperatingPoint, &[])?;
    close(
        plot.value("v(port)", 0).unwrap(),
        Complex::real(0.7501249875012499),
    );
    close(
        plot.value(&format!("i({})", binding.terminal_probe), 0)
            .unwrap(),
        Complex::real(0.124987501249875),
    );
    let ac = run(
        &mut circuit,
        AnalysisKind::Ac,
        &["lin", "1", "1000", "1000"],
    )?;
    let zc = Complex::new(0.001, -1. / (TAU * 1000. * 2e-6));
    close(
        ac.value("v(port)", 0).unwrap(),
        zc / (zc + Complex::real(10.)),
    );
    Ok(())
}

fn voltage_network(
    series_resistance: f64,
    parallel_capacitance: f64,
    load: f64,
    noiseless_load: bool,
) -> SpiceResult<(Circuit, String)> {
    use ngspice_rs::devices::{
        ResistorNoise,
        passive_network::{PreparedVoltageNetwork, VoltageNetworkCoefficients},
        sources::{AcInput, SourceInputs},
    };

    let mut circuit = Circuit::new();
    let out = circuit.add_node("out")?;
    circuit.add_device(Box::new(
        Resistor::new("rload", [out, NodeId::GROUND], load)?.with_noise(ResistorNoise {
            noisy: !noiseless_load,
            ..Default::default()
        }),
    ))?;
    let network = PreparedVoltageNetwork::prepare(
        VoltageNetworkCoefficients {
            series_resistance,
            parallel_capacitance,
            temperature: None,
        },
        SourceInputs {
            dc: Some(1.),
            ac: Some(AcInput {
                magnitude: Some(1.),
                phase: None,
            }),
            waveform: None,
        },
    )?;
    let binding = network.bind(&mut circuit, "v1", [out, NodeId::GROUND])?;
    assert_eq!(binding.excitation, "v1");
    Ok((circuit, binding.terminal_probe))
}

#[test]
fn signed_series_voltage_impedance_preserves_dc_excitation() -> SpiceResult<()> {
    for (r, expected) in [(0., 1.), (2., 0.6), (-1., 1.5)] {
        let (mut circuit, probe) = voltage_network(r, 0., 3., false)?;
        let plot = run(&mut circuit, AnalysisKind::OperatingPoint, &[])?;
        close(plot.value("v(out)", 0).unwrap(), Complex::real(expected));
        close(
            plot.value(&format!("i({probe})"), 0).unwrap(),
            Complex::real(-expected / 3.),
        );
    }
    Ok(())
}

#[test]
fn voltage_current_identities_exclude_internal_shunt_from_terminal_current() -> SpiceResult<()> {
    let (mut circuit, probe) = voltage_network(2., 1e-6, 3., false)?;
    let ac = run(&mut circuit, AnalysisKind::Ac, &["lin", "1", "10k", "10k"])?;
    let output = ac.value("v(out)", 0).unwrap();
    close(
        output,
        Complex::new(0.5966083459066635, -0.0449832095176976),
    );
    let external_current = -output / Complex::real(3.);
    close(
        ac.value(&format!("i({probe})"), 0).unwrap(),
        external_current,
    );
    close(
        ac.value("i(v1)", 0).unwrap(),
        external_current - output * Complex::new(0., TAU * 1e4 * 1e-6),
    );
    Ok(())
}

#[test]
fn voltage_series_impedance_noise_uses_explicit_nonnegative_thermal_law() -> SpiceResult<()> {
    for (r, expected) in [(1000., 2.035687076996364e-9), (-500., 5.757792546071801e-9)] {
        let (mut circuit, _) = voltage_network(r, 0., 1000., true)?;
        let noise = run(
            &mut circuit,
            AnalysisKind::Noise,
            &["v(out)", "v1", "lin", "2", "100", "200"],
        )?;
        for index in 0..noise.point_count() {
            let got = noise.value("onoise_spectrum", index).unwrap().re;
            assert!(
                (got - expected).abs() <= 3e-4 * expected,
                "{got} vs {expected}"
            );
        }
    }
    Ok(())
}
