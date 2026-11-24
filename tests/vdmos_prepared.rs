//! Circuit-coupled VDMOS DC goldens from ngspice 47.
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};
use std::path::Path;

fn run(source: &str) -> SpiceResult<ngspice_rs::analysis::Plot> {
    let card = source.split(".control").next().unwrap_or(source);
    let deck = parse_deck_text(Path::new("vdmos.cir"), &format!("{card}\n.op\n.end\n"));
    let parsed = Parser::new().parse_deck(&deck)?;
    let mut circuit = Circuit::from_netlist(&parsed)?;
    AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, [""; 0])
        .resolve()?
        .run(&mut circuit, &AnalysisContext::default())
}
fn close(name: &str, actual: f64, c: f64) {
    let tolerance = 2e-4_f64.mul_add(c.abs(), 1e-5);
    assert!(
        (actual - c).abs() <= tolerance,
        "{name} Rust {actual:e}, ngspice47 {c:e}, tolerance {tolerance:e}"
    );
}
#[test]
fn circuit_coupled_vdmos_nmos_pmos_series_and_diode() {
    for (name, source, drain, supply_current) in [
        (
            "N",
            include_str!("fixtures/vdmos/n_load.cir"),
            0.06158834,
            -1.19384,
        ),
        (
            "P",
            include_str!("fixtures/vdmos/p_load.cir"),
            -0.126767,
            1.187323,
        ),
        (
            "series",
            include_str!("fixtures/vdmos/n_series.cir"),
            0.3009264,
            -1.16991,
        ),
        (
            "diode",
            include_str!("fixtures/vdmos/n_diode.cir"),
            -0.598882,
            0.01117514,
        ),
    ] {
        let plot = run(source).unwrap_or_else(|error| panic!("{name}: {error}"));
        let v = plot
            .value("v(drain)", 0)
            .unwrap_or_else(|| panic!("{name}: missing Vdrain"));
        let i = plot
            .value("i(vdd)", 0)
            .unwrap_or_else(|| panic!("{name}: missing source current"));
        close(&format!("{name} Vdrain"), v.re, drain);
        close(&format!("{name} supply current"), i.re, supply_current);
    }
}
#[test]
fn unsupported_thermal_network_cannot_be_silently_ignored() {
    let source = "VDMOS thermal refusal\nvdd d 0 1\nvg g 0 4\nm1 d g 0 dut thermal\n.model dut vdmos(vto=3 kp=20 rthjc=1 rthca=2 cthj=10u)\n";
    assert!(run(source).is_err());
}

fn ac(source: &str) -> SpiceResult<ngspice_rs::analysis::Plot> {
    use ngspice_rs::analysis::RunConfig;
    let card = source.split(".control").next().unwrap_or(source);
    let deck = parse_deck_text(
        Path::new("vdmos-ac.cir"),
        &format!("{card}\n.ac lin 1 100k 100k\n.end\n"),
    );
    let parsed = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&parsed)?;
    let request = config.plan_for(&parsed.analyses[0])?;
    let mut circuit = config.circuit(&parsed)?;
    request.run(&mut circuit, &config.context())
}

#[test]
fn native_circuit_coupled_small_signal_n_p_and_series() {
    for (name, source, real, imag) in [
        (
            "N",
            include_str!("fixtures/vdmos/n_ac.cir"),
            -0.0652793,
            7.197459e-5,
        ),
        (
            "P",
            include_str!("fixtures/vdmos/p_ac.cir"),
            -0.143518,
            1.673841e-4,
        ),
        (
            "series",
            include_str!("fixtures/vdmos/n_series_ac.cir"),
            -0.0777147,
            4.281995e-4,
        ),
    ] {
        let plot = ac(source).unwrap_or_else(|error| panic!("AC {name}: {error}"));
        let v = plot
            .value("v(drain)", 0)
            .unwrap_or_else(|| panic!("AC {name}: missing drain"));
        close(&format!("AC {name} real"), v.re, real);
        close(&format!("AC {name} imaginary"), v.im, imag);
    }
}
