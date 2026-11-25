//! Native passive expression laws through the real parser and solver.
use std::path::Path;

use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, semantic_diff, source::parse_deck_text, write_netlist};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};

fn run(body: &str, kind: AnalysisKind, args: &[&str]) -> SpiceResult<Plot> {
    let deck = parse_deck_text(
        Path::new("passive.cir"),
        &format!("passive\n{body}\n.end\n"),
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let mut circuit = Circuit::from_netlist(&netlist)?;
    AnalysisInput::with_arguments(kind, args.iter().copied())
        .resolve()?
        .run(&mut circuit, &AnalysisContext::default())
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:e} != {expected:e}"
    );
}

#[test]
fn resistance_uses_control_voltage_and_actual_terminal_current() -> SpiceResult<()> {
    let plot = run(
        "vin a 0 2\nvcontrol ctrl 0 1\nr1 a 0 r={1000*(1+v(ctrl))}",
        AnalysisKind::OperatingPoint,
        &[],
    )?;
    close(plot.value("i(vin)", 0).unwrap().re, -1e-3, 1e-10);
    Ok(())
}

#[test]
fn capacitance_and_charge_have_distinct_ac_derivatives() -> SpiceResult<()> {
    let plot = run(
        "vc c 0 dc 1 ac 1\nvq q 0 dc 1 ac 1\nc1 c 0 c={1u*(1+v(c))}\nc2 q 0 q={1u*(v(q)+v(q)^2)}",
        AnalysisKind::Ac,
        &["lin", "1", "1000", "1000"],
    )?;
    let omega = 2. * std::f64::consts::PI * 1000.;
    close(plot.value("i(vc)", 0).unwrap().im, -omega * 2e-6, 1e-10);
    close(plot.value("i(vq)", 0).unwrap().im, -omega * 3e-6, 1e-10);
    Ok(())
}

#[test]
fn capacitance_and_charge_distinguish_controlled_transient_storage() -> SpiceResult<()> {
    let plot = run(
        "vc c 0 PWL(0 1 1m 2)\nvq q 0 PWL(0 1 1m 2)\nc1 c 0 c={1u*(1+v(c))}\nc2 q 0 q={1u*(v(q)+v(q)^2)}",
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u"],
    )?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        if !(20e-6..=980e-6).contains(&t) {
            continue;
        }
        let voltage = 1. + t * 1000.;
        close(
            plot.value("i(vc)", index).unwrap().re,
            -(1. + voltage) * 1e-3,
            1e-5,
        );
        close(
            plot.value("i(vq)", index).unwrap().re,
            -(1. + 2. * voltage) * 1e-3,
            1e-5,
        );
    }
    Ok(())
}

#[test]
fn variable_inductance_uses_current_derivative_not_flux_product() -> SpiceResult<()> {
    let plot = run(
        "iin 0 a PWL(0 0 1m 1)\nl1 a 0 l={1m*(1+i(l1))}",
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u"],
    )?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        if !(20e-6..=980e-6).contains(&t) {
            continue;
        }
        close(plot.value("v(a)", index).unwrap().re, 1. + t * 1000., 1e-3);
    }
    Ok(())
}

#[test]
fn round_trip_preserves_static_and_runtime_value_categories() -> SpiceResult<()> {
    let parse =
        |text: &str| Parser::new().parse_deck(&parse_deck_text(Path::new("roundtrip.cir"), text));
    let original = parse(
        "title\n.param r=1k\nr1 a 0 {r}\nr2 a 0 {r*(1+v(a))}\nc1 a 0 q={1u*v(a)^2}\nl1 a 0 l={1m*(1+time)}\n.end\n",
    )?;
    let text = write_netlist(&original)?;
    let restored = parse(&text)?;
    assert!(semantic_diff(&original, &restored).is_none());
    Ok(())
}
