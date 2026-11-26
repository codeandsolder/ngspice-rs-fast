//! Prepared PWL resistor segment selection and physical branch identity.
use std::path::Path;

use ngspice_rs::analysis::{Plot, RunConfig};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::SpiceResult;

const KNOTS: &str = "nseg=4 x0=-1 y0=-1 x1=0 y1=2 x2=1 y2=0 x3=2 y3=2 x4=3 y4=3";
const INVERSE_KNOTS: &str = "nseg=4 x0=-1 y0=-1 x1=2 y1=0 x2=0 y2=1 x3=2 y3=2 x4=3 y4=3";

fn simulate(body: &str) -> SpiceResult<Vec<Plot>> {
    let deck = parse_deck_text(
        Path::new("prepared-pwl.cir"),
        &format!("test\n{body}\n.end\n"),
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    netlist
        .analyses
        .iter()
        .map(|analysis| {
            config
                .plan_for(analysis)?
                .run(&mut circuit, &config.context())
        })
        .collect()
}

fn near(plot: &Plot, name: &str, index: usize, expected: f64) {
    let actual = plot
        .value(name, index)
        .unwrap_or_else(|| panic!("missing {name}"))
        .re;
    assert!(
        (actual - expected).abs() < 1e-10,
        "{name} at {index}: {actual:.17e} != {expected:.17e}"
    );
}

#[test]
fn voltage_defined_multiroot_hints_are_advisory() {
    for (segment, root) in [(1, -1. / 3.), (2, 0.5), (3, 1.5), (4, 1.5)] {
        let plots=simulate(&format!("i1 0 a 1\nr1 a 0 curve initial_segment={segment}\n.model curve prepared_vpwlr ({KNOTS})\n.op")).unwrap();
        near(&plots[0], "v(a)", 0, root);
        near(&plots[0], "i(r1)", 0, 1.);
    }
}

#[test]
fn current_defined_multiroot_hints_are_advisory() {
    for (segment, root) in [(1, -1. / 3.), (2, 0.5), (3, 1.5), (4, 1.5)] {
        let plots=simulate(&format!("v1 a 0 1\nr1 a 0 curve initial_segment={segment}\n.model curve prepared_ipwlr ({INVERSE_KNOTS})\n.op")).unwrap();
        near(&plots[0], "i(r1)", 0, root);
        near(&plots[0], "i(v1)", 0, -root);
    }
}

#[test]
fn simultaneous_floating_resistor_hints_preserve_terminal_currents() {
    let plots=simulate(&format!("vcm cm 0 10\ni1 cm a 1\ni2 cm b 1\nr1 a cm curve initial_segment=1\nr2 b cm curve initial_segment=3\n.model curve prepared_vpwlr ({KNOTS})\n.op")).unwrap();
    near(&plots[0], "v(a)", 0, 10. - 1. / 3.);
    near(&plots[0], "v(b)", 0, 11.5);
    near(&plots[0], "i(r1)", 0, 1.);
    near(&plots[0], "i(r2)", 0, 1.);
    near(&plots[0], "i(vcm)", 0, 0.);
}

#[test]
fn prepared_segment_validation_rejects_invalid_hints_and_knots() {
    for hint in ["0", "5", "1.5", "-1"] {
        assert!(simulate(&format!("i1 0 a 1\nr1 a 0 curve initial_segment={hint}\n.model curve prepared_vpwlr ({KNOTS})\n.op")).is_err());
    }
    for knots in [
        "nseg=2 x0=0 y0=0 x1=0 y1=1 x2=2 y2=3",
        "nseg=2 x0=0 y0=0 x1=1 y1=-1 x2=2 y2=3",
        "nseg=2 x0=0 y0=0 x1=1 y1=1 x2=2",
        "nseg=2 x0=0 y0=0 x1=1 y1=1 x2=2 y2=3 foo=1",
    ] {
        assert!(simulate(&format!("i1 0 a 1\nr1 a 0 curve initial_segment=1\n.model curve prepared_vpwlr ({knots})\n.op")).is_err());
    }
}
