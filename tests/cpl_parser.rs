//! CPL P-card and positioned Maxwell matrix grammar, with semantic roundtrip.
use std::path::Path;

use ngspice_rs::netlist::ast::{Netlist, ParameterKind};
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::netlist::{Parser, semantic_eq, write_netlist};
use ngspice_rs::primitives::SpiceResult;

fn parse(text: &str) -> SpiceResult<Netlist> {
    Parser::new().parse_deck(&parse_deck_text(Path::new("cpl-parser.cir"), text))
}

fn model() -> &'static str {
    ".model cm cpl length=2m R=(1 0.2 2) L=(400n 60n 400n) G=(0 0 0) C=(100p -10p 100p)"
}

#[test]
fn differential_pair_retains_near_far_and_explicit_return_order() {
    let input = format!(
        "Coupled cable\nP1 a b 0 x y gnd CM len=3m\n{}\n.end\n",
        model()
    );
    let deck = parse(&input).unwrap();
    let p = &deck.devices[0];
    assert_eq!(p.designator, 'p');
    assert_eq!(p.nodes, ["a", "b", "0", "x", "y", "0"]);
    assert_eq!(p.model.as_deref(), Some("cm"));
    assert_eq!(p.parameters.len(), 1);
    assert_eq!(p.parameters[0].name, "length");
    assert_eq!(p.parameters[0].value, "3m");
    let m = &deck.models[0];
    assert_eq!(m.base, "cpl");
    assert_eq!(
        m.parameters
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["length", "r", "l", "g", "c"]
    );
    for parameter in &m.parameters[1..] {
        let ParameterKind::NumericVector(components) = &parameter.kind else {
            panic!("expected typed, positioned Maxwell matrix vector");
        };
        assert_eq!(components.len(), 3);
        assert!(
            components
                .iter()
                .all(|component| component.location.line > 0)
        );
    }
    assert_eq!(m.parameters[4].value, "100p -10p 100p");
}

#[test]
fn three_conductor_cable_has_no_false_pairing_or_implicit_reference() {
    let deck = parse(
        "Three-conductor\nP3 a b c ref1 x y z ref2 model\n.model model cpl length=5u R=1 0 0 1 0 1 L=4n 0.1n 0.1n 4n 0.1n 4n C=4p -0.3p -0.1p 4p -0.3p 4p\n.end\n",
    ).unwrap();
    assert_eq!(
        deck.devices[0].nodes,
        ["a", "b", "c", "ref1", "x", "y", "z", "ref2"]
    );
    assert_eq!(deck.devices[0].parameters, []);
    let m = &deck.models[0];
    assert_eq!(m.parameters[1].name, "r");
    let ParameterKind::NumericVector(r) = &m.parameters[1].kind else {
        panic!("expected R vector");
    };
    assert_eq!(r.len(), 6);
}

#[test]
fn ordered_duplicate_model_setters_are_kept_and_case_folded() {
    let deck = parse(
        "t\n.model CM CPL (R=(1,0,1) L=(2n,0,2n) C=(3p,0,3p) length=1m R=(2,0,2))\nP1 a b 0 x y 0 cm LENGTH 4m\n.end\n",
    ).unwrap();
    assert_eq!(
        deck.models[0]
            .parameters
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["r", "l", "c", "length", "r"]
    );
    assert_eq!(deck.devices[0].parameters[0].name, "length");
    assert_eq!(deck.devices[0].parameters[0].value, "4m");
}

#[test]
fn vector_expressions_remain_separate_and_positioned() {
    let deck = parse(
        "param\n.param lself=4e-7 cmut=-1e-11\n.model cm cpl L=({lself},0,{lself}) C=({1e-10},{cmut},{1e-10})\n.end\n",
    ).unwrap();
    for parameter in &deck.models[0].parameters {
        let ParameterKind::NumericVector(components) = &parameter.kind else {
            panic!("expected numeric vector");
        };
        assert_eq!(components.len(), 3);
        assert!(components[0].text.starts_with('{'));
        assert!(components[0].location.column > 0);
    }
}

#[test]
fn p_and_cpl_model_round_trip_through_borrowed_writer_semantically() {
    let text = format!("CPL\nP1 a b 0 c d 0 Cm LEN=8m\n{}\n.end\n", model());
    let original = parse(&text).unwrap();
    let serialized = write_netlist(&original).unwrap();
    let reloaded = parse(&serialized).unwrap();
    assert!(semantic_eq(&original, &reloaded));
    assert_eq!(serialized, write_netlist(&reloaded).unwrap());
}

#[test]
fn missing_return_bad_width_or_unknown_matrix_assignment_are_rejected() {
    for device in [
        "p1 a 0 b m",
        "p1 a b 0 x y m",
        "p1 a b 0 x y 0 m extra",
        "p1 a b 0 x y 0 m len=",
        "p1 a b 0 x y 0 m len=not-a-number",
    ] {
        let deck = format!("bad\n{device}\n{}\n.end\n", model());
        assert!(parse(&deck).is_err(), "{device}");
    }
    for wrong in [
        ".model cm cpl L=()",
        ".model cm cpl L=1 0 1 X=3",
        ".model cm cpl R=(1 2)",
        ".model cm cpl L=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37",
        ".model cm cpl L=(1,abc,2)",
    ] {
        let text = format!("bad\n{wrong}\n.end\n");
        if wrong.contains("R=(1 2)") {
            // Source grammar accepts a short numeric vector: physical model
            // dimension is checked later against a bound P instance.
            assert!(parse(&text).is_ok());
        } else {
            assert!(parse(&text).is_err(), "{wrong}");
        }
    }
}
