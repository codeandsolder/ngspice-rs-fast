//! T/O grammar parity: instance-specific syntax and one ordered setter parser.

use std::path::Path;

use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::{Netlist, ParameterKind};
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{SpiceError, SpiceResult};

fn parse(body: &str) -> SpiceResult<Netlist> {
    Parser::new().parse_deck(&parse_deck_text(
        Path::new("line-setters.cir"),
        &format!("line setter cases\n{body}\n.model m ltra l=250n c=100p len=1\n.end\n"),
    ))
}

fn names_and_values(body: &str) -> Vec<(String, String)> {
    let parsed = parse(body).unwrap();
    parsed.devices[0]
        .parameters
        .iter()
        .map(|assignment| (assignment.name.clone(), assignment.value.clone()))
        .collect()
}

#[test]
fn both_line_families_keep_case_optional_equals_and_duplicate_order() {
    for (prefix, setters, expected) in [
        (
            "t1 a 0 b 0",
            "Z0 = 25 zO 50 TD=1n v1 {1+2} ic=3,4 V1=9",
            vec![
                ("z0", "25"),
                ("zo", "50"),
                ("td", "1n"),
                ("v1", "{1+2}"),
                ("ic", "3,4"),
                ("v1", "9"),
            ],
        ),
        (
            "o1 a 0 b 0 M",
            "V1=1 i2 5 ic=3,4 v1 7",
            vec![("v1", "1"), ("i2", "5"), ("ic", "3,4"), ("v1", "7")],
        ),
    ] {
        let card = format!("{prefix} {setters}");
        let parsed = parse(&card).unwrap();
        let device = &parsed.devices[0];
        assert_eq!(device.nodes, ["a", "0", "b", "0"], "{prefix}");
        assert_eq!(
            device.model.as_deref(),
            if prefix.starts_with('o') {
                Some("m")
            } else {
                None
            },
        );
        assert_eq!(
            names_and_values(&card),
            expected
                .into_iter()
                .map(|(name, value)| (name.to_owned(), value.to_owned()))
                .collect::<Vec<_>>(),
            "{card}"
        );
    }
}

#[test]
fn both_line_families_share_ordered_ic_component_projection_and_locations() {
    for prefix in ["t1 a 0 b 0 z0=50", "o1 a 0 b 0 m"] {
        for arity in 1..=4 {
            let values = ["11", "22", "33", "44"]
                .into_iter()
                .take(arity)
                .collect::<Vec<_>>();
            let card = format!("{prefix} IC = {}", values.join(","));
            let parsed = parse(&card).unwrap();
            let assignment = parsed.devices[0].parameters.last().unwrap();
            assert_eq!(assignment.name, "ic");
            assert_eq!(assignment.location.line, 2);
            assert_eq!(assignment.location.column, card.find("IC").unwrap() + 1);
            let ParameterKind::InitialConditions(components) = &assignment.kind else {
                panic!("IC was not parsed as an ordered vector: {card}");
            };
            assert_eq!(components.len(), arity);
            for ((component, expected_name), expected_value) in
                components.iter().zip(["v1", "i1", "v2", "i2"]).zip(&values)
            {
                assert_eq!(&component.name, expected_name);
                assert_eq!(&component.value.text, expected_value);
                assert_eq!(component.value.location.line, 2);
                assert!(component.value.location.column > assignment.location.column);
            }
        }
    }
}

#[test]
fn setters_on_folded_physical_lines_preserve_source_locations() {
    for prefix in ["t1 a 0 b 0 z0=50", "o1 a 0 b 0 m"] {
        let parsed = parse(&format!("{prefix}\n+ v1=2 ic=3,4")).unwrap();
        let parameters = &parsed.devices[0].parameters;
        let last = parameters.last().unwrap();
        assert_eq!(last.name, "ic");
        assert_eq!(last.location.line, 3, "{prefix}");
        let ParameterKind::InitialConditions(values) = &last.kind else {
            panic!("expected IC vector");
        };
        assert_eq!(values[0].value.location.line, 3);
        assert_eq!(values[1].value.location.line, 3);
    }
}

#[test]
fn line_specific_errors_keep_context_and_do_not_backtrack_after_setter() {
    for (card, expected) in [
        ("t1 a 0 b 0 55 z0=50", "leading value"),
        ("o1 a 0 b 0 m 55 v1=1", "leading value"),
        ("o1 a 0 b 0", "lossy-line model name"),
        (
            "t1 a 0 b 0 z0=50 unknown=1",
            "unknown transmission-line parameter",
        ),
        ("o1 a 0 b 0 m z0=50", "unknown lossy-line parameter"),
        ("t1 a 0 b 0 z0=50 ic=1,2,3,4,5", "too many"),
        ("o1 a 0 b 0 m ic=1,2,3,4,5", "too many"),
        ("t1 a 0 b 0 z0=", "finite numeric literal"),
        ("o1 a 0 b 0 m ic=", "finite numeric literal"),
    ] {
        let error: SpiceError = parse(card).unwrap_err();
        let text = error.to_string();
        assert!(text.contains(expected), "{card}: {text}");
    }
}
