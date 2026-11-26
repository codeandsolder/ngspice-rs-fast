//! GitHub #8: numeric source syntax only, not time evaluation/solver parity.
use std::path::Path;

use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::{Netlist, ParameterKind, SourceWaveform};
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, SpiceError};

fn parse(body: &str) -> Result<Netlist, SpiceError> {
    Parser::new().parse_deck(&parse_deck_text(
        Path::new("wave.cir"),
        &format!("Title\n{body}"),
    ))
}

#[test]
fn transient_fixture_has_positioned_pulse_and_unvalidated_request() {
    let path = ngspice_conformance::fixtures::workspace_root()
        .unwrap()
        .join("conformance/netlists/rc_transient.cir");
    let netlist = Parser::new().parse_file(path).unwrap();
    assert_eq!(netlist.devices.len(), 3);
    let source = netlist.device("v1").unwrap();
    assert_eq!(source.nodes, ["in", "0"]);
    let setter = &source.parameters[0];
    assert_eq!(setter.name, "pulse");
    assert_eq!(setter.location.line, 2);
    let ParameterKind::Waveform(SourceWaveform::Pulse(pulse)) = &setter.kind else {
        panic!("PULSE AST")
    };
    assert_eq!(pulse.initial.source.text, "0");
    assert_eq!(pulse.pulsed.source.text, "5");
    assert!(pulse.period.is_some());
    assert_eq!(netlist.analyses[0].kind, AnalysisKind::Transient);
    assert_eq!(netlist.analyses[0].arguments, ["0.5u", "5u"]);
}

#[test]
fn all_pulse_optional_fields_remain_omitted_or_positioned() {
    let fields = ["0", "5V", "1n", "2n", "3n", "4u", "5u"];
    for count in 2..=7 {
        for spelling in [
            fields[..count].join(" "),
            format!("({})", fields[..count].join(", ")),
        ] {
            let netlist = parse(&format!("V1 a 0 PULSE {spelling}\n")).unwrap();
            let setter = &netlist.devices[0].parameters[0];
            assert_eq!(setter.value, spelling);
            let ParameterKind::Waveform(SourceWaveform::Pulse(pulse)) = &setter.kind else {
                panic!("PULSE AST")
            };
            assert_eq!(pulse.initial.source.text, fields[0]);
            assert_eq!(pulse.pulsed.source.text, fields[1]);
            let timing = [
                &pulse.delay,
                &pulse.rise,
                &pulse.fall,
                &pulse.width,
                &pulse.period,
            ];
            for (index, field) in timing.iter().enumerate() {
                assert_eq!(
                    field.as_ref().map(|v| v.source.text.as_str()),
                    fields.get(index + 2).filter(|_| index + 2 < count).copied()
                );
            }
        }
    }
}

#[test]
fn mixed_source_setters_keep_duplicates_and_leading_dc_last() {
    let netlist = parse("V1 a 0 7 AC PULSE(0 5) DC=2 PWL(0,1,2u,3) AC=2 90 PULSE(1 4) dc 3\nI1 a 0 DC 2m PWL=0 1m 2u -3m AC\n").unwrap();
    let p = &netlist.devices[0].parameters;
    assert_eq!(
        p.iter()
            .map(|p| (p.name.as_str(), p.value.as_str()))
            .collect::<Vec<_>>(),
        [
            ("acmag", "1"),
            ("acphase", "0"),
            ("pulse", "(0 5)"),
            ("dc", "2"),
            ("pwl", "(0,1,2u,3)"),
            ("acmag", "2"),
            ("acphase", "90"),
            ("pulse", "(1 4)"),
            ("dc", "3"),
            ("dc", "7")
        ]
    );
    assert_eq!(p[9].kind, ParameterKind::Scalar);
    let p = &netlist.devices[1].parameters;
    assert_eq!(
        p.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
        ["dc", "pwl", "acmag", "acphase"]
    );
    let ParameterKind::Waveform(SourceWaveform::Pwl(points)) = &p[1].kind else {
        panic!("PWL AST")
    };
    assert_eq!(points.len(), 2);
    assert_eq!(points[0].value.source.text, "1m");
    assert_eq!(points[1].time.source.text, "2u");
    assert_eq!(points[1].value.source.text, "-3m");
}

#[test]
fn syntax_does_not_sort_validate_or_repair_pwl_times() {
    let netlist = parse("I1 a 0 pwl(-1 2 0 3 0 4 -2 5)\nV1 a 0 pwl(0 1)\n").unwrap();
    let ParameterKind::Waveform(SourceWaveform::Pwl(points)) =
        &netlist.devices[0].parameters[0].kind
    else {
        panic!("PWL AST")
    };
    assert_eq!(
        points
            .iter()
            .map(|p| p.time.source.text.as_str())
            .collect::<Vec<_>>(),
        ["-1", "0", "0", "-2"]
    );
    assert_eq!(netlist.devices[1].parameters.len(), 1);
}

#[test]
fn continuations_and_unicode_nodes_keep_byte_locations_and_raw_spelling() {
    let netlist = parse("V1 α GND PULSE(0 5V\n+ 1n, 2n)\n").unwrap();
    let setter = &netlist.devices[0].parameters[0];
    assert_eq!(setter.location.line, 2);
    assert_eq!(setter.location.column, 11);
    assert_eq!(setter.value, "(0 5V 1n, 2n)");
    let ParameterKind::Waveform(SourceWaveform::Pulse(pulse)) = &setter.kind else {
        panic!("PULSE AST")
    };
    assert_eq!(pulse.initial.source.location.column, 17);
    assert_eq!(pulse.delay.as_ref().unwrap().source.location.line, 3);
    assert_eq!(pulse.delay.as_ref().unwrap().source.location.column, 3);
    assert_eq!(netlist.devices[0].nodes, ["α", "0"]);
    for (body, bad) in [
        ("V1 α 0 pulse(0 1e999)", "1e999"),
        ("V1 α 0 pulse(0 1) dc=1e999", "1e999"),
        ("I1 α 0 pwl(0 1 2 1e999)", "1e999"),
        ("V1 α 0 pulse(0 1,,2)", ",,2"),
    ] {
        let expected = body.find(bad).unwrap() + 1 + usize::from(bad == ",,2");
        assert!(
            matches!(parse(body), Err(SpiceError::Parse { location, .. })
            if location.column == expected),
            "{body}"
        );
    }
}

#[test]
fn malformed_waveforms_are_committed_parse_errors() {
    for body in [
        "pulse",
        "pulse()",
        "pulse(0)",
        "pulse 0",
        "pulse=",
        "pulse==0 1",
        "pulse(0 1",
        "pulse(0 1))",
        "pulse((0 1))",
        "pulse(0 1,)",
        "pulse(,0 1)",
        "pulse(0,,1)",
        "pulse(0 1 2 3 4 5 6 7 8)",
        "pulse 0 1 2 3 4 5 6 7 8",
        "sin(0)",
        "sin(0 1 2 3 4 5 6)",
        "sine 0 1 2 3 4 5 6",
        "exp(0 1 2 3 4 5 6)",
        "sffm(0 1 2 3 4 5 6 7 8)",
        "am(0 1 2 3 4 5 6 7 8)",
        "am(0 1e999)",
        "sin(0 1",
        "exp(,0 1)",
        "pwl(0 1) r",
        "pwl(0 1) r=",
        "pwl(0 1) td=1e999",
        "pulse(0 1e999)",
        "pulse 0 1e999",
        "pulse(0 1 1e999)",
        "pwl(0 1 2)",
        "pwl 0 1 2",
        "pwl(0)",
        "pwl(0,)",
        "pwl(0 1 2 1e999)",
        "pwl 0 1 2 1e999",
        "pwl(0 1) =2",
    ] {
        let error = parse(&format!("V1 a 0 {body}\n")).unwrap_err();
        assert!(
            matches!(error, SpiceError::Parse{ ref location,..} if location.line==2 && location.path()==Path::new("wave.cir")),
            "{body}: {error}"
        );
    }
}

#[test]
fn unsupported_waveform_extensions_never_partially_succeed() {
    for body in [
        "pulse(0 \"1\")",
        "pwl file=\"values.txt\"",
        "external",
        "pulse(0 1) junk",
    ] {
        let error = parse(&format!("I1 a 0 {body}\n")).unwrap_err();
        assert!(error.is_not_yet_ported(), "{body}: {error}");
        assert!(error.to_string().contains("inp2i.c"), "{error}");
    }
}

#[test]
fn vector_work_limit_is_explicit() {
    let fields = "0 1 ".repeat(2048);
    let netlist = parse(&format!("V1 a 0 pwl({fields})")).unwrap();
    let ParameterKind::Waveform(SourceWaveform::Pwl(points)) =
        &netlist.devices[0].parameters[0].kind
    else {
        panic!("PWL AST")
    };
    assert_eq!(points.len(), 2048);
    let fields = "0 1 ".repeat(2049);
    assert!(matches!(
        parse(&format!("V1 a 0 pwl({fields})")),
        Err(SpiceError::Parse { .. })
    ));
}

#[test]
fn source_functions_keep_positioned_fields_and_omissions() {
    use ngspice_rs::netlist::ast::SourceFunction;
    for (card, function, keyword, count) in [
        (
            "V1 a 0 SIN(0.5 2 1k 0.1m 20 30)",
            SourceFunction::Sin,
            "sin",
            6,
        ),
        ("I1 a 0 sine 0 1m", SourceFunction::Sin, "sine", 2),
        (
            "V1 a 0 exp(1, 3, 0.2m, 0.1m, 0.6m, 0.2m)",
            SourceFunction::Exp,
            "exp",
            6,
        ),
        ("V1 a 0 exp=(1 3 0.2m)", SourceFunction::Exp, "exp", 3),
        (
            "I1 a 0 SFFM 0 1m 10k 2 1k 0 90 45",
            SourceFunction::Sffm,
            "sffm",
            8,
        ),
        ("V1 a 0 am(0.1 1 0.5 1k)", SourceFunction::Am, "am", 4),
    ] {
        let netlist = parse(card).unwrap();
        let setter = &netlist.devices[0].parameters[0];
        assert_eq!(setter.name, keyword, "{card}");
        assert_eq!(
            setter.location.column,
            card.to_ascii_lowercase().find(keyword).unwrap() + 1
        );
        let ParameterKind::Waveform(SourceWaveform::Function(wave)) = &setter.kind else {
            panic!("{card}: function AST")
        };
        assert_eq!(wave.function, function, "{card}");
        assert_eq!(wave.values.len(), count, "{card}");
        assert!(count <= function.fields().len());
        // Every field keeps its spelling and byte column.
        for value in &wave.values {
            let column = value.source.location.column;
            assert_eq!(
                &card[column - 1..column - 1 + value.source.text.len()],
                value.source.text
            );
        }
    }
}

#[test]
fn pulse_count_and_pwl_options_are_ordered_positioned_setters() {
    let netlist = parse("V1 a 0 PULSE(0 1 0 1n 1n 1u 2u 3)\n").unwrap();
    let ParameterKind::Waveform(SourceWaveform::Pulse(pulse)) =
        &netlist.devices[0].parameters[0].kind
    else {
        panic!("PULSE AST")
    };
    let count = pulse.count.as_ref().unwrap();
    assert_eq!(
        (count.source.text.as_str(), count.source.location.column),
        ("3", 32)
    );
    // Seven fields leave the count omitted.
    let netlist = parse("V1 a 0 PULSE(0 1 0 1n 1n 1u 2u)\n").unwrap();
    let ParameterKind::Waveform(SourceWaveform::Pulse(pulse)) =
        &netlist.devices[0].parameters[0].kind
    else {
        panic!("PULSE AST")
    };
    assert!(pulse.count.is_none());
    // r=/td= are scalar setters in card order (C applies them in order), with
    // or without '=' (INPgetTok gobbles it), including expressions.
    let netlist = parse("I1 a 0 td=1u PWL(0 0 1u 1) R 0 td {d}\n").unwrap();
    let p = &netlist.devices[0].parameters;
    assert_eq!(
        p.iter()
            .map(|p| (p.name.as_str(), p.value.as_str(), p.location.column))
            .collect::<Vec<_>>(),
        [
            ("td", "1u", 8),
            ("pwl", "(0 0 1u 1)", 14),
            ("r", "0", 28),
            ("td", "{d}", 32)
        ]
    );
    assert_eq!(p[0].kind, ParameterKind::Scalar);
    assert!(matches!(p[3].kind, ParameterKind::Expression(_)));
}

#[test]
fn expression_coefficients_preserve_physical_positions_and_typed_sites() {
    use ngspice_rs::netlist::elaborate::{SiteKind, literalize};
    let source = parse(
        ".param a=2 edge=0.125\nV1 α 0 PWL(0 0\n+ {edge}, 'a*2' {2*edge} 0) r={0} td={-edge/2}",
    )
    .unwrap();
    let ParameterKind::Waveform(SourceWaveform::Pwl(points)) =
        &source.devices[0].parameters[0].kind
    else {
        panic!("expected PWL");
    };
    assert_eq!(points[1].time.source.location.line, 4);
    assert_eq!(points[1].time.source.location.column, 3);
    assert_eq!(points[1].value.source.text, "'a*2'");
    assert!(points[1].value.expression.as_ref().unwrap().quoted);
    let elaborated = literalize(&source).unwrap();
    let sites = &elaborated.sites;
    for (index, expected, spelling) in [
        (2, 0.125, "{edge}"),
        (3, 4., "'a*2'"),
        (4, 0.25, "{2*edge}"),
    ] {
        let site = sites
            .iter()
            .find(|site| {
                site.kind
                    == (SiteKind::DeviceVectorEntry {
                        device: 0,
                        parameter: 0,
                        entry: index,
                    })
            })
            .unwrap();
        assert_eq!(site.original, spelling);
        assert_eq!(site.read_back, expected);
        assert_eq!(site.location.line, 4);
    }
    let ParameterKind::Waveform(SourceWaveform::Pwl(resolved)) =
        &elaborated.netlist.devices[0].parameters[0].kind
    else {
        panic!("expected resolved PWL");
    };
    assert!(
        resolved
            .iter()
            .all(|p| p.time.expression.is_none() && p.value.expression.is_none())
    );
    assert_eq!(
        resolved[1].time.source.location,
        points[1].time.source.location
    );
    assert!(points[1].time.expression.is_some());
}

#[test]
fn malformed_and_unresolvable_expression_fields_fail_at_their_own_locations() {
    use ngspice_rs::netlist::elaborate::literalize;
    for card in [
        "pulse(0 {1+})",
        "pwl(0 1 {2})",
        "sin(0 1,)",
        "exp(0 1 {2} 3 4 5 {6})",
        "pulse 0 1 {2} 3 4 5 6 7 {8}",
    ] {
        assert!(
            matches!(
                parse(&format!("V1 a 0 {card}")),
                Err(SpiceError::Parse { .. })
            ),
            "{card}"
        );
    }
    for (field, message) in [
        ("{missing}", "missing"),
        ("{1/0}", "resolving sin vector coefficient"),
    ] {
        let source = parse(&format!("V1 α 0 SIN(0 1\n+ {field})")).unwrap();
        let error = literalize(&source).unwrap_err();
        assert!(
            matches!(&error, SpiceError::Parse { location, .. } if location.line == 3 && location.column >= 3),
            "{error}"
        );
        assert!(error.to_string().contains(message), "{error}");
    }
}
