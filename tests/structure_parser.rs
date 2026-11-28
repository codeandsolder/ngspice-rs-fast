//! Structure parsing and elaboration preserve hierarchical identity, including
//! definitions and models obtained through source-relative include/library cards.
use ngspice_rs::analysis::RunConfig;
use ngspice_rs::devices::{Circuit, ModelContext, ModelResolver};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use std::path::Path;

#[test]
fn committed_subcircuit_and_source_relative_library_decks_elaborate() {
    let root = ngspice_conformance::fixtures::workspace_root()
        .unwrap()
        .join("conformance");
    let n = Parser::new()
        .parse_file(root.join("netlists/subckt_divider.cir"))
        .unwrap();
    let circuit = Circuit::from_netlist(&n).unwrap();
    assert_eq!(
        circuit
            .devices()
            .iter()
            .map(|device| device.name())
            .collect::<Vec<_>>(),
        ["v1", "r.x1.r1", "r2"]
    );
    // The included divider uses the typical section's 1 kΩ sheet resistor,
    // not the fast section's 500 Ω model. Solve the actual elaborated deck.
    let n = Parser::new()
        .parse_file(root.join("parser/sources/main.cir"))
        .unwrap();
    let config = RunConfig::from_netlist(&n).unwrap();
    let mut circuit = config.circuit(&n).unwrap();
    assert_eq!(
        circuit
            .devices()
            .iter()
            .map(|device| device.name())
            .collect::<Vec<_>>(),
        ["v1", "r.x1.r1", "r2"]
    );
    let plot = config
        .plan_for(&n.analyses[0])
        .unwrap()
        .run(&mut circuit, &config.context())
        .unwrap();
    assert!((plot.column("v(out)").unwrap()[0].re - 5.).abs() < 1e-12);
}

#[test]
fn x_instance_factory_failure_is_atomic_even_without_a_definition() {
    let n = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("x.cir"),
            "title\nr1 old 0 1k\nxnew fresh 0 missing k={base}\n",
        ))
        .unwrap();
    let models = ModelResolver::new(&n.models).unwrap();
    let mut c = Circuit::new();
    c.add_instance(&n.devices[0], &models, &ModelContext::default())
        .unwrap();
    c.finalize().unwrap();
    let before = (
        c.nodes().nodes().to_vec(),
        c.device_count(),
        c.unknown_count(),
        c.branch_rows(0),
    );
    assert!(
        c.add_instance(&n.devices[1], &models, &ModelContext::default())
            .is_err()
    );
    assert_eq!(
        (
            c.nodes().nodes().to_vec(),
            c.device_count(),
            c.unknown_count(),
            c.branch_rows(0)
        ),
        before
    );
}

#[test]
fn forward_models_in_ancestors_and_local_scopes_survive_round_trip() {
    let deck = "scope models
.subckt outer d g s b
r1 d s LOCAL
q1 d g s QROOT
.subckt inner a b
r2 a b local
q2 a b 0 QROOT
m2 a b 0 0 MoS.BaSe
.ends inner
.model local r
.ends outer
.subckt sibling a b
r3 a b rootres
.model local r
.ends sibling
.model rootres r
.model Qroot NPN
.model mos.base.12 NMOS
.end
";
    let parser = Parser::new();
    let n = parser
        .parse_deck(&parse_deck_text(Path::new("scopes.cir"), deck))
        .unwrap();
    let outer = &n.subcircuits[0];
    let inner = &outer.subcircuits[0];
    assert_eq!(outer.devices[0].model.as_deref(), Some("local"));
    assert_eq!(outer.devices[1].nodes, ["d", "g", "s"]);
    assert_eq!(inner.devices[0].model.as_deref(), Some("local"));
    assert_eq!(inner.devices[1].model.as_deref(), Some("qroot"));
    assert_eq!(inner.devices[2].model.as_deref(), Some("mos.base"));
    assert_eq!(
        n.subcircuits[1].devices[0].model.as_deref(),
        Some("rootres")
    );
    let written = ngspice_rs::netlist::write_netlist(&n).unwrap();
    let again = parser
        .parse_deck(&parse_deck_text(Path::new("written.cir"), &written))
        .unwrap();
    assert!(ngspice_rs::netlist::semantic_eq(&n, &again));
    assert_eq!(ngspice_rs::netlist::write_netlist(&again).unwrap(), written);
}

#[test]
fn sibling_and_child_declarations_do_not_leak_into_ancestor_or_other_body() {
    for deck in [
        "title\nr1 a b inner\n.subckt cell a b\n.model inner r\n.ends\n",
        "title\n.subckt first a b\n.model sibling r\n.ends\n.subckt second a b\nr1 a b sibling\n.ends\n",
        "title\n.subckt first a b\nr1 a b sibling\n.ends\n.subckt second a b\n.model sibling r\n.ends\n",
        "title\n.subckt parent a b\nm1 a b 0 0 hidden\n.subckt child a b\n.model hidden.1 NMOS\n.ends\n.ends\n",
    ] {
        assert!(
            Parser::new()
                .parse_deck(&parse_deck_text(Path::new("isolated.cir"), deck))
                .is_err(),
            "{deck}"
        );
    }
}

#[test]
fn repeated_include_occurrences_bind_to_their_own_lexical_ancestors() {
    let scratch = ngspice_conformance::ngspice::Scratch::new("lexical-model-index").unwrap();
    let root = scratch.path().join("root.cir");
    let child = scratch.path().join("child.inc");
    std::fs::write(&child, ".subckt cell a b\nr1 a b local\n.ends cell\n").unwrap();
    let first = "title\n.subckt first a b\n.include child.inc\n.model local r\n.ends first\n";
    let second = ".subckt second a b\n.include child.inc\n.model local r\n.ends second\n.end\n";
    std::fs::write(&root, format!("{first}{second}")).unwrap();
    let n = Parser::new().parse_file(&root).unwrap();
    let a = &n.subcircuits[0].subcircuits[0];
    let b = &n.subcircuits[1].subcircuits[0];
    assert_eq!(a.devices[0].model.as_deref(), Some("local"));
    assert_eq!(b.devices[0].model.as_deref(), Some("local"));
    assert_eq!(a.devices[0].location, b.devices[0].location);
    assert!(std::sync::Arc::ptr_eq(
        &a.cards[0].source,
        &b.cards[0].source
    ));
    assert_ne!(a.cards[0].include_chain, b.cards[0].include_chain);
    // The second occurrence cannot borrow the first parent's declaration,
    // even though the resolver reuses the very same raw card snapshot.
    let second = second.replace(".model local r\n", "");
    std::fs::write(&root, format!("{first}{second}")).unwrap();
    let error = Parser::new().parse_file(&root).unwrap_err();
    assert!(error.to_string().contains("child.inc:2:"), "{error}");
}

#[test]
fn nesting_budget_reports_the_semantic_opening_error() {
    use std::fmt::Write;
    for depth in [64, 65] {
        let mut deck = String::from("title\n");
        for n in 0..depth {
            writeln!(deck, ".subckt nested{n} a b").unwrap();
        }
        deck.push_str("r1 a b rootmodel\n");
        for _ in 0..depth {
            deck.push_str(".ends\n");
        }
        deck.push_str(".model rootmodel r\n.end\n");
        let parsed = Parser::new().parse_deck(&parse_deck_text(Path::new("depth.cir"), &deck));
        if depth == 64 {
            let n = parsed.unwrap();
            let mut sub = &n.subcircuits[0];
            for _ in 1..depth {
                sub = &sub.subcircuits[0];
            }
            assert_eq!(sub.devices[0].model.as_deref(), Some("rootmodel"));
        } else {
            let error = parsed.unwrap_err();
            assert!(
                matches!(error, ngspice_rs::primitives::SpiceError::Parse { location, message } if location.line == 66 && message.contains("nesting limit (64)"))
            );
        }
    }
}
