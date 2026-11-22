//! URC uniform distributed RC lines (GitHub #85): the `inp2u.c` grammar, the
//! `urc` model schema and `urcsetup.c`'s expansion into generated R/C/D
//! elements with C's names, through the production parser, circuit and
//! analyses. The committed C goldens `m10_urc_tran`, `m10_urc_ac` and
//! `m10_urc_diode_tran` are compared by `cargo xtask golden verify`.
use std::path::Path;

use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot};
use ngspice_rs::devices::{Circuit, ModelContext};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, Real, SpiceError, SpiceResult};

fn parse(body: &str) -> SpiceResult<ngspice_rs::netlist::ast::Netlist> {
    Parser::new().parse_deck(&parse_deck_text(
        Path::new("urc.cir"),
        &format!("urc\n{body}\n.end\n"),
    ))
}

fn circuit(body: &str) -> SpiceResult<Circuit> {
    Circuit::from_netlist(&parse(body)?)
}

fn op(body: &str) -> SpiceResult<Plot> {
    AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
        .resolve()
        .and_then(|plan| plan.run(&mut circuit(body)?, &AnalysisContext::default()))
}

fn names(c: &Circuit) -> Vec<String> {
    c.devices().iter().map(|d| d.name().to_owned()).collect()
}

fn parameter(c: &Circuit, device: &str, key: &str) -> Real {
    c.device(device)
        .unwrap_or_else(|| panic!("no {device}"))
        .observation_parameter(key, &ModelContext::default())
        .unwrap()
        .unwrap_or_else(|| panic!("{device} has no {key}"))
}

fn message(error: &SpiceError) -> String {
    error.to_string()
}

#[test]
fn the_grammar_keeps_three_terminals_the_model_and_ordered_setters() {
    let netlist = parse("u1 A B Gnd LINE l 1m n=2.6 l=2m\n.model line urc urc k=2").unwrap();
    let u = &netlist.devices[0];
    assert_eq!(u.designator, 'u');
    assert_eq!(u.nodes, ["a", "b", "0"]);
    assert_eq!(u.model.as_deref(), Some("line"));
    let setters: Vec<_> = u
        .parameters
        .iter()
        .map(|p| (p.name.as_str(), p.value.as_str()))
        .collect();
    assert_eq!(setters, [("l", "1m"), ("n", "2.6"), ("l", "2m")]);
    assert_eq!(netlist.models[0].base, "urc");
}

#[test]
fn malformed_cards_are_refused() {
    // No model, a leading value, an unknown setter, too few terminals.
    let error = parse("u1 a b 0").unwrap_err();
    assert!(message(&error).contains("model"), "{error}");
    let error = parse("u1 a b 0 m 5 l=1m\n.model m urc").unwrap_err();
    assert!(matches!(error, SpiceError::Unsupported { .. }), "{error}");
    let error = parse("u1 a b 0 m l=1m k=2\n.model m urc").unwrap_err();
    assert!(
        message(&error).contains("unknown URC parameter 'k'"),
        "{error}"
    );
    let error = parse("u1 a b m\n.model m urc").unwrap_err();
    assert!(message(&error).contains("model"), "{error}");
}

#[test]
fn the_expansion_follows_urcsetup_names_and_order() {
    // Short line: wnorm < 35, so C's minimum of three sections.
    let c = circuit("v1 in 0 1\nu1 in out ref m l=1e-5\nr1 out 0 1k\nvr ref 0 0\n.model m urc")
        .unwrap();
    assert_eq!(
        names(&c),
        [
            "v1", "u1", "u1#rlo1", "u1#rhi1", "u1#clo1", "u1#chi1", "u1#rlo2", "u1#rhi2",
            "u1#clo2", "u1#chi2", "u1#rlo3", "u1#rhi3", "u1#clo3", "r1", "vr",
        ]
    );
    let nodes: Vec<_> = c.nodes().nodes().iter().map(|n| n.name.clone()).collect();
    for internal in ["u1#hi1", "u1#lo1", "u1#hi2", "u1#lo2", "u1#hi3"] {
        assert!(nodes.contains(&internal.to_owned()), "{nodes:?}");
    }
    assert!(!nodes.iter().any(|n| n == "u1#lo3"), "{nodes:?}");
    // The instance itself stays, without terminals or load, as in C.
    let line = c.device("u1").unwrap();
    assert_eq!(line.designator(), 'u');
    assert!(line.terminals().is_empty());
    assert_eq!(parameter(&c, "u1", "l"), 1e-5);
    assert_eq!(parameter(&c, "u1", "n"), 3.);
    // Topology: rlo1 from the first terminal, the last section meets in hi3,
    // the hi side returns to the second terminal; shunts go to the reference.
    let terminal_names = |device: &str| -> Vec<String> {
        c.device(device)
            .unwrap()
            .terminals()
            .iter()
            .map(|id| c.nodes().node(*id).unwrap().name.clone())
            .collect()
    };
    assert_eq!(terminal_names("u1#rlo1"), ["in", "u1#lo1"]);
    assert_eq!(terminal_names("u1#rhi1"), ["u1#hi1", "out"]);
    assert_eq!(terminal_names("u1#rlo3"), ["u1#lo2", "u1#hi3"]);
    assert_eq!(terminal_names("u1#rhi3"), ["u1#hi3", "u1#hi2"]);
    assert_eq!(terminal_names("u1#clo3"), ["u1#hi3", "ref"]);
    assert_eq!(terminal_names("u1#chi2"), ["u1#hi2", "ref"]);
    // Values: r0 = 10 ohms, c0 = 1e-17 F, p = 1.5, geometric by section.
    let (p, r0, c0): (Real, Real, Real) = (1.5, 1e-5 * 1000., 1e-5 * 1e-12);
    let r1 = (r0 * (p - 1.)) / ((2. * p.powf(3.)) - 2.);
    let c1 = (c0 * (p - 1.)) / (p.powf(2.) * (p + 1.) - 2.);
    assert_eq!(parameter(&c, "u1#rlo1", "resistance"), r1);
    assert_eq!(parameter(&c, "u1#rhi2", "resistance"), p * r1);
    assert_eq!(parameter(&c, "u1#rlo3", "resistance"), p * p * r1);
    assert_eq!(parameter(&c, "u1#clo1", "capacitance"), c1);
    assert_eq!(parameter(&c, "u1#clo3", "capacitance"), p * p * c1);
}

#[test]
fn n_is_rounded_like_an_if_integer_setter() {
    let c = circuit("v1 in 0 1\nu1 in out 0 m l=1m n=1.5\nr1 out 0 1k\n.model m urc").unwrap();
    // floor(1.5 + 0.5) = 2 sections: hi1, lo1, hi2.
    assert!(c.device("u1#rhi2").is_some());
    assert!(c.device("u1#rlo3").is_none());
    assert!(c.device("u1#chi2").is_none());
}

#[test]
fn the_dc_solution_is_the_series_resistance() {
    // r0 = 1m x 1meg/m = 1k in series with 1k: out = 0.5 V, the internal
    // nodes on the resistive divider and no current into the reference.
    let plot = op(
        "v1 in 0 1\nu1 in out ref m l=1m n=4\nr1 out 0 1k\nvr ref 0 0.3\n\
                   .model m urc rperl=1meg",
    )
    .unwrap();
    let value = |name: &str| plot.value(name, 0).unwrap().re;
    assert!((value("v(out)") - 0.5).abs() < 1e-12);
    assert!((value("v(u1#hi4)") - 0.75).abs() < 1e-12);
    assert!(value("i(vr)").abs() < 1e-15);
}

#[test]
fn isperl_selects_the_diode_ladder() {
    let c = circuit(
        "v1 in 0 1\nu1 in out 0 m l=1m n=2\nr1 out 0 1k\n\
         .model m urc isperl=1e-12 rsperl=10",
    )
    .unwrap();
    assert_eq!(
        names(&c),
        [
            "v1", "u1", "u1#rlo1", "u1#rhi1", "u1#dlo1", "u1#dhi1", "u1#rlo2", "u1#rhi2",
            "u1#dlo2", "r1"
        ]
    );
    for diode in ["u1#dlo1", "u1#dhi1", "u1#dlo2"] {
        assert_eq!(c.device(diode).unwrap().designator(), 'd');
    }
    // area = p^(i-1).
    assert_eq!(parameter(&c, "u1#dlo1", "area"), 1.);
    assert_eq!(parameter(&c, "u1#dlo2", "area"), 1.5);
    // rd = l x lumps x RSPERL = 0.02 ohm > 0: each diode has its internal
    // series-resistance node, hidden like C's `#internal`.
    let internal = c
        .nodes()
        .nodes()
        .iter()
        .filter(|n| n.kind == ngspice_rs::primitives::NodeKind::Internal)
        .count();
    assert_eq!(internal, 3);
    assert!(op("v1 in 0 1\nu1 in out 0 m l=1m\nr1 out 0 1k\n.model m urc isperl=1e-12").is_ok());
}

#[test]
fn lines_inside_subcircuits_get_hierarchical_names() {
    let c = circuit(
        "x1 in out line\n.subckt line a b\nu1 a b 0 lm l=1m n=2\n.model lm urc(rperl=1k)\n\
         .ends\nv1 in 0 1\nr1 out 0 1k",
    )
    .unwrap();
    assert!(c.device("u.x1.u1#rlo1").is_some(), "{:?}", names(&c));
    assert!(c.nodes().get("u.x1.u1#hi2").is_some());
}

#[test]
fn degenerate_and_unsupported_inputs_are_explicit_errors() {
    let refuse = |body: &str, needle: &str| {
        let error = circuit(&format!("v1 in 0 1\n{body}\nr1 out 0 1k")).unwrap_err();
        assert!(message(&error).contains(needle), "{body}: {error}");
    };
    refuse("u1 in out 0 m\n.model m urc", "without l=");
    refuse("u1 in out 0 m l=0\n.model m urc", "l must satisfy");
    refuse("u1 in out 0 m l=1m n=0\n.model m urc", "builds no sections");
    refuse("u1 in out 0 m l=1m\n.model m urc k=1", "different from 1");
    refuse(
        "u1 in out 0 m l=1m\n.model m urc cperl=0",
        "CPERL must be positive",
    );
    refuse(
        "u1 in out 0 m l=1m\n.model m urc isperl=0",
        "isperl must satisfy",
    );
    refuse(
        "u1 in out 0 m l=1m\n.model m urc bogus=1",
        "unsupported setter 'bogus'",
    );
    refuse(
        "u1 in out 0 m l=1m\n.model m urc level=1",
        "level on a urc model",
    );
    refuse("u1 in out 0 m l=1m n=100000\n.model m urc", "exceed");
    refuse("u1 in out 0 nomodel l=1m", "not defined");
    refuse("u1 in out 0 m l=1m\n.model m d", "wrong model family");
    // A user node spelled like a generated one is refused, not merged.
    refuse(
        "u1 in out 0 m l=1m\nr2 out u1#hi1 1k\n.model m urc",
        "internal-node name collision",
    );
}

#[test]
fn a_failed_expansion_leaves_the_circuit_untouched() {
    let netlist = parse("u1 in out 0 m l=1m n=0\n.model m urc").unwrap();
    let models = ngspice_rs::devices::ModelResolver::new(&netlist.models).unwrap();
    let mut c = Circuit::new();
    assert!(
        c.add_instances(&netlist.devices, &models, &ModelContext::default())
            .is_err()
    );
    assert_eq!(c.device_count(), 0);
    assert!(c.nodes().nodes().iter().all(|n| n.name == "0"));
}

#[test]
fn the_ac_response_falls_with_frequency() {
    let plot = AnalysisInput::with_arguments(AnalysisKind::Ac, ["dec", "1", "1k", "1g"])
        .resolve()
        .and_then(|plan| {
            plan.run(
                &mut circuit(
                    "v1 in 0 dc 0 ac 1\nu1 in out 0 m l=1m\nr1 out 0 1meg\n\
                 .model m urc rperl=1e7 cperl=1e-7",
                )
                .unwrap(),
                &AnalysisContext::default(),
            )
        })
        .unwrap();
    let out: Vec<Real> = plot
        .column("v(out)")
        .unwrap()
        .iter()
        .map(|v| v.magnitude())
        .collect();
    assert!((out[0] - 1e6 / 1.01e6).abs() < 1e-3, "{out:?}");
    assert!(out.windows(2).all(|w| w[1] < w[0]), "{out:?}");
    assert!(*out.last().unwrap() < 1e-6, "{out:?}");
}

#[test]
fn pole_zero_and_sensitivity_are_explicitly_refused() {
    let deck = "v1 in 0 dc 1 ac 1\nu1 in out 0 m l=1m n=2\nr1 out 0 10k\n\
                .model m urc rperl=1meg cperl=1n";
    // C aborts `.pz` (URCsetup re-run as DEVpzSetup: "device already exists").
    let error =
        AnalysisInput::with_arguments(AnalysisKind::PoleZero, ["in", "0", "out", "0", "vol", "pz"])
            .resolve()
            .and_then(|plan| plan.run(&mut circuit(deck).unwrap(), &AnalysisContext::default()))
            .unwrap_err();
    assert!(message(&error).contains("URC line u1"), "{error}");
    // C's `.sens` also lists the URC's own parameters (zero sensitivities).
    let error = AnalysisInput::with_arguments(AnalysisKind::Sensitivity, ["v", "(", "out", ")"])
        .resolve()
        .and_then(|plan| plan.run(&mut circuit(deck).unwrap(), &AnalysisContext::default()))
        .unwrap_err();
    assert!(error.is_not_yet_ported(), "{error}");
    // `.noise` runs: the instance is noiseless (DEVnoise = NULL) and the
    // generated resistors carry the noise.
    let plot = AnalysisInput::with_arguments(
        AnalysisKind::Noise,
        ["v(out)", "v1", "dec", "1", "1k", "1meg"],
    )
    .resolve()
    .and_then(|plan| plan.run(&mut circuit(deck).unwrap(), &AnalysisContext::default()));
    assert!(plot.is_ok(), "{plot:?}");
}

#[test]
fn programmatic_urc_terminal_spelling_does_not_change_setup_numbering() {
    // AST callers need not have normalized external node spellings. In
    // particular, external terminals must never be treated as ladder nodes.
    let mut parsed =
        parse("u1 a b ref m l=1m n=3\nv1 a 0 1\nr1 b 0 1k\nvr ref 0 0\n.model m urc").unwrap();
    let canonical = Circuit::from_netlist(&parsed).unwrap();
    parsed.devices[0].nodes = vec!["A".into(), "B".into(), "REF".into()];
    let mixed = Circuit::from_netlist(&parsed).unwrap();
    assert_eq!(canonical.unknowns(), mixed.unknowns());
    assert_eq!(
        canonical.unknown_names().as_ref(),
        mixed.unknown_names().as_ref()
    );
    for index in 0..canonical.device_count() {
        assert_eq!(
            canonical.setup_elements(index).unwrap(),
            mixed.setup_elements(index).unwrap()
        );
    }
}

#[test]
fn generated_urc_terminal_collision_is_case_insensitive_for_future_cards() {
    let mut parsed = parse("u1 in out 0 m l=1m n=2\nr1 u1#hi1 0 1k\n.model m urc").unwrap();
    parsed.devices[1].nodes[0] = "U1#HI1".into();
    let models = ngspice_rs::devices::ModelResolver::new(&parsed.models).unwrap();
    let mut circuit = Circuit::new();
    let error = circuit
        .add_instances(&parsed.devices, &models, &ModelContext::default())
        .unwrap_err();
    assert!(error.to_string().contains("collision"), "{error}");
    assert_eq!(circuit.device_count(), 0);
    assert_eq!(circuit.nodes().len(), 1);
}

fn limited_circuit(
    body: &str,
    limits: ngspice_rs::devices::ElaborationLimits,
) -> SpiceResult<Circuit> {
    let netlist = parse(body)?;
    let models = ngspice_rs::devices::ModelResolver::new(&netlist.models)?;
    let mut c = Circuit::with_elaboration_limits(limits);
    c.add_instances(&netlist.devices, &models, &ModelContext::default())?;
    c.finalize()?;
    Ok(c)
}

#[test]
fn capacitor_and_diode_ladders_use_exact_aggregate_resource_counts() {
    use ngspice_rs::devices::ElaborationLimits;
    for n in [1, 3, 7] {
        for (model, primed, slots) in [
            ("", false, 2),
            ("isperl=1e-12 rsperl=0", false, 3),
            ("isperl=1e-12 rsperl=10", true, 3),
        ] {
            let body = format!("u1 in out 0 m l=1m n={n}\n.model m urc {model}");
            let shunts = 2 * n - 1;
            let limits = ElaborationLimits {
                devices: 4 * n,
                nodes: 2 + shunts * (1 + usize::from(primed)),
                state_slots: shunts * slots,
            };
            let c = limited_circuit(&body, limits).unwrap();
            assert_eq!(c.device_count(), limits.devices);
            assert_eq!(c.nodes().len() - 1, limits.nodes);
            assert_eq!(c.state_len(), limits.state_slots);
            let ordinary = circuit(&body).unwrap();
            assert_eq!(names(&c), names(&ordinary));
            assert_eq!(c.unknown_names(), ordinary.unknown_names());
            for (smaller, label) in [
                (
                    ElaborationLimits {
                        devices: limits.devices - 1,
                        ..limits
                    },
                    "devices",
                ),
                (
                    ElaborationLimits {
                        nodes: limits.nodes - 1,
                        ..limits
                    },
                    "non-ground nodes",
                ),
                (
                    ElaborationLimits {
                        state_slots: limits.state_slots - 1,
                        ..limits
                    },
                    "state slots",
                ),
            ] {
                let error = limited_circuit(&body, smaller).unwrap_err();
                assert!(matches!(error, SpiceError::Parse { .. }), "{error}");
                assert!(
                    message(&error).contains("elaboration resource limit")
                        && message(&error).contains(label),
                    "{error}"
                );
            }
        }
    }
}

#[test]
fn aggregate_generated_terminals_are_deduplicated_across_plans_and_existing_nodes() {
    use ngspice_rs::devices::{ElaborationLimits, ModelResolver};
    let mut netlist = parse("u1 a a 0 m l=1m n=1\nu2 b a 0 m l=1m n=1\n.model m urc").unwrap();
    // Hand-built ASTs retain written case and aliases.
    netlist.devices[0].nodes[1] = "A".into();
    netlist.devices[1].nodes[2] = "GND".into();
    let models = ModelResolver::new(&netlist.models).unwrap();
    let mut c = Circuit::with_elaboration_limits(ElaborationLimits {
        devices: 8,
        nodes: 4,
        state_slots: 4,
    });
    c.add_node("a").unwrap();
    c.add_instances(&netlist.devices, &models, &ModelContext::default())
        .unwrap();
    c.finalize().unwrap();
    assert_eq!(c.device_count(), 8);
    assert_eq!(c.nodes().len(), 5);
    assert_eq!(c.state_len(), 4);
}

#[test]
fn aggregate_default_limit_precedes_any_large_ladder_materialization() {
    use ngspice_rs::devices::ModelResolver;
    use std::fmt::Write as _;
    // Each line is individually valid with 10,000 sections; seven lines
    // exceed the default 250,000 total materialized records.
    let mut body = String::new();
    for i in 1..=7 {
        writeln!(body, "u{i} a b 0 m l=1m n=10000").unwrap();
    }
    body.push_str(".model m urc k=1.0001");
    let netlist = parse(&body).unwrap();
    let models = ModelResolver::new(&netlist.models).unwrap();
    let mut c = Circuit::new();
    // This collision would fail the first group immediately if it were
    // materialized before checking the entire batch's aggregate plan.
    c.add_node("u1#hi1").unwrap();
    let before = c.nodes().nodes().to_vec();
    let error = c
        .add_instances(&netlist.devices, &models, &ModelContext::default())
        .unwrap_err();
    assert!(
        message(&error).contains("280000 devices exceed 250000"),
        "{error}"
    );
    assert_eq!(c.nodes().nodes(), before);
    assert_eq!(c.device_count(), 0);
}

#[test]
fn repeated_batches_and_late_ordinary_state_failure_preserve_finalized_bindings() {
    use ngspice_rs::devices::{ElaborationLimits, ModelResolver};
    let context = ModelContext::default();
    let initial = parse("rkeep keep 0 1k\n.model m urc").unwrap();
    let mut c = Circuit::with_elaboration_limits(ElaborationLimits {
        devices: 10,
        nodes: 12,
        state_slots: 5,
    });
    let models = ModelResolver::new(&initial.models).unwrap();
    c.add_instances(&initial.devices, &models, &context)
        .unwrap();
    let urc = parse("u1 a b 0 m l=1m n=1\n.model m urc").unwrap();
    let models = ModelResolver::new(&urc.models).unwrap();
    c.add_instances(&urc.devices, &models, &context).unwrap();
    c.finalize().unwrap();
    let before_nodes = c.nodes().nodes().to_vec();
    let before_names = names(&c);
    let before_unknowns = c.unknowns().clone();
    let before_pattern = c.matrix_pattern().unwrap();
    let bad = parse("u2 b c 0 m l=1m n=1\nc2 c 0 1n\n.model m urc").unwrap();
    // The generated group alone fits; the ordinary capacitor contributes
    // another two actual state slots after binding, so the whole batch fails.
    let models = ModelResolver::new(&bad.models).unwrap();
    let error = c
        .add_instances(&bad.devices, &models, &context)
        .unwrap_err();
    assert!(
        message(&error).contains("6 state slots exceed 5"),
        "{error}"
    );
    assert_eq!(c.nodes().nodes(), before_nodes);
    assert_eq!(names(&c), before_names);
    assert_eq!(c.unknowns(), &before_unknowns);
    assert!(std::sync::Arc::ptr_eq(
        &before_pattern,
        &c.matrix_pattern().unwrap()
    ));
    assert_eq!(c.state_len(), 2);
    c.add_instances(&bad.devices[..1], &models, &context)
        .unwrap();
    c.finalize().unwrap();
    assert_eq!(c.device_count(), 9);
    assert_eq!(c.state_len(), 4);
    let third = parse("u3 c d 0 m l=1m n=1\n.model m urc").unwrap();
    let models = ModelResolver::new(&third.models).unwrap();
    let error = c
        .add_instances(&third.devices, &models, &context)
        .unwrap_err();
    assert!(message(&error).contains("13 devices exceed 10"), "{error}");
    assert_eq!(c.device_count(), 9);
    assert_eq!(c.state_len(), 4);
}
