//! #16 run configuration: precedence, validation and per-deck isolation.
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, RunConfig, RunOverrides};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, SpiceError};
use std::path::Path;

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("cfg.cir"),
            &format!("t\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn config(options: &str) -> Result<RunConfig, SpiceError> {
    RunConfig::from_netlist(&deck(&format!("r1 a 0 1k\n{options}")))
}

#[test]
fn defaults_are_unchanged() {
    let c = config("").unwrap();
    assert_eq!(c.context(), AnalysisContext::default());
    assert_eq!(c.context().temperature, 27.0);
    assert_eq!(c.context().nominal_temperature, 27.0);
    assert_eq!(*c.transient(), Default::default());
    assert!(c.applied().is_empty() && c.method().is_none() && c.maxord().is_none());
    let tran = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"]);
    assert_eq!(c.plan(tran.clone()).unwrap(), tran.resolve().unwrap());
}

#[test]
fn ordered_overrides_are_case_insensitive_and_last_wins() {
    let c =
        config(".options TEMP=10 tnom=20\n.OPTION Temp=50 RELTOL=1e-4\n.opt reltol=2e-4").unwrap();
    assert_eq!(c.context().temperature, 50.0);
    assert_eq!(c.context().nominal_temperature, 20.0);
    assert_eq!(c.context().reltol, 2e-4);
    assert_eq!(c.applied().len(), 5);
    assert_eq!(c.applied()[0].name, "temp");
    assert_eq!(c.applied()[0].value, "10");
}

#[test]
fn overrides_beat_deck_and_request_arguments_beat_deck() {
    let overrides = RunOverrides {
        temperature: Some(85.0),
        nominal_temperature: None,
    };
    let n = deck("r1 a 0 1k\n.options temp=10 tnom=20 reltol=1e-3 vntol=2u abstol=3p");
    let c = RunConfig::from_options(&n.options, &overrides).unwrap();
    assert_eq!(c.context().temperature, 85.0);
    assert_eq!(c.context().nominal_temperature, 20.0);
    assert!(
        RunConfig::from_options(
            &n.options,
            &RunOverrides {
                temperature: Some(f64::NAN),
                ..Default::default()
            }
        )
        .is_err()
    );

    let request = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m", "RTOL=5e-5"]);
    let merged = c.plan(request).unwrap();
    assert_eq!(c.context().model_context().reltol, 1e-3);
    assert_eq!(
        merged.dc_settings().newton.reltol,
        ngspice_rs::primitives::parse_spice_number("5e-5").unwrap()
    );
    assert_eq!(
        merged.dc_settings().newton.vntol,
        c.transient().vntol.unwrap()
    );
    assert_eq!(
        merged.dc_settings().newton.abstol,
        c.transient().abstol.unwrap()
    );
    // M4 forwards the same physical tolerances to the reusable DC Newton solve.
    let op = c
        .plan(AnalysisInput::new(AnalysisKind::OperatingPoint))
        .unwrap();
    assert_eq!(op.dc_settings().newton.reltol, c.context().reltol);
    assert_eq!(op.dc_settings().newton.vntol, c.transient().vntol.unwrap());
    assert_eq!(
        op.dc_settings().newton.abstol,
        c.transient().abstol.unwrap()
    );
}

#[test]
fn invalid_values_are_errors() {
    for options in [
        ".options temp=-273.15",
        ".options temp=-300",
        ".options temp=hot",
        ".options reltol=0",
        ".options gshunt=-1",
        ".options minbreak=-1n",
        ".options minbreak=abc",
        ".options lteabstol=0",
        ".options ltereltol=-1",
        ".options ltetrtol=abc",
        ".options newtrunc=1",
        ".options reltol=-1e-3",
        ".options vntol=abc",
        ".options abstol=0",
        ".options maxord=0",
        ".options maxord=7",
        ".options maxord=gear",
        ".options method=euler",
        ".options method=2",
        ".options temp",
        ".options reltol",
        ".options method",
    ] {
        let error = config(options).expect_err(options);
        assert!(!error.is_not_yet_ported(), "{options}: {error}");
    }
}

#[test]
fn unknown_unimplemented_and_conflicting_options_are_errors() {
    for options in [
        ".options bogus=1",
        ".options no_auto_gnd",
        ".options RELTO=1",
    ] {
        assert!(
            matches!(config(options), Err(SpiceError::Parse { .. })),
            "{options}"
        );
    }
    for options in [
        // srcsteps/gminsteps/gminfactor/itl1 are implemented (tests/dc_continuation.rs);
        // itl2/itl4/itl6/gmin/xmu and the documented no-ops in tests/options_coverage.rs;
        // noopiter in tests/convergence.rs.
        ".options bypass=1",
    ] {
        assert!(
            config(options).unwrap_err().is_not_yet_ported(),
            "{options}"
        );
    }
    let error = config(".options reltol=1m\n.options reltol").unwrap_err();
    assert!(
        error.to_string().contains("flag and with a value"),
        "{error}"
    );
}

#[test]
fn method_and_maxord_reach_the_companion_driver() {
    use ngspice_rs::maths::integrator::IntegrationMethod;
    let c = config(".options method=Gear maxord=2").unwrap();
    assert_eq!(
        (c.method(), c.maxord()),
        (Some(ngspice_rs::analysis::IntegrationFamily::Gear), Some(2))
    );
    let ordinary = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"]);
    for (options, method, order) in [
        (".options method=trap", IntegrationMethod::Trapezoidal, 2),
        (
            ".options method=trapezoidal",
            IntegrationMethod::Trapezoidal,
            2,
        ),
        (
            ".options method=gear",
            IntegrationMethod::Gear { order: 2 },
            2,
        ),
        (".options maxord=2", IntegrationMethod::Trapezoidal, 2),
        (
            ".options method=gear maxord=1",
            IntegrationMethod::Gear { order: 1 },
            1,
        ),
    ] {
        let c = config(options).unwrap();
        let plan = c.plan(ordinary.clone()).unwrap();
        let settings = plan.transient().unwrap();
        assert_eq!(
            (settings.method(), settings.max_order()),
            (method, order),
            "{options}"
        );
        let ac = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "3", "1", "3"]);
        assert_eq!(c.plan(ac.clone()).unwrap(), ac.resolve().unwrap());
    }
    // An explicit request selection beats the deck's.
    let c = config(".options method=gear maxord=1").unwrap();
    let explicit = AnalysisInput::with_arguments(
        AnalysisKind::Transient,
        ["1u", "1m", "method=trap", "maxord=2"],
    );
    assert_eq!(
        c.plan(explicit.clone()).unwrap(),
        explicit.resolve().unwrap()
    );
    // Gear orders up to ngspice's 6 are forwarded (#98); the companion driver
    // follows dctran.c, which never runs above order 2.
    let request = config(".options method=gear maxord=4")
        .unwrap()
        .plan(ordinary.clone())
        .unwrap();
    assert_eq!(
        (
            request.transient().unwrap().method(),
            request.transient().unwrap().max_order()
        ),
        (IntegrationMethod::Gear { order: 4 }, 4)
    );
    assert!(config(".options method=gear maxord=7").is_err());
    // trtol/chgtol are companion truncation options.
    let c = config(".options trtol=3 chgtol=1e-13").unwrap();
    assert_eq!(
        (c.transient().trtol, c.transient().chgtol),
        (Some(3.), Some(1e-13))
    );
    let request = c.plan(ordinary.clone()).unwrap();
    assert_eq!(
        (
            request.transient().unwrap().trtol(),
            request.transient().unwrap().chgtol()
        ),
        (3., c.transient().chgtol.unwrap())
    );
    assert!(config(".options trtol=0").is_err());
}

#[test]
fn minbreak_is_real_valued_and_resolved_per_transient() {
    let c = config(".param spacing=2n\n.options minbreak={spacing}").unwrap();
    let input = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"]);
    let plan = c.plan(input.clone()).unwrap();
    assert_eq!(
        plan.transient().unwrap().min_break(),
        c.transient().minbreak.unwrap()
    );
    let explicit =
        AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m", "minbreak=5n"]);
    assert_eq!(
        c.plan(explicit.clone()).unwrap(),
        explicit.resolve().unwrap()
    );
    let reset = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m", "minbreak=0"]);
    assert_eq!(c.plan(reset).unwrap(), input.resolve().unwrap());
    let invalid =
        AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m", "minbreak=-1"]);
    assert!(c.plan(invalid).is_err());
}

#[test]
fn voltage_lte_request_overrides_deck_and_does_not_leak() {
    let c = config(".options newtrunc lteabstol=1u ltereltol=0.01 ltetrtol=10").unwrap();
    let request = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"]);
    let explicit = AnalysisInput::with_arguments(
        AnalysisKind::Transient,
        [
            "1u",
            "1m",
            "newtrunc=1",
            "lteabstol=1u",
            "ltereltol=0.01",
            "ltetrtol=10",
        ],
    );
    assert_eq!(
        c.plan(request.clone()).unwrap(),
        explicit.resolve().unwrap()
    );
    let disabled =
        AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m", "newtrunc=0"]);
    assert_eq!(c.plan(disabled).unwrap(), request.resolve().unwrap());
    let invalid =
        AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m", "ltetrtol=0"]);
    assert!(c.plan(invalid).is_err());
}

#[test]
fn deck_options_take_effect_in_a_real_transient_and_do_not_leak() {
    let rc = "v1 in 0 dc 1\nr1 in out 1k\nc1 out 0 1u\n.tran 1m 5m";
    let loose = deck(&format!("{rc}\n.options reltol=0.5 vntol=1 abstol=1"));
    let strict = deck(rc);
    let run = |n: &Netlist| {
        let c = RunConfig::from_netlist(n).unwrap();
        let mut circuit = c.circuit(n).unwrap();
        let request = c.plan_for(&n.analyses[0]).unwrap();
        let plot = request.run(&mut circuit, &c.context());
        (c, plot)
    };
    let (c_loose, loose_plot) = run(&loose);
    let (c_strict, strict_plot) = run(&strict);
    assert_eq!(c_loose.context().reltol, 0.5);
    assert_eq!(
        c_strict.context().reltol,
        ngspice_rs::devices::DEFAULT_RELTOL
    );
    // Both decks simulate; the strict deck is unaffected by the earlier one.
    assert!(
        loose_plot.is_ok() && strict_plot.is_ok(),
        "{loose_plot:?} {strict_plot:?}"
    );
    assert_eq!(c_strict.context(), AnalysisContext::default());
}

#[test]
fn temperatures_reach_elaboration_and_plain_elaboration_refuses_options() {
    let n = deck("r1 a 0 rm\n.model rm r(r=1k tc1=0.01 tnom=27)\nv1 a 0 1\n.options temp=127");
    let c = RunConfig::from_netlist(&n).unwrap();
    let mut hot = c.circuit(&n).unwrap();
    let mut cold = RunConfig::default().circuit(&n).unwrap();
    let op = |circuit: &mut ngspice_rs::devices::Circuit, ctx: &AnalysisContext| {
        AnalysisInput::new(AnalysisKind::OperatingPoint)
            .resolve()
            .and_then(|plan| plan.run(circuit, ctx))
            .unwrap()
            .value("i(v1)", 0)
            .unwrap()
            .re
    };
    let hot_i = op(&mut hot, &c.context());
    let cold_i = op(&mut cold, &AnalysisContext::default());
    assert!((hot_i - cold_i).abs() > 1e-9, "{hot_i} {cold_i}");
    assert!(matches!(
        ngspice_rs::devices::Circuit::from_netlist(&n),
        Err(SpiceError::Unsupported { .. })
    ));
}

#[test]
fn top_level_globals_are_accepted_by_flat_elaboration() {
    let n = deck("v1 a 0 1\nr1 a gnd 1k\n.global a gnd");
    assert!(RunConfig::from_netlist(&n).is_ok());
    assert!(ngspice_rs::devices::Circuit::from_netlist(&n).is_ok());
}

#[test]
fn parsed_ic_and_nodeset_cards_travel_with_every_request() {
    // #27: the evaluated cards (braced expressions against .param, duplicates
    // in deck order) are attached to each analysis request instead of rejected.
    let netlist = deck(
        "r1 a 0 1k\n.param half=0.5\n.ic v(a)=1 v(a)={half*3}\n.nodeset v(a)=2\n.op\n.tran 1u 1m",
    );
    let config = RunConfig::from_netlist(&netlist).unwrap();
    for analysis in &netlist.analyses {
        let request = config.plan_for(analysis).unwrap();
        let ic: Vec<_> = request
            .initial_conditions()
            .iter()
            .map(|c| (c.node.as_str(), c.value))
            .collect();
        // Correctly rounded read-back of the numparam-formatted value.
        assert_eq!(ic, [("a", 1.), ("a", 1.5)]);
        assert_eq!(request.nodesets().len(), 1);
        assert_eq!(request.nodesets()[0].value, 2.);
        assert_eq!(request.nodesets()[0].location.line, 5);
    }
    // An undefined parameter is an error, not a silently dropped entry.
    let netlist = deck("r1 a 0 1k\n.ic v(a)={nope}\n.tran 1u 1m");
    assert!(RunConfig::from_netlist(&netlist).is_err());
}

#[test]
fn tran_uic_is_a_request_flag_the_companion_driver_runs() {
    let netlist = deck("v1 a 0 0\nr1 a b 1k\nc1 b 0 1u\n.tran 1u 10u uic");
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    assert!(request.uic());
    assert!(!netlist.analyses[0].arguments.iter().any(|a| a == "uic"));
    let mut circuit = config.circuit(&netlist).unwrap();
    request.run(&mut circuit, &config.context()).unwrap();
}

#[test]
fn scalar_temperature_directive_has_frontend_precedence() {
    for options in [
        ".temp -30\n.options temp=75 tnom=20",
        ".options temp=75 tnom=20\n.temp -30",
        ".temp 75\n.temp -30\n.options tnom=20",
        ".param tt=-35\n.temp {tt+5}\n.options tnom=20 temp=75",
        ".param tt=-30\n.temp 'tt'\n.options tnom=20",
    ] {
        let c = config(options).unwrap();
        assert_eq!(c.context().temperature, -30., "{options}");
        assert_eq!(c.context().nominal_temperature, 20., "{options}");
    }
    let n = deck("r1 a 0 1k\n.temp -30\n.option temp=75 tnom=20");
    let overridden = RunConfig::from_options(
        &n.options,
        &RunOverrides {
            temperature: Some(85.),
            nominal_temperature: None,
        },
    )
    .unwrap();
    assert_eq!(overridden.context().temperature, 85.);
    assert_eq!(overridden.context().nominal_temperature, 20.);
    for directive in [".temp -273.15", ".temp -300"] {
        let error = config(directive).unwrap_err();
        assert!(matches!(error, SpiceError::Parse { .. }), "{error}");
        assert!(error.to_string().contains("cfg.cir:3"), "{error}");
    }
}

#[test]
fn temperature_directives_follow_instantiated_scopes() {
    let body = ".subckt outer a\n.subckt inner b params: tt=75\n.temp {tt}\nr1 b 0 1k\n.ends\nxinner a inner tt=-30\n.ends";
    let unused = config(body).unwrap();
    assert_eq!(unused.context().temperature, 27.);
    let instantiated = config(&format!("{body}\n.temp 75\nxouter a outer")).unwrap();
    assert_eq!(instantiated.context().temperature, -30.);
}
