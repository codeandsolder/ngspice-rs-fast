//! #110 / #107: every `.option` name this port accepts either takes effect,
//! is a documented no-op (`RunConfig::ignored`) or fails explicitly; option
//! values may be `{expr}`/`'expr'` evaluated against top-level `.param`.
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot, RunConfig, RunOverrides};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, SpiceError};
use std::path::Path;

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("opts.cir"),
            &format!("t\n{body}\n.end\n"),
        ))
        .unwrap_or_else(|error| panic!("{error}\n{body}"))
}

fn config(options: &str) -> Result<RunConfig, SpiceError> {
    RunConfig::from_netlist(&deck(&format!("r1 a 0 1k\n{options}")))
}

/// Run the deck's first analysis through `RunConfig`, as the CLI does.
fn simulate(body: &str) -> Result<Plot, SpiceError> {
    let n = deck(body);
    let c = RunConfig::from_netlist(&n)?;
    let request = c.plan_for(&n.analyses[0])?;
    let mut circuit = c.circuit(&n)?;
    request.run(&mut circuit, &c.context())
}

fn input_for(kind: AnalysisKind) -> AnalysisInput {
    let text = match kind {
        AnalysisKind::OperatingPoint => "",
        AnalysisKind::DcSweep => "v1 0 1 0.1",
        AnalysisKind::Ac | AnalysisKind::SParameter | AnalysisKind::Distortion => "dec 1 1k 10k",
        AnalysisKind::Noise => "v(out) v1 dec 1 1k 10k",
        AnalysisKind::PoleZero => "in 0 out 0 vol pz",
        AnalysisKind::Sensitivity => "v ( out )",
        AnalysisKind::TransferFunction => "v ( out ) v1",
        AnalysisKind::Transient => "1u 10u",
        AnalysisKind::PeriodicSteadyState => "1k 1m out",
        AnalysisKind::HarmonicBalance => "1k 3",
        AnalysisKind::Fourier => unreachable!("not a driver"),
    };
    AnalysisInput::with_arguments(kind, text.split_whitespace())
}

fn stage_limit(plan: &ngspice_rs::analysis::AnalysisPlan) -> Option<usize> {
    plan.dc_settings()
        .continuation
        .stage_max_iterations
        .map(ngspice_rs::analysis::bias::IterationLimit::get)
}

fn tran() -> AnalysisInput {
    AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"])
}

fn dc() -> AnalysisInput {
    AnalysisInput::with_arguments(AnalysisKind::DcSweep, ["v1", "0", "1", "0.1"])
}

/// ngspice's default Newton tolerances (`CKTreltol`, `CKTvoltTol`): two
/// converged solutions of one circuit reached along different iterates
/// agree only to these (`NIiter` stops once the update is within them).
const NIITER_RELTOL: f64 = 1e-3;
const NIITER_VNTOL: f64 = 1e-6;

fn close(a: f64, b: f64, rel: f64, abs: f64) {
    assert!(
        (a - b).abs() <= rel * a.abs().max(b.abs()) + abs,
        "{a:e} vs {b:e}"
    );
}

// ---- options with an effect -------------------------------------------------

#[test]
fn physical_shunts_match_independent_nodal_equations() {
    let circuit = "v1 in 0 dc 1 ac 1\nr1 in out 1k";
    let plot = simulate(&format!("{circuit}\n.options rshunt=1k\n.op")).unwrap();
    close(plot.value("v(out)", 0).unwrap().re, 0.5, 1e-12, 0.);
    close(plot.value("i(v1)", 0).unwrap().re, -1.5e-3, 1e-12, 0.);
    // An extreme finite resistance must retain its real reciprocal.
    let large = simulate(&format!("{circuit}\n.options rshunt=1e15\n.op")).unwrap();
    close(large.value("i(v1)", 0).unwrap().re, -2e-15, 0., 5e-19);
    let frequency = 1000.;
    let omega_c = std::f64::consts::TAU * frequency * 1e-6;
    let denominator = 1. + (omega_c * 1000.).powi(2);
    let ac = simulate(&format!("{circuit}\n.options cshunt=1u\n.ac lin 1 1k 1k")).unwrap();
    let output = ac.value("v(out)", 0).unwrap();
    close(output.re, denominator.recip(), 1e-12, 0.);
    close(output.im, -omega_c * 1000. / denominator, 1e-12, 0.);
    let current = ac.value("i(v1)", 0).unwrap();
    close(current.re, (output.re - 1.) / 1000., 1e-12, 0.);
    close(current.im, output.im / 1000. - omega_c, 1e-12, 0.);
    let unshunted = simulate(&format!("{circuit}\n.op")).unwrap();
    close(unshunted.value("v(out)", 0).unwrap().re, 1., 1e-12, 0.);
}

#[test]
fn topology_shunts_agree_with_explicit_reactive_storage() {
    let circuit = "v1 in 0 pulse(0 1 1u 1u 1u 1m 2m)\nr1 in out 1k";
    let analysis = ".tran 1u 50u";
    for method in ["trap", "gear"] {
        let generated = simulate(&format!(
            "{circuit}\n.options rshunt=1k cshunt=1n method={method}\n{analysis}"
        ))
        .unwrap();
        let explicit = simulate(&format!(
            "{circuit}\nrsi in 0 1k\nrso out 0 1k\ncsi in 0 1n\ncso out 0 1n\n.options method={method}\n{analysis}"
        )).unwrap();
        assert_eq!(generated.point_count(), explicit.point_count());
        for point in 0..generated.point_count() {
            for variable in ["time", "v(out)", "i(v1)"] {
                close(
                    generated.value(variable, point).unwrap().re,
                    explicit.value(variable, point).unwrap().re,
                    1e-8,
                    1e-12,
                );
            }
        }
    }
}

#[test]
fn topology_shunts_validate_and_preserve_branch_coordinates() {
    for invalid in [
        ".options rshunt=0",
        ".options rshunt=-1",
        ".options rshunt=1e-320",
        ".options cshunt=-1",
        ".options cshunt=abc",
    ] {
        assert!(config(invalid).is_err(), "{invalid}");
    }
    let n = deck(
        "v1 a 0 1\nr1 a b 1k\nd1 b 0 dm\n.model dm d(is=1e-14 rs=10)\n.options rshunt=1e15 cshunt=1p",
    );
    let c = RunConfig::from_netlist(&n).unwrap();
    let circuit = c.circuit(&n).unwrap();
    let nodes = circuit.nodes().nodes();
    let terminals = nodes
        .iter()
        .filter(|node| node.kind == ngspice_rs::primitives::NodeKind::Terminal)
        .count();
    let internal = nodes
        .iter()
        .filter(|node| node.kind == ngspice_rs::primitives::NodeKind::Internal)
        .count();
    assert!(internal > 0);
    let resistors = circuit
        .devices()
        .iter()
        .filter(|d| d.name().starts_with("r__rshunt_"))
        .count();
    let capacitors = circuit
        .devices()
        .iter()
        .filter(|d| d.name().starts_with("capac") && d.name().ends_with("shunt"))
        .count();
    assert_eq!(resistors, terminals + internal);
    assert_eq!(capacitors, terminals);
}

#[test]
fn gmin_is_the_junction_gmin_of_every_analysis() {
    let c = config(".options gmin=1u").unwrap();
    assert_eq!(c.context().gmin, 1e-6);
    assert_eq!(c.context().model_context().gmin, 1e-6);
    assert_eq!(config("").unwrap().context().gmin, 1e-12);
    // Reverse-biased diode: |I| = IS + gmin * 10 V (dioload.c adds CKTgmin).
    let reverse = |options: &str| {
        simulate(&format!(
            "v1 a 0 -10\nd1 a 0 dm\n.model dm d(is=1e-14)\n{options}\n.op"
        ))
        .unwrap()
        .value("i(v1)", 0)
        .unwrap()
        .re
    };
    close(reverse(".options gmin=1u").abs(), 1e-14 + 1e-5, 1e-6, 0.);
    close(reverse("").abs(), 1e-14 + 1e-11, 1e-6, 0.);
    close(reverse(".options gmin=0").abs(), 1e-14, 1e-6, 0.);
    // Small-signal: the same gmin is the reverse junction's conductance.
    let ac = simulate(
        "v1 a 0 -10 ac 1\nd1 a 0 dm\n.model dm d(is=1e-14)\n.options gmin=1u\n.ac lin 1 1k 1k",
    )
    .unwrap();
    close(ac.value("i(v1)", 0).unwrap().re.abs(), 1e-6, 1e-6, 0.);
    // The transient sees it through the same context.
    let tran =
        simulate("v1 a 0 -10\nd1 a 0 dm\n.model dm d(is=1e-14)\n.options gmin=1u\n.tran 1u 10u")
            .unwrap();
    let last = tran.point_count() - 1;
    close(
        tran.value("i(v1)", last).unwrap().re.abs(),
        1e-14 + 1e-5,
        1e-6,
        0.,
    );
    // BJT/MOS1 junctions use it as well (reverse-biased base-collector).
    let bjt = |options: &str| {
        simulate(&format!(
            "vc c 0 10\nq1 c 0 0 qm\n.model qm npn(is=1e-16)\n{options}\n.op"
        ))
        .unwrap()
        .value("i(vc)", 0)
        .unwrap()
        .re
        .abs()
    };
    // bjtload.c: gmin across base-collector plus the substrate gmin, which a
    // vertical NPN connects to the collector: 2 * 1 uS * 10 V.
    close(bjt(".options gmin=1u"), 2e-5, 1e-6, 1e-12);
    // A lateral PNP's substrate gmin connects to the base instead.
    let pnp = simulate("vc c 0 -10\nq1 c 0 0 qm\n.model qm pnp(is=1e-16)\n.options gmin=1u\n.op")
        .unwrap()
        .value("i(vc)", 0)
        .unwrap()
        .re
        .abs();
    close(pnp, 1e-5, 1e-6, 1e-12);
    assert!(bjt("") < 1e-9, "{}", bjt(""));
    let mos = |options: &str| {
        simulate(&format!(
            "vd d 0 5\nm1 d 0 0 0 mm\n.model mm nmos(vto=1)\n{options}\n.op"
        ))
        .unwrap()
        .value("i(vd)", 0)
        .unwrap()
        .re
        .abs()
    };
    assert!(
        mos(".options gmin=1u") > 4e-6,
        "{}",
        mos(".options gmin=1u")
    );
    for bad in [".options gmin=-1e-12", ".options gmin=abc", ".options gmin"] {
        let error = config(bad).unwrap_err();
        assert!(!error.is_not_yet_ported(), "{bad}: {error}");
    }
}

#[test]
fn itl1_reaches_dc_analyses_and_the_companion_initial_bias() {
    // niiter.c raises any limit below 100 to 100: the deck value is stored and
    // forwarded as C's effective limit.
    let c = config(".options itl1=37").unwrap();
    assert_eq!(c.dc().itl1, Some(100));
    for kind in [AnalysisKind::OperatingPoint, AnalysisKind::Ac] {
        let request = c.plan(input_for(kind)).unwrap();
        assert_eq!(request.dc_settings().newton.max_iterations, 100);
    }
    assert_eq!(
        c.plan(dc()).unwrap().dc_settings().newton.max_iterations,
        100
    );
    assert_eq!(
        c.plan(tran()).unwrap().dc_settings().newton.max_iterations,
        100
    );
    let above = config(".options itl1=250").unwrap();
    assert_eq!(
        above
            .plan(dc())
            .unwrap()
            .dc_settings()
            .newton
            .max_iterations,
        250
    );
    for low in ["0", "1", "99", "100"] {
        let c = config(&format!(".options itl1={low}")).unwrap();
        assert_eq!(c.dc().itl1, Some(100), "itl1={low}");
    }
    // C converges this deck with `itl1=2 gminsteps=0 srcsteps=0` (v(b) =
    // 0.69289, 5 iterations): so does the port, at the same bias.
    let deck = "v1 a 0 5\nr1 a b 1k\nd1 b 0 dm\n.model dm d(is=1e-14)\n.op";
    let plain = simulate(deck).unwrap();
    let low = simulate(&format!("{deck}\n.options itl1=2 gminsteps=0 srcsteps=0")).unwrap();
    close(
        low.value("v(b)", 0).unwrap().re,
        plain.value("v(b)", 0).unwrap().re,
        1e-9,
        1e-12,
    );
    close(low.value("v(b)", 0).unwrap().re, 0.69289, 1e-4, 0.);
    // The same holds for the initial bias of a nonlinear transient.
    let tran_deck = "v1 a 0 5\nr1 a b 1k\nd1 b 0 dm\n.model dm d(is=1e-14)\n.tran 1u 10u";
    assert!(
        simulate(&format!(
            "{tran_deck}\n.options itl1=1 gminsteps=0 srcsteps=0"
        ))
        .is_ok()
    );
    // The forwarded limit is a real bound: the literal request budget of one
    // iteration (below C's floor, so a port-only knob) fails with diagnostics.
    let n = deck_with_request(deck, &["maxiter=1", "gminsteps=0", "srcsteps=0"]);
    let error = n.unwrap_err();
    assert!(error.to_string().contains("Newton"), "{error}");
}

/// Run `body`'s first analysis with extra explicit request arguments.
fn deck_with_request(body: &str, extra: &[&str]) -> Result<Plot, SpiceError> {
    let n = deck(body);
    let c = RunConfig::from_netlist(&n)?;
    let mut request = AnalysisInput::from(&n.analyses[0]);
    request
        .arguments
        .extend(extra.iter().map(|a| (*a).to_owned()));
    let request = c.plan(request)?;
    let mut circuit = c.circuit(&n)?;
    request.run(&mut circuit, &c.context())
}

#[test]
fn itl2_bounds_the_warm_started_dc_sweep_points() {
    let c = config(".options itl2=7").unwrap();
    assert_eq!(c.dc().itl2, Some(100));
    assert_eq!(c.plan(dc()).unwrap().point_iterations(), Some(100));
    let above = config(".options itl2=300").unwrap();
    assert_eq!(above.plan(dc()).unwrap().point_iterations(), Some(300));
    // The warm start exists only in .dc; the continuation stage limit reaches
    // every analysis that runs the DC bias (tests below).
    for request in [
        AnalysisInput::new(AnalysisKind::OperatingPoint),
        input_for(AnalysisKind::Ac),
        tran(),
    ] {
        let request = c.plan(request).unwrap();
        assert_eq!(request.point_iterations(), None);
        assert_eq!(stage_limit(&request), Some(100));
    }
    // One large sweep step (0 V -> 5 V) on a diode: the first point is
    // trivial, the second needs several damped Newton iterations from the
    // previous point. With a three-iteration full-solve budget and no
    // continuation, the point fails unless the warm start takes it.
    let sweep = "v1 a 0 0\nr1 a b 1k\nd1 b 0 dm\n.model dm d(is=1e-14)\n.dc v1 0 5 5";
    // `limiting=global continuation=ladder`: the legacy damped Newton and
    // ladder policies this budget was chosen for.
    let tight = [
        "maxiter=3",
        "gminsteps=0",
        "srcsteps=0",
        "limiting=global",
        "continuation=ladder",
    ];
    let error = deck_with_request(sweep, &tight).unwrap_err();
    assert!(error.to_string().contains("Newton"), "{error}");
    // A warm-start bound of 2 is too small too: it fails, falls back to the
    // three-iteration full solve, and that fails as well.
    let mut bounded = tight.to_vec();
    bounded.push("trcvmaxiter=2");
    assert!(deck_with_request(sweep, &bounded).is_err());
    // Deck itl2=5 is C's effective 100: the warm start converges the point,
    // although a literal bound of 5 (or the full-solve budget of 3) would not.
    let mut literal = tight.to_vec();
    literal.push("trcvmaxiter=5");
    assert!(deck_with_request(sweep, &literal).is_err());
    let plot = deck_with_request(&format!("{sweep}\n.options itl2=5"), &tight).unwrap();
    let reference = simulate(sweep).unwrap();
    assert_eq!(plot.point_count(), 2);
    // Two converged solves of one point along different iterates agree to
    // NIiter's tolerance (update within reltol |v| + vntol, returning the
    // last load's iterate, CKTrhsOld), not exactly: here 2.9e-4 relative.
    close(
        plot.value("v(b)", 1).unwrap().re,
        reference.value("v(b)", 1).unwrap().re,
        NIITER_RELTOL,
        NIITER_VNTOL,
    );
    for bad in [".options itl2=10001", ".options itl2=abc"] {
        assert!(!config(bad).unwrap_err().is_not_yet_ported(), "{bad}");
    }
    // IF_INTEGER rounds: floor(0.5 + 1.5) = 2 (raised to 100 for the limit).
    let rounded = config(".options itl2=1.5").unwrap();
    assert_eq!(rounded.dc().itl2, Some(100));
    assert_eq!(rounded.dc().itl2_written, Some(2));
    assert_eq!(config(".options itl2=0").unwrap().dc().itl2, Some(100));
    // The request key is validated by the sweep driver.
    let mut circuit = RunConfig::from_netlist(&deck(sweep))
        .unwrap()
        .circuit(&deck(sweep))
        .unwrap();
    let request = AnalysisInput::with_arguments(
        AnalysisKind::DcSweep,
        ["v1", "0", "1", "0.5", "trcvmaxiter=0"],
    );
    let error = request
        .resolve()
        .and_then(|plan| plan.run(&mut circuit, &AnalysisContext::default()))
        .unwrap_err();
    assert!(error.to_string().contains("trcvmaxiter"), "{error}");
}

#[test]
fn itl2_bounds_every_continuation_stage_of_the_dc_bias() {
    // cktop.c: the direct solve runs NIiter(itl1), every dynamic/spice3 gmin
    // and source-stepping stage NIiter(itl2); CKTop serves .op, .ac, the .dc
    // first point and the .tran initial bias alike.
    let op = AnalysisInput::new(AnalysisKind::OperatingPoint);
    let ac = AnalysisInput::with_arguments(AnalysisKind::Ac, ["dec", "1", "1k", "10k"]);
    assert_eq!(
        stage_limit(&config("").unwrap().plan(op.clone()).unwrap()),
        None
    );
    for (options, stage) in [
        (".options itl2=300", 300),
        (".options itl2=7", 100),
        // itl1 alone: C keeps the stages at its default itl2 (50 -> 100).
        (".options itl1=5000", 100),
        (".options itl1=5000 itl2=250", 250),
    ] {
        let c = config(options).unwrap();
        for request in [op.clone(), ac.clone(), dc(), tran()] {
            let kind = request.kind;
            let request = c.plan(request).unwrap();
            assert_eq!(stage_limit(&request), Some(stage), "{options} {kind:?}");
        }
    }
    let c = config(".options itl1=5000").unwrap();
    assert_eq!(
        c.plan(op.clone())
            .unwrap()
            .dc_settings()
            .newton
            .max_iterations,
        5000
    );
    // Explicit request entries win.
    let mut explicit = op.clone();
    explicit.arguments.push("stagemaxiter=7".into());
    assert_eq!(stage_limit(&c.plan(explicit).unwrap()), Some(7));

    // Observable: a diode the damped direct Newton cannot reach in three
    // iterations. With maxiter=3 and no deck limits every gmin stage gets
    // three iterations too and the bias fails; deck itl2=5 (C's effective
    // 100) gives the stages 100 and gmin stepping converges, in .op, .ac and
    // the .tran initial bias alike.
    let body = "v1 a 0 5 ac 1\nr1 a b 1k\nd1 b 0 dm\n.model dm d(is=1e-14)";
    // `limiting=global continuation=ladder`: the legacy damped Newton and
    // ladder policies this budget was chosen for.
    let tight = [
        "maxiter=3",
        "srcsteps=0",
        "limiting=global",
        "continuation=ladder",
    ];
    let reference = simulate(&format!("{body}\n.op")).unwrap();
    let expected = reference.value("v(b)", 0).unwrap().re;
    for analysis in [".op", ".ac dec 1 1k 10k", ".tran 1u 2u"] {
        let plain = format!("{body}\n{analysis}");
        let error = deck_with_request(&plain, &tight).unwrap_err();
        assert!(error.to_string().contains("gmin"), "{analysis}: {error}");
        let limited = format!("{body}\n{analysis}\n.options itl2=5");
        let plot = deck_with_request(&limited, &tight).unwrap();
        if analysis == ".op" {
            // As above: NIiter's tolerance (here 4.9e-6 relative apart).
            close(
                plot.value("v(b)", 0).unwrap().re,
                expected,
                NIITER_RELTOL,
                NIITER_VNTOL,
            );
        }
        // The stage limit is the bound, not the direct-solve limit: a literal
        // stagemaxiter=3 fails as before.
        let mut literal = tight.to_vec();
        literal.push("stagemaxiter=3");
        assert!(deck_with_request(&limited, &literal).is_err(), "{analysis}");
    }
}

#[test]
fn stage_iteration_limit_is_separate_from_the_direct_limit() {
    use ngspice_rs::analysis::bias::{DcStrategy, solve_dc_with};
    let n = deck("v1 a 0 5\nr1 a b 1k\nd1 b 0 dm\n.model dm d(is=1e-14)\n.op");
    let circuit = RunConfig::from_netlist(&n).unwrap().circuit(&n).unwrap();
    let plan = AnalysisInput::with_arguments(
        AnalysisKind::OperatingPoint,
        [
            "maxiter=3",
            "stagemaxiter=40",
            "srcsteps=0",
            "continuation=ladder",
        ],
    )
    .resolve()
    .unwrap();
    let settings = plan.dc_settings();
    assert_eq!(settings.newton.max_iterations, 3);
    assert_eq!(
        settings
            .continuation
            .stage_max_iterations
            .map(ngspice_rs::analysis::bias::IterationLimit::get),
        Some(40)
    );
    let context = AnalysisContext::default().model_context();
    let solved = solve_dc_with(&circuit, &context, settings, &[], None, None).unwrap();
    let stages = &solved.report.stages;
    assert_eq!(stages[0].strategy, DcStrategy::Direct);
    // NIiter fails after the pass that exceeds the limit: 3 + 1.
    assert_eq!(stages[0].iterations, 4);
    assert!(
        stages[1..]
            .iter()
            .all(|s| s.strategy == DcStrategy::GminStepping)
    );
    assert!(stages[1..].iter().all(|s| s.iterations <= 41));
    assert!(stages[1..].iter().any(|s| s.iterations > 3), "{stages:?}");
    // Gmin stepping's closing zero-gmin solve is bounded by the direct limit
    // (itl1), not the stage limit: C's `spice3_gmin`/`dynamic_gmin`/`new_gmin`
    // end with `NIiter(ckt, iterlim)` (`cktop.c`).
    let closing = stages.last().unwrap();
    assert_eq!(closing.strategy, DcStrategy::GminStepping);
    assert_eq!(closing.gmin, 0.);
    assert!(closing.iterations <= 3, "{closing:?}");
    // Budget: the direct limit for the direct and closing solves, plus the
    // stage limit for every gmin stage in between, each plus the pass a
    // failing NIiter makes beyond its limit.
    assert_eq!(
        solved.report.budget,
        2 * (3 + 1) + (stages.len() - 2) * (40 + 1)
    );
    for bad in ["stagemaxiter=0", "stagemaxiter=10001", "stagemaxiter=2.5"] {
        let request = AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, [bad]);
        assert!(request.resolve().is_err(), "{bad}");
    }
}

#[test]
fn itl4_is_the_companion_newton_limit_per_timepoint() {
    let c = config(".options itl4=250").unwrap();
    assert_eq!(c.transient().itl4, Some(250));
    assert_eq!(
        c.plan(tran())
            .unwrap()
            .transient()
            .unwrap()
            .max_iterations(),
        250
    );
    // niiter.c: below 100 (including C's nominal default 10) means 100.
    for low in ["0", "3", "10", "99"] {
        let c = config(&format!(".options itl4={low}")).unwrap();
        assert_eq!(c.transient().itl4, Some(100), "itl4={low}");
    }
    assert_eq!(
        c.plan(AnalysisInput::new(AnalysisKind::OperatingPoint))
            .unwrap()
            .transient(),
        None
    );
    let deck_text = "v1 a 0 pulse(0 5 1u 1u 1u 5u 20u)\nr1 a b 1k\nd1 b 0 dm\n\
                     .model dm d(is=1e-14)\n.tran 0.1u 3u";
    let stats = |options: &str, extra: &[&str]| {
        let n = deck(&format!("{deck_text}\n{options}"));
        let c = RunConfig::from_netlist(&n).unwrap();
        let mut request = AnalysisInput::from(&n.analyses[0]);
        request
            .arguments
            .extend(extra.iter().map(|a| (*a).to_owned()));
        let request = c.plan(request).unwrap();
        let mut circuit = c.circuit(&n).unwrap();
        ngspice_rs::analysis::companion_transient(&mut circuit, &request, &c.context())
            .unwrap()
            .1
    };
    // As in C (identical iteration/timepoint/rejection counts for itl4 =
    // 1/3/10/99), deck values below 100 change nothing.
    let default = stats("", &[]);
    for low in ["1", "3", "10", "99", "100"] {
        assert_eq!(stats(&format!(".options itl4={low}"), &[]), default);
    }
    // The limit is a real per-timepoint bound: the literal request knob below
    // C's floor forces more rejected trials (dctran.c cuts the step by 8 on
    // non-convergence), and explicit request arguments beat the deck.
    let starved = stats(".options itl4=200", &["tranmaxiter=3"]);
    assert!(
        starved.rejected > default.rejected,
        "{starved:?} vs {default:?}"
    );
    for bad in [".options itl4", ".options itl4=10001"] {
        assert!(!config(bad).unwrap_err().is_not_yet_ported(), "{bad}");
    }
    // IF_INTEGER rounds 2.5 to 3, which NIiter raises to 100.
    assert_eq!(
        config(".options itl4=2.5").unwrap().transient().itl4,
        Some(100)
    );
}

#[test]
fn xmu_weights_the_companion_trapezoidal_rule() {
    let c = config(".options xmu=0.3").unwrap();
    assert_eq!(c.transient().xmu, Some(0.3));
    assert_eq!(
        c.plan(tran()).unwrap().transient().unwrap().xmu(),
        c.transient().xmu.unwrap()
    );
    let rc = "v1 a 0 pulse(0 1 0 1n 1n 1 2)\nr1 a b 1k\nc1 b 0 1u\n.tran 10u 2m";
    let at = |plot: &Plot, t: f64| {
        let time = plot.column("time").unwrap();
        let i = time.iter().position(|v| v.re >= t).unwrap();
        plot.value("v(b)", i).unwrap().re
    };
    let default = simulate(rc).unwrap();
    let explicit = simulate(&format!("{rc}\n.options xmu=0.5")).unwrap();
    assert_eq!(default, explicit);
    let damped = simulate(&format!("{rc}\n.options xmu=0.2")).unwrap();
    assert_ne!(damped, default);
    // Both remain accurate RC charging curves.
    for plot in [&default, &damped] {
        close(at(plot, 1e-3), 1. - (-1f64).exp(), 0.02, 0.);
    }
    for bad in [".options xmu=0.6", ".options xmu=-0.1", ".options xmu=abc"] {
        assert!(!config(bad).unwrap_err().is_not_yet_ported(), "{bad}");
    }
}

#[test]
fn itl6_is_srcsteps() {
    let c = config(".options itl6=3").unwrap();
    assert_eq!(c.dc().srcsteps, Some(3));
    assert_eq!(c.applied()[0].name, "itl6");
    assert_eq!(
        c.plan(AnalysisInput::new(AnalysisKind::OperatingPoint))
            .unwrap()
            .dc_settings()
            .continuation
            .schedule,
        ngspice_rs::analysis::bias::ContinuationSchedule::Ngspice(
            ngspice_rs::analysis::bias::NgspiceStepping::new(1, 3, 10., 50).unwrap()
        )
    );
    // One setting: last occurrence wins across both spellings.
    assert_eq!(
        config(".options itl6=3 srcsteps=5").unwrap().dc().srcsteps,
        Some(5)
    );
    assert_eq!(
        config(".options srcsteps=5 itl6=0").unwrap().dc().srcsteps,
        Some(0)
    );
    assert!(config(".options itl6=1001").is_err());
}

#[test]
fn iteration_and_stepping_limits_preserve_c_integer_rounding() {
    for (name, limit) in [
        ("itl1", 10_000),
        ("itl2", 10_000),
        ("itl4", 10_000),
        ("itl6", 1_000),
        ("srcsteps", 1_000),
        ("gminsteps", 100),
    ] {
        // A 100-step gmin ladder requires a near-unity factor to stay below
        // MAX_GMIN; the default factor of 10 would overflow its conductance.
        let factor = if name == "gminsteps" {
            " gminfactor=1.01"
        } else {
            ""
        };
        for valid in [limit.to_string(), format!("{limit}.49")] {
            assert!(
                config(&format!(".options {name}={valid}{factor}")).is_ok(),
                "{name}={valid}"
            );
        }
        for invalid in [(limit + 1).to_string(), format!("{limit}.5")] {
            assert!(
                config(&format!(".options {name}={invalid}{factor}")).is_err(),
                "{name}={invalid}"
            );
        }
    }
}

#[test]
fn existing_effective_options_are_still_applied() {
    let c = config(
        ".options temp=50 tnom=20 reltol=1e-4 vntol=2u abstol=3p chgtol=1e-13 trtol=5 \
         method=gear maxord=1 srcsteps=2 gminsteps=4 gminfactor=5",
    )
    .unwrap();
    assert_eq!(c.context().temperature, 50.);
    assert_eq!(c.context().nominal_temperature, 20.);
    let t = c.transient();
    assert_eq!(
        (c.context().reltol, t.vntol, t.abstol, t.chgtol, t.trtol),
        (1e-4, Some(2e-6), Some(3e-12), Some(1e-13), Some(5.))
    );
    assert_eq!(
        (c.method(), c.maxord()),
        (Some(ngspice_rs::analysis::IntegrationFamily::Gear), Some(1))
    );
    assert_eq!(
        (c.dc().srcsteps, c.dc().gminsteps, c.dc().gminfactor),
        (Some(2), Some(4), Some(5.))
    );
    assert_eq!(c.applied().len(), 12);
    assert!(c.ignored().is_empty());
}

#[test]
fn every_analysis_with_an_operating_point_gets_the_deck_tolerances() {
    // C's CKTop reads CKTreltol/CKTvoltTol/CKTabstol for every analysis that
    // solves an operating point; `.disto` and `.sens` once dropped them, so
    // their operating points ran at the default reltol (the disto_* and
    // sens_diode goldens tighten RELTOL and depend on it).
    let c = config(".options reltol=1e-9 vntol=2u abstol=3p").unwrap();
    for kind in [
        AnalysisKind::OperatingPoint,
        AnalysisKind::DcSweep,
        AnalysisKind::Ac,
        AnalysisKind::Noise,
        AnalysisKind::Distortion,
        AnalysisKind::PoleZero,
        AnalysisKind::Sensitivity,
        AnalysisKind::TransferFunction,
        AnalysisKind::SParameter,
        AnalysisKind::PeriodicSteadyState,
        AnalysisKind::HarmonicBalance,
    ] {
        let request = c.plan(input_for(kind)).unwrap();
        let newton = &request.dc_settings().newton;
        assert_eq!(newton.reltol, c.context().reltol, "{kind:?}");
        assert_eq!(newton.vntol, c.transient().vntol.unwrap(), "{kind:?}");
        assert_eq!(newton.abstol, c.transient().abstol.unwrap(), "{kind:?}");
    }
}

// ---- documented no-ops ------------------------------------------------------

#[test]
fn print_control_flags_are_documented_no_ops() {
    for flag in [
        "acct",
        "noacct",
        "list",
        "nomod",
        "nopage",
        "node",
        "opts",
        "noinit",
        "norefvalue",
    ] {
        let c = config(&format!(".options {flag}")).unwrap();
        assert!(c.applied().is_empty(), "{flag}");
        assert_eq!(c.ignored().len(), 1, "{flag}");
        assert_eq!(c.ignored()[0].name, flag);
        assert_eq!(c.ignored()[0].value, None);
        assert!(c.ignored()[0].reason.contains("print"), "{flag}");
        // Nothing reaches any request.
        for request in [tran(), AnalysisInput::new(AnalysisKind::OperatingPoint)] {
            assert_eq!(
                c.plan(request.clone()).unwrap(),
                request.resolve().unwrap(),
                "{flag}"
            );
        }
        let error = config(&format!(".options {flag}=1")).unwrap_err();
        assert!(matches!(error, SpiceError::Parse { .. }), "{flag}: {error}");
    }
    // A run with every flag is identical to one without.
    let rc = "v1 a 0 1\nr1 a b 1k\nc1 b 0 1u\n.tran 10u 1m";
    assert_eq!(
        simulate(&format!(
            "{rc}\n.options acct noacct list nomod nopage node opts noinit norefvalue"
        ))
        .unwrap(),
        simulate(rc).unwrap()
    );
}

#[test]
fn options_c_ignores_are_documented_no_ops() {
    for (name, value) in [
        ("itl3", "4"),
        ("itl5", "5000"),
        ("cptime", "1e3"),
        ("limtim", "2"),
        ("limpts", "201"),
        ("lvlcod", "1"),
        ("lvltim", "2"),
    ] {
        let c = config(&format!(".options {name}={value}")).unwrap();
        assert!(c.applied().is_empty(), "{name}");
        assert_eq!(c.ignored()[0].value.as_deref(), Some(value));
        assert!(
            c.ignored()[0].reason.contains("ignored by ngspice"),
            "{name}"
        );
        assert!(config(&format!(".options {name}=abc")).is_err(), "{name}");
        assert!(config(&format!(".options {name}")).is_err(), "{name}");
    }
    // Validated as C's IF_INTEGER: 2.5 rounds, a word does not parse.
    assert!(config(".options itl3=2.5").is_ok());
    assert!(config(".options cptime=2.5").is_ok());
}

#[test]
fn integral_option_storage_preserves_the_validated_c_range() {
    // IF_INTEGER is a signed C integer even when the accepting field is
    // unsigned. Values above i32::MAX must not silently truncate or saturate.
    assert!(config(".options itl3=2147483647").is_ok());
    for outside in ["2147483648", "4294967295", "-2147483648"] {
        assert!(
            config(&format!(".options itl3={outside}")).is_err(),
            "out-of-C-range value: {outside}"
        );
    }

    let c = config(".options nfreqs=101 nperiods=100 polydegree=16 fourgridsize=100000 itl1=10000")
        .unwrap();
    let four = c.fourier_settings().unwrap();
    assert_eq!(
        (four.nfreqs, four.nperiods, four.polydegree, four.gridsize),
        (101, 100, 16, 100_000)
    );
    assert_eq!(c.dc().itl1, Some(10_000));
    for outside in [
        "nfreqs=102",
        "nperiods=101",
        "polydegree=17",
        "fourgridsize=100001",
        "itl1=10001",
    ] {
        assert!(config(&format!(".options {outside}")).is_err(), "{outside}");
    }
}

#[test]
fn unread_frontend_variables_and_bypass_are_documented_no_ops() {
    for options in [
        ".options post",
        ".options post=2",
        ".options ingold=2",
        ".options post=csdf",
    ] {
        let c = config(options).unwrap();
        assert_eq!(c.ignored().len(), 1, "{options}");
        assert!(c.ignored()[0].reason.contains("nothing reads"), "{options}");
    }
    let c = config(".options bypass=0").unwrap();
    assert!(c.ignored()[0].reason.contains("never bypasses"));
    assert!(config(".options bypass=1").unwrap_err().is_not_yet_ported());
    assert!(!config(".options bypass").unwrap_err().is_not_yet_ported());
}

/// A 1e-17 S diagonal at node `a` with the lowest Markowitz product, coupled
/// by VCCSs to `b`: SPARSE takes it as the first pivot only when `pivrel`
/// lets it, and then loses `v(a)` to cancellation. ngspice with
/// `.options sparse` gives v(a) = -3.45016 by default and -9.89 with
/// `pivrel=1e-30 pivtol=0`.
const TINY_PIVOT_DECK: &str = "r1 a 0 1e17
g1 a 0 b 0 -0.7
g2 b 0 a 0 -1.3
r2 b 0 1.7
r3 b c 1.1
r4 c 0 2.9
r5 c d 1.3
r6 d 0 3.1
r7 b d 0.9
r8 c e 1
r9 e 0 1
r10 d e 1";

#[test]
fn pivot_thresholds_reach_every_circuit_factorization() {
    let c = config(".options pivtol=1e-10 pivrel=0.5").unwrap();
    assert_eq!(c.context().pivot.absolute, 1e-10);
    assert_eq!(c.context().pivot.relative, 0.5);
    assert_eq!(c.context().model_context().pivot, c.context().pivot);
    assert_eq!(c.applied().len(), 2);
    let defaults = config("").unwrap().context().pivot;
    assert_eq!((defaults.absolute, defaults.relative), (1e-13, 1e-3));
    assert_eq!(
        config(".param p=1e-12\n.options pivtol={p}")
            .unwrap()
            .context()
            .pivot
            .absolute,
        1e-12
    );
    // spOrderAndFactor would silently fall back to its default here.
    for bad in [
        ".options pivtol=-1e-13",
        ".options pivrel=0",
        ".options pivrel=1.5",
        ".options pivrel=abc",
        ".options pivtol",
    ] {
        let error = config(bad).unwrap_err();
        assert!(!error.is_not_yet_ported(), "{bad}: {error}");
    }

    let exact = -3.450_156_379_816_799;
    let run = |sources: &str, analysis: &str, options: &str| {
        let plot = simulate(&format!(
            "{TINY_PIVOT_DECK}\n{sources}\n{options}\n{analysis}"
        ))
        .unwrap();
        plot.value("v(a)", plot.point_count() - 1).unwrap().re
    };
    let dc = "i1 0 a 1.1\ni2 0 b 2.3";
    let ac = "i1 0 a dc 0 ac 1.1\ni2 0 b dc 0 ac 2.3";
    let loose = ".options pivrel=1e-30 pivtol=0";
    for (sources, analysis) in [
        (dc, ".op"),
        (dc, ".dc i1 1.1 1.1 1"),
        (ac, ".ac lin 1 1k 1k"),
        (dc, ".tran 1u 2u"),
    ] {
        close(run(sources, analysis, ""), exact, 1e-9, 0.);
        let lost = run(sources, analysis, loose);
        assert!((lost - exact).abs() > 0.1, "{analysis}: v(a) = {lost}");
    }
}

#[test]
fn unknown_and_unported_options_still_fail() {
    assert!(config(".options seed=5").is_ok());
    assert!(config(".options rndseed=5").is_ok());
    for invalid in [
        ".options seed=0",
        ".options seed=1.5",
        ".options seed=random",
        ".options seed={x}",
    ] {
        assert!(config(invalid).is_err(), "{invalid}");
    }
    for unknown in [".options bogus=1", ".options postt", ".options itl7=1"] {
        assert!(
            matches!(config(unknown), Err(SpiceError::Parse { .. })),
            "{unknown}"
        );
    }
    for pending in [
        ".options gshunt=1e-12",
        ".options numdgt=8",
        ".options savecurrents",
        ".options scale=1u",
        ".options klu",
    ] {
        assert!(
            config(pending).unwrap_err().is_not_yet_ported(),
            "{pending}"
        );
    }
}

// ---- expression values (#107) -------------------------------------------------

#[test]
fn braced_and_quoted_option_values_evaluate_against_params() {
    let c = config(
        ".param t=40 g=1u half=0.5\n.options temp={t+10} tnom='t-5' gmin={g/10} \
         reltol={2*1m} itl1={10*15} xmu={half/2} cptime='half'",
    )
    .unwrap();
    assert_eq!(c.context().temperature, 50.);
    assert_eq!(c.context().nominal_temperature, 35.);
    close(c.context().gmin, 1e-7, 1e-12, 0.);
    close(c.context().reltol, 2e-3, 1e-12, 0.);
    assert_eq!(c.dc().itl1, Some(150));
    assert_eq!(c.transient().xmu, Some(0.25));
    assert_eq!(c.applied()[0].value, "{t+10}");
    assert_eq!(c.applied()[1].value, "'t-5'");
    assert_eq!(c.ignored()[0].value.as_deref(), Some("'half'"));
    // Parameters defined after the option card are visible (one root scope).
    let late = config(".options temp={t}\n.param t=60").unwrap();
    assert_eq!(late.context().temperature, 60.);
}

#[test]
fn option_expression_failures_are_explicit() {
    let undefined = config(".options temp={nope}").unwrap_err();
    assert!(matches!(undefined, SpiceError::Parse { .. }), "{undefined}");
    assert!(
        undefined.to_string().contains("option 'temp'"),
        "{undefined}"
    );
    // Range checks apply to the evaluated value.
    assert!(config(".param n=10001\n.options itl1={n}").is_err());
    assert!(config(".param n=-1\n.options reltol={n}").is_err());
    assert!(config(".options temp={-300}").is_err());
    // method takes a word.
    let error = config(".param m=1\n.options method={m}").unwrap_err();
    assert!(error.to_string().contains("method"), "{error}");
    // Unknown and unported names fail as such, before evaluation.
    assert!(matches!(
        config(".options bogus={nope}"),
        Err(SpiceError::Parse { message, .. }) if message.contains("unknown option")
    ));
    assert!(
        config(".options gshunt={nope}")
            .unwrap_err()
            .is_not_yet_ported()
    );
    // Without a parameter scope the value cannot be evaluated.
    let n = deck("r1 a 0 1k\n.param t=40\n.options temp={t}");
    let error = RunConfig::from_options(&n.options, &RunOverrides::default()).unwrap_err();
    assert!(matches!(error, SpiceError::Unsupported { .. }), "{error}");
}

#[test]
fn removed_diffsol_backend_is_rejected_whatever_the_deck_options() {
    // RunConfig forwards deck options without interpreting `backend=`; the
    // `.tran` dispatcher refuses the removed adaptive BDF backend explicitly,
    // so no option can be reported as honoured by it.
    for options in [
        "",
        ".options itl2=300",
        ".options itl4=20 xmu=0.3",
        ".options method=gear maxord=2 trtol=3 chgtol=1e-13",
    ] {
        let error = simulate(&format!(
            "v1 a 0 dc 1\nr1 a b 1k\nc1 b 0 1u\n{options}\n.tran 1u 10u backend=diffsol"
        ))
        .unwrap_err();
        assert!(
            matches!(error, SpiceError::Unsupported { .. })
                && error.to_string().contains("backend=diffsol"),
            "{options}: {error}"
        );
    }
}
