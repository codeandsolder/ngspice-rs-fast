//! Scalar measurements: C's two-pass ordering, deck parameters, and atomic errors.
use ngspice_conformance::measurement::{self, ActualSection, Analysis};
use ngspice_conformance::ngspice::{Ngspice, Scratch};
use ngspice_rs::analysis::{Plot, PlotFlags, Variable, measure};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::eval::ParamScope;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, Complex};
use std::fs;
use std::path::Path;

fn evaluate(body: &str) -> ngspice_rs::primitives::SpiceResult<Vec<measure::Measurement>> {
    let parsed =
        Parser::new().parse_deck_with_output(&parse_deck_text(Path::new("scalar.cir"), body))?;
    let scope = ParamScope::for_netlist(&parsed.netlist)?;
    let mut plot = Plot::builder("tran1", "Transient Analysis", PlotFlags::Real);
    plot.push_variable(Variable::new("time", "time"));
    plot.push_variable(Variable::new("v(out)", "voltage"));
    let mut plot = plot.build();
    for t in [0., 0.5, 1.] {
        plot.push_point(&[Complex::real(t), Complex::real(2. * t)])?;
    }
    measure::resolve_with_scope(&plot, AnalysisKind::Transient, &parsed.measurements, &scope)
}

const DECK: &str = "Scalar\n.param scale=3\n.func twice(x) {2*x}\n.measure tran scaled param='peak*scale'\n.measure tran peak max v(out)\n.measure tran chained expr='twice(scaled)+1'\n.end\n";

#[test]
fn scalar_measurements_use_two_passes_but_return_source_order() {
    let results = evaluate(DECK).unwrap();
    assert_eq!(
        results.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
        ["scaled", "peak", "chained"]
    );
    assert_eq!(
        results.iter().map(|r| r.value).collect::<Vec<_>>(),
        [6., 2., 13.]
    );
    assert_eq!(results[0].unit, "scalar");
    for expression in ["missing", "1/0", "sqrt(-1)", "later+1"] {
        assert!(
            evaluate(&format!(
                "Bad\n.measure tran first param='{expression}'\n.measure tran later param=1\n.end\n"
            ))
            .is_err()
        );
    }
}

#[test]
fn result_overlay_shadows_the_base_and_prior_results_case_insensitively() {
    let values = evaluate(
        "Overlay\n.param peak=9 base=3\n.func shift(x) {x+peak+base}\n\
         .measure tran first param='shift(1)'\n\
         .measure tran peak max v(out)\n\
         .measure tran sample find v(out) at={peak/4}\n\
         .measure tran PEAK find v(out) at=.75\n\
         .measure tran peak param='PEAK+base'\n\
         .measure tran latest param='shift(0)'\n.end\n",
    )
    .unwrap();
    // The vector pass sees peak=2, then PEAK=1.5. The scalar pass sees
    // the last vector value, and a self-named scalar can read its prior value.
    assert_eq!(
        values.iter().map(|result| result.value).collect::<Vec<_>>(),
        [5.5, 2., 1., 1.5, 4.5, 7.5],
    );
    assert_eq!(
        values
            .iter()
            .map(|result| result.name.to_ascii_lowercase())
            .collect::<Vec<_>>(),
        ["first", "peak", "sample", "peak", "peak", "latest"],
    );
}

#[test]
fn many_result_records_keep_early_and_root_bindings_accessible() {
    use std::fmt::Write as _;
    use std::time::Instant;

    // 20,000 names refer to the earliest result and an outer parameter
    // through a user function, the previous worst-case ancestor traversal.
    let mut text = String::from(
        "Long measurement sequence\n.param base=3\n.func shift(x) {x+seed+base}\n.meas tran seed max v(out)\n",
    );
    for i in 0..20_000_u32 {
        writeln!(text, ".meas tran r{i} param='shift({i})'").unwrap();
    }
    text.push_str(".meas tran last param='r19999+seed+base'\n.end\n");
    let start = Instant::now();
    let records = evaluate(&text).unwrap();
    eprintln!(
        "20,002 measurement records including parse: {:?}",
        start.elapsed()
    );
    assert_eq!(records.len(), 20_002);
    assert_eq!(records[0].value, 2.);
    for (i, row) in (0..20_000_u32).zip(&records[1..20_001]) {
        assert_eq!(row.name, format!("r{i}"));
        assert_eq!(row.value, f64::from(i) + 5.);
    }
    assert_eq!(records[20_001].value, 20_009.);
}

#[test]
fn scalar_measurements_round_trip_and_malformed_tails_fail() {
    let parser = Parser::new();
    let parsed = parser
        .parse_deck_with_output(&parse_deck_text(Path::new("scalar.cir"), DECK))
        .unwrap();
    let written = ngspice_rs::netlist::write_netlist(&parsed.netlist).unwrap();
    let again = parser
        .parse_deck_with_output(&parse_deck_text(Path::new("again.cir"), &written))
        .unwrap();
    assert!(ngspice_rs::netlist::semantic_eq(
        &parsed.netlist,
        &again.netlist
    ));
    for card in ["param=", "param='1+2' garbage", "param='1+'"] {
        assert!(
            parser
                .parse_deck_with_output(&parse_deck_text(
                    Path::new("bad.cir"),
                    &format!("Bad\n.meas tran x {card}\n.end\n")
                ))
                .is_err()
        );
    }
}

fn compare_live(tag: &str, text: &str) {
    let scratch = Scratch::new(tag).expect("scratch directory");
    let deck = scratch.path().join("deck.cir");
    let raw = scratch.path().join("out.raw");
    fs::write(&deck, text).unwrap();
    let binary = Ngspice::pinned().expect("pinned C reference");
    let output = binary
        .run(scratch.path(), &["-b", "deck.cir"])
        .expect("run C");
    output
        .require_success(tag)
        .unwrap_or_else(|error| panic!("{error}"));
    let report = measurement::parse(
        &String::from_utf8_lossy(&output.stdout),
        &String::from_utf8_lossy(&output.stderr),
    )
    .unwrap_or_else(|error| panic!("{tag}: {error}\n{}", output.transcript()));
    let ours = ngspice_rs::cli::simulate::run(&deck, &raw, true).unwrap();
    let [plot] = ours.plots.as_slice() else {
        panic!("{tag}: expected one analysis plot");
    };
    assert_eq!(plot.analysis, AnalysisKind::Transient);
    let actual = [ActualSection {
        analysis: Analysis::Tran,
        run: 0,
        values: plot
            .measurements
            .iter()
            .map(|row| (row.name.clone(), row.value))
            .collect(),
    }];
    measurement::compare(&report, &actual, |_| 1e-5_f64.next_down())
        .unwrap_or_else(|error| panic!("{tag}: {error}\n{}", output.transcript()));
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN"]
fn scalar_measurements_match_live_c() {
    compare_live(
        "m9-meas-expr",
        &DECK.replace("chained expr=", "chained param=").replace(
            "Scalar\n",
            "Scalar\nV1 out 0 pwl(0 0 1 2)\nR1 out 0 1k\n.tran .01 1\n",
        ),
    );
}

#[test]
fn invalid_scalar_expression_preserves_existing_rawfile() {
    let scratch = Scratch::new("m9-meas-atomic").unwrap();
    let dir = scratch.path();
    let deck = dir.join("deck.cir");
    let raw = dir.join("out.raw");
    fs::write(&raw, b"previous output").unwrap();
    fs::write(&deck, "Bad scalar\nV1 out 0 1\nR1 out 0 1k\n.tran .01 1\n.measure tran invalid param='missing+1'\n.end\n").unwrap();
    assert!(ngspice_rs::cli::simulate::run(&deck, &raw, true).is_err());
    assert_eq!(fs::read(&raw).unwrap(), b"previous output");
}

#[test]
fn numeric_setter_expressions_resolve_parameters_and_prior_results() {
    let deck = "Setters\n.param half=.5 one=1\n.measure tran peak max v(out)\n.measure tran sample find v(out) at={half}\n.measure tran crossing when v(out)='peak/2' rise={one} td={half/2}\n.measure tran area integ v(out) from={half} to={one}\n.end\n";
    let results = evaluate(deck).unwrap();
    assert_eq!(
        results.iter().map(|r| r.value).collect::<Vec<_>>(),
        [2., 1., 0.5, 0.75]
    );
    let parser = Parser::new();
    let parsed = parser
        .parse_deck_with_output(&parse_deck_text(Path::new("setters.cir"), deck))
        .unwrap();
    let written = ngspice_rs::netlist::write_netlist(&parsed.netlist).unwrap();
    let again = parser
        .parse_deck_with_output(&parse_deck_text(Path::new("again.cir"), &written))
        .unwrap();
    assert!(ngspice_rs::netlist::semantic_eq(
        &parsed.netlist,
        &again.netlist
    ));
    for setter in ["at={missing}", "at={1/0}", "at={sqrt(-1)}"] {
        assert!(evaluate(&format!("Bad\n.meas tran x find v(out) {setter}\n.end\n")).is_err());
    }
    assert!(evaluate("Bad count\n.meas tran x when v(out)={1} rise={1.5}\n.end\n").is_err());
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN"]
fn numeric_setter_integration_expressions_match_live_c() {
    compare_live(
        "m9-setters",
        "Setters\n.param half=.5 one=1\nV1 out 0 pwl(0 0 1 2)\nR1 out 0 1k\n.tran .01 1\n.measure tran sample find v(out) at={half}\n.measure tran crossing when v(out)={one} rise={one} td={half/2}\n.measure tran area integ v(out) from={half} to={one}\n.end\n",
    );
}

#[test]
fn numeric_setter_expressions_share_the_measurement_scope() {
    let results = evaluate("Setters\n.param lo=.25 hi=.75 target=1 count=1\n.meas tran peak max v(out) from={lo} to='hi'\n.meas tran cross when v(out)={target} rise={count} td={lo}\n.meas tran at_cross find v(out) at={cross}\n.meas tran delay trig v(out) val={target} rise='count' td={lo} targ at={hi}\n.end\n").unwrap();
    assert_eq!(
        results.iter().map(|r| r.value).collect::<Vec<_>>(),
        [1., 0.5, 1., 0.25]
    );
    for setter in ["rise={0}", "rise={1.5}", "rise={missing}", "td={1/0}"] {
        assert!(
            evaluate(&format!(
                "Bad\n.meas tran x when v(out)={{1}} {setter}\n.end\n"
            ))
            .is_err()
        );
    }
    let text = "Round trip\n.meas tran x max v(out) from={.2} to='.8'\n.end\n";
    let parser = Parser::new();
    let parsed = parser
        .parse_deck_with_output(&parse_deck_text(Path::new("a.cir"), text))
        .unwrap();
    let written = ngspice_rs::netlist::write_netlist(&parsed.netlist).unwrap();
    let again = parser
        .parse_deck_with_output(&parse_deck_text(Path::new("b.cir"), &written))
        .unwrap();
    assert!(ngspice_rs::netlist::semantic_eq(
        &parsed.netlist,
        &again.netlist
    ));
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN"]
fn numeric_setter_expressions_match_live_c() {
    compare_live(
        "m9-meas-setters",
        "Setters\n.param lo=.25 hi=.75 target=1 count=1\nV1 out 0 pwl(0 0 1 2)\nR1 out 0 1k\n.tran .01 1\n.meas tran peak max v(out) from={lo} to='hi'\n.meas tran cross when v(out)={target} rise={count} td={lo}\n.meas tran at_cross find v(out) at={target/2}\n.meas tran delay trig v(out) val={target} rise='count' td={lo} targ at={hi}\n.end\n",
    );
}
