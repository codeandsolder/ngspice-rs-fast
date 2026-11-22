//! Opt-in #43 live C comparison for `.measure`/`.meas`.
//!
//! Run it explicitly with the C reference binary:
//!
//! ```text
//! NGSPICE_BIN=/abs/path/to/ngspice \
//!   cargo test -p ngspice-rs --test c_measure_reference -- --ignored
//! ```
//!
//! Ordinary `cargo test` runs never need C: the test is `#[ignore]`d. C prints
//! every measurement with five fractional digits (`%.5e`, `measure_get_precision()`),
//! which bounds how closely the two can be compared; the tolerance below is a
//! few of C's printed ULPs, not a relaxed physical bound.
//!
//! Nothing here reads or rewrites a committed golden.

use std::fs;
use std::path::Path;

use ngspice_conformance::measurement::{self, ActualSection, Analysis};
use ngspice_conformance::ngspice::{Ngspice, Scratch};

use ngspice_rs::analysis::{RunConfig, measure};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, Real};

/// The deck's `.measure` results, evaluated over the port's own plot.
fn port_results(deck: &str) -> Vec<(String, Real)> {
    let text = format!("Measure oracle\n{deck}\n.end\n");
    let parsed = Parser::new()
        .parse_deck_with_output(&parse_deck_text(Path::new("measure.cir"), &text))
        .expect("the deck parses");
    let netlist = &parsed.netlist;
    let card = netlist.analyses.first().expect("one analysis");
    let config = RunConfig::from_netlist(netlist).expect("the deck configures");
    let request = config.plan_for(card).expect("the analysis request");
    let mut circuit = config.circuit(netlist).expect("the circuit builds");
    let plot = request
        .run(&mut circuit, &config.context())
        .expect("the run succeeds");
    assert_eq!(card.kind, request.kind());
    measure::resolve(&plot, card.kind, &parsed.measurements)
        .expect("every measurement resolves")
        .into_iter()
        .map(|result| (result.name, result.value))
        .collect()
}

/// Runs both engines and checks the complete result set, with the original
/// few-printed-ULP bound and the actual requested analysis.
fn compare(tag: &str, kind: AnalysisKind, deck: &str) {
    let scratch = Scratch::new(tag).expect("scratch directory");
    fs::write(
        scratch.path().join("measure.cir"),
        format!("Measure oracle\n{deck}\n.end\n"),
    )
    .expect("write the deck");
    let binary = Ngspice::pinned().expect("pinned C reference");
    let output = binary
        .run(scratch.path(), &["-b", "measure.cir"])
        .expect("run C");
    output
        .require_success(tag)
        .unwrap_or_else(|error| panic!("{error}"));
    let theirs = measurement::parse(
        &String::from_utf8_lossy(&output.stdout),
        &String::from_utf8_lossy(&output.stderr),
    )
    .unwrap_or_else(|error| panic!("{tag}: {error}\n{}", output.transcript()));
    let analysis = Analysis::keyword(kind.as_str()).expect("supported measurement analysis");
    let actual = [ActualSection {
        analysis,
        run: 0,
        values: port_results(deck),
    }];
    measurement::compare(&theirs, &actual, |reference| 3e-5 * reference.abs() + 1e-9)
        .unwrap_or_else(|error| panic!("{tag}: {error}\n{}", output.transcript()));
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; intentional failed and repeated C measurements"]
fn real_reports_keep_failed_cards_and_repeated_runs_separate() {
    use ngspice_conformance::measurement::Outcome;
    let scratch = Scratch::new("measure-diagnostics").unwrap();
    let binary = Ngspice::pinned().unwrap();
    let run = |text: &str| {
        fs::write(scratch.path().join("probe.cir"), text).unwrap();
        let output = binary.run(scratch.path(), &["-b", "probe.cir"]).unwrap();
        output.require_success("measurement probe").unwrap();
        measurement::parse(
            &String::from_utf8_lossy(&output.stdout),
            &String::from_utf8_lossy(&output.stderr),
        )
        .unwrap_or_else(|error| panic!("{error}\n{}", output.transcript()))
    };
    let repeated = run(
        "Repeated measurements\nv1 in 0 0\nr1 in 0 1k\n.dc v1 0 1 .1\n.meas dc stack max v(in)\n.control\nrun\nrun\nquit\n.endc\n.end\n",
    );
    let actual = [
        ActualSection {
            analysis: Analysis::Dc,
            run: 0,
            values: vec![("stack".into(), 1.)],
        },
        ActualSection {
            analysis: Analysis::Dc,
            run: 1,
            values: vec![("stack".into(), 1.)],
        },
    ];
    measurement::compare(&repeated, &actual, |_| 1e-9).unwrap();
    let failures = run(
        "Failed measurements\nv1 in 0 0\nr1 in 0 1k\n.dc v1 0 1 .1\n.meas dc missing param='unknown+1'\n.meas dc nohit when v(in)=2\n.meas dc duplicate max v(in)\n.meas dc duplicate min v(in)\n.end\n",
    );
    assert_eq!(failures.sections.len(), 1);
    let rows = &failures.sections[0].rows;
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].outcome, Outcome::Failed);
    assert_eq!(rows[1].name, "duplicate");
    assert_eq!(rows[2].name, "duplicate");
    assert_eq!(failures.failures.len(), 1);
    assert_eq!(failures.failures[0].name, "nohit");
    assert_eq!(failures.failures[0].run, Some(0));
    assert!(measurement::compare(&failures, &actual[..1], |_| 1e-9).is_err());
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; runs a temporary C deck out of process"]
fn an_rc_step_measures_delay_rise_and_statistics_like_c() {
    compare(
        "tran",
        AnalysisKind::Transient,
        "v1 in 0 pulse(0 1 0 1n 1n 1 2)\nr1 in out 1k\nc1 out 0 1u\n.tran 1u 5m\n\
         .meas tran tdelay trig v(in) val=0.5 rise=1 targ v(out) val=0.5 rise=1\n\
         .meas tran trise trig v(out) val=0.1 rise=1 targ v(out) val=0.9 rise=1\n\
         .meas tran trise10 trig v(out) val=0.1 rise=1 targ v(out) val=0.9 cross=1\n\
         .meas tran vmax max v(out)\n\
         .meas tran vmin min v(out)\n\
         .meas tran vavg avg v(out)\n\
         .meas tran vrms rms v(out)\n\
         .meas tran iinteg integ v(out) from=1m to=4m\n\
         .meas tran vat find v(out) at=1m",
    );
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; runs a temporary C deck out of process"]
fn an_ac_sweep_measures_magnitude_db_and_a_frequency_like_c() {
    compare(
        "ac",
        AnalysisKind::Ac,
        "v1 in 0 ac 1\nr1 in out 1k\nc1 out 0 1u\n.ac dec 10 10 10k\n\
         .meas ac f_at find vm(out) at=1591\n\
         .meas ac vmax max vm(out)\n\
         .meas ac vdc max vdb(out)\n\
         .meas ac vavg avg vm(out)\n\
         .meas ac vrms rms vm(out)",
    );
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; runs a temporary C deck out of process"]
fn a_dc_sweep_measures_values_and_integrals_like_c() {
    compare(
        "dc",
        AnalysisKind::DcSweep,
        "v1 in 0 0\nr1 in out 1k\nr2 out 0 1k\n.dc v1 0 2 0.5\n\
         .meas dc vmid find v(out) at=1\n\
         .meas dc vmax max v(out)\n\
         .meas dc vmin min v(out)\n\
         .meas dc vavg avg v(out)\n\
         .meas dc vint integ v(out)",
    );
}

#[test]
#[ignore = "requires NGSPICE_BIN"]
fn sp_measurements_match_c() {
    compare(
        "sp",
        AnalysisKind::SParameter,
        "v1 a 0 dc 0 portnum 1\nv2 b 0 dc 0 portnum 2\nr1 a b 50\nr2 a 0 100\nr3 b 0 100\nc1 b 0 1n\n.sp dec 10 1k 1meg\n .meas sp transmission max vm(S_2_1)\n.meas sp atfreq find vr(S_2_1) at=10k\n.print sp all",
    );
}

#[test]
#[ignore = "requires NGSPICE_BIN; extended measurement variants"]
fn extended_measurements_match_c() {
    compare(
        "extended",
        AnalysisKind::DcSweep,
        "V1 in 0 0\nR1 in 0 1k\n.dc v1 0 2 .25\n.meas dc crossing WHEN v(in)=.5 rise=1 td=.1\n.meas dc found FIND v(in) WHEN v(in)=.75\n.meas dc slope DERIV v(in) AT=1\n.meas dc slopewhen DERIV v(in) WHEN v(in)=1.25\n.meas dc low MIN_AT v(in)\n.meas dc high MAX_AT v(in)\n.meas dc range PP v(in)\n.meas dc area INTEG v(in)\n.meas dc rms RMS v(in)",
    );
}

#[test]
#[ignore = "requires NGSPICE_BIN; phase/gain margins"]
fn margins_match_c() {
    compare(
        "margins",
        AnalysisKind::Ac,
        "V1 in 0 0 ac 1\nE1 drive 0 in 0 10\nR1 drive a 1k\nC1 a 0 1u\nE2 b 0 a 0 1\nR2 b c 1k\nC2 c 0 1u\nE3 d 0 c 0 1\nR3 d out 1k\nC3 out 0 1u\n.ac dec 100 1 1meg\n.meas ac pm PHASE_MARGIN v(out)\n.meas ac gm GAIN_MARGIN v(out)",
    );
}

#[test]
#[ignore = "requires NGSPICE_BIN; independent trigger and target TD"]
fn trigger_target_delays_have_independent_search_origins() {
    compare(
        "independent-td",
        AnalysisKind::Transient,
        "V1 out 0 pulse(0 1 0 1u 1u 400u 1m)\nR1 out 0 1k\n.tran 1u 4m\n.meas tran delay TRIG v(out) VAL=.5 RISE=1 TD=.5m TARG v(out) VAL=.5 RISE=1 TD=1.5m\n.meas tran whenlate WHEN v(out)=.5 RISE=1 TD=2.5m",
    );
}

#[test]
#[ignore = "requires NGSPICE_BIN; moving vector references"]
fn vector_equality_crossings_match_c() {
    compare(
        "vector-equality",
        AnalysisKind::DcSweep,
        "V1 in 0 0\nR1 in 0 1k\nE1 ref 0 in 0 .5\nV2 target ref .375\n.dc v1 0 2 .2\n.meas dc rising WHEN v(in)=v(target) RISE=1\n.meas dc falling WHEN v(target)=v(in) FALL=1\n.meas dc found FIND v(ref) WHEN v(in)=v(target)\n.meas dc slope DERIV v(ref) WHEN v(in)=v(target)",
    );
}

#[test]
fn vector_equality_uses_both_sampled_slopes() {
    let deck = "V1 in 0 0\nR1 in 0 1k\nE1 ref 0 in 0 .5\nV2 target ref .375\n.dc v1 0 2 .2\n.meas dc rising WHEN v(in)=v(target) RISE=1\n.meas dc falling WHEN v(target)=v(in) FALL=1\n.meas dc found FIND v(ref) WHEN v(in)=v(target)\n.meas dc slope DERIV v(ref) WHEN v(in)=v(target)";
    let results = port_results(deck);
    for ((name, value), expected) in results.iter().zip([0.75, 0.75, 0.375, 0.5]) {
        assert!((value - expected).abs() < 1e-12, "{name}: {value}");
    }
    assert_eq!(results.len(), 4);
}
