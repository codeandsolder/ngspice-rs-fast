//! End-to-end, source-backed Newton allocation workloads (#135).
//!
//! Run each ignored benchmark separately under Valgrind DHAT to attribute
//! per-Newton-load heap allocation counts/bytes, and natively for throughput.
//! Reference cases exercise multiple accepted/rejected Newton loads and deep
//! BSIM state vectors, not synthetic hand-stamped matrices.
use std::{hint::black_box, path::Path, time::Instant};

use ngspice_rs::analysis::RunConfig;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};

fn fixture(name: &str, deck: &str, repeats: usize) {
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(Path::new(name), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&parsed).unwrap();
    let plan = config.plan_for(&parsed.analyses[0]).unwrap();
    let mut signature = None;
    let mut elapsed_ns = 0_u128;
    for _ in 0..repeats {
        let mut circuit = config.circuit(&parsed).unwrap();
        let states = circuit.state_history().unwrap().len();
        let start = Instant::now();
        let plot = black_box(plan.run(&mut circuit, &config.context()).unwrap());
        elapsed_ns += start.elapsed().as_nanos();
        let checksum = plot.rows().flatten().fold(0_u64, |sum, value| {
            sum.rotate_left(7) ^ value.re.to_bits() ^ value.im.to_bits()
        });
        let actual = (states, plot.point_count(), checksum);
        if let Some(previous) = signature {
            assert_eq!(
                previous, actual,
                "the same physical deck changed between independent runs"
            );
        }
        signature = Some(actual);
    }
    println!(
        "newton_workspace_{name}: runs={repeats} states={} points={} checksum={:#x} total_ns={elapsed_ns} mean_ns={}",
        signature.unwrap().0,
        signature.unwrap().1,
        signature.unwrap().2,
        elapsed_ns / u128::try_from(repeats).unwrap(),
    );
}

#[test]
#[ignore = "opt-in physical BSIM3 nested DC continuation and DHAT allocation workload"]
fn bsim3_nested_dc_workload() {
    fixture(
        "bsim3_dc",
        include_str!("../conformance/netlists/m11_bsim3_dc.cir"),
        3,
    );
}

#[test]
#[ignore = "opt-in BSIM4 NQS/body-state transient Newton DHAT workload"]
fn bsim4_transient_workload() {
    fixture(
        "bsim4_transient",
        include_str!("../conformance/netlists/m11_bsim4_tran.cir"),
        3,
    );
}

#[test]
#[ignore = "opt-in BSIM3 capacitor/charge transient Newton DHAT workload"]
fn bsim3_transient_workload() {
    fixture(
        "bsim3_transient",
        include_str!("../conformance/netlists/m11_bsim3_tran.cir"),
        3,
    );
}
