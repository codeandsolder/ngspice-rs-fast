//! Bounded deterministic performance workloads. See docs/port/PERFORMANCE.md.
use std::time::Duration;

#[path = "performance_workloads/decks.rs"]
mod decks;
#[path = "performance_workloads/matrix.rs"]
mod matrix;

fn report(workload: &str, phase: &str, operations: u32, elapsed: Duration) {
    println!(
        "workload={workload},phase={phase},operations={operations},elapsed_ns={}",
        elapsed.as_nanos()
    );
}

#[test]
fn small_workloads_are_valid() {
    matrix::real("smoke_ladder", 8, 1, 2);
    matrix::real("smoke_mesh", 4, 4, 2);
    matrix::frequency_ensemble(4, 4, 3);
    decks::smoke();
}

#[test]
#[ignore = "opt-in sparse ladder performance workload"]
fn sparse_ladder() {
    matrix::real("sparse_ladder", 1024, 1, 32);
}

#[test]
#[ignore = "opt-in sparse mesh performance workload"]
fn sparse_mesh() {
    matrix::real("sparse_mesh", 32, 32, 24);
}

#[test]
#[ignore = "opt-in independent complex frequency ensemble"]
fn complex_frequency_ensemble() {
    matrix::frequency_ensemble(24, 24, 64);
}

#[test]
#[ignore = "opt-in RC transient workload"]
fn rc_transient() {
    decks::rc();
}

#[test]
#[ignore = "opt-in nonlinear mixed-device workload"]
fn nonlinear_mixed() {
    decks::nonlinear();
}

#[test]
#[ignore = "opt-in compact model array workload"]
fn compact_models() {
    decks::compact();
}

#[test]
#[ignore = "opt-in RF multiport workload"]
fn rf_multiport() {
    decks::rf();
}

#[test]
#[ignore = "opt-in long LTRA convolution workload"]
fn ltra_history() {
    decks::ltra();
}

#[test]
#[ignore = "opt-in independent parameter ensemble"]
fn parameter_ensemble() {
    decks::parameters();
}

#[test]
#[ignore = "opt-in multi-frequency AC factorization and ordered observation workload"]
fn independent_ac_frequency_tiles() {
    use std::fmt::Write as _;
    use std::hint::black_box;
    use std::path::Path;
    use std::time::Instant;

    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};

    for nodes in [8, 32, 128, 512] {
        let mut deck = String::from("frequency tile ladder\nv0 n0 0 dc 0 ac 1\n");
        for index in 0..nodes {
            writeln!(deck, "r{index} n{index} n{} 1k", index + 1).unwrap();
            writeln!(deck, "c{index} n{} 0 1n", index + 1).unwrap();
        }
        writeln!(deck, "rload n{nodes} 0 1k").unwrap();
        deck.push_str(".ac lin 65 1k 1meg\n.end\n");
        let netlist = Parser::new()
            .parse_deck(&parse_deck_text(Path::new("ac-frequency-bench.cir"), &deck))
            .unwrap();
        let config = RunConfig::from_netlist(&netlist).unwrap();
        let plan = config.plan_for(&netlist.analyses[0]).unwrap();
        let mut samples = Vec::new();
        let mut expected = None;
        for _ in 0..5 {
            let mut circuit = config.circuit(&netlist).unwrap();
            let start = Instant::now();
            let plot = plan.run(&mut circuit, &config.context()).unwrap();
            samples.push(start.elapsed());
            assert_eq!(plot.point_count(), 65);
            assert!(plot.is_finite());
            let observed = plot.value(&format!("v(n{nodes})"), 64).unwrap();
            let bits = [observed.re.to_bits(), observed.im.to_bits()];
            if let Some(expected) = expected {
                assert_eq!(expected, bits, "AC output changed between independent runs");
            }
            expected = Some(bits);
            black_box(&plot);
        }
        samples.sort_unstable();
        println!(
            "workload=independent_ac_frequency_tiles,nodes={nodes},frequencies=65,median_ns={}",
            samples[2].as_nanos(),
        );
    }
}
