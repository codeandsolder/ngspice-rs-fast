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
