//! Opt-in semantic comparison benchmark. Each test is a separate workload;
//! run it directly under Valgrind DHAT to measure process allocations. The
//! fixture-only case isolates parsing costs from comparison allocations.
use std::fmt::Write;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::netlist::{Parser, semantic_diff, semantic_eq};

const INSTANCES: usize = 2048;
const REPEATS: usize = 16;

fn fixture() -> (Netlist, Netlist) {
    let mut input = String::from("semantic comparison\n.subckt s a b\n.param base=1k\n");
    for i in 0..INSTANCES {
        writeln!(
            input,
            "r{i} a n{i} {{base*(1+2*3)}}\nb{i} n{i} b v=v(a)*sin(v(b))+{i}"
        )
        .unwrap();
    }
    input.push_str(".ends\nx1 a b s\n.end\n");
    let parser = |path: &str| {
        Parser::new()
            .parse_deck(&parse_deck_text(Path::new(path), &input))
            .unwrap()
    };
    (parser("left.cir"), parser("right.cir"))
}

#[test]
#[ignore = "opt-in fixture allocation baseline"]
fn fixture_only() {
    black_box(fixture());
}

#[test]
#[ignore = "opt-in equal deck semantic comparison"]
fn equal_decks() {
    let (left, right) = fixture();
    let start = Instant::now();
    for _ in 0..REPEATS {
        assert!(semantic_eq(black_box(&left), black_box(&right)));
        assert!(semantic_diff(black_box(&left), black_box(&right)).is_none());
    }
    println!(
        "{} equal comparisons, {:.6} s",
        REPEATS * 2,
        start.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "opt-in first-field mismatch comparison"]
fn early_mismatch() {
    let (left, mut right) = fixture();
    right.title.push('!');
    let start = Instant::now();
    for _ in 0..REPEATS {
        assert!(!semantic_eq(black_box(&left), black_box(&right)));
        let diff = semantic_diff(black_box(&left), black_box(&right)).unwrap();
        assert!(diff.starts_with("title"));
        black_box(diff);
    }
    println!(
        "{} early comparisons, {:.6} s",
        REPEATS * 2,
        start.elapsed().as_secs_f64()
    );
}
