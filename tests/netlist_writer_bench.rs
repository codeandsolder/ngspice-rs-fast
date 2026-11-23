//! Opt-in normalized writer measurement, shared with the pre-change baseline.
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::netlist::{Parser, semantic_eq, write_netlist};

const REPEATS: usize = 24;

fn parse(text: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(Path::new("writer-bench.cir"), text))
        .unwrap()
}

fn models() -> Netlist {
    let mut text = String::from("model lists\n");
    for i in 0..128 {
        write!(text, ".model d{i} d(").unwrap();
        for j in 0..128 {
            write!(text, "is={j}e-14 rs={j} ").unwrap();
        }
        text.push_str(")\n");
    }
    text.push_str(".end\n");
    parse(&text)
}

fn waveforms() -> Netlist {
    let mut text = String::from("repeated pwl\n");
    for i in 0..128 {
        write!(text, "v{i} n{i} 0 PWL(").unwrap();
        for j in 0..512 {
            write!(text, " {j}u {j}m").unwrap();
        }
        text.push_str(")\n");
    }
    text.push_str(".end\n");
    parse(&text)
}

fn nested() -> Netlist {
    let mut text = String::from("nested behavioural decks\n");
    for i in 0..32 {
        writeln!(
            text,
            ".subckt s{i} a b\n.param base=1k\n.func scale(x) {{x*base}}"
        )
        .unwrap();
        for j in 0..32 {
            writeln!(
                text,
                "r{j} a n{j} {{scale(1+{j})}}\nb{j} n{j} b v=v(a)*sin(v(b))+{j}"
            )
            .unwrap();
        }
    }
    for i in (0..32).rev() {
        writeln!(text, ".ends s{i}").unwrap();
    }
    text.push_str(".end\n");
    parse(&text)
}

fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

fn run(netlist: &Netlist) {
    let expected = write_netlist(netlist).unwrap();
    let reparsed = parse(&expected);
    assert!(semantic_eq(netlist, &reparsed));
    assert_eq!(write_netlist(&reparsed).unwrap(), expected);
    drop(reparsed);
    let checksum = digest(expected.as_bytes());
    let start = Instant::now();
    for _ in 0..REPEATS {
        assert_eq!(
            black_box(write_netlist(black_box(netlist)).unwrap()),
            expected
        );
    }
    println!(
        "writer iterations={REPEATS} bytes={} digest={checksum:016x} seconds={:.6}",
        expected.len(),
        start.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "opt-in paired writer benchmark"]
fn model_lists() {
    run(&models());
}

#[test]
#[ignore = "opt-in paired writer benchmark"]
fn pwl_lists() {
    run(&waveforms());
}

#[test]
#[ignore = "opt-in paired writer benchmark"]
fn nested_expressions() {
    run(&nested());
}
