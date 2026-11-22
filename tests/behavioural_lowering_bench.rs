//! Opt-in frontend benchmark. Run with
//! `cargo test --test behavioural_lowering_bench -- --ignored --nocapture`.
//! Parse outside timing and exercise the full clone/edit/lower operation.
//! The duplicate-name stress case intentionally measures lowering before
//! circuit construction rejects duplicate devices.

use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use ngspice_rs::netlist::{Parser, behavioural::lower_nonlinear_sources, source::parse_deck_text};

fn scoped_deck(duplicates: bool) -> String {
    let mut deck = String::from("lowering benchmark\n");
    for parent in 0..16 {
        writeln!(deck, ".subckt parent{parent} a").unwrap();
        for child in 0..4 {
            writeln!(deck, ".subckt child{child} a").unwrap();
            for index in 0..128 {
                let target = if duplicates { index / 16 } else { index };
                writeln!(deck, "r{target} n{index} 0 1k").unwrap();
                writeln!(deck, "b{index} out{index} 0 v=i(r{target})+i(r{target})").unwrap();
                writeln!(deck, "e{index} e{index} 0 value={{v(out{index})}}").unwrap();
                writeln!(
                    deck,
                    "g{index} g{index} 0 table {{v(n{index})}} = (0,0) (1,2)"
                )
                .unwrap();
            }
            writeln!(deck, ".ends child{child}").unwrap();
        }
        writeln!(deck, ".ends parent{parent}").unwrap();
    }
    deck.push_str(".end\n");
    deck
}

#[test]
#[ignore = "opt-in frontend lowering benchmark"]
fn large_scoped_decks() {
    for duplicates in [false, true] {
        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("benchmark.cir"),
                &scoped_deck(duplicates),
            ))
            .unwrap();
        let expected = lower_nonlinear_sources(&deck).unwrap();
        let mut samples = Vec::new();
        for _ in 0..7 {
            let start = Instant::now();
            let lowered = black_box(lower_nonlinear_sources(black_box(&deck)).unwrap());
            samples.push(start.elapsed());
            assert_eq!(lowered, expected);
        }
        samples.sort_unstable();
        eprintln!(
            "duplicates={duplicates} median={:?} all={samples:?}",
            samples[3]
        );
    }
}
