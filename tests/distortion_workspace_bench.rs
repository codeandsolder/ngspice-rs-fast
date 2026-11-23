//! Reproducible distortion sweep benchmarks, using the complete analysis API.
use ngspice_rs::analysis::RunConfig;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use std::{fmt::Write as _, hint::black_box, path::Path, time::Instant};

fn sweep(branches: u32, steps: u32, intermodulation: bool) {
    let mut deck = String::from(
        "Volterra workspace benchmark\n.model dm d(is=1e-14 n=1.5)\n.option reltol=1e-7\n",
    );
    for i in 0..branches {
        writeln!(deck, "i{i} 0 n{i} dc 1m distof1 2u 30 distof2 1u 15\nr{i} n{i} 0 10k\nd{i} n{i} 0 dm\nc{i} n{i} 0 100n").unwrap();
    }
    let ratio = if intermodulation { " 1.7" } else { "" };
    writeln!(deck, ".disto lin {steps} 1k 4k{ratio}\n.end").unwrap();
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("disto-bench.cir"), &deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let start = Instant::now();
    let plots = black_box(request.run_plots(&mut circuit, &config.context()).unwrap());
    let elapsed = start.elapsed().as_nanos();
    assert_eq!(plots.len(), if intermodulation { 3 } else { 2 });
    let mut digest = 0_u64;
    let mut points = 0;
    for plot in &plots {
        points = plot.point_count();
        assert_eq!(
            plot.variable_count(),
            usize::try_from(branches).unwrap() + 1
        );
        assert!(points >= 2);
        for row in plot.rows() {
            for value in row {
                assert!(value.is_finite());
                digest = (digest ^ value.re.to_bits()).wrapping_mul(0x100_0000_01b3)
                    ^ value.im.to_bits();
            }
        }
    }
    println!("volterra,{branches},{steps},{intermodulation},{points},{digest:016x},{elapsed}");
}

#[test]
#[ignore = "release performance benchmark"]
fn harmonic_small() {
    sweep(4, 2000, false);
}
#[test]
#[ignore = "release performance benchmark"]
fn harmonic_medium() {
    sweep(32, 1000, false);
}
#[test]
#[ignore = "release performance benchmark"]
fn harmonic_large() {
    sweep(128, 400, false);
}
#[test]
#[ignore = "release performance benchmark"]
fn intermod_small() {
    sweep(4, 2000, true);
}
#[test]
#[ignore = "release performance benchmark"]
fn intermod_medium() {
    sweep(32, 1000, true);
}
#[test]
#[ignore = "release performance benchmark"]
fn intermod_large() {
    sweep(128, 400, true);
}
