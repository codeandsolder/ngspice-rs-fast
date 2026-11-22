//! Compare the former scan/allocate pathways with the indexed waveform and
//! heap scheduler on actual generated parsed decks. Timings exclude parsing.
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use ngspice_rs::devices::{Circuit, Limit, LinearSystem, PwlSource, Waveform};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};

fn system(deck: &str) -> LinearSystem {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("schedule-bench.cir"), deck))
        .unwrap();
    Circuit::from_netlist(&netlist)
        .unwrap()
        .linear_system()
        .unwrap()
}

#[test]
#[ignore = "opt-in waveform scheduling benchmark"]
fn sparse_sources_and_large_pwl_deck() {
    const SOURCES: u32 = 4096;
    // Match the parser's existing maximum of 4096 numeric fields.
    const KNOTS: u32 = 2048;
    let mut deck = String::from("sparse schedules\nr1 out 0 1k\n");
    for source in 0..SOURCES {
        write!(deck, "i{source} out 0 pwl(").unwrap();
        for index in 0..8 {
            write!(deck, " {} {}", index * SOURCES + source, index % 2).unwrap();
        }
        deck.push_str(")\n");
    }
    deck.push_str(".end\n");
    let sparse = system(&deck);
    let end = f64::from(8 * SOURCES);
    let start = Instant::now();
    let heap = sparse.breakpoints_in(0., end).unwrap().collect::<Vec<_>>();
    let heap_elapsed = start.elapsed();
    assert_eq!(heap.len(), (8 * SOURCES) as usize);
    let start = Instant::now();
    let mut scan = sparse
        .sources
        .iter()
        .map(|source| source.waveform.breakpoints_in(0., end).unwrap().peekable())
        .collect::<Vec<_>>();
    let mut original = Vec::with_capacity(heap.len());
    while let Some(time) = scan
        .iter_mut()
        .filter_map(|source| source.peek().copied())
        .min_by(f64::total_cmp)
    {
        original.push(time);
        for source in &mut scan {
            while source
                .next_if(|t| t.partial_cmp(&time) == Some(std::cmp::Ordering::Equal))
                .is_some()
            {}
        }
    }
    println!(
        "{SOURCES} sources/{} events: scan {:?}, heap {heap_elapsed:?}",
        heap.len(),
        start.elapsed()
    );
    assert_eq!(
        original.iter().map(|t| t.to_bits()).collect::<Vec<_>>(),
        heap.iter().map(|t| t.to_bits()).collect::<Vec<_>>()
    );
    let mut deck = String::from("large PWL\nv1 out 0 pwl(");
    for index in 0..KNOTS {
        if index % 32 == 0 && index != 0 {
            deck.push_str("\n+");
        }
        write!(deck, " {index} {}", index % 2).unwrap();
    }
    deck.push_str(")\nr1 out 0 1k\n.end\n");
    let large = system(&deck);
    benchmark_lookup(&large.sources[0].waveform, KNOTS);
    // The programmatic waveform API has no parser field limit.
    const LARGE_KNOTS: u32 = 65_536;
    let large_waveform = Waveform::Pwl(
        PwlSource::plain(
            (0..LARGE_KNOTS)
                .map(|i| (f64::from(i), f64::from(i % 2)))
                .collect(),
        )
        .unwrap(),
    );
    benchmark_lookup(&large_waveform, LARGE_KNOTS);
}

fn benchmark_lookup(waveform: &Waveform, knots: u32) {
    const REQUESTS: u32 = 8192;
    let Waveform::Pwl(pwl) = waveform else {
        panic!("not PWL");
    };
    assert_eq!(pwl.knots().len(), knots as usize);
    let times = (0..REQUESTS)
        .map(|index| f64::from(index * 7919 % knots) + 0.125)
        .collect::<Vec<_>>();
    let start = Instant::now();
    let indexed = times
        .iter()
        .map(|&time| black_box(waveform.value_at(black_box(time), Limit::Right).unwrap()))
        .collect::<Vec<_>>();
    let indexed_elapsed = start.elapsed();
    let start = Instant::now();
    let original = times
        .iter()
        .map(|&time| {
            let knots = black_box(pwl.knots());
            knots
                .windows(2)
                .find_map(|w| {
                    let ((t0, v0), (t1, v1)) = (w[0], w[1]);
                    (time <= t1).then(|| {
                        let fraction = (time - t0) / (t1 - t0);
                        v0 + fraction * (v1 - v0)
                    })
                })
                .unwrap_or(knots[knots.len() - 1].1)
        })
        .collect::<Vec<_>>();
    println!(
        "{knots} knots/{REQUESTS} evaluations: scan {:?}, indexed {indexed_elapsed:?}",
        start.elapsed()
    );
    assert_eq!(
        original.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        indexed.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
