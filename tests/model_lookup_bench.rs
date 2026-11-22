//! Opt-in declaration-index setup and lookup benchmark. Run the ignored
//! tests in separate processes (also suitable for DHAT allocation profiling).
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use ngspice_rs::devices::ModelResolver;
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::{DeviceInstance, ModelCard};
use ngspice_rs::netlist::source::parse_deck_text;

const GROUPS: usize = 512;
const BINS: usize = 8;
const QUERIES: usize = 8192;

#[derive(Clone, Copy)]
enum Workload {
    Index,
    Exact,
    Binned,
}

fn fixture(workload: Workload) -> (Vec<ModelCard>, Vec<DeviceInstance>) {
    let deck = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("model-lookup-bench.cir"),
            "model lookup\nm1 d g 0 0 family0 l=1u w=1u\n.model family0.1 nmos level=8 lmin=0.5u lmax=2u wmin=0.5u wmax=2u vth0=0.4\n.end\n",
        ))
        .unwrap();
    let template = &deck.models[0];
    let mut cards = Vec::with_capacity(GROUPS * BINS);
    let mut devices = Vec::with_capacity(GROUPS);
    for group in 0..GROUPS {
        let base = format!("family{group}");
        for bin in 1..=BINS {
            let mut card = template.clone();
            card.name = format!("{base}.{bin}");
            if bin != 1 {
                // Force a full reverse walk of the relevant bin group.
                card.parameters
                    .iter_mut()
                    .find(|p| p.name == "lmin")
                    .unwrap()
                    .value = "5u".into();
                card.parameters
                    .iter_mut()
                    .find(|p| p.name == "lmax")
                    .unwrap()
                    .value = "8u".into();
            }
            cards.push(card);
        }
        let mut device = deck.devices[0].clone();
        device.model = Some(if matches!(workload, Workload::Exact) {
            format!("{base}.1")
        } else {
            base
        });
        devices.push(device);
    }
    (cards, devices)
}

fn benchmark(workload: Workload) {
    let (cards, devices) = fixture(workload);
    let start = Instant::now();
    let resolver = ModelResolver::new(black_box(&cards)).unwrap();
    let setup = start.elapsed();
    println!(
        "{} declarations, index setup {:.6} s",
        cards.len(),
        setup.as_secs_f64()
    );
    if matches!(workload, Workload::Index) {
        black_box(resolver);
        return;
    }
    let start = Instant::now();
    for query in 0..QUERIES {
        let group = query % GROUPS;
        let selected = resolver
            .resolve(black_box(&devices[group]))
            .unwrap()
            .unwrap();
        assert!(std::ptr::eq(selected.card(), &cards[group * BINS]));
        black_box(selected);
    }
    println!("{QUERIES} lookups, {:.6} s", start.elapsed().as_secs_f64());
}

#[test]
#[ignore = "opt-in model declaration index setup benchmark"]
fn index_only() {
    benchmark(Workload::Index);
}

#[test]
#[ignore = "opt-in exact declaration lookup benchmark"]
fn exact_lookup() {
    benchmark(Workload::Exact);
}

#[test]
#[ignore = "opt-in binned declaration lookup benchmark"]
fn binned_lookup() {
    benchmark(Workload::Binned);
}
