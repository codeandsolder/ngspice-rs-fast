//! Opt-in matched layout/refresh workload. Invoke each test independently
//! under time and Valgrind DHAT; the device laws are identical in both modes.
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use ngspice_rs::devices::{Circuit, ModelContext, noise::NoisePlan};
use ngspice_rs::maths::Vector;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};

const DEVICES: u32 = 512;
const FREQUENCIES: u32 = 256;

fn benchmark(rebind: bool) {
    let mut text = String::from("noise layout benchmark\n");
    for model in 0..32 {
        writeln!(text, ".model r{model} r(r=1k kf=1e-12)").unwrap();
    }
    for index in 0..DEVICES {
        writeln!(text, "r{index} a b r{}", index % 32).unwrap();
    }
    text.push_str(".end\n");
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("noise-workspace-bench.cir"),
            &text,
        ))
        .unwrap();
    let circuit = Circuit::from_netlist(&netlist).unwrap();
    let run = circuit.prepare_run();
    let bias = Vector::from_slice(&[1., 0.]);
    let model = |step| ModelContext::default().with_frequency(f64::from(step) + 1.);
    let mut plan = NoisePlan::bind(&run, &model(0), &bias, None).unwrap();
    let start = Instant::now();
    for step in 0..FREQUENCIES {
        if rebind {
            plan = NoisePlan::bind(&run, &model(step), &bias, None).unwrap();
        } else {
            plan.refresh(&run, &model(step), &bias, None).unwrap();
        }
        assert_eq!(plan.source_count(), 2 * DEVICES as usize);
        black_box(&plan);
    }
    println!(
        "{DEVICES} devices, {FREQUENCIES} frequencies, rebind={rebind}: {:?}",
        start.elapsed()
    );
}

#[test]
#[ignore = "opt-in noise layout comparison"]
fn rebuild_every_frequency() {
    benchmark(true);
}

#[test]
#[ignore = "opt-in noise layout comparison"]
fn stable_layout_refresh() {
    benchmark(false);
}
