
//! Reproducible allocation and CPU workloads for many saved device quantities.
//!
//! Run the two ignored workloads independently (also suitable for DHAT).
use ngspice_rs::{
    analysis::RunConfig,
    devices::{AnalysisMode, Circuit, LoadRequest, ModelContext},
    maths::Vector,
    netlist::{Parser, ast::Netlist, source::parse_deck_text},
    primitives::Complex,
};
use std::{fmt::Write as _, hint::black_box, path::Path, time::Instant};

const OBSERVED: usize = 128;

fn fixture(branches: usize) -> (Netlist, Vec<String>) {
    let mut deck = String::from("Observation benchmark\nv1 in 0 sin(0 1 1meg)\n");
    for i in 0..branches {
        writeln!(deck, "r{i} in n{i} 1k\nc{i} n{i} 0 1n").unwrap();
    }
    deck.push_str(".tran 10n 5u 0 10n\n.end\n");
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("observation-bench.cir"), &deck))
        .unwrap();
    let names = (0..OBSERVED)
        .flat_map(|i| {
            [
                format!("i(r{i})"),
                format!("@r{i}[p]"),
                format!("@c{i}[i]"),
                format!("@c{i}[capacitance]"),
            ]
        })
        .collect();
    (netlist, names)
}

fn digest(hash: u64, value: Complex) -> u64 {
    assert!(value.is_finite());
    (hash ^ value.re.to_bits()).wrapping_mul(0x100_0000_01b3) ^ value.im.to_bits()
}

fn direct(points: u32) {
    let (netlist, names) = fixture(1024);
    let mut circuit = Circuit::from_netlist(&netlist).unwrap();
    circuit.set_observations(&names).unwrap();
    let history = circuit.state_history().unwrap();
    let model = ModelContext::default();
    let run = circuit.prepare_run();
    let mut solution = Vector::zeros(circuit.unknown_count());
    assert_eq!(circuit.observations().len(), names.len());
    let start = Instant::now();
    let mut hash = 0_u64;
    for point in 0..points {
        let voltage = (f64::from(point) * 0.01).sin();
        for (i, value) in solution.as_mut_slice().iter_mut().enumerate() {
            *value = voltage * f64::from(u32::try_from(i % 17).unwrap()) / 17.;
        }
        let values = run
            .observe_real(
                &LoadRequest {
                    mode: AnalysisMode::OperatingPoint,
                    solution: black_box(&solution),
                    model_context: &model,
                    integration: None,
                    history: &history,
                    forcing: None,
                },
                &[],
                None,
                None,
            )
            .unwrap();
        assert_eq!(values.len(), names.len());
        hash = values.into_iter().fold(hash, digest);
    }
    println!(
        "direct,{points},{},{hash:016x},{}",
        names.len(),
        start.elapsed().as_nanos()
    );
}

#[test]
#[ignore = "opt-in isolated observation allocation baseline"]
fn fixture_only() {
    direct(0);
}

#[test]
#[ignore = "opt-in isolated observation workload"]
fn many_saved_devices() {
    direct(512);
}

#[test]
#[ignore = "opt-in complete transient workload"]
fn many_saved_transient() {
    let (netlist, names) = fixture(256);
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    circuit.set_observations(&names).unwrap();
    let start = Instant::now();
    let plot = request.run(&mut circuit, &config.context()).unwrap();
    let elapsed = start.elapsed();
    assert!(plot.point_count() >= 500);
    let mut hash = 0_u64;
    for row in plot.rows() {
        hash = row.iter().copied().fold(hash, digest);
    }
    for name in &names {
        assert!(plot.variable_index(name).is_some());
    }
    println!(
        "transient,{},{},{hash:016x},{}",
        plot.point_count(),
        plot.variables().len(),
        elapsed.as_nanos()
    );
}
