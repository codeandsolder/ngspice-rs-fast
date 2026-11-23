//! Repeated prepared Newton and AC evaluations of a nonlinear B-source
//! network. Run under Valgrind DHAT for allocation counts, or natively for
//! representative CPU times. Deliberately separates device stamps from solve.

use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use ngspice_rs::devices::{AnalysisMode, Circuit, IterationPhase, LoadRequest, ModelContext};
use ngspice_rs::maths::Vector;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::Real;

const NETWORK: &str = "vin in 0 dc 0.7
vb b 0 dc -0.4
vx x 0 dc 0.25
rin in 0 1k
rb b 0 2k
rx x 0 3k
b1 out 0 v=v(in)>0 ? (log(1+v(in))+sin(v(b))+2*i(vx)) : sqrt(-v(in))
r1 out 0 1k
b2 0 out2 i=1m*pwl(v(in), -1,-2, 0,0, 1,3, 2,3.5) + 2m*v(out)
r2 out2 0 1k
b3 out3 0 v=exp(v(in)/2)+sqrt(1+v(out2)*v(out2)) + i(vb)*100
r3 out3 0 1k
b4 0 out4 i=1m*tanh(v(out3)) + 1m*cos(v(in)) + v(out2)*1m
r4 out4 0 1k
b5 out5 0 v=v(in)>0.3 ? v(in)^2 + abs(v(b)) : v(out3)
r5 out5 0 1k
";

fn build_circuit() -> Circuit {
    let deck = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("behavioural_workspace_bench.cir"),
            &format!("behavioural runtime benchmark\n{NETWORK}.end\n"),
        ))
        .unwrap();
    Circuit::from_netlist(&deck).unwrap()
}

fn point(circuit: &Circuit) -> Vector {
    let mut x = Vector::zeros(circuit.unknown_count());
    for (n, value) in x.as_mut_slice().iter_mut().enumerate() {
        *value = 0.2 + (n % 5) as Real * 0.075;
    }
    x
}

fn stamp_bits(circuit: &Circuit, x: &Vector, repeats: usize) -> (Vec<u64>, f64) {
    let history = circuit.state_history().unwrap();
    let mut trial = history.trial_in(IterationPhase::Float, None).unwrap();
    let mut matrix = circuit.new_matrix().unwrap();
    let mut rhs = Vector::zeros(circuit.unknown_count());
    let context = ModelContext::default();
    let request = LoadRequest {
        mode: AnalysisMode::OperatingPoint,
        solution: x,
        model_context: &context,
        integration: None,
        history: &history,
        forcing: None,
    };
    let prepared = circuit.prepare_run();
    let mut run = || {
        matrix.clear();
        rhs.clear();
        prepared
            .load(&request, &mut matrix, &mut rhs, &mut trial)
            .unwrap();
        let mut checksum = 0_u64;
        for (i, value) in rhs.as_slice().iter().enumerate() {
            checksum ^= value.to_bits().rotate_left(i as u32 % 64);
        }
        black_box(checksum)
    };
    for _ in 0..8 {
        run();
    }
    let now = Instant::now();
    let checksum = (0..repeats).fold(0u64, |sum, _| sum ^ run());
    let seconds = now.elapsed().as_secs_f64();
    let mut bits = Vec::new();
    for row in 0..circuit.unknown_count() {
        for col in 0..circuit.unknown_count() {
            bits.push(matrix.get(row, col).to_bits());
        }
    }
    bits.extend(rhs.as_slice().iter().map(|v| v.to_bits()));
    bits.push(checksum);
    (bits, seconds)
}

#[test]
fn repeated_prepared_b_sources_preserve_exact_stamps_across_independent_circuits() {
    let circuit = build_circuit();
    let x = point(&circuit);
    let (original, _) = stamp_bits(&circuit, &x, 8);
    let (again, _) = stamp_bits(&circuit, &x, 8);
    assert_eq!(again, original);
    let copy = build_circuit();
    let (cloned, _) = stamp_bits(&copy, &x, 8);
    assert_eq!(cloned, original);

    // The same device must recover when the selected ternary branches change,
    // then return to its old bias, without borrowing stale values or gradients.
    let in_row = circuit
        .unknowns()
        .node_row(circuit.nodes().get("in").unwrap())
        .unwrap();
    for bias in [-0.7, 0.8, -0.2, 0.4, 1.3] {
        let mut perturbed = x.clone();
        perturbed.as_mut_slice()[in_row] = bias;
        let current = stamp_bits(&circuit, &perturbed, 4).0;
        let independent = stamp_bits(&copy, &perturbed, 4).0;
        assert_eq!(current, independent, "independent candidate at {bias}");
    }
    assert_eq!(stamp_bits(&circuit, &x, 8).0, original);

    // An independent AC linearization exercises the same compiled gradient
    // after all repeated DC loads, without retaining another plot.
    let a = circuit
        .small_signal_system(&ModelContext::default(), &x)
        .unwrap();
    let b = copy
        .small_signal_system(&ModelContext::default(), &x)
        .unwrap();
    for row in 0..circuit.unknown_count() {
        for col in 0..circuit.unknown_count() {
            assert_eq!(a.a.get(row, col).to_bits(), b.a.get(row, col).to_bits());
        }
    }
}

/// Run this in its own process under Valgrind DHAT to attribute per-call
/// allocations to expression evaluation instead of just total elapsed time.
#[test]
#[ignore = "opt-in prepared Newton and AC device workload; use Valgrind DHAT for allocation attribution"]
fn measure_prepared_behavioural_stamp_workload() {
    let circuit = build_circuit();
    let x = point(&circuit);
    for repeats in [1000, 1000, 1000] {
        let (bits, seconds) = stamp_bits(&circuit, &x, repeats);
        println!(
            "5 nonlinear B sources; {repeats} prepared Newton loads in {:.3}ms, checksum={:x}",
            seconds * 1e3,
            bits.last().unwrap()
        );
    }
    let context = ModelContext::default();
    let start = Instant::now();
    for _ in 0..1000 {
        black_box(circuit.small_signal_system(&context, &x).unwrap());
    }
    println!(
        "1000 whole-circuit AC Jacobians in {:.3}ms",
        start.elapsed().as_secs_f64() * 1e3
    );
}
