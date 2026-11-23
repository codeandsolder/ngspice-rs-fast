//! The same finalized diode circuits loaded with and without the homogeneous
//! planning optimization. Inspect raw MNA/state bits, not merely solved plots.
use super::Circuit;

fn diode_circuits(count: usize, model_groups: bool, rich: bool) -> (Circuit, Circuit) {
    use crate::netlist::{Parser, source::parse_deck_text};
    use std::{fmt::Write, path::Path};

    let mut deck = String::from("batched device load\n");
    deck.push_str("v1 n0 0 dc 0\nr1 n0 n1 100\n");
    // ngspice rejects unused model declarations: each test size has
    // precisely the models its source cards actually reference.
    const FULL_PHYSICS: &str = "is=1e-14 n=1.1 rs=2 cjo=30p tt=5n vj=0.8 m=0.4 fc=0.5          bv=5.1 ibv=5m nbv=1.4 jsw=2e-15 ns=1.6 cjsw=4p vjsw=0.8 mjsw=0.3          fcs=0.4 isr=1e-11 nr=2.2 ikf=1m ikr=1u ikp=1u          jtun=1e-13 jtunsw=1e-14 ntun=20";
    if count >= 2 || (count > 0 && !model_groups) {
        if rich {
            writeln!(&mut deck, ".model dm d({FULL_PHYSICS})").unwrap();
        } else {
            deck.push_str(".model dm d(is=1e-14 rs=2 cjo=2p tt=3n)\n");
        }
    }
    if count > 0 && model_groups {
        deck.push_str(".model dn d(is=2e-14 rs=0 cjo=4p tt=1n)\n");
    }
    for index in 0..count {
        let model = if model_groups && index % 5 == 0 {
            "dn"
        } else {
            "dm"
        };
        let node = index % 8;
        if rich {
            let area = 1 + index % 4;
            let perimeter = 1 + index % 7;
            let temperature = 5 + (index % 5) * 20;
            writeln!(
                &mut deck,
                "d{index} n{node} 0 {model} area={area} pj={perimeter} temp={temperature}"
            )
            .unwrap();
        } else {
            writeln!(&mut deck, "d{index} n{node} 0 {model}").unwrap();
        }
        if index % 23 == 0 {
            writeln!(&mut deck, "r{index} n{node} 0 2k").unwrap();
        }
    }
    deck.push_str(".op\n.end\n");
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("diode-batches.cir"), &deck))
        .unwrap();
    let batched = Circuit::from_netlist(&parsed).unwrap();
    let mut scalar = Circuit::from_netlist(&parsed).unwrap();
    let active: usize = batched
        .bindings
        .diode_batches
        .iter()
        .map(|batch| batch.len)
        .sum();
    if count >= 2 {
        assert!(
            active >= count.saturating_sub(2),
            "count={count}, active={active}"
        );
    } else {
        assert_eq!(active, 0);
    }
    assert!(
        batched
            .bindings
            .diode_batches
            .iter()
            .all(|batch| batch.len >= 2 && batch.len <= super::DIODE_TILE)
    );
    scalar.bindings.diode_batches.clear();
    assert_eq!(batched.unknown_count(), scalar.unknown_count());
    (batched, scalar)
}

fn assert_same_load(
    label: &str,
    first: (
        &crate::maths::SparseMatrix,
        &crate::maths::Vector,
        &crate::devices::TrialState,
    ),
    second: (
        &crate::maths::SparseMatrix,
        &crate::maths::Vector,
        &crate::devices::TrialState,
    ),
) {
    let (matrix_a, rhs_a, trial_a) = first;
    let (matrix_b, rhs_b, trial_b) = second;
    let n = rhs_a.len();
    assert_eq!(rhs_b.len(), n);
    for row in 0..n {
        assert_eq!(
            rhs_a.get(row).unwrap().to_bits(),
            rhs_b.get(row).unwrap().to_bits(),
            "{label} row={row} RHS"
        );
        for col in 0..n {
            assert_eq!(
                matrix_a.get(row, col).to_bits(),
                matrix_b.get(row, col).to_bits(),
                "{label} row={row} col={col} MNA"
            );
        }
    }
    assert_eq!(
        trial_a
            .values()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        trial_b
            .values()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        "{label} trial"
    );
    assert_eq!(trial_a.is_nonconvergent(), trial_b.is_nonconvergent());
}

/// Obtain real accepted C-style junction values for subsequent predictor,
/// FIX and FLOAT comparisons rather than fabricating an unrealistic history.
fn seed_initial(
    circuit: &Circuit,
    history: &mut crate::devices::StateHistory,
    solution: &crate::maths::Vector,
    context: &crate::devices::ModelContext,
) {
    use crate::devices::{AnalysisMode, LoadRequest};
    let mut matrix = circuit.new_matrix().unwrap();
    let mut rhs = crate::maths::Vector::zeros(circuit.unknown_count());
    let mut trial = history.trial();
    circuit
        .load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution,
                model_context: context,
                integration: None,
                history,
                forcing: None,
            },
            &mut matrix,
            &mut rhs,
            &mut trial,
        )
        .unwrap();
    history.initialize(trial).unwrap();
}

/// Two independently finalized copies of the same circuit exercise
/// production `load_prepared` with and without the immutable batch plan.
/// The scalar path is not a reimplementation: it is the exact same
/// stamp loop with an empty homogeneous plan.
fn compare_diode_load(
    count: usize,
    model_groups: bool,
    phase: crate::devices::IterationPhase,
    limiting: bool,
    c_jacobian: bool,
    temperature: f64,
    bad_node: Option<&str>,
    rich: bool,
) {
    use crate::devices::{AnalysisMode, LoadRequest, ModelContext};
    use crate::maths::Vector;

    let (batched, scalar) = diode_circuits(count, model_groups, rich);
    let n = batched.unknown_count();
    let mut solution = Vector::zeros(n);
    for row in 0..n {
        solution
            .set(
                row,
                f64::from(u32::try_from(row % 17).unwrap()).mul_add(0.017, -0.12),
            )
            .unwrap();
    }
    if rich {
        for (name, voltage) in [
            ("n0", -6.),
            ("n1", 0.8),
            ("n2", -0.08),
            ("n3", 1.),
            ("n4", -5.5),
            ("n5", 0.55),
            ("n6", -0.3),
            ("n7", 0.),
        ] {
            if let Some(node) = batched.nodes.get(name)
                && let Some(row) = batched.bindings.unknowns.node_row(node)
            {
                solution.set(row, voltage).unwrap();
            }
        }
    }
    if let Some(name) = bad_node {
        let node = batched.nodes.get(name).unwrap();
        let row = batched.bindings.unknowns.node_row(node).unwrap();
        solution.set(row, 1_000.).unwrap();
    }
    let context = ModelContext::new(temperature, 27.);
    let mut history_a = batched.state_history().unwrap();
    let mut history_b = scalar.state_history().unwrap();

    // Predictor/limiter runs have a completed, equivalent accepted
    // physical state; compare the actual C MODEINIT path to obtain it.
    if phase != crate::devices::IterationPhase::Junction {
        seed_initial(&batched, &mut history_a, &solution, &context);
        seed_initial(&scalar, &mut history_b, &solution, &context);
    }
    let previous_a = (phase != crate::devices::IterationPhase::Junction).then(|| history_a.trial());
    let previous_b = (phase != crate::devices::IterationPhase::Junction).then(|| history_b.trial());
    let mut trial_a = if phase == crate::devices::IterationPhase::Junction {
        history_a.trial()
    } else {
        history_a.trial_in(phase, previous_a.as_ref()).unwrap()
    }
    .with_device_limiting(limiting)
    .with_c_jacobian(c_jacobian);
    let mut trial_b = if phase == crate::devices::IterationPhase::Junction {
        history_b.trial()
    } else {
        history_b.trial_in(phase, previous_b.as_ref()).unwrap()
    }
    .with_device_limiting(limiting)
    .with_c_jacobian(c_jacobian);
    let mut matrix_a = batched.new_matrix().unwrap();
    let mut matrix_b = scalar.new_matrix().unwrap();
    let mut rhs_a = Vector::zeros(n);
    let mut rhs_b = Vector::zeros(n);
    let result_a = batched.load(
        &LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &solution,
            model_context: &context,
            integration: None,
            history: &history_a,
            forcing: None,
        },
        &mut matrix_a,
        &mut rhs_a,
        &mut trial_a,
    );
    let result_b = scalar.load(
        &LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &solution,
            model_context: &context,
            integration: None,
            history: &history_b,
            forcing: None,
        },
        &mut matrix_b,
        &mut rhs_b,
        &mut trial_b,
    );
    assert_eq!(
        result_a.as_ref().err().map(ToString::to_string),
        result_b.as_ref().err().map(ToString::to_string),
        "count={count}"
    );
    assert_same_load(
        &format!("count={count}"),
        (&matrix_a, &rhs_a, &trial_a),
        (&matrix_b, &rhs_b, &trial_b),
    );
}

#[test]
fn diode_tiles_are_bitwise_equivalent_to_scalar_load_across_modes_and_sizes() {
    use crate::devices::IterationPhase;
    for &count in &[0, 1, 2, 3, 63, 64, 65, 66, 127, 128, 129, 131] {
        compare_diode_load(
            count,
            count % 2 == 1,
            IterationPhase::Junction,
            false,
            false,
            27.,
            None,
            false,
        );
    }
    for &phase in &[
        IterationPhase::Junction,
        IterationPhase::Fix,
        IterationPhase::Float,
        IterationPhase::Predict,
    ] {
        compare_diode_load(68, true, phase, true, true, -35., None, false);
        compare_diode_load(129, false, phase, false, false, 120., None, false);
    }
}

#[test]
fn full_physical_diode_tiles_match_scalar_over_boundaries_and_phases() {
    use crate::devices::IterationPhase;
    for &count in &[4, 63, 64, 65, 129] {
        for &phase in &[
            IterationPhase::Junction,
            IterationPhase::Float,
            IterationPhase::Fix,
            IterationPhase::Predict,
        ] {
            compare_diode_load(count, count % 2 == 1, phase, true, true, -35., None, true);
        }
    }
}

#[test]
fn diode_tile_defers_later_lane_error_until_the_original_stamp_position() {
    use crate::devices::IterationPhase;
    // dn has no internal series node, so 1000 V overflows its physical
    // exponential. d0 in the same model group stamps successfully
    // before d5 fails; neither the error nor the prefix load can move.
    compare_diode_load(
        68,
        true,
        IterationPhase::Junction,
        false,
        false,
        27.,
        Some("n5"),
        false,
    );
}

/// A parameter walk must not invalidate the topology-bound physical tile.
/// A new prepared point may own replacement diode recipes, but every batch
/// still commits at the original order and each new context has its own
/// thermal inputs without changing the model family or state layout.
#[test]
fn diode_tiles_follow_swept_instance_replacements_without_stale_physics() {
    use crate::devices::{AnalysisMode, LoadRequest, ModelContext};
    use crate::maths::Vector;

    let (batched, scalar) = diode_circuits(77, false, false);
    assert_eq!(
        batched
            .bindings
            .diode_batches
            .iter()
            .map(|b| b.len)
            .sum::<usize>(),
        77
    );
    let history_a = batched.state_history().unwrap();
    let history_b = scalar.state_history().unwrap();
    let run_a = batched.prepare_run();
    let run_b = scalar.prepare_run();
    let n = batched.unknown_count();
    let mut bias = Vector::zeros(n);
    for row in 0..n {
        bias.set(
            row,
            (f64::from(u32::try_from(row % 9).unwrap()) - 4.) * 0.025,
        )
        .unwrap();
    }
    let base = ModelContext::default();
    for &(area, temp) in &[(1., 27.), (0.25, -10.), (3., 110.), (0.25, -10.), (1., 27.)] {
        let effective = |circuit: &Circuit| {
            let context = base
                .with_instance_override(
                    circuit
                        .instance_override("d5", "area", area, &base)
                        .unwrap(),
                )
                .unwrap();
            context
                .with_instance_override(
                    circuit
                        .instance_override("d13", "temp", temp, &context)
                        .unwrap(),
                )
                .unwrap()
        };
        let ctx_a = effective(&batched);
        let ctx_b = effective(&scalar);
        let mut matrix_a = batched.new_matrix().unwrap();
        let mut matrix_b = scalar.new_matrix().unwrap();
        let mut rhs_a = Vector::zeros(n);
        let mut rhs_b = Vector::zeros(n);
        let mut trial_a = history_a.trial();
        let mut trial_b = history_b.trial();
        run_a
            .load(
                &LoadRequest {
                    mode: AnalysisMode::OperatingPoint,
                    solution: &bias,
                    model_context: &ctx_a,
                    history: &history_a,
                    forcing: None,
                    integration: None,
                },
                &mut matrix_a,
                &mut rhs_a,
                &mut trial_a,
            )
            .unwrap();
        run_b
            .load(
                &LoadRequest {
                    mode: AnalysisMode::OperatingPoint,
                    solution: &bias,
                    model_context: &ctx_b,
                    history: &history_b,
                    forcing: None,
                    integration: None,
                },
                &mut matrix_b,
                &mut rhs_b,
                &mut trial_b,
            )
            .unwrap();
        assert_same_load(
            &format!("area={area} temp={temp}"),
            (&matrix_a, &rhs_a, &trial_a),
            (&matrix_b, &rhs_b, &trial_b),
        );
    }
}

/// Opt-in end-to-end circuit *load and solve* cost for 1..4096 common
/// physical diodes. The same finalized graph is used in each mode; the
/// only difference is clearing the immutable homogeneous execution plan.
/// This tests small-circuit overhead as well as large-device throughput.
#[test]
#[ignore = "manual scalar-versus-tiled diode load and solve throughput"]
fn diode_tile_load_and_solve_benchmark() {
    use crate::devices::{AnalysisMode, LoadRequest, ModelContext};
    use crate::maths::Vector;
    use crate::netlist::{Parser, source::parse_deck_text};
    use std::{fmt::Write, hint::black_box, path::Path, time::Instant};

    for family in ["simple", "full"] {
        for count in [1, 4, 64, 256, 1_024, 4_096] {
            let model = if family == "full" {
                "is=1e-14 n=1.1 rs=2 cjo=30p tt=5n vj=0.8 m=0.4 fc=0.5              bv=5.1 ibv=5m nbv=1.4 jsw=2e-15 ns=1.6 cjsw=4p              isr=1e-11 nr=2.2 ikf=1m ikr=1u ikp=1u              jtun=1e-13 jtunsw=1e-14 ntun=20"
            } else {
                "is=1e-14 cjo=2p tt=3n"
            };
            let mut deck =
                format!("diode benchmark\nv0 n0 0 dc 0\nr0 n0 n1 100\n.model dm d({model})\n");
            for index in 0..count {
                if family == "full" {
                    writeln!(&mut deck, "d{index} n{} 0 dm pj=4", index % 8).unwrap();
                } else {
                    writeln!(&mut deck, "d{index} n{} 0 dm", index % 8).unwrap();
                }
            }
            deck.push_str(".op\n.end\n");
            let parsed = Parser::new()
                .parse_deck(&parse_deck_text(Path::new("tile-benchmark.cir"), &deck))
                .unwrap();
            let context = ModelContext::default();
            for batched in [false, true] {
                let mut circuit = Circuit::from_netlist(&parsed).unwrap();
                let lanes = circuit
                    .bindings
                    .diode_batches
                    .iter()
                    .map(|b| b.len)
                    .sum::<usize>();
                if !batched {
                    circuit.bindings.diode_batches.clear();
                }
                let history = circuit.state_history().unwrap();
                let run = circuit.prepare_run();
                let solution = Vector::zeros(circuit.unknown_count());
                let request = LoadRequest {
                    mode: AnalysisMode::OperatingPoint,
                    solution: &solution,
                    model_context: &context,
                    integration: None,
                    history: &history,
                    forcing: None,
                };
                let mut matrix = circuit.new_matrix().unwrap();
                let mut rhs = Vector::zeros(circuit.unknown_count());
                let trials = (65_536 / count).max(32);
                let mut records = Vec::with_capacity(5);
                for repeat in 0..6 {
                    let mut load_ns = 0_u128;
                    let mut solve_ns = 0_u128;
                    for _ in 0..trials {
                        matrix.clear();
                        rhs.clear();
                        let mut trial = history.trial();
                        let start = Instant::now();
                        run.load(&request, &mut matrix, &mut rhs, &mut trial)
                            .unwrap();
                        load_ns += start.elapsed().as_nanos();
                        black_box(&trial);
                        let start = Instant::now();
                        black_box(matrix.solve(&rhs).unwrap());
                        solve_ns += start.elapsed().as_nanos();
                    }
                    if repeat > 0 {
                        records.push((load_ns, solve_ns));
                    }
                }
                let mut loads = records.iter().map(|&(load, _)| load).collect::<Vec<_>>();
                let mut solves = records.iter().map(|&(_, solve)| solve).collect::<Vec<_>>();
                loads.sort_unstable();
                solves.sort_unstable();
                println!(
                    "diode_family={family},diode_count={count},active_lanes={lanes},batched={batched},load_ns_per_trial={},solve_ns_per_trial={},total_ns_per_trial={}",
                    loads[2] / trials as u128,
                    solves[2] / trials as u128,
                    (loads[2] + solves[2]) / trials as u128
                );
            }
        }
    }
}
