//! #113: independent candidate worker ownership and bounded deterministic scheduling.
use std::num::NonZeroUsize;
use std::path::Path;

use ngspice_rs::analysis::optimization::objectives::{
    ObjectiveRequests, Observation, ObservationReduction, ObservationValue, PlotObjectivePlan,
    ScenarioPlan,
};
use ngspice_rs::analysis::optimization::parameters::{CandidateTopology, CircuitParameterPlan};
use ngspice_rs::analysis::optimization::sessions::{
    CandidateFailure, CandidateJob, CandidateResult, CandidateScenario, CandidateWorker,
    ScenarioAnalyses, ScenarioStart, run_parallel,
};
use ngspice_rs::analysis::{AnalysisInput, AnalysisPlan, RunConfig};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::{Netlist, RequestedVector, VectorRequest};
use ngspice_rs::netlist::eval::EvalLimits;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, Real, SourceLoc};

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("optimization-sessions.cir"),
            &format!("Session execution\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn setup(body: &str, inputs: &[&str]) -> (CircuitParameterPlan, Vec<AnalysisPlan>, ScenarioPlan) {
    let source = deck(body);
    let config = RunConfig::from_netlist(&source).unwrap();
    let analyses = vec![
        config
            .plan(AnalysisInput::new(AnalysisKind::OperatingPoint))
            .unwrap(),
    ];
    let mut circuit = config.circuit(&source).unwrap();
    let full = analyses[0].run(&mut circuit, &config.context()).unwrap();
    let output = PlotObjectivePlan::bind(
        &full,
        AnalysisKind::OperatingPoint,
        ObjectiveRequests {
            observations: vec![Observation {
                name: "out".into(),
                vector: VectorRequest {
                    vector: RequestedVector::Named {
                        name: "v(out)".into(),
                        component: None,
                    },
                    location: SourceLoc::new(
                        Path::new("optimization-sessions.cir").to_path_buf(),
                        1,
                        1,
                    ),
                },
                reduction: ObservationReduction::Last,
            }],
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    (
        CircuitParameterPlan::new(source, inputs).unwrap(),
        analyses,
        ScenarioPlan::new(vec![output]),
    )
}

fn simple() -> (CircuitParameterPlan, Vec<AnalysisPlan>, ScenarioPlan) {
    setup(
        ".param r=1k\nv1 in 0 10\nr1 in out {r}\nr2 out 0 1k",
        &["r"],
    )
}

fn candidate(id: u64, resistance: Real) -> CandidateJob {
    CandidateJob {
        id,
        values: vec![resistance],
    }
}

fn output_value(outcome: &CandidateResult) -> Real {
    let plots = outcome.outcome.as_ref().unwrap();
    let obs = &plots[0][0].observations[0];
    let ObservationValue::Scalar(point) = obs.value.as_ref().unwrap() else {
        panic!("expected scalar observation");
    };
    point.value.re
}

fn independent<'a>(
    analyses: &'a [AnalysisPlan],
    outputs: &'a ScenarioPlan,
) -> CandidateScenario<'a> {
    CandidateScenario {
        start: ScenarioStart::Independent,
        analyses: ScenarioAnalyses::Prepared(analyses),
        objectives: outputs,
    }
}

#[test]
fn serial_owned_worker_keeps_aba_and_repeated_candidates_independent() {
    let (plan, analyses, outputs) = simple();
    let scenarios = [independent(&analyses, &outputs)];
    let mut worker = CandidateWorker::new(&plan, EvalLimits::default()).unwrap();
    let mut received = Vec::new();
    let stats = worker
        .run_bounded(
            [
                candidate(20, 2000.),
                candidate(21, 5000.),
                candidate(20, 2000.),
                candidate(22, 2000.),
            ],
            &scenarios,
            NonZeroUsize::new(10).unwrap(),
            |outcome| received.push(outcome),
        )
        .unwrap();
    assert_eq!(stats.candidates, 4);
    assert_eq!(stats.succeeded, 4);
    assert_eq!(stats.failed, 0);
    assert_eq!(stats.rebuilt, 0);
    assert_eq!(stats.peak_active_workers, 1);
    assert_eq!(stats.peak_outstanding, 1);
    assert_eq!(stats.peak_reorder_pending, 0);
    assert_eq!(
        received.iter().map(|result| result.id).collect::<Vec<_>>(),
        [20, 21, 20, 22]
    );
    for item in &received {
        assert_eq!(item.topology, Some(CandidateTopology::Reused));
    }
    assert_eq!(output_value(&received[0]), output_value(&received[2]));
    assert_eq!(output_value(&received[2]), output_value(&received[3]));
    assert!((output_value(&received[0]) - 10. / 3.).abs() < 1e-11);
    assert!((output_value(&received[1]) - 10. / 6.).abs() < 1e-11);
}

#[test]
fn failures_are_per_candidate_and_do_not_leak_into_following_runs() {
    let (plan, analyses, outputs) = setup(
        ".param x=2\nv1 in 0 10\nr1 in out {1000+1/(x-5)}\nr2 out 0 1000",
        &["x"],
    );
    let scenarios = [independent(&analyses, &outputs)];
    let mut worker = CandidateWorker::new(&plan, EvalLimits::default()).unwrap();
    let mut received = Vec::new();
    let stats = worker
        .run_bounded(
            [
                candidate(1, 3.),
                candidate(2, 5.),
                candidate(3, 3.),
                candidate(4, 4.),
            ],
            &scenarios,
            NonZeroUsize::new(4).unwrap(),
            |result| received.push(result),
        )
        .unwrap();
    assert_eq!(stats.succeeded, 3);
    assert_eq!(stats.failed, 1);
    assert_eq!(received[1].topology, None);
    assert!(matches!(
        received[1].outcome,
        Err(CandidateFailure::Preparation(_))
    ));
    assert_eq!(
        output_value(&received[0]).to_bits(),
        output_value(&received[2]).to_bits()
    );
    assert!(received[3].outcome.is_ok());
}

#[test]
fn candidate_resolved_inputs_reapply_numeric_context_per_job() {
    let (plan, _, outputs) = simple();
    let requests = [AnalysisInput::new(AnalysisKind::OperatingPoint)];
    let scenarios = [CandidateScenario {
        start: ScenarioStart::Independent,
        analyses: ScenarioAnalyses::Candidate(&requests),
        objectives: &outputs,
    }];
    let mut worker = CandidateWorker::new(&plan, EvalLimits::default()).unwrap();
    let first = worker.evaluate(&candidate(5, 1000.), &scenarios);
    let next = worker.evaluate(&candidate(6, 4000.), &scenarios);
    assert!(first.outcome.is_ok());
    assert!((output_value(&next) - 2.).abs() < 1e-11);
}

#[test]
fn multiple_scenarios_enforce_independent_reset_or_explicit_continuation() {
    let (plan, analyses, outputs) = simple();
    let scenarios = [
        independent(&analyses, &outputs),
        CandidateScenario {
            start: ScenarioStart::Continue,
            analyses: ScenarioAnalyses::Prepared(&analyses),
            objectives: &outputs,
        },
        independent(&analyses, &outputs),
    ];
    let mut worker = CandidateWorker::new(&plan, EvalLimits::default()).unwrap();
    let one = worker.evaluate(&candidate(1, 2500.), &scenarios);
    assert_eq!(one.outcome.as_ref().unwrap().len(), 3);
    let values = one.outcome.unwrap();
    for result in &values {
        let ObservationValue::Scalar(point) = result[0].observations[0].value.as_ref().unwrap()
        else {
            panic!("expected scalar");
        };
        assert!((point.value.re - 10. / 3.5).abs() < 1e-11);
    }
    let invalid = [CandidateScenario {
        start: ScenarioStart::Continue,
        analyses: ScenarioAnalyses::Prepared(&analyses),
        objectives: &outputs,
    }];
    let error = worker.run_bounded(
        [candidate(1, 2500.)],
        &invalid,
        NonZeroUsize::new(1).unwrap(),
        |_| {},
    );
    assert!(error.is_err());
}

#[test]
fn parallel_stream_preserves_order_and_matches_serial_numeric_results() {
    let (plan, analyses, outputs) = simple();
    let scenarios = [independent(&analyses, &outputs)];
    let work = (0..37)
        .map(|i| candidate(1000 + i, 1000. + (i % 13) as Real * 17.))
        .collect::<Vec<_>>();
    let mut serial = CandidateWorker::new(&plan, EvalLimits::default()).unwrap();
    let mut serial_values = Vec::new();
    serial
        .run_bounded(
            work.iter().map(|item| candidate(item.id, item.values[0])),
            &scenarios,
            NonZeroUsize::new(40).unwrap(),
            |result| serial_values.push((result.id, output_value(&result).to_bits())),
        )
        .unwrap();
    let mut parallel_values = Vec::new();
    let stats = run_parallel(
        &plan,
        &scenarios,
        work,
        NonZeroUsize::new(4).unwrap(),
        NonZeroUsize::new(40).unwrap(),
        EvalLimits::default(),
        |result| parallel_values.push((result.id, output_value(&result).to_bits())),
    )
    .unwrap();
    assert_eq!(stats.candidates, 37);
    assert_eq!(stats.failed, 0);
    assert!(stats.peak_active_workers >= 1 && stats.peak_active_workers <= 4);
    assert!(stats.peak_outstanding >= stats.peak_active_workers);
    assert!(stats.peak_outstanding <= 8);
    assert!(stats.peak_reorder_pending <= stats.peak_outstanding);
    assert_eq!(parallel_values, serial_values);
}

#[test]
fn parallel_tail_resource_limit_and_rebuilt_topology_are_explicit() {
    let (plan, analyses, outputs) = setup(
        ".param a=1k b=2k c=3k d=4k\nv1 in 0 10\nr1 in out {a}\nr2 out 0 {b}\nr3 out 0 {c}\nr4 out 0 {d}",
        &["a", "b", "c", "d"],
    );
    let scenarios = [independent(&analyses, &outputs)];
    let jobs = (0..9)
        .map(|i| CandidateJob {
            id: i,
            values: vec![
                1000. + i as Real,
                2000. + i as Real,
                3000. + i as Real,
                4000. + i as Real,
            ],
        })
        .collect::<Vec<_>>();
    let mut results = Vec::new();
    let stats = run_parallel(
        &plan,
        &scenarios,
        jobs,
        NonZeroUsize::new(3).unwrap(),
        NonZeroUsize::new(9).unwrap(),
        EvalLimits::default(),
        |result| results.push((result.id, result.topology)),
    )
    .unwrap();
    assert_eq!(stats.candidates, 9);
    assert_eq!(results.len(), 9);
    assert_eq!(results[0].1, Some(CandidateTopology::Reused));
    assert_eq!(
        stats.rebuilt, 0,
        "four safe scalar setters must reuse topology"
    );
    assert_eq!(stats.failed, 0);
    let mut called = false;
    let result = run_parallel(
        &plan,
        &scenarios,
        (0..9).map(|i| CandidateJob {
            id: i,
            values: vec![1000., 2000., 3000., 4000.],
        }),
        NonZeroUsize::new(3).unwrap(),
        NonZeroUsize::new(8).unwrap(),
        EvalLimits::default(),
        |_| called = true,
    );
    assert!(result.is_err());
    assert!(!called);
}

#[test]
fn deck_analysis_expressions_rebind_per_candidate_without_stale_timing() {
    let source = deck(
        ".param r=1k stop=10u\nv1 in 0 10\nr1 in out {r}\nr2 out 0 1k\nc1 out 0 1u\n.tran 1u {stop}",
    );
    let cards = source.analyses.clone();
    let config = RunConfig::from_netlist(&source).unwrap();
    let analysis = config.plan_for(&cards[0]).unwrap();
    let mut baseline = config.circuit(&source).unwrap();
    let header = analysis.run(&mut baseline, &config.context()).unwrap();
    let objective = PlotObjectivePlan::bind(
        &header,
        AnalysisKind::Transient,
        ObjectiveRequests {
            observations: vec![Observation {
                name: "transient".into(),
                vector: VectorRequest {
                    vector: RequestedVector::Named {
                        name: "v(out)".into(),
                        component: None,
                    },
                    location: SourceLoc::new(
                        Path::new("optimization-sessions.cir").to_path_buf(),
                        1,
                        1,
                    ),
                },
                reduction: ObservationReduction::Last,
            }],
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    let outputs = ScenarioPlan::new(vec![objective]);
    let plan = CircuitParameterPlan::new(source, &["r", "stop"]).unwrap();
    let scenarios = [CandidateScenario {
        start: ScenarioStart::Independent,
        analyses: ScenarioAnalyses::Deck(&cards),
        objectives: &outputs,
    }];
    let mut worker = CandidateWorker::new(&plan, EvalLimits::default()).unwrap();
    let mut last_times = Vec::new();
    for (id, stop) in [(5, 10e-6), (6, 20e-6), (7, 10e-6)] {
        let result = worker.evaluate(
            &CandidateJob {
                id,
                values: vec![1000., stop],
            },
            &scenarios,
        );
        let result = result.outcome.unwrap();
        let ObservationValue::Scalar(point) = result[0][0].observations[0].value.as_ref().unwrap()
        else {
            panic!("expected transient scalar");
        };
        last_times.push(point.coordinate.unwrap());
    }
    assert!((last_times[0] - 10e-6).abs() < 1e-14);
    assert!((last_times[1] - 20e-6).abs() < 1e-14);
    assert_eq!(last_times[0].to_bits(), last_times[2].to_bits());
}

#[test]
fn transient_uic_seeds_restart_independently_across_scenarios_and_jobs() {
    use ngspice_rs::analysis::NodeCondition;

    let source = deck(".param r=1k\nr1 a 0 {r}\nc1 a 0 1u");
    let config = RunConfig::from_netlist(&source).unwrap();
    let seed = |value: Real| {
        let mut input = AnalysisInput::with_arguments(AnalysisKind::Transient, ["10u", "1m"]);
        input.uic = true;
        input.initial_conditions = vec![NodeCondition {
            node: "a".into(),
            value,
            location: SourceLoc::new(Path::new("optimization-sessions.cir").to_path_buf(), 1, 1),
        }];
        input
    };
    let (positive, negative) = ([seed(2.)], [seed(-2.)]);
    let mut baseline = config.circuit(&source).unwrap();
    let analysis = config.plan(positive[0].clone()).unwrap();
    let plot = analysis.run(&mut baseline, &config.context()).unwrap();
    let requests = ObjectiveRequests {
        observations: vec![Observation {
            name: "voltage".into(),
            vector: VectorRequest {
                vector: RequestedVector::Named {
                    name: "v(a)".into(),
                    component: None,
                },
                location: SourceLoc::new(
                    Path::new("optimization-sessions.cir").to_path_buf(),
                    1,
                    1,
                ),
            },
            reduction: ObservationReduction::Last,
        }],
        ..ObjectiveRequests::default()
    };
    let objectives = ScenarioPlan::new(vec![
        PlotObjectivePlan::bind(&plot, AnalysisKind::Transient, requests).unwrap(),
    ]);
    let plan = CircuitParameterPlan::new(source, &["r"]).unwrap();
    let scenarios = [
        CandidateScenario {
            start: ScenarioStart::Independent,
            analyses: ScenarioAnalyses::Candidate(&positive),
            objectives: &objectives,
        },
        CandidateScenario {
            start: ScenarioStart::Independent,
            analyses: ScenarioAnalyses::Candidate(&negative),
            objectives: &objectives,
        },
    ];
    let mut worker = CandidateWorker::new(&plan, EvalLimits::default()).unwrap();
    let calculate = |value: &ngspice_rs::analysis::optimization::objectives::PlotObjectives| {
        let ObservationValue::Scalar(point) = value.observations[0].value.as_ref().unwrap() else {
            panic!("transient scalar missing");
        };
        point.value.re
    };
    let first = worker.evaluate(&candidate(42, 1000.), &scenarios);
    let second = worker.evaluate(&candidate(43, 2000.), &scenarios);
    let third = worker.evaluate(&candidate(44, 1000.), &scenarios);
    let a = first.outcome.unwrap();
    let b = second.outcome.unwrap();
    let c = third.outcome.unwrap();
    let positive_a = calculate(&a[0][0]);
    let negative_a = calculate(&a[1][0]);
    assert!(positive_a > 0. && negative_a < 0.);
    assert!((positive_a + negative_a).abs() < 1e-11);
    assert!(calculate(&b[0][0]) > positive_a);
    assert_eq!(positive_a.to_bits(), calculate(&c[0][0]).to_bits());
    assert_eq!(negative_a.to_bits(), calculate(&c[1][0]).to_bits());
}

#[test]
#[ignore = "manual: benchmark persistent workers against scalar re-elaboration"]
fn benchmark_ten_thousand_independent_candidates() {
    let (plan, analyses, outputs) = simple();
    let scenarios = [independent(&analyses, &outputs)];
    for workers in [1, 4] {
        let start = std::time::Instant::now();
        let mut count = 0;
        let stats = run_parallel(
            &plan,
            &scenarios,
            (0..10_000).map(|i| candidate(i, 1000. + (i % 1000) as Real)),
            NonZeroUsize::new(workers).unwrap(),
            NonZeroUsize::new(10_000).unwrap(),
            EvalLimits::default(),
            |result| {
                assert!(result.outcome.is_ok());
                count += 1;
            },
        )
        .unwrap();
        assert_eq!(count, 10_000);
        eprintln!(
            "workers={workers} elapsed={:?} candidates={} rebuilt={}",
            start.elapsed(),
            stats.candidates,
            stats.rebuilt
        );
    }
}
