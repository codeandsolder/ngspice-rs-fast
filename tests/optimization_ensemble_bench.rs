//! Opt-in large independent optimization-candidate sweeps (#116).
//! The generator, worker scheduler and objective sink remain bounded even
//! at 100k candidates. The optimizer and any search heuristics are external.

use std::num::NonZeroUsize;
use std::path::Path;
use std::time::Instant;

use ngspice_rs::analysis::optimization::objectives::{
    ObjectiveRequests, Observation, ObservationReduction, ObservationValue, PlotObjectivePlan,
    ScenarioPlan,
};
use ngspice_rs::analysis::optimization::parameters::CircuitParameterPlan;
use ngspice_rs::analysis::optimization::sessions::{
    CandidateJob, CandidateScenario, EnsembleStatistics, ScenarioAnalyses, ScenarioStart,
    run_parallel,
};
use ngspice_rs::analysis::{AnalysisInput, AnalysisPlan, RunConfig};
use ngspice_rs::netlist::ast::{RequestedVector, VectorRequest};
use ngspice_rs::netlist::eval::EvalLimits;
use ngspice_rs::netlist::{Parser, ast::Netlist, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, SourceLoc};

#[derive(Debug, Clone, Copy)]
enum Scenario {
    Dc,
    Ac,
    Transient,
}

#[derive(Debug, Clone, Copy)]
enum Directions {
    Coordinate,
    SparseThree,
    Dense,
}

fn baseline(index: usize) -> f64 {
    1000. + index as f64 * 7.
}

fn direction(seed: u64, dimension: usize) -> f64 {
    let mut value = seed ^ (dimension as u64).wrapping_mul(0x9e3779b97f4a7c15);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^= value >> 31;
    ((value % 2001) as f64 - 1000.) * 0.000025
}

fn job(index: u64, dimensions: usize, mode: Directions) -> CandidateJob {
    let seed = index / 2;
    let sign = if index % 2 == 0 { 1. } else { -1. };
    let values = (0..dimensions)
        .map(|axis| {
            let selected = match mode {
                Directions::Coordinate => axis == seed as usize % dimensions,
                Directions::SparseThree => {
                    (0..3).any(|slot| axis == (seed as usize + slot * 7) % dimensions)
                }
                Directions::Dense => true,
            };
            let delta = if selected {
                sign * direction(seed, axis)
            } else {
                0.
            };
            baseline(axis) * (1. + delta)
        })
        .collect();
    CandidateJob { id: index, values }
}

fn setup(
    dimensions: usize,
    kind: Scenario,
) -> (CircuitParameterPlan, Vec<AnalysisPlan>, ScenarioPlan) {
    let mut body = String::from("Candidate ensemble\nv1 in 0 dc 10 ac 1\n");
    let mut tunables = Vec::with_capacity(dimensions);
    for axis in 0..dimensions {
        let name = format!("p{axis}");
        body.push_str(&format!(".param {name}={}\n", baseline(axis)));
        let (positive, negative) = if axis == 0 {
            ("in", "out")
        } else {
            ("out", "0")
        };
        body.push_str(&format!("r{axis} {positive} {negative} {{{name}}}\n"));
        tunables.push(name);
    }
    body.push_str("d1 out 0 nonlinear\n.model nonlinear d(is=1e-12 n=1.4 cjo=1e-12)\n.end\n");
    let source: Netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("optimization-cloud.cir"), &body))
        .unwrap();
    let config = RunConfig::from_netlist(&source).unwrap();
    let request = match kind {
        Scenario::Dc => AnalysisInput::new(AnalysisKind::OperatingPoint),
        Scenario::Ac => {
            AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "8", "100", "8000"])
        }
        Scenario::Transient => {
            AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "40u", "0", "500n"])
        }
    };
    let analysis = config.plan(request).unwrap();
    let mut circuit = config.circuit(&source).unwrap();
    let reference = analysis.run(&mut circuit, &config.context()).unwrap();
    let outputs = ScenarioPlan::new(vec![
        PlotObjectivePlan::bind(
            &reference,
            match kind {
                Scenario::Dc => AnalysisKind::OperatingPoint,
                Scenario::Ac => AnalysisKind::Ac,
                Scenario::Transient => AnalysisKind::Transient,
            },
            ObjectiveRequests {
                observations: vec![Observation {
                    name: "output".into(),
                    vector: VectorRequest {
                        vector: RequestedVector::Named {
                            name: "v(out)".into(),
                            component: None,
                        },
                        location: SourceLoc::new(
                            Path::new("optimization-cloud.cir").to_path_buf(),
                            1,
                            1,
                        ),
                    },
                    reduction: ObservationReduction::Last,
                }],
                ..ObjectiveRequests::default()
            },
        )
        .unwrap(),
    ]);
    let inputs: Vec<&str> = tunables.iter().map(String::as_str).collect();
    let plan = CircuitParameterPlan::new(source, &inputs).unwrap();
    (plan, vec![analysis], outputs)
}

fn run(
    dimensions: usize,
    mode: Directions,
    count: usize,
    workers: usize,
    scenario: Scenario,
) -> (EnsembleStatistics, u64, f64) {
    let (plan, analyses, objectives) = setup(dimensions, scenario);
    let scenarios = [CandidateScenario {
        start: ScenarioStart::Independent,
        analyses: ScenarioAnalyses::Prepared(&analyses),
        objectives: &objectives,
    }];
    let mut digest = 0_u64;
    let mut expected_id = 0_u64;
    let start = Instant::now();
    let statistics = run_parallel(
        &plan,
        &scenarios,
        (0..count as u64).map(|index| job(index, dimensions, mode)),
        NonZeroUsize::new(workers).unwrap(),
        NonZeroUsize::new(count).unwrap(),
        EvalLimits::default(),
        |result| {
            assert_eq!(result.id, expected_id);
            expected_id += 1;
            let sample = result
                .outcome
                .ok()
                .and_then(|scenarios| {
                    let observation = &scenarios[0][0].observations[0];
                    match observation.value.as_ref().ok()? {
                        ObservationValue::Scalar(point) => Some(point.value.re.to_bits()),
                        _ => None,
                    }
                })
                .unwrap_or(u64::MAX);
            digest = digest.wrapping_mul(0x9e3779b185ebca87).wrapping_add(sample);
        },
    )
    .unwrap();
    let seconds = start.elapsed().as_secs_f64();
    assert_eq!(statistics.candidates, count);
    assert_eq!(expected_id, count as u64);
    assert!(statistics.peak_active_workers <= workers);
    assert!(statistics.peak_outstanding <= workers * 2);
    assert!(statistics.peak_reorder_pending <= statistics.peak_outstanding);
    (statistics, digest, seconds)
}

fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmHWM:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

#[test]
fn bounded_candidate_cloud_smoke() {
    for (dimensions, mode) in [
        (8, Directions::Coordinate),
        (8, Directions::SparseThree),
        (64, Directions::Dense),
        (128, Directions::Coordinate),
    ] {
        let (serial, a, _) = run(dimensions, mode, 16, 1, Scenario::Dc);
        let (parallel, b, _) = run(dimensions, mode, 16, 4, Scenario::Dc);
        assert_eq!(a, b);
        assert_eq!(serial.succeeded, parallel.succeeded);
        assert_eq!(serial.failed, parallel.failed);
        assert_eq!(serial.rebuilt, parallel.rebuilt);
        assert_eq!(serial.peak_active_workers, 1);
        assert_eq!(serial.peak_reorder_pending, 0);
    }
}

#[test]
#[ignore = "manual release benchmark: 1k/10k/100k independent candidates"]
fn large_independent_candidate_cloud() {
    for (dimensions, mode, count, scenario) in [
        (8, Directions::Coordinate, 100_000, Scenario::Dc),
        (8, Directions::SparseThree, 10_000, Scenario::Dc),
        (64, Directions::Coordinate, 10_000, Scenario::Dc),
        (64, Directions::SparseThree, 10_000, Scenario::Dc),
        (64, Directions::Dense, 1_000, Scenario::Dc),
        (128, Directions::Dense, 1_000, Scenario::Ac),
        (8, Directions::SparseThree, 1_000, Scenario::Transient),
    ] {
        let mut reference = None;
        for workers in [1, 4] {
            let (stats, digest, elapsed) = run(dimensions, mode, count, workers, scenario);
            if let Some((baseline_digest, baseline_stats)) = reference {
                assert_eq!(digest, baseline_digest);
                assert_eq!(
                    (stats.succeeded, stats.failed, stats.rebuilt),
                    baseline_stats
                );
            } else {
                reference = Some((digest, (stats.succeeded, stats.failed, stats.rebuilt)));
            }
            println!(
                "workload=optimization_cloud,kind={scenario:?},dimensions={dimensions},directions={mode:?},candidates={count},workers={workers},elapsed_ns={},candidate_per_second={:.1},succeeded={},failed={},rebuilt={},peak_active={},peak_outstanding={},peak_reorder={},rss_peak_kib={:?},digest={digest:#018x}",
                (elapsed * 1e9) as u128,
                count as f64 / elapsed,
                stats.succeeded,
                stats.failed,
                stats.rebuilt,
                stats.peak_active_workers,
                stats.peak_outstanding,
                stats.peak_reorder_pending,
                peak_rss_kib(),
            );
        }
    }
}
