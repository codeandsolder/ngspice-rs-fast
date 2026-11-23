//! Replayable grid-plan regression: exact stepping semantics and aggregate work.
use ngspice_rs::analysis::sweep::{SweepSpec, SweepTarget};
use ngspice_rs::analysis::{AnalysisInput, PlanningLimits, RunConfig};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::AnalysisKind;
use std::path::Path;

fn plan(kind: AnalysisKind, arguments: &str) -> ngspice_rs::analysis::AnalysisPlan {
    AnalysisInput::with_arguments(kind, arguments.split_whitespace())
        .resolve()
        .unwrap()
}

fn bits(points: impl Iterator<Item = f64>) -> Vec<u64> {
    points.map(f64::to_bits).collect()
}

#[test]
fn spectral_descriptors_replay_identically_without_storing_coordinates() {
    for (kind, args) in [
        (AnalysisKind::Ac, "lin 3 0.1 0.3"),
        (AnalysisKind::Ac, "dec 10 1 1000"),
        (AnalysisKind::Ac, "oct 4 1 64"),
        (AnalysisKind::SParameter, "lin 3 1 3"),
        (AnalysisKind::SParameter, "dec 4 1 100"),
        (AnalysisKind::Noise, "v(out) v1 lin 3 1 3"),
        (AnalysisKind::Distortion, "lin 3 1k 4k"),
        (AnalysisKind::Sensitivity, "v ( out ) ac dec 1 1k 100k"),
        (AnalysisKind::Sensitivity, "v ( out ) ac lin 2 1k 2k"),
    ] {
        let p = plan(kind, args);
        let iter = p.frequencies().unwrap();
        let n = iter.len();
        let first = bits(iter.clone());
        assert_eq!(first.len(), n);
        assert_eq!(first, bits(iter.clone()), "{kind:?} {args}");
        assert_eq!(first, bits(p.frequencies().unwrap()), "{kind:?} {args}");
    }
    let small = plan(AnalysisKind::Ac, "lin 3 1 3");
    let big = plan(AnalysisKind::Ac, "lin 100000 1 100000");
    assert_eq!(std::mem::size_of_val(&small), std::mem::size_of_val(&big));
    assert_eq!(big.frequencies().unwrap().len(), 100_000);
}

#[test]
fn repeated_addition_and_celsius_kelvin_accumulation_remain_distinct() {
    let ac = plan(AnalysisKind::Ac, "lin 3 0.1 0.3");
    let step = (0.3_f64 - 0.1) / 2.;
    let want = bits([0.1_f64, 0.1 + step, (0.1 + step) + step].into_iter());
    assert_eq!(bits(ac.frequencies().unwrap()), want);

    let descending = SweepSpec {
        target: SweepTarget::VoltageSource("v1".into()),
        start: 0.3,
        stop: 0.,
        step: -0.1,
    };
    let exact = descending.grid().unwrap();
    let accumulated = descending.accumulated_grid().unwrap();
    assert_eq!(exact[3].to_bits(), 0_f64.to_bits());
    assert_eq!(
        accumulated[3].to_bits(),
        (((0.3_f64 - 0.1) - 0.1) - 0.1).to_bits()
    );
    assert_ne!(exact[3].to_bits(), accumulated[3].to_bits());

    let kelvin = SweepSpec {
        target: SweepTarget::Temperature,
        start: 0.,
        stop: 1e-12,
        step: 1e-15,
    };
    assert!(
        kelvin
            .accumulated_grid()
            .unwrap_err()
            .to_string()
            .contains("makes no progress")
    );
}

#[test]
fn aggregate_limit_is_checked_before_a_deck_can_run() {
    let text = "budget\nv1 a 0 1\nr1 a 0 1k\n.ac lin 4 1 4\n.ac lin 4 1 4\n.ac lin 4 1 4\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("budget.cir"), text))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    assert_eq!(netlist.analyses.len(), 3);
    assert_eq!(
        config
            .plans_for_with_limits(
                &netlist.analyses,
                PlanningLimits {
                    max_grid_points: 12,
                    ..PlanningLimits::default()
                }
            )
            .unwrap()
            .len(),
        3
    );
    let err = config
        .plans_for_with_limits(
            &netlist.analyses,
            PlanningLimits {
                max_grid_points: 11,
                ..PlanningLimits::default()
            },
        )
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("aggregate deterministic analysis grid")
    );
    assert!(
        config
            .plans_for_with_limits(
                &netlist.analyses,
                PlanningLimits {
                    max_grid_points: 0,
                    ..PlanningLimits::default()
                }
            )
            .is_err()
    );
}

#[test]
fn all_grid_input_errors_are_rejected_at_resolution() {
    for (kind, args) in [
        (AnalysisKind::Ac, "lin 100000 1e20 1.0000000000000002e20"),
        (AnalysisKind::Ac, "dec 3 0 1k"),
        (
            AnalysisKind::Noise,
            "v(out) v1 lin 100000 1e20 1.000000000000001e20",
        ),
        (AnalysisKind::Sensitivity, "v ( out ) ac lin 2 1k 1k"),
        (AnalysisKind::DcSweep, "temp 0 1e-12 1e-15"),
        (AnalysisKind::DcSweep, "v1 0 1e6 1"),
    ] {
        assert!(
            AnalysisInput::with_arguments(kind, args.split_whitespace())
                .resolve()
                .is_err(),
            "invalid request accepted: {kind:?} {args}"
        );
    }
}

#[test]
fn aggregate_card_cap_applies_to_jobs_without_grid_coordinates() {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("many-ops.cir"),
            "op cards\nr1 a 0 1k\n.op\n.op\n.op\n.end\n",
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    assert_eq!(
        config
            .plans_for_with_limits(
                &netlist.analyses,
                PlanningLimits {
                    max_jobs: 3,
                    max_grid_points: 0
                },
            )
            .unwrap()
            .len(),
        3
    );
    assert!(
        config
            .plans_for_with_limits(
                &netlist.analyses,
                PlanningLimits {
                    max_jobs: 2,
                    max_grid_points: 0
                },
            )
            .unwrap_err()
            .to_string()
            .contains("aggregate analysis job limit")
    );
}

#[test]
fn large_multi_analysis_deck_uses_descriptor_sized_storage() {
    // The input contains 10.1 million requested frequency coordinates:
    // 100 cards fit the default work budget, 101 do not.
    let mut text = String::from("many grids\nv1 a 0 1\nr1 a 0 1k\n");
    for _ in 0..101 {
        text.push_str(".ac lin 100000 1 100000\n");
    }
    text.push_str(".end\n");
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("many-grids.cir"), &text))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let started = std::time::Instant::now();
    let plans = config.plans_for(&netlist.analyses[..100]).unwrap();
    let elapsed = started.elapsed();
    assert_eq!(plans.len(), 100);
    assert_eq!(
        plans
            .iter()
            .map(|p| p.frequencies().unwrap().len())
            .sum::<usize>(),
        10_000_000
    );
    assert!(std::mem::size_of_val(plans.as_slice()) < 1_000_000);
    assert!(config.plans_for(&netlist.analyses).is_err());
    println!(
        "100 x 100k-point .ac plans validated in {elapsed:?}; inline plan bytes={}",
        std::mem::size_of_val(plans.as_slice())
    );
}

#[test]
fn aggregate_budget_counts_nested_dc_products_and_excludes_adaptive_time_steps() {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("mixed-grids.cir"),
            "mixed\nv1 a 0 1\nv2 b 0 2\nr1 a b 1k\n.dc v1 0 4 1 v2 0 2 1\n.tran 1u 1m\n.end\n",
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    assert_eq!(
        config
            .plans_for_with_limits(
                &netlist.analyses,
                PlanningLimits {
                    max_grid_points: 15,
                    ..PlanningLimits::default()
                }
            )
            .unwrap()
            .len(),
        2
    );
    assert!(
        config
            .plans_for_with_limits(
                &netlist.analyses,
                PlanningLimits {
                    max_grid_points: 14,
                    ..PlanningLimits::default()
                }
            )
            .is_err()
    );
}
