//! #140: source value sites share the #117 slot DAG and #112 candidate states.
use std::path::Path;
use std::sync::Arc;

use ngspice_rs::analysis::optimization::parameters::{CandidateTopology, CircuitParameterPlan};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::{Netlist, ParameterKind};
use ngspice_rs::netlist::eval::{
    EvalBudget, EvalLimits, FunctionScope, ParameterPlan, ParameterScopeSpec,
};
use ngspice_rs::netlist::source::parse_deck_text;

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("compiled-sites.cir"),
            &format!("Value site candidate tests\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn equivalent_matrix(
    candidate: &mut ngspice_rs::analysis::optimization::parameters::CircuitCandidate<'_>,
    expected: &Netlist,
) {
    let context = candidate.model_context();
    let actual = candidate
        .circuit_mut()
        .linear_system_with_context(&context)
        .unwrap();
    let reference = Circuit::from_netlist(expected)
        .unwrap()
        .linear_system()
        .unwrap();
    assert_eq!(actual.a.rows(), reference.a.rows());
    for r in 0..actual.a.rows() {
        for c in 0..actual.a.cols() {
            let expected = reference.a.get(r, c);
            let got = actual.a.get(r, c);
            assert!(
                (got - expected).abs() <= 1e-12 * expected.abs().max(1.),
                "{r},{c}: {got:e} != {expected:e}"
            );
        }
    }
}

#[test]
fn bound_site_function_free_names_follow_nested_shadowed_slots() {
    let root = deck(".param x=2 unused=10\n.func f(v)={v+shift}");
    let child = deck(".param shift=3 flat={x-x} alias={flat+2}");
    let site = deck("r1 a 0 {f(alias)+shift}");
    let expression = match &site.devices[0].parameters[0].kind {
        ParameterKind::Expression(expression) => expression,
        other => panic!("expected parameter expression: {other:?}"),
    };
    let functions = FunctionScope::for_netlist(&root, &EvalBudget::default()).unwrap();
    let mut plan = ParameterPlan::new();
    let top = plan
        .add_scope(
            ParameterScopeSpec {
                parent: None,
                functions: Some(Arc::clone(&functions)),
                bindings: &[],
                cards: &root.params,
                instance_overrides: false,
                tunables: &["x", "unused"],
            },
            &mut EvalBudget::default(),
        )
        .unwrap();
    let child_id = plan
        .add_scope(
            ParameterScopeSpec {
                parent: Some(top),
                functions: Some(functions),
                bindings: &[],
                cards: &child.params,
                instance_overrides: false,
                tunables: &["shift"],
            },
            &mut EvalBudget::default(),
        )
        .unwrap();
    let bound = plan
        .bind_expression(child_id, expression, &mut EvalBudget::default())
        .unwrap();
    assert_eq!(bound.inputs(), &[0, 2]);
    let mut current = plan.session();
    for (candidate, result, dirty) in [
        ([5., 10., 3.], 8., false),
        ([5., 10., 4.], 10., true),
        ([2., 10., 4.], 10., false),
        ([2., 10., 3.], 8., true),
    ] {
        let result: f64 = result;
        let previous = current.clone();
        current
            .update(&candidate, &mut EvalBudget::default())
            .unwrap();
        assert_eq!(bound.changed_between(&previous, &current), dirty);
        let direct = current
            .evaluate(child_id, expression, &mut EvalBudget::default())
            .unwrap();
        let compiled = current
            .evaluate_bound(child_id, expression, &bound, &mut EvalBudget::default())
            .unwrap();
        assert_eq!(direct.to_bits(), result.to_bits());
        assert_eq!(compiled.to_bits(), direct.to_bits());
    }
    let old = current.clone();
    assert!(
        current
            .update(&[f64::NAN, 10., 3.], &mut EvalBudget::default())
            .is_err()
    );
    assert!(!bound.changed_between(&old, &current));
}

#[test]
fn unchanged_derived_value_sites_skip_re_evaluation_but_keep_effective_overlays() {
    let source = ".param r=1k b=2k unused=5 flat={r-r} derived={flat+1k}\n\
v1 n 0 1\nr1 n 0 {derived}\nr2 n 0 {b}";
    let plan = CircuitParameterPlan::new(deck(source), &["r", "b", "unused"]).unwrap();
    let mut session = plan.session().unwrap();
    for (r, b, unused, count) in [
        (3000., 2000., 5., 0),
        (3000., 2500., 5., 1),
        (5000., 2500., 9., 0),
        (5000., 2000., 9., 1),
        (3000., 2500., 5., 1),
        (3000., 2500., 5., 0),
    ] {
        session
            .update(&[r, b, unused], &mut EvalBudget::default())
            .unwrap();
        assert_eq!(session.last_site_evaluations(), count, "{r} {b} {unused}");
        assert_eq!(session.topology(), CandidateTopology::Reused);
        equivalent_matrix(
            &mut session,
            &deck(&format!(
                ".param r={r} b={b} unused={unused} flat={{r-r}} derived={{flat+1k}}\n\
v1 n 0 1\nr1 n 0 {{derived}}\nr2 n 0 {{b}}"
            )),
        );
    }
}

#[test]
fn direct_dirty_expression_with_unchanged_result_skips_decimal_round_trip() {
    let source = deck(".param x=1000\nv1 n 0 1\nr1 n 0 {x-x+2000}");
    let plan = CircuitParameterPlan::new(source, &["x"]).unwrap();
    let mut session = plan.session().unwrap();
    assert_eq!(plan.sites().len(), 1);
    for (x, evaluated, rounded) in [(1500., 1, 0), (3000., 1, 0), (1000., 1, 0), (1000., 0, 0)] {
        session.update(&[x], &mut EvalBudget::default()).unwrap();
        assert_eq!(session.last_site_evaluations(), evaluated);
        assert_eq!(session.last_site_roundings(), rounded);
        assert_eq!(session.site_value(0), Some(2000.));
        equivalent_matrix(
            &mut session,
            &deck(&format!(".param x={x}\nv1 n 0 1\nr1 n 0 {{x-x+2000}}")),
        );
    }
}

#[test]
fn site_expression_failure_and_budget_exhaustion_leave_full_candidate_unchanged() {
    let plan =
        CircuitParameterPlan::new(deck(".param x=4\nv1 n 0 1\nr1 n 0 {1/(x-3)}"), &["x"]).unwrap();
    let mut session = plan.session().unwrap();
    session.update(&[5.], &mut EvalBudget::default()).unwrap();
    assert_eq!(session.last_site_evaluations(), 1);
    let stable = session.model_context();
    let stable_config = session.config().params().unwrap().get("x");
    assert!(session.update(&[3.], &mut EvalBudget::default()).is_err());
    assert_eq!(session.model_context(), stable);
    assert_eq!(session.config().params().unwrap().get("x"), stable_config);
    assert_eq!(session.last_site_evaluations(), 1);
    assert!(
        session
            .update(
                &[6.],
                &mut EvalBudget::new(EvalLimits {
                    nodes: 0,
                    ..EvalLimits::default()
                })
            )
            .is_err()
    );
    assert_eq!(session.model_context(), stable);
    session.update(&[4.], &mut EvalBudget::default()).unwrap();
    assert_eq!(session.last_site_evaluations(), 1);
    equivalent_matrix(
        &mut session,
        &deck(".param x=4\nv1 n 0 1\nr1 n 0 {1/(x-3)}"),
    );
}

#[test]
fn compiled_sites_remain_isolated_across_parallel_aba_sessions() {
    let plan = Arc::new(
        CircuitParameterPlan::new(
            deck(".param r=1k drive=1\nv1 n 0 {drive}\nr1 n 0 {r}"),
            &["r", "drive"],
        )
        .unwrap(),
    );
    std::thread::scope(|threads| {
        for offset in [0., 100., 200., 300.] {
            let plan = Arc::clone(&plan);
            threads.spawn(move || {
                let mut session = plan.session().unwrap();
                for (r, drive, count) in [
                    (1000. + offset, 2., if offset == 0. { 1 } else { 2 }),
                    (1000. + offset, 3., 1),
                    (1000., 1., if offset == 0. { 1 } else { 2 }),
                ] {
                    session
                        .update(&[r, drive], &mut EvalBudget::default())
                        .unwrap();
                    assert_eq!(session.last_site_evaluations(), count);
                    assert_eq!(session.topology(), CandidateTopology::Reused);
                    equivalent_matrix(
                        &mut session,
                        &deck(&format!(
                            ".param r={r} drive={drive}\nv1 n 0 {{drive}}\nr1 n 0 {{r}}"
                        )),
                    );
                }
            });
        }
    });
}

#[test]
fn typed_site_table_retains_device_model_analysis_and_hint_provenance() {
    use ngspice_rs::netlist::elaborate::SiteKind;

    let source = deck(
        ".param r=1k isat=1e-14 dt=1u ic=2\n\
v1 in 0 1\nr1 in 0 {r}\nd1 in 0 dm\n.model dm d(is={isat})\n\
.tran {dt} 10u\n.ic v(in)={ic}\n.nodeset v(in)={ic+1}",
    );
    let plan = CircuitParameterPlan::new(source, &["r", "isat", "dt", "ic"]).unwrap();
    let kinds: Vec<_> = plan.sites().iter().map(|site| site.kind()).collect();
    for expected in [
        SiteKind::DeviceParameter {
            device: 1,
            parameter: 0,
        },
        SiteKind::ModelParameter {
            model: 0,
            parameter: 0,
        },
        SiteKind::AnalysisArgument {
            analysis: 0,
            argument: 0,
        },
        SiteKind::InitialCondition { card: 0, entry: 0 },
        SiteKind::Nodeset { card: 0, entry: 0 },
    ] {
        assert!(
            kinds.contains(&expected),
            "missing {expected:?} in {kinds:?}"
        );
    }
    let mut session = plan.session().unwrap();
    for (index, site) in plan.sites().iter().enumerate() {
        assert!(site.location().to_string().contains("compiled-sites.cir"));
        assert!(!site.inputs().is_empty());
        assert_eq!(
            session.site_value(index).unwrap().to_bits(),
            site.baseline().to_bits()
        );
    }
    session
        .update(&[1000., 1e-14, 2e-6, 2.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.last_site_evaluations(), 1);
    assert_eq!(session.topology(), CandidateTopology::Rebuilt);
}

/// Manual latency and dependency-work comparison across ten thousand candidates.
/// Only one of 128 dependency-affected sites changes per transition because
/// the others depend on a derived parameter with an exactly cancelling value.
/// Full scalar re-elaboration is sampled separately using the same deck.
#[test]
#[ignore = "manual 10k-candidate optimization preparation benchmark"]
fn benchmark_compiled_site_dirty_count_and_scalar_rebuilds() {
    use std::fmt::Write as _;
    use std::hint::black_box;
    use std::time::Instant;

    const SITES: usize = 128;
    const CANDIDATES: usize = 10_000;
    const SCALAR_SAMPLES: usize = 500;

    let mut body = String::from(".param x=1k flat={x-x}\nv1 n 0 1\nr1 n 0 {x}\n");
    for site in 2..=SITES {
        writeln!(body, "r{site} n 0 {{flat+{}}}", 1000 + site).unwrap();
    }
    let initial = Instant::now();
    let plan = CircuitParameterPlan::new(deck(&body), &["x"]).unwrap();
    let compile = initial.elapsed();
    assert_eq!(plan.sites().len(), SITES);
    let mut session = plan.session().unwrap();

    let values = |i: usize| 1000. + f64::from(u32::try_from(i % 101).unwrap()) * 0.125;
    let start = Instant::now();
    let mut site_evaluations = 0_usize;
    let mut site_roundings = 0_usize;
    let mut reused = 0_usize;
    let mut rebuilt = 0_usize;
    for candidate in 0..CANDIDATES {
        session
            .update(&[values(candidate)], &mut EvalBudget::default())
            .unwrap();
        site_evaluations += session.last_site_evaluations();
        site_roundings += session.last_site_roundings();
        match black_box(session.topology()) {
            CandidateTopology::Reused => reused += 1,
            CandidateTopology::Rebuilt => rebuilt += 1,
        }
    }
    let elapsed = start.elapsed();
    assert_eq!(reused, CANDIDATES);
    assert_eq!(rebuilt, 0);
    assert_eq!(site_evaluations, CANDIDATES - 1);
    assert_eq!(site_roundings, CANDIDATES - 1);

    let start = Instant::now();
    for candidate in 0..SCALAR_SAMPLES {
        let scalar = body.replacen(
            ".param x=1k",
            &format!(".param x={:.17e}", values(candidate)),
            1,
        );
        black_box(Circuit::from_netlist(&deck(&scalar)).unwrap());
    }
    let scalar_elapsed = start.elapsed();

    eprintln!(
        "compiled_site_bench sites={SITES} candidate_count={CANDIDATES} prepare_compile_ms={:.3} prepared_ms={:.3} dirty_sites={site_evaluations} numparam_roundings={site_roundings} reused={reused} rebuilt={rebuilt} scalar_samples={SCALAR_SAMPLES} scalar_ms={:.3}",
        compile.as_secs_f64() * 1000.,
        elapsed.as_secs_f64() * 1000.,
        scalar_elapsed.as_secs_f64() * 1000.,
    );
}
