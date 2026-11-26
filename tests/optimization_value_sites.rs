//! #140: bound optimization value sites and transactional dirty evaluation.
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ngspice_rs::analysis::optimization::parameters::{CandidateTopology, CircuitParameterPlan};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::elaborate::SiteKind;
use ngspice_rs::netlist::eval::{
    EvalBudget, EvalLimits, FunctionScope, ParameterPlan, ParameterScopeSpec,
};
use ngspice_rs::netlist::source::parse_deck_text;

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("optimization-value-sites.cir"),
            &format!("Bound values\n{body}\n.end\n"),
        ))
        .unwrap()
}

#[test]
fn typed_sites_keep_original_provenance_and_independent_inputs() {
    let source = deck(
        ".param x=2 y=3\nv1 in 0 {x}\nr1 in 0 {x+y}\n.tran {x*1e-9} {y*1e-8}\n.ic v(in)={x}\n.nodeset v(in)={y}\n.model dm d(is={y})\nd1 in 0 dm",
    );
    let plan = CircuitParameterPlan::new(source, &["x", "y"]).unwrap();
    let sites = plan.sites();
    assert!(
        sites
            .iter()
            .any(|site| matches!(site.kind, SiteKind::DeviceParameter { .. }))
    );
    assert!(
        sites
            .iter()
            .any(|site| matches!(site.kind, SiteKind::AnalysisArgument { .. }))
    );
    assert!(
        sites
            .iter()
            .any(|site| matches!(site.kind, SiteKind::InitialCondition { .. }))
    );
    assert!(
        sites
            .iter()
            .any(|site| matches!(site.kind, SiteKind::Nodeset { .. }))
    );
    assert!(
        sites
            .iter()
            .any(|site| matches!(site.kind, SiteKind::ModelParameter { .. }))
    );
    assert!(sites.iter().all(|site| !site.inputs().is_empty()));
    assert!(
        sites
            .iter()
            .all(|site| !site.location.to_string().is_empty())
    );
    let mut session = plan.session().unwrap();
    for (index, site) in sites.iter().enumerate() {
        assert_eq!(session.site_value(index), Some(site.baseline));
    }
    assert_eq!(session.site_value(sites.len()), None);
    session
        .update(&[5., 3.], &mut EvalBudget::default())
        .unwrap();
    assert!(session.last_site_evaluations() > 0);
    assert_eq!(session.site_value(sites.len()), None);
}

#[test]
fn exact_dependency_dirty_and_raw_equal_skip_rounding() {
    let source = deck(
        ".param x=2 y=3 flat={x-x}\nv1 in 0 2\nr1 in 0 {1000+flat}\nr2 in 0 {2000+(x-x)}\nr3 in 0 {3000+y}",
    );
    let plan = CircuitParameterPlan::new(source, &["x", "y"]).unwrap();
    let mut session = plan.session().unwrap();
    assert_eq!(plan.sites().len(), 3);
    session
        .update(&[8., 3.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.last_site_evaluations(), 1);
    assert_eq!(session.last_site_roundings(), 0);
    assert_eq!(session.topology(), CandidateTopology::Reused);
    for (index, expected) in [1000., 2000., 3003.].into_iter().enumerate() {
        assert_eq!(session.site_value(index), Some(expected));
    }
    session
        .update(&[8., 4.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.last_site_evaluations(), 1);
    assert_eq!(session.last_site_roundings(), 1);
    session
        .update(&[8., 4.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.last_site_evaluations(), 0);
    assert_eq!(session.last_site_roundings(), 0);
    session
        .update(&[2., 3.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.last_site_roundings(), 1);
    assert_eq!(session.site_value(2), Some(3003.));
}

#[test]
fn bound_function_names_shadowing_and_scalar_evaluator_agree() {
    let root = deck(".param x=2 y=3 sum={x+y}\n.func f(z)={2*z+sum}");
    let child = deck(".param y=7 sum={x+y}");
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
                tunables: &["x", "y"],
            },
            &mut EvalBudget::default(),
        )
        .unwrap();
    let sub = plan
        .add_scope(
            ParameterScopeSpec {
                parent: Some(top),
                functions: Some(functions),
                bindings: &[],
                cards: &child.params,
                instance_overrides: false,
                tunables: &[],
            },
            &mut EvalBudget::default(),
        )
        .unwrap();
    let probe = deck("r1 in 0 {f(x)+sum}");
    let expr = match &probe.devices[0].parameters[0].kind {
        ngspice_rs::netlist::ast::ParameterKind::Expression(expr) => expr,
        other => panic!("expected expression: {other:?}"),
    };
    let bound = plan
        .bind_expression(sub, expr, &mut EvalBudget::default())
        .unwrap();
    assert_eq!(bound.inputs(), &[0]); // the nested scope shadows the outer y
    let mut state = plan.session();
    for input in [[4., 5.], [8., 2.], [4., 5.]] {
        state.update(&input, &mut EvalBudget::default()).unwrap();
        let direct = state
            .evaluate(sub, expr, &mut EvalBudget::default())
            .unwrap();
        let indexed = state
            .evaluate_bound(expr, &bound, &mut EvalBudget::default())
            .unwrap();
        assert_eq!(indexed.to_bits(), direct.to_bits());
    }
}

#[test]
fn rollback_preserves_cached_values_after_invalid_and_budget_failure() {
    let source = deck(".param x=2 reciprocal={1/x}\nv1 in 0 {x}\nr1 in 0 {1000+reciprocal}");
    let plan = CircuitParameterPlan::new(source, &["x"]).unwrap();
    let mut session = plan.session().unwrap();
    session.update(&[4.], &mut EvalBudget::default()).unwrap();
    let saved = plan
        .sites()
        .iter()
        .enumerate()
        .map(|(i, _)| session.site_value(i).unwrap())
        .collect::<Vec<_>>();
    let previous = session.last_site_evaluations();
    assert!(session.update(&[0.], &mut EvalBudget::default()).is_err());
    assert_eq!(session.last_site_evaluations(), previous);
    for (i, value) in saved.iter().enumerate() {
        assert_eq!(session.site_value(i).unwrap().to_bits(), value.to_bits());
    }
    let mut tight = EvalBudget::new(EvalLimits {
        nodes: 1,
        ..EvalLimits::default()
    });
    assert!(session.update(&[5.], &mut tight).is_err());
    session.update(&[4.], &mut EvalBudget::default()).unwrap();
    assert_eq!(session.last_site_evaluations(), 0);
    session.update(&[2.], &mut EvalBudget::default()).unwrap();
    assert_eq!(session.topology(), CandidateTopology::Reused);
}

#[test]
fn independent_sessions_keep_separate_site_caches() {
    let plan = Arc::new(
        CircuitParameterPlan::new(
            deck(".param x=2 y=3\nv1 in 0 {x}\nr1 in 0 {1000+y}"),
            &["x", "y"],
        )
        .unwrap(),
    );
    std::thread::scope(|threads| {
        for (x, y) in [(4., 5.), (8., 3.), (2., 7.), (12., 6.)] {
            let plan = Arc::clone(&plan);
            threads.spawn(move || {
                let mut session = plan.session().unwrap();
                session.update(&[x, y], &mut EvalBudget::default()).unwrap();
                assert_eq!(session.topology(), CandidateTopology::Reused);
                let values = plan
                    .sites()
                    .iter()
                    .enumerate()
                    .map(|(index, _)| session.site_value(index).unwrap())
                    .collect::<Vec<_>>();
                assert!(values.contains(&x));
                assert!(values.contains(&(1000. + y)));
            });
        }
    });
}

fn workload(sites: usize) -> Netlist {
    let mut input = String::from(".param x=1 y=2\nv1 in 0 1\n");
    for i in 0..sites {
        let variable = if i == 0 { "x" } else { "y" };
        input.push_str(&format!("r{} in 0 {{{}+{}}}\n", i + 1, 1000 + i, variable));
    }
    deck(&input)
}

#[test]
#[ignore = "run manually; use Valgrind DHAT externally for allocation counts"]
fn benchmark_ten_thousand_candidates_and_one_hundred_twenty_eight_sites() {
    let plan = CircuitParameterPlan::new(workload(128), &["x", "y"]).unwrap();
    let mut session = plan.session().unwrap();
    assert_eq!(plan.sites().len(), 128);
    let start = Instant::now();
    let mut evaluated = 0;
    let mut rounded = 0;
    for i in 0..10_000 {
        session
            .update(&[i as f64 + 2., 2.], &mut EvalBudget::default())
            .unwrap();
        evaluated += session.last_site_evaluations();
        rounded += session.last_site_roundings();
    }
    let elapsed = start.elapsed();
    assert_eq!(evaluated, 10_000);
    assert_eq!(rounded, 10_000);
    eprintln!(
        "10k candidate updates: {elapsed:?}; 128 sites/candidate; evaluated {evaluated}; rounded {rounded}; naive evaluations {}",
        10_000 * 128
    );
}

#[test]
fn waveform_candidate_sites_rebuild_and_restore_immutable_forcing() {
    use ngspice_rs::devices::{Limit, TransientTiming};
    let source =
        deck(".param a=2 period=.5\nv1 in 0 pwl(0 0 {period/2} {a} {period} 0) r=0\nr1 in 0 1k");
    let plan = CircuitParameterPlan::new(source, &["a", "period"]).unwrap();
    let entries = plan
        .sites()
        .iter()
        .map(|site| match site.kind {
            SiteKind::DeviceVectorEntry {
                device: 0,
                parameter: 0,
                entry,
            } => entry,
            other => panic!("unexpected site {other:?}"),
        })
        .collect::<Vec<_>>();
    assert_eq!(entries, [2, 3, 4]);
    let mut session = plan.session().unwrap();
    for (coords, value) in [
        ([2., 0.5], 2.),
        ([4., 1.], 2.),
        ([6., 0.5], 6.),
        ([2., 0.5], 2.),
    ] {
        session.update(&coords, &mut EvalBudget::default()).unwrap();
        let mut system = session.circuit().linear_system().unwrap();
        system
            .bind_transient_timing(&TransientTiming::new(0.015625, 2.).unwrap())
            .unwrap();
        let actual = system.sources[0]
            .waveform
            .value_at(0.25, Limit::Right)
            .unwrap();
        assert_eq!(actual, value);
    }
    assert_eq!(session.topology(), CandidateTopology::Reused);
    assert!(
        session
            .update(&[2., 0.], &mut EvalBudget::default())
            .is_err()
    );
    let mut system = session.circuit().linear_system().unwrap();
    system
        .bind_transient_timing(&TransientTiming::new(0.015625, 2.).unwrap())
        .unwrap();
    assert_eq!(
        system.sources[0]
            .waveform
            .value_at(0.25, Limit::Right)
            .unwrap(),
        2.
    );
}
