//! #117: reusable parameter dependency plans for independent candidate vectors.
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::eval::{
    EvalBudget, EvalLimits, FunctionScope, ParamBinding, ParamScope, ParameterPlan,
    ParameterScopeSpec,
};
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::SourceLoc;
use std::path::{Path, PathBuf};
use std::sync::Arc;

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("candidate.cir"),
            &format!("t\n{body}\n.end\n"),
        ))
        .unwrap()
}

#[test]
fn nested_candidate_vectors_match_independent_scalar_resolution() {
    let root = deck(".param a=2 b=3 alias={a+b} unrelated=7\n.func twice(x)={x*2+alias}");
    let child = deck(".param a={a+1} c={twice(a)} shadow=9 fixed={shadow*2}");
    let grandchild = deck(".param d={c+b} alias=100");
    let mut budget = EvalBudget::default();
    let functions = FunctionScope::for_netlist(&root, &budget).unwrap();
    let mut plan = ParameterPlan::new();
    let top = plan
        .add_scope(
            ParameterScopeSpec {
                parent: None,
                functions: Some(Arc::clone(&functions)),
                bindings: &[],
                cards: &root.params,
                instance_overrides: false,
                tunables: &["B", "a"],
            },
            &mut budget,
        )
        .unwrap();
    let sub = plan
        .add_scope(
            ParameterScopeSpec {
                parent: Some(top),
                functions: Some(Arc::clone(&functions)),
                bindings: &[],
                cards: &child.params,
                instance_overrides: false,
                tunables: &[],
            },
            &mut budget,
        )
        .unwrap();
    let nested = plan
        .add_scope(
            ParameterScopeSpec {
                parent: Some(sub),
                functions: Some(Arc::clone(&functions)),
                bindings: &[],
                cards: &grandchild.params,
                instance_overrides: false,
                tunables: &["alias"],
            },
            &mut budget,
        )
        .unwrap();
    assert_eq!(plan.inputs()[0].name, "b");
    assert_eq!(plan.inputs()[0].slot.scope(), top);
    assert_eq!(plan.inputs()[2].slot.scope(), nested);
    let mut session = plan.session();
    let independent = plan.session();
    for (b, a, alias) in [(5., 2., 120.), (8., 4., 90.), (5., 2., 120.)] {
        session
            .update(&[b, a, alias], &mut EvalBudget::default())
            .unwrap();
        let scalar_root = Arc::new(
            ParamScope::for_netlist(&deck(&format!(
                ".param a={a} b={b} alias={{a+b}} unrelated=7\n.func twice(x)={{x*2+alias}}"
            )))
            .unwrap(),
        );
        let scalar_child = Arc::new(
            ParamScope::resolve(
                Some(Arc::clone(&scalar_root)),
                &[],
                &child.params,
                &mut EvalBudget::default(),
            )
            .unwrap(),
        );
        let scalar_grandchild = ParamScope::resolve(
            Some(Arc::clone(&scalar_child)),
            &[],
            &deck(&format!(".param d={{c+b}} alias={alias}")).params,
            &mut EvalBudget::default(),
        )
        .unwrap();
        for (scope, scalar, names) in [
            (
                top,
                scalar_root.as_ref(),
                &["a", "b", "alias", "unrelated"][..],
            ),
            (
                sub,
                scalar_child.as_ref(),
                &["a", "c", "fixed", "alias"][..],
            ),
            (nested, &scalar_grandchild, &["d", "alias", "c"][..]),
        ] {
            for name in names {
                assert_eq!(session.get(scope, name), scalar.get(name), "{name}");
            }
        }
    }
    assert_eq!(independent.get(top, "alias"), Some(5.));
    assert_eq!(independent.get(nested, "d"), Some(14.));
    assert_eq!(session.get(nested, "d"), Some(18.));
    let inherited = plan.slot(nested, "B").unwrap();
    assert_eq!(inherited, plan.inputs()[0].slot);
    assert_eq!(session.get_slot(inherited), Some(5.));
}

#[test]
fn exact_dirty_propagation_and_tunable_cut_points() {
    let netlist =
        deck(".param x=2 y={x*3} z={y+1} flat={x-x} downstream={flat+1} independent={9*2}");
    let mut plan = ParameterPlan::new();
    let root = plan
        .add_scope(
            ParameterScopeSpec {
                parent: None,
                functions: None,
                bindings: &[],
                cards: &netlist.params,
                instance_overrides: false,
                tunables: &["x", "y"],
            },
            &mut EvalBudget::default(),
        )
        .unwrap();
    let mut session = plan.session();
    let mut budget = EvalBudget::default();
    // y is independently supplied; x cannot invalidate it or z. flat is
    // recomputed but unchanged, so its own downstream is not evaluated.
    assert_eq!(session.update(&[3., 6.], &mut budget).unwrap(), 1);
    assert_eq!(budget.nodes_used(), 3);
    assert_eq!(session.get(root, "z"), Some(7.));
    assert_eq!(session.update(&[3., 8.], &mut budget).unwrap(), 1);
    assert_eq!(session.get(root, "z"), Some(9.));
    assert_eq!(session.update(&[3., 8.], &mut budget).unwrap(), 0);
    assert_eq!(session.get(root, "downstream"), Some(1.));
    let expression = &deck(".param site={x+y+z}").params[0].assignments[0].expression;
    assert_eq!(
        session.evaluate(root, expression, &mut budget).unwrap(),
        20.
    );
}

#[test]
fn failed_updates_are_atomic_and_preserve_sign_of_zero() {
    let netlist = deck(".param x=2 y={x+1} z={1/x} signed=0 copy={signed}");
    let mut plan = ParameterPlan::new();
    let root = plan
        .add_scope(
            ParameterScopeSpec {
                parent: None,
                functions: None,
                bindings: &[],
                cards: &netlist.params,
                instance_overrides: false,
                tunables: &["x", "signed"],
            },
            &mut EvalBudget::default(),
        )
        .unwrap();
    let mut session = plan.session();
    let mut budget = EvalBudget::default();
    assert!(
        session
            .update(&[0., -0.], &mut budget)
            .unwrap_err()
            .to_string()
            .contains("division by zero")
    );
    assert_eq!(session.get(root, "x"), Some(2.));
    assert_eq!(session.get(root, "y"), Some(3.));
    assert_eq!(
        session.get(root, "copy").unwrap().to_bits(),
        0_f64.to_bits()
    );
    assert!(session.update(&[3., f64::NAN], &mut budget).is_err());
    assert!(session.update(&[3.], &mut budget).is_err());
    let mut tight = EvalBudget::new(EvalLimits {
        nodes: 3,
        ..EvalLimits::default()
    });
    assert!(session.update(&[3., -0.], &mut tight).is_err());
    assert_eq!(session.get(root, "y"), Some(3.));
    assert_eq!(session.update(&[3., -0.], &mut budget).unwrap(), 3);
    assert_eq!(session.get(root, "y"), Some(4.));
    assert_eq!(
        session.get(root, "copy").unwrap().to_bits(),
        (-0_f64).to_bits()
    );
}

#[test]
fn instance_precedence_and_invalid_plan_compilation() {
    let netlist = deck(".param x={undefined} y={x*2}");
    let bindings = [ParamBinding {
        name: "x".into(),
        value: 3.,
        location: SourceLoc::new(PathBuf::from("instance.cir"), 1, 1),
    }];
    let mut plan = ParameterPlan::new();
    let spec = |tunables| ParameterScopeSpec {
        parent: None,
        functions: None,
        bindings: &bindings,
        cards: &netlist.params,
        instance_overrides: true,
        tunables,
    };
    let mut budget = EvalBudget::default();
    assert!(plan.add_scope(spec(&["x", "X"]), &mut budget).is_err());
    assert!(plan.add_scope(spec(&["missing"]), &mut budget).is_err());
    let root = plan.add_scope(spec(&["x"]), &mut budget).unwrap();
    assert_eq!(plan.inputs().len(), 1);
    let mut session = plan.session();
    assert_eq!(session.update(&[7.], &mut budget).unwrap(), 1);
    assert_eq!(session.get(root, "y"), Some(14.));
    let cycle = deck(".param a={b} b={a}");
    let error = plan
        .add_scope(
            ParameterScopeSpec {
                parent: None,
                functions: None,
                bindings: &[],
                cards: &cycle.params,
                instance_overrides: false,
                tunables: &["a"],
            },
            &mut budget,
        )
        .unwrap_err();
    assert!(error.to_string().contains("circular parameter definition"));
    assert_eq!(plan.inputs().len(), 1);
}

#[test]
fn sibling_scopes_and_concurrent_sessions_are_independent() {
    let root = deck(".param x={undefined} x=2 k=4 total={x+k}");
    let child = deck(".param k={k+1} result={total+k}");
    let mut plan = ParameterPlan::new();
    let mut budget = EvalBudget::default();
    let top = plan
        .add_scope(
            ParameterScopeSpec {
                parent: None,
                functions: None,
                bindings: &[],
                cards: &root.params,
                instance_overrides: false,
                tunables: &["x"],
            },
            &mut budget,
        )
        .unwrap();
    let child_spec = |tunables| ParameterScopeSpec {
        parent: Some(top),
        functions: None,
        bindings: &[],
        cards: &child.params,
        instance_overrides: false,
        tunables,
    };
    assert!(plan.add_scope(child_spec(&["x"]), &mut budget).is_err());
    let left = plan.add_scope(child_spec(&["k"]), &mut budget).unwrap();
    let right = plan.add_scope(child_spec(&[]), &mut budget).unwrap();
    let left_result = plan.slot(left, "result").unwrap();
    let right_result = plan.slot(right, "result").unwrap();
    assert_ne!(left_result, right_result);
    std::thread::scope(|threads| {
        let plan = &plan;
        let first = threads.spawn(move || {
            let mut session = plan.session();
            session
                .update(&[10., 20.], &mut EvalBudget::default())
                .unwrap();
            (
                session.get_slot(left_result),
                session.get_slot(right_result),
            )
        });
        let second = threads.spawn(move || {
            let mut session = plan.session();
            session
                .update(&[30., 50.], &mut EvalBudget::default())
                .unwrap();
            (
                session.get_slot(left_result),
                session.get_slot(right_result),
            )
        });
        assert_eq!(first.join().unwrap(), (Some(34.), Some(19.)));
        assert_eq!(second.join().unwrap(), (Some(84.), Some(39.)));
    });
    let baseline = plan.session();
    assert_eq!(baseline.get_slot(left_result), Some(11.));
    assert_eq!(baseline.get_slot(right_result), Some(11.));
}
