//! Static statistical realization is scoped, deterministic, and independent
//! of runtime Newton and candidate evaluation order.
use std::path::Path;

use ngspice_rs::analysis::config::RunConfig;
use ngspice_rs::analysis::optimization::parameters::CircuitParameterPlan;
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{
    Parser,
    ast::Netlist,
    eval::{EvalBudget, ParamScope},
    source::parse_deck_text,
};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("statistical.cir"),
            &format!("statistics\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn op(circuit: &mut Circuit) -> SpiceResult<Plot> {
    AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
        .resolve()?
        .run(circuit, &AnalysisContext::default())
}

#[test]
fn same_seed_same_draw_different_seed_different_draw() -> SpiceResult<()> {
    let source = |seed| {
        deck(&format!(
            ".options seed={seed}\n.param a=agauss(10,2,3) b=gauss(10,.2,3) c=aunif(10,2) d=unif(10,.2) e1=limit(10,2)"
        ))
    };
    let first = ParamScope::for_netlist(&source(41))?;
    let second = ParamScope::for_netlist(&source(41))?;
    let other = ParamScope::for_netlist(&source(42))?;
    for name in ["a", "b", "c", "d", "e1"] {
        assert_eq!(first.get(name), second.get(name));
    }
    for name in ["a", "b", "c", "d"] {
        assert_ne!(first.get(name), other.get(name));
    }
    for name in ["c", "d"] {
        assert!((8. ..=12.).contains(&first.get(name).unwrap()));
    }
    assert!([8., 12.].contains(&first.get("e1").unwrap()));
    Ok(())
}

#[test]
fn statistical_calls_in_macros_have_independent_occurrences_and_static_arguments() -> SpiceResult<()>
{
    let source = deck(
        ".options seed=13\n.func sample(a) {aunif(a,0.1)}\n.param x={sample(1)-sample(1)}\n.param a=agauss(2,0,3) b=gauss(4,0.1,0)\nb1 out 0 v=sample(1+2)\nb2 other 0 v=sample(1+2)",
    );
    let scope = ParamScope::for_netlist(&source)?;
    assert_ne!(scope.get("x"), Some(0.));
    assert_eq!(scope.get("a"), Some(2.));
    assert_eq!(scope.get("b"), Some(4.));
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let plot = op(&mut circuit)?;
    let out = plot.value("v(out)", 0).unwrap().re;
    let other = plot.value("v(other)", 0).unwrap().re;
    assert!((2.9..=3.1).contains(&out) && (2.9..=3.1).contains(&other));
    assert_ne!(out, other);
    Ok(())
}

#[test]
fn subcircuit_instances_are_independent_but_shared_parameters_remain_correlated() -> SpiceResult<()>
{
    let source = deck(
        ".options seed=72\n.param common=aunif(1,0.1)\n.subckt cell a b c\n.param local=aunif(1,0.1)\nb1 a 0 v=local\nb2 b 0 v=local\nb3 c 0 v=common\n.ends\nx1 a1 b1 c1 cell\nx2 a2 b2 c2 cell",
    );
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let plot = op(&mut circuit)?;
    let get = |name| plot.value(name, 0).unwrap().re;
    assert_eq!(get("v(a1)"), get("v(b1)"));
    assert_eq!(get("v(a2)"), get("v(b2)"));
    assert_ne!(get("v(a1)"), get("v(a2)"));
    assert_eq!(get("v(c1)"), get("v(c2)"));
    let mut rebuilt = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let second = op(&mut rebuilt)?;
    for name in ["v(a1)", "v(a2)", "v(c1)"] {
        assert_eq!(get(name), second.value(name, 0).unwrap().re);
    }
    Ok(())
}

#[test]
fn candidate_aba_retains_the_normalized_realization_on_dependent_sites() -> SpiceResult<()> {
    let source = deck(
        ".options seed=17\n.param nominal=1k r={gauss(nominal,.1,3)}\nv1 in 0 1\nr1 in 0 {r}\nr2 in 0 {gauss(nominal,.1,3)}",
    );
    let plan = CircuitParameterPlan::new(source, &["nominal"])?;
    let mut session = plan.session()?;
    let run = |session: &mut ngspice_rs::analysis::optimization::parameters::CircuitCandidate<
        '_,
    >|
     -> SpiceResult<f64> {
        let context = session.analysis_context();
        Ok(
            AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
                .resolve()?
                .run(session.circuit_mut(), &context)?
                .value("i(v1)", 0)
                .unwrap()
                .re,
        )
    };
    let a = run(&mut session)?;
    session.update(&[2000.], &mut EvalBudget::default())?;
    let b = run(&mut session)?;
    assert!((2. * b - a).abs() < 1e-12, "{a} {b}");
    session.update(&[1000.], &mut EvalBudget::default())?;
    assert_eq!(a, run(&mut session)?);
    Ok(())
}

#[test]
fn statistical_values_do_not_change_with_transient_time_or_newton_loads() -> SpiceResult<()> {
    let source = deck(".options seed=81\nb1 out 0 v=agauss(2,0.2,3)+1000*time\nr1 out 0 1k");
    let mut circuit = RunConfig::from_netlist(&source)?.circuit(&source)?;
    let bias = op(&mut circuit)?.value("v(out)", 0).unwrap().re;
    let plot = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"])
        .resolve()?
        .run(&mut circuit, &AnalysisContext::default())?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        let actual = plot.value("v(out)", index).unwrap().re;
        assert!((actual - bias - 1000. * t).abs() < 1e-10);
    }
    Ok(())
}

#[test]
fn behavioural_candidates_keep_draws_across_reelaboration_and_hierarchy() -> SpiceResult<()> {
    let source = deck(
        ".options seed=55\n.param nominal=10\n.subckt cell a b params: n=nominal\n.param local=gauss(n,.1,3)\nb1 a 0 v=local\nb2 b 0 v=gauss(n,.1,3)\n.ends\nx1 a1 b1 cell\nx2 a2 b2 cell",
    );
    let plan = CircuitParameterPlan::new(source, &["nominal"])?;
    let mut candidate = plan.session()?;
    let run = |candidate: &mut ngspice_rs::analysis::optimization::parameters::CircuitCandidate<
        '_,
    >|
     -> SpiceResult<[f64; 4]> {
        let context = candidate.analysis_context();
        let plot =
            AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
                .resolve()?
                .run(candidate.circuit_mut(), &context)?;
        Ok(["v(a1)", "v(b1)", "v(a2)", "v(b2)"].map(|name| plot.value(name, 0).unwrap().re))
    };
    let first = run(&mut candidate)?;
    assert_ne!(first[0], first[2]);
    assert_ne!(first[1], first[3]);
    candidate.update(&[20.], &mut EvalBudget::default())?;
    let second = run(&mut candidate)?;
    for (a, b) in first.into_iter().zip(second) {
        assert!((b - 2. * a).abs() < 1e-12, "{a} {b}");
    }
    candidate.update(&[10.], &mut EvalBudget::default())?;
    assert_eq!(first, run(&mut candidate)?);
    Ok(())
}
