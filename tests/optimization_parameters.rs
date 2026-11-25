//! #112: topology-aware, independently reproducible reusable candidate plans.
use std::path::Path;
use std::sync::Arc;

use ngspice_rs::analysis::optimization::parameters::{CandidateTopology, CircuitParameterPlan};
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::eval::{EvalBudget, FunctionScope, ParameterPlan, ParameterScopeSpec};
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::AnalysisKind;

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("optimization-parameters.cir"),
            &format!("Optimization candidates\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn close(value: f64, reference: f64) {
    let difference = (value - reference).abs();
    let limit = 1e-12_f64.mul_add(reference.abs().max(value.abs()), 1e-14);
    assert!(difference <= limit, "{value:e} vs {reference:e}");
}

fn same_linear(
    candidate: &mut ngspice_rs::analysis::optimization::parameters::CircuitCandidate<'_>,
    independent: &Netlist,
) {
    let context = candidate.model_context();
    let actual = candidate
        .circuit_mut()
        .linear_system_with_context(&context)
        .unwrap();
    let expected = Circuit::from_netlist(independent)
        .unwrap()
        .linear_system()
        .unwrap();
    assert_eq!(actual.a.rows(), expected.a.rows());
    for row in 0..actual.a.rows() {
        for col in 0..actual.a.cols() {
            close(actual.a.get(row, col), expected.a.get(row, col));
            close(actual.e.get(row, col), expected.e.get(row, col));
        }
    }
    let actual = actual.dc_rhs(None).unwrap();
    let expected = expected.dc_rhs(None).unwrap();
    for (a, b) in actual.as_slice().iter().zip(expected.as_slice()) {
        close(*a, *b);
    }
    // The full operating-point analysis must receive the candidate model
    // overlay as well; matching only hand-assembled matrices is insufficient.
    let op = AnalysisInput::new(AnalysisKind::OperatingPoint)
        .resolve()
        .unwrap();
    let candidate_context = candidate.analysis_context();
    let actual = op.run(candidate.circuit_mut(), &candidate_context).unwrap();
    let mut independent = Circuit::from_netlist(independent).unwrap();
    let expected = op
        .run(&mut independent, &AnalysisContext::default())
        .unwrap();
    assert_eq!(actual.variables().len(), expected.variables().len());
    for variable in expected.variables() {
        close(
            actual.value(&variable.name, 0).unwrap().re,
            expected.value(&variable.name, 0).unwrap().re,
        );
    }
}

#[test]
fn compiled_numeric_setters_reuse_session_topology_and_match_scalar_aba() {
    let source = deck(
        ".param r=1k drive=2 derived={r*2} unrelated=7 reciprocal={1/(r-500)}\nv1 in 0 {drive}\nr1 in 0 {derived}",
    );
    let plan = CircuitParameterPlan::new(source, &["r", "drive", "unrelated"]).unwrap();
    assert_eq!(plan.inputs().len(), 3);
    let mut session = plan.session().unwrap();
    assert_eq!(session.topology(), CandidateTopology::Reused);
    for (r, drive, unused) in [(3000., 3., 7.), (1500., 4., 9.), (3000., 3., 7.)] {
        let computed = session
            .update(&[r, drive, unused], &mut EvalBudget::default())
            .unwrap();
        // Both the device's derived resistance and the independent diagnostic
        // reciprocal node respond to r; no other definition is evaluated.
        assert!(computed <= 2);
        assert_eq!(session.topology(), CandidateTopology::Reused);
        assert_eq!(session.config().params().unwrap().get("r"), Some(r));
        assert_eq!(session.config().params().unwrap().get("drive"), Some(drive));
        same_linear(
            &mut session,
            &deck(&format!(
                ".param r={r} drive={drive} derived={{r*2}} unrelated={unused} reciprocal={{1/(r-500)}}\nv1 in 0 {{drive}}\nr1 in 0 {{derived}}"
            )),
        );
    }
    assert_eq!(
        session
            .update(&[3000., 3., 7.], &mut EvalBudget::default())
            .unwrap(),
        0
    );
    // A scalar zero resistor is physically clamped by the ngspice factory.
    // The typed sweep setter forbids zero; the scalar fallback must accept it.
    session
        .update(&[0., 3., 7.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.topology(), CandidateTopology::Rebuilt);
    same_linear(
        &mut session,
        &deck(
            ".param r=0 drive=3 derived={r*2} unrelated=7 reciprocal={1/(r-500)}\nv1 in 0 {drive}\nr1 in 0 {derived}",
        ),
    );
    session
        .update(&[3000., 3., 7.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.topology(), CandidateTopology::Reused);
    let stable = session.model_context();
    assert!(
        session
            .update(&[500., 3., 7.], &mut EvalBudget::default())
            .is_err()
    );
    assert_eq!(session.topology(), CandidateTopology::Reused);
    assert_eq!(session.model_context(), stable);
    same_linear(
        &mut session,
        &deck(
            ".param r=3000 drive=3 derived={r*2} unrelated=7 reciprocal={1/(r-500)}\nv1 in 0 {drive}\nr1 in 0 {derived}",
        ),
    );
    assert!(
        session
            .update(&[f64::NAN, 3., 7.], &mut EvalBudget::default())
            .is_err()
    );
}

#[test]
fn arbitrary_number_of_independent_resistors_uses_validated_sparse_overlay() {
    let source = deck(
        ".param a=1k b=2k c=3k d=4k\nv1 in 0 10\nr1 in 0 {a}\nr2 in 0 {b}\nr3 in 0 {c}\nr4 in 0 {d}",
    );
    let plan = CircuitParameterPlan::new(source, &["a", "b", "c", "d"]).unwrap();
    let mut session = plan.session().unwrap();
    session
        .update(&[1500., 2500., 3500., 4500.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.topology(), CandidateTopology::Reused);
    same_linear(
        &mut session,
        &deck(
            ".param a=1500 b=2500 c=3500 d=4500\nv1 in 0 10\nr1 in 0 {a}\nr2 in 0 {b}\nr3 in 0 {c}\nr4 in 0 {d}",
        ),
    );
    session
        .update(&[1000., 2000., 3000., 4000.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.topology(), CandidateTopology::Reused);
}

#[test]
fn one_hundred_twenty_eight_typed_resistor_sites_keep_numeric_fidelity_and_aba_state() {
    const N: usize = 128;
    let netlist = |values: &[f64]| {
        let mut body = String::from("v1 in 0 dc 10\n");
        for (index, value) in values.iter().enumerate() {
            body.push_str(&format!(".param p{index}={value:.17e}\n"));
            let (pos, neg) = if index == 0 {
                ("in", "out")
            } else {
                ("out", "0")
            };
            body.push_str(&format!("r{index} {pos} {neg} {{p{index}}}\n"));
        }
        deck(&body)
    };
    let starting: Vec<_> = (0..N).map(|index| 1000. + index as f64 * 7.).collect();
    let names: Vec<_> = (0..N).map(|index| format!("p{index}")).collect();
    let inputs: Vec<_> = names.iter().map(String::as_str).collect();
    let plan = CircuitParameterPlan::new(netlist(&starting), &inputs).unwrap();
    let mut session = plan.session().unwrap();
    let a: Vec<_> = starting
        .iter()
        .enumerate()
        .map(|(index, value)| value * (1. + (index % 13) as f64 * 0.001))
        .collect();
    let b: Vec<_> = starting
        .iter()
        .enumerate()
        .map(|(index, value)| value * (1. + (index % 11) as f64 * 0.002))
        .collect();
    for values in [&a, &b, &a, &starting] {
        session.update(values, &mut EvalBudget::default()).unwrap();
        assert_eq!(session.topology(), CandidateTopology::Reused);
        same_linear(&mut session, &netlist(values));
    }
    let mut invalid = a.clone();
    invalid[0] = f64::NAN;
    assert!(
        session
            .update(&invalid, &mut EvalBudget::default())
            .is_err()
    );
    assert_eq!(session.topology(), CandidateTopology::Reused);
    same_linear(&mut session, &netlist(&starting));
}

#[test]
fn multiple_instance_setters_on_one_device_preserve_order_and_atomicity() {
    // One independent voltage source supports three independent typed
    // numeric setters (DC bias, AC magnitude and AC phase) without a model
    // selector or geometry change.
    let netlist = |bias: f64, magnitude: f64, phase: f64| {
        deck(&format!(
            ".param bias={bias} mag={magnitude} phase={phase}\n\
             v1 in 0 dc {{bias}} ac {{mag}} {{phase}}\n\
             r1 in out 1k\n\
             r2 out 0 2k"
        ))
    };
    let plan = CircuitParameterPlan::new(netlist(10., 1., 0.), &["bias", "mag", "phase"]).unwrap();
    let mut session = plan.session().unwrap();
    let ac = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "1", "1k", "1k"])
        .resolve()
        .unwrap();
    for &(bias, magnitude, phase) in &[(8., 2., 30.), (7., 1.5, 120.), (8., 2., 30.), (10., 1., 0.)]
    {
        session
            .update(&[bias, magnitude, phase], &mut EvalBudget::default())
            .unwrap();
        assert_eq!(session.topology(), CandidateTopology::Reused);
        let expected_netlist = netlist(bias, magnitude, phase);
        same_linear(&mut session, &expected_netlist);
        let actual_context = session.analysis_context();
        let actual = ac.run(session.circuit_mut(), &actual_context).unwrap();
        let mut fresh = Circuit::from_netlist(&expected_netlist).unwrap();
        let expected = ac.run(&mut fresh, &AnalysisContext::default()).unwrap();
        for variable in expected.variables() {
            let a = actual.value(&variable.name, 0).unwrap();
            let b = expected.value(&variable.name, 0).unwrap();
            close(a.re, b.re);
            close(a.im, b.im);
        }
    }
    assert!(
        session
            .update(&[f64::NAN, 3., 45.], &mut EvalBudget::default())
            .is_err()
    );
    assert_eq!(session.topology(), CandidateTopology::Reused);
    same_linear(&mut session, &netlist(10., 1., 0.));
}

#[test]
fn model_selectors_and_geometry_rebuild_and_failure_is_atomic() {
    let source = deck(
        ".param size=2 isat=1e-14 temp=27\n.option temp={temp}\nv1 in 0 2\nr1 in a 1k\nd1 a 0 dm area={size}\n.model dm d(is={isat})",
    );
    let plan = CircuitParameterPlan::new(source, &["size", "isat", "temp"]).unwrap();
    let mut session = plan.session().unwrap();
    assert_eq!(session.topology(), CandidateTopology::Reused);
    session
        .update(&[3., 2e-14, 70.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.topology(), CandidateTopology::Rebuilt);
    assert_eq!(session.config().context().temperature, 70.);
    assert_eq!(session.model_context().temperature, 70.);
    assert_eq!(session.circuit().device_count(), 3);
    assert!(session.circuit().device("d1").is_some());
    assert!(
        session
            .update(&[0., 2e-14, 70.], &mut EvalBudget::default())
            .is_err()
    );
    assert_eq!(session.topology(), CandidateTopology::Rebuilt);
    assert_eq!(session.config().context().temperature, 70.);
    assert!(session.circuit().device("d1").is_some());
    session
        .update(&[2., 1e-14, 27.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(session.topology(), CandidateTopology::Reused);
}

#[test]
fn nested_formals_and_scoped_params_follow_the_scalar_expansion() {
    let netlist = |s: f64| {
        deck(&format!(
            ".param base={s}\n.subckt cell p q params: gain=3\n.param local={{base*gain}}\nrchild p q {{local}}\n.ends\nv1 in 0 5\nx1 in 0 cell params: gain=2"
        ))
    };
    let plan = CircuitParameterPlan::new(netlist(100.), &["base"]).unwrap();
    let mut session = plan.session().unwrap();
    for value in [300., 500., 300., 100.] {
        session
            .update(&[value], &mut EvalBudget::default())
            .unwrap();
        assert_eq!(session.topology(), CandidateTopology::Rebuilt);
        same_linear(&mut session, &netlist(value));
    }
}

#[test]
fn scoped_functions_and_aliases_bind_only_true_transitive_inputs() {
    let source = deck(
        ".param a=2 b=3 unused=9 sum={a+b}\n.func gain(x)={2*x+sum}\nv1 n 0 1\nr1 n 0 {gain(a)}",
    );
    let functions = FunctionScope::for_netlist(&source, &EvalBudget::default()).unwrap();
    let mut plan = ParameterPlan::new();
    let root = plan
        .add_scope(
            ParameterScopeSpec {
                parent: None,
                functions: Some(functions),
                bindings: &[],
                cards: &source.params,
                instance_overrides: false,
                tunables: &["b", "unused", "a"],
            },
            &mut EvalBudget::default(),
        )
        .unwrap();
    let expression = match &source.devices[1].parameters[0].kind {
        ngspice_rs::netlist::ast::ParameterKind::Expression(expr) => expr,
        other => panic!("expected expression: {other:?}"),
    };
    let bound = plan
        .bind_expression(root, expression, &mut EvalBudget::default())
        .unwrap();
    assert_eq!(bound.inputs(), &[0, 2]);
    assert!(!bound.affected(&[3., 9., 2.], plan.inputs()) && !bound.inputs().contains(&1));
    let mut candidate = plan.session();
    candidate
        .update(&[5., 9., 4.], &mut EvalBudget::default())
        .unwrap();
    let snapshot = candidate.resolved_scope(root).unwrap();
    assert_eq!(snapshot.get("sum"), Some(9.));
    assert_eq!(
        snapshot
            .evaluate(expression, &mut EvalBudget::default())
            .unwrap(),
        17.
    );
}

#[test]
fn independent_concurrent_sessions_never_share_candidate_or_topology_state() {
    let source = deck(".param r=1k\nv1 a 0 3\nr1 a 0 {r}");
    let plan = Arc::new(CircuitParameterPlan::new(source, &["r"]).unwrap());
    std::thread::scope(|threads| {
        for resistance in [100., 200., 300., 400.] {
            let plan = Arc::clone(&plan);
            threads.spawn(move || {
                let mut session = plan.session().unwrap();
                session
                    .update(&[resistance], &mut EvalBudget::default())
                    .unwrap();
                assert_eq!(session.topology(), CandidateTopology::Reused);
                same_linear(
                    &mut session,
                    &deck(&format!(".param r={resistance}\nv1 a 0 3\nr1 a 0 {{r}}")),
                );
            });
        }
    });
}

/// Diagnostic only: compare complete candidate preparation with independently
/// parsed and validated scalar decks, without asserting a performance ratio.
#[test]
#[ignore = "opt-in candidate preparation benchmark"]
fn benchmark_repeated_candidate_preparation() {
    use std::fmt::Write as _;
    use std::hint::black_box;
    use std::time::Instant;

    const COUNT: usize = 128;
    for (model, source, inputs) in [
        (
            "two-setter",
            ".param a=1k b=2k\nv1 in 0 1\nr1 in out {a}\nr2 out 0 {b}",
            &["a", "b"][..],
        ),
        (
            "four-setter",
            ".param a=1k b=2k c=3k d=4k\nv1 in 0 1\nr1 in out {a}\nr2 out 0 {b}\nr3 out 0 {c}\nr4 out 0 {d}",
            &["a", "b", "c", "d"][..],
        ),
    ] {
        let candidate_value = |i: usize, index: usize| {
            let offset = f64::from(u32::try_from((i * 17 + index * 23) % 127).unwrap());
            let coordinate = f64::from(u32::try_from(index).unwrap());
            1000.0_f64.mul_add(coordinate + 1., offset * 5.)
        };
        let source = deck(source);
        let start = Instant::now();
        let plan = CircuitParameterPlan::new(source, inputs).unwrap();
        let compile_time = start.elapsed();
        let mut session = plan.session().unwrap();
        let mut reused = std::time::Duration::ZERO;
        let mut scalar = std::time::Duration::ZERO;
        let mut fast = 0;
        let mut rebuilt = 0;
        for i in 0..COUNT {
            let vector: Vec<_> = (0..inputs.len()).map(|j| candidate_value(i, j)).collect();
            let start = Instant::now();
            black_box(session.update(&vector, &mut EvalBudget::default()).unwrap());
            reused += start.elapsed();
            match session.topology() {
                CandidateTopology::Reused => fast += 1,
                CandidateTopology::Rebuilt => rebuilt += 1,
            }
            let mut source = String::new();
            for (name, &value) in inputs.iter().zip(&vector) {
                writeln!(source, ".param {name}={value:.17e}").unwrap();
            }
            source.push_str("v1 in 0 1\nr1 in out {a}\nr2 out 0 {b}\n");
            if inputs.len() == 4 {
                source.push_str("r3 out 0 {c}\nr4 out 0 {d}\n");
            }
            let start = Instant::now();
            let fresh = Circuit::from_netlist(&deck(&source)).unwrap();
            black_box(fresh);
            scalar += start.elapsed();
        }
        println!(
            "workload=candidate_parameter_prepare,model={model},compile_ns={},candidates={COUNT},fast={fast},rebuild={rebuilt},reuse_ns={},fresh_scalar_ns={},ratio={:.3}",
            compile_time.as_nanos(),
            reused.as_nanos(),
            scalar.as_nanos(),
            reused.as_secs_f64() / scalar.as_secs_f64()
        );
    }
}

#[test]
fn hidden_runtime_passive_dependencies_invalidate_only_changed_laws() {
    let plan = CircuitParameterPlan::new(
        deck(
            ".options seed=17 reltol=1e-8\n.param drive=2 a=500 b=500 gain={a+b} unused=1\n.func law(x) {gain*(1+v(in))}\nv1 in 0 {drive}\nr1 in 0 {law(unused)}",
        ),
        &["drive", "a", "b", "unused"],
    )
    .unwrap();
    let mut session = plan.session().unwrap();
    for (values, topology) in [
        ([3., 600., 400., 7.], CandidateTopology::Reused),
        ([3., 1000., 500., 7.], CandidateTopology::Rebuilt),
        ([2., 500., 500., 1.], CandidateTopology::Reused),
    ] {
        session.update(&values, &mut EvalBudget::default()).unwrap();
        assert_eq!(session.topology(), topology);
        let context = session.analysis_context();
        let op = AnalysisInput::new(AnalysisKind::OperatingPoint)
            .resolve()
            .unwrap();
        let actual = op.run(session.circuit_mut(), &context).unwrap();
        let [drive, a, b, _] = values;
        close(
            actual.value("i(v1)", 0).unwrap().re,
            -drive / ((a + b) * (1. + drive)),
        );
    }
}

#[test]
fn lexically_static_passive_records_the_actual_numeric_expression() {
    let plan = CircuitParameterPlan::new(
        deck(".param gain=1000\n.func v(x) {gain*x}\nv1 a 0 2\nr1 a 0 {v(2)}"),
        &["gain"],
    )
    .unwrap();
    let mut session = plan.session().unwrap();
    for gain in [2000., 500., 1000.] {
        session.update(&[gain], &mut EvalBudget::default()).unwrap();
        assert_eq!(session.topology(), CandidateTopology::Reused);
        let context = session.analysis_context();
        let op = AnalysisInput::new(AnalysisKind::OperatingPoint)
            .resolve()
            .unwrap();
        let plot = op.run(session.circuit_mut(), &context).unwrap();
        close(plot.value("i(v1)", 0).unwrap().re, -1. / gain);
    }
}

#[test]
fn embedded_numparam_and_lowered_source_coefficients_invalidate_compiled_laws() {
    let plan = CircuitParameterPlan::new(
        deck(".param drive=0.5 gain=2\nv1 in 0 {drive}\nb1 out 0 v=pwl(v(in),0,0,1,{gain})\ne1 polyout 0 poly(1) in 0 0 {gain}"),
        &["drive", "gain"],
    ).unwrap();
    let mut session = plan.session().unwrap();
    for (values, topology) in [
        ([0.25, 2.], CandidateTopology::Reused),
        ([0.25, 3.], CandidateTopology::Rebuilt),
        ([0.5, 2.], CandidateTopology::Reused),
    ] {
        session.update(&values, &mut EvalBudget::default()).unwrap();
        assert_eq!(session.topology(), topology);
        let context = session.analysis_context();
        let op = AnalysisInput::new(AnalysisKind::OperatingPoint)
            .resolve()
            .unwrap();
        let plot = op.run(session.circuit_mut(), &context).unwrap();
        for name in ["v(out)", "v(polyout)"] {
            close(plot.value(name, 0).unwrap().re, values[0] * values[1]);
        }
    }
}

#[test]
fn only_effective_option_dependencies_invalidate_baseline_configuration() {
    let plan = CircuitParameterPlan::new(
        deck(".param drive=1 a=10 b=17 temperature={a+b}\n.options temp={temperature} seed=17\nv1 in 0 {drive}\nr1 in 0 1k"),
        &["drive", "a", "b"],
    ).unwrap();
    let mut session = plan.session().unwrap();
    for (values, topology) in [
        ([2., 11., 16.], CandidateTopology::Reused),
        ([2., 11., 20.], CandidateTopology::Rebuilt),
        ([3., 10., 17.], CandidateTopology::Reused),
    ] {
        session.update(&values, &mut EvalBudget::default()).unwrap();
        assert_eq!(session.topology(), topology);
        assert_eq!(session.model_context().temperature, values[1] + values[2]);
        let context = session.analysis_context();
        let op = AnalysisInput::new(AnalysisKind::OperatingPoint)
            .resolve()
            .unwrap();
        let plot = op.run(session.circuit_mut(), &context).unwrap();
        close(plot.value("v(in)", 0).unwrap().re, values[0]);
    }
}

/// Diagnostic: repeated prepared candidates with a fixed, lexically defined
/// nonlinear law, versus fresh parsed and elaborated decks.
#[test]
#[ignore = "opt-in nonlinear candidate preparation benchmark"]
fn benchmark_behavioural_candidate_preparation() {
    use std::fmt::Write as _;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    const COUNT: u32 = 256;
    let mut body = String::from(
        ".options seed=17 reltol=1e-8\n.func law(x) {gain*(1+v(in))}\nv1 in 0 {drive}\n",
    );
    for index in 0..64 {
        writeln!(body, "r{index} in 0 {{law({index})}}").unwrap();
    }
    let source = |drive: f64| deck(&format!(".param drive={drive:.17e} gain=1000\n{body}"));
    let start = Instant::now();
    let plan = CircuitParameterPlan::new(source(1.), &["drive"]).unwrap();
    let compile = start.elapsed();
    let mut session = plan.session().unwrap();
    let mut prepared = Duration::ZERO;
    let mut scalar = Duration::ZERO;
    for index in 0..COUNT {
        let drive = f64::from(index).mul_add(0.005, 1.);
        let start = Instant::now();
        black_box(
            session
                .update(&[drive], &mut EvalBudget::default())
                .unwrap(),
        );
        prepared += start.elapsed();
        assert_eq!(session.topology(), CandidateTopology::Reused);
        let start = Instant::now();
        let fresh = source(drive);
        let config = ngspice_rs::analysis::RunConfig::from_netlist(&fresh).unwrap();
        black_box(config.circuit(&fresh).unwrap());
        scalar += start.elapsed();
    }
    println!(
        "workload=behavioural_candidate_prepare,devices=65,candidates={COUNT},compile_ns={},prepared_ns={},fresh_ns={},ratio={:.5}",
        compile.as_nanos(),
        prepared.as_nanos(),
        scalar.as_nanos(),
        prepared.as_secs_f64() / scalar.as_secs_f64()
    );
}
