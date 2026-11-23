//! Adjoint parameter sensitivities integrated with compiled candidate bindings.
use std::path::Path;

use ngspice_rs::analysis::AnalysisInput;
use ngspice_rs::analysis::optimization::derivatives::{
    DerivativeRequest, DerivativeStatus, DirectOutput, evaluate,
};
use ngspice_rs::analysis::optimization::parameters::{CandidateTopology, CircuitParameterPlan};
use ngspice_rs::netlist::eval::{EvalBudget, EvalLimits};
use ngspice_rs::netlist::{Parser, ast::Netlist, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, Complex};

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("optimization-derivatives.cir"),
            &format!("Physical gradients\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn output() -> DirectOutput {
    DirectOutput::Voltage {
        pos: "out".into(),
        neg: None,
    }
}

fn requests(analysis: AnalysisKind, frequency: Option<f64>) -> DerivativeRequest {
    DerivativeRequest {
        output: output(),
        analysis,
        frequency,
    }
}

fn full_candidate_voltage(
    plan: &CircuitParameterPlan,
    candidate: &[f64],
    analysis: AnalysisKind,
    frequency: Option<f64>,
) -> Complex {
    let mut session = plan.session().unwrap();
    session
        .update(candidate, &mut EvalBudget::default())
        .unwrap();
    let input = if analysis == AnalysisKind::Ac {
        AnalysisInput::with_arguments(analysis, ["lin", "1", "10000", "10000"])
    } else {
        AnalysisInput::new(analysis)
    };
    let request = session.config().plan(input).unwrap();
    let context = session.analysis_context();
    let plot = request.run(session.circuit_mut(), &context).unwrap();
    let value = plot.value("v(out)", 0).unwrap();
    assert_eq!(plot.point_count(), 1);
    let _ = frequency;
    value
}

fn physical_gradient_fixture() -> CircuitParameterPlan {
    CircuitParameterPlan::new(
        deck(
            ".param r=1k factor=2 unused=4
v1 in 0 dc 10 ac 1
r1 in out {r*factor}
r2 out 0 1k
c1 out 0 1n",
        ),
        &["r", "factor", "unused"],
    )
    .unwrap()
}

fn compare_gradient_to_resimulation(analysis: AnalysisKind, frequency: Option<f64>) {
    let plan = physical_gradient_fixture();
    let base = [1500., 2., 4.];
    let mut candidate = plan.session().unwrap();
    candidate.update(&base, &mut EvalBudget::default()).unwrap();
    assert_eq!(candidate.topology(), CandidateTopology::Reused);
    let derivatives = evaluate(
        &mut candidate,
        &requests(analysis, frequency),
        EvalLimits::default(),
    )
    .unwrap();
    assert_eq!(derivatives.len(), base.len());
    let expected_names = ["r", "factor", "unused"];
    for (input, name) in derivatives.iter().zip(expected_names) {
        assert_eq!(input.input, name);
    }
    for index in 0..2 {
        let h = base[index] * 0.002;
        let mut plus = base;
        let mut minus = base;
        plus[index] += h;
        minus[index] -= h;
        let upper = full_candidate_voltage(&plan, &plus, analysis, frequency);
        let lower = full_candidate_voltage(&plan, &minus, analysis, frequency);
        let reference = Complex::new(
            (upper.re - lower.re) / (2. * h),
            (upper.im - lower.im) / (2. * h),
        );
        let DerivativeStatus::Defined(value) = derivatives[index].status else {
            panic!("gradient {index} invalid: {:?}", derivatives[index].status);
        };
        let tolerance = 4e-4 * reference.re.abs().max(reference.im.abs()).max(1e-10);
        assert!(
            (value.re - reference.re).abs() < tolerance
                && (value.im - reference.im).abs() < tolerance,
            "{analysis:?} {index}: adjoint {value:?} central {reference:?}, tol={tolerance:e}",
        );
    }
    assert_eq!(
        derivatives[2].status,
        DerivativeStatus::Defined(Complex::ZERO)
    );
}

#[test]
fn compiled_resistor_chain_uses_one_dc_sensitivity_walk() {
    compare_gradient_to_resimulation(AnalysisKind::OperatingPoint, None);
}

#[test]
fn compiled_resistor_chain_uses_one_small_signal_adjoint() {
    compare_gradient_to_resimulation(AnalysisKind::Ac, Some(10000.));
}

#[test]
fn unsupported_transient_and_invalid_ac_frequency_return_typed_status() {
    let plan = physical_gradient_fixture();
    let mut candidate = plan.session().unwrap();
    candidate
        .update(&[1500., 2., 4.], &mut EvalBudget::default())
        .unwrap();
    for request in [
        requests(AnalysisKind::Transient, None),
        requests(AnalysisKind::Ac, None),
        requests(AnalysisKind::Ac, Some(f64::NAN)),
    ] {
        let derivatives = evaluate(&mut candidate, &request, EvalLimits::default()).unwrap();
        assert_eq!(derivatives.len(), 3);
        assert!(
            derivatives
                .iter()
                .all(|value| matches!(value.status, DerivativeStatus::Unsupported(_)))
        );
    }
}

#[test]
fn cusp_and_structural_model_changes_do_not_produce_fake_gradients() {
    let plan = CircuitParameterPlan::new(
        deck(
            ".param p=0
v1 in 0 dc 1
r1 in out {1k+abs(p)}
r2 out 0 2k",
        ),
        &["p"],
    )
    .unwrap();
    let mut candidate = plan.session().unwrap();
    let result = evaluate(
        &mut candidate,
        &requests(AnalysisKind::OperatingPoint, None),
        EvalLimits::default(),
    )
    .unwrap();
    assert_eq!(result.len(), 1);
    assert!(
        matches!(result[0].status, DerivativeStatus::NonSmooth(_)),
        "{result:?}"
    );
    let rebuild = CircuitParameterPlan::new(
        deck(
            ".param r=1k
v1 in 0 dc 1
r1 in out {r}
r2 out 0 2k
.options reltol={r/1e9}",
        ),
        &["r"],
    )
    .unwrap();
    let mut candidate = rebuild.session().unwrap();
    candidate
        .update(&[1500.], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(candidate.topology(), CandidateTopology::Rebuilt);
    let result = evaluate(
        &mut candidate,
        &requests(AnalysisKind::OperatingPoint, None),
        EvalLimits::default(),
    )
    .unwrap();
    assert!(
        matches!(result[0].status, DerivativeStatus::Unsupported(_)),
        "{result:?}"
    );
}

#[test]
fn nonlinear_ac_bias_is_not_confused_with_c_reset_sensitivity() {
    let plan = CircuitParameterPlan::new(
        deck(
            ".param r=1k
v1 in 0 dc 0.7 ac 1
r1 in out {r}
d1 out 0 dm
.model dm d (is=1e-14)",
        ),
        &["r"],
    )
    .unwrap();
    let mut candidate = plan.session().unwrap();
    candidate
        .update(&[1100.], &mut EvalBudget::default())
        .unwrap();
    let result = evaluate(
        &mut candidate,
        &requests(AnalysisKind::Ac, Some(1000.)),
        EvalLimits::default(),
    )
    .unwrap();
    assert!(
        matches!(result[0].status, DerivativeStatus::Unsupported(_)),
        "{result:?}"
    );
}

#[test]
fn shared_nonlinear_model_parameter_sums_all_instance_adjoint_partials() {
    let plan = CircuitParameterPlan::new(
        deck(
            ".param isat=1e-14\nv1 in 0 dc 0.8\nr1 in out 1k\nd1 out 0 dm\nd2 out 0 dm\n.model dm d (is={isat})",
        ),
        &["isat"],
    ).unwrap();
    let mut candidate = plan.session().unwrap();
    candidate
        .update(&[1.2e-14], &mut EvalBudget::default())
        .unwrap();
    assert_eq!(candidate.topology(), CandidateTopology::Rebuilt);
    let measured = evaluate(
        &mut candidate,
        &requests(AnalysisKind::OperatingPoint, None),
        EvalLimits::default(),
    )
    .unwrap();
    let DerivativeStatus::Defined(actual) = measured[0].status else {
        panic!("model derivative missing: {measured:?}");
    };
    let upper = full_candidate_voltage(&plan, &[1.212e-14], AnalysisKind::OperatingPoint, None);
    let lower = full_candidate_voltage(&plan, &[1.188e-14], AnalysisKind::OperatingPoint, None);
    let reference = (upper.re - lower.re) / (2.4e-16);
    let relative_error = ((actual.re - reference) / reference).abs();
    assert!(
        relative_error < 0.015 && actual.im.abs() <= 1e-16,
        "model partial summed across two diodes: actual={actual:?}, reference={reference:e}, relative error={relative_error:e}",
    );
}

#[test]
fn one_root_input_can_contribute_through_multiple_device_parameters() {
    let plan = CircuitParameterPlan::new(
        deck(".param r=1k\nv1 in 0 dc 3\nr1 in out {r}\nr2 out 0 {2*r}"),
        &["r"],
    )
    .unwrap();
    let mut candidate = plan.session().unwrap();
    candidate
        .update(&[1500.], &mut EvalBudget::default())
        .unwrap();
    let derivative = evaluate(
        &mut candidate,
        &requests(AnalysisKind::OperatingPoint, None),
        EvalLimits::default(),
    )
    .unwrap();
    let DerivativeStatus::Defined(value) = derivative[0].status else {
        panic!("invalid derivative: {derivative:?}");
    };
    assert!(
        value.re.abs() < 1e-9,
        "two resistor contributions should cancel, got {value:?}"
    );
    assert!(value.im.abs() < 1e-12);
}

#[test]
fn persistent_worker_gradient_reuses_candidate_state_without_cross_candidate_leakage() {
    use ngspice_rs::analysis::optimization::sessions::{
        CandidateDerivativeFailure, CandidateJob, CandidateWorker,
    };
    let plan = physical_gradient_fixture();
    let mut worker = CandidateWorker::new(&plan, EvalLimits::default()).unwrap();
    let request = requests(AnalysisKind::OperatingPoint, None);
    let a = CandidateJob {
        id: 11,
        values: vec![1500., 2., 4.],
    };
    let b = CandidateJob {
        id: 12,
        values: vec![1700., 2., 4.],
    };
    let first = worker.evaluate_gradient(&a, &request);
    assert_eq!(first.id, 11);
    assert_eq!(first.topology, Some(CandidateTopology::Reused));
    let first = first.outcome.unwrap();
    let second = worker.evaluate_gradient(&b, &request).outcome.unwrap();
    assert_ne!(first[0].status, second[0].status);
    let again = worker.evaluate_gradient(&a, &request).outcome.unwrap();
    assert_eq!(
        first, again,
        "independent A/B/A gradient state was not restored"
    );
    let bad = CandidateJob {
        id: 13,
        values: vec![f64::NAN, 2., 4.],
    };
    let failed = worker.evaluate_gradient(&bad, &request);
    assert_eq!(failed.id, 13);
    assert!(failed.topology.is_none());
    assert!(matches!(
        failed.outcome,
        Err(CandidateDerivativeFailure::Preparation(_))
    ));
    assert_eq!(
        worker.evaluate_gradient(&a, &request).outcome.unwrap(),
        first
    );
}

#[test]
#[ignore = "opt-in one-adjoint-vs-per-coordinate-circuit-simulation workload"]
fn parameter_fit_gradient_benchmark() {
    use std::fmt::Write as _;
    use std::hint::black_box;
    use std::time::Instant;
    let mut source = String::new();
    let mut names = Vec::new();
    for i in 0..32 {
        let name = format!("p{i}");
        names.push(name.clone());
        writeln!(source, ".param {name}=1k").unwrap();
    }
    source.push_str(
        "v1 in 0 dc 1
",
    );
    for (i, name) in names.iter().enumerate() {
        writeln!(source, "r{i} in out {{{name}}}").unwrap();
    }
    source.push_str("rload out 0 1k");
    let plan = CircuitParameterPlan::new(
        deck(&source),
        &names.iter().map(String::as_str).collect::<Vec<_>>(),
    )
    .unwrap();
    let values = vec![1000.; names.len()];
    let mut candidate = plan.session().unwrap();
    candidate
        .update(&values, &mut EvalBudget::default())
        .unwrap();
    let start = Instant::now();
    let derivatives = black_box(
        evaluate(
            &mut candidate,
            &requests(AnalysisKind::OperatingPoint, None),
            EvalLimits::default(),
        )
        .unwrap(),
    );
    let derivative_time = start.elapsed();
    assert!(
        derivatives
            .iter()
            .all(|value| matches!(value.status, DerivativeStatus::Defined(_))),
        "{derivatives:?}"
    );
    let start = Instant::now();
    for i in 0..values.len() {
        for delta in [-10., 10.] {
            let mut point = values.clone();
            point[i] += delta;
            black_box(full_candidate_voltage(
                &plan,
                &point,
                AnalysisKind::OperatingPoint,
                None,
            ));
        }
    }
    println!(
        "optimizer-gradient inputs={} adjoint_ns={} independent_simulations=64 independent_ns={}",
        values.len(),
        derivative_time.as_nanos(),
        start.elapsed().as_nanos()
    );
}
