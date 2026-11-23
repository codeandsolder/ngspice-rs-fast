//! Model recipe, geometry, contextual arithmetic and atomic elaboration contracts.
use ngspice_rs::devices::{
    AnalysisMode, Circuit, ModelContext, ModelFamily, ModelResolver, PassiveParameters,
    StampContext,
};
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{NodeId, SpiceError};
use std::path::Path;

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("passive.cir"),
            &format!("passives\n{body}\n.end\n"),
        ))
        .unwrap()
}
fn parameters(n: &Netlist, index: usize) -> Result<PassiveParameters, SpiceError> {
    ModelResolver::new(&n.models)?
        .resolve(&n.devices[index])?
        .unwrap()
        .passive_parameters(&n.devices[index])
}
fn effective(body: &str) -> Result<f64, SpiceError> {
    parameters(&deck(body), 0)?.effective_value(&ModelContext::default())
}
fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1e-12 * want.abs() + 1e-24,
        "{got} != {want}"
    );
}

#[test]
fn scalar_models_and_ordered_instance_setters_preserve_raw_ast() {
    for (d, keyword, primary, scalar, family) in [
        ('r', "r", "resistance", "2k", ModelFamily::Resistor),
        ('c', "cap", "capacitance", "2u", ModelFamily::Capacitor),
        ('l', "ind", "inductance", "2m", ModelFamily::Inductor),
    ] {
        let n = deck(&format!(
            "{d}1 a 0 mdl\n{d}2 a 0 3 mdl {primary}=4 {primary}=5\n{d}3 a 0 mdl 6 {primary}=7\n{d}4 a 0 8 mdl 9 {primary}=10\n.model mdl {d}({keyword}=1 {keyword}={scalar})\n.model mdl {d}({keyword}=99)"
        ));
        let before = n.clone();
        let model = parameters(&n, 0).unwrap();
        assert_eq!(model.family(), family);
        close(
            model.nominal_value(),
            ngspice_rs::primitives::parse_spice_number(scalar).unwrap(),
        );
        for (index, want) in [(1, 5.0), (2, 6.0), (3, 9.0)] {
            close(parameters(&n, index).unwrap().nominal_value(), want);
        }
        assert!(Circuit::from_netlist(&n).is_ok());
        assert_eq!(n, before);
    }
    close(effective("r1 a 0 mdl\n.model mdl res(r=2k)").unwrap(), 2e3);
}

#[test]
fn resistor_sheet_precedence_defaults_and_two_sided_corrections() {
    // RSH selects geometry ahead of model R; instance scalar selects ahead of both.
    let n = deck(
        "r1 a 0 mdl l=8u w=4u\nr2 a 0 mdl\nr3 a 0 7k mdl l=1u w=1u\n.model mdl r(r=9k rsh=100 l=12u defw=6u short=1u narrow=0.5u)",
    );
    close(parameters(&n, 0).unwrap().nominal_value(), 200.0);
    close(parameters(&n, 1).unwrap().nominal_value(), 200.0);
    close(parameters(&n, 2).unwrap().nominal_value(), 7e3);
    close(
        effective("r1 a 0 mdl\n.model mdl r(rsh=100)").unwrap(),
        100.0,
    );
    close(
        effective("r1 a 0 mdl\n.model mdl r(r=2k rsh=0)").unwrap(),
        2e3,
    );
}

#[test]
fn capacitor_area_sidewall_geometry_and_scalar_precedence() {
    let n = deck(
        "c1 a 0 mdl l=10u w=6u\nc2 a 0 mdl l=12u\nc3 a 0 7u mdl\n.model mdl c(cj=2m cjsw=3n defw=8u short=2u narrow=1u)",
    );
    close(
        parameters(&n, 0).unwrap().nominal_value(),
        2e-3 * 5e-6 * 8e-6 + 3e-9 * 2.0 * 13e-6,
    );
    close(
        parameters(&n, 1).unwrap().nominal_value(),
        2e-3 * 7e-6 * 10e-6 + 3e-9 * 2.0 * 17e-6,
    );
    close(parameters(&n, 2).unwrap().nominal_value(), 7e-6);
    close(
        effective("c1 a 0 mdl l=10u w=6u\n.model mdl c(cap=2u cj=2m)").unwrap(),
        2e-6,
    );
    close(
        effective("c1 a 0 mdl l=4u\n.model mdl c(cjsw=1n)").unwrap(),
        1e-9 * 2.0 * 14e-6,
    );
}

#[test]
fn temperature_coefficients_scale_and_parallel_multiplicity() {
    let context = ModelContext::new(77.0, 22.0);
    for (d, keyword, primary, want) in [
        ('r', "r", "resistance", 180.0),
        ('c', "cap", "capacitance", 720.0),
        ('l', "ind", "inductance", 180.0),
    ] {
        let n = deck(&format!(
            "{d}1 a 0 mdl {primary}=100 tc1=0.01 tc2=0.001 temp=40 scale=3 m=2\n.model mdl {d}({keyword}=90 tc1=99 tc2=99 tnom=30)"
        ));
        let p = parameters(&n, 0).unwrap();
        close(p.effective_value(&context).unwrap(), want); // dt=10, factor=1.2
        close(p.effective_value(&context).unwrap(), want);
        assert_eq!(p.multiplicity(), 2.0);
    }
    // Independent per-coefficient override; no override/default is cached into AST.
    let n = deck("r1 a 0 mdl tc1=0\n.model mdl r(r=100 tc1=1 tc2=0.001)");
    let p = parameters(&n, 0).unwrap();
    close(
        p.effective_value(&context).unwrap(),
        100.0 * (1.0 + 0.001 * 55.0 * 55.0),
    );
    close(p.effective_value(&ModelContext::default()).unwrap(), 100.0);
    assert!(n.models[0].parameters.iter().all(|p| p.name != "tnom"));
}

#[test]
fn negative_instance_resistance_is_retained_but_models_and_cl_are_positive() {
    close(effective("r1 a 0 -2k mdl m=2\n.model mdl r").unwrap(), -1e3);
    for body in [
        "r1 a 0 mdl\n.model mdl r(r=-2k)",
        "c1 a 0 -2u mdl\n.model mdl c",
        "l1 a 0 -2m mdl\n.model mdl l",
    ] {
        assert!(effective(body).is_err(), "{body}");
    }
}

#[test]
fn missing_values_and_invalid_geometry_fail_without_c_fallbacks() {
    for body in [
        "r1 a 0 mdl\n.model mdl r",
        "r1 a 0 mdl\n.model mdl r(rsh=0)",
        "r1 a 0 mdl l=2u\n.model mdl r(rsh=100 short=1u)",
        "r1 a 0 mdl w=2u\n.model mdl r(rsh=100 narrow=1u)",
        "c1 a 0 mdl\n.model mdl c(cj=1m)",
        "c1 a 0 mdl l=2u\n.model mdl c(cj=1m short=2u)",
        "c1 a 0 mdl l=2u w=1u\n.model mdl c(cj=1m narrow=2u)",
        "l1 a 0 mdl\n.model mdl l",
    ] {
        let error = Circuit::from_netlist(&deck(body)).unwrap_err();
        assert!(matches!(error, SpiceError::Parse { .. }), "{body}: {error}");
        assert!(error.to_string().contains("passive.cir:"), "{error}");
    }
}

#[test]
fn all_raw_setters_are_validated_even_if_overwritten_or_unused_by_precedence() {
    for setter in [
        "r=0",
        "r=-1 r=2k",
        "rsh=-1",
        "defw=0",
        "l=-1",
        "short=-1u",
        "narrow=-1u",
        "tnom=-273.15",
        "noisy=0",
        "res=2k",
        "w=1u",
    ] {
        let n = deck(&format!("r1 a 0 1k mdl\n.model mdl r({setter})"));
        assert!(parameters(&n, 0).is_err(), "{setter}");
    }
    for (d, setter) in [
        ('r', "w=0"),
        ('r', "m=0"),
        ('c', "capacitance=-1 capacitance=1u"),
        ('l', "inductance=-1"),
        ('l', "scale=-1"),
        ('r', "temp=-273.15"),
        ('r', "ac=0"),
        ('r', "tc=1"),
        ('c', "bv_max=1"),
        ('l', "nt=2"),
    ] {
        let n = deck(&format!("{d}1 a 0 mdl {setter}\n.model mdl {d}"));
        assert!(parameters(&n, 0).is_err(), "{d} {setter}");
    }
    for (base, setters) in [
        ("c", "defl=12u"),
        ("c", "cox=1m"),
        ("c", "del=-1u"),
        ("c", "thick=0"),
        ("l", "csect=-1u"),
        ("l", "nt=0"),
        ("l", "mu=0"),
    ] {
        assert!(
            parameters(
                &deck(&format!("{base}1 a 0 1 mdl\n.model mdl {base}({setters})")),
                0
            )
            .is_err()
        );
    }
}

#[test]
fn nonfinite_programmatic_values_and_arithmetic_overflow_are_rejected() {
    for text in ["NaN", "inf", "1e999", "{expr}"] {
        let mut n = deck("r1 a 0 mdl\n.model mdl r(r=2k)");
        n.models[0].parameters[0].value = text.into();
        assert!(parameters(&n, 0).is_err());
        let mut n = deck("r1 a 0 mdl m=2\n.model mdl r(r=2k)");
        n.devices[0].parameters[0].value = text.into();
        assert!(parameters(&n, 0).is_err());
    }
    // Written but overwritten/unused inputs must still be validated. The
    // parser correctly rejects nonfinite literals before reaching this path.
    let mut n = deck("r1 a 0 1k mdl dtemp=5 dtemp=0\n.model mdl r(tce=2 tce=0)");
    n.models[0].parameters[0].value = "1e999".into();
    assert!(parameters(&n, 0).is_err());
    n.models[0].parameters[0].value = "2".into();
    let bad = n.devices[0]
        .parameters
        .iter_mut()
        .find(|p| p.name == "dtemp")
        .unwrap();
    bad.value = "1e999".into();
    assert!(parameters(&n, 0).is_err());
    for body in [
        "r1 a 0 mdl\n.model mdl r(rsh=1e300 l=1e300 defw=1e-300)",
        "r1 a 0 mdl\n.model mdl r(rsh=1 narrow=1e308)",
        "c1 a 0 mdl l=1e300 w=1e300\n.model mdl c(cj=1)",
        "c1 a 0 mdl l=1e308 w=1e308\n.model mdl c(cjsw=1)",
        "r1 a 0 mdl temp=40\n.model mdl r(r=1 tc1=1e308 tnom=20)",
        "c1 a 0 mdl temp=40\n.model mdl c(cap=1 tc2=1e308 tnom=20)",
        "l1 a 0 mdl scale=2\n.model mdl l(ind=1e308)",
        "c1 a 0 mdl m=2\n.model mdl c(cap=1e308)",
        "r1 a 0 mdl m=1e308\n.model mdl r(r=1e-300)",
        "r1 a 0 mdl\n.model mdl r(r=1e-320)",
        "r1 a 0 mdl temp=30\n.model mdl r(r=1 tc1=-1 tnom=20)",
        "c1 a 0 mdl temp=30\n.model mdl c(cap=1 tc1=-0.1 tnom=20)",
    ] {
        assert!(Circuit::from_netlist(&deck(body)).is_err(), "{body}");
    }
}

#[test]
fn failed_instance_elaboration_preserves_all_circuit_namespaces() {
    let mut circuit =
        Circuit::from_netlist(&deck("v1 old 0 1\nl1 old mid 1m\nr1 mid 0 1k")).unwrap();
    let nodes = circuit.nodes().nodes().to_vec();
    let names: Vec<_> = circuit
        .devices()
        .iter()
        .map(|d| d.name().to_owned())
        .collect();
    let rows: Vec<_> = (0..circuit.device_count())
        .map(|i| circuit.branch_rows(i))
        .collect();
    let unknowns = circuit.unknown_count();
    for body in [
        "r2 new 0 missing\n.model missing r(r=1k)",
        "r2 new 0 mdl\n.model mdl c(cap=1u)",
        "r2 new 0 mdl\n.model mdl r(rsh=100 narrow=10u)",
        "c2 new 0 mdl\n.model mdl c(cj=1m)",
        "l2 new 0 mdl\n.model mdl l(ind=1m mu=0)",
        "r2 new 0 mdl temp=40\n.model mdl r(r=1 tc1=-1 tnom=20)",
        "r2 new 0 mdl\n.model mdl r(level=2 r=1)",
        "v1 new 0 1",
        "l2 new 0 mdl\n.model mdl l(ind=1e308)",
    ] {
        let mut n = deck(body);
        if n.devices[0].model.as_deref() == Some("missing") {
            n.models.clear();
        }
        if body.contains("1e308") {
            n.devices[0].nodes.push("extra".into());
        }
        let resolver = ModelResolver::new(&n.models).unwrap();
        assert!(
            circuit
                .add_instance(&n.devices[0], &resolver, &ModelContext::default())
                .is_err(),
            "{body}"
        );
        assert_eq!(circuit.nodes().nodes(), nodes);
        assert_eq!(
            circuit
                .devices()
                .iter()
                .map(|d| d.name().to_owned())
                .collect::<Vec<_>>(),
            names
        );
        assert_eq!(
            (0..circuit.device_count())
                .map(|i| circuit.branch_rows(i))
                .collect::<Vec<_>>(),
            rows
        );
        assert_eq!(circuit.unknown_count(), unknowns);
    }
}

#[test]
fn repeated_contextual_assembly_is_immutable_and_numbering_stays_bound() {
    let n = deck(
        "v1 a 0 1\nr1 a b rm\nl1 b 0 lm\nc1 a 0 cm\n.model rm r(r=1k tc1=0.01)\n.model lm l(ind=1m tc1=0.01)\n.model cm c(cap=1u tc1=0.01)",
    );
    let mut circuit = Circuit::from_netlist(&n).unwrap();
    let nodes = circuit.nodes().nodes().to_vec();
    let rows = circuit.branch_rows(2).unwrap();
    let cold = circuit.linear_system().unwrap();
    let hot = circuit
        .linear_system_with_context(&ModelContext::new(77.0, 27.0))
        .unwrap();
    let restored = circuit.linear_system().unwrap();
    assert_eq!(cold.a, restored.a);
    assert_eq!(cold.e, restored.e);
    assert_ne!(cold.a, hot.a);
    assert_ne!(cold.e, hot.e);
    assert_eq!(rows, circuit.branch_rows(2).unwrap());
    assert_eq!(circuit.nodes().nodes(), nodes);
    let error = circuit.linear_system_with_context(&ModelContext::new(f64::NAN, 27.0));
    assert!(error.is_err());
    assert_eq!(circuit.nodes().nodes(), nodes);
}

#[test]
fn real_stamp_uses_explicit_temperature_and_initial_conditions_remain_visible() {
    let n = deck("r1 a 0 mdl\n.model mdl r(r=1k tc1=0.01)");
    let circuit = Circuit::from_netlist(&n).unwrap();
    let unknowns = circuit.unknowns();
    let nodes = circuit.nodes();
    let mut matrix = SparseMatrix::new(1, 1);
    let mut rhs = Vector::zeros(1);
    let solution = Vector::zeros(1);
    let mut context = StampContext {
        matrix: &mut matrix,
        rhs: ngspice_rs::devices::traits::StampRhs::dense(&mut rhs),
        unknowns,
        nodes,
        solution: &solution,
        temperature: 77.0,
        nominal_temperature: 27.0,
        gmin: ngspice_rs::devices::DEFAULT_GMIN,
        reltol: ngspice_rs::devices::DEFAULT_RELTOL,
        frequency: 0.,
        mode: AnalysisMode::OperatingPoint,
        branches: 0..0,
        controls: &[],
        mutual: &[],
        integration: None,
        states: ngspice_rs::devices::DeviceState::none(),
        forcing: None,
        source_factor: 1.,
        voltage_source_value: &mut 0.,
    };
    circuit.devices()[0].stamp(&mut context).unwrap();
    close(matrix.get(0, 0), 1.0 / 1500.0);
    for (d, setter, model) in [('c', "cap", "1u"), ('l', "ind", "1m")] {
        let n = deck(&format!(
            "{d}1 a 0 mdl ic=2\n.model mdl {d}({setter}={model})"
        ));
        let p = parameters(&n, 0).unwrap();
        assert_eq!(p.initial_condition(), Some(2.0));
        assert!(
            Circuit::from_netlist(&n)
                .unwrap()
                .linear_system()
                .unwrap()
                .has_initial_conditions
        );
    }
    // No change to literal factories or negative scalar resistance contract.
    let r =
        ngspice_rs::devices::Resistor::new("r", [NodeId::new(1), NodeId::GROUND], -1.0).unwrap();
    assert_eq!(r.conductance(), -1.0);
}

#[test]
fn unused_models_and_unknown_setters_cannot_disappear_behind_supported_instances() {
    let n = deck("r1 a 0 mdl\n.model mdl r(r=2k)\n.model unused r(kf=1)");
    assert!(
        Circuit::from_netlist(&n)
            .unwrap_err()
            .to_string()
            .contains("unused model declarations")
    );
    let mut n = deck("r1 a 0 mdl\n.model mdl r(r=2k)");
    n.devices[0].model = Some("wrong".into());
    assert!(
        ModelResolver::new(&n.models)
            .unwrap()
            .resolve(&n.devices[0])
            .is_err()
    );
}

#[test]
fn model_backed_capacitors_and_inductors_own_companion_state() {
    use ngspice_rs::devices::LoadRequest;
    use ngspice_rs::maths::integrator::DEFAULT_XMU;
    use ngspice_rs::maths::{IntegrationMethod, StepHistory};
    let n = deck("c1 a 0 cmod l=2 w=1\nl1 a 0 lmod\n.model cmod c(cj=1u)\n.model lmod l(ind=1m)");
    let circuit = Circuit::from_netlist(&n).unwrap();
    assert_eq!(circuit.state_rows(0), Some(0..2));
    assert_eq!(circuit.state_rows(1), Some(2..4));
    let mut history = circuit.state_history().unwrap();
    let solution = Vector::from_slice(&[1.0, 0.5]);
    let context = ModelContext::default();
    fn request<'a>(
        mode: AnalysisMode,
        integration: Option<&'a ngspice_rs::maths::Coefficients>,
        history: &'a ngspice_rs::devices::StateHistory,
        solution: &'a Vector,
        model_context: &'a ModelContext,
    ) -> LoadRequest<'a> {
        LoadRequest {
            mode,
            solution,
            model_context,
            integration,
            history,
            forcing: None,
        }
    }
    let mut trial = history.trial();
    circuit
        .load(
            &request(
                AnalysisMode::OperatingPoint,
                None,
                &history,
                &solution,
                &context,
            ),
            &mut SparseMatrix::new(2, 2),
            &mut Vector::zeros(2),
            &mut trial,
        )
        .unwrap();
    // Effective C = cj * l * w = 2 uF; flux = 1 mH * 0.5 A.
    close(trial.values()[0], 2e-6);
    close(trial.values()[2], 5e-4);
    circuit
        .accept_point(&solution, None, &mut history, trial)
        .unwrap();
    let c = StepHistory::new()
        .trial(IntegrationMethod::Trapezoidal, 1, 1e-6, DEFAULT_XMU)
        .unwrap();
    let mut matrix = SparseMatrix::new(2, 2);
    circuit
        .load(
            &request(
                AnalysisMode::Transient {
                    time: 1e-6,
                    dt: 1e-6,
                },
                Some(&c),
                &history,
                &solution,
                &context,
            ),
            &mut matrix,
            &mut Vector::zeros(2),
            &mut history.trial(),
        )
        .unwrap();
    matrix.fold_duplicates();
    close(matrix.get(0, 0), 2.0);
    close(matrix.get(1, 1), -1e-3 / 1e-6);
}

/// Repeated bound loads (including a verified fixed-stamp replay) must
/// match unbound stamping bit-for-bit with ground and aliased terminals.
#[test]
fn fixed_passive_batches_match_unbound_stamps() {
    use ngspice_rs::devices::LoadRequest;
    use ngspice_rs::maths::integrator::DEFAULT_XMU;
    use ngspice_rs::maths::{IntegrationMethod, StepHistory};

    let netlist = deck(
        "r1 a b rm\nr2 0 b 4\nr3 a a 2\nc1 a b cm\nc2 a a 0.125\nc3 0 b 0.25\nl1 a 0 0.25\n.model rm r(r=8 tc1=0.01)\n.model cm c(cap=0.5)",
    );
    let circuit = Circuit::from_netlist(&netlist).unwrap();
    let n = circuit.unknown_count();
    let solution = Vector::from_slice(&[2., -1., 0.5]);
    assert_eq!(n, solution.len());
    let mut history = circuit.state_history().unwrap();
    let model = ModelContext::default();
    let mut initial = history.trial();
    circuit
        .load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &solution,
                model_context: &model,
                integration: None,
                history: &history,
                forcing: None,
            },
            &mut circuit.new_matrix().unwrap(),
            &mut Vector::zeros(n),
            &mut initial,
        )
        .unwrap();
    circuit
        .accept_point(&solution, None, &mut history, initial)
        .unwrap();
    let coefficients = StepHistory::new()
        .trial(IntegrationMethod::Trapezoidal, 1, 0.125, DEFAULT_XMU)
        .unwrap();
    for temperature in [27., 77., 77., 27., 77.] {
        let model = ModelContext::new(temperature, 27.);
        let request = LoadRequest {
            mode: AnalysisMode::Transient {
                time: 0.125,
                dt: 0.125,
            },
            solution: &solution,
            model_context: &model,
            integration: Some(&coefficients),
            history: &history,
            forcing: None,
        };
        let mut reference = SparseMatrix::new(n, n);
        let mut bound = circuit.new_matrix().unwrap();
        let (mut ra, mut rb) = (Vector::zeros(n), Vector::zeros(n));
        let (mut ta, mut tb) = (history.trial(), history.trial());
        circuit
            .load(&request, &mut reference, &mut ra, &mut ta)
            .unwrap();
        circuit
            .load(&request, &mut bound, &mut rb, &mut tb)
            .unwrap();
        for row in 0..n {
            for col in 0..n {
                assert_eq!(
                    reference.get(row, col).to_bits(),
                    bound.get(row, col).to_bits(),
                    "{row},{col}"
                );
            }
            assert_eq!(
                ra.get(row).unwrap().to_bits(),
                rb.get(row).unwrap().to_bits()
            );
        }
        for (a, b) in ta.values().iter().zip(tb.values()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

#[test]
fn modeled_noise_uses_the_same_resistor_kernel_and_model_identity() {
    use ngspice_rs::devices::noise::{DeviceNoise, NoiseContext};
    use ngspice_rs::devices::{Device, Resistor};

    let n = deck(
        "r1 a 0 rm m=2 w=4 l=2 temp=50\nr2 a 0 rm noisy=0\nc1 a 0 cm\nl1 a 0 lm\n.model rm r(r=8 kf=0.25 af=2 ef=1.5)\n.model cm c(cap=1u)\n.model lm l(ind=1m)",
    );
    let circuit = Circuit::from_netlist(&n).unwrap();
    let model = ModelContext::new(77., 27.);
    let bias = Vector::from_slice(&[2., 0.]);
    let unknowns = circuit.unknowns();
    let context = NoiseContext {
        model_context: &model,
        unknowns,
        bias: &bias,
        states: None,
        controls: &[],
    };
    let p = parameters(&n, 0).unwrap();
    let literal = Resistor::new("r1", [NodeId::new(1), NodeId::GROUND], 4.)
        .unwrap()
        .with_noise(p.resistor_noise("rm"));
    let mut modeled_sources = Vec::new();
    let mut literal_sources = Vec::new();
    assert_eq!(
        circuit.devices()[0]
            .noise(&context, &mut modeled_sources)
            .unwrap(),
        literal.noise(&context, &mut literal_sources).unwrap()
    );
    assert_eq!(modeled_sources, literal_sources);
    for device in &circuit.devices()[1..] {
        modeled_sources.clear();
        assert_eq!(
            device.noise(&context, &mut modeled_sources).unwrap(),
            DeviceNoise::Noiseless
        );
        assert!(modeled_sources.is_empty());
    }
}

/// C accepts these (`resparam.c` raises a resistance in `[0, RESMIN)` to
/// `RESMIN = 1e-12` with a warning, a later setter replacing it; zero
/// capacitance and inductance are legal values, and a geometry capacitor
/// without `cj`/`cjsw` is zero), so the port does too.
#[test]
fn c_accepted_zero_and_tiny_values() {
    assert_eq!(
        effective("r1 a 0 mdl resistance=0\n.model mdl r").unwrap(),
        1e-12
    );
    assert_eq!(
        effective("r1 a 0 mdl resistance=-1e-15\n.model mdl r").unwrap(),
        -1e-12
    );
    assert_eq!(
        effective("r1 a 0 mdl resistance=0 resistance=1k\n.model mdl r").unwrap(),
        1e3
    );
    assert_eq!(
        effective("l1 a 0 mdl inductance=0\n.model mdl l").unwrap(),
        0.0
    );
    assert_eq!(
        effective("c1 a 0 mdl capacitance=0\n.model mdl c").unwrap(),
        0.0
    );
    assert_eq!(effective("c1 a 0 mdl l=1u\n.model mdl c").unwrap(), 0.0);
}

#[test]
fn prepared_passive_values_match_sensitivity_records_bitwise() {
    let n = deck(
        "\
r1 a 0 rm m=3.7 scale=1.13 dtemp=17.3
r2 b 0 rm temp=41.9 dtemp=100 tce=0 ac=63.1
r3 c 0 1e-15 rm
r4 d 0 rp m=1.7 scale=1.31 temp=73.2 tc2=0.000013
c1 e 0 cm l=13.7u w=4.19u m=2.2 scale=1.013 dtemp=11.7 tc1=0
c2 f 0 cp l=3.41u w=1.77u temp=52.1 tc2=0.000041
l1 g 0 lm m=2.7 scale=0.71 dtemp=6.9
l2 h 0 lm nt=31.7 m=1.9 temp=71.9 dtemp=15
l3 i 0 3.2m lm nt=5
.model rm r(rsh=17.9 l=13.2u defw=4.1u short=0.42u narrow=0.11u tce=0.3 tc1=1 tc2=2 tnom=19.7)
.model rp r(r=237.9 tc1=0.00317 tc2=0.00009 tnom=22.1)
.model cm c(thick=13.1n di=4.7 cjsw=1.329n del=0.11u narrow=0.19u defw=3u tc1=0.001 tc2=0.0000013)
.model cp c(cj=1.337m cjsw=2.193n narrow=0.21u short=0.07u tc1=0.00373 tc2=0.0000021 tnom=29.1)
.model lm l(dia=7.3m csect=1u length=11.9m nt=22.9 mu=2.13 tc1=0.00091 tc2=0.000013 tnom=21.3)
",
    );
    let circuit = Circuit::from_netlist(&n).unwrap();
    for context in [
        ModelContext::default(),
        ModelContext::new(77.3, 19.6),
        ModelContext::new(-9.5, 23.7),
    ] {
        for (index, ast) in n.devices.iter().enumerate() {
            let recipe = parameters(&n, index).unwrap();
            let dev = circuit.device(&ast.name).unwrap();
            let description = dev.sensitivity(&context).unwrap();
            let (_, records) = description.records().unwrap();
            let observed = |name| dev.observation_parameter(name, &context).unwrap().unwrap();
            match ast.designator {
                'r' => {
                    assert_eq!(
                        recipe.nominal_value().to_bits(),
                        records.value("resistance").unwrap().to_bits(),
                        "{}",
                        ast.name
                    );
                    assert_eq!(
                        observed("conductance").to_bits(),
                        records.value("conduct").unwrap().to_bits(),
                        "{}",
                        ast.name
                    );
                }
                'c' => assert_eq!(
                    recipe.effective_value(&context).unwrap().to_bits(),
                    (records.value("capac").unwrap() * records.value("m").unwrap()).to_bits(),
                    "{}",
                    ast.name
                ),
                'l' => assert_eq!(
                    recipe.coupling_value(&context).unwrap().to_bits(),
                    records.value("induct").unwrap().to_bits(),
                    "{}",
                    ast.name
                ),
                _ => panic!("unexpected passive"),
            }
        }
    }
}

#[test]
fn temperature_time_geometry_sweeps_reuse_setup_and_preserve_given_precedence() {
    let n = deck(
        "\
r1 a 0 rm
c1 b 0 cm l=4u
l1 c 0 lm nt=10
.model rm r(rsh=100 l=10u defw=5u)
.model cm c(cj=2m cjsw=3n defw=7u del=0.1u)
.model lm l(length=20m dia=5m nt=12)
",
    );
    let context = ModelContext::new(67.3, 25.1);
    let circuit = Circuit::from_netlist_with_context(&n, &context).unwrap();
    for (name, keyword, value, nominal_keyword, replay_keyword) in [
        ("r1", "w", 2.5e-6, "conductance", "conduct"),
        ("c1", "l", 8.0e-6, "capacitance", "capac"),
        ("c1", "w", 9.0e-6, "capacitance", "capac"),
        ("l1", "nt", 19., "inductance", "induct"),
    ] {
        let original = circuit.device(name).unwrap();
        let before = original
            .observation_parameter(nominal_keyword, &context)
            .unwrap()
            .unwrap();
        let swept = original
            .with_instance_parameter(keyword, value, &context)
            .unwrap();
        let got = swept
            .observation_parameter(nominal_keyword, &context)
            .unwrap()
            .unwrap();
        let replay = swept.sensitivity(&context).unwrap();
        let (_, records) = replay.records().unwrap();
        assert_eq!(
            got.to_bits(),
            records.value(replay_keyword).unwrap().to_bits(),
            "{name}: {keyword}"
        );
        assert_ne!(got.to_bits(), before.to_bits());
        assert_eq!(
            original
                .observation_parameter(nominal_keyword, &context)
                .unwrap()
                .unwrap()
                .to_bits(),
            before.to_bits()
        );
    }
}

#[test]
fn new_passive_inputs_validate_geometry_and_runtime_domains() {
    for body in [
        "r1 a 0 rm dtemp=-400\n.model rm r(r=1k)",
        "r1 a 0 rm ac=1e-320\n.model rm r(r=1k)",
        "r1 a 0 rm temp=80\n.model rm r(r=1k tce=1e308)",
        "l1 a 0 lm nt=2\n.model lm l(ind=1m)",
        "l1 a 0 lm\n.model lm l(nt=10 length=1m)",
        "l1 a 0 lm\n.model lm l(nt=1e200 length=1m dia=1m)",
        "c1 a 0 cm l=3u\n.model cm c(thick=1e-320 di=1e300)",
    ] {
        assert!(Circuit::from_netlist(&deck(body)).is_err(), "{body}");
    }
}
