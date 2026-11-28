//! Physical resistor metadata and the immutable per-point override (#35).
//! The override replaces the *supplied scalar*; the effective resistance is
//! recomputed from the context, and no device, model recipe or AST changes.
use ngspice_rs::devices::{
    AnalysisMode, Capacitor, Circuit, IndependentSource, Inductor, LoadRequest,
    MAX_RESISTOR_OVERRIDES, ModelContext, Resistor, ResistorOrigin, Waveform,
};
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{Complex, NodeId};
use std::path::Path;

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("override.cir"),
            &format!("override\n{body}\n.end\n"),
        ))
        .unwrap()
}
fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1e-12 * want.abs() + 1e-24,
        "{got} != {want}"
    );
}
/// `v9` is a resistor, `r9` a voltage source, `rc` a capacitor, `r_l` an inductor:
/// names deliberately contradict SPICE's first-letter convention.
fn misnamed() -> Circuit {
    let mut c = Circuit::new();
    let a = c.add_node("a").unwrap();
    let b = c.add_node("b").unwrap();
    c.add_device(Box::new(Resistor::new("v9", [a, b], 1e3).unwrap()))
        .unwrap();
    c.add_device(Box::new(
        IndependentSource::new(
            "r9",
            [a, NodeId::GROUND],
            true,
            1.,
            Complex::ZERO,
            Waveform::Constant(1.),
        )
        .unwrap(),
    ))
    .unwrap();
    c.add_device(Box::new(
        Capacitor::new("rc", [b, NodeId::GROUND], 1e-6, None).unwrap(),
    ))
    .unwrap();
    c.add_device(Box::new(
        Inductor::new("r_l", [b, NodeId::GROUND], 1e-3, None).unwrap(),
    ))
    .unwrap();
    c.finalize().unwrap();
    c
}
fn load_a00(c: &Circuit, context: &ModelContext) -> f64 {
    let history = c.state_history().unwrap();
    let mut trial = history.trial();
    let mut matrix = SparseMatrix::new(c.unknown_count(), c.unknown_count());
    c.load(
        &LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &Vector::zeros(c.unknown_count()),
            model_context: context,
            integration: None,
            history: &history,
            forcing: None,
        },
        &mut matrix,
        &mut Vector::zeros(c.unknown_count()),
        &mut trial,
    )
    .unwrap();
    matrix.get(0, 0)
}

#[test]
fn metadata_comes_from_the_device_not_the_name_prefix() {
    let c = misnamed();
    let (index, metadata) = c.resistor("V9").unwrap(); // case-insensitive
    assert_eq!(index, 0);
    assert_eq!(metadata.origin, ResistorOrigin::Literal);
    assert_eq!((metadata.supplied, metadata.multiplicity), (1e3, 1.));
    for not_a_resistor in ["r9", "rc", "r_l", "missing", "r1.r", "tc1", ""] {
        assert!(c.resistor(not_a_resistor).is_none(), "{not_a_resistor}");
        assert!(c.resistor_override(not_a_resistor, 1e3).is_err());
    }
    let n = deck("r1 a 0 rm scale=2 m=4\n.model rm r(r=1k)");
    let c = Circuit::from_netlist(&n).unwrap();
    let (_, metadata) = c.resistor("r1").unwrap();
    assert_eq!(metadata.origin, ResistorOrigin::ModelBacked);
    // The supplied scalar excludes temperature, scale and multiplicity.
    assert_eq!((metadata.supplied, metadata.multiplicity), (1e3, 4.));
    // Model-backed capacitors/inductors are not resistors.
    let n = deck("c1 a 0 cm\nl1 a 0 lm\n.model cm c(cap=1u)\n.model lm l(ind=1m)");
    let c = Circuit::from_netlist(&n).unwrap();
    assert!(c.resistor("c1").is_none() && c.resistor("l1").is_none());
}

#[test]
fn override_construction_rejects_unusable_supplied_values() {
    let c = misnamed();
    let ok = c.resistor_override("v9", 500.).unwrap();
    assert_eq!((ok.device(), ok.supplied()), (0, 500.));
    // Negative nonzero scalars stay allowed, as for literal instances.
    assert!(c.resistor_override("v9", -50.).is_ok());
    for bad in [0., -0., f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e-320] {
        assert!(c.resistor_override("v9", bad).is_err(), "{bad}");
    }
}

#[test]
fn context_holds_a_bounded_duplicate_free_set_of_overrides() {
    assert_eq!(MAX_RESISTOR_OVERRIDES, 2);
    let n = deck("r1 a 0 1k\nr2 a 0 1k\nr3 a 0 1k");
    let c = Circuit::from_netlist(&n).unwrap();
    let context = ModelContext::default();
    assert!(context.resistor_overrides.iter().all(Option::is_none));
    let one = context
        .with_resistor_override(c.resistor_override("r1", 1.).unwrap())
        .unwrap();
    let two = one
        .with_resistor_override(c.resistor_override("r2", 2.).unwrap())
        .unwrap();
    assert_eq!(context, ModelContext::default()); // Copy: the original is unchanged
    assert!(
        one.with_resistor_override(c.resistor_override("r1", 9.).unwrap())
            .unwrap_err()
            .to_string()
            .contains("duplicate")
    );
    assert!(
        two.with_resistor_override(c.resistor_override("r3", 3.).unwrap())
            .unwrap_err()
            .to_string()
            .contains("too many")
    );
}

#[test]
fn every_assembly_and_load_path_sees_the_override_without_mutating_devices() {
    let n = deck("r1 a 0 1k");
    let before = n.clone();
    let mut c = Circuit::from_netlist(&n).unwrap();
    let context = ModelContext::default()
        .with_resistor_override(c.resistor_override("r1", 250.).unwrap())
        .unwrap();
    let zero = Vector::zeros(c.unknown_count());
    close(
        c.linear_system_with_context(&context).unwrap().a.get(0, 0),
        1. / 250.,
    );
    close(
        c.small_signal_system(&context, &zero).unwrap().a.get(0, 0),
        1. / 250.,
    );
    close(load_a00(&c, &context), 1. / 250.);
    // The circuit, its resistor and the AST are exactly as before.
    close(c.linear_system().unwrap().a.get(0, 0), 1. / 1e3);
    close(load_a00(&c, &ModelContext::default()), 1. / 1e3);
    assert_eq!(c.resistor("r1").unwrap().1.supplied, 1e3);
    assert_eq!(n, before);
}

#[test]
fn model_backed_override_keeps_temperature_tc_scale_and_multiplicity() {
    // R = supplied * (1 + 0.01*20) * scale / m = supplied * 0.6 at 47 C, tnom 27 C.
    let n = deck("r1 a 0 rm scale=2 m=4\n.model rm r(r=1k tc1=0.01)");
    let hot = ModelContext::new(47., 27.);
    let mut c = Circuit::from_netlist_with_context(&n, &hot).unwrap();
    close(
        c.linear_system_with_context(&hot).unwrap().a.get(0, 0),
        1. / 600.,
    );
    for (supplied, effective) in [(1e3, 600.), (2e3, 1200.), (-3e3, -1800.)] {
        let target = c.resistor_override("r1", supplied).unwrap();
        close(c.effective_resistance(&target, &hot).unwrap(), effective);
        let context = hot.with_resistor_override(target).unwrap();
        close(
            c.linear_system_with_context(&context).unwrap().a.get(0, 0),
            1. / effective,
        );
        close(
            c.small_signal_system(&context, &Vector::zeros(1))
                .unwrap()
                .a
                .get(0, 0),
            1. / effective,
        );
        close(load_a00(&c, &context), 1. / effective);
    }
    // At nominal temperature the same supplied scalar gives scale/m only.
    let nominal = ModelContext::default();
    let target = c.resistor_override("r1", 2e3).unwrap();
    close(c.effective_resistance(&target, &nominal).unwrap(), 1e3);
    // A temperature that makes the factor nonpositive is an error, not a clamp.
    let n = deck("r1 a 0 rm\n.model rm r(r=1k tc1=-0.1)");
    let c = Circuit::from_netlist(&n).unwrap();
    let target = c.resistor_override("r1", 1e3).unwrap();
    assert!(
        c.effective_resistance(&target, &ModelContext::new(87., 27.))
            .is_err()
    );
    assert!(
        c.effective_resistance(&target, &ModelContext::new(f64::NAN, 27.))
            .is_err()
    );
}

#[test]
fn instance_temp_still_overrides_the_swept_circuit_temperature() {
    let n = deck("r1 a 0 rm temp=77\n.model rm r(r=1k tc1=0.01 tnom=27)");
    let c = Circuit::from_netlist(&n).unwrap();
    let target = c.resistor_override("r1", 1e3).unwrap();
    for temperature in [0., 27., 127.] {
        close(
            c.effective_resistance(&target, &ModelContext::new(temperature, 27.))
                .unwrap(),
            1e3 * 1.5,
        );
    }
}

#[test]
fn stale_or_mismatched_overrides_are_errors() {
    let a = Circuit::from_netlist(&deck("r1 x 0 1k\nr2 x 0 1k\nr3 x 0 1k")).unwrap();
    let late = ModelContext::default()
        .with_resistor_override(a.resistor_override("r3", 5.).unwrap())
        .unwrap();
    // Ordinal 2 does not exist in a one-device circuit.
    let mut small = Circuit::from_netlist(&deck("r1 x 0 1k")).unwrap();
    let error = small.linear_system_with_context(&late).unwrap_err();
    assert!(error.to_string().contains("stale"), "{error}");
    assert!(small.small_signal_system(&late, &Vector::zeros(1)).is_err());
    assert!(
        small
            .effective_resistance(
                &a.resistor_override("r3", 5.).unwrap(),
                &ModelContext::default()
            )
            .is_err()
    );
    // Ordinal 0 exists but is a capacitor here.
    let first = ModelContext::default()
        .with_resistor_override(a.resistor_override("r1", 5.).unwrap())
        .unwrap();
    let mut other = Circuit::from_netlist(&deck("c1 x 0 1u\nr1 x 0 1k")).unwrap();
    let error = other.linear_system_with_context(&first).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("foreign or stale resistor override"),
        "{error}"
    );
}

#[test]
fn resistor_overrides_preserve_observed_recipe_and_noise() {
    use ngspice_rs::devices::noise::NoisePlan;

    let original =
        "r1 a 0 rm scale=2 m=4 temp=47\nr2 a 0 1k noisy=0\n.model rm r(r=1k tc1=0.01 kf=1e-12)";
    let c = Circuit::from_netlist(&deck(original)).unwrap();
    let explicit = Circuit::from_netlist(&deck(
        "r1 a 0 2k rm scale=2 m=4 temp=47\nr2 a 0 2k noisy=0\n.model rm r(r=1k tc1=0.01 kf=1e-12)",
    ))
    .unwrap();
    let context = ModelContext::default()
        .with_resistor_override(c.resistor_override("r1", 2000.).unwrap())
        .unwrap()
        .with_resistor_override(c.resistor_override("r2", 2000.).unwrap())
        .unwrap();
    let bias = Vector::from_slice(&[2.]);
    let swept = NoisePlan::bind(&c.prepare_run(), &context, &bias, None).unwrap();
    let expected = NoisePlan::bind(
        &explicit.prepare_run(),
        &ModelContext::default(),
        &bias,
        None,
    )
    .unwrap();
    assert_eq!(
        swept.instances().collect::<Vec<_>>(),
        expected.instances().collect::<Vec<_>>()
    );
    assert_eq!(swept.instances().len(), 1);
    drop(swept);
    let mut c = c;
    c.set_observations(&[
        "@r1[resistance]".into(),
        "@r1[m]".into(),
        "@r1[temp]".into(),
        "@r1[conductance]".into(),
    ])
    .unwrap();
    let observed = c.observe_parameters(&context).unwrap();
    for (got, want) in observed.iter().zip([4800., 4., 47., 1. / 1200.]) {
        close(got.re, want);
    }
    assert_eq!(c.resistor("r1").unwrap().1.supplied, 1000.);
}

#[test]
fn contextual_literal_sweep_preserves_default_model_and_noisy_flag() {
    use ngspice_rs::devices::noise::{DeviceNoise, NoiseContext};
    let c = Circuit::from_netlist(&deck("r1 a 0 1k noisy=0")).unwrap();
    let context = ModelContext::default();
    let replacement = c.devices()[0]
        .with_instance_parameter("tc1", 0.01, &context)
        .unwrap();
    assert_eq!(
        replacement.resistor_metadata().unwrap().origin,
        ResistorOrigin::Literal
    );
    let sensitivity = replacement.sensitivity(&context).unwrap();
    assert_eq!(sensitivity.model(), None);
    let (_, instance) = sensitivity.records().unwrap();
    assert_eq!(instance.value("tc").unwrap().to_bits(), 0.01_f64.to_bits());
    let bias = Vector::from_slice(&[1.]);
    assert_eq!(
        replacement
            .noise(
                &NoiseContext {
                    model_context: &context,
                    unknowns: c.unknowns(),
                    bias: &bias,
                    states: None,
                    controls: &[]
                },
                &mut Vec::new()
            )
            .unwrap(),
        DeviceNoise::Noiseless
    );
    let replacement = replacement
        .with_instance_parameter("m", 2., &context)
        .unwrap();
    assert_eq!(
        replacement
            .noise(
                &NoiseContext {
                    model_context: &context,
                    unknowns: c.unknowns(),
                    bias: &bias,
                    states: None,
                    controls: &[]
                },
                &mut Vec::new()
            )
            .unwrap(),
        DeviceNoise::Noiseless
    );
}

#[test]
fn equal_size_foreign_and_reordered_override_targets_are_rejected() {
    let a = Circuit::from_netlist(&deck("r1 x 0 1k\nr2 x 0 2k")).unwrap();
    let mut b = Circuit::from_netlist(&deck("r2 x 0 2k\nr1 x 0 1k")).unwrap();
    let context = ModelContext::default();
    let target = a.resistor_override("r1", 500.).unwrap();
    let foreign = context.with_resistor_override(target).unwrap();
    assert!(b.effective_resistance(&target, &context).is_err());
    assert!(b.linear_system_with_context(&foreign).is_err());
    let target = b.resistor_override("r1", 500.).unwrap();
    let local = context.with_resistor_override(target).unwrap();
    close(
        b.linear_system_with_context(&local).unwrap().a.get(0, 0),
        1. / 500. + 1. / 2e3,
    );
    b.swap_devices(0, 1).unwrap();
    b.finalize().unwrap();
    assert!(b.linear_system_with_context(&local).is_err());
    let rebound = context
        .with_resistor_override(b.resistor_override("r1", 500.).unwrap())
        .unwrap();
    close(
        b.linear_system_with_context(&rebound).unwrap().a.get(0, 0),
        1. / 500. + 1. / 2e3,
    );
}

#[derive(Debug, Clone)]
struct PreparationProbe {
    nodes: [NodeId; 2],
    value: f64,
    preparations: std::rc::Rc<std::cell::Cell<usize>>,
}

impl ngspice_rs::devices::Device for PreparationProbe {
    fn name(&self) -> &str {
        "rprobe"
    }
    fn designator(&self) -> char {
        'r'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.nodes
    }
    fn state_count(&self) -> usize {
        1
    }
    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        match keyword {
            "x" => Some("x"),
            "y" => Some("y"),
            _ => None,
        }
    }
    fn with_instance_parameter(
        &self,
        _: &str,
        value: f64,
        _: &ModelContext,
    ) -> ngspice_rs::primitives::SpiceResult<Box<dyn ngspice_rs::devices::Device>> {
        self.preparations.set(self.preparations.get() + 1);
        Ok(Box::new(Self {
            value: self.value * 10. + value,
            ..self.clone()
        }))
    }
    fn stamp(
        &self,
        context: &mut ngspice_rs::devices::traits::StampContext<'_>,
    ) -> ngspice_rs::primitives::SpiceResult<()> {
        context.states.set(0, self.value)?;
        context.stamp(self.nodes[0], self.nodes[0], self.value)
    }
    fn convergence_hook(&self) -> Option<&dyn ngspice_rs::devices::DeviceConvergence> {
        (self.value > 10.).then_some(self)
    }
    fn assemble_linear(
        &self,
        context: &mut ngspice_rs::devices::LinearContext<'_>,
    ) -> ngspice_rs::primitives::SpiceResult<()> {
        context.nodal(self.nodes, self.value, false)
    }
    fn observation_parameter(
        &self,
        keyword: &str,
        _: &ModelContext,
    ) -> ngspice_rs::primitives::SpiceResult<Option<f64>> {
        Ok((keyword == "value").then_some(self.value))
    }
}

impl ngspice_rs::devices::DeviceConvergence for PreparationProbe {
    fn convergence_test(
        &self,
        context: &ngspice_rs::devices::traits::ConvergenceContext<'_>,
    ) -> ngspice_rs::primitives::SpiceResult<bool> {
        Ok(context.states[0].to_bits() != self.value.to_bits())
    }
}

#[test]
fn prepared_point_reuses_ordered_setters_for_load_convergence_and_observation() {
    use ngspice_rs::devices::ConvergenceRequest;
    let mut c = Circuit::new();
    let preparations = std::rc::Rc::new(std::cell::Cell::new(0));
    let node = c.add_node("a").unwrap();
    c.add_device(Box::new(PreparationProbe {
        nodes: [node, NodeId::GROUND],
        value: 1.,
        preparations: preparations.clone(),
    }))
    .unwrap();
    c.finalize().unwrap();
    c.set_observations(&["@rprobe[value]".into()]).unwrap();
    let base = ModelContext::default();
    let x = c.instance_override("rprobe", "x", 2., &base).unwrap();
    let y = c.instance_override("rprobe", "y", 3., &base).unwrap();
    let a = base
        .with_instance_override(x)
        .unwrap()
        .with_instance_override(y)
        .unwrap();
    let b = base
        .with_instance_override(y)
        .unwrap()
        .with_instance_override(x)
        .unwrap();
    let run = c.prepare_run();
    let history = c.state_history().unwrap();
    let before = history.accepted(1).map(<[f64]>::to_vec);
    let zero = Vector::zeros(1);
    preparations.set(0);
    for (model, value, count) in [
        (a, 123., 2),
        (a, 123., 2),
        (b, 132., 4),
        (a, 123., 6),
        (a.with_frequency(-0.), 123., 8),
        (a, 123., 10),
        (
            ModelContext {
                temperature: 47.,
                ..a
            },
            123.,
            12,
        ),
    ] {
        let request = LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &zero,
            model_context: &model,
            integration: None,
            history: &history,
            forcing: None,
        };
        for _ in 0..8 {
            let mut trial = history.trial();
            let mut matrix = c.new_matrix().unwrap();
            run.load(&request, &mut matrix, &mut Vector::zeros(1), &mut trial)
                .unwrap();
            assert_eq!(matrix.get(0, 0), value);
            assert_eq!(trial.values()[0], value);
            assert!(
                !run.convergence_test(&ConvergenceRequest {
                    solution: &zero,
                    previous: &zero,
                    history: &history,
                    trial: &trial,
                    model_context: &model,
                    time: 0.,
                    integration: None,
                    reltol: 1e-3,
                    abstol: 1e-12,
                    vntol: 1e-6,
                })
                .unwrap()
            );
            assert_eq!(
                run.observe_real(&request, &[], None, Some(&trial)).unwrap(),
                [Complex::real(value)]
            );
        }
        assert_eq!(
            run.observe_parameters(&model).unwrap(),
            [Complex::real(value)]
        );
        assert_eq!(
            run.small_signal_system_at(&model, &zero, None)
                .unwrap()
                .a
                .get(0, 0),
            value
        );
        assert_eq!(preparations.get(), count);
        assert_eq!(history.accepted(1).map(<[f64]>::to_vec), before);
    }
    let mut duplicate = a;
    duplicate.instance_overrides = [Some(x), Some(x)];
    assert!(
        run.observe_parameters(&duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate instance")
    );
    let last = ModelContext {
        temperature: 47.,
        ..a
    };
    let count = preparations.get();
    assert_eq!(
        run.observe_parameters(&last).unwrap(),
        [Complex::real(123.)]
    );
    assert_eq!(preparations.get(), count);
    assert_eq!(
        c.prepare_run().observe_parameters(&last).unwrap(),
        [Complex::real(123.)]
    );
    assert_eq!(preparations.get(), count + 2);
    assert_eq!(
        c.device("rprobe")
            .unwrap()
            .observation_parameter("value", &base)
            .unwrap(),
        Some(1.)
    );
}

#[test]
fn prepared_scope_rejects_foreign_and_stale_overrides_and_failed_transition_preserves_point() {
    let n = deck("r1 a 0 1k");
    let mut c = Circuit::from_netlist(&n).unwrap();
    c.set_observations(&["@r1[resistance]".into()]).unwrap();
    let other = Circuit::from_netlist(&n).unwrap();
    let base = ModelContext::default();
    let target = c.resistor_override("r1", 250.).unwrap();
    let own = base.with_resistor_override(target).unwrap();
    {
        let run = c.prepare_run();
        assert_eq!(run.observe_parameters(&own).unwrap(), [Complex::real(250.)]);
        let foreign = base
            .with_resistor_override(other.resistor_override("r1", 500.).unwrap())
            .unwrap();
        assert!(
            run.observe_parameters(&foreign)
                .unwrap_err()
                .to_string()
                .contains("foreign or stale")
        );
        let conflict = own
            .with_instance_override(c.instance_override("r1", "m", 2., &base).unwrap())
            .unwrap();
        assert!(
            run.observe_parameters(&conflict)
                .unwrap_err()
                .to_string()
                .contains("both a resistor and an instance")
        );
        assert_eq!(run.observe_parameters(&own).unwrap(), [Complex::real(250.)]);
    }
    c.add_node("new").unwrap();
    c.finalize().unwrap();
    assert!(
        c.prepare_run()
            .small_signal_system_at(&own, &Vector::zeros(c.unknown_count()), None)
            .unwrap_err()
            .to_string()
            .contains("foreign or stale")
    );
}

#[test]
#[ignore = "manual throughput benchmark; both paths execute the same physical load"]
fn prepared_context_load_benchmark() {
    use std::hint::black_box;
    use std::time::Instant;
    const LOADS: usize = 3000;
    for (label, body, instance, parameter, value) in [
        (
            "resistor",
            "r1 a 0 rm m=2\n.model rm r(r=1k tc1=0.01)",
            "r1",
            "m",
            3.,
        ),
        (
            "bjt",
            "q1 c b 0 qm\n.model qm npn(is=1e-15 bf=100)",
            "q1",
            "area",
            2.,
        ),
        (
            "mos1",
            "m1 d g 0 0 mm w=10u l=1u\n.model mm nmos level=1 vto=0.5 kp=100u",
            "m1",
            "w",
            12e-6,
        ),
        (
            "bsim4",
            "m1 d g 0 0 mm w=1u l=0.1u\n.model mm nmos level=54 version=4.8.3",
            "m1",
            "w",
            1.2e-6,
        ),
    ] {
        let c = Circuit::from_netlist(&deck(body)).unwrap();
        let base = ModelContext::default();
        let model = base
            .with_instance_override(
                c.instance_override(instance, parameter, value, &base)
                    .unwrap(),
            )
            .unwrap();
        let history = c.state_history().unwrap();
        let mut bias = Vector::zeros(c.unknown_count());
        for (node, v) in [("a", 1.), ("b", 0.4), ("c", 1.), ("d", 1.), ("g", 0.7)] {
            if let Some(row) = c
                .nodes()
                .get(node)
                .and_then(|node| c.unknowns().node_row(node))
            {
                bias.as_mut_slice()[row] = v;
            }
        }
        let request = LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &bias,
            model_context: &model,
            integration: None,
            history: &history,
            forcing: None,
        };
        let run = c.prepare_run();
        let load = |reuse: bool| {
            let mut trial = history.trial().with_device_limiting(false);
            let mut matrix = c.new_matrix().unwrap();
            let mut rhs = Vector::zeros(c.unknown_count());
            if reuse {
                run.load(&request, &mut matrix, &mut rhs, &mut trial)
                    .unwrap();
            } else {
                c.load(&request, &mut matrix, &mut rhs, &mut trial).unwrap();
            }
            (matrix, rhs, trial)
        };
        let (a, ar, at) = load(false);
        let (b, br, bt) = load(true);
        for r in 0..c.unknown_count() {
            for col in 0..c.unknown_count() {
                assert_eq!(
                    a.get(r, col).to_bits(),
                    b.get(r, col).to_bits(),
                    "{label} [{r},{col}]"
                );
            }
        }
        assert_eq!(
            ar.as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            br.as_slice()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            at.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            bt.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let measure = |reuse| {
            let start = Instant::now();
            for _ in 0..LOADS {
                black_box(load(reuse));
            }
            start.elapsed().as_secs_f64()
        };
        let mut cold = [0.; 3];
        let mut hot = [0.; 3];
        for repeat in 0..3 {
            if repeat == 1 {
                hot[repeat] = measure(true);
                cold[repeat] = measure(false);
            } else {
                cold[repeat] = measure(false);
                hot[repeat] = measure(true);
            }
        }
        cold.sort_by(f64::total_cmp);
        hot.sort_by(f64::total_cmp);
        eprintln!(
            "{label}: {LOADS} loads; one-shot {:.3} ms, prepared {:.3} ms, {:.3}x; bit-identical",
            cold[1] * 1000.,
            hot[1] * 1000.,
            cold[1] / hot[1]
        );
    }
}

#[test]
fn replacement_can_add_and_remove_convergence_capability() {
    use ngspice_rs::devices::ConvergenceRequest;
    for (initial, override_value) in [(1., 2.), (20., -195.)] {
        let mut circuit = Circuit::new();
        let node = circuit.add_node("a").unwrap();
        circuit
            .add_device(Box::new(PreparationProbe {
                nodes: [node, NodeId::GROUND],
                value: initial,
                preparations: Default::default(),
            }))
            .unwrap();
        circuit.finalize().unwrap();
        let base = ModelContext::default();
        let replacement = base
            .with_instance_override(
                circuit
                    .instance_override("rprobe", "x", override_value, &base)
                    .unwrap(),
            )
            .unwrap();
        let run = circuit.prepare_run();
        let history = circuit.state_history().unwrap();
        let trial = history.trial();
        let zero = Vector::zeros(1);
        for (model, expected) in [
            (base, initial > 10.),
            (replacement, initial < 10.),
            (base, initial > 10.),
            (replacement, initial < 10.),
        ] {
            assert_eq!(
                run.convergence_test(&ConvergenceRequest {
                    solution: &zero,
                    previous: &zero,
                    history: &history,
                    trial: &trial,
                    model_context: &model,
                    time: 0.,
                    integration: None,
                    reltol: 1e-3,
                    abstol: 1e-12,
                    vntol: 1e-6,
                })
                .unwrap(),
                expected
            );
        }
    }
}
