//! Prepared-model ABI contract exercised through real Circuit/analysis loads.
//! The analytic polynomial is a source-compiler stand-in; the physical
//! equations live outside the engine and are only called through Kernel.

use std::sync::Arc;

use model_kernel_api::batch::{BatchOutput, PreparedTile};
use model_kernel_api::{
    ABI_VERSION, Artifact, Binding, Capability, Descriptor, Equation, EquationKind, Error, Kernel,
    Parameter, Point,
};
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput};
use ngspice_rs::devices::noise::{NoiseContext, NoiseKind};
use ngspice_rs::devices::prepared_model::{InstanceTemperature, PreparedDevice};
use ngspice_rs::devices::{
    AnalysisMode, Circuit, IndependentSource, LoadRequest, ModelContext, Resistor, Waveform,
};
use ngspice_rs::maths::integrator::DEFAULT_XMU;
use ngspice_rs::maths::{Coefficients, IntegrationMethod, SparseMatrix, StepHistory, Vector};
use ngspice_rs::primitives::{AnalysisKind, Complex, NodeId, Real};

#[derive(Debug, Default)]
struct PolynomialKernel;

#[derive(Debug)]
struct Polynomial {
    g: Real,
    cubic: Real,
    c: Real,
    charge_cubic: Real,
    temperature: Real,
}

impl Kernel for PolynomialKernel {
    type Prepared = Polynomial;

    fn prepare(
        &self,
        parameters: &[Real],
        temperature: Real,
    ) -> model_kernel_api::Result<Polynomial> {
        let [g, cubic, c, charge_cubic] = parameters else {
            return Err(Error::Domain(
                "polynomial artifact has four parameters".into(),
            ));
        };
        if [*g, *cubic, *c, *charge_cubic, temperature]
            .iter()
            .any(|v| !v.is_finite())
            || *g <= 0.
            || *cubic < 0.
            || *c < 0.
            || *charge_cubic < 0.
        {
            return Err(Error::Domain(
                "invalid polynomial physical parameters".into(),
            ));
        }
        Ok(Polynomial {
            g: *g,
            cubic: *cubic,
            c: *c,
            charge_cubic: *charge_cubic,
            temperature,
        })
    }

    fn evaluate(
        &self,
        prepared: &Polynomial,
        voltages: &[Real],
        out: &mut Point<'_>,
    ) -> model_kernel_api::Result<()> {
        let [positive, negative] = voltages else {
            return Err(Error::Domain(
                "polynomial kernel requires two terminals".into(),
            ));
        };
        let v = positive - negative;
        let v2 = v * v;
        let current = prepared.g * v + prepared.cubic * v * v2;
        let conductance = prepared.g + 3. * prepared.cubic * v2;
        let charge = prepared.c * v + prepared.charge_cubic * v * v2;
        let capacitance = prepared.c + 3. * prepared.charge_cubic * v2;
        out.current.copy_from_slice(&[current, -current]);
        out.conductance
            .copy_from_slice(&[conductance, -conductance, -conductance, conductance]);
        out.charge.copy_from_slice(&[charge, -charge]);
        out.capacitance
            .copy_from_slice(&[capacitance, -capacitance, -capacitance, capacitance]);
        Ok(())
    }

    fn evaluate_noise(
        &self,
        prepared: &Polynomial,
        voltages: &[Real],
        _frequency: Real,
        sink: &mut dyn model_kernel_api::noise::Sink,
    ) -> model_kernel_api::Result<()> {
        use model_kernel_api::noise::{Injection, Source};
        let [positive, negative] = voltages else {
            return Err(Error::Shape("polynomial noise input"));
        };
        let voltage = positive - negative;
        let conductance = prepared.g + 3. * prepared.cubic * voltage * voltage;
        sink.push(Source {
            name: "thermal",
            density: 4.
                * ngspice_rs::devices::noise::BOLTZMANN
                * prepared.temperature
                * conductance,
            injections: &[
                Injection {
                    row: 0,
                    re: 1.,
                    im: 0.,
                },
                Injection {
                    row: 1,
                    re: -1.,
                    im: 0.,
                },
            ],
        })
    }
}

fn descriptor(charge: Capability) -> Descriptor {
    Descriptor {
        family: "analytic_polynomial".into(),
        fingerprint: [0x35; 16],
        abi: ABI_VERSION,
        equations: vec![
            Equation {
                name: "p".into(),
                kind: EquationKind::Electrical,
                binding: Binding::Terminal,
            },
            Equation {
                name: "n".into(),
                kind: EquationKind::Electrical,
                binding: Binding::Terminal,
            },
        ],
        parameters: ["g", "cubic", "c", "charge_cubic"]
            .into_iter()
            .zip([1e-3, 2e-4, 1e-6, 2e-7])
            .map(|(name, default)| Parameter {
                name: name.into(),
                default,
                changes_topology: false,
            })
            .collect(),
        charge,
        noise: Capability::Computed,
        trial: model_kernel_api::trial::TrialCapabilities::ENGINE,
        taylor: model_kernel_api::taylor::TaylorCapability::Unavailable,
    }
}

fn artifact() -> Arc<Artifact<PolynomialKernel>> {
    Artifact::new(descriptor(Capability::Computed), PolynomialKernel).unwrap()
}

fn circuit(artifact: &Arc<Artifact<PolynomialKernel>>, g: Real) -> Circuit {
    let mut circuit = Circuit::new();
    let a = circuit.add_node("a").unwrap();
    let b = circuit.add_node("b").unwrap();
    circuit
        .add_device(Box::new(
            PreparedDevice::new(
                artifact,
                "x1",
                vec![a, b],
                &[g, 2e-4, 1e-6, 2e-7],
                InstanceTemperature::circuit(27.),
            )
            .unwrap(),
        ))
        .unwrap();
    circuit.finalize().unwrap();
    circuit
}

fn close(actual: Real, expected: Real) {
    assert!(
        (actual - expected).abs() <= 2e-12 * expected.abs().max(1e-10),
        "{actual:.17e} != {expected:.17e}"
    );
}

#[test]
fn artifact_schema_enforces_versions_slots_and_structural_mutation() {
    let mut schema = descriptor(Capability::Computed);
    schema.abi += 1;
    assert!(Artifact::new(schema.clone(), PolynomialKernel).is_err());
    schema.abi = ABI_VERSION;
    schema.parameters[1].name = "g".into();
    assert!(Artifact::new(schema.clone(), PolynomialKernel).is_err());
    schema.parameters[1].name = "cubic".into();
    let model = Artifact::new(schema, PolynomialKernel).unwrap();
    let first = PreparedDevice::new(
        &model,
        "x1",
        vec![NodeId::GROUND, NodeId::GROUND],
        &[1e-3, 2e-4, 1e-6, 2e-7],
        InstanceTemperature::circuit(27.),
    )
    .unwrap();
    let updated = first.with_parameter_named("G", 5e-3).unwrap();
    assert!(updated.with_parameter_named("not_existing", 1.).is_err());
    assert!(first.with_parameter(0, -1.).is_err());
    assert!(first.with_parameter(9, 1.).is_err());
    let _ = updated;
}

#[test]
fn homogeneous_tile_matches_scalar_equations_and_conserves_kcl_and_charge() {
    let kernel = PolynomialKernel;
    let model = artifact();
    let prepared = [
        kernel.prepare(&[1e-3, 2e-4, 1e-6, 2e-7], 300.15).unwrap(),
        kernel.prepare(&[2e-3, 1e-4, 3e-6, 3e-7], 308.15).unwrap(),
        kernel.prepare(&[3e-3, 4e-4, 2e-6, 1e-7], 263.15).unwrap(),
    ];

    let mut out = BatchOutput::new(2, prepared.len()).unwrap();
    let voltage = [0.5, 0., 1.1, 0.3, -0.4, 0.2];
    let candidates: &[(&[Real], Real)] = &[
        (&[1e-3, 2e-4, 1e-6, 2e-7], 300.15),
        (&[2e-3, 1e-4, 3e-6, 3e-7], 308.15),
        (&[3e-3, 4e-4, 2e-6, 1e-7], 263.15),
    ];
    PreparedTile::new(&model, candidates)
        .unwrap()
        .evaluate(&voltage, &mut out)
        .unwrap();
    for (lane, data) in prepared.iter().enumerate() {
        let v = voltage[2 * lane] - voltage[2 * lane + 1];
        close(out.current[2 * lane], data.g * v + data.cubic * v.powi(3));
        close(
            out.charge[2 * lane],
            data.c * v + data.charge_cubic * v.powi(3),
        );
    }
    assert!(BatchOutput::new(usize::MAX, 2).is_err());
    assert!(BatchOutput::new(2, 0).is_err());
}

fn load(
    circuit: &Circuit,
    history: &ngspice_rs::devices::StateHistory,
    solution: &[Real],
    integration: Option<&Coefficients>,
) -> (SparseMatrix, Vector, ngspice_rs::devices::TrialState) {
    let n = circuit.unknown_count();
    let mut matrix = SparseMatrix::new(n, n);
    let mut rhs = Vector::zeros(n);
    let mut trial = history.trial();
    let mode = integration.map_or(AnalysisMode::OperatingPoint, |coeff| {
        AnalysisMode::Transient {
            time: coeff.dt(),
            dt: coeff.dt(),
        }
    });
    circuit
        .load(
            &LoadRequest {
                mode,
                solution: &Vector::from_slice(solution),
                model_context: &ModelContext::default(),
                integration,
                history,
                forcing: None,
            },
            &mut matrix,
            &mut rhs,
            &mut trial,
        )
        .unwrap();
    matrix.fold_duplicates();
    (matrix, rhs, trial)
}

#[test]
fn prepared_model_stamps_exact_newton_equations_and_ac_linearization() {
    let c = circuit(&artifact(), 1e-3);
    assert_eq!(c.unknown_count(), 2);
    assert_eq!(c.state_len(), 4);
    let history = c.state_history().unwrap();
    let (matrix, rhs, trial) = load(&c, &history, &[0.5, 0.], None);
    let i = 1e-3 * 0.5 + 2e-4 * (0.5_f64).powi(3);
    let g = 1e-3 + 3. * 2e-4 * 0.5_f64.powi(2);
    close(matrix.get(0, 0), g);
    close(matrix.get(0, 1), -g);
    close(rhs.as_slice()[0], -i + g * 0.5);
    close(trial.values()[0], 1e-6 * 0.5 + 2e-7 * 0.5_f64.powi(3));
    assert_eq!(trial.values()[1], 0.);

    let system = c
        .small_signal_system_at(
            &ModelContext::default(),
            &Vector::from_slice(&[0.5, 0.]),
            None,
        )
        .unwrap();
    let capacitance = 1e-6 + 3. * 2e-7 * 0.5_f64.powi(2);
    close(system.a.get(0, 0), g);
    close(system.e.get(0, 0), capacitance);
    close(system.e.get(0, 1), -capacitance);
}

#[test]
fn transient_accept_reject_and_candidate_reprepare_are_isolated() {
    let model = artifact();
    let c = circuit(&model, 1e-3);
    let mut history = c.state_history().unwrap();
    let (.., first) = load(&c, &history, &[0.5, 0.], None);
    c.accept_point(&Vector::from_slice(&[0.5, 0.]), None, &mut history, first)
        .unwrap();

    let steps = StepHistory::new();
    let coefficients = steps
        .trial(IntegrationMethod::Trapezoidal, 1, 1e-6, DEFAULT_XMU)
        .unwrap();
    let (_, _, failed_trial) = load(&c, &history, &[0.75, 0.], Some(&coefficients));
    let accepted = history.accepted(1).unwrap().to_vec();
    assert_eq!(
        history.accepted(1).unwrap(),
        accepted,
        "trial must not commit"
    );
    drop(failed_trial);
    let (_, _, accepted_trial) = load(&c, &history, &[0.75, 0.], Some(&coefficients));
    assert_eq!(
        accepted_trial.values()[0],
        1e-6 * 0.75 + 2e-7 * 0.75_f64.powi(3)
    );
    c.accept_point(
        &Vector::from_slice(&[0.75, 0.]),
        Some(1e-6),
        &mut history,
        accepted_trial,
    )
    .unwrap();
    assert_ne!(history.accepted(1).unwrap(), accepted);

    let altered = circuit(&model, 2e-3);
    let (matrix, _, _) = load(
        &altered,
        &altered.state_history().unwrap(),
        &[0.5, 0.],
        None,
    );
    close(matrix.get(0, 0), 2e-3 + 3. * 2e-4 * 0.5_f64.powi(2));
    close(
        load(&c, &c.state_history().unwrap(), &[0.5, 0.], None)
            .0
            .get(0, 0),
        1e-3 + 3. * 2e-4 * 0.5_f64.powi(2),
    );
}

#[test]
fn real_operating_point_driver_accepts_prepared_native_device() {
    let model = artifact();
    let mut circuit = Circuit::new();
    let positive = circuit.add_node("in").unwrap();
    let out = circuit.add_node("out").unwrap();
    circuit
        .add_device(Box::new(
            IndependentSource::new(
                "v1",
                [positive, NodeId::GROUND],
                true,
                1.,
                Complex::ZERO,
                Waveform::Constant(1.),
            )
            .unwrap(),
        ))
        .unwrap();
    circuit
        .add_device(Box::new(
            Resistor::new("r1", [positive, out], 1000.).unwrap(),
        ))
        .unwrap();
    circuit
        .add_device(Box::new(
            PreparedDevice::new(
                &model,
                "x1",
                vec![out, NodeId::GROUND],
                &[1e-3, 2e-4, 1e-6, 2e-7],
                InstanceTemperature::circuit(27.),
            )
            .unwrap(),
        ))
        .unwrap();
    circuit.finalize().unwrap();
    let plan = AnalysisInput::new(AnalysisKind::OperatingPoint)
        .resolve()
        .unwrap();
    let result = plan.run(&mut circuit, &AnalysisContext::default()).unwrap();
    let voltage = result.value("v(out)", 0).unwrap().re;
    let current = 1e-3 * voltage + 2e-4 * voltage.powi(3);
    // Default Newton tolerances are ngspice's, not machine-precision solve
    // tolerances; check the physical KCL residual against that solve policy.
    let residual = (1. - voltage) / 1000. - current;
    assert!(residual.abs() <= 5e-10 * current.abs(), "{residual:e}");
    assert!(voltage > 0. && voltage < 1.);
}

fn source_driven_model(step: bool) -> Circuit {
    let model = artifact();
    let mut circuit = Circuit::new();
    let input = circuit.add_node("in").unwrap();
    let output = circuit.add_node("out").unwrap();
    let waveform = if step {
        Waveform::Step {
            before: 0.,
            after: 1.,
            time: 3e-6,
        }
    } else {
        Waveform::Constant(0.)
    };
    circuit
        .add_device(Box::new(
            IndependentSource::new(
                "v1",
                [input, NodeId::GROUND],
                true,
                0.,
                Complex::real(1.),
                waveform,
            )
            .unwrap(),
        ))
        .unwrap();
    circuit
        .add_device(Box::new(
            Resistor::new("r1", [input, output], 1000.).unwrap(),
        ))
        .unwrap();
    circuit
        .add_device(Box::new(
            PreparedDevice::new(
                &model,
                "x1",
                vec![output, NodeId::GROUND],
                &[1e-3, 2e-4, 1e-6, 2e-7],
                InstanceTemperature::circuit(27.),
            )
            .unwrap(),
        ))
        .unwrap();
    circuit.finalize().unwrap();
    circuit
}

#[test]
fn ac_driver_uses_the_native_prepared_charge_jacobian() {
    let mut circuit = source_driven_model(false);
    let request = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "1", "1k", "1k"])
        .resolve()
        .unwrap();
    let plot = request
        .run(&mut circuit, &AnalysisContext::default())
        .unwrap();
    let actual = plot.value("v(out)", 0).unwrap();
    let resistance: f64 = 1e-3;
    let admittance = Complex::new(resistance + 1e-3, 2. * std::f64::consts::PI * 1000. * 1e-6);
    let expected = Complex::real(resistance) / admittance;
    assert!((actual.re - expected.re).abs() < 1e-8);
    assert!((actual.im - expected.im).abs() < 1e-8);
}

#[test]
fn transient_driver_integrates_and_accepts_native_prepared_charge() {
    let mut circuit = source_driven_model(true);
    let request =
        AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "10u", "0", "500n"])
            .resolve()
            .unwrap();
    let plot = request
        .run(&mut circuit, &AnalysisContext::default())
        .unwrap();
    assert!(plot.point_count() >= 5);
    let mut early = false;
    let mut late = false;
    for index in 0..plot.point_count() {
        let time = plot.value("time", index).unwrap().re;
        let output = plot.value("v(out)", index).unwrap().re;
        assert!(output.is_finite(), "t={time:e}");
        if time < 2e-6 {
            early = true;
            assert!(output.abs() < 1e-8, "t={time:e} out={output:e}");
        }
        if time > 4e-6 {
            late = true;
            assert!(output > 0. && output < 1., "t={time:e} out={output:e}");
        }
    }
    assert!(early && late);
}

#[test]
fn prepared_model_noise_is_source_backed_not_assumed_zero() {
    let c = circuit(&artifact(), 1e-3);
    let device = &c.devices()[0];
    let bias = Vector::from_slice(&[0.4, 0.]);
    let model_context = ModelContext::default().with_frequency(1.);
    let context = NoiseContext {
        model_context: &model_context,
        unknowns: c.unknowns(),
        bias: &bias,
        states: None,
        controls: &[],
    };
    let mut generators = Vec::new();
    let identity = device.noise(&context, &mut generators).unwrap();
    assert_eq!(generators.len(), 1);
    assert!(matches!(
        identity,
        ngspice_rs::devices::noise::DeviceNoise::Sources {
            family: ngspice_rs::devices::noise::NoiseFamily::Prepared,
            ..
        }
    ));
    match &generators[0].kind {
        NoiseKind::Generalized {
            density,
            injections,
        } => {
            close(
                *density,
                4. * ngspice_rs::devices::noise::BOLTZMANN
                    * 300.15
                    * (1e-3 + 3. * 2e-4 * 0.4_f64.powi(2)),
            );
            assert_eq!(injections.len(), 2);
        }
        other => panic!("unexpected prepared noise {other:?}"),
    }
    let mut unavailable = descriptor(Capability::Computed);
    unavailable.noise = Capability::Unavailable;
    let factory = Artifact::new(unavailable, PolynomialKernel).unwrap();
    let mut circuit = Circuit::new();
    let a = circuit.add_node("a").unwrap();
    circuit
        .add_device(Box::new(
            PreparedDevice::new(
                &factory,
                "x1",
                vec![a, NodeId::GROUND],
                &[1e-3, 2e-4, 1e-6, 2e-7],
                InstanceTemperature::circuit(27.),
            )
            .unwrap(),
        ))
        .unwrap();
    circuit.finalize().unwrap();
    let bias = Vector::from_slice(&[0.4]);
    let context = NoiseContext {
        model_context: &model_context,
        unknowns: circuit.unknowns(),
        bias: &bias,
        states: None,
        controls: &[],
    };
    assert!(
        circuit.devices()[0]
            .noise(&context, &mut generators)
            .is_err()
    );
}

#[test]
fn model_temperature_changes_require_preparation_invalidation() {
    let model = artifact();
    let circuit = circuit(&model, 1e-3);
    let context = ModelContext::new(37., 27.);
    let error = circuit.devices()[0]
        .validate_dc_inputs(&context)
        .unwrap_err()
        .to_string();
    assert!(error.contains("temperature changed"), "{error}");
}

#[test]
fn noise_driver_consumes_prepared_model_generators_and_ac_charge() {
    let mut circuit = source_driven_model(false);
    let request = AnalysisInput::with_arguments(
        AnalysisKind::Noise,
        ["v(out)", "v1", "lin", "2", "1k", "2k"],
    )
    .resolve()
    .unwrap();
    let plots = request
        .run_plots(&mut circuit, &AnalysisContext::default())
        .unwrap();
    assert_eq!(plots.len(), 2);
    for index in 0..plots[0].point_count() {
        let frequency = plots[0].value("frequency", index).unwrap().re;
        let actual = plots[0].value("onoise_spectrum", index).unwrap().re;
        let omega_c = 2. * std::f64::consts::PI * frequency * 1e-6;
        let total_g = 2e-3_f64;
        let density = 4. * ngspice_rs::devices::noise::BOLTZMANN * 300.15 * total_g
            / (total_g * total_g + omega_c * omega_c);
        assert!(
            (actual - density.sqrt()).abs() < density.sqrt() * 1e-9,
            "f={frequency:e}, observed={actual:e}, reference={:e}",
            density.sqrt()
        );
    }
}

#[test]
#[ignore = "opt-in release benchmark of real engine loads; never a timing assertion"]
fn prepared_kernel_load_benchmark_against_in_tree_behavioural_model() {
    use std::hint::black_box;
    use std::path::Path;
    use std::time::Instant;

    use ngspice_rs::netlist::{Parser, source::parse_deck_text};

    let native = source_driven_model(false);
    let source = "analytic behavioural reference
v1 in 0 dc 0 ac 1
r1 in out 1k
b1 out 0 i=0.001*v(out)+0.0002*v(out)^3
.end
";
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("prepared-model-benchmark.cir"),
            source,
        ))
        .unwrap();
    let behavioural = Circuit::from_netlist(&parsed).unwrap();
    let load_one = |circuit: &Circuit, iterations: usize| {
        let history = circuit.state_history().unwrap();
        let start = Instant::now();
        let n = circuit.unknown_count();
        let output = circuit
            .unknowns()
            .node_row(circuit.nodes().get("out").unwrap())
            .unwrap();
        let mut digest = 0.;
        for index in 0..iterations {
            let bias = 0.1 + 0.01 * (index % 50) as f64;
            let mut solution = Vector::zeros(n);
            solution.as_mut_slice()[output] = bias;
            let mut matrix = SparseMatrix::new(n, n);
            let mut rhs = Vector::zeros(n);
            let mut trial = history.trial();
            circuit
                .load(
                    &LoadRequest {
                        mode: AnalysisMode::OperatingPoint,
                        solution: &solution,
                        model_context: &ModelContext::default(),
                        integration: None,
                        history: &history,
                        forcing: None,
                    },
                    &mut matrix,
                    &mut rhs,
                    &mut trial,
                )
                .unwrap();
            digest += black_box(matrix.get(output, output) + rhs.as_slice()[output]);
        }
        (start.elapsed(), digest)
    };
    const REPEATS: usize = 10_000;
    let (a_time, a_digest) = load_one(&native, REPEATS);
    let (b_time, b_digest) = load_one(&behavioural, REPEATS);
    close(a_digest, b_digest);
    println!(
        "prepared_model_loads=10000 native_ns_per_load={:.1} behavioural_ns_per_load={:.1} digest={a_digest:.14e}",
        a_time.as_nanos() as f64 / REPEATS as f64,
        b_time.as_nanos() as f64 / REPEATS as f64,
    );
}

#[test]
fn declared_physical_zero_and_unavailable_channels_have_distinct_semantics() {
    let mut unavailable = descriptor(Capability::Unavailable);
    unavailable.noise = Capability::Zero;
    let model = Artifact::new(unavailable, PolynomialKernel).unwrap();
    let mut circuit = Circuit::new();
    let a = circuit.add_node("a").unwrap();
    circuit
        .add_device(Box::new(
            PreparedDevice::new(
                &model,
                "x1",
                vec![a, NodeId::GROUND],
                &[1e-3, 2e-4, 1e-6, 2e-7],
                InstanceTemperature::circuit(27.),
            )
            .unwrap(),
        ))
        .unwrap();
    circuit.finalize().unwrap();
    assert_eq!(circuit.state_len(), 0);
    let history = circuit.state_history().unwrap();
    let (_, _, _) = load(&circuit, &history, &[0.4], None);
    let bias = Vector::from_slice(&[0.4]);
    assert!(
        circuit
            .small_signal_system(&ModelContext::default(), &bias)
            .is_err()
    );
    let model_context = ModelContext::default().with_frequency(1.);
    let context = NoiseContext {
        model_context: &model_context,
        unknowns: circuit.unknowns(),
        bias: &bias,
        states: None,
        controls: &[],
    };
    let mut sources = Vec::new();
    let result = circuit.devices()[0].noise(&context, &mut sources).unwrap();
    assert!(matches!(
        result,
        ngspice_rs::devices::noise::DeviceNoise::Noiseless
    ));
    assert!(sources.is_empty());

    // The fixture returns nonzero charge; falsely labeling it physically
    // zero is detected as a model-contract violation.
    let model = Artifact::new(descriptor(Capability::Zero), PolynomialKernel).unwrap();
    let mut other = Circuit::new();
    let a = other.add_node("a").unwrap();
    other
        .add_device(Box::new(
            PreparedDevice::new(
                &model,
                "x2",
                vec![a, NodeId::GROUND],
                &[1e-3, 2e-4, 1e-6, 2e-7],
                InstanceTemperature::circuit(27.),
            )
            .unwrap(),
        ))
        .unwrap();
    other.finalize().unwrap();
    let n = other.unknown_count();
    let history = other.state_history().unwrap();
    let mut matrix = SparseMatrix::new(n, n);
    let mut rhs = Vector::zeros(n);
    let error = other
        .load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &Vector::from_slice(&[0.4]),
                model_context: &ModelContext::default(),
                integration: None,
                history: &history,
                forcing: None,
            },
            &mut matrix,
            &mut rhs,
            &mut history.trial(),
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("nonzero declared-zero storage"),
        "{error}"
    );
}

/// A genuine generalized DAE: two electrical KCL rows, a heat-balance
/// equation and a recovery-state equation. It is not a transistor surrogate:
/// it verifies the ABI's unit domains and circuit state ownership.
#[derive(Debug, Default)]
struct ElectrothermalKernel;

#[derive(Debug)]
struct ElectrothermalPrepared {
    ambient_kelvin: Real,
    g: Real,
    temperature_coefficient: Real,
    c: Real,
    coupling: Real,
    heat_conductance: Real,
    heat_capacitance: Real,
    relaxation_conductance: Real,
    recovery_time: Real,
}

impl Kernel for ElectrothermalKernel {
    type Prepared = ElectrothermalPrepared;

    fn prepare(
        &self,
        parameters: &[Real],
        temperature_kelvin: Real,
    ) -> model_kernel_api::Result<Self::Prepared> {
        let [g, k, c, coupling, gth, cth, gq, tau] = parameters else {
            return Err(Error::Domain(
                "electrothermal artifact requires eight parameter slots".into(),
            ));
        };
        if !temperature_kelvin.is_finite()
            || temperature_kelvin <= 0.
            || parameters.iter().any(|value| !value.is_finite())
            || *g <= 0.
            || *c < 0.
            || *gth <= 0.
            || *cth < 0.
            || *gq <= 0.
            || *tau <= 0.
        {
            return Err(Error::Domain(
                "electrothermal prepared model parameter domain".into(),
            ));
        }
        Ok(ElectrothermalPrepared {
            ambient_kelvin: temperature_kelvin,
            g: *g,
            temperature_coefficient: *k,
            c: *c,
            coupling: *coupling,
            heat_conductance: *gth,
            heat_capacitance: *cth,
            relaxation_conductance: *gq,
            recovery_time: *tau,
        })
    }

    fn evaluate(
        &self,
        prepared: &Self::Prepared,
        voltages: &[Real],
        output: &mut Point<'_>,
    ) -> model_kernel_api::Result<()> {
        let [positive, negative, thermal, recovery] = voltages else {
            return Err(Error::Domain("electrothermal input shape".into()));
        };
        let thermal = thermal - prepared.ambient_kelvin;
        let v = positive - negative;
        let g = prepared.g
            * (1.
                + prepared.temperature_coefficient * (prepared.ambient_kelvin - 300.15 + thermal));
        if g <= 0. {
            return Err(Error::Domain(
                "negative physical electrothermal conductivity".into(),
            ));
        }
        let dgdt = prepared.g * prepared.temperature_coefficient;
        let i = g * v;
        let flow = [
            i,
            -i,
            prepared.heat_conductance * thermal - i * v,
            prepared.relaxation_conductance * (recovery - v),
        ];
        output.current.copy_from_slice(&flow);
        output.conductance.copy_from_slice(&[
            g,
            -g,
            dgdt * v,
            0.,
            -g,
            g,
            -dgdt * v,
            0.,
            -2. * g * v,
            2. * g * v,
            prepared.heat_conductance - dgdt * v * v,
            0.,
            -prepared.relaxation_conductance,
            prepared.relaxation_conductance,
            0.,
            prepared.relaxation_conductance,
        ]);
        let q = prepared.c * v + prepared.coupling * recovery;
        output.charge.copy_from_slice(&[
            q,
            -q,
            prepared.heat_capacitance * thermal,
            prepared.recovery_time * prepared.relaxation_conductance * recovery,
        ]);
        output.capacitance.copy_from_slice(&[
            prepared.c,
            -prepared.c,
            0.,
            prepared.coupling,
            -prepared.c,
            prepared.c,
            0.,
            -prepared.coupling,
            0.,
            0.,
            prepared.heat_capacitance,
            0.,
            0.,
            0.,
            0.,
            prepared.recovery_time * prepared.relaxation_conductance,
        ]);
        Ok(())
    }
}

const ELECTROTHERMAL_PARAMETERS: [Real; 8] = [1e-3, 0.02, 1e-6, 2e-7, 0.1, 1e-3, 5e-4, 1e-6];

fn electrothermal_descriptor() -> Descriptor {
    Descriptor {
        family: "verified_electrothermal_dae".into(),
        fingerprint: [0x57; 16],
        abi: ABI_VERSION,
        equations: [
            ("p", EquationKind::Electrical),
            ("n", EquationKind::Electrical),
            ("temperature", EquationKind::Thermal),
            ("recovery", EquationKind::Internal),
        ]
        .into_iter()
        .map(|(name, kind)| Equation {
            name: name.into(),
            kind,
            binding: Binding::Terminal,
        })
        .collect(),
        parameters: [
            "conductance",
            "tempco",
            "capacitance",
            "coupling",
            "heat_conductance",
            "heat_capacitance",
            "recovery_conductance",
            "recovery_time",
        ]
        .into_iter()
        .zip(ELECTROTHERMAL_PARAMETERS)
        .map(|(name, default)| Parameter {
            name: name.into(),
            default,
            changes_topology: false,
        })
        .collect(),
        charge: Capability::Computed,
        noise: Capability::Zero,
        trial: model_kernel_api::trial::TrialCapabilities::ENGINE,
        taylor: model_kernel_api::taylor::TaylorCapability::Unavailable,
    }
}

#[test]
fn prepared_kernel_supports_thermal_and_internal_dae_rows_without_electrical_unit_mixing() {
    use model_kernel_api::PointBuffer;
    let descriptor = electrothermal_descriptor();
    let kernel = ElectrothermalKernel;
    let prepared = kernel.prepare(&ELECTROTHERMAL_PARAMETERS, 300.15).unwrap();
    let bias = [0.6, -0.2, 305.15, 0.1];
    let mut buffer = PointBuffer::new(4).unwrap();
    let mut output = buffer.point();
    output.reset(Capability::Computed);
    kernel.evaluate(&prepared, &bias, &mut output).unwrap();
    output.validate(&descriptor).unwrap();
    let electrical = output.current[0] + output.current[1];
    close(electrical, 0.);
    assert!(
        output.current[2] != 0.,
        "heat balance is not an electrical KCL current"
    );
    assert!(
        output.current[3] != 0.,
        "recovery state must admit a generalized residual"
    );
    assert!(output.charge[2] > 0.);
    assert!(output.charge[3] > 0.);

    let previous = output.conductance[0];
    let opposite = output.conductance[4];
    output.conductance[0] = previous + 0.1;
    output.conductance[4] = opposite - 0.1;
    // Column KCL still holds, but the electrical common-mode voltage
    // erroneously changes device current: the new row-gauge gate detects it.
    assert!(output.validate(&descriptor).is_err());
    output.conductance[0] = previous;
    output.conductance[4] = opposite;

    output.conductance[8] += 0.1;
    assert!(
        output.validate(&descriptor).is_err(),
        "thermal power must depend on electrical voltage differences"
    );
}

#[test]
fn prepared_electrothermal_recovery_equations_stamp_into_real_mna_rows() {
    let artifact = Artifact::new(electrothermal_descriptor(), ElectrothermalKernel).unwrap();
    let mut circuit = Circuit::new();
    let p = circuit.add_node("p").unwrap();
    let temperature = circuit.add_node("temperature").unwrap();
    let recovery = circuit.add_node("recovery").unwrap();
    circuit
        .add_device(Box::new(
            PreparedDevice::new(
                &artifact,
                "xthermal",
                vec![p, NodeId::GROUND, temperature, recovery],
                &ELECTROTHERMAL_PARAMETERS,
                InstanceTemperature::circuit(27.),
            )
            .unwrap(),
        ))
        .unwrap();
    circuit.finalize().unwrap();
    let history = circuit.state_history().unwrap();
    let solution = Vector::from_slice(&[0.5, 10., 0.2]);
    let mut mna = SparseMatrix::new(3, 3);
    let mut rhs = Vector::zeros(3);
    let mut trial = history.trial();
    circuit
        .load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &solution,
                model_context: &ModelContext::default(),
                integration: None,
                history: &history,
                forcing: None,
            },
            &mut mna,
            &mut rhs,
            &mut trial,
        )
        .unwrap();
    mna.fold_duplicates();
    let g = ELECTROTHERMAL_PARAMETERS[0] * 1.2;
    let dgdt = ELECTROTHERMAL_PARAMETERS[0] * 0.02;
    let p_row = circuit.unknowns().node_row(p).unwrap();
    let thermal_row = circuit.unknowns().node_row(temperature).unwrap();
    let recovery_row = circuit.unknowns().node_row(recovery).unwrap();
    close(mna.get(p_row, p_row), g);
    close(mna.get(p_row, thermal_row), dgdt * 0.5);
    close(mna.get(thermal_row, p_row), -2. * g * 0.5);
    close(
        mna.get(thermal_row, thermal_row),
        0.1 - dgdt * 0.5_f64.powi(2),
    );
    close(mna.get(recovery_row, p_row), -5e-4);
    close(mna.get(recovery_row, recovery_row), 5e-4);
    let initial = trial.values();
    assert_eq!(initial.len(), 8);
    close(initial[4], 1e-3 * 10.);
    close(initial[6], 1e-6 * 5e-4 * 0.2);

    let ac = circuit
        .small_signal_system_at(&ModelContext::default(), &solution, None)
        .unwrap();
    close(ac.e.get(thermal_row, thermal_row), 1e-3);
    close(ac.e.get(recovery_row, recovery_row), 1e-6 * 5e-4);
    close(ac.e.get(p_row, recovery_row), 2e-7);
    close(ac.e.get(p_row, thermal_row), 0.);

    // The generalized thermal energy and Qp recovery coordinates use the
    // same accepted/rejected history lifecycle as ordinary junction charge.
    let mut accepted = circuit.state_history().unwrap();
    let dc_solution = Vector::from_slice(&[0.5, 10., 0.2]);
    let (.., dc_trial) = load(&circuit, &accepted, dc_solution.as_slice(), None);
    circuit
        .accept_point(&dc_solution, None, &mut accepted, dc_trial)
        .unwrap();
    let stored = accepted.accepted(1).unwrap().to_vec();
    let step = StepHistory::new();
    let coeff = step
        .trial(IntegrationMethod::Trapezoidal, 1, 1e-6, DEFAULT_XMU)
        .unwrap();
    let transient_solution = [0.75, 10., 0.3];
    let (loaded, _, discarded) = load(&circuit, &accepted, &transient_solution, Some(&coeff));
    close(loaded.get(p_row, recovery_row), coeff.ag()[0] * 2e-7);
    close(
        loaded.get(thermal_row, thermal_row),
        0.1 - dgdt * 0.75_f64.powi(2) + coeff.ag()[0] * 1e-3,
    );
    close(
        loaded.get(recovery_row, recovery_row),
        5e-4 + coeff.ag()[0] * 1e-6 * 5e-4,
    );
    close(discarded.values()[4], 1e-3 * 10.);
    close(discarded.values()[6], 1e-6 * 5e-4 * 0.3);
    drop(discarded);
    assert_eq!(accepted.accepted(1).unwrap(), &stored);
    let (_, _, committed) = load(&circuit, &accepted, &transient_solution, Some(&coeff));
    circuit
        .accept_point(
            &Vector::from_slice(&transient_solution),
            Some(1e-6),
            &mut accepted,
            committed,
        )
        .unwrap();
    assert_ne!(accepted.accepted(1).unwrap(), &stored);
}

#[test]
fn prepared_equation_names_and_electrical_domain_are_structural_schema() {
    let model = Artifact::new(electrothermal_descriptor(), ElectrothermalKernel).unwrap();
    assert!(
        PreparedDevice::new(
            &model,
            "bad",
            vec![NodeId::GROUND; 4],
            &ELECTROTHERMAL_PARAMETERS,
            InstanceTemperature::circuit(-273.15)
        )
        .is_err()
    );
    let mut descriptor = electrothermal_descriptor();
    descriptor.equations[2].name = "p".into();
    assert!(Artifact::new(descriptor.clone(), ElectrothermalKernel).is_err());
    descriptor.equations[2].name = "temperature".into();
    descriptor.equations[1].kind = EquationKind::Internal;
    assert!(
        Artifact::new(descriptor, ElectrothermalKernel).is_err(),
        "at least two electrical terminals required to close KCL"
    );
}

fn vdmos_fixture() -> Arc<Artifact<model_kernel_vdmos::artifact::VdmosKernel>> {
    use model_kernel_vdmos::{Polarity, artifact::artifact, compile_ngspice};
    let model = compile_ngspice(
        Polarity::N,
        [
            ("vto", 3.),
            ("kp", 2.),
            ("rd", 0.02),
            ("rg", 1.),
            ("rs", 0.01),
            ("rb", 0.015),
            ("cgs", 1e-9),
            ("cgdmin", 2e-11),
            ("cgdmax", 8e-10),
            ("cjo", 1e-10),
            ("kf", 1e-15),
        ],
    )
    .unwrap();
    artifact(Arc::new(model), false, [0x71; 16]).unwrap()
}

fn vdmos_circuit(
    model: &Arc<Artifact<model_kernel_vdmos::artifact::VdmosKernel>>,
    lanes: usize,
) -> Circuit {
    vdmos_circuit_with(model, lanes, false, false)
}

fn vdmos_circuit_with(
    model: &Arc<Artifact<model_kernel_vdmos::artifact::VdmosKernel>>,
    lanes: usize,
    step: bool,
    scalar: bool,
) -> Circuit {
    let mut circuit = Circuit::new();
    for lane in 0..lanes {
        let drain = circuit.add_node(&format!("d{lane}")).unwrap();
        let gate = circuit.add_node(&format!("g{lane}")).unwrap();
        for (name, node, dc, ac) in [
            (format!("vd{lane}"), drain, 0.25, 0.),
            (format!("vg{lane}"), gate, 4., 1.),
        ] {
            circuit
                .add_device(Box::new(
                    IndependentSource::new(
                        name,
                        [node, NodeId::GROUND],
                        true,
                        dc,
                        Complex::real(ac),
                        if step && dc == 4. {
                            Waveform::Step {
                                before: 4.,
                                after: 4.5,
                                time: 3e-6,
                            }
                        } else {
                            Waveform::Constant(dc)
                        },
                    )
                    .unwrap(),
                ))
                .unwrap();
        }
        let instance_model = if scalar {
            vdmos_fixture()
        } else {
            Arc::clone(model)
        };
        PreparedDevice::bind(
            &mut circuit,
            &instance_model,
            &format!("m{lane}"),
            &[drain, gate, NodeId::GROUND],
            &[1. + lane as f64],
            InstanceTemperature::circuit(27.),
        )
        .unwrap();
    }
    circuit.finalize().unwrap();
    circuit
}

#[test]
fn actual_vdmos_artifact_binds_hidden_rows_solves_and_reprepares_candidates() {
    use ngspice_rs::analysis::{bias::solve_dc, newton::NewtonOptions};
    use ngspice_rs::devices::sweep::ParameterKey;
    let model = vdmos_fixture();
    let mut circuit = vdmos_circuit(&model, 3);
    let hidden = model
        .descriptor()
        .equations
        .iter()
        .filter(|e| e.binding == Binding::Hidden)
        .count();
    assert!(hidden >= 3);
    assert_eq!(circuit.unknown_count(), 3 * (hidden + 4));
    assert_eq!(
        circuit.state_len(),
        3 * (3 * model.descriptor().equations.len() + 1)
    );
    let context = ModelContext::default();
    let options = NewtonOptions {
        reltol: 1e-10,
        abstol: 1e-13,
        vntol: 1e-10,
        ..NewtonOptions::default()
    };
    let solve =
        |context: &ModelContext| solve_dc(&circuit, context, &options, &[], None, None).unwrap();
    let baseline = solve(&context);
    // Compare each real engine operating point, including every private
    // series node, with the physical residual of the same public artifact.
    let current_at = |lane: usize, solution: &Vector| {
        let device = &circuit.devices()[3 * lane + 2];
        let coordinates: Vec<_> = device
            .terminals()
            .iter()
            .map(|&node| {
                circuit
                    .unknowns()
                    .node_row(node)
                    .and_then(|row| solution.get(row))
                    .unwrap_or(0.)
            })
            .collect();
        let prepared = model.prepare(&[1. + lane as f64], 300.15).unwrap();
        let mut buffer = model_kernel_api::PointBuffer::new(coordinates.len()).unwrap();
        model
            .evaluate(&prepared, &coordinates, &mut buffer.point())
            .unwrap();
        let point = buffer.point();
        for (row, equation) in model.descriptor().equations.iter().enumerate() {
            if equation.binding == Binding::Hidden {
                assert!(
                    point.current[row].abs() < 1e-9,
                    "{}: {}",
                    equation.name,
                    point.current[row]
                );
            }
        }
        let drain = model
            .descriptor()
            .equations
            .iter()
            .position(|e| e.name == "drain")
            .unwrap();
        let branch = circuit.branch_rows(3 * lane).unwrap().start;
        close(point.current[drain], -solution.get(branch).unwrap());
        point.current[drain]
    };
    let first = current_at(0, &baseline.values);
    let second = current_at(1, &baseline.values);
    let third = current_at(2, &baseline.values);
    assert!(first > 0. && second > first && third > second);
    let target = circuit.bind_instance_target("m1", "MULTIPLIER").unwrap();
    assert_eq!(target.parameter(), ParameterKey::Prepared(0));
    let changed = context
        .with_instance_override(target.at(4.).unwrap())
        .unwrap();
    let candidate = solve(&changed);
    let branch = |lane: usize| circuit.branch_rows(3 * lane).unwrap().start;
    close(
        candidate.values.get(branch(0)).unwrap(),
        baseline.values.get(branch(0)).unwrap(),
    );
    close(
        candidate.values.get(branch(2)).unwrap(),
        baseline.values.get(branch(2)).unwrap(),
    );
    assert!(-candidate.values.get(branch(1)).unwrap() > third);
    let hot = solve(&ModelContext::new(57., 27.));
    assert!(hot.values.is_finite());
    assert_ne!(hot.values.get(branch(0)), baseline.values.get(branch(0)));
    let again = solve(&context);
    close(
        again.values.get(branch(1)).unwrap(),
        baseline.values.get(branch(1)).unwrap(),
    );

    let ac = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "3", "1k", "1meg"])
        .resolve()
        .unwrap()
        .run_plots(&mut circuit, &AnalysisContext::default())
        .unwrap();
    assert_eq!(ac[0].point_count(), 3);
    for row in 0..3 {
        let current = ac[0].value("i(vg0)", row).unwrap();
        assert!(current.is_finite() && current.im.abs() > 0.);
    }
    let noise = AnalysisInput::with_arguments(
        AnalysisKind::Noise,
        ["v(d0)", "vg0", "lin", "3", "1k", "1meg"],
    )
    .resolve()
    .unwrap()
    .run_plots(&mut circuit, &AnalysisContext::default())
    .unwrap();
    assert_eq!(noise[0].point_count(), 3);
}

#[test]
fn prepared_hidden_node_transactions_rollback_on_invalid_or_duplicate_instance() {
    let model = vdmos_fixture();
    let mut circuit = Circuit::new();
    let drain = circuit.add_node("d").unwrap();
    let gate = circuit.add_node("g").unwrap();
    let ports = [drain, gate, NodeId::GROUND];
    let initial = circuit.nodes().len();
    assert!(
        PreparedDevice::bind(
            &mut circuit,
            &model,
            "m1",
            &ports,
            &[-1.],
            InstanceTemperature::circuit(27.)
        )
        .is_err()
    );
    assert_eq!(circuit.nodes().len(), initial);
    assert!(circuit.devices().is_empty());
    PreparedDevice::bind(
        &mut circuit,
        &model,
        "m1",
        &ports,
        &[1.],
        InstanceTemperature::circuit(27.),
    )
    .unwrap();
    let allocated = circuit.nodes().len();
    assert!(allocated > initial);
    assert!(
        PreparedDevice::bind(
            &mut circuit,
            &model,
            "m1",
            &ports,
            &[1.],
            InstanceTemperature::circuit(27.)
        )
        .is_err()
    );
    assert_eq!(circuit.nodes().len(), allocated);
    assert_eq!(circuit.devices().len(), 1);
}

const VDMOS_OP_DECK: &str = include_str!("fixtures/prepared_vdmos/op.cir");
const VDMOS_OP_REFERENCE: &str = include_str!("fixtures/prepared_vdmos/op.expected");

#[test]
fn actual_vdmos_operating_point_matches_c_reference() {
    check_vdmos_reference(VDMOS_OP_REFERENCE);
}

#[test]
#[ignore = "requires independent ngspice 47 in the test runtime"]
fn actual_vdmos_live_reference() {
    use ngspice_conformance::ngspice::{Ngspice, Scratch};
    let scratch = Scratch::new("vdmos-engine").unwrap();
    std::fs::write(scratch.path().join("ref.cir"), VDMOS_OP_DECK).unwrap();
    let result = Ngspice::pinned()
        .unwrap()
        .run(scratch.path(), &["-b", "ref.cir"])
        .unwrap();
    result.require_success("VDMOS engine").unwrap();
    check_vdmos_reference(&String::from_utf8_lossy(&result.stdout));
}

fn check_vdmos_reference(output: &str) {
    use ngspice_rs::analysis::{bias::solve_dc, newton::NewtonOptions};
    let model = vdmos_fixture();
    let circuit = vdmos_circuit(&model, 3);
    let options = NewtonOptions {
        reltol: 1e-10,
        abstol: 1e-13,
        vntol: 1e-10,
        ..NewtonOptions::default()
    };
    let solution = solve_dc(
        &circuit,
        &ModelContext::default(),
        &options,
        &[],
        None,
        None,
    )
    .unwrap()
    .values;
    for lane in 0..3 {
        let name = format!("i(vd{lane})");
        let reference: f64 = output
            .lines()
            .filter_map(|line| line.split_once('='))
            .find(|(left, _)| left.trim() == name)
            .unwrap_or_else(|| panic!("missing {name}: {output}"))
            .1
            .trim()
            .parse()
            .unwrap();
        let actual = solution
            .get(circuit.branch_rows(lane * 3).unwrap().start)
            .unwrap();
        assert!(
            (actual - reference).abs() <= 1e-6 * reference.abs() + 1e-10,
            "{name}: {actual:e} vs {reference:e}"
        );
    }
}

#[test]
fn vdmos_tiled_transient_matches_independent_scalar_instances() {
    let model = vdmos_fixture();
    let mut tiled = vdmos_circuit_with(&model, 3, true, false);
    let mut scalar = vdmos_circuit_with(&model, 3, true, true);
    let request =
        AnalysisInput::with_arguments(AnalysisKind::Transient, ["100n", "6u", "0", "50n"])
            .resolve()
            .unwrap();
    let a = request
        .run(&mut tiled, &AnalysisContext::default())
        .unwrap();
    let b = request
        .run(&mut scalar, &AnalysisContext::default())
        .unwrap();
    assert_eq!(a.point_count(), b.point_count());
    assert!(a.point_count() > 5);
    let mut first = None;
    let mut last = None;
    for row in 0..a.point_count() {
        for name in [
            "time", "i(vd0)", "i(vd1)", "i(vd2)", "i(vg0)", "i(vg1)", "i(vg2)",
        ] {
            let x = a.value(name, row).unwrap().re;
            let y = b.value(name, row).unwrap().re;
            assert!(x.is_finite() && y.is_finite());
            assert_eq!(x.to_bits(), y.to_bits(), "{name} at {row}");
        }
        let time = a.value("time", row).unwrap().re;
        let drain = a.value("i(vd0)", row).unwrap().re;
        if time < 2e-6 {
            first = Some(drain);
        }
        if time > 5e-6 {
            last = Some(drain);
        }
    }
    assert!(last.unwrap().abs() > first.unwrap().abs());
}

#[test]
#[ignore = "opt-in real engine load benchmark"]
fn vdmos_tile_engine_benchmark() {
    use std::hint::black_box;
    use std::time::Instant;
    let model = vdmos_fixture();
    let lanes = 128;
    let tiled = vdmos_circuit_with(&model, lanes, false, false);
    let scalar = vdmos_circuit_with(&model, lanes, false, true);
    let measurement = |circuit: &Circuit| {
        let run = circuit.prepare_run();
        let history = circuit.state_history().unwrap();
        let n = circuit.unknown_count();
        let context = ModelContext::default();
        let mut solution = Vector::zeros(n);
        for lane in 0..lanes {
            for (node, value) in [(format!("d{lane}"), 0.25), (format!("g{lane}"), 4.)] {
                let row = circuit
                    .unknowns()
                    .node_row(circuit.nodes().get(&node).unwrap())
                    .unwrap();
                solution.as_mut_slice()[row] = value;
            }
        }
        let mut matrix = circuit.new_matrix().unwrap();
        let mut rhs = Vector::zeros(n);
        let mut trial = history.trial();
        // Warm the fixed-stamp and prepared tile caches before timing.
        let request = LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &solution,
            model_context: &context,
            integration: None,
            history: &history,
            forcing: None,
        };
        run.load(&request, &mut matrix, &mut rhs, &mut trial)
            .unwrap();
        let start = Instant::now();
        for _ in 0..300 {
            matrix.clear();
            rhs.clear();
            run.load(&request, &mut matrix, &mut rhs, &mut trial)
                .unwrap();
            black_box((&matrix, &rhs));
        }
        let elapsed = start.elapsed().as_nanos() as f64 / 300.;
        let snapshot: Vec<_> = matrix
            .bound_values()
            .unwrap()
            .1
            .iter()
            .chain(rhs.as_slice())
            .map(|v| v.to_bits())
            .collect();
        (elapsed, snapshot)
    };
    let mut tiles = Vec::new();
    let mut scalars = Vec::new();
    for round in 0..12 {
        let ((t, a), (s, b)) = if round % 2 == 0 {
            (measurement(&tiled), measurement(&scalar))
        } else {
            let b = measurement(&scalar);
            (measurement(&tiled), b)
        };
        assert_eq!(a, b);
        if round > 0 {
            tiles.push(t);
            scalars.push(s);
        }
    }
    tiles.sort_by(f64::total_cmp);
    scalars.sort_by(f64::total_cmp);
    let (tile, scalar) = (tiles[5], scalars[5]);
    println!(
        "VDMOS {lanes} lanes: tiled {tile:.0} ns/load, scalar {scalar:.0} ns/load, ratio {:.4}",
        tile / scalar
    );
}

#[derive(Debug)]
struct CountingKernel {
    preparations: Arc<std::sync::atomic::AtomicUsize>,
}

impl Kernel for CountingKernel {
    type Prepared = Polynomial;

    fn prepare(&self, values: &[Real], ambient: Real) -> model_kernel_api::Result<Polynomial> {
        self.preparations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        PolynomialKernel.prepare(values, ambient)
    }

    fn evaluate(
        &self,
        prepared: &Polynomial,
        coordinates: &[Real],
        out: &mut Point<'_>,
    ) -> model_kernel_api::Result<()> {
        PolynomialKernel.evaluate(prepared, coordinates, out)
    }
    fn evaluate_noise(
        &self,
        prepared: &Polynomial,
        coordinates: &[Real],
        frequency: Real,
        sink: &mut dyn model_kernel_api::noise::Sink,
    ) -> model_kernel_api::Result<()> {
        PolynomialKernel.evaluate_noise(prepared, coordinates, frequency, sink)
    }
}

fn counted_circuit() -> (Circuit, Arc<std::sync::atomic::AtomicUsize>) {
    use ngspice_rs::devices::prepared_model::AmbientTemperature;
    use std::sync::atomic::AtomicUsize;
    let preparations = Arc::new(AtomicUsize::new(0));
    let model = Artifact::new(
        descriptor(Capability::Computed),
        CountingKernel {
            preparations: Arc::clone(&preparations),
        },
    )
    .unwrap();
    let mut circuit = Circuit::new();
    for lane in 0..129 {
        let node = circuit.add_node(&format!("n{lane}")).unwrap();
        let temperature = if lane == 128 {
            InstanceTemperature {
                circuit_celsius: 27.,
                ambient: AmbientTemperature::AbsoluteKelvin(300.15),
            }
        } else {
            InstanceTemperature::circuit(27.)
        };
        circuit
            .add_device(Box::new(
                PreparedDevice::new(
                    &model,
                    format!("m{lane}"),
                    vec![node, NodeId::GROUND],
                    &[1e-3, 2e-4, 1e-6, 2e-7],
                    temperature,
                )
                .unwrap(),
            ))
            .unwrap();
    }
    circuit.finalize().unwrap();
    (circuit, preparations)
}

fn counted_load(
    run: &ngspice_rs::devices::PreparedRun<'_>,
    context: &ModelContext,
) -> ngspice_rs::primitives::SpiceResult<Vec<u64>> {
    let n = run.circuit().unknown_count();
    let history = run.circuit().state_history()?;
    let mut matrix = SparseMatrix::new(n, n);
    let mut rhs = Vector::zeros(n);
    let mut trial = history.trial();
    run.load(
        &LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &Vector::from_slice(&vec![0.25; n]),
            model_context: context,
            integration: None,
            history: &history,
            forcing: None,
        },
        &mut matrix,
        &mut rhs,
        &mut trial,
    )?;
    Ok((0..n).map(|row| matrix.get(row, row).to_bits()).collect())
}

#[test]
fn prepared_run_reuses_unaffected_tiles_across_context_changes() {
    use ngspice_rs::devices::noise::NoisePlan;
    use std::{sync::atomic::Ordering, time::Instant};
    let start = Instant::now();
    let (circuit, preparations) = counted_circuit();
    let initial_bind = start.elapsed();
    let run = circuit.prepare_run();
    let count = || preparations.load(Ordering::Relaxed);
    assert_eq!(count(), 129);
    let context = ModelContext::default();
    let start = Instant::now();
    run.validate_context(&context).unwrap();
    let initial_transfer = start.elapsed();
    assert_eq!(count(), 129);
    let original = counted_load(&run, &context).unwrap();
    assert_eq!(counted_load(&run, &context).unwrap(), original);
    assert_eq!(
        counted_load(&run, &context.with_frequency(1000.)).unwrap(),
        original
    );
    assert_eq!(count(), 129);

    let target = circuit.bind_instance_target("m3", "g").unwrap();
    let second = circuit.bind_instance_target("m3", "c").unwrap();
    let changed = context
        .with_instance_override(target.at(2e-3).unwrap())
        .unwrap()
        .with_instance_override(second.at(2e-6).unwrap())
        .unwrap();
    let start = Instant::now();
    run.validate_context(&changed).unwrap();
    let one_lane = start.elapsed();
    assert_eq!(count(), 130, "two setters on one lane must prepare once");
    let new = counted_load(&run, &changed).unwrap();
    let row = circuit
        .unknowns()
        .node_row(circuit.nodes().get("n3").unwrap())
        .unwrap();
    assert_eq!(new.iter().zip(&original).filter(|(a, b)| a != b).count(), 1);
    assert!(f64::from_bits(new[row]) > f64::from_bits(original[row]));
    let bias = Vector::from_slice(&vec![0.25; circuit.unknown_count()]);
    let linear = run.small_signal_system_at(&changed, &bias, None).unwrap();
    for (row, bits) in new.iter().enumerate() {
        assert_eq!(linear.a.get(row, row).to_bits(), *bits);
    }
    let noise = NoisePlan::bind(&run, &changed.with_frequency(1000.), &bias, None).unwrap();
    assert_eq!(noise.source_count(), 129);
    assert_eq!(count(), 130, "AC and noise use the same physical lane");
    assert_eq!(counted_load(&run, &context).unwrap(), original);
    assert_eq!(count(), 131);

    let warm = ModelContext::new(37., 27.);
    assert_eq!(counted_load(&run, &warm).unwrap(), original);
    assert_eq!(count(), 131 + 128, "fixed ambient singleton is unchanged");
    assert_eq!(counted_load(&run, &context).unwrap(), original);
    assert_eq!(count(), 131 + 256);
    println!(
        "129-lane bind {initial_bind:?}; initial ownership transfer {initial_transfer:?}; one changed lane {one_lane:?}"
    );
}

#[test]
fn prepared_tiles_roll_back_all_groups_and_keep_live_runs_independent() {
    use std::sync::atomic::Ordering;
    let (circuit, preparations) = counted_circuit();
    let a = circuit.prepare_run();
    let b = circuit.prepare_run();
    let context = ModelContext::default();
    let original = counted_load(&a, &context).unwrap();
    let changed = context
        .with_instance_override(
            circuit
                .bind_instance_target("m3", "g")
                .unwrap()
                .at(2e-3)
                .unwrap(),
        )
        .unwrap();
    let b_point = counted_load(&b, &changed).unwrap();
    assert_ne!(b_point, original);
    assert_eq!(counted_load(&a, &context).unwrap(), original);
    assert_eq!(preparations.load(Ordering::Relaxed), 258);
    let invalid = changed
        .with_instance_override(
            circuit
                .bind_instance_target("m80", "g")
                .unwrap()
                .at(-1.)
                .unwrap(),
        )
        .unwrap();
    assert!(counted_load(&a, &invalid).is_err());
    let after_failure = preparations.load(Ordering::Relaxed);
    assert_eq!(counted_load(&a, &context).unwrap(), original);
    assert_eq!(counted_load(&b, &changed).unwrap(), b_point);
    assert_eq!(preparations.load(Ordering::Relaxed), after_failure);
    assert_eq!(counted_load(&a, &changed).unwrap(), b_point);
    assert_eq!(preparations.load(Ordering::Relaxed), after_failure + 1);
    assert_eq!(counted_load(&a, &context).unwrap(), original);
    assert_eq!(preparations.load(Ordering::Relaxed), after_failure + 2);
}

#[derive(Debug, Default)]
struct TrialPolynomial;

impl Kernel for TrialPolynomial {
    type Prepared = Polynomial;

    fn prepare(
        &self,
        parameters: &[Real],
        temperature: Real,
    ) -> model_kernel_api::Result<Polynomial> {
        PolynomialKernel.prepare(parameters, temperature)
    }

    fn evaluate(
        &self,
        prepared: &Polynomial,
        coordinates: &[Real],
        point: &mut Point<'_>,
    ) -> model_kernel_api::Result<()> {
        PolynomialKernel.evaluate(prepared, coordinates, point)
    }

    fn initialize_trial(
        &self,
        _: &Polynomial,
        coordinates: &mut [Real],
    ) -> model_kernel_api::Result<()> {
        coordinates[0] = coordinates[1] + 0.1;
        Ok(())
    }

    fn limit_trial(
        &self,
        _: &Polynomial,
        reference: model_kernel_api::trial::TrialReference<'_>,
        coordinates: &mut [Real],
    ) -> model_kernel_api::Result<()> {
        use model_kernel_api::trial::TrialReference;
        let limit = match reference {
            TrialReference::PreviousNewton(_) => 0.25,
            TrialReference::AcceptedStep(_) => 0.125,
        };
        let reference = reference.coordinates();
        let last = reference[0] - reference[1];
        let raw = coordinates[0] - coordinates[1];
        coordinates[0] = coordinates[1] + raw.clamp(last - limit, last + limit);
        Ok(())
    }

    fn trial_convergence(
        &self,
        _: &Polynomial,
        input: model_kernel_api::trial::ConvergenceInput<'_>,
    ) -> model_kernel_api::Result<model_kernel_api::trial::Convergence> {
        let old = input.previous;
        let next = input.candidate;
        let delta =
            (next.coordinates[0] - next.coordinates[1]) - (old.coordinates[0] - old.coordinates[1]);
        input.compare_residual(
            0,
            old.jacobian[0].mul_add(delta, old.residual[0]),
            next.residual[0],
        )
    }
}

fn trial_artifact() -> Arc<Artifact<TrialPolynomial>> {
    use model_kernel_api::trial::{TrialCapabilities, TrialPolicy};
    let mut desc = descriptor(Capability::Computed);
    desc.trial = TrialCapabilities {
        initialization: TrialPolicy::Kernel,
        limiting: TrialPolicy::Kernel,
        convergence: TrialPolicy::Kernel,
    };
    desc.noise = Capability::Zero;
    Artifact::new(desc, TrialPolynomial).unwrap()
}

fn trial_circuit(count: usize) -> Circuit {
    let artifact = trial_artifact();
    let mut circuit = Circuit::new();
    for index in 0..count {
        let node = circuit.add_node(&format!("p{index}")).unwrap();
        PreparedDevice::bind(
            &mut circuit,
            &artifact,
            &format!("x{index}"),
            &[node, NodeId::GROUND],
            &[1e-3, 2e-4, 1e-6, 2e-7],
            InstanceTemperature::circuit(27.),
        )
        .unwrap();
    }
    circuit.finalize().unwrap();
    circuit
}

fn trial_load(
    circuit: &Circuit,
    history: &ngspice_rs::devices::StateHistory,
    proposed: &[Real],
    mut trial: ngspice_rs::devices::TrialState,
    integration: Option<&Coefficients>,
) -> (SparseMatrix, Vector, ngspice_rs::devices::TrialState) {
    let n = circuit.unknown_count();
    let mut matrix = SparseMatrix::new(n, n);
    let mut rhs = Vector::zeros(n);
    let mode = integration.map_or(AnalysisMode::OperatingPoint, |coeff| {
        AnalysisMode::Transient {
            time: coeff.dt(),
            dt: coeff.dt(),
        }
    });
    circuit
        .load(
            &LoadRequest {
                mode,
                solution: &Vector::from_slice(proposed),
                model_context: &ModelContext::default(),
                integration,
                history,
                forcing: None,
            },
            &mut matrix,
            &mut rhs,
            &mut trial,
        )
        .unwrap();
    matrix.fold_duplicates();
    (matrix, rhs, trial)
}

#[test]
fn kernel_trial_policy_preserves_reference_provenance_and_exact_acceptance() {
    use ngspice_rs::devices::{ConvergenceRequest, IterationPhase};
    // One instance uses the scalar path; three share the same homogeneous tile.
    for count in [1, 3] {
        let circuit = trial_circuit(count);
        let mut history = circuit.state_history().unwrap();
        let proposed = vec![2.; count];
        let (matrix, rhs, cold) = trial_load(
            &circuit,
            &history,
            &proposed,
            history.trial_in(IterationPhase::Junction, None).unwrap(),
            None,
        );
        assert!(cold.is_nonconvergent());
        assert!(
            circuit
                .accept_point(
                    &Vector::from_slice(&proposed),
                    None,
                    &mut history,
                    cold.clone()
                )
                .is_err()
        );
        assert_eq!(history.depth(), 0);
        let x: Real = 0.1;
        let g = 1e-3 + 6e-4 * x * x;
        let i = 1e-3 * x + 2e-4 * x.powi(3);
        for lane in 0..count {
            close(cold.values()[7 * lane + 4], x);
            close(matrix.get(lane, lane), g);
            close(rhs.as_slice()[lane], g * x - i);
        }
        let (_, _, limited) = trial_load(
            &circuit,
            &history,
            &proposed,
            history.trial_in(IterationPhase::Fix, Some(&cold)).unwrap(),
            None,
        );
        assert!(limited.is_nonconvergent());
        for lane in 0..count {
            close(limited.values()[7 * lane + 4], 0.35);
        }
        let (_, _, exact) = trial_load(&circuit, &history, &proposed, history.trial(), None);
        assert!(!exact.is_nonconvergent());
        for lane in 0..count {
            close(exact.values()[7 * lane + 4], 2.);
        }
        circuit
            .accept_point(&Vector::from_slice(&proposed), None, &mut history, exact)
            .unwrap();
        let accepted = history.accepted(1).unwrap().to_vec();
        let coeff = StepHistory::new()
            .trial(IntegrationMethod::Trapezoidal, 1, 1e-6, DEFAULT_XMU)
            .unwrap();
        let predictor = history.trial_in(IterationPhase::Predict, None).unwrap();
        let (matrix, rhs, rejected) = trial_load(
            &circuit,
            &history,
            &vec![3.; count],
            predictor,
            Some(&coeff),
        );
        assert!(rejected.is_nonconvergent());
        assert!(
            circuit
                .accept_point(
                    &Vector::from_slice(&vec![3.; count]),
                    Some(1e-6),
                    &mut history,
                    rejected.clone()
                )
                .is_err()
        );
        assert_eq!(history.accepted(1).unwrap(), accepted);
        for lane in 0..count {
            let x: Real = 2.125;
            let stored = 1e-6 * x + 2e-7 * x.powi(3);
            let accepted_charge = 1e-6 * 2. + 2e-7 * 8.;
            let current = 1e-3 * x + 2e-4 * x.powi(3) + (stored - accepted_charge) / 1e-6;
            let jac = 1e-3 + 6e-4 * x * x + (1e-6 + 6e-7 * x * x) / 1e-6;
            close(rejected.values()[7 * lane + 4], x);
            close(rejected.values()[7 * lane], stored);
            close(matrix.get(lane, lane), jac);
            close(rhs.as_slice()[lane], jac * x - current);
        }
        assert_eq!(history.accepted(1).unwrap(), accepted);
        let explicit = history
            .trial_in(IterationPhase::Predict, None)
            .unwrap()
            .with_initial_conditions(true);
        let (_, _, explicit) =
            trial_load(&circuit, &history, &vec![3.; count], explicit, Some(&coeff));
        assert!(!explicit.is_nonconvergent());
        for lane in 0..count {
            close(explicit.values()[7 * lane + 4], 3.);
        }
        let (_, _, last_exact) = trial_load(&circuit, &history, &proposed, history.trial(), None);
        let previous = Vector::from_slice(&proposed);
        let changed = Vector::from_slice(&vec![2.2; count]);
        let context = ModelContext::default();
        let request = ConvergenceRequest {
            solution: &changed,
            previous: &previous,
            trial: &last_exact,
            history: &history,
            model_context: &context,
            time: 0.,
            integration: None,
            reltol: 1e-8,
            abstol: 1e-12,
            vntol: 1e-6,
        };
        assert!(circuit.convergence_test(&request).unwrap());
        assert!(
            !circuit
                .convergence_test(&ConvergenceRequest {
                    solution: &previous,
                    ..request
                })
                .unwrap()
        );
    }
}

#[derive(Debug)]
struct TrialElectrothermal;

impl Kernel for TrialElectrothermal {
    type Prepared = ElectrothermalPrepared;

    fn prepare(
        &self,
        parameters: &[Real],
        temperature: Real,
    ) -> model_kernel_api::Result<Self::Prepared> {
        ElectrothermalKernel.prepare(parameters, temperature)
    }

    fn evaluate(
        &self,
        prepared: &Self::Prepared,
        coordinates: &[Real],
        point: &mut Point<'_>,
    ) -> model_kernel_api::Result<()> {
        ElectrothermalKernel.evaluate(prepared, coordinates, point)
    }

    fn initialize_trial(
        &self,
        prepared: &Self::Prepared,
        coordinates: &mut [Real],
    ) -> model_kernel_api::Result<()> {
        coordinates[2] = prepared.ambient_kelvin + 5.;
        coordinates[3] = 0.125;
        Ok(())
    }

    fn limit_trial(
        &self,
        _: &Self::Prepared,
        reference: model_kernel_api::trial::TrialReference<'_>,
        coordinates: &mut [Real],
    ) -> model_kernel_api::Result<()> {
        let reference = reference.coordinates();
        coordinates[2] = coordinates[2].clamp(reference[2] - 2., reference[2] + 2.);
        coordinates[3] = coordinates[3].clamp(reference[3] - 0.1, reference[3] + 0.1);
        Ok(())
    }
}

#[test]
fn kernel_trial_policy_stamps_limited_thermal_and_internal_coordinates() {
    use model_kernel_api::trial::{TrialCapabilities, TrialPolicy};
    use ngspice_rs::devices::IterationPhase;

    let mut descriptor = electrothermal_descriptor();
    descriptor.trial = TrialCapabilities {
        initialization: TrialPolicy::Kernel,
        limiting: TrialPolicy::Kernel,
        convergence: TrialPolicy::Engine,
    };
    let artifact = Artifact::new(descriptor, TrialElectrothermal).unwrap();
    // A one-lane instance and a homogeneous tile must stamp the same
    // affine residuals even when local thermal and internal coordinates
    // differ from the circuit's proposed MNA coordinates.
    for count in [1, 2] {
        let mut circuit = Circuit::new();
        let mut nodes = Vec::new();
        for index in 0..count {
            let p = circuit.add_node(&format!("p{index}")).unwrap();
            let t = circuit.add_node(&format!("t{index}")).unwrap();
            let q = circuit.add_node(&format!("q{index}")).unwrap();
            nodes.push([p, t, q]);
            PreparedDevice::bind(
                &mut circuit,
                &artifact,
                &format!("x{index}"),
                &[p, NodeId::GROUND, t, q],
                &ELECTROTHERMAL_PARAMETERS,
                InstanceTemperature::circuit(27.),
            )
            .unwrap();
        }
        circuit.finalize().unwrap();
        let n = circuit.unknown_count();
        let rows: Vec<_> = nodes
            .iter()
            .map(|nodes| nodes.map(|node| circuit.unknowns().node_row(node).unwrap()))
            .collect();
        let mut solution = Vector::zeros(n);
        for [p, t, q] in &rows {
            solution.set(*p, 0.5).unwrap();
            solution.set(*t, 10.).unwrap();
            solution.set(*q, 0.8).unwrap();
        }
        let history = circuit.state_history().unwrap();
        let (matrix, rhs, cold) = trial_load(
            &circuit,
            &history,
            solution.as_slice(),
            history.trial_in(IterationPhase::Junction, None).unwrap(),
            None,
        );
        assert!(cold.is_nonconvergent());
        let prepared = artifact
            .prepare(&ELECTROTHERMAL_PARAMETERS, 300.15)
            .unwrap();
        let mut buffer = model_kernel_api::PointBuffer::new(4).unwrap();
        let mut point = buffer.point();
        let actual = [0.5, 0., 305.15, 0.125];
        artifact.evaluate(&prepared, &actual, &mut point).unwrap();
        let mna = [0.5, 0., 5., 0.125];
        for (lane, [p, t, q]) in rows.iter().enumerate() {
            let local_rows = [Some(*p), None, Some(*t), Some(*q)];
            for (i, row) in local_rows.into_iter().enumerate() {
                let Some(row) = row else {
                    continue;
                };
                let expected_rhs = (0..4).fold(-point.current[i], |r, j| {
                    point.conductance[4 * i + j].mul_add(mna[j], r)
                });
                close(rhs.as_slice()[row], expected_rhs);
                for (j, col) in local_rows.into_iter().enumerate() {
                    if let Some(col) = col {
                        close(matrix.get(row, col), point.conductance[4 * i + j]);
                    }
                }
                close(cold.values()[13 * lane + 2 * i], point.charge[i]);
            }
            close(cold.values()[13 * lane + 10], 305.15);
            close(cold.values()[13 * lane + 11], 0.125);
        }
        let (_, _, limited) = trial_load(
            &circuit,
            &history,
            solution.as_slice(),
            history.trial_in(IterationPhase::Fix, Some(&cold)).unwrap(),
            None,
        );
        assert!(limited.is_nonconvergent());
        for lane in 0..count {
            close(limited.values()[13 * lane + 10], 307.15);
            close(limited.values()[13 * lane + 11], 0.225);
        }
        let (_, _, exact) = trial_load(
            &circuit,
            &history,
            solution.as_slice(),
            history.trial(),
            None,
        );
        assert!(!exact.is_nonconvergent());
        for lane in 0..count {
            close(exact.values()[13 * lane + 10], 310.15);
            close(exact.values()[13 * lane + 11], 0.8);
        }
    }
}

#[test]
fn explicit_bias_constraints_do_not_invoke_kernel_initialization_or_limiting() {
    use model_kernel_api::trial::{TrialCapabilities, TrialPolicy};
    use ngspice_rs::analysis::bias::{
        DcOutcome, DcSettings, DcStrategy, NodeForcing, solve_dc_forced,
    };
    // This witness intentionally declares specialized trial capabilities
    // without implementing those callbacks. The producer's default callbacks
    // return Unsupported if invoked, so the real constrained driver must
    // keep the entire load exact rather than merely converge eventually.
    let mut descriptor = descriptor(Capability::Computed);
    descriptor.trial = TrialCapabilities {
        initialization: TrialPolicy::Kernel,
        limiting: TrialPolicy::Kernel,
        convergence: TrialPolicy::Engine,
    };
    let artifact = Artifact::new(descriptor, PolynomialKernel).unwrap();
    let mut circuit = Circuit::new();
    let p = circuit.add_node("p").unwrap();
    PreparedDevice::bind(
        &mut circuit,
        &artifact,
        "x1",
        &[p, NodeId::GROUND],
        &[1e-3, 2e-4, 1e-6, 2e-7],
        InstanceTemperature::circuit(27.),
    )
    .unwrap();
    circuit.finalize().unwrap();
    let row = circuit.unknowns().node_row(p).unwrap();
    let settings = DcSettings::default();
    let result = solve_dc_forced(
        &circuit,
        &ModelContext::default(),
        &settings,
        &[],
        None,
        None,
        &NodeForcing {
            initial: vec![(row, 2.)],
            nodesets: Vec::new(),
        },
    )
    .unwrap();
    assert_eq!(
        result.report.outcome,
        DcOutcome::Converged(DcStrategy::Direct)
    );
    close(result.solution.values.get(row).unwrap(), 2.);
    assert!(!result.solution.trial.is_nonconvergent());
    close(result.solution.trial.values()[4], 2.);
    close(result.solution.trial.values()[0], 3.6e-6);
}

#[test]
fn sequential_runs_lease_prepared_data_without_sharing_live_candidates() {
    use std::sync::atomic::Ordering;
    let (circuit, preparations) = counted_circuit();
    let context = ModelContext::default();
    let original = counted_load(&circuit.prepare_run(), &context).unwrap();
    assert_eq!(preparations.load(Ordering::Relaxed), 129);
    let target = circuit.bind_instance_target("m3", "g").unwrap();
    let changed = context
        .with_instance_override(target.at(2e-3).unwrap())
        .unwrap();
    let updated = counted_load(&circuit.prepare_run(), &changed).unwrap();
    assert_ne!(original, updated);
    assert_eq!(preparations.load(Ordering::Relaxed), 130);
    assert_eq!(
        counted_load(&circuit.prepare_run(), &context).unwrap(),
        original
    );
    assert_eq!(preparations.load(Ordering::Relaxed), 131);
}

#[derive(Debug)]
struct CountingThermal {
    preparations: Arc<std::sync::atomic::AtomicUsize>,
}

impl Kernel for CountingThermal {
    type Prepared = ElectrothermalPrepared;

    fn prepare(
        &self,
        parameters: &[Real],
        temperature: Real,
    ) -> model_kernel_api::Result<Self::Prepared> {
        self.preparations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        ElectrothermalKernel.prepare(parameters, temperature)
    }

    fn evaluate(
        &self,
        prepared: &Self::Prepared,
        coordinates: &[Real],
        point: &mut Point<'_>,
    ) -> model_kernel_api::Result<()> {
        ElectrothermalKernel.evaluate(prepared, coordinates, point)
    }
}

#[test]
fn fixed_ambient_tile_updates_thermal_row_offsets_without_physical_preparation() {
    use ngspice_rs::devices::prepared_model::AmbientTemperature;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let preparations = Arc::new(AtomicUsize::new(0));
    let artifact = Artifact::new(
        electrothermal_descriptor(),
        CountingThermal {
            preparations: Arc::clone(&preparations),
        },
    )
    .unwrap();
    let mut circuit = Circuit::new();
    let mut nodes = Vec::new();
    for index in 0..2 {
        let p = circuit.add_node(&format!("p{index}")).unwrap();
        let t = circuit.add_node(&format!("t{index}")).unwrap();
        let q = circuit.add_node(&format!("q{index}")).unwrap();
        nodes.push([p, t, q]);
        PreparedDevice::bind(
            &mut circuit,
            &artifact,
            &format!("x{index}"),
            &[p, NodeId::GROUND, t, q],
            &ELECTROTHERMAL_PARAMETERS,
            InstanceTemperature {
                circuit_celsius: 27.,
                ambient: AmbientTemperature::AbsoluteKelvin(300.15),
            },
        )
        .unwrap();
    }
    circuit.finalize().unwrap();
    let run = circuit.prepare_run();
    let n = circuit.unknown_count();
    let mut solution = Vector::zeros(n);
    let rows: Vec<_> = nodes
        .iter()
        .map(|nodes| nodes.map(|node| circuit.unknowns().node_row(node).unwrap()))
        .collect();
    for [p, t, q] in &rows {
        solution.set(*p, 0.5).unwrap();
        solution.set(*t, 10.).unwrap();
        solution.set(*q, 0.2).unwrap();
    }
    let history = circuit.state_history().unwrap();
    for temperature in [27., 37., 27.] {
        let context = ModelContext::new(temperature, 27.);
        let mut matrix = SparseMatrix::new(n, n);
        let mut rhs = Vector::zeros(n);
        let mut trial = history.trial();
        run.load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &solution,
                model_context: &context,
                integration: None,
                history: &history,
                forcing: None,
            },
            &mut matrix,
            &mut rhs,
            &mut trial,
        )
        .unwrap();
        matrix.fold_duplicates();
        let rise = temperature - 27. + 10.;
        let g = ELECTROTHERMAL_PARAMETERS[0] * (1. + 0.02 * rise);
        for [p, t, _] in &rows {
            close(matrix.get(*p, *p), g);
            close(
                matrix.get(*p, *t),
                ELECTROTHERMAL_PARAMETERS[0] * 0.02 * 0.5,
            );
        }
        let linear = run
            .small_signal_system_at(&context, &solution, None)
            .unwrap();
        for [p, _, _] in &rows {
            close(linear.a.get(*p, *p), g);
        }
        assert_eq!(preparations.load(Ordering::Relaxed), 2);
    }
}

#[test]
#[ignore = "opt-in real prepared candidate preparation timing"]
fn vdmos_candidate_preparation_benchmark() {
    use std::time::Instant;
    let model = vdmos_fixture();
    let start = Instant::now();
    let circuit = vdmos_circuit(&model, 128);
    let binding = start.elapsed();
    let run = circuit.prepare_run();
    let context = ModelContext::default();
    let start = Instant::now();
    run.validate_context(&context).unwrap();
    let initial = start.elapsed();
    let target = circuit.bind_instance_target("m3", "multiplier").unwrap();
    let mut changed = Vec::with_capacity(31);
    let mut all = Vec::with_capacity(31);
    for round in 0..31 {
        let single = context
            .with_instance_override(target.at(4.5 + f64::from(round % 2)).unwrap())
            .unwrap();
        let start = Instant::now();
        run.validate_context(&single).unwrap();
        changed.push(start.elapsed());
        let warm = ModelContext::new(35. + f64::from(round % 2), 27.);
        let start = Instant::now();
        run.validate_context(&warm).unwrap();
        all.push(start.elapsed());
        run.validate_context(&context).unwrap();
    }
    changed.sort_unstable();
    all.sort_unstable();
    println!(
        "128 VDMOS: initial bind {binding:?}, transfer {initial:?}, one-lane median {:?}, all-lane median {:?}",
        changed[15], all[15]
    );
}

#[test]
fn prepared_tiles_rebuild_after_membership_and_artifact_edits() {
    let (mut circuit, preparations) = counted_circuit();
    let context = ModelContext::default();
    let old_target = circuit.bind_instance_target("m3", "g").unwrap();
    let original = counted_load(&circuit.prepare_run(), &context).unwrap();
    assert_eq!(original.len(), 129);
    let artifact = Artifact::new(
        descriptor(Capability::Computed),
        CountingKernel {
            preparations: Arc::clone(&preparations),
        },
    )
    .unwrap();
    // Same model schema but distinct exact artifact: old tile membership is
    // invalid, even though dimensions and names can remain unchanged.
    let node = circuit.nodes().get("n3").unwrap();
    circuit
        .replace_device(
            3,
            Box::new(
                PreparedDevice::new(
                    &artifact,
                    "m3",
                    vec![node, NodeId::GROUND],
                    &[2e-3, 2e-4, 1e-6, 2e-7],
                    InstanceTemperature::circuit(27.),
                )
                .unwrap(),
            ),
        )
        .unwrap();
    circuit.finalize().unwrap();
    let stale = context
        .with_instance_override(old_target.at(3e-3).unwrap())
        .unwrap();
    assert!(counted_load(&circuit.prepare_run(), &stale).is_err());
    let rebuilt = counted_load(&circuit.prepare_run(), &context).unwrap();
    let row = circuit.unknowns().node_row(node).unwrap();
    assert_eq!(
        rebuilt
            .iter()
            .zip(&original)
            .filter(|(a, b)| a != b)
            .count(),
        1
    );
    assert!(f64::from_bits(rebuilt[row]) > f64::from_bits(original[row]));
    let target = circuit.bind_instance_target("m3", "g").unwrap();
    let changed = context
        .with_instance_override(target.at(4e-3).unwrap())
        .unwrap();
    let before = preparations.load(std::sync::atomic::Ordering::Relaxed);
    let next = counted_load(&circuit.prepare_run(), &changed).unwrap();
    assert!(f64::from_bits(next[row]) > f64::from_bits(rebuilt[row]));
    assert_eq!(
        preparations.load(std::sync::atomic::Ordering::Relaxed),
        before + 1
    );
}
