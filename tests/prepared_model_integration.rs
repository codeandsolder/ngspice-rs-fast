//! Prepared-model ABI contract exercised through real Circuit/analysis loads.
//! The analytic polynomial is a source-compiler stand-in; the physical
//! equations live outside the engine and are only called through Kernel.

use std::sync::Arc;

use model_kernel_api::batch::{BatchOutput, PreparedTile};
use model_kernel_api::noise::{Injection, Sink, Source};
use model_kernel_api::{
    ABI_VERSION, Artifact, Binding, Capability, Descriptor, Equation, EquationKind, Error, Kernel,
    Parameter, Point, PointBuffer, Result,
};
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput};
use ngspice_rs::devices::noise::{NoiseContext, NoiseSource};
use ngspice_rs::devices::prepared_model::PreparedDevice;
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
}

impl Kernel for PolynomialKernel {
    type Prepared = Polynomial;

    fn prepare(&self, parameters: &[Real], temperature: Real) -> Result<Polynomial> {
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
        })
    }

    fn evaluate(
        &self,
        prepared: &Polynomial,
        voltages: &[Real],
        out: &mut Point<'_>,
    ) -> Result<()> {
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
        frequency: Real,
        sink: &mut dyn Sink,
    ) -> Result<()> {
        let [positive, negative] = voltages else {
            return Err(Error::Shape("polynomial noise input shape mismatch"));
        };
        let voltage = positive - negative;
        let conductance = prepared.g + 3. * prepared.cubic * voltage * voltage;
        // Explicit frequency dependence tests refresh independently of AC.
        let density = 4. * ngspice_rs::devices::noise::BOLTZMANN * 300.15 * conductance;
        sink.push(Source {
            name: "thermal",
            density: density * (1. + frequency / 1000.),
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
                Arc::clone(artifact),
                "x1",
                vec![a, b],
                &[g, 2e-4, 1e-6, 2e-7],
                27.,
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
        Arc::clone(&model),
        "x1",
        vec![NodeId::GROUND, NodeId::GROUND],
        &[1e-3, 2e-4, 1e-6, 2e-7],
        27.,
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
    let model = artifact();
    let candidates: [(&[Real], Real); 3] = [
        (&[1e-3, 2e-4, 1e-6, 2e-7], 300.15),
        (&[2e-3, 1e-4, 3e-6, 3e-7], 308.15),
        (&[3e-3, 4e-4, 2e-6, 1e-7], 263.15),
    ];
    let tile = PreparedTile::new(&model, &candidates).unwrap();
    let mut out = BatchOutput::new(2, candidates.len()).unwrap();
    let voltage = [0.5, 0., 1.1, 0.3, -0.4, 0.2];
    tile.evaluate(&voltage, &mut out).unwrap();
    for (lane, (data, _)) in candidates.iter().enumerate() {
        let v = voltage[2 * lane] - voltage[2 * lane + 1];
        close(out.current[2 * lane], data[0] * v + data[1] * v.powi(3));
        close(out.charge[2 * lane], data[2] * v + data[3] * v.powi(3));
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
                Arc::clone(&model),
                "x1",
                vec![out, NodeId::GROUND],
                &[1e-3, 2e-4, 1e-6, 2e-7],
                27.,
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
                Arc::clone(&model),
                "x1",
                vec![output, NodeId::GROUND],
                &[1e-3, 2e-4, 1e-6, 2e-7],
                27.,
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
    let model_context = ModelContext::default().with_frequency(1000.);
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
    match &generators[0] {
        NoiseSource::Generalized {
            density,
            injections,
            ..
        } => {
            close(
                *density,
                8. * ngspice_rs::devices::noise::BOLTZMANN
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
                Arc::clone(&factory),
                "x1",
                vec![a, NodeId::GROUND],
                &[1e-3, 2e-4, 1e-6, 2e-7],
                27.,
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
        let density = 4.
            * ngspice_rs::devices::noise::BOLTZMANN
            * 300.15
            * (total_g + 1e-3 * frequency / 1000.)
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
                Arc::clone(&model),
                "x1",
                vec![a, NodeId::GROUND],
                &[1e-3, 2e-4, 1e-6, 2e-7],
                27.,
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
    let model_context = ModelContext::default().with_frequency(1000.);
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
                Arc::clone(&model),
                "x2",
                vec![a, NodeId::GROUND],
                &[1e-3, 2e-4, 1e-6, 2e-7],
                27.,
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
        error.to_string().contains("declared-zero storage"),
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

    fn prepare(&self, parameters: &[Real], temperature_kelvin: Real) -> Result<Self::Prepared> {
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
    ) -> Result<()> {
        let [positive, negative, thermal, recovery] = voltages else {
            return Err(Error::Domain("electrothermal input shape".into()));
        };
        let v = positive - negative;
        let g = prepared.g * (1. + prepared.temperature_coefficient * (thermal - 300.15));
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
            prepared.heat_conductance * (thermal - prepared.ambient_kelvin) - i * v,
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
            prepared.heat_capacitance * (thermal - prepared.ambient_kelvin),
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
    }
}

#[test]
fn prepared_kernel_supports_thermal_and_internal_dae_rows_without_electrical_unit_mixing() {
    let descriptor = electrothermal_descriptor();
    let kernel = ElectrothermalKernel;
    let prepared = kernel.prepare(&ELECTROTHERMAL_PARAMETERS, 300.15).unwrap();
    let bias = [0.6, -0.2, 305.15, 0.1];
    let mut output = PointBuffer::new(4).unwrap();
    output.point().reset(Capability::Computed);
    kernel
        .evaluate(&prepared, &bias, &mut output.point())
        .unwrap();
    output.point().validate(&descriptor).unwrap();
    let output = output.point();
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
                Arc::clone(&artifact),
                "xthermal",
                vec![p, NodeId::GROUND, temperature, recovery],
                &ELECTROTHERMAL_PARAMETERS,
                27.,
            )
            .unwrap(),
        ))
        .unwrap();
    circuit.finalize().unwrap();
    let history = circuit.state_history().unwrap();
    let solution = Vector::from_slice(&[0.5, 37., 0.2]);
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
    let dc_solution = Vector::from_slice(&[0.5, 37., 0.2]);
    let (.., dc_trial) = load(&circuit, &accepted, dc_solution.as_slice(), None);
    circuit
        .accept_point(&dc_solution, None, &mut accepted, dc_trial)
        .unwrap();
    let stored = accepted.accepted(1).unwrap().to_vec();
    let step = StepHistory::new();
    let coeff = step
        .trial(IntegrationMethod::Trapezoidal, 1, 1e-6, DEFAULT_XMU)
        .unwrap();
    let transient_solution = [0.75, 37., 0.3];
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
            Arc::clone(&model),
            "bad",
            vec![NodeId::GROUND; 4],
            &ELECTROTHERMAL_PARAMETERS,
            -273.15
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
