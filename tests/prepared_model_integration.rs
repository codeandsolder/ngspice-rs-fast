//! Prepared-model ABI contract exercised through real Circuit/analysis loads.
//! The analytic polynomial is a source-compiler stand-in; the physical
//! equations live outside the engine and are only called through Kernel.

use std::sync::Arc;

use ngspice_rs::analysis::{AnalysisContext, AnalysisInput};
use ngspice_rs::devices::prepared_model::batch::{BatchInput, BatchOutput, evaluate_tile};
use ngspice_rs::devices::prepared_model::{
    ABI_VERSION, Artifact, Capability, Descriptor, Kernel, Parameter, Point,
};
use ngspice_rs::devices::{
    AnalysisMode, Circuit, IndependentSource, LoadRequest, ModelContext, Resistor, Waveform,
};
use ngspice_rs::maths::integrator::DEFAULT_XMU;
use ngspice_rs::maths::{Coefficients, IntegrationMethod, SparseMatrix, StepHistory, Vector};
use ngspice_rs::primitives::{AnalysisKind, Complex, NodeId, Real, SpiceError, SpiceResult};

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

    fn prepare(&self, parameters: &[Real], temperature: Real) -> SpiceResult<Polynomial> {
        let [g, cubic, c, charge_cubic] = parameters else {
            return Err(SpiceError::circuit(
                "polynomial artifact has four parameters",
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
            return Err(SpiceError::circuit(
                "invalid polynomial physical parameters",
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
    ) -> SpiceResult<()> {
        let [positive, negative] = voltages else {
            return Err(SpiceError::circuit(
                "polynomial kernel requires two terminals",
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
}

fn descriptor(charge: Capability) -> Descriptor {
    Descriptor {
        family: "analytic-polynomial".into(),
        fingerprint: [0x35; 16],
        abi: ABI_VERSION,
        terminals: 2,
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
        noise: Capability::Unavailable,
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
            artifact
                .instance("x1", vec![a, b], &[g, 2e-4, 1e-6, 2e-7], 27.)
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
    let first = model
        .instance(
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
    let model = PolynomialKernel;
    let prepared = [
        model.prepare(&[1e-3, 2e-4, 1e-6, 2e-7], 27.).unwrap(),
        model.prepare(&[2e-3, 1e-4, 3e-6, 3e-7], 35.).unwrap(),
        model.prepare(&[3e-3, 4e-4, 2e-6, 1e-7], -10.).unwrap(),
    ];

    let mut out = BatchOutput::new(2, prepared.len()).unwrap();
    let voltage = [0.5, 0., 1.1, 0.3, -0.4, 0.2];
    evaluate_tile(
        &model,
        &BatchInput {
            prepared: &prepared,
            voltages: &voltage,
            charge: Capability::Computed,
        },
        &mut out,
    )
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
            model
                .instance(
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
            model
                .instance(
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
