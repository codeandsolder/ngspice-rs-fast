use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use model_kernel_api::{
    ABI_VERSION, Artifact, Binding, Capability, Descriptor, Equation, EquationKind, Kernel,
    Parameter, Point, trial::TrialCapabilities,
};

use super::{InstanceTemperature, PreparedDevice};
use crate::devices::{
    Circuit, ModelContext,
    circuit::{CandidateNumericOverride, preparation_metrics},
};
use crate::primitives::{NodeId, Real};

#[derive(Debug)]
struct CountingConductance(Arc<AtomicUsize>);

impl Kernel for CountingConductance {
    type Prepared = Real;

    fn prepare(&self, values: &[Real], _: Real) -> model_kernel_api::Result<Real> {
        self.0.fetch_add(1, Ordering::Relaxed);
        match values {
            [g] if g.is_finite() && *g > 0. => Ok(*g),
            _ => Err(model_kernel_api::Error::Domain("conductance".into())),
        }
    }

    fn evaluate(
        &self,
        g: &Real,
        coordinates: &[Real],
        out: &mut Point<'_>,
    ) -> model_kernel_api::Result<()> {
        let [p, n] = coordinates else {
            return Err(model_kernel_api::Error::Shape("coordinates"));
        };
        let current = g * (p - n);
        out.current.copy_from_slice(&[current, -current]);
        out.conductance.copy_from_slice(&[*g, -*g, -*g, *g]);
        out.charge.fill(0.);
        out.capacitance.fill(0.);
        Ok(())
    }
}

fn counted_circuit() -> (Circuit, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let artifact = Artifact::new(
        Descriptor {
            family: "candidate_conductance".into(),
            fingerprint: [1; 16],
            abi: ABI_VERSION,
            equations: ["p", "n"]
                .into_iter()
                .map(|name| Equation {
                    name: name.into(),
                    kind: EquationKind::Electrical,
                    binding: Binding::Terminal,
                })
                .collect(),
            parameters: vec![Parameter {
                name: "g".into(),
                default: 1e-3,
                changes_topology: false,
            }],
            charge: Capability::Zero,
            noise: Capability::Zero,
            trial: TrialCapabilities::ENGINE,
            taylor: model_kernel_api::taylor::TaylorCapability::Unavailable,
        },
        CountingConductance(Arc::clone(&count)),
    )
    .unwrap();
    let mut circuit = Circuit::new();
    for index in 0..129 {
        let node = circuit.add_node(&format!("n{index}")).unwrap();
        circuit
            .add_device(Box::new(
                PreparedDevice::new(
                    &artifact,
                    format!("m{index}"),
                    vec![node, NodeId::GROUND],
                    &[1e-3],
                    InstanceTemperature::circuit(27.),
                )
                .unwrap(),
            ))
            .unwrap();
    }
    circuit.finalize().unwrap();
    (circuit, count)
}

#[test]
fn optimizer_commit_leases_validation_preparation_and_preserves_failed_candidate() {
    let (mut circuit, count) = counted_circuit();
    let context = ModelContext::default();
    let first = circuit.bind_instance_target("m3", "g").unwrap();
    let second = circuit.bind_instance_target("m80", "g").unwrap();
    circuit.prepare_run().validate_context(&context).unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 129);
    let valid = CandidateNumericOverride::Instance(first.at(2e-3).unwrap());
    circuit
        .commit_candidate_overrides(&context, vec![valid])
        .unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 130);
    circuit.clear_candidate_analysis_state();
    let _ = preparation_metrics::take();
    circuit.prepare_run().validate_dc_inputs(&context).unwrap();
    assert_eq!(preparation_metrics::take().contexts, 0);
    assert_eq!(count.load(Ordering::Relaxed), 130);

    let invalid = CandidateNumericOverride::Instance(second.at(-1.).unwrap());
    let other = CandidateNumericOverride::Instance(first.at(3e-3).unwrap());
    assert!(
        circuit
            .commit_candidate_overrides(&context, vec![other, invalid])
            .is_err()
    );
    let after_failure = count.load(Ordering::Relaxed);
    let run = circuit.prepare_run();
    run.validate_dc_inputs(&context).unwrap();
    assert_eq!(count.load(Ordering::Relaxed), after_failure);
    drop(run);

    circuit
        .commit_candidate_overrides(&context, vec![other])
        .unwrap();
    assert_eq!(count.load(Ordering::Relaxed), after_failure + 1);
    circuit.prepare_run().validate_dc_inputs(&context).unwrap();
    assert_eq!(count.load(Ordering::Relaxed), after_failure + 1);
}

#[test]
fn preparation_key_includes_analysis_phase_and_all_elaboration_settings() {
    use crate::primitives::AnalysisKind;
    let original = ModelContext::default();
    let cases = [
        ModelContext {
            analysis_kind: Some(AnalysisKind::Transient),
            ..original
        },
        ModelContext {
            bias_analysis: Some(AnalysisKind::OperatingPoint),
            ..original
        },
        ModelContext {
            configured_gmin: 1e-8,
            ..original
        },
        ModelContext {
            geometry_scale: 2.,
            ..original
        },
        ModelContext {
            shunts: crate::devices::models::TopologyShunts {
                resistance: Some(1e6),
                ..original.shunts
            },
            ..original
        },
        ModelContext {
            shunts: crate::devices::models::TopologyShunts {
                capacitance: 1e-12,
                ..original.shunts
            },
            ..original
        },
    ];
    assert!(original.same_preparation(&original));
    for case in cases {
        assert!(!original.same_preparation(&case));
        assert!(!case.same_preparation(&original));
    }
}

#[test]
fn active_prepared_view_rejects_context_transaction_before_preparing() {
    let (circuit, count) = counted_circuit();
    let run = circuit.prepare_run();
    let context = ModelContext::default();
    let held = run.point(&context).unwrap();
    let changed = context
        .with_instance_override(
            circuit
                .bind_instance_target("m3", "g")
                .unwrap()
                .at(2e-3)
                .unwrap(),
        )
        .unwrap();
    assert!(run.validate_context(&changed).is_err());
    assert_eq!(count.load(Ordering::Relaxed), 129);
    drop(held);
    run.validate_context(&changed).unwrap();
    assert_eq!(count.load(Ordering::Relaxed), 130);
}
