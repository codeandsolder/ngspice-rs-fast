//! K mutual inductance through the device layer (GitHub #80): registry entry,
//! elaboration and reference resolution, the inductive-system check, and the
//! coupled-flux DC/companion/linear stamps of `Circuit::load` and
//! `Circuit::linear_system` (C `indload.c`, `mutacld.c`, `muttemp.c`).
use std::path::Path;

use ngspice_rs::devices::{
    AnalysisMode, Circuit, LoadRequest, ModelContext, MutualInductance, MutualLimits, Registry,
    StorageKind,
};
use ngspice_rs::maths::integrator::DEFAULT_XMU;
use ngspice_rs::maths::{IntegrationMethod, SparseMatrix, StepHistory, Vector};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{SpiceError, SpiceResult};

fn circuit(body: &str) -> SpiceResult<Circuit> {
    let netlist = Parser::new().parse_deck(&parse_deck_text(
        Path::new("k.cir"),
        &format!("mutual\n{body}\n.end\n"),
    ))?;
    Circuit::from_netlist(&netlist)
}

fn error(body: &str) -> SpiceError {
    circuit(body).expect_err("expected a failure")
}

fn close(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1e-12 * want.abs().max(1e-12),
        "{got} != {want}"
    );
}

/// `l1 a 0 1m`, `l2 b 0 4m` with resistors, coupled by `k`: M = k 2m.
const PAIR: &str = "r1 a 0 1\nr2 b 0 1\nl1 a 0 1m\nl2 b 0 4m";

#[test]
fn the_registry_ports_k_with_the_ind_references() {
    let registry = Registry::with_builtins();
    let entry = registry.get('K').unwrap();
    assert_eq!(entry.support, ngspice_rs::devices::DeviceSupport::Ported);
    assert!(entry.c_reference.contains("inp2k.c"), "{entry:?}");
    assert!(entry.c_reference.contains("ind/mutsetup.c"), "{entry:?}");
    assert!(!entry.c_reference.contains("cpl"), "{entry:?}");
}

#[test]
fn k_devices_have_no_terminals_branches_or_state() {
    let c = circuit(&format!("{PAIR}\nk1 l1 l2 0.5")).unwrap();
    let k = c.device("k1").unwrap();
    assert_eq!(k.designator(), 'k');
    assert!(k.terminals().is_empty());
    assert_eq!(k.branch_currents(), 0);
    assert_eq!(k.state_count(), 0);
    let coupling = k.mutual_coupling().unwrap();
    assert_eq!(coupling.coefficient, 0.5);
    let names: Vec<_> = coupling.inductors.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, ["l1", "l2"]);
    assert!(coupling.location.is_some());
    // M = k sqrt(L1 L2) = 0.5 * 2m on both branch rows.
    let terms = c.mutual_terms(&ModelContext::default()).unwrap();
    let rows = |name: &str| {
        let index = c.devices().iter().position(|d| d.name() == name).unwrap();
        (index, c.branch_rows(index).unwrap().start)
    };
    let ((i1, r1), (i2, r2)) = (rows("l1"), rows("l2"));
    assert_eq!(terms[i1].len(), 1);
    assert_eq!(terms[i1][0].row, r2);
    close(terms[i1][0].inductance, 1e-3);
    assert_eq!(terms[i2][0].row, r1);
    close(terms[i2][0].inductance, 1e-3);
}

#[test]
fn constructor_validation() {
    let reference = |name: &str| ngspice_rs::devices::ControlReference {
        name: name.to_owned(),
        location: None,
    };
    assert!(MutualInductance::new("k1", vec![reference("l1")], 0.5).is_err());
    assert!(MutualInductance::new("k1", vec![reference("l1"), reference("l2")], f64::NAN).is_err());
    let itself = MutualInductance::new("k1", vec![reference("l1"), reference("L1")], 0.5);
    assert!(matches!(itself, Err(SpiceError::Unsupported { .. })));
    let k = MutualInductance::new("k1", vec![reference("l1"), reference("l2")], -1.0).unwrap();
    assert_eq!(k.coefficient(), -1.0);
}

#[test]
fn references_must_name_existing_inductors() {
    let missing = error(&format!("{PAIR}\nk1 l1 l3 0.5"));
    assert!(matches!(missing, SpiceError::Parse { .. }), "{missing}");
    assert!(
        missing
            .to_string()
            .contains("k1: coupling to non-existent inductor l3"),
        "{missing}"
    );
    let resistor = error(&format!("{PAIR}\nk1 l1 r2 0.5")).to_string();
    assert!(resistor.contains("r2 is not an inductor"), "{resistor}");
    let itself = error(&format!("{PAIR}\nk1 l1 L1 0.5")).to_string();
    assert!(itself.contains("to itself"), "{itself}");
    // A K card after its inductors or before them resolves the same way.
    circuit("k1 l1 l2 0.5\nr1 a 0 1\nr2 b 0 1\nl1 a 0 1m\nl2 b 0 4m").unwrap();
}

#[test]
fn inductive_systems_must_be_positive_semidefinite() {
    let message = |body: &str| error(&format!("{PAIR}\nl3 c 0 2m\nr3 c 0 1\n{body}")).to_string();
    // |k| > 1 stores negative energy (C warns "is not positive definite").
    let big = message("k1 l1 l2 1.01");
    assert!(big.contains("not positive semidefinite"), "{big}");
    assert!(big.contains("l1 l2") && big.contains("k1"), "{big}");
    let negative = message("k1 l1 l2 -1.5");
    assert!(negative.contains("not positive semidefinite"), "{negative}");
    // Each |k| < 1 but the set is inconsistent: k12 = k13 = 0.9, k23 = -0.9.
    let set = message("k1 l1 l2 0.9\nk2 l1 l3 0.9\nk3 l2 l3 -0.9");
    assert!(set.contains("system l1 l2 l3 coupled"), "{set}");
    assert!(set.contains("k1 k2 k3"), "{set}");
    // Duplicate couplings of one pair are summed (C sums its loads).
    let summed = message("k1 l1 l2 0.6\nk2 l2 l1 0.6");
    assert!(summed.contains("not positive semidefinite"), "{summed}");
    // Ideal (|k| = 1) and partial couplings are accepted.
    for body in [
        "k1 l1 l2 1",
        "k1 l1 l2 -1",
        "k1 l1 l2 l3 1",
        "k1 l1 l2 0.5\nk2 l2 l3 0.5",
        "k1 l1 l2 0.5\nk2 l1 l2 0.4",
    ] {
        circuit(&format!("{PAIR}\nl3 c 0 2m\nr3 c 0 1\n{body}"))
            .unwrap_or_else(|e| panic!("{body}: {e}"));
    }
}

#[test]
fn multi_inductor_cards_couple_every_pair() {
    let c = circuit(&format!("{PAIR}\nl3 c 0 2m\nr3 c 0 1\nk1 l1 l2 l3 0.5")).unwrap();
    let terms = c.mutual_terms(&ModelContext::default()).unwrap();
    let index = |name: &str| c.devices().iter().position(|d| d.name() == name).unwrap();
    assert_eq!(terms[index("l1")].len(), 2);
    assert_eq!(terms[index("l2")].len(), 2);
    assert_eq!(terms[index("l3")].len(), 2);
    assert!(terms[index("k1")].is_empty());
}

#[test]
fn subcircuit_k_cards_use_hierarchical_names() {
    let body = ".subckt xf p s\nl1 p 0 1m\nl2 s 0 4m\nk1 l1 l2 0.9\n.ends\n\
                r1 a 0 1\nr2 b 0 1\nr3 c 0 1\nr4 d 0 1\nx1 a b xf\nx2 c d xf";
    let c = circuit(body).unwrap();
    for k in ["k.x1.k1", "k.x2.k1"] {
        let coupling = c.device(k).unwrap().mutual_coupling().unwrap();
        let prefix = &k[2..4];
        for reference in coupling.inductors {
            assert!(
                reference.name.starts_with(&format!("l.{prefix}.")),
                "{k}: {reference:?}"
            );
        }
    }
    // A body K card cannot reach a top-level inductor: its name is renamed too.
    let outer = error(
        ".subckt xf p\nla p 0 1m\nk1 la lt 0.5\n.ends\nr1 a 0 1\nlt t 0 1m\nrt t 0 1\nx1 a xf",
    );
    assert!(
        outer.to_string().contains("non-existent inductor l.x1.lt"),
        "{outer}"
    );
    // A top-level K card may name a flattened inductor explicitly.
    circuit(&format!(
        ".subckt xf p\nla p 0 1m\n.ends\nrx x 0 1\nx1 x xf\n{PAIR}\nk1 l1 l.x1.la 0.5"
    ))
    .unwrap();
}

#[test]
fn model_backed_inductors_couple_through_their_undivided_inductance() {
    // MUTtemp uses INDinduct (after TC and scale, before /m): M = k sqrt(2m 4m),
    // while l1 itself stamps 2m / 2.
    let c =
        circuit("r1 a 0 1\nr2 b 0 1\nl1 a 0 lm m=2\nl2 b 0 4m\nk1 l1 l2 0.5\n.model lm l(ind=2m)")
            .unwrap();
    let index = |name: &str| c.devices().iter().position(|d| d.name() == name).unwrap();
    let value = c.devices()[index("l1")]
        .inductance(&ModelContext::default())
        .unwrap()
        .unwrap();
    close(value.effective, 1e-3);
    close(value.coupling_base, 2e-3);
    let terms = c.mutual_terms(&ModelContext::default()).unwrap();
    close(
        terms[index("l1")][0].inductance,
        0.5 * (2e-3_f64 * 4e-3).sqrt(),
    );
}

#[test]
fn multiplicity_can_make_the_stamped_system_indefinite_where_c_stays_silent() {
    // l1 stamps 10m / 2 = 5m but M = 0.9 sqrt(10m 40m) = 18m > sqrt(5m 40m):
    // the simulated matrix is indefinite although C's own check (INDinduct
    // on the diagonal, muttemp.c) passes and C prints nothing.
    let deck = "r1 a 0 1\nr2 b 0 1\nl1 a 0 lm m=2\nl2 b 0 lm2\nk1 l1 l2 0.9\n\
                .model lm l(ind=10m)\n.model lm2 l(ind=40m)";
    let message = error(deck).to_string();
    assert!(
        message.contains("not positive semidefinite as stamped"),
        "{message}"
    );
    assert!(message.contains("multiplicity m"), "{message}");
    assert!(!message.contains("C only warns"), "{message}");
    // With |k| > 1 C's matrix fails too, and the message says C warns.
    let message = error(&deck.replace("0.9", "1.5")).to_string();
    assert!(message.contains("C only warns"), "{message}");
    assert!(!message.contains("multiplicity m"), "{message}");
}

#[test]
fn model_backed_storage_elements_seed_uic_and_join_truncation_control() {
    let c = circuit(
        "r1 a 0 1\nr2 b 0 1\nl1 a 0 lm ic=10m m=2\nc1 b 0 cm ic=2 m=3\n\
         .model lm l(ind=2m tc1=0.01)\n.model cm c(cap=1u tc1=0.02)",
    )
    .unwrap();
    let index = |name: &str| c.devices().iter().position(|d| d.name() == name).unwrap();
    let hot = ModelContext::new(77.0, 27.0);
    let inductor = c.devices()[index("l1")]
        .storage_element(&hot)
        .unwrap()
        .unwrap();
    assert_eq!(inductor.kind, StorageKind::Inductor);
    close(inductor.value, 2e-3 * 1.5 / 2.0);
    assert_eq!(inductor.initial, Some(10e-3));
    let capacitor = c.devices()[index("c1")]
        .storage_element(&hot)
        .unwrap()
        .unwrap();
    assert_eq!(capacitor.kind, StorageKind::Capacitor);
    close(capacitor.value, 1e-6 * 2.0 * 3.0);
    assert_eq!(capacitor.initial, Some(2.0));
    for name in ["l1", "c1"] {
        assert_eq!(c.devices()[index(name)].truncation_slots(), [0], "{name}");
    }
    assert!(c.devices()[index("r1")].truncation_slots().is_empty());
}

fn load(
    circuit: &Circuit,
    history: &ngspice_rs::devices::StateHistory,
    solution: &[f64],
    integration: Option<&ngspice_rs::maths::Coefficients>,
) -> (SparseMatrix, Vector, ngspice_rs::devices::TrialState) {
    let n = circuit.unknown_count();
    let mut matrix = SparseMatrix::new(n, n);
    let mut rhs = Vector::zeros(n);
    let mut trial = history.trial();
    let mode = integration.map_or(AnalysisMode::OperatingPoint, |c| AnalysisMode::Transient {
        time: 1.0,
        dt: c.dt(),
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
fn coupled_flux_drives_dc_state_companions_and_linear_assembly() {
    // Only the two inductors: nodes a, b; branch rows 2 (l1) and 3 (l2).
    let mut c = circuit("l1 a 0 1m\nl2 b 0 4m\nk1 l1 l2 0.5").unwrap();
    let m = 1e-3;
    let index =
        |c: &Circuit, name: &str| c.devices().iter().position(|d| d.name() == name).unwrap();
    let (b1, b2) = (
        c.branch_rows(index(&c, "l1")).unwrap().start,
        c.branch_rows(index(&c, "l2")).unwrap().start,
    );
    let (s1, s2) = (
        c.state_rows(index(&c, "l1")).unwrap().start,
        c.state_rows(index(&c, "l2")).unwrap().start,
    );
    let mut x = vec![0.0; c.unknown_count()];
    x[b1] = 2.0;
    x[b2] = -1.0;
    // DC: shorts, no mutual matrix entry; the recorded flux is coupled.
    let mut history = c.state_history();
    let (matrix, _, trial) = load(&c, &history, &x, None);
    assert_eq!(matrix.get(b1, b2), 0.0);
    close(trial.values()[s1], 1e-3 * 2.0 - m);
    close(trial.values()[s2], -4e-3 + m * 2.0);
    c.accept_point(&Vector::from_slice(&x), None, &mut history, trial)
        .unwrap();
    // Backward Euler from that point: the cross entries are -ag0 M and the
    // history source uses the coupled flux, so v = ag0 (flux - flux_prev).
    let steps = StepHistory::new();
    let be = steps
        .trial(IntegrationMethod::Trapezoidal, 1, 1e-6, DEFAULT_XMU)
        .unwrap();
    let ag0 = be.ag()[0];
    let mut y = x.clone();
    y[b1] = 3.0;
    let (matrix, rhs, trial) = load(&c, &history, &y, Some(&be));
    close(matrix.get(b1, b2), -ag0 * m);
    close(matrix.get(b2, b1), -ag0 * m);
    close(matrix.get(b1, b1), -ag0 * 1e-3);
    let flux1 = 1e-3 * 3.0 - m;
    close(trial.values()[s1], flux1);
    // Row b1: v(a) - ag0 L1 i1 - ag0 M i2 = derivative - ag0 flux1.
    close(trial.values()[s1 + 1], ag0 * (flux1 - (1e-3 * 2.0 - m)));
    close(rhs.as_slice()[b1], trial.values()[s1 + 1] - ag0 * flux1);
    // Truncation control sees the coupled flux slot.
    assert_eq!(c.devices()[index(&c, "l1")].truncation_slots(), [0]);
    // E x' + A x = b: -M between the branch rows, -L on the diagonal.
    let system = c.linear_system().unwrap();
    close(system.e.get(b1, b2), -m);
    close(system.e.get(b2, b1), -m);
    close(system.e.get(b1, b1), -1e-3);
    close(system.e.get(b2, b2), -4e-3);
}

#[test]
fn independent_coupling_groups_keep_pair_order_and_local_diagnostics() {
    let body =
        "l1 a 0 1m\nl2 b 0 4m\nl3 c 0 2m\nl4 d 0 8m\nk1 l1 l2 0.25\nk2 l3 l4 0.75\nk3 l2 l1 -0.125";
    let c = circuit(body).unwrap();
    let index = |name: &str| c.devices().iter().position(|d| d.name() == name).unwrap();
    let terms = c.mutual_terms(&ModelContext::default()).unwrap();
    let a = &terms[index("l1")];
    let b = &terms[index("l3")];
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 1);
    assert_eq!(a[0].row, c.branch_rows(index("l2")).unwrap().start);
    assert_eq!(a[1].row, a[0].row);
    close(a[0].inductance, -0.00025);
    close(a[1].inductance, 0.0005);
    close(b[0].inductance, 0.003);
    // A failure in the second component reports only its member inductors
    // and coupling cards, not cards interleaved from the first component.
    let error = error(&format!("{body}\nk4 l4 l3 0.5")).to_string();
    assert!(error.contains("system l3 l4 coupled by k2 k4"), "{error}");
    assert!(!error.contains("l1 l2"), "{error}");

    let context = ModelContext::default()
        .with_instance_override(
            c.instance_override("k2", "k", 0.5, &ModelContext::default())
                .unwrap(),
        )
        .unwrap();
    let updated = c.mutual_terms(&context).unwrap();
    close(updated[index("l3")][0].inductance, 0.002);
    assert_eq!(updated[index("l1")], terms[index("l1")]);
    assert_eq!(c.mutual_terms(&ModelContext::default()).unwrap(), terms);
}

fn bounded_circuit(body: &str, limits: MutualLimits) -> SpiceResult<Circuit> {
    let netlist = Parser::new().parse_deck(&parse_deck_text(
        Path::new("limits.cir"),
        &format!("mutual limits\n{body}\n.end\n"),
    ))?;
    Circuit::from_netlist_with_mutual_limits(&netlist, &ModelContext::default(), limits)
}

fn resource_error(error: SpiceError, expected: &str) {
    assert!(
        matches!(
            error,
            SpiceError::Unsupported {
                location: Some(_),
                ..
            }
        ),
        "{error}"
    );
    assert!(error.to_string().contains(expected), "{error}");
}

#[test]
fn pair_and_group_limits_are_preflighted_before_expansion() {
    use std::fmt::Write;
    let mut large = String::from("klarge");
    for i in 0..10_000 {
        write!(large, " l{i}").unwrap();
    }
    large.push_str(" 0.01");
    resource_error(
        bounded_circuit(&large, MutualLimits::default()).unwrap_err(),
        "mutual pair limit exceeded",
    );
    let overlap = "l1 a 0 1\nl2 b 0 1\nl3 c 0 1\nl4 d 0 1\nk1 l1 l2 l3 0.1\nk2 l2 l3 l4 0.1";
    resource_error(
        bounded_circuit(
            overlap,
            MutualLimits {
                pairs: 5,
                ..MutualLimits::default()
            },
        )
        .unwrap_err(),
        "k2: mutual pair limit exceeded",
    );
    resource_error(
        bounded_circuit(
            overlap,
            MutualLimits {
                group_inductors: 3,
                ..MutualLimits::default()
            },
        )
        .unwrap_err(),
        "connected-group size limit exceeded",
    );
    bounded_circuit(
        overlap,
        MutualLimits {
            pairs: 6,
            group_inductors: 4,
            ..MutualLimits::default()
        },
    )
    .unwrap();
    resource_error(
        bounded_circuit(
            overlap,
            MutualLimits {
                storage_bytes: 0,
                ..MutualLimits::default()
            },
        )
        .unwrap_err(),
        "mutual storage limit exceeded",
    );
    resource_error(
        bounded_circuit(
            overlap,
            MutualLimits {
                validation_work: 0,
                ..MutualLimits::default()
            },
        )
        .unwrap_err(),
        "mutual validation work limit exceeded",
    );
    bounded_circuit(
        "l1 a 0 1",
        MutualLimits {
            storage_bytes: 0,
            validation_work: 0,
            ..MutualLimits::default()
        },
    )
    .unwrap();
}

#[test]
fn failed_binding_does_not_publish_partial_terms_or_setup() {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("limits.cir"),
            "mutual limits\nl1 a 0 1\nl2 b 0 1\nk1 l1 l2 0.5\n.end\n",
        ))
        .unwrap();
    let models = ngspice_rs::devices::models::ModelResolver::new(&netlist.models).unwrap();
    let mut circuit = Circuit::with_mutual_limits(MutualLimits {
        pairs: 0,
        ..MutualLimits::default()
    });
    circuit
        .add_instances(&netlist.devices, &models, &ModelContext::default())
        .unwrap();
    circuit.rebuild_unknowns();
    resource_error(
        circuit.mutual_terms(&ModelContext::default()).unwrap_err(),
        "mutual pair limit",
    );
    resource_error(circuit.setup_elements(2).unwrap_err(), "mutual pair limit");
    resource_error(circuit.finalize().unwrap_err(), "mutual pair limit");
}

#[test]
fn sparse_validation_preserves_repeated_pair_order_and_zero_inductance() {
    let c = bounded_circuit(
        "l1 a 0 1\nl2 b 0 1\nl0 c 0 0\nk1 l1 l2 1e16\nk2 l2 l1 -1e16\nk3 l1 l2 0.25\nk0 l0 l1 100",
        MutualLimits {
            dense_bytes: 0,
            ..MutualLimits::default()
        },
    )
    .unwrap();
    let terms = c.mutual_terms(&ModelContext::default()).unwrap();
    assert_eq!(
        terms[0]
            .iter()
            .map(|term| term.inductance.to_bits())
            .collect::<Vec<_>>(),
        [0.0_f64, 0.25, -1e16, 1e16].map(f64::to_bits)
    );
    assert_eq!(
        terms[1]
            .iter()
            .map(|term| term.inductance.to_bits())
            .collect::<Vec<_>>(),
        [0.25_f64, -1e16, 1e16].map(f64::to_bits)
    );
    assert_eq!(terms[2][0].inductance, 0.0);
}

#[test]
fn large_sparse_chain_needs_no_dense_matrix() {
    use std::fmt::Write;
    let nodes = 10_000;
    let mut deck = String::new();
    for i in 0..nodes {
        writeln!(deck, "l{i} n{i} 0 1").unwrap();
    }
    for i in 1..nodes {
        writeln!(deck, "k{i} l{} l{i} 0.5", i - 1).unwrap();
    }
    let c = bounded_circuit(
        &deck,
        MutualLimits {
            dense_bytes: 0,
            ..MutualLimits::default()
        },
    )
    .unwrap();
    let terms = c.mutual_terms(&ModelContext::default()).unwrap();
    assert_eq!(terms.iter().map(Vec::len).sum::<usize>(), 2 * (nodes - 1));
}
