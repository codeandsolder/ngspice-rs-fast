//! C reference DC branch layout, exercised through the real Circuit MNA
//! assembler while the separate coupled transient solver is under development.
use std::path::Path;

use ngspice_rs::devices::Circuit;
use ngspice_rs::devices::cpl::dc::Cpl;
use ngspice_rs::devices::cpl::plan::{CplMatrixKind, CplPlan};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::NodeId;

#[test]
fn one_wire_stamp_matches_cplload_dc_branch_equations() {
    let plan = CplPlan::prepare(1, 1., &[1.], &[100e-9], &[0.], &[10e-12]).unwrap();
    let mut circuit = Circuit::new();
    let a = circuit.add_node("a").unwrap();
    let b = circuit.add_node("b").unwrap();
    circuit
        .add_device(Box::new(
            Cpl::new(
                "p1",
                "line",
                vec![a, NodeId::GROUND, b, NodeId::GROUND],
                plan,
            )
            .unwrap(),
        ))
        .unwrap();
    let system = circuit.linear_system().unwrap();
    let m = &system.a;
    assert_eq!(m.rows(), 4);
    assert_eq!(m.get(0, 0), 0.1e-12);
    assert_eq!(m.get(1, 1), 0.1e-12);
    assert_eq!(m.get(0, 1), 0.1e-12);
    assert_eq!(m.get(1, 0), 0.1e-12);
    assert_eq!(m.get(0, 2), 1.);
    assert_eq!(m.get(1, 3), 1.);
    assert_eq!(m.get(2, 2), 1.);
    assert_eq!(m.get(2, 3), 1.);
    assert_eq!(m.get(3, 0), 1.);
    assert_eq!(m.get(3, 1), -1.);
    assert_eq!(m.get(3, 2), -1.);
    assert_eq!(m.get(3, 3), 0.);
}

// One real network shared by immutable linear assembly and Newton MNA paths.
fn loaded_line_circuit() -> Circuit {
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("cpl-dc-compare.cir"),
            "DC P-line reference\nV1 a 0 DC 1\nR1 b 0 1\n.end\n",
        ))
        .unwrap();
    let mut circuit = Circuit::from_netlist(&parsed).unwrap();
    let a = circuit.add_node("a").unwrap();
    let b = circuit.add_node("b").unwrap();
    let plan = CplPlan::prepare(1, 1., &[1.], &[100e-9], &[0.], &[10e-12]).unwrap();
    circuit
        .add_device(Box::new(
            Cpl::new(
                "p1",
                "line",
                vec![a, NodeId::GROUND, b, NodeId::GROUND],
                plan,
            )
            .unwrap(),
        ))
        .unwrap();
    circuit
}

#[test]
fn full_circuit_dc_matches_installed_ngspice_one_volt_one_ohm_line_and_load() {
    // C reference: V(a)=1 V, V(b)=0.5 V, CPL near/far branch currents
    // of +0.5 A/-0.5 A for a 1 V source and 1 ohm load.
    let mut circuit = loaded_line_circuit();
    let system = circuit.linear_system().unwrap();
    let solution = system.a.solve(&system.dc_rhs(None).unwrap()).unwrap();
    // Both nodal rows are the first two unknowns (input deck V1 a, R1 b).
    assert!((solution.get(0).unwrap() - 1.).abs() < 1e-12);
    assert!((solution.get(1).unwrap() - 0.5).abs() < 1e-12);
}

#[test]
fn newton_operating_point_uses_the_same_cplload_dc_equations() {
    let mut circuit = loaded_line_circuit();
    circuit.finalize().unwrap();
    let solution = ngspice_rs::analysis::bias::solve_dc(
        &circuit,
        &ngspice_rs::devices::ModelContext::default(),
        &ngspice_rs::analysis::newton::NewtonOptions::default(),
        &[],
        None,
        None,
    )
    .unwrap();
    assert!((solution.values.get(0).unwrap() - 1.).abs() < 1e-12);
    assert!((solution.values.get(1).unwrap() - 0.5).abs() < 1e-12);
}

#[test]
fn c_setup_clamps_zero_and_tiny_series_resistance_before_dc_load() {
    // ngspice reference: R=0, R=1u, R=100u all give Vout=0.9999000
    // for a 1 V source, one 1 m P line and one 1 ohm load.
    for resistance in [0., 1e-6, 1e-4] {
        let plan = CplPlan::prepare(1, 1., &[resistance], &[100e-9], &[], &[10e-12]).unwrap();
        assert_eq!(
            plan.matrix(CplMatrixKind::Resistance).get(0, 0),
            Some(resistance)
        );
        let mut circuit = Circuit::new();
        let near = circuit.add_node("near").unwrap();
        let far = circuit.add_node("far").unwrap();
        circuit
            .add_device(Box::new(
                Cpl::new(
                    "p1",
                    "cable",
                    vec![near, NodeId::GROUND, far, NodeId::GROUND],
                    plan,
                )
                .unwrap(),
            ))
            .unwrap();
        let system = circuit.linear_system().unwrap();
        assert_eq!(system.a.get(3, 2), -1e-4);
    }
}

#[test]
fn dc_uses_each_diagonal_resistance_without_dropping_mutual_model_terms() {
    let plan = CplPlan::prepare(
        3,
        2.,
        &[1., 0.25, 0.4, 2., 0.7, 3.],
        &[100e-9, 2e-9, 3e-9, 110e-9, 4e-9, 120e-9],
        &[0., 0., 0.],
        &[10e-12, -1e-12, -1e-12, 10e-12, -1e-12, 10e-12],
    )
    .unwrap();
    assert_eq!(plan.matrix(CplMatrixKind::Resistance).get(0, 1), Some(0.25));
    let mut circuit = Circuit::new();
    let near = [
        circuit.add_node("near0").unwrap(),
        circuit.add_node("near1").unwrap(),
        circuit.add_node("near2").unwrap(),
    ];
    let far = [
        circuit.add_node("far0").unwrap(),
        circuit.add_node("far1").unwrap(),
        circuit.add_node("far2").unwrap(),
    ];
    circuit
        .add_device(Box::new(
            Cpl::new(
                "p3",
                "line",
                vec![
                    near[0],
                    near[1],
                    near[2],
                    NodeId::GROUND,
                    far[0],
                    far[1],
                    far[2],
                    NodeId::GROUND,
                ],
                plan,
            )
            .unwrap(),
        ))
        .unwrap();
    let system = circuit.linear_system().unwrap();
    assert_eq!(system.a.rows(), 12);
    for (i, expected) in [2., 4., 6.].into_iter().enumerate() {
        let first = 6 + 2 * i;
        let second = first + 1;
        assert_eq!(system.a.get(second, first), -expected);
        assert_eq!(system.a.get(second, i), 1.);
        assert_eq!(system.a.get(second, 3 + i), -1.);
        assert_eq!(system.a.get(first, second), 1.);
    }
    // Upstream C uses only R_ii for its DC equations, so no cross-wire
    // off-diagonal branch loads, even though the plan retains R_ij.
    assert_eq!(system.a.get(7, 8), 0.);
    assert_eq!(system.a.get(9, 6), 0.);
}

#[test]
fn explicit_near_and_far_returns_are_retained_but_not_stamped_by_c_dc() {
    let plan = CplPlan::prepare(1, 1., &[1.], &[100e-9], &[0.], &[10e-12]).unwrap();
    let mut circuit = Circuit::new();
    let a = circuit.add_node("a").unwrap();
    let ref1 = circuit.add_node("ref1").unwrap();
    let b = circuit.add_node("b").unwrap();
    let ref2 = circuit.add_node("ref2").unwrap();
    circuit
        .add_device(Box::new(
            Cpl::new("p1", "line", vec![a, ref1, b, ref2], plan).unwrap(),
        ))
        .unwrap();
    let system = circuit.linear_system().unwrap();
    assert_eq!(system.a.rows(), 6);
    for row in [1, 3] {
        for col in 0..6 {
            assert_eq!(system.a.get(row, col), 0.);
            assert_eq!(system.a.get(col, row), 0.);
        }
    }
}

#[test]
fn refuses_mismatched_terminal_groups_and_unrepresentable_series_resistance() {
    let plan = CplPlan::prepare(1, 1., &[1.], &[100e-9], &[0.], &[10e-12]).unwrap();
    assert!(Cpl::new("p1", "line", vec![NodeId::GROUND], plan).is_err());
    let overflow = CplPlan::prepare(1, 1e308, &[1e308], &[100e-9], &[0.], &[10e-12]).unwrap();
    assert!(Cpl::new("p1", "line", vec![NodeId::GROUND; 4], overflow,).is_err());
}
