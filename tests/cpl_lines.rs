//! Independent analytic distributed multiconductor transmission-line checks.
//!
//! Includes differential/common eigenmodes, actual cross-conductor coupling,
//! zero-frequency resistive coupling, and explicit invalid model rejection.
use std::sync::Arc;

use ngspice_rs::devices::MnaUnknowns;
use ngspice_rs::devices::cpl::CoupledLine;
use ngspice_rs::devices::cpl::plan::{CoupledLineLimits, CoupledLinePlan};
use ngspice_rs::maths::Matrix;
use ngspice_rs::primitives::{Complex, NodeId, NodeTable};

fn model(n: usize, length: f64, r: &[f64], l: &[f64], g: &[f64], c: &[f64]) -> CoupledLinePlan {
    CoupledLinePlan::from_packed(n, length, r, l, g, c, CoupledLineLimits::default()).unwrap()
}
#[track_caller]
fn near(actual: Complex, expected: Complex, relative: f64) {
    let scale = expected.magnitude().max(1e-3);
    let error = (actual - expected).magnitude();
    assert!(
        error <= relative * scale,
        "got {actual:?} instead of {expected:?}, absolute error={error}, relative={}",
        error / scale
    );
}
fn analytic_lossless(l: f64, c: f64, length: f64, hz: f64) -> (Complex, Complex) {
    let phase = std::f64::consts::TAU * hz * length * (l * c).sqrt();
    let y0 = (c / l).sqrt();
    (
        Complex::imaginary(-phase.cos() / phase.sin() * y0),
        Complex::imaginary(phase.sin().recip() * y0),
    )
}

#[test]
fn scalar_lossless_line_matches_exact_trigonometric_solution() {
    let line = model(1, 1., &[0.], &[8e-9], &[0.], &[3e-12]);
    for hz in [100e6, 300e6, 1e9, 1.5e9] {
        let y = line.admittance_at(hz).unwrap();
        let (diagonal, far) = analytic_lossless(8e-9, 3e-12, 1., hz);
        near(y.get(0, 0).unwrap(), diagonal, 2e-11);
        near(y.get(0, 1).unwrap(), far, 2e-11);
        near(y.get(1, 0).unwrap(), far, 2e-11);
        near(y.get(1, 1).unwrap(), diagonal, 2e-11);
    }
}

#[test]
fn dc_series_resistance_couples_both_ends_correctly() {
    // A positive definite R and zero G give Ydc = [[K,-K],[-K,K]],
    // K = inverse(R*length). This must hold even if L,C are coupled.
    let length = 2.;
    let plan = model(
        2,
        length,
        &[3., 1., 4.],
        &[9e-9, 2e-9, 5e-9],
        &[0., 0., 0.],
        &[4e-12, -0.5e-12, 3e-12],
    );
    let y = plan.admittance_at(0.).unwrap();
    let inv_det = 1. / (3. * 4. - 1.);
    let expected = [[4. * inv_det, -inv_det], [-inv_det, 3. * inv_det]];
    for (i, row) in expected.iter().enumerate() {
        for (j, &entry) in row.iter().enumerate() {
            let physical = entry / length;
            near(y.get(i, j).unwrap(), Complex::real(physical), 2e-12);
            near(y.get(i, j + 2).unwrap(), Complex::real(-physical), 2e-12);
            near(y.get(i + 2, j).unwrap(), Complex::real(-physical), 2e-12);
            near(y.get(i + 2, j + 2).unwrap(), Complex::real(physical), 2e-12);
        }
    }
}

#[test]
fn coupled_2line_even_odd_modes_match_independent_orthonormal_transform() {
    // U[:,0]=(1,1)/sqrt(2); U[:,1]=(1,-1)/sqrt(2).
    // L_even=8nH, L_odd=4nH, C_even=2.5pF, C_odd=3.5pF.
    let line = model(
        2,
        1.,
        &[0., 0., 0.],
        &[6e-9, 2e-9, 6e-9],
        &[0., 0., 0.],
        &[3e-12, -0.5e-12, 3e-12],
    );
    for hz in [50e6, 300e6, 1e9, 1.5e9] {
        let matrix = line.admittance_at(hz).unwrap();
        let (even_d, even_o) = analytic_lossless(8e-9, 2.5e-12, 1., hz);
        let (odd_d, odd_o) = analytic_lossless(4e-9, 3.5e-12, 1., hz);
        let expected_diag = (even_d + odd_d) * Complex::real(0.5);
        let expected_cross = (even_d - odd_d) * Complex::real(0.5);
        let expected_far_diag = (even_o + odd_o) * Complex::real(0.5);
        let expected_far_cross = (even_o - odd_o) * Complex::real(0.5);
        for (a, b, target) in [
            (0, 0, expected_diag),
            (1, 1, expected_diag),
            (0, 1, expected_cross),
            (1, 0, expected_cross),
            (2, 2, expected_diag),
            (3, 3, expected_diag),
            (2, 3, expected_cross),
            (3, 2, expected_cross),
            (0, 2, expected_far_diag),
            (1, 3, expected_far_diag),
            (0, 3, expected_far_cross),
            (1, 2, expected_far_cross),
            (2, 0, expected_far_diag),
            (3, 1, expected_far_diag),
            (2, 1, expected_far_cross),
            (3, 0, expected_far_cross),
        ] {
            near(matrix.get(a, b).unwrap(), target, 3e-10);
        }
    }
}

#[test]
fn uncoupled_multiple_conductors_do_not_generate_false_crosstalk() {
    let plan = model(
        3,
        0.7,
        &[0.25, 0., 0., 0.5, 0., 0.75],
        &[4e-9, 0., 0., 6e-9, 0., 8e-9],
        &[0., 0., 0., 0., 0., 0.],
        &[2e-12, 0., 0., 3e-12, 0., 4e-12],
    );
    let y = plan.admittance_at(1e9).unwrap();
    for i in 0..6 {
        for j in 0..6 {
            if i % 3 != j % 3 {
                near(y.get(i, j).unwrap(), Complex::ZERO, 2e-10);
            }
        }
    }
}

#[test]
fn model_rejects_nonphysical_singular_and_oversized_materials() {
    let limit = CoupledLineLimits::default();
    assert!(CoupledLinePlan::from_packed(0, 1., &[], &[], &[], &[], limit).is_err());
    assert!(
        CoupledLinePlan::from_packed(
            2,
            1.,
            &[1., 0.],
            &[1., 0., 1.],
            &[0., 0., 0.],
            &[1., 0., 1.],
            limit
        )
        .is_err()
    );
    assert!(CoupledLinePlan::from_packed(1, 0., &[1.], &[1.], &[0.], &[1.], limit).is_err());
    assert!(CoupledLinePlan::from_packed(1, f64::NAN, &[1.], &[1.], &[0.], &[1.], limit).is_err());
    assert!(CoupledLinePlan::from_packed(1, 1., &[-1.], &[1.], &[0.], &[1.], limit).is_err());
    assert!(CoupledLinePlan::from_packed(1, 1., &[1.], &[-1.], &[0.], &[1.], limit).is_err());
    assert!(CoupledLinePlan::from_packed(1, 1., &[1.], &[1.], &[-1.], &[1.], limit).is_err());
    assert!(CoupledLinePlan::from_packed(1, 1., &[1.], &[1.], &[0.], &[-1.], limit).is_err());
    assert!(
        CoupledLinePlan::from_packed(
            1,
            1.,
            &[1.],
            &[1.],
            &[0.],
            &[1.],
            CoupledLineLimits {
                max_conductors: 0,
                ..limit
            }
        )
        .is_err()
    );
    assert!(
        CoupledLinePlan::from_packed(
            1,
            1.,
            &[1.],
            &[1.],
            &[0.],
            &[1.],
            CoupledLineLimits {
                max_workspace_bytes: 0,
                ..limit
            }
        )
        .is_err()
    );
    assert!(
        CoupledLinePlan::from_packed(1, 1., &[1.], &[1.], &[0.], &[1.], limit)
            .unwrap()
            .admittance_at(f64::MAX)
            .is_err()
    );
    let mut asymmetric = Matrix::zeros(2, 2).unwrap();
    asymmetric.set(0, 0, 3.).unwrap();
    asymmetric.set(1, 1, 3.).unwrap();
    asymmetric.set(0, 1, 1.).unwrap();
    asymmetric.set(1, 0, 0.).unwrap();
    let l = CoupledLinePlan::unpack_symmetric(2, &[8e-9, 0., 8e-9], limit).unwrap();
    let g = CoupledLinePlan::unpack_symmetric(2, &[0., 0., 0.], limit).unwrap();
    let c = CoupledLinePlan::unpack_symmetric(2, &[3e-12, 0., 3e-12], limit).unwrap();
    assert!(CoupledLinePlan::new(1., asymmetric, l, g, c, limit).is_err());
}

#[test]
fn singular_ideal_dc_line_is_not_silently_replaced_with_an_open() {
    let line = model(1, 0.5, &[0.], &[1e-6], &[0.], &[1e-9]);
    assert!(line.admittance_at(0.).is_err());
    assert!(line.admittance_at(1e6).is_ok());
}

#[test]
fn distinct_input_and_output_return_nodes_stamp_exact_kcl_signs() {
    let line = Arc::new(model(
        2,
        2.,
        &[3., 0., 4.],
        &[9e-9, 0., 5e-9],
        &[0., 0., 0.],
        &[4e-12, 0., 3e-12],
    ));
    let device = CoupledLine::new(
        "p1",
        line,
        vec![NodeId::new(1), NodeId::new(2)],
        NodeId::new(3),
        vec![NodeId::new(4), NodeId::new(5)],
        NodeId::new(6),
    )
    .unwrap();
    let mut nodes = NodeTable::new();
    for i in 1..=6 {
        nodes.intern(&format!("node{i}")).unwrap();
    }
    let mut mna = MnaUnknowns::new();
    mna.rebuild(&nodes);
    let stamps = device.nodal_stamps_at(0., &mna).unwrap();
    let mut dense = [[0.; 6]; 6];
    for (row, col, value) in stamps {
        assert_eq!(value.im, 0.);
        dense[row][col] += value.re;
    }
    // Every row and column sums to zero under a global voltage shift:
    // returns are not silently replaced by ground.
    for row in &dense {
        near(Complex::real(row.iter().sum()), Complex::ZERO, 2e-10);
    }
    for col in 0..6 {
        let sum = dense.iter().map(|row| row[col]).sum::<f64>();
        near(Complex::real(sum), Complex::ZERO, 2e-10);
    }
    near(Complex::real(dense[0][0]), Complex::real(1. / 6.), 2e-12);
    near(Complex::real(dense[0][2]), Complex::real(-1. / 6.), 2e-12);
    near(Complex::real(dense[0][3]), Complex::real(-1. / 6.), 2e-12);
    near(Complex::real(dense[0][5]), Complex::real(1. / 6.), 2e-12);
}

#[test]
fn native_cpl_parser_preserves_conductor_and_return_order() {
    use ngspice_rs::netlist::ast::ParameterKind;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;
    let deck = "native coupled pair\n\
        p1 in1 in2 retin out1 out2 retout coupled len=0.3048\n\
        .model coupled cpl R=0.5 0 0.5 L=247.3n 31.65n 247.3n\n\
        + C=31.4p -2.45p 31.4p G=0 0 0 length=1\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("cpl.cir"), deck))
        .unwrap();
    assert_eq!(netlist.devices.len(), 1);
    assert_eq!(
        netlist.devices[0].nodes,
        ["in1", "in2", "retin", "out1", "out2", "retout"]
    );
    assert_eq!(netlist.devices[0].model.as_deref(), Some("coupled"));
    assert_eq!(netlist.models.len(), 1);
    assert_eq!(netlist.models[0].base, "cpl");
    for name in ["r", "l", "g", "c"] {
        let parameter = netlist.models[0]
            .parameters
            .iter()
            .find(|p| p.name == name)
            .unwrap();
        let ParameterKind::NumericVector(entries) = &parameter.kind else {
            panic!("{name} not vector")
        };
        assert_eq!(entries.len(), 3);
    }
    // Writer output is also a valid native CPL deck, with the complete
    // positioned matrix coefficient vectors in the same conductor order.
    let written = ngspice_rs::netlist::writer::write_netlist(&netlist).unwrap();
    let reloaded = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("cpl_rewritten.cir"), &written))
        .unwrap();
    assert!(ngspice_rs::netlist::semantic_eq(&netlist, &reloaded));
}

#[test]
fn native_p_device_solves_dc_and_ac_with_model_and_instance_length_override() {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;
    let deck = "p device native\n\
        v1 in 0 dc 1 ac 1\n\
        rload out 0 100\n\
        p1 in 0 out 0 coupled len 1\n\
        .model coupled cpl R=10 L=8n C=3p G=0 length=0.5\n\
        .op\n.ac lin 1 1e6 1e6\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("cpl_native.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let analytic = model(1, 1., &[10.], &[8e-9], &[0.], &[3e-12]);
    let y = analytic.admittance_at(1e6).unwrap();
    let y21 = y.get(1, 0).unwrap();
    let y22 = y.get(1, 1).unwrap();
    let expected = -y21 / (y22 + Complex::real(0.01));
    for plan in config.plans_for(&netlist.analyses).unwrap() {
        let mut circuit = config.circuit(&netlist).unwrap();
        let output = plan.run(&mut circuit, &config.context()).unwrap();
        if plan.kind() == ngspice_rs::primitives::AnalysisKind::OperatingPoint {
            near(
                output.value("v(out)", 0).unwrap(),
                Complex::real(100. / 110.),
                1e-11,
            );
        } else {
            near(output.value("v(out)", 0).unwrap(), expected, 5e-11);
        }
    }
}

#[test]
fn arbitrary_noncommuting_rlgc_matches_independent_runge_kutta_propagation() {
    // The real R/L/G/C matrices have distinct eigenvectors. Exact modal
    // separation into scalar lines is impossible; independently integrate
    // the full first-order physical telegrapher system with RK4.
    let resistance = [[1.0, 0.3], [0.3, 2.0]];
    let inductance = [[100e-9, 22e-9], [22e-9, 150e-9]];
    let conductance = [[0.001, 0.00015], [0.00015, 0.003]];
    let capacitance = [[5e-12, -0.7e-12], [-0.7e-12, 8e-12]];
    let length = 0.2;
    let line = model(
        2,
        length,
        &[resistance[0][0], resistance[0][1], resistance[1][1]],
        &[inductance[0][0], inductance[0][1], inductance[1][1]],
        &[conductance[0][0], conductance[0][1], conductance[1][1]],
        &[capacitance[0][0], capacitance[0][1], capacitance[1][1]],
    );
    for hz in [1e6, 250e6, 1e9] {
        let omega = std::f64::consts::TAU * hz;
        let mut operator = [[Complex::ZERO; 4]; 4];
        for r in 0..2 {
            for c in 0..2 {
                operator[r][c + 2] = -Complex::new(resistance[r][c], omega * inductance[r][c]);
                operator[r + 2][c] = -Complex::new(conductance[r][c], omega * capacitance[r][c]);
            }
        }
        let derivative = |state: &[Complex; 4]| -> [Complex; 4] {
            std::array::from_fn(|r| {
                (0..4).fold(Complex::ZERO, |acc, c| acc + operator[r][c] * state[c])
            })
        };
        let mut reference = [[Complex::ZERO; 4]; 4];
        let steps = 4096usize;
        let delta = length / f64::from(u16::try_from(steps).unwrap());
        for input in 0..4 {
            let mut state = [Complex::ZERO; 4];
            state[input] = Complex::real(1.);
            for _ in 0..steps {
                let k1 = derivative(&state);
                let stage = std::array::from_fn(|k| state[k] + k1[k] * Complex::real(delta * 0.5));
                let k2 = derivative(&stage);
                let stage = std::array::from_fn(|k| state[k] + k2[k] * Complex::real(delta * 0.5));
                let k3 = derivative(&stage);
                let stage = std::array::from_fn(|k| state[k] + k3[k] * Complex::real(delta));
                let k4 = derivative(&stage);
                for row in 0..4 {
                    state[row] = state[row]
                        + (k1[row]
                            + k2[row] * Complex::real(2.)
                            + k3[row] * Complex::real(2.)
                            + k4[row])
                            * Complex::real(delta / 6.);
                }
            }
            for row in 0..4 {
                reference[row][input] = state[row];
            }
        }
        let (observed, scale) = line.transfer_at(hz).unwrap();
        for row in 0..4 {
            for col in 0..4 {
                let conversion = match (row < 2, col < 2) {
                    (true, false) => 1. / scale,
                    (false, true) => scale,
                    _ => 1.,
                };
                near(
                    observed.get(row, col).unwrap(),
                    reference[row][col] * Complex::real(conversion),
                    2e-8,
                );
            }
        }
    }
}

#[test]
fn coupled_plan_accepts_the_physical_rc_diffusion_limit() {
    // A purely RC distributed cable has a well-defined lossy transmission
    // function despite L=0. The scalar analytical admittance follows
    // gamma=sqrt(ZY), Y0=gamma/Z, Y11=Y0*coth(gamma*length),
    // Y12=-Y0*csch(gamma*length).
    let length = 0.2;
    let resistance = 10.;
    let capacitance = 1e-9;
    let cable = model(1, length, &[resistance], &[0.], &[0.], &[capacitance]);
    for hz in [1e3, 1e6, 10e6] {
        let omega = std::f64::consts::TAU * hz;
        let imaginary = omega * resistance * capacitance;
        let root = (imaginary / 2.).sqrt();
        let gamma = Complex::new(root, root);
        let distance = gamma * Complex::real(length);
        let sine = Complex::new(
            distance.re.sinh() * distance.im.cos(),
            distance.re.cosh() * distance.im.sin(),
        );
        let cosine = Complex::new(
            distance.re.cosh() * distance.im.cos(),
            distance.re.sinh() * distance.im.sin(),
        );
        let admittance = gamma / Complex::real(resistance);
        let diag = admittance * cosine / sine;
        let offdiag = -admittance / sine;
        let result = cable.admittance_at(hz).unwrap();
        near(result.get(0, 0).unwrap(), diag, 3e-11);
        near(result.get(1, 1).unwrap(), diag, 3e-11);
        near(result.get(0, 1).unwrap(), offdiag, 3e-11);
        near(result.get(1, 0).unwrap(), offdiag, 3e-11);
    }
}

#[test]
fn lossless_cpl_dc_short_is_exact_branch_constraint_while_ac_is_distributed() {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;
    let deck = "lossless cable with exact DC short\n\
        v1 in 0 dc 1 ac 1\n\
        rload out 0 100\n\
        p1 in 0 out 0 cable\n\
        .model cable cpl R=0 L=8n C=3p G=0 length=1\n\
        .op\n.ac lin 1 250e6 250e6\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("lossless_p.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let analytic = model(1, 1., &[0.], &[8e-9], &[0.], &[3e-12]);
    assert!(analytic.admittance_at(0.).is_err());
    let y = analytic.admittance_at(250e6).unwrap();
    let ac_expected = -y.get(1, 0).unwrap() / (y.get(1, 1).unwrap() + Complex::real(0.01));
    for plan in config.plans_for(&netlist.analyses).unwrap() {
        let mut circuit = config.circuit(&netlist).unwrap();
        let output = plan.run(&mut circuit, &config.context()).unwrap();
        if plan.kind() == ngspice_rs::primitives::AnalysisKind::OperatingPoint {
            near(output.value("v(out)", 0).unwrap(), Complex::real(1.), 1e-12);
        } else {
            near(output.value("v(out)", 0).unwrap(), ac_expected, 5e-10);
        }
    }
}
