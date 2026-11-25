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
                conductors: 0,
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
                workspace_bytes: 0,
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

fn run_transient_cpl(body: &str) -> ngspice_rs::analysis::Plot {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;
    let text = format!("CPL transient\n{body}\n.end\n");
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("cpl_transient.cir"), &text))
        .unwrap();
    let config = RunConfig::from_netlist(&parsed).unwrap();
    let plan = config.plan_for(&parsed.analyses[0]).unwrap();
    let mut circuit = config.circuit(&parsed).unwrap();
    plan.run(&mut circuit, &config.context()).unwrap()
}

fn transient_column(plot: &ngspice_rs::analysis::Plot, name: &str) -> Vec<f64> {
    plot.column(name)
        .unwrap()
        .iter()
        .map(|value| value.re)
        .collect()
}

#[test]
fn coupled_multiconductor_transient_preserves_initial_dc_steady_state() {
    // With constant source, the accepted t=0 DC fluxes/charges initialize
    // the entire spatial ladder, so the first transient step contains no
    // unphysical impulse, and repeated transient integration remains steady.
    let plot = run_transient_cpl(
        "vin source 0 dc 1\n\
        rsource source a 50\n\
        pline a 0 b 0 cable\n\
        rload b 0 50\n\
        .model cable cpl r=10 l=250n c=100p g=0 length=1 sections=16\n\
        .tran 1n 20n",
    );
    let voltages = transient_column(&plot, "v(b)");
    assert!(voltages.len() >= 10);
    let expected = 50. / 110.;
    for (index, &voltage) in voltages.iter().enumerate() {
        assert!(
            (voltage - expected).abs() < 1e-8,
            "steady DC voltage at point {index}: {voltage} vs {expected}"
        );
    }
}

#[test]
fn coupled_multiconductor_transient_propagates_a_delayed_pulse_causally() {
    // 50-ohm matched distributed line: delay=length*sqrt(LC)=1 ns,
    // independent of the section setting. The far end cannot respond before the pulse and
    // should approach the half-amplitude matched limit after the delay.
    let plot = run_transient_cpl(
        "vin source 0 pulse(0 1 2n 0.5n 0.5n 3n 8n)\n\
        rs source a 50\n\
        p1 a 0 b 0 cable\n\
        rl b 0 50\n\
        .model cable cpl r=0 l=250n c=100p g=0 length=0.2 sections=32\n\
        .tran 0.1n 6n",
    );
    let values = transient_column(&plot, "v(b)");
    let times = transient_column(&plot, "time");
    assert_eq!(values.len(), times.len());
    assert!(values.len() > 25);
    for (&time, &voltage) in times.iter().zip(&values) {
        if time <= 3e-9 {
            assert!(
                voltage.abs() < 1e-10,
                "noncausal response at {time}: {voltage}"
            );
        }
    }
    let reached = times
        .iter()
        .zip(&values)
        .filter(|(time, _)| **time > 3.5e-9 && **time < 5.5e-9)
        .map(|(_, value)| *value)
        .fold(0.0_f64, f64::max);
    assert!(
        (reached - 0.5).abs() < 0.06,
        "matched pulse peak: {reached}"
    );
}

#[test]
fn four_conductor_lc_corpus_transient_against_c_reference() {
    // Adapted only at the source/load boundary from ngspice's
    // tests/transmission/cpl3_4_line.cir. The R/L/G/C per-length upper
    // triangles and 6.3 length are exactly those of the C fixture.
    // C ngspice 47 generated the comparison samples on 2026-11-24.
    let plot = run_transient_cpl(
        "vin in 0 pulse(0 1 1n 0.2n 0.2n 5n 12n)\n\
        rs in a1 100\n\
        r2 a2 0 100\n\
        r3 a3 0 100\n\
        r4 a4 0 100\n\
        p1 a1 a2 a3 a4 0 b1 b2 b3 b4 0 coupled\n\
        rl1 b1 0 100\n\
        rl2 b2 0 100\n\
        rl3 b3 0 100\n\
        rl4 b4 0 100\n\
        .model coupled cpl\n\
        + r=0.3 0 0 0 0.3 0 0 0.3 0 0.3\n\
        + l=9n 5.4n 0 0 9n 5.4n 0 9n 5.4n 9n\n\
        + g=0 0 0 0 0 0 0 0 0 0\n\
        + c=0.35p -0.03p 0 0 0.35p -0.03p 0 0.35p -0.03p 0.35p\n\
        + length=6.3 sections=64\n\
        .tran 0.05n 6n 0 0.05n",
    );
    let times = transient_column(&plot, "time");
    assert!(times.len() > 60);
    // Interpolated from the 134-row ngspice 47 run on the source/loads
    // above. This tests the C waveform shape, while the separate physical
    // DC/AC oracles remain authoritative for continuous-line accuracy.
    const SAMPLES: [f64; 6] = [1.2e-9, 1.6e-9, 2e-9, 3e-9, 4e-9, 5e-9];
    const C_WAVEFORMS: [(&str, [f64; 6]); 4] = [
        (
            "v(a1)",
            [
                0.572_275_5,
                0.601_133_1,
                0.529_333_3,
                0.502_224_1,
                0.500_068_1,
                0.499_862_6,
            ],
        ),
        (
            "v(b1)",
            [
                0.031_441_0,
                0.438_039_2,
                0.469_346_5,
                0.496_472_3,
                0.499_911_6,
                0.500_247_9,
            ],
        ),
        (
            "v(b2)",
            [
                -0.050_908_0,
                -0.061_744_5,
                -0.030_107_0,
                -0.004_432_8,
                -0.000_593_8,
                -0.000_089_3,
            ],
        ),
        (
            "v(b4)",
            [
                -0.031_441_0,
                -0.000_384_0,
                0.005_661_5,
                -0.000_270_8,
                -0.000_275_8,
                -0.000_018_2,
            ],
        ),
    ];
    for (name, reference) in C_WAVEFORMS {
        let values = transient_column(&plot, name);
        assert_eq!(values.len(), times.len());
        assert!(values.iter().all(|v| v.is_finite()));
        for (&sample, &expected) in SAMPLES.iter().zip(&reference) {
            let k = times
                .partition_point(|&time| time < sample)
                .clamp(1, times.len() - 1);
            let value = values[k - 1]
                + (values[k] - values[k - 1]) * (sample - times[k - 1]) / (times[k] - times[k - 1]);
            assert!(
                (value - expected).abs() < 0.010,
                "{name} at {sample:.5e}: Rust={value}, C={expected}"
            );
        }
    }
    // 6.3 * 0.3 ohm adds real cable resistance. In contrast to C's
    // ~0.500 V asymptote, Rust's ~0.495 V is the independent physical
    // DC divider: 100 / (100 + 6.3 * 0.3 + 100).
    let physical_dc = 100. / (201.89);
    let last = *transient_column(&plot, "v(b1)").last().unwrap();
    assert!(
        (last - physical_dc).abs() < 1e-3,
        "physical DC limit: {last}"
    );
}

#[test]
fn coupled_transient_gear_history_is_causal_and_matches_trapezoidal_waveform() {
    const BODY: &str = "vin source 0 pulse(0 1 2n 0.5n 0.5n 3n 8n)\n\
        rs source a 50\n\
        p1 a 0 b 0 cable\n\
        rl b 0 50\n\
        .model cable cpl r=0 l=250n c=100p g=0 length=0.2 sections=32\n\
        .tran 0.05n 6n 0 0.05n";
    let trap = run_transient_cpl(BODY);
    let gear = run_transient_cpl(&BODY.replace(".tran", ".options method=gear maxord=2\n.tran"));
    let trap_time = transient_column(&trap, "time");
    let trap_out = transient_column(&trap, "v(b)");
    let gear_time = transient_column(&gear, "time");
    let gear_out = transient_column(&gear, "v(b)");
    let interpolate = |time: f64, times: &[f64], values: &[f64]| {
        let index = times
            .partition_point(|&x| x < time)
            .clamp(1, times.len() - 1);
        values[index - 1]
            + (values[index] - values[index - 1]) * (time - times[index - 1])
                / (times[index] - times[index - 1])
    };
    for sample in [1e-9, 2e-9, 2.5e-9, 3e-9, 3.5e-9, 4e-9, 5e-9] {
        let trap_v = interpolate(sample, &trap_time, &trap_out);
        let gear_v = interpolate(sample, &gear_time, &gear_out);
        let tolerance = if sample <= 2e-9 { 1e-7 } else { 0.05 };
        assert!(
            (trap_v - gear_v).abs() < tolerance,
            "Gear/trap at {sample}: {gear_v} versus {trap_v}"
        );
    }
}

#[test]
fn extremely_attenuating_passive_line_retains_finite_physical_norton_y() {
    // The full 1000-metre DC ABCD has exp(1000) terms and overflows f64.
    // The physical port Y does not: for R=G=1 (per metre) its infinite-
    // line characteristic admittance is exactly 1 siemens, and its
    // end-to-end coupling decays to zero.
    let line = model(1, 1000., &[1.], &[1e-6], &[1.], &[1e-9]);
    let admittance = line.admittance_at(0.).unwrap();
    for diagonal in [0, 1] {
        near(
            admittance.get(diagonal, diagonal).unwrap(),
            Complex::real(1.),
            3e-12,
        );
    }
    for (r, c) in [(0, 1), (1, 0)] {
        near(admittance.get(r, c).unwrap(), Complex::ZERO, 3e-12);
    }
    assert!(
        line.transfer_at(0.).is_err(),
        "full ABCD is not representable even when physical Norton Y is"
    );
}

#[test]
fn noncommuting_long_line_y_satisfies_matrix_characteristic_riccati() {
    // At DC a semi-infinite reciprocal line has Yc R Yc = G.
    // R and G here do *not* commute: no independent scalar modal
    // decomposition by raw conductor can satisfy this identity.
    let line = model(
        2,
        1000.,
        &[2., 0.2, 1.],
        &[1e-6, 0., 2e-6],
        &[1., 0.1, 3.],
        &[1e-9, 0., 2e-9],
    );
    let y = line.admittance_at(0.).unwrap();
    let mut characteristic = ngspice_rs::maths::dense_complex::DenseComplex::zeros(2).unwrap();
    let mut resistance = ngspice_rs::maths::dense_complex::DenseComplex::zeros(2).unwrap();
    let expected = [[1., 0.1], [0.1, 3.]];
    let series = [[2., 0.2], [0.2, 1.]];
    for row in 0..2 {
        for col in 0..2 {
            characteristic
                .set(row, col, y.get(row, col).unwrap())
                .unwrap();
            resistance
                .set(row, col, Complex::real(series[row][col]))
                .unwrap();
        }
    }
    let result = characteristic
        .mul(&resistance)
        .unwrap()
        .mul(&characteristic)
        .unwrap();
    for row in 0..2 {
        for col in 0..2 {
            near(
                result.get(row, col).unwrap(),
                Complex::real(expected[row][col]),
                1e-9,
            );
            near(y.get(row, col + 2).unwrap(), Complex::ZERO, 2e-11);
            near(y.get(row + 2, col).unwrap(), Complex::ZERO, 2e-11);
            near(y.get(row, col).unwrap(), y.get(col, row).unwrap(), 1e-10);
        }
    }
}

#[test]
fn very_long_passive_line_solves_native_dc_and_ac_without_transfer_overflow() {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;
    let deck = "1000 metre passive line\n\
        v1 source 0 dc 1 ac 1\n\
        rs source a 10\n\
        p1 a 0 b 0 cable\n\
        rl b 0 10\n\
        .model cable cpl r=1 l=1u c=1n g=1 length=1000\n\
        .op\n.ac lin 1 100 100\n.end\n";
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("long_cpl.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&parsed).unwrap();
    for plan in config.plans_for(&parsed.analyses).unwrap() {
        let mut circuit = config.circuit(&parsed).unwrap();
        let plot = plan.run(&mut circuit, &config.context()).unwrap();
        let near_voltage = plot.value("v(a)", 0).unwrap();
        let far_voltage = plot.value("v(b)", 0).unwrap();
        if plan.kind() == ngspice_rs::primitives::AnalysisKind::OperatingPoint {
            near(near_voltage, Complex::real(1. / 11.), 1e-10);
        } else {
            assert!((near_voltage.magnitude() - 1. / 11.).abs() < 1e-5);
        }
        near(far_voltage, Complex::ZERO, 1e-8);
    }
}

#[test]
fn lossy_coupled_three_conductor_y_is_reciprocal_and_positive_real() {
    // All material matrices are symmetric passive, but their eigenvectors
    // differ. For arbitrary complex port voltages the dissipated power is
    // Re(V* Y V)>=0 and the full physical Y remains reciprocal.
    let line = model(
        3,
        0.65,
        &[2., 0.3, 0.1, 1., 0.2, 1.5],
        &[7e-9, 1e-9, 0.5e-9, 8e-9, 0.7e-9, 6e-9],
        &[0.001, 0.0002, 0., 0.002, 0.0001, 0.003],
        &[3e-12, -0.2e-12, 0., 2.5e-12, -0.3e-12, 4e-12],
    );
    let stimuli = [
        [
            Complex::new(1., 0.25),
            Complex::new(-0.3, 0.4),
            Complex::new(0.7, -0.6),
            Complex::new(0.2, 0.1),
            Complex::new(-0.5, -0.3),
            Complex::new(0.1, 0.8),
        ],
        [
            Complex::real(1.),
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
            Complex::ZERO,
        ],
    ];
    for hz in [0., 20e6, 250e6, 1e9] {
        let y = line.admittance_at(hz).unwrap();
        for row in 0..6 {
            for col in 0..6 {
                near(y.get(row, col).unwrap(), y.get(col, row).unwrap(), 1e-9);
            }
        }
        for voltage in stimuli {
            let mut power = 0.;
            let mut absolute = 0.;
            for row in 0..6 {
                let mut current = Complex::ZERO;
                for col in 0..6 {
                    current = current + y.get(row, col).unwrap() * voltage[col];
                }
                power += voltage[row].re * current.re + voltage[row].im * current.im;
                absolute += voltage[row].magnitude() * current.magnitude();
            }
            assert!(
                power >= -1e-10 * absolute.max(1e-12),
                "negative dissipated power {power} versus absolute {absolute} at {hz} Hz"
            );
        }
    }
}

#[test]
fn ideal_series_voltage_constraints_with_large_shunt_remain_exact() {
    // Zero R and L means zero total series impedance at DC. There is no
    // finite Y-matrix, but the conductor is still an ideal wire with
    // distributed shunt G*length; native branch MNA is the correct form.
    let plan = model(1, 1000., &[0.], &[0.], &[1.], &[1e-9]);
    assert!(plan.admittance_at(0.).is_err());
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;
    let deck = "ideal zero series cable\n\
        v1 source 0 dc 1\n\
        rs source a 10\n\
        p1 a 0 b 0 cable\n\
        rl b 0 10\n\
        .model cable cpl r=0 l=0 c=1n g=1 length=1000\n\
        .op\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("ideal_series.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let plot = config.plans_for(&netlist.analyses).unwrap()[0]
        .run(&mut circuit, &config.context())
        .unwrap();
    near(
        plot.value("v(a)", 0).unwrap(),
        Complex::real(1. / 10002.),
        2e-9,
    );
    near(
        plot.value("v(b)", 0).unwrap(),
        Complex::real(1. / 10002.),
        2e-9,
    );
}

#[test]
fn ideal_zero_series_transient_matches_exact_lumped_shunt() {
    // An ideal series-short cable with arbitrary C/G has no finite
    // two-ended Y, yet its transient constraint is exactly valid. A
    // distributed zero-impedance scalar line is a wire and total shunt
    // conductance/capacitance, independent of section count.
    let cable = run_transient_cpl(
        "v1 src 0 pulse(0 1 2n 0.25n 0.25n 3n 8n)\n\
         rs src a 100\n\
         p1 a 0 b 0 cable\n\
         rl b 0 100\n\
         .model cable cpl r=0 l=0 c=100p g=0.002 length=1 sections=8\n\
         .tran 0.1n 16n 0 0.1n",
    );
    let lumped = run_transient_cpl(
        "v1 src 0 pulse(0 1 2n 0.25n 0.25n 3n 8n)\n\
         rs src b 100\n\
         rl b 0 100\n\
         rsh b 0 500\n\
         csh b 0 100p\n\
         .tran 0.1n 16n 0 0.1n",
    );
    let expected_times = transient_column(&lumped, "time");
    let expected_voltages = transient_column(&lumped, "v(b)");
    let actual_times = transient_column(&cable, "time");
    let actual_voltages = transient_column(&cable, "v(b)");
    let interpolation = |time: f64, times: &[f64], values: &[f64]| {
        let position = times
            .partition_point(|&candidate| candidate < time)
            .clamp(1, times.len() - 1);
        values[position - 1]
            + (values[position] - values[position - 1]) * (time - times[position - 1])
                / (times[position] - times[position - 1])
    };
    for sample in [
        1e-9, 2e-9, 2.5e-9, 3e-9, 4e-9, 5e-9, 6e-9, 7e-9, 10e-9, 12e-9, 15e-9,
    ] {
        let expected = interpolation(sample, &expected_times, &expected_voltages);
        let actual = interpolation(sample, &actual_times, &actual_voltages);
        assert!(
            (actual - expected).abs() < 2e-5,
            "zero-Z distributed cable and exactly equivalent lumped shunt at {sample}: \
             {actual} vs {expected}"
        );
    }
}

#[test]
fn reciprocal_lossless_modes_have_distinct_exact_wavefronts_and_matched_amplitudes() {
    // L=50² C, hence both independently known even/odd impedances are 50 Ω.
    // C_even=60 pF/m, C_odd=140 pF/m, length=0.2 m: 0.6/1.4 ns delays.
    // Exciting only conductor 1 launches equal even and odd waves; their
    // sum/difference at the far end is not a decoupled-line approximation.
    let plot = run_transient_cpl(
        "vin source 0 pulse(0 1 2n 0.4n 0.4n 3n 10n)\n\
         rs source a1 50\n\
         rz a2 0 50\n\
         p1 a1 a2 0 b1 b2 0 cable\n\
         rl1 b1 0 50\n\
         rl2 b2 0 50\n\
         .model cable cpl r=0 0 0 g=0 0 0 l=250n -100n 250n c=100p -40p 100p length=0.2\n\
         .tran 0.025n 6n 0 0.025n",
    );
    let times = transient_column(&plot, "time");
    let b1 = transient_column(&plot, "v(b1)");
    let b2 = transient_column(&plot, "v(b2)");
    let pulse =
        |t: f64| ((t - 2e-9) / 0.4e-9).clamp(0., 1.) - ((t - 5.4e-9) / 0.4e-9).clamp(0., 1.);
    assert!(times.len() > 100);
    for ((&time, &first), &second) in times.iter().zip(&b1).zip(&b2) {
        let even = pulse(time - 0.6e-9);
        let odd = pulse(time - 1.4e-9);
        let first_expected = 0.25 * (even + odd);
        let second_expected = 0.25 * (even - odd);
        assert!(
            (first - first_expected).abs() < 2e-8,
            "b1 at {time:.17e}: {first:.17e} vs {first_expected:.17e}"
        );
        assert!(
            (second - second_expected).abs() < 2e-8,
            "b2 at {time:.17e}: {second:.17e} vs {second_expected:.17e}"
        );
    }
}

#[test]
fn lossless_modal_history_starts_at_dc_bias_without_an_artificial_wavefront() {
    let plot = run_transient_cpl(
        "vin source 0 dc 1\n\
         rs source a 50\n\
         p1 a 0 b 0 cable\n\
         rl b 0 75\n\
         .model cable cpl r=0 g=0 l=250n c=100p length=0.2\n\
         .tran 0.05n 5n",
    );
    let voltages = transient_column(&plot, "v(b)");
    assert!(voltages.len() > 10);
    for voltage in voltages {
        assert!(
            (voltage - 0.6).abs() < 1e-10,
            "DC initialized wave: {voltage:.17e}"
        );
    }
}

#[test]
fn noncommuting_three_conductor_modal_transport_matches_independent_reciprocal_basis() {
    // Construct independent power-normalized modal bases V = 8 B and
    // I = B^(-T)/8, B lower triangular with B21=B32=0.2, B31=0.04.
    // L = V diag(1,2,3) V^T nH, C = I diag(1,2,3) I^T nF.
    // The 1 m cable has three known unequal delays and noncommuting L,C.
    // Both ends are terminated in Y=I I^T using physical resistors, and
    // a 20 mA Norton source on conductor 3 excites all three modes.
    let plot = run_transient_cpl(
        "iin 0 a3 pulse(0 0.02 1n 0.4n 0.4n 4n 20n)\n\
         ra12 a1 a2 320\nra23 a2 a3 320\n\
         ra1 a1 0 76.19047619047619\nra2 a2 0 100\nra3 a3 0 80\n\
         rb12 b1 b2 320\nrb23 b2 b3 320\n\
         rb1 b1 0 76.19047619047619\nrb2 b2 0 100\nrb3 b3 0 80\n\
         p1 a1 a2 a3 0 b1 b2 b3 0 cable\n\
         .model cable cpl r=0 0 0 0 0 0 g=0 0 0 0 0 0\n\
         + l=64n 12.8n 2.56n 130.56n 26.112n 197.2224n\n\
         + c=16.875p -6.25p 0 33.125p -9.375p 46.875p length=1\n\
         .tran 0.025n 8n 0 0.025n",
    );
    let times = transient_column(&plot, "time");
    let outputs = ["v(b1)", "v(b2)", "v(b3)"].map(|name| transient_column(&plot, name));
    let pulse =
        |t: f64| ((t - 1e-9) / 0.4e-9).clamp(0., 1.) - ((t - 5.4e-9) / 0.4e-9).clamp(0., 1.);
    assert!(times.len() > 200);
    for (index, &time) in times.iter().enumerate() {
        let [a, b, c] = [1e-9, 2e-9, 3e-9].map(|delay| pulse(time - delay));
        let expected = [
            0.0256 * a,
            0.00512 * a + 0.128 * b,
            0.001024 * a + 0.0256 * b + 0.64 * c,
        ];
        for (output, target) in outputs.iter().zip(expected) {
            let actual = output[index];
            assert!(
                (actual - target).abs() < 2e-8,
                "noncommuting three-mode transport at {time:.17e}: {actual:.17e} vs {target:.17e}"
            );
        }
    }
}

#[test]
fn mismatched_modal_line_preserves_bidirectional_reflection_series() {
    // For 50 ohms, source 25 ohms and load 100 ohms, the source/load
    // reflection coefficients are -1/3 and +1/3, respectively. The
    // first far-end amplitude is 8/9; each round trip multiplies by -1/9.
    let plot = run_transient_cpl(
        "vin source 0 pulse(0 1 1n 0.4n 0.4n 20n 40n)\n\
         rs source a 25\n\
         p1 a 0 b 0 cable\n\
         rl b 0 100\n\
         .model cable cpl r=0 g=0 l=250n c=100p length=0.2\n\
         .tran 0.025n 12n 0 0.025n",
    );
    let times = transient_column(&plot, "time");
    let values = transient_column(&plot, "v(b)");
    assert!(times.len() > 300);
    for (&time, &actual) in times.iter().zip(&values) {
        let expected = (0_i32..6).fold(0., |sum, round_trip| {
            let delay = (1. + 2. * f64::from(round_trip)) * 1e-9;
            let pulse = ((time - delay - 1e-9) / 0.4e-9).clamp(0., 1.);
            ((8. / 9.) * (-1.0_f64 / 9.).powi(round_trip)).mul_add(pulse, sum)
        });
        assert!(
            (actual - expected).abs() < 2e-8,
            "multiple reflection at {time:.17e}: {actual:.17e} vs {expected:.17e}"
        );
    }
}

#[test]
fn partially_singular_series_matches_independent_modal_pi_and_ideal_shunt() {
    // Rank-one coupled R and L: the common mode sees 20 ohms and 100 nH,
    // while the differential mode is an ideal series constraint. Both
    // modes have 100 pF and 2 mS shunt loading over the entire cable.
    // Compare against separately elaborated ordinary R/L/C primitives;
    // the reference shares neither the hybrid nor CPL stamping kernel.
    let cable = run_transient_cpl(
        "v1 src 0 pulse(0 1 2n 0.25n 0.25n 3n 8n)\n\
         rs1 src a1 100\nrs2 a2 0 100\n\
         p1 a1 a2 0 b1 b2 0 cable\n\
         rl1 b1 0 100\nrl2 b2 0 100\n\
         .model cable cpl r=10 10 10 l=50n 50n 50n c=100p 0 100p g=0.002 0 0.002 length=1 sections=4\n\
         .tran 0.02n 16n 0 0.02n",
    );
    let reference = run_transient_cpl(
        "ve es 0 pulse(0 0.5 2n 0.25n 0.25n 3n 8n)\n\
         vo os 0 pulse(0 0.5 2n 0.25n 0.25n 3n 8n)\n\
         rse es e0 100\nrso os o 100\n\
         r0 e0 m0 5\nl0 m0 e1 25n\n\
         r1 e1 m1 5\nl1 m1 e2 25n\n\
         r2 e2 m2 5\nl2 m2 e3 25n\n\
         r3 e3 m3 5\nl3 m3 e4 25n\n\
         c0 e0 0 12.5p\nrg0 e0 0 4000\n\
         c1 e1 0 25p\nrg1 e1 0 2000\n\
         c2 e2 0 25p\nrg2 e2 0 2000\n\
         c3 e3 0 25p\nrg3 e3 0 2000\n\
         c4 e4 0 12.5p\nrg4 e4 0 4000\n\
         co o 0 100p\nro o 0 500\n\
         rle e4 0 100\nrlo o 0 100\n\
         .tran 0.02n 16n 0 0.02n",
    );
    let times = transient_column(&reference, "time");
    let even = transient_column(&reference, "v(e4)");
    let odd = transient_column(&reference, "v(o)");
    let actual_times = transient_column(&cable, "time");
    let first = transient_column(&cable, "v(b1)");
    let second = transient_column(&cable, "v(b2)");
    let interpolate = |time: f64, values: &[f64]| {
        let j = times
            .partition_point(|&candidate| candidate < time)
            .clamp(1, times.len() - 1);
        let weight = (time - times[j - 1]) / (times[j] - times[j - 1]);
        (values[j] - values[j - 1]).mul_add(weight, values[j - 1])
    };
    for ((&time, &a), &b) in actual_times.iter().zip(&first).zip(&second) {
        let e = interpolate(time, &even);
        let o = interpolate(time, &odd);
        assert!(
            (a - e - o).abs() < 2e-5 && (b - e + o).abs() < 2e-5,
            "partially singular coupled cable at {time:.17e}: ({a:.17e},{b:.17e}) vs ({:.17e},{:.17e})",
            e + o,
            e - o
        );
    }
}

#[test]
fn long_rank_one_dc_series_has_exact_ideal_mode_without_transfer_overflow() {
    // Even mode exp(-1000) attenuation, odd mode an ideal wire shunted
    // by 1000 S. Native DC initialization must preserve both at once.
    let plot = run_transient_cpl(
        "v1 src 0 1\nrs1 src a1 1\nrs2 a2 0 1\n\
         p1 a1 a2 0 b1 b2 0 cable\n\
         rl1 b1 0 1\nrl2 b2 0 1\n\
         .model cable cpl r=0.5 0.5 0.5 l=0 0 0 c=0 0 0 g=1 0 1 length=1000 sections=8\n\
         .tran 0.1n 1n 0 0.1n",
    );
    let a = transient_column(&plot, "v(b1)");
    let b = transient_column(&plot, "v(b2)");
    let expected = 0.5 / 1002.;
    assert!(
        (a[0] - expected).abs() < 1e-12 && (b[0] + expected).abs() < 1e-12,
        "DC mixed ideal/attenuating modes: {}, {}",
        a[0],
        b[0]
    );
    for (&a, &b) in a.iter().zip(&b) {
        assert!(
            (a - b - 2. * expected).abs() < 1e-12,
            "exact ideal mode changed: {a}, {b}"
        );
    }
}

#[test]
fn modal_trials_are_repeatable_across_rejection_and_history_pruning() {
    use ngspice_rs::devices::{
        AnalysisMode, Circuit, DelayAcceptance, LoadRequest, ModelContext, StateHistory,
    };
    use ngspice_rs::maths::{SparseMatrix, Vector};

    let plan = Arc::new(model(
        2,
        0.2,
        &[0.; 3],
        &[250e-9, 30e-9, 300e-9],
        &[0.; 3],
        &[100e-12, -15e-12, 80e-12],
    ));
    let modes = plan.exact_lc_modes().unwrap();
    let shortest = modes
        .total_delays
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let longest = modes.total_delays.iter().copied().fold(0., f64::max);
    let dt = shortest * 0.25;
    let mut circuit = Circuit::new();
    let input = [
        circuit.add_node("a1").unwrap(),
        circuit.add_node("a2").unwrap(),
    ];
    let output = [
        circuit.add_node("b1").unwrap(),
        circuit.add_node("b2").unwrap(),
    ];
    circuit
        .add_device(Box::new(
            CoupledLine::new(
                "p1",
                plan,
                input.to_vec(),
                NodeId::GROUND,
                output.to_vec(),
                NodeId::GROUND,
            )
            .unwrap(),
        ))
        .unwrap();
    circuit.finalize().unwrap();
    let dimension = circuit.unknown_count();
    assert_eq!(dimension, 8);
    assert_eq!(circuit.state_len(), 0);

    let load = |history: &StateHistory, mode, solution: &Vector| {
        let mut trial = history.trial();
        let mut matrix = SparseMatrix::new(dimension, dimension);
        let mut rhs = Vector::zeros(dimension);
        circuit
            .load(
                &LoadRequest {
                    mode,
                    solution,
                    model_context: &ModelContext::default(),
                    integration: None,
                    history,
                    forcing: None,
                },
                &mut matrix,
                &mut rhs,
                &mut trial,
            )
            .unwrap();
        (trial, matrix, rhs)
    };
    let acceptance = |start| DelayAcceptance {
        steps: [dt; 3],
        min_break: dt * 1e-6,
        start,
        reltol: 1e-3,
        abstol: 1e-12,
    };
    let mut history = circuit.state_history().unwrap();
    let initial = Vector::from_slice(&[0.1, 0.2, 0.3, 0.4, 0.01, -0.02, 0.03, -0.04]);
    let (trial, _, _) = load(&history, AnalysisMode::OperatingPoint, &initial);
    circuit
        .accept_transient_point(&initial, 0., &mut history, trial, &acceptance(true))
        .unwrap();

    for step in 1..=100 {
        let time = f64::from(step) * dt;
        let solution = Vector::from_slice(&[time / dt, 0.2, 0.3, 0.4, 0.01, -0.02, 0.03, -0.04]);
        let previous = history.clone();
        let mode = AnalysisMode::Transient { time, dt };
        let (_, before_matrix, before_rhs) = load(&history, mode, &solution);
        // Oversized, discarded trial exercises the affine implicit modal
        // stamp rather than only the accepted-sample interpolation path.
        let rejected_dt = 2. * longest;
        let rejected_time = time - dt + rejected_dt;
        let (_, _, _) = load(
            &history,
            AnalysisMode::Transient {
                time: rejected_time,
                dt: rejected_dt,
            },
            &initial,
        );
        assert_eq!(history, previous);
        let (trial, after_matrix, after_rhs) = load(&history, mode, &solution);
        assert_eq!(after_matrix, before_matrix);
        assert_eq!(after_rhs, before_rhs);
        circuit
            .accept_transient_point(&solution, time, &mut history, trial, &acceptance(false))
            .unwrap();
        let wave = history.delay(0).unwrap().as_waveform().unwrap();
        assert_eq!(wave.last_time(), Some(time));
        assert!(wave.time(0).unwrap() <= time - longest);
        assert!(wave.len() <= 2 + (longest / dt).ceil() as usize);
    }
}

#[test]
fn matched_noncommuting_modal_cable_does_not_propagate_roundoff_as_breakpoints() {
    // Independent reciprocal bases with unequal delays 0.7 ns and 1.1 ns.
    // Both ends have the matched physical admittance I I^T. The far-end
    // incident waves therefore cancel, even though their individual
    // physical voltage/current summands are not zero.
    let plot = run_transient_cpl(
        "i0 0 a0 pwl(0 0 .5n 0 .7n -4.02490707249183229e-03 2.1n -4.02490707249183229e-03 2.4n 0 5n 0)\n\
         i1 0 a1 pwl(0 0 .5n 0 .7n 1.85040525873951187e-03 2.1n 1.85040525873951187e-03 2.4n 0 5n 0)\n\
         rag0 a0 0 2.98673262776638602e+01\nra0x1 a0 a1 2.64995705127014446e+02\nrag1 a1 0 3.19199052608936995e+01\n\
         rbg0 b0 0 2.98673262776638602e+01\nrb0x1 b0 b1 2.64995705127014446e+02\nrbg1 b1 0 3.19199052608936995e+01\n\
         p1 a0 a1 0 b0 b1 0 cable\n\
         .model cable cpl r=0 0 0 g=0 0 0 l=9.55723104263021698e-08 1.60457956954219321e-08 1.58410964411988147e-07\n\
         + c=1.30392673177890839e-10 -1.32077612288939703e-11 1.92296847325850063e-10 length=0.2\n\
         .tran 0.05n 5n",
    );
    let times = transient_column(&plot, "time");
    let first = transient_column(&plot, "v(b0)");
    let second = transient_column(&plot, "v(b1)");
    assert!(
        times.len() < 1000,
        "roundoff caused {} accepted transient points",
        times.len()
    );
    let pulse =
        |t: f64| ((t - 0.5e-9) / 0.2e-9).clamp(0., 1.) - ((t - 2.1e-9) / 0.3e-9).clamp(0., 1.);
    for ((&t, &a), &b) in times.iter().zip(&first).zip(&second) {
        let p = pulse(t - 0.7e-9);
        let q = pulse(t - 1.1e-9);
        assert!(
            (a - (-0.054018275760412925_f64).mul_add(p, 0.002104498808104603 * q)).abs() < 5e-9
        );
        assert!((b - 0.0207765131828798 * q).abs() < 5e-9);
    }
}

#[test]
fn matched_full_order_cable_preserves_all_wavefronts_without_event_explosion() {
    use std::fmt::Write as _;

    // Generate independent reciprocal bases. I is lower bidiagonal, so
    // the matched physical port admittance I I^T is tridiagonal and can
    // be assembled using only ordinary resistors. V=I^(-T) is dense
    // upper triangular; the resulting L and C do not commute.
    for n in [8_usize, 16, 32] {
        let index = |i: usize| f64::from(u32::try_from(i).unwrap());
        let diagonal: Vec<_> = (0..n)
            .map(|i| 0.13 + 0.006 * index((5 * i + 3) % 11))
            .collect();
        let sub: Vec<_> = (0..n)
            .map(|i| {
                if i == 0 {
                    0.
                } else {
                    -0.015 - 0.002 * index((7 * i + 2) % 13)
                }
            })
            .collect();
        let slowness: Vec<_> = (0..n)
            .map(|i| (3. + 4. * (index((7 * i) % n) + 0.25) / index(n)) * 1e-9)
            .collect();
        let source: Vec<_> = (0..n)
            .map(|i| 0.001 * (index((11 * i + 5) % 19) - 9.))
            .collect();
        let mut current = vec![vec![0.; n]; n];
        let mut voltage = vec![vec![0.; n]; n];
        for (i, row) in current.iter_mut().enumerate() {
            row[i] = diagonal[i];
            if i > 0 {
                row[i - 1] = sub[i];
            }
        }
        for j in 0..n {
            for i in (0..=j).rev() {
                let next = if i + 1 < n {
                    sub[i + 1] * voltage[i + 1][j]
                } else {
                    0.
                };
                voltage[i][j] = (f64::from(u8::from(i == j)) - next) / diagonal[i];
            }
        }
        let packed_material = |basis: &[Vec<f64>]| {
            let mut entries = Vec::new();
            for (i, left) in basis.iter().enumerate() {
                for right in &basis[i..] {
                    let element = left
                        .iter()
                        .zip(right)
                        .zip(&slowness)
                        .map(|((&a, &b), &s)| a * b * s)
                        .sum::<f64>();
                    entries.push(format!("{element:.17e}"));
                }
            }
            entries.join(" ")
        };
        let mut deck = String::new();
        for (i, &amplitude) in source.iter().enumerate() {
            writeln!(
                deck,
                "i{i} 0 a{i} pwl(0 0 .5n 0 .7n {amplitude:.17e} 2.1n {amplitude:.17e} 2.4n 0 5n 0)"
            )
            .unwrap();
        }
        for end in ['a', 'b'] {
            for i in 0..n {
                let shunt = diagonal[i] * diagonal[i]
                    + sub[i] * sub[i]
                    + if i > 0 { sub[i] * diagonal[i - 1] } else { 0. }
                    + if i + 1 < n {
                        sub[i + 1] * diagonal[i]
                    } else {
                        0.
                    };
                assert!(shunt > 0.);
                writeln!(deck, "r{end}g{i} {end}{i} 0 {:.17e}", 1. / shunt).unwrap();
                if i > 0 {
                    writeln!(
                        deck,
                        "r{end}x{i} {end}{} {end}{i} {:.17e}",
                        i - 1,
                        -1. / (sub[i] * diagonal[i - 1])
                    )
                    .unwrap();
                }
            }
        }
        let terminals = |end: char| {
            (0..n)
                .map(|i| format!("{end}{i}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let zero = vec!["0"; n * (n + 1) / 2].join(" ");
        writeln!(
            deck,
            "p1 {} 0 {} 0 cable\n.model cable cpl r={zero} g={zero} l={} c={} length=0.2\n.tran 0.05n 5n",
            terminals('a'),
            terminals('b'),
            packed_material(&voltage),
            packed_material(&current),
        )
        .unwrap();

        let plot = run_transient_cpl(&deck);
        let times = transient_column(&plot, "time");
        assert!(
            (100..1500).contains(&times.len()),
            "{n} conductors produced {} accepted points",
            times.len()
        );
        let launch: Vec<_> = (0..n)
            .map(|mode| {
                0.5 * voltage
                    .iter()
                    .zip(&source)
                    .map(|(row, &drive)| row[mode] * drive)
                    .sum::<f64>()
            })
            .collect();
        let pulse =
            |t: f64| ((t - 0.5e-9) / 0.2e-9).clamp(0., 1.) - ((t - 2.1e-9) / 0.3e-9).clamp(0., 1.);
        for (conductor, row) in voltage.iter().enumerate() {
            let actual = transient_column(&plot, &format!("v(b{conductor})"));
            for (&time, &observed) in times.iter().zip(&actual) {
                let expected = row
                    .iter()
                    .zip(&launch)
                    .zip(&slowness)
                    .map(|((&basis, &amplitude), &delay)| {
                        basis * amplitude * pulse(time - 0.2 * delay)
                    })
                    .sum::<f64>();
                assert!(
                    (observed - expected).abs() < 5e-9,
                    "{n} conductors, output {conductor} at {time:.17e}: {observed:.17e} vs {expected:.17e}"
                );
            }
        }
    }
}

#[test]
fn long_rank_deficient_cable_preserves_ideal_ac_mode_and_attenuated_mode() {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;

    // Common mode Z=1+jw*1u, Y=1+jw*1n: a matched order-one
    // impedance line with >1000 nepers of attenuation. Differential
    // mode Z=0, Y=1+jw*1n is an exact equipotential shunt line.
    // A source at a1 excites both modes equally. The ideal mode is a
    // single voltage shared across the cable, with shunt 1000*Y and
    // two one-ohm end terminations. No finite nodal Y exists.
    let deck = "long rank-one coupled line\n\
         v1 src 0 dc 1 ac 1\nrs1 src a1 1\nrs2 a2 0 1\n\
         p1 a1 a2 0 b1 b2 0 cable\nrl1 b1 0 1\nrl2 b2 0 1\n\
         .model cable cpl r=0.5 0.5 0.5 l=0.5u 0.5u 0.5u g=1 0 1 c=1n 0 1n length=1000 sections=8\n\
         .ac lin 5 100 100meg\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("rank_one_ac.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let plot = config.plans_for(&netlist.analyses).unwrap()[0]
        .run(&mut circuit, &config.context())
        .unwrap();
    for point in 0..5 {
        let hz = 100. + f64::from(u32::try_from(point).unwrap()) * (100e6 - 100.) / 4.;
        let omega = std::f64::consts::TAU * hz;
        let shunt = Complex::new(1., omega * 1e-9);
        let ideal = Complex::real(0.5) / (Complex::real(2.) + Complex::real(1000.) * shunt);
        let a1 = plot.value("v(a1)", point).unwrap();
        let a2 = plot.value("v(a2)", point).unwrap();
        near((a1 - a2) * Complex::real(0.5), ideal, 2e-10);
        near(plot.value("v(b1)", point).unwrap(), ideal, 2e-10);
        near(plot.value("v(b2)", point).unwrap(), -ideal, 2e-10);
    }
}

#[test]
fn ac_scattering_of_nilpotent_series_and_shunt_is_exact() {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;
    for (material, shunt) in [
        ("r=1000 l=1m g=0 c=0", false),
        ("r=0 l=0 g=1000 c=1m", true),
    ] {
        let deck = format!(
            "nilpotent cable\nv1 src 0 dc 1 ac 1\nrs src a 1\n\
             p1 a 0 b 0 cable\nrl b 0 1\n\
             .model cable cpl {material} length=1000 sections=8\n\
             .ac lin 3 100 100meg\n.end\n"
        );
        let netlist = Parser::new()
            .parse_deck(&parse_deck_text(Path::new("nilpotent_ac.cir"), &deck))
            .unwrap();
        let config = RunConfig::from_netlist(&netlist).unwrap();
        let mut circuit = config.circuit(&netlist).unwrap();
        let plot = config.plans_for(&netlist.analyses).unwrap()[0]
            .run(&mut circuit, &config.context())
            .unwrap();
        for point in 0..3 {
            let hz = 100. + f64::from(u32::try_from(point).unwrap()) * (100e6 - 100.) / 2.;
            let immittance = Complex::new(1e6, std::f64::consts::TAU * hz);
            let far = Complex::real(1.) / (Complex::real(2.) + immittance);
            near(plot.value("v(b)", point).unwrap(), far, 2e-10);
            let expected_near = if shunt { far } else { Complex::real(1.) - far };
            near(plot.value("v(a)", point).unwrap(), expected_near, 2e-10);
        }
    }
}

#[test]
fn noncommuting_singular_ac_cable_retains_exact_voltage_constraint() {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;

    // Z is rank one in the physical basis; the shunt matrix does not
    // commute with it. V2 is spatially constant. Shift W = V1 + q V2,
    // q = Y12/Y11, to obtain an independent scalar telegrapher equation
    // for W and I1. Integrating the I2 equation then gives a lumped
    // shunt L*(Y22-Y12²/Y11) plus q times the net I1 entering the cable.
    // The scalar mode has >1000 nepers of loss throughout this sweep.
    let deck = "noncommuting ideal/attenuating line\n\
        v1 src 0 dc 1 ac 1\nrs1 src a1 1\nrs2 a2 0 1\n\
        p1 a1 a2 0 b1 b2 0 cable\nrl1 b1 0 1\nrl2 b2 0 1\n\
        .model cable cpl r=1 0 0 l=1u 0 0 g=2 0.5 1 c=2n -0.25n 1n length=1000 sections=8\n\
        .ac lin 5 100 100meg\n.end\n";
    let parsed = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("singular_noncommuting.cir"),
            deck,
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&parsed).unwrap();
    let mut circuit = config.circuit(&parsed).unwrap();
    let plot = config.plans_for(&parsed.analyses).unwrap()[0]
        .run(&mut circuit, &config.context())
        .unwrap();
    let one = Complex::real(1.);
    let two = Complex::real(2.);
    for point in 0..5 {
        let omega = std::f64::consts::TAU * plot.value("frequency", point).unwrap().re;
        let z = Complex::new(1., omega * 1e-6);
        let y11 = Complex::new(2., omega * 2e-9);
        let y12 = Complex::new(0.5, -omega * 0.25e-9);
        let y22 = Complex::new(1., omega * 1e-9);
        let q = y12 / y11;
        let effective = y22 - y12 * q;
        let ratio = y11 / z;
        let real = (0.5 * (ratio.magnitude() + ratio.re)).sqrt();
        let characteristic = Complex::new(real, ratio.im / (2. * real));
        let current = characteristic / (one + characteristic);
        let ideal = -q * current / (two + Complex::real(1000.) * effective + two * q * q * current);
        near(plot.value("v(a2)", point).unwrap(), ideal, 2e-10);
        near(plot.value("v(b2)", point).unwrap(), ideal, 2e-10);
        near(
            plot.value("v(b1)", point).unwrap(),
            -q * ideal * current,
            2e-10,
        );
        near(
            plot.value("v(a1)", point).unwrap(),
            (one + q * ideal) / (one + characteristic) - q * ideal,
            2e-10,
        );
    }
}
