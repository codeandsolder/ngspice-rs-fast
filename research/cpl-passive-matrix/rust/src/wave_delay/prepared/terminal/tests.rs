use crate::wave_delay::WaveNetwork;

fn two_mode_coupled() -> Result<WaveNetwork, &'static str> {
    let scattering = vec![
        0.1, 0.015, 0.8, 0.03, 0.015, 0.1, 0.03, 0.8, 0.8, 0.03, 0.1, 0.015, 0.03, 0.8, 0.015, 0.1,
    ];
    WaveNetwork::new(
        vec![1e-10, 1.3e-10],
        3,
        scattering,
        vec![0., 0.],
        vec![0., 0.],
        true,
    )
}

fn inverse_2x2(matrix: &[f64; 4]) -> [f64; 4] {
    let det = matrix[0].mul_add(matrix[3], -matrix[1] * matrix[2]);
    [
        matrix[3] / det,
        -matrix[1] / det,
        -matrix[2] / det,
        matrix[0] / det,
    ]
}
fn apply(matrix: &[f64; 4], input: &[f64; 2]) -> [f64; 2] {
    [
        matrix[0].mul_add(input[0], matrix[1] * input[1]),
        matrix[2].mul_add(input[0], matrix[3] * input[1]),
    ]
}

#[test]
fn modal_equations_hold_for_every_coupled_near_far_newton_trial() -> Result<(), &'static str> {
    let line = two_mode_coupled()?;
    let prepared = line.prepare_step(1.7e-9)?;
    let equations = prepared.modal_terminal_equations()?;
    assert_eq!(equations.modes(), 2);
    let width = 4;
    for incident in [
        [0.3, -0.25, 0.5, 0.12],
        [-1.2, 0.2, 0.75, 0.3],
        [0., 0., 0., 0.],
    ] {
        let out = prepared.evaluate(&incident[..2], &incident[2..])?;
        let emitted = [out.reflected_near(), out.transmitted_far()].concat();
        let mut alpha = vec![0.; width];
        let mut beta = vec![0.; width];
        for k in 0..width {
            alpha[k] = incident[k] + emitted[k];
            beta[k] = incident[k] - emitted[k];
        }
        for row in 0..width {
            let mut residual = 0.;
            for col in 0..width {
                residual =
                    equations.voltage_matrix()[row * width + col].mul_add(alpha[col], residual);
                residual =
                    equations.current_matrix()[row * width + col].mul_add(beta[col], residual);
            }
            assert!(
                (residual - equations.rhs()[row]).abs() < 2e-12,
                "modal MNA row {row}: lhs={residual}, rhs={}",
                equations.rhs()[row]
            );
        }
    }
    Ok(())
}

#[test]
fn fully_coupled_physical_mna_rows_stamps_both_returns_and_branch_currents()
-> Result<(), &'static str> {
    let line = two_mode_coupled()?;
    let prepared = line.prepare_step(1.9e-9)?;
    let eq = prepared.modal_terminal_equations()?;
    let voltage_inv = [1.3, 0.22, -0.04, 0.85];
    let current_inv = [0.9, -0.1, 0.2, 1.1];
    let physical = eq.project_physical(&voltage_inv, &current_inv)?;
    let pv = inverse_2x2(&voltage_inv);
    let pi = inverse_2x2(&current_inv);
    let incident = [0.6, 0.25, -0.3, 0.5];
    let predicted = prepared.evaluate(&incident[..2], &incident[2..])?;
    let emitted = [predicted.reflected_near(), predicted.transmitted_far()].concat();
    let mut voltage = [0.; 4];
    let mut current = [0.; 4];
    for end in 0..2 {
        let alpha = [
            incident[2 * end] + emitted[2 * end],
            incident[2 * end + 1] + emitted[2 * end + 1],
        ];
        let beta = [
            incident[2 * end] - emitted[2 * end],
            incident[2 * end + 1] - emitted[2 * end + 1],
        ];
        voltage[2 * end..2 * end + 2].copy_from_slice(&apply(&pv, &alpha));
        current[2 * end..2 * end + 2].copy_from_slice(&apply(&pi, &beta));
    }
    let recovered = physical.incident_from_physical(&voltage, &current)?;
    for (&actual, &expected) in recovered.iter().zip(&incident) {
        assert!((actual - expected).abs() < 5e-13);
    }
    let evaluated = prepared.evaluate(&recovered[..2], &recovered[2..])?;
    for (&actual, &expected) in evaluated
        .transmitted_far()
        .iter()
        .zip(predicted.transmitted_far())
    {
        assert!((actual - expected).abs() < 5e-13);
    }
    let signals = [Some(0), Some(1), Some(2), Some(3)];
    let returns = [Some(8), None, Some(9), None];
    let branches = [4, 5, 6, 7];
    let stamp = physical.mna_stamp(&signals, &returns, &branches)?;
    let mut unknown = [0.; 10];
    unknown[8] = 0.12;
    unknown[9] = -0.06;
    for col in 0..4 {
        unknown[col] = voltage[col] + returns[col].map_or(0., |index| unknown[index]);
        unknown[branches[col]] = current[col];
    }
    let mut residual = [0.; 10];
    for term in &stamp.coefficients {
        residual[term.row] = term
            .coefficient
            .mul_add(unknown[term.col], residual[term.row]);
    }
    for term in &stamp.rhs {
        residual[term.row] -= term.value;
    }
    for (row, &value) in residual[4..8].iter().enumerate() {
        assert!(
            value.abs() < 3e-12,
            "physical MNA branch row={row}: {value}"
        );
    }
    for col in 0..4 {
        assert!((residual[col] - current[col]).abs() < 1e-12);
    }
    assert!((residual[8] + current[0]).abs() < 1e-12);
    assert!((residual[9] + current[2]).abs() < 1e-12);
    assert_eq!(physical.modes(), 2);
    assert_eq!(physical.rhs(), eq.rhs());
    assert_eq!(physical.current_matrix().len(), 16);
    assert_eq!(physical.voltage_matrix().len(), 16);
    Ok(())
}

#[test]
fn invalid_physical_projection_and_reused_current_indices_are_rejected() -> Result<(), &'static str>
{
    let line = two_mode_coupled()?;
    let eq = line.prepare_step(1e-9)?.modal_terminal_equations()?;
    let mat = [1., 0., 0., 1.];
    assert!(eq.project_physical(&mat[..3], &mat).is_err());
    assert!(eq.project_physical(&[f64::NAN, 0., 0., 1.], &mat).is_err());
    let physical = eq.project_physical(&mat, &mat)?;
    assert!(
        physical
            .mna_stamp(&[None; 3], &[None; 4], &[0, 1, 2, 3])
            .is_err()
    );
    assert!(
        physical
            .mna_stamp(&[None; 4], &[None; 4], &[0, 1, 1, 3])
            .is_err()
    );
    Ok(())
}

#[test]
fn inversion_free_grounded_end_branch_constraint_remains_representable() -> Result<(), &'static str>
{
    // A static reflection -1 is an ideal short. No matrix inverse of
    // (I+S) should ever be attempted by branch-equation formation.
    let line = WaveNetwork::new(
        vec![1e-10],
        1,
        vec![-1., 0., 0., -1.],
        vec![0.],
        vec![0.],
        true,
    )?;
    let eq = line.prepare_step(2e-9)?.modal_terminal_equations()?;
    assert_eq!(eq.modes(), 1);
    assert!(eq.current_matrix().iter().all(|v| v.is_finite()));
    assert!(eq.voltage_matrix().iter().all(|v| v.is_finite()));
    let physical = eq.project_physical(&[1.], &[1.])?;
    let stamp = physical.mna_stamp(&[Some(0), Some(1)], &[None, None], &[2, 3])?;
    assert!(stamp.coefficients.iter().all(|v| v.coefficient.is_finite()));
    Ok(())
}
