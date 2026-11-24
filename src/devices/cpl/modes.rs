//! Reciprocal LC modal bases for physically exact transmission-line delays.
//!
//! The C-positive-definite metric reduces the arbitrary coupled Maxwell
//! line to a symmetric eigenproblem without assuming that L and C commute.
//! In particular, C=B*Bᵀ, H=Bᵀ*L*B=Q*diag(λ)*Qᵀ, and the reciprocal physical
//! voltage/current bases are B⁻ᵀQ and BQ. Power-normalizing those bases
//! yields passive modal travelling waves and exact delays length*sqrt(λ).
//! Degenerate RC/RL diffusion remains in the general nonmodal plan.
use super::plan::CoupledLinePlan;
use crate::primitives::numeric::{scale_power, split_power};
use crate::primitives::{Real, SpiceResult};

/// Eigenvectors are row-major physical conductor × mode. In every mode,
/// C*V=I, L*I=λV and Vᵀ*I=Identity. The listed characteristic admittance
/// is 1/sqrt(λ) in these metric-normalized coordinates, not the physical
/// conductor admittance. See the power normalization of PhysicalModalLine.
#[derive(Debug, Clone)]
pub struct CoupledLineModes {
    /// Physical signal conductors, excluding the two return terminals.
    pub conductors: usize,
    /// Reciprocal voltage transform, physical conductor × modal index.
    pub voltage_basis: Vec<Real>,
    /// Reciprocal current transform, physical conductor × modal index.
    pub current_basis: Vec<Real>,
    /// Positive travelling-wave current/voltage ratio in modal coordinates.
    pub characteristic_admittances: Vec<Real>,
    /// Exact individual LC propagation times across the entire cable.
    pub total_delays: Vec<Real>,
}

fn normalized_cholesky(c: &[Real], order: usize, scale: Real) -> SpiceResult<Vec<Real>> {
    let mut lower = vec![0.0_f64; order * order];
    for row in 0..order {
        for col in 0..=row {
            let mut value = c[row * order + col] / scale;
            for k in 0..col {
                value = (-lower[row * order + k]).mul_add(lower[col * order + k], value);
            }
            if row == col {
                if !value.is_finite() || value <= 0. {
                    return Err(super::invalid(
                        "LC modal propagation requires positive definite capacitance",
                    ));
                }
                lower[row * order + col] = value.sqrt();
            } else {
                let entry = value / lower[col * order + col];
                if !entry.is_finite() {
                    return Err(super::invalid("nonfinite normalized C Cholesky factor"));
                }
                lower[row * order + col] = entry;
            }
        }
    }
    Ok(lower)
}

/// Eigenvalues and eigenvectors of a small real symmetric matrix.
/// The scaled cyclic Jacobi updates preserve the columns of the eigenbasis;
/// a budget and residual check avoid silent acceptance of degenerate work.
fn symmetric_eigenvectors(mut a: Vec<Real>, order: usize) -> SpiceResult<(Vec<Real>, Vec<Real>)> {
    let mut vectors = vec![0.0_f64; order * order];
    for row in 0..order {
        vectors[row * order + row] = 1.;
    }
    let max = a.iter().fold(0.0_f64, |largest, &v| largest.max(v.abs()));
    if !max.is_finite() || max <= 0. {
        return Err(super::invalid("LC modal material has no positive spectrum"));
    }
    for value in &mut a {
        *value /= max;
    }
    // Treat only roundoff-level offdiagonal residuals as diagonal.
    let dimension = f64::from(
        u32::try_from(order).map_err(|_| super::invalid("LC mode dimension out of range"))?,
    );
    let tolerance = 8. * dimension * Real::EPSILON;
    let mut converged = false;
    for _sweep in 0..100 {
        let mut maximum = 0.0_f64;
        for row in 0..order {
            for col in row + 1..order {
                maximum = maximum.max(a[row * order + col].abs());
            }
        }
        if maximum <= tolerance {
            converged = true;
            break;
        }
        for first in 0..order {
            for second in first + 1..order {
                let ab = a[first * order + second];
                if ab == 0. {
                    continue;
                }
                let gap = a[second * order + second] - a[first * order + first];
                let twice = 2. * ab;
                let tangent = if gap == 0. {
                    1.
                } else {
                    let magnitude = twice.abs() / (gap.abs() + gap.hypot(twice));
                    if gap.is_sign_negative() == ab.is_sign_negative() {
                        magnitude
                    } else {
                        -magnitude
                    }
                };
                let cosine = 1. / tangent.hypot(1.);
                let sine = tangent * cosine;
                let ap = a[first * order + first];
                let aq = a[second * order + second];
                a[first * order + first] = ap - tangent * ab;
                a[second * order + second] = aq + tangent * ab;
                a[first * order + second] = 0.;
                a[second * order + first] = 0.;
                for row in 0..order {
                    if row != first && row != second {
                        let x = a[row * order + first];
                        let y = a[row * order + second];
                        let new_x = cosine.mul_add(x, -sine * y);
                        let new_y = sine.mul_add(x, cosine * y);
                        a[row * order + first] = new_x;
                        a[first * order + row] = new_x;
                        a[row * order + second] = new_y;
                        a[second * order + row] = new_y;
                    }
                    let x = vectors[row * order + first];
                    let y = vectors[row * order + second];
                    vectors[row * order + first] = cosine.mul_add(x, -sine * y);
                    vectors[row * order + second] = sine.mul_add(x, cosine * y);
                }
            }
        }
    }
    if !converged {
        return Err(super::invalid(
            "LC modal Jacobi decomposition did not converge",
        ));
    }
    let mut indices: Vec<_> = (0..order).collect();
    indices.sort_by(|&a_idx, &b_idx| a[a_idx * order + a_idx].total_cmp(&a[b_idx * order + b_idx]));
    let eigenvalues = indices
        .iter()
        .map(|&col| a[col * order + col] * max)
        .collect::<Vec<_>>();
    let mut reordered = vec![0.0_f64; order * order];
    for row in 0..order {
        for (mode, &col) in indices.iter().enumerate() {
            reordered[row * order + mode] = vectors[row * order + col];
        }
    }
    Ok((eigenvalues, reordered))
}

impl CoupledLinePlan {
    /// Exact LC modal basis and physical flight times, including all
    /// cross-conductor mixing. This is the input for the separate physically
    /// passive static-R/G, exact-delay scattering realization.
    ///
    /// # Errors
    /// L or C is singular/nonpositive, numerical orthogonality cannot be
    /// represented, a delay overflows/underflows, or eigenwork fails.
    pub fn exact_lc_modes(&self) -> SpiceResult<CoupledLineModes> {
        let n = self.conductors();
        let c = self.c.data();
        let l = self.l.data();
        let cscale = c.iter().fold(0.0_f64, |largest, &v| largest.max(v.abs()));
        let lscale = l.iter().fold(0.0_f64, |largest, &v| largest.max(v.abs()));
        if cscale <= 0. || lscale <= 0. {
            return Err(super::invalid(
                "LC modal delay requires nonzero inductance and capacitance",
            ));
        }
        let lower = normalized_cholesky(c, n, cscale)?;
        let mut product = vec![0.0_f64; n * n];
        let mut symmetric = vec![0.0_f64; n * n];
        for row in 0..n {
            for col in 0..n {
                let mut value = 0.;
                for k in 0..n {
                    value = (l[row * n + k] / lscale).mul_add(lower[k * n + col], value);
                }
                product[row * n + col] = value;
            }
        }
        for row in 0..n {
            for col in 0..n {
                let mut value = 0.;
                for k in 0..n {
                    value = lower[k * n + row].mul_add(product[k * n + col], value);
                }
                symmetric[row * n + col] = value;
            }
        }
        for row in 0..n {
            for col in 0..row {
                let averaged = symmetric[row * n + col].midpoint(symmetric[col * n + row]);
                symmetric[row * n + col] = averaged;
                symmetric[col * n + row] = averaged;
            }
        }
        let (normalized_eigenvalues, eigenvectors) = symmetric_eigenvectors(symmetric, n)?;
        let root_cscale = cscale.sqrt();
        let mut modes = CoupledLineModes {
            conductors: n,
            voltage_basis: vec![0.; n * n],
            current_basis: vec![0.; n * n],
            characteristic_admittances: Vec::with_capacity(n),
            total_delays: Vec::with_capacity(n),
        };
        for (mode, &lambda) in normalized_eigenvalues.iter().enumerate() {
            // Multiplication order must not overflow just because the
            // normalized eigenvalue exceeds one: the physical L*C may be
            // representable even when lambda*lscale is not.
            let (a, ae) = split_power(lambda);
            let (b, be) = split_power(lscale);
            let (c, ce) = split_power(cscale);
            let physical_lambda = scale_power(a * b * c, ae + be + ce);
            if !physical_lambda.is_finite() || physical_lambda <= 0. {
                return Err(super::invalid(
                    "LC modal eigenvalue cannot produce positive delay",
                ));
            }
            let speed_inverse = physical_lambda.sqrt();
            let admittance = 1. / speed_inverse;
            let delay = self.length * speed_inverse;
            if !delay.is_finite() || delay <= 0. || !admittance.is_finite() {
                return Err(super::invalid(
                    "LC modal delay/admittance not representable",
                ));
            }
            modes.characteristic_admittances.push(admittance);
            modes.total_delays.push(delay);
            for row in 0..n {
                let mut current = 0.;
                for col in 0..=row {
                    current = lower[row * n + col].mul_add(eigenvectors[col * n + mode], current);
                }
                modes.current_basis[row * n + mode] = current * root_cscale;
            }
            for row in (0..n).rev() {
                let mut voltage = eigenvectors[row * n + mode];
                for k in row + 1..n {
                    voltage = (-lower[k * n + row])
                        .mul_add(modes.voltage_basis[k * n + mode] * root_cscale, voltage);
                }
                modes.voltage_basis[row * n + mode] =
                    (voltage / lower[row * n + row]) / root_cscale;
            }
        }
        if modes
            .voltage_basis
            .iter()
            .chain(&modes.current_basis)
            .any(|value| !value.is_finite())
        {
            return Err(super::invalid("LC modal physical transforms are nonfinite"));
        }
        Ok(modes)
    }
}

#[cfg(test)]
mod tests {
    use super::CoupledLinePlan;
    use crate::devices::cpl::plan::CoupledLineLimits;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= tolerance,
            "{actual:.17e} vs {expected:.17e}, tolerance {tolerance:.3e}"
        );
    }

    #[test]
    fn exact_scalar_wave_speed_and_characteristic_impedance() {
        let line = CoupledLinePlan::from_packed(
            1,
            0.2,
            &[0.],
            &[250e-9],
            &[0.],
            &[100e-12],
            CoupledLineLimits::default(),
        )
        .unwrap();
        let modes = line.exact_lc_modes().unwrap();
        assert_eq!(modes.conductors, 1);
        close(modes.total_delays[0], 1e-9, 1e-24);
        let y = modes.characteristic_admittances[0];
        let pv = modes.voltage_basis[0] / y.sqrt();
        let pi = modes.current_basis[0] * y.sqrt();
        close(pv / pi, 50., 1e-12);
        close(modes.voltage_basis[0] * modes.current_basis[0], 1., 2e-15);
    }

    #[test]
    fn symmetric_two_wire_even_and_odd_have_distinct_exact_flights() {
        let line = CoupledLinePlan::from_packed(
            2,
            1.,
            &[0., 0., 0.],
            &[10e-9, 2e-9, 10e-9],
            &[0., 0., 0.],
            &[4e-12, -1e-12, 4e-12],
            CoupledLineLimits::default(),
        )
        .unwrap();
        let modes = line.exact_lc_modes().unwrap();
        close(modes.total_delays[0], (36e-21_f64).sqrt(), 2e-25);
        close(modes.total_delays[1], (40e-21_f64).sqrt(), 2e-25);
        assert!(modes.total_delays[0] < modes.total_delays[1]);
        for mode in 0..2 {
            let v0 = modes.voltage_basis[mode];
            let v1 = modes.voltage_basis[2 + mode];
            close(v0.abs(), v1.abs(), 1e-6);
            if mode == 0 {
                assert!(v0.signum() == v1.signum());
            } else {
                assert!(v0.signum() != v1.signum());
            }
        }
    }

    #[test]
    fn noncommuting_three_wire_lc_materials_preserve_physical_reciprocity() {
        let line = CoupledLinePlan::from_packed(
            3,
            0.78,
            &[0.; 6],
            &[7e-9, 1.1e-9, -0.3e-9, 8.5e-9, 0.6e-9, 6.1e-9],
            &[0.; 6],
            &[3.1e-12, -0.21e-12, 0.08e-12, 2.6e-12, -0.15e-12, 4.1e-12],
            CoupledLineLimits::default(),
        )
        .unwrap();
        let modes = line.exact_lc_modes().unwrap();
        let n = modes.conductors;
        for mode in 0..n {
            let lambda = (modes.total_delays[mode] / line.length).powi(2);
            for row in 0..n {
                let mut cv = 0.;
                let mut li = 0.;
                for k in 0..n {
                    cv += line.c.data()[row * n + k] * modes.voltage_basis[k * n + mode];
                    li += line.l.data()[row * n + k] * modes.current_basis[k * n + mode];
                }
                close(cv, modes.current_basis[row * n + mode], 1e-17);
                close(li, lambda * modes.voltage_basis[row * n + mode], 1e-13);
            }
        }
        for first in 0..n {
            for second in 0..n {
                let mut overlap = 0.;
                for row in 0..n {
                    overlap += modes.voltage_basis[row * n + first]
                        * modes.current_basis[row * n + second];
                }
                close(overlap, f64::from(u8::from(first == second)), 5e-12);
            }
        }
    }

    #[test]
    fn large_noncommuting_cables_reconstruct_every_physical_modal_equation() {
        use crate::maths::Matrix;
        for order in [3, 8, 16, 32] {
            let mut left = vec![0.0_f64; order * order];
            let mut right = vec![0.0_f64; order * order];
            for row in 0..order {
                for col in 0..order {
                    let signed = |x: usize| f64::from(u16::try_from(x % 11).unwrap()) - 5.;
                    left[row * order + col] = if row == col {
                        1.
                    } else {
                        0.012 * signed(row * 7 + col * 13 + 2)
                    };
                    right[row * order + col] = if row == col {
                        1.3
                    } else {
                        0.009 * signed(row * 17 + col * 3 + 4)
                    };
                }
            }
            let mut inductance = Matrix::zeros(order, order).unwrap();
            let mut capacitance = Matrix::zeros(order, order).unwrap();
            for row in 0..order {
                for col in 0..order {
                    let mut l = 0.;
                    let mut c = 0.;
                    for k in 0..order {
                        l += left[k * order + row] * left[k * order + col];
                        c += right[k * order + row] * right[k * order + col];
                    }
                    inductance.set(row, col, l * 4e-9).unwrap();
                    capacitance.set(row, col, c * 2e-12).unwrap();
                }
            }
            let line = CoupledLinePlan::new(
                0.83,
                Matrix::zeros(order, order).unwrap(),
                inductance,
                Matrix::zeros(order, order).unwrap(),
                capacitance,
                CoupledLineLimits::default(),
            )
            .unwrap();
            let modes = line.exact_lc_modes().unwrap();
            assert_eq!(modes.conductors, order);
            let mut summed_lambdas = 0.;
            for (mode, &delay) in modes.total_delays.iter().enumerate() {
                let lambda = (delay / line.length).powi(2);
                assert!(lambda.is_finite() && lambda > 0.);
                summed_lambdas += lambda;
                for row in 0..order {
                    let (mut cv, mut li) = (0.0_f64, 0.0_f64);
                    for k in 0..order {
                        cv +=
                            line.c.data()[row * order + k] * modes.voltage_basis[k * order + mode];
                        li +=
                            line.l.data()[row * order + k] * modes.current_basis[k * order + mode];
                    }
                    let expected_i = modes.current_basis[row * order + mode];
                    let expected_li = lambda * modes.voltage_basis[row * order + mode];
                    assert!(
                        (cv - expected_i).abs() <= 2e-11 * expected_i.abs().max(1e-14),
                        "C V=I: {order} conductors mode={mode} row={row}, residual={:.17e}",
                        cv - expected_i
                    );
                    assert!(
                        (li - expected_li).abs() <= 2e-11 * expected_li.abs().max(1e-23),
                        "L I=lambda V: {order} conductors mode={mode} row={row}, residual={:.17e}",
                        li - expected_li
                    );
                }
                for other in 0..order {
                    let mut overlap = 0.;
                    for row in 0..order {
                        overlap += modes.voltage_basis[row * order + mode]
                            * modes.current_basis[row * order + other];
                    }
                    assert!(
                        (overlap - f64::from(u8::from(mode == other))).abs() < 3e-11,
                        "modal reciprocity: order={order}, ({mode},{other}), overlap={overlap:.17e}"
                    );
                }
            }
            let mut trace_lc = 0.;
            for row in 0..order {
                for col in 0..order {
                    trace_lc += line.l.data()[row * order + col] * line.c.data()[col * order + row];
                }
            }
            assert!(
                (trace_lc - summed_lambdas).abs() <= trace_lc.abs() * 5e-11,
                "sum of modal eigenvalues differs from physical trace(LC): order={order}"
            );
        }
    }

    #[test]
    fn physically_singular_material_uses_nonmodal_diffusive_transient_path() {
        for (l, c) in [(0., 2e-12), (5e-9, 0.)] {
            let line = CoupledLinePlan::from_packed(
                1,
                1.,
                &[1.],
                &[l],
                &[1e-3],
                &[c],
                CoupledLineLimits::default(),
            )
            .unwrap();
            assert!(line.exact_lc_modes().is_err());
        }
    }

    #[test]
    fn large_eigenvalues_do_not_overflow_before_reciprocal_c_scaling() {
        let line = CoupledLinePlan::from_packed(
            2,
            1.,
            &[0., 0., 0.],
            &[1e308, 0., 1e308],
            &[0., 0., 0.],
            &[1e-308, 0.75e-308, 1e-308],
            CoupledLineLimits::default(),
        )
        .unwrap();
        let modes = line.exact_lc_modes().unwrap();
        close(modes.total_delays[0], 0.5, 4e-14);
        close(modes.total_delays[1], 1.75_f64.sqrt(), 4e-14);
    }

    #[test]
    fn extreme_reciprocal_lc_scales_keep_finite_modal_basis_and_delay() {
        for (l, c) in [(1e200, 1e-200), (1e-200, 1e200)] {
            let line = CoupledLinePlan::from_packed(
                1,
                1.,
                &[0.],
                &[l],
                &[0.],
                &[c],
                CoupledLineLimits::default(),
            )
            .unwrap();
            let modes = line.exact_lc_modes().unwrap();
            close(modes.total_delays[0], 1., 4e-15);
            close(modes.voltage_basis[0] * modes.current_basis[0], 1., 4e-15);
        }
    }
}
