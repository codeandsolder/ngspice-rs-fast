//! A physically passive, fully coupled static R/G scattering section.
//!
//! Strang splitting of lossless LC propagation and lossy R/G creates
//! exact modal delays and passive static stages. Within the static
//! stage, use a symmetric conductive pi network: two shunts G*dx/2
//! and a series multi-conductor resistor matrix R*dx.
//!
//! A single scaled partial-pivoting dense solve converts this network
//! into power-normalized 2N by 2N scattering without ever inverting
//! R or G. This remains valid for PSD singular R, G, zero series
//! loss, zero shunt loss and independent or mutual coupling.
//! The static pi approximation is second order in dx, preserving the
//! order of the overall symmetric LC/RG split.

use super::matrix_is_passive;

/// Solve a dense physical multi-right-hand-side system in place.
/// Rows are scaled by their largest coefficient to control units in
/// reciprocal modal bases before partial pivoting.
fn solve_many(
    mut a: Vec<f64>,
    mut rhs: Vec<f64>,
    order: usize,
    columns: usize,
) -> Result<Vec<f64>, &'static str> {
    for row in 0..order {
        let scale = (0..order)
            .map(|col| a[row * order + col].abs())
            .fold(0.0_f64, f64::max);
        if scale == 0. || !scale.is_finite() {
            return Err("singular modal static pi coefficient row");
        }
        for col in 0..order {
            a[row * order + col] /= scale;
        }
        for col in 0..columns {
            rhs[row * columns + col] /= scale;
        }
    }
    for pivot_col in 0..order {
        let pivot_row = (pivot_col..order)
            .max_by(|&a_row, &b_row| {
                a[a_row * order + pivot_col]
                    .abs()
                    .total_cmp(&a[b_row * order + pivot_col].abs())
            })
            .ok_or("empty static pi pivot candidates")?;
        let pivot = a[pivot_row * order + pivot_col];
        if !pivot.is_finite() || pivot.abs() <= 1e-14 {
            return Err("static pi modal coefficient matrix is singular or ill-conditioned");
        }
        if pivot_row != pivot_col {
            for col in 0..order {
                a.swap(pivot_row * order + col, pivot_col * order + col);
            }
            for col in 0..columns {
                rhs.swap(pivot_row * columns + col, pivot_col * columns + col);
            }
        }
        let diagonal = a[pivot_col * order + pivot_col];
        for row in pivot_col + 1..order {
            let factor = a[row * order + pivot_col] / diagonal;
            a[row * order + pivot_col] = 0.;
            for col in pivot_col + 1..order {
                a[row * order + col] =
                    (-factor).mul_add(a[pivot_col * order + col], a[row * order + col]);
            }
            for col in 0..columns {
                rhs[row * columns + col] =
                    (-factor).mul_add(rhs[pivot_col * columns + col], rhs[row * columns + col]);
            }
        }
    }
    for row in (0..order).rev() {
        for column in 0..columns {
            let mut result = rhs[row * columns + column];
            for next in row + 1..order {
                result = (-a[row * order + next]).mul_add(rhs[next * columns + column], result);
            }
            result /= a[row * order + row];
            if !result.is_finite() {
                return Err("nonfinite static coupled scattering coefficient");
            }
            rhs[row * columns + column] = result;
        }
    }
    Ok(rhs)
}

/// Prepare one real symmetric passive static R/G pi stage in a
/// fully coupled power-normalized LC modal basis.
///
/// R and G are full physical conductor matrices per unit length,
/// and the segment length is a positive physical distance. The modal
/// voltage/current bases are full row-major matrices mapping
/// *power-normalized modal waves* to physical conductor voltage
/// and inward current. The two bases must be power-reciprocal.
///
/// Returns row-major two-port scattering with all near and far modes
/// retained and no resistance inversion or fake mutual losses.
///
/// # Errors
/// Wrong matrix shapes, nonphysical segment, non-reciprocal modal
/// basis, singular static solve, or loss of full-matrix passivity.
pub fn coupled_static_rg_pi(
    resistance: &[f64],
    conductance: &[f64],
    modal_voltage: &[f64],
    modal_current: &[f64],
    modes: usize,
    cell_length: f64,
) -> Result<Vec<f64>, &'static str> {
    if modes == 0 || modes > 32 || !cell_length.is_finite() || cell_length <= 0. {
        return Err("invalid physical static line section");
    }
    let size = modes
        .checked_mul(modes)
        .ok_or("static line matrix dimension overflow")?;
    if [resistance, conductance, modal_voltage, modal_current]
        .into_iter()
        .any(|matrix| matrix.len() != size || matrix.iter().any(|v| !v.is_finite()))
    {
        return Err("nonfinite or incomplete physical Maxwell/modal matrices");
    }
    for left in 0..modes {
        for right in 0..modes {
            let mut power = 0.;
            for conductor in 0..modes {
                power = modal_voltage[conductor * modes + left]
                    .mul_add(modal_current[conductor * modes + right], power);
            }
            let expected = if left == right { 1. } else { 0. };
            if (power - expected).abs() > 2e-9 {
                return Err("physical modal bases violate reciprocal power normalization");
            }
        }
    }

    let width = modes
        .checked_mul(2)
        .ok_or("line scattering width overflow")?;
    let order = modes
        .checked_mul(3)
        .ok_or("static pi branch dimension overflow")?;
    let mut matrix = vec![0.; order * order];
    let mut rhs = vec![0.; order * width];
    let mut shunt = vec![0.; size];
    for row in 0..modes {
        for col in 0..modes {
            let mut value = 0.;
            for conductor in 0..modes {
                value = conductance[row * modes + conductor]
                    .mul_add(modal_voltage[conductor * modes + col], value);
            }
            shunt[row * modes + col] = value * (cell_length * 0.5);
        }
    }
    for row in 0..modes {
        let near = row;
        let far = row + modes;
        let series = row + width;
        for col in 0..modes {
            let pvoltage = modal_voltage[row * modes + col];
            let pcurrent = modal_current[row * modes + col];
            let ghalf = shunt[row * modes + col];
            // Near/far port currents equal half shunt plus/minus
            // the same internal physical series branch current.
            matrix[near * order + col] = pcurrent + ghalf;
            matrix[far * order + modes + col] = pcurrent + ghalf;
            rhs[near * width + col] = pcurrent - ghalf;
            rhs[far * width + modes + col] = pcurrent - ghalf;
            // Series voltage constraint, valid even for R singular.
            matrix[series * order + col] = -pvoltage;
            matrix[series * order + modes + col] = pvoltage;
            matrix[series * order + width + col] = resistance[row * modes + col] * cell_length;
            rhs[series * width + col] = pvoltage;
            rhs[series * width + modes + col] = -pvoltage;
        }
        matrix[near * order + width + row] = 1.;
        matrix[far * order + width + row] = -1.;
    }
    let full = solve_many(matrix, rhs, order, width)?;
    let mut scattering = vec![0.; width * width];
    scattering.copy_from_slice(&full[..width * width]);
    matrix_is_passive(&scattering, width)?;
    Ok(scattering)
}

#[cfg(test)]
mod tests;
