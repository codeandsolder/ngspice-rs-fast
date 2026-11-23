//! High-frequency lossless modal structure of the *full coupled* LC system.
//!
//! This is the causal wavefront and characteristic admittance needed before
//! the lossy CPL frequency-fitting and convolution solver can be assembled.
//! No approximation discards off-diagonal Maxwell coupling, and L,C need not
//! commute. Mode voltage/current bases are reciprocal transposes, preserving
//! instantaneous port power and the sign of bidirectional launched waves.

use crate::maths::Matrix;
use crate::primitives::{Real, SpiceError, SpiceResult};

/// The reciprocal voltage and current transformations of the modal line.
/// The columns are sorted by ascending physical propagation delay.
#[derive(Debug, Clone, PartialEq)]
pub struct CplModes {
    delays: Vec<Real>,
    admittances: Vec<Real>,
    voltage_basis: Matrix,
    current_basis: Matrix,
    characteristic_admittance: Matrix,
}

impl CplModes {
    /// Factor the symmetric LC similarity once and retain the modal projection
    /// required by the coupled transient engine.
    ///
    /// # Errors
    /// Ill-conditioned LC, nonfinite admittance, or invalid propagation modes.
    pub fn prepare(length: Real, inductance: &Matrix, capacitance: &Matrix) -> SpiceResult<Self> {
        propagation_modes(length, inductance, capacitance)
    }

    /// One lossless high-frequency propagation time per mode, shortest first.
    #[must_use]
    pub fn delays(&self) -> &[Real] {
        &self.delays
    }

    /// One modal current/voltage ratio in reciprocal physical impedance.
    #[must_use]
    pub fn modal_admittances(&self) -> &[Real] {
        &self.admittances
    }

    /// Columns mapping modal voltages to physical conductor voltages.
    #[must_use]
    pub const fn voltage_basis(&self) -> &Matrix {
        &self.voltage_basis
    }

    /// Columns mapping modal currents to physical conductor currents.
    #[must_use]
    pub const fn current_basis(&self) -> &Matrix {
        &self.current_basis
    }

    /// Physical matched load conductance: I = Y0 V. This matrix includes
    /// off-diagonal cross-conductor conductances even for lossless lines.
    #[must_use]
    pub const fn characteristic_admittance(&self) -> &Matrix {
        &self.characteristic_admittance
    }
}

/// The spectrum of the coupled LC product equals the symmetric similarity
/// transformed by the Cholesky factor of C.
/// Positive definiteness ensures real propagation delays without assuming
/// that L and C commute or that their eigenvectors are already aligned.
fn propagation_modes(
    length: Real,
    inductance: &Matrix,
    capacitance: &Matrix,
) -> SpiceResult<CplModes> {
    let order = inductance.rows();
    let scale = |matrix: &Matrix| -> SpiceResult<Real> {
        let maximum = (0..order)
            .filter_map(|row| matrix.get(row, row))
            .fold(0.0_f64, Real::max);
        if !maximum.is_finite() || maximum <= 0. {
            return Err(SpiceError::circuit(
                "CPL propagation needs positive L and C scales",
            ));
        }
        Ok(maximum)
    };
    let inductance_scale = scale(inductance)?;
    let capacitance_scale = scale(capacitance)?;

    // Cholesky of C/c_scale: lower-triangular S.
    let mut lower = Matrix::zeros(order, order)?;
    for row in 0..order {
        for col in 0..=row {
            let mut entry = capacitance
                .get(row, col)
                .ok_or_else(|| SpiceError::circuit("CPL capacitance index"))?
                / capacitance_scale;
            for index in 0..col {
                let left = lower.get(row, index).unwrap_or(0.);
                let right = lower.get(col, index).unwrap_or(0.);
                entry = (-left).mul_add(right, entry);
            }
            if row == col {
                if entry <= 0. || !entry.is_finite() {
                    return Err(SpiceError::circuit(
                        "CPL capacitance Cholesky is singular or indefinite",
                    ));
                }
                lower.set(row, col, entry.sqrt())?;
            } else {
                let pivot = lower.get(col, col).unwrap_or(0.);
                let value = entry / pivot;
                if !value.is_finite() {
                    return Err(SpiceError::circuit("CPL modal basis is nonfinite"));
                }
                lower.set(row, col, value)?;
            }
        }
    }

    let mut similarity = Matrix::zeros(order, order)?;
    for row in 0..order {
        for col in row..order {
            let mut value = 0.;
            for inner in 0..order {
                let s_left = lower.get(inner, row).unwrap_or(0.);
                for other in 0..order {
                    let l = inductance.get(inner, other).unwrap_or(0.) / inductance_scale;
                    let s_right = lower.get(other, col).unwrap_or(0.);
                    value = (s_left * l).mul_add(s_right, value);
                }
            }
            if !value.is_finite() {
                return Err(SpiceError::circuit("CPL similarity transform overflowed"));
            }
            similarity.set(row, col, value)?;
            if row != col {
                similarity.set(col, row, value)?;
            }
        }
    }
    let (eigenvalues, eigenvectors) = similarity.symmetric_eigendecomposition()?;
    let magnitude = length * inductance_scale.sqrt() * capacitance_scale.sqrt();
    if !magnitude.is_finite() || magnitude <= 0. {
        return Err(SpiceError::circuit("CPL propagation time scale overflows"));
    }

    let mut delays = Vec::with_capacity(order);
    let mut admittances = Vec::with_capacity(order);
    for value in eigenvalues {
        if !value.is_finite() || value <= 0. {
            return Err(SpiceError::circuit("CPL propagation mode is not positive"));
        }
        let root = value.sqrt();
        let delay = magnitude * root;
        let admittance = capacitance_scale.sqrt() / (inductance_scale.sqrt() * root);
        if !delay.is_finite() || delay <= 0. || !admittance.is_finite() || admittance <= 0. {
            return Err(SpiceError::circuit(
                "CPL propagation delay or modal admittance is not finite",
            ));
        }
        delays.push(delay);
        admittances.push(admittance);
    }
    let (current_basis, voltage_basis) = modal_bases(&lower, &eigenvectors)?;
    let characteristic_admittance = matched_admittance(&current_basis, &admittances)?;
    Ok(CplModes {
        delays,
        admittances,
        voltage_basis,
        current_basis,
        characteristic_admittance,
    })
}
/// Build the power-reciprocal modal bases without explicitly inverting S.
fn modal_bases(lower: &Matrix, eigenvectors: &Matrix) -> SpiceResult<(Matrix, Matrix)> {
    let order = lower.rows();
    let mut current_basis = Matrix::zeros(order, order)?;
    let mut voltage_basis = Matrix::zeros(order, order)?;
    // I = S Q i_modal, V = S^-T Q v_modal. Their bases are reciprocal
    // transposes: V^T I = v_modal^T i_modal (power normalization).
    for mode in 0..order {
        for row in 0..order {
            let mut value = 0.;
            for col in 0..=row {
                value = lower
                    .get(row, col)
                    .unwrap_or(0.)
                    .mul_add(eigenvectors.get(col, mode).unwrap_or(0.), value);
            }
            current_basis.set(row, mode, value)?;
        }
        for row in (0..order).rev() {
            let mut value = eigenvectors.get(row, mode).unwrap_or(0.);
            for col in row + 1..order {
                value = (-lower.get(col, row).unwrap_or(0.))
                    .mul_add(voltage_basis.get(col, mode).unwrap_or(0.), value);
            }
            value /= lower.get(row, row).unwrap_or(0.);
            if !value.is_finite() {
                return Err(SpiceError::circuit("CPL modal voltage basis is nonfinite"));
            }
            voltage_basis.set(row, mode, value)?;
        }
    }
    Ok((current_basis, voltage_basis))
}

/// The characteristic matrix is the modal current-basis congruence of\n/// the diagonal modal admittances; no coupling terms are omitted.
fn matched_admittance(current_basis: &Matrix, admittances: &[Real]) -> SpiceResult<Matrix> {
    let order = current_basis.rows();
    let mut characteristic_admittance = Matrix::zeros(order, order)?;
    for row in 0..order {
        for col in row..order {
            let mut value = 0.;
            for (mode, &admittance) in admittances.iter().enumerate() {
                let left = current_basis.get(row, mode).unwrap_or(0.);
                let right = current_basis.get(col, mode).unwrap_or(0.);
                value = (left * admittance).mul_add(right, value);
            }
            if !value.is_finite() {
                return Err(SpiceError::circuit(
                    "CPL characteristic admittance overflows",
                ));
            }
            characteristic_admittance.set(row, col, value)?;
            if row != col {
                characteristic_admittance.set(col, row, value)?;
            }
        }
    }
    Ok(characteristic_admittance)
}
