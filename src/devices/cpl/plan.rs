//! Bounded Maxwell R/L/G/C preparation for KSPICE coupled lines.
//!
//! Packed triangular storage follows C: (00,01,..,0n,11,..,nn).
//! R/G may also specify only n diagonal entries or default to zero.
//! Original physical values are retained without dropping mutual coupling.

use crate::maths::Matrix;
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Maximum number of conductors supported by upstream CPL.
pub const MAX_CONDUCTORS: usize = 8;

/// A per-unit-length Maxwell parameter matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CplMatrixKind {
    /// Series resistance (ohm/unit).
    Resistance,
    /// Series inductance (henry/unit).
    Inductance,
    /// Shunt conductance (siemens/unit).
    Conductance,
    /// Shunt capacitance (farad/unit).
    Capacitance,
}

impl CplMatrixKind {
    const fn index(self) -> usize {
        match self {
            Self::Resistance => 0,
            Self::Inductance => 1,
            Self::Conductance => 2,
            Self::Capacitance => 3,
        }
    }
    const fn name(self) -> &'static str {
        match self {
            Self::Resistance => "R",
            Self::Inductance => "L",
            Self::Conductance => "G",
            Self::Capacitance => "C",
        }
    }
    const fn requires_modes(self) -> bool {
        matches!(self, Self::Inductance | Self::Capacitance)
    }
}

/// Immutable validated R/L/G/C matrices for one line instance.
#[derive(Debug, Clone, PartialEq)]
pub struct CplPlan {
    conductors: usize,
    length: Real,
    matrices: [Matrix; 4],
    modal_delays: Vec<Real>,
}

impl CplPlan {
    /// Expand C triangular vectors to symmetric row-major matrices.
    ///
    /// # Errors
    /// Invalid conductor count, length, packed dimensions, nonfinite values,
    /// nonpassive R/G, or missing/ill-conditioned propagation modes in L/C.
    pub fn prepare(
        conductors: usize,
        length: Real,
        r: &[Real],
        l: &[Real],
        g: &[Real],
        c: &[Real],
    ) -> SpiceResult<Self> {
        if !(1..=MAX_CONDUCTORS).contains(&conductors) {
            return Err(SpiceError::circuit(format!(
                "CPL needs 1..={MAX_CONDUCTORS} conductors; got {conductors}"
            )));
        }
        if !length.is_finite() || length <= 0. {
            return Err(SpiceError::circuit(
                "CPL length must be positive and finite",
            ));
        }
        let kinds = [
            CplMatrixKind::Resistance,
            CplMatrixKind::Inductance,
            CplMatrixKind::Conductance,
            CplMatrixKind::Capacitance,
        ];
        let packed = [r, l, g, c];
        let mut matrices = Vec::with_capacity(4);
        for (kind, values) in kinds.into_iter().zip(packed) {
            let matrix = unpack(conductors, kind, values)?;
            validate_spectrum(conductors, kind, &matrix)?;
            matrices.push(matrix);
        }
        let matrices: [Matrix; 4] = matrices
            .try_into()
            .map_err(|_| SpiceError::circuit("CPL internal matrix count"))?;
        let modal_delays = propagation_delays(
            length,
            &matrices[CplMatrixKind::Inductance.index()],
            &matrices[CplMatrixKind::Capacitance.index()],
        )?;
        Ok(Self {
            conductors,
            length,
            matrices,
            modal_delays,
        })
    }

    /// Number of coupled conductors (excluding the return references).
    #[must_use]
    pub const fn conductors(&self) -> usize {
        self.conductors
    }

    /// Line length in the per-unit matrices' common distance unit.
    #[must_use]
    pub const fn length(&self) -> Real {
        self.length
    }

    /// Original full symmetric row-major physical coefficients.
    #[must_use]
    pub const fn matrix(&self, kind: CplMatrixKind) -> &Matrix {
        &self.matrices[kind.index()]
    }

    /// Propagation delays of all coupled lossless modes, smallest first.
    ///
    /// These are the physical roots of the full matrix LC, not independent
    /// conductor delays. Resistive and dielectric losses affect attenuation
    /// and convolution; they do not change these high-frequency wavefronts.
    #[must_use]
    pub fn modal_delays(&self) -> &[Real] {
        &self.modal_delays
    }
}

fn unpack(order: usize, kind: CplMatrixKind, values: &[Real]) -> SpiceResult<Matrix> {
    let triangle = order * (order + 1) / 2;
    let diagonals_allowed = !kind.requires_modes();
    let diagonal = diagonals_allowed && values.len() == order;
    let zero = diagonals_allowed && values.is_empty();
    if values.len() != triangle && !diagonal && !zero {
        return Err(SpiceError::circuit(format!(
            "CPL {} expects {triangle} upper-triangular entries or a valid R/G diagonal/default; got {}",
            kind.name(),
            values.len()
        )));
    }
    if values.iter().any(|entry| !entry.is_finite()) {
        return Err(SpiceError::circuit(format!(
            "CPL {} has a nonfinite coefficient",
            kind.name()
        )));
    }
    let mut matrix = Matrix::zeros(order, order)?;
    if diagonal && (values.len() != triangle) {
        for (index, value) in values.iter().copied().enumerate() {
            matrix.set(index, index, value)?;
        }
    } else if !zero {
        let mut source = values.iter().copied();
        for row in 0..order {
            for col in row..order {
                let value = source
                    .next()
                    .ok_or_else(|| SpiceError::circuit("CPL matrix length mismatch"))?;
                matrix.set(row, col, value)?;
                if row != col {
                    matrix.set(col, row, value)?;
                }
            }
        }
    }
    Ok(matrix)
}

fn validate_spectrum(order: usize, kind: CplMatrixKind, matrix: &Matrix) -> SpiceResult<()> {
    let diagonals = (0..order)
        .map(|index| matrix.get(index, index).unwrap_or(f64::NAN))
        .collect::<Vec<_>>();
    if diagonals.iter().any(|v| !v.is_finite() || *v < 0.)
        || (kind.requires_modes() && diagonals.contains(&0.))
    {
        return Err(SpiceError::circuit(format!(
            "CPL {} has an invalid diagonal",
            kind.name()
        )));
    }

    // Diagonal congruence preserves definiteness while normalizing physical
    // magnitudes; it works across highly different capacitance/inductance units.
    let mut normalized = Matrix::zeros(order, order)?;
    for row in 0..order {
        let row_root = diagonals[row].sqrt();
        for (col, diagonal) in diagonals.iter().enumerate().skip(row) {
            let col_root = diagonal.sqrt();
            let physical = matrix.get(row, col).unwrap_or(f64::NAN);
            if row_root == 0. || col_root == 0. {
                if physical != 0. {
                    return Err(SpiceError::circuit(format!(
                        "CPL {} couples a zero diagonal to a live conductor",
                        kind.name()
                    )));
                }
                continue;
            }
            let relative = physical / row_root / col_root;
            if !relative.is_finite() {
                return Err(SpiceError::circuit(format!(
                    "CPL {} has an unrepresentable normalized coupling",
                    kind.name()
                )));
            }
            normalized.set(row, col, relative)?;
            if row != col {
                normalized.set(col, row, relative)?;
            }
        }
    }
    let eigenvalues = normalized.symmetric_eigenvalues()?;
    let minimum = eigenvalues.first().copied().unwrap_or(-1.);
    let uncertainty = 64.
        * Real::EPSILON
        * Real::from(u32::try_from(order).map_err(|_| SpiceError::circuit("CPL order overflow"))?);
    if minimum < -uncertainty || (kind.requires_modes() && minimum <= uncertainty) {
        return Err(SpiceError::circuit(format!(
            "CPL {} {} (smallest scaled eigenvalue {minimum:.8e})",
            kind.name(),
            if kind.requires_modes() {
                "has missing or ill-conditioned propagation modes"
            } else {
                "is not passive positive-semidefinite"
            }
        )));
    }
    Ok(())
}

/// The spectrum of the coupled LC product equals the symmetric similarity
/// transformed by the Cholesky factor of C.
/// Positive definiteness ensures real propagation delays without assuming
/// that L and C commute or that their eigenvectors are already aligned.
fn propagation_delays(
    length: Real,
    inductance: &Matrix,
    capacitance: &Matrix,
) -> SpiceResult<Vec<Real>> {
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
    let eigenvalues = similarity.symmetric_eigenvalues()?;
    let magnitude = length * inductance_scale.sqrt() * capacitance_scale.sqrt();
    if !magnitude.is_finite() || magnitude <= 0. {
        return Err(SpiceError::circuit("CPL propagation time scale overflows"));
    }
    eigenvalues
        .into_iter()
        .map(|value| {
            if !value.is_finite() || value <= 0. {
                return Err(SpiceError::circuit("CPL propagation mode is not positive"));
            }
            let delay = magnitude * value.sqrt();
            if !delay.is_finite() || delay <= 0. {
                return Err(SpiceError::circuit("CPL propagation delay is not finite"));
            }
            Ok(delay)
        })
        .collect()
}
