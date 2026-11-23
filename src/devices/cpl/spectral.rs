//! Real-positive-Laplace samples of the full lossy coupled telegrapher system.
//!
//! This kernel supplies physically exact frequency-axis samples of C's
//! *effective* lossy R/L/G/C matrices for subsequent rational fitting,
//! with all mutual Maxwell terms retained. cplsetup.c clamps every R entry
//! (including off-diagonal mutual resistance) to at least 1e-4 before
//! fitting. The immutable source `CplPlan` still preserves original values.
//! It is not an AC/PZ device registration or a time-domain approximation.
//!
//! In the positive real Laplace half-axis at s>0 the line's per-unit matrices
//! are Z(s)=R+sL and Y(s)=G+sC. Factor the better-scaled pair
//! Z(s)/s=L+R/s and Y(s)/s=C+G/s with the same normalized Cholesky/Jacobi
//! transformation used by the lossless high-frequency wavefront.
//!
//! Each resulting mode has a characteristic admittance, propagation
//! exponent, and current basis. The exact symmetric two-port consists of
//! equal self-admittance blocks and negative cross-admittance blocks,
//! obtained from a full modal congruence of the hyperbolic kernels.
//! Hyperbolic kernels are evaluated through exp(-u) and expm1(-2u), retaining
//! accuracy for small u and avoiding overflow for large u.

use super::modal::{CplModes, matched_admittance};
use super::plan::{CplMatrixKind, CplPlan};
use crate::maths::Matrix;
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Real Laplace-axis two-port sample, including every coupled conductor.
#[derive(Debug, Clone, PartialEq)]
pub struct CplSpectralSample {
    rate: Real,
    modes: CplModes,
    attenuation: Vec<Real>,
    self_admittance: Matrix,
    cross_admittance: Matrix,
}

impl CplSpectralSample {
    /// Construct a positive-real sample of the entire Maxwell cable.
    ///
    /// # Errors
    /// Invalid sampling rate, nonfinite effective matrices, singular
    /// transmission modes or unrepresentable two-port admittances.
    pub fn prepare(plan: &CplPlan, rate: Real) -> SpiceResult<Self> {
        if !rate.is_finite() || rate <= 0. {
            return Err(SpiceError::circuit(
                "CPL Laplace sample rate must be positive and finite",
            ));
        }
        let mut series_impedance = Matrix::zeros(plan.conductors(), plan.conductors())?;
        let mut shunt_admittance = Matrix::zeros(plan.conductors(), plan.conductors())?;
        for row in 0..plan.conductors() {
            for col in 0..plan.conductors() {
                let inductance = plan
                    .matrix(CplMatrixKind::Inductance)
                    .get(row, col)
                    .unwrap_or(0.);
                let resistance = plan
                    .matrix(CplMatrixKind::Resistance)
                    .get(row, col)
                    .unwrap_or(0.);
                let capacitance = plan
                    .matrix(CplMatrixKind::Capacitance)
                    .get(row, col)
                    .unwrap_or(0.);
                let conductance = plan
                    .matrix(CplMatrixKind::Conductance)
                    .get(row, col)
                    .unwrap_or(0.);
                // cplsetup.c:475 clamps every packed R entry before
                // building the transient polynomials, not only Rii.
                let reference_resistance = resistance.max(1e-4);
                let z_value = reference_resistance.mul_add(1. / rate, inductance);
                let y_value = conductance.mul_add(1. / rate, capacitance);
                if !z_value.is_finite() || !y_value.is_finite() {
                    return Err(SpiceError::circuit(
                        "CPL lossy Laplace coefficients overflow",
                    ));
                }
                series_impedance.set(row, col, z_value)?;
                shunt_admittance.set(row, col, y_value)?;
            }
        }

        let modes = CplModes::prepare(plan.length(), &series_impedance, &shunt_admittance)?;
        let mut attenuation = Vec::with_capacity(plan.conductors());
        let mut near_coefficients = Vec::with_capacity(plan.conductors());
        let mut across_coefficients = Vec::with_capacity(plan.conductors());
        for (&delay, &admittance) in modes.delays().iter().zip(modes.modal_admittances()) {
            // If rate*delay overflows, the exact passive limit is a
            // fully attenuated wave with coth(infinity)=1, csch(infinity)=0.
            let exponent = rate * delay;
            let decay = (-exponent).exp();
            let denom = -(-2. * exponent).exp_m1();
            if denom <= 0. || !denom.is_finite() {
                return Err(SpiceError::circuit(
                    "CPL Laplace propagation exponent is unresolvable",
                ));
            }
            let self_coefficient = admittance * decay.mul_add(decay, 1.) / denom;
            let across_coefficient = admittance * (2. * decay) / denom;
            if !self_coefficient.is_finite() || !across_coefficient.is_finite() {
                return Err(SpiceError::circuit(
                    "CPL Laplace two-port admittance overflows",
                ));
            }
            attenuation.push(decay);
            near_coefficients.push(self_coefficient);
            across_coefficients.push(across_coefficient);
        }
        let self_admittance = matched_admittance(modes.current_basis(), &near_coefficients)?;
        let cross_admittance = matched_admittance(modes.current_basis(), &across_coefficients)?;
        Ok(Self {
            rate,
            modes,
            attenuation,
            self_admittance,
            cross_admittance,
        })
    }

    /// Sampling point (Laplace inverse seconds), strictly positive.
    #[must_use]
    pub const fn rate(&self) -> Real {
        self.rate
    }

    /// Complete frequency-dependent coupled modal structure.
    #[must_use]
    pub const fn modes(&self) -> &CplModes {
        &self.modes
    }

    /// Real Laplace-axis modal propagation factors, between zero and one.
    #[must_use]
    pub fn attenuation(&self) -> &[Real] {
        &self.attenuation
    }

    /// Physical near-near and far-far small-signal port admittance block.
    #[must_use]
    pub const fn self_admittance(&self) -> &Matrix {
        &self.self_admittance
    }

    /// Magnitude of the near-far / far-near admittance block.
    /// Offdiagonal two-port blocks have the *negative* of this matrix.
    #[must_use]
    pub const fn cross_admittance(&self) -> &Matrix {
        &self.cross_admittance
    }
}
