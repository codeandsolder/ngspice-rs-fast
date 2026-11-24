//! Native modal-CPL to physical passive exact-delay stage-network bridge.
//!
//! The upstream coupled-line modal diagonalizer provides reciprocal
//! voltage/current bases and per-mode physical characteristic admittance.
//! Its modal voltage/current coordinates are *not* normalized to unit
//! power. Scattering requires the reciprocal power-normalized bases:
//!
//! Scale each native modal voltage basis column by the inverse square
//! root of its characteristic modal admittance, and the matching
//! current basis column by the positive square root.
//! The transformed bases remain power-reciprocal, so inverse transforms
//! are transposes of the opposite basis (no matrix inverse).
//!
//! Use a symmetric coupled passive R/G pi stage per cable segment,
//! exact LC modal propagation and prepared physical terminal MNA
//! equations. This module works directly with the modal values already
//! prepared by the existing native modal diagonalizer, without
//! repeating its LC eigensolver.

use super::{
    PhysicalPortEquations, PreparedWaveStep, WaveBatch, WaveMnaStamp, WaveNetwork,
    coupled_static_rg_pi,
};

/// The complete already-diagonalized LC eigenbasis and physical
/// reciprocal series/shunt material matrices of a native P model.
#[derive(Debug, Clone, Copy)]
pub struct NativeModalCoefficients<'a> {
    pub voltage_basis: &'a [f64],
    pub current_basis: &'a [f64],
    pub characteristic_admittances: &'a [f64],
    pub total_delays: &'a [f64],
    pub resistance: &'a [f64],
    pub conductance: &'a [f64],
    pub length: f64,
}

/// Concrete spatial discretization and earliest exterior source
/// arrival bounds for a physically coupled line transient.
#[derive(Debug)]
pub struct WaveDiscretization {
    pub sections: usize,
    pub near_front: Vec<f64>,
    pub far_front: Vec<f64>,
    pub prune_histories: bool,
}

/// Reciprocal normalization for native modal voltage/current columns.
/// The physical scaling is derived from each mode's admittance,
/// and preserves the full noncommuting conductor basis.
///
/// # Errors
/// Invalid dimensions or nonfinite normalized basis.
fn power_normalize_modal_bases(
    voltage: &[f64],
    current: &[f64],
    admittances: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
    let modes = admittances.len();
    let size = modes
        .checked_mul(modes)
        .ok_or("modal square size overflow")?;
    if voltage.len() != size || current.len() != size {
        return Err("invalid reciprocal modal basis dimensions");
    }
    let mut pv = vec![0.; size];
    let mut pi = vec![0.; size];
    for row in 0..modes {
        for (mode, &y) in admittances.iter().enumerate() {
            if !y.is_finite() || y <= 0. {
                return Err("invalid modal characteristic admittance");
            }
            let index = row * modes + mode;
            let root = y.sqrt();
            pv[index] = voltage[index] / root;
            pi[index] = current[index] * root;
            if !pv[index].is_finite() || !pi[index].is_finite() {
                return Err("nonfinite power-normalized modal basis");
            }
        }
    }
    Ok((pv, pi))
}

#[derive(Debug)]
pub struct PhysicalModalLine {
    network: WaveNetwork,
    modes: usize,
    modal_voltage_to_physical: Vec<f64>,
    modal_current_to_physical: Vec<f64>,
    physical_voltage_to_modal: Vec<f64>,
    physical_current_to_modal: Vec<f64>,
}

#[derive(Debug)]
pub struct PreparedPhysicalModalLine {
    prepared: PreparedWaveStep,
    equations: PhysicalPortEquations,
}

impl PhysicalModalLine {
    /// Construct from a native LC modal decomposition and full physical
    /// per-unit-length R and G matrices, for a lossless-delay plus
    /// reciprocal passive resistive pi-stage Strang realization.
    ///
    /// Raw native basis columns are row-major conductor x modal mode;
    /// the modal characteristic admittances and total propagation
    /// delays are positive physical values, not dimensionless proxies.
    ///
    /// # Errors
    /// Nonphysical modal data, nonreciprocal bases, negative/passivity-
    /// violating static R/G, malformed dimensions or resource exhaustion.
    pub fn from_native_modes(
        coefficients: NativeModalCoefficients<'_>,
        setup: WaveDiscretization,
    ) -> Result<Self, &'static str> {
        let NativeModalCoefficients {
            voltage_basis: native_voltage,
            current_basis: native_current,
            characteristic_admittances: native_modal_admittance,
            total_delays: physical_total_delay,
            resistance: series_r,
            conductance: shunt_g,
            length,
        } = coefficients;
        let WaveDiscretization {
            sections,
            near_front,
            far_front,
            prune_histories,
        } = setup;
        let modes = native_modal_admittance.len();
        let n2 = modes
            .checked_mul(modes)
            .ok_or("CPL model mode count overflow")?;
        if modes == 0
            || native_voltage.len() != n2
            || native_current.len() != n2
            || series_r.len() != n2
            || shunt_g.len() != n2
            || physical_total_delay.len() != modes
            || !length.is_finite()
            || length <= 0.
            || sections == 0
        {
            return Err("CPL modal model shape/geometry mismatch");
        }
        if native_voltage
            .iter()
            .chain(native_current)
            .chain(series_r)
            .chain(shunt_g)
            .any(|v| !v.is_finite())
            || native_modal_admittance
                .iter()
                .chain(physical_total_delay)
                .any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err("nonfinite/nonphysical coupled-line modal values");
        }
        let (pv, pi) =
            power_normalize_modal_bases(native_voltage, native_current, native_modal_admittance)?;
        let segment = length
            / f64::from(
                u32::try_from(sections).map_err(|_| "CPL section count exceeds supported size")?,
            );
        if !segment.is_finite() || segment <= 0. {
            return Err("CPL section length is not representable");
        }
        let half_denominator = 2.
            * f64::from(
                u32::try_from(sections).map_err(|_| "CPL section count exceeds supported size")?,
            );
        let mut half = Vec::with_capacity(modes);
        for &total in physical_total_delay {
            let value = total / half_denominator;
            if !value.is_finite() || value <= 0. {
                return Err("CPL exact modal half-delay cannot be represented");
            }
            half.push(value);
        }
        let static_s = coupled_static_rg_pi(series_r, shunt_g, &pv, &pi, modes, segment)?;
        // Power-reciprocal modal bases give inverse transforms exactly
        // by transpose. This includes all Maxwell cross-conductor terms.
        let mut voltage_to_modal = vec![0.; n2];
        let mut current_to_modal = vec![0.; n2];
        for conductor in 0..modes {
            for mode in 0..modes {
                voltage_to_modal[mode * modes + conductor] = pi[conductor * modes + mode];
                current_to_modal[mode * modes + conductor] = pv[conductor * modes + mode];
            }
        }
        let network = WaveNetwork::new(
            half,
            sections,
            static_s,
            near_front,
            far_front,
            prune_histories,
        )?;
        Ok(Self {
            network,
            modes,
            modal_voltage_to_physical: pv,
            modal_current_to_physical: pi,
            physical_voltage_to_modal: voltage_to_modal,
            physical_current_to_modal: current_to_modal,
        })
    }

    /// Prepare the exact-delay line for one global simulation timestep,
    /// and construct the full physical branch-current equations once.
    ///
    /// # Errors
    /// Unrepresentable step or invalid physical modal transform.
    pub fn prepare(&self, time: f64) -> Result<PreparedPhysicalModalLine, &'static str> {
        let prepared = self.network.prepare_step(time)?;
        let equations = prepared.modal_terminal_equations()?.project_physical(
            &self.physical_voltage_to_modal,
            &self.physical_current_to_modal,
        )?;
        Ok(PreparedPhysicalModalLine {
            prepared,
            equations,
        })
    }

    /// Accept one fully resolved, native-physical terminal trial as a
    /// single atomic physical-state and modal-wave history transaction.
    ///
    /// # Errors
    /// Stale/foreign/malformed accepted step or resource exhaustion.
    pub fn accept(&mut self, trial: WaveBatch) -> Result<(), &'static str> {
        self.network.accept_prepared(trial)
    }

    #[must_use]
    pub const fn modes(&self) -> usize {
        self.modes
    }

    /// Physical conductor voltage as a linear map of power-normalized
    /// modal voltage, provided for diagnostic and basis cross-checks.
    #[must_use]
    pub fn modal_voltage_basis(&self) -> &[f64] {
        &self.modal_voltage_to_physical
    }

    /// Physical conductor current as a linear map of power-normalized
    /// modal current, provided for diagnostic and basis cross-checks.
    #[must_use]
    pub fn modal_current_basis(&self) -> &[f64] {
        &self.modal_current_to_physical
    }

    #[must_use]
    pub fn histories(&self) -> usize {
        self.network.retained_samples()
    }
}

impl PreparedPhysicalModalLine {
    #[must_use]
    pub const fn internal_steps(&self) -> usize {
        self.prepared.internal_steps()
    }

    #[must_use]
    pub const fn equations(&self) -> &PhysicalPortEquations {
        &self.equations
    }

    /// Native MNA-ready physical KCL/constraint stamp (signal and
    /// return nodes and unique branch-current equations).
    ///
    /// # Errors
    /// Mis-sized or conflicting physical sparse matrix indices.
    pub fn stamp(
        &self,
        signal: &[Option<usize>],
        returns: &[Option<usize>],
        branch: &[usize],
    ) -> Result<WaveMnaStamp, &'static str> {
        self.equations.mna_stamp(signal, returns, branch)
    }

    /// Construct the complete accepted-history candidate from converged
    /// physical differential signal-return voltages and inward line
    /// branch currents. Does not mutate accepted line state.
    ///
    /// # Errors
    /// Invalid terminal wave or nonfinite candidate.
    pub fn trial_physical(
        &self,
        voltage: &[f64],
        current: &[f64],
    ) -> Result<WaveBatch, &'static str> {
        let incident = self.equations.incident_from_physical(voltage, current)?;
        let n = self.equations.modes();
        self.prepared.evaluate(&incident[..n], &incident[n..])
    }
}

#[cfg(test)]
mod tests;
