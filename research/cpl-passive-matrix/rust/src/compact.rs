//! Coalesced matrix-pole realization of the same positive-real rational fit.
//!
//! NNLS may leave hundreds of active rank-one PSD atoms but only a handful
//! of distinct shared pole rates. Coalesce all atoms sharing a rate into
//! symmetric PSD A/B matrices; one n-vector filter per pole then replaces
//! one scalar history per active atom. The fit remains passive by construction.
//! This module intentionally implements only the LOCAL port h1 kernel.

use std::collections::BTreeMap;

use crate::{Pole, integration_weights, next_port_id};

#[derive(Debug, Clone)]
struct RateBlock {
    rate: f64,
    high: Vec<f64>,
    low: Vec<f64>,
    accepted_filtered: Vec<f64>,
}

/// General positive-real local matrix admittance with shared real poles.
#[derive(Debug)]
pub struct MatrixPassivePort {
    owner_id: u64,
    order: usize,
    blocks: Vec<RateBlock>,
    accepted_voltage: Vec<f64>,
    revision: u64,
}

/// A pure trial whose data can be committed only from its matching revision.
#[derive(Debug, Clone)]
pub struct MatrixTrial {
    pub current: Vec<f64>,
    pub jacobian: Vec<f64>,
    new_voltage: Vec<f64>,
    updated_filters: Vec<Vec<f64>>,
    revision: u64,
    owner_id: u64,
}

/// Step-constant positive-semidefinite Newton Jacobian and accepted-history
/// source; reuse one prepared step across every Newton candidate iteration.
#[derive(Debug, Clone)]
pub struct PreparedStep {
    order: usize,
    revision: u64,
    owner_id: u64,
    jacobian: Vec<f64>,
    history_rhs: Vec<f64>,
    filter_base: Vec<Vec<f64>>,
    alphas: Vec<f64>,
}

impl PreparedStep {
    /// Physical current Jacobian of this time step (row-major n by n).
    #[must_use]
    pub fn jacobian(&self) -> &[f64] {
        &self.jacobian
    }

    /// Contribution solely from the already accepted filter state.
    #[must_use]
    pub fn accepted_history_rhs(&self) -> &[f64] {
        &self.history_rhs
    }

    /// Evaluate a trial conductor voltage with a precomputed linear system,
    /// without redoing exponentials or matrix assembly per Newton iterate.
    ///
    /// # Errors
    /// Invalid voltage width or nonfinite current / filter update.
    pub fn evaluate(&self, voltage: &[f64]) -> Result<MatrixTrial, &'static str> {
        if voltage.len() != self.order || voltage.iter().any(|value| !value.is_finite()) {
            return Err("invalid candidate matrix-port voltage");
        }
        let mut current = self.history_rhs.clone();
        for (row, value) in current.iter_mut().enumerate() {
            for (col, &voltage_entry) in voltage.iter().enumerate() {
                *value = self.jacobian[row * self.order + col].mul_add(voltage_entry, *value);
            }
        }
        let mut updated_filters = Vec::with_capacity(self.alphas.len());
        for (&alpha, base) in self.alphas.iter().zip(&self.filter_base) {
            let mut filter = vec![0.; self.order];
            for (row, value) in filter.iter_mut().enumerate() {
                *value = alpha.mul_add(voltage[row], base[row]);
            }
            updated_filters.push(filter);
        }
        if current
            .iter()
            .chain(updated_filters.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err("nonfinite precomputed matrix-port candidate");
        }
        Ok(MatrixTrial {
            current,
            jacobian: self.jacobian.clone(),
            new_voltage: voltage.to_vec(),
            updated_filters,
            revision: self.revision,
            owner_id: self.owner_id,
        })
    }
}

impl MatrixPassivePort {
    /// Aggregate rank-one positive-semidefinite fit atoms by shared pole rate.
    ///
    /// Every positive relaxation rate accumulates two nonnegative
    /// weighted sums of symmetric rank-one outer products. Both resulting
    /// matrices are PSD by construction, without a fresh eigendecomposition.
    ///
    /// # Errors
    /// Invalid directions/rates/weights/voltages or arithmetic overflow.
    pub fn from_rank_one(initial_voltage: Vec<f64>, terms: &[Pole]) -> Result<Self, &'static str> {
        let order = initial_voltage.len();
        if order == 0 || initial_voltage.iter().any(|v| !v.is_finite()) {
            return Err("invalid accepted matrix-port voltage");
        }
        let mut groups: BTreeMap<u64, RateBlock> = BTreeMap::new();
        for term in terms {
            if term.direction.len() != order
                || !term.rate.is_finite()
                || term.rate <= 0.
                || !term.high.is_finite()
                || !term.low.is_finite()
                || term.high < 0.
                || term.low < 0.
            {
                return Err("invalid positive-real shared-pole coefficient");
            }
            let norm = term
                .direction
                .iter()
                .map(|entry| entry * entry)
                .sum::<f64>();
            if !norm.is_finite() || (norm - 1.).abs() > 1e-9 {
                return Err("nonunit shared-pole PSD atom");
            }
            let block = groups
                .entry(term.rate.to_bits())
                .or_insert_with(|| RateBlock {
                    rate: term.rate,
                    high: vec![0.; order * order],
                    low: vec![0.; order * order],
                    accepted_filtered: initial_voltage.clone(),
                });
            for row in 0..order {
                for col in 0..order {
                    let offset = order * row + col;
                    let outer = term.direction[row] * term.direction[col];
                    block.high[offset] = term.high.mul_add(outer, block.high[offset]);
                    block.low[offset] = term.low.mul_add(outer, block.low[offset]);
                    if !block.high[offset].is_finite() || !block.low[offset].is_finite() {
                        return Err("matrix fitting coefficient overflow");
                    }
                }
            }
        }
        Ok(Self {
            owner_id: next_port_id()?,
            order,
            blocks: groups.into_values().collect(),
            accepted_voltage: initial_voltage,
            revision: 0,
        })
    }

    /// Number of retained scalar convolution states after PSD coalescing.
    #[must_use]
    pub const fn state_scalars(&self) -> usize {
        self.order * self.blocks.len()
    }

    /// Prepare the accepted-history right-hand side and positive-semidefinite
    /// Newton Jacobian once per candidate integration step. Every subsequent
    /// Newton iteration can reuse this object unchanged.
    ///
    /// # Errors
    /// Invalid step, overflow of fixed history or the matrix coefficients.
    pub fn prepare_step(&self, dt: f64) -> Result<PreparedStep, &'static str> {
        let mut jacobian = vec![0.; self.order * self.order];
        let mut history_rhs = vec![0.; self.order];
        let mut filter_base = Vec::with_capacity(self.blocks.len());
        let mut alphas = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (decay, alpha, beta) = integration_weights(block.rate, dt)?;
            let mut base = vec![0.; self.order];
            for (row, value) in base.iter_mut().enumerate() {
                *value = decay.mul_add(
                    block.accepted_filtered[row],
                    beta * self.accepted_voltage[row],
                );
            }
            for (row, history_value) in history_rhs.iter_mut().enumerate() {
                for (col, &base_value) in base.iter().enumerate() {
                    let idx = row * self.order + col;
                    let high = block.high[idx];
                    let low = block.low[idx];
                    *history_value = (low - high).mul_add(base_value, *history_value);
                    jacobian[idx] += (1. - alpha).mul_add(high, alpha * low);
                }
            }
            filter_base.push(base);
            alphas.push(alpha);
        }
        if jacobian
            .iter()
            .chain(&history_rhs)
            .chain(filter_base.iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err("nonfinite prepared matrix port Jacobian or history source");
        }
        Ok(PreparedStep {
            order: self.order,
            revision: self.revision,
            owner_id: self.owner_id,
            jacobian,
            history_rhs,
            filter_base,
            alphas,
        })
    }

    /// Shorthand for one accepted-history trial; callers with multiple Newton
    /// iterations should prepare the step once and reuse its evaluator.
    ///
    /// # Errors
    /// Invalid dt, voltage or nonfinite coupled candidate current.
    pub fn trial(&self, dt: f64, voltage: &[f64]) -> Result<MatrixTrial, &'static str> {
        self.prepare_step(dt)?.evaluate(voltage)
    }

    /// Commit one successful candidate; stale trials cannot replace newer
    /// accepted history.
    ///
    /// # Errors
    /// Malformed/stale trial or revision overflow.
    pub fn accept(&mut self, trial: MatrixTrial) -> Result<(), &'static str> {
        if trial.revision != self.revision || trial.owner_id != self.owner_id {
            return Err("stale or foreign coupled-matrix filter trial");
        }
        if trial.updated_filters.len() != self.blocks.len() {
            return Err("wrong shared-pole block count");
        }
        if trial.new_voltage.len() != self.order {
            return Err("wrong shared-pole voltage width");
        }
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or("matrix revision overflow")?;
        for (block, filtered) in self.blocks.iter_mut().zip(trial.updated_filters) {
            block.accepted_filtered = filtered;
        }
        self.accepted_voltage = trial.new_voltage;
        self.revision = next_revision;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PassivePort;

    fn terms() -> Vec<Pole> {
        let diagonal = 1. / 2.0_f64.sqrt();
        vec![
            Pole {
                rate: 1e5,
                direction: vec![1., 0., 0.],
                high: 0.1,
                low: 0.03,
            },
            Pole {
                rate: 1e5,
                direction: vec![0., 1., 0.],
                high: 0.15,
                low: 0.2,
            },
            Pole {
                rate: 1e5,
                direction: vec![diagonal, diagonal, 0.],
                high: 0.2,
                low: 0.01,
            },
            Pole {
                rate: 1e8,
                direction: vec![0., 0., 1.],
                high: 0.3,
                low: 0.05,
            },
            Pole {
                rate: 1e8,
                direction: vec![0., diagonal, diagonal],
                high: 0.05,
                low: 0.3,
            },
            Pole {
                rate: 1e8,
                direction: vec![diagonal, 0., -diagonal],
                high: 0.04,
                low: 0.02,
            },
        ]
    }

    fn compare(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        let scale = expected
            .iter()
            .fold(1e-12_f64, |largest, value| largest.max(value.abs()));
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance * scale,
                "{actual:0.15e} vs {expected:0.15e}"
            );
        }
    }

    #[test]
    fn aggregation_matches_individual_psd_atoms_through_accepted_and_rejected_steps()
    -> Result<(), &'static str> {
        let specs = terms();
        let initial = vec![0.2, 0.1, -0.5];
        let mut atom = PassivePort::new(initial.clone(), specs.clone())?;
        let mut compact = MatrixPassivePort::from_rank_one(initial, specs.as_slice())?;
        assert_eq!(atom.state_scalars(), 6);
        assert_eq!(compact.state_scalars(), 6); // n=3, two shared poles.
        for (dt, voltage) in [
            (1e-12, [1., 0., 0.]),
            (1e-6, [0.2, -0.1, 1.]),
            (1e-3, [0., 0., 0.]),
            (1e6, [5., -2., 3.]),
        ] {
            let atom_trial = atom.trial(dt, &voltage)?;
            let compact_trial = compact.trial(dt, &voltage)?;
            compare(&compact_trial.current, &atom_trial.current, 1e-12);
            compare(&compact_trial.jacobian, &atom_trial.jacobian, 1e-12);
            // Evaluate another rejected Newton iterate at the same
            // candidate time. It must not affect the accepted state.
            let _ = atom.trial(dt, &[7., 8., 9.])?;
            let _ = compact.trial(dt, &[7., 8., 9.])?;
            atom.accept(atom_trial)?;
            compact.accept(compact_trial)?;
        }
        Ok(())
    }

    #[test]
    fn many_active_rank_one_terms_share_only_one_vector_state_per_pole() -> Result<(), &'static str>
    {
        let mut specs = Vec::new();
        for term in 0..100 {
            let mut axis = vec![0.; 8];
            axis[term % 8] = 1.;
            specs.push(Pole {
                rate: if term % 2 == 0 { 1e5 } else { 1e9 },
                direction: axis,
                high: 0.01,
                low: 0.02,
            });
        }
        let starting = vec![1.; 8];
        let atom = PassivePort::new(starting.clone(), specs.clone())?;
        let compact = MatrixPassivePort::from_rank_one(starting, &specs)?;
        assert_eq!(atom.state_scalars(), 100);
        assert_eq!(compact.state_scalars(), 16);
        let trial = compact.trial(1e-7, &[1.; 8])?;
        assert!(trial.current.iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    fn coupled_matrix_trial_is_passive_and_recovers_endpoints_for_extreme_dt()
    -> Result<(), &'static str> {
        let port = MatrixPassivePort::from_rank_one(vec![0.; 3], &terms())?;
        for dt in [1e-20, 1e-6, 1., 1e250] {
            let trial = port.trial(dt, &[1., 2., 3.])?;
            for direction in [[1., 0., 0.], [0., 1., 0.], [1., -2., 3.], [-1., 3., 0.]] {
                let energy = (0..3)
                    .map(|row| {
                        direction[row]
                            * (0..3)
                                .map(|col| trial.jacobian[row * 3 + col] * direction[col])
                                .sum::<f64>()
                    })
                    .sum::<f64>();
                assert!(energy >= -1e-12);
            }
        }
        Ok(())
    }

    #[test]
    fn invalid_and_stale_trials_cannot_modify_accepted_compact_state() -> Result<(), &'static str> {
        let mut port = MatrixPassivePort::from_rank_one(vec![0.; 3], &terms())?;
        let before = port.trial(1e-4, &[1., 0., 0.])?;
        assert!(port.trial(-1., &[1., 0., 0.]).is_err());
        assert!(port.trial(1e-4, &[1., f64::NAN, 0.]).is_err());
        let candidate = port.trial(1e-4, &[1., 2., 3.])?;
        port.accept(candidate)?;
        assert!(port.accept(before).is_err());
        let next = port.trial(1e-4, &[0., 0., 0.])?;
        assert!(next.current.iter().all(|v| v.is_finite()));
        Ok(())
    }
    #[test]
    fn prepared_step_reuses_exact_jacobian_and_history_rhs_across_newton_trials()
    -> Result<(), &'static str> {
        let specs = terms();
        let original = vec![1., 0.3, -0.7];
        let mut compact = MatrixPassivePort::from_rank_one(original.clone(), &specs)?;
        let mut independent = PassivePort::new(original, specs)?;
        let prepared = compact.prepare_step(2e-6)?;
        let jacobian = prepared.jacobian().to_vec();
        let history = prepared.accepted_history_rhs().to_vec();
        let mut accepted = None;
        for candidate in [[0., 0., 0.], [2., -3., 1.], [-1., 1., 5.], [1., 1., 1.]] {
            let fused = prepared.evaluate(&candidate)?;
            let reference = independent.trial(2e-6, &candidate)?;
            compare(&fused.jacobian, &reference.jacobian, 1e-12);
            compare(&fused.current, &reference.current, 1e-12);
            assert_eq!(fused.jacobian, jacobian);
            for row in 0..3 {
                let predicted = (0..3).fold(history[row], |value, col| {
                    jacobian[3 * row + col].mul_add(candidate[col], value)
                });
                assert!((fused.current[row] - predicted).abs() < 1e-12);
            }
            accepted = Some(fused);
        }
        let accepted = accepted.ok_or("expected saved trial")?;
        compact.accept(accepted)?;
        independent.accept(independent.trial(2e-6, &[1., 1., 1.])?)?;
        let candidate = prepared.evaluate(&[0., 0., 0.])?;
        assert!(compact.accept(candidate).is_err());
        let next = compact.prepare_step(1e-4)?.evaluate(&[0., 0., 0.])?;
        let independent_next = independent.trial(1e-4, &[0., 0., 0.])?;
        compare(&next.current, &independent_next.current, 1e-12);
        Ok(())
    }
    #[test]
    fn prepared_trials_are_owned_by_one_port_even_if_revisions_match() -> Result<(), &'static str> {
        let specs = terms();
        let mut first = MatrixPassivePort::from_rank_one(vec![0.; 3], &specs)?;
        let mut second = MatrixPassivePort::from_rank_one(vec![0.; 3], &specs)?;
        let from_first = first.prepare_step(1e-6)?.evaluate(&[1., 2., 3.])?;
        assert!(second.accept(from_first.clone()).is_err());
        first.accept(from_first)?;
        let from_second = second.prepare_step(1e-6)?.evaluate(&[2., 1., 0.])?;
        assert!(first.accept(from_second.clone()).is_err());
        second.accept(from_second)?;
        Ok(())
    }
}
