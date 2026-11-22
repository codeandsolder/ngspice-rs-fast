//! A charge-storage branch with its adjacent C state slots. Device
//! equations choose when to seed, integrate and store the branch; the
//! history lookup and companion construction are independent of the model.

use crate::devices::DeviceState;
use crate::maths::integrator::{ChargeHistory, Coefficients, Companion};
use crate::primitives::{Real, SpiceError, SpiceResult};

#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct ChargeBranch {
    slot: usize,
    pub(in crate::devices) charge: Real,
    pub(in crate::devices) capacitance: Real,
}

impl ChargeBranch {
    pub(in crate::devices) const fn new(slot: usize, charge: Real, capacitance: Real) -> Self {
        Self {
            slot,
            charge,
            capacitance,
        }
    }

    /// C's MODEINITTRAN charge write. Must precede integration when used.
    pub(in crate::devices) fn seed(self, states: &mut DeviceState<'_>) -> SpiceResult<()> {
        states.patch_state1(self.slot, self.charge)
    }

    /// C's initial-transient derivative write, after `NIintegrate`.
    pub(in crate::devices) fn seed_derivative(
        self,
        states: &mut DeviceState<'_>,
        derivative: Real,
    ) -> SpiceResult<()> {
        states.patch_state1(self.slot + 1, derivative)
    }

    pub(in crate::devices) fn store(self, states: &mut DeviceState<'_>) -> SpiceResult<()> {
        states.set(self.slot, self.charge)
    }

    pub(in crate::devices) fn zero_derivative(
        self,
        states: &mut DeviceState<'_>,
    ) -> SpiceResult<()> {
        states.set(self.slot + 1, 0.)
    }

    /// Retain a derivative/current slot that the C device does not update
    /// in this mode, including a small-signal capacitance carried across
    /// analyses. A newer Newton iterate takes precedence over carried state.
    pub(in crate::devices) fn retain_derivative(
        self,
        states: &mut DeviceState<'_>,
    ) -> SpiceResult<Real> {
        let slot = self.slot + 1;
        let value = states
            .iterate(slot)
            .or_else(|| states.carried(slot))
            .unwrap_or(0.);
        states.set(slot, value)?;
        Ok(value)
    }

    /// The derivative state adjacent to this charge.
    pub(in crate::devices) const fn derivative_slot(self) -> usize {
        self.slot + 1
    }

    /// `NIintegrate` stores the derivative, but does not advance accepted
    /// history or write the charge slot. The device owns those operations.
    #[inline]
    pub(in crate::devices) fn integrate(
        self,
        states: &mut DeviceState<'_>,
        coefficients: &Coefficients,
    ) -> SpiceResult<Companion> {
        let mut history = ChargeHistory::new(self.charge);
        for age in 1..=coefficients.charge_history_len() {
            let Some(charge) = states.accepted(age, self.slot) else {
                return Err(missing_history("missing accepted junction charge"));
            };
            history.push(charge)?;
        }
        let previous = if coefficients.needs_previous_derivative() {
            let Some(derivative) = states.accepted(1, self.slot + 1) else {
                return Err(missing_history("missing accepted junction derivative"));
            };
            Some(derivative)
        } else {
            None
        };
        let companion = coefficients.integrate(&history, previous, self.capacitance)?;
        states.set(self.slot + 1, companion.derivative)?;
        Ok(companion)
    }
}

#[cold]
#[inline(never)]
fn missing_history(message: &str) -> SpiceError {
    SpiceError::circuit(message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::{IterationPhase, StateHistory};
    use crate::maths::integrator::{IntegrationMethod, StepHistory};

    #[test]
    fn charge_lifecycle_does_not_commit_rejected_trials() {
        let mut history = StateHistory::new(2);
        let mut initial = history.trial();
        let branch = ChargeBranch::new(0, 2., 3.);
        {
            let mut states = history.device(&mut initial, 0..2).unwrap();
            branch.store(&mut states).unwrap();
            branch.zero_derivative(&mut states).unwrap();
        }
        history.commit(initial).unwrap();
        let steps = StepHistory::new();
        let coefficients = steps
            .trial(IntegrationMethod::Trapezoidal, 1, 0.25, 0.5)
            .unwrap();
        let branch = ChargeBranch::new(0, 5., 3.);
        let mut trial = history.trial_in(IterationPhase::Predict, None).unwrap();
        {
            let mut states = history.device(&mut trial, 0..2).unwrap();
            let companion = branch.integrate(&mut states, &coefficients).unwrap();
            assert_eq!(companion.derivative, 12.);
            assert_eq!(companion.conductance, 12.);
            assert!(states.trial(0).unwrap().is_nan());
            branch.seed(&mut states).unwrap();
            let companion = branch.integrate(&mut states, &coefficients).unwrap();
            assert_eq!(companion.derivative, 0.);
            branch
                .seed_derivative(&mut states, companion.derivative)
                .unwrap();
            branch.store(&mut states).unwrap();
            assert_eq!(states.accepted(1, 0), Some(5.));
        }
        assert_eq!(history.accepted(1), Some(&[2., 0.][..]));
        drop(trial);
        assert_eq!(history.accepted(1), Some(&[2., 0.][..]));
    }

    #[test]
    fn retained_derivative_prefers_iterate_then_carried_state() {
        let branch = ChargeBranch::new(0, 2., 3.);
        let history = StateHistory::new(2).with_carried(Some(vec![1., 4.]));
        let mut first = history.trial();
        {
            let mut states = history.device(&mut first, 0..2).unwrap();
            assert_eq!(branch.retain_derivative(&mut states).unwrap(), 4.);
            states.set(1, 7.).unwrap();
        }
        let mut trial = history
            .trial_in(IterationPhase::Float, Some(&first))
            .unwrap();
        let mut states = history.device(&mut trial, 0..2).unwrap();
        assert_eq!(branch.retain_derivative(&mut states).unwrap(), 7.);
        assert_eq!(states.trial(1), Some(7.));
        let empty = StateHistory::new(2);
        let mut fresh = empty.trial();
        let mut states = empty.device(&mut fresh, 0..2).unwrap();
        assert_eq!(branch.retain_derivative(&mut states).unwrap(), 0.);
    }

    #[test]
    fn trapezoidal_and_gear_read_the_correct_history() {
        let mut history = StateHistory::new(2);
        let mut steps = StepHistory::new();
        for (q, dq) in [(1., 0.5), (2., 1.5), (4., 3.)] {
            let mut trial = history.trial();
            let mut states = history.device(&mut trial, 0..2).unwrap();
            states.set(0, q).unwrap();
            states.set(1, dq).unwrap();
            history.commit(trial).unwrap();
            let coeff = steps
                .trial(IntegrationMethod::Trapezoidal, 1, 0.25, 0.5)
                .unwrap();
            steps.accept(&coeff);
        }
        for (method, order) in [
            (IntegrationMethod::Trapezoidal, 2),
            (IntegrationMethod::Gear { order: 3 }, 3),
        ] {
            let coeff = steps.trial(method, order, 0.5, 0.5).unwrap();
            let mut trial = history.trial();
            let mut states = history.device(&mut trial, 0..2).unwrap();
            let branch = ChargeBranch::new(0, 6., 4.);
            let actual = branch.integrate(&mut states, &coeff).unwrap();
            let expected = coeff.integrate(&[6., 4., 2., 1.], Some(3.), 4.).unwrap();
            assert_eq!(actual, expected);
            assert_eq!(states.trial(1), Some(expected.derivative));
        }
    }
}
