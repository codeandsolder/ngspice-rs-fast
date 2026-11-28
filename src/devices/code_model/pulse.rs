//! Transactional deterministic pulse qualification. A rejected transition
//! never publishes a full binary pulse. A short stimulus above the explicit
//! rejection threshold publishes a directional ambiguity at the original
//! propagation deadline. Arrival windows in the interval analysis layer
//! are a separate contract.

use super::{Drive, Logic, Strength};
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub(super) struct PulseFilter {
    minimum: Real,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    from: Drive,
    to: Drive,
    since: Real,
    due: Real,
}

#[derive(Clone, Debug, Default)]
pub(super) struct PulseState {
    pending: Option<Candidate>,
    scheduled: BTreeMap<u64, Drive>,
}

impl PulseState {
    pub(super) fn next_wakeup(&self) -> Option<Real> {
        self.pending
            .map(|candidate| candidate.due.to_bits())
            .into_iter()
            .chain(self.scheduled.first_key_value().map(|(&time, _)| time))
            .min()
            .map(Real::from_bits)
    }

    fn schedule(&mut self, time: Real, drive: Drive) {
        let key = time.to_bits();
        // Same output ordering as transport events: an earlier or equal
        // newly posted delivery supersedes later pending deliveries.
        drop(self.scheduled.split_off(&key));
        self.scheduled.insert(key, drive);
    }

    fn deliver(&mut self, now: Real, delivered: &mut Drive) {
        let key = now.to_bits();
        // At an exact qualification deadline the preceding transition
        // is definite. Process it before an input reversal at that time.
        if let Some(candidate) = self.pending
            && candidate.due <= now
        {
            self.pending = None;
            self.schedule(candidate.due, candidate.to);
        }
        while let Some((&time, _)) = self.scheduled.first_key_value() {
            if time > key {
                break;
            }
            if let Some((_, drive)) = self.scheduled.pop_first() {
                *delivered = drive;
            }
        }
    }

    fn tail(&self, delivered: Drive) -> Drive {
        self.scheduled
            .last_key_value()
            .map_or(delivered, |(_, &drive)| drive)
    }
}

/// A possible excursion retains direction only when the electrical
/// connection and both endpoint levels are definite. Otherwise the
/// connected result itself is uncertain.
fn ambiguity(from: Drive, to: Drive) -> Drive {
    if from.strength == Strength::HIGH_IMPEDANCE && to.strength == Strength::HIGH_IMPEDANCE {
        return Drive::default();
    }
    let logic = if from.strength == to.strength {
        match (from.logic, to.logic) {
            (Logic::Zero, Logic::One | Logic::Rising) => Logic::Rising,
            (Logic::One, Logic::Zero | Logic::Falling) => Logic::Falling,
            (Logic::Rising, Logic::Zero | Logic::One) => Logic::Rising,
            (Logic::Falling, Logic::Zero | Logic::One) => Logic::Falling,
            _ => Logic::Unknown,
        }
    } else {
        Logic::Unknown
    };
    Drive {
        logic,
        strength: from.strength.max(to.strength),
    }
}

impl PulseFilter {
    pub(super) fn new(minimum: Real) -> SpiceResult<Self> {
        if !minimum.is_finite() || minimum < 0. {
            return Err(SpiceError::circuit("invalid pulse rejection threshold"));
        }
        Ok(Self { minimum })
    }

    /// Qualification and delivery are both model-state events; neither
    /// mutates the accepted network outside its enclosing trial.
    pub(super) fn evaluate(
        self,
        now: Real,
        desired: Drive,
        propagation: impl FnOnce(Drive, Drive) -> Real,
        requested: &mut Drive,
        delivered: &mut Drive,
        state: &mut PulseState,
    ) -> SpiceResult<()> {
        state.deliver(now, delivered);
        if desired == *requested {
            return Ok(());
        }
        *requested = desired;
        if let Some(candidate) = state.pending.take()
            && now > candidate.since
            && now - candidate.since >= self.minimum
        {
            state.schedule(candidate.due, ambiguity(candidate.from, candidate.to));
        }
        let from = state.tail(*delivered);
        if desired == from {
            return Ok(());
        }
        let delay = propagation(from, desired);
        let due = now + delay;
        if !due.is_finite() || (delay > 0. && due <= now) {
            return Err(SpiceError::circuit(
                "pulse propagation deadline outside representable time range",
            ));
        }
        if delay == 0. {
            state.schedule(now, desired);
            state.deliver(now, delivered);
        } else {
            state.pending = Some(Candidate {
                from,
                to: desired,
                since: now,
                due,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::code_model::Delays;

    struct Harness {
        filter: PulseFilter,
        delays: Delays,
        state: PulseState,
        requested: Drive,
        delivered: Drive,
    }

    impl Harness {
        fn new(initial: Logic, rejection: Real, rise: Real, fall: Real) -> SpiceResult<Self> {
            let drive = Drive::strong(initial);
            Ok(Self {
                filter: PulseFilter::new(rejection)?,
                delays: Delays::new(rise, fall)?,
                state: PulseState::default(),
                requested: drive,
                delivered: drive,
            })
        }

        fn sample(&mut self, time: Real, desired: Logic, expected: Logic) -> SpiceResult<()> {
            self.filter.evaluate(
                time,
                Drive::strong(desired),
                |from, to| self.delays.transition(from, to),
                &mut self.requested,
                &mut self.delivered,
                &mut self.state,
            )?;
            assert_eq!(self.delivered, Drive::strong(expected), "at {time}");
            Ok(())
        }
    }

    #[test]
    fn short_qualifying_pulses_retain_direction_and_deadline() -> SpiceResult<()> {
        for (base, opposite, phase) in [
            (Logic::Zero, Logic::One, Logic::Rising),
            (Logic::One, Logic::Zero, Logic::Falling),
        ] {
            for threshold in [0., 1., 2., 2.5] {
                let mut h = Harness::new(base, threshold, 10., 10.)?;
                let expected = if threshold <= 2. { phase } else { base };
                h.sample(20., opposite, base)?;
                h.sample(22., base, base)?;
                h.sample(30., base, expected)?;
                h.sample(32., base, base)?;
                assert_eq!(h.state.next_wakeup(), None);
            }
        }
        Ok(())
    }

    #[test]
    fn full_propagation_equality_is_not_a_glitch() -> SpiceResult<()> {
        let mut h = Harness::new(Logic::Zero, 1., 10., 10.)?;
        h.sample(20., Logic::One, Logic::Zero)?;
        h.sample(30., Logic::Zero, Logic::One)?;
        h.sample(40., Logic::Zero, Logic::Zero)?;
        Ok(())
    }

    #[test]
    fn shorter_asymmetric_return_supersedes_unseen_glitch() -> SpiceResult<()> {
        let mut h = Harness::new(Logic::Zero, 1., 10., 2.)?;
        h.sample(20., Logic::One, Logic::Zero)?;
        h.sample(22., Logic::Zero, Logic::Zero)?;
        h.sample(24., Logic::Zero, Logic::Zero)?;
        h.sample(30., Logic::Zero, Logic::Zero)?;
        assert_eq!(h.state.next_wakeup(), None);
        Ok(())
    }
}
