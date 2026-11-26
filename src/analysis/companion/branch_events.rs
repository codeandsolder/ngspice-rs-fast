//! Localize executed behavioural guard transitions using disposable physical
//! derivative-array solves against the same accepted integration history.

use super::{AttemptedStep, Boundary, Driver, ReadyStep, StepClock, Trial, TrialPoint, failure};
use crate::maths::Vector;
use crate::primitives::{Real, SpiceResult};

/// Adjacent representable times enclosing a branch transition. If the
/// right time itself admits the old left germ, use it; otherwise the
/// left point is the last representable old-branch state.
#[derive(Clone, Copy)]
pub(super) struct BranchBracket {
    pub left: Real,
    pub right: Real,
}

pub(super) enum Localization {
    Event(BranchBracket),
    Retry(Real),
}

impl Driver<'_> {
    /// A rate-only truncation estimate cannot see a jump when both endpoint
    /// derivatives vanish. Compare the actual derived coordinate increment
    /// with the endpoint rate envelope, without imposing another DAE row.
    pub(super) fn derived_storage_retry(
        &self,
        attempt: &AttemptedStep,
        trial: &crate::devices::TrialState,
    ) -> SpiceResult<Option<super::NextStep>> {
        let dt = attempt.clock.next.delta;
        let tol = &self.settings.tolerances;
        let old = self
            .history
            .accepted(1)
            .ok_or_else(|| failure("missing accepted derivative history"))?;
        for (device, truncation) in self.circuit.devices().iter().zip(&self.truncation) {
            for slot in truncation.slots.iter().filter(|slot| slot.derived) {
                let [q0, r0] = old[slot.base..=slot.base + 1] else {
                    unreachable!()
                };
                let [q1, r1] = trial.values()[slot.base..=slot.base + 1] else {
                    unreachable!()
                };
                let error = (q1 - q0 - dt * Real::midpoint(r0, r1)).abs();
                let envelope = dt * 0.5 * (r1 - r0).abs();
                let tolerance = tol.reltol.mul_add(q0.abs().max(q1.abs()), tol.vntol);
                if error > envelope + tolerance {
                    if dt * 0.5 <= self.settings.min_step() {
                        return Err(failure(format!(
                            "{}: unresolved derivative input increment at t = {:e}; change {:e} is inconsistent with endpoint rates {:e}, {:e}",
                            device.name(),
                            attempt.clock.time,
                            q1 - q0,
                            r0,
                            r1
                        )));
                    }
                    return Ok(Some(super::NextStep {
                        delta: dt * 0.5,
                        order: super::Order::One,
                    }));
                }
            }
        }
        Ok(None)
    }

    pub(super) fn limit_branch_probe(&self, ready: &mut ReadyStep) {
        if self.accepted_guards.is_empty() {
            return;
        }
        let start = ready.clock.time;
        let end = ready.end_time();
        let duration = end - start;
        if let Some(offset) = self
            .accepted_guards
            .predicted_offset(duration, self.timing.step())
        {
            let at = (start + offset).max(start.next_up());
            if at < end {
                // This is only a proposed inspection point, not a physical
                // event. A converged guard observation must confirm it.
                ready.clock.next.delta = at - start;
                ready.event_landing = None;
                ready.probe_landing = Some(at);
                ready.boundary = Boundary::Ordinary;
            }
        }
    }

    pub(super) fn branch_probe_step(
        start: StepClock,
        original: &AttemptedStep,
        end: Real,
    ) -> ReadyStep {
        let mut clock = start;
        clock.next = original.clock.next;
        clock.next.delta = end - start.time;
        ReadyStep {
            clock,
            boundary: Boundary::Ordinary,
            event_landing: None,
            probe_landing: Some(end),
        }
    }

    fn probe_branch(
        &self,
        start: StepClock,
        original: &AttemptedStep,
        time: Real,
        left: bool,
        seed: &Vector,
    ) -> SpiceResult<Trial> {
        let mut ready = Self::branch_probe_step(start, original, time);
        // Bind the exact probe time rather than reconstituting it by
        // adding a rounded delta. For a right probe no event is accepted.
        if left {
            ready.limit_to_event(time);
        }
        let attempt = ready.begin(&self.steps, &self.settings)?;
        self.derivative_trial(&attempt, seed)
    }

    pub(super) fn localize_branch(
        &mut self,
        start: StepClock,
        seed: &Vector,
        original: &AttemptedStep,
        endpoint: &TrialPoint,
    ) -> SpiceResult<Localization> {
        let mut bracket = BranchBracket {
            left: start.time,
            right: original.clock.time,
        };
        let device = self
            .accepted_guards
            .changed_device(&endpoint.guards)
            .and_then(|index| self.circuit.devices().get(index))
            .map_or("behavioural expression", |device| device.name())
            .to_owned();
        // Fast path for a representable root: its left and right germs at
        // the same time select different branches. Taylor predictions are
        // not trusted without this independent solve.
        self.count_branch_probe()?;
        if original.event_start.is_none()
            && let Trial::Converged(left) =
                self.probe_branch(start, original, bracket.right, true, seed)?
            && self.accepted_guards.same_regions(&left.guards)
        {
            return Ok(Localization::Event(bracket));
        }
        // A non-representable root (or an imperfect prediction) requires
        // an actual bracket. Probe the candidate trajectory, not a linear
        // interpolation of algebraic variables or a stale Newton iterate.
        for _ in 0..128 {
            if bracket.left.next_up() >= bracket.right {
                return Ok(Localization::Event(bracket));
            }
            let at = Real::midpoint(bracket.left, bracket.right);
            self.count_branch_probe()?;
            match self.probe_branch(start, original, at, false, seed)? {
                Trial::Converged(point) => {
                    if self.accepted_guards.same_regions(&point.guards) {
                        bracket.left = at;
                    } else {
                        bracket.right = at;
                    }
                }
                Trial::NotConverged { .. } => {
                    let retry = Real::midpoint(start.time, at);
                    if retry <= start.time || retry - start.time <= self.settings.min_step() {
                        return Err(failure(format!(
                            "{device}: unresolved behavioural branch crossing at t = {at:e}"
                        )));
                    }
                    return Ok(Localization::Retry(retry));
                }
            }
        }
        Err(failure(format!(
            "{device}: behavioural branch isolation budget exhausted at t = {:e}..{:e}",
            bracket.left, bracket.right
        )))
    }

    fn count_branch_probe(&mut self) -> SpiceResult<()> {
        if self.stats.accepted + self.stats.rejected >= self.settings.max_steps {
            return Err(failure("work limit during behavioural branch isolation"));
        }
        self.stats.rejected += 1;
        Ok(())
    }

    pub(super) fn branch_left_point(
        &self,
        start: StepClock,
        seed: &Vector,
        bracket: BranchBracket,
        attempt: &AttemptedStep,
    ) -> SpiceResult<TrialPoint> {
        // At a non-representable root the exact right time may already
        // select the new region even when evaluated with a left germ.
        // Reuse the last representable old-region point's storage, never
        // the new region's storage, for the continuity check.
        if bracket.left.next_up() < bracket.right {
            return Err(failure(
                "behavioural branch left germ changed after localization",
            ));
        }
        if bracket.left <= start.time {
            let values = self
                .history
                .accepted(1)
                .ok_or_else(|| failure("missing accepted branch history"))?;
            let mut state = self.history.trial();
            for (slot, &value) in values.iter().enumerate() {
                self.history.set_candidate_slot(&mut state, slot, value)?;
            }
            return Ok(TrialPoint {
                x: seed.clone(),
                state,
                guards: self.accepted_guards.clone(),
            });
        }
        match self.probe_branch(start, attempt, bracket.left, false, seed)? {
            Trial::Converged(point) if self.accepted_guards.same_regions(&point.guards) => {
                Ok(point)
            }
            _ => Err(failure(format!(
                "could not reproduce behavioural left germ at t = {:e}",
                bracket.left
            ))),
        }
    }
}
