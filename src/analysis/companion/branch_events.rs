//! Localize executed behavioural guard transitions using disposable physical
//! derivative-array solves against the same accepted integration history.

use super::{
    AcceptedCandidate, AcceptedPoint, AttemptedStep, Boundary, DerivativePredictor, Driver,
    NextStep, Order, OutputSelection, PhysicalTrial, ReadyStep, StepClock, StepDecision,
    TrialPoint, failure,
};
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

/// Only accepted start data may anchor a localized crossing. Newton seeds
/// and bracket candidates are disposable and never become this baseline.
struct BranchSearch {
    start: StepClock,
    seed: Vector,
    event: Option<BranchBracket>,
}

impl BranchSearch {
    fn retry(
        &mut self,
        attempt: AttemptedStep,
        next: NextStep,
        min_step: Real,
    ) -> SpiceResult<ReadyStep> {
        self.event = None;
        let mut ready = attempt.retry(next, min_step)?;
        ready.clock.time = self.start.time;
        Ok(ready)
    }
}

enum BranchResolution {
    Point(TrialPoint),
    Probe(ReadyStep),
}

impl Driver<'_> {
    /// Physical derivative-array trials are separate from the companion
    /// Newton loop: guard probes and germ transitions do not need ordinary
    /// transient stamping or mixed-signal preview state.
    pub(super) fn advance_physical(
        &mut self,
        mut ready: ReadyStep,
        mut iterate: Vector,
        output: OutputSelection<'_>,
    ) -> SpiceResult<AcceptedPoint> {
        let min_step = self.settings.min_step();
        let mut search = BranchSearch {
            start: ready.clock,
            seed: iterate.clone(),
            event: None,
        };
        let physical_event = self.next_physical_event(search.start.time);
        loop {
            self.check_step_budget(search.start.time)?;
            if let Some(event) = physical_event {
                ready.limit_to_event(event);
            }
            if let Some(bracket) = search.event {
                ready.limit_to_event(bracket.right);
                if ready.end_time() < bracket.right {
                    search.event = None;
                }
            } else {
                self.limit_branch_probe(&mut ready)?;
            }
            let attempt = ready.begin(&self.steps, &self.settings)?;
            let point = match self.derivative_trial(&attempt, &iterate)? {
                PhysicalTrial::NotConverged { last } => {
                    iterate = last;
                    self.stats.rejected += 1;
                    let next = NextStep {
                        delta: attempt.clock.next.delta / 8.,
                        order: Order::One,
                    };
                    ready = search.retry(attempt, next, min_step)?;
                    continue;
                }
                PhysicalTrial::Converged(point) => point,
            };
            let point = match self.resolve_branch(&mut search, &attempt, point)? {
                BranchResolution::Point(point) => point,
                BranchResolution::Probe(probe) => {
                    ready = probe;
                    iterate = search.seed.clone();
                    self.stats.rejected += 1;
                    continue;
                }
            };
            let decision = if let Some(next) = self.derived_storage_retry(&attempt, &point.state)? {
                StepDecision::Retry(next)
            } else {
                attempt.decision(&self.settings, &self.steps, |coefficients| {
                    self.truncation_limit(coefficients, &point.state, &point.x, attempt.clock.time)
                })?
            };
            match decision {
                StepDecision::Retry(next) => {
                    iterate = point.x;
                    self.stats.rejected += 1;
                    ready = search.retry(attempt, next, min_step)?;
                }
                StepDecision::Accept(next) => {
                    return self.accept_physical(attempt, next, point, &output);
                }
            }
        }
    }

    fn resolve_branch(
        &mut self,
        search: &mut BranchSearch,
        attempt: &AttemptedStep,
        point: TrialPoint,
    ) -> SpiceResult<BranchResolution> {
        let changed = !self.accepted_guards.same_regions(&point.guards);
        if let Some(bracket) = search.event {
            return Ok(BranchResolution::Point(if changed {
                self.count_branch_probe()?;
                self.branch_left_point(search.start, &search.seed, bracket, attempt)?
            } else {
                point
            }));
        }
        if changed {
            let ready = match self.localize_branch(search.start, &search.seed, attempt, &point)? {
                Localization::Event(bracket) => {
                    let mut ready = Self::branch_probe_step(search.start, attempt, bracket.right);
                    ready.limit_to_event(bracket.right);
                    search.event = Some(bracket);
                    ready
                }
                Localization::Retry(at) => Self::branch_probe_step(search.start, attempt, at),
            };
            return Ok(BranchResolution::Probe(ready));
        }
        if let Some(at) = self.backward_branch_probe(search.start, attempt, &point)? {
            return Ok(BranchResolution::Probe(Self::branch_probe_step(
                search.start,
                attempt,
                at,
            )));
        }
        Ok(BranchResolution::Point(point))
    }

    fn accept_physical(
        &mut self,
        attempt: AttemptedStep,
        mut next: NextStep,
        point: TrialPoint,
        output: &OutputSelection<'_>,
    ) -> SpiceResult<AcceptedPoint> {
        let event = self.crosses_physical_event(&attempt);
        let TrialPoint {
            x,
            state,
            guards,
            seed,
        } = if event {
            next.order = Order::One;
            self.derivative_event(&attempt, &point)?
        } else {
            point
        };
        let time = attempt.clock.time;
        let limit = if attempt.event_start.is_some() && time >= self.settings.stop {
            crate::devices::Limit::Left
        } else {
            crate::devices::Limit::Right
        };
        let candidate = AcceptedCandidate {
            output: self.candidate_output(&attempt, &x, &state, output, || {
                self.observe_derivative_point(&x, Some(&state), time, limit, output.selected)
            })?,
            mixed: None,
            solution: x,
            state,
        };
        let accepted = self.commit_candidate(attempt, next, candidate, None)?;
        self.accepted_guards = guards;
        self.derivative_predictor = Some(DerivativePredictor {
            time,
            state: accepted.solution.clone(),
            seed,
        });
        Ok(accepted)
    }

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
                let range = slot.base..=slot.base + 1;
                let Some(&[q0, r0]) = old.get(range.clone()) else {
                    return Err(failure("missing accepted derivative slots"));
                };
                let Some(&[q1, r1]) = trial.values().get(range) else {
                    return Err(failure("missing trial derivative slots"));
                };
                let error = (-dt).mul_add(Real::midpoint(r0, r1), q1 - q0).abs();
                let envelope = dt * 0.5 * (r1 - r0).abs();
                let tolerance = tol.reltol.mul_add(q0.abs().max(q1.abs()), tol.vntol);
                if error > envelope + tolerance {
                    if dt * 0.5 <= self.settings.min_step() {
                        return Err(failure(format!(
                            "{}: unresolved derivative input increment at t = {:e}; change {:e} is inconsistent with endpoint rates {r0:e}, {r1:e}",
                            device.name(),
                            attempt.clock.time,
                            q1 - q0
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

    pub(super) fn limit_branch_probe(&self, ready: &mut ReadyStep) -> SpiceResult<()> {
        if self.accepted_guards.is_empty() {
            return Ok(());
        }
        let start = ready.clock.time;
        let end = ready.end_time();
        let duration = end - start;
        if let Some(offset) = self
            .accepted_guards
            .predicted_offset(duration, self.timing.step())
            .map_err(|error| {
                failure(format!(
                    "behavioural branch prediction at t = {start:e}: {error}"
                ))
            })?
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
        Ok(())
    }

    /// If the endpoint's backward germ predicts a hidden interior root,
    /// inspect that time before accepting two identical region signatures.
    pub(super) fn backward_branch_probe(
        &self,
        start: StepClock,
        attempt: &AttemptedStep,
        point: &TrialPoint,
    ) -> SpiceResult<Option<Real>> {
        let end = attempt.clock.time;
        let Some(offset) = point
            .guards
            .predicted_backward_offset(
                end - start.time,
                self.timing.step(),
                attempt.event_start.is_some(),
            )
            .map_err(|error| {
                failure(format!(
                    "behavioural backward branch prediction at t = {end:e}: {error}"
                ))
            })?
        else {
            return Ok(None);
        };
        let at = end - offset;
        // Adjacent floating-point endpoint roots are already covered by
        // the endpoint germ comparison. Avoid an endless probe loop.
        Ok((at > start.time.next_up() && at < end.next_down()).then_some(at))
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
    ) -> SpiceResult<PhysicalTrial> {
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
            && let PhysicalTrial::Converged(left) =
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
                PhysicalTrial::Converged(point) => {
                    if self.accepted_guards.same_regions(&point.guards) {
                        bracket.left = at;
                    } else {
                        bracket.right = at;
                    }
                }
                PhysicalTrial::NotConverged { .. } => {
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
                seed: self
                    .derivative_predictor
                    .as_ref()
                    .ok_or_else(|| failure("missing accepted physical predictor"))?
                    .seed
                    .clone(),
            });
        }
        match self.probe_branch(start, attempt, bracket.left, false, seed)? {
            PhysicalTrial::Converged(point) if self.accepted_guards.same_regions(&point.guards) => {
                Ok(point)
            }
            _ => Err(failure(format!(
                "could not reproduce behavioural left germ at t = {:e}",
                bracket.left
            ))),
        }
    }
}
