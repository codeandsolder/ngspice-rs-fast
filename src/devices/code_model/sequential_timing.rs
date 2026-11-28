//! Transactional sequential timing checks. This is a deterministic contract
//! over resolved input events, not a simulator-specific timing-corner model.

use super::source_logic::{AsyncPolicy, Edge, Operation};
use super::{Drive, EventContext, Logic, ModelDiagnostic};
use crate::primitives::{Real, SpiceError, SpiceResult};

#[derive(Clone, Copy, Debug, Default)]
pub(super) enum ViolationPolicy {
    #[default]
    Error,
    Unknown,
    Warn,
}

/// Optional timing declarations. Explicit zero means no required interval;
/// omission never substitutes a finite propagation delay.
#[derive(Clone, Debug, Default)]
pub(super) struct TimingSpec {
    setup: Option<Vec<Real>>,
    hold: Option<Vec<Real>>,
    clock_min: Option<[Real; 2]>,
    async_min: Option<[Real; 2]>,
    recovery: Option<[Real; 2]>,
    policy: Option<ViolationPolicy>,
}

impl TimingSpec {
    pub(super) fn set_vector(&mut self, name: &str, values: Vec<Real>) -> SpiceResult<()> {
        if values.iter().any(|value| !value.is_finite() || *value < 0.) {
            return Err(SpiceError::circuit("invalid sequential timing interval"));
        }
        let pair = |values: Vec<Real>| {
            values
                .try_into()
                .map_err(|_| SpiceError::circuit(format!("{name} requires two intervals")))
        };
        match name {
            "timing_setup" => self.setup = Some(values),
            "timing_hold" => self.hold = Some(values),
            "timing_clock_min" => self.clock_min = Some(pair(values)?),
            "timing_async_min" => self.async_min = Some(pair(values)?),
            "timing_recovery" => self.recovery = Some(pair(values)?),
            _ => return Err(SpiceError::circuit("unregistered timing vector")),
        }
        Ok(())
    }

    pub(super) fn set_policy(&mut self, value: Real) -> SpiceResult<()> {
        self.policy = Some(match value {
            0. => ViolationPolicy::Error,
            1. => ViolationPolicy::Unknown,
            2. => ViolationPolicy::Warn,
            _ => {
                return Err(SpiceError::circuit(
                    "invalid sequential timing violation policy",
                ));
            }
        });
        Ok(())
    }

    pub(super) fn bind(
        &self,
        operation: Operation,
        inputs: usize,
    ) -> SpiceResult<Option<TimingContract>> {
        if self.setup.is_none()
            && self.hold.is_none()
            && self.clock_min.is_none()
            && self.async_min.is_none()
            && self.recovery.is_none()
            && self.policy.is_none()
        {
            return Ok(None);
        }
        let (clock, edge, asynchronous) = match operation {
            Operation::Clocked {
                kind,
                trigger,
                asynchronous,
            } => (kind.arity() - 1, trigger, asynchronous),
            // Transparent latches capture the final data at the closing
            // edge. Setup/hold are relative to that edge, not the opening.
            Operation::Enabled(active) => (
                1,
                if active == Logic::One {
                    Edge::Falling
                } else {
                    Edge::Rising
                },
                None,
            ),
            _ => {
                return Err(SpiceError::circuit(
                    "sequential timing requires clocked or enabled storage",
                ));
            }
        };
        if asynchronous.is_none() && (self.async_min.is_some() || self.recovery.is_some()) {
            return Err(SpiceError::circuit(
                "asynchronous timing requires preset/clear ports",
            ));
        }
        let data = |values: &Option<Vec<Real>>| -> SpiceResult<Vec<Real>> {
            match values {
                Some(values) if values.len() == clock => Ok(values.clone()),
                Some(_) => Err(SpiceError::circuit(
                    "setup/hold vector must match data input count",
                )),
                None => Ok(vec![0.; clock]),
            }
        };
        Ok(Some(TimingContract {
            inputs,
            clock,
            edge,
            asynchronous,
            setup: data(&self.setup)?,
            hold: data(&self.hold)?,
            clock_min: self.clock_min.unwrap_or([0.; 2]),
            async_min: self.async_min.unwrap_or([0.; 2]),
            recovery: self.recovery.unwrap_or([0.; 2]),
            policy: self.policy.unwrap_or_default(),
        }))
    }
}

#[derive(Clone, Debug)]
pub(super) struct TimingContract {
    inputs: usize,
    clock: usize,
    edge: Edge,
    asynchronous: Option<AsyncPolicy>,
    setup: Vec<Real>,
    hold: Vec<Real>,
    clock_min: [Real; 2],
    async_min: [Real; 2],
    recovery: [Real; 2],
    policy: ViolationPolicy,
}

#[derive(Clone, Debug)]
pub(super) struct TimingState {
    previous: Vec<Logic>,
    changed: Vec<Option<Real>>,
    capture: Option<Real>,
}

fn check(
    violations: &mut Vec<ModelDiagnostic>,
    kind: &'static str,
    input: usize,
    previous: Option<Real>,
    now: Real,
    minimum: Real,
) {
    // Strict inequality: exact deadline equality is valid. Compare against
    // the same representable deadline used by an event scheduler, avoiding
    // loss from subtracting nearly equal absolute timestamps.
    if minimum > 0.
        && let Some(previous) = previous
        && (now <= previous || now < previous + minimum)
    {
        violations.push(ModelDiagnostic {
            kind,
            input,
            elapsed: now - previous,
            minimum,
        });
    }
}

fn may_trigger(edge: Edge, before: Logic, after: Logic) -> bool {
    before != after
        && match edge {
            Edge::Rising => before != Logic::One && after != Logic::Zero,
            Edge::Falling => before != Logic::Zero && after != Logic::One,
            Edge::Both => true,
        }
}

impl TimingContract {
    pub(super) fn initial_state(&self) -> TimingState {
        TimingState {
            previous: vec![Logic::Unknown; self.inputs],
            changed: vec![None; self.inputs],
            capture: None,
        }
    }

    fn capture_possible(&self, inputs: &[Drive]) -> bool {
        self.asynchronous.is_none_or(|policy| {
            let preset = inputs[self.clock + 1].logic;
            let clear = inputs[self.clock + 2].logic;
            preset != policy.preset && clear != policy.clear
        })
    }

    /// Validate a resolved input event before a sequential state transition.
    /// Returns true if the explicit policy requires unknown output.
    /// State belongs to the event engine's disposable trial; a failed check
    /// does not publish input history, timer changes or captured state.
    pub(super) fn observe(
        &self,
        context: EventContext,
        inputs: &[Drive],
        state: &mut TimingState,
        diagnostics: &mut Vec<ModelDiagnostic>,
    ) -> SpiceResult<bool> {
        if inputs.len() != self.inputs {
            return Err(SpiceError::circuit(
                "sequential timing input dimension mismatch",
            ));
        }
        if context.initial {
            for (previous, input) in state.previous.iter_mut().zip(inputs) {
                *previous = input.logic;
            }
            // Initial levels establish history, but no observed assertion
            // edge exists before time zero.
            state.changed.fill(None);
            state.capture = None;
            return Ok(false);
        }
        let now = context.time;
        let mut violations = Vec::new();
        let capture_possible = self.capture_possible(inputs);
        if capture_possible {
            for (index, (&previous, input)) in state
                .previous
                .iter()
                .zip(inputs)
                .take(self.clock)
                .enumerate()
            {
                if previous != input.logic {
                    check(
                        &mut violations,
                        "hold",
                        index,
                        state.capture,
                        now,
                        self.hold[index],
                    );
                }
            }
        } else {
            // A known asynchronous assertion supersedes the pending
            // synchronous storage capture and its hold interval.
            state.capture = None;
        }
        let before = state.previous[self.clock];
        let after = inputs[self.clock].logic;
        if before != after {
            let minimum = match before {
                Logic::Zero => self.clock_min[0],
                Logic::One => self.clock_min[1],
                Logic::Unknown => 0.,
            };
            check(
                &mut violations,
                "clock/gate pulse width",
                self.clock,
                state.changed[self.clock],
                now,
                minimum,
            );
        }
        if let Some(policy) = self.asynchronous {
            for (offset, active) in [policy.preset, policy.clear].into_iter().enumerate() {
                let index = self.clock + 1 + offset;
                if state.previous[index] == active && inputs[index].logic != active {
                    check(
                        &mut violations,
                        "asynchronous pulse width",
                        index,
                        state.changed[index],
                        now,
                        self.async_min[offset],
                    );
                }
            }
        }
        for (index, (previous, input)) in state.previous.iter_mut().zip(inputs).enumerate() {
            if *previous != input.logic {
                state.changed[index] = Some(now);
                *previous = input.logic;
            }
        }
        if capture_possible && may_trigger(self.edge, before, after) {
            for (index, &minimum) in self.setup.iter().enumerate() {
                check(
                    &mut violations,
                    "setup",
                    index,
                    state.changed[index],
                    now,
                    minimum,
                );
            }
            if self.asynchronous.is_some() {
                for (offset, &minimum) in self.recovery.iter().enumerate() {
                    let index = self.clock + 1 + offset;
                    check(
                        &mut violations,
                        "asynchronous recovery",
                        index,
                        state.changed[index],
                        now,
                        minimum,
                    );
                }
            }
            state.capture = Some(now);
        }
        if violations.is_empty() {
            return Ok(false);
        }
        if matches!(self.policy, ViolationPolicy::Error) {
            let violation = &violations[0];
            return Err(SpiceError::circuit(format!(
                "{} violation at t={now:e}: input {} interval {:e} < required {:e}",
                violation.kind,
                violation.input + 1,
                violation.elapsed,
                violation.minimum
            )));
        }
        // A known preset/clear drives the result independently of a
        // violating clock or data interval. An async-width violation is
        // visible even when the other asynchronous input is inactive.
        let unknown = matches!(self.policy, ViolationPolicy::Unknown)
            && (capture_possible
                || violations
                    .iter()
                    .any(|violation| violation.kind == "asynchronous pulse width"));
        diagnostics.extend(violations);
        Ok(unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::super::source_logic::AsyncPolicy;
    use super::*;

    fn contract(policy: Real) -> SpiceResult<TimingContract> {
        let mut spec = TimingSpec::default();
        for (name, values) in [
            ("timing_setup", vec![2.]),
            ("timing_hold", vec![1.]),
            ("timing_clock_min", vec![3., 3.]),
            ("timing_async_min", vec![2., 2.]),
            ("timing_recovery", vec![2., 2.]),
        ] {
            spec.set_vector(name, values)?;
        }
        spec.set_policy(policy)?;
        let operation = Operation::from_value(
            11.,
            4,
            Some(AsyncPolicy {
                preset: Logic::One,
                clear: Logic::One,
                both: Logic::Unknown,
            }),
        )?;
        spec.bind(operation, 4)?
            .ok_or_else(|| SpiceError::circuit("missing timing contract"))
    }

    fn sample(
        contract: &TimingContract,
        state: &mut TimingState,
        time: Real,
        bits: [Logic; 4],
    ) -> SpiceResult<bool> {
        contract.observe(
            EventContext {
                time,
                delta: 0,
                initial: time == 0.,
            },
            &bits.map(Drive::strong),
            state,
            &mut Vec::new(),
        )
    }

    #[test]
    fn exact_intervals_and_transactional_violations() -> SpiceResult<()> {
        use Logic::{One, Zero};
        let timing = contract(0.)?;
        let mut state = timing.initial_state();
        sample(&timing, &mut state, 0., [Zero, Zero, Zero, Zero])?;
        sample(&timing, &mut state, 1., [One, Zero, Zero, Zero])?;
        // Setup equality at t=3 and hold equality at t=4 are valid.
        sample(&timing, &mut state, 3., [One, One, Zero, Zero])?;
        sample(&timing, &mut state, 4., [Zero, One, Zero, Zero])?;
        let mut rejected = state.clone();
        let err = sample(&timing, &mut rejected, 5., [Zero, Zero, Zero, Zero]).unwrap_err();
        assert!(err.to_string().contains("pulse width"));
        sample(&timing, &mut state, 6., [Zero, Zero, Zero, Zero])?;
        sample(&timing, &mut state, 7., [One, Zero, Zero, Zero])?;
        sample(&timing, &mut state, 9., [One, One, Zero, Zero])?;
        let err = sample(&timing, &mut state.clone(), 9.5, [Zero, One, Zero, Zero]).unwrap_err();
        assert!(err.to_string().contains("hold"));
        Ok(())
    }

    #[test]
    fn async_recovery_and_pulse_width_are_distinct() -> SpiceResult<()> {
        use Logic::{One, Zero};
        let timing = contract(0.)?;
        let mut state = timing.initial_state();
        sample(&timing, &mut state, 0., [Zero, Zero, Zero, Zero])?;
        sample(&timing, &mut state, 1., [Zero, Zero, One, Zero])?;
        let err = sample(&timing, &mut state.clone(), 2., [Zero, Zero, Zero, Zero]).unwrap_err();
        assert!(err.to_string().contains("asynchronous pulse"));
        sample(&timing, &mut state, 3., [Zero, Zero, Zero, Zero])?;
        let err = sample(&timing, &mut state.clone(), 4., [Zero, One, Zero, Zero]).unwrap_err();
        assert!(err.to_string().contains("recovery"));
        sample(&timing, &mut state, 5., [Zero, One, Zero, Zero])?;
        let timing = contract(1.)?;
        let mut state = timing.initial_state();
        sample(&timing, &mut state, 0., [Zero, Zero, Zero, Zero])?;
        sample(&timing, &mut state, 1., [One, Zero, Zero, Zero])?;
        assert!(sample(&timing, &mut state, 2., [One, One, Zero, Zero])?);
        Ok(())
    }
}
