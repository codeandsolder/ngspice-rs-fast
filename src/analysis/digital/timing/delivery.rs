//! Typed causal arrivals at the event queue boundary.

use super::TimingWindow;
use crate::devices::code_model::{Drive, Logic, Output};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// One guaranteed transition whose physical time is bounded by an
/// expression. The two boundary deliveries share this entire expression.
#[derive(Clone, Debug)]
pub struct Arrival {
    window: TimingWindow,
    from: Drive,
    to: Drive,
}

impl Arrival {
    /// Construct a transition between stable drives.
    ///
    /// # Errors
    /// An endpoint is itself a directional phase, or the drives agree.
    pub fn new(from: Drive, to: Drive, window: TimingWindow) -> SpiceResult<Self> {
        if from == to
            || [from, to]
                .iter()
                .any(|d| matches!(d.logic, Logic::Rising | Logic::Falling))
        {
            return Err(SpiceError::circuit("invalid causal transition endpoints"));
        }
        Ok(Self { window, from, to })
    }

    /// The pre-transition drive.
    #[must_use = "the endpoint describes the drive before the arrival"]
    pub const fn from(&self) -> Drive {
        self.from
    }

    /// The post-transition drive.
    #[must_use = "the endpoint describes the drive after the arrival"]
    pub const fn to(&self) -> Drive {
        self.to
    }

    /// The complete causal arrival expression.
    #[must_use = "the expression retains shared uncertainty origins"]
    pub const fn window(&self) -> &TimingWindow {
        &self.window
    }

    /// Whether this is the same announced transition.
    #[must_use = "identity pairs the two boundaries of a transition"]
    pub fn same_transition(&self, other: &Self) -> bool {
        self.from == other.from && self.to == other.to && self.window.same_expression(&other.window)
    }

    /// Displayed uncertainty while the event may or may not have arrived.
    #[must_use = "the phase is the drive between arrival bounds"]
    pub fn phase(&self) -> Drive {
        phase_between(self.from, self.to)
    }

    /// Whether the physical transition may still be unresolved at time.
    #[must_use = "the result identifies the active uncertainty interval"]
    pub fn active_at(&self, time: Real) -> bool {
        let bounds = self.window.bounds();
        bounds.minimum() <= time && time < bounds.maximum()
    }
}

// A single uncertain transition between two stable drive values.
pub(in crate::analysis::digital) fn phase_between(from: Drive, to: Drive) -> Drive {
    if from == to {
        return from;
    }
    let logic = if from.strength == to.strength {
        match (from.logic, to.logic) {
            (Logic::Zero, Logic::One) => Logic::Rising,
            (Logic::One, Logic::Zero) => Logic::Falling,
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

/// Resolved input and, when qualified, its latest causal transition.
#[derive(Clone, Debug)]
pub struct TimedInput {
    /// Current drive (R/F or X inside a bounded transition).
    pub drive: Drive,
    /// Latest transition, retained after its completion for comparisons.
    pub arrival: Option<Arrival>,
}

/// Result of evaluating one timed output port.
#[derive(Clone, Debug)]
pub enum TimedOutput {
    /// No new transition; preserve existing queued events.
    Retain,
    /// Ordinary scalar event.
    Scalar(Output),
    /// A guaranteed transition with an absolute causal arrival expression.
    Arrival(Arrival),
    /// Multiple causally ordered transitions requested by a single evaluation.
    Waveform(Vec<Arrival>),
}
