//! Causal timing constraints, expressed in the relative rather than
//! absolute arrival domain. Bounds may be sound enclosures, so a
//! possible violation is not reported as a definite one.

use super::{Bounds, TimingWindow};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Whether the enclosure proves that a timing condition is satisfied or
/// violated, or still permits both outcomes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Certainty {
    /// Every possible separation satisfies the condition.
    Satisfied,
    /// The enclosure admits a violation but does not establish one.
    PossibleViolation,
    /// Every possible separation violates the condition.
    DefiniteViolation,
}

/// Checked relative separation for one condition.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConstraintResult {
    /// Later minus earlier, after cancellation of shared delay origins.
    pub separation: Bounds,
    /// Required minimum separation, in seconds.
    pub minimum: Real,
    /// Classification of the observed enclosure.
    pub certainty: Certainty,
}

impl ConstraintResult {
    /// Validate a minimum separation against already computed relative
    /// bounds. Equality with the minimum satisfies the constraint.
    ///
    /// # Errors
    /// Negative or nonfinite minimum.
    pub fn from_bounds(separation: Bounds, minimum: Real) -> SpiceResult<Self> {
        if !minimum.is_finite() || minimum < 0. {
            return Err(SpiceError::circuit("invalid timing constraint minimum"));
        }
        let certainty = if separation.minimum() >= minimum {
            Certainty::Satisfied
        } else if separation.maximum() < minimum {
            Certainty::DefiniteViolation
        } else {
            Certainty::PossibleViolation
        };
        Ok(Self {
            separation,
            minimum,
            certainty,
        })
    }

    /// Check the minimum interval between two causal arrivals.
    ///
    /// # Errors
    /// Invalid minimum or timing-expression comparison failure.
    pub fn minimum(
        later: &TimingWindow,
        earlier: &TimingWindow,
        minimum: Real,
        budget: usize,
    ) -> SpiceResult<Self> {
        Self::from_bounds(later.relative_to(earlier, budget)?, minimum)
    }
}

/// A data transition checked against a clock edge. Both constraints
/// share the same signed relative enclosure, even when absolute
/// arrival envelopes overlap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ApertureResult {
    /// Capture minus data arrival, in seconds.
    pub capture_minus_data: Bounds,
    /// Data may fall in the setup aperture (including exact coincidence).
    pub setup: Certainty,
    /// Data may fall in the hold aperture (strictly after capture).
    pub hold: Certainty,
}

impl ApertureResult {
    /// Check a data edge against the setup/hold aperture. A zero
    /// declaration disables that side. At exact coincidence, the
    /// data-before-clock convention assigns the event to setup.
    ///
    /// # Errors
    /// Invalid intervals or timing-expression comparison failure.
    pub fn check(
        capture: &TimingWindow,
        data: &TimingWindow,
        setup: Real,
        hold: Real,
        budget: usize,
    ) -> SpiceResult<Self> {
        Self::from_bounds(capture.relative_to(data, budget)?, setup, hold)
    }

    /// Classify both aperture sides using an existing signed enclosure.
    ///
    /// # Errors
    /// Negative or nonfinite setup/hold.
    pub fn from_bounds(capture_minus_data: Bounds, setup: Real, hold: Real) -> SpiceResult<Self> {
        Self::from_ordered_bounds(capture_minus_data, setup, hold, true)
    }

    /// Classify both aperture sides, assigning a coincident edge to setup
    /// if data is evaluated first and to hold if a later delta delivers it.
    ///
    /// # Errors
    /// Negative or nonfinite setup/hold.
    pub fn from_ordered_bounds(
        capture_minus_data: Bounds,
        setup: Real,
        hold: Real,
        data_first: bool,
    ) -> SpiceResult<Self> {
        if [setup, hold].iter().any(|v| !v.is_finite() || *v < 0.) {
            return Err(SpiceError::circuit("invalid setup/hold aperture"));
        }
        let low = capture_minus_data.minimum();
        let high = capture_minus_data.maximum();
        let after = |value: Real| value < 0. || (value == 0. && !data_first);
        let setup = if setup == 0. || after(high) || low >= setup {
            Certainty::Satisfied
        } else if !after(low) && high < setup {
            Certainty::DefiniteViolation
        } else {
            Certainty::PossibleViolation
        };
        let hold = if hold == 0. || !after(low) || high <= -hold {
            Certainty::Satisfied
        } else if after(high) && low > -hold {
            Certainty::DefiniteViolation
        } else {
            Certainty::PossibleViolation
        };
        Ok(Self {
            capture_minus_data,
            setup,
            hold,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::digital::timing::{DelayWindow, Origin};

    #[test]
    fn common_path_is_cancelled_for_setup_hold_and_pulse_width() -> SpiceResult<()> {
        let root =
            TimingWindow::exact(0.)?.after(DelayWindow::new(15., 30.)?, Origin::shared(0))?;
        let data = root.after(DelayWindow::new(10., 30.)?, Origin::shared(1))?;
        let clock = root.after(DelayWindow::new(40., 60.)?, Origin::shared(2))?;
        for (setup, expected) in [
            (10., Certainty::Satisfied),
            (11., Certainty::PossibleViolation),
            (51., Certainty::DefiniteViolation),
        ] {
            let result = ApertureResult::check(&clock, &data, setup, 10., 100)?;
            assert_eq!(result.capture_minus_data, Bounds::new(10., 50.)?);
            assert_eq!(result.setup, expected);
            assert_eq!(result.hold, Certainty::Satisfied);
            assert_eq!(
                ConstraintResult::minimum(&clock, &data, setup, 100)?.certainty,
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn both_sides_and_exact_boundaries_classified_without_guessing_order() -> SpiceResult<()> {
        let capture =
            TimingWindow::exact(100.)?.after(DelayWindow::new(0., 10.)?, Origin::shared(0))?;
        let data = TimingWindow::exact(105.)?;
        let result = ApertureResult::check(&capture, &data, 2., 3., 100)?;
        assert_eq!(result.capture_minus_data, Bounds::new(-5., 5.)?);
        assert_eq!(result.setup, Certainty::PossibleViolation);
        assert_eq!(result.hold, Certainty::PossibleViolation);
        for (dt, setup, hold) in [
            (-3., Certainty::Satisfied, Certainty::Satisfied),
            (-2., Certainty::Satisfied, Certainty::DefiniteViolation),
            (0., Certainty::DefiniteViolation, Certainty::Satisfied),
            (1., Certainty::DefiniteViolation, Certainty::Satisfied),
            (2., Certainty::Satisfied, Certainty::Satisfied),
        ] {
            let result = ApertureResult::from_bounds(Bounds::new(dt, dt)?, 2., 3.)?;
            assert_eq!((result.setup, result.hold), (setup, hold));
        }
        Ok(())
    }
}
