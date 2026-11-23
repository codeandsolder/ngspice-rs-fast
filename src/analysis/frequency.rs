//! Bounded, incremental frequency traversal for the spectral analyses.
//!
//! Each caller owns the reference-specific spacing formula and singleton
//! policy. The shared iterator keeps the actual repeated additions or
//! multiplications, endpoint tolerance and work budget in one place.

use crate::analysis::linear::unsupported;
use crate::primitives::{Real, SpiceResult};

pub(super) const MAX_POINTS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Advance {
    Single,
    Add(Real),
    Multiply(Real),
}

/// A validated, replayable spectral grid. Only coordinates needed by a
/// consumer are produced; validation does not retain an entire sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FrequencyPlan {
    start: Real,
    advance: Advance,
    count: usize,
}

impl FrequencyPlan {
    pub(super) fn new(
        start: Real,
        stop: Real,
        tolerance: Real,
        advance: Advance,
        analysis: &'static str,
    ) -> SpiceResult<Self> {
        // Run the reference loop exactly once at admission, including the
        // final nonadvance/point-budget check. A replay cannot fail later,
        // after a plot has begun emitting rows.
        let count = FrequencyGrid::new(start, stop, tolerance, advance, analysis)?
            .try_fold(0usize, |count, next| next.map(|_| count + 1))?;
        Ok(Self {
            start,
            advance,
            count,
        })
    }

    /// Sensitivity uses count_steps and multiplies even for a non-increasing
    /// factor; it has no endpoint-tolerance/monotonicity test.
    pub(super) fn counted(start: Real, factor: Real, count: usize) -> SpiceResult<Self> {
        let plan = Self {
            start,
            advance: Advance::Multiply(factor),
            count,
        };
        for frequency in plan.values() {
            if !(frequency.is_finite() && frequency > 0.) {
                return Err(super::linear::unsupported(format!(
                    ".sens AC frequency {frequency} is not positive and finite (C would silently use the DC operating-point matrix for a zero frequency)"
                )));
            }
        }
        Ok(plan)
    }

    pub(super) const fn len(&self) -> usize {
        self.count
    }

    pub(super) const fn values(&self) -> FrequencyValues {
        FrequencyValues {
            next: self.start,
            advance: self.advance,
            remaining: self.count,
        }
    }
}

/// Fresh repeated-arithmetic traversal with the exact validated length.
#[derive(Clone)]
pub(super) struct FrequencyValues {
    next: Real,
    advance: Advance,
    remaining: usize,
}

impl Iterator for FrequencyValues {
    type Item = Real;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let frequency = self.next;
        self.remaining -= 1;
        if self.remaining != 0 {
            self.next = match self.advance {
                Advance::Single => frequency,
                Advance::Add(delta) => frequency + delta,
                Advance::Multiply(delta) => frequency * delta,
            };
        }
        Some(frequency)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for FrequencyValues {}

/// A frequency sweep with the C-style inclusive, tolerance-adjusted end.
pub(super) struct FrequencyGrid {
    next: Option<Real>,
    end: Real,
    advance: Advance,
    emitted: usize,
    analysis: &'static str,
}

impl FrequencyGrid {
    pub(super) fn new(
        start: Real,
        stop: Real,
        tolerance: Real,
        advance: Advance,
        analysis: &'static str,
    ) -> SpiceResult<Self> {
        let end = stop + tolerance;
        let valid_advance = match advance {
            Advance::Single => true,
            Advance::Add(delta) => delta.is_finite() && delta > 0.,
            Advance::Multiply(delta) => delta.is_finite() && delta > 1.,
        };
        if !start.is_finite() || !end.is_finite() || !valid_advance {
            return Err(unsupported(format!(
                "{analysis} frequency sweep cannot advance finitely"
            )));
        }
        Ok(Self {
            next: (start <= end).then_some(start),
            end,
            advance,
            emitted: 0,
            analysis,
        })
    }
}

impl Iterator for FrequencyGrid {
    type Item = SpiceResult<Real>;

    fn next(&mut self) -> Option<Self::Item> {
        let frequency = self.next.take()?;
        if self.emitted >= MAX_POINTS {
            return Some(Err(unsupported(format!(
                "{} frequency loop exceeded {MAX_POINTS} points",
                self.analysis
            ))));
        }
        let next = match self.advance {
            Advance::Single => None,
            Advance::Add(delta) => Some(frequency + delta),
            Advance::Multiply(delta) => Some(frequency * delta),
        };
        if let Some(next) = next {
            if next <= frequency {
                return Some(Err(unsupported(format!(
                    "{} frequency step does not advance at {frequency:e}",
                    self.analysis
                ))));
            }
            self.next = (next <= self.end).then_some(next);
        }
        self.emitted += 1;
        Some(Ok(frequency))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_matches_the_original_incremental_iterator_bitwise() {
        let cases = [
            (0.1, 0.3, 1e-15, Advance::Add(0.1)),
            (1., 1e4, 1e-12, Advance::Multiply(10f64.powf(0.1))),
            (1e20, 1e20, 0., Advance::Single),
            (42., 42., 0., Advance::Add(1.)),
            (3., 1., 0., Advance::Single),
        ];
        for (start, stop, tolerance, advance) in cases {
            let from_reference = FrequencyGrid::new(start, stop, tolerance, advance, ".ac")
                .unwrap()
                .collect::<SpiceResult<Vec<_>>>()
                .unwrap();
            let plan = FrequencyPlan::new(start, stop, tolerance, advance, ".ac").unwrap();
            let replay = || plan.values().map(Real::to_bits).collect::<Vec<_>>();
            let want: Vec<_> = from_reference.iter().map(|v| v.to_bits()).collect();
            assert_eq!(replay(), want);
            assert_eq!(replay(), want);
            assert_eq!(plan.len(), want.len());
        }
    }

    #[test]
    fn invalid_future_steps_are_rejected_during_plan_construction() {
        // There is a representable endpoint, but its nominal sub-ULP
        // increment cannot move the initial value.
        assert!(
            FrequencyPlan::new(1e20, 1.0000000000000002e20, 0., Advance::Add(0.1), ".ac").is_err()
        );
        // A finite multiplier can legitimately jump past the stop in one
        // step; it is not an error under the reference grid policy.
        let skipped =
            FrequencyPlan::new(1., 1e308, 0., Advance::Multiply(Real::MAX), ".ac").unwrap();
        assert_eq!(skipped.values().collect::<Vec<_>>(), [1.]);
        assert!(
            FrequencyPlan::new(1., Real::MAX, Real::MAX, Advance::Multiply(2.), ".ac").is_err()
        );
        assert!(FrequencyPlan::new(0., 1e9, 0., Advance::Add(1.), ".ac").is_err());
    }

    #[test]
    fn grid_accumulates_the_same_rounded_frequencies_as_c() {
        let start = 0.1;
        let delta = 0.1;
        let grid = FrequencyGrid::new(start, 0.3, 1e-15, Advance::Add(delta), ".ac")
            .unwrap()
            .collect::<SpiceResult<Vec<_>>>()
            .unwrap();
        assert_eq!(grid, [start, start + delta, (start + delta) + delta]);
    }

    #[test]
    fn single_point_sweep_does_not_attempt_to_step() {
        let grid = FrequencyGrid::new(42., 42., 0., Advance::Single, ".noise")
            .unwrap()
            .collect::<SpiceResult<Vec<_>>>()
            .unwrap();
        assert_eq!(grid, [42.]);
    }

    #[test]
    fn stalled_step_fails_before_repeating_the_frequency() {
        let mut grid = FrequencyGrid::new(1e20, 2e20, 0., Advance::Add(1.), ".noise").unwrap();
        let error = grid.next().unwrap().unwrap_err();
        assert!(error.to_string().contains("does not advance"));
        assert!(grid.next().is_none());
    }

    #[test]
    fn work_bound_and_nonfinite_endpoint_are_checked() {
        let mut grid = FrequencyGrid::new(0., 1e9, 0., Advance::Add(1.), ".ac").unwrap();
        assert_eq!(grid.by_ref().take(MAX_POINTS).count(), MAX_POINTS);
        assert!(grid.next().unwrap().is_err());
        assert!(grid.next().is_none());
        assert!(
            FrequencyGrid::new(1., Real::MAX, Real::MAX, Advance::Multiply(2.), ".ac").is_err()
        );
    }
}
