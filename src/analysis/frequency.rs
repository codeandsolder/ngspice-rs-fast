//! Bounded, incremental frequency traversal for the spectral analyses.
//!
//! Each caller owns the reference-specific spacing formula and singleton
//! policy. The shared iterator keeps the actual repeated additions or
//! multiplications, endpoint tolerance and work budget in one place.

use crate::analysis::linear::unsupported;
use crate::primitives::{Real, SpiceResult};

pub(super) const MAX_POINTS: usize = 100_000;

#[derive(Clone, Copy)]
pub(super) enum Advance {
    Single,
    Add(Real),
    Multiply(Real),
}

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
