//! Compensated reductions for cancellation-sensitive analysis quantities.

use crate::primitives::Real;

/// Neumaier accumulation: retain the rounding error of each addition even
/// when the new term exceeds the running total in magnitude.
#[derive(Clone, Copy, Default)]
pub(super) struct CompensatedSum {
    sum: Real,
    correction: Real,
}

impl CompensatedSum {
    pub(super) const fn with_initial(value: Real) -> Self {
        Self {
            sum: value,
            correction: 0.,
        }
    }

    pub(super) const fn negate(&mut self) {
        self.sum = -self.sum;
        self.correction = -self.correction;
    }

    pub(super) const fn add(&mut self, value: Real) {
        let next = self.sum + value;
        if next.is_finite() {
            self.correction += if self.sum.abs() >= value.abs() {
                (self.sum - next) + value
            } else {
                (value - next) + self.sum
            };
        } else {
            // Preserve ordinary IEEE propagation for sensitivity's block-local
            // nonfinite handling; compensation cannot recover an overflow.
            self.correction = 0.;
        }
        self.sum = next;
    }

    pub(super) const fn total(self) -> Real {
        if self.correction == 0. {
            self.sum
        } else {
            self.sum + self.correction
        }
    }
}

/// One-pass norm with a running magnitude scale, so neither large nor tiny
/// terms need to be squared in the original units.
#[derive(Default)]
pub(super) struct ScaledSquares {
    scale: Real,
    squares: CompensatedSum,
}

impl ScaledSquares {
    pub(super) fn add(&mut self, magnitude: Real) {
        if magnitude == 0. {
            return;
        }
        if magnitude > self.scale {
            let ratio = self.scale / magnitude;
            let previous = self.squares.total() * ratio * ratio;
            self.scale = magnitude;
            self.squares = CompensatedSum::with_initial(previous);
            self.squares.add(1.);
        } else {
            let ratio = magnitude / self.scale;
            self.squares.add(ratio * ratio);
        }
    }

    pub(super) fn norm(&self) -> Real {
        self.scale * self.squares.total().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::CompensatedSum;

    #[test]
    fn retains_small_terms_on_both_sides_of_large_cancellation() {
        for values in [[1e16, 1., -1e16], [1., 1e16, -1e16]] {
            let mut sum = CompensatedSum::default();
            for value in values {
                sum.add(value);
            }
            assert_eq!(sum.total(), 1.);
        }
    }
}
