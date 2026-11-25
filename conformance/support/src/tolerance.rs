//! Shared numerical budgets for reference comparisons, not solver convergence.

/// Relative and absolute error budget for a scalar or a complex magnitude.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// Fraction of the reference magnitude.
    pub relative: f64,
    /// Absolute floor in the compared quantity’s units.
    pub absolute: f64,
}

impl Tolerance {
    /// Absolute error allowed for the given reference magnitude.
    #[must_use = "the returned bound determines whether the numerical error is acceptable"]
    pub const fn bound(self, reference: f64) -> f64 {
        self.relative.mul_add(reference.abs(), self.absolute)
    }
}

/// Linear derivatives have independent closed-form checks and do not amplify
/// nonlinear operating-point error. Keep the tight historical comparison.
pub const SENSITIVITY: Tolerance = Tolerance {
    relative: 1e-8,
    absolute: 1e-12,
};

/// Engineering-scale agreement with C's finite-difference nonlinear sensitivity.
///
/// The 0.1% relative budget follows SPICE's default accuracy scale, not a
/// claim about the accuracy of the physical derivative. Independent analytic
/// and convergence checks retain their own numerical budgets.
pub const SENSITIVITY_NONLINEAR: Tolerance = Tolerance {
    relative: 1e-3,
    absolute: 1e-9,
};

/// Same relative budget, with the existing absolute cancellation floor for
/// the MESA level-2 reference's weak and unused parameters.
pub const SENSITIVITY_CANCELLED: Tolerance = Tolerance {
    absolute: 5e-8,
    ..SENSITIVITY_NONLINEAR
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonlinear_reference_budget_is_engineering_scale() {
        for reference in [-1e6_f64, -1e-3, 0., 1e-3, 1e6] {
            let bound = SENSITIVITY_NONLINEAR.bound(reference);
            assert!(bound >= reference.abs() * 1e-3);
            assert!(bound >= 1e-9);
            assert!(bound <= reference.abs().mul_add(1.001e-3, 1.001e-9));
            assert!(SENSITIVITY.bound(reference) < bound);
            assert!(SENSITIVITY_CANCELLED.bound(reference) >= bound);
        }
    }
}
