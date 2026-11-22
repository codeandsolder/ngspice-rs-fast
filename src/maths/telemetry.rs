//! Read-only sparse solver work and storage measurements.
use crate::maths::sparse13::Status;
use crate::primitives::SpiceResult;

/// Numeric factor slots and their structural origins, measured from a valid LU.
/// Counts include stored zero values, not only nonzero numerical coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FactorStorage {
    /// Order requested by the external caller.
    pub external_order: usize,
    /// Number of rows actually translated by SPARSE.
    pub translated_order: usize,
    /// Handles created as original entries (including structural zeroes).
    pub original_entries: usize,
    /// All allocated element slots in the valid factor, including fill-ins.
    pub factor_entries: usize,
    /// Logical size of the numeric factor buffer, excluding spare capacity and metadata.
    pub numeric_bytes: usize,
}

impl FactorStorage {
    /// Slots initially created by elimination rather than assembly.
    #[must_use]
    pub const fn fill_entries(self) -> usize {
        self.factor_entries - self.original_entries
    }
}

/// Work performed during factorization. For a standalone snapshot this refers
/// to the single producing operation, including any immediate retry.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FactorWork {
    /// Calls to threshold order-and-factor, including retained pivot traversals.
    pub order_attempts: u64,
    /// Actual starts of Markowitz pivot search (not retained-order traversals).
    pub pivot_searches: u64,
    /// Numerical refactors using a complete symbolic traversal plan.
    pub fixed_refactors: u64,
    /// Completed valid numeric factors, including small-pivot warnings.
    pub successful_factors: u64,
    /// Fixed-order zero diagonal failures.
    pub zero_pivots: u64,
    /// Singular factorization outcomes.
    pub singular_failures: u64,
    /// Non-finite or unrepresentable factorization outcomes.
    pub nonfinite_failures: u64,
    /// Factor storage or workspace capacity failures.
    pub resource_failures: u64,
    /// Attempts rejected because coefficients or a traversal plan are invalid.
    pub invalid_failures: u64,
    /// Rejected inputs or failed frame construction before factorization.
    pub assembly_rejections: u64,
    /// Immediate reload-and-order retries after a zero diagonal.
    pub zero_pivot_retries: u64,
}

/// Only one compact outcome is stored in a numerical frame. The public
/// counters are accumulated at operation boundaries, not copied into every
/// retained frame.
#[derive(Debug, Clone, Copy)]
pub(super) enum AttemptPath {
    Order { searched: bool },
    Refactor,
    Rejected,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct FactorAttempt {
    pub(super) path: AttemptPath,
    pub(super) status: Status,
}

impl FactorWork {
    pub(super) const fn record(&mut self, attempt: FactorAttempt) {
        let start = match attempt.path {
            AttemptPath::Order { searched } => {
                if searched {
                    self.pivot_searches = self.pivot_searches.saturating_add(1);
                }
                Some(&mut self.order_attempts)
            }
            AttemptPath::Refactor => Some(&mut self.fixed_refactors),
            AttemptPath::Rejected => None,
        };
        if let Some(counter) = start {
            *counter = counter.saturating_add(1);
        }
        let counter = match attempt.status {
            Status::Okay | Status::SmallPivot => &mut self.successful_factors,
            Status::ZeroDiag => &mut self.zero_pivots,
            Status::Singular => &mut self.singular_failures,
            Status::NonFinite => &mut self.nonfinite_failures,
            Status::Resource => &mut self.resource_failures,
            Status::InvalidPlan | Status::NeedsReload => &mut self.invalid_failures,
        };
        *counter = counter.saturating_add(1);
    }

    pub(super) fn assembly<T>(&mut self, result: SpiceResult<T>) -> SpiceResult<T> {
        result.inspect_err(|_| {
            self.assembly_rejections = self.assembly_rejections.saturating_add(1);
        })
    }
}

impl std::ops::AddAssign for FactorWork {
    fn add_assign(&mut self, other: Self) {
        let Self {
            order_attempts,
            pivot_searches,
            fixed_refactors,
            successful_factors,
            zero_pivots,
            singular_failures,
            nonfinite_failures,
            resource_failures,
            invalid_failures,
            assembly_rejections,
            zero_pivot_retries,
        } = other;
        self.order_attempts = self.order_attempts.saturating_add(order_attempts);
        self.pivot_searches = self.pivot_searches.saturating_add(pivot_searches);
        self.fixed_refactors = self.fixed_refactors.saturating_add(fixed_refactors);
        self.successful_factors = self.successful_factors.saturating_add(successful_factors);
        self.zero_pivots = self.zero_pivots.saturating_add(zero_pivots);
        self.singular_failures = self.singular_failures.saturating_add(singular_failures);
        self.nonfinite_failures = self.nonfinite_failures.saturating_add(nonfinite_failures);
        self.resource_failures = self.resource_failures.saturating_add(resource_failures);
        self.invalid_failures = self.invalid_failures.saturating_add(invalid_failures);
        self.assembly_rejections = self.assembly_rejections.saturating_add(assembly_rejections);
        self.zero_pivot_retries = self.zero_pivot_retries.saturating_add(zero_pivot_retries);
    }
}

/// Cumulative work of one shared pattern's cache transactions. Independent
/// factor snapshots and their solves do not mutate this record.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PatternWork {
    /// Factorization work, including failed attempts.
    pub factor: FactorWork,
    /// Bit-identical input matches that use retained numeric factors.
    pub exact_input_hits: u64,
    /// Successful real RHS solves made through the cache transaction.
    pub rhs_solves: u64,
    /// Failed RHS solve requests made through the cache transaction.
    pub rhs_failures: u64,
    /// Actual transitions from real to complex storage.
    pub real_to_complex: u64,
    /// Actual transitions from complex to real storage.
    pub complex_to_real: u64,
}

impl PatternWork {
    pub(super) const fn record_rhs<T>(&mut self, result: &SpiceResult<T>) {
        let counter = if result.is_ok() {
            &mut self.rhs_solves
        } else {
            &mut self.rhs_failures
        };
        *counter = counter.saturating_add(1);
    }
}

/// Consistent snapshot of a shared pattern's work and current valid factors.
/// Cloning, extending or recovering a poisoned pattern cache starts a new record.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PatternTelemetry {
    /// Work since construction or cache reset.
    pub work: PatternWork,
    /// Current valid storage, absent after an unsuccessful factorization or before factoring.
    pub storage: Option<FactorStorage>,
}
