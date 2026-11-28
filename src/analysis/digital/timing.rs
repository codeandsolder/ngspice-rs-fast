//! Causal timing uncertainty, separate from an unknown logical state.
//!
//! Delay variables are nonnegative, independently bounded except where
//! an explicit origin is shared. Fanout clones a window's expression;
//! reconvergence can cancel that common uncertainty before comparing
//! setup/hold intervals or the order of conflicting transitions.

use crate::primitives::{ExactSum, Real, SpiceError, SpiceResult};
use std::collections::BTreeMap;
mod conditional;
mod feasibility;
pub(crate) use conditional::proven_order;
pub use conditional::{
    Conditional, GuardStatus, Guarded, OrderCase, OrderPartition, TimingGuard, TimingRelation,
};
pub use feasibility::ProjectedInterval;
use std::sync::Arc;
mod constraints;
pub use constraints::{ApertureResult, Certainty, ConstraintResult};
mod delivery;
mod waveform;
pub(super) use delivery::phase_between;
pub use delivery::{Arrival, TimedInput, TimedOutput};
pub use waveform::{DriveTrace, GuardedWaveform, ProjectedDriveChange, WaveformProjection};

/// Identity of one uncertain delay variable. Reusing an identity expresses
/// the same normalized uncertainty, not another independent random draw.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Origin {
    generated: bool,
    identity: u64,
}

impl Origin {
    /// Explicit shared identity assigned by a timing-event producer.
    #[must_use = "the identity establishes delay correlation"]
    pub const fn shared(identity: u64) -> Self {
        Self {
            generated: false,
            identity,
        }
    }

    /// Allocate an independent delay variable, disjoint from explicitly
    /// assigned shared origins.
    ///
    /// # Errors
    /// Identity space exhausted.
    pub fn independent() -> SpiceResult<Self> {
        Ok(Self {
            generated: true,
            identity: super::external_identity()?,
        })
    }
}

/// Finite inclusive lower and upper time bounds, in seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    minimum: Real,
    maximum: Real,
}

impl Bounds {
    /// Construct finite ordered bounds. Relative times may be negative.
    ///
    /// # Errors
    /// Nonfinite or reversed bounds.
    pub fn new(minimum: Real, maximum: Real) -> SpiceResult<Self> {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err(SpiceError::circuit("invalid timing bounds"));
        }
        Ok(Self { minimum, maximum })
    }

    /// Earliest possible time, or least relative separation.
    #[must_use = "the bound is the earliest possible time"]
    pub const fn minimum(self) -> Real {
        self.minimum
    }

    /// Latest possible time, or greatest relative separation.
    #[must_use = "the bound is the latest possible time"]
    pub const fn maximum(self) -> Real {
        self.maximum
    }

    /// Whether this window has no timing uncertainty.
    #[must_use = "the result distinguishes exact and uncertain events"]
    pub fn is_exact(self) -> bool {
        self.minimum == self.maximum
    }
}

/// Nonnegative propagation interval, not a logic value or a slew rate.
#[derive(Clone, Copy, Debug)]
pub struct DelayWindow(Bounds);

impl DelayWindow {
    /// Construct a finite nonnegative delay interval.
    ///
    /// # Errors
    /// Invalid or negative delay.
    pub fn new(minimum: Real, maximum: Real) -> SpiceResult<Self> {
        if minimum < 0. {
            return Err(SpiceError::circuit("negative timing delay"));
        }
        Ok(Self(Bounds::new(minimum, maximum)?))
    }

    /// The inclusive propagation interval.
    #[must_use = "the bounds describe the propagation delay"]
    pub const fn bounds(self) -> Bounds {
        self.0
    }
}

/// An immutable causal arrival-time expression. Copying it at fanout
/// preserves the identity of every upstream uncertain delay.
#[derive(Clone)]
pub struct TimingWindow(Arc<Node>);

#[derive(Debug)]
struct Node {
    depth: usize,
    bounds: Bounds,
    expression: Expression,
}

impl std::fmt::Debug for TimingWindow {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TimingWindow")
            .field("bounds", &self.0.bounds)
            .field("depth", &self.0.depth)
            .finish()
    }
}

// Release uniquely owned ancestry iteratively. Shared nodes remain alive
// until their last reference is removed; the final owner drains them too.
impl Drop for Node {
    fn drop(&mut self) {
        fn enqueue(expression: Expression, pending: &mut Vec<TimingWindow>) {
            match expression {
                Expression::Exact(_) => {}
                Expression::Delay { parent, .. } => pending.push(parent),
                Expression::Extremum { inputs, .. } => pending.extend(inputs),
            }
        }
        let mut pending = Vec::new();
        enqueue(
            std::mem::replace(&mut self.expression, Expression::Exact(0.)),
            &mut pending,
        );
        while let Some(window) = pending.pop() {
            if let Some(mut node) = Arc::into_inner(window.0) {
                enqueue(
                    std::mem::replace(&mut node.expression, Expression::Exact(0.)),
                    &mut pending,
                );
            }
        }
    }
}

#[derive(Debug)]
enum Expression {
    Exact(Real),
    Delay {
        parent: TimingWindow,
        delay: DelayWindow,
        origin: Origin,
    },
    Extremum {
        kind: Extremum,
        inputs: Vec<TimingWindow>,
    },
}

#[derive(Clone, Copy, Debug)]
enum Extremum {
    Earliest,
    Latest,
}

impl Extremum {
    const fn inverted(self) -> Self {
        match self {
            Self::Earliest => Self::Latest,
            Self::Latest => Self::Earliest,
        }
    }

    fn combine(self, a: Bounds, b: Bounds) -> Bounds {
        let op = match self {
            Self::Earliest => Real::min,
            Self::Latest => Real::max,
        };
        Bounds {
            minimum: op(a.minimum, b.minimum),
            maximum: op(a.maximum, b.maximum),
        }
    }
}

impl TimingWindow {
    /// An exact nonnegative physical event time.
    ///
    /// # Errors
    /// Negative or nonfinite time.
    pub fn exact(time: Real) -> SpiceResult<Self> {
        if time < 0. {
            return Err(SpiceError::circuit("negative timing event"));
        }
        Ok(Self(Arc::new(Node {
            depth: 0,
            bounds: Bounds::new(time, time)?,
            expression: Expression::Exact(time),
        })))
    }

    /// Propagate this arrival through another causal delay. Distinct
    /// origins are independent. Shared origins are correlated even if
    /// their minimum and maximum delays differ.
    ///
    /// # Errors
    /// Nonrepresentable physical arrival time.
    pub fn after(&self, delay: DelayWindow, origin: Origin) -> SpiceResult<Self> {
        let d = delay.bounds();
        if d.minimum == 0. && d.maximum == 0. {
            return Ok(self.clone());
        }
        let bounds = Bounds::new(
            self.0.bounds.minimum + d.minimum,
            self.0.bounds.maximum + d.maximum,
        )?;
        if (d.minimum > 0. && bounds.minimum == self.0.bounds.minimum)
            || (d.maximum > 0. && bounds.maximum == self.0.bounds.maximum)
            || ((!d.is_exact() || !self.0.bounds.is_exact()) && bounds.is_exact())
        {
            return Err(SpiceError::circuit(
                "timing ambiguity below physical time resolution",
            ));
        }
        if bounds.is_exact() {
            return Self::exact(bounds.minimum);
        }
        let bounds = {
            let add = |a: Real, b: Real, upper: bool| -> SpiceResult<Real> {
                let mut sum = ExactSum::new();
                sum.add_product(a, 1.);
                sum.add_product(b, 1.);
                round_outward(sum, upper)
            };
            Bounds::new(
                add(self.0.bounds.minimum, d.minimum, false)?,
                add(self.0.bounds.maximum, d.maximum, true)?,
            )?
        };
        Ok(Self(Arc::new(Node {
            depth: self
                .0
                .depth
                .checked_add(1)
                .ok_or_else(|| SpiceError::circuit("timing expression depth overflow"))?,
            bounds,
            expression: Expression::Delay {
                parent: self.clone(),
                delay,
                origin,
            },
        })))
    }

    /// Arrival at the earliest of the supplied causal inputs, as for a
    /// rising OR or falling AND. Preserve all origins at reconvergence.
    ///
    /// # Errors
    /// Empty input set.
    pub fn earliest(inputs: Vec<Self>) -> SpiceResult<Self> {
        Self::extremum(Extremum::Earliest, inputs)
    }

    /// Arrival at the latest of the supplied causal inputs, as for a
    /// rising AND or falling OR. Preserve all origins at reconvergence.
    ///
    /// # Errors
    /// Empty input set.
    pub fn latest(inputs: Vec<Self>) -> SpiceResult<Self> {
        Self::extremum(Extremum::Latest, inputs)
    }

    fn extremum(kind: Extremum, mut inputs: Vec<Self>) -> SpiceResult<Self> {
        let Some(first) = inputs.first() else {
            return Err(SpiceError::circuit("empty timing extremum"));
        };
        if inputs.len() == 1 {
            return Ok(inputs.remove(0));
        }
        let bounds = inputs.iter().skip(1).fold(first.bounds(), |old, input| {
            kind.combine(old, input.bounds())
        });
        Ok(Self(Arc::new(Node {
            depth: inputs
                .iter()
                .map(|input| input.0.depth)
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| SpiceError::circuit("timing expression depth overflow"))?,
            bounds,
            expression: Expression::Extremum { kind, inputs },
        })))
    }

    /// Whether both handles identify the same causal arrival expression.
    #[must_use = "identity distinguishes completion of a previously announced transition"]
    pub fn same_expression(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Absolute arrival interval. Because all expressions are monotone in
    /// every delay variable, the two extrema are jointly attainable.
    #[must_use = "the bounds describe absolute arrival uncertainty"]
    pub fn bounds(&self) -> Bounds {
        self.0.bounds
    }

    /// Enclose the relative time `self - reference`, factoring common
    /// causal origins before taking a difference. Affine paths give exact
    /// relative bounds; min/max expressions yield sound enclosures, not
    /// necessarily tight ones. An explicit work budget bounds arbitrary
    /// reconvergent expression expansion without silently dropping terms.
    ///
    /// # Errors
    /// Work budget exhausted or nonfinite arithmetic.
    pub fn relative_to(&self, reference: &Self, mut budget: usize) -> SpiceResult<Bounds> {
        relative(self, reference, &mut budget)
    }
}

#[derive(Clone, Copy)]
struct Shift {
    delay: DelayWindow,
    origin: Origin,
}

/// A disposable comparison job. Offsets only contain delays *after* the
/// current expression; equal subexpressions cancel without traversing
/// their ancestors. No recursion through an adversarial circuit depth.
struct Compare<'a> {
    left: &'a TimingWindow,
    right: &'a TimingWindow,
    left_shifts: Vec<Shift>,
    right_shifts: Vec<Shift>,
}

enum Work<'a> {
    Compare(Compare<'a>),
    Combine { kind: Extremum, count: usize },
}

fn charge(budget: &mut usize, work: usize) -> SpiceResult<()> {
    *budget = budget
        .checked_sub(work)
        .ok_or_else(|| SpiceError::circuit("timing uncertainty comparison work limit exceeded"))?;
    Ok(())
}

// The exact accumulator contains sums of binary64 numbers, so any
// nonzero residual is at least one smallest subnormal. Round outward
// only when the nearest representable result lies inside the interval.
fn round_outward(mut sum: ExactSum, upper: bool) -> SpiceResult<Real> {
    let rounded = sum.rounded_sum();
    if !rounded.is_finite() {
        return Err(SpiceError::circuit(
            "timing comparison outside representable range",
        ));
    }
    sum.add_product(rounded, -1.);
    let residual = sum.rounded_sum();
    Ok(if upper && residual > 0. {
        rounded.next_up()
    } else if !upper && residual < 0. {
        rounded.next_down()
    } else {
        rounded
    })
}

fn affine_difference(
    left: Real,
    right: Real,
    lhs: &[Shift],
    rhs: &[Shift],
    budget: &mut usize,
) -> SpiceResult<Bounds> {
    charge(budget, lhs.len().saturating_add(rhs.len()))?;
    // Each origin is one normalized variable in [0,1]. Subtract the
    // coefficients before bounding its contribution, including cases
    // where the same origin has a different range on each path.
    let mut origins = BTreeMap::<Origin, Vec<(Bounds, Real)>>::new();
    let mut lower = ExactSum::new();
    lower.add_product(left, 1.);
    lower.add_product(right, -1.);
    for (shifts, sign) in [(lhs, 1.), (rhs, -1.)] {
        for shift in shifts {
            let bounds = shift.delay.bounds();
            if bounds.is_exact() {
                lower.add_product(bounds.minimum, sign);
            } else {
                origins
                    .entry(shift.origin)
                    .or_default()
                    .push((bounds, sign));
            }
        }
    }
    let mut upper = lower.clone();
    for terms in origins.values() {
        let mut width = ExactSum::new();
        for &(bounds, sign) in terms {
            width.add_product(bounds.maximum, sign);
            width.add_product(bounds.minimum, -sign);
        }
        let increasing = width.rounded_sum() >= 0.;
        for &(bounds, sign) in terms {
            let (lo, hi) = if increasing {
                (bounds.minimum, bounds.maximum)
            } else {
                (bounds.maximum, bounds.minimum)
            };
            lower.add_product(lo, sign);
            upper.add_product(hi, sign);
        }
    }
    Bounds::new(round_outward(lower, false)?, round_outward(upper, true)?)
}

fn relative(left: &TimingWindow, right: &TimingWindow, budget: &mut usize) -> SpiceResult<Bounds> {
    let mut work = vec![Work::Compare(Compare {
        left,
        right,
        left_shifts: Vec::new(),
        right_shifts: Vec::new(),
    })];
    let mut results = Vec::new();
    while let Some(task) = work.pop() {
        charge(budget, 1)?;
        match task {
            Work::Combine { kind, count } => {
                let start = results
                    .len()
                    .checked_sub(count)
                    .ok_or_else(|| SpiceError::circuit("invalid timing expression comparison"))?;
                let mut values = results.drain(start..);
                let first = values
                    .next()
                    .ok_or_else(|| SpiceError::circuit("empty timing comparison"))?;
                let value = values.fold(first, |a, b| kind.combine(a, b));
                results.push(value);
            }
            Work::Compare(mut job) => loop {
                charge(budget, 1)?;
                if Arc::ptr_eq(&job.left.0, &job.right.0) {
                    results.push(affine_difference(
                        0.,
                        0.,
                        &job.left_shifts,
                        &job.right_shifts,
                        budget,
                    )?);
                    break;
                }
                // Align chain depths before peeling delays so a shared
                // ancestor cancels even with a large absolute timestamp.
                if job.left.0.depth >= job.right.0.depth
                    && let Expression::Delay {
                        parent,
                        delay,
                        origin,
                    } = &job.left.0.expression
                {
                    job.left_shifts.push(Shift {
                        delay: *delay,
                        origin: *origin,
                    });
                    job.left = parent;
                    continue;
                }
                match (&job.left.0.expression, &job.right.0.expression) {
                    (
                        Expression::Delay {
                            parent,
                            delay,
                            origin,
                        },
                        Expression::Exact(_) | Expression::Extremum { .. },
                    ) => {
                        job.left_shifts.push(Shift {
                            delay: *delay,
                            origin: *origin,
                        });
                        job.left = parent;
                    }
                    (
                        _,
                        Expression::Delay {
                            parent,
                            delay,
                            origin,
                        },
                    ) => {
                        job.right_shifts.push(Shift {
                            delay: *delay,
                            origin: *origin,
                        });
                        job.right = parent;
                    }
                    (Expression::Exact(a), Expression::Exact(b)) => {
                        results.push(affine_difference(
                            *a,
                            *b,
                            &job.left_shifts,
                            &job.right_shifts,
                            budget,
                        )?);
                        break;
                    }
                    (Expression::Extremum { kind, inputs }, _) => {
                        charge(
                            budget,
                            inputs.len().saturating_mul(
                                1_usize
                                    .saturating_add(job.left_shifts.len())
                                    .saturating_add(job.right_shifts.len()),
                            ),
                        )?;
                        work.push(Work::Combine {
                            kind: *kind,
                            count: inputs.len(),
                        });
                        for input in inputs {
                            work.push(Work::Compare(Compare {
                                left: input,
                                right: job.right,
                                left_shifts: job.left_shifts.clone(),
                                right_shifts: job.right_shifts.clone(),
                            }));
                        }
                        break;
                    }
                    (_, Expression::Extremum { kind, inputs }) => {
                        charge(
                            budget,
                            inputs.len().saturating_mul(
                                1_usize
                                    .saturating_add(job.left_shifts.len())
                                    .saturating_add(job.right_shifts.len()),
                            ),
                        )?;
                        work.push(Work::Combine {
                            kind: kind.inverted(),
                            count: inputs.len(),
                        });
                        for input in inputs {
                            work.push(Work::Compare(Compare {
                                left: job.left,
                                right: input,
                                left_shifts: job.left_shifts.clone(),
                                right_shifts: job.right_shifts.clone(),
                            }));
                        }
                        break;
                    }
                }
            },
        }
    }
    match results.as_slice() {
        [result] => Ok(*result),
        _ => Err(SpiceError::circuit("invalid timing comparison result")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconvergent_paths_cancel_common_timing_origin() -> SpiceResult<()> {
        let root =
            TimingWindow::exact(0.)?.after(DelayWindow::new(15., 30.)?, Origin::shared(0))?;
        let data = root.after(DelayWindow::new(10., 30.)?, Origin::shared(1))?;
        let clock = root.after(DelayWindow::new(40., 60.)?, Origin::shared(2))?;
        assert_eq!(data.bounds(), Bounds::new(25., 60.)?);
        assert_eq!(clock.bounds(), Bounds::new(55., 90.)?);
        assert_eq!(clock.relative_to(&data, 100)?, Bounds::new(10., 50.)?);
        assert_eq!(data.relative_to(&clock, 100)?, Bounds::new(-50., -10.)?);
        let unrelated =
            TimingWindow::exact(0.)?.after(DelayWindow::new(55., 90.)?, Origin::shared(3))?;
        assert_eq!(unrelated.relative_to(&data, 100)?, Bounds::new(-5., 65.)?);
        Ok(())
    }

    #[test]
    fn shared_origin_with_distinct_ranges_and_nested_extrema() -> SpiceResult<()> {
        let a = TimingWindow::exact(2.)?.after(DelayWindow::new(1., 4.)?, Origin::shared(7))?;
        let b = TimingWindow::exact(0.)?.after(DelayWindow::new(4., 8.)?, Origin::shared(7))?;
        assert_eq!(b.relative_to(&a, 100)?, Bounds::new(1., 2.)?);
        let first = TimingWindow::earliest(vec![a.clone(), b.clone()])?;
        let last = TimingWindow::latest(vec![a.clone(), b.clone()])?;
        assert_eq!(first.bounds(), a.bounds());
        assert_eq!(last.bounds(), b.bounds());
        assert_eq!(first.relative_to(&a, 100)?, Bounds::new(0., 0.)?);
        assert_eq!(last.relative_to(&b, 100)?, Bounds::new(0., 0.)?);
        let shifted = last.after(DelayWindow::new(1., 3.)?, Origin::shared(8))?;
        assert_eq!(shifted.relative_to(&last, 100)?, Bounds::new(1., 3.)?);
        assert!(shifted.relative_to(&first, 1).is_err());
        Ok(())
    }

    #[test]
    fn deep_shared_ancestry_cancels_with_constant_comparison_work() -> SpiceResult<()> {
        let mut root = TimingWindow::exact(1e12)?;
        for origin in 0..10_000 {
            root = root.after(DelayWindow::new(1., 2.)?, Origin::shared(origin))?;
        }
        let a = root.after(DelayWindow::new(1., 2.)?, Origin::shared(10_000))?;
        let b = root
            .after(DelayWindow::new(2., 4.)?, Origin::shared(10_001))?
            .after(DelayWindow::new(1., 1.)?, Origin::shared(10_002))?;
        assert_eq!(b.relative_to(&a, 20)?, Bounds::new(1., 4.)?);
        drop(root);
        drop(a);
        drop(b);
        Ok(())
    }

    #[test]
    fn extremum_relative_enclosure_contains_sampled_correlated_arrivals() -> SpiceResult<()> {
        let root = TimingWindow::exact(0.)?.after(DelayWindow::new(2., 7.)?, Origin::shared(1))?;
        let a = root.after(DelayWindow::new(1., 8.)?, Origin::shared(2))?;
        let b = root.after(DelayWindow::new(4., 5.)?, Origin::shared(3))?;
        let c = TimingWindow::exact(3.)?.after(DelayWindow::new(2., 7.)?, Origin::shared(2))?;
        let joined = TimingWindow::latest(vec![TimingWindow::earliest(vec![a, b])?, c.clone()])?;
        let bounds = joined.relative_to(&c, 1000)?;
        for x in [0., 0.25, 0.5, 0.75, 1.] {
            for y in [0., 0.25, 0.5, 0.75, 1.] {
                for z in [0., 0.25, 0.5, 0.75, 1.] {
                    let root: Real = 2. + 5. * x;
                    let a = root + 1. + 7. * y;
                    let b = root + 4. + z;
                    let c = 5. + 5. * y;
                    let actual = a.min(b).max(c) - c;
                    assert!(bounds.minimum() <= actual && actual <= bounds.maximum());
                }
            }
        }
        Ok(())
    }

    #[test]
    fn separately_constructed_large_arrivals_cancel_without_losing_small_offsets() -> SpiceResult<()>
    {
        let a = TimingWindow::exact(1e16)?.after(DelayWindow::new(4., 8.)?, Origin::shared(1))?;
        let b = TimingWindow::exact(1e16)?.after(DelayWindow::new(5., 9.)?, Origin::shared(1))?;
        assert_eq!(b.relative_to(&a, 100)?, Bounds::new(1., 1.)?);
        Ok(())
    }

    #[test]
    fn invalid_or_subresolution_intervals_fail_explicitly() -> SpiceResult<()> {
        assert!(DelayWindow::new(-1., 1.).is_err());
        assert!(DelayWindow::new(2., 1.).is_err());
        assert!(TimingWindow::earliest(Vec::new()).is_err());
        let root = TimingWindow::exact(1.)?;
        assert!(
            root.after(DelayWindow::new(0., Real::EPSILON / 4.)?, Origin::shared(1))
                .is_err()
        );
        Ok(())
    }
}
