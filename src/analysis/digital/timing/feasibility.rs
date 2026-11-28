//! Exact feasibility of joint timing guards. Every finite binary64
//! constant is an integer multiple of 2^-1074. Affine inequalities
//! therefore have exact integer coefficients; only their variables are
//! real-valued normalized origins. Fourier-Motzkin elimination preserves
//! strict inequalities and uses positive integer scaling, never a
//! rounded quotient or an epsilon. Min/max selectors are disjunctions
//! over their children, shared by every use of the same expression.

use super::conditional::{Comparison, TimingRelation};
use super::{Expression, Extremum, Node, Origin, TimingWindow, charge};
use crate::primitives::{Real, SpiceError, SpiceResult};
use num_bigint::{BigInt, Sign};
use num_integer::Integer;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Variable {
    Origin(Origin),
    Observation,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
struct Affine {
    constant: BigInt,
    terms: BTreeMap<Variable, BigInt>,
}

fn units(value: Real) -> BigInt {
    let bits = value.to_bits();
    let biased = (bits >> 52) & 0x7ff;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    let significand = if biased == 0 {
        fraction
    } else {
        fraction | (1 << 52)
    };
    let magnitude = BigInt::from(significand) << biased.saturating_sub(1);
    if value.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

fn words(value: &BigInt) -> usize {
    usize::try_from(value.bits().div_ceil(64))
        .unwrap_or(usize::MAX)
        .max(1)
}

fn multiply(lhs: &BigInt, rhs: &BigInt, budget: &mut usize) -> SpiceResult<BigInt> {
    charge(budget, words(lhs).saturating_mul(words(rhs)))?;
    Ok(lhs * rhs)
}

impl Affine {
    fn exact(value: Real) -> Self {
        Self {
            constant: units(value),
            terms: BTreeMap::new(),
        }
    }

    fn delay(mut self, origin: Origin, minimum: Real, maximum: Real) -> Self {
        let minimum = units(minimum);
        let width = units(maximum) - &minimum;
        self.constant += minimum;
        if width.sign() != Sign::NoSign {
            *self.terms.entry(Variable::Origin(origin)).or_default() += width;
        }
        self
    }

    fn difference(lhs: &Self, rhs: &Self, budget: &mut usize) -> SpiceResult<Self> {
        charge(
            budget,
            lhs.terms
                .len()
                .saturating_add(rhs.terms.len())
                .saturating_add(1),
        )?;
        let mut result = lhs.clone();
        result.constant -= &rhs.constant;
        for (&origin, coefficient) in &rhs.terms {
            *result.terms.entry(origin).or_default() -= coefficient;
        }
        result
            .terms
            .retain(|_, coefficient| coefficient.sign() != Sign::NoSign);
        Ok(result)
    }

    fn normalized(mut self, budget: &mut usize) -> SpiceResult<Self> {
        // Primitive integer row, with its sign preserved. Common powers
        // of two (including the binary64 unit scale) disappear here.
        let mut divisor = BigInt::ZERO;
        for coefficient in std::iter::once(&self.constant).chain(self.terms.values()) {
            charge(budget, words(&divisor).saturating_mul(words(coefficient)))?;
            divisor = divisor.gcd(coefficient);
            if divisor == BigInt::from(1) {
                return Ok(self);
            }
        }
        if divisor.sign() != Sign::NoSign {
            charge(
                budget,
                self.terms
                    .len()
                    .saturating_add(1)
                    .saturating_mul(words(&divisor)),
            )?;
            self.constant /= &divisor;
            for coefficient in self.terms.values_mut() {
                *coefficient /= &divisor;
            }
        }
        Ok(self)
    }
}

/// expression <= 0, or expression < 0 when strict.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct Row {
    expression: Affine,
    strict: bool,
}

impl Row {
    fn comparison(
        lhs: &Affine,
        rhs: &Affine,
        relation: TimingRelation,
        rows: &mut Vec<Self>,
        budget: &mut usize,
    ) -> SpiceResult<()> {
        let mut push = |a: &Affine, b: &Affine, strict| -> SpiceResult<()> {
            let expression = Affine::difference(a, b, budget)?.normalized(budget)?;
            rows.push(Self { expression, strict });
            Ok(())
        };
        match relation {
            TimingRelation::Before => push(lhs, rhs, true),
            TimingRelation::NoLater => push(lhs, rhs, false),
            TimingRelation::After => push(rhs, lhs, true),
            TimingRelation::NoEarlier => push(rhs, lhs, false),
            TimingRelation::Coincident => {
                push(lhs, rhs, false)?;
                push(rhs, lhs, false)
            }
        }
    }

    fn constant_satisfied(&self) -> bool {
        match self.expression.constant.sign() {
            Sign::Minus => true,
            Sign::NoSign => !self.strict,
            Sign::Plus => false,
        }
    }

    fn eliminate(
        upper: &Self,
        lower: &Self,
        origin: Variable,
        budget: &mut usize,
    ) -> SpiceResult<Self> {
        let positive = &upper.expression.terms[&origin];
        let negative = -&lower.expression.terms[&origin];
        // positive * lower + (-negative coefficient) * upper removes
        // the origin. Both scales are positive and strictness survives
        // if either original inequality is strict.
        let constant = multiply(&upper.expression.constant, &negative, budget)?
            + multiply(&lower.expression.constant, positive, budget)?;
        let mut terms = BTreeMap::<Variable, BigInt>::new();
        for (row, scale) in [(upper, &negative), (lower, positive)] {
            for (&key, coefficient) in &row.expression.terms {
                if key != origin {
                    *terms.entry(key).or_default() += multiply(coefficient, scale, budget)?;
                }
            }
        }
        terms.retain(|_, coefficient| coefficient.sign() != Sign::NoSign);
        Ok(Self {
            expression: Affine { constant, terms }.normalized(budget)?,
            strict: upper.strict || lower.strict,
        })
    }
}

/// Drop satisfied constants and retain only the strictest identical row.
/// Returning None establishes an actual contradiction, not an exhausted
/// numerical tolerance.
fn simplify(rows: Vec<Row>, budget: &mut usize) -> SpiceResult<Option<Vec<Row>>> {
    charge(budget, rows.len())?;
    let mut unique = BTreeMap::<Affine, bool>::new();
    for row in rows {
        if row.expression.terms.is_empty() {
            if !row.constant_satisfied() {
                return Ok(None);
            }
        } else {
            *unique.entry(row.expression).or_default() |= row.strict;
        }
    }
    Ok(Some(
        unique
            .into_iter()
            .map(|(expression, strict)| Row { expression, strict })
            .collect(),
    ))
}

fn eliminate(
    rows: Vec<Row>,
    preserve: Option<Variable>,
    budget: &mut usize,
) -> SpiceResult<Option<Vec<Row>>> {
    let Some(mut rows) = simplify(rows, budget)? else {
        return Ok(None);
    };
    let mut origins: BTreeSet<_> = rows
        .iter()
        .flat_map(|row| row.expression.terms.keys().copied())
        .collect();
    if let Some(variable) = preserve {
        origins.insert(variable);
    }
    charge(budget, origins.len().saturating_mul(2))?;
    for origin in origins {
        // Delay origins are normalized in [0,1]; the observation
        // variable is a physical time in units of 2^-1074.
        rows.push(Row {
            expression: Affine {
                constant: BigInt::ZERO,
                terms: BTreeMap::from([(origin, BigInt::from(-1))]),
            },
            strict: false,
        });
        rows.push(Row {
            expression: Affine {
                constant: if origin == Variable::Observation {
                    -units(Real::MAX)
                } else {
                    BigInt::from(-1)
                },
                terms: BTreeMap::from([(origin, BigInt::from(1))]),
            },
            strict: false,
        });
    }
    loop {
        // Choose the smallest elimination cross product, not an
        // arbitrary chronological origin order. Isolated one-sided
        // variables disappear without pairing.
        let mut counts = BTreeMap::<Variable, (usize, usize)>::new();
        for row in &rows {
            charge(budget, row.expression.terms.len().saturating_add(1))?;
            for (&origin, coefficient) in &row.expression.terms {
                if Some(origin) == preserve {
                    continue;
                }
                let (positive, negative) = counts.entry(origin).or_default();
                if coefficient.sign() == Sign::Plus {
                    *positive = positive.saturating_add(1);
                } else {
                    *negative = negative.saturating_add(1);
                }
            }
        }
        let Some((origin, (positive, negative))) = counts
            .into_iter()
            .min_by_key(|(_, (p, n))| p.saturating_mul(*n))
        else {
            return Ok(Some(rows));
        };
        charge(budget, positive.saturating_mul(negative))?;
        let mut upper = Vec::new();
        let mut lower = Vec::new();
        let mut next = Vec::new();
        for row in rows {
            match row.expression.terms.get(&origin).map(BigInt::sign) {
                Some(Sign::Plus) => upper.push(row),
                Some(Sign::Minus) => lower.push(row),
                _ => next.push(row),
            }
        }
        for u in &upper {
            for l in &lower {
                next.push(Row::eliminate(u, l, origin, budget)?);
            }
        }
        let Some(reduced) = simplify(next, budget)? else {
            return Ok(None);
        };
        rows = reduced;
    }
}

fn feasible(rows: Vec<Row>, budget: &mut usize) -> SpiceResult<bool> {
    Ok(eliminate(rows, None, budget)?.is_some())
}

#[derive(Clone, Debug)]
struct RationalBoundary {
    numerator: BigInt,
    denominator: BigInt,
    inclusive: bool,
}

impl RationalBoundary {
    fn compare(&self, other: &Self, budget: &mut usize) -> SpiceResult<Ordering> {
        Ok(
            multiply(&self.numerator, &other.denominator, budget)?.cmp(&multiply(
                &other.numerator,
                &self.denominator,
                budget,
            )?),
        )
    }

    fn compare_time(&self, time: Real, budget: &mut usize) -> SpiceResult<Ordering> {
        Ok(self
            .numerator
            .cmp(&multiply(&units(time), &self.denominator, budget)?))
    }

    fn time_satisfies(&self, time: Real, lower: bool, budget: &mut usize) -> SpiceResult<bool> {
        let cmp = self.compare_time(time, budget)?;
        Ok(if lower {
            cmp == Ordering::Less || (cmp == Ordering::Equal && self.inclusive)
        } else {
            cmp == Ordering::Greater || (cmp == Ordering::Equal && self.inclusive)
        })
    }

    /// Directed conversion on the ordered nonnegative binary64 lattice.
    /// No rounded quotient or arbitrary epsilon is involved.
    fn represented(&self, lower: bool, budget: &mut usize) -> SpiceResult<Option<Real>> {
        let mut lo = 0_u64;
        let mut hi = Real::MAX.to_bits();
        if !self.time_satisfies(if lower { Real::MAX } else { 0. }, lower, budget)? {
            return Ok(None);
        }
        while lo < hi {
            let span = hi - lo;
            let mid = lo + if lower { span / 2 } else { span.div_ceil(2) };
            let accepted = self.time_satisfies(Real::from_bits(mid), lower, budget)?;
            if lower {
                if accepted {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            } else if accepted {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Ok(Some(Real::from_bits(lo)))
    }
}

/// A nonempty inclusive interval on the representable physical-time
/// lattice. Strict real endpoints are converted to the adjacent float
/// only when equality would not satisfy the original guard.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectedInterval {
    first: Real,
    last: Real,
}

impl ProjectedInterval {
    /// First representable observation satisfying the guard.
    #[must_use = "the endpoint is the first feasible observation"]
    pub const fn first(self) -> Real {
        self.first
    }
    /// Last representable observation satisfying the guard.
    #[must_use = "the endpoint is the last feasible observation"]
    pub const fn last(self) -> Real {
        self.last
    }
    /// Whether this representable physical time satisfies the interval.
    #[must_use = "the interval determines feasibility at the observation"]
    pub fn contains(self, time: Real) -> bool {
        (self.first..=self.last).contains(&time)
    }
}

fn project(rows: Vec<Row>, budget: &mut usize) -> SpiceResult<Option<ProjectedInterval>> {
    let Some(rows) = eliminate(rows, Some(Variable::Observation), budget)? else {
        return Ok(None);
    };
    let mut lower = RationalBoundary {
        numerator: BigInt::ZERO,
        denominator: BigInt::from(1),
        inclusive: true,
    };
    let mut upper = RationalBoundary {
        numerator: units(Real::MAX),
        denominator: BigInt::from(1),
        inclusive: true,
    };
    for row in rows {
        let coefficient = row
            .expression
            .terms
            .get(&Variable::Observation)
            .ok_or_else(|| SpiceError::circuit("missing projected time coefficient"))?;
        let (bound, lower_bound) = match coefficient.sign() {
            Sign::Plus => (
                RationalBoundary {
                    numerator: -row.expression.constant,
                    denominator: coefficient.clone(),
                    inclusive: !row.strict,
                },
                false,
            ),
            Sign::Minus => (
                RationalBoundary {
                    numerator: row.expression.constant,
                    denominator: -coefficient,
                    inclusive: !row.strict,
                },
                true,
            ),
            Sign::NoSign => return Err(SpiceError::circuit("zero projected time coefficient")),
        };
        let current = if lower_bound { &mut lower } else { &mut upper };
        let cmp = bound.compare(current, budget)?;
        if cmp == Ordering::Equal {
            current.inclusive &= bound.inclusive;
        } else if (lower_bound && cmp == Ordering::Greater)
            || (!lower_bound && cmp == Ordering::Less)
        {
            *current = bound;
        }
    }
    let Some(first) = lower.represented(true, budget)? else {
        return Ok(None);
    };
    let Some(last) = upper.represented(false, budget)? else {
        return Ok(None);
    };
    Ok((first <= last).then_some(ProjectedInterval { first, last }))
}

struct Graph<'a> {
    nodes: Vec<&'a TimingWindow>,
    indices: BTreeMap<*const Node, usize>,
    choices: Vec<usize>,
}

impl<'a> Graph<'a> {
    fn new(
        comparisons: &'a [Comparison],
        observations: &'a [ObservationComparison],
        budget: &mut usize,
    ) -> SpiceResult<Self> {
        let mut pending: Vec<_> = comparisons
            .iter()
            .flat_map(|c| [&c.lhs, &c.rhs])
            .chain(observations.iter().map(|comparison| &comparison.window))
            .collect();
        let mut seen = BTreeSet::new();
        let mut nodes = Vec::new();
        while let Some(window) = pending.pop() {
            charge(budget, 1)?;
            if !seen.insert(Arc::as_ptr(&window.0)) {
                continue;
            }
            nodes.push(window);
            match &window.0.expression {
                Expression::Exact(_) => {}
                Expression::Delay { parent, .. } => pending.push(parent),
                Expression::Extremum { inputs, .. } => {
                    charge(budget, inputs.len())?;
                    pending.extend(inputs);
                }
            }
        }
        nodes.sort_by_key(|window| window.0.depth);
        let indices = nodes
            .iter()
            .enumerate()
            .map(|(index, window)| (Arc::as_ptr(&window.0), index))
            .collect();
        let choices = nodes
            .iter()
            .filter_map(|window| match &window.0.expression {
                Expression::Extremum { inputs, .. } => Some(inputs.len()),
                _ => None,
            })
            .collect();
        Ok(Self {
            nodes,
            indices,
            choices,
        })
    }

    fn index(&self, window: &TimingWindow) -> usize {
        self.indices[&Arc::as_ptr(&window.0)]
    }

    fn branch(
        &self,
        comparisons: &[Comparison],
        choices: &[usize],
        observations: &[ObservationComparison],
        budget: &mut usize,
    ) -> SpiceResult<Vec<Row>> {
        let mut forms = Vec::<Affine>::with_capacity(self.nodes.len());
        let mut rows = Vec::new();
        let mut selectors = choices.iter().copied();
        for window in &self.nodes {
            charge(budget, 1)?;
            let affine = match &window.0.expression {
                Expression::Exact(value) => Affine::exact(*value),
                Expression::Delay {
                    parent,
                    delay,
                    origin,
                } => {
                    let parent = &forms[self.index(parent)];
                    charge(budget, parent.terms.len().saturating_add(1))?;
                    let bounds = delay.bounds();
                    parent
                        .clone()
                        .delay(*origin, bounds.minimum(), bounds.maximum())
                }
                Expression::Extremum { kind, inputs } => {
                    let index = selectors
                        .next()
                        .ok_or_else(|| SpiceError::circuit("missing timing extremum selector"))?;
                    let selected = &forms[self.index(&inputs[index])];
                    for (i, input) in inputs.iter().enumerate() {
                        if i == index {
                            continue;
                        }
                        let other = &forms[self.index(input)];
                        let (lhs, rhs) = match kind {
                            Extremum::Earliest => (selected, other),
                            Extremum::Latest => (other, selected),
                        };
                        // Equality is allowed for more than one selected
                        // child. This is a feasibility union, not the
                        // public disjoint order partition.
                        rows.push(Row {
                            expression: Affine::difference(lhs, rhs, budget)?.normalized(budget)?,
                            strict: false,
                        });
                    }
                    charge(budget, selected.terms.len().saturating_add(1))?;
                    selected.clone()
                }
            };
            forms.push(affine);
        }
        for comparison in comparisons {
            let lhs = &forms[self.index(&comparison.lhs)];
            let rhs = &forms[self.index(&comparison.rhs)];
            if comparison.offset == 0. {
                Row::comparison(lhs, rhs, comparison.relation, &mut rows, budget)?;
            } else {
                charge(budget, rhs.terms.len().saturating_add(1))?;
                let mut rhs = rhs.clone();
                rhs.constant += units(comparison.offset);
                Row::comparison(lhs, &rhs, comparison.relation, &mut rows, budget)?;
            }
        }
        if !observations.is_empty() {
            let observation = Affine {
                constant: BigInt::ZERO,
                terms: BTreeMap::from([(Variable::Observation, BigInt::from(1))]),
            };
            for comparison in observations {
                Row::comparison(
                    &forms[self.index(&comparison.window)],
                    &observation,
                    comparison.relation,
                    &mut rows,
                    budget,
                )?;
            }
        }
        Ok(rows)
    }
}

#[derive(Clone, Debug)]
pub(super) struct ObservationComparison {
    pub(super) window: TimingWindow,
    pub(super) relation: TimingRelation,
}

fn visit_branches(
    comparisons: &[Comparison],
    observations: &[ObservationComparison],
    budget: &mut usize,
    mut visit: impl FnMut(Vec<Row>, &mut usize) -> SpiceResult<bool>,
) -> SpiceResult<bool> {
    let graph = Graph::new(comparisons, observations, budget)?;
    let mut selectors = vec![0; graph.choices.len()];
    loop {
        if visit(
            graph.branch(comparisons, &selectors, observations, budget)?,
            budget,
        )? {
            return Ok(true);
        }
        // Iterative mixed-radix enumeration; shared extrema select one
        // child for all their consumers. Never assume that selections
        // in two separately compared expressions are independent.
        let mut advanced = false;
        for (selector, &count) in selectors.iter_mut().zip(&graph.choices).rev() {
            charge(budget, 1)?;
            *selector += 1;
            if *selector < count {
                advanced = true;
                break;
            }
            *selector = 0;
        }
        if !advanced {
            return Ok(false);
        }
    }
}

pub(super) fn guard_feasible(comparisons: &[Comparison], budget: &mut usize) -> SpiceResult<bool> {
    visit_branches(comparisons, &[], budget, feasible)
}

pub(super) fn guard_projection(
    comparisons: &[Comparison],
    observations: &[ObservationComparison],
    budget: &mut usize,
) -> SpiceResult<Vec<ProjectedInterval>> {
    let mut intervals = Vec::new();
    visit_branches(comparisons, observations, budget, |rows, budget| {
        if let Some(interval) = project(rows, budget)? {
            intervals.push(interval);
        }
        Ok(false)
    })?;
    intervals.sort_by(|a, b| {
        a.first
            .total_cmp(&b.first)
            .then_with(|| a.last.total_cmp(&b.last))
    });
    let mut merged: Vec<ProjectedInterval> = Vec::new();
    for interval in intervals {
        if let Some(last) = merged.last_mut()
            && interval.first <= last.last.next_up()
        {
            last.last = last.last.max(interval.last);
        } else {
            merged.push(interval);
        }
    }
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::super::{DelayWindow, GuardStatus, TimingGuard};
    use super::*;

    #[test]
    fn exact_dyadic_units_span_all_finite_binary64_exponents() {
        assert_eq!(units(Real::from_bits(1)), BigInt::from(1));
        assert_eq!(units(-Real::from_bits(1)), BigInt::from(-1));
        assert_eq!(units(1.), BigInt::from(1) << 1074_u32);
        assert_eq!(units(Real::MAX).bits(), 2098);
    }

    #[test]
    fn midpoint_cannot_be_strictly_first_or_last() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let v = Origin::shared(1);
        let t = TimingWindow::exact(100.)?;
        let a = t
            .after(DelayWindow::new(0., 10.)?, u)?
            .after(DelayWindow::new(0., 10.)?, v)?;
        let b = t.after(DelayWindow::new(0., 20.)?, u)?;
        let c = t.after(DelayWindow::new(0., 20.)?, v)?;
        for relation in [TimingRelation::Before, TimingRelation::After] {
            let guard = TimingGuard::compare(a.clone(), b.clone(), relation)
                .and(TimingGuard::compare(a.clone(), c.clone(), relation));
            assert_eq!(guard.status(100_000)?, GuardStatus::Impossible);
        }
        let guard = TimingGuard::compare(a.clone(), b.clone(), TimingRelation::Coincident)
            .and(TimingGuard::compare(a, c, TimingRelation::Coincident));
        assert_ne!(guard.status(100_000)?, GuardStatus::Impossible);
        Ok(())
    }

    #[test]
    fn exact_strict_feasibility_including_shared_extremum() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let v = Origin::shared(1);
        let t = TimingWindow::exact(10.)?;
        let a = t.after(DelayWindow::new(0., 2.)?, u)?;
        let b = t.after(DelayWindow::new(0., 2.)?, v)?;
        let earliest = TimingWindow::earliest(vec![a.clone(), b.clone()])?;
        let latest = TimingWindow::latest(vec![a.clone(), b.clone()])?;
        let middle = t
            .after(DelayWindow::new(0., 1.)?, u)?
            .after(DelayWindow::new(0., 1.)?, v)?;
        let impossible =
            TimingGuard::compare(earliest.clone(), middle.clone(), TimingRelation::After);
        assert_eq!(impossible.status(100_000)?, GuardStatus::Impossible);
        let possible = TimingGuard::compare(earliest, middle.clone(), TimingRelation::Before)
            .and(TimingGuard::compare(middle, latest, TimingRelation::Before));
        assert_ne!(possible.status(100_000)?, GuardStatus::Impossible);
        let at_zero = TimingGuard::compare(a, t.clone(), TimingRelation::Coincident)
            .and(TimingGuard::compare(b, t, TimingRelation::Coincident));
        assert_ne!(at_zero.status(100_000)?, GuardStatus::Impossible);
        Ok(())
    }

    #[test]
    fn strict_rows_cannot_erase_a_zero_width_contradiction() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let t = TimingWindow::exact(0.)?;
        let a = t.after(DelayWindow::new(0., Real::from_bits(1))?, u)?;
        let end = TimingWindow::exact(Real::from_bits(1))?;
        let impossible = TimingGuard::compare(a.clone(), t, TimingRelation::After)
            .and(TimingGuard::compare(a, end, TimingRelation::Coincident));
        assert_ne!(impossible.status(100_000)?, GuardStatus::Impossible);
        let zero = Affine {
            constant: BigInt::ZERO,
            terms: BTreeMap::from([(Variable::Origin(u), BigInt::from(1))]),
        };
        let neg = Affine {
            constant: BigInt::ZERO,
            terms: BTreeMap::from([(Variable::Origin(u), BigInt::from(-1))]),
        };
        assert!(!feasible(
            vec![
                Row {
                    expression: zero,
                    strict: true
                },
                Row {
                    expression: neg,
                    strict: false
                }
            ],
            &mut 100_000
        )?);
        Ok(())
    }
    #[test]
    fn projection_finds_interior_guarded_boundary_and_respects_strictness() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let a = TimingWindow::exact(10.)?.after(DelayWindow::new(0., 20.)?, u)?;
        let b = TimingWindow::exact(15.)?.after(DelayWindow::new(0., 10.)?, u)?;
        let guard = TimingGuard::compare(a.clone(), b.clone(), TimingRelation::Before);
        let intervals = guard.observation_intervals(
            &[(a, TimingRelation::NoLater), (b, TimingRelation::After)],
            100_000,
        )?;
        assert_eq!(
            intervals,
            [ProjectedInterval {
                first: 10.,
                last: 20_f64.next_down()
            }]
        );
        Ok(())
    }

    #[test]
    fn shared_extrema_projection_keeps_disconnected_point_branches() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let v = Origin::shared(1);
        let zero = TimingWindow::exact(0.)?;
        let one = TimingWindow::exact(1.)?;
        let a = zero.after(DelayWindow::new(0., 1.)?, u)?;
        let b = zero.after(DelayWindow::new(0., 1.)?, v)?;
        let first = TimingWindow::earliest(vec![a.clone(), b.clone()])?;
        let last = TimingWindow::latest(vec![a.clone(), b.clone()])?;
        let sum = a.after(DelayWindow::new(0., 2.)?, v)?;
        let guard = TimingGuard::compare(first, zero, TimingRelation::Coincident)
            .and(TimingGuard::compare(last, one, TimingRelation::Coincident));
        let intervals =
            guard.observation_intervals(&[(sum, TimingRelation::Coincident)], 100_000)?;
        assert_eq!(
            intervals,
            [
                ProjectedInterval {
                    first: 1.,
                    last: 1.
                },
                ProjectedInterval {
                    first: 2.,
                    last: 2.
                }
            ]
        );
        Ok(())
    }

    #[test]
    fn projection_rounds_exact_nonrepresentable_rational_boundary_inward() -> SpiceResult<()> {
        // x=1/3, hence 1+x=4/3 exactly, not a binary64 value.
        let u = Origin::shared(0);
        let origin = TimingWindow::exact(0.)?;
        let triple = origin.after(DelayWindow::new(0., 3.)?, u)?;
        let shifted = TimingWindow::exact(1.)?.after(DelayWindow::new(0., 1.)?, u)?;
        let guard =
            TimingGuard::compare(triple, TimingWindow::exact(1.)?, TimingRelation::Coincident);
        let intervals =
            guard.observation_intervals(&[(shifted.clone(), TimingRelation::NoLater)], 100_000)?;
        assert_eq!(intervals[0].first, (4_f64 / 3.).next_up());
        let points =
            guard.observation_intervals(&[(shifted, TimingRelation::Coincident)], 100_000)?;
        assert!(points.is_empty());
        Ok(())
    }
}
