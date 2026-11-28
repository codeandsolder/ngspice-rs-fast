//! A symbolic partition by physical arrival order. Each case is guarded
//! by simultaneous equality inside groups and strict order between them.
//! Relative bounds reject simple contradictions; exact joint
//! feasibility rules out impossible orders from correlated paths.

use super::{Expression, Extremum, Origin, TimingWindow, charge, relative};
use crate::primitives::{ExactSum, Real, SpiceError, SpiceResult};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::Arc;

/// One ordered partition of the input arrival expressions. A case holds
/// precisely when all members of each group arrive simultaneously and
/// every group precedes the following group.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrderCase {
    groups: Vec<Vec<usize>>,
}

impl OrderCase {
    /// Input indices, grouped by simultaneous arrival in physical order.
    #[must_use = "the groups describe the conditional order"]
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }
}

/// Fast path for a proven total preorder. Compare adjacent expressions
/// in absolute-bound order; transitivity establishes all other pairs.
/// Any uncertainty goes to the complete guarded partition.
pub(crate) fn proven_order(
    mut windows: Vec<(usize, &TimingWindow)>,
    budget: &mut usize,
) -> SpiceResult<Option<OrderCase>> {
    let n = windows.len();
    charge(
        budget,
        n.saturating_mul(
            usize::try_from(n.checked_ilog2().unwrap_or(0))
                .unwrap_or(usize::MAX)
                .saturating_add(1),
        ),
    )?;
    windows.sort_by(|(_, a), (_, b)| {
        let a = a.bounds();
        let b = b.bounds();
        a.minimum()
            .total_cmp(&b.minimum())
            .then_with(|| a.maximum().total_cmp(&b.maximum()))
    });
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut previous: Option<&TimingWindow> = None;
    for (index, window) in windows {
        if let (Some(prior), Some(group)) = (previous, groups.last_mut()) {
            let b = relative(window, prior, budget)?;
            if b.is_exact() && b.minimum() == 0. {
                group.push(index);
                continue;
            }
            if b.minimum() <= 0. {
                return Ok(None);
            }
        }
        groups.push(vec![index]);
        previous = Some(window);
    }
    for group in &mut groups {
        group.sort_unstable();
    }
    Ok(Some(OrderCase { groups }))
}

/// Result of a sound feasibility proof about an ordering guard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuardStatus {
    /// The conjunction is proven contradictory.
    Impossible,
    /// All required comparisons hold for every realization.
    Always,
    /// A feasible realization exists; universality is not proven.
    Conditional,
}

/// A comparison of two physical arrival expressions. Inclusive
/// comparisons describe delivery at an exact observation boundary
/// without splitting a single half-space into a strict case and tie.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TimingRelation {
    /// Strictly earlier.
    Before,
    /// Earlier or simultaneous.
    NoLater,
    /// Simultaneous.
    Coincident,
    /// Later or simultaneous.
    NoEarlier,
    /// Strictly later.
    After,
}

impl TimingRelation {
    fn classify(self, bounds: super::Bounds, threshold: Real) -> (bool, bool) {
        let lo = bounds.minimum();
        let hi = bounds.maximum();
        match self {
            Self::Before => (lo < threshold, hi < threshold),
            Self::NoLater => (lo <= threshold, hi <= threshold),
            Self::Coincident => (
                lo <= threshold && hi >= threshold,
                lo == threshold && hi == threshold,
            ),
            Self::NoEarlier => (hi >= threshold, lo >= threshold),
            Self::After => (hi > threshold, lo > threshold),
        }
    }

    const fn complement(self) -> &'static [Self] {
        match self {
            Self::Before => &[Self::NoEarlier],
            Self::NoLater => &[Self::After],
            Self::Coincident => &[Self::Before, Self::After],
            Self::NoEarlier => &[Self::Before],
            Self::After => &[Self::NoLater],
        }
    }

    fn accepts(self, order: Ordering) -> bool {
        match self {
            Self::Before => order == Ordering::Less,
            Self::NoLater => order != Ordering::Greater,
            Self::Coincident => order == Ordering::Equal,
            Self::NoEarlier => order != Ordering::Less,
            Self::After => order == Ordering::Greater,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct Comparison {
    pub(super) lhs: TimingWindow,
    pub(super) rhs: TimingWindow,
    pub(super) relation: TimingRelation,
    pub(super) offset: Real,
}

/// A conjunction of physical-time comparisons on shared delay origins.
/// Empty conjunction is true; guards from independent partitions may
/// be combined without making their uncertainty variables independent.
#[derive(Clone, Debug, Default)]
pub struct TimingGuard {
    comparisons: Vec<Comparison>,
}

impl TimingGuard {
    /// One exact physical-time comparison, preserving both expressions.
    #[must_use = "the guard constrains a joint timing realization"]
    pub fn compare(lhs: TimingWindow, rhs: TimingWindow, relation: TimingRelation) -> Self {
        Self {
            comparisons: vec![Comparison {
                lhs,
                rhs,
                relation,
                offset: 0.,
            }],
        }
    }

    /// Compare a physical separation with an exact finite constant:
    /// `lhs - rhs relation offset`. The offset is not an event time,
    /// so sub-ULP or nonrepresentable absolute deadlines do not round
    /// the constraint or create spurious delivery events.
    ///
    /// # Errors
    /// Nonfinite comparison offset.
    pub fn separation(
        lhs: TimingWindow,
        rhs: TimingWindow,
        relation: TimingRelation,
        offset: Real,
    ) -> SpiceResult<Self> {
        if !offset.is_finite() {
            return Err(SpiceError::circuit(
                "nonfinite timing separation constraint",
            ));
        }
        Ok(Self {
            comparisons: vec![Comparison {
                lhs,
                rhs,
                relation,
                offset,
            }],
        })
    }

    /// Conjoin two guarded conditions, retaining the original expressions.
    #[must_use = "the conjunction retains upstream causal constraints"]
    pub fn and(mut self, other: Self) -> Self {
        self.comparisons.extend(other.comparisons);
        self
    }

    /// Check a guard with fast relative enclosures and strict-order
    /// cycle detection, followed by exact affine feasibility including
    /// shared min/max selections when the fast proofs are inconclusive.
    ///
    /// # Errors
    /// Exhausted comparison budget or invalid relative arithmetic.
    pub fn status(&self, mut budget: usize) -> SpiceResult<GuardStatus> {
        self.status_with_budget(&mut budget)
    }

    pub(super) fn status_with_budget(&self, budget: &mut usize) -> SpiceResult<GuardStatus> {
        let mut uncertain = Vec::new();
        for comparison in &self.comparisons {
            let b = relative(&comparison.lhs, &comparison.rhs, budget)?;
            let (possible, always) = comparison.relation.classify(b, comparison.offset);
            if !possible {
                return Ok(GuardStatus::Impossible);
            }
            if !always {
                uncertain.push(comparison.clone());
            }
        }
        if uncertain.is_empty() {
            return Ok(GuardStatus::Always);
        }
        if self.has_strict_cycle(budget)?
            || !super::feasibility::guard_feasible(&uncertain, budget)?
        {
            return Ok(GuardStatus::Impossible);
        }
        Ok(GuardStatus::Conditional)
    }

    /// Project this guard and a set of event-time comparisons against
    /// an observation time onto the representable nonnegative time axis.
    /// Each tuple means `event relation observation`. All delay origins
    /// and extremum selections remain shared. Returned intervals are
    /// sorted, disjoint, and inclusive on the binary64 lattice.
    ///
    /// # Errors
    /// Invalid expression or exhausted exact feasibility budget.
    pub fn observation_intervals(
        &self,
        events: &[(TimingWindow, TimingRelation)],
        mut budget: usize,
    ) -> SpiceResult<Vec<super::ProjectedInterval>> {
        self.observation_intervals_with_budget(events, &mut budget)
    }

    pub(super) fn observation_intervals_with_budget(
        &self,
        events: &[(TimingWindow, TimingRelation)],
        budget: &mut usize,
    ) -> SpiceResult<Vec<super::ProjectedInterval>> {
        let observations = events
            .iter()
            .map(
                |(window, relation)| super::feasibility::ObservationComparison {
                    window: window.clone(),
                    relation: *relation,
                },
            )
            .collect::<Vec<_>>();
        super::feasibility::guard_projection(&self.comparisons, &observations, budget)
    }

    // Equality is a pair of non-strict edges. A strongly connected
    // component containing any strict inequality is contradictory,
    // including across independently constructed partitions which
    // share the underlying arrival expressions.
    fn has_strict_cycle(&self, budget: &mut usize) -> SpiceResult<bool> {
        charge(budget, self.comparisons.len().saturating_mul(4))?;
        let mut graph = DiGraph::<(), bool>::new();
        let mut nodes = BTreeMap::new();
        for comparison in &self.comparisons {
            // Weighted cycles are decided by affine feasibility, not
            // by the unweighted strict-order shortcut.
            if comparison.offset != 0. {
                continue;
            }
            let mut node = |window: &TimingWindow| -> NodeIndex {
                *nodes
                    .entry(Arc::as_ptr(&window.0))
                    .or_insert_with(|| graph.add_node(()))
            };
            let lhs = node(&comparison.lhs);
            let rhs = node(&comparison.rhs);
            match comparison.relation {
                TimingRelation::Before => {
                    graph.add_edge(lhs, rhs, true);
                }
                TimingRelation::NoLater => {
                    graph.add_edge(lhs, rhs, false);
                }
                TimingRelation::Coincident => {
                    graph.add_edge(lhs, rhs, false);
                    graph.add_edge(rhs, lhs, false);
                }
                TimingRelation::NoEarlier => {
                    graph.add_edge(rhs, lhs, false);
                }
                TimingRelation::After => {
                    graph.add_edge(rhs, lhs, true);
                }
            }
        }
        let components = petgraph::algo::kosaraju_scc(&graph);
        let mut component = vec![0; graph.node_count()];
        for (index, nodes) in components.iter().enumerate() {
            for node in nodes {
                component[node.index()] = index;
            }
        }
        Ok(graph.edge_references().any(|edge| {
            *edge.weight() && component[edge.source().index()] == component[edge.target().index()]
        }))
    }

    /// Evaluate the conjunction at a concrete joint origin realization.
    ///
    /// # Errors
    /// Missing/invalid origin assignment or exhausted work budget.
    pub fn holds(&self, origins: &BTreeMap<Origin, Real>, mut budget: usize) -> SpiceResult<bool> {
        self.holds_with_budget(origins, &mut budget)
    }

    pub(super) fn holds_with_budget(
        &self,
        origins: &BTreeMap<Origin, Real>,
        budget: &mut usize,
    ) -> SpiceResult<bool> {
        for comparison in &self.comparisons {
            let lhs = exact_at(&comparison.lhs, origins, budget)?;
            let mut rhs = exact_at(&comparison.rhs, origins, budget)?;
            rhs.add_product(comparison.offset, 1.);
            if !comparison.relation.accepts(lhs.exact_cmp(&rhs)) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// A guarded result of evaluating a complete order partition. Each
/// payload belongs to one disjoint case, not an independent possible
/// state at an arbitrary time.
#[derive(Clone, Debug)]
pub struct Conditional<T> {
    partition: OrderPartition,
    payloads: Vec<T>,
}

impl<T> Conditional<T> {
    /// The guarded payloads in case order.
    pub fn alternatives(&self) -> impl Iterator<Item = (&OrderCase, &T)> {
        self.partition.cases.iter().zip(&self.payloads)
    }

    /// Recover the guard for each payload, for composition with upstream
    /// conditions.
    pub fn guarded(&self) -> impl Iterator<Item = (TimingGuard, &T)> {
        self.alternatives()
            .map(|(case, payload)| (self.partition.guard(case), payload))
    }

    /// Select the result under the exact shared-origin assignment.
    ///
    /// # Errors
    /// Missing/invalid origin assignment or exhausted work budget.
    pub fn realize(&self, origins: &BTreeMap<Origin, Real>, budget: usize) -> SpiceResult<&T> {
        let index = self.partition.realize_index(origins, budget)?;
        self.payloads
            .get(index)
            .ok_or_else(|| SpiceError::circuit("missing conditional payload"))
    }
}

/// A complete disjoint set of guarded payloads. Constructed from a
/// verified order partition, or by composing such sets under conjunction.
/// Unlike a set of independently possible values, a branch retains the
/// same origin variables when it is combined with another signal.
#[derive(Clone, Debug)]
pub struct Guarded<T> {
    branches: Vec<(TimingGuard, T)>,
}

impl<T> Guarded<T> {
    // Internal algorithms construct a complete partition by exact
    // case splitting; do not expose unchecked guard lists to callers.
    pub(super) fn from_partitioned(branches: Vec<(TimingGuard, T)>) -> SpiceResult<Self> {
        if branches.is_empty() {
            return Err(SpiceError::circuit("empty timing partition"));
        }
        Ok(Self { branches })
    }

    /// An unconditional payload, without uncertain case selection.
    pub fn unconditional(payload: T) -> Self {
        Self {
            branches: vec![(TimingGuard::default(), payload)],
        }
    }

    /// Retain the guards of a verified order partition.
    #[must_use = "the partition retains every feasible guard"]
    pub fn from_conditional(conditional: Conditional<T>) -> Self {
        let Conditional {
            partition,
            payloads,
        } = conditional;
        let branches = partition
            .cases
            .iter()
            .zip(payloads)
            .map(|(case, payload)| (partition.guard(case), payload))
            .collect();
        Self { branches }
    }

    // A priority decision list forms a complete partition: each match
    // excludes all preceding predicates, and the fallback excludes
    // every predicate. A predicate can be false in more than one
    // disjoint relation (the complement of coincidence).
    pub(crate) fn first_match(
        choices: impl IntoIterator<Item = (TimingWindow, TimingRelation, TimingWindow, T)>,
        fallback: T,
        budget: &mut usize,
    ) -> SpiceResult<Self>
    where
        T: Clone,
    {
        let mut pending = vec![TimingGuard::default()];
        let mut branches = Vec::new();
        for (lhs, relation, rhs, payload) in choices {
            if pending.is_empty() {
                break;
            }
            let predicate = TimingGuard::compare(lhs.clone(), rhs.clone(), relation);
            match predicate.status_with_budget(budget)? {
                GuardStatus::Impossible => continue,
                GuardStatus::Always => {
                    branches.extend(pending.drain(..).map(|guard| (guard, payload.clone())));
                    break;
                }
                GuardStatus::Conditional => {}
            }
            let mut next = Vec::new();
            for guard in pending {
                charge(budget, 1)?;
                let selected = guard.clone().and(predicate.clone());
                if selected.status_with_budget(budget)? != GuardStatus::Impossible {
                    branches.push((selected, payload.clone()));
                }
                for &opposite in relation.complement() {
                    let rejected =
                        guard
                            .clone()
                            .and(TimingGuard::compare(lhs.clone(), rhs.clone(), opposite));
                    if rejected.status_with_budget(budget)? != GuardStatus::Impossible {
                        next.push(rejected);
                    }
                }
            }
            pending = next;
        }
        branches.extend(pending.into_iter().map(|guard| (guard, fallback.clone())));
        Self::from_partitioned(branches)
    }

    /// Inspect each mutually exclusive branch and its payload.
    pub fn branches(&self) -> impl Iterator<Item = (&TimingGuard, &T)> {
        self.branches
            .iter()
            .map(|(guard, payload)| (guard, payload))
    }

    /// Transform each payload without changing its guard.
    ///
    /// # Errors
    /// Payload transformation failure.
    pub fn try_map<U>(
        &self,
        mut transform: impl FnMut(&T) -> SpiceResult<U>,
    ) -> SpiceResult<Guarded<U>> {
        let branches = self
            .branches
            .iter()
            .map(|(guard, payload)| Ok((guard.clone(), transform(payload)?)))
            .collect::<SpiceResult<Vec<_>>>()?;
        Ok(Guarded { branches })
    }

    /// Compose two complete partitions over the same shared-origin
    /// domain. Infeasible conjunctions are removed by exact feasibility.
    ///
    /// # Errors
    /// Exhausted budget or payload transformation failure.
    pub fn zip<U, V>(
        &self,
        other: &Guarded<U>,
        mut budget: usize,
        transform: impl FnMut(&T, &U) -> SpiceResult<V>,
    ) -> SpiceResult<Guarded<V>> {
        self.zip_with_budget(other, &mut budget, transform)
    }

    pub(crate) fn zip_with_budget<U, V>(
        &self,
        other: &Guarded<U>,
        budget: &mut usize,
        mut transform: impl FnMut(&T, &U) -> SpiceResult<V>,
    ) -> SpiceResult<Guarded<V>> {
        let mut branches = Vec::new();
        for (lhs, a) in &self.branches {
            for (rhs, b) in &other.branches {
                charge(budget, 1)?;
                let guard = lhs.clone().and(rhs.clone());
                if guard.status_with_budget(budget)? != GuardStatus::Impossible {
                    branches.push((guard, transform(a, b)?));
                }
            }
        }
        if branches.is_empty() {
            return Err(SpiceError::circuit("empty composed timing partition"));
        }
        Ok(Guarded { branches })
    }

    /// Expand each payload into a complete guarded partition, conjoining
    /// its existing condition with the newly selected conditions.
    ///
    /// # Errors
    /// Exhausted budget or payload transformation failure.
    pub fn try_flat_map<U>(
        &self,
        mut budget: usize,
        mut transform: impl FnMut(&T) -> SpiceResult<Guarded<U>>,
    ) -> SpiceResult<Guarded<U>> {
        let mut branches = Vec::new();
        for (guard, payload) in &self.branches {
            for (next, result) in transform(payload)?.branches {
                charge(&mut budget, 1)?;
                let guard = guard.clone().and(next);
                if guard.status_with_budget(&mut budget)? != GuardStatus::Impossible {
                    branches.push((guard, result));
                }
            }
        }
        if branches.is_empty() {
            return Err(SpiceError::circuit("empty expanded timing partition"));
        }
        Ok(Guarded { branches })
    }

    /// Select the unique payload for a concrete joint origin assignment.
    ///
    /// # Errors
    /// Missing origin, invalid realization, exhausted budget, or
    /// inconsistent branch coverage.
    pub fn realize(&self, origins: &BTreeMap<Origin, Real>, mut budget: usize) -> SpiceResult<&T> {
        self.realize_with_budget(origins, &mut budget)
    }

    pub(super) fn realize_with_budget(
        &self,
        origins: &BTreeMap<Origin, Real>,
        budget: &mut usize,
    ) -> SpiceResult<&T> {
        let mut selected = None;
        for (guard, payload) in &self.branches {
            charge(budget, 1)?;
            if guard.holds_with_budget(origins, budget)? {
                if selected.is_some() {
                    return Err(SpiceError::circuit("overlapping realized timing branches"));
                }
                selected = Some(payload);
            }
        }
        selected.ok_or_else(|| SpiceError::circuit("no realized timing branch"))
    }
}

/// A disjoint symbolic partition of the joint delay-origin domain.
/// Every retained case has a jointly feasible guard, and every
/// concrete realization belongs to exactly one case.
#[derive(Clone, Debug)]
pub struct OrderPartition {
    windows: Vec<TimingWindow>,
    cases: Vec<OrderCase>,
}

#[derive(Clone, Copy)]
struct Relations {
    less: bool,
    equal: bool,
    greater: bool,
}

impl Relations {
    fn reverse(self) -> Self {
        Self {
            less: self.greater,
            greater: self.less,
            equal: self.equal,
        }
    }
}

impl OrderPartition {
    /// Construct the complete partition of feasible physical orders.
    /// No assumption of independent absolute timing windows is made.
    /// The shared budget bounds comparison, case expansion and cloning
    /// work; exhaustion is an explicit error, never a partial partition.
    ///
    /// # Errors
    /// Invalid expression, comparison failure, or exhausted work budget.
    pub fn new(windows: Vec<TimingWindow>, mut budget: usize) -> SpiceResult<Self> {
        if let Some(case) = proven_order(windows.iter().enumerate().collect(), &mut budget)? {
            return Ok(Self {
                windows,
                cases: vec![case],
            });
        }
        let count = windows.len();
        charge(&mut budget, count.saturating_mul(count))?;
        let mut relations = vec![
            vec![
                Relations {
                    less: false,
                    equal: true,
                    greater: false
                };
                count
            ];
            count
        ];
        for i in 0..count {
            for j in 0..i {
                let separation = relative(&windows[i], &windows[j], &mut budget)?;
                let relation = Relations {
                    less: separation.minimum() < 0.,
                    equal: separation.minimum() <= 0. && separation.maximum() >= 0.,
                    greater: separation.maximum() > 0.,
                };
                relations[i][j] = relation;
                relations[j][i] = relation.reverse();
            }
        }
        let mut cases = vec![OrderCase { groups: Vec::new() }];
        // Insert each next event exactly once into the existing order:
        // either simultaneous with one group or between two groups.
        // This enumerates ordered partitions without duplicate cases,
        // including zero-measure equality branches at interval boundaries.
        for (event, relation) in relations.iter().enumerate() {
            let mut next = Vec::new();
            for case in cases {
                let groups = &case.groups;
                for insertion in 0..=groups.len() {
                    for simultaneous in [false, true] {
                        if simultaneous && insertion == groups.len() {
                            continue;
                        }
                        charge(&mut budget, event.saturating_add(1))?;
                        let allowed = groups.iter().enumerate().all(|(i, group)| {
                            group.iter().all(|&other| {
                                let r = relation[other];
                                if i < insertion {
                                    r.greater
                                } else if i == insertion && simultaneous {
                                    r.equal
                                } else {
                                    r.less
                                }
                            })
                        });
                        if allowed {
                            let mut groups = groups.clone();
                            if simultaneous {
                                groups[insertion].push(event);
                            } else {
                                groups.insert(insertion, vec![event]);
                            }
                            next.push(OrderCase { groups });
                        }
                    }
                }
            }
            cases = next;
        }
        let mut partition = Self {
            windows,
            cases: Vec::new(),
        };
        for case in cases {
            if partition.guard(&case).status_with_budget(&mut budget)? != GuardStatus::Impossible {
                partition.cases.push(case);
            }
        }
        if partition.cases.is_empty() {
            return Err(SpiceError::circuit("empty conditional timing partition"));
        }
        Ok(partition)
    }

    /// The mutually exclusive and jointly feasible order cases.
    #[must_use = "retain each case's ordering guard"]
    pub fn cases(&self) -> &[OrderCase] {
        &self.cases
    }

    fn guard(&self, case: &OrderCase) -> TimingGuard {
        let mut comparisons = Vec::new();
        let mut previous: Option<usize> = None;
        for group in &case.groups {
            if let Some((&first, rest)) = group.split_first() {
                if let Some(prior) = previous {
                    comparisons.push(Comparison {
                        lhs: self.windows[prior].clone(),
                        rhs: self.windows[first].clone(),
                        relation: TimingRelation::Before,
                        offset: 0.,
                    });
                }
                for &other in rest {
                    comparisons.push(Comparison {
                        lhs: self.windows[first].clone(),
                        rhs: self.windows[other].clone(),
                        relation: TimingRelation::Coincident,
                        offset: 0.,
                    });
                }
                previous = Some(first);
            }
        }
        TimingGuard { comparisons }
    }

    /// Evaluate a deterministic reducer separately under every ordering
    /// guard. The payload can contain state, output requests and timing
    /// expressions; no endpoint or event count is guessed for the union.
    ///
    /// # Errors
    /// Reducer failure or exhausted case-evaluation budget.
    pub fn evaluate<T>(
        self,
        mut budget: usize,
        mut reducer: impl FnMut(&OrderCase) -> SpiceResult<T>,
    ) -> SpiceResult<Conditional<T>> {
        let mut payloads = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            charge(&mut budget, self.windows.len().saturating_add(1))?;
            payloads.push(reducer(case)?);
        }
        Ok(Conditional {
            partition: self,
            payloads,
        })
    }

    /// Select the unique case for one assignment of normalized delay
    /// origins. Evaluates min/max and shared affine paths without
    /// rounding away relative order at a large absolute timestamp.
    ///
    /// # Errors
    /// Missing/invalid origin assignment or exhausted work budget.
    pub fn realize(
        &self,
        origins: &BTreeMap<Origin, Real>,
        budget: usize,
    ) -> SpiceResult<&OrderCase> {
        let index = self.realize_index(origins, budget)?;
        self.cases
            .get(index)
            .ok_or_else(|| SpiceError::circuit("missing conditional order case"))
    }

    fn realize_index(
        &self,
        origins: &BTreeMap<Origin, Real>,
        mut budget: usize,
    ) -> SpiceResult<usize> {
        let values = self
            .windows
            .iter()
            .map(|window| exact_at(window, origins, &mut budget))
            .collect::<SpiceResult<Vec<_>>>()?;
        let mut indices: Vec<_> = (0..values.len()).collect();
        // Charge an upper bound for comparison work before invoking the
        // sort (whose comparator cannot return a resource error).
        let n = indices.len();
        charge(
            &mut budget,
            n.saturating_mul(
                usize::try_from(n.checked_ilog2().unwrap_or(0))
                    .unwrap_or(usize::MAX)
                    .saturating_add(1),
            ),
        )?;
        indices.sort_by(|&a, &b| values[a].exact_cmp(&values[b]).then_with(|| a.cmp(&b)));
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for index in indices {
            if let Some(group) = groups.last_mut()
                && values[group[0]].exact_cmp(&values[index]) == Ordering::Equal
            {
                group.push(index);
                continue;
            }
            groups.push(vec![index]);
        }
        for (index, case) in self.cases.iter().enumerate() {
            charge(&mut budget, n.saturating_add(1))?;
            if case.groups == groups {
                return Ok(index);
            }
        }
        Err(SpiceError::circuit(
            "realization missing from conditional timing partition",
        ))
    }
}

enum Evaluation<'a> {
    Visit(&'a TimingWindow),
    Delay(super::DelayWindow, Origin),
    Extremum(Extremum, usize),
}

pub(super) fn exact_at(
    window: &TimingWindow,
    origins: &BTreeMap<Origin, Real>,
    budget: &mut usize,
) -> SpiceResult<ExactSum> {
    let mut tasks = vec![Evaluation::Visit(window)];
    let mut values = Vec::<ExactSum>::new();
    while let Some(task) = tasks.pop() {
        charge(budget, 1)?;
        match task {
            Evaluation::Visit(window) => match &window.0.expression {
                Expression::Exact(time) => {
                    let mut sum = ExactSum::new();
                    sum.add_product(*time, 1.);
                    values.push(sum);
                }
                Expression::Delay {
                    parent,
                    delay,
                    origin,
                } => {
                    tasks.push(Evaluation::Delay(*delay, *origin));
                    tasks.push(Evaluation::Visit(parent));
                }
                Expression::Extremum { kind, inputs } => {
                    charge(budget, inputs.len())?;
                    tasks.push(Evaluation::Extremum(*kind, inputs.len()));
                    tasks.extend(inputs.iter().map(Evaluation::Visit));
                }
            },
            Evaluation::Delay(delay, origin) => {
                let sum = values
                    .last_mut()
                    .ok_or_else(|| SpiceError::circuit("missing timing realization parent"))?;
                let bounds = delay.bounds();
                sum.add_product(bounds.minimum(), 1.);
                if !bounds.is_exact() {
                    let fraction = origins
                        .get(&origin)
                        .copied()
                        .filter(|value| (0. ..=1.).contains(value))
                        .ok_or_else(|| {
                            SpiceError::circuit("missing or invalid normalized timing origin")
                        })?;
                    // Evaluate minimum + u*(maximum-minimum) as three
                    // exact products, not a rounded difference or 1-u.
                    sum.add_product(bounds.maximum(), fraction);
                    sum.add_product(bounds.minimum(), -fraction);
                }
            }
            Evaluation::Extremum(kind, count) => {
                let start = values
                    .len()
                    .checked_sub(count)
                    .ok_or_else(|| SpiceError::circuit("invalid timing realization extremum"))?;
                let mut candidates = values.drain(start..);
                let first = candidates
                    .next()
                    .ok_or_else(|| SpiceError::circuit("empty timing realization extremum"))?;
                let result = candidates.fold(first, |old, next| {
                    let cmp = next.exact_cmp(&old);
                    if matches!(
                        (kind, cmp),
                        (Extremum::Earliest, Ordering::Less)
                            | (Extremum::Latest, Ordering::Greater)
                    ) {
                        next
                    } else {
                        old
                    }
                });
                values.push(result);
            }
        }
    }
    if values.len() != 1 {
        return Err(SpiceError::circuit("invalid timing realization stack"));
    }
    values
        .pop()
        .ok_or_else(|| SpiceError::circuit("missing timing realization"))
}

#[cfg(test)]
mod tests {
    use super::super::DelayWindow;
    use super::*;

    #[test]
    fn pairwise_possible_orders_do_not_imply_joint_feasibility() -> SpiceResult<()> {
        // A = t+u+v is the midpoint of B = t+2u and C = t+2v.
        // Pairwise each can precede or follow either other event, but A
        // cannot be strictly first or last in any physical realization.
        let u = Origin::shared(0);
        let v = Origin::shared(1);
        let base = TimingWindow::exact(100.)?;
        let a = base
            .after(DelayWindow::new(0., 10.)?, u)?
            .after(DelayWindow::new(0., 10.)?, v)?;
        let b = base.after(DelayWindow::new(0., 20.)?, u)?;
        let c = base.after(DelayWindow::new(0., 20.)?, v)?;
        let partition = OrderPartition::new(vec![a, b, c], 100_000)?;
        assert_eq!(partition.cases().len(), 3);
        for i in 0..=16 {
            for j in 0..=16 {
                let values = BTreeMap::from([(u, f64::from(i) / 16.), (v, f64::from(j) / 16.)]);
                let expected = match i.cmp(&j) {
                    Ordering::Less => vec![vec![1], vec![0], vec![2]],
                    Ordering::Equal => vec![vec![0, 1, 2]],
                    Ordering::Greater => vec![vec![2], vec![0], vec![1]],
                };
                assert_eq!(partition.realize(&values, 100_000)?.groups(), expected);
            }
        }
        Ok(())
    }

    #[test]
    fn realization_orders_shared_extrema_and_exact_ties() -> SpiceResult<()> {
        let u = Origin::shared(0);
        let v = Origin::shared(1);
        let a = TimingWindow::exact(0.)?.after(DelayWindow::new(10., 20.)?, u)?;
        let b = TimingWindow::exact(0.)?.after(DelayWindow::new(10., 20.)?, v)?;
        let first = TimingWindow::earliest(vec![a.clone(), b.clone()])?;
        let last = TimingWindow::latest(vec![a.clone(), b.clone()])?;
        let partition = OrderPartition::new(vec![a, b, first, last], 100_000)?;
        for (x, y, expected) in [
            (0., 1., vec![vec![0, 2], vec![1, 3]]),
            (1., 0., vec![vec![1, 2], vec![0, 3]]),
            (0.5, 0.5, vec![vec![0, 1, 2, 3]]),
        ] {
            assert_eq!(
                partition
                    .realize(&BTreeMap::from([(u, x), (v, y)]), 100_000)?
                    .groups(),
                expected
            );
        }
        assert!(
            partition
                .realize(&BTreeMap::from([(u, 0.)]), 100_000)
                .is_err()
        );
        assert!(OrderPartition::new(partition.windows.clone(), 1).is_err());
        Ok(())
    }

    #[test]
    fn point_realization_preserves_sub_ulp_order() -> SpiceResult<()> {
        // Both absolute rounded values are indistinguishable at this
        // timestamp, but the shared normalized origin proves their
        // strict order for every nonzero assignment.
        let u = Origin::shared(0);
        let base = TimingWindow::exact(1e16)?;
        let a = base.after(DelayWindow::new(0., 4.)?, u)?;
        let b = base.after(DelayWindow::new(0., 8.)?, u)?;
        let partition = OrderPartition::new(vec![a, b], 100_000)?;
        for fraction in [Real::from_bits(1), 1e-100, 0.1, 1.] {
            assert_eq!(
                partition
                    .realize(&BTreeMap::from([(u, fraction)]), 100_000)?
                    .groups(),
                [vec![0], vec![1]]
            );
        }
        assert_eq!(
            partition
                .realize(&BTreeMap::from([(u, 0.)]), 100_000)?
                .groups(),
            [vec![0, 1]]
        );
        Ok(())
    }
    #[test]
    fn conjoined_guards_detect_strict_cycles_across_partitions() -> SpiceResult<()> {
        let a = TimingWindow::exact(0.)?.after(DelayWindow::new(10., 20.)?, Origin::shared(0))?;
        let b = TimingWindow::exact(0.)?.after(DelayWindow::new(10., 20.)?, Origin::shared(1))?;
        let c = TimingWindow::exact(0.)?.after(DelayWindow::new(10., 20.)?, Origin::shared(2))?;
        let before = |lhs: TimingWindow, rhs: TimingWindow| TimingGuard {
            comparisons: vec![Comparison {
                lhs,
                rhs,
                relation: TimingRelation::Before,
                offset: 0.,
            }],
        };
        let cycle = before(a.clone(), b.clone())
            .and(before(b, c.clone()))
            .and(before(c, a));
        assert_eq!(cycle.status(100_000)?, GuardStatus::Impossible);
        Ok(())
    }
}
