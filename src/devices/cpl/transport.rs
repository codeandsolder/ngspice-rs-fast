//! Passive split-generator transport for positive-definite L and C.
//!
//! Every static R/G stage is a reciprocal dissipative scattering relation.
//! Exact modal half-delays surround the stages (Strang splitting). For
//! mutually converting modes, stream continuous delayed PWL waveforms
//! through the stages with energy-contractive history reduction. For
//! separable modes or one stage, compile the inexpensive exact impulse
//! response and integrate it over finite boundary ramps. Both methods
//! produce an affine trial relation and commit only accepted histories.
mod continuous;

use continuous::{ContinuousHistory, ContinuousPlan};
use std::{
    cmp::{Ordering, Reverse},
    collections::{BTreeMap, BinaryHeap},
    sync::Arc,
};

use super::{
    CoupledLine, invalid, lossless::ModalBasis, plan::CoupledLinePlan,
    scattering::modal_static_scattering, wide::Wide,
};
use crate::{
    devices::{delay::DelayContext, traits::StampContext},
    primitives::{Real, SpiceResult},
};

#[derive(Debug, Clone, Copy)]
struct Arrival {
    time: Real,
    node: usize,
}
impl PartialEq for Arrival {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Arrival {}
impl Ord for Arrival {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time
            .total_cmp(&other.time)
            .then(self.node.cmp(&other.node))
    }
}
impl PartialOrd for Arrival {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Increment {
    time: Wide,
    values: Vec<Real>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Knot {
    time: Wide,
    cumulative: Wide,
    impulse: Real,
}

type Trace = Vec<Knot>;
type Queue = BTreeMap<Arrival, Increment>;

// The response to a unit ramp over [start,end], evaluated at time.
// Integrate the cumulative impulse response over the finite lag interval
// and normalize by its duration. When there are no arrivals in that
// interval, return the exact constant directly: subtracting two large
// ramp integrals would lose tiny accepted timesteps to cancellation.
fn segment_response(trace: &[Knot], time: Real, start: Real, end: Real) -> Wide {
    let low = Wide::from_real(time) - Wide::from_real(end);
    let high = Wide::from_real(time) - Wide::from_real(start);
    let duration = Wide::from_real(end) - Wide::from_real(start);
    let index = trace.partition_point(|knot| (knot.time - low).hi <= 0.);
    let mut response = index
        .checked_sub(1)
        .map_or(Wide::ZERO, |i| trace[i].cumulative);
    for knot in &trace[index..] {
        let remaining = high - knot.time;
        if remaining.hi <= 0. {
            break;
        }
        let fraction = remaining / duration;
        response =
            response + Wide::from_real(knot.impulse) * Wide::from_real(fraction.hi.clamp(0., 1.));
    }
    response
}

#[derive(Debug, Clone, PartialEq)]
struct Kernel {
    horizon: Real,
    pending: Queue,
    traces: Vec<Trace>,
    records: usize,
    // Sum of the allocated trace capacities, not just live records.
    trace_capacity: usize,
    remaining_roundoff: Vec<Real>,
}

impl Kernel {
    fn empty(ports: usize) -> Self {
        Self {
            horizon: 0.,
            pending: BTreeMap::new(),
            traces: vec![Vec::new(); ports * ports],
            records: 0,
            trace_capacity: 0,
            remaining_roundoff: vec![0.; ports],
        }
    }

    // Account for the actual trace allocation and a conservative per-entry
    // estimate of the B-tree node and separately allocated incident vector.
    fn workspace(&self, ports: usize) -> usize {
        self.trace_capacity
            .saturating_mul(std::mem::size_of::<Knot>())
            .saturating_add(
                self.traces
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Trace>()),
            )
            .saturating_add(
                self.pending
                    .len()
                    .saturating_mul(Self::arrival_bytes(ports)),
            )
    }

    const fn arrival_bytes(ports: usize) -> usize {
        3 * (std::mem::size_of::<Arrival>() + std::mem::size_of::<Increment>())
            + 32
            + ports * std::mem::size_of::<Real>()
    }

    fn push(&mut self, row: usize, time: Wide, values: &[Real]) -> SpiceResult<()> {
        let ports = values.len();
        for (col, &impulse) in values.iter().enumerate() {
            if impulse == 0. {
                continue;
            }
            let trace = &mut self.traces[row * ports + col];
            let cumulative = trace
                .last()
                .map_or(Wide::ZERO, |previous| previous.cumulative)
                + Wide::from_real(impulse);
            if !cumulative.hi.is_finite() {
                return Err(invalid("nonfinite CPL ramp response"));
            }
            let capacity = trace.capacity();
            trace.push(Knot {
                time,
                cumulative,
                impulse,
            });
            self.records += 1;
            self.trace_capacity += trace.capacity() - capacity;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Segment {
    start: Real,
    end: Real,
    delta: Vec<Real>,
}

#[derive(Debug, Clone, PartialEq)]
struct Corner {
    time: Real,
    duration: Real,
    changes: Vec<(usize, Real)>,
}

/// Accepted boundary waveform and shared, incrementally compiled response.
#[derive(Debug, Clone, PartialEq)]
pub struct CplHistory {
    time: Option<Real>,
    input: Vec<Real>,
    input_error: Vec<Real>,
    previous_input: Vec<Real>,
    previous_error: Vec<Real>,
    previous_time: Option<Real>,
    initial_output: Vec<Wide>,
    segments: Vec<Segment>,
    corners: Vec<Corner>,
    kernel: Arc<Kernel>,
    continuous: Option<Arc<ContinuousHistory>>,
    limit: usize,
}

/// Prepared replacement of accepted coupled-line state.
#[derive(Debug, Clone, PartialEq)]
pub struct CplUpdate {
    reset: bool,
    next: CplHistory,
    breakpoints: Vec<Real>,
}

impl CplUpdate {
    pub(crate) const fn is_reset(&self) -> bool {
        self.reset
    }
    pub(crate) fn into_breakpoints(self) -> Vec<Real> {
        self.breakpoints
    }
}

impl CplHistory {
    /// Number of retained boundary and response records.
    #[must_use]
    pub fn len(&self) -> usize {
        usize::from(self.time.is_some())
            + self.segments.len()
            + self.corners.len()
            + self.kernel.records
            + self.kernel.pending.len()
            + self.continuous.as_deref().map_or(0, ContinuousHistory::len)
    }

    /// Whether the accepted graph has been initialized.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.time.is_none()
    }

    pub(crate) fn clear(&mut self) {
        self.time = None;
        self.previous_time = None;
        self.segments.clear();
        self.corners.clear();
        self.kernel = Arc::new(Kernel::empty(self.input.len()));
        if let Some(history) = &mut self.continuous {
            Arc::make_mut(history).clear();
        }
        self.initial_output.fill(Wide::ZERO);
        for values in [
            &mut self.input,
            &mut self.input_error,
            &mut self.previous_input,
            &mut self.previous_error,
        ] {
            values.fill(0.);
        }
    }

    fn boundary_bytes(&self) -> usize {
        let ports = self.input.len();
        let segment_bytes =
            std::mem::size_of::<Segment>() + 16 + ports * std::mem::size_of::<Real>();
        let corner_bytes =
            std::mem::size_of::<Corner>() + 16 + ports * std::mem::size_of::<(usize, Real)>();
        self.segments
            .capacity()
            .saturating_add(1)
            .saturating_mul(segment_bytes)
            .saturating_add(
                self.corners
                    .capacity()
                    .saturating_add(1)
                    .saturating_mul(corner_bytes),
            )
            .saturating_add(ports * (4 * std::mem::size_of::<Real>() + std::mem::size_of::<Wide>()))
            .saturating_add(
                self.continuous
                    .as_deref()
                    .map_or(0, ContinuousHistory::workspace),
            )
    }

    pub(crate) fn check(&self, update: &CplUpdate) -> SpiceResult<()> {
        let next = &update.next;
        let time = next
            .time
            .ok_or_else(|| invalid("uninitialized CPL update"))?;
        let width = self.input.len();
        if !time.is_finite() || (!update.reset && self.time.is_none_or(|old| time <= old)) {
            return Err(invalid(format!(
                "invalid CPL accepted time {time:e}, previous {:?}",
                self.time
            )));
        }
        let (bytes, limit) = if let (Some(old), Some(new)) = (&self.continuous, &next.continuous) {
            // The transactional update shares unchanged immutable wave blocks.
            // Count the simultaneous versions, not two independent copies.
            let bytes = self
                .boundary_bytes()
                .saturating_add(next.boundary_bytes().saturating_sub(new.workspace()))
                .saturating_add(new.unique_workspace(old))
                .saturating_add(self.kernel.workspace(width))
                .saturating_add(next.kernel.workspace(width));
            (bytes, self.limit.saturating_mul(2))
        } else {
            (
                next.boundary_bytes()
                    .saturating_add(next.kernel.workspace(width)),
                self.limit,
            )
        };
        if next.limit != self.limit || bytes > limit {
            return Err(invalid(format!(
                "CPL accepted history budget: {bytes} estimated bytes, limit {limit} bytes, {} boundary segments, {} response knots, {} arrivals at {time:e}",
                next.segments.len(),
                next.kernel.records,
                next.kernel.pending.len()
            )));
        }
        for (name, values) in [
            ("input", &next.input),
            ("input uncertainty", &next.input_error),
            ("previous input", &next.previous_input),
            ("previous uncertainty", &next.previous_error),
        ] {
            if values.len() != width || values.iter().any(|value| !value.is_finite()) {
                return Err(invalid(format!(
                    "nonfinite or mismatched CPL {name}: {values:?}, width {width}, time {time:e}, previous {:?}",
                    self.time
                )));
            }
        }
        if next.initial_output.len() != width
            || next
                .initial_output
                .iter()
                .any(|value| !value.hi.is_finite())
        {
            return Err(invalid("nonfinite or mismatched CPL initial output"));
        }
        if next.segments.last().is_some_and(|segment| {
            segment.start >= segment.end
                || segment.end > time
                || segment.delta.len() != width
                || segment.delta.iter().any(|value| !value.is_finite())
        }) {
            return Err(invalid("invalid CPL boundary segment"));
        }
        if update
            .breakpoints
            .iter()
            .any(|point| !point.is_finite() || *point <= time)
        {
            return Err(invalid("invalid CPL future breakpoint"));
        }
        Ok(())
    }

    pub(crate) fn apply_validated(&mut self, update: &CplUpdate) {
        self.clone_from(&update.next);
    }
}

#[derive(Debug, Clone, Copy)]
struct Edge {
    target: usize,
    delay: Real,
    gain: Real,
}

#[derive(Debug, Clone)]
pub(super) struct LossyPlan {
    basis: ModalBasis,
    scattering: Vec<Vec<(usize, Real)>>,
    half: Vec<Real>,
    launch: Vec<Edge>,
    fronts: Vec<Vec<Real>>,
    reachable: Vec<Vec<usize>>,
    first_at_node: Vec<Vec<Real>>,
    // A certified absolute terminal path mass bound, if one exists.
    path_bound: Option<Vec<Real>>,
    roundoff_budget: Vec<Real>,
    continuous: Option<ContinuousPlan>,
    sections: usize,
    terminal: usize,
    limit: usize,
}

impl LossyPlan {
    pub(super) fn prepare(
        basis: ModalBasis,
        plan: &CoupledLinePlan,
        sections: usize,
    ) -> SpiceResult<Self> {
        let n = plan.conductors();
        let ports = 2 * n;
        let terminal = sections * ports;
        let count = Real::from(u32::try_from(sections).map_err(|_| invalid("section count"))?);
        let half: Vec<_> = basis
            .delays
            .iter()
            .map(|delay| delay / (2. * count))
            .collect();
        if half.iter().any(|delay| !delay.is_finite() || *delay <= 0.) {
            return Err(invalid("unrepresentable modal section delay"));
        }
        let matrix = modal_static_scattering(plan, &basis, sections)?;
        let scattering = (0..ports)
            .map(|input| {
                (0..ports)
                    .filter_map(|output| {
                        let gain = matrix[output * ports + input];
                        (gain != 0.).then_some((output, gain))
                    })
                    .collect()
            })
            .collect();
        // Both the trial and accepted response can coexist. Account for
        // the queue entry, vector, ordered-map overhead and prefix knots.
        let limit = plan.workspace_limit() / 2;
        let launch = (0..ports)
            .map(|port| Edge {
                target: if port < n {
                    port
                } else {
                    (sections - 1) * ports + port
                },
                delay: half[port % n],
                gain: 1.,
            })
            .collect();
        let mut result = Self {
            basis,
            scattering,
            half,
            launch,
            fronts: Vec::new(),
            reachable: Vec::new(),
            first_at_node: Vec::new(),
            path_bound: None,
            roundoff_budget: Vec::new(),
            continuous: None,
            sections,
            terminal,
            limit,
        };
        result.first_at_node = (0..ports).map(|port| result.first_arrivals(port)).collect();
        result.fronts = result
            .first_at_node
            .iter()
            .map(|distances| distances[result.terminal..].to_vec())
            .collect();
        result.reachable = result
            .fronts
            .iter()
            .map(|distances| {
                distances
                    .iter()
                    .enumerate()
                    .filter_map(|(row, delay)| delay.is_finite().then_some(row))
                    .collect()
            })
            .collect();
        result.continuous = ContinuousPlan::prepare(&result)?;
        if result.continuous.is_none() {
            result.prepare_path_budget();
        } else {
            result.first_at_node.clear();
            result.roundoff_budget = vec![0.; ports];
        }
        Ok(result)
    }

    fn prepare_path_budget(&mut self) {
        self.path_bound = self.absolute_path_bound();
        let ports = self.launch.len();
        self.roundoff_budget = if self.path_bound.is_some() {
            (0..ports)
                .map(|port| self.source_roundoff_budget(port))
                .collect()
        } else {
            self.first_at_node.clear();
            vec![0.; ports]
        };
    }

    pub(super) const fn shortest(&self) -> Real {
        match &self.continuous {
            Some(plan) => plan.step(),
            None => self.basis.shortest,
        }
    }

    // Every section shares the same matrix. Store it once and derive the
    // neighbouring stage index from the outgoing mode and direction.
    fn outgoing(&self, node: usize) -> impl Iterator<Item = Edge> + '_ {
        let ports = self.launch.len();
        let section = node / ports;
        let n = self.half.len();
        self.scattering[node % ports]
            .iter()
            .map(move |&(output, gain)| {
                let mode = output % n;
                let (target, delay) = if output < n {
                    if section == 0 {
                        (self.terminal + output, self.half[mode])
                    } else {
                        ((section - 1) * ports + n + mode, 2. * self.half[mode])
                    }
                } else if section + 1 == self.sections {
                    (self.terminal + output, self.half[mode])
                } else {
                    ((section + 1) * ports + mode, 2. * self.half[mode])
                };
                Edge {
                    target,
                    delay,
                    gain,
                }
            })
    }

    // First physical arrival for every reachable terminal, independent of
    // the numerical integration step or which input waves happen to be zero.
    fn first_arrivals(&self, port: usize) -> Vec<Real> {
        let ports = self.launch.len();
        let mut distances = vec![Real::INFINITY; self.terminal + ports];
        let mut queue = BinaryHeap::new();
        let launch = self.launch[port];
        distances[launch.target] = launch.delay;
        queue.push(Reverse(Arrival {
            time: launch.delay,
            node: launch.target,
        }));
        while let Some(Reverse(event)) = queue.pop() {
            if event.time > distances[event.node] || event.node >= self.terminal {
                continue;
            }
            for edge in self.outgoing(event.node) {
                let time = event.time + edge.delay;
                if time < distances[edge.target] {
                    distances[edge.target] = time;
                    queue.push(Reverse(Arrival {
                        time,
                        node: edge.target,
                    }));
                }
            }
        }
        distances
    }

    // Upper bound on the sum of absolute path gains from each internal
    // incident port to *all* external terminals. Use two scalar directional
    // envelopes per stage, independent of the number of modes. All
    // products/sums round upwards when verifying the supersolution.
    // Some passive networks rely on signed cancellation and have no
    // finite absolute-path envelope; those retain the exact graph.
    fn absolute_path_bound(&self) -> Option<Vec<Real>> {
        let n = self.half.len();
        let ports = self.launch.len();
        let mut reflection: Real = 0.;
        let mut transmission: Real = 0.;
        for (input, column) in self.scattering.iter().enumerate() {
            let mut reflected: Real = 0.;
            let mut transmitted: Real = 0.;
            for &(output, gain) in column {
                let total = if input / n == output / n {
                    &mut reflected
                } else {
                    &mut transmitted
                };
                *total = (*total + gain.abs()).next_up();
            }
            reflection = reflection.max(reflected);
            transmission = transmission.max(transmitted);
        }
        let mut weights = vec![1.; 2 * self.sections];
        let mut sums = weights.clone();
        for _ in 0..(8 * self.sections + 64) {
            for stage in 0..self.sections {
                let left = if stage == 0 {
                    1.
                } else {
                    weights[2 * (stage - 1) + 1]
                };
                let right = if stage + 1 == self.sections {
                    1.
                } else {
                    weights[2 * (stage + 1)]
                };
                sums[2 * stage] =
                    ((reflection * left).next_up() + (transmission * right).next_up()).next_up();
                sums[2 * stage + 1] =
                    ((transmission * left).next_up() + (reflection * right).next_up()).next_up();
            }
            if sums.iter().zip(&weights).all(|(sum, weight)| sum <= weight) {
                return Some(
                    (0..self.terminal + ports)
                        .map(|node| {
                            if node >= self.terminal {
                                1.
                            } else {
                                weights[2 * (node / ports) + (node % ports) / n]
                            }
                        })
                        .collect(),
                );
            }
            if sums.iter().any(|sum| !sum.is_finite()) {
                return None;
            }
            for (weight, sum) in weights.iter_mut().zip(&sums) {
                // Slack is used only to find a tight supersolution; the
                // explicit outward-rounded inequality above certifies it.
                *weight = weight.max(1.000_001_f64.mul_add(*sum, 1e-6).next_up());
            }
        }
        None
    }

    // Scale the aggregate rounding allowance against the weakest reachable
    // terminal's strongest path, not the largest through coefficient.
    // This protects weak but real crosstalk between strongly disparate modes.
    fn source_roundoff_budget(&self, source: usize) -> Real {
        let ports = self.launch.len();
        let mut best = vec![0.; self.terminal + ports];
        let mut queue = BinaryHeap::new();
        let launch = self.launch[source];
        best[launch.target] = 1.;
        queue.push(Arrival {
            time: 1.,
            node: launch.target,
        });
        while let Some(event) = queue.pop() {
            if event.time < best[event.node] || event.node >= self.terminal {
                continue;
            }
            for edge in self.outgoing(event.node) {
                let gain = (event.time * edge.gain.abs().min(1.)).next_down().max(0.);
                if gain > best[edge.target] {
                    best[edge.target] = gain;
                    queue.push(Arrival {
                        time: gain,
                        node: edge.target,
                    });
                }
            }
        }
        let minimum = self.reachable[source]
            .iter()
            .map(|&row| best[self.terminal + row])
            .fold(1., Real::min);
        (minimum * (64. * Real::EPSILON)).next_down().max(0.)
    }

    fn prune_roundoff(&self, budget: &mut [Real], key: Arrival, values: &mut [Real]) {
        let Some(weights) = &self.path_bound else {
            return;
        };
        let weight = weights[key.node];
        for (source, value) in values.iter_mut().enumerate() {
            if *value == 0. {
                continue;
            }
            // Retain the earliest arrival at every node (hence every
            // shortest terminal path), allowing only arithmetic roundoff
            // in the sum of its positive physical delays.
            let first = self.first_at_node[source][key.node];
            let path_roundoff = (8. * Real::EPSILON).mul_add(
                Real::from(u32::try_from(self.terminal + self.launch.len()).unwrap_or(u32::MAX)),
                1.,
            );
            if key.time <= first * path_roundoff {
                continue;
            }
            let cost = (value.abs() * weight).next_up();
            let remaining = &mut budget[source];
            if cost <= *remaining {
                *remaining = (*remaining - cost).next_down().max(0.);
                *value = 0.;
            }
        }
    }

    pub(super) fn new_history(&self) -> CplHistory {
        let zeros = vec![0.; self.launch.len()];
        CplHistory {
            time: None,
            input: zeros.clone(),
            input_error: zeros.clone(),
            previous_input: zeros.clone(),
            previous_error: zeros.clone(),
            previous_time: None,
            initial_output: vec![Wide::ZERO; zeros.len()],
            segments: Vec::new(),
            corners: Vec::new(),
            kernel: Arc::new(Kernel::empty(self.launch.len())),
            continuous: self
                .continuous
                .as_ref()
                .map(|plan| Arc::new(plan.empty(self.launch.len()))),
            limit: self.limit,
        }
    }

    pub(super) fn start(
        &self,
        line: &CoupledLine,
        context: &DelayContext<'_>,
    ) -> SpiceResult<CplUpdate> {
        let mut next = self.new_history();
        next.time = Some(context.time);
        if !context.initial_conditions {
            let incident = self.basis.physical_waves(line, context, 1.)?;
            let reflected = self.basis.physical_waves(line, context, -1.)?;
            let ports = self.launch.len();
            next.input.copy_from_slice(&incident[..ports]);
            next.input_error.copy_from_slice(&incident[ports..]);
            for (output, value) in next.initial_output.iter_mut().zip(&reflected[..ports]) {
                *output = Wide::from_real(*value);
            }
        }
        if let Some(plan) = &self.continuous {
            next.continuous = Some(Arc::new(plan.start(
                context.time,
                context.reltol,
                &next.input,
            )));
        }
        // All delays are unity at DC, so the static distributed R/G
        // solution also initializes the exact split network's boundary.
        Ok(CplUpdate {
            reset: true,
            next,
            breakpoints: Vec::new(),
        })
    }

    fn send(
        &self,
        kernel: &mut Kernel,
        time: Wide,
        edge: Edge,
        values: &[Real],
    ) -> SpiceResult<()> {
        let arrival = time + Wide::from_real(edge.delay);
        if !arrival.hi.is_finite() || arrival.hi <= time.hi {
            return Err(invalid(
                "modal delay smaller than representable simulation time",
            ));
        }
        let key = Arrival {
            time: arrival.hi,
            node: edge.target,
        };
        if let Some(event) = kernel.pending.get_mut(&key) {
            for (target, value) in event.values.iter_mut().zip(values) {
                *target = edge.gain.mul_add(*value, *target);
            }
            return Ok(());
        }
        let mut incoming: Vec<_> = values.iter().map(|value| edge.gain * value).collect();
        self.prune_roundoff(&mut kernel.remaining_roundoff, key, &mut incoming);
        if incoming.iter().all(|value| *value == 0.) {
            return Ok(());
        }
        if kernel.pending.len() >= self.limit / Kernel::arrival_bytes(values.len()) {
            return Err(invalid("CPL arrival workspace exceeded"));
        }
        kernel.pending.insert(
            key,
            Increment {
                time: arrival,
                values: incoming,
            },
        );
        Ok(())
    }

    fn extend_kernel(
        &self,
        kernel: &mut Kernel,
        horizon: Real,
        boundary_bytes: usize,
    ) -> SpiceResult<()> {
        let ports = self.launch.len();
        if kernel.horizon == 0. {
            kernel
                .remaining_roundoff
                .copy_from_slice(&self.roundoff_budget);
            for (port, edge) in self.launch.iter().enumerate() {
                let mut unit = vec![0.; ports];
                unit[port] = 1.;
                self.send(kernel, Wide::ZERO, *edge, &unit)?;
            }
        }
        let mut work = 0_usize;
        kernel.trace_capacity = kernel.traces.iter().map(Vec::capacity).sum();
        while kernel
            .pending
            .first_key_value()
            .is_some_and(|(key, _)| key.time <= horizon)
        {
            let (key, mut event) = kernel
                .pending
                .pop_first()
                .ok_or_else(|| invalid("missing CPL arrival"))?;
            self.prune_roundoff(&mut kernel.remaining_roundoff, key, &mut event.values);
            if event.values.iter().all(|value| *value == 0.) {
                continue;
            }
            work += 1;
            if work > (self.limit / std::mem::size_of::<Knot>()).saturating_mul(8) {
                return Err(invalid("CPL arrival work budget exceeded"));
            }
            if key.node >= self.terminal {
                kernel.push(key.node - self.terminal, event.time, &event.values)?;
            } else {
                for edge in self.outgoing(key.node) {
                    self.send(kernel, event.time, edge, &event.values)?;
                }
            }
            let bytes = kernel.workspace(ports).saturating_add(boundary_bytes);
            if bytes > self.limit {
                return Err(invalid(format!(
                    "CPL response workspace exceeded: {bytes} estimated bytes, {} knots, {} pending, {boundary_bytes} boundary bytes, {work} work at relative time {:e}, limit {} bytes",
                    kernel.records,
                    kernel.pending.len(),
                    key.time,
                    self.limit
                )));
            }
        }
        if kernel.workspace(ports).saturating_add(boundary_bytes) > self.limit {
            return Err(invalid("CPL initial response workspace exceeded"));
        }
        kernel.horizon = horizon;
        Ok(())
    }

    fn kernel_at(&self, history: &CplHistory, time: Real) -> SpiceResult<Arc<Kernel>> {
        let start = history
            .time
            .ok_or_else(|| invalid("missing accepted CPL boundary"))?;
        let horizon = time
            - history
                .segments
                .first()
                .map_or(start, |segment| segment.start);
        if !horizon.is_finite() || time <= start {
            return Err(invalid("nonpositive CPL trial duration"));
        }
        let mut kernel = Arc::clone(&history.kernel);
        if horizon > kernel.horizon
            && (kernel.horizon == 0.
                || kernel
                    .pending
                    .first_key_value()
                    .is_some_and(|(key, _)| key.time <= horizon))
        {
            self.extend_kernel(
                Arc::make_mut(&mut kernel),
                horizon,
                history.boundary_bytes(),
            )?;
        }
        Ok(kernel)
    }

    fn known_output(&self, history: &CplHistory, kernel: &Kernel, time: Real) -> Vec<Wide> {
        let ports = self.launch.len();
        let mut known = history.initial_output.clone();
        for segment in &history.segments {
            for (source, &delta) in segment.delta.iter().enumerate() {
                if delta == 0. {
                    continue;
                }
                let delta = Wide::from_real(delta);
                for &row in &self.reachable[source] {
                    let gain = segment_response(
                        &kernel.traces[row * ports + source],
                        time,
                        segment.start,
                        segment.end,
                    );
                    known[row] = known[row] + delta * gain;
                }
            }
        }
        known
    }

    pub(super) fn stamp(
        &self,
        line: &CoupledLine,
        context: &mut StampContext<'_>,
        time: Real,
    ) -> SpiceResult<()> {
        let history = context
            .states
            .delay_history()
            .ok_or_else(|| invalid("missing CPL graph history"))?
            .as_cpl()?;
        let start = history
            .time
            .ok_or_else(|| invalid("missing CPL boundary"))?;
        let ports = self.launch.len();
        let (known, coefficients) =
            if let (Some(plan), Some(state)) = (&self.continuous, &history.continuous) {
                let output = plan.outputs(
                    self,
                    state,
                    start,
                    time,
                    history
                        .boundary_bytes()
                        .saturating_add(history.kernel.workspace(ports)),
                )?;
                let known = history
                    .initial_output
                    .iter()
                    .zip(&output)
                    .map(|(&baseline, wave)| baseline + Wide::from_real(wave.known))
                    .collect::<Vec<_>>();
                let coefficients = output
                    .iter()
                    .map(|wave| {
                        (0..ports)
                            .map(|source| Wide::from_real(wave.component(source + 1)))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                (known, coefficients)
            } else {
                let kernel = self.kernel_at(history, time)?;
                let known = self.known_output(history, &kernel, time);
                let coefficients = kernel
                    .traces
                    .chunks_exact(ports)
                    .map(|row| {
                        row.iter()
                            .map(|trace| segment_response(trace, time, start, time))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                (known, coefficients)
            };
        let first = context.branch(0)?;
        for (row, col, value) in line.terminal_kcl(context.unknowns, first)? {
            context.matrix.add(row, col, value.re)?;
        }
        let n = self.basis.delays.len();
        for (row, (&constant, coefficients)) in known.iter().zip(&coefficients).enumerate() {
            let mut rhs = constant;
            for (&gain, &input) in coefficients.iter().zip(&history.input) {
                rhs = rhs - gain * Wide::from_real(input);
            }
            if !rhs.hi.is_finite() || coefficients.iter().any(|value| !value.hi.is_finite()) {
                return Err(invalid("nonfinite CPL endpoint relation"));
            }
            context.rhs.add_to(first + row, rhs.hi)?;
            for physical in 0..ports {
                let (node, ret) = line.physical_pair(physical);
                let mut voltage = 0.;
                let mut current = 0.;
                for mode in 0..n {
                    let port = (physical / n) * n + mode;
                    let index = (physical % n) * n + mode;
                    let delta = if row == port { 1. } else { 0. };
                    let gain = coefficients[port].hi;
                    voltage = self.basis.current_basis[index].mul_add(delta - gain, voltage);
                    current = self.basis.voltage_basis[index].mul_add(-delta - gain, current);
                }
                for (column, value) in [
                    (context.unknowns.checked_node_row(node)?, voltage),
                    (context.unknowns.checked_node_row(ret)?, -voltage),
                    (Some(first + physical), current),
                ] {
                    if let Some(column) = column
                        && value != 0.
                    {
                        context.matrix.add(first + row, column, value)?;
                    }
                }
            }
        }
        Ok(())
    }

    // A finished response has finite support (including distortionless
    // lossy modes). Once a whole segment has crossed that support its
    // contribution is constant; fold it into the baseline exactly.
    fn prune_finished_response(history: &mut CplHistory, time: Real) {
        if history.kernel.horizon == 0. || !history.kernel.pending.is_empty() {
            return;
        }
        let tail = history
            .kernel
            .traces
            .iter()
            .filter_map(|trace| trace.last())
            .map(|knot| knot.time.hi)
            .fold(0., Real::max);
        let count = history
            .segments
            .partition_point(|segment| time - segment.end >= tail);
        let ports = history.input.len();
        for segment in history.segments.drain(..count) {
            for (row, target) in history.initial_output.iter_mut().enumerate() {
                for (source, &delta) in segment.delta.iter().enumerate() {
                    if let Some(knot) = history.kernel.traces[row * ports + source].last() {
                        *target = *target + Wide::from_real(delta) * knot.cumulative;
                    }
                }
            }
        }
        let corners = history
            .corners
            .partition_point(|corner| time - corner.time >= tail);
        history.corners.drain(..corners);
    }

    pub(super) fn accept(
        &self,
        line: &CoupledLine,
        history: &CplHistory,
        context: &DelayContext<'_>,
    ) -> SpiceResult<CplUpdate> {
        let start = history
            .time
            .ok_or_else(|| invalid("missing accepted CPL boundary"))?;
        let dt = context.time - start;
        let incident = self.basis.physical_waves(line, context, 1.)?;
        let reflected = self.basis.physical_waves(line, context, -1.)?;
        let ports = self.launch.len();
        let mut next = history.clone();
        let direct = if let (Some(plan), Some(state)) = (&self.continuous, &history.continuous) {
            Some((plan, state))
        } else {
            next.kernel = self.kernel_at(history, context.time)?;
            None
        };
        let mut segment = Segment {
            start,
            end: context.time,
            delta: vec![0.; ports],
        };
        let mut breakpoints = Vec::new();
        let mut corners = Vec::new();
        for port in 0..ports {
            let delta = incident[port] - history.input[port];
            segment.delta[port] = delta;
            let (change, uncertainty) = history.previous_time.map_or(
                (delta, history.input_error[port] + incident[ports + port]),
                |previous_time| {
                    let ratio = dt / (start - previous_time);
                    let predicted = ratio * (history.input[port] - history.previous_input[port]);
                    let error = ratio.mul_add(
                        history.previous_error[port] + history.input_error[port],
                        history.input_error[port] + incident[ports + port],
                    );
                    (delta - predicted, error)
                },
            );
            // Roundoff/tolerance affects only scheduling. The convolution
            // always retains the actual accepted endpoint, even for small
            // waves, so uncertainty cannot accumulate into extrapolation.
            if change.abs() > uncertainty
                && ((direct.is_none() && history.segments.is_empty())
                    || history.previous_time.is_none()
                    || change.abs() > context.reltol.mul_add(incident[port].abs(), context.abstol))
            {
                corners.push((port, change));
                for delay in self.fronts[port].iter().copied().chain(
                    direct
                        .as_ref()
                        .map(|_| self.basis.delays[port % self.half.len()]),
                ) {
                    let point = start + delay;
                    if point.is_finite() && point > context.time {
                        breakpoints.push(point);
                    }
                }
            }
            next.previous_input[port] = history.input[port];
            next.previous_error[port] = history.input_error[port];
            next.input[port] = incident[port];
            next.input_error[port] = incident[ports + port];
        }
        next.previous_time = Some(start);
        next.time = Some(context.time);
        if let Some((plan, state)) = direct {
            next.continuous = Some(Arc::new(
                plan.advance(
                    self,
                    state,
                    start,
                    context.time,
                    history
                        .boundary_bytes()
                        .saturating_add(history.kernel.workspace(ports)),
                    (&incident[..ports], &segment.delta),
                )?,
            ));
            return Ok(CplUpdate {
                reset: false,
                next,
                breakpoints,
            });
        }
        if segment.delta.iter().any(|value| *value != 0.) {
            next.segments.push(segment);
        }
        if !corners.is_empty() {
            next.corners.push(Corner {
                time: start,
                duration: dt,
                changes: corners,
            });
        }
        Self::prune_finished_response(&mut next, context.time);
        self.secondary_breakpoints(&next, &reflected[..ports], context, dt, &mut breakpoints);
        Ok(CplUpdate {
            reset: false,
            next,
            breakpoints,
        })
    }
    fn secondary_breakpoints(
        &self,
        history: &CplHistory,
        reflected: &[Real],
        context: &DelayContext<'_>,
        dt: Real,
        breakpoints: &mut Vec<Real>,
    ) {
        let ports = self.launch.len();
        // Schedule the next resolvable secondary corner for each output,
        // not every tiny reflection. All arrivals remain in the response.
        for (row, &amplitude) in reflected.iter().enumerate() {
            let mut earliest = Real::INFINITY;
            let tolerance = context.reltol.mul_add(amplitude.abs(), context.abstol);
            for corner in &history.corners {
                let ratio = dt / corner.duration;
                let lag = context.time - corner.time;
                for &(source, change) in &corner.changes {
                    let trace = &history.kernel.traces[row * ports + source];
                    let index = trace.partition_point(|knot| knot.time.hi <= lag);
                    for knot in &trace[index..] {
                        let point = corner.time + knot.time.hi;
                        if point >= earliest {
                            break;
                        }
                        if point > context.time && (knot.impulse * change * ratio).abs() > tolerance
                        {
                            earliest = point;
                            break;
                        }
                    }
                }
            }
            if earliest.is_finite() {
                breakpoints.push(earliest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::cpl::CoupledLineLimits;

    #[test]
    fn bounded_path_roundoff_preserves_first_arrivals_and_small_transfer_coefficients() {
        let material = CoupledLinePlan::from_packed(
            2,
            1.,
            &[0.05, 0.001, 0.08],
            &[250e-9, 0., 400e-9],
            &[0.000001, 0., 0.000002],
            &[100e-12, 0., 160e-12],
            CoupledLineLimits::default(),
        )
        .unwrap();
        let basis = ModalBasis::prepare(&material).unwrap().unwrap();
        let mut plan = LossyPlan::prepare(basis, &material, 4).unwrap();
        // Exercise the exact path compiler independently of the selected
        // continuous production representation.
        plan.first_at_node = (0..4).map(|source| plan.first_arrivals(source)).collect();
        plan.prepare_path_budget();
        let bound = plan.path_bound.as_ref().unwrap();
        for node in 0..plan.terminal {
            let sum = plan
                .outgoing(node)
                .map(|edge| edge.gain.abs() * bound[edge.target])
                .sum::<Real>();
            assert!(sum <= bound[node], "absolute path bound at node {node}");
        }
        let mut exact = plan.clone();
        exact.path_bound = None;
        let mut reference = Kernel::empty(4);
        exact.extend_kernel(&mut reference, 30e-9, 0).unwrap();
        let mut bounded = Kernel::empty(4);
        plan.extend_kernel(&mut bounded, 30e-9, 0).unwrap();
        assert!(bounded.records < reference.records);
        for (index, (full, reduced)) in reference.traces.iter().zip(&bounded.traces).enumerate() {
            let source = index % 4;
            assert_eq!(full.first(), reduced.first());
            let mut i = 0;
            let mut j = 0;
            let mut omitted = 0.;
            while i < full.len() || j < reduced.len() {
                let a = full.get(i);
                let b = reduced.get(j);
                match (a, b) {
                    (Some(a), Some(b)) if a.time.hi == b.time.hi => {
                        omitted += (a.impulse - b.impulse).abs();
                        i += 1;
                        j += 1;
                    }
                    (Some(a), Some(b)) if a.time.hi < b.time.hi => {
                        omitted += a.impulse.abs();
                        i += 1;
                    }
                    (Some(a), None) => {
                        omitted += a.impulse.abs();
                        i += 1;
                    }
                    (_, Some(b)) => {
                        omitted += b.impulse.abs();
                        j += 1;
                    }
                    (None, None) => break,
                }
            }
            let used = plan.roundoff_budget[source] - bounded.remaining_roundoff[source];
            // Distinguish the certified discarded-path mass from floating
            // arithmetic in the two independently accumulated kernels.
            let arithmetic =
                full.iter().map(|knot| knot.impulse.abs()).sum::<Real>() * 64. * Real::EPSILON;
            assert!(
                omitted <= used + arithmetic,
                "trace {index}: {omitted:e} > {used:e} + {arithmetic:e}"
            );
        }
    }
}
