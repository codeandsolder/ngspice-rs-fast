//! Passive split-generator transport for positive-definite L and C.
//!
//! Every static R/G stage is a reciprocal dissipative scattering relation.
//! Exact modal half-delays surround the stages (Strang splitting). Linear
//! scattering commutes with convolution, so compile the boundary impulse
//! response once, incrementally, rather than propagating every input knot
//! through the same internal paths. Its integral is a continuous piecewise
//! linear ramp response. Convolution uses finite accepted PWL segments,
//! not extrapolated slopes or cancellation of unbounded ramps. Only
//! accepted solutions commit the response frontier and new segments.
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
}

impl Kernel {
    fn empty(ports: usize) -> Self {
        Self {
            horizon: 0.,
            pending: BTreeMap::new(),
            traces: vec![Vec::new(); ports * ports],
            records: 0,
        }
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
            trace.push(Knot {
                time,
                cumulative,
                impulse,
            });
            self.records += 1;
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
        if next.limit != self.limit || next.len() > self.limit {
            return Err(invalid(format!(
                "CPL accepted history budget: {} records, limit {}, {} boundary segments, {} response knots, {} arrivals at {time:e}",
                next.len(),
                self.limit,
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
        let limit = plan.workspace_limit() / (2 * (128 + ports * std::mem::size_of::<Real>()));
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
            sections,
            terminal,
            limit,
        };
        result.fronts = (0..ports).map(|port| result.first_arrivals(port)).collect();
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
        Ok(result)
    }

    pub(super) const fn shortest(&self) -> Real {
        self.basis.shortest
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
        distances.split_off(self.terminal)
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
        // All delays are unity at DC, so the static distributed R/G
        // solution also initializes the exact split network's boundary.
        Ok(CplUpdate {
            reset: true,
            next,
            breakpoints: Vec::new(),
        })
    }

    fn send(&self, queue: &mut Queue, time: Wide, edge: Edge, values: &[Real]) -> SpiceResult<()> {
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
        if queue.len() >= self.limit && !queue.contains_key(&key) {
            return Err(invalid("CPL arrival workspace exceeded"));
        }
        let event = queue.entry(key).or_insert_with(|| Increment {
            time: arrival,
            values: vec![0.; values.len()],
        });
        for (target, value) in event.values.iter_mut().zip(values) {
            *target = edge.gain.mul_add(*value, *target);
        }
        Ok(())
    }

    fn extend_kernel(
        &self,
        kernel: &mut Kernel,
        horizon: Real,
        boundary_records: usize,
    ) -> SpiceResult<()> {
        let ports = self.launch.len();
        if kernel.horizon == 0. {
            for (port, edge) in self.launch.iter().enumerate() {
                let mut unit = vec![0.; ports];
                unit[port] = 1.;
                self.send(&mut kernel.pending, Wide::ZERO, *edge, &unit)?;
            }
        }
        let mut work = 0_usize;
        while kernel
            .pending
            .first_key_value()
            .is_some_and(|(key, _)| key.time <= horizon)
        {
            let (key, event) = kernel
                .pending
                .pop_first()
                .ok_or_else(|| invalid("missing CPL arrival"))?;
            if event.values.iter().all(|value| *value == 0.) {
                continue;
            }
            work += 1;
            if work > self.limit.saturating_mul(8) {
                return Err(invalid("CPL arrival work budget exceeded"));
            }
            if key.node >= self.terminal {
                kernel.push(key.node - self.terminal, event.time, &event.values)?;
            } else {
                for edge in self.outgoing(key.node) {
                    self.send(&mut kernel.pending, event.time, edge, &event.values)?;
                }
            }
            if kernel.records + kernel.pending.len() + boundary_records > self.limit {
                return Err(invalid("CPL response workspace exceeded"));
            }
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
        if horizon > kernel.horizon && (kernel.horizon == 0. || !kernel.pending.is_empty()) {
            self.extend_kernel(
                Arc::make_mut(&mut kernel),
                horizon,
                history.segments.len() + history.corners.len() + 3,
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
        let kernel = self.kernel_at(history, time)?;
        let known = self.known_output(history, &kernel, time);
        let first = context.branch(0)?;
        for (row, col, value) in line.terminal_kcl(context.unknowns, first)? {
            context.matrix.add(row, col, value.re)?;
        }
        let n = self.basis.delays.len();
        let ports = self.launch.len();
        for (row, &constant) in known.iter().enumerate() {
            let mut rhs = constant;
            let mut coefficients = Vec::with_capacity(ports);
            for source in 0..ports {
                let gain =
                    segment_response(&kernel.traces[row * ports + source], time, start, time);
                rhs = rhs - gain * Wide::from_real(history.input[source]);
                coefficients.push(gain.hi);
            }
            if !rhs.hi.is_finite() || coefficients.iter().any(|value| !value.is_finite()) {
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
                    let gain = coefficients[port];
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
        next.kernel = self.kernel_at(history, context.time)?;
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
                && (history.segments.is_empty()
                    || change.abs() > context.abstol + context.reltol * incident[port].abs())
            {
                corners.push((port, change));
                for &delay in &self.fronts[port] {
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
        next.previous_time = Some(start);
        next.time = Some(context.time);
        Self::prune_finished_response(&mut next, context.time);
        // Schedule the next resolvable secondary corner for each output,
        // not every tiny reflection. All arrivals remain in the response.
        for (row, &amplitude) in reflected[..ports].iter().enumerate() {
            let mut earliest = Real::INFINITY;
            let tolerance = context.abstol + context.reltol * amplitude.abs();
            for corner in &next.corners {
                let ratio = dt / corner.duration;
                let lag = context.time - corner.time;
                for &(source, change) in &corner.changes {
                    let trace = &next.kernel.traces[row * ports + source];
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
        Ok(CplUpdate {
            reset: false,
            next,
            breakpoints,
        })
    }
}
