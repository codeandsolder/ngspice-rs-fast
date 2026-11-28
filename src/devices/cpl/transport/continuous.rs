//! Continuous-time method of steps for coupled, lossy modal transport.
//!
//! Each stage operates on the exact physical delayed PWL incident waves.
//! Histories retain only the portions that can still be consumed by a
//! neighboring stage or terminal. Trial waveforms are affine in the new
//! boundary endpoint, and reduction preserves endpoints, causal fronts
//! and the wave-energy inequality for every affine combination.
mod energy;
mod history;
mod reduce;

use energy::scattering_energy_factor;
use history::{KnownWave, RetainedWindow, WaveRead};

use super::{LossyPlan, Real, SpiceResult, invalid};

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Affine {
    pub(super) known: Real,
    // Empty for a wave independent of the trial endpoint.
    pub(super) gains: Vec<Real>,
}

impl Affine {
    pub(super) const fn zero() -> Self {
        Self::scalar(0.)
    }

    const fn scalar(known: Real) -> Self {
        Self {
            known,
            gains: Vec::new(),
        }
    }

    pub(super) fn component(&self, index: usize) -> Real {
        if index == 0 {
            self.known
        } else {
            self.gains.get(index - 1).copied().unwrap_or(0.)
        }
    }

    fn set(&mut self, index: usize, value: Real, width: usize) {
        if index == 0 {
            self.known = value;
        } else if value != 0. || !self.gains.is_empty() {
            self.gains.resize(width - 1, 0.);
            self.gains[index - 1] = value;
        }
    }

    fn add_scaled(&mut self, wave: &Self, gain: Real) {
        self.known = gain.mul_add(wave.known, self.known);
        if !wave.gains.is_empty() {
            self.gains.resize(wave.gains.len(), 0.);
            for (target, &value) in self.gains.iter_mut().zip(&wave.gains) {
                *target = gain.mul_add(value, *target);
            }
        }
    }

    fn interpolate(first: &Self, last: &Self, fraction: Real) -> Self {
        if fraction <= 0. {
            return first.clone();
        }
        if fraction >= 1. {
            return last.clone();
        }
        let mut result = first.clone();
        result.known = (last.known - first.known).mul_add(fraction, first.known);
        let width = first.gains.len().max(last.gains.len());
        if width != 0 {
            result.gains.resize(width, 0.);
            for (index, value) in result.gains.iter_mut().enumerate() {
                let a = first.component(index + 1);
                let b = last.component(index + 1);
                *value = (b - a).mul_add(fraction, a);
            }
        }
        result
    }

    fn evaluate(&self, delta: &[Real]) -> Real {
        self.gains
            .iter()
            .zip(delta)
            .fold(self.known, |value, (&gain, &delta)| {
                gain.mul_add(delta, value)
            })
    }

    fn finite(&self) -> bool {
        self.known.is_finite() && self.gains.iter().all(|value| value.is_finite())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Point<Value> {
    time: Real,
    wave: Value,
}

type Knot<Value = Affine> = Point<Value>;
type Wave<Value = Affine> = Vec<Knot<Value>>;

/// A PWL value is either a realized scalar or an affine function of all
/// trial endpoints. Ordinary causal accepts need no per-knot coefficient
/// vector or affine storage overhead. Both representations use the same
/// propagation and energy-certified reduction.
trait WaveValue: Clone + PartialEq {
    fn scalar(known: Real) -> Self;
    fn known(&self) -> Real;
    fn width(&self) -> usize;
    fn component(&self, index: usize) -> Real;
    fn set(&mut self, index: usize, value: Real, width: usize);
    fn add_scaled(&mut self, wave: &Self, gain: Real);
    fn interpolate(first: &Self, last: &Self, fraction: Real) -> Self;
    fn boundary(
        known: Real,
        port: usize,
        fraction: Real,
        width: usize,
        delta: Option<&[Real]>,
    ) -> SpiceResult<Self>;
    fn evaluate(&self, delta: &[Real]) -> Real;
    fn finite(&self) -> bool;
    fn allocated_bytes(&self) -> usize;
}

impl WaveValue for Real {
    fn scalar(known: Real) -> Self {
        known
    }
    fn known(&self) -> Real {
        *self
    }
    fn width(&self) -> usize {
        1
    }
    fn component(&self, index: usize) -> Real {
        if index == 0 { *self } else { 0. }
    }
    fn set(&mut self, index: usize, value: Real, _width: usize) {
        debug_assert_eq!(index, 0);
        *self = value;
    }
    fn add_scaled(&mut self, wave: &Self, gain: Real) {
        *self = gain.mul_add(*wave, *self);
    }
    fn interpolate(first: &Self, last: &Self, fraction: Real) -> Self {
        if fraction <= 0. {
            *first
        } else if fraction >= 1. {
            *last
        } else {
            (*last - *first).mul_add(fraction, *first)
        }
    }
    fn boundary(
        known: Real,
        port: usize,
        fraction: Real,
        _width: usize,
        delta: Option<&[Real]>,
    ) -> SpiceResult<Self> {
        let delta = delta
            .and_then(|values| values.get(port))
            .ok_or_else(|| invalid("missing realized CPL endpoint"))?;
        Ok(fraction.mul_add(*delta, known))
    }
    fn evaluate(&self, _delta: &[Real]) -> Real {
        *self
    }
    fn finite(&self) -> bool {
        self.is_finite()
    }
    fn allocated_bytes(&self) -> usize {
        0
    }
}

impl WaveValue for Affine {
    fn scalar(known: Real) -> Self {
        Self::scalar(known)
    }
    fn known(&self) -> Real {
        self.known
    }
    fn width(&self) -> usize {
        self.gains.len() + 1
    }
    fn component(&self, index: usize) -> Real {
        self.component(index)
    }
    fn set(&mut self, index: usize, value: Real, width: usize) {
        self.set(index, value, width);
    }
    fn add_scaled(&mut self, wave: &Self, gain: Real) {
        self.add_scaled(wave, gain);
    }
    fn interpolate(first: &Self, last: &Self, fraction: Real) -> Self {
        Self::interpolate(first, last, fraction)
    }
    fn boundary(
        known: Real,
        port: usize,
        fraction: Real,
        width: usize,
        delta: Option<&[Real]>,
    ) -> SpiceResult<Self> {
        let mut result = Self::scalar(known);
        if let Some(delta) = delta {
            result.known = fraction.mul_add(delta[port], known);
        } else {
            result.set(port + 1, fraction, width);
        }
        Ok(result)
    }
    fn evaluate(&self, delta: &[Real]) -> Real {
        self.evaluate(delta)
    }
    fn finite(&self) -> bool {
        self.finite()
    }
    fn allocated_bytes(&self) -> usize {
        self.gains.capacity() * std::mem::size_of::<Real>()
    }
}

fn wave_bytes<Value: WaveValue>(wave: &Vec<Point<Value>>) -> usize {
    wave.capacity()
        .saturating_mul(std::mem::size_of::<Point<Value>>())
        .saturating_add(
            wave.iter()
                .map(|knot| knot.wave.allocated_bytes())
                .sum::<usize>(),
        )
}

fn histories_bytes<Value: WaveValue>(waves: &[Vec<Point<Value>>]) -> usize {
    waves
        .len()
        .saturating_mul(std::mem::size_of::<Vec<Point<Value>>>())
        .saturating_add(waves.iter().map(wave_bytes).sum::<usize>())
}

fn sample<W: WaveRead + ?Sized>(wave: &W, time: Real, origin: Real) -> SpiceResult<W::Value> {
    if time < origin {
        return Ok(W::Value::scalar(0.));
    }
    let upper = wave.partition_point(|knot| knot.time <= time);
    sample_at(wave, time, upper, &mut 0)
}

fn sample_at<W: WaveRead + ?Sized>(
    wave: &W,
    time: Real,
    upper: usize,
    hint: &mut usize,
) -> SpiceResult<W::Value> {
    let first = wave
        .get_hint(upper.saturating_sub(1), hint)
        .ok_or_else(|| invalid("empty CPL wave history"))?;
    if upper == 0 {
        let roundoff = 8. * Real::EPSILON * first.time.abs().max(time.abs());
        if first.time - time > roundoff {
            return Err(invalid(format!(
                "CPL wave history begins at {:e}, requested {time:e}",
                first.time
            )));
        }
        return Ok(first.wave.clone());
    }
    let Some(last) = wave.get_hint(upper, hint) else {
        let roundoff = 8. * Real::EPSILON * first.time.abs().max(time.abs());
        if time - first.time > roundoff {
            return Err(invalid(format!(
                "CPL wave history ends at {:e}, requested {time:e}",
                first.time
            )));
        }
        return Ok(first.wave.clone());
    };
    let fraction = (time - first.time) / (last.time - first.time);
    Ok(W::Value::interpolate(&first.wave, &last.wave, fraction))
}

// A cursor consumes a monotonically increasing time sequence without
// searching the same immutable history at every scattering knot.
struct WaveCursor<'a, W: WaveRead + ?Sized> {
    wave: &'a W,
    upper: usize,
    hint: usize,
    origin: Real,
}

impl<'a, W: WaveRead + ?Sized> WaveCursor<'a, W> {
    fn new(wave: &'a W, time: Real, origin: Real) -> Self {
        Self {
            wave,
            upper: wave.partition_point(|knot| knot.time <= time),
            hint: 0,
            origin,
        }
    }

    fn sample(&mut self, time: Real) -> SpiceResult<W::Value> {
        if time < self.origin {
            return Ok(W::Value::scalar(0.));
        }
        while self
            .wave
            .get_hint(self.upper, &mut self.hint)
            .is_some_and(|knot| knot.time <= time)
        {
            self.upper += 1;
        }
        sample_at(self.wave, time, self.upper, &mut self.hint)
    }
}

// Both the scattered curve and its protected times are already sorted.
// Insert only missing points; leave the curve borrowed if every protected
// time already exists. The interpolation is identical to random access.
fn augment_curve<'a, Value: WaveValue>(
    curve: &'a [Knot<Value>],
    times: &[Real],
    origin: Real,
) -> SpiceResult<std::borrow::Cow<'a, [Knot<Value>]>> {
    if times.iter().all(|time| {
        curve
            .binary_search_by(|knot| knot.time.total_cmp(time))
            .is_ok()
    }) {
        return Ok(std::borrow::Cow::Borrowed(curve));
    }
    let mut output = Vec::with_capacity(curve.len().saturating_add(times.len()));
    let mut cursor = WaveCursor::new(curve, curve[0].time, origin);
    let mut index = 0;
    for &time in times {
        let first = index;
        while curve.get(index).is_some_and(|knot| knot.time < time) {
            index += 1;
        }
        output.extend_from_slice(&curve[first..index]);
        if curve
            .get(index)
            .is_some_and(|knot| knot.time.total_cmp(&time).is_eq())
        {
            output.push(curve[index].clone());
            index += 1;
        } else {
            output.push(Knot {
                time,
                wave: cursor.sample(time)?,
            });
        }
    }
    output.extend_from_slice(&curve[index..]);
    Ok(std::borrow::Cow::Owned(output))
}

fn shifted_times<W: WaveRead + ?Sized>(
    wave: &W,
    delay: Real,
    start: Real,
    end: Real,
    times: &mut Vec<Real>,
) {
    let first = wave.partition_point(|knot| knot.time + delay <= start);
    let mut hint = 0;
    for index in first..wave.len() {
        let Some(knot) = wave.get_hint(index, &mut hint) else {
            break;
        };
        let time = knot.time + delay;
        if time >= end {
            break;
        }
        if time > start {
            times.push(time);
        }
    }
}

fn sort_times(times: &mut Vec<Real>) {
    times.sort_unstable_by(Real::total_cmp);
    times.dedup();
}

// A flat run needs its first and last endpoints, but none of its
// interior knots. Removing such knots is exact, including for affine
// trials; it also avoids treating them as protected boundary corners.
fn append<Value: PartialEq>(wave: &mut Vec<Point<Value>>, knot: Point<Value>) {
    if wave.len() >= 2 {
        let last = wave.len() - 1;
        if wave[last - 1].wave == wave[last].wave && wave[last].wave == knot.wave {
            wave[last] = knot;
            return;
        }
    }
    wave.push(knot);
}

fn prune<Value>(wave: &mut Vec<Point<Value>>, oldest: Real) {
    let count = wave
        .partition_point(|knot| knot.time <= oldest)
        .saturating_sub(1);
    wave.drain(..count);
    // Long trials may have grown the vector substantially. Do not retain
    // obsolete capacity indefinitely after the consumed prefix is removed.
    if wave.capacity() > wave.len().saturating_mul(4).max(16) {
        wave.shrink_to_fit();
    }
}

/// Boundary data for an accepted interval. Front markers describe the
/// start of the new ramp, not its end.
#[derive(Clone, Copy)]
pub(super) struct BoundaryStep<'a> {
    pub(super) input: &'a [Real],
    pub(super) delta: &'a [Real],
    pub(super) fronts: &'a [bool],
}

// Preserve transitions into or out of exact constant intervals, even
// below the solver's significance threshold. A resolved slope corner
// also retains its physically delayed first and ballistic arrivals.
pub(super) fn boundary_front(previous: Option<Real>, delta: Real, resolved: bool) -> bool {
    resolved || previous.map_or(delta != 0., |old| (old == 0.) != (delta == 0.))
}

fn shifted_fronts(
    fronts: &[Real],
    pending: Option<Real>,
    delay: Real,
    start: Real,
    end: Real,
    times: &mut Vec<Real>,
) {
    let first = fronts.partition_point(|&time| time + delay <= start);
    for &corner in &fronts[first..] {
        let time = corner + delay;
        if time >= end {
            break;
        }
        if time > start {
            times.push(time);
        }
    }
    if let Some(corner) = pending {
        let time = corner + delay;
        if time > start && time < end {
            times.push(time);
        }
    }
}

// Preserve exact zero support of each affine component independently.
// Equal-valued nonzero samples can merely be floating-point plateaus
// and do not independently denote a source wavefront.
fn zero_fronts<Value: WaveValue>(curve: &[Knot<Value>], times: &mut Vec<Real>) {
    for triple in curve.windows(3) {
        let width = triple
            .iter()
            .map(|knot| knot.wave.width())
            .max()
            .unwrap_or(1);
        if (0..width).any(|column| {
            triple[1].wave.component(column) == 0.
                && (triple[0].wave.component(column) == 0.)
                    != (triple[2].wave.component(column) == 0.)
        }) {
            times.push(triple[1].time);
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct ContinuousPlan {
    step: Real,
    delays: Vec<Real>,
    // For each outgoing stage port, exact first paths and straight modal
    // flights from boundary sources. Never reduce across these arrivals.
    first_outgoing: Vec<Real>,
    ballistic: Vec<[Real; 2]>,
    scattering_rows: Vec<Vec<(usize, Real)>>,
    boundary_retention: Vec<Real>,
    tolerance_divisor: Real,
    energy_factor: Real,
}

impl ContinuousPlan {
    pub(super) fn prepare(plan: &LossyPlan) -> SpiceResult<Option<Self>> {
        let ports = plan.launch.len();
        let n = plan.half.len();
        // Separable modes and a single stage have an inexpensive exact
        // impulse graph, with no combinatorial mode conversion paths.
        if plan.sections == 1
            || plan
                .scattering
                .iter()
                .enumerate()
                .all(|(input, column)| column.iter().all(|&(output, _)| input % n == output % n))
        {
            return Ok(None);
        }
        let step = plan.half.iter().copied().fold(Real::INFINITY, Real::min) * 2.;
        let mut delays = Vec::with_capacity(plan.terminal);
        let mut ballistic = Vec::with_capacity(plan.terminal);
        let entries = plan
            .terminal
            .checked_mul(ports)
            .ok_or_else(|| invalid("CPL protected arrival table size overflow"))?;
        if entries.saturating_mul(std::mem::size_of::<Real>()) > plan.limit {
            return Err(invalid("CPL protected arrival table workspace exceeded"));
        }
        let mut first_outgoing = vec![Real::INFINITY; entries];
        let mut boundary_retention = (0..ports)
            .map(|port| plan.half[port % n])
            .collect::<Vec<_>>();
        for node in 0..plan.terminal {
            let stage = node / ports;
            let row = node % ports;
            let mode = row % n;
            let half = plan.half[mode];
            let flight = 2. * half;
            delays.push(
                if (row < n && stage == 0) || (row >= n && stage + 1 == plan.sections) {
                    half
                } else {
                    flight
                },
            );
            let mut times = [0.; 2];
            for (side, time) in times.iter_mut().enumerate() {
                let distance = if side == 0 {
                    stage
                } else {
                    plan.sections - 1 - stage
                };
                let distance =
                    Real::from(u32::try_from(distance).map_err(|_| invalid("CPL stage index"))?);
                *time = flight.mul_add(distance, half);
                let source = side * n + mode;
                boundary_retention[source] = boundary_retention[source].max(*time);
            }
            ballistic.push(times);
        }
        for (source, (distances, retention)) in plan
            .first_at_node
            .iter()
            .zip(&mut boundary_retention)
            .enumerate()
        {
            for stage in 0..plan.sections {
                for (input, column) in plan.scattering.iter().enumerate() {
                    let time = distances[stage * ports + input];
                    if !time.is_finite() {
                        continue;
                    }
                    for &(row, _) in column {
                        let output = stage * ports + row;
                        let first = &mut first_outgoing[output * ports + source];
                        *first = first.min(time);
                    }
                }
            }
            for row in first_outgoing.chunks_exact(ports) {
                if row[source].is_finite() {
                    *retention = retention.max(row[source]);
                }
            }
        }
        let tolerance_divisor = 16.
            * Real::from(u32::try_from(plan.sections).map_err(|_| invalid("CPL section count"))?);
        let mut scattering_rows = vec![Vec::new(); ports];
        for (input, column) in plan.scattering.iter().enumerate() {
            for &(output, gain) in column {
                scattering_rows[output].push((input, gain));
            }
        }
        let energy_factor = scattering_energy_factor(&scattering_rows);
        Ok(Some(Self {
            step,
            delays,
            first_outgoing,
            ballistic,
            scattering_rows,
            boundary_retention,
            tolerance_divisor,
            energy_factor,
        }))
    }

    pub(super) const fn step(&self) -> Real {
        self.step
    }

    pub(super) fn empty(&self, ports: usize) -> ContinuousHistory {
        ContinuousHistory {
            origin: 0.,
            reltol: 0.,
            initial_input: vec![0.; ports],
            boundary: vec![KnownWave::default(); ports],
            fronts: vec![Vec::new(); ports],
            outgoing: vec![KnownWave::default(); self.delays.len()],
        }
    }

    pub(super) fn start(&self, time: Real, reltol: Real, input: &[Real]) -> ContinuousHistory {
        let mut history = self.empty(input.len());
        history.origin = time;
        history.reltol = reltol;
        history.initial_input.copy_from_slice(input);
        for wave in history.boundary.iter_mut().chain(&mut history.outgoing) {
            *wave = KnownWave::single(Point { time, wave: 0. });
        }
        history
    }

    fn compress<Value: WaveValue>(
        &self,
        step: &Step<'_, Value>,
        node: usize,
        curve: &[Knot<Value>],
        terminal_sample: Option<Real>,
    ) -> SpiceResult<Wave<Value>> {
        let history = step.history;
        let start = curve[0].time;
        let end = curve[curve.len() - 1].time;
        if curve.iter().all(|knot| knot.wave == curve[0].wave) {
            return Ok(vec![curve[0].clone(), curve[curve.len() - 1].clone()]);
        }
        let mut times = Vec::new();
        let ports = history.boundary.len();
        let n = ports / 2;
        let first = &self.first_outgoing[node * ports..(node + 1) * ports];
        let pending = |port: usize| {
            step.fronts
                .is_none_or(|fronts| fronts[port])
                .then_some(step.start)
        };
        for (source, (&delay, fronts)) in first.iter().zip(&history.fronts).enumerate() {
            if delay.is_finite() {
                shifted_fronts(fronts, pending(source), delay, start, end, &mut times);
            }
        }
        for (side, &delay) in self.ballistic[node].iter().enumerate() {
            let source = side * n + node % n;
            shifted_fronts(
                &history.fronts[source],
                pending(source),
                delay,
                start,
                end,
                &mut times,
            );
        }
        zero_fronts(curve, &mut times);
        if let Some(time) = terminal_sample
            && time > start
            && time < end
        {
            times.push(time);
        }
        sort_times(&mut times);
        let augmented = augment_curve(curve, &times, history.origin)?;
        let tolerance = (history.reltol / self.tolerance_divisor).max(128. * Real::EPSILON);
        let known_scale = history.outgoing[node]
            .iter()
            .map(|knot| knot.wave.abs())
            .chain(curve.iter().map(|knot| knot.wave.known().abs()))
            .fold(0., Real::max);
        let mut output = vec![augmented[0].clone()];
        let mut first = 0;
        for time in times.into_iter().chain(std::iter::once(end)) {
            let last = augmented.partition_point(|knot| knot.time < time);
            reduce::reduce(
                &augmented[first..=last],
                ports + 1,
                tolerance,
                known_scale,
                self.energy_factor,
                &mut output,
            );
            first = last;
        }
        Ok(output)
    }

    // Ordinary transient steps are no longer than the shortest complete
    // interstage flight. A terminal wave at the new time can therefore
    // depend only on accepted incident history. Evaluate just the two
    // terminal stages during Newton loads; build the interior trial
    // history once when accepting. Oversized steps retain the full affine
    // method of steps, including feedback within the trial interval.
    pub(super) fn outputs(
        &self,
        plan: &LossyPlan,
        history: &ContinuousHistory,
        start: Real,
        end: Real,
        boundary_bytes: usize,
    ) -> SpiceResult<Vec<Affine>> {
        if end - self.step > start {
            return Ok(self
                .trial(plan, history, start, end, boundary_bytes)?
                .output);
        }
        let ports = plan.launch.len();
        let n = plan.half.len();
        let mut result = Vec::with_capacity(ports);
        for row in 0..ports {
            let stage = if row < n { 0 } else { plan.sections - 1 };
            let time = end - plan.half[row % n];
            let wave = if time <= start {
                Affine::scalar(sample(
                    &history.outgoing[stage * ports + row],
                    time,
                    history.origin,
                )?)
            } else {
                let mut value = Affine::zero();
                for &(port, gain) in &self.scattering_rows[row] {
                    let (node, delay) = source(plan, stage, port);
                    let incident =
                        node.map_or(&history.boundary[port], |index| &history.outgoing[index]);
                    let incident = sample(incident, (time - delay).min(start), history.origin)?;
                    value.add_scaled(&Affine::scalar(incident), gain);
                }
                value
            };
            if !wave.finite() {
                return Err(invalid("nonfinite CPL terminal wave"));
            }
            result.push(wave);
        }
        Ok(result)
    }

    fn trial(
        &self,
        plan: &LossyPlan,
        history: &ContinuousHistory,
        start: Real,
        end: Real,
        boundary_bytes: usize,
    ) -> SpiceResult<Trial> {
        self.propagate(plan, history, start, end, boundary_bytes, None)
    }

    // Ordinary loads depend only on accepted history. Their accepted
    // wave propagation can therefore store native scalars, not affine
    // coefficient vectors. Oversized trials retain the exact affine
    // construction used by their implicit endpoint load.
    pub(super) fn advance(
        &self,
        plan: &LossyPlan,
        history: &ContinuousHistory,
        start: Real,
        end: Real,
        boundary_bytes: usize,
        boundary: BoundaryStep<'_>,
    ) -> SpiceResult<ContinuousHistory> {
        if end - self.step <= start && end <= start + self.step {
            self.advance_scalar(plan, history, start, end, boundary_bytes, boundary)
        } else {
            let trial = self.trial(plan, history, start, end, boundary_bytes)?;
            self.accept(
                history,
                trial,
                start,
                end,
                boundary,
                (boundary_bytes, plan.limit.saturating_mul(2)),
            )
        }
    }

    // Ordinary accepted steps cannot read a newly generated internal
    // wave: every internal flight is at least the complete step length.
    // Materialize the next history one stage at a time, rather than
    // retaining a full trial and then cloning/merging a second history.
    fn advance_scalar(
        &self,
        plan: &LossyPlan,
        history: &ContinuousHistory,
        start: Real,
        end: Real,
        boundary_bytes: usize,
        boundary: BoundaryStep<'_>,
    ) -> SpiceResult<ContinuousHistory> {
        let ports = plan.launch.len();
        let trial = Trial::<Real> {
            outgoing: vec![Vec::new(); plan.terminal],
            output: Vec::new(),
        };
        let step = Step {
            plan,
            history,
            trial: &trial,
            start,
            end,
            at: start,
            next: end,
            realized: Some(boundary.delta),
            fronts: Some(boundary.fronts),
        };
        // The old state and new state share complete immutable blocks.
        // Charge each shared allocation only once against the original
        // total workspace, along with the current stage scratch.
        let limit = plan.limit.saturating_mul(2);
        let stage_headers = ports * std::mem::size_of::<Wave<Real>>();
        let (mut next, mut bytes) = self.successor(
            history,
            start,
            end,
            boundary,
            (boundary_bytes, limit, trial.workspace()),
        )?;
        for stage in 0..plan.sections {
            bytes = bytes.saturating_add(stage_headers);
            let curves = self.compress_stage(&step, stage, &mut bytes, limit)?;
            for (row, curve) in curves.into_iter().enumerate() {
                let node = stage * ports + row;
                let cutoff = end - self.delays[node];
                let old = &history.outgoing[node];
                let curve_bytes = wave_bytes(&curve);
                let window = RetainedWindow::new(old, &curve, start, cutoff);
                let allocation = window.allocation_bound(old, curve.len());
                if bytes.saturating_add(allocation) > limit {
                    return Err(invalid(
                        "CPL accepted window construction workspace exceeded",
                    ));
                }
                let wave = window.merge(old, curve, cutoff, boundary.delta)?;
                bytes = bytes
                    .saturating_sub(curve_bytes)
                    .saturating_add(wave.unique_workspace(old));
                next.outgoing.push(wave);
            }
            bytes = bytes.saturating_sub(stage_headers);
        }
        Ok(next)
    }

    fn propagate<Value: WaveValue>(
        &self,
        plan: &LossyPlan,
        history: &ContinuousHistory,
        start: Real,
        end: Real,
        boundary_bytes: usize,
        realized: Option<&[Real]>,
    ) -> SpiceResult<Trial<Value>> {
        let ports = plan.launch.len();
        let n = plan.half.len();
        let mut trial: Trial<Value> = Trial {
            outgoing: vec![Vec::new(); plan.terminal],
            output: Vec::new(),
        };
        let mut at = start;
        let mut substeps = 0_usize;
        loop {
            if at >= end {
                break;
            }
            let mut next = (at + self.step).min(end);
            loop {
                if next - self.step <= at {
                    break;
                }
                next = next.next_down();
            }
            if next <= at || !next.is_finite() {
                return Err(invalid(
                    "CPL internal flight is smaller than representable time",
                ));
            }
            substeps += 1;
            if substeps > plan.limit / std::mem::size_of::<Knot<Value>>() {
                return Err(invalid("CPL method-of-steps work budget exceeded"));
            }
            let context = Step {
                plan,
                history,
                trial: &trial,
                start,
                end,
                at,
                next,
                realized,
                fronts: None,
            };
            let mut curves = Vec::with_capacity(plan.terminal);
            let mut bytes = boundary_bytes
                .saturating_add(trial.workspace())
                .saturating_add(plan.terminal * std::mem::size_of::<Wave<Value>>());
            for stage in 0..plan.sections {
                curves.extend(self.compress_stage(
                    &context,
                    stage,
                    &mut bytes,
                    plan.limit.saturating_mul(2),
                )?);
            }
            for (node, (wave, curve)) in trial.outgoing.iter_mut().zip(curves).enumerate() {
                let skip = usize::from(!wave.is_empty());
                for knot in curve.into_iter().skip(skip) {
                    append(wave, knot);
                }
                prune(wave, next - self.delays[node]);
            }
            at = next;
        }
        for row in 0..ports {
            let node = if row < n {
                row
            } else {
                (plan.sections - 1) * ports + row
            };
            let time = end - plan.half[row % n];
            trial.output.push(trial.sample(history, node, start, time)?);
        }
        if trial.output.iter().any(|wave| !wave.finite()) {
            return Err(invalid("nonfinite CPL trial output"));
        }
        Ok(trial)
    }

    // Bound temporary waveform memory independently of conductor order.
    // A stage normally shares all incident samples between its outputs.
    // When that matrix of samples would be too large, reduce batches of
    // rows and, if necessary, overlapping time chunks. Chunk boundaries
    // are exact retained endpoints, so the energy certificates compose.
    fn compress_stage<Value: WaveValue>(
        &self,
        step: &Step<'_, Value>,
        stage: usize,
        bytes: &mut usize,
        limit: usize,
    ) -> SpiceResult<Vec<Wave<Value>>> {
        let plan = step.plan;
        let ports = plan.launch.len();
        let n = plan.half.len();
        let input = step.stage(stage);
        let mut output = vec![Vec::new(); ports];
        let affine = step.realized.is_none()
            && input
                .sources
                .iter()
                .any(|&(_, delay)| step.next - delay > step.start);
        let per_knot = std::mem::size_of::<Knot<Value>>()
            + usize::from(affine) * ports * std::mem::size_of::<Real>();
        let metadata = input.times.capacity() * std::mem::size_of::<Real>()
            + input.sources.capacity() * std::mem::size_of::<(Option<usize>, Real)>()
            + ports
                * (std::mem::size_of::<IncidentCursor<'_, Value>>()
                    + std::mem::size_of::<Value>()
                    + usize::from(affine) * ports * std::mem::size_of::<Real>());
        let available = limit.saturating_sub(bytes.saturating_add(metadata));
        // Reserve space for augmentation and reduction scratch, not
        // just the raw scattered curve.
        let batch = (available
            / per_knot
                .saturating_mul(input.times.len())
                .saturating_mul(4)
                .max(1))
        .clamp(1, ports);
        let mut first_row = 0;
        while first_row < ports {
            let last_row = (first_row + batch).min(ports);
            let mut first_time = 0;
            while first_time + 1 < input.times.len() {
                let available = limit.saturating_sub(bytes.saturating_add(metadata));
                let count = available / (per_knot * (last_row - first_row) * 4);
                if count < 2 {
                    return Err(invalid(format!(
                        "CPL stage waveform workspace exceeded at {:e}: stage {stage}, row {first_row}, retained {boundary} bytes, trial {bytes} bytes, metadata {metadata}, available {available}, {} times",
                        step.next,
                        input.times.len(),
                        boundary = step.history.workspace()
                    )));
                }
                let last_time = first_time
                    .saturating_add(count - 1)
                    .min(input.times.len() - 1);
                let stage_curves = input.curves(
                    &self.scattering_rows[first_row..last_row],
                    &input.times[first_time..=last_time],
                )?;
                let scratch = histories_bytes(&stage_curves);
                for (row, curve) in (first_row..last_row).zip(&stage_curves) {
                    let terminal =
                        (row < n && stage == 0) || (row >= n && stage + 1 == plan.sections);
                    let terminal_sample = terminal.then(|| step.end - plan.half[row % n]);
                    let compressed =
                        self.compress(step, stage * ports + row, curve, terminal_sample)?;
                    let previous = wave_bytes(&output[row]);
                    let skip = usize::from(!output[row].is_empty());
                    for knot in compressed.into_iter().skip(skip) {
                        append(&mut output[row], knot);
                    }
                    *bytes = bytes
                        .saturating_sub(previous)
                        .saturating_add(wave_bytes(&output[row]));
                    if bytes.saturating_add(metadata).saturating_add(scratch) > limit {
                        return Err(invalid(format!(
                            "CPL trial waveform workspace exceeded at {:e}: stage {stage}, row {row}, retained {boundary} bytes, trial {bytes} bytes, metadata {metadata}, scratch {scratch}, {} times",
                            step.next,
                            input.times.len(),
                            boundary = step.history.workspace()
                        )));
                    }
                }
                first_time = last_time;
            }
            first_row = last_row;
        }
        Ok(output)
    }

    fn accept<Value: WaveValue>(
        &self,
        history: &ContinuousHistory,
        trial: Trial<Value>,
        start: Real,
        end: Real,
        boundary: BoundaryStep<'_>,
        budget: (usize, usize),
    ) -> SpiceResult<ContinuousHistory> {
        let (boundary_bytes, limit) = budget;
        let (mut next, mut bytes) = self.successor(
            history,
            start,
            end,
            boundary,
            (boundary_bytes, limit, trial.workspace()),
        )?;
        for (node, (old, new)) in history.outgoing.iter().zip(trial.outgoing).enumerate() {
            let cutoff = end - self.delays[node];
            let window = RetainedWindow::new(old, &new, start, cutoff);
            let allocation = window.allocation_bound(old, new.len());
            if bytes.saturating_add(allocation) > limit {
                return Err(invalid(
                    "CPL accepted window construction workspace exceeded",
                ));
            }
            let previous = wave_bytes(&new);
            let wave = window.merge(old, new, cutoff, boundary.delta)?;
            bytes = bytes
                .saturating_sub(previous)
                .saturating_add(wave.unique_workspace(old));
            next.outgoing.push(wave);
        }
        Ok(next)
    }

    fn successor(
        &self,
        history: &ContinuousHistory,
        start: Real,
        end: Real,
        boundary: BoundaryStep<'_>,
        budget: (usize, usize, usize),
    ) -> SpiceResult<(ContinuousHistory, usize)> {
        let (boundary_bytes, limit, trial_bytes) = budget;
        let mut next = ContinuousHistory {
            origin: history.origin,
            reltol: history.reltol,
            initial_input: history.initial_input.clone(),
            boundary: Vec::with_capacity(history.boundary.len()),
            fronts: Vec::with_capacity(history.fronts.len()),
            outgoing: Vec::with_capacity(history.outgoing.len()),
        };
        let mut bytes = boundary_bytes
            .saturating_add(boundary_bytes.saturating_sub(history.workspace()))
            .saturating_add(next.headers())
            .saturating_add(trial_bytes);
        for (port, old) in history.boundary.iter().enumerate() {
            let value = boundary.input[port] - next.initial_input[port];
            if !value.is_finite() {
                return Err(invalid("nonfinite accepted CPL boundary wave"));
            }
            let new = vec![Point {
                time: end,
                wave: value,
            }];
            let cutoff = end - self.boundary_retention[port];
            let markers = &history.fronts[port];
            let first = markers.partition_point(|&time| time <= cutoff);
            let record = boundary.fronts[port] && start > cutoff;
            let count = markers.len() - first + usize::from(record);
            let marker_bytes = count.saturating_mul(std::mem::size_of::<Real>());
            if bytes.saturating_add(marker_bytes) > limit {
                return Err(invalid("CPL protected front workspace exceeded"));
            }
            let mut fronts = Vec::with_capacity(count);
            fronts.extend_from_slice(&markers[first..]);
            if record {
                fronts.push(start);
            }
            bytes = bytes.saturating_add(fronts.capacity() * std::mem::size_of::<Real>());
            next.fronts.push(fronts);
            let window = RetainedWindow::new(old, &new, start, cutoff);
            if bytes.saturating_add(window.allocation_bound(old, new.len())) > limit {
                return Err(invalid(
                    "CPL accepted boundary construction workspace exceeded",
                ));
            }
            let wave = window.merge(old, new, cutoff, &[])?;
            bytes = bytes.saturating_add(wave.unique_workspace(old));
            next.boundary.push(wave);
        }
        Ok((next, bytes))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ContinuousHistory {
    origin: Real,
    reltol: Real,
    initial_input: Vec<Real>,
    boundary: Vec<KnownWave>,
    fronts: Vec<Vec<Real>>,
    outgoing: Vec<KnownWave>,
}

impl ContinuousHistory {
    pub(super) fn workspace(&self) -> usize {
        self.headers()
            .saturating_add(self.front_bytes())
            .saturating_add(
                self.boundary
                    .iter()
                    .chain(&self.outgoing)
                    .map(KnownWave::workspace)
                    .sum::<usize>(),
            )
    }

    const fn headers(&self) -> usize {
        (self.boundary.capacity() + self.outgoing.capacity()) * std::mem::size_of::<KnownWave>()
            + self.initial_input.capacity() * std::mem::size_of::<Real>()
            + self.fronts.capacity() * std::mem::size_of::<Vec<Real>>()
    }

    fn front_bytes(&self) -> usize {
        self.fronts
            .iter()
            .map(|times| times.capacity() * std::mem::size_of::<Real>())
            .sum()
    }

    pub(super) fn unique_workspace(&self, old: &Self) -> usize {
        self.headers()
            .saturating_add(self.front_bytes())
            .saturating_add(
                self.boundary
                    .iter()
                    .zip(&old.boundary)
                    .chain(self.outgoing.iter().zip(&old.outgoing))
                    .map(|(wave, previous)| wave.unique_workspace(previous))
                    .sum::<usize>(),
            )
    }

    pub(super) fn len(&self) -> usize {
        self.boundary
            .iter()
            .chain(&self.outgoing)
            .map(KnownWave::len)
            .sum()
    }

    pub(super) fn clear(&mut self) {
        for wave in self.boundary.iter_mut().chain(&mut self.outgoing) {
            wave.clear();
        }
        self.initial_input.fill(0.);
        for fronts in &mut self.fronts {
            fronts.clear();
        }
        self.origin = 0.;
        self.reltol = 0.;
    }
}

struct Trial<Value = Affine> {
    outgoing: Vec<Wave<Value>>,
    output: Vec<Value>,
}

impl<Value: WaveValue> Trial<Value> {
    fn workspace(&self) -> usize {
        histories_bytes(&self.outgoing)
            .saturating_add(self.output.capacity() * std::mem::size_of::<Value>())
            .saturating_add(
                self.output
                    .iter()
                    .map(WaveValue::allocated_bytes)
                    .sum::<usize>(),
            )
    }

    fn sample(
        &self,
        history: &ContinuousHistory,
        node: usize,
        start: Real,
        time: Real,
    ) -> SpiceResult<Value> {
        if time <= start {
            Ok(Value::scalar(sample(
                &history.outgoing[node],
                time,
                history.origin,
            )?))
        } else {
            sample(&self.outgoing[node], time, history.origin)
        }
    }
}

struct Step<'a, Value> {
    plan: &'a LossyPlan,
    history: &'a ContinuousHistory,
    trial: &'a Trial<Value>,
    start: Real,
    end: Real,
    at: Real,
    next: Real,
    realized: Option<&'a [Real]>,
    fronts: Option<&'a [bool]>,
}

fn source(plan: &LossyPlan, stage: usize, port: usize) -> (Option<usize>, Real) {
    let ports = plan.launch.len();
    let n = plan.half.len();
    let mode = port % n;
    let node = if port < n {
        stage
            .checked_sub(1)
            .map(|neighbor| neighbor * ports + n + mode)
    } else {
        (stage + 1 < plan.sections).then_some((stage + 1) * ports + mode)
    };
    let delay = if node.is_some() {
        2. * plan.half[mode]
    } else {
        plan.half[mode]
    };
    (node, delay)
}

impl<Value: WaveValue> Step<'_, Value> {
    fn stage(&self, stage: usize) -> Stage<'_, '_, Value> {
        let ports = self.plan.launch.len();
        let sources: Vec<_> = (0..ports)
            .map(|port| source(self.plan, stage, port))
            .collect();
        let mut times = vec![self.at, self.next];
        for (port, &(node, delay)) in sources.iter().enumerate() {
            if let Some(node) = node {
                shifted_times(
                    &self.history.outgoing[node],
                    delay,
                    self.at,
                    self.next,
                    &mut times,
                );
                shifted_times(
                    &self.trial.outgoing[node],
                    delay,
                    self.at,
                    self.next,
                    &mut times,
                );
            } else {
                shifted_times(
                    &self.history.boundary[port],
                    delay,
                    self.at,
                    self.next,
                    &mut times,
                );
            }
        }
        sort_times(&mut times);
        Stage {
            step: self,
            sources,
            times,
        }
    }
}

struct IncidentCursor<'a, Value: WaveValue> {
    port: usize,
    delay: Real,
    accepted: WaveCursor<'a, KnownWave>,
    trial: Option<WaveCursor<'a, Wave<Value>>>,
}

impl<'a, Value: WaveValue> IncidentCursor<'a, Value> {
    fn new(
        step: &Step<'a, Value>,
        port: usize,
        node: Option<usize>,
        delay: Real,
        first_time: Real,
    ) -> Self {
        let time = first_time - delay;
        let origin = step.history.origin;
        let accepted = node.map_or(&step.history.boundary[port], |node| {
            &step.history.outgoing[node]
        });
        Self {
            port,
            delay,
            accepted: WaveCursor::new(accepted, time.min(step.start), origin),
            trial: node.map(|node| {
                WaveCursor::new(&step.trial.outgoing[node], time.max(step.start), origin)
            }),
        }
    }

    fn sample(&mut self, time: Real, step: &Step<'_, Value>) -> SpiceResult<Value> {
        let time = time - self.delay;
        if time <= step.start {
            return Ok(Value::scalar(self.accepted.sample(time)?));
        }
        if let Some(trial) = &mut self.trial {
            return trial.sample(time);
        }
        let known = self.accepted.sample(step.start)?;
        let fraction = ((time - step.start) / (step.end - step.start)).clamp(0., 1.);
        Value::boundary(
            known,
            self.port,
            fraction,
            step.plan.launch.len() + 1,
            step.realized,
        )
    }
}

struct Stage<'a, 'b, Value> {
    step: &'a Step<'b, Value>,
    sources: Vec<(Option<usize>, Real)>,
    times: Vec<Real>,
}

impl<Value: WaveValue> Stage<'_, '_, Value> {
    fn curves(&self, rows: &[Vec<(usize, Real)>], times: &[Real]) -> SpiceResult<Vec<Wave<Value>>> {
        let mut curves: Vec<Wave<Value>> = (0..rows.len())
            .map(|_| Vec::with_capacity(times.len()))
            .collect();
        let mut cursors: Vec<_> = self
            .sources
            .iter()
            .enumerate()
            .map(|(port, &(node, delay))| {
                IncidentCursor::new(self.step, port, node, delay, times[0])
            })
            .collect();
        let mut incident = Vec::with_capacity(self.sources.len());
        for &time in times {
            incident.clear();
            for cursor in &mut cursors {
                incident.push(cursor.sample(time, self.step)?);
            }
            for (curve, row) in curves.iter_mut().zip(rows) {
                let mut wave = Value::scalar(0.);
                for &(port, gain) in row {
                    wave.add_scaled(&incident[port], gain);
                }
                if !wave.finite() {
                    return Err(invalid("nonfinite CPL stage wave"));
                }
                curve.push(Knot { time, wave });
            }
        }
        Ok(curves)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::cpl::{
        CoupledLineLimits,
        lossless::ModalBasis,
        plan::CoupledLinePlan,
        transport::{Kernel, segment_response},
    };

    #[test]
    fn protected_curve_merge_retains_exact_knots_and_interpolation() {
        let curve = [(0., 0.), (0.2, 1.), (0.4, -0.5), (0.9, 0.4), (1., 0.)]
            .map(|(time, wave)| Knot { time, wave });
        assert!(matches!(
            augment_curve(&curve, &[0.2, 0.4], 0.).unwrap(),
            std::borrow::Cow::Borrowed(_)
        ));
        let times = [0.1, 0.2, 0.3, 0.4, 0.41, 0.6, 0.9, 0.99];
        let result = augment_curve(&curve, &times, 0.).unwrap();
        let mut expected: Vec<_> = curve.iter().map(|knot| knot.time).chain(times).collect();
        sort_times(&mut expected);
        assert_eq!(result.len(), expected.len());
        for (knot, time) in result.iter().zip(expected) {
            assert_eq!(knot.time, time);
            assert_eq!(knot.wave, sample(&curve, time, 0.).unwrap());
        }
    }

    #[test]
    fn monotonic_cursor_matches_random_access_for_scalar_and_affine_waves() {
        let knots = [(0., 0.), (0.1, 1.), (0.15, -0.25), (0.8, 0.5), (1., 0.)]
            .map(|(time, wave)| Knot { time, wave });
        let affine = knots
            .iter()
            .map(|knot| {
                let mut wave = Affine::scalar(knot.wave);
                wave.set(1, 2. * knot.wave, 3);
                wave.set(2, 1. - knot.wave, 3);
                Knot {
                    time: knot.time,
                    wave,
                }
            })
            .collect::<Vec<_>>();
        let times = [-0.2, 0., 0.01, 0.1, 0.1, 0.12, 0.15, 0.3, 0.8, 0.85, 1., 1.];
        let mut scalar_cursor = WaveCursor::new(&knots, times[0], 0.);
        let mut affine_cursor = WaveCursor::new(&affine, times[0], 0.);
        for time in times {
            assert_eq!(
                scalar_cursor.sample(time).unwrap(),
                sample(&knots, time, 0.).unwrap()
            );
            assert_eq!(
                affine_cursor.sample(time).unwrap(),
                sample(&affine, time, 0.).unwrap()
            );
        }
        assert!(scalar_cursor.sample(1.1).is_err());
        assert!(WaveCursor::new(&knots[2..], 0.05, 0.).sample(0.05).is_err());
    }

    fn dense_plan(sections: usize) -> LossyPlan {
        let material = CoupledLinePlan::from_packed(
            3,
            1.,
            &[20., 2., 3., 30., 4., 40.],
            &[250e-9, 0., 0., 356.15e-9, 0., 461.7e-9],
            &[0.001, -0.0001, -0.0001, 0.002, -0.0002, 0.003],
            &[100e-12, 0., 0., 142.46e-12, 0., 184.68e-12],
            CoupledLineLimits::default(),
        )
        .unwrap();
        let basis = ModalBasis::prepare(&material).unwrap().unwrap();
        LossyPlan::prepare(basis, &material, sections).unwrap()
    }

    fn test_fronts(
        input: &[Real],
        delta: &[Real],
        previous: Option<&[Real]>,
        reltol: Real,
    ) -> Vec<bool> {
        input
            .iter()
            .zip(delta)
            .enumerate()
            .map(|(port, (&value, &change))| {
                let old = previous.map(|previous| previous[port]);
                let resolved = old.is_none_or(|old| (change - old).abs() > reltol * value.abs());
                boundary_front(old, change, resolved && change != 0.)
            })
            .collect()
    }

    #[test]
    fn protected_fronts_distinguish_flat_transitions_and_resolved_corners() {
        assert!(boundary_front(None, 1., false));
        assert!(!boundary_front(None, 0., false));
        assert!(boundary_front(Some(0.), 1e-30, false));
        assert!(boundary_front(Some(1e-30), 0., false));
        assert!(!boundary_front(Some(1.), 1., false));
        assert!(boundary_front(Some(1.), -1., true));
        let mut times = Vec::new();
        shifted_fronts(&[0., 1., 2., 3.], Some(4.), 0.5, 1.5, 4.6, &mut times);
        assert_eq!(times, [2.5, 3.5, 4.5]);
        let curve = [(0., 0.), (1., 0.), (2., 1.), (3., 1.), (4., 0.), (5., 0.)]
            .map(|(time, wave)| Knot { time, wave });
        times.clear();
        zero_fronts(&curve, &mut times);
        assert_eq!(times, [1., 4.]);
        let affine: Vec<_> = curve
            .iter()
            .map(|point| {
                let mut wave = Affine::scalar(1.);
                wave.set(1, point.wave, 2);
                Knot {
                    time: point.time,
                    wave,
                }
            })
            .collect();
        times.clear();
        zero_fronts(&affine, &mut times);
        assert_eq!(times, [1., 4.]);
    }

    fn ramp_response(plan: &LossyPlan, source: usize, reltol: Real) -> (Vec<Vec<Real>>, usize) {
        let continuous = plan.continuous.as_ref().unwrap();
        let ports = plan.launch.len();
        let mut previous = vec![0.; ports];
        let mut previous_delta = None;
        let mut history = continuous.start(0., reltol, &previous);
        let mut records = history.len();
        let mut result = Vec::new();
        let mut start = 0.;
        for step in 1..=120 {
            let end = Real::from(step) * 0.1e-9;
            let mut input = vec![0.; ports];
            input[source] = (end / 0.6e-9).min(1.);
            let delta: Vec<_> = input.iter().zip(&previous).map(|(a, b)| a - b).collect();
            let fronts = test_fronts(&input, &delta, previous_delta.as_deref(), reltol);
            let loaded = continuous
                .outputs(plan, &history, start, end, history.workspace())
                .unwrap();
            let accepted = continuous
                .advance(
                    plan,
                    &history,
                    start,
                    end,
                    history.workspace(),
                    BoundaryStep {
                        input: &input,
                        delta: &delta,
                        fronts: &fronts,
                    },
                )
                .unwrap();
            let mut actual = Vec::with_capacity(ports);
            let modes = ports / 2;
            for (row, load) in loaded.iter().enumerate() {
                let node = if row < modes {
                    row
                } else {
                    (plan.sections - 1) * ports + row
                };
                let time = end - plan.half[row % modes];
                let value = sample(&accepted.outgoing[node], time, accepted.origin).unwrap();
                assert!((load.evaluate(&delta) - value).abs() < 1e-12);
                actual.push(value);
            }
            result.push(actual);
            history = accepted;
            records = records.max(history.len());
            previous = input;
            previous_delta = Some(delta);
            start = end;
        }
        (result, records)
    }

    #[test]
    fn continuous_transport_matches_every_exact_modal_column() {
        let mut plan = dense_plan(8);
        plan.path_bound = None;
        let mut reference = Kernel::empty(6);
        plan.extend_kernel(&mut reference, 12e-9, 0).unwrap();
        let mut worst: Real = 0.;
        for source in 0..6 {
            let (response, records) = ramp_response(&plan, source, 1e-4);
            for (sample, outputs) in response.iter().enumerate() {
                let time = Real::from(u32::try_from(sample + 1).unwrap()) * 0.1e-9;
                for (row, &actual) in outputs.iter().enumerate() {
                    let expected =
                        segment_response(&reference.traces[row * 6 + source], time, 0., 0.6e-9).hi;
                    let error = (actual - expected).abs();
                    worst = worst.max(error);
                    assert!(
                        error < 2e-5,
                        "source {source} row {row} at {time:e}: {actual:e} != {expected:e}"
                    );
                    if time < plan.fronts[source][row] - 1e-20 {
                        assert!(actual.abs() < 1e-13, "premature arrival {source} -> {row}");
                    }
                }
            }
            eprintln!("source {source}: peak {records} retained knots");
        }
        eprintln!("all-column maximum absolute error {worst:e}");
    }

    #[test]
    fn overlapping_sources_retain_resolved_fronts_without_protecting_ramp_samples() {
        let mut plan = dense_plan(8);
        plan.path_bound = None;
        let mut reference = Kernel::empty(6);
        plan.extend_kernel(&mut reference, 12e-9, 0).unwrap();
        let continuous = plan.continuous.as_ref().unwrap();
        let mut previous = [0.; 6];
        let mut previous_delta = None;
        let mut history = continuous.start(0., 1e-4, &previous);
        let mut segments = Vec::new();
        let mut start = 0.;
        let mut saw_marker = false;
        let arrival = 17. * 0.1e-9 + continuous.first_outgoing[1];
        for step in 1..=120 {
            let end = Real::from(step) * 0.1e-9;
            let pulse = |first: i32, rise: i32, last: i32, fall: i32| {
                (Real::from(step - first) / Real::from(rise)).clamp(0., 1.)
                    - (Real::from(step - last) / Real::from(fall)).clamp(0., 1.)
            };
            let input = [
                pulse(2, 6, 43, 6),
                0.01 * pulse(17, 5, 61, 7),
                0.,
                0.,
                -0.2 * pulse(21, 11, 70, 8),
                0.,
            ];
            let delta: Vec<_> = input.iter().zip(&previous).map(|(a, b)| a - b).collect();
            let fronts = test_fronts(&input, &delta, previous_delta.as_deref(), history.reltol);
            if step == 18 {
                assert!(fronts[1]);
            }
            if (19..=22).contains(&step) {
                assert!(!fronts[1], "interior point of linear ramp");
            }
            if delta.iter().any(|&value| value != 0.) {
                segments.push((start, end, delta.clone()));
            }
            let loaded = continuous
                .outputs(&plan, &history, start, end, history.workspace())
                .unwrap();
            let accepted = continuous
                .advance(
                    &plan,
                    &history,
                    start,
                    end,
                    history.workspace(),
                    BoundaryStep {
                        input: &input,
                        delta: &delta,
                        fronts: &fronts,
                    },
                )
                .unwrap();
            if start < arrival && end > arrival {
                saw_marker = accepted.outgoing[0]
                    .iter()
                    .any(|point| point.time.to_bits() == arrival.to_bits());
            }
            for (row, wave) in loaded.iter().enumerate() {
                let actual = wave.evaluate(&delta);
                let expected: Real = segments
                    .iter()
                    .map(|(from, to, changes)| {
                        changes
                            .iter()
                            .enumerate()
                            .map(|(source, &change)| {
                                change
                                    * segment_response(
                                        &reference.traces[row * 6 + source],
                                        end,
                                        *from,
                                        *to,
                                    )
                                    .hi
                            })
                            .sum::<Real>()
                    })
                    .sum();
                assert!(
                    (actual - expected).abs() < 2e-5,
                    "row {row} at {end:e}: {actual:e} != {expected:e}"
                );
            }
            history = accepted;
            previous = input;
            previous_delta = Some(delta);
            start = end;
        }
        assert!(
            saw_marker,
            "overlapping source front was not retained exactly"
        );
    }

    #[test]
    fn dense_sixty_four_stage_transport_refines_without_path_explosion() {
        let plan = dense_plan(64);
        for source in [0, 1, 2] {
            let (coarse, coarse_records) = ramp_response(&plan, source, 1e-3);
            let (fine, fine_records) = ramp_response(&plan, source, 1e-4);
            let error = coarse
                .iter()
                .flatten()
                .zip(fine.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0., Real::max);
            eprintln!(
                "source {source}: refinement {error:e}; {coarse_records}/{fine_records} knots"
            );
            assert!(error < 2e-5);
            assert!(coarse_records < 25_000 && fine_records < 50_000);
        }
    }

    #[test]
    fn accepted_history_retains_delay_windows_not_simulation_age() {
        let plan = dense_plan(8);
        let continuous = plan.continuous.as_ref().unwrap();
        let mut previous = [0.; 6];
        let mut previous_delta = None;
        let mut history = continuous.start(0., 1e-3, &previous);
        let mut peaks = [0; 4];
        let mut start = 0.;
        for step in 1..=480 {
            let end = Real::from(step) * 0.25e-9;
            let phase = Real::from(step % 120);
            let mut input = [0.; 6];
            input[1] = (phase / 4.).min(1.) * ((40. - phase) / 4.).clamp(0., 1.);
            input[4] =
                0.3 * ((phase - 40.) / 4.).clamp(0., 1.) * ((80. - phase) / 4.).clamp(0., 1.);
            let delta: Vec<_> = input.iter().zip(&previous).map(|(a, b)| a - b).collect();
            let fronts = test_fronts(&input, &delta, previous_delta.as_deref(), history.reltol);
            let trial = continuous
                .trial(&plan, &history, start, end, history.workspace())
                .unwrap();
            history = continuous
                .accept(
                    &history,
                    trial,
                    start,
                    end,
                    BoundaryStep {
                        input: &input,
                        delta: &delta,
                        fronts: &fronts,
                    },
                    (history.workspace(), plan.limit.saturating_mul(2)),
                )
                .unwrap();
            let cycle = usize::try_from((step - 1) / 120).unwrap();
            peaks[cycle] = peaks[cycle].max(history.len());
            for (wave, delay) in history.outgoing.iter().zip(&continuous.delays) {
                assert!(wave.len() < 2 || wave.get_hint(1, &mut 0).unwrap().time >= end - delay);
            }
            previous = input;
            previous_delta = Some(delta);
            start = end;
        }
        eprintln!("periodic retained knot peaks: {peaks:?}");
        assert!(peaks[3] <= 2 * peaks[1]);
    }

    #[test]
    fn oversized_affine_trial_matches_exact_reference_and_is_repeatable() {
        let mut plan = dense_plan(4);
        plan.path_bound = None;
        let mut reference = Kernel::empty(6);
        plan.extend_kernel(&mut reference, 12e-9, 0).unwrap();
        let continuous = plan.continuous.as_ref().unwrap();
        let zero = [0.; 6];
        let history = continuous.start(0., 1e-5, &zero);
        let end = 7e-9;
        let trial = continuous
            .trial(&plan, &history, 0., end, history.workspace())
            .unwrap();
        let repeated = continuous
            .trial(&plan, &history, 0., end, history.workspace())
            .unwrap();
        assert_eq!(trial.output, repeated.output);
        assert_eq!(trial.outgoing, repeated.outgoing);
        let input = [1., -0.4, 0.8, -0.7, 0.2, -0.1];
        for (row, wave) in trial.output.iter().enumerate() {
            let expected: Real = input
                .iter()
                .enumerate()
                .map(|(source, &value)| {
                    value * segment_response(&reference.traces[row * 6 + source], end, 0., end).hi
                })
                .sum();
            assert!((wave.evaluate(&input) - expected).abs() < 2e-5);
        }
        let accepted = continuous
            .accept(
                &history,
                trial,
                0.,
                end,
                BoundaryStep {
                    input: &input,
                    delta: &input,
                    fronts: &[true; 6],
                },
                (history.workspace(), plan.limit.saturating_mul(2)),
            )
            .unwrap();
        assert_eq!(history, continuous.start(0., 1e-5, &zero));
        let next = continuous
            .trial(&plan, &accepted, end, 12e-9, accepted.workspace())
            .unwrap();
        for (row, wave) in next.output.iter().enumerate() {
            let expected: Real = input
                .iter()
                .enumerate()
                .map(|(source, &value)| {
                    value * segment_response(&reference.traces[row * 6 + source], 12e-9, 0., end).hi
                })
                .sum();
            assert!((wave.evaluate(&zero) - expected).abs() < 2e-5);
        }
    }
}
