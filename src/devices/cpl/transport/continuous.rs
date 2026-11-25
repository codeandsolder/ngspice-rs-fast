//! Continuous-time method of steps for coupled, lossy modal transport.
//!
//! Each stage operates on the exact physical delayed PWL incident waves.
//! Histories retain only the portions that can still be consumed by a
//! neighboring stage or terminal. Trial waveforms are affine in the new
//! boundary endpoint, and reduction preserves endpoints, causal fronts
//! and the wave-energy inequality for every affine combination.
mod reduce;

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
type KnownWave = Wave<Real>;

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

fn sample<Value: WaveValue>(wave: &[Point<Value>], time: Real, origin: Real) -> SpiceResult<Value> {
    if time < origin {
        return Ok(Value::scalar(0.));
    }
    let upper = wave.partition_point(|knot| knot.time <= time);
    let first = wave
        .get(upper.saturating_sub(1))
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
    let Some(last) = wave.get(upper) else {
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
    Ok(Value::interpolate(&first.wave, &last.wave, fraction))
}

fn shifted_times<Value>(
    wave: &[Point<Value>],
    delay: Real,
    start: Real,
    end: Real,
    times: &mut Vec<Real>,
) {
    let first = wave.partition_point(|knot| knot.time + delay <= start);
    for knot in &wave[first..] {
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
        Ok(Some(Self {
            step,
            delays,
            first_outgoing,
            ballistic,
            scattering_rows,
            boundary_retention,
            tolerance_divisor,
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
            boundary: vec![Vec::new(); ports],
            outgoing: vec![Vec::new(); self.delays.len()],
        }
    }

    pub(super) fn start(&self, time: Real, reltol: Real, input: &[Real]) -> ContinuousHistory {
        let mut history = self.empty(input.len());
        history.origin = time;
        history.reltol = reltol;
        history.initial_input.copy_from_slice(input);
        for wave in history.boundary.iter_mut().chain(&mut history.outgoing) {
            wave.push(Point { time, wave: 0. });
        }
        history
    }

    fn compress<Value: WaveValue>(
        &self,
        history: &ContinuousHistory,
        node: usize,
        curve: &[Knot<Value>],
        width: usize,
        terminal_sample: Option<Real>,
    ) -> SpiceResult<Wave<Value>> {
        let start = curve[0].time;
        let end = curve[curve.len() - 1].time;
        if curve.iter().all(|knot| knot.wave == curve[0].wave) {
            return Ok(vec![curve[0].clone(), curve[curve.len() - 1].clone()]);
        }
        let mut times = Vec::new();
        let ports = history.boundary.len();
        let n = ports / 2;
        let first = &self.first_outgoing[node * ports..(node + 1) * ports];
        for (wave, &delay) in history.boundary.iter().zip(first) {
            if delay.is_finite() {
                shifted_times(wave, delay, start, end, &mut times);
            }
        }
        for (side, &delay) in self.ballistic[node].iter().enumerate() {
            let source = side * n + node % n;
            shifted_times(&history.boundary[source], delay, start, end, &mut times);
        }
        if let Some(time) = terminal_sample
            && time > start
            && time < end
        {
            times.push(time);
        }
        sort_times(&mut times);
        let mut augmented = curve.to_vec();
        for &time in &times {
            if curve
                .binary_search_by(|knot| knot.time.total_cmp(&time))
                .is_err()
            {
                augmented.push(Knot {
                    time,
                    wave: sample(curve, time, history.origin)?,
                });
            }
        }
        augmented.sort_unstable_by(|a, b| a.time.total_cmp(&b.time));
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
                width,
                tolerance,
                known_scale,
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
        boundary: (&[Real], &[Real]),
    ) -> SpiceResult<ContinuousHistory> {
        let (input, delta) = boundary;
        if end - self.step <= start {
            let trial =
                self.propagate::<Real>(plan, history, start, end, boundary_bytes, Some(delta))?;
            self.accept(history, trial, start, end, input, delta)
        } else {
            let trial = self.trial(plan, history, start, end, boundary_bytes)?;
            self.accept(history, trial, start, end, input, delta)
        }
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
            };
            let mut curves = Vec::with_capacity(plan.terminal);
            let mut bytes = boundary_bytes
                .saturating_add(trial.workspace())
                .saturating_add(plan.terminal * std::mem::size_of::<Wave<Value>>());
            for stage in 0..plan.sections {
                curves.extend(self.compress_stage(&context, stage, &mut bytes)?);
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
            + input.sources.capacity() * std::mem::size_of::<(Option<usize>, Real)>();
        let available = plan.limit.saturating_sub(bytes.saturating_add(metadata));
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
                let available = plan.limit.saturating_sub(bytes.saturating_add(metadata));
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
                    let compressed = self.compress(
                        step.history,
                        stage * ports + row,
                        curve,
                        ports + 1,
                        terminal_sample,
                    )?;
                    let previous = wave_bytes(&output[row]);
                    let skip = usize::from(!output[row].is_empty());
                    for knot in compressed.into_iter().skip(skip) {
                        append(&mut output[row], knot);
                    }
                    *bytes = bytes
                        .saturating_sub(previous)
                        .saturating_add(wave_bytes(&output[row]));
                    if bytes.saturating_add(metadata).saturating_add(scratch) > plan.limit {
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
        input: &[Real],
        delta: &[Real],
    ) -> SpiceResult<ContinuousHistory> {
        // Build only the live portion of each accepted delay window.
        // Cloning the full old window and then appending before pruning
        // doubles capacity on ordinary steps even when most of it expires.
        let mut next = ContinuousHistory {
            origin: history.origin,
            reltol: history.reltol,
            initial_input: history.initial_input.clone(),
            boundary: history.boundary.clone(),
            outgoing: Vec::with_capacity(history.outgoing.len()),
        };
        for (node, (old, new)) in history.outgoing.iter().zip(trial.outgoing).enumerate() {
            let cutoff = end - self.delays[node];
            let first = old
                .partition_point(|knot| knot.time <= cutoff)
                .saturating_sub(1);
            let incoming = new.partition_point(|knot| knot.time <= start);
            let mut wave = Vec::with_capacity(old.len() - first + new.len() - incoming);
            wave.extend_from_slice(&old[first..]);
            for knot in new.into_iter().skip(incoming) {
                let value = knot.wave.evaluate(delta);
                if !value.is_finite() {
                    return Err(invalid("nonfinite accepted CPL internal wave"));
                }
                append(
                    &mut wave,
                    Point {
                        time: knot.time,
                        wave: value,
                    },
                );
            }
            prune(&mut wave, cutoff);
            next.outgoing.push(wave);
        }
        for (port, wave) in next.boundary.iter_mut().enumerate() {
            let value = input[port] - next.initial_input[port];
            if !value.is_finite() {
                return Err(invalid("nonfinite accepted CPL boundary wave"));
            }
            append(
                wave,
                Point {
                    time: end,
                    wave: value,
                },
            );
            prune(wave, end - self.boundary_retention[port]);
        }
        Ok(next)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ContinuousHistory {
    origin: Real,
    reltol: Real,
    initial_input: Vec<Real>,
    boundary: Vec<KnownWave>,
    outgoing: Vec<KnownWave>,
}

impl ContinuousHistory {
    pub(super) fn workspace(&self) -> usize {
        histories_bytes(&self.boundary)
            .saturating_add(histories_bytes(&self.outgoing))
            .saturating_add(self.initial_input.capacity() * std::mem::size_of::<Real>())
    }

    pub(super) fn len(&self) -> usize {
        self.boundary
            .iter()
            .chain(&self.outgoing)
            .map(Vec::len)
            .sum()
    }

    pub(super) fn clear(&mut self) {
        for wave in self.boundary.iter_mut().chain(&mut self.outgoing) {
            wave.clear();
        }
        self.initial_input.fill(0.);
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
    fn input(&self, port: usize, node: Option<usize>, time: Real) -> SpiceResult<Value> {
        if let Some(node) = node {
            return self.trial.sample(self.history, node, self.start, time);
        }
        let known = sample(
            &self.history.boundary[port],
            time.min(self.start),
            self.history.origin,
        )?;
        if time <= self.start {
            return Ok(Value::scalar(known));
        }
        let fraction = ((time - self.start) / (self.end - self.start)).clamp(0., 1.);
        Value::boundary(
            known,
            port,
            fraction,
            self.plan.launch.len() + 1,
            self.realized,
        )
    }

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
        let mut incident = Vec::with_capacity(self.sources.len());
        for &time in times {
            incident.clear();
            for (port, &(node, delay)) in self.sources.iter().enumerate() {
                incident.push(self.step.input(port, node, time - delay)?);
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

    fn ramp_response(plan: &LossyPlan, source: usize, reltol: Real) -> (Vec<Vec<Real>>, usize) {
        let continuous = plan.continuous.as_ref().unwrap();
        let ports = plan.launch.len();
        let mut previous = vec![0.; ports];
        let mut history = continuous.start(0., reltol, &previous);
        let mut records = history.len();
        let mut result = Vec::new();
        let mut start = 0.;
        for step in 1..=120 {
            let end = Real::from(step) * 0.1e-9;
            let mut input = vec![0.; ports];
            input[source] = (end / 0.6e-9).min(1.);
            let delta: Vec<_> = input.iter().zip(&previous).map(|(a, b)| a - b).collect();
            let loaded = continuous
                .outputs(plan, &history, start, end, history.workspace())
                .unwrap();
            if end - continuous.step() <= start {
                let trial = continuous
                    .propagate::<Real>(
                        plan,
                        &history,
                        start,
                        end,
                        history.workspace(),
                        Some(&delta),
                    )
                    .unwrap();
                for (load, &accepted) in loaded.iter().zip(&trial.output) {
                    assert!((load.evaluate(&delta) - accepted).abs() < 1e-12);
                }
                result.push(trial.output.clone());
                history = continuous
                    .accept(&history, trial, start, end, &input, &delta)
                    .unwrap();
            } else {
                let trial = continuous
                    .trial(plan, &history, start, end, history.workspace())
                    .unwrap();
                for (load, accepted) in loaded.iter().zip(&trial.output) {
                    assert!((load.evaluate(&delta) - accepted.evaluate(&delta)).abs() < 1e-12);
                }
                result.push(
                    trial
                        .output
                        .iter()
                        .map(|wave| wave.evaluate(&delta))
                        .collect(),
                );
                history = continuous
                    .accept(&history, trial, start, end, &input, &delta)
                    .unwrap();
            }
            records = records.max(history.len());
            previous = input;
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
            let trial = continuous
                .trial(&plan, &history, start, end, history.workspace())
                .unwrap();
            history = continuous
                .accept(&history, trial, start, end, &input, &delta)
                .unwrap();
            let cycle = usize::try_from((step - 1) / 120).unwrap();
            peaks[cycle] = peaks[cycle].max(history.len());
            for (wave, delay) in history.outgoing.iter().zip(&continuous.delays) {
                assert!(wave.len() < 2 || wave[1].time >= end - delay);
            }
            previous = input;
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
            .accept(&history, trial, 0., end, &input, &input)
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
