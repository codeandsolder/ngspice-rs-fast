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
pub(super) struct Knot {
    time: Real,
    wave: Affine,
}

type Wave = Vec<Knot>;

fn wave_bytes(wave: &Wave) -> usize {
    wave.capacity()
        .saturating_mul(std::mem::size_of::<Knot>())
        .saturating_add(
            wave.iter()
                .map(|knot| knot.wave.gains.capacity() * std::mem::size_of::<Real>())
                .sum::<usize>(),
        )
}

fn histories_bytes(waves: &[Wave]) -> usize {
    waves
        .len()
        .saturating_mul(std::mem::size_of::<Wave>())
        .saturating_add(waves.iter().map(wave_bytes).sum::<usize>())
}

fn sample(wave: &[Knot], time: Real, origin: Real) -> SpiceResult<Affine> {
    if time < origin {
        return Ok(Affine::zero());
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
    Ok(Affine::interpolate(&first.wave, &last.wave, fraction))
}

fn shifted_times(wave: &[Knot], delay: Real, start: Real, end: Real, times: &mut Vec<Real>) {
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

fn prune(wave: &mut Wave, oldest: Real) {
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
        Ok(Some(Self {
            step,
            delays,
            first_outgoing,
            ballistic,
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
            wave.push(Knot {
                time,
                wave: Affine::zero(),
            });
        }
        history
    }

    fn compress(
        &self,
        history: &ContinuousHistory,
        node: usize,
        curve: &[Knot],
        width: usize,
    ) -> SpiceResult<Wave> {
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
        let mut output = vec![augmented[0].clone()];
        let mut first = 0;
        for time in times.into_iter().chain(std::iter::once(end)) {
            let last = augmented.partition_point(|knot| knot.time < time);
            reduce::reduce(&augmented[first..=last], width, tolerance, &mut output);
            first = last;
        }
        Ok(output)
    }

    pub(super) fn trial(
        &self,
        plan: &LossyPlan,
        history: &ContinuousHistory,
        start: Real,
        end: Real,
        boundary_bytes: usize,
    ) -> SpiceResult<Trial> {
        let ports = plan.launch.len();
        let n = plan.half.len();
        let mut trial = Trial {
            outgoing: vec![Vec::new(); plan.terminal],
            output: Vec::new(),
        };
        let mut at = start;
        let mut substeps = 0_usize;
        while at < end {
            let mut next = (at + self.step).min(end);
            while next - self.step > at {
                next = next.next_down();
            }
            if next <= at || !next.is_finite() {
                return Err(invalid(
                    "CPL internal flight is smaller than representable time",
                ));
            }
            substeps += 1;
            if substeps > plan.limit / std::mem::size_of::<Knot>() {
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
            };
            let mut curves = Vec::with_capacity(plan.terminal);
            let mut bytes = boundary_bytes
                .saturating_add(trial.workspace())
                .saturating_add(plan.terminal * std::mem::size_of::<Wave>());
            for stage in 0..plan.sections {
                let stage_curves = context.stage(stage, plan.limit.saturating_sub(bytes))?;
                let scratch = histories_bytes(&stage_curves);
                for (row, curve) in stage_curves.iter().enumerate() {
                    let compressed =
                        self.compress(history, stage * ports + row, curve, ports + 1)?;
                    bytes = bytes.saturating_add(wave_bytes(&compressed));
                    if bytes.saturating_add(scratch) > plan.limit {
                        return Err(invalid("CPL trial waveform workspace exceeded"));
                    }
                    curves.push(compressed);
                }
            }
            for (node, (wave, curve)) in trial.outgoing.iter_mut().zip(curves).enumerate() {
                if wave.is_empty() {
                    wave.extend(curve);
                } else {
                    wave.extend(curve.into_iter().skip(1));
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

    pub(super) fn accept(
        &self,
        history: &ContinuousHistory,
        trial: Trial,
        start: Real,
        end: Real,
        input: &[Real],
        delta: &[Real],
    ) -> SpiceResult<ContinuousHistory> {
        let mut next = history.clone();
        for (node, (wave, new)) in next.outgoing.iter_mut().zip(trial.outgoing).enumerate() {
            for knot in new.into_iter().filter(|knot| knot.time > start) {
                let value = knot.wave.evaluate(delta);
                if !value.is_finite() {
                    return Err(invalid("nonfinite accepted CPL internal wave"));
                }
                wave.push(Knot {
                    time: knot.time,
                    wave: Affine::scalar(value),
                });
            }
            prune(wave, end - self.delays[node]);
        }
        for (port, wave) in next.boundary.iter_mut().enumerate() {
            let value = input[port] - next.initial_input[port];
            if !value.is_finite() {
                return Err(invalid("nonfinite accepted CPL boundary wave"));
            }
            wave.push(Knot {
                time: end,
                wave: Affine::scalar(value),
            });
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
    boundary: Vec<Wave>,
    outgoing: Vec<Wave>,
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

pub(super) struct Trial {
    outgoing: Vec<Wave>,
    pub(super) output: Vec<Affine>,
}

impl Trial {
    fn workspace(&self) -> usize {
        histories_bytes(&self.outgoing)
    }

    fn sample(
        &self,
        history: &ContinuousHistory,
        node: usize,
        start: Real,
        time: Real,
    ) -> SpiceResult<Affine> {
        let wave = if time <= start {
            &history.outgoing[node]
        } else {
            &self.outgoing[node]
        };
        sample(wave, time, history.origin)
    }
}

struct Step<'a> {
    plan: &'a LossyPlan,
    history: &'a ContinuousHistory,
    trial: &'a Trial,
    start: Real,
    end: Real,
    at: Real,
    next: Real,
}

impl Step<'_> {
    fn source(&self, stage: usize, port: usize) -> (Option<usize>, Real) {
        let ports = self.plan.launch.len();
        let n = self.plan.half.len();
        let mode = port % n;
        let node = if port < n {
            stage
                .checked_sub(1)
                .map(|neighbor| neighbor * ports + n + mode)
        } else {
            (stage + 1 < self.plan.sections).then_some((stage + 1) * ports + mode)
        };
        let delay = if node.is_some() {
            2. * self.plan.half[mode]
        } else {
            self.plan.half[mode]
        };
        (node, delay)
    }

    fn input(&self, port: usize, node: Option<usize>, time: Real) -> SpiceResult<Affine> {
        if let Some(node) = node {
            return self.trial.sample(self.history, node, self.start, time);
        }
        if time <= self.start {
            return sample(&self.history.boundary[port], time, self.history.origin);
        }
        let mut wave = sample(
            &self.history.boundary[port],
            self.start,
            self.history.origin,
        )?;
        let fraction = ((time - self.start) / (self.end - self.start)).clamp(0., 1.);
        wave.set(port + 1, fraction, self.plan.launch.len() + 1);
        Ok(wave)
    }

    fn stage(&self, stage: usize, limit: usize) -> SpiceResult<Vec<Wave>> {
        let ports = self.plan.launch.len();
        let sources: Vec<_> = (0..ports).map(|port| self.source(stage, port)).collect();
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
        if times
            .len()
            .saturating_mul(ports)
            .saturating_mul(std::mem::size_of::<Knot>())
            > limit
        {
            return Err(invalid("CPL stage waveform workspace exceeded"));
        }
        let mut curves: Vec<Wave> = (0..ports)
            .map(|_| Vec::with_capacity(times.len()))
            .collect();
        for time in times {
            let mut output = vec![Affine::zero(); ports];
            for (port, (&(node, delay), column)) in
                sources.iter().zip(&self.plan.scattering).enumerate()
            {
                let input = self.input(port, node, time - delay)?;
                for &(row, gain) in column {
                    output[row].add_scaled(&input, gain);
                }
            }
            for (curve, wave) in curves.iter_mut().zip(output) {
                if !wave.finite() {
                    return Err(invalid("nonfinite CPL stage wave"));
                }
                curve.push(Knot { time, wave });
            }
        }
        if histories_bytes(&curves) > limit {
            return Err(invalid("CPL affine stage waveform workspace exceeded"));
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
            let trial = continuous
                .trial(plan, &history, start, end, history.workspace())
                .unwrap();
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
