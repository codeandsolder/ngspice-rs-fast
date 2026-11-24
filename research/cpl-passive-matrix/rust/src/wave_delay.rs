//! Exact-front multiconductor modal delay scattering with accepted histories.
//!
//! A passive reciprocal static scattering matrix couples the modal waves
//! across each R/G section. Neighboring sections are connected by exact,
//! mode-specific LC propagation delays. This module is self-contained
//! research, not yet an implementation of the native simulator Device.
//!
//! Each accepted wave trace retains the interpolation bracket needed by
//! subsequent delayed samples, while physical wavefront metadata prevents
//! interpolation from anticipating a signal before its first possible arrival.

use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, VecDeque};

use crate::next_port_id;

/// A frozen scattering network trial; only its owning current revision can
/// accept it. All computed wave arrays are validated before any mutation.
#[derive(Debug)]
pub struct WaveTrial {
    owner_id: u64,
    revision: u64,
    time: f64,
    incoming_near: Vec<f64>,
    incoming_far: Vec<f64>,
    outgoing_near: Vec<f64>,
    outgoing_far: Vec<f64>,
    static_near: Vec<Vec<f64>>,
    static_far: Vec<Vec<f64>>,
}

impl WaveTrial {
    /// Power-normalized reflected waves seen at the near exterior port.
    #[must_use]
    pub fn reflected_near(&self) -> &[f64] {
        &self.outgoing_near
    }

    /// Power-normalized waves leaving the far exterior port.
    #[must_use]
    pub fn transmitted_far(&self) -> &[f64] {
        &self.outgoing_far
    }
}

#[derive(Debug)]
struct Trace {
    times: VecDeque<f64>,
    samples: VecDeque<Vec<f64>>,
    first_possible: Vec<f64>,
}

impl Trace {
    fn new(first_possible: Vec<f64>) -> Self {
        let width = first_possible.len();
        Self {
            times: VecDeque::from([0.]),
            samples: VecDeque::from([vec![0.; width]]),
            first_possible,
        }
    }

    fn value(&self, time: f64, mode: usize) -> Result<f64, &'static str> {
        if mode >= self.first_possible.len() || !time.is_finite() {
            return Err("invalid modal wave history request");
        }
        if time < self.first_possible[mode] || time < 0. {
            return Ok(0.);
        }
        let latest = self
            .times
            .back()
            .copied()
            .ok_or("empty accepted wave history")?;
        if time > latest {
            return Err("future modal history sample requires timestep truncation");
        }
        let oldest = self
            .times
            .front()
            .copied()
            .ok_or("missing accepted modal history")?;
        if time < oldest {
            return Err("required wave interpolation bracket was pruned");
        }

        // Lower bound of the first accepted sample strictly greater than
        // requested time. VecDeque can be wrapped; logical get() still
        // permits correct bounded binary search without copying a slice.
        let mut start = 0;
        let mut end = self.times.len();
        while start < end {
            let mid = usize::midpoint(start, end);
            if self.times.get(mid).copied().ok_or("history index error")? <= time {
                start = mid + 1;
            } else {
                end = mid;
            }
        }
        let lower = start
            .checked_sub(1)
            .ok_or("missing earlier history bracket")?;
        let y0 = self
            .samples
            .get(lower)
            .and_then(|v| v.get(mode))
            .copied()
            .ok_or("missing accepted modal value")?;
        if start == self.times.len() {
            return Ok(y0);
        }
        let t0 = self.times.get(lower).copied().ok_or("history time lost")?;
        let t1 = self.times.get(start).copied().ok_or("history time lost")?;
        let y1 = self
            .samples
            .get(start)
            .and_then(|v| v.get(mode))
            .copied()
            .ok_or("history modal component lost")?;
        let alpha = (time - t0) / (t1 - t0);
        let value = (1. - alpha).mul_add(y0, alpha * y1);
        if !value.is_finite() {
            return Err("delayed interpolation overflowed");
        }
        Ok(value)
    }

    fn prune(&mut self, first_needed: f64) {
        while self.times.len() > 1 && self.times.get(1).is_some_and(|&t| t <= first_needed) {
            self.times.pop_front();
            self.samples.pop_front();
        }
    }

    fn reserve_more(&mut self, count: usize) -> Result<(), &'static str> {
        self.times
            .try_reserve(count)
            .map_err(|_| "wave timestamp allocation failed")?;
        self.samples
            .try_reserve(count)
            .map_err(|_| "wave sample allocation failed")
    }

    fn reserve(&mut self) -> Result<(), &'static str> {
        self.times
            .try_reserve(1)
            .map_err(|_| "wave history timestamp reserve failed")?;
        self.samples
            .try_reserve(1)
            .map_err(|_| "wave history sample reserve failed")
    }

    fn push(&mut self, t: f64, value: Vec<f64>) {
        self.times.push_back(t);
        self.samples.push_back(value);
    }
}

/// Heap node ordered by first physical arrival time, then output id.
#[derive(Debug, Clone, Copy)]
struct Arrival {
    time: f64,
    node: usize,
}
impl PartialEq for Arrival {
    fn eq(&self, other: &Self) -> bool {
        self.time.to_bits() == other.time.to_bits() && self.node == other.node
    }
}
impl Eq for Arrival {}
impl PartialOrd for Arrival {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Arrival {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time
            .total_cmp(&other.time)
            .then_with(|| self.node.cmp(&other.node))
    }
}

fn matrix_is_passive(scattering: &[f64], width: usize) -> Result<(), &'static str> {
    let tol = 2e-10;
    for row in 0..width {
        for col in 0..row {
            if (scattering[row * width + col] - scattering[col * width + row]).abs() > tol {
                return Err("static coupled scattering must be reciprocal");
            }
        }
    }
    let mut gram = vec![0.; width * width];
    for row in 0..width {
        for col in row..width {
            let value = (0..width)
                .map(|k| scattering[k * width + row] * scattering[k * width + col])
                .sum::<f64>();
            if !value.is_finite() {
                return Err("static scattering Gram matrix overflow");
            }
            gram[row * width + col] = value;
            gram[col * width + row] = value;
        }
    }
    // Bounded symmetric Jacobi decomposition of SᵀS. Its eigenvalues
    // are the squared singular values, hence ≤1 iff power contractive.
    // Scale is ≤1 for a truly passive matrix, so fixed absolute
    // convergence bounds are meaningful here.
    let iterations = width
        .checked_mul(width)
        .and_then(|n| n.checked_mul(100))
        .ok_or("static scattering matrix eigensolver work overflow")?;
    let mut converged = width <= 1;
    for _ in 0..iterations {
        let mut best = 0.;
        let (mut pivot, mut partner) = (0, 0);
        for i in 0..width {
            for j in i + 1..width {
                let off = gram[i * width + j].abs();
                if off > best {
                    best = off;
                    pivot = i;
                    partner = j;
                }
            }
        }
        if best <= 1e-13 {
            converged = true;
            break;
        }
        let app = gram[pivot * width + pivot];
        let aqq = gram[partner * width + partner];
        let apq = gram[pivot * width + partner];
        let tau = (aqq - app) / (2. * apq);
        let rotation = if tau == 0. {
            1.
        } else {
            tau.signum() / (tau.abs() + (tau.mul_add(tau, 1.)).sqrt())
        };
        let cosine = 1. / (rotation.mul_add(rotation, 1.)).sqrt();
        let sin = cosine * rotation;
        for k in 0..width {
            if k == pivot || k == partner {
                continue;
            }
            let kp = gram[k * width + pivot];
            let kq = gram[k * width + partner];
            let new_p = cosine.mul_add(kp, -sin * kq);
            let new_q = sin.mul_add(kp, cosine * kq);
            gram[k * width + pivot] = new_p;
            gram[pivot * width + k] = new_p;
            gram[k * width + partner] = new_q;
            gram[partner * width + k] = new_q;
        }
        gram[pivot * width + pivot] = rotation.mul_add(-apq, app);
        gram[partner * width + partner] = rotation.mul_add(apq, aqq);
        gram[pivot * width + partner] = 0.;
        gram[partner * width + pivot] = 0.;
    }
    if !converged {
        return Err("static coupled-scattering eigenvalue calculation did not converge");
    }
    if (0..width).any(|i| gram[i * width + i] > 1. + tol) {
        return Err("active static scattering matrix violates power passivity");
    }
    Ok(())
}

fn shortest_wavefronts(
    modes: usize,
    stages: usize,
    static_scattering: &[f64],
    half_delay: &[f64],
    near_front: &[f64],
    far_front: &[f64],
) -> Vec<f64> {
    let width = modes * 2;
    let nodes = width * stages;
    let mut edges = vec![Vec::<(usize, f64)>::new(); nodes];
    let mut fronts = vec![f64::INFINITY; nodes];
    for stage in 0..stages {
        for output in 0..width {
            let destination = stage * width + output;
            for side in 0..2 {
                for mode in 0..modes {
                    let input = side * modes + mode;
                    if static_scattering[output * width + input] == 0. {
                        continue;
                    }
                    let half = half_delay[mode];
                    if side == 0 {
                        if stage == 0 {
                            fronts[destination] = fronts[destination].min(near_front[mode] + half);
                        } else {
                            let predecessor = (stage - 1) * width + modes + mode;
                            edges[predecessor].push((destination, 2. * half));
                        }
                    } else if stage + 1 == stages {
                        fronts[destination] = fronts[destination].min(far_front[mode] + half);
                    } else {
                        let predecessor = (stage + 1) * width + mode;
                        edges[predecessor].push((destination, 2. * half));
                    }
                }
            }
        }
    }
    let mut heap = BinaryHeap::new();
    for (node, &time) in fronts.iter().enumerate() {
        if time.is_finite() {
            heap.push(Reverse(Arrival { time, node }));
        }
    }
    while let Some(Reverse(Arrival { time, node })) = heap.pop() {
        if time > fronts[node] {
            continue;
        }
        for &(next, delta) in &edges[node] {
            let candidate = time + delta;
            if candidate < fronts[next] {
                fronts[next] = candidate;
                heap.push(Reverse(Arrival {
                    time: candidate,
                    node: next,
                }));
            }
        }
    }
    fronts
}

/// One physically passive N-mode lossy split network. Only accepted
/// histories can be read during trial evaluation. Mutable state has a
/// unique owner identity and cannot be cloned into independent devices.
#[derive(Debug)]
pub struct WaveNetwork {
    owner_id: u64,
    modes: usize,
    stages: usize,
    half_delays: Vec<f64>,
    static_scattering: Vec<f64>,
    output_near: Vec<Trace>,
    output_far: Vec<Trace>,
    input_near: Trace,
    input_far: Trace,
    time: f64,
    revision: u64,
    prune_histories: bool,
    input_energy: f64,
    output_energy: f64,
}

impl WaveNetwork {
    /// Construct a network with arbitrary PSD R/G cross-mode scattering
    /// already expressed as one power-normalized real symmetric matrix,
    /// and positive per-mode half-cell delays.
    ///
    /// Source front time may be infinity when the boundary is unexcited.
    /// Matrix passivity is checked by the singular values of its full
    /// 2N×2N scattering matrix, not a subset of diagonal conductors.
    ///
    /// # Errors
    /// Invalid dimensions, zero delay, nonpassive/active scattering,
    /// nonreciprocity, nonfinite inputs or resource exhaustion.
    pub fn new(
        half_delays: Vec<f64>,
        stages: usize,
        scattering: Vec<f64>,
        near_front: Vec<f64>,
        far_front: Vec<f64>,
        prune_histories: bool,
    ) -> Result<Self, &'static str> {
        let modes = half_delays.len();
        let width = modes.checked_mul(2).ok_or("modal port width overflow")?;
        let slots = width
            .checked_mul(stages)
            .ok_or("wavefront graph size overflow")?;
        if modes == 0 || stages == 0 || slots == 0 || stages > 4096 || modes > 32 {
            return Err("invalid mode or passive stage count");
        }
        if scattering.len() != width.checked_mul(width).ok_or("scattering size overflow")?
            || near_front.len() != modes
            || far_front.len() != modes
        {
            return Err("invalid passive coupled-scattering matrix layout");
        }
        if half_delays.iter().any(|t| !t.is_finite() || *t <= 0.)
            || near_front
                .iter()
                .chain(&far_front)
                .any(|t| t.is_nan() || *t < 0.)
            || scattering.iter().any(|x| !x.is_finite())
        {
            return Err("invalid physical delay, source start or static scattering");
        }
        if half_delays.iter().any(|t| !(*t * 2.).is_finite()) {
            return Err("modal propagation delay overflows");
        }
        matrix_is_passive(&scattering, width)?;
        let fronts = shortest_wavefronts(
            modes,
            stages,
            &scattering,
            &half_delays,
            &near_front,
            &far_front,
        );
        let mut output_near = Vec::with_capacity(stages);
        let mut output_far = Vec::with_capacity(stages);
        for stage in 0..stages {
            let start = stage * width;
            output_near.push(Trace::new(fronts[start..start + modes].to_vec()));
            output_far.push(Trace::new(fronts[start + modes..start + width].to_vec()));
        }
        Ok(Self {
            owner_id: next_port_id()?,
            modes,
            stages,
            half_delays,
            static_scattering: scattering,
            output_near,
            output_far,
            input_near: Trace::new(near_front),
            input_far: Trace::new(far_front),
            time: 0.,
            revision: 0,
            prune_histories,
            input_energy: 0.,
            output_energy: 0.,
        })
    }

    fn delayed(
        &self,
        trace: &Trace,
        time: f64,
        double_delay: bool,
    ) -> Result<Vec<f64>, &'static str> {
        let mut result = Vec::with_capacity(self.modes);
        for (mode, &half) in self.half_delays.iter().enumerate() {
            let delay = if double_delay { 2. * half } else { half };
            result.push(trace.value(time - delay, mode)?);
        }
        Ok(result)
    }

    /// Read accepted histories for a complete, nonmutating trial. The
    /// timestep must not exceed the smallest half-stage propagation
    /// delay, preventing the required delayed sample from being in
    /// an unaccepted future timestep.
    ///
    /// # Errors
    /// Unsupported timestep, invalid waves or missing accepted history.
    pub fn trial(
        &self,
        time: f64,
        incident_near: &[f64],
        incident_far: &[f64],
    ) -> Result<WaveTrial, &'static str> {
        if !time.is_finite() || time <= self.time {
            return Err("trial time must be finite and ahead of last accepted time");
        }
        let dt = time - self.time;
        let shortest = self
            .half_delays
            .iter()
            .copied()
            .reduce(f64::min)
            .ok_or("empty line propagation delay")?;
        if dt > shortest * (1. + 1e-12) {
            return Err("modal delay requires a shorter trial timestep");
        }
        if incident_near.len() != self.modes
            || incident_far.len() != self.modes
            || incident_near
                .iter()
                .chain(incident_far)
                .any(|v| !v.is_finite())
        {
            return Err("invalid exterior incident modal wave");
        }
        for (trace, incident) in [
            (&self.input_near, incident_near),
            (&self.input_far, incident_far),
        ] {
            if incident
                .iter()
                .enumerate()
                .any(|(mode, &wave)| wave != 0. && time < trace.first_possible[mode])
            {
                return Err("external excitation precedes its declared wavefront");
            }
        }
        let mut static_near = Vec::with_capacity(self.stages);
        let mut static_far = Vec::with_capacity(self.stages);
        let width = 2 * self.modes;
        for stage in 0..self.stages {
            let left = if stage == 0 {
                self.delayed(&self.input_near, time, false)?
            } else {
                self.delayed(&self.output_far[stage - 1], time, true)?
            };
            let right = if stage + 1 == self.stages {
                self.delayed(&self.input_far, time, false)?
            } else {
                self.delayed(&self.output_near[stage + 1], time, true)?
            };
            let mut input = left;
            input.extend(right);
            let mut output = vec![0.; width];
            for (row, value) in output.iter_mut().enumerate() {
                for (col, &sample) in input.iter().enumerate() {
                    *value = self.static_scattering[row * width + col].mul_add(sample, *value);
                }
                if !value.is_finite() {
                    return Err("static coupled-scattering trial overflow");
                }
            }
            static_near.push(output[..self.modes].to_vec());
            static_far.push(output[self.modes..].to_vec());
        }
        Ok(WaveTrial {
            owner_id: self.owner_id,
            revision: self.revision,
            time,
            incoming_near: incident_near.to_vec(),
            incoming_far: incident_far.to_vec(),
            outgoing_near: self.delayed(&self.output_near[0], time, false)?,
            outgoing_far: self.delayed(&self.output_far[self.stages - 1], time, false)?,
            static_near,
            static_far,
        })
    }

    fn validate_trial(&self, trial: &WaveTrial) -> Result<(f64, f64), &'static str> {
        if trial.revision != self.revision || trial.owner_id != self.owner_id {
            return Err("stale or foreign modal wave transaction");
        }
        if !trial.time.is_finite() || trial.time <= self.time {
            return Err("invalid accepted modal wave time");
        }
        if trial.static_near.len() != self.stages || trial.static_far.len() != self.stages {
            return Err("invalid accepted scattering stage count");
        }
        for sample in trial.static_near.iter().chain(&trial.static_far).chain([
            &trial.incoming_near,
            &trial.incoming_far,
            &trial.outgoing_near,
            &trial.outgoing_far,
        ]) {
            if sample.len() != self.modes || sample.iter().any(|v| !v.is_finite()) {
                return Err("malformed or nonfinite accepted modal wave");
            }
        }
        let squared = |wave: &[f64]| wave.iter().map(|v| v * v).sum::<f64>();
        let dt = trial.time - self.time;
        let energy_in = dt.mul_add(
            squared(&trial.incoming_near) + squared(&trial.incoming_far),
            self.input_energy,
        );
        let energy_out = dt.mul_add(
            squared(&trial.outgoing_near) + squared(&trial.outgoing_far),
            self.output_energy,
        );
        if !energy_in.is_finite() || !energy_out.is_finite() {
            return Err("accepted modal scattering energy overflow");
        }
        Ok((energy_in, energy_out))
    }

    /// Atomically accept a complete candidate. All validation and
    /// fallible history capacity reservations happen before any accepted
    /// time, wave or energy is mutated.
    ///
    /// # Errors
    /// Rejected/stale/foreign candidate or history allocation failure.
    pub fn accept(&mut self, trial: WaveTrial) -> Result<(), &'static str> {
        let (new_input_energy, new_output_energy) = self.validate_trial(&trial)?;
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or("wave revision exhausted")?;
        let maximum_delay = self
            .half_delays
            .iter()
            .copied()
            .reduce(f64::max)
            .ok_or("empty modal delays during pruning")?;
        for trace in self
            .output_near
            .iter_mut()
            .chain(&mut self.output_far)
            .chain([&mut self.input_near, &mut self.input_far])
        {
            trace.reserve()?;
        }
        let t = trial.time;
        for (trace, wave) in self.output_near.iter_mut().zip(trial.static_near) {
            trace.push(t, wave);
        }
        for (trace, wave) in self.output_far.iter_mut().zip(trial.static_far) {
            trace.push(t, wave);
        }
        self.input_near.push(t, trial.incoming_near);
        self.input_far.push(t, trial.incoming_far);
        self.time = t;
        self.revision = next_revision;
        self.input_energy = new_input_energy;
        self.output_energy = new_output_energy;
        if self.prune_histories {
            let first_needed = (-2.0_f64).mul_add(maximum_delay, t);
            for trace in self
                .output_near
                .iter_mut()
                .chain(&mut self.output_far)
                .chain([&mut self.input_near, &mut self.input_far])
            {
                trace.prune(first_needed);
            }
        }
        Ok(())
    }

    /// Current total number of accepted wave timestamps across all traces.
    #[must_use]
    pub fn retained_samples(&self) -> usize {
        self.output_near
            .iter()
            .chain(&self.output_far)
            .chain([&self.input_near, &self.input_far])
            .map(|trace| trace.times.len())
            .sum()
    }

    /// Accepted passive power-wave energy integrals (for conservation tests).
    #[must_use]
    pub const fn energies(&self) -> (f64, f64) {
        (self.input_energy, self.output_energy)
    }
}

mod native_modal;
pub use native_modal::{
    NativeModalCoefficients, PhysicalModalLine, PreparedPhysicalModalLine, WaveDiscretization,
};

mod static_pi;
pub use static_pi::coupled_static_rg_pi;

mod prepared;
pub use prepared::{
    MnaRhs, MnaTriplet, ModalPortEquations, PhysicalPortEquations, PreparedWaveStep, WaveBatch,
    WaveMnaStamp,
};

#[cfg(test)]
mod tests;
