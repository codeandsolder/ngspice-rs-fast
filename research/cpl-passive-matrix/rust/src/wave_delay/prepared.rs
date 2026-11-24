//! An internal causal method of steps for arbitrary global timestep sizes.
//! Every intermediate wave is affine in the 2N unknown terminal incident
//! waves at the global endpoint. Preparation is reusable across Newton
//! trials and never mutates accepted history.

use super::{Trace, WaveNetwork, WaveTrial};

mod events;
mod terminal;
pub use terminal::{MnaRhs, MnaTriplet, ModalPortEquations, PhysicalPortEquations, WaveMnaStamp};

#[derive(Debug, Clone)]
struct Affine {
    base: f64,
    jac: Vec<f64>,
}
impl Affine {
    fn constant(base: f64, width: usize) -> Self {
        Self {
            base,
            jac: vec![0.; width],
        }
    }
    fn terminal(column: usize, alpha: f64, offset: f64, width: usize) -> Self {
        let mut result = Self::constant(offset, width);
        result.jac[column] = alpha;
        result
    }
    fn interpolate(lower: &Self, upper: &Self, alpha: f64) -> Self {
        Self {
            base: alpha.mul_add(upper.base - lower.base, lower.base),
            jac: lower
                .jac
                .iter()
                .zip(&upper.jac)
                .map(|(&lo, &hi)| alpha.mul_add(hi - lo, lo))
                .collect(),
        }
    }
    fn add_scaled(&mut self, other: &Self, scale: f64) {
        self.base = scale.mul_add(other.base, self.base);
        for (dst, &src) in self.jac.iter_mut().zip(&other.jac) {
            *dst = scale.mul_add(src, *dst);
        }
    }
    fn evaluate(&self, unknown: &[f64]) -> Result<f64, &'static str> {
        let value = self
            .jac
            .iter()
            .zip(unknown)
            .fold(self.base, |acc, (&j, &v)| j.mul_add(v, acc));
        if !value.is_finite() {
            return Err("nonfinite prepared modal wave");
        }
        Ok(value)
    }
}

#[derive(Debug)]
struct LocalTrace<'a> {
    accepted: &'a Trace,
    initial_time: f64,
    times: Vec<f64>,
    values: Vec<Vec<Affine>>,
}
impl<'a> LocalTrace<'a> {
    const fn new(accepted: &'a Trace, initial_time: f64) -> Self {
        Self {
            accepted,
            initial_time,
            times: vec![],
            values: vec![],
        }
    }
    fn insert(&mut self, time: f64, wave: Vec<Affine>) -> Result<(), &'static str> {
        if time <= self.times.last().copied().unwrap_or(self.initial_time) {
            return Err("nonmonotone prepared internal time");
        }
        self.times.push(time);
        self.values.push(wave);
        Ok(())
    }
    fn mode(&self, time: f64, index: usize, width: usize) -> Result<Affine, &'static str> {
        if !time.is_finite() {
            return Err("invalid retarded time");
        }
        if time < self.accepted.first_possible[index] {
            return Ok(Affine::constant(0., width));
        }
        if time <= self.initial_time {
            return Ok(Affine::constant(self.accepted.value(time, index)?, width));
        }
        let mut lo = 0;
        let mut hi = self.times.len();
        while lo < hi {
            let mid = usize::midpoint(lo, hi);
            if self.times[mid] < time {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo >= self.times.len() {
            return Err("in-step delay needs unresolved future wave");
        }
        if self.times[lo].to_bits() == time.to_bits() {
            return self.values[lo]
                .get(index)
                .cloned()
                .ok_or("missing modal value");
        }
        let (low_time, low_wave) = if lo == 0 {
            (
                self.initial_time,
                Affine::constant(self.accepted.value(self.initial_time, index)?, width),
            )
        } else {
            (self.times[lo - 1], self.values[lo - 1][index].clone())
        };
        let alpha = (time - low_time) / (self.times[lo] - low_time);
        Ok(Affine::interpolate(
            &low_wave,
            &self.values[lo][index],
            alpha,
        ))
    }
    fn retarded(
        &self,
        time: f64,
        delays: &[f64],
        factor: f64,
        width: usize,
    ) -> Result<Vec<Affine>, &'static str> {
        delays
            .iter()
            .enumerate()
            .map(|(index, &delay)| self.mode((-factor).mul_add(delay, time), index, width))
            .collect()
    }
}

#[derive(Debug)]
struct PreparedKnot {
    time: f64,
    near_in: Vec<Affine>,
    far_in: Vec<Affine>,
    near_out: Vec<Affine>,
    far_out: Vec<Affine>,
    static_near: Vec<Vec<Affine>>,
    static_far: Vec<Vec<Affine>>,
}
impl PreparedKnot {
    fn evaluate(
        &self,
        owner_id: u64,
        revision: u64,
        unknown: &[f64],
    ) -> Result<WaveTrial, &'static str> {
        let eval = |formulas: &[Affine]| -> Result<Vec<f64>, &'static str> {
            formulas.iter().map(|f| f.evaluate(unknown)).collect()
        };
        Ok(WaveTrial {
            owner_id,
            revision,
            time: self.time,
            incoming_near: eval(&self.near_in)?,
            incoming_far: eval(&self.far_in)?,
            outgoing_near: eval(&self.near_out)?,
            outgoing_far: eval(&self.far_out)?,
            static_near: self
                .static_near
                .iter()
                .map(|a| eval(a))
                .collect::<Result<Vec<_>, _>>()?,
            static_far: self
                .static_far
                .iter()
                .map(|a| eval(a))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

/// One global timestep prepared by resolving its inner causal delays.
/// The terminal Jacobians are exact for the assumed piecewise-linear
/// endpoint incident waves and their linearly interpolated histories.
#[derive(Debug)]
pub struct PreparedWaveStep {
    owner_id: u64,
    revision: u64,
    modes: usize,
    final_time: f64,
    near_front: Vec<f64>,
    far_front: Vec<f64>,
    knots: Vec<PreparedKnot>,
}

impl PreparedWaveStep {
    #[must_use]
    pub const fn internal_steps(&self) -> usize {
        self.knots.len()
    }

    #[must_use]
    pub fn reflected_jacobian(&self) -> Vec<f64> {
        self.knots.last().map_or_else(Vec::new, |k| {
            k.near_out
                .iter()
                .flat_map(|a| a.jac.iter().copied())
                .collect()
        })
    }
    #[must_use]
    pub fn transmitted_jacobian(&self) -> Vec<f64> {
        self.knots.last().map_or_else(Vec::new, |k| {
            k.far_out
                .iter()
                .flat_map(|a| a.jac.iter().copied())
                .collect()
        })
    }

    /// Evaluate any terminal Newton trial without rerunning the expensive
    /// modal internal propagation.
    ///
    /// # Errors
    /// Nonfinite, wrongly sized or physically premature excitation,
    /// or arithmetic overflow.
    pub fn evaluate(&self, near: &[f64], far: &[f64]) -> Result<WaveBatch, &'static str> {
        if near.len() != self.modes
            || far.len() != self.modes
            || near.iter().chain(far).any(|v| !v.is_finite())
        {
            return Err("invalid prepared exterior wave");
        }
        if near
            .iter()
            .zip(&self.near_front)
            .any(|(&v, &front)| v != 0. && self.final_time < front)
            || far
                .iter()
                .zip(&self.far_front)
                .any(|(&v, &front)| v != 0. && self.final_time < front)
        {
            return Err("prepared exterior excitation before physical front");
        }
        let mut unknown = Vec::with_capacity(2 * self.modes);
        unknown.extend_from_slice(near);
        unknown.extend_from_slice(far);
        let mut samples = Vec::with_capacity(self.knots.len());
        for knot in &self.knots {
            samples.push(knot.evaluate(self.owner_id, self.revision, &unknown)?);
        }
        Ok(WaveBatch {
            owner_id: self.owner_id,
            revision: self.revision,
            samples,
        })
    }
}

/// A complete candidate of all internal knots in one global timestep.
#[derive(Debug)]
pub struct WaveBatch {
    owner_id: u64,
    revision: u64,
    samples: Vec<WaveTrial>,
}
impl WaveBatch {
    #[must_use]
    pub fn reflected_near(&self) -> &[f64] {
        self.samples
            .last()
            .map_or(&[] as &[f64], WaveTrial::reflected_near)
    }
    #[must_use]
    pub fn transmitted_far(&self) -> &[f64] {
        self.samples
            .last()
            .map_or(&[] as &[f64], WaveTrial::transmitted_far)
    }
    #[must_use]
    pub const fn internal_steps(&self) -> usize {
        self.samples.len()
    }
}

fn endpoint_formula(
    trace: &Trace,
    start: f64,
    final_time: f64,
    time: f64,
    column_offset: usize,
    width: usize,
) -> Result<Vec<Affine>, &'static str> {
    let mut result = Vec::with_capacity(trace.first_possible.len());
    for (index, &front) in trace.first_possible.iter().enumerate() {
        if time < front {
            result.push(Affine::constant(0., width));
        } else {
            let anchor = start.max(front);
            let alpha = if final_time <= anchor {
                1.
            } else {
                (time - anchor) / (final_time - anchor)
            };
            let old = if front <= start {
                trace.value(start, index)?
            } else {
                0.
            };
            result.push(Affine::terminal(
                column_offset + index,
                alpha,
                (1. - alpha) * old,
                width,
            ));
        }
    }
    Ok(result)
}

impl WaveNetwork {
    /// Prepare the internal causal propagation for a global timestep.
    /// The circuit Newton solver may call evaluate many times on the
    /// same immutable affine response with different terminal waves.
    /// No accepted history is mutated at preparation time.
    ///
    /// # Errors
    /// Nonrepresentable time intervals, insufficient allocation,
    /// missing accepted history or nonfinite propagated coefficients.
    pub fn prepare_step(&self, final_time: f64) -> Result<PreparedWaveStep, &'static str> {
        if !final_time.is_finite() || final_time <= self.time {
            return Err("global timestep must advance representable time");
        }
        let first_near = self.input_near.first_possible.clone();
        let first_far = self.input_far.first_possible.clone();
        let width = 2 * self.modes;
        let times = self.prepared_event_times(final_time)?;
        let total = times.len();
        let mut local_near: Vec<LocalTrace<'_>> = self
            .output_near
            .iter()
            .map(|t| LocalTrace::new(t, self.time))
            .collect();
        let mut local_far: Vec<LocalTrace<'_>> = self
            .output_far
            .iter()
            .map(|t| LocalTrace::new(t, self.time))
            .collect();
        let mut launch_near = LocalTrace::new(&self.input_near, self.time);
        let mut launch_far = LocalTrace::new(&self.input_far, self.time);
        let mut knots = Vec::new();
        knots
            .try_reserve_exact(total)
            .map_err(|_| "causal internal wave allocation failed")?;
        for time in times {
            let incoming_near =
                endpoint_formula(&self.input_near, self.time, final_time, time, 0, width)?;
            let incoming_far = endpoint_formula(
                &self.input_far,
                self.time,
                final_time,
                time,
                self.modes,
                width,
            )?;
            let mut static_near = Vec::with_capacity(self.stages);
            let mut static_far = Vec::with_capacity(self.stages);
            // All current-time scatter matrices are computed solely
            // from earlier retarded waves, then the entire current
            // timestep is inserted as the next local accepted knot.
            for stage in 0..self.stages {
                let mut arriving = if stage == 0 {
                    launch_near.retarded(time, &self.half_delays, 1., width)?
                } else {
                    local_far[stage - 1].retarded(time, &self.half_delays, 2., width)?
                };
                arriving.extend(if stage + 1 == self.stages {
                    launch_far.retarded(time, &self.half_delays, 1., width)?
                } else {
                    local_near[stage + 1].retarded(time, &self.half_delays, 2., width)?
                });
                let mut leaving = Vec::with_capacity(width);
                for row in 0..width {
                    let mut result = Affine::constant(0., width);
                    for (col, source) in arriving.iter().enumerate() {
                        result.add_scaled(source, self.static_scattering[row * width + col]);
                    }
                    if !result.base.is_finite() || result.jac.iter().any(|v| !v.is_finite()) {
                        return Err("nonfinite prepared scattering coefficients");
                    }
                    leaving.push(result);
                }
                static_near.push(leaving[..self.modes].to_vec());
                static_far.push(leaving[self.modes..].to_vec());
            }
            for stage in 0..self.stages {
                local_near[stage].insert(time, static_near[stage].clone())?;
                local_far[stage].insert(time, static_far[stage].clone())?;
            }
            launch_near.insert(time, incoming_near.clone())?;
            launch_far.insert(time, incoming_far.clone())?;
            let returned_near = local_near[0].retarded(time, &self.half_delays, 1., width)?;
            let transmitted_far =
                local_far[self.stages - 1].retarded(time, &self.half_delays, 1., width)?;
            knots.push(PreparedKnot {
                time,
                near_in: incoming_near,
                far_in: incoming_far,
                near_out: returned_near,
                far_out: transmitted_far,
                static_near,
                static_far,
            });
        }
        Ok(PreparedWaveStep {
            owner_id: self.owner_id,
            revision: self.revision,
            modes: self.modes,
            final_time,
            near_front: first_near,
            far_front: first_far,
            knots,
        })
    }

    fn prepared_event_times(&self, final_time: f64) -> Result<Vec<f64>, &'static str> {
        let interval = final_time - self.time;
        let shortest = self
            .half_delays
            .iter()
            .copied()
            .reduce(f64::min)
            .ok_or("no modal propagation delays")?;
        let rounded = (interval / shortest * 1.01).ceil();
        let count = positive_integer_f64_to_usize(rounded)
            .ok_or("unrepresentable internal propagation step count")?;
        let mut times = Vec::new();
        times
            .try_reserve_exact(count)
            .map_err(|_| "causal internal time allocation failed")?;
        let denominator = exact_usize_to_f64(count)?;
        for index in 1..=count {
            let time = if index == count {
                final_time
            } else {
                let fraction = exact_usize_to_f64(index)? / denominator;
                interval.mul_add(fraction, self.time)
            };
            if time <= times.last().copied().unwrap_or(self.time) {
                return Err("internal substep below floating-point time resolution");
            }
            times.push(time);
        }
        let extra = events::propagated_breakpoints(self, final_time);
        times
            .try_reserve(extra.len())
            .map_err(|_| "causal breakpoint scheduling allocation failed")?;
        times.extend(extra);
        times.sort_by(f64::total_cmp);
        times.dedup_by(|a, b| a.to_bits() == b.to_bits());
        Ok(times)
    }

    /// Atomically commit the entire internally resolved global step,
    /// retaining all physical-delay interpolation knots. No partial
    /// wave history or energy is committed on an invalid batch.
    ///
    /// # Errors
    /// Stale/foreign state, invalid arrays or time, overflow, or
    /// failed history capacity pre-reservation.
    pub fn accept_prepared(&mut self, batch: WaveBatch) -> Result<(), &'static str> {
        if batch.owner_id != self.owner_id || batch.revision != self.revision {
            return Err("foreign or stale prepared coupled line wave");
        }
        if batch.samples.is_empty() {
            return Err("empty prepared modal wave transaction");
        }
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or("modal wave accepted revision exhausted")?;
        let mut input_energy = self.input_energy;
        let mut output_energy = self.output_energy;
        let mut previous_time = self.time;
        for sample in &batch.samples {
            if sample.owner_id != self.owner_id
                || sample.revision != self.revision
                || !sample.time.is_finite()
                || sample.time <= previous_time
                || sample.static_near.len() != self.stages
                || sample.static_far.len() != self.stages
            {
                return Err("malformed prepared modal acceptance ordering");
            }
            for wave in sample.static_near.iter().chain(&sample.static_far).chain([
                &sample.incoming_near,
                &sample.incoming_far,
                &sample.outgoing_near,
                &sample.outgoing_far,
            ]) {
                if wave.len() != self.modes || wave.iter().any(|v| !v.is_finite()) {
                    return Err("invalid prepared modal wave dimension or value");
                }
            }
            let squared = |wave: &[f64]| wave.iter().map(|v| v * v).sum::<f64>();
            let dt = sample.time - previous_time;
            input_energy = dt.mul_add(
                squared(&sample.incoming_near) + squared(&sample.incoming_far),
                input_energy,
            );
            output_energy = dt.mul_add(
                squared(&sample.outgoing_near) + squared(&sample.outgoing_far),
                output_energy,
            );
            if !input_energy.is_finite() || !output_energy.is_finite() {
                return Err("prepared modal energy integration overflow");
            }
            previous_time = sample.time;
        }
        let count = batch.samples.len();
        for trace in self
            .output_near
            .iter_mut()
            .chain(&mut self.output_far)
            .chain([&mut self.input_near, &mut self.input_far])
        {
            trace.reserve_more(count)?;
        }
        // No fallible operations remain between here and complete commit.
        for step in batch.samples {
            for (trace, wave) in self.output_near.iter_mut().zip(step.static_near) {
                trace.push(step.time, wave);
            }
            for (trace, wave) in self.output_far.iter_mut().zip(step.static_far) {
                trace.push(step.time, wave);
            }
            self.input_near.push(step.time, step.incoming_near);
            self.input_far.push(step.time, step.incoming_far);
        }
        self.time = previous_time;
        self.revision = next_revision;
        self.input_energy = input_energy;
        self.output_energy = output_energy;
        if self.prune_histories {
            let maximum = self
                .half_delays
                .iter()
                .copied()
                .reduce(f64::max)
                .ok_or("no modal propagation delay")?;
            let oldest_needed = (-2.0_f64).mul_add(maximum, previous_time);
            for trace in self
                .output_near
                .iter_mut()
                .chain(&mut self.output_far)
                .chain([&mut self.input_near, &mut self.input_far])
            {
                trace.prune(oldest_needed);
            }
        }
        Ok(())
    }
}

fn positive_integer_f64_to_usize(value: f64) -> Option<usize> {
    if !value.is_finite() || value < 1. || value > 9_007_199_254_740_992. {
        return None;
    }
    let bits = value.to_bits();
    let exponent = i32::try_from((bits >> 52) & 0x7ff).ok()? - 1023;
    if exponent < 0 {
        return None;
    }
    let mantissa = u128::from((bits & ((1_u64 << 52) - 1)) | (1_u64 << 52));
    let integer = if exponent >= 52 {
        mantissa.checked_shl(u32::try_from(exponent - 52).ok()?)?
    } else {
        mantissa >> u32::try_from(52 - exponent).ok()?
    };
    usize::try_from(integer).ok()
}
fn exact_usize_to_f64(index: usize) -> Result<f64, &'static str> {
    let numeric = u64::try_from(index).map_err(|_| "index wider than u64")?;
    if numeric > 9_007_199_254_740_992 {
        return Err("internal time subdivision exceeds exact f64 indexing");
    }
    let high = u32::try_from(numeric >> 32).map_err(|_| "high index overflow")?;
    let low = u32::try_from(numeric & 0xffff_ffff).map_err(|_| "low index overflow")?;
    Ok(f64::from(high).mul_add(4_294_967_296., f64::from(low)))
}

#[cfg(test)]
#[path = "prepared_tests.rs"]
mod prepared_tests;
