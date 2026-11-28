//! Resolved prepared source forcing with exact external limits.
//!
//! The producer owns symbolic source semantics. This module consumes its
//! validated physical knots without applying native PULSE or PWL defaults.
use std::sync::Arc;

use model_normalize_ngspice::waveform::{
    ExponentialForcing, Forcing, Knot, Repeat, ResolvedWaveform,
};

use crate::devices::functions::{affine_value, scaled_interval_difference};
use crate::devices::pulse::MAX_CYCLE;
use crate::devices::{Limit, WaveformEvent, WaveformEventKind};
use crate::maths::taylor::Jet;
use crate::primitives::{Real, SpiceError, SpiceResult};

fn invalid(message: &str) -> SpiceError {
    SpiceError::circuit(format!("prepared waveform: {message}"))
}

/// Immutable occurrence-bound forcing and its independent DC bias.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedForcing {
    dc: Real,
    idle_in_pop: bool,
    law: Law,
}

#[derive(Debug, Clone, PartialEq)]
enum Law {
    Pwl(SidedPwl),
    Sine(SignedSine),
    Exponential(ExponentialForcing),
}

#[derive(Debug, Clone, PartialEq)]
struct SidedPwl {
    knots: Arc<[Knot]>,
    repeat: Option<Repeat>,
    min_spacing: Real,
}

#[derive(Clone, Copy)]
enum CycleOrHold {
    Cycle(Real),
    Hold(Real),
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SignedSine {
    offset: Real,
    amplitude: Real,
    frequency: Real,
    delay: Real,
    damping: Real,
    hold: bool,
    phase: Real,
    stop_after: Option<Real>,
}

impl PreparedForcing {
    /// Consume the shared producer representation, validating its numerical
    /// invariants at the engine boundary before any source is stamped.
    /// # Errors
    /// Invalid ordinates, geometry, period, or sine coefficients.
    pub fn new(resolved: ResolvedWaveform) -> SpiceResult<Self> {
        if !resolved.dc.is_finite() || !resolved.startup.is_finite() {
            return Err(invalid("nonfinite DC bias or transient startup"));
        }
        let law = match resolved.forcing {
            Forcing::PiecewiseLinear { knots, repeat } => Law::Pwl(SidedPwl::new(knots, repeat)?),
            Forcing::Exponential(exp) => {
                if ![
                    exp.initial,
                    exp.pulsed,
                    exp.rise_delay,
                    exp.rise_tau,
                    exp.fall_delay,
                    exp.fall_tau,
                ]
                .into_iter()
                .all(Real::is_finite)
                    || exp.rise_tau <= 0.
                    || exp.fall_tau <= 0.
                {
                    return Err(invalid("invalid exponential coefficients"));
                }
                Law::Exponential(exp)
            }
            Forcing::Sine {
                offset,
                amplitude,
                frequency,
                delay,
                damping,
                hold_before_delay,
                phase,
                stop_after,
            } => {
                if ![offset, amplitude, frequency, delay, damping, phase]
                    .iter()
                    .all(|v| v.is_finite())
                    || stop_after
                        .is_some_and(|stop| !stop.is_finite() || !(delay + stop).is_finite())
                {
                    return Err(invalid("invalid sine coefficients"));
                }
                Law::Sine(SignedSine {
                    offset,
                    amplitude,
                    frequency,
                    delay,
                    damping,
                    hold: hold_before_delay,
                    phase: phase.to_radians(),
                    stop_after,
                })
            }
        };
        Ok(Self {
            dc: resolved.dc,
            idle_in_pop: resolved.idle_in_pop,
            law,
        })
    }

    /// Source-defined DC bias, not the left-hand transient limit at zero.
    #[must_use = "the returned bias is independent of transient forcing"]
    pub const fn dc(&self) -> Real {
        self.dc
    }

    /// Whether periodic operating-point preparation freezes this source.
    #[must_use = "the result determines whether a periodic analysis freezes the source"]
    pub const fn idle_in_pop(&self) -> bool {
        self.idle_in_pop
    }

    /// Whether affine endpoint interpolation is exact between events.
    #[must_use = "the result selects affine or continuous forcing evaluation"]
    pub const fn is_piecewise_linear(&self) -> bool {
        matches!(self.law, Law::Pwl(_))
    }

    /// Evaluate the requested external limit without modifying event state.
    /// # Errors
    /// Nonfinite or unresolvable time or forcing.
    pub fn value_at(&self, time: Real, limit: Limit) -> SpiceResult<Real> {
        if !time.is_finite() {
            return Err(invalid("nonfinite time"));
        }
        let value = match &self.law {
            Law::Pwl(pwl) => pwl.value_at(time, limit)?,
            Law::Sine(sine) => sine.value_at(time, limit),
            Law::Exponential(exp) => exponential_value(*exp, time),
        };
        if !value.is_finite() {
            return Err(invalid("nonfinite forcing"));
        }
        Ok(value)
    }

    /// Analytic coefficients in increasing absolute time, with the selected
    /// one-sided segment even at discontinuities and repeat boundaries.
    pub(super) fn jet(
        &self,
        time: Real,
        scale: Real,
        order: usize,
        limit: Limit,
    ) -> SpiceResult<Jet> {
        Ok(match &self.law {
            Law::Pwl(pwl) => {
                let (value, slope) = pwl.local_affine(time, scale, limit)?;
                Jet::affine(order, value, slope)
            }
            Law::Sine(sine) => sine
                .jet(time, scale, order, limit)
                .with_value(self.value_at(time, limit)?),
            Law::Exponential(exp) => exponential_jet(*exp, time, scale, order, limit)
                .with_value(self.value_at(time, limit)?),
        })
    }

    /// Lazy event sequence in the closed time window. Both source kinds use
    /// exactly the same event times as the external-limit evaluator.
    /// # Errors
    /// Invalid window or unrepresentable periodic geometry.
    pub fn breakpoints_in(&self, start: Real, end: Real) -> SpiceResult<PreparedEvents> {
        if !start.is_finite() || !end.is_finite() || start > end {
            return Err(invalid("invalid event window"));
        }
        let events = match &self.law {
            Law::Pwl(pwl) => Events::Pwl(pwl.events(start, end)?),
            Law::Sine(sine) => Events::Finite(FiniteEvents::new(
                [
                    sine.hold.then_some(sine.delay),
                    sine.stop_after
                        .filter(|stop| !sine.hold || *stop > 0.)
                        .map(|stop| sine.delay + stop),
                ],
                start,
                end,
            )),
            Law::Exponential(exp) => Events::Finite(FiniteEvents::new(
                [Some(exp.rise_delay), Some(exp.fall_delay)],
                start,
                end,
            )),
        };
        Ok(PreparedEvents(events))
    }
}

impl SignedSine {
    fn coordinate(self, time: Real, limit: Limit) -> (Real, bool) {
        if self.hold && (time < self.delay || (time <= self.delay && limit == Limit::Left)) {
            return (0., true);
        }
        if let Some(stop) = self.stop_after {
            let end = self.delay + stop;
            if time >= end {
                return (stop, time > end || limit == Limit::Right);
            }
        }
        (time - self.delay, false)
    }

    fn angle(self, elapsed: Real) -> Real {
        (elapsed * self.frequency % 1.).mul_add(std::f64::consts::TAU, self.phase)
    }

    fn value_at(self, time: Real, limit: Limit) -> Real {
        if self.amplitude == 0. {
            return self.offset;
        }
        let (elapsed, _) = self.coordinate(time, limit);
        (self.amplitude * (-self.damping * elapsed).exp())
            .mul_add(self.angle(elapsed).sin(), self.offset)
    }

    fn jet(self, time: Real, scale: Real, order: usize, limit: Limit) -> Jet {
        let (elapsed, held) = self.coordinate(time, limit);
        if self.amplitude == 0. || held {
            return Jet::constant(order, self.value_at(time, limit));
        }
        let angular_scale = self.frequency * scale * std::f64::consts::TAU;
        let (sin, _) = Jet::affine(order, self.angle(elapsed), angular_scale).sin_cos();
        let envelope = Jet::affine(order, -self.damping * elapsed, -self.damping * scale)
            .exp()
            .scale(self.amplitude);
        envelope.multiply(&sin).offset(self.offset)
    }
}

fn exponential_term(time: Real, delay: Real, tau: Real) -> Real {
    -(-((time - delay).max(0.) / tau)).exp_m1()
}

fn exponential_value(exp: ExponentialForcing, time: Real) -> Real {
    let rise = exponential_term(time, exp.rise_delay, exp.rise_tau);
    let fall = exponential_term(time, exp.fall_delay, exp.fall_tau);
    (exp.pulsed - exp.initial).mul_add(rise - fall, exp.initial)
}

fn exponential_jet(
    exp: ExponentialForcing,
    time: Real,
    scale: Real,
    order: usize,
    limit: Limit,
) -> Jet {
    let term = |delay: Real, tau: Real| {
        if time < delay || (time <= delay && limit == Limit::Left) {
            Jet::constant(order, 0.)
        } else {
            Jet::affine(order, -(time - delay) / tau, -scale / tau)
                .exp()
                .scale(-1.)
                .offset(1.)
        }
    };
    term(exp.rise_delay, exp.rise_tau)
        .add(&term(exp.fall_delay, exp.fall_tau).scale(-1.))
        .scale(exp.pulsed - exp.initial)
        .offset(exp.initial)
}

impl SidedPwl {
    fn new(knots: Vec<Knot>, repeat: Option<Repeat>) -> SpiceResult<Self> {
        if knots.is_empty()
            || !knots
                .iter()
                .all(|k| k.time.is_finite() && k.left.is_finite() && k.right.is_finite())
            || !knots.windows(2).all(|pair| pair[0].time < pair[1].time)
        {
            return Err(invalid("knots must be finite and strictly ordered"));
        }
        let mut min_spacing = knots
            .windows(2)
            .map(|p| p[1].time - p[0].time)
            .fold(Real::INFINITY, Real::min);
        if let Some(r) = repeat {
            if !r.period.is_finite()
                || r.period <= 0.
                || !r.origin.is_finite()
                || knots[0].time.abs().to_bits() != 0
                || knots[knots.len() - 1].time >= r.period
                || r.hold.is_some_and(|v| !v.is_finite())
                || r.limit
                    .is_some_and(|limit| !limit.time.is_finite() || !limit.value.is_finite())
            {
                return Err(invalid("invalid repeating knot geometry"));
            }
            min_spacing = min_spacing.min(r.period - knots[knots.len() - 1].time);
        }
        Ok(Self {
            knots: knots.into(),
            repeat,
            min_spacing,
        })
    }

    fn absolute_time(&self, cycle: Real, time: Real) -> Real {
        self.repeat.map_or(time, |r| {
            if time.abs().to_bits() == 0 {
                Self::cycle_start(r, cycle)
            } else {
                r.origin + cycle.mul_add(r.period, time)
            }
        })
    }

    const fn cycle_start(r: Repeat, cycle: Real) -> Real {
        cycle.mul_add(r.period, r.origin)
    }

    // Use the generated absolute boundary, rather than testing rounded local
    // phase for equality. The same representation is used by the iterator.
    fn cycle_at(&self, time: Real, r: Repeat) -> SpiceResult<Real> {
        let mut cycle = ((time - r.origin) / r.period).floor();
        if !cycle.is_finite() || cycle.abs() >= MAX_CYCLE {
            return Err(invalid("unrepresentable cycle index"));
        }
        if time < Self::cycle_start(r, cycle) {
            cycle -= 1.;
        } else if time >= Self::cycle_start(r, cycle + 1.) {
            cycle += 1.;
        }
        let start = Self::cycle_start(r, cycle);
        let end = Self::cycle_start(r, cycle + 1.);
        if cycle.abs() >= MAX_CYCLE || !(start <= time && time < end && end.is_finite()) {
            return Err(invalid("unrepresentable cycle boundaries"));
        }
        if !self.spacing_proven(cycle, cycle, r) {
            // A short query need not reject a representable cycle merely
            // because the inexpensive uniform rounding bound is inconclusive.
            let mut previous = start;
            for knot in self.knots.iter().skip(1) {
                let next = self.absolute_time(cycle, knot.time);
                if !(previous < next && next < end) {
                    return Err(invalid("unrepresentable periodic knot spacing"));
                }
                previous = next;
            }
        }
        Ok(cycle)
    }

    // Both the fused cycle-plus-knot operation and the addition of origin
    // round once. The combined error of two adjacent events is at most two
    // ULPs at an upper bound on the magnitude of either intermediate.
    // Strict separation also excludes rounding-to-even ties. This uniform
    // proof is independent of the number of cycles in the requested window.
    fn spacing_proven(&self, first: Real, last: Real, r: Repeat) -> bool {
        let end = last + 1.;
        let scale = Self::cycle_start(r, first)
            .abs()
            .max(Self::cycle_start(r, end).abs())
            .max((first * r.period).abs())
            .max((end * r.period).abs());
        self.min_spacing > 2. * (scale.next_up() - scale)
    }

    #[expect(
        clippy::float_cmp,
        reason = "hold ends at the exact generated repeat origin"
    )]
    fn cycle_or_hold(&self, time: Real, limit: Limit) -> SpiceResult<CycleOrHold> {
        Ok(match self.repeat {
            Some(r) => {
                if let Some(end) = r.limit
                    && (time > end.time || (time == end.time && limit == Limit::Right))
                {
                    CycleOrHold::Hold(end.value)
                } else if let Some(hold) = r.hold
                    && (time < r.origin || (time == r.origin && limit == Limit::Left))
                {
                    CycleOrHold::Hold(hold)
                } else {
                    CycleOrHold::Cycle(self.cycle_at(time, r)?)
                }
            }
            None => CycleOrHold::Cycle(0.),
        })
    }

    fn next_endpoint(&self, cycle: Real, upper: usize) -> Option<(Real, Real)> {
        if upper < self.knots.len() {
            Some((self.time_at(cycle, upper), self.knots[upper].left))
        } else {
            self.repeat
                .map(|r| (Self::cycle_start(r, cycle + 1.), self.knots[0].left))
        }
    }

    /// The one-sided derivative uses the same rounded absolute interval
    /// as the scalar evaluator. No difference of rounded waveform samples
    /// is involved.
    fn segment_slope(
        &self,
        cycle: Real,
        upper: usize,
        time: Real,
        scale: Real,
    ) -> SpiceResult<Real> {
        let (t0, v0, t1, v1) = if upper == 0 {
            let Some(r) = self.repeat else {
                return Ok(0.);
            };
            // At a repeat boundary the incoming segment belongs to the
            // previous cycle. Validate that cycle as well as the new one.
            let previous = self.cycle_at(time.next_down(), r)?;
            let last = self.knots.len() - 1;
            (
                self.time_at(previous, last),
                self.knots[last].right,
                self.time_at(cycle, 0),
                self.knots[0].left,
            )
        } else {
            let Some((end, value)) = self.next_endpoint(cycle, upper) else {
                return Ok(0.);
            };
            (
                self.time_at(cycle, upper - 1),
                self.knots[upper - 1].right,
                end,
                value,
            )
        };
        if !(t0.is_finite() && t1.is_finite() && t0 < t1 && (t0..=t1).contains(&time)) {
            return Err(invalid("unrepresentable Taylor interpolation interval"));
        }
        Ok(scaled_interval_difference(v0, v1, scale, t0, t1))
    }

    fn value_at(&self, time: Real, limit: Limit) -> SpiceResult<Real> {
        self.local_affine(time, 0., limit).map(|(value, _)| value)
    }

    #[expect(
        clippy::float_cmp,
        reason = "external limits are selected at exact generated event times"
    )]
    fn local_affine(&self, time: Real, scale: Real, limit: Limit) -> SpiceResult<(Real, Real)> {
        let cycle = match self.cycle_or_hold(time, limit)? {
            CycleOrHold::Cycle(cycle) => cycle,
            CycleOrHold::Hold(hold) => return Ok((hold, 0.)),
        };
        let upper = self
            .knots
            .partition_point(|k| self.absolute_time(cycle, k.time) <= time);
        if upper == 0 {
            return Ok((self.knots[0].left, 0.));
        }
        let i = upper - 1;
        let left = self.knots[i];
        let t0 = self.time_at(cycle, i);
        let (value, segment_upper) = if time == t0 {
            match limit {
                Limit::Left => (left.left, i),
                Limit::Right => (left.right, upper),
            }
        } else {
            (self.interpolate(cycle, upper, time, t0, left.right)?, upper)
        };
        let slope = if scale == 0. {
            0.
        } else {
            self.segment_slope(cycle, segment_upper, time, scale)?
        };
        Ok((value, slope))
    }

    fn interpolate(
        &self,
        cycle: Real,
        upper: usize,
        time: Real,
        t0: Real,
        v0: Real,
    ) -> SpiceResult<Real> {
        let Some((t1, v1)) = self.next_endpoint(cycle, upper) else {
            return Ok(v0);
        };
        if !(t0 < time && time < t1) {
            return Err(invalid("unrepresentable interpolation interval"));
        }
        let span = t1 - t0;
        let fraction = if span.is_finite() {
            (time - t0) / span
        } else {
            t0.mul_add(-0.5, time * 0.5) / t0.mul_add(-0.5, t1 * 0.5)
        };
        Ok(affine_value(v0, v1, fraction))
    }

    fn time_at(&self, cycle: Real, knot: usize) -> Real {
        self.absolute_time(cycle, self.knots[knot].time)
    }

    fn events(&self, start: Real, end: Real) -> SpiceResult<PwlEvents> {
        let cutoff = self.repeat.and_then(|r| r.limit).map(|limit| limit.time);
        let termination = cutoff.filter(|time| (start..=end).contains(time));
        let end = cutoff.map_or(end, |time| end.min(time));
        if cutoff.is_some_and(|time| time <= start) {
            return Ok(PwlEvents {
                pwl: self.clone(),
                cycle: 0.,
                index: 0,
                end,
                done: true,
                cutoff: termination,
            });
        }
        let (cycle, done) = match self.repeat {
            Some(r) if r.hold.is_some() && end < r.origin => (0., true),
            Some(r) => {
                let first = if r.hold.is_some() && start < r.origin {
                    0.
                } else {
                    self.cycle_at(start, r)?
                };
                let last_time = if termination.is_some() {
                    end.next_down().max(start)
                } else {
                    end
                };
                let last = self.cycle_at(last_time, r)?;
                // Each endpoint cycle has been checked exactly if necessary.
                // For an arbitrarily long window require a uniform proof:
                // iterating all intervening cycles would destroy laziness.
                if first < last && !self.spacing_proven(first, last, r) {
                    return Err(invalid(
                        "unrepresentable periodic knot spacing over event window",
                    ));
                }
                (first, false)
            }
            None => (0., false),
        };
        let index = self
            .knots
            .partition_point(|k| self.absolute_time(cycle, k.time) < start);
        Ok(PwlEvents {
            pwl: self.clone(),
            cycle,
            index,
            end,
            done,
            cutoff: termination,
        })
    }
}

/// Lazy finite-window sequence of exact prepared source events.
#[derive(Debug, Clone)]
pub struct PreparedEvents(Events);

#[derive(Debug, Clone)]
enum Events {
    Pwl(PwlEvents),
    Finite(FiniteEvents),
}

#[derive(Debug, Clone)]
struct FiniteEvents([Option<Real>; 2]);

impl FiniteEvents {
    fn new(mut times: [Option<Real>; 2], start: Real, end: Real) -> Self {
        for time in &mut times {
            *time = time.filter(|time| (start..=end).contains(time));
        }
        if let [Some(a), Some(b)] = times {
            if a > b {
                times.swap(0, 1);
            } else if a.to_bits() == b.to_bits() || (a == 0. && b == 0.) {
                times[1] = None;
            }
        }
        Self(times)
    }

    fn next_time(&mut self) -> Option<Real> {
        self.0[0].take().or_else(|| self.0[1].take())
    }
}

#[derive(Debug, Clone)]
struct PwlEvents {
    pwl: SidedPwl,
    cycle: Real,
    index: usize,
    end: Real,
    done: bool,
    cutoff: Option<Real>,
}

impl PwlEvents {
    fn next_time(&mut self) -> Option<Real> {
        if self.done {
            return self.cutoff.take();
        }
        if self.index == self.pwl.knots.len() {
            if self.pwl.repeat.is_none() {
                self.done = true;
                return None;
            }
            self.cycle += 1.;
            self.index = 0;
        }
        let time = self.pwl.time_at(self.cycle, self.index);
        if time > self.end || self.cutoff.is_some_and(|cutoff| time >= cutoff) {
            self.done = true;
            return self.cutoff.take();
        }
        self.index += 1;
        Some(time)
    }
}

impl Iterator for PreparedEvents {
    type Item = WaveformEvent;
    fn next(&mut self) -> Option<Self::Item> {
        let time = match &mut self.0 {
            Events::Pwl(pwl) => pwl.next_time()?,
            Events::Finite(events) => events.next_time()?,
        };
        Some(WaveformEvent {
            time,
            kind: WaveformEventKind::PreparedKnot,
        })
    }
}
