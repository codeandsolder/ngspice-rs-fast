//! Resolved prepared source forcing with exact external limits.
//!
//! The producer owns symbolic source semantics. This module consumes its
//! validated physical knots without applying native PULSE or PWL defaults.
use std::sync::Arc;

use model_normalize_ngspice::waveform::{Forcing, Knot, Repeat, ResolvedWaveform};

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
}

impl PreparedForcing {
    /// Consume the shared producer representation, validating its numerical
    /// invariants at the engine boundary before any source is stamped.
    /// # Errors
    /// Invalid ordinates, geometry, period, or sine coefficients.
    pub fn new(resolved: ResolvedWaveform) -> SpiceResult<Self> {
        if !resolved.dc.is_finite() {
            return Err(invalid("nonfinite DC bias"));
        }
        let law = match resolved.forcing {
            Forcing::PiecewiseLinear { knots, repeat } => Law::Pwl(SidedPwl::new(knots, repeat)?),
            Forcing::Sine {
                offset,
                amplitude,
                frequency,
                delay,
                damping,
                hold_before_delay,
            } => {
                if ![offset, amplitude, frequency, delay, damping]
                    .iter()
                    .all(|v| v.is_finite())
                    || amplitude < 0.
                    || frequency <= 0.
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
            Law::Sine(sine) => sine.value_at(time),
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
            Law::Sine(sine) => Events::Single(
                (sine.hold && (start..=end).contains(&sine.delay)).then_some(sine.delay),
            ),
        };
        Ok(PreparedEvents(events))
    }
}

impl SignedSine {
    fn value_at(self, time: Real) -> Real {
        if self.amplitude == 0. || (self.hold && time <= self.delay) {
            return self.offset;
        }
        let elapsed = time - self.delay;
        let phase = (elapsed * self.frequency % 1.) * std::f64::consts::TAU;
        (self.amplitude * (-self.damping * elapsed).exp()).mul_add(phase.sin(), self.offset)
    }

    fn jet(self, time: Real, scale: Real, order: usize, limit: Limit) -> Jet {
        if self.amplitude == 0.
            || (self.hold && (time < self.delay || (limit == Limit::Left && time <= self.delay)))
        {
            return Jet::constant(order, self.offset);
        }
        let elapsed = time - self.delay;
        let phase = (elapsed * self.frequency % 1.) * std::f64::consts::TAU;
        let angular_scale = self.frequency * scale * std::f64::consts::TAU;
        let (sin, _) = Jet::affine(order, phase, angular_scale).sin_cos();
        let envelope = Jet::affine(order, -self.damping * elapsed, -self.damping * scale)
            .exp()
            .scale(self.amplitude);
        envelope.multiply(&sin).offset(self.offset)
    }
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
                if let Some(hold) = r.hold
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
        let (cycle, done) = match self.repeat {
            Some(r) if r.hold.is_some() && end < r.origin => (0., true),
            Some(r) => {
                let first = if r.hold.is_some() && start < r.origin {
                    0.
                } else {
                    self.cycle_at(start, r)?
                };
                let last = self.cycle_at(end, r)?;
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
        })
    }
}

/// Lazy finite-window sequence of exact prepared source events.
#[derive(Debug, Clone)]
pub struct PreparedEvents(Events);

#[derive(Debug, Clone)]
enum Events {
    Pwl(PwlEvents),
    Single(Option<Real>),
}

#[derive(Debug, Clone)]
struct PwlEvents {
    pwl: SidedPwl,
    cycle: Real,
    index: usize,
    end: Real,
    done: bool,
}

impl PwlEvents {
    fn next_time(&mut self) -> Option<Real> {
        if self.done {
            return None;
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
        if time > self.end {
            self.done = true;
            return None;
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
            Events::Single(time) => time.take()?,
        };
        Some(WaveformEvent {
            time,
            kind: WaveformEventKind::PreparedKnot,
        })
    }
}
