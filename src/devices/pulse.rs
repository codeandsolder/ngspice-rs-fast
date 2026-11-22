//! Analytic periodic PULSE forcing with explicit one-sided limits and lazy corners.
//!
//! Upstream behavior: `src/spicelib/devices/vsrc/vsrcload.c` (value, `case PULSE`),
//! `vsrcacct.c` (breakpoints) and the identical current-source files. Defaults that
//! depend on the analysis (`CKTstep`, `CKTfinalTime`) live in [`PulseSpec`] and are
//! resolved by [`PulseSpec::resolve`]; [`Pulse`] itself is fully specified.
//!
//! The eighth field (`PHASE` in `vsrcload.c`) follows ngspice's default
//! compatibility mode: a positive value `NP` limits the waveform to `NP` periods
//! (`tmax = NP * PER` after the delay; non-integer counts cut a pulse short),
//! after which the source holds V1. The `xs` compatibility mode, where the same
//! field is a phase in degrees, is not modelled (the port has no `ngbehavior`).
//!
//! Deliberate differences from C: negative delay is rejected rather than shifting
//! the pulse earlier; [`Pulse::new`] accepts zero-duration edges (C substitutes `CKTstep` for a
//! nonpositive rise/fall, which [`PulseSpec::resolve`] reproduces); and every
//! boundary has a defined left and right limit instead of a single sampled value.

use crate::devices::Limit;
use crate::primitives::{Real, SpiceError, SpiceResult};

/// The largest cycle index whose start time is still resolvable in `f64`.
pub(super) const MAX_CYCLE: Real = 4_503_599_627_370_496.0; // 2^52

/// Cycle decomposition in the reference C order. Callers choose when to
/// fold (load uses a strict boundary, accept an inclusive one) and when to
/// reject unresolvable cycle numbers; neither policy belongs in this kernel.
#[derive(Clone, Copy)]
pub(super) struct CyclePosition {
    index: Real,
    pub(super) base: Real,
    pub(super) phase: Real,
}

pub(super) fn cycle_position(elapsed: Real, period: Real) -> CyclePosition {
    let index = (elapsed / period).floor();
    let base = period * index;
    let phase = elapsed - base;
    CyclePosition { index, base, phase }
}

impl CyclePosition {
    pub(super) const fn resolvable(self) -> bool {
        self.index.is_finite() && self.index < MAX_CYCLE && self.base.is_finite()
    }
}

/// Skip a monotone prefix of cycles with no visible corners. A large delay
/// can round many consecutive cycles to the same absolute time. Search the
/// exactly representable cycle indices instead of stepping through them.
pub(super) fn seek_cycle(first: Real, mut omit: impl FnMut(Real) -> bool) -> Option<Real> {
    if first >= MAX_CYCLE {
        return None;
    }
    if !omit(first) {
        return Some(first);
    }
    let mut low = first + 1.;
    let mut high = MAX_CYCLE;
    // The search interval contains at most 2^52 integer indices.
    for _ in 0..52 {
        if low >= high {
            break;
        }
        let middle = low.midpoint(high).floor();
        if omit(middle) {
            low = middle + 1.;
        } else {
            high = middle;
        }
    }
    (low < MAX_CYCLE).then_some(low)
}

fn invalid(message: impl Into<String>) -> SpiceError {
    SpiceError::circuit(format!("invalid PULSE waveform: {}", message.into()))
}

/// Analysis quantities that parameterize C's waveform defaults.
///
/// `step` is `CKTstep` (the `.tran` print step) and `final_time` is
/// `CKTfinalTime` (the stop time).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientTiming {
    step: Real,
    final_time: Real,
}
impl TransientTiming {
    /// Creates timing from the `.tran` step and stop time.
    ///
    /// # Errors
    /// Either value is nonfinite or not strictly positive.
    pub fn new(step: Real, final_time: Real) -> SpiceResult<Self> {
        if !(step.is_finite() && step > 0. && final_time.is_finite() && final_time > 0.) {
            return Err(SpiceError::circuit(
                "transient timing needs finite positive step and stop time",
            ));
        }
        Ok(Self { step, final_time })
    }
    /// The `.tran` step (`CKTstep`).
    #[must_use]
    pub const fn step(&self) -> Real {
        self.step
    }
    /// The stop time (`CKTfinalTime`).
    #[must_use]
    pub const fn final_time(&self) -> Real {
        self.final_time
    }
}

/// Parsed PULSE fields before analysis-specific timing defaults are resolved.
///
/// A PULSE as written: omitted trailing fields stay `None` until an analysis
/// supplies [`TransientTiming`]. Fields must be supplied as a contiguous prefix
/// (`delay`, `rise`, `fall`, `width`, `period`), as in the netlist syntax.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseSpec {
    /// V1, the initial level.
    pub initial: Real,
    /// V2, the pulsed level.
    pub pulsed: Real,
    /// TD, seconds (default 0).
    pub delay: Option<Real>,
    /// TR (nonpositive or omitted: `CKTstep`).
    pub rise: Option<Real>,
    /// TF (nonpositive or omitted: `CKTstep`).
    pub fall: Option<Real>,
    /// PW (negative or omitted: `CKTfinalTime`; exactly five fields: 0).
    pub width: Option<Real>,
    /// PER (nonpositive or omitted: `CKTfinalTime`).
    pub period: Option<Real>,
    /// NP, the eighth field: a positive value is the number of pulses
    /// (`tmax = NP * PER`); zero, negative or omitted means unlimited.
    pub count: Option<Real>,
}
impl PulseSpec {
    /// Validate input-only constraints and return the number of optional
    /// coefficients. Timing-dependent sums/products are checked at resolve.
    pub(super) fn validate(&self) -> SpiceResult<usize> {
        let optional = [
            self.delay,
            self.rise,
            self.fall,
            self.width,
            self.period,
            self.count,
        ];
        let given =
            crate::devices::functions::prefix("PULSE", &[self.initial, self.pulsed], &optional)?;
        if self.delay.is_some_and(|delay| delay < 0.) {
            return Err(invalid(
                "delay/rise/fall/width must be nonnegative and period positive",
            ));
        }
        let explicit_rise = self.rise.unwrap_or(0.).max(0.);
        let explicit_width = self.width.unwrap_or(0.).max(0.);
        let explicit_fall = self.fall.unwrap_or(0.).max(0.);
        if !(self.pulsed - self.initial).is_finite()
            || !(explicit_rise + explicit_width + explicit_fall).is_finite()
        {
            return Err(invalid("level difference or edge sum overflows"));
        }
        if let (Some(count), Some(period)) = (self.count, self.period)
            && count > 0.
            && period > 0.
            && !(count * period).is_finite()
        {
            return Err(invalid("pulse count times period overflows"));
        }
        Ok(given)
    }

    /// Resolves C's defaults (vsrcload.c, case PULSE) for one transient run.
    ///
    /// # Errors
    /// Nonfinite fields, a gap in the supplied prefix, negative delay, or
    /// resolved timing that fails the validating pulse constructor.
    pub fn resolve(&self, timing: &TransientTiming) -> SpiceResult<Pulse> {
        let given = self.validate()?;
        let positive_or = |value: Option<Real>, default: Real| match value {
            Some(v) if v > 0. => v,
            _ => default,
        };
        // C: exactly five fields (TD, TR, TF but no PW) means PW = 0.
        let width = match (given, self.width) {
            (3, _) => 0.,
            (_, Some(w)) if w >= 0. => w,
            _ => timing.final_time,
        };
        Pulse::new(
            self.initial,
            self.pulsed,
            self.delay.unwrap_or(0.),
            positive_or(self.rise, timing.step),
            positive_or(self.fall, timing.step),
            width,
            positive_or(self.period, timing.final_time),
        )?
        .with_count(self.count.unwrap_or(0.))
    }
}

/// A validated, fully specified periodic pulse.
///
/// Starting at `delay`, each cycle of length `period` rises linearly from
/// `initial` to `pulsed` over `rise`, holds for `width`, falls over `fall` and
/// then idles at `initial`. A cycle shorter than `rise + width + fall` is cut
/// off at the period boundary (a jump), as C does. Zero-duration edges are
/// jumps whose left and right limits differ; use [`Limit`] to choose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pulse {
    initial: Real,
    pulsed: Real,
    delay: Real,
    rise: Real,
    fall: Real,
    width: Real,
    period: Real,
    /// `NP * PER` (time after `delay` past which the source holds V1).
    stop: Option<Real>,
}
impl Pulse {
    /// Creates a pulse.
    ///
    /// # Errors
    /// Any field nonfinite; negative delay, rise, fall or width; nonpositive
    /// period; or level differences / edge sums that overflow.
    pub fn new(
        initial: Real,
        pulsed: Real,
        delay: Real,
        rise: Real,
        fall: Real,
        width: Real,
        period: Real,
    ) -> SpiceResult<Self> {
        let all = [initial, pulsed, delay, rise, fall, width, period];
        if all.iter().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite field"));
        }
        if delay < 0. || rise < 0. || fall < 0. || width < 0. || period <= 0. {
            return Err(invalid(
                "delay/rise/fall/width must be nonnegative and period positive",
            ));
        }
        if !(pulsed - initial).is_finite() || !(rise + width + fall).is_finite() {
            return Err(invalid("level difference or edge sum overflows"));
        }
        Ok(Self {
            initial,
            pulsed,
            delay,
            rise,
            fall,
            width,
            period,
            stop: None,
        })
    }

    /// Limits the pulse train to `count` periods (the eighth PULSE field in
    /// ngspice's default compatibility mode, `vsrcload.c`): past
    /// `delay + count * period` the value is V1. Zero or negative means
    /// unlimited, as in C.
    ///
    /// # Errors
    /// `count` nonfinite or `count * period` overflows.
    pub fn with_count(mut self, count: Real) -> SpiceResult<Self> {
        if !count.is_finite() {
            return Err(invalid("nonfinite pulse count"));
        }
        self.stop = if count > 0. {
            let stop = count * self.period;
            if !stop.is_finite() {
                return Err(invalid("pulse count times period overflows"));
            }
            Some(stop)
        } else {
            None
        };
        Ok(self)
    }
    /// The time after `delay` past which the source holds V1, if limited.
    #[must_use]
    pub const fn stop(&self) -> Option<Real> {
        self.stop
    }
    /// V1.
    #[must_use]
    pub const fn initial(&self) -> Real {
        self.initial
    }
    /// V2.
    #[must_use]
    pub const fn pulsed(&self) -> Real {
        self.pulsed
    }
    /// TD.
    #[must_use]
    pub const fn delay(&self) -> Real {
        self.delay
    }
    /// TR.
    #[must_use]
    pub const fn rise(&self) -> Real {
        self.rise
    }
    /// TF.
    #[must_use]
    pub const fn fall(&self) -> Real {
        self.fall
    }
    /// PW.
    #[must_use]
    pub const fn width(&self) -> Real {
        self.width
    }
    /// PER.
    #[must_use]
    pub const fn period(&self) -> Real {
        self.period
    }

    /// Forcing at time `t` using the requested one-sided limit.
    ///
    /// Away from corners both limits agree. Before `delay` the value is
    /// `initial`; at exactly `delay` the left limit is `initial` and the right
    /// limit is the start of the first rise.
    ///
    /// # Errors
    /// `t` nonfinite, or so many periods from the origin that the in-cycle
    /// position is unresolvable.
    pub fn value_at(&self, t: Real, limit: Limit) -> SpiceResult<Real> {
        if !t.is_finite() {
            return Err(invalid("nonfinite evaluation time"));
        }
        let s = t - self.delay;
        if !s.is_finite() {
            return Err(invalid("evaluation time overflows"));
        }
        // vsrcload.c, `case PULSE`, verbatim: C's pulse is continuous (TR and
        // TF default to CKTstep), so both limits are the same value.
        let _ = limit;
        let mut time = s;
        if let Some(stop) = self.stop
            && time > stop
        {
            return Ok(self.initial);
        }
        if time > self.period {
            let cycle = cycle_position(time, self.period);
            if !cycle.resolvable() {
                return Err(invalid(
                    "evaluation time is too many periods from the origin",
                ));
            }
            time = cycle.phase;
        }
        let (v1, v2) = (self.initial, self.pulsed);
        let (tr, tf, pw) = (self.rise, self.fall, self.width);
        Ok(if time <= 0. || time >= tr + pw + tf {
            v1
        } else if time >= tr && time <= tr + pw {
            v2
        } else if time > 0. && time < tr {
            v1 + (v2 - v1) * time / tr
        } else {
            v2 + (v1 - v2) * (time - (tr + pw)) / tf
        })
    }

    /// Lazily enumerates the corner and jump times in the closed window
    /// `[t0, t1]`, ascending and strictly increasing, one cycle at a time.
    ///
    /// Per cycle the corners are the start, end of rise, start of fall and end
    /// of fall, limited to offsets before the next cycle starts. The sequence is
    /// never materialized: take only what a run needs.
    ///
    /// # Errors
    /// Nonfinite or reversed window, or a window so many periods long that
    /// cycle start times are unresolvable.
    pub fn breakpoints_in(&self, t0: Real, t1: Real) -> SpiceResult<PulseBreakpoints> {
        PulseBreakpoints::new(*self, t0, t1)
    }
}

/// Lazy corner iterator for a [`Pulse`] over a finite window.
#[derive(Debug, Clone)]
pub struct PulseBreakpoints {
    pulse: Pulse,
    t0: Real,
    t1: Real,
    offsets: [Real; 4],
    count: usize,
    cycle: Real,
    index: usize,
    last: Option<Real>,
    done: bool,
    /// A count-limited train's cut time, when it truncates an edge or the
    /// plateau (a jump to V1 that is not already a corner).
    cut: Option<Real>,
    /// The remaining breakpoints once the count is exhausted.
    tail: Option<[Option<Real>; 2]>,
}
impl PulseBreakpoints {
    fn new(pulse: Pulse, t0: Real, t1: Real) -> SpiceResult<Self> {
        if !(t0.is_finite() && t1.is_finite() && t0 <= t1) {
            return Err(invalid("breakpoint window must be finite with t0 <= t1"));
        }
        let all = [
            0.,
            pulse.rise,
            pulse.rise + pulse.width,
            pulse.rise + pulse.width + pulse.fall,
        ];
        let mut offsets = [0.; 4];
        let mut count = 0;
        for offset in all {
            if offset < pulse.period && (count == 0 || offset > offsets[count - 1]) {
                offsets[count] = offset;
                count += 1;
            }
        }
        let cut = pulse.stop.and_then(|stop| {
            let active = (pulse.rise + pulse.width + pulse.fall).min(pulse.period);
            let offset = cycle_position(stop, pulse.period).phase;
            (offset > 0. && offset < active).then_some(pulse.delay + stop)
        });
        let last_cycle = ((t1 - pulse.delay) / pulse.period).floor();
        if !last_cycle.is_finite() || last_cycle >= MAX_CYCLE {
            return Err(invalid("breakpoint window spans too many periods"));
        }
        let first = ((t0 - pulse.delay) / pulse.period).floor();
        // A finite train has only one request after its stop. Do not start
        // past that request's cycle, or a late window would invent a corner.
        let first = pulse
            .stop
            .map_or(first, |stop| first.min((stop / pulse.period).floor()));
        // Start one cycle early so rounding can never skip a corner at t0.
        let cycle = if first > 1. { first - 1. } else { 0. };
        Ok(Self {
            pulse,
            t0,
            t1,
            offsets,
            count,
            cycle,
            index: 0,
            last: None,
            done: false,
            cut,
            tail: None,
        })
    }
}
impl Iterator for PulseBreakpoints {
    type Item = Real;
    fn next(&mut self) -> Option<Real> {
        while !self.done && self.last.is_none_or(|last| last < self.t1) {
            if let Some(tail) = &mut self.tail {
                // After `tmax`: the cut (if any), then C's last request.
                let Some(t) = tail.iter_mut().find_map(Option::take) else {
                    self.done = true;
                    break;
                };
                if t >= self.t0 && t <= self.t1 && self.last.is_none_or(|last| t > last) {
                    self.last = Some(t);
                    return Some(t);
                }
                continue;
            }
            if self.index >= self.count {
                self.index = 0;
                let Some(cycle) = seek_cycle(self.cycle + 1., |cycle| {
                    let cycle_start = self.pulse.period * cycle;
                    let local_last = cycle_start + self.offsets[self.count - 1];
                    // Do not skip the cycle containing the first request
                    // after a finite train's stop, even if its corners
                    // round to already emitted times.
                    if self.pulse.stop.is_some_and(|stop| local_last > stop) {
                        return false;
                    }
                    let end = self.pulse.delay + local_last;
                    end < self.t0 || self.last.is_some_and(|last| end <= last)
                }) else {
                    self.done = true;
                    break;
                };
                self.cycle = cycle;
            }
            let cycle_start = self.pulse.period * self.cycle;
            let local = cycle_start + self.offsets[self.index];
            let t = self.pulse.delay + local;
            self.index += 1;
            if !t.is_finite() {
                self.done = true;
            } else if self.pulse.stop.is_some_and(|stop| local > stop) {
                // C's VSRCaccept stops once `time > tmax`, but the request it
                // made at its last step before that is the first corner after
                // `tmax`; landing there too keeps the step sequence C's.
                self.tail = Some([self.cut.take(), Some(t)]);
            } else if t > self.t1 {
                self.done = true;
            } else if t >= self.t0 && self.last.is_none_or(|last| t > last) {
                self.last = Some(t);
                return Some(t);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const L: Limit = Limit::Left;
    const R: Limit = Limit::Right;

    fn pulse(rise: Real, fall: Real, width: Real, period: Real) -> Pulse {
        Pulse::new(1., 5., 2., rise, fall, width, period).unwrap()
    }
    fn v(p: &Pulse, t: Real, limit: Limit) -> Real {
        p.value_at(t, limit).unwrap()
    }

    #[test]
    fn corners_with_finite_edges() {
        let p = pulse(1., 2., 3., 10.);
        // Delay boundary is continuous because the rise starts from V1.
        assert_eq!((v(&p, 2., L), v(&p, 2., R)), (1., 1.));
        assert_eq!(v(&p, 1., R), 1.);
        assert_eq!(v(&p, 2.5, R), 3.);
        assert_eq!((v(&p, 3., L), v(&p, 3., R)), (5., 5.));
        assert_eq!(v(&p, 4., R), 5.);
        // Width end / start of fall, then middle and end of fall.
        assert_eq!((v(&p, 6., L), v(&p, 6., R)), (5., 5.));
        assert_eq!(v(&p, 7., R), 3.);
        assert_eq!((v(&p, 8., L), v(&p, 8., R)), (1., 1.));
        assert_eq!(v(&p, 11.9, R), 1.);
    }

    #[test]
    fn zero_duration_edges_follow_the_c_formula() {
        // vsrcload.c has one value per time: `time <= 0` and
        // `time >= TR + PW + TF` are V1 whatever the limit. (A deck cannot
        // produce zero edges; TR and TF default to CKTstep.)
        let p = pulse(0., 0., 3., 10.);
        assert_eq!((v(&p, 2., L), v(&p, 2., R)), (1., 1.));
        assert_eq!((v(&p, 5., L), v(&p, 5., R)), (1., 1.));
        assert_eq!(v(&p, 3.5, R), 5.);
        assert_eq!(v(&p, 6., R), 1.);
        let spike = pulse(0., 0., 0., 10.);
        assert_eq!((v(&spike, 2., L), v(&spike, 2., R)), (1., 1.));
        assert_eq!(v(&spike, 2.5, R), 1.);
    }

    #[test]
    fn multiple_periods_fold_as_vsrcload_does() {
        let p = pulse(0., 0., 3., 10.);
        for k in 0..4 {
            let base = 2. + 10. * f64::from(k);
            assert_eq!((v(&p, base, L), v(&p, base, R)), (1., 1.), "k={k}");
            assert_eq!(v(&p, base + 1.5, R), 5., "k={k}");
            assert_eq!(v(&p, base + 3., R), 1., "k={k}");
            assert_eq!(v(&p, base + 7., R), 1., "k={k}");
        }
        // Slow rise cut off by a short period: `time > PER` folds strictly
        // after the period, so at time = PER the rise is still evaluated.
        let cut = pulse(8., 1., 0., 4.);
        assert_eq!(v(&cut, 6., R), 1. + 4. * 4. / 8.);
        assert_eq!(v(&cut, 5., R), 1. + 4. * 3. / 8.);
        assert_eq!(v(&cut, 10., L), 1.);
    }

    #[test]
    fn evaluation_failures() {
        let p = pulse(1., 1., 1., 10.);
        assert!(p.value_at(f64::NAN, R).is_err());
        assert!(p.value_at(f64::INFINITY, L).is_err());
        let tiny = Pulse::new(0., 1., 0., 0., 0., 0., 1e-300).unwrap();
        assert!(tiny.value_at(1., R).is_err());
        assert!(p.value_at(MAX_CYCLE * p.period() + p.delay(), R).is_err());
        let finite = p.with_count(2.).unwrap();
        assert_eq!(finite.value_at(Real::MAX, R).unwrap(), p.initial());
    }

    #[test]
    fn constructor_rejections() {
        let ok = |a: [Real; 7]| Pulse::new(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
        assert!(ok([0., 1., 0., 1., 1., 1., 5.]).is_ok());
        for bad in [
            [f64::NAN, 1., 0., 1., 1., 1., 5.],
            [0., f64::INFINITY, 0., 1., 1., 1., 5.],
            [0., 1., -1., 1., 1., 1., 5.],
            [0., 1., 0., -1., 1., 1., 5.],
            [0., 1., 0., 1., -1., 1., 5.],
            [0., 1., 0., 1., 1., -1., 5.],
            [0., 1., 0., 1., 1., 1., 0.],
            [0., 1., 0., 1., 1., 1., -5.],
            [-f64::MAX, f64::MAX, 0., 1., 1., 1., 5.],
            [0., 1., 0., f64::MAX, f64::MAX, f64::MAX, 5.],
        ] {
            assert!(ok(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn breakpoints_are_lazy_windowed_and_strictly_increasing() {
        let p = pulse(1., 2., 3., 10.);
        let all: Vec<_> = p.breakpoints_in(0., 25.).unwrap().collect();
        assert_eq!(all, [2., 3., 6., 8., 12., 13., 16., 18., 22., 23.]);
        // Closed window, starting mid-run.
        let window: Vec<_> = p.breakpoints_in(12., 16.).unwrap().collect();
        assert_eq!(window, [12., 13., 16.]);
        // Zero-duration edges merge coincident corners.
        let q = pulse(0., 0., 3., 10.);
        let corners: Vec<_> = q.breakpoints_in(0., 15.).unwrap().collect();
        assert_eq!(corners, [2., 5., 12., 15.]);
        // An enormous but finite run is enumerated incrementally, not expanded.
        let mut lazy = pulse(1e-9, 1e-9, 1e-9, 1e-6)
            .breakpoints_in(0., 1e6)
            .unwrap();
        assert_eq!(lazy.next(), Some(2.));
        assert_eq!(lazy.by_ref().take(5).count(), 5);
    }

    #[test]
    fn finite_train_windows_are_slices_of_the_complete_corner_sequence() {
        for count in [0.25, 1., 1.4, 1.9, 3., 8.25] {
            let pulse = pulse(1., 2., 3., 10.).with_count(count).unwrap();
            let full: Vec<_> = pulse.breakpoints_in(0., 300.).unwrap().collect();
            for start in 0..300 {
                let start = Real::from(start) + 0.25;
                let end = (start + 9.5).min(300.);
                let expected: Vec<_> = full
                    .iter()
                    .copied()
                    .filter(|&t| t >= start && t <= end)
                    .collect();
                assert_eq!(
                    pulse
                        .breakpoints_in(start, end)
                        .unwrap()
                        .collect::<Vec<_>>(),
                    expected,
                    "count {count}, window {start}..{end}"
                );
            }
        }
    }

    #[test]
    fn cycle_overlapping_corners_are_dropped() {
        // rise+width+fall exceeds the period: only corners before the next start.
        let p = pulse(1., 1., 5., 4.);
        let all: Vec<_> = p.breakpoints_in(0., 12.).unwrap().collect();
        assert_eq!(all, [2., 3., 6., 7., 10., 11.]);
    }

    #[test]
    fn breakpoint_window_failures() {
        let p = pulse(1., 1., 1., 10.);
        assert!(p.breakpoints_in(5., 1.).is_err());
        assert!(p.breakpoints_in(0., f64::NAN).is_err());
        assert!(p.breakpoints_in(f64::NEG_INFINITY, 1.).is_err());
        let tiny = Pulse::new(0., 1., 0., 0., 0., 0., 1e-300).unwrap();
        assert!(tiny.breakpoints_in(0., 1.).is_err());
    }

    #[test]
    fn spec_resolves_c_defaults() {
        let timing = TransientTiming::new(1e-3, 1.).unwrap();
        let spec = |fields: [Option<Real>; 5]| PulseSpec {
            initial: 0.,
            pulsed: 1.,
            delay: fields[0],
            rise: fields[1],
            fall: fields[2],
            width: fields[3],
            period: fields[4],
            count: None,
        };
        // Two fields: TD=0, TR=TF=step, PW=PER=final.
        let p = spec([None; 5]).resolve(&timing).unwrap();
        assert_eq!(
            (p.delay(), p.rise(), p.fall(), p.width(), p.period()),
            (0., 1e-3, 1e-3, 1., 1.)
        );
        // Exactly five fields: PW defaults to zero, not the stop time.
        let p = spec([Some(0.1), Some(0.2), Some(0.3), None, None])
            .resolve(&timing)
            .unwrap();
        assert_eq!((p.width(), p.period()), (0., 1.));
        // Four fields: PW defaults to the stop time.
        let p = spec([Some(0.1), Some(0.2), None, None, None])
            .resolve(&timing)
            .unwrap();
        assert_eq!((p.fall(), p.width()), (1e-3, 1.));
        // Nonpositive TR/TF/PER and negative PW fall back; zero PW is kept.
        let p = spec([Some(0.), Some(0.), Some(-1.), Some(-1.), Some(0.)])
            .resolve(&timing)
            .unwrap();
        assert_eq!(
            (p.rise(), p.fall(), p.width(), p.period()),
            (1e-3, 1e-3, 1., 1.)
        );
        let p = spec([Some(0.), Some(0.1), Some(0.1), Some(0.), Some(0.5)])
            .resolve(&timing)
            .unwrap();
        assert_eq!((p.width(), p.period()), (0., 0.5));
    }

    #[test]
    fn spec_rejections() {
        let timing = TransientTiming::new(1e-3, 1.).unwrap();
        let base = PulseSpec {
            initial: 0.,
            pulsed: 1.,
            delay: None,
            rise: None,
            fall: None,
            width: None,
            period: None,
            count: None,
        };
        let gap = PulseSpec {
            rise: Some(1.),
            ..base
        };
        assert!(gap.resolve(&timing).is_err());
        let negative_delay = PulseSpec {
            delay: Some(-1.),
            ..base
        };
        assert!(negative_delay.resolve(&timing).is_err());
        let nonfinite = PulseSpec {
            delay: Some(f64::NAN),
            ..base
        };
        assert!(nonfinite.resolve(&timing).is_err());
        let count_gap = PulseSpec {
            count: Some(2.),
            ..base
        };
        assert!(count_gap.resolve(&timing).is_err());
        assert!(pulse(1., 1., 1., 10.).with_count(f64::INFINITY).is_err());
        assert!(pulse(1., 1., 1., 1e300).with_count(1e300).is_err());
        assert!(TransientTiming::new(0., 1.).is_err());
        assert!(TransientTiming::new(1., f64::INFINITY).is_err());
        assert!(TransientTiming::new(f64::NAN, 1.).is_err());
    }

    #[test]
    fn pulse_count_holds_v1_after_count_periods() {
        // Three pulses: delay 2, period 10, high on [3, 6], fall to 1 at 8.
        let p = pulse(1., 2., 3., 10.).with_count(3.).unwrap();
        assert_eq!(p.stop(), Some(30.));
        assert_eq!(v(&p, 25., R), 5.);
        assert_eq!(v(&p, 32., R), 1.);
        assert_eq!(v(&p, 32.5, R), 1.);
        assert_eq!(v(&p, 1e6, L), 1.);
        // Integer counts end at a cycle boundary (continuous). After it only
        // C's last request remains: the next corner (end of the next rise).
        let corners: Vec<_> = p.breakpoints_in(0., 100.).unwrap().collect();
        assert_eq!(
            corners,
            [
                2., 3., 6., 8., 12., 13., 16., 18., 22., 23., 26., 28., 32., 33.
            ]
        );
        // Zero and negative counts are unlimited, as in C (`PHASE > 0`).
        for count in [0., -2.] {
            let q = pulse(1., 2., 3., 10.).with_count(count).unwrap();
            assert_eq!(q.stop(), None);
            assert_eq!(v(&q, 43., R), 5.);
        }
        // A fractional count cuts the plateau: a jump to V1 at 2 + 1.4 * 10.
        let cut = pulse(1., 2., 3., 10.).with_count(1.4).unwrap();
        assert_eq!(v(&cut, 15., R), 5.);
        // vsrcload.c: `time > tmax` holds V1; at tmax itself the shape.
        assert_eq!((v(&cut, 16., L), v(&cut, 16., R)), (5., 5.));
        assert_eq!(v(&cut, 16.5, R), 1.);
        let corners: Vec<_> = cut.breakpoints_in(0., 100.).unwrap().collect();
        assert_eq!(corners, [2., 3., 6., 8., 12., 13., 16., 18.]);
        // A cut in the idle part is no corner.
        let idle = pulse(1., 2., 3., 10.).with_count(1.9).unwrap();
        let corners: Vec<_> = idle.breakpoints_in(0., 100.).unwrap().collect();
        assert_eq!(corners, [2., 3., 6., 8., 12., 13., 16., 18., 22.]);
        assert_eq!(v(&idle, 19.5, R), 1.);
        // Through the spec's eighth field.
        let timing = TransientTiming::new(1e-3, 1.).unwrap();
        let spec = PulseSpec {
            initial: 0.,
            pulsed: 1.,
            delay: Some(0.),
            rise: Some(0.1),
            fall: Some(0.1),
            width: Some(0.1),
            period: Some(0.5),
            count: Some(1.),
        };
        let p = spec.resolve(&timing).unwrap();
        assert_eq!(p.stop(), Some(0.5));
        assert_eq!(p.value_at(0.65, R).unwrap(), 0.);
    }
    #[test]
    fn input_validation_does_not_invent_transient_timing() {
        let spec = PulseSpec {
            initial: 0.,
            pulsed: 1.,
            delay: None,
            rise: None,
            fall: None,
            width: None,
            period: None,
            count: None,
        };
        assert_eq!(spec.validate().unwrap(), 0);
        assert!(
            spec.resolve(&TransientTiming::new(1e-3, 1.).unwrap())
                .is_ok()
        );
        assert!(
            spec.resolve(&TransientTiming::new(Real::MAX, 1.).unwrap())
                .is_err()
        );
        let overflow = PulseSpec {
            initial: -Real::MAX,
            pulsed: Real::MAX,
            ..spec
        };
        assert!(overflow.validate().is_err());
    }
    #[test]
    fn collapsed_cycles_advance_to_next_representable_event_without_scanning() {
        let delay = 1e16_f64;
        let end = delay.next_up();
        let pulse = Pulse::new(0., 1., delay, 1e-13, 1e-13, 1e-13, 1e-12).unwrap();
        // Two trillion periods fit in one ulp of the absolute time.
        assert_eq!(
            pulse
                .breakpoints_in(delay, end)
                .unwrap()
                .collect::<Vec<_>>(),
            [delay, end]
        );
        assert_eq!(
            pulse
                .breakpoints_in(delay, delay)
                .unwrap()
                .collect::<Vec<_>>(),
            [delay]
        );
        let finite = pulse.with_count(0.25).unwrap();
        assert_eq!(
            finite
                .breakpoints_in(delay, end)
                .unwrap()
                .collect::<Vec<_>>(),
            [delay]
        );
        // A finite train whose stop is on the rounding threshold still
        // publishes the *first* corner after the stop, even after skipping.
        let long = pulse.with_count(1e12).unwrap();
        assert_eq!(
            long.breakpoints_in(delay, end).unwrap().collect::<Vec<_>>(),
            [delay, end]
        );
    }

    #[test]
    fn cycle_seek_is_exact_and_bounded() {
        for start in [1., MAX_CYCLE - 1.] {
            assert_eq!(seek_cycle(start, |_| false), Some(start));
            assert_eq!(seek_cycle(start, |_| true), None);
        }
        let threshold = MAX_CYCLE - 3.;
        assert_eq!(seek_cycle(1., |cycle| cycle < threshold), Some(threshold));
        assert_eq!(seek_cycle(MAX_CYCLE, |_| false), None);
    }
}
