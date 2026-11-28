//! Adaptive trapezoidal / Gear companion-model transient driver.
//!
//! This is the SPICE-compatible `.tran` backend: capacitors and inductors are
//! and M4's bounded nonlinear device charges are stamped as companion models
//! from accepted charge/flux history
//! ([`crate::devices::Circuit::load`]), each timestep is a trial that is either
//! rejected (nothing advances) or accepted atomically
//! ([`crate::devices::Circuit::accept_point`]). It is the only transient
//! backend.
//!
//! The control flow follows the transient loop of `dctran.c` and its helpers
//! (`ckttrunc.c`, `cktterr.c`, `niinteg.c`, `nicomcof.c`). In short:
//!
//! * **Initial point**: the DC bias (shared M4 Newton/continuation for nonlinear
//!   circuits) with the sources at their `t = 0` left
//!   limit and the `.ic` node voltages imposed (forced in every Newton load of
//!   a nonlinear bias, with `.nodeset` forced in its `MODEINITJCT`/`MODEINITFIX`
//!   loads), or, with `uic`, the charges and fluxes of the capacitor/inductor
//!   initial conditions and of the nonlinear devices' instance initial
//!   conditions without any solve (see [`crate::analysis::initial`] and
//!   `docs/port/TRANSIENT.md`). The charge/flux
//!   state fills the whole accepted history (C copies `CKTstate0` into
//!   `CKTstate1..3`) with zero derivative.
//! * **Step size**: first step `min(stop/100, tstep)/10`, cut at the `t = 0`
//!   breakpoint to `0.1 min(stop/50, gap to the next breakpoint)/10`; maximum
//!   step `tmax`, or `min(tstep, (stop - start)/50)` (`traninit.c`); minimum
//!   step `1e-11 maxstep`. `tstep` is therefore a step ceiling, never an output
//!   spacing. The loop is a literal port of `dctran.c` as the reference binary
//!   is built (XSPICE defined, no event-driven instances), so the accepted
//!   time points are bit-identical whenever the device loads are: the time
//!   advances by `CKTtime += CKTdelta` and a rejection subtracts the same
//!   delta again, `CKTsaveDelta`, `CKTbreak` and `firsttime` follow C, and
//!   every trial's Newton solve starts from the previous trial's last iterate
//!   (C's `CKTrhsOld`), rejected or not.
//! * **Order policy**: the first step and the first step after every source
//!   breakpoint use order 1 (backward Euler). After an accepted order-1 step the
//!   order is raised to 2 when `maxord > 1` and the order-2 truncation estimate
//!   allows a step more than 5 % larger. Like `dctran.c`
//!   (`CKTdeltaOld` is filled with `CKTmaxStep`) the estimate may already be
//!   probed on the second step, using the maximum step as the placeholder for
//!   the not-yet-existing older step ([`crate::maths::StepHistory::with_fill`]).
//!   `maxord=1` keeps backward Euler throughout. `dctran.c` has no other order
//!   change: it never raises the order above 2, so `maxord` 3 to 6 (accepted
//!   for both methods, as in C) run exactly like `maxord=2`; the Gear order
//!   3–6 coefficients of [`crate::maths::integrator`] are not reached from
//!   here (#98, verified bit-identical in the C reference binary).
//! * **Truncation error**: per capacitor/inductor divided differences of the
//!   charge/flux ([`crate::maths::Coefficients::truncation_timestep`], `CKTterr`)
//!   bound the next step to `min(2 dt, limit)`. A trial whose bound is not
//!   above `0.9 dt` is rejected and retried with the bound. The first step is
//!   never checked, as in C.
//! * **Breakpoints**: C's table ([`BreakTable`], `CKTsetBreak`/`CKTclrBreak`
//!   with `CKTminBreak = 10 CKTdelmin`) is filled at each accepted point by
//!   the sources' accept hooks ([`SourceBreaks`], `VSRCaccept`/`ISRCaccept`).
//!   A step that would pass `CKTbreaks[0]` is cut to end there (at
//!   `CKTtime + (CKTbreaks[0] - CKTtime)`), or the distance is halved when
//!   `t + 1.9 dt` passes it; the step ending at a breakpoint evaluates the
//!   forcing with the left limit, other steps with the right (only port-only
//!   jumps tell them apart). Devices with a delay history (transmission lines,
//!   [`crate::devices::delay`]) add entries through the same `CKTsetBreak`
//!   when they accept a point (`TRAaccept`), in C's `CKTaccept` type order
//!   between the current and the voltage sources.
//! * **Failures**: a step at or below the minimum is retried once at the
//!   minimum, then the run fails ("timestep too small"); a work limit bounds
//!   accepted + rejected steps (C has none); accept-hook and solver failures
//!   abort the run.
//!
//! # Output
//!
//! Like C's rawfile, the plot holds **every accepted time point** with
//! `time >= tstart` (no interpolation and no resampling onto the `.tran` step),
//! so no sample ever interpolates across a breakpoint and every breakpoint inside
//! the run has a sample. C emits no second sample for a jump; neither does this
//! driver: the sample at a breakpoint is the left limit and the next sample is
//! the first step of the right-hand segment.
//!
//! # Newton hook (M4)
//!
//! A trial is an iteration `load -> solve -> converged?`. Linear circuits take
//! exactly one solve per trial (followed by a state-only reload at the
//! solution); circuits with nonlinear devices iterate with C's voltage/current
//! tolerance test up to `itl4` times (request `tranmaxiter=`, default 100:
//! C's `NIiter()` raises its nominal default of 10, and any limit below 100,
//! to 100) and report non-convergence by shrinking the step by eight with order
//! 1, as `dctran.c` does.
//!
//! The nonlinear initial bias is the shared DC solve (`dctran.c` calls `CKTop`
//! with `CKTdcMaxIter`): request `maxiter=` (deck `itl1`), `srcsteps=`,
//! `gminsteps=` and `gminfactor=` configure it exactly as they configure `.op`
//! (see `docs/port/DC_CONTINUATION.md`). `xmu=` is the trapezoidal weighting of
//! `nicomcof.c` (default 0.5).

mod branch_events;
mod schedule;
pub use schedule::{OutputSampling, TimingSegment, TransientSchedule};
mod interpolation;

use crate::devices::physical_dae::BranchTrace;
use crate::devices::{
    AnalysisMode, BreakTable, Circuit, Forcing, LinearSystem, LoadRequest, SourceBreaks,
    SourceKind, TransientTiming, TrialState,
};
use crate::maths::integrator::TruncationTolerances;
use crate::maths::{Coefficients, IntegrationMethod, StepHistory, Vector};
use crate::primitives::{Complex, Real, SpiceError, SpiceResult, almost_equal_ulps};

use crate::analysis::initial::{self, Hints, VoltageTolerance};
use crate::analysis::linear::{number, plot, unsupported};
use crate::analysis::newton::{CircuitLoad, CircuitNewton};
use crate::analysis::{AnalysisContext, AnalysisPlan, Plot, PointSink};

/// Counters describing one companion transient run.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TransientStats {
    /// Accepted time points after the initial operating point.
    pub accepted: usize,
    /// Trials discarded for truncation error or non-convergence.
    pub rejected: usize,
    /// Accepted steps that ended exactly on a source breakpoint or the stop time.
    pub breakpoints: usize,
    /// Smallest accepted step (0 before any step).
    pub min_step: Real,
    /// Largest accepted step.
    pub max_step: Real,
    /// Observed transient-attempt and LTRA preparation work.
    pub ltra: crate::devices::ltra::LtraWork,
    /// Probes of opt-in device-local event sources at accepted states.
    pub device_event_boundary_probes: usize,
    /// Converged candidate probes of opt-in device event sources.
    pub device_event_trial_probes: usize,
    /// Rejected trials localized to a device crossing.
    pub device_event_retries: usize,
    /// Accepted landings on previously scheduled device boundaries.
    pub device_event_landings: usize,
}

/// Convergence and truncation tolerances. `vntol` applies to node voltages and
/// `abstol` to branch currents (`NIconvTest`); truncation uses `reltol`,
/// `abstol` (on the charge derivative, a current), `chgtol` and `trtol`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Tolerances {
    reltol: Real,
    vntol: Real,
    abstol: Real,
    chgtol: Real,
    trtol: Real,
}

/// A validated `.tran` request for the companion backend.
#[derive(Debug, Clone, PartialEq)]
pub struct TransientPlan {
    step: Real,
    stop: Real,
    start: Real,
    max_step: Real,
    schedule: Option<TransientSchedule>,
    minbreak: Real,
    voltage_lte: Option<crate::analysis::plan::VoltageLte>,
    method: IntegrationMethod,
    max_order: u8,
    tolerances: Tolerances,
    max_steps: usize,
    /// Newton iterations per timepoint (`itl4`, C `CKTtranMaxIter`).
    tran_max_iter: usize,
    /// Trapezoidal weighting (`.option xmu`, C `CKTxmu`), in `[0, 0.5]`.
    xmu: Real,
}

impl TransientPlan {
    fn time_bounds(arguments: &[String]) -> SpiceResult<[Real; 4]> {
        let positional: Vec<_> = arguments.iter().map(String::as_str).collect();
        if !(2..=4).contains(&positional.len()) {
            return Err(unsupported(".tran requires tstep tstop [tstart [tmax]]"));
        }
        let step_size = number(positional.first().copied(), "step time")?;
        let end_time = number(positional.get(1).copied(), "stop time")?;
        let start = number(
            Some(positional.get(2).copied().unwrap_or("0")),
            "start time",
        )?;
        let tmax = number(Some(positional.get(3).copied().unwrap_or("0")), "tmax")?;
        if step_size <= 0. || end_time <= 0. || start < 0. || start >= end_time || tmax < 0. {
            return Err(unsupported(
                "invalid transient time bounds (need tstep > 0, tstop > 0, 0 <= tstart < tstop, tmax >= 0)",
            ));
        }
        // traninit.c: tmax = 0 selects min(tstep, (tstop - tstart)/50).
        let max_step = if tmax > 0. {
            tmax
        } else {
            step_size.min((end_time - start) / 50.)
        };
        Ok([step_size, end_time, start, max_step])
    }

    pub(super) fn resolve(
        positional: &[String],
        options: crate::analysis::plan::TransientOptions,
        newton: &crate::analysis::newton::NewtonOptions,
    ) -> SpiceResult<Self> {
        Ok(Self::from_bounds(
            Self::time_bounds(positional)?,
            options,
            newton,
        ))
    }

    pub(super) const fn from_bounds(
        [output_step, end_time, start_time, max_step]: [Real; 4],
        options: crate::analysis::plan::TransientOptions,
        newton: &crate::analysis::newton::NewtonOptions,
    ) -> Self {
        Self {
            step: output_step,
            stop: end_time,
            start: start_time,
            max_step,
            schedule: None,
            minbreak: options.minbreak,
            voltage_lte: options.voltage_lte,
            method: options.method,
            max_order: options.max_order,
            tolerances: Tolerances {
                reltol: newton.reltol,
                vntol: newton.vntol,
                abstol: newton.abstol,
                chgtol: options.chgtol,
                trtol: options.trtol,
            },
            max_steps: options.max_steps,
            tran_max_iter: options.tran_max_iter,
            xmu: options.xmu,
        }
    }

    pub(super) fn from_schedule(
        schedule: TransientSchedule,
        options: crate::analysis::plan::TransientOptions,
        newton: &crate::analysis::newton::NewtonOptions,
    ) -> Self {
        let mut plan = Self::from_bounds(schedule.bounds(), options, newton);
        plan.schedule = Some(schedule);
        plan
    }

    /// Nominal integration startup and forcing-default time hint.
    #[must_use = "the time hint determines initial integration and implicit forcing timing"]
    pub const fn step(&self) -> Real {
        self.step
    }

    pub(super) fn requested_output_points(&self) -> usize {
        self.schedule
            .as_ref()
            .map_or(0, TransientSchedule::requested_points)
    }

    fn interpolates(&self) -> bool {
        self.schedule
            .as_ref()
            .is_some_and(|schedule| schedule.sampling() == OutputSampling::Interpolate)
    }

    /// Requested first output time.
    #[must_use = "the requested first output time determines which transient points are retained"]
    pub const fn start(&self) -> Real {
        self.start
    }

    /// Final simulation time.
    #[must_use = "the returned end time bounds transient advancement"]
    pub const fn stop(&self) -> Real {
        self.stop
    }

    /// The requested integration family and effective Gear order.
    #[must_use = "the returned integration rule determines the companion equations and effective Gear order"]
    pub const fn method(&self) -> IntegrationMethod {
        self.method
    }

    /// The requested maximum integration order.
    #[must_use = "the returned limit bounds the order of adaptive integration trials"]
    pub const fn max_order(&self) -> u8 {
        self.max_order
    }

    /// Newton iteration limit at each accepted or rejected timepoint.
    #[must_use = "the returned limit bounds Newton work for each transient trial"]
    pub const fn max_iterations(&self) -> usize {
        self.tran_max_iter
    }

    /// Trapezoidal weighting in [0, 0.5].
    #[must_use = "the returned weight is needed to construct the configured trapezoidal coefficients"]
    pub const fn xmu(&self) -> Real {
        self.xmu
    }

    /// Charge (flux) absolute truncation tolerance.
    #[must_use = "the returned absolute charge or flux tolerance controls truncation acceptance"]
    pub const fn chgtol(&self) -> Real {
        self.tolerances.chgtol
    }

    /// Truncation error overestimation factor.
    #[must_use = "the returned overestimation factor scales the truncation error estimate"]
    pub const fn trtol(&self) -> Real {
        self.tolerances.trtol
    }

    /// Reuse the resolved integrator settings for an autonomous one-period map.
    pub(super) fn over_duration(mut self, duration: Real) -> Self {
        self.stop = duration;
        self.start = 0.;
        self.schedule = None;
        self
    }

    pub(super) fn with_max_steps(mut self, steps: usize) -> Self {
        self.max_steps = steps;
        self
    }

    pub(super) fn over_period(mut self, period: Real, resolution: Real) -> Self {
        self.step = period / resolution;
        self.max_step = self.step;
        self.over_duration(period)
    }

    /// Resolved analog breakpoint merge distance (C `CKTminBreak`).
    #[must_use = "the returned interval determines analog breakpoint coalescing"]
    pub fn min_break(&self) -> Real {
        if self.minbreak == 0. {
            10. * self.min_step()
        } else {
            self.minbreak
        }
    }

    /// C `CKTdelmin`.
    fn min_step(&self) -> Real {
        1e-11
            * self
                .schedule
                .as_ref()
                .map_or(self.max_step, TransientSchedule::minimum_ceiling)
    }
}

fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "transient".to_owned(),
        message: message.into(),
    }
}

struct TrialPoint {
    x: Vector,
    state: TrialState,
    guards: BranchTrace,
    seed: DerivativeSeed,
}

enum PhysicalTrial {
    Converged(TrialPoint),
    NotConverged { last: Vector },
}

enum Trial {
    Converged {
        x: Vector,
        state: TrialState,
    },
    /// The iteration limit was reached; `last` is the iterate of the final
    /// load (C's `CKTrhsOld`, the next trial's starting point).
    NotConverged {
        last: Vector,
    },
}

/// The first transient step uses MODEINITTRAN, including every retry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    First,
    Continuing,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Order {
    One,
    Two,
}

impl Order {
    const fn number(self) -> u8 {
        match self {
            Self::One => 1,
            Self::Two => 2,
        }
    }
}

/// C retains `CKTbreak` on rejection, but no longer requests left-limit forcing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Boundary {
    Ordinary,
    Forced,
    Retained,
}

impl Boundary {
    fn at_break(self) -> bool {
        self != Self::Ordinary
    }

    const fn limit(self) -> crate::devices::Limit {
        match self {
            Self::Forced => crate::devices::Limit::Left,
            Self::Ordinary | Self::Retained => crate::devices::Limit::Right,
        }
    }

    const fn retry(self) -> Self {
        match self {
            Self::Forced => Self::Retained,
            other => other,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct NextStep {
    delta: Real,
    order: Order,
}

#[derive(Clone, Copy)]
struct StepClock {
    time: Real,
    next: NextStep,
    save_delta: Real,
    phase: Phase,
}

struct AcceptedCursor {
    clock: StepClock,
    boundary: Boundary,
}

#[derive(Clone, Copy)]
struct ReadyStep {
    clock: StepClock,
    boundary: Boundary,
    event_landing: Option<Real>,
    probe_landing: Option<Real>,
}

struct AttemptedStep {
    clock: StepClock,
    boundary: Boundary,
    coefficients: Coefficients,
    event_start: Option<Real>,
    exact_start: Option<Real>,
}

#[derive(PartialEq, Debug)]
enum StepDecision {
    Accept(NextStep),
    Retry(NextStep),
}

impl AcceptedCursor {
    fn initial(settings: &TransientPlan) -> Self {
        Self {
            clock: StepClock {
                time: 0.,
                // dctran.c: delta = MIN(CKTfinalTime/100, CKTstep)/10.
                next: NextStep {
                    delta: (settings.stop / 100.).min(settings.step) / 10.,
                    order: Order::One,
                },
                save_delta: settings.stop / 50.,
                phase: Phase::First,
            },
            boundary: Boundary::Retained,
        }
    }

    fn prepare(
        self,
        settings: &TransientPlan,
        breaks: &mut BreakTable,
        uic: bool,
    ) -> SpiceResult<ReadyStep> {
        let mut clock = self.clock;
        let min_step = settings.min_step();
        let ceiling = settings
            .schedule
            .as_ref()
            .map(|schedule| schedule.ceiling_at(clock.time));
        clock.next.delta = clock
            .next
            .delta
            .min(ceiling.map_or(settings.max_step, |segment| segment.step));
        // First timepoint after a breakpoint: reduce the order and step.
        if almost_equal_ulps(clock.time, breaks.first(), 100)
            || breaks.first() - clock.time <= min_step
        {
            clock.next.order = Order::One;
            clock.next.delta = clock
                .next
                .delta
                .min(0.1 * clock.save_delta.min(breaks.second() - breaks.first()));
            if clock.phase == Phase::First {
                if uic {
                    breaks.set(clock.time, settings.step)?;
                }
                clock.next.delta /= 10.;
            }
            // XSPICE builds do not raise delta to 2 * CKTdelmin here.
        }
        while (breaks.first() <= clock.time + breaks.min_break()
            || almost_equal_ulps(breaks.first(), clock.time, 100))
            && breaks.first() < settings.stop
        {
            breaks.clear();
        }

        let mut boundary = Boundary::Ordinary;
        if clock.time + clock.next.delta > breaks.first() {
            boundary = Boundary::Forced;
            clock.save_delta = clock.next.delta;
            clock.next.delta = breaks.first() - clock.time;
        } else if !almost_equal_ulps(clock.time + clock.next.delta, settings.stop, 100)
            && clock.next.delta.mul_add(1.9, clock.time) > breaks.first()
        {
            clock.save_delta = clock.next.delta;
            clock.next.delta = (breaks.first() - clock.time) / 2.0;
        }
        let mut ready = ReadyStep {
            clock,
            boundary,
            event_landing: None,
            probe_landing: None,
        };
        if let Some(segment) = ceiling {
            ready.limit_to_sample(segment.until);
        }
        Ok(ready)
    }
}

impl ReadyStep {
    // An output grid is not a source breakpoint: keep the integration
    // order and one-sided forcing policy unless a real breakpoint coincides.
    fn limit_to_sample(&mut self, at: Real) {
        let distance = at - self.clock.time;
        if distance <= 0. {
            return;
        }
        let end = self.end_time();
        if self.clock.next.delta >= distance || end >= at {
            if end > at {
                self.boundary = Boundary::Ordinary;
                self.event_landing = None;
            }
            self.clock.next.delta = distance;
            self.probe_landing = Some(at);
        } else if !almost_equal_ulps(end, at, 100) && self.clock.next.delta * 1.9 > distance {
            self.clock.next.delta = distance / 2.;
            self.boundary = Boundary::Ordinary;
            self.event_landing = None;
            self.probe_landing = None;
        }
    }

    fn shorten(&mut self, maximum: Real) {
        if self.clock.next.delta > maximum {
            self.clock.next.delta = maximum;
            self.event_landing = None;
            self.probe_landing = None;
            self.boundary = Boundary::Ordinary;
        }
    }

    /// A digital or DAC-ramp boundary is not merged into the analog source
    /// breakpoint table: distinct physical event times must not be lost to
    /// its `CKTminBreak` coalescing policy.
    fn limit_to_event(&mut self, at: Real) {
        let distance = at - self.clock.time;
        if distance > 0. && self.end_time() >= at {
            if self.clock.next.delta > distance {
                self.clock.save_delta = self.clock.next.delta;
            }
            self.clock.next.delta = distance;
            self.boundary = Boundary::Forced;
            self.event_landing = Some(at);
            self.probe_landing = None;
        }
    }

    fn end_time(&self) -> Real {
        self.event_landing
            .or(self.probe_landing)
            .unwrap_or(self.clock.time + self.clock.next.delta)
    }

    fn begin(self, steps: &StepHistory, settings: &TransientPlan) -> SpiceResult<AttemptedStep> {
        let end = self.end_time();
        let mut clock = self.clock;
        let event_start = self.event_landing.map(|_| clock.time);
        let exact_start = self
            .event_landing
            .or(self.probe_landing)
            .map(|_| clock.time);
        clock.time = end;
        let coefficients = steps.trial(
            settings.method,
            clock.next.order.number(),
            clock.next.delta,
            settings.xmu,
        )?;
        Ok(AttemptedStep {
            clock,
            boundary: self.boundary,
            coefficients,
            event_start,
            exact_start,
        })
    }
}

impl AttemptedStep {
    fn retry(self, mut next: NextStep, min_step: Real) -> SpiceResult<ReadyStep> {
        let mut clock = self.clock;
        // C adds then subtracts; restoring the previous accepted time would
        // change rounding. Only this retry cursor, never the history, moves.
        clock.time = self.exact_start.unwrap_or(clock.time - clock.next.delta);
        if next.delta <= min_step {
            if clock.next.delta > min_step {
                next.delta = min_step;
            } else {
                return Err(failure(format!(
                    "timestep too small at t = {:e} (minimum {min_step:e})",
                    clock.time
                )));
            }
        }
        clock.next = next;
        Ok(ReadyStep {
            clock,
            boundary: self.boundary.retry(),
            event_landing: None,
            probe_landing: None,
        })
    }

    const fn accept(self, mut next: NextStep) -> AcceptedCursor {
        let mut clock = self.clock;
        clock.phase = Phase::Continuing;
        if self.event_start.is_some() {
            next.order = Order::One;
        }
        clock.next = next;
        AcceptedCursor {
            clock,
            boundary: self.boundary,
        }
    }

    /// Probe the next step without changing the attempted coefficients or order.
    fn decision(
        &self,
        settings: &TransientPlan,
        steps: &StepHistory,
        mut truncate: impl FnMut(&Coefficients) -> SpiceResult<Real>,
    ) -> SpiceResult<StepDecision> {
        let mut next = self.clock.next;
        if self.clock.phase == Phase::First || steps.is_empty() {
            return Ok(StepDecision::Accept(next));
        }
        let delta = next.delta;
        next.delta = (2. * delta).min(truncate(&self.coefficients)?);
        if next.delta > 0.9 * delta {
            // dctran.c never raises above order 2. Keep the probed timestep
            // even when the order-2 probe does not justify raising the order.
            if next.order == Order::One
                && settings.max_order > 1
                && steps.available_order(settings.method) >= 2
            {
                let probe = steps.trial(settings.method, 2, delta, settings.xmu)?;
                next.order = Order::Two;
                next.delta = (2. * delta).min(truncate(&probe)?);
                if next.delta <= 1.05 * delta {
                    next.order = Order::One;
                }
            }
            Ok(StepDecision::Accept(next))
        } else {
            Ok(StepDecision::Retry(next))
        }
    }
}

/// A deterministic closed grid for shooting output. It constrains the
/// next proposed step without introducing an artificial source breakpoint.
struct SampleGrid {
    index: u16,
    intervals: u16,
    stop: Real,
}

impl SampleGrid {
    const fn new(intervals: u16, stop: Real) -> Self {
        Self {
            index: 0,
            intervals,
            stop,
        }
    }

    fn time(&self) -> Real {
        if self.index == self.intervals {
            self.stop
        } else {
            self.stop * (Real::from(self.index) / Real::from(self.intervals))
        }
    }

    fn at_sample(&self, time: Real) -> bool {
        almost_equal_ulps(time, self.time(), 100)
    }

    fn advance(&mut self, on_sample: bool) -> SpiceResult<Real> {
        if on_sample {
            if self.index >= self.intervals {
                return Err(failure("periodic output grid ended before the orbit"));
            }
            self.index += 1;
        }
        Ok(self.time())
    }
}

enum OutputGrid {
    Orbit(SampleGrid),
    Scheduled(schedule::ScheduledGrid),
}

impl OutputGrid {
    fn time(&self) -> Real {
        match self {
            Self::Orbit(grid) => grid.time(),
            Self::Scheduled(grid) => grid.time(),
        }
    }

    fn at_sample(&self, time: Real) -> bool {
        match self {
            Self::Orbit(grid) => grid.at_sample(time),
            Self::Scheduled(grid) => time.to_bits() == grid.time().to_bits(),
        }
    }

    fn advance(&mut self, on_sample: bool) -> SpiceResult<Real> {
        match self {
            Self::Orbit(grid) => grid.advance(on_sample),
            Self::Scheduled(grid) => grid.advance(on_sample),
        }
    }
}

#[derive(Clone, Copy)]
struct OutputSelection<'a> {
    sample_at: Option<Real>,
    capture_boundary: bool,
    selected: Option<&'a [usize]>,
}

struct CandidateOutput {
    observed: Option<Vec<Complex>>,
    periodic_boundary: Option<Box<PeriodicBoundary>>,
}

/// Converged point with all fallible observations and mixed-domain
/// preparation complete, ready to publish after releasing trial borrows.
struct AcceptedCandidate {
    output: CandidateOutput,
    mixed: Option<super::digital::MixedPoint>,
    solution: Vector,
    state: TrialState,
}

struct InitialCandidate {
    solution: Vector,
    state: TrialState,
    reported_bias: Option<Vector>,
}

struct AcceptedPoint {
    cursor: AcceptedCursor,
    solution: Vector,
    reported_bias: Option<Vector>,
    observed: Option<Vec<Complex>>,
    device_breaks: Vec<Real>,
    periodic_boundary: Option<Box<PeriodicBoundary>>,
}

#[derive(Clone, Copy)]
enum Acceptance<'a> {
    Initial,
    Step(&'a AttemptedStep),
}

struct TruncationDevice {
    rows: std::ops::Range<usize>,
    slots: Vec<TruncationSlot>,
}

struct TruncationSlot {
    base: usize,
    explicit_derivative: bool,
    derived: bool,
}

/// Accepted voltage history used only when newtrunc is requested. Rejected
/// trials never enter this history. Breakpoint landings start a new predictor
/// epoch; an algebraic jump must not force an arbitrarily small next step.
struct VoltageHistory {
    options: crate::analysis::plan::VoltageLte,
    rows: Vec<usize>,
    accepted: std::collections::VecDeque<Vector>,
}

impl VoltageHistory {
    fn new(circuit: &Circuit, options: crate::analysis::plan::VoltageLte) -> Self {
        let rows = circuit
            .physical_voltage_nodes()
            .filter_map(|node| circuit.unknowns().node_row(node.id))
            .collect();
        Self {
            options,
            rows,
            accepted: std::collections::VecDeque::new(),
        }
    }

    fn accept(&mut self, solution: &Vector, acceptance: Acceptance<'_>) {
        if matches!(acceptance, Acceptance::Step(attempt) if attempt.boundary == Boundary::Forced) {
            self.accepted.clear();
            return;
        }
        self.accepted.push_front(solution.clone());
        self.accepted
            .truncate(usize::from(IntegrationMethod::MAX_GEAR_ORDER) + 1);
    }

    fn limit(&self, coefficients: &Coefficients, solution: &Vector) -> SpiceResult<Real> {
        let order = usize::from(coefficients.order());
        if self.accepted.len() <= order {
            return Ok(Real::INFINITY);
        }
        let dt = coefficients.dt();
        let (multiplier, exponent) = match coefficients.method() {
            IntegrationMethod::Trapezoidal => (2., 1. / Real::from(coefficients.order())),
            IntegrationMethod::Gear { .. } => {
                let mut ratio = 0.;
                for age in 0..=order {
                    ratio += coefficients
                        .delta_old(age)
                        .ok_or_else(|| failure("voltage predictor step history is too short"))?
                        / dt;
                }
                (ratio, 1. / (Real::from(coefficients.order()) + 1.))
            }
        };
        let mut bound = Real::INFINITY;
        for &row in &self.rows {
            let mut past = [0.; IntegrationMethod::MAX_GEAR_ORDER as usize + 1];
            for (value, accepted) in past.iter_mut().zip(&self.accepted) {
                *value = accepted.as_slice()[row];
            }
            let predicted = coefficients.predict(&past[..=order])?;
            let corrected = solution.as_slice()[row];
            let error = (corrected - predicted).abs();
            if error > 0. {
                let tolerance = self
                    .options
                    .relative
                    .mul_add(corrected.abs().max(predicted.abs()), self.options.absolute);
                // C CKTtrunc's predictor/corrector voltage bound. Evaluate
                // the ratio before multiplying by dt; an unbounded proposal
                // is harmless because the driver already caps growth.
                let ratio = (tolerance / error) * self.options.factor * multiplier;
                bound = bound.min(dt * ratio.powf(exponent));
            }
        }
        Ok(bound)
    }
}

/// A disposable physical solution's derivative seed. It becomes a
/// predictor only when the corresponding physical point is accepted.
#[derive(Clone)]
struct DerivativeSeed {
    rate: Vector,
    order_hint: usize,
}

impl DerivativeSeed {
    const fn guess(&self) -> initial::DerivativeGuess<'_> {
        initial::DerivativeGuess {
            order_hint: self.order_hint,
            rate: Some(&self.rate),
        }
    }
}

/// Most recent accepted physical point, retained solely as a Newton seed.
struct DerivativePredictor {
    time: Real,
    state: Vector,
    seed: DerivativeSeed,
}

impl DerivativePredictor {
    fn at(&self, time: Real) -> Option<Vector> {
        let delta = time - self.time;
        let guess = Vector::from(
            self.state
                .as_slice()
                .iter()
                .zip(self.seed.rate.as_slice())
                .map(|(&x, &rate)| {
                    let predicted = rate.mul_add(delta, x);
                    // Extrapolation can land on a zero of a singular residual.
                    // Within the rounding envelope of the cancelling terms,
                    // prefer exact zero as a disposable Newton seed. The
                    // physical residual and rank checks still certify the germ.
                    let scale = x.abs().max((rate * delta).abs());
                    if scale.is_finite() && predicted.abs() <= 2. * Real::EPSILON * scale {
                        0.
                    } else {
                        predicted
                    }
                })
                .collect::<Vec<_>>(),
        );
        guess.is_finite().then_some(guess)
    }
}

struct EventDevice<'a> {
    source: &'a dyn crate::devices::traits::DeviceTransientEvents,
    rows: std::ops::Range<usize>,
}

struct Driver<'a> {
    circuit: &'a Circuit,
    prepared: crate::devices::PreparedRun<'a>,
    system: &'a LinearSystem,
    settings: TransientPlan,
    timing: TransientTiming,
    model_context: crate::devices::ModelContext,
    /// Newton/continuation settings of the nonlinear initial bias.
    bias: crate::analysis::bias::DcSettings,
    history: crate::devices::StateHistory,
    steps: StepHistory,
    hints: Hints,
    uic: bool,
    /// `true` for rows that carry branch currents (`abstol`), else `vntol`.
    branch_row: Vec<bool>,
    /// Global damping's watched rows, fixed by finalized device topology.
    limited_rows: Option<Vec<bool>>,
    /// Each device's state rows and the absolute state index of every
    /// charge it hands `CKTterr` (`truncation_slots`), fixed for the run.
    truncation: Vec<TruncationDevice>,
    event_devices: Vec<EventDevice<'a>>,
    voltage_history: Option<VoltageHistory>,
    stats: TransientStats,
    /// Reuse the ordinary Newton MNA/RHS/trial storage across timepoint
    /// acceptance and rejected smaller-step retries.
    workspace: crate::analysis::newton::CircuitNewtonWorkspace,
    mixed: Option<&'a mut super::digital::MixedSignal>,
    index_aware: bool,
    derivative_predictor: Option<DerivativePredictor>,
    physical_events: Option<std::iter::Peekable<crate::devices::SystemBreakpoints>>,
    accepted_guards: BranchTrace,
}

/// Runs an ordinary `.tran` with the companion backend, returning the plot and
/// run statistics.
///
/// `request` uses the same positional `tstep tstop [tstart [tmax]]` arguments
/// as C plus named `method` (`trap` default, `trapezoidal`, `gear`), `maxord`
/// (1 to 6; only 1 versus more than 1 matters, as in `dctran.c`), `rtol`, `vntol`, `abstol`, `chgtol`, `trtol`, `maxsteps`,
/// `tranmaxiter` (Newton iterations per timepoint, `itl4`), `xmu` (trapezoidal
/// weighting) and the initial-bias `maxiter`, `srcsteps`, `gminsteps`,
/// `gminfactor`.
///
/// # Errors
///
/// Invalid or unsupported requests (`maxord` outside 1..=6, unknown options), `.ic` or
/// `.nodeset` entries naming unknown nodes, an `.ic` contradicting ideal
/// sources, `uic` initial conditions that would need an impulse at `t = 0+`,
/// structural or singular systems, a failing accept hook, an exceeded
/// breakpoint/work limit, or "timestep too small". A failed initialization
/// returns before any device accept hook ran or any plot row exists.
pub fn companion_transient(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
) -> SpiceResult<(Plot, TransientStats)> {
    let mut collector = super::results::PlotCollector::default();
    let stats = companion_transient_into(circuit, request, context, &mut collector)?;
    let plot = collector
        .finish(1)?
        .pop()
        .ok_or_else(|| super::results::lifecycle_error("transient driver returned no plot"))?;
    Ok((plot, stats))
}

/// [`companion_transient`], handing every accepted point to `sink` as it is
/// accepted instead of storing it, as C's `dctran.c` hands each one to
/// `CKTdump`/`OUTpData`.
///
/// `sink` receives the plot header before the initial operating point is
/// solved, then exactly the rows [`companion_transient`] would store, followed
/// by a successful end event. The returned statistics contain no plot data.
///
/// # Errors
///
/// As [`companion_transient`], and whatever `sink` reports.
pub fn companion_transient_into(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    sink: &mut dyn PointSink,
) -> SpiceResult<TransientStats> {
    let settings = request
        .transient()
        .ok_or_else(|| unsupported("companion requires a transient plan"))?
        .clone();
    let result = run_transient(
        circuit,
        request,
        context,
        settings,
        TransientRun::Ordinary,
        sink,
    )?;
    circuit.record_final_transient_state(&result.history)?;
    Ok(result.stats)
}

/// Run a typed mixed-signal transient using the ordinary companion solver.
/// Digital model state is settled outside Newton iterations, and DAC output
/// ramps are read-only during each analog attempt.
///
/// # Errors
/// The ordinary transient errors, invalid mixed-signal bindings, unsupported
/// UIC initialization, or an exceeded digital event/initialization budget.
pub fn mixed_signal_transient(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    mixed: &mut super::digital::MixedSignal,
) -> SpiceResult<(Plot, TransientStats)> {
    let mut collector = super::results::PlotCollector::default();
    let stats = mixed_signal_transient_into(circuit, request, context, mixed, &mut collector)?;
    let plot = collector
        .finish(1)?
        .pop()
        .ok_or_else(|| super::results::lifecycle_error("mixed transient returned no plot"))?;
    Ok((plot, stats))
}

/// Streaming form of [`mixed_signal_transient`].
///
/// # Errors
/// As [`mixed_signal_transient`], and any sink failure.
pub fn mixed_signal_transient_into(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    mixed: &mut super::digital::MixedSignal,
    sink: &mut dyn PointSink,
) -> SpiceResult<TransientStats> {
    let settings = request
        .transient()
        .ok_or_else(|| unsupported("mixed signal requires a transient plan"))?
        .clone();
    if request.uic() {
        return Err(unsupported("mixed-signal UIC initialization"));
    }
    mixed.activate(circuit)?;
    let result = run_transient_inner(
        circuit,
        request,
        context,
        settings,
        TransientRun::Ordinary,
        sink,
        Some(mixed),
    );
    mixed.deactivate();
    let result = result?;
    circuit.record_final_transient_state(&result.history)?;
    Ok(result.stats)
}

/// The shooting map uses the very same adaptive step and acceptance loop as
/// ordinary transient analysis, but never publishes intermediate orbit trials.
#[derive(Clone, Copy)]
pub(super) enum TransientRun<'a> {
    Ordinary,
    Seed,
    Shooting {
        seed: &'a Vector,
        points: u16,
        output: OrbitOutput<'a>,
    },
}

impl TransientRun<'_> {
    fn output(self, circuit: &Circuit) -> SpiceResult<super::linear::LinearOutput> {
        let (name, title) = match self {
            TransientRun::Ordinary | TransientRun::Seed => ("tran1", "Transient Analysis"),
            TransientRun::Shooting { .. } => ("pss1", "Periodic Steady State Analysis"),
        };
        plot(
            circuit,
            name,
            title,
            Some(("time", "time")),
            false,
            crate::devices::passive_network::ReportPhase::Transient,
        )
    }
}

/// The probes and verification see only physical state, never optional
/// observations. The emission pass reuses the verified endpoint's transient
/// proposal to ask exactly the selected device quantities at phase zero.
#[derive(Clone, Copy)]
pub(super) enum OrbitOutput<'a> {
    Probe,
    Verify,
    Emit(&'a PeriodicBoundary),
}

/// A converged, pre-acceptance transient load at the periodic boundary.
/// Its trial revision and accepted history belong together: reusing the
/// post-acceptance history would double-count the last integration step.
pub(super) struct PeriodicBoundary {
    history: crate::devices::StateHistory,
    coefficients: Coefficients,
    trial: TrialState,
    solution: Vector,
    time: Real,
    forcing: Forcing,
}

impl PeriodicBoundary {
    fn capture(
        history: &crate::devices::StateHistory,
        attempt: &AttemptedStep,
        trial: &TrialState,
        solution: &Vector,
        timing: TransientTiming,
    ) -> Self {
        Self {
            history: history.clone(),
            coefficients: attempt.coefficients.clone(),
            trial: trial.clone(),
            solution: solution.clone(),
            time: attempt.clock.time,
            forcing: Forcing {
                limit: attempt.boundary.limit(),
                timing,
            },
        }
    }

    fn observe(
        &self,
        prepared: &crate::devices::PreparedRun<'_>,
        context: &crate::devices::ModelContext,
        selected: Option<&[usize]>,
    ) -> SpiceResult<Vec<Complex>> {
        let mut attempt = prepared.transient_attempt(
            &self.history,
            context,
            self.time,
            &self.coefficients,
            self.forcing,
        )?;
        attempt.observe_selected(&self.solution, &self.trial, selected)
    }
}

pub(super) struct TransientEnd {
    pub solution: Vector,
    pub history: crate::devices::StateHistory,
    pub initial_state: Option<Vec<Real>>,
    pub charge_scales: Option<Vec<Real>>,
    pub boundary: Option<Box<PeriodicBoundary>>,
    pub stats: TransientStats,
}

pub(super) fn run_transient(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    settings: TransientPlan,
    purpose: TransientRun<'_>,
    sink: &mut dyn PointSink,
) -> SpiceResult<TransientEnd> {
    run_transient_inner(circuit, request, context, settings, purpose, sink, None)
}

fn truncation_devices(circuit: &Circuit) -> SpiceResult<Vec<TruncationDevice>> {
    circuit
        .devices()
        .iter()
        .enumerate()
        .map(|(index, device)| {
            let rows = circuit
                .state_rows(index)
                .ok_or_else(|| failure("missing state range"))?;
            let bases = device
                .truncation_slots()
                .into_iter()
                .map(|slot| TruncationSlot {
                    base: rows.start + slot,
                    explicit_derivative: device
                        .initial_expression_storage()
                        .is_some_and(|storage| storage.slot == slot),
                    derived: device.initial_expression_storage().is_some_and(|storage| {
                        storage.slot == slot
                            && storage.policy == crate::devices::InitialStoragePolicy::Derived
                    }),
                })
                .collect();
            Ok(TruncationDevice { rows, slots: bases })
        })
        .collect()
}

fn event_devices(circuit: &Circuit) -> SpiceResult<Vec<EventDevice<'_>>> {
    circuit
        .devices()
        .iter()
        .enumerate()
        .filter_map(|(index, device)| device.transient_events().map(|source| (index, source)))
        .map(|(index, source)| {
            Ok(EventDevice {
                source,
                rows: circuit
                    .state_rows(index)
                    .ok_or_else(|| failure("missing event device state range"))?,
            })
        })
        .collect()
}

fn run_transient_inner(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    settings: TransientPlan,
    purpose: TransientRun<'_>,
    sink: &mut dyn PointSink,
    mixed: Option<&mut super::digital::MixedSignal>,
) -> SpiceResult<TransientEnd> {
    let bias = request.dc_settings().clone();
    let uic = request.uic() && !matches!(purpose, TransientRun::Shooting { .. });
    if uic && let Some(line) = circuit.devices().iter().find(|d| d.delay_line().is_some()) {
        // traload.c/ltraload.c start the delayed waves from the instance
        // `ic=`/`v1 i1 v2 i2` under MODEUIC, but the port's impulse check
        // (initial::check_impulse_free) has no transmission-line
        // formulation yet.
        let reference = if line.designator() == 'o' {
            "src/spicelib/devices/ltra/ltraload.c (MODEINITTRAN | MODEUIC)"
        } else {
            "src/spicelib/devices/tra/traload.c (MODEINITTRAN | MODEUIC)"
        };
        return Err(SpiceError::not_yet_ported(
            format!(
                ".tran uic with transmission line {} (initial waves from its ic=)",
                line.name()
            ),
            reference,
        ));
    }
    // Resolve row-indexed hints only after topology changes have been finalized.
    circuit.finalize()?;
    let hints = initial::resolve(circuit, request)?;
    let model_context = context.model_context();
    let nonlinear = circuit.devices().iter().any(|d| d.is_nonlinear());
    let mut system = if nonlinear {
        circuit
            .prepare_run()
            .transient_forcing_system(&model_context)?
    } else {
        circuit.linear_system_with_context(&model_context)?
    };
    let timing = TransientTiming::new(settings.step, settings.stop)?;
    system.bind_transient_timing(&timing)?;
    if !matches!(purpose, TransientRun::Ordinary) {
        if let Some(source) = system.sources.iter().find(|source| {
            !matches!(
                source.waveform.as_ref(),
                crate::devices::Waveform::Constant(_)
            )
        }) {
            return Err(unsupported(format!(
                ".pss requires autonomous forcing; {} has a time-dependent waveform",
                source.name
            )));
        }
        if let Some(device) = circuit.devices().iter().find(|device| {
            device.delay_line().is_some()
                || device.depends_on_time()
                || matches!(device.designator(), 's' | 'w')
        }) {
            return Err(unsupported(format!(
                ".pss cannot restart the history or explicit time dependence of {}",
                device.name()
            )));
        }
    }
    let circuit: &Circuit = circuit;
    let branch_row = crate::analysis::bias::branch_rows(circuit);
    let limited_rows = crate::analysis::bias::limited_rows(circuit);
    let truncation = truncation_devices(circuit)?;
    let event_devices = event_devices(circuit)?;
    // A timed receiver must not miss a whole pulse because the ordinary
    // analog breakpoint table coalesced its two known source corners.
    let physical_events = (!event_devices.is_empty())
        .then(|| {
            system
                .breakpoints_in(0., settings.stop)
                .map(Iterator::peekable)
        })
        .transpose()?;
    let driver = Driver {
        circuit,
        prepared: circuit.prepare_run(),
        system: &system,
        timing,
        model_context,
        bias,
        history: circuit.state_history()?,
        steps: StepHistory::with_fill(settings.max_step),
        hints,
        uic,
        branch_row,
        limited_rows,
        truncation,
        event_devices,
        voltage_history: settings
            .voltage_lte
            .map(|options| VoltageHistory::new(circuit, options)),
        stats: TransientStats::default(),
        workspace: crate::analysis::newton::CircuitNewtonWorkspace::default(),
        mixed,
        index_aware: false,
        derivative_predictor: None,
        physical_events,
        accepted_guards: BranchTrace::default(),
        settings,
    };
    if let Some(schedule) = driver
        .settings
        .schedule
        .as_ref()
        .filter(|schedule| schedule.sampling() == OutputSampling::Interpolate)
    {
        let mut interpolated = interpolation::InterpolatedSink::new(schedule, uic, sink)?;
        driver.run(purpose, &mut interpolated)
    } else {
        driver.run(purpose, sink)
    }
}

impl Driver<'_> {
    fn observe_initial(
        &self,
        solution: &Vector,
        selected: Option<&[usize]>,
        report_bias: bool,
    ) -> SpiceResult<Vec<Complex>> {
        if self.index_aware && !report_bias {
            return self.observe_derivative_point(
                solution,
                None,
                0.,
                crate::devices::Limit::Right,
                selected,
            );
        }
        let request = LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution,
            model_context: &self.model_context,
            integration: None,
            history: &self.history,
            forcing: None,
        };
        let forcing = Some((
            0.,
            Forcing {
                limit: crate::devices::Limit::Left,
                timing: self.timing,
            },
        ));
        selected.map_or_else(
            || self.prepared.observe_real(&request, &[], forcing, None),
            |indices| {
                self.prepared
                    .observe_real_selected(&request, &[], forcing, None, indices)
            },
        )
    }

    fn observe_derivative_point(
        &self,
        solution: &Vector,
        state: Option<&TrialState>,
        time: Real,
        limit: crate::devices::Limit,
        selected: Option<&[usize]>,
    ) -> SpiceResult<Vec<Complex>> {
        let forcing = Forcing {
            limit,
            timing: self.timing,
        };
        let request = LoadRequest {
            mode: AnalysisMode::Instantaneous { time },
            solution,
            model_context: &self.model_context,
            integration: None,
            history: &self.history,
            forcing: Some(forcing),
        };
        selected.map_or_else(
            || {
                self.prepared
                    .observe_real(&request, &[], Some((time, forcing)), state)
            },
            |indices| {
                self.prepared.observe_real_selected(
                    &request,
                    &[],
                    Some((time, forcing)),
                    state,
                    indices,
                )
            },
        )
    }

    fn run(
        mut self,
        purpose: TransientRun<'_>,
        sink: &mut dyn PointSink,
    ) -> SpiceResult<TransientEnd> {
        let TransientPlan { stop, start, .. } = self.settings;
        let start = if self.settings.interpolates() {
            0.
        } else {
            start
        };
        // traninit.c: CKTdelmin; dctran.c (XSPICE build): CKTminBreak.
        let mut breaks = BreakTable::new(stop, self.settings.min_break())?;
        let mut sources = SourceBreaks::new(self.system);
        let output = purpose.output(self.circuit)?;
        sink.begin(&output.header)?;
        let mut selected = crate::analysis::results::SelectedOutput::bind(
            sink,
            &output,
            self.circuit.unknown_count(),
            true,
        )?;
        let mut point = Vec::with_capacity(output.header.variable_count());
        let mut accepted = match purpose {
            TransientRun::Shooting { seed, output, .. } => self.initialize_orbit(
                seed,
                output,
                selected
                    .as_ref()
                    .map(crate::analysis::results::SelectedOutput::observations),
            )?,
            TransientRun::Ordinary | TransientRun::Seed => self.initialize()?,
        };
        let initial_state = match purpose {
            TransientRun::Shooting { .. } => self.history.accepted(1).map(<[Real]>::to_vec),
            _ => None,
        };
        let mut charge_scales = initial_state
            .as_ref()
            .map(|values| vec![0.0_f64; values.len()]);
        let mut sample = match purpose {
            TransientRun::Shooting { points, .. } => {
                Some(OutputGrid::Orbit(SampleGrid::new(points, stop)))
            }
            _ => self
                .settings
                .schedule
                .as_ref()
                .filter(|schedule| schedule.sampling() == OutputSampling::Land)
                .map(|schedule| schedule.grid().map(OutputGrid::Scheduled))
                .transpose()?,
        };
        let capture_boundary = matches!(
            purpose,
            TransientRun::Shooting {
                output: OrbitOutput::Verify,
                ..
            }
        );
        loop {
            self.accumulate_charge_scales(charge_scales.as_deref_mut())?;
            let time = accepted.cursor.clock.time;
            let on_sample = sample.as_ref().is_none_or(|grid| grid.at_sample(time));
            let output_time = sample.as_ref().map_or(time, OutputGrid::time);
            self.accept_breaks(&mut accepted, &mut sources, &mut breaks)?;
            if on_sample && accepted.cursor.clock.time >= start {
                self.emit_accepted(
                    &accepted,
                    output_time,
                    &output,
                    sink,
                    &mut selected,
                    &mut point,
                )?;
            }
            let exact_stop = self.settings.schedule.is_some()
                || self.physical_events.is_some()
                || sources.has_exact();
            if (exact_stop && time >= stop) || (!exact_stop && almost_equal_ulps(time, stop, 100)) {
                break;
            }
            let AcceptedPoint {
                cursor, solution, ..
            } = accepted;
            let next_sample = sample
                .as_mut()
                .map(|grid| grid.advance(on_sample))
                .transpose()?;
            let mut ready = cursor.prepare(&self.settings, &mut breaks, self.uic)?;
            if let Some(at) = next_sample {
                ready.limit_to_sample(at);
            }
            self.limit_source_step(&mut ready, &sources, time)?;
            accepted = self.advance(
                ready,
                solution,
                OutputSelection {
                    sample_at: next_sample,
                    capture_boundary,
                    selected: selected
                        .as_ref()
                        .map(crate::analysis::results::SelectedOutput::observations),
                },
            )?;
        }
        sink.end()?;
        // C keeps the final load's instance fields, not values re-evaluated
        // at the final solution (the load can lag that solution by an iterate).
        // Publish only after the entire run and every sink write succeeded.
        self.stats.ltra = self.prepared.ltra_work();
        Ok(TransientEnd {
            solution: accepted.solution,
            history: self.history,
            initial_state,
            charge_scales,
            boundary: accepted.periodic_boundary,
            stats: self.stats,
        })
    }

    fn emit_accepted(
        &self,
        accepted: &AcceptedPoint,
        time: Real,
        output: &super::linear::LinearOutput,
        sink: &mut dyn PointSink,
        selected: &mut Option<crate::analysis::results::SelectedOutput>,
        point: &mut Vec<Complex>,
    ) -> SpiceResult<()> {
        // C (dctran.c) writes no t = 0 row under uic.
        if accepted.cursor.clock.time == 0. && self.uic && !self.settings.interpolates() {
            return Ok(());
        }
        let solution = accepted
            .reported_bias
            .as_ref()
            .unwrap_or(&accepted.solution);
        let initial;
        let observations = match accepted.observed.as_deref() {
            Some(values) => values,
            None if accepted.cursor.clock.time == 0. => {
                initial = self.observe_initial(
                    solution,
                    selected
                        .as_ref()
                        .map(crate::analysis::results::SelectedOutput::observations),
                    accepted.reported_bias.is_some(),
                )?;
                &initial
            }
            None => return Err(failure("missing accepted output observations")),
        };
        push(
            output,
            sink,
            selected.as_mut(),
            point,
            time,
            solution,
            observations,
        )
    }

    fn accumulate_charge_scales(&self, scales: Option<&mut [Real]>) -> SpiceResult<()> {
        if let Some(scales) = scales {
            let state = self
                .history
                .accepted(1)
                .ok_or_else(|| failure("periodic charge history is empty"))?;
            for TruncationDevice { slots, .. } in &self.truncation {
                for TruncationSlot { base: slot, .. } in slots {
                    let slot = *slot;
                    scales[slot] = scales[slot].max(state[slot].abs());
                }
            }
        }
        Ok(())
    }

    fn accept_breaks(
        &self,
        accepted: &mut AcceptedPoint,
        sources: &mut SourceBreaks<'_>,
        breaks: &mut BreakTable,
    ) -> SpiceResult<()> {
        // nextTime: CKTaccept (device hooks ran when the point was
        // committed), the sources' breakpoint requests, then output.
        // CKTaccept runs DEVaccept in dev.c type order: isrc, then tra
        // (the delay lines' requests), then vsrc.
        sources.accept(
            SourceKind::Current,
            accepted.cursor.clock.time,
            accepted.cursor.boundary.at_break(),
            &self.timing,
            breaks,
        )?;
        for time in std::mem::take(&mut accepted.device_breaks) {
            breaks.set(accepted.cursor.clock.time, time)?;
        }
        sources.accept(
            SourceKind::Voltage,
            accepted.cursor.clock.time,
            accepted.cursor.boundary.at_break(),
            &self.timing,
            breaks,
        )?;
        if accepted.cursor.clock.time > breaks.first() {
            breaks.clear();
        }
        Ok(())
    }

    /// A shooting candidate supplies the MNA unknowns, not a DC bias.
    /// Rebuild device charge and flux at that point using an exact load;
    /// the ordinary first-step backward-Euler initialization then reconstructs
    /// derivatives without reusing a stale accepted integration history.
    fn initialize_orbit(
        &mut self,
        seed: &Vector,
        output: OrbitOutput<'_>,
        selected: Option<&[usize]>,
    ) -> SpiceResult<AcceptedPoint> {
        if seed.len() != self.circuit.unknown_count() || !seed.is_finite() {
            return Err(failure("invalid periodic orbit restart vector"));
        }
        let trial = self.initial_state(seed, &[], false)?;
        let device_breaks = self.commit_point(Acceptance::Initial, seed, trial)?;
        let observed = match output {
            OrbitOutput::Probe | OrbitOutput::Verify => Vec::new(),
            OrbitOutput::Emit(boundary) => {
                boundary.observe(&self.prepared, &self.model_context, selected)?
            }
        };
        Ok(AcceptedPoint {
            cursor: AcceptedCursor::initial(&self.settings),
            solution: seed.clone(),
            reported_bias: None,
            observed: Some(observed),
            device_breaks,
            periodic_boundary: None,
        })
    }

    fn limit_source_step(
        &self,
        ready: &mut ReadyStep,
        sources: &SourceBreaks<'_>,
        time: Real,
    ) -> SpiceResult<()> {
        self.limit_mixed_step(ready, time)?;
        // The derivative solver's merged iterator already contains
        // every native and prepared event. Do not schedule it twice.
        if !self.index_aware
            && let Some(at) = sources.next_exact(time, self.settings.stop)?
        {
            ready.limit_to_event(at);
        }
        Ok(())
    }

    fn limit_mixed_step(&self, ready: &mut ReadyStep, time: Real) -> SpiceResult<()> {
        if let Some(mixed) = self.mixed.as_deref() {
            if mixed.boundary() && time > 0. {
                ready.clock.next.order = Order::One;
                ready.shorten(0.1 * ready.clock.save_delta);
            }
            if let Some(at) = mixed.next_ramp_boundary(time)? {
                ready.limit_to_event(at);
            }
        }
        Ok(())
    }

    fn mixed_retry(
        &self,
        attempt: &AttemptedStep,
        solution: &Vector,
    ) -> SpiceResult<Option<NextStep>> {
        let Some(mixed) = self.mixed.as_deref() else {
            return Ok(None);
        };
        let time = attempt.clock.time;
        let min_step = self.settings.min_step();
        if let Some(at) = mixed.first_crossing(self.circuit, solution, time)?
            && time - at > min_step
        {
            let start = attempt
                .exact_start
                .unwrap_or(time - attempt.clock.next.delta);
            let delta = (at - start).max(min_step).max(start.next_up() - start);
            if delta < attempt.clock.next.delta {
                return Ok(Some(NextStep {
                    delta,
                    order: Order::One,
                }));
            }
        }
        Ok(None)
    }

    /// Cold-indexed event sources probe only accepted local state; no
    /// per-device scan or additional event queue in the analog-only path.
    fn next_device_boundary(&self, time: Real) -> SpiceResult<Option<Real>> {
        if self.event_devices.is_empty() {
            return Ok(None);
        }
        let history = self
            .history
            .accepted(1)
            .ok_or_else(|| failure("missing accepted event state"))?;
        let mut next = None::<Real>;
        for event in &self.event_devices {
            let slots = history
                .get(event.rows.clone())
                .ok_or_else(|| failure("invalid event state window"))?;
            if let Some(at) = event.source.next_boundary(slots)? {
                if !at.is_finite() || at <= time {
                    return Err(failure("invalid device event boundary"));
                }
                next = Some(next.map_or(at, |prior| prior.min(at)));
            }
        }
        Ok(next)
    }

    fn refine_device_events(
        &self,
        solution: &Vector,
        state: &mut TrialState,
        time: Real,
    ) -> SpiceResult<Option<Real>> {
        if self.event_devices.is_empty() {
            return Ok(None);
        }
        let history = self
            .history
            .accepted(1)
            .ok_or_else(|| failure("missing accepted event state"))?;
        let mut next = None::<Real>;
        for event in &self.event_devices {
            let context = crate::devices::traits::TransientEventContext {
                accepted: history
                    .get(event.rows.clone())
                    .ok_or_else(|| failure("invalid event state window"))?,
                time,
                solution,
                unknowns: self.circuit.unknowns(),
            };
            if let Some(at) = self
                .history
                .edit_candidate(state, event.rows.clone(), |slots| {
                    event.source.refine_trial(&context, slots)
                })?
            {
                if !at.is_finite() || at > time {
                    return Err(failure("invalid device crossing time"));
                }
                next = Some(next.map_or(at, |prior| prior.min(at)));
            }
        }
        Ok(next)
    }

    fn device_retry(
        &self,
        attempt: &AttemptedStep,
        solution: &Vector,
        state: &mut TrialState,
    ) -> SpiceResult<Option<NextStep>> {
        let time = attempt.clock.time;
        let min_step = self.settings.min_step();
        if let Some(at) = self.refine_device_events(solution, state, time)?
            && time - at > min_step
        {
            let start = attempt
                .exact_start
                .unwrap_or(time - attempt.clock.next.delta);
            let delta = (at - start).max(min_step).max(start.next_up() - start);
            if delta < attempt.clock.next.delta {
                return Ok(Some(NextStep {
                    delta,
                    order: Order::One,
                }));
            }
        }
        Ok(None)
    }

    fn prepare_mixed_point(
        &self,
        solution: &Vector,
        time: Real,
        preview: Option<super::digital::EventPreview>,
    ) -> SpiceResult<Option<super::digital::MixedPoint>> {
        self.mixed
            .as_deref()
            .map(|mixed| mixed.prepare_point(self.circuit, solution, time, preview))
            .transpose()
    }

    fn commit_mixed_point(&mut self, point: Option<super::digital::MixedPoint>) {
        if let (Some(mixed), Some(point)) = (self.mixed.as_deref_mut(), point) {
            mixed.commit_point(point);
        }
    }

    fn preview_events(
        &self,
        ready: &mut ReadyStep,
    ) -> SpiceResult<Option<super::digital::EventPreview>> {
        let preview = self
            .mixed
            .as_deref()
            .map(|mixed| mixed.preview(ready.end_time()))
            .transpose()?
            .flatten();
        if let Some(time) = preview
            .as_ref()
            .and_then(super::digital::EventPreview::boundary)
        {
            ready.limit_to_event(time);
        }
        Ok(preview)
    }

    fn next_physical_event(&mut self, time: Real) -> Option<Real> {
        let events = self.physical_events.as_mut()?;
        while events.peek().is_some_and(|&event| event <= time) {
            events.next();
        }
        events.peek().copied()
    }

    fn crosses_physical_event(&self, attempt: &AttemptedStep) -> bool {
        self.index_aware && attempt.event_start.is_some() && attempt.clock.time < self.settings.stop
    }

    fn check_step_budget(&self, time: Real) -> SpiceResult<()> {
        if self.stats.accepted + self.stats.rejected >= self.settings.max_steps {
            return Err(failure(format!(
                "work limit of {} accepted + rejected steps reached at t = {time:e}",
                self.settings.max_steps
            )));
        }
        Ok(())
    }

    /// Advance through rejected trials. Only Newton's starting iterate moves
    /// between attempts; device and integration histories remain accepted-only.
    fn advance(
        &mut self,
        ready: ReadyStep,
        iterate: Vector,
        output: OutputSelection<'_>,
    ) -> SpiceResult<AcceptedPoint> {
        if self.index_aware {
            return self.advance_physical(ready, iterate, output);
        }
        self.advance_companion(ready, iterate, output)
    }

    fn advance_companion(
        &mut self,
        mut ready: ReadyStep,
        mut iterate: Vector,
        output: OutputSelection<'_>,
    ) -> SpiceResult<AcceptedPoint> {
        let min_step = self.settings.min_step();
        let physical_event = self.next_physical_event(ready.clock.time);
        let device_boundary = self.next_device_boundary(ready.clock.time)?;
        self.stats.device_event_boundary_probes += self.event_devices.len();
        loop {
            self.check_step_budget(ready.clock.time)?;
            if let Some(event) = physical_event
                .into_iter()
                .chain(device_boundary)
                .reduce(Real::min)
            {
                ready.limit_to_event(event);
            }
            let preview = self.preview_events(&mut ready)?;
            let attempt = ready.begin(&self.steps, &self.settings)?;
            let mut prepared = self.prepared.transient_attempt(
                &self.history,
                &self.model_context,
                attempt.clock.time,
                &attempt.coefficients,
                Forcing {
                    limit: attempt.boundary.limit(),
                    timing: self.timing,
                },
            )?;
            // The candidate step borrows accepted history read-only; move
            // the disposable numeric workspace out and back around the
            // solve so retries reuse it without borrowing history mutably.
            let mut workspace = std::mem::take(&mut self.workspace);
            let outcome = self.solve_trial(&attempt, &mut prepared, &iterate, &mut workspace);
            self.workspace = workspace;
            match outcome? {
                Trial::NotConverged { last } => {
                    iterate = last;
                    self.stats.rejected += 1;
                    let next = NextStep {
                        delta: attempt.clock.next.delta / 8.,
                        order: Order::One,
                    };
                    drop(prepared);
                    ready = attempt.retry(next, min_step)?;
                }
                Trial::Converged { x, mut state } => {
                    if let Some(next) = self.mixed_retry(&attempt, &x)? {
                        iterate = x;
                        self.stats.rejected += 1;
                        drop(prepared);
                        ready = attempt.retry(next, min_step)?;
                        ready.boundary = Boundary::Forced;
                        continue;
                    }
                    self.stats.device_event_trial_probes += self.event_devices.len();
                    if let Some(next) = self.device_retry(&attempt, &x, &mut state)? {
                        iterate = x;
                        self.stats.rejected += 1;
                        self.stats.device_event_retries += 1;
                        drop(prepared);
                        ready = attempt.retry(next, min_step)?;
                        ready.boundary = Boundary::Forced;
                        ready.limit_to_event(ready.clock.time + ready.clock.next.delta);
                        continue;
                    }
                    let decision =
                        attempt.decision(&self.settings, &self.steps, |coefficients| {
                            self.truncation_limit(coefficients, &state, &x, attempt.clock.time)
                        })?;
                    match decision {
                        StepDecision::Retry(next) => {
                            iterate = x;
                            self.stats.rejected += 1;
                            drop(prepared);
                            ready = attempt.retry(next, min_step)?;
                        }
                        StepDecision::Accept(next) => {
                            let candidate = self.prepare_accepted_candidate(
                                &attempt,
                                &mut prepared,
                                (x, state),
                                &output,
                                preview,
                            )?;
                            // Release immutable history before publication.
                            drop(prepared);
                            return self.commit_candidate(
                                attempt,
                                next,
                                candidate,
                                device_boundary,
                            );
                        }
                    }
                }
            }
        }
    }

    fn prepare_accepted_candidate(
        &self,
        attempt: &AttemptedStep,
        prepared: &mut crate::devices::circuit::PreparedTransient<'_>,
        (x, state): (Vector, TrialState),
        output: &OutputSelection<'_>,
        preview: Option<super::digital::EventPreview>,
    ) -> SpiceResult<AcceptedCandidate> {
        Ok(AcceptedCandidate {
            mixed: self.prepare_mixed_point(&x, attempt.clock.time, preview)?,
            output: self.candidate_output(attempt, &x, &state, output, || {
                match output.selected {
                    Some(indices) => prepared.observe_selected(&x, &state, Some(indices)),
                    None => prepared.observe(&x, &state),
                }
            })?,
            solution: x,
            state,
        })
    }

    fn commit_candidate(
        &mut self,
        attempt: AttemptedStep,
        next: NextStep,
        candidate: AcceptedCandidate,
        device_boundary: Option<Real>,
    ) -> SpiceResult<AcceptedPoint> {
        let event = self.crosses_physical_event(&attempt);
        let device_breaks = self.commit_point(
            Acceptance::Step(&attempt),
            &candidate.solution,
            candidate.state,
        )?;
        self.commit_mixed_point(candidate.mixed);
        self.stats.device_event_landings +=
            usize::from(device_boundary == Some(attempt.clock.time));
        if event {
            self.history.restart_integration()?;
            self.steps.clear();
        }
        Ok(AcceptedPoint {
            cursor: attempt.accept(next),
            solution: candidate.solution,
            reported_bias: None,
            observed: candidate.output.observed,
            device_breaks,
            periodic_boundary: candidate.output.periodic_boundary,
        })
    }

    fn candidate_output(
        &self,
        attempt: &AttemptedStep,
        x: &Vector,
        state: &TrialState,
        output: &OutputSelection<'_>,
        observe: impl FnOnce() -> SpiceResult<Vec<Complex>>,
    ) -> SpiceResult<CandidateOutput> {
        let time = attempt.clock.time;
        let observed = if (time < self.settings.start && !self.settings.interpolates())
            || output.sample_at.is_some_and(|at| {
                if self.settings.schedule.is_some() {
                    time.to_bits() != at.to_bits()
                } else {
                    !almost_equal_ulps(time, at, 100)
                }
            }) {
            None
        } else {
            Some(observe()?)
        };
        let periodic_boundary = (output.capture_boundary
            && almost_equal_ulps(time, self.settings.stop, 100))
        .then(|| {
            Box::new(PeriodicBoundary::capture(
                &self.history,
                attempt,
                state,
                x,
                self.timing,
            ))
        });
        Ok(CandidateOutput {
            observed,
            periodic_boundary,
        })
    }

    /// The sole acceptance boundary: commit device/delay history, then advance
    /// integration history and statistics for an ordinary transient step.
    fn commit_point(
        &mut self,
        acceptance: Acceptance<'_>,
        solution: &Vector,
        trial: TrialState,
    ) -> SpiceResult<Vec<Real>> {
        let (time, dt, start) = match acceptance {
            Acceptance::Initial => (0., self.settings.max_step, true),
            Acceptance::Step(attempt) => (attempt.clock.time, attempt.coefficients.dt(), false),
        };
        let delay = self.delay_acceptance(dt, start);
        let breaks = self.circuit.accept_transient_point(
            solution,
            time,
            &mut self.history,
            trial,
            &delay,
        )?;
        if let Some(history) = &mut self.voltage_history {
            history.accept(solution, acceptance);
        }
        match acceptance {
            Acceptance::Initial => {}
            Acceptance::Step(attempt) => {
                self.steps.accept(&attempt.coefficients);
                self.record_step(dt, attempt.boundary.at_break());
            }
        }
        Ok(breaks)
    }

    fn record_step(&mut self, step: Real, on_breakpoint: bool) {
        let stats = &mut self.stats;
        stats.min_step = if stats.accepted == 0 {
            step
        } else {
            stats.min_step.min(step)
        };
        stats.max_step = stats.max_step.max(step);
        stats.accepted += 1;
        stats.breakpoints += usize::from(on_breakpoint);
    }

    /// The initial point of the run: the solution `x` at `t = 0` and the
    /// accepted charge/flux state filling the whole history.
    ///
    /// * Ordinary run (C `CKTop` with `MODETRANOP`): `A x = b(0-)` with the
    ///   sources at their left limit; `.ic` node voltages are enforced as
    ///   `cktload.c`'s forced rows during this solve only
    ///   ([`crate::analysis::bias::NodeForcing`]),
    ///   instance `ic=` of C/L/D/Q is ignored (MOS1 `ic=` only moves its
    ///   `MODEINITJCT` start) and `.nodeset` cannot change a linear point.
    /// * `uic` (C `NIiter` returns after one `CKTload`): no solve; charges and
    ///   fluxes come from the capacitor/inductor initial values (see
    ///   [`initial::uic_start`]), nonlinear devices load once at their
    ///   instance initial conditions, and [`initial::check_impulse_free`]
    ///   rejects capacitor/inductor initial conditions that would need an
    ///   impulse (junction charges are not part of that check, as in C).
    ///
    /// Nothing is committed (no accept hook, no history) until every check has
    /// passed, so a failed initialization leaves no partial state.
    fn initialize(&mut self) -> SpiceResult<AcceptedPoint> {
        // Solve the analog and event operating point to a common fixed point.
        // The provisional digital state is rolled back if initialization
        // fails; analog integration history is committed exactly once.
        for _ in 0..self.bias.newton.max_iterations {
            let InitialCandidate {
                solution: x,
                state: trial,
                reported_bias,
            } = self.initial_candidate()?;
            let mixed_point = self
                .mixed
                .as_deref()
                .map(|mixed| mixed.prepare_point(self.circuit, &x, 0., None))
                .transpose()?;
            if let (Some(mixed), Some(point)) = (self.mixed.as_deref_mut(), mixed_point) {
                let changed = mixed.initial_changed(&point)?;
                mixed.commit_point(point);
                if changed {
                    continue;
                }
            }
            let device_breaks = self.commit_point(Acceptance::Initial, &x, trial)?;
            if let Some(mixed) = self.mixed.as_deref_mut() {
                mixed.accept_initial();
            }
            return Ok(AcceptedPoint {
                cursor: AcceptedCursor::initial(&self.settings),
                solution: x,
                reported_bias,
                observed: None,
                device_breaks,
                periodic_boundary: None,
            });
        }
        Err(failure("mixed-signal operating point did not settle"))
    }

    fn initial_uic_candidate(&mut self) -> SpiceResult<(Vector, TrialState)> {
        let start = initial::uic_start(self.circuit, &self.hints, &self.model_context)?;
        // C's single MODETRANOP|MODEUIC|MODEINITJCT load: nonlinear
        // devices evaluate at their instance initial conditions.
        let mut trial = self.initial_state(&start.x, &start.charges, true)?;
        if let Some(consistent) = self.derivative_initial(&start.x, &trial, true)? {
            return Ok(consistent);
        }
        let mut right_rhs = self
            .system
            .transient_rhs(0., crate::devices::Limit::Right)?;
        if let Some(mixed) = self.mixed.as_deref() {
            mixed.initial_rhs(self.circuit, &mut right_rhs)?;
        }
        let consistent = initial::check_impulse_free(
            &self.prepared,
            &self.system.a,
            &right_rhs,
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &start.x,
                model_context: &self.model_context,
                integration: None,
                history: &self.history,
                forcing: Some(Forcing {
                    limit: crate::devices::Limit::Right,
                    timing: self.timing,
                }),
            },
            &trial,
            self.physical_tolerances(),
            &self.bias.newton,
        )?;
        // Keep the device-flagged UIC history, but start nonlinear
        // Newton iteration at the solved algebraic point rather than
        // guessing every unspecified node/branch at zero.
        for (slot, value) in consistent.canonical {
            self.history.set_candidate_slot(&mut trial, slot, value)?;
        }
        Ok((consistent.solution.unwrap_or(start.x), trial))
    }

    fn initial_candidate(&mut self) -> SpiceResult<InitialCandidate> {
        if self.uic {
            let (solution, state) = self.initial_uic_candidate()?;
            return Ok(InitialCandidate {
                solution,
                state,
                reported_bias: None,
            });
        }
        let mut rhs = self.system.transient_startup_rhs()?;
        if let Some(mixed) = self.mixed.as_deref() {
            mixed.initial_rhs(self.circuit, &mut rhs)?;
        }
        let tolerances = &self.settings.tolerances;
        let constraints = initial::irredundant_constraints(
            self.circuit,
            &rhs,
            &self.hints.initial,
            VoltageTolerance {
                reltol: tolerances.reltol,
                vntol: tolerances.vntol,
            },
        )?;
        // dctran.c calls CKTop (NIiter) for every circuit, linear or not.
        let solution = {
            // CKTic: the Newton guess holds the .nodeset, then the .ic
            // values; cktload.c forces the .ic rows in every load of the
            // MODETRANOP solve and the .nodeset rows in its MODEINITJCT/
            // MODEINITFIX loads.
            let mut seed = Vector::zeros(self.circuit.unknown_count());
            for hint in self.hints.nodesets.iter().chain(&self.hints.initial) {
                seed.as_mut_slice()[hint.row] = hint.value;
            }
            // Every entry is forced, as cktload.c does;
            // `irredundant_constraints` only rejects contradictions.
            let nodes = crate::analysis::bias::NodeForcing {
                initial: self
                    .hints
                    .initial
                    .iter()
                    .map(|hint| (hint.row, hint.value))
                    .collect(),
                nodesets: initial::forced_nodesets(&self.hints.nodesets),
            };
            crate::analysis::bias::solve_dc_forced_prepared(
                &self.prepared,
                &self.model_context,
                &self.bias,
                &[],
                Some(&seed),
                // MODETRANOP: native sources at 0-, prepared sources at 0+.
                Some(Forcing {
                    limit: crate::devices::Limit::Left,
                    timing: self.timing,
                }),
                &nodes,
            )?
            .solution
        };
        let x = solution.values;
        initial::check_implied(
            &x,
            &constraints,
            VoltageTolerance {
                reltol: tolerances.reltol,
                vntol: tolerances.vntol,
            },
        )?;
        // A nonlinear bias's converged trial is the DC-mode load at `x`
        // (charges, zero derivatives) and also carries the operating
        // point's discrete switch states, which a fresh reload in the
        // initial phase would re-derive from the instance flags.
        let trial = solution.trial;
        if let Some((solution, state)) = self.derivative_initial(&x, &trial, false)? {
            let reported_bias = self
                .circuit
                .devices()
                .iter()
                .any(|device| device.exact_startup_storage().is_some())
                .then_some(x);
            return Ok(InitialCandidate {
                solution,
                state,
                reported_bias,
            });
        }
        Ok(InitialCandidate {
            solution: x,
            state: trial,
            reported_bias: None,
        })
    }

    fn derivative_initial(
        &mut self,
        x: &Vector,
        trial: &TrialState,
        uic: bool,
    ) -> SpiceResult<Option<(Vector, TrialState)>> {
        if self.mixed.is_some() || !initial::has_derived_storage(self.circuit) {
            return Ok(None);
        }
        let request = LoadRequest {
            mode: AnalysisMode::Instantaneous { time: 0. },
            solution: x,
            model_context: &self.model_context,
            integration: None,
            history: &self.history,
            forcing: Some(Forcing {
                limit: crate::devices::Limit::Right,
                timing: self.timing,
            }),
        };
        let tolerance = &self.settings.tolerances;
        let Some(consistent) = initial::consistent_derivative_start(
            &self.prepared,
            &request,
            trial,
            uic,
            initial::Tolerances {
                reltol: tolerance.reltol,
                vntol: tolerance.vntol,
                abstol: tolerance.abstol,
                chgtol: tolerance.chgtol,
            },
            &self.bias.newton,
        )?
        else {
            return Ok(None);
        };
        let TrialPoint {
            x,
            state: trial,
            guards,
            seed,
        } = self.physical_candidate(&request, consistent)?;
        self.accepted_guards = guards;
        self.derivative_predictor = Some(DerivativePredictor {
            time: 0.,
            state: x.clone(),
            seed,
        });
        self.physical_events = Some(
            self.system
                .breakpoints_in(0., self.settings.stop)?
                .peekable(),
        );
        self.index_aware = true;
        Ok(Some((x, trial)))
    }

    /// The charge/flux state of the initial point: a DC-mode load at `x`
    /// (`q = C v`, `flux = L i`, zero derivative) with the listed absolute
    /// state slots overwritten. `uic` marks C's `uic` initial load, in which
    /// nonlinear devices evaluate (and store their junction voltages and
    /// charges) at their instance initial conditions
    /// ([`crate::devices::TrialState::with_initial_conditions`]).
    fn initial_state(
        &self,
        x: &Vector,
        overrides: &[(usize, Real)],
        uic: bool,
    ) -> SpiceResult<TrialState> {
        let n = self.circuit.unknown_count();
        let mut trial = self.history.trial().with_initial_conditions(uic);
        self.prepared.load(
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: x,
                model_context: &self.model_context,
                integration: None,
                history: &self.history,
                forcing: None,
            },
            &mut self.circuit.new_matrix()?,
            &mut Vector::zeros(n),
            &mut trial,
        )?;
        for (slot, value) in overrides {
            self.history.set_candidate_slot(&mut trial, *slot, *value)?;
        }
        Ok(trial)
    }

    /// One trial point: C `NIiter` from `previous` (`CKTrhsOld`: the last
    /// iterate of the previous trial, accepted or not; ngspice is built
    /// without `PREDICTOR`), `MODEINITTRAN` before the first accepted step
    /// (`firsttime`) and `MODEINITPRED` after, for linear and nonlinear
    /// circuits alike. The trial is that of the last Newton load, one iterate
    /// behind the returned solution, as C's `CKTstate0`.
    fn physical_candidate(
        &self,
        request: &LoadRequest<'_>,
        consistent: initial::DerivativePoint,
    ) -> SpiceResult<TrialPoint> {
        let x = consistent.solution;
        let trial = self.prepared.physical_candidate_state(
            &LoadRequest {
                solution: &x,
                ..*request
            },
            &consistent.canonical,
        )?;
        Ok(TrialPoint {
            x,
            state: trial,
            guards: consistent.guards,
            seed: DerivativeSeed {
                rate: consistent.rate,
                order_hint: consistent.order_hint,
            },
        })
    }

    const fn physical_tolerances(&self) -> initial::Tolerances {
        let t = &self.settings.tolerances;
        initial::Tolerances {
            reltol: t.reltol,
            vntol: t.vntol,
            abstol: t.abstol,
            chgtol: t.chgtol,
        }
    }

    fn derivative_event(
        &self,
        attempt: &AttemptedStep,
        left: &TrialPoint,
    ) -> SpiceResult<TrialPoint> {
        let request = LoadRequest {
            mode: AnalysisMode::Instantaneous {
                time: attempt.clock.time,
            },
            solution: &left.x,
            model_context: &self.model_context,
            integration: None,
            history: &self.history,
            forcing: Some(Forcing {
                limit: crate::devices::Limit::Right,
                timing: self.timing,
            }),
        };
        let consistent = initial::consistent_derivative_event(
            &self.prepared,
            &request,
            &left.state,
            self.physical_tolerances(),
            self.settings.tran_max_iter,
            left.seed.guess(),
        )?;
        self.physical_candidate(&request, consistent)
    }

    fn derivative_trial(
        &self,
        attempt: &AttemptedStep,
        previous: &Vector,
    ) -> SpiceResult<PhysicalTrial> {
        let predictor = self.derivative_predictor.as_ref();
        let prediction = predictor.and_then(|point| point.at(attempt.clock.time));
        let guess = predictor.map_or(
            initial::DerivativeGuess {
                order_hint: 2,
                rate: None,
            },
            |point| point.seed.guess(),
        );
        let request = LoadRequest {
            mode: AnalysisMode::Transient {
                time: attempt.clock.time,
                dt: attempt.coefficients.dt(),
            },
            solution: prediction.as_ref().unwrap_or(previous),
            model_context: &self.model_context,
            integration: Some(&attempt.coefficients),
            history: &self.history,
            forcing: Some(Forcing {
                limit: if attempt.event_start.is_some() {
                    crate::devices::Limit::Left
                } else {
                    crate::devices::Limit::Right
                },
                timing: self.timing,
            }),
        };
        let solve = |request: &LoadRequest<'_>, rate| {
            initial::consistent_derivative_step(
                &self.prepared,
                request,
                self.physical_tolerances(),
                self.settings.tran_max_iter,
                initial::DerivativeGuess {
                    order_hint: guess.order_hint,
                    rate,
                },
            )
        };
        let rate = guess.rate;
        let outcome = match solve(&request, rate) {
            // An extrapolated state can leave the domain of an otherwise
            // smooth constitutive law. Fall back to the ordinary Newton
            // seed; do not retry away a detected physical ambiguity.
            Err(SpiceError::Circuit { .. }) | Ok(initial::DerivativeTrial::NotConverged(_))
                if prediction.is_some() || rate.is_some() =>
            {
                solve(
                    &LoadRequest {
                        solution: previous,
                        ..request
                    },
                    None,
                )
            }
            result => result,
        }?;
        let consistent = match outcome {
            initial::DerivativeTrial::Converged(point) => point,
            initial::DerivativeTrial::NotConverged(last) => {
                return Ok(PhysicalTrial::NotConverged { last });
            }
        };
        let point = self.physical_candidate(
            &LoadRequest {
                mode: AnalysisMode::Instantaneous {
                    time: attempt.clock.time,
                },
                integration: None,
                ..request
            },
            consistent,
        )?;
        Ok(PhysicalTrial::Converged(point))
    }

    fn solve_trial(
        &self,
        attempt: &AttemptedStep,
        prepared: &mut crate::devices::circuit::PreparedTransient<'_>,
        previous: &Vector,
        workspace: &mut crate::analysis::newton::CircuitNewtonWorkspace,
    ) -> SpiceResult<Trial> {
        let tolerance = &self.settings.tolerances;
        let options = crate::analysis::newton::NewtonOptions {
            max_iterations: self.settings.tran_max_iter,
            reltol: tolerance.reltol,
            vntol: tolerance.vntol,
            abstol: tolerance.abstol,
            limiting: self.bias.newton.limiting,
            diag_gmin: self.circuit.continuation_diagonal(),
            ..crate::analysis::newton::NewtonOptions::default()
        };
        // dctran.c: MODEINITTRAN (first step) or MODEINITPRED for the first
        // load of a timepoint, MODEINITFLOAT afterwards.
        let policy = if attempt.clock.phase == Phase::First {
            crate::analysis::newton::PhasePolicy::InitialTransient
        } else {
            crate::analysis::newton::PhasePolicy::Predicted
        };
        let solved = CircuitNewton {
            branch_rows: &self.branch_row,
            limited_rows: self.limited_rows.as_deref(),
            options: &options,
            load: CircuitLoad::Transient {
                attempt: prepared,
                uic: self.uic,
            },
            workspace,
        }
        .solve(previous, policy, None, |_, _, _| Ok(()));
        match solved {
            Ok(solved) => Ok(Trial::Converged {
                x: solved.values,
                state: solved.trial,
            }),
            Err(crate::analysis::newton::NewtonFailure {
                last_iterate: Some(last),
                ..
            }) => Ok(Trial::NotConverged { last }),
            Err(failure) => Err(failure.error),
        }
    }

    /// The delay-line quantities of an accepted point or a converged trial
    /// whose step is `dt`: C `CKTdeltaOld[0..3]` (`dt`, then the two latest
    /// accepted steps; steps not taken yet read as the maximum step, as
    /// `dctran.c` initializes `CKTdeltaOld`) and the `CKTminBreak` that
    /// `TRAaccept` compares sample spacing with: the XSPICE build's
    /// configured merge distance (default `10 CKTdelmin`), shared with the [`BreakTable`].
    fn delay_acceptance(&self, dt: Real, start: bool) -> crate::devices::DelayAcceptance {
        let accepted = self.steps.accepted();
        let older = |age: usize| accepted.get(age).copied().unwrap_or(self.settings.max_step);
        crate::devices::DelayAcceptance {
            steps: [dt, older(0), older(1)],
            min_break: self.settings.min_break(),
            start,
            reltol: self.settings.tolerances.reltol,
            abstol: self.settings.tolerances.abstol,
        }
    }

    /// C `CKTtrunc`: the smallest step bound over every charge-storage element,
    /// every discrete-state device ([`crate::devices::Device::timestep_limit`],
    /// `swtrunc.c`) and every delay line (`tratrunc.c`), from the converged
    /// trial `solution` at `time` and the accepted history.
    fn truncation_limit(
        &self,
        coefficients: &Coefficients,
        trial: &TrialState,
        solution: &Vector,
        time: Real,
    ) -> SpiceResult<Real> {
        let tolerances = &self.settings.tolerances;
        let tolerances = TruncationTolerances {
            reltol: tolerances.reltol,
            abstol: tolerances.abstol,
            chgtol: tolerances.chgtol,
            trtol: tolerances.trtol,
        };
        let order = usize::from(coefficients.order());
        let mut limit = Real::INFINITY;
        for (device, TruncationDevice { rows, slots: bases }) in
            self.circuit.devices().iter().zip(&self.truncation)
        {
            for &TruncationSlot {
                base,
                explicit_derivative,
                ..
            } in bases
            {
                let accepted = |age: usize, offset: usize| {
                    self.history
                        .accepted(age)
                        .and_then(|vector| vector.get(base + offset))
                        .copied()
                        .ok_or_else(|| failure("accepted state history is too short"))
                };
                let mut storage = [0.0; IntegrationMethod::MAX_GEAR_ORDER as usize + 2];
                let bound = if self.index_aware || explicit_derivative {
                    let derivative = &mut storage[..=order];
                    derivative[0] = trial.values()[base + 1];
                    for (age, value) in derivative.iter_mut().enumerate().skip(1) {
                        *value = accepted(age, 1)?;
                    }
                    coefficients.truncation_timestep_from_derivatives(
                        [trial.values()[base], accepted(1, 0)?],
                        derivative,
                        &tolerances,
                    )?
                } else {
                    // The trial charge, then order + 1 accepted ones.
                    let charge = &mut storage[..order + 2];
                    charge[0] = trial.values()[base];
                    for (age, value) in charge.iter_mut().enumerate().skip(1) {
                        *value = accepted(age, 0)?;
                    }
                    let derivative = [trial.values()[base + 1], accepted(1, 1)?];
                    coefficients.truncation_timestep(charge, derivative, &tolerances)?
                };
                limit = limit.min(bound);
            }
            // Discrete-state bounds (`swtrunc.c`), from the same trial.
            if !rows.is_empty() {
                let bound = device.timestep_limit(&crate::devices::TruncationContext {
                    trial: trial
                        .slice(rows.clone())
                        .ok_or_else(|| failure("trial state is too short"))?,
                    accepted: self
                        .history
                        .accepted(1)
                        .and_then(|vector| vector.get(rows.clone())),
                    dt: coefficients.dt(),
                })?;
                if let Some(bound) = bound {
                    if !bound.is_finite() {
                        return Err(failure(format!(
                            "{}: nonfinite timestep limit",
                            device.name()
                        )));
                    }
                    limit = limit.min(bound);
                }
            }
        }
        if let Some(history) = &self.voltage_history {
            limit = limit.min(history.limit(coefficients, solution)?);
        }
        if let Some(bound) = self.circuit.delay_timestep_limit(
            &self.history,
            solution,
            time,
            &self.delay_acceptance(coefficients.dt(), false),
        )? {
            limit = limit.min(bound);
        }
        Ok(limit)
    }
}

/// Deliver an accepted row to the consumer without copying it.
fn push(
    output: &super::linear::LinearOutput,
    sink: &mut dyn PointSink,
    selected: Option<&mut crate::analysis::results::SelectedOutput>,
    point: &mut Vec<Complex>,
    time: Real,
    x: &Vector,
    observed: &[Complex],
) -> SpiceResult<()> {
    if let Some(selected) = selected {
        return selected.real(sink, Some(time), x.as_slice(), observed);
    }
    point.clear();
    point.push(Complex::real(time));
    point.extend(x.as_slice().iter().map(|v| Complex::real(*v)));
    output.append_reports(point);
    point.extend_from_slice(observed);
    output.header.check_point(point)?;
    sink.point(point)
}

#[cfg(test)]
mod step_control_tests {
    use super::*;

    fn settings() -> TransientPlan {
        TransientPlan {
            step: 1.,
            stop: 10.,
            start: 0.,
            max_step: 1.,
            schedule: None,
            minbreak: 0.,
            voltage_lte: None,
            max_order: 2,
            method: IntegrationMethod::Trapezoidal,
            xmu: 0.5,
            max_steps: 100,
            tran_max_iter: 10,
            tolerances: Tolerances {
                reltol: 1e-3,
                vntol: 1e-6,
                abstol: 1e-12,
                chgtol: 1e-14,
                trtol: 7.,
            },
        }
    }

    fn ready() -> ReadyStep {
        ReadyStep {
            clock: StepClock {
                time: 0.25,
                next: NextStep {
                    delta: 0.5,
                    order: Order::One,
                },
                save_delta: 0.5,
                phase: Phase::Continuing,
            },
            boundary: Boundary::Forced,
            event_landing: None,
            probe_landing: None,
        }
    }

    fn proposal(delta: Real) -> NextStep {
        NextStep {
            delta,
            order: Order::One,
        }
    }

    #[test]
    fn rejected_breakpoint_preserves_c_flag_but_uses_right_limit() {
        let steps = StepHistory::with_fill(1.);
        let settings = settings();
        let attempt = ready().begin(&steps, &settings).unwrap();
        assert_eq!(attempt.boundary.limit(), crate::devices::Limit::Left);
        let retry = attempt.retry(proposal(0.125), 0.01).unwrap();
        assert_eq!(retry.clock.time, 0.25);
        assert!(retry.boundary.at_break());
        let attempt = retry.begin(&steps, &settings).unwrap();
        assert_eq!(attempt.boundary.limit(), crate::devices::Limit::Right);
        assert_eq!(attempt.clock.time, 0.375);
    }

    #[test]
    fn minimum_step_gets_exactly_one_retry() {
        let steps = StepHistory::with_fill(1.);
        let settings = settings();
        let retry = ready()
            .begin(&steps, &settings)
            .unwrap()
            .retry(proposal(0.001), 0.01)
            .unwrap();
        assert_eq!(retry.clock.next.delta, 0.01);
        assert!(
            retry
                .begin(&steps, &settings)
                .unwrap()
                .retry(proposal(0.001), 0.01)
                .is_err()
        );
    }

    #[test]
    fn rejection_uses_c_add_then_subtract_rounding() {
        let mut ready = ready();
        ready.clock.time = 0.1;
        ready.clock.next.delta = 0.2;
        let expected = (ready.clock.time + ready.clock.next.delta) - ready.clock.next.delta;
        assert_ne!(expected.to_bits(), ready.clock.time.to_bits());
        let retry = ready
            .begin(&StepHistory::with_fill(1.), &settings())
            .unwrap()
            .retry(proposal(0.02), 0.001)
            .unwrap();
        assert_eq!(retry.clock.time.to_bits(), expected.to_bits());
    }

    #[test]
    fn sample_landings_preserve_history_order_and_retry_origin() {
        let mut ready = ready();
        ready.clock.time = 0.1;
        ready.clock.next.delta = 0.4;
        ready.clock.next.order = Order::Two;
        ready.limit_to_sample(0.3);
        assert_eq!(ready.end_time().to_bits(), 0.3_f64.to_bits());
        assert_eq!(ready.boundary, Boundary::Ordinary);
        assert_eq!(ready.clock.next.order, Order::Two);
        let mut steps = StepHistory::with_fill(1.);
        let first = settings().method;
        steps.accept(&steps.trial(first, 1, 0.1, 0.5).unwrap());
        let attempt = ready.begin(&steps, &settings()).unwrap();
        assert_eq!(
            attempt
                .retry(proposal(0.02), 0.001)
                .unwrap()
                .clock
                .time
                .to_bits(),
            0.1_f64.to_bits()
        );
        ready.limit_to_event(0.3);
        assert_eq!(ready.boundary, Boundary::Forced);
        ready.shorten(0.01);
        assert!(ready.end_time() < 0.3);
        assert_eq!(ready.boundary, Boundary::Ordinary);
    }

    #[test]
    fn order_probe_is_a_decision_not_a_mutation() {
        let settings = settings();
        let mut steps = StepHistory::with_fill(1.);
        let initial = ready().begin(&steps, &settings).unwrap();
        steps.accept(&initial.coefficients);
        let attempt = ready().begin(&steps, &settings).unwrap();
        let delta = attempt.coefficients.dt();
        let before = steps.accepted().to_vec();
        assert_eq!(
            attempt
                .decision(&settings, &steps, |_| Ok(0.9 * delta))
                .unwrap(),
            StepDecision::Retry(proposal(0.9 * delta))
        );
        for (bound, order) in [(1.05 * delta, Order::One), (1.06 * delta, Order::Two)] {
            let mut orders = Vec::new();
            let decision = attempt
                .decision(&settings, &steps, |coefficients| {
                    orders.push(coefficients.order());
                    Ok(if coefficients.order() == 1 {
                        delta
                    } else {
                        bound
                    })
                })
                .unwrap();
            assert_eq!(orders, [1, 2]);
            assert_eq!(
                decision,
                StepDecision::Accept(NextStep {
                    delta: bound,
                    order
                })
            );
        }
        assert_eq!(attempt.coefficients.order(), 1);
        assert_eq!(attempt.clock.next, proposal(delta));
        assert_eq!(steps.accepted(), before);
    }

    #[test]
    fn first_step_retry_retains_initial_phase_until_acceptance() {
        let settings = settings();
        let steps = StepHistory::with_fill(1.);
        let mut ready = ready();
        ready.clock.phase = Phase::First;
        let attempt = ready.begin(&steps, &settings).unwrap();
        let retry = attempt.retry(proposal(0.125), 0.01).unwrap();
        assert_eq!(retry.clock.phase, Phase::First);
        let attempt = retry.begin(&steps, &settings).unwrap();
        assert_eq!(
            attempt
                .decision(&settings, &steps, |_| panic!(
                    "first step must not probe truncation"
                ))
                .unwrap(),
            StepDecision::Accept(proposal(0.125))
        );
        let accepted = attempt.accept(proposal(0.125));
        assert_eq!(accepted.clock.phase, Phase::Continuing);
        assert!(steps.accepted().is_empty());
    }
}
