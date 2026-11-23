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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientPlan {
    step: Real,
    stop: Real,
    start: Real,
    max_step: Real,
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
        let [output_step, end_time, start_time, max_step] = Self::time_bounds(positional)?;
        Ok(Self {
            step: output_step,
            stop: end_time,
            start: start_time,
            max_step,
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
        })
    }

    /// Requested output time step.
    #[must_use]
    pub const fn step(&self) -> Real {
        self.step
    }

    /// Final simulation time.
    #[must_use]
    pub const fn stop(&self) -> Real {
        self.stop
    }

    /// The requested integration family and effective Gear order.
    #[must_use]
    pub const fn method(&self) -> IntegrationMethod {
        self.method
    }

    /// The requested maximum integration order.
    #[must_use]
    pub const fn max_order(&self) -> u8 {
        self.max_order
    }

    /// Newton iteration limit at each accepted or rejected timepoint.
    #[must_use]
    pub const fn max_iterations(&self) -> usize {
        self.tran_max_iter
    }

    /// Trapezoidal weighting in [0, 0.5].
    #[must_use]
    pub const fn xmu(&self) -> Real {
        self.xmu
    }

    /// Charge (flux) absolute truncation tolerance.
    #[must_use]
    pub const fn chgtol(&self) -> Real {
        self.tolerances.chgtol
    }

    /// Truncation error overestimation factor.
    #[must_use]
    pub const fn trtol(&self) -> Real {
        self.tolerances.trtol
    }

    /// C `CKTdelmin`.
    fn min_step(&self) -> Real {
        1e-11 * self.max_step
    }
}

fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "transient".to_owned(),
        message: message.into(),
    }
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

struct ReadyStep {
    clock: StepClock,
    boundary: Boundary,
}

struct AttemptedStep {
    clock: StepClock,
    boundary: Boundary,
    coefficients: Coefficients,
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
        clock.next.delta = clock.next.delta.min(settings.max_step);
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
        Ok(ReadyStep { clock, boundary })
    }
}

impl ReadyStep {
    fn begin(self, steps: &StepHistory, settings: &TransientPlan) -> SpiceResult<AttemptedStep> {
        let mut clock = self.clock;
        clock.time += clock.next.delta;
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
        })
    }
}

impl AttemptedStep {
    fn retry(self, mut next: NextStep, min_step: Real) -> SpiceResult<ReadyStep> {
        let mut clock = self.clock;
        // C adds then subtracts; restoring the previous accepted time would
        // change rounding. Only this retry cursor, never the history, moves.
        clock.time -= clock.next.delta;
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
        })
    }

    const fn accept(self, next: NextStep) -> AcceptedCursor {
        let mut clock = self.clock;
        clock.phase = Phase::Continuing;
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
        if self.clock.phase == Phase::First {
            return Ok(StepDecision::Accept(next));
        }
        let delta = next.delta;
        next.delta = (2. * delta).min(truncate(&self.coefficients)?);
        if next.delta > 0.9 * delta {
            // dctran.c never raises above order 2. Keep the probed timestep
            // even when the order-2 probe does not justify raising the order.
            if next.order == Order::One && settings.max_order > 1 {
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

struct AcceptedPoint {
    cursor: AcceptedCursor,
    solution: Vector,
    observed: Vec<Complex>,
    device_breaks: Vec<Real>,
}

#[derive(Clone, Copy)]
enum Acceptance<'a> {
    Initial,
    Step(&'a AttemptedStep),
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
    truncation: Vec<(std::ops::Range<usize>, Vec<usize>)>,
    stats: TransientStats,
    /// Reuse the ordinary Newton MNA/RHS/trial storage across timepoint
    /// acceptance and rejected smaller-step retries.
    workspace: crate::analysis::newton::CircuitNewtonWorkspace,
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
    let settings = *request
        .transient()
        .ok_or_else(|| unsupported("companion requires a transient plan"))?;
    let bias = request.dc_settings().clone();
    if request.uic()
        && let Some(line) = circuit.devices().iter().find(|d| d.delay_line().is_some())
    {
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
    // Unknown/ground nodes in .ic/.nodeset fail before anything is assembled.
    let hints = initial::resolve(circuit, request)?;
    let model_context = context.model_context();
    circuit.finalize()?;
    let nonlinear = circuit.devices().iter().any(|d| d.is_nonlinear());
    let mut system = if nonlinear {
        circuit.small_signal_system(&model_context, &Vector::zeros(circuit.unknown_count()))?
    } else {
        circuit.linear_system_with_context(&model_context)?
    };
    let timing = TransientTiming::new(settings.step, settings.stop)?;
    system.bind_transient_timing(&timing)?;
    let circuit: &Circuit = circuit;
    let branch_row = crate::analysis::bias::branch_rows(circuit);
    let limited_rows = crate::analysis::bias::limited_rows(circuit);
    let truncation = circuit
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
                .map(|slot| rows.start + slot)
                .collect();
            Ok((rows, bases))
        })
        .collect::<SpiceResult<_>>()?;
    let driver = Driver {
        circuit,
        prepared: circuit.prepare_run(),
        system: &system,
        settings,
        timing,
        model_context,
        bias,
        history: circuit.state_history()?,
        steps: StepHistory::with_fill(settings.max_step),
        hints,
        uic: request.uic(),
        branch_row,
        limited_rows,
        truncation,
        stats: TransientStats::default(),
        workspace: crate::analysis::newton::CircuitNewtonWorkspace::default(),
    };
    driver.run(sink)
}

impl Driver<'_> {
    fn observe_initial(
        &self,
        solution: &Vector,
        selected: Option<&[usize]>,
    ) -> SpiceResult<Vec<Complex>> {
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

    fn run(mut self, sink: &mut dyn PointSink) -> SpiceResult<TransientStats> {
        let TransientPlan { stop, start, .. } = self.settings;
        // traninit.c: CKTdelmin; dctran.c (XSPICE build): CKTminBreak.
        let min_step = self.settings.min_step();
        let mut breaks = BreakTable::new(stop, 10.0 * min_step)?;
        let mut sources = SourceBreaks::new(self.system);
        let plot = plot(
            self.circuit,
            "tran1",
            "Transient Analysis",
            Some(("time", "time")),
            false,
        )?;
        sink.begin(&plot)?;
        let mut selected = crate::analysis::results::SelectedOutput::bind(
            sink,
            &plot,
            self.circuit.unknown_count(),
            true,
        )?;
        let mut point = Vec::with_capacity(plot.variable_count());
        let mut accepted = self.initialize()?;
        loop {
            // nextTime: CKTaccept (device hooks ran when the point was
            // committed), the sources' breakpoint requests, then output.
            // CKTaccept runs DEVaccept in dev.c type order: isrc, then tra
            // (the delay lines' requests), then vsrc.
            sources.accept(
                SourceKind::Current,
                accepted.cursor.clock.time,
                accepted.cursor.boundary.at_break(),
                &self.timing,
                &mut breaks,
            )?;
            for time in std::mem::take(&mut accepted.device_breaks) {
                breaks.set(accepted.cursor.clock.time, time)?;
            }
            sources.accept(
                SourceKind::Voltage,
                accepted.cursor.clock.time,
                accepted.cursor.boundary.at_break(),
                &self.timing,
                &mut breaks,
            )?;
            if accepted.cursor.clock.time > breaks.first() {
                breaks.clear();
            }
            if accepted.cursor.clock.time == 0. {
                // C (dctran.c) writes no t = 0 row under uic.
                if start <= 0. && !self.uic {
                    let observed = self.observe_initial(
                        &accepted.solution,
                        selected
                            .as_ref()
                            .map(crate::analysis::results::SelectedOutput::observations),
                    )?;
                    push(
                        &plot,
                        sink,
                        selected.as_mut(),
                        &mut point,
                        0.,
                        &accepted.solution,
                        &observed,
                    )?;
                }
            } else if accepted.cursor.clock.time >= start {
                push(
                    &plot,
                    sink,
                    selected.as_mut(),
                    &mut point,
                    accepted.cursor.clock.time,
                    &accepted.solution,
                    &accepted.observed,
                )?;
            }
            if almost_equal_ulps(accepted.cursor.clock.time, stop, 100) {
                break;
            }
            let AcceptedPoint {
                cursor, solution, ..
            } = accepted;
            let ready = cursor.prepare(&self.settings, &mut breaks, self.uic)?;
            accepted = self.advance(
                ready,
                solution,
                selected
                    .as_ref()
                    .map(crate::analysis::results::SelectedOutput::observations),
            )?;
        }
        sink.end()?;
        // C keeps the final load's instance fields, not values re-evaluated
        // at the final solution (the load can lag that solution by an iterate).
        // Publish only after the entire run and every sink write succeeded.
        self.circuit.record_final_transient_state(&self.history)?;
        self.stats.ltra = self.prepared.ltra_work();
        Ok(self.stats)
    }

    /// Advance through rejected trials. Only Newton's starting iterate moves
    /// between attempts; device and integration histories remain accepted-only.
    fn advance(
        &mut self,
        mut ready: ReadyStep,
        mut iterate: Vector,
        selected: Option<&[usize]>,
    ) -> SpiceResult<AcceptedPoint> {
        let min_step = self.settings.min_step();
        loop {
            if self.stats.accepted + self.stats.rejected >= self.settings.max_steps {
                return Err(failure(format!(
                    "work limit of {} accepted + rejected steps reached at t = {:e}",
                    self.settings.max_steps, ready.clock.time
                )));
            }
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
                Trial::Converged { x, state } => {
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
                            let observed = if attempt.clock.time < self.settings.start {
                                Vec::new()
                            } else if let Some(selected) = selected {
                                prepared.observe_selected(&x, &state, Some(selected))?
                            } else {
                                prepared.observe(&x, &state)?
                            };
                            // Release the proposal's immutable history and exclusive scratch
                            // before accepting, which changes the accepted history revision.
                            drop(prepared);
                            let device_breaks =
                                self.commit_point(Acceptance::Step(&attempt), &x, state)?;
                            return Ok(AcceptedPoint {
                                cursor: attempt.accept(next),
                                solution: x,
                                observed,
                                device_breaks,
                            });
                        }
                    }
                }
            }
        }
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
        let rhs = self.system.transient_rhs(0., crate::devices::Limit::Left)?;
        let (x, trial) = if self.uic {
            let start = initial::uic_start(self.circuit, &self.hints, &self.model_context)?;
            // C's single MODETRANOP|MODEUIC|MODEINITJCT load: nonlinear
            // devices evaluate at their instance initial conditions.
            let trial = self.initial_state(&start.x, &start.charges, true)?;
            let tolerances = &self.settings.tolerances;
            initial::check_impulse_free(
                self.circuit,
                &self.system.a,
                &self
                    .system
                    .transient_rhs(0., crate::devices::Limit::Right)?,
                &start.x,
                initial::Tolerances {
                    reltol: tolerances.reltol,
                    vntol: tolerances.vntol,
                    abstol: tolerances.abstol,
                },
                &self.model_context,
            )?;
            (start.x, trial)
        } else {
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
            let (x, solved) = {
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
                let solution = crate::analysis::bias::solve_dc_forced_prepared(
                    &self.prepared,
                    &self.model_context,
                    &self.bias,
                    &[],
                    Some(&seed),
                    // MODETRANOP: the sources' time functions at 0-.
                    Some(Forcing {
                        limit: crate::devices::Limit::Left,
                        timing: self.timing,
                    }),
                    &nodes,
                )?
                .solution;
                (solution.values, Some(solution.trial))
            };
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
            let trial = match solved {
                Some(trial) => trial,
                None => self.initial_state(&x, &[], false)?,
            };
            (x, trial)
        };
        // C copies CKTstate0 into CKTstate1..3; the derivative is zero. The
        // port fills every retained vector (ACCEPTED_DEPTH, sized for Gear
        // order 6); dctran.c's orders 1-2 never read beyond CKTstate3.
        // The t = 0 point also starts every transmission line's delay
        // history (traload.c, MODEINITTRAN); it requests no breakpoints.
        let device_breaks = self.commit_point(Acceptance::Initial, &x, trial)?;
        Ok(AcceptedPoint {
            cursor: AcceptedCursor::initial(&self.settings),
            solution: x,
            observed: Vec::new(),
            device_breaks,
        })
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
            self.history.set_initial_slot(&mut trial, *slot, *value)?;
        }
        Ok(trial)
    }

    /// One trial point: C `NIiter` from `previous` (`CKTrhsOld`: the last
    /// iterate of the previous trial, accepted or not; ngspice is built
    /// without `PREDICTOR`), `MODEINITTRAN` before the first accepted step
    /// (`firsttime`) and `MODEINITPRED` after, for linear and nonlinear
    /// circuits alike. The trial is that of the last Newton load, one iterate
    /// behind the returned solution, as C's `CKTstate0`.
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
    /// `10 CKTdelmin`, the same merge distance as the [`BreakTable`].
    fn delay_acceptance(&self, dt: Real, start: bool) -> crate::devices::DelayAcceptance {
        let accepted = self.steps.accepted();
        let older = |age: usize| accepted.get(age).copied().unwrap_or(self.settings.max_step);
        crate::devices::DelayAcceptance {
            steps: [dt, older(0), older(1)],
            min_break: 10. * self.settings.min_step(),
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
        for (device, (rows, bases)) in self.circuit.devices().iter().zip(&self.truncation) {
            for &base in bases {
                let accepted = |age: usize, offset: usize| {
                    self.history
                        .accepted(age)
                        .and_then(|vector| vector.get(base + offset))
                        .copied()
                        .ok_or_else(|| failure("accepted state history is too short"))
                };
                // The trial charge, then `order + 1` accepted ones.
                let mut storage = [0.0; IntegrationMethod::MAX_GEAR_ORDER as usize + 2];
                let charge = storage
                    .get_mut(..order + 2)
                    .ok_or_else(|| failure("integration order out of range"))?;
                charge[0] = trial.values()[base];
                for (age, value) in charge.iter_mut().enumerate().skip(1) {
                    *value = accepted(age, 0)?;
                }
                let derivative = [trial.values()[base + 1], accepted(1, 1)?];
                limit =
                    limit.min(coefficients.truncation_timestep(charge, derivative, &tolerances)?);
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
    header: &Plot,
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
    point.extend_from_slice(observed);
    header.check_point(point)?;
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
