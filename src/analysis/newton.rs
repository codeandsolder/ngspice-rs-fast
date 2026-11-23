//! Reusable, disposable Newton loads for DC and companion transient solves.
//!
//! C references: `niiter.c`, `niconv.c`, `cktop.c`. Step control follows
//! [`StepLimiting`]: by default ([`StepLimiting::Device`]) the phased solve
//! leaves it to the devices, which limit their own junction/FET voltages as
//! ngspice does (`DEVpnjlim`/`DEVfetlim`/`DEVlimvds`, see
//! [`crate::devices::limiting`]) and mark limited loads nonconvergent; the
//! port's earlier bounded global voltage-step damping remains available as
//! the [`StepLimiting::Global`] fallback policy and is what the unphased
//! [`solve`]/[`solve_counted`] apply (their loads cannot limit).
//! No trial or continuation stage invokes device acceptance hooks.
//!
//! [`solve_phased`] additionally tags each load with C's `MODEINITF` phase
//! ([`IterationPhase`], `niiter.c`), hands it the previous load's trial (C's
//! `CKTstate0` survives between iterations) and refuses to converge on a load
//! a device marked nonconvergent (C `CKTnoncon`), as the S/W switches need.
use crate::devices::circuit::PreparedTransient;
use crate::devices::{
    AnalysisMode, ConvergenceRequest, Forcing, IterationPhase, LoadRequest, ModelContext,
    PreparedRun, StateHistory, TrialState,
};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Largest accepted per-solve Newton iteration limit (`maxiter`, deck `itl1`).
pub const MAX_ITERATIONS: usize = 10_000;

/// How Newton bounds the step between iterates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StepLimiting {
    /// ngspice's policy (`niiter.c` applies no global damping by default):
    /// in [`solve_phased`] the devices limit their own controlling voltages
    /// in every load ([`crate::devices::limiting`]) and the iterate is never
    /// damped. Nonlinear devices without limiting in C (behavioural sources,
    /// switches) take full steps, as in C.
    #[default]
    Device,
    /// The port's original bounded policy, kept as a fallback: devices load
    /// exactly at the iterate and the largest nodal step is scaled down to
    /// [`NewtonOptions::voltage_step`] (on the rows of
    /// `bias::limited_rows`). Loads for this policy must forbid
    /// device limiting ([`crate::devices::TrialState::with_device_limiting`]).
    Global,
}

impl StepLimiting {
    /// Whether devices limit their own voltages (trials allow it).
    #[must_use]
    pub const fn is_device(self) -> bool {
        matches!(self, Self::Device)
    }
}

/// Finite work and physical voltage/current tolerances for Newton iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NewtonOptions {
    /// Maximum loads/solves per continuation stage.
    pub max_iterations: usize,
    /// Relative iterate tolerance (C `CKTreltol`).
    pub reltol: Real,
    /// Absolute voltage tolerance (V).
    pub vntol: Real,
    /// Absolute current tolerance (A).
    pub abstol: Real,
    /// Maximum nodal voltage change per iteration (V) under
    /// [`StepLimiting::Global`] (and in the unphased [`solve`]); limits trial
    /// exponentials.
    pub voltage_step: Real,
    /// Step control of [`solve_phased`]; see [`StepLimiting`].
    pub limiting: StepLimiting,
    /// C `CKTdiagGmin`: the conductance gmin stepping adds to every
    /// internal diagonal element of SPARSE's frame before each factorization
    /// (`LoadGmin` in `SMPreorder`/`SMPluFac`); zero otherwise.
    pub diag_gmin: Real,
    /// C `MODEDC && CKThadNodeset`: the solve has nodesets, so `NIiter`'s
    /// `ipass` keeps the first `MODEINITFLOAT` pass after `MODEINITFIX`
    /// from converging.
    pub had_nodeset: bool,
}
impl Default for NewtonOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            // ngspice's defaults (cktntask.c): CKTreltol, CKTvoltTol, CKTabstol.
            reltol: crate::devices::DEFAULT_RELTOL,
            vntol: 1e-6,
            abstol: 1e-12,
            voltage_step: 0.2,
            limiting: StepLimiting::Device,
            diag_gmin: 0.,
            had_nodeset: false,
        }
    }
}
impl NewtonOptions {
    /// Validate tolerances and bounded work before the first load.
    ///
    /// # Errors
    /// Returns a numerical error for invalid tolerances, damping limits, or iteration budgets.
    pub fn validate(&self) -> SpiceResult<()> {
        if !(1..=MAX_ITERATIONS).contains(&self.max_iterations)
            || [self.reltol, self.vntol, self.abstol, self.voltage_step]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.)
            || !self.diag_gmin.is_finite()
            || self.diag_gmin < 0.
        {
            return Err(failure(
                "invalid Newton tolerances, voltage limit or iteration budget",
            ));
        }
        Ok(())
    }
}

/// A converged Newton solution and the trial of the load that produced it.
#[derive(Debug)]
pub struct NewtonSolution<T> {
    /// Converged physical MNA values.
    pub values: Vector,
    /// Trial data belonging to `values`, not to the preceding iterate.
    pub trial: T,
    /// The solve of the last load (C's `CKTrhs` when `NIiter` returns: the
    /// iterate after `values`, which `.sens` multiplies its `delta_Y` by).
    pub next: Vector,
    /// Number of solves, bounded by [`NewtonOptions::max_iterations`].
    pub iterations: usize,
}

/// Solve `J(x) x_new = b_equivalent(x)` using fresh, immutable trial loads.
///
/// `branch_rows[r]` selects current tolerance for unknown `r` and voltage
/// tolerance for its KVL equation. Other rows use voltage iterate/current KCL
/// tolerances, as `NIiter` does (see [`solve_phased`]): no device convergence
/// tests, and the global voltage-step damping of [`StepLimiting::Global`].
/// The callback must not commit state.
///
/// # Errors
/// Invalid dimensions/settings, load failures, singular/nonfinite systems, or
/// exhaustion of the iteration budget. No acceptance hooks are called.
pub fn solve<T>(
    initial: &Vector,
    branch_rows: &[bool],
    options: &NewtonOptions,
    load: impl FnMut(&Vector) -> SpiceResult<(SparseMatrix, Vector, T)>,
) -> SpiceResult<NewtonSolution<T>> {
    solve_counted(initial, branch_rows, options, load).map_err(|failure| failure.error)
}

/// A failed Newton solve and the number of iterations it consumed.
#[derive(Debug, Clone, PartialEq)]
pub struct NewtonFailure {
    /// Why the solve failed (unchanged from [`solve`]).
    pub error: SpiceError,
    /// Load attempts, including a failed load. C checks the iteration limit
    /// after a solve, so an exhausted solve includes the over-limit pass.
    pub iterations: usize,
    /// At the iteration limit: the iterate the last load used. `NIiter`
    /// returns `E_ITERLIM` before swapping `CKTrhs` and `CKTrhsOld`, so this is
    /// the `CKTrhsOld` the next solve (`dctran.c`'s retry) starts from.
    pub last_iterate: Option<Vector>,
}

/// [`solve`] that also reports the work a failed solve used, so a caller owning a
/// total work budget (DC continuation) charges exactly what was spent.
///
/// # Errors
/// As [`solve`], with the iteration count in [`NewtonFailure`].
pub fn solve_counted<T>(
    initial: &Vector,
    branch_rows: &[bool],
    options: &NewtonOptions,
    load: impl FnMut(&Vector) -> SpiceResult<(SparseMatrix, Vector, T)>,
) -> Result<NewtonSolution<T>, NewtonFailure> {
    solve_counted_limited(initial, branch_rows, None, options, load)
}

/// Apply voltage-step damping only to selected matrix rows.
///
/// [`solve_counted`] whose voltage-step damping watches only the rows marked
/// in `limited_rows` (all non-branch rows when `None`). Devices that C never
/// limits, such as behavioural sources (`asrcload.c` has no limiting), opt out
/// through [`crate::devices::Device::limits_voltage_steps`]; see
/// [`crate::analysis::bias::limited_rows`].
///
/// # Errors
/// As [`solve_counted`], plus a `limited_rows` of the wrong length.
pub fn solve_counted_limited<T>(
    initial: &Vector,
    branch_rows: &[bool],
    limited_rows: Option<&[bool]>,
    options: &NewtonOptions,
    mut load: impl FnMut(&Vector) -> SpiceResult<(SparseMatrix, Vector, T)>,
) -> Result<NewtonSolution<T>, NewtonFailure> {
    iterate(
        initial,
        branch_rows,
        limited_rows,
        options,
        Phases {
            first: IterationPhase::Junction,
            previous: None,
            nonconvergent: |_: &T| false,
            damped: true,
            initial_transient: false,
            nodesets: options.had_nodeset,
        },
        |x: &Vector, _, _: Option<&T>| load(x),
        |_: &Vector, _: &Vector, _: &T| Ok(false),
    )
}

/// How a phased solve starts (C `NIiter`'s initial `MODEINITF`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhasePolicy {
    /// A DC operating point from scratch: `MODEINITJCT` for the first load,
    /// `MODEINITFIX` until the iterate first converges, then `MODEINITFLOAT`.
    OperatingPoint,
    /// A point continuing accepted history (a transient timepoint or a
    /// warm-started DC sweep point): `MODEINITPRED`, then `MODEINITFLOAT`.
    Predicted,
    /// The first transient timepoint (`MODEINITTRAN`): loaded as
    /// [`Self::Predicted`], but `NIiter` reorders on its first two passes.
    InitialTransient,
}

/// [`solve_counted`] for device trials with C's Newton phases.
///
/// `load(x, phase, previous)` must build its trial with
/// [`StateHistory::trial_in(phase, previous)`](crate::devices::StateHistory::trial_in). A load whose
/// trial [`TrialState::is_nonconvergent`] never ends the solve (C: `CKTnoncon`
/// set by a device). Otherwise `NIconvTest` compares successive iterates and
/// then calls `device_test(new, old, trial)`, the devices' convergence tests
/// (C `CKTconvTest`, see [`crate::devices::Circuit::convergence_test`]),
/// which returns `true` when a device has not converged. The mode sequence
/// and reordering are `NIiter`'s; see `iterate`. A `continued` trial (the converged trial
/// of a preceding continuation stage, as `cktop.c` continues its stepping
/// stages in `MODEINITFLOAT` with `CKTstate0` intact) starts the solve in
/// `MODEINITFLOAT` with that trial as the previous iterate instead. Devices
/// without discrete state load identically in every phase, so their solves
/// and iteration counts match [`solve_counted`].
///
/// `limited_rows` selects the rows watched by voltage-step damping, as in
/// [`solve_counted_limited`].
///
/// # Errors
/// As [`solve_counted_limited`].
#[expect(
    clippy::too_many_arguments,
    reason = "keeps the positional interface explicit at the reference solver or device boundary"
)]
pub fn solve_phased(
    initial: &Vector,
    branch_rows: &[bool],
    limited_rows: Option<&[bool]>,
    options: &NewtonOptions,
    policy: PhasePolicy,
    continued: Option<TrialState>,
    mut load: impl FnMut(
        &Vector,
        IterationPhase,
        Option<&TrialState>,
    ) -> SpiceResult<(SparseMatrix, Vector, TrialState)>,
    device_test: impl FnMut(&Vector, &Vector, &TrialState) -> SpiceResult<bool>,
) -> Result<NewtonSolution<TrialState>, NewtonFailure> {
    let first = match (policy, &continued) {
        (_, Some(_)) => IterationPhase::Float,
        (PhasePolicy::OperatingPoint, None) => IterationPhase::Junction,
        (PhasePolicy::Predicted | PhasePolicy::InitialTransient, None) => IterationPhase::Predict,
    };
    iterate(
        initial,
        branch_rows,
        limited_rows,
        options,
        Phases {
            first,
            previous: continued,
            nonconvergent: TrialState::is_nonconvergent,
            damped: !options.limiting.is_device(),
            initial_transient: policy == PhasePolicy::InitialTransient,
            nodesets: options.had_nodeset,
        },
        |x: &Vector, phase, previous: Option<&TrialState>| {
            let loaded = load(x, phase, previous)?;
            if loaded.2.device_limiting() != options.limiting.is_device() {
                return Err(failure(
                    "Newton load trial does not match the step-limiting policy",
                ));
            }
            Ok(loaded)
        },
        device_test,
    )
}

/// The two circuit-load modes driven by the real Newton solver. Keeping
/// transient-only state together prevents DC callers from accidentally
/// providing integration coefficients or the transient convergence time.
pub(super) enum CircuitLoad<'a, 'p> {
    Bias {
        prepared: &'a PreparedRun<'a>,
        context: &'a ModelContext,
        history: &'a StateHistory,
        forcing: Option<Forcing>,
    },
    Transient {
        attempt: &'a mut PreparedTransient<'p>,
        uic: bool,
    },
}

/// The shared device-backed Newton driver. Each iteration creates its trial
/// from accepted history and the previous disposable trial, stamps the circuit,
/// and runs device convergence tests with the same options that loaded it.
/// Acceptance is deliberately left to the outer analysis.
pub(super) struct CircuitNewton<'a, 'p> {
    pub branch_rows: &'a [bool],
    pub limited_rows: Option<&'a [bool]>,
    pub options: &'a NewtonOptions,
    pub load: CircuitLoad<'a, 'p>,
}

impl CircuitNewton<'_, '_> {
    /// The bias point may add forcing rows after the circuit load; ordinary
    /// transient trials pass a no-op postload. The phase sequence and device
    /// state never escape the common driver until Newton has converged.
    pub(super) fn solve(
        &mut self,
        initial: &Vector,
        policy: PhasePolicy,
        continued: Option<TrialState>,
        mut after_load: impl FnMut(IterationPhase, &mut SparseMatrix, &mut Vector) -> SpiceResult<()>,
    ) -> Result<NewtonSolution<TrialState>, NewtonFailure> {
        // Both stamping and convergence read the same attempt-bound model
        // and accepted history, with no second caller-supplied identity.
        let (prepared, context, history, forcing, time, uic, mut attempt) = match &mut self.load {
            CircuitLoad::Bias {
                prepared,
                context,
                history,
                forcing,
            } => (*prepared, *context, *history, *forcing, 0., None, None),
            CircuitLoad::Transient { attempt, uic } => (
                attempt.run(),
                attempt.context(),
                attempt.history(),
                None,
                attempt.time(),
                Some(*uic),
                Some(&mut **attempt),
            ),
        };
        let options = self.options;
        let circuit = prepared.circuit();
        solve_phased(
            initial,
            self.branch_rows,
            self.limited_rows,
            options,
            policy,
            continued,
            |guess, phase, previous| {
                let mut matrix = circuit.new_matrix()?;
                let mut rhs = Vector::zeros(circuit.unknown_count());
                let mut trial = history
                    .trial_in(phase, previous)?
                    .with_device_limiting(options.limiting.is_device())
                    .with_convergence_tolerances(options.reltol, options.abstol);
                if let Some(uic) = uic {
                    trial = trial
                        .with_initial_transient(
                            policy == PhasePolicy::InitialTransient
                                && phase == IterationPhase::Predict,
                        )
                        .with_uic(uic);
                }
                if let Some(attempt) = &mut attempt {
                    attempt.load(guess, &mut matrix, &mut rhs, &mut trial)?;
                } else {
                    prepared.load(
                        &LoadRequest {
                            mode: AnalysisMode::OperatingPoint,
                            solution: guess,
                            model_context: context,
                            integration: None,
                            history,
                            forcing,
                        },
                        &mut matrix,
                        &mut rhs,
                        &mut trial,
                    )?;
                }
                after_load(phase, &mut matrix, &mut rhs)?;
                Ok((matrix, rhs, trial))
            },
            |new, old, trial| {
                prepared.convergence_test(&ConvergenceRequest {
                    solution: new,
                    previous: old,
                    trial,
                    history,
                    model_context: context,
                    time,
                    reltol: options.reltol,
                    abstol: options.abstol,
                    vntol: options.vntol,
                })
            },
        )
    }
}

struct Phases<T, F> {
    first: IterationPhase,
    previous: Option<T>,
    nonconvergent: F,
    /// Apply the global voltage-step damping ([`StepLimiting::Global`]).
    damped: bool,
    /// `MODEINITTRAN`: `NIiter` reorders on the first iteration and again
    /// after it.
    initial_transient: bool,
    /// `MODEDC` with nodesets (`CKThadNodeset`): the first `MODEINITFLOAT`
    /// iterate after `MODEINITFIX` never converges (`ipass`).
    nodesets: bool,
}

/// Counts load attempts and owns the retry iterate for all Newton entry points.
#[derive(Default)]
struct Progress {
    iterations: usize,
    last_iterate: Option<Vector>,
}

fn iterate<T, F: Fn(&T) -> bool>(
    initial: &Vector,
    branch_rows: &[bool],
    limited_rows: Option<&[bool]>,
    options: &NewtonOptions,
    phases: Phases<T, F>,
    load: impl FnMut(&Vector, IterationPhase, Option<&T>) -> SpiceResult<(SparseMatrix, Vector, T)>,
    device_test: impl FnMut(&Vector, &Vector, &T) -> SpiceResult<bool>,
) -> Result<NewtonSolution<T>, NewtonFailure> {
    let mut progress = Progress::default();
    iterate_inner(
        initial,
        branch_rows,
        limited_rows,
        options,
        phases,
        load,
        device_test,
        &mut progress,
    )
    .map_err(|error| NewtonFailure {
        error,
        iterations: progress.iterations,
        last_iterate: progress.last_iterate,
    })
}

/// C `NIiter` (`niiter.c`, the SPARSE path) over disposable loads.
///
/// Each pass loads at the present iterate, factors the matrix of the
/// circuit (reordering under `NISHOULDREORDER`: every `MODEINITJCT` load,
/// the load after it, and the first two loads of `MODEINITTRAN`; otherwise
/// refactoring, which after a zero pivot reloads at the same iterate and
/// reorders) with `diag_gmin` on the diagonal, and solves. Past
/// `max_iterations` it fails. A load that set `CKTnoncon` (a device trial
/// marked nonconvergent) and the first pass never converge; otherwise
/// `NIconvTest` compares the new iterate with the previous one (`reltol`
/// with `vntol` for node rows and `abstol` for branch rows) and then runs the
/// devices' convergence tests (`device_test(new, old, trial)`, C
/// `CKTconvTest`). Mode changes follow `NIiter`: `MODEINITJCT` to
/// `MODEINITFIX`, which ends at the first converged pass, `MODEINITPRED`
/// (and `MODEINITTRAN`) to `MODEINITFLOAT`, where a converged pass returns
/// the iterate that pass loaded (C's `CKTrhsOld`, within the tolerances of
/// the new one) with that load's trial (C's `CKTstate0`); no further load
/// is made.
#[expect(
    clippy::too_many_arguments,
    reason = "keeps the positional interface explicit at the reference solver or device boundary"
)]
fn iterate_inner<T, F: Fn(&T) -> bool>(
    initial: &Vector,
    branch_rows: &[bool],
    limited_rows: Option<&[bool]>,
    options: &NewtonOptions,
    phases: Phases<T, F>,
    mut load: impl FnMut(&Vector, IterationPhase, Option<&T>) -> SpiceResult<(SparseMatrix, Vector, T)>,
    mut device_test: impl FnMut(&Vector, &Vector, &T) -> SpiceResult<bool>,
    progress: &mut Progress,
) -> SpiceResult<NewtonSolution<T>> {
    options.validate()?;
    let n = initial.len();
    if n == 0
        || n != branch_rows.len()
        || limited_rows.is_some_and(|rows| rows.len() != n)
        || !initial.is_finite()
    {
        return Err(failure(
            "Newton requires a nonempty square system, finite solution and matching row kinds",
        ));
    }
    let mut guess = initial.clone();
    let mut phase = phases.first;
    let mut previous = phases.previous;
    let mut ordering = None;
    let mut reorder = phases.initial_transient;
    let mut ipass = false;
    loop {
        // NIiter increments before checking the CKTload return code.
        progress.iterations += 1;
        let iterno = progress.iterations;
        let (mut matrix, rhs, trial) = load(&guess, phase, previous.as_ref())?;
        let flagged = (phases.nonconvergent)(&trial);
        check(&matrix, &rhs, n)?;
        matrix.fold_duplicates();
        if phase == IterationPhase::Junction {
            reorder = true;
        }
        let mut next = match linearised_solve(&matrix, &rhs, &mut ordering, reorder, options)? {
            crate::maths::NiStep::Solved(next) => {
                reorder = false;
                next
            }
            crate::maths::NiStep::ZeroPivot => {
                // E_SINGULAR from SMPluFac: NISHOULDREORDER and `continue`,
                // reloading at the same iterate.
                reorder = true;
                previous = Some(trial);
                continue;
            }
        };
        if phases.damped {
            damp(
                &mut next,
                &guess,
                branch_rows,
                limited_rows,
                options.voltage_step,
            );
        }
        if !next.is_finite() {
            return Err(failure("nonfinite Newton iterate"));
        }
        if iterno > options.max_iterations {
            progress.last_iterate = Some(guess);
            return Err(failure(format!(
                "Newton iteration limit ({}) reached",
                options.max_iterations
            )));
        }
        let mut noncon = if flagged || iterno == 1 {
            true
        } else {
            !converged(&next, &guess, branch_rows, options) || device_test(&next, &guess, &trial)?
        };
        match phase {
            IterationPhase::Float => {
                if phases.nodesets {
                    noncon |= ipass;
                    ipass = false;
                }
                if !noncon {
                    // NIiter returns before swapping CKTrhs and CKTrhsOld:
                    // the solution ngspice keeps (CKTdump writes CKTrhsOld,
                    // the next solve starts from it) is the iterate the last
                    // load used, the one CKTstate0 belongs to.
                    return Ok(NewtonSolution {
                        values: guess,
                        trial,
                        next,
                        iterations: iterno,
                    });
                }
            }
            IterationPhase::Junction => {
                phase = IterationPhase::Fix;
                reorder = true;
            }
            IterationPhase::Fix => {
                if !noncon {
                    phase = IterationPhase::Float;
                }
                ipass = true;
            }
            IterationPhase::Predict => {
                if phases.initial_transient && iterno <= 1 {
                    reorder = true;
                }
                phase = IterationPhase::Float;
            }
        }
        previous = Some(trial);
        guess = next;
    }
}
/// [`StepLimiting::Global`]: scale the step so no watched nodal voltage moves
/// by more than `limit` (all non-branch rows when `limited_rows` is `None`).
fn damp(
    next: &mut Vector,
    guess: &Vector,
    branch_rows: &[bool],
    limited_rows: Option<&[bool]>,
    limit: Real,
) {
    let largest_voltage_step = next
        .as_slice()
        .iter()
        .zip(guess.as_slice())
        .zip(branch_rows)
        .enumerate()
        .filter(|(row, (_, branch))| !**branch && limited_rows.is_none_or(|limited| limited[*row]))
        .map(|(_, ((new, old), _))| (new - old).abs())
        .fold(0., Real::max);
    let damping = (limit / largest_voltage_step).min(1.);
    if damping < 1. {
        for (new, old) in next.as_mut_slice().iter_mut().zip(guess.as_slice()) {
            *new = old + damping * (*new - old);
        }
    }
}
/// One `NIiter` factor-and-solve on the raw matrix ([`crate::maths::solve_ni`]):
/// there is no row equilibration or conditioning guard; ngspice has neither.
fn linearised_solve(
    matrix: &SparseMatrix,
    rhs: &Vector,
    ordering: &mut Option<crate::maths::SparseSymbolic>,
    reorder: bool,
    options: &NewtonOptions,
) -> SpiceResult<crate::maths::NiStep> {
    crate::maths::solve_ni(matrix, rhs, ordering, reorder, options.diag_gmin)
}

fn check(matrix: &SparseMatrix, rhs: &Vector, n: usize) -> SpiceResult<()> {
    if matrix.rows() != n || matrix.cols() != n || rhs.len() != n || !rhs.is_finite() {
        return Err(failure("invalid Newton load dimensions or nonfinite RHS"));
    }
    Ok(())
}
fn converged(new: &Vector, old: &Vector, branch: &[bool], options: &NewtonOptions) -> bool {
    new.as_slice()
        .iter()
        .zip(old.as_slice())
        .zip(branch)
        .all(|((new, old), branch)| {
            let relative_tolerance = options.reltol * new.abs().max(old.abs());
            let absolute_tolerance = if *branch {
                options.abstol
            } else {
                options.vntol
            };
            (new - old).abs() <= relative_tolerance + absolute_tolerance
        })
}
fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "Newton".into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn solves_nonlinear_equation_and_returns_state_at_solution() {
        let initial = Vector::zeros(1);
        let solution = solve(&initial, &[false], &NewtonOptions::default(), |x| {
            let v = x.as_slice()[0];
            let mut a = SparseMatrix::new(1, 1);
            a.add(0, 0, v.exp())?;
            let mut b = Vector::zeros(1);
            b.add_to(0, 2. - v.exp() + v.exp() * v)?;
            Ok((a, b, v))
        })
        .unwrap();
        assert!((solution.values.as_slice()[0] - 2_f64.ln()).abs() < 1e-9);
        assert_eq!(solution.trial, solution.values.as_slice()[0]);
    }
    #[test]
    fn counted_solve_reports_the_iterations_used_by_success_and_failure() {
        let identity = |_: &Vector| -> SpiceResult<(SparseMatrix, Vector, ())> {
            let mut a = SparseMatrix::new(1, 1);
            a.add(0, 0, 1.)?;
            Ok((a, Vector::zeros(1), ()))
        };
        // x_{n+1} = x_n + 1 never converges.
        let drifting = |x: &Vector| -> SpiceResult<(SparseMatrix, Vector, ())> {
            let mut a = SparseMatrix::new(1, 1);
            a.add(0, 0, 1.)?;
            let mut b = Vector::zeros(1);
            b.add_to(0, x.as_slice()[0] + 1.)?;
            Ok((a, b, ()))
        };
        let far = Vector::zeros(1);
        let options = NewtonOptions {
            max_iterations: 3,
            ..NewtonOptions::default()
        };
        // NIiter fails after the solve that exceeds the limit: 3 + 1 passes.
        let failure = solve_counted(&far, &[false], &options, drifting).unwrap_err();
        assert_eq!(failure.iterations, 4);
        assert!(failure.error.to_string().contains("iteration limit"));
        assert_eq!(
            solve(&far, &[false], &options, drifting).unwrap_err(),
            failure.error
        );
        // MODEINITJCT, MODEINITFIX, then a converged MODEINITFLOAT pass.
        let solved = solve_counted(&Vector::zeros(1), &[false], &options, identity).unwrap();
        assert_eq!(solved.iterations, 3);
        let rejected = NewtonOptions {
            max_iterations: 0,
            ..options
        };
        let failure = solve_counted(&far, &[false], &rejected, identity).unwrap_err();
        assert_eq!(failure.iterations, 0);
    }
    #[test]
    fn failed_load_is_charged_to_the_attempt_budget() {
        let mut loads = 0;
        let error = failure("device load failed");
        let failed = solve_counted(
            &Vector::zeros(1),
            &[false],
            &NewtonOptions::default(),
            |_| {
                loads += 1;
                if loads == 2 {
                    return Err(error.clone());
                }
                let mut matrix = SparseMatrix::new(1, 1);
                matrix.add(0, 0, 1.)?;
                Ok((matrix, Vector::zeros(1), ()))
            },
        )
        .unwrap_err();
        assert_eq!(failed.error, error);
        assert_eq!(loads, 2);
        assert_eq!(failed.iterations, loads);
        assert!(failed.last_iterate.is_none());
    }

    #[test]
    fn rejects_singular_homogeneous_system_and_invalid_settings() {
        assert!(
            solve(
                &Vector::zeros(1),
                &[false],
                &NewtonOptions::default(),
                |_| Ok((SparseMatrix::new(1, 1), Vector::zeros(1), ()))
            )
            .is_err()
        );
        assert!(
            NewtonOptions {
                abstol: Real::NAN,
                ..NewtonOptions::default()
            }
            .validate()
            .is_err()
        );
    }
}
