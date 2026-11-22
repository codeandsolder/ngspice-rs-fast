//! Nonlinear DC operating points with bounded continuation (`cktop.c`).
//!
//! The policy and its diagnostics are typed: [`DcSettings`] resolves Newton
//! limits and a [`ContinuationPolicy`] (ngspice's adaptive `CKTop`
//! strategies by default, or the port's fixed gmin/source ladders, plus a
//! total work budget); [`solve_dc_with`] returns a [`DcReport`] of every
//! attempt and stage on success *and* failure. [`solve_dc`] is the compatible
//! default wrapper. See `docs/port/DC_CONTINUATION.md` for the contract and
//! for the remaining differences from ngspice.
use crate::analysis::newton::{
    self, CircuitLoad, CircuitNewton, NewtonFailure, NewtonOptions, NewtonSolution, PhasePolicy,
};
use crate::devices::{
    Circuit, Forcing, IterationPhase, ModelContext, PreparedRun, StateHistory, TrialState,
};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Default nodal-gmin schedule (S): one decade per stage, 1e-3 down to 1e-12.
pub const DEFAULT_GMIN_SCHEDULE: [Real; 10] = [
    1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12,
];
/// First value (S) of a schedule built from `gminsteps`/`gminfactor`.
pub const GMIN_START: Real = 1e-3;
/// Default ratio between consecutive scheduled gmin values.
pub const DEFAULT_GMIN_FACTOR: Real = 10.;
/// Largest accepted `gminfactor`.
pub const MAX_GMIN_FACTOR: Real = 1e6;
/// Largest scheduled artificial nodal gmin (S).
pub const MAX_GMIN: Real = 1.;
/// Default number of equal source increments.
pub const DEFAULT_SOURCE_STEPS: u16 = 20;
/// Default temporary nodal gmin (S) held during source stepping.
pub const DEFAULT_SOURCE_GMIN: Real = 1e-8;
/// Maximum gmin steps, or explicit ladder entries. The ngspice
/// `spice3_gmin` schedule includes both endpoints (steps + 1 entries).
pub const MAX_GMIN_STAGES: usize = 100;
/// Most equal source increments accepted (a schedule has one more scale).
pub const MAX_SOURCE_STEPS: usize = 1_000;
/// Largest explicit total Newton-iteration budget.
pub const MAX_TOTAL_ITERATIONS: usize = 10_000_000;

fn uniform_scales(steps: u16) -> impl Iterator<Item = Real> {
    (0..=steps).map(move |step| Real::from(step) / Real::from(steps))
}

fn invalid(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "DC settings".into(),
        message: message.into(),
    }
}

/// A checked count with domain-specific inclusive bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedCount<const MIN: usize, const MAX: usize>(usize);

impl<const MIN: usize, const MAX: usize> BoundedCount<MIN, MAX> {
    /// Constructs a count within its inclusive bounds.
    ///
    /// # Errors
    /// The count is outside the bounds.
    pub fn new(value: usize) -> SpiceResult<Self> {
        if (MIN..=MAX).contains(&value) {
            Ok(Self(value))
        } else {
            Err(invalid(format!(
                "count must be in {MIN}..={MAX}, not {value}"
            )))
        }
    }

    /// The validated count.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// A fixed ngspice schedule has at least two increments.
pub type FixedSteps<const MAX: usize> = BoundedCount<2, MAX>;
/// Bounded Newton iteration limit (written adaptation base or stage limit).
pub type IterationLimit = BoundedCount<1, { newton::MAX_ITERATIONS }>;
/// Bounded total work limit for a DC solve.
pub type IterationBudget = BoundedCount<1, MAX_TOTAL_ITERATIONS>;

/// ngspice's count encoding, translated once at the request boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stepping<const MAX: usize> {
    /// No attempt.
    Disabled,
    /// C's adaptive strategy.
    Adaptive,
    /// C's fixed strategy with a bounded count.
    Fixed(FixedSteps<MAX>),
}

impl<const MAX: usize> Stepping<MAX> {
    /// Decode C's 0 (disabled), 1 (adaptive), or fixed count.
    ///
    /// # Errors
    /// The count exceeds the supported domain.
    pub fn from_count(count: usize) -> SpiceResult<Self> {
        match count {
            0 => Ok(Self::Disabled),
            1 => Ok(Self::Adaptive),
            _ => FixedSteps::new(count).map(Self::Fixed),
        }
    }
}

/// A validated, decreasing custom nodal-gmin ladder. Empty disables it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GminSchedule(Vec<Real>);

impl GminSchedule {
    /// Validate a custom ladder.
    ///
    /// # Errors
    /// Nonfinite, nonpositive, oversized or nondecreasing stages.
    pub fn new(values: Vec<Real>) -> SpiceResult<Self> {
        validate_gmin_schedule(&values, MAX_GMIN_STAGES)?;
        Ok(Self(values))
    }

    /// The temporary nodal-gmin values, in execution order.
    #[must_use]
    pub const fn as_slice(&self) -> &[Real] {
        self.0.as_slice()
    }
}

/// Source stepping: strictly increasing scales followed by a separate
/// full-source, zero-gmin solve. Temporary gmin and scales are immutable.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceStepping {
    scales: Vec<Real>,
    gmin: Real,
}

impl SourceStepping {
    /// Validate a custom source ladder.
    ///
    /// # Errors
    /// Empty/oversized, nonfinite, out-of-range or non-increasing scales, or
    /// an invalid temporary gmin.
    pub fn new(scales: Vec<Real>, gmin: Real) -> SpiceResult<Self> {
        if scales.is_empty() || scales.len() > MAX_SOURCE_STEPS + 1 {
            return Err(invalid(format!(
                "source schedule needs 1..={} scales, not {}",
                MAX_SOURCE_STEPS + 1,
                scales.len()
            )));
        }
        let mut previous = Real::NEG_INFINITY;
        for scale in &scales {
            if !scale.is_finite() || !(0. ..=1.).contains(scale) {
                return Err(invalid(format!(
                    "source scale {scale} must be finite and in [0, 1]"
                )));
            }
            if *scale <= previous {
                return Err(invalid(format!(
                    "source scales must strictly increase (no progress at {scale})"
                )));
            }
            previous = *scale;
        }
        if !gmin.is_finite() || !(0. ..=MAX_GMIN).contains(&gmin) {
            return Err(invalid(format!(
                "source-stepping gmin {gmin} must be finite and in [0, {MAX_GMIN}] S"
            )));
        }
        Ok(Self { scales, gmin })
    }

    /// Equal increments including both endpoints.
    ///
    /// # Errors
    /// Invalid step count or temporary gmin.
    pub fn uniform(steps: usize, gmin: Real) -> SpiceResult<Self> {
        let steps = BoundedCount::<1, MAX_SOURCE_STEPS>::new(steps)?.get();
        let steps = u16::try_from(steps).map_err(|_| invalid("source step count exceeds u16"))?;
        Self::new(uniform_scales(steps).collect(), gmin)
    }

    /// The validated source scales.
    #[must_use]
    pub const fn scales(&self) -> &[Real] {
        self.scales.as_slice()
    }

    /// The temporary nodal gmin.
    #[must_use]
    pub const fn gmin(&self) -> Real {
        self.gmin
    }
}

/// C's default `CKTdcTrcvMaxIter` as written (`cktntask.c`, deck `itl2`):
/// the adaptive ngspice strategies compare each stage's iterations with a
/// quarter and three quarters of it (`iters <= itl2 / 4`).
pub const NGSPICE_ADAPT_ITERATIONS: usize = 50;
/// Most stages one adaptive ngspice strategy may run. A port-only bound: C
/// stops only at its step-size floors, which can take far longer.
pub const MAX_ADAPTIVE_STAGES: usize = 1_000;
/// `dynamic_gmin`/`new_gmin` start from `OldGmin = 1e-2` S divided by the factor.
const DYNAMIC_GMIN_START: Real = 1e-2;
/// `dynamic_gmin` gives up once a failed step's factor is below this, and
/// never shrinks the factor below it after a slow stage.
const DYNAMIC_FACTOR_FLOOR: Real = 1.00005;
/// `new_gmin` never shrinks its factor below 3 after a slow stage.
const TRUE_GMIN_SLOW_FACTOR: Real = 3.;
/// `gillespie_src`: first source increment, smallest increment, increment cap
/// after a failure and smallest progress worth retrying.
const GILLESPIE_FIRST_RAISE: Real = 1e-3;
const GILLESPIE_MIN_RAISE: Real = 1e-7;
const GILLESPIE_MAX_RAISE: Real = 0.01;
const GILLESPIE_MIN_PROGRESS: Real = 1e-8;
/// `gillespie_src`'s zero-source gmin ladder spans ten decades above `gmin`.
const GILLESPIE_GMIN_DECADES: usize = 10;

/// ngspice's `CKTop` strategies: adaptive gmin tries both nodal and junction
/// gmin, adaptive source stepping uses Gillespie; fixed counts use Spice3.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NgspiceStepping {
    /// C `CKTnumGminSteps`, decoded at construction.
    pub gmin: Stepping<MAX_GMIN_STAGES>,
    /// C `CKTnumSrcSteps`, decoded at construction.
    pub source: Stepping<MAX_SOURCE_STEPS>,
    gmin_factor: Real,
    adapt_iterations: IterationLimit,
}

impl Default for NgspiceStepping {
    fn default() -> Self {
        Self {
            gmin: Stepping::Adaptive,
            source: Stepping::Adaptive,
            gmin_factor: DEFAULT_GMIN_FACTOR,
            adapt_iterations: BoundedCount(NGSPICE_ADAPT_ITERATIONS),
        }
    }
}

impl NgspiceStepping {
    /// Decode C's step counts and validate the factor and adaptation base.
    ///
    /// # Errors
    /// Out-of-range counts, an invalid factor or invalid adaptation base.
    pub fn new(
        gmin_steps: usize,
        source_steps: usize,
        gmin_factor: Real,
        adapt_iterations: usize,
    ) -> SpiceResult<Self> {
        check_factor(gmin_factor)?;
        Ok(Self {
            gmin: Stepping::from_count(gmin_steps)?,
            source: Stepping::from_count(source_steps)?,
            gmin_factor,
            adapt_iterations: IterationLimit::new(adapt_iterations)?,
        })
    }

    /// C's written adaptation base, independent of the effective stage limit.
    #[must_use]
    pub const fn adapt_iterations(&self) -> usize {
        self.adapt_iterations.get()
    }

    /// Ratio used for artificial or junction gmin stepping.
    #[must_use]
    pub const fn gmin_factor(&self) -> Real {
        self.gmin_factor
    }

    /// Resolve the fixed gmin ladder for the configured junction gmin.
    /// Adaptive and disabled strategies have no fixed ladder.
    ///
    /// # Errors
    /// A fixed ladder is not finite, positive, bounded and decreasing.
    pub fn fixed_gmin_ladder(&self, gmin: Real) -> SpiceResult<Option<Vec<Real>>> {
        match self.gmin {
            Stepping::Fixed(steps) => self.spice3_ladder(gmin, steps).map(Some),
            Stepping::Disabled | Stepping::Adaptive => Ok(None),
        }
    }

    fn spice3_ladder(
        &self,
        gmin: Real,
        steps: FixedSteps<MAX_GMIN_STAGES>,
    ) -> SpiceResult<Vec<Real>> {
        let steps = steps.get();
        let mut value = gmin;
        for _ in 0..steps {
            value *= self.gmin_factor;
        }
        let mut ladder = Vec::with_capacity(steps + 1);
        for _ in 0..=steps {
            ladder.push(value);
            value /= self.gmin_factor;
        }
        validate_gmin_schedule(&ladder, MAX_GMIN_STAGES + 1).map_err(|_| {
            invalid(format!(
                "gminsteps={steps} with gminfactor={} and gmin={gmin:e} S gives the ladder \
                 {:e}..{gmin:e} S; each value must be finite, positive, at most \
                 {MAX_GMIN} S and decreasing",
                self.gmin_factor, ladder[0]
            ))
        })?;
        Ok(ladder)
    }
}

/// Disjoint strategies: custom ladder data cannot coexist with `CKTop` data.
#[derive(Debug, Clone, PartialEq)]
pub enum ContinuationSchedule {
    /// ngspice's `CKTop` strategies.
    Ngspice(NgspiceStepping),
    /// The port's fixed, deterministic ladders.
    Ladder {
        /// Validated nodal gmin ladder (empty disables the attempt).
        gmin: GminSchedule,
        /// Validated source ladder (None disables the attempt).
        source: Option<SourceStepping>,
    },
}

/// Bounded DC continuation used after direct Newton fails.
///
/// Order: direct Newton (not part of this policy), then gmin stepping, then
/// source stepping, as [`Self::schedule`] defines them. Each strategy ends in
/// a solve with **full sources, zero artificial nodal gmin and the configured
/// junction gmin**; only that solve may be returned. Temporary stages never
/// touch device history, accept hooks or stored source values.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationPolicy {
    /// Strategy family and its validated data.
    pub schedule: ContinuationSchedule,
    /// Total Newton iterations over *all* stages of one solve. `None` selects
    /// `NewtonOptions::max_iterations` for the direct solve plus the stage
    /// limit for every continuation stage, i.e. no tighter than the per-stage
    /// limits already imply; `Some(n)` caps the work at `n`.
    pub max_total_iterations: Option<IterationBudget>,
    /// Newton iteration limit of every gmin/source-stepping stage, including
    /// source stepping's final full-source solve (C: `NIiter(ckt,
    /// CKTdcTrcvMaxIter)`, deck `itl2`). Gmin stepping's closing zero-gmin
    /// solve is not a stage: like the direct solve it uses
    /// `NewtonOptions::max_iterations` (C: `dynamic_gmin`, `spice3_gmin` and
    /// `new_gmin` end with `NIiter(ckt, iterlim)`, i.e. `CKTdcMaxIter`, deck
    /// `itl1`). `None` keeps `NewtonOptions::max_iterations` for the stages too.
    pub stage_max_iterations: Option<IterationLimit>,
    /// Skip the direct Newton attempt and start with continuation (C
    /// `CKTnoOpIter`, deck `.options noopiter`).
    pub skip_direct: bool,
}

/// The ngspice schedule with C's defaults (`gminsteps=1`, `srcsteps=1`,
/// `gminfactor=10`, `itl2=50`): `dynamic_gmin`, `new_gmin`, `gillespie_src`.
impl Default for ContinuationPolicy {
    fn default() -> Self {
        Self::ngspice(NgspiceStepping::default())
    }
}

impl ContinuationPolicy {
    /// ngspice's strategies with the given parameters.
    #[must_use]
    pub fn ngspice(stepping: NgspiceStepping) -> Self {
        Self {
            schedule: ContinuationSchedule::Ngspice(stepping),
            ..Self::disabled()
        }
    }

    /// The port's original fixed ladders: nodal gmin 1e-3 down to 1e-12 S one
    /// decade per stage, then twenty equal source increments at 1e-8 S.
    #[must_use]
    pub fn ladder() -> Self {
        Self {
            schedule: ContinuationSchedule::Ladder {
                gmin: GminSchedule(DEFAULT_GMIN_SCHEDULE.to_vec()),
                source: Some(SourceStepping {
                    scales: uniform_scales(DEFAULT_SOURCE_STEPS).collect(),
                    gmin: DEFAULT_SOURCE_GMIN,
                }),
            },
            ..Self::disabled()
        }
    }

    /// No continuation: a failed direct Newton solve is final.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            schedule: ContinuationSchedule::Ladder {
                gmin: GminSchedule(Vec::new()),
                source: None,
            },
            max_total_iterations: None,
            stage_max_iterations: None,
            skip_direct: false,
        }
    }

    /// The ngspice schedule from C's step counts over C's defaults: `None`
    /// keeps a default, `Some(0)` disables a strategy (see
    /// [`NgspiceStepping`]).
    ///
    /// # Errors
    /// Out-of-range counts or an invalid factor/adaptation base.
    pub fn from_ngspice_steps(
        source_steps: Option<usize>,
        gmin_steps: Option<usize>,
        gmin_factor: Option<Real>,
        adapt_iterations: Option<usize>,
    ) -> SpiceResult<Self> {
        NgspiceStepping::new(
            gmin_steps.unwrap_or(1),
            source_steps.unwrap_or(1),
            gmin_factor.unwrap_or(DEFAULT_GMIN_FACTOR),
            adapt_iterations.unwrap_or(NGSPICE_ADAPT_ITERATIONS),
        )
        .map(Self::ngspice)
    }

    /// Geometric gmin schedule `start / factor^k`, `k = 0..steps`. `steps == 0`
    /// is an empty (disabled) schedule.
    ///
    /// # Errors
    /// Too many steps, an invalid `start`/`factor`, or a value that underflows.
    pub fn geometric_gmin(start: Real, factor: Real, steps: usize) -> SpiceResult<GminSchedule> {
        check_factor(factor)?;
        if !(start.is_finite() && 0. < start && start <= MAX_GMIN) {
            return Err(invalid(format!(
                "gmin start must be finite and in (0, {MAX_GMIN}] S, not {start}"
            )));
        }
        if steps > MAX_GMIN_STAGES {
            return Err(invalid(format!(
                "gmin steps must be in 0..={MAX_GMIN_STAGES}, not {steps}"
            )));
        }
        // The default decade ladder is returned exactly, not recomputed.
        let ladder = start == GMIN_START
            && factor == DEFAULT_GMIN_FACTOR
            && steps <= DEFAULT_GMIN_SCHEDULE.len();
        if ladder {
            return Ok(GminSchedule(DEFAULT_GMIN_SCHEDULE[..steps].to_vec()));
        }
        let steps = i32::try_from(steps).map_err(|_| invalid("gmin step count exceeds i32"))?;
        let schedule: Vec<Real> = (0..steps).map(|k| start / factor.powi(k)).collect();
        GminSchedule::new(schedule)
    }

    /// The ladder schedule from `srcsteps`/`gminsteps`/`gminfactor`-style
    /// counts over the ladder defaults ([`Self::ladder`]). `None` keeps the
    /// default; `Some(0)` disables a strategy. A `gminfactor` without
    /// `gminsteps` rebuilds the default number of stages.
    ///
    /// # Errors
    /// Out-of-range counts, an invalid factor or an underflowing schedule.
    pub fn from_steps(
        source_steps: Option<usize>,
        gmin_steps: Option<usize>,
        gmin_factor: Option<Real>,
    ) -> SpiceResult<Self> {
        let gmin = Self::geometric_gmin(
            GMIN_START,
            gmin_factor.unwrap_or(DEFAULT_GMIN_FACTOR),
            gmin_steps.unwrap_or(DEFAULT_GMIN_SCHEDULE.len()),
        )?;
        let steps = source_steps.unwrap_or_else(|| usize::from(DEFAULT_SOURCE_STEPS));
        let source = (steps > 0)
            .then(|| SourceStepping::uniform(steps, DEFAULT_SOURCE_GMIN))
            .transpose()?;
        Ok(Self {
            schedule: ContinuationSchedule::Ladder { gmin, source },
            ..Self::disabled()
        })
    }
}

fn check_factor(factor: Real) -> SpiceResult<()> {
    if factor.is_finite() && factor > 1. && factor <= MAX_GMIN_FACTOR {
        Ok(())
    } else {
        Err(invalid(format!(
            "gminfactor must be finite and in (1, {MAX_GMIN_FACTOR}], not {factor}"
        )))
    }
}

fn validate_gmin_schedule(schedule: &[Real], max_entries: usize) -> SpiceResult<()> {
    if schedule.len() > max_entries {
        return Err(invalid(format!(
            "gmin schedule has {} stages; at most {max_entries}",
            schedule.len()
        )));
    }
    let mut previous = Real::INFINITY;
    for gmin in schedule {
        if !gmin.is_finite() || *gmin <= 0. || *gmin > MAX_GMIN {
            return Err(invalid(format!(
                "scheduled gmin {gmin:e} must be finite and in (0, {MAX_GMIN}] S"
            )));
        }
        if *gmin >= previous {
            return Err(invalid(format!(
                "gmin schedule must strictly decrease (no progress at {gmin:e})"
            )));
        }
        previous = *gmin;
    }
    Ok(())
}

/// Newton limits plus continuation policy for one DC solve.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DcSettings {
    /// Per-stage Newton tolerances and iteration limit.
    pub newton: NewtonOptions,
    /// Continuation used after direct Newton fails.
    pub continuation: ContinuationPolicy,
}

impl DcSettings {
    /// Validate Newton tolerances; the continuation schedule is constructed valid.
    ///
    /// # Errors
    /// Invalid Newton options.
    pub fn validate(&self) -> SpiceResult<()> {
        self.newton.validate()
    }
}

/// How a DC solve was attempted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DcStrategy {
    /// Plain Newton from the seed at full sources and zero artificial gmin.
    Direct,
    /// Decreasing artificial nodal gmin at full sources (the ladder,
    /// `dynamic_gmin` or `spice3_gmin`).
    GminStepping,
    /// Decreasing *junction* gmin of every device at full sources (ngspice
    /// `new_gmin`, "true gmin stepping").
    JunctionGminStepping,
    /// Increasing source scale (the ladder with a temporary nodal gmin,
    /// `gillespie_src` or `spice3_src`).
    SourceStepping,
}

impl DcStrategy {
    /// Name used in diagnostics.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct Newton",
            Self::GminStepping => "gmin stepping",
            Self::JunctionGminStepping => "true gmin stepping",
            Self::SourceStepping => "source stepping",
        }
    }
}

/// One Newton solve (continuation stage).
#[derive(Debug, Clone, PartialEq)]
pub struct DcStage {
    /// The strategy this stage belongs to.
    pub strategy: DcStrategy,
    /// Source multiplier used (`1` is full sources).
    pub source_scale: Real,
    /// Artificial uniform nodal gmin (S) added to non-branch rows. This is
    /// *not* the device's junction gmin, which is always present.
    pub gmin: Real,
    /// The junction gmin (S) of every device in this stage when true gmin
    /// stepping replaced the configured one; `None` for the configured value.
    pub junction_gmin: Option<Real>,
    /// Newton iterations spent, charged against the total budget.
    pub iterations: usize,
    /// `None` if the stage converged, else the failure message.
    pub error: Option<String>,
}

impl DcStage {
    /// Whether the stage converged.
    #[must_use]
    pub const fn converged(&self) -> bool {
        self.error.is_none()
    }

    /// Whether this stage solved the *original* equations: full sources, zero
    /// artificial gmin and the configured junction gmin. Only such a stage's
    /// solution is ever returned.
    #[must_use]
    pub fn is_unregularized(&self) -> bool {
        self.source_scale.to_bits() == 1.0_f64.to_bits()
            && self.gmin == 0.
            && self.junction_gmin.is_none()
    }
}

/// Result of one strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// The strategy's final unregularized solve converged.
    Converged,
    /// A stage (or the final solve) failed.
    Failed,
    /// Reached but configured off (empty gmin schedule / no source stepping).
    Disabled,
}

/// One strategy attempt, in the order tried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DcAttempt {
    /// Which strategy.
    pub strategy: DcStrategy,
    /// How it ended.
    pub outcome: AttemptOutcome,
    /// For failures, the stage and error; empty otherwise.
    pub detail: String,
}

/// Final classification of a solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DcOutcome {
    /// Rejected before any Newton solve (invalid settings, seed or overrides).
    #[default]
    Rejected,
    /// Converged through the given strategy.
    Converged(DcStrategy),
    /// A structural, device or other non-numerical failure; never retried.
    NonRetryableFailure,
    /// Direct Newton and every enabled strategy failed numerically.
    Exhausted,
    /// The total iteration budget ran out first.
    BudgetExhausted,
}

/// Diagnostics for one DC solve, returned on success and failure.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DcReport {
    /// Strategies in the order they were reached.
    pub attempts: Vec<DcAttempt>,
    /// Every Newton solve performed, in order.
    pub stages: Vec<DcStage>,
    /// Newton iterations spent over all stages (`<= budget`).
    pub total_iterations: usize,
    /// Effective total iteration budget.
    pub budget: usize,
    /// How the solve ended.
    pub outcome: DcOutcome,
}

impl DcReport {
    /// One-line, human-readable account of the attempts and the work spent.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = self
            .attempts
            .iter()
            .map(|attempt| {
                let name = attempt.strategy.name();
                match attempt.outcome {
                    AttemptOutcome::Converged => format!("{name}: converged"),
                    AttemptOutcome::Failed => format!("{name}: failed ({})", attempt.detail),
                    AttemptOutcome::Disabled => format!("{name}: disabled"),
                }
            })
            .collect();
        parts.push(format!(
            "{} of {} budgeted Newton iterations used",
            self.total_iterations, self.budget
        ));
        parts.join("; ")
    }
}

/// A successful DC solve and how it was reached.
#[derive(Debug)]
pub struct DcSolution {
    /// The solution of the original (full-source, zero-artificial-gmin) system.
    pub solution: NewtonSolution<TrialState>,
    /// Attempts and stages that produced it.
    pub report: DcReport,
}

/// A failed DC solve with the diagnostics collected before it failed.
#[derive(Debug, Clone, PartialEq)]
pub struct DcFailure {
    /// The error. Device/structural failures are reported unchanged; an
    /// exhausted continuation is a [`SpiceError::Numerical`] naming every attempt.
    pub error: SpiceError,
    /// Attempts and stages before the failure (boxed to keep `Result` small).
    pub report: Box<DcReport>,
}

impl std::fmt::Display for DcFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for DcFailure {}

impl From<DcFailure> for SpiceError {
    fn from(failure: DcFailure) -> Self {
        failure.error
    }
}

/// Row kinds shared by DC and companion Newton iteration.
/// Rows Newton's global voltage-step damping watches, or `None` for every
/// non-branch row (the historical policy, kept whenever no device opts out).
///
/// When some nonlinear device does not want limiting
/// ([`crate::devices::Device::limits_voltage_steps`], behavioural sources),
/// only the node rows of the nonlinear devices that do (junctions) are
/// watched, so a large but exact step at a behavioural output is not damped
/// to 0.2 V per iteration.
pub(crate) fn limited_rows(circuit: &Circuit) -> Option<Vec<bool>> {
    let devices = circuit.devices();
    if !devices
        .iter()
        .any(|device| device.is_nonlinear() && !device.limits_voltage_steps())
    {
        return None;
    }
    let mut rows = vec![false; circuit.unknown_count()];
    for device in devices
        .iter()
        .filter(|device| device.is_nonlinear() && device.limits_voltage_steps())
    {
        for terminal in device.terminals() {
            if let Some(row) = circuit.unknowns().node_row(*terminal) {
                rows[row] = true;
            }
        }
    }
    Some(rows)
}

pub(crate) fn branch_rows(circuit: &Circuit) -> Vec<bool> {
    let mut kinds = vec![false; circuit.unknown_count()];
    for i in 0..circuit.device_count() {
        for row in circuit.branch_rows(i).unwrap_or(0..0) {
            kinds[row] = true;
        }
    }
    kinds
}

/// Constraints applied to node rows during a DC operating-point solve.
///
/// Node rows a DC solve forces in its loads, the exact form of `cktload.c`'s
/// `.nodeset`/`.ic` stamping: each listed row's equation is replaced by
/// `x[row] = value * scale`, where `scale` is the source-stepping factor
/// (C multiplies both by `CKTsrcFact`).
///
/// * [`Self::initial`]: `.ic` rows, forced in **every** load of the solve
///   (C: `MODETRANOP` without `MODEUIC`, the transient operating point).
/// * [`Self::nodesets`]: `.nodeset` rows, forced only in the
///   `MODEINITJCT`/`MODEINITFIX` loads ([`crate::devices::IterationPhase`]
///   `Junction`/`Fix`) and released afterwards, so they select a solution
///   without changing it. A row in both lists is forced to its `.ic` value.
///
/// A row that keeps a branch-current entry (a node on a voltage source,
/// inductor or E/H output) gets C's `1e10` conductance instead of the unit
/// diagonal (see `crate::analysis::initial`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NodeForcing {
    /// `(row, value)` pairs of the `.ic` constraints.
    pub initial: Vec<(usize, Real)>,
    /// `(row, value)` pairs of the `.nodeset` hints.
    pub nodesets: Vec<(usize, Real)>,
}

impl NodeForcing {
    /// No forced rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.initial.is_empty() && self.nodesets.is_empty()
    }

    fn validate(&self, n: usize) -> SpiceResult<()> {
        if self
            .initial
            .iter()
            .chain(&self.nodesets)
            .any(|(row, value)| *row >= n || !value.is_finite())
        {
            return Err(SpiceError::circuit("invalid forced DC node row"));
        }
        Ok(())
    }

    /// Replaces the forced rows of `a x = b` (see the type docs) as
    /// `cktload.c` does: `ZeroNoncurRow` zeroes the row's elements in place
    /// (they stay in the matrix structure, so its ordering is kept), then the
    /// diagonal is set to 1 and the RHS to the scaled value. `nodesets`
    /// selects whether the `.nodeset` rows are forced in this load; `branch`
    /// flags the branch-current rows, whose columns C leaves alone.
    fn apply(
        &self,
        a: &mut SparseMatrix,
        b: &mut Vector,
        scale: Real,
        nodesets: bool,
        branch: &[bool],
    ) -> SpiceResult<()> {
        let mut rows = std::collections::BTreeMap::new();
        if nodesets {
            rows.extend(self.nodesets.iter().copied());
        }
        rows.extend(self.initial.iter().copied());
        for (row, value) in rows {
            // A row that keeps a branch-current entry gets C's `1e10`
            // conductance (`= 1e10` for a nodeset, `+= 1e10` onto the zeroed
            // diagonal for an `.ic`: the same value).
            let currents = a.zero_row(row, |col| branch.get(col).copied().unwrap_or(false));
            a.set(row, row, if currents { 1e10 } else { 1. })?;
            b.as_mut_slice()[row] = if currents {
                1e10 * value * scale
            } else {
                value * scale
            };
        }
        Ok(())
    }
}

/// The node hints common to every ordinary operating-point solve. Transient
/// initialization deliberately has a separate path for enforced IC rows.
pub(super) struct OperatingPointStart {
    seed: Vector,
    nodes: NodeForcing,
}

impl OperatingPointStart {
    pub(super) fn resolve(
        circuit: &Circuit,
        request: &crate::analysis::AnalysisPlan,
    ) -> SpiceResult<Self> {
        let hints = crate::analysis::initial::resolve(circuit, request)?;
        let nodes = NodeForcing {
            initial: Vec::new(),
            nodesets: crate::analysis::initial::forced_nodesets(&hints.nodesets),
        };
        let mut seed = Vector::zeros(circuit.unknown_count());
        for hint in hints.nodesets {
            seed.as_mut_slice()[hint.row] = hint.value;
        }
        Ok(Self { seed, nodes })
    }

    pub(super) fn solve(
        &self,
        circuit: &Circuit,
        model: &ModelContext,
        settings: &DcSettings,
    ) -> SpiceResult<NewtonSolution<TrialState>> {
        self.solve_from(&circuit.prepare_run(), model, settings, &self.seed)
    }

    /// Another operating point in the same analysis: keep MODEINITJCT and
    /// MODEINITFIX node forcing, but warm-start from the previous solution.
    pub(super) fn solve_from(
        &self,
        prepared: &PreparedRun<'_>,
        model: &ModelContext,
        settings: &DcSettings,
        seed: &Vector,
    ) -> SpiceResult<NewtonSolution<TrialState>> {
        Ok(solve_dc_forced_prepared(
            prepared,
            model,
            settings,
            &[],
            Some(seed),
            None,
            &self.nodes,
        )?
        .solution)
    }
}

/// Solve a finalized circuit without committing device history or accept hooks,
/// using the default [`ContinuationPolicy`]. Compatible wrapper over
/// [`solve_dc_with`] that drops the report.
///
/// # Errors
/// Invalid inputs, device/structural failures, singular final system, or bounded
/// convergence failure. Temporary continuation solutions are never accepted.
pub fn solve_dc(
    circuit: &Circuit,
    context: &ModelContext,
    options: &NewtonOptions,
    overrides: &[(&str, f64)],
    initial: Option<&Vector>,
    forcing: Option<Forcing>,
) -> SpiceResult<NewtonSolution<TrialState>> {
    let settings = DcSettings {
        newton: *options,
        continuation: ContinuationPolicy::default(),
    };
    solve_dc_with(circuit, context, &settings, overrides, initial, forcing)
        .map(|solved| solved.solution)
        .map_err(SpiceError::from)
}

/// Solve a finalized circuit under explicit [`DcSettings`], reporting attempts.
///
/// Independent source overrides are typed, checked, and loaded as the sources'
/// DC values (`dctrcurv.c` sets `dcValue` and `dcGiven`); the circuit is not
/// modified. `forcing` makes the solve C's MODETRANOP bias: sources load their
/// time functions at 0 with its one-sided limit, scaled by `CKTsrcFact`.
/// Direct Newton runs first. Only a [`SpiceError::Numerical`] failure starts
/// continuation (gmin stepping, then source stepping, as enabled and within the
/// total iteration budget); any other error is returned at once, unretried.
/// Every success is a solve with **zero artificial nodal gmin and full source
/// values**, so continuation never changes the requested physical solution. The
/// device's own junction gmin is separate and always present.
///
/// # Errors
/// [`DcFailure`] carrying the error and report: invalid settings/inputs,
/// device/structural failures, or bounded convergence failure.
pub fn solve_dc_with(
    circuit: &Circuit,
    context: &ModelContext,
    settings: &DcSettings,
    overrides: &[(&str, f64)],
    initial: Option<&Vector>,
    forcing: Option<Forcing>,
) -> Result<DcSolution, DcFailure> {
    solve_dc_forced(
        circuit,
        context,
        settings,
        overrides,
        initial,
        forcing,
        &NodeForcing::default(),
    )
}

/// Run a DC solve with initial-condition and nodeset row constraints.
///
/// [`solve_dc_with`] with `.ic`/`.nodeset` row forcing ([`NodeForcing`]) in
/// every Newton load of every strategy. The returned point satisfies the
/// forced `.ic` rows and the physical equations of every other row; the
/// `.nodeset` rows are released before convergence.
///
/// # Errors
/// As [`solve_dc_with`], plus a forced row outside the system.
pub fn solve_dc_forced(
    circuit: &Circuit,
    context: &ModelContext,
    settings: &DcSettings,
    overrides: &[(&str, f64)],
    initial: Option<&Vector>,
    forcing: Option<Forcing>,
    nodes: &NodeForcing,
) -> Result<DcSolution, DcFailure> {
    solve_dc_forced_prepared(
        &circuit.prepare_run(),
        context,
        settings,
        overrides,
        initial,
        forcing,
        nodes,
    )
}

pub(super) fn solve_dc_forced_prepared(
    prepared: &PreparedRun<'_>,
    context: &ModelContext,
    settings: &DcSettings,
    overrides: &[(&str, f64)],
    initial: Option<&Vector>,
    forcing: Option<Forcing>,
    nodes: &NodeForcing,
) -> Result<DcSolution, DcFailure> {
    let history = prepared
        .circuit()
        .state_history()
        .map_err(|error| DcFailure {
            error,
            report: Box::new(DcReport::default()),
        })?;
    solve(
        prepared,
        context,
        settings,
        overrides,
        &Start {
            initial,
            forcing,
            history: &history,
            policy: PhasePolicy::OperatingPoint,
            nodes,
        },
    )
}

/// Continue a DC solve from an accepted previous state.
///
/// [`solve_dc_forced`] continuing an accepted state `history` (C `CKTstate1..`),
/// starting the direct Newton attempt in `policy`'s phase: a DC sweep point
/// after the first runs [`PhasePolicy::Predicted`] (`dctrcurv.c` sets
/// `MODEINITPRED`). Gmin/source-stepping strategies always restart in
/// [`PhasePolicy::OperatingPoint`] against the same history (`CKTop` with
/// `MODEINITJCT`; the rotated states are not cleared).
/// Only devices with discrete state (switches) read the history; nothing is
/// committed to it.
///
/// # Errors
/// As [`solve_dc_with`], or a history that does not match the circuit.
#[expect(
    clippy::too_many_arguments,
    reason = "keeps the positional interface explicit at the reference solver or device boundary"
)]
pub fn solve_dc_from(
    prepared: &PreparedRun<'_>,
    context: &ModelContext,
    settings: &DcSettings,
    overrides: &[(&str, f64)],
    initial: Option<&Vector>,
    history: &StateHistory,
    policy: PhasePolicy,
    nodes: &NodeForcing,
) -> Result<DcSolution, DcFailure> {
    solve(
        prepared,
        context,
        settings,
        overrides,
        &Start {
            initial,
            forcing: None,
            history,
            policy,
            nodes,
        },
    )
}

/// Where a DC solve starts: seed, transient-bias forcing, accepted history and
/// the direct attempt's Newton phase policy.
struct Start<'a> {
    initial: Option<&'a Vector>,
    forcing: Option<Forcing>,
    history: &'a StateHistory,
    policy: PhasePolicy,
    nodes: &'a NodeForcing,
}

fn solve(
    prepared: &PreparedRun<'_>,
    context: &ModelContext,
    settings: &DcSettings,
    overrides: &[(&str, f64)],
    start: &Start<'_>,
) -> Result<DcSolution, DcFailure> {
    let mut report = DcReport::default();
    match run(prepared, context, settings, overrides, start, &mut report) {
        Ok(solution) => Ok(DcSolution { solution, report }),
        Err(error) => Err(DcFailure {
            error,
            report: Box::new(report),
        }),
    }
}

/// A stage failure, classified for the strategy driver.
enum Halt {
    /// Numerical failure: the next strategy may still succeed.
    Retry(String),
    /// Anything else (structural, device, unsupported): never masked by retries.
    Fatal(SpiceError),
    /// The total iteration budget ended the attempt.
    Budget(String),
}

/// How one continuation stage modifies the original equations.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Point {
    /// Source multiplier (C `CKTsrcFact`).
    scale: Real,
    /// Artificial nodal gmin on non-branch rows (C `CKTdiagGmin`).
    gmin: Real,
    /// Replacement junction gmin of every device (C `CKTgmin` in `new_gmin`).
    junction: Option<Real>,
}

impl Point {
    /// Full sources, no artificial gmin, the configured junction gmin.
    const ORIGINAL: Self = Self::nodal(1., 0.);

    const fn nodal(scale: Real, gmin: Real) -> Self {
        Self {
            scale,
            gmin,
            junction: None,
        }
    }
}

struct Engine<'a> {
    prepared: &'a PreparedRun<'a>,
    context: &'a ModelContext,
    history: &'a StateHistory,
    policy: PhasePolicy,
    branches: Vec<bool>,
    /// Rows watched by Newton's voltage-step damping ([`limited_rows`]).
    limited: Option<Vec<bool>>,
    /// MODETRANOP source forcing ([`Start::forcing`]).
    forcing: Option<Forcing>,
    nodes: &'a NodeForcing,
    newton: NewtonOptions,
    /// Per-stage limit of the gmin/source-stepping strategies.
    stage_limit: usize,
    remaining: usize,
}

#[derive(Clone, Copy)]
enum GminKind {
    Nodal,
    Junction,
}

impl GminKind {
    const fn strategy(self) -> DcStrategy {
        match self {
            Self::Nodal => DcStrategy::GminStepping,
            Self::Junction => DcStrategy::JunctionGminStepping,
        }
    }

    const fn point(self, gmin: Real) -> Point {
        match self {
            Self::Nodal => Point::nodal(1., gmin),
            Self::Junction => Point {
                scale: 1.,
                gmin: 0.,
                junction: Some(gmin),
            },
        }
    }
}

/// A solve identifies both its equation modifications and its iteration policy.
/// The original-equation closing stages cannot accidentally retain a gmin.
#[derive(Clone, Copy)]
enum Stage {
    Direct,
    Gmin(GminKind, Real),
    CloseGmin(GminKind),
    Source { scale: Real, gmin: Real },
    CloseSource,
}

impl Stage {
    const fn strategy(self) -> DcStrategy {
        match self {
            Self::Direct => DcStrategy::Direct,
            Self::Gmin(kind, _) | Self::CloseGmin(kind) => kind.strategy(),
            Self::Source { .. } | Self::CloseSource => DcStrategy::SourceStepping,
        }
    }

    const fn point(self) -> Point {
        match self {
            Self::Direct | Self::CloseGmin(_) | Self::CloseSource => Point::ORIGINAL,
            Self::Gmin(kind, gmin) => kind.point(gmin),
            Self::Source { scale, gmin } => Point::nodal(scale, gmin),
        }
    }

    const fn uses_direct_limit(self) -> bool {
        matches!(self, Self::Direct | Self::CloseGmin(_))
    }
}

/// Each executable strategy owns its seed, stage data and closing semantics.
/// Only the disabled entries carry a reporting-only strategy tag.
enum Plan<'p> {
    Disabled(DcStrategy),
    Direct(&'p Vector),
    FixedGmin {
        start: &'p Vector,
        ladder: Vec<Real>,
    },
    FixedSource {
        start: &'p Vector,
        scales: Vec<Real>,
        gmin: Real,
    },
    Dynamic {
        kind: GminKind,
        factor: Real,
        adapt: IterationLimit,
    },
    Gillespie(IterationLimit),
}

impl Plan<'_> {
    const fn strategy(&self) -> DcStrategy {
        match self {
            Self::Disabled(strategy) => *strategy,
            Self::Direct(_) => DcStrategy::Direct,
            Self::FixedGmin { .. } => DcStrategy::GminStepping,
            Self::Dynamic { kind, .. } => kind.strategy(),
            Self::FixedSource { .. } | Self::Gillespie(_) => DcStrategy::SourceStepping,
        }
    }

    /// Upper bound on the number of solves at ITL1 and ITL2 respectively.
    const fn solve_counts(&self) -> (usize, usize) {
        match self {
            Self::Disabled(_) => (0, 0),
            Self::Direct(_) => (1, 0),
            Self::FixedGmin { ladder, .. } => (1, ladder.len()),
            Self::FixedSource { scales, .. } => (0, scales.len() + 1),
            Self::Dynamic { .. } => (1, MAX_ADAPTIVE_STAGES),
            Self::Gillespie(_) => (0, 1 + GILLESPIE_GMIN_DECADES + 1 + MAX_ADAPTIVE_STAGES),
        }
    }
}

impl ContinuationPolicy {
    /// Resolve the validated schedule into ordered attempts.
    fn plan<'a>(
        &self,
        seed: &'a Vector,
        zero: &'a Vector,
        gmin: Real,
    ) -> SpiceResult<Vec<Plan<'a>>> {
        let mut plan = vec![if self.skip_direct {
            Plan::Disabled(DcStrategy::Direct)
        } else {
            Plan::Direct(seed)
        }];
        match &self.schedule {
            ContinuationSchedule::Ladder { gmin, source } => {
                plan.push(if gmin.as_slice().is_empty() {
                    Plan::Disabled(DcStrategy::GminStepping)
                } else {
                    Plan::FixedGmin {
                        start: seed,
                        ladder: gmin.as_slice().to_vec(),
                    }
                });
                plan.push(source.as_ref().map_or(
                    Plan::Disabled(DcStrategy::SourceStepping),
                    |source| Plan::FixedSource {
                        start: zero,
                        scales: source.scales.clone(),
                        gmin: source.gmin,
                    },
                ));
            }
            ContinuationSchedule::Ngspice(stepping) => {
                match stepping.gmin {
                    Stepping::Disabled => plan.push(Plan::Disabled(DcStrategy::GminStepping)),
                    Stepping::Adaptive => {
                        for kind in [GminKind::Nodal, GminKind::Junction] {
                            plan.push(Plan::Dynamic {
                                kind,
                                factor: stepping.gmin_factor,
                                adapt: stepping.adapt_iterations,
                            });
                        }
                    }
                    Stepping::Fixed(steps) => plan.push(Plan::FixedGmin {
                        start: seed,
                        ladder: stepping.spice3_ladder(gmin, steps)?,
                    }),
                }
                plan.push(match stepping.source {
                    Stepping::Disabled => Plan::Disabled(DcStrategy::SourceStepping),
                    Stepping::Adaptive => Plan::Gillespie(stepping.adapt_iterations),
                    Stepping::Fixed(steps) => {
                        let count = steps.get();
                        let steps = u16::try_from(count)
                            .map_err(|_| invalid("source step count exceeds u16"))?;
                        // The last Spice3 scale is the explicit original solve,
                        // unlike the custom ladder which has an extra solve.
                        Plan::FixedSource {
                            start: zero,
                            scales: uniform_scales(steps).take(count).collect(),
                            gmin: 0.,
                        }
                    }
                });
            }
        }
        Ok(plan)
    }
}

/// `NIiter` may take its configured limit plus the failing extra pass.
fn iteration_budget(plan: &[Plan<'_>], direct: usize, stage: usize) -> usize {
    plan.iter().fold(0_usize, |total, planned| {
        let (direct_count, stage_count) = planned.solve_counts();
        total
            .saturating_add(direct_count.saturating_mul(direct.saturating_add(1)))
            .saturating_add(stage_count.saturating_mul(stage.saturating_add(1)))
    })
}

impl Engine<'_> {
    /// One disposable Newton solve of the typed stage; charges the budget.
    /// `continued` is the preceding stage's converged trial, if any.
    fn stage(
        &mut self,
        report: &mut DcReport,
        stage: Stage,
        guess: &Vector,
        continued: Option<TrialState>,
    ) -> Result<NewtonSolution<TrialState>, Halt> {
        let Point {
            scale,
            gmin,
            junction,
        } = stage.point();
        let strategy = stage.strategy();
        let describe = || {
            junction.map_or_else(
                || format!("scale {scale}, gmin {gmin:e} S"),
                |junction| format!("scale {scale}, junction gmin {junction:e} S"),
            )
        };
        // A Newton solve needs two passes at least (the first never
        // converges), and a failing one uses `max_iterations + 1` (NIiter
        // fails after the solve that exceeds the limit).
        if self.remaining < 2 {
            return Err(Halt::Budget(format!(
                "total iteration budget ({}) exhausted before {}",
                report.budget,
                describe()
            )));
        }
        // C bounds the direct solve and the gmin strategies' closing solve by
        // `iterlim` (itl1) and every other continuation solve by itl2
        // (`cktop.c`: `spice3_gmin`/`dynamic_gmin`/`new_gmin` end with
        // `NIiter(ckt, iterlim)`; `gillespie_src`/`spice3_src` ignore it).
        let limit = if stage.uses_direct_limit() {
            self.newton.max_iterations
        } else {
            self.stage_limit
        };
        let options = NewtonOptions {
            max_iterations: limit.min(self.remaining - 1),
            // cktop.c's gmin stepping sets CKTdiagGmin, which SMPreorder and
            // SMPluFac add to SPARSE's diagonal (LoadGmin).
            diag_gmin: gmin,
            had_nodeset: !self.nodes.nodesets.is_empty(),
            ..self.newton
        };
        let reduced = options.max_iterations < limit;
        // Sources load their value scaled by CKTsrcFact (vsrcload.c).
        let context = junction
            .map_or(*self.context, |junction| self.context.with_gmin(junction))
            .with_source_factor(scale);
        let result = CircuitNewton {
            prepared: self.prepared,
            context: &context,
            history: self.history,
            branch_rows: &self.branches,
            limited_rows: self.limited.as_deref(),
            options: &options,
            load: CircuitLoad::Bias {
                forcing: self.forcing,
            },
        }
        .solve(
            guess,
            // Continuation strategies restart like CKTop; only the direct
            // attempt may continue a predicted point.
            if strategy == DcStrategy::Direct {
                self.policy
            } else {
                PhasePolicy::OperatingPoint
            },
            continued,
            |phase, matrix, rhs| {
                // cktload.c: nodeset rows only in MODEINITJCT/MODEINITFIX;
                // .ic rows in every load, both scaled by CKTsrcFact.
                let hinted = matches!(phase, IterationPhase::Junction | IterationPhase::Fix);
                self.nodes.apply(matrix, rhs, scale, hinted, &self.branches)
            },
        );
        let (iterations, error) = match &result {
            Ok(solved) => (solved.iterations, None),
            Err(NewtonFailure {
                error, iterations, ..
            }) => (*iterations, Some(error)),
        };
        self.remaining = self.remaining.saturating_sub(iterations);
        report.total_iterations += iterations;
        report.stages.push(DcStage {
            strategy,
            source_scale: scale,
            gmin,
            junction_gmin: junction,
            iterations,
            error: error.map(ToString::to_string),
        });
        // A failure that used the whole budget-reduced allowance may only have
        // been cut short by the budget, so it ends the attempt.
        let cut_short = reduced && iterations >= options.max_iterations;
        match result {
            Ok(solved) => Ok(solved),
            Err(NewtonFailure {
                error: error @ SpiceError::Numerical { .. },
                ..
            }) => {
                let detail = format!("{}: {error}", describe());
                Err(if cut_short {
                    Halt::Budget(format!("{detail} (iteration limit reduced by the budget)"))
                } else {
                    Halt::Retry(detail)
                })
            }
            Err(NewtonFailure { error, .. }) => Err(Halt::Fatal(error)),
        }
    }

    /// Walk fixed stages and an explicit original-equation closing stage.
    fn walk(
        &mut self,
        report: &mut DcReport,
        start: &Vector,
        stages: impl Iterator<Item = Stage>,
        closing: Stage,
    ) -> Result<NewtonSolution<TrialState>, Halt> {
        let mut guess = start.clone();
        let mut continued = None;
        for stage in stages {
            let solved = self.stage(report, stage, &guess, continued)?;
            guess = solved.values;
            continued = Some(solved.trial);
        }
        self.stage(report, closing, &guess, continued)
    }

    /// A stage continuing `from` (a converged stage), or restarting from the
    /// zero vector in `MODEINITJCT` without one (C zeroes `CKTrhsOld` and
    /// `CKTstate0` before its adaptive strategies).
    fn step(
        &mut self,
        report: &mut DcReport,
        from: Option<&NewtonSolution<TrialState>>,
        stage: Stage,
    ) -> Result<NewtonSolution<TrialState>, Halt> {
        if let Some(from) = from {
            self.stage(report, stage, &from.values, Some(from.trial.clone()))
        } else {
            let zero = Vector::zeros(self.prepared.circuit().unknown_count());
            self.stage(report, stage, &zero, None)
        }
    }

    /// `dynamic_gmin` (artificial nodal gmin) or, with `junction`, `new_gmin`
    /// (the devices' junction gmin), both from `1e-2 / gminfactor` S down to
    /// the configured junction gmin with C's adaptive factor, then the
    /// original equations at the direct limit.
    fn dynamic_gmin(
        &mut self,
        report: &mut DcReport,
        kind: GminKind,
        initial_factor: Real,
        adapt: IterationLimit,
    ) -> Result<NewtonSolution<TrialState>, Halt> {
        let junction = matches!(kind, GminKind::Junction);
        // gtarget = MAX(CKTgmin, CKTgshunt); gshunt is not ported (0).
        let target = self.context.gmin;
        let slow_floor = if junction {
            TRUE_GMIN_SLOW_FACTOR
        } else {
            DYNAMIC_FACTOR_FLOOR
        };
        let (quick, slow) = (adapt.get() / 4, 3 * adapt.get() / 4);
        let mut factor = initial_factor;
        let mut previous = DYNAMIC_GMIN_START;
        let mut gmin = previous / factor;
        let mut saved: Option<NewtonSolution<TrialState>> = None;
        for _ in 0..MAX_ADAPTIVE_STAGES {
            match self.step(report, saved.as_ref(), Stage::Gmin(kind, gmin)) {
                Ok(solved) => {
                    let iterations = solved.iterations;
                    saved = Some(solved);
                    if gmin <= target {
                        let last = saved.as_ref();
                        return self.step(report, last, Stage::CloseGmin(kind));
                    }
                    if iterations <= quick {
                        factor = (factor * factor.sqrt()).min(initial_factor);
                    }
                    if iterations > slow {
                        factor = factor.sqrt().max(slow_floor);
                    }
                    previous = gmin;
                    if gmin < factor * target {
                        factor = gmin / target;
                        gmin = target;
                    } else {
                        gmin /= factor;
                    }
                }
                Err(Halt::Retry(detail)) => {
                    if factor < DYNAMIC_FACTOR_FLOOR {
                        return Err(Halt::Retry(format!("last step failed ({detail})")));
                    }
                    factor = factor.sqrt().sqrt();
                    gmin = previous / factor;
                }
                Err(halt) => return Err(halt),
            }
        }
        Err(Halt::Retry(format!(
            "stopped after {MAX_ADAPTIVE_STAGES} adaptive stages at {} {gmin:e} S",
            if junction { "junction gmin" } else { "gmin" }
        )))
    }

    /// `gillespie_src`: the circuit with every source off (with a ten-decade
    /// artificial gmin ladder from `gmin * 1e10` if that fails), then C's
    /// adaptive source factor up to full sources.
    fn gillespie(
        &mut self,
        report: &mut DcReport,
        adapt: IterationLimit,
    ) -> Result<NewtonSolution<TrialState>, Halt> {
        let mut saved = match self.step(
            report,
            None,
            Stage::Source {
                scale: 0.,
                gmin: 0.,
            },
        ) {
            Ok(solved) => solved,
            Err(Halt::Retry(_)) => {
                // diagGmin = (gshunt <= 0 ? gmin : gshunt) * 10^10, then / 10.
                let mut gmin = self.context.gmin;
                for _ in 0..GILLESPIE_GMIN_DECADES {
                    gmin *= 10.;
                }
                let mut last = None;
                for _ in 0..=GILLESPIE_GMIN_DECADES {
                    last = Some(self.step(
                        report,
                        last.as_ref(),
                        Stage::Source { scale: 0., gmin },
                    )?);
                    gmin /= 10.;
                }
                last.ok_or_else(|| Halt::Retry("empty zero-source gmin ladder".into()))?
            }
            Err(halt) => return Err(halt),
        };
        let (quick, slow) = (adapt.get() / 4, 3 * adapt.get() / 4);
        let mut converged: Real = 0.;
        let mut raise = GILLESPIE_FIRST_RAISE;
        let mut scale = converged + raise;
        for _ in 0..MAX_ADAPTIVE_STAGES {
            match self.step(
                report,
                Some(&saved),
                if scale.to_bits() == 1.0_f64.to_bits() {
                    Stage::CloseSource
                } else {
                    Stage::Source { scale, gmin: 0. }
                },
            ) {
                Ok(solved) => {
                    let iterations = solved.iterations;
                    converged = scale;
                    saved = solved;
                    scale = converged + raise;
                    if iterations <= quick {
                        raise *= 1.5;
                    }
                    if iterations > slow {
                        raise *= 0.5;
                    }
                }
                Err(Halt::Retry(detail)) => {
                    if scale - converged < GILLESPIE_MIN_PROGRESS {
                        return Err(Halt::Retry(format!(
                            "stalled at source scale {converged} ({detail})"
                        )));
                    }
                    raise = (raise / 10.).min(GILLESPIE_MAX_RAISE);
                    // C retries the last converged factor before raising it again.
                    scale = converged;
                }
                Err(halt) => return Err(halt),
            }
            scale = scale.min(1.);
            if converged >= 1. {
                return Ok(saved);
            }
            if raise < GILLESPIE_MIN_RAISE {
                return Err(Halt::Retry(format!(
                    "source increment fell below {GILLESPIE_MIN_RAISE:e} at scale {converged}"
                )));
            }
        }
        Err(Halt::Retry(format!(
            "stopped after {MAX_ADAPTIVE_STAGES} adaptive stages at source scale {converged}"
        )))
    }

    /// Run one planned strategy and record its entry in the attempt list.
    fn attempt(
        &mut self,
        report: &mut DcReport,
        plan: &Plan<'_>,
    ) -> Result<NewtonSolution<TrialState>, Halt> {
        let strategy = plan.strategy();
        let result = match plan {
            Plan::Disabled(_) => Err(Halt::Retry(String::new())),
            Plan::Direct(seed) => self.stage(report, Stage::Direct, seed, None),
            Plan::FixedGmin { start, ladder } => self.walk(
                report,
                start,
                ladder
                    .iter()
                    .map(|gmin| Stage::Gmin(GminKind::Nodal, *gmin)),
                Stage::CloseGmin(GminKind::Nodal),
            ),
            Plan::FixedSource {
                start,
                scales,
                gmin,
            } => self.walk(
                report,
                start,
                scales.iter().map(|scale| Stage::Source {
                    scale: *scale,
                    gmin: *gmin,
                }),
                Stage::CloseSource,
            ),
            Plan::Dynamic {
                kind,
                factor,
                adapt,
            } => self.dynamic_gmin(report, *kind, *factor, *adapt),
            Plan::Gillespie(adapt) => self.gillespie(report, *adapt),
        };
        let (outcome, detail) = match &result {
            Ok(_) => (AttemptOutcome::Converged, String::new()),
            Err(_) if matches!(plan, Plan::Disabled(_)) => {
                (AttemptOutcome::Disabled, String::new())
            }
            Err(Halt::Retry(detail) | Halt::Budget(detail)) => {
                (AttemptOutcome::Failed, detail.clone())
            }
            Err(Halt::Fatal(error)) => (AttemptOutcome::Failed, error.to_string()),
        };
        report.attempts.push(DcAttempt {
            strategy,
            outcome,
            detail,
        });
        result
    }
}

fn run(
    prepared: &PreparedRun<'_>,
    context: &ModelContext,
    settings: &DcSettings,
    overrides: &[(&str, f64)],
    start: &Start<'_>,
    report: &mut DcReport,
) -> SpiceResult<NewtonSolution<TrialState>> {
    let circuit = prepared.circuit();
    let Start {
        initial,
        forcing,
        history,
        policy: phases,
        nodes,
    } = *start;
    settings.validate()?;
    nodes.validate(circuit.unknown_count())?;
    if !settings.newton.limiting.is_device()
        && let Some(device) = circuit.devices().iter().find(|d| d.has_start_settings())
    {
        return Err(SpiceError::Unsupported {
            feature: format!(
                "{}: `off`/MOS1 `ic=` start voltages need ngspice's device limiting \
                 (limiting=device, the default), not limiting=global",
                device.name()
            ),
            location: None,
        });
    }
    if history.len() != circuit.state_len() {
        return Err(SpiceError::circuit(
            "DC state history does not match the circuit numbering",
        ));
    }
    let options = &settings.newton;
    let policy = &settings.continuation;
    let n = circuit.unknown_count();
    if initial.is_some_and(|x| x.len() != n || !x.is_finite()) {
        return Err(SpiceError::circuit("invalid DC initial solution"));
    }
    if forcing.is_some() && !overrides.is_empty() {
        return Err(SpiceError::circuit(
            "combine neither source overrides nor transient-bias forcing",
        ));
    }
    let zero = Vector::zeros(n);
    let mut loaded = *context;
    for (name, value) in overrides {
        loaded = loaded.with_instance_override(circuit.source_override(name, *value, context)?)?;
    }
    let context = &loaded;
    // Linear circuits take the same CKTop/NIiter path as nonlinear ones, as
    // in ngspice: the same iterations, pivot order and continuation fallbacks.
    let seed = initial.unwrap_or(&zero);
    let stage_limit = policy
        .stage_max_iterations
        .map_or(options.max_iterations, IterationLimit::get);
    let plan = policy.plan(seed, &zero, context.gmin)?;
    let budget = policy.max_total_iterations.map_or_else(
        || iteration_budget(&plan, options.max_iterations, stage_limit),
        IterationBudget::get,
    );
    report.budget = budget;
    let mut engine = Engine {
        prepared,
        context,
        history,
        policy: phases,
        branches: branch_rows(circuit),
        limited: limited_rows(circuit),
        forcing,
        nodes,
        newton: *options,
        stage_limit,
        remaining: budget,
    };
    for planned in &plan {
        match engine.attempt(report, planned) {
            Ok(solved) => {
                report.outcome = DcOutcome::Converged(planned.strategy());
                return Ok(solved);
            }
            Err(Halt::Retry(_)) => {}
            Err(Halt::Fatal(error)) => {
                report.outcome = DcOutcome::NonRetryableFailure;
                return Err(error);
            }
            Err(Halt::Budget(_)) => {
                report.outcome = DcOutcome::BudgetExhausted;
                return Err(SpiceError::Numerical {
                    context: "DC continuation".into(),
                    message: format!(
                        "total iteration budget exhausted: {}; raise maxiter or \
                         max_total_iterations, or simplify the circuit",
                        report.summary()
                    ),
                });
            }
        }
    }
    report.outcome = DcOutcome::Exhausted;
    Err(SpiceError::Numerical {
        context: "DC continuation".into(),
        message: format!(
            "direct Newton and every enabled continuation strategy failed: {}; \
             {OPTRAN_NOT_PORTED}",
            report.summary()
        ),
    })
}

/// Appended to an exhausted continuation's error: ngspice's `CKTop` would
/// next run its transient operating-point fallback, which is not ported.
pub const OPTRAN_NOT_PORTED: &str = "ngspice's next fallback, the transient operating point \
     (src/spicelib/analysis/optran.c, OPtran), is not ported";
