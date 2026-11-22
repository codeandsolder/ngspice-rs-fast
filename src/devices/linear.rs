//! Immutable linear MNA assembly, distinct from timestep-dependent companions.
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::devices::MnaUnknowns;
use crate::devices::functions::{FunctionSpec, PwlSource, SourceFunction};
use crate::devices::pulse::{Pulse, PulseSpec, TransientTiming};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};

/// Which one-sided limit to take where a waveform has a jump.
///
/// Away from jumps both limits agree. Transient drivers evaluate the forcing at
/// the *end* of a segment with [`Limit::Left`] and at the *start* of the next
/// segment with [`Limit::Right`]; they never interpolate across a jump.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Limit {
    /// The limit approaching `t` from earlier times.
    Left,
    /// The limit approaching `t` from later times (the value just after `t`).
    Right,
}

/// How a companion transient load evaluates independent-source forcing.
///
/// A load at `AnalysisMode::Transient { time, .. }` evaluates every source at
/// `time` with `limit`: [`Limit::Left`] for the step that *ends* at a source
/// breakpoint, [`Limit::Right`] for a step that starts at or after one (the
/// step's end time is then past the breakpoint, so either limit agrees with the
/// strictly later value). `timing` supplies C's PULSE defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Forcing {
    /// Which one-sided limit to take at a jump.
    pub limit: Limit,
    /// The `.tran` quantities that resolve PULSE defaults.
    pub timing: TransientTiming,
}

/// A bounded source waveform. Breakpoints are explicit and values finite.
///
/// Time forcing is evaluated with [`Waveform::value_at`] and its corners and
/// jumps are enumerated lazily with [`Waveform::breakpoints_in`]; DC and AC
/// excitations are separate fields of a source and never derived from this.
#[derive(Debug, Clone, PartialEq)]
pub enum Waveform {
    /// Time-independent forcing.
    Constant(Real),
    /// RFSPICE PORT accumulation in `VSRCload` order. Tones are (amplitude, radians/s).
    PortSeries {
        /// Preceding non-PORT voltage source time function.
        baseline: Box<Self>,
        /// Accumulated cosine excitations, flattened to bound recursion.
        tones: Vec<(Real, Real)>,
    },
    /// Right-continuous jump at a nonnegative time.
    Step {
        /// Value before the jump.
        before: Real,
        /// Value after the jump.
        after: Real,
        /// Jump time.
        time: Real,
    },
    /// Validated PWL knots, with optional C delay and repetition.
    Pwl(PwlSource),
    /// A fully specified periodic pulse; see [`Pulse`].
    Pulse(Pulse),
    /// A PULSE as written, with C's analysis-dependent defaults still pending.
    /// [`LinearSystem::bind_transient_timing`] (or [`Waveform::resolve`]) turns
    /// it into [`Waveform::Pulse`]; evaluating it earlier is an error.
    PulseDefaults(PulseSpec),
    /// A fully specified SIN/EXP/SFFM/AM; see [`SourceFunction`].
    Function(SourceFunction),
    /// SIN/EXP/SFFM/AM as written, with C's analysis-dependent defaults
    /// pending; resolved like [`Waveform::PulseDefaults`].
    FunctionDefaults(FunctionSpec),
}
impl Waveform {
    /// Checks finite values and strictly increasing nonnegative knot times.
    /// # Errors
    /// Invalid source data.
    pub fn validate(&self) -> SpiceResult<()> {
        let valid = match self {
            Self::Constant(v) => v.is_finite(),
            Self::PortSeries { baseline, tones } => {
                baseline.validate()?;
                tones.iter().all(|(a, w)| a.is_finite() && w.is_finite())
            }
            Self::Step {
                before,
                after,
                time,
            } => before.is_finite() && after.is_finite() && time.is_finite() && *time >= 0.,
            Self::PulseDefaults(spec) => {
                spec.validate()?;
                true
            }
            // Pulse and PWL/function values are only constructed through
            // validating constructors, so they need no further checks here.
            Self::Pulse(_) | Self::Pwl(_) | Self::Function(_) => true,
            Self::FunctionDefaults(spec) => {
                spec.validate()?;
                true
            }
        };
        if !valid {
            return Err(SpiceError::circuit("invalid source waveform"));
        }
        Ok(())
    }

    /// Resolves analysis-dependent defaults; other variants are returned as is.
    ///
    /// # Errors
    /// The resolved pulse is invalid.
    pub fn resolve(&self, timing: &TransientTiming) -> SpiceResult<Self> {
        match self {
            Self::PortSeries { baseline, tones } => Ok(Self::PortSeries {
                baseline: Box::new(baseline.resolve(timing)?),
                tones: tones.clone(),
            }),
            Self::PulseDefaults(spec) => Ok(Self::Pulse(spec.resolve(timing)?)),
            Self::FunctionDefaults(spec) => Ok(Self::Function(spec.resolve(timing)?)),
            other => Ok(other.clone()),
        }
    }

    /// True when the forcing is linear between consecutive
    /// [`Waveform::breakpoints_in`] times (constant, step, PWL and PULSE
    /// forms), so a segment's end values determine it exactly. SIN/EXP/SFFM/AM
    /// are not.
    #[must_use]
    pub const fn is_piecewise_linear(&self) -> bool {
        !matches!(
            self,
            Self::Function(_) | Self::FunctionDefaults(_) | Self::PortSeries { .. }
        )
    }

    /// The value C loads in OP/DC analyses when the source has no explicit DC
    /// value: the transient function at `time = 0` (`vsrcload.c`, `MODEDC`),
    /// which is the left limit at zero. Needs no analysis timing.
    ///
    /// # Errors
    /// The waveform is invalid.
    pub fn time_zero(&self) -> SpiceResult<Real> {
        match self {
            Self::PortSeries { baseline, tones } => {
                Ok(baseline.time_zero()? + tones.iter().map(|(a, _)| a).sum::<Real>())
            }
            Self::PulseDefaults(spec) => {
                self.validate()?;
                Ok(spec.initial)
            }
            Self::FunctionDefaults(spec) => spec.time_zero(),
            other => other.value_at(0., Limit::Left),
        }
    }

    /// Time forcing at `t` with an explicit one-sided limit.
    ///
    /// This is the evaluation API for transient drivers. `limit` only matters at
    /// a jump (a [`Waveform::Step`] time, a zero-duration pulse edge, or a pulse
    /// period boundary that cuts a ramp); elsewhere both limits are equal.
    ///
    /// # Errors
    /// Nonfinite `t`, unresolved [`Waveform::PulseDefaults`], or a time too many
    /// periods from the origin to resolve.
    pub fn value_at(&self, t: Real, limit: Limit) -> SpiceResult<Real> {
        if !t.is_finite() {
            return Err(SpiceError::circuit("nonfinite waveform evaluation time"));
        }
        let left = limit == Limit::Left;
        Ok(match self {
            Self::Constant(v) => *v,
            Self::PortSeries { baseline, tones } => {
                baseline.value_at(t, limit)?
                    + tones.iter().map(|(a, w)| a * (w * t).cos()).sum::<Real>()
            }
            Self::Step {
                before,
                after,
                time,
            } => {
                let before_jump = if left { t <= *time } else { t < *time };
                if before_jump { *before } else { *after }
            }
            Self::Pwl(pwl) => return pwl.value_at(t, limit),
            Self::Pulse(pulse) => return pulse.value_at(t, limit),
            Self::Function(function) => return function.value_at(t, limit),
            Self::PulseDefaults(_) => {
                return Err(SpiceError::circuit(
                    "PULSE defaults (TR/TF/PW/PER) are unresolved; bind transient timing first",
                ));
            }
            Self::FunctionDefaults(_) => {
                return Err(SpiceError::circuit(
                    "SIN/EXP/SFFM/AM defaults are unresolved; bind transient timing first",
                ));
            }
        })
    }

    /// Like [`Self::value_at`], but resolves a [`Waveform::PulseDefaults`]
    /// against `timing` first, so unbound source devices can be loaded.
    ///
    /// # Errors
    /// As [`Self::value_at`], or the pulse is invalid after resolution.
    pub fn value_at_timed(
        &self,
        t: Real,
        limit: Limit,
        timing: &TransientTiming,
    ) -> SpiceResult<Real> {
        match self {
            Self::PortSeries { baseline, tones } => Ok(baseline
                .value_at_timed(t, limit, timing)?
                + tones.iter().map(|(a, w)| a * (w * t).cos()).sum::<Real>()),
            Self::PulseDefaults(spec) => spec.resolve(timing)?.value_at(t, limit),
            Self::FunctionDefaults(spec) => spec.resolve(timing)?.value_at(t, limit),
            other => other.value_at(t, limit),
        }
    }

    /// Value at time t; left limit is used at a segment's terminating jump.
    /// Returns NaN where [`Waveform::value_at`] would fail; prefer `value_at`.
    #[must_use]
    pub fn value(&self, t: Real, left_limit: bool) -> Real {
        let limit = if left_limit {
            Limit::Left
        } else {
            Limit::Right
        };
        self.value_at(t, limit).unwrap_or(Real::NAN)
    }

    /// Lazily enumerates corner and jump times in the closed window `[t0, t1]`,
    /// ascending without repeats.
    ///
    /// Periodic pulses are generated one cycle at a time and never expanded;
    /// callers bound a run by taking only what they need. Constant waveforms
    /// have none; Step has its jump; Pwl has its knots.
    ///
    /// # Errors
    /// Nonfinite or reversed window, unresolved [`Waveform::PulseDefaults`], or a
    /// window spanning more periods than can be resolved.
    pub fn breakpoints_in(&self, t0: Real, t1: Real) -> SpiceResult<WaveformBreakpoints> {
        if !(t0.is_finite() && t1.is_finite() && t0 <= t1) {
            return Err(SpiceError::circuit(
                "breakpoint window must be finite with t0 <= t1",
            ));
        }
        let (fixed, kind) = match self {
            Self::Constant(_) => (vec![], WaveformEventKind::FunctionCorner),
            Self::PortSeries { baseline, .. } => return baseline.breakpoints_in(t0, t1),
            Self::Step { time, .. } => (vec![*time], WaveformEventKind::StepJump),
            Self::Pulse(pulse) => {
                return Ok(WaveformBreakpoints(Inner::Pulse(
                    pulse.breakpoints_in(t0, t1)?,
                )));
            }
            Self::Pwl(pwl) => {
                return Ok(WaveformBreakpoints(Inner::Pwl(pwl.breakpoints_in(t0, t1)?)));
            }
            Self::Function(function) => (function.corners(), WaveformEventKind::FunctionCorner),
            Self::PulseDefaults(_) | Self::FunctionDefaults(_) => {
                return Err(SpiceError::circuit(
                    "source defaults are unresolved; bind transient timing before enumerating breakpoints",
                ));
            }
        };
        Ok(WaveformBreakpoints(Inner::Fixed(
            fixed
                .into_iter()
                .filter(|t| (t0..=t1).contains(t))
                .map(|time| WaveformEvent { time, kind })
                .collect::<Vec<_>>()
                .into_iter(),
        )))
    }
}

/// Provenance of an event in a resolved source waveform.
///
/// A waveform corner
/// is not necessarily an accepted simulator breakpoint: C's accept callbacks
/// apply their own rounding, coalescing and source-specific scheduling policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformEventKind {
    /// Start of a pulse (also a possible transition at a truncated period).
    PulsePeriodStart,
    /// An ordinary rise/plateau/fall corner.
    PulseEdge,
    /// A fractional pulse count terminates an active pulse.
    PulseCountCut,
    /// C's final accept request beyond the pulse count; not a forcing corner.
    PulseTrailingRequest,
    /// A supplied PWL knot.
    PwlKnot,
    /// A PWL end knot that also starts the next repetition (possibly a jump).
    PwlRepeatBoundary,
    /// A programmatically constructed step discontinuity.
    StepJump,
    /// SIN/EXP/SFFM/AM delay or slope corner; C does not request a breakpoint.
    FunctionCorner,
}

impl WaveformEventKind {
    /// Whether the comparator requires both solvers to sample this nominal
    /// corner. Actual landings remain subject to C's source-specific accept
    /// routines; a missing sample is reported rather than silently ignored.
    #[must_use]
    pub const fn requires_landing(self) -> bool {
        matches!(
            self,
            Self::PulsePeriodStart
                | Self::PulseEdge
                | Self::PulseCountCut
                | Self::PwlKnot
                | Self::PwlRepeatBoundary
                | Self::StepJump
        )
    }
}

/// One event, at its exact generated time before any comparison tolerance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaveformEvent {
    /// Absolute time.
    pub time: Real,
    /// Its origin in the waveform.
    pub kind: WaveformEventKind,
}

/// A finalized circuit's independent source, resolved for one transient plan.
///
/// The same validated waveform type supplies simulation forcing and events.
/// Source identity is separate from time because different sources may have
/// coincident corners.
#[derive(Debug, Clone)]
pub struct SourceTimeline<'a> {
    /// Elaborated instance name (including its subcircuit path).
    pub name: &'a str,
    /// Physical source type.
    pub kind: SourceKind,
    /// Analysis-resolved waveform.
    pub waveform: Waveform,
}

/// Lazy breakpoint sequence of one [`Waveform`] (see [`Waveform::breakpoints_in`]).
#[derive(Debug, Clone)]
pub struct WaveformBreakpoints(Inner);
#[derive(Debug, Clone)]
enum Inner {
    Fixed(std::vec::IntoIter<WaveformEvent>),
    Pulse(crate::devices::pulse::PulseBreakpoints),
    Pwl(crate::devices::functions::PwlBreakpoints),
}
impl WaveformBreakpoints {
    /// Advance the same sequence as the time-only iterator, retaining event
    /// provenance. Use this when deciding which events require C samples.
    pub fn next_event(&mut self) -> Option<WaveformEvent> {
        match &mut self.0 {
            Inner::Fixed(it) => it.next(),
            Inner::Pulse(it) => it.next_event(),
            Inner::Pwl(it) => it.next_event(),
        }
    }
}
impl Iterator for WaveformBreakpoints {
    type Item = Real;
    fn next(&mut self) -> Option<Real> {
        self.next_event().map(|event| event.time)
    }
}

/// A finite event ordered by its time bits and then source ordinal.
#[derive(Debug, Clone, Copy)]
struct ScheduledCorner {
    time: Real,
    source: usize,
}

impl PartialEq for ScheduledCorner {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for ScheduledCorner {}
impl PartialOrd for ScheduledCorner {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ScheduledCorner {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time
            .total_cmp(&other.time)
            .then_with(|| self.source.cmp(&other.source))
    }
}

/// Lazy, merged breakpoints of every source in a [`LinearSystem`].
///
/// One heap entry per nonempty source. Advancement is logarithmic in the
/// number of active sources, not a scan over every source for every event.
#[derive(Debug, Clone)]
pub struct SystemBreakpoints {
    sources: Vec<WaveformBreakpoints>,
    pending: BinaryHeap<Reverse<ScheduledCorner>>,
}
impl SystemBreakpoints {
    fn new(mut sources: Vec<WaveformBreakpoints>) -> Self {
        let pending = BinaryHeap::from(
            sources
                .iter_mut()
                .enumerate()
                .filter_map(|(source, iterator)| {
                    iterator
                        .next()
                        .map(|time| Reverse(ScheduledCorner { time, source }))
                })
                .collect::<Vec<_>>(),
        );
        Self { sources, pending }
    }

    fn advance(&mut self, source: usize) {
        if let Some(time) = self.sources[source].next() {
            self.pending.push(Reverse(ScheduledCorner { time, source }));
        }
    }
}
impl Iterator for SystemBreakpoints {
    type Item = Real;
    fn next(&mut self) -> Option<Real> {
        let Reverse(first) = self.pending.pop()?;
        self.advance(first.source);
        // Match the old total-order minimum, including -0 ahead of +0,
        // but coalesce both signed zeros by numeric equality.
        while self.pending.peek().is_some_and(|Reverse(event)| {
            event.time.partial_cmp(&first.time) == Some(Ordering::Equal)
        }) {
            let Reverse(event) = self.pending.pop()?;
            self.advance(event.source);
        }
        Some(first.time)
    }
}

/// Physical type of an independent source, independent of its instance name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceKind {
    /// A voltage constraint, with a branch-current unknown.
    Voltage,
    /// A current injection into terminal KCL rows.
    Current,
}

/// A source bound to RHS rows with signed contributions.
#[derive(Debug, Clone)]
pub struct LinearSource {
    /// Instance name (for sweep selection).
    pub name: String,
    /// Physical target type; not inferred from a programmatic device name.
    pub kind: SourceKind,
    /// Signed RHS locations, omitting ground.
    pub rows: Vec<(usize, Real)>,
    /// Operating-point value.
    pub dc: Real,
    /// Small-signal phasor.
    pub ac: Complex,
    /// Transient forcing.
    pub waveform: Waveform,
}

/// Preassembled state-independent `E x' + A x = b(t)`.
#[derive(Debug, Clone)]
pub struct LinearSystem {
    /// Static conductances and branch constraints.
    pub a: SparseMatrix,
    /// Capacitive/inductive operator, not companion stamps.
    pub e: SparseMatrix,
    /// Independent sources.
    pub sources: Vec<LinearSource>,
    /// Rows with explicit device initial conditions (not yet supported).
    pub has_initial_conditions: bool,
    /// Contributions of `e` that one device adds to one element as a single
    /// `omega`-scaled sum ([`LinearContext::dynamic_sum`]); empty for most
    /// systems.
    pub e_groups: Vec<crate::maths::DynamicGroup>,
}
impl LinearSystem {
    /// Empty assembly for n unknowns.
    #[must_use]
    pub const fn new(n: usize) -> Self {
        Self {
            a: SparseMatrix::new(n, n),
            e: SparseMatrix::new(n, n),
            sources: vec![],
            has_initial_conditions: false,
            e_groups: vec![],
        }
    }
    /// DC forcing, optionally overriding one independent source.
    ///
    /// # Errors
    /// An out-of-range source row or nonfinite source sum.
    pub fn dc_rhs(&self, sweep: Option<(&str, Real)>) -> SpiceResult<Vector> {
        let mut rhs = Vector::zeros(self.a.rows());
        for s in &self.sources {
            let value = sweep
                .filter(|(name, _)| s.name.eq_ignore_ascii_case(name))
                .map_or(s.dc, |(_, v)| v);
            for (row, sign) in &s.rows {
                rhs.add_to(*row, sign * value)?;
            }
        }
        if !rhs.is_finite() {
            return Err(SpiceError::circuit("non-finite source sum"));
        }
        Ok(rhs)
    }
    /// Binds the analysis quantities that C uses for PULSE defaults (`CKTstep`,
    /// `CKTfinalTime`), turning every [`Waveform::PulseDefaults`] into a
    /// resolved [`Waveform::Pulse`]. Call once before time forcing is evaluated.
    ///
    /// # Errors
    /// A pulse is invalid after resolution; the system is left unchanged.
    pub fn bind_transient_timing(&mut self, timing: &TransientTiming) -> SpiceResult<()> {
        let resolved = self
            .sources
            .iter()
            .map(|s| s.waveform.resolve(timing))
            .collect::<SpiceResult<Vec<_>>>()?;
        for (source, waveform) in self.sources.iter_mut().zip(resolved) {
            source.waveform = waveform;
        }
        Ok(())
    }
    /// Real transient RHS `b(t)` with an explicit one-sided limit at jumps.
    ///
    /// # Errors
    /// Nonfinite time, unresolved PULSE defaults or a nonfinite source sum.
    pub fn transient_rhs(&self, t: Real, limit: Limit) -> SpiceResult<Vector> {
        let mut rhs = Vector::zeros(self.a.rows());
        for s in &self.sources {
            let value = s.waveform.value_at(t, limit)?;
            for (r, sign) in &s.rows {
                rhs.add_to(*r, sign * value)?;
            }
        }
        if !rhs.is_finite() {
            return Err(SpiceError::circuit("non-finite source sum"));
        }
        Ok(rhs)
    }
    /// Clear the assembled operators and forcing, retaining outer storage.
    pub(crate) fn clear(&mut self) {
        self.a.clear();
        self.e.clear();
        self.sources.clear();
        self.e_groups.clear();
        self.has_initial_conditions = false;
    }

    /// Complex AC forcing.
    pub fn ac_rhs(&self) -> Vec<Complex> {
        let mut rhs = Vec::new();
        self.ac_rhs_into(&mut rhs);
        rhs
    }

    /// Rebuild the AC forcing using caller-owned storage.
    pub(crate) fn ac_rhs_into(&self, rhs: &mut Vec<Complex>) {
        rhs.clear();
        rhs.resize(self.a.rows(), Complex::ZERO);
        for s in &self.sources {
            for (r, sign) in &s.rows {
                rhs[*r] = rhs[*r] + Complex::real(*sign) * s.ac;
            }
        }
    }
    /// Lazily merges every source's corners and jumps in the closed window
    /// `[t0, t1]` into one ascending, duplicate-free sequence.
    ///
    /// Nothing is materialized: a periodic pulse over a long run yields one
    /// cycle at a time, so a driver must bound how many it consumes.
    ///
    /// # Errors
    /// Invalid window or unresolved PULSE defaults.
    pub fn breakpoints_in(&self, t0: Real, t1: Real) -> SpiceResult<SystemBreakpoints> {
        let sources = self
            .sources
            .iter()
            .map(|s| s.waveform.breakpoints_in(t0, t1))
            .collect::<SpiceResult<_>>()?;
        Ok(SystemBreakpoints::new(sources))
    }
}

/// Per-device immutable equation assembly context.
pub struct LinearContext<'a> {
    /// Explicit circuit/nominal temperatures for immutable model evaluation.
    pub model_context: &'a crate::devices::models::ModelContext,
    /// Operators and forcing being assembled.
    pub system: &'a mut LinearSystem,
    /// Node numbering, with ground eliminated.
    pub unknowns: &'a MnaUnknowns,
    /// The first allocated branch row for this device, if any.
    pub branch: Option<usize>,
    /// Branch rows of the devices this one senses
    /// ([`crate::devices::Device::controlling_sources`]), in order.
    pub controls: &'a [usize],
    /// Mutual-inductance terms of this device's branch equation (empty except
    /// for coupled inductors); see [`crate::devices::traits::MutualTerm`].
    pub mutual: &'a [crate::devices::traits::MutualTerm],
    /// This device's slots of the bias point's converged state, for
    /// small-signal assemblies built from a solved operating point
    /// ([`crate::devices::Circuit::small_signal_system_at`]); `None` otherwise.
    pub states: Option<&'a [Real]>,
    /// Set while [`crate::devices::Circuit::pole_zero_terms_at`] collects
    /// `CKTpzLoad`'s terms. A device whose C pole-zero load adds a
    /// conductance and an `s` term to one element in separate statements
    /// (`b4pzld.c`) records each statement here, in C's order, as a
    /// conductance-only or capacitance-only term; the collector then uses
    /// them instead of merging the device's A and E stamps per element.
    pub pole_zero_statements: Option<&'a mut Vec<crate::maths::pencil::PencilTerm>>,
    /// For a pole-zero assembly, the number of `CKTpzSetup` runs so far,
    /// this one included (see
    /// [`crate::devices::setup::MatrixSetup::pole_zero_setups`]); zero
    /// otherwise.
    pub pole_zero_setups: u32,
}
impl LinearContext<'_> {
    /// Stamps a two-terminal nodal operator into A or E.
    ///
    /// # Errors
    /// A matrix position outside the assembled system.
    pub fn nodal(&mut self, terminals: [NodeId; 2], value: Real, dynamic: bool) -> SpiceResult<()> {
        let matrix = if dynamic {
            &mut self.system.e
        } else {
            &mut self.system.a
        };
        nodal_stamp(matrix, self.unknowns, terminals, value)
    }
    /// Adds the dynamic terms `terms` to `E[row][col]` as C adds
    /// `x1 + x2 + ...` (each `cap * CKTomega`) in one step: the complex
    /// assembly scales each term by `omega` and adds their sum. Ground rows
    /// or columns are skipped.
    ///
    /// # Errors
    /// An out-of-range position.
    pub fn dynamic_sum(&mut self, row: NodeId, col: NodeId, terms: &[Real]) -> SpiceResult<()> {
        self.scaled_dynamic_sum(row, col, terms, 1.)
    }
    /// [`Self::dynamic_sum`] multiplied by `scale` after the sum, as C's
    /// `m * (x1 + x2 + ...)`.
    ///
    /// # Errors
    /// An out-of-range position.
    pub fn scaled_dynamic_sum(
        &mut self,
        row: NodeId,
        col: NodeId,
        terms: &[Real],
        scale: Real,
    ) -> SpiceResult<()> {
        let (Some(r), Some(c)) = (self.unknowns.node_row(row), self.unknowns.node_row(col)) else {
            return Ok(());
        };
        self.dynamic_group(r, c, terms, scale, 1.)
    }
    /// Adds `scale * (x1 + x2 + ...)` to the imaginary part of the AC matrix
    /// at (`row`, `col`), the `x` already including `CKTomega` (C's
    /// frequency-dependent `PSacload` terms): unlike
    /// [`Self::scaled_dynamic_sum`] the complex assembly does not multiply
    /// them by `omega`. Only for a device whose small-signal load is
    /// reassembled at every frequency
    /// ([`crate::devices::Device::small_signal_depends_on_frequency`]).
    ///
    /// # Errors
    /// An out-of-range position.
    pub fn prescaled_dynamic_sum(
        &mut self,
        row: NodeId,
        col: NodeId,
        terms: &[Real],
        scale: Real,
    ) -> SpiceResult<()> {
        let (Some(r), Some(c)) = (self.unknowns.node_row(row), self.unknowns.node_row(col)) else {
            return Ok(());
        };
        let start = self.system.e.triplets().len();
        for term in terms {
            self.system.e.add(r, c, scale * *term)?;
        }
        let end = self.system.e.triplets().len();
        if end > start && self.system.e.overflow().is_none() {
            self.system.e_groups.push(crate::maths::DynamicGroup {
                range: start..end,
                terms: terms.to_vec(),
                scale,
                divisor: 1.,
                prescaled: true,
            });
        }
        Ok(())
    }
    /// Adds `value`, which already includes `CKTomega`, to the imaginary part
    /// of the AC matrix at matrix rows (`r`, `c`); see
    /// [`Self::prescaled_dynamic_sum`].
    ///
    /// # Errors
    /// An out-of-range position.
    pub fn dynamic_prescaled(&mut self, r: usize, c: usize, value: Real) -> SpiceResult<()> {
        let start = self.system.e.triplets().len();
        self.system.e.add(r, c, value)?;
        let end = self.system.e.triplets().len();
        if end > start && self.system.e.overflow().is_none() {
            self.system.e_groups.push(crate::maths::DynamicGroup {
                range: start..end,
                terms: vec![value],
                scale: 1.,
                divisor: 1.,
                prescaled: true,
            });
        }
        Ok(())
    }
    /// Adds `scale * (x1 + x2 + ...) / divisor` to `E[r][c]` (matrix rows,
    /// not nodes), each `x` being a term times `omega`, as one C stamp.
    ///
    /// # Errors
    /// An out-of-range position.
    pub fn dynamic_group(
        &mut self,
        r: usize,
        c: usize,
        terms: &[Real],
        scale: Real,
        divisor: Real,
    ) -> SpiceResult<()> {
        let start = self.system.e.triplets().len();
        for term in terms {
            self.system.e.add(r, c, scale * *term / divisor)?;
        }
        let end = self.system.e.triplets().len();
        // Only an unbound (triplet-list) E keeps the contributions apart.
        if end > start
            && (terms.len() > 1
                || scale.to_bits() != 1_f64.to_bits()
                || divisor.to_bits() != 1_f64.to_bits())
            && self.system.e.overflow().is_none()
        {
            self.system.e_groups.push(crate::maths::DynamicGroup {
                range: start..end,
                terms: terms.to_vec(),
                scale,
                divisor,
                prescaled: false,
            });
        }
        Ok(())
    }
    /// Stamps a voltage/inductor branch, positive current from terminal 0 to 1.
    ///
    /// # Errors
    /// The branch-row binding is absent or a matrix position is out of range.
    pub fn branch(&mut self, terminals: [NodeId; 2]) -> SpiceResult<usize> {
        let branch = self
            .branch
            .ok_or_else(|| SpiceError::circuit("missing branch-row binding"))?;
        branch_stamp(&mut self.system.a, self.unknowns, terminals, branch)?;
        Ok(branch)
    }
}

pub(crate) fn nodal_stamp(
    matrix: &mut SparseMatrix,
    unknowns: &MnaUnknowns,
    terminals: [NodeId; 2],
    value: Real,
) -> SpiceResult<()> {
    for (node, sign) in [(terminals[0], 1.), (terminals[1], -1.)] {
        for (other, other_sign) in [(terminals[0], 1.), (terminals[1], -1.)] {
            if let (Some(r), Some(c)) = (unknowns.node_row(node), unknowns.node_row(other)) {
                matrix.add(r, c, value * sign * other_sign)?;
            }
        }
    }
    Ok(())
}
pub(crate) fn branch_stamp(
    matrix: &mut SparseMatrix,
    unknowns: &MnaUnknowns,
    terminals: [NodeId; 2],
    branch: usize,
) -> SpiceResult<()> {
    for (node, sign) in [(terminals[0], 1.), (terminals[1], -1.)] {
        if let Some(row) = unknowns.node_row(node) {
            matrix.add(row, branch, sign)?;
            matrix.add(branch, row, sign)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan_merge(mut sources: Vec<std::iter::Peekable<WaveformBreakpoints>>) -> Vec<u64> {
        let mut result = Vec::new();
        while let Some(time) = sources
            .iter_mut()
            .filter_map(|source| source.peek().copied())
            .min_by(Real::total_cmp)
        {
            result.push(time.to_bits());
            for source in &mut sources {
                while source
                    .next_if(|t| t.partial_cmp(&time) == Some(Ordering::Equal))
                    .is_some()
                {}
            }
        }
        result
    }

    #[test]
    fn heap_schedule_matches_scan_with_signed_zero_ties_and_fractional_cycles() {
        let mut waveforms = vec![
            Waveform::Pwl(PwlSource::plain(vec![(-0., 1.), (1., 2.)]).unwrap()),
            Waveform::Pwl(PwlSource::new(vec![(-0., 1.), (1., 2.)], 0., Some(0.)).unwrap()),
            Waveform::Pulse(
                Pulse::new(0., 2., 0., 0., 0., 1., 2.)
                    .unwrap()
                    .with_count(2.25)
                    .unwrap(),
            ),
            Waveform::Constant(0.),
        ];
        for i in 0..128 {
            let i = Real::from(i);
            waveforms.push(Waveform::Pwl(
                PwlSource::plain(vec![(i * 0.125, 0.), (i * 0.125 + 0.1, 1.)]).unwrap(),
            ));
        }
        for (start, end) in [
            (-0., 20.),
            (0.01, 1.),
            (2., 8.25),
            (10., 100.),
            (150., 200.),
        ] {
            let iterators = waveforms
                .iter()
                .map(|waveform| waveform.breakpoints_in(start, end).unwrap())
                .collect::<Vec<_>>();
            let expected = scan_merge(iterators.iter().cloned().map(Iterator::peekable).collect());
            let mut actual = SystemBreakpoints::new(iterators);
            let first = actual.next();
            let clone = actual.clone();
            let tail = actual.map(Real::to_bits).collect::<Vec<_>>();
            assert_eq!(tail, clone.map(Real::to_bits).collect::<Vec<_>>());
            assert_eq!(
                first
                    .into_iter()
                    .map(Real::to_bits)
                    .chain(tail)
                    .collect::<Vec<_>>(),
                expected
            );
            if start == 0. {
                assert_eq!(first.unwrap().to_bits(), (-0.0_f64).to_bits());
            }
        }
    }
}
