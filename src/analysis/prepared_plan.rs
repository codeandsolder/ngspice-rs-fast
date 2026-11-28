//! Bind source analysis intent to a prepared circuit without depending on the
//! source translator. Physical occurrence names are supplied by the caller;
//! the solver must not infer them from the original source spelling.
use std::collections::BTreeSet;
use std::path::PathBuf;

use model_analysis_plan::sweep::{SweepExecution, SweepTarget};
use model_analysis_plan::timing::{IntegrationCeiling, SavedOutput, StepControl};
use model_analysis_plan::{
    AnalysisKind as SourceKind, AnalysisPlan as SourcePlan, BoundCoordinate,
    BoundIndependentSweeps, Directive, DirectiveOrigin, Disposition, PlanEntry, PlanKind,
};
use model_normalize_ngspice::ltspice::transient::TransientModifier;
use model_preprocess_command::WorkBudget;

use super::optimization::parameters::CandidateDeckPlan;
use super::sweep::{DcAxisTarget, DcSweepAxis};
use super::{
    AnalysisInput, AnalysisPlan, OutputSampling, PlanningLimits, Plot, RunConfig, TimingSegment,
    TransientSchedule,
};
use crate::devices::circuit::CandidateEdit;
use crate::devices::subckt::PreparedDeclarations;
use crate::netlist::ast::Netlist;
use crate::netlist::eval::{EvalBudget, ParamScope};
use crate::primitives::{AnalysisKind, Real, SourceLoc, SpiceError, SpiceResult};

/// An explicitly resolved target in the electrical artifact's namespace.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum NativeTarget {
    /// Independent root parameter.
    Parameter(String),
    /// Circuit temperature in Celsius.
    Temperature,
    /// Expanded independent voltage or current source.
    Source(String),
    /// Expanded resistance setter in ohms.
    Resistance(String),
    /// Expanded instance and native parameter.
    Instance {
        /// Expanded native device name.
        name: String,
        /// Native scalar setter name.
        parameter: String,
    },
    /// A native model declaration and scalar parameter.
    Model {
        /// Lexical declaration path.
        scope: Vec<String>,
        /// Native model name.
        name: String,
        /// Native scalar setter name.
        parameter: String,
    },
}

impl NativeTarget {
    fn canonical(self) -> Self {
        match self {
            Self::Parameter(name) => Self::Parameter(name.to_ascii_lowercase()),
            Self::Source(name) => Self::Source(name.to_ascii_lowercase()),
            Self::Resistance(name) => Self::Resistance(name.to_ascii_lowercase()),
            Self::Instance { name, parameter } => Self::Instance {
                name: name.to_ascii_lowercase(),
                parameter: parameter.to_ascii_lowercase(),
            },
            Self::Model {
                scope,
                name,
                parameter,
            } => Self::Model {
                scope: scope
                    .into_iter()
                    .map(|name| name.to_ascii_lowercase())
                    .collect(),
                name: name.to_ascii_lowercase(),
                parameter: parameter.to_ascii_lowercase(),
            },
            Self::Temperature => Self::Temperature,
        }
    }

    fn continuation(&self) -> DcAxisTarget {
        match self {
            Self::Source(name) => DcAxisTarget::Source(name.clone()),
            Self::Resistance(name) => DcAxisTarget::Resistance(name.clone()),
            Self::Temperature => DcAxisTarget::Temperature,
            Self::Instance { name, parameter } => DcAxisTarget::Instance {
                instance: name.clone(),
                parameter: parameter.clone(),
            },
            Self::Parameter(name) => DcAxisTarget::Parameter(name.clone()),
            Self::Model {
                scope,
                name,
                parameter,
            } => DcAxisTarget::Model {
                scope: scope.clone(),
                name: name.clone(),
                parameter: parameter.clone(),
            },
        }
    }

    fn structural(&self) -> bool {
        matches!(self, Self::Parameter(_) | Self::Model { .. })
    }

    fn edit(&self, value: Real) -> Option<CandidateEdit<'_>> {
        match self {
            Self::Parameter(_) | Self::Temperature => None,
            Self::Source(name) => Some(CandidateEdit::Source { name, value }),
            Self::Resistance(name) => Some(CandidateEdit::Instance {
                name,
                parameter: "resistance",
                value,
            }),
            Self::Instance { name, parameter } => Some(CandidateEdit::Instance {
                name,
                parameter,
                value,
            }),
            Self::Model {
                scope,
                name,
                parameter,
            } => Some(CandidateEdit::Model {
                scope,
                name,
                parameter,
                value,
            }),
        }
    }
}

fn location(entry: &PlanEntry) -> SourceLoc {
    let (path, line) = match &entry.origin {
        DirectiveOrigin::Source {
            path, first_line, ..
        }
        | DirectiveOrigin::SourceOccurrence {
            path, first_line, ..
        } => (path.clone(), *first_line),
        DirectiveOrigin::Generated {
            artifact,
            first_line,
            ..
        } => (PathBuf::from(artifact), *first_line),
    };
    SourceLoc::new(path, u32::try_from(line).unwrap_or(u32::MAX), 1)
}

fn at(entry: &PlanEntry, message: impl Into<String>) -> SpiceError {
    SpiceError::parse(location(entry), message)
}

fn map_target(
    entry: &PlanEntry,
    target: &SweepTarget,
    resolve: &mut impl FnMut(&PlanEntry, &SweepTarget) -> SpiceResult<NativeTarget>,
) -> SpiceResult<NativeTarget> {
    let bound = resolve(entry, target)?.canonical();
    // Reject a resolver that changes the declared physical quantity.
    let compatible = matches!(
        (target, &bound),
        (SweepTarget::Parameter { .. }, NativeTarget::Parameter(_))
            | (SweepTarget::Temperature, NativeTarget::Temperature)
            | (SweepTarget::Source { .. }, NativeTarget::Source(_))
            | (
                SweepTarget::Device { .. },
                NativeTarget::Instance { .. }
                    | NativeTarget::Resistance(_)
                    | NativeTarget::Source(_)
            )
            | (
                SweepTarget::ModelParameter { .. },
                NativeTarget::Model { .. }
            )
    );
    if !compatible {
        return Err(at(
            entry,
            format!("incompatible native binding for {target:?}"),
        ));
    }
    Ok(bound)
}

fn native_kind(kind: SourceKind, entry: &PlanEntry) -> SpiceResult<AnalysisKind> {
    match kind {
        SourceKind::OperatingPoint => Ok(AnalysisKind::OperatingPoint),
        SourceKind::DcSweep => Ok(AnalysisKind::DcSweep),
        SourceKind::Ac => Ok(AnalysisKind::Ac),
        SourceKind::Transient => Ok(AnalysisKind::Transient),
        SourceKind::Noise => Ok(AnalysisKind::Noise),
        SourceKind::TransferFunction => Ok(AnalysisKind::TransferFunction),
        _ => Err(at(entry, "source analysis has no qualified native driver")),
    }
}

/// Collect the enclosing native analysis kinds and bias selection needed when
/// attaching a source-prepared initialization sidecar to an electrical deck.
///
/// # Errors
/// A source request has no qualified native analysis kind.
pub fn initialization_selection(plan: &SourcePlan) -> SpiceResult<BTreeSet<(AnalysisKind, bool)>> {
    plan.entries
        .iter()
        .filter(|entry| entry.is_analysis())
        .map(selection)
        .collect()
}

fn selection(entry: &PlanEntry) -> SpiceResult<(AnalysisKind, bool)> {
    match &entry.directive {
        Directive::Analysis { analysis, uic, .. } => Ok((native_kind(*analysis, entry)?, *uic)),
        Directive::DcSweep { .. } => Ok((AnalysisKind::DcSweep, false)),
        Directive::Transient { intent } => Ok((
            AnalysisKind::Transient,
            intent.modes.contains(TransientModifier::Uic),
        )),
        Directive::ScheduledTransient { intent } => Ok((AnalysisKind::Transient, intent.uic)),
        _ => Err(at(entry, "expected prepared analysis request")),
    }
}

struct Job<'a> {
    entry_index: usize,
    entry: &'a PlanEntry,
    sweeps: BoundIndependentSweeps<'a>,
    targets: Vec<NativeTarget>,
    dc_targets: Vec<NativeTarget>,
}

/// The original declaration and resolved coordinates remain attached to each
/// repetition, including duplicate coordinates in source order.
pub struct PreparedResult<'a> {
    /// Index in the original source plan.
    pub entry_index: usize,
    /// Original analysis declaration.
    pub entry: &'a PlanEntry,
    /// Zero-based repetition ordinal, innermost axis fastest.
    pub ordinal: usize,
    /// Source-axis identities and physical coordinates.
    pub coordinates: Vec<BoundCoordinate<'a>>,
    /// Every plot produced by this one independent run, in driver order.
    pub plots: Vec<Plot>,
}

/// A validated source-coordinate plan. Only the current candidate circuit and
/// its plots are materialized during execution; prior candidates never
/// supply mutable device, Newton, or integration state.
pub struct PreparedAnalysisPlan<'a> {
    circuit: CandidateDeckPlan,
    jobs: Vec<Job<'a>>,
}

impl<'a> PreparedAnalysisPlan<'a> {
    /// Bind independent grids against the nominal electrical lexical scope.
    /// The resolver maps every source target to an explicit native identity;
    /// for unchanged root parameters and temperature it can simply return
    /// the corresponding native target. Source declarations requiring
    /// unimplemented observation or execution policies are rejected.
    ///
    /// # Errors
    /// Invalid plan or electrical artifact, unqualified directive or target,
    /// invalid coordinate grid, duplicate target or exceeded job budget.
    pub fn bind(
        source: Netlist,
        plan: &'a SourcePlan,
        declarations: PreparedDeclarations<'_>,
        mut resolve: impl FnMut(&PlanEntry, &SweepTarget) -> SpiceResult<NativeTarget>,
        work: &WorkBudget,
    ) -> SpiceResult<Self> {
        plan.validate_supported()
            .map_err(|error| SpiceError::circuit(error.to_string()))?;
        if plan.kind != PlanKind::Deck || plan.statistical_sample.is_some() {
            return Err(SpiceError::circuit(
                "prepared analysis requires an executable nonsampled deck",
            ));
        }
        for entry in &plan.entries {
            let requires_initialization = entry.disposition == Disposition::Initialization
                || matches!(&entry.directive, Directive::Options { settings } if settings.iter().any(|setting| setting.disposition == Disposition::Initialization));
            if requires_initialization && source.prepared_initialization.is_none() {
                return Err(at(
                    entry,
                    "source initialization requires the prepared initialization sidecar",
                ));
            }
            if entry.is_analysis() {
                let (kind, uic) = selection(entry)?;
                if let Some(policy) = &source.prepared_initialization {
                    policy.validate_selection(kind, uic)?;
                }
            }
        }
        let nominal = ParamScope::for_netlist(&source)?;
        let mut budget = EvalBudget::for_netlist(&source)?;
        let mut jobs = Vec::new();
        let mut parameters = BTreeSet::new();
        let mut total = 0usize;
        for (index, entry) in plan.entries.iter().enumerate() {
            if !entry.is_analysis() {
                if entry.disposition == Disposition::RequiresBinding
                    && !matches!(&entry.directive, Directive::Sweep { axis } if axis.execution == SweepExecution::Independent)
                {
                    return Err(at(
                        entry,
                        "unqualified prepared analysis or observation directive",
                    ));
                }
                if let Directive::Options { settings } = &entry.directive
                    && settings
                        .iter()
                        .any(|setting| setting.disposition == Disposition::RequiresBinding)
                {
                    return Err(at(entry, "unqualified prepared analysis option"));
                }
                continue;
            }
            if !entry.scope.is_empty() {
                return Err(at(entry, "scoped analysis requires occurrence binding"));
            }
            let sweeps = plan
                .bind_independent_sweeps(
                    index,
                    |decl, _, raw| {
                        if !decl.scope.is_empty() {
                            return Err(at(decl, "scoped sweep requires occurrence binding").into());
                        }
                        Ok(crate::netlist::prepared_initialization::evaluate_scalar(
                            raw,
                            &location(decl),
                            &nominal,
                            None,
                            &mut budget,
                        )?)
                    },
                    work,
                )
                .map_err(|error| at(entry, error.to_string()))?;
            total = total
                .checked_add(sweeps.count())
                .filter(|&count| count <= PlanningLimits::default().max_jobs)
                .ok_or_else(|| at(entry, "prepared analysis job limit exceeded"))?;
            let mut seen = BTreeSet::new();
            let mut targets = Vec::new();
            for axis in sweeps.axes() {
                let target = map_target(axis.entry, &axis.axis.target, &mut resolve)?;
                if !seen.insert(target.clone()) {
                    return Err(at(axis.entry, "duplicate independent sweep target"));
                }
                if let NativeTarget::Parameter(name) = &target {
                    parameters.insert(name.clone());
                }
                targets.push(target);
            }
            let dc_targets = match &entry.directive {
                Directive::DcSweep { axes } => {
                    let mut targets = Vec::with_capacity(axes.len());
                    for axis in axes {
                        if axis.execution != SweepExecution::Continuation {
                            return Err(at(entry, "DC axis is not a continuation"));
                        }
                        let target = map_target(entry, &axis.target, &mut resolve)?;
                        if !seen.insert(target.clone()) {
                            return Err(at(entry, "duplicate independent or continuation target"));
                        }
                        if let NativeTarget::Parameter(name) = &target {
                            parameters.insert(name.clone());
                        }
                        targets.push(target);
                    }
                    targets
                }
                Directive::Analysis {
                    analysis: SourceKind::DcSweep,
                    ..
                } if entry.disposition != Disposition::NativeRequest => {
                    return Err(at(
                        entry,
                        "source DC operands require typed occurrence binding",
                    ));
                }
                _ => Vec::new(),
            };
            jobs.push(Job {
                entry_index: index,
                entry,
                sweeps,
                targets,
                dc_targets,
            });
        }
        if jobs.is_empty() {
            return Err(SpiceError::circuit(
                "prepared plan has no executable analysis",
            ));
        }
        let names = parameters.iter().map(String::as_str).collect::<Vec<_>>();
        let circuit = CandidateDeckPlan::new(source, &names, declarations)?;
        Ok(Self { circuit, jobs })
    }

    /// Execute each candidate independently, delivering its plots and original
    /// coordinates before preparing the next candidate. Bounds and timing are
    /// resolved against that candidate, not the nominal parameter snapshot.
    ///
    /// # Errors
    /// Candidate preparation, unsupported timing or analysis policy, numerical
    /// failure, or result consumer failure. Previously delivered results remain valid.
    pub fn run(
        &self,
        work: &WorkBudget,
        mut result: impl FnMut(PreparedResult<'_>) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        let mut session = self.circuit.session();
        let mut budget = EvalBudget::for_netlist(self.circuit.source())?;
        let defaults = self
            .circuit
            .inputs()
            .iter()
            .map(|input| input.default)
            .collect::<Vec<_>>();
        let mut total_points = 0usize;
        for job in &self.jobs {
            for ordinal in 0..job.sweeps.count() {
                let coordinates = job
                    .sweeps
                    .candidate(ordinal)
                    .and_then(Iterator::collect::<anyhow::Result<Vec<_>>>)
                    .map_err(|error| at(job.entry, error.to_string()))?;
                let mut values = defaults.clone();
                let mut temperature = None;
                let mut edits = Vec::new();
                for (target, coordinate) in job.targets.iter().zip(&coordinates) {
                    match target {
                        NativeTarget::Parameter(name) => {
                            let index = self
                                .circuit
                                .inputs()
                                .iter()
                                .position(|input| &input.name == name)
                                .ok_or_else(|| at(coordinate.entry, "unbound root parameter"))?;
                            values[index] = coordinate.value;
                        }
                        NativeTarget::Temperature => temperature = Some(coordinate.value),
                        _ => {
                            if let Some(edit) = target.edit(coordinate.value) {
                                edits.push(edit);
                            }
                        }
                    }
                }
                let plots = if job.dc_targets.iter().any(NativeTarget::structural) {
                    let nominal = session.resolve_config(&values, temperature, &mut budget)?;
                    let axes = bind_dc_axes(job, &nominal, work, &mut budget)?;
                    let request = dc_plan(job.entry, &nominal, axes.clone())?;
                    total_points = charge_points(total_points, &request, job.entry)?;
                    let plot = super::sweep::structural::run(&request, |point| {
                        let mut candidate_values = values.clone();
                        let mut candidate_temperature = temperature;
                        let mut candidate_edits = edits.clone();
                        for (target, value) in job.dc_targets.iter().zip(point) {
                            match target {
                                NativeTarget::Parameter(name) => {
                                    let index = self
                                        .circuit
                                        .inputs()
                                        .iter()
                                        .position(|input| &input.name == name)
                                        .ok_or_else(|| {
                                            at(job.entry, "unbound continuation parameter")
                                        })?;
                                    candidate_values[index] = value;
                                }
                                NativeTarget::Temperature => candidate_temperature = Some(value),
                                _ => {
                                    if let Some(edit) = target.edit(value) {
                                        candidate_edits.push(edit);
                                    }
                                }
                            }
                        }
                        let (circuit, config) = session.prepare(
                            &candidate_values,
                            candidate_temperature,
                            &candidate_edits,
                            &mut budget,
                        )?;
                        let plan = dc_plan(job.entry, &config, axes.clone())?;
                        Ok((circuit, plan, config.context()))
                    })?;
                    vec![plot]
                } else {
                    let (mut circuit, config) =
                        session.prepare(&values, temperature, &edits, &mut budget)?;
                    let request = bind_request(job, &config, work, &mut budget)?;
                    total_points = charge_points(total_points, &request, job.entry)?;
                    request.run_plots(&mut circuit, &config.context())?
                };
                result(PreparedResult {
                    entry_index: job.entry_index,
                    entry: job.entry,
                    ordinal,
                    coordinates,
                    plots,
                })?;
            }
        }
        Ok(())
    }
}

fn charge_points(total: usize, request: &AnalysisPlan, entry: &PlanEntry) -> SpiceResult<usize> {
    total
        .checked_add(request.requested_grid_points())
        .filter(|&count| count <= PlanningLimits::default().max_grid_points)
        .ok_or_else(|| at(entry, "aggregate prepared analysis grid limit exceeded"))
}

fn bind_dc_axes(
    job: &Job<'_>,
    config: &RunConfig,
    work: &WorkBudget,
    budget: &mut EvalBudget,
) -> SpiceResult<Vec<DcSweepAxis>> {
    let Directive::DcSweep { axes } = &job.entry.directive else {
        return Err(at(job.entry, "missing typed DC axes"));
    };
    let location = location(job.entry);
    axes.iter()
        .zip(&job.dc_targets)
        .map(|(axis, target)| {
            let values = axis
                .bind(
                    |raw| Ok(config.prepared_scalar(raw, &location, budget)?),
                    work,
                )
                .map_err(|error| at(job.entry, error.to_string()))?;
            let coordinates = (0..values.len())
                .map(|index| {
                    values
                        .value(index)
                        .ok_or_else(|| at(job.entry, "missing continuation coordinate"))
                })
                .collect::<SpiceResult<Vec<_>>>()?;
            Ok(DcSweepAxis {
                target: target.continuation(),
                coordinates: coordinates.into(),
            })
        })
        .collect()
}

fn dc_plan(
    entry: &PlanEntry,
    config: &RunConfig,
    axes: Vec<DcSweepAxis>,
) -> SpiceResult<AnalysisPlan> {
    let mut input = AnalysisInput::new(AnalysisKind::DcSweep);
    input.dc_axes = Some(axes);
    input.set_location(location(entry));
    config.plan(input)
}

fn bind_request(
    job: &Job<'_>,
    config: &RunConfig,
    work: &WorkBudget,
    budget: &mut EvalBudget,
) -> SpiceResult<AnalysisPlan> {
    let entry = job.entry;
    let location = location(entry);
    let mut scalar = |raw: &str| config.prepared_scalar(raw, &location, budget);
    let mut input = match &entry.directive {
        Directive::ScheduledTransient { intent } => {
            let bound = intent
                .bind(|raw| Ok(scalar(raw)?))
                .map_err(|error| at(entry, error.to_string()))?;
            // This is native integration with an explicit physical ceiling.
            // No arbitrary hard cap may replace the source automatic algorithm.
            if !matches!(bound.step_control, StepControl::RunLevel { level: 3, .. }) {
                return Err(at(
                    entry,
                    "source-specific RUNLVL or legacy minimum timestep not qualified",
                ));
            }
            let IntegrationCeiling::Explicit(ceiling) = bound.ceiling else {
                return Err(at(
                    entry,
                    "automatic source integration ceiling not qualified",
                ));
            };
            let output = bound
                .output
                .iter()
                .map(|segment| TimingSegment {
                    until: segment.until,
                    step: segment.step,
                })
                .collect();
            let ceiling = ceiling
                .iter()
                .map(|segment| TimingSegment {
                    until: segment.until,
                    step: segment.step,
                })
                .collect();
            let sampling = match bound.saved_output {
                SavedOutput::AcceptedSteps => OutputSampling::Accepted,
                SavedOutput::LinearInterpolation => OutputSampling::Interpolate,
            };
            let hint = bound
                .output
                .first()
                .ok_or_else(|| at(entry, "missing timing segment"))?
                .step;
            let mut input = AnalysisInput::new(AnalysisKind::Transient);
            input.uic = bound.uic;
            input.transient_schedule = Some(TransientSchedule::new(
                bound.start,
                output,
                ceiling,
                hint,
                sampling,
            )?);
            input
        }
        Directive::Transient { intent } => {
            let bound = intent
                .bind(|raw| Ok(scalar(raw)?))
                .map_err(|error| at(entry, error.to_string()))?;
            let timing = bound
                .native_timing()
                .map_err(|error| at(entry, error.to_string()))?;
            let mut input = AnalysisInput::with_arguments(
                AnalysisKind::Transient,
                timing.map(|value| value.to_string()),
            );
            input.uic = bound.modes.contains(TransientModifier::Uic);
            input
        }
        Directive::DcSweep { .. } => {
            return dc_plan(entry, config, bind_dc_axes(job, config, work, budget)?);
        }
        Directive::Analysis {
            analysis,
            arguments,
            uic,
        } => {
            let kind = native_kind(*analysis, entry)?;
            let mut arguments = arguments.clone();
            match kind {
                AnalysisKind::OperatingPoint => {}
                AnalysisKind::Ac
                    if arguments.len() == 4
                        && ["lin", "dec", "oct"]
                            .iter()
                            .any(|mode| arguments[0].eq_ignore_ascii_case(mode)) =>
                {
                    for argument in &mut arguments[1..] {
                        *argument = scalar(argument)?.to_string();
                    }
                }
                AnalysisKind::Transient if entry.disposition == Disposition::NativeRequest => {
                    for argument in &mut arguments {
                        *argument = scalar(argument)?.to_string();
                    }
                }
                _ if entry.disposition == Disposition::NativeRequest => {}
                _ => {
                    return Err(at(
                        entry,
                        "source analysis operands require a qualified native binding",
                    ));
                }
            }
            let mut input = AnalysisInput::with_arguments(kind, arguments);
            input.uic = *uic;
            input
        }
        _ => return Err(at(entry, "expected prepared analysis request")),
    };
    // Preserve the original source location through native plan diagnostics.
    input.set_location(location);
    config.plan(input)
}
