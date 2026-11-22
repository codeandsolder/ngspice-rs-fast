//! Reusable schema-bound objective plans.
//!
//! One ordered plan per scenario;
//! one fresh sink per candidate. No solver-specific fitness policy,
//! formatting, rawfile round-trip or per-candidate full plot is involved.
//!
//! Operations that need arbitrary history use a bounded raw-column
//! projection, not the producer's whole result schema.

use crate::analysis::measure::online::{
    AxisTracker, BoundMeasure, MeasureState, PointQuery, QueryFailure,
};
use crate::analysis::measure::{self, Axis, Measurement};
use crate::analysis::results::{Plot, PointSink, lifecycle_error};
use crate::analysis::selection::{Column, resolve_request};
use crate::analysis::{AnalysisContext, AnalysisPlan, FourierAnalysis, fourier};
use crate::devices::Circuit;
use crate::netlist::ast::{FourierCard, MeasureCard, MeasureEvent, MeasureRequest, VectorRequest};
use crate::netlist::eval::ParamScope;
use crate::primitives::{AnalysisKind, Complex, Real, SpiceError, SpiceResult};

mod queries;
use queries::{QueryPlan, QueryRows};

/// A requested observation independent of measurement-card expression bindings.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Caller-provided identity, returned without canonicalizing the name.
    pub name: String,
    /// The same vector grammar and components as output selection.
    pub vector: VectorRequest,
    /// How the accepted samples are reduced.
    pub reduction: ObservationReduction,
}

/// Output shape, not a fitness function. Complex values remain complex unless
/// the requested vector explicitly projects a scalar AC component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ObservationReduction {
    /// Last accepted value; supports single-point operating-point outputs.
    Last,
    /// Linear query at a physical time, frequency or DC scale coordinate.
    /// The scale must be monotonic in either direction; a direction change
    /// makes a single coordinate query ambiguous.
    At(Real),
    /// Selected trace with an explicit maximum number of accepted samples.
    /// No decimation, interpolation or duplicate-coordinate merging.
    Trace {
        /// Maximum retained samples.
        max_points: usize,
    },
}

/// A result sample, with the physical coordinate when the plot has one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservationPoint {
    /// Time, frequency or DC sweep coordinate, if applicable.
    pub coordinate: Option<Real>,
    /// Projected complex or scalar value.
    pub value: Complex,
}

/// Typed value data, suitable for a caller's objective or constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum ObservationValue {
    /// Single evaluated value.
    Scalar(ObservationPoint),
    /// Stored accepted samples, in driver order.
    Trace(Vec<ObservationPoint>),
}

/// An individual observation failed to produce a usable value.
#[derive(Debug, Clone, PartialEq)]
pub enum ObservationFailure {
    /// Query lies outside the accepted coordinate range.
    OutOfRange {
        /// Requested coordinate.
        at: Real,
        /// Actual endpoints, or None for an empty stream.
        range: Option<(Real, Real)>,
    },
    /// Coincident samples disagree at the requested coordinate.
    Discontinuous,
    /// No accepted point to return.
    Empty,
    /// Trace storage would exceed its declared limit.
    TraceLimit {
        /// Maximum allowed sample count.
        max_points: usize,
    },
    /// Bad coordinate, operand or allocation; preserve the underlying error.
    Invalid(SpiceError),
}

/// One ordered result. Duplicate request names retain independent entries.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationResult {
    /// Caller-provided identity.
    pub name: String,
    /// Resolved vector unit.
    pub unit: String,
    /// The value, or explicit unusable-result status.
    pub value: Result<ObservationValue, ObservationFailure>,
}

/// Requirements for one expected output plot. Unrequested outputs get an
/// empty specification rather than disappearing from the ordinal schedule.
#[derive(Debug, Clone, Default)]
pub struct ObjectiveRequests {
    /// Measurement cards in source order (vector pass, then scalar pass).
    pub measurements: Vec<MeasureCard>,
    /// Caller objectives in their requested order.
    pub observations: Vec<Observation>,
    /// Fourier transforms using the existing final-period semantics.
    pub fourier: Vec<FourierCard>,
    /// Maximum raw projected cells retained for operations requiring history.
    /// Zero rejects any positive-size retained history; streaming-only plans
    /// need no nonzero limit. Trace limits are specified separately.
    pub retained_cell_limit: usize,
}

struct BoundObservation {
    name: String,
    column: Column,
    reduction: ObservationReduction,
}

/// Last-value expressions are evaluated once, after the final accepted
/// sample. Only their shared raw columns and optional physical scale need
/// to survive each point callback.
struct LastProjection {
    columns: Vec<usize>,
    axis: Option<usize>,
}

impl LastProjection {
    fn bind(
        observations: &mut [BoundObservation],
        trace_axis: Option<usize>,
    ) -> SpiceResult<Option<Self>> {
        let mut columns = Vec::new();
        let mut present = false;
        for observation in observations.iter() {
            if matches!(observation.reduction, ObservationReduction::Last) {
                present = true;
                columns.extend(observation.column.input_columns());
            }
        }
        if !present {
            return Ok(None);
        }
        columns.extend(trace_axis);
        columns.sort_unstable();
        columns.dedup();
        for observation in observations.iter_mut() {
            if matches!(observation.reduction, ObservationReduction::Last) {
                observation.column.remap_inputs(&columns)?;
            }
        }
        let axis = trace_axis
            .map(|index| {
                columns
                    .binary_search(&index)
                    .map_err(|_| invalid("missing last observation scale"))
            })
            .transpose()?;
        Ok(Some(Self { columns, axis }))
    }
}

struct Projection {
    header: Plot,
    columns: Vec<usize>,
    limit: usize,
}

/// Bound objective requirements for one output in the scenario schedule.
pub struct PlotObjectivePlan {
    kind: AnalysisKind,
    source: Plot,
    measurements: Vec<MeasureCard>,
    reductions: Vec<Option<BoundMeasure>>,
    observations: Vec<BoundObservation>,
    queries: QueryPlan,
    last: Option<LastProjection>,
    traces: Vec<usize>,
    fourier: Vec<FourierCard>,
    axis: Option<AxisTracker>,
    trace_axis: Option<usize>,
    projection: Option<Projection>,
    required: Vec<usize>,
}

fn invalid(message: impl Into<String>) -> SpiceError {
    SpiceError::circuit(message)
}

fn requested_columns(
    header: &Plot,
    request: &MeasureRequest,
    indices: &mut Vec<usize>,
) -> SpiceResult<()> {
    fn event(
        event: &MeasureEvent,
        vector: &mut impl FnMut(&VectorRequest) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        let mut event = event;
        while let MeasureEvent::Delayed { event: inner, .. } = event {
            event = inner;
        }
        match event {
            MeasureEvent::At { .. } => Ok(()),
            MeasureEvent::Crossing { operand, .. } => vector(operand),
            MeasureEvent::VectorCrossing {
                operand, reference, ..
            } => {
                vector(operand)?;
                vector(reference)
            }
            MeasureEvent::Delayed { .. } => Err(invalid("invalid delayed event")),
        }
    }
    let mut vector = |request: &VectorRequest| -> SpiceResult<()> {
        indices.extend(resolve_request(header, request)?.input_columns());
        Ok(())
    };
    let mut request = request;
    while let MeasureRequest::Deferred { template, .. } = request {
        request = template;
    }
    match request {
        MeasureRequest::Find { operand, .. } | MeasureRequest::Statistic { operand, .. } => {
            vector(operand)
        }
        MeasureRequest::AtEvent {
            operand, event: e, ..
        } => {
            vector(operand)?;
            event(e, &mut vector)
        }
        MeasureRequest::When { event: e, .. } => event(e, &mut vector),
        MeasureRequest::TrigTarg { trig, targ, .. } => {
            event(trig, &mut vector)?;
            event(targ, &mut vector)
        }
        MeasureRequest::Parameter(_) => Ok(()),
        MeasureRequest::Deferred { .. } => Err(invalid("invalid deferred measurement template")),
    }
}

fn bind_projection(
    header: &Plot,
    kind: AnalysisKind,
    mut retained: Vec<usize>,
    needs_history: bool,
    retained_cell_limit: usize,
    required: &mut Vec<usize>,
) -> SpiceResult<Option<Projection>> {
    Ok(if needs_history {
        let name = measure::axis_name(header, kind)?;
        let column = header
            .variable_index(name)
            .ok_or_else(|| invalid("retained objective requires its physical axis"))?;
        retained.push(column);
        retained.sort_unstable();
        retained.dedup();
        required.extend(retained.iter().copied());
        let projected = Plot::new(
            header.name.clone(),
            header.plotname.clone(),
            header.flags(),
            retained
                .iter()
                .map(|&index| header.variables()[index].clone())
                .collect(),
        );
        Some(Projection {
            header: projected,
            columns: retained,
            limit: retained_cell_limit,
        })
    } else {
        None
    })
}

fn bind_axes(
    header: &Plot,
    kind: AnalysisKind,
    has_measurements: bool,
    observations: &[Observation],
) -> SpiceResult<(Option<AxisTracker>, Option<usize>)> {
    // Direct observations are not restricted to the .measure grammar.
    // Frequency-bearing noise, distortion and AC sensitivity plots use
    // their physical scale; integrated/one-point plots have none.
    let scale = measure::axis_name(header, kind).ok().or_else(|| {
        (matches!(
            kind,
            AnalysisKind::Noise | AnalysisKind::Distortion | AnalysisKind::Sensitivity
        ) && header.variable_index("frequency").is_some())
        .then_some("frequency")
    });
    let axis = if has_measurements {
        Some(AxisTracker::bind(header, kind)?)
    } else if observations
        .iter()
        .any(|request| matches!(request.reduction, ObservationReduction::At(_)))
    {
        Some(AxisTracker::bind_named(
            header,
            scale.ok_or_else(|| invalid("point query requires a physical scale"))?,
        )?)
    } else {
        None
    };
    let trace_axis = scale.and_then(|name| header.variable_index(name));
    Ok((axis, trace_axis))
}

impl PlotObjectivePlan {
    /// Bind one output's requirements against its immutable column schema.
    /// The header may come from a prepared schema or a representative run;
    /// its rows are never copied into the plan.
    ///
    /// # Errors
    /// Missing vectors/scales, wrong analysis kinds, invalid static query or
    /// record limits. A later run with a different schema is rejected.
    pub fn bind(
        header: &Plot,
        kind: AnalysisKind,
        requests: ObjectiveRequests,
    ) -> SpiceResult<Self> {
        let ObjectiveRequests {
            measurements,
            observations,
            fourier,
            retained_cell_limit,
        } = requests;
        measure::bounded_count(measurements.iter())?;
        let (axis, trace_axis) = bind_axes(header, kind, !measurements.is_empty(), &observations)?;
        let mut required = Vec::new();
        if let Some(axis) = &axis {
            required.push(axis.column());
        }
        if !observations.is_empty()
            && let Some(column) = trace_axis
        {
            required.push(column);
        }
        let mut retained = Vec::new();
        let mut reductions = Vec::with_capacity(measurements.len());
        let mut needs_history = !fourier.is_empty();
        for card in &measurements {
            if card.analysis != kind {
                return Err(SpiceError::Unsupported {
                    feature: format!("measurement {} names a different analysis", card.name),
                    location: Some(card.analysis_location.clone()),
                });
            }
            let bound = BoundMeasure::bind(header, card)?;
            if let Some(bound) = &bound {
                bound.inputs(&mut required);
                // Validate static thresholds / counts at plan construction.
                bound.start(card)?;
            } else if !matches!(card.request, MeasureRequest::Parameter(_)) {
                needs_history = true;
                requested_columns(header, &card.request, &mut retained)?;
            }
            reductions.push(bound);
        }
        let mut observations = observations
            .into_iter()
            .map(|request| {
                if let ObservationReduction::At(at) = request.reduction
                    && !at.is_finite()
                {
                    return Err(invalid("objective query coordinate must be finite"));
                }
                let column = resolve_request(header, &request.vector)?;
                required.extend(column.input_columns());
                Ok(BoundObservation {
                    name: request.name,
                    column,
                    reduction: request.reduction,
                })
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        let queries = QueryPlan::bind(&mut observations)?;
        let last = LastProjection::bind(&mut observations, trace_axis)?;
        let traces = observations
            .iter()
            .enumerate()
            .filter_map(|(index, observation)| {
                matches!(observation.reduction, ObservationReduction::Trace { .. }).then_some(index)
            })
            .collect();
        if !fourier.is_empty() {
            if kind != AnalysisKind::Transient {
                return Err(invalid("Fourier objectives require a transient plot"));
            }
            for card in &fourier {
                for vector in &card.vectors {
                    retained.extend(resolve_request(header, vector)?.input_columns());
                }
            }
        }
        let projection = bind_projection(
            header,
            kind,
            retained,
            needs_history,
            retained_cell_limit,
            &mut required,
        )?;
        required.sort_unstable();
        required.dedup();
        Ok(Self {
            kind,
            source: header.header(),
            measurements,
            reductions,
            observations,
            queries,
            last,
            traces,
            fourier,
            axis,
            trace_axis,
            projection,
            required,
        })
    }

    /// Ordered raw observation indices required by the objectives, including
    /// the scale whenever interpolation or trace metadata needs it. This is
    /// the producer boundary for selective loading/output in #120.
    #[must_use]
    pub fn required_columns(&self) -> &[usize] {
        &self.required
    }

    /// Raw indices that must be retained for postprocessing. Empty for purely
    /// online reductions, regardless of the number of accepted points.
    #[must_use]
    pub fn retained_columns(&self) -> &[usize] {
        self.projection
            .as_ref()
            .map_or(&[], |projection| &projection.columns)
    }

    /// The expected source header, with no rows.
    #[must_use]
    pub const fn header(&self) -> &Plot {
        &self.source
    }
}

impl BoundObservation {
    fn point(
        &self,
        row: &[Complex],
        axis: Option<usize>,
    ) -> Result<ObservationPoint, ObservationFailure> {
        let coordinate = axis.map(|index| row[index]);
        if coordinate.is_some_and(|value| !value.is_finite()) {
            return Err(ObservationFailure::Invalid(invalid(
                "nonfinite observation coordinate",
            )));
        }
        Ok(ObservationPoint {
            coordinate: coordinate.map(|value| value.re),
            value: self
                .column
                .value(row)
                .map_err(ObservationFailure::Invalid)?,
        })
    }
}

enum ObservationState {
    Last,
    Query(PointQuery),
    Trace(Result<Vec<ObservationPoint>, ObservationFailure>),
}

impl ObservationState {
    const fn new(bound: &BoundObservation) -> Self {
        match bound.reduction {
            ObservationReduction::Last => Self::Last,
            ObservationReduction::At(at) => Self::Query(PointQuery::new(at, false)),
            ObservationReduction::Trace { .. } => Self::Trace(Ok(Vec::new())),
        }
    }

    fn observe(
        &mut self,
        bound: &BoundObservation,
        row: &[Complex],
        axis: Option<Real>,
        trace_axis: Option<usize>,
    ) {
        match self {
            Self::Last => {}
            Self::Query(query) => {
                if let Some(x) = axis {
                    query.observe(x, || bound.column.value(row));
                }
            }
            Self::Trace(trace) => {
                let Ok(values) = trace else {
                    return;
                };
                let ObservationReduction::Trace { max_points } = bound.reduction else {
                    return;
                };
                let result = if values.len() == max_points {
                    Err(ObservationFailure::TraceLimit { max_points })
                } else {
                    bound.point(row, trace_axis).and_then(|point| {
                        values.try_reserve(1).map_err(|error| {
                            ObservationFailure::Invalid(invalid(format!(
                                "allocating objective trace: {error}"
                            )))
                        })?;
                        values.push(point);
                        Ok(())
                    })
                };
                if let Err(error) = result {
                    *trace = Err(error);
                }
            }
        }
    }

    fn finish(
        self,
        bound: &BoundObservation,
        axis: Option<&AxisTracker>,
        last: Option<(&[Complex], Option<usize>)>,
    ) -> ObservationResult {
        let value = match self {
            Self::Last => last
                .ok_or(ObservationFailure::Empty)
                .and_then(|(row, axis)| bound.point(row, axis))
                .map(ObservationValue::Scalar),
            Self::Trace(trace) => trace.map(ObservationValue::Trace),
            Self::Query(query) => (|| {
                let ObservationReduction::At(at) = bound.reduction else {
                    return Err(ObservationFailure::Invalid(invalid(
                        "objective state mismatch",
                    )));
                };
                let range = axis
                    .ok_or_else(|| ObservationFailure::Invalid(invalid("missing objective axis")))?
                    .query_range()
                    .map_err(ObservationFailure::Invalid)?;
                query
                    .finish()
                    .map(|value| {
                        ObservationValue::Scalar(ObservationPoint {
                            coordinate: Some(at),
                            value,
                        })
                    })
                    .map_err(|error| match error {
                        QueryFailure::Outside => ObservationFailure::OutOfRange { at, range },
                        QueryFailure::Discontinuous => ObservationFailure::Discontinuous,
                        QueryFailure::Invalid(error) => ObservationFailure::Invalid(error),
                    })
            })(),
        };
        ObservationResult {
            name: bound.name.clone(),
            unit: bound.column.unit().into(),
            value,
        }
    }
}

/// Results of a completed output.
///
/// Measurement bindings are transactional
/// within a plot; a failed vector/parameter publishes no partial measurement
/// namespace. Independent observations and Fourier results remain accessible.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotObjectives {
    /// Measurement records in source order, or their first evaluation error.
    pub measurements: SpiceResult<Vec<Measurement>>,
    /// Scalar/trace observations in request order.
    pub observations: Vec<ObservationResult>,
    /// Fourier results, or a typed postprocessing error.
    pub fourier: SpiceResult<Vec<FourierAnalysis>>,
    /// Number of accepted points seen.
    pub accepted_points: usize,
    /// Peak number of cells held in the raw history projection, excluding
    /// explicitly returned traces. Zero for streaming-only objectives.
    pub retained_cells: usize,
}

/// Mutable row state, independent of where the immutable plan is stored.
pub(in crate::analysis) struct ActivePlot {
    axis: Option<AxisTracker>,
    measurements: Vec<Option<MeasureState>>,
    observations: Vec<ObservationState>,
    query_rows: QueryRows,
    last_row: Vec<Complex>,
    retained: SpiceResult<Option<Plot>>,
    scratch: Vec<Complex>,
    rows: usize,
    peak_retained: usize,
}

impl ActivePlot {
    pub(in crate::analysis) fn new(plan: &PlotObjectivePlan) -> SpiceResult<Self> {
        let measurements = plan
            .reductions
            .iter()
            .zip(&plan.measurements)
            .map(|(bound, card)| bound.as_ref().map(|bound| bound.start(card)).transpose())
            .collect::<SpiceResult<Vec<_>>>()?;
        let observations = plan
            .observations
            .iter()
            .map(ObservationState::new)
            .collect();
        Ok(Self {
            axis: plan.axis.as_ref().map(AxisTracker::fresh),
            measurements,
            observations,
            query_rows: QueryRows::new(&plan.queries),
            last_row: Vec::with_capacity(plan.last.as_ref().map_or(0, |last| last.columns.len())),
            retained: Ok(plan
                .projection
                .as_ref()
                .map(|projection| projection.header.header())),
            scratch: Vec::with_capacity(plan.retained_columns().len()),
            rows: 0,
            peak_retained: 0,
        })
    }

    pub(in crate::analysis) fn point(
        &mut self,
        plan: &PlotObjectivePlan,
        row: &[Complex],
    ) -> SpiceResult<()> {
        plan.source.check_point(row)?;
        self.rows = self
            .rows
            .checked_add(1)
            .ok_or_else(|| lifecycle_error("objective point count overflows usize"))?;
        let axis = self.axis.as_mut().and_then(|axis| axis.observe(row));
        if let Some(x) = axis
            && self
                .axis
                .as_ref()
                .is_some_and(AxisTracker::is_nondecreasing)
        {
            for ((state, bound), card) in self
                .measurements
                .iter_mut()
                .zip(&plan.reductions)
                .zip(&plan.measurements)
            {
                if let (Some(state), Some(bound)) = (state, bound) {
                    state.observe(bound, card, x, row);
                }
            }
        }
        if let Some(last) = &plan.last {
            self.last_row.clear();
            self.last_row
                .extend(last.columns.iter().map(|&index| row[index]));
        }
        for &index in &plan.traces {
            self.observations[index].observe(&plan.observations[index], row, None, plan.trace_axis);
        }
        if let Some(x) = axis {
            self.query_rows.observe(
                &plan.queries,
                &plan.observations,
                &mut self.observations,
                x,
                row,
            );
        }
        if let (Ok(Some(retained)), Some(projection)) = (&mut self.retained, &plan.projection) {
            let next = retained.cells().len().checked_add(projection.columns.len());
            if next.is_none_or(|next| next > projection.limit) {
                self.retained = Err(invalid(format!(
                    "objective retained-cell limit of {} exceeded",
                    projection.limit
                )));
            } else {
                self.scratch.clear();
                self.scratch
                    .extend(projection.columns.iter().map(|&index| row[index]));
                match retained.push_point(&self.scratch) {
                    Ok(()) => self.peak_retained = next.unwrap_or(0),
                    Err(error) => self.retained = Err(error),
                }
            }
        }
        Ok(())
    }

    pub(in crate::analysis) fn finish(
        self,
        plan: &PlotObjectivePlan,
        scope: &ParamScope,
        settings: Option<fourier::FourierSettings>,
    ) -> PlotObjectives {
        let measurements = (|| {
            if plan.measurements.is_empty() {
                return Ok(Vec::new());
            }
            let axis = self
                .axis
                .as_ref()
                .ok_or_else(|| invalid("missing measurement axis"))?;
            // Axis failures precede every individual measurement, including
            // PARAM-only plots. Resolve deferred values in the original two
            // passes; do not publish a result until evaluation succeeds.
            axis.range()?;
            let mut online = self
                .measurements
                .into_iter()
                .zip(&plan.reductions)
                .zip(&plan.measurements)
                .map(|((state, bound), card)| {
                    state
                        .zip(bound.as_ref())
                        .map(|(state, bound)| state.finish(bound, card, axis))
                })
                .collect::<Vec<_>>();
            let mut retained_axis = None;
            measure::resolve_ordered(plan.kind, plan.measurements.iter(), scope, |index, card| {
                if let Some(result) = online[index].take() {
                    return result;
                }
                let retained = self
                    .retained
                    .as_ref()
                    .map_err(Clone::clone)?
                    .as_ref()
                    .ok_or_else(|| invalid("measurement requires retained samples"))?;
                if retained_axis.is_none() {
                    retained_axis = Some(Axis::of(retained, plan.kind)?);
                }
                let axis = retained_axis
                    .as_ref()
                    .ok_or_else(|| invalid("missing retained axis"))?;
                measure::evaluate(retained, axis, card)
            })
        })();
        let last = plan
            .last
            .as_ref()
            .filter(|_| self.rows != 0)
            .map(|plan| (self.last_row.as_slice(), plan.axis));
        let observations = self
            .observations
            .into_iter()
            .zip(&plan.observations)
            .map(|(state, bound)| state.finish(bound, self.axis.as_ref(), last))
            .collect();
        let fourier = (|| {
            if plan.fourier.is_empty() {
                return Ok(Vec::new());
            }
            let plot = self
                .retained
                .as_ref()
                .map_err(Clone::clone)?
                .as_ref()
                .ok_or_else(|| invalid("Fourier requires retained samples"))?;
            let cards = fourier::resolve_cards(&plan.fourier, scope)?;
            match settings {
                Some(settings) => fourier::resolve_with_settings(plot, plan.kind, &cards, settings),
                None => fourier::resolve(plot, plan.kind, &cards),
            }
        })();
        PlotObjectives {
            measurements,
            observations,
            fourier,
            accepted_points: self.rows,
            retained_cells: self.peak_retained,
        }
    }
}

/// Immutable ordinal plan for a scenario consisting of one or more analyses.
///
/// Independent scenarios can each bind once and reuse their plans for any
/// number of model/parameter candidates with the same output schemas.
pub struct ScenarioPlan {
    plots: Vec<PlotObjectivePlan>,
}

impl ScenarioPlan {
    /// Bind the ordered expected plots. Include ignored bias/summary outputs
    /// with empty requests; never route by a plot-name substring.
    #[must_use]
    pub const fn new(plots: Vec<PlotObjectivePlan>) -> Self {
        Self { plots }
    }

    /// Plans in the same order in which the sink must receive them.
    #[must_use]
    pub fn plots(&self) -> &[PlotObjectivePlan] {
        &self.plots
    }

    /// New independent candidate state. No saved measurements, traces or
    /// partially evaluated namespace survives a previous candidate.
    #[must_use]
    pub fn sink<'a>(&'a self, scope: &'a ParamScope) -> ScenarioSink<'a> {
        ScenarioSink {
            plan: self,
            scope,
            active: None,
            results: Vec::with_capacity(self.plots.len()),
            failed: false,
        }
    }

    /// Execute analyses against a prepared candidate, in caller-supplied
    /// schedule order. Solver errors (including failed convergence) abort
    /// the candidate; no partial result vector is returned.
    ///
    /// # Errors
    /// Underlying analysis or output errors, including unexpected schemas,
    /// incomplete plot lifecycles or failed numerical solves.
    pub fn run(
        &self,
        analyses: &[AnalysisPlan],
        circuit: &mut Circuit,
        context: &AnalysisContext,
        scope: &ParamScope,
    ) -> Result<Vec<PlotObjectives>, ScenarioFailure> {
        let mut sink = self.sink(scope);
        for (ordinal, analysis) in analyses.iter().enumerate() {
            if let Err(error) = analysis.run_into(circuit, context, &mut sink) {
                return Err(if sink.failed {
                    ScenarioFailure::Output(error)
                } else {
                    ScenarioFailure::Execution {
                        analysis: ordinal,
                        error,
                    }
                });
            }
        }
        sink.finish().map_err(ScenarioFailure::Output)
    }
}

/// A scenario that did not finish its analysis/output lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScenarioFailure {
    /// Solver or circuit error, including nonconvergence. Preserve the
    /// actual `SpiceError` rather than interpreting diagnostic message text.
    Execution {
        /// Index in the ordered analysis input list.
        analysis: usize,
        /// Original failure.
        error: SpiceError,
    },
    /// The producer or consumer violated the bound output contract.
    Output(SpiceError),
}

/// Multi-plot objective sink. A failure poisons the sink, so a caller cannot
/// accidentally retrieve a prefix and mistake it for a completed candidate.
pub struct ScenarioSink<'a> {
    plan: &'a ScenarioPlan,
    scope: &'a ParamScope,
    active: Option<ActivePlot>,
    results: Vec<PlotObjectives>,
    failed: bool,
}

impl ScenarioSink<'_> {
    /// Complete the candidate only after every expected plot ended.
    ///
    /// # Errors
    /// Unexpected/missing plots, active unfinished plot, or a poisoned sink.
    pub fn finish(self) -> SpiceResult<Vec<PlotObjectives>> {
        if self.failed || self.active.is_some() || self.results.len() != self.plan.plots.len() {
            return Err(lifecycle_error(
                "objective scenario did not complete every planned plot",
            ));
        }
        Ok(self.results)
    }

    const fn checked(&mut self, result: SpiceResult<()>) -> SpiceResult<()> {
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

impl PointSink for ScenarioSink<'_> {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        let result = (|| {
            if self.failed || self.active.is_some() || header.point_count() != 0 {
                return Err(lifecycle_error("invalid objective plot begin"));
            }
            let plan = self
                .plan
                .plots
                .get(self.results.len())
                .ok_or_else(|| lifecycle_error("unexpected objective plot"))?;
            if header.schema() != plan.source.schema()
                || header.name != plan.source.name
                || header.plotname != plan.source.plotname
            {
                return Err(lifecycle_error("objective plot identity or schema changed"));
            }
            self.active = Some(ActivePlot::new(plan)?);
            Ok(())
        })();
        self.checked(result)
    }

    fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        let result = if self.failed {
            Err(lifecycle_error("objective sink is poisoned"))
        } else {
            self.active
                .as_mut()
                .ok_or_else(|| lifecycle_error("objective row without an active plot"))
                .and_then(|active| active.point(&self.plan.plots[self.results.len()], values))
        };
        self.checked(result)
    }

    fn end(&mut self) -> SpiceResult<()> {
        let result = (|| {
            if self.failed {
                return Err(lifecycle_error("objective sink is poisoned"));
            }
            let active = self
                .active
                .take()
                .ok_or_else(|| lifecycle_error("objective completion without active plot"))?;
            self.results.push(active.finish(
                &self.plan.plots[self.results.len()],
                self.scope,
                None,
            ));
            Ok(())
        })();
        self.checked(result)
    }
}
