//! Explicit reusable simulation sessions for scripted analysis.
//!
//! Parameter changes share the optimization backend's validated numeric
//! bindings. Each independent run resets solver state but retains prepared
//! topology; output ownership belongs to the caller's point sink.

use super::batch::{BoundConsumers, OutputRoutes, PlannedOutput};
use super::fourier::{self, FourierSettings};
use super::measure::{self, Measurement};
use super::optimization::parameters::{CandidateTopology, CircuitCandidate, CircuitParameterPlan};
use super::results::PlotCollector;
use super::selection::{RowProjection, Selection};
use super::{AnalysisInput, AnalysisPlan, Plot, PointSink};
use crate::devices::sweep::InstanceOverride;
use crate::netlist::ast::{AnalysisCard, FourierCard, MeasureCard, OutputCards};
use crate::netlist::eval::{EvalBudget, EvalLimits, ParamScope};
use crate::netlist::expr::ParameterExpression;
use crate::primitives::{AnalysisKind, Complex, Real, SpiceError, SpiceResult};

pub mod vectors;

use vectors::TimeGrid;

/// A retained plot with its originating analysis and requested transient grid.
pub struct ControlPlot {
    /// Analysis or postprocessor that produced the plot.
    pub kind: AnalysisKind,
    /// Typed plot and data.
    pub plot: Plot,
    /// Requested output grid for transient linearization, if applicable.
    pub grid: Option<TimeGrid>,
}

/// Shared frontend cards to route for an explicitly requested run.
pub(crate) struct ControlBatchCards<'a> {
    pub output: &'a OutputCards,
    pub measurements: &'a [MeasureCard],
    pub fourier: &'a [FourierCard],
    pub script_measurements: &'a [MeasureCard],
    pub script_observations: &'a [String],
    pub deck_run: bool,
}

pub(crate) struct ControlBatchBudget<'a> {
    pub plots: usize,
    pub cells: usize,
    pub grid: &'a mut usize,
}

pub(crate) struct ControlBatch {
    pub plots: Vec<ControlPlot>,
    pub measurements: Vec<Measurement>,
}

struct ActiveControlOutput {
    projection: RowProjection,
    consumers: BoundConsumers,
}

/// Route the ordinary batch postprocessors and retain only the requested
/// plot projection. This is also a streaming sink: full driver rows need
/// not be accumulated before a measurement or .print can consume them.
struct ControlBatchSink<'a, 'b> {
    planned: &'a [PlannedOutput<'a>],
    select: bool,
    scope: &'a ParamScope,
    settings: Option<FourierSettings>,
    collector: &'b mut PlotCollector,
    measurements: &'b mut Vec<Measurement>,
    emit: &'b mut dyn FnMut(&str) -> SpiceResult<()>,
    active: Option<ActiveControlOutput>,
    next: usize,
}

impl ControlBatchSink<'_, '_> {
    fn current(&self) -> SpiceResult<&PlannedOutput<'_>> {
        self.planned
            .get(self.next)
            .ok_or_else(|| SpiceError::circuit("control driver exceeded its plot schedule"))
    }

    fn selection(&self, header: &Plot) -> SpiceResult<Selection> {
        let planned = self.current()?;
        if self.select {
            planned.selection(header)
        } else {
            Selection::resolve(header, planned.kind, &[])
        }
    }

    fn finish_consumers(&mut self, consumers: BoundConsumers) -> SpiceResult<()> {
        let reports = consumers.finish(self.scope, self.settings, &mut self.emit)?;
        if !reports.measurements.is_empty() {
            (self.emit)(&measure::to_text(&reports.measurements))?;
        }
        if !reports.fourier.is_empty() {
            (self.emit)(&fourier::to_text(&reports.fourier))?;
        }
        self.measurements.extend(reports.measurements);
        self.next += 1;
        Ok(())
    }
}

impl PointSink for ControlBatchSink<'_, '_> {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        if self.active.is_some() {
            return Err(SpiceError::circuit(
                "control driver began a second active plot",
            ));
        }
        let selection = self.selection(header)?;
        let consumers = self.current()?.bind_consumers(header)?;
        consumers.begin_text(&mut self.emit)?;
        self.collector.begin(&selection.header(header)?)?;
        self.active = Some(ActiveControlOutput {
            projection: selection.bind(header)?,
            consumers,
        });
        Ok(())
    }

    fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| SpiceError::circuit("control row without active plot"))?;
        if !active.consumers.is_empty() {
            active.consumers.point(values, &mut self.emit)?;
        }
        self.collector.point(active.projection.project(values)?)
    }

    fn end(&mut self) -> SpiceResult<()> {
        let active = self
            .active
            .take()
            .ok_or_else(|| SpiceError::circuit("control end without active plot"))?;
        self.finish_consumers(active.consumers)?;
        self.collector.end()
    }

    fn complete(&mut self, plot: Plot) -> SpiceResult<()> {
        if self.active.is_some() {
            return Err(SpiceError::circuit(
                "control completed a second active plot",
            ));
        }
        let selection = self.selection(&plot)?;
        let mut consumers = self.current()?.bind_consumers(&plot.header())?;
        consumers.begin_text(&mut self.emit)?;
        if !consumers.is_empty() {
            for row in plot.rows() {
                consumers.point(row, &mut self.emit)?;
            }
        }
        self.finish_consumers(consumers)?;
        self.collector.complete(selection.into_plot(plot)?)
    }
}

/// A mutable simulation session with independently validated parameter inputs.
pub struct ControlSession<'a> {
    plan: &'a CircuitParameterPlan,
    candidate: CircuitCandidate<'a>,
    values: Vec<Real>,
    limits: EvalLimits,
    edits: Vec<InstanceOverride>,
}

impl<'a> ControlSession<'a> {
    /// Bind a reusable circuit and initialize its independent coordinates.
    ///
    /// # Errors
    /// Invalid baseline circuit or parameter bindings.
    pub fn new(plan: &'a CircuitParameterPlan, limits: EvalLimits) -> SpiceResult<Self> {
        Ok(Self {
            plan,
            candidate: plan.session()?,
            values: plan.inputs().iter().map(|input| input.default).collect(),
            limits,
            edits: Vec::new(),
        })
    }

    /// The current value of a bound independent parameter.
    #[must_use]
    pub fn parameter(&self, name: &str) -> Option<Real> {
        self.plan
            .inputs()
            .iter()
            .position(|input| input.name.eq_ignore_ascii_case(name))
            .map(|index| self.values[index])
    }

    /// Transactionally change one independent coordinate. A failed update
    /// preserves the previous parameter vector, configuration and circuit.
    ///
    /// # Errors
    /// Unknown parameter, invalid numeric value or rejected circuit update.
    pub fn alter_parameter(&mut self, name: &str, value: Real) -> SpiceResult<CandidateTopology> {
        self.alter_parameters([(name, value)])
    }

    /// Apply several coordinate changes as one transaction; no intermediate
    /// parameter vector needs to be a valid physical circuit.
    ///
    /// # Errors
    /// Unknown input names or a rejected candidate.
    pub fn alter_parameters<'b>(
        &mut self,
        changes: impl IntoIterator<Item = (&'b str, Real)>,
    ) -> SpiceResult<CandidateTopology> {
        let mut next = self.values.clone();
        for (name, value) in changes {
            let index = self
                .plan
                .inputs()
                .iter()
                .position(|input| input.name.eq_ignore_ascii_case(name))
                .ok_or_else(|| SpiceError::circuit(format!("unknown control parameter {name}")))?;
            next[index] = value;
        }
        self.candidate
            .update(&next, &mut EvalBudget::new(self.limits))?;
        self.values = next;
        if !self.edits.is_empty() {
            let context = self.candidate.model_context();
            self.candidate
                .circuit_mut()
                .set_control_overrides(&context, &[])?;
            self.edits.clear();
        }
        Ok(self.candidate.topology())
    }

    /// Change one elaborated numeric instance parameter without rebuilding
    /// topology. Later edits of the same canonical site replace earlier ones.
    /// A parameter reset restores the deck's parameter-dependent device values.
    ///
    /// # Errors
    /// Unknown device/setter, invalid value, topology-changing edit, or an
    /// edit that would exceed the explicit retained site limit.
    pub fn alter_instance(
        &mut self,
        name: &str,
        parameter: Option<&str>,
        value: Real,
        max_edits: usize,
    ) -> SpiceResult<()> {
        let circuit = self.candidate.circuit();
        let device = circuit
            .device(name)
            .ok_or_else(|| SpiceError::circuit(format!("unknown control instance {name}")))?;
        let parameter = match parameter {
            Some(parameter) => parameter,
            None => match device.designator().to_ascii_lowercase() {
                'r' => "resistance",
                'c' => "capacitance",
                'l' => "inductance",
                'v' | 'i' => "dc",
                'k' => "k",
                _ => {
                    return Err(SpiceError::circuit(format!(
                        "specify an instance parameter for {name}"
                    )));
                }
            },
        };
        let target = if parameter.eq_ignore_ascii_case("dc")
            && matches!(device.designator().to_ascii_lowercase(), 'v' | 'i')
        {
            circuit.bind_source_target(name)?
        } else {
            circuit.bind_instance_target(name, parameter)?
        };
        let mut edits = self.edits.clone();
        edits.retain(|edit| edit.target() != target);
        if edits.len() >= max_edits {
            return Err(SpiceError::circuit("control instance edit budget exceeded"));
        }
        edits.push(target.at(value)?);
        let context = self.candidate.model_context();
        self.candidate
            .circuit_mut()
            .set_control_overrides(&context, &edits)?;
        self.edits = edits;
        Ok(())
    }

    /// The validated candidate parameter scope.
    ///
    /// # Errors
    /// Missing parameter scope, an invalid candidate configuration.
    pub fn scope(&self) -> SpiceResult<&ParamScope> {
        self.candidate
            .config()
            .params()
            .ok_or_else(|| SpiceError::circuit("control candidate has no parameter scope"))
    }

    /// Evaluate a scalar expression over the current validated parameters
    /// and caller-provided script bindings.
    ///
    /// # Errors
    /// Undefined variables, evaluation budget or numerical errors.
    pub fn evaluate(
        &self,
        expression: &ParameterExpression,
        lookup: impl Fn(&str) -> Option<Real>,
        budget: &mut EvalBudget,
    ) -> SpiceResult<Real> {
        self.scope()?.evaluate_with(expression, lookup, budget)
    }

    pub(crate) fn plan(&self, card: &AnalysisCard) -> SpiceResult<AnalysisPlan> {
        self.candidate.config().plan_for(card)
    }

    pub(crate) fn plans(&self, cards: &[AnalysisCard]) -> SpiceResult<Vec<AnalysisPlan>> {
        self.candidate.config().plans_for(cards)
    }

    fn expand_frontend(
        &self,
        cards: &ControlBatchCards<'_>,
        scope: &ParamScope,
    ) -> SpiceResult<Option<crate::devices::subckt::ExpandedNetlist>> {
        let source = self.plan.source();
        if !source.subcircuits.iter().any(|sub| {
            !sub.output.is_empty() || !sub.measurements.is_empty() || !sub.fourier.is_empty()
        }) {
            return Ok(None);
        }
        let literal = crate::netlist::elaborate::literalize_with(
            source,
            std::sync::Arc::new(scope.clone()),
            &mut EvalBudget::new(self.limits),
        )?;
        crate::devices::subckt::expand_with_output(
            &literal.netlist,
            &literal.scope,
            crate::devices::subckt::SubcircuitLimits::default(),
            cards.output,
            cards.measurements,
            cards.fourier,
        )
        .map(Some)
    }

    /// Execute one explicit batch in normal deck order, retaining a bounded
    /// collection of typed plots and streaming requested report consumers.
    pub(crate) fn collect_batch(
        &mut self,
        plans: &[AnalysisPlan],
        cards: ControlBatchCards<'_>,
        budget: ControlBatchBudget<'_>,
        mut emit: impl FnMut(&str) -> SpiceResult<()>,
    ) -> SpiceResult<ControlBatch> {
        let grid = plans.iter().try_fold(0usize, |sum, plan| {
            sum.checked_add(plan.requested_grid_points())
                .ok_or_else(|| SpiceError::circuit("control analysis grid count overflow"))
        })?;
        let remaining = budget
            .grid
            .checked_sub(grid)
            .ok_or_else(|| SpiceError::circuit("control deterministic grid budget exceeded"))?;
        let expected = plans.iter().try_fold(0usize, |sum, plan| {
            sum.checked_add(plan.plot_count())
                .ok_or_else(|| SpiceError::circuit("control plot count overflow"))
        })?;
        if expected > budget.plots {
            return Err(SpiceError::circuit("control plot count budget exceeded"));
        }
        let schedule = super::batch::schedule(plans);
        let scope = self.scope()?.clone();
        let expanded = self.expand_frontend(&cards, &scope)?;
        let (output, measurements, fourier) = expanded.as_ref().map_or(
            (cards.output, cards.measurements, cards.fourier),
            |expanded| {
                (
                    &expanded.output,
                    expanded.measurements.as_slice(),
                    expanded.fourier.as_slice(),
                )
            },
        );
        let direct_output;
        let (output, measurements, fourier) = if cards.deck_run {
            (output, measurements, fourier)
        } else {
            direct_output = OutputCards {
                saves: output.saves.clone(),
                prints: Vec::new(),
            };
            (&direct_output, &[][..], &[][..])
        };
        let fourier = fourier::resolve_cards(fourier, &scope)?;
        let cards = ControlBatchCards {
            output,
            measurements,
            fourier: &fourier,
            ..cards
        };
        let routes = OutputRoutes::new(&schedule, cards.output, cards.measurements, cards.fourier)?;
        *budget.grid = remaining;
        let mut collector = PlotCollector::bounded(budget.plots, budget.cells);
        let mut origins = Vec::with_capacity(expected);
        let mut measurements = Vec::new();
        self.candidate
            .circuit_mut()
            .clear_candidate_analysis_state();
        for entry in &schedule {
            let plan = &plans[entry.card_index];
            let planned = routes.plan_outputs(entry);
            self.prepare_observations(entry.kind, &cards)?;
            let mut sink = ControlBatchSink {
                planned: &planned,
                select: !cards.output.saves.is_empty(),
                scope: &scope,
                settings: self.candidate.config().fourier_settings(),
                collector: &mut collector,
                measurements: &mut measurements,
                emit: &mut emit,
                active: None,
                next: 0,
            };
            self.execute_prepared(plan, &mut sink)?;
            self.candidate.circuit_mut().note_completed(entry.kind);
            if sink.active.is_some() || sink.next != planned.len() {
                return Err(SpiceError::circuit(
                    "control driver did not complete planned plots",
                ));
            }
            for index in 0..plan.plot_count() {
                let kind = if index == 0 && plan.retains_bias() {
                    AnalysisKind::OperatingPoint
                } else {
                    entry.kind
                };
                let grid = if kind == AnalysisKind::Transient {
                    plan.transient().map(|settings| TimeGrid {
                        start: settings.start(),
                        stop: settings.stop(),
                        step: settings.step(),
                    })
                } else {
                    None
                };
                origins.push((kind, grid));
            }
        }
        let plots = origins
            .into_iter()
            .zip(collector.finish(expected)?)
            .map(|((kind, grid), plot)| ControlPlot { kind, plot, grid })
            .collect();
        Ok(ControlBatch {
            plots,
            measurements,
        })
    }

    fn prepare_observations(
        &mut self,
        kind: AnalysisKind,
        cards: &ControlBatchCards<'_>,
    ) -> SpiceResult<()> {
        let mut names =
            super::observations::requests(kind, cards.output, cards.measurements, cards.fourier);
        names.extend(super::observations::requests(
            kind,
            &OutputCards::default(),
            cards.script_measurements,
            &[],
        ));
        let supported = matches!(
            kind,
            AnalysisKind::OperatingPoint
                | AnalysisKind::DcSweep
                | AnalysisKind::Ac
                | AnalysisKind::Transient
        );
        if supported {
            names.extend_from_slice(cards.script_observations);
        }
        let circuit = self.candidate.circuit_mut();
        circuit.set_observations(&names)?;
        if !supported && !circuit.observations().is_empty() {
            return Err(SpiceError::not_yet_ported(
                "device observations for this analysis",
                "src/frontend/outitf.c (OUTpData)",
            ));
        }
        Ok(())
    }

    /// Run a requested analysis from independent initial state, streaming
    /// directly to the supplied consumer.
    ///
    /// # Errors
    /// Analysis planning, solver or consumer failure.
    pub fn run_into(&mut self, input: AnalysisInput, sink: &mut dyn PointSink) -> SpiceResult<()> {
        let plan = self.candidate.config().plan(input)?;
        self.execute(&plan, sink)
    }

    /// Retain all output plots from one run under explicit plot and cell limits.
    /// Use [`Self::run_into`] for streaming without plot retention.
    ///
    /// # Errors
    /// Planning, simulation or plot-retention budget failure.
    pub fn run(
        &mut self,
        input: AnalysisInput,
        max_plots: usize,
        max_cells: usize,
    ) -> SpiceResult<Vec<Plot>> {
        let plan = self.candidate.config().plan(input)?;
        let expected = plan.plot_count();
        let mut collector = super::results::PlotCollector::bounded(max_plots, max_cells);
        self.execute(&plan, &mut collector)?;
        collector.finish(expected)
    }

    fn execute(&mut self, plan: &AnalysisPlan, sink: &mut dyn PointSink) -> SpiceResult<()> {
        self.candidate
            .circuit_mut()
            .clear_candidate_analysis_state();
        self.execute_prepared(plan, sink)
    }

    fn execute_prepared(
        &mut self,
        plan: &AnalysisPlan,
        sink: &mut dyn PointSink,
    ) -> SpiceResult<()> {
        let mut context = self.candidate.analysis_context();
        if !self.edits.is_empty() {
            // Persistent edits already include all candidate overlays. The
            // remaining context slots belong to the individual sweep point.
            context.candidate_model_overrides = None;
        }
        plan.run_into(self.candidate.circuit_mut(), &context, sink)
    }
}
