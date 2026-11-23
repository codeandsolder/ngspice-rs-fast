//! Explicit reusable simulation sessions for scripted analysis.
//!
//! Parameter changes share the optimization backend's validated numeric
//! bindings. Each independent run resets solver state but retains prepared
//! topology; output ownership belongs to the caller's point sink.

use super::optimization::parameters::{CandidateTopology, CircuitCandidate, CircuitParameterPlan};
use super::{AnalysisInput, AnalysisPlan, Plot, PointSink};
use crate::netlist::ast::AnalysisCard;
use crate::netlist::eval::{EvalBudget, EvalLimits, ParamScope};
use crate::netlist::expr::ParameterExpression;
use crate::primitives::{AnalysisKind, Real, SpiceError, SpiceResult};

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

/// A mutable simulation session with independently validated parameter inputs.
pub struct ControlSession<'a> {
    plan: &'a CircuitParameterPlan,
    candidate: CircuitCandidate<'a>,
    values: Vec<Real>,
    limits: EvalLimits,
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
        Ok(self.candidate.topology())
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

    /// Execute one explicit batch in normal deck order, retaining a bounded
    /// collection of typed plots. The cumulative grid budget is charged before
    /// invoking any driver, including when the numerical run later fails.
    pub(crate) fn collect_batch(
        &mut self,
        plans: &[AnalysisPlan],
        max_plots: usize,
        max_cells: usize,
        grid_budget: &mut usize,
    ) -> SpiceResult<Vec<ControlPlot>> {
        let grid = plans.iter().try_fold(0usize, |sum, plan| {
            sum.checked_add(plan.requested_grid_points())
                .ok_or_else(|| SpiceError::circuit("control analysis grid count overflow"))
        })?;
        let remaining = grid_budget
            .checked_sub(grid)
            .ok_or_else(|| SpiceError::circuit("control deterministic grid budget exceeded"))?;
        let expected = plans.iter().try_fold(0usize, |sum, plan| {
            sum.checked_add(plan.plot_count())
                .ok_or_else(|| SpiceError::circuit("control plot count overflow"))
        })?;
        if expected > max_plots {
            return Err(SpiceError::circuit("control plot count budget exceeded"));
        }
        *grid_budget = remaining;
        let mut collector = super::results::PlotCollector::bounded(max_plots, max_cells);
        let mut origins = Vec::with_capacity(expected);
        for entry in super::batch::schedule(plans) {
            let plan = &plans[entry.card_index];
            self.execute(plan, &mut collector)?;
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
        Ok(origins
            .into_iter()
            .zip(collector.finish(expected)?)
            .map(|((kind, grid), plot)| ControlPlot { kind, plot, grid })
            .collect())
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
        let context = self.candidate.analysis_context();
        self.candidate
            .circuit_mut()
            .clear_candidate_analysis_state();
        plan.run_into(self.candidate.circuit_mut(), &context, sink)
    }
}
