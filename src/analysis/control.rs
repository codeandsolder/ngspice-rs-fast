//! Explicit reusable simulation sessions for scripted analysis.
//!
//! Parameter changes share the optimization backend's validated numeric
//! bindings. Each independent run resets solver state but retains prepared
//! topology; output ownership belongs to the caller's point sink.

use super::optimization::parameters::{CandidateTopology, CircuitCandidate, CircuitParameterPlan};
use super::{AnalysisInput, AnalysisPlan, Plot, PointSink};
use crate::netlist::eval::{EvalBudget, EvalLimits};
use crate::primitives::{Real, SpiceError, SpiceResult};

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
        let index = self
            .plan
            .inputs()
            .iter()
            .position(|input| input.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| SpiceError::circuit(format!("unknown control parameter {name}")))?;
        let previous = self.values[index];
        self.values[index] = value;
        if let Err(error) = self
            .candidate
            .update(&self.values, &mut EvalBudget::new(self.limits))
        {
            self.values[index] = previous;
            return Err(error);
        }
        Ok(self.candidate.topology())
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
    /// Use [Self::run_into] for streaming without plot retention.
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
