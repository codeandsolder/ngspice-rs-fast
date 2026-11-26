//! Exact, unintegrated real residuals for a memoryless physical DAE.
//!
//! The same prepared devices and original MNA coordinates are used for both
//! F(x,t) and Q(x,t), so the equation is F + dQ/dt = 0. No integration
//! coefficient, time discretization, device limiting, or artificial gmin
//! is introduced by this evaluator.

use crate::devices::{
    AnalysisMode, Forcing, Limit, LoadRequest, ModelContext, PreparedRun, StateHistory,
    TransientTiming, TrialState,
};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Exact physical residuals and their Jacobians with respect to the state.
pub struct PhysicalDaePoint {
    /// Static physical residual, including independent time forcing.
    pub residual: Vector,
    /// Signed charge and flux by equation row.
    pub charge: Vector,
    /// Jacobian of the static residual.
    pub conductance: SparseMatrix,
    /// Jacobian of charge and flux.
    pub capacitance: SparseMatrix,
}

/// A topology-bound evaluator reusing the static matrix, charge operator,
/// and disposable trial storage. Evaluating a point never publishes history.
pub struct PhysicalDae<'run, 'circuit> {
    run: &'run PreparedRun<'circuit>,
    model: ModelContext,
    timing: TransientTiming,
    linear: crate::devices::linear::LinearSystem,
    history: StateHistory,
    trial: Option<TrialState>,
    rhs: Vector,
    point: PhysicalDaePoint,
}

impl<'run, 'circuit> PhysicalDae<'run, 'circuit> {
    pub(super) fn new(
        run: &'run PreparedRun<'circuit>,
        model: ModelContext,
        timing: TransientTiming,
        mut linear: crate::devices::linear::LinearSystem,
    ) -> SpiceResult<Self> {
        let circuit = run.circuit();
        let n = circuit.unknown_count();
        if model.source_factor.to_bits() != 1_f64.to_bits() {
            return Err(SpiceError::circuit(
                "physical DAE requires an unscaled source context",
            ));
        }
        linear.bind_transient_timing(&timing)?;
        Ok(Self {
            run,
            model,
            timing,
            linear,
            history: circuit.independent_state_history()?,
            trial: None,
            rhs: Vector::zeros(n),
            point: PhysicalDaePoint {
                residual: Vector::zeros(n),
                charge: Vector::zeros(n),
                conductance: circuit.new_matrix()?,
                capacitance: SparseMatrix::new(n, n),
            },
        })
    }

    /// Analytic coefficients of the independent forcing in the same
    /// physical equation coordinates as the residual.
    ///
    /// # Errors
    /// As [`crate::devices::linear::LinearSystem::transient_taylor_rhs`].
    pub fn forcing_taylor_rhs(
        &self,
        time: Real,
        scale: Real,
        order: usize,
        limit: Limit,
    ) -> SpiceResult<Vec<Vector>> {
        self.linear.transient_taylor_rhs(time, scale, order, limit)
    }

    /// Evaluate at an explicitly chosen time and side of a source event.
    ///
    /// # Errors
    /// Nonfinite state/time, invalid constitutive law or unsupported physics.
    pub fn evaluate(
        &mut self,
        state: &Vector,
        time: Real,
        limit: Limit,
    ) -> SpiceResult<&PhysicalDaePoint> {
        if !time.is_finite() {
            return Err(SpiceError::circuit("nonfinite physical DAE time"));
        }
        let point = &mut self.point;
        point.conductance.clear();
        point.capacitance.clear();
        self.rhs.clear();
        let mut trial = self.history.trial_reusing(self.trial.take());
        let loaded = self.run.load(
            &LoadRequest {
                mode: AnalysisMode::Instantaneous { time },
                solution: state,
                model_context: &self.model,
                integration: None,
                history: &self.history,
                forcing: Some(Forcing {
                    limit,
                    timing: self.timing,
                }),
            },
            &mut point.conductance,
            &mut self.rhs,
            &mut trial,
        );
        self.trial = Some(trial);
        loaded?;
        point
            .conductance
            .mul_vector_into(state, &mut point.residual)?;
        for (residual, rhs) in point
            .residual
            .as_mut_slice()
            .iter_mut()
            .zip(self.rhs.as_slice())
        {
            *residual -= rhs;
        }
        self.linear.e.mul_vector_into(state, &mut point.charge)?;
        for term in self.linear.e.triplets() {
            point.capacitance.add(term.row, term.col, term.value)?;
        }
        self.run.dae_nonlinear_charge(
            &self.model,
            state,
            time,
            &mut point.charge,
            &mut point.capacitance,
        )?;
        point.capacitance.fold_duplicates();
        if !point.residual.is_finite() || !point.charge.is_finite() {
            return Err(SpiceError::circuit("nonfinite physical DAE residual"));
        }
        Ok(point)
    }
}
