//! Exact, unintegrated real residuals for a memoryless physical DAE.
//!
//! The same prepared devices and original MNA coordinates are used for both
//! F(x,t) and Q(x,t), so the equation is F + dQ/dt = 0. No integration
//! coefficient, time discretization, device limiting, or artificial gmin
//! is introduced by this evaluator.

pub(crate) mod guards;
pub(crate) use guards::{BranchTrace, LocalGuard, Region};

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

    /// Assemble exact one-sided physical Taylor coefficients from the same prepared
    /// device topology and analytic independent forcing.
    ///
    /// # Errors
    /// Invalid state jet, unsupported nonlinear Taylor kernel, singular
    /// constitutive expansion or nonfinite result.
    pub fn evaluate_taylor(
        &self,
        state: &[Vector],
        time: Real,
        scale: Real,
        limit: Limit,
    ) -> SpiceResult<PhysicalDaeTaylor> {
        self.evaluate_taylor_inner(state, time, scale, limit, false)
    }

    pub(crate) fn evaluate_taylor_with_guards(
        &self,
        state: &[Vector],
        time: Real,
        scale: Real,
        limit: Limit,
    ) -> SpiceResult<PhysicalDaeTaylor> {
        self.evaluate_taylor_inner(state, time, scale, limit, true)
    }

    fn evaluate_taylor_inner(
        &self,
        state: &[Vector],
        time: Real,
        scale: Real,
        limit: Limit,
        collect_guards: bool,
    ) -> SpiceResult<PhysicalDaeTaylor> {
        use crate::maths::taylor::{Jet, MAX_ORDER};
        let n = self.run.circuit().unknown_count();
        if state.is_empty()
            || state.len() > MAX_ORDER + 1
            || state
                .iter()
                .any(|vector| vector.len() != n || !vector.is_finite())
            || n.saturating_mul(state.len()) > 8_388_608
        {
            return Err(SpiceError::circuit(
                "invalid physical Taylor state or resource limit",
            ));
        }
        let order = state.len() - 1;
        let mut forcing = self.forcing_taylor_rhs(time, scale, order, limit)?;
        let reversed;
        let (state, time) = if limit == Limit::Left {
            reversed = reverse_odd_vectors(state);
            negate_odd_vectors(&mut forcing);
            (reversed.as_slice(), Jet::affine(order, time, -scale))
        } else {
            (state, Jet::affine(order, time, scale))
        };
        let mut output = PhysicalDaeTaylor::linear(&self.linear, state, forcing)?;
        output.guards = collect_guards.then(BranchTrace::default);
        self.run
            .dae_nonlinear_taylor(&self.model, state, &time, &mut output)?;
        if limit == Limit::Left {
            negate_odd_vectors(&mut output.residual);
            negate_odd_vectors(&mut output.charge);
            for matrices in [&mut output.conductance, &mut output.capacitance] {
                for matrix in matrices.iter_mut().skip(1).step_by(2) {
                    let (rows, cols) = (matrix.rows(), matrix.cols());
                    let old = std::mem::replace(matrix, SparseMatrix::new(rows, cols));
                    for term in old.triplets() {
                        matrix.add(term.row, term.col, -term.value)?;
                    }
                }
            }
        }
        output.validate()?;
        Ok(output)
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

fn negate_odd_vectors(vectors: &mut [Vector]) {
    for vector in vectors.iter_mut().skip(1).step_by(2) {
        for value in vector.as_mut_slice() {
            *value = -*value;
        }
    }
}

fn reverse_odd_vectors(vectors: &[Vector]) -> Vec<Vector> {
    let mut reversed = vectors.to_vec();
    negate_odd_vectors(&mut reversed);
    reversed
}

/// Coefficients of the physical equations and their state Jacobians.
///
/// Index k denotes the coefficient of the k-th normalized time power.
/// Jacobian coefficients act by convolution on corrections to state jets.
pub struct PhysicalDaeTaylor {
    /// Static residual coefficients.
    pub residual: Vec<Vector>,
    /// Charge/flux coefficients.
    pub charge: Vec<Vector>,
    /// Static Jacobian coefficients.
    pub conductance: Vec<SparseMatrix>,
    /// Charge/flux Jacobian coefficients.
    pub capacitance: Vec<SparseMatrix>,
    pub(crate) guards: Option<BranchTrace>,
    pub(crate) candidate_states: Vec<(usize, Real)>,
}

impl PhysicalDaeTaylor {
    fn linear(
        system: &crate::devices::linear::LinearSystem,
        state: &[Vector],
        forcing: Vec<Vector>,
    ) -> SpiceResult<Self> {
        let n = system.a.rows();
        let mut result = Self {
            residual: forcing,
            charge: Vec::with_capacity(state.len()),
            conductance: Vec::with_capacity(state.len()),
            capacitance: Vec::with_capacity(state.len()),
            guards: None,
            candidate_states: Vec::new(),
        };
        for (degree, x) in state.iter().enumerate() {
            let mut f = system.a.mul_vector(x)?;
            for (a, b) in f
                .as_mut_slice()
                .iter_mut()
                .zip(result.residual[degree].as_slice())
            {
                *a -= b;
            }
            result.residual[degree] = f;
            result.charge.push(system.e.mul_vector(x)?);
            result.conductance.push(if degree == 0 {
                system.a.clone()
            } else {
                SparseMatrix::new(n, n)
            });
            result.capacitance.push(if degree == 0 {
                system.e.clone()
            } else {
                SparseMatrix::new(n, n)
            });
        }
        Ok(result)
    }

    fn validate(&mut self) -> SpiceResult<()> {
        for matrix in self.conductance.iter_mut().chain(&mut self.capacitance) {
            matrix.fold_duplicates();
            if matrix
                .triplets()
                .iter()
                .any(|entry| !entry.value.is_finite())
            {
                return Err(SpiceError::circuit("nonfinite physical Taylor Jacobian"));
            }
        }
        if self
            .residual
            .iter()
            .chain(&self.charge)
            .any(|vector| !vector.is_finite())
        {
            return Err(SpiceError::circuit("nonfinite physical Taylor residual"));
        }
        Ok(())
    }
}

/// Device-local bindings and accumulation target for exact physical jets.
/// Construction is restricted to the prepared physical DAE evaluator.
pub struct TaylorContext<'a> {
    pub(crate) device: usize,
    pub(crate) state_rows: std::ops::Range<usize>,
    pub(crate) model: &'a ModelContext,
    pub(crate) unknowns: &'a crate::devices::MnaUnknowns,
    pub(crate) branch: Option<usize>,
    pub(crate) controls: &'a [usize],
    pub(crate) state: &'a [Vector],
    pub(crate) time: &'a crate::maths::taylor::Jet,
    pub(crate) output: &'a mut PhysicalDaeTaylor,
}

impl TaylorContext<'_> {
    /// Retain the value chosen by the physical germ, not the scalar value
    /// at an isolated branch boundary.
    pub(crate) fn candidate_state(&mut self, slot: usize, value: Real) -> SpiceResult<()> {
        if self.output.guards.is_some() {
            let row = self
                .state_rows
                .start
                .checked_add(slot)
                .filter(|&row| self.state_rows.contains(&row))
                .ok_or_else(|| SpiceError::circuit("invalid physical candidate slot"))?;
            if !value.is_finite() {
                return Err(SpiceError::circuit("nonfinite physical candidate state"));
            }
            self.output.candidate_states.push((row, value));
        }
        Ok(())
    }

    pub(crate) fn variable(&self, column: Option<usize>) -> SpiceResult<crate::maths::taylor::Jet> {
        use crate::maths::taylor::Jet;
        column.map_or_else(
            || Ok(Jet::constant(self.time.order(), 0.)),
            |column| {
                Jet::from_coefficients(
                    self.state
                        .iter()
                        .map(|x| x.get(column).unwrap_or(Real::NAN))
                        .collect(),
                )
            },
        )
    }

    pub(crate) fn static_value(
        &mut self,
        row: Option<usize>,
        jet: &crate::maths::taylor::Jet,
        scale: Real,
    ) -> SpiceResult<()> {
        if let Some(row) = row {
            for (vector, &value) in self.output.residual.iter_mut().zip(jet.coefficients()) {
                vector.add_to(row, scale * value)?;
            }
        }
        Ok(())
    }

    pub(crate) fn static_gradient(
        &mut self,
        row: Option<usize>,
        column: Option<usize>,
        jet: &crate::maths::taylor::Jet,
        scale: Real,
    ) -> SpiceResult<()> {
        if let (Some(row), Some(column)) = (row, column) {
            for (matrix, &value) in self.output.conductance.iter_mut().zip(jet.coefficients()) {
                matrix.add(row, column, scale * value)?;
            }
        }
        Ok(())
    }

    pub(crate) fn linear_static(
        &mut self,
        row: Option<usize>,
        column: Option<usize>,
        coefficient: Real,
    ) -> SpiceResult<()> {
        if let (Some(row), Some(column)) = (row, column)
            && coefficient != 0.
        {
            self.output.conductance[0].add(row, column, coefficient)?;
            for (vector, x) in self.output.residual.iter_mut().zip(self.state) {
                vector.add_to(row, coefficient * x.as_slice()[column])?;
            }
        }
        Ok(())
    }

    pub(crate) fn linear_charge(
        &mut self,
        row: Option<usize>,
        column: Option<usize>,
        coefficient: Real,
    ) -> SpiceResult<()> {
        if let (Some(row), Some(column)) = (row, column)
            && coefficient != 0.
        {
            self.output.capacitance[0].add(row, column, coefficient)?;
            for (vector, x) in self.output.charge.iter_mut().zip(self.state) {
                vector.add_to(row, coefficient * x.as_slice()[column])?;
            }
        }
        Ok(())
    }
}
