//! Exact, unintegrated real residuals for a memoryless physical DAE.
//!
//! The same prepared devices and original MNA coordinates are used for both
//! F(x,t) and Q(x,t), so the equation is F + dQ/dt = 0. No integration
//! coefficient, time discretization, device limiting, or artificial gmin
//! is introduced by this evaluator.

mod guards;

use crate::devices::{
    AnalysisMode, Forcing, Limit, LoadRequest, ModelContext, PreparedRun, StateHistory,
    TransientTiming, TrialState,
};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Region {
    Sign(std::cmp::Ordering),
    Interval(usize),
    Discrete(u64),
}

/// One operation's active branch and the adjacent candidate boundaries in
/// its operand coordinate. The slot and operand discriminator are stable
/// across evaluations of the same compiled program.
#[derive(Debug, Clone)]
pub(crate) struct LocalGuard {
    pub slot: usize,
    pub operand: u8,
    pub region: Region,
    pub argument: crate::maths::taylor::Jet,
    pub boundaries: [Option<Real>; 2],
}

#[derive(Debug, Clone)]
struct Guard {
    device: usize,
    local: LocalGuard,
}

/// Observations belong to a disposable physical point until it is accepted.
#[derive(Debug, Default, Clone)]
pub(crate) struct BranchTrace {
    guards: Vec<Guard>,
    coefficients: usize,
}

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
    regularization: Real,
    history: StateHistory,
    trial: Option<TrialState>,
    rhs: Vector,
    point: PhysicalDaePoint,
    storage_bindings: Vec<(usize, String)>,
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
        let regularization = circuit.continuation_diagonal();
        circuit.stamp_nodal_regularization(&mut linear.a, regularization)?;
        Ok(Self {
            run,
            model,
            timing,
            linear,
            regularization,
            history: circuit.independent_state_history()?,
            trial: None,
            rhs: Vector::zeros(n),
            storage_bindings: run.physical_taylor_storage_bindings(&model)?,
            point: PhysicalDaePoint {
                residual: Vector::zeros(n),
                charge: Vector::zeros(n),
                conductance: circuit.new_matrix()?,
                capacitance: SparseMatrix::new(n, n),
            },
        })
    }

    pub(crate) fn storage_bindings(&self) -> &[(usize, String)] {
        &self.storage_bindings
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
            for storage in output.storage.values_mut() {
                for coefficients in std::iter::once(&mut storage.values).chain(
                    storage
                        .gradients
                        .iter_mut()
                        .map(|(_, coefficients)| coefficients),
                ) {
                    for value in coefficients.iter_mut().skip(1).step_by(2) {
                        *value = -*value;
                    }
                }
            }
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
        if !output
            .storage
            .keys()
            .copied()
            .eq(self.storage_bindings.iter().map(|(slot, _)| *slot))
        {
            return Err(SpiceError::circuit(
                "physical Taylor storage does not match declared history bindings",
            ));
        }
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
        self.run
            .circuit()
            .stamp_nodal_regularization(&mut point.conductance, self.regularization)?;
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

/// A device-local stored charge, in its history units rather than the
/// signed sum of all device contributions to a global MNA equation.
pub(crate) struct LocalChargeTaylor {
    pub(crate) values: Vec<Real>,
    pub(crate) gradients: Vec<(usize, Vec<Real>)>,
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
    pub(crate) storage: BTreeMap<usize, LocalChargeTaylor>,
    storage_coefficients: usize,
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
            storage: BTreeMap::new(),
            storage_coefficients: 0,
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

/// A normalized local constitutive voltage and its absolute MNA incidence.
#[derive(Clone, Copy)]
pub(crate) struct TaylorControl {
    pub(crate) rows: [Option<usize>; 2],
    pub(crate) polarity: Real,
}

impl TaylorControl {
    pub(crate) fn input<const N: usize>(
        self,
        index: usize,
        context: &TaylorContext<'_>,
    ) -> SpiceResult<crate::maths::taylor::differential::Differential<N>> {
        let v = context
            .variable(self.rows[0])?
            .add(&context.variable(self.rows[1])?.scale(-1.))
            .scale(self.polarity);
        crate::maths::taylor::differential::Differential::variable(index, v)
    }
}

/// Convert sparse local partials to absolute MNA columns, combining aliases
/// before stamping or recording device-local charge.
fn absolute_gradients<const N: usize>(
    law: &crate::maths::taylor::differential::Differential<N>,
    controls: &[TaylorControl; N],
) -> Vec<(usize, crate::maths::taylor::Jet)> {
    let mut absolute = std::collections::BTreeMap::<usize, crate::maths::taylor::Jet>::new();
    for (derivative, control) in law.gradient.iter().zip(controls) {
        if let Some(derivative) = derivative
            && control.rows[0] != control.rows[1]
        {
            for (row, sign) in control
                .rows
                .into_iter()
                .zip([control.polarity, -control.polarity])
            {
                if let Some(row) = row {
                    let term = derivative.clone().scale(sign);
                    let entry = if let Some(old) = absolute.remove(&row) {
                        old.add(&term)
                    } else {
                        term
                    };
                    absolute.insert(row, entry);
                }
            }
        }
    }
    absolute
        .into_iter()
        .filter(|(_, gradient)| !gradient.is_zero())
        .collect()
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

    /// Retain a device's own charge jet and sparse state gradient for
    /// independent initial storage and companion constraints.
    pub(crate) fn record_charge(
        &mut self,
        slot: usize,
        value: &crate::maths::taylor::Jet,
        gradients: impl IntoIterator<Item = (usize, crate::maths::taylor::Jet)>,
    ) -> SpiceResult<()> {
        if slot.saturating_add(1) >= self.state_rows.len() {
            return Err(SpiceError::circuit("invalid physical Taylor charge slot"));
        }
        let degree = self.state.len();
        let mut charge = LocalChargeTaylor {
            values: value.coefficients().to_vec(),
            gradients: Vec::new(),
        };
        let account = |output: &mut PhysicalDaeTaylor| -> SpiceResult<()> {
            output.storage_coefficients = output
                .storage_coefficients
                .checked_add(degree)
                .filter(|&count| count <= 8_388_608)
                .ok_or_else(|| {
                    SpiceError::circuit("physical Taylor charge coefficient budget exceeded")
                })?;
            Ok(())
        };
        value.check()?;
        if charge.values.len() != degree {
            return Err(SpiceError::circuit("invalid physical Taylor charge order"));
        }
        account(self.output)?;
        for (column, gradient) in gradients {
            gradient.check()?;
            if column >= self.state[0].len() || gradient.coefficients().len() != degree {
                return Err(SpiceError::circuit(
                    "invalid physical Taylor charge gradient",
                ));
            }
            account(self.output)?;
            charge
                .gradients
                .push((column, gradient.coefficients().to_vec()));
        }
        if self
            .output
            .storage
            .insert(self.state_rows.start + slot, charge)
            .is_some()
        {
            return Err(SpiceError::circuit("duplicate physical Taylor charge slot"));
        }
        Ok(())
    }

    /// Stamp one scalar constitutive law between two rows, controlled by
    /// the difference of two columns. Static and charge equations use the
    /// same incidence and chain rule; aliased terminals cancel exactly.
    pub(crate) fn branch_taylor(
        &mut self,
        rows: [Option<usize>; 2],
        columns: [Option<usize>; 2],
        value: &crate::maths::taylor::Jet,
        gradient: &crate::maths::taylor::Jet,
        charge: bool,
    ) -> SpiceResult<()> {
        if rows[0] == rows[1] {
            return Ok(());
        }
        let (values, jacobians) = if charge {
            (&mut self.output.charge, &mut self.output.capacitance)
        } else {
            (&mut self.output.residual, &mut self.output.conductance)
        };
        for (row, sign) in rows.into_iter().zip([1., -1.]) {
            if let Some(row) = row {
                for (vector, &coefficient) in values.iter_mut().zip(value.coefficients()) {
                    vector.add_to(row, sign * coefficient)?;
                }
                if columns[0] != columns[1] {
                    for (column, orientation) in columns.into_iter().zip([sign, -sign]) {
                        if let Some(column) = column {
                            for (matrix, &coefficient) in
                                jacobians.iter_mut().zip(gradient.coefficients())
                            {
                                matrix.add(row, column, orientation * coefficient)?;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Stamp a normalized multi-coordinate law, optionally preserving its
    /// separate history coordinate (without physical polarity/multiplier).
    pub(crate) fn local_law<const N: usize>(
        &mut self,
        rows: [Option<usize>; 2],
        law: &crate::maths::taylor::differential::Differential<N>,
        controls: &[TaylorControl; N],
        scale: Real,
        charge_slot: Option<usize>,
    ) -> SpiceResult<()> {
        let gradients = absolute_gradients(law, controls);
        let physical = gradients
            .iter()
            .map(|(row, g)| (*row, g.clone().scale(scale)))
            .collect::<Vec<_>>();
        self.branch_gradients(
            rows,
            &law.value.clone().scale(scale),
            &physical,
            charge_slot.is_some(),
        )?;
        if let Some(slot) = charge_slot {
            self.record_charge(slot, &law.value, gradients)?;
        }
        Ok(())
    }

    /// Stamp a multi-coordinate constitutive law using its absolute sparse
    /// state derivatives. The caller has already combined aliased columns.
    pub(crate) fn branch_gradients(
        &mut self,
        rows: [Option<usize>; 2],
        value: &crate::maths::taylor::Jet,
        gradients: &[(usize, crate::maths::taylor::Jet)],
        charge: bool,
    ) -> SpiceResult<()> {
        if rows[0] == rows[1] {
            return Ok(());
        }
        value.check()?;
        let (values, jacobians) = if charge {
            (&mut self.output.charge, &mut self.output.capacitance)
        } else {
            (&mut self.output.residual, &mut self.output.conductance)
        };
        for (row, sign) in rows.into_iter().zip([1., -1.]) {
            if let Some(row) = row {
                for (vector, &coefficient) in values.iter_mut().zip(value.coefficients()) {
                    vector.add_to(row, sign * coefficient)?;
                }
                for (column, gradient) in gradients {
                    gradient.check()?;
                    for (matrix, &coefficient) in jacobians.iter_mut().zip(gradient.coefficients())
                    {
                        matrix.add(row, *column, sign * coefficient)?;
                    }
                }
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
