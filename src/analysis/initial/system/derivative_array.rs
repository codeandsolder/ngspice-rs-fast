//! Coupled physical derivative arrays. A normalized time coefficient X[k]
//! represents x^(k)(0) * h^k/k!. The equation in degree k is
//! F[k] + (k+1)/h Q[k+1] = 0, with state Jacobians acting by convolution.
//! Higher-degree variables are projected away before initial constraints
//! are selected. No arbitrary zero derivative gauge enters the prefix.

use super::rank;
mod recovery;
use crate::analysis::initial::{
    CapacitorInitial, ChargeInitial, DerivativeGuess, FixedCoordinate, ImpulseStorage,
    LocalChargeInitial, Tolerances,
    constraints::{AbsoluteRow, VoltageVertex},
    failure,
};
use crate::devices::physical_dae::{PhysicalDae, PhysicalDaeTaylor};
use crate::devices::{
    Circuit, InitialCoordinateHistory, InitialStoragePolicy, Limit, LoadRequest, StateHistory,
};
use crate::maths::integrator::{ChargeHistory, Coefficients};
use crate::maths::taylor::MAX_ORDER;
use crate::maths::{SparseMatrix, Triplet, Vector};
use crate::primitives::{Real, SpiceResult};

const MAX_ARRAY_ENTRIES: usize = 8_388_608;
const MAX_ARRAY_VARIABLES: usize = 1_048_576;
const MAX_ARRAY_ROWS: usize = 1_048_576;

/// `PhysicalDaeTaylor` validates and folds every Jacobian before returning it.
/// Restrict constraint traversal to the requested row rather than scanning
/// the entire derivative matrix for each independent storage coordinate.
fn folded_row(matrix: &SparseMatrix, row: usize) -> &[Triplet] {
    let entries = matrix.triplets();
    let start = entries.partition_point(|term| term.row < row);
    let end = start + entries[start..].partition_point(|term| term.row == row);
    &entries[start..end]
}

/// Account for raw contributions before inserting them, including duplicates
/// that will later be folded. Storage and companion rows share the same cap.
struct Stamps<'a> {
    matrix: &'a mut SparseMatrix,
    remaining: usize,
}

impl Stamps<'_> {
    fn add(&mut self, row: usize, col: usize, value: Real) -> SpiceResult<()> {
        self.remaining = self
            .remaining
            .checked_sub(1)
            .ok_or_else(|| failure("derivative array exceeds its stamp budget"))?;
        self.matrix.add(row, col, value)
    }
}

#[derive(Clone, Copy, Default)]
struct ResidualScale {
    /// Magnitude of the actual physical quantity being constrained.
    physical: Real,
    /// Sum of magnitudes before cancellation, for floating-point error.
    backward: Real,
}

impl ResidualScale {
    const fn equal(value: Real) -> Self {
        Self {
            physical: value,
            backward: value,
        }
    }

    const fn comparison(actual: Real, target: Real, backward: Real) -> Self {
        let physical = actual.abs().max(target.abs());
        Self {
            physical,
            backward: backward.max(physical),
        }
    }

    fn roundoff(self) -> Real {
        (32. * Real::EPSILON * self.backward).max(Real::MIN_POSITIVE)
    }

    const fn finite(self) -> bool {
        self.physical.is_finite() && self.backward.is_finite()
    }
}

#[derive(Clone, Copy)]
enum Constraint<'a> {
    Coordinate(usize, &'a FixedCoordinate),
    Voltage(&'a CapacitorInitial),
    Charge(usize, &'a ChargeInitial),
    LocalCharge(&'a LocalChargeInitial),
    Companion(CompanionConstraint<'a>),
}

impl<'a> Constraint<'a> {
    fn optional(self) -> bool {
        matches!(self,Self::Charge(_,charge) if charge.policy==InitialStoragePolicy::Derived)
            || matches!(self, Self::LocalCharge(charge) if charge.optional)
            || matches!(self, Self::Companion(c) if c.derived)
    }

    fn initial_name(self) -> Option<(&'a str, &'static str)> {
        match self {
            Self::Coordinate(_, coordinate) => Some((&coordinate.name, "coordinate")),
            Self::Voltage(cap) => Some((&cap.name, "voltage")),
            Self::Charge(_, charge) if charge.policy == InitialStoragePolicy::Physical => {
                Some((&charge.name, "charge"))
            }
            Self::LocalCharge(charge) if !charge.optional => Some((&charge.name, "charge")),
            _ => None,
        }
    }

    fn stamp(
        self,
        row: usize,
        state: &[Vector],
        time_scale: Real,
        point: &PhysicalDaeTaylor,
        matrix: &mut Stamps<'_>,
    ) -> SpiceResult<(Real, ResidualScale)> {
        if let Self::Companion(companion) = self {
            return companion.stamp(row, state, time_scale, point, matrix);
        }
        let state = &state[0];
        match self {
            Self::Coordinate(column, coordinate) => {
                matrix.add(row, column, 1.)?;
                let value = state.as_slice()[column];
                Ok((
                    value - coordinate.value,
                    ResidualScale::comparison(value, coordinate.value, 0.),
                ))
            }
            Self::Voltage(cap) => {
                let positive = crate::analysis::initial::potential(state, cap.positive);
                let negative = crate::analysis::initial::potential(state, cap.negative);
                for (vertex, sign) in [(cap.positive, 1.), (cap.negative, -1.)] {
                    if let VoltageVertex::Node(AbsoluteRow(column)) = vertex {
                        matrix.add(row, column, sign)?;
                    }
                }
                // Subtract common-mode voltage before the specified IC.
                // A large common mode must not loosen a differential IC.
                let actual = positive - negative;
                Ok((
                    actual - cap.voltage,
                    ResidualScale::comparison(actual, cap.voltage, positive.abs() + negative.abs()),
                ))
            }
            Self::Charge(equation, charge) => {
                let mut backward = 0.;
                for term in folded_row(&point.capacitance[0], equation) {
                    matrix.add(row, term.col, term.value)?;
                    backward = term
                        .value
                        .abs()
                        .mul_add(state.as_slice()[term.col].abs(), backward);
                }
                let value = point.charge[0].as_slice()[equation];
                Ok((
                    value - charge.target,
                    ResidualScale::comparison(value, charge.target, backward),
                ))
            }
            Self::LocalCharge(charge) => {
                let local = point
                    .storage
                    .get(&charge.slot)
                    .ok_or_else(|| failure("missing device-local Taylor charge"))?;
                let mut backward = 0.;
                for (column, gradient) in &local.gradients {
                    matrix.add(row, *column, gradient[0])?;
                    backward = gradient[0]
                        .abs()
                        .mul_add(state.as_slice()[*column].abs(), backward);
                }
                let value = local.values[0];
                Ok((
                    value - charge.target,
                    ResidualScale::comparison(value, charge.target, backward),
                ))
            }
            Self::Companion(_) => Err(failure("unexpected companion constraint")),
        }
    }

    fn absolute(self, tol: Tolerances) -> Real {
        match self {
            Self::Voltage(..) => tol.vntol,
            Self::Charge(_, charge) if charge.policy == InitialStoragePolicy::Derived => tol.vntol,
            Self::Charge(..) | Self::LocalCharge(..) => tol.chgtol,
            Self::Companion(c) if c.derived => tol.vntol,
            Self::Coordinate(..) | Self::Companion(_) => tol.abstol,
        }
    }

    fn tolerance(self, scale: ResidualScale, tol: Tolerances) -> Real {
        let requested = tol.reltol.mul_add(scale.physical, self.absolute(tol));
        // A selected numerical companion is constrained by finite-precision
        // history differences, not by an exact prescribed IC. Its backward
        // error floor cannot be applied to mandatory initial storage.
        if matches!(self, Self::Companion(_)) {
            requested.max(scale.roundoff())
        } else {
            requested
        }
    }
}

fn constraints(storage: &ImpulseStorage) -> Vec<Constraint<'_>> {
    storage
        .fixed_coordinates
        .iter()
        .map(|(&AbsoluteRow(row), item)| Constraint::Coordinate(row, item))
        .chain(storage.capacitors.iter().map(Constraint::Voltage))
        .chain(storage.local_charges.iter().map(Constraint::LocalCharge))
        .chain(
            storage
                .charges
                .iter()
                .map(|(&AbsoluteRow(row), item)| Constraint::Charge(row, item)),
        )
        .collect()
}

/// A storage coordinate in the same units as its accepted history slots.
#[derive(Clone, Copy)]
enum StoredQuantity {
    Coordinate(usize),
    Voltage(VoltageVertex, VoltageVertex, Real),
    Equation(usize, Real),
    LocalCharge(usize),
}

impl StoredQuantity {
    fn value(
        self,
        state: &[Vector],
        point: &PhysicalDaeTaylor,
        degree: usize,
    ) -> SpiceResult<Real> {
        Ok(match self {
            Self::Coordinate(column) => state[degree].as_slice()[column],
            Self::Voltage(p, n, scale) => {
                scale
                    * (crate::analysis::initial::potential(&state[degree], p)
                        - crate::analysis::initial::potential(&state[degree], n))
            }
            Self::Equation(row, scale) => scale * point.charge[degree].as_slice()[row],
            Self::LocalCharge(slot) => {
                point
                    .storage
                    .get(&slot)
                    .ok_or_else(|| failure("missing device-local Taylor charge"))?
                    .values[degree]
            }
        })
    }

    fn gradient(
        self,
        point: &PhysicalDaeTaylor,
        degree: usize,
        mut visit: impl FnMut(usize, Real) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        match self {
            Self::Coordinate(column) if degree == 0 => visit(column, 1.),
            Self::Voltage(p, n, scale) if degree == 0 => {
                for (vertex, factor) in [(p, scale), (n, -scale)] {
                    if let VoltageVertex::Node(AbsoluteRow(column)) = vertex {
                        visit(column, factor)?;
                    }
                }
                Ok(())
            }
            Self::Equation(row, scale) => {
                for term in folded_row(&point.capacitance[degree], row) {
                    visit(term.col, scale * term.value)?;
                }
                Ok(())
            }
            Self::LocalCharge(slot) => {
                let local = point
                    .storage
                    .get(&slot)
                    .ok_or_else(|| failure("missing device-local Taylor charge"))?;
                for (column, gradient) in &local.gradients {
                    visit(*column, gradient[degree])?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy)]
struct CompanionConstraint<'a> {
    quantity: StoredQuantity,
    slot: usize,
    derived: bool,
    history: &'a StateHistory,
    coefficients: &'a Coefficients,
}

impl CompanionConstraint<'_> {
    fn stamp(
        self,
        row: usize,
        state: &[Vector],
        time_scale: Real,
        point: &PhysicalDaeTaylor,
        matrix: &mut Stamps<'_>,
    ) -> SpiceResult<(Real, ResidualScale)> {
        let accepted = |age, offset| {
            self.history
                .accepted(age)
                .and_then(|values| values.get(self.slot + offset))
                .copied()
                .ok_or_else(|| failure("missing independent storage history"))
        };
        let value = self.quantity.value(state, point, 0)?;
        let rate = self.quantity.value(state, point, 1)? / time_scale;
        let mut history = ChargeHistory::new(value);
        for age in 1..=self.coefficients.charge_history_len() {
            history.push(accepted(age, 0)?)?;
        }
        let previous = self
            .coefficients
            .needs_previous_derivative()
            .then(|| accepted(1, 1))
            .transpose()?;
        let companion = self.coefficients.integrate(&history, previous, 1.)?;
        let physical = rate.abs().max(companion.derivative.abs());
        let mut scale = rate.abs() + companion.derivative.abs();
        let width = state[0].len();
        self.quantity.gradient(point, 0, |column, gradient| {
            let term = -companion.conductance * gradient;
            matrix.add(row, column, term)?;
            scale += (term * state[0].as_slice()[column]).abs();
            let term = gradient / time_scale;
            matrix.add(row, width + column, term)?;
            scale += (term * state[1].as_slice()[column]).abs();
            Ok(())
        })?;
        self.quantity.gradient(point, 1, |column, gradient| {
            let term = gradient / time_scale;
            matrix.add(row, column, term)?;
            scale += (term * state[0].as_slice()[column]).abs();
            Ok(())
        })?;
        Ok((
            rate - companion.derivative,
            ResidualScale {
                physical,
                backward: scale,
            },
        ))
    }
}

fn companions<'a>(
    circuit: &Circuit,
    storage: &ImpulseStorage,
    history: &'a StateHistory,
    coefficients: &'a Coefficients,
) -> SpiceResult<Vec<Constraint<'a>>> {
    let mut result = Vec::new();
    let mut add = |quantity, slot, derived| {
        result.push(Constraint::Companion(CompanionConstraint {
            quantity,
            slot,
            derived,
            history,
            coefficients,
        }));
    };
    for charge in &storage.local_charges {
        add(StoredQuantity::LocalCharge(charge.slot), charge.slot, false);
    }
    for (&AbsoluteRow(row), charge) in &storage.charges {
        add(
            StoredQuantity::Equation(row, 1.),
            charge.slot,
            charge.policy == InitialStoragePolicy::Derived,
        );
    }
    for cap in &storage.capacitors {
        use crate::devices::traits::VoltageStorageHistory;
        let quantity = match cap.history {
            VoltageStorageHistory::Linear(scale) => {
                StoredQuantity::Voltage(cap.positive, cap.negative, scale)
            }
            VoltageStorageHistory::Polynomial { equation_scale, .. } => StoredQuantity::Equation(
                cap.equation
                    .ok_or_else(|| failure("missing nonlinear voltage evolution row"))?
                    .0,
                equation_scale,
            ),
        };
        add(quantity, cap.slot, false);
    }
    for (&AbsoluteRow(row), coordinate) in &storage.fixed_coordinates {
        let slots = circuit
            .state_rows(coordinate.device)
            .ok_or_else(|| failure("missing coordinate history"))?;
        let (quantity, slot) = match coordinate.history {
            InitialCoordinateHistory::EquationCharge { slot, scale } => {
                (StoredQuantity::Equation(coordinate.equation.0, scale), slot)
            }
            InitialCoordinateHistory::Coordinate { slot } => {
                (StoredQuantity::Coordinate(row), slot)
            }
        };
        if slot.saturating_add(1) >= slots.len() {
            return Err(failure("invalid companion coordinate history"));
        }
        add(quantity, slots.start + slot, false);
    }
    Ok(result)
}

struct Array {
    matrix: SparseMatrix,
    residual: Vec<Real>,
    scale: Vec<ResidualScale>,
    physical_rows: usize,
    late: Vec<usize>,
    optional: Vec<usize>,
}

impl Array {
    fn build(
        point: &PhysicalDaeTaylor,
        state: &[Vector],
        scale: Real,
        initial: &[Constraint<'_>],
    ) -> SpiceResult<Self> {
        let width = state[0].len();
        let order = state.len() - 1;
        let variables = width
            .checked_mul(state.len())
            .filter(|&n| n <= MAX_ARRAY_VARIABLES)
            .ok_or_else(|| failure("derivative array exceeds its variable budget"))?;
        let physical_rows = width * order;
        let rows = physical_rows
            .checked_add(initial.len())
            .filter(|&n| n <= MAX_ARRAY_ROWS)
            .ok_or_else(|| failure("derivative array exceeds its row budget"))?;
        let mut array = Self {
            matrix: SparseMatrix::new(rows, variables),
            residual: crate::primitives::filled_vec(rows, 0., "derivative array residual")?,
            scale: crate::primitives::filled_vec(
                rows,
                ResidualScale::default(),
                "derivative array residual scale",
            )?,
            physical_rows,
            late: Vec::new(),
            optional: Vec::new(),
        };
        let mut degree = 1.;
        let mut stamps = Stamps {
            matrix: &mut array.matrix,
            remaining: MAX_ARRAY_ENTRIES,
        };
        for k in 0..order {
            let multiplier = degree / scale;
            let offset = k * width;
            for (r, (&f, &q)) in point.residual[k]
                .as_slice()
                .iter()
                .zip(point.charge[k + 1].as_slice())
                .enumerate()
            {
                array.residual[offset + r] = multiplier.mul_add(q, f);
                array.scale[offset + r] =
                    ResidualScale::equal(multiplier.mul_add(q.abs(), f.abs()));
            }
            for (j, coefficient_state) in state.iter().enumerate().take(k + 2) {
                for (matrix, factor) in [
                    (point.conductance.get(k.wrapping_sub(j)), 1.),
                    (point.capacitance.get(k + 1 - j), multiplier),
                ] {
                    let Some(matrix) = matrix else { continue };
                    for term in matrix.triplets() {
                        let row = offset + term.row;
                        let coefficient = factor * term.value;
                        stamps.add(row, j * width + term.col, coefficient)?;
                        array.scale[row] = ResidualScale::equal(coefficient.abs().mul_add(
                            coefficient_state.as_slice()[term.col].abs(),
                            array.scale[row].backward,
                        ));
                    }
                }
            }
            degree += 1.;
        }
        for (index, constraint) in initial.iter().copied().enumerate() {
            let row = physical_rows + index;
            let (residual, scale) = constraint.stamp(row, state, scale, point, &mut stamps)?;
            array.residual[row] = residual;
            array.scale[row] = scale;
            if constraint.optional() {
                array.optional.push(row);
            } else {
                array.late.push(row);
            }
        }
        array.matrix.fold_duplicates();
        if array.residual.iter().any(|v| !v.is_finite())
            || array.scale.iter().any(|scale| !scale.finite())
        {
            return Err(failure("nonfinite derivative array residual or scale"));
        }
        Ok(array)
    }

    fn merit(
        &self,
        initial: &[Constraint<'_>],
        independent: &[bool],
        tol: Tolerances,
    ) -> (Real, Real) {
        let mut user: Real = 0.;
        let mut roundoff: Real = 0.;
        for (row, (&residual, &scale)) in self.residual.iter().zip(&self.scale).enumerate() {
            let tolerance = if row < self.physical_rows {
                // Mixed-dimensional original equations retain their
                // stamp scale as the residual's physical reference.
                tol.reltol
                    .mul_add(scale.physical, tol.abstol.min(tol.vntol).min(tol.chgtol))
            } else {
                let constraint = initial[row - self.physical_rows];
                if (constraint.optional() || matches!(constraint, Constraint::Companion(_)))
                    && !independent[row]
                {
                    continue;
                }
                constraint.tolerance(scale, tol)
            };
            user = user.max(residual.abs() / tolerance);
            roundoff = roundoff.max(residual.abs() / scale.roundoff());
        }
        (user, roundoff)
    }

    fn initial_violation(&self, initial: &[Constraint<'_>], tol: Tolerances) -> Option<String> {
        initial.iter().copied().enumerate().filter_map(|(index,constraint)|{
            let (name,quantity)=constraint.initial_name()?;
            let row=self.physical_rows+index;
            let residual=self.residual[row];
            let tolerance=constraint.tolerance(self.scale[row],tol);
            (residual.abs()>tolerance).then_some((residual.abs()/tolerance,name,quantity,residual,tolerance))
        }).max_by(|a,b|a.0.total_cmp(&b.0)).map(|(_,name,quantity,residual,tolerance)|format!(
            "{name}: initial {quantity} constraint differs by {residual:e} (tolerance {tolerance:e}); the requested state may require an impulse"
        ))
    }
}

pub(in crate::analysis::initial) struct RightLimit {
    pub state: Vector,
    pub rate: Vector,
    pub charge: Vector,
    pub charge_rate: Vector,
    pub local_history: Vec<(usize, Real)>,
    pub order_hint: usize,
    pub guards: crate::devices::physical_dae::BranchTrace,
    pub candidate_states: Vec<(usize, Real)>,
    pub terminal_currents: crate::devices::physical_dae::PhysicalTerminalCurrents,
}

#[derive(Clone, Copy)]
struct LocalPoint<'a> {
    time: Real,
    scale: Real,
    limit: Limit,
    order_hint: usize,
    recovery: Option<&'a recovery::Budget>,
    rate_seed: Option<&'a Vector>,
}

impl LocalPoint<'_> {
    fn evaluate(
        self,
        dae: &PhysicalDae<'_, '_>,
        state: &[Vector],
    ) -> SpiceResult<PhysicalDaeTaylor> {
        if let Some(budget) = self.recovery {
            budget.charge(state)?;
        }
        dae.evaluate_taylor(state, self.time, self.scale, self.limit)
    }

    fn prefix_rank(self, dae: &PhysicalDae<'_, '_>, state: &[Vector]) -> SpiceResult<usize> {
        let point = self.evaluate(dae, state)?;
        let array = Array::build(&point, state, self.scale, &[])?;
        rank::projected_rank(&array.matrix, 2 * state[0].len())
    }
}

fn extend(state: &mut Vec<Vector>) -> bool {
    let width = state[0].len();
    if state.len() > MAX_ORDER || width.saturating_mul(state.len() + 1) > MAX_ARRAY_VARIABLES {
        return false;
    }
    state.push(Vector::zeros(width));
    true
}

fn order_limit() -> crate::primitives::SpiceError {
    failure(
        "physical DAE derivative order or state budget exhausted: required coordinates may be noncausal or underdetermined, or have an unresolved singular nonlinear tangent",
    )
}

fn stable_rank_order(
    dae: &PhysicalDae<'_, '_>,
    state: &mut Vec<Vector>,
    local: LocalPoint<'_>,
) -> SpiceResult<bool> {
    let mut previous = local.prefix_rank(dae, state)?;
    while extend(state) {
        let next = local.prefix_rank(dae, state)?;
        if next == previous {
            return Ok(true);
        }
        previous = next;
    }
    Ok(false)
}

enum ArrayNewton {
    Converged,
    NeedOrder { nonlinear_residual: bool },
    NotConverged(String),
}

fn apply_step(state: &mut [Vector], correction: &Vector) -> SpiceResult<()> {
    let width = state[0].len();
    for (jet, delta) in state
        .iter_mut()
        .zip(correction.as_slice().chunks_exact(width))
    {
        for (value, &delta) in jet.as_mut_slice().iter_mut().zip(delta) {
            *value += delta;
        }
        if !jet.is_finite() {
            return Err(failure("nonfinite derivative array Newton iterate"));
        }
    }
    Ok(())
}

fn newton(
    dae: &PhysicalDae<'_, '_>,
    state: &mut [Vector],
    local: LocalPoint<'_>,
    initial: &[Constraint<'_>],
    tol: Tolerances,
    iterations: usize,
) -> SpiceResult<ArrayNewton> {
    let width = state[0].len();
    let mut best: Option<(Real, Vec<Vector>)> = None;
    let mut violation = None;
    for _ in 0..iterations {
        let point = local.evaluate(dae, state)?;
        let array = Array::build(&point, state, local.scale, initial)?;
        let rhs = array
            .residual
            .iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let step =
            rank::projected_step(&array.matrix, &rhs, &array.late, &array.optional, 2 * width)?;
        if step.prefix_rank < 2 * width {
            // A nonlinear coefficient can vanish at a provisional seed,
            // then recover after the determined coordinates are corrected.
            // Do not confuse a deficient tangent with a noncausal DAE.
            // The expanded array must prove the whole prefix before use.
            let nonlinear_residual = array.residual[..array.physical_rows]
                .iter()
                .zip(&array.scale)
                .zip(&step.independent)
                .any(|((&residual, scale), &independent)| {
                    !independent && residual.abs() > scale.roundoff()
                });
            apply_step(state, &step.values)?;
            return Ok(ArrayNewton::NeedOrder { nonlinear_residual });
        }
        let (ratio, roundoff) = array.merit(initial, &step.independent, tol);
        if ratio > 1. {
            violation = array.initial_violation(initial, tol);
        }
        if ratio <= 1. {
            if roundoff <= 1. {
                return Ok(ArrayNewton::Converged);
            }
            // A recovery probe must reach rounding consistency in the
            // original physical rows. A small residual near a repeated
            // nonlinear root is not by itself a rank certificate.
            let physical_roundoff = array.residual[..array.physical_rows]
                .iter()
                .zip(&array.scale)
                .all(|(residual, scale)| residual.abs() <= scale.roundoff());
            if local.recovery.is_some() && physical_roundoff {
                return Ok(ArrayNewton::Converged);
            }
            if best.as_ref().is_some_and(|(score, _)| roundoff >= *score) {
                break;
            }
            best = Some((roundoff, state.to_vec()));
        }
        apply_step(state, &step.values)?;
    }
    if let Some((_, best)) = best.filter(|_| local.recovery.is_none()) {
        state.clone_from_slice(&best);
        Ok(ArrayNewton::Converged)
    } else {
        Ok(ArrayNewton::NotConverged(violation.unwrap_or_else(|| {
            "physical derivative array Newton iteration did not converge".to_owned()
        })))
    }
}

pub(in crate::analysis::initial) enum SolveOutcome {
    Converged(Box<RightLimit>),
    NotConverged { last: Vector, reason: String },
}

impl SolveOutcome {
    fn require(self) -> SpiceResult<RightLimit> {
        match self {
            Self::Converged(right) => Ok(*right),
            Self::NotConverged { reason, .. } => Err(failure(reason)),
        }
    }
}

/// Determine a consistent initial state and first derivative without imposing
/// artificial initial values on dependent higher derivatives. Increase the
/// order until the projected DAE rank is stable before considering optional
/// storage constraints, then recheck rank at the nonlinear solution.
pub(in crate::analysis::initial) fn solve_right_limit(
    dae: &PhysicalDae<'_, '_>,
    seed: &Vector,
    storage: &ImpulseStorage,
    scale: Real,
    tol: Tolerances,
    iterations: usize,
) -> SpiceResult<RightLimit> {
    solve(
        dae,
        seed,
        &constraints(storage),
        LocalPoint {
            time: 0.,
            scale,
            limit: Limit::Right,
            order_hint: 2,
            recovery: None,
            rate_seed: None,
        },
        tol,
        iterations,
    )?
    .require()
}

pub(in crate::analysis::initial) fn solve_event(
    dae: &PhysicalDae<'_, '_>,
    request: &LoadRequest<'_>,
    storage: &ImpulseStorage,
    tol: Tolerances,
    iterations: usize,
    guess: DerivativeGuess<'_>,
) -> SpiceResult<RightLimit> {
    let forcing = request
        .forcing
        .ok_or_else(|| failure("missing derivative event forcing"))?;
    let crate::devices::AnalysisMode::Instantaneous { time } = request.mode else {
        return Err(failure("invalid derivative event mode"));
    };
    solve(
        dae,
        request.solution,
        &constraints(storage),
        LocalPoint {
            time,
            scale: forcing.timing.step(),
            limit: Limit::Right,
            order_hint: guess.order_hint,
            recovery: None,
            rate_seed: guess.rate,
        },
        tol,
        iterations,
    )?
    .require()
}

pub(in crate::analysis::initial) fn solve_step(
    dae: &PhysicalDae<'_, '_>,
    circuit: &Circuit,
    request: &LoadRequest<'_>,
    storage: &ImpulseStorage,
    tol: Tolerances,
    iterations: usize,
    guess: DerivativeGuess<'_>,
) -> SpiceResult<SolveOutcome> {
    let forcing = request
        .forcing
        .ok_or_else(|| failure("missing derivative step forcing"))?;
    let coefficients = request
        .integration
        .ok_or_else(|| failure("missing derivative step coefficients"))?;
    let crate::devices::AnalysisMode::Transient { time, .. } = request.mode else {
        return Err(failure("invalid derivative step mode"));
    };
    solve(
        dae,
        request.solution,
        &companions(circuit, storage, request.history, coefficients)?,
        LocalPoint {
            time,
            scale: forcing.timing.step(),
            limit: forcing.limit,
            order_hint: guess.order_hint,
            recovery: None,
            rate_seed: guess.rate,
        },
        tol,
        iterations,
    )
}

struct Candidate {
    right: RightLimit,
    mandatory_rank: usize,
    uncertainty: Option<Vector>,
}

enum CoreOutcome {
    Converged(Box<Candidate>),
    NotConverged { last: Vector, reason: String },
    Exhausted(Option<Vec<Vector>>),
}

fn solve(
    dae: &PhysicalDae<'_, '_>,
    seed: &Vector,
    initial: &[Constraint<'_>],
    local: LocalPoint<'_>,
    tol: Tolerances,
    iterations: usize,
) -> SpiceResult<SolveOutcome> {
    let mut state = vec![seed.clone()];
    for _ in 0..local.order_hint.max(2) {
        if !extend(&mut state) {
            return Err(order_limit());
        }
    }
    if let Some(rate) = local.rate_seed {
        let normalized = Vector::from(
            rate.as_slice()
                .iter()
                .map(|value| value * local.scale)
                .collect::<Vec<_>>(),
        );
        if normalized.len() == seed.len() && normalized.is_finite() {
            state[1] = normalized;
        }
    }
    match solve_core(dae, state, initial, local, tol, iterations)? {
        CoreOutcome::Converged(candidate) => Ok(SolveOutcome::Converged(Box::new(candidate.right))),
        CoreOutcome::NotConverged { last, reason } => {
            Ok(SolveOutcome::NotConverged { last, reason })
        }
        CoreOutcome::Exhausted(Some(seed)) => {
            recovery::recover(dae, seed, initial, local, tol, iterations)
                .map(|right| SolveOutcome::Converged(Box::new(right)))
        }
        CoreOutcome::Exhausted(None) => Err(order_limit()),
    }
}

fn solve_core(
    dae: &PhysicalDae<'_, '_>,
    mut state: Vec<Vector>,
    initial: &[Constraint<'_>],
    local: LocalPoint<'_>,
    tol: Tolerances,
    iterations: usize,
) -> SpiceResult<CoreOutcome> {
    let mut recovery_seed = None;
    loop {
        if !stable_rank_order(dae, &mut state, local)? {
            break;
        }
        match newton(dae, &mut state, local, initial, tol, iterations)? {
            ArrayNewton::Converged => {}
            ArrayNewton::NeedOrder { nonlinear_residual } => {
                if nonlinear_residual && local.recovery.is_none() && recovery_seed.is_none() {
                    recovery_seed = Some(state.clone());
                }
                if !extend(&mut state) {
                    break;
                }
                continue;
            }
            ArrayNewton::NotConverged(reason) => {
                return Ok(CoreOutcome::NotConverged {
                    last: state.remove(0),
                    reason,
                });
            }
        }
        let rank = local.prefix_rank(dae, &state)?;
        if !extend(&mut state) {
            break;
        }
        if local.prefix_rank(dae, &state)? != rank {
            continue;
        }
        // Rank equality alone does not establish the affine residual.
        match newton(dae, &mut state, local, initial, tol, iterations)? {
            ArrayNewton::Converged => {}
            ArrayNewton::NeedOrder { nonlinear_residual } => {
                if nonlinear_residual && local.recovery.is_none() && recovery_seed.is_none() {
                    recovery_seed = Some(state.clone());
                }
                continue;
            }
            ArrayNewton::NotConverged(reason) => {
                return Ok(CoreOutcome::NotConverged {
                    last: state.remove(0),
                    reason,
                });
            }
        }
        if local.prefix_rank(dae, &state)? != rank {
            continue;
        }
        return candidate(dae, &state, initial, local)
            .map(|candidate| CoreOutcome::Converged(Box::new(candidate)));
    }
    Ok(CoreOutcome::Exhausted(recovery_seed))
}

fn candidate(
    dae: &PhysicalDae<'_, '_>,
    state: &[Vector],
    initial: &[Constraint<'_>],
    local: LocalPoint<'_>,
) -> SpiceResult<Candidate> {
    if let Some(budget) = local.recovery {
        budget.charge(state)?;
    }
    let point = dae.evaluate_taylor_with_guards(state, local.time, local.scale, local.limit)?;
    let (mandatory_rank, uncertainty) = if local.recovery.is_some() {
        let array = Array::build(&point, state, local.scale, initial)?;
        let prefix = 2 * state[0].len();
        let physical_rank =
            rank::probe_space(&array.matrix, &array.late, &array.optional, prefix, 0)?.rank;
        let bounds = array
            .scale
            .iter()
            .map(|scale| scale.roundoff())
            .collect::<Vec<_>>();
        let uncertainty = rank::projected_uncertainty(
            &array.matrix,
            &bounds,
            &array.late,
            &array.optional,
            prefix,
        )?;
        (physical_rank, Some(uncertainty))
    } else {
        (0, None)
    };
    let to_rate = |vector: &Vector| {
        Vector::from_slice(
            &vector
                .as_slice()
                .iter()
                .map(|value| value / local.scale)
                .collect::<Vec<_>>(),
        )
    };
    Ok(Candidate {
        right: RightLimit {
            state: state[0].clone(),
            rate: to_rate(&state[1]),
            charge: point.charge[0].clone(),
            charge_rate: to_rate(&point.charge[1]),
            local_history: point
                .storage
                .iter()
                .flat_map(|(&slot, charge)| {
                    [
                        (slot, charge.values[0]),
                        (slot + 1, charge.values[1] / local.scale),
                    ]
                })
                .collect(),
            // The two extra orders were rank/affine lookahead checks;
            // retain a lower bound so the next trial still repeats them.
            order_hint: state.len().saturating_sub(3).max(2),
            guards: point.guards.unwrap_or_default(),
            candidate_states: point.candidate_states,
            terminal_currents: point.terminal_currents,
        },
        mandatory_rank,
        uncertainty,
    })
}

#[cfg(test)]
mod assembly_tests {
    use super::*;

    #[test]
    fn folded_storage_rows_include_only_their_own_terms() {
        let mut matrix = SparseMatrix::new(8, 8);
        for (row, col, value) in [(5, 2, 3.), (1, 4, 2.), (5, 0, 1.), (5, 2, -1.)] {
            matrix.add(row, col, value).unwrap();
        }
        matrix.fold_duplicates();
        assert_eq!(
            folded_row(&matrix, 5),
            &[
                Triplet {
                    row: 5,
                    col: 0,
                    value: 1.
                },
                Triplet {
                    row: 5,
                    col: 2,
                    value: 2.
                }
            ]
        );
        for row in [0, 2, 4, 6, 7] {
            assert!(folded_row(&matrix, row).is_empty());
        }
    }

    #[test]
    fn raw_stamp_budget_counts_repeated_contributions_before_folding() {
        let mut matrix = SparseMatrix::new(1, 1);
        let mut stamps = Stamps {
            matrix: &mut matrix,
            remaining: 2,
        };
        stamps.add(0, 0, 1.).unwrap();
        stamps.add(0, 0, -1.).unwrap();
        assert!(stamps.add(0, 0, 2.).is_err());
        matrix.fold_duplicates();
        assert_eq!(matrix.get(0, 0), 0.);
    }
}
