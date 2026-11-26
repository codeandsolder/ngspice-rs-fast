//! Coupled physical derivative arrays. A normalized time coefficient X[k]
//! represents x^(k)(0) * h^k/k!. The equation in degree k is
//! F[k] + (k+1)/h Q[k+1] = 0, with state Jacobians acting by convolution.
//! Higher-degree variables are projected away before initial constraints
//! are selected. No arbitrary zero derivative gauge enters the prefix.

use super::rank;
use crate::analysis::initial::{
    ChargeInitial, ImpulseStorage, Tolerances,
    constraints::{AbsoluteRow, VoltageVertex},
    failure,
};
use crate::devices::physical_dae::{PhysicalDae, PhysicalDaeTaylor};
use crate::devices::{
    Circuit, InitialCoordinateHistory, InitialStoragePolicy, Limit, LoadRequest, StateHistory,
};
use crate::maths::integrator::{ChargeHistory, Coefficients};
use crate::maths::taylor::MAX_ORDER;
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Real, SpiceResult};

const MAX_ARRAY_ENTRIES: usize = 8_388_608;
const MAX_ARRAY_VARIABLES: usize = 1_048_576;

#[derive(Clone, Copy)]
enum Constraint<'a> {
    Coordinate(usize, Real),
    Voltage(VoltageVertex, VoltageVertex, Real),
    Charge(usize, &'a ChargeInitial),
    Companion(CompanionConstraint<'a>),
}

impl Constraint<'_> {
    fn optional(self) -> bool {
        matches!(self,Self::Charge(_,charge) if charge.policy==InitialStoragePolicy::Derived)
            || matches!(self, Self::Companion(c) if c.derived)
    }

    fn stamp(
        self,
        row: usize,
        state: &[Vector],
        time_scale: Real,
        point: &PhysicalDaeTaylor,
        matrix: &mut SparseMatrix,
    ) -> SpiceResult<(Real, Real)> {
        if let Self::Companion(companion) = self {
            return companion.stamp(row, state, time_scale, point, matrix);
        }
        let state = &state[0];
        let mut scale = 0.;
        let value = match self {
            Self::Coordinate(column, target) => {
                matrix.add(row, column, 1.)?;
                let value = state.as_slice()[column];
                scale = value.abs().max(target.abs());
                value - target
            }
            Self::Voltage(positive, negative, target) => {
                let mut value = -target;
                scale = target.abs();
                for (vertex, sign) in [(positive, 1.), (negative, -1.)] {
                    if let VoltageVertex::Node(AbsoluteRow(column)) = vertex {
                        matrix.add(row, column, sign)?;
                        let voltage = state.as_slice()[column];
                        value = sign.mul_add(voltage, value);
                        scale += voltage.abs();
                    }
                }
                value
            }
            Self::Companion(_) => return Err(failure("unexpected companion constraint")),
            Self::Charge(equation, charge) => {
                for term in point.capacitance[0]
                    .triplets()
                    .iter()
                    .filter(|term| term.row == equation)
                {
                    matrix.add(row, term.col, term.value)?;
                    scale = term
                        .value
                        .abs()
                        .mul_add(state.as_slice()[term.col].abs(), scale);
                }
                let value = point.charge[0].as_slice()[equation];
                scale = scale.max(value.abs()).max(charge.target.abs());
                value - charge.target
            }
        };
        Ok((value, scale))
    }

    fn absolute(self, tol: Tolerances) -> Real {
        match self {
            Self::Voltage(..) => tol.vntol,
            Self::Charge(_, charge) if charge.policy == InitialStoragePolicy::Derived => tol.vntol,
            Self::Charge(..) => tol.chgtol,
            Self::Companion(c) if c.derived => tol.vntol,
            Self::Coordinate(..) | Self::Companion(_) => tol.abstol,
        }
    }
}

fn constraints(storage: &ImpulseStorage) -> Vec<Constraint<'_>> {
    storage
        .fixed_coordinates
        .iter()
        .map(|(&AbsoluteRow(row), item)| Constraint::Coordinate(row, item.value))
        .chain(
            storage
                .capacitors
                .iter()
                .map(|item| Constraint::Voltage(item.positive, item.negative, item.voltage)),
        )
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
}

impl StoredQuantity {
    fn value(self, state: &[Vector], point: &PhysicalDaeTaylor, degree: usize) -> Real {
        match self {
            Self::Coordinate(column) => state[degree].as_slice()[column],
            Self::Voltage(p, n, scale) => {
                scale
                    * (crate::analysis::initial::potential(&state[degree], p)
                        - crate::analysis::initial::potential(&state[degree], n))
            }
            Self::Equation(row, scale) => scale * point.charge[degree].as_slice()[row],
        }
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
                for term in point.capacitance[degree]
                    .triplets()
                    .iter()
                    .filter(|term| term.row == row)
                {
                    visit(term.col, scale * term.value)?;
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
        matrix: &mut SparseMatrix,
    ) -> SpiceResult<(Real, Real)> {
        let accepted = |age, offset| {
            self.history
                .accepted(age)
                .and_then(|values| values.get(self.slot + offset))
                .copied()
                .ok_or_else(|| failure("missing independent storage history"))
        };
        let value = self.quantity.value(state, point, 0);
        let rate = self.quantity.value(state, point, 1) / time_scale;
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
        Ok((rate - companion.derivative, scale))
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
    for (&AbsoluteRow(row), charge) in &storage.charges {
        add(
            StoredQuantity::Equation(row, 1.),
            charge.slot,
            charge.policy == InitialStoragePolicy::Derived,
        );
    }
    for cap in &storage.capacitors {
        add(
            StoredQuantity::Voltage(cap.positive, cap.negative, cap.scale),
            cap.slot,
            false,
        );
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
    scale: Vec<Real>,
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
            .ok_or_else(|| failure("derivative array row count overflow"))?;
        let mut array = Self {
            matrix: SparseMatrix::new(rows, variables),
            residual: crate::primitives::filled_vec(rows, 0., "derivative array residual")?,
            scale: crate::primitives::filled_vec(rows, 0., "derivative array residual scale")?,
            physical_rows,
            late: Vec::new(),
            optional: Vec::new(),
        };
        let mut degree = 1.;
        let mut entries = 0_usize;
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
                array.scale[offset + r] = multiplier.mul_add(q.abs(), f.abs());
            }
            for (j, coefficient_state) in state.iter().enumerate().take(k + 2) {
                for (matrix, factor) in [
                    (point.conductance.get(k.wrapping_sub(j)), 1.),
                    (point.capacitance.get(k + 1 - j), multiplier),
                ] {
                    let Some(matrix) = matrix else { continue };
                    entries = entries.saturating_add(matrix.triplets().len());
                    if entries > MAX_ARRAY_ENTRIES {
                        return Err(failure("derivative array exceeds its stamp budget"));
                    }
                    for term in matrix.triplets() {
                        let row = offset + term.row;
                        let coefficient = factor * term.value;
                        array.matrix.add(row, j * width + term.col, coefficient)?;
                        array.scale[row] = coefficient.abs().mul_add(
                            coefficient_state.as_slice()[term.col].abs(),
                            array.scale[row],
                        );
                    }
                }
            }
            degree += 1.;
        }
        for (index, constraint) in initial.iter().copied().enumerate() {
            let row = physical_rows + index;
            let (residual, scale) =
                constraint.stamp(row, state, scale, point, &mut array.matrix)?;
            array.residual[row] = residual;
            array.scale[row] = scale;
            if constraint.optional() {
                array.optional.push(row);
            } else {
                array.late.push(row);
            }
        }
        array.matrix.fold_duplicates();
        if array
            .residual
            .iter()
            .chain(&array.scale)
            .any(|v| !v.is_finite())
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
            let absolute = if row < self.physical_rows {
                // Physical equations have mixed dimensions. Their Jacobian
                // contribution scale is the primary backward-error measure.
                // Use the smallest configured absolute tolerance as a
                // fallback; mandatory ICs below retain their actual units.
                tol.abstol.min(tol.vntol).min(tol.chgtol)
            } else {
                let constraint = initial[row - self.physical_rows];
                if (constraint.optional() || matches!(constraint, Constraint::Companion(_)))
                    && !independent[row]
                {
                    continue;
                }
                constraint.absolute(tol)
            };
            user = user.max(residual.abs() / tol.reltol.mul_add(scale, absolute));
            roundoff = roundoff
                .max(residual.abs() / (32. * Real::EPSILON * scale).max(Real::MIN_POSITIVE));
        }
        (user, roundoff)
    }
}

pub(in crate::analysis::initial) struct RightLimit {
    pub state: Vector,
    pub rate: Vector,
    pub charge: Vector,
    pub charge_rate: Vector,
    pub order_hint: usize,
}

#[derive(Clone, Copy)]
struct LocalPoint {
    time: Real,
    scale: Real,
    limit: Limit,
    order_hint: usize,
}

impl LocalPoint {
    fn evaluate(
        self,
        dae: &PhysicalDae<'_, '_>,
        state: &[Vector],
    ) -> SpiceResult<PhysicalDaeTaylor> {
        dae.evaluate_taylor(state, self.time, self.scale, self.limit)
    }

    fn prefix_rank(self, dae: &PhysicalDae<'_, '_>, state: &[Vector]) -> SpiceResult<usize> {
        let point = self.evaluate(dae, state)?;
        let array = Array::build(&point, state, self.scale, &[])?;
        rank::projected_rank(&array.matrix, 2 * state[0].len())
    }
}

fn extend(state: &mut Vec<Vector>) -> SpiceResult<()> {
    let width = state[0].len();
    if state.len() > MAX_ORDER || width.saturating_mul(state.len() + 1) > MAX_ARRAY_VARIABLES {
        return Err(failure(
            "physical DAE derivative order or state budget exhausted: required coordinates may be noncausal or underdetermined",
        ));
    }
    state.push(Vector::zeros(width));
    Ok(())
}

fn stable_rank_order(
    dae: &PhysicalDae<'_, '_>,
    state: &mut Vec<Vector>,
    local: LocalPoint,
) -> SpiceResult<()> {
    let mut previous = local.prefix_rank(dae, state)?;
    loop {
        extend(state)?;
        let next = local.prefix_rank(dae, state)?;
        if next == previous {
            return Ok(());
        }
        previous = next;
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ArrayNewton {
    Converged,
    NeedOrder,
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
    local: LocalPoint,
    initial: &[Constraint<'_>],
    tol: Tolerances,
    iterations: usize,
) -> SpiceResult<ArrayNewton> {
    let width = state[0].len();
    let mut best: Option<(Real, Vec<Vector>)> = None;
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
            apply_step(state, &step.values)?;
            return Ok(ArrayNewton::NeedOrder);
        }
        let (ratio, roundoff) = array.merit(initial, &step.independent, tol);
        if ratio <= 1. {
            if roundoff <= 1. {
                return Ok(ArrayNewton::Converged);
            }
            if best.as_ref().is_some_and(|(score, _)| roundoff >= *score) {
                break;
            }
            best = Some((roundoff, state.to_vec()));
        }
        apply_step(state, &step.values)?;
    }
    if let Some((_, best)) = best {
        state.clone_from_slice(&best);
        Ok(ArrayNewton::Converged)
    } else {
        Err(failure(
            "physical right-limit derivative array did not converge; initial constraints may require an impulse",
        ))
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
        },
        tol,
        iterations,
    )
}

pub(in crate::analysis::initial) fn solve_event(
    dae: &PhysicalDae<'_, '_>,
    request: &LoadRequest<'_>,
    storage: &ImpulseStorage,
    tol: Tolerances,
    iterations: usize,
    order_hint: usize,
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
            order_hint,
        },
        tol,
        iterations,
    )
}

pub(in crate::analysis::initial) fn solve_step(
    dae: &PhysicalDae<'_, '_>,
    circuit: &Circuit,
    request: &LoadRequest<'_>,
    storage: &ImpulseStorage,
    tol: Tolerances,
    iterations: usize,
    order_hint: usize,
) -> SpiceResult<RightLimit> {
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
            order_hint,
        },
        tol,
        iterations,
    )
}

fn solve(
    dae: &PhysicalDae<'_, '_>,
    seed: &Vector,
    initial: &[Constraint<'_>],
    local: LocalPoint,
    tol: Tolerances,
    iterations: usize,
) -> SpiceResult<RightLimit> {
    let mut state = vec![seed.clone()];
    for _ in 0..local.order_hint.max(2) {
        extend(&mut state)?;
    }
    loop {
        stable_rank_order(dae, &mut state, local)?;
        if newton(dae, &mut state, local, initial, tol, iterations)? == ArrayNewton::NeedOrder {
            extend(&mut state)?;
            continue;
        }
        let rank = local.prefix_rank(dae, &state)?;
        extend(&mut state)?;
        if local.prefix_rank(dae, &state)? == rank {
            // Re-solve the extended array: rank equality alone does not
            // establish its affine residual or the chosen right-hand branch.
            if newton(dae, &mut state, local, initial, tol, iterations)? == ArrayNewton::NeedOrder
                || local.prefix_rank(dae, &state)? != rank
            {
                continue;
            }
            let point = local.evaluate(dae, &state)?;
            let to_rate = |vector: &Vector| {
                Vector::from_slice(
                    &vector
                        .as_slice()
                        .iter()
                        .map(|value| value / local.scale)
                        .collect::<Vec<_>>(),
                )
            };
            return Ok(RightLimit {
                state: state[0].clone(),
                rate: to_rate(&state[1]),
                charge: point.charge[0].clone(),
                charge_rate: to_rate(&point.charge[1]),
                // The two extra orders were rank/affine lookahead checks;
                // retain a lower bound so the next trial still repeats them.
                order_hint: state.len().saturating_sub(3).max(2),
            });
        }
    }
}
