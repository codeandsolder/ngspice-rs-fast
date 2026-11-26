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
use crate::devices::InitialStoragePolicy;
use crate::devices::physical_dae::{PhysicalDae, PhysicalDaeTaylor};
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
}

impl Constraint<'_> {
    fn optional(self) -> bool {
        matches!(self,Self::Charge(_,charge) if charge.policy==InitialStoragePolicy::Derived)
    }

    fn stamp(
        self,
        row: usize,
        state: &Vector,
        point: &PhysicalDaeTaylor,
        matrix: &mut SparseMatrix,
    ) -> SpiceResult<(Real, Real)> {
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
            Self::Charge(equation, charge) => {
                for term in point.capacitance[0]
                    .triplets()
                    .iter()
                    .filter(|term| term.row == equation)
                {
                    matrix.add(row, term.col, term.value)?;
                    scale += term.value.abs() * state.as_slice()[term.col].abs();
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
            Self::Coordinate(..) => tol.abstol,
            Self::Voltage(..) => tol.vntol,
            Self::Charge(_, charge) if charge.policy == InitialStoragePolicy::Derived => tol.vntol,
            Self::Charge(..) => tol.chgtol,
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
                array.scale[offset + r] = f.abs() + multiplier * q.abs();
            }
            for j in 0..=k + 1 {
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
                        array.scale[row] += coefficient.abs() * state[j].as_slice()[term.col].abs();
                    }
                }
            }
            degree += 1.;
        }
        for (index, constraint) in initial.iter().copied().enumerate() {
            let row = physical_rows + index;
            let (residual, scale) = constraint.stamp(row, &state[0], point, &mut array.matrix)?;
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
                if constraint.optional() && !independent[row] {
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
}

fn initial_prefix_rank(
    dae: &PhysicalDae<'_, '_>,
    state: &[Vector],
    scale: Real,
) -> SpiceResult<usize> {
    let point = dae.evaluate_taylor(state, 0., scale)?;
    let array = Array::build(&point, state, scale, &[])?;
    rank::projected_rank(&array.matrix, 2 * state[0].len())
}

fn extend(state: &mut Vec<Vector>) -> SpiceResult<()> {
    let width = state[0].len();
    if state.len() > MAX_ORDER || width.saturating_mul(state.len() + 1) > MAX_ARRAY_VARIABLES {
        return Err(failure(
            "physical DAE initial derivative order or state budget exhausted",
        ));
    }
    state.push(Vector::zeros(width));
    Ok(())
}

fn stable_rank_order(
    dae: &PhysicalDae<'_, '_>,
    state: &mut Vec<Vector>,
    scale: Real,
) -> SpiceResult<()> {
    let mut previous = initial_prefix_rank(dae, state, scale)?;
    loop {
        extend(state)?;
        let next = initial_prefix_rank(dae, state, scale)?;
        if next == previous {
            return Ok(());
        }
        previous = next;
    }
}

fn newton(
    dae: &PhysicalDae<'_, '_>,
    state: &mut [Vector],
    scale: Real,
    initial: &[Constraint<'_>],
    tol: Tolerances,
    iterations: usize,
) -> SpiceResult<()> {
    let width = state[0].len();
    let mut best: Option<(Real, Vec<Vector>)> = None;
    for _ in 0..iterations {
        let point = dae.evaluate_taylor(state, 0., scale)?;
        let array = Array::build(&point, state, scale, initial)?;
        let rhs = array
            .residual
            .iter()
            .map(|value| -value)
            .collect::<Vec<_>>();
        let step = rank::complete(
            &array.matrix,
            &rhs,
            &[],
            &array.late,
            &array.optional,
            2 * width,
        )?;
        let (ratio, roundoff) = array.merit(initial, &step.independent, tol);
        if ratio <= 1. {
            if roundoff <= 1. {
                return Ok(());
            }
            if best.as_ref().is_some_and(|(score, _)| roundoff >= *score) {
                break;
            }
            best = Some((roundoff, state.to_vec()));
        }
        for (jet, delta) in state
            .iter_mut()
            .zip(step.values.as_slice().chunks_exact(width))
        {
            for (value, &delta) in jet.as_mut_slice().iter_mut().zip(delta) {
                *value += delta;
            }
        }
    }
    if let Some((_, best)) = best {
        state.clone_from_slice(&best);
        Ok(())
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
    let initial = constraints(storage);
    let mut state = vec![
        seed.clone(),
        Vector::zeros(seed.len()),
        Vector::zeros(seed.len()),
    ];
    loop {
        stable_rank_order(dae, &mut state, scale)?;
        newton(dae, &mut state, scale, &initial, tol, iterations)?;
        let rank = initial_prefix_rank(dae, &state, scale)?;
        extend(&mut state)?;
        if initial_prefix_rank(dae, &state, scale)? == rank {
            // Re-solve the extended array: rank equality alone does not
            // establish its affine residual or the chosen right-hand branch.
            newton(dae, &mut state, scale, &initial, tol, iterations)?;
            let point = dae.evaluate_taylor(&state, 0., scale)?;
            let to_rate = |vector: &Vector| {
                Vector::from_slice(
                    &vector
                        .as_slice()
                        .iter()
                        .map(|value| value / scale)
                        .collect::<Vec<_>>(),
                )
            };
            return Ok(RightLimit {
                state: state[0].clone(),
                rate: to_rate(&state[1]),
                charge: point.charge[0].clone(),
                charge_rate: to_rate(&point.charge[1]),
            });
        }
    }
}
