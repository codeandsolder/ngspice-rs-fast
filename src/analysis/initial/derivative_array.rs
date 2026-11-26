//! Coefficient-space Newton system for the right-limit physical DAE.
//!
//! For x(s) = sum x[k] s^k, t = h*s, write the exact differentiated
//! residual F[k] + (k+1)/h Q[k+1] = 0. The Newton derivative is a
//! convolution of the Taylor coefficients of the physical Jacobians
//! with the correction to x. This preserves the derivative-array
//! structure: a high-order free coefficient cannot be made to constrain
//! a lower one by interpolation noise in an unrelated block.
//!
//! Retain all original equations for residual verification. Rank selection
//! is only a solve strategy, never a relaxation of a physical equation.

use super::{failure, system::rank};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Real, SpiceResult};

/// Bounds for a single derivative-array solve, including sparse fill work.
const MAX_COORDINATES: usize = 32_768;
const MAX_STAMPS: usize = 2_000_000;
const MAX_RANK_WORK: usize = 20_000_000;

use crate::devices::physical_jet::ResidualJet;

/// A constraint on the actual initial state, expressed as a Newton
/// correction. Physical storage wins over optional derived input seeds.
pub(super) struct InitialConstraint {
    pub(super) terms: Vec<(usize, Real)>,
    pub(super) residual: Real,
    pub(super) optional: bool,
}

pub(super) struct DerivativeArray {
    matrix: SparseMatrix,
    rhs: Vec<Real>,
    optional: Vec<usize>,
    dimension: usize,
    order: usize,
}

/// One projected Newton correction, including the independent initial
/// constraints selected by the rank-aware solve.
pub(super) struct Correction {
    pub(super) coefficients: Vec<Vector>,
    pub(super) retained: Vec<bool>,
}

fn derivative_factor(index: usize, scale: Real) -> SpiceResult<Real> {
    let index = u32::try_from(index).map_err(|_| failure("derivative order exceeds range"))?;
    let factor = Real::from(index) / scale;
    if !factor.is_finite() {
        return Err(failure("right-limit derivative scaling overflow"));
    }
    Ok(factor)
}

impl DerivativeArray {
    /// Assemble one truncated derivative array at the supplied Taylor
    /// trajectory. A degree-p trajectory imposes physical residual orders
    /// zero through p-1. Its highest coefficient need not be determined.
    pub(super) fn assemble(
        static_part: &ResidualJet,
        charge: &ResidualJet,
        constraints: &[InitialConstraint],
        scale: Real,
    ) -> SpiceResult<Self> {
        let order = static_part.values.len();
        let dimension = static_part.values.first().map_or(0, Vector::len);
        let levels = order
            .checked_add(1)
            .ok_or_else(|| failure("right-limit derivative order overflow"))?;
        let columns = levels
            .checked_mul(dimension)
            .filter(|&count| count <= MAX_COORDINATES)
            .ok_or_else(|| failure("right-limit derivative coordinate limit exceeded"))?;
        let rows = order
            .checked_mul(dimension)
            .and_then(|count| count.checked_add(constraints.len()))
            .filter(|&count| count <= MAX_COORDINATES)
            .ok_or_else(|| failure("right-limit derivative equation limit exceeded"))?;
        let valid_jet = |jet: &ResidualJet, count: usize| {
            jet.values.len() == count
                && jet.jacobians.len() == count
                && jet
                    .values
                    .iter()
                    .all(|v| v.len() == dimension && v.is_finite())
                && jet
                    .jacobians
                    .iter()
                    .all(|m| m.rows() == dimension && m.cols() == dimension)
        };
        if order == 0
            || dimension == 0
            || !scale.is_finite()
            || scale == 0.
            || !valid_jet(static_part, order)
            || !valid_jet(charge, levels)
        {
            return Err(failure("invalid right-limit physical residual jet"));
        }
        let mut matrix = SparseMatrix::new(rows, columns);
        let mut rhs = Vec::with_capacity(rows);
        let mut optional = Vec::new();
        let mut budget = MAX_STAMPS;
        for degree in 0..order {
            let factor = derivative_factor(degree + 1, scale)?;
            rhs.extend(
                static_part.values[degree]
                    .as_slice()
                    .iter()
                    .zip(charge.values[degree + 1].as_slice())
                    .map(|(&f, &q)| -factor.mul_add(q, f)),
            );
            // A coefficient's contribution uses only the compatible
            // Taylor order. No numerical rank noise above this band.
            for coefficient in 0..=degree + 1 {
                for (jet, index, multiplier) in [
                    (static_part, degree.checked_sub(coefficient), 1.),
                    (charge, Some(degree + 1 - coefficient), factor),
                ] {
                    let Some(index) = index else { continue };
                    let entries = jet.jacobians[index].triplets();
                    budget = budget.checked_sub(entries.len()).ok_or_else(|| {
                        failure("right-limit derivative matrix stamp limit exceeded")
                    })?;
                    for term in entries {
                        matrix.add(
                            degree * dimension + term.row,
                            coefficient * dimension + term.col,
                            multiplier * term.value,
                        )?;
                    }
                }
            }
        }
        for constraint in constraints {
            let row = rhs.len();
            budget = budget
                .checked_sub(constraint.terms.len())
                .ok_or_else(|| failure("right-limit initial constraint stamp limit exceeded"))?;
            rhs.push(-constraint.residual);
            if constraint.optional {
                optional.push(row);
            }
            for &(column, value) in &constraint.terms {
                if column >= columns {
                    return Err(failure("right-limit constraint outside derivative array"));
                }
                matrix.add(row, column, value)?;
            }
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(failure("nonfinite right-limit derivative residual"));
        }
        matrix.fold_duplicates();
        Ok(Self {
            matrix,
            rhs,
            optional,
            dimension,
            order,
        })
    }

    /// Backward-error check of every original equation, including rows
    /// dependent in the rank-reduced system. Only unretained optional ICs
    /// are excluded. Each coefficient row is scaled independently.
    pub(super) fn residual_ratio(
        &self,
        trajectory: &[Vector],
        retained: &[bool],
        relative: Real,
        absolute: Real,
    ) -> SpiceResult<Real> {
        let mut scales = vec![0.; self.rhs.len()];
        for term in self.matrix.triplets() {
            let value = trajectory[term.col / self.dimension].as_slice()[term.col % self.dimension];
            scales[term.row] = term.value.abs().mul_add(value.abs(), scales[term.row]);
        }
        let physical_rows = self.order * self.dimension;
        let mut ratio: Real = 0.;
        for (row, residual) in self.rhs.iter().enumerate() {
            if self.optional.binary_search(&row).is_ok()
                && !retained.get(row - physical_rows).copied().unwrap_or(false)
            {
                continue;
            }
            let scale = scales[row];
            let tolerance = relative.mul_add(scale, absolute);
            if !scale.is_finite() || !tolerance.is_finite() || tolerance <= 0. {
                return Err(failure("invalid right-limit residual scaling"));
            }
            ratio = ratio.max(residual.abs() / tolerance);
        }
        Ok(ratio)
    }

    /// Solve only when the requested coefficient coordinates are fixed by
    /// the residual and initial constraints. Unneeded high-order freedoms
    /// may be gauged after that proof; never gauge a required derivative.
    pub(super) fn solve(
        &self,
        required: &[usize],
        gauges: &[usize],
    ) -> SpiceResult<Option<Correction>> {
        let Some(result) = rank::complete_projected(
            &self.matrix,
            &self.rhs,
            gauges,
            &[],
            &self.optional,
            required,
            MAX_RANK_WORK,
        )?
        else {
            return Ok(None);
        };
        let coefficients = result
            .values
            .as_slice()
            .chunks_exact(self.dimension)
            .map(Vector::from_slice)
            .collect();
        Ok(Some(Correction {
            coefficients,
            retained: result.independent[self.order * self.dimension..].to_vec(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Arbitrary length feed-forward chain, including current/storage rows.
    /// The first equation is prescribed input, each remaining equation
    /// differentiates its predecessor. All required coordinates must be
    /// determined by the physical rows, not zero gauges.
    fn chain(length: usize, order: usize, scale: Real) -> DerivativeArray {
        let mut static_values = Vec::new();
        let mut charge_values = Vec::new();
        let mut static_jacobians = Vec::new();
        let mut charge_jacobians = Vec::new();
        // x[0](t) = 2 + 3t + 5t² + 7t³.
        let source = [2., 3. * scale, 5. * scale.powi(2), 7. * scale.powi(3)];
        for degree in 0..=order {
            let mut g = SparseMatrix::new(length, length);
            let mut c = SparseMatrix::new(length, length);
            let mut f = Vector::zeros(length);
            if degree == 0 {
                for row in 0..length {
                    g.add(row, row, 1.).unwrap();
                    if row > 0 {
                        c.add(row, row - 1, -1.).unwrap();
                    }
                }
                g.fold_duplicates();
                c.fold_duplicates();
            }
            if degree < order {
                f.as_mut_slice()[0] = -source.get(degree).copied().unwrap_or(0.);
                static_values.push(f);
                static_jacobians.push(g);
            }
            charge_values.push(Vector::zeros(length));
            charge_jacobians.push(c);
        }
        DerivativeArray::assemble(
            &ResidualJet {
                values: static_values,
                jacobians: static_jacobians,
            },
            &ResidualJet {
                values: charge_values,
                jacobians: charge_jacobians,
            },
            &[],
            scale,
        )
        .unwrap()
    }

    #[test]
    fn nested_derivatives_require_enough_physical_orders() {
        for length in [2, 4, 24] {
            let required = (0..length).collect::<Vec<_>>();
            assert!(
                chain(length, length - 1, 0.125)
                    .solve(&required, &[])
                    .unwrap()
                    .is_none()
            );
            let result = chain(length, length, 0.125)
                .solve(&required, &[])
                .unwrap()
                .unwrap();
            let expected = [2., 3., 10., 42.];
            for (index, &actual) in result.coefficients[0].as_slice().iter().enumerate() {
                let expected = expected.get(index).copied().unwrap_or(0.);
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "{index}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn initial_conditions_fix_only_independent_state() {
        // x' = 2x; the genuine independent initial coordinate is supplied
        // by a physical constraint, and a conflicting optional seed loses.
        let mut g = SparseMatrix::new(1, 1);
        g.add(0, 0, -2.).unwrap();
        let mut c = SparseMatrix::new(1, 1);
        c.add(0, 0, 1.).unwrap();
        let empty = || SparseMatrix::new(1, 1);
        let jet = |jacobians: Vec<SparseMatrix>| ResidualJet {
            values: jacobians.iter().map(|_| Vector::zeros(1)).collect(),
            jacobians,
        };
        let array = DerivativeArray::assemble(
            &jet(vec![g, empty()]),
            &jet(vec![c, empty(), empty()]),
            &[
                InitialConstraint {
                    terms: vec![(0, 1.)],
                    residual: -3.,
                    optional: false,
                },
                InitialConstraint {
                    terms: vec![(0, 1.)],
                    residual: -5.,
                    optional: true,
                },
            ],
            0.25,
        )
        .unwrap();
        let solution = array.solve(&[0, 1], &[]).unwrap().unwrap();
        assert_eq!(solution.retained, [true, false]);
        assert_eq!(
            solution
                .coefficients
                .iter()
                .map(|v| v.as_slice()[0])
                .collect::<Vec<_>>(),
            [3., 1.5, 0.375]
        );
    }
}
