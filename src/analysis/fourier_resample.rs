//! Local polynomial interpolation on the physical Fourier time axis.
//!
//! Geometry is independent of the trace and can be reused. Exact coordinates
//! read the original sample group; between coordinates choose a local stencil
//! on one continuous span. No artificial perturbation of repeated times and
//! no global-coordinate power-basis fit are necessary.

use super::ordered_samples::{Location, OrderedAxis, SampleError, same_coordinate};
use super::reduction::CompensatedSum;
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::ops::Range;

const MAX_ORDER: usize = 17;

fn failure(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "Fourier interpolation".into(),
        message: message.into(),
    }
}

enum Stencil {
    Direct(Location),
    Polynomial {
        start: usize,
        weights: Range<usize>,
        anchor: usize,
    },
}

/// Shared sampling geometry and packed local Lagrange coefficients.
pub(super) struct PolynomialPlan {
    stencils: Vec<Stencil>,
    weights: Vec<Real>,
}

impl PolynomialPlan {
    pub(super) fn new(axis: &OrderedAxis<'_>, grid: &[Real], degree: usize) -> SpiceResult<Self> {
        if degree == 0 || degree >= MAX_ORDER || axis.len() <= degree {
            return Err(failure(
                "insufficient samples for the requested polynomial degree",
            ));
        }
        let times = axis.as_slice();
        let jumps: Vec<_> = times
            .windows(2)
            .enumerate()
            .filter_map(|(i, pair)| same_coordinate(pair[0], pair[1]).then_some(i))
            .collect();
        let mut stencils = Vec::with_capacity(grid.len());
        let mut weights = Vec::with_capacity(grid.len() * (degree + 1));
        let mut scratch = [0.; MAX_ORDER];
        for &x in grid {
            let location = axis
                .locate(x)
                .ok_or_else(|| failure("polynomial grid outside the time axis"))?;
            let Location::Between(pair) = location else {
                stencils.push(Stencil::Direct(location));
                continue;
            };
            let split = jumps.partition_point(|&left| left < pair.left);
            let lo = if split == 0 { 0 } else { jumps[split - 1] + 1 };
            let hi = jumps.get(split).map_or(times.len(), |&left| left + 1);
            let max_order = (degree + 1).min(hi - lo);
            let mut stencil = Stencil::Direct(Location::Between(pair));
            for order in (3..=max_order).rev() {
                let start = pair
                    .right()
                    .saturating_sub((order - 1).div_ceil(2))
                    .clamp(lo, hi - order);
                let nodes = &times[start..start + order];
                let row = &mut scratch[..order];
                if weights_for(nodes, x, row) {
                    let offset = weights.len();
                    weights.extend_from_slice(row);
                    let near = if x <= Real::midpoint(pair.x0, pair.x1) {
                        pair.left
                    } else {
                        pair.right()
                    };
                    stencil = Stencil::Polynomial {
                        start,
                        weights: offset..weights.len(),
                        anchor: near - start,
                    };
                    break;
                }
            }
            // Linear interpolation is the well-conditioned fallback, including
            // continuous spans too short for the requested polynomial.
            stencils.push(stencil);
        }
        Ok(Self { stencils, weights })
    }

    pub(super) fn sample(&self, grid: &[Real], input: &[Real]) -> SpiceResult<Vec<Real>> {
        self.stencils
            .iter()
            .zip(grid)
            .map(|(stencil, &x)| {
                let value = match stencil {
                    Stencil::Direct(location) => {
                        location
                            .sample(x, |i| Ok(input[i]))
                            .map_err(|error| match error {
                                SampleError::Discontinuous => {
                                    failure("polynomial grid meets a discontinuity")
                                }
                                SampleError::OutOfRange => {
                                    failure("polynomial grid outside the time axis")
                                }
                                SampleError::Nonfinite => failure("nonfinite linear interpolation"),
                                SampleError::Operand(error) => error,
                            })?
                    }
                    Stencil::Polynomial {
                        start,
                        weights,
                        anchor,
                    } => {
                        let row = &self.weights[weights.clone()];
                        local_value(row, &input[*start..*start + row.len()], *anchor)
                    }
                };
                if value.is_finite() {
                    Ok(value)
                } else {
                    Err(failure("nonfinite local polynomial interpolation"))
                }
            })
            .collect()
    }

    /// Conservative peak plan storage and geometry work for preflight. The
    /// work includes retrying all lower orders and both axis index searches.
    pub(super) fn budget(rows: usize, count: usize, order: usize) -> (usize, usize) {
        let index_cost = rows.checked_ilog2().unwrap_or(0) as usize + 1;
        let polynomial_cost = (3..=order).map(|n| n * n).sum::<usize>();
        let work = rows.saturating_add(count.saturating_mul(2 * index_cost + polynomial_cost));
        let bytes = rows
            .saturating_mul(size_of::<usize>())
            .saturating_add(count.saturating_mul(size_of::<Stencil>() + order * size_of::<Real>()));
        (work, bytes)
    }
}

/// Ratio of two local coordinate differences. Halving only when an
/// intermediate difference overflows preserves small ordinary intervals.
fn ratio(x: Real, y: Real, a: Real, b: Real) -> Real {
    let numerator = x - y;
    let denominator = a - b;
    if numerator.is_finite() && denominator.is_finite() {
        numerator / denominator
    } else {
        let half_y = y * 0.5;
        let half_b = b * 0.5;
        let half_x = x * 0.5;
        let half_a = a * 0.5;
        (half_x - half_y) / (half_a - half_b)
    }
}

fn weights_for(nodes: &[Real], x: Real, weights: &mut [Real]) -> bool {
    let mut norm = 0.;
    for (i, weight) in weights.iter_mut().enumerate() {
        *weight = 1.;
        for (j, &node) in nodes.iter().enumerate() {
            if i != j {
                *weight *= ratio(x, node, nodes[i], node);
            }
        }
        norm += weight.abs();
    }
    // The L1 norm of interpolation weights bounds amplification of sample
    // error. A conservative 8 roundings per node bounds ordinary rounding
    // error by 8 * order * epsilon * norm times the sample scale.
    // Reduce order rather than evaluate an ill-conditioned stencil whose
    // estimated rounding error exceeds sqrt(epsilon) of that scale.
    let order = u32::try_from(nodes.len()).unwrap_or(u32::MAX);
    let error_bound = 8. * Real::from(order) * Real::EPSILON * norm;
    error_bound <= Real::EPSILON.sqrt()
}

fn local_value(weights: &[Real], values: &[Real], anchor_index: usize) -> Real {
    let anchor = values[anchor_index];
    let mut sum = CompensatedSum::default();
    sum.add(anchor);
    for (&weight, &value) in weights.iter().zip(values) {
        sum.add(weight * (value - anchor));
    }
    let result = sum.total();
    if result.is_finite() {
        return result;
    }

    // Opposite-sign extreme ordinates or intermediate weighted terms may
    // overflow even though the polynomial value is representable. Scale
    // locally and only on that exceptional path.
    let scale = values.iter().fold(0_f64, |m, value| m.max(value.abs()));
    if scale == 0. {
        return anchor;
    }
    let scaled_anchor = anchor / scale;
    let mut sum = CompensatedSum::default();
    sum.add(scaled_anchor);
    for (&weight, &value) in weights.iter().zip(values) {
        let difference = value - anchor;
        let scaled = if difference.is_finite() {
            difference / scale
        } else {
            value / scale - scaled_anchor
        };
        sum.add(weight * scaled);
    }
    sum.total() * scale
}

#[cfg(test)]
mod tests {
    use super::{OrderedAxis, PolynomialPlan};

    #[test]
    fn local_polynomial_reproduces_cubic_at_large_absolute_times() {
        let offset = 1e12;
        let times = [0., 1., 3., 4., 7., 9.].map(|x| offset + x);
        let polynomial = |t: f64| {
            let x = t - offset;
            x.powi(3) - 2. * x + 5.
        };
        let values = times.map(polynomial);
        let grid = [0., 0.5, 2., 3.5, 8., 9.].map(|x| offset + x);
        let axis = OrderedAxis::borrowed(&times).unwrap();
        let plan = PolynomialPlan::new(&axis, &grid, 3).unwrap();
        let actual = plan.sample(&grid, &values).unwrap();
        for (&t, got) in grid.iter().zip(actual) {
            let want = polynomial(t);
            assert!((got - want).abs() <= 16. * f64::EPSILON * want.abs().max(1.));
        }
    }

    #[test]
    fn repeated_times_are_exact_groups_and_split_polynomial_stencils() {
        let axis = OrderedAxis::borrowed(&[0., 1., 2., 2., 3., 4.]).unwrap();
        let grid = [0., 0.5, 1.5, 2.5, 3.5, 4.];
        let values = [-0., 1., 4., 10., 11., 14.];
        let plan = PolynomialPlan::new(&axis, &grid, 4).unwrap();
        let result = plan.sample(&grid, &values).unwrap();
        assert_eq!(result[0].to_bits(), (-0_f64).to_bits());
        for (got, want) in result[1..].iter().zip([0.25, 2.25, 10.25, 12.25, 14.]) {
            assert!((got - want).abs() <= 8. * f64::EPSILON * want);
        }
        let exact = [2.];
        let plan = PolynomialPlan::new(&axis, &exact, 4).unwrap();
        assert!(plan.sample(&exact, &values).is_err());
        assert_eq!(
            plan.sample(&exact, &[0., 1., 4., 4., 5., 7.]).unwrap(),
            [4.]
        );
    }

    #[test]
    fn ill_conditioned_stencil_reduces_degree_instead_of_amplifying_roundoff() {
        let times = [0., f64::EPSILON, 1., 2., 3.];
        let grid = [0.5];
        let axis = OrderedAxis::borrowed(&times).unwrap();
        let plan = PolynomialPlan::new(&axis, &grid, 4).unwrap();
        assert_eq!(plan.sample(&grid, &times).unwrap(), [0.5]);
        assert!(plan.weights.len() < 5);
    }

    #[test]
    fn local_cubic_matches_high_precision_interpolant_on_c_saved_samples() {
        let times = [
            0.003993280000000206,
            0.003994280000000206,
            0.003995280000000206,
            0.003996280000000206,
        ];
        let values = [
            0.4577895393556249,
            0.4640679166130364,
            0.47034771240757456,
            0.4766286788236575,
        ];
        let grid = [0.0039949, 0.003995];
        let axis = OrderedAxis::borrowed(&times).unwrap();
        let plan = PolynomialPlan::new(&axis, &grid, 3).unwrap();
        let result = plan.sample(&grid, &values).unwrap();
        // Decimal 80-digit Lagrange evaluation of the exact binary64 inputs.
        // C's absolute-power fit differs from these by about 3e-13 and 8e-13.
        for (got, want) in result
            .into_iter()
            .zip([0.46796123867108974, 0.4685892409228076])
        {
            assert!((got - want).abs() <= 8. * f64::EPSILON * want);
        }
    }
}
