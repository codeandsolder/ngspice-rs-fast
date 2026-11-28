//! Certified scattering-energy bounds for passive waveform reduction.
use super::Real;

// Directed-rounding enclosure for the small immutable scattering Gram
// matrix. All interval operations enclose their exact real results,
// including subnormal products. Failure to certify never grants slack.
#[derive(Clone, Copy)]
struct EnergyInterval {
    lower: Real,
    upper: Real,
}

impl EnergyInterval {
    fn dot(left: &[Real], right: &[Real]) -> Self {
        let mut result = Self {
            lower: 0.,
            upper: 0.,
        };
        for (&a, &b) in left.iter().zip(right) {
            result.lower = a.mul_add(b, result.lower).next_down();
            result.upper = a.mul_add(b, result.upper).next_up();
        }
        result
    }

    fn product(self, other: Self) -> Self {
        let products = [
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        ];
        Self {
            lower: products
                .into_iter()
                .fold(Real::INFINITY, Real::min)
                .next_down(),
            upper: products
                .into_iter()
                .fold(Real::NEG_INFINITY, Real::max)
                .next_up(),
        }
    }

    fn subtract(self, other: Self) -> Self {
        Self {
            lower: (self.lower - other.upper).next_down(),
            upper: (self.upper - other.lower).next_up(),
        }
    }

    const fn reciprocal_positive(self) -> Self {
        Self {
            lower: self.upper.recip().next_down(),
            upper: self.lower.recip().next_up(),
        }
    }
}

// Interval LDL^T of qI - S*S^T. Strictly positive lower pivot
// endpoints certify positive definiteness without trusting a computed
// eigenvalue. The matrix is small and prepared only once per cable.
fn positive_energy_margin(gram: &[EnergyInterval], order: usize, bound: Real) -> bool {
    let mut residual: Vec<_> = gram
        .iter()
        .enumerate()
        .map(|(index, &entry)| {
            let diagonal = if index / order == index % order {
                bound
            } else {
                0.
            };
            EnergyInterval {
                lower: diagonal,
                upper: diagonal,
            }
            .subtract(entry)
        })
        .collect();
    for pivot in 0..order {
        let diagonal = residual[pivot * order + pivot];
        if diagonal.lower <= 0. || !diagonal.upper.is_finite() || !diagonal.lower.is_finite() {
            return false;
        }
        let inverse = diagonal.reciprocal_positive();
        if !inverse.upper.is_finite() {
            return false;
        }
        for col in pivot + 1..order {
            let factor = residual[col * order + pivot].product(inverse);
            if !factor.lower.is_finite() || !factor.upper.is_finite() {
                return false;
            }
            for row in col..order {
                let entry = row * order + col;
                residual[entry] =
                    residual[entry].subtract(factor.product(residual[row * order + pivot]));
                if !residual[entry].lower.is_finite() || !residual[entry].upper.is_finite() {
                    return false;
                }
            }
        }
    }
    true
}

// If ||S||² <= q <= 1, a reduced output may use up to 1/q of
// its exact scattered energy. Summing the per-output prefix inequalities
// still bounds total outgoing energy by incoming energy. Reserve half of
// the certified physical dissipation and keep the interpolation error
// tolerance unchanged. With no proven margin use the original bound 1.
pub(super) fn scattering_energy_factor(rows: &[Vec<(usize, Real)>]) -> Real {
    let order = rows.len();
    let mut dense = vec![vec![0.; order]; order];
    for (values, row) in dense.iter_mut().zip(rows) {
        for &(column, gain) in row {
            values[column] = gain;
        }
    }
    let mut gram = Vec::with_capacity(order * order);
    let mut gershgorin: Real = 0.;
    for left in &dense {
        let mut total: Real = 0.;
        for right in &dense {
            let dot = EnergyInterval::dot(left, right);
            if !dot.lower.is_finite() || !dot.upper.is_finite() {
                return 1.;
            }
            total = (total + dot.lower.abs().max(dot.upper.abs())).next_up();
            gram.push(dot);
        }
        gershgorin = gershgorin.max(total);
    }
    if gershgorin < 1. {
        return gershgorin.midpoint(1.).next_up().min(1.);
    }
    let mut slack = 0.5;
    for _ in 0..20 {
        let bound = 1. - slack;
        if positive_energy_margin(&gram, order, bound) {
            return bound.midpoint(1.).next_up().min(1.);
        }
        slack *= 0.5;
    }
    1.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_scattering_dissipation_bounds_compression_energy() {
        let rows = vec![vec![(0, 0.3), (1, 0.4)], vec![(0, 0.4), (1, -0.3)]];
        let factor = scattering_energy_factor(&rows);
        assert!(factor >= 0.25 && factor < 0.626);
        for input in [[1., 1.], [-1., 2.], [0.25, -0.8]] {
            let incoming = input.iter().map(|&x| x * x).sum::<Real>();
            let scattered = rows
                .iter()
                .map(|row| {
                    let value = row
                        .iter()
                        .map(|&(column, gain)| gain * input[column])
                        .sum::<Real>();
                    value * value
                })
                .sum::<Real>();
            assert!(scattered <= factor * incoming);
        }
        let dense: Vec<_> = (0..3)
            .map(|row| {
                (0..3)
                    .map(|col| {
                        let identity = if row == col { 0.99 } else { 0. };
                        (col, identity - 0.79 / 3.)
                    })
                    .collect()
            })
            .collect();
        let factor = scattering_energy_factor(&dense);
        assert!(factor >= 0.99 * 0.99 && factor < 1.);
        assert_eq!(scattering_energy_factor(&[vec![(0, 1.)]]), 1.);
        assert_eq!(
            scattering_energy_factor(&[vec![(0, 1.), (1, 1.)], vec![]]),
            1.
        );
    }
}
