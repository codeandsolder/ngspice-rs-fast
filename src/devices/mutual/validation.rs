//! Bounded validation of the normalized, coalesced inductance matrix.
//!
//! A sparse certificate is deliberately stronger than the historical
//! eigenvalue tolerance. It never turns an ambiguous matrix into acceptance.

use super::{CouplingSite, MutualLimits};
use crate::maths::{Matrix, dense::symmetric_eigenvalue_cost};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// An upper budget for the work of both diagonal conventions over all groups.
pub(in crate::devices) struct ValidationBudget {
    limits: MutualLimits,
    remaining: usize,
}

impl ValidationBudget {
    pub(in crate::devices) const fn new(limits: MutualLimits) -> Self {
        Self {
            limits,
            remaining: limits.validation_work,
        }
    }

    /// One pass through group rows and the original contributions, including
    /// accumulation, bound construction and the row scan.
    pub(in crate::devices) fn charge_sparse(
        &mut self,
        nodes: usize,
        pairs: usize,
        site: CouplingSite<'_>,
    ) -> SpiceResult<()> {
        let work = sparse_work(nodes, pairs)
            .ok_or_else(|| site.resource("mutual validation work count overflows"))?;
        self.charge(work, site)
    }

    fn charge(&mut self, work: usize, site: CouplingSite<'_>) -> SpiceResult<()> {
        self.remaining = self.remaining.checked_sub(work).ok_or_else(|| {
            site.resource(format_args!(
                "mutual validation work limit exceeded (need {work}, {} remaining of {})",
                self.remaining, self.limits.validation_work
            ))
        })?;
        Ok(())
    }

    /// `cells` is one entry per unordered off-diagonal pair, with repeated K
    /// contributions already rounded in their original order.
    ///
    /// An ordinary or weighted Gershgorin bound can prove that every
    /// eigenvalue lies well within the existing acceptance region. Otherwise
    /// use the original dense Jacobi calculation, charged before allocation.
    /// Returns the smallest eigenvalue only when the matrix is indefinite.
    pub(in crate::devices) fn check(
        &mut self,
        nodes: usize,
        pairs: usize,
        cells: impl Iterator<Item = (usize, usize, Real)> + Clone,
        site: CouplingSite<'_>,
    ) -> SpiceResult<Option<Real>> {
        let count = u32::try_from(nodes)
            .map_err(|_| site.resource("mutual group exceeds compact validation range"))?;
        // max |lambda| >= 1 for a unit-diagonal symmetric matrix. Using half
        // the minimum existing tolerance leaves a wide margin for rounding
        // and eigensolver error, even on the exact dominance boundary.
        let minimum_tolerance = 64.0 * Real::from(count) * Real::EPSILON;
        let certificate_limit = 0.5_f64.mul_add(minimum_tolerance, 1.0);
        self.charge_sparse(nodes, pairs, site)?;
        let mut rows = site.reserve(nodes)?;
        rows.resize(nodes, 0.0);
        let radius = row_radius(&mut rows, cells.clone(), None)?;
        if radius <= certificate_limit {
            return Ok(None);
        }

        // Gershgorin after diagonal similarity with positive weights is an
        // equally rigorous but often much tighter certificate (e.g. stars
        // or nonuniform chains). The weights are arbitrary positive numbers;
        // only the outward-rounded bound, not their derivation, matters.
        if radius.is_finite()
            && sparse_work(nodes, pairs).is_some_and(|work| work <= self.remaining)
        {
            self.charge_sparse(nodes, pairs, site)?;
            let mut weights = site.reserve(nodes)?;
            weights.extend(rows.iter().map(|sum| 1.0 + sum));
            if row_radius(&mut rows, cells.clone(), Some(&weights))? <= certificate_limit {
                return Ok(None);
            }
        }

        let (bytes, work) = symmetric_eigenvalue_cost(nodes)
            .ok_or_else(|| site.resource("dense mutual validation cost overflows"))?;
        if bytes > self.limits.dense_bytes {
            return Err(site.resource(format_args!(
                "dense mutual validation for {nodes} inductors requires {bytes} bytes; limit is {}",
                self.limits.dense_bytes
            )));
        }
        let work = work
            .checked_add(pairs)
            .and_then(|work| work.checked_add(nodes))
            .ok_or_else(|| site.resource("dense mutual validation work count overflows"))?;
        self.charge(work, site)?;

        let mut matrix = Matrix::zeros(nodes, nodes)?;
        for row in 0..nodes {
            matrix.set(row, row, 1.0)?;
        }
        for (row, col, value) in cells {
            matrix.set(row, col, value)?;
            matrix.set(col, row, value)?;
        }
        let eigenvalues = matrix.symmetric_eigenvalues()?;
        let largest = eigenvalues.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let smallest = eigenvalues.first().copied().unwrap_or(0.0);
        Ok((smallest < -(minimum_tolerance * largest)).then_some(smallest))
    }
}

fn sparse_work(nodes: usize, pairs: usize) -> Option<usize> {
    nodes.checked_mul(3)?.checked_add(pairs.checked_mul(3)?)
}

/// Round a nonnegative sum upward; zero is an exact no-op.
fn add_upper(sum: Real, value: Real) -> Real {
    if value == 0.0 {
        sum
    } else {
        (sum + value).next_up()
    }
}

fn row_radius(
    rows: &mut [Real],
    cells: impl Iterator<Item = (usize, usize, Real)>,
    weights: Option<&[Real]>,
) -> SpiceResult<Real> {
    rows.fill(0.0);
    for (row, col, value) in cells {
        if !value.is_finite() {
            return Err(SpiceError::Numerical {
                context: "mutual inductive system".to_owned(),
                message: "nonfinite normalized mutual inductance".to_owned(),
            });
        }
        let magnitude = value.abs();
        if magnitude == 0.0 {
            continue;
        }
        let (a, b) = weights.map_or((magnitude, magnitude), |weights| {
            (
                (magnitude * weights[col]).next_up(),
                (magnitude * weights[row]).next_up(),
            )
        });
        rows[row] = add_upper(rows[row], a);
        rows[col] = add_upper(rows[col], b);
    }
    Ok(weights.map_or_else(
        || rows.iter().copied().fold(0.0, Real::max),
        |weights| {
            rows.iter()
                .zip(weights)
                .map(|(sum, weight)| (sum / weight).next_up())
                .fold(0.0, Real::max)
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SITE: CouplingSite<'static> = CouplingSite {
        name: "test",
        location: None,
    };

    fn compare_dense(nodes: usize, cells: &[(usize, usize, Real)]) -> bool {
        let mut matrix = Matrix::zeros(nodes, nodes).unwrap();
        for row in 0..nodes {
            matrix.set(row, row, 1.0).unwrap();
        }
        for &(row, col, value) in cells {
            matrix.set(row, col, value).unwrap();
            matrix.set(col, row, value).unwrap();
        }
        let actual = ValidationBudget::new(MutualLimits::default()).check(
            nodes,
            cells.len(),
            cells.iter().copied(),
            SITE,
        );
        let eigenvalues = match matrix.symmetric_eigenvalues() {
            Ok(values) => values,
            Err(old_error) => {
                // A proof can succeed even when the numerical iteration stalls.
                // Otherwise the fallback must return the original solver error.
                match actual {
                    Ok(None) => {
                        let limits = MutualLimits {
                            dense_bytes: 0,
                            ..MutualLimits::default()
                        };
                        assert_eq!(
                            ValidationBudget::new(limits)
                                .check(nodes, cells.len(), cells.iter().copied(), SITE)
                                .unwrap(),
                            None
                        );
                    }
                    Err(error) => assert_eq!(error.to_string(), old_error.to_string()),
                    Ok(Some(value)) => panic!("dense fallback unexpectedly returned {value}"),
                }
                return false;
            }
        };
        let largest = eigenvalues.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let tolerance = 64.0 * Real::from(u32::try_from(nodes).unwrap()) * Real::EPSILON * largest;
        let expected = eigenvalues[0] < -tolerance;
        let actual = actual.unwrap();
        assert_eq!(actual.is_some(), expected, "nodes={nodes}, cells={cells:?}");
        if let Some(value) = actual {
            assert_eq!(value.to_bits(), eigenvalues[0].to_bits());
        }
        true
    }

    #[test]
    fn sparse_and_dense_classification_agree() {
        let mut state = 0x7135_33b9_u32;
        let mut converged = 0;
        for nodes in 2..=8 {
            for pass in 0..128 {
                let mut cells = Vec::new();
                for row in 0..nodes {
                    for col in row + 1..nodes {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let signed = Real::from(state % 17) - 8.0;
                        let value = if pass % 2 == 0 {
                            signed / 8.0
                        } else {
                            signed / 64.0
                        };
                        if value != 0.0 {
                            cells.push((row, col, value));
                        }
                    }
                }
                converged += usize::from(compare_dense(nodes, &cells));
            }
        }
        eprintln!("old eigensolver converged on {converged}/896 randomized matrices");
        assert!(converged >= 500);
        for magnitude in [1.0, 1.0 + 16.0 * Real::EPSILON, 1.0 + 512.0 * Real::EPSILON] {
            for sign in [-1.0, 1.0] {
                assert!(compare_dense(2, &[(0, 1, sign * magnitude)]));
            }
        }
    }

    #[test]
    fn certificates_keep_large_sparse_systems_out_of_dense_storage() {
        let limits = MutualLimits {
            dense_bytes: 0,
            ..MutualLimits::default()
        };
        let nodes = 10_000;
        let chain = (0..nodes - 1).map(|row| (row, row + 1, 0.5));
        assert_eq!(
            ValidationBudget::new(limits)
                .check(nodes, nodes - 1, chain, SITE)
                .unwrap(),
            None
        );
        // Ordinary Gershgorin fails at the centre, but weighting the centre
        // by its larger degree proves this nine-leaf star positive definite.
        let star = (1..10).map(|leaf| (0, leaf, 0.3));
        assert_eq!(
            ValidationBudget::new(limits)
                .check(10, 9, star, SITE)
                .unwrap(),
            None
        );
        let triangle = [(0, 1, 0.9), (0, 2, 0.9), (1, 2, 0.9)];
        let error = ValidationBudget::new(limits)
            .check(3, 3, triangle.into_iter(), SITE)
            .unwrap_err();
        assert!(error.to_string().contains("dense mutual validation"));
    }

    #[test]
    fn work_budget_covers_all_groups_and_rejects_nonfinite_cells() {
        let limits = MutualLimits {
            validation_work: 18,
            ..MutualLimits::default()
        };
        let mut budget = ValidationBudget::new(limits);
        for _ in 0..2 {
            assert_eq!(
                budget.check(2, 1, [(0, 1, 0.5)].into_iter(), SITE).unwrap(),
                None
            );
        }
        assert!(budget.check(2, 1, [(0, 1, 0.5)].into_iter(), SITE).is_err());
        assert!(
            ValidationBudget::new(MutualLimits::default())
                .check(2, 1, [(0, 1, Real::NAN)].into_iter(), SITE)
                .is_err()
        );
        assert!(sparse_work(usize::MAX, 1).is_none());
        assert!(symmetric_eigenvalue_cost(usize::MAX).is_none());
    }
}
