//! Autonomous single shooting in scaled MNA coordinates.
//!
//! Solve F(x,T)-x=0 with a fixed node coordinate as a phase condition.
//! The remaining state coordinates and log(T) form a square Newton problem.

use crate::maths::{Matrix, Vector};
use crate::primitives::{Real, SpiceError, SpiceResult};

pub(super) fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "periodic steady state".into(),
        message: message.into(),
    }
}

/// One period of a nonconstant autonomous solution.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicOrbit {
    pub(super) solution: ShootingSolution,
    pub(super) topology: crate::devices::circuit::TopologyId,
    pub(super) phase_row: usize,
    pub(super) charge_residual: Real,
    pub(super) accepted_steps: usize,
    pub(super) rejected_steps: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ShootingSolution {
    pub initial: Vector,
    pub period: Real,
    pub residual: Real,
    pub transverse_bound: Real,
    pub map_evaluations: usize,
}

impl PeriodicOrbit {
    /// MNA solution at the chosen phase section.
    #[must_use = "the returned solution supplies the initial MNA state at the selected phase section"]
    pub const fn initial(&self) -> &Vector {
        &self.solution.initial
    }

    /// Fundamental period in seconds.
    #[must_use = "the returned period determines the integration interval for one orbit"]
    pub const fn period(&self) -> Real {
        self.solution.period
    }

    /// Fundamental frequency in hertz.
    #[must_use = "the returned frequency determines the physical spacing of the orbit's spectral harmonics"]
    pub const fn frequency(&self) -> Real {
        self.solution.period.recip()
    }

    /// Maximum scaled state mismatch at the orbit boundary.
    #[must_use = "the returned mismatch is needed to assess periodic state closure"]
    pub const fn residual(&self) -> Real {
        self.solution.residual
    }

    /// Maximum scaled charge/flux mismatch at the orbit boundary.
    #[must_use = "the returned mismatch is needed to assess charge and flux closure separately from voltage closure"]
    pub const fn charge_residual(&self) -> Real {
        self.charge_residual
    }

    /// Estimated transverse contraction bound (excluding phase).
    #[must_use = "the returned bound estimates stability transverse to the neutral phase direction"]
    pub const fn transverse_bound(&self) -> Real {
        self.solution.transverse_bound
    }

    /// Number of one-period map evaluations, including Jacobian probes.
    #[must_use = "the returned count measures orbit-search work including finite-difference probes"]
    pub const fn map_evaluations(&self) -> usize {
        self.solution.map_evaluations
    }

    /// Total accepted adaptive timesteps, including startup and Jacobian probes.
    #[must_use = "the returned count measures accepted integration work across all orbit-search evaluations"]
    pub const fn accepted_steps(&self) -> usize {
        self.accepted_steps
    }

    /// Total rejected adaptive timesteps.
    #[must_use = "the returned count measures rejected adaptive trials during the orbit search"]
    pub const fn rejected_steps(&self) -> usize {
        self.rejected_steps
    }
}

pub(super) struct Shooting {
    pub phase_row: usize,
    pub scales: Vector,
    pub period_guess: Real,
    pub tolerance: Real,
    pub iterations: usize,
}

struct Candidate {
    state: Vector,
    log_period: Real,
}

impl Candidate {
    fn period(&self, guess: Real) -> Real {
        guess * self.log_period.exp()
    }
}

impl Shooting {
    fn residual(
        &self,
        candidate: &Candidate,
        map: &mut impl FnMut(&Vector, Real) -> SpiceResult<Vector>,
    ) -> SpiceResult<Vector> {
        let endpoint = map(&candidate.state, candidate.period(self.period_guess))?;
        if endpoint.len() != self.scales.len() || !endpoint.is_finite() {
            return Err(failure("invalid transient endpoint"));
        }
        Ok(Vector::from_slice(
            &endpoint
                .as_slice()
                .iter()
                .zip(candidate.state.as_slice())
                .zip(self.scales.as_slice())
                .map(|((end, start), scale)| (end - start) / scale)
                .collect::<Vec<_>>(),
        ))
    }

    /// Perturb dimensionless state or log-period coordinates. The phase
    /// coordinate is held fixed, so its column is the period derivative.
    fn displaced(&self, base: &Candidate, direction: &Vector, factor: Real) -> Candidate {
        let mut state = base.state.clone();
        let mut log_period = base.log_period;
        for (row, ((value, &delta), &scale)) in state
            .as_mut_slice()
            .iter_mut()
            .zip(direction.as_slice())
            .zip(self.scales.as_slice())
            .enumerate()
        {
            if row == self.phase_row {
                log_period = factor.mul_add(delta, log_period);
            } else {
                *value = (factor * delta).mul_add(scale, *value);
            }
        }
        Candidate { state, log_period }
    }

    fn jacobian(
        &self,
        candidate: &Candidate,
        map: &mut impl FnMut(&Vector, Real) -> SpiceResult<Vector>,
    ) -> SpiceResult<Matrix> {
        let n = self.scales.len();
        let mut jacobian = Matrix::zeros(n, n)?;
        let h = (self.tolerance.sqrt() * 0.1).max(1e-4);
        let mut basis = Vector::zeros(n);
        for column in 0..n {
            basis.set(column, 1.)?;
            let plus = self.residual(&self.displaced(candidate, &basis, h), map)?;
            let minus = self.residual(&self.displaced(candidate, &basis, -h), map)?;
            basis.set(column, 0.)?;
            for (row, (&a, &b)) in plus.as_slice().iter().zip(minus.as_slice()).enumerate() {
                jacobian.set(row, column, (a - b) / (2. * h))?;
            }
        }
        Ok(jacobian)
    }

    /// Project the monodromy onto the fixed-node Poincare section. The
    /// period derivative is parallel to the flow; its phase component must
    /// be nonzero. An induced norm of a matrix power bounds the spectral
    /// radius without relying on a fragile eigenvalue ranking.
    fn stability(&self, jacobian: &Matrix) -> SpiceResult<Real> {
        let phase = self.phase_row;
        let n = self.scales.len();
        let value = |row, col| jacobian.get(row, col).unwrap_or(Real::NAN);
        let speed = value(phase, phase);
        if !speed.is_finite() || speed.abs() <= self.tolerance.sqrt() {
            return Err(failure(
                "phase section is tangent to the orbit or has no oscillation",
            ));
        }
        let indices: Vec<_> = (0..n).filter(|&i| i != phase).collect();
        let mut transverse = Matrix::zeros(n - 1, n - 1)?;
        for (i, &row) in indices.iter().enumerate() {
            let direction = value(row, phase) / speed;
            for (j, &column) in indices.iter().enumerate() {
                let monodromy = value(row, column) + Real::from(u8::from(row == column));
                transverse.set(i, j, (-direction).mul_add(value(phase, column), monodromy))?;
            }
        }
        contraction_bound(&transverse)
    }

    pub(super) fn solve(
        &self,
        seed: Vector,
        map: &mut impl FnMut(&Vector, Real) -> SpiceResult<Vector>,
    ) -> SpiceResult<ShootingSolution> {
        let mut candidate = Candidate {
            state: seed,
            log_period: 0.,
        };
        let mut evaluations = 0;
        let mut counted_map = |state: &Vector, period: Real| {
            evaluations += 1;
            map(state, period)
        };
        let mut residual = self.residual(&candidate, &mut counted_map)?;
        for _ in 0..self.iterations {
            let norm = residual.max_abs();
            if !norm.is_finite() {
                return Err(failure("nonfinite shooting residual"));
            }
            let jacobian = self.jacobian(&candidate, &mut counted_map)?;
            if norm <= self.tolerance {
                let transverse_bound = self.stability(&jacobian)?;
                return Ok(ShootingSolution {
                    period: candidate.period(self.period_guess),
                    initial: candidate.state,
                    residual: norm,
                    transverse_bound,
                    map_evaluations: evaluations,
                });
            }
            let rhs =
                Vector::from_slice(&residual.as_slice().iter().map(|v| -v).collect::<Vec<_>>());
            let direction = jacobian.solve(&rhs).map_err(|error| {
                failure(format!(
                    "singular shooting Jacobian: {error}; check phase and frequency guess"
                ))
            })?;
            let mut factor = 1.;
            let mut improved = false;
            for _ in 0..12 {
                let next = self.displaced(&candidate, &direction, factor);
                let period = next.period(self.period_guess);
                if next.state.is_finite()
                    && period.is_finite()
                    && (0.125..=8.).contains(&(period / self.period_guess))
                    && let next_residual = self.residual(&next, &mut counted_map)?
                    && next_residual.max_abs() < norm
                {
                    candidate = next;
                    residual = next_residual;
                    improved = true;
                    break;
                }
                factor *= 0.5;
            }
            if !improved {
                return Err(failure(format!(
                    "shooting line search stalled at scaled residual {norm:e}; adjust startup or frequency guess"
                )));
            }
        }
        Err(failure(format!(
            "shooting iteration limit reached at scaled residual {:e}",
            residual.max_abs()
        )))
    }
}

fn contraction_bound(matrix: &Matrix) -> SpiceResult<Real> {
    let n = matrix.rows();
    let mut power = matrix.clone();
    for exponent in 1..=32 {
        let norm = power
            .data()
            .chunks_exact(n)
            .map(|row| row.iter().map(|value| value.abs()).sum::<Real>())
            .fold(0., Real::max);
        if !norm.is_finite() {
            break;
        }
        if norm < 0.98 {
            return Ok(norm.powf(1. / Real::from(exponent)));
        }
        let mut next = Matrix::zeros(n, n)?;
        for column in 0..n {
            let vector = Vector::from_slice(
                &matrix
                    .data()
                    .chunks_exact(n)
                    .map(|row| row[column])
                    .collect::<Vec<_>>(),
            );
            let product = power.mul_vector(&vector)?;
            for (row, &value) in product.as_slice().iter().enumerate() {
                next.set(row, column, value)?;
            }
        }
        power = next;
    }
    Err(failure(
        "unstable or insufficiently resolved transverse limit-cycle contraction",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transverse_test_accepts_nonnormal_stable_and_rejects_unstable_maps() {
        let mut matrix = Matrix::zeros(2, 2).unwrap();
        matrix.set(0, 0, 0.5).unwrap();
        matrix.set(1, 1, 0.5).unwrap();
        matrix.set(0, 1, 4.).unwrap();
        assert!(contraction_bound(&matrix).unwrap() < 1.);
        matrix.set(1, 1, 1.1).unwrap();
        assert!(contraction_bound(&matrix).is_err());
    }
}
