//! Research-only, bounded one-port passive rank-one common-pole filter.
//!
//! A real symmetric positive-real admittance is represented by
//! Each positive real relaxation rate has two nonnegative scalar weights,
//! multiplying a pair of positive-real Debye functions and a symmetric
//! positive-semidefinite outer product. Thus the matrix response is
//! positive-real for every Laplace rate in the closed right half plane.
//! everywhere in the right half plane. This is only the LOCAL h1 port
//! kernel; propagation between cable ends remains a separate problem.

pub mod compact;

use std::sync::atomic::{AtomicU64, Ordering};

/// Allocate a non-reused owner identity for a mutable accepted-history
/// object. Even ports with identical model and state must not accept
/// one another's candidate transitions.
pub(crate) fn next_port_id() -> Result<u64, &'static str> {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| "exhausted passive-port identities")
}

#[derive(Debug, Clone)]
pub struct Pole {
    pub rate: f64,
    pub direction: Vec<f64>,
    pub high: f64,
    pub low: f64,
}

#[derive(Debug, Clone)]
struct AcceptedPole {
    spec: Pole,
    filtered_projection: f64,
}

#[derive(Debug)]
pub struct PassivePort {
    owner_id: u64,
    order: usize,
    accepted_voltage: Vec<f64>,
    poles: Vec<AcceptedPole>,
    revision: u64,
}

/// Candidate result of an uncommitted integration step.
#[derive(Debug, Clone)]
pub struct Trial {
    pub current: Vec<f64>,
    pub jacobian: Vec<f64>,
    new_voltage: Vec<f64>,
    filters: Vec<f64>,
    revision: u64,
    owner_id: u64,
}

fn dot(lhs: &[f64], rhs: &[f64]) -> Result<f64, &'static str> {
    if lhs.len() != rhs.len() {
        return Err("projection width mismatch");
    }
    let value = lhs
        .iter()
        .zip(rhs)
        .fold(0., |sum, (&a, &b)| a.mul_add(b, sum));
    if !value.is_finite() {
        return Err("nonfinite modal projection");
    }
    Ok(value)
}

pub(crate) fn integration_weights(rate: f64, dt: f64) -> Result<(f64, f64, f64), &'static str> {
    if !rate.is_finite() || rate <= 0. || !dt.is_finite() || dt <= 0. {
        return Err("invalid passive pole rate or integration time step");
    }
    let h = rate * dt;
    if h.is_infinite() {
        // A valid finite rate and step can overflow when multiplied,
        // but the exact passive limit remains well-defined.
        return Ok((0., 1., 0.));
    }
    let decay = (-h).exp();
    let one_minus_decay = -(-h).exp_m1();
    let alpha = if h < 1e-3 {
        // aG = (h - 1 + exp(-h))/h. The direct formula loses all
        // significant bits for h << sqrt(machine epsilon).
        let polynomial = h.mul_add(
            h.mul_add(
                h.mul_add(h.mul_add(1. / 720., -1. / 120.), 1. / 24.),
                -1. / 6.,
            ),
            0.5,
        );
        h * polynomial
    } else {
        1. - one_minus_decay / h
    };
    let beta = one_minus_decay - alpha;
    if !(0. ..=1.).contains(&alpha) || !(0. ..=1.).contains(&beta) {
        return Err("unstable exponential-integration weights");
    }
    Ok((decay, alpha, beta))
}

impl PassivePort {
    /// Create a ready-to-evaluate port state with its DC equilibrium
    /// history initialized from the accepted initial voltage.
    ///
    /// # Errors
    /// Nonunit modal direction, negative/invalid coefficient, bad width
    /// or nonfinite accepted voltage.
    pub fn new(initial_voltage: Vec<f64>, specs: Vec<Pole>) -> Result<Self, &'static str> {
        let order = initial_voltage.len();
        if order == 0 || initial_voltage.iter().any(|v| !v.is_finite()) {
            return Err("invalid initial passive-port voltage");
        }
        let mut poles = Vec::with_capacity(specs.len());
        for spec in specs {
            if spec.direction.len() != order
                || !spec.rate.is_finite()
                || spec.rate <= 0.
                || !spec.high.is_finite()
                || !spec.low.is_finite()
                || spec.high < 0.
                || spec.low < 0.
            {
                return Err("invalid shared-pole matrix term");
            }
            let norm = dot(&spec.direction, &spec.direction)?;
            if (norm - 1.).abs() > 1e-9 {
                return Err("shared-pole direction must be unit length");
            }
            let filtered_projection = dot(&spec.direction, &initial_voltage)?;
            poles.push(AcceptedPole {
                spec,
                filtered_projection,
            });
        }
        Ok(Self {
            owner_id: next_port_id()?,
            order,
            accepted_voltage: initial_voltage,
            poles,
            revision: 0,
        })
    }

    /// Evaluate the exact convolution of a *linearly interpolated* trial
    /// voltage across the proposed time step. No accepted state is
    /// changed. The resulting Jacobian is PSD for every positive step
    /// because its per-rank-one coefficient is a convex combination
    /// of the two nonnegative passive endpoint weights.
    ///
    /// # Errors
    /// Invalid step, overflow, or dimension mismatch.
    pub fn trial(&self, dt: f64, voltage: &[f64]) -> Result<Trial, &'static str> {
        if voltage.len() != self.order || voltage.iter().any(|v| !v.is_finite()) {
            return Err("invalid trial port voltage");
        }
        let mut current = vec![0.; self.order];
        let mut jacobian = vec![0.; self.order * self.order];
        let mut filters = Vec::with_capacity(self.poles.len());
        for pole in &self.poles {
            let (decay, alpha, beta) = integration_weights(pole.spec.rate, dt)?;
            let projection_old = dot(&pole.spec.direction, &self.accepted_voltage)?;
            let projection_new = dot(&pole.spec.direction, voltage)?;
            let filtered = alpha.mul_add(
                projection_new,
                decay.mul_add(pole.filtered_projection, beta * projection_old),
            );
            if !filtered.is_finite() {
                return Err("nonfinite accepted-mode trial filter");
            }
            filters.push(filtered);
            let contrast = pole.spec.low - pole.spec.high;
            let current_scalar = contrast.mul_add(filtered, pole.spec.high * projection_new);
            let slope = (1. - alpha).mul_add(pole.spec.high, alpha * pole.spec.low);
            if !slope.is_finite() || slope < 0. || !current_scalar.is_finite() {
                return Err("nonfinite/passivity-violating local port tangent");
            }
            for row in 0..self.order {
                let signed = pole.spec.direction[row];
                current[row] = signed.mul_add(current_scalar, current[row]);
                for (col, &direction) in pole.spec.direction.iter().enumerate() {
                    let offset = row * self.order + col;
                    jacobian[offset] = (signed * slope).mul_add(direction, jacobian[offset]);
                }
            }
        }
        if current.iter().chain(&jacobian).any(|v| !v.is_finite()) {
            return Err("nonfinite coupled-port current or tangent");
        }
        Ok(Trial {
            current,
            jacobian,
            new_voltage: voltage.to_vec(),
            filters,
            revision: self.revision,
            owner_id: self.owner_id,
        })
    }

    /// Commit only a successful full-port trial; stale or rejected Newton
    /// iterations cannot change accepted convolution state.
    ///
    /// # Errors
    /// Stale accepted revision or wrong pole count.
    pub fn accept(&mut self, trial: Trial) -> Result<(), &'static str> {
        if self.revision != trial.revision || self.owner_id != trial.owner_id {
            return Err("stale or foreign passive-port acceptance");
        }
        if trial.filters.len() != self.poles.len() {
            return Err("wrong passive-port filter state count");
        }
        if trial.new_voltage.len() != self.order {
            return Err("wrong passive-port voltage width");
        }
        let next_revision = self
            .revision
            .checked_add(1)
            .ok_or("revision counter exhausted")?;
        for (pole, &updated) in self.poles.iter_mut().zip(&trial.filters) {
            pole.filtered_projection = updated;
        }
        self.accepted_voltage = trial.new_voltage;
        self.revision = next_revision;
        Ok(())
    }

    /// Frequency-domain real/imaginary matrix blocks, for independent
    /// numerical regression without integrating a trial.
    ///
    /// # Errors
    /// Negative/nonfinite frequency, or overflowed response.
    pub fn at_omega(&self, omega: f64) -> Result<(Vec<f64>, Vec<f64>), &'static str> {
        if !omega.is_finite() || omega < 0. {
            return Err("invalid local port angular frequency");
        }
        let mut real = vec![0.; self.order * self.order];
        let mut imaginary = vec![0.; self.order * self.order];
        for pole in &self.poles {
            let theta = omega.atan2(pole.spec.rate);
            let (sin, cos) = theta.sin_cos();
            let real_scale = pole.spec.high.mul_add(sin * sin, pole.spec.low * cos * cos);
            let imag_scale = (pole.spec.high - pole.spec.low) * sin * cos;
            for row in 0..self.order {
                for col in 0..self.order {
                    let offset = row * self.order + col;
                    let projection = pole.spec.direction[row] * pole.spec.direction[col];
                    real[offset] = real_scale.mul_add(projection, real[offset]);
                    imaginary[offset] = imag_scale.mul_add(projection, imaginary[offset]);
                }
            }
        }
        if real.iter().chain(&imaginary).any(|v| !v.is_finite()) {
            return Err("nonfinite characteristic matrix response");
        }
        Ok((real, imaginary))
    }

    /// Per-time step accepted state count; each retained filter is one scalar.
    #[must_use]
    pub const fn state_scalars(&self) -> usize {
        self.poles.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(high: f64, low: f64) -> Result<PassivePort, &'static str> {
        PassivePort::new(
            vec![0.],
            vec![Pole {
                rate: 3.,
                direction: vec![1.],
                high,
                low,
            }],
        )
    }

    #[test]
    fn scalar_ramp_has_exact_positive_convex_jacobian() -> Result<(), &'static str> {
        let port = scalar(0.01, 0.1)?;
        for step in [1e-14, 1e-8, 0.01, 0.4, 1e2, 1e250] {
            let trial = port.trial(step, &[1.])?;
            assert!(trial.current[0].is_finite());
            assert!(trial.jacobian[0] >= 0.01 && trial.jacobian[0] <= 0.1);
            assert!((trial.current[0] - trial.jacobian[0]).abs() < 1e-11);
            if step < 1e-8 {
                assert!(trial.current[0] < 0.010_000_01);
            } else if step > 1e2 {
                assert!((trial.current[0] - 0.1).abs() < 1e-3);
            }
        }
        Ok(())
    }

    #[test]
    fn zero_step_rejected_and_old_accepted_state_is_unchanged_by_trials() -> Result<(), &'static str>
    {
        let mut port = scalar(1., 2.)?;
        let first = port.trial(0.5, &[1.])?;
        let rejected = port.trial(0.5, &[10.])?;
        assert!(port.trial(0., &[1.]).is_err());
        assert!(port.trial(0.5, &[f64::NAN]).is_err());
        assert!(port.trial(0.5, &[f64::MAX]).is_err());
        port.accept(first.clone())?;
        let second = port.trial(0.5, &[3.])?;
        assert!(port.accept(rejected).is_err());
        assert!(port.accept(first).is_err());
        port.accept(second)?;
        Ok(())
    }

    #[test]
    fn three_wire_cross_coupling_and_jacobian_passivity() -> Result<(), &'static str> {
        let direction = vec![1. / (3.0_f64).sqrt(); 3];
        let mut specs = vec![];
        for idx in 0..3 {
            let mut axis = vec![0.; 3];
            axis[idx] = 1.;
            specs.push(Pole {
                rate: 1e5,
                direction: axis,
                high: 0.1,
                low: 0.07,
            });
        }
        specs.push(Pole {
            rate: 1e7,
            direction,
            high: 0.3,
            low: 0.01,
        });
        let port = PassivePort::new(vec![0., 0., 0.], specs)?;
        let trial = port.trial(2e-8, &[1., 0., 0.])?;
        assert!(trial.current[1].abs() > 0.);
        for candidate in [[1., 0., 0.], [0., 1., 0.], [1., 1., -2.], [-1., 2., 0.3]] {
            let form = (0..3)
                .map(|i| {
                    (0..3)
                        .map(|j| candidate[i] * trial.jacobian[3 * i + j] * candidate[j])
                        .sum::<f64>()
                })
                .sum::<f64>();
            assert!(form >= 0.);
        }
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(trial.jacobian[i * 3 + j], trial.jacobian[j * 3 + i]);
            }
        }
        Ok(())
    }

    #[test]
    fn dc_and_high_frequency_limits_are_finite_positive_real() -> Result<(), &'static str> {
        let port = scalar(0.01, 0.1)?;
        let (at_dc, imag_dc) = port.at_omega(0.)?;
        assert!((at_dc[0] - 0.1).abs() < 1e-12);
        assert_eq!(imag_dc[0], 0.);
        let (at_high, imag_high) = port.at_omega(1e300)?;
        assert!((at_high[0] - 0.01).abs() < 1e-12);
        assert!(imag_high[0].is_finite());
        assert!(port.at_omega(-1.).is_err());
        Ok(())
    }
    #[test]
    fn identical_models_from_distinct_ports_cannot_exchange_trials() -> Result<(), &'static str> {
        let mut first = scalar(0.01, 0.1)?;
        let mut second = scalar(0.01, 0.1)?;
        let candidate = first.trial(1e-4, &[0.75])?;
        assert!(second.accept(candidate.clone()).is_err());
        first.accept(candidate)?;
        let newer = second.trial(1e-4, &[0.2])?;
        assert!(first.accept(newer.clone()).is_err());
        second.accept(newer)?;
        Ok(())
    }
}
