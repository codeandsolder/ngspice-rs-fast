//! Error control for the first backward-Euler interval of a new integration
//! epoch. Pre-event charge differences cannot estimate the right-hand germ.

use super::{
    AttemptedStep, Boundary, Driver, NextStep, Order, ReadyStep, StepDecision, Tolerances, Trial,
    TrialPoint, failure,
};
use crate::devices::{Forcing, TrialState};
use crate::maths::Vector;
use crate::primitives::{Real, SpiceResult};

pub(super) enum RestartProbe {
    /// No representable interior point, or no dynamic quantities to bound.
    Unneeded,
    Midpoint {
        dt: Real,
        point: TrialPoint,
    },
    Retry {
        next: NextStep,
        last: Vector,
    },
}

/// Both backward-Euler solves start at the same accepted charge. If the
/// right-hand germ is smooth, subtracting the extrapolated midpoint
/// increment from the full increment cancels the leading O(h) term.
/// The residual, scaled for the actual (possibly rounded) midpoint,
/// estimates the full backward-Euler O(h²) local error. No pre-jump
/// derivative is needed, and no provisional midpoint is published.
fn restart_error(initial: Real, middle: Real, final_value: Real, dt: Real, half: Real) -> Real {
    let fraction = half / dt;
    let residual = (-(1. / fraction)).mul_add(middle - initial, final_value - initial);
    residual.abs() / (2. * (1. - fraction))
}

fn charge_ratio(
    values: [Real; 3],
    rates: [Real; 2],
    dt: Real,
    half: Real,
    tolerance: Tolerances,
) -> Real {
    let charge = values
        .into_iter()
        .map(Real::abs)
        .fold(tolerance.chgtol, Real::max);
    let rate = rates.into_iter().map(Real::abs).fold(0., Real::max);
    let bound = (tolerance.reltol * charge)
        .max(dt * tolerance.reltol.mul_add(rate, tolerance.abstol))
        * tolerance.trtol;
    restart_error(values[0], values[1], values[2], dt, half) / bound
}

impl Driver<'_> {
    pub(super) fn restart_probe(
        &mut self,
        ready: ReadyStep,
        iterate: &Vector,
    ) -> SpiceResult<RestartProbe> {
        if self.truncation.iter().all(|device| device.slots.is_empty())
            && self.voltage_history.is_none()
        {
            return Ok(RestartProbe::Unneeded);
        }
        let start = ready.clock.time;
        let end = ready.end_time();
        let at = start.midpoint(end);
        if at <= start || at >= end {
            return Ok(RestartProbe::Unneeded);
        }
        let dt = at - start;
        let mut midpoint = ready;
        midpoint.clock.next = NextStep {
            delta: dt,
            order: Order::One,
        };
        midpoint.boundary = Boundary::Ordinary;
        midpoint.event_landing = None;
        midpoint.probe_landing = Some(at);
        let attempt = midpoint.begin(&self.steps, &self.settings)?;
        self.stats.restart_error_probes += 1;
        let mut prepared = self.prepared.transient_attempt(
            &self.history,
            &self.model_context,
            at,
            &attempt.coefficients,
            Forcing {
                limit: crate::devices::Limit::Right,
                timing: self.timing,
            },
        )?;
        let mut workspace = std::mem::take(&mut self.workspace);
        let outcome = self.solve_trial(&attempt, &mut prepared, iterate, &mut workspace);
        self.workspace = workspace;
        drop(prepared);
        let mut point = match outcome? {
            Trial::Converged(point) => point,
            Trial::NotConverged { last } => {
                return Ok(RestartProbe::Retry {
                    next: NextStep {
                        delta: dt,
                        order: Order::One,
                    },
                    last,
                });
            }
        };
        // The extra sample may discover an event hidden from the original
        // endpoint. Retrying at the midpoint (or its localized crossing)
        // makes that event go through the ordinary transactional path.
        self.stats.device_event_trial_probes += self.event_devices.len();
        let mixed = self.mixed_retry(&attempt, &point.x)?;
        let device = self.device_retry(&attempt, &point.x, &mut point.state)?;
        if let Some(next) = mixed
            .into_iter()
            .chain(device)
            .min_by(|a, b| a.delta.total_cmp(&b.delta))
        {
            return Ok(RestartProbe::Retry {
                next,
                last: point.x,
            });
        }
        if self
            .device_discontinuity(&point.state)?
            .is_some_and(|time| time > start)
        {
            return Ok(RestartProbe::Retry {
                next: NextStep {
                    delta: dt,
                    order: Order::One,
                },
                last: point.x,
            });
        }
        Ok(RestartProbe::Midpoint { dt, point })
    }

    pub(super) fn restart_decision(
        &self,
        attempt: &AttemptedStep,
        state: &TrialState,
        solution: &Vector,
        midpoint: &RestartProbe,
    ) -> SpiceResult<Option<StepDecision>> {
        let RestartProbe::Midpoint { dt: half, point } = midpoint else {
            return Ok(None);
        };
        let dt = attempt.coefficients.dt();
        let accepted = self
            .history
            .accepted(1)
            .ok_or_else(|| failure("missing restart history"))?;
        let tolerance = self.settings.tolerances;
        let mut ratio: Real = 0.;
        for slot in self.truncation.iter().flat_map(|device| &device.slots) {
            let base = slot.base;
            let values = [
                accepted[base],
                point.state.values()[base],
                state.values()[base],
            ];
            let rates = [point.state.values()[base + 1], state.values()[base + 1]];
            let estimate = charge_ratio(values, rates, dt, *half, tolerance);
            if !estimate.is_finite() {
                return Err(failure("nonfinite restart charge error estimate"));
            }
            ratio = ratio.max(estimate);
        }
        if let Some(history) = &self.voltage_history
            && let Some(initial) = history.accepted.front()
        {
            for &row in &history.rows {
                let values = [
                    initial.as_slice()[row],
                    point.x.as_slice()[row],
                    solution.as_slice()[row],
                ];
                let magnitude = values.into_iter().map(Real::abs).fold(0., Real::max);
                let bound = history
                    .options
                    .relative
                    .mul_add(magnitude, history.options.absolute)
                    * history.options.factor;
                let estimate = restart_error(values[0], values[1], values[2], dt, *half) / bound;
                if !estimate.is_finite() {
                    return Err(failure("nonfinite restart voltage error estimate"));
                }
                ratio = ratio.max(estimate);
            }
        }
        if !ratio.is_finite() {
            return Err(failure("nonfinite restart error estimate"));
        }
        // The estimator is an O(h²) local error, not the ordinary
        // divided-difference timestep extrapolation across accepted history.
        let delta = if ratio == 0. {
            2. * dt
        } else {
            dt * (0.9 / ratio.sqrt()).min(2.)
        };
        let next = NextStep {
            delta,
            order: Order::One,
        };
        Ok(Some(if ratio > 1. {
            StepDecision::Retry(next)
        } else {
            StepDecision::Accept(next)
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_germ_residual_is_full_backward_euler_error() {
        // q' = 2 t. BE increment from zero is 2 h², exact is h².
        // Test a deliberately unequal subdivision as well as a midpoint.
        for half in [0.25, 0.5, 0.75] {
            assert_eq!(restart_error(3., 3. + 2. * half * half, 5., 1., half), 1.);
        }
        assert_eq!(restart_error(3., 4., 5., 1., 0.5), 0.);
    }
}
