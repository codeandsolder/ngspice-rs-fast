//! Causal delayed modal launches from existing accepted waveform history.
//!
//! The width 2n channels hold near then far launched waves. Trial loads read
//! only committed samples; no history mutations occur in this module.

use super::modal::CplModes;
use super::plan::MAX_CONDUCTORS;
use crate::devices::delay::WaveformHistory;
use crate::primitives::{Real, SpiceError, SpiceResult};

fn error(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "CPL accepted delay history".into(),
        message: message.into(),
    }
}

#[derive(Debug, Clone, Copy)]
struct Bracket {
    earlier: usize,
    later: usize,
    fraction: Real,
}

/// Locate accepted knot samples, with constant prehistory and no extrapolation.
fn bracket(history: &WaveformHistory, target: Real) -> SpiceResult<Bracket> {
    let size = history.len();
    let first = history
        .time(0)
        .ok_or_else(|| error("missing initial CPL time"))?;
    let last = history
        .time(size - 1)
        .ok_or_else(|| error("missing newest CPL time"))?;
    if target <= first {
        return Ok(Bracket {
            earlier: 0,
            later: 0,
            fraction: 0.,
        });
    }
    if target >= last {
        let roundoff = 8. * Real::EPSILON * last.abs().max(target.abs());
        if target - last > roundoff {
            return Err(error(format!(
                "CPL wave at {target:e} exceeds newest accepted time {last:e}; shorten the trial step"
            )));
        }
        return Ok(Bracket {
            earlier: size - 1,
            later: size - 1,
            fraction: 0.,
        });
    }
    let mut earlier = 0;
    let mut later = size - 1;
    while later - earlier > 1 {
        let midpoint = earlier + (later - earlier) / 2;
        let middle = history
            .time(midpoint)
            .ok_or_else(|| error("missing CPL knot"))?;
        if middle <= target {
            earlier = midpoint;
        } else {
            later = midpoint;
        }
    }
    let start = history
        .time(earlier)
        .ok_or_else(|| error("missing earlier CPL time"))?;
    if target.partial_cmp(&start) == Some(std::cmp::Ordering::Equal) {
        return Ok(Bracket {
            earlier,
            later: earlier,
            fraction: 0.,
        });
    }
    let end = history
        .time(later)
        .ok_or_else(|| error("missing later CPL time"))?;
    let fraction = (target - start) / (end - start);
    if !(0. ..=1.).contains(&fraction) {
        return Err(error("invalid CPL interpolation fraction"));
    }
    Ok(Bracket {
        earlier,
        later,
        fraction,
    })
}

fn sample(history: &WaveformHistory, bracket: Bracket, component: usize) -> SpiceResult<Real> {
    let before = history
        .value(bracket.earlier, component)
        .ok_or_else(|| error("missing previous CPL launch"))?;
    let after = history
        .value(bracket.later, component)
        .ok_or_else(|| error("missing later CPL launch"))?;
    let value = crate::devices::functions::affine_value(before, after, bracket.fraction);
    if !value.is_finite() {
        return Err(error("CPL interpolation overflow"));
    }
    Ok(value)
}

impl CplModes {
    /// Causal received waves, near from far and far from near. The first
    /// accepted sample provides a constant initial prehistory.
    ///
    /// # Errors
    /// Missing data, invalid time, wrong widths or a noncausal sample.
    pub fn delayed_launches_into(
        &self,
        history: &WaveformHistory,
        time: Real,
        near_received: &mut [Real],
        far_received: &mut [Real],
    ) -> SpiceResult<()> {
        let delays = self.delays();
        let n = delays.len();
        if !time.is_finite() || time < 0. {
            return Err(error("CPL time must be finite and nonnegative"));
        }
        if history.width() != 2 * n || history.is_empty() || history.time(0) != Some(0.) {
            return Err(error(format!(
                "CPL needs nonempty width {} history at t=0",
                2 * n
            )));
        }
        if near_received.len() != n || far_received.len() != n {
            return Err(error(format!(
                "both CPL receiving buffers must have width {n}"
            )));
        }
        let mut near = [0.; MAX_CONDUCTORS];
        let mut far = [0.; MAX_CONDUCTORS];
        for (mode, &delay) in delays.iter().enumerate() {
            let accepted = bracket(history, time - delay)?;
            near[mode] = sample(history, accepted, n + mode)?;
            far[mode] = sample(history, accepted, mode)?;
        }
        near_received.copy_from_slice(&near[..n]);
        far_received.copy_from_slice(&far[..n]);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::cpl::plan::CplPlan;
    use crate::devices::delay::WaveformUpdate;

    fn modes() -> CplModes {
        CplPlan::prepare(2, 0.5, &[], &[1e-6, 0., 4e-6], &[], &[1e-12, 0., 1e-12])
            .unwrap()
            .modes()
            .clone()
    }

    fn history() -> WaveformHistory {
        let mut history = WaveformHistory::new(4);
        history
            .apply(&WaveformUpdate {
                reset: Some(vec![
                    (0., vec![10., 20., 30., 40.]),
                    (1e-9, vec![12., 24., 36., 48.]),
                    (2e-9, vec![14., 28., 42., 56.]),
                ]),
                ..WaveformUpdate::default()
            })
            .unwrap();
        history
    }

    #[test]
    fn modal_delays_sample_opposite_ports_without_writing_accepted_history() {
        let modes = modes();
        assert!((modes.delays()[0] - 0.5e-9).abs() < 1e-24);
        assert!((modes.delays()[1] - 1e-9).abs() < 1e-24);
        let history = history();
        let before = history.clone();
        let mut near = [0.; 2];
        let mut far = [0.; 2];
        modes
            .delayed_launches_into(&history, 2.25e-9, &mut near, &mut far)
            .unwrap();
        for (actual, expected) in near.into_iter().zip([40.5, 50.]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        for (actual, expected) in far.into_iter().zip([13.5, 25.]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(history, before);
    }

    #[test]
    fn early_preflight_and_exact_knots_keep_initial_and_accepted_values() {
        let modes = modes();
        let history = history();
        let mut near = [0.; 2];
        let mut far = [0.; 2];
        modes
            .delayed_launches_into(&history, 0.25e-9, &mut near, &mut far)
            .unwrap();
        assert_eq!(near, [30., 40.]);
        assert_eq!(far, [10., 20.]);
        modes
            .delayed_launches_into(&history, 1.5e-9, &mut near, &mut far)
            .unwrap();
        assert!((near[0] - 36.).abs() < 1e-12);
        assert!((near[1] - 44.).abs() < 1e-12);
        assert!((far[0] - 12.).abs() < 1e-12);
        assert!((far[1] - 22.).abs() < 1e-12);
    }

    #[test]
    fn causality_and_invalid_updates_leave_outputs_and_history_unchanged() {
        let modes = modes();
        let mut history = history();
        let before = history.clone();
        let mut near = [111.; 2];
        let mut far = [222.; 2];
        assert!(
            modes
                .delayed_launches_into(&history, 10e-9, &mut near, &mut far)
                .is_err()
        );
        assert!(
            history
                .apply(&WaveformUpdate {
                    append: Some((1e-9, vec![0.; 4])),
                    ..WaveformUpdate::default()
                })
                .is_err()
        );
        for time in [f64::NAN, f64::INFINITY, -1.] {
            assert!(
                modes
                    .delayed_launches_into(&history, time, &mut near, &mut far)
                    .is_err()
            );
        }
        assert!(
            modes
                .delayed_launches_into(&WaveformHistory::new(3), 0., &mut near, &mut far)
                .is_err()
        );
        assert_eq!(near, [111.; 2]);
        assert_eq!(far, [222.; 2]);
        assert_eq!(history, before);
    }

    #[test]
    fn opposite_sign_extreme_waves_interpolate_without_overflow() {
        let modes = CplPlan::prepare(1, 0.5, &[], &[1e-6], &[], &[1e-12])
            .unwrap()
            .modes()
            .clone();
        let mut history = WaveformHistory::new(2);
        history
            .apply(&WaveformUpdate {
                reset: Some(vec![
                    (0., vec![f64::MAX, f64::MAX]),
                    (1e-9, vec![-f64::MAX, -f64::MAX]),
                ]),
                ..WaveformUpdate::default()
            })
            .unwrap();
        let mut near = [0.];
        let mut far = [0.];
        modes
            .delayed_launches_into(&history, 1e-9, &mut near, &mut far)
            .unwrap();
        assert!(near[0].is_finite() && near[0].abs() <= 1e294);
        assert!(far[0].is_finite() && far[0].abs() <= 1e294);
    }
}
