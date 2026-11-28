//! Explicit `LTspice` voltage-controlled switch laws.
//!
//! The source-defined control transfer is separate from ngspice's discrete
//! SW/CSW state machine. This module contains only memoryless branch physics;
//! the outer switch owns any discrete state and sparse row binding.
//! Reference: `LTspice` help, S. Voltage Controlled Switch.

mod germ;

use self::germ::{Germ, angle, sech_squared, tanh};
use crate::devices::physical_dae::LocalGuard;
use crate::maths::taylor::Jet;
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Transfer {
    Discrete,
    Cubic,
    Arctangent,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Direction {
    Bidirectional,
    Offset(Real),
    OneWay { offset: Real, epsilon: Real },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Law {
    pub(super) transfer: Transfer,
    pub(super) limit: Option<Real>,
    pub(super) direction: Direction,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Jacobian<T = Real> {
    pub(super) voltage: T,
    pub(super) control: T,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Point<T = Real> {
    pub(super) current: T,
    /// Exact derivative of the physical current law for Newton and transient.
    pub(super) jacobian: Jacobian<T>,
    /// Reference AC/noise load (level-1 limiting uses a chord rather than a tangent).
    pub(super) small_signal: Jacobian<T>,
}

impl<T: Clone> Point<T> {
    fn exact(current: T, voltage: T, control: T) -> Self {
        let jacobian = Jacobian { voltage, control };
        Self {
            current,
            small_signal: jacobian.clone(),
            jacobian,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Bias {
    pub(super) voltage: Real,
    pub(super) control: Real,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Conductance {
    pub(super) on: Real,
    pub(super) off: Real,
    pub(super) threshold: Real,
    pub(super) width: Real,
    pub(super) gmin: Real,
}

impl Law {
    pub(super) const fn new(
        level: u8,
        hysteresis: Real,
        limit: Option<Real>,
        direction: Direction,
    ) -> Self {
        let transfer = if level == 2 {
            Transfer::Arctangent
        } else if hysteresis < 0. {
            Transfer::Cubic
        } else {
            Transfer::Discrete
        };
        let limit = match (transfer, limit) {
            (Transfer::Arctangent, Some(value)) if value > 0. => Some(value),
            (Transfer::Arctangent, _) => Some(10.),
            (_, Some(value)) if value != 0. => Some(value.abs()),
            _ => None,
        };
        Self {
            transfer,
            limit,
            direction,
        }
    }

    pub(super) const fn discrete(self) -> bool {
        matches!(self.transfer, Transfer::Discrete)
    }

    fn controlled<T: Germ>(
        self,
        p: Conductance,
        control: T,
        closed: bool,
        guards: &mut Option<Vec<LocalGuard>>,
    ) -> SpiceResult<(T, T)> {
        if self.discrete() {
            return Ok((
                control.constant(if closed { p.on } else { p.off }),
                control.constant(0.),
            ));
        }
        if p.off <= 0. || p.on <= 0. || p.width <= 0. {
            return Err(SpiceError::circuit(
                "continuous switch requires positive finite on/off conductance and transition width",
            ));
        }
        // Difference of logarithms avoids the overflow of Gon/Goff.
        let low = p.off.ln();
        let high = p.on.ln();
        let span = high - low;
        let delta = control.offset(-p.threshold);
        let (log_g, log_slope) = match self.transfer {
            Transfer::Cubic => {
                let x = delta.clone().divide(&delta.constant(p.width))?;
                x.guard(0, &[-1., 1.], guards);
                if x.compare(-1.) != Ordering::Greater {
                    return Ok((x.constant(p.off), x.constant(0.)));
                }
                if x.compare(1.) != Ordering::Less {
                    return Ok((x.constant(p.on), x.constant(0.)));
                }
                let fraction = x
                    .clone()
                    .scale(0.5)
                    .offset(0.5)
                    .with_value(x.value().mul_add(0.5, 0.5));
                let smooth = fraction
                    .clone()
                    .multiply(&fraction)
                    .multiply(&fraction.clone().scale(-2.).offset(3.));
                let slope = fraction
                    .clone()
                    .multiply(&fraction.clone().scale(-1.).offset(1.))
                    .scale(3. * span)
                    .divide(&fraction.constant(p.width))?;
                let log = smooth
                    .clone()
                    .scale(span)
                    .offset(low)
                    .with_value(span.mul_add(smooth.value(), low));
                (log, slope)
            }
            Transfer::Arctangent => {
                // atan2 retains the correct half-plane even for very large
                // control/width ratios. The derivative uses a scaled norm.
                let (angle, slope) = angle(delta, p.width)?;
                let coefficient = span / std::f64::consts::PI;
                let midpoint = Real::midpoint(low, high);
                let log = angle
                    .clone()
                    .scale(coefficient)
                    .offset(midpoint)
                    .with_value(coefficient.mul_add(angle.value(), midpoint));
                (log, slope.scale(coefficient))
            }
            Transfer::Discrete => {
                return Err(SpiceError::circuit("invalid smooth-switch transfer"));
            }
        };
        let g = log_g.exp();
        Ok((g.clone(), g.multiply(&log_slope)))
    }

    /// Current and both independent partials at the same constitutive point.
    pub(super) fn evaluate(self, p: Conductance, bias: Bias, closed: bool) -> SpiceResult<Point> {
        let point = self.evaluate_germ(p, bias.voltage, bias.control, closed, &mut None)?;
        if [
            point.current,
            point.jacobian.voltage,
            point.jacobian.control,
            point.small_signal.voltage,
            point.small_signal.control,
        ]
        .into_iter()
        .all(Real::is_finite)
        {
            Ok(point)
        } else {
            Err(SpiceError::circuit(
                "nonfinite LTspice switch constitutive point",
            ))
        }
    }

    /// The physical Taylor load differentiates the actual iteration law,
    /// not the reference AC chord or implicit epsilon-join tangent.
    pub(super) fn taylor(
        self,
        p: Conductance,
        voltage: Jet,
        control: Jet,
        guards: &mut Option<Vec<LocalGuard>>,
    ) -> SpiceResult<Point<Jet>> {
        self.evaluate_germ(p, voltage, control, false, guards)
    }

    fn evaluate_germ<T: Germ>(
        self,
        p: Conductance,
        mut voltage: T,
        control: T,
        closed: bool,
        guards: &mut Option<Vec<LocalGuard>>,
    ) -> SpiceResult<Point<T>> {
        let (mut g, mut g_control) = self.controlled(p, control, closed, guards)?;
        if let Direction::Offset(offset) = self.direction {
            voltage = voltage.offset(offset);
            voltage.guard(1, &[0.], guards);
            // Level 1 retains the off resistance in the opposite direction.
            let sign = voltage.compare(0.);
            if sign != Ordering::Equal && (sign == Ordering::Greater) == offset.is_sign_positive() {
                g = g.constant(p.off);
                g_control = g.constant(0.);
            }
        }
        let mut point = match (self.transfer, self.limit) {
            (Transfer::Arctangent, Some(limit)) => {
                if let Direction::OneWay { offset, epsilon } = self.direction {
                    directional_current(
                        g,
                        g_control,
                        limit,
                        voltage.clone(),
                        offset,
                        epsilon,
                        guards,
                    )?
                } else {
                    limited_current(g, g_control, limit, &voltage)?
                }
            }
            (_, Some(limit)) => {
                let linear = g.clone().multiply(&voltage);
                linear.guard(2, &[-limit, limit], guards);
                if linear.compare(limit) == Ordering::Greater
                    || linear.compare(-limit) == Ordering::Less
                {
                    // Level 1 uses a chord in AC and noise, not the physical tangent.
                    let current = voltage.constant(limit.copysign(voltage.value()));
                    let zero = voltage.constant(0.);
                    let mut point = Point::exact(current, zero.clone(), zero);
                    point.small_signal.voltage =
                        voltage.constant(limit).divide(&voltage.clone().abs())?;
                    point
                } else {
                    Point::exact(linear, g, g_control.multiply(&voltage))
                }
            }
            _ => Point::exact(
                g.clone().multiply(&voltage),
                g,
                g_control.multiply(&voltage),
            ),
        };
        if self.transfer == Transfer::Cubic {
            point.small_signal.control = point.small_signal.control.scale(0.99);
        }
        // Level 1 applies gmin to the shifted voltage; level 2 to the
        // physical branch voltage, even when reverse conduction is blocked.
        point.current = voltage.constant(p.gmin).mul_add(&voltage, &point.current);
        point.jacobian.voltage = point.jacobian.voltage.offset(p.gmin);
        point.small_signal.voltage = point.small_signal.voltage.offset(p.gmin);
        Ok(point)
    }
}

fn limited_current<T: Germ>(g: T, g_control: T, limit: Real, voltage: &T) -> SpiceResult<Point<T>> {
    let linear = g.clone().multiply(voltage);
    let argument = linear.clone().divide(&linear.constant(limit))?;
    let sech2 = sech_squared(&argument)?;
    // The tiny-argument scalar evaluation is a roundoff shortcut, not a
    // different physical tangent. Retain the analytic tanh germ.
    let mut current = tanh(argument.clone())?.scale(limit);
    if argument.value().abs() < 1e-8 {
        current = current.with_value(linear.value());
    }
    let control_slope = if sech2.value() == 0. {
        voltage.constant(0.)
    } else {
        g_control.multiply(&sech2).multiply(voltage)
    };
    Ok(Point::exact(current, g.multiply(&sech2), control_slope))
}

// Solve sinh(2*j)=a with the measured fixed iteration count. Stable tanh
// and sech avoid overflow in the original hyperbolic expression. The second
// result is d j / d ln(a), not the derivative of the converged root.
fn epsilon_join<T: Germ>(a: T) -> SpiceResult<(T, T, T)> {
    let mut join = a.clone().scale(0.5);
    let mut derivative = join.clone();
    if a.value().is_infinite() {
        let zero = a.constant(0.);
        return Ok((a, zero.clone(), zero));
    }
    for _ in 0..10 {
        let doubled = join.clone().scale(2.);
        let tail = doubled.clone().scale(-1.).exp();
        let denominator = tail.clone().mul_add(&tail, &tail.constant(1.));
        let sech = tail.scale(2.).divide(&denominator)?;
        let tanh = tanh(doubled)?;
        let scaled = a.clone().multiply(&sech);
        derivative = tanh
            .clone()
            .multiply(&tanh.clone().add(&scaled.clone().scale(-1.)))
            .mul_add(&derivative, &scaled.clone().scale(0.5));
        join = join
            .constant(0.5)
            .mul_add(&scaled.add(&tanh.scale(-1.)), &join);
    }
    // Reference AC differentiates the implicit equation at the final
    // approximate join, not the ten actual iterations.
    let tail = join.clone().scale(-2.).exp();
    let denominator = tail.clone().mul_add(&tail, &tail.constant(1.));
    let implicit = a.multiply(&tail).divide(&denominator)?;
    Ok((join, derivative, implicit))
}

/// The one-way branch conducts from the second terminal to the first.
/// At epsilon a quadratic joins the shifted tanh; the vendor uses exactly
/// ten Newton steps from a/2, not the converged asinh(a)/2 root.
fn directional_current<T: Germ>(
    g: T,
    g_control: T,
    limit: Real,
    voltage: T,
    offset: Real,
    epsilon: Real,
    guards: &mut Option<Vec<LocalGuard>>,
) -> SpiceResult<Point<T>> {
    let forward = voltage.clone().offset(offset).scale(-1.);
    forward.guard(3, &[0.], guards);
    if forward.compare(0.) != Ordering::Greater {
        let zero = forward.constant(0.);
        return Ok(Point::exact(zero.clone(), zero.clone(), zero));
    }
    if epsilon == 0. {
        return limited_current(g, g_control, limit, &forward.scale(-1.));
    }
    let a = g.clone().scale(epsilon).divide(&g.constant(limit))?;
    let (join, join_log_derivative, implicit_derivative) = epsilon_join(a)?;
    let log_slope = g_control.divide(&g)?;
    // Compare in terminal-voltage coordinates at the exact shifted join.
    let boundary_voltage = -offset - epsilon;
    voltage.guard(4, &[boundary_voltage], guards);
    if voltage.compare(boundary_voltage) == Ordering::Greater {
        let fraction = forward.clone().divide(&forward.constant(epsilon))?;
        let boundary = tanh(join.clone())?.scale(limit);
        let current = boundary
            .clone()
            .scale(-1.)
            .multiply(&fraction)
            .multiply(&fraction);
        let voltage_slope = boundary
            .scale(2.)
            .divide(&fraction.constant(epsilon))?
            .multiply(&fraction);
        let sech2 = sech_squared(&join)?;
        let derivative = |join_derivative: T| {
            if sech2.value() == 0. {
                fraction.constant(0.)
            } else {
                sech2
                    .clone()
                    .scale(-limit)
                    .multiply(&join_derivative)
                    .multiply(&log_slope)
                    .multiply(&fraction)
                    .multiply(&fraction)
            }
        };
        let mut point = Point::exact(current, voltage_slope, derivative(join_log_derivative));
        point.small_signal.control = derivative(implicit_derivative);
        return Ok(point);
    }
    let overdrive = g
        .clone()
        .multiply(&voltage.scale(-1.).offset(boundary_voltage))
        .divide(&g.constant(limit))?;
    let argument = overdrive.clone().add(&join);
    let sech2 = sech_squared(&argument)?;
    let derivative = |join_derivative: T| {
        if sech2.value() == 0. {
            sech2.constant(0.)
        } else {
            sech2
                .clone()
                .scale(-limit)
                .multiply(&overdrive.clone().add(&join_derivative))
                .multiply(&log_slope)
        }
    };
    let mut point = Point::exact(
        tanh(argument)?.scale(-limit),
        g.multiply(&sech2),
        derivative(join_log_derivative),
    );
    point.small_signal.control = derivative(implicit_derivative);
    Ok(point)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(level: u8, direction: Direction) -> Law {
        Law::new(level, -1., Some(0.05), direction)
    }

    fn check_rows(data: &str, laws: &[Law]) -> SpiceResult<()> {
        let conductance = Conductance {
            on: 10.,
            off: 0.01,
            threshold: 0.,
            width: 1.,
            gmin: 1e-9,
        };
        for (index, line) in data
            .lines()
            .filter(|line| !line.starts_with('#'))
            .enumerate()
        {
            let values: Vec<Real> = line
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<_, _>>()
                .map_err(|_| SpiceError::circuit("invalid numeric fixture"))?;
            assert_eq!(values.len(), laws.len() + 2);
            let bias = Bias {
                voltage: values[0],
                control: values[1],
            };
            for (i, law) in laws.iter().enumerate() {
                let point = law.evaluate(conductance, bias, false)?;
                let expected = values[i + 2];
                assert!(
                    (point.current - expected).abs() < 5e-9_f64.mul_add(expected.abs(), 3e-13),
                    "row {index}, law {law:?}: {:?} != {expected}",
                    point.current
                );
            }
        }
        Ok(())
    }

    #[test]
    fn cubic_ac_control_derivative_matches_measured_grid() -> SpiceResult<()> {
        let law = Law::new(1, -1., None, Direction::Bidirectional);
        for line in include_str!("../../../tests/fixtures/ltspice-switch/cubic-ac-grid.tsv")
            .lines()
            .filter(|line| !line.starts_with('#'))
        {
            let values = line
                .split_whitespace()
                .map(str::parse::<Real>)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| SpiceError::circuit("invalid AC fixture"))?;
            let [roff, width, normalized, expected]: [Real; 4] = values
                .try_into()
                .map_err(|_| SpiceError::circuit("invalid AC fixture width"))?;
            let p = Conductance {
                on: 1.,
                off: roff.recip(),
                threshold: 0.,
                width,
                gmin: 1e-9,
            };
            let point = law.evaluate(
                p,
                Bias {
                    voltage: -0.3,
                    control: normalized * width,
                },
                false,
            )?;
            assert!(
                (-point.small_signal.control - expected).abs()
                    < 1e-10_f64.mul_add(expected.abs(), 1e-14)
            );
        }
        Ok(())
    }

    #[test]
    fn measured_limits_offsets_and_directional_smoothing() -> SpiceResult<()> {
        check_rows(
            include_str!("../../../tests/fixtures/ltspice-switch/directional.tsv"),
            &[
                model(1, Direction::Bidirectional),
                model(1, Direction::Offset(0.5)),
                model(2, Direction::Bidirectional),
                model(
                    2,
                    Direction::OneWay {
                        offset: 0.5,
                        epsilon: 0.,
                    },
                ),
                model(
                    2,
                    Direction::OneWay {
                        offset: 0.,
                        epsilon: 0.,
                    },
                ),
                model(
                    2,
                    Direction::OneWay {
                        offset: 0.5,
                        epsilon: 0.2,
                    },
                ),
                model(
                    2,
                    Direction::OneWay {
                        offset: 0.,
                        epsilon: 0.2,
                    },
                ),
            ],
        )?;
        check_rows(
            include_str!("../../../tests/fixtures/ltspice-switch/offset.tsv"),
            &[
                model(1, Direction::Offset(-0.5)),
                model(1, Direction::Bidirectional),
                model(1, Direction::Offset(0.5)),
                model(2, Direction::Bidirectional),
                model(
                    2,
                    Direction::OneWay {
                        offset: -0.5,
                        epsilon: 0.,
                    },
                ),
                model(
                    2,
                    Direction::OneWay {
                        offset: 0.,
                        epsilon: 0.2,
                    },
                ),
            ],
        )
    }

    #[test]
    fn measured_epsilon_join_at_twenty_five_conductances() -> SpiceResult<()> {
        let law = model(
            2,
            Direction::OneWay {
                offset: 0.,
                epsilon: 0.2,
            },
        );
        let data = include_str!("../../../tests/fixtures/ltspice-switch/epsilon-join.tsv");
        for line in data.lines().filter(|line| !line.starts_with('#')) {
            let values: Vec<Real> = line
                .split_whitespace()
                .map(str::parse)
                .collect::<Result<_, _>>()
                .map_err(|_| SpiceError::circuit("invalid numeric fixture"))?;
            assert_eq!(values.len(), 5);
            let p = Conductance {
                on: values[0],
                off: values[0],
                threshold: 0.,
                width: 1.,
                gmin: 1e-12,
            };
            for (voltage, expected) in [-0.1, -0.2, -0.21, -0.3].into_iter().zip(&values[1..]) {
                let point = law.evaluate(
                    p,
                    Bias {
                        voltage,
                        control: 0.,
                    },
                    false,
                )?;
                assert!(
                    (point.current - expected).abs() < 5e-9_f64.mul_add(expected.abs(), 3e-13),
                    "conductance {}, voltage {voltage}: {} != {expected}",
                    p.on,
                    point.current
                );
            }
        }
        Ok(())
    }

    #[test]
    fn measured_boundary_and_signed_parameter_semantics() -> SpiceResult<()> {
        let p = Conductance {
            on: 10.,
            off: 0.01,
            threshold: 0.,
            width: 1.,
            gmin: 1e-9,
        };
        for offset in [-0.5, 0.5] {
            let law = Law::new(1, -1., None, Direction::Offset(offset));
            let point = law.evaluate(
                p,
                Bias {
                    voltage: -offset,
                    control: 2.,
                },
                false,
            )?;
            assert!((point.small_signal.voltage - (10. + p.gmin)).abs() < 1e-12);
        }
        let law = model(
            2,
            Direction::OneWay {
                offset: 0.5,
                epsilon: 0.2,
            },
        );
        let point = law.evaluate(
            p,
            Bias {
                voltage: -0.7,
                control: 2.,
            },
            false,
        )?;
        assert!((point.small_signal.voltage - 0.139_280_320_957_840_9).abs() < 1e-9);
        assert_eq!(
            Law::new(1, -1., Some(0.), Direction::Bidirectional).limit,
            None
        );
        assert_eq!(
            Law::new(1, -1., Some(-0.05), Direction::Bidirectional).limit,
            Some(0.05)
        );
        assert_eq!(
            Law::new(2, -1., Some(-0.05), Direction::Bidirectional).limit,
            Some(10.)
        );
        assert_eq!(
            Law::new(2, -1., Some(0.), Direction::Bidirectional).limit,
            Some(10.)
        );
        Ok(())
    }

    #[test]
    fn directional_jacobian_matches_finite_differences() -> SpiceResult<()> {
        let p = Conductance {
            on: 10.,
            off: 0.01,
            threshold: 0.,
            width: 1.,
            gmin: 1e-9,
        };
        let step = 1e-6;
        for direction in [
            Direction::Offset(-0.5),
            Direction::Offset(0.5),
            Direction::OneWay {
                offset: 0.5,
                epsilon: 0.,
            },
            Direction::OneWay {
                offset: 0.5,
                epsilon: 0.2,
            },
        ] {
            let level = if matches!(direction, Direction::Offset(_)) {
                1
            } else {
                2
            };
            let law = model(level, direction);
            for control in [-2., -0.3, 0.2, 2.] {
                for voltage in [
                    -1., -0.9, -0.71, -0.69, -0.61, -0.51, -0.49, -0.1, 0.1, 0.49, 0.51, 0.9,
                ] {
                    let bias = Bias { voltage, control };
                    let point = law.evaluate(p, bias, false)?;
                    let i = |voltage, control| {
                        law.evaluate(p, Bias { voltage, control }, false)
                            .map(|p| p.current)
                    };
                    let dv =
                        (i(voltage + step, control)? - i(voltage - step, control)?) / (2. * step);
                    let dc =
                        (i(voltage, control + step)? - i(voltage, control - step)?) / (2. * step);
                    assert!(
                        (dv - point.jacobian.voltage).abs() < 1e-7_f64.mul_add(dv.abs(), 1e-9),
                        "{law:?} {bias:?} {dv} {point:?}"
                    );
                    assert!(
                        (dc - point.jacobian.control).abs() < 1e-7_f64.mul_add(dc.abs(), 1e-9),
                        "{law:?} {bias:?} {dc} {point:?}"
                    );
                }
            }
        }
        Ok(())
    }
}
