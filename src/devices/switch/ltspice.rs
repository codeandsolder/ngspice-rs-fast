//! Explicit `LTspice` voltage-controlled switch laws.
//!
//! The source-defined control transfer is separate from ngspice's discrete
//! SW/CSW state machine. This module contains only memoryless branch physics;
//! the outer switch owns any discrete state and sparse row binding.
//! Reference: `LTspice` help, S. Voltage Controlled Switch.

use crate::primitives::{Real, SpiceError, SpiceResult};

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
pub(super) struct Jacobian {
    pub(super) voltage: Real,
    pub(super) control: Real,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Point {
    pub(super) current: Real,
    /// Exact derivative of the physical current law for Newton and transient.
    pub(super) jacobian: Jacobian,
    /// Reference AC/noise load (level-1 limiting uses a chord rather than a tangent).
    pub(super) small_signal: Jacobian,
}

impl Point {
    const fn exact(current: Real, voltage: Real, control: Real) -> Self {
        let jacobian = Jacobian { voltage, control };
        Self {
            current,
            jacobian,
            small_signal: jacobian,
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

    fn controlled(self, p: Conductance, control: Real, closed: bool) -> SpiceResult<(Real, Real)> {
        if self.discrete() {
            return Ok((if closed { p.on } else { p.off }, 0.));
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
        let delta = control - p.threshold;
        let x = delta / p.width;
        let (log_g, log_slope) = match self.transfer {
            Transfer::Cubic => {
                if x <= -1. {
                    return Ok((p.off, 0.));
                }
                if x >= 1. {
                    return Ok((p.on, 0.));
                }
                let fraction = x.mul_add(0.5, 0.5);
                let smooth = fraction * fraction * 2.0f64.mul_add(-fraction, 3.);
                (
                    span.mul_add(smooth, low),
                    span * (3. * fraction * (1. - fraction)) / p.width,
                )
            }
            Transfer::Arctangent => {
                // atan2 retains the correct half-plane even for very large
                // control/width ratios. The derivative uses a scaled norm.
                let angle = delta.atan2(p.width);
                let scale = delta.abs().max(p.width);
                let normalized_width = p.width / scale;
                let normalized_delta = delta / scale;
                let denom =
                    normalized_delta.mul_add(normalized_delta, normalized_width * normalized_width);
                let slope = if delta.is_infinite() {
                    0.
                } else {
                    (normalized_width / denom) / scale
                };
                (
                    (span / std::f64::consts::PI).mul_add(angle, Real::midpoint(low, high)),
                    (span / std::f64::consts::PI) * slope,
                )
            }
            Transfer::Discrete => {
                return Err(SpiceError::circuit("invalid smooth-switch transfer"));
            }
        };
        let g = log_g.exp();
        Ok((g, g * log_slope))
    }

    /// Current and both independent partials at the same constitutive point.
    pub(super) fn evaluate(self, p: Conductance, bias: Bias, closed: bool) -> SpiceResult<Point> {
        let (mut g, mut g_control) = self.controlled(p, bias.control, closed)?;
        let mut voltage = bias.voltage;
        let mut regularization_voltage = voltage;
        if let Direction::Offset(offset) = self.direction {
            voltage += offset;
            regularization_voltage = voltage;
            // Level 1 retains the off resistance in the opposite direction.
            if voltage != 0. && voltage.is_sign_positive() == offset.is_sign_positive() {
                g = p.off;
                g_control = 0.;
            }
        }
        let mut point = match (self.transfer, self.limit) {
            (Transfer::Arctangent, Some(limit)) => {
                if let Direction::OneWay { offset, epsilon } = self.direction {
                    directional_current(g, g_control, limit, voltage, offset, epsilon)
                } else {
                    limited_current(g, g_control, limit, voltage)
                }
            }
            (_, Some(limit)) if (g * voltage).abs() > limit => {
                // LTspice level 1 uses the limited current/voltage chord
                // in AC and noise, although the physical tangent is zero.
                let mut point = Point::exact(limit.copysign(voltage), 0., 0.);
                point.small_signal.voltage = limit / voltage.abs();
                point
            }
            _ => Point::exact(g * voltage, g, g_control * voltage),
        };
        // Independent AC sweeps show a constant 0.99 factor on the
        // cubic control derivative. Preserve the exact physical Newton
        // derivative; only the reference small-signal load is scaled.
        if self.transfer == Transfer::Cubic {
            point.small_signal.control *= 0.99;
        }
        // Level 1 applies gmin to the shifted voltage; level 2 applies it to
        // the physical branch voltage, even when reverse conduction is blocked.
        // Independent noise measurements show that both levels include gmin
        // in the thermal generator.
        point.current = p.gmin.mul_add(regularization_voltage, point.current);
        point.jacobian.voltage += p.gmin;
        point.small_signal.voltage += p.gmin;
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
}

/// Stable derivative of tanh, including the nearly saturated region where
/// computing 1 - tanh(x)^2 would lose the small remaining conductance.
fn sech_squared(argument: Real) -> Real {
    let tail = (-2. * argument.abs()).exp();
    4. * tail / (1. + tail).powi(2)
}

fn limited_current(g: Real, g_control: Real, limit: Real, voltage: Real) -> Point {
    let linear = g * voltage;
    let argument = linear / limit;
    let sech2 = sech_squared(argument);
    let current = if argument.abs() < 1e-8 {
        linear
    } else {
        limit * argument.tanh()
    };
    let control_slope = if sech2 == 0. {
        0.
    } else {
        (g_control * sech2) * voltage
    };
    Point::exact(current, g * sech2, control_slope)
}

// Solve sinh(2*j)=a with the measured fixed iteration count. Stable tanh
// and sech avoid overflow in the original hyperbolic expression. The second
// result is d j / d ln(a), not the derivative of the converged root.
fn epsilon_join(a: Real) -> (Real, Real, Real) {
    let mut join = 0.5 * a;
    let mut derivative = join;
    if a.is_infinite() {
        return (a, 0., 0.);
    }
    for _ in 0..10 {
        let doubled = 2. * join;
        let tail = (-doubled).exp();
        let sech = 2. * tail / tail.mul_add(tail, 1.);
        let tanh = doubled.tanh();
        let scaled = a * sech;
        derivative = (tanh * (tanh - scaled)).mul_add(derivative, 0.5 * scaled);
        join = 0.5_f64.mul_add(scaled - tanh, join);
    }
    // The reference AC law differentiates the implicit root equation at
    // the final approximate join, not the ten actual iterations.
    let tail = (-2. * join).exp();
    let implicit = a * tail / tail.mul_add(tail, 1.);
    (join, derivative, implicit)
}

/// The one-way branch conducts from the second terminal to the first.
/// At epsilon, a quadratic joins the shifted tanh with equal value and
/// slope in the converged limit. `LTspice` 24.0.12 uses ten Newton steps
/// from a/2, not the exact asinh(a)/2 root; its high-a transfer differs
/// substantially. Propagate the derivative of those same iterations.
fn directional_current(
    g: Real,
    g_control: Real,
    limit: Real,
    voltage: Real,
    offset: Real,
    epsilon: Real,
) -> Point {
    let forward = -(voltage + offset);
    if forward <= 0. {
        return Point::exact(0., 0., 0.);
    }
    if epsilon == 0. {
        return limited_current(g, g_control, limit, -forward);
    }
    let a = g * epsilon / limit;
    let (join, join_log_derivative, implicit_derivative) = epsilon_join(a);
    let log_slope = g_control / g;
    // Compare in terminal-voltage coordinates to select the tanh branch
    // at the exact shifted join, even if subtracting the offset rounds.
    let boundary_voltage = -offset - epsilon;
    if voltage > boundary_voltage {
        let fraction = forward / epsilon;
        let boundary = limit * join.tanh();
        let current = -boundary * fraction * fraction;
        let voltage_slope = 2. * (boundary / epsilon) * fraction;
        let sech2 = sech_squared(join);
        let derivative = |join_derivative: Real| {
            if sech2 == 0. {
                0.
            } else {
                -limit * sech2 * join_derivative * log_slope * fraction * fraction
            }
        };
        let mut point = Point::exact(current, voltage_slope, derivative(join_log_derivative));
        point.small_signal.control = derivative(implicit_derivative);
        return point;
    }
    let overdrive = g * (boundary_voltage - voltage) / limit;
    let argument = overdrive + join;
    let sech2 = sech_squared(argument);
    let derivative = |join_derivative: Real| {
        if sech2 == 0. {
            0.
        } else {
            -limit * sech2 * (overdrive + join_derivative) * log_slope
        }
    };
    let mut point = Point::exact(
        -limit * argument.tanh(),
        g * sech2,
        derivative(join_log_derivative),
    );
    point.small_signal.control = derivative(implicit_derivative);
    point
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
