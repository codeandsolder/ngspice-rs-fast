//! Explicit `LTspice` voltage-controlled switch laws.
//!
//! The source-defined control transfer is separate from ngspice's discrete
//! SW/CSW state machine. This module contains only memoryless branch physics;
//! the outer switch owns any discrete state and sparse row binding.
//! Reference: `LTspice` help, S. Voltage Controlled Switch.

use crate::devices::physical_dae::LocalGuard;
use crate::maths::taylor::{
    constitutive::{Constitutive, Dual, sech_squared},
    differential::Differential,
};
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

// Reference AC exceptions are metadata from the same physical branch,
// not a second implementation of its current law.
enum Reference<T> {
    Physical,
    Chord(Real),
    Join { factor: Real, conductance: T },
}

struct Evaluation<T> {
    current: T,
    reference: Reference<T>,
}

impl<T> Evaluation<T> {
    const fn physical(current: T) -> Self {
        Self {
            current,
            reference: Reference::Physical,
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

    fn controlled<T: Constitutive>(
        self,
        p: Conductance,
        control: T,
        closed: bool,
        guard: &mut impl FnMut(usize, &T, Real),
    ) -> SpiceResult<T> {
        if self.discrete() {
            return Ok(control.constant_like(if closed { p.on } else { p.off }));
        }
        if p.off <= 0. || p.on <= 0. || p.width <= 0. {
            return Err(SpiceError::circuit(
                "continuous switch requires positive finite on/off conductance and transition width",
            ));
        }
        let low = p.off.ln();
        let high = p.on.ln();
        let span = high - low;
        let delta = control.offset(-p.threshold);
        let log_g = match self.transfer {
            Transfer::Cubic => {
                let x = delta.divide_constant(p.width);
                guard(0, &x, -1.);
                guard(1, &x, 1.);
                if x.compare(-1.) != Ordering::Greater {
                    return Ok(x.constant_like(p.off));
                }
                if x.compare(1.) != Ordering::Less {
                    return Ok(x.constant_like(p.on));
                }
                let fraction = x.affine(0.5, 0.5);
                let smooth = fraction
                    .clone()
                    .square()
                    .multiply(&fraction.affine(-2., 3.));
                smooth.affine(span, low)
            }
            Transfer::Arctangent => delta
                .atan_scaled(p.width)?
                .affine(span / std::f64::consts::PI, Real::midpoint(low, high)),
            Transfer::Discrete => {
                return Err(SpiceError::circuit("invalid smooth-switch transfer"));
            }
        };
        Ok(log_g.exp())
    }

    /// Current and both independent partials at the same constitutive point.
    pub(super) fn evaluate(self, p: Conductance, bias: Bias, closed: bool) -> SpiceResult<Point> {
        let voltage = Dual::<2>::variable(0, bias.voltage)?;
        let control = Dual::<2>::variable(1, bias.control)?;
        let result = self.physical(p, voltage, control, closed, |_, _, _| {})?;
        let current = result.current;
        let jacobian = Jacobian {
            voltage: current.gradient[0],
            control: current.gradient[1],
        };
        let mut small_signal = jacobian;
        match result.reference {
            Reference::Physical => {}
            Reference::Chord(slope) => small_signal.voltage = slope + p.gmin,
            Reference::Join {
                factor,
                conductance,
            } => {
                small_signal.control = if factor == 0. {
                    0.
                } else {
                    factor * (conductance.gradient[1] / conductance.value)
                };
            }
        }
        if self.transfer == Transfer::Cubic {
            small_signal.control *= 0.99;
        }
        if ![small_signal.voltage, small_signal.control]
            .into_iter()
            .all(Real::is_finite)
        {
            return Err(SpiceError::circuit(
                "nonfinite LTspice switch small-signal point",
            ));
        }
        Ok(Point {
            current: current.value,
            jacobian,
            small_signal,
        })
    }

    pub(super) fn taylor(
        self,
        p: Conductance,
        voltage: Differential<2>,
        control: Differential<2>,
        guards: &mut Vec<LocalGuard>,
    ) -> SpiceResult<Differential<2>> {
        if self.discrete() {
            return Err(SpiceError::circuit(
                "discrete switch has no memoryless Taylor law",
            ));
        }
        Ok(self
            .physical(p, voltage, control, false, |slot, argument, boundary| {
                guards.push(LocalGuard::sign(slot, 0, argument.value.clone(), boundary));
            })?
            .current)
    }

    fn physical<T: Constitutive>(
        self,
        p: Conductance,
        mut voltage: T,
        control: T,
        closed: bool,
        mut guard: impl FnMut(usize, &T, Real),
    ) -> SpiceResult<Evaluation<T>> {
        voltage.check()?;
        control.check()?;
        let mut g = self.controlled(p, control, closed, &mut guard)?;
        if let Direction::Offset(offset) = self.direction {
            voltage = voltage.offset(offset);
            guard(2, &voltage, 0.);
            let sign = voltage.compare(0.);
            if (sign == Ordering::Greater && offset.is_sign_positive())
                || (sign == Ordering::Less && offset.is_sign_negative())
            {
                g = g.constant_like(p.off);
            }
        }
        let mut result = match (self.transfer, self.limit) {
            (Transfer::Arctangent, Some(limit)) => {
                if let Direction::OneWay { offset, epsilon } = self.direction {
                    directional_current(g, limit, voltage.clone(), offset, epsilon, &mut guard)?
                } else {
                    Evaluation::physical(limited_current(g, limit, voltage.clone())?)
                }
            }
            (_, limit) => {
                let raw = g.multiply(&voltage);
                if let Some(limit) = limit {
                    guard(3, &raw, -limit);
                    guard(4, &raw, limit);
                    if raw.compare(-limit) == Ordering::Less
                        || raw.compare(limit) == Ordering::Greater
                    {
                        Evaluation {
                            current: raw.constant_like(limit.copysign(voltage.value())),
                            reference: Reference::Chord(limit / voltage.value().abs()),
                        }
                    } else {
                        Evaluation::physical(raw)
                    }
                } else {
                    Evaluation::physical(raw)
                }
            }
        };
        // Offset level 1 regularizes the shifted voltage; level 2 uses the
        // physical terminal voltage, even for blocked reverse conduction.
        let current = result.current.value();
        result.current = result
            .current
            .add(&voltage.clone().scale(p.gmin))
            .with_value(p.gmin.mul_add(voltage.value(), current));
        result.current.check()?;
        Ok(result)
    }
}

fn limited_current<T: Constitutive>(g: T, limit: Real, voltage: T) -> SpiceResult<T> {
    let linear = g.multiply(&voltage);
    let argument = linear.clone().divide_constant(limit);
    if argument.value().abs() < 1e-8 {
        return Ok(linear);
    }
    Ok(argument.tanh()?.scale(limit))
}

// The finite ten-step Newton law itself is shared across scalar and Taylor
// arithmetic. Derivatives propagate through every actual iteration.
fn epsilon_join<T: Constitutive>(a: &T) -> SpiceResult<T> {
    let mut join = a.clone().scale(0.5);
    for _ in 0..10 {
        let doubled = join.clone().scale(2.);
        let tail = doubled.clone().scale(-1.).exp();
        let sech = tail.clone().scale(2.).divide(&tail.square().offset(1.))?;
        let scaled = a.clone().multiply(&sech);
        let step = scaled.add(&doubled.tanh()?.scale(-1.));
        let next = step.value().mul_add(0.5, join.value());
        join = join.add(&step.scale(0.5)).with_value(next);
    }
    Ok(join)
}

/// The small-signal convention uses the derivative of the implicit root at
/// the final approximate join. Only that reference derivative is separate
/// from the exact ten-step constitutive derivative.
fn implicit_join(a: Real, join: Real) -> Real {
    let tail = (-2. * join).exp();
    a * tail / tail.mul_add(tail, 1.)
}

fn directional_current<T: Constitutive>(
    g: T,
    limit: Real,
    voltage: T,
    offset: Real,
    epsilon: Real,
    guard: &mut impl FnMut(usize, &T, Real),
) -> SpiceResult<Evaluation<T>> {
    let forward = voltage.clone().offset(offset).scale(-1.);
    guard(5, &forward, 0.);
    if forward.compare(0.) != Ordering::Greater {
        return Ok(Evaluation::physical(forward.constant_like(0.)));
    }
    if epsilon == 0. {
        return Ok(Evaluation::physical(limited_current(
            g,
            limit,
            forward.scale(-1.),
        )?));
    }
    let a = g.clone().scale(epsilon).divide_constant(limit);
    // Compare in terminal coordinates, not rounded forward distance.
    let boundary_voltage = -offset - epsilon;
    guard(6, &voltage, boundary_voltage);
    let quadratic = voltage.compare(boundary_voltage) == Ordering::Greater;
    if a.value().is_infinite() {
        let current = if quadratic {
            forward.divide_constant(epsilon).square().scale(-limit)
        } else {
            forward.constant_like(-limit)
        };
        return Ok(Evaluation::physical(current));
    }
    let join = epsilon_join(&a)?;
    let implicit = implicit_join(a.value(), join.value());
    let (current, factor) = if quadratic {
        let fraction = forward.divide_constant(epsilon).square();
        let sech2 = sech_squared(join.value());
        let factor = if sech2 == 0. {
            0.
        } else {
            -limit * sech2 * implicit * fraction.value()
        };
        let boundary = join.tanh()?.scale(-limit);
        (boundary.multiply(&fraction), factor)
    } else {
        let distance = voltage.scale(-1.).offset(boundary_voltage);
        let overdrive = g.clone().multiply(&distance).divide_constant(limit);
        let argument = overdrive.clone().add(&join);
        let sech2 = sech_squared(argument.value());
        let factor = if sech2 == 0. {
            0.
        } else {
            -limit * sech2 * (overdrive.value() + implicit)
        };
        (argument.tanh()?.scale(-limit), factor)
    };
    Ok(Evaluation {
        current,
        reference: Reference::Join {
            factor,
            conductance: g,
        },
    })
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
