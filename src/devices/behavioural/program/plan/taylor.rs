//! Directional Taylor evaluation of the existing immutable value/gradient
//! plan. All roots share the same right-hand branch decisions. Only the
//! active path of a ternary is evaluated, as in the ordinary scalar plan.

use super::{Binary, Environment, Instruction, Operator, Plan, Slot, Unary, describe};
use crate::devices::behavioural::program::{
    EXP_LIMIT, divide, power, power_base_slope, power_exponent_slope, power_h, pwr, unary_value,
};
use crate::maths::taylor::Taylor;
use crate::primitives::{Real, SpiceError, SpiceResult, almost_equal_ulps};

fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "behavioural right-limit derivative".into(),
        message: message.into(),
    }
}

struct Evaluator<'a> {
    plan: &'a Plan,
    variables: &'a [Taylor],
    time: &'a Taylor,
    environment: &'a Environment<'a>,
    values: Vec<Option<Taylor>>,
    pending: Vec<Slot>,
    budget: &'a mut usize,
}

impl Plan {
    pub(in crate::devices::behavioural::program) fn right_taylor(
        &self,
        root: Slot,
        derivatives: &[Slot],
        variables: &[Taylor],
        time: &Taylor,
        environment: &Environment<'_>,
        budget: &mut usize,
    ) -> SpiceResult<(Taylor, Vec<Taylor>)> {
        if variables.iter().any(|value| value.order() != time.order()) {
            return Err(failure("incompatible input Taylor orders"));
        }
        let mut evaluator = Evaluator {
            plan: self,
            variables,
            time,
            environment,
            values: vec![None; self.instructions.len()],
            pending: Vec::new(),
            budget,
        };
        let value = evaluator.evaluate(root)?;
        let gradient = derivatives
            .iter()
            .map(|&slot| evaluator.evaluate(slot))
            .collect::<SpiceResult<_>>()?;
        Ok((value, gradient))
    }
}

impl Evaluator<'_> {
    fn evaluate(&mut self, root: Slot) -> SpiceResult<Taylor> {
        self.pending.push(root);
        while let Some(&slot) = self.pending.last() {
            if self.values[slot].is_none() {
                let instruction = self.plan.instructions[slot];
                if let Some(input) = self.missing(instruction) {
                    self.pending.push(input);
                    continue;
                }
                let work = (self.time.order() + 1).saturating_pow(2);
                *self.budget = self
                    .budget
                    .checked_sub(work)
                    .ok_or_else(|| failure("Taylor plan work budget exceeded"))?;
                self.values[slot] =
                    Some(self.ready(instruction).map_err(|error| {
                        failure(format!("{}: {error}", describe(&instruction)))
                    })?);
            }
            self.pending.pop();
        }
        self.read(root).cloned()
    }

    fn read(&self, slot: Slot) -> SpiceResult<&Taylor> {
        self.values[slot]
            .as_ref()
            .ok_or_else(|| failure("missing Taylor instruction dependency"))
    }

    fn missing(&self, instruction: Instruction) -> Option<Slot> {
        let missing = |slot: Slot| self.values[slot].is_none().then_some(slot);
        match instruction {
            Instruction::Unary(_, a)
            | Instruction::Pwl(a, _)
            | Instruction::PwlDerivative(a, _)
            | Instruction::Table(a, _)
            | Instruction::TableSlope(a, _) => missing(a),
            Instruction::Operator(_, a, b) | Instruction::Binary(_, a, b) => {
                missing(a).or_else(|| missing(b))
            }
            Instruction::Ternary(c, a, b) => missing(c).or_else(|| {
                self.values[c]
                    .as_ref()
                    .and_then(|c| missing(if c.right_sign() == 0. { b } else { a }))
            }),
            _ => None,
        }
    }

    fn ready(&self, instruction: Instruction) -> SpiceResult<Taylor> {
        let order = self.time.order();
        let constant = |value| Taylor::constant(value, order);
        match instruction {
            Instruction::Constant(bits) => {
                constant(Real::from_bits(bits)).with_base(Real::from_bits(bits))
            }
            Instruction::Variable(index) => Ok(self.variables[index].clone()),
            Instruction::Time => Ok(self.time.clone()),
            Instruction::Temperature => Ok(constant(self.environment.temperature)),
            Instruction::Frequency => Ok(constant(self.environment.frequency)),
            Instruction::Unary(op, a) => unary(op, self.read(a)?),
            Instruction::Operator(op, a, b) => {
                let (a, b) = (self.read(a)?, self.read(b)?);
                match op {
                    Operator::Plus => a.add(b),
                    Operator::Minus => a.sub(b),
                    Operator::Times => a.mul(b),
                    Operator::Divide => {
                        let fudge = self.environment.gmin * 1e-20;
                        let adjusted = b.base() + if b.right_sign() < 0. { -fudge } else { fudge };
                        let denominator = b.clone().with_base(adjusted)?;
                        a.div(&denominator)?
                            .with_base(if b.base() == 0. && b.direction() < 0. {
                                a.base() / adjusted
                            } else {
                                divide(a.base(), b.base(), fudge)
                            })
                    }
                    Operator::Power => native_power(a, b, Power::Caret),
                }
            }
            Instruction::Binary(op, a, b) => {
                let (a, b) = (self.read(a)?, self.read(b)?);
                match op {
                    Binary::Pow => native_power(a, b, Power::Ordinary),
                    Binary::Pwr => native_power(a, b, Power::Signed),
                    Binary::PowBaseSlope => base_slope(a, b, false),
                    Binary::PwrBaseSlope => base_slope(a, b, true),
                    Binary::PowExponentSlope => exponent_slope(a, b, false),
                    Binary::PwrExponentSlope => exponent_slope(a, b, true),
                    Binary::Min | Binary::Max => {
                        let sign = a.sub(b)?.right_sign();
                        let select_a = if op == Binary::Min {
                            sign <= 0.
                        } else {
                            sign > 0.
                        };
                        Ok(if select_a { a.clone() } else { b.clone() })
                    }
                }
            }
            Instruction::Ternary(c, a, b) => self
                .read(if self.read(c)?.right_sign() == 0. {
                    b
                } else {
                    a
                })
                .cloned(),
            Instruction::Pwl(a, table) | Instruction::PwlDerivative(a, table) => {
                let arg = self.read(a)?;
                let [(x0, y0), (x1, y1)] =
                    self.plan.pwl[table].right_segment(arg.base(), arg.direction());
                let slope = (y1 - y0) / (x1 - x0);
                if matches!(instruction, Instruction::PwlDerivative(..)) {
                    constant(slope).with_base(slope)
                } else {
                    arg.clone()
                        .with_base(arg.base() - x0)?
                        .scale(slope)?
                        .with_base(slope.mul_add(arg.base() - x0, y0))
                }
            }
            Instruction::Table(a, table) | Instruction::TableSlope(a, table) => {
                let (value, slope) = self.plan.tables[table].right_taylor(self.read(a)?)?;
                Ok(if matches!(instruction, Instruction::TableSlope(..)) {
                    slope
                } else {
                    value
                })
            }
        }
    }
}

#[expect(
    clippy::float_cmp,
    reason = "exact switches and integral exponent branches define the right-hand law"
)]
fn unary(op: Unary, x: &Taylor) -> SpiceResult<Taylor> {
    let base = x.base();
    let value = unary_value(op, base);
    let order = x.order();
    let constant = |value| Taylor::constant(value, order).with_base(value);
    let one = Taylor::constant(1., order);
    let sign = x.right_sign();
    if x.is_constant() {
        return constant(value);
    }
    match op {
        Unary::Abs => x.scale(sign),
        Unary::Uminus => x.scale(-1.),
        Unary::Sgn => constant(sign),
        Unary::Ustep => constant(if sign < 0. { 0. } else { 1. }),
        Unary::Uramp => {
            if sign < 0. {
                constant(0.)
            } else {
                Ok(x.clone())
            }
        }
        Unary::Ustep2 => {
            if sign < 0. {
                constant(0.)
            } else if base > 1. || (base == 1. && x.direction() > 0.) {
                constant(1.)
            } else {
                Ok(x.clone())
            }
        }
        Unary::Eq0 | Unary::Ne0 | Unary::Gt0 | Unary::Lt0 | Unary::Ge0 | Unary::Le0 => {
            constant(unary_value(op, sign))
        }
        Unary::Floor => constant(if base.fract() == 0. && x.direction() < 0. {
            base - 1.
        } else {
            value
        }),
        Unary::Ceil => constant(if base.fract() == 0. && x.direction() > 0. {
            base + 1.
        } else {
            value
        }),
        Unary::Nint => constant(if base.fract().abs() == 0.5 {
            if x.direction() < 0. {
                base.floor()
            } else {
                base.ceil()
            }
        } else {
            value
        }),
        Unary::Exp => {
            if base > EXP_LIMIT || (base == EXP_LIMIT && x.direction() > 0.) {
                constant(1e99)
            } else {
                x.exp(value)
            }
        }
        Unary::Log => x.log(value),
        Unary::Log10 => x
            .log(base.ln())?
            .scale(std::f64::consts::LOG10_E)?
            .with_base(value),
        Unary::Sqrt => x.sqrt(value),
        Unary::Sin | Unary::Cos | Unary::Tan => {
            let (s, c) = x.trig_pair(
                unary_value(Unary::Sin, base),
                unary_value(Unary::Cos, base),
                false,
            )?;
            match op {
                Unary::Sin => Ok(s),
                Unary::Cos => Ok(c),
                _ => s.div(&c)?.with_base(value),
            }
        }
        Unary::Sinh | Unary::Cosh | Unary::Tanh => {
            if op == Unary::Tanh {
                // Stable at large finite arguments: use exp(-2|x|), not
                // an overflowing ratio of sinh and cosh.
                let arg = x.scale(-2. * sign)?;
                let decay = arg.exp(arg.base().exp())?;
                return one
                    .sub(&decay)?
                    .div(&one.add(&decay)?)?
                    .scale(sign)?
                    .with_base(value);
            }
            let (s, c) = x.trig_pair(base.sinh(), base.cosh(), true)?;
            Ok(if op == Unary::Sinh { s } else { c })
        }
        Unary::Asin | Unary::Acos | Unary::Asinh | Unary::Acosh | Unary::Atan | Unary::Atanh => {
            let square = x.mul(x)?;
            let denominator = match op {
                Unary::Asin | Unary::Acos => {
                    let d = one.sub(&square)?;
                    d.sqrt(d.base().sqrt())?
                }
                Unary::Asinh => {
                    let d = one.add(&square)?;
                    d.sqrt(d.base().sqrt())?
                }
                Unary::Acosh => {
                    let d = square.sub(&one)?;
                    d.sqrt(d.base().sqrt())?
                }
                Unary::Atan => one.add(&square)?,
                _ => one.sub(&square)?,
            };
            let numerator = if op == Unary::Acos {
                one.scale(-1.)?
            } else {
                one
            };
            x.compose_derivative(&numerator.div(&denominator)?, value)
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Power {
    Ordinary,
    Caret,
    Signed,
}

fn integral_power(a: &Taylor, exponent: Real) -> SpiceResult<Taylor> {
    if (Real::from(i32::MIN)..=Real::from(i32::MAX)).contains(&exponent) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "caller supplies a finite integral exponent in the i32 range"
        )]
        let exponent = exponent as i32;
        return a.powi(exponent);
    }
    if a.base() == 0. && exponent > 0. {
        // A power beyond i32::MAX has no nonzero coefficient within the
        // bounded right-limit order.
        return Ok(Taylor::constant(0., a.order()));
    }
    let sign = if a.base() < 0. && exponent.rem_euclid(2.) != 0. {
        -1.
    } else {
        1.
    };
    let magnitude = a.scale(if a.base() < 0. { -1. } else { 1. })?;
    let arg = magnitude.log(magnitude.base().ln())?.scale(exponent)?;
    arg.exp(arg.base().exp())?.scale(sign)
}

#[expect(
    clippy::float_cmp,
    reason = "native power kernels distinguish exact zero and integer branches"
)]
fn native_power(a: &Taylor, b: &Taylor, law: Power) -> SpiceResult<Taylor> {
    let value = match law {
        Power::Ordinary => power(a.base(), b.base()),
        Power::Caret => power_h(a.base(), b.base()),
        Power::Signed => pwr(a.base(), b.base()),
    };
    let constant = |value| Taylor::constant(value, a.order()).with_base(value);
    if a.is_constant() && b.is_constant() {
        return constant(value);
    }
    let sign = a.right_sign();
    if sign == 0. {
        if b.base() == 0. && !b.is_constant() {
            return Err(failure("nonregular exponent at a zero base"));
        }
        return constant(value);
    }
    if sign < 0. && law != Power::Signed {
        if almost_equal_ulps(b.base().round_ties_even(), b.base(), 10) {
            return integral_power(a, b.base().round())?.with_base(if a.base() == 0. {
                integral_power(a, b.base().round())?.base()
            } else {
                value
            });
        }
        return constant(0.);
    }
    let magnitude = a.scale(if sign < 0. { -1. } else { 1. })?;
    let sign = if sign < 0. && law == Power::Signed {
        -1.
    } else {
        1.
    };
    let result = if b.is_constant() && b.base().fract() == 0. {
        integral_power(&magnitude, b.base())?.scale(sign)?
    } else {
        let arg = magnitude.log(magnitude.base().ln())?.mul(b)?;
        arg.exp(arg.base().exp())?.scale(sign)?
    };
    // At a zero base the open right-hand branch can differ from an isolated
    // native sample (e.g. pow(0,0)). Preserve the actual right limit.
    if a.base() == 0. {
        Ok(result)
    } else {
        result.with_base(value)
    }
}

#[expect(
    clippy::float_cmp,
    reason = "the scalar law has separate exact zero/integer power branches"
)]
fn base_slope(a: &Taylor, b: &Taylor, signed: bool) -> SpiceResult<Taylor> {
    let value = power_base_slope(a.base(), b.base(), signed);
    if a.is_constant() && b.is_constant() {
        return Taylor::constant(value, a.order()).with_base(value);
    }
    if b.is_constant() && b.base() == 0. {
        return Ok(Taylor::constant(0., a.order()));
    }
    if a.right_sign() == 0. {
        if b.base() > 1. {
            return Ok(Taylor::constant(0., a.order()));
        }
        if b.is_constant() && b.base() == 1. {
            return Ok(Taylor::constant(1., a.order()));
        }
        return Err(failure("nonregular base derivative at zero"));
    }
    let one = Taylor::constant(1., a.order());
    let (base, exponent) = if signed {
        (
            a.scale(if a.right_sign() < 0. { -1. } else { 1. })?,
            b.clone(),
        )
    } else if a.right_sign() < 0. {
        if !almost_equal_ulps(b.base().round_ties_even(), b.base(), 10) {
            return Ok(Taylor::constant(0., a.order()));
        }
        (a.clone(), Taylor::constant(b.base().round(), a.order()))
    } else {
        (a.clone(), b.clone())
    };
    let result = exponent.mul(&native_power(&base, &exponent.sub(&one)?, Power::Caret)?)?;
    if a.base() == 0. {
        Ok(result)
    } else {
        result.with_base(value)
    }
}

fn exponent_slope(a: &Taylor, b: &Taylor, signed: bool) -> SpiceResult<Taylor> {
    let value = power_exponent_slope(a.base(), b.base(), signed);
    if a.is_constant() && b.is_constant() {
        return Taylor::constant(value, a.order()).with_base(value);
    }
    if a.right_sign() == 0. || (!signed && a.right_sign() < 0.) {
        return Ok(Taylor::constant(0., a.order()));
    }
    let magnitude = a.scale(if a.right_sign() < 0. { -1. } else { 1. })?;
    native_power(
        a,
        b,
        if signed {
            Power::Signed
        } else {
            Power::Ordinary
        },
    )?
    .mul(&magnitude.log(magnitude.base().ln())?)?
    .with_base(value)
}
