//! Local right-hand Taylor propagation through the compiled expression DAG.
//! Conditional expressions remain lazy, and branch choice follows the
//! first nonzero coefficient of the input germ instead of a finite sample.

use super::{Binary, Instruction, Operator, Plan, Slot, Unary};
use crate::devices::behavioural::program::{
    EXP_LIMIT, Environment, TWO_PI, divide, modulus, power, power_base_slope, power_exponent_slope,
    power_h, pwr, unary_value,
};
use crate::maths::taylor::Jet;
use crate::primitives::{Real, SpiceError, SpiceResult, almost_equal_ulps};
use std::cmp::Ordering;

const MAX_JET_ENTRIES: usize = 8_388_608;

pub(in crate::devices::behavioural) struct TaylorEvaluation {
    pub value: Jet,
    pub gradient: Vec<Jet>,
}

struct Evaluator<'a> {
    plan: &'a Plan,
    environment: &'a Environment<'a>,
    variables: &'a [Jet],
    time: &'a Jet,
    values: Vec<Option<Jet>>,
    pending: Vec<Slot>,
}

impl Plan {
    pub(in crate::devices::behavioural::program) fn evaluate_taylor(
        &self,
        root: Slot,
        derivatives: &[Slot],
        environment: &Environment<'_>,
        variables: &[Jet],
        time: &Jet,
    ) -> SpiceResult<TaylorEvaluation> {
        if variables.iter().any(|jet| jet.order() != time.order())
            || self.instructions.len().saturating_mul(time.order() + 1) > MAX_JET_ENTRIES
        {
            return Err(SpiceError::circuit(
                "behavioural Taylor evaluation resource limit or inconsistent order",
            ));
        }
        let mut evaluator = Evaluator {
            plan: self,
            environment,
            variables,
            time,
            values: vec![None; self.instructions.len()],
            pending: Vec::new(),
        };
        let value = evaluator.eval(root)?.clone();
        let mut gradient = Vec::with_capacity(derivatives.len());
        for &slot in derivatives {
            gradient.push(evaluator.eval(slot)?.clone());
        }
        Ok(TaylorEvaluation { value, gradient })
    }
}

fn missing(node: Instruction, values: &[Option<Jet>]) -> Option<Slot> {
    let absent = |slot: Slot| values[slot].is_none().then_some(slot);
    match node {
        Instruction::Unary(_, a)
        | Instruction::Pwl(a, _)
        | Instruction::PwlDerivative(a, _)
        | Instruction::Table(a, _)
        | Instruction::TableSlope(a, _) => absent(a),
        Instruction::Operator(_, a, b) | Instruction::Binary(_, a, b) => {
            absent(a).or_else(|| absent(b))
        }
        Instruction::Ternary(c, a, b) => absent(c).or_else(|| {
            values[c].as_ref().and_then(|value| {
                absent(if value.sign() == Ordering::Equal {
                    b
                } else {
                    a
                })
            })
        }),
        _ => None,
    }
}

impl Evaluator<'_> {
    fn eval(&mut self, root: Slot) -> SpiceResult<&Jet> {
        self.pending.push(root);
        while let Some(&slot) = self.pending.last() {
            if self.values[slot].is_none() {
                let instruction = self.plan.instructions[slot];
                if let Some(input) = missing(instruction, &self.values) {
                    self.pending.push(input);
                    continue;
                }
                let value = self.ready(instruction)?;
                value.check()?;
                self.values[slot] = Some(value);
            }
            self.pending.pop();
        }
        self.read(root)
    }

    fn read(&self, slot: Slot) -> SpiceResult<&Jet> {
        self.values[slot]
            .as_ref()
            .ok_or_else(|| SpiceError::circuit("missing behavioural Taylor operand"))
    }

    fn ready(&self, node: Instruction) -> SpiceResult<Jet> {
        let constant = |value| Jet::constant(self.time.order(), value);
        Ok(match node {
            Instruction::Constant(bits) => constant(Real::from_bits(bits)),
            Instruction::Variable(index) => self.variables[index].clone(),
            Instruction::Time => self.time.clone(),
            Instruction::Temperature => constant(self.environment.temperature),
            Instruction::Frequency => constant(self.environment.frequency),
            Instruction::Unary(op, a) => unary(op, self.read(a)?)?,
            Instruction::Operator(op, a, b) => operator(
                op,
                self.read(a)?,
                self.read(b)?,
                self.environment.gmin * 1e-20,
            )?,
            Instruction::Binary(op, a, b) => binary(op, self.read(a)?, self.read(b)?)?,
            Instruction::Ternary(c, a, b) => self
                .read(if self.read(c)?.sign() == Ordering::Equal {
                    b
                } else {
                    a
                })?
                .clone(),
            Instruction::Pwl(a, table) | Instruction::PwlDerivative(a, table) => {
                let x = self.read(a)?;
                let table = &self.plan.pwl[table];
                let upper = match table.order {
                    super::super::PwlOrder::Ascending => Ordering::Less,
                    super::super::PwlOrder::Descending => Ordering::Greater,
                };
                let [(x0, y0), (x1, y1)] =
                    table.segment_by(|boundary| x.compare(boundary) == Some(upper));
                let slope = (y1 - y0) / (x1 - x0);
                if matches!(node, Instruction::PwlDerivative(..)) {
                    constant(slope)
                } else {
                    x.clone().offset(-x0).scale(slope).offset(y0)
                }
            }
            Instruction::Table(a, table) | Instruction::TableSlope(a, table) => {
                let x = self.read(a)?;
                let (value, slope, quadratic) = self.plan.tables[table]
                    .local_polynomial(x.value(), |boundary| x.compare(boundary));
                let delta = x.clone().with_value(0.);
                if matches!(node, Instruction::TableSlope(..)) {
                    delta.scale(2. * quadratic).offset(slope)
                } else {
                    delta
                        .multiply(&delta)
                        .scale(quadratic)
                        .add(&delta.scale(slope))
                        .offset(value)
                }
            }
        })
    }
}

fn unary(op: Unary, x: &Jet) -> SpiceResult<Jet> {
    let constant = |value| Jet::constant(x.order(), value);
    if x.is_constant() {
        return Ok(constant(unary_value(op, x.value())));
    }
    let value = unary_value(op, x.value());
    let sign = x.sign();
    Ok(match op {
        Unary::Abs => {
            if sign == Ordering::Less {
                x.clone().scale(-1.)
            } else {
                x.clone()
            }
        }
        Unary::Uminus => x.clone().scale(-1.),
        Unary::Exp => {
            if x.clone().offset(-EXP_LIMIT).sign() == Ordering::Greater {
                constant(1e99)
            } else {
                x.exp()
            }
        }
        Unary::Log => x.ln()?,
        Unary::Log10 => x.ln()?.scale(std::f64::consts::LOG10_E).with_value(value),
        Unary::Sqrt => x.sqrt()?,
        Unary::Sin => x.clone().with_value(modulus(x.value(), TWO_PI)).sin_cos().0,
        Unary::Cos => x.clone().with_value(modulus(x.value(), TWO_PI)).sin_cos().1,
        Unary::Tan => {
            let (sine, cosine) = x
                .clone()
                .with_value(modulus(x.value(), std::f64::consts::PI))
                .sin_cos();
            sine.divide(&cosine)?.with_value(value)
        }
        Unary::Sinh => x.sinh_cosh().0,
        Unary::Cosh => x.sinh_cosh().1,
        Unary::Tanh => x.tanh(),
        Unary::Acos | Unary::Asin | Unary::Acosh | Unary::Asinh | Unary::Atan | Unary::Atanh => {
            x.compose_derivative(value, &inverse_slope(op, x)?)
        }
        Unary::Uramp => {
            if sign == Ordering::Less {
                constant(0.)
            } else {
                x.clone()
            }
        }
        Unary::Ustep2 => {
            if sign == Ordering::Less {
                constant(0.)
            } else if x.clone().offset(-1.).sign() == Ordering::Greater {
                constant(1.)
            } else {
                x.clone()
            }
        }
        Unary::Sgn
        | Unary::Ustep
        | Unary::Eq0
        | Unary::Ne0
        | Unary::Gt0
        | Unary::Lt0
        | Unary::Ge0
        | Unary::Le0 => constant(unary_value(
            op,
            match sign {
                Ordering::Less => -1.,
                Ordering::Equal => 0.,
                Ordering::Greater => 1.,
            },
        )),
        Unary::Floor | Unary::Ceil | Unary::Nint => constant(discrete_limit(op, x)),
    })
}

fn discrete_limit(op: Unary, x: &Jet) -> Real {
    let value = x.value();
    match (op, x.direction()) {
        (Unary::Floor, Ordering::Less) if value.fract() == 0. => value - 1.,
        (Unary::Ceil, Ordering::Greater) if value.fract() == 0. => value + 1.,
        (Unary::Nint, Ordering::Less) if value.fract().abs().to_bits() == 0.5_f64.to_bits() => {
            value.floor()
        }
        (Unary::Nint, Ordering::Greater) if value.fract().abs().to_bits() == 0.5_f64.to_bits() => {
            value.ceil()
        }
        _ => unary_value(op, value),
    }
}

fn inverse_slope(op: Unary, x: &Jet) -> SpiceResult<Jet> {
    let square = x.multiply(x);
    let radicand = match op {
        Unary::Acos | Unary::Asin | Unary::Atanh => square.scale(-1.).offset(1.),
        Unary::Acosh => square.offset(-1.),
        _ => square.offset(1.),
    };
    let denominator = if matches!(op, Unary::Atan | Unary::Atanh) {
        radicand
    } else {
        radicand.sqrt()?
    };
    let slope = denominator.reciprocal()?;
    Ok(if op == Unary::Acos {
        slope.scale(-1.)
    } else {
        slope
    })
}

fn operator(op: Operator, a: &Jet, b: &Jet, fudge: Real) -> SpiceResult<Jet> {
    Ok(match op {
        Operator::Plus => a.clone().add(b),
        Operator::Minus => a.clone().add(&b.clone().scale(-1.)),
        Operator::Times => a.multiply(b),
        Operator::Divide => {
            let offset = if b.sign() == Ordering::Less {
                -fudge
            } else {
                fudge
            };
            let result = a.divide(&b.clone().offset(offset))?;
            if b.value() == 0. {
                result
            } else {
                result.with_value(divide(a.value(), b.value(), fudge))
            }
        }
        Operator::Power => power_jet(a, b, PowerKind::Operator)?,
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PowerKind {
    Operator,
    Ordinary,
    Signed,
}

fn power_jet(a: &Jet, b: &Jet, kind: PowerKind) -> SpiceResult<Jet> {
    let constant = |value| Jet::constant(a.order(), value);
    let scalar = match kind {
        PowerKind::Operator => power_h(a.value(), b.value()),
        PowerKind::Ordinary => power(a.value(), b.value()),
        PowerKind::Signed => pwr(a.value(), b.value()),
    };
    if a.is_constant() && b.is_constant() {
        return Ok(constant(scalar));
    }
    if a.sign() == Ordering::Equal {
        if kind == PowerKind::Ordinary || b.value() > 0. {
            return Ok(constant(0.));
        }
        return Err(SpiceError::circuit(
            "zero base has no regular Taylor power at this exponent",
        ));
    }
    let negative = a.sign() == Ordering::Less;
    let result = if kind == PowerKind::Signed && negative {
        a.clone().scale(-1.).power(b)?.scale(-1.)
    } else if negative {
        // Negative-base ordinary powers use the existing quasi-integer
        // exponent convention. No derivative w.r.t. exponent in that branch.
        if almost_equal_ulps(b.value().round_ties_even(), b.value(), 10) {
            a.power(&constant(b.value().round()))?
        } else {
            constant(0.)
        }
    } else {
        a.power(b)?
    };
    // At a zero crossing the mathematical right limit can differ from the
    // reference scalar pow(0,0); preserve the branch-local expansion.
    Ok(if a.value() == 0. {
        result
    } else {
        result.with_value(scalar)
    })
}

fn power_partial(a: &Jet, b: &Jet, signed: bool, exponent: bool) -> SpiceResult<Jet> {
    let constant = |value| Jet::constant(a.order(), value);
    let scalar = if exponent {
        power_exponent_slope(a.value(), b.value(), signed)
    } else {
        power_base_slope(a.value(), b.value(), signed)
    };
    if a.is_constant() && b.is_constant() {
        return Ok(constant(scalar));
    }
    let negative = a.sign() == Ordering::Less;
    if exponent {
        if a.sign() == Ordering::Equal || (!signed && negative) {
            return Ok(constant(0.));
        }
        let magnitude = if negative {
            a.clone().scale(-1.)
        } else {
            a.clone()
        };
        return Ok(power_jet(
            a,
            b,
            if signed {
                PowerKind::Signed
            } else {
                PowerKind::Ordinary
            },
        )?
        .multiply(&magnitude.ln()?));
    }
    if a.sign() == Ordering::Equal {
        return match b.compare(1.) {
            Some(Ordering::Greater) => Ok(constant(0.)),
            Some(Ordering::Equal) => Ok(constant(1.)),
            _ => Err(SpiceError::circuit(
                "singular zero-base Taylor power derivative",
            )),
        };
    }
    if b.is_constant() && b.value() == 0. {
        return Ok(constant(0.));
    }
    if negative && !signed {
        if !almost_equal_ulps(b.value().round_ties_even(), b.value(), 10) {
            return Ok(constant(0.));
        }
        let exponent = b.value().round();
        return Ok(a.power(&constant(exponent - 1.))?.scale(exponent));
    }
    let magnitude = if signed && negative {
        a.clone().scale(-1.)
    } else {
        a.clone()
    };
    Ok(magnitude.power(&b.clone().offset(-1.))?.multiply(b))
}

fn binary(op: Binary, a: &Jet, b: &Jet) -> SpiceResult<Jet> {
    Ok(match op {
        Binary::Pow => power_jet(a, b, PowerKind::Ordinary)?,
        Binary::Pwr => power_jet(a, b, PowerKind::Signed)?,
        Binary::PowBaseSlope => power_partial(a, b, false, false)?,
        Binary::PwrBaseSlope => power_partial(a, b, true, false)?,
        Binary::PowExponentSlope => power_partial(a, b, false, true)?,
        Binary::PwrExponentSlope => power_partial(a, b, true, true)?,
        Binary::Min | Binary::Max => {
            let comparison = a.clone().add(&b.clone().scale(-1.)).sign();
            if (op == Binary::Min && comparison == Ordering::Greater)
                || (op == Binary::Max && comparison != Ordering::Greater)
            {
                b.clone()
            } else {
                a.clone()
            }
        }
    })
}
