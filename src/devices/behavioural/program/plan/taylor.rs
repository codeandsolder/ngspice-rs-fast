//! Local right-hand Taylor propagation through the compiled expression DAG.
//! Conditional expressions remain lazy, and branch choice follows the
//! first nonzero coefficient of the input germ instead of a finite sample.

use super::{Binary, Instruction, Operator, Plan, Slot, Unary};
use crate::devices::behavioural::program::{
    EXP_LIMIT, Environment, PowerLaw, PowerPart, TWO_PI, divide, modulus, power, power_h, pwr,
    unary_value,
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
        environment: &Environment<'_>,
        variables: &[Jet],
        time: &Jet,
    ) -> SpiceResult<TaylorEvaluation> {
        if variables.iter().any(|jet| jet.order() != time.order())
            || (root + 1)
                .saturating_mul(time.order() + 1)
                .saturating_mul(2)
                > MAX_JET_ENTRIES
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
            values: vec![None; root + 1],
            pending: Vec::new(),
        };
        let value = evaluator.eval(root)?.clone();
        let gradient = evaluator.gradient(root)?;
        Ok(TaylorEvaluation { value, gradient })
    }
}

fn missing(node: Instruction, values: &[Option<Jet>]) -> Option<Slot> {
    let absent = |slot: Slot| values[slot].is_none().then_some(slot);
    match node {
        Instruction::Unary(_, a) | Instruction::Pwl(a, _) | Instruction::Table(a, _) => absent(a),
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
            Instruction::Pwl(a, table) => {
                let x = self.read(a)?;
                let [(x0, y0), (x1, y1)] = self.pwl_segment(x, table);
                x.clone()
                    .offset(-x0)
                    .scale((y1 - y0) / (x1 - x0))
                    .offset(y0)
            }
            Instruction::Table(a, table) => {
                let x = self.read(a)?;
                let (value, slope, quadratic) = self.table_polynomial(x, table);
                let delta = x.clone().with_value(0.);
                delta
                    .multiply(&delta)
                    .scale(quadratic)
                    .add(&delta.scale(slope))
                    .offset(value)
            }
            Instruction::PwlDerivative(..) | Instruction::TableSlope(..) => {
                return Err(SpiceError::circuit(
                    "derivative-only instruction in physical value graph",
                ));
            }
        })
    }
    fn pwl_segment(&self, x: &Jet, table: usize) -> [(Real, Real); 2] {
        let table = &self.plan.pwl[table];
        let upper = match table.order {
            super::super::PwlOrder::Ascending => Ordering::Less,
            super::super::PwlOrder::Descending => Ordering::Greater,
        };
        table.segment_by(|boundary| x.compare(boundary) == Some(upper))
    }

    fn table_polynomial(&self, x: &Jet, table: usize) -> (Real, Real, Real) {
        self.plan.tables[table].local_polynomial(x.value(), |boundary| x.compare(boundary))
    }

    /// Only the active differentiable children of the executed value graph.
    /// Inactive conditional branches and discrete controls are not evaluated
    /// by the reverse pass.
    fn active_inputs(&self, node: Instruction) -> SpiceResult<[Option<Slot>; 2]> {
        Ok(match node {
            Instruction::Unary(op, a) if !discrete(op) => [Some(a), None],
            Instruction::Pwl(a, _) | Instruction::Table(a, _) => [Some(a), None],
            Instruction::Operator(_, a, b)
            | Instruction::Binary(Binary::Power(_, PowerPart::Value), a, b) => [Some(a), Some(b)],
            Instruction::Binary(op @ (Binary::Min | Binary::Max), a, b) => [
                Some(select_extremum(op, self.read(a)?, self.read(b)?, a, b)),
                None,
            ],
            Instruction::Ternary(c, a, b) => [
                Some(if self.read(c)?.sign() == Ordering::Equal {
                    b
                } else {
                    a
                }),
                None,
            ],
            _ => [None, None],
        })
    }

    /// Reverse-mode differentiation of the actual local value kernels.
    /// The ordinary scalar solver retains its legacy symbolic Newton
    /// slopes; those are not exact derivatives of clipped or regularized
    /// functions and must not be used to establish physical DAE rank.
    fn gradient(&self, root: Slot) -> SpiceResult<Vec<Jet>> {
        let constant = |value| Jet::constant(self.time.order(), value);
        let mut gradient = vec![constant(0.); self.variables.len()];
        let mut active = vec![[None; 2]; root + 1];
        let mut depends = vec![false; root + 1];
        for slot in 0..=root {
            if self.values[slot].is_some() {
                let node = self.plan.instructions[slot];
                active[slot] = self.active_inputs(node)?;
                depends[slot] = matches!(node, Instruction::Variable(_))
                    || active[slot].iter().flatten().any(|&child| depends[child]);
            }
        }
        let mut adjoints = vec![None; root + 1];
        adjoints[root] = Some(constant(1.));
        for slot in (0..=root).rev() {
            let Some(adjoint) = adjoints[slot].take() else {
                continue;
            };
            if !depends[slot] || adjoint.is_zero() {
                continue;
            }
            let node = self.plan.instructions[slot];
            if let Instruction::Variable(variable) = node {
                gradient[variable] = gradient[variable].clone().add(&adjoint);
                gradient[variable].check()?;
                continue;
            }
            for (side, child) in active[slot].into_iter().enumerate() {
                let Some(child) = child.filter(|&child| depends[child]) else {
                    continue;
                };
                let partial = self.partial(slot, node, side)?;
                let contribution = adjoint.multiply(&partial);
                contribution.check()?;
                let sum = match adjoints[child].take() {
                    Some(old) => old.add(&contribution),
                    None => contribution,
                };
                sum.check()?;
                adjoints[child] = Some(sum);
            }
        }
        Ok(gradient)
    }

    fn partial(&self, slot: Slot, node: Instruction, side: usize) -> SpiceResult<Jet> {
        let constant = |value| Jet::constant(self.time.order(), value);
        Ok(match node {
            Instruction::Unary(op, a) => unary_partial(op, self.read(a)?, self.read(slot)?)?,
            Instruction::Operator(op, a, b) => operator_partial(
                op,
                self.read(a)?,
                self.read(b)?,
                self.read(slot)?,
                side,
                self.environment.gmin * 1e-20,
            )?,
            Instruction::Binary(Binary::Power(law, PowerPart::Value), a, b) => {
                typed_power_partial(law, self.read(a)?, self.read(b)?, side != 0)?
            }
            Instruction::Binary(Binary::Min | Binary::Max, ..) | Instruction::Ternary(..) => {
                constant(1.)
            }
            Instruction::Pwl(a, table) => {
                let [(x0, y0), (x1, y1)] = self.pwl_segment(self.read(a)?, table);
                constant((y1 - y0) / (x1 - x0))
            }
            Instruction::Table(a, table) => {
                let x = self.read(a)?;
                let (_, slope, quadratic) = self.table_polynomial(x, table);
                x.clone().with_value(0.).scale(2. * quadratic).offset(slope)
            }
            _ => return Err(SpiceError::circuit("invalid physical Taylor partial")),
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

const fn discrete(op: Unary) -> bool {
    matches!(
        op,
        Unary::Sgn
            | Unary::Ustep
            | Unary::Ceil
            | Unary::Floor
            | Unary::Nint
            | Unary::Eq0
            | Unary::Ne0
            | Unary::Gt0
            | Unary::Lt0
            | Unary::Ge0
            | Unary::Le0
    )
}

fn unary_partial(op: Unary, x: &Jet, value: &Jet) -> SpiceResult<Jet> {
    let constant = |value| Jet::constant(x.order(), value);
    let sign = x.sign();
    Ok(match op {
        Unary::Abs => constant(match sign {
            Ordering::Less => -1.,
            Ordering::Equal => 0.,
            Ordering::Greater => 1.,
        }),
        Unary::Uminus => constant(-1.),
        Unary::Exp if x.compare(EXP_LIMIT) == Some(Ordering::Greater) => constant(0.),
        Unary::Exp => value.clone(),
        Unary::Log => x.reciprocal()?,
        Unary::Log10 => x.reciprocal()?.scale(std::f64::consts::LOG10_E),
        Unary::Sqrt => value.reciprocal()?.scale(0.5),
        Unary::Sin => x.clone().with_value(modulus(x.value(), TWO_PI)).sin_cos().1,
        Unary::Cos => x
            .clone()
            .with_value(modulus(x.value(), TWO_PI))
            .sin_cos()
            .0
            .scale(-1.),
        Unary::Tan => value.multiply(value).offset(1.),
        Unary::Sinh => x.sinh_cosh().1,
        Unary::Cosh => x.sinh_cosh().0,
        Unary::Tanh => value.multiply(value).scale(-1.).offset(1.),
        Unary::Acos | Unary::Asin | Unary::Acosh | Unary::Asinh | Unary::Atan | Unary::Atanh => {
            inverse_slope(op, x)?
        }
        Unary::Uramp => constant(if sign == Ordering::Less { 0. } else { 1. }),
        Unary::Ustep2 => constant(
            if sign == Ordering::Less || x.compare(1.) == Some(Ordering::Greater) {
                0.
            } else {
                1.
            },
        ),
        _ => constant(0.),
    })
}

fn operator_partial(
    op: Operator,
    a: &Jet,
    b: &Jet,
    value: &Jet,
    side: usize,
    fudge: Real,
) -> SpiceResult<Jet> {
    let constant = |value| Jet::constant(a.order(), value);
    Ok(match op {
        Operator::Plus => constant(1.),
        Operator::Minus => constant(if side == 0 { 1. } else { -1. }),
        Operator::Times => {
            if side == 0 {
                b.clone()
            } else {
                a.clone()
            }
        }
        Operator::Divide => {
            let denominator = b.clone().offset(if b.sign() == Ordering::Less {
                -fudge
            } else {
                fudge
            });
            if side == 0 {
                denominator.reciprocal()?
            } else {
                value.divide(&denominator)?.scale(-1.)
            }
        }
        Operator::Power => power_partial(a, b, false, side != 0)?,
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
    if b.is_constant() && b.value() == 0. {
        return Ok(constant(0.));
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

/// Sine and cosine of the principal complex-power phase. Reduce only the
/// center, then propagate the displacement from its exact (possibly
/// integral) phase, so integer branches never acquire spurious derivatives.
fn principal_phase(b: &Jet) -> (Jet, Jet) {
    let pi = std::f64::consts::PI;
    let center = b.value().rem_euclid(2.);
    let (s0, c0) = if b.value().fract() == 0. {
        (0., if center == 0. { 1. } else { -1. })
    } else {
        (pi * center).sin_cos()
    };
    let (s, c) = b.clone().with_value(0.).scale(pi).sin_cos();
    (
        s.clone().scale(c0).add(&c.clone().scale(s0)),
        c.scale(c0).add(&s.scale(-s0)),
    )
}

fn magnitude(a: &Jet) -> Jet {
    if a.sign() == Ordering::Less {
        a.clone().scale(-1.)
    } else {
        a.clone()
    }
}

fn nonregular_power() -> SpiceError {
    SpiceError::circuit("power law has no regular local physical Taylor expansion")
}

fn typed_power_jet(law: PowerLaw, a: &Jet, b: &Jet) -> SpiceResult<Jet> {
    match law {
        PowerLaw::NativePow => return power_jet(a, b, PowerKind::Ordinary),
        PowerLaw::NativePwr => return power_jet(a, b, PowerKind::Signed),
        _ => {}
    }
    let scalar = law.evaluate(PowerPart::Value, a.value(), b.value());
    let constant = |value| Jet::constant(a.order(), value);
    if a.is_constant() && b.is_constant() {
        return Ok(constant(scalar));
    }
    if a.sign() == Ordering::Equal {
        if b.sign() != Ordering::Greater {
            return Err(nonregular_power());
        }
        // The caret law's measured floor at exactly zero is a separate
        // value branch, not a surrogate derivative of the positive-base law.
        return Ok(if law == PowerLaw::SimetrixCaret {
            b.clone().scale(1e-308_f64.ln()).exp()
        } else {
            constant(0.)
        });
    }
    let negative = a.sign() == Ordering::Less;
    let result = match law {
        PowerLaw::SimetrixCaret if negative => {
            if !b.is_constant() || b.value().fract() != 0. {
                return Err(nonregular_power());
            }
            a.power(b)?
        }
        PowerLaw::SimetrixAbsolute => magnitude(a).power(b)?,
        PowerLaw::SimetrixRealPart if negative => {
            magnitude(a).power(b)?.multiply(&principal_phase(b).1)
        }
        PowerLaw::SimetrixCaret | PowerLaw::SimetrixRealPart => a.power(b)?,
        _ => return Err(nonregular_power()),
    };
    // At a zero crossing use the actual one-sided expansion, not the
    // separately regularized scalar value at the isolated boundary.
    Ok(if a.value() == 0. {
        result
    } else {
        result.with_value(scalar)
    })
}

fn zero_base_partial(law: PowerLaw, a: &Jet, b: &Jet, exponent: bool) -> SpiceResult<Jet> {
    let constant = |value| Jet::constant(a.order(), value);
    if exponent {
        if b.sign() != Ordering::Greater {
            return Err(nonregular_power());
        }
        if law != PowerLaw::SimetrixCaret {
            return Ok(constant(0.));
        }
        if b.is_constant() && b.value().fract() == 0. {
            // The isolated exact-integer branch has no exponent derivative.
            return Err(nonregular_power());
        }
        return Ok(b
            .clone()
            .scale(1e-308_f64.ln())
            .exp()
            .scale(1e-308_f64.ln()));
    }
    if b.is_zero() {
        return Ok(constant(0.));
    }
    if law == PowerLaw::SimetrixCaret && (!b.is_constant() || b.value().fract() != 0.) {
        // A finite scalar floor at zero is not a physical base derivative.
        return Err(nonregular_power());
    }
    match b.compare(1.) {
        Some(Ordering::Greater) => Ok(constant(0.)),
        Some(Ordering::Equal) if law != PowerLaw::SimetrixAbsolute => Ok(constant(1.)),
        _ => Err(nonregular_power()),
    }
}

fn typed_power_partial(law: PowerLaw, a: &Jet, b: &Jet, exponent: bool) -> SpiceResult<Jet> {
    match law {
        PowerLaw::NativePow => return power_partial(a, b, false, exponent),
        PowerLaw::NativePwr => return power_partial(a, b, true, exponent),
        _ => {}
    }
    if a.sign() == Ordering::Equal {
        return zero_base_partial(law, a, b, exponent);
    }
    let negative = a.sign() == Ordering::Less;
    let mag = magnitude(a);
    if exponent {
        if negative && law == PowerLaw::SimetrixCaret {
            return Err(nonregular_power());
        }
        let log = mag.ln()?;
        let value = mag.power(b)?;
        if negative && law == PowerLaw::SimetrixRealPart {
            let (sin, cos) = principal_phase(b);
            return Ok(value.multiply(&log.multiply(&cos).add(&sin.scale(-std::f64::consts::PI))));
        }
        return Ok(value.multiply(&log));
    }
    if b.is_zero() {
        return Ok(Jet::constant(a.order(), 0.));
    }
    let shifted = b.clone().offset(-1.);
    if law == PowerLaw::SimetrixCaret {
        return Ok(a.power(&shifted)?.multiply(b));
    }
    let mut slope = mag.power(&shifted)?.multiply(b);
    if negative {
        slope = slope.scale(-1.);
        if law == PowerLaw::SimetrixRealPart {
            slope = slope.multiply(&principal_phase(b).1);
        }
    }
    Ok(slope)
}

fn select_extremum<T>(op: Binary, a: &Jet, b: &Jet, left: T, right: T) -> T {
    let comparison = a.clone().add(&b.clone().scale(-1.)).sign();
    if (op == Binary::Min && comparison == Ordering::Greater)
        || (op == Binary::Max && comparison != Ordering::Greater)
    {
        right
    } else {
        left
    }
}

fn binary(op: Binary, a: &Jet, b: &Jet) -> SpiceResult<Jet> {
    match op {
        Binary::Power(law, PowerPart::Value) => typed_power_jet(law, a, b),
        Binary::Min | Binary::Max => Ok(select_extremum(op, a, b, a, b).clone()),
        Binary::Power(_, PowerPart::Base | PowerPart::Exponent) => Err(SpiceError::circuit(
            "derivative-only instruction in physical value graph",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::behavioural::program::Program;
    use crate::netlist::Parser;
    use crate::primitives::SourceLoc;

    fn evaluate(text: &str, variables: &[Jet]) -> TaylorEvaluation {
        try_evaluate(text, variables).unwrap()
    }

    fn try_evaluate(text: &str, variables: &[Jet]) -> SpiceResult<TaylorEvaluation> {
        let expression = Parser::new()
            .parse_behavioural_expression(
                text,
                &SourceLoc::new(std::path::PathBuf::from("physical-jacobian.cir"), 1, 1),
                true,
            )
            .unwrap()
            .root;
        let program = Program::compile(&expression).unwrap();
        let values: Vec<_> = variables.iter().map(Jet::value).collect();
        let environment = Environment {
            values: &values,
            time: 0.,
            temperature: 27.,
            gmin: 1e-12,
            frequency: 0.,
        };
        program.evaluate_taylor(
            &environment,
            variables,
            &Jet::affine(variables[0].order(), 0., 1.),
        )
    }

    fn close(actual: &Jet, expected: &[Real]) {
        assert_eq!(actual.coefficients().len(), expected.len());
        for (&a, &b) in actual.coefficients().iter().zip(expected) {
            assert!(
                (a - b).abs() <= 2e-13 * b.abs().max(1e-20),
                "{a:e} != {b:e}"
            );
        }
    }

    #[test]
    fn physical_jacobian_is_zero_on_clipped_exponential_branch() {
        let result = evaluate("exp(v(a))", &[Jet::affine(3, 230., 1.)]);
        close(&result.value, &[1e99, 0., 0., 0.]);
        close(&result.gradient[0], &[0.; 4]);
        let result = evaluate("exp(v(a))", &[Jet::affine(3, 0., 1.)]);
        close(&result.value, &[1., 1., 0.5, 1. / 6.]);
        close(&result.gradient[0], &[1., 1., 0.5, 1. / 6.]);
    }

    #[test]
    fn physical_division_jacobian_regularizes_only_the_value_denominator() {
        let result = evaluate(
            "v(a)/v(b)",
            &[Jet::constant(3, 1e-32), Jet::affine(3, 1e-32, 1e-32)],
        );
        close(&result.value, &[0.5, -0.25, 0.125, -0.0625]);
        close(&result.gradient[0], &[5e31, -2.5e31, 1.25e31, -6.25e30]);
        close(&result.gradient[1], &[-2.5e31, 2.5e31, -1.875e31, 1.25e31]);
    }

    #[test]
    fn physical_reverse_pass_accumulates_shared_operands_and_ignores_inactive_branches() {
        let result = evaluate("v(a)*v(a)+v(a)^3", &[Jet::affine(3, 2., 1.)]);
        close(&result.value, &[12., 16., 7., 1.]);
        close(&result.gradient[0], &[16., 14., 3., 0.]);
        let result = evaluate(
            "v(a)>0 ? 2*v(b) : sqrt(-1)",
            &[Jet::affine(3, 1., 1.), Jet::affine(3, 2., 3.)],
        );
        close(&result.value, &[4., 6., 0., 0.]);
        close(&result.gradient[0], &[0.; 4]);
        close(&result.gradient[1], &[2., 0., 0., 0.]);
    }

    #[test]
    fn physical_reverse_pass_uses_the_same_germ_as_value_selection() {
        let result = evaluate(
            "min(v(a),v(b))",
            &[Jet::affine(3, 1., 1.), Jet::affine(3, 1., -1.)],
        );
        close(&result.value, &[1., -1., 0., 0.]);
        close(&result.gradient[0], &[0.; 4]);
        close(&result.gradient[1], &[1., 0., 0., 0.]);
        for (direction, slope) in [(1., -1.), (-1., 2.)] {
            let result = evaluate("pwl(v(a),0,0,1,2,2,1)", &[Jet::affine(3, 1., direction)]);
            close(&result.value, &[2., slope * direction, 0., 0.]);
            close(&result.gradient[0], &[slope, 0., 0., 0.]);
        }
    }
    #[test]
    fn typed_power_laws_preserve_distinct_negative_base_germs() {
        for (name, sign) in [
            ("simetrix_abs_pow", 1.),
            ("simetrix_caret_pow", -1.),
            ("simetrix_pow", -1.),
        ] {
            let result = evaluate(&format!("{name}(v(a),3)"), &[Jet::affine(3, -2., 1.)]);
            close(&result.value, &[8. * sign, -12. * sign, 6. * sign, -sign]);
            close(
                &result.gradient[0],
                &[-12. * sign, 12. * sign, -3. * sign, 0.],
            );
        }
    }

    #[test]
    fn real_component_power_differentiates_principal_phase() {
        let result = evaluate(
            "simetrix_pow(v(a),v(b))",
            &[Jet::constant(2, -2.), Jet::affine(2, 2., 1.)],
        );
        let log = 2_f64.ln();
        let pi = std::f64::consts::PI;
        let second = log.mul_add(log, -pi * pi);
        close(&result.value, &[4., 4. * log, 2. * second]);
        close(
            &result.gradient[0],
            &[-4., -2. - 4. * log, -2. * (second + log)],
        );
        close(
            &result.gradient[1],
            &[
                4. * log,
                4. * second,
                2. * log * log.mul_add(log, -3. * pi * pi),
            ],
        );
    }

    #[test]
    fn typed_zero_crossing_uses_the_selected_germ_or_rejects_nonregular_rank() {
        for direction in [-1., 1.] {
            let result = evaluate("simetrix_abs_pow(v(a),2)", &[Jet::affine(2, 0., direction)]);
            close(&result.value, &[0., 0., 1.]);
            close(&result.gradient[0], &[0., 2. * direction, 0.]);
            let result = evaluate("simetrix_abs_pow(v(a),1)", &[Jet::affine(2, 0., direction)]);
            close(&result.value, &[0., 1., 0.]);
            close(&result.gradient[0], &[direction, 0., 0.]);
        }
        assert!(try_evaluate("simetrix_abs_pow(v(a),1)", &[Jet::constant(2, 0.)]).is_err());
        assert!(try_evaluate("simetrix_caret_pow(v(a),0.5)", &[Jet::affine(2, 0., 1.)]).is_err());
        assert!(
            try_evaluate(
                "simetrix_caret_pow(v(a),v(b))",
                &[Jet::constant(2, -2.), Jet::affine(2, 2., 1.)],
            )
            .is_err()
        );
    }
}
