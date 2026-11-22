//! Sparse symbolic differentiation in the instruction interner.
//!
//! A derivative is a default instruction plus ordered variable-specific
//! overrides. The default is obtained by applying the *same* C symbolic
//! rule to absent variables, not by assuming that the result is zero:
//! constant folding can produce signed zero or a non-finite constant.
//! Intermediate instructions are never evaluated unless reachable from
//! the value or a requested derivative.

use super::{Binary, Instruction, Operator, Plan, PlanBuilder, Slot, Unary};
use crate::devices::behavioural::program::{c_pow, unary_value, unsupported};
use crate::netlist::bexpr::BExpr;
use crate::primitives::{Real, SpiceResult};
use std::ops::Range;

const MAX_DERIVATIVE_VISITS: usize = 131_072;

struct SparseDerivative {
    default: Slot,
    overrides: Range<usize>,
}

impl Instruction {
    // Only these children supply derivatives; the ternary condition is
    // still a value operand, but C never differentiates it.
    const fn derivative_inputs(self) -> [Option<Slot>; 2] {
        match self {
            Self::Unary(_, a)
            | Self::Pwl(a, _)
            | Self::PwlDerivative(a, _)
            | Self::Table(a, _)
            | Self::TableSlope(a, _) => [Some(a), None],
            Self::Operator(_, a, b) | Self::Binary(_, a, b) | Self::Ternary(_, a, b) => {
                [Some(a), Some(b)]
            }
            _ => [None, None],
        }
    }

    const fn inputs(self) -> [Option<Slot>; 3] {
        if let Self::Ternary(c, a, b) = self {
            [Some(c), Some(a), Some(b)]
        } else {
            let [a, b] = self.derivative_inputs();
            [a, b, None]
        }
    }

    fn remap(self, map: &[Slot]) -> Self {
        match self {
            Self::Unary(op, a) => Self::Unary(op, map[a]),
            Self::Operator(op, a, b) => Self::Operator(op, map[a], map[b]),
            Self::Binary(op, a, b) => Self::Binary(op, map[a], map[b]),
            Self::Ternary(c, a, b) => Self::Ternary(map[c], map[a], map[b]),
            Self::Pwl(a, t) => Self::Pwl(map[a], t),
            Self::PwlDerivative(a, t) => Self::PwlDerivative(map[a], t),
            Self::Table(a, t) => Self::Table(map[a], t),
            Self::TableSlope(a, t) => Self::TableSlope(map[a], t),
            leaf => leaf,
        }
    }
}

impl PlanBuilder {
    pub(in crate::devices::behavioural::program) fn sparse_derivatives(
        &mut self,
        root: Slot,
        quantities: usize,
        expr: &BExpr,
    ) -> SpiceResult<Vec<Slot>> {
        if quantities == 0 {
            return Ok(Vec::new());
        }
        let source_count = self.plan.instructions.len();
        let zero = self.constant(0.)?;
        // Do not differentiate condition-only subgraphs. They remain in the
        // value plan and retain their original lazy evaluation semantics.
        let mut needed = vec![false; source_count];
        let mut pending = vec![root];
        while let Some(slot) = pending.pop() {
            if !core::mem::replace(&mut needed[slot], true) {
                pending.extend(
                    self.plan.instructions[slot]
                        .derivative_inputs()
                        .into_iter()
                        .flatten(),
                );
            }
        }
        let mut partials: Vec<SparseDerivative> = Vec::with_capacity(source_count);
        // All overrides use a single arena, including leaves. No per-node or
        // per-variable heap allocations; earlier ranges survive Vec growth.
        let mut entries: Vec<(usize, Slot)> = Vec::new();
        let mut remaining = MAX_DERIVATIVE_VISITS;
        for (source, needed) in needed.into_iter().enumerate() {
            if !needed {
                partials.push(SparseDerivative {
                    default: zero,
                    overrides: 0..0,
                });
                continue;
            }
            let instruction = self.plan.instructions[source];
            let children = instruction
                .derivative_inputs()
                .map(|child| child.map(|i| &partials[i]));
            let defaults = children.map(|child| child.map_or(zero, |sparse| sparse.default));
            charge(&mut remaining, expr)?;
            let default = self.derive(instruction, defaults, zero)?;
            let start = entries.len();
            if let Instruction::Variable(variable) = instruction {
                entries.push((variable, self.constant(1.)?));
            } else {
                let [mut left, mut right] =
                    children.map(|child| child.map_or(0..0, |sparse| sparse.overrides.clone()));
                while !left.is_empty() || !right.is_empty() {
                    let a = peek_override(&entries, &left);
                    let b = peek_override(&entries, &right);
                    let variable = match (a, b) {
                        (Some(a), Some(b)) => a.0.min(b.0),
                        (Some(a), None) => a.0,
                        (None, Some(b)) => b.0,
                        (None, None) => break,
                    };
                    let a = take_override(a, &mut left, variable, defaults[0]);
                    let b = take_override(b, &mut right, variable, defaults[1]);
                    charge(&mut remaining, expr)?;
                    let derivative = self.derive(instruction, [a, b], zero)?;
                    if derivative != default {
                        entries.push((variable, derivative));
                    }
                }
            }
            partials.push(SparseDerivative {
                default,
                overrides: start..entries.len(),
            });
        }
        let result = &partials[root];
        let mut derivatives = vec![result.default; quantities];
        for &(variable, slot) in &entries[result.overrides.clone()] {
            derivatives[variable] = slot;
        }
        Ok(derivatives)
    }

    /// Remove intermediate operations discarded by C's simplifications
    /// and restore value-first, gradient-order postorder construction.
    /// In particular, strict evaluation must not execute an unreachable
    /// nonfinite constant or domain error created during differentiation.
    pub(in crate::devices::behavioural::program) fn finish_reachable(
        self,
        root: Slot,
        mut derivatives: Vec<Slot>,
    ) -> (Plan, Slot, Vec<Slot>) {
        let mut map = vec![usize::MAX; self.plan.instructions.len()];
        let mut instructions = Vec::new();
        let mut pending = Vec::new();
        for root in core::iter::once(root).chain(derivatives.iter().copied()) {
            pending.push(root);
            while let Some(&slot) = pending.last() {
                if map[slot] == usize::MAX {
                    let instruction = self.plan.instructions[slot];
                    if let Some(child) = instruction
                        .inputs()
                        .into_iter()
                        .flatten()
                        .find(|&i| map[i] == usize::MAX)
                    {
                        pending.push(child);
                        continue;
                    }
                    map[slot] = instructions.len();
                    instructions.push(instruction.remap(&map));
                }
                pending.pop();
            }
        }
        let root = map[root];
        for derivative in &mut derivatives {
            *derivative = map[*derivative];
        }
        let has_ternary = instructions
            .iter()
            .any(|op| matches!(op, Instruction::Ternary(..)));
        (
            Plan {
                instructions,
                has_ternary,
                pwl: self.plan.pwl,
                tables: self.plan.tables,
            },
            root,
            derivatives,
        )
    }

    fn literal(&self, slot: Slot) -> Option<Real> {
        if let Instruction::Constant(bits) = self.plan.instructions[slot] {
            Some(Real::from_bits(bits))
        } else {
            None
        }
    }

    fn constant(&mut self, value: Real) -> SpiceResult<Slot> {
        self.intern(Instruction::Constant(value.to_bits()))
    }

    // C mkb folds constant pairs *before* zero/one identities.
    fn operator(&mut self, op: Operator, left: Slot, right: Slot) -> SpiceResult<Slot> {
        let (l, r) = (self.literal(left), self.literal(right));
        if let (Some(l), Some(r)) = (l, r) {
            return self.constant(match op {
                Operator::Times => l * r,
                Operator::Divide => l / r,
                Operator::Plus => l + r,
                Operator::Minus => l - r,
                Operator::Power => c_pow(l, r),
            });
        }
        match op {
            Operator::Times => {
                if l == Some(0.) {
                    return Ok(left);
                }
                if r == Some(0.) || l == Some(1.) {
                    return Ok(right);
                }
                if r == Some(1.) {
                    return Ok(left);
                }
            }
            Operator::Divide => {
                if l == Some(0.) || r == Some(1.) {
                    return Ok(left);
                }
            }
            Operator::Plus => {
                if l == Some(0.) {
                    return Ok(right);
                }
                if r == Some(0.) {
                    return Ok(left);
                }
            }
            Operator::Minus => {
                if r == Some(0.) {
                    return Ok(left);
                }
                if l == Some(0.) {
                    return self.unary(Unary::Uminus, right);
                }
            }
            Operator::Power => {
                if r == Some(0.) {
                    return self.constant(1.);
                }
                if r == Some(1.) {
                    return Ok(left);
                }
            }
        }
        self.intern(Instruction::Operator(op, left, right))
    }

    fn times(&mut self, a: Slot, b: Slot) -> SpiceResult<Slot> {
        self.operator(Operator::Times, a, b)
    }

    fn over(&mut self, a: Slot, b: Slot) -> SpiceResult<Slot> {
        self.operator(Operator::Divide, a, b)
    }

    fn squared(&mut self, a: Slot) -> SpiceResult<Slot> {
        let two = self.constant(2.)?;
        self.operator(Operator::Power, a, two)
    }

    fn unary(&mut self, op: Unary, a: Slot) -> SpiceResult<Slot> {
        if let Some(a) = self.literal(a) {
            self.constant(unary_value(op, a))
        } else {
            self.intern(Instruction::Unary(op, a))
        }
    }

    fn ternary(&mut self, condition: Slot, a: Slot, b: Slot) -> SpiceResult<Slot> {
        if let Some(condition) = self.literal(condition) {
            return Ok(if condition == 0. { b } else { a });
        }
        if let (Some(a_value), Some(b_value)) = (self.literal(a), self.literal(b))
            && a_value.partial_cmp(&b_value) == Some(core::cmp::Ordering::Equal)
        {
            return Ok(a);
        }
        self.intern(Instruction::Ternary(condition, a, b))
    }

    fn log_abs(&mut self, a: Slot) -> SpiceResult<Slot> {
        let abs = self.unary(Unary::Abs, a)?;
        self.unary(Unary::Log, abs)
    }

    // Ordinary/signed powers use opposite kernels for a constant exponent.
    // Binary function construction (C mkf over a comma) never folds pairs.
    fn power_derivative(
        &mut self,
        kernel: Binary,
        a: Slot,
        b: Slot,
        da: Slot,
        db: Slot,
    ) -> SpiceResult<Slot> {
        if let Some(exponent) = self.literal(b) {
            let shifted = self.constant(exponent - 1.)?;
            let inner = self.intern(Instruction::Binary(
                if kernel == Binary::Pow {
                    Binary::Pwr
                } else {
                    Binary::Pow
                },
                a,
                shifted,
            ))?;
            let outer = self.times(b, inner)?;
            return self.times(outer, da);
        }
        let outer = self.intern(Instruction::Binary(kernel, a, b))?;
        if kernel == Binary::Pow && self.literal(a).is_some() {
            let log = self.log_abs(a)?;
            let inner = self.times(db, log)?;
            return self.times(outer, inner);
        }
        let div = self.over(da, a)?;
        let left = self.times(b, div)?;
        let log = self.log_abs(a)?;
        let right = self.times(db, log)?;
        let inner = self.operator(Operator::Plus, left, right)?;
        self.times(outer, inner)
    }

    fn unary_slope(&mut self, op: Unary, u: Slot, zero: Slot) -> SpiceResult<Slot> {
        let one = self.constant(1.)?;
        match op {
            Unary::Abs => self.unary(Unary::Sgn, u),
            Unary::Acos | Unary::Asin | Unary::Acosh | Unary::Asinh => {
                let square = self.squared(u)?;
                let radicand = match op {
                    Unary::Acos | Unary::Asin => self.operator(Operator::Minus, one, square)?,
                    Unary::Acosh => self.operator(Operator::Minus, square, one)?,
                    _ => self.operator(Operator::Plus, square, one)?,
                };
                let denominator = self.unary(Unary::Sqrt, radicand)?;
                let numerator = if op == Unary::Acos {
                    self.constant(-1.)?
                } else {
                    one
                };
                self.over(numerator, denominator)
            }
            Unary::Atan | Unary::Atanh => {
                let square = self.squared(u)?;
                let denominator = if op == Unary::Atan {
                    self.operator(Operator::Plus, square, one)?
                } else {
                    self.operator(Operator::Minus, one, square)?
                };
                self.over(one, denominator)
            }
            Unary::Cos => {
                let sin = self.unary(Unary::Sin, u)?;
                self.unary(Unary::Uminus, sin)
            }
            Unary::Cosh => self.unary(Unary::Sinh, u),
            Unary::Exp => self.unary(Unary::Exp, u),
            Unary::Log => self.over(one, u),
            Unary::Log10 => {
                let numerator = self.constant(std::f64::consts::LOG10_E)?;
                self.over(numerator, u)
            }
            Unary::Sin => self.unary(Unary::Cos, u),
            Unary::Sinh => self.unary(Unary::Cosh, u),
            Unary::Sqrt => {
                let two = self.constant(2.)?;
                let sqrt = self.unary(Unary::Sqrt, u)?;
                let denominator = self.times(two, sqrt)?;
                self.over(one, denominator)
            }
            Unary::Tan | Unary::Tanh => {
                let tangent = self.unary(op, u)?;
                let square = self.squared(tangent)?;
                self.operator(
                    if op == Unary::Tan {
                        Operator::Plus
                    } else {
                        Operator::Minus
                    },
                    one,
                    square,
                )
            }
            Unary::Uramp => self.unary(Unary::Ustep, u),
            Unary::Ustep2 => {
                let a = self.unary(Unary::Ustep, u)?;
                let shifted = self.operator(Operator::Minus, u, one)?;
                let b = self.unary(Unary::Ustep, shifted)?;
                self.operator(Operator::Minus, a, b)
            }
            Unary::Uminus => self.constant(-1.),
            Unary::Sgn
            | Unary::Ustep
            | Unary::Eq0
            | Unary::Ne0
            | Unary::Gt0
            | Unary::Lt0
            | Unary::Ge0
            | Unary::Le0
            | Unary::Floor
            | Unary::Ceil
            | Unary::Nint => Ok(zero),
        }
    }

    fn derive(
        &mut self,
        instruction: Instruction,
        [da, db]: [Slot; 2],
        zero: Slot,
    ) -> SpiceResult<Slot> {
        match instruction {
            Instruction::Constant(_)
            | Instruction::Time
            | Instruction::Temperature
            | Instruction::Frequency
            | Instruction::Variable(_) => Ok(zero),
            Instruction::Operator(op @ (Operator::Plus | Operator::Minus), _, _) => {
                self.operator(op, da, db)
            }
            Instruction::Operator(op @ (Operator::Times | Operator::Divide), a, b) => {
                let left = self.times(da, b)?;
                let right = self.times(a, db)?;
                if op == Operator::Times {
                    self.operator(Operator::Plus, left, right)
                } else {
                    let numerator = self.operator(Operator::Minus, left, right)?;
                    let denominator = self.squared(b)?;
                    self.over(numerator, denominator)
                }
            }
            Instruction::Operator(Operator::Power, a, b)
            | Instruction::Binary(Binary::Pow, a, b) => {
                self.power_derivative(Binary::Pow, a, b, da, db)
            }
            Instruction::Binary(Binary::Pwr, a, b) => {
                self.power_derivative(Binary::Pwr, a, b, da, db)
            }
            Instruction::Ternary(c, ..) => self.ternary(c, da, db),
            Instruction::Binary(function @ (Binary::Min | Binary::Max), a, b) => {
                let difference = self.operator(Operator::Minus, a, b)?;
                let comparison = self.unary(
                    if function == Binary::Min {
                        Unary::Lt0
                    } else {
                        Unary::Gt0
                    },
                    difference,
                )?;
                self.ternary(comparison, da, db)
            }
            Instruction::Unary(op, u) => {
                let slope = self.unary_slope(op, u, zero)?;
                self.times(slope, da)
            }
            Instruction::Pwl(u, t) => {
                let slope = if let Some(x) = self.literal(u) {
                    self.constant(self.plan.pwl[t].slope(x))?
                } else {
                    self.intern(Instruction::PwlDerivative(u, t))?
                };
                self.times(slope, da)
            }
            Instruction::Table(u, t) => {
                let slope = if let Some(x) = self.literal(u) {
                    self.constant(self.plan.tables[t].evaluate(x).1)?
                } else {
                    self.intern(Instruction::TableSlope(u, t))?
                };
                self.times(slope, da)
            }
            Instruction::PwlDerivative(..) | Instruction::TableSlope(..) => self.times(zero, da),
        }
    }
}

fn peek_override(entries: &[(usize, Slot)], range: &Range<usize>) -> Option<(usize, Slot)> {
    (!range.is_empty()).then(|| entries[range.start])
}

fn take_override(
    entry: Option<(usize, Slot)>,
    range: &mut Range<usize>,
    variable: usize,
    default: Slot,
) -> Slot {
    if let Some((index, slot)) = entry
        && index == variable
    {
        range.start += 1;
        slot
    } else {
        default
    }
}

fn charge(remaining: &mut usize, expr: &BExpr) -> SpiceResult<()> {
    *remaining = remaining
        .checked_sub(1)
        .ok_or_else(|| unsupported(expr, "behavioural derivative expansion limit exceeded"))?;
    Ok(())
}
