//! The previous tree differentiator, retained only as a regression oracle.

use super::{Binary, Node, Operator, Real, Tree, Unary, c_pow, unary_value};
use std::sync::Arc;

thread_local! {
    static TREE_ALLOCATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn new_node(node: Node) -> Tree {
    TREE_ALLOCATIONS.with(|count| count.set(count.get() + 1));
    Arc::new(node)
}

pub(super) fn take_allocations() -> usize {
    TREE_ALLOCATIONS.with(|count| count.replace(0))
}

pub(super) fn constant(value: Real) -> Tree {
    new_node(Node::Constant(value))
}

/// `mkcon(0.0)` / `mkcon(1.0)` for a variable.
fn zero() -> Tree {
    constant(0.)
}

/// `mkb()` for the arithmetic operators: constant folding (plain C
/// arithmetic and `pow`, not the `PT*` functions) and the 0/1 identities.
fn mk_operator(op: Operator, left: Tree, right: Tree) -> Tree {
    if let (Some(l), Some(r)) = (left.constant(), right.constant()) {
        return constant(match op {
            Operator::Times => l * r,
            Operator::Divide => l / r,
            Operator::Plus => l + r,
            Operator::Minus => l - r,
            Operator::Power => c_pow(l, r),
        });
    }
    let (l, r) = (left.constant(), right.constant());
    match op {
        Operator::Times => {
            // C's order: a zero factor (either side) wins, then a unit one.
            if l == Some(0.) {
                return left;
            } else if r == Some(0.) || l == Some(1.) {
                return right;
            } else if r == Some(1.) {
                return left;
            }
        }
        Operator::Divide => {
            if l == Some(0.) || r == Some(1.) {
                return left;
            }
        }
        Operator::Plus => {
            if l == Some(0.) {
                return right;
            } else if r == Some(0.) {
                return left;
            }
        }
        Operator::Minus => {
            if r == Some(0.) {
                return left;
            } else if l == Some(0.) {
                return mk_unary(Unary::Uminus, right);
            }
        }
        Operator::Power => {
            if r == Some(0.) {
                return constant(1.);
            } else if r == Some(1.) {
                return left;
            }
        }
    }
    new_node(Node::Operator(op, left, right))
}

/// `mkb(PT_TERN, condition, mkb(PT_COMMA, then, otherwise))`.
fn mk_ternary(condition: Tree, then: Tree, otherwise: Tree) -> Tree {
    if let Some(condition) = condition.constant() {
        return if condition == 0. { otherwise } else { then };
    }
    if let (Some(a), Some(b)) = (then.constant(), otherwise.constant())
        && a.partial_cmp(&b) == Some(std::cmp::Ordering::Equal)
    {
        return then;
    }
    new_node(Node::Ternary(condition, then, otherwise))
}

/// `mkf()` for a one-argument function: a constant argument is folded.
fn mk_unary(function: Unary, argument: Tree) -> Tree {
    argument.constant().map_or_else(
        || new_node(Node::Unary(function, argument)),
        |x| constant(unary_value(function, x)),
    )
}

/// `mkf(PTF_POW/PTF_PWR, mkb(PT_COMMA, a, b))`: never folded (a comma node
/// is not a constant).
fn mk_binary(function: Binary, a: &Tree, b: Tree) -> Tree {
    new_node(Node::Binary(function, a.clone(), b))
}

/// `mkb(PT_TIMES, ...)` shorthand.
fn times(left: Tree, right: Tree) -> Tree {
    mk_operator(Operator::Times, left, right)
}

/// `mkb(PT_DIVIDE, ...)` shorthand.
fn over(left: Tree, right: Tree) -> Tree {
    mk_operator(Operator::Divide, left, right)
}

/// `mkb(PT_POWER, u, mkcon(2.0))`.
fn squared(u: &Tree) -> Tree {
    mk_operator(Operator::Power, u.clone(), constant(2.))
}

/// `mkf(PTF_LOG, mkf(PTF_ABS, a))`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PowerRule {
    Ordinary,
    Signed,
}

fn differentiate_power(rule: PowerRule, a: &Tree, b: &Tree, variable: usize) -> Tree {
    let (base, exponent) = match rule {
        PowerRule::Ordinary => (Binary::PowBaseSlope, Binary::PowExponentSlope),
        PowerRule::Signed => (Binary::PwrBaseSlope, Binary::PwrExponentSlope),
    };
    let left = times(mk_binary(base, a, b.clone()), differentiate(a, variable));
    let db = differentiate(b, variable);
    if db.constant() == Some(0.) {
        left
    } else {
        mk_operator(
            Operator::Plus,
            left,
            times(mk_binary(exponent, a, b.clone()), db),
        )
    }
}

/// The outer derivative of a unary function, constructed with C's tree
/// helpers. The chain rule is applied once by the main differentiator.
fn unary_slope(function: Unary, u: &Tree) -> Tree {
    let one = || constant(1.);
    match function {
        Unary::Abs => mk_unary(Unary::Sgn, u.clone()),
        Unary::Acos => over(
            constant(-1.),
            mk_unary(Unary::Sqrt, mk_operator(Operator::Minus, one(), squared(u))),
        ),
        Unary::Acosh => over(
            one(),
            mk_unary(Unary::Sqrt, mk_operator(Operator::Minus, squared(u), one())),
        ),
        Unary::Asin => over(
            one(),
            mk_unary(Unary::Sqrt, mk_operator(Operator::Minus, one(), squared(u))),
        ),
        Unary::Asinh => over(
            one(),
            mk_unary(Unary::Sqrt, mk_operator(Operator::Plus, squared(u), one())),
        ),
        Unary::Atan => over(one(), mk_operator(Operator::Plus, squared(u), one())),
        Unary::Atanh => over(one(), mk_operator(Operator::Minus, one(), squared(u))),
        Unary::Cos => mk_unary(Unary::Uminus, mk_unary(Unary::Sin, u.clone())),
        Unary::Cosh => mk_unary(Unary::Sinh, u.clone()),
        Unary::Exp => mk_unary(Unary::Exp, u.clone()),
        Unary::Log => over(one(), u.clone()),
        Unary::Log10 => over(constant(std::f64::consts::LOG10_E), u.clone()),
        Unary::Sin => mk_unary(Unary::Cos, u.clone()),
        Unary::Sinh => mk_unary(Unary::Cosh, u.clone()),
        Unary::Sqrt => over(one(), times(constant(2.), mk_unary(Unary::Sqrt, u.clone()))),
        Unary::Tan => mk_operator(
            Operator::Plus,
            one(),
            squared(&mk_unary(Unary::Tan, u.clone())),
        ),
        Unary::Tanh => mk_operator(
            Operator::Minus,
            one(),
            squared(&mk_unary(Unary::Tanh, u.clone())),
        ),
        Unary::Uramp => mk_unary(Unary::Ustep, u.clone()),
        Unary::Ustep2 => mk_operator(
            Operator::Minus,
            mk_unary(Unary::Ustep, u.clone()),
            mk_unary(Unary::Ustep, mk_operator(Operator::Minus, u.clone(), one())),
        ),
        Unary::Uminus => constant(-1.),
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
        | Unary::Nint => zero(),
    }
}

/// `PTdifferentiate` (`inpptree.c`): the derivative tree of `p` with respect
/// to quantity `variable`, built with C's `mkb`/`mkf` simplifications so its
/// evaluation is C's operation for operation. The PSPICE/HSPICE/LTspice
/// `newcompat` variants are not modelled.
pub(super) fn differentiate(p: &Tree, variable: usize) -> Tree {
    match &**p {
        Node::Constant(_) | Node::Time | Node::Temperature | Node::Frequency => zero(),
        Node::Variable(index) => constant(if *index == variable { 1. } else { 0. }),
        Node::Operator(op @ (Operator::Plus | Operator::Minus), a, b) => {
            mk_operator(*op, differentiate(a, variable), differentiate(b, variable))
        }
        // d(a * b) = d(a) * b + a * d(b)
        Node::Operator(Operator::Times, a, b) => mk_operator(
            Operator::Plus,
            times(differentiate(a, variable), b.clone()),
            times(a.clone(), differentiate(b, variable)),
        ),
        // d(a / b) = (d(a) * b - a * d(b)) / b^2
        Node::Operator(Operator::Divide, a, b) => over(
            mk_operator(
                Operator::Minus,
                times(differentiate(a, variable), b.clone()),
                times(a.clone(), differentiate(b, variable)),
            ),
            squared(b),
        ),
        Node::Operator(Operator::Power, a, b) | Node::Binary(Binary::Pow, a, b) => {
            differentiate_power(PowerRule::Ordinary, a, b, variable)
        }
        Node::Ternary(condition, then, otherwise) => mk_ternary(
            condition.clone(),
            differentiate(then, variable),
            differentiate(otherwise, variable),
        ),
        Node::Binary(function @ (Binary::Min | Binary::Max), a, b) => {
            // min(a,b)' = lt0(a-b) ? a' : b'; max uses gt0.
            let comparison = if *function == Binary::Min {
                Unary::Lt0
            } else {
                Unary::Gt0
            };
            mk_ternary(
                mk_unary(
                    comparison,
                    mk_operator(Operator::Minus, a.clone(), b.clone()),
                ),
                differentiate(a, variable),
                differentiate(b, variable),
            )
        }
        Node::Binary(Binary::Pwr, a, b) => differentiate_power(PowerRule::Signed, a, b, variable),
        Node::Binary(
            Binary::PowBaseSlope
            | Binary::PwrBaseSlope
            | Binary::PowExponentSlope
            | Binary::PwrExponentSlope,
            ..,
        ) => panic!("derivative instruction is not a source expression"),
        Node::Unary(function, u) => times(unary_slope(*function, u), differentiate(u, variable)),
        Node::Pwl(u, table) => {
            let slope = u.constant().map_or_else(
                || new_node(Node::PwlDerivative(Arc::clone(u), Arc::clone(table))),
                |x| constant(table.slope(x)),
            );
            times(slope, differentiate(u, variable))
        }
        Node::Table(u, map) => {
            let slope = u.constant().map_or_else(
                || new_node(Node::TableSlope(Arc::clone(u), Arc::clone(map))),
                |x| constant(map.evaluate(x).1),
            );
            times(slope, differentiate(u, variable))
        }
        Node::PwlDerivative(u, _) | Node::TableSlope(u, _) => {
            times(zero(), differentiate(u, variable))
        }
    }
}
