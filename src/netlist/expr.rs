//! Parameter-expression syntax tree (bounded numparam subset).
//!
//! These types describe **syntax only**. Nothing here is evaluated, resolved or
//! range-checked: a later evaluation pass (GitHub #15) walks [`Expr`] with its
//! own environment. The grammar and its limits are documented in
//! `docs/port/PARAM_EXPRESSIONS.md`; the C behaviour it follows lives in
//! `src/frontend/numparam/xpressn.c` (`formula()`, `fetchoperator()`,
//! `fmathS`) and the `.param` line splitting in `inpcom.c`
//! (`inp_split_multi_param_lines()`).
//!
//! # Accepted syntax
//!
//! ```text
//! sum     := [sign] term { ('+' | '-') term }
//! term    := factor { ('*' | '/') factor }          (left associative)
//! factor  := atom { ('^' | '**') atom }             (left associative, as in C)
//! atom    := number | '-' number | identifier | call | '(' sum ')'
//! call    := function '(' sum { ',' sum } ')'
//!          | name '(' [ sum { ',' sum } ] ')'     (user `.func`, see below)
//! ```
//!
//! A call to a name outside the [`Function`] allowlist is an
//! [`ExprKind::UserCall`]; whether a `.func` defines it is decided during
//! evaluation ([`crate::netlist::eval::FunctionScope`]).
//!
//! Precedence, tightest first: `^`/`**`, then `*` `/`, then `+` `-`. A sign at
//! the start of an expression, group or argument binds like C's
//! additive-level operator (`-2^2` is `-(2^2)`, `-a*b+1` is `(-(a*b))+1`).
//! After a binary operator C accepts only `-` directly before a numeric
//! literal, which then binds tightest (`2*-3^2` is `2*((-3)^2)`); the parser
//! rejects the other C-erroring forms (`2*+3`, `2*-a`, `2*-(3)`).

use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};

/// A half-open source span.
///
/// Physical endpoints can be on different lines; the separate logical byte
/// offsets address the retained spelling in the parser input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceSpan {
    /// First byte of the spanned text.
    pub start: SourceLoc,
    /// Physical location just past the last byte.
    pub end: SourceLoc,
    /// Half-open byte range in the parser's shared logical input.
    pub logical: std::ops::Range<usize>,
}

impl SourceSpan {
    pub(crate) fn from_positions(
        start: crate::netlist::source::SourcePosition<'_>,
        end: crate::netlist::source::SourcePosition<'_>,
    ) -> Self {
        Self {
            start: start.location(),
            end: end.location(),
            logical: start.logical_offset()..end.logical_offset(),
        }
    }

    /// The byte range relative to the containing span, for slicing its text.
    #[must_use]
    pub fn relative_to(&self, container: &Self) -> Option<std::ops::Range<usize>> {
        let start = self.logical.start.checked_sub(container.logical.start)?;
        let end = self.logical.end.checked_sub(container.logical.start)?;
        (end <= container.len()).then_some(start..end)
    }

    /// Span length in bytes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.logical.end - self.logical.start
    }

    /// True when the span covers no bytes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A parsed expression together with the text it was written as.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterExpression {
    /// The expression text exactly as written: without the surrounding braces
    /// when [`braced`](Self::braced), including any inner whitespace.
    pub text: String,
    /// True for a `{...}` or `'...'` expression, false for an unbraced
    /// `.param` value or a bare name at an `X`/`.subckt` parameter site.
    pub braced: bool,
    /// True when the delimiters were single quotes (`'...'`) rather than braces.
    /// C's `inpcom.c` `inp_change_quotes()` rewrites every quote pair to a
    /// brace pair before numparam runs, so a quoted expression means exactly
    /// the same as the braced one; the flag only preserves the spelling for
    /// the writer and dumps. Implies [`braced`](Self::braced).
    pub quoted: bool,
    /// Span of [`text`](Self::text), excluding enclosing delimiters.
    pub span: SourceSpan,
    /// The syntax tree. `root.span` excludes surrounding whitespace.
    pub root: Expr,
}

impl ParameterExpression {
    /// The expression as written, delimiters included: `{text}`, `'text'` or
    /// the bare text.
    #[must_use]
    pub fn spelling(&self) -> String {
        if self.quoted {
            format!("'{}'", self.text)
        } else if self.braced {
            format!("{{{}}}", self.text)
        } else {
            self.text.clone()
        }
    }

    /// Names referenced by the expression, lowercased, in source order with
    /// duplicates kept. Function names are not references.
    #[must_use]
    pub fn references(&self) -> Vec<&str> {
        self.root
            .preorder()
            .filter_map(|node| match &node.kind {
                ExprKind::Identifier(name) => Some(name.as_str()),
                _ => None,
            })
            .collect()
    }
}

/// One syntax-tree node with its byte span.
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    /// What the node is.
    pub kind: ExprKind,
    /// The text the node covers (parentheses of a [`ExprKind::Group`] and a
    /// signed literal's sign included).
    pub span: SourceSpan,
}

impl Expr {
    /// Measure an existing subtree against the remaining construction budget.
    /// Used before copying it into a generated behavioural expression.
    pub(super) fn check_tree(&self, depth: usize, budget: &mut TreeBudget) -> SpiceResult<usize> {
        budget.charge(depth, 1, &self.span.start)?;
        let mut height = depth;
        let mut child = |expr: &Self| -> SpiceResult<()> {
            height = height.max(expr.check_tree(depth + 1, budget)?);
            Ok(())
        };
        match &self.kind {
            ExprKind::Number { .. } | ExprKind::Identifier(_) => {}
            ExprKind::Unary { operand, .. } | ExprKind::Group(operand) => child(operand)?,
            ExprKind::Binary { lhs, rhs, .. } => {
                child(lhs)?;
                child(rhs)?;
            }
            ExprKind::Call { arguments, .. } | ExprKind::UserCall { arguments, .. } => {
                for argument in arguments {
                    child(argument)?;
                }
            }
        }
        Ok(height)
    }

    /// Visit a syntax tree in source order without recursive stack growth.
    /// A binary node visits its left subtree before the right; a call is
    /// visited before its arguments.
    pub(super) fn preorder(&self) -> impl Iterator<Item = &Self> {
        let mut pending = vec![self];
        std::iter::from_fn(move || {
            let node = pending.pop()?;
            match &node.kind {
                ExprKind::Number { .. } | ExprKind::Identifier(_) => {}
                ExprKind::Unary { operand, .. } | ExprKind::Group(operand) => {
                    pending.push(operand);
                }
                ExprKind::Binary { lhs, rhs, .. } => {
                    pending.push(rhs);
                    pending.push(lhs);
                }
                ExprKind::Call { arguments, .. } | ExprKind::UserCall { arguments, .. } => {
                    pending.extend(arguments.iter().rev());
                }
            }
            Some(node)
        })
    }
}

/// Node shapes. Parsed trees obey both syntax nesting and structural limits.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    /// A finite numeric literal. `value` already includes the SPICE scale
    /// factor (`2.5meg` is `2.5e6`, `5V` is `5`); `spelling` is the original
    /// text. Overflowing spellings such as `1e999` are rejected, never stored.
    Number {
        /// Value with the scale factor applied.
        value: Real,
        /// Original spelling, including suffix and unit letters.
        spelling: String,
    },
    /// A parameter reference, lowercased (identifiers are case-insensitive).
    /// Never a function name.
    Identifier(String),
    /// A prefix sign.
    Unary {
        /// The sign.
        op: UnaryOp,
        /// Its operand.
        operand: Box<Expr>,
    },
    /// A binary arithmetic operation.
    Binary {
        /// The operator.
        op: BinaryOp,
        /// Left operand.
        lhs: Box<Expr>,
        /// Right operand.
        rhs: Box<Expr>,
    },
    /// An allowlisted function call; `arguments.len() == function.arity()`.
    Call {
        /// The function.
        function: Function,
        /// Arguments in source order.
        arguments: Vec<Expr>,
    },
    /// A parenthesised sub-expression, kept so spans and shape survive.
    Group(Box<Expr>),
    /// A call to a name outside the [`Function`] allowlist: a user `.func`
    /// (C: `inpcom.c` `inp_expand_macro_in_str()`), resolved during
    /// evaluation in the scope of the site. Any argument count (including
    /// zero) is syntactically valid; arity is checked against the definition.
    /// Names of numparam built-ins outside the allowlist (`agauss`, `limit`,
    /// ...) parse here too, so a `.func` may define them; without one they are
    /// reported as not yet ported when evaluated.
    UserCall {
        /// Lowercased function name.
        name: String,
        /// Arguments in source order.
        arguments: Vec<Expr>,
    },
}

/// Prefix signs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// `+x`
    Plus,
    /// `-x`
    Minus,
}

/// Binary arithmetic operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// `+`
    Add,
    /// `-`
    Sub,
    /// `*`
    Mul,
    /// `/`
    Div,
    /// `^` or `**` (both spellings are one operator in C). Left associative:
    /// `2^3^2` is `(2^3)^2 = 64` in C. C's default `^` is `pow(fabs(x), y)`
    /// (xpressn.c `operate()`), so an evaluator must not use plain `powf`.
    Pow,
}

/// Maximum parenthesis/call nesting accepted by the parser; deeper input is a
/// committed error instead of unbounded recursion.
pub const MAX_NESTING: usize = 64;

/// Maximum structural height (including leaves) of a parsed expression.
///
/// Flat left-associative chains count toward this limit, independently of
/// syntactic nesting. Matches the behavioural compiler's depth envelope.
pub const MAX_TREE_DEPTH: usize = 4 * MAX_NESTING;

/// Maximum number of syntax nodes in one expression, including embedded
/// numparam values. Wide calls are limited independently of tree height.
pub const MAX_TREE_NODES: usize = 100_000;

/// A cumulative node budget for generated trees. Depth is checked separately
/// because even a small left-deep tree can exhaust a recursive consumer.
pub(super) struct TreeBudget {
    remaining: usize,
}

impl TreeBudget {
    pub(super) const fn new(remaining: usize) -> Self {
        Self { remaining }
    }

    pub(super) fn charge(
        &mut self,
        depth: usize,
        nodes: usize,
        location: &SourceLoc,
    ) -> SpiceResult<()> {
        if depth > MAX_TREE_DEPTH {
            return Err(SpiceError::parse(
                location.clone(),
                format!("expression structural depth limit ({MAX_TREE_DEPTH}) exceeded"),
            ));
        }
        self.remaining = self.remaining.checked_sub(nodes).ok_or_else(|| {
            SpiceError::parse(
                location.clone(),
                format!("expression node limit ({MAX_TREE_NODES}) exceeded"),
            )
        })?;
        Ok(())
    }
}

macro_rules! functions {
    ($($variant:ident => $name:literal / $arity:literal,)*) => {
        /// The allowlisted numparam functions (a subset of `fmathS` in
        /// `xpressn.c`). Randomised functions (`agauss`, `gauss`, `unif`,
        /// `aunif`, `limit`), `ternary_fcn`, and the string-argument `vec`/`var`
        /// are deliberately excluded.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Function {
            $(
                #[doc = concat!("`", $name, "`")]
                $variant,
            )*
        }

        impl Function {
            /// Every allowlisted function, in `fmathS` order.
            pub const ALL: &'static [Function] = &[$(Self::$variant,)*];

            /// Looks a function up by (case-insensitive) name.
            #[must_use]
            pub fn from_name(name: &str) -> Option<Self> {
                match name.to_ascii_lowercase().as_str() {
                    $($name => Some(Self::$variant),)*
                    _ => None,
                }
            }

            /// Canonical lowercase name.
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)*
                }
            }

            /// Required argument count: 2 for `pow`/`pwr`/`max`/`min`, else 1.
            #[must_use]
            pub const fn arity(self) -> usize {
                match self {
                    $(Self::$variant => $arity,)*
                }
            }
        }
    };
}

functions! {
    Sqr => "sqr" / 1,
    Sqrt => "sqrt" / 1,
    Sin => "sin" / 1,
    Cos => "cos" / 1,
    Exp => "exp" / 1,
    Ln => "ln" / 1,
    Arctan => "arctan" / 1,
    Abs => "abs" / 1,
    Pow => "pow" / 2,
    Pwr => "pwr" / 2,
    Max => "max" / 2,
    Min => "min" / 2,
    Int => "int" / 1,
    Log => "log" / 1,
    Log10 => "log10" / 1,
    Sinh => "sinh" / 1,
    Cosh => "cosh" / 1,
    Tanh => "tanh" / 1,
    Sgn => "sgn" / 1,
    Ceil => "ceil" / 1,
    Floor => "floor" / 1,
    Asin => "asin" / 1,
    Acos => "acos" / 1,
    Atan => "atan" / 1,
    Asinh => "asinh" / 1,
    Acosh => "acosh" / 1,
    Atanh => "atanh" / 1,
    Tan => "tan" / 1,
    Nint => "nint" / 1,
}

/// C `fmathS` names outside the allowlist. A call to one parses as
/// [`ExprKind::UserCall`] and is rejected as not yet ported during evaluation
/// unless a `.func` of that name is in scope.
pub const EXCLUDED_FUNCTIONS: &[&str] = &[
    "ternary_fcn",
    "agauss",
    "gauss",
    "unif",
    "aunif",
    "limit",
    "vec",
    "var",
];
