//! Winnow grammar for bounded numparam expressions over `&str`.
//!
//! Follows `src/frontend/numparam/xpressn.c` (`formula()`, `fetchoperator()`,
//! `fetchnumber()`, `fetchid()`, `fmathS`): `^`/`**` binds tighter than `*` `/`,
//! which bind tighter than `+` `-`, all left associative (`2^3^2` is 64 in C).
//! A sign at the start of an expression, group or argument behaves like the
//! additive operator (`-2^2` is -4); after a binary operator C only accepts
//! `-` before a numeric literal. All binary levels use incremental left folds.
//!
//! Every failure after a recognised prefix is a committed error carrying a
//! byte column. Nothing is evaluated.

use crate::primitives::{SourceLoc, SpiceError, SpiceResult, parse_numparam_number_prefix};
use winnow::Parser as _;
use winnow::combinator::{opt, peek, separated};
use winnow::error::{AddContext, ErrMode, ModalResult, ParserError};
use winnow::stream::{Stateful, Stream};
use winnow::token::{any, literal, one_of, take, take_while};

use crate::netlist::expr::{
    BinaryOp, EXCLUDED_FUNCTIONS, Expr, ExprKind, Function, MAX_NESTING, MAX_TREE_DEPTH,
    MAX_TREE_NODES, ParameterExpression, SourceSpan, UnaryOp,
};
use crate::netlist::source::SourcePosition;
use crate::netlist::token::{Token, TokenKind};

pub(super) const C_REFERENCE: &str = "src/frontend/numparam/xpressn.c";

/// Position context, syntactic nesting, and remaining expression-tree budget.
#[derive(Debug)]
pub(super) struct Ctx<'a> {
    pub origin: SourcePosition<'a>,
    /// Byte length of the parsed text.
    pub total: usize,
    pub depth: usize,
    pub remaining_nodes: usize,
    /// `.func` formals named like a built-in function: a bare use binds to
    /// the formal, a call is an error (as in C's numparam). Empty elsewhere.
    pub shadowing: &'a [String],
    /// Control vectors and variables may share names with callable functions.
    pub control_names: bool,
}

impl Ctx<'_> {
    /// Charge a node before constructing its parent or allocating child boxes.
    pub(super) fn node<T>(
        &mut self,
        at: usize,
        child_height: usize,
        build: impl FnOnce() -> T,
    ) -> Res<Built<T>> {
        if child_height >= MAX_TREE_DEPTH {
            return Err(cut(
                at,
                format!("expression structural depth limit ({MAX_TREE_DEPTH}) exceeded"),
            ));
        }
        self.remaining_nodes = self.remaining_nodes.checked_sub(1).ok_or_else(|| {
            cut(
                at,
                format!("expression node limit ({MAX_TREE_NODES}) exceeded"),
            )
        })?;
        Ok(Built {
            node: build(),
            height: child_height + 1,
        })
    }

    pub(super) fn location(&self, remaining: usize) -> SourceLoc {
        self.position(remaining).location()
    }

    pub(super) const fn position(&self, remaining: usize) -> SourcePosition<'_> {
        self.origin.advance(self.total - remaining)
    }
}

/// Parser-only bookkeeping; the public AST has no redundant cached metadata.
pub(super) struct Built<T> {
    pub node: T,
    pub height: usize,
}

pub(super) type In<'a> = Stateful<&'a str, Ctx<'a>>;
pub(super) type Res<T> = ModalResult<T, Fail>;

/// Error carrying the remaining byte count where it occurred, so the entry
/// points can compute an absolute byte column.
#[derive(Debug)]
pub(super) struct Fail {
    pub remaining: usize,
    pub message: String,
    /// Valid numparam outside the bounded subset (`NotYetPorted`, not Parse).
    pub unsupported: bool,
    /// An already positioned error from a nested parse.
    pub resolved: Option<SpiceError>,
}

impl<I: Stream> ParserError<I> for Fail {
    type Inner = Self;

    fn from_input(input: &I) -> Self {
        Self {
            remaining: input.eof_offset(),
            message: "unexpected input".to_owned(),
            unsupported: false,
            resolved: None,
        }
    }

    fn into_inner(self) -> std::result::Result<Self::Inner, Self> {
        Ok(self)
    }
}

impl<I: Stream> AddContext<I, &'static str> for Fail {
    fn add_context(mut self, _input: &I, _start: &I::Checkpoint, expected: &'static str) -> Self {
        self.message = format!("expected {expected}");
        self
    }
}

pub(super) fn cut(remaining: usize, message: impl Into<String>) -> ErrMode<Fail> {
    ErrMode::Cut(Fail {
        remaining,
        message: message.into(),
        unsupported: false,
        resolved: None,
    })
}

pub(super) fn cut_unsupported(remaining: usize, message: impl Into<String>) -> ErrMode<Fail> {
    ErrMode::Cut(Fail {
        remaining,
        message: message.into(),
        unsupported: true,
        resolved: None,
    })
}

/// Converts a grammar failure to the port's error type.
pub(super) fn into_error(origin: SourcePosition<'_>, total: usize, fail: Fail) -> SpiceError {
    if let Some(error) = fail.resolved {
        return error;
    }
    let location = Ctx {
        origin,
        total,
        depth: 0,
        remaining_nodes: MAX_TREE_NODES,
        shadowing: &[],
        control_names: false,
    }
    .location(fail.remaining);
    if fail.unsupported {
        SpiceError::not_yet_ported(format!("{location}: {}", fail.message), C_REFERENCE)
    } else {
        SpiceError::parse(location, fail.message)
    }
}

pub(super) fn ws(input: &mut In<'_>) -> Res<()> {
    take_while(0.., |c: char| c.is_ascii_whitespace())
        .void()
        .parse_next(input)
}

pub(super) const fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_'
}

pub(super) const fn is_ident_continue(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.'
}

fn span(input: &In<'_>, start: usize, end: usize) -> SourceSpan {
    SourceSpan::from_positions(input.state.position(start), input.state.position(end))
}

/// Describes the character that cannot start/continue an expression here.
fn describe(c: char) -> String {
    match c {
        '%' | '\\' | '!' | '<' | '>' | '=' | '&' | '|' | '?' | ':' => format!(
            "operator '{c}' is outside the bounded subset (only + - * / ^ ** are parsed; \
             comparison, logical, ternary, '%' and '\\' operators are not)"
        ),
        '\'' | '"' => "quoted text inside an expression is outside the bounded subset \
                        (single quotes may only delimit a whole value, like braces)"
            .to_owned(),
        '{' => "nested '{' is not supported inside an expression".to_owned(),
        '}' => "unmatched '}'".to_owned(),
        ')' => "unmatched ')'".to_owned(),
        other => format!("unexpected '{other}'"),
    }
}

const fn is_unsupported(c: char) -> bool {
    matches!(
        c,
        '%' | '\\' | '!' | '<' | '>' | '=' | '&' | '|' | '?' | ':' | '\'' | '"'
    )
}

fn unexpected(remaining: usize, c: char, context: &str) -> ErrMode<Fail> {
    let message = if matches!(c, ')' | '}' | '{') || is_unsupported(c) {
        describe(c)
    } else {
        format!("{} {context}", describe(c))
    };
    if is_unsupported(c) {
        cut_unsupported(remaining, message)
    } else {
        cut(remaining, message)
    }
}

fn enter(input: &mut In<'_>, remaining: usize) -> Res<()> {
    input.state.depth += 1;
    if input.state.depth > MAX_NESTING {
        return Err(cut(
            remaining,
            format!("expression nesting limit ({MAX_NESTING}) exceeded"),
        ));
    }
    Ok(())
}

const fn leave(input: &mut In<'_>) {
    input.state.depth = input.state.depth.saturating_sub(1);
}

/// `sum := [sign] term { ('+' | '-') term }`
fn sum(input: &mut In<'_>) -> Res<Built<Expr>> {
    ws.parse_next(input)?;
    let open = input.eof_offset();
    let lead = opt(one_of(['+', '-'])).parse_next(input)?;
    let first = term.parse_next(input)?;
    let first = match lead {
        Some(sign) => {
            let op = if sign == '-' {
                UnaryOp::Minus
            } else {
                UnaryOp::Plus
            };
            let span = SourceSpan {
                start: input.state.location(open),
                end: first.node.span.end.clone(),
                logical: input.state.position(open).logical_offset()..first.node.span.logical.end,
            };
            input.state.node(open, first.height, || Expr {
                kind: ExprKind::Unary {
                    op,
                    operand: Box::new(first.node),
                },
                span,
            })?
        }
        None => first,
    };
    let mut lhs = first;
    loop {
        ws(input)?;
        let at = input.eof_offset();
        let Some(op) = opt(additive).parse_next(input)? else {
            return Ok(lhs);
        };
        let rhs = term(input)?;
        lhs = binary(input, at, op, lhs, rhs)?;
    }
}

fn additive(input: &mut In<'_>) -> Res<BinaryOp> {
    ws.parse_next(input)?;
    one_of(['+', '-'])
        .map(|c| {
            if c == '+' {
                BinaryOp::Add
            } else {
                BinaryOp::Sub
            }
        })
        .parse_next(input)
}

fn binary(
    input: &mut In<'_>,
    at: usize,
    op: BinaryOp,
    lhs: Built<Expr>,
    rhs: Built<Expr>,
) -> Res<Built<Expr>> {
    let span = SourceSpan {
        start: lhs.node.span.start.clone(),
        end: rhs.node.span.end.clone(),
        logical: lhs.node.span.logical.start..rhs.node.span.logical.end,
    };
    input.state.node(at, lhs.height.max(rhs.height), || Expr {
        kind: ExprKind::Binary {
            op,
            lhs: Box::new(lhs.node),
            rhs: Box::new(rhs.node),
        },
        span,
    })
}

/// Multiplication/division and power are separate left folds. No reassociation.
fn term(input: &mut In<'_>) -> Res<Built<Expr>> {
    let mut lhs = power(input)?;
    loop {
        ws(input)?;
        let at = input.eof_offset();
        let op = if input.input.starts_with("**") {
            return Ok(lhs);
        } else if input.input.starts_with('*') {
            BinaryOp::Mul
        } else if input.input.starts_with('/') {
            BinaryOp::Div
        } else {
            return Ok(lhs);
        };
        take(1usize).void().parse_next(input)?;
        let rhs = power(input)?;
        lhs = binary(input, at, op, lhs, rhs)?;
    }
}

fn power(input: &mut In<'_>) -> Res<Built<Expr>> {
    let mut lhs = operand(input)?;
    loop {
        ws(input)?;
        let at = input.eof_offset();
        let width = if input.input.starts_with("**") {
            2usize
        } else if input.input.starts_with('^') {
            1
        } else {
            return Ok(lhs);
        };
        take(width).void().parse_next(input)?;
        let rhs = operand(input)?;
        lhs = binary(input, at, BinaryOp::Pow, lhs, rhs)?;
    }
}

/// Every operand position is mandatory, so failures here are committed.
fn operand(input: &mut In<'_>) -> Res<Built<Expr>> {
    ws.parse_next(input)?;
    let start = input.eof_offset();
    match peek(opt(any)).parse_next(input)? {
        None => Err(cut(
            start,
            "expected an operand, found the end of the expression",
        )),
        Some('(') => group(input),
        Some(c) if c.is_ascii_digit() || c == '.' => number(input),
        Some('-') => signed_number(input),
        Some(c) if is_ident_start(c) => identifier_or_call(input),
        Some(')') => Err(cut(start, "expected an operand, found ')'")),
        Some(c) => Err(unexpected(start, c, "where an operand is required")),
    }
}

fn number(input: &mut In<'_>) -> Res<Built<Expr>> {
    let start = input.eof_offset();
    let before = input.input;
    // numparam's fetchnumber(): %lG plus parseunit(), not INPevaluate.
    let Some(parsed) = parse_numparam_number_prefix(before) else {
        return Err(cut(start, "malformed numeric literal"));
    };
    let digits = input.input[..parsed.consumed].chars().count();
    take(digits).void().parse_next(input)?;
    // fetchnumber() swallows the unit letters after the scale factor.
    take_while(0.., |c: char| c.is_ascii_alphabetic())
        .void()
        .parse_next(input)?;
    let end = input.eof_offset();
    let spelling = &before[..before.len() - input.input.len()];
    if !parsed.value.is_finite() {
        return Err(cut(
            start,
            format!("numeric literal '{spelling}' overflows to a non-finite value"),
        ));
    }
    let span = span(input, start, end);
    input.state.node(start, 0, || Expr {
        kind: ExprKind::Number {
            value: parsed.value,
            spelling: spelling.to_owned(),
        },
        span,
    })
}

/// `-` directly before a numeric literal, valid after a binary operator
/// (xpressn.c's `negate` flag, which only `fetchnumber()` consumes).
fn signed_number(input: &mut In<'_>) -> Res<Built<Expr>> {
    let start = input.eof_offset();
    literal("-").void().parse_next(input)?;
    ws.parse_next(input)?;
    let here = input.eof_offset();
    match peek(opt(any)).parse_next(input)? {
        Some(c) if c.is_ascii_digit() || c == '.' => {}
        _ => {
            return Err(cut(
                here,
                "a second sign is accepted only directly before a numeric literal \
                 (C numparam rejects `-` before names, groups and calls here)",
            ));
        }
    }
    let literal_value = number(input)?;
    let span = SourceSpan {
        start: input.state.location(start),
        end: literal_value.node.span.end.clone(),
        logical: input.state.position(start).logical_offset()..literal_value.node.span.logical.end,
    };
    input.state.node(start, literal_value.height, || Expr {
        kind: ExprKind::Unary {
            op: UnaryOp::Minus,
            operand: Box::new(literal_value.node),
        },
        span,
    })
}

fn group(input: &mut In<'_>) -> Res<Built<Expr>> {
    let start = input.eof_offset();
    literal("(").void().parse_next(input)?;
    enter(input, start)?;
    let inner = sum.parse_next(input)?;
    close(input, start)?;
    leave(input);
    let span = span(input, start, input.eof_offset());
    input.state.node(start, inner.height, || Expr {
        kind: ExprKind::Group(Box::new(inner.node)),
        span,
    })
}

fn close(input: &mut In<'_>, open: usize) -> Res<()> {
    ws.parse_next(input)?;
    let here = input.eof_offset();
    match opt(literal(")")).parse_next(input)? {
        Some(_) => Ok(()),
        None => match peek(opt(any)).parse_next(input)? {
            None => Err(cut(
                here,
                format!(
                    "expected ')' to close the '(' at column {}",
                    input.state.location(open).column
                ),
            )),
            Some(c) => Err(unexpected(
                here,
                c,
                "after a complete expression (missing operator?)",
            )),
        },
    }
}

/// Whether `name` (lower case) is a built-in numparam function name, allowlisted
/// or excluded.
pub(super) fn is_builtin_name(name: &str) -> bool {
    Function::from_name(name).is_some() || EXCLUDED_FUNCTIONS.contains(&name)
}

fn identifier_or_call(input: &mut In<'_>) -> Res<Built<Expr>> {
    let start = input.eof_offset();
    let name: &str = (one_of(is_ident_start), take_while(0.., is_ident_continue))
        .take()
        .parse_next(input)?;
    let after = input.eof_offset();
    let call = peek((ws, opt(literal("(")))).parse_next(input)?.1.is_some();
    let lowered = name.to_ascii_lowercase();
    if input.state.shadowing.contains(&lowered) {
        if call {
            // C's numparam substitutes the formal into the call and the
            // deck fails ("Formula() error"; checked against the C binary).
            return Err(cut(
                start,
                format!(
                    "'{name}' is a parameter of this function, so the built-in of that \
                     name cannot be called in its body"
                ),
            ));
        }
        let span = span(input, start, after);
        return input.state.node(start, 0, || Expr {
            kind: ExprKind::Identifier(lowered),
            span,
        });
    }
    if !call {
        if !input.state.control_names && is_builtin_name(&lowered) {
            return Err(cut(
                start,
                format!("function name '{name}' requires an argument list"),
            ));
        }
        let span = span(input, start, after);
        return input.state.node(start, 0, || Expr {
            kind: ExprKind::Identifier(lowered),
            span,
        });
    }
    let Some(function) = Function::from_name(&lowered) else {
        return user_call(input, start, lowered);
    };
    ws.parse_next(input)?;
    let open = input.eof_offset();
    literal("(").void().parse_next(input)?;
    enter(input, open)?;
    let arguments: Vec<Built<Expr>> = separated(1.., sum, (ws, literal(","))).parse_next(input)?;
    close(input, open)?;
    leave(input);
    if arguments.len() != function.arity() {
        return Err(cut(
            start,
            format!(
                "function '{}' takes {} argument(s), found {}",
                function.name(),
                function.arity(),
                arguments.len()
            ),
        ));
    }
    let height = arguments.iter().map(|arg| arg.height).max().unwrap_or(0);
    let span = span(input, start, input.eof_offset());
    input.state.node(start, height, || Expr {
        kind: ExprKind::Call {
            function,
            arguments: arguments.into_iter().map(|arg| arg.node).collect(),
        },
        span,
    })
}

/// `name(args)` for a name outside the allowlist: a user `.func` call
/// (resolved during evaluation). Zero arguments are allowed (`f()`).
fn user_call(input: &mut In<'_>, start: usize, name: String) -> Res<Built<Expr>> {
    ws.parse_next(input)?;
    let open = input.eof_offset();
    literal("(").void().parse_next(input)?;
    enter(input, open)?;
    let empty = peek((ws, opt(literal(")")))).parse_next(input)?.1.is_some();
    let arguments: Vec<Built<Expr>> = if empty {
        Vec::new()
    } else {
        separated(1.., sum, (ws, literal(","))).parse_next(input)?
    };
    close(input, open)?;
    leave(input);
    let height = arguments.iter().map(|arg| arg.height).max().unwrap_or(0);
    let span = span(input, start, input.eof_offset());
    input.state.node(start, height, || Expr {
        kind: ExprKind::UserCall {
            name,
            arguments: arguments.into_iter().map(|arg| arg.node).collect(),
        },
        span,
    })
}

/// Parses `text` (the content of a `{...}` expression, or an unbraced `.param`
/// value / bare name) with the supplied logical-to-physical origin.
///
/// # Errors
///
/// A committed [`SpiceError::Parse`] (malformed, with the byte column of the
/// offending text) or [`SpiceError::NotYetPorted`] (valid numparam outside the
/// bounded subset).
pub(super) fn parse_expression(
    text: &str,
    origin: SourcePosition<'_>,
    braced: bool,
) -> SpiceResult<ParameterExpression> {
    parse_delimited(text, origin, braced, false)
}

/// As [`parse_expression`], recording whether the delimiters were quotes.
pub(super) fn parse_delimited(
    text: &str,
    origin: SourcePosition<'_>,
    braced: bool,
    quoted: bool,
) -> SpiceResult<ParameterExpression> {
    parse_func_body(text, origin, braced, quoted, &[])
}

/// As [`parse_delimited`] for a `.func` body whose formals in `shadowing`
/// reuse built-in function names.
pub(super) fn parse_func_body(
    text: &str,
    origin: SourcePosition<'_>,
    braced: bool,
    quoted: bool,
    shadowing: &[String],
) -> SpiceResult<ParameterExpression> {
    parse_with_budget(text, origin, braced, quoted, shadowing, MAX_TREE_NODES)
        .map(|(expression, _, _)| expression)
}

/// Parse an embedded value with the enclosing expression's remaining budget.
pub(super) fn parse_with_budget(
    text: &str,
    origin: SourcePosition<'_>,
    braced: bool,
    quoted: bool,
    shadowing: &[String],
    remaining_nodes: usize,
) -> SpiceResult<(ParameterExpression, usize, usize)> {
    parse_with_names(
        text,
        origin,
        braced,
        quoted,
        shadowing,
        remaining_nodes,
        false,
    )
}

pub(super) fn parse_control_expression(
    text: &str,
    origin: SourcePosition<'_>,
) -> SpiceResult<ParameterExpression> {
    parse_with_names(text, origin, false, false, &[], MAX_TREE_NODES, true)
        .map(|(expression, _, _)| expression)
}

fn parse_with_names(
    text: &str,
    origin: SourcePosition<'_>,
    braced: bool,
    quoted: bool,
    shadowing: &[String],
    remaining_nodes: usize,
    control_names: bool,
) -> SpiceResult<(ParameterExpression, usize, usize)> {
    let mut input = In {
        input: text,
        state: Ctx {
            origin,
            total: text.len(),
            depth: 0,
            remaining_nodes,
            shadowing,
            control_names,
        },
    };
    let run = |input: &mut In<'_>| -> Res<Built<Expr>> {
        let root = sum(input)?;
        ws.parse_next(input)?;
        let here = input.eof_offset();
        if let Some(c) = peek(opt(any)).parse_next(input)? {
            return Err(unexpected(
                here,
                c,
                "after a complete expression (missing operator?)",
            ));
        }
        Ok(root)
    };
    if text.trim().is_empty() {
        return Err(SpiceError::parse(origin.location(), "empty expression"));
    }
    match run(&mut input) {
        Ok(root) => Ok((
            ParameterExpression {
                text: text.to_owned(),
                braced: braced || quoted,
                quoted,
                span: SourceSpan::from_positions(origin, origin.advance(text.len())),
                root: root.node,
            },
            root.height,
            input.state.remaining_nodes,
        )),
        Err(ErrMode::Backtrack(fail) | ErrMode::Cut(fail)) => {
            Err(into_error(origin, text.len(), fail))
        }
        Err(ErrMode::Incomplete(_)) => unreachable!("complete input"),
    }
}

/// True for a token that holds a delimited expression: a `{...}` token, or a
/// single-quoted `'...'` token. C's `inp_change_quotes()` (`inpcom.c`) turns
/// every single-quote pair into a brace pair before numparam runs, so both
/// spellings mean the same expression. Double-quoted strings stay strings.
pub(super) fn is_expression_token(token: &Token) -> bool {
    match token.kind {
        TokenKind::Expression(_) => true,
        TokenKind::Quoted(_) => token.text.starts_with('\''),
        _ => false,
    }
}

/// Parses the text of a `{...}` token (tokenizer-matched braces) or of a
/// single-quoted `'...'` token (see [`is_expression_token`]).
///
/// # Errors
///
/// As [`parse_expression`].
pub(super) fn from_brace_token(token: &Token) -> SpiceResult<ParameterExpression> {
    if let Some(inner) = token
        .text
        .strip_prefix('\'')
        .and_then(|rest| rest.strip_suffix('\''))
        .filter(|_| token.text.len() >= 2)
    {
        return parse_delimited(inner, token.position().advance(1), true, true);
    }
    let inner = token
        .text
        .strip_prefix('{')
        .and_then(|rest| rest.strip_suffix('}'))
        .unwrap_or(&token.text);
    parse_expression(inner, token.position().advance(1), true)
}

/// Parses a token that is a bare identifier, as an unbraced expression.
pub(super) fn from_name_token(token: &Token) -> SpiceResult<ParameterExpression> {
    parse_expression(&token.text, token.position(), false)
}
