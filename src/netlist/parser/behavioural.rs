//! Behavioural-source cards: `Bname n+ n- v=expr|i=expr [setters]`
//! (`src/spicelib/parser/inp2b.c`, `devices/asrc/asrc.c`) and the nonlinear
//! E/G/F/H forms that `src/frontend/inpcom.c` rewrites into B sources or
//! XSPICE code models before `INP2E`..`INP2H` run:
//!
//! ```text
//! Ename n+ n- value={expr} | vol={expr} [B setters]       (inp_compat)
//! Gname n+ n- value={expr} | cur={expr} [m=val]
//! Ename n+ n- table {expr} [=] (x0,y0) (x1,y1) ...        (XSPICE pwl)
//! Gname n+ n- table {expr} [=] (x0,y0) (x1,y1) ... [m=val]
//! Ename n+ n- nc+ nc- table=(x0, y0, x1, y1, ...)         (LTspice form)
//! Ename n+ n- poly(n) nc1+ nc1- ... c0 c1 c2 ...          (enhtrans.c)
//! Gname n+ n- poly(n) nc1+ nc1- ... c0 c1 ... [m=val]
//! Fname n+ n- poly(n) v1 ... c0 c1 ... [m=val]
//! Hname n+ n- poly(n) v1 ... c0 c1 ...
//! Ename n+ n- nc+ nc- c0 c1 ...    (implicit POLY(1), inp_poly_2g6_compat)
//! ```
//!
//! The B expression is parsed from the raw card text ([`super::bexpression`])
//! because C hands the B-source parser the text, not a token stream; setters
//! after it are ordinary tokens again.

use crate::primitives::SpiceError;
use winnow::Parser as _;
use winnow::combinator::cut_err;
use winnow::error::ErrMode;
use winnow::stream::Stream as _;
use winnow::token::any;

use crate::netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind, poly_dimension};
use crate::netlist::behavioural::forms::B_SETTERS;
use crate::netlist::bexpr::BehaviouralExpression;
use crate::netlist::token::{Token, TokenKind};

use super::grammar::{Failure, Input, ParsedCard, Result, location};
use super::syntax::{
    canonical_node, equals, malformed, named, named_device, skip_terminal_punctuation,
    terminal_name, value,
};

pub(super) fn behavioural_card(input: &mut Input<'_>) -> Result<ParsedCard> {
    let (instance, _) = named_device("b")
        .verify(|&(token, _): &(&Token, char)| token.text.len() > 1)
        .parse_next(input)?;
    cut_err(move |input: &mut Input<'_>| b_body(input, instance)).parse_next(input)
}

fn b_body(input: &mut Input<'_>, instance: &Token) -> Result<ParsedCard> {
    let auto_gnd = input.state.auto_gnd;
    let positive = terminal_name(input, "positive terminal")?;
    let negative = terminal_name(input, "negative terminal")?;
    let keyword = input.input.first();
    let output = match keyword {
        Some(token) if token.is_keyword("v") || token.is_keyword("i") => {
            token.text.to_ascii_lowercase()
        }
        _ => {
            return Err(malformed(
                input,
                "expected v=expression or i=expression (C: Bname n+ n- v=expr | i=expr)",
            ));
        }
    };
    let keyword = any.parse_next(input)?;
    let equal = equals
        .parse_next(input)
        .map_err(|_| malformed(input, "expected '=' after v or i"))?;
    let verbatim = is_verbatim(&input.state.card.raw);
    let expression = expression_after(input, equal, verbatim)?;
    let mut parameters = vec![behavioural(&output, keyword, expression)];
    parameters.extend(b_setters(input, &output)?);
    Ok(ParsedCard::Device(DeviceInstance {
        name: instance.text.to_ascii_lowercase(),
        designator: 'b',
        nodes: vec![
            canonical_node(positive, auto_gnd),
            canonical_node(negative, auto_gnd),
        ],
        model: None,
        parameters,
        location: input.state.card.location.clone(),
    }))
}

/// `inp_bsource_compat()` leaves a card alone when, after `inp_remove_ws()`,
/// it contains `=pwl(`; numparam then evaluates its `{...}` groups.
pub(super) fn is_verbatim(raw: &str) -> bool {
    super::controlled::remove_ws(raw)
        .to_ascii_lowercase()
        .contains("=pwl(")
}

/// The assignment holding a parsed behavioural expression.
pub(super) fn behavioural(
    name: &str,
    at: &Token,
    expression: BehaviouralExpression,
) -> ParameterAssignment {
    ParameterAssignment {
        name: name.to_owned(),
        value: expression.text.clone(),
        kind: ParameterKind::Behavioural(Box::new(expression)),
        location: at.location.clone(),
    }
}

/// Parses the raw card text after `after` (normally the `=` token) as a
/// behavioural expression and skips the tokens it covered.
pub(super) fn expression_after(
    input: &mut Input<'_>,
    after: &Token,
    verbatim: bool,
) -> Result<BehaviouralExpression> {
    let raw = &input.state.card.raw;
    let start = raw_offset(after) + after.text.len();
    let column = after.location.column + u32::try_from(after.text.len()).unwrap_or(u32::MAX);
    let (expression, consumed) = super::bexpression::parse_prefix(
        &raw[start.min(raw.len())..],
        &input.state.card.location,
        column,
        verbatim,
        input.state.auto_gnd,
    )
    .map_err(|error| ErrMode::Cut(Failure(error)))?;
    skip_to(input, start + consumed)?;
    Ok(expression)
}

/// Byte offset of a token in its card's joined text (`tokenize` columns are
/// one-based byte offsets).
const fn raw_offset(token: &Token) -> usize {
    token.location.column.saturating_sub(1) as usize
}

/// Consumes every token that starts before `end`; a token that starts before
/// `end` but extends past it was split by the expression and is an error.
fn skip_to(input: &mut Input<'_>, end: usize) -> Result<()> {
    while let Some(token) = input.input.first() {
        let start = raw_offset(token);
        if start >= end {
            break;
        }
        if start + token.text.len() > end {
            return Err(ErrMode::Cut(Failure(SpiceError::parse(
                token
                    .location
                    .at_column(u32::try_from(end + 1).unwrap_or(u32::MAX)),
                format!(
                    "unexpected text after a complete behavioural expression in '{}'",
                    token.text
                ),
            ))));
        }
        input.next_token();
    }
    Ok(())
}

/// `name=value` instance setters after a B expression. `output` is the
/// expression's own name (`v` or `i`), for the duplicate diagnostic.
fn b_setters(input: &mut Input<'_>, output: &str) -> Result<Vec<ParameterAssignment>> {
    let mut setters = Vec::new();
    while let Some(token) = input.input.first() {
        let lowered = token.text.to_ascii_lowercase();
        let is_setter = input
            .input
            .get(1)
            .is_some_and(|next| next.kind == TokenKind::Equals);
        if is_setter && matches!(lowered.as_str(), "v" | "i") {
            return Err(malformed(
                input,
                &format!(
                    "a B source takes exactly one v= or i= expression ({output}= was already given)"
                ),
            ));
        }
        if !is_setter || !B_SETTERS.contains(&lowered.as_str()) {
            return Err(malformed(
                input,
                &format!(
                    "unexpected '{}' after the behavioural expression; expected one of {} as name=value",
                    token.text,
                    B_SETTERS.join(", ")
                ),
            ));
        }
        setters.push(setter(input)?);
    }
    Ok(setters)
}

fn setter(input: &mut Input<'_>) -> Result<ParameterAssignment> {
    let (token, _, value) = (any, equals, value).parse_next(input)?;
    Ok(named(
        &token.text.to_ascii_lowercase(),
        token.location.clone(),
        value,
    ))
}

fn is_named(input: &Input<'_>, keyword: &str) -> bool {
    input.input.first().is_some_and(|t| t.is_keyword(keyword))
        && input
            .input
            .get(1)
            .is_some_and(|t| t.kind == TokenKind::Equals)
}

/// A value token: a number, `{expr}` or `'expr'`.
fn is_value_token(token: &Token) -> bool {
    token.number().is_some() || super::expression::is_expression_token(token)
}

/// `VALUE=`/`VOL=` (E) or `VALUE=`/`CUR=` (G): `inp_compat()` moves the
/// expression into a B source driving an internal node.
pub(super) fn value_form(
    input: &mut Input<'_>,
    designator: char,
    keyword: &Token,
) -> Result<Vec<ParameterAssignment>> {
    any.parse_next(input)?;
    let equal = equals
        .parse_next(input)
        .map_err(|_| malformed(input, "expected '=' after VALUE/VOL/CUR"))?;
    let expression = expression_after(input, equal, false)?;
    let mut parameters = vec![behavioural("value", keyword, expression)];
    if designator == 'e' {
        // inp_compat copies everything after the expression onto the B card.
        parameters.extend(b_setters(input, "value")?);
    } else if is_named(input, "m") {
        // The G's own gain becomes m (inp_compat, "find multiplier m").
        parameters.push(setter(input)?);
    }
    if !input.input.is_empty() {
        return Err(malformed(
            input,
            &format!(
                "unexpected '{}' after the VALUE expression",
                input.input[0].text
            ),
        ));
    }
    Ok(parameters)
}

/// `TABLE {expr} [=] (x, y) ...` (two output nodes, E/G) or, with
/// `control_nodes`, the `LTspice` `table=(x0, y0, x1, y1, ...)` after the two
/// controlling nodes (E only). Points become ordered `x`/`y` setters.
pub(super) fn table_form(
    input: &mut Input<'_>,
    designator: char,
    control_nodes: bool,
) -> Result<Vec<ParameterAssignment>> {
    let keyword = any.parse_next(input)?;
    let mut parameters = Vec::new();
    let _ = winnow::combinator::opt(equals).parse_next(input)?;
    if control_nodes {
        parameters.push(ParameterAssignment {
            name: "table".to_owned(),
            value: String::new(),
            kind: ParameterKind::Flag,
            location: keyword.location.clone(),
        });
    } else {
        let Some(token) = input
            .input
            .first()
            .filter(|t| super::expression::is_expression_token(t))
        else {
            return Err(malformed(
                input,
                "expected the TABLE input expression in braces (TABLE {expr} = (x0,y0) ...)",
            ));
        };
        let inner = token
            .text
            .get(1..token.text.len().saturating_sub(1))
            .unwrap_or_default();
        let verbatim = inner.trim_start().to_ascii_lowercase().starts_with("pwl(");
        let expression = super::bexpression::parse_complete(
            inner,
            &token.location,
            token.location.column + 1,
            verbatim,
            input.state.auto_gnd,
        )
        .map_err(|error| ErrMode::Cut(Failure(error)))?;
        parameters.push(behavioural("table", keyword, expression));
        any.parse_next(input)?;
        let _ = winnow::combinator::opt(equals).parse_next(input)?;
    }
    let mut values = Vec::new();
    loop {
        skip_terminal_punctuation(input)?;
        let Some(token) = input.input.first() else {
            break;
        };
        if is_named(input, "m") {
            break;
        }
        if !is_value_token(token) {
            return Err(malformed(
                input,
                &format!(
                    "expected a TABLE point value (a number or {{expression}}), found '{}'",
                    token.text
                ),
            ));
        }
        let value = value.parse_next(input)?;
        values.push(value);
    }
    if values.is_empty() || values.len() % 2 != 0 {
        return Err(malformed(
            input,
            &format!(
                "TABLE needs (x, y) pairs; found {} value(s) (C: missing token)",
                values.len()
            ),
        ));
    }
    for (index, value) in values.into_iter().enumerate() {
        let name = if index % 2 == 0 { "x" } else { "y" };
        let location = value.token().location.clone();
        parameters.push(named(name, location, value));
    }
    if is_named(input, "m") {
        if designator != 'g' {
            return Err(malformed(
                input,
                "m= is not accepted on an E TABLE source (C reads it as a missing point)",
            ));
        }
        parameters.push(setter(input)?);
    }
    if !input.input.is_empty() {
        return Err(malformed(
            input,
            &format!(
                "unexpected '{}' after the TABLE points",
                input.input[0].text
            ),
        ));
    }
    Ok(parameters)
}

/// `POLY(n)` after the output nodes: `n` controlling node pairs (E/G) or
/// sources (F/H), then the SPICE2 coefficients.
pub(super) fn poly_form(
    input: &mut Input<'_>,
    designator: char,
    nodes: &mut Vec<String>,
) -> Result<Vec<ParameterAssignment>> {
    let keyword = any.parse_next(input)?;
    skip_terminal_punctuation(input)?;
    let count = input
        .input
        .first()
        .and_then(|token| poly_dimension(&token.text))
        .ok_or_else(|| {
            malformed(
                input,
                "expected a positive integer POLY dimension (POLY(n), n <= 64)",
            )
        })?;
    let dimension = any.parse_next(input)?;
    let mut parameters = vec![ParameterAssignment {
        name: "poly".to_owned(),
        value: dimension.text.clone(),
        kind: ParameterKind::Scalar,
        location: keyword.location.clone(),
    }];
    controls(input, designator, count, nodes, &mut parameters)?;
    coefficients(input, designator, &mut parameters)?;
    Ok(parameters)
}

fn controls(
    input: &mut Input<'_>,
    designator: char,
    count: usize,
    nodes: &mut Vec<String>,
    parameters: &mut Vec<ParameterAssignment>,
) -> Result<()> {
    let auto_gnd = input.state.auto_gnd;
    if matches!(designator, 'e' | 'g') {
        for _ in 0..2 * count {
            let node = terminal_name(input, "controlling node of the POLY source")?;
            nodes.push(canonical_node(node, auto_gnd));
        }
    } else {
        for _ in 0..count {
            let source = terminal_name(input, "controlling source of the POLY source")?;
            parameters.push(ParameterAssignment {
                name: "control".to_owned(),
                value: source.text.to_ascii_lowercase(),
                kind: ParameterKind::Instance,
                location: source.location.clone(),
            });
        }
    }
    Ok(())
}

/// SPICE2 coefficients, then an optional `m=` (G/F only; `enhtrans.c` drops
/// it with a warning on E/H, which the port reports as an error instead).
pub(super) fn coefficients(
    input: &mut Input<'_>,
    designator: char,
    parameters: &mut Vec<ParameterAssignment>,
) -> Result<()> {
    let mut found = 0usize;
    loop {
        skip_terminal_punctuation(input)?;
        let Some(token) = input.input.first() else {
            break;
        };
        if is_named(input, "m") {
            break;
        }
        if !is_value_token(token) {
            return Err(malformed(
                input,
                &format!(
                    "expected a POLY coefficient (a number or {{expression}}), found '{}'",
                    token.text
                ),
            ));
        }
        let value = value.parse_next(input)?;
        let location = value.token().location.clone();
        parameters.push(named("coef", location, value));
        found += 1;
    }
    if found == 0 {
        return Err(malformed(
            input,
            "a POLY source needs at least one coefficient (C: number of connections differs \
             from poly dimension)",
        ));
    }
    if is_named(input, "m") {
        if !matches!(designator, 'g' | 'f') {
            return Err(ErrMode::Cut(Failure(SpiceError::Unsupported {
                feature: format!(
                    "m= on an {} POLY source: enhtrans.c ignores it with a warning; remove it",
                    designator.to_ascii_uppercase()
                ),
                location: Some(location(input)),
            })));
        }
        parameters.push(setter(input)?);
    }
    if !input.input.is_empty() {
        return Err(malformed(
            input,
            &format!(
                "unexpected '{}' after the POLY coefficients",
                input.input[0].text
            ),
        ));
    }
    Ok(())
}
