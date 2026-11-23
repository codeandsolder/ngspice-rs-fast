//! Shared four-terminal and ordered setter grammar for T/O transmission lines.
//!
//! The instance-specific parsers retain their different model requirements,
//! positional-value rejection and precise diagnostic wording.

use winnow::Parser as _;
use winnow::combinator::{alt, cut_err, opt, peek, repeat};
use winnow::token::any;

use crate::netlist::ast::ParameterAssignment;
use crate::netlist::token::{Token, TokenKind};

use super::grammar::{Input, Result};
use super::ic;
use super::syntax::{canonical_node, equals, malformed, name, named, value};

const LINE_IC_COMPONENTS: &[&str] = &["v1", "i1", "v2", "i2"];

pub(super) fn four_terminals(input: &mut Input<'_>) -> Result<Vec<String>> {
    let auto_gnd = input.state.auto_gnd;
    let mut nodes = Vec::with_capacity(4);
    for what in [
        "port 1 positive node",
        "port 1 negative node",
        "port 2 positive node",
        "port 2 negative node",
    ] {
        let node = name(what).parse_next(input)?;
        nodes.push(canonical_node(node, auto_gnd));
    }
    Ok(nodes)
}

/// Keeps every setter in source order: later scalar/IC assignments override
/// earlier ones in the actual device setter, not in the parser.
pub(super) fn parameters(
    input: &mut Input<'_>,
    scalars: &[&str],
    description: &str,
    expected: &str,
) -> Result<Vec<ParameterAssignment>> {
    repeat(
        0..,
        alt((
            |input: &mut Input<'_>| ic::vector_with_components(input, LINE_IC_COMPONENTS),
            |input: &mut Input<'_>| scalar(input, scalars),
            |input: &mut Input<'_>| invalid(input, description, expected),
        )),
    )
    .parse_next(input)
}

fn scalar(input: &mut Input<'_>, scalars: &[&str]) -> Result<ParameterAssignment> {
    let token = any
        .verify(|token: &Token| {
            token.kind == TokenKind::Word && scalars.iter().any(|keyword| token.is_keyword(keyword))
        })
        .parse_next(input)?;
    let (_, value) = cut_err((opt(equals), value)).parse_next(input)?;
    Ok(named(
        &token.text.to_ascii_lowercase(),
        token.location.clone(),
        value,
    ))
}

fn invalid(
    input: &mut Input<'_>,
    description: &str,
    expected: &str,
) -> Result<ParameterAssignment> {
    let token = peek(any).parse_next(input)?;
    Err(malformed(
        input,
        &format!(
            "unknown {description} parameter '{}' (expected {expected})",
            token.text
        ),
    ))
}
