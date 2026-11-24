//! P/CPL distributed multiconductor transmission-line instance grammar.
//!
//! C input: Pname in1..inN input-return out1..outN output-return model
//!          optional length followed by a positive distance.
//! The two return terminals remain distinct; model R/L/G/C matrices use
//! this exact conductor ordering.
use winnow::Parser as _;
use winnow::combinator::{cut_err, opt};
use winnow::error::ErrMode;
use winnow::token::any;

use super::grammar::{Failure, Input, ParsedCard, Result};
use super::syntax::{canonical_node, equals, malformed, name, named, named_device, value};
use crate::netlist::ast::DeviceInstance;
use crate::netlist::token::Token;
use crate::primitives::SpiceError;

pub(super) fn cpl_card(input: &mut Input<'_>) -> Result<ParsedCard> {
    let (instance, _) = named_device("p").parse_next(input)?;
    cut_err(move |input: &mut Input<'_>| body(input, instance)).parse_next(input)
}
fn body(input: &mut Input<'_>, instance: &Token) -> Result<ParsedCard> {
    let tokens = &input.input;
    if tokens.len() < 5 {
        return Err(malformed(
            input,
            "P/CPL needs input/output conductors, two returns, and a model",
        ));
    }
    let last = tokens.len();
    let has_length = last >= 3
        && ((tokens[last - 2].is_keyword("length") || tokens[last - 2].is_keyword("len"))
            || (last >= 4
                && (tokens[last - 3].is_keyword("length") || tokens[last - 3].is_keyword("len"))));
    let model_end = if has_length {
        if tokens[last - 2].is_keyword("length") || tokens[last - 2].is_keyword("len") {
            last - 2
        } else {
            last - 3
        }
    } else {
        last
    };
    let node_count = model_end.saturating_sub(1);
    if node_count < 4 || node_count % 2 != 0 {
        return Err(malformed(
            input,
            "P/CPL requires N input conductors, one input return, N output conductors, one output return, then a model",
        ));
    }
    let auto_gnd = input.state.auto_gnd;
    let mut nodes = Vec::new();
    nodes.try_reserve(node_count).map_err(|_| {
        ErrMode::Cut(Failure(SpiceError::parse(
            input.state.card.location.clone(),
            "cannot allocate CPL terminals",
        )))
    })?;
    for _ in 0..node_count {
        let token = name("P/CPL conductor or return terminal").parse_next(input)?;
        nodes.push(canonical_node(token, auto_gnd));
    }
    let model = name("P/CPL model declaration").parse_next(input)?;
    let mut parameters = Vec::new();
    if has_length {
        let key = any
            .verify(|token: &Token| token.is_keyword("length") || token.is_keyword("len"))
            .parse_next(input)?;
        let (_, length) = cut_err((opt(equals), value)).parse_next(input)?;
        parameters.push(named("length", key.location.clone(), length));
    }
    if !input.input.is_empty() {
        return Err(malformed(
            input,
            "unexpected trailing CPL instance parameters",
        ));
    }
    Ok(ParsedCard::Device(DeviceInstance {
        name: instance.text.to_ascii_lowercase(),
        designator: 'p',
        nodes,
        model: Some(model.text.to_ascii_lowercase()),
        parameters,
        location: input.state.card.location.clone(),
    }))
}
