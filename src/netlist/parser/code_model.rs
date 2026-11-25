//! Scalar XSPICE analog ports and model-bound code-model instances.
//! The declared model selects the backend at elaboration; other code models
//! do not acquire rational-transfer semantics merely by using an A card.
use crate::netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind};
use crate::netlist::token::TokenKind;
use winnow::Parser as _;
use winnow::combinator::{cut_err, opt};
use winnow::token::any;

use super::grammar::{Input, ParsedCard, Result};
use super::syntax::{canonical_node, malformed, name, named_device, punctuation};

pub(super) fn code_model_card(input: &mut Input<'_>) -> Result<ParsedCard> {
    let (instance, designator) = named_device("a").parse_next(input)?;
    cut_err(|input: &mut Input<'_>| {
        let (input_nodes, input_mode) = port(input, "input")?;
        let (output_nodes, output_mode) = port(input, "output")?;
        let model = name("code model name").parse_next(input)?;
        Ok(ParsedCard::Device(DeviceInstance {
            name: instance.text.to_ascii_lowercase(),
            designator,
            nodes: output_nodes.into_iter().chain(input_nodes).collect(),
            model: Some(model.text.to_ascii_lowercase()),
            parameters: vec![input_mode, output_mode],
            location: instance.location.clone(),
        }))
    })
    .parse_next(input)
}

fn port(input: &mut Input<'_>, direction: &str) -> Result<([String; 2], ParameterAssignment)> {
    let token = input
        .input
        .first()
        .ok_or_else(|| malformed(input, "missing code model port"))?;
    let spelling = token.text.to_ascii_lowercase();
    let (current, differential, explicit) = match spelling.as_str() {
        "%v" => (false, false, true),
        "%vd" => (false, true, true),
        "%i" => (true, false, true),
        "%id" => (true, true, true),
        _ if spelling.starts_with('%') => {
            return Err(malformed(input, "unsupported analog port type"));
        }
        _ => (false, false, false),
    };
    let location = token.location.clone();
    if explicit {
        any.parse_next(input)?;
        punctuation(TokenKind::LParen)
            .context("'(' after port type")
            .parse_next(input)?;
    }
    let positive = name("positive analog port node").parse_next(input)?;
    let positive = canonical_node(positive, input.state.auto_gnd);
    let negative = if differential {
        opt(punctuation(TokenKind::Comma)).parse_next(input)?;
        let negative = name("negative analog port node").parse_next(input)?;
        canonical_node(negative, input.state.auto_gnd)
    } else {
        "0".to_owned()
    };
    if explicit {
        punctuation(TokenKind::RParen)
            .context("')' after port nodes")
            .parse_next(input)?;
    }
    Ok((
        [positive, negative],
        ParameterAssignment {
            name: direction.to_owned(),
            value: if current { "current" } else { "voltage" }.to_owned(),
            kind: ParameterKind::Textual,
            location,
        },
    ))
}
