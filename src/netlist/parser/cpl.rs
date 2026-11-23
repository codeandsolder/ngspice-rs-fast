//! P/CPL multiconductor line grammar, preserving ordered physical matrices.
//!
//! Pname in1 [in2 ...] reference_in out1 [out2 ...] reference_out model
//!       [length=physical_length]
//! .model model CPL [length=value] R=(packed upper triangle) ...
//!
//! A CPL terminal group always has one explicit return at each end. The
//! model's R/L/G/C matrices are physical numeric vectors, not independent
//! scalar setters. Exactly which dimensions a model requires is validated
//! when the instance is bound to its conductor count.

use winnow::Parser as _;
use winnow::combinator::{cut_err, opt};
use winnow::token::any;

use crate::netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind, PositionedValue};
use crate::netlist::token::{Token, TokenKind};

use super::grammar::{Input, ParsedCard, Result};
use super::syntax::{canonical_node, equals, malformed, named, named_device, value};

pub(super) fn p_card(input: &mut Input<'_>) -> Result<ParsedCard> {
    let (instance, _) = named_device("p").parse_next(input)?;
    cut_err(|input: &mut Input<'_>| p_body(input, instance)).parse_next(input)
}

fn p_body(input: &mut Input<'_>, instance: &Token) -> Result<ParsedCard> {
    let mut components = Vec::new();
    while let Some(token) = input.input.first() {
        if token.is_keyword("length") || token.is_keyword("len") {
            break;
        }
        if !token.is_name_like() {
            return Err(malformed(
                input,
                "expected CPL conductor, return, or model name",
            ));
        }
        let position = any.parse_next(input)?;
        components.push(position);
        if components.len() > 2 * crate::devices::cpl::plan::MAX_CONDUCTORS + 3 {
            return Err(malformed(
                input,
                "CPL terminal count exceeds eight conductors",
            ));
        }
    }
    // 2n+2 electrical terminals and one final model reference.
    if components.len() < 5 || components.len() % 2 == 0 {
        return Err(malformed(
            input,
            "CPL needs n near conductors, near return, n far conductors, far return, and model",
        ));
    }
    let conductors = (components.len() - 3) / 2;
    if !(1..=crate::devices::cpl::plan::MAX_CONDUCTORS).contains(&conductors) {
        return Err(malformed(input, "CPL conductor order out of range"));
    }
    let model = components
        .pop()
        .ok_or_else(|| malformed(input, "CPL model name missing"))?;
    if !matches!(model.kind, TokenKind::Word) {
        return Err(malformed(
            input,
            "CPL model must be a named model, not a number",
        ));
    }
    let nodes = components
        .iter()
        .map(|node| canonical_node(node, input.state.auto_gnd))
        .collect();
    let mut parameters = Vec::new();
    if input
        .input
        .first()
        .is_some_and(|token| token.is_keyword("length") || token.is_keyword("len"))
    {
        let keyword = any.parse_next(input)?;
        opt(equals).parse_next(input)?;
        let value = cut_err(value).parse_next(input)?;
        parameters.push(named("length", keyword.location.clone(), value));
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

/// Parse ordered numeric-vector model setters, including parameter expressions.
/// The model is not yet an executable device. This front-end representation
/// retains original component positions for later scope resolution.
///
/// A vector has 1..=36 entries (the largest CPL upper triangle). Full matrix
/// dimension checking occurs when associated with an instance.
pub(super) fn model_parameters(input: &mut Input<'_>) -> Result<Vec<ParameterAssignment>> {
    let mut parameters = Vec::new();
    loop {
        let Some(next) = input.input.first() else {
            break;
        };
        if next.kind == TokenKind::RParen {
            break;
        }
        if next.kind == TokenKind::Comma {
            any.parse_next(input)?;
            continue;
        }
        if !matches!(next.kind, TokenKind::Word)
            || !["r", "l", "g", "c", "len", "length"]
                .iter()
                .any(|name| next.is_keyword(name))
        {
            return Err(malformed(
                input,
                "expected CPL R/L/G/C vector or length setter",
            ));
        }
        let keyword = any.parse_next(input)?;
        let name = keyword.text.to_ascii_lowercase();
        opt(equals).parse_next(input)?;
        if matches!(name.as_str(), "length" | "len") {
            let scalar = cut_err(value).parse_next(input)?;
            parameters.push(named("length", keyword.location.clone(), scalar));
            continue;
        }
        let parenthesized = input
            .input
            .first()
            .is_some_and(|token| token.kind == TokenKind::LParen);
        if parenthesized {
            any.parse_next(input)?;
        }
        let mut components = Vec::new();
        loop {
            let Some(token) = input.input.first() else {
                if parenthesized {
                    return Err(malformed(input, "unclosed CPL matrix vector"));
                }
                break;
            };
            if token.kind == TokenKind::RParen {
                if parenthesized {
                    any.parse_next(input)?;
                }
                break;
            }
            if token.kind == TokenKind::Comma {
                any.parse_next(input)?;
                continue;
            }
            if !matches!(
                token.kind,
                TokenKind::Number(_) | TokenKind::Expression(_) | TokenKind::Quoted(_)
            ) {
                break;
            }
            if components.len() == 36 {
                return Err(malformed(input, "CPL matrix vector exceeds 36 entries"));
            }
            let component = cut_err(value).parse_next(input)?;
            let token = component.token();
            components.push(PositionedValue {
                text: token.text.clone(),
                location: token.location.clone(),
            });
        }
        if components.is_empty() {
            return Err(malformed(
                input,
                "CPL matrix vector requires at least one value",
            ));
        }
        let text = components
            .iter()
            .map(|component| component.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        parameters.push(ParameterAssignment {
            name,
            value: text,
            kind: ParameterKind::NumericVector(components),
            location: keyword.location.clone(),
        });
    }
    Ok(parameters)
}
