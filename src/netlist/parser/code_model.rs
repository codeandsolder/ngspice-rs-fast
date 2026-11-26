//! Ordered, model-independent XSPICE A-instance ports.
use crate::netlist::ast::{CodePort, CodePortKind, CodeTerminal, DeviceInstance};
use crate::netlist::token::{Token, TokenKind};
use winnow::Parser as _;
use winnow::combinator::{cut_err, opt};
use winnow::token::any;

use super::grammar::{Input, ParsedCard, Result};
use super::syntax::{canonical_node, malformed, name, named_device, punctuation};

pub(super) fn code_model_card(input: &mut Input<'_>) -> Result<ParsedCard> {
    let (instance, designator) = named_device("a").parse_next(input)?;
    cut_err(|input: &mut Input<'_>| {
        let schema = input
            .input
            .last()
            .and_then(|token| input.state.declared_models.code_model(&token.text));
        let mut ports = Vec::new();
        while input.input.len() > 1 {
            let default = schema
                .and_then(|schema| schema.ports().get(ports.len()))
                .map_or(CodePortKind::Default, |port| port.default);
            ports.push(port(input, default)?);
        }
        if ports.is_empty() {
            return Err(malformed(input, "A instance requires at least one port"));
        }
        let model = terminal_name(input)?;
        Ok(ParsedCard::Device(DeviceInstance {
            name: instance.text.to_ascii_lowercase(),
            designator,
            nodes: Vec::new(),
            code_ports: ports,
            model: Some(model.text.to_ascii_lowercase()),
            parameters: Vec::new(),
            location: instance.location.clone(),
        }))
    })
    .parse_next(input)
}

fn marker(input: &mut Input<'_>, spelling: &str) -> Result<()> {
    any.verify(|token: &Token| token.text == spelling)
        .void()
        .parse_next(input)
}

fn terminal_name<'a>(input: &mut Input<'a>) -> Result<&'a Token> {
    let token = name("code model terminal or model name").parse_next(input)?;
    if matches!(token.text.as_str(), "[" | "]" | "~") || token.text.starts_with('%') {
        return Err(malformed(input, "invalid code model name"));
    }
    Ok(token)
}

fn port(input: &mut Input<'_>, default: CodePortKind) -> Result<CodePort> {
    let token = input
        .input
        .first()
        .ok_or_else(|| malformed(input, "missing code model port"))?;
    let kind = match token.text.to_ascii_lowercase().as_str() {
        "%v" => CodePortKind::Voltage,
        "%vd" => CodePortKind::VoltageDifferential,
        "%i" => CodePortKind::Current,
        "%id" => CodePortKind::CurrentDifferential,
        "%d" => CodePortKind::Digital,
        "%g" => CodePortKind::Conductance,
        "%gd" => CodePortKind::ConductanceDifferential,
        value if value.starts_with('%') => {
            return Err(malformed(input, "unsupported code model port type"));
        }
        _ => CodePortKind::Default,
    };
    let kind = if kind == CodePortKind::Default {
        default
    } else {
        any.parse_next(input)?;
        kind
    };
    let vector = opt(|input: &mut Input<'_>| marker(input, "["))
        .parse_next(input)?
        .is_some();
    let terminals = if vector {
        let mut terminals = Vec::new();
        while input.input.first().is_some_and(|token| token.text != "]") {
            if !terminals.is_empty() {
                opt(punctuation(TokenKind::Comma)).parse_next(input)?;
            }
            terminals.push(element(input, kind)?);
        }
        marker(input, "]")?;
        if terminals.is_empty() {
            return Err(malformed(input, "empty code model vector"));
        }
        terminals
    } else {
        vec![element(input, kind)?]
    };
    Ok(CodePort {
        kind,
        vector,
        terminals,
    })
}

fn element(input: &mut Input<'_>, kind: CodePortKind) -> Result<CodeTerminal> {
    let grouped = opt(punctuation(TokenKind::LParen))
        .parse_next(input)?
        .is_some();
    let inverted = opt(|input: &mut Input<'_>| marker(input, "~"))
        .parse_next(input)?
        .is_some();
    if inverted && !matches!(kind, CodePortKind::Digital | CodePortKind::Default) {
        return Err(malformed(input, "inverted analog terminal"));
    }
    let node = |token: &Token, auto_gnd| {
        if matches!(kind, CodePortKind::Digital | CodePortKind::Default) {
            token.text.to_ascii_lowercase()
        } else {
            canonical_node(token, auto_gnd)
        }
    };
    let positive = node(terminal_name(input)?, input.state.auto_gnd);
    let negative = if kind.differential() {
        opt(punctuation(TokenKind::Comma)).parse_next(input)?;
        Some(node(terminal_name(input)?, input.state.auto_gnd))
    } else {
        None
    };
    if grouped {
        punctuation(TokenKind::RParen).parse_next(input)?;
    }
    Ok(CodeTerminal {
        positive,
        negative,
        inverted,
    })
}
