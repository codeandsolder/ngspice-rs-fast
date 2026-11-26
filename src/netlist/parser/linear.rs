//! Winnow grammars for `src/spicelib/parser/inp2{r,c,l,v,i}.c`.
//!
//! `INPdevParse()` (`inpdpar.c`) handles leading values and named parameters;
//! scalar names come from `res/res.c`, `cap/cap.c`, and `ind/ind.c`. Cuts after
//! recognised prefixes prevent optional/repeated parsers from hiding errors.

use crate::primitives::{SourceLoc, SpiceError};
use winnow::Parser as _;
use winnow::combinator::{alt, cut_err, opt, peek, repeat};
use winnow::error::ErrMode;
use winnow::token::any;

use crate::netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind};
use crate::netlist::token::{Token, TokenKind};

use super::grammar::{Failure, Input, ParsedCard, Result, gap, keyword, location};
use super::syntax::{
    assignment, canonical_node, equals, leading_value, malformed, name as node, named,
    token_designator, value,
};

pub(super) fn device_card(input: &mut Input<'_>) -> Result<ParsedCard> {
    let (name, designator) = any
        .verify_map(|token: &Token| {
            let designator = token_designator(token)?;
            matches!(designator, 'r' | 'c' | 'l' | 'v' | 'i').then_some((token, designator))
        })
        .parse_next(input)?;
    // Once the device prefix matches, failure is definitive. Without this cut,
    // alt could fall through to the generic unported-device branch.
    let (positive, negative) =
        cut_err((node("positive terminal"), node("negative terminal"))).parse_next(input)?;
    let (model, parameters) = cut_err(|input: &mut Input<'_>| {
        if matches!(designator, 'v' | 'i') {
            source_parameters(input).map(|parameters| (None, parameters))
        } else {
            passive_parameters(input, designator)
        }
    })
    .parse_next(input)?;
    Ok(ParsedCard::Device(DeviceInstance {
        code_ports: Vec::new(),
        name: name.text.to_ascii_lowercase(),
        designator,
        nodes: vec![
            canonical_node(positive, input.state.auto_gnd),
            canonical_node(negative, input.state.auto_gnd),
        ],
        model,
        parameters,
        location: input.state.card.location.clone(),
    }))
}

const fn primary_name(designator: char) -> &'static str {
    match designator {
        'r' => "resistance",
        'c' => "capacitance",
        _ => "inductance",
    }
}

/// INP2R/C/L set a pre-model scalar before named setters. A scalar immediately
/// after the model is `INPdevParse`'s leading value, applied *after* named setters.
/// No scalar is injected for a model-only/geometry instance. Declared numeric
/// names cannot steal the first numeric slot; model semantics stay in devices.
fn passive_parameters(
    input: &mut Input<'_>,
    designator: char,
) -> Result<(Option<String>, Vec<ParameterAssignment>)> {
    let primary = primary_name(designator);
    let leading = opt(|input: &mut Input<'_>| passive_leading(input, primary)).parse_next(input)?;
    let model = opt(passive_model).parse_next(input)?;
    let after_model = if model.is_some() {
        opt(leading_value).parse_next(input)?
    } else {
        None
    };
    let remaining: Vec<ParameterAssignment> =
        repeat(0.., alt((passive_assignment, invalid_passive))).parse_next(input)?;
    let mut parameters = Vec::new();
    if let Some(parameter) = leading {
        parameters.push(parameter);
    }
    parameters.extend(remaining);
    if let Some(value) = after_model {
        parameters.push(assignment(primary, value));
    }
    if model.is_none()
        && !parameters.iter().any(|parameter| {
            parameter.name == primary || (designator == 'c' && parameter.name == "charge")
        })
    {
        return Err(ErrMode::Cut(Failure(SpiceError::parse(
            location(input),
            format!("expected {primary} or a declared passive model"),
        ))));
    }
    Ok((
        model.map(|token| token.text.to_ascii_lowercase()),
        parameters,
    ))
}

fn passive_model<'a>(input: &mut Input<'a>) -> Result<&'a Token> {
    let designator = input
        .state
        .card
        .designator()
        .ok_or_else(|| malformed(input, "expected a passive device card"))?;
    // C reserves r/c/l for scalar assignment. inpcom's named-assignment
    // preprocessing also preserves keyword=value even if keyword is declared
    // as a model. Bare declared keywords in this slot *are* model references.
    let assignment = input
        .input
        .get(1)
        .is_some_and(|token| token.kind == TokenKind::Equals);
    let declared = input.state.declared_models;
    if input
        .input
        .first()
        .is_some_and(|token| token.number().is_some() && declared.contains(&token.text, false))
    {
        return Err(gap(input, "numeric-looking passive model references"));
    }
    if assignment
        && input.input.first().is_some_and(|token| {
            declared.contains(&token.text, false) && scalar_name(designator, &token.text).is_none()
        })
    {
        return Err(malformed(
            input,
            "passive model reference does not take '='",
        ));
    }
    any.verify(|token: &Token| {
        token.is_name_like()
            && !assignment
            && !token.text.eq_ignore_ascii_case(&designator.to_string())
            && declared.contains(&token.text, false)
    })
    .parse_next(input)
}

// Runtime quantities are identified in the parsed tree, not by matching
// substrings (e.g. a parameter called "time_constant" is still static).
fn behavioural_passive_value(
    input: &mut Input<'_>,
    name: &str,
    force: bool,
) -> Result<Option<ParameterAssignment>> {
    let Some(token) = input
        .input
        .first()
        .filter(|token| super::expression::is_expression_token(token))
    else {
        return if force {
            Err(malformed(
                input,
                "charge law requires a braced or quoted expression",
            ))
        } else {
            Ok(None)
        };
    };
    let expression = super::bexpression::parse_complete(
        &token.text,
        token.position(),
        false,
        input.state.auto_gnd,
    );
    let tree = match expression {
        Ok(tree) => tree,
        Err(error) if force => return Err(ErrMode::Cut(Failure(error))),
        Err(_) => return Ok(None),
    };
    let mut runtime = force;
    tree.root.visit(&mut |node| {
        use crate::netlist::bexpr::BExprKind;
        runtime |= match &node.kind {
            BExprKind::Voltage { .. } | BExprKind::Current(_) => true,
            BExprKind::Name(name) => matches!(name.as_str(), "time" | "temper" | "hertz"),
            BExprKind::Call { name, .. } => name == "ddt",
            _ => false,
        };
    });
    if !runtime {
        return Ok(None);
    }
    let at = any.parse_next(input)?;
    Ok(Some(super::behavioural::behavioural(name, at, tree)))
}

fn passive_leading(input: &mut Input<'_>, name: &str) -> Result<ParameterAssignment> {
    behavioural_passive_value(input, name, false)?.map_or_else(
        || {
            leading_value
                .map(|value| assignment(name, value))
                .parse_next(input)
        },
        Ok,
    )
}

fn passive_assignment(input: &mut Input<'_>) -> Result<ParameterAssignment> {
    let designator = input
        .state
        .card
        .designator()
        .ok_or_else(|| malformed(input, "expected a passive device card"))?;
    let (token, name) = any
        .verify_map(move |token: &Token| {
            scalar_name(designator, &token.text).map(|name| (token, name))
        })
        .parse_next(input)?;
    // INPgetTok() gobbles '=': both tc1=0.01 and tc1 0.01 are legal.
    cut_err(|input: &mut Input<'_>| {
        opt(equals).parse_next(input)?;
        if (name == primary_name(designator) || name == "charge")
            && let Some(mut parameter) = behavioural_passive_value(input, &name, name == "charge")?
        {
            parameter.location = token.location.clone();
            return Ok(parameter);
        }
        let value = value(input)?;
        Ok(named(&name, token.location.clone(), value))
    })
    .parse_next(input)
}

fn scalar_name(designator: char, text: &str) -> Option<String> {
    let name = text.to_ascii_lowercase();
    let canonical = match (designator, name.as_str()) {
        ('c', "q" | "charge") => "charge",
        ('r', "noise") => "noisy",
        ('r', "r" | "resistance")
        | ('c', "c" | "cap" | "capacitance")
        | ('l', "l" | "inductance") => primary_name(designator),
        (_, "temp" | "dtemp" | "m" | "tc1" | "tc2" | "scale")
        | ('r' | 'c', "w" | "l" | "bv_max")
        | ('r', "ac" | "tc" | "tce" | "noisy" | "initial_segment")
        | ('c' | 'l', "ic")
        | ('l', "nt") => &name,
        _ => return None,
    };
    Some(canonical.to_owned())
}

fn invalid_passive(input: &mut Input<'_>) -> Result<ParameterAssignment> {
    peek(any).parse_next(input)?;
    Err(gap(
        input,
        "undeclared passive model names, expressions or non-scalar parameters",
    ))
}

/// `INP2V()`/`INP2I()` apply leading DC after `INPdevParse()` named parameters.
/// VSRCtemp/ISRCtemp default bare AC to magnitude 1 and phase 0.
fn source_parameters(input: &mut Input<'_>) -> Result<Vec<ParameterAssignment>> {
    let leading = opt(leading_value).parse_next(input)?;
    let mut parameters = repeat(
        0..,
        alt((
            dc_parameters,
            ac_parameters,
            distortion_parameters,
            port_parameter,
            super::waveform::parameters,
            super::waveform::pwl_options,
            invalid_source,
        )),
    )
    .fold(Vec::new, |mut parameters, chunk| {
        parameters.extend(chunk);
        parameters
    })
    .parse_next(input)?;
    if let Some(value) = leading {
        parameters.push(assignment("dc", value));
    }
    // No explicit source value is legal: keep the implicit DC zero implicit.
    Ok(parameters)
}

fn dc_parameters(input: &mut Input<'_>) -> Result<Vec<ParameterAssignment>> {
    let (token, (_, value)) = (keyword("dc"), cut_err((opt(equals), value))).parse_next(input)?;
    Ok(vec![named("dc", token.location.clone(), value)])
}

fn ac_parameters(input: &mut Input<'_>) -> Result<Vec<ParameterAssignment>> {
    let token = keyword("ac").parse_next(input)?;
    cut_err((
        opt(equals),
        ac_value("acmag", "1", &token.location),
        ac_value("acphase", "0", &token.location),
    ))
    .map(|(_, magnitude, phase)| vec![magnitude, phase])
    .parse_next(input)
}

/// `distof1 [mag [phase]]` / `distof2 [mag [phase]]` (`vsrc.c`/`isrc.c`
/// `IF_REALVEC` setters, `vsrcpar.c`: a missing magnitude is 1 and a missing
/// phase 0), kept as the ordered pair `distof<k>mag`, `distof<k>phase`.
fn distortion_parameters(input: &mut Input<'_>) -> Result<Vec<ParameterAssignment>> {
    let (token, which) = alt((
        keyword("distof1").map(|token| (token, 1)),
        keyword("distof2").map(|token| (token, 2)),
    ))
    .parse_next(input)?;
    let (magnitude, phase) = if which == 1 {
        ("distof1mag", "distof1phase")
    } else {
        ("distof2mag", "distof2phase")
    };
    cut_err((
        opt(equals),
        ac_value(magnitude, "1", &token.location),
        ac_value(phase, "0", &token.location),
    ))
    .map(|(_, magnitude, phase)| vec![magnitude, phase])
    .parse_next(input)
}

fn ac_value<'a>(
    name: &'static str,
    default: &'static str,
    default_location: &SourceLoc,
) -> impl winnow::Parser<Input<'a>, ParameterAssignment, ErrMode<Failure>> {
    let default_location = default_location.clone();
    move |input: &mut Input<'a>| {
        if input.input.first().is_some_and(|token| {
            matches!(token.kind, TokenKind::Quoted(_))
                && !super::expression::is_expression_token(token)
        }) {
            return Err(gap(input, "double-quoted AC parameter strings"));
        }
        opt(leading_value)
            .map(|value| {
                value.map_or_else(
                    || ParameterAssignment {
                        name: name.to_owned(),
                        value: default.to_owned(),
                        kind: ParameterKind::Scalar,
                        location: default_location.clone(),
                    },
                    |token| assignment(name, token),
                )
            })
            .parse_next(input)
    }
}

/// The RFSPICE port setters of a voltage source (`vsrc.c`: `portnum`, `z0`,
/// `pwr`, `freq`, `phase`), each `name [=] value`, kept as ordered scalar
/// assignments: `vsrcpar.c` applies them in deck order and `portnum` reads the
/// `z0` set so far. Current sources have no such parameters.
fn port_parameter(input: &mut Input<'_>) -> Result<Vec<ParameterAssignment>> {
    let voltage = input.state.card.designator() == Some('v');
    let (token, name) = any
        .verify_map(move |token: &Token| {
            let name = token.text.to_ascii_lowercase();
            (voltage && matches!(name.as_str(), "portnum" | "z0" | "pwr" | "freq" | "phase"))
                .then_some((token, name))
        })
        .parse_next(input)?;
    let (_, value) = cut_err((opt(equals), value)).parse_next(input)?;
    Ok(vec![named(&name, token.location.clone(), value)])
}

fn invalid_source(input: &mut Input<'_>) -> Result<Vec<ParameterAssignment>> {
    let token = peek(any).parse_next(input)?;
    if matches!(
        token.kind,
        TokenKind::Equals | TokenKind::LParen | TokenKind::RParen | TokenKind::Comma
    ) {
        return Err(malformed(input, "expected a named source parameter"));
    }
    Err(gap(
        input,
        "unsupported source waveforms, expressions or additional source parameters",
    ))
}
