//! Bounded `.model` grammar (`inpdomod.c`, `inpgmod.c`, `inpfindl.c`).
//!
//! Retains scalar assignments and bounded bare type flags for D/BJT/MOS/R/C/L
//! and the SW/CSW switch models (`sw.c`/`csw.c` `SWmPTable`/`CSWmPTable`),
//! plus the URC line model (`urc.c` `URCmPTable`, whose `urc` flag is a no-op)
//! and the lossy-line model (`ltra.c` `LTRAmPTable`, with its `IF_FLAG` options).
//! The base token remains distinct from ordered tail flag setters.
//! Device parameter validity, default levels, selector rounding, model lookup
//! and availability are elaboration concerns, not claims made by this grammar.

use winnow::Parser as _;
use winnow::combinator::{alt, cut_err, opt, peek, repeat};
use winnow::error::ErrMode;
use winnow::token::any;

use crate::devices::code_model::source::SourceModel;
use crate::netlist::ast::{
    ModelCard, NumericVectorEntry, ParameterAssignment, ParameterKind, PositionedValue,
};
use crate::netlist::token::{Token, TokenKind};

use super::grammar::{Failure, Input, ParsedCard, Result, gap, keyword};
use super::syntax::{Value, equals, malformed, name, named, punctuation, value};

pub(super) fn model_card(input: &mut Input<'_>) -> Result<ParsedCard> {
    keyword(".model").parse_next(input)?;
    cut_err(model_body).parse_next(input)
}

fn model_body(input: &mut Input<'_>) -> Result<ParsedCard> {
    let (model_name, base) = (name("model name"), model_base).parse_next(input)?;
    // C gobbles commas as delimiters. We retain a single optional outer pair
    // of parentheses, diagnosing an unmatched pair instead of ignoring it.
    let parameters: Vec<ParameterAssignment> = alt((
        (
            punctuation(TokenKind::LParen),
            cut_err((
                |input: &mut Input<'_>| parameters(input, &base.text),
                punctuation(TokenKind::RParen).context("')' after model parameters"),
            )),
        )
            .map(|(_, (parameters, _))| parameters),
        |input: &mut Input<'_>| parameters(input, &base.text),
    ))
    .parse_next(input)?;
    // INPfindLev scans the first level. Keep its literal value here, without
    // injecting defaults or rounding it into a backend selector. Preserve all
    // assignments separately, including duplicates for future model setters.
    let level = parameters
        .iter()
        .find(|parameter| parameter.name == "level")
        .map(|parameter| {
            crate::primitives::parse_spice_number(&parameter.value).ok_or_else(|| {
                ErrMode::Cut(Failure(crate::primitives::SpiceError::parse(
                    parameter.location.clone(),
                    "expected a literal model level",
                )))
            })
        })
        .transpose()?;
    Ok(ParsedCard::Model(ModelCard {
        name: model_name.text.to_ascii_lowercase(),
        base: base.text.to_ascii_lowercase(),
        level,
        parameters,
        location: input.state.card.location.clone(),
    }))
}

fn model_base<'a>(input: &mut Input<'a>) -> Result<&'a Token> {
    let base = peek(name("model type")).parse_next(input)?;
    if crate::devices::ModelFamily::from_base(&base.text).is_none() {
        return Err(gap(
            input,
            "model family outside registered scalar and vector model syntax",
        ));
    }
    any.parse_next(input)
}

fn parameters(input: &mut Input<'_>, base: &str) -> Result<Vec<ParameterAssignment>> {
    repeat(
        0..,
        alt((
            punctuation(TokenKind::Comma).value(None),
            (|input: &mut Input<'_>| cpl_vector(input, base)).map(Some),
            (|input: &mut Input<'_>| code_vector(input, base)).map(Some),
            (|input: &mut Input<'_>| code_boolean(input, base)).map(Some),
            (|input: &mut Input<'_>| legacy_partition(input, base)).map(Some),
            (|input: &mut Input<'_>| super::flags::model(input, base)).map(Some),
            (|input: &mut Input<'_>| version_assignment(input, base)).map(Some),
            scalar_assignment.map(Some),
            invalid_parameter,
        )),
    )
    .fold(Vec::new, |mut parameters, parameter| {
        if let Some(parameter) = parameter {
            parameters.push(parameter);
        }
        parameters
    })
    .parse_next(input)
}

// BSIM1 declares XPART as a flag, while BSIM2/3/4 declare a real
// scalar. Retain the source shape here and validate it after LEVEL dispatch.
fn legacy_partition(input: &mut Input<'_>, base: &str) -> Result<ParameterAssignment> {
    if !matches!(base, "nmos" | "pmos")
        || !input
            .input
            .first()
            .is_some_and(|token| token.is_keyword("xpart"))
        || input.input.get(1).is_some_and(|next| {
            next.kind == TokenKind::Equals
                || next.number().is_some()
                || super::expression::is_expression_token(next)
        })
    {
        return Err(winnow::error::ErrMode::Backtrack(Failure(
            crate::primitives::SpiceError::parse(
                input.state.card.location.clone(),
                "expected bare partition flag",
            ),
        )));
    }
    let token = any.parse_next(input)?;
    Ok(ParameterAssignment {
        name: "xpart".into(),
        value: String::new(),
        kind: ParameterKind::Flag,
        location: token.location.clone(),
    })
}

/// `INPfindVer`/`IF_STRING` retain a bare dotted VERSION token, not a float.
/// Only MOS model VERSION sites gain this syntax; other word-valued setters
/// keep their existing refusal. Quoted/expression versions are not substituted.
fn version_assignment(input: &mut Input<'_>, base: &str) -> Result<ParameterAssignment> {
    peek(
        any.verify(|token: &Token| matches!(base, "nmos" | "pmos") && token.is_keyword("version")),
    )
    .parse_next(input)?;
    let keyword = any.parse_next(input)?;
    opt(equals).parse_next(input)?;
    let token = cut_err(
        any.verify(|token: &Token| {
            token.kind == TokenKind::Word || token.number().is_some_and(f64::is_finite)
        })
        .context("a bare MOS version literal"),
    )
    .parse_next(input)?;
    Ok(ParameterAssignment {
        name: "version".into(),
        value: token.text.clone(),
        kind: if token.number().is_some() {
            ParameterKind::Scalar
        } else {
            ParameterKind::Textual
        },
        location: keyword.location.clone(),
    })
}

fn scalar_assignment(input: &mut Input<'_>) -> Result<ParameterAssignment> {
    // A flag is not a scalar assignment, even when it is spelled as a word -
    // unless an `=` follows it. `d` is both: a bare `d` is a model-type flag
    // (`tests/flags_ic_parser.rs`), while `.model m nmf(d=0.24u)` is MESA's
    // device depth, which C's `INPpSetP` reads as an assignment. The reference
    // binary answers 4.162441e-04 A for that card against 1.926577e-03 A for
    // the default depth, so the `=` has to settle it before the refusal.
    let flag = input.input.first().is_some_and(|token| {
        [
            "off",
            "nchan",
            "pchan",
            "thermal",
            "d",
            "npn",
            "pnp",
            "nmos",
            "pmos",
            "sw",
            "csw",
            "njf",
            "pjf",
            "nmf",
            "pmf",
            "nhfet",
            "phfet",
            "urc",
            "ltra",
            "nocontrol",
            "steplimit",
            "nosteplimit",
            "lininterp",
            "quadinterp",
            "mixedinterp",
            "truncnr",
            "truncdontcut",
            "sens_area",
            "sens_l",
            "sens_w",
        ]
        .iter()
        .any(|flag| token.is_keyword(flag))
    });
    let assigned = input
        .input
        .get(1)
        .is_some_and(|token| token.kind == TokenKind::Equals);
    if flag && !assigned {
        return Err(gap(input, "non-scalar model flags"));
    }
    let token = any
        .verify(|token: &Token| matches!(token.kind, TokenKind::Word))
        .parse_next(input)?;
    let name = token.text.to_ascii_lowercase();
    opt(equals).parse_next(input)?;
    if name == "level"
        && input
            .input
            .first()
            .is_some_and(super::expression::is_expression_token)
    {
        // INPfindLev reads the literal level; ModelCard::level cannot carry an
        // unevaluated expression, so refuse instead of defaulting it.
        return Err(gap(input, "expression-valued model level selectors"));
    }
    let value = cut_err(value).parse_next(input)?;
    Ok(named(&name, token.location.clone(), value))
}

fn invalid_parameter(input: &mut Input<'_>) -> Result<Option<ParameterAssignment>> {
    let token =
        peek(any.verify(|token: &Token| token.kind != TokenKind::RParen)).parse_next(input)?;
    if matches!(token.kind, TokenKind::Expression(_) | TokenKind::Quoted(_)) {
        return Err(gap(input, "non-scalar model parameters"));
    }
    Err(malformed(input, "expected a scalar model assignment"))
}

/// Upper-triangular CPL numeric-vector setter in physical conductor order.
/// Retains every source span and bracket expression for literalization.
fn cpl_vector(input: &mut Input<'_>, base: &str) -> Result<ParameterAssignment> {
    peek(any.verify(|token: &Token| {
        matches!(base, "cpl" | "cpllines")
            && token.kind == TokenKind::Word
            && ["r", "l", "g", "c"]
                .iter()
                .any(|name| token.is_keyword(name))
    }))
    .parse_next(input)?;
    let setter = any.parse_next(input)?;
    opt(equals).parse_next(input)?;
    let mut entries = Vec::new();
    while let Some(token) = input.input.first() {
        if token.kind == TokenKind::RParen
            || (token.kind == TokenKind::Word
                && ["r", "l", "g", "c", "length", "sections"]
                    .iter()
                    .any(|name| token.is_keyword(name)))
        {
            break;
        }
        if token.kind == TokenKind::Comma {
            any.parse_next(input)?;
            continue;
        }
        let component = cut_err(value).parse_next(input)?;
        let (source, expression) = match component {
            Value::Literal(token) => (token, None),
            Value::Expression(token, expression) => (token, Some(expression)),
        };
        entries.push(NumericVectorEntry {
            source: PositionedValue {
                text: source.text.clone(),
                location: source.location.clone(),
            },
            expression,
        });
        if entries.len() > 528 {
            return Err(malformed(
                input,
                "CPL matrix vector exceeds 32-conductor bound",
            ));
        }
    }
    if entries.is_empty() {
        return Err(malformed(
            input,
            "CPL R/L/G/C requires at least one numeric coefficient",
        ));
    }
    let text = entries
        .iter()
        .map(|entry| entry.source.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    Ok(ParameterAssignment {
        name: setter.text.to_ascii_lowercase(),
        value: text,
        kind: ParameterKind::NumericVector(entries),
        location: setter.location.clone(),
    })
}

/// XSPICE uses square brackets, which are otherwise valid parts of node
/// names. Parse delimiters only at the model-vector site; no global token
/// splitting changes bus node names. Numeric and expression entries retain
/// exact physical spans.
const fn boolean_literal(text: &str) -> Option<&'static str> {
    if text.eq_ignore_ascii_case("true") {
        Some("1")
    } else if text.eq_ignore_ascii_case("false") {
        Some("0")
    } else {
        None
    }
}

fn code_boolean(input: &mut Input<'_>, base: &str) -> Result<ParameterAssignment> {
    let Some(setter) = input.input.first() else {
        return Err(winnow::error::ErrMode::Backtrack(Failure(
            crate::primitives::SpiceError::circuit("expected boolean setter"),
        )));
    };
    let shape = SourceModel::from_base(base)
        .and_then(|schema| schema.parameter(&setter.text.to_ascii_lowercase()));
    let value_index = 1 + usize::from(
        input
            .input
            .get(1)
            .is_some_and(|token| token.kind == TokenKind::Equals),
    );
    let literal = input
        .input
        .get(value_index)
        .and_then(|token| boolean_literal(&token.text));
    if shape != Some(crate::devices::code_model::source::ParameterShape::Boolean)
        || literal.is_none()
    {
        return Err(winnow::error::ErrMode::Backtrack(Failure(
            crate::primitives::SpiceError::circuit("not a boolean literal setter"),
        )));
    }
    let setter = any.parse_next(input)?;
    opt(equals).parse_next(input)?;
    any.parse_next(input)?;
    Ok(ParameterAssignment {
        name: setter.text.to_ascii_lowercase(),
        value: literal.unwrap_or_default().into(),
        kind: ParameterKind::Scalar,
        location: setter.location.clone(),
    })
}

fn code_vector_entry(
    input: &mut Input<'_>,
    token: &Token,
    text: &str,
    prefix: usize,
    closed: bool,
    boolean: bool,
) -> Result<NumericVectorEntry> {
    if prefix == 0
        && !closed
        && matches!(token.kind, TokenKind::Expression(_) | TokenKind::Quoted(_))
    {
        let Value::Expression(source, expression) = value(input)? else {
            return Err(malformed(
                input,
                "invalid code-model coefficient expression",
            ));
        };
        return Ok(NumericVectorEntry {
            source: PositionedValue {
                text: source.text.clone(),
                location: source.location.clone(),
            },
            expression: Some(expression),
        });
    }
    let text = if boolean {
        boolean_literal(text).unwrap_or(text)
    } else {
        text
    };
    if text.contains(['[', ']'])
        || !crate::primitives::parse_spice_number(text).is_some_and(f64::is_finite)
    {
        return Err(malformed(input, "expected finite code-model coefficient"));
    }
    let position = token.position().advance(prefix).location();
    let entry = NumericVectorEntry {
        source: PositionedValue {
            text: text.to_owned(),
            location: position,
        },
        expression: None,
    };
    any.parse_next(input)?;
    Ok(entry)
}

fn code_vector(input: &mut Input<'_>, base: &str) -> Result<ParameterAssignment> {
    peek(any.verify(|token: &Token| {
        SourceModel::from_base(base)
            .and_then(|schema| schema.parameter(&token.text.to_ascii_lowercase()))
            .is_some_and(crate::devices::code_model::source::ParameterShape::vector)
    }))
    .parse_next(input)?;
    let setter = any.parse_next(input)?;
    let boolean = SourceModel::from_base(base)
        .and_then(|schema| schema.parameter(&setter.text.to_ascii_lowercase()))
        .is_some_and(crate::devices::code_model::source::ParameterShape::boolean);
    opt(equals).parse_next(input)?;
    cut_err(|input: &mut Input<'_>| {
        let mut entries = Vec::new();
        let mut first = true;
        let mut comma = false;
        loop {
            let tokens = input.input.clone();
            let token = tokens
                .first()
                .ok_or_else(|| malformed(input, "unterminated code-model coefficient vector"))?;
            let text = if first {
                token
                    .text
                    .strip_prefix('[')
                    .ok_or_else(|| malformed(input, "expected '[' at start of code-model vector"))?
            } else {
                token.text.as_str()
            };
            let prefix = usize::from(first);
            first = false;
            let (text, closed) = text
                .strip_suffix(']')
                .map_or((text, false), |text| (text, true));
            if text.is_empty() {
                any.parse_next(input)?;
                if closed {
                    if comma {
                        return Err(malformed(input, "missing coefficient after comma"));
                    }
                    break;
                }
                if prefix == 1 {
                    continue;
                }
                return Err(malformed(input, "empty code-model vector token"));
            }
            if token.kind == TokenKind::Comma {
                if comma || entries.is_empty() {
                    return Err(malformed(input, "misplaced coefficient comma"));
                }
                any.parse_next(input)?;
                comma = true;
                continue;
            }
            entries.push(code_vector_entry(
                input, token, text, prefix, closed, boolean,
            )?);
            comma = false;
            if closed {
                break;
            }
        }
        let value = entries
            .iter()
            .map(|entry| entry.source.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(ParameterAssignment {
            name: setter.text.to_ascii_lowercase(),
            value,
            kind: ParameterKind::NumericVector(entries),
            location: setter.location.clone(),
        })
    })
    .parse_next(input)
}
