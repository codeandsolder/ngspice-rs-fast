//! Shared positioned-token primitives for scalar device and model grammars.

use crate::primitives::SpiceError;
use winnow::Parser as _;
use winnow::combinator::{cut_err, opt, peek, repeat};
use winnow::error::ErrMode;
use winnow::token::any;

use crate::primitives::SourceLoc;

use crate::netlist::ast::{ParameterAssignment, ParameterKind};
use crate::netlist::expr::ParameterExpression;
use crate::netlist::token::{Token, TokenKind};

use super::grammar::{Failure, Input, Result, gap};

pub(super) fn name<'a>(
    expected: &'static str,
) -> impl winnow::Parser<Input<'a>, &'a Token, ErrMode<Failure>> {
    any.verify(|token: &Token| token.is_name_like())
        .context(expected)
}

/// The ASCII-folded first character of a token. Linear devices historically
/// recognize the prefix regardless of token kind; other instance grammars
/// additionally require a name-like token.
pub(super) fn token_designator(token: &Token) -> Option<char> {
    token.text.chars().next().map(|c| c.to_ascii_lowercase())
}

/// Match a name-like device token and return its already checked designator.
/// The verification failure backtracks; body parsers establish their own cut.
pub(super) fn named_device<'a>(
    designators: &'static str,
) -> impl winnow::Parser<Input<'a>, (&'a Token, char), ErrMode<Failure>> {
    any.verify_map(move |token: &'a Token| {
        let designator = token_designator(token)?;
        (token.is_name_like() && designators.contains(designator)).then_some((token, designator))
    })
}

/// A node or instance name after the separators skipped by C's token readers.
/// This permissive rule is distinct from the balanced numeric-vector grammar.
pub(super) fn terminal_name<'a>(
    input: &mut Input<'a>,
    expected: &'static str,
) -> Result<&'a Token> {
    skip_terminal_punctuation(input)?;
    name(expected).parse_next(input)
}

pub(super) fn skip_terminal_punctuation(input: &mut Input<'_>) -> Result<()> {
    repeat(0.., terminal_punctuation).parse_next(input)
}

fn terminal_punctuation<'a>(input: &mut Input<'a>) -> Result<&'a Token> {
    any.verify(|token: &Token| {
        matches!(
            token.kind,
            TokenKind::LParen | TokenKind::RParen | TokenKind::Comma
        )
    })
    .parse_next(input)
}

pub(super) fn punctuation<'a>(
    kind: TokenKind,
) -> impl winnow::Parser<Input<'a>, &'a Token, ErrMode<Failure>> {
    any.verify(move |token: &Token| token.kind == kind)
}

pub(super) fn canonical_node(token: &Token, auto_gnd: bool) -> String {
    // Node-only aliasing is the port's AST contract. inpcom.c's
    // inp_fix_gnd_name() replaces more broadly in card text; deliberately keep
    // model/value/instance spelling unchanged here (see ARCHITECTURE.md).
    let name = token.text.to_ascii_lowercase();
    if auto_gnd && name == "gnd" {
        "0".to_owned()
    } else {
        name
    }
}

pub(super) fn equals<'a>(input: &mut Input<'a>) -> Result<&'a Token> {
    any.verify(|token: &Token| token.kind == TokenKind::Equals)
        .parse_next(input)
}

/// A scalar value site: a finite literal or a parsed (unevaluated) `{...}`.
pub(super) enum Value<'a> {
    Literal(&'a Token),
    Expression(&'a Token, Box<ParameterExpression>),
}

impl Value<'_> {
    pub(super) const fn token(&self) -> &Token {
        match self {
            Self::Literal(token) | Self::Expression(token, _) => token,
        }
    }
}

/// Only a numeric prefix or a brace expression claims an optional positional
/// slot. Once claimed, overflow or a malformed expression is a committed
/// syntax error, not a missing optional value. Bare names are never claimed:
/// they may be model names (C does not substitute unbraced names in device
/// cards).
pub(super) fn leading_value<'a>(input: &mut Input<'a>) -> Result<Value<'a>> {
    peek(any.verify(|token: &Token| {
        token.number().is_some() || super::expression::is_expression_token(token)
    }))
    .parse_next(input)?;
    cut_err(value).parse_next(input)
}

/// A finite numeric literal or a brace expression.
pub(super) fn value<'a>(input: &mut Input<'a>) -> Result<Value<'a>> {
    if let Some(token) =
        opt(any.verify(super::expression::is_expression_token)).parse_next(input)?
    {
        let expression = super::expression::from_brace_token(token)
            .map_err(|error| ErrMode::Cut(Failure(error)))?;
        return Ok(Value::Expression(token, Box::new(expression)));
    }
    literal.map(Value::Literal).parse_next(input)
}

/// A strictly numeric literal. Expressions are a gap at sites that have not
/// adopted [`value`] (waveform and IC-vector components).
pub(super) fn literal<'a>(input: &mut Input<'a>) -> Result<&'a Token> {
    if input.input.first().is_some_and(|token| {
        matches!(
            token.kind,
            TokenKind::Word | TokenKind::Expression(_) | TokenKind::Quoted(_)
        )
    }) {
        return Err(gap(
            input,
            "parameter expressions, model references or extended numeric syntax",
        ));
    }
    any.verify(|token: &Token| token.number().is_some_and(f64::is_finite))
        .context("a finite numeric literal")
        .parse_next(input)
}

/// Positional/leading assignment located at its value token.
pub(super) fn assignment(name: &str, value: Value<'_>) -> ParameterAssignment {
    let location = value.token().location.clone();
    named(name, location, value)
}

/// Named assignment located at its setter keyword.
pub(super) fn named(name: &str, location: SourceLoc, value: Value<'_>) -> ParameterAssignment {
    let text = value.token().text.clone();
    ParameterAssignment {
        name: name.to_owned(),
        value: text,
        kind: match value {
            Value::Literal(_) => ParameterKind::Scalar,
            Value::Expression(_, expression) => ParameterKind::Expression(expression),
        },
        location,
    }
}

pub(super) fn malformed(input: &Input<'_>, message: &str) -> ErrMode<Failure> {
    ErrMode::Cut(Failure(SpiceError::parse(
        super::grammar::location(input),
        message,
    )))
}

#[cfg(test)]
mod tests {
    use super::{named_device, token_designator};
    use crate::netlist::parser::{behavioural, grammar::Context};
    use crate::netlist::{RawCard, TokenKind, source::parse_deck_text};
    use crate::primitives::SpiceError;
    use std::{collections::BTreeSet, path::Path};
    use winnow::{Parser as _, error::ErrMode, stream::TokenSlice};

    fn card(text: &str) -> RawCard {
        let deck = parse_deck_text(Path::new("prefix.cir"), &format!("Test\n{text}\n"));
        RawCard::parse(&deck.lines[0]).unwrap()
    }

    fn input<'a>(card: &'a RawCard, declared_models: &'a BTreeSet<String>) -> super::Input<'a> {
        super::Input {
            input: TokenSlice::new(&card.tokens),
            state: Context {
                card,
                auto_gnd: true,
                declared_models,
            },
        }
    }

    #[test]
    fn named_device_backtracks_and_returns_the_original_token() {
        let card = card("D1 a 0 model");
        let models = BTreeSet::new();
        let mut stream = input(&card, &models);
        let Err(ErrMode::Backtrack(failure)) = named_device("efgh").parse_next(&mut stream) else {
            panic!("a different device family must backtrack");
        };
        let SpiceError::Parse { location, .. } = failure.0 else {
            panic!("a prefix mismatch must retain the parser error category");
        };
        assert_eq!(location, card.tokens[0].location);
        assert_eq!(stream.input.first(), Some(&card.tokens[0]));
        let (token, designator) = named_device("d").parse_next(&mut stream).unwrap();
        assert!(std::ptr::eq(token, &card.tokens[0]));
        assert_eq!(designator, 'd');
        assert_eq!(stream.input.first(), Some(&card.tokens[1]));
    }

    #[test]
    fn named_device_preserves_the_token_kind_filter() {
        let mut card = card("D1 a 0 model");
        card.tokens[0].kind = TokenKind::Quoted("D1".to_owned());
        let models = BTreeSet::new();
        let mut stream = input(&card, &models);
        assert_eq!(token_designator(&card.tokens[0]), Some('d'));
        assert!(matches!(
            named_device("d").parse_next(&mut stream),
            Err(ErrMode::Backtrack(_))
        ));
        assert_eq!(stream.input.first(), Some(&card.tokens[0]));
    }

    #[test]
    fn behavioural_prefix_requires_a_suffix_and_body_errors_cut() {
        let models = BTreeSet::new();
        let bare = card("B a 0 v=1");
        let mut stream = input(&bare, &models);
        assert!(matches!(
            behavioural::behavioural_card(&mut stream),
            Err(ErrMode::Backtrack(_))
        ));
        assert_eq!(stream.input.first(), Some(&bare.tokens[0]));
        let named = card("B1 a");
        let mut stream = input(&named, &models);
        assert!(matches!(
            behavioural::behavioural_card(&mut stream),
            Err(ErrMode::Cut(_))
        ));
    }
}
