//! Lossless transmission-line grammar: `src/spicelib/parser/inp2t.c` and
//! the `TRApTable` of `src/spicelib/devices/tra/tra.c`.
//!
//! ```text
//! Tname p1 n1 p2 n2 [z0|zo [=] v] [td [=] v] [f [=] v] [nl [=] v]
//!       [v1|i1|v2|i2 [=] v] [rel [=] v] [abs [=] v] [ic [=] v1[,i1[,v2[,i2]]]]
//! ```
//!
//! `INP2T` reads four nodes and hands the rest to `INPdevParse`: setters are
//! kept in written order (the device applies them in turn, last wins), the
//! `=` is optional as in C's tokenizer, and `ic=` is an `IF_REALVEC` of one
//! to four values stored as [`ParameterKind::InitialConditions`] components
//! `v1`, `i1`, `v2`, `i2`. Values are finite literals or `{...}`
//! expressions (evaluated by elaboration); the `ic=` vector takes literals.
//!
//! Explicit gaps: a leading positional value (C's `waslead`, silently
//! ignored by `INP2T`) is refused; any other word is a parse error ("unknown
//! parameter" in C). Requiring `z0` and validating values belongs to the
//! device (`crate::devices::tline`).

use crate::primitives::SpiceError;
use winnow::Parser as _;
use winnow::combinator::cut_err;
use winnow::error::ErrMode;

use crate::netlist::ast::DeviceInstance;
use crate::netlist::token::Token;

use super::grammar::{Failure, Input, ParsedCard, Result, location};
use super::line_grammar;
use super::syntax::named_device;

/// Scalar setters allowed by this device's C parameter table.
const SCALARS: [&str; 11] = [
    "z0", "zo", "td", "f", "nl", "v1", "v2", "i1", "i2", "rel", "abs",
];

pub(super) fn tline_card(input: &mut Input<'_>) -> Result<ParsedCard> {
    let (instance, _) = named_device("t").parse_next(input)?;
    cut_err(move |input: &mut Input<'_>| body(input, instance)).parse_next(input)
}

fn body(input: &mut Input<'_>, instance: &Token) -> Result<ParsedCard> {
    let nodes = line_grammar::four_terminals(input)?;
    if input
        .input
        .first()
        .is_some_and(|token| token.number().is_some())
    {
        return Err(ErrMode::Cut(Failure(SpiceError::Unsupported {
            feature: "a leading value after the transmission-line nodes (C silently ignores it)"
                .into(),
            location: Some(location(input)),
        })));
    }
    let parameters = line_grammar::parameters(
        input,
        &SCALARS,
        "transmission-line",
        "z0, zo, td, f, nl, v1, i1, v2, i2, rel, abs or ic",
    )?;
    Ok(ParsedCard::Device(DeviceInstance {
        name: instance.text.to_ascii_lowercase(),
        designator: 't',
        nodes,
        model: None,
        parameters,
        location: input.state.card.location.clone(),
    }))
}
