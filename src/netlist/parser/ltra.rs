//! Lossy transmission-line grammar: `src/spicelib/parser/inp2o.c` and the
//! `LTRApTable` of `src/spicelib/devices/ltra/ltra.c`.
//!
//! ```text
//! Oname p1 n1 p2 n2 model [v1|i1|v2|i2 [=] v] [ic [=] v1[,i1[,v2[,i2]]]]
//! ```
//!
//! `INP2O` reads four nodes and a model name, then hands the rest to
//! `INPdevParse`: setters are kept in written order (last wins), the `=` is
//! optional as in C's tokenizer, and `ic=` is an `IF_REALVEC` of one to four
//! values stored as [`ParameterKind::InitialConditions`] components `v1`,
//! `i1`, `v2`, `i2` (`LTRAparam`'s fallthrough order). Values are finite
//! literals or `{...}` expressions (evaluated by elaboration); the `ic=`
//! vector takes literals.
//!
//! The model name is required: a missing one makes `INPgetMod` fail ("unable
//! to find definition of model"), after which C's default `O` model has no
//! length and setup fails. Explicit gaps: a leading positional value (C's
//! `waslead`, silently ignored by `INP2O`) is refused; any other word is a
//! parse error ("unknown parameter" in C). Model lookup and validation belong
//! to the device (`crate::devices::ltra`).

use crate::primitives::SpiceError;
use winnow::Parser as _;
use winnow::combinator::cut_err;
use winnow::error::ErrMode;

use crate::netlist::ast::DeviceInstance;
use crate::netlist::token::Token;

use super::grammar::{Failure, Input, ParsedCard, Result, location};
use super::line_grammar;
use super::syntax::{malformed, name, named_device};

/// Scalar setters allowed by this device's C parameter table.
const SCALARS: [&str; 4] = ["v1", "v2", "i1", "i2"];

pub(super) fn ltra_card(input: &mut Input<'_>) -> Result<ParsedCard> {
    let (instance, _) = named_device("o").parse_next(input)?;
    cut_err(move |input: &mut Input<'_>| body(input, instance)).parse_next(input)
}

fn body(input: &mut Input<'_>, instance: &Token) -> Result<ParsedCard> {
    let nodes = line_grammar::four_terminals(input)?;
    if input.input.is_empty() {
        return Err(malformed(
            input,
            "expected the lossy-line model name (C: unable to find definition of model)",
        ));
    }
    let model = name("lossy-line model name").parse_next(input)?;
    if input
        .input
        .first()
        .is_some_and(|token| token.number().is_some())
    {
        return Err(ErrMode::Cut(Failure(SpiceError::Unsupported {
            feature: "a leading value after the lossy-line model (C silently ignores it)".into(),
            location: Some(location(input)),
        })));
    }
    let parameters =
        line_grammar::parameters(input, &SCALARS, "lossy-line", "v1, i1, v2, i2 or ic")?;
    Ok(ParsedCard::Device(DeviceInstance {
        name: instance.text.to_ascii_lowercase(),
        designator: 'o',
        nodes,
        model: Some(model.text.to_ascii_lowercase()),
        parameters,
        location: input.state.card.location.clone(),
    }))
}
