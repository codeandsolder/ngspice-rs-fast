//! Source syntax from inp2v.c/inp2i.c, `INPgetValue` in inpgval.c and
//! VSRCparam/ISRCparam. This is not VSRCload/ISRCload time evaluation.

use winnow::Parser as _;
use winnow::combinator::{alt, cut_err, opt};
use winnow::token::any;

use crate::netlist::ast::{
    FunctionWaveform, NumericVectorEntry, ParameterAssignment, ParameterKind, PulseWaveform,
    PwlPoint, SourceWaveform, WaveformKind,
};

use crate::netlist::token::Token;

use super::grammar::{Input, Result, keyword};
use super::syntax::{Value, equals, malformed, named, value};
use super::vector::{numeric, positioned};

/// The vector-valued transient setters of `vsrc.c`/`isrc.c` (`IF_REALVEC`).
pub(super) fn parameters(input: &mut Input<'_>) -> Result<Vec<ParameterAssignment>> {
    let (token, kind) = any
        .verify_map(|token: &Token| {
            WaveformKind::from_keyword(&token.text).map(|kind| (token, kind))
        })
        .parse_next(input)?;
    cut_err(|input: &mut Input<'_>| {
        opt(equals).parse_next(input)?;
        // C reads at most eight PULSE coefficients, six SIN/EXP and eight
        // SFFM/AM ones; more would be silently ignored there, so they are a
        // syntax error here. PWL is bounded to 2048 pairs. Runtime
        // restrictions (times, slopes, defaults) are not syntax.
        let vector = numeric(input, kind.maximum_fields(), value)?;
        if vector.values.len() < 2 {
            return Err(malformed(
                input,
                "waveform requires at least two numeric fields",
            ));
        }
        let waveform = match kind {
            WaveformKind::Function(function) => {
                SourceWaveform::Function(Box::new(FunctionWaveform {
                    function,
                    values: vector.values.iter().map(coefficient).collect(),
                }))
            }
            WaveformKind::Stochastic(function) => SourceWaveform::Stochastic {
                function,
                values: vector.values.iter().map(coefficient).collect(),
            },
            WaveformKind::Pulse => {
                let value = |index: usize| vector.values.get(index).map(coefficient);
                SourceWaveform::Pulse(Box::new(PulseWaveform {
                    initial: coefficient(&vector.values[0]),
                    pulsed: coefficient(&vector.values[1]),
                    delay: value(2),
                    rise: value(3),
                    fall: value(4),
                    width: value(5),
                    period: value(6),
                    count: value(7),
                }))
            }
            WaveformKind::Pwl => {
                let (pairs, remainder) = vector.values.as_chunks::<2>();
                if !remainder.is_empty() {
                    return Err(malformed(
                        input,
                        "PWL requires strictly paired time/value fields",
                    ));
                }
                SourceWaveform::Pwl(
                    pairs
                        .iter()
                        .map(|[time, value]| PwlPoint {
                            time: coefficient(time),
                            value: coefficient(value),
                        })
                        .collect(),
                )
            }
        };
        Ok(vec![ParameterAssignment {
            name: token.text.to_ascii_lowercase(),
            value: vector.text,
            kind: ParameterKind::Waveform(waveform),
            location: token.location.clone(),
        }])
    })
    .parse_next(input)
}

/// PWL repeat/delay scalars (`IP("r", VSRC_R)`, `IP("td", VSRC_TD)` in
/// `vsrc.c`/`isrc.c`). They stay ordered setters: whether they apply depends
/// on the waveform set before them, which device elaboration checks.
pub(super) fn pwl_options(input: &mut Input<'_>) -> Result<Vec<ParameterAssignment>> {
    let token = alt((keyword("r"), keyword("td"))).parse_next(input)?;
    let (_, value) = cut_err((opt(equals), value)).parse_next(input)?;
    Ok(vec![named(
        &token.text.to_ascii_lowercase(),
        token.location.clone(),
        value,
    )])
}

fn coefficient(value: &Value<'_>) -> NumericVectorEntry {
    NumericVectorEntry {
        source: positioned(value.token()),
        expression: match value {
            Value::Literal(_) => None,
            Value::Expression(_, expression) => Some(expression.clone()),
        },
    }
}
