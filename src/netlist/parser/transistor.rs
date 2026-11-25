//! Bounded Q/M instance grammars from `inp2q.c`, `inp2m.c`, `bjt/bjt.c`
//! and `mos1/mos1.c`. The declaration index follows C's earliest-model terminal
//! scan, including forward references and node/model-name collisions. It does
//! not select or validate a model backend or inject ports/defaults. An `M`
//! card also recognises a binned reference: a name with no declaration of its
//! own but a declared `<name>.<digits>` model, as `inp2m.c` falls back to
//! `INPgetModBin` and `subckt.c` uses `model_name_match`. Which bin (if any)
//! applies is decided later by `devices::models::ModelResolver`.

use winnow::Parser as _;
use winnow::combinator::{alt, cut_err, opt, peek, repeat};
use winnow::token::any;

use crate::netlist::ast::{DeviceInstance, ParameterAssignment};
use crate::netlist::token::{Token, TokenKind};

use super::grammar::{Input, ParsedCard, Result, gap};
use super::syntax::{
    assignment, canonical_node, equals, leading_value, malformed, name, named, named_device, value,
};

type Connections<'a> = (Vec<&'a Token>, &'a Token);

pub(super) fn transistor_card<'a>(input: &mut Input<'a>) -> Result<ParsedCard> {
    let (instance, designator) = named_device("qm").parse_next(input)?;
    let (connections, parameters) = cut_err(|input: &mut Input<'a>| {
        let connections = if designator == 'q' {
            bjt_connections(input)?
        } else {
            mos_connections(input)?
        };
        let parameters = parameters(input, designator)?;
        Ok((connections, parameters))
    })
    .parse_next(input)?;
    let (nodes, model) = connections;
    Ok(ParsedCard::Device(DeviceInstance {
        name: instance.text.to_ascii_lowercase(),
        designator,
        nodes: nodes
            .into_iter()
            .map(|node| canonical_node(node, input.state.auto_gnd))
            .collect(),
        model: Some(model.text.to_ascii_lowercase()),
        parameters,
        location: input.state.card.location.clone(),
    }))
}

fn is_declared_model(input: &Input<'_>, token: &Token) -> bool {
    token.is_name_like()
        && input
            .state
            .declared_models
            .contains(&token.text, input.state.card.designator() == Some('m'))
}

fn declared_model<'a>(input: &mut Input<'a>) -> Result<&'a Token> {
    let token = peek(name("declared model name")).parse_next(input)?;
    if !is_declared_model(input, token) {
        return Err(malformed(
            input,
            "expected a model declared before .end (forward declarations are allowed)",
        ));
    }
    // inpcom.c's inp_get_number_terminals assumes BJT model names contain an
    // alphabetic character, so a purely numeric name is not a supported Q form.
    if input.state.card.designator() == Some('q')
        && !token.text.bytes().any(|byte| byte.is_ascii_alphabetic())
    {
        return Err(malformed(
            input,
            "BJT model name must contain an ASCII alphabetic character",
        ));
    }
    // Numeric-looking model references are outside the bounded identifier
    // grammar. In particular, ngspice-47+ rejects Q models named 123 or 123n
    // during its front-end preprocessing, before terminal binding.
    if token.number().is_some() {
        return Err(gap(input, "numeric-looking transistor model names"));
    }
    any.parse_next(input)
}

fn bjt_connections<'a>(input: &mut Input<'a>) -> Result<Connections<'a>> {
    let (collector, base, emitter) = (
        name("collector terminal"),
        name("base terminal"),
        name("emitter terminal"),
    )
        .parse_next(input)?;
    let mut nodes = vec![collector, base, emitter];
    let candidate = peek(name("BJT model name or substrate terminal")).parse_next(input)?;
    // INP2Q recognises a declared model as soon as the first three ports exist.
    if !is_declared_model(input, candidate) {
        nodes.push(name("substrate terminal").parse_next(input)?);
        // VBIC's fifth/thermal terminal is deliberately outside this grammar.
        if input
            .input
            .get(1)
            .is_some_and(|token| is_declared_model(input, token))
            && input
                .input
                .first()
                .is_some_and(|token| !is_declared_model(input, token))
        {
            return Err(gap(input, "BJT fifth/thermal terminal syntax"));
        }
    }
    let model = declared_model(input)?;
    // Keep an omitted substrate omitted; INP2Q's implicit ground is elaboration.
    Ok((nodes, model))
}

fn mos_connections<'a>(input: &mut Input<'a>) -> Result<Connections<'a>> {
    let (drain, gate, source) = (
        name("drain terminal"),
        name("gate terminal"),
        name("source terminal"),
    )
        .parse_next(input)?;
    let candidate = peek(name("bulk terminal")).parse_next(input)?;
    // C scans for a declared model immediately after the third port for
    // VDMOS; do not manufacture a bulk terminal. Family validation happens
    // after the entire model catalog is resolved, so an ordinary MOS with
    // omitted bulk is still a loud elaboration error, not reinterpreted.
    if is_declared_model(input, candidate) {
        if !input.state.declared_models.is_vdmos(&candidate.text) {
            return Err(malformed(input, "MOS instance requires a bulk terminal"));
        }
        return Ok((vec![drain, gate, source], declared_model(input)?));
    }
    let bulk = name("bulk terminal").parse_next(input)?;
    let candidate = peek(name("MOS model name")).parse_next(input)?;
    if !is_declared_model(input, candidate) {
        // The VDMOS self-heating syntax has two additional explicit thermal
        // terminals after D/G/S, with no bulk node. It is resolved against
        // the model family during elaboration, not guessed by token count.
        if input
            .input
            .get(1)
            .is_some_and(|next| input.state.declared_models.is_vdmos(&next.text))
        {
            let case = name("VDMOS case thermal terminal").parse_next(input)?;
            let model = declared_model(input)?;
            return Ok((vec![drain, gate, source, bulk, case], model));
        }
        // C's SOI/HiSIM/thermal variants can take 6–7 ports; retain a loud gap.
        if input
            .input
            .iter()
            .skip(1)
            .take(3)
            .any(|token| is_declared_model(input, token))
        {
            return Err(gap(input, "MOS extra/thermal terminal syntax"));
        }
    }
    let model = declared_model(input)?;
    Ok((vec![drain, gate, source, bulk], model))
}

fn parameters(input: &mut Input<'_>, designator: char) -> Result<Vec<ParameterAssignment>> {
    let leading = if designator == 'q' {
        opt(leading_value).parse_next(input)?
    } else {
        None
    };
    let mut parameters: Vec<ParameterAssignment> = repeat(
        0..,
        alt((
            super::flags::transistor_instance,
            super::ic::vector,
            |input: &mut Input<'_>| scalar_assignment(input, designator),
            invalid_parameter,
        )),
    )
    .parse_next(input)?;
    if let Some(area) = leading {
        // INP2Q, like INP2D, applies the leading area after INPdevParse.
        parameters.push(assignment("area", area));
    }
    Ok(parameters)
}

fn scalar_assignment(input: &mut Input<'_>, designator: char) -> Result<ParameterAssignment> {
    let token = any
        .verify(move |token: &Token| {
            let key = token.text.to_ascii_lowercase();
            match designator {
                'q' => matches!(
                    key.as_str(),
                    "area" | "areab" | "areac" | "m" | "icvbe" | "icvce" | "temp" | "dtemp"
                ),
                _ => matches!(
                    key.as_str(),
                    "m" | "l"
                        | "w"
                        | "ad"
                        | "as"
                        | "pd"
                        | "ps"
                        | "nrd"
                        | "nrs"
                        | "icvds"
                        | "icvgs"
                        | "icvbs"
                        | "temp"
                        | "dtemp"
                        // bsim3/b3.c input-only syntax, not backend availability.
                        | "nqsmod"
                        | "acnqsmod"
                        | "geo"
                        | "delvto"
                        | "mulu0"
                        // bsim4/b4.c input-only syntax, likewise.
                        | "delvt0"
                        | "geomod"
                        | "min"
                        | "nf"
                        | "ngcon"
                        | "rbdb"
                        | "rbsb"
                        | "rbpb"
                        | "rbps"
                        | "rbpd"
                        | "rbodymod"
                        | "rgatemod"
                        | "rgeomod"
                        | "sa"
                        | "sb"
                        | "sc"
                        | "sd"
                        | "sca"
                        | "scb"
                        | "scc"
                        | "trnqsmod"
                        | "wnflag"
                        | "xgw"
                        | "mult_fn"
                        | "mult_i"
                        | "mult_q"
                ),
            }
        })
        .parse_next(input)?;
    let (_, value) = cut_err((opt(equals), value)).parse_next(input)?;
    Ok(named(
        &token.text.to_ascii_lowercase(),
        token.location.clone(),
        value,
    ))
}

fn invalid_parameter(input: &mut Input<'_>) -> Result<ParameterAssignment> {
    let token = peek(any).parse_next(input)?;
    if input.state.card.designator() == Some('m')
        && (token.number().is_some() || super::expression::is_expression_token(token))
    {
        return Err(malformed(
            input,
            "no unlabeled parameter permitted on MOSFET",
        ));
    }
    if matches!(
        token.kind,
        TokenKind::Equals | TokenKind::LParen | TokenKind::RParen | TokenKind::Comma
    ) {
        return Err(malformed(
            input,
            "expected a named scalar transistor parameter",
        ));
    }
    Err(gap(
        input,
        "unsupported transistor flags or additional/non-scalar parameters",
    ))
}
