//! Normalized deck writer: [`Netlist`] in, SPICE text out (GitHub #20).
//!
//! The writer serializes the **raw, unevaluated, unflattened** AST. It is not a
//! debug dump (see `spice-rs parse`), not an evaluated/expanded deck (that is
//! elaboration, #15/#18) and not a byte-exact source reproducer. Its contract is
//! *semantic*: re-parsing the output through the production [`Parser`] yields a
//! netlist that is [`semantic_eq`](crate::netlist::semantic::semantic_eq) to the input,
//! and writing that netlist again yields the identical text (fixed point).
//!
//! # Normalization contract
//!
//! Lexical:
//! - One logical card per line, no continuation lines, no comments, no blank
//!   lines; `\n` terminators; 2-space indentation per `.subckt` nesting level.
//! - Line 1 is the title, verbatim (it may not contain a line break or end in
//!   whitespace). Identifiers are written as stored (the parser lowercases
//!   them; `gnd` aliasing is already applied, so re-parse with the same
//!   [`Parser`] configuration, in particular `auto_gnd`).
//! - Numeric and expression spellings are preserved exactly as stored
//!   (`1meg`, `5V`, `{ a + b }`); numbers are never re-formatted.
//! - Directives use their canonical names: `.options`, `.include`, `.lib`.
//!
//! Card order: `Netlist::cards`/`Subcircuit::cards` is authoritative. Cards are
//! written in that order (so forward model references, subcircuit placement and
//! directive order are unchanged); the typed vectors are only addressed by the
//! card indexes. Directives are never reordered or hoisted. The root `.end` is
//! written only if the deck had one.
//!
//! Parameter application order: every device/model/subcircuit assignment list is
//! written in stored (application) order, duplicates retained, so that C's
//! "later setter wins" semantics survive. The *position* a value had in the
//! source is not meaningful beyond that order, so values are written by name:
//! - R/C/L: a first scalar of the primary name (`resistance`, ...) on a
//!   model-less instance is written positionally (`r1 a b 1k tc1=1m`);
//!   otherwise every value is a named setter (`rpost a 0 rm resistance=5k
//!   resistance=4k`), which is how C's post-model positional value (applied
//!   last) is kept last. Likewise D/Q's leading area is written as `area=...`
//!   at its stored (last) position, and V/I's leading DC as `dc <value>`.
//! - V/I: `dc v`, `ac mag phase` (explicit defaults written out),
//!   `pulse(...)`/`pwl(...)`/`sin(...)` (or `sine`)/`exp(...)`/`sffm(...)`/
//!   `am(...)` and PWL `td=`/`r=` in stored order. Optional fields that
//!   were omitted stay omitted; PWL pairs and other fields are space separated
//!   inside one pair of parentheses (the stored vector text is re-spelled, so
//!   it is excluded from semantic comparison).
//! - E/F/G/H (linear gain forms): `e1 n+ n- nc+ nc- <gain>`,
//!   `f1 n+ n- vname <gain>`; parentheses and the HSPICE `vcvs`-style keyword
//!   are not written. A gain stored last after `m=` setters (C's leading value,
//!   applied after the named setters) is written positionally before them; a
//!   gain stored first is written as `gain=`. Other orders are refused.
//! - D/Q/M: bare flags (`off`), `name=value` scalars, `ic=(a,b[,c])` vectors
//!   with omitted trailing components omitted. An omitted Q substrate stays
//!   omitted.
//! - `.model`: `.model name base(flag name=value ...)`; level spelling is
//!   whatever the `level=` assignment says.
//! - `.subckt name terminals [params: name=value ...]` ... `.ends name`;
//!   X cards are `xname nodes target name=value ...` without a `params:` marker
//!   (the marker is not represented in the AST; both spellings parse the same).
//! - `.param name=text ...`: expression text is preserved (braced expressions as
//!   `{text}`); the writer verifies at write time that the text re-parses to the
//!   stored syntax tree, and that each `.param` card re-parses to the same
//!   assignments, instead of re-printing from the tree. Parentheses are
//!   therefore exactly those of the source, which are always sufficient.
//! - `.func name(p1,p2) body`: the body keeps its delimiters (`{...}`, `'...'`
//!   or bare) and text, verified to re-parse like `.param`.
//! - Single-quoted expressions (`'a*2'`) keep their quotes everywhere; C's
//!   `inp_change_quotes()` makes them identical to braces.
//! - Analysis cards: `.name` plus the stored argument tokens joined with single
//!   spaces (no space around `(`, `)`, `=` or before `,`). A `.tran` `uic` flag
//!   is written after the positional arguments, before any `name=value` options.
//! - `.ic`/`.nodeset`: one card per stored card, `v(node)=value` entries in
//!   order (values keep their literal spelling or braced expression text).
//!
//! # Includes
//!
//! A parsed [`Netlist`] built with `parse_file` contains resolved include/lib
//! content in its typed vectors and `cards`, marked by a non-empty
//! `include_chain`. The writer emits the **directive** (`.include path`,
//! `.lib path section`) and **skips** all cards whose chain is non-empty:
//! inlining would silently flatten source structure, break source-relative path
//! semantics and duplicate library files into every deck. The directive's path
//! is written with its original spelling (quotes included) when that spelling
//! still decodes to the stored `path`; otherwise `path` is written bare when it
//! is a plain word and double-quoted with `\\`/`\"` escapes otherwise. Paths are
//! never rewritten: a written deck must live where its relative includes
//! resolve, and re-parsing it with `parse_file` re-reads the same files.
//! A scope that opens inside an included file but closes outside it (or the
//! reverse) cannot be represented and is an error.
//!
//! # Source locations
//!
//! Locations (file, line, column, spans, joined card text) describe a specific
//! text and always differ on re-parse. Compare with
//! [`semantic_eq`](crate::netlist::semantic::semantic_eq)/
//! [`semantic_diff`](crate::netlist::semantic::semantic_diff), which ignore them.
//!
//! # Errors
//!
//! [`write_netlist`] returns [`SpiceError::Unsupported`] (with the offending
//! card's location when known) for anything that would not re-parse to the same
//! semantics: unknown designators or parameter names, kinds a card cannot carry,
//! non-contiguous PULSE fields, inconsistent expression text, inconsistent card
//! indexes, names that are not single tokens or would be treated as comments,
//! and scopes crossing include boundaries.

use std::fmt::Write as _;

use crate::primitives::{AnalysisKind, SourceLoc, SpiceError, SpiceResult, parse_spice_number};

use crate::netlist::ast::{
    AnalysisCard, DeviceInstance, FuncCard, FuncSpelling, GlobalCard, IncludeDirective, ModelCard,
    Netlist, NodeHintCard, NodeHintValue, OptionCard, ParamCard, ParameterAssignment,
    ParameterKind, PositionedValue, ScopeEntry, ScopeView, ScopedCardKind, SourceWaveform,
    Subcircuit,
};
use crate::netlist::behavioural::forms::{
    B_SETTERS, NonlinearForm, PolyControls, PolyForm, TableForm, TableInput,
};
use crate::netlist::expr::ParameterExpression;
use crate::netlist::semantic::{bexpr_form, expr_form};
use crate::netlist::source::{LogicalLine, parse_deck_text, strip_comment};
use crate::netlist::token::{Token, TokenKind, tokenize};
use crate::netlist::{Parser, parse_parameter_expression};

/// Serializes `netlist` as a normalized deck. See the [module documentation](self)
/// for the contract.
///
/// # Errors
///
/// [`SpiceError::Unsupported`] for constructs that cannot be written so that
/// they re-parse to the same semantics.
pub fn write_netlist(netlist: &Netlist) -> SpiceResult<String> {
    if netlist.title.contains(['\n', '\r']) || netlist.title != netlist.title.trim_end() {
        return Err(refuse(
            "the title must be one line without trailing whitespace",
            Some(&netlist.location),
        ));
    }
    let mut writer = Writer { out: String::new() };
    writer.out.push_str(&netlist.title);
    writer.out.push('\n');
    writer.scope(ScopeView::from(netlist), 0, false)?;
    Ok(writer.out)
}

fn refuse(what: impl Into<String>, location: Option<&SourceLoc>) -> SpiceError {
    SpiceError::Unsupported {
        feature: format!("netlist writer: {}", what.into()),
        location: location.cloned(),
    }
}

struct Writer {
    out: String,
}

impl Writer {
    fn line(&mut self, depth: usize, text: &str, location: &SourceLoc) -> SpiceResult<()> {
        if text.contains(['\n', '\r']) || strip_comment(text, false) != text {
            return Err(refuse(
                format!("card would not survive comment/line handling: {text:?}"),
                Some(location),
            ));
        }
        for _ in 0..depth {
            self.out.push_str("  ");
        }
        self.out.push_str(text);
        self.out.push('\n');
        Ok(())
    }

    fn scope(&mut self, scope: ScopeView<'_>, depth: usize, body: bool) -> SpiceResult<()> {
        scope
            .validate_order()
            .map_err(|error| refuse(error.to_string(), None))?;
        for card in scope.cards {
            // Cards read from an include/lib file are represented by their
            // directive; they are counted but not written.
            let skip = !card.include_chain.is_empty();
            let location = &card.source.location;
            match scope
                .entry(card.kind)
                .map_err(|error| refuse(error.to_string(), Some(location)))?
            {
                ScopeEntry::Device(item) => {
                    if !skip {
                        self.device(item, depth)?;
                    }
                }
                ScopeEntry::Model(item) => {
                    if !skip {
                        self.model(item, depth)?;
                    }
                }
                ScopeEntry::Subcircuit(sub) => {
                    check_closed(sub, &card.include_chain)?;
                    if !skip {
                        self.subcircuit(sub, depth)?;
                    }
                }
                ScopeEntry::Analysis(item) => {
                    if !skip {
                        self.analysis(item, depth)?;
                    }
                }
                ScopeEntry::Include(item) => {
                    if !skip {
                        self.include(item, depth)?;
                    }
                }
                ScopeEntry::Param(item) => {
                    if !skip {
                        self.param(item, depth)?;
                    }
                }
                ScopeEntry::Func(item) => {
                    if !skip {
                        self.func(item, depth)?;
                    }
                }
                ScopeEntry::Options(options) => {
                    if !skip {
                        if card
                            .source
                            .tokens
                            .first()
                            .is_some_and(|token| token.text.eq_ignore_ascii_case("set"))
                        {
                            self.settings(options, depth, "set")?;
                        } else {
                            self.options(options, depth)?;
                        }
                    }
                }
                ScopeEntry::Global(item) => {
                    if !skip {
                        self.global(item, depth)?;
                    }
                }
                ScopeEntry::InitialCondition(card_) => {
                    if !skip {
                        self.hints(".ic", card_, depth)?;
                    }
                }
                ScopeEntry::Nodeset(card_) => {
                    if !skip {
                        self.hints(".nodeset", card_, depth)?;
                    }
                }
                // `.save`/`.print` requests (`ast::OutputCards`), `.measure`
                // requests (`ParsedDeck::measurements`) and `.four` requests
                // (`ParsedDeck::fourier`) are returned beside the netlist, so
                // their typed index lives outside the netlist and the card is
                // reproduced from its own spelling.
                ScopeEntry::Save(_)
                | ScopeEntry::Print(_)
                | ScopeEntry::Measure(_)
                | ScopeEntry::Fourier(_)
                | ScopeEntry::Control => {
                    if !skip {
                        self.line(depth, card.source.raw.trim(), location)?;
                    }
                }
                // The closing `.ends` is written by `subcircuit`; it was
                // verified there to be the final card.
                ScopeEntry::Ends => {
                    if !body {
                        return Err(refuse("`.ends` outside a subcircuit", Some(location)));
                    }
                }
                ScopeEntry::End => {
                    if body {
                        return Err(refuse("`.end` inside a subcircuit", Some(location)));
                    }
                    if !skip {
                        self.line(depth, ".end", location)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn subcircuit(&mut self, sub: &Subcircuit, depth: usize) -> SpiceResult<()> {
        let location = &sub.location;
        let mut parts = vec![".subckt".to_owned(), node(&sub.name, location)?];
        for terminal in &sub.terminals {
            parts.push(node(terminal, location)?);
        }
        if !sub.parameters.is_empty() {
            parts.push("params:".to_owned());
            for parameter in &sub.parameters {
                parts.push(named_value(parameter, true)?);
            }
        }
        self.line(depth, &parts.join(" "), location)?;
        self.scope(ScopeView::from(sub), depth + 1, true)?;
        self.line(depth, &format!(".ends {}", sub.name), &sub.end_location)
    }

    fn analysis(&mut self, card: &AnalysisCard, depth: usize) -> SpiceResult<()> {
        let mut text = format!(".{}", card.kind.as_str());
        let mut arguments: Vec<&str> = card.arguments.iter().map(String::as_str).collect();
        if card.uic {
            if card.kind != AnalysisKind::Transient {
                return Err(refuse("uic is only valid on .tran", Some(&card.location)));
            }
            // The flag follows the positional arguments and precedes any
            // `name=value` driver options, where the parser accepts it.
            let mut at = 0;
            while at < arguments.len() && arguments.get(at + 1) != Some(&"=") {
                at += 1;
            }
            arguments.insert(at, "uic");
        }
        let mut previous: Option<&str> = None;
        for argument in &arguments {
            let glued =
                matches!(*argument, "(" | ")" | "," | "=") || matches!(previous, Some("(" | "="));
            if !glued {
                text.push(' ');
            }
            text.push_str(argument);
            previous = Some(argument);
        }
        // The arguments must re-tokenize to exactly the stored tokens.
        let tokens = lex(&text, &card.location)?;
        if tokens
            .iter()
            .skip(1)
            .map(|t| t.text.as_str())
            .ne(arguments.iter().copied())
        {
            return Err(refuse(
                "analysis arguments do not re-tokenize to the stored tokens",
                Some(&card.location),
            ));
        }
        self.line(depth, &text, &card.location)
    }

    fn include(&mut self, include: &IncludeDirective, depth: usize) -> SpiceResult<()> {
        let location = &include.location;
        let path = match single_token(&include.path_spelling, location) {
            Some(token) if path_token(&token) == Some(include.path.as_str()) => {
                include.path_spelling.clone()
            }
            _ => quote_path(&include.path, location)?,
        };
        let text = match &include.section {
            Some(section) => format!(".lib {path} {}", quote_path(section, location)?),
            None => format!(".include {path}"),
        };
        if include.path.is_empty() {
            return Err(refuse("empty include path", Some(location)));
        }
        self.line(depth, &text, location)
    }

    fn param(&mut self, card: &ParamCard, depth: usize) -> SpiceResult<()> {
        let location = &card.location;
        let mut text = ".param".to_owned();
        for assignment in &card.assignments {
            check_expression(&assignment.expression, location)?;
            let value = assignment.expression.spelling();
            let _ = write!(text, " {}={value}", assignment.name);
        }
        if card.assignments.is_empty() {
            return Err(refuse("empty .param card", Some(location)));
        }
        // The whole card, including the extent of each unbraced value, must
        // re-parse to the same assignments.
        let deck = parse_deck_text(location.path(), &format!("t\n{text}\n"));
        let reparsed = Parser::new().parse_deck(&deck).map_err(|error| {
            refuse(
                format!("`.param` card does not re-parse: {error}"),
                Some(location),
            )
        })?;
        let same = reparsed.params.len() == 1
            && reparsed.params[0].assignments.len() == card.assignments.len()
            && reparsed.params[0]
                .assignments
                .iter()
                .zip(&card.assignments)
                .all(|(a, b)| {
                    a.name == b.name
                        && a.expression.braced == b.expression.braced
                        && a.expression.quoted == b.expression.quoted
                        && a.expression.text == b.expression.text
                        && expr_form(&a.expression.root) == expr_form(&b.expression.root)
                });
        if !same {
            return Err(refuse(
                "`.param` card does not re-parse to the same assignments",
                Some(location),
            ));
        }
        self.line(depth, &text, location)
    }

    /// `.func name(p1,p2) body`, the body in its original delimiters. Like
    /// `.param`, the card is verified to re-parse to the same definition.
    fn func(&mut self, card: &FuncCard, depth: usize) -> SpiceResult<()> {
        let location = &card.location;
        // A formal named like a built-in only parses inside its own card;
        // the whole-card re-parse below still compares the tree.
        let shadows = card.parameters.iter().any(|p| {
            crate::netlist::expr::Function::from_name(&p.name).is_some()
                || crate::netlist::expr::EXCLUDED_FUNCTIONS.contains(&p.name.as_str())
        });
        if !shadows {
            check_expression(&card.body, location)?;
        }
        let formals: Vec<&str> = card.parameters.iter().map(|p| p.name.as_str()).collect();
        let text = match card.spelling {
            FuncSpelling::Func => format!(
                ".func {}({}) {}",
                card.name,
                formals.join(","),
                card.body.spelling()
            ),
            FuncSpelling::Param => format!(
                ".param {}({})={}",
                card.name,
                formals.join(","),
                card.body.spelling()
            ),
        };
        let deck = parse_deck_text(location.path(), &format!("t\n{text}\n"));
        let reparsed = Parser::new().parse_deck(&deck).map_err(|error| {
            refuse(
                format!("`.func` card does not re-parse: {error}"),
                Some(location),
            )
        })?;
        let same = reparsed.functions.len() == 1 && {
            let other = &reparsed.functions[0];
            other.name == card.name
                && other.spelling == card.spelling
                && other
                    .parameters
                    .iter()
                    .map(|p| p.name.as_str())
                    .eq(formals.iter().copied())
                && other.body.braced == card.body.braced
                && other.body.quoted == card.body.quoted
                && other.body.text == card.body.text
                && expr_form(&other.body.root) == expr_form(&card.body.root)
        };
        if !same {
            return Err(refuse(
                "`.func` card does not re-parse to the same definition",
                Some(location),
            ));
        }
        self.line(depth, &text, location)
    }

    fn options(&mut self, card: &OptionCard, depth: usize) -> SpiceResult<()> {
        self.settings(card, depth, ".options")
    }

    fn settings(&mut self, card: &OptionCard, depth: usize, prefix: &str) -> SpiceResult<()> {
        let location = &card.location;
        if card.settings.is_empty() {
            return Err(refuse("empty .options card", Some(location)));
        }
        let mut text = prefix.to_owned();
        for setting in &card.settings {
            if prefix == "set"
                && !matches!(
                    setting.name.as_str(),
                    "sqrnoise"
                        | "ngbehavior"
                        | "filetype"
                        | "nfreqs"
                        | "nperiods"
                        | "polydegree"
                        | "fourgridsize"
                )
            {
                return Err(refuse("unsupported front-end setting", Some(location)));
            }
            if !matches!(
                single_token(&setting.name, location),
                Some(Token {
                    kind: TokenKind::Word,
                    ..
                })
            ) {
                return Err(refuse(
                    format!("option name {:?} is not a plain word", setting.name),
                    Some(location),
                ));
            }
            text.push(' ');
            text.push_str(&setting.name);
            if setting.expression.is_some() && setting.value.is_none() {
                return Err(refuse(
                    format!(
                        "option {:?} has an expression but no value text",
                        setting.name
                    ),
                    Some(location),
                ));
            }
            if let Some(value) = &setting.value {
                let expression = setting.expression.is_some();
                let ok = match single_token(&value.text, location) {
                    Some(Token {
                        kind: TokenKind::Word,
                        ..
                    }) => !expression,
                    Some(Token {
                        kind: TokenKind::Number(v),
                        ..
                    }) => v.is_finite() && !expression,
                    Some(Token {
                        kind: TokenKind::Expression(_) | TokenKind::Quoted(_),
                        ..
                    }) => expression,
                    _ => false,
                };
                if !ok {
                    return Err(refuse(
                        format!(
                            "option value {:?} is not a word, finite number or parsed \
                             expression",
                            value.text
                        ),
                        Some(location),
                    ));
                }
                text.push('=');
                text.push_str(&value.text);
            }
        }
        self.line(depth, &text, location)
    }

    fn hints(&mut self, name: &str, card: &NodeHintCard, depth: usize) -> SpiceResult<()> {
        if card.entries.is_empty() {
            return Err(refuse(format!("empty {name} card"), Some(&card.location)));
        }
        let mut text = name.to_owned();
        for hint in &card.entries {
            let (kind, value) = match &hint.value {
                NodeHintValue::Literal { text, .. } => (ParameterKind::Scalar, text.clone()),
                NodeHintValue::Expression(expression) => (
                    ParameterKind::Expression(expression.clone()),
                    expression.spelling(),
                ),
            };
            let value = value_text(
                &ParameterAssignment {
                    name: format!("v({})", hint.node),
                    value,
                    kind,
                    location: hint.value_location.clone(),
                },
                false,
            )?;
            let _ = write!(text, " v({})={value}", node(&hint.node, &hint.location)?);
        }
        self.line(depth, &text, &card.location)
    }

    fn global(&mut self, card: &GlobalCard, depth: usize) -> SpiceResult<()> {
        if card.nodes.is_empty() {
            return Err(refuse("empty .global card", Some(&card.location)));
        }
        let mut text = ".global".to_owned();
        for global in &card.nodes {
            text.push(' ');
            text.push_str(&node(&global.name, &card.location)?);
        }
        self.line(depth, &text, &card.location)
    }

    fn model(&mut self, model: &ModelCard, depth: usize) -> SpiceResult<()> {
        let location = &model.location;
        let base = model.base.as_str();
        if !matches!(
            base,
            "d" | "npn"
                | "pnp"
                | "nmos"
                | "pmos"
                | "r"
                | "res"
                | "c"
                | "l"
                | "sw"
                | "csw"
                | "njf"
                | "pjf"
                | "urc"
                | "ltra"
                | "txl"
                | "nmf"
                | "pmf"
                // `inpdomod.c`'s HFET spellings of the same families.
                | "nhfet"
                | "phfet"
        ) {
            return Err(refuse(
                format!("model type {base:?} is outside the supported syntax"),
                Some(location),
            ));
        }
        let mut items = Vec::new();
        for parameter in &model.parameters {
            match &parameter.kind {
                ParameterKind::Flag => {
                    let allowed: &[&str] = match base {
                        "d" => &["d"],
                        "npn" | "pnp" => &["npn", "pnp"],
                        "nmos" | "pmos" => &["nmos", "pmos"],
                        "sw" => &["sw"],
                        "csw" => &["csw"],
                        "njf" | "pjf" => &["njf", "pjf"],
                        "urc" => &["urc"],
                        "txl" => &["txl"],
                        "ltra" => &[
                            "ltra",
                            "nocontrol",
                            "steplimit",
                            "nosteplimit",
                            "lininterp",
                            "quadinterp",
                            "mixedinterp",
                            "truncnr",
                            "truncdontcut",
                        ],
                        "nmf" | "pmf" | "nhfet" | "phfet" => &["nmf", "pmf", "nhfet", "phfet"],
                        _ => &[],
                    };
                    if !allowed.contains(&parameter.name.as_str()) || !parameter.value.is_empty() {
                        return Err(refuse(
                            format!("flag {:?} is not valid for a {base} model", parameter.name),
                            Some(location),
                        ));
                    }
                    items.push(parameter.name.clone());
                }
                ParameterKind::Textual
                    if matches!(base, "nmos" | "pmos") && parameter.name == "version" =>
                {
                    if !matches!(
                        single_token(&parameter.value, &parameter.location),
                        Some(Token {
                            kind: TokenKind::Word,
                            ..
                        })
                    ) {
                        return Err(refuse(
                            "MOS version must be one bare literal token",
                            Some(location),
                        ));
                    }
                    items.push(named_value(parameter, true)?);
                }
                ParameterKind::Expression(_) if parameter.name == "level" => {
                    return Err(refuse("expression-valued model level", Some(location)));
                }
                _ => items.push(named_value(parameter, false)?),
            }
        }
        let mut text = format!(".model {} {base}", node(&model.name, location)?);
        if !items.is_empty() {
            let _ = write!(text, "({})", items.join(" "));
        }
        self.line(depth, &text, location)
    }

    fn device(&mut self, device: &DeviceInstance, depth: usize) -> SpiceResult<()> {
        let location = &device.location;
        let designator = device.designator;
        if !device.name.to_ascii_lowercase().starts_with(designator) {
            return Err(refuse(
                format!(
                    "instance {:?} does not start with '{designator}'",
                    device.name
                ),
                Some(location),
            ));
        }
        let nonlinear = NonlinearForm::validate(device)?;
        if designator == 'b' || nonlinear.is_some() {
            let line = behavioural_card(device, nonlinear)?;
            return self.line(depth, &line, location);
        }
        let mut parts = vec![node(&device.name, location)?];
        let setters = device_prefix(device, &mut parts)?;
        match designator {
            'r' | 'c' | 'l' => passive_parameters(device, &mut parts)?,
            'v' | 'i' => source_parameters(device, &mut parts)?,
            'e'..='h' => controlled_parameters(device, &mut parts)?,
            'k' => mutual_parameters(device, &mut parts)?,
            's' | 'w' => switch_parameters(setters, designator, &mut parts)?,
            't' => tline_parameters(device, &mut parts)?,
            'o' => ltra_parameters(device, &mut parts)?,
            'u' => urc_parameters(device, &mut parts)?,
            'y' => {
                for p in &device.parameters {
                    if p.name != "len"
                        || !matches!(p.kind, ParameterKind::Scalar | ParameterKind::Expression(_))
                    {
                        return Err(refuse("TXL instance setter (only len=)", Some(&p.location)));
                    }
                    parts.push(named_value(p, false)?);
                }
            }
            'x' => {
                for parameter in &device.parameters {
                    parts.push(named_value(parameter, true)?);
                }
            }
            _ => transistor_parameters(device, &mut parts)?,
        }
        self.line(depth, &parts.join(" "), location)
    }
}

// Write and validate the common positional device fields before its setters.
fn device_prefix<'a>(
    device: &'a DeviceInstance,
    parts: &mut Vec<String>,
) -> SpiceResult<&'a [ParameterAssignment]> {
    let location = &device.location;
    let designator = device.designator;
    let count = device.nodes.len();
    let count_ok = match designator {
        'r' | 'c' | 'l' | 'v' | 'i' | 'd' | 'f' | 'h' | 'w' => count == 2,
        'q' => count == 3 || count == 4,
        'm' | 'e' | 'g' | 's' | 't' | 'o' | 'y' => count == 4,
        'j' | 'u' | 'z' => count == 3,
        'x' => true,
        'k' => count == 0,
        _ => {
            return Err(refuse(
                format!("device designator '{designator}' has no writer"),
                Some(location),
            ));
        }
    };
    if !count_ok {
        return Err(refuse(
            format!("'{designator}' instance with {count} nodes"),
            Some(location),
        ));
    }
    for n in &device.nodes {
        parts.push(node(n, location)?);
    }
    let mut setters = device.parameters.as_slice();
    if designator == 'w' {
        // INP2W reads the controlling source before the model name.
        match setters.split_first() {
            Some((control, rest))
                if control.name == "control" && control.kind == ParameterKind::Instance =>
            {
                parts.push(node(&control.value, &control.location)?);
                setters = rest;
            }
            _ => {
                return Err(refuse(
                    "'w' instance without a leading controlling source",
                    Some(location),
                ));
            }
        }
    }
    match (designator, &device.model) {
        ('v' | 'i' | 'e'..='h' | 'k' | 't', Some(_)) => {
            return Err(refuse("source with a model", Some(location)));
        }
        ('d' | 'q' | 'm' | 'x' | 's' | 'w' | 'j' | 'u' | 'o' | 'z' | 'y', None) => {
            return Err(refuse("device without a model/target", Some(location)));
        }
        (_, Some(model)) => {
            if designator == 'q'
                && (!model.bytes().any(|b| b.is_ascii_alphabetic())
                    || parse_spice_number(model).is_some())
            {
                return Err(refuse(
                    "BJT model name must be a non-numeric word with a letter",
                    Some(location),
                ));
            }
            if designator == 'm' && parse_spice_number(model).is_some() {
                return Err(refuse("numeric-looking MOS model name", Some(location)));
            }
            parts.push(node(model, location)?);
        }
        (_, None) => {}
    }
    Ok(setters)
}

/// An opening card in `chain` must be closed by a final `.ends` in the same
/// file (same include chain length), else the scope crosses an include boundary.
fn check_closed(sub: &Subcircuit, chain: &[SourceLoc]) -> SpiceResult<()> {
    match sub.cards.last() {
        Some(last)
            if last.kind == ScopedCardKind::Ends && last.include_chain.len() == chain.len() =>
        {
            Ok(())
        }
        _ => Err(refuse(
            format!(
                "subcircuit '{}' is not closed by `.ends` in the file that opened it",
                sub.name
            ),
            Some(&sub.location),
        )),
    }
}

fn lex(text: &str, location: &SourceLoc) -> SpiceResult<Vec<Token>> {
    tokenize(&LogicalLine::single(location.clone(), text.to_owned())).map_err(|error| {
        refuse(
            format!("text {text:?} does not tokenize: {error}"),
            Some(location),
        )
    })
}

/// `text` as one token whose spelling is exactly `text`, safe from comment
/// stripping.
fn single_token(text: &str, location: &SourceLoc) -> Option<Token> {
    if text.is_empty() || strip_comment(text, false) != text {
        return None;
    }
    let mut tokens = lex(text, location).ok()?;
    (tokens.len() == 1 && tokens[0].text == text).then(|| tokens.remove(0))
}

fn path_token(token: &Token) -> Option<&str> {
    match &token.kind {
        TokenKind::Quoted(decoded) => Some(decoded),
        TokenKind::Word | TokenKind::Number(_) => Some(&token.text),
        _ => None,
    }
}

fn quote_path(path: &str, location: &SourceLoc) -> SpiceResult<String> {
    if path.is_empty() || path.contains(['\n', '\r']) {
        return Err(refuse(
            "path or section is empty or contains a line break",
            Some(location),
        ));
    }
    if let Some(token) = single_token(path, location)
        && path_token(&token) == Some(path)
        && !matches!(token.kind, TokenKind::Quoted(_))
    {
        return Ok(path.to_owned());
    }
    let mut quoted = String::from("\"");
    for character in path.chars() {
        if matches!(character, '"' | '\\') {
            quoted.push('\\');
        }
        quoted.push(character);
    }
    quoted.push('"');
    Ok(quoted)
}

/// A node/instance/model name: one word or number token.
fn node(name: &str, location: &SourceLoc) -> SpiceResult<String> {
    match single_token(name, location) {
        Some(Token {
            kind: TokenKind::Word | TokenKind::Number(_),
            ..
        }) => Ok(name.to_owned()),
        _ => Err(refuse(
            format!("{name:?} is not a single name token"),
            Some(location),
        )),
    }
}

/// The expression text must re-parse to the stored tree, so writing the text
/// preserves the grouping (parentheses) of the source.
fn check_expression(expression: &ParameterExpression, location: &SourceLoc) -> SpiceResult<()> {
    let reparsed = parse_parameter_expression(&expression.text, location).map_err(|error| {
        refuse(
            format!(
                "expression {:?} does not re-parse: {error}",
                expression.text
            ),
            Some(location),
        )
    })?;
    if expr_form(&reparsed.root) != expr_form(&expression.root) {
        return Err(refuse(
            format!(
                "expression text {:?} does not match its stored syntax tree",
                expression.text
            ),
            Some(location),
        ));
    }
    Ok(())
}

/// The spelling of a scalar-or-expression value (`text_ok` also allows a
/// single `Textual` token, used at `.subckt`/X parameter sites).
fn value_text(parameter: &ParameterAssignment, text_ok: bool) -> SpiceResult<String> {
    let location = &parameter.location;
    let bad = |what: &str| {
        refuse(
            format!("parameter {:?}: {what}", parameter.name),
            Some(location),
        )
    };
    let token = single_token(&parameter.value, location);
    match &parameter.kind {
        ParameterKind::Scalar => match token {
            Some(Token {
                kind: TokenKind::Number(v),
                ..
            }) if v.is_finite() => {}
            _ => return Err(bad("scalar value is not one finite numeric token")),
        },
        ParameterKind::Textual if text_ok => {
            if !matches!(
                token,
                Some(Token {
                    kind: TokenKind::Word | TokenKind::Quoted(_),
                    ..
                })
            ) {
                return Err(bad("textual value is not one word or quoted token"));
            }
        }
        ParameterKind::Expression(expression) => {
            check_expression(expression, location)?;
            let consistent = parameter.value == expression.spelling();
            let kind_ok = matches!(
                (&token, expression.braced, expression.quoted),
                (
                    Some(Token {
                        kind: TokenKind::Expression(_),
                        ..
                    }),
                    true,
                    false
                ) | (
                    Some(Token {
                        kind: TokenKind::Quoted(_),
                        ..
                    }),
                    true,
                    true
                ) | (
                    Some(Token {
                        kind: TokenKind::Word,
                        ..
                    }),
                    false,
                    false
                )
            );
            if !consistent || !kind_ok {
                return Err(bad("value text does not match its expression"));
            }
        }
        _ => return Err(bad("value kind is not allowed here")),
    }
    Ok(parameter.value.clone())
}

fn named_value(parameter: &ParameterAssignment, text_ok: bool) -> SpiceResult<String> {
    let location = &parameter.location;
    if !matches!(
        single_token(&parameter.name, location),
        Some(Token {
            kind: TokenKind::Word,
            ..
        })
    ) {
        return Err(refuse(
            format!("parameter name {:?} is not a plain word", parameter.name),
            Some(location),
        ));
    }
    Ok(format!(
        "{}={}",
        parameter.name,
        value_text(parameter, text_ok)?
    ))
}

fn positioned_number<'a>(value: &'a PositionedValue, location: &SourceLoc) -> SpiceResult<&'a str> {
    match single_token(&value.text, location) {
        Some(Token {
            kind: TokenKind::Number(v),
            ..
        }) if v.is_finite() => Ok(&value.text),
        _ => Err(refuse(
            format!(
                "vector component {:?} is not a finite numeric literal",
                value.text
            ),
            Some(location),
        )),
    }
}

/// Write validated positioned numeric literals without intermediate owned
/// strings, shared by waveform and initial-condition vectors.
fn numeric_vector_text<'a>(
    prefix: &str,
    fields: impl IntoIterator<Item = &'a PositionedValue>,
    separator: char,
    location: &SourceLoc,
) -> SpiceResult<String> {
    let mut text = String::from(prefix);
    text.push('(');
    for (index, field) in fields.into_iter().enumerate() {
        if index != 0 {
            text.push(separator);
        }
        text.push_str(positioned_number(field, location)?);
    }
    text.push(')');
    Ok(text)
}

fn passive_parameters(device: &DeviceInstance, parts: &mut Vec<String>) -> SpiceResult<()> {
    let designator = device.designator;
    let primary = match designator {
        'r' => "resistance",
        'c' => "capacitance",
        _ => "inductance",
    };
    let allowed = |name: &str| match (designator, name) {
        (_, "temp" | "dtemp" | "m" | "tc1" | "tc2" | "scale")
        | ('r' | 'c', "w" | "l" | "bv_max")
        | ('r', "ac" | "tc" | "tce" | "noisy")
        | ('c' | 'l', "ic")
        | ('l', "nt") => true,
        (_, name) => name == primary,
    };
    for parameter in &device.parameters {
        if !allowed(&parameter.name) || matches!(parameter.kind, ParameterKind::Textual) {
            return Err(refuse(
                format!("parameter {:?} on '{designator}' instance", parameter.name),
                Some(&parameter.location),
            ));
        }
    }
    if device.model.is_none() && !device.parameters.iter().any(|p| p.name == primary) {
        return Err(refuse(
            format!("model-less '{designator}' instance without {primary}"),
            Some(&device.location),
        ));
    }
    let positional = device.model.is_none()
        && device
            .parameters
            .first()
            .is_some_and(|parameter| parameter.name == primary);
    for (index, parameter) in device.parameters.iter().enumerate() {
        if positional && index == 0 {
            parts.push(value_text(parameter, false)?);
        } else {
            parts.push(named_value(parameter, false)?);
        }
    }
    Ok(())
}

/// S/W (`parser/switch.rs`): only bare `on`/`off` flags follow the model, in
/// their written order; W's `control` was already written before the model.
/// U (`parser/urc.rs`): the ordered `l=`/`n=` setters, nothing else.
fn urc_parameters(device: &DeviceInstance, parts: &mut Vec<String>) -> SpiceResult<()> {
    for parameter in &device.parameters {
        if !matches!(
            parameter.kind,
            ParameterKind::Scalar | ParameterKind::Expression(_)
        ) || !matches!(parameter.name.as_str(), "l" | "n")
        {
            return Err(refuse(
                format!("parameter {:?} on 'u' instance", parameter.name),
                Some(&parameter.location),
            ));
        }
        parts.push(named_value(parameter, false)?);
    }
    Ok(())
}

fn switch_parameters(
    setters: &[ParameterAssignment],
    designator: char,
    parts: &mut Vec<String>,
) -> SpiceResult<()> {
    for parameter in setters {
        if parameter.kind != ParameterKind::Flag
            || !matches!(parameter.name.as_str(), "on" | "off")
            || !parameter.value.is_empty()
        {
            return Err(refuse(
                format!("parameter {:?} on '{designator}' instance", parameter.name),
                Some(&parameter.location),
            ));
        }
        parts.push(parameter.name.clone());
    }
    Ok(())
}

/// T (`parser/tline.rs`): `TRApTable` scalars and the `ic` vector, in order.
fn tline_parameters(device: &DeviceInstance, parts: &mut Vec<String>) -> SpiceResult<()> {
    const SCALARS: [&str; 11] = [
        "z0", "zo", "td", "f", "nl", "v1", "v2", "i1", "i2", "rel", "abs",
    ];
    line_parameters(device, &SCALARS, parts)
}

/// O (`parser/ltra.rs`): `LTRApTable` scalars and the `ic` vector, in order.
fn ltra_parameters(device: &DeviceInstance, parts: &mut Vec<String>) -> SpiceResult<()> {
    line_parameters(device, &["v1", "v2", "i1", "i2"], parts)
}

/// The ordered scalar setters and `ic` vector of a T or O line.
fn line_parameters(
    device: &DeviceInstance,
    scalars: &[&str],
    parts: &mut Vec<String>,
) -> SpiceResult<()> {
    const IC: [&str; 4] = ["v1", "i1", "v2", "i2"];
    for parameter in &device.parameters {
        let location = &parameter.location;
        match &parameter.kind {
            ParameterKind::InitialConditions(components) if parameter.name == "ic" => {
                if components.is_empty()
                    || components.len() > IC.len()
                    || components.iter().zip(IC).any(|(c, n)| c.name != n)
                {
                    return Err(refuse("malformed ic vector", Some(location)));
                }
                parts.push(numeric_vector_text(
                    "ic=",
                    components.iter().map(|component| &component.value),
                    ',',
                    location,
                )?);
            }
            ParameterKind::Scalar | ParameterKind::Expression(_)
                if scalars.contains(&parameter.name.as_str()) =>
            {
                parts.push(named_value(parameter, false)?);
            }
            _ => {
                return Err(refuse(
                    format!(
                        "parameter {:?} on '{}' instance",
                        parameter.name, device.designator
                    ),
                    Some(location),
                ));
            }
        }
    }
    Ok(())
}

/// E/F/G/H (`parser/controlled.rs`): the F/H controlling source first, then
/// the gain slot, then any `m=`/`gain=` tail. A gain stored last after other
/// setters was C's leading value (applied after the named setters), so it is
/// written positionally; a gain stored first is written as `gain=`. Any other
/// order cannot be re-parsed to the same setter sequence and is refused.
fn controlled_parameters(device: &DeviceInstance, parts: &mut Vec<String>) -> SpiceResult<()> {
    let designator = device.designator;
    let mut setters = device.parameters.as_slice();
    if matches!(designator, 'f' | 'h') {
        match setters.split_first() {
            Some((control, rest))
                if control.name == "control" && control.kind == ParameterKind::Instance =>
            {
                parts.push(node(&control.value, &control.location)?);
                setters = rest;
            }
            _ => {
                return Err(refuse(
                    format!("'{designator}' instance without a leading controlling source"),
                    Some(&device.location),
                ));
            }
        }
    }
    let multiplier = matches!(designator, 'f' | 'g');
    for parameter in setters {
        let allowed = parameter.name == "gain" || (multiplier && parameter.name == "m");
        if !allowed
            || !matches!(
                parameter.kind,
                ParameterKind::Scalar | ParameterKind::Expression(_)
            )
        {
            return Err(refuse(
                format!("parameter {:?} on '{designator}' instance", parameter.name),
                Some(&parameter.location),
            ));
        }
    }
    let unrepresentable = || {
        refuse(
            format!("'{designator}' setter order cannot be written as a linear gain card"),
            Some(&device.location),
        )
    };
    match setters {
        [gain] if gain.name == "gain" => parts.push(value_text(gain, false)?),
        [first, rest @ ..] if first.name == "gain" => {
            if rest.first().is_some_and(|next| next.name != "m") {
                return Err(unrepresentable());
            }
            for parameter in setters {
                parts.push(named_value(parameter, false)?);
            }
        }
        [tail @ .., gain]
            if gain.name == "gain" && tail.first().is_some_and(|first| first.name == "m") =>
        {
            parts.push(value_text(gain, false)?);
            for parameter in tail {
                parts.push(named_value(parameter, false)?);
            }
        }
        _ if !setters.iter().any(|parameter| parameter.name == "gain") => {
            return Err(refuse(
                format!("'{designator}' instance without a gain"),
                Some(&device.location),
            ));
        }
        _ => return Err(unrepresentable()),
    }
    Ok(())
}

/// The behavioural expression text must re-parse to the stored tree.
fn behavioural_text(parameter: &ParameterAssignment) -> SpiceResult<String> {
    let location = &parameter.location;
    let ParameterKind::Behavioural(expression) = &parameter.kind else {
        return Err(refuse(
            format!(
                "parameter {:?} is not a behavioural expression",
                parameter.name
            ),
            Some(location),
        ));
    };
    let reparsed = Parser::new()
        .parse_behavioural_expression(&expression.text, location, expression.verbatim)
        .map_err(|error| {
            refuse(
                format!(
                    "behavioural expression {:?} does not re-parse: {error}",
                    expression.text
                ),
                Some(location),
            )
        })?;
    if expression.text.contains(['\n', '\r'])
        || bexpr_form(&reparsed.root) != bexpr_form(&expression.root)
    {
        return Err(refuse(
            format!(
                "behavioural expression text {:?} does not match its stored syntax tree",
                expression.text
            ),
            Some(location),
        ));
    }
    Ok(expression.text.clone())
}

/// B sources and the nonlinear E/G/F/H forms (`parser/behavioural.rs`):
/// `b1 n+ n- v=<text> [setters]`, `e1 n+ n- value=<text> [setters]`,
/// `e1 n+ n- table {<text>} = (x, y) ... [m=..]`,
/// `e1 n+ n- nc+ nc- table=(x0, y0, ...)` and
/// `e1 n+ n- poly(n) <controls> <coefficients> [m=..]`.
fn behavioural_card(
    device: &DeviceInstance,
    nonlinear: Option<NonlinearForm<'_>>,
) -> SpiceResult<String> {
    let location = &device.location;
    let designator = device.designator;
    if device.model.is_some() {
        return Err(refuse("behavioural source with a model", Some(location)));
    }
    let mut parts = vec![node(&device.name, location)?];
    let [positive, negative, controls @ ..] = device.nodes.as_slice() else {
        return Err(refuse(
            format!("'{designator}' instance with fewer than two nodes"),
            Some(location),
        ));
    };
    parts.push(node(positive, location)?);
    parts.push(node(negative, location)?);
    match nonlinear {
        Some(NonlinearForm::Value(form)) => {
            parts.push(format!("value={}", behavioural_text(form.input)?));
            for parameter in form.setters {
                parts.push(named_value(parameter, false)?);
            }
        }
        Some(NonlinearForm::Table(form)) => behavioural_table(device, form, &mut parts)?,
        Some(NonlinearForm::Poly(form)) => behavioural_poly(device, form, &mut parts)?,
        None => {
            let Some((first, setters)) = device.parameters.split_first() else {
                return Err(refuse("B source without behavioural form", Some(location)));
            };
            if designator != 'b'
                || !matches!(first.name.as_str(), "v" | "i")
                || !controls.is_empty()
            {
                return Err(refuse("invalid B source form", Some(location)));
            }
            parts.push(format!("{}={}", first.name, behavioural_text(first)?));
            for parameter in setters {
                parts.push(behavioural_setter(parameter, designator, B_SETTERS)?);
            }
        }
    }
    Ok(parts.join(" "))
}

fn behavioural_setter(
    parameter: &ParameterAssignment,
    designator: char,
    allowed: &[&str],
) -> SpiceResult<String> {
    if allowed.contains(&parameter.name.as_str()) {
        named_value(parameter, false)
    } else {
        Err(refuse(
            format!("parameter {:?} on '{designator}' instance", parameter.name),
            Some(&parameter.location),
        ))
    }
}

fn behavioural_table(
    device: &DeviceInstance,
    form: TableForm<'_>,
    parts: &mut Vec<String>,
) -> SpiceResult<()> {
    match form.input {
        TableInput::Voltage([positive, negative]) => {
            parts.push(node(positive, &device.location)?);
            parts.push(node(negative, &device.location)?);
            let mut values = Vec::with_capacity(form.points.len() * 2);
            for (x, y) in form.points {
                values.push(value_text(x.parameter, false)?);
                values.push(value_text(y.parameter, false)?);
            }
            parts.push(format!("table=({})", values.join(", ")));
        }
        TableInput::Expression(parameter, _) => {
            parts.push(format!("table {{{}}} =", behavioural_text(parameter)?));
            for (x, y) in form.points {
                let x = value_text(x.parameter, false)?;
                let y = value_text(y.parameter, false)?;
                parts.push(format!("({x}, {y})"));
            }
        }
    }
    for parameter in form.setters {
        parts.push(named_value(parameter, false)?);
    }
    Ok(())
}

fn behavioural_poly(
    device: &DeviceInstance,
    form: PolyForm<'_>,
    parts: &mut Vec<String>,
) -> SpiceResult<()> {
    let spelling = value_text(form.spelling, false)?;
    parts.push(format!("poly({spelling})"));
    match form.controls {
        PolyControls::Voltage(pairs) => {
            for control in pairs.iter().flatten() {
                parts.push(node(control, &device.location)?);
            }
        }
        PolyControls::Current(sources) => {
            for control in sources {
                parts.push(node(&control.value, &control.location)?);
            }
        }
    }
    for coefficient in form.coefficients {
        parts.push(value_text(coefficient.parameter, false)?);
    }
    for parameter in form.setters {
        parts.push(named_value(parameter, false)?);
    }
    Ok(())
}

/// K (`parser/mutual.rs`): the inductor references `inductor1`, `inductor2`,
/// … in order, then exactly one `coefficient`, written positionally (a named
/// `k=`/`coefficient=` setter re-parses to the same setter).
fn mutual_parameters(device: &DeviceInstance, parts: &mut Vec<String>) -> SpiceResult<()> {
    let Some((coupling, inductors)) = device.parameters.split_last() else {
        return Err(refuse(
            "'k' instance without inductors and a coupling",
            Some(&device.location),
        ));
    };
    if inductors.len() < 2 {
        return Err(refuse(
            "'k' instance with fewer than two inductors",
            Some(&device.location),
        ));
    }
    for (index, inductor) in inductors.iter().enumerate() {
        if inductor.kind != ParameterKind::Instance
            || inductor.name != format!("inductor{}", index + 1)
        {
            return Err(refuse(
                format!("parameter {:?} on 'k' instance", inductor.name),
                Some(&inductor.location),
            ));
        }
        if parse_spice_number(&inductor.value).is_some() {
            return Err(refuse(
                "numeric-looking inductor name on a 'k' instance",
                Some(&inductor.location),
            ));
        }
        parts.push(node(&inductor.value, &inductor.location)?);
    }
    if coupling.name != "coefficient"
        || !matches!(
            coupling.kind,
            ParameterKind::Scalar | ParameterKind::Expression(_)
        )
    {
        return Err(refuse(
            format!("parameter {:?} on 'k' instance", coupling.name),
            Some(&coupling.location),
        ));
    }
    parts.push(value_text(coupling, false)?);
    Ok(())
}

fn source_parameters(device: &DeviceInstance, parts: &mut Vec<String>) -> SpiceResult<()> {
    let mut parameters = device.parameters.iter().peekable();
    while let Some(parameter) = parameters.next() {
        let location = &parameter.location;
        match (&parameter.kind, parameter.name.as_str()) {
            (ParameterKind::Scalar | ParameterKind::Expression(_), "dc") => {
                parts.push("dc".to_owned());
                parts.push(value_text(parameter, false)?);
            }
            (ParameterKind::Scalar | ParameterKind::Expression(_), "acmag") => {
                let Some(phase) = parameters.next_if(|next| next.name == "acphase") else {
                    return Err(refuse("acmag without acphase", Some(location)));
                };
                parts.push("ac".to_owned());
                parts.push(value_text(parameter, false)?);
                parts.push(value_text(phase, false)?);
            }
            (
                ParameterKind::Scalar | ParameterKind::Expression(_),
                name @ ("distof1mag" | "distof2mag"),
            ) => {
                let phase_name = if name == "distof1mag" {
                    "distof1phase"
                } else {
                    "distof2phase"
                };
                let Some(phase) = parameters.next_if(|next| next.name == phase_name) else {
                    return Err(refuse(
                        format!("{name} without {phase_name}"),
                        Some(location),
                    ));
                };
                parts.push(name.trim_end_matches("mag").to_owned());
                parts.push(value_text(parameter, false)?);
                parts.push(value_text(phase, false)?);
            }
            (ParameterKind::Waveform(waveform), name) => {
                parts.push(waveform_text(waveform, name, location)?);
            }
            // PWL delay/repeat setters stay separate ordered scalars (vsrc.c).
            (ParameterKind::Scalar | ParameterKind::Expression(_), name @ ("r" | "td")) => {
                parts.push(format!("{name}={}", value_text(parameter, false)?));
            }
            // RFSPICE port setters of a V source, ordered like the PWL ones.
            (
                ParameterKind::Scalar | ParameterKind::Expression(_),
                name @ ("portnum" | "z0" | "pwr" | "freq" | "phase"),
            ) if device.designator == 'v' => {
                parts.push(format!("{name}={}", value_text(parameter, false)?));
            }
            _ => {
                return Err(refuse(
                    format!("source parameter {:?}", parameter.name),
                    Some(location),
                ));
            }
        }
    }
    Ok(())
}

fn waveform_text(
    waveform: &SourceWaveform,
    name: &str,
    location: &SourceLoc,
) -> SpiceResult<String> {
    let kind = waveform.kind();
    let keyword = kind.keyword(name);
    let fields = match waveform {
        SourceWaveform::Pulse(pulse) => {
            let optional = [
                &pulse.delay,
                &pulse.rise,
                &pulse.fall,
                &pulse.width,
                &pulse.period,
                &pulse.count,
            ];
            let mut fields = vec![&pulse.initial, &pulse.pulsed];
            let mut omitted = false;
            for value in optional {
                match value {
                    Some(value) if !omitted => fields.push(value),
                    Some(_) => {
                        return Err(refuse(
                            "PULSE field present after an omitted field",
                            Some(location),
                        ));
                    }
                    None => omitted = true,
                }
            }
            fields
        }
        SourceWaveform::Pwl(points) => {
            if points.is_empty() || points.len() > kind.maximum_fields() / 2 {
                return Err(refuse("PWL needs 1 to 2048 points", Some(location)));
            }
            points.iter().flat_map(|p| [&p.time, &p.value]).collect()
        }
        SourceWaveform::Function(function) => {
            let maximum = kind.maximum_fields();
            if !(2..=maximum).contains(&function.values.len()) {
                return Err(refuse(
                    format!(
                        "{} needs 2 to {maximum} fields",
                        function.function.keyword()
                    ),
                    Some(location),
                ));
            }
            function.values.iter().collect()
        }
    };
    if name != keyword {
        return Err(refuse(
            format!("waveform setter named {name:?} holds a {keyword}"),
            Some(location),
        ));
    }
    numeric_vector_text(keyword, fields, ' ', location)
}

fn transistor_parameters(device: &DeviceInstance, parts: &mut Vec<String>) -> SpiceResult<()> {
    let designator = device.designator;
    let scalars: &[&str] = match designator {
        'd' => &[
            "pj", "area", "w", "l", "m", "ic", "temp", "dtemp", "lm", "lp", "wm", "wp",
        ],
        'q' => &[
            "area", "areab", "areac", "m", "icvbe", "icvce", "temp", "dtemp",
        ],
        'j' => &["area", "m", "ic-vds", "ic-vgs", "temp", "dtemp"],
        'z' => &[
            "area", "m", "l", "w", "td", "ts", "dtemp", "temp", "icvds", "icvgs",
        ],
        _ => &[
            "m", "l", "w", "ad", "as", "pd", "ps", "nrd", "nrs", "icvds", "icvgs", "icvbs", "temp",
            "dtemp", "nqsmod", "acnqsmod", "geo", "delvto", "mulu0",
            // The BSIM4 instance keywords the M grammar accepts (bsim4/b4.c).
            "delvt0", "geomod", "min", "nf", "ngcon", "rbdb", "rbsb", "rbpb", "rbps", "rbpd",
            "rbodymod", "rgatemod", "rgeomod", "sa", "sb", "sc", "sd", "sca", "scb", "scc",
            "trnqsmod", "wnflag", "xgw", "mult_fn", "mult_i", "mult_q",
        ],
    };
    for parameter in &device.parameters {
        let location = &parameter.location;
        match &parameter.kind {
            ParameterKind::Flag if parameter.name == "off" && parameter.value.is_empty() => {
                parts.push("off".to_owned());
            }
            ParameterKind::InitialConditions(components) if parameter.name == "ic" => {
                let names: &[&str] = match designator {
                    'q' => &["icvbe", "icvce"],
                    'm' => &["icvds", "icvgs", "icvbs"],
                    'j' => &["ic-vds", "ic-vgs"],
                    'z' => &["icvds", "icvgs"],
                    _ => &[],
                };
                if components.is_empty()
                    || components.len() > names.len()
                    || components.iter().zip(names).any(|(c, n)| c.name != *n)
                {
                    return Err(refuse("malformed ic vector", Some(location)));
                }
                parts.push(numeric_vector_text(
                    "ic=",
                    components.iter().map(|component| &component.value),
                    ',',
                    location,
                )?);
            }
            ParameterKind::Scalar | ParameterKind::Expression(_)
                if scalars.contains(&parameter.name.as_str()) =>
            {
                parts.push(named_value(parameter, false)?);
            }
            _ => {
                return Err(refuse(
                    format!("parameter {:?} on '{designator}' instance", parameter.name),
                    Some(location),
                ));
            }
        }
    }
    Ok(())
}
