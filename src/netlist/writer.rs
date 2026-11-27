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

use std::io::{self, Write};

use crate::primitives::{AnalysisKind, SourceLoc, SpiceError, SpiceResult, parse_spice_number};

use crate::netlist::ast::{
    AnalysisCard, CodePortKind, DeviceInstance, FuncCard, FuncSpelling, GlobalCard,
    IncludeDirective, ModelCard, Netlist, NodeHintCard, NodeHintValue, NumericVectorEntry,
    OptionCard, ParamCard, ParameterAssignment, ParameterKind, PositionedValue, ScopeEntry,
    ScopeView, ScopedCardKind, SourceWaveform, Subcircuit,
};
use crate::netlist::behavioural::forms::{
    B_SETTERS, NonlinearForm, PolyControls, PolyForm, TableForm, TableInput,
};
use crate::netlist::expr::ParameterExpression;
use crate::netlist::semantic::equal as semantic_equal;
use crate::netlist::source::{parse_deck_text, strip_comment};
use crate::netlist::token::{Lexeme, TokenKind, scan_next};
use crate::netlist::{Parser, parse_parameter_expression};

/// Resource limits for normalized emission.
///
/// Byte limits count emitted UTF-8
/// bytes, including indentation and newlines. The title consumes bytes but
/// is not a card. The card budget counts visited ordered card occurrences,
/// including skipped include content in a visited scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteLimits {
    /// Maximum total emitted bytes.
    pub bytes: usize,
    /// Maximum bytes in any emitted line (including the title).
    pub card_bytes: usize,
    /// Maximum visited logical cards.
    pub cards: usize,
    /// Maximum nested subcircuit depth; capped at the parser's limit of 64.
    pub depth: usize,
}

impl Default for WriteLimits {
    fn default() -> Self {
        Self {
            bytes: 64 * 1024 * 1024,
            card_bytes: 16 * 1024 * 1024,
            cards: 1_000_000,
            depth: 64,
        }
    }
}

/// Serializes a normalized deck to a string using the bounded emission kernel.
///
/// # Errors
/// Invalid or unrepresentable AST, exceeded resource limits or allocation failure.
pub fn write_netlist(netlist: &Netlist) -> SpiceResult<String> {
    let mut collector = Collector(Vec::new());
    write_netlist_to(netlist, &mut collector, WriteLimits::default())?;
    String::from_utf8(collector.0)
        .map_err(|error| refuse(error.to_string(), Some(&netlist.location)))
}

/// Emit normalized cards to a caller-owned sink without flushing it.
///
/// On error the sink may already contain a prefix of the deck (including a
/// partially written card on I/O failure); atomic publication belongs to the
/// caller. The same validation and order apply as in [`write_netlist`].
///
/// # Errors
/// Invalid/unrepresentable AST, exceeded resource limits or sink I/O failure.
pub fn write_netlist_to(
    netlist: &Netlist,
    sink: &mut dyn Write,
    limits: WriteLimits,
) -> SpiceResult<()> {
    if netlist.prepared_initialization.is_some() {
        return Err(refuse(
            "prepared initial conditions require their typed metadata, not native deck serialization",
            Some(&netlist.location),
        ));
    }
    if netlist.title.contains(['\n', '\r']) || netlist.title != netlist.title.trim_end() {
        return Err(refuse(
            "the title must be one line without trailing whitespace",
            Some(&netlist.location),
        ));
    }
    let mut writer = Writer {
        sink,
        buffer: String::new(),
        remaining: limits.bytes,
        cards: limits.cards,
        card_bytes: limits.card_bytes,
        depth: limits.depth.min(64),
    };
    writer.emit(0, &netlist.title, &netlist.location, true)?;
    writer.scope(ScopeView::from(netlist), 0, false)
}

struct Collector(Vec<u8>);

impl Write for Collector {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.try_reserve(bytes.len()).map_err(io::Error::other)?;
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn refuse(what: impl Into<String>, location: Option<&SourceLoc>) -> SpiceError {
    SpiceError::Unsupported {
        feature: format!("netlist writer: {}", what.into()),
        location: location.cloned(),
    }
}

struct Card<'a> {
    text: &'a mut String,
    limit: usize,
    location: &'a SourceLoc,
}

impl Card<'_> {
    fn check_length(&self, additional: usize) -> SpiceResult<()> {
        if additional > self.limit.saturating_sub(self.text.len()) {
            return Err(refuse(
                "card/total byte limit exceeded",
                Some(self.location),
            ));
        }
        Ok(())
    }

    fn push_str(&mut self, text: &str) -> SpiceResult<()> {
        self.check_length(text.len())?;
        let needed = self.text.len() + text.len();
        if needed > self.text.capacity() {
            // Amortized growth within the configured card budget.
            let capacity = needed
                .max(self.text.capacity().saturating_mul(2))
                .min(self.limit);
            self.text
                .try_reserve_exact(capacity - self.text.len())
                .map_err(|error| {
                    refuse(
                        format!("card buffer allocation: {error}"),
                        Some(self.location),
                    )
                })?;
        }
        self.text.push_str(text);
        Ok(())
    }

    fn push(&mut self, character: char) -> SpiceResult<()> {
        self.push_str(character.encode_utf8(&mut [0; 4]))
    }

    fn word(&mut self, word: &str) -> SpiceResult<()> {
        if !self.text.is_empty() {
            self.push(' ')?;
        }
        self.push_str(word)
    }

    fn expression(&mut self, expression: &ParameterExpression) -> SpiceResult<()> {
        match (expression.braced, expression.quoted) {
            (true, true) => {
                self.push('\'')?;
                self.push_str(&expression.text)?;
                self.push('\'')
            }
            (true, false) => {
                self.push('{')?;
                self.push_str(&expression.text)?;
                self.push('}')
            }
            (false, false) => self.push_str(&expression.text),
            (false, true) => Err(refuse(
                "quoted expression without delimiters",
                Some(self.location),
            )),
        }
    }
}

struct Writer<'a> {
    sink: &'a mut dyn Write,
    buffer: String,
    remaining: usize,
    cards: usize,
    card_bytes: usize,
    depth: usize,
}

fn emit_line(
    sink: &mut dyn Write,
    remaining: &mut usize,
    card_bytes: usize,
    depth: usize,
    text: &str,
    location: &SourceLoc,
    title: bool,
) -> SpiceResult<()> {
    let indent = 2 * depth; // depth is checked against the parser limit
    let bytes = text
        .len()
        .checked_add(indent + 1)
        .filter(|&size| size <= card_bytes && size <= *remaining)
        .ok_or_else(|| refuse("card/total byte limit exceeded", Some(location)))?;
    if !title && (text.contains(['\n', '\r']) || strip_comment(text, false) != text) {
        return Err(refuse(
            format!("card would not survive comment/line handling: {text:?}"),
            Some(location),
        ));
    }
    sink.write_all(&[b' '; 128][..indent])
        .and_then(|()| sink.write_all(text.as_bytes()))
        .and_then(|()| sink.write_all(b"\n"))
        .map_err(|error| SpiceError::io(location.path(), &error))?;
    *remaining -= bytes;
    Ok(())
}

impl Writer<'_> {
    fn emit(
        &mut self,
        depth: usize,
        text: &str,
        location: &SourceLoc,
        title: bool,
    ) -> SpiceResult<()> {
        emit_line(
            self.sink,
            &mut self.remaining,
            self.card_bytes,
            depth,
            text,
            location,
            title,
        )
    }

    fn line(&mut self, depth: usize, text: &str, location: &SourceLoc) -> SpiceResult<()> {
        self.emit(depth, text, location, false)
    }

    fn card<'a>(&'a mut self, depth: usize, location: &'a SourceLoc) -> SpiceResult<Card<'a>> {
        let limit = self
            .card_bytes
            .min(self.remaining)
            .checked_sub(2 * depth + 1)
            .ok_or_else(|| refuse("card/total byte limit exceeded", Some(location)))?;
        self.buffer.clear();
        Ok(Card {
            text: &mut self.buffer,
            limit,
            location,
        })
    }

    fn finish_card(&mut self, depth: usize, location: &SourceLoc) -> SpiceResult<()> {
        emit_line(
            self.sink,
            &mut self.remaining,
            self.card_bytes,
            depth,
            &self.buffer,
            location,
            false,
        )
    }

    fn scope(&mut self, scope: ScopeView<'_>, depth: usize, body: bool) -> SpiceResult<()> {
        if depth > self.depth {
            return Err(refuse("subcircuit depth limit exceeded", None));
        }
        self.cards = self
            .cards
            .checked_sub(scope.cards.len())
            .ok_or_else(|| refuse("card count limit exceeded", None))?;
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
                ScopeEntry::Subcircuit(sub) => {
                    check_closed(sub, &card.include_chain)?;
                    if !skip {
                        self.subcircuit(sub, depth)?;
                    }
                }
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
                _ if skip => {}
                ScopeEntry::Device(item) => self.device(item, depth)?,
                ScopeEntry::Model(item) => self.model(item, depth)?,
                ScopeEntry::Analysis(item) => self.analysis(item, depth)?,
                ScopeEntry::Include(item) => self.include(item, depth)?,
                ScopeEntry::Param(item) => self.param(item, depth)?,
                ScopeEntry::Func(item) => self.func(item, depth)?,
                ScopeEntry::Options(options) => {
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
                ScopeEntry::Global(item) => self.global(item, depth)?,
                ScopeEntry::InitialCondition(item) => self.hints(".ic", item, depth)?,
                ScopeEntry::Nodeset(item) => self.hints(".nodeset", item, depth)?,
                // The typed output indexes live outside the netlist.
                // Reproduce the original output or control card spelling.
                ScopeEntry::Save(_)
                | ScopeEntry::Print(_)
                | ScopeEntry::Measure(_)
                | ScopeEntry::Fourier(_)
                | ScopeEntry::Control => self.line(depth, card.source.raw.trim(), location)?,
            }
        }
        Ok(())
    }

    fn subcircuit(&mut self, sub: &Subcircuit, depth: usize) -> SpiceResult<()> {
        let location = &sub.location;
        let mut parts = self.card(depth, location)?;
        parts.word(".subckt")?;
        parts.word(node(&parts, &sub.name, location)?)?;
        for terminal in &sub.terminals {
            parts.word(node(&parts, terminal, location)?)?;
        }
        if !sub.parameters.is_empty() {
            parts.word("params:")?;
            for parameter in &sub.parameters {
                named_value(&mut parts, parameter, true)?;
            }
        }
        self.finish_card(depth, location)?;
        self.scope(ScopeView::from(sub), depth + 1, true)?;
        let mut parts = self.card(depth, &sub.end_location)?;
        parts.word(".ends")?;
        parts.word(&sub.name)?;
        self.finish_card(depth, &sub.end_location)
    }

    fn analysis(&mut self, card: &AnalysisCard, depth: usize) -> SpiceResult<()> {
        let location = &card.location;
        let at = if card.uic {
            if !matches!(
                card.kind,
                AnalysisKind::Transient | AnalysisKind::PeriodicSteadyState
            ) {
                return Err(refuse("uic is only valid on .tran or .pss", Some(location)));
            }
            card.arguments
                .windows(2)
                .position(|pair| pair[1] == "=")
                .unwrap_or(card.arguments.len())
        } else {
            card.arguments.len()
        };
        let arguments = || {
            card.arguments[..at]
                .iter()
                .map(String::as_str)
                .chain(card.uic.then_some("uic"))
                .chain(card.arguments[at..].iter().map(String::as_str))
        };
        let mut parts = self.card(depth, location)?;
        parts.push('.')?;
        parts.push_str(card.kind.as_str())?;
        let mut previous = None;
        for argument in arguments() {
            let glued =
                matches!(argument, "(" | ")" | "," | "=") || matches!(previous, Some("(" | "="));
            if !glued {
                parts.push(' ')?;
            }
            parts.push_str(argument)?;
            previous = Some(argument);
        }
        let mut cursor = 0;
        let text = parts.text.as_str();
        let mut next = || scan_next(text, &mut cursor, |_| location.clone());
        let _ = next()?;
        for expected in arguments() {
            if next()?.is_none_or(|token| token.text != expected) {
                return Err(refuse(
                    "analysis arguments do not re-tokenize to the stored tokens",
                    Some(location),
                ));
            }
        }
        if next()?.is_some() {
            return Err(refuse(
                "analysis arguments do not re-tokenize to the stored tokens",
                Some(location),
            ));
        }
        self.finish_card(depth, location)
    }

    fn include(&mut self, include: &IncludeDirective, depth: usize) -> SpiceResult<()> {
        let location = &include.location;
        if include.path.is_empty() {
            return Err(refuse("empty include path", Some(location)));
        }
        let mut parts = self.card(depth, location)?;
        parts.word(if include.section.is_some() {
            ".lib"
        } else {
            ".include"
        })?;
        parts.check_length(include.path_spelling.len())?;
        if single_token(&include.path_spelling, location)
            .is_some_and(|token| path_token(&token) == Some(include.path.as_str()))
        {
            parts.word(&include.path_spelling)?;
        } else {
            quote_path(&mut parts, &include.path, location)?;
        }
        if let Some(section) = &include.section {
            quote_path(&mut parts, section, location)?;
        }
        self.finish_card(depth, location)
    }

    fn param(&mut self, card: &ParamCard, depth: usize) -> SpiceResult<()> {
        let location = &card.location;
        let mut parts = self.card(depth, location)?;
        parts.word(".param")?;
        for assignment in &card.assignments {
            parts.word(&assignment.name)?;
            parts.push('=')?;
            parts.expression(&assignment.expression)?;
        }
        if card.assignments.is_empty() {
            return Err(refuse("empty .param card", Some(location)));
        }
        let deck = parse_deck_text(location.path(), &format!("t\n{}\n", parts.text));
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
                .all(|(a, b)| a.name == b.name && semantic_equal(&a.expression, &b.expression));
        if !same {
            return Err(refuse(
                "`.param` card does not match the stored assignments",
                Some(location),
            ));
        }
        self.finish_card(depth, location)
    }

    /// `.func name(p1,p2) body`, the body in its original delimiters. Like
    /// `.param`, the card is verified to re-parse to the same definition.
    fn func(&mut self, card: &FuncCard, depth: usize) -> SpiceResult<()> {
        let location = &card.location;
        let mut parts = self.card(depth, location)?;
        let (prefix, delimiter) = match card.spelling {
            FuncSpelling::Func => (".func", ' '),
            FuncSpelling::Param => (".param", '='),
        };
        parts.word(prefix)?;
        parts.word(&card.name)?;
        parts.push('(')?;
        for (i, parameter) in card.parameters.iter().enumerate() {
            if i != 0 {
                parts.push(',')?;
            }
            parts.push_str(&parameter.name)?;
        }
        parts.push(')')?;
        parts.push(delimiter)?;
        parts.expression(&card.body)?;
        let deck = parse_deck_text(location.path(), &format!("t\n{}\n", parts.text));
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
                    .eq(card.parameters.iter().map(|p| p.name.as_str()))
                && semantic_equal(&other.body, &card.body)
        };
        if !same {
            return Err(refuse(
                "`.func` card does not re-parse to the same definition",
                Some(location),
            ));
        }
        self.finish_card(depth, location)
    }

    fn options(&mut self, card: &OptionCard, depth: usize) -> SpiceResult<()> {
        self.settings(card, depth, ".options")
    }

    fn settings(&mut self, card: &OptionCard, depth: usize, prefix: &str) -> SpiceResult<()> {
        let location = &card.location;
        if card.settings.is_empty() {
            return Err(refuse("empty .options card", Some(location)));
        }
        let mut parts = self.card(depth, location)?;
        parts.word(prefix)?;
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
            parts.check_length(setting.name.len())?;
            if !matches!(
                single_token(&setting.name, location),
                Some(Lexeme {
                    kind: TokenKind::Word,
                    ..
                })
            ) {
                return Err(refuse(
                    format!("option name {:?} is not a plain word", setting.name),
                    Some(location),
                ));
            }
            parts.word(&setting.name)?;
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
                parts.check_length(value.text.len())?;
                let expression = setting.expression.is_some();
                let ok = match single_token(&value.text, location) {
                    Some(Lexeme {
                        kind: TokenKind::Word,
                        ..
                    }) => !expression,
                    Some(Lexeme {
                        kind: TokenKind::Number(v),
                        ..
                    }) => v.is_finite() && !expression,
                    Some(Lexeme {
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
                if let Some(expression) = &setting.expression {
                    check_value_expression(&parts, &value.text, expression, location)?;
                }
                parts.push('=')?;
                parts.push_str(&value.text)?;
            }
        }
        self.finish_card(depth, location)
    }

    fn hints(&mut self, name: &str, card: &NodeHintCard, depth: usize) -> SpiceResult<()> {
        if card.entries.is_empty() {
            return Err(refuse(format!("empty {name} card"), Some(&card.location)));
        }
        let mut parts = self.card(depth, &card.location)?;
        parts.word(name)?;
        for hint in &card.entries {
            parts.word("v(")?;
            parts.push_str(node(&parts, &hint.node, &hint.location)?)?;
            parts.push_str(")=")?;
            match &hint.value {
                NodeHintValue::Literal { text, value } => {
                    parts.check_length(text.len())?;
                    if !matches!(single_token(text, &hint.value_location), Some(Lexeme {kind: TokenKind::Number(v), ..}) if v.is_finite() && v.to_bits() == value.to_bits())
                    {
                        return Err(refuse(
                            "node hint literal does not match its finite numeric value",
                            Some(&hint.value_location),
                        ));
                    }
                    parts.push_str(text)?;
                }
                NodeHintValue::Expression(expression) => {
                    parts.check_length(expression.text.len())?;
                    check_expression(expression, &hint.value_location)?;
                    let start = parts.text.len();
                    parts.expression(expression)?;
                    if !expression_token_matches(
                        &parts.text[start..],
                        expression,
                        &hint.value_location,
                    ) {
                        return Err(refuse(
                            "node hint value text does not match its expression",
                            Some(&hint.value_location),
                        ));
                    }
                }
            }
        }
        self.finish_card(depth, &card.location)
    }

    fn global(&mut self, card: &GlobalCard, depth: usize) -> SpiceResult<()> {
        if card.nodes.is_empty() {
            return Err(refuse("empty .global card", Some(&card.location)));
        }
        let mut parts = self.card(depth, &card.location)?;
        parts.word(".global")?;
        for global in &card.nodes {
            parts.word(node(&parts, &global.name, &card.location)?)?;
        }
        self.finish_card(depth, &card.location)
    }

    fn model(&mut self, model: &ModelCard, depth: usize) -> SpiceResult<()> {
        let location = &model.location;
        let base = model.base.as_str();
        if crate::devices::ModelFamily::from_base(base).is_none() {
            return Err(refuse(
                format!("model type {base:?} is outside the supported syntax"),
                Some(location),
            ));
        }
        let mut parts = self.card(depth, location)?;
        parts.word(".model")?;
        parts.word(node(&parts, &model.name, location)?)?;
        parts.word(base)?;
        if !model.parameters.is_empty() {
            parts.push('(')?;
        }
        for (index, parameter) in model.parameters.iter().enumerate() {
            if index != 0 {
                parts.push(' ')?;
            }
            match &parameter.kind {
                ParameterKind::NumericVector(entries)
                    if matches!(base, "cpl" | "cpllines")
                        && ["r", "l", "g", "c"].contains(&parameter.name.as_str()) =>
                {
                    append_cpl_material(&mut parts, parameter, entries)?;
                }
                ParameterKind::NumericVector(entries)
                    if crate::devices::code_model::source::SourceModel::from_base(base)
                        .and_then(|schema| schema.parameter(&parameter.name))
                        .is_some_and(
                            crate::devices::code_model::source::ParameterShape::vector,
                        ) =>
                {
                    append_transfer_vector(&mut parts, parameter, entries)?;
                }
                ParameterKind::Flag => {
                    let allowed = model_flags(base);
                    if !allowed.contains(&parameter.name.as_str()) || !parameter.value.is_empty() {
                        return Err(refuse(
                            format!("flag {:?} is not valid for a {base} model", parameter.name),
                            Some(location),
                        ));
                    }
                    parts.push_str(&parameter.name)?;
                }
                ParameterKind::Textual
                    if matches!(base, "nmos" | "pmos") && parameter.name == "version" =>
                {
                    parts.check_length(parameter.value.len())?;
                    if !matches!(
                        single_token(&parameter.value, &parameter.location),
                        Some(Lexeme {
                            kind: TokenKind::Word,
                            ..
                        })
                    ) {
                        return Err(refuse(
                            "MOS version must be one bare literal token",
                            Some(location),
                        ));
                    }
                    append_named_value(&mut parts, parameter, true)?;
                }
                ParameterKind::Expression(_) if parameter.name == "level" => {
                    return Err(refuse("expression-valued model level", Some(location)));
                }
                _ => append_named_value(&mut parts, parameter, false)?,
            }
        }
        if !model.parameters.is_empty() {
            parts.push(')')?;
        }
        self.finish_card(depth, location)
    }

    fn device(&mut self, device: &DeviceInstance, depth: usize) -> SpiceResult<()> {
        let location = &device.location;
        let designator = device.designator;
        let mut parts = self.card(depth, location)?;
        parts.check_length(device.name.len())?;
        if device
            .name
            .chars()
            .next()
            .is_none_or(|c| c.to_ascii_lowercase() != designator)
        {
            return Err(refuse(
                format!(
                    "instance {:?} does not start with '{designator}'",
                    device.name
                ),
                Some(location),
            ));
        }
        if matches!(designator, 'e'..='h')
            && device.parameters.first().is_some_and(|parameter| {
                matches!(parameter.name.as_str(), "value" | "table" | "poly")
            })
        {
            parts.check_length(device.parameters.len().saturating_sub(1))?;
            for parameter in &device.parameters {
                parts.check_length(parameter.value.len())?;
            }
        }
        let nonlinear = NonlinearForm::validate(device)?;
        if designator == 'b' || nonlinear.is_some() {
            behavioural_card(device, nonlinear, &mut parts)?;
            return self.finish_card(depth, location);
        }
        parts.word(node(&parts, &device.name, location)?)?;
        if designator == 'a' {
            code_model_ports(device, &mut parts)?;
            return self.finish_card(depth, location);
        }
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
            'p' => {
                for parameter in &device.parameters {
                    if parameter.name != "length"
                        || !matches!(
                            parameter.kind,
                            ParameterKind::Scalar | ParameterKind::Expression(_)
                        )
                    {
                        return Err(refuse(
                            "P/CPL instance only accepts length=",
                            Some(&parameter.location),
                        ));
                    }
                    named_value(&mut parts, parameter, false)?;
                }
            }
            'y' => {
                for p in &device.parameters {
                    if p.name != "len"
                        || !matches!(p.kind, ParameterKind::Scalar | ParameterKind::Expression(_))
                    {
                        return Err(refuse("TXL instance setter (only len=)", Some(&p.location)));
                    }
                    named_value(&mut parts, p, false)?;
                }
            }
            'x' => {
                for parameter in &device.parameters {
                    named_value(&mut parts, parameter, true)?;
                }
            }
            _ => transistor_parameters(device, &mut parts)?,
        }
        self.finish_card(depth, location)
    }
}

fn model_flags(base: &str) -> &'static [&'static str] {
    match base {
        "d" => &["d"],
        "npn" | "pnp" => &["npn", "pnp"],
        "nmos" | "pmos" => &["nmos", "pmos"],
        "sw" => &["sw"],
        "ltspice_switch" => &["oneway"],
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
    }
}

// Write and validate the common positional device fields before its setters.
/// Write ordered typed scalar/vector ports, retaining type changes within
/// vectors. Internal lowered A cards remain unrepresentable.
fn code_model_ports(device: &DeviceInstance, parts: &mut Card<'_>) -> SpiceResult<()> {
    let location = &device.location;
    if device.code_ports.is_empty() || !device.nodes.is_empty() || !device.parameters.is_empty() {
        return Err(refuse(
            "expected ordered typed A-instance ports",
            Some(location),
        ));
    }
    for port in &device.code_ports {
        if port.terminals.is_empty() || (!port.vector && port.terminals.len() != 1) {
            return Err(refuse(
                "invalid A-instance port cardinality",
                Some(location),
            ));
        }
        let mut kind = port.terminals[0].kind;
        if let Some(marker) = kind.marker() {
            parts.word(marker)?;
        }
        if port.vector {
            parts.word("[")?;
        } else if kind.marker().is_some() {
            parts.push('(')?;
        }
        for terminal in &port.terminals {
            if terminal.kind != kind {
                let marker = terminal.kind.marker().ok_or_else(|| {
                    refuse(
                        "unresolved type change inside A-instance vector",
                        Some(location),
                    )
                })?;
                parts.word(marker)?;
                kind = terminal.kind;
            }
            if terminal.negative.is_some() != kind.differential()
                || (terminal.inverted
                    && !matches!(kind, CodePortKind::Digital | CodePortKind::Default))
            {
                return Err(refuse("invalid A-instance terminal", Some(location)));
            }
            if terminal.inverted {
                parts.word("~")?;
            }
            parts.word(code_port_name(parts, &terminal.positive, location)?)?;
            if let Some(negative) = &terminal.negative {
                parts.word(code_port_name(parts, negative, location)?)?;
            }
        }
        if port.vector {
            parts.word("]")?;
        } else if kind.marker().is_some() {
            parts.push(')')?;
        }
    }
    let model = device
        .model
        .as_deref()
        .ok_or_else(|| refuse("A instance needs a declared model", Some(location)))?;
    parts.word(code_port_name(parts, model, location)?)
}

fn code_port_name<'a>(
    parts: &Card<'_>,
    value: &'a str,
    location: &SourceLoc,
) -> SpiceResult<&'a str> {
    if value.starts_with('%') || value.contains(['[', ']', '~']) {
        return Err(refuse("invalid A-instance name", Some(location)));
    }
    node(parts, value, location)
}

fn device_prefix<'a>(
    device: &'a DeviceInstance,
    parts: &mut Card<'_>,
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
        'p' => count >= 4 && count % 2 == 0,
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
        parts.word(node(parts, n, location)?)?;
    }
    let mut setters = device.parameters.as_slice();
    if designator == 'w' {
        // INP2W reads the controlling source before the model name.
        match setters.split_first() {
            Some((control, rest))
                if control.name == "control" && control.kind == ParameterKind::Instance =>
            {
                parts.word(node(parts, &control.value, &control.location)?)?;
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
        ('d' | 'q' | 'm' | 'x' | 's' | 'w' | 'j' | 'u' | 'o' | 'z' | 'y' | 'p', None) => {
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
            parts.word(node(parts, model, location)?)?;
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

fn single_token<'a>(text: &'a str, location: &SourceLoc) -> Option<Lexeme<'a>> {
    if text.is_empty() || strip_comment(text, false) != text {
        return None;
    }
    let mut cursor = 0;
    let token = scan_next(text, &mut cursor, |_| location.clone()).ok()??;
    (token.text == text && cursor == text.len()).then_some(token)
}

fn path_token<'a>(token: &'a Lexeme<'_>) -> Option<&'a str> {
    match &token.kind {
        TokenKind::Quoted(decoded) => Some(decoded),
        TokenKind::Word | TokenKind::Number(_) => Some(token.text),
        _ => None,
    }
}

fn quote_path(parts: &mut Card<'_>, path: &str, location: &SourceLoc) -> SpiceResult<()> {
    parts.check_length(path.len())?;
    if path.is_empty() || path.contains(['\n', '\r']) {
        return Err(refuse("empty or multiline include path", Some(location)));
    }
    if let Some(token) = single_token(path, location)
        && path_token(&token) == Some(path)
        && !matches!(token.kind, TokenKind::Quoted(_))
    {
        return parts.word(path);
    }
    parts.word("\"")?;
    for character in path.chars() {
        if matches!(character, '"' | '\\') {
            parts.push('\\')?;
        }
        parts.push(character)?;
    }
    parts.push('"')
}

/// A node/instance/model name: one word or number token.
fn node<'a>(parts: &Card<'_>, name: &'a str, location: &SourceLoc) -> SpiceResult<&'a str> {
    parts.check_length(name.len())?;
    match single_token(name, location) {
        Some(Lexeme {
            kind: TokenKind::Word | TokenKind::Number(_),
            ..
        }) => Ok(name),
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
    if !semantic_equal(&reparsed.root, &expression.root) {
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
fn expression_token_matches(
    text: &str,
    expression: &ParameterExpression,
    location: &SourceLoc,
) -> bool {
    let inner = match (expression.braced, expression.quoted) {
        (true, true) => text.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')),
        (true, false) => text.strip_prefix('{').and_then(|s| s.strip_suffix('}')),
        (false, false) => Some(text),
        (false, true) => None,
    };
    inner == Some(expression.text.as_str())
        && matches!(
            (
                single_token(text, location).map(|token| token.kind),
                expression.braced,
                expression.quoted
            ),
            (Some(TokenKind::Expression(_)), true, false)
                | (Some(TokenKind::Quoted(_)), true, true)
                | (Some(TokenKind::Word), false, false)
        )
}

fn check_value_expression(
    parts: &Card<'_>,
    text: &str,
    expression: &ParameterExpression,
    location: &SourceLoc,
) -> SpiceResult<()> {
    parts.check_length(expression.text.len())?;
    check_expression(expression, location)?;
    if !expression_token_matches(text, expression, location) {
        return Err(refuse(
            "value text does not match its expression",
            Some(location),
        ));
    }
    Ok(())
}

fn value_text<'a>(
    parts: &Card<'_>,
    parameter: &'a ParameterAssignment,
    text_ok: bool,
) -> SpiceResult<&'a str> {
    let location = &parameter.location;
    let bad = |what: &str| {
        refuse(
            format!("parameter {:?}: {what}", parameter.name),
            Some(location),
        )
    };
    parts.check_length(parameter.value.len())?;
    match &parameter.kind {
        ParameterKind::Scalar => match single_token(&parameter.value, location) {
            Some(Lexeme {
                kind: TokenKind::Number(v),
                ..
            }) if v.is_finite() => {}
            _ => return Err(bad("scalar value is not one finite numeric token")),
        },
        ParameterKind::Textual if text_ok => {
            if !matches!(
                single_token(&parameter.value, location),
                Some(Lexeme {
                    kind: TokenKind::Word | TokenKind::Quoted(_),
                    ..
                })
            ) {
                return Err(bad("textual value is not one word or quoted token"));
            }
        }
        ParameterKind::Expression(expression) => {
            check_value_expression(parts, &parameter.value, expression, location)?;
        }
        _ => return Err(bad("value kind is not allowed here")),
    }
    Ok(&parameter.value)
}

/// Write a transfer coefficient or initial-condition vector without
/// losing numeric expressions or source spelling.
fn append_transfer_vector(
    parts: &mut Card<'_>,
    parameter: &ParameterAssignment,
    entries: &[crate::netlist::ast::NumericVectorEntry],
) -> SpiceResult<()> {
    parts.push_str(&parameter.name)?;
    parts.push_str("=[")?;
    for (index, entry) in entries.iter().enumerate() {
        if index != 0 {
            parts.push(' ')?;
        }
        append_vector_entry(parts, entry)?;
    }
    parts.push(']')?;
    Ok(())
}

/// Write the evaluated independent CPL vector coefficients as ordinary
/// numeric tokens, retaining physical scale suffixes and rejecting injection.
fn append_cpl_material(
    parts: &mut Card<'_>,
    parameter: &ParameterAssignment,
    entries: &[crate::netlist::ast::NumericVectorEntry],
) -> SpiceResult<()> {
    parts.push_str(&parameter.name)?;
    parts.push('=')?;
    for (position, entry) in entries.iter().enumerate() {
        if position != 0 {
            parts.push(' ')?;
        }
        match single_token(&entry.source.text, &entry.source.location) {
            Some(Lexeme {
                kind: TokenKind::Number(value),
                ..
            }) if value.is_finite() => {
                parts.push_str(&entry.source.text)?;
            }
            _ => {
                return Err(refuse(
                    "CPL vector coefficient must be a finite literal",
                    Some(&entry.source.location),
                ));
            }
        }
    }
    Ok(())
}

fn append_named_value(
    parts: &mut Card<'_>,
    parameter: &ParameterAssignment,
    text_ok: bool,
) -> SpiceResult<()> {
    let location = &parameter.location;
    parts.check_length(parameter.name.len())?;
    if !matches!(
        single_token(&parameter.name, location),
        Some(Lexeme {
            kind: TokenKind::Word,
            ..
        })
    ) {
        return Err(refuse(
            format!("parameter name {:?} is not a plain word", parameter.name),
            Some(location),
        ));
    }
    parts.push_str(&parameter.name)?;
    parts.push('=')?;
    parts.push_str(value_text(parts, parameter, text_ok)?)
}

fn named_value(
    parts: &mut Card<'_>,
    parameter: &ParameterAssignment,
    text_ok: bool,
) -> SpiceResult<()> {
    parts.push(' ')?;
    append_named_value(parts, parameter, text_ok)
}

fn positioned_number<'a>(
    parts: &Card<'_>,
    value: &'a PositionedValue,
    location: &SourceLoc,
) -> SpiceResult<&'a str> {
    parts.check_length(value.text.len())?;
    match single_token(&value.text, location) {
        Some(Lexeme {
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

fn numeric_vector_text<'a>(
    parts: &mut Card<'_>,
    prefix: &str,
    fields: impl IntoIterator<Item = &'a PositionedValue>,
    separator: char,
    location: &SourceLoc,
) -> SpiceResult<()> {
    parts.word(prefix)?;
    parts.push('(')?;
    for (index, field) in fields.into_iter().enumerate() {
        if index != 0 {
            parts.push(separator)?;
        }
        parts.push_str(positioned_number(parts, field, location)?)?;
    }
    parts.push(')')
}

fn passive_parameters(device: &DeviceInstance, parts: &mut Card<'_>) -> SpiceResult<()> {
    let designator = device.designator;
    let primary = match designator {
        'r' => "resistance",
        'c' => "capacitance",
        _ => "inductance",
    };
    let allowed = |name: &str| match (designator, name) {
        (_, name) if crate::netlist::network_parameter(designator, name) => true,
        (_, "temp" | "dtemp" | "m" | "tc1" | "tc2" | "scale")
        | ('r' | 'c', "w" | "l" | "bv_max")
        | ('r', "ac" | "tc" | "tce" | "noisy")
        | ('c', "charge")
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
    if device.model.is_none()
        && !device
            .parameters
            .iter()
            .any(|p| p.name == primary || (designator == 'c' && p.name == "charge"))
    {
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
        if matches!(parameter.kind, ParameterKind::Behavioural(_)) {
            if parameter.name != primary && !(designator == 'c' && parameter.name == "charge") {
                return Err(refuse(
                    "invalid behavioural passive setter",
                    Some(&parameter.location),
                ));
            }
            let text = behavioural_text(parts, parameter)?;
            parts.push(' ')?;
            parts.push_str(if parameter.name == "charge" {
                "q"
            } else {
                primary
            })?;
            parts.push('=')?;
            parts.push_str(text)?;
        } else if positional && index == 0 {
            parts.word(value_text(parts, parameter, false)?)?;
        } else {
            named_value(parts, parameter, false)?;
        }
    }
    Ok(())
}

/// S/W (`parser/switch.rs`): only bare `on`/`off` flags follow the model, in
/// their written order; W's `control` was already written before the model.
/// U (`parser/urc.rs`): the ordered `l=`/`n=` setters, nothing else.
fn urc_parameters(device: &DeviceInstance, parts: &mut Card<'_>) -> SpiceResult<()> {
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
        named_value(parts, parameter, false)?;
    }
    Ok(())
}

fn switch_parameters(
    setters: &[ParameterAssignment],
    designator: char,
    parts: &mut Card<'_>,
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
        parts.word(&parameter.name)?;
    }
    Ok(())
}

/// T (`parser/tline.rs`): `TRApTable` scalars and the `ic` vector, in order.
fn tline_parameters(device: &DeviceInstance, parts: &mut Card<'_>) -> SpiceResult<()> {
    const SCALARS: [&str; 11] = [
        "z0", "zo", "td", "f", "nl", "v1", "v2", "i1", "i2", "rel", "abs",
    ];
    line_parameters(device, &SCALARS, parts)
}

/// O (`parser/ltra.rs`): `LTRApTable` scalars and the `ic` vector, in order.
fn ltra_parameters(device: &DeviceInstance, parts: &mut Card<'_>) -> SpiceResult<()> {
    line_parameters(device, &["v1", "v2", "i1", "i2"], parts)
}

/// The ordered scalar setters and `ic` vector of a T or O line.
fn line_parameters(
    device: &DeviceInstance,
    scalars: &[&str],
    parts: &mut Card<'_>,
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
                numeric_vector_text(
                    parts,
                    "ic=",
                    components.iter().map(|component| &component.value),
                    ',',
                    location,
                )?;
            }
            ParameterKind::Scalar | ParameterKind::Expression(_)
                if scalars.contains(&parameter.name.as_str()) =>
            {
                named_value(parts, parameter, false)?;
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
fn controlled_parameters(device: &DeviceInstance, parts: &mut Card<'_>) -> SpiceResult<()> {
    let designator = device.designator;
    let mut setters = device.parameters.as_slice();
    if matches!(designator, 'f' | 'h') {
        match setters.split_first() {
            Some((control, rest))
                if control.name == "control" && control.kind == ParameterKind::Instance =>
            {
                parts.word(node(parts, &control.value, &control.location)?)?;
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
        [gain] if gain.name == "gain" => parts.word(value_text(parts, gain, false)?)?,
        [first, rest @ ..] if first.name == "gain" => {
            if rest.first().is_some_and(|next| next.name != "m") {
                return Err(unrepresentable());
            }
            for parameter in setters {
                named_value(parts, parameter, false)?;
            }
        }
        [tail @ .., gain]
            if gain.name == "gain" && tail.first().is_some_and(|first| first.name == "m") =>
        {
            parts.word(value_text(parts, gain, false)?)?;
            for parameter in tail {
                named_value(parts, parameter, false)?;
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
fn behavioural_text<'a>(
    parts: &Card<'_>,
    parameter: &'a ParameterAssignment,
) -> SpiceResult<&'a str> {
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
    parts.check_length(expression.text.len())?;
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
    if expression.text.contains(['\n', '\r']) || !semantic_equal(&reparsed.root, &expression.root) {
        return Err(refuse(
            format!(
                "behavioural expression text {:?} does not match its stored syntax tree",
                expression.text
            ),
            Some(location),
        ));
    }
    Ok(&expression.text)
}

/// B sources and the nonlinear E/G/F/H forms (`parser/behavioural.rs`):
/// `b1 n+ n- v=<text> [setters]`, `e1 n+ n- value=<text> [setters]`,
/// `e1 n+ n- table {<text>} = (x, y) ... [m=..]`,
/// `e1 n+ n- nc+ nc- table=(x0, y0, ...)` and
/// `e1 n+ n- poly(n) <controls> <coefficients> [m=..]`.
fn behavioural_card(
    device: &DeviceInstance,
    nonlinear: Option<NonlinearForm<'_>>,
    parts: &mut Card<'_>,
) -> SpiceResult<()> {
    let location = &device.location;
    let designator = device.designator;
    if device.model.is_some() {
        return Err(refuse("behavioural source with a model", Some(location)));
    }
    parts.word(node(parts, &device.name, location)?)?;
    let [positive, negative, controls @ ..] = device.nodes.as_slice() else {
        return Err(refuse(
            format!("'{designator}' instance with fewer than two nodes"),
            Some(location),
        ));
    };
    parts.word(node(parts, positive, location)?)?;
    parts.word(node(parts, negative, location)?)?;
    match nonlinear {
        Some(NonlinearForm::Value(form)) => {
            parts.word("value=")?;
            parts.push_str(behavioural_text(parts, form.input)?)?;
            for parameter in form.setters {
                named_value(parts, parameter, false)?;
            }
        }
        Some(NonlinearForm::Table(form)) => behavioural_table(device, form, parts)?,
        Some(NonlinearForm::Poly(form)) => behavioural_poly(device, form, parts)?,
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
            parts.word(&first.name)?;
            parts.push('=')?;
            parts.push_str(behavioural_text(parts, first)?)?;
            for parameter in setters {
                behavioural_setter(parts, parameter, designator, B_SETTERS)?;
            }
        }
    }
    Ok(())
}

fn behavioural_setter(
    parts: &mut Card<'_>,
    parameter: &ParameterAssignment,
    designator: char,
    allowed: &[&str],
) -> SpiceResult<()> {
    if allowed.contains(&parameter.name.as_str()) {
        named_value(parts, parameter, false)
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
    parts: &mut Card<'_>,
) -> SpiceResult<()> {
    match form.input {
        TableInput::Voltage([positive, negative]) => {
            parts.word(node(parts, positive, &device.location)?)?;
            parts.word(node(parts, negative, &device.location)?)?;
            parts.word("table=(")?;
            for (index, (x, y)) in form.points.into_iter().enumerate() {
                if index != 0 {
                    parts.push_str(", ")?;
                }
                parts.push_str(value_text(parts, x.parameter, false)?)?;
                parts.push_str(", ")?;
                parts.push_str(value_text(parts, y.parameter, false)?)?;
            }
            parts.push(')')?;
        }
        TableInput::Expression(parameter, _) => {
            parts.word("table {")?;
            parts.push_str(behavioural_text(parts, parameter)?)?;
            parts.push_str("} =")?;
            for (x, y) in form.points {
                parts.word("(")?;
                parts.push_str(value_text(parts, x.parameter, false)?)?;
                parts.push_str(", ")?;
                parts.push_str(value_text(parts, y.parameter, false)?)?;
                parts.push(')')?;
            }
        }
    }
    for parameter in form.setters {
        named_value(parts, parameter, false)?;
    }
    Ok(())
}

fn behavioural_poly(
    device: &DeviceInstance,
    form: PolyForm<'_>,
    parts: &mut Card<'_>,
) -> SpiceResult<()> {
    parts.word("poly(")?;
    parts.push_str(value_text(parts, form.spelling, false)?)?;
    parts.push(')')?;
    match form.controls {
        PolyControls::Voltage(pairs) => {
            for control in pairs.iter().flatten() {
                parts.word(node(parts, control, &device.location)?)?;
            }
        }
        PolyControls::Current(sources) => {
            for control in sources {
                parts.word(node(parts, &control.value, &control.location)?)?;
            }
        }
    }
    for coefficient in form.coefficients {
        parts.word(value_text(parts, coefficient.parameter, false)?)?;
    }
    for parameter in form.setters {
        named_value(parts, parameter, false)?;
    }
    Ok(())
}

/// K (`parser/mutual.rs`): the inductor references `inductor1`, `inductor2`,
/// … in order, then exactly one `coefficient`, written positionally (a named
/// `k=`/`coefficient=` setter re-parses to the same setter).
fn mutual_parameters(device: &DeviceInstance, parts: &mut Card<'_>) -> SpiceResult<()> {
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
        parts.word(node(parts, &inductor.value, &inductor.location)?)?;
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
    parts.word(value_text(parts, coupling, false)?)?;
    Ok(())
}

fn source_parameters(device: &DeviceInstance, parts: &mut Card<'_>) -> SpiceResult<()> {
    let mut parameters = device.parameters.iter().peekable();
    while let Some(parameter) = parameters.next() {
        let location = &parameter.location;
        match (&parameter.kind, parameter.name.as_str()) {
            (ParameterKind::Scalar | ParameterKind::Expression(_), "dc") => {
                parts.word("dc")?;
                parts.word(value_text(parts, parameter, false)?)?;
            }
            (ParameterKind::Scalar | ParameterKind::Expression(_), "acmag") => {
                let Some(phase) = parameters.next_if(|next| next.name == "acphase") else {
                    return Err(refuse("acmag without acphase", Some(location)));
                };
                parts.word("ac")?;
                parts.word(value_text(parts, parameter, false)?)?;
                parts.word(value_text(parts, phase, false)?)?;
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
                parts.word(name.trim_end_matches("mag"))?;
                parts.word(value_text(parts, parameter, false)?)?;
                parts.word(value_text(parts, phase, false)?)?;
            }
            (ParameterKind::Waveform(waveform), name) => {
                waveform_text(parts, waveform, name, location)?;
            }
            // PWL delay/repeat setters stay separate ordered scalars (vsrc.c).
            (ParameterKind::Scalar | ParameterKind::Expression(_), "r" | "td") => {
                named_value(parts, parameter, false)?;
            }
            (ParameterKind::Scalar | ParameterKind::Expression(_), name)
                if crate::netlist::network_parameter(device.designator, name) =>
            {
                named_value(parts, parameter, false)?;
            }
            // RFSPICE port setters of a V source, ordered like the PWL ones.
            (
                ParameterKind::Scalar | ParameterKind::Expression(_),
                "portnum" | "z0" | "pwr" | "freq" | "phase",
            ) if device.designator == 'v' => {
                named_value(parts, parameter, false)?;
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

fn append_vector_entry(parts: &mut Card<'_>, entry: &NumericVectorEntry) -> SpiceResult<()> {
    if let Some(expression) = &entry.expression {
        check_value_expression(
            parts,
            &entry.source.text,
            expression,
            &entry.source.location,
        )?;
        parts.push_str(&entry.source.text)
    } else {
        parts.push_str(positioned_number(
            parts,
            &entry.source,
            &entry.source.location,
        )?)
    }
}

fn waveform_vector_text<'a>(
    parts: &mut Card<'_>,
    keyword: &str,
    fields: impl IntoIterator<Item = &'a NumericVectorEntry>,
) -> SpiceResult<()> {
    parts.word(keyword)?;
    parts.push('(')?;
    for (index, entry) in fields.into_iter().enumerate() {
        if index != 0 {
            parts.push(' ')?;
        }
        append_vector_entry(parts, entry)?;
    }
    parts.push(')')
}

fn waveform_text(
    parts: &mut Card<'_>,
    waveform: &SourceWaveform,
    name: &str,
    location: &SourceLoc,
) -> SpiceResult<()> {
    let kind = waveform.kind();
    let keyword = kind.keyword(name);
    if name != keyword {
        return Err(refuse(
            format!("waveform setter named {name:?} holds a {keyword}"),
            Some(location),
        ));
    }
    match waveform {
        SourceWaveform::Pulse(pulse) => {
            let optional = [
                &pulse.delay,
                &pulse.rise,
                &pulse.fall,
                &pulse.width,
                &pulse.period,
                &pulse.count,
            ];
            let mut omitted = false;
            for value in optional {
                if value.is_none() {
                    omitted = true;
                } else if omitted {
                    return Err(refuse(
                        "PULSE field present after an omitted field",
                        Some(location),
                    ));
                }
            }
            waveform_vector_text(
                parts,
                keyword,
                [&pulse.initial, &pulse.pulsed]
                    .into_iter()
                    .chain(optional.into_iter().filter_map(Option::as_ref)),
            )
        }
        SourceWaveform::Pwl(points) => {
            if points.is_empty() || points.len() > kind.maximum_fields() / 2 {
                return Err(refuse("PWL needs 1 to 2048 points", Some(location)));
            }
            waveform_vector_text(
                parts,
                keyword,
                points.iter().flat_map(|p| [&p.time, &p.value]),
            )
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
            waveform_vector_text(parts, keyword, &function.values)
        }
        SourceWaveform::Stochastic { values, .. } => {
            let maximum = kind.maximum_fields();
            if !(2..=maximum).contains(&values.len()) {
                return Err(refuse(
                    format!("{keyword} requires 2 to {maximum} fields"),
                    Some(location),
                ));
            }
            waveform_vector_text(parts, keyword, values)
        }
    }
}

fn transistor_parameters(device: &DeviceInstance, parts: &mut Card<'_>) -> SpiceResult<()> {
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
                parts.word("off")?;
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
                numeric_vector_text(
                    parts,
                    "ic=",
                    components.iter().map(|component| &component.value),
                    ',',
                    location,
                )?;
            }
            ParameterKind::Scalar | ParameterKind::Expression(_)
                if scalars.contains(&parameter.name.as_str()) =>
            {
                named_value(parts, parameter, false)?;
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
