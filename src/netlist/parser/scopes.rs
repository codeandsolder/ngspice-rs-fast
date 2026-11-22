//! Ordered scope assembly; parsing X references is deliberately not elaboration.

use super::ParsedDeck;
use super::grammar::{self, ParsedCard};
use super::save::OutputCard;
use crate::netlist::ast::{
    AnalysisCard, DeviceInstance, FourierCard, FuncCard, GlobalCard, IncludeDirective, MeasureCard,
    ModelCard, Netlist, NodeHintCard, OptionCard, OutputCards, ParamCard, ScopedCard,
    ScopedCardKind, Subcircuit,
};
use crate::netlist::card::{DotCommand, RawCard};
use crate::netlist::source::Deck;
use crate::primitives::{SourceLoc, SpiceError, SpiceResult};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

pub(super) struct InputCard {
    pub source: Arc<RawCard>,
    pub include_chain: Arc<[SourceLoc]>,
    pub resolved_path: Option<PathBuf>,
    pub selected_section: Option<crate::netlist::ast::LibrarySection>,
}

impl From<RawCard> for InputCard {
    fn from(source: RawCard) -> Self {
        Self {
            source: Arc::new(source),
            include_chain: Arc::default(),
            resolved_path: None,
            selected_section: None,
        }
    }
}

#[derive(Default)]
struct Scope {
    devices: Vec<DeviceInstance>,
    models: Vec<ModelCard>,
    subcircuits: Vec<Subcircuit>,
    analyses: Vec<AnalysisCard>,
    includes: Vec<IncludeDirective>,
    options: Vec<OptionCard>,
    globals: Vec<GlobalCard>,
    initial_conditions: Vec<NodeHintCard>,
    nodesets: Vec<NodeHintCard>,
    params: Vec<ParamCard>,
    functions: Vec<FuncCard>,
    output: OutputCards,
    measurements: Vec<MeasureCard>,
    fourier: Vec<FourierCard>,
    cards: Vec<ScopedCard>,
}

impl Scope {
    /// Output requests travel beside the netlist, with scope-local ordinals.
    /// Body-local save/measure/Fourier requests remain attached for expansion;
    /// print requests in bodies remain explicitly unsupported.
    fn record_output(
        &mut self,
        output: OutputCard,
        card: &RawCard,
        opening: Option<(&str, &SourceLoc)>,
    ) -> SpiceResult<ScopedCardKind> {
        match output {
            OutputCard::Save(save) => Ok(ScopedCardKind::Save(push_index(
                &mut self.output.saves,
                save,
            ))),
            OutputCard::Print(print) => {
                reject_in_body(
                    card,
                    opening,
                    ".print",
                    "src/frontend/dotcards.c (ft_savedotargs/com_save2)",
                )?;
                Ok(ScopedCardKind::Print(push_index(
                    &mut self.output.prints,
                    print,
                )))
            }
        }
    }

    fn into_definition(self, mut definition: Subcircuit) -> SpiceResult<Subcircuit> {
        definition.end_location = self
            .cards
            .last()
            .ok_or_else(|| {
                SpiceError::parse(
                    definition.location.clone(),
                    "closed subcircuit has no .ends card",
                )
            })?
            .source
            .location
            .clone();
        definition.devices = self.devices;
        definition.models = self.models;
        definition.subcircuits = self.subcircuits;
        definition.analyses = self.analyses;
        definition.includes = self.includes;
        definition.params = self.params;
        definition.functions = self.functions;
        definition.options = self.options;
        definition.globals = self.globals;
        definition.initial_conditions = self.initial_conditions;
        definition.nodesets = self.nodesets;
        definition.output = self.output;
        definition.measurements = self.measurements;
        definition.fourier = self.fourier;
        definition.cards = self.cards;
        Ok(definition)
    }
}

pub(super) fn assemble(
    deck: &Deck,
    cards: &[SpiceResult<InputCard>],
    auto_gnd: bool,
) -> SpiceResult<ParsedDeck> {
    let mut cursor = 0;
    let scope = scope(cards, &mut cursor, auto_gnd, &BTreeSet::new(), None, 0)?;
    Ok(ParsedDeck {
        netlist: Netlist {
            title: deck.title.clone(),
            path: deck.path.clone(),
            location: deck.title_location.clone(),
            devices: scope.devices,
            models: scope.models,
            subcircuits: scope.subcircuits,
            analyses: scope.analyses,
            includes: scope.includes,
            cards: scope.cards,
            params: scope.params,
            functions: scope.functions,
            options: scope.options,
            globals: scope.globals,
            initial_conditions: scope.initial_conditions,
            nodesets: scope.nodesets,
        },
        output: scope.output,
        measurements: scope.measurements,
        fourier: scope.fourier,
    })
}

// Read-only forward-name scan, bounded to this body and excluding children and
// control scripts. Ancestors are visible; siblings/children never leak names.
fn model_names(
    cards: &[SpiceResult<InputCard>],
    start: usize,
    inherited: &BTreeSet<String>,
) -> BTreeSet<String> {
    let mut names = inherited.clone();
    let mut depth = 0usize;
    let mut in_control = false;
    for entry in &cards[start..] {
        let Ok(entry) = entry else { continue };
        match entry.source.dot_command() {
            Some(DotCommand::Control) => in_control = true,
            Some(DotCommand::Endc) => in_control = false,
            Some(DotCommand::Subckt) if !in_control => depth += 1,
            Some(DotCommand::Ends) if !in_control => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            Some(DotCommand::End) => break,
            Some(DotCommand::Model) if !in_control && depth == 0 => {
                if let Some(name) = entry.source.tokens.get(1).filter(|t| t.is_name_like()) {
                    names.insert(name.text.to_ascii_lowercase());
                }
            }
            _ => {}
        }
    }
    names
}

#[derive(Default, PartialEq, Eq)]
enum ControlPhase {
    #[default]
    Outside,
    BeforeRun,
    AfterRun,
    Quit,
}

#[derive(Default)]
struct ControlScript {
    phase: ControlPhase,
    deck_ran: bool,
}

impl ControlScript {
    /// Parse a normal card, a supported control command, or a block delimiter.
    /// A consumed command/delimiter has no typed card, but remains in deck order.
    fn parse(
        &mut self,
        card: &RawCard,
        in_body: bool,
        auto_gnd: bool,
        declared: &BTreeSet<String>,
    ) -> SpiceResult<Option<ParsedCard>> {
        if matches!(
            card.dot_command(),
            Some(DotCommand::Control | DotCommand::Endc)
        ) {
            let begin = card.dot_command() == Some(&DotCommand::Control);
            if in_body || card.tokens.len() != 1 || begin == (self.phase != ControlPhase::Outside) {
                return Err(SpiceError::parse(
                    card.location.clone(),
                    "invalid or nested .control/.endc settings block",
                ));
            }
            self.phase = if begin {
                ControlPhase::BeforeRun
            } else {
                ControlPhase::Outside
            };
            return Ok(None);
        }
        if self.phase == ControlPhase::Outside {
            return grammar::parse_card(card, auto_gnd, declared).map(Some);
        }
        if self.phase == ControlPhase::Quit {
            return Err(SpiceError::not_yet_ported(
                format!("{}: commands after quit", card.location),
                "src/frontend/control.c",
            ));
        }
        if let Some(head) = card.tokens.first()
            && (head.is_keyword("run") || head.is_keyword("quit"))
        {
            if card.tokens.len() != 1 || (head.is_keyword("run") && self.deck_ran) {
                return Err(SpiceError::not_yet_ported(
                    format!(
                        "{}: repeated or argument-bearing control run/quit",
                        card.location
                    ),
                    "src/frontend/runcoms.c",
                ));
            }
            if head.is_keyword("run") {
                self.phase = ControlPhase::AfterRun;
                self.deck_ran = true;
            } else {
                if self.phase != ControlPhase::AfterRun {
                    return Err(SpiceError::not_yet_ported(
                        format!("{}: quit before run", card.location),
                        "src/frontend/inp.c, src/frontend/control.c",
                    ));
                }
                self.phase = ControlPhase::Quit;
            }
            return Ok(None);
        }
        if card.tokens.first().is_some_and(|t| t.is_keyword("fourier")) {
            if self.phase != ControlPhase::AfterRun {
                return Err(SpiceError::parse(
                    card.location.clone(),
                    "fourier needs a preceding run in this control block",
                ));
            }
            let mut converted = card.clone();
            converted.tokens[0].text = ".four".into();
            converted.kind = crate::netlist::card::CardKind::DotCommand(DotCommand::Analysis(
                crate::primitives::AnalysisKind::Fourier,
            ));
            let mut parsed = grammar::parse_card(&converted, auto_gnd, declared)?;
            if let ParsedCard::Fourier(card) = &mut parsed {
                card.frontend_command = true;
            }
            return Ok(Some(parsed));
        }
        if self.deck_ran && card.tokens.first().is_some_and(|t| t.is_keyword("set")) {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: set after run; set Fourier variables before run",
                    card.location
                ),
                "src/frontend/variable.c",
            ));
        }
        grammar::parse_frontend_setting(card, auto_gnd, declared).map(Some)
    }
}

/// Append a scoped item and return its stable ordinal in that typed collection.
fn push_index<T>(items: &mut Vec<T>, item: T) -> usize {
    let index = items.len();
    items.push(item);
    index
}

fn scope(
    cards: &[SpiceResult<InputCard>],
    cursor: &mut usize,
    auto_gnd: bool,
    inherited: &BTreeSet<String>,
    opening: Option<(&str, &SourceLoc)>,
    depth: usize,
) -> SpiceResult<Scope> {
    let declared = model_names(cards, *cursor, inherited);
    let mut result = Scope::default();
    let mut definitions = BTreeSet::new();
    let mut control = ControlScript::default();
    while let Some(entry) = cards.get(*cursor) {
        let entry = entry.as_ref().map_err(Clone::clone)?;
        *cursor += 1;
        let card = &entry.source;
        let Some(parsed) = control.parse(card, opening.is_some(), auto_gnd, &declared)? else {
            result.cards.push(ordered(entry, ScopedCardKind::Control));
            continue;
        };
        let kind = match parsed {
            ParsedCard::Device(d) => ScopedCardKind::Device(push_index(&mut result.devices, d)),
            ParsedCard::Model(m) => ScopedCardKind::Model(push_index(&mut result.models, m)),
            ParsedCard::Analysis(a) => {
                ScopedCardKind::Analysis(push_index(&mut result.analyses, a))
            }
            ParsedCard::Options(o) => ScopedCardKind::Options(push_index(&mut result.options, o)),
            ParsedCard::Global(g) => ScopedCardKind::Global(push_index(&mut result.globals, g)),
            ParsedCard::InitialCondition(c) => {
                ScopedCardKind::InitialCondition(push_index(&mut result.initial_conditions, c))
            }
            ParsedCard::Nodeset(c) => ScopedCardKind::Nodeset(push_index(&mut result.nodesets, c)),
            ParsedCard::Param(p) => ScopedCardKind::Param(push_index(&mut result.params, p)),
            ParsedCard::Func(f) => ScopedCardKind::Func(push_index(&mut result.functions, f)),
            ParsedCard::Output(output) => result.record_output(output, card, opening)?,
            ParsedCard::Measure(measure) => {
                ScopedCardKind::Measure(push_index(&mut result.measurements, measure))
            }
            ParsedCard::Fourier(fourier) => {
                ScopedCardKind::Fourier(push_index(&mut result.fourier, fourier))
            }
            ParsedCard::Include(mut i) => {
                i.resolved_path.clone_from(&entry.resolved_path);
                i.selected_section.clone_from(&entry.selected_section);
                ScopedCardKind::Include(push_index(&mut result.includes, i))
            }
            ParsedCard::Subckt(s) => {
                if !definitions.insert(s.name.clone()) {
                    return Err(SpiceError::parse(
                        s.location,
                        format!("duplicate subcircuit definition '{}' in this scope", s.name),
                    ));
                }
                if depth >= 64 {
                    return Err(SpiceError::parse(
                        s.location,
                        "subcircuit nesting limit (64) exceeded",
                    ));
                }
                let body = scope(
                    cards,
                    cursor,
                    auto_gnd,
                    &declared,
                    Some((&s.name, &s.location)),
                    depth + 1,
                )?;
                let s = body.into_definition(s)?;
                ScopedCardKind::Subcircuit(push_index(&mut result.subcircuits, s))
            }
            ParsedCard::Ends(name) => {
                check_ends(name.as_deref(), opening, &card.location)?;
                result.cards.push(ordered(entry, ScopedCardKind::Ends));
                return Ok(result);
            }
            ParsedCard::End => {
                check_end(opening, &card.location)?;
                result.cards.push(ordered(entry, ScopedCardKind::End));
                return Ok(result);
            }
            ParsedCard::LibStart(_) | ParsedCard::LibEnd(_) => {
                return Err(SpiceError::parse(
                    card.location.clone(),
                    "library section markers require .lib path section resolution",
                ));
            }
        };
        result.cards.push(ordered(entry, kind));
    }
    if control.phase != ControlPhase::Outside {
        return Err(SpiceError::circuit("missing .endc for settings block"));
    }
    if let Some((name, location)) = opening {
        return Err(SpiceError::parse(
            location.clone(),
            format!("missing .ends for '{name}'"),
        ));
    }
    Ok(result)
}

fn check_ends(
    name: Option<&str>,
    opening: Option<(&str, &SourceLoc)>,
    location: &SourceLoc,
) -> SpiceResult<()> {
    let Some((expected, _)) = opening else {
        return Err(SpiceError::parse(location.clone(), "unmatched .ends"));
    };
    if name.is_some_and(|name| name != expected) {
        return Err(SpiceError::parse(
            location.clone(),
            format!("mismatched .ends: expected '{expected}'"),
        ));
    }
    Ok(())
}

fn check_end(opening: Option<(&str, &SourceLoc)>, location: &SourceLoc) -> SpiceResult<()> {
    if let Some((name, _)) = opening {
        return Err(SpiceError::parse(
            location.clone(),
            format!(".end before .ends for '{name}'"),
        ));
    }
    Ok(())
}

fn ordered(entry: &InputCard, kind: ScopedCardKind) -> ScopedCard {
    ScopedCard {
        kind,
        source: Arc::clone(&entry.source),
        include_chain: Arc::clone(&entry.include_chain),
    }
}

// Body-local options/globals/output cards need per-subcircuit storage and
// flattening rules (inpcom.c/subckt.c); until then they must not be dropped or
// hoisted silently.
fn reject_in_body(
    card: &RawCard,
    opening: Option<(&str, &SourceLoc)>,
    what: &str,
    c_reference: &'static str,
) -> SpiceResult<()> {
    if opening.is_some() {
        return Err(SpiceError::not_yet_ported(
            format!("{}: {what} inside a .subckt body", card.location),
            c_reference,
        ));
    }
    Ok(())
}
