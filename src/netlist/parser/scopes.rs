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
use std::borrow::Cow;
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
    let models = ModelIndex::build(cards);
    let mut cursor = 0;
    let scope = scope(cards, &mut cursor, auto_gnd, &models, None, 0)?;
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

const MAX_SCOPE_DEPTH: usize = 64;

/// Only declarations local to this lexical body. The parent edge represents
/// ancestor visibility without copying any inherited names.
#[derive(Debug, Default)]
struct LexicalModels {
    body_start: usize,
    parent: Option<usize>,
    names: BTreeSet<String>,
    bins: BTreeSet<String>,
}

impl LexicalModels {
    fn insert(&mut self, name: &str) {
        let name = name.to_ascii_lowercase();
        // Bin candidates are identified during indexing, not by scanning a
        // prefix range on every MOS terminal/model disambiguation.
        if let Some((base, suffix)) = name.rsplit_once('.')
            && !suffix.is_empty()
            && suffix.bytes().all(|byte| byte.is_ascii_digit())
        {
            self.bins.insert(base.to_owned());
        }
        self.names.insert(name);
    }
}

/// Occurrence-based structural index, independent of semantic validity.
/// In particular two visits to the same included RawCard have distinct
/// scope identities. The ordered body-start keys are unique and monotone.
struct ModelIndex {
    scopes: Vec<LexicalModels>,
    #[cfg(test)]
    inspected_cards: usize,
}

impl ModelIndex {
    /// One read-only pass, with the same lexical control and terminator
    /// policy as the semantic parser's former forward-name scans. Errors
    /// remain in the input for ordered replay; this pass never emits one.
    /// Do not allocate scope records beyond the semantic nesting budget.
    fn build(cards: &[SpiceResult<InputCard>]) -> Self {
        let mut scopes = vec![LexicalModels::default()];
        let mut active = 0;
        let mut depth = 0usize;
        let mut in_control = false;
        #[cfg(test)]
        let mut inspected_cards = 0;
        for (ordinal, entry) in cards.iter().enumerate() {
            #[cfg(test)]
            {
                inspected_cards += 1;
            }
            let Ok(entry) = entry else { continue };
            match entry.source.dot_command() {
                Some(DotCommand::Control) => in_control = true,
                Some(DotCommand::Endc) => in_control = false,
                Some(DotCommand::Subckt) if !in_control => {
                    depth += 1;
                    if depth <= MAX_SCOPE_DEPTH {
                        let child = LexicalModels {
                            body_start: ordinal + 1,
                            parent: Some(active),
                            ..LexicalModels::default()
                        };
                        active = scopes.len();
                        scopes.push(child);
                    }
                }
                Some(DotCommand::Ends) if !in_control => {
                    if depth == 0 {
                        break;
                    }
                    if depth <= MAX_SCOPE_DEPTH
                        && let Some(parent) = scopes[active].parent
                    {
                        active = parent;
                    }
                    depth -= 1;
                }
                Some(DotCommand::End) => break,
                Some(DotCommand::Model) if !in_control && depth <= MAX_SCOPE_DEPTH => {
                    if let Some(name) = entry.source.tokens.get(1).filter(|t| t.is_name_like()) {
                        scopes[active].insert(&name.text);
                    }
                }
                _ => {}
            }
        }
        Self {
            scopes,
            #[cfg(test)]
            inspected_cards,
        }
    }

    fn visibility(&self, body_start: usize) -> SpiceResult<ModelVisibility<'_>> {
        let scope = self
            .scopes
            .binary_search_by_key(&body_start, |scope| scope.body_start)
            .map_err(|_| SpiceError::circuit("missing lexical model scope"))?;
        Ok(ModelVisibility {
            scopes: &self.scopes,
            scope: Some(scope),
        })
    }
}

/// Borrowed model membership for grammar lookahead. Non-body grammar
/// entry points (include directives, resolved measures) need no declarations.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ModelVisibility<'a> {
    scopes: &'a [LexicalModels],
    scope: Option<usize>,
}

impl ModelVisibility<'_> {
    pub(super) fn contains(self, name: &str, allow_bins: bool) -> bool {
        let name = if name.bytes().any(|byte| byte.is_ascii_uppercase()) {
            Cow::Owned(name.to_ascii_lowercase())
        } else {
            Cow::Borrowed(name)
        };
        let mut current = self.scope;
        while let Some(index) = current {
            let scope = &self.scopes[index];
            if scope.names.contains(name.as_ref())
                || (allow_bins && scope.bins.contains(name.as_ref()))
            {
                return true;
            }
            current = scope.parent;
        }
        false
    }
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
        declared: ModelVisibility<'_>,
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
    models: &ModelIndex,
    opening: Option<(&str, &SourceLoc)>,
    depth: usize,
) -> SpiceResult<Scope> {
    let declared = models.visibility(*cursor)?;
    let mut result = Scope::default();
    let mut definitions = BTreeSet::new();
    let mut control = ControlScript::default();
    while let Some(entry) = cards.get(*cursor) {
        let entry = entry.as_ref().map_err(Clone::clone)?;
        *cursor += 1;
        let card = &entry.source;
        let Some(parsed) = control.parse(card, opening.is_some(), auto_gnd, declared)? else {
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
                if depth >= MAX_SCOPE_DEPTH {
                    return Err(SpiceError::parse(
                        s.location,
                        "subcircuit nesting limit (64) exceeded",
                    ));
                }
                let body = scope(
                    cards,
                    cursor,
                    auto_gnd,
                    models,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::source::parse_deck_text;
    use std::fmt::Write;
    use std::path::Path;

    fn cards(body: &str) -> Vec<SpiceResult<InputCard>> {
        let deck = parse_deck_text(Path::new("scope-index.cir"), &format!("Title\n{body}"));
        deck.lines
            .iter()
            .map(|line| RawCard::parse(line).map(Into::into))
            .collect()
    }

    // Former implementation, kept only as an independent differential
    // oracle for lexical membership and the old operation count.
    fn reference(
        cards: &[SpiceResult<InputCard>],
        start: usize,
        inherited: &BTreeSet<String>,
    ) -> (BTreeSet<String>, usize) {
        let mut names = inherited.clone();
        let mut depth = 0usize;
        let mut in_control = false;
        let mut inspected = 0;
        for entry in &cards[start..] {
            inspected += 1;
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
        (names, inspected)
    }

    fn reference_contains(names: &BTreeSet<String>, candidate: &str, bins: bool) -> bool {
        let name = candidate.to_ascii_lowercase();
        names.contains(&name)
            || (bins
                && names.iter().any(|model| {
                    model
                        .strip_prefix(&name)
                        .and_then(|rest| rest.strip_prefix('.'))
                        .is_some_and(|suffix| {
                            !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit())
                        })
                }))
    }

    fn compare_index(
        cards: &[SpiceResult<InputCard>],
        index: &ModelIndex,
        candidates: &[&str],
    ) -> (usize, usize) {
        let mut visible = Vec::new();
        let mut visits = 0;
        let mut copied = 0;
        for scope in &index.scopes {
            let empty = BTreeSet::new();
            let inherited = scope.parent.map_or(&empty, |parent| &visible[parent]);
            copied += inherited.len();
            let (reference, inspected) = reference(cards, scope.body_start, inherited);
            visits += inspected;
            let view = index.visibility(scope.body_start).unwrap();
            for candidate in candidates {
                for bins in [false, true] {
                    assert_eq!(
                        view.contains(candidate, bins),
                        reference_contains(&reference, candidate, bins),
                        "body {} candidate {candidate} bins {bins}",
                        scope.body_start
                    );
                }
            }
            visible.push(reference);
        }
        (visits, copied)
    }

    #[test]
    fn index_preserves_lexical_membership_with_malformed_future_cards() {
        // Exhaust a deterministic mixture of structural/control/lexical
        // errors, including misplaced terminators and repeated model names.
        // No semantic validation is performed by the index itself.
        let choices = [
            ".subckt child a b",
            ".ends child",
            ".control",
            ".endc",
            ".model RM r",
            ".model Foo.12 NMOS",
            ".model foo.a NMOS",
            ".model deep.2.3 NMOS",
            ".model Qm NPN",
            "rbroken a {",
            "r1 a b RM",
            ".end",
            ".model",
            ".model rm r",
        ];
        let candidates = [
            "rm", "RM", "foo", "foo.12", "foo.a", "deep", "deep.2", "deep.2.3", "QM", "absent",
        ];
        let mut rng = 0x7f4a7c159e3779b9_u64;
        for _ in 0..512 {
            let mut body = String::new();
            for _ in 0..96 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let choice = usize::try_from(rng % u64::try_from(choices.len()).unwrap()).unwrap();
                writeln!(body, "{}", choices[choice]).unwrap();
            }
            let cards = cards(&body);
            let index = ModelIndex::build(&cards);
            compare_index(&cards, &index, &candidates);
            assert!(index.inspected_cards <= cards.len());
        }
    }

    #[test]
    fn wide_and_deep_scopes_store_each_local_declaration_once() {
        let mut body = String::new();
        const ROOTS: usize = 256;
        const SIBLINGS: usize = 256;
        for model in 0..ROOTS {
            writeln!(body, ".model root{model} r").unwrap();
        }
        for scope in 0..SIBLINGS {
            writeln!(body, ".subckt sibling{scope} a b\n.model local r\n.ends").unwrap();
        }
        for scope in 0..MAX_SCOPE_DEPTH {
            writeln!(body, ".subckt nested{scope} a b\n.model inside{scope} r").unwrap();
        }
        // This deliberately exceeds the parser's semantic depth limit. The
        // index must scan across it, but never allocate a 65th scope record.
        body.push_str(".subckt excessive a b\n.model ignored r\n.ends\n");
        for _ in 0..MAX_SCOPE_DEPTH {
            body.push_str(".ends\n");
        }
        body.push_str(".model final r\n.end\n.model after r\n");
        let cards = cards(&body);
        let index = ModelIndex::build(&cards);
        assert_eq!(index.scopes.len(), 1 + SIBLINGS + MAX_SCOPE_DEPTH);
        assert_eq!(
            index.scopes.iter().map(|s| s.names.len()).sum::<usize>(),
            ROOTS + SIBLINGS + MAX_SCOPE_DEPTH + 1
        );
        assert_eq!(index.scopes[0].names.len(), ROOTS + 1);
        for scope in &index.scopes[1..=SIBLINGS] {
            assert_eq!(scope.parent, Some(0));
            assert_eq!(scope.names.len(), 1);
        }
        for (depth, scope) in index.scopes[SIBLINGS + 1..].iter().enumerate() {
            assert_eq!(
                scope.parent,
                Some(if depth == 0 { 0 } else { SIBLINGS + depth })
            );
        }
        let (visits, copied) = compare_index(
            &cards,
            &index,
            &[
                "root0", "root255", "local", "inside0", "inside63", "final", "after", "ignored",
            ],
        );
        assert_eq!(index.inspected_cards, cards.len() - 1);
        assert!(visits >= 3 * index.inspected_cards);
        assert!(copied >= ROOTS * (SIBLINGS + MAX_SCOPE_DEPTH));
    }
}
