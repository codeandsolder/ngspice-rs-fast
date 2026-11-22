//! Incremental semantic deck parser built with winnow token-stream combinators.
//!
//! Dispatch follows `src/spicelib/parser/inppas2.c`, `INPpas2()`;
//! device grammars follow `inp2r.c`, `inp2c.c`, `inp2l.c`, `inp2v.c`, and
//! `inp2i.c`, plus `inp2d.c`, `inp2q.c` and `inp2m.c` for bounded D/Q/M forms,
//! `inp2e.c`..`inp2h.c` for linear controlled sources, `inp2k.c` for mutual
//! inductance, `inp2s.c`/`inp2w.c` for switches, `inp2u.c` for URC lines and `inp2o.c` for
//! lossy lines.
//! Scalar model cards follow
//! `inpdomod.c`/`inpgmod.c`. Dot-card dispatch follows `inp2dot.c`, not the front-end
//! `parse-bison.y` expression grammar.
//!
//! The implemented subset is M1a (scalar R/C/L, DC/AC sources, analysis cards)
//! plus M1b model cards, two-terminal D, three/four-terminal Q and
//! four-terminal M instances, bounded flags/IC vectors and numeric PULSE/PWL/SIN/EXP/SFFM/AM. Q/M use declared names for terminal disambiguation;
//! scoped subcircuits/X and source-relative include/library resolution. Model
//! types/backend availability and parameter validity are not checked yet.
//! Other constructs fail explicitly, never silently dropping cards. Values stay
//! textual; evaluation and circuit elaboration are separate passes. See `docs/port/ROADMAP.md` for the remaining M1 work.

use std::path::Path;

use crate::primitives::SpiceResult;

use crate::netlist::ast::{FourierCard, MeasureCard, Netlist, OutputCards};
use crate::netlist::card::{DotCommand, RawCard};
use crate::netlist::source::{Deck, fold_bounded, load_bounded};

use crate::netlist::source::budget::SourceBudget;
pub use crate::netlist::source::budget::SourceLimits;

mod behavioural;
mod bexpression;
mod controlled;
mod diode;
mod expression;
mod flags;
mod fourier;
mod func;
mod grammar;
mod hints;
mod ic;
mod jfet;
mod linear;
mod ltra;
mod measure;
mod mesfet;
mod model;
mod mutual;
mod options;
mod param;
mod resolution;
mod save;
mod scopes;
mod structure;
mod switch;
mod syntax;
mod tline;
mod transistor;
mod txl;
mod urc;
mod vector;
mod waveform;

/// Turns decks into [`Netlist`]s for the currently supported syntax subset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Parser {
    auto_gnd: bool,
}

/// A parsed deck: the semantic netlist, the `.save`/`.print` cards, the
/// `.measure` cards and the `.four` cards.
///
/// The output and post-processing cards are returned beside the netlist rather
/// than inside it. They describe what an analysis should write and what should
/// be measured or transformed afterwards, not how the circuit is built, so they
/// never reach device elaboration; separating them also keeps the netlist
/// unchanged for every consumer that only simulates the circuit. See
/// [`Parser::parse_file_with_output`], `docs/port/OUTPUT_SELECTION.md`,
/// `docs/port/MEASURE.md` and `docs/port/FOURIER.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedDeck {
    /// The semantic netlist.
    pub netlist: Netlist,
    /// The `.save`/`.print` cards the deck contains, typed and positioned.
    pub output: OutputCards,
    /// The `.measure`/`.meas` cards the deck contains, in deck order, typed and
    /// positioned. Measurements are evaluated over the **full** plot, so the
    /// output selection never hides a measurable vector.
    pub measurements: Vec<MeasureCard>,
    /// The `.four` cards the deck contains, in deck order, typed and
    /// positioned. A `.four` card is evaluated over the **full** plot as well,
    /// so the output selection never hides a transformed vector and the written
    /// rawfile never changes because a `.four` exists.
    pub fourier: Vec<FourierCard>,
}

impl Parser {
    /// A parser with ngspice's default behaviour, where `gnd` means ground.
    #[must_use]
    pub const fn new() -> Self {
        Self { auto_gnd: true }
    }

    /// A parser with the aliasing rule set explicitly.
    ///
    /// `auto_gnd == false` matches ngspice with `no_auto_gnd` set, where `gnd`
    /// is an ordinary node.
    #[must_use]
    pub const fn with_auto_gnd(auto_gnd: bool) -> Self {
        Self { auto_gnd }
    }

    /// Whether `gnd` is aliased to node `0`.
    #[must_use]
    pub const fn auto_gnd(self) -> bool {
        self.auto_gnd
    }

    /// Builds a semantic netlist, stopping at `.end` as `INP2dot()` does.
    ///
    /// Analysis arguments are preserved without validation, per the AST
    /// contract; parsing a request does not imply its driver is implemented.
    /// Deck fragments without `.end` are accepted, as by `INPpas2()`.
    /// Includes are retained without I/O; use [`Parser::parse_file`] for source
    /// resolution. Nested subcircuits own ordered cards and local declarations.
    /// Q/M require a model declaration in their scope or an ancestor before
    /// `.end`; forward declarations work. A read-only name index disambiguates optional ports,
    /// without validating model type/backend or applying selector/default rules.
    /// D references may remain unresolved. None of these are simulation inputs
    /// until the later elaboration pass validates them.
    ///
    /// `.save`/`.print` and `.measure` cards are validated here (a malformed
    /// request is an error, never a dropped card) and returned by
    /// [`Parser::parse_deck_with_output`].
    ///
    /// # Errors
    ///
    /// Returns [`crate::primitives::SpiceError::Parse`] for malformed supported syntax and
    /// [`crate::primitives::SpiceError::NotYetPorted`] for constructs outside the current subset.
    /// No partially parsed netlist is returned on failure.
    pub fn parse_deck(self, deck: &Deck) -> SpiceResult<Netlist> {
        Ok(self.parse_deck_with_output(deck)?.netlist)
    }

    /// Builds a semantic netlist and reports the deck's `.save`/`.print` cards
    /// and `.measure` cards.
    ///
    /// # Errors
    ///
    /// The same failures as [`Parser::parse_deck`].
    pub fn parse_deck_with_output(self, deck: &Deck) -> SpiceResult<ParsedDeck> {
        self.parse_deck_with_limits_and_output(deck, SourceLimits::default())
    }

    /// Parse a caller-owned folded deck with explicit admission and lexical limits.
    ///
    /// # Errors
    /// Invalid source structure, unsupported syntax or exhausted resource limits.
    pub fn parse_deck_with_limits_and_output(
        self,
        deck: &Deck,
        limits: SourceLimits,
    ) -> SpiceResult<ParsedDeck> {
        let mut budget = SourceBudget::new(limits);
        budget.admit_deck(deck)?;
        self.parse_admitted(deck, &mut budget)
    }

    /// Parse borrowed, potentially large text without first constructing an
    /// unbounded owned deck. Includes remain unresolved as in [`Self::parse_deck`].
    ///
    /// # Errors
    /// Invalid source structure, unsupported syntax or exhausted resource limits.
    pub fn parse_text(
        self,
        file: &Path,
        text: &str,
        limits: SourceLimits,
    ) -> SpiceResult<ParsedDeck> {
        let location = crate::primitives::SourceLoc::new(file.to_path_buf(), 1, 1);
        let mut budget = SourceBudget::new(limits);
        budget.validate(&location)?;
        budget.begin_file(&location)?;
        budget.bytes(text.len(), &location)?;
        let deck = fold_bounded(file, text, true, &mut budget)?;
        self.parse_admitted(&deck, &mut budget)
    }

    fn parse_admitted(self, deck: &Deck, budget: &mut SourceBudget) -> SpiceResult<ParsedDeck> {
        scopes::assemble(deck, &prepare_cards(deck, budget)?, self.auto_gnd)
    }

    /// Parses `text` (whose first byte is at `location`) as one complete
    /// behavioural-source expression ([`crate::netlist::bexpr`]). `verbatim` selects
    /// the `=pwl(` line rules (numparam `{...}` values, no literal rounding).
    ///
    /// # Errors
    ///
    /// [`crate::primitives::SpiceError::Parse`] for malformed syntax or trailing text.
    pub fn parse_behavioural_expression(
        self,
        text: &str,
        location: &crate::primitives::SourceLoc,
        verbatim: bool,
    ) -> SpiceResult<crate::netlist::bexpr::BehaviouralExpression> {
        bexpression::parse_complete(
            text,
            super::source::SourcePosition::physical(location),
            verbatim,
            self.auto_gnd,
        )
    }

    /// Loads `path`, resolves source-relative `.include`/`.lib` directives and
    /// parses ordered scoped cards with default [`SourceLimits`]. Included files
    /// are fragments (no title); the root's first physical line is its title.
    ///
    /// # Errors
    ///
    /// Fails if the file cannot be read, or for the syntax errors and unported
    /// constructs described by [`Parser::parse_deck`].
    pub fn parse_file(self, path: impl AsRef<Path>) -> SpiceResult<Netlist> {
        Ok(self.parse_file_with_output(path)?.netlist)
    }

    /// Loads `path` and reports the deck's `.save`/`.print` and `.measure`
    /// cards as well.
    ///
    /// # Errors
    ///
    /// The same failures as [`Parser::parse_file`].
    pub fn parse_file_with_output(self, path: impl AsRef<Path>) -> SpiceResult<ParsedDeck> {
        self.parse_file_with_limits_and_output(path, SourceLimits::default())
    }

    /// Resolves and parses a file with explicit source work limits.
    ///
    /// Paths are relative to the file containing each directive, not the process
    /// working directory. Canonical file/section dependency cycles are rejected.
    /// No home/environment/search-path substitution is performed.
    ///
    /// # Errors
    /// I/O, malformed structure, unavailable syntax, cycles or exhausted limits.
    pub fn parse_file_with_limits(
        self,
        path: impl AsRef<Path>,
        limits: SourceLimits,
    ) -> SpiceResult<Netlist> {
        Ok(self
            .parse_file_with_limits_and_output(path, limits)?
            .netlist)
    }

    /// Resolves and parses a file with explicit limits and reports its output
    /// cards as well.
    ///
    /// # Errors
    ///
    /// The same failures as [`Parser::parse_file_with_limits`].
    pub fn parse_file_with_limits_and_output(
        self,
        path: impl AsRef<Path>,
        limits: SourceLimits,
    ) -> SpiceResult<ParsedDeck> {
        let (deck, cards) = resolution::resolve(path.as_ref(), limits)?;
        scopes::assemble(&deck, &cards, self.auto_gnd)
    }
}

/// Parses one unevaluated parameter expression.
///
/// For instance, parse the text of a
/// `{...}` value. `location` is the position of the first byte of `text`
/// (its column anchors every span and diagnostic). Whitespace is allowed
/// between tokens, as inside braces. See [`crate::netlist::expr`] for the grammar.
///
/// # Errors
///
/// [`crate::primitives::SpiceError::Parse`] for malformed syntax and
/// [`crate::primitives::SpiceError::NotYetPorted`] for valid numparam outside the
/// bounded subset (other operators, functions or quoting).
pub fn parse_parameter_expression(
    text: &str,
    location: &crate::primitives::SourceLoc,
) -> SpiceResult<crate::netlist::expr::ParameterExpression> {
    expression::parse_expression(
        text,
        super::source::SourcePosition::physical(location),
        false,
    )
}

/// Cache lexical results for ordered replay; later errors do not mask earlier
/// semantic failures. Forward declaration lookup belongs to each scope.
fn prepare_cards(
    deck: &Deck,
    budget: &mut SourceBudget,
) -> SpiceResult<Vec<SpiceResult<scopes::InputCard>>> {
    let mut cards = Vec::new();
    for line in &deck.lines {
        budget.allocation(cards.try_reserve(1), &line.location)?;
        if let Err(error) = budget.card(&line.location) {
            cards.push(Err(error));
            break;
        }
        let card = RawCard::parse_bounded(line, budget);
        let end = card
            .as_ref()
            .is_ok_and(|c| c.dot_command() == Some(&DotCommand::End));
        cards.push(card.map(Into::into));
        if end || budget.exhausted() {
            break;
        }
    }
    Ok(cards)
}

impl Default for Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// Classifies every card in a deck.
///
/// # Errors
///
/// Propagates tokenizer errors.
pub fn classify_deck(deck: &Deck) -> SpiceResult<Vec<RawCard>> {
    classify_deck_with_limits(deck, SourceLimits::default())
}

/// Classify a caller-owned deck under explicit source and lexical limits.
/// Unlike semantic parsing this visits every card, including after `.end`.
///
/// # Errors
/// Invalid lexical syntax or exhausted source/work limits.
pub fn classify_deck_with_limits(deck: &Deck, limits: SourceLimits) -> SpiceResult<Vec<RawCard>> {
    let mut budget = SourceBudget::new(limits);
    budget.admit_deck(deck)?;
    classify_admitted(deck, &mut budget)
}

fn classify_admitted(deck: &Deck, budget: &mut SourceBudget) -> SpiceResult<Vec<RawCard>> {
    let mut out = Vec::new();
    for line in &deck.lines {
        budget.card(&line.location)?;
        budget.allocation(out.try_reserve(1), &line.location)?;
        out.push(RawCard::parse_bounded(line, budget)?);
    }
    Ok(out)
}

/// Loads a deck from disk and classifies it.
///
/// # Errors
///
/// Fails if the file cannot be read or a card cannot be tokenized.
pub fn load_classified(path: impl AsRef<Path>) -> SpiceResult<(Deck, Vec<RawCard>)> {
    load_classified_with_limits(path, SourceLimits::default())
}

/// Load and classify a file under one shared source and lexical budget.
/// This does not expand included sources.
///
/// # Errors
/// Invalid UTF-8, I/O, lexical syntax or exhausted source/work limits.
pub fn load_classified_with_limits(
    path: impl AsRef<Path>,
    limits: SourceLimits,
) -> SpiceResult<(Deck, Vec<RawCard>)> {
    let mut budget = SourceBudget::new(limits);
    let deck = load_bounded(path.as_ref(), &mut budget)?;
    let cards = classify_admitted(&deck, &mut budget)?;
    Ok((deck, cards))
}

/// Evaluates a measurement's numeric setters, retaining positioned diagnostics.
/// Reuses the original winnow card grammar to check ranges and crossing counts.
///
/// # Errors
/// Expression evaluation or measurement grammar failures.
pub(crate) fn resolve_measure_card(
    card: &MeasureCard,
    scope: &crate::netlist::eval::SequentialBindings<'_>,
    budget: &mut crate::netlist::eval::EvalBudget,
) -> SpiceResult<MeasureCard> {
    let crate::netlist::ast::MeasureRequest::Deferred {
        card: source,
        auto_gnd,
        ..
    } = &card.request
    else {
        return Ok(card.clone());
    };
    let mut prepared = (**source).clone();
    for token in &mut prepared.tokens {
        if expression::is_expression_token(token) {
            let value = scope.evaluate(&measure::numeric_expression(token)?, budget)?;
            token.kind = crate::netlist::token::TokenKind::Number(value);
            token.text = crate::netlist::elaborate::format_literal(value);
        }
    }
    match grammar::parse_card(&prepared, *auto_gnd, &std::collections::BTreeSet::new())? {
        grammar::ParsedCard::Measure(card) => Ok(card),
        _ => Err(crate::primitives::SpiceError::parse(
            card.location.clone(),
            "expected a measurement card",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{Parser, classify_deck};
    use crate::netlist::source::parse_deck_text;
    use std::path::Path;

    const DECK: &str = "\
RC divider
v1 in 0 dc 5
r1 in out 1k
r2 out 0 1k
.model dm d(is=1e-14)
.subckt pair a b
r1 a b 10k
.ends pair
.tran 1u 10u
.end
";

    #[test]
    fn classification_still_works_on_unported_syntax() {
        let deck = parse_deck_text(Path::new("rc.cir"), DECK);
        let cards = classify_deck(&deck).expect("classifies");
        assert_eq!(cards.len(), 9);
        assert_eq!(cards.iter().filter(|c| c.kind.is_device()).count(), 4);
        assert_eq!(cards.iter().filter(|c| c.kind.is_dot_command()).count(), 5);
    }

    #[test]
    fn parsing_retains_scoped_devices() {
        let deck = parse_deck_text(Path::new("rc.cir"), DECK);
        let netlist = Parser::new().parse_deck(&deck).unwrap();
        assert_eq!(netlist.devices.len(), 3);
        assert_eq!(netlist.subcircuits[0].devices.len(), 1);
        assert_eq!(netlist.subcircuits[0].location.line, 6);
    }

    #[test]
    fn parser_remembers_the_gnd_rule() {
        assert!(Parser::new().auto_gnd());
        assert!(!Parser::with_auto_gnd(false).auto_gnd());
        assert_eq!(Parser::default(), Parser::new());
    }
}
