//! One admission and lexical-work budget for all semantic deck entrypoints.

use crate::netlist::source::Deck;
use crate::primitives::{SourceLoc, SpiceError, SpiceResult};

/// Limits for source admission and ordered lexical work.
///
/// The byte/file/card visit limits charge repeated includes in full. Folding
/// limits charge unique physical interpretations (root title and fragment
/// views are separate); token work charges both initial lexing and replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLimits {
    /// Maximum include depth below the root (at most 64).
    pub depth: usize,
    /// Total logical file visits, including the root.
    pub files: usize,
    /// Total admitted UTF-8 source bytes, including unselected sections.
    pub bytes: usize,
    /// Total processed card occurrences, including directives and terminators.
    pub cards: usize,
    /// Aggregate physical lines folded, including blank/comment lines.
    pub physical_lines: usize,
    /// Aggregate logical cards constructed, including unselected sections.
    pub folded_cards: usize,
    /// Maximum byte length of one folded logical card (or title).
    pub card_bytes: usize,
    /// Aggregate tokens constructed or replayed from the lexical cache.
    pub tokens: usize,
}

impl Default for SourceLimits {
    fn default() -> Self {
        Self {
            depth: 64,
            files: 1024,
            bytes: 16 * 1024 * 1024,
            cards: 1_000_000,
            physical_lines: 2_000_000,
            folded_cards: 1_000_000,
            card_bytes: 4 * 1024 * 1024,
            tokens: 4_000_000,
        }
    }
}

/// Counters spend before storage creation. A hard admission/lexical error
/// terminates traversal; ordinary syntax errors can still be replayed in
/// order. Actual allocation failures use the same terminal path.
pub(in crate::netlist) struct SourceBudget {
    pub(crate) limits: SourceLimits,
    files: usize,
    bytes: usize,
    cards: usize,
    lines: usize,
    folded: usize,
    tokens: usize,
    exhausted: bool,
}

fn spend(
    remaining: &mut usize,
    count: usize,
    location: &SourceLoc,
    label: &str,
) -> SpiceResult<()> {
    *remaining = remaining.checked_sub(count).ok_or_else(|| {
        SpiceError::parse(location.clone(), format!("source {label} limit exceeded"))
    })?;
    Ok(())
}

impl SourceBudget {
    pub(crate) const fn new(limits: SourceLimits) -> Self {
        Self {
            limits,
            files: limits.files,
            bytes: limits.bytes,
            cards: limits.cards,
            lines: limits.physical_lines,
            folded: limits.folded_cards,
            tokens: limits.tokens,
            exhausted: false,
        }
    }

    pub(crate) fn validate(&self, location: &SourceLoc) -> SpiceResult<()> {
        if self.limits.depth > 64 {
            return Err(SpiceError::parse(
                location.clone(),
                "source depth limit must be at most 64",
            ));
        }
        Ok(())
    }

    pub(crate) fn begin_file(&mut self, location: &SourceLoc) -> SpiceResult<usize> {
        spend(&mut self.files, 1, location, "file work")?;
        Ok(self.bytes)
    }

    pub(crate) fn bytes(&mut self, count: usize, location: &SourceLoc) -> SpiceResult<()> {
        spend(&mut self.bytes, count, location, "byte work")
    }

    pub(crate) fn physical(&mut self, count: usize, location: &SourceLoc) -> SpiceResult<()> {
        spend(&mut self.lines, count, location, "physical line work")
    }

    pub(crate) fn folded(&mut self, location: &SourceLoc) -> SpiceResult<()> {
        spend(&mut self.folded, 1, location, "folded card work")
    }

    pub(crate) fn card_size(&self, count: usize, location: &SourceLoc) -> SpiceResult<()> {
        if count > self.limits.card_bytes {
            return Err(SpiceError::parse(
                location.clone(),
                "source logical card byte limit exceeded",
            ));
        }
        Ok(())
    }

    pub(crate) fn card(&mut self, location: &SourceLoc) -> SpiceResult<()> {
        self.cards = self.cards.checked_sub(1).ok_or_else(|| {
            SpiceError::parse(
                location.clone(),
                format!(
                    "source card work limit exceeded ({} cards, counting repeated includes)",
                    self.limits.cards
                ),
            )
        })?;
        Ok(())
    }

    pub(crate) fn tokens(&mut self, count: usize, location: &SourceLoc) -> SpiceResult<()> {
        let result = spend(&mut self.tokens, count, location, "token work");
        self.exhausted |= result.is_err();
        result
    }

    pub(crate) const fn remaining_tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) const fn exhausted(&self) -> bool {
        self.exhausted
    }

    pub(crate) fn allocation(
        &mut self,
        result: Result<(), std::collections::TryReserveError>,
        location: &SourceLoc,
    ) -> SpiceResult<()> {
        result.map_err(|error| {
            self.exhausted = true;
            SpiceError::parse(
                location.clone(),
                format!("allocating source storage: {error}"),
            )
        })
    }

    /// An externally constructed Deck already owns its folded input. Charge
    /// its accessible metadata and text, not unavailable discarded comments,
    /// whitespace or empty physical lines. No parser allocation precedes it.
    pub(crate) fn admit_deck(&mut self, deck: &Deck) -> SpiceResult<()> {
        self.validate(&deck.title_location)?;
        self.begin_file(&deck.title_location)?;
        self.bytes(deck.title.len(), &deck.title_location)?;
        self.physical(1, &deck.title_location)?;
        self.card_size(deck.title.len(), &deck.title_location)?;
        for line in &deck.lines {
            self.bytes(line.text.len(), &line.location)?;
            self.physical(line.retained_physical_lines(), &line.location)?;
            self.folded(&line.location)?;
            self.card_size(line.text.len(), &line.location)?;
        }
        Ok(())
    }
}

pub(in crate::netlist) fn copy_text(
    text: &str,
    budget: &mut SourceBudget,
    location: &SourceLoc,
) -> SpiceResult<String> {
    let mut owned = String::new();
    budget.allocation(owned.try_reserve_exact(text.len()), location)?;
    owned.push_str(text);
    Ok(owned)
}
