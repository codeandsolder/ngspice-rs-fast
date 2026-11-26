//! Source-relative preprocessing, referencing inpcom.c's `inp_readall` and
//! library section processing. Deliberately no sourcepath/env/home expansion.

use std::cell::OnceCell;
use std::collections::BTreeMap;
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use crate::primitives::{SourceLoc, SpiceError, SpiceResult};
use petgraph::algo::has_path_connecting;
use petgraph::graph::{DiGraph, NodeIndex};

use super::grammar::{self, ParsedCard};
use super::scopes::{InputCard, ModelVisibility};
use crate::netlist::ast::LibrarySection;
use crate::netlist::card::{DotCommand, RawCard};
use crate::netlist::source::admission::SourceAdmission;
use crate::netlist::source::{Deck, fold_bounded, read_admitted};

use crate::netlist::source::budget::{SourceBudget, SourceLimits};

type SourceKey = (PathBuf, Option<String>);
type CardResult = SpiceResult<Arc<RawCard>>;
type CachedCard = OnceCell<CachedLexical>;
struct CachedLexical {
    card: Result<Arc<RawCard>, Box<SpiceError>>,
    cost: usize,
}
type SectionIndex = BTreeMap<String, Section>;

#[derive(Clone)]
struct Section {
    range: Range<usize>,
    boundaries: LibrarySection,
}

/// A snapshot is folded once for each interpretation (root title or fragment).
/// Lexical work is lazy: unselected ordinary cards and text after .end never
/// need tokenization. Errors are replayed only when their occurrence is visited.
struct Fragment {
    deck: Deck,
    cards: Vec<CachedCard>,
    sections: OnceCell<SpiceResult<SectionIndex>>,
}

impl Fragment {
    fn new(deck: Deck, budget: &mut SourceBudget) -> SpiceResult<Self> {
        let mut cards = Vec::new();
        budget.allocation(
            cards.try_reserve_exact(deck.lines.len()),
            &deck.title_location,
        )?;
        cards.resize_with(deck.lines.len(), OnceCell::new);
        Ok(Self {
            deck,
            cards,
            sections: OnceCell::new(),
        })
    }

    fn card(&self, index: usize, budget: &mut SourceBudget) -> CardResult {
        let line = &self.deck.lines[index];
        let slot = &self.cards[index];
        if let Some(cached) = slot.get() {
            budget.tokens(cached.cost, &line.location)?;
            return cached
                .card
                .as_ref()
                .map(Arc::clone)
                .map_err(|error| error.as_ref().clone());
        }
        let before = budget.remaining_tokens();
        let card = RawCard::parse_bounded(line, budget).map(Arc::new);
        if budget.exhausted() {
            return card;
        }
        let cached = slot.get_or_init(|| CachedLexical {
            cost: before - budget.remaining_tokens(),
            card: card.map_err(Box::new),
        });
        cached
            .card
            .as_ref()
            .map(Arc::clone)
            .map_err(|error| error.as_ref().clone())
    }

    fn section(
        &self,
        name: &str,
        location: &SourceLoc,
        budget: &mut SourceBudget,
    ) -> SpiceResult<Section> {
        self.sections
            .get_or_init(|| index_sections(self, budget))
            .as_ref()
            .map_err(Clone::clone)?
            .get(name)
            .cloned()
            .ok_or_else(|| {
                SpiceError::parse(
                    location.clone(),
                    format!("library section '{name}' not found"),
                )
            })
    }
}

enum SnapshotContent {
    // The root's title is part of its fragment view, if a section in the
    // root itself is selected. Preserve the same bytes rather than re-read it.
    RootText(String),
    Folded(Rc<Fragment>),
}

struct SourceSnapshot {
    bytes: usize,
    content: SnapshotContent,
}

impl SourceSnapshot {
    fn fragment(&mut self, path: &Path, budget: &mut SourceBudget) -> SpiceResult<Rc<Fragment>> {
        match &self.content {
            SnapshotContent::Folded(fragment) => Ok(Rc::clone(fragment)),
            SnapshotContent::RootText(text) => {
                let fragment = Rc::new(Fragment::new(
                    fold_bounded(path, text, false, budget)?,
                    budget,
                )?);
                self.content = SnapshotContent::Folded(Rc::clone(&fragment));
                Ok(fragment)
            }
        }
    }
}

struct Expansion {
    fragment: Rc<Fragment>,
    range: Range<usize>,
    node: NodeIndex,
    chain: Arc<[SourceLoc]>,
}

struct Resolver<'a> {
    admission: Option<&'a SourceAdmission>,
    budget: SourceBudget,
    sources: BTreeMap<PathBuf, SourceSnapshot>,
    dependencies: DiGraph<SourceKey, ()>,
    nodes: BTreeMap<SourceKey, NodeIndex>,
}

pub(super) fn resolve(
    path: &Path,
    limits: SourceLimits,
    admission: Option<&SourceAdmission>,
) -> SpiceResult<(Deck, Vec<SpiceResult<InputCard>>)> {
    let location = SourceLoc::new(path.to_path_buf(), 1, 1);
    let mut resolver = Resolver::new(limits, admission);
    resolver.budget.validate(&location)?;
    let canonical = fs::canonicalize(path).map_err(|e| SpiceError::io(path, &e))?;
    let remaining = resolver.budget.begin_file(&location)?;
    let (text, bytes) = read_admitted(&canonical, remaining, &location, admission)?;
    resolver.budget.bytes(bytes, &location)?;
    // Keep the caller's root path/title API; fragments use canonical paths.
    let deck = fold_bounded(path, &text, true, &mut resolver.budget)?;
    let fragment = Fragment::new(deck, &mut resolver.budget)?;
    resolver.sources.insert(
        canonical.clone(),
        SourceSnapshot {
            bytes,
            content: SnapshotContent::RootText(text),
        },
    );
    let root = resolver.node((canonical, None));
    let mut output = Vec::new();
    if let Err(error) = resolver.expand(
        &fragment,
        0..fragment.deck.lines.len(),
        root,
        &Arc::default(),
        &mut output,
    ) {
        // Replay resolution errors alongside cards so an earlier unsupported
        // directive/device keeps its diagnostic instead of a later I/O error.
        resolver
            .budget
            .allocation(output.try_reserve(1), &location)?;
        output.push(Err(error));
    }
    Ok((fragment.deck, output))
}

impl<'a> Resolver<'a> {
    fn new(limits: SourceLimits, admission: Option<&'a SourceAdmission>) -> Self {
        Self {
            admission,
            budget: SourceBudget::new(limits),
            sources: BTreeMap::new(),
            dependencies: DiGraph::new(),
            nodes: BTreeMap::new(),
        }
    }

    /// Each visit consumes the original full file length, even a cache hit
    /// selecting an empty library section. Only physical I/O is deduplicated.
    fn fragment(&mut self, path: &Path, location: &SourceLoc) -> SpiceResult<Rc<Fragment>> {
        if let Some(admission) = self.admission {
            admission.check_cancelled(location)?;
        }
        let remaining = self.budget.begin_file(location)?;
        if let Some(source) = self.sources.get_mut(path) {
            self.budget.bytes(source.bytes, location)?;
            return source.fragment(path, &mut self.budget);
        }
        let (text, bytes) = read_admitted(path, remaining, location, self.admission)?;
        self.budget.bytes(bytes, location)?;
        let deck = fold_bounded(path, &text, false, &mut self.budget)?;
        let fragment = Rc::new(Fragment::new(deck, &mut self.budget)?);
        self.sources.insert(
            path.to_path_buf(),
            SourceSnapshot {
                bytes,
                content: SnapshotContent::Folded(Rc::clone(&fragment)),
            },
        );
        Ok(fragment)
    }

    fn node(&mut self, key: SourceKey) -> NodeIndex {
        *self
            .nodes
            .entry(key.clone())
            .or_insert_with(|| self.dependencies.add_node(key))
    }

    fn include(
        &mut self,
        entry: &mut InputCard,
        parent: NodeIndex,
        chain: &[SourceLoc],
    ) -> SpiceResult<Option<Expansion>> {
        let parsed = grammar::parse_card(&entry.source, true, ModelVisibility::default())?;
        let ParsedCard::Include(directive) = parsed else {
            return Ok(None);
        };
        if chain.len() >= self.budget.limits.depth {
            return Err(SpiceError::parse(
                directive.location,
                "source include depth limit exceeded",
            ));
        }
        // An explicit transformation receives the canonical source path
        // and runs in its directory, so linked output must use that same
        // physical origin (including when the root was opened via a symlink).
        // Preserve the caller's root spelling in diagnostic locations.
        let origin = if self.admission.is_some_and(SourceAdmission::transforms) {
            &self
                .dependencies
                .node_weight(parent)
                .ok_or_else(|| {
                    SpiceError::parse(directive.location.clone(), "missing source origin")
                })?
                .0
        } else {
            directive.location.path()
        };
        let target = origin
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(&directive.path);
        let canonical = fs::canonicalize(&target).map_err(|e| {
            SpiceError::parse(
                directive.location.clone(),
                format!("cannot resolve source '{}': {e}", target.display()),
            )
        })?;
        let child = self.node((canonical.clone(), directive.section.clone()));
        // Keep graph identity and cycle-test ordering unchanged.
        if has_path_connecting(&self.dependencies, child, parent, None) {
            return Err(SpiceError::parse(
                directive.location,
                format!(
                    "source cycle through '{}'{}",
                    canonical.display(),
                    directive
                        .section
                        .as_ref()
                        .map_or(String::new(), |s| format!(" section '{s}'"))
                ),
            ));
        }
        self.dependencies.update_edge(parent, child, ());
        let fragment = self
            .fragment(&canonical, &directive.location)
            .map_err(|e| {
                SpiceError::parse(
                    directive.location.clone(),
                    format!("loading source '{}': {e}", canonical.display()),
                )
            })?;
        entry.resolved_path = Some(canonical);
        let range = match directive.section {
            Some(section) => {
                let selected = fragment.section(&section, &directive.location, &mut self.budget)?;
                entry.selected_section = Some(selected.boundaries);
                selected.range
            }
            None => 0..fragment.deck.lines.len(),
        };
        let chain = chain
            .iter()
            .cloned()
            .chain(std::iter::once(directive.location))
            .collect();
        Ok(Some(Expansion {
            fragment,
            range,
            node: child,
            chain,
        }))
    }

    fn expand(
        &mut self,
        fragment: &Fragment,
        range: Range<usize>,
        parent: NodeIndex,
        chain: &Arc<[SourceLoc]>,
        output: &mut Vec<SpiceResult<InputCard>>,
    ) -> SpiceResult<bool> {
        for index in range {
            let line = &fragment.deck.lines[index];
            self.budget.card(&line.location)?;
            // Reserve a result slot before attempting lexing. A terminal
            // error is replayed once, not cached or repeated for every card.
            self.budget
                .allocation(output.try_reserve(1), &line.location)?;
            let card = match fragment.card(index, &mut self.budget) {
                Ok(card) => card,
                Err(error) => {
                    if self.budget.exhausted() {
                        return Err(error);
                    }
                    output.push(Err(error));
                    continue;
                }
            };
            let end = card.dot_command() == Some(&DotCommand::End);
            let include = matches!(
                card.dot_command(),
                Some(DotCommand::Include | DotCommand::Lib)
            );
            let mut entry = InputCard {
                source: card,
                include_chain: Arc::clone(chain),
                resolved_path: None,
                selected_section: None,
            };
            let expansion = if include {
                self.include(&mut entry, parent, chain)?
            } else {
                None
            };
            output.push(Ok(entry));
            if let Some(expansion) = expansion {
                if self.expand(
                    &expansion.fragment,
                    expansion.range,
                    expansion.node,
                    &expansion.chain,
                    output,
                )? {
                    return Ok(true);
                }
            } else if end {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// Validate section boundaries even in unselected blocks; do not tokenize or
/// resolve ordinary cards in those blocks. Section names are case-insensitive.
fn index_sections(fragment: &Fragment, budget: &mut SourceBudget) -> SpiceResult<SectionIndex> {
    let mut current: Option<(String, Arc<RawCard>, usize)> = None;
    let mut sections = BTreeMap::new();
    for (index, line) in fragment.deck.lines.iter().enumerate() {
        let keyword = line.text.split_whitespace().next().unwrap_or_default();
        if keyword.eq_ignore_ascii_case(".lib") || keyword.eq_ignore_ascii_case(".endl") {
            let card = fragment.card(index, budget)?;
            match grammar::parse_card(&card, true, ModelVisibility::default())? {
                ParsedCard::LibStart(name) => {
                    if current.is_some() {
                        return Err(SpiceError::parse(
                            line.location.clone(),
                            "nested library sections are unsupported",
                        ));
                    }
                    if sections.contains_key(&name) {
                        return Err(SpiceError::parse(
                            line.location.clone(),
                            format!("duplicate library section '{name}'"),
                        ));
                    }
                    current = Some((name, card, index + 1));
                }
                ParsedCard::LibEnd(name) => {
                    let Some((expected, opening, start)) = current.take() else {
                        return Err(SpiceError::parse(line.location.clone(), "unmatched .endl"));
                    };
                    if name.as_deref().is_some_and(|n| n != expected) {
                        return Err(SpiceError::parse(
                            line.location.clone(),
                            format!("mismatched .endl: expected '{expected}'"),
                        ));
                    }
                    sections.insert(
                        expected.clone(),
                        Section {
                            range: start..index,
                            boundaries: LibrarySection {
                                name: expected,
                                opening,
                                closing: card,
                            },
                        },
                    );
                }
                ParsedCard::Include(_) => {}
                _ => {
                    return Err(SpiceError::parse(
                        line.location.clone(),
                        "expected a library directive",
                    ));
                }
            }
        }
    }
    if let Some((name, start, _)) = current {
        return Err(SpiceError::parse(
            start.location.clone(),
            format!("missing .endl for '{name}'"),
        ));
    }
    Ok(sections)
}

#[cfg(test)]
mod tests {
    use super::{Resolver, SourceLimits};
    use crate::primitives::SourceLoc;
    use std::fs;
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn snapshot_survives_changes_and_charges_original_bytes() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ngspice-source-snapshot-{}-{}.lib",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let original = ".lib tt\nr1 a 0 1k\n+ tc1=0.1\n.endl tt\n.lib ff\nmalformed {\n.endl ff\n";
        fs::write(&path, original).unwrap();
        let mut resolver = Resolver::new(
            SourceLimits {
                files: 3,
                bytes: 2 * original.len(),
                ..SourceLimits::default()
            },
            None,
        );
        let location = SourceLoc::new(PathBuf::from("root.cir"), 2, 1);
        let first = resolver.fragment(&path, &location).unwrap();
        let selected = first
            .section("tt", &location, &mut resolver.budget)
            .unwrap();
        let card = first
            .card(selected.range.start, &mut resolver.budget)
            .unwrap();
        assert_eq!(card.tokens.last().unwrap().location.line, 3);
        assert!(first.cards[3].get().is_some());
        assert!(
            first.cards[4].get().is_none(),
            "unselected malformed body is not tokenized"
        );
        fs::write(&path, "new bytes").unwrap();
        let second = resolver.fragment(&path, &location).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(Rc::ptr_eq(&first, &second));
        let second_section = second
            .section("tt", &location, &mut resolver.budget)
            .unwrap();
        assert!(Arc::ptr_eq(
            &selected.boundaries.opening,
            &second_section.boundaries.opening
        ));
        assert!(Arc::ptr_eq(
            &card,
            &second
                .card(second_section.range.start, &mut resolver.budget)
                .unwrap()
        ));
        assert_eq!(resolver.sources.len(), 1);
        assert!(
            resolver
                .fragment(&path, &location)
                .err()
                .unwrap()
                .to_string()
                .contains("byte work limit")
        );
        assert!(
            resolver
                .fragment(&path, &location)
                .err()
                .unwrap()
                .to_string()
                .contains("file work limit")
        );
    }
}
