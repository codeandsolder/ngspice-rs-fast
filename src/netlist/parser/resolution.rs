//! Source-relative preprocessing, referencing inpcom.c's `inp_readall` and
//! library section processing. Deliberately no sourcepath/env/home expansion.

use std::cell::OnceCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use crate::primitives::{SourceLoc, SpiceError, SpiceResult};
use petgraph::algo::has_path_connecting;
use petgraph::graph::{DiGraph, NodeIndex};

use super::grammar::{self, ParsedCard};
use super::scopes::InputCard;
use crate::netlist::ast::LibrarySection;
use crate::netlist::card::{DotCommand, RawCard};
use crate::netlist::source::{Deck, parse_deck_text, parse_fragment_text};

/// Per-parse source work limits. Repeated includes consume budgets again;
/// sharing a dependency never suppresses its ordered content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceLimits {
    /// Maximum nested includes below the root. Values above 64 are rejected.
    pub depth: usize,
    /// Total logical file visits including the root and repeated includes.
    pub files: usize,
    /// Total logical UTF-8 source bytes, including unselected library sections.
    pub bytes: usize,
    /// Total processed cards, including directives and structural terminators.
    ///
    /// C ngspice (`inpcom.c`, `inp_readall`) has no such cap; the port keeps one
    /// only as a budget against runaway repeated includes. The default is four
    /// times the default subcircuit device budget (`SubcircuitLimits::max_devices`
    /// in the devices layer, 250,000), so a flat deck that the expansion budget
    /// accepts is not rejected here for its size.
    pub cards: usize,
}

impl Default for SourceLimits {
    fn default() -> Self {
        Self {
            depth: 64,
            files: 1024,
            bytes: 16 * 1024 * 1024,
            cards: 1_000_000,
        }
    }
}

type SourceKey = (PathBuf, Option<String>);
type CardResult = SpiceResult<Arc<RawCard>>;
type CachedCard = OnceCell<Result<Arc<RawCard>, Box<SpiceError>>>;
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
    fn new(deck: Deck) -> Self {
        let cards = (0..deck.lines.len()).map(|_| OnceCell::new()).collect();
        Self {
            deck,
            cards,
            sections: OnceCell::new(),
        }
    }

    fn card(&self, index: usize) -> CardResult {
        self.cards[index]
            .get_or_init(|| {
                RawCard::parse(&self.deck.lines[index])
                    .map(Arc::new)
                    .map_err(Box::new)
            })
            .as_ref()
            .map(Arc::clone)
            .map_err(|error| error.as_ref().clone())
    }

    fn section(&self, name: &str, location: &SourceLoc) -> SpiceResult<Section> {
        self.sections
            .get_or_init(|| index_sections(self))
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
    fn fragment(&mut self, path: &Path) -> Rc<Fragment> {
        match &self.content {
            SnapshotContent::Folded(fragment) => Rc::clone(fragment),
            SnapshotContent::RootText(text) => {
                let fragment = Rc::new(Fragment::new(parse_fragment_text(path, text)));
                self.content = SnapshotContent::Folded(Rc::clone(&fragment));
                fragment
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

struct Resolver {
    limits: SourceLimits,
    files: usize,
    bytes: usize,
    cards: usize,
    sources: BTreeMap<PathBuf, SourceSnapshot>,
    dependencies: DiGraph<SourceKey, ()>,
    nodes: BTreeMap<SourceKey, NodeIndex>,
}

pub(super) fn resolve(
    path: &Path,
    limits: SourceLimits,
) -> SpiceResult<(Deck, Vec<SpiceResult<InputCard>>)> {
    let location = SourceLoc::new(path.to_path_buf(), 1, 1);
    if limits.depth > 64 {
        return Err(SpiceError::parse(
            location,
            "source depth limit must be at most 64",
        ));
    }
    let mut resolver = Resolver::new(limits);
    let canonical = fs::canonicalize(path).map_err(|e| SpiceError::io(path, &e))?;
    let remaining = resolver.begin_file(&location)?;
    let text = read_text(&canonical, remaining, &location)?;
    let bytes = text.len();
    resolver.bytes += bytes;
    // Keep the caller's root path/title API; fragments use canonical paths.
    let fragment = Fragment::new(parse_deck_text(path, &text));
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
        output.push(Err(error));
    }
    Ok((fragment.deck, output))
}

fn byte_limit(location: &SourceLoc) -> SpiceError {
    SpiceError::parse(location.clone(), "source byte work limit exceeded")
}

fn read_text(path: &Path, remaining: usize, location: &SourceLoc) -> SpiceResult<String> {
    let mut bytes = Vec::new();
    let file = File::open(path).map_err(|e| SpiceError::io(path, &e))?;
    file.take(
        u64::try_from(remaining)
            .unwrap_or(u64::MAX)
            .saturating_add(1),
    )
    .read_to_end(&mut bytes)
    .map_err(|e| SpiceError::io(path, &e))?;
    if bytes.len() > remaining {
        return Err(byte_limit(location));
    }
    String::from_utf8(bytes)
        .map_err(|e| SpiceError::parse(location.clone(), format!("source is not UTF-8: {e}")))
}

impl Resolver {
    fn new(limits: SourceLimits) -> Self {
        Self {
            limits,
            files: 0,
            bytes: 0,
            cards: 0,
            sources: BTreeMap::new(),
            dependencies: DiGraph::new(),
            nodes: BTreeMap::new(),
        }
    }

    fn begin_file(&mut self, location: &SourceLoc) -> SpiceResult<usize> {
        if self.files >= self.limits.files {
            return Err(SpiceError::parse(
                location.clone(),
                "source file work limit exceeded",
            ));
        }
        self.files += 1;
        Ok(self.limits.bytes.saturating_sub(self.bytes))
    }

    /// Each visit consumes the original full file length, even a cache hit
    /// selecting an empty library section. Only physical I/O is deduplicated.
    fn fragment(&mut self, path: &Path, location: &SourceLoc) -> SpiceResult<Rc<Fragment>> {
        let remaining = self.begin_file(location)?;
        if let Some(source) = self.sources.get_mut(path) {
            if source.bytes > remaining {
                return Err(byte_limit(location));
            }
            self.bytes += source.bytes;
            return Ok(source.fragment(path));
        }
        let text = read_text(path, remaining, location)?;
        let bytes = text.len();
        self.bytes += bytes;
        let fragment = Rc::new(Fragment::new(parse_fragment_text(path, &text)));
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
        let parsed = grammar::parse_card(&entry.source, true, &BTreeSet::new())?;
        let ParsedCard::Include(directive) = parsed else {
            return Ok(None);
        };
        if chain.len() >= self.limits.depth {
            return Err(SpiceError::parse(
                directive.location,
                "source include depth limit exceeded",
            ));
        }
        let target = directive
            .location
            .path()
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
                let selected = fragment.section(&section, &directive.location)?;
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
            if self.cards >= self.limits.cards {
                return Err(SpiceError::parse(
                    line.location.clone(),
                    format!(
                        "source card work limit exceeded ({} cards, counting repeated includes)",
                        self.limits.cards
                    ),
                ));
            }
            self.cards += 1;
            let card = match fragment.card(index) {
                Ok(card) => card,
                Err(error) => {
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
fn index_sections(fragment: &Fragment) -> SpiceResult<SectionIndex> {
    let mut current: Option<(String, Arc<RawCard>, usize)> = None;
    let mut sections = BTreeMap::new();
    for (index, line) in fragment.deck.lines.iter().enumerate() {
        let keyword = line.text.split_whitespace().next().unwrap_or_default();
        if keyword.eq_ignore_ascii_case(".lib") || keyword.eq_ignore_ascii_case(".endl") {
            let card = fragment.card(index)?;
            match grammar::parse_card(&card, true, &BTreeSet::new())? {
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
        let mut resolver = Resolver::new(SourceLimits {
            files: 3,
            bytes: 2 * original.len(),
            ..SourceLimits::default()
        });
        let location = SourceLoc::new(PathBuf::from("root.cir"), 2, 1);
        let first = resolver.fragment(&path, &location).unwrap();
        let selected = first.section("tt", &location).unwrap();
        let card = first.card(selected.range.start).unwrap();
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
        let second_section = second.section("tt", &location).unwrap();
        assert!(Arc::ptr_eq(
            &selected.boundaries.opening,
            &second_section.boundaries.opening
        ));
        assert!(Arc::ptr_eq(
            &card,
            &second.card(second_section.range.start).unwrap()
        ));
        assert_eq!(resolver.sources.len(), 1);
        assert_eq!(resolver.files, 2);
        assert_eq!(resolver.bytes, 2 * original.len());
        assert!(
            resolver
                .fragment(&path, &location)
                .err()
                .unwrap()
                .to_string()
                .contains("byte work limit")
        );
        assert_eq!(resolver.files, 3);
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
