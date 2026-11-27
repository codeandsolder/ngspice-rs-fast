//! Subcircuit instance expansion (GitHub #18).
//!
//! This is the device-side half of `X` elaboration: it turns the parsed
//! [`Netlist`] into a deck in which every `X` instance has been replaced by its
//! definition's body, with ports bound, node/device identities made unique and
//! every body `{...}` site literalized against that instance's parameter scope.
//! [`crate::devices::Circuit::from_netlist_with_context`] runs it before any device is
//! built, so an unsupported or inconsistent deck never leaves a partial circuit
//! behind.
//!
//! The C equivalent is `src/frontend/subckt.c` (`inp_subcktexpand()`,
//! `translate()`, `gettrans()`, `collect_global_nodes()`), with the parameter
//! environment from `src/frontend/numparam/spicenum.c`.
//!
//! # Naming (C: `translate_node_name`, `translate_inst_name`)
//!
//! Ground `0` and every `.global` node keep their top-level name. Every other
//! node inside an instance is renamed `<instance-path>.<node>`, where the path
//! is the dotted chain of instance names from the root (`xout`, then
//! `xout.x1`). Devices are renamed `<designator>.<instance-path>.<name>`
//! (`r.xout.r2`, `r.xout.x1.r1`); an `X` instance's own name drops the
//! duplicated designator. An F/H controlling-source reference is renamed the
//! same way (`vin` inside `x1` becomes `v.x1.vin`), so it always names a device
//! of the same instance, as in C. Two instances of the same definition therefore never
//! share a node, a device or a branch current, whatever the values of their
//! internal nodes.
//!
//! # Precedence (C: numparam)
//!
//! Highest first: an instance override (`x1 in out div rval=3k`), then a body
//! `.param`, then the formal default on the `.subckt` line. Duplicate setters
//! are applied in written order, so the last one wins. A formal default is
//! evaluated in the body's own scope, so it may reference a body `.param`, and
//! it sees the caller's scope through the parent link.
//!
//! # Not ported
//!
//! Nested `.subckt` definitions inside a body, unresolved source directives,
//! and analysis cards inside a body are rejected explicitly rather than ignored.
//! Resolved includes already contribute their materialized cards in source order.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::devices::binning::{DeclarationIndex, bin_base};
use crate::netlist::ast::{
    CodePortKind, DeviceInstance, FourierCard, MeasureCard, ModelCard, Netlist, NodeHintCard,
    OptionCard, OutputCards, ParamAssignment, ParamCard, ParameterAssignment, ParameterKind,
    PrintCard, RequestedVector, SaveCard, ScopeEntry, ScopeView, ScopedCardKind, Subcircuit,
};
use crate::netlist::bexpr::BExprKind;
use crate::netlist::eval::{EvalBudget, FunctionScope, ParamBinding, ParamScope};
use crate::netlist::expr::{Expr, ExprKind, ParameterExpression, SourceSpan};
use crate::primitives::{NodeTable, Real, SourceLoc, SpiceError, SpiceResult};
use petgraph::Direction;
use petgraph::algo::tarjan_scc;
use petgraph::graph::{DiGraph, NodeIndex};

/// The C reference for expansion used in diagnostics.
pub const C_REFERENCE: &str =
    "src/frontend/subckt.c (inp_subcktexpand, translate, collect_global_nodes)";

/// Bounds on one expansion.
///
/// Recursive definitions are rejected up front ([`expand_subcircuits`] builds
/// the definition graph and reports the cycle), so these bounds are the backstop
/// against a legal but exploding deck: a depth limit that a chain of distinct
/// definitions cannot exceed unnoticed, and a device budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubcircuitLimits {
    /// Maximum number of nested instance frames.
    pub max_depth: usize,
    /// Maximum number of devices the expansion may produce.
    pub max_devices: usize,
}

impl Default for SubcircuitLimits {
    fn default() -> Self {
        Self {
            max_depth: 32,
            max_devices: 250_000,
        }
    }
}

/// A deck with every `X` instance replaced by its definition's body.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpandedNetlist {
    /// Source-prepared, occurrence-bound passive initialization equations.
    pub prepared_initialization: crate::netlist::prepared_initialization::BoundInitializations,
    /// Top-level devices in deck order, `X` instances expanded in place.
    pub devices: Vec<DeviceInstance>,
    /// Explicit prepared forcing resolved in each flattened instance scope.
    pub prepared_waveforms: Vec<(String, super::prepared_waveform::PreparedForcing)>,
    /// Root `.model` cards in declaration order, then each instance's renamed
    /// local models in first-reference order.
    pub models: Vec<ModelCard>,
    pub(crate) statistical_model_origins: BTreeSet<String>,
    /// Instantiated options in expanded card order, with local expressions resolved.
    pub options: Vec<OptionCard>,
    /// Renamed, literalized initial conditions in expanded card order.
    pub initial_conditions: Vec<NodeHintCard>,
    /// Renamed, literalized operating-point hints in expanded card order.
    pub nodesets: Vec<NodeHintCard>,
    /// Expanded save/print requests.
    pub output: OutputCards,
    /// Instantiated measurements (C does not rename their operands).
    pub measurements: Vec<MeasureCard>,
    /// Fourier cards hoisted once per used definition, without operand renaming.
    pub fourier: Vec<FourierCard>,
}

/// A subcircuit actual connection may be used in either domain. Do not
/// canonicalize digital names as analog ground or .global nodes.
#[derive(Default)]
struct HierarchyBindings {
    analog: BTreeMap<String, String>,
    digital: BTreeMap<String, String>,
}

/// Expand every subcircuit instance in `netlist`.
///
/// `root` is the deck's resolved top-level parameter scope; `netlist`'s
/// top-level `{...}` sites must already be literalized against it (see
/// [`crate::netlist::elaborate::literalize`]). Body sites are evaluated here, in
/// each instance's own scope. The inputs are never modified: everything is
/// returned in [`ExpandedNetlist`], so a failing deck leaves the caller's
/// circuit, node table and AST untouched.
///
/// # Errors
/// Unknown subcircuit names, arity mismatches, circular definitions, nested
/// definitions or unresolved includes/body analysis cards, non-literal parameter values,
/// parameter evaluation failures and exceeded [`SubcircuitLimits`]. Each error
/// carries the location of the card that caused it, which for a body device is
/// the location inside the definition.
pub fn expand_subcircuits(
    netlist: &Netlist,
    root: &Arc<ParamScope>,
    limits: SubcircuitLimits,
) -> SpiceResult<ExpandedNetlist> {
    expand_with_output(netlist, root, limits, &OutputCards::default(), &[], &[])
}

/// Expand circuit and front-end cards together in source order.
///
/// C `translate` renames `.save`, `.ic`, and `.nodeset`; `.measure` and
/// `.four` remain unchanged. `inp_spsource` hoists `.four` before expansion
/// and extracts `.measure` afterwards.
///
/// # Errors
/// The same expansion and expression errors as [`expand_subcircuits`].
pub fn expand_with_output(
    netlist: &Netlist,
    root: &Arc<ParamScope>,
    limits: SubcircuitLimits,
    output: &OutputCards,
    measurements: &[MeasureCard],
    fourier: &[FourierCard],
) -> SpiceResult<ExpandedNetlist> {
    expand_with_prepared_output(netlist, root, limits, output, measurements, fourier, &[])
}

/// Expand a deck with typed prepared source declarations, binding each in its
/// actual lexical occurrence scope before constructing the physical device.
/// # Errors
/// Invalid, duplicate or unbound source identity, scope or waveform.
pub fn expand_prepared(
    netlist: &Netlist,
    root: &Arc<ParamScope>,
    limits: SubcircuitLimits,
    waveforms: &[model_normalize_ngspice::waveform::PreparedWaveform],
) -> SpiceResult<ExpandedNetlist> {
    expand_with_prepared_output(
        netlist,
        root,
        limits,
        &OutputCards::default(),
        &[],
        &[],
        waveforms,
    )
}

fn expand_with_prepared_output(
    netlist: &Netlist,
    root: &Arc<ParamScope>,
    limits: SubcircuitLimits,
    output: &OutputCards,
    measurements: &[MeasureCard],
    fourier: &[FourierCard],
    waveforms: &[model_normalize_ngspice::waveform::PreparedWaveform],
) -> SpiceResult<ExpandedNetlist> {
    let waveforms = PreparedIndex::new(netlist, waveforms)?;
    let sequence = expansion_plan(ScopeView::from(netlist), output, measurements)?;
    let definitions = definitions(netlist);
    check_acyclic(&definitions, &netlist.devices)?;
    let active = netlist.active_subcircuit_indices();
    let mut hoisted_fourier = Vec::new();
    let root_view = ScopeView::from(netlist);
    for card in &netlist.cards {
        let entry = root_view
            .entry(card.kind)
            .map_err(|error| SpiceError::parse(card.source.location.clone(), error.to_string()))?;
        match (card.kind, entry) {
            (_, ScopeEntry::Fourier(index)) => {
                hoisted_fourier.extend(side_card(fourier, index, &card.source.location)?.cloned());
            }
            (ScopedCardKind::Subcircuit(index), ScopeEntry::Subcircuit(definition))
                if active.contains(&index) =>
            {
                hoisted_fourier.extend(definition.fourier.iter().cloned());
            }
            _ => {}
        }
    }
    if netlist.cards.is_empty() {
        hoisted_fourier.extend_from_slice(fourier);
    }
    let mut budget = EvalBudget::default();
    let binder = crate::netlist::prepared_initialization::Binder::new(netlist, root, &mut budget)?;
    let mut expander = Expander {
        binder,
        netlist,
        definitions,
        limits,
        budget,
        devices: Vec::new(),
        waveforms,
        prepared_waveforms: Vec::new(),
        models: Vec::new(),
        statistical_model_origins: BTreeSet::new(),
        frontend_cards: 0,
        frontend: Frontend {
            fourier: hoisted_fourier,
            ..Frontend::default()
        },
        global_nodes: netlist
            .global_node_names()
            .into_iter()
            .map(str::to_ascii_lowercase)
            .chain(["0".to_owned()])
            .collect(),
        emitted_locals: BTreeSet::new(),
        binned_bases: BTreeSet::new(),
        root_functions: root.functions().cloned(),
        plans: BTreeMap::new(),
    };
    let mut locals = vec![expander.root_models(root)];
    let directives = Directives {
        lexical_scope: &[],
        devices: &netlist.devices,
        definition: "",
        sequence: &sequence,
        initial_conditions: &netlist.initial_conditions,
        nodesets: &netlist.nodesets,
    };
    expander.expand_body(
        "",
        0,
        &HierarchyBindings::default(),
        root,
        &mut locals,
        &directives,
    )?;
    check_flattened_model_names(&expander.models, &expander.emitted_locals)?;
    check_binned_bases(
        &expander.models,
        &expander.emitted_locals,
        &expander.binned_bases,
    )?;
    Ok(ExpandedNetlist {
        prepared_initialization: expander
            .binder
            .map_or_else(Default::default, |binder| binder.bound),
        devices: expander.devices,
        prepared_waveforms: expander.prepared_waveforms,
        models: expander.models,
        statistical_model_origins: expander.statistical_model_origins,
        options: expander.frontend.options,
        initial_conditions: expander.frontend.initial_conditions,
        nodesets: expander.frontend.nodesets,
        output: expander.frontend.output,
        measurements: expander.frontend.measurements,
        fourier: expander.frontend.fourier,
    })
}

/// A declaration is matched once against physical artifact provenance and
/// lexical definition identity. Multiple occurrences then bind independently.
type WaveformKey = (String, String, String);
struct PreparedIndex<'a>(
    BTreeMap<WaveformKey, &'a model_normalize_ngspice::waveform::PreparedWaveform>,
);

impl<'a> PreparedIndex<'a> {
    fn key(device: &DeviceInstance, definition: &str) -> Option<WaveformKey> {
        let artifact = device.location.path().file_name()?.to_str()?;
        Some((
            artifact.to_owned(),
            definition.to_ascii_lowercase(),
            device.name.to_ascii_lowercase(),
        ))
    }

    fn new(
        netlist: &Netlist,
        declarations: &'a [model_normalize_ngspice::waveform::PreparedWaveform],
    ) -> SpiceResult<Self> {
        let mut index = BTreeMap::new();
        if declarations.is_empty() {
            return Ok(Self(index));
        }
        let mut identities = BTreeMap::<WaveformKey, usize>::new();
        for (definition, devices) in std::iter::once(("", netlist.devices.as_slice())).chain(
            netlist
                .subcircuits
                .iter()
                .map(|definition| (definition.name.as_str(), definition.devices.as_slice())),
        ) {
            for device in devices
                .iter()
                .filter(|device| matches!(device.designator, 'v' | 'i'))
            {
                if let Some(key) = Self::key(device, definition) {
                    *identities.entry(key).or_default() += 1;
                }
            }
        }
        for declaration in declarations {
            if declaration.scope.len() > 1 {
                return Err(SpiceError::circuit(
                    "nested prepared source definition is unsupported",
                ));
            }
            let key = (
                declaration.artifact.clone(),
                declaration
                    .scope
                    .first()
                    .map_or_else(String::new, |name| name.to_ascii_lowercase()),
                declaration.source.to_ascii_lowercase(),
            );
            if identities.get(&key) != Some(&1) || index.insert(key, declaration).is_some() {
                return Err(SpiceError::circuit(format!(
                    "ambiguous, duplicate or missing prepared source {} in {}",
                    declaration.source, declaration.artifact
                )));
            }
        }
        Ok(Self(index))
    }

    fn get(
        &self,
        device: &DeviceInstance,
        definition: &str,
    ) -> Option<&'a model_normalize_ngspice::waveform::PreparedWaveform> {
        if self.0.is_empty() || !matches!(device.designator, 'v' | 'i') {
            return None;
        }
        self.0.get(&Self::key(device, definition)?).copied()
    }
}

/// One frame's model identity. Ordinary model cards are moved into the
/// flattened deck once; statistical declarations retain their defining scope
/// and immutable source until each physical occurrence is bound.
#[derive(Debug)]
enum ModelMaterial {
    Emitted(usize),
    Pending(ModelCard),
    Statistical {
        card: ModelCard,
        owner: Arc<ParamScope>,
    },
}

#[derive(Debug)]
struct LocalModel {
    name: String,
    material: ModelMaterial,
    root: bool,
}

impl LocalModel {
    const fn statistical(&self) -> bool {
        matches!(self.material, ModelMaterial::Statistical { .. })
    }
}

#[derive(Default)]
struct Frontend {
    options: Vec<OptionCard>,
    initial_conditions: Vec<NodeHintCard>,
    nodesets: Vec<NodeHintCard>,
    output: OutputCards,
    measurements: Vec<MeasureCard>,
    fourier: Vec<FourierCard>,
}

struct Directives<'a> {
    /// Declaration identity, not the chain of invoking X instances. The
    /// current native definition index contains root lexical definitions.
    lexical_scope: &'a [String],
    devices: &'a [DeviceInstance],
    definition: &'a str,
    sequence: &'a [ExpansionItem<'a>],
    initial_conditions: &'a [NodeHintCard],
    nodesets: &'a [NodeHintCard],
}

#[derive(Clone, Copy)]
enum ExpansionItem<'a> {
    Device(usize),
    Frontend {
        location: &'a SourceLoc,
        action: FrontendAction<'a>,
    },
}

#[derive(Clone, Copy)]
enum FrontendAction<'a> {
    Options(&'a OptionCard),
    InitialCondition(usize),
    Nodeset(usize),
    Save(&'a SaveCard),
    Print(&'a PrintCard),
    Measure(&'a MeasureCard),
    NoOutput,
}

fn side_card<'a, T>(
    cards: &'a [T],
    index: usize,
    location: &SourceLoc,
) -> SpiceResult<Option<&'a T>> {
    if cards.is_empty() {
        return Ok(None);
    }
    cards
        .get(index)
        .map(Some)
        .ok_or_else(|| SpiceError::parse(location.clone(), "invalid frontend side-card index"))
}

/// Compile immutable lexical order once. Indexed entries are validated at
/// the boundary; output side channels follow their original occurrence order
/// rather than matching physical source locations.
fn expansion_plan<'a>(
    scope: ScopeView<'a>,
    output: &'a OutputCards,
    measurements: &'a [MeasureCard],
) -> SpiceResult<Vec<ExpansionItem<'a>>> {
    // File admission splices the selected source cards into this very
    // scope. Keep directives as provenance, not as deferred execution.
    // Syntax-only/programmatic decks with unresolved source references
    // must still fail rather than silently omit their contents.
    if let Some(include) = scope.includes.iter().find(|include| {
        include.resolved_path.is_none()
            || (include.section.is_some() && include.selected_section.is_none())
    }) {
        return Err(SpiceError::parse(
            include.location.clone(),
            format!(
                "unresolved source directive '{}' requires file admission",
                include.path
            ),
        ));
    }
    if scope.cards.is_empty() {
        // Programmatically constructed decks may have no source cards.
        return Ok((0..scope.devices.len())
            .map(ExpansionItem::Device)
            .collect());
    }
    scope
        .validate_order()
        .map_err(|error| SpiceError::circuit(error.to_string()))?;
    let mut sequence = Vec::new();
    for card in scope.cards {
        let location = &card.source.location;
        let entry = scope
            .entry(card.kind)
            .map_err(|error| SpiceError::parse(location.clone(), error.to_string()))?;
        let action = match (card.kind, entry) {
            (ScopedCardKind::Device(index), _) => {
                sequence.push(ExpansionItem::Device(index));
                continue;
            }
            (_, ScopeEntry::Options(option)) => FrontendAction::Options(option),
            (ScopedCardKind::InitialCondition(index), _) => FrontendAction::InitialCondition(index),
            (ScopedCardKind::Nodeset(index), _) => FrontendAction::Nodeset(index),
            (_, ScopeEntry::Save(index)) => side_card(&output.saves, index, location)?
                .map_or(FrontendAction::NoOutput, FrontendAction::Save),
            (_, ScopeEntry::Print(index)) => side_card(&output.prints, index, location)?
                .map_or(FrontendAction::NoOutput, FrontendAction::Print),
            (_, ScopeEntry::Measure(index)) => side_card(measurements, index, location)?
                .map_or(FrontendAction::NoOutput, FrontendAction::Measure),
            (_, ScopeEntry::Control) => FrontendAction::NoOutput,
            _ => continue,
        };
        sequence.push(ExpansionItem::Frontend { location, action });
    }
    Ok(sequence)
}

struct DefinitionPlan<'a> {
    sequence: Vec<ExpansionItem<'a>>,
    parameters: Vec<ParamCard>,
    functions: Option<Arc<FunctionScope>>,
}

struct Expander<'a> {
    binder: Option<crate::netlist::prepared_initialization::Binder<'a>>,
    netlist: &'a Netlist,
    definitions: BTreeMap<String, &'a Subcircuit>,
    limits: SubcircuitLimits,
    budget: EvalBudget,
    devices: Vec<DeviceInstance>,
    waveforms: PreparedIndex<'a>,
    prepared_waveforms: Vec<(String, super::prepared_waveform::PreparedForcing)>,
    models: Vec<ModelCard>,
    statistical_model_origins: BTreeSet<String>,
    frontend: Frontend,
    frontend_cards: usize,
    global_nodes: BTreeSet<String>,
    /// Lowercased names of the body-local models that were renamed and emitted.
    emitted_locals: BTreeSet<String>,
    /// Lowercased flattened base names (`x1.nch`) of body-local binned sets.
    binned_bases: BTreeSet<String>,
    /// The deck's top-level `.func` definitions.
    root_functions: Option<Arc<FunctionScope>>,
    /// Lexical declarations are independent of an instance's bound values.
    plans: BTreeMap<String, Arc<DefinitionPlan<'a>>>,
}

/// Definition lookup: first declaration wins, like `ModelResolver` and C's
/// `INPmakeMod`-style duplicate policy. Names are case-insensitive.
fn definitions(netlist: &Netlist) -> BTreeMap<String, &Subcircuit> {
    let mut map: BTreeMap<String, &Subcircuit> = BTreeMap::new();
    for definition in &netlist.subcircuits {
        map.entry(definition.name.to_ascii_lowercase())
            .or_insert(definition);
    }
    map
}

/// Reject recursion an instance can actually reach.
///
/// The definition graph is directed: an edge `a -> b` means `a`'s body
/// instantiates `b`. A strongly connected component of more than one node, or a
/// self-edge, is a cycle. Unknown targets are not reported here: expansion sees
/// the invoking card and reports the instance location. Only definitions
/// reachable from a top-level `X` are checked, so a cyclic definition that
/// nothing instantiates is dead text exactly like any other unused definition.
fn check_acyclic(
    definitions: &BTreeMap<String, &Subcircuit>,
    devices: &[DeviceInstance],
) -> SpiceResult<()> {
    let mut graph: DiGraph<String, ()> = DiGraph::new();
    let mut index: BTreeMap<String, NodeIndex> = BTreeMap::new();
    for name in definitions.keys() {
        index.insert(name.clone(), graph.add_node(name.clone()));
    }
    for (name, body) in definitions {
        for device in &body.devices {
            if device.designator != 'x' {
                continue;
            }
            let Some(target) = device.model.as_ref().map(|name| name.to_ascii_lowercase()) else {
                continue;
            };
            if let Some(to) = index.get(&target)
                && !graph.contains_edge(index[name], *to)
            {
                graph.add_edge(index[name], *to, ());
            }
        }
    }
    let mut cyclic: Vec<Vec<NodeIndex>> = {
        // Breadth-first from every top-level `X` target; a strongly connected
        // component is either wholly reachable or wholly unreachable.
        let mut reachable: BTreeSet<NodeIndex> = BTreeSet::new();
        let mut queue: std::collections::VecDeque<NodeIndex> = devices
            .iter()
            .filter(|device| device.designator == 'x')
            .filter_map(|device| device.model.as_ref())
            .filter_map(|name| index.get(&name.to_ascii_lowercase()).copied())
            .collect();
        while let Some(node) = queue.pop_front() {
            if !reachable.insert(node) {
                continue;
            }
            queue.extend(graph.neighbors_directed(node, Direction::Outgoing));
        }
        tarjan_scc(&graph)
            .into_iter()
            .filter(|component| component.iter().any(|node| reachable.contains(node)))
            .filter(|component| {
                component.len() > 1 || graph.find_edge(component[0], component[0]).is_some()
            })
            .collect()
    };
    if cyclic.is_empty() {
        return Ok(());
    }
    cyclic.sort_by_key(|component| component.iter().map(|&node| graph[node].clone()).min());
    let component = &cyclic[0];
    let start = component
        .iter()
        .copied()
        .min_by_key(|&node| graph[node].clone())
        .ok_or_else(|| SpiceError::circuit("empty cyclic subcircuit component"))?;
    let path = shortest_cycle(&graph, component, start);
    let chain: Vec<String> = path
        .iter()
        .map(|&node| format!("'{}'", graph[node]))
        .collect();
    let body = definitions[&graph[start]];
    Err(SpiceError::parse(
        body.location.clone(),
        format!(
            "circular subcircuit definition: {}; expansion would not terminate",
            chain.join(" -> ")
        ),
    ))
}

/// Reject a declaration that would join (or shadow) a body-local binning set.
///
/// A body's `nch.1`, `nch.2` become `x1.nch.1`, `x1.nch.2` and its `M`
/// references `x1.nch`, which `ModelResolver` bins. A root `.model x1.nch`
/// (exact names win over bins) or `.model x1.nch.7` (an extra candidate)
/// would silently change the selection, so either is reported.
fn check_binned_bases(
    models: &[ModelCard],
    renamed: &BTreeSet<String>,
    bases: &BTreeSet<String>,
) -> SpiceResult<()> {
    for card in models {
        let name = card.name.to_ascii_lowercase();
        if renamed.contains(&name) {
            continue;
        }
        // The strict-prefix base precedes the exact name in the old sorted
        // collision scan when both happen to match.
        if let Some(base) = bin_base(&name)
            .filter(|base| bases.contains(*base))
            .or_else(|| bases.get(&name).map(String::as_str))
        {
            return Err(SpiceError::parse(
                card.location.clone(),
                format!(
                    "model '{}' collides with the binned subcircuit models '{base}.<n>'; \
                     rename the top-level model",
                    card.name
                ),
            ));
        }
    }
    Ok(())
}

/// Reject a per-instance rename that collides with another flattened model.
///
/// A body-local model is renamed `<instance path>.<name>`, which is exactly the
/// shape a root declaration may already use (`.model x1.am ...`). The resolver
/// keeps the first declaration for a name, so a collision would silently solve
/// the device against the wrong card. Root-to-root duplicates keep their
/// pre-existing first-declaration behaviour; only a generated rename is
/// reported, because the deck never wrote that name.
fn check_flattened_model_names(
    models: &[ModelCard],
    renamed: &BTreeSet<String>,
) -> SpiceResult<()> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for card in models {
        *counts.entry(card.name.to_ascii_lowercase()).or_default() += 1;
    }
    if let Some(name) = counts
        .iter()
        .find(|(name, count)| **count > 1 && renamed.contains(*name))
        .map(|(name, _)| name.clone())
    {
        return Err(SpiceError::circuit(format!(
            "duplicate flattened model name '{name}': a per-instance subcircuit model \
             collides with another model declaration; rename the top-level model"
        )));
    }
    Ok(())
}

/// Shortest cycle through `start` inside `component`, as `start .. start`.
fn shortest_cycle(
    graph: &DiGraph<String, ()>,
    component: &[NodeIndex],
    start: NodeIndex,
) -> Vec<NodeIndex> {
    let mut parent: BTreeMap<NodeIndex, NodeIndex> = BTreeMap::new();
    let mut queue = std::collections::VecDeque::from([start]);
    while let Some(node) = queue.pop_front() {
        let mut next: Vec<NodeIndex> = graph
            .neighbors_directed(node, Direction::Outgoing)
            .filter(|target| component.contains(target))
            .collect();
        next.sort_by_key(|&node| graph[node].clone());
        for target in next {
            if target == start {
                let mut chain = vec![node];
                let mut at = node;
                while at != start {
                    at = parent[&at];
                    chain.push(at);
                }
                chain.reverse();
                chain.push(start);
                return chain;
            }
            if let std::collections::btree_map::Entry::Vacant(slot) = parent.entry(target) {
                slot.insert(node);
                queue.push_back(target);
            }
        }
    }
    vec![start, start]
}

impl<'a> Expander<'a> {
    fn directive(
        &mut self,
        location: &SourceLoc,
        action: FrontendAction<'_>,
        directives: &Directives<'_>,
        terminals: &BTreeMap<String, String>,
        path: &str,
        scope: &Arc<ParamScope>,
    ) -> SpiceResult<()> {
        self.frontend_cards += 1;
        if self.frontend_cards > self.limits.max_devices {
            return Err(SpiceError::parse(
                location.clone(),
                format!(
                    "subcircuit expansion exceeds the front-end card budget ({})",
                    self.limits.max_devices
                ),
            ));
        }
        match action {
            FrontendAction::Options(option) => {
                let mut option = option.clone();
                for setting in &mut option.settings {
                    if let Some(expression) = setting.expression.take() {
                        let value = scope.evaluate(&expression, &mut self.budget)?;
                        setting.value = Some(crate::netlist::ast::PositionedValue {
                            text: crate::netlist::elaborate::format_literal(value),
                            location: setting.location.clone(),
                        });
                    }
                }
                self.frontend.options.push(option);
            }
            FrontendAction::InitialCondition(index) | FrontendAction::Nodeset(index) => {
                let initial = matches!(action, FrontendAction::InitialCondition(_));
                let cards = if initial {
                    directives.initial_conditions
                } else {
                    directives.nodesets
                };
                let mut hint = cards[index].clone();
                for entry in &mut hint.entries {
                    entry.node = self.rewrite_node(&entry.node, terminals, path);
                }
                if initial {
                    self.frontend.initial_conditions.push(hint);
                } else {
                    self.frontend.nodesets.push(hint);
                }
            }
            FrontendAction::Save(save) => self.output_directive(save, terminals, path),
            FrontendAction::Print(print) => self.frontend.output.prints.push(print.clone()),
            FrontendAction::Measure(measure) => self.frontend.measurements.push(measure.clone()),
            FrontendAction::NoOutput => {}
        }
        Ok(())
    }

    /// Rename saved vectors at this instance while retaining print cards'
    /// reference behavior and order.
    fn output_directive(
        &mut self,
        save: &SaveCard,
        terminals: &BTreeMap<String, String>,
        path: &str,
    ) {
        let mut save = save.clone();
        for request in &mut save.requests {
            match &mut request.vector {
                RequestedVector::Voltage { positive, negative }
                | RequestedVector::Component {
                    positive, negative, ..
                } => {
                    *positive = self.rewrite_node(positive, terminals, path);
                    if let Some(negative) = negative {
                        *negative = self.rewrite_node(negative, terminals, path);
                    }
                }
                RequestedVector::Current { device } => {
                    *device = device_name(device, device.chars().next().unwrap_or_default(), path);
                }
                RequestedVector::Named { name, .. } if name.starts_with('@') => {
                    if let Some((device, parameter)) = name[1..].split_once('[') {
                        *name = format!(
                            "@{}[{parameter}",
                            device_name(device, device.chars().next().unwrap_or_default(), path)
                        );
                    }
                }
                RequestedVector::All | RequestedVector::Named { .. } => {}
            }
        }
        self.frontend.output.saves.push(save);
    }

    /// The root frame's models: every top-level declaration, unrenamed.
    fn root_models(&mut self, scope: &Arc<ParamScope>) -> DeclarationIndex<LocalModel> {
        let mut index = DeclarationIndex::new();
        for card in &self.netlist.models {
            let material = if crate::netlist::elaborate::model_uses_statistics(card, scope) {
                ModelMaterial::Statistical {
                    card: card.clone(),
                    owner: Arc::clone(scope),
                }
            } else {
                let index = self.models.len();
                self.models.push(card.clone());
                ModelMaterial::Emitted(index)
            };
            index.insert_first(&card.name, || LocalModel {
                name: card.name.clone(),
                material,
                root: true,
            });
        }
        index
    }

    /// One frame's literalized models, renamed to `<path>.<name>`. They stay
    /// out of the flattened deck until a device in this frame or a descendant
    /// frame references one, so unused declarations are not invented.
    fn local_models(
        cards: Vec<ModelCard>,
        path: &str,
        scope: &Arc<ParamScope>,
    ) -> DeclarationIndex<LocalModel> {
        let mut index = DeclarationIndex::new();
        for card in cards {
            let source_name = card.name.clone();
            index.insert_first(&source_name, || LocalModel {
                name: format!("{path}.{source_name}"),
                material: if crate::netlist::elaborate::model_uses_statistics(&card, scope) {
                    ModelMaterial::Statistical {
                        card,
                        owner: Arc::clone(scope),
                    }
                } else {
                    ModelMaterial::Pending(card)
                },
                root: false,
            });
        }
        index
    }

    /// Expand one scope level: top-level devices at the root, a definition's
    /// body inside an instance.
    fn expand_body(
        &mut self,
        path: &str,
        depth: usize,
        terminals: &HierarchyBindings,
        scope: &Arc<ParamScope>,
        locals: &mut Vec<DeclarationIndex<LocalModel>>,
        directives: &Directives<'_>,
    ) -> SpiceResult<()> {
        if let Some(binder) = &mut self.binder {
            binder.nodes(
                directives.lexical_scope,
                scope,
                |node| rewrite_node(&self.global_nodes, node, &terminals.analog, path),
                &mut self.budget,
            )?;
        }
        let devices = directives.devices;
        for &item in directives.sequence {
            let index = match item {
                ExpansionItem::Device(index) => index,
                ExpansionItem::Frontend { location, action } => {
                    self.directive(location, action, directives, &terminals.analog, path, scope)?;
                    continue;
                }
            };
            let original = devices.get(index).ok_or_else(|| {
                SpiceError::parse(
                    self.netlist.location.clone(),
                    "invalid device card index during expansion",
                )
            })?;
            let mut instance = original.clone();
            if instance.designator == 'x' {
                let target = instance
                    .model
                    .as_deref()
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                let Some(definition) = self.definitions.get(&target).copied() else {
                    return Err(SpiceError::parse(
                        instance.location.clone(),
                        format!(
                            "unknown subcircuit '{}' instantiated by {}",
                            instance.model.as_deref().unwrap_or_default(),
                            instance.name
                        ),
                    ));
                };
                self.expand_instance(
                    definition,
                    &instance,
                    (path, depth, terminals),
                    scope,
                    locals,
                )?;
                continue;
            }
            for node in &mut instance.nodes {
                *node = self.rewrite_node(node, &terminals.analog, path);
            }
            self.rewrite_code_ports(&mut instance, terminals, path);
            if self.devices.len() >= self.limits.max_devices {
                return Err(SpiceError::parse(
                    instance.location.clone(),
                    format!(
                        "subcircuit expansion produced more than {} devices",
                        self.limits.max_devices
                    ),
                ));
            }
            self.rewrite_model(&mut instance, scope, locals)?;
            self.rewrite_parameter_references(&mut instance, &terminals.analog, path);
            instance.name = device_name(&instance.name, instance.designator, path);
            self.bind_prepared_waveform(original, directives.definition, &instance, scope)?;
            if let Some(binder) = &mut self.binder {
                binder.device(
                    original,
                    &instance.name,
                    directives.lexical_scope,
                    scope,
                    &mut self.budget,
                )?;
            }
            self.devices.push(instance);
        }
        Ok(())
    }

    /// Rewrite instance-local device and node references in parameters.
    fn rewrite_parameter_references(
        &self,
        instance: &mut DeviceInstance,
        terminals: &BTreeMap<String, String>,
        path: &str,
    ) {
        // F/H controlling sources are translated like instance names
        // (`translate()` calls `translate_inst_name` for them), so they
        // always name a device of the same instance.
        for parameter in &mut instance.parameters {
            if let ParameterKind::Behavioural(expression) = &mut parameter.kind {
                // Node and source names inside a behavioural expression are
                // translated like the card's own (C expands the B line
                // text, `translate()` renaming v(...)/i(...) operands).
                expression.root.visit_mut(&mut |node| match &mut node.kind {
                    BExprKind::Voltage { positive, negative } => {
                        *positive = self.rewrite_node(positive, terminals, path);
                        if let Some(negative) = negative {
                            *negative = self.rewrite_node(negative, terminals, path);
                        }
                    }
                    BExprKind::Current(source) => {
                        let designator = source
                            .chars()
                            .next()
                            .unwrap_or_default()
                            .to_ascii_lowercase();
                        *source = device_name(source, designator, path);
                    }
                    _ => {}
                });
                continue;
            }
            if parameter.kind == ParameterKind::Instance {
                let designator = parameter
                    .value
                    .chars()
                    .next()
                    .unwrap_or_default()
                    .to_ascii_lowercase();
                parameter.value = device_name(&parameter.value, designator, path);
            }
        }
    }

    /// Bind a resolved waveform to the flattened source occurrence.
    fn bind_prepared_waveform(
        &mut self,
        original: &DeviceInstance,
        definition: &str,
        instance: &DeviceInstance,
        scope: &Arc<ParamScope>,
    ) -> SpiceResult<()> {
        if let Some(declaration) = self.waveforms.get(original, definition) {
            let mut ordinal = 0_u64;
            let site = scope
                .statistical_site(b"device", &original.name)
                .map(|site| site.field(b"prepared_waveform"));
            let resolved = declaration
                .resolve(|text| {
                    let text = text.trim();
                    let expression = crate::netlist::parser::parse_parameter_expression(
                        text.strip_prefix('{')
                            .and_then(|text| text.strip_suffix('}'))
                            .unwrap_or(text),
                        &original.location,
                    )?;
                    let field = site.map(|site| site.field(&ordinal.to_le_bytes()));
                    ordinal += 1;
                    Ok(scope.evaluate_at(&expression, field, &mut self.budget)?)
                })
                .map_err(|error| {
                    SpiceError::parse(
                        original.location.clone(),
                        format!("prepared source {}: {error}", instance.name),
                    )
                })?;
            self.prepared_waveforms.push((
                instance.name.clone(),
                super::prepared_waveform::PreparedForcing::new(resolved)?,
            ));
        }
        Ok(())
    }

    /// Expand one `X` card: bind ports, build the instance's parameter scope,
    /// literalize its body and recurse.
    fn expand_instance(
        &mut self,
        definition: &'a Subcircuit,
        instance: &DeviceInstance,
        frame: (&str, usize, &HierarchyBindings),
        parent: &Arc<ParamScope>,
        locals: &mut Vec<DeclarationIndex<LocalModel>>,
    ) -> SpiceResult<()> {
        let (path, depth, parent_terminals) = frame;
        if depth + 1 > self.limits.max_depth {
            return Err(SpiceError::parse(
                instance.location.clone(),
                format!(
                    "subcircuit nesting deeper than {} levels at {}",
                    self.limits.max_depth, instance.name
                ),
            ));
        }
        if definition.terminals.len() != instance.nodes.len() {
            return Err(SpiceError::parse(
                instance.location.clone(),
                format!(
                    "subcircuit '{}' has {} terminal(s), but {} supplies {}",
                    definition.name,
                    definition.terminals.len(),
                    instance.name,
                    instance.nodes.len()
                ),
            ));
        }
        if !definition.subcircuits.is_empty() {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: .subckt '{}' defines a nested subcircuit",
                    definition.location, definition.name
                ),
                C_REFERENCE,
            ));
        }
        if let Some(card) = definition.analyses.first() {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: analysis card inside .subckt '{}'",
                    card.location, definition.name
                ),
                C_REFERENCE,
            ));
        }
        let mut terminals = HierarchyBindings::default();
        for (formal, actual) in definition.terminals.iter().zip(&instance.nodes) {
            let analog_formal = NodeTable::canonical_name(formal, self.netlist.auto_gnd);
            let analog_actual = NodeTable::canonical_name(actual, self.netlist.auto_gnd);
            terminals.analog.insert(
                analog_formal,
                self.rewrite_node(&analog_actual, &parent_terminals.analog, path),
            );
            terminals.digital.insert(
                formal.clone(),
                Self::rewrite_digital_node(actual, &parent_terminals.digital, path),
            );
        }
        let (scope, plan) = self.instance_scope(definition, instance, parent)?;
        let child_path = if path.is_empty() {
            instance.name.clone()
        } else {
            format!("{path}.{}", instance.name)
        };
        let body =
            crate::netlist::elaborate::literalize_body(definition, &scope, &mut self.budget)?;
        locals.push(Self::local_models(body.models, &child_path, &scope));
        let result = self.expand_body(
            &child_path,
            depth + 1,
            &terminals,
            &scope,
            locals,
            &Directives {
                lexical_scope: std::slice::from_ref(&definition.name),
                devices: &body.devices,
                definition: &definition.name,
                sequence: &plan.sequence,
                initial_conditions: &body.initial_conditions,
                nodesets: &body.nodesets,
            },
        );
        locals.pop();
        result
    }

    /// The instance's parameter scope: overrides as bound values, formal
    /// defaults as the first `.param` card, body `.param` cards after them.
    fn instance_scope(
        &mut self,
        definition: &'a Subcircuit,
        instance: &DeviceInstance,
        parent: &Arc<ParamScope>,
    ) -> SpiceResult<(Arc<ParamScope>, Arc<DefinitionPlan<'a>>)> {
        for setting in definition.options.iter().flat_map(|card| &card.settings) {
            if matches!(setting.name.as_str(), "seed" | "rndseed") {
                return Err(SpiceError::parse(
                    setting.location.clone(),
                    "statistical seed belongs to the top-level deck, not a subcircuit",
                ));
            }
        }
        let mut bindings = Vec::with_capacity(instance.parameters.len());
        for parameter in &instance.parameters {
            bindings.push(ParamBinding {
                name: parameter.name.clone(),
                value: self.parameter_value(parameter, parent)?,
                location: parameter.location.clone(),
            });
        }
        let plan = self.definition_plan(definition)?;
        let scope = Arc::new(ParamScope::resolve_instance_scoped(
            Some(Arc::clone(parent)),
            &instance.name,
            plan.functions.clone(),
            &bindings,
            &plan.parameters,
            &mut self.budget,
        )?);
        Ok((scope, plan))
    }

    /// Cache only immutable lexical syntax. Formal defaults and body
    /// parameters are still evaluated with each caller's bindings and budget.
    fn definition_plan(
        &mut self,
        definition: &'a Subcircuit,
    ) -> SpiceResult<Arc<DefinitionPlan<'a>>> {
        let key = definition.name.to_ascii_lowercase();
        if let Some(plan) = self.plans.get(&key) {
            return Ok(Arc::clone(plan));
        }
        let mut parameters = Vec::new();
        if !definition.parameters.is_empty() {
            let assignments = definition
                .parameters
                .iter()
                .map(formal_assignment)
                .collect::<SpiceResult<_>>()?;
            parameters.push(ParamCard {
                assignments,
                location: definition.location.clone(),
            });
        }
        parameters.extend(definition.params.iter().cloned());
        // Functions see their lexical definition and root, not the caller's local functions.
        let functions = if definition.functions.is_empty() {
            self.root_functions.clone()
        } else {
            Some(Arc::new(FunctionScope::new(
                self.root_functions.clone(),
                &definition.functions,
                &self.budget,
            )?))
        };
        let sequence = expansion_plan(
            ScopeView::from(definition),
            &definition.output,
            &definition.measurements,
        )?;
        let plan = Arc::new(DefinitionPlan {
            sequence,
            parameters,
            functions,
        });
        self.plans.insert(key, Arc::clone(&plan));
        Ok(plan)
    }

    /// One instance parameter's value, evaluated at the call site.
    fn parameter_value(
        &mut self,
        parameter: &ParameterAssignment,
        scope: &ParamScope,
    ) -> SpiceResult<Real> {
        match &parameter.kind {
            ParameterKind::Scalar => finite_number(parameter),
            ParameterKind::Expression(expression) => scope.evaluate(expression, &mut self.budget),
            _ => Err(SpiceError::not_yet_ported(
                format!(
                    "{}: subcircuit instance parameter '{}' = {}",
                    parameter.location, parameter.name, parameter.value
                ),
                "src/frontend/numparam/spicenum.c",
            )),
        }
    }

    /// Rename a body node. Ground and `.global` nodes are exempt; a formal
    /// terminal is replaced by the actual node it was bound to; anything else is
    /// prefixed with the instance path.
    fn rewrite_code_ports(
        &self,
        instance: &mut DeviceInstance,
        terminals: &HierarchyBindings,
        path: &str,
    ) {
        for terminal in instance
            .code_ports
            .iter_mut()
            .flat_map(|port| &mut port.terminals)
        {
            let rewrite = |name: &str| {
                if terminal.kind == CodePortKind::Digital {
                    Self::rewrite_digital_node(name, &terminals.digital, path)
                } else {
                    self.rewrite_node(name, &terminals.analog, path)
                }
            };
            terminal.positive = rewrite(&terminal.positive);
            if let Some(negative) = &mut terminal.negative {
                *negative = rewrite(negative);
            }
        }
    }

    fn rewrite_digital_node(
        name: &str,
        terminals: &BTreeMap<String, String>,
        path: &str,
    ) -> String {
        terminals.get(name).cloned().unwrap_or_else(|| {
            if path.is_empty() {
                name.to_owned()
            } else {
                format!("{path}.{name}")
            }
        })
    }

    fn rewrite_node(&self, name: &str, terminals: &BTreeMap<String, String>, path: &str) -> String {
        rewrite_node(&self.global_nodes, name, terminals, path)
    }

    /// Rewrite one model reference to the flattened name of the declaration it
    /// resolves to, emitting a local declaration the first time it is used.
    /// An unresolved reference keeps its name; the model resolver reports it
    /// with the instance location.
    fn rewrite_model(
        &mut self,
        instance: &mut DeviceInstance,
        scope: &ParamScope,
        locals: &mut [DeclarationIndex<LocalModel>],
    ) -> SpiceResult<()> {
        let Some(reference) = instance.model.as_deref() else {
            return Ok(());
        };
        let occurrence = || {
            scope
                .statistical_site(b"device", &instance.name)
                .ok_or_else(|| {
                    SpiceError::parse(
                        instance.location.clone(),
                        "statistical device has no physical occurrence",
                    )
                })
        };
        for frame in locals.iter_mut().rev() {
            if let Some(local) = frame.get_mut(reference) {
                let name = if local.statistical() {
                    let site = occurrence()?;
                    let name = format!("{}.__stat_{}", local.name, occurrence_suffix(site));
                    self.emit_statistical_model(local, &name, site)?;
                    name
                } else {
                    self.emit_model(local);
                    local.name.clone()
                };
                instance.model = Some(name);
                return Ok(());
            }
            // A frame declaring only `<name>.<digits>` bins also captures
            // MOS model names. Each group is already in declaration order.
            if instance.designator.eq_ignore_ascii_case(&'m')
                && let Some(base) = self.rewrite_binned(reference, frame, &occurrence)?
            {
                instance.model = Some(base);
                return Ok(());
            }
        }
        Ok(())
    }

    fn emit_model(&mut self, local: &mut LocalModel) {
        let material = std::mem::replace(
            &mut local.material,
            ModelMaterial::Emitted(self.models.len()),
        );
        match material {
            ModelMaterial::Pending(mut card) => {
                card.name.clone_from(&local.name);
                self.emitted_locals.insert(local.name.to_ascii_lowercase());
                self.models.push(card);
            }
            other => local.material = other,
        }
    }

    fn emit_statistical_model(
        &mut self,
        local: &LocalModel,
        name: &str,
        site: crate::netlist::eval::statistics::PreparedSite,
    ) -> SpiceResult<()> {
        let mut card = match &local.material {
            ModelMaterial::Statistical { card, owner } => {
                crate::netlist::elaborate::realize_model(card, owner, site, &mut self.budget)?
            }
            ModelMaterial::Pending(card) => card.clone(),
            ModelMaterial::Emitted(index) => self
                .models
                .get(*index)
                .ok_or_else(|| SpiceError::circuit("missing emitted model declaration"))?
                .clone(),
        };
        name.clone_into(&mut card.name);
        self.emitted_locals.insert(name.to_ascii_lowercase());
        if local.root {
            self.statistical_model_origins
                .insert(local.name.to_ascii_lowercase());
        }
        self.models.push(card);
        Ok(())
    }

    /// Bind an M reference to a frame's bin set, emitting the ordered
    /// declarations once and leaving geometry selection to `ModelResolver`.
    fn rewrite_binned(
        &mut self,
        key: &str,
        frame: &mut DeclarationIndex<LocalModel>,
        occurrence: &impl Fn() -> SpiceResult<crate::netlist::eval::statistics::PreparedSite>,
    ) -> SpiceResult<Option<String>> {
        let (ids, entries) = frame.bin_slots_mut(key);
        let Some(&first_id) = ids.first() else {
            return Ok(None);
        };
        let Some(base) = bin_base(&entries[first_id].name).map(str::to_owned) else {
            return Ok(None);
        };
        if ids.iter().any(|&id| entries[id].statistical()) {
            let site = occurrence()?;
            let sampled_base = format!("{base}.__stat_{}", occurrence_suffix(site));
            for &id in ids {
                let local = &entries[id];
                let (_, suffix) = local
                    .name
                    .rsplit_once('.')
                    .ok_or_else(|| SpiceError::circuit("invalid model bin identity"))?;
                self.emit_statistical_model(local, &format!("{sampled_base}.{suffix}"), site)?;
            }
            self.binned_bases.insert(sampled_base.to_ascii_lowercase());
            return Ok(Some(sampled_base));
        }
        for &id in ids {
            self.emit_model(&mut entries[id]);
        }
        if !base.eq_ignore_ascii_case(key) {
            self.binned_bases.insert(base.to_ascii_lowercase());
        }
        Ok(Some(base))
    }
}

fn occurrence_suffix(site: crate::netlist::eval::statistics::PreparedSite) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut suffix = String::with_capacity(64);
    for byte in site.occurrence.bytes() {
        suffix.push(char::from(HEX[usize::from(byte >> 4)]));
        suffix.push(char::from(HEX[usize::from(byte & 15)]));
    }
    suffix
}

fn rewrite_node(
    global_nodes: &BTreeSet<String>,
    name: &str,
    terminals: &BTreeMap<String, String>,
    path: &str,
) -> String {
    if global_nodes.contains(&name.to_ascii_lowercase()) {
        return name.to_owned();
    }
    if let Some(actual) = terminals.get(name) {
        return actual.clone();
    }
    if path.is_empty() {
        return name.to_owned();
    }
    format!("{path}.{name}")
}

/// `translate_inst_name`: the designator prefix is not repeated for an `X`.
fn device_name(name: &str, designator: char, path: &str) -> String {
    if path.is_empty() {
        name.to_owned()
    } else if designator == 'x' {
        format!("{path}.{name}")
    } else {
        format!("{designator}.{path}.{name}")
    }
}

/// One formal default as a `.param` assignment, so that evaluation orders it
/// with the body's own `.param` cards (C: numparam's shared environment).
fn formal_assignment(formal: &ParameterAssignment) -> SpiceResult<ParamAssignment> {
    let expression = match &formal.kind {
        ParameterKind::Expression(expression) => (**expression).clone(),
        ParameterKind::Scalar => {
            literal_expression(&formal.value, finite_number(formal)?, &formal.location)
        }
        _ => {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: subcircuit formal '{}' = {}",
                    formal.location, formal.name, formal.value
                ),
                "src/frontend/numparam/spicenum.c",
            ));
        }
    };
    Ok(ParamAssignment {
        name: formal.name.clone(),
        name_span: SourceSpan {
            start: formal.location.clone(),
            end: formal
                .location
                .at_column(formal.location.column + byte_len(&formal.name)),
            logical: 0..formal.name.len(),
        },
        expression,
    })
}

/// A finite scalar parameter value, read as numparam reads subcircuit call
/// and `params:` values (`fetchnumber()`: `%lG` times the unit); anything
/// else is rejected explicitly.
fn finite_number(parameter: &ParameterAssignment) -> SpiceResult<Real> {
    crate::primitives::parse_numparam_number(&parameter.value)
        .filter(|value| value.is_finite())
        .ok_or_else(|| {
            SpiceError::parse(
                parameter.location.clone(),
                format!(
                    "parameter '{}' = {} is not a finite number",
                    parameter.name, parameter.value
                ),
            )
        })
}

/// Wrap a literal scalar as an expression node, for uniform evaluation.
fn literal_expression(text: &str, value: Real, start: &SourceLoc) -> ParameterExpression {
    let span = SourceSpan {
        start: start.clone(),
        end: start.at_column(start.column + byte_len(text)),
        logical: 0..text.len(),
    };
    ParameterExpression {
        text: text.to_owned(),
        braced: false,
        quoted: false,
        span: span.clone(),
        root: Expr {
            kind: ExprKind::Number {
                value,
                spelling: text.to_owned(),
            },
            span,
        },
    }
}

const fn byte_len(text: &str) -> usize {
    text.len()
}
