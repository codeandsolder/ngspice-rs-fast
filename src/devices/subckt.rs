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
//! Nested `.subckt` definitions inside a body, `.include`/`.lib` directives, and
//! analysis cards inside a body are rejected explicitly rather than ignored.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::devices::binning::{NameMatch, model_name_match};
use crate::netlist::ast::{
    DeviceInstance, FourierCard, MeasureCard, ModelCard, Netlist, NodeHintCard, OptionCard,
    OutputCards, ParamAssignment, ParamCard, ParameterAssignment, ParameterKind, PrintCard,
    RequestedVector, SaveCard, ScopeEntry, ScopeView, ScopedCardKind, Subcircuit,
};
use crate::netlist::bexpr::BExprKind;
use crate::netlist::eval::{EvalBudget, FunctionScope, ParamBinding, ParamScope};
use crate::netlist::expr::{Expr, ExprKind, ParameterExpression, SourceSpan};
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};
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
    /// Top-level devices in deck order, `X` instances expanded in place.
    pub devices: Vec<DeviceInstance>,
    /// Root `.model` cards in declaration order, then each instance's renamed
    /// local models in first-reference order.
    pub models: Vec<ModelCard>,
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
/// definitions or body `.include`/analysis cards, non-literal parameter values,
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
    let mut expander = Expander {
        netlist,
        definitions,
        limits,
        budget: EvalBudget::default(),
        devices: Vec::new(),
        models: netlist.models.clone(),
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
    let mut locals = vec![expander.root_models()];
    let directives = Directives {
        devices: &netlist.devices,
        sequence: &sequence,
        initial_conditions: &netlist.initial_conditions,
        nodesets: &netlist.nodesets,
    };
    expander.expand_body("", 0, &BTreeMap::new(), root, &mut locals, &directives)?;
    check_flattened_model_names(&expander.models, &expander.emitted_locals)?;
    check_binned_bases(
        &expander.models,
        &expander.emitted_locals,
        &expander.binned_bases,
    )?;
    Ok(ExpandedNetlist {
        devices: expander.devices,
        models: expander.models,
        options: expander.frontend.options,
        initial_conditions: expander.frontend.initial_conditions,
        nodesets: expander.frontend.nodesets,
        output: expander.frontend.output,
        measurements: expander.frontend.measurements,
        fourier: expander.frontend.fourier,
    })
}

/// One frame's model declarations: canonical local name to flattened name and
/// the card as it will be declared at top level.
#[derive(Debug, Clone)]
struct LocalModel {
    name: String,
    card: ModelCard,
    emitted: bool,
    /// Declaration index of the (first) card within its frame; binned sets are
    /// emitted in this order because C's last declared matching bin wins.
    order: usize,
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
    devices: &'a [DeviceInstance],
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
    netlist: &'a Netlist,
    definitions: BTreeMap<String, &'a Subcircuit>,
    limits: SubcircuitLimits,
    budget: EvalBudget,
    devices: Vec<DeviceInstance>,
    models: Vec<ModelCard>,
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
        if let Some(base) = bases
            .iter()
            .find(|base| model_name_match(base, &name).is_some())
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
    fn root_models(&self) -> BTreeMap<String, LocalModel> {
        let mut map: BTreeMap<String, LocalModel> = BTreeMap::new();
        for (order, card) in self.netlist.models.iter().enumerate() {
            map.entry(card.name.to_ascii_lowercase())
                .or_insert_with(|| LocalModel {
                    name: card.name.clone(),
                    card: card.clone(),
                    emitted: true,
                    order,
                });
        }
        map
    }

    /// One frame's literalized models, renamed to `<path>.<name>`. They stay
    /// out of the flattened deck until a device in this frame or a descendant
    /// frame references one, so unused declarations are not invented.
    fn local_models(cards: &[ModelCard], path: &str) -> BTreeMap<String, LocalModel> {
        let mut map: BTreeMap<String, LocalModel> = BTreeMap::new();
        for (order, card) in cards.iter().enumerate() {
            let name = format!("{path}.{}", card.name);
            let renamed = ModelCard {
                name: name.clone(),
                ..card.clone()
            };
            map.entry(card.name.to_ascii_lowercase())
                .or_insert_with(|| LocalModel {
                    name,
                    card: renamed,
                    emitted: false,
                    order,
                });
        }
        map
    }

    /// Expand one scope level: top-level devices at the root, a definition's
    /// body inside an instance.
    fn expand_body(
        &mut self,
        path: &str,
        depth: usize,
        terminals: &BTreeMap<String, String>,
        scope: &Arc<ParamScope>,
        locals: &mut Vec<BTreeMap<String, LocalModel>>,
        directives: &Directives<'_>,
    ) -> SpiceResult<()> {
        let devices = directives.devices;
        for &item in directives.sequence {
            let index = match item {
                ExpansionItem::Device(index) => index,
                ExpansionItem::Frontend { location, action } => {
                    self.directive(location, action, directives, terminals, path, scope)?;
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
            for node in &mut instance.nodes {
                *node = self.rewrite_node(node, terminals, path);
            }
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
                self.expand_instance(definition, &instance, path, depth, scope, locals)?;
                continue;
            }
            if self.devices.len() >= self.limits.max_devices {
                return Err(SpiceError::parse(
                    instance.location.clone(),
                    format!(
                        "subcircuit expansion produced more than {} devices",
                        self.limits.max_devices
                    ),
                ));
            }
            self.rewrite_model(&mut instance, locals);
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
            instance.name = device_name(&instance.name, instance.designator, path);
            self.devices.push(instance);
        }
        Ok(())
    }

    /// Expand one `X` card: bind ports, build the instance's parameter scope,
    /// literalize its body and recurse.
    fn expand_instance(
        &mut self,
        definition: &'a Subcircuit,
        instance: &DeviceInstance,
        path: &str,
        depth: usize,
        parent: &Arc<ParamScope>,
        locals: &mut Vec<BTreeMap<String, LocalModel>>,
    ) -> SpiceResult<()> {
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
        if !definition.includes.is_empty() {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: .include/.lib inside .subckt '{}'",
                    definition.location, definition.name
                ),
                "src/frontend/inpcom.c (lexical include expansion)",
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
        let mut terminals: BTreeMap<String, String> = BTreeMap::new();
        for (formal, actual) in definition.terminals.iter().zip(&instance.nodes) {
            terminals.insert(formal.clone(), actual.clone());
        }
        let (scope, plan) = self.instance_scope(definition, instance, parent)?;
        let child_path = if path.is_empty() {
            instance.name.clone()
        } else {
            format!("{path}.{}", instance.name)
        };
        let body =
            crate::netlist::elaborate::literalize_body(definition, &scope, &mut self.budget)?;
        locals.push(Self::local_models(&body.models, &child_path));
        let result = self.expand_body(
            &child_path,
            depth + 1,
            &terminals,
            &scope,
            locals,
            &Directives {
                devices: &body.devices,
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
    fn rewrite_node(&self, name: &str, terminals: &BTreeMap<String, String>, path: &str) -> String {
        if self.global_nodes.contains(&name.to_ascii_lowercase()) {
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

    /// Rewrite one model reference to the flattened name of the declaration it
    /// resolves to, emitting a local declaration the first time it is used.
    /// An unresolved reference keeps its name; the model resolver reports it
    /// with the instance location.
    fn rewrite_model(
        &mut self,
        instance: &mut DeviceInstance,
        locals: &mut [BTreeMap<String, LocalModel>],
    ) {
        let Some(reference) = instance.model.clone() else {
            return;
        };
        let key = reference.to_ascii_lowercase();
        for index in (0..locals.len()).rev() {
            let Some(local) = locals[index].get(&key) else {
                // subckt.c matches M models with model_name_match, so a frame
                // declaring only `<name>.<digits>` bins also captures the name.
                if instance.designator.eq_ignore_ascii_case(&'m')
                    && self.rewrite_binned(instance, &key, &mut locals[index])
                {
                    return;
                }
                continue;
            };
            let name = local.name.clone();
            let emitted = local.emitted;
            let card = local.card.clone();
            if !emitted {
                self.models.push(card);
                self.emitted_locals.insert(name.to_ascii_lowercase());
                if let Some(slot) = locals[index].get_mut(&key) {
                    slot.emitted = true;
                }
            }
            instance.model = Some(name);
            return;
        }
    }

    /// Bind an `M` reference to one frame's `<key>.<digits>` binning set:
    /// emit every bin of the set (declaration order) and rewrite the reference
    /// to the set's flattened base name, leaving the choice of bin to
    /// `ModelResolver`. `false` when the frame has no such bin.
    fn rewrite_binned(
        &mut self,
        instance: &mut DeviceInstance,
        key: &str,
        frame: &mut BTreeMap<String, LocalModel>,
    ) -> bool {
        let mut bins: Vec<(&String, &mut LocalModel)> = frame
            .iter_mut()
            .filter(|(name, _)| model_name_match(key, name) == Some(NameMatch::Bin))
            .collect();
        bins.sort_by_key(|(_, local)| local.order);
        let Some((first_key, first)) = bins.first() else {
            return false;
        };
        // `<path>.nch.1` minus the `.1` the original `nch.1` adds to `nch`.
        let suffix = first_key.len() - key.len();
        let base = first.name[..first.name.len() - suffix].to_owned();
        let renamed = !first.name.eq_ignore_ascii_case(first_key);
        for (_, local) in &mut bins {
            if !local.emitted {
                self.models.push(local.card.clone());
                self.emitted_locals.insert(local.name.to_ascii_lowercase());
                local.emitted = true;
            }
        }
        if renamed {
            self.binned_bases.insert(base.to_ascii_lowercase());
        }
        instance.model = Some(base);
        true
    }
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
