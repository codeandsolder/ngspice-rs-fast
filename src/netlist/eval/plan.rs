//! Reusable structural parameter scheduling and isolated candidate values.

use super::{
    Bindings, DefinitionState, DependencyReference, EvalBudget, Failure, FunctionScope,
    ParamBinding, ParamScope, ParamState, ScopeDefinitions, canonical_lookup, collect_definitions,
    evaluate_definition, evaluate_root,
};
use crate::netlist::ast::ParamCard;
use crate::netlist::expr::ParameterExpression;
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

/// Identity of a lexical parameter scope within a prepared plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParameterScopeId(usize);

/// Stable handle to an active entry in its defining scope, not to a name
/// resolved again for each candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParameterSlot {
    scope: ParameterScopeId,
    local: usize,
}

impl ParameterSlot {
    /// Defining scope of this entry.
    #[must_use]
    pub const fn scope(self) -> ParameterScopeId {
        self.scope
    }
}

/// One independent candidate coordinate, in the order supplied to the plan.
#[derive(Debug, Clone)]
pub struct ParameterInput {
    /// Slot in the defining scope (not an inherited alias).
    pub slot: ParameterSlot,
    /// Canonical local name.
    pub name: String,
    /// Value in the independently resolved baseline deck.
    pub default: Real,
}

/// Structural inputs for one lexical scope. Parent scopes must be added first.
pub struct ParameterScopeSpec<'a> {
    /// Parent parameter scope, if any.
    pub parent: Option<ParameterScopeId>,
    /// Lexically visible functions; pass the definition's function scope for instances.
    pub functions: Option<Arc<FunctionScope>>,
    /// Pre-evaluated formal or instance bindings.
    pub bindings: &'a [ParamBinding],
    /// Definitions in deck order, including formal defaults.
    pub cards: &'a [ParamCard],
    /// Whether instance bindings supersede cards rather than rejecting redefinitions.
    pub instance_overrides: bool,
    /// Active local names selected as independent numeric coordinates.
    pub tunables: &'a [&'a str],
}

struct PlannedScope {
    baseline: Arc<ParamScope>,
    parent: Option<ParameterScopeId>,
    start: usize,
}

struct PlannedNode {
    scope: ParameterScopeId,
    local: usize,
    expression: Option<ParameterExpression>,
    references: HashMap<String, usize>,
    rank: Option<usize>,
    dependents: Vec<usize>,
    baseline: Option<Real>,
}

struct ScopeSchedule {
    sequence: Vec<usize>,
    references: Vec<HashMap<String, usize>>,
}

/// Immutable dependency schedule once borrowed by candidate sessions.
///
/// Build scopes in parent-before-child order, then share this plan among
/// independent sessions. Selecting an active definition as tunable replaces
/// its evaluated value, not its name or lexical visibility. Only numeric
/// evaluation is supported; changing topology or function definitions needs
/// a new plan. The original deck is validated before any inputs are replaced.
/// Scope IDs and slots belong to the plan that issued them.
#[derive(Default)]
pub struct ParameterPlan {
    scopes: Vec<PlannedScope>,
    nodes: Vec<PlannedNode>,
    inputs: Vec<ParameterInput>,
}

impl ParameterPlan {
    /// Empty plan.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            scopes: Vec::new(),
            nodes: Vec::new(),
            inputs: Vec::new(),
        }
    }

    /// Candidate coordinate order.
    #[must_use]
    pub fn inputs(&self) -> &[ParameterInput] {
        &self.inputs
    }

    /// Resolved original scope, including provenance and function environment.
    #[must_use]
    pub fn baseline(&self, scope: ParameterScopeId) -> Option<&ParamScope> {
        self.scopes
            .get(scope.0)
            .map(|scope| scope.baseline.as_ref())
    }

    fn lookup_slot(&self, mut scope: Option<ParameterScopeId>, name: &str) -> Option<usize> {
        let key = canonical_lookup(name);
        while let Some(id) = scope {
            let current = self.scopes.get(id.0)?;
            if let Some(&local) = current.baseline.active.get(key.as_ref()) {
                return Some(current.start + local);
            }
            scope = current.parent;
        }
        None
    }

    /// Resolve a name to its stable slot once, for repeated observation.
    #[must_use]
    pub fn slot(&self, scope: ParameterScopeId, name: &str) -> Option<ParameterSlot> {
        let node = &self.nodes[self.lookup_slot(Some(scope), name)?];
        Some(ParameterSlot {
            scope: node.scope,
            local: node.local,
        })
    }

    fn node_index(&self, slot: ParameterSlot) -> Option<usize> {
        let scope = self.scopes.get(slot.scope.0)?;
        scope.baseline.entries().get(slot.local)?;
        Some(scope.start + slot.local)
    }

    /// Compile a scope using the scalar resolver's existing graph, reference
    /// scanner and baseline evaluator. Record reverse edges across lexical
    /// parent scopes as well as within this scope.
    ///
    /// # Errors
    /// Invalid parents, duplicate/unknown/nonlocal tunables, or any ordinary
    /// parameter resolution error. A failed compilation leaves the plan unchanged.
    pub fn add_scope(
        &mut self,
        spec: ParameterScopeSpec<'_>,
        budget: &mut EvalBudget,
    ) -> SpiceResult<ParameterScopeId> {
        let parent = spec
            .parent
            .map(|id| {
                self.scopes
                    .get(id.0)
                    .map(|s| Arc::clone(&s.baseline))
                    .ok_or_else(|| SpiceError::circuit("unknown parameter plan parent"))
            })
            .transpose()?;
        let policy = if spec.instance_overrides {
            Bindings::Win
        } else {
            Bindings::Reject
        };
        let definitions = collect_definitions(spec.bindings, spec.cards, budget.limits(), policy)?;
        let start = self.nodes.len();
        let scope = ParameterScopeId(self.scopes.len());
        let inputs = select_tunables(&definitions, spec.tunables)?;
        let selected: HashSet<_> = inputs.iter().map(|(local, _)| *local).collect();
        let ScopeSchedule {
            sequence,
            references,
        } = self.compile_schedule(&spec, &definitions, &selected, budget)?;
        let mut nodes: Vec<_> = definitions
            .entries
            .iter()
            .zip(references)
            .enumerate()
            .map(|(local, (entry, references))| {
                let expression = match entry.state {
                    DefinitionState::Pending(expr) if !selected.contains(&local) => {
                        Some(expr.clone())
                    }
                    _ => None,
                };
                PlannedNode {
                    scope,
                    local,
                    expression,
                    references,
                    rank: None,
                    dependents: Vec::new(),
                    baseline: None,
                }
            })
            .collect();
        for (rank, &local) in sequence.iter().enumerate() {
            if nodes[local].expression.is_some() {
                nodes[local].rank = Some(start + rank);
            }
        }
        let baseline = Arc::new(ParamScope::resolve_definitions(
            parent,
            spec.functions,
            definitions,
            sequence,
            budget,
        )?);
        for (node, entry) in nodes.iter_mut().zip(baseline.entries()) {
            if let ParamState::Resolved(value) = entry.state {
                node.baseline = Some(value);
            }
        }
        let inputs = inputs
            .into_iter()
            .map(|(local, name)| {
                let default = nodes[local]
                    .baseline
                    .ok_or_else(|| SpiceError::circuit("active tunable has no baseline value"))?;
                Ok(ParameterInput {
                    slot: ParameterSlot { scope, local },
                    name,
                    default,
                })
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        self.nodes.extend(nodes);
        let mut dependencies = Vec::new();
        for dependent in start..self.nodes.len() {
            // A reference map is keyed by canonical name. A slot can only be
            // reached through its one name in this lexical environment.
            dependencies.extend(self.nodes[dependent].references.values().copied());
            while let Some(dependency) = dependencies.pop() {
                self.nodes[dependency].dependents.push(dependent);
            }
        }
        self.inputs.extend(inputs);
        self.scopes.push(PlannedScope {
            baseline,
            parent: spec.parent,
            start,
        });
        Ok(scope)
    }

    fn compile_schedule(
        &self,
        spec: &ParameterScopeSpec<'_>,
        definitions: &ScopeDefinitions<'_>,
        selected: &HashSet<usize>,
        budget: &mut EvalBudget,
    ) -> SpiceResult<ScopeSchedule> {
        let start = self.nodes.len();
        let mut references = vec![HashMap::new(); definitions.entries.len()];
        let mut missing_dependency = false;
        let sequence = definitions.schedule(
            spec.parent.and_then(|id| self.baseline(id)),
            spec.functions.as_deref(),
            budget,
            |local, reference| {
                if selected.contains(&local) {
                    return;
                }
                let (name, dependency) = match reference {
                    DependencyReference::Local(index) => (
                        definitions.entries[index].name.as_str(),
                        Some(start + index),
                    ),
                    DependencyReference::Parent(name) => {
                        (name, self.lookup_slot(spec.parent, name))
                    }
                };
                if let Some(index) = dependency {
                    references[local].entry(name.to_owned()).or_insert(index);
                } else {
                    missing_dependency = true;
                }
            },
        )?;
        if missing_dependency {
            return Err(SpiceError::circuit(
                "parameter plan dependency missing from parent",
            ));
        }
        Ok(ScopeSchedule {
            sequence,
            references,
        })
    }

    /// Fresh independent session at the validated baseline.
    #[must_use]
    pub fn session(&self) -> CandidateParameters<'_> {
        CandidateParameters {
            plan: self,
            values: self.nodes.iter().map(|node| node.baseline).collect(),
            staged: vec![None; self.nodes.len()],
            changed: Vec::new(),
            ready: BinaryHeap::new(),
            queued: vec![false; self.nodes.len()],
        }
    }
}

fn select_tunables(
    definitions: &ScopeDefinitions<'_>,
    tunables: &[&str],
) -> SpiceResult<Vec<(usize, String)>> {
    let mut selected = HashSet::new();
    let mut inputs = Vec::with_capacity(tunables.len());
    for name in tunables {
        let key = canonical_lookup(name);
        let Some(&local) = definitions.active.get(key.as_ref()) else {
            return Err(SpiceError::circuit(format!(
                "unknown local tunable '{name}'"
            )));
        };
        if !selected.insert(local) {
            return Err(SpiceError::circuit(format!("duplicate tunable '{name}'")));
        }
        inputs.push((local, key.into_owned()));
    }
    Ok(inputs)
}

/// Numeric state owned by one worker or lane. Candidate updates are atomic:
/// errors retain the previous successful candidate and discard all staged values.
pub struct CandidateParameters<'a> {
    plan: &'a ParameterPlan,
    values: Vec<Option<Real>>,
    staged: Vec<Option<Real>>,
    changed: Vec<usize>,
    ready: BinaryHeap<Reverse<(usize, usize)>>,
    queued: Vec<bool>,
}

impl CandidateParameters<'_> {
    /// Lookup with normal case-insensitive lexical shadowing.
    #[must_use]
    pub fn get(&self, scope: ParameterScopeId, name: &str) -> Option<Real> {
        self.plan
            .lookup_slot(Some(scope), name)
            .and_then(|i| self.values[i])
    }

    /// Read a previously resolved slot without name lookup.
    #[must_use]
    pub fn get_slot(&self, slot: ParameterSlot) -> Option<Real> {
        self.plan
            .node_index(slot)
            .and_then(|index| self.values[index])
    }

    /// Evaluate a site expression against the selected scope and current candidate.
    ///
    /// # Errors
    /// Invalid scope or ordinary expression evaluation errors.
    pub fn evaluate(
        &self,
        scope: ParameterScopeId,
        expression: &ParameterExpression,
        budget: &mut EvalBudget,
    ) -> SpiceResult<Real> {
        let baseline = self
            .plan
            .baseline(scope)
            .ok_or_else(|| SpiceError::circuit("unknown parameter plan scope"))?;
        evaluate_root(
            expression,
            &|name| self.get(scope, name),
            baseline.functions.as_deref(),
            budget,
        )
        .map_err(Failure::into_error)
    }

    /// Materialize a resolved lexical scope from the current candidate
    /// without re-running parameter definitions. Parent chains carry the
    /// same candidate values and function environments. Scope metadata and
    /// superseded declarations are preserved.
    ///
    /// # Errors
    /// The requested scope does not belong to this plan.
    pub fn resolved_scope(&self, scope: ParameterScopeId) -> SpiceResult<Arc<ParamScope>> {
        let mut chain = Vec::new();
        let mut current = Some(scope);
        while let Some(id) = current {
            let node = self
                .plan
                .scopes
                .get(id.0)
                .ok_or_else(|| SpiceError::circuit("unknown parameter plan scope"))?;
            chain.push(node);
            current = node.parent;
        }
        let mut parent = None;
        for node in chain.into_iter().rev() {
            let mut resolved = (*node.baseline).clone();
            resolved.parent = parent;
            for (entry, value) in resolved.entries.iter_mut().zip(&self.values[node.start..]) {
                if let Some(value) = value {
                    entry.state = ParamState::Resolved(*value);
                }
            }
            parent = Some(Arc::new(resolved));
        }
        parent.ok_or_else(|| SpiceError::circuit("empty parameter scope chain"))
    }

    /// Set an arbitrary full candidate vector in plan input order. Returns
    /// the count of derived definitions actually evaluated. Only dependents
    /// of changed values are scheduled, in original topological order.
    ///
    /// # Errors
    /// Wrong vector length, nonfinite input, or expression/budget failure.
    pub fn update(&mut self, candidate: &[Real], budget: &mut EvalBudget) -> SpiceResult<usize> {
        if candidate.len() != self.plan.inputs.len() {
            return Err(SpiceError::circuit(format!(
                "expected {} parameter inputs, found {}",
                self.plan.inputs.len(),
                candidate.len()
            )));
        }
        for (input, value) in self.plan.inputs.iter().zip(candidate) {
            if !value.is_finite() {
                let entry =
                    &self.plan.scopes[input.slot.scope.0].baseline.entries[input.slot.local];
                return Err(SpiceError::parse(
                    entry.location.clone(),
                    format!("binding '{}' is not finite", input.name),
                ));
            }
        }
        let result = self.stage_candidate(candidate, budget);
        for node in self.changed.drain(..) {
            let staged = self.staged[node].take();
            if result.is_ok() {
                self.values[node] = staged;
            }
        }
        for Reverse((_, index)) in self.ready.drain() {
            self.queued[index] = false;
        }
        result
    }

    fn stage_candidate(
        &mut self,
        candidate: &[Real],
        budget: &mut EvalBudget,
    ) -> SpiceResult<usize> {
        for (input, &value) in self.plan.inputs.iter().zip(candidate) {
            let index = self.plan.scopes[input.slot.scope.0].start + input.slot.local;
            self.stage(index, value);
        }
        let mut evaluated = 0;
        while let Some(Reverse((_, index))) = self.ready.pop() {
            self.queued[index] = false;
            let node = &self.plan.nodes[index];
            let scope = &self.plan.scopes[node.scope.0];
            let entry = &scope.baseline.entries[node.local];
            let expression = node
                .expression
                .as_ref()
                .ok_or_else(|| SpiceError::circuit("scheduled parameter has no expression"))?;
            let lookup = |name: &str| {
                node.references
                    .get(name)
                    .and_then(|&slot| self.staged[slot].or(self.values[slot]))
            };
            let value = evaluate_definition(
                expression,
                &lookup,
                scope.baseline.functions.as_deref(),
                budget,
                &entry.name,
                &entry.location,
            )?;
            evaluated += 1;
            self.stage(index, value);
        }
        Ok(evaluated)
    }

    fn stage(&mut self, index: usize, value: Real) {
        if self.values[index].is_some_and(|old| old.to_bits() == value.to_bits()) {
            return;
        }
        self.staged[index] = Some(value);
        self.changed.push(index);
        for &dependent in &self.plan.nodes[index].dependents {
            if !self.queued[dependent]
                && let Some(rank) = self.plan.nodes[dependent].rank
            {
                self.queued[dependent] = true;
                self.ready.push(Reverse((rank, dependent)));
            }
        }
    }
}
