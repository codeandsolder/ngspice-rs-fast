//! Immutable, reusable optimizer parameter bindings and transactional candidates.
//!
//! The scalar .param DAG is compiled once by `ParameterPlan` (#117); site
//! dependencies are resolved by that same evaluator, including .func captures.
//! The existing finalized `InstanceTarget` (#134) supplies the only numeric
//! device setter binding. All changes not proven safe for an instance overlay
//! go through the ordinary validated scalar elaborator; in particular a
//! candidate never silently reuses an obsolete matrix or state layout.

use std::sync::Arc;

use crate::analysis::RunConfig;
use crate::devices::circuit::Circuit;
use crate::devices::models::ModelContext;
use crate::devices::sweep::{InstanceTarget, MAX_INSTANCE_OVERRIDES, MAX_RESISTOR_OVERRIDES};
use crate::netlist::ast::{Netlist, NodeHintValue, ParameterKind};
use crate::netlist::elaborate::{self, SiteKind};
use crate::netlist::eval::{
    BoundExpression, CandidateParameters, EvalBudget, FunctionScope, ParameterPlan,
    ParameterScopeId, ParameterScopeSpec,
};
use crate::netlist::expr::ParameterExpression;
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};

/// What candidate preparation did to the finalized circuit.
///
/// A rebuilt candidate needs fresh solver and device/analysis identities and
/// observation bindings, even if its new node names look identical.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateTopology {
    /// Reused the original validated circuit and its topology identity.
    Reused,
    /// Re-elaborated the deck; old typed targets and state must not be reused.
    Rebuilt,
}

#[derive(Debug, Clone, Copy)]
enum NumericRoute {
    Resistor(InstanceTarget),
    Instance(InstanceTarget),
    Rebuild,
}

#[derive(Debug, Clone, Copy)]
enum RouteSpec {
    Resistor(usize),
    Instance { device: usize, parameter: usize },
    Rebuild,
}

/// One immutable source value site compiled against the parameter slot DAG.
/// Source identity and provenance remain stable across candidate sessions.
pub struct CandidateSite {
    kind: SiteKind,
    expression: ParameterExpression,
    inputs: BoundExpression,
    unrounded: Real,
    baseline: Real,
    route: RouteSpec,
}

impl CandidateSite {
    /// Original location in the input netlist, including the original
    /// device/model, analysis argument or node-hint index.
    #[must_use]
    pub const fn kind(&self) -> SiteKind {
        self.kind
    }

    /// Source expression location, used for scalar-equivalent diagnostics.
    #[must_use]
    pub const fn location(&self) -> &SourceLoc {
        &self.expression.span.start
    }

    /// Original validated, numparam-rounded numeric value.
    #[must_use]
    pub const fn baseline(&self) -> Real {
        self.baseline
    }

    /// Independent candidate coordinates that could change this site.
    #[must_use]
    pub fn inputs(&self) -> &[usize] {
        self.inputs.inputs()
    }
}

/// Shareable immutable preparation of the original source and validated baseline.
///
/// Any number of independent sessions may borrow it. Inputs are top-level `.param` coordinates; nested instance formal scopes
/// are evaluated by the ordinary subcircuit elaborator when affected.
pub struct CircuitParameterPlan {
    source: Netlist,
    parameters: ParameterPlan,
    root: ParameterScopeId,
    config: RunConfig,
    sites: Vec<CandidateSite>,
    requires_rebuild: bool,
}

impl CircuitParameterPlan {
    /// Compile a circuit and bind independent top-level `.param` inputs.
    /// The untouched scalar deck is fully elaborated and validated first.
    ///
    /// # Errors
    /// Ordinary netlist, parameter, option, model or circuit validation; an
    /// unknown/duplicate independent coordinate or dependency scan failure.
    pub fn new(source: Netlist, tunables: &[&str]) -> SpiceResult<Self> {
        let mut budget = EvalBudget::default();
        let functions = FunctionScope::for_netlist(&source, &budget)?;
        let mut parameters = ParameterPlan::new();
        let root = parameters.add_scope(
            ParameterScopeSpec {
                parent: None,
                functions: Some(functions),
                bindings: &[],
                cards: &source.params,
                instance_overrides: false,
                tunables,
            },
            &mut budget,
        )?;
        let scope = Arc::new(
            parameters
                .baseline(root)
                .ok_or_else(|| SpiceError::circuit("missing parameter plan root"))?
                .clone(),
        );
        let elaborated = elaborate::literalize_with(&source, scope, &mut budget)?;
        let config = RunConfig::from_netlist(&source)?;
        let baseline = config.circuit(&source)?;
        let mut sites = Vec::new();
        for site in &elaborated.sites {
            let expression = expression_at(&source, site.kind).ok_or_else(|| {
                SpiceError::circuit("literalization site lost its original expression")
            })?;
            let inputs = parameters.bind_expression(root, expression, &mut budget)?;
            if inputs.inputs().is_empty() {
                continue;
            }
            let route = if let SiteKind::DeviceParameter { device, parameter } = site.kind {
                bind_numeric_site(&source, &baseline, device, parameter)
            } else {
                RouteSpec::Rebuild
            };
            sites.push(CandidateSite {
                kind: site.kind,
                expression: expression.clone(),
                inputs,
                unrounded: site.value,
                baseline: site.read_back,
                route,
            });
        }
        // Other consumers of root `.param` are resolved by full elaboration:
        // options, nested formal scopes, behavioural expression rewriting,
        // and output expressions can change independently of scalar sites.
        let requires_rebuild = !source.subcircuits.is_empty()
            || !source.options.is_empty()
            || source.cards.iter().any(|card| {
                matches!(
                    card.kind,
                    crate::netlist::ast::ScopedCardKind::Measure(_)
                        | crate::netlist::ast::ScopedCardKind::Fourier(_)
                        | crate::netlist::ast::ScopedCardKind::Save(_)
                        | crate::netlist::ast::ScopedCardKind::Print(_)
                )
            })
            || source.devices.iter().any(|device| {
                device
                    .parameters
                    .iter()
                    .any(|parameter| matches!(parameter.kind, ParameterKind::Behavioural(_)))
            });
        Ok(Self {
            source,
            parameters,
            root,
            config,
            sites,
            requires_rebuild,
        })
    }

    /// Coordinate schema in original caller order.
    #[must_use]
    pub fn inputs(&self) -> &[crate::netlist::eval::ParameterInput] {
        self.parameters.inputs()
    }

    /// Immutable parameter-dependent numeric sites in original deck order.
    /// Sites that cannot change with the selected tunables are not retained.
    #[must_use]
    pub fn sites(&self) -> &[CandidateSite] {
        &self.sites
    }

    /// A fresh independent candidate session with its own mutable circuit
    /// and its own topology-bound instance setters. This prevents cross-lane
    /// solver state leaks and never copies one session's device bindings.
    ///
    /// # Errors
    /// Candidate baseline elaboration failure or stale typed setter.
    pub fn session(&self) -> SpiceResult<CircuitCandidate<'_>> {
        let scope = Arc::new(
            self.parameters
                .baseline(self.root)
                .ok_or_else(|| SpiceError::circuit("missing parameter plan root"))?
                .clone(),
        );
        let context = self.config.context().model_context();
        let baseline = Circuit::from_netlist_with_scoped_limits(
            &self.source,
            &context,
            crate::devices::MutualLimits::default(),
            crate::devices::ElaborationLimits::default(),
            Some(scope),
        )?;
        let routes = self
            .sites
            .iter()
            .map(|site| site.route.bind(&self.source, &baseline))
            .collect();
        Ok(CircuitCandidate {
            plan: self,
            baseline,
            routes,
            parameters: self.parameters.session(),
            values: self.inputs().iter().map(|input| input.default).collect(),
            site_values: self.sites.iter().map(|site| site.baseline).collect(),
            raw_site_values: self.sites.iter().map(|site| site.unrounded).collect(),
            last_site_evaluations: 0,
            last_site_roundings: 0,
            prepared: None,
        })
    }
}

fn expression_at(source: &Netlist, kind: SiteKind) -> Option<&ParameterExpression> {
    match kind {
        SiteKind::DeviceParameter { device, parameter } => {
            match &source.devices.get(device)?.parameters.get(parameter)?.kind {
                ParameterKind::Expression(expression) => Some(expression),
                _ => None,
            }
        }
        SiteKind::ModelParameter { model, parameter } => {
            match &source.models.get(model)?.parameters.get(parameter)?.kind {
                ParameterKind::Expression(expression) => Some(expression),
                _ => None,
            }
        }
        SiteKind::AnalysisArgument { analysis, argument } => source
            .analyses
            .get(analysis)?
            .expressions
            .iter()
            .find(|expression| expression.index == argument)
            .map(|value| &value.expression),
        SiteKind::InitialCondition { card, entry } => {
            match &source
                .initial_conditions
                .get(card)?
                .entries
                .get(entry)?
                .value
            {
                NodeHintValue::Expression(expression) => Some(expression),
                NodeHintValue::Literal { .. } => None,
            }
        }
        SiteKind::Nodeset { card, entry } => {
            match &source.nodesets.get(card)?.entries.get(entry)?.value {
                NodeHintValue::Expression(expression) => Some(expression),
                NodeHintValue::Literal { .. } => None,
            }
        }
    }
}

fn bind_numeric_site(
    source: &Netlist,
    circuit: &Circuit,
    device: usize,
    parameter: usize,
) -> RouteSpec {
    let Some(card) = source.devices.get(device) else {
        return RouteSpec::Rebuild;
    };
    let Some(setter) = card.parameters.get(parameter) else {
        return RouteSpec::Rebuild;
    };
    // A repeated setter may be superseded by a later literal, or precede
    // a model selection. The original ordered recipe must then be replayed.
    if card.parameters[parameter + 1..]
        .iter()
        .any(|next| next.name == setter.name)
    {
        return RouteSpec::Rebuild;
    }
    match (card.designator, setter.name.as_str()) {
        ('r', "resistance") if circuit.bind_resistor_target(&card.name).is_ok() => {
            RouteSpec::Resistor(device)
        }
        ('v' | 'i', "dc") if circuit.bind_source_target(&card.name).is_ok() => {
            RouteSpec::Instance { device, parameter }
        }
        _ if card.model.is_none()
            && circuit
                .bind_instance_target(&card.name, &setter.name)
                .is_ok() =>
        {
            RouteSpec::Instance { device, parameter }
        }
        // Model parameter binning and geometry are resolved by the factory.
        _ => RouteSpec::Rebuild,
    }
}

impl RouteSpec {
    fn bind(self, source: &Netlist, circuit: &Circuit) -> NumericRoute {
        match self {
            Self::Resistor(device) => source
                .devices
                .get(device)
                .and_then(|card| circuit.bind_resistor_target(&card.name).ok())
                .map_or(NumericRoute::Rebuild, NumericRoute::Resistor),
            Self::Instance { device, parameter } => {
                let Some(card) = source.devices.get(device) else {
                    return NumericRoute::Rebuild;
                };
                let Some(setter) = card.parameters.get(parameter) else {
                    return NumericRoute::Rebuild;
                };
                let target = if matches!((card.designator, setter.name.as_str()), ('v' | 'i', "dc"))
                {
                    circuit.bind_source_target(&card.name)
                } else {
                    circuit.bind_instance_target(&card.name, &setter.name)
                };
                target.map_or(NumericRoute::Rebuild, NumericRoute::Instance)
            }
            Self::Rebuild => NumericRoute::Rebuild,
        }
    }
}

struct PreparedCandidate {
    context: ModelContext,
    config: RunConfig,
    circuit: Option<Circuit>,
}

/// Candidate-local numeric site evaluation transaction. None of the
/// recorded values reaches the live session until validation succeeds.
#[derive(Default)]
struct PendingSiteEdits {
    updated: Vec<(usize, Real, Real)>,
    evaluated: usize,
    rounded: usize,
}

/// Independent candidate state with transactional updates.
///
/// A failed parameter evaluation, device setter or re-elaboration retains the prior
/// validated parameter vector, model context and circuit simultaneously.
pub struct CircuitCandidate<'a> {
    plan: &'a CircuitParameterPlan,
    baseline: Circuit,
    routes: Vec<NumericRoute>,
    parameters: CandidateParameters<'a>,
    values: Vec<Real>,
    /// Last committed read-back value of each prepared source site.
    site_values: Vec<Real>,
    /// Before numparam decimal formatting; changed raw bits alone trigger it.
    raw_site_values: Vec<Real>,
    last_site_evaluations: usize,
    last_site_roundings: usize,
    prepared: Option<PreparedCandidate>,
}

impl CircuitCandidate<'_> {
    /// The baseline circuit or the current candidate's own finalized circuit.
    #[must_use]
    pub fn circuit(&self) -> &Circuit {
        self.prepared
            .as_ref()
            .and_then(|prepared| prepared.circuit.as_ref())
            .unwrap_or(&self.baseline)
    }

    /// Mutable circuit for one analysis driver. Candidate transitions clear
    /// previously carried analysis state without changing finalized numbering.
    pub fn circuit_mut(&mut self) -> &mut Circuit {
        self.prepared
            .as_mut()
            .and_then(|prepared| prepared.circuit.as_mut())
            .unwrap_or(&mut self.baseline)
    }

    /// Validated configuration of the current candidate.
    #[must_use]
    pub fn config(&self) -> &RunConfig {
        self.prepared
            .as_ref()
            .map_or(&self.plan.config, |prepared| &prepared.config)
    }

    /// Model preparation, including all selected typed numeric replacements.
    #[must_use]
    pub fn model_context(&self) -> ModelContext {
        self.prepared.as_ref().map_or_else(
            || self.plan.config.context().model_context(),
            |prepared| prepared.context,
        )
    }

    /// The complete analysis context for this numeric candidate. All normal
    /// analysis drivers obtain their typed device overrides through this
    /// context's existing `model_context()` conversion; passing the baseline
    /// `RunConfig::context()` would omit candidate values.
    #[must_use]
    pub fn analysis_context(&self) -> crate::analysis::AnalysisContext {
        self.config()
            .context()
            .with_candidate_model_overrides(self.model_context())
    }

    /// Whether the original topology identity is still valid.
    #[must_use]
    pub fn topology(&self) -> CandidateTopology {
        if self
            .prepared
            .as_ref()
            .is_some_and(|prepared| prepared.circuit.is_some())
        {
            CandidateTopology::Rebuilt
        } else {
            CandidateTopology::Reused
        }
    }

    /// Current committed typed value at a source site, with numparam
    /// read-back applied once when it last changed. Never creates a deck.
    #[must_use]
    pub fn site_value(&self, site: usize) -> Option<Real> {
        self.site_values.get(site).copied()
    }

    /// Number of source value sites evaluated by the most recent successful
    /// candidate transition. A repeated identical candidate evaluates none.
    #[must_use]
    pub const fn last_site_evaluations(&self) -> usize {
        self.last_site_evaluations
    }

    /// Number of source-site numparam decimal formatting and read-back
    /// operations in the most recent successful candidate transition.
    #[must_use]
    pub const fn last_site_roundings(&self) -> usize {
        self.last_site_roundings
    }

    /// Apply a full vector of finite independent coordinates. Returns the
    /// number of derived `.param` expressions recomputed by the shared DAG.
    /// Candidate geometry, context and circuit are published only on success.
    ///
    /// # Errors
    /// Invalid coordinates, expression, model or device physics, or any
    /// validated scalar re-elaboration failure. Errors do not mutate state.
    pub fn update(&mut self, candidate: &[Real], budget: &mut EvalBudget) -> SpiceResult<usize> {
        if candidate.len() == self.values.len()
            && candidate
                .iter()
                .zip(&self.values)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        {
            self.last_site_evaluations = 0;
            self.last_site_roundings = 0;
            return Ok(0);
        }
        let mut numeric = self.parameters.clone();
        let evaluated = numeric.update(candidate, budget)?;
        let (prepared, edits) = self.prepare(&numeric, budget)?;
        self.parameters = numeric;
        for (site, raw, value) in edits.updated {
            self.raw_site_values[site] = raw;
            self.site_values[site] = value;
        }
        self.last_site_evaluations = edits.evaluated;
        self.last_site_roundings = edits.rounded;
        self.baseline.clear_candidate_analysis_state();
        self.prepared = Some(prepared);
        self.values.clear();
        self.values.extend_from_slice(candidate);
        Ok(evaluated)
    }

    fn evaluate_site(
        &self,
        index: usize,
        site: &CandidateSite,
        parameters: &CandidateParameters<'_>,
        budget: &mut EvalBudget,
        edits: &mut PendingSiteEdits,
    ) -> SpiceResult<Real> {
        if !site.inputs.changed_between(&self.parameters, parameters) {
            return Ok(self.site_values[index]);
        }
        let raw =
            parameters.evaluate_bound(self.plan.root, &site.expression, &site.inputs, budget)?;
        edits.evaluated += 1;
        // A changed direct reference can produce the identical result
        // (e.g. x-x). Round only when the raw binary64 result changes.
        if raw.to_bits() == self.raw_site_values[index].to_bits() {
            return Ok(self.site_values[index]);
        }
        let text = elaborate::format_literal(raw);
        let rounded = crate::primitives::parse_spice_number(&text)
            .ok_or_else(|| SpiceError::circuit("candidate scalar cannot be read back"))?;
        edits.rounded += 1;
        edits.updated.push((index, raw, rounded));
        Ok(rounded)
    }

    fn prepare(
        &self,
        parameters: &CandidateParameters<'_>,
        budget: &mut EvalBudget,
    ) -> SpiceResult<(PreparedCandidate, PendingSiteEdits)> {
        let mut context = self.plan.config.context().model_context();
        let mut structural = self.plan.requires_rebuild;
        let mut resistor_count = 0_usize;
        let mut instance_count = 0_usize;
        let mut edits = PendingSiteEdits::default();
        for (index, (site, &route)) in self.plan.sites.iter().zip(&self.routes).enumerate() {
            let value = self.evaluate_site(index, site, parameters, budget, &mut edits)?;
            if value.to_bits() == site.baseline.to_bits() {
                continue;
            }
            match route {
                NumericRoute::Resistor(target) => {
                    resistor_count += 1;
                    if resistor_count > MAX_RESISTOR_OVERRIDES {
                        structural = true;
                    } else if !structural {
                        if let Ok(next) = target
                            .at_resistance(value)
                            .and_then(|replacement| context.with_resistor_override(replacement))
                        {
                            context = next;
                        } else {
                            structural = true;
                        }
                    }
                }
                NumericRoute::Instance(target) => {
                    instance_count += 1;
                    if instance_count > MAX_INSTANCE_OVERRIDES {
                        structural = true;
                    } else if !structural {
                        if let Ok(next) = target
                            .at(value)
                            .and_then(|replacement| context.with_instance_override(replacement))
                        {
                            context = next;
                        } else {
                            structural = true;
                        }
                    }
                }
                NumericRoute::Rebuild => structural = true,
            }
        }
        if !structural && self.baseline.validate_candidate_context(&context).is_ok() {
            // The circuit/configuration shape is unchanged, but independent
            // parameter values must still be visible to downstream objectives.
            let scope = parameters.resolved_scope(self.plan.root)?;
            return Ok((
                PreparedCandidate {
                    context,
                    config: self.plan.config.clone().with_parameter_scope(scope),
                    circuit: None,
                },
                edits,
            ));
        }
        // Every dimension, including non-instance model and topology-changing
        // coordinates, is handled by this general scalar-equivalent path.
        // Failed structural validation cannot escape through the cached matrix.
        let scope = parameters.resolved_scope(self.plan.root)?;
        let config = RunConfig::from_netlist_with_scope(&self.plan.source, Arc::clone(&scope))?;
        let context = config.context().model_context();
        let circuit = Circuit::from_netlist_with_scoped_limits(
            &self.plan.source,
            &context,
            crate::devices::MutualLimits::default(),
            crate::devices::ElaborationLimits::default(),
            Some(scope),
        )?;
        Ok((
            PreparedCandidate {
                context,
                config,
                circuit: Some(circuit),
            },
            edits,
        ))
    }
}
