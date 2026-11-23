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
use crate::analysis::sens::{PhysicalParameter, conditioned_derivative};
use crate::devices::circuit::Circuit;
use crate::devices::models::ModelContext;
use crate::devices::sweep::{InstanceTarget, MAX_INSTANCE_OVERRIDES, MAX_RESISTOR_OVERRIDES};
use crate::netlist::ast::{Netlist, NodeHintValue, ParameterKind};
use crate::netlist::elaborate::{self, SiteKind};
use crate::netlist::eval::{
    BoundExpression, CandidateParameters, EvalBudget, EvalLimits, FunctionScope, ParameterPlan,
    ParameterScopeId, ParameterScopeSpec,
};
use crate::netlist::expr::ParameterExpression;
use crate::primitives::{Complex, Real, SourceLoc, SpiceError, SpiceResult};

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

/// Immutable numeric site from the original netlist, indexed by the
/// shared parameter plan.
pub struct CandidateSite {
    /// Original site category and position.
    pub kind: SiteKind,
    /// Source expression location.
    pub location: SourceLoc,
    /// Baseline value after numparam decimal read-back.
    pub baseline: Real,
    raw_baseline: Real,
    expression: ParameterExpression,
    inputs: BoundExpression,
    route: RouteSpec,
}

impl CandidateSite {
    /// Independent input coordinates which may affect this site.
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
    /// Input coordinates affecting numeric options rather than device stamps.
    option_dependencies: Vec<usize>,
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
                location: site.location.clone(),
                expression: expression.clone(),
                inputs,
                baseline: site.read_back,
                raw_baseline: site.value,
                route,
            });
        }
        let mut option_dependencies = Vec::new();
        for card in &source.options {
            for setting in &card.settings {
                if let Some(expression) = &setting.expression {
                    option_dependencies.extend(
                        parameters
                            .bind_expression(root, expression, &mut budget)?
                            .inputs(),
                    );
                }
            }
        }
        option_dependencies.sort_unstable();
        option_dependencies.dedup();
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
            option_dependencies,
            requires_rebuild,
        })
    }

    /// Coordinate schema in original caller order.
    #[must_use]
    pub fn inputs(&self) -> &[crate::netlist::eval::ParameterInput] {
        self.parameters.inputs()
    }

    /// Parameter-dependent site schema and original source provenance.
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
            site_raw: self.sites.iter().map(|site| site.raw_baseline).collect(),
            site_values: self.sites.iter().map(|site| site.baseline).collect(),
            site_updates: Vec::new(),
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

struct SiteUpdate {
    index: usize,
    raw: Real,
    value: Real,
}

/// One differentiable parameter site, retaining the typed device identity
/// and its scalar chain-rule coefficient with respect to an input.
pub(super) struct GradientSite {
    pub parameter: PhysicalParameter,
    pub scale: Real,
}

/// Whether a candidate root input cannot have a valid scalar chain rule.
#[derive(Debug, Clone)]
pub(super) enum GradientSiteFailure {
    Unsupported(String),
    NonSmooth(String),
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
    site_raw: Vec<Real>,
    site_values: Vec<Real>,
    site_updates: Vec<SiteUpdate>,
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

    /// Build a one-off physically re-elaborated circuit for a sensitivity
    /// walk. A typed candidate overlay changes the *primal* stamps but device
    /// sensitivity records still describe the original device literals;
    /// evaluating those records against an overlaid baseline would produce
    /// an inconsistent Jacobian and a false gradient. Re-elaborate exactly
    /// once per requested direct objective, never per parameter.
    pub(super) fn physical_sensitivity_circuit(&self) -> SpiceResult<Circuit> {
        let scope = self.parameters.resolved_scope(self.plan.root)?;
        let context = self.config().context().model_context();
        Circuit::from_netlist_with_scoped_limits(
            &self.plan.source,
            &context,
            crate::devices::MutualLimits::default(),
            crate::devices::ElaborationLimits::default(),
            Some(scope),
        )
    }

    /// Stable name of one independent root coordinate in plan order.
    pub(super) fn parameter_input_name(&self, index: usize) -> &str {
        &self.plan.inputs()[index].name
    }

    /// Differentiate the compiled scalar parameter DAG without solving a
    /// circuit for each coordinate. The physical derivative driver performs
    /// one fresh elaboration, even if the optimized candidate's limited
    /// numeric overlay capacity forced a structural rebuild. Unsupported
    /// model/analysis sites invalidate only their dependent inputs.
    pub(super) fn gradient_sites(
        &self,
        limits: EvalLimits,
    ) -> Vec<Result<Vec<GradientSite>, GradientSiteFailure>> {
        let mut output: Vec<Result<Vec<GradientSite>, GradientSiteFailure>> =
            (0..self.values.len()).map(|_| Ok(Vec::new())).collect();
        // Numeric options can change the solver's convergence regime.
        // Unknown nested or behavioural expressions have no validated
        // differentiable-site binding, though black-box evaluation still works.
        if !self.plan.source.subcircuits.is_empty()
            || self.plan.source.devices.iter().any(|device| {
                device
                    .parameters
                    .iter()
                    .any(|parameter| matches!(parameter.kind, ParameterKind::Behavioural(_)))
            })
        {
            for result in &mut output {
                *result = Err(GradientSiteFailure::Unsupported(
                    "subcircuit or behavioural parameter dependency lacks a validated derivative binding"
                        .to_owned(),
                ));
            }
            return output;
        }
        for &input in &self.plan.option_dependencies {
            output[input] = Err(GradientSiteFailure::Unsupported(
                "input affects numerical analysis options".to_owned(),
            ));
        }
        for (site_index, site) in self.plan.sites.iter().enumerate() {
            for &input in site.inputs.inputs() {
                if output[input].is_err() {
                    continue;
                }
                let Some(physical_parameter) = (match (site.kind, site.route) {
                    (
                        SiteKind::DeviceParameter { device, parameter },
                        RouteSpec::Resistor(_) | RouteSpec::Instance { .. },
                    ) => self.plan.source.devices.get(device).and_then(|card| {
                        card.parameters.get(parameter).map(|parameter| {
                            PhysicalParameter::Instance {
                                instance: card.name.to_ascii_lowercase(),
                                keyword: parameter.name.clone(),
                            }
                        })
                    }),
                    (SiteKind::ModelParameter { model, parameter }, _) => {
                        self.plan.source.models.get(model).and_then(|card| {
                            card.parameters.get(parameter).map(|parameter| {
                                PhysicalParameter::Model {
                                    model: card.name.to_ascii_lowercase(),
                                    keyword: parameter.name.clone(),
                                }
                            })
                        })
                    }
                    _ => None,
                }) else {
                    output[input] = Err(GradientSiteFailure::Unsupported(format!(
                        "input affects a structural, model or unbound site at {}",
                        site.location
                    )));
                    continue;
                };
                let mut candidate_values = self.values.clone();
                let base = self.site_values[site_index];
                let derivative = conditioned_derivative(self.values[input], |point| {
                    candidate_values[input] = point;
                    let mut scratch = self.parameters.clone();
                    let mut budget = EvalBudget::new(limits);
                    if scratch.update(&candidate_values, &mut budget).is_err() {
                        return Ok(None);
                    }
                    let raw =
                        match scratch.evaluate_bound(&site.expression, &site.inputs, &mut budget) {
                            Ok(value) if value.is_finite() => value,
                            _ => return Ok(None),
                        };
                    let literal = elaborate::format_literal(raw);
                    let Some(value) = crate::primitives::parse_spice_number(&literal) else {
                        return Ok(None);
                    };
                    Ok(Some(Complex::real(value - base)))
                });
                match derivative {
                    Ok(Some((scale, _))) if scale.re.is_finite() && scale.im == 0. => {
                        if scale.re != 0. {
                            if let Ok(ref mut links) = output[input] {
                                links.push(GradientSite {
                                    parameter: physical_parameter,
                                    scale: scale.re,
                                });
                            }
                        }
                    }
                    Ok(_) => {
                        output[input] = Err(GradientSiteFailure::NonSmooth(format!(
                            "non-smooth or unresolvable scalar parameter transformation at {}",
                            site.location,
                        )))
                    }
                    Err(error) => {
                        output[input] = Err(GradientSiteFailure::NonSmooth(format!(
                            "scalar parameter derivative failed at {}: {error}",
                            site.location,
                        )))
                    }
                }
            }
        }
        output
    }

    /// Current committed typed site value by stable plan site index.
    #[must_use]
    pub fn site_value(&self, index: usize) -> Option<Real> {
        self.site_values.get(index).copied()
    }

    /// Site expressions evaluated by the last successful update.
    #[must_use]
    pub const fn last_site_evaluations(&self) -> usize {
        self.last_site_evaluations
    }

    /// Decimal materializations by the last successful update.
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
        let (prepared, evaluations, roundings) = self.prepare(&numeric, budget)?;
        for change in self.site_updates.drain(..) {
            self.site_raw[change.index] = change.raw;
            self.site_values[change.index] = change.value;
        }
        self.last_site_evaluations = evaluations;
        self.last_site_roundings = roundings;
        self.parameters = numeric;
        self.baseline.clear_candidate_analysis_state();
        self.prepared = Some(prepared);
        self.values.clear();
        self.values.extend_from_slice(candidate);
        Ok(evaluated)
    }

    fn prepare(
        &mut self,
        parameters: &CandidateParameters<'_>,
        budget: &mut EvalBudget,
    ) -> SpiceResult<(PreparedCandidate, usize, usize)> {
        self.site_updates.clear();
        let mut evaluations = 0;
        let mut roundings = 0;
        let mut context = self.plan.config.context().model_context();
        let mut structural = self.plan.requires_rebuild;
        let mut resistor_count = 0_usize;
        let mut instance_count = 0_usize;
        for (index, (site, &route)) in self.plan.sites.iter().zip(&self.routes).enumerate() {
            let value = if site.inputs.changed_between(&self.parameters, parameters) {
                evaluations += 1;
                let raw = parameters.evaluate_bound(&site.expression, &site.inputs, budget)?;
                if raw.to_bits() == self.site_raw[index].to_bits() {
                    self.site_values[index]
                } else {
                    // Preserve numparam decimal read-back for device setters.
                    let text = elaborate::format_literal(raw);
                    let value = crate::primitives::parse_spice_number(&text).ok_or_else(|| {
                        SpiceError::circuit("candidate scalar cannot be read back")
                    })?;
                    roundings += 1;
                    self.site_updates.push(SiteUpdate { index, raw, value });
                    value
                }
            } else {
                self.site_values[index]
            };
            // Every effective non-baseline site must be applied to each fresh
            // context, even when it was unchanged since the previous candidate.
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
                evaluations,
                roundings,
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
            evaluations,
            roundings,
        ))
    }
}
