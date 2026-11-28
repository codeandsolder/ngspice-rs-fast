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
use crate::devices::circuit::{CandidateNumericOverride, Circuit};
use crate::devices::models::ModelContext;
use crate::devices::sweep::InstanceTarget;
use crate::netlist::ast::Netlist;
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
    #[must_use = "the returned dependency indices determine which coordinate changes invalidate this site"]
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
    option_dependencies: BoundExpression,
    behavioural_dependencies: BoundExpression,
    statistical_model_dependencies: BoundExpression,
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
        let mut budget = EvalBudget::for_netlist(&source)?;
        let functions = FunctionScope::for_netlist(&source, &budget)?;
        let mut parameters = ParameterPlan::new();
        let root = parameters.add_scope(
            ParameterScopeSpec {
                parent: None,
                instance: None,
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
        let elaborated = elaborate::literalize_with(&source, Arc::clone(&scope), &mut budget)?;
        let config = RunConfig::from_netlist(&source)?;
        let baseline = config.circuit(&source)?;
        let mut sites = Vec::new();
        for site in &elaborated.sites {
            let expression = &site.expression;
            let inputs = parameters
                .bind_expression(root, expression, &mut budget)?
                .at(site.statistical);
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
        let option_dependencies = parameters.bind_expressions(
            root,
            source.options.iter().flat_map(|card| {
                card.settings
                    .iter()
                    .filter_map(|setting| setting.expression.as_deref())
            }),
            &mut budget,
        )?;
        // The original AST does not reveal runtime laws hidden in lexical
        // functions, or coefficients inserted by controlled-source lowering.
        // Inspect the actual macro-expanded laws rather than the card syntax.
        let laws = crate::netlist::behavioural::candidate_laws(&source, &scope, &mut budget)?;
        let behavioural_dependencies = parameters.bind_behavioural(root, &laws, &mut budget)?;
        // Nested formal scopes still require independent validation through
        // the full elaborator. Root options need revalidation only when their
        // actual parameter reads differ from the validated baseline.
        let statistical_model_dependencies = parameters.bind_expressions(
            root,
            source
                .models
                .iter()
                .filter(|card| elaborate::model_uses_statistics(card, &scope))
                .flat_map(elaborate::model_expressions),
            &mut budget,
        )?;
        let requires_rebuild = !source.subcircuits.is_empty();
        Ok(Self {
            source,
            parameters,
            root,
            config,
            sites,
            option_dependencies,
            behavioural_dependencies,
            statistical_model_dependencies,
            requires_rebuild,
        })
    }

    pub(crate) const fn source(&self) -> &Netlist {
        &self.source
    }

    /// Coordinate schema in original caller order.
    #[must_use = "the returned schema defines the required coordinate order for candidate updates"]
    pub fn inputs(&self) -> &[crate::netlist::eval::ParameterInput] {
        self.parameters.inputs()
    }

    /// Parameter-dependent site schema and original source provenance.
    #[must_use = "the returned site schema maps dependent values to their original source provenance"]
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
        _ if (card.model.is_none()
            || circuit
                .device(&card.name)
                .is_some_and(crate::devices::Device::model_candidate_setters_are_stable))
            && circuit
                .bind_instance_target(&card.name, &setter.name)
                .is_ok() =>
        {
            RouteSpec::Instance { device, parameter }
        }
        // Uncertified model selection and geometry are resolved by the factory.
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

/// Prefer the allocation-free two-slot context. On overflow, move
/// already validated site identities into a sparse worker-local list.
fn append_numeric_override(
    fixed: &mut ModelContext,
    sparse: &mut Option<Vec<CandidateNumericOverride>>,
    baseline: ModelContext,
    next: CandidateNumericOverride,
    capacity: usize,
) -> SpiceResult<()> {
    if let Some(values) = sparse {
        values.push(next);
        return Ok(());
    }
    let updated = match next {
        CandidateNumericOverride::Resistor(value) => fixed.with_resistor_override(value),
        CandidateNumericOverride::Instance(value) => fixed.with_instance_override(value),
    };
    if let Ok(updated) = updated {
        *fixed = updated;
        return Ok(());
    }
    let mut values = Vec::new();
    values.try_reserve(capacity).map_err(|error| {
        SpiceError::circuit(format!(
            "cannot allocate candidate numeric overlay: {error}"
        ))
    })?;
    values.extend(
        fixed
            .resistor_overrides
            .iter()
            .flatten()
            .copied()
            .map(CandidateNumericOverride::Resistor),
    );
    values.extend(
        fixed
            .instance_overrides
            .iter()
            .flatten()
            .copied()
            .map(CandidateNumericOverride::Instance),
    );
    values.push(next);
    *sparse = Some(values);
    *fixed = baseline;
    Ok(())
}

impl CircuitCandidate<'_> {
    /// The baseline circuit or the current candidate's own finalized circuit.
    #[must_use = "the returned finalized circuit reflects the current candidate rather than necessarily the baseline"]
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
    #[must_use = "the returned run configuration includes the current candidate's validated option replacements"]
    pub fn config(&self) -> &RunConfig {
        self.prepared
            .as_ref()
            .map_or(&self.plan.config, |prepared| &prepared.config)
    }

    /// Model preparation, including all selected typed numeric replacements.
    #[must_use = "the returned context carries the selected typed model replacements needed for device preparation"]
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
    #[must_use = "the returned context combines run configuration and candidate model overrides; a baseline context loses replacements"]
    pub fn analysis_context(&self) -> crate::analysis::AnalysisContext {
        self.config()
            .context()
            .with_candidate_model_overrides(self.model_context())
    }

    /// Whether the original topology identity is still valid.
    #[must_use = "the returned classification determines whether baseline topology identity can be reused"]
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

    /// Differentiate one bound site against one root coordinate using the
    /// same scalar DAG and literal readback used in candidate preparation.
    fn gradient_site(
        &self,
        site_index: usize,
        input: usize,
        limits: EvalLimits,
    ) -> Result<Option<GradientSite>, GradientSiteFailure> {
        let site = &self.plan.sites[site_index];
        let Some(physical_parameter) = (match (site.kind, site.route) {
            (
                SiteKind::DeviceParameter { device, parameter },
                RouteSpec::Resistor(_) | RouteSpec::Instance { .. },
            ) => self.plan.source.devices.get(device).and_then(|card| {
                card.parameters
                    .get(parameter)
                    .map(|parameter| PhysicalParameter::Instance {
                        instance: card.name.to_ascii_lowercase(),
                        keyword: parameter.name.clone(),
                    })
            }),
            (SiteKind::ModelParameter { model, parameter }, _) => {
                self.plan.source.models.get(model).and_then(|card| {
                    card.parameters
                        .get(parameter)
                        .map(|parameter| PhysicalParameter::Model {
                            model: card.name.to_ascii_lowercase(),
                            keyword: parameter.name.clone(),
                        })
                })
            }
            _ => None,
        }) else {
            return Err(GradientSiteFailure::Unsupported(format!(
                "input affects a structural, model or unbound site at {}",
                site.location
            )));
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
            let raw = match scratch.evaluate_bound(&site.expression, &site.inputs, &mut budget) {
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
            Ok(Some((scale, _))) if scale.re.is_finite() && scale.im == 0. => Ok((scale.re != 0.)
                .then_some(GradientSite {
                    parameter: physical_parameter,
                    scale: scale.re,
                })),
            Ok(_) => Err(GradientSiteFailure::NonSmooth(format!(
                "non-smooth or unresolvable scalar parameter transformation at {}",
                site.location,
            ))),
            Err(error) => Err(GradientSiteFailure::NonSmooth(format!(
                "scalar parameter derivative failed at {}: {error}",
                site.location,
            ))),
        }
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
        // Nested scopes do not yet have a validated physical derivative
        // binding. Runtime law coefficients invalidate only the inputs
        // that actually affect their compiled numerical values.
        if !self.plan.source.subcircuits.is_empty() {
            for result in &mut output {
                *result = Err(GradientSiteFailure::Unsupported(
                    "subcircuit parameter dependency lacks a validated derivative binding"
                        .to_owned(),
                ));
            }
            return output;
        }
        for &input in self.plan.behavioural_dependencies.inputs() {
            output[input] = Err(GradientSiteFailure::Unsupported(
                "behavioural parameter dependency lacks a validated derivative binding".to_owned(),
            ));
        }
        for &input in self.plan.statistical_model_dependencies.inputs() {
            output[input] = Err(GradientSiteFailure::Unsupported(
                "input affects occurrence-bound model realization".to_owned(),
            ));
        }
        for &input in self.plan.option_dependencies.inputs() {
            output[input] = Err(GradientSiteFailure::Unsupported(
                "input affects numerical analysis options".to_owned(),
            ));
        }
        for (site_index, site) in self.plan.sites.iter().enumerate() {
            for &input in site.inputs.inputs() {
                if output[input].is_err() {
                    continue;
                }
                match self.gradient_site(site_index, input, limits) {
                    Ok(Some(link)) => {
                        if let Ok(ref mut links) = output[input] {
                            links.push(link);
                        }
                    }
                    Ok(None) => {}
                    Err(error) => output[input] = Err(error),
                }
            }
        }
        output
    }

    /// Current committed typed site value by stable plan site index.
    #[must_use = "the returned option distinguishes a committed typed value from an invalid site index"]
    pub fn site_value(&self, index: usize) -> Option<Real> {
        self.site_values.get(index).copied()
    }

    /// Site expressions evaluated by the last successful update.
    #[must_use = "the returned count measures dependency-driven expression evaluation work for the last successful update"]
    pub const fn last_site_evaluations(&self) -> usize {
        self.last_site_evaluations
    }

    /// Decimal materializations by the last successful update.
    #[must_use = "the returned count measures decimal materialization work for the last successful update"]
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
        let baseline_context = self.plan.config.context().model_context();
        let mut fixed_context = baseline_context;
        let mut structural = self.plan.requires_rebuild
            || self
                .plan
                .statistical_model_dependencies
                .changed_from_baseline(parameters)
            || self
                .plan
                .option_dependencies
                .changed_from_baseline(parameters)
            || self
                .plan
                .behavioural_dependencies
                .changed_from_baseline(parameters);
        // Preserve the small fixed-context route with zero heap allocation
        // when one or two ordinary sites change. Beyond its actual capacity,
        // promote the exact accumulated sites to an unbounded worker overlay.
        let mut extra: Option<Vec<CandidateNumericOverride>> = None;
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
            if structural {
                continue;
            }
            let selected = match route {
                NumericRoute::Resistor(target) => target
                    .at_resistance(value)
                    .map(CandidateNumericOverride::Resistor),
                NumericRoute::Instance(target) => target
                    .at_source(value, baseline_context.geometry_scale)
                    .map(CandidateNumericOverride::Instance),
                NumericRoute::Rebuild => {
                    structural = true;
                    continue;
                }
            };
            if let Ok(replacement) = selected {
                append_numeric_override(
                    &mut fixed_context,
                    &mut extra,
                    baseline_context,
                    replacement,
                    self.plan.sites.len(),
                )?;
            } else {
                structural = true;
            }
        }
        let (context, overrides) = extra.map_or_else(
            || (fixed_context, Vec::new()),
            |sparse| (baseline_context, sparse),
        );
        if !structural {
            // Resolve all other fallible candidate metadata before publishing
            // the numeric overlay and its validated physical preparation.
            let scope = parameters.resolved_scope(self.plan.root)?;
            if self
                .baseline
                .commit_candidate_overrides(&context, overrides)
                .is_ok()
            {
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
        }
        let prepared = self.prepare_structural(parameters)?;
        self.baseline.reset_candidate_overrides();
        Ok((prepared, evaluations, roundings))
    }

    fn prepare_structural(
        &self,
        parameters: &CandidateParameters<'_>,
    ) -> SpiceResult<PreparedCandidate> {
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
        Ok(PreparedCandidate {
            context,
            config,
            circuit: Some(circuit),
        })
    }
}
