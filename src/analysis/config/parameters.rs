//! Candidate coordinate binding and independent circuit preparation.
use std::collections::HashSet;
use std::sync::Arc;

use super::RunConfig;
use crate::analysis::AnalysisPlan;
use crate::devices::Circuit;
use crate::devices::circuit::CandidateTopology;
use crate::netlist::ast::{Netlist, ParameterAssignment, ParameterKind, ScopeView};
use crate::netlist::eval::{
    CandidateParameters, EvalBudget, FunctionScope, ParamScope, ParameterPlan, ParameterScopeId,
    ParameterScopeSpec,
};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Explicit setter occurrence in a parsed lexical scope, never a name lookup
/// or a runtime device ordinal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SetterSite {
    /// Device parameter.
    Device {
        /// Instance index.
        device: usize,
        /// Ordered setter index.
        parameter: usize,
    },
    /// Model parameter.
    Model {
        /// Declaration index.
        model: usize,
        /// Ordered setter index.
        parameter: usize,
    },
    /// Formal subcircuit default.
    Formal {
        /// Ordered setter index.
        parameter: usize,
    },
    /// Active local parameter definition (root names use the dependency plan).
    Parameter {
        /// Card index.
        card: usize,
        /// Assignment index.
        assignment: usize,
    },
}

/// A bound numeric setter. When scoped, the declaration applies to each
/// instance of that subcircuit; an X-card's own overrides retain precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CandidateSetter {
    /// Immediate subcircuit definition index, or root.
    pub definition: Option<usize>,
    /// Numeric setter occurrence.
    pub site: SetterSite,
}

fn invalid_site() -> SpiceError {
    SpiceError::circuit("invalid or nonnumeric candidate setter site")
}

fn scalar(setter: &ParameterAssignment) -> bool {
    matches!(
        setter.kind,
        ParameterKind::Scalar | ParameterKind::Expression(_)
    )
}

impl CandidateSetter {
    fn validate(self, deck: &Netlist) -> SpiceResult<()> {
        let definition = self
            .definition
            .map(|i| deck.subcircuits.get(i).ok_or_else(invalid_site))
            .transpose()?;
        let view = definition.map_or_else(|| ScopeView::from(deck), ScopeView::from);
        let found = match self.site {
            SetterSite::Device { device, parameter } => view
                .devices
                .get(device)
                .and_then(|d| d.parameters.get(parameter))
                .is_some_and(scalar),
            SetterSite::Model { model, parameter } => {
                let Some(declaration) = view.models.get(model) else {
                    return Err(invalid_site());
                };
                // First declaration wins, as in the model resolver.
                let active = view
                    .models
                    .iter()
                    .position(|m| m.name.eq_ignore_ascii_case(&declaration.name));
                active == Some(model) && declaration.parameters.get(parameter).is_some_and(scalar)
            }
            SetterSite::Formal { parameter } => definition
                .and_then(|s| s.parameters.get(parameter))
                .is_some_and(scalar),
            SetterSite::Parameter { card, assignment } => {
                let Some(definition) = definition else {
                    return Err(invalid_site());
                };
                let Some(entry) = definition
                    .params
                    .get(card)
                    .and_then(|c| c.assignments.get(assignment))
                else {
                    return Err(invalid_site());
                };
                !definition
                    .params
                    .iter()
                    .enumerate()
                    .skip(card)
                    .any(|(ci, c)| {
                        c.assignments.iter().enumerate().any(|(ai, a)| {
                            (ci > card || ai > assignment)
                                && a.name.eq_ignore_ascii_case(&entry.name)
                        })
                    })
            }
        };
        if found { Ok(()) } else { Err(invalid_site()) }
    }

    fn assign(self, deck: &mut Netlist, value: Real) -> SpiceResult<()> {
        let spelling = value.to_string();
        let (devices, models, formals, params) = match self.definition {
            Some(i) => {
                let scope = &mut deck.subcircuits[i];
                (
                    &mut scope.devices,
                    &mut scope.models,
                    Some(&mut scope.parameters),
                    &mut scope.params,
                )
            }
            None => (&mut deck.devices, &mut deck.models, None, &mut deck.params),
        };
        let setter = match self.site {
            SetterSite::Device { device, parameter } => &mut devices[device].parameters[parameter],
            SetterSite::Model { model, parameter } => {
                let model = &mut models[model];
                if model.parameters[parameter]
                    .name
                    .eq_ignore_ascii_case("level")
                    && !model.parameters[..parameter]
                        .iter()
                        .any(|p| p.name.eq_ignore_ascii_case("level"))
                {
                    model.level = Some(value);
                }
                &mut model.parameters[parameter]
            }
            SetterSite::Formal { parameter } => &mut formals.ok_or_else(invalid_site)?[parameter],
            SetterSite::Parameter { card, assignment } => {
                let expression = &mut params[card].assignments[assignment].expression;
                *expression = crate::netlist::parser::parse_parameter_expression(
                    &spelling,
                    &expression.span.start,
                )?;
                return Ok(());
            }
        };
        setter.value = spelling;
        setter.kind = ParameterKind::Scalar;
        Ok(())
    }
}

/// Immutable input schema and reusable root dependency plan. Sessions own
/// their numeric evaluator and original validated layout.
pub struct CandidatePlan<'a> {
    deck: &'a Netlist,
    parameters: ParameterPlan,
    root: ParameterScopeId,
    setters: Vec<CandidateSetter>,
}

impl<'a> CandidatePlan<'a> {
    /// Compile root tunables followed by explicit ordered setter coordinates.
    ///
    /// # Errors
    /// Invalid/duplicate tunables or setter sites, or invalid root definitions.
    pub fn compile(
        deck: &'a Netlist,
        tunables: &[&str],
        setters: &[CandidateSetter],
    ) -> SpiceResult<Self> {
        let mut budget = EvalBudget::default();
        let functions = FunctionScope::for_netlist(deck, &budget)?;
        let mut parameters = ParameterPlan::new();
        let root = parameters.add_scope(
            ParameterScopeSpec {
                parent: None,
                functions: Some(functions),
                bindings: &[],
                cards: &deck.params,
                instance_overrides: false,
                tunables,
            },
            &mut budget,
        )?;
        let mut seen = HashSet::new();
        for &setter in setters {
            setter.validate(deck)?;
            if !seen.insert(setter) {
                return Err(SpiceError::circuit("duplicate candidate setter site"));
            }
        }
        Ok(Self {
            deck,
            parameters,
            root,
            setters: setters.to_vec(),
        })
    }

    /// Number of independent coordinates.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.parameters.inputs().len() + self.setters.len()
    }

    /// Create an independent lane, validating the original circuit.
    ///
    /// # Errors
    /// Invalid original circuit, configuration or analysis.
    pub fn session(&self) -> SpiceResult<CandidateSession<'_>> {
        let evaluator = self.parameters.session();
        let scope = evaluator.resolved_scope(self.root)?;
        let config = RunConfig::with_resolved_scope(self.deck, scope.clone())?;
        let baseline = Circuit::from_netlist_with_scope(
            self.deck,
            &config.context().model_context(),
            Default::default(),
            Default::default(),
            Some(scope.clone()),
            None,
        )?;
        let baseline = CandidateTopology::new(baseline)?;
        config.plans_for(&self.deck.analyses)?;
        Ok(CandidateSession {
            plan: self,
            evaluator,
            baseline,
            scope,
            root_values: self
                .parameters
                .inputs()
                .iter()
                .map(|input| input.default.to_bits())
                .collect(),
            working: (!self.setters.is_empty()).then(|| self.deck.clone()),
            direct_values: vec![None; self.setters.len()],
        })
    }
}

/// Comparison against the independently validated original setup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateLayout {
    /// Same structural layout and selected compact-model bins.
    Same,
    /// Structurally different layout or selected backend/bin.
    Changed,
}

/// One independent circuit and its resolved analyses. Equal layouts do not
/// imply equal numeric factors or transferable accepted state.
pub struct PreparedCandidate {
    /// Independently finalized circuit.
    pub circuit: Circuit,
    /// Candidate configuration.
    pub config: RunConfig,
    /// Resolved analysis plans.
    pub analyses: Vec<AnalysisPlan>,
    /// Structural comparison against the original.
    pub layout: CandidateLayout,
    /// Dirty derived root definitions evaluated for the update.
    pub evaluated_definitions: usize,
}

/// Lane-local evaluator and never-simulated reference layout.
pub struct CandidateSession<'a> {
    plan: &'a CandidatePlan<'a>,
    evaluator: CandidateParameters<'a>,
    baseline: CandidateTopology,
    scope: Arc<ParamScope>,
    root_values: Vec<u64>,
    working: Option<Netlist>,
    direct_values: Vec<Option<u64>>,
}

impl CandidateSession<'_> {
    /// Prepare an independent candidate of arbitrary dimension.
    ///
    /// # Errors
    /// Invalid vector, dependency evaluation, options, device/model physics
    /// or analysis limits. Previously returned circuits remain independent.
    pub fn prepare(&mut self, values: &[Real]) -> SpiceResult<PreparedCandidate> {
        if values.len() != self.plan.input_count() || values.iter().any(|v| !v.is_finite()) {
            return Err(SpiceError::circuit("invalid candidate input vector"));
        }
        let (root, direct) = values.split_at(self.plan.parameters.inputs().len());
        let evaluated_definitions = self.evaluator.update(root, &mut EvalBudget::default())?;
        if self
            .root_values
            .iter()
            .copied()
            .ne(root.iter().map(|value| value.to_bits()))
        {
            self.scope = self.evaluator.resolved_scope(self.plan.root)?;
            for (saved, value) in self.root_values.iter_mut().zip(root) {
                *saved = value.to_bits();
            }
        }
        // Keep a private editable syntax tree and update only changed
        // coordinates. A failed physical preparation does not expose this
        // tree: the next complete vector replaces every changed coordinate.
        if let Some(deck) = &mut self.working {
            for ((&site, &value), previous) in self
                .plan
                .setters
                .iter()
                .zip(direct)
                .zip(&mut self.direct_values)
            {
                if *previous != Some(value.to_bits()) {
                    site.assign(deck, value)?;
                    *previous = Some(value.to_bits());
                }
            }
        }
        let deck = self.working.as_ref().unwrap_or(self.plan.deck);
        let scope = Arc::clone(&self.scope);
        let config = RunConfig::with_resolved_scope(deck, scope.clone())?;
        let circuit = Circuit::from_netlist_with_scope(
            deck,
            &config.context().model_context(),
            Default::default(),
            Default::default(),
            Some(scope),
            Some(&self.baseline),
        )?;
        let analyses = config.plans_for(&deck.analyses)?;
        let layout = if self.baseline.shares_layout(&circuit) {
            CandidateLayout::Same
        } else {
            CandidateLayout::Changed
        };
        Ok(PreparedCandidate {
            circuit,
            config,
            analyses,
            layout,
            evaluated_definitions,
        })
    }
}
