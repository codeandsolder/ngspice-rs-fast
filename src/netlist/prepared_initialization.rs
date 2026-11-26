//! Bind the producer's typed initialization sidecar to the parsed electrical
//! artifacts. No source-specific syntax is interpreted in the solver.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use model_normalize_ngspice::param_expression::{SpiceExpression, SpiceExpressionContext};
use model_normalize_ngspice::passive_ic::{
    Analysis, BiasConstraint, InitialConditions, NodeInitialCondition, PassiveInitialCondition,
    PassiveKind, ResolvedDefaults,
};

use crate::devices::{
    StorageKind,
    passive_bias::{NodeInitialization, PassiveBias, PassiveInitialization},
};
use crate::netlist::{
    ast::{DeviceInstance, IncludeDirective, Netlist, ScopedCard, Subcircuit},
    eval::{EvalBudget, ParamScope},
};
use crate::primitives::{AnalysisKind, Real, SourceLoc, SpiceError, SpiceResult};

type Scope = Vec<String>;
type Identity = (PathBuf, Scope, String);

/// Source-prepared policy and canonical artifact identities. A native deck
/// has no such policy; attaching one cannot change the meaning of native IC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedInitialConditions {
    source: Arc<InitialConditions>,
    artifacts: BTreeMap<String, PathBuf>,
    analyses: BTreeSet<(AnalysisKind, bool)>,
}

impl PreparedInitialConditions {
    /// Associate the parsed electrical artifacts with the producer sidecar
    /// for the explicitly selected enclosing jobs, including UIC flags.
    /// Artifact paths are relative to the emitted bundle directory. All
    /// device declarations must match an exact parsed lexical identity.
    ///
    /// # Errors
    /// Invalid artifact path, missing or duplicate declaration, wrong storage
    /// designator, or an empty or unsupported analysis selection.
    pub fn attach(
        netlist: &mut Netlist,
        directory: &Path,
        source: InitialConditions,
        analyses: impl IntoIterator<Item = (AnalysisKind, bool)>,
    ) -> SpiceResult<()> {
        let analyses = analyses.into_iter().collect::<BTreeSet<_>>();
        if analyses.is_empty()
            || analyses.iter().any(|&(kind, uic)| {
                source_analysis(kind, uic).is_none() || (uic && kind != AnalysisKind::Transient)
            })
        {
            return Err(SpiceError::parse(
                netlist.location.clone(),
                "invalid prepared initialization analysis selection",
            ));
        }
        let fail = |message| SpiceError::parse(netlist.location.clone(), message);
        let root = directory
            .canonicalize()
            .map_err(|e| SpiceError::io(directory, &e))?;
        let mut artifacts = BTreeMap::new();
        for name in source
            .devices
            .iter()
            .map(|d| &d.artifact)
            .chain(source.nodes.iter().map(|n| &n.artifact))
        {
            if artifacts.contains_key(name) {
                continue;
            }
            let relative = Path::new(name);
            if relative.as_os_str().is_empty()
                || !relative
                    .components()
                    .all(|c| matches!(c, Component::Normal(_)))
            {
                return Err(fail(format!("invalid prepared artifact {name}")));
            }
            let path = root.join(relative);
            let canonical = path.canonicalize().map_err(|e| SpiceError::io(&path, &e))?;
            if !canonical.starts_with(&root) {
                return Err(fail(format!("prepared artifact escapes bundle: {name}")));
            }
            artifacts.insert(name.clone(), canonical);
        }
        let policy = Self {
            source: Arc::new(source),
            artifacts,
            analyses,
        };
        policy.validate(netlist)?;
        netlist.prepared_initialization = Some(policy);
        Ok(())
    }

    fn identity(&self, declaration: &PassiveInitialCondition) -> Identity {
        (
            self.artifacts[&declaration.artifact].clone(),
            canonical_scope(&declaration.scope),
            declaration.device.to_ascii_lowercase(),
        )
    }

    fn validate(&self, netlist: &Netlist) -> SpiceResult<()> {
        let mut written = BTreeMap::new();
        collect_devices(&netlist.devices, &[], &netlist.subcircuits, &mut written);
        let scopes = scope_files(netlist);
        let mut seen = BTreeSet::new();
        for d in &self.source.devices {
            let id = self.identity(d);
            let expected = match d.kind {
                PassiveKind::Capacitor => 'c',
                PassiveKind::Inductor => 'l',
            };
            if written.get(&id) != Some(&Some(expected)) || !seen.insert(id) {
                return Err(SpiceError::parse(
                    netlist.location.clone(),
                    format!(
                        "missing, duplicate or incompatible prepared passive {} in {}",
                        d.device, d.artifact
                    ),
                ));
            }
        }
        for n in &self.source.nodes {
            if !scopes
                .get(&canonical_scope(&n.scope))
                .is_some_and(|paths| paths.contains(&self.artifacts[&n.artifact]))
            {
                return Err(SpiceError::parse(
                    netlist.location.clone(),
                    format!(
                        "missing prepared node hint scope {:?} in {}",
                        n.scope, n.artifact
                    ),
                ));
            }
        }
        Ok(())
    }
}

fn canonical_scope(scope: &[String]) -> Scope {
    scope.iter().map(|part| part.to_ascii_lowercase()).collect()
}

fn collect_devices(
    devices: &[DeviceInstance],
    scope: &[String],
    definitions: &[Subcircuit],
    written: &mut BTreeMap<Identity, Option<char>>,
) {
    for d in devices {
        let id = (
            d.location.path().to_path_buf(),
            scope.to_vec(),
            d.name.to_ascii_lowercase(),
        );
        written
            .entry(id)
            .and_modify(|kind| *kind = None)
            .or_insert(Some(d.designator));
    }
    for definition in definitions {
        let mut child = scope.to_vec();
        child.push(definition.name.to_ascii_lowercase());
        collect_devices(
            &definition.devices,
            &child,
            &definition.subcircuits,
            written,
        );
    }
}

// Only artifacts admitted into the selected scope may bind node declarations.
// Include directives count even if all their electrical lines were stripped.
fn files(
    location: &SourceLoc,
    cards: &[ScopedCard],
    includes: &[IncludeDirective],
) -> BTreeSet<PathBuf> {
    std::iter::once(location.path().to_path_buf())
        .chain(
            cards
                .iter()
                .map(|card| card.source.location.path().to_path_buf()),
        )
        .chain(
            includes
                .iter()
                .filter_map(|include| include.resolved_path.clone()),
        )
        .collect()
}

fn scope_files(netlist: &Netlist) -> BTreeMap<Scope, BTreeSet<PathBuf>> {
    fn collect(
        definitions: &[Subcircuit],
        parent: &[String],
        scopes: &mut BTreeMap<Scope, BTreeSet<PathBuf>>,
    ) {
        for d in definitions {
            let mut scope = parent.to_vec();
            scope.push(d.name.to_ascii_lowercase());
            // Match the native first-definition-wins lexical lookup.
            if !scopes.contains_key(&scope) {
                scopes.insert(scope.clone(), files(&d.location, &d.cards, &d.includes));
                collect(&d.subcircuits, &scope, scopes);
            }
        }
    }
    let mut scopes = BTreeMap::new();
    scopes.insert(
        Vec::new(),
        files(&netlist.location, &netlist.cards, &netlist.includes),
    );
    collect(&netlist.subcircuits, &[], &mut scopes);
    scopes
}

/// Resolved occurrence equations from one expansion, in source order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoundInitializations {
    pub(crate) analyses: Option<BTreeSet<(AnalysisKind, bool)>>,
    pub(crate) devices: Vec<(String, PassiveInitialization)>,
    pub(crate) nodes: Vec<(String, NodeInitialization)>,
}

#[derive(Debug)]
struct Job {
    kind: AnalysisKind,
    analysis: Analysis,
    defaults: ResolvedDefaults,
}

/// All declaration lookup is indexed before walking occurrences. The actual
/// scalar evaluation reuses the existing lexical parameter/function scope.
pub(crate) struct Binder<'a> {
    devices: BTreeMap<Identity, &'a PassiveInitialCondition>,
    nodes: BTreeMap<Scope, Vec<(&'a Path, &'a NodeInitialCondition)>>,
    jobs: Vec<Job>,
    uic: Option<Job>,
    seen_nodes: BTreeSet<String>,
    pub(crate) bound: BoundInitializations,
}

fn source_analysis(kind: AnalysisKind, uic: bool) -> Option<Analysis> {
    Some(match kind {
        AnalysisKind::OperatingPoint => Analysis::OperatingPoint,
        AnalysisKind::Transient => Analysis::Transient { uic },
        AnalysisKind::Ac => Analysis::Ac,
        AnalysisKind::DcSweep => Analysis::DcSweep,
        AnalysisKind::Noise => Analysis::Noise,
        AnalysisKind::TransferFunction => Analysis::TransferFunction,
        AnalysisKind::Sensitivity => Analysis::Sensitivity,
        _ => return None,
    })
}

/// Source-side normalized scalar grammar includes lazy conditionals and
/// comparisons not present in the native numparam subset. Reuse the
/// producer's compiled evaluator and delegate source functions to the
/// already-bound native lexical scope.
struct ExpressionContext<'a> {
    scope: &'a ParamScope,
    analysis: Analysis,
    location: &'a SourceLoc,
    budget: &'a mut EvalBudget,
}

impl SpiceExpressionContext for ExpressionContext<'_> {
    fn symbol(&mut self, name: &str) -> anyhow::Result<Real> {
        if name.eq_ignore_ascii_case("analysis") {
            Ok(self.analysis.source_code())
        } else {
            self.scope
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("undefined prepared parameter {name}"))
        }
    }

    fn function(&mut self, name: &str, arguments: &[SpiceExpression]) -> anyhow::Result<Real> {
        let args = arguments
            .iter()
            .map(|argument| argument.evaluate_with(self))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let code = self.analysis.source_code();
        self.scope
            .evaluate_prepared_call(
                name,
                &args,
                self.location,
                |symbol| symbol.eq_ignore_ascii_case("analysis").then_some(code),
                self.budget,
            )
            .map_err(Into::into)
    }
}

fn evaluate(
    text: &str,
    location: &SourceLoc,
    scope: &ParamScope,
    analysis: Analysis,
    budget: &mut EvalBudget,
) -> SpiceResult<Real> {
    budget.charge_prepared(text.len(), location)?;
    let result = SpiceExpression::parse(text).and_then(|expression| {
        expression.evaluate_with(&mut ExpressionContext {
            scope,
            analysis,
            location,
            budget,
        })
    });
    result.map_err(|error| SpiceError::parse(location.clone(), error.to_string()))
}

fn physical_constraint(bias: BiasConstraint) -> SpiceResult<Option<PassiveBias>> {
    Ok(Some(match bias {
        BiasConstraint::None => return Ok(None),
        BiasConstraint::CapacitorHard { voltage } => PassiveBias::Voltage {
            voltage,
            conductance: None,
        },
        BiasConstraint::CapacitorSoft {
            voltage,
            resistance,
        } => PassiveBias::Voltage {
            voltage,
            conductance: Some(resistance.recip()),
        },
        BiasConstraint::InductorHard { current } => PassiveBias::Current {
            current,
            conductance: 0.,
        },
        BiasConstraint::InductorSoft {
            current,
            conductance,
        } => PassiveBias::Current {
            current,
            conductance,
        },
        BiasConstraint::NodeSoft { .. } => {
            return Err(SpiceError::circuit(
                "node policy returned for passive device",
            ));
        }
    }))
}

impl<'a> Binder<'a> {
    pub(crate) fn new(
        netlist: &'a Netlist,
        root: &ParamScope,
        budget: &mut EvalBudget,
    ) -> SpiceResult<Option<Self>> {
        let Some(policy) = &netlist.prepared_initialization else {
            return Ok(None);
        };
        policy.validate(netlist)?;
        let kinds = policy.analyses.clone();
        let mut jobs = Vec::new();
        let mut uic = None;
        for &(kind, use_ic) in &kinds {
            let analysis = source_analysis(kind, use_ic).ok_or_else(|| {
                SpiceError::parse(
                    netlist.location.clone(),
                    format!(
                        "prepared initial conditions unsupported for {}",
                        kind.as_str()
                    ),
                )
            })?;
            let defaults = policy
                .source
                .resolve_defaults(|text| {
                    evaluate(text, &netlist.location, root, analysis, budget).map_err(Into::into)
                })
                .map_err(|error| SpiceError::parse(netlist.location.clone(), error.to_string()))?;
            let job = Job {
                kind,
                analysis,
                defaults,
            };
            if use_ic {
                uic = Some(job);
            } else {
                jobs.push(job);
            }
        }
        let devices = policy
            .source
            .devices
            .iter()
            .map(|d| (policy.identity(d), d))
            .collect();
        let mut nodes: BTreeMap<Scope, Vec<_>> = BTreeMap::new();
        for n in &policy.source.nodes {
            nodes
                .entry(canonical_scope(&n.scope))
                .or_default()
                .push((policy.artifacts[&n.artifact].as_path(), n));
        }
        Ok(Some(Self {
            devices,
            nodes,
            jobs,
            uic,
            seen_nodes: BTreeSet::new(),
            bound: BoundInitializations {
                analyses: Some(kinds),
                ..BoundInitializations::default()
            },
        }))
    }

    pub(crate) fn device(
        &mut self,
        original: &DeviceInstance,
        name: &str,
        lexical_scope: &[String],
        scope: &ParamScope,
        budget: &mut EvalBudget,
    ) -> SpiceResult<()> {
        let key = (
            original.location.path().to_path_buf(),
            canonical_scope(lexical_scope),
            original.name.to_ascii_lowercase(),
        );
        let Some(d) = self.devices.get(&key) else {
            return Ok(());
        };
        let location = &original.location;
        let mut constraints = Vec::new();
        let mut storage_ic = None;
        for job in &self.jobs {
            let resolved = d
                .resolve(job.analysis, job.defaults, |text| {
                    evaluate(text, location, scope, job.analysis, budget).map_err(Into::into)
                })
                .map_err(|e| SpiceError::parse(location.clone(), e.to_string()))?;
            if let Some(bias) = physical_constraint(resolved.bias)? {
                constraints.push((job.kind, bias));
            }
        }
        if let Some(job) = &self.uic {
            storage_ic = d
                .resolve(job.analysis, job.defaults, |text| {
                    evaluate(text, location, scope, job.analysis, budget).map_err(Into::into)
                })
                .map_err(|e| SpiceError::parse(location.clone(), e.to_string()))?
                .storage_ic;
        }
        let kind = match d.kind {
            PassiveKind::Capacitor => StorageKind::Capacitor,
            PassiveKind::Inductor => StorageKind::Inductor,
        };
        self.bound.devices.push((
            name.to_owned(),
            PassiveInitialization::new(kind, storage_ic, constraints)?,
        ));
        Ok(())
    }

    pub(crate) fn nodes(
        &mut self,
        lexical_scope: &[String],
        scope: &ParamScope,
        rename: impl Fn(&str) -> String,
        budget: &mut EvalBudget,
    ) -> SpiceResult<()> {
        let Some(nodes) = self.nodes.get(&canonical_scope(lexical_scope)) else {
            return Ok(());
        };
        for &(path, n) in nodes {
            let name = rename(&n.node.to_ascii_lowercase());
            if !self.seen_nodes.insert(name.to_ascii_lowercase()) {
                continue;
            }
            let location = SourceLoc::new(path.to_path_buf(), 1, 1);
            let mut constraints = Vec::new();
            let mut hint = 0.;
            for job in &self.jobs {
                let (_, bias) = n
                    .resolve(job.analysis, job.defaults, |text| {
                        evaluate(text, &location, scope, job.analysis, budget).map_err(Into::into)
                    })
                    .map_err(|e| SpiceError::parse(location.clone(), e.to_string()))?;
                match bias {
                    BiasConstraint::None => (),
                    BiasConstraint::NodeSoft {
                        voltage,
                        resistance,
                    } => constraints.push((job.kind, voltage, resistance.recip())),
                    _ => {
                        return Err(SpiceError::parse(
                            location.clone(),
                            "passive constraint returned for node",
                        ));
                    }
                }
            }
            if let Some(job) = &self.uic {
                hint = n
                    .resolve(job.analysis, job.defaults, |text| {
                        evaluate(text, &location, scope, job.analysis, budget).map_err(Into::into)
                    })
                    .map_err(|e| SpiceError::parse(location.clone(), e.to_string()))?
                    .0;
            }
            self.bound
                .nodes
                .push((name, NodeInitialization::new(hint, location, constraints)?));
        }
        Ok(())
    }
}
