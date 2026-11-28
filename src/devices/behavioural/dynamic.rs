//! Solver-owned dynamic expression lowering and signed storage integration.
//! The expression compiler never owns mutable time history.

use crate::devices::nonlinear::charge::ChargeBranch;
use crate::devices::traits::{AnalysisMode, StampContext};
use crate::primitives::{Real, SpiceError, SpiceResult};

/// History-only source and present-quantity coefficient for the current
/// trial. A rejected trial leaves accepted history unchanged.
pub(super) fn integrate(
    context: &mut StampContext<'_>,
    quantity: Real,
) -> SpiceResult<(Real, Real)> {
    let storage = ChargeBranch::new(0, quantity, 1.);
    let result = if let AnalysisMode::Transient { dt, .. } = context.mode {
        let coefficients = context.integration.ok_or_else(|| {
            SpiceError::circuit("dynamic residual requires integration coefficients")
        })?;
        if coefficients.dt().to_bits() != dt.to_bits() {
            return Err(SpiceError::circuit("dynamic residual timestep mismatch"));
        }
        let companion = storage.integrate(&mut context.states, coefficients)?;
        (companion.current, companion.conductance)
    } else {
        storage.zero_derivative(&mut context.states)?;
        (0., 0.)
    };
    storage.store(&mut context.states)?;
    Ok(result)
}

use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};

use crate::netlist::ast::{CodePortKind, DeviceInstance, ParameterAssignment, ParameterKind};
use crate::netlist::bexpr::{
    BExpr, BExprKind, BehaviouralExpression, CurrentPolicy, DynamicOperator,
};
use crate::primitives::{NodeTable, SourceLoc};

pub(super) const DERIVATIVE: &str = "__dynamic_derivative";
const SAMPLED_CURRENT: &str = "__dynamic_sampled_current";

/// The private operator kind is the single source of truth for factory
/// dispatch and exclusion from source-declared node identities.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::devices) enum GeneratedOperator {
    Derivative,
    SampledCurrent,
}

pub(in crate::devices) fn generated_operator(
    instance: &DeviceInstance,
) -> Option<GeneratedOperator> {
    if instance.designator != 'b' {
        return None;
    }
    let [parameter] = instance.parameters.as_slice() else {
        return None;
    };
    match (parameter.name.as_str(), &parameter.kind) {
        (DERIVATIVE, ParameterKind::Behavioural(_)) => Some(GeneratedOperator::Derivative),
        (SAMPLED_CURRENT, ParameterKind::Instance) => Some(GeneratedOperator::SampledCurrent),
        _ => None,
    }
}

fn expressions(instance: &DeviceInstance) -> impl Iterator<Item = &BehaviouralExpression> {
    instance
        .parameters
        .iter()
        .filter_map(|parameter| match &parameter.kind {
            ParameterKind::Behavioural(expression) => Some(expression.as_ref()),
            _ => None,
        })
}

fn dynamic_operator(expr: &BExpr) -> Option<DynamicOperator> {
    match &expr.kind {
        BExprKind::Call { name, .. } => DynamicOperator::from_name(name),
        _ => None,
    }
}

pub(in crate::devices) struct NetworkReferences<'a> {
    existing: &'a [crate::devices::ElaboratedDevice],
    behavioural: BTreeMap<String, SourceLoc>,
    pub current: BTreeSet<String>,
    pub coupled: BTreeSet<String>,
}

#[derive(Default)]
struct CurrentTargets {
    probes: BTreeSet<usize>,
    zero: BTreeSet<String>,
    instantaneous: BTreeSet<String>,
}

/// A genuine voltage-imposing or winding branch, not merely an internal
/// constitutive row or an artificial observation probe. This selection is
/// intentionally independent of the factory's current-coordinate allocation.
const fn is_instantaneous_branch(designator: char, voltage_output: bool) -> bool {
    matches!(designator, 'v' | 'e' | 'h' | 'l') || (designator == 'b' && voltage_output)
}

/// A physical output current coordinate supplied by the selected factory.
fn has_native_current(
    instance: &DeviceInstance,
    models: &crate::devices::models::ModelResolver<'_>,
) -> SpiceResult<bool> {
    Ok(match instance.designator {
        'v' | 'e' | 'h' | 'l' => true,
        'c' if instance.parameters.iter().any(|p| p.name == "pn_control") => true,
        'r' => expressions(instance).next().is_some(),
        'c' => {
            expressions(instance).next().is_some()
                || models.resolve(instance)?.is_some_and(|model| {
                    model.family() == crate::devices::ModelFamily::PolynomialCapacitor
                })
        }
        'b' => instance
            .parameters
            .iter()
            .any(|parameter| parameter.name == "v"),
        'a' => {
            models.resolve(instance)?.is_some_and(|model| {
                matches!(
                    model.levels().backend,
                    crate::devices::models::ModelBackend::RationalTransfer
                        | crate::devices::models::ModelBackend::FrequencyTransfer
                )
            }) || instance
                .parameters
                .iter()
                .any(|parameter| parameter.name == "v")
        }
        _ => false,
    })
}

impl<'a> NetworkReferences<'a> {
    /// Electrical consumers before absent partition, dynamic lowering or
    /// numerical simplification. Zero-copy storage has only scalar parameters;
    /// absent partition never changes current reads in surviving expressions.
    pub(in crate::devices) fn collect(
        instances: &[DeviceInstance],
        existing: &'a [crate::devices::ElaboratedDevice],
    ) -> Self {
        let mut behavioural = BTreeMap::new();
        for expression in instances.iter().flat_map(expressions) {
            expression.root.visit(&mut |node| {
                if let BExprKind::Current(name) = &node.kind {
                    behavioural
                        .entry(name.to_ascii_lowercase())
                        .or_insert_with(|| node.span.start.clone());
                }
            });
        }
        let current = behavioural
            .keys()
            .cloned()
            .chain(
                instances
                    .iter()
                    .flat_map(|instance| instance.parameters.iter())
                    .filter(|parameter| {
                        parameter.name == "control" && parameter.kind == ParameterKind::Instance
                    })
                    .map(|parameter| parameter.value.to_ascii_lowercase()),
            )
            .chain(
                instances
                    .iter()
                    .flat_map(|instance| &instance.code_ports)
                    .flat_map(|port| &port.terminals)
                    .filter(|terminal| terminal.kind == CodePortKind::BranchCurrent)
                    .map(|terminal| terminal.positive.to_ascii_lowercase()),
            )
            .chain(
                existing
                    .iter()
                    .flat_map(|device| device.controlling_sources())
                    .map(|control| control.name.to_ascii_lowercase()),
            )
            .collect();
        let coupled = instances
            .iter()
            .filter(|instance| instance.designator == 'k')
            .flat_map(|instance| &instance.parameters)
            .filter(|parameter| parameter.kind == ParameterKind::Instance)
            .map(|parameter| parameter.value.to_ascii_lowercase())
            .chain(
                existing
                    .iter()
                    .filter_map(|device| device.mutual_coupling())
                    .flat_map(|coupling| coupling.inductors)
                    .map(|inductor| inductor.name.to_ascii_lowercase()),
            )
            .collect();
        Self {
            existing,
            behavioural,
            current,
            coupled,
        }
    }
}

fn measured_targets(
    instances: &[DeviceInstance],
    models: &crate::devices::models::ModelResolver<'_>,
    resistors: &crate::devices::pwl_resistor::BoundResistors,
    references: &NetworkReferences<'_>,
    absent: &crate::devices::passive_network::AbsentStorage,
) -> SpiceResult<CurrentTargets> {
    let referenced = &references.behavioural;
    let mut targets = CurrentTargets::default();
    if referenced.is_empty() {
        return Ok(targets);
    }
    for name in referenced.keys() {
        if absent.names.contains(name) {
            targets.zero.insert(name.clone());
        }
        if absent.windings.contains(name) {
            targets.instantaneous.insert(name.clone());
        }
    }
    for device in references.existing {
        let name = device.name().to_ascii_lowercase();
        if referenced.contains_key(&name)
            && is_instantaneous_branch(device.designator(), device.findable_branch().is_some())
        {
            targets.instantaneous.insert(name);
        }
    }
    for (index, instance) in instances.iter().enumerate() {
        let name = instance.name.to_ascii_lowercase();
        let Some(location) = referenced.get(&name) else {
            continue;
        };
        if is_instantaneous_branch(
            instance.designator,
            instance.parameters.iter().any(|p| p.name == "v"),
        ) {
            targets.instantaneous.insert(name.clone());
        }
        if !resistors.contains_key(&instance.name) && !has_native_current(instance, models)? {
            // A series probe reads the first physical terminal. A terminal-less
            // object or a collapsed/unknown port cannot be sensed this way.
            if instance.nodes.is_empty() || (instance.designator == 'a' && instance.model.is_some())
            {
                return Err(SpiceError::parse(
                    location.clone(),
                    format!(
                        "{} has no analog current measurement terminal",
                        instance.name
                    ),
                ));
            }
            targets.probes.insert(index);
        }
    }
    Ok(targets)
}

/// Extract dynamic operators once, after lexical resolution and hierarchy
/// expansion. Their outputs are solver-owned internal nodes; the original
/// expression becomes a static program reading these coordinates.
/// The common non-dynamic path borrows the input without cloning it.
pub(in crate::devices) struct ExpandedInstances<'a> {
    pub instances: Cow<'a, [DeviceInstance]>,
    pub probe_aliases: Vec<String>,
}

pub(in crate::devices) fn expand<'a>(
    instances: &'a [DeviceInstance],
    nodes: &NodeTable,
    models: &crate::devices::models::ModelResolver<'_>,
    references: &NetworkReferences<'_>,
    capacity: usize,
    resistors: &crate::devices::pwl_resistor::BoundResistors,
    absent: &crate::devices::passive_network::AbsentStorage,
) -> SpiceResult<ExpandedInstances<'a>> {
    let count = dynamic_count(instances);
    let CurrentTargets {
        probes: measured,
        zero,
        instantaneous,
    } = measured_targets(instances, models, resistors, references, absent)?;
    if count == 0 && measured.is_empty() && zero.is_empty() {
        return Ok(ExpandedInstances {
            instances: Cow::Borrowed(instances),
            probe_aliases: Vec::new(),
        });
    }
    // Selection may remove wrappers, zero-copy reads become constants, and
    // stateless samples of a single control row share one signal. Reserve an
    // upper bound but enforce the device budget on actual materialization.
    let total = instances
        .len()
        .saturating_add(count)
        .saturating_add(measured.len())
        .min(capacity);
    let (mut occupied_nodes, occupied_devices) = occupied_names(
        instances,
        nodes,
        references
            .existing
            .iter()
            .map(|device| device.name())
            .chain(absent.names.iter().map(String::as_str)),
    );
    occupied_nodes.extend(absent.terminals.iter().cloned());
    let mut expansion = Expansion {
        nodes: occupied_nodes,
        devices: occupied_devices,
        zero_currents: zero,
        instantaneous,
        renames: BTreeMap::new(),
        sampled_nodes: BTreeMap::new(),
        capacity,
        serial: 0,
        output: Vec::new(),
    };
    expansion
        .output
        .try_reserve_exact(total)
        .map_err(|error| SpiceError::circuit(format!("dynamic expression allocation: {error}")))?;
    let mut probe_aliases = Vec::new();
    probe_aliases
        .try_reserve_exact(measured.len())
        .map_err(|error| SpiceError::circuit(format!("current probe allocation: {error}")))?;
    let mut senses = BTreeMap::new();
    for index in measured {
        let (node, source) = expansion.names("sense", 'v')?;
        probe_aliases.push(node.clone());
        let original = &instances[index];
        expansion
            .renames
            .insert(original.name.to_ascii_lowercase(), source.clone());
        senses.insert(
            index,
            DeviceInstance {
                code_ports: Vec::new(),
                name: source,
                designator: 'v',
                nodes: vec![original.nodes[0].clone(), node],
                model: None,
                parameters: vec![ParameterAssignment {
                    name: "dc".to_owned(),
                    value: "0".to_owned(),
                    kind: ParameterKind::Scalar,
                    location: original.location.clone(),
                }],
                location: original.location.clone(),
            },
        );
    }
    for (index, instance) in instances.iter().enumerate() {
        let mut instance = instance.clone();
        for parameter in &mut instance.parameters {
            if let ParameterKind::Behavioural(expression) = &mut parameter.kind {
                expansion.extract(&mut expression.root)?;
            }
        }
        if let Some(sense) = senses.remove(&index) {
            instance.nodes[0].clone_from(&sense.nodes[1]);
            expansion.push(sense)?;
        }
        expansion.push(instance)?;
    }
    Ok(ExpandedInstances {
        instances: Cow::Owned(expansion.output),
        probe_aliases,
    })
}

/// Reserve names used by the input and all resolved expressions before generating nodes.
fn occupied_names<'n>(
    instances: &[DeviceInstance],
    nodes: &NodeTable,
    existing_names: impl Iterator<Item = &'n str>,
) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut occupied_nodes: BTreeSet<String> =
        nodes.nodes().iter().map(|node| node.name.clone()).collect();
    let mut occupied_devices: BTreeSet<String> =
        existing_names.map(str::to_ascii_lowercase).collect();
    for instance in instances {
        occupied_devices.insert(instance.name.to_ascii_lowercase());
        occupied_devices.extend(
            instance
                .parameters
                .iter()
                .filter(|parameter| matches!(parameter.kind, ParameterKind::Instance))
                .map(|parameter| parameter.value.to_ascii_lowercase()),
        );
        occupied_nodes.extend(instance.nodes.iter().map(|name| name.to_ascii_lowercase()));
        for expression in expressions(instance) {
            expression.root.visit(&mut |expr| match &expr.kind {
                BExprKind::Voltage { positive, negative } => {
                    occupied_nodes.insert(positive.to_ascii_lowercase());
                    if let Some(negative) = negative {
                        occupied_nodes.insert(negative.to_ascii_lowercase());
                    }
                }
                BExprKind::Current(name) => {
                    occupied_devices.insert(name.to_ascii_lowercase());
                }
                _ => {}
            });
        }
    }
    (occupied_nodes, occupied_devices)
}

fn dynamic_count(instances: &[DeviceInstance]) -> usize {
    let mut count = 0usize;
    for expression in instances.iter().flat_map(expressions) {
        expression.root.visit(&mut |expr| {
            if dynamic_operator(expr).is_some() {
                count = count.saturating_add(1);
            }
        });
    }
    count
}

struct Expansion {
    nodes: BTreeSet<String>,
    devices: BTreeSet<String>,
    zero_currents: BTreeSet<String>,
    instantaneous: BTreeSet<String>,
    renames: BTreeMap<String, String>,
    sampled_nodes: BTreeMap<String, String>,
    capacity: usize,
    serial: usize,
    output: Vec<DeviceInstance>,
}

impl Expansion {
    fn push(&mut self, device: DeviceInstance) -> SpiceResult<()> {
        if self.output.len() >= self.capacity {
            return Err(SpiceError::circuit(
                "dynamic expression exceeds device elaboration limit",
            ));
        }
        self.output.try_reserve(1).map_err(|error| {
            SpiceError::circuit(format!("dynamic expression allocation: {error}"))
        })?;
        self.output.push(device);
        Ok(())
    }

    fn names(&mut self, prefix: &str, designator: char) -> SpiceResult<(String, String)> {
        loop {
            let serial = self.serial;
            self.serial = serial
                .checked_add(1)
                .ok_or_else(|| SpiceError::circuit("dynamic expression name overflow"))?;
            let node = format!("__{prefix}_{serial}");
            let device = format!("{designator}__{prefix}_{serial}");
            if !self.nodes.contains(&node) && !self.devices.contains(&device) {
                self.nodes.insert(node.clone());
                self.devices.insert(device.clone());
                return Ok((node, device));
            }
        }
    }

    fn extract(&mut self, root: &mut BExpr) -> SpiceResult<()> {
        let mut result = Ok(());
        root.visit_mut(&mut |expr| {
            if result.is_err() {
                return;
            }
            match dynamic_operator(expr) {
                Some(DynamicOperator::Derivative) => result = self.extract_derivative(expr),
                Some(DynamicOperator::Current(policy)) => {
                    result = self.extract_sample(expr, policy);
                }
                None => {
                    if let BExprKind::Current(name) = &mut expr.kind {
                        let key = name.to_ascii_lowercase();
                        if self.zero_currents.contains(&key) {
                            expr.kind = zero_current();
                        } else if let Some(source) = self.renames.get(&key) {
                            name.clone_from(source);
                        }
                    }
                }
            }
        });
        result
    }

    fn extract_sample(&mut self, expr: &mut BExpr, policy: CurrentPolicy) -> SpiceResult<()> {
        let invalid = || {
            SpiceError::parse(
                expr.span.start.clone(),
                "current sampling takes one physical current reference i(instance)",
            )
        };
        let BExprKind::Call { arguments, .. } = &expr.kind else {
            return Err(invalid());
        };
        let [argument] = arguments.as_slice() else {
            return Err(invalid());
        };
        let mut argument = argument;
        while let BExprKind::Group(inner) = &argument.kind {
            argument = inner;
        }
        let BExprKind::Current(name) = &argument.kind else {
            return Err(invalid());
        };
        let key = name.to_ascii_lowercase();
        let requires_branch =
            policy == CurrentPolicy::BranchOrSampled && self.instantaneous.contains(&key);
        if self.zero_currents.contains(&key) {
            if requires_branch {
                return Err(SpiceError::parse(
                    expr.span.start.clone(),
                    format!("{name}: required physical branch is absent (zero multiplicity)"),
                ));
            }
            expr.kind = zero_current();
            return Ok(());
        }
        if requires_branch {
            // Classification preceded sensing aliases: an observation-only
            // source cannot acquire a voltage-imposing branch policy.
            expr.kind = BExprKind::Current(self.renames.get(&key).cloned().unwrap_or(key));
            return Ok(());
        }
        let name = self.renames.get(&key).cloned().unwrap_or(key);
        let node = if let Some(node) = self.sampled_nodes.get(&name) {
            node.clone()
        } else {
            let (node, device) = self.names("sampled_current", 'b')?;
            let location = expr.span.start.clone();
            self.push(DeviceInstance {
                code_ports: Vec::new(),
                name: device,
                designator: 'b',
                nodes: vec![node.clone()],
                model: None,
                parameters: vec![ParameterAssignment {
                    name: SAMPLED_CURRENT.to_owned(),
                    value: name.clone(),
                    kind: ParameterKind::Instance,
                    location: location.clone(),
                }],
                location,
            })?;
            self.sampled_nodes.insert(name, node.clone());
            node
        };
        expr.kind = BExprKind::Voltage {
            positive: node,
            negative: None,
        };
        Ok(())
    }

    fn extract_derivative(&mut self, expr: &mut BExpr) -> SpiceResult<()> {
        let BExprKind::Call { arguments, .. } = &mut expr.kind else {
            return Err(SpiceError::circuit("invalid dynamic expression"));
        };
        if arguments.len() != 1 {
            return Err(SpiceError::parse(
                expr.span.start.clone(),
                "ddt takes exactly one argument",
            ));
        }
        let mut input = arguments
            .pop()
            .ok_or_else(|| SpiceError::circuit("missing derivative input"))?;
        // Inner operators are emitted first. Each occurrence gets its own
        // accepted-state history, regardless of structural equality.
        self.extract(&mut input)?;
        let (node, device) = self.names("dynamic", 'b')?;
        let location = expr.span.start.clone();
        let span = input.span.clone();
        self.push(DeviceInstance {
            code_ports: Vec::new(),
            name: device,
            designator: 'b',
            nodes: vec![node.clone(), "0".to_owned()],
            model: None,
            parameters: vec![ParameterAssignment {
                name: DERIVATIVE.to_owned(),
                value: String::new(),
                kind: ParameterKind::Behavioural(Box::new(BehaviouralExpression {
                    text: String::new(),
                    span,
                    verbatim: true,
                    scalar_alternative: None,
                    root: input,
                })),
                location: location.clone(),
            }],
            location,
        })?;
        expr.kind = BExprKind::Voltage {
            positive: node,
            negative: None,
        };
        Ok(())
    }
}

fn zero_current() -> BExprKind {
    BExprKind::Number {
        value: 0.,
        spelling: "0".to_owned(),
    }
}
