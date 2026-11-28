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

use crate::netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind};
use crate::netlist::bexpr::{BExpr, BExprKind, BehaviouralExpression};
use crate::primitives::NodeTable;

const DERIVATIVE: &str = "__dynamic_derivative";

/// The marker is emitted by the resolved-expression expansion, never by
/// the netlist parser. Ordinary behavioural sources cannot acquire it.
pub(in crate::devices) fn is_generated(instance: &DeviceInstance) -> bool {
    instance.designator == 'b'
        && matches!(instance.parameters.as_slice(), [parameter] if parameter.name == DERIVATIVE
            && matches!(parameter.kind, ParameterKind::Behavioural(_)))
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
        BExprKind::Call { name, .. } if name == "ddt" => Some(DynamicOperator::Derivative),
        BExprKind::Call { name, .. } if name == "sampled_current" => {
            Some(DynamicOperator::SampledCurrent)
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum DynamicOperator {
    Derivative,
    SampledCurrent,
}

#[derive(Default)]
struct CurrentTargets {
    probes: BTreeSet<usize>,
    zero: BTreeSet<String>,
}

// A source-prepared storage network with resolved m=0 has no physical
// current coordinate. Keep its scalar validation in NetworkPlan, but do not
// attempt to bind the absent branch when a behavioral expression reads it.
fn removed_network(instance: &DeviceInstance) -> SpiceResult<bool> {
    if matches!(instance.designator, 'c' | 'l')
        && instance
            .parameters
            .iter()
            .any(|p| crate::netlist::network_parameter(instance.designator, &p.name))
        && let Some(m) = instance.parameters.iter().rfind(|p| p.name == "m")
    {
        return Ok(crate::devices::factory::scalar_number(m)? == 0.);
    }
    Ok(false)
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

fn measured_targets(
    instances: &[DeviceInstance],
    models: &crate::devices::models::ModelResolver<'_>,
    resistors: &crate::devices::pwl_resistor::BoundResistors,
) -> SpiceResult<CurrentTargets> {
    let mut referenced = BTreeMap::new();
    for expression in instances.iter().flat_map(expressions) {
        expression.root.visit(&mut |node| {
            if let BExprKind::Current(name) = &node.kind {
                referenced
                    .entry(name.to_ascii_lowercase())
                    .or_insert_with(|| node.span.start.clone());
            }
        });
    }
    if referenced.is_empty() {
        return Ok(CurrentTargets::default());
    }
    let mut targets = CurrentTargets::default();
    for (index, instance) in instances.iter().enumerate() {
        let name = instance.name.to_ascii_lowercase();
        let Some(location) = referenced.get(&name) else {
            continue;
        };
        if removed_network(instance)? {
            targets.zero.insert(name);
        } else if !resistors.contains_key(&instance.name) && !has_native_current(instance, models)?
        {
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

pub(in crate::devices) fn expand<'a, 'n>(
    instances: &'a [DeviceInstance],
    nodes: &NodeTable,
    models: &crate::devices::models::ModelResolver<'_>,
    existing_names: impl Iterator<Item = &'n str>,
    capacity: usize,
    resistors: &crate::devices::pwl_resistor::BoundResistors,
) -> SpiceResult<ExpandedInstances<'a>> {
    let count = dynamic_count(instances);
    let CurrentTargets {
        probes: measured,
        zero,
    } = measured_targets(instances, models, resistors)?;
    if count == 0 && measured.is_empty() && zero.is_empty() {
        return Ok(ExpandedInstances {
            instances: Cow::Borrowed(instances),
            probe_aliases: Vec::new(),
        });
    }
    let total = instances
        .len()
        .checked_add(count)
        .and_then(|total| total.checked_add(measured.len()))
        .filter(|&total| total <= capacity)
        .ok_or_else(|| {
            SpiceError::circuit("dynamic expression exceeds device elaboration limit")
        })?;
    let (occupied_nodes, occupied_devices) = occupied_names(instances, nodes, existing_names);
    let mut expansion = Expansion {
        nodes: occupied_nodes,
        devices: occupied_devices,
        zero_currents: zero,
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
    let mut renames = BTreeMap::new();
    for index in measured {
        let (node, source) = expansion.names("sense", 'v')?;
        probe_aliases.push(node.clone());
        let original = &instances[index];
        renames.insert(original.name.to_ascii_lowercase(), source.clone());
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
                expression.root.visit_mut(&mut |expr| {
                    if let BExprKind::Current(name) = &mut expr.kind
                        && let Some(source) = renames.get(&name.to_ascii_lowercase())
                    {
                        name.clone_from(source);
                    }
                });
                expansion.extract(&mut expression.root)?;
            }
        }
        if let Some(sense) = senses.remove(&index) {
            instance.nodes[0].clone_from(&sense.nodes[1]);
            expansion.output.push(sense);
        }
        expansion.output.push(instance);
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
    serial: usize,
    output: Vec<DeviceInstance>,
}

impl Expansion {
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
                Some(DynamicOperator::SampledCurrent) => result = self.extract_sample(expr),
                None => {
                    if let BExprKind::Current(name) = &expr.kind
                        && self.zero_currents.contains(&name.to_ascii_lowercase())
                    {
                        expr.kind = zero_current();
                    }
                }
            }
        });
        result
    }

    fn extract_sample(&mut self, expr: &mut BExpr) -> SpiceResult<()> {
        let invalid = || {
            SpiceError::parse(
                expr.span.start.clone(),
                "sampled_current takes one physical current reference, sampled_current(i(instance))",
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
        if self.zero_currents.contains(&name.to_ascii_lowercase()) {
            expr.kind = zero_current();
            return Ok(());
        }
        let (node, device) = self.names("sampled_current", 'b')?;
        let location = expr.span.start.clone();
        self.output.push(DeviceInstance {
            code_ports: Vec::new(),
            name: device,
            designator: 'b',
            nodes: vec![node.clone()],
            model: None,
            parameters: vec![ParameterAssignment {
                name: super::sampled::MARKER.to_owned(),
                value: name.clone(),
                kind: ParameterKind::Instance,
                location: location.clone(),
            }],
            location,
        });
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
        self.output.push(DeviceInstance {
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
                    root: input,
                })),
                location: location.clone(),
            }],
            location,
        });
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
