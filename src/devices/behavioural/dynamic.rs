//! Solver-owned integration of a signed storage residual. The expression
//! compiler supplies its value and Jacobian, never a mutable time history.

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

fn is_derivative(expr: &BExpr) -> bool {
    matches!(&expr.kind, BExprKind::Call { name, .. } if name == "ddt")
}

/// A physical output current coordinate supplied by the selected factory.
fn has_native_current(
    instance: &DeviceInstance,
    models: &crate::devices::models::ModelResolver<'_>,
) -> SpiceResult<bool> {
    Ok(match instance.designator {
        'v' | 'e' | 'h' | 'l' => true,
        'r' | 'c' => expressions(instance).next().is_some(),
        'b' => instance
            .parameters
            .iter()
            .any(|parameter| parameter.name == "v"),
        'a' => {
            models.resolve(instance)?.is_some_and(|model| {
                model.levels().backend == crate::devices::models::ModelBackend::RationalTransfer
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
) -> SpiceResult<BTreeSet<usize>> {
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
        return Ok(BTreeSet::new());
    }
    let mut targets = BTreeSet::new();
    for (index, instance) in instances.iter().enumerate() {
        if let Some(location) = referenced.get(&instance.name.to_ascii_lowercase())
            && !has_native_current(instance, models)?
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
            targets.insert(index);
        }
    }
    Ok(targets)
}

/// Extract dynamic operators once, after lexical resolution and hierarchy
/// expansion. Their outputs are solver-owned internal nodes; the original
/// expression becomes a static program reading these coordinates.
/// The common non-dynamic path borrows the input without cloning it.
pub(in crate::devices) fn expand<'a, 'n>(
    instances: &'a [DeviceInstance],
    nodes: &NodeTable,
    models: &crate::devices::models::ModelResolver<'_>,
    existing_names: impl Iterator<Item = &'n str>,
    capacity: usize,
) -> SpiceResult<Cow<'a, [DeviceInstance]>> {
    let mut count = 0usize;
    for expression in instances.iter().flat_map(expressions) {
        expression.root.visit(&mut |expr| {
            if is_derivative(expr) {
                count = count.saturating_add(1);
            }
        });
    }
    let measured = measured_targets(instances, models)?;
    if count == 0 && measured.is_empty() {
        return Ok(Cow::Borrowed(instances));
    }
    let total = instances
        .len()
        .checked_add(count)
        .and_then(|total| total.checked_add(measured.len()))
        .filter(|&total| total <= capacity)
        .ok_or_else(|| {
            SpiceError::circuit("dynamic expression exceeds device elaboration limit")
        })?;
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
    let mut expansion = Expansion {
        nodes: occupied_nodes,
        devices: occupied_devices,
        serial: 0,
        output: Vec::new(),
    };
    expansion
        .output
        .try_reserve_exact(total)
        .map_err(|error| SpiceError::circuit(format!("dynamic expression allocation: {error}")))?;
    let mut senses = BTreeMap::new();
    let mut renames = BTreeMap::new();
    for index in measured {
        let (node, source) = expansion.names("sense", 'v')?;
        let original = &instances[index];
        renames.insert(original.name.to_ascii_lowercase(), source.clone());
        senses.insert(
            index,
            DeviceInstance {
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
    Ok(Cow::Owned(expansion.output))
}

struct Expansion {
    nodes: BTreeSet<String>,
    devices: BTreeSet<String>,
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
            if result.is_ok() && is_derivative(expr) {
                result = self.extract_one(expr);
            }
        });
        result
    }

    fn extract_one(&mut self, expr: &mut BExpr) -> SpiceResult<()> {
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
