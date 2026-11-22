//! Plan current-sense rewrites on original, scope-local device identities.
//!
//! C visits references in expanded deck order, including nested definitions.
//! Generated devices never participate in the original-name index. Each
//! original device owns its first-node edit, occurrence renames and optional
//! sense source; applying the plan expands its ordered card range only once.

use std::collections::HashMap;

use crate::netlist::ast::{
    DeviceInstance, Netlist, ParameterKind, ScopeView, ScopedCard, ScopedCardKind, Subcircuit,
};
use crate::netlist::bexpr::BExprKind;
use crate::primitives::{SpiceError, SpiceResult};

use super::{forms::NonlinearForm, instance, lower_device, scalar};

/// A device ordinal in the immutable original vector of the current scope.
#[derive(Clone, Copy)]
struct DeviceId(usize);

#[derive(Default)]
struct DeviceEdit {
    first_node: Option<String>,
    renames: Vec<(usize, String)>,
    sense_source: Option<Box<DeviceInstance>>,
}

#[derive(Default)]
struct MeasurementTargets {
    simple: bool,
    measured: Vec<DeviceId>,
    visited: bool,
}

/// All edits for one lexical scope, and corresponding plans for its children.
pub(super) struct ScopeEdits {
    devices: Vec<DeviceEdit>,
    children: Vec<Self>,
}

impl ScopeEdits {
    pub(super) fn build(scope: ScopeView<'_>, serial: &mut usize) -> SpiceResult<Self> {
        if !scope.cards.is_empty() {
            scope
                .validate_order()
                .map_err(|error| SpiceError::circuit(error.to_string()))?;
        }
        let mut targets = None;

        let mut plan = Self {
            devices: (0..scope.devices.len())
                .map(|_| DeviceEdit::default())
                .collect(),
            children: Vec::with_capacity(scope.subcircuits.len()),
        };
        // Without source cards, semantic vector order is the established
        // fallback. Otherwise the shared scope invariant guarantees that
        // every device/child appears exactly once in its own vector order.
        if scope.cards.is_empty() {
            for index in 0..scope.devices.len() {
                plan.references(scope.devices, DeviceId(index), &mut targets, serial)?;
            }
            for child in scope.subcircuits {
                plan.children
                    .push(Self::build(ScopeView::from(child), serial)?);
            }
        } else {
            for card in scope.cards {
                match card.kind {
                    ScopedCardKind::Device(index) => {
                        plan.references(scope.devices, DeviceId(index), &mut targets, serial)?;
                    }
                    ScopedCardKind::Subcircuit(index) => {
                        plan.children.push(Self::build(
                            ScopeView::from(&scope.subcircuits[index]),
                            serial,
                        )?);
                    }
                    _ => {}
                }
            }
        }
        Ok(plan)
    }

    fn index_targets(devices: &[DeviceInstance]) -> SpiceResult<HashMap<&str, MeasurementTargets>> {
        let mut targets = HashMap::<&str, MeasurementTargets>::new();
        for (index, device) in devices.iter().enumerate() {
            let target = targets.entry(&device.name).or_default();
            if matches!(device.designator, 'e' | 'h') && NonlinearForm::validate(device)?.is_none()
            {
                target.simple = true;
            } else if !device.nodes.is_empty() {
                target.measured.push(DeviceId(index));
            }
        }
        Ok(targets)
    }

    fn references<'a>(
        &mut self,
        devices: &'a [DeviceInstance],
        device: DeviceId,
        targets: &mut Option<HashMap<&'a str, MeasurementTargets>>,
        serial: &mut usize,
    ) -> SpiceResult<()> {
        let mut occurrence = 0;
        for parameter in &devices[device.0].parameters {
            let ParameterKind::Behavioural(expression) = &parameter.kind else {
                continue;
            };
            let mut result: SpiceResult<()> = Ok(());
            expression.root.visit(&mut |node| {
                if result.is_err() {
                    return;
                }
                if let BExprKind::Current(name) = &node.kind {
                    let offset = node
                        .span
                        .start
                        .column
                        .saturating_sub(expression.span.start.column)
                        as usize;
                    result = (|| {
                        if !name.starts_with('v')
                            && measured_reference(&expression.text, offset).ok_or_else(|| {
                                SpiceError::parse(
                                    node.span.start.clone(),
                                    "current-reference position splits a UTF-8 character",
                                )
                            })?
                        {
                            if targets.is_none() {
                                *targets = Some(Self::index_targets(devices)?);
                            }
                            self.reference(
                                devices,
                                device,
                                occurrence,
                                name,
                                targets
                                    .as_mut()
                                    .and_then(|index| index.get_mut(name.as_str())),
                                serial,
                            )?;
                        }
                        occurrence += 1;
                        Ok(())
                    })();
                }
            });
            result?;
        }
        Ok(())
    }

    fn reference(
        &mut self,
        devices: &[DeviceInstance],
        device: DeviceId,
        occurrence: usize,
        name: &str,
        target: Option<&mut MeasurementTargets>,
        serial: &mut usize,
    ) -> SpiceResult<()> {
        let source = format!("v_{name}");
        if !target.as_ref().is_some_and(|target| target.simple) {
            self.devices[device.0]
                .renames
                .push((occurrence, source.clone()));
        }
        let Some(target) = target else {
            return Ok(());
        };

        // C increments the serial for *every* matching non-simple target,
        // even after it was visited. Only the first reference can edit
        // these targets or insert a source; subsequent references advance
        // the serial in constant time, including duplicate instance names.
        let next = serial
            .checked_add(target.measured.len())
            .ok_or_else(|| SpiceError::circuit("current-sense source serial overflow"))?;
        if !target.visited {
            for (offset, &id) in target.measured.iter().enumerate() {
                let device = &devices[id.0];
                let original = &device.nodes[0];
                let measured = if original.contains("_vmeas") {
                    original.clone()
                } else {
                    let name = format!("{original}_vmeas_{}", *serial + offset);
                    self.devices[id.0].first_node = Some(name.clone());
                    name
                };
                if offset == 0 {
                    let base = original
                        .split_once("_vmeas_")
                        .map_or(original.as_str(), |(base, _)| base);
                    self.devices[id.0].sense_source = Some(Box::new(instance(
                        source.clone(),
                        'v',
                        vec![base.to_owned(), measured],
                        vec![scalar("dc", "0", &device.location)],
                        &device.location,
                    )));
                }
            }
            target.visited = true;
        }
        *serial = next;
        Ok(())
    }

    pub(super) fn apply(self, netlist: &mut Netlist) -> SpiceResult<()> {
        self.apply_scope(
            &mut netlist.devices,
            &mut netlist.cards,
            &mut netlist.subcircuits,
        )
    }

    fn apply_scope(
        self,
        devices: &mut Vec<DeviceInstance>,
        cards: &mut Vec<ScopedCard>,
        children: &mut [Subcircuit],
    ) -> SpiceResult<()> {
        let originals = std::mem::take(devices);
        let mut out = Vec::with_capacity(originals.len());
        let mut ranges = Vec::with_capacity(originals.len());
        for (mut device, edit) in originals.into_iter().zip(self.devices) {
            let start = out.len();
            if let Some(node) = edit.first_node {
                device.nodes[0] = node;
            }
            rename_occurrences(&mut device, &edit.renames);
            lower_device(device, &mut out)?;
            out.extend(edit.sense_source.map(|source| *source));
            ranges.push(start..out.len());
        }

        // Generated cards inherit the original card's provenance, but each
        // receives its own semantic ordinal. Include positions and all other
        // indexed collections remain unchanged.
        let original_cards = std::mem::take(cards);
        cards.reserve(original_cards.len() + out.len() - ranges.len());
        for card in original_cards {
            if let ScopedCardKind::Device(index) = card.kind {
                let range = ranges[index].clone();
                if let Some(last) = range.clone().next_back() {
                    for index in range.start..last {
                        cards.push(ScopedCard {
                            kind: ScopedCardKind::Device(index),
                            ..card.clone()
                        });
                    }
                    cards.push(ScopedCard {
                        kind: ScopedCardKind::Device(last),
                        ..card
                    });
                }
            } else {
                cards.push(card);
            }
        }
        *devices = out;
        for (child, plan) in children.iter_mut().zip(self.children) {
            plan.apply_scope(&mut child.devices, &mut child.cards, &mut child.subcircuits)?;
        }
        Ok(())
    }
}

/// Whether `inp_meas_current()` rewrites the `i(` at byte `offset` of an
/// expression's text: after stripping whitespace inside parentheses, the
/// previous character must be arithmetic, `{`, `=` or an outside space.
fn measured_reference(text: &str, offset: usize) -> Option<bool> {
    let before = text.get(..offset.min(text.len()))?;
    let trimmed = before.trim_end();
    let Some(previous) = trimmed.chars().last() else {
        return Some(true);
    };
    if trimmed.len() < before.len() {
        let depth = trimmed.chars().fold(0i32, |depth, c| match c {
            '(' => depth + 1,
            ')' => depth - 1,
            _ => depth,
        });
        if depth <= 0 {
            return Some(true);
        }
    }
    Some("+-*/()<>?:|&^!%\\{='".contains(previous))
}

fn rename_occurrences(device: &mut DeviceInstance, renames: &[(usize, String)]) {
    if renames.is_empty() {
        return;
    }
    let mut seen = 0;
    let mut remaining = renames.iter();
    let mut next = remaining.next();
    for parameter in &mut device.parameters {
        let ParameterKind::Behavioural(expression) = &mut parameter.kind else {
            continue;
        };
        expression.root.visit_mut(&mut |node| {
            if let BExprKind::Current(source) = &mut node.kind {
                if let Some((occurrence, name)) = next
                    && seen == *occurrence
                {
                    name.clone_into(source);
                    next = remaining.next();
                }
                seen += 1;
            }
        });
    }
}
