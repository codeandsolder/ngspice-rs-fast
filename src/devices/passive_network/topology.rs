//! Topology selectors are resolved once, from expanded electrical references,
//! before allocating any physical-network or measurement-sensor devices.

use crate::{
    netlist::ast::{CodePortKind, DeviceInstance, ParameterKind},
    primitives::{Real, SpiceError, SpiceResult},
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub(in crate::devices) struct ReferenceGraph {
    controls: BTreeSet<String>,
    coupled: BTreeSet<String>,
}

/// Retain the selected topology across incremental edits. An edit that
/// changes a selection must re-elaborate the original occurrence graph.
#[derive(Default)]
pub(in crate::devices) struct TopologyChoices {
    voltage: BTreeMap<String, bool>,
    winding: BTreeMap<String, bool>,
}

impl TopologyChoices {
    pub(in crate::devices) fn is_empty(&self) -> bool {
        self.voltage.is_empty() && self.winding.is_empty()
    }

    pub(in crate::devices) fn check_append(&self, graph: &ReferenceGraph) -> SpiceResult<()> {
        for (choices, references) in [
            (&self.voltage, &graph.controls),
            (&self.winding, &graph.coupled),
        ] {
            if let Some(name) = references
                .iter()
                .find(|name| choices.get(*name) == Some(&false))
            {
                return Err(SpiceError::circuit(format!(
                    "electrical reference changes deferred topology of {name}; re-elaborate the deck"
                )));
            }
        }
        Ok(())
    }

    pub(in crate::devices) fn check_device(
        &self,
        device: &dyn crate::devices::Device,
    ) -> SpiceResult<()> {
        if self.is_empty() {
            return Ok(());
        }
        let mut graph = ReferenceGraph::default();
        graph.include_device(device);
        self.check_append(&graph)
    }

    pub(in crate::devices) fn check_replacement(
        &self,
        previous: &dyn crate::devices::Device,
        replacement: &dyn crate::devices::Device,
    ) -> SpiceResult<()> {
        if self.is_empty() {
            return Ok(());
        }
        self.check_device(replacement)?;
        let name = previous.name().to_ascii_lowercase();
        let mut old = ReferenceGraph::default();
        old.include_device(previous);
        let mut new = ReferenceGraph::default();
        new.include_device(replacement);
        if self.voltage.contains_key(&name)
            || self.winding.contains_key(&name)
            || old
                .controls
                .difference(&new.controls)
                .any(|name| self.voltage.contains_key(name))
            || old
                .coupled
                .difference(&new.coupled)
                .any(|name| self.winding.contains_key(name))
        {
            return Err(SpiceError::circuit(
                "replacement changes a deferred network recipe or its reference graph; re-elaborate the deck",
            ));
        }
        Ok(())
    }

    pub(in crate::devices) fn extend(&mut self, choices: Self) {
        self.voltage.extend(choices.voltage);
        self.winding.extend(choices.winding);
    }
}

impl ReferenceGraph {
    pub(in crate::devices) fn include_device(&mut self, device: &dyn crate::devices::Device) {
        self.controls.extend(
            device
                .controlling_sources()
                .iter()
                .map(|reference| reference.name.to_ascii_lowercase()),
        );
        if let Some(coupling) = device.mutual_coupling() {
            self.coupled.extend(
                coupling
                    .inductors
                    .iter()
                    .map(|reference| reference.name.to_ascii_lowercase()),
            );
        }
    }

    pub(in crate::devices) fn choices(&self, instances: &[DeviceInstance]) -> TopologyChoices {
        let mut choices = TopologyChoices::default();
        for instance in instances {
            let name = instance.name.to_ascii_lowercase();
            for parameter in &instance.parameters {
                match parameter.name.as_str() {
                    "pn_ccontrol" => {
                        choices
                            .voltage
                            .insert(name.clone(), self.controls.contains(&name));
                    }
                    "pn_rseries_uncoupled" => {
                        choices
                            .winding
                            .insert(name.clone(), self.coupled.contains(&name));
                    }
                    _ => {}
                }
            }
        }
        choices
    }

    pub(in crate::devices) fn has_selectors(instances: &[DeviceInstance]) -> bool {
        instances.iter().any(|instance| {
            instance
                .parameters
                .iter()
                .any(|p| matches!(p.name.as_str(), "pn_ccontrol" | "pn_rseries_uncoupled"))
        })
    }

    pub(in crate::devices) fn collect(instances: &[DeviceInstance], required: bool) -> Self {
        let mut graph = Self::default();
        if !required {
            return graph;
        }
        for instance in instances {
            for parameter in &instance.parameters {
                match &parameter.kind {
                    ParameterKind::Instance => {
                        let targets = if instance.designator == 'k' {
                            &mut graph.coupled
                        } else {
                            &mut graph.controls
                        };
                        targets.insert(parameter.value.to_ascii_lowercase());
                    }
                    ParameterKind::Behavioural(expression) => {
                        graph.controls.extend(
                            expression
                                .root
                                .current_references()
                                .into_iter()
                                .map(str::to_ascii_lowercase),
                        );
                    }
                    _ => {}
                }
            }
            graph.controls.extend(
                instance
                    .code_ports
                    .iter()
                    .flat_map(|port| &port.terminals)
                    .filter(|terminal| terminal.kind == CodePortKind::BranchCurrent)
                    .map(|terminal| terminal.positive.to_ascii_lowercase()),
            );
        }
        graph
    }

    pub(super) fn select(
        &self,
        instance: &DeviceInstance,
        values: &mut BTreeMap<&str, Real>,
    ) -> SpiceResult<()> {
        let invalid = |message| SpiceError::parse(instance.location.clone(), message);
        let name = instance.name.to_ascii_lowercase();
        if let Some(capacitance) = values.remove("pn_ccontrol") {
            if values.contains_key("pn_cparallel") || values.contains_key("pn_cseries_shunt") {
                return Err(invalid(
                    "control-dependent capacitance conflicts with explicit topology",
                ));
            }
            let key = if self.controls.contains(&name) {
                "pn_cseries_shunt"
            } else {
                "pn_cparallel"
            };
            values.insert(key, capacitance);
        }
        if let Some(resistance) = values.remove("pn_rseries_uncoupled") {
            if values.contains_key("pn_rseries") {
                return Err(invalid(
                    "uncoupled series resistance conflicts with explicit series resistance",
                ));
            }
            values.insert(
                "pn_rseries",
                if self.coupled.contains(&name) {
                    0.
                } else {
                    resistance
                },
            );
        }
        if let Some(extra) = values.remove("pn_terminal_bias_zero_series") {
            let zero_series = values.get("pn_rseries").copied().unwrap_or(0.) == 0.;
            let mut selected = false;
            for key in [
                "pn_terminal_bias_op",
                "pn_terminal_bias_dc",
                "pn_terminal_bias_ac",
                "pn_terminal_bias_tran",
            ] {
                if let Some(g) = values.get_mut(key) {
                    selected = true;
                    if zero_series {
                        *g += extra;
                        if !g.is_finite() {
                            return Err(invalid("nonfinite selected terminal bias conductance"));
                        }
                    }
                }
            }
            if !selected {
                return Err(invalid(
                    "zero-series bias correction requires a terminal bias equation",
                ));
            }
        }
        Ok(())
    }
}
