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

impl ReferenceGraph {
    pub(in crate::devices) fn collect(instances: &[DeviceInstance]) -> Self {
        let mut graph = Self::default();
        if !instances.iter().any(|instance| {
            instance
                .parameters
                .iter()
                .any(|p| matches!(p.name.as_str(), "pn_ccontrol" | "pn_rseries_uncoupled"))
        }) {
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
