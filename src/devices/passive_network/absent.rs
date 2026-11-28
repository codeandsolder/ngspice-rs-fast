//! Source-visible zero-multiplicity storage, with no physical MNA objects.

use std::{borrow::Cow, collections::BTreeSet};

use crate::{
    netlist::{
        ast::{CodePortKind, DeviceInstance, ParameterKind},
        bexpr::BExprKind,
    },
    primitives::{NodeTable, SpiceError, SpiceResult},
};

use super::NetworkPlan;

/// Names and written terminals survive removing a physical storage network.
/// Only terminals which have no real node in the circuit observe zero.
#[derive(Default, Clone)]
pub(in crate::devices) struct AbsentStorage {
    pub(in crate::devices) names: BTreeSet<String>,
    pub(in crate::devices) terminals: BTreeSet<String>,
    zero_voltage_reads: BTreeSet<String>,
}

impl AbsentStorage {
    pub(in crate::devices) fn contains(&self, name: &str) -> bool {
        self.names.contains(&name.to_lowercase())
    }

    pub(in crate::devices) fn isolated<'a>(
        &'a self,
        nodes: &'a NodeTable,
    ) -> impl Iterator<Item = &'a str> {
        self.terminals
            .iter()
            .filter(|name| nodes.get(name).is_none())
            .map(String::as_str)
    }

    /// An incremental edit cannot silently invalidate an already compiled
    /// observation-only voltage read. A new complete source elaboration can.
    pub(in crate::devices) fn validate_voltage_binding(
        &self,
        nodes: &NodeTable,
    ) -> SpiceResult<()> {
        if let Some(name) = self
            .zero_voltage_reads
            .iter()
            .find(|name| nodes.get(name).is_some())
        {
            return Err(SpiceError::circuit(format!(
                "connecting observation-only node {name} requires source re-elaboration"
            )));
        }
        Ok(())
    }

    /// Partition only validated, explicitly prepared zero-copy networks.
    /// Work on a staged copy; no names or electrical objects are published
    /// if subsequent expression expansion or circuit elaboration fails.
    pub(in crate::devices) fn partition<'a>(
        &mut self,
        instances: &'a [DeviceInstance],
        nodes: &NodeTable,
        graph: &super::topology::ReferenceGraph,
    ) -> SpiceResult<Cow<'a, [DeviceInstance]>> {
        let mut removed = BTreeSet::new();
        for (index, instance) in instances.iter().enumerate() {
            if !NetworkPlan::is_removed(instance, graph)? {
                continue;
            }
            if !self.names.insert(instance.name.to_lowercase()) {
                return Err(SpiceError::parse(
                    instance.location.clone(),
                    format!("duplicate instance name '{}'", instance.name),
                ));
            }
            self.terminals.extend(
                instance
                    .nodes
                    .iter()
                    .map(|name| NodeTable::canonical_name(name, nodes.auto_gnd())),
            );
            removed.insert(index);
        }
        if self.names.is_empty() {
            return Ok(Cow::Borrowed(instances));
        }

        // A written terminal shared with any surviving instance is not
        // observation-only, regardless of deck order or hierarchy.
        let mut isolated = self.isolated(nodes).collect::<BTreeSet<_>>();
        for (index, instance) in instances.iter().enumerate() {
            if removed.contains(&index) {
                continue;
            }
            for name in instance.nodes.iter().chain(
                instance
                    .code_ports
                    .iter()
                    .flat_map(|port| &port.terminals)
                    .filter(|terminal| {
                        !matches!(
                            terminal.kind,
                            CodePortKind::Digital | CodePortKind::BranchCurrent
                        )
                    })
                    .flat_map(|terminal| {
                        std::iter::once(&terminal.positive).chain(terminal.negative.iter())
                    }),
            ) {
                isolated.remove(NodeTable::canonical_name(name, nodes.auto_gnd()).as_str());
            }
        }

        let isolated: BTreeSet<String> = isolated.into_iter().map(str::to_owned).collect();
        let mut zero_voltage_reads = BTreeSet::new();
        // Rewrite resolved behavioral reads before measurement expansion:
        // no zero-volt sensor or artificial branch is allowed to be created.
        let mut result = Vec::new();
        result
            .try_reserve_exact(instances.len() - removed.len())
            .map_err(|error| {
                SpiceError::circuit(format!("absent storage partition allocation: {error}"))
            })?;
        for (index, instance) in instances.iter().enumerate() {
            if removed.contains(&index) {
                continue;
            }
            let mut instance = instance.clone();
            for parameter in &mut instance.parameters {
                if let ParameterKind::Behavioural(expression) = &mut parameter.kind {
                    expression.root.visit_mut(&mut |expr| match &mut expr.kind {
                        BExprKind::Current(name) if self.contains(name) => {
                            expr.kind = BExprKind::Number {
                                value: 0.,
                                spelling: "0".into(),
                            };
                        }
                        BExprKind::Voltage { positive, negative } => {
                            let mut zero = |name: &mut String| {
                                let canonical = NodeTable::canonical_name(name, nodes.auto_gnd());
                                if isolated.contains(&canonical) {
                                    zero_voltage_reads.insert(canonical);
                                    *name = "0".into();
                                }
                            };
                            zero(positive);
                            if let Some(negative) = negative {
                                zero(negative);
                            }
                        }
                        _ => {}
                    });
                }
            }
            result.push(instance);
        }
        self.zero_voltage_reads.extend(zero_voltage_reads);
        Ok(Cow::Owned(result))
    }
}
