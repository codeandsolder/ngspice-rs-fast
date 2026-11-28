//! Resource accounting for circuit elaboration, independent of MNA numbering.
use super::{
    models::{ModelContext, ModelResolver},
    txl::TxlPlan,
    urc::UrcPlan,
};
use crate::netlist::ast::{CodePortKind, DeviceInstance};
use crate::primitives::{NodeId, NodeTable, Real, SourceLoc, SpiceError, SpiceResult};
use std::collections::BTreeSet;

/// Aggregate materialized circuit limits. A generated element counts just
/// like an ordinary instance; these limits also apply to programmatic edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElaborationLimits {
    /// Maximum device records, including URC markers and generated devices.
    pub devices: usize,
    /// Maximum distinct non-ground nodes, including setup-private nodes.
    pub nodes: usize,
    /// Maximum declared device state slots (one scalar per slot).
    pub state_slots: usize,
}

impl Default for ElaborationLimits {
    fn default() -> Self {
        Self {
            devices: super::subckt::SubcircuitLimits::default().max_devices,
            nodes: 1_000_000,
            state_slots: 4_000_000,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct ElaborationSize {
    pub(super) devices: usize,
    pub(super) nodes: usize,
    pub(super) states: usize,
}

pub(super) fn resource_error(
    location: Option<&SourceLoc>,
    message: impl std::fmt::Display,
) -> SpiceError {
    let message = format!("elaboration resource limit: {message}");
    match location {
        Some(at) => SpiceError::parse(at.clone(), message),
        None => SpiceError::circuit(message),
    }
}

impl ElaborationSize {
    pub(super) const fn new(devices: usize, nodes: usize, states: usize) -> Self {
        Self {
            devices,
            nodes,
            states,
        }
    }

    pub(super) fn checked_add(
        self,
        extra: Self,
        location: Option<&SourceLoc>,
    ) -> SpiceResult<Self> {
        let add = |a: usize, b| {
            a.checked_add(b)
                .ok_or_else(|| resource_error(location, "aggregate size overflows"))
        };
        Ok(Self {
            devices: add(self.devices, extra.devices)?,
            nodes: add(self.nodes, extra.nodes)?,
            states: add(self.states, extra.states)?,
        })
    }
}

impl ElaborationLimits {
    pub(super) fn check(
        self,
        size: ElaborationSize,
        location: Option<&SourceLoc>,
    ) -> SpiceResult<()> {
        for (name, current, limit) in [
            ("devices", size.devices, self.devices),
            ("non-ground nodes", size.nodes, self.nodes),
            ("state slots", size.states, self.state_slots),
        ] {
            if current > limit {
                return Err(resource_error(
                    location,
                    format!("aggregate {current} {name} exceed {limit}"),
                ));
            }
        }
        NodeId::from_index(size.nodes).map_err(|_| {
            resource_error(location, "node count exceeds compact node identity range")
        })?;
        std::alloc::Layout::array::<super::circuit::ElaboratedDevice>(size.devices)
            .map_err(|_| resource_error(location, "device storage allocation layout overflows"))?;
        let node_records = size
            .nodes
            .checked_add(1)
            .ok_or_else(|| resource_error(location, "node record count overflows"))?;
        std::alloc::Layout::array::<crate::primitives::Node>(node_records)
            .map_err(|_| resource_error(location, "node storage allocation layout overflows"))?;
        std::alloc::Layout::array::<Real>(size.states)
            .map_err(|_| resource_error(location, "state storage allocation layout overflows"))?;
        Ok(())
    }
}

/// Reserve countable staging storage fallibly before any publication.
pub(super) fn reserve<T>(
    entries: &mut Vec<T>,
    additional: usize,
    location: Option<&SourceLoc>,
) -> SpiceResult<()> {
    entries.try_reserve_exact(additional).map_err(|error| {
        resource_error(
            location,
            format!("cannot reserve {additional} entries: {error}"),
        )
    })
}

pub(super) enum ExpansionKind {
    Urc(UrcPlan),
    Txl(TxlPlan),
    Network(super::passive_network::NetworkPlan),
}

/// Only special expansion cards need a stored plan. Ordinary decks borrow
/// their original instance slice, with no per-card plan allocation.
pub(super) struct PlannedExpansion {
    pub(super) ordinal: usize,
    pub(super) kind: ExpansionKind,
    size: ElaborationSize,
}

impl PlannedExpansion {
    fn prepare(
        ordinal: usize,
        instance: &DeviceInstance,
        models: &ModelResolver<'_>,
        context: &ModelContext,
        graph: &super::passive_network::topology::ReferenceGraph,
    ) -> SpiceResult<Option<Self>> {
        let kind =
            if let Some(plan) = super::passive_network::NetworkPlan::prepare(instance, graph)? {
                ExpansionKind::Network(plan)
            } else {
                match instance.designator.to_ascii_lowercase() {
                    'u' => ExpansionKind::Urc(UrcPlan::prepare(instance, models, context)?),
                    'y' => ExpansionKind::Txl(TxlPlan::prepare(instance, models, context)?),
                    _ => return Ok(None),
                }
            };
        let size = match &kind {
            ExpansionKind::Urc(plan) => plan.size(&instance.location)?,
            ExpansionKind::Txl(plan) => plan.size(),
            ExpansionKind::Network(plan) => plan.size(),
        };
        Ok(Some(Self {
            ordinal,
            kind,
            size,
        }))
    }

    /// At bind time count only missing distinct terminals. Both generated
    /// plan kinds always intern their entire written terminal list.
    pub(super) fn size_at_bind(
        &self,
        instance: &DeviceInstance,
        nodes: &NodeTable,
    ) -> SpiceResult<ElaborationSize> {
        let missing = instance
            .nodes
            .iter()
            .filter(|name| nodes.get(name).is_none())
            .map(|name| NodeTable::canonical_name(name, nodes.auto_gnd()))
            .collect::<BTreeSet<_>>()
            .len();
        self.size.checked_add(
            ElaborationSize::new(0, missing, 0),
            Some(&instance.location),
        )
    }
}

/// Preflight all generated groups before materializing even the first ladder.
/// Includes one device per ordinary card, generated nodes/state and distinct
/// written terminals of the generated groups. Ordinary private node/state
/// sizes are checked as each factory binds.
pub(super) struct PlannedBatch<'a> {
    pub(super) instances: &'a [DeviceInstance],
    pub(super) expansions: Vec<PlannedExpansion>,
    pub(super) devices: usize,
    pub(super) deck_nodes: BTreeSet<String>,
    pub(super) resistors: Option<&'a super::pwl_resistor::BoundResistors>,
}

impl<'a> PlannedBatch<'a> {
    pub(super) fn prepare(
        instances: &'a [DeviceInstance],
        models: &ModelResolver<'_>,
        context: &ModelContext,
        nodes: &NodeTable,
        existing: ElaborationSize,
        limits: ElaborationLimits,
        graph: &super::passive_network::topology::ReferenceGraph,
    ) -> SpiceResult<Self> {
        let mut known = existing;
        let mut expansions = Vec::new();
        let mut generated_terminals = BTreeSet::new();
        for (ordinal, instance) in instances.iter().enumerate() {
            let at = Some(&instance.location);
            let Some(plan) = PlannedExpansion::prepare(ordinal, instance, models, context, graph)?
            else {
                known = known.checked_add(ElaborationSize::new(1, 0, 0), at)?;
                limits.check(known, at)?;
                continue;
            };
            let mut extra = plan.size;
            for name in &instance.nodes {
                if nodes.get(name).is_none()
                    && generated_terminals.insert(NodeTable::canonical_name(name, nodes.auto_gnd()))
                {
                    extra = extra.checked_add(ElaborationSize::new(0, 1, 0), at)?;
                }
            }
            known = known.checked_add(extra, at)?;
            limits.check(known, at)?;
            expansions.try_reserve(1).map_err(|error| {
                resource_error(at, format!("cannot reserve expansion plan: {error}"))
            })?;
            expansions.push(plan);
        }
        let deck_nodes = instances
            .iter()
            .filter(|instance| !super::behavioural::dynamic::is_generated(instance))
            .flat_map(|i| {
                i.nodes.iter().chain(
                    i.code_ports
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
                )
            })
            .map(|name| NodeTable::canonical_name(name, nodes.auto_gnd()))
            .collect();
        Ok(Self {
            instances,
            expansions,
            devices: known.devices - existing.devices,
            deck_nodes,
            resistors: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_dimensions_and_reserve_fail_without_wrapping() {
        let unrestricted = ElaborationLimits {
            devices: usize::MAX,
            nodes: usize::MAX,
            state_slots: usize::MAX,
        };
        for size in [
            ElaborationSize::new(usize::MAX, 0, 0),
            ElaborationSize::new(0, usize::MAX, 0),
            ElaborationSize::new(0, 0, usize::MAX),
        ] {
            assert!(
                size.checked_add(ElaborationSize::new(1, 1, 1), None)
                    .is_err()
            );
        }
        assert!(
            unrestricted
                .check(ElaborationSize::new(0, 0, usize::MAX), None)
                .is_err()
        );
        assert!(
            unrestricted
                .check(ElaborationSize::new(usize::MAX, 0, 0), None)
                .is_err()
        );
        if usize::BITS > u32::BITS {
            assert!(
                unrestricted
                    .check(ElaborationSize::new(0, usize::MAX, 0), None)
                    .is_err()
            );
        }
        let mut nodes = NodeTable::new();
        let retained = nodes.intern("keep").unwrap();
        let before = nodes.nodes().to_vec();
        let mut final_records = vec![retained];
        let result = nodes.transaction(|nodes| {
            nodes.intern("temporary")?;
            reserve(&mut final_records, usize::MAX, None)
        });
        assert!(result.unwrap_err().to_string().contains("reserve"));
        assert_eq!(nodes.nodes(), before);
        assert_eq!(final_records, [retained]);
    }
}
