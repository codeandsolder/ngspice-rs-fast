//! Resource accounting for circuit elaboration, independent of MNA numbering.
use super::{
    models::{ModelContext, ModelResolver},
    txl::TxlPlan,
    urc::UrcPlan,
};
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{NodeId, NodeTable, Real, SourceLoc, SpiceError, SpiceResult};
use std::collections::BTreeSet;

/// Aggregate materialized circuit limits. A generated element counts just
/// like an ordinary instance; these limits also apply to programmatic edits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElaborationLimits {
    /// Maximum device records, including URC markers and generated devices.
    pub max_devices: usize,
    /// Maximum distinct non-ground nodes, including setup-private nodes.
    pub max_nodes: usize,
    /// Maximum declared device state slots (one scalar per slot).
    pub max_state_slots: usize,
}

impl Default for ElaborationLimits {
    fn default() -> Self {
        Self {
            max_devices: super::subckt::SubcircuitLimits::default().max_devices,
            max_nodes: 1_000_000,
            max_state_slots: 4_000_000,
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
            ("devices", size.devices, self.max_devices),
            ("non-ground nodes", size.nodes, self.max_nodes),
            ("state slots", size.states, self.max_state_slots),
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

pub(super) enum PlannedKind {
    Ordinary,
    Urc(UrcPlan),
    Txl(TxlPlan),
}

/// The generated plan owns validated setters and exact internal resource
/// counts; ordinary devices have a one-record lower bound until factory bind.
pub(super) struct PlannedInstance<'a> {
    pub(super) instance: &'a DeviceInstance,
    pub(super) kind: PlannedKind,
    size: ElaborationSize,
}

impl<'a> PlannedInstance<'a> {
    fn prepare(
        instance: &'a DeviceInstance,
        models: &ModelResolver<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Self> {
        let kind = match instance.designator {
            'u' => PlannedKind::Urc(UrcPlan::prepare(instance, models, context)?),
            'y' => PlannedKind::Txl(TxlPlan::prepare(instance, models, context)?),
            _ => PlannedKind::Ordinary,
        };
        let size = match &kind {
            PlannedKind::Ordinary => ElaborationSize::new(1, 0, 0),
            PlannedKind::Urc(plan) => plan.size(&instance.location)?,
            PlannedKind::Txl(plan) => plan.size(),
        };
        Ok(Self {
            instance,
            kind,
            size,
        })
    }

    pub(super) const fn is_exact(&self) -> bool {
        !matches!(self.kind, PlannedKind::Ordinary)
    }

    /// At bind time count only missing distinct terminals. Both generated
    /// plan kinds always intern their entire written terminal list.
    pub(super) fn size_at_bind(&self, nodes: &NodeTable) -> SpiceResult<ElaborationSize> {
        if !self.is_exact() {
            return Ok(self.size);
        }
        let missing = self
            .instance
            .nodes
            .iter()
            .filter(|name| nodes.get(name).is_none())
            .map(|name| NodeTable::canonical_name(name, nodes.auto_gnd()))
            .collect::<BTreeSet<_>>()
            .len();
        self.size.checked_add(
            ElaborationSize::new(0, missing, 0),
            Some(&self.instance.location),
        )
    }
}

/// Preflight all generated groups before materializing even the first ladder.
/// Includes one device per ordinary card, generated nodes/state and distinct
/// written terminals of the generated groups. Ordinary private node/state
/// sizes are checked as each factory binds.
pub(super) struct PlannedBatch<'a> {
    pub(super) instances: Vec<PlannedInstance<'a>>,
    pub(super) devices: usize,
    pub(super) deck_nodes: BTreeSet<String>,
}

impl<'a> PlannedBatch<'a> {
    pub(super) fn prepare(
        instances: &'a [DeviceInstance],
        models: &ModelResolver<'_>,
        context: &ModelContext,
        nodes: &NodeTable,
        existing: ElaborationSize,
        limits: ElaborationLimits,
    ) -> SpiceResult<Self> {
        let first = instances.first().map(|i| &i.location);
        let mut known = existing.checked_add(ElaborationSize::new(instances.len(), 0, 0), first)?;
        limits.check(known, first)?;
        let mut plans = Vec::new();
        reserve(&mut plans, instances.len(), first)?;
        let mut generated_terminals = BTreeSet::new();
        for instance in instances {
            let at = Some(&instance.location);
            let plan = PlannedInstance::prepare(instance, models, context)?;
            if plan.is_exact() {
                let mut extra = plan.size;
                extra.devices -= 1; // the minimum counted every input card already
                for name in &instance.nodes {
                    if nodes.get(name).is_none()
                        && generated_terminals
                            .insert(NodeTable::canonical_name(name, nodes.auto_gnd()))
                    {
                        extra = extra.checked_add(ElaborationSize::new(0, 1, 0), at)?;
                    }
                }
                known = known.checked_add(extra, at)?;
                limits.check(known, at)?;
            }
            plans.push(plan);
        }
        let deck_nodes = instances
            .iter()
            .flat_map(|i| &i.nodes)
            .map(|name| NodeTable::canonical_name(name, nodes.auto_gnd()))
            .collect();
        Ok(Self {
            instances: plans,
            devices: known.devices - existing.devices,
            deck_nodes,
        })
    }
}
