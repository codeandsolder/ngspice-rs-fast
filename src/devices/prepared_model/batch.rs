//! Bounded homogeneous prepared-instance tiles. The circuit keeps the
//! original device stamp order; only constitutive evaluation is grouped.

use std::{collections::BTreeMap, sync::Arc};

use model_kernel_api::{
    Artifact, Kernel, Point,
    batch::{BatchOutput, PreparedTile},
};

use super::{PreparedDevice, kernel_error};
use crate::devices::traits::{Device, MnaUnknowns};
use crate::maths::Vector;
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

/// Exact identity of an alive immutable artifact. It is neither a persisted
/// fingerprint nor a substitute for the producer's identity checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key(usize);

impl Key {
    pub(super) fn new<K: Kernel>(artifact: &Arc<Artifact<K>>) -> Self {
        Self(Arc::as_ptr(artifact).cast::<()>() as usize)
    }
}

/// Type-erased numerical workspace, built once from effective instances.
pub trait Tile {
    /// Evaluate every lane at the same circuit Newton/bias point.
    /// # Errors
    /// Invalid coordinates or family evaluation.
    fn evaluate(&mut self, unknowns: &MnaUnknowns, solution: &Vector) -> SpiceResult<()>;

    /// Borrow one validated output lane without allocating.
    /// # Errors
    /// Lane ordinal out of range.
    fn point(&mut self, lane: usize) -> SpiceResult<Point<'_>>;
}

pub(super) struct BoundTile<K: Kernel> {
    prepared: PreparedTile<K>,
    rows: Vec<(NodeId, Real)>,
    coordinates: Vec<Real>,
    output: BatchOutput,
}

impl<K: Kernel> BoundTile<K> {
    pub(super) fn new(artifact: &Arc<Artifact<K>>, instances: &[&dyn Device]) -> SpiceResult<Self> {
        let width = artifact.descriptor().equations.len();
        let mut candidates = Vec::with_capacity(instances.len());
        let mut rows = Vec::new();
        for instance in instances {
            let device = (*instance as &dyn std::any::Any)
                .downcast_ref::<PreparedDevice<K>>()
                .ok_or_else(|| SpiceError::circuit("foreign prepared batch family"))?;
            if !Arc::ptr_eq(artifact, &device.artifact) {
                return Err(SpiceError::circuit("foreign prepared batch artifact"));
            }
            candidates.push((device.values.as_ref(), device.temperature.ambient_kelvin()));
            rows.extend(
                device
                    .terminals
                    .iter()
                    .zip(&artifact.descriptor().equations)
                    .map(|(&node, equation)| {
                        let offset = if equation.kind == model_kernel_api::EquationKind::Thermal {
                            device.temperature.circuit_celsius + super::CELSIUS_TO_KELVIN
                        } else {
                            0.
                        };
                        (node, offset)
                    }),
            );
        }
        let prepared = PreparedTile::new(artifact, &candidates).map_err(kernel_error)?;
        let output = BatchOutput::new(width, instances.len()).map_err(kernel_error)?;
        let coordinates = vec![0.; rows.len()];
        Ok(Self {
            prepared,
            rows,
            coordinates,
            output,
        })
    }
}

impl<K: Kernel> Tile for BoundTile<K> {
    fn evaluate(&mut self, unknowns: &MnaUnknowns, solution: &Vector) -> SpiceResult<()> {
        for (value, &(node, offset)) in self.coordinates.iter_mut().zip(&self.rows) {
            *value = unknowns
                .node_row(node)
                .and_then(|row| solution.get(row))
                .unwrap_or(0.)
                + offset;
        }
        self.prepared
            .evaluate(&self.coordinates, &mut self.output)
            .map_err(kernel_error)
    }

    fn point(&mut self, lane: usize) -> SpiceResult<Point<'_>> {
        self.output.point(lane).map_err(kernel_error)
    }
}

struct Group {
    tile: Box<dyn Tile>,
    evaluated: bool,
}

/// A small tile caps scratch footprint and allows kernels to vectorize without
/// assuming that a whole model family fits in one contiguous allocation.
const TILE_LANES: usize = 64;

#[derive(Default)]
pub(crate) struct Plan {
    bindings: Vec<Option<(usize, usize)>>,
    groups: Vec<Group>,
}

impl Plan {
    pub(crate) fn new<'a>(
        device_count: usize,
        devices: impl Iterator<Item = (usize, &'a dyn Device)>,
    ) -> SpiceResult<Self> {
        let mut families: BTreeMap<Key, Vec<(usize, &dyn Device)>> = BTreeMap::new();
        for (index, device) in devices {
            if let Some(key) = device.prepared_batch_key() {
                families.entry(key).or_default().push((index, device));
            }
        }
        let mut plan = Self::default();
        for family in families.into_values() {
            for tile in family.chunks(TILE_LANES).filter(|tile| tile.len() > 1) {
                if plan.bindings.is_empty() {
                    plan.bindings.resize(device_count, None);
                }
                let instances: Vec<_> = tile.iter().map(|&(_, device)| device).collect();
                let group = plan.groups.len();
                let prepared = instances[0].prepare_batch(&instances)?;
                for (lane, &(index, _)) in tile.iter().enumerate() {
                    plan.bindings[index] = Some((group, lane));
                }
                plan.groups.push(Group {
                    tile: prepared,
                    evaluated: false,
                });
            }
        }
        Ok(plan)
    }

    pub(crate) fn begin(&mut self) {
        for group in &mut self.groups {
            group.evaluated = false;
        }
    }

    pub(crate) fn point(
        &mut self,
        device: usize,
        unknowns: &MnaUnknowns,
        solution: &Vector,
    ) -> SpiceResult<Option<Point<'_>>> {
        let Some((group, lane)) = self.bindings.get(device).copied().flatten() else {
            return Ok(None);
        };
        let group = &mut self.groups[group];
        if !group.evaluated {
            group.tile.evaluate(unknowns, solution)?;
            group.evaluated = true;
        }
        group.tile.point(lane).map(Some)
    }
}
