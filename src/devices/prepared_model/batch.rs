//! Bounded homogeneous prepared-instance tiles. The circuit keeps the
//! original device stamp order; only constitutive evaluation is grouped.

use std::{collections::BTreeMap, sync::Arc};

use model_kernel_api::{
    Artifact, Kernel, Point,
    batch::{BatchOutput, PreparedTile},
};

use super::{PreparedDevice, evaluation_error, kernel_error};
use crate::devices::traits::{Device, MnaUnknowns};
use crate::maths::Vector;
use crate::primitives::{Real, SpiceError, SpiceResult};

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
    fn evaluate(&mut self, solution: &Vector) -> SpiceResult<()>;

    /// Borrow one validated output lane without allocating.
    /// # Errors
    /// Lane ordinal out of range.
    fn point(&mut self, lane: usize) -> SpiceResult<Point<'_>>;
}

pub(super) struct BoundTile<K: Kernel> {
    prepared: PreparedTile<K>,
    rows: Vec<(Option<usize>, Option<Real>)>,
    coordinates: Vec<Real>,
    output: BatchOutput,
}

impl<K: Kernel> BoundTile<K> {
    pub(super) fn new(
        artifact: &Arc<Artifact<K>>,
        instances: &[&dyn Device],
        unknowns: &MnaUnknowns,
    ) -> SpiceResult<Self> {
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
            for (&node, equation) in device
                .terminals
                .iter()
                .zip(&artifact.descriptor().equations)
            {
                let offset = (equation.kind == model_kernel_api::EquationKind::Thermal)
                    .then_some(device.temperature.circuit_celsius + super::CELSIUS_TO_KELVIN);
                rows.push((unknowns.checked_node_row(node)?, offset));
            }
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
    fn evaluate(&mut self, solution: &Vector) -> SpiceResult<()> {
        for (value, &(row, offset)) in self.coordinates.iter_mut().zip(&self.rows) {
            let coordinate = match row {
                Some(row) => solution.get(row).ok_or_else(|| {
                    SpiceError::circuit("prepared tile solution dimension mismatch")
                })?,
                None => 0.,
            };
            *value = offset.map_or(coordinate, |offset| coordinate + offset);
        }
        self.prepared
            .evaluate(&self.coordinates, &mut self.output)
            .map_err(evaluation_error)
    }

    fn point(&mut self, lane: usize) -> SpiceResult<Point<'_>> {
        self.output.point(lane).map_err(kernel_error)
    }
}

/// Inputs that determine a lane's preparation and thermal row offset.
pub struct Recipe<'a> {
    pub(super) key: Key,
    pub(super) values: &'a [Real],
    pub(super) ambient: Real,
    pub(super) circuit_temperature: Real,
}

struct Lane {
    index: usize,
    key: Key,
    values: Vec<u64>,
    ambient: u64,
    circuit_temperature: u64,
}

impl Lane {
    fn new(index: usize, device: &dyn Device) -> SpiceResult<Self> {
        let recipe = device
            .prepared_recipe()
            .ok_or_else(|| SpiceError::circuit("missing prepared lane recipe"))?;
        Ok(Self {
            index,
            key: recipe.key,
            values: recipe.values.iter().map(|value| value.to_bits()).collect(),
            ambient: recipe.ambient.to_bits(),
            circuit_temperature: recipe.circuit_temperature.to_bits(),
        })
    }

    fn matches(&self, device: &dyn Device) -> bool {
        device.prepared_recipe().is_some_and(|recipe| {
            self.key == recipe.key
                && self.ambient == recipe.ambient.to_bits()
                && self.circuit_temperature == recipe.circuit_temperature.to_bits()
                && self
                    .values
                    .iter()
                    .copied()
                    .eq(recipe.values.iter().map(|value| value.to_bits()))
        })
    }
}

struct Group {
    lanes: Vec<Lane>,
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
        unknowns: &MnaUnknowns,
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
                let prepared = instances[0].prepare_batch(&instances, unknowns)?;
                for (lane, &(index, _)) in tile.iter().enumerate() {
                    plan.bindings[index] = Some((group, lane));
                }
                let lanes = tile
                    .iter()
                    .map(|&(index, device)| Lane::new(index, device))
                    .collect::<SpiceResult<_>>()?;
                plan.groups.push(Group {
                    lanes,
                    tile: prepared,
                    evaluated: false,
                });
            }
        }
        Ok(plan)
    }

    /// Reuse unchanged groups across context transitions. Construct every
    /// affected tile before committing any replacement, so a failed
    /// preparation leaves the previous plan usable.
    pub(crate) fn rebind<'a>(
        &mut self,
        device: impl Fn(usize) -> &'a dyn Device,
        unknowns: &MnaUnknowns,
    ) -> SpiceResult<()> {
        let mut replacements = Vec::new();
        for (index, group) in self.groups.iter().enumerate() {
            if group
                .lanes
                .iter()
                .all(|lane| lane.matches(device(lane.index)))
            {
                continue;
            }
            let instances: Vec<_> = group.lanes.iter().map(|lane| device(lane.index)).collect();
            let lanes = group
                .lanes
                .iter()
                .zip(&instances)
                .map(|(lane, &instance)| Lane::new(lane.index, instance))
                .collect::<SpiceResult<_>>()?;
            let tile = instances[0].prepare_batch(&instances, unknowns)?;
            replacements.push((
                index,
                Group {
                    lanes,
                    tile,
                    evaluated: false,
                },
            ));
        }
        for (index, group) in replacements {
            self.groups[index] = group;
        }
        self.begin();
        Ok(())
    }

    pub(crate) fn begin(&mut self) {
        for group in &mut self.groups {
            group.evaluated = false;
        }
    }

    pub(crate) fn point(
        &mut self,
        device: usize,
        solution: &Vector,
    ) -> SpiceResult<Option<Point<'_>>> {
        let Some((group, lane)) = self.bindings.get(device).copied().flatten() else {
            return Ok(None);
        };
        let group = &mut self.groups[group];
        if !group.evaluated {
            group.tile.evaluate(solution)?;
            group.evaluated = true;
        }
        group.tile.point(lane).map(Some)
    }
}
