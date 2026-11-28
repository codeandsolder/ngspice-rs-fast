//! Bounded homogeneous prepared-instance tiles. The circuit keeps the
//! original device stamp order; only constitutive evaluation is grouped.

use std::{
    cell::{RefCell, RefMut},
    collections::BTreeMap,
    rc::Rc,
    sync::Arc,
};

use model_kernel_api::{
    Artifact, Kernel, Point,
    batch::{BatchOutput, LaneTransaction, LaneUpdate, PreparedTile},
    trial::{Convergence, ConvergenceInput, TrialInput, TrialOutcome},
};

use super::{Physical, PreparedDevice, evaluation_error, kernel_error};
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

/// A constitutive lane paired with its actual evaluation coordinates.
pub struct EvaluatedLane<'a> {
    /// Current, charge, and derivatives at the evaluation point.
    pub point: Point<'a>,
    /// Coordinates used to construct the affine Newton source.
    pub coordinates: &'a [Real],
}

/// One typed, contiguous physical tile. Shared only by the run-local
/// lane views and its homogeneous numerical workspace.
#[derive(Debug)]
struct SharedTile<K: Kernel> {
    artifact: Arc<Artifact<K>>,
    data: PreparedTile<K>,
}

#[derive(Debug)]
pub(super) struct PreparedLane<K: Kernel> {
    tile: Rc<RefCell<SharedTile<K>>>,
    ordinal: usize,
}

impl<K: Kernel> PreparedLane<K> {
    pub(super) fn evaluate(
        &self,
        coordinates: &[Real],
        point: &mut Point<'_>,
    ) -> model_kernel_api::Result<()> {
        self.tile
            .borrow()
            .data
            .evaluate_lane(self.ordinal, coordinates, point)
    }
    pub(super) fn noise(
        &self,
        coordinates: &[Real],
        frequency: Real,
        sink: &mut dyn model_kernel_api::noise::Sink,
    ) -> model_kernel_api::Result<()> {
        self.tile
            .borrow()
            .data
            .evaluate_lane_noise(self.ordinal, coordinates, frequency, sink)
    }
    pub(super) fn linearize(
        &self,
        input: TrialInput<'_>,
        coordinates: &mut [Real],
    ) -> model_kernel_api::Result<TrialOutcome> {
        self.tile
            .borrow()
            .data
            .linearize_lane_trial(self.ordinal, input, coordinates)
    }
    pub(super) fn convergence(
        &self,
        input: ConvergenceInput<'_>,
    ) -> model_kernel_api::Result<Convergence> {
        self.tile
            .borrow()
            .data
            .check_lane_trial_convergence(self.ordinal, input)
    }
}

/// A fully validated heterogeneous tile update. The exclusive tile borrow
/// survives until commit or discard, while all other affected tiles stage.
pub trait StagedTile {
    /// Install already-prepared lanes; no physics or allocation occurs here.
    fn commit(self: Box<Self>);
}

impl<K: Kernel> StagedTile for LaneTransaction<'_, K> {
    fn commit(self: Box<Self>) {
        (*self).commit();
    }
}

/// Type erasure occurs at the tile boundary, never per physical lane.
pub trait PhysicalTile {
    /// Prepare all changed lanes before any mutation.
    /// # Errors
    /// Invalid candidate inputs or lane identifiers.
    fn stage(&mut self, updates: &[LaneUpdate<'_>]) -> SpiceResult<Box<dyn StagedTile + '_>>;
}

impl<K: Kernel> PhysicalTile for SharedTile<K> {
    fn stage(&mut self, updates: &[LaneUpdate<'_>]) -> SpiceResult<Box<dyn StagedTile + '_>> {
        Ok(Box::new(
            self.data
                .stage_replacements(&self.artifact, updates)
                .map_err(kernel_error)?,
        ))
    }
}

/// Prebound MNA row and optional absolute-coordinate offset per local coordinate.
pub type BoundRows = Vec<(Option<usize>, Option<Real>)>;

/// Type-erased numerical workspace, built once from effective instances.
pub trait Tile {
    /// Exclusively borrow the physical tile for a transactional replacement.
    fn physical(&self) -> RefMut<'_, dyn PhysicalTile>;

    /// Bind unpublished candidate views to the exact typed lanes.
    /// # Errors
    /// Foreign artifacts or incompatible lanes.
    fn bind(&self, instances: &[&dyn Device]) -> SpiceResult<()>;

    /// Resolve new row offsets before committing any physical replacement.
    /// # Errors
    /// Foreign artifacts or invalid topology.
    fn rows_for(&self, instances: &[&dyn Device], unknowns: &MnaUnknowns)
    -> SpiceResult<BoundRows>;

    /// Install already-resolved row offsets after all staged updates commit.
    fn replace_rows(&mut self, rows: BoundRows);

    /// Evaluate every lane at the same circuit Newton/bias point.
    /// # Errors
    /// Invalid coordinates or family evaluation.
    fn evaluate(
        &mut self,
        solution: &Vector,
        linearize: &mut dyn FnMut(usize, &mut [Real]) -> SpiceResult<()>,
    ) -> SpiceResult<()>;

    /// Borrow one validated output lane without allocating.
    /// # Errors
    /// Lane ordinal out of range.
    fn point(&mut self, lane: usize) -> SpiceResult<EvaluatedLane<'_>>;
}

pub(super) struct BoundTile<K: Kernel> {
    physical: Rc<RefCell<SharedTile<K>>>,
    width: usize,
    rows: BoundRows,
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
        let rows = Self::bound_rows(artifact, instances, unknowns)?;
        let data = instances
            .iter()
            .map(|instance| Self::typed(artifact, *instance)?.take_prepared())
            .collect::<SpiceResult<Vec<_>>>()?;
        let prepared = PreparedTile::from_prepared(artifact, data).map_err(kernel_error)?;
        let output = BatchOutput::new(width, instances.len()).map_err(kernel_error)?;
        let coordinates = vec![0.; rows.len()];
        let tile = Self {
            physical: Rc::new(RefCell::new(SharedTile {
                artifact: Arc::clone(artifact),
                data: prepared,
            })),
            width,
            rows,
            coordinates,
            output,
        };
        tile.bind(instances)?;
        Ok(tile)
    }

    fn typed<'a>(
        artifact: &Arc<Artifact<K>>,
        instance: &'a dyn Device,
    ) -> SpiceResult<&'a PreparedDevice<K>> {
        let device = (instance as &dyn std::any::Any)
            .downcast_ref::<PreparedDevice<K>>()
            .ok_or_else(|| SpiceError::circuit("foreign prepared batch family"))?;
        if !Arc::ptr_eq(artifact, &device.artifact) {
            return Err(SpiceError::circuit("foreign prepared batch artifact"));
        }
        Ok(device)
    }

    fn bound_rows(
        artifact: &Arc<Artifact<K>>,
        instances: &[&dyn Device],
        unknowns: &MnaUnknowns,
    ) -> SpiceResult<BoundRows> {
        let mut rows = Vec::new();
        for instance in instances {
            let device = Self::typed(artifact, *instance)?;
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
        Ok(rows)
    }
}

impl<K: Kernel> Tile for BoundTile<K> {
    fn physical(&self) -> RefMut<'_, dyn PhysicalTile> {
        RefMut::map(self.physical.borrow_mut(), |tile| {
            tile as &mut dyn PhysicalTile
        })
    }

    fn bind(&self, instances: &[&dyn Device]) -> SpiceResult<()> {
        let tile = self.physical.borrow();
        if instances.len() != tile.data.lanes() {
            return Err(SpiceError::circuit("prepared lane membership changed"));
        }
        for (ordinal, instance) in instances.iter().enumerate() {
            let device = Self::typed(&tile.artifact, *instance)?;
            *device.physical.borrow_mut() = Physical::Lane(PreparedLane {
                tile: Rc::clone(&self.physical),
                ordinal,
            });
        }
        Ok(())
    }

    fn rows_for(
        &self,
        instances: &[&dyn Device],
        unknowns: &MnaUnknowns,
    ) -> SpiceResult<BoundRows> {
        Self::bound_rows(&self.physical.borrow().artifact, instances, unknowns)
    }

    fn replace_rows(&mut self, rows: BoundRows) {
        self.rows = rows;
    }

    fn evaluate(
        &mut self,
        solution: &Vector,
        linearize: &mut dyn FnMut(usize, &mut [Real]) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        for (value, &(row, offset)) in self.coordinates.iter_mut().zip(&self.rows) {
            let coordinate = match row {
                Some(row) => solution.get(row).ok_or_else(|| {
                    SpiceError::circuit("prepared tile solution dimension mismatch")
                })?,
                None => 0.,
            };
            *value = offset.map_or(coordinate, |offset| coordinate + offset);
        }
        for (lane, coordinates) in self.coordinates.chunks_exact_mut(self.width).enumerate() {
            linearize(lane, coordinates)?;
        }
        self.physical
            .borrow()
            .data
            .evaluate(&self.coordinates, &mut self.output)
            .map_err(evaluation_error)
    }

    fn point(&mut self, lane: usize) -> SpiceResult<EvaluatedLane<'_>> {
        let point = self.output.point(lane).map_err(kernel_error)?;
        Ok(EvaluatedLane {
            point,
            coordinates: &self.coordinates[lane * self.width..(lane + 1) * self.width],
        })
    }
}

/// Inputs that determine a lane's preparation and thermal row offset.
pub struct Recipe<'a> {
    pub(super) key: Key,
    pub(super) values: &'a [Real],
    pub(super) ambient: Real,
}

struct Lane {
    index: usize,
    key: Key,
    values: Vec<u64>,
    ambient: u64,
    needs_linearization: bool,
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
            needs_linearization: device.needs_prepared_linearization(),
        })
    }

    fn same_physics(&self, recipe: &Recipe<'_>) -> bool {
        self.key == recipe.key
            && self.ambient == recipe.ambient.to_bits()
            && self
                .values
                .iter()
                .copied()
                .eq(recipe.values.iter().map(|value| value.to_bits()))
    }
}

struct Group {
    lanes: Vec<Lane>,
    tile: Box<dyn Tile>,
    evaluated: bool,
}

struct GroupUpdate<'a> {
    lanes: Vec<Lane>,
    rows: BoundRows,
    changed: Vec<LaneUpdate<'a>>,
}

impl<'a> GroupUpdate<'a> {
    fn new(
        group: &Group,
        device: &impl Fn(usize) -> &'a dyn Device,
        unknowns: &MnaUnknowns,
    ) -> SpiceResult<Self> {
        let instances: Vec<_> = group.lanes.iter().map(|lane| device(lane.index)).collect();
        let rows = group.tile.rows_for(&instances, unknowns)?;
        let mut lanes = Vec::with_capacity(instances.len());
        let mut changed = Vec::new();
        for (ordinal, (old, instance)) in group.lanes.iter().zip(&instances).enumerate() {
            let recipe = instance
                .prepared_recipe()
                .ok_or_else(|| SpiceError::circuit("missing prepared lane recipe"))?;
            if !old.same_physics(&recipe) {
                changed.push(LaneUpdate {
                    lane: ordinal,
                    values: recipe.values,
                    ambient_kelvin: recipe.ambient,
                });
            }
            lanes.push(Lane::new(old.index, *instance)?);
        }
        group.tile.bind(&instances)?;
        Ok(Self {
            lanes,
            rows,
            changed,
        })
    }
}

/// A small tile caps scratch footprint and allows kernels to vectorize without
/// assuming that a whole model family fits in one contiguous allocation.
const TILE_LANES: usize = 64;

#[derive(Default)]
pub(crate) struct Plan {
    device_count: usize,
    bindings: Vec<Option<(usize, usize)>>,
    groups: Vec<Group>,
    singletons: Vec<(usize, Key)>,
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
        let mut plan = Self {
            device_count,
            ..Self::default()
        };
        for family in families.into_values() {
            for tile in family.chunks(TILE_LANES) {
                if let [(index, device)] = tile {
                    device.activate_prepared_scalar()?;
                    let key = device
                        .prepared_batch_key()
                        .ok_or_else(|| SpiceError::circuit("missing singleton artifact"))?;
                    plan.singletons.push((*index, key));
                    continue;
                }
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

    /// Rebind every unpublished effective device to one physical owner.
    /// All changed physical lanes and row offsets stage before any commit.
    /// On failure the previous complete candidate remains usable.
    pub(crate) fn rebind<'a>(
        &mut self,
        device: impl Fn(usize) -> &'a dyn Device,
        unknowns: &MnaUnknowns,
    ) -> SpiceResult<()> {
        let members = self
            .groups
            .iter()
            .flat_map(|group| group.lanes.iter().map(|lane| (lane.index, lane.key)))
            .chain(self.singletons.iter().copied());
        if members
            .clone()
            .any(|(index, key)| device(index).prepared_batch_key() != Some(key))
        {
            let next = Self::new(
                self.device_count,
                unknowns,
                members.map(|(index, _)| (index, device(index))),
            )?;
            *self = next;
            return Ok(());
        }
        for &(index, _) in &self.singletons {
            device(index).activate_prepared_scalar()?;
        }
        let updates = self
            .groups
            .iter()
            .map(|group| GroupUpdate::new(group, &device, unknowns))
            .collect::<SpiceResult<Vec<_>>>()?;
        // Each borrow is exclusive to a disjoint typed tile. The entire
        // collection of transactions must stage successfully before commit.
        let mut physical = self
            .groups
            .iter()
            .map(|group| group.tile.physical())
            .collect::<Vec<_>>();
        let staged = physical
            .iter_mut()
            .zip(&updates)
            .map(|(tile, update)| {
                if update.changed.is_empty() {
                    Ok(None)
                } else {
                    tile.stage(&update.changed).map(Some)
                }
            })
            .collect::<SpiceResult<Vec<_>>>()?;
        for transaction in staged.into_iter().flatten() {
            transaction.commit();
        }
        drop(physical);
        for (group, update) in self.groups.iter_mut().zip(updates) {
            group.lanes = update.lanes;
            group.tile.replace_rows(update.rows);
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
        mut linearize: impl FnMut(usize, &mut [Real]) -> SpiceResult<()>,
    ) -> SpiceResult<Option<EvaluatedLane<'_>>> {
        let Some((group, lane)) = self.bindings.get(device).copied().flatten() else {
            return Ok(None);
        };
        let group = &mut self.groups[group];
        if !group.evaluated {
            group.tile.evaluate(solution, &mut |lane, coordinates| {
                let lane = &group.lanes[lane];
                if lane.needs_linearization {
                    linearize(lane.index, coordinates)?;
                }
                Ok(())
            })?;
            group.evaluated = true;
        }
        group.tile.point(lane).map(Some)
    }
}
