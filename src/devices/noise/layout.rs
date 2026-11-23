//! Stable C visitation order, generator slots and transfer rows.
use std::collections::BTreeMap;
use std::ops::Range;

use super::{BoundTransfer, DeviceNoise, NoiseContext, NoiseFamily, NoiseKind, NoiseSource};
use crate::devices::circuit::PreparedPoint;
use crate::devices::{Circuit, MnaUnknowns, ModelContext, PreparedRun, StateSnapshot};
use crate::maths::Vector;
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult};

impl BoundTransfer {
    pub(crate) fn bind(unknowns: &MnaUnknowns, nodes: [NodeId; 2]) -> SpiceResult<Self> {
        Ok(Self {
            nodes,
            rows: [
                unknowns.checked_node_row(nodes[0])?,
                unknowns.checked_node_row(nodes[1])?,
            ],
        })
    }

    fn refresh(
        previous: Option<Self>,
        unknowns: &MnaUnknowns,
        nodes: [NodeId; 2],
    ) -> SpiceResult<Self> {
        match previous {
            Some(pair) if pair.nodes == nodes => Ok(pair),
            _ => Self::bind(unknowns, nodes),
        }
    }

    pub(crate) fn difference(self, solution: &[Complex]) -> Complex {
        let at = |row: Option<usize>| row.map_or(Complex::ZERO, |r| solution[r]);
        at(self.rows[0]) - at(self.rows[1])
    }

    pub(crate) const fn rows(self) -> [Option<usize>; 2] {
        self.rows
    }
}

/// Frequency-local law and bound transfer(s) of one generator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundNoiseSource {
    kind: NoiseKind,
    primary: BoundTransfer,
    secondary: Option<BoundTransfer>,
}

impl BoundNoiseSource {
    pub(crate) fn bind(
        source: NoiseSource,
        unknowns: &MnaUnknowns,
        previous: Option<Self>,
    ) -> SpiceResult<Self> {
        let primary =
            BoundTransfer::refresh(previous.map(|old| old.primary), unknowns, source.nodes)?;
        let secondary = match source.kind {
            NoiseKind::CorrelatedThermal { second, .. } => Some(BoundTransfer::refresh(
                previous.and_then(|old| old.secondary),
                unknowns,
                second,
            )?),
            _ => None,
        };
        Ok(Self {
            kind: source.kind,
            primary,
            secondary,
        })
    }

    pub(crate) fn output_density(self, adjoint: &[Complex], frequency: Real) -> SpiceResult<Real> {
        self.kind.output_density(
            self.primary.difference(adjoint),
            self.secondary.map(|pair| pair.difference(adjoint)),
            frequency,
        )
    }

    pub(crate) fn rf_amplitude(self) -> SpiceResult<Option<Real>> {
        self.kind.rf_amplitude()
    }

    pub(crate) fn transfer(self, adjoint: &[Complex]) -> Complex {
        self.primary.difference(adjoint)
    }
}

/// Immutable identity and ordered generator slots of a noisy instance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoiseInstance {
    /// Ordinal in the finalized circuit.
    pub device: usize,
    /// Instance name used by C's output columns.
    pub name: String,
    /// Reference family and naming convention.
    pub family: NoiseFamily,
    /// Model identity; None for the default model.
    pub model: Option<String>,
    /// Generator suffixes, in the device's output order.
    pub suffixes: Vec<&'static str>,
    /// Whether C reports the instance total after the generators.
    pub total: bool,
    range: Range<usize>,
}

/// The noise layout bound to one finalized circuit, with double-buffered
/// numeric slots.
///
/// Model ranks and names are constructed only during binding. Failed
/// refreshes leave the entire previous spectral point intact.
pub struct NoisePlan<'a> {
    circuit: &'a Circuit,
    instances: Vec<NoiseInstance>,
    by_device: Vec<Option<usize>>,
    sources: Vec<BoundNoiseSource>,
    staging: Vec<BoundNoiseSource>,
    scratch: Vec<NoiseSource>,
}

struct NoiseVisit<'a> {
    circuit: &'a Circuit,
    point: &'a PreparedPoint,
    context: &'a ModelContext,
    bias: &'a Vector,
    state: Option<&'a [Real]>,
}

impl<'a> NoiseVisit<'a> {
    fn new(
        circuit: &'a Circuit,
        point: &'a PreparedPoint,
        context: &'a ModelContext,
        bias: &'a Vector,
        state: Option<&'a StateSnapshot>,
    ) -> SpiceResult<Self> {
        if bias.len() != circuit.unknown_count() || !bias.is_finite() {
            return Err(SpiceError::circuit("invalid noise bias dimensions/values"));
        }
        Ok(Self {
            circuit,
            point,
            context,
            bias,
            state: circuit.snapshot_values(state)?,
        })
    }

    fn describe(
        &self,
        index: usize,
        scratch: &mut Vec<NoiseSource>,
    ) -> SpiceResult<(&str, DeviceNoise<'_>)> {
        let device = self.point.device(self.circuit, index);
        let rows = self
            .circuit
            .state_rows(index)
            .ok_or_else(|| SpiceError::circuit("circuit numbering is stale; finalize first"))?;
        let controls = self.circuit.controls(index)?;
        let states = self
            .state
            .map(|state| {
                state
                    .get(rows)
                    .ok_or_else(|| SpiceError::circuit("noise bias state is too short"))
            })
            .transpose()?;
        scratch.clear();
        let description = device.noise(
            &NoiseContext {
                model_context: self.context,
                unknowns: self.circuit.unknowns(),
                bias: self.bias,
                states,
                controls,
            },
            scratch,
        )?;
        if matches!(description, DeviceNoise::Noiseless) != scratch.is_empty() {
            return Err(SpiceError::circuit(format!(
                "{}: inconsistent noise description and generator count",
                device.name()
            )));
        }
        if let Some(bad) = scratch.iter().find(|source| !source.kind.validate()) {
            return Err(SpiceError::Numerical {
                context: format!("noise of {}", device.name()),
                message: format!("nonfinite or out-of-domain generator {}", bad.suffix),
            });
        }
        Ok((device.name(), description))
    }
}

impl<'a> NoisePlan<'a> {
    /// Bind C's family, reverse model creation and reverse instance order.
    /// Call once before spectral evaluation.
    ///
    /// # Errors
    /// Invalid numbering, bias, state, generator description or transfer.
    pub fn bind(
        run: &PreparedRun<'a>,
        context: &ModelContext,
        bias: &Vector,
        state: Option<&StateSnapshot>,
    ) -> SpiceResult<Self> {
        let circuit = run.circuit();
        let point = run.point(context)?;
        let visit = NoiseVisit::new(circuit, &point, context, bias, state)?;
        let mut models = BTreeMap::new();
        let mut entries = Vec::new();
        let mut sources = Vec::new();
        let mut scratch = Vec::new();
        for index in 0..circuit.device_count() {
            let (name, description) = visit.describe(index, &mut scratch)?;
            let DeviceNoise::Sources {
                family,
                model,
                total,
            } = description
            else {
                continue;
            };
            let next_rank = models.len();
            let rank = *models.entry((family, model)).or_insert(next_rank);
            let start = sources.len();
            for &source in &scratch {
                sources.push(BoundNoiseSource::bind(source, circuit.unknowns(), None)?);
            }
            entries.push((
                rank,
                NoiseInstance {
                    device: index,
                    name: name.to_owned(),
                    family,
                    model: model.map(str::to_owned),
                    suffixes: scratch.iter().map(|source| source.suffix).collect(),
                    total,
                    range: start..sources.len(),
                },
            ));
        }
        entries.sort_unstable_by(|(rank_a, a), (rank_b, b)| {
            a.family
                .cmp(&b.family)
                .then(rank_b.cmp(rank_a))
                .then(b.device.cmp(&a.device))
        });
        let instances: Vec<_> = entries.into_iter().map(|(_, instance)| instance).collect();
        let mut by_device = vec![None; circuit.device_count()];
        for (slot, instance) in instances.iter().enumerate() {
            by_device[instance.device] = Some(slot);
        }
        Ok(Self {
            circuit,
            instances,
            by_device,
            staging: sources.clone(),
            sources,
            scratch,
        })
    }

    /// Immutable instances and current numeric sources, in C visitation order.
    pub fn instances(
        &self,
    ) -> impl ExactSizeIterator<Item = (&NoiseInstance, &[BoundNoiseSource])> {
        self.instances
            .iter()
            .map(|instance| (instance, &self.sources[instance.range.clone()]))
    }

    /// Number of generators, excluding instance totals.
    #[must_use]
    pub const fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Validate and refresh laws at the new frequency/bias. Reuse the
    /// published transfer rows when their node pairs are unchanged.
    ///
    /// # Errors
    /// Foreign topology, invalid bias/state/law/transfer or changed identity
    /// (including a change between noisy and noiseless).
    pub fn refresh(
        &mut self,
        run: &PreparedRun<'a>,
        context: &ModelContext,
        bias: &Vector,
        state: Option<&StateSnapshot>,
    ) -> SpiceResult<()> {
        if !std::ptr::eq(self.circuit, run.circuit()) {
            return Err(SpiceError::circuit(
                "noise plan belongs to a different finalized circuit",
            ));
        }
        let point = run.point(context)?;
        let visit = NoiseVisit::new(self.circuit, &point, context, bias, state)?;
        for (index, slot) in self.by_device.iter().enumerate() {
            let (name, description) = visit.describe(index, &mut self.scratch)?;
            match (slot, description) {
                (None, DeviceNoise::Noiseless) => {}
                (
                    Some(slot),
                    DeviceNoise::Sources {
                        family,
                        model,
                        total,
                    },
                ) => {
                    let instance = &self.instances[*slot];
                    if instance.name != name
                        || instance.family != family
                        || instance.model.as_deref() != model
                        || instance.total != total
                        || instance.suffixes.len() != self.scratch.len()
                        || instance
                            .suffixes
                            .iter()
                            .zip(&self.scratch)
                            .any(|(&suffix, source)| suffix != source.suffix)
                    {
                        return Err(changed_layout(name));
                    }
                    for (offset, &source) in self.scratch.iter().enumerate() {
                        let slot = instance.range.start + offset;
                        self.staging[slot] = BoundNoiseSource::bind(
                            source,
                            self.circuit.unknowns(),
                            Some(self.sources[slot]),
                        )?;
                    }
                }
                _ => return Err(changed_layout(name)),
            }
        }
        std::mem::swap(&mut self.sources, &mut self.staging);
        Ok(())
    }
}

fn changed_layout(name: &str) -> SpiceError {
    SpiceError::circuit(format!(
        "noise generator layout changed between frequencies for {name}"
    ))
}
