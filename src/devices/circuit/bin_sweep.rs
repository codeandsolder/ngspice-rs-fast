//! Geometry candidates retain the literalized, lexically expanded source graph.
//! Model selection and setup must use written dimensions, not a previously
//! selected model's physical setter snapshot.

use super::{CandidateNumericOverride, Circuit, ModelContext};
use crate::devices::models::ModelResolver;
use crate::netlist::ast::{DeviceInstance, ModelCard, ParameterAssignment, ParameterKind};
use crate::primitives::{NodeTable, Real, SpiceError, SpiceResult};
use std::collections::BTreeMap;

#[expect(
    clippy::redundant_pub_crate,
    reason = "crate-restricted visibility is required by the unreachable_pub lint"
)]
pub(crate) struct BinnedDeck {
    devices: Vec<DeviceInstance>,
    models: Vec<ModelCard>,
    binned: BTreeMap<String, usize>,
    noise_analysis: bool,
}

#[derive(Clone, Copy)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "crate-restricted visibility is required by the unreachable_pub lint"
)]
pub(crate) struct GeometryEdit<'a> {
    pub(crate) instance: &'a str,
    pub(crate) parameter: &'a str,
    pub(crate) value: Real,
}

impl BinnedDeck {
    pub(super) fn capture(
        devices: &[DeviceInstance],
        models: &[ModelCard],
        resolver: &ModelResolver<'_>,
        noise_analysis: bool,
    ) -> Option<Self> {
        let binned: BTreeMap<_, _> = devices
            .iter()
            .enumerate()
            .filter(|(_, instance)| {
                instance.designator.eq_ignore_ascii_case(&'m')
                    && instance.model.as_ref().is_some_and(|name| {
                        resolver.model(name).is_none()
                            && resolver.bin_candidates(name).next().is_some()
                    })
            })
            .map(|(index, instance)| (instance.name.to_ascii_lowercase(), index))
            .collect();
        if binned.is_empty() {
            return None;
        }
        Some(Self {
            devices: devices.to_vec(),
            models: models.to_vec(),
            binned,
            noise_analysis,
        })
    }

    pub(crate) fn contains(&self, name: &str, parameter: &str) -> bool {
        matches!(parameter, "l" | "w" | "nf" | "wnflag")
            && self.binned.contains_key(&name.to_ascii_lowercase())
    }

    pub(crate) fn resolver(&self, scale: Real) -> SpiceResult<ModelResolver<'_>> {
        Ok(ModelResolver::new(&self.models)?.with_bin_options(
            crate::devices::binning::BinOptions {
                scale,
                ..Default::default()
            },
        ))
    }

    fn geometry_override<'a>(
        &self,
        baseline: &'a Circuit,
        context: &ModelContext,
        value: crate::devices::sweep::InstanceOverride,
    ) -> SpiceResult<Option<GeometryEdit<'a>>> {
        if value.topology() != baseline.topology_id()? {
            return Err(SpiceError::circuit("stale geometry sweep override"));
        }
        let device = baseline
            .devices
            .get(value.device())
            .ok_or_else(|| SpiceError::circuit("stale sweep device ordinal"))?;
        let name = device
            .numeric_parameter_name(value.parameter())
            .ok_or_else(|| SpiceError::circuit("missing canonical setter"))?;
        if !self.contains(device.name(), name) {
            return Ok(None);
        }
        let raw = if matches!(name, "l" | "w") {
            value.value() / context.geometry_scale
        } else {
            value.value()
        };
        if !raw.is_finite() {
            return Err(SpiceError::circuit("invalid written geometry override"));
        }
        Ok(Some(GeometryEdit {
            instance: device.name(),
            parameter: name,
            value: raw,
        }))
    }

    fn edited_instances(
        &self,
        baseline: &Circuit,
        context: &ModelContext,
        edits: &[GeometryEdit<'_>],
    ) -> SpiceResult<BTreeMap<usize, DeviceInstance>> {
        // Earlier numeric candidates/control edits affect the same geometry
        // that the resolver must see. Convert physical values back to the
        // written scale; explicit sweep coordinates are appended last.
        let mut existing = Vec::new();
        let slots = context.instance_overrides.iter().flatten().copied();
        if baseline.runtime.control_overrides.is_empty() {
            existing.extend(slots);
            existing.extend(
                baseline
                    .runtime
                    .candidate_overrides
                    .iter()
                    .filter_map(instance_override),
            );
        } else {
            existing.extend(
                baseline
                    .runtime
                    .control_overrides
                    .iter()
                    .filter_map(instance_override),
            );
            existing.extend(slots);
        }
        let mut geometry = Vec::new();
        for value in existing {
            if let Some(edit) = self.geometry_override(baseline, context, value)? {
                geometry.push(edit);
            }
        }
        geometry.extend_from_slice(edits);
        let mut instances = BTreeMap::new();
        for edit in geometry {
            let &index = self
                .binned
                .get(&edit.instance.to_ascii_lowercase())
                .ok_or_else(|| SpiceError::circuit("unknown binned geometry target"))?;
            let instance = instances
                .entry(index)
                .or_insert_with(|| self.devices[index].clone());
            instance.parameters.push(ParameterAssignment {
                name: edit.parameter.to_owned(),
                // The grid coordinate is a written value, not a previously
                // scaled physical setter. Preserve its binary64 round trip.
                value: edit.value.to_string(),
                kind: ParameterKind::Scalar,
                location: instance.location.clone(),
            });
        }
        Ok(instances)
    }

    /// Reuse an owned sweep candidate's unaffected devices and finalized
    /// bindings when the changed instances select the same declarations and
    /// allocate exactly the same nodes, state and matrix elements.
    ///
    /// A scratch node prefix lets the ordinary factory build a complete new
    /// physical instance without modifying the candidate. Commit the entire
    /// replacement set only after every structural check succeeds.
    pub(crate) fn try_update(
        &self,
        baseline: &Circuit,
        circuit: &mut Circuit,
        resolver: &ModelResolver<'_>,
        edits: &[GeometryEdit<'_>],
        context: &ModelContext,
    ) -> SpiceResult<bool> {
        let mut replacements = Vec::new();
        let mut setup = *context;
        setup.noise_analysis = self.noise_analysis;
        setup.resistor_overrides.fill(None);
        setup.instance_overrides.fill(None);
        for instance in self
            .edited_instances(baseline, context, edits)?
            .into_values()
        {
            let ordinal = circuit
                .device_index(&instance.name)
                .ok_or_else(|| SpiceError::circuit("missing binned sweep device"))?;
            let original = &circuit.devices[ordinal];
            let selected = resolver.select_bin(&instance)?;
            if selected.map(|model| model.name.as_str()) != original.model_name() {
                return Ok(false);
            }
            let prefix = original
                .origin
                .internal_nodes
                .first()
                .map_or(circuit.nodes.len(), |node| node.index());
            let mut nodes = NodeTable::with_auto_gnd(circuit.nodes.auto_gnd());
            for node in &circuit.nodes.nodes()[1..prefix] {
                let id = nodes.intern(&node.name)?;
                nodes.set_kind(id, node.kind);
            }
            let replacement = nodes.transaction(|nodes| {
                crate::devices::factory::instantiate_with_models(&instance, nodes, resolver, &setup)
            })?;
            let expected = &original.origin.internal_nodes;
            if nodes.len() != prefix + expected.len()
                || !nodes.nodes()[prefix..]
                    .iter()
                    .zip(expected)
                    .all(|(node, &id)| circuit.nodes.nodes().get(id.index()) == Some(node))
            {
                return Ok(false);
            }
            let replacement = (ordinal, replacement);
            if circuit
                .validate_replacements(std::slice::from_ref(&replacement))
                .is_err()
            {
                return Ok(false);
            }
            let mut elements = Vec::new();
            let mut matrix = crate::devices::setup::MatrixSetup {
                unknowns: &circuit.bindings.unknowns,
                branches: circuit.bindings.branch_rows[ordinal].clone(),
                controls: circuit.controls(ordinal)?,
                coupled: circuit
                    .bindings
                    .mutual
                    .setup_rows
                    .get(&ordinal)
                    .map_or(&[][..], Vec::as_slice),
                pole_zero: false,
                pole_zero_setups: 0,
                elements: &mut elements,
            };
            replacement.1.setup_matrix(&mut matrix)?;
            if elements != circuit.setup_elements(ordinal)? {
                return Ok(false);
            }
            replacements.push(replacement);
        }
        for (ordinal, replacement) in replacements {
            circuit.devices[ordinal].device = replacement;
        }
        circuit.clear_candidate_analysis_state();
        Ok(true)
    }

    /// Instantiate the same expanded graph with updated written geometry.
    /// This is the ordinary model resolver and device factory, not a
    /// numerical override of the previous declaration.
    pub(crate) fn candidate(
        &self,
        baseline: &Circuit,
        resolver: &ModelResolver<'_>,
        edits: &[GeometryEdit<'_>],
        context: &ModelContext,
    ) -> SpiceResult<Circuit> {
        let mut devices = self.devices.clone();
        for (index, instance) in self.edited_instances(baseline, context, edits)? {
            devices[index] = instance;
        }
        let mut setup = *context;
        setup.noise_analysis = self.noise_analysis;
        setup.resistor_overrides.fill(None);
        setup.instance_overrides.fill(None);
        let mut circuit = Circuit {
            nodes: NodeTable::with_auto_gnd(baseline.nodes.auto_gnd()),
            mutual_limits: baseline.mutual_limits,
            elaboration_limits: baseline.elaboration_limits,
            ..Circuit::default()
        };
        circuit.add_instances(&devices, resolver, &setup)?;
        circuit.add_topology_shunts(setup.shunts)?;
        circuit.finalize()?;
        circuit.prepared_mutual_terms(&setup)?;
        let observations = baseline
            .observations()
            .iter()
            .map(|observation| observation.name().to_owned())
            .collect::<Vec<_>>();
        circuit.set_observations(&observations)?;
        Ok(circuit)
    }
}

impl Circuit {
    /// Remap numeric overlays to a newly elaborated topology by device name
    /// and canonical parameter, retaining already physical setter values.
    pub(crate) fn rebind_sweep_context(
        &mut self,
        original: &Self,
        mut context: ModelContext,
    ) -> SpiceResult<ModelContext> {
        use super::CandidateNumericOverride::{Instance, Resistor};
        let identity = original.topology_id()?;
        let rebind =
            |target: CandidateNumericOverride| -> SpiceResult<Option<CandidateNumericOverride>> {
                let ordinal = target.device();
                let device = original
                    .devices
                    .get(ordinal)
                    .ok_or_else(|| SpiceError::circuit("stale sweep device ordinal"))?;
                match target {
                    Resistor(value) => {
                        if value.topology() != identity {
                            return Err(SpiceError::circuit("stale resistor override"));
                        }
                        Ok(Some(Resistor(
                            self.bind_resistor_target(device.name())?
                                .at_resistance(value.supplied())?,
                        )))
                    }
                    Instance(value) => {
                        if value.topology() != identity {
                            return Err(SpiceError::circuit("stale instance override"));
                        }
                        let name = device
                            .numeric_parameter_name(value.parameter())
                            .ok_or_else(|| SpiceError::circuit("missing canonical setter"))?;
                        if original
                            .binned_deck()
                            .is_some_and(|deck| deck.contains(device.name(), name))
                        {
                            return Ok(None);
                        }
                        let target = if name == "dc" && device.input_source().is_some() {
                            self.bind_source_target(device.name())?
                        } else {
                            self.bind_instance_target(device.name(), name)?
                        };
                        Ok(Some(Instance(target.at(value.value())?)))
                    }
                }
            };
        for slot in &mut context.resistor_overrides {
            if let Some(value) = *slot {
                *slot = match rebind(Resistor(value))? {
                    Some(Resistor(value)) => Some(value),
                    _ => None,
                };
            }
        }
        for slot in &mut context.instance_overrides {
            if let Some(value) = *slot {
                *slot = match rebind(Instance(value))? {
                    Some(Instance(value)) => Some(value),
                    _ => None,
                };
            }
        }
        let candidates = original
            .runtime
            .candidate_overrides
            .iter()
            .copied()
            .filter_map(|value| rebind(value).transpose())
            .collect::<SpiceResult<Vec<_>>>()?;
        let controls = original
            .runtime
            .control_overrides
            .iter()
            .copied()
            .filter_map(|value| rebind(value).transpose())
            .collect::<SpiceResult<Vec<_>>>()?;
        self.runtime.candidate_overrides = candidates;
        self.runtime.control_overrides = controls;
        Ok(context)
    }
}

const fn instance_override(
    target: &CandidateNumericOverride,
) -> Option<crate::devices::sweep::InstanceOverride> {
    match *target {
        CandidateNumericOverride::Instance(value) => Some(value),
        CandidateNumericOverride::Resistor(_) => None,
    }
}
