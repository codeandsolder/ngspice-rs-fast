//! Geometry candidates retain the literalized, lexically expanded source graph.
//! Model selection and setup must use written dimensions, not a previously
//! selected model's physical setter snapshot.

use super::{Circuit, ModelContext};
use crate::devices::models::ModelResolver;
use crate::netlist::ast::{DeviceInstance, ModelCard, ParameterAssignment, ParameterKind};
use crate::primitives::{Real, SpiceResult};
use std::collections::BTreeMap;

pub(crate) struct BinnedDeck {
    devices: Vec<DeviceInstance>,
    models: Vec<ModelCard>,
    binned: BTreeMap<String, usize>,
    noise_analysis: bool,
}

#[derive(Clone, Copy)]
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
        for edit in edits {
            let index = self.binned[&edit.instance.to_ascii_lowercase()];
            let instance = &mut devices[index];
            instance.parameters.push(ParameterAssignment {
                name: edit.parameter.to_owned(),
                // Round-trip the grid's actual binary64 coordinate, without
                // applying the global geometry scale twice.
                value: edit.value.to_string(),
                kind: ParameterKind::Scalar,
                location: instance.location.clone(),
            });
        }
        let mut setup = *context;
        setup.noise_analysis = self.noise_analysis;
        setup.resistor_overrides.fill(None);
        setup.instance_overrides.fill(None);
        let mut circuit = Circuit {
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
        original: &Circuit,
        mut context: ModelContext,
    ) -> SpiceResult<ModelContext> {
        use super::CandidateNumericOverride::{Instance, Resistor};
        use crate::primitives::SpiceError;
        let identity = original.topology_id()?;
        let rebind = |target: super::CandidateNumericOverride| -> SpiceResult<super::CandidateNumericOverride> {
            let ordinal = target.device();
            let device = original.devices.get(ordinal).ok_or_else(|| SpiceError::circuit("stale sweep device ordinal"))?;
            match target {
                Resistor(value) => {
                    if value.topology() != identity {
                        return Err(SpiceError::circuit("stale resistor override"));
                    }
                    Ok(Resistor(self.bind_resistor_target(device.name())?.at_resistance(value.supplied())?))
                }
                Instance(value) => {
                    if value.topology() != identity {
                        return Err(SpiceError::circuit("stale instance override"));
                    }
                    let name = device.numeric_parameter_name(value.parameter()).ok_or_else(|| SpiceError::circuit("missing canonical setter"))?;
                    if original.binned_deck().is_some_and(|deck| deck.contains(device.name(), name)) {
                        return Err(SpiceError::circuit("binned geometry cannot be combined with a pre-existing geometry override"));
                    }
                    let target = if name == "dc" && device.input_source().is_some() {
                        self.bind_source_target(device.name())?
                    } else {
                        self.bind_instance_target(device.name(), name)?
                    };
                    Ok(Instance(target.at(value.value())?))
                }
            }
        };
        for slot in &mut context.resistor_overrides {
            if let Some(value) = *slot
                && let Resistor(value) = rebind(Resistor(value))?
            {
                *slot = Some(value);
            }
        }
        for slot in &mut context.instance_overrides {
            if let Some(value) = *slot
                && let Instance(value) = rebind(Instance(value))?
            {
                *slot = Some(value);
            }
        }
        let candidates = original
            .runtime
            .candidate_overrides
            .iter()
            .copied()
            .map(&rebind)
            .collect::<SpiceResult<Vec<_>>>()?;
        let controls = original
            .runtime
            .control_overrides
            .iter()
            .copied()
            .map(rebind)
            .collect::<SpiceResult<Vec<_>>>()?;
        self.runtime.candidate_overrides = candidates;
        self.runtime.control_overrides = controls;
        Ok(context)
    }
}
