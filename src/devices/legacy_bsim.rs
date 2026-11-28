//! Source binding for the distinct prepared Berkeley BSIM1/2 devices.
//! Physical current, charge and noise equations are owned by the shared kernel.

use std::sync::Arc;

use model_compact_intrinsic::{
    BoundParameters, ModelVariant, ParameterScope, ParameterType, ParameterValue,
    PreparedLegacyBsimModel, prepare_legacy_bsim_model,
};
use model_kernel_api::Artifact;
use model_kernel_bsim::{INSTANCE_PARAMETERS, Instance, LegacyBsimKernel, legacy_bsim_artifact};
use sha2::{Digest, Sha256};

use super::{
    Device, ModelBackend, ModelContext, ModelFamily, ResolvedModel,
    models::PreparationKey,
    prepared_model::{
        AmbientTemperature, InstanceTemperature, PreparedDevice, SourceBinding, SourceParameter,
        SourceStart,
    },
    schema::finite_literal,
};
use crate::{
    netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind},
    primitives::{NodeAppend, SpiceError, SpiceResult, parse_spice_number},
};

const GMIN: usize = INSTANCE_PARAMETERS.len() - 1;

const fn source_parameter(slot: usize) -> SourceParameter {
    SourceParameter {
        name: INSTANCE_PARAMETERS[slot],
        slot,
        offset: 0.,
    }
}

const SOURCE_PARAMETERS: &[SourceParameter] = &[
    source_parameter(0),
    source_parameter(1),
    source_parameter(2),
    source_parameter(3),
    source_parameter(4),
    source_parameter(5),
    source_parameter(6),
    source_parameter(7),
    source_parameter(8),
];

struct Compiled {
    physical: Arc<PreparedLegacyBsimModel>,
    defaults: Vec<ParameterAssignment>,
    fingerprint: [u8; 32],
}

fn field_error(field: &ParameterAssignment, message: impl std::fmt::Display) -> SpiceError {
    SpiceError::parse(
        field.location.clone(),
        format!("legacy BSIM {}: {message}", field.name),
    )
}

fn model_value(variant: ModelVariant, field: &ParameterAssignment) -> SpiceResult<ParameterValue> {
    let descriptor = variant
        .parameter(ParameterScope::Model, &field.name)
        .ok_or_else(|| field_error(field, "unknown model parameter"))?;
    match descriptor.kind {
        ParameterType::Real => finite_literal(field).map(ParameterValue::Real),
        ParameterType::Flag if field.kind == ParameterKind::Flag => Ok(ParameterValue::Flag(true)),
        _ => Err(field_error(
            field,
            "wrong parameter syntax for the selected family",
        )),
    }
}

fn compile(model: &ResolvedModel<'_>) -> SpiceResult<Compiled> {
    let variant = match model.levels().backend {
        ModelBackend::Bsim1 => ModelVariant::Bsim1,
        ModelBackend::Bsim2 => ModelVariant::Bsim2,
        _ => return Err(SpiceError::circuit("foreign legacy BSIM backend")),
    };
    let card = model.card();
    let mut pmos = model.family() == ModelFamily::Pmos;
    let mut fields = Vec::new();
    let mut defaults = Vec::new();
    let mut hash = Sha256::new();
    hash.update(match variant {
        ModelVariant::Bsim1 => b"bsim1/v1",
        _ => b"bsim2/v1",
    });
    for field in &card.parameters {
        if field.name == "level" {
            continue;
        }
        if variant
            .parameter(ParameterScope::Model, &field.name)
            .is_some()
        {
            let value = model_value(variant, field)?;
            match field.name.as_str() {
                "nmos" => pmos = false,
                "pmos" => pmos = true,
                _ => {
                    hash.update(Sha256::digest(field.name.as_bytes()));
                    match &value {
                        ParameterValue::Real(value) => hash.update(value.to_bits().to_le_bytes()),
                        ParameterValue::Flag(value) => hash.update([u8::from(*value)]),
                        _ => return Err(field_error(field, "unsupported model field type")),
                    }
                    fields.push((field.name.as_str(), value));
                }
            }
        } else {
            // INPgetMod permits instance defaults on the declaration.
            // INPdevParse applies those defaults in reverse source order;
            // declaration-level multiplicity is deliberately discarded.
            InstanceInput::new(0.).apply(variant, field)?;
            if field.name != "m" {
                defaults.push(field.clone());
            }
        }
    }
    defaults.reverse();
    hash.update([u8::from(pmos)]);
    fields.extend([
        ("nmos", ParameterValue::Flag(!pmos)),
        ("pmos", ParameterValue::Flag(pmos)),
    ]);
    let bound = BoundParameters::bind(variant, ParameterScope::Model, fields)
        .map_err(|error| SpiceError::parse(card.location.clone(), error.to_string()))?;
    let physical = prepare_legacy_bsim_model(&bound)
        .map_err(|error| SpiceError::parse(card.location.clone(), error.to_string()))?;
    Ok(Compiled {
        physical: Arc::new(physical),
        defaults,
        fingerprint: hash.finalize().into(),
    })
}

struct InstanceInput {
    slots: [f64; 10],
    voltages: [Option<f64>; 3],
    off: bool,
}

impl InstanceInput {
    fn new(gmin: f64) -> Self {
        Self {
            slots: Instance {
                gmin,
                ..Instance::default()
            }
            .slots(),
            voltages: [None; 3],
            off: false,
        }
    }

    fn apply(&mut self, variant: ModelVariant, field: &ParameterAssignment) -> SpiceResult<()> {
        let descriptor = variant
            .parameter(ParameterScope::Instance, &field.name)
            .ok_or_else(|| field_error(field, "unknown instance parameter"))?;
        match (field.name.as_str(), descriptor.kind, &field.kind) {
            ("off", ParameterType::Flag, ParameterKind::Flag) => self.off = true,
            ("ic", ParameterType::RealVector, ParameterKind::InitialConditions(values)) => {
                if values.is_empty() || values.len() > self.voltages.len() {
                    return Err(field_error(field, "IC requires VDS[,VGS[,VBS]]"));
                }
                for (index, value) in values.iter().enumerate() {
                    let voltage = parse_spice_number(&value.value.text)
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| {
                            SpiceError::parse(
                                value.value.location.clone(),
                                "nonfinite or unresolved legacy BSIM IC",
                            )
                        })?;
                    self.voltages[index] = Some(voltage);
                }
            }
            ("vds" | "vgs" | "vbs", ParameterType::Real, _) => {
                let index = match field.name.as_str() {
                    "vds" => 0,
                    "vgs" => 1,
                    _ => 2,
                };
                self.voltages[index] = Some(finite_literal(field)?);
            }
            (_, ParameterType::Real, _) => {
                let index = INSTANCE_PARAMETERS[..GMIN]
                    .iter()
                    .position(|name| *name == field.name)
                    .ok_or_else(|| field_error(field, "unsupported numeric instance setter"))?;
                // Legacy BSIM1/2 setters receive physical lengths/areas,
                // without the global MOS scale convention of later families.
                self.slots[index] = finite_literal(field)?;
            }
            _ => return Err(field_error(field, "wrong instance parameter syntax")),
        }
        Ok(())
    }

    const fn reference(&self) -> Instance {
        let [
            length,
            width,
            multiplier,
            drain_area,
            source_area,
            drain_perimeter,
            source_perimeter,
            drain_squares,
            source_squares,
            gmin,
        ] = self.slots;
        Instance {
            length,
            width,
            multiplier,
            drain_area,
            source_area,
            drain_perimeter,
            source_perimeter,
            drain_squares,
            source_squares,
            gmin,
        }
    }

    fn shape(&self, compiled: &Compiled, instance: &DeviceInstance) -> SpiceResult<[bool; 2]> {
        let bound = BoundParameters::bind(
            compiled.physical.variant(),
            ParameterScope::Instance,
            INSTANCE_PARAMETERS[..GMIN]
                .iter()
                .copied()
                .zip(self.slots[..GMIN].iter().copied().map(ParameterValue::Real)),
        )
        .map_err(|error| SpiceError::parse(instance.location.clone(), error.to_string()))?;
        // Use the producer's exact topology decision. Invalid local geometry
        // is never placed inside the declaration-scoped artifact cache.
        let prepared = compiled
            .physical
            .prepare_instance(&bound)
            .map_err(|error| SpiceError::parse(instance.location.clone(), error.to_string()))?;
        Ok([
            prepared.drain.series_conductance > 0.,
            prepared.source.series_conductance > 0.,
        ])
    }
}

fn artifact_for(
    model: &ResolvedModel<'_>,
    compiled: &Compiled,
    input: &InstanceInput,
    instance: &DeviceInstance,
    [drain_series, source_series]: [bool; 2],
) -> SpiceResult<Arc<Artifact<LegacyBsimKernel>>> {
    let cached = model.prepare(
        PreparationKey::LegacyBsimTopology {
            drain_series,
            source_series,
        },
        || {
            let mut hash = Sha256::new();
            hash.update(compiled.fingerprint);
            hash.update([u8::from(drain_series), u8::from(source_series)]);
            let digest = hash.finalize();
            let mut fingerprint = [0; 16];
            fingerprint.copy_from_slice(&digest[..16]);
            legacy_bsim_artifact(
                Arc::clone(&compiled.physical),
                input.reference(),
                fingerprint,
            )
            .map_err(|error| SpiceError::parse(instance.location.clone(), error.to_string()))
        },
    )?;
    Ok(Arc::clone(cached.as_ref()))
}

pub(super) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
    context: &ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    if instance.nodes.len() != 4 {
        return Err(SpiceError::parse(
            instance.location.clone(),
            "legacy BSIM requires four terminals",
        ));
    }
    let compiled = model.prepare(PreparationKey::Inputs, || compile(model))?;
    let mut input = InstanceInput::new(context.gmin);
    for field in compiled.defaults.iter().chain(&instance.parameters) {
        input.apply(compiled.physical.variant(), field)?;
    }
    let shape = input.shape(&compiled, instance)?;
    let artifact = artifact_for(model, &compiled, &input, instance, shape)?;
    let [drain_series, source_series] = shape;
    let source = SourceBinding {
        model_name: model.card().name.clone(),
        designator: 'm',
        parameters: SOURCE_PARAMETERS,
        gmin_slot: Some(GMIN),
        model_ambient: AmbientTemperature::Circuit,
        relative_temperature_slot: None,
        absolute_temperature_slot: None,
        start: Some(SourceStart {
            external: [0, 1, 2, 3],
            intrinsic: [
                if drain_series { 4 } else { 0 },
                1,
                if source_series {
                    4 + usize::from(drain_series)
                } else {
                    2
                },
                3,
            ],
            voltages: input.voltages,
            off: input.off,
        }),
    };
    nodes.transaction(|nodes| {
        let terminals = nodes.intern_array::<4>(&instance.nodes)?;
        let device = PreparedDevice::bind_nodes(
            nodes,
            &artifact,
            &instance.name,
            &terminals,
            &input.slots,
            InstanceTemperature::circuit(context.temperature),
        )?
        .with_source(source);
        Ok(Box::new(device) as Box<dyn Device>)
    })
}
