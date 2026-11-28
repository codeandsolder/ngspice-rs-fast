//! Bind the explicit source-selected diode family to the shared physical kernel.
//! Ordinary native D declarations continue to use the native diode factory.

use std::sync::Arc;

use model_kernel_api::Artifact;
use model_kernel_diode::{
    Instance, PreparedModel,
    artifact::{DiodeKernel, artifact},
    compile_simetrix,
};
use sha2::{Digest, Sha256};

use super::{
    Device, ModelContext, ResolvedModel,
    models::PreparationKey,
    prepared_model::{
        AmbientTemperature, InstanceTemperature, PreparedDevice, SourceBinding, SourceParameter,
    },
    schema::finite_literal,
};
use crate::{
    netlist::ast::DeviceInstance,
    primitives::{NodeAppend, SpiceError, SpiceResult},
};

const CELSIUS_TO_KELVIN: f64 = 273.15;
const AREA: usize = 0;
const MULTIPLIER: usize = 1;
const GMIN: usize = 2;
const RELATIVE_TEMPERATURE: usize = 3;
const ABSOLUTE_TEMPERATURE: usize = 4;

const AREA_PARAMETER: SourceParameter = SourceParameter {
    name: "area",
    slot: AREA,
    offset: 0.,
};
const MULTIPLIER_PARAMETER: SourceParameter = SourceParameter {
    name: "m",
    slot: MULTIPLIER,
    offset: 0.,
};
const RELATIVE_PARAMETER: SourceParameter = SourceParameter {
    name: "dtemp",
    slot: RELATIVE_TEMPERATURE,
    offset: 0.,
};
const ABSOLUTE_PARAMETER: SourceParameter = SourceParameter {
    name: "temp",
    slot: ABSOLUTE_TEMPERATURE,
    offset: CELSIUS_TO_KELVIN,
};
const RELATIVE_PARAMETERS: &[SourceParameter] =
    &[AREA_PARAMETER, MULTIPLIER_PARAMETER, RELATIVE_PARAMETER];
const ABSOLUTE_PARAMETERS: &[SourceParameter] = &[
    AREA_PARAMETER,
    MULTIPLIER_PARAMETER,
    RELATIVE_PARAMETER,
    ABSOLUTE_PARAMETER,
];

struct Compiled {
    model: PreparedModel,
    fingerprint: [u8; 32],
}

fn compile(model: &ResolvedModel<'_>) -> SpiceResult<Compiled> {
    let card = model.card();
    let fields = card
        .parameters
        .iter()
        .map(|field| Ok((field.name.as_str(), finite_literal(field)?)))
        .collect::<SpiceResult<Vec<_>>>()?;
    let compiled = compile_simetrix(fields.iter().copied())
        .map_err(|error| SpiceError::parse(card.location.clone(), error.to_string()))?;
    // Hash the ordered source fields (including aliases/overrides), not
    // instance values, names or traversal order. The topology hash below
    // distinguishes the two independent descriptor shape decisions.
    let mut hash = Sha256::new();
    hash.update(b"simetrix-diode-level1/v1");
    for (name, value) in fields {
        hash.update(Sha256::digest(name.to_ascii_lowercase().as_bytes()));
        hash.update(value.to_bits().to_le_bytes());
    }
    Ok(Compiled {
        model: compiled,
        fingerprint: hash.finalize().into(),
    })
}

fn instance_input(instance: &DeviceInstance, context: &ModelContext) -> SpiceResult<Instance> {
    if instance.nodes.len() != 2 {
        return Err(SpiceError::parse(
            instance.location.clone(),
            "SIMetrix diode requires two terminals",
        ));
    }
    let mut input = Instance {
        analysis_temperature: context.temperature + CELSIUS_TO_KELVIN,
        gmin: context.gmin,
        ..Instance::default()
    };
    for field in &instance.parameters {
        let value = finite_literal(field)?;
        match field.name.as_str() {
            "area" => input.area = value,
            "m" => input.multiplier = value,
            "temp" => input.absolute_temperature = Some(value + CELSIUS_TO_KELVIN),
            "dtemp" => input.relative_temperature = value,
            _ => {
                return Err(SpiceError::Unsupported {
                    feature: format!("SIMetrix diode instance parameter {}", field.name),
                    location: Some(field.location.clone()),
                });
            }
        }
    }
    Ok(input)
}

fn prepared_artifact(
    model: &ResolvedModel<'_>,
    compiled: &Compiled,
    input: Instance,
    instance: &DeviceInstance,
) -> SpiceResult<Arc<Artifact<DiodeKernel>>> {
    let error = |message: String| SpiceError::parse(instance.location.clone(), message);
    // Instance-specific failures must not poison the shared declaration cache.
    let bound = compiled
        .model
        .bind(input)
        .map_err(|err| error(err.to_string()))?;
    let resistance = bound.series_resistance();
    let series = resistance != 0.;
    if series && !resistance.recip().is_finite() {
        return Err(error("nonrepresentable diode series conductance".into()));
    }
    let absolute_temperature = input.absolute_temperature.is_some();
    let cached = model.prepare(
        PreparationKey::PreparedTopology {
            series,
            absolute_temperature,
        },
        || {
            let mut hash = Sha256::new();
            hash.update(compiled.fingerprint);
            hash.update([u8::from(series), u8::from(absolute_temperature)]);
            let digest = hash.finalize();
            let mut fingerprint = [0; 16];
            fingerprint.copy_from_slice(&digest[..16]);
            artifact(compiled.model, input, fingerprint).map_err(|err| error(err.to_string()))
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
    let input = instance_input(instance, context)?;
    let compiled = model.prepare(PreparationKey::Inputs, || compile(model))?;
    let artifact = prepared_artifact(model, &compiled, input, instance)?;
    let mut values = vec![
        input.area,
        input.multiplier,
        input.gmin,
        input.relative_temperature,
    ];
    if let Some(temperature) = input.absolute_temperature {
        values.push(temperature);
    }
    let temperature = compiled.model.parameters().temperature;
    let model_ambient = temperature.absolute.map_or(
        AmbientTemperature::DeltaKelvin(temperature.relative),
        AmbientTemperature::AbsoluteKelvin,
    );
    let absolute = input.absolute_temperature.is_some();
    let source = SourceBinding {
        model_name: model.card().name.clone(),
        designator: 'd',
        parameters: if absolute {
            ABSOLUTE_PARAMETERS
        } else {
            RELATIVE_PARAMETERS
        },
        gmin_slot: Some(GMIN),
        model_ambient,
        relative_temperature_slot: Some(RELATIVE_TEMPERATURE),
        start: None,
        absolute_temperature_slot: absolute.then_some(ABSOLUTE_TEMPERATURE),
    };
    nodes.transaction(|nodes| {
        let terminals = nodes.intern_array::<2>(&instance.nodes)?;
        let device = PreparedDevice::bind_nodes(
            nodes,
            &artifact,
            &instance.name,
            &terminals,
            &values,
            InstanceTemperature::circuit(context.temperature),
        )?
        .with_source(source);
        Ok(Box::new(device) as Box<dyn Device>)
    })
}
