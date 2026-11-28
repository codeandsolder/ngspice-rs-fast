//! Occurrence-resolved `LTspice` diode dispatch and prepared kernel binding.
//! The numerical transport and charge equations live in model-kernel-diode.

use std::sync::Arc;

use model_kernel_api::{Artifact, Kernel};
use model_kernel_diode::{
    artifact::{fowler_nordheim, ltspice},
    compile_ltspice,
    fowler_nordheim::Model as FnModel,
    ltspice_semiconductor::{Instance as SemiconductorInstance, SourceModel},
};
use sha2::{Digest, Sha256};

use crate::{
    netlist::ast::{DeviceInstance, ModelCard, ParameterKind},
    primitives::{NodeAppend, Real, SpiceError, SpiceResult},
};

use super::{
    Device, LevelSelection, ModelBackend, ModelContext, ResolvedModel,
    models::PreparationKey,
    prepared_model::{
        AmbientTemperature, InstanceTemperature, PreparedDevice, SourceBinding, SourceParameter,
    },
    schema::finite_literal,
};

/// Source dialect identity; never a fixed physical model family.
pub(super) const MODEL_FAMILY: &str = "ltspice_diode";
const CELSIUS_TO_KELVIN: Real = 273.15;

const IDEAL_SELECTORS: &[&str] = &["ron", "roff", "vfwd", "vrev", "rrev"];

/// The final resolved LEVEL has priority over the presence of ideal
/// selectors. Other finite levels use the base ideal/semiconductor family.
pub(super) fn select(card: &ModelCard) -> SpiceResult<LevelSelection> {
    let mut first_raw = None;
    let mut last = 1.;
    let mut ideal = false;
    for field in &card.parameters {
        if field.name.eq_ignore_ascii_case("level") {
            let raw = finite_literal(field)?;
            first_raw.get_or_insert(raw);
            last = raw;
        } else if IDEAL_SELECTORS.contains(&field.name.as_str()) {
            ideal = true;
        }
    }
    if card.level != first_raw {
        return Err(SpiceError::parse(
            card.location.clone(),
            "resolved LTspice LEVEL cache does not match the first ordered setter",
        ));
    }
    let (selector, backend) = if (1.5..2.5).contains(&last) {
        (2, ModelBackend::LtspiceFowlerNordheim)
    } else if ideal {
        (0, ModelBackend::LtspiceIdealDiode)
    } else {
        (1, ModelBackend::LtspiceSemiconductorDiode)
    };
    Ok(LevelSelection {
        first_raw,
        selector,
        backend,
        applied: crate::primitives::c_integer(last).and_then(|value| u8::try_from(value).ok()),
    })
}

fn source_error(instance: &DeviceInstance, error: impl std::fmt::Display) -> SpiceError {
    SpiceError::parse(instance.location.clone(), error.to_string())
}

fn unsupported(instance: &DeviceInstance, name: &str) -> SpiceError {
    SpiceError::Unsupported {
        feature: format!("LTspice diode instance parameter {name}"),
        location: Some(instance.location.clone()),
    }
}

fn model_fields<'a>(model: &ResolvedModel<'a>) -> SpiceResult<Vec<(&'a str, Real)>> {
    model
        .card()
        .parameters
        .iter()
        .map(|field| Ok((field.name.as_str(), finite_literal(field)?)))
        .collect()
}

fn fingerprint(tag: &[u8], fields: &[(&str, Real)], context: &[u8]) -> [u8; 16] {
    let mut hash = Sha256::new();
    hash.update(tag);
    hash.update(context);
    for &(name, value) in fields {
        hash.update(Sha256::digest(name.as_bytes()));
        hash.update(value.to_bits().to_le_bytes());
    }
    let mut fingerprint = [0; 16];
    fingerprint.copy_from_slice(&hash.finalize()[..16]);
    fingerprint
}

const fn parameter(name: &'static str, slot: usize, offset: Real) -> SourceParameter {
    SourceParameter { name, slot, offset }
}

const FN_PARAMETERS: &[SourceParameter] = &[
    parameter("l", 0, 0.),
    parameter("w", 1, 0.),
    parameter("m", 2, 0.),
    parameter("n", 3, 0.),
];

const SEMICONDUCTOR_PARAMETERS: &[SourceParameter] = &[
    parameter("area", 0, 0.),
    parameter("perim", 1, 0.),
    parameter("m", 2, 0.),
    parameter("n", 3, 0.),
];

const SEMICONDUCTOR_ABSOLUTE_PARAMETERS: &[SourceParameter] = &[
    parameter("area", 0, 0.),
    parameter("perim", 1, 0.),
    parameter("m", 2, 0.),
    parameter("n", 3, 0.),
    parameter("temp", 5, CELSIUS_TO_KELVIN),
];

fn bind<K: Kernel + 'static>(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    artifact: &Arc<Artifact<K>>,
    values: &[Real],
    temperature: InstanceTemperature,
    source: SourceBinding,
) -> SpiceResult<Box<dyn Device>> {
    nodes.transaction(|nodes| {
        let terminals = nodes.intern_array::<2>(&instance.nodes)?;
        let device = PreparedDevice::bind_nodes(
            nodes,
            artifact,
            &instance.name,
            &terminals,
            values,
            temperature,
        )?
        .with_source(source);
        Ok(Box::new(device) as Box<dyn Device>)
    })
}

fn fn_model(
    model: &ResolvedModel<'_>,
) -> SpiceResult<Arc<Artifact<fowler_nordheim::FowlerNordheimKernel>>> {
    let fields = model_fields(model)?;
    let mut law = FnModel::default();
    for &(name, value) in &fields {
        match name {
            "tox" => law.oxide_thickness = value,
            "l" => law.length = value,
            "w" => law.width = value,
            // The measured F-N law has no semiconductor, ideal, or rating
            // terms; recognized fields for those families do not select them.
            "level" | "d" | "area" | "perim" | "isw" | "ns" | "rsw" | "cjsw" | "vjsw" | "mjsw"
            | "fcs" | "vp" | "ilimit" | "revilimit" | "epsilon" | "revepsilon" | "vpk" | "ipk"
            | "iave" | "irms" | "diss" => {}
            _ if IDEAL_SELECTORS.contains(&name)
                || model_kernel_diode::SOURCE_PARAMETERS.contains(&name) => {}
            _ => {
                return Err(SpiceError::parse(
                    model.card().location.clone(),
                    format!("unknown Fowler–Nordheim diode model parameter {name}"),
                ));
            }
        }
    }
    fowler_nordheim::artifact(
        law,
        fingerprint(b"ltspice-fowler-nordheim/v1", &fields, &[]),
    )
    .map_err(|error| SpiceError::parse(model.card().location.clone(), error.to_string()))
}

pub(super) fn instantiate_fn(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
    context: &ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    let artifact = model.prepare(PreparationKey::Inputs, || fn_model(model))?;
    let mut values: Vec<_> = artifact
        .descriptor()
        .parameters
        .iter()
        .map(|p| p.default)
        .collect();
    for field in &instance.parameters {
        if field.name == "off" && field.kind == ParameterKind::Flag {
            continue;
        }
        let value = finite_literal(field)?;
        match field.name.as_str() {
            "l" => values[0] = value,
            "w" => values[1] = value,
            "m" => values[2] = value,
            "n" => values[3] = value,
            // Independent witnesses established these are not F-N geometry
            // or temperature coefficients. Area is distinct from L*W.
            "area" | "d" | "temp" => {}
            name => return Err(unsupported(instance, name)),
        }
    }
    bind(
        instance,
        nodes,
        artifact.as_ref(),
        &values,
        InstanceTemperature::circuit(context.temperature),
        SourceBinding {
            model_name: model.card().name.clone(),
            designator: 'd',
            parameters: FN_PARAMETERS,
            gmin_slot: None,
            model_ambient: AmbientTemperature::Circuit,
            relative_temperature_slot: None,
            absolute_temperature_slot: None,
        },
    )
}

struct CompiledSemiconductor {
    model: SourceModel,
    fingerprint: [u8; 16],
}

fn compile_semiconductor(
    model: &ResolvedModel<'_>,
    context: &ModelContext,
) -> SpiceResult<CompiledSemiconductor> {
    let mut fields = model_fields(model)?;
    fields.insert(0, ("tnom", context.nominal_temperature));
    let compiled = compile_ltspice(fields.iter().copied())
        .map_err(|error| SpiceError::parse(model.card().location.clone(), error.to_string()))?;
    Ok(CompiledSemiconductor {
        model: compiled,
        fingerprint: fingerprint(b"ltspice-semiconductor/v1", &fields, &[]),
    })
}

fn semiconductor_input(
    instance: &DeviceInstance,
    context: &ModelContext,
) -> SpiceResult<(SemiconductorInstance, bool)> {
    let mut input = SemiconductorInstance {
        temperature: context.temperature + CELSIUS_TO_KELVIN,
        gmin: context.gmin,
        ..SemiconductorInstance::default()
    };
    let mut absolute = false;
    for field in &instance.parameters {
        if field.name == "off" && field.kind == ParameterKind::Flag {
            continue;
        }
        let value = finite_literal(field)?;
        match field.name.as_str() {
            "area" => input.area = value,
            "perim" => input.perimeter = Some(value),
            "m" => input.multiplier = value,
            "n" => input.series = value,
            "temp" => {
                input.temperature = value + CELSIUS_TO_KELVIN;
                absolute = true;
            }
            // Native PJ is ignored by LTspice; it is not an alias for PERIM
            // in this source dialect.
            "pj" => {}
            name => return Err(unsupported(instance, name)),
        }
    }
    Ok((input, absolute))
}

pub(super) fn instantiate_semiconductor(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    model: &ResolvedModel<'_>,
    context: &ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    let nominal = context.nominal_temperature.to_bits();
    let compiled = model.prepare(PreparationKey::Nominal(nominal), || {
        compile_semiconductor(model, context)
    })?;
    let (input, absolute) = semiconductor_input(instance, context)?;
    let prepared = compiled
        .model
        .bind(input)
        .map_err(|error| source_error(instance, error))?;
    for branch in [prepared.bulk, prepared.sidewall].into_iter().flatten() {
        let resistance = branch.resistance();
        if resistance != 0. && !resistance.recip().is_finite() {
            return Err(source_error(
                instance,
                "nonrepresentable diode series conductance",
            ));
        }
    }
    let shape = u8::from(prepared.bulk.is_some())
        | (u8::from(prepared.sidewall.is_some()) << 1)
        | (u8::from(
            prepared
                .bulk
                .is_some_and(|branch| branch.resistance() != 0.),
        ) << 2)
        | (u8::from(prepared.independent_sidewall) << 3);
    let artifact = model.prepare(
        PreparationKey::LtspiceDiodeTopology {
            nominal,
            shape,
            absolute_temperature: absolute,
        },
        || {
            let mut hash = Sha256::new();
            hash.update(compiled.fingerprint);
            hash.update([shape, u8::from(absolute)]);
            let mut fingerprint = [0; 16];
            fingerprint.copy_from_slice(&hash.finalize()[..16]);
            ltspice::artifact(&compiled.model, input, absolute, fingerprint)
                .map_err(|error| source_error(instance, error))
        },
    )?;
    let mut values = vec![
        input.area,
        input.perimeter.unwrap_or(0.),
        input.multiplier,
        input.series,
        input.gmin,
    ];
    if absolute {
        values.push(input.temperature);
    }
    bind(
        instance,
        nodes,
        artifact.as_ref(),
        &values,
        InstanceTemperature::circuit(context.temperature),
        SourceBinding {
            model_name: model.card().name.clone(),
            designator: 'd',
            parameters: if absolute {
                SEMICONDUCTOR_ABSOLUTE_PARAMETERS
            } else {
                SEMICONDUCTOR_PARAMETERS
            },
            gmin_slot: Some(4),
            model_ambient: AmbientTemperature::Circuit,
            relative_temperature_slot: None,
            absolute_temperature_slot: absolute.then_some(5),
        },
    )
}
