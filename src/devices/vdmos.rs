//! Source-level binding for the independently compiled VDMOS kernel.
//! Numerical residuals, derivatives and topology layout are producer-owned.

use std::sync::Arc;

use model_kernel_api::Artifact;
use model_kernel_vdmos::{
    Polarity,
    artifact::{VdmosKernel, artifact},
    compile_ngspice,
};
use sha2::{Digest, Sha256};

use crate::devices::models::{PreparationKey, ResolvedModel};
use crate::devices::prepared_model::{AmbientTemperature, PreparedDevice};
use crate::devices::{Device, ModelContext};
use crate::netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind};
use crate::primitives::{NodeAppend, Real, SpiceError, SpiceResult};

fn scalar(parameter: &ParameterAssignment) -> SpiceResult<Real> {
    if parameter.kind != ParameterKind::Scalar {
        return Err(SpiceError::Unsupported {
            feature: format!(
                "VDMOS parameter {} requires a resolved scalar",
                parameter.name
            ),
            location: Some(parameter.location.clone()),
        });
    }
    crate::primitives::parse_spice_number(&parameter.value)
        .filter(|value| value.is_finite())
        .ok_or_else(|| SpiceError::circuit(format!("invalid VDMOS parameter {}", parameter.name)))
}

fn prepare_model(
    resolved: &ResolvedModel<'_>,
    nominal: Real,
    thermal: bool,
) -> SpiceResult<Arc<Artifact<VdmosKernel>>> {
    let card = resolved.card();
    let mut polarity = if card.base == "vdmosp" {
        Polarity::P
    } else {
        Polarity::N
    };
    let mut values = Vec::with_capacity(card.parameters.len());
    for parameter in &card.parameters {
        match parameter.name.as_str() {
            "pchan" | "nchan" if parameter.kind == ParameterKind::Flag => {
                polarity = if parameter.name == "pchan" {
                    Polarity::P
                } else {
                    Polarity::N
                };
            }
            "level" => {
                if scalar(parameter)?.to_bits() != 1_f64.to_bits() {
                    return Err(SpiceError::circuit("VDMOS has only native level 1"));
                }
            }
            name => values.push((name, scalar(parameter)?)),
        }
    }
    // Feed the compiler's own temperature setter, rather than editing its
    // derived data. An explicit model TNOM takes precedence.
    if !values.iter().any(|(name, _)| *name == "tnom") {
        values.push(("tnom", nominal));
    }
    let mut digest = Sha256::new();
    digest.update(b"vdmos-abi3\0");
    digest.update([u8::from(polarity == Polarity::P), u8::from(thermal)]);
    for (name, value) in &values {
        digest.update(name.as_bytes());
        digest.update([0]);
        digest.update(value.to_bits().to_le_bytes());
    }
    let hash = digest.finalize();
    let mut fingerprint = [0; 16];
    fingerprint.copy_from_slice(&hash[..16]);
    let model = compile_ngspice(polarity, values)
        .map_err(|error| SpiceError::circuit(format!("{}: {error}", card.name)))?;
    artifact(Arc::new(model), thermal, fingerprint)
        .map_err(|error| SpiceError::circuit(format!("{}: {error}", card.name)))
}

/// Resolve the source instance and bind the producer's immutable row layout.
pub(super) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut NodeAppend<'_>,
    resolved: &ResolvedModel<'_>,
    context: &ModelContext,
) -> SpiceResult<Box<dyn Device>> {
    let mut multiplier = 1.;
    let mut temperature = None;
    let mut dtemp = 0.;
    let mut thermal = false;
    for parameter in &instance.parameters {
        match parameter.name.as_str() {
            "m" => multiplier = scalar(parameter)?,
            "temp" => temperature = Some(scalar(parameter)?),
            "dtemp" => dtemp = scalar(parameter)?,
            "thermal" if parameter.kind == ParameterKind::Flag => thermal = true,
            other => {
                return Err(SpiceError::Unsupported {
                    feature: format!(
                        "{}: unsupported VDMOS instance setter {other}",
                        instance.name
                    ),
                    location: Some(parameter.location.clone()),
                });
            }
        }
    }
    if temperature.is_some() && dtemp != 0. {
        return Err(SpiceError::circuit("VDMOS cannot combine TEMP and DTEMP"));
    }
    if thermal
        && !resolved
            .card()
            .parameters
            .iter()
            .any(|parameter| parameter.name == "rthjc")
    {
        return Err(SpiceError::circuit("thermal VDMOS needs explicit RTHJC"));
    }
    let shared = resolved.prepare(
        PreparationKey::NativeArtifact {
            nominal: context.nominal_temperature.to_bits(),
            thermal,
        },
        || prepare_model(resolved, context.nominal_temperature, thermal),
    )?;
    let device = PreparedDevice::bind(
        Arc::clone(&shared),
        &instance.name,
        &instance.nodes,
        nodes,
        &[multiplier],
        context.temperature,
        temperature.map_or(
            AmbientTemperature::CircuitOffset(dtemp),
            AmbientTemperature::Fixed,
        ),
    )?
    .with_limiting()
    .with_setter("m", 0)?
    .with_identity(
        resolved.card().name.clone(),
        'm',
        crate::devices::setup::CDeviceType::Vdmos,
        crate::devices::noise::NoiseFamily::Vdmos,
    );
    Ok(Box::new(device))
}
