//! Ordered BSIM model setter dispatch shared by the family-specific schemas.
//!
//! C's `INPdomodel`/`INPgetMod` admit the family/version before invoking setters.
//! `create_model` prepends model-level instance defaults; `INPdevParse` applies those
//! defaults before the instance's own parameters. The typed `IFparm` descriptor
//! controls all validation, conversions, aliases and given-bit provenance.

use std::sync::Arc;

use crate::devices::ifparam::{Device, Inputs};
use crate::devices::models::{ModelFamily, ResolvedModel};
use crate::netlist::ast::{DeviceInstance, ParameterAssignment};
use crate::primitives::{Real, SpiceResult};

/// Explicit setter dispatch result, not a prepared physical model.
pub(super) struct OrderedModelInputs {
    pub(super) inputs: Inputs,
    pub(super) polarity: i8,
    pub(super) instance_defaults: Arc<[ParameterAssignment]>,
}

pub(super) fn dispatch(
    device: &'static Device,
    model: &ResolvedModel<'_>,
) -> SpiceResult<OrderedModelInputs> {
    let mut inputs = Inputs::new(device, true);
    let mut polarity = if model.family() == ModelFamily::Nmos {
        1
    } else {
        -1
    };
    let mut defaults = Vec::new();
    for p in &model.card().parameters {
        if p.name.eq_ignore_ascii_case("level") {
            continue;
        }
        if device.model.get(&p.name).is_some() {
            inputs.apply(p, 1.)?;
            if p.name.eq_ignore_ascii_case("nmos") {
                polarity = 1;
            }
            if p.name.eq_ignore_ascii_case("pmos") {
                polarity = -1;
            }
        } else {
            // Validate even M, which C discards at model level. Preserve the
            // source assignment so geometry scale is applied only later.
            let mut check = Inputs::new(device, false);
            check.apply(p, 1.)?;
            if !p.name.eq_ignore_ascii_case("m") {
                defaults.push(p.clone());
            }
        }
    }
    defaults.reverse();
    Ok(OrderedModelInputs {
        inputs,
        polarity,
        instance_defaults: defaults.into(),
    })
}

pub(super) fn instance_inputs(
    device: &'static Device,
    defaults: &[ParameterAssignment],
    instance: &DeviceInstance,
    scale: Real,
) -> SpiceResult<Inputs> {
    let mut inputs = Inputs::new(device, false);
    for p in defaults.iter().chain(&instance.parameters) {
        inputs.apply(p, scale)?;
    }
    Ok(inputs)
}
