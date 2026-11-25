//! BSIM4.8 (`bsim4/`, levels 14 and 54): the generated input schema and a
//! statement-by-statement port of C's setup, temperature, check, load,
//! small-signal, pole-zero, noise and convergence routines.
//!
//! The C-shaped modules keep C's struct members (`fields.rs`, generated from
//! `bsim4def.h`) and see the circuit through `host.rs`; `device.rs` wires
//! them into the solver. Equations are verified against the pinned FMA-free
//! C oracle and independent physical numerical invariants. Deliberate fused
//! arithmetic and stable high-dynamic-range formulas need not preserve C's
//! intermediate rounding (`tests/bsim4_device.rs`, opt-in `c_bsim4_*` tests,
//! and `docs/port/BSIM.md`).
//!
//! BSIM4 was developed by the UC Berkeley BSIM Research Group. The translated
//! files follow the upstream notice: Copyright (c) 2025 University of
//! California, licensed under the Educational Community License, Version 2.0
//! (see `NOTICE` and `LICENSE-ECL-2.0`).
use crate::devices::ifparam::{Device, ParameterTable};
use std::sync::Arc;

use crate::devices::bsim_model_inputs::{self, OrderedModelInputs};
use crate::devices::models::{PreparationKey, ResolvedModel};
use crate::devices::schema::temperature_kelvin;
use crate::netlist::ast::{DeviceInstance, ModelCard, ParameterAssignment};
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};

use crate::devices::ifparam::version_text;

pub use crate::devices::ifparam::{
    Access, AppliedParameter, Diagnostic, InputType, Inputs, Parameter, Value,
};

mod defaults_data;

// The C-shaped port. These modules follow their C files statement by
// statement over C's own struct members (`fields.rs`), so C's mixed-case
// member and local names are kept.
mod acld;
mod apply;
mod asks;
mod check;
mod cvtest;
mod device;
mod elements;
pub(super) mod fields;
mod geo;
mod host;
mod load;
mod noise;
/// Generated `IFparm` order, identifiers and setter/given-field mappings.
pub mod parameters;
mod pzld;
mod sens;
mod set;
mod stamp_common;
mod temp;

/// The generated literal model defaults in C setup order (`b4set.c`).
///
/// Generated from `b4set.c`'s given-guarded literal assignments. Independent
/// defaults execute as typed generated assignments; selector fallbacks and
/// conditional defaults remain in the ordered setup stages.
#[must_use]
pub const fn literal_defaults() -> &'static [(&'static str, f64)] {
    defaults_data::LITERALS
}

/// The BSIM4 interface descriptor the shared input plumbing uses.
pub(in crate::devices) static DEVICE: Device = Device {
    reference: "bsim4/b4.c",
    model: ParameterTable::new(parameters::MODEL),
    instance: ParameterTable::new(parameters::INSTANCE),
    uninteresting: parameters::UNINTERESTING,
    ic_names: ["icvds", "icvgs", "icvbs"],
    model_real: bsim4_model_real,
};

/// `b4mpar.c`: TNOM adds `CONSTCtoK`; NDEP converts above 1e20 and NSD/NGATE
/// above 1e23 (and their L/W/P coefficients likewise) from cm^-3 to m^-3.
fn bsim4_model_real(p: &Parameter, value: Real, location: &SourceLoc) -> SpiceResult<Real> {
    if p.canonical == "tnom" {
        return temperature_kelvin(value, location);
    }
    let field = p.c_field.unwrap_or_default();
    let threshold = if field.ends_with("ndep") {
        1e20
    } else if field.ends_with("nsd") || field.ends_with("ngate") {
        1e23
    } else {
        Real::INFINITY
    };
    Ok(if value > threshold {
        value * 1e-6
    } else {
        value
    })
}

/// C's `IF_UNINTERESTING` rows, as `TABLE:keyword` pairs.
#[must_use]
pub const fn uninteresting() -> &'static [&'static str] {
    DEVICE.uninteresting
}

/// Validated explicit model inputs, distinct from a setup/defaulted model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelParameters {
    /// Applied model setters; unset/defaulted values are deliberately absent.
    pub inputs: Inputs,
    /// Last effective polarity flag, initially the model family (+1 NMOS/-1 PMOS).
    pub polarity: i8,
    instance_defaults: Arc<[ParameterAssignment]>,
    location: SourceLoc,
}

impl ModelParameters {
    /// Apply reversed model-level instance defaults, then instance setters
    /// (`inpgmod.c`/`inpdpar.c`). C discards model-level `m`, not instance `m`.
    ///
    /// # Errors
    /// Unknown/ask-only setters, invalid shapes, nonfinite input, integer or
    /// scaled-geometry overflow. No physical setup check is performed: the
    /// `b4set.c`/`b4temp.c` stage is not ported.
    pub fn instance_parameters(
        &self,
        instance: &DeviceInstance,
        scale: Real,
    ) -> SpiceResult<Inputs> {
        if instance.designator != 'm' || !scale.is_finite() || scale <= 0. {
            return Err(SpiceError::parse(
                instance.location.clone(),
                "BSIM4 needs an M instance and a finite positive scale",
            ));
        }
        // BSIM4 has three- and four-terminal forms; the bulk node is optional.
        if !matches!(instance.nodes.len(), 3 | 4) {
            return Err(SpiceError::parse(
                instance.location.clone(),
                "BSIM4 needs three or four terminals: drain, gate, source[, bulk]",
            ));
        }
        bsim_model_inputs::instance_inputs(&DEVICE, &self.instance_defaults, instance, scale)
    }
}

/// Validate explicit BSIM4 model setters, including their instance-default
/// fallback, in C order. This does **not** apply `b4set.c` defaults or claim
/// that `b4check.c`/`b4soachk.c` diagnostics have run.
///
/// # Errors
/// Wrong family/level/version, unknown/read-only setters or invalid typed input.
pub fn model_parameters(model: &ResolvedModel<'_>) -> SpiceResult<ModelParameters> {
    model
        .prepare(PreparationKey::Inputs, || parse_model_parameters(model))
        .map(|parameters| (*parameters).clone())
}

fn parse_model_parameters(model: &ResolvedModel<'_>) -> SpiceResult<ModelParameters> {
    if model.levels().backend != crate::devices::ModelBackend::Bsim4 {
        return Err(SpiceError::parse(
            model.card().location.clone(),
            "BSIM4 input schema requires NMOS/PMOS level 14 or 54",
        ));
    }
    let OrderedModelInputs {
        inputs,
        polarity,
        instance_defaults,
    } = bsim_model_inputs::dispatch(&DEVICE, model)?;
    Ok(ModelParameters {
        inputs,
        polarity,
        instance_defaults,
        location: model.card().location.clone(),
    })
}

/// Select the BSIM4 directory from the **first** VERSION token, as
/// `inpdomod.c`/`INPfindVer` do. Only the current `bsim4/` family is being
/// ported; the `bsim4v5`/`bsim4v6`/`bsim4v7` variants are refused by name.
///
/// # Errors
/// Unsupported versions, or nonliteral version input.
pub fn select_version(card: &ModelCard) -> SpiceResult<&str> {
    let version = card
        .parameters
        .iter()
        .find(|p| p.name.eq_ignore_ascii_case("version"));
    let Some(p) = version else {
        return Ok("");
    };
    let text = version_text(p)?;
    let reference = if text.starts_with("4.5") {
        "src/spicelib/devices/bsim4v5/"
    } else if text.starts_with("4.6") {
        "src/spicelib/devices/bsim4v6/"
    } else if text.starts_with("4.7") {
        "src/spicelib/devices/bsim4v7/"
    } else if text.starts_with("4.8") || text.is_empty() {
        return Ok(text);
    } else {
        "src/spicelib/parser/inpdomod.c, inpfindv.c (no supported BSIM4 VERSION)"
    };
    Err(SpiceError::not_yet_ported(
        format!("{}: model '{}' BSIM4 version {text}", p.location, card.name),
        reference,
    ))
}

/// The factory entry: [`device::Bsim4`] runs `b4set.c`/`b4temp.c` before
/// interning any node, so every unported path fails before a partial
/// circuit exists.
pub(crate) fn instantiate(
    instance: &DeviceInstance,
    nodes: &mut crate::primitives::NodeAppend<'_>,
    model: &ResolvedModel<'_>,
    context: &crate::devices::ModelContext,
) -> SpiceResult<Box<dyn crate::devices::Device>> {
    device::Bsim4::instantiate(instance, nodes, model, context)
}
