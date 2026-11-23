//! CPL model interpretation after deck/subcircuit parameter literalization.
//!
//! This stage binds an ordered P card to a model declaration and expands its
//! packed physical matrices. It does not construct an executable P device.
//! Model matrices stay distinct, including mutual coupling and negative
//! Maxwell capacitance off-diagonal entries.

use super::plan::{CplPlan, MAX_CONDUCTORS};
use crate::netlist::ast::{
    DeviceInstance, ModelCard, ParameterAssignment, ParameterKind, PositionedValue,
};
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult, parse_spice_number};

fn invalid(at: &SourceLoc, message: impl std::fmt::Display) -> SpiceError {
    SpiceError::parse(at.clone(), message.to_string())
}

fn scalar(parameter: &ParameterAssignment) -> SpiceResult<Real> {
    if !matches!(parameter.kind, ParameterKind::Scalar) {
        return Err(invalid(
            &parameter.location,
            format!(
                "CPL {} requires a literalized numeric scalar, got {}",
                parameter.name, parameter.value
            ),
        ));
    }
    parse_spice_number(&parameter.value)
        .filter(|value| value.is_finite())
        .ok_or_else(|| invalid(&parameter.location, "CPL scalar is nonnumeric or nonfinite"))
}

fn vector(
    parameter: &ParameterAssignment,
    components: &[PositionedValue],
) -> SpiceResult<Vec<Real>> {
    components
        .iter()
        .map(|component| {
            parse_spice_number(&component.text)
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    invalid(
                        &component.location,
                        format!(
                            "CPL {} matrix component {} must be literalized and finite",
                            parameter.name, component.text
                        ),
                    )
                })
        })
        .collect()
}

/// Bind an instance to validated Maxwell matrices after top-level or
/// subcircuit parameter literalization.
///
/// Ordered model and instance setters obey last-assignment-wins, with an
/// instance length taking precedence over any model length. No physical
/// coefficient is synthesized except the optional zero R/G matrices.
///
/// # Errors
/// Wrong device/model, malformed terminal groups, missing physical matrices,
/// unliteralized parameters, matrix dimensions, passivity or modal errors.
pub fn bind(instance: &DeviceInstance, card: &ModelCard) -> SpiceResult<CplPlan> {
    if instance.designator != 'p' {
        return Err(invalid(
            &instance.location,
            "CPL model requires a P instance",
        ));
    }
    if !matches!(card.base.as_str(), "cpl" | "cpllines") {
        return Err(invalid(&card.location, "expected a CPL model declaration"));
    }
    if instance.model.as_deref() != Some(card.name.as_str()) {
        return Err(invalid(
            &instance.location,
            format!("P instance does not reference CPL model {}", card.name),
        ));
    }
    let Some(ports) = instance.nodes.len().checked_sub(2) else {
        return Err(invalid(
            &instance.location,
            "CPL terminal group is incomplete",
        ));
    };
    if ports % 2 != 0 || !(1..=MAX_CONDUCTORS).contains(&(ports / 2)) {
        return Err(invalid(
            &instance.location,
            "CPL requires n near conductors, a near return, n far conductors, and a far return (n=1..8)",
        ));
    }
    let conductors = ports / 2;
    let mut values: [Option<Vec<Real>>; 4] = std::array::from_fn(|_| None);
    let mut model_length = None;
    for parameter in &card.parameters {
        if parameter.name == "length" {
            model_length = Some(scalar(parameter)?);
            continue;
        }
        let key = match parameter.name.as_str() {
            "r" => 0,
            "l" => 1,
            "g" => 2,
            "c" => 3,
            _ => {
                return Err(invalid(
                    &parameter.location,
                    format!("unknown CPL model setter {}", parameter.name),
                ));
            }
        };
        let ParameterKind::NumericVector(components) = &parameter.kind else {
            return Err(invalid(
                &parameter.location,
                format!("CPL {} needs a numeric matrix vector", parameter.name),
            ));
        };
        values[key] = Some(vector(parameter, components)?);
    }
    let mut length = model_length;
    for parameter in &instance.parameters {
        if parameter.name != "length" {
            return Err(invalid(
                &parameter.location,
                format!("unsupported CPL instance setter {}", parameter.name),
            ));
        }
        length = Some(scalar(parameter)?);
    }
    let length = length.ok_or_else(|| {
        invalid(
            &instance.location,
            "CPL line length must be defined by the instance or model",
        )
    })?;
    let empty = [];
    let r = values[0].as_deref().unwrap_or(&empty);
    let l = values[1].as_deref().unwrap_or(&empty);
    let g = values[2].as_deref().unwrap_or(&empty);
    let c = values[3].as_deref().unwrap_or(&empty);
    CplPlan::prepare(conductors, length, r, l, g, c).map_err(|error| {
        invalid(
            &instance.location,
            format!("CPL {}: {error}", instance.name),
        )
    })
}
