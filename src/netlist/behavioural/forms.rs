//! Validated views of the nonlinear controlled-source card syntax.
//!
//! The ordered AST retains spelling and provenance; consumers use these
//! views rather than independently interpreting the generic setter list.
use crate::netlist::ast::{DeviceInstance, ParameterAssignment, ParameterKind, poly_dimension};
use crate::netlist::bexpr::{BExpr, BExprKind, BehaviouralExpression};
use crate::netlist::expr::{MAX_TREE_NODES, ParameterExpression, SourceSpan};
use crate::primitives::{Real, SpiceError, SpiceResult, parse_spice_number};

use super::malformed;

pub(in crate::netlist) const B_SETTERS: &[&str] = &[
    "m",
    "tc1",
    "tc2",
    "temp",
    "dtemp",
    "reciproctc",
    "reciprocm",
];

pub(in crate::netlist) struct Numeric<'a> {
    pub parameter: &'a ParameterAssignment,
    value: NumericValue<'a>,
}

enum NumericValue<'a> {
    Literal(Real),
    Expression(&'a ParameterExpression),
}

impl<'a> Numeric<'a> {
    fn validate(parameter: &'a ParameterAssignment) -> SpiceResult<Self> {
        let value = match &parameter.kind {
            ParameterKind::Scalar => NumericValue::Literal(
                parse_spice_number(&parameter.value)
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| invalid(parameter, "expected a finite numeric literal"))?,
            ),
            ParameterKind::Expression(expression) => NumericValue::Expression(expression),
            _ => return Err(invalid(parameter, "expected a numeric parameter")),
        };
        Ok(Self { parameter, value })
    }

    pub(super) fn literal_zero(&self) -> bool {
        matches!(self.value, NumericValue::Literal(value) if value == 0.)
    }

    pub(super) fn expression(&self) -> BExpr {
        let parameter = self.parameter;
        let kind = match self.value {
            NumericValue::Literal(value) => BExprKind::Number {
                value,
                spelling: parameter.value.clone(),
            },
            NumericValue::Expression(expression) => BExprKind::Value(Box::new(expression.clone())),
        };
        BExpr {
            kind,
            span: SourceSpan {
                start: parameter.location.clone(),
                end: parameter.location.clone(),
                logical: 0..0,
            },
        }
    }
}

fn invalid(parameter: &ParameterAssignment, message: &str) -> SpiceError {
    SpiceError::parse(
        parameter.location.clone(),
        format!("{}: {message}", parameter.name),
    )
}

fn setters(parameters: &[ParameterAssignment], allowed: &[&str]) -> SpiceResult<()> {
    if allowed == ["m"] && parameters.len() > 1 {
        return Err(invalid(
            &parameters[1],
            "only one multiplier is allowed on this nonlinear form",
        ));
    }
    for parameter in parameters {
        if !allowed.contains(&parameter.name.as_str()) {
            return Err(invalid(parameter, "unexpected nonlinear-source setter"));
        }
        Numeric::validate(parameter)?;
    }
    Ok(())
}

pub(in crate::netlist) struct ValueForm<'a> {
    pub input: &'a ParameterAssignment,
    pub setters: &'a [ParameterAssignment],
}

pub(in crate::netlist) enum TableInput<'a> {
    Expression(&'a ParameterAssignment, &'a BehaviouralExpression),
    Voltage(&'a [String; 2]),
}

pub(in crate::netlist) struct TableForm<'a> {
    pub input: TableInput<'a>,
    pub location: &'a crate::primitives::SourceLoc,
    pub points: Vec<(Numeric<'a>, Numeric<'a>)>,
    pub setters: &'a [ParameterAssignment],
}

pub(in crate::netlist) enum PolyControls<'a> {
    Voltage(&'a [[String; 2]]),
    Current(&'a [ParameterAssignment]),
}

pub(in crate::netlist) struct PolyForm<'a> {
    pub spelling: &'a ParameterAssignment,
    pub dimension: usize,
    pub controls: PolyControls<'a>,
    pub coefficients: Vec<Numeric<'a>>,
    pub setters: &'a [ParameterAssignment],
}

pub(in crate::netlist) enum NonlinearForm<'a> {
    Value(ValueForm<'a>),
    Table(TableForm<'a>),
    Poly(PolyForm<'a>),
}

impl<'a> NonlinearForm<'a> {
    pub(in crate::netlist) fn validate(device: &'a DeviceInstance) -> SpiceResult<Option<Self>> {
        if !matches!(device.designator, 'e'..='h') {
            return Ok(None);
        }
        let is_form =
            |p: &ParameterAssignment| matches!(p.name.as_str(), "value" | "table" | "poly");
        let Some((first, rest)) = device.parameters.split_first() else {
            return Ok(None);
        };
        if !is_form(first) {
            if rest.iter().any(is_form) {
                return Err(malformed(
                    device,
                    "nonlinear form must be the first parameter",
                ));
            }
            return Ok(None);
        }
        if device.model.is_some() {
            return Err(malformed(device, "nonlinear source must not name a model"));
        }
        let [_, _, controls @ ..] = device.nodes.as_slice() else {
            return Err(malformed(device, "nonlinear form needs two output nodes"));
        };
        let multiplier = if matches!(device.designator, 'f' | 'g') {
            &["m"][..]
        } else {
            &[]
        };
        match first.name.as_str() {
            "value" if matches!(device.designator, 'e' | 'g') => {
                if !controls.is_empty() || !matches!(first.kind, ParameterKind::Behavioural(_)) {
                    return Err(malformed(
                        device,
                        "VALUE needs two output nodes and a behavioural expression",
                    ));
                }
                setters(
                    rest,
                    if device.designator == 'e' {
                        B_SETTERS
                    } else {
                        multiplier
                    },
                )?;
                Ok(Some(Self::Value(ValueForm {
                    input: first,
                    setters: rest,
                })))
            }
            "table" if matches!(device.designator, 'e' | 'g') => {
                Self::table(device, first, rest, controls, multiplier).map(Some)
            }
            "poly" => Self::poly(device, first, rest, controls, multiplier).map(Some),
            _ => Err(malformed(device, "unsupported nonlinear source form")),
        }
    }
    fn table(
        device: &'a DeviceInstance,
        first: &'a ParameterAssignment,
        rest: &'a [ParameterAssignment],
        controls: &'a [String],
        multiplier: &[&str],
    ) -> SpiceResult<Self> {
        let input = match (&first.kind, controls) {
            (ParameterKind::Behavioural(expression), []) => {
                TableInput::Expression(first, expression)
            }
            (ParameterKind::Flag, [_, _]) if device.designator == 'e' && first.value.is_empty() => {
                TableInput::Voltage(
                    controls
                        .try_into()
                        .map_err(|_| malformed(device, "invalid TABLE voltage controls"))?,
                )
            }
            _ => {
                return Err(malformed(
                    device,
                    "TABLE needs an expression or the four-node E form",
                ));
            }
        };
        let mut remaining = rest;
        let mut points = Vec::new();
        while let [x, y, tail @ ..] = remaining {
            if x.name != "x" || y.name != "y" {
                break;
            }
            points.push((Numeric::validate(x)?, Numeric::validate(y)?));
            remaining = tail;
        }
        if points.is_empty() {
            return Err(malformed(device, "TABLE needs (x, y) pairs"));
        }
        setters(remaining, multiplier)?;
        Ok(Self::Table(TableForm {
            input,
            location: &first.location,
            points,
            setters: remaining,
        }))
    }

    fn poly(
        device: &'a DeviceInstance,
        first: &'a ParameterAssignment,
        rest: &'a [ParameterAssignment],
        controls: &'a [String],
        multiplier: &[&str],
    ) -> SpiceResult<Self> {
        let dimension = if first.kind == ParameterKind::Scalar {
            poly_dimension(&first.value)
        } else {
            None
        }
        .ok_or_else(|| malformed(device, "POLY needs a positive integer dimension"))?;
        let (controls, remaining) = if matches!(device.designator, 'e' | 'g') {
            let (pairs, tail) = controls.as_chunks::<2>();
            if !tail.is_empty() || pairs.len() != dimension {
                return Err(malformed(
                    device,
                    "POLY has the wrong number of controlling nodes",
                ));
            }
            (PolyControls::Voltage(pairs), rest)
        } else {
            let Some((currents, remaining)) = rest.split_at_checked(dimension) else {
                return Err(malformed(device, "POLY needs controlling sources"));
            };
            if !controls.is_empty()
                || currents.iter().any(|p| {
                    p.name != "control" || p.kind != ParameterKind::Instance || p.value.is_empty()
                })
            {
                return Err(malformed(
                    device,
                    "POLY needs instance-name controlling sources",
                ));
            }
            (PolyControls::Current(currents), remaining)
        };
        let count = remaining.iter().take_while(|p| p.name == "coef").count();
        if count == 0 {
            return Err(malformed(device, "POLY needs at least one coefficient"));
        }
        if count > MAX_TREE_NODES {
            return Err(malformed(
                device,
                "POLY coefficient count exceeds expression node limit",
            ));
        }
        let (coefficients, remaining) = remaining.split_at(count);
        let coefficients = coefficients
            .iter()
            .map(Numeric::validate)
            .collect::<SpiceResult<_>>()?;
        setters(remaining, multiplier)?;
        Ok(Self::Poly(PolyForm {
            spelling: first,
            dimension,
            controls,
            coefficients,
            setters: remaining,
        }))
    }
}
