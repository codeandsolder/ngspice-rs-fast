//! Typed adjoint derivatives for smooth optimizer objectives.
//!
//! One physical sensitivity analysis solves the operating point and adjoint,
//! then the existing device perturbation walk projects the requested partials.
//! The compiled candidate parameter DAG supplies scalar chain coefficients
//! without additional circuit solves. All unsupported or nondifferentiable
//! coordinates retain explicit status; arbitrary black-box objectives continue
//! to use the independent candidate evaluation API.

use crate::analysis::AnalysisInput;
use crate::analysis::optimization::parameters::{CircuitCandidate, GradientSiteFailure};
use crate::analysis::reduction::CompensatedSum;
use crate::analysis::sens::{self, PhysicalParameter, PhysicalPartial, PhysicalProbe};
use crate::netlist::eval::EvalLimits;
use crate::primitives::{AnalysisKind, Complex, Real, SpiceResult};

/// A linear readout whose physical derivative can be projected with one
/// adjoint, independently of the number of tunable parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DirectOutput {
    /// Voltage relative to ground or another named node.
    Voltage {
        /// Positive node name.
        pos: String,
        /// Negative node name, absent for ground.
        neg: Option<String>,
    },
    /// Current through a source with a findable branch.
    Current {
        /// Source carrying a findable current branch.
        source: String,
    },
}

impl From<&DirectOutput> for PhysicalProbe {
    fn from(value: &DirectOutput) -> Self {
        match value {
            DirectOutput::Voltage { pos, neg } => Self::Voltage {
                pos: pos.clone(),
                neg: neg.clone(),
            },
            DirectOutput::Current { source } => Self::Current {
                source: source.clone(),
            },
        }
    }
}

/// One direct DC/AC objective; transient and event objectives report a typed unsupported status.
#[derive(Debug, Clone, PartialEq)]
pub struct DerivativeRequest {
    /// Objective variable with explicit node/branch identity.
    pub output: DirectOutput,
    /// DC operating point or AC small-signal analysis.
    pub analysis: AnalysisKind,
    /// Positive finite AC frequency, absent for DC.
    pub frequency: Option<Real>,
}

/// A derivative can be absent without making other input coordinates invalid.
#[derive(Debug, Clone, PartialEq)]
pub enum DerivativeStatus {
    /// Physical gradient with respect to this root parameter coordinate.
    Defined(Complex),
    /// Analytically unsupported structure, device setter or objective.
    Unsupported(String),
    /// A step encountered discontinuous, non-finite or poorly resolved
    /// physical parameters; no synthetic zero/gradient is reported.
    NonSmooth(String),
}

/// The result for a single independent input in the parameter-plan order.
#[derive(Debug, Clone, PartialEq)]
pub struct InputDerivative {
    /// The original coordinate name.
    pub input: String,
    /// Valid differentiated result or an explicit reason for invalidity.
    pub status: DerivativeStatus,
}

/// Evaluate requested input derivatives without one simulation per input.
///
/// Nonlinear AC is rejected because C-reset sensitivity is not the biased AC
/// derivative. Unsupported parameters and objectives have typed statuses;
/// ordinary black-box objective evaluation remains available unchanged.
///
/// # Errors
/// Failure to assemble/solve the common operating point or the physical
/// sensitivity adjoint, or invalid resource allocation.
pub fn evaluate(
    candidate: &mut CircuitCandidate<'_>,
    request: &DerivativeRequest,
    limits: EvalLimits,
) -> SpiceResult<Vec<InputDerivative>> {
    let status = match (request.analysis, request.frequency) {
        (AnalysisKind::OperatingPoint, None) => None,
        (AnalysisKind::Ac, Some(f)) if f.is_finite() && f > 0. => None,
        (AnalysisKind::Ac, _) => Some("AC derivatives require a positive finite frequency"),
        _ => Some("only fixed-point DC and small-signal AC direct readouts are differentiable"),
    };
    let links = candidate.gradient_sites(limits);
    let mut entries: Vec<_> = links
        .iter()
        .enumerate()
        .map(|(index, links)| InputDerivative {
            input: candidate.parameter_input_name(index).to_owned(),
            status: match links {
                Ok(_) => DerivativeStatus::Defined(Complex::ZERO),
                Err(GradientSiteFailure::Unsupported(reason)) => {
                    DerivativeStatus::Unsupported(reason.clone())
                }
                Err(GradientSiteFailure::NonSmooth(reason)) => {
                    DerivativeStatus::NonSmooth(reason.clone())
                }
            },
        })
        .collect();
    if let Some(reason) = status {
        for entry in &mut entries {
            entry.status = DerivativeStatus::Unsupported(reason.to_owned());
        }
        return Ok(entries);
    }
    if request.analysis == AnalysisKind::Ac
        && candidate
            .circuit()
            .devices()
            .iter()
            .any(|device| device.is_nonlinear())
    {
        for (entry, links) in entries.iter_mut().zip(&links) {
            if links.as_ref().is_ok_and(|links| !links.is_empty()) {
                entry.status = DerivativeStatus::Unsupported(
                    "nonlinear AC sensitivity uses C's reset state, not the biased AC objective"
                        .into(),
                );
            }
        }
        return Ok(entries);
    }

    let mut selected: Vec<PhysicalParameter> = Vec::new();
    for valid in links.iter().filter_map(|result| result.as_ref().ok()) {
        for link in valid {
            if !selected.contains(&link.parameter) {
                selected.push(link.parameter.clone());
            }
        }
    }
    if selected.is_empty() {
        return Ok(entries);
    }
    let config = candidate.config().clone();
    let context = config.context();
    let plan = config.plan(AnalysisInput::new(AnalysisKind::OperatingPoint))?;
    let mut physical = candidate.physical_sensitivity_circuit()?;
    let partials = sens::physical_partials(
        &mut physical,
        &plan,
        &context,
        &PhysicalProbe::from(&request.output),
        request.frequency,
        &selected,
    )?;
    for (result, bound) in entries.iter_mut().zip(links) {
        if let Ok(bound) = bound {
            result.status = project_input(&bound, &selected, &partials);
        }
    }
    Ok(entries)
}

/// Project one coordinate through its validated device partials with
/// compensated summation and explicit invalid-result statuses.
fn project_input(
    bound: &[crate::analysis::optimization::parameters::GradientSite],
    selected: &[PhysicalParameter],
    partials: &[PhysicalPartial],
) -> DerivativeStatus {
    let mut real = CompensatedSum::default();
    let mut imaginary = CompensatedSum::default();
    for site in bound {
        let Some(slot) = selected
            .iter()
            .position(|parameter| *parameter == site.parameter)
        else {
            return DerivativeStatus::Unsupported(
                "typed physical parameter selection lost its slot".to_owned(),
            );
        };
        match partials[slot] {
            PhysicalPartial::Valid(value) => {
                CompensatedSum::with_initial(value.re).multiply_into(site.scale, &mut real);
                CompensatedSum::with_initial(value.im).multiply_into(site.scale, &mut imaginary);
            }
            PhysicalPartial::NonSmooth => {
                return DerivativeStatus::NonSmooth(format!(
                    "{} has no stable physical derivative",
                    site.parameter.describe()
                ));
            }
            PhysicalPartial::Missing => {
                return DerivativeStatus::Unsupported(format!(
                    "{} has no typed device sensitivity record",
                    site.parameter.describe()
                ));
            }
        }
    }
    let value = Complex::new(real.total(), imaginary.total());
    if value.is_finite() {
        DerivativeStatus::Defined(value)
    } else {
        DerivativeStatus::NonSmooth("nonfinite chain-rule projection".to_owned())
    }
}
