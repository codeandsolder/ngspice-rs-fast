//! Source-selected reporting laws over physical current and voltage rows.
//! Reporting is deliberately separate from electrical loading.
use std::collections::BTreeMap;

use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

/// Which physical current an output convention starts from.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum CurrentReference {
    /// Current into the positive external terminal.
    #[default]
    Terminal,
    /// Main storage or ideal forcing branch current.
    Primary,
}

/// A reported current may include a nonphysical voltage-proportional term.
///
/// This term is an output convention only. It never changes KCL, noise,
/// charge, flux, convergence, or any controlled device's physical load.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct CurrentReportingLaw {
    /// Physical current coordinate to report.
    pub reference: CurrentReference,
    /// Signed additive conductance multiplying the external differential voltage.
    pub voltage_conductance: Real,
}

/// Explicit current conventions for each kind of analysis output.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct CurrentReporting {
    /// DC sweep and operating-point reporting.
    pub steady: CurrentReportingLaw,
    /// Complex frequency-domain reporting.
    pub small_signal: CurrentReportingLaw,
    /// Time-domain reporting.
    pub transient: CurrentReportingLaw,
}

impl CurrentReporting {
    pub(super) fn validate(self) -> SpiceResult<Self> {
        if [self.steady, self.small_signal, self.transient]
            .iter()
            .any(|law| !law.voltage_conductance.is_finite())
        {
            return Err(SpiceError::circuit(
                "nonfinite reported-current conductance",
            ));
        }
        Ok(self)
    }

    pub(super) fn needs_primary(self) -> bool {
        [self.steady, self.small_signal, self.transient]
            .iter()
            .any(|law| law.reference == CurrentReference::Primary)
    }
}

/// Normalized scalar contract. Relative parallel-resistor factors multiply
/// the actual aggregate physical shunt conductance after multiplicity.
pub(super) fn parse(
    values: &BTreeMap<&str, Real>,
    parallel_conductance: Real,
) -> SpiceResult<Option<CurrentReporting>> {
    if !values.keys().any(|key| key.starts_with("pn_report_")) {
        return Ok(None);
    }
    let law = |phase: &str| -> SpiceResult<CurrentReportingLaw> {
        let get = |suffix: &str, default| {
            values
                .get(format!("pn_report_{phase}{suffix}").as_str())
                .copied()
                .unwrap_or(default)
        };
        let reference = match get("", 1.) {
            0. => CurrentReference::Primary,
            1. => CurrentReference::Terminal,
            _ => {
                return Err(SpiceError::circuit(
                    "reported current selector must be 0 (primary) or 1 (terminal)",
                ));
            }
        };
        Ok(CurrentReportingLaw {
            reference,
            voltage_conductance: get("_rparallel_factor", 0.)
                .mul_add(parallel_conductance, get("_g", 0.)),
        })
    };
    CurrentReporting {
        steady: law("dc")?,
        small_signal: law("ac")?,
        transient: law("tran")?,
    }
    .validate()
    .map(Some)
}

/// Resolve a physical controlling-current selector independently of reporting.
pub(super) fn parse_control(
    values: &BTreeMap<&str, Real>,
) -> SpiceResult<Option<CurrentReference>> {
    match values.get("pn_control").copied() {
        None => Ok(None),
        Some(0.) => Ok(Some(CurrentReference::Primary)),
        Some(1.) => Ok(Some(CurrentReference::Terminal)),
        _ => Err(SpiceError::circuit(
            "current control selector must be 0 (primary) or 1 (terminal)",
        )),
    }
}

/// Current identity description attached to an ideal current probe during elaboration.
#[derive(Debug, Clone)]
pub(in crate::devices) struct CurrentReportSpec {
    pub(in crate::devices) original: String,
    pub(in crate::devices) primary: String,
    pub(in crate::devices) terminals: [NodeId; 2],
    pub(in crate::devices) reporting: Option<CurrentReporting>,
    pub(in crate::devices) control: Option<CurrentReference>,
}
