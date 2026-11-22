//! Classic MOS charge evolution. The channel evaluation supplies intrinsic
//! depletion charges and Meyer half-capacitances; this module owns their
//! accepted/trial history and companion currents, not matrix topology.

use super::{Eval, slot};
use crate::devices::nonlinear::charge::ChargeBranch;
use crate::devices::{DeviceState, IterationPhase};
use crate::maths::integrator::Coefficients;
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Fold the bulk junction companions into the device-frame conduction
/// terms in C's drain-then-source order. DC still records complete state.
pub(super) fn integrate_bulk(
    e: &mut Eval,
    states: &mut DeviceState<'_>,
    integration: Option<&Coefficients>,
) -> SpiceResult<()> {
    let drain = ChargeBranch::new(slot::QBD, e.qbd, e.capbd);
    let source = ChargeBranch::new(slot::QBS, e.qbs, e.capbs);
    drain.store(states)?;
    source.store(states)?;
    if let Some(coefficients) = integration {
        let companion = drain.integrate(states, coefficients)?;
        e.gbd += companion.conductance;
        e.cbd += companion.derivative;
        e.cd -= companion.derivative;
        let companion = source.integrate(states, coefficients)?;
        e.gbs += companion.conductance;
        e.cbs += companion.derivative;
    } else {
        drain.zero_derivative(states)?;
        source.zero_derivative(states)?;
    }
    Ok(())
}

/// The three gate branches in source, drain, bulk order.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct GateCompanions {
    pub(super) conductance: [Real; 3],
    pub(super) equivalent: [Real; 3],
}

fn accepted(states: &DeviceState<'_>, slot: usize, name: &str, label: &str) -> SpiceResult<Real> {
    states
        .accepted(1, slot)
        .ok_or_else(|| SpiceError::circuit(format!("{name}: missing accepted {label} state")))
}

/// C's Meyer half-capacitances averaged with the accepted half; predictor
/// extrapolation uses accepted charges, other transient loads accumulate
/// voltage increments. MODEINITTRAN stores charge but omits gate companions.
pub(super) fn integrate_gates(
    e: &Eval,
    overlap: [Real; 3],
    states: &mut DeviceState<'_>,
    integration: Option<&Coefficients>,
    name: &str,
    label: &str,
) -> SpiceResult<GateCompanions> {
    let gate = [e.vgs, e.vgd, e.vgb];
    let transient = integration.is_some();
    let predict = states.phase() == IterationPhase::Predict;
    let mut cap = [0.; 3];
    for (k, value) in cap.iter_mut().enumerate() {
        states.set(slot::CAPG[k], e.half[k])?;
        *value = if transient {
            e.half[k] + accepted(states, slot::CAPG[k], name, label)? + overlap[k]
        } else {
            2. * e.half[k] + overlap[k]
        };
    }
    let mut charge = [0.; 3];
    if let Some(coefficients) = integration {
        if predict {
            let xfact = coefficients.predictor_ratio();
            for (k, value) in charge.iter_mut().enumerate() {
                *value = states.predicted(slot::QG[k], xfact).ok_or_else(|| {
                    SpiceError::circuit(format!("{name}: missing accepted {label} charge"))
                })?;
            }
        } else {
            let vgs1 = accepted(states, slot::VGS, name, label)?;
            let gate1 = [
                vgs1,
                vgs1 - accepted(states, slot::VDS, name, label)?,
                vgs1 - accepted(states, slot::VBS, name, label)?,
            ];
            for (k, value) in charge.iter_mut().enumerate() {
                *value =
                    (gate[k] - gate1[k]) * cap[k] + accepted(states, slot::QG[k], name, label)?;
            }
        }
    } else {
        for (k, value) in charge.iter_mut().enumerate() {
            *value = gate[k] * cap[k];
        }
    }

    let branches: [ChargeBranch; 3] =
        std::array::from_fn(|k| ChargeBranch::new(slot::QG[k], charge[k], cap[k]));
    for branch in branches {
        branch.store(states)?;
    }

    let mut result = GateCompanions::default();
    if let Some(coefficients) = integration.filter(|_| !(predict && states.initial_transient())) {
        let ag0 = coefficients.ag()[0];
        for (k, branch) in branches.into_iter().enumerate() {
            let companion = branch.integrate(states, coefficients)?;
            result.conductance[k] = companion.conductance;
            result.equivalent[k] = companion.derivative - ag0 * branch.charge;
        }
        // Do not cancel the two ag0*q terms: NIintegrate's ceq and the
        // subsequent voltage linearization have independent rounding.
        for (k, equivalent) in result.equivalent.iter_mut().enumerate() {
            *equivalent = *equivalent - result.conductance[k] * gate[k] + ag0 * charge[k];
        }
    } else {
        for branch in branches {
            branch.zero_derivative(states)?;
        }
    }
    Ok(result)
}
