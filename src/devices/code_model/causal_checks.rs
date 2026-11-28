//! Shared timing constraints for arrival-aware sequential primitives.
use super::causal_history::{COMPARISON_BUDGET, InputEdge, PortHistory, sample_joint};
use super::sequential_timing::{TimingContract, ViolationPolicy};
use super::source_logic::AsyncPolicy;
use super::{Logic, ModelDiagnostic};
use crate::analysis::digital::timing::{ApertureResult, Bounds, Certainty, ConstraintResult};
use crate::primitives::SpiceResult;

pub(super) fn capture_enabled(
    timing: &TimingContract,
    ports: &[PortHistory],
    edge: &InputEdge,
) -> SpiceResult<bool> {
    let Some(policy) = timing.asynchronous else {
        return Ok(true);
    };
    controls_may_be_inactive(policy, &ports[timing.clock + 1..], edge)
}

/// An asynchronous mask permits synchronous capture only if both
/// controls can be inactive in the *same* timing realization.
pub(super) fn controls_may_be_inactive(
    policy: AsyncPolicy,
    ports: &[PortHistory],
    edge: &InputEdge,
) -> SpiceResult<bool> {
    let controls = sample_joint(ports, edge)?;
    let enabled = controls
        .branches()
        .any(|(_, values)| values[0].logic != policy.preset && values[1].logic != policy.clear);
    Ok(enabled)
}

pub(super) fn aperture(
    timing: &TimingContract,
    capture: &InputEdge,
    data: &InputEdge,
    index: usize,
    diagnostics: &mut Vec<ModelDiagnostic>,
) -> SpiceResult<bool> {
    if !data.logic_changed() || (timing.setup[index] == 0. && timing.hold[index] == 0.) {
        return Ok(false);
    }
    let result = ApertureResult::check_ordered(
        &capture.window,
        &data.window,
        timing.setup[index],
        timing.hold[index],
        data.precedes_at_equality(capture),
        COMPARISON_BUDGET,
    )?;
    let bounds = result.capture_minus_data;
    let mut violated = false;
    for (kind, certainty, minimum, elapsed) in [
        ("setup", result.setup, timing.setup[index], bounds),
        (
            "hold",
            result.hold,
            timing.hold[index],
            Bounds::new(-bounds.maximum(), -bounds.minimum())?,
        ),
    ] {
        violated |= report(
            timing.policy,
            kind,
            index,
            ConstraintResult {
                separation: elapsed,
                minimum,
                certainty,
            },
            diagnostics,
        );
    }
    Ok(violated)
}

pub(super) fn check_widths(
    timing: &TimingContract,
    ports: &[PortHistory],
    diagnostics: &mut Vec<ModelDiagnostic>,
    changes: &[Option<InputEdge>],
) -> SpiceResult<bool> {
    let clock = timing.clock;
    let mut poisoned = false;
    for (index, edge) in changes.iter().enumerate().skip(clock) {
        let Some(edge) = edge.as_ref().filter(|edge| edge.logic_changed()) else {
            continue;
        };
        let (kind, minimum) = if index == clock {
            (
                "clock/gate pulse width",
                match edge.from.logic {
                    Logic::Zero => timing.clock_min[0],
                    Logic::One => timing.clock_min[1],
                    _ => 0.,
                },
            )
        } else if let Some(policy) = timing.asynchronous {
            let active = if index == clock + 1 {
                policy.preset
            } else {
                policy.clear
            };
            (
                "asynchronous pulse width",
                if edge.from.logic == active && edge.to.logic != active {
                    timing.async_min[index - clock - 1]
                } else {
                    0.
                },
            )
        } else {
            continue;
        };
        if minimum == 0. {
            continue;
        }
        if let Some(previous) = ports[index]
            .edges()
            .iter()
            .rev()
            .filter(|edge| edge.logic_changed())
            .nth(1)
        {
            let result = ConstraintResult::minimum(
                &edge.window,
                &previous.window,
                minimum,
                COMPARISON_BUDGET,
            )?;
            let unknown = report(timing.policy, kind, index, result, diagnostics);
            poisoned |= unknown && (index != clock || capture_enabled(timing, ports, edge)?);
        }
    }
    Ok(poisoned)
}

pub(super) fn check_capture(
    timing: &TimingContract,
    ports: &[PortHistory],
    diagnostics: &mut Vec<ModelDiagnostic>,
    capture: &InputEdge,
) -> SpiceResult<bool> {
    if !capture_enabled(timing, ports, capture)? {
        return Ok(false);
    }
    let mut poisoned = false;
    for (index, port) in ports.iter().enumerate().take(timing.clock) {
        for edge in port.edges() {
            poisoned |= aperture(timing, capture, edge, index, diagnostics)?;
        }
    }
    if let Some(policy) = timing.asynchronous {
        for (offset, active) in [policy.preset, policy.clear].into_iter().enumerate() {
            let index = timing.clock + 1 + offset;
            let minimum = timing.recovery[offset];
            if minimum == 0. {
                continue;
            }
            for edge in ports[index].edges() {
                if edge.from.logic != active || edge.to.logic == active {
                    continue;
                }
                // Recovery is the setup aperture of an asynchronous release.
                let result = ApertureResult::check_ordered(
                    &capture.window,
                    &edge.window,
                    minimum,
                    0.,
                    edge.precedes_at_equality(capture),
                    COMPARISON_BUDGET,
                )?;
                let bounds = result.capture_minus_data;
                let certainty = result.setup;
                poisoned |= report(
                    timing.policy,
                    "asynchronous recovery",
                    index,
                    ConstraintResult {
                        separation: bounds,
                        minimum,
                        certainty,
                    },
                    diagnostics,
                );
            }
        }
    }
    Ok(poisoned)
}

fn report(
    policy: ViolationPolicy,
    kind: &'static str,
    input: usize,
    result: ConstraintResult,
    diagnostics: &mut Vec<ModelDiagnostic>,
) -> bool {
    if result.certainty == Certainty::Satisfied {
        return false;
    }
    diagnostics.push(ModelDiagnostic {
        fatal: matches!(policy, ViolationPolicy::Error),
        kind,
        input,
        elapsed: result.separation.minimum(),
        maximum_elapsed: result.separation.maximum(),
        minimum: result.minimum,
        certainty: result.certainty,
    });
    matches!(policy, ViolationPolicy::Unknown)
}
