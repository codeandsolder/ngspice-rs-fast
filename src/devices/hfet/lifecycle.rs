//! Shared load lifecycle for the HFET and MESA families. Device-specific
//! limiting, current equations and matrix/state stamping remain in each model.

use crate::devices::nonlinear::charge::ChargeBranch;
use crate::devices::{AnalysisMode, DeviceState, IterationPhase, StampContext};
use crate::maths::integrator::Companion;
use crate::primitives::{Real, SpiceError, SpiceResult};

/// Gate/drain voltage and current linearization used by the C convergence test.
#[derive(Clone, Copy)]
pub(super) struct Linearization {
    pub(super) voltage: [Real; 2],
    pub(super) gate_current: Real,
    pub(super) drain_current: Real,
    pub(super) transconductance: Real,
    pub(super) output_conductance: Real,
    pub(super) source_conductance: Real,
    pub(super) drain_conductance: Real,
}

impl Linearization {
    /// The common two-junction prediction (HFET1's thermionic cross terms
    /// require its own expression for the limited phase).
    #[expect(
        clippy::suboptimal_flops,
        reason = "preserve the C non-fused prediction order"
    )]
    pub(super) const fn predict(self, voltage: [Real; 2]) -> [Real; 2] {
        let source_delta = voltage[0] - self.voltage[0];
        let drain_delta = voltage[1] - self.voltage[1];
        let channel_delta = source_delta - drain_delta;
        [
            self.gate_current
                + self.drain_conductance * drain_delta
                + self.source_conductance * source_delta,
            self.drain_current
                + self.transconductance * source_delta
                + self.output_conductance * channel_delta
                - self.drain_conductance * drain_delta,
        ]
    }
}

/// The phase chosen before evaluating a device's physical equations.
#[derive(Debug, Clone, Copy)]
pub(super) enum Start {
    Conditions,
    Exact,
    Fixed,
    HeldOff {
        raw: [Real; 2],
    },
    Limited {
        junction_limited: bool,
        currents: [Real; 2],
    },
}

impl Start {
    /// Select the initial/solved or predicted junction voltages, then invoke
    /// family-specific limiting only with complete finite prior state.
    pub(super) fn select<const N: usize>(
        states: &DeviceState<'_>,
        raw: [Real; 2],
        ic: [Real; 2],
        off: bool,
        voltage_slots: [usize; 2],
        history_slots: [usize; N],
        limit: impl FnOnce([Real; 2], [Real; N]) -> ([Real; 2], Self),
    ) -> ([Real; 2], Self) {
        if states.initial_conditions() {
            let [vds, vgs] = ic;
            return ([vgs, vgs - vds], Self::Conditions);
        }
        if !states.device_limiting() {
            return (raw, Self::Exact);
        }
        let phase = states.phase();
        let voltage = match phase {
            IterationPhase::Junction => {
                let v = if off { 0. } else { -1. };
                return ([v; 2], Self::Fixed);
            }
            IterationPhase::Fix if off => return ([0.; 2], Self::HeldOff { raw }),
            IterationPhase::Predict => voltage_slots.map(|slot| states.predicted(slot, 1.)),
            IterationPhase::Fix | IterationPhase::Float => raw.map(Some),
        };
        let [Some(source), Some(drain)] = voltage else {
            return (raw, Self::Exact);
        };
        let voltage = [source, drain];
        let mut previous = [0.; N];
        for (value, slot) in previous.iter_mut().zip(history_slots) {
            let old = if phase == IterationPhase::Predict {
                states.accepted(1, slot)
            } else {
                states.iterate(slot)
            };
            let Some(old) = old.filter(|value| value.is_finite()) else {
                return (voltage, Self::Exact);
            };
            *value = old;
        }
        limit(voltage, previous)
    }

    pub(super) fn report(
        self,
        states: &mut DeviceState<'_>,
        point: Linearization,
    ) -> SpiceResult<()> {
        let nonconvergent = match self {
            Self::Fixed => true,
            Self::Conditions | Self::Exact => false,
            Self::HeldOff { raw } => states
                .convergence_tolerances()
                .is_some_and(|tolerances| exceeds(point.predict(raw), point, tolerances, false)),
            Self::Limited {
                junction_limited,
                currents,
            } => {
                junction_limited
                    || states
                        .convergence_tolerances()
                        .is_some_and(|tolerances| exceeds(currents, point, tolerances, true))
            }
        };
        if nonconvergent {
            states.report_nonconvergence()?;
        }
        Ok(())
    }
}

/// The limited gate-current comparison is inclusive, unlike the held-off
/// test and the drain-current comparison. Never reorder or fuse C arithmetic.
#[expect(
    clippy::suboptimal_flops,
    reason = "retain the non-fused C tolerance calculation"
)]
const fn exceeds(
    predicted: [Real; 2],
    point: Linearization,
    tolerances: (Real, Real),
    inclusive_gate: bool,
) -> bool {
    let [gate, drain] = predicted;
    let (relative, absolute) = tolerances;
    let gate_error = (gate - point.gate_current).abs();
    let gate_limit = relative * gate.abs().max(point.gate_current.abs()) + absolute;
    let gate_failed = if inclusive_gate {
        gate_error >= gate_limit
    } else {
        gate_error > gate_limit
    };
    gate_failed
        || (drain - point.drain_current).abs()
            > relative * drain.abs().max(point.drain_current.abs()) + absolute
}

/// Latest accepted finite value, or the same fallback used in the C port.
pub(super) fn accepted(states: &DeviceState<'_>, slot: usize, fallback: Real) -> Real {
    states
        .accepted(1, slot)
        .filter(|value| value.is_finite())
        .unwrap_or(fallback)
}

/// Prime-prime voltages are carried without extrapolation during prediction.
pub(super) fn prime_prime(
    states: &DeviceState<'_>,
    aliased: bool,
    limited: Real,
    slot: usize,
    solved: impl FnOnce() -> Real,
) -> Real {
    if aliased {
        return limited;
    }
    if states.phase() == IterationPhase::Predict {
        return states
            .accepted(1, slot)
            .filter(|value| value.is_finite())
            .unwrap_or_else(solved);
    }
    solved()
}

/// Charge and optional integration results in the caller's reference order.
pub(super) struct ChargeStep<const N: usize> {
    pub(super) charges: [Real; N],
    pub(super) companions: Option<[Companion; N]>,
}

/// Incremental charge storage shared by the three families. HFET1 supplies
/// an additional drain-source branch. Seed all accepted charges before
/// accumulation, integrate in the supplied order, then seed all derivatives.
#[expect(
    clippy::suboptimal_flops,
    reason = "preserve non-fused incremental charge arithmetic"
)]
#[expect(clippy::float_cmp, reason = "C companion timestep identity is exact")]
pub(super) fn charge_step<const N: usize>(
    context: &mut StampContext<'_>,
    family: &str,
    slots: [usize; N],
    voltage: [Real; N],
    previous: [Real; N],
    capacitance: [Real; N],
) -> SpiceResult<ChargeStep<N>> {
    let seed =
        context.states.initial_transient() && context.states.phase() == IterationPhase::Predict;
    if seed {
        for ((slot, voltage), cap) in slots.into_iter().zip(voltage).zip(capacitance) {
            ChargeBranch::new(slot, cap * voltage, cap).seed(&mut context.states)?;
        }
    }
    let branches: [ChargeBranch; N] = std::array::from_fn(|i| {
        let charge =
            accepted(&context.states, slots[i], 0.) + capacitance[i] * (voltage[i] - previous[i]);
        ChargeBranch::new(slots[i], charge, capacitance[i])
    });
    if seed {
        for branch in branches {
            branch.seed(&mut context.states)?;
        }
    }
    let mut result = ChargeStep {
        charges: branches.map(|branch| branch.charge),
        companions: None,
    };
    if let Some(coefficients) = context.integration {
        let AnalysisMode::Transient { dt, .. } = context.mode else {
            return Err(SpiceError::circuit(format!(
                "{family} charge companion outside transient"
            )));
        };
        if dt != coefficients.dt() {
            return Err(SpiceError::circuit(format!(
                "{family} charge timestep mismatch"
            )));
        }
        let mut companions = [Companion {
            derivative: 0.,
            conductance: 0.,
            current: 0.,
        }; N];
        for (companion, branch) in companions.iter_mut().zip(branches) {
            *companion = branch.integrate(&mut context.states, coefficients)?;
        }
        if seed {
            for (branch, companion) in branches.into_iter().zip(companions) {
                branch.seed_derivative(&mut context.states, companion.derivative)?;
            }
        }
        result.companions = Some(companions);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::StateHistory;

    const SLOTS: [usize; 2] = [0, 1];

    #[test]
    fn initialization_precedes_limiting_and_missing_history_is_exact() {
        let history = StateHistory::new(2);
        let raw = [4., 3.];
        let ic = [2., 1.];
        for (phase, off, conditions, limiting, expected) in [
            (IterationPhase::Junction, false, true, true, [1., -1.]),
            (IterationPhase::Junction, false, false, false, raw),
            (IterationPhase::Junction, false, false, true, [-1.; 2]),
            (IterationPhase::Junction, true, false, true, [0.; 2]),
            (IterationPhase::Fix, true, false, true, [0.; 2]),
            (IterationPhase::Float, false, false, true, raw),
            (IterationPhase::Predict, false, false, true, raw),
        ] {
            let mut trial = history
                .trial_in(phase, None)
                .unwrap()
                .with_initial_conditions(conditions)
                .with_device_limiting(limiting);
            let states = history.device(&mut trial, 0..2).unwrap();
            let (voltage, start) =
                Start::select(&states, raw, ic, off, SLOTS, SLOTS, |_, _| unreachable!());
            assert_eq!(voltage, expected);
            match (conditions, limiting, phase, off) {
                (true, _, _, _) => assert!(matches!(start, Start::Conditions)),
                (false, false, _, _) => assert!(matches!(start, Start::Exact)),
                (false, true, IterationPhase::Junction, _) => {
                    assert!(matches!(start, Start::Fixed))
                }
                (false, true, IterationPhase::Fix, true) => {
                    assert!(matches!(start, Start::HeldOff { .. }))
                }
                _ => assert!(matches!(start, Start::Exact)),
            }
        }
    }

    #[test]
    fn predicted_voltage_extrapolates_but_prime_prime_carries_latest() {
        let mut history = StateHistory::new(2);
        for values in [[1., 2.], [3., 4.]] {
            let mut trial = history.trial();
            {
                let mut states = history.device(&mut trial, 0..2).unwrap();
                states.set(0, values[0]).unwrap();
                states.set(1, values[1]).unwrap();
            }
            history.commit(trial).unwrap();
        }
        let mut trial = history.trial_in(IterationPhase::Predict, None).unwrap();
        let states = history.device(&mut trial, 0..2).unwrap();
        let (voltage, start) = Start::select(
            &states,
            [7., 8.],
            [0.; 2],
            false,
            SLOTS,
            SLOTS,
            |predicted, previous| {
                assert_eq!(predicted, [5., 6.]);
                assert_eq!(previous, [3., 4.]);
                (
                    predicted,
                    Start::Limited {
                        junction_limited: false,
                        currents: [0.; 2],
                    },
                )
            },
        );
        assert_eq!(voltage, [5., 6.]);
        assert!(matches!(start, Start::Limited { .. }));
        assert_eq!(
            prime_prime(&states, false, 9., 0, || 7.).to_bits(),
            3_f64.to_bits()
        );
        assert_eq!(
            prime_prime(&states, true, 9., 0, || 7.).to_bits(),
            9_f64.to_bits()
        );
    }

    #[test]
    fn convergence_respects_gate_boundary_and_junction_limiter() {
        let point = Linearization {
            voltage: [0.; 2],
            gate_current: 0.,
            drain_current: 0.,
            transconductance: 0.,
            output_conductance: 0.,
            source_conductance: 1.,
            drain_conductance: 0.,
        };
        let history = StateHistory::new(0);
        for (start, expected) in [
            (Start::HeldOff { raw: [1., 0.] }, false),
            (
                Start::Limited {
                    junction_limited: false,
                    currents: [1., 0.],
                },
                true,
            ),
            (
                Start::Limited {
                    junction_limited: false,
                    currents: [0., 1.],
                },
                false,
            ),
            (
                Start::Limited {
                    junction_limited: true,
                    currents: [0.; 2],
                },
                true,
            ),
            (Start::Fixed, true),
            (Start::Exact, false),
            (Start::Conditions, false),
        ] {
            // Each case must begin with an independent Newton load.
            let mut trial = history
                .trial_in(IterationPhase::Float, None)
                .unwrap()
                .with_convergence_tolerances(0., 1.);
            start
                .report(&mut history.device(&mut trial, 0..0).unwrap(), point)
                .unwrap();
            assert_eq!(trial.is_nonconvergent(), expected);
        }
    }
}
