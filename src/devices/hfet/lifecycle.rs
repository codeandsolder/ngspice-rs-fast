//! Shared load lifecycle for the HFET and MESA families. Device-specific
//! limiting, current equations and matrix/state stamping remain in each model.

use crate::devices::limiting;
use crate::devices::nonlinear::charge::ChargeBranch;
use crate::devices::{AnalysisMode, DeviceState, IterationPhase};
use crate::maths::integrator::{Coefficients, Companion};
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
        reason = "preserve C linearization and convergence arithmetic order"
    )]
    pub(super) const fn predict(self, voltage: [Real; 2]) -> [Real; 2] {
        let source_delta = voltage[0] - self.voltage[0];
        let drain_delta = voltage[1] - self.voltage[1];
        let channel_delta = source_delta - drain_delta;
        [
            self.source_conductance.mul_add(
                source_delta,
                self.drain_conductance
                    .mul_add(drain_delta, self.gate_current),
            ),
            (-self.drain_conductance).mul_add(
                drain_delta,
                self.output_conductance.mul_add(
                    channel_delta,
                    self.transconductance
                        .mul_add(source_delta, self.drain_current),
                ),
            ),
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
        limit: impl FnOnce([Real; 2], [Real; 2], [Real; N]) -> ([Real; 2], Self),
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
        let read_history = |slot| {
            let old = if phase == IterationPhase::Predict {
                states.accepted(1, slot)
            } else {
                states.iterate(slot)
            };
            old.filter(|value| value.is_finite())
        };
        let [Some(previous_source), Some(previous_drain)] = voltage_slots.map(read_history) else {
            return (voltage, Self::Exact);
        };
        let mut previous = [0.; N];
        for (value, slot) in previous.iter_mut().zip(history_slots) {
            let Some(old) = read_history(slot) else {
                return (voltage, Self::Exact);
            };
            *value = old;
        }
        limit(voltage, [previous_source, previous_drain], previous)
    }

    /// HFET2 and MESA use the same current linearization and two pnjlim calls
    /// followed by fetlim. The model supplies each junction's thermal and
    /// critical voltages and the threshold.
    pub(super) fn limit_junctions(
        voltage: [Real; 2],
        previous: [Real; 2],
        history: [Real; 6],
        thermal: [Real; 2],
        critical: [Real; 2],
        threshold: Real,
    ) -> ([Real; 2], Self) {
        let [
            gate_current,
            drain_current,
            transconductance,
            output_conductance,
            source_conductance,
            drain_conductance,
        ] = history;
        let currents = Linearization {
            voltage: previous,
            gate_current,
            drain_current,
            transconductance,
            output_conductance,
            source_conductance,
            drain_conductance,
        }
        .predict(voltage);
        let source = limiting::pnjlim(voltage[0], previous[0], thermal[0], critical[0]);
        let drain = limiting::pnjlim(voltage[1], previous[1], thermal[1], critical[1]);
        let limited = [
            limiting::fetlim(source.value, previous[0], threshold),
            limiting::fetlim(drain.value, previous[1], threshold),
        ];
        (
            limited,
            Self::Limited {
                junction_limited: source.limited || drain.limited,
                currents,
            },
        )
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
    reason = "preserve C convergence comparison rounding"
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
    let gate_limit = relative.mul_add(gate.abs().max(point.gate_current.abs()), absolute);
    let gate_failed = if inclusive_gate {
        gate_error >= gate_limit
    } else {
        gate_error > gate_limit
    };
    gate_failed
        || (drain - point.drain_current).abs()
            > relative.mul_add(drain.abs().max(point.drain_current.abs()), absolute)
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

/// An incremental junction or drain-source charge and its state slot.
/// Previous voltage is resolved before any initial-transient history writes.
#[derive(Clone, Copy)]
pub(super) struct ChargeInput {
    slot: usize,
    voltage: Real,
    previous: Real,
    capacitance: Real,
}

impl ChargeInput {
    pub(super) const fn new(slot: usize, voltage: Real, previous: Real, capacitance: Real) -> Self {
        Self {
            slot,
            voltage,
            previous,
            capacitance,
        }
    }
}

/// Charge and optional integration results in the caller's reference order.
pub(super) struct ChargeStep<const N: usize> {
    pub(super) charges: [Real; N],
    pub(super) companions: Option<[Companion; N]>,
}

/// Incremental charge storage shared by the three families. HFET1 supplies
/// an additional drain-source branch. Seed all accepted charges before
/// accumulation, integrate in the supplied order, then seed all derivatives.
#[expect(clippy::float_cmp, reason = "C companion timestep identity is exact")]
pub(super) fn charge_step<const N: usize>(
    states: &mut DeviceState<'_>,
    mode: AnalysisMode,
    integration: Option<&Coefficients>,
    family: &str,
    inputs: [ChargeInput; N],
) -> SpiceResult<ChargeStep<N>> {
    let seed = states.initial_transient() && states.phase() == IterationPhase::Predict;
    if seed {
        for input in inputs {
            ChargeBranch::new(
                input.slot,
                input.capacitance * input.voltage,
                input.capacitance,
            )
            .seed(states)?;
        }
    }
    let branches = inputs.map(|input| {
        let charge = input.capacitance.mul_add(
            input.voltage - input.previous,
            accepted(states, input.slot, 0.),
        );
        ChargeBranch::new(input.slot, charge, input.capacitance)
    });
    if seed {
        for branch in branches {
            branch.seed(states)?;
        }
    }
    let mut result = ChargeStep {
        charges: branches.map(|branch| branch.charge),
        companions: None,
    };
    if let Some(coefficients) = integration {
        let AnalysisMode::Transient { dt, .. } = mode else {
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
            *companion = branch.integrate(states, coefficients)?;
        }
        if seed {
            for (branch, companion) in branches.into_iter().zip(companions) {
                branch.seed_derivative(states, companion.derivative)?;
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
                Start::select(&states, raw, ic, off, SLOTS, [], |_, _, _| unreachable!());
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
            [],
            |predicted, previous, []| {
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

    #[test]
    fn incremental_charges_seed_and_integrate_without_accepting_trial() {
        use crate::maths::integrator::{IntegrationMethod, StepHistory};

        let mut history = StateHistory::new(6);
        let mut initial = history.trial();
        {
            let mut states = history.device(&mut initial, 0..6).unwrap();
            for (slot, value) in [2., 1., 4., 2., 8., 3.].into_iter().enumerate() {
                states.set(slot, value).unwrap();
            }
        }
        history.commit(initial).unwrap();
        let inputs = [
            ChargeInput::new(0, 3., 1., 2.),
            ChargeInput::new(2, 4., 2., 3.),
            ChargeInput::new(4, 5., 3., 4.),
        ];
        {
            let mut trial = history.trial();
            let mut states = history.device(&mut trial, 0..6).unwrap();
            let step =
                charge_step(&mut states, AnalysisMode::DcSweep, None, "HFET1", inputs).unwrap();
            assert_eq!(step.charges, [6., 10., 16.]);
            assert!(step.companions.is_none());
            assert_eq!(states.accepted(1, 0), Some(2.));
        }
        let coefficients = StepHistory::new()
            .trial(IntegrationMethod::Trapezoidal, 1, 0.25, 0.5)
            .unwrap();
        {
            let mut trial = history
                .trial_in(IterationPhase::Predict, None)
                .unwrap()
                .with_initial_transient(true);
            let mut states = history.device(&mut trial, 0..6).unwrap();
            let step = charge_step(
                &mut states,
                AnalysisMode::Transient {
                    time: 0.25,
                    dt: 0.25,
                },
                Some(&coefficients),
                "HFET1",
                inputs,
            )
            .unwrap();
            assert_eq!(step.charges, [10., 18., 28.]);
            let companions = step.companions.unwrap();
            assert_eq!(companions.map(|branch| branch.conductance), [8., 12., 16.]);
            assert_eq!(companions.map(|branch| branch.derivative), [0.; 3]);
            for (slot, charge) in [0, 2, 4].into_iter().zip(step.charges) {
                assert_eq!(states.accepted(1, slot), Some(charge));
                assert_eq!(states.accepted(1, slot + 1), Some(0.));
            }
        }
        assert_eq!(history.accepted(1), Some(&[2., 1., 4., 2., 8., 3.][..]));
    }
}
