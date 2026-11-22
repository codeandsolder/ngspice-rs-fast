//! Shared classic-FET load phases, gate charge lifecycle and fixed stamping.
//! Physical current equations, thermal memory and Statz laws stay in their
//! respective device modules.

#![expect(
    clippy::similar_names,
    reason = "C equation symbols preserve the reference operations"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "Preserve the non-fused C operation order"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "C stamp and equation symbols"
)]

use super::slot;
use crate::devices::limiting;
use crate::devices::nonlinear::charge::ChargeBranch;
use crate::devices::{AnalysisMode, DeviceState, IterationPhase, StampContext};
use crate::maths::integrator::Coefficients;
use crate::primitives::{NodeId, Real, SpiceError, SpiceResult};

/// A validated load mode. Companion integration is present exactly for
/// transient loads and its timestep equals the trial timestep.
#[derive(Clone, Copy)]
pub(in crate::devices) enum LoadPhase<'a> {
    Static {
        update_statz: bool,
    },
    Transient {
        coefficients: &'a Coefficients,
        seed: bool,
    },
}

impl<'a> LoadPhase<'a> {
    #[expect(clippy::float_cmp, reason = "C companion timestep identity is exact")]
    pub(in crate::devices) fn new(
        mode: AnalysisMode,
        integration: Option<&'a Coefficients>,
        states: &DeviceState<'_>,
        family: &str,
    ) -> SpiceResult<Self> {
        match (mode, integration) {
            (AnalysisMode::Ac { .. }, _) => Err(SpiceError::circuit(format!(
                "{family} AC needs small-signal assembly"
            ))),
            (AnalysisMode::Transient { dt, .. }, Some(coefficients)) if dt == coefficients.dt() => {
                Ok(Self::Transient {
                    coefficients,
                    seed: states.phase() == IterationPhase::Predict && states.initial_transient(),
                })
            }
            (AnalysisMode::Transient { .. }, Some(_)) => Err(SpiceError::circuit(format!(
                "{family} charge timestep mismatch"
            ))),
            (AnalysisMode::Transient { .. }, None) => Err(SpiceError::circuit(format!(
                "{family} transient needs companion integration"
            ))),
            (_, Some(_)) => Err(SpiceError::circuit(format!(
                "{family} charge companion outside transient"
            ))),
            (mode, None) => Ok(Self::Static {
                update_statz: mode == AnalysisMode::DcSweep || states.initial_conditions(),
            }),
        }
    }

    pub(in crate::devices) const fn step(self) -> Option<Real> {
        match self {
            Self::Static { .. } => None,
            Self::Transient { coefficients, .. } => Some(coefficients.dt()),
        }
    }

    pub(in crate::devices) const fn updates_statz(self) -> bool {
        match self {
            Self::Static { update_statz } => update_statz,
            Self::Transient { .. } => true,
        }
    }

    /// Select the normalized gate voltages using the validated timestep.
    pub(in crate::devices) fn junctions(
        self,
        j: Junctions,
        context: &StampContext<'_>,
        terminals: &[NodeId],
        inner: [NodeId; 3],
        initial: &crate::devices::initial::InstanceInitial,
    ) -> ([Real; 2], Start) {
        let raw = normalized(j.pol, inner.map(|n| context.node_voltage(n)));
        let ic = start_conditions(terminals, initial, |n| context.node_voltage(n));
        let ratio = match self {
            Self::Static { .. } => 1.,
            Self::Transient { coefficients, .. } => coefficients.predictor_ratio(),
        };
        select_junctions(j, &context.states, raw, ic, ratio)
    }
}

/// How the classic FET loads obtained the voltages of one load.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::devices) enum Start {
    /// The `uic` initial load (`MODETRANOP | MODEUIC | MODEINITJCT`).
    Conditions,
    /// A load outside a device-limited Newton solve: the supplied solution.
    Exact,
    /// `MODEINITJCT` or `MODEINITTRAN`: fixed voltages with C's `icheck`
    /// still set, so always nonconvergent.
    Fixed,
    /// An `off` instance held at zero in `MODEINITFIX`, with the iterate's
    /// normalized `[vgs, vgd]`. C marks every such load nonconvergent and
    /// leaves `MODEINITFIX` only through its continuation fallbacks, ending
    /// in `OPtran` (`optran.c`), which is not ported. The port instead
    /// judges the held load as the devices with a `DEVconvTest` do: it
    /// converges once the currents at the held voltages agree with their
    /// linear prediction at the iterate.
    HeldOff { raw: [Real; 2] },
    /// `MODEINITPRED`, `MODEINITFLOAT` or `MODEINITFIX`: limited against the
    /// previous state, with C's `icheck` and current predictions.
    Limited {
        icheck: bool,
        cghat: Real,
        cdhat: Real,
    },
}

/// What [`select_junctions`] needs of a device: polarity, `off`, `kT/q`,
/// `vcrit` and the `DEVfetlim` threshold (`JFETtThreshold`, or the model
/// `VTO` for level 2).
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct Junctions {
    pub(in crate::devices) pol: Real,
    pub(in crate::devices) off: bool,
    pub(in crate::devices) vt: Real,
    pub(in crate::devices) vcrit: Real,
    pub(in crate::devices) vto: Real,
}

/// `jfetload.c`/`jfet2load.c`'s normalized `[vgs, vgd]` for this load and
/// how they were obtained.
///
/// `raw` is the iterate's (C `CKTrhsOld`) and `ic` the `uic` initial
/// conditions `[IC-VDS, IC-VGS]`. `MODEINITJCT` starts at `-1` (zero for an
/// `off` instance, which is also held at zero in `MODEINITFIX`),
/// `MODEINITTRAN` at the accepted (`CKTstate1`) voltages. `MODEINITPRED`
/// extrapolates the accepted voltages with `DEVpred`'s
/// `(1 + xfact) state1 - xfact state2` and compares with `CKTstate1` (C
/// copies it into `CKTstate0`); other loads compare the iterate with the
/// previous load. The comparison yields C's `cghat` and `cdhat`, then
/// `DEVpnjlim` and `DEVfetlim` limit both voltages.
fn select_junctions(
    j: Junctions,
    states: &DeviceState<'_>,
    raw: [Real; 2],
    ic: [Real; 2],
    xfact: Real,
) -> ([Real; 2], Start) {
    use limiting::Linearization;
    match Linearization::classify(states) {
        Linearization::InitialConditions => {
            let vds = j.pol * ic[0];
            let vgs = j.pol * ic[1];
            return ([vgs, vgs - vds], Start::Conditions);
        }
        Linearization::Exact => return (raw, Start::Exact),
        Linearization::Initial => {
            let v = if j.off { 0. } else { -1. };
            return ([v, v], Start::Fixed);
        }
        Linearization::InitialTransient => {
            let state1 = |slot: usize, fallback: Real| states.accepted(1, slot).unwrap_or(fallback);
            return (
                [state1(slot::VGS, raw[0]), state1(slot::VGD, raw[1])],
                Start::Fixed,
            );
        }
        Linearization::Limited => {}
    }
    let [vgs, vgd] = match states.phase() {
        IterationPhase::Fix if j.off => return ([0.; 2], Start::HeldOff { raw }),
        IterationPhase::Predict => {
            let Some(predicted) = limiting::predicted_values(states, [slot::VGS, slot::VGD], xfact)
            else {
                return (raw, Start::Exact);
            };
            predicted
        }
        _ => raw,
    };
    let Some([vgs0, vgd0, cg0, cd0, gm0, gds0, ggs0, ggd0]) = limiting::previous_values(
        states,
        [
            slot::VGS,
            slot::VGD,
            slot::CG,
            slot::CD,
            slot::GM,
            slot::GDS,
            slot::GGS,
            slot::GGD,
        ],
    ) else {
        return ([vgs, vgd], Start::Exact);
    };
    let [cghat, cdhat] = predict_currents(
        [vgs, vgd],
        [vgs0, vgd0],
        [cg0, cd0],
        [gm0, gds0, ggs0, ggd0],
    );
    let gs = limiting::pnjlim(vgs, vgs0, j.vt, j.vcrit);
    let gd = limiting::pnjlim(vgd, vgd0, j.vt, j.vcrit);
    let vgs = limiting::fetlim(gs.value, vgs0, j.vto);
    let vgd = limiting::fetlim(gd.value, vgd0, j.vto);
    (
        [vgs, vgd],
        Start::Limited {
            icheck: gs.limited || gd.limited,
            cghat,
            cdhat,
        },
    )
}

/// Reference current linearization, also used for the held-off convergence
/// rule. Preserve the left-associated C arithmetic.
fn predict_currents(
    [vgs, vgd]: [Real; 2],
    [vgs0, vgd0]: [Real; 2],
    [cg0, cd0]: [Real; 2],
    [gm0, gds0, ggs0, ggd0]: [Real; 4],
) -> [Real; 2] {
    let delvgs = vgs - vgs0;
    let delvgd = vgd - vgd0;
    let delvds = delvgs - delvgd;
    [
        cg0 + ggd0 * delvgd + ggs0 * delvgs,
        cd0 + gm0 * delvgs + gds0 * delvds - ggd0 * delvgd,
    ]
}

/// The values `jfetload.c`/`jfet2load.c` store and stamp, per device,
/// after the charge companions: C's `vgs`, `vgd`, `vds`, `cg`, `cgd`, `cd`,
/// `gm`, `gds`, `ggs`, `ggd`.
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) struct Loaded {
    pub(in crate::devices) vgs: Real,
    pub(in crate::devices) vgd: Real,
    pub(in crate::devices) vds: Real,
    pub(in crate::devices) cg: Real,
    pub(in crate::devices) cgd: Real,
    pub(in crate::devices) cd: Real,
    pub(in crate::devices) gm: Real,
    pub(in crate::devices) gds: Real,
    pub(in crate::devices) ggs: Real,
    pub(in crate::devices) ggd: Real,
}

impl Loaded {
    /// Construct from the channel and both gate diodes without changing the
    /// two additions/subtractions of the C load.
    pub(in crate::devices) fn from_currents(
        [vgs, vgd]: [Real; 2],
        (cdrain, gm, gds): (Real, Real, Real),
        (cgs, ggs): (Real, Real),
        (cgd, ggd): (Real, Real),
    ) -> Self {
        Self {
            vgs,
            vgd,
            vds: vgs - vgd,
            cg: cgs + cgd,
            cgd,
            cd: cdrain - cgd,
            gm,
            gds,
            ggs,
            ggd,
        }
    }

    /// Fold the gate charge companions into the per-unit C currents and
    /// conductances in source-then-drain order.
    fn integrate_gates(
        &mut self,
        states: &mut DeviceState<'_>,
        coefficients: &Coefficients,
        [source, drain]: [ChargeBranch; 2],
    ) -> SpiceResult<[Real; 2]> {
        let gs = source.integrate(states, coefficients)?;
        self.ggs += gs.conductance;
        self.cg += gs.derivative;
        let gd = drain.integrate(states, coefficients)?;
        self.ggd += gd.conductance;
        self.cg += gd.derivative;
        self.cd -= gd.derivative;
        self.cgd += gd.derivative;
        Ok([gs.derivative, gd.derivative])
    }
}

/// Different DC/initial-history policies of the three reference models.
/// The optional Parker drain charge contributes current but no conductance.
#[derive(Clone, Copy)]
pub(in crate::devices) enum GateUpdate {
    Depletion([ChargeBranch; 2]),
    Parker {
        gates: [ChargeBranch; 2],
        drain: ChargeBranch,
    },
    Statz([ChargeBranch; 2]),
}

impl GateUpdate {
    pub(in crate::devices) fn apply(
        self,
        phase: LoadPhase<'_>,
        states: &mut DeviceState<'_>,
        loaded: &mut Loaded,
    ) -> SpiceResult<()> {
        let (gates, drain) = match self {
            Self::Depletion(gates) | Self::Statz(gates) => (gates, None),
            Self::Parker { gates, drain } => (gates, Some(drain)),
        };
        let branches = gates.into_iter().chain(drain);
        match phase {
            LoadPhase::Transient { coefficients, seed } => {
                // MESFET increments already start from its DC charge offset.
                // Preserve the port's distinct initial-transient policy.
                let seed = seed && !matches!(self, Self::Statz(_));
                if seed {
                    for branch in branches.clone() {
                        branch.seed(states)?;
                    }
                }
                let derivatives = loaded.integrate_gates(states, coefficients, gates)?;
                let drain_derivative = if let Some(drain) = drain {
                    let derivative = drain.integrate(states, coefficients)?.derivative;
                    loaded.cd += derivative;
                    Some(derivative)
                } else {
                    None
                };
                if seed {
                    for (branch, derivative) in branches
                        .clone()
                        .zip(derivatives.into_iter().chain(drain_derivative))
                    {
                        branch.seed_derivative(states, derivative)?;
                    }
                }
            }
            LoadPhase::Static { .. } => match self {
                Self::Depletion(_) => {
                    for branch in branches.clone() {
                        branch.zero_derivative(states)?;
                    }
                }
                Self::Parker { .. } => {
                    // The Parker model retains derivatives in DC; its
                    // priority is iterate, accepted, carried, then zero.
                    for branch in branches.clone() {
                        carry(states, branch.derivative_slot())?;
                    }
                }
                Self::Statz(_) => {
                    for branch in branches {
                        branch.store(states)?;
                        branch.zero_derivative(states)?;
                    }
                    return Ok(());
                }
            },
        }
        for branch in branches {
            branch.store(states)?;
        }
        Ok(())
    }
}

/// Retain the most recent finite state using the Parker model's C ordering.
/// A nonfinite iterate is treated as zero, not replaced by an older value.
pub(in crate::devices) fn carry(states: &mut DeviceState<'_>, slot: usize) -> SpiceResult<()> {
    let value = states
        .iterate(slot)
        .or_else(|| states.accepted(1, slot))
        .or_else(|| states.carried(slot))
        .filter(|v| v.is_finite())
        .unwrap_or(0.);
    states.set(slot, value)
}

/// The two Statz implementations integrate the same charge increments but
/// JFET2 averages the partials while MESFET keeps the first evaluation.
#[derive(Debug, Clone, Copy)]
pub(in crate::devices) enum StatzCapacitance {
    Averaged,
    Present,
}

pub(in crate::devices) fn incremental_statz(
    states: &DeviceState<'_>,
    [vgs, vgd]: [Real; 2],
    rule: StatzCapacitance,
    missing: impl Fn(&str) -> SpiceError,
    mut evaluate: impl FnMut(Real, Real) -> SpiceResult<(Real, Real, Real)>,
) -> SpiceResult<[ChargeBranch; 2]> {
    let accepted = |slot, name| states.accepted(1, slot).ok_or_else(|| missing(name));
    let vgs1 = accepted(slot::VGS, "vgs")?;
    let vgd1 = accepted(slot::VGD, "vgd")?;
    let qgs1 = accepted(slot::QGS, "qgs")?;
    let qgd1 = accepted(slot::QGD, "qgd")?;
    let (qa, cgsna, cgdna) = evaluate(vgs, vgd)?;
    let (qb, _, cgdnb) = evaluate(vgs1, vgd)?;
    let (qc, cgsnc, _) = evaluate(vgs, vgd1)?;
    let (qd, _, _) = evaluate(vgs1, vgd1)?;
    let qgs = qgs1 + 0.5 * (qa - qb + qc - qd);
    let qgd = qgd1 + 0.5 * (qa - qc + qb - qd);
    let (capgs, capgd) = match rule {
        StatzCapacitance::Averaged => {
            // Preserve the reference sum-then-scale order, including overflow.
            let gate_source_sum = cgsna + cgsnc;
            let gate_drain_sum = cgdna + cgdnb;
            (0.5 * gate_source_sum, 0.5 * gate_drain_sum)
        }
        StatzCapacitance::Present => (cgsna, cgdna),
    };
    Ok(gate_charges(qgs, qgd, capgs, capgd))
}

pub(in crate::devices) const fn gate_charges(
    qgs: Real,
    qgd: Real,
    capgs: Real,
    capgd: Real,
) -> [ChargeBranch; 2] {
    [
        ChargeBranch::new(slot::QGS, qgs, capgs),
        ChargeBranch::new(slot::QGD, qgd, capgd),
    ]
}

/// Physical drain/gate/source voltages to device-frame gate voltages.
pub(in crate::devices) fn normalized(pol: Real, [d, g, s]: [Real; 3]) -> [Real; 2] {
    [pol * (g - s), pol * (g - d)]
}

/// All three C setups allocate this exact `TSTALLOC` sequence. Include
/// aliased prime connections; `MatrixSetup` itself omits ground entries.
pub(in crate::devices) fn setup_gate_matrix(
    setup: &mut crate::devices::setup::MatrixSetup<'_>,
    terminals: &[NodeId],
    inner: [NodeId; 3],
) {
    let [d, g, s] = [terminals[0], terminals[1], terminals[2]];
    let [dp, _, sp] = inner;
    for (row, col) in [
        (d, dp),
        (g, dp),
        (g, sp),
        (s, sp),
        (dp, d),
        (dp, g),
        (dp, sp),
        (sp, g),
        (sp, s),
        (sp, dp),
        (d, d),
        (g, g),
        (s, s),
        (dp, dp),
        (sp, sp),
    ] {
        setup.nodes(row, col);
    }
}

impl Start {
    fn report(self, states: &mut DeviceState<'_>, l: &Loaded) -> SpiceResult<()> {
        let nonconvergent = match self {
            Self::Fixed => true,
            Self::Conditions | Self::Exact => false,
            Self::HeldOff { raw } => {
                let predicted = predict_currents(
                    raw,
                    [l.vgs, l.vgd],
                    [l.cg, l.cd],
                    [l.gm, l.gds, l.ggs, l.ggd],
                );
                states
                    .convergence_tolerances()
                    .is_some_and(|tolerances| exceeds(predicted, l, tolerances, false))
            }
            Self::Limited {
                icheck,
                cghat,
                cdhat,
            } => {
                icheck
                    || states
                        .convergence_tolerances()
                        .is_some_and(|tolerances| exceeds([cghat, cdhat], l, tolerances, true))
            }
        };
        if nonconvergent {
            states.report_nonconvergence()?;
        }
        Ok(())
    }
}

/// The limited gate comparison is inclusive; held-off and drain comparisons
/// are strict. This distinction is observable at the convergence boundary.
fn exceeds(
    [cghat, cdhat]: [Real; 2],
    l: &Loaded,
    (reltol, abstol): (Real, Real),
    inclusive: bool,
) -> bool {
    let gate_error = (cghat - l.cg).abs();
    let gate_limit = reltol * cghat.abs().max(l.cg.abs()) + abstol;
    let gate_failed = if inclusive {
        gate_error >= gate_limit
    } else {
        gate_error > gate_limit
    };
    gate_failed || (cdhat - l.cd).abs() > reltol * cdhat.abs().max(l.cd.abs()) + abstol
}

/// The end of `jfetload.c`/`jfet2load.c`: the convergence check (`icheck`,
/// then the gate and drain currents against their linear prediction from
/// the previous load), the `CKTstate0` writes and the current vector and Y
/// matrix in C's order and arithmetic, every stamp scaled by `m`.
/// `terminals` are the external d, g, s and `inner` the intrinsic d', g, s'.
pub(in crate::devices) struct GateStamp<'a> {
    pub(in crate::devices) terminals: &'a [NodeId],
    pub(in crate::devices) inner: [NodeId; 3],
    pub(in crate::devices) polarity: Real,
    pub(in crate::devices) multiplicity: Real,
    pub(in crate::devices) series: [Real; 2],
}

impl GateStamp<'_> {
    pub(in crate::devices) fn finish(
        self,
        context: &mut StampContext<'_>,
        start: Start,
        l: &Loaded,
    ) -> SpiceResult<()> {
        let Self {
            terminals,
            inner,
            polarity: pol,
            multiplicity: m,
            series: [gdpr, gspr],
        } = self;
        start.report(&mut context.states, l)?;
        for (slot, value) in [
            (slot::VGS, l.vgs),
            (slot::VGD, l.vgd),
            (slot::CG, l.cg),
            (slot::CD, l.cd),
            (slot::CGD, l.cgd),
            (slot::GM, l.gm),
            (slot::GDS, l.gds),
            (slot::GGS, l.ggs),
            (slot::GGD, l.ggd),
        ] {
            context.states.set(slot, value)?;
        }
        let t = pol;
        let Loaded {
            vgs,
            vgd,
            vds,
            cg,
            cgd,
            cd,
            gm,
            gds,
            ggs,
            ggd,
        } = *l;
        let ceqgd = t * (cgd - ggd * vgd);
        let ceqgs = t * ((cg - cgd) - ggs * vgs);
        let cdreq = t * ((cd + cgd) - gds * vds - gm * vgs);
        let [d, g, s] = [0, 1, 2].map(|k| terminals[k]);
        let [dp, _, sp] = inner;
        context.stamp_rhs(g, m * (-ceqgs - ceqgd))?;
        context.stamp_rhs(dp, m * (-cdreq + ceqgd))?;
        context.stamp_rhs(sp, m * (cdreq + ceqgs))?;
        context.stamp_fixed(
            [
                (d, dp, -gdpr),
                (g, dp, -ggd),
                (g, sp, -ggs),
                (s, sp, -gspr),
                (dp, d, -gdpr),
                (dp, g, gm - ggd),
                (dp, sp, -gds - gm),
                (sp, g, -ggs - gm),
                (sp, s, -gspr),
                (sp, dp, -gds),
                (d, d, gdpr),
                (g, g, ggd + ggs),
                (s, s, gspr),
                (dp, dp, gdpr + gds + ggd),
                (sp, sp, gspr + gds + gm + ggs),
            ]
            .map(|(r, c, value)| (r, c, m * value)),
        )?;
        Ok(())
    }
}

/// The `uic` initial conditions `[IC-VDS, IC-VGS]`: unset components are
/// the external terminal voltages of the solution (`jfetic.c`,
/// `jfet2ic.c`).
fn start_conditions(
    terminals: &[NodeId],
    initial: &crate::devices::initial::InstanceInitial,
    voltage: impl Fn(NodeId) -> Real,
) -> [Real; 2] {
    let [d, g, s] = [0, 1, 2].map(|k| terminals[k]);
    [
        initial.values[0].unwrap_or_else(|| voltage(d) - voltage(s)),
        initial.values[1].unwrap_or_else(|| voltage(g) - voltage(s)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::devices::StateHistory;
    use crate::maths::integrator::{IntegrationMethod, StepHistory};

    const DRAIN: usize = slot::COUNT;
    const COUNT: usize = DRAIN + 2;

    fn history() -> StateHistory {
        let mut history = StateHistory::new(COUNT);
        let mut trial = history.trial();
        {
            let mut states = history.device(&mut trial, 0..COUNT).unwrap();
            for index in 0..COUNT {
                states.set(index, 0.).unwrap();
            }
            for (index, value) in [
                (slot::VGS, -0.25),
                (slot::VGD, -0.5),
                (slot::QGS, 2.),
                (slot::QGS + 1, 1.),
                (slot::QGD, 4.),
                (slot::QGD + 1, 2.),
                (DRAIN, 6.),
                (DRAIN + 1, 3.),
            ] {
                states.set(index, value).unwrap();
            }
        }
        history.commit(trial).unwrap();
        history
    }

    fn charges() -> [ChargeBranch; 2] {
        gate_charges(5., 9., 2., 3.)
    }

    fn loaded() -> Loaded {
        Loaded::from_currents([-0.25, -0.5], (5., 2., 3.), (1., 0.5), (2., 0.25))
    }

    #[test]
    fn phase_rejects_invalid_companion_contexts_before_load() {
        let history = history();
        let mut trial = history
            .trial_in(IterationPhase::Predict, None)
            .unwrap()
            .with_initial_transient(true);
        let states = history.device(&mut trial, 0..COUNT).unwrap();
        let coefficients = StepHistory::new()
            .trial(IntegrationMethod::Trapezoidal, 1, 0.25, 0.5)
            .unwrap();
        let transient = AnalysisMode::Transient {
            time: 0.25,
            dt: 0.25,
        };
        for (mode, integration) in [
            (AnalysisMode::Ac { frequency: 10. }, None),
            (AnalysisMode::OperatingPoint, Some(&coefficients)),
            (AnalysisMode::DcSweep, Some(&coefficients)),
            (transient, None),
            (
                AnalysisMode::Transient {
                    time: 0.25,
                    dt: 0.5,
                },
                Some(&coefficients),
            ),
        ] {
            assert!(LoadPhase::new(mode, integration, &states, "test").is_err());
        }
        let phase = LoadPhase::new(transient, Some(&coefficients), &states, "test").unwrap();
        assert!(matches!(phase, LoadPhase::Transient { seed: true, .. }));
        assert!(phase.updates_statz());
        assert_eq!(phase.step().map(Real::to_bits), Some(0.25_f64.to_bits()));
        assert!(
            !LoadPhase::new(AnalysisMode::OperatingPoint, None, &states, "test")
                .unwrap()
                .updates_statz()
        );
        assert!(
            LoadPhase::new(AnalysisMode::DcSweep, None, &states, "test")
                .unwrap()
                .updates_statz()
        );
    }

    #[test]
    fn startup_selects_ic_off_and_first_transient_history() {
        let history = history();
        let junctions = Junctions {
            pol: -1.,
            off: true,
            vt: 0.025,
            vcrit: 0.7,
            vto: -2.,
        };
        let raw = [0.4, -0.7];
        let ic = [2., 1.];
        for (phase, conditions, limiting, initial, expected, start) in [
            (
                IterationPhase::Junction,
                true,
                true,
                false,
                [-1., 1.],
                Start::Conditions,
            ),
            (
                IterationPhase::Junction,
                false,
                true,
                false,
                [0.; 2],
                Start::Fixed,
            ),
            (
                IterationPhase::Fix,
                false,
                true,
                false,
                [0.; 2],
                Start::HeldOff { raw },
            ),
            (
                IterationPhase::Predict,
                false,
                true,
                true,
                [-0.25, -0.5],
                Start::Fixed,
            ),
            (
                IterationPhase::Predict,
                false,
                false,
                true,
                raw,
                Start::Exact,
            ),
        ] {
            let mut trial = history
                .trial_in(phase, None)
                .unwrap()
                .with_initial_conditions(conditions)
                .with_device_limiting(limiting)
                .with_initial_transient(initial);
            let states = history.device(&mut trial, 0..COUNT).unwrap();
            assert_eq!(
                select_junctions(junctions, &states, raw, ic, 0.5),
                (expected, start)
            );
        }
    }

    #[test]
    fn predictor_uses_accepted_voltages_and_nonunit_ratio() {
        let mut history = history();
        let mut accepted = history.trial();
        {
            let mut states = history.device(&mut accepted, 0..COUNT).unwrap();
            for slot in 0..COUNT {
                states.set(slot, 0.).unwrap();
            }
            states.set(slot::VGS, -0.125).unwrap();
            states.set(slot::VGD, -0.375).unwrap();
            states.set(slot::CG, 1.).unwrap();
            states.set(slot::CD, 2.).unwrap();
            states.set(slot::GGS, 4.).unwrap();
            states.set(slot::GGD, 8.).unwrap();
            states.set(slot::GM, 16.).unwrap();
        }
        history.commit(accepted).unwrap();
        let mut trial = history
            .trial_in(IterationPhase::Predict, None)
            .unwrap()
            .with_device_limiting(true);
        let states = history.device(&mut trial, 0..COUNT).unwrap();
        let junctions = Junctions {
            pol: 1.,
            off: false,
            vt: 0.025,
            vcrit: 0.7,
            vto: -2.,
        };
        let expected = [-0.0625, -0.3125];
        let (voltage, start) = select_junctions(junctions, &states, [-0.9, -0.8], [0.; 2], 0.5);
        assert_eq!(voltage, expected);
        assert_eq!(
            start,
            Start::Limited {
                icheck: false,
                cghat: 1.75,
                cdhat: 2.5
            }
        );
    }

    #[test]
    fn initial_transient_seed_is_family_specific_and_trial_local() {
        let history = history();
        let coefficients = StepHistory::new()
            .trial(IntegrationMethod::Trapezoidal, 1, 0.25, 0.5)
            .unwrap();
        for (update, seeds, extra) in [
            (GateUpdate::Depletion(charges()), true, false),
            (
                GateUpdate::Parker {
                    gates: charges(),
                    drain: ChargeBranch::new(DRAIN, 11., 4.),
                },
                true,
                true,
            ),
            (GateUpdate::Statz(charges()), false, false),
        ] {
            let mut trial = history
                .trial_in(IterationPhase::Predict, None)
                .unwrap()
                .with_initial_transient(true);
            let mut states = history.device(&mut trial, 0..COUNT).unwrap();
            let phase = LoadPhase::new(
                AnalysisMode::Transient {
                    time: 0.25,
                    dt: 0.25,
                },
                Some(&coefficients),
                &states,
                "test",
            )
            .unwrap();
            let mut l = loaded();
            update.apply(phase, &mut states, &mut l).unwrap();
            let derivatives = if seeds { [0.; 2] } else { [12., 20.] };
            assert_eq!(states.trial(slot::QGS + 1), Some(derivatives[0]));
            assert_eq!(states.trial(slot::QGD + 1), Some(derivatives[1]));
            assert_eq!(
                states.accepted(1, slot::QGS),
                Some(if seeds { 5. } else { 2. })
            );
            assert_eq!(states.trial(slot::QGS), Some(5.));
            assert_eq!(states.trial(slot::QGD), Some(9.));
            assert_eq!(l.ggs.to_bits(), 8.5_f64.to_bits());
            assert_eq!(l.ggd.to_bits(), 12.25_f64.to_bits());
            assert_eq!(
                l.cg.to_bits(),
                (3. + derivatives[0] + derivatives[1]).to_bits()
            );
            assert_eq!(l.cd.to_bits(), (3. - derivatives[1]).to_bits());
            if extra {
                assert_eq!(states.trial(DRAIN), Some(11.));
                assert_eq!(states.accepted(1, DRAIN), Some(11.));
                assert_eq!(l.gds.to_bits(), 3_f64.to_bits());
            }
        }
        // Rejected first-step attempts may patch their private state1 view;
        // the original accepted history must stay untouched.
        let accepted = history.accepted(1).unwrap();
        assert_eq!(&accepted[slot::QGS..], &[2., 1., 4., 2., 6., 3.]);
    }

    #[test]
    fn dc_derivatives_are_zeroed_or_carried_by_family() {
        let history = history();
        for (update, expected) in [
            (GateUpdate::Depletion(charges()), [0., 0.]),
            (GateUpdate::Statz(charges()), [0., 0.]),
            (
                GateUpdate::Parker {
                    gates: charges(),
                    drain: ChargeBranch::new(DRAIN, 11., 4.),
                },
                [1., 2.],
            ),
        ] {
            let mut trial = history.trial();
            let mut states = history.device(&mut trial, 0..COUNT).unwrap();
            let mut point = loaded();
            let phase = LoadPhase::new(AnalysisMode::DcSweep, None, &states, "test").unwrap();
            update.apply(phase, &mut states, &mut point).unwrap();
            assert_eq!(states.trial(slot::QGS + 1), Some(expected[0]));
            assert_eq!(states.trial(slot::QGD + 1), Some(expected[1]));
            assert_eq!(states.trial(slot::QGS), Some(5.));
            assert_eq!(point.cd.to_bits(), 3_f64.to_bits());
            assert_eq!(point.ggs.to_bits(), 0.5_f64.to_bits());
        }
    }

    #[test]
    fn parker_drain_companion_adds_only_current_in_trap_and_gear() {
        let mut history = history();
        let mut steps = StepHistory::new();
        let first = steps
            .trial(IntegrationMethod::Trapezoidal, 1, 0.25, 0.5)
            .unwrap();
        steps.accept(&first);
        // Two accepted steps provide history for Gear2.
        let mut accepted = history.trial();
        {
            let mut states = history.device(&mut accepted, 0..COUNT).unwrap();
            for index in 0..COUNT {
                states.set(index, 0.).unwrap();
            }
            for (index, value) in [
                (slot::QGS, 3.),
                (slot::QGS + 1, 2.),
                (slot::QGD, 5.),
                (slot::QGD + 1, 3.),
                (DRAIN, 7.),
                (DRAIN + 1, 4.),
            ] {
                states.set(index, value).unwrap();
            }
        }
        history.commit(accepted).unwrap();
        for (method, order) in [
            (IntegrationMethod::Trapezoidal, 2),
            (IntegrationMethod::Gear { order: 2 }, 2),
        ] {
            let coefficients = steps.trial(method, order, 0.5, 0.5).unwrap();
            let mut trial = history.trial();
            let mut states = history.device(&mut trial, 0..COUNT).unwrap();
            let phase = LoadPhase::new(
                AnalysisMode::Transient {
                    time: 0.75,
                    dt: 0.5,
                },
                Some(&coefficients),
                &states,
                "test",
            )
            .unwrap();
            let gs = coefficients.integrate(&[5., 3., 2.], Some(2.), 2.).unwrap();
            let gd = coefficients.integrate(&[9., 5., 4.], Some(3.), 3.).unwrap();
            let ds = coefficients
                .integrate(&[11., 7., 6.], Some(4.), 4.)
                .unwrap();
            let mut l = loaded();
            GateUpdate::Parker {
                gates: charges(),
                drain: ChargeBranch::new(DRAIN, 11., 4.),
            }
            .apply(phase, &mut states, &mut l)
            .unwrap();
            assert_eq!(
                l.cd.to_bits(),
                (3. - gd.derivative + ds.derivative).to_bits()
            );
            assert_eq!(
                l.cg.to_bits(),
                (3. + gs.derivative + gd.derivative).to_bits()
            );
            assert_eq!(l.gds.to_bits(), 3_f64.to_bits());
            assert_eq!(
                states.trial(DRAIN + 1).map(Real::to_bits),
                Some(ds.derivative.to_bits())
            );
        }
    }

    #[test]
    fn convergence_distinguishes_gate_equality_from_drain_equality() {
        let l = Loaded::from_currents([0.; 2], (0., 0., 0.), (0., 1.), (0., 0.));
        let history = StateHistory::new(0);
        for (start, expected) in [
            (Start::HeldOff { raw: [1., 0.] }, false),
            (
                Start::Limited {
                    icheck: false,
                    cghat: 1.,
                    cdhat: 0.,
                },
                true,
            ),
            (
                Start::Limited {
                    icheck: false,
                    cghat: 0.,
                    cdhat: 1.,
                },
                false,
            ),
            (
                Start::Limited {
                    icheck: true,
                    cghat: 0.,
                    cdhat: 0.,
                },
                true,
            ),
            (Start::Fixed, true),
            (Start::Conditions, false),
            (Start::Exact, false),
        ] {
            let mut trial = history
                .trial_in(IterationPhase::Float, None)
                .unwrap()
                .with_convergence_tolerances(0., 1.);
            start
                .report(&mut history.device(&mut trial, 0..0).unwrap(), &l)
                .unwrap();
            assert_eq!(trial.is_nonconvergent(), expected);
        }
    }
}
