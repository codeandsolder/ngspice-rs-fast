//! Junction diode conduction/reverse tunneling and depletion charge stages.
//! Ownership: BSIM4 load inventory fields cbs/cbd/gbs/gbd and capbs/capbd;
//! charge also writes QBS/QBD (no integration here).
use std::cmp::Ordering;

use super::{DEXP, EXP_THRESHOLD, Instance, LoadCkt, MIN_EXP, Model, SizeParam, state};
use crate::devices::bsim_physics::{JunctionCapacitance, depletion};
use crate::devices::nonlinear::exponential_junction_at_argument;
use crate::maths::taylor::Constitutive;
use crate::primitives::SpiceResult;

/// Body-source and body-drain voltage, selecting bulk-resistance internal
/// nodes when present. Both are already polarity-normalized and limited.
#[derive(Clone, Copy)]
pub(super) struct JunctionBias<S = f64> {
    pub(super) source: S,
    pub(super) drain: S,
}

/// Temperature-prepared diode quantities for one physical junction.
/// The drain/source selection occurs once, not inside the constitutive law.
#[derive(Clone, Copy)]
struct JunctionDiode {
    saturation: f64,
    thermal: f64,
    breakdown_voltage: f64,
    breakdown_coefficient: f64,
    breakdown_at_zero: f64,
    forward_voltage: f64,
    forward_current: f64,
    forward_slope: f64,
    reverse_voltage: f64,
    reverse_current: f64,
    reverse_slope: f64,
}

impl JunctionDiode {
    /// The change of the reverse breakdown exponential from equilibrium.
    /// In the same unclipped branch, factor the zero-bias exponential
    /// and use expm1; never cancel across an exponential clamp.
    fn breakdown_difference<S: Constitutive>(
        self,
        argument: &S,
        reverse_argument: &S,
        reverse_exp: &S,
        clipped: bool,
    ) -> S {
        if argument.primal().abs() < 0.5
            && self.thermal > 0.
            && self.thermal.is_finite()
            && self.breakdown_coefficient.is_finite()
            && self.breakdown_voltage / self.thermal <= EXP_THRESHOLD
            && !clipped
            && reverse_argument.primal().is_finite()
            && self.breakdown_at_zero.is_finite()
        {
            argument
                .clone()
                .scale(-1.)
                .exp_m1()
                .scale(-self.breakdown_at_zero)
        } else {
            argument
                .constant(self.breakdown_at_zero)
                .add(&reverse_exp.clone().scale(-self.breakdown_coefficient))
        }
    }

    /// C diode modes select distinct forward and reverse continuations.
    /// Return the physical current and its partial, including gmin.
    fn evaluate<S: Constitutive>(
        self,
        selector: i32,
        voltage: &S,
        gmin: f64,
    ) -> SpiceResult<Option<(S, S)>> {
        if self.saturation <= 0. {
            return Ok(Some((voltage.clone().scale(gmin), voltage.constant(gmin))));
        }
        let argument = voltage.clone().divide(&voltage.constant(self.thermal))?;
        let (current, slope) = match selector {
            0 => self.unclipped(voltage, &argument)?,
            1 => self.forward_limited(voltage, &argument),
            2 => self.both_limited(voltage, &argument)?,
            _ => return Ok(None),
        };
        Ok(Some((
            voltage.constant(gmin).mul_add(voltage, &current),
            slope.offset(gmin),
        )))
    }

    fn unclipped<S: Constitutive>(self, voltage: &S, argument: &S) -> SpiceResult<(S, S)> {
        let reverse_argument = voltage
            .clone()
            .offset(self.breakdown_voltage)
            .divide(&voltage.constant(self.thermal))?;
        let reverse = reverse_argument.clone().scale(-1.).exp();
        let (forward, forward_slope) = exponential(argument, self.thermal, self.saturation);
        let difference = self.breakdown_difference(argument, &reverse_argument, &reverse, false);
        let reverse_slope = reverse
            .scale(self.breakdown_coefficient)
            .divide(&voltage.constant(self.thermal))?;
        Ok((
            voltage
                .constant(self.saturation)
                .mul_add(&difference, &forward),
            voltage
                .constant(self.saturation)
                .mul_add(&reverse_slope, &forward_slope),
        ))
    }

    fn forward_limited<S: Constitutive>(self, voltage: &S, argument: &S) -> (S, S) {
        if argument.compare(-EXP_THRESHOLD) == Some(Ordering::Less) {
            (
                voltage.constant(self.saturation * (MIN_EXP - 1.)),
                voltage.constant(0.),
            )
        } else if voltage
            .compare(self.forward_voltage)
            .is_some_and(Ordering::is_le)
        {
            exponential(argument, self.thermal, self.saturation)
        } else {
            let slope = self.forward_current / self.thermal;
            let edge_argument = self.forward_voltage / self.thermal;
            let edge = if edge_argument.abs() < 0.5 {
                self.saturation * edge_argument.exp_m1()
            } else {
                self.forward_current - self.saturation
            };
            (
                voltage.constant(slope).mul_add(
                    &voltage.clone().offset(-self.forward_voltage),
                    &voltage.constant(edge),
                ),
                voltage.constant(slope),
            )
        }
    }

    fn both_limited<S: Constitutive>(self, voltage: &S, argument: &S) -> SpiceResult<(S, S)> {
        if voltage.compare(self.reverse_voltage) == Some(Ordering::Less) {
            let (difference, derivative) = limited_forward(argument, self.thermal)?;
            let edge = voltage.constant(self.reverse_slope).mul_add(
                &voltage.clone().offset(-self.reverse_voltage),
                &voltage.constant(self.reverse_current),
            );
            let slope = derivative.mul_add(&edge, &difference.clone().scale(self.reverse_slope));
            Ok((difference.multiply(&edge), slope))
        } else if voltage
            .compare(self.forward_voltage)
            .is_some_and(Ordering::is_le)
        {
            let (difference, derivative) = limited_forward(argument, self.thermal)?;
            let reverse_argument = voltage
                .clone()
                .offset(self.breakdown_voltage)
                .divide(&voltage.constant(self.thermal))?;
            let clipped = reverse_argument.compare(EXP_THRESHOLD) == Some(Ordering::Greater);
            let (reverse, reverse_slope) = if clipped {
                (voltage.constant(MIN_EXP), voltage.constant(0.))
            } else {
                let exponential = reverse_argument.clone().scale(-1.).exp();
                let slope = exponential
                    .clone()
                    .divide(&voltage.constant(self.thermal))?;
                (exponential, slope)
            };
            let reverse_difference =
                self.breakdown_difference(argument, &reverse_argument, &reverse, clipped);
            Ok((
                difference.add(&reverse_difference).scale(self.saturation),
                voltage
                    .constant(self.breakdown_coefficient)
                    .mul_add(&reverse_slope, &derivative)
                    .scale(self.saturation),
            ))
        } else {
            Ok((
                voltage.constant(self.forward_slope).mul_add(
                    &voltage.clone().offset(-self.forward_voltage),
                    &voltage.constant(self.forward_current),
                ),
                voltage.constant(self.forward_slope),
            ))
        }
    }
}

/// The same cancellation- and overflow-safe exponential law used by the
/// scalar junctions, lifted from its scaled conductance rather than exp(v/Vt).
fn exponential<S: Constitutive>(argument: &S, thermal: f64, saturation: f64) -> (S, S) {
    let (value, slope) = exponential_junction_at_argument(argument.primal(), thermal, saturation);
    let slope = argument.clone().exp_with_primal(slope);
    let current = argument
        .clone()
        .compose_derivative(value, &slope.clone().scale(thermal));
    (current, slope)
}

fn limited_forward<S: Constitutive>(argument: &S, thermal: f64) -> SpiceResult<(S, S)> {
    if argument.compare(-EXP_THRESHOLD) == Some(Ordering::Less) {
        Ok((argument.constant(MIN_EXP - 1.), argument.constant(0.)))
    } else if argument.primal().abs() < 0.5 {
        let difference = argument.clone().exp_m1();
        let slope = difference
            .clone()
            .offset(1.)
            .divide(&argument.constant(thermal))?;
        Ok((difference, slope))
    } else {
        let exponential = argument.clone().exp();
        let slope = exponential.clone().divide(&argument.constant(thermal))?;
        Ok((exponential.offset(-1.), slope))
    }
}

/// One trap-assisted tunnelling law, shared by area, sidewall and
/// gate-sidewall terms on both physical junctions. C stores the negative
/// of its exponent derivative because the tunnelling current is subtracted.
fn reverse_tunneling<S: Constitutive>(
    voltage: S,
    thermal: f64,
    threshold: f64,
) -> SpiceResult<(S, S)> {
    let distance = voltage.clone().scale(-1.).offset(threshold);
    let normalized = -voltage.primal() / thermal;
    let (value, derivative) = if distance.compare(threshold * 1e-3) == Some(Ordering::Less) {
        (normalized * 1e3, voltage.constant(1e3 / thermal))
    } else {
        let reciprocal = voltage.constant(1.).divide(&distance)?;
        let derivative = if threshold > 0.
            && thermal > 0.
            && threshold.is_finite()
            && thermal.is_finite()
            && voltage.primal().is_finite()
        {
            let factor = voltage.constant(threshold).divide(&distance)?;
            factor
                .clone()
                .multiply(&factor.divide(&voltage.constant(thermal))?)
        } else {
            // Keep the reference nonphysical argument propagation.
            voltage
                .clone()
                .multiply(&reciprocal)
                .mul_add(&reciprocal, &reciprocal)
                .scale(threshold / thermal)
        };
        (normalized * threshold * reciprocal.primal(), derivative)
    };
    // The independently rationalized slope avoids cancellation of the two
    // quotient partials at very large reverse bias. Use it to lift the
    // original rounded argument rather than differentiating the quotient.
    let argument = voltage.compose_derivative(value, &derivative.clone().scale(-1.));
    let (exponential, slope) = DEXP(argument.clone());
    let difference = if argument.primal().abs() < 0.5 {
        argument.exp_m1()
    } else {
        exponential.offset(-1.)
    };
    Ok((difference, slope.multiply(&derivative)))
}

/// Evaluate the four load-owned junction conduction members.
#[expect(
    clippy::suboptimal_flops,
    reason = "ordered saturation-current geometry products must match temperature preparation of the limiting continuations"
)]
pub(super) fn current(
    model: &Model,
    here: &mut Instance,
    size: &SizeParam,
    ckt: &dyn LoadCkt,
    bias: JunctionBias,
) -> SpiceResult<()> {
    current_constitutive(model, here, size, ckt.gmin(), bias)?;
    Ok(())
}

pub(super) fn current_constitutive<S: Constitutive>(
    model: &Model,
    here: &mut Instance,
    size: &SizeParam,
    gmin: f64,
    bias: JunctionBias<S>,
) -> SpiceResult<[S; 2]> {
    let gate_perimeter = size.weffCJ * here.nf;
    let source_saturation = if here.Aseff <= 0. && here.Pseff <= 0. {
        0.
    } else {
        here.Aseff * model.SjctTempSatCurDensity
            + here.Pseff * model.SjctSidewallTempSatCurDensity
            + gate_perimeter * model.SjctGateSidewallTempSatCurDensity
    };
    let drain_saturation = if here.Adeff <= 0. && here.Pdeff <= 0. {
        0.
    } else {
        here.Adeff * model.DjctTempSatCurDensity
            + here.Pdeff * model.DjctSidewallTempSatCurDensity
            + gate_perimeter * model.DjctGateSidewallTempSatCurDensity
    };
    let source = JunctionDiode {
        saturation: source_saturation,
        thermal: model.vtm * model.SjctEmissionCoeff,
        breakdown_voltage: model.bvs,
        breakdown_coefficient: model.xjbvs,
        breakdown_at_zero: here.XExpBVS,
        forward_voltage: here.vjsmFwd,
        forward_current: here.IVjsmFwd,
        forward_slope: here.SslpFwd,
        reverse_voltage: here.vjsmRev,
        reverse_current: here.IVjsmRev,
        reverse_slope: here.SslpRev,
    };
    let drain = JunctionDiode {
        saturation: drain_saturation,
        thermal: model.vtm * model.DjctEmissionCoeff,
        breakdown_voltage: model.bvd,
        breakdown_coefficient: model.xjbvd,
        breakdown_at_zero: here.XExpBVD,
        forward_voltage: here.vjdmFwd,
        forward_current: here.IVjdmFwd,
        forward_slope: here.DslpFwd,
        reverse_voltage: here.vjdmRev,
        reverse_current: here.IVjdmRev,
        reverse_slope: here.DslpRev,
    };
    let (source_current, source_slope) = source
        .evaluate(model.dioMod, &bias.source, gmin)?
        .ok_or_else(|| {
            crate::primitives::SpiceError::circuit("invalid BSIM4 junction diode mode")
        })?;
    let (drain_current, drain_slope) = drain
        .evaluate(model.dioMod, &bias.drain, gmin)?
        .ok_or_else(|| {
            crate::primitives::SpiceError::circuit("invalid BSIM4 junction diode mode")
        })?;
    let source_tunnels = [
        (here.SjctTempRevSatCur, model.njtsstemp, model.vtss),
        (here.SswTempRevSatCur, model.njtsswstemp, model.vtssws),
        (here.SswgTempRevSatCur, model.njtsswgstemp, model.vtsswgs),
    ];
    let drain_tunnels = [
        (here.DjctTempRevSatCur, model.njtsdtemp, model.vtsd),
        (here.DswTempRevSatCur, model.njtsswdtemp, model.vtsswd),
        (here.DswgTempRevSatCur, model.njtsswgdtemp, model.vtsswgd),
    ];
    let apply_tunnels =
        |voltage: S, terms: [(f64, f64, f64); 3], current: S, slope: S| -> SpiceResult<(S, S)> {
            let evaluate = |(density, emission, threshold)| {
                let (difference, derivative) =
                    reverse_tunneling(voltage.clone(), model.vtm0 * emission, threshold)?;
                Ok::<_, crate::primitives::SpiceError>((
                    difference.scale(density),
                    derivative.scale(density),
                ))
            };
            let [area, sidewall, gate] = terms.map(evaluate);
            let (area, area_slope) = area?;
            let (sidewall, sidewall_slope) = sidewall?;
            let (gate, gate_slope) = gate?;
            Ok((
                current.add(&area.add(&sidewall).add(&gate).scale(-1.)),
                slope.add(&area_slope.add(&sidewall_slope).add(&gate_slope)),
            ))
        };
    let (source, source_slope) =
        apply_tunnels(bias.source, source_tunnels, source_current, source_slope)?;
    let (drain, drain_slope) =
        apply_tunnels(bias.drain, drain_tunnels, drain_current, drain_slope)?;
    here.cbs = source.primal();
    here.gbs = source_slope.primal();
    here.cbd = drain.primal();
    here.gbd = drain_slope.primal();
    Ok([source, drain])
}

/// Evaluate depletion charge and capacitance (C's finished label).
pub(super) fn charge(
    model: &Model,
    here: &mut Instance,
    size: &SizeParam,
    ckt: &mut dyn LoadCkt,
    bias: JunctionBias,
) -> SpiceResult<()> {
    charge_constitutive(model, here, size, bias, |slot, charge| {
        ckt.set_state0(slot, *charge)
    })?;
    Ok(())
}

pub(super) fn charge_constitutive<S: Constitutive>(
    model: &Model,
    here: &mut Instance,
    size: &SizeParam,
    bias: JunctionBias<S>,
    mut record: impl FnMut(usize, &S),
) -> SpiceResult<[S; 2]> {
    let (source_charge, source_capacitance) = depletion(
        &bias.source,
        [
            JunctionCapacitance {
                zero: model.SunitAreaTempJctCap * here.Aseff,
                potential: model.PhiBS,
                grading: model.SbulkJctBotGradingCoeff,
            },
            JunctionCapacitance {
                zero: model.SunitLengthSidewallTempJctCap * here.Pseff,
                potential: model.PhiBSWS,
                grading: model.SbulkJctSideGradingCoeff,
            },
            JunctionCapacitance {
                zero: model.SunitLengthGateSidewallTempJctCap * size.weffCJ * here.nf,
                potential: model.PhiBSWGS,
                grading: model.SbulkJctGateSideGradingCoeff,
            },
        ],
        |charge| record(state::QBS, charge),
    )?;
    here.capbs = source_capacitance.primal();
    let (drain_charge, drain_capacitance) = depletion(
        &bias.drain,
        [
            JunctionCapacitance {
                zero: model.DunitAreaTempJctCap * here.Adeff,
                potential: model.PhiBD,
                grading: model.DbulkJctBotGradingCoeff,
            },
            JunctionCapacitance {
                zero: model.DunitLengthSidewallTempJctCap * here.Pdeff,
                potential: model.PhiBSWD,
                grading: model.DbulkJctSideGradingCoeff,
            },
            JunctionCapacitance {
                zero: model.DunitLengthGateSidewallTempJctCap * size.weffCJ * here.nf,
                potential: model.PhiBSWGD,
                grading: model.DbulkJctGateSideGradingCoeff,
            },
        ],
        |charge| record(state::QBD, charge),
    )?;
    here.capbd = drain_capacitance.primal();
    Ok([source_charge, drain_charge])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diode() -> JunctionDiode {
        JunctionDiode {
            saturation: 1e-12,
            thermal: 0.026,
            breakdown_voltage: 0.026 * 2_f64.ln(),
            breakdown_coefficient: 1.,
            breakdown_at_zero: 0.5,
            forward_voltage: 0.1,
            forward_current: 1e-12 * (0.1_f64 / 0.026).exp(),
            forward_slope: 1e-12 * ((0.1_f64 / 0.026).exp() + 0.5 * (-0.1_f64 / 0.026).exp())
                / 0.026,
            reverse_voltage: -1.,
            reverse_current: 1e-5,
            reverse_slope: -1e-4,
        }
    }

    #[test]
    fn all_diode_modes_preserve_tiny_bias_and_differential_conductance() {
        for selector in [0, 1, 2] {
            let diode = diode();
            let gain = if selector == 1 { 1. } else { 1.5 };
            let expected = gain * diode.saturation / diode.thermal;
            for voltage in [-1e-22, -1e-18, 1e-18, 1e-22] {
                let (current, slope) = diode.evaluate(selector, &voltage, 0.).unwrap().unwrap();
                assert!((current / (expected * voltage) - 1.).abs() < 8. * f64::EPSILON);
                assert!((slope / expected - 1.).abs() < 8. * f64::EPSILON);
            }
        }
    }

    #[test]
    fn diode_clips_and_continuations_retain_their_defined_regions() {
        let mut diode = diode();
        diode.breakdown_voltage = 40. * diode.thermal;
        diode.breakdown_at_zero = MIN_EXP;
        let (current, slope) = diode.evaluate(2, &0., 0.).unwrap().unwrap();
        assert_eq!(current.to_bits(), 0_f64.to_bits());
        assert!((slope / (diode.saturation / diode.thermal) - 1.).abs() < f64::EPSILON);
        let (current, _) = diode.evaluate(0, &0., 0.).unwrap().unwrap();
        // Mode zero retains the C zero-bias floor but uses the unclipped
        // reverse exponential, unlike mode two.
        assert!(current > 0.);
        let (current, slope) = diode
            .evaluate(1, &(-36. * diode.thermal), 0.)
            .unwrap()
            .unwrap();
        assert_eq!(current, diode.saturation * (MIN_EXP - 1.));
        assert_eq!(slope, 0.);
        diode.forward_voltage = 1e-18;
        diode.forward_current = diode.saturation;
        let (current, _) = diode.evaluate(1, &2e-18, 0.).unwrap().unwrap();
        assert!(
            (current / (2e-18 * diode.saturation / diode.thermal) - 1.).abs() < 4. * f64::EPSILON
        );
    }

    fn close(actual: f64, expected: f64, absolute: f64) {
        assert!(
            (actual - expected).abs() <= absolute + 8e-12 * expected.abs(),
            "{actual:.17e} != {expected:.17e}"
        );
    }

    fn sample(jet: &crate::maths::taylor::Jet, time: f64) -> f64 {
        jet.coefficients()
            .iter()
            .rev()
            .fold(0., |value, coefficient| value.mul_add(time, *coefficient))
    }

    #[test]
    fn diode_modes_share_current_and_partial_germs() -> SpiceResult<()> {
        use crate::maths::taylor::{Jet, differential::Differential};
        let diode = diode();
        let gmin = 1e-14;
        let order = 8;
        for selector in [0, 1, 2] {
            for bias in [-1.25, -0.95, -0.7, -0.1, -1e-18, 0., 0.05, 0.15, 1.] {
                let rate = 0.01;
                let voltage = Differential::<1>::variable(0, Jet::affine(order, bias, rate))?;
                let (current, slope) = diode.evaluate(selector, &voltage, gmin)?.unwrap();
                let (scalar_current, scalar_slope) =
                    diode.evaluate(selector, &bias, gmin)?.unwrap();
                close(current.value.value(), scalar_current, 1e-28);
                close(slope.value.value(), scalar_slope, 1e-28);
                let partial = current.gradient[0].as_ref().unwrap();
                for (&actual, &expected) in partial
                    .coefficients()
                    .iter()
                    .zip(slope.value.coefficients())
                {
                    close(actual, expected, 1e-25);
                }
                for time in [-0.03, 0.03] {
                    let (expected, expected_slope) = diode
                        .evaluate(selector, &rate.mul_add(time, bias), gmin)?
                        .unwrap();
                    close(sample(&current.value, time), expected, 1e-25);
                    close(sample(&slope.value, time), expected_slope, 1e-25);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn diode_continuation_selects_the_right_hand_germ_at_joins() -> SpiceResult<()> {
        use crate::maths::taylor::{Jet, differential::Differential};
        let diode = diode();
        let gmin = 1e-14;
        for (selector, bias) in [
            (1, diode.forward_voltage),
            (2, diode.forward_voltage),
            (2, diode.reverse_voltage),
        ] {
            for rate in [-0.01, 0.01] {
                let voltage = Differential::<1>::variable(0, Jet::affine(8, bias, rate))?;
                let (current, slope) = diode.evaluate(selector, &voltage, gmin)?.unwrap();
                let time = 0.03;
                let (expected, expected_slope) = diode
                    .evaluate(selector, &rate.mul_add(time, bias), gmin)?
                    .unwrap();
                close(sample(&current.value, time), expected, 1e-25);
                close(sample(&slope.value, time), expected_slope, 1e-25);
            }
        }
        Ok(())
    }

    #[test]
    fn reverse_tunneling_shares_the_physical_derivative_across_exponential_regions()
    -> SpiceResult<()> {
        use crate::maths::taylor::{Jet, differential::Differential};
        for (bias, rate) in [
            (-1e18, 1e15),
            (-100., 0.001),
            (-8., 0.001),
            (-3., 0.001),
            (-0.02, 0.001),
            (0., 0.001),
            (0.5, 0.001),
            (9.995, 0.001),
        ] {
            let voltage = Differential::<1>::variable(0, Jet::affine(8, bias, rate))?;
            let (difference, derivative) = reverse_tunneling(voltage, 0.1, 10.)?;
            let (value, slope) = reverse_tunneling(bias, 0.1, 10.)?;
            close(difference.value.value(), value, 1e-24);
            close(derivative.value.value(), slope, 1e-24);
            for (degree, &expected) in derivative.value.coefficients().iter().enumerate() {
                let actual = difference.gradient[0]
                    .as_ref()
                    .map_or(0., |gradient| gradient.coefficients()[degree]);
                close(-actual, expected, 1e-300);
            }
            for time in [-0.03, 0.03] {
                let (value, slope) = reverse_tunneling(rate.mul_add(time, bias), 0.1, 10.)?;
                close(sample(&difference.value, time), value, 1e-24);
                close(sample(&derivative.value, time), slope, 1e-24);
            }
        }
        Ok(())
    }

    #[test]
    fn reverse_tunneling_retains_tiny_current_and_large_reverse_derivative() {
        let (difference, derivative) = reverse_tunneling(-1e-18, 0.1, 10.).unwrap();
        assert!((difference / 1e-17 - 1.).abs() < 4. * f64::EPSILON);
        assert!((derivative / 10. - 1.).abs() < 4. * f64::EPSILON);
        let (_, derivative) = reverse_tunneling(-1e18, 0.1, 10.).unwrap();
        assert!((derivative / (super::super::MAX_EXP * 1e-33) - 1.).abs() < 4. * f64::EPSILON);
    }
}
