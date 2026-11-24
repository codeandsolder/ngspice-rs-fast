//! Junction diode conduction/reverse tunneling and depletion charge stages.
//! Ownership: BSIM4 load inventory fields cbs/cbd/gbs/gbd and capbs/capbd;
//! charge also writes QBS/QBD (no integration here).
use super::{DEXP, EXP_THRESHOLD, Instance, LoadCkt, MIN_EXP, Model, SizeParam, state};
use crate::devices::nonlinear::{exponential_junction_at_argument, reverse_depletion};

/// Body-source and body-drain voltage, selecting bulk-resistance internal
/// nodes when present. Both are already polarity-normalized and limited.
#[derive(Clone, Copy)]
pub(super) struct JunctionBias {
    pub(super) source: f64,
    pub(super) drain: f64,
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
    #[expect(
        clippy::suboptimal_flops,
        reason = "subtract the same rounded breakdown product prepared at zero bias, so the clipped branch remains exactly zero at equilibrium"
    )]
    fn breakdown_difference(
        self,
        argument: f64,
        reverse_argument: f64,
        reverse_exp: f64,
        clipped: bool,
    ) -> f64 {
        if argument.abs() < 0.5
            && self.thermal > 0.
            && self.thermal.is_finite()
            && self.breakdown_coefficient.is_finite()
            && self.breakdown_voltage / self.thermal <= EXP_THRESHOLD
            && !clipped
            && reverse_argument.is_finite()
            && self.breakdown_at_zero.is_finite()
        {
            -self.breakdown_at_zero * (-argument).exp_m1()
        } else {
            self.breakdown_at_zero - self.breakdown_coefficient * reverse_exp
        }
    }

    /// C diode modes select distinct forward and reverse continuations.
    /// Return current/conductance without merging the leakage or gmin laws.
    fn evaluate(self, selector: i32, voltage: f64, gmin: f64) -> Option<(f64, f64)> {
        if self.saturation <= 0. {
            return Some((gmin * voltage, gmin));
        }
        let argument = voltage / self.thermal;
        let (current, slope) = match selector {
            0 => {
                let reverse_argument = (self.breakdown_voltage + voltage) / self.thermal;
                let reverse = (-reverse_argument).exp();
                let (forward, forward_slope) =
                    exponential_junction_at_argument(argument, self.thermal, self.saturation);
                let difference =
                    self.breakdown_difference(argument, reverse_argument, reverse, false);
                (
                    self.saturation.mul_add(difference, forward),
                    self.saturation.mul_add(
                        self.breakdown_coefficient * reverse / self.thermal,
                        forward_slope,
                    ),
                )
            }
            1 => {
                if argument < -EXP_THRESHOLD {
                    (self.saturation * (MIN_EXP - 1.), 0.)
                } else if voltage <= self.forward_voltage {
                    exponential_junction_at_argument(argument, self.thermal, self.saturation)
                } else {
                    let slope = self.forward_current / self.thermal;
                    let edge_argument = self.forward_voltage / self.thermal;
                    let edge = if edge_argument.abs() < 0.5 {
                        self.saturation * edge_argument.exp_m1()
                    } else {
                        self.forward_current - self.saturation
                    };
                    (slope.mul_add(voltage - self.forward_voltage, edge), slope)
                }
            }
            2 => {
                if voltage < self.reverse_voltage {
                    let (difference, derivative) = limited_forward(argument, self.thermal);
                    let edge = self
                        .reverse_slope
                        .mul_add(voltage - self.reverse_voltage, self.reverse_current);
                    (
                        difference * edge,
                        derivative.mul_add(edge, difference * self.reverse_slope),
                    )
                } else if voltage <= self.forward_voltage {
                    let (difference, derivative) = limited_forward(argument, self.thermal);
                    let reverse_argument = (self.breakdown_voltage + voltage) / self.thermal;
                    let clipped = reverse_argument > EXP_THRESHOLD;
                    let (reverse, reverse_slope) = if clipped {
                        (MIN_EXP, 0.)
                    } else {
                        let exponential = (-reverse_argument).exp();
                        (exponential, exponential / self.thermal)
                    };
                    let reverse_difference =
                        self.breakdown_difference(argument, reverse_argument, reverse, clipped);
                    (
                        self.saturation * (difference + reverse_difference),
                        self.saturation
                            * self
                                .breakdown_coefficient
                                .mul_add(reverse_slope, derivative),
                    )
                } else {
                    (
                        self.forward_slope
                            .mul_add(voltage - self.forward_voltage, self.forward_current),
                        self.forward_slope,
                    )
                }
            }
            _ => return None,
        };
        Some((gmin.mul_add(voltage, current), slope + gmin))
    }
}

fn limited_forward(argument: f64, thermal: f64) -> (f64, f64) {
    if argument < -EXP_THRESHOLD {
        (MIN_EXP - 1., 0.)
    } else if argument.abs() < 0.5 {
        let difference = argument.exp_m1();
        (difference, (1. + difference) / thermal)
    } else {
        let exponential = argument.exp();
        (exponential - 1., exponential / thermal)
    }
}

/// One trap-assisted tunnelling law, shared by area, sidewall and
/// gate-sidewall terms on both physical junctions. C stores the negative
/// of its exponent derivative because the tunnelling current is subtracted.
fn reverse_tunneling(voltage: f64, thermal: f64, threshold: f64) -> (f64, f64) {
    let distance = threshold - voltage;
    let (argument, derivative) = if distance < threshold * 1e-3 {
        (-voltage / thermal * 1e3, 1e3 / thermal)
    } else {
        let reciprocal = 1. / distance;
        let derivative = if threshold > 0.
            && thermal > 0.
            && threshold.is_finite()
            && thermal.is_finite()
            && voltage.is_finite()
        {
            let factor = threshold / distance;
            factor * (factor / thermal)
        } else {
            // Keep the reference nonphysical argument propagation.
            threshold / thermal * (voltage * reciprocal).mul_add(reciprocal, reciprocal)
        };
        (-voltage / thermal * threshold * reciprocal, derivative)
    };
    let (exponential, slope) = DEXP(argument);
    let difference = if argument.abs() < 0.5 {
        argument.exp_m1()
    } else {
        exponential - 1.
    };
    (difference, slope * derivative)
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
) {
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
    if let Some((current, slope)) = source.evaluate(model.dioMod, bias.source, ckt.gmin()) {
        here.cbs = current;
        here.gbs = slope;
    }
    if let Some((current, slope)) = drain.evaluate(model.dioMod, bias.drain, ckt.gmin()) {
        here.cbd = current;
        here.gbd = slope;
    }
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
    for (voltage, terms, current, slope) in [
        (bias.source, source_tunnels, &mut here.cbs, &mut here.gbs),
        (bias.drain, drain_tunnels, &mut here.cbd, &mut here.gbd),
    ] {
        let [
            (area, area_slope),
            (sidewall, sidewall_slope),
            (gate, gate_slope),
        ] = terms.map(|(density, emission, threshold)| {
            let (difference, derivative) =
                reverse_tunneling(voltage, model.vtm0 * emission, threshold);
            (density * difference, density * derivative)
        });
        *current -= area + sidewall + gate;
        *slope += area_slope + sidewall_slope + gate_slope;
    }
}

#[derive(Clone, Copy)]
struct JunctionCapacitance {
    zero: f64,
    potential: f64,
    grading: f64,
}

/// The reverse integral shares the diode/BJT depletion law. The forward
/// BSIM4 continuation is its quadratic Taylor polynomial at zero bias.
/// State writes remain in the reference bottom/sidewall/gate order.
fn depletion(
    voltage: f64,
    parts: [JunctionCapacitance; 3],
    slot: usize,
    ckt: &mut dyn LoadCkt,
) -> f64 {
    let [bottom, side, gate] = parts;
    if voltage == 0. {
        ckt.set_state0(slot, 0.);
        bottom.zero + side.zero + gate.zero
    } else if voltage < 0. {
        let mut capacitance = 0.;
        for (index, part) in parts.into_iter().enumerate() {
            if part.zero > 0. {
                let (charge, cap) =
                    reverse_depletion(voltage, part.zero, part.potential, part.grading);
                if index == 0 {
                    ckt.set_state0(slot, charge);
                    capacitance = cap;
                } else {
                    ckt.set_state0(slot, ckt.state0(slot) + charge);
                    capacitance += cap;
                }
            } else if index == 0 {
                ckt.set_state0(slot, 0.);
            }
        }
        capacitance
    } else {
        let cap0 = bottom.zero + side.zero + gate.zero;
        let increment = voltage
            * (bottom.zero * bottom.grading / bottom.potential
                + side.zero * side.grading / side.potential
                + gate.zero * gate.grading / gate.potential);
        ckt.set_state0(slot, voltage * 0.5_f64.mul_add(increment, cap0));
        cap0 + increment
    }
}

/// Evaluate depletion charge and capacitance (C's finished label).
pub(super) fn charge(
    model: &Model,
    here: &mut Instance,
    size: &SizeParam,
    ckt: &mut dyn LoadCkt,
    bias: JunctionBias,
) {
    here.capbs = depletion(
        bias.source,
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
        state::QBS,
        ckt,
    );
    here.capbd = depletion(
        bias.drain,
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
        state::QBD,
        ckt,
    );
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
                let (current, slope) = diode.evaluate(selector, voltage, 0.).unwrap();
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
        let (current, slope) = diode.evaluate(2, 0., 0.).unwrap();
        assert_eq!(current.to_bits(), 0_f64.to_bits());
        assert!((slope / (diode.saturation / diode.thermal) - 1.).abs() < f64::EPSILON);
        let (current, _) = diode.evaluate(0, 0., 0.).unwrap();
        // Mode zero retains the C zero-bias floor but uses the unclipped
        // reverse exponential, unlike mode two.
        assert!(current > 0.);
        let (current, slope) = diode.evaluate(1, -36. * diode.thermal, 0.).unwrap();
        assert_eq!(current, diode.saturation * (MIN_EXP - 1.));
        assert_eq!(slope, 0.);
        diode.forward_voltage = 1e-18;
        diode.forward_current = diode.saturation;
        let (current, _) = diode.evaluate(1, 2e-18, 0.).unwrap();
        assert!(
            (current / (2e-18 * diode.saturation / diode.thermal) - 1.).abs() < 4. * f64::EPSILON
        );
    }

    #[test]
    fn reverse_tunneling_retains_tiny_current_and_large_reverse_derivative() {
        let (difference, derivative) = reverse_tunneling(-1e-18, 0.1, 10.);
        assert!((difference / 1e-17 - 1.).abs() < 4. * f64::EPSILON);
        assert!((derivative / 10. - 1.).abs() < 4. * f64::EPSILON);
        let (_, derivative) = reverse_tunneling(-1e18, 0.1, 10.);
        assert!((derivative / (super::super::MAX_EXP * 1e-33) - 1.).abs() < 4. * f64::EPSILON);
    }
}
