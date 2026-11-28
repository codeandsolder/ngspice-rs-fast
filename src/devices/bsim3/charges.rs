//! BSIM3v3.3 charge (C-V) model, a port of `b3ld.c`'s charge block.
//!
//! The supported charge path is BSIM3 `capMod == 3`.
//!
//! This ports the `capMod == 3` path (C's default, the "charge thickness"
//! model): the C-V preamble (`VbseffCV`, `CoxWL`, the `noff`/`voffcv`
//! `Vgsteff`), the CTM charge/capacitance block, bias-dependent overlap
//! capacitances for `capMod >= 2`, and source/drain junction C-V. `capMod` 0,
//! 1 and 2 and every NQS mode are explicit refusals: each is a different
//! equation set, not a tuning of this one.
//!
//! Every output keeps C's sign convention: charges on the gate, bulk and
//! drain and the nine `c*xb` capacitance derivatives, before the multiplicity
//! multiplier and before the overlap-charge adjustment the caller performs.
#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols keep their original distinct names for source-level audit"
)]
#![expect(
    clippy::too_many_lines,
    reason = "a complete translated C numerical routine remains contiguous for equation parity review"
)]

use super::dc::{Bias, DcEvaluation};
use super::defaults::ModelDefaults;
use super::setup::InstanceSetup;
use super::temperature::TemperatureModel;
use crate::devices::bsim_physics::{
    ChargeChain, JunctionCapacitance, channel_inversion, charge_overdrive, depletion,
    depletion_charge, overlap, smooth_drain, smooth_positive, thermal_softplus,
};
use crate::maths::taylor::Constitutive;
use crate::primitives::{Real, SpiceError, SpiceResult};
use std::cmp::Ordering;

const EXP_THRESHOLD: Real = 34.0;
const MIN_EXP: Real = 1.713_908_431e-15;
const MAX_EXP: Real = 5.834_617_425e14;
/// `b3ld.c` `EPSSI` (rounded, not the platform's permittivity).
const EPS_SI: Real = 1.035_94e-10;
/// `b3ld.c` `DELTA_3`.
const DELTA_3: Real = 0.02;
/// `b3ld.c` `DELTA_4`.
const DELTA_4: Real = 0.02;

/// Intrinsic charges, the nine capacitance derivatives and the junction C-V
/// of one `b3ld.c` charge evaluation at a fixed bias.
///
/// Charges are C's instance values before the multiplicity multiplier and
/// before the overlap-charge (`qgdo`/`qgso`/`cgbo`) adjustment: `qgate`,
/// `qbulk` and `qdrn` are the CTM results, exactly as C has them when the
/// `capMod == 3` block ends. `qbs`/`qbd` are C's `CKTstate0` junction charges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChargeEvaluation<S = Real> {
    /// Intrinsic gate charge (C `qgate` at the end of the CTM block).
    pub qgate: S,
    /// Intrinsic bulk charge (C `qbulk`).
    pub qbulk: S,
    /// Intrinsic drain charge (C `qdrn`).
    pub qdrn: S,
    /// Intrinsic source charge (C `qsrc`, from the XPART partition).
    pub qsrc: S,
    /// C `here->BSIM3cggb`.
    pub cggb: Real,
    /// C `here->BSIM3cgdb`.
    pub cgdb: Real,
    /// C `here->BSIM3cgsb`.
    pub cgsb: Real,
    /// C `here->BSIM3cdgb`.
    pub cdgb: Real,
    /// C `here->BSIM3cddb`.
    pub cddb: Real,
    /// C `here->BSIM3cdsb`.
    pub cdsb: Real,
    /// C `here->BSIM3cbgb`.
    pub cbgb: Real,
    /// C `here->BSIM3cbdb`.
    pub cbdb: Real,
    /// C `here->BSIM3cbsb`.
    pub cbsb: Real,
    /// Bias-dependent gate-drain overlap capacitance (C `here->BSIM3cgdo`).
    pub cgdo: Real,
    /// Bias-dependent gate-source overlap capacitance (C `here->BSIM3cgso`).
    pub cgso: Real,
    /// Gate-bulk overlap capacitance (C `pParam->BSIM3cgbo`).
    pub cgbo: Real,
    /// Gate-drain overlap charge (C `qgdo`), for the gate/drain adjustment.
    pub qgdo: S,
    /// Gate-source overlap charge (C `qgso`).
    pub qgso: S,
    /// Source junction capacitance (C `here->BSIM3capbs`).
    pub capbs: Real,
    /// Drain junction capacitance (C `here->BSIM3capbd`).
    pub capbd: Real,
    /// Source junction charge (C `CKTstate0[here->BSIM3qbs]`).
    pub qbs: S,
    /// Drain junction charge (C `CKTstate0[here->BSIM3qbd]`).
    pub qbd: S,
    /// Inversion charge for the noise model (C `here->BSIM3qinv`).
    pub qinv: S,
}

impl ChargeEvaluation {
    /// C's overlap-charge adjustment and `qd`/`qb` derivation, in C's order.
    ///
    /// `mode` is the DC mode (>= 0 normal, < 0 inverse). Returns
    /// `CKTstate0`'s `qg`, `qd` and `qb` (C's `qgate`/`qdrn` are updated in
    /// place because the terminal currents read them).
    #[must_use = "the returned charges include the overlap contributions and terminal-mode redistribution"]
    pub fn with_overlap(&self, mode: i8, vgb: Real) -> OverlapCharges {
        self.with_overlap_constitutive(mode, vgb)
    }
}

impl<S> ChargeEvaluation<S> {
    pub(super) fn with_overlap_constitutive(&self, mode: i8, vgb: S) -> OverlapCharges<S>
    where
        S: Constitutive,
    {
        let qgb = vgb.scale(self.cgbo);
        let qgate = self.qgate.clone().add(&self.qgdo).add(&self.qgso).add(&qgb);
        let qbulk = self.qbulk.clone().add(&qgb.scale(-1.));
        let (qdrn, qsrc) = if mode >= 0 {
            let qdrn = self.qdrn.clone().add(&self.qgdo.clone().scale(-1.));
            let qsrc = qgate.clone().add(&qbulk).add(&qdrn).scale(-1.);
            (qdrn, qsrc)
        } else {
            let qsrc = self.qdrn.clone().add(&self.qgso.clone().scale(-1.));
            let qdrn = qgate.clone().add(&qbulk).add(&qsrc).scale(-1.);
            (qdrn, qsrc)
        };
        OverlapCharges {
            qg: qgate.clone(),
            qd: qdrn.clone().add(&self.qbd.clone().scale(-1.)),
            qb: qbulk.clone().add(&self.qbd).add(&self.qbs),
            qgate,
            qbulk,
            qdrn,
            qsrc,
        }
    }
}

/// C's `qgate`/`qbulk`/`qdrn`/`qsrc` after the overlap-charge adjustment and
/// the `CKTstate0` charge values `b3ask.c` reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlapCharges<S = Real> {
    /// Gate charge including overlaps and gate-bulk.
    pub qgate: S,
    /// Bulk charge including gate-bulk.
    pub qbulk: S,
    /// Drain charge including the gate-drain overlap.
    pub qdrn: S,
    /// Source charge (C solves it from the node-charge sum).
    pub qsrc: S,
    /// C `CKTstate0[here->BSIM3qg]`.
    pub qg: S,
    /// C `CKTstate0[here->BSIM3qd]`.
    pub qd: S,
    /// C `CKTstate0[here->BSIM3qb]`.
    pub qb: S,
}

/// The BSIM3 charge core for one defaulted model and one set-up instance.
#[derive(Debug, Clone, Copy)]
pub struct ChargeModel<'a> {
    /// The defaulted model (`b3set.c`).
    pub model: &'a ModelDefaults,
    /// The model-level temperature pass (`b3temp.c`).
    pub temperature: &'a TemperatureModel,
    /// This instance's effective parameters (`b3temp.c`).
    pub setup: &'a InstanceSetup,
}

impl ChargeModel<'_> {
    /// Evaluate the charge model at `bias`, reusing the DC evaluation's
    /// intermediates.
    ///
    /// # Errors
    /// A `capMod` other than 3 or any NQS mode (`NotYetPorted` naming the C
    /// file), or a nonfinite result.
    pub fn evaluate(&self, dc: &DcEvaluation, bias: Bias) -> SpiceResult<ChargeEvaluation> {
        self.evaluate_constitutive(dc, bias)
    }

    /// The complete CTM charge law shares the scalar and differential
    /// value path. C's scalar capacitance stamp uses the reference
    /// partial convention, while the differential values retain the
    /// derivatives of the actual physical charge.
    pub(super) fn evaluate_constitutive<S: Constitutive>(
        &self,
        dc: &DcEvaluation<S>,
        bias: Bias<S>,
    ) -> SpiceResult<ChargeEvaluation<S>> {
        let cap_mod = self.model.values.capmod.value;
        if cap_mod != 3 {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "BSIM3 capMod={cap_mod} charge model (only capMod=3 is ported; \
                     set capMod=3 or omit it)"
                ),
                "src/spicelib/devices/bsim3/b3ld.c (capMod 0/1/2 charge blocks)",
            ));
        }
        if self.setup.nqs_mod != 0 || self.setup.acnqs_mod != 0 {
            return Err(SpiceError::not_yet_ported(
                "BSIM3 NQS charge model (nqsMod/acnqsMod)",
                "src/spicelib/devices/bsim3/b3ld.c (NQS charge partition and qdef)",
            ));
        }
        let p = self.setup;
        let m = self.model;
        let t = self.temperature;
        let cv = &dc.cv;
        // C's junction C-V and overlap charges use the raw terminal bias;
        // the CTM block uses the mode-selected magnitudes (`Vgs`, `Vds`).
        let vbs = bias.vbs;
        let vgs = bias.vgs;
        let vds = bias.vds;
        let vds_mode = &dc.vds;

        // C-V preamble: effective body bias and thermal overdrive.
        let (vbseff_cv, dvbseff_cv_dvb) = if cv.vbseff.compare(0.) == Some(Ordering::Less) {
            (cv.vbseff.clone(), 1.)
        } else {
            (cv.phis.clone().scale(-1.).offset(cv.phi), -cv.d_phis_dvb)
        };
        let cox = m.oxide_capacitance;
        let area = p.weff_cv * p.leff_cv;
        let noff = cv.n.clone().scale(p.binned.noff);
        let dnoff_dvd = p.binned.noff * cv.d_n_dvd;
        let dnoff_dvb = p.binned.noff * cv.d_n_dvb;
        let (vgsteff, voltage_partial, factor_partial) =
            thermal_softplus(&cv.vgst.clone().offset(-p.binned.voffcv), &noff, cv.vtm)?;
        let voltage_partial = voltage_partial.primal();
        let factor_partial = factor_partial.primal();
        let dvgsteff_dvg = voltage_partial * cv.d_vgs_eff_dvg;
        let dvgsteff_dvd = factor_partial.mul_add(dnoff_dvd, -voltage_partial * cv.d_vth_dvd);
        let dvgsteff_dvb = factor_partial.mul_add(dnoff_dvb, -voltage_partial * cv.d_vth_dvb);

        // Flatband and accumulation centroid.
        let vfbzb = p.vfbzb;
        let v3 = vbseff_cv
            .clone()
            .add(&cv.vgs_eff.clone().scale(-1.))
            .offset(vfbzb - DELTA_3);
        let (flatband_drop, slope) = smooth_positive(v3, DELTA_3 * vfbzb.abs())?;
        let vfbeff = flatband_drop.clone().scale(-1.).offset(vfbzb);
        let dvfbeff_dvg = slope.primal() * cv.d_vgs_eff_dvg;
        let dvfbeff_dvb = -slope.primal() * dvbseff_cv_dvb;

        let tox = m.values.tox.value;
        let tox_cm = 1e8 * tox;
        let argument = cv
            .vgs_eff
            .clone()
            .add(&vbseff_cv.clone().scale(-1.))
            .offset(-vfbzb)
            .scale(p.binned.acde / tox_cm);
        let ldeb = p.derived.ldeb;
        let (centroid, centroid_partial) =
            if argument.compare(-EXP_THRESHOLD) != Some(Ordering::Greater) {
                (argument.constant(ldeb * MIN_EXP), 0.)
            } else if argument.compare(EXP_THRESHOLD) != Some(Ordering::Less) {
                (argument.constant(ldeb * MAX_EXP), 0.)
            } else {
                let value = argument.exp().scale(ldeb);
                let partial = value.primal() * p.binned.acde / tox_cm;
                (value, partial)
            };
        let centroid = smooth_drain(vds.constant(ldeb), centroid, 1e-3 * tox)?;
        let tcen = centroid.effective;
        let centroid_partial = centroid_partial * centroid.drain_partial.primal();
        let dtcen_dvg = centroid_partial * cv.d_vgs_eff_dvg;
        let dtcen_dvb = -centroid_partial * dvbseff_cv_dvb;
        let coxeff = effective_oxide(&tcen, cox)?;
        let oxide_partial = -coxeff.primal() * coxeff.primal() / EPS_SI;
        let dcoxeff_dvg = oxide_partial * dtcen_dvg;
        let dcoxeff_dvb = oxide_partial * dtcen_dvb;
        let cox_wl_cen = coxeff.clone().scale(area);
        let qac0 = cox_wl_cen.clone().scale(-1.).multiply(&flatband_drop);
        let qov_cox = qac0.primal() / coxeff.primal();
        let dqac0_dvg = qov_cox.mul_add(dcoxeff_dvg, cox_wl_cen.primal() * dvfbeff_dvg);
        let dqac0_dvb = qov_cox.mul_add(dcoxeff_dvb, cox_wl_cen.primal() * dvfbeff_dvb);

        // Poly-gate depletion share of bulk charge.
        let k1ox = p.derived.k1ox;
        let depletion_argument = cv
            .vgs_eff
            .clone()
            .add(&vfbeff.scale(-1.))
            .add(&vbseff_cv.scale(-1.))
            .add(&vgsteff.clone().scale(-1.));
        let (charge, slope) = depletion_charge(k1ox, depletion_argument)?;
        let qsub0 = cox_wl_cen.clone().multiply(&charge);
        let t2 = cox_wl_cen.primal() * slope.primal();
        let qov_cox = qsub0.primal() / coxeff.primal();
        let dqsub0_dvg = qov_cox.mul_add(
            dcoxeff_dvg,
            t2 * (cv.d_vgs_eff_dvg - dvfbeff_dvg - dvgsteff_dvg),
        );
        let dqsub0_dvd = -t2 * dvgsteff_dvd;
        let dqsub0_dvb = qov_cox.mul_add(
            dcoxeff_dvb,
            -t2 * (dvfbeff_dvb + dvbseff_cv_dvb + dvgsteff_dvb),
        );

        let (vgdp, dvgdp_dvg) = charge_overdrive(
            vgsteff.clone(),
            k1ox,
            p.binned.moin,
            cv.vtm,
            p.derived.sqrtphi,
        )?;
        let dvgdp_dvg = dvgdp_dvg.primal();

        // Bias-dependent channel thickness for inversion charge.
        let threshold_offset = cv.vth.clone().offset(-vfbzb - cv.phi).scale(4.);
        let tox_cm = 2. * tox_cm;
        let (field, dfield_dvd, dfield_dvb) =
            if threshold_offset.compare(0.) != Some(Ordering::Less) {
                (
                    vgsteff.clone().add(&threshold_offset).scale(tox_cm.recip()),
                    4.0_f64.mul_add(cv.d_vth_dvd, dvgsteff_dvd) / tox_cm,
                    4.0_f64.mul_add(cv.d_vth_dvb, dvgsteff_dvb) / tox_cm,
                )
            } else {
                (
                    vgsteff.offset(1e-20).scale(tox_cm.recip()),
                    dvgsteff_dvd / tox_cm,
                    dvgsteff_dvb / tox_cm,
                )
            };
        let power = field.clone().ln()?.scale(0.7).exp();
        let divisor = power.clone().offset(1.);
        let tcen = field.constant(1.9e-9).divide(&divisor)?;
        let centroid_partial =
            -tcen.primal() * 0.7 * power.primal() / (field.primal() * divisor.primal());
        let coxeff = effective_oxide(&tcen, cox)?;
        let oxide_partial = -coxeff.primal() * coxeff.primal() / EPS_SI * centroid_partial;
        let dcoxeff_dvg = oxide_partial * dvgsteff_dvg / tox_cm;
        let dcoxeff_dvd = oxide_partial * dfield_dvd;
        let dcoxeff_dvb = oxide_partial * dfield_dvb;
        let cox_wl_cen = coxeff.clone().scale(area);

        // Effective drain voltage for C-V.
        let abulk_cv = cv.abulk0.clone().scale(p.thermal.abulkcvfactor);
        let dabulk_cv_dvb = p.thermal.abulkcvfactor * cv.d_abulk0_dvb;
        let vdsat_cv = vgdp.clone().divide(&abulk_cv)?;
        let smoothing = smooth_drain(vdsat_cv.clone(), vds_mode.clone(), DELTA_4)?;
        let vdseff_cv = smoothing.effective;
        let dvdseff_cv_dvg = smoothing.saturation_partial.primal() * dvgdp_dvg / abulk_cv.primal();
        let dvdseff_cv_dvd = smoothing.drain_partial.primal();
        // Preserve the BSIM3 scalar reference capacitance convention at
        // the negative-root branch. The charge law differentiates its
        // actual effective drain voltage instead.
        let saturation_factor = if vdsat_cv.primal() - vds_mode.primal() - DELTA_4 < 0.
            && vds_mode.primal().abs() > 0.
        {
            smoothing
                .saturation_partial
                .primal()
                .mul_add(vdsat_cv.primal(), smoothing.saturation_excess.primal())
        } else {
            smoothing.saturation_partial.primal() * vdsat_cv.primal()
        };
        let dvdseff_cv_dvb = -saturation_factor * dabulk_cv_dvb / abulk_cv.primal();

        let [inversion, bulk, source] = channel_inversion(
            &vgdp,
            &vdseff_cv,
            &abulk_cv,
            &cox_wl_cen,
            m.values.xpart.value,
            1e-20,
        )?;
        let chain = ChargeChain {
            overdrive: [dvgsteff_dvg, dvgsteff_dvd, dvgsteff_dvb].map(|slope| dvgdp_dvg * slope),
            drain: [
                dvdseff_cv_dvg * dvgsteff_dvg,
                dvdseff_cv_dvg.mul_add(dvgsteff_dvd, dvdseff_cv_dvd),
                dvdseff_cv_dvg.mul_add(dvgsteff_dvb, dvdseff_cv_dvb),
            ],
            bulk: [0., 0., dabulk_cv_dvb],
            relative_oxide: [dcoxeff_dvg, dcoxeff_dvd, dcoxeff_dvb]
                .map(|slope| slope / coxeff.primal()),
        };
        let [cgg1, cgd1, cgb1] = chain.apply(&inversion);
        let [cbg1, cbd1, cbb1] = chain.apply(&bulk);
        let [csg, csd, mut csb] = chain.apply(&source);
        let qgate = inversion.value;
        let qinoi = qgate.clone();
        let qbulk = bulk.value;
        let qsrc = source.value;

        // Fold the accumulation/subthreshold charges into gate and bulk.
        let accumulation = qac0.add(&qsub0);
        let qgate = qgate.add(&accumulation).add(&qbulk.clone().scale(-1.));
        let qbulk = qbulk.add(&accumulation.scale(-1.));
        let qdrn = qgate.clone().add(&qbulk).add(&qsrc).scale(-1.);

        let cbg = cbg1 - dqac0_dvg - dqsub0_dvg;
        let cbd = cbd1 - dqsub0_dvd;
        let mut cbb = cbb1 - dqac0_dvb - dqsub0_dvb;

        let cgg = cgg1 - cbg;
        let cgd = cgd1 - cbd;
        let mut cgb = cgb1 - cbb;
        cgb *= cv.d_vbseff_dvb;
        cbb *= cv.d_vbseff_dvb;
        csb *= cv.d_vbseff_dvb;

        let cggb = cgg;
        let cgsb = -(cgg + cgd + cgb);
        let cgdb = cgd;
        let cdgb = -(cgg + cbg + csg);
        let cdsb = cgg + cgd + cgb + cbg + cbd + cbb + csg + csd + csb;
        let cddb = -(cgd + cbd + csd);
        let cbgb = cbg;
        let cbsb = -(cbg + cbd + cbb);
        let cbdb = cbd;

        // ---- Overlap capacitances (capMod 3 uses the capMod >= 2 branch).
        let cgdo_param = p.cgdo;
        let cgso_param = p.cgso;
        let ck = p.binned.ckappa;
        let vgd = vgs.clone().add(&vds.clone().scale(-1.));

        let (cgdo, qgdo) = overlap(vgd, cgdo_param, p.weff_cv * p.binned.cgdl, ck)?;
        let (cgso, qgso) = overlap(vgs, cgso_param, p.weff_cv * p.binned.cgsl, ck)?;

        let cgbo = p.derived.cgbo;

        // ---- Junction C-V (b3ld.c "bulk-drain and bulk-source depletion
        // capacitances"). C uses the un-temperature-scaled gate-sidewall field
        // (`BSIM3unitLengthGateSidewallJctCap`, not `...TempJctCap`); that
        // quirk is reproduced.
        let source = junction_parts(t, m, p.source_area, p.source_perimeter, p.weff);
        let drain = junction_parts(t, m, p.drain_area, p.drain_perimeter, p.weff);
        let (qbs, capbs) = depletion(&vbs, source, |_| {})?;
        let (qbd, capbd) = depletion(&vbs.add(&vds.scale(-1.)), drain, |_| {})?;

        let result = ChargeEvaluation {
            qgate,
            qbulk,
            qdrn,
            qsrc,
            cggb,
            cgdb,
            cgsb,
            cdgb,
            cddb,
            cdsb,
            cbgb,
            cbdb,
            cbsb,
            cgdo: cgdo.primal(),
            cgso: cgso.primal(),
            cgbo,
            qgdo,
            qgso,
            capbs: capbs.primal(),
            capbd: capbd.primal(),
            qbs,
            qbd,
            qinv: qinoi.scale(-1.),
        };
        for (name, value) in [
            ("qgate", result.qgate.primal()),
            ("qbulk", result.qbulk.primal()),
            ("qdrn", result.qdrn.primal()),
            ("qsrc", result.qsrc.primal()),
            ("qbs", result.qbs.primal()),
            ("qbd", result.qbd.primal()),
            ("capbs", result.capbs),
            ("capbd", result.capbd),
            ("cggb", result.cggb),
            ("cgdb", result.cgdb),
            ("cgsb", result.cgsb),
            ("cdgb", result.cdgb),
            ("cddb", result.cddb),
            ("cdsb", result.cdsb),
            ("cbgb", result.cbgb),
            ("cbdb", result.cbdb),
            ("cbsb", result.cbsb),
            ("cgdo", result.cgdo),
            ("cgso", result.cgso),
        ] {
            if !value.is_finite() {
                return Err(SpiceError::Numerical {
                    context: "BSIM3 charge evaluation (bsim3/b3ld.c)".into(),
                    message: format!("{name} is not finite"),
                });
            }
        }
        Ok(result)
    }
}

/// Series oxide/centroid capacitance in the nonsingular form, including
/// the zero-centroid limit.
fn effective_oxide<S: Constitutive>(centroid: &S, oxide: Real) -> SpiceResult<S> {
    centroid
        .constant(oxide * EPS_SI)
        .divide(&centroid.clone().scale(oxide).offset(EPS_SI))
}

/// `b3ld.c`'s zero-bias junction capacitances for one junction, split into the
/// bottom, sidewall and gate-sidewall parts.
fn junction_parts(
    temperature: &TemperatureModel,
    model: &ModelDefaults,
    area: Real,
    perimeter: Real,
    weff: Real,
) -> [JunctionCapacitance; 3] {
    let bottom = temperature.unit_area_temp_jct_cap * area;
    let sidewall = temperature.unit_length_sidewall_temp_jct_cap;
    // C reads the *model* field here (no CJSWG temperature scaling).
    let gate_sidewall = model.values.cjswg.value;
    let (side, gate) = if perimeter < weff {
        (0., gate_sidewall * perimeter)
    } else {
        (sidewall * (perimeter - weff), gate_sidewall * weff)
    };
    [
        JunctionCapacitance {
            zero: bottom,
            potential: temperature.phi_b,
            grading: model.values.mj.value,
        },
        JunctionCapacitance {
            zero: side,
            potential: temperature.phi_bsw,
            grading: model.values.mjsw.value,
        },
        JunctionCapacitance {
            zero: gate,
            potential: temperature.phi_bswg,
            grading: model.values.mjswg.value,
        },
    ]
}
