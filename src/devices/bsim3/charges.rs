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
    depletion_charge, overlap, smooth_drain, smooth_positive,
};
use crate::primitives::{Real, SpiceError, SpiceResult};

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
pub struct ChargeEvaluation {
    /// Intrinsic gate charge (C `qgate` at the end of the CTM block).
    pub qgate: Real,
    /// Intrinsic bulk charge (C `qbulk`).
    pub qbulk: Real,
    /// Intrinsic drain charge (C `qdrn`).
    pub qdrn: Real,
    /// Intrinsic source charge (C `qsrc`, from the XPART partition).
    pub qsrc: Real,
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
    pub qgdo: Real,
    /// Gate-source overlap charge (C `qgso`).
    pub qgso: Real,
    /// Source junction capacitance (C `here->BSIM3capbs`).
    pub capbs: Real,
    /// Drain junction capacitance (C `here->BSIM3capbd`).
    pub capbd: Real,
    /// Source junction charge (C `CKTstate0[here->BSIM3qbs]`).
    pub qbs: Real,
    /// Drain junction charge (C `CKTstate0[here->BSIM3qbd]`).
    pub qbd: Real,
    /// Inversion charge for the noise model (C `here->BSIM3qinv`).
    pub qinv: Real,
}

impl ChargeEvaluation {
    /// C's overlap-charge adjustment and `qd`/`qb` derivation, in C's order.
    ///
    /// `mode` is the DC mode (>= 0 normal, < 0 inverse). Returns
    /// `CKTstate0`'s `qg`, `qd` and `qb` (C's `qgate`/`qdrn` are updated in
    /// place because the terminal currents read them).
    #[must_use = "the returned charges include the overlap contributions and terminal-mode redistribution"]
    pub fn with_overlap(&self, mode: i8, vgb: Real) -> OverlapCharges {
        let qgd = self.qgdo;
        let qgs = self.qgso;
        let qgb = self.cgbo * vgb;
        let qgate = self.qgate + qgd + qgs + qgb;
        let qbulk = self.qbulk - qgb;
        let (qdrn, qsrc) = if mode >= 0 {
            let qdrn = self.qdrn - qgd;
            (qdrn, -(qgate + qbulk + qdrn))
        } else {
            let qsrc = self.qdrn - qgs;
            (-(qgate + qbulk + qsrc), qsrc)
        };
        OverlapCharges {
            qgate,
            qbulk,
            qdrn,
            qsrc,
            // C stores the state after the adjustment; the junction charges
            // are already on the bulk/drain nodes.
            qg: qgate,
            qd: qdrn - self.qbd,
            qb: qbulk + self.qbd + self.qbs,
        }
    }
}

/// C's `qgate`/`qbulk`/`qdrn`/`qsrc` after the overlap-charge adjustment and
/// the `CKTstate0` charge values `b3ask.c` reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlapCharges {
    /// Gate charge including overlaps and gate-bulk.
    pub qgate: Real,
    /// Bulk charge including gate-bulk.
    pub qbulk: Real,
    /// Drain charge including the gate-drain overlap.
    pub qdrn: Real,
    /// Source charge (C solves it from the node-charge sum).
    pub qsrc: Real,
    /// C `CKTstate0[here->BSIM3qg]`.
    pub qg: Real,
    /// C `CKTstate0[here->BSIM3qd]`.
    pub qd: Real,
    /// C `CKTstate0[here->BSIM3qb]`.
    pub qb: Real,
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
        let vds_mode = dc.vds;

        // ---- b3ld.c C-V preamble: VbseffCV, CoxWL and the noff VgsteffCV.
        let (vbseff_cv, dvbseff_cv_dvb) = if cv.vbseff < 0.0 {
            (cv.vbseff, 1.0)
        } else {
            // C `VbseffCV = phi - Phis`.
            (cv.phi - cv.phis, -cv.d_phis_dvb)
        };
        let cox = m.oxide_capacitance;
        let cox_wl = cox * p.weff_cv * p.leff_cv;

        let noff = cv.n * p.binned.noff;
        let dnoff_dvd = p.binned.noff * cv.d_n_dvd;
        let dnoff_dvb = p.binned.noff * cv.d_n_dvb;
        let t0 = cv.vtm * noff;
        let voffcv = p.binned.voffcv;
        let vgst_nvt = (cv.vgst - voffcv) / t0;
        let (vgsteff, dvgsteff_dvg, dvgsteff_dvd, dvgsteff_dvb) = if vgst_nvt > EXP_THRESHOLD {
            (
                cv.vgst - voffcv,
                cv.d_vgs_eff_dvg,
                -cv.d_vth_dvd,
                -cv.d_vth_dvb,
            )
        } else if vgst_nvt < -EXP_THRESHOLD {
            let vgsteff = t0 * MIN_EXP.ln_1p();
            let dvd = vgsteff / noff;
            (vgsteff, 0.0, dvd * dnoff_dvd, dvd * dnoff_dvb)
        } else {
            let exp_vgst = vgst_nvt.exp();
            let vgsteff = t0 * exp_vgst.ln_1p();
            let dvg = exp_vgst / (1.0 + exp_vgst);
            let dvd = (vgsteff / noff).mul_add(
                dnoff_dvd,
                -dvg * ((cv.vgst - voffcv) / noff).mul_add(dnoff_dvd, cv.d_vth_dvd),
            );
            let dvb = (vgsteff / noff).mul_add(
                dnoff_dvb,
                -dvg * ((cv.vgst - voffcv) / noff).mul_add(dnoff_dvb, cv.d_vth_dvb),
            );
            (vgsteff, dvg * cv.d_vgs_eff_dvg, dvd, dvb)
        };

        // ---- capMod == 3, the charge-thickness model (CTM).
        let vfbzb = p.vfbzb;
        let v3 = vfbzb - cv.vgs_eff + vbseff_cv - DELTA_3;
        let (flatband_drop, t1) = smooth_positive(v3, DELTA_3 * vfbzb.abs())?;
        let vfbeff = vfbzb - flatband_drop;
        let dvfbeff_dvg = t1 * cv.d_vgs_eff_dvg;
        let dvfbeff_dvb = -t1 * dvbseff_cv_dvb;

        let tox = m.values.tox.value;
        // C `Tox = 1.0e8 * model->BSIM3tox` (centimetres for the ACDE exponent).
        let mut tox_cm = 1.0e8 * tox;
        let t0 = (cv.vgs_eff - vbseff_cv - vfbzb) / tox_cm;
        let dt0_dvg = cv.d_vgs_eff_dvg / tox_cm;
        let dt0_dvb = -dvbseff_cv_dvb / tox_cm;

        let acde = p.binned.acde;
        let ldeb = p.derived.ldeb;
        let tmp = t0 * acde;
        let (tcen, mut dtcen_dvg, mut dtcen_dvb) = if tmp > -EXP_THRESHOLD && tmp < EXP_THRESHOLD {
            let tcen = ldeb * tmp.exp();
            let mut dvg = acde * tcen;
            let dvb = dvg * dt0_dvb;
            dvg *= dt0_dvg;
            (tcen, dvg, dvb)
        } else if tmp <= -EXP_THRESHOLD {
            (ldeb * MIN_EXP, 0.0, 0.0)
        } else {
            (ldeb * MAX_EXP, 0.0, 0.0)
        };

        let link = 1.0e-3 * tox;
        let centroid = smooth_drain(ldeb, tcen, link)?;
        let tcen = centroid.effective;
        dtcen_dvg *= centroid.drain_partial;
        dtcen_dvb *= centroid.drain_partial;

        let ccen = EPS_SI / tcen;
        let t2 = cox / (cox + ccen);
        let coxeff = t2 * ccen;
        let t3 = -ccen / tcen;
        let mut dcoxeff_dvg = t2 * t2 * t3;
        let dcoxeff_dvb = dcoxeff_dvg * dtcen_dvb;
        dcoxeff_dvg *= dtcen_dvg;
        let cox_wl_cen = cox_wl * coxeff / cox;

        let qac0 = -cox_wl_cen * flatband_drop;
        let qov_cox = qac0 / coxeff;
        let dqac0_dvg = qov_cox.mul_add(dcoxeff_dvg, cox_wl_cen * dvfbeff_dvg);
        let dqac0_dvb = qov_cox.mul_add(dcoxeff_dvb, cox_wl_cen * dvfbeff_dvb);

        // Poly-gate depletion share of the bulk charge.
        let k1ox = p.derived.k1ox;
        let t3 = cv.vgs_eff - vfbeff - vbseff_cv - vgsteff;
        let (charge, slope) = depletion_charge(k1ox, t3)?;
        let qsub0 = cox_wl_cen * charge;
        let t2 = cox_wl_cen * slope;
        let qov_cox = qsub0 / coxeff;
        let dqsub0_dvg = qov_cox.mul_add(
            dcoxeff_dvg,
            t2 * (cv.d_vgs_eff_dvg - dvfbeff_dvg - dvgsteff_dvg),
        );
        let dqsub0_dvd = -t2 * dvgsteff_dvd;
        let dqsub0_dvb = qov_cox.mul_add(
            dcoxeff_dvb,
            -t2 * (dvfbeff_dvb + dvbseff_cv_dvb + dvgsteff_dvb),
        );

        let (vgdp, dvgdp_dvg) =
            charge_overdrive(vgsteff, k1ox, p.binned.moin, cv.vtm, p.derived.sqrtphi)?;

        // Bias-dependent channel thickness for the inversion charge.
        let t3 = 4.0 * (cv.vth - vfbzb - cv.phi);
        tox_cm += tox_cm;
        let (t0, dt0_dvd, dt0_dvb) = if t3 >= 0.0 {
            (
                (vgsteff + t3) / tox_cm,
                4.0f64.mul_add(cv.d_vth_dvd, dvgsteff_dvd) / tox_cm,
                4.0f64.mul_add(cv.d_vth_dvb, dvgsteff_dvb) / tox_cm,
            )
        } else {
            (
                (vgsteff + 1.0e-20) / tox_cm,
                dvgsteff_dvd / tox_cm,
                dvgsteff_dvb / tox_cm,
            )
        };
        let tmp = (0.7 * t0.ln()).exp();
        let t1 = 1.0 + tmp;
        let t2 = 0.7 * tmp / (t0 * tox_cm);
        let tcen = 1.9e-9 / t1;
        let mut dtcen_dvg2 = -1.9e-9 * t2 / t1 / t1;
        let mut dtcen_dvd = tox_cm * dtcen_dvg2;
        let dtcen_dvb2 = dtcen_dvd * dt0_dvb;
        dtcen_dvd *= dt0_dvd;
        dtcen_dvg2 *= dvgsteff_dvg;

        let ccen = EPS_SI / tcen;
        let t0 = cox / (cox + ccen);
        let coxeff = t0 * ccen;
        let t1 = -ccen / tcen;
        let mut dcoxeff_dvg = t0 * t0 * t1;
        let dcoxeff_dvd = dcoxeff_dvg * dtcen_dvd;
        let dcoxeff_dvb = dcoxeff_dvg * dtcen_dvb2;
        dcoxeff_dvg *= dtcen_dvg2;
        let cox_wl_cen = cox_wl * coxeff / cox;

        // Effective drain voltage for C-V.
        let abulk_cv = cv.abulk0 * p.thermal.abulkcvfactor;
        let dabulk_cv_dvb = p.thermal.abulkcvfactor * cv.d_abulk0_dvb;
        let vdsat_cv = vgdp / abulk_cv;
        let smoothing = smooth_drain(vdsat_cv, vds_mode, DELTA_4)?;
        let vdseff_cv = smoothing.effective;
        let dvdseff_cv_dvg = smoothing.saturation_partial * dvgdp_dvg / abulk_cv;
        let dvdseff_cv_dvd = smoothing.drain_partial;
        // The BSIM3 capMod=3 scalar load uses (1 - T5) instead of
        // (T4 - T5) in the body derivative of the negative-root branch.
        // Its published capacitance/stamp convention therefore differs
        // from the exact derivative of its charge by the saturation gap.
        // Keep this discrepancy at the scalar load boundary; the shared
        // smoothing law itself returns the actual physical partials.
        let saturation_factor = if vdsat_cv - vds_mode - DELTA_4 < 0. && vds_mode.abs() > 0. {
            smoothing
                .saturation_partial
                .mul_add(vdsat_cv, smoothing.saturation_excess)
        } else {
            smoothing.saturation_partial * vdsat_cv
        };
        let dvdseff_cv_dvb = -saturation_factor * dabulk_cv_dvb / abulk_cv;

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
            relative_oxide: [dcoxeff_dvg, dcoxeff_dvd, dcoxeff_dvb].map(|slope| slope / coxeff),
        };
        let [cgg1, cgd1, cgb1] = chain.apply(&inversion);
        let [cbg1, cbd1, cbb1] = chain.apply(&bulk);
        let [csg, csd, mut csb] = chain.apply(&source);
        let qgate = inversion.value;
        let qinoi = qgate;
        let qbulk = bulk.value;
        let qsrc = source.value;

        // Fold the accumulation/subthreshold charges into gate and bulk.
        let qgate = qgate + qac0 + qsub0 - qbulk;
        let qbulk = qbulk - (qac0 + qsub0);
        let qdrn = -(qgate + qbulk + qsrc);

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
        let vgd = vgs - vds;

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
        let (qbd, capbd) = depletion(&(vbs - vds), drain, |_| {})?;

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
            cgdo,
            cgso,
            cgbo,
            qgdo,
            qgso,
            capbs,
            capbd,
            qbs,
            qbd,
            qinv: -qinoi,
        };
        for (name, value) in [
            ("qgate", result.qgate),
            ("qbulk", result.qbulk),
            ("qdrn", result.qdrn),
            ("qsrc", result.qsrc),
            ("qbs", result.qbs),
            ("qbd", result.qbd),
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
