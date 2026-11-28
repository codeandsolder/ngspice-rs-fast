//! BSIM3v3.3 DC evaluation, a port of the bias-dependent part of `b3ld.c`.
//!
//! This is the physics core only: it takes terminal voltages and returns the
//! DC currents, conductances and observables C stores on the instance. It does
//! not limit voltages (`DEVfetlim`, `DEVpnjlim`, `DEVlimvds`), bypass points or
//! assemble matrix stamps — those belong to the solver loop and the device
//! implementation. Charges (`capMod`), NQS, AC and noise are not evaluated.
#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols keep their original distinct names for source-level audit"
)]
#![expect(
    clippy::float_cmp,
    reason = "C device branches compare floating-point values exactly, including nonfinite cases"
)]
#![expect(
    clippy::too_many_lines,
    reason = "a complete translated C numerical routine remains contiguous for equation parity review"
)]

use super::defaults::ModelDefaults;
use super::setup::InstanceSetup;
use super::temperature::TemperatureModel;
use crate::devices::bsim_physics::{
    SaturationRoot, bsim3_junction, poly_depletion, rational_floor, reciprocal_floor, smooth_drain,
};
use crate::primitives::{Real, SpiceResult};

const EXP_THRESHOLD: Real = 34.0;
const MIN_EXP: Real = 1.713_908_431e-15;
const MAX_EXP: Real = 5.834_617_425e14;
const EPS_SI: Real = 1.035_94e-10;
const CHARGE_Q: Real = 1.602_19e-19;

/// Terminal bias applied to the intrinsic device (the `dNodePrime`/
/// `sNodePrime` reference C uses), already multiplied by the model polarity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bias {
    /// Body-source voltage.
    pub vbs: Real,
    /// Gate-source voltage.
    pub vgs: Real,
    /// Drain-source voltage.
    pub vds: Real,
}

/// Bias-dependent intermediates `b3ld.c` computes before the charge block and
/// the charge preamble (`b3acld.c`/`b3ld.c` C-V) reuses.
///
/// They are carried out of the DC evaluation rather than recomputed so the
/// charge model sees exactly the values (and rounding) C's single pass used.
/// Derivatives are with respect to the polarity-scaled device bias.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CvIntermediates {
    /// Thermal voltage (C `model->BSIM3vtm`).
    pub vtm: Real,
    /// Oxide capacitance per area (C `model->BSIM3cox`).
    pub cox: Real,
    /// Oxide thickness in metres (C `model->BSIM3tox`).
    pub tox: Real,
    /// Surface potential parameter (C `pParam->BSIM3phi`).
    pub phi: Real,
    /// Effective gate-source voltage after poly depletion.
    pub vgs_eff: Real,
    /// Its gate derivative.
    pub d_vgs_eff_dvg: Real,
    /// Threshold voltage.
    pub vth: Real,
    /// Its drain derivative.
    pub d_vth_dvd: Real,
    /// Its body derivative.
    pub d_vth_dvb: Real,
    /// `vgs_eff - vth`, before the subthreshold smoothing.
    pub vgst: Real,
    /// Smoothed body-source voltage.
    pub vbseff: Real,
    /// Its body derivative.
    pub d_vbseff_dvb: Real,
    /// Surface potential at `vbseff` (C `Phis`).
    pub phis: Real,
    /// Its body derivative (C `dPhis_dVb`).
    pub d_phis_dvb: Real,
    /// Square root of the surface potential (C `sqrtPhis`).
    pub sqrt_phis: Real,
    /// C `dsqrtPhis_dVb`.
    pub d_sqrt_phis_dvb: Real,
    /// Bulk-charge factor before the mobility gate term.
    pub abulk0: Real,
    /// Its body derivative.
    pub d_abulk0_dvb: Real,
    /// Subthreshold slope factor.
    pub n: Real,
    /// Its drain derivative.
    pub d_n_dvd: Real,
    /// Its body derivative.
    pub d_n_dvb: Real,
    /// Smoothed gate overdrive used by the DC current.
    pub vgsteff: Real,
    /// Its gate derivative.
    pub d_vgsteff_dvg: Real,
    /// Its drain derivative.
    pub d_vgsteff_dvd: Real,
    /// Its body derivative.
    pub d_vgsteff_dvb: Real,
}

/// DC currents, conductances and observables from one `b3ld.c` evaluation.
///
/// Currents are the C instance values (`here->BSIM3cd`, `cbs`, ...) before the
/// multiplicity multiplier, which `*ask.c` applies separately.
#[derive(Debug, Clone, PartialEq)]
pub struct DcEvaluation {
    /// Drain current (C `here->BSIM3cd`).
    pub ids: Real,
    /// Transconductance (C `here->BSIM3gm`).
    pub gm: Real,
    /// Output conductance (C `here->BSIM3gds`).
    pub gds: Real,
    /// Body transconductance (C `here->BSIM3gmbs`).
    pub gmbs: Real,
    /// Substrate current (C `here->BSIM3csub`).
    pub isub: Real,
    /// Substrate-current derivatives (C `gbgs`, `gbds`, `gbbs`).
    pub gbgs: Real,
    /// Substrate-current drain derivative.
    pub gbds: Real,
    /// Substrate-current body derivative.
    pub gbbs: Real,
    /// Source junction current (C `here->BSIM3cbs`).
    pub cbs: Real,
    /// Drain junction current (C `here->BSIM3cbd`).
    pub cbd: Real,
    /// Source junction conductance, including gmin.
    pub gbs: Real,
    /// Drain junction conductance, including gmin.
    pub gbd: Real,
    /// Source-body voltage used, after mode selection.
    pub vbs: Real,
    /// Drain-body voltage computed as `vbs - vds`.
    pub vbd: Real,
    /// Gate-source voltage used, after mode selection.
    pub vgs: Real,
    /// Drain-source voltage used, after mode selection.
    pub vds: Real,
    /// Normal (+1) or inverse (-1) mode.
    pub mode: i8,
    /// Threshold voltage reported by `@m1[vth]` (C `here->BSIM3von`).
    pub von: Real,
    /// Effective gate overdrive.
    pub vgsteff: Real,
    /// Effective drain voltage.
    pub vdseff: Real,
    /// Saturation voltage.
    pub vdsat: Real,
    /// Bulk-charge factor.
    pub abulk: Real,
    /// Effective mobility.
    pub ueff: Real,
    /// Series resistance used.
    pub rds: Real,
    /// C `here->BSIM3thetavth`.
    pub thetavth: Real,
    /// C `here->BSIM3AbovVgst2Vtm`.
    pub abulk_over_vgst2vtm: Real,
    /// Intermediates the charge model (`b3ld.c` C-V) reads back.
    pub cv: CvIntermediates,
}

/// The BSIM3 DC core for one defaulted model and one set-up instance.
#[derive(Debug, Clone, Copy)]
pub struct DcModel<'a> {
    /// The defaulted model (`b3set.c`).
    pub model: &'a ModelDefaults,
    /// The model-level temperature pass (`b3temp.c`).
    pub temperature: &'a TemperatureModel,
    /// This instance's effective parameters (`b3temp.c`).
    pub setup: &'a InstanceSetup,
}

impl DcModel<'_> {
    /// Evaluate the DC currents and conductances at `bias`.
    ///
    /// `gmin` is C's `CKTgmin`, added to both junction conductances.
    ///
    /// # Errors
    /// A nonfinite intermediate that C would silently propagate into the
    /// matrix.
    pub fn evaluate(&self, bias: Bias, gmin: Real) -> SpiceResult<DcEvaluation> {
        let p = self.setup;
        let m = self.model;
        let t = self.temperature;
        let type_ = Real::from(p.polarity);
        let cox = m.oxide_capacitance;
        let tox = m.values.tox.value;
        let vtm = t.vtm;
        let temp = t.temp;
        let tnom = t.tnom;
        let factor1 = t.factor1;

        // Source/drain junction diode DC model begins.
        let nvtm = vtm * m.values.nj.value;
        let source_sat_current = p.source_sat_current;
        let drain_sat_current = p.drain_sat_current;
        let vbd_in = bias.vbs - bias.vds;

        let junction = |sat, v, vjsm, isevjsm| {
            let continuation = (m.values.ijth.value != 0.).then_some((vjsm, isevjsm));
            bsim3_junction(v, sat, nvtm, continuation, gmin)
        };
        let (cbs, gbs) = junction(source_sat_current, bias.vbs, p.vjsm, p.isevjsm)?;
        let (cbd, gbd) = junction(drain_sat_current, vbd_in, p.vjdm, p.isevjdm)?;

        // Normal versus inverse mode. The inverse case mirrors gate/drain.
        let (mode, vds, vgs, vbs) = if bias.vds >= 0.0 {
            (1, bias.vds, bias.vgs, bias.vbs)
        } else {
            (-1, -bias.vds, bias.vgs - bias.vds, bias.vbs - bias.vds)
        };

        // Body-effect and depletion charge.
        let vbsc = p.derived.vbsc;
        let phi = p.derived.phi;
        let t0 = vbs - vbsc - 0.001;
        let t1 = 0.004f64.mul_add(-vbsc, t0 * t0).sqrt();
        let vbseff_raw = vbsc + f64::midpoint(t0, t1);
        let dvbseff_dvb = f64::midpoint(1.0, t0 / t1);
        let vbseff = if vbseff_raw < vbs { vbs } else { vbseff_raw };
        let (sqrt_phis, dsqrt_phis_dvb, phis, dphis_dvb);
        if vbseff > 0.0 {
            let t0 = phi / (phi + vbseff);
            phis = phi * t0;
            dphis_dvb = -t0 * t0;
            sqrt_phis = p.derived.phis3 / 0.5f64.mul_add(vbseff, phi);
            dsqrt_phis_dvb = -0.5 * sqrt_phis * sqrt_phis / p.derived.phis3;
        } else {
            phis = phi - vbseff;
            dphis_dvb = -1.0;
            sqrt_phis = phis.sqrt();
            dsqrt_phis_dvb = -0.5 / sqrt_phis;
        }
        let xdep = p.derived.xdep0 * sqrt_phis / p.derived.sqrtphi;
        let dxdep_dvb = (p.derived.xdep0 / p.derived.sqrtphi) * dsqrt_phis_dvb;

        let leff = p.leff;
        let v0 = p.derived.vbi - phi;
        let t3 = xdep.sqrt();

        // dvt2 body-bias dependence of the threshold.
        let (t1, slope) = rational_floor(p.binned.dvt2 * vbseff, -0.5, 1., 8.)?;
        let t2 = p.binned.dvt2 * slope;
        let lt1 = factor1 * t3 * t1;
        let dlt1_dvb = factor1 * t3.mul_add(t2, 0.5 / t3 * t1 * dxdep_dvb);
        let (t1, slope) = rational_floor(p.binned.dvt2w * vbseff, -0.5, 1., 8.)?;
        let t2 = p.binned.dvt2w * slope;
        let ltw = factor1 * t3 * t1;
        let dltw_dvb = factor1 * t3.mul_add(t2, 0.5 / t3 * t1 * dxdep_dvb);

        // Narrow/short-channel threshold shifts.
        let (theta0, dtheta0_dvb) = {
            let t0 = -0.5 * p.binned.dvt1 * leff / lt1;
            if t0 > -EXP_THRESHOLD {
                let t1 = t0.exp();
                let dt1_dvb = -t0 / lt1 * t1 * dlt1_dvb;
                (
                    t1 * 2.0f64.mul_add(t1, 1.0),
                    4.0f64.mul_add(t1, 1.0) * dt1_dvb,
                )
            } else {
                let t1 = MIN_EXP;
                (t1 * 2.0f64.mul_add(t1, 1.0), 0.0)
            }
        };
        let thetavth = p.binned.dvt0 * theta0;
        let delt_vth = thetavth * v0;
        let ddelt_vth_dvb = p.binned.dvt0 * dtheta0_dvb * v0;

        let (t2w, dt2w_dvb) = {
            let t0 = -0.5 * p.binned.dvt1w * p.weff * leff / ltw;
            if t0 > -EXP_THRESHOLD {
                let t1 = t0.exp();
                let dt1_dvb = -t0 / ltw * t1 * dltw_dvb;
                (
                    t1 * 2.0f64.mul_add(t1, 1.0),
                    4.0f64.mul_add(t1, 1.0) * dt1_dvb,
                )
            } else {
                let t1 = MIN_EXP;
                (t1 * 2.0f64.mul_add(t1, 1.0), 0.0)
            }
        };
        let t0 = p.binned.dvt0w * t2w;
        let t2 = t0 * v0;
        let dt2_dvb = p.binned.dvt0w * dt2w_dvb * v0;

        let temp_ratio = (temp - tnom) / tnom;
        let length_ratio = p.binned.nlx / leff;
        let root_offset = length_ratio / ((1.0 + length_ratio).sqrt() + 1.0);
        let t1 = p
            .binned
            .kt2
            .mul_add(vbseff, p.binned.kt1 + p.binned.kt1l / leff)
            .mul_add(temp_ratio, p.derived.k1ox * root_offset * p.derived.sqrtphi);
        let effective_width_offset = p.weff + p.binned.w0;
        let tmp2 = tox * phi / effective_width_offset;

        let t3eta = p.binned.etab.mul_add(vbseff, p.binned.eta0);
        let (t3eta, t4) = rational_floor(t3eta, 1e-4, 0., 2e4)?;
        let ddibl_sft_dvd = t3eta * p.derived.theta0vb0;
        let dibl_sft = ddibl_sft_dvd * vds;

        // `here->BSIM3vth0` is the instance value after DELVTO.
        let vth = p.binned.k3b.mul_add(vbseff, p.binned.k3).mul_add(
            tmp2,
            p.derived.k2ox.mul_add(
                -vbseff,
                p.derived.k1ox.mul_add(
                    sqrt_phis,
                    p.binned.k1.mul_add(-p.derived.sqrtphi, type_ * p.vth0),
                ),
            ) - delt_vth
                - t2,
        ) + t1
            - dibl_sft;
        let von = vth;
        let dvth_dvb = p.binned.kt2.mul_add(
            temp_ratio,
            (p.binned.etab * vds * p.derived.theta0vb0).mul_add(
                -t4,
                p.binned.k3b.mul_add(
                    tmp2,
                    p.derived.k1ox.mul_add(dsqrt_phis_dvb, -p.derived.k2ox)
                        - ddelt_vth_dvb
                        - dt2_dvb,
                ),
            ),
        );
        let dvth_dvd = -ddibl_sft_dvd;

        // Subthreshold slope factor n.
        let tmp2n = p.binned.nfactor * EPS_SI / xdep;
        let tmp3n = p
            .binned
            .cdscd
            .mul_add(vds, p.binned.cdscb.mul_add(vbseff, p.binned.cdsc));
        let tmp4 = (tmp3n.mul_add(theta0, tmp2n) + p.binned.cit) / cox;
        let (n, slope) = rational_floor(tmp4, -0.5, 1., 8.)?;
        let dn_dvb = p.binned.cdscb.mul_add(
            theta0,
            tmp3n.mul_add(dtheta0_dvb, -tmp2n / xdep * dxdep_dvb),
        ) / cox
            * slope;
        let dn_dvd = p.binned.cdscd * theta0 / cox * slope;

        let (vgs_eff, dvgseff_dvg) =
            poly_depletion(vgs, p.vfb + phi, p.binned.ngate, EPS_SI, cox, CHARGE_Q)?;
        let vgst = vgs_eff - vth;

        // Effective gate overdrive (Vgsteff), with C's three regimes.
        let t10 = 2.0 * n * vtm;
        let vgst_nvt = vgst / t10;
        let exp_arg = 2.0f64.mul_add(p.binned.voff, -vgst) / t10;
        let (vgsteff, mut dvgsteff_dvg, dvgsteff_dvd, dvgsteff_dvb);
        if vgst_nvt > EXP_THRESHOLD {
            vgsteff = vgst;
            dvgsteff_dvg = dvgseff_dvg;
            dvgsteff_dvd = -dvth_dvd;
            dvgsteff_dvb = -dvth_dvb;
        } else if exp_arg > EXP_THRESHOLD {
            let t0 = (vgst - p.binned.voff) / (n * vtm);
            let exp_vgst = t0.exp();
            vgsteff = vtm * p.derived.cdep0 / cox * exp_vgst;
            dvgsteff_dvg = vgsteff / (n * vtm);
            dvgsteff_dvd = -dvgsteff_dvg * (t0 * vtm).mul_add(dn_dvd, dvth_dvd);
            dvgsteff_dvb = -dvgsteff_dvg * (t0 * vtm).mul_add(dn_dvb, dvth_dvb);
            dvgsteff_dvg *= dvgseff_dvg;
        } else {
            let exp_vgst = vgst_nvt.exp();
            let t1 = t10 * exp_vgst.ln_1p();
            let dt1_dvg = exp_vgst / (1.0 + exp_vgst);
            let dt1_dvb = (t1 / n).mul_add(dn_dvb, -dt1_dvg * (vgst / n).mul_add(dn_dvb, dvth_dvb));
            let dt1_dvd = (t1 / n).mul_add(dn_dvd, -dt1_dvg * (vgst / n).mul_add(dn_dvd, dvth_dvd));

            let dt2_dvg = -cox / (vtm * p.derived.cdep0) * exp_arg.exp();
            let t2v = t10.mul_add(-dt2_dvg, 1.0);
            let dt2_dvd = ((t2v - 1.0) / n).mul_add(
                dn_dvd,
                -dt2_dvg * (2.0 * vtm * exp_arg).mul_add(-dn_dvd, dvth_dvd),
            );
            let dt2_dvb = ((t2v - 1.0) / n).mul_add(
                dn_dvb,
                -dt2_dvg * (2.0 * vtm * exp_arg).mul_add(-dn_dvb, dvth_dvb),
            );

            vgsteff = t1 / t2v;
            let t3 = t2v * t2v;
            dvgsteff_dvg = t1.mul_add(-dt2_dvg, t2v * dt1_dvg) / t3 * dvgseff_dvg;
            dvgsteff_dvd = t1.mul_add(-dt2_dvd, t2v * dt1_dvd) / t3;
            dvgsteff_dvb = t1.mul_add(-dt2_dvb, t2v * dt1_dvb) / t3;
        }

        // Effective channel width.
        let t9 = sqrt_phis - p.derived.sqrtphi;
        let raw_width = 2.0f64.mul_add(-p.binned.dwb.mul_add(t9, p.binned.dwg * vgsteff), p.weff);
        let (weff, slope) = rational_floor(raw_width, 2e-8, 0., 1e8)?;
        let dweff_dvg = -2.0 * p.binned.dwg * slope;
        let dweff_dvb = -2.0 * p.binned.dwb * dsqrt_phis_dvb * slope;

        // Series resistance.
        let t0 = p.binned.prwb.mul_add(t9, p.binned.prwg * vgsteff);
        let (resistance_factor, slope) = rational_floor(t0, -0.9, 1., 20.)?;
        let rds = p.thermal.rds0 * resistance_factor;
        let drds_dvg = p.thermal.rds0 * p.binned.prwg * slope;
        let drds_dvb = p.thermal.rds0 * p.binned.prwb * dsqrt_phis_dvb * slope;

        // Bulk-charge factor.
        let t1 = 0.5 * p.derived.k1ox / sqrt_phis;
        let dt1_dvb = -t1 / sqrt_phis * dsqrt_phis_dvb;
        let t9 = (p.binned.xj * xdep).sqrt();
        let tmp1 = 2.0f64.mul_add(t9, leff);
        let t5 = leff / tmp1;
        let tmp2a = p.binned.a0 * t5;
        let tmp3a = p.weff + p.binned.b1;
        let tmp4a = p.binned.b0 / tmp3a;
        let t2a = tmp2a + tmp4a;
        let dt2_dvb = -t9 / tmp1 / xdep * dxdep_dvb;
        let t6 = t5 * t5;
        let t7 = t5 * t6;
        let mut abulk0 = t1.mul_add(t2a, 1.0);
        let mut dabulk0_dvb = t2a.mul_add(dt1_dvb, t1 * tmp2a * dt2_dvb);
        let t8 = p.binned.ags * p.binned.a0 * t7;
        let mut dabulk_dvg = -t1 * t8;
        let mut abulk = dabulk_dvg.mul_add(vgsteff, abulk0);
        let mut dabulk_dvb =
            (t8 * vgsteff).mul_add(-(3.0 * t1).mul_add(dt2_dvb, dt1_dvb), dabulk0_dvb);
        let slope;
        (abulk0, slope) = rational_floor(abulk0, 0.1, 0., 20.)?;
        dabulk0_dvb *= slope;
        let slope;
        (abulk, slope) = rational_floor(abulk, 0.1, 0., 20.)?;
        dabulk_dvb *= slope;
        dabulk_dvg *= slope;
        let (t0k, slope) = reciprocal_floor(p.binned.keta * vbseff, -0.9, 20.)?;
        let dt0_dvb = p.binned.keta * slope;
        dabulk_dvg *= t0k;
        dabulk_dvb = abulk.mul_add(dt0_dvb, dabulk_dvb * t0k);
        dabulk0_dvb = abulk0.mul_add(dt0_dvb, dabulk0_dvb * t0k);
        abulk *= t0k;
        abulk0 *= t0k;

        // Mobility, by MOBMOD.
        let mut denomi_dvg;
        let mut denomi_dvd;
        let mut denomi_dvb;
        let t5m;
        if m.values.mobmod.value == 1 {
            let t0 = vgsteff + vth + vth;
            let t2 = p.binned.uc.mul_add(vbseff, p.binned.ua);
            let t3 = t0 / tox;
            t5m = t3 * p.binned.ub.mul_add(t3, t2);
            denomi_dvg = (2.0 * p.binned.ub).mul_add(t3, t2) / tox;
            denomi_dvd = denomi_dvg * 2.0 * dvth_dvd;
            denomi_dvb = p.binned.uc.mul_add(t3, denomi_dvg * 2.0 * dvth_dvb);
        } else if m.values.mobmod.value == 2 {
            t5m = vgsteff / tox
                * (p.binned.uc.mul_add(vbseff, p.binned.ua) + p.binned.ub * vgsteff / tox);
            denomi_dvg = (p.binned.uc.mul_add(vbseff, p.binned.ua)
                + 2.0 * p.binned.ub * vgsteff / tox)
                / tox;
            denomi_dvd = 0.0;
            denomi_dvb = vgsteff * p.binned.uc / tox;
        } else {
            let t0 = vgsteff + vth + vth;
            let t2 = p.binned.uc.mul_add(vbseff, 1.0);
            let t3 = t0 / tox;
            let t4 = t3 * p.binned.ub.mul_add(t3, p.binned.ua);
            t5m = t4 * t2;
            denomi_dvg = (2.0 * p.binned.ub).mul_add(t3, p.binned.ua) * t2 / tox;
            denomi_dvd = denomi_dvg * 2.0 * dvth_dvd;
            denomi_dvb = p.binned.uc.mul_add(t4, denomi_dvg * 2.0 * dvth_dvb);
        }
        let (denomi, slope) = rational_floor(t5m, -0.8, 1., 10.)?;
        denomi_dvg *= slope;
        denomi_dvd *= slope;
        denomi_dvb *= slope;
        let u0temp = p.u0temp;
        let ueff = u0temp / denomi;
        let t9c = -ueff / denomi;
        let dueff_dvg = t9c * denomi_dvg;
        let dueff_dvd = t9c * denomi_dvd;
        let dueff_dvb = t9c * denomi_dvb;

        // Saturation drain voltage.
        let wvcox = weff * p.thermal.vsattemp * cox;
        let wvcoxrds = wvcox * rds;
        let esat = 2.0 * p.thermal.vsattemp / ueff;
        let esatl = esat * leff;
        let t0e = -esatl / ueff;
        let desatl_dvg = t0e * dueff_dvg;
        let desatl_dvd = t0e * dueff_dvd;
        let desatl_dvb = t0e * dueff_dvb;

        let a1v = p.binned.a1;
        let a2v = p.binned.a2;
        let (lambda, dlambda_dvg) = if a1v == 0.0 {
            (a2v, 0.0)
        } else if a1v > 0.0 {
            let t0 = 1.0 - a2v;
            let t1 = a1v.mul_add(-vgsteff, t0) - 0.0001;
            let t2 = 0.0004f64.mul_add(t0, t1 * t1).sqrt();
            (
                a2v + t0 - f64::midpoint(t1, t2),
                0.5 * a1v * (1.0 + t1 / t2),
            )
        } else {
            let t1 = a1v.mul_add(vgsteff, a2v) - 0.0001;
            let t2 = 0.0004f64.mul_add(a2v, t1 * t1).sqrt();
            (f64::midpoint(t1, t2), 0.5 * a1v * (1.0 + t1 / t2))
        };

        let vgst2vtm = 2.0f64.mul_add(vtm, vgsteff);
        let abulk_over_vgst2vtm = abulk / vgst2vtm;
        let (tmp2v, tmp3v) = if rds > 0.0 {
            (
                drds_dvg / rds + dweff_dvg / weff,
                drds_dvb / rds + dweff_dvb / weff,
            )
        } else {
            (dweff_dvg / weff, dweff_dvb / weff)
        };

        let (vdsat, dvdsat_dvg, dvdsat_dvd, dvdsat_dvb, tmp1v);
        if rds == 0.0 && lambda == 1.0 {
            let t0 = 1.0 / abulk.mul_add(esatl, vgst2vtm);
            tmp1v = 0.0;
            let t1v = t0 * t0;
            let t2 = vgst2vtm * t0;
            let t3 = esatl * vgst2vtm;
            vdsat = t3 * t0;
            let dt0_dvg = -(esatl.mul_add(dabulk_dvg, abulk * desatl_dvg) + 1.0) * t1v;
            let dt0_dvd = -(abulk * desatl_dvd) * t1v;
            let dt0_dvb = -dabulk_dvb.mul_add(esatl, abulk * desatl_dvb) * t1v;
            dvdsat_dvg = esatl.mul_add(t0, t2.mul_add(desatl_dvg, t3 * dt0_dvg));
            dvdsat_dvd = t2.mul_add(desatl_dvd, t3 * dt0_dvd);
            dvdsat_dvb = t2.mul_add(desatl_dvb, t3 * dt0_dvb);
        } else {
            tmp1v = dlambda_dvg / (lambda * lambda);
            let t9 = abulk * wvcoxrds;
            let t8 = abulk * t9;
            let t7 = vgst2vtm * t9;
            let t6 = vgst2vtm * wvcoxrds;
            let t0 = 2.0 * abulk * (t9 - 1.0 + 1.0 / lambda);
            let dt0_dvg = 2.0
                * (2.0f64.mul_add(t9, 1.0 / lambda) - 1.0)
                    .mul_add(dabulk_dvg, abulk.mul_add(-tmp1v, t8 * tmp2v));
            let dt0_dvb = 2.0
                * f64::mul_add(
                    1.0 / lambda - 1.0,
                    dabulk_dvb,
                    t8 * (2.0 / abulk).mul_add(dabulk_dvb, tmp3v),
                );
            let t1v = 3.0f64.mul_add(t7, abulk.mul_add(esatl, vgst2vtm * (2.0 / lambda - 1.0)));
            let dt1_dvg = 3.0f64.mul_add(
                t6.mul_add(dabulk_dvg, t7.mul_add(tmp2v, t9)),
                esatl.mul_add(
                    dabulk_dvg,
                    abulk.mul_add(
                        desatl_dvg,
                        (2.0 * vgst2vtm).mul_add(-tmp1v, 2.0 / lambda - 1.0),
                    ),
                ),
            );
            let dt1_dvb = 3.0f64.mul_add(
                t7.mul_add(tmp3v, t6 * dabulk_dvb),
                esatl.mul_add(dabulk_dvb, abulk * desatl_dvb),
            );
            let dt1_dvd = abulk * desatl_dvd;
            let t2v = vgst2vtm * 2.0f64.mul_add(t6, esatl);
            let dt2_dvg = t6.mul_add(
                (2.0 * vgst2vtm).mul_add(tmp2v, 4.0),
                vgst2vtm.mul_add(desatl_dvg, esatl),
            );
            let dt2_dvb = vgst2vtm * (2.0 * t6).mul_add(tmp3v, desatl_dvb);
            let dt2_dvd = vgst2vtm * desatl_dvd;
            let root = SaturationRoot::new(&t0, t1v, t2v)?;
            vdsat = root.value;
            dvdsat_dvg = root.derivative(&dt0_dvg, &dt1_dvg, &dt2_dvg)?;
            dvdsat_dvb = root.derivative(&dt0_dvb, &dt1_dvb, &dt2_dvb)?;
            dvdsat_dvd = root.derivative(&0., &dt1_dvd, &dt2_dvd)?;
        }

        let smoothing = smooth_drain(vdsat, vds, p.binned.delta)?;
        let vdseff = if smoothing.effective > vds {
            vds
        } else {
            smoothing.effective
        };
        let dvdseff_dvg = smoothing.saturation_partial * dvdsat_dvg;
        let dvdseff_dvb = smoothing.saturation_partial * dvdsat_dvb;
        let dvdseff_dvd = smoothing
            .saturation_partial
            .mul_add(dvdsat_dvd, smoothing.drain_partial);
        let diff_vds = smoothing.drain_excess;

        // Early voltage: saturation, CLM and DIBL components.
        let tmp4v = 1.0 - 0.5 * abulk * vdsat / vgst2vtm;
        let t9v = wvcoxrds * vgsteff;
        let t8v = t9v / vgst2vtm;
        let t0v = (2.0 * t9v).mul_add(tmp4v, esatl + vdsat);
        let t7v = 2.0 * wvcoxrds * tmp4v;
        let dt0_dvg = t8v.mul_add(
            -f64::mul_add(
                vdsat,
                dabulk_dvg,
                abulk.mul_add(dvdsat_dvg, -(abulk * vdsat / vgst2vtm)),
            ),
            t7v.mul_add(f64::mul_add(tmp2v, vgsteff, 1.0), desatl_dvg + dvdsat_dvg),
        );
        let dt0_dvb = t8v.mul_add(
            -abulk.mul_add(dvdsat_dvb, dabulk_dvb * vdsat),
            (t7v * tmp3v).mul_add(vgsteff, desatl_dvb + dvdsat_dvb),
        );
        let dt0_dvd = (t8v * abulk).mul_add(-dvdsat_dvd, desatl_dvd + dvdsat_dvd);
        let t9e = wvcoxrds * abulk;
        let t1e = 2.0 / lambda - 1.0 + t9e;
        let dt1_dvg = wvcoxrds.mul_add(abulk.mul_add(tmp2v, dabulk_dvg), -2.0 * tmp1v);
        let dt1_dvb = t9e.mul_add(tmp3v, dabulk_dvb * wvcoxrds);
        let vasat = t0v / t1e;
        let dvasat_dvg = vasat.mul_add(-dt1_dvg, dt0_dvg) / t1e;
        let dvasat_dvb = vasat.mul_add(-dt1_dvb, dt0_dvb) / t1e;
        let dvasat_dvd = dt0_dvd / t1e;

        let (vaclm, dvaclm_dvg, dvaclm_dvd, dvaclm_dvb) = if p.binned.pclm > 0.0
            && diff_vds > 1.0e-10
        {
            let t0 = 1.0 / (p.binned.pclm * abulk * p.derived.litl);
            let dt0_dvb = -t0 / abulk * dabulk_dvb;
            let dt0_dvg = -t0 / abulk * dabulk_dvg;
            let t2 = vgsteff / esatl;
            let t1 = leff * (abulk + t2);
            let dt1_dvg = leff * (t2.mul_add(-desatl_dvg, 1.0) / esatl + dabulk_dvg);
            let dt1_dvb = leff * (dabulk_dvb - t2 * desatl_dvb / esatl);
            // b3ld.c divides by Esat (not EsatL) here; keep that.
            let dt1_dvd = -t2 * desatl_dvd / esat;
            let t9 = t0 * t1;
            let vaclm = t9 * diff_vds;
            (
                vaclm,
                (t1 * diff_vds).mul_add(dt0_dvg, t9.mul_add(-dvdseff_dvg, t0 * dt1_dvg * diff_vds)),
                t9.mul_add(1.0 - dvdseff_dvd, t0 * dt1_dvd * diff_vds),
                t9.mul_add(-dvdseff_dvb, t0.mul_add(dt1_dvb, dt0_dvb * t1) * diff_vds),
            )
        } else {
            (MAX_EXP, 0.0, 0.0, 0.0)
        };

        let (vadibl, dvadibl_dvg, dvadibl_dvd, dvadibl_dvb) = if p.derived.thetarout > 0.0 {
            let t8 = abulk * vdsat;
            let t0 = vgst2vtm * t8;
            let dt0_dvg =
                (vgst2vtm * vdsat).mul_add(dabulk_dvg, (vgst2vtm * abulk).mul_add(dvdsat_dvg, t8));
            let dt0_dvb = vgst2vtm * abulk.mul_add(dvdsat_dvb, dabulk_dvb * vdsat);
            let dt0_dvd = vgst2vtm * abulk * dvdsat_dvd;
            let t1 = vgst2vtm + t8;
            let dt1_dvg = f64::mul_add(vdsat, dabulk_dvg, abulk.mul_add(dvdsat_dvg, 1.0));
            let dt1_dvb = dabulk_dvb.mul_add(vdsat, abulk * dvdsat_dvb);
            let dt1_dvd = abulk * dvdsat_dvd;
            let t9 = t1 * t1;
            let t2 = p.derived.thetarout;
            let mut vadibl = (vgst2vtm - t0 / t1) / t2;
            let mut dvadibl_dvg = (1.0 - dt0_dvg / t1 + t0 * dt1_dvg / t9) / t2;
            let mut dvadibl_dvb = (-dt0_dvb / t1 + t0 * dt1_dvb / t9) / t2;
            let mut dvadibl_dvd = (-dt0_dvd / t1 + t0 * dt1_dvd / t9) / t2;
            let (factor, slope) = reciprocal_floor(p.binned.pdiblcb * vbseff, -0.9, 20.)?;
            dvadibl_dvb = (vadibl * p.binned.pdiblcb).mul_add(slope, dvadibl_dvb * factor);
            dvadibl_dvg *= factor;
            dvadibl_dvd *= factor;
            vadibl *= factor;
            (vadibl, dvadibl_dvg, dvadibl_dvd, dvadibl_dvb)
        } else {
            (MAX_EXP, 0.0, 0.0, 0.0)
        };

        let t8a = p.binned.pvag / esatl;
        let t9a = t8a * vgsteff;
        let (t0a, slope) = rational_floor(t9a, -0.9, 1., 20.)?;
        let dt0_dvg = t8a * (1.0 - vgsteff * desatl_dvg / esatl) * slope;
        let dt0_dvb = -t9a * desatl_dvb / esatl * slope;
        let dt0_dvd = -t9a * desatl_dvd / esatl * slope;

        let tmp1a = vaclm * vaclm;
        let tmp2a2 = vadibl * vadibl;
        let mut tmp3a2 = vaclm + vadibl;
        let t1a = vaclm * vadibl / tmp3a2;
        tmp3a2 *= tmp3a2;
        let dt1_dvg = f64::mul_add(tmp2a2, dvaclm_dvg, tmp1a * dvadibl_dvg) / tmp3a2;
        let dt1_dvd = f64::mul_add(tmp2a2, dvaclm_dvd, tmp1a * dvadibl_dvd) / tmp3a2;
        let dt1_dvb = f64::mul_add(tmp2a2, dvaclm_dvb, tmp1a * dvadibl_dvb) / tmp3a2;
        let va = f64::mul_add(t0a, t1a, vasat);
        let dva_dvg = f64::mul_add(t0a, dt1_dvg, f64::mul_add(t1a, dt0_dvg, dvasat_dvg));
        let dva_dvd = f64::mul_add(t0a, dt1_dvd, f64::mul_add(t1a, dt0_dvd, dvasat_dvd));
        let dva_dvb = f64::mul_add(t0a, dt1_dvb, f64::mul_add(t1a, dt0_dvb, dvasat_dvb));

        let (vascbe, dvascbe_dvg, dvascbe_dvd, dvascbe_dvb) = if p.binned.pscbe2 > 0.0 {
            if diff_vds > p.binned.pscbe1 * p.derived.litl / EXP_THRESHOLD {
                let t0 = p.binned.pscbe1 * p.derived.litl / diff_vds;
                let vascbe = leff * t0.exp() / p.binned.pscbe2;
                let t1 = t0 * vascbe / diff_vds;
                (
                    vascbe,
                    t1 * dvdseff_dvg,
                    -t1 * (1.0 - dvdseff_dvd),
                    t1 * dvdseff_dvb,
                )
            } else {
                (MAX_EXP * leff / p.binned.pscbe2, 0.0, 0.0, 0.0)
            }
        } else {
            (MAX_EXP, 0.0, 0.0, 0.0)
        };

        // Drain current and its derivatives.
        let coxwovl = cox * weff / leff;
        let beta = ueff * coxwovl;
        let dbeta_dvg = coxwovl.mul_add(dueff_dvg, beta * dweff_dvg / weff);
        let dbeta_dvd = coxwovl * dueff_dvd;
        let dbeta_dvb = coxwovl.mul_add(dueff_dvb, beta * dweff_dvb / weff);

        let t0 = 1.0 - 0.5 * abulk * vdseff / vgst2vtm;
        let dt0_dvg =
            -0.5 * f64::mul_add(
                vdseff,
                dabulk_dvg,
                abulk.mul_add(dvdseff_dvg, -(abulk * vdseff / vgst2vtm)),
            ) / vgst2vtm;
        let dt0_dvd = -0.5 * abulk * dvdseff_dvd / vgst2vtm;
        let dt0_dvb = -0.5 * dabulk_dvb.mul_add(vdseff, abulk * dvdseff_dvb) / vgst2vtm;

        let fgche1 = vgsteff * t0;
        let dfgche1_dvg = f64::mul_add(vgsteff, dt0_dvg, t0);
        let dfgche1_dvd = vgsteff * dt0_dvd;
        let dfgche1_dvb = vgsteff * dt0_dvb;

        let t9f = vdseff / esatl;
        let fgche2 = 1.0 + t9f;
        let dfgche2_dvg = t9f.mul_add(-desatl_dvg, dvdseff_dvg) / esatl;
        let dfgche2_dvd = t9f.mul_add(-desatl_dvd, dvdseff_dvd) / esatl;
        let dfgche2_dvb = t9f.mul_add(-desatl_dvb, dvdseff_dvb) / esatl;

        let gche = beta * fgche1 / fgche2;
        let dgche_dvg =
            gche.mul_add(-dfgche2_dvg, fgche1.mul_add(dbeta_dvg, beta * dfgche1_dvg)) / fgche2;
        let dgche_dvd =
            gche.mul_add(-dfgche2_dvd, fgche1.mul_add(dbeta_dvd, beta * dfgche1_dvd)) / fgche2;
        let dgche_dvb =
            gche.mul_add(-dfgche2_dvb, fgche1.mul_add(dbeta_dvb, beta * dfgche1_dvb)) / fgche2;

        let t0i = gche.mul_add(rds, 1.0);
        let t9i = vdseff / t0i;
        let idl = gche * t9i;
        let didl_dvg =
            (idl * gche / t0i).mul_add(-drds_dvg, t9i.mul_add(dgche_dvg, gche * dvdseff_dvg) / t0i);
        let didl_dvd = t9i.mul_add(dgche_dvd, gche * dvdseff_dvd) / t0i;
        let didl_dvb =
            (idl * drds_dvb).mul_add(-gche, t9i.mul_add(dgche_dvb, gche * dvdseff_dvb)) / t0i;

        let t9s = diff_vds / va;
        let t0s = 1.0 + t9s;
        let idsa = idl * t0s;
        let didsa_dvg = t0s.mul_add(didl_dvg, -(idl * t9s.mul_add(dva_dvg, dvdseff_dvg) / va));
        let didsa_dvd = t0s.mul_add(
            didl_dvd,
            idl * t9s.mul_add(-dva_dvd, 1.0 - dvdseff_dvd) / va,
        );
        let didsa_dvb = t0s.mul_add(didl_dvb, -(idl * t9s.mul_add(dva_dvb, dvdseff_dvb) / va));

        let t9c2 = diff_vds / vascbe;
        let t0c = 1.0 + t9c2;
        let ids = idsa * t0c;
        let mut gds = f64::mul_add(
            t0c,
            didsa_dvd,
            idsa * f64::mul_add(t9c2, -dvascbe_dvd, 1.0 - dvdseff_dvd) / vascbe,
        );
        let mut gmbs = f64::mul_add(
            t0c,
            didsa_dvb,
            -(idsa * f64::mul_add(t9c2, dvascbe_dvb, dvdseff_dvb) / vascbe),
        );
        let mut gm = f64::mul_add(
            t0c,
            didsa_dvg,
            -(idsa * f64::mul_add(t9c2, dvascbe_dvg, dvdseff_dvg) / vascbe),
        );

        gds = gm.mul_add(dvgsteff_dvd, gds);
        gmbs = gm.mul_add(dvgsteff_dvb, gmbs);
        gm *= dvgsteff_dvg;
        gmbs *= dvbseff_dvb;

        // Substrate current.
        let tmp = p.binned.alpha1.mul_add(leff, p.binned.alpha0);
        let (isub, gbgs, gbds, gbbs) = if tmp <= 0.0 || p.binned.beta0 <= 0.0 {
            (0.0, 0.0, 0.0, 0.0)
        } else {
            let t2 = tmp / leff;
            let (t1, dt1_dvg, dt1_dvd, dt1_dvb) = if diff_vds > p.binned.beta0 / EXP_THRESHOLD {
                let t0 = -p.binned.beta0 / diff_vds;
                let t1 = t2 * diff_vds * t0.exp();
                let t3 = t1 / diff_vds * (t0 - 1.0);
                (
                    t1,
                    t3 * dvdseff_dvg,
                    t3 * (dvdseff_dvd - 1.0),
                    t3 * dvdseff_dvb,
                )
            } else {
                let t3 = t2 * MIN_EXP;
                (
                    t3 * diff_vds,
                    -t3 * dvdseff_dvg,
                    t3 * (1.0 - dvdseff_dvd),
                    -t3 * dvdseff_dvb,
                )
            };
            let isub = t1 * idsa;
            let mut gbg = idsa.mul_add(dt1_dvg, t1 * didsa_dvg);
            let mut gbd = idsa.mul_add(dt1_dvd, t1 * didsa_dvd);
            let mut gbb = idsa.mul_add(dt1_dvb, t1 * didsa_dvb);
            gbd = gbg.mul_add(dvgsteff_dvd, gbd);
            gbb = gbg.mul_add(dvgsteff_dvb, gbb);
            gbg *= dvgsteff_dvg;
            gbb *= dvbseff_dvb;
            (isub, gbg, gbd, gbb)
        };

        let result = DcEvaluation {
            ids,
            gm,
            gds,
            gmbs,
            isub,
            gbgs,
            gbds,
            gbbs,
            cbs,
            cbd,
            gbs,
            gbd,
            vbs,
            vbd: vbs - vds,
            vgs,
            vds,
            mode,
            von,
            vgsteff,
            vdseff,
            vdsat,
            abulk,
            ueff,
            rds,
            thetavth,
            abulk_over_vgst2vtm,
            cv: CvIntermediates {
                vtm,
                cox,
                tox,
                phi,
                vgs_eff,
                d_vgs_eff_dvg: dvgseff_dvg,
                vth,
                d_vth_dvd: dvth_dvd,
                d_vth_dvb: dvth_dvb,
                vgst,
                vbseff,
                d_vbseff_dvb: dvbseff_dvb,
                phis,
                d_phis_dvb: dphis_dvb,
                sqrt_phis,
                d_sqrt_phis_dvb: dsqrt_phis_dvb,
                abulk0,
                d_abulk0_dvb: dabulk0_dvb,
                n,
                d_n_dvd: dn_dvd,
                d_n_dvb: dn_dvb,
                vgsteff,
                d_vgsteff_dvg: dvgsteff_dvg,
                d_vgsteff_dvd: dvgsteff_dvd,
                d_vgsteff_dvb: dvgsteff_dvb,
            },
        };
        for (name, value) in [
            ("ids", result.ids),
            ("gm", result.gm),
            ("gds", result.gds),
            ("gmbs", result.gmbs),
            ("von", result.von),
            ("ueff", result.ueff),
            ("vdseff", result.vdseff),
            ("cbs", result.cbs),
            ("cbd", result.cbd),
        ] {
            if !value.is_finite() {
                return Err(crate::primitives::SpiceError::Numerical {
                    context: "BSIM3 DC evaluation (bsim3/b3ld.c)".into(),
                    message: format!("{name} is not finite"),
                });
            }
        }
        Ok(result)
    }
}

/// `@inst[param]` values C's `b3ask.c` reads from the instance after a load.
impl DcEvaluation {
    /// The value of a DC operating-point ask keyword, before multiplicity.
    /// These are C's `IFparm` keywords; `cbs`/`cbd` are capacitances in C and are
    /// therefore absent, while `ibs`/`ibd` are the junction currents. Charge,
    /// NQS, noise and gate-current asks are not part of this core.
    #[must_use = "the returned option distinguishes a supported operating-point observation from an unknown keyword"]
    pub fn ask(&self, keyword: &str) -> Option<Real> {
        match keyword.to_ascii_lowercase().as_str() {
            "id" => Some(self.ids),
            "gm" => Some(self.gm),
            "gds" => Some(self.gds),
            "gmbs" => Some(self.gmbs),
            "gbd" => Some(self.gbds),
            "gbs" => Some(self.gbs),
            "vth" => Some(self.von),
            "vbs" => Some(self.vbs),
            "vgs" => Some(self.vgs),
            "vds" => Some(self.vds),
            "vdsat" => Some(self.vdsat),
            "ibs" => Some(self.cbs),
            "ibd" => Some(self.cbd),
            _ => None,
        }
    }
}
