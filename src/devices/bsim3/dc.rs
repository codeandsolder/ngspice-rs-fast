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
    clippy::too_many_lines,
    reason = "a complete translated C numerical routine remains contiguous for equation parity review"
)]

use super::defaults::ModelDefaults;
use super::setup::InstanceSetup;
use super::temperature::TemperatureModel;
use crate::devices::bsim_physics::{
    SlopeParameters, ThresholdParameters, ThresholdRollOff, bsim3_body_limit, bsim3_junction,
    bsim3_overdrive, channel_conductance, channel_rolloff, dibl_early_voltage, impact_ionization,
    poly_depletion, rational_floor, reciprocal_floor, saturation_early_voltage, saturation_shape,
    saturation_voltage, series_conductance, smooth_drain, substrate_early_voltage,
    surface_potential,
};
use crate::maths::taylor::{Constitutive, Jet, differential::Differential};
use crate::primitives::{Real, SpiceResult};
use std::cmp::Ordering;

const MAX_EXP: Real = 5.834_617_425e14;
const EPS_SI: Real = 1.035_94e-10;
const CHARGE_Q: Real = 1.602_19e-19;

/// Terminal bias applied to the intrinsic device (the `dNodePrime`/
/// `sNodePrime` reference C uses), already multiplied by the model polarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bias<S = Real> {
    /// Body-source voltage.
    pub vbs: S,
    /// Gate-source voltage.
    pub vgs: S,
    /// Drain-source voltage.
    pub vds: S,
}

/// Bias-dependent intermediates `b3ld.c` computes before the charge block and
/// the charge preamble (`b3acld.c`/`b3ld.c` C-V) reuses.
///
/// They are carried out of the DC evaluation rather than recomputed so the
/// charge model sees exactly the values (and rounding) C's single pass used.
/// Derivatives are with respect to the polarity-scaled device bias.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CvIntermediates<S = Real> {
    /// Thermal voltage (C `model->BSIM3vtm`).
    pub vtm: Real,
    /// Oxide capacitance per area (C `model->BSIM3cox`).
    pub cox: Real,
    /// Oxide thickness in metres (C `model->BSIM3tox`).
    pub tox: Real,
    /// Surface potential parameter (C `pParam->BSIM3phi`).
    pub phi: Real,
    /// Effective gate-source voltage after poly depletion.
    pub vgs_eff: S,
    /// Its gate derivative.
    pub d_vgs_eff_dvg: Real,
    /// Threshold voltage.
    pub vth: S,
    /// Its drain derivative.
    pub d_vth_dvd: Real,
    /// Its body derivative.
    pub d_vth_dvb: Real,
    /// `vgs_eff - vth`, before the subthreshold smoothing.
    pub vgst: S,
    /// Smoothed body-source voltage.
    pub vbseff: S,
    /// Its body derivative.
    pub d_vbseff_dvb: Real,
    /// Surface potential at `vbseff` (C `Phis`).
    pub phis: S,
    /// Its body derivative (C `dPhis_dVb`).
    pub d_phis_dvb: Real,
    /// Square root of the surface potential (C `sqrtPhis`).
    pub sqrt_phis: S,
    /// C `dsqrtPhis_dVb`.
    pub d_sqrt_phis_dvb: Real,
    /// Bulk-charge factor before the mobility gate term.
    pub abulk0: S,
    /// Its body derivative.
    pub d_abulk0_dvb: Real,
    /// Subthreshold slope factor.
    pub n: S,
    /// Its drain derivative.
    pub d_n_dvd: Real,
    /// Its body derivative.
    pub d_n_dvb: Real,
    /// Smoothed gate overdrive used by the DC current.
    pub vgsteff: S,
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
pub struct DcEvaluation<S = Real> {
    /// Drain current (C `here->BSIM3cd`).
    pub ids: S,
    /// Transconductance (C `here->BSIM3gm`).
    pub gm: Real,
    /// Output conductance (C `here->BSIM3gds`).
    pub gds: Real,
    /// Body transconductance (C `here->BSIM3gmbs`).
    pub gmbs: Real,
    /// Substrate current (C `here->BSIM3csub`).
    pub isub: S,
    /// Substrate-current derivatives (C `gbgs`, `gbds`, `gbbs`).
    pub gbgs: Real,
    /// Substrate-current drain derivative.
    pub gbds: Real,
    /// Substrate-current body derivative.
    pub gbbs: Real,
    /// Source junction current (C `here->BSIM3cbs`).
    pub cbs: S,
    /// Drain junction current (C `here->BSIM3cbd`).
    pub cbd: S,
    /// Source junction conductance, including gmin.
    pub gbs: Real,
    /// Drain junction conductance, including gmin.
    pub gbd: Real,
    /// Source-body voltage used, after mode selection.
    pub vbs: S,
    /// Drain-body voltage computed as `vbs - vds`.
    pub vbd: S,
    /// Gate-source voltage used, after mode selection.
    pub vgs: S,
    /// Drain-source voltage used, after mode selection.
    pub vds: S,
    /// Normal (+1) or inverse (-1) mode.
    pub mode: i8,
    /// Threshold voltage reported by `@m1[vth]` (C `here->BSIM3von`).
    pub von: S,
    /// Effective gate overdrive.
    pub vgsteff: S,
    /// Effective drain voltage.
    pub vdseff: S,
    /// Saturation voltage.
    pub vdsat: S,
    /// Bulk-charge factor.
    pub abulk: S,
    /// Effective mobility.
    pub ueff: S,
    /// Series resistance used.
    pub rds: S,
    /// C `here->BSIM3thetavth`.
    pub thetavth: S,
    /// C `here->BSIM3AbovVgst2Vtm`.
    pub abulk_over_vgst2vtm: S,
    /// Intermediates the charge model (`b3ld.c` C-V) reads back.
    pub cv: CvIntermediates<S>,
}

impl<S> CvIntermediates<S> {
    fn map_values<T>(self, mut map: impl FnMut(S) -> T) -> CvIntermediates<T> {
        CvIntermediates {
            vtm: self.vtm,
            cox: self.cox,
            tox: self.tox,
            phi: self.phi,
            vgs_eff: map(self.vgs_eff),
            d_vgs_eff_dvg: self.d_vgs_eff_dvg,
            vth: map(self.vth),
            d_vth_dvd: self.d_vth_dvd,
            d_vth_dvb: self.d_vth_dvb,
            vgst: map(self.vgst),
            vbseff: map(self.vbseff),
            d_vbseff_dvb: self.d_vbseff_dvb,
            phis: map(self.phis),
            d_phis_dvb: self.d_phis_dvb,
            sqrt_phis: map(self.sqrt_phis),
            d_sqrt_phis_dvb: self.d_sqrt_phis_dvb,
            abulk0: map(self.abulk0),
            d_abulk0_dvb: self.d_abulk0_dvb,
            n: map(self.n),
            d_n_dvd: self.d_n_dvd,
            d_n_dvb: self.d_n_dvb,
            vgsteff: map(self.vgsteff),
            d_vgsteff_dvg: self.d_vgsteff_dvg,
            d_vgsteff_dvd: self.d_vgsteff_dvd,
            d_vgsteff_dvb: self.d_vgsteff_dvb,
        }
    }
}

impl<S> DcEvaluation<S> {
    fn map_values<T>(self, mut map: impl FnMut(S) -> T) -> DcEvaluation<T> {
        DcEvaluation {
            ids: map(self.ids),
            gm: self.gm,
            gds: self.gds,
            gmbs: self.gmbs,
            isub: map(self.isub),
            gbgs: self.gbgs,
            gbds: self.gbds,
            gbbs: self.gbbs,
            cbs: map(self.cbs),
            cbd: map(self.cbd),
            gbs: self.gbs,
            gbd: self.gbd,
            vbs: map(self.vbs),
            vbd: map(self.vbd),
            vgs: map(self.vgs),
            vds: map(self.vds),
            mode: self.mode,
            von: map(self.von),
            vgsteff: map(self.vgsteff),
            vdseff: map(self.vdseff),
            vdsat: map(self.vdsat),
            abulk: map(self.abulk),
            ueff: map(self.ueff),
            rds: map(self.rds),
            thetavth: map(self.thetavth),
            abulk_over_vgst2vtm: map(self.abulk_over_vgst2vtm),
            cv: self.cv.map_values(map),
        }
    }
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

impl Bias {
    /// Raw normalized body/gate/drain coordinates, in that order.
    pub(super) fn differential(self) -> SpiceResult<Bias<Differential<3>>> {
        Ok(Bias {
            vbs: Differential::variable(0, Jet::constant(0, self.vbs))?,
            vgs: Differential::variable(1, Jet::constant(0, self.vgs))?,
            vds: Differential::variable(2, Jet::constant(0, self.vds))?,
        })
    }
}

/// Convert a raw physical derivative to the reference-oriented terminal
/// stamp coordinates. Reverse mode changes the local basis, not the
/// underlying junction orientation.
pub(super) fn mode_partials(law: &Differential<3>, mode: i8) -> [Real; 3] {
    let [body, gate, drain] = law
        .gradient
        .each_ref()
        .map(|g| g.as_ref().map_or(0., Jet::value));
    let drain = if mode >= 0 {
        drain
    } else {
        -drain - body - gate
    };
    [gate, drain, body]
}

impl DcEvaluation<Differential<3>> {
    fn physical_stamp(self) -> DcEvaluation {
        let [gm, gds, gmbs] = mode_partials(&self.ids, self.mode);
        let [gbgs, gbds, gbbs] = mode_partials(&self.isub, self.mode);
        let mut scalar = self.map_values(|law| law.value.value());
        scalar.gm = gm;
        scalar.gds = gds;
        scalar.gmbs = gmbs;
        scalar.gbgs = gbgs;
        scalar.gbds = gbds;
        scalar.gbbs = gbbs;
        scalar
    }
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
        self.evaluate_constitutive(bias, gmin)
    }

    pub(super) fn evaluate_physical(&self, bias: Bias, gmin: Real) -> SpiceResult<DcEvaluation> {
        Ok(self
            .evaluate_constitutive(bias.differential()?, gmin)?
            .physical_stamp())
    }

    /// The scalar load and physical Taylor evaluation share the entire
    /// constitutive value path. Scalar reference stamp partials are
    /// retained separately (in the original local coordinate convention);
    /// the differential values carry their exact physical partial series.
    pub(super) fn evaluate_constitutive<S: Constitutive>(
        &self,
        bias: Bias<S>,
        gmin: Real,
    ) -> SpiceResult<DcEvaluation<S>> {
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
        let vbd_in = bias.vbs.clone().add(&bias.vds.clone().scale(-1.));

        let junction = |sat, v, vjsm, isevjsm| {
            let continuation = (m.values.ijth.value != 0.).then_some((vjsm, isevjsm));
            bsim3_junction(v, sat, nvtm, continuation, gmin)
        };
        let (cbs, gbs) = junction(source_sat_current, bias.vbs.clone(), p.vjsm, p.isevjsm)?;
        let (cbd, gbd) = junction(drain_sat_current, vbd_in, p.vjdm, p.isevjdm)?;

        // Normal versus inverse mode. The inverse case mirrors gate/drain.
        let (mode, vds, vgs, vbs) = if bias.vds.compare(0.) == Some(Ordering::Less) {
            let drain = bias.vds.scale(-1.);
            (
                -1,
                drain.clone(),
                bias.vgs.add(&drain),
                bias.vbs.add(&drain),
            )
        } else {
            (1, bias.vds, bias.vgs, bias.vbs)
        };
        let gbs = gbs.primal();
        let gbd = gbd.primal();

        // Body-effect and depletion charge.
        let vbsc = p.derived.vbsc;
        let phi = p.derived.phi;
        let (vbseff, body_partial) = bsim3_body_limit(&vbs, vbsc)?;
        let dvbseff_dvb = body_partial.primal();
        let surface = surface_potential(&vbseff, phi, Some(p.derived.phis3))?;
        let phis = surface.potential;
        let sqrt_phis = surface.root;
        let dphis_dvb = surface.potential_partial.primal();
        let dsqrt_phis_dvb = surface.root_partial.primal();
        let xdep = sqrt_phis.clone().scale(p.derived.xdep0 / p.derived.sqrtphi);
        let dxdep_dvb = (p.derived.xdep0 / p.derived.sqrtphi) * dsqrt_phis_dvb;

        let leff = p.leff;
        let v0 = p.derived.vbi - phi;
        let short = channel_rolloff(
            &vbseff,
            &xdep,
            p.binned.dvt2,
            factor1,
            -0.5 * p.binned.dvt1 * leff,
            ThresholdRollOff::Bsim3,
        )?;
        let narrow = channel_rolloff(
            &vbseff,
            &xdep,
            p.binned.dvt2w,
            factor1,
            -0.5 * p.binned.dvt1w * p.weff * leff,
            ThresholdRollOff::Bsim3,
        )?;
        let body_depth = [[0., 0., 1.], [0., 0., dxdep_dvb]];
        let short_slopes = short.project(&body_depth);
        let narrow_slopes = narrow.project(&body_depth);
        let theta0 = short.value;
        let thetavth = theta0.clone().scale(p.binned.dvt0);

        let temp_ratio = (temp - tnom) / tnom;
        let length_ratio = p.binned.nlx / leff;
        let root_offset = length_ratio / ((1.0 + length_ratio).sqrt() + 1.0);
        let width_shift = tox * phi / (p.weff + p.binned.w0);
        let temperature_offset = (p.binned.kt1 + p.binned.kt1l / leff)
            .mul_add(temp_ratio, p.derived.k1ox * root_offset * p.derived.sqrtphi);
        let threshold = ThresholdParameters {
            fixed: p.binned.k3.mul_add(
                width_shift,
                p.binned.k1.mul_add(-p.derived.sqrtphi, type_ * p.vth0) + temperature_offset,
            ),
            surface: p.derived.k1ox,
            body: p.binned.kt2.mul_add(
                temp_ratio,
                p.binned.k3b.mul_add(width_shift, -p.derived.k2ox),
            ),
            short: -p.binned.dvt0 * v0,
            narrow: -p.binned.dvt0w * v0,
            eta0: p.binned.eta0,
            etab: p.binned.etab,
            theta0vb0: p.derived.theta0vb0,
        }
        .evaluate(&vbseff, &vds, &sqrt_phis, &theta0, &narrow.value)?;
        let [_, dvth_dvd, dvth_dvb] = threshold.project(&[
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 0., dsqrt_phis_dvb],
            short_slopes,
            narrow_slopes,
        ]);
        let vth = threshold.value;
        let von = vth.clone();

        let slope = SlopeParameters {
            depletion: p.binned.nfactor * EPS_SI,
            coupling: [p.binned.cdsc, p.binned.cdscb, p.binned.cdscd],
            interface: p.binned.cit,
            oxide: cox,
        }
        .evaluate(&vbseff, &vds, &xdep, &theta0)?;
        let [_, dn_dvd, dn_dvb] = slope.project(&[
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 0., dxdep_dvb],
            short_slopes,
        ]);
        let n = slope.value;

        let (vgs_eff, dvgseff_dvg) = poly_depletion(
            vgs.clone(),
            p.vfb + phi,
            p.binned.ngate,
            EPS_SI,
            cox,
            CHARGE_Q,
        )?;
        let dvgseff_dvg = dvgseff_dvg.primal();
        let vgst = vgs_eff.clone().add(&vth.clone().scale(-1.));

        // Effective gate overdrive (Vgsteff), with C's three regimes.
        let (vgsteff, voltage_partial, factor_partial) =
            bsim3_overdrive(&vgst, &n, vtm, p.binned.voff, cox / p.derived.cdep0)?;
        let voltage_partial = voltage_partial.primal();
        let factor_partial = factor_partial.primal();
        let dvgsteff_dvg = voltage_partial * dvgseff_dvg;
        let dvgsteff_dvd = factor_partial.mul_add(dn_dvd, -voltage_partial * dvth_dvd);
        let dvgsteff_dvb = factor_partial.mul_add(dn_dvb, -voltage_partial * dvth_dvb);

        // Effective width and source/drain resistance.
        let root_shift = sqrt_phis.clone().offset(-p.derived.sqrtphi);
        let raw_width = root_shift
            .clone()
            .scale(-2. * p.binned.dwb)
            .add(&vgsteff.clone().scale(-2. * p.binned.dwg))
            .offset(p.weff);
        let (weff, width_partial) = rational_floor(raw_width, 2e-8, 0., 1e8)?;
        let dweff_dvg = -2. * p.binned.dwg * width_partial.primal();
        let dweff_dvb = -2. * p.binned.dwb * dsqrt_phis_dvb * width_partial.primal();
        let resistance_argument = root_shift
            .scale(p.binned.prwb)
            .add(&vgsteff.clone().scale(p.binned.prwg));
        let (resistance_factor, resistance_partial) =
            rational_floor(resistance_argument, -0.9, 1., 20.)?;
        let rds = resistance_factor.scale(p.thermal.rds0);
        let drds_dvg = p.thermal.rds0 * p.binned.prwg * resistance_partial.primal();
        let drds_dvb =
            p.thermal.rds0 * p.binned.prwb * dsqrt_phis_dvb * resistance_partial.primal();

        // Bulk-charge factor. Preserve the reference stamp's local
        // derivatives before the effective-body chain is applied.
        let body_effect = sqrt_phis
            .constant(0.5 * p.derived.k1ox)
            .divide(&sqrt_phis)?;
        let junction_depth = xdep.clone().scale(p.binned.xj).sqrt()?;
        let length = junction_depth.clone().scale(2.).offset(leff);
        let length_ratio = xdep.constant(leff).divide(&length)?;
        let geometry = length_ratio
            .clone()
            .scale(p.binned.a0)
            .offset(p.binned.b0 / (p.weff + p.binned.b1));
        let gate_coupling = length_ratio
            .clone()
            .multiply(&length_ratio)
            .multiply(&length_ratio)
            .scale(p.binned.ags * p.binned.a0);
        let raw_bulk0 = body_effect.clone().mul_add(&geometry, &vds.constant(1.));
        let gate_slope = body_effect.clone().scale(-1.).multiply(&gate_coupling);
        let raw_bulk = gate_slope.clone().mul_add(&vgsteff, &raw_bulk0);
        let body_effect_partial = -body_effect.primal() / sqrt_phis.primal() * dsqrt_phis_dvb;
        let ratio_partial = -junction_depth.primal() / length.primal() / xdep.primal() * dxdep_dvb;
        let raw_bulk0_partial = geometry.primal().mul_add(
            body_effect_partial,
            body_effect.primal() * p.binned.a0 * length_ratio.primal() * ratio_partial,
        );
        let raw_bulk_partial = (gate_coupling.primal() * vgsteff.primal()).mul_add(
            -(3. * body_effect.primal()).mul_add(ratio_partial, body_effect_partial),
            raw_bulk0_partial,
        );
        let (bulk0, bulk0_partial) = rational_floor(raw_bulk0, 0.1, 0., 20.)?;
        let (bulk, bulk_partial) = rational_floor(raw_bulk, 0.1, 0., 20.)?;
        let (keta, keta_partial) =
            reciprocal_floor(vbseff.clone().scale(p.binned.keta), -0.9, 20.)?;
        let dketa_dvb = p.binned.keta * keta_partial.primal();
        let dabulk0_dvb = bulk0.primal().mul_add(
            dketa_dvb,
            raw_bulk0_partial * bulk0_partial.primal() * keta.primal(),
        );
        let dabulk_dvb = bulk.primal().mul_add(
            dketa_dvb,
            raw_bulk_partial * bulk_partial.primal() * keta.primal(),
        );
        let dabulk_dvg = gate_slope.primal() * bulk_partial.primal() * keta.primal();
        let abulk0 = bulk0.multiply(&keta);
        let abulk = bulk.multiply(&keta);

        // Mobility, by MOBMOD. The same field and coefficient equations
        // are used by both arithmetic instantiations.
        let mobility_field = if m.values.mobmod.value == 2 {
            vgsteff.clone().scale(tox.recip())
        } else {
            vgsteff
                .clone()
                .add(&vth.clone().scale(2.))
                .scale(tox.recip())
        };
        let (raw_mobility, raw_gate, raw_drain, raw_body) =
            if m.values.mobmod.value == 1 || m.values.mobmod.value == 2 {
                let coefficient = vbseff.clone().scale(p.binned.uc).offset(p.binned.ua);
                let raw = mobility_field
                    .clone()
                    .multiply(&mobility_field.clone().scale(p.binned.ub).add(&coefficient));
                let gate =
                    (2. * p.binned.ub).mul_add(mobility_field.primal(), coefficient.primal()) / tox;
                if m.values.mobmod.value == 1 {
                    (
                        raw,
                        gate,
                        gate * 2. * dvth_dvd,
                        p.binned
                            .uc
                            .mul_add(mobility_field.primal(), gate * 2. * dvth_dvb),
                    )
                } else {
                    (raw, gate, 0., p.binned.uc * mobility_field.primal())
                }
            } else {
                let body_factor = vbseff.clone().scale(p.binned.uc).offset(1.);
                let field_factor = mobility_field.clone().multiply(
                    &mobility_field
                        .clone()
                        .scale(p.binned.ub)
                        .offset(p.binned.ua),
                );
                let gate = (2. * p.binned.ub).mul_add(mobility_field.primal(), p.binned.ua)
                    * body_factor.primal()
                    / tox;
                let body = p
                    .binned
                    .uc
                    .mul_add(field_factor.primal(), gate * 2. * dvth_dvb);
                (
                    field_factor.multiply(&body_factor),
                    gate,
                    gate * 2. * dvth_dvd,
                    body,
                )
            };
        let (mobility_denominator, mobility_partial) = rational_floor(raw_mobility, -0.8, 1., 10.)?;
        let ueff = vds.constant(p.u0temp).divide(&mobility_denominator)?;
        let mobility_scale =
            -ueff.primal() / mobility_denominator.primal() * mobility_partial.primal();
        let [dueff_dvg, dueff_dvd, dueff_dvb] =
            [raw_gate, raw_drain, raw_body].map(|slope| slope * mobility_scale);

        // Saturation drain voltage.
        let wvcox = weff.clone().scale(p.thermal.vsattemp * cox);
        let wvcoxrds = wvcox.clone().multiply(&rds);
        let esat = vds.constant(2. * p.thermal.vsattemp).divide(&ueff)?;
        let esatl = esat.clone().scale(leff);
        let field_scale = -esatl.primal() / ueff.primal();
        let [desatl_dvg, desatl_dvd, desatl_dvb] =
            [dueff_dvg, dueff_dvd, dueff_dvb].map(|slope| slope * field_scale);
        let (lambda, dlambda_dvg) = saturation_shape(&vgsteff, p.binned.a1, p.binned.a2)?;
        let vgst2vtm = vgsteff.clone().offset(2. * vtm);
        let abulk_over_vgst2vtm = abulk.clone().divide(&vgst2vtm)?;
        let resistance_slopes = [
            wvcox
                .primal()
                .mul_add(drds_dvg, wvcoxrds.primal() * dweff_dvg / weff.primal()),
            0.,
            wvcox
                .primal()
                .mul_add(drds_dvb, wvcoxrds.primal() * dweff_dvb / weff.primal()),
        ];
        let bulk_slopes = [dabulk_dvg, 0., dabulk_dvb];
        let field_slopes = [desatl_dvg, desatl_dvd, desatl_dvb];
        let shape_slopes = [dlambda_dvg.primal(), 0., 0.];
        let saturation = saturation_voltage(&vgst2vtm, &abulk, &esatl, &wvcoxrds, &lambda)?;
        let saturation_slopes = saturation.project(&[
            [1., 0., 0.],
            bulk_slopes,
            field_slopes,
            resistance_slopes,
            shape_slopes,
        ]);
        let [dvdsat_dvg, dvdsat_dvd, dvdsat_dvb] = saturation_slopes;
        let vdsat = saturation.value;

        let smoothing = smooth_drain(vdsat.clone(), vds.clone(), p.binned.delta)?;
        let vdseff = if smoothing.effective.primal() > vds.primal() {
            vds.clone()
        } else {
            smoothing.effective
        };
        let dvdseff_dvg = smoothing.saturation_partial.primal() * dvdsat_dvg;
        let dvdseff_dvb = smoothing.saturation_partial.primal() * dvdsat_dvb;
        let dvdseff_dvd = smoothing
            .saturation_partial
            .primal()
            .mul_add(dvdsat_dvd, smoothing.drain_partial.primal());
        let diff_vds = smoothing.drain_excess;

        // Early voltage: saturation, CLM and DIBL components.
        let early =
            saturation_early_voltage(&vgsteff, &abulk, &vdsat, &esatl, &wvcoxrds, &lambda, vtm)?;
        let [dvasat_dvg, dvasat_dvd, dvasat_dvb] = early.project(&[
            [1., 0., 0.],
            bulk_slopes,
            saturation_slopes,
            field_slopes,
            resistance_slopes,
            shape_slopes,
        ]);
        let vasat = early.value;
        let (vaclm, dvaclm_dvg, dvaclm_dvd, dvaclm_dvb) =
            if p.binned.pclm > 0. && diff_vds.compare(1e-10) == Some(Ordering::Greater) {
                let factor = vds
                    .constant((p.binned.pclm * p.derived.litl).recip())
                    .divide(&abulk)?;
                let over_field = vgsteff.clone().divide(&esatl)?;
                let numerator = abulk.clone().add(&over_field).scale(leff);
                let clm_slope = factor.clone().multiply(&numerator);
                let vaclm = clm_slope.clone().multiply(&diff_vds);
                let factor_partial = -factor.primal() / abulk.primal();
                let numerator_dvg = leff
                    * (over_field.primal().mul_add(-desatl_dvg, 1.) / esatl.primal() + dabulk_dvg);
                let numerator_dvb =
                    leff * (dabulk_dvb - over_field.primal() * desatl_dvb / esatl.primal());
                // b3ld.c's scalar reference stamp divides by Esat here.
                let numerator_dvd = -over_field.primal() * desatl_dvd / esat.primal();
                let dvg = clm_slope.primal().mul_add(
                    -dvdseff_dvg,
                    diff_vds.primal()
                        * factor.primal().mul_add(
                            numerator_dvg,
                            numerator.primal() * factor_partial * dabulk_dvg,
                        ),
                );
                let dvd = clm_slope.primal().mul_add(
                    1. - dvdseff_dvd,
                    factor.primal() * numerator_dvd * diff_vds.primal(),
                );
                let dvb = clm_slope.primal().mul_add(
                    -dvdseff_dvb,
                    diff_vds.primal()
                        * factor.primal().mul_add(
                            numerator_dvb,
                            numerator.primal() * factor_partial * dabulk_dvb,
                        ),
                );
                (vaclm, dvg, dvd, dvb)
            } else {
                (vds.constant(MAX_EXP), 0., 0., 0.)
            };

        let (vadibl, dvadibl_dvg, dvadibl_dvd, dvadibl_dvb) = if p.derived.thetarout > 0. {
            let early = dibl_early_voltage(
                &vgst2vtm,
                &abulk,
                &vdsat,
                &vbseff,
                p.derived.thetarout,
                p.binned.pdiblcb,
            )?;
            let [dg, dd, db] =
                early.project(&[[1., 0., 0.], bulk_slopes, saturation_slopes, [0., 0., 1.]]);
            (early.value, dg, dd, db)
        } else {
            (vds.constant(MAX_EXP), 0., 0., 0.)
        };

        let pvag_argument = vgsteff.clone().scale(p.binned.pvag).divide(&esatl)?;
        let pvag_value = pvag_argument.primal();
        let (pvag, pvag_partial) = rational_floor(pvag_argument, -0.9, 1., 20.)?;
        let pvag_slope = pvag_partial.primal();
        let dt0_dvg = p.binned.pvag / esatl.primal()
            * (1. - vgsteff.primal() * desatl_dvg / esatl.primal())
            * pvag_slope;
        let dt0_dvb = -pvag_value * desatl_dvb / esatl.primal() * pvag_slope;
        let dt0_dvd = -pvag_value * desatl_dvd / esatl.primal() * pvag_slope;
        let early_sum = vaclm.clone().add(&vadibl);
        let combined = vaclm.clone().divide(&early_sum)?.multiply(&vadibl);
        let clm_weight = (vadibl.primal() / early_sum.primal()).powi(2);
        let dibl_weight = (vaclm.primal() / early_sum.primal()).powi(2);
        let combine = |clm: Real, dibl: Real| clm_weight.mul_add(clm, dibl_weight * dibl);
        let va = pvag.clone().mul_add(&combined, &vasat);
        let early_slope = |pvag_derivative: Real, clm: Real, dibl: Real, saturation: Real| {
            pvag.primal().mul_add(
                combine(clm, dibl),
                combined.primal().mul_add(pvag_derivative, saturation),
            )
        };
        let dva_dvg = early_slope(dt0_dvg, dvaclm_dvg, dvadibl_dvg, dvasat_dvg);
        let dva_dvd = early_slope(dt0_dvd, dvaclm_dvd, dvadibl_dvd, dvasat_dvd);
        let dva_dvb = early_slope(dt0_dvb, dvaclm_dvb, dvadibl_dvb, dvasat_dvb);

        let (vascbe, slope) = substrate_early_voltage(
            &diff_vds,
            leff,
            p.binned.pscbe1 * p.derived.litl,
            p.binned.pscbe2,
            p.binned.pscbe2 > 0.,
        )?;
        let dvascbe_dvg = -slope.primal() * dvdseff_dvg;
        let dvascbe_dvd = slope.primal() * (1. - dvdseff_dvd);
        let dvascbe_dvb = -slope.primal() * dvdseff_dvb;

        // Drain current and its derivatives.
        let coxwovl = weff.clone().scale(cox / leff);
        let beta = ueff.clone().multiply(&coxwovl);
        let dbeta_dvg = coxwovl
            .primal()
            .mul_add(dueff_dvg, beta.primal() * dweff_dvg / weff.primal());
        let dbeta_dvd = coxwovl.primal() * dueff_dvd;
        let dbeta_dvb = coxwovl
            .primal()
            .mul_add(dueff_dvb, beta.primal() * dweff_dvb / weff.primal());
        let channel = channel_conductance(&vgsteff, &abulk, &vdseff, &esatl, &beta, vtm)?;
        let [dg, dd, db] = channel.project(&[
            [1., 0., 0.],
            bulk_slopes,
            [dvdseff_dvg, dvdseff_dvd, dvdseff_dvb],
            field_slopes,
            [dbeta_dvg, dbeta_dvd, dbeta_dvb],
        ]);
        let series = series_conductance(&channel.value, &rds)?;
        let [sg, sd, sb] = series.project(&[[dg, dd, db], [drds_dvg, 0., drds_dvb]]);
        let idl = series.value.clone().multiply(&vdseff);
        let didl_dvg = series
            .value
            .primal()
            .mul_add(dvdseff_dvg, sg * vdseff.primal());
        let didl_dvd = series
            .value
            .primal()
            .mul_add(dvdseff_dvd, sd * vdseff.primal());
        let didl_dvb = series
            .value
            .primal()
            .mul_add(dvdseff_dvb, sb * vdseff.primal());
        let early_ratio = diff_vds.clone().divide(&va)?;
        let early_factor = early_ratio.clone().offset(1.);
        let idsa = idl.clone().multiply(&early_factor);
        let corrected = |derivative: Real, drain_derivative: Real, early_derivative: Real| {
            early_factor.primal().mul_add(
                derivative,
                idl.primal()
                    * early_ratio
                        .primal()
                        .mul_add(-early_derivative, drain_derivative)
                    / va.primal(),
            )
        };
        let didsa_dvg = corrected(didl_dvg, -dvdseff_dvg, dva_dvg);
        let didsa_dvd = corrected(didl_dvd, 1. - dvdseff_dvd, dva_dvd);
        let didsa_dvb = corrected(didl_dvb, -dvdseff_dvb, dva_dvb);

        let substrate_ratio = diff_vds.clone().divide(&vascbe)?;
        let substrate_factor = substrate_ratio.clone().offset(1.);
        let ids = idsa.clone().multiply(&substrate_factor);
        let corrected = |derivative: Real, drain_derivative: Real, early_derivative: Real| {
            substrate_factor.primal().mul_add(
                derivative,
                idsa.primal()
                    * substrate_ratio
                        .primal()
                        .mul_add(-early_derivative, drain_derivative)
                    / vascbe.primal(),
            )
        };
        let gm_local = corrected(didsa_dvg, -dvdseff_dvg, dvascbe_dvg);
        let gds = gm_local.mul_add(
            dvgsteff_dvd,
            corrected(didsa_dvd, 1. - dvdseff_dvd, dvascbe_dvd),
        );
        let gmbs = gm_local.mul_add(
            dvgsteff_dvb,
            corrected(didsa_dvb, -dvdseff_dvb, dvascbe_dvb),
        ) * dvbseff_dvb;
        let gm = gm_local * dvgsteff_dvg;

        // Substrate current.
        let alpha = p.binned.alpha1.mul_add(leff, p.binned.alpha0);
        let (impact, slope) = impact_ionization(&diff_vds, alpha / leff, p.binned.beta0)?;
        let isub = impact.clone().multiply(&idsa);
        let corrected = |derivative: Real, drain_derivative: Real| {
            idsa.primal().mul_add(
                slope.primal() * drain_derivative,
                impact.primal() * derivative,
            )
        };
        let gbgs_local = corrected(didsa_dvg, -dvdseff_dvg);
        let gbds = gbgs_local.mul_add(dvgsteff_dvd, corrected(didsa_dvd, 1. - dvdseff_dvd));
        let gbbs =
            gbgs_local.mul_add(dvgsteff_dvb, corrected(didsa_dvb, -dvdseff_dvb)) * dvbseff_dvb;
        let gbgs = gbgs_local * dvgsteff_dvg;

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
            vbd: vbs.clone().add(&vds.clone().scale(-1.)),
            vbs,
            vgs,
            vds,
            mode,
            von,
            vgsteff: vgsteff.clone(),
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
            ("ids", result.ids.primal()),
            ("gm", result.gm),
            ("gds", result.gds),
            ("gmbs", result.gmbs),
            ("von", result.von.primal()),
            ("ueff", result.ueff.primal()),
            ("vdseff", result.vdseff.primal()),
            ("cbs", result.cbs.primal()),
            ("cbd", result.cbd.primal()),
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
