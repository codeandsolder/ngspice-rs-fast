//! Intrinsic electrostatics and channel transport from b4ld.c.
#![expect(
    non_snake_case,
    reason = "C-translated equation symbols identify their upstream terms"
)]
#![expect(
    clippy::similar_names,
    reason = "C equation names preserve the reference distinctions"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "Remaining translated product-rule sums and branch expressions have no demonstrated cancellation defect; indiscriminate software FMA would add portable-target cost. See BSIM_CHANNEL_NUMERICS.md."
)]
#![expect(
    clippy::float_cmp,
    reason = "C branch predicates compare floating point exactly"
)]
#![expect(
    clippy::too_many_lines,
    reason = "preserve cohesive C physics equation sequence"
)]
use super::bias::LoadBias;
use super::{
    CHARGE, Charge_q, DEXP, EPS0, EPSSI, EXP_THRESHOLD, Instance, LoadCkt, MAX_EXP, MIN_EXP, MM,
    Model, SizeParam, smooth_drain, smooth_positive, sqrt,
};

use crate::devices::bsim_physics::{
    SlopeParameters, ThresholdParameters, ThresholdRollOff, bsim4_body_limit, channel_conductance,
    channel_rolloff, dibl_early_voltage, effective_overdrive, impact_ionization, poly_depletion,
    rational_floor, reciprocal_floor, saturation_early_voltage, saturation_shape,
    saturation_voltage, series_conductance, substrate_early_voltage, surface_potential,
};
use crate::maths::taylor::{Constitutive, Local};
use crate::primitives::SpiceResult;
use std::cmp::Ordering;

/// Body potential, threshold, effective gate bias and their independent derivatives.
pub(super) struct Electrostatics<S = f64> {
    pub(super) Leff: f64,
    pub(super) Lpe_Vb: f64,
    pub(super) Phis: S,
    pub(super) Vbs: S,
    pub(super) Vbseff: S,
    pub(super) Vds: S,
    pub(super) Vgs: S,
    pub(super) Vgs_eff: S,
    pub(super) vgs_eff_raw: S,
    pub(super) vgd_eff_raw: S,
    pub(super) Vgst: S,
    pub(super) Vgsteff: S,
    pub(super) Vth: S,
    pub(super) Vth_NarrowW: f64,
    pub(super) Vtm: f64,
    pub(super) Vtm0: f64,
    pub(super) Xdep: S,
    pub(super) dPhis_dVb: f64,
    pub(super) dVbseff_dVb: f64,
    pub(super) dVgs_eff_dVg: f64,
    pub(super) dVgsteff_dVb: f64,
    pub(super) dVgsteff_dVd: f64,
    pub(super) dVgsteff_dVg: f64,
    pub(super) dVth_dVb: f64,
    pub(super) dVth_dVd: f64,
    pub(super) dXdep_dVb: f64,
    pub(super) dn_dVb: f64,
    pub(super) dn_dVd: f64,
    pub(super) dsqrtPhis_dVb: f64,
    pub(super) epsrox: f64,
    pub(super) epssub: f64,
    pub(super) n: S,
    pub(super) sqrtPhis: S,
    pub(super) toxe: f64,
}

/// Unscaled channel current, effective drain voltage, bulk charge factor and transport quantities used by leakage/C-V.
pub(super) struct Transport<S = f64> {
    pub(super) Abulk0_Q: S,
    pub(super) Coxeff: S,
    pub(super) Rds: S,
    pub(super) Vdseff: S,
    pub(super) beta: S,
    pub(super) unscaled_drain_current: S,
    pub(super) dAbulk0_Q_dVb: f64,
    pub(super) dVdseff_dVb: f64,
    pub(super) dVdseff_dVd: f64,
    pub(super) dVdseff_dVg: f64,
    pub(super) gche: S,
    pub(super) substrate_current: S,
    pub(super) gate_conductance: S,
    pub(super) source_conductance: S,
    pub(super) drain_conductance: S,
}

/// Threshold and effective gate-voltage equations, in the reference orientation.
pub(super) fn electrostatics(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    ckt: &dyn LoadCkt,
    bias: &LoadBias,
) -> SpiceResult<Electrostatics> {
    electrostatics_constitutive(
        model,
        here,
        pParam,
        ckt.temp(),
        bias.vbs,
        bias.vgs,
        bias.vds,
    )
}

/// Exact electrostatics for both ordinary reference loading and physical jets.
/// Instance fields record only the primal operating point, never a derivative
/// or an alternate physical value path.
pub(super) fn electrostatics_constitutive<S: Constitutive>(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    temperature: f64,
    vbs: S,
    vgs: S,
    vds: S,
) -> SpiceResult<Electrostatics<S>> {
    let vbd = vbs.clone().add(&vds.clone().scale(-1.));
    let vgd = vgs.clone().add(&vds.clone().scale(-1.));
    let (Vds, Vgs, Vbs) = if vds.compare(0.) == Some(Ordering::Less) {
        here.mode = -1;
        (vds.scale(-1.), vgd.clone(), vbd)
    } else {
        here.mode = 1;
        (vds, vgs.clone(), vbs)
    };
    let (epsrox, toxe, epssub) = if model.mtrlMod != 0 {
        (3.9, model.eot, EPS0 * model.epsrsub)
    } else {
        (model.epsrox, model.toxe, EPSSI)
    };
    let (Vbseff, body_partial) = bsim4_body_limit(&Vbs, here.vbsc, pParam.phi)?;
    let dVbseff_dVb = body_partial.primal();
    let surface = surface_potential(&Vbseff, pParam.phi, None)?;
    let Phis = surface.potential;
    let sqrtPhis = surface.root;
    let dPhis_dVb = surface.potential_partial.primal();
    let dsqrtPhis_dVb = surface.root_partial.primal();
    let depth_factor = pParam.Xdep0 / pParam.sqrtPhi;
    let Xdep = sqrtPhis.clone().scale(depth_factor);
    let dXdep_dVb = depth_factor * dsqrtPhis_dVb;
    let Leff = pParam.leff;
    let Vtm = model.vtm;
    let Vtm0 = model.vtm0;
    let V0 = pParam.vbi - pParam.phi;
    let short = channel_rolloff(
        &Vbseff,
        &Xdep,
        pParam.dvt2,
        model.factor1,
        pParam.dvt1 * Leff,
        ThresholdRollOff::Bsim4,
    )?;
    let narrow = channel_rolloff(
        &Vbseff,
        &Xdep,
        pParam.dvt2w,
        model.factor1,
        pParam.dvt1w * pParam.weff * Leff,
        ThresholdRollOff::Bsim4,
    )?;
    let body_depth = [[0., 0., 1.], [0., 0., dXdep_dVb]];
    let short_slopes = short.project(&body_depth);
    let narrow_slopes = narrow.project(&body_depth);
    let Theta0 = short.value;
    here.thetavth = pParam.dvt0 * Theta0.primal();
    let TempRatio = temperature / model.tnom - 1.;
    let length_ratio = pParam.lpe0 / Leff;
    let root_offset = length_ratio / ((1. + length_ratio).sqrt() + 1.);
    let temperature_offset = (pParam.kt1 + pParam.kt1l / Leff)
        .mul_add(TempRatio, pParam.k1ox * root_offset * pParam.sqrtPhi);
    let Vth_NarrowW = toxe * pParam.phi / (pParam.weff + pParam.w0);
    let Lpe_Vb = sqrt(1. + pParam.lpeb / Leff);
    let threshold = ThresholdParameters {
        fixed: pParam.k3.mul_add(
            Vth_NarrowW,
            (pParam.k1 * Lpe_Vb).mul_add(-pParam.sqrtPhi, f64::from(model.r#type) * here.vth0)
                + temperature_offset,
        ),
        surface: pParam.k1ox * Lpe_Vb,
        body: pParam
            .kt2
            .mul_add(TempRatio, pParam.k3b.mul_add(Vth_NarrowW, -here.k2ox)),
        short: -pParam.dvt0 * V0,
        narrow: -pParam.dvt0w * V0,
        eta0: here.eta0,
        etab: pParam.etab,
        theta0vb0: pParam.theta0vb0,
    }
    .evaluate(&Vbseff, &Vds, &sqrtPhis, &Theta0, &narrow.value)?;
    let [_, mut dVth_dVd, mut dVth_dVb] = threshold.project(&[
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 0., dsqrtPhis_dVb],
        short_slopes,
        narrow_slopes,
    ]);
    let mut Vth = threshold.value;
    here.nstar = Vtm / Charge_q * (model.coxe + epssub / Xdep.primal() + pParam.cit);
    let slope = SlopeParameters {
        depletion: pParam.nfactor * epssub,
        coupling: [pParam.cdsc, pParam.cdscb, pParam.cdscd],
        interface: pParam.cit,
        oxide: model.coxe,
    }
    .evaluate(&Vbseff, &Vds, &Xdep, &Theta0)?;
    let [_, dn_dVd, dn_dVb] = slope.project(&[
        [0., 0., 1.],
        [0., 1., 0.],
        [0., 0., dXdep_dVb],
        short_slopes,
    ]);
    let n = slope.value;
    if pParam.dvtp0 > 0. {
        let exponent = Vds.clone().scale(-pParam.dvtp1);
        let (factor, factor_partial) = if exponent.compare(-EXP_THRESHOLD) == Some(Ordering::Less) {
            (exponent.constant(MIN_EXP), 0.)
        } else {
            let value = exponent.exp();
            let partial = -pParam.dvtp1 * value.primal();
            (value, partial)
        };
        let length_ratio = factor.offset(1.).scale(pParam.dvtp0 / Leff);
        let thermal = if model.tempMod < 2 { Vtm } else { Vtm0 };
        let shift = length_ratio.clone().ln_1p()?.scale(-thermal);
        let shift_partial =
            -thermal * pParam.dvtp0 * factor_partial / (Leff * (1. + length_ratio.primal()));
        dVth_dVd -= dn_dVd * shift.primal() + n.primal() * shift_partial;
        dVth_dVb -= shift.primal() * dn_dVb;
        Vth = Vth.add(&n.clone().multiply(&shift).scale(-1.));
    }
    if pParam.dvtp4 != 0. && pParam.dvtp2factor != 0. {
        let (factor, partial) = DEXP(Vds.clone().scale(2. * pParam.dvtp4));
        let den = factor.clone().offset(1.);
        let shift = factor.offset(-1.).divide(&den)?.scale(pParam.dvtp2factor);
        dVth_dVd -=
            pParam.dvtp2factor * pParam.dvtp4 * 4. * partial.primal() / den.primal().powi(2);
        Vth = Vth.add(&shift.scale(-1.));
    }
    here.von = Vth.primal();
    let poly_threshold = here.vfb + pParam.phi;
    let poly_permittivity = if model.mtrlMod == 0 {
        EPSSI
    } else {
        model.epsrgate * EPS0
    };
    let (vgs_eff, dvgs_eff_dvg) = poly_depletion(
        vgs,
        poly_threshold,
        pParam.ngate,
        poly_permittivity,
        model.coxe,
        CHARGE,
    )?;
    let (vgd_eff, dvgd_eff_dvg) = poly_depletion(
        vgd,
        poly_threshold,
        pParam.ngate,
        poly_permittivity,
        model.coxe,
        CHARGE,
    )?;
    here.vgs_eff = vgs_eff.primal();
    here.vgd_eff = vgd_eff.primal();
    here.dvgs_eff_dvg = dvgs_eff_dvg.primal();
    here.dvgd_eff_dvg = dvgd_eff_dvg.primal();
    let (Vgs_eff, dVgs_eff_dVg) = if here.mode > 0 {
        (vgs_eff.clone(), here.dvgs_eff_dvg)
    } else {
        (vgd_eff.clone(), here.dvgd_eff_dvg)
    };
    let Vgst = Vgs_eff.clone().add(&Vth.clone().scale(-1.));
    let (Vgsteff, voltage_partial, factor_partial) = effective_overdrive(
        &Vgst,
        &n,
        Vtm,
        pParam.mstar,
        pParam.voffcbn,
        model.coxe / pParam.cdep0,
    )?;
    here.Vgsteff = Vgsteff.primal();
    let dVgsteff_dVg = voltage_partial.primal() * dVgs_eff_dVg;
    let dVgsteff_dVd = factor_partial
        .primal()
        .mul_add(dn_dVd, -voltage_partial.primal() * dVth_dVd);
    let dVgsteff_dVb = factor_partial
        .primal()
        .mul_add(dn_dVb, -voltage_partial.primal() * dVth_dVb);
    Ok(Electrostatics {
        Leff,
        Lpe_Vb,
        Phis,
        Vbs,
        Vbseff,
        Vds,
        Vgs,
        Vgs_eff,
        vgs_eff_raw: vgs_eff,
        vgd_eff_raw: vgd_eff,
        Vgst,
        Vgsteff,
        Vth,
        Vth_NarrowW,
        Vtm,
        Vtm0,
        Xdep,
        dPhis_dVb,
        dVbseff_dVb,
        dVgs_eff_dVg,
        dVgsteff_dVb,
        dVgsteff_dVd,
        dVgsteff_dVg,
        dVth_dVb,
        dVth_dVd,
        dXdep_dVb,
        dn_dVb,
        dn_dVd,
        dsqrtPhis_dVb,
        epsrox,
        epssub,
        n,
        sqrtPhis,
        toxe,
    })
}

/// Mobility degradation, before the rational denominator continuation.
/// The local coordinates are effective gate overdrive, threshold, and body
/// potential. Reference load partials use these coordinates, whereas physical
/// Taylor arithmetic differentiates the same value law through terminal bias.
fn mobility_degradation<S: Constitutive>(
    model: &Model,
    here: &Instance,
    p: &SizeParam,
    electro: &Electrostatics<S>,
) -> SpiceResult<(S, [f64; 3])> {
    let v = &electro.Vgsteff;
    let threshold = &electro.Vth;
    let body = &electro.Vbseff;
    let tox = electro.toxe;
    let material_shift = if model.mtrlMod != 0 && model.mtrlCompatMod == 0 {
        2. * f64::from(model.r#type) * (model.phig - model.easub - 0.5 * model.Eg0 + 0.45)
    } else {
        0.
    };
    let ua = p.ua;
    let ub = p.ub;
    let uc = p.uc;
    let ud = p.ud;
    if model.mobMod == 3 || !(0..=6).contains(&model.mobMod) {
        let field_scale = 1e-8 / tox / 6.;
        let field = v.clone().offset(here.vtfbphi1).scale(field_scale);
        let power = field.clone().ln()?.scale(p.eu).exp();
        let coefficient = body.clone().scale(uc).offset(ua);
        let denominator = v.clone().scale(0.5 / p.VgsteffVth).offset(0.5);
        let scattering = denominator.clone().ln()?.scale(-p.ucs).exp().scale(ud);
        let gate = coefficient.primal() * power.primal() * p.eu * field_scale / field.primal()
            - 0.5 * p.ucs * scattering.primal() / denominator.primal() / p.VgsteffVth;
        let bulk = power.primal() * uc;
        return Ok((
            power.multiply(&coefficient).add(&scattering),
            [gate, 0., bulk],
        ));
    }
    let fixed_threshold = model.mobMod >= 4;
    let family = if fixed_threshold {
        model.mobMod - 4
    } else {
        model.mobMod
    };
    let coulomb_threshold = if fixed_threshold {
        v.constant(here.vtfbphi1)
    } else {
        threshold.clone()
    };
    let softened = coulomb_threshold
        .clone()
        .multiply(&coulomb_threshold)
        .offset(0.0001)
        .sqrt()?;
    let screening = v.clone().add(&softened.clone().scale(2.));
    let scaled_threshold = coulomb_threshold.clone().scale(tox).divide(&screening)?;
    let coulomb = scaled_threshold
        .clone()
        .multiply(&scaled_threshold)
        .scale(ud);
    let coulomb_gate = -2. * coulomb.primal() / screening.primal();
    let coulomb_threshold_partial = if fixed_threshold {
        0.
    } else {
        let threshold = coulomb_threshold.primal();
        let ratio = tox / screening.primal();
        2. * (ud * ratio * ratio * threshold + coulomb_gate * threshold / softened.primal())
    };
    let (field_term, gate_partial, threshold_partial, body_partial) = if family == 2 {
        let field = v.clone().offset(here.vtfbphi1).scale(tox.recip());
        let power = field.clone().ln()?.scale(p.eu).exp();
        let coefficient = body.clone().scale(uc).offset(ua);
        let gate_partial = coefficient.primal() * power.primal() * p.eu / field.primal() / tox;
        let body_partial = power.primal() * uc;
        (power.multiply(&coefficient), gate_partial, 0., body_partial)
    } else {
        let field = if fixed_threshold {
            v.clone().offset(here.vtfbphi1 - material_shift)
        } else {
            v.clone()
                .add(&threshold.clone().scale(2.))
                .offset(-material_shift)
        }
        .scale(tox.recip());
        let (term, gate, bulk) = if family == 0 {
            let coefficient = body.clone().scale(uc).offset(ua);
            let slope = (coefficient.primal() + 2. * ub * field.primal()) / tox;
            let bulk = uc * field.primal();
            let term = field.clone().multiply(&field.scale(ub).add(&coefficient));
            (term, slope, bulk)
        } else {
            let coefficient = body.clone().scale(uc).offset(1.);
            let field_term = field.clone().multiply(&field.clone().scale(ub).offset(ua));
            let slope = (ua + 2. * ub * field.primal()) * coefficient.primal() / tox;
            let bulk = uc * field_term.primal();
            (field_term.multiply(&coefficient), slope, bulk)
        };
        (
            term,
            gate,
            if fixed_threshold { 0. } else { 2. * gate },
            bulk,
        )
    };
    Ok((
        field_term.add(&coulomb),
        [
            gate_partial + coulomb_gate,
            threshold_partial + coulomb_threshold_partial,
            body_partial,
        ],
    ))
}

/// Channel mobility, saturation, current, substrate current and series resistances.
pub(super) fn transport(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    bias: &LoadBias,
    electrostatics: &Electrostatics,
) -> SpiceResult<Transport> {
    transport_constitutive(
        model,
        here,
        pParam,
        [bias.vbs, bias.vgs, bias.vds, bias.vdes, bias.vses],
        electrostatics,
    )
}

/// Convert reference partials in (effective overdrive, oriented drain,
/// effective body) to normalized terminal partials. This is separate from
/// the exact physical derivatives already present in the generic value.
fn terminal_partials<S>(partials: [f64; 3], electro: &Electrostatics<S>) -> [f64; 3] {
    let [gate, drain, body] = partials;
    [
        gate * electro.dVgsteff_dVg,
        gate.mul_add(electro.dVgsteff_dVd, drain),
        gate.mul_add(electro.dVgsteff_dVb, body) * electro.dVbseff_dVb,
    ]
}

fn external_conductance<S: Constitutive>(
    p: &SizeParam,
    gate: S,
    body: S,
    minimum: f64,
    resistance: f64,
    conductance: f64,
) -> SpiceResult<Local<S, 2>> {
    let gate = Local::new(gate, [1., 0.]);
    let body = Local::new(body, [0., 1.]);
    let (effective, _) = smooth_positive(gate.offset(-p.vfbsd), 2.5e-5)?;
    let argument = effective
        .constant(1.)
        .divide(&effective.scale(p.prwg).offset(1.))?
        .add(&body.scale(-p.prwb));
    let (positive, _) = smooth_positive(argument, 0.0025)?;
    let r = positive.scale(resistance).offset(minimum);
    r.constant(conductance)
        .divide(&r.scale(conductance).offset(1.))
}

/// Reference and physical transport share the complete value sequence.
/// The stack-only local tangent tracks the three conventional Newton
/// coordinates even when the value is a differential time series.
pub(super) fn transport_constitutive<S: Constitutive>(
    model: &Model,
    here: &mut Instance,
    p: &SizeParam,
    [vbs, vgs, vds, vdes, vses]: [S; 5],
    electro: &Electrostatics<S>,
) -> SpiceResult<Transport<S>> {
    let leff = electro.Leff;
    let vtm = electro.Vtm;
    let epssub = electro.epssub;
    let c = |x| Local::new(electro.Vgsteff.constant(x), [0.; 3]);
    let gate = Local::new(electro.Vgsteff.clone(), [1., 0., 0.]);
    let drain = Local::new(electro.Vds.clone(), [0., 1., 0.]);
    let body = Local::new(electro.Vbseff.clone(), [0., 0., 1.]);
    let root = Local::new(electro.sqrtPhis.clone(), [0., 0., electro.dsqrtPhis_dVb]);
    let depth = Local::new(electro.Xdep.clone(), [0., 0., electro.dXdep_dVb]);
    let root_shift = root.clone().offset(-p.sqrtPhi);
    let width_argument = gate
        .clone()
        .scale(p.dwg)
        .add(&root_shift.clone().scale(p.dwb))
        .scale(-2.)
        .offset(p.weff);
    let (width, _) = rational_floor(width_argument, 2e-8, 0., 1e8)?;
    let rds = if model.rdsMod == 1 {
        c(0.)
    } else {
        let argument = c(1.)
            .divide(&gate.clone().scale(p.prwg).offset(1.))?
            .add(&root_shift.scale(p.prwb));
        let (positive, _) = smooth_positive(argument, 0.0025)?;
        let r = positive.scale(p.rds0).offset(p.rdswmin);
        here.grdsw = if r.primal() > 0. {
            here.nf / r.primal()
        } else {
            0.
        };
        r
    };
    let body_effect = c(0.5 * p.k1ox * electro.Lpe_Vb)
        .divide(&root)?
        .offset(here.k2ox - p.k3b * electro.Vth_NarrowW);
    let geometry = c(leff).divide(&depth.scale(p.xj).sqrt()?.scale(2.).offset(leff))?;
    let bulk0 = body_effect.clone().mul_add(
        &geometry.clone().scale(p.a0).offset(p.b0 / (p.weff + p.b1)),
        &c(1.),
    );
    let cubic = geometry.clone().multiply(&geometry).multiply(&geometry);
    let bulk = body_effect
        .multiply(&cubic)
        .scale(-p.ags * p.a0)
        .mul_add(&gate, &bulk0);
    let (bulk0, _) = rational_floor(bulk0, 0.1, 0., 20.)?;
    let (bulk, _) = rational_floor(bulk, 0.1, 0., 20.)?;
    here.Abulk = bulk.primal();
    let (bulk_scale, _) = reciprocal_floor(body.clone().scale(p.keta), -0.9, 20.)?;
    let bulk = bulk.multiply(&bulk_scale);
    let bulk_cv = if p.ketac == p.keta {
        bulk0.multiply(&bulk_scale)
    } else {
        let (factor, _) = reciprocal_floor(body.clone().scale(p.ketac), -0.9, 20.)?;
        bulk0.multiply(&factor)
    };

    let (degradation, [dg, dt, db]) = mobility_degradation(model, here, p, electro)?;
    let raw = Local::new(
        degradation,
        [dg, dt * electro.dVth_dVd, dt.mul_add(electro.dVth_dVb, db)],
    );
    let (denominator, _) = rational_floor(raw, -0.8, 1., 10.)?;
    let mobility = c(here.u0temp).divide(&denominator)?;
    here.ueff = mobility.primal();
    let resistance_product = width
        .clone()
        .scale(here.vsattemp * model.coxe)
        .multiply(&rds);
    let mut field = c(2. * here.vsattemp * leff).divide(&mobility)?;
    here.EsatL = field.primal();
    let drive = gate.clone().offset(2. * vtm);
    let (shape, _) = saturation_shape(&gate, p.a1, p.a2)?;
    let saturation = saturation_voltage(&drive, &bulk, &field, &resistance_product, &shape)?.value;
    here.vdsat = saturation.primal();
    let smoothing = smooth_drain(saturation.clone(), drain.clone(), p.delta)?;
    let effective = smoothing.effective;
    let excess = smoothing.drain_excess;
    here.Vdseff = effective.primal();

    if model.lambdaGiven && model.lambda > 0. {
        let correction = c(p.lambda / leff).divide(&mobility)?;
        let ratio = excess.clone().scale(leff / p.litl).divide(&field)?;
        let square = ratio.offset(1.);
        let square = square.clone().multiply(&square);
        let overshoot = c(1.).add(&c(-2.).divide(&square.offset(1.))?);
        let mut factor = correction.mul_add(&overshoot, &c(1.));
        // Reference Newton stamp discards the slope when the correction
        // rounds to unity; the physical value germ retains it.
        if factor.primal() == 1. {
            factor.partials = [0.; 3];
        }
        field = field.multiply(&factor);
        here.EsatL = field.primal();
    }
    let vasat = saturation_early_voltage(
        &gate,
        &bulk,
        &saturation,
        &field,
        &resistance_product,
        &shape,
        vtm,
    )?
    .value;
    let centroid_argument = gate
        .clone()
        .offset(here.vtfbphi2)
        .scale(1. / (2e8 * here.toxp));
    let centroid_power = centroid_argument.ln()?.scale(model.bdos * 0.7).exp();
    let centroid = c(model.ados * 1.9e-9).divide(&centroid_power.offset(1.))?;
    let oxide = c(epssub * here.coxp).divide(&centroid.scale(here.coxp).offset(epssub))?;
    here.Coxeff = oxide.primal();
    let oxide_width = oxide.clone().multiply(&width).scale(leff.recip());
    let beta = mobility.multiply(&oxide_width);
    here.AbovVgst2Vtm = bulk.primal() / drive.primal();
    let gche = channel_conductance(&gate, &bulk, &effective, &field, &beta, vtm)?.value;
    let idl = series_conductance(&gche, &rds)?.value;
    let fp = if p.fprout <= 0. {
        c(1.)
    } else {
        c(1.).divide(&c(p.fprout * sqrt(leff)).divide(&drive)?.offset(1.))?
    };
    let pvag_argument = gate.clone().scale(p.pvag).divide(&field)?;
    let (pvag, _) = rational_floor(pvag_argument, -0.9, 1., 20.)?;
    let (cclm, vaclm) = if p.pclm > MIN_EXP && excess.compare(1e-10) == Some(Ordering::Greater) {
        let length = saturation.clone().scale(leff).divide(&field)?.offset(leff);
        let resistance_factor = rds.clone().mul_add(&idl, &c(1.));
        let cclm = fp
            .clone()
            .multiply(&pvag)
            .multiply(&resistance_factor)
            .multiply(&length)
            .scale(1. / (p.pclm * p.litl));
        let vaclm = cclm.clone().multiply(&excess);
        (cclm, vaclm)
    } else {
        (c(MAX_EXP), c(MAX_EXP))
    };
    let vadibl = if p.thetaRout > MIN_EXP {
        dibl_early_voltage(&drive, &bulk, &saturation, &body, p.thetaRout, p.pdiblb)?
            .value
            .multiply(&pvag)
    } else {
        c(MAX_EXP)
    };
    let vadits = if p.pdits > MIN_EXP {
        let argument = drain.scale(p.pditsd);
        let exponential = if argument.compare(EXP_THRESHOLD) == Some(Ordering::Greater) {
            c(MAX_EXP)
        } else {
            argument.exp()
        };
        exponential
            .scale(1. + model.pditsl * leff)
            .offset(1.)
            .scale(p.pdits.recip())
            .multiply(&fp)
    } else {
        c(MAX_EXP)
    };
    let (vascbe, _) = substrate_early_voltage(
        &excess,
        leff,
        p.pscbe1 * p.litl,
        p.pscbe2,
        p.pscbe2 > 0. && p.pscbe1 >= 0.,
    )?;
    let idsa = idl
        .multiply(&excess.clone().divide(&vadibl)?.offset(1.))
        .multiply(&excess.clone().divide(&vadits)?.offset(1.));
    let clm = vaclm.divide(&vasat)?.ln_1p()?.divide(&cclm)?.offset(1.);
    let idsa = idsa.multiply(&clm);
    let alpha = p.alpha1.mul_add(leff, p.alpha0);
    let (impact, _) = impact_ionization(&excess, alpha / leff, p.beta0)?;
    let substrate = impact.multiply(&idsa.clone().multiply(&effective));
    [here.gbgs, here.gbds, here.gbbs] = terminal_partials(substrate.partials, electro);
    let ids = idsa.multiply(&excess.divide(&vascbe)?.offset(1.));
    let current = ids.clone().multiply(&effective);
    let mut current = Local::new(current.value, terminal_partials(current.partials, electro));
    if model.vtlGiven && model.vtl > 0. {
        // Use the reference stamp's velocity derivative convention while
        // differentiating the actual velocity law in the physical germ.
        let inverse = c(1. / leff).divide(&oxide_width)?.divide(&gate)?;
        let velocity = current.value.clone().multiply(&inverse.value);
        let [gm, gd, gb] = current.partials;
        let t11 = inverse.primal();
        let t10 = -t11 / gate.primal() * current.primal();
        let velocity = Local::new(
            velocity,
            [
                t10.mul_add(electro.dVgsteff_dVg, gm * t11),
                t10.mul_add(electro.dVgsteff_dVd, gd * t11),
                t10.mul_add(electro.dVgsteff_dVb, gb * t11),
            ],
        );
        let ratio = velocity.scale(1. / (p.vtl * p.tfactor));
        let factor = if ratio.compare(0.) == Some(Ordering::Greater) {
            let exponent = f64::from(2 * MM);
            ratio
                .ln()?
                .scale(exponent)
                .exp()
                .offset(1.)
                .ln()?
                .scale(-1. / exponent)
                .exp()
        } else {
            c(1.)
        };
        current = current.multiply(&factor);
    }
    [here.gm, here.gds, here.gmbs] = current.partials;
    here.IdovVds = ids.primal().max(model.idovvdsc);
    let gate_conductance = if here.rgateMod > 1 || here.trnqsMod != 0 || here.acnqsMod != 0 {
        let mut law = beta
            .clone()
            .scale(p.xrcrg2 * model.vtm)
            .add(&ids)
            .scale(p.xrcrg1 * here.nf);
        if here.rgateMod == 2 {
            law = law
                .clone()
                .scale(here.grgeltd)
                .divide(&law.offset(here.grgeltd))?;
        }
        [here.gcrgg, here.gcrgd, here.gcrgb] = terminal_partials(law.partials, electro);
        here.gcrgs = -(here.gcrgg + here.gcrgd + here.gcrgb);
        law.value
    } else {
        electro.Vgsteff.constant(here.gcrg)
    };
    let (source_conductance, drain_conductance) = if model.rdsMod != 0 {
        let source = external_conductance(
            p,
            vgs.clone(),
            vbs.clone(),
            p.rswmin,
            p.rs0,
            here.sourceConductance,
        )?;
        let drain = external_conductance(
            p,
            vgs.add(&vds.clone().scale(-1.)),
            vbs.add(&vds.clone().scale(-1.)),
            p.rdwmin,
            p.rd0,
            here.drainConductance,
        )?;
        here.gstotd = 0.;
        here.gstotg = vses.primal() * source.partials[0];
        here.gstotb = vses.primal() * source.partials[1];
        here.gstots = -(here.gstotg + here.gstotb);
        let drop = vdes.primal() - vds.primal();
        here.gdtots = 0.;
        here.gdtotg = drop * drain.partials[0];
        here.gdtotb = drop * drain.partials[1];
        here.gdtotd = -(here.gdtotg + here.gdtotb);
        (source.value, drain.value)
    } else {
        here.gstotd = 0.;
        here.gstotg = 0.;
        here.gstotb = 0.;
        here.gstots = 0.;
        here.gdtotd = 0.;
        here.gdtotg = 0.;
        here.gdtotb = 0.;
        here.gdtots = 0.;
        (electro.Vgsteff.constant(0.), electro.Vgsteff.constant(0.))
    };
    let result = Transport {
        dAbulk0_Q_dVb: bulk_cv.partials[2],
        dVdseff_dVg: effective.partials[0],
        dVdseff_dVd: effective.partials[1],
        dVdseff_dVb: effective.partials[2],
        Abulk0_Q: bulk_cv.value,
        Coxeff: oxide.value,
        Rds: rds.value,
        Vdseff: effective.value,
        beta: beta.value,
        unscaled_drain_current: current.value,
        gche: gche.value,
        substrate_current: substrate.value,
        gate_conductance,
        source_conductance,
        drain_conductance,
    };
    here.csub = result.substrate_current.primal();
    here.gcrg = result.gate_conductance.primal();
    here.gstot = result.source_conductance.primal();
    here.gdtot = result.drain_conductance.primal();
    Ok(result)
}
