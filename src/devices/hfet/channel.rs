//! Common intrinsic HFET channel equations from hfeta and hfeta2.
//! Family-specific temperature preparation and gate currents stay with the devices.

use super::HfetInstance;
use crate::devices::jfet::{self, CHARGE};
use crate::primitives::Real;

/// Carrier-density cutoff shared by both C devices.
const N_MIN: Real = 1e-38;

#[derive(Debug, Clone, Copy)]
pub(super) struct ChannelModel {
    pub(super) eta: Real,
    pub(super) vs: Real,
    pub(super) rsi: Real,
    pub(super) rdi: Real,
    pub(super) vsigmat: Real,
    pub(super) vsigma: Real,
    pub(super) sigma0: Real,
    pub(super) eta2: Real,
    pub(super) gamma: Real,
    pub(super) m: Real,
    pub(super) eta1: Real,
    pub(super) d1: Real,
    pub(super) epsi: Real,
    pub(super) mc: Real,
    pub(super) pm: Real,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ChannelTemperature {
    pub(super) vt: Real,
    pub(super) tmu: Real,
    pub(super) tvto: Real,
    pub(super) delta_sqr: Real,
    pub(super) n0: Real,
    pub(super) two_exponentials: bool,
    pub(super) n02: Real,
    pub(super) vt2: Real,
    pub(super) nmax: Real,
    pub(super) gchi0: Real,
    pub(super) imax: Real,
    pub(super) tlambda: Real,
    pub(super) vt1: Real,
    pub(super) cf: Real,
}

/// Active-channel intermediates needed by HFET1's gate and drain corrections.
#[derive(Debug, Clone, Copy)]
pub(super) struct Saturation {
    pub(super) vsate: Real,
    pub(super) delvsatevgt: Real,
    pub(super) delvgtvgs: Real,
    pub(super) sigma: Real,
    pub(super) vgte: Real,
    pub(super) delvgtevgt: Real,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct IntrinsicChannel {
    pub(super) cdrain: Real,
    pub(super) gm: Real,
    pub(super) gds: Real,
    pub(super) capgs: Real,
    pub(super) capgd: Real,
    pub(super) saturation: Option<Saturation>,
}

impl IntrinsicChannel {
    /// The C cutoff returns before checking active-channel quantities.
    pub(super) fn is_finite(self) -> bool {
        [self.cdrain, self.gm, self.gds, self.capgs, self.capgd]
            .iter()
            .all(|v| v.is_finite())
    }
}

/// Evaluate the normalized intrinsic bias (nonnegative Vds).
#[inline]
#[expect(
    clippy::similar_names,
    clippy::many_single_char_names,
    reason = "C equation symbols retained for numerical audit"
)]
pub(super) fn evaluate(
    model: &ChannelModel,
    derived: &ChannelTemperature,
    instance: &HfetInstance,
    vgs: Real,
    vds: Real,
) -> IntrinsicChannel {
    let vt = derived.vt;
    let etavth = model.eta * vt;
    let vl = model.vs / derived.tmu * instance.l;
    let rt = model.rsi + model.rdi;
    let vgt0 = vgs - derived.tvto;
    let (rolloff, rolloff_slope) = super::rolloff((vgt0 - model.vsigmat) / model.vsigma);
    let sigma = model.sigma0 * rolloff;
    let vgt = sigma.mul_add(vds, vgt0);
    let u = 0.5 * vgt / vt - 1.;
    let smoothed = jfet::smoothed_extrema(u, 0., derived.delta_sqr.sqrt());
    let vgte = (2. * vt) * (1. + smoothed.high);
    let (nsn, slope) = super::soft_density(derived.n0, vgt, etavth);
    let (nsm, delnsmvgt) = if derived.two_exponentials {
        let thermal = model.eta2 * vt;
        let nsc = derived.n02 * ((vgt + derived.tvto - derived.vt2) / thermal).exp();
        super::series_density(nsn, slope, nsc, thermal)
    } else {
        (nsn, slope)
    };
    if nsm < N_MIN {
        return IntrinsicChannel {
            cdrain: 0.,
            gm: 0.,
            gds: 0.,
            capgs: derived.cf,
            capgd: derived.cf,
            saturation: None,
        };
    }
    let c = (nsm / derived.nmax).powf(model.gamma);
    let q = (1. + c).powf(1. / model.gamma);
    let ns = nsm / q;
    let gchi = derived.gchi0 * ns;
    let gch = gchi / gchi.mul_add(rt, 1.);
    let gchim = derived.gchi0 * nsm;
    let h = ((2. * gchim).mul_add(model.rsi, 1.) + vgte * vgte / (vl * vl)).sqrt();
    let p = gchim.mul_add(model.rsi, 1.) + h;
    let isatm = gchim * vgte / p;
    let g = (isatm / derived.imax).powf(model.gamma);
    let isat = isatm / (1. + g).powf(1. / model.gamma);
    let vsate = isat / gch;
    let d = (vds / vsate).powf(model.m);
    let e = (1. + d).powf(1. / model.m);
    let delidgch = vds * derived.tlambda.mul_add(vds, 1.) / e;
    let cdrain = gch * delidgch;
    let delidvsate = cdrain * d / vsate / (1. + d);
    let delidvds = gch / e
        * derived
            .tlambda
            .mul_add(vds, derived.tlambda.mul_add(vds, 1.) / (1. + d));
    let a = gchi.mul_add(rt, 1.);
    let delgchgchi = 1. / (a * a);
    let delgchins = derived.gchi0;
    let delnsnsm = ns / nsm / (1. + c);
    let delvgtevgt = smoothed.high_weights[0];
    let delvsateisat = 1. / gch;
    let delisatisatm = isat / isatm / (1. + g);
    let delisatmvgte = gchim * (p - vgte * vgte / (vl * vl * h)) / (p * p);
    let delvsategch = -vsate / gch;
    let delisatmgchim = vgte * (gchim * model.rsi).mul_add(-(1. + 1. / h), p) / (p * p);
    let delvgtvgs = (vds * model.sigma0 / model.vsigma).mul_add(-rolloff_slope, 1.);
    let p = delgchgchi * delgchins * delnsnsm * delnsmvgt;
    let delvsatevgt = delvsategch.mul_add(
        p,
        delvsateisat
            * delisatisatm
            * (delisatmgchim * derived.gchi0).mul_add(delnsmvgt, delisatmvgte * delvgtevgt),
    );
    let g = delidvsate.mul_add(delvsatevgt, delidgch * p);
    let gm = g * delvgtvgs;
    let gds = g.mul_add(sigma, delidvds);
    // Capacitances.
    let temp = model.eta1 * vt;
    let cg1 = 1. / temp.mul_add((-(vgs - derived.vt1) / temp).exp(), model.d1 / model.epsi);
    let cgc = instance.w * instance.l * (CHARGE * delnsnsm * delnsmvgt).mul_add(delvgtvgs, cg1);
    let vdse = vds * (1. + (vds / vsate).powf(model.mc)).powf(-1. / model.mc);
    let a = (vsate - vdse) / 2.0f64.mul_add(vsate, -vdse);
    let a = a * a;
    let third = 2.0 / 3.0;
    let p = (1. - model.pm).mul_add((-vds / vsate).exp(), model.pm);
    let capgs = derived.cf + 2. * third * cgc * (1. - a) / (1. + p);
    let a = vsate / 2.0f64.mul_add(vsate, -vdse);
    let a = a * a;
    let capgd = derived.cf + 2. * p * third * cgc * (1. - a) / (1. + p);
    IntrinsicChannel {
        cdrain,
        gm,
        gds,
        capgs,
        capgd,
        saturation: Some(Saturation {
            vsate,
            delvsatevgt,
            delvgtvgs,
            sigma,
            vgte,
            delvgtevgt,
        }),
    }
}
