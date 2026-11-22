//! Parker-Skellern JFET/MESFET, JFET level 2 (`jfet2`).
//!
//! C references, read as behaviour only:
//!
//! - `jfet2/jfet2.c` and `jfet2parm.h` (`JFET2pTable`, `JFET2mPTable`): the
//!   instance and model setters; `jfet2mpar.c`/`jfet2par.c` apply them in card
//!   order and `jfet2set.c` (`JFET2setup`) the defaults, internal nodes and
//!   state layout;
//! - `jfet2/jfet2temp.c` (`JFET2temp`): IS, PB (`tGatePot`), CGS/CGD scaling
//!   and `vcrit`, then `psmodel.c` `PSinstanceinit` (`xiwoo`, `za`, `alpha`,
//!   `d3`);
//! - `jfet2/psmodel.c`: `PSids` (gate diodes with breakdown, the
//!   subthreshold/dual power-law/velocity-saturation channel, the rate
//!   dependent threshold modulation through the delayed gate voltages
//!   `vgstrap`/`vtrap` and the thermal reduction through the average power
//!   `pave`), `qgg`/`PScharge` (Statz gate charge) and `PSacload`;
//! - `jfet2/jfet2load.c` (`JFET2load`): `MODEINITJCT` start voltages,
//!   `DEVpnjlim`/`DEVfetlim` limiting, inverse mode, charge integration and
//!   the stamps; `jfet2acld.c` (`JFET2acLoad`), `jfet2trun.c`
//!   (`JFET2trunc`), `jfet2ask.c` (`JFET2ask`) and `jfet2ic.c`.
//!
//! The load and the AC load follow C operation by operation through the
//! level-1 frame they share (see [`crate::devices::jfet`]), so the port is
//! bit-identical to a C build without fused multiply-adds.
//!
//! # State
//!
//! C's `JFET2numStates` layout. Besides the gate charges, the model keeps
//! three filtered quantities from one accepted point to the next (C
//! `JFET2vgstrap`, `JFET2vtrap`, `JFET2pave`): in transient each load
//! evaluates `x = h x(accepted) + (1 - h) x(now)` with
//! `h = (tau / (tau + dt / 4))^4` (`TAUG` for the gate voltages, `TAUD` for
//! the power), reading only the last *accepted* value. Outside transient
//! `h = 0`. As in C the slots follow `PSids`' arguments, so in inverse mode
//! the "gate-drain" trap holds the gate-source voltage.
//!
//! The Statz gate charge of `PScharge` is incremental in transient:
//! `qgs = qgs(accepted) + (q(vgs, vgd) - q(vgs1, vgd) + q(vgs, vgd1) -
//! q(vgs1, vgd1)) / 2` with the accepted junction voltages `vgs1`, `vgd1`
//! (and symmetrically for `qgd`). As in C a plain operating point leaves the
//! charge slots alone, while the `uic` initial load and DC sweeps store the
//! total charge in both (`PScharge` outside transient), so truncation
//! control (`jfet2trun.c`, `CKTterr` on both charges) sees C's values.
//!
//! As in C, the Newton matrix stamps each gate charge as a two-terminal
//! capacitance (no `d qgs / d vgd` cross terms) and integrates `CDS * vds`
//! into the drain current without stamping its conductance.
//!
//! # Deliberate divergences
//!
//! - As for level 1: no bypass, the held `off` rule, and `FC > 0.95` is
//!   rejected instead of clamped. `PSids` also writes `POWR_BEFORE` (`CKTstate1`) outside
//!   transient; that value is only ever multiplied by `h = 0` there and is
//!   replaced before the first transient step, so the port does not write
//!   it.
//! - C has no pole-zero load for level 2 (`DEVpzLoad = NULL`, so `.pz`
//!   silently omits the device); the port refuses `.pz` explicitly.
//! - Parameters whose zero or negative values divide by zero or take roots
//!   of negative numbers in `psmodel.c` are range-checked (`VBD`, `XI`, `P`,
//!   `Q` positive; `Z`, `TAUD`, `TAUG`, `VST`, `DELTA`, `IBD`, `CDS`
//!   nonnegative), and any nonfinite evaluation is an error.
//!
//! `.sens` replays the family-specific DC and AC reset loads; gate-lag
//! and thermal memory remain part of the Parker model, not the shared
//! gate lifecycle.

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

use crate::devices::jfet::{
    self, IC_COMPONENTS, Instance, K_OVER_Q,
    lifecycle::{self, GateStamp, GateUpdate, Junctions, LoadPhase, Loaded, carry},
    p, required,
};
use crate::devices::limiting;
use crate::devices::nonlinear::charge::ChargeBranch;
use crate::devices::nonlinear::thermal::{TemperatureCache, TemperatureKey};
use crate::devices::schema::{
    ScalarDomain as D, ScalarParameter as P, ScalarSchema, ScalarUnit as U,
};
use crate::devices::sensitivity::{
    AcReset, CELSIUS_TO_KELVIN, ParameterScope, RecordPair, SensitivityFamily, SensitivityKernel,
    SensitivityLoad, SensitivityMode, SensitivityParameter, SensitivityRecord,
};
use crate::devices::{Device, LinearContext, ModelContext, ResolvedModel, StampContext};
use crate::maths::Vector;
use crate::netlist::ast::DeviceInstance;
use crate::primitives::{NodeAppend, NodeId, Real, SpiceError, SpiceResult};

/// `psmodel.c` `FX`: below `FX` thermal voltages a junction exponential is
/// dropped (and `FX * vst` is the numerical cut-off of the channel).
const FX: Real = -10.0;
/// `psmodel.c` `MX`: the largest exponential argument.
const MX: Real = 40.0;
/// `psmodel.c` `EMX = exp(MX)`.
#[expect(
    clippy::excessive_precision,
    reason = "reference C literal spelling is preserved for numerical parity"
)]
const EMX: Real = 2.353_852_668_370_199_842e17;
/// `psmodel.h` `FOURTH`: the relaxation filters use a quarter step.
const FOURTH: Real = 0.25;
/// `jfet2temp.c` uses a fixed 1.11 eV band gap for IS.
const EG: Real = 1.11;

/// Slots of the state vector: C's `JFET2numStates` layout (`jfet2defs.h`),
/// level 1's followed by the drain-source charge and the filters.
mod slot {
    pub(super) use crate::devices::jfet::slot::*;
    /// Drain-source charge `CDS * vds` (C `JFET2qds`) and its derivative.
    pub(super) const QDS: usize = 13;
    /// Average power (C `JFET2pave`) and filtered gate voltages (C
    /// `JFET2vtrap`, `JFET2vgstrap`).
    pub(super) const PAVE: usize = 15;
    pub(super) const VGDTRAP: usize = 16;
    pub(super) const VGSTRAP: usize = 17;
    /// C `JFET2unknown`: allocated, never written.
    pub(super) const UNKNOWN: usize = 18;
    pub(super) const COUNT: usize = 19;
}

/// Model setters of `jfet2parm.h` with the defaults of `jfet2set.c`. The two
/// spellings of VTO (`vt0`, `vto`) and of PB (`vbi`, `pb`) and `hfgam`
/// (default: `lfgam`) have no schema default.
const MODEL: &[P] = &[
    p("acgam", U::Dimensionless, D::Finite, Some(0.)),
    p("af", U::Dimensionless, D::Finite, Some(1.)),
    p("beta", U::AmperePerVoltSquared, D::NonNegative, Some(1e-4)),
    p("cds", U::Farad, D::NonNegative, Some(0.)),
    p("cgd", U::Farad, D::NonNegative, Some(0.)),
    p("cgs", U::Farad, D::NonNegative, Some(0.)),
    p("delta", U::Dimensionless, D::NonNegative, Some(0.)),
    p("hfeta", U::Dimensionless, D::Finite, Some(0.)),
    p("hfe1", U::InverseVolt, D::Finite, Some(0.)),
    p("hfe2", U::InverseVolt, D::Finite, Some(0.)),
    p("hfg1", U::InverseVolt, D::Finite, Some(0.)),
    p("hfg2", U::InverseVolt, D::Finite, Some(0.)),
    p("mvst", U::InverseVolt, D::Finite, Some(0.)),
    p("mxi", U::Dimensionless, D::Finite, Some(0.)),
    p("fc", U::Dimensionless, D::NonNegative, Some(0.5)),
    p("ibd", U::Ampere, D::NonNegative, Some(0.)),
    p("is", U::Ampere, D::Positive, Some(1e-14)),
    // jfet2noi.c input: accepted for model compatibility; `.noise` itself is
    // refused until jfet2noi.c is ported.
    p("kf", U::Dimensionless, D::Finite, Some(0.)),
    p("lambda", U::InverseVolt, D::NonNegative, Some(0.)),
    p("lfgam", U::Dimensionless, D::Finite, Some(0.)),
    p("lfg1", U::InverseVolt, D::Finite, Some(0.)),
    p("lfg2", U::InverseVolt, D::Finite, Some(0.)),
    p("n", U::Dimensionless, D::Positive, Some(1.)),
    p("p", U::Dimensionless, D::Positive, Some(2.)),
    p("vbi", U::Volt, D::Positive, None),
    p("pb", U::Volt, D::Positive, None),
    p("q", U::Dimensionless, D::Positive, Some(2.)),
    p("rd", U::Ohm, D::NonNegative, Some(0.)),
    p("rs", U::Ohm, D::NonNegative, Some(0.)),
    p("taud", U::Second, D::NonNegative, Some(0.)),
    p("taug", U::Second, D::NonNegative, Some(0.)),
    p("vbd", U::Volt, D::Positive, Some(1.)),
    // "version number of PS model": stored but read nowhere in jfet2/.
    p("ver", U::Dimensionless, D::Finite, Some(0.)),
    p("vst", U::Volt, D::NonNegative, Some(0.)),
    p("vt0", U::Volt, D::Finite, None),
    p("vto", U::Volt, D::Finite, None),
    p("xc", U::Dimensionless, D::Finite, Some(0.)),
    p("xi", U::Dimensionless, D::Positive, Some(1000.)),
    p("z", U::Dimensionless, D::NonNegative, Some(1.)),
    p("hfgam", U::Dimensionless, D::Finite, None),
    p("tnom", U::Celsius, D::Temperature, None),
];

/// Validated model card. `None` records C's "not given".
#[derive(Debug, Clone, Copy)]
struct Model {
    /// C `JFET2type`: +1 for NJF, -1 for PJF.
    pol: Real,
    acgam: Real,
    beta: Real,
    cds: Real,
    cgd: Real,
    cgs: Real,
    delta: Real,
    hfeta: Real,
    hfe1: Real,
    hfe2: Real,
    hfg1: Real,
    hfg2: Real,
    hfgam: Real,
    mvst: Real,
    mxi: Real,
    fc: Real,
    ibd: Real,
    is: Real,
    lambda: Real,
    lfgam: Real,
    lfg1: Real,
    lfg2: Real,
    n: Real,
    p: Real,
    /// C `JFET2phi` (`vbi`/`pb`).
    pb: Real,
    q: Real,
    rd: Real,
    rs: Real,
    taud: Real,
    taug: Real,
    vbd: Real,
    vst: Real,
    vto: Real,
    xc: Real,
    xi: Real,
    z: Real,
    tnom: Option<Real>,
}

/// Every temperature- and area-dependent quantity of one load
/// (`jfet2temp.c`, `PSinstanceinit` and the area factors of `psmodel.c`).
#[derive(Debug, Clone, Copy)]
struct Operating {
    /// `kT/q` at the device temperature (limiting and `vcrit`).
    vt: Real,
    /// `N kT/q` (`psmodel.h` `NVT`).
    nvt: Real,
    /// C `JFET2vcrit` (from the per-device saturation current).
    vcrit: Real,
    /// `JFET2tSatCur * area`, `IBD * area`, `BETA * area`, `DELTA / area`.
    isat: Real,
    ibd: Real,
    beta: Real,
    delta: Real,
    /// C `JFET2tGatePot` (`VBI`).
    pb: Real,
    /// `JFET2tCGS * area`, `JFET2tCGD * area`, `CDS * area`.
    czgs: Real,
    czgd: Real,
    capds: Real,
    /// C `JFET2corDepCap` (`VMAX`), `JFET2xiwoo`, model `JFET2za`,
    /// `JFET2alpha`, `JFET2d3`.
    vmax: Real,
    xi_woo: Real,
    za: Real,
    alpha: Real,
    d3: Real,
}

/// The accepted history a transient load filters against (`psmodel.h`
/// `VGSTRAP_BEFORE`, `VGDTRAP_BEFORE`, `POWR_BEFORE`) and the step
/// (`STEP = CKTdelta`).
#[derive(Debug, Clone, Copy)]
struct Memory {
    step: Real,
    vgstrap: Real,
    vgdtrap: Real,
    powr: Real,
}

/// `PSids` at its argument voltages: the drain current, its partials with
/// respect to `vgs` at fixed `vds` and `vds` at fixed `vgs`, both gate diode
/// currents and conductances, and the new filtered state.
#[derive(Debug, Clone, Copy)]
struct Ids {
    current: Real,
    gm: Real,
    gds: Real,
    igs: Real,
    igd: Real,
    ggs: Real,
    ggd: Real,
    vgstrap: Real,
    vgdtrap: Real,
    pave: Real,
}

/// Everything one normalized bias point loads, per device (before `m`), in
/// `jfet2load.c`'s orientation after the inverse-mode exchange.
#[derive(Debug, Clone, Copy)]
struct Point {
    /// Channel current from d' to s' and its partials (`cd + cgd`, `gm`,
    /// `gds` of `jfet2load.c`).
    channel: (Real, Real, Real),
    /// Gate-source and gate-drain diode currents and conductances (with
    /// `gmin`).
    gs: (Real, Real),
    gd: (Real, Real),
    /// The filtered state to store (`vgstrap`, `vtrap`, `pave`).
    vgstrap: Real,
    vgdtrap: Real,
    pave: Real,
}

/// `psmodel.c` `qgg`: Statz gate charge and its partials `(q, cgs, cgd)`.
#[expect(
    clippy::too_many_arguments,
    reason = "keeps the positional interface explicit at the reference solver or device boundary"
)]
fn qgg(
    vgs: Real,
    vgd: Real,
    gamma: Real,
    pb: Real,
    alpha: Real,
    vto: Real,
    vmax: Real,
    xc: Real,
    cgso: Real,
    cgdo: Real,
) -> (Real, Real, Real) {
    let vds = vgs - vgd;
    let d1_xc = 1. - xc;
    let gate = jfet::smoothed_extrema(vgs, vgd, alpha.sqrt());
    let veff = gamma.mul_add(vds, gate.high);
    let vnr = d1_xc * (veff - vto);
    let threshold = jfet::smoothed_extrema(vnr, 0., 0.2);
    let vnew = if vnr < 0. {
        xc.mul_add(veff, d1_xc * vto) + threshold.high
    } else {
        veff - threshold.low
    };
    let derivative = d1_xc.mul_add(threshold.high_weights[0], xc);
    let voltage = vnew.min(vmax);
    let qrt = (1. - voltage / pb).sqrt();
    let (ext, cgso_eff) = if vnew < vmax {
        (0., cgso * derivative / qrt)
    } else {
        let vx = 0.5 * (vnew - vmax);
        let par = 1. + vx / (pb - vmax);
        (vx * (1. + par) / qrt, cgso * derivative / qrt * par)
    };
    let [cplus, cminus] = gate.high_weights;
    let cgs = cgso_eff.mul_add(cplus + gamma, cgdo * (cminus + gamma));
    let cgd = cgso_eff.mul_add(cminus - gamma, cgdo * (cplus - gamma));
    let charge = cgso.mul_add(
        voltage / 1_f64.midpoint(qrt) + ext,
        cgdo * gamma.mul_add(vds, gate.low),
    );
    (charge, cgs, cgd)
}

/// One forward gate diode of `PSids` (`isat`, `N Vt`, `gmin`): current and
/// conductance.
fn forward_diode(v: Real, isat: Real, nvt: Real, gmin: Real) -> (Real, Real) {
    let arg = v / nvt;
    if arg > FX {
        if arg < MX {
            let (current, conductance) =
                crate::devices::nonlinear::exponential_junction_at_argument(arg, nvt, isat);
            (gmin.mul_add(v, current), conductance + gmin)
        } else {
            let zz = isat * EMX;
            (
                gmin.mul_add(v, zz.mul_add(arg - MX + 1., -isat)),
                zz / nvt + gmin,
            )
        }
    } else {
        (gmin.mul_add(v, -isat), gmin)
    }
}

/// Adds `PSids`' reverse "breakdown" conduction (`IBD`, `VBD`) to a diode.
fn breakdown((mut i, mut g): (Real, Real), v: Real, ibd: Real, vbd: Real) -> (Real, Real) {
    let arg = -v / vbd;
    if arg > FX {
        if arg < MX {
            let (current, conductance) =
                crate::devices::nonlinear::exponential_junction_at_argument(arg, vbd, ibd);
            g += conductance;
            i -= current;
        } else {
            let zz = ibd * EMX;
            g += zz / vbd;
            i -= zz.mul_add(arg - MX + 1., -ibd);
        }
    } else {
        i += ibd;
    }
    (i, g)
}

/// `(tau / (tau + step / 4))^4` (`psmodel.c`).
fn relaxation(tau: Real, step: Real) -> Real {
    let mut h = tau / (tau + step * FOURTH);
    h *= h;
    h *= h;
    h
}

impl Model {
    /// `psmodel.c` `PSids` at (`vgs`, `vgd`) with `vgs >= vgd`; `memory` is
    /// the accepted filter state of a transient load (`TRAN_ANAL`).
    #[expect(
        clippy::similar_names,
        reason = "related reference equation variables retain their distinct C names"
    )]
    fn ids(
        &self,
        op: &Operating,
        vgs: Real,
        vgd: Real,
        gmin: Real,
        memory: Option<&Memory>,
    ) -> Ids {
        // Gate junction diodes.
        let (igs, ggs) = breakdown(
            forward_diode(vgs, op.isat, op.nvt, gmin),
            vgs,
            op.ibd,
            self.vbd,
        );
        let (igd, ggd) = breakdown(
            forward_diode(vgd, op.isat, op.nvt, gmin),
            vgd,
            op.ibd,
            self.vbd,
        );

        // Drain current and derivatives.
        let vdst = vgs - vgd;
        // Rate-dependent threshold modulation.
        let (h, vgdtrap, vgstrap) = memory.map_or((0., vgd, vgs), |memory| {
            let h = relaxation(self.taug, memory.step);
            (
                h,
                h * memory.vgdtrap + (1. - h) * vgd,
                h * memory.vgstrap + (1. - h) * vgs,
            )
        });
        let (lfg, lfg1, lfg2) = (self.lfgam, self.lfg1, self.lfg2);
        let (hfg, hfg1, hfg2) = (self.hfgam, self.hfg1, self.hfg2);
        let (hfe, hfe1, hfe2) = (self.hfeta, self.hfe1, self.hfe2);
        let mut vgst = vgs - self.vto;
        vgst -= (lfg - lfg1 * vgstrap + lfg2 * vgdtrap) * vgdtrap;
        let eta = hfe - hfe1 * vgdtrap + hfe2 * vgstrap;
        let dvgs = vgstrap - vgs;
        vgst += eta * dvgs;
        let gam = hfg - hfg1 * vgstrap + hfg2 * vgdtrap;
        let dvgd = vgdtrap - vgd;
        vgst += gam * dvgd;
        let (mut idrain, mut gm, mut gds);
        {
            // Exponential subthreshold effect ids(vgst, vdst).
            let mvst = self.mvst;
            let vst = self.vst * (1. + mvst * vdst);
            if vgst > FX * vst {
                let (vgt, subfac, derivative);
                let large = MX * vst;
                if vgst > large {
                    // Numerically large.
                    subfac = EMX + 1.;
                    derivative = EMX / subfac;
                    vgt = derivative.mul_add(vgst - large, large);
                } else {
                    // Limit gate bias exponentially.
                    let exp = (vgst / vst).exp();
                    subfac = 1. + exp;
                    vgt = vst * exp.ln_1p();
                    derivative = exp / subfac;
                }
                // Dual power-law ids(vgt, vdst).
                let m_q = self.q;
                let p_m_q = self.p - m_q;
                let dvpd_dvdst = op.d3 * vgt.powf(p_m_q);
                let vdp = vdst * dvpd_dvdst;
                // Early saturation effect ids(vgt, vdp).
                let za = op.za;
                let mxi = self.mxi;
                let vsat_fac = vgt / (mxi * vgt + op.xi_woo);
                let vsat = vgt / (1. + vsat_fac);
                let half = vsat * 0.5;
                let aa = za.mul_add(vdp, half);
                let a_aa = za.mul_add(vdp, -half);
                let knee = half * self.z.sqrt();
                let rpt = aa.hypot(knee);
                let a_rpt = a_aa.hypot(knee);
                // Rationalized difference of the two positive roots.
                let vdt = vsat * (za * vdp / rpt.midpoint(a_rpt));
                let dvdt_dvdp = za * (aa / rpt - a_aa / a_rpt);
                let dvdt_dvgt = (vdt - vdp * dvdt_dvdp) * (1. + mxi * vsat_fac * vsat_fac)
                    / (1. + vsat_fac)
                    / vgt;
                // Intrinsic Q-law FET equation ids(vgt, vdt).
                let exponent = m_q - 1.;
                let full = vgt.powf(exponent);
                let ratio = vdt / vgt;
                if ratio.abs() < 0.5 {
                    let increment = (exponent * (-ratio).ln_1p()).exp_m1();
                    gds = full.mul_add(increment, full);
                    gm = -full * increment;
                } else {
                    gds = (vgt - vdt).powf(exponent);
                    gm = full - gds;
                }
                idrain = vdt.mul_add(gds, vgt * gm);
                gds *= m_q;
                gm *= m_q;
                gm += gds * dvdt_dvgt;
                gds *= dvdt_dvdp;
                gm += gds * p_m_q * vdp / vgt;
                gds *= dvpd_dvdst;
                let arg = derivative;
                if vst != 0. {
                    gds += gm * self.vst * mvst * (vgt - vgst * arg) / vst;
                }
                gm *= arg;
            } else {
                // In extreme cut-off (numerically).
                idrain = 0.;
                gm = 0.;
                gds = 0.;
            }
        }
        let arg = h * gam
            + (1. - h) * (hfe1 * dvgs - hfg2 * dvgd + 2. * lfg2 * vgdtrap - lfg1 * vgstrap + lfg);
        gds += gm * arg;
        gm *= 1. - h * eta + (1. - h) * (hfe2 * dvgs - hfg1 * dvgd + lfg1 * vgdtrap) - arg;

        // Channel length modulation and beta scaling.
        let arg = op.beta * (1. + self.lambda * vdst);
        gm *= arg;
        gds = op.beta * self.lambda * idrain + gds * arg;
        idrain *= arg;

        // Thermal reduction of drain current.
        let delta = op.delta;
        let (h, pave, powr_before) = memory.map_or_else(
            || {
                // `POWR_NOW = POWR_BEFORE = pAverage`.
                let pave = vdst * idrain;
                (0., pave, pave)
            },
            |memory| {
                let h = relaxation(self.taud, memory.step);
                let pave = h * memory.powr + (1. - h) * vdst * idrain;
                (h, pave, memory.powr)
            },
        );
        let pfac = 1. + pave * delta;
        idrain /= pfac;
        let arg = (h * delta * powr_before + 1.) / pfac / pfac;
        Ids {
            current: idrain,
            gm: gm * arg,
            gds: gds * arg - (1. - h) * delta * idrain * idrain,
            igs,
            igd,
            ggs,
            ggd,
            vgstrap,
            vgdtrap,
            pave,
        }
    }

    /// `PScharge`'s `qgg` with this device's parameters.
    fn qgg(&self, op: &Operating, vgs: Real, vgd: Real) -> (Real, Real, Real) {
        qgg(
            vgs, vgd, self.acgam, op.pb, op.alpha, self.vto, op.vmax, self.xc, op.czgs, op.czgd,
        )
    }

    /// `psmodel.c` `PSacload`: the AC transconductance and output
    /// conductance `(Gm, xGm, Gds, xGds)` at angular frequency `omega` from
    /// the operating point's `gm`, `gds` and drain current `ids` (C's stored
    /// `JFET2cd`).
    #[expect(
        clippy::too_many_arguments,
        reason = "keeps the positional interface explicit at the reference solver or device boundary"
    )]
    #[expect(
        clippy::similar_names,
        reason = "related reference equation variables retain their distinct C names"
    )]
    fn acload(
        &self,
        op: &Operating,
        vgs: Real,
        vgd: Real,
        ids: Real,
        omega: Real,
        gm: Real,
        gds: Real,
    ) -> (Real, Real, Real, Real) {
        let vds = vgs - vgd;
        let lfgam = self.lfgam;
        let lfg1 = self.lfg1;
        let lfg2 = self.lfg2 * vgd;
        let hfg1 = self.hfg1;
        let hfg2 = self.hfg2 * vgd;
        let hfeta = self.hfeta;
        let hfe1 = self.hfe1;
        let hfe2 = self.hfe2 * vgs;
        let hfgam = self.hfgam - hfg1 * vgs + hfg2;
        let eta = hfeta - hfe1 * vgd + hfe2;
        let lfga = lfgam - lfg1 * vgs + lfg2 + lfg2;
        let gmo = gm / (1. - lfga + lfg1 * vgd);

        let wtg = self.taug * omega;
        let wtgdet = 1. + wtg * wtg;
        let gwtgdet = gmo / wtgdet;

        let arg = hfgam - lfga;
        let gdsi = arg * gwtgdet;
        let gdsr = arg * gmo - gdsi;
        let gmi = (eta + lfg1 * vgd) * gwtgdet + gdsi;

        let xgds = wtg * gdsi;
        let gds = gds + gdsr;
        let xgm = -wtg * gmi;
        let gm = gmi + gmo * (1. - eta - hfgam);

        let delta = op.delta;
        let wtd = self.taud * omega;
        let wtddet = 1. + wtd * wtd;
        let fac = delta * ids;
        let del = 1. / (1. - fac * vds);
        let dd = (del - 1.) / wtddet;
        let dr = del - dd;
        let di = wtd * dd;

        let cdsqr = fac * ids * del * wtd / wtddet;
        (
            dr * gm - di * xgm,
            di * gm + dr * xgm,
            dr * gds - di * xgds + cdsqr * wtd,
            di * gds + dr * xgds + cdsqr,
        )
    }
}

/// Parker-Skellern JFET with series resistance, Statz gate charge, a
/// drain-source capacitance and temperature scaling.
#[derive(Debug)]
pub struct Jfet2 {
    name: String,
    /// The `.model` card this instance resolved to (`JFET2modName`), which
    /// `.sens` groups records by.
    model_name: String,
    /// External d, g, s followed by any internal source/drain nodes.
    terminals: Vec<NodeId>,
    /// d', g, s': the nodes the intrinsic device sees.
    inner: [NodeId; 3],
    model: Model,
    instance: Instance,
    derived: TemperatureCache<Operating>,
    /// `OFF` and `IC-VDS`/`IC-VGS` (C `JFET2off`, `JFET2icVDS`, `JFET2icVGS`).
    initial: crate::devices::initial::InstanceInitial,
}

impl Jfet2 {
    pub(crate) fn instantiate(
        i: &DeviceInstance,
        nodes: &mut NodeAppend<'_>,
        resolved: &ResolvedModel<'_>,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        if resolved.levels().backend != crate::devices::ModelBackend::Jfet2 || i.nodes.len() != 3 {
            return Err(SpiceError::circuit(format!(
                "{}: JFET2 needs level 2 and three terminals",
                i.name
            )));
        }
        let (initial, setters) = crate::devices::initial::split(&i.parameters, &IC_COMPONENTS)?;
        let (pol, scalars) = jfet::card_scalars(resolved);
        // `inpgmod.c::create_model`: see `Jfet::instantiate`.
        let (scalars, model_defaults) = crate::devices::schema::split_model_setters(
            scalars,
            &MODEL_SCHEMA,
            &crate::devices::jfet::INSTANCE_SCHEMA,
        );
        let m = MODEL_SCHEMA.validate(scalars, &resolved.card().location)?;
        let r = |name: &str| required(&m, name);
        // JFET2_MOD_VTO (`vt0`, alias `vto`) and JFET2_MOD_PB (`vbi`, alias
        // `pb`): setters apply in card order, so the later spelling wins.
        let vto = jfet::last_setter(&m, &["vt0", "vto"]).unwrap_or(-2.);
        let pb = jfet::last_setter(&m, &["vbi", "pb"]).unwrap_or(1.);
        let lfgam = r("lfgam")?;
        let model = Model {
            pol,
            acgam: r("acgam")?,
            beta: r("beta")?,
            cds: r("cds")?,
            cgd: r("cgd")?,
            cgs: r("cgs")?,
            delta: r("delta")?,
            hfeta: r("hfeta")?,
            hfe1: r("hfe1")?,
            hfe2: r("hfe2")?,
            hfg1: r("hfg1")?,
            hfg2: r("hfg2")?,
            // jfet2parm.h: HFGAM defaults to the model's LFGAM.
            hfgam: m.get("hfgam").map_or(lfgam, |v| v.value),
            mvst: r("mvst")?,
            mxi: r("mxi")?,
            fc: r("fc")?,
            ibd: r("ibd")?,
            is: r("is")?,
            lambda: r("lambda")?,
            lfgam,
            lfg1: r("lfg1")?,
            lfg2: r("lfg2")?,
            n: r("n")?,
            p: r("p")?,
            pb,
            q: r("q")?,
            rd: r("rd")?,
            rs: r("rs")?,
            taud: r("taud")?,
            taug: r("taug")?,
            vbd: r("vbd")?,
            vst: r("vst")?,
            vto,
            xc: r("xc")?,
            xi: r("xi")?,
            z: r("z")?,
            tnom: m.get("tnom").map(|v| v.value),
        };
        jfet::check_fc(&i.name, model.fc)?;
        let instance = jfet::instance_setters(&model_defaults, &setters, &i.location)?;
        let mut device = Self {
            name: i.name.clone(),
            model_name: resolved.card().name.clone(),
            terminals: vec![],
            inner: [NodeId::GROUND; 3],
            model,
            instance,
            derived: TemperatureCache::new(),
            initial,
        };
        // Validate every derivation before interning nodes.
        device.operating(context)?;
        (device.terminals, device.inner) = jfet::intern_nodes(i, nodes, model.rd, model.rs)?;
        Ok(Box::new(device))
    }

    /// `jfet2temp.c` and `PSinstanceinit` for this instance, with
    /// `psmodel.c`'s area scaling.
    fn operating(&self, context: &ModelContext) -> SpiceResult<Operating> {
        self.derived
            .get_or_derive(TemperatureKey::context(context), || {
                self.derive_operating(context)
            })
    }

    fn derive_operating(&self, context: &ModelContext) -> SpiceResult<Operating> {
        let (model, instance) = (&self.model, &self.instance);
        let (tnom, temp) = jfet::temperatures(&self.name, model.tnom, instance, context)?;
        let junction = jfet::gate_junction(model.pb, tnom, temp);
        let vt = temp * K_OVER_Q;
        let ratio1 = temp / tnom - 1.;
        let saturation = model.is * (ratio1 * EG / vt).exp();
        let pb = junction.pb;
        let tcgs = model.cgs * junction.cjfact * junction.cjfact1;
        let tcgd = model.cgd * junction.cjfact * junction.cjfact1;
        // PSinstanceinit.
        let woo = pb - model.vto;
        let xi_woo = model.xi * woo;
        let area = instance.area;
        let operating = Operating {
            vt,
            nvt: temp * K_OVER_Q * model.n,
            vcrit: limiting::critical_voltage(vt, saturation),
            isat: saturation * area,
            ibd: model.ibd * area,
            beta: model.beta * area,
            delta: model.delta / area,
            pb,
            czgs: tcgs * area,
            czgd: tcgd * area,
            capds: model.cds * area,
            vmax: model.fc * pb,
            xi_woo,
            za: (1. + model.z).sqrt() / 2.,
            alpha: xi_woo * xi_woo / (model.xi + 1.) / (model.xi + 1.) / 4.,
            d3: model.p / model.q / woo.powf(model.p - model.q),
        };
        let finite = [
            operating.nvt,
            operating.vcrit,
            operating.isat,
            operating.beta,
            operating.delta,
            operating.czgs,
            operating.czgd,
            operating.vmax,
            operating.xi_woo,
            operating.alpha,
            operating.d3,
        ]
        .iter()
        .all(|v| v.is_finite());
        if !(finite && pb.is_finite() && pb > 0.)
            || operating.isat <= 0.
            || operating.czgs < 0.
            || operating.czgd < 0.
        {
            return Err(SpiceError::circuit(format!(
                "{}: JFET2 parameters are out of range at {temp:.2} K (nonfinite value, \
                 VBI - VTO <= 0 with P != Q, or nonpositive PB/IS after temperature scaling)",
                self.name
            )));
        }
        Ok(operating)
    }

    /// `jfet2load.c`'s DC evaluation at normalized `vgs`, `vgd`: `PSids` in
    /// normal mode, or with the junctions exchanged in inverse mode (`gds +=
    /// gm`, `gm = -gm`, the current reversed).
    fn point(
        &self,
        op: &Operating,
        vgs: Real,
        vgd: Real,
        gmin: Real,
        memory: Option<&Memory>,
    ) -> SpiceResult<Point> {
        let point = if vgs - vgd < 0. {
            let r = self.model.ids(op, vgd, vgs, gmin, memory);
            Point {
                channel: (-r.current, -r.gm, r.gds + r.gm),
                gs: (r.igd, r.ggd),
                gd: (r.igs, r.ggs),
                vgstrap: r.vgstrap,
                vgdtrap: r.vgdtrap,
                pave: r.pave,
            }
        } else {
            let r = self.model.ids(op, vgs, vgd, gmin, memory);
            Point {
                channel: (r.current, r.gm, r.gds),
                gs: (r.igs, r.ggs),
                gd: (r.igd, r.ggd),
                vgstrap: r.vgstrap,
                vgdtrap: r.vgdtrap,
                pave: r.pave,
            }
        };
        let values = [
            point.channel.0,
            point.channel.1,
            point.channel.2,
            point.gs.0,
            point.gs.1,
            point.gd.0,
            point.gd.1,
            point.vgstrap,
            point.vgdtrap,
            point.pave,
        ];
        if values.iter().any(|v| !v.is_finite()) {
            return Err(SpiceError::Numerical {
                context: format!("JFET2 {}", self.name),
                message: "nonfinite Parker-Skellern channel/junction equations".into(),
            });
        }
        Ok(point)
    }

    /// Normalized `[vgs, vgd]` from physical node voltages of d', g, s'.
    fn normalized(&self, v: [Real; 3]) -> [Real; 2] {
        lifecycle::normalized(self.model.pol, v)
    }

    fn numerical(&self, message: &str) -> SpiceError {
        SpiceError::Numerical {
            context: format!("JFET2 {}", self.name),
            message: message.into(),
        }
    }

    /// The bias-dependent ask quantities of `jfet2ask.c` at the solution
    /// `voltage` (no limiting, DC filters): `vgs`, `vgd`, the
    /// multiplicity-scaled `gm`, `gds`, `ggs`, `ggd`, `igd`, and `vtrap`,
    /// `vpave` (normalized polarity, as C stores them). In transient every
    /// quantity but `vgs`/`vgd` depends on the accepted filter state or the
    /// charge companion, so those are refused there.
    fn operating_ask(
        &self,
        key: &str,
        context: &ModelContext,
        voltage: &dyn Fn(NodeId) -> Real,
        transient: bool,
    ) -> SpiceResult<Option<Real>> {
        if !matches!(
            key,
            "vgs" | "vgd" | "gm" | "gds" | "ggs" | "ggd" | "igd" | "vtrap" | "vpave"
        ) {
            return Ok(None);
        }
        let [vgs, vgd] = self.normalized(self.inner.map(voltage));
        if transient && !matches!(key, "vgs" | "vgd") {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "{}: transient @{}[{key}] (C reads the filtered state and charge companion)",
                    self.name, self.name
                ),
                "src/spicelib/devices/jfet2/jfet2ask.c, psmodel.c",
            ));
        }
        let op = self.operating(context)?;
        let point = self.point(&op, vgs, vgd, context.gmin, None)?;
        let m = self.instance.m;
        Ok(Some(match key {
            "vgs" => vgs,
            "vgd" => vgd,
            "gm" => m * point.channel.1,
            "gds" => m * point.channel.2,
            "ggs" => m * point.gs.1,
            "ggd" => m * point.gd.1,
            "igd" => m * point.gd.0,
            "vtrap" => point.vgdtrap,
            _ => point.pave,
        }))
    }
}

/// Rewrites `slot` with its value before this load (C leaves `CKTstate0`
/// untouched there): the previous load's, else the accepted or carried one,
/// else zero.
impl Device for Jfet2 {
    /// `jfet2set.c`: C's device type and `TSTALLOC` order.
    fn c_device_type(&self) -> crate::devices::setup::CDeviceType {
        crate::devices::setup::CDeviceType::Jfet2
    }

    /// `jfet2set.c` `TSTALLOC` order.
    fn setup_matrix(
        &self,
        setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        lifecycle::setup_gate_matrix(setup, &self.terminals, self.inner);
        Ok(true)
    }
    /// `jfet2ask.c` scalar asks: `area` (C reports `area * m`), `m`, `temp`,
    /// `dtemp`, `ic-vds` and `ic-vgs` (when given).
    fn observation_parameter(
        &self,
        key: &str,
        context: &ModelContext,
    ) -> SpiceResult<Option<Real>> {
        Ok(jfet::observation_parameter(
            &self.instance,
            &self.initial,
            key,
            context,
        ))
    }
    fn observation_operating(
        &self,
        key: &str,
        context: &ModelContext,
        voltage: &dyn Fn(NodeId) -> Real,
        transient: bool,
    ) -> SpiceResult<Option<Real>> {
        self.operating_ask(key, context, voltage, transient)
    }
    /// `jfet2ask.c` reports `id`, `ig` and `is` in the device's normalized
    /// polarity (`JFET2cd`, `JFET2cg` are stored before the `type` factor).
    fn observation_current_sign(&self) -> Real {
        self.model.pol
    }
    fn name(&self) -> &str {
        &self.name
    }
    fn designator(&self) -> char {
        'j'
    }
    fn terminals(&self) -> &[NodeId] {
        &self.terminals
    }
    fn is_nonlinear(&self) -> bool {
        true
    }
    fn has_start_settings(&self) -> bool {
        self.initial.off
    }
    /// With `TAUG` or `TAUD` the small-signal load of `PSacload` is not
    /// affine in `j omega`, so AC re-assembles it at every frequency (the
    /// operating point itself does not depend on it).
    fn small_signal_depends_on_frequency(&self) -> bool {
        self.model.taug != 0. || self.model.taud != 0.
    }
    fn state_count(&self) -> usize {
        slot::COUNT
    }
    /// `jfet2trun.c`: both gate charges control the timestep (not `qds`).
    fn truncation_slots(&self) -> Vec<usize> {
        vec![slot::QGS, slot::QGD]
    }
    fn instance_parameter(&self, keyword: &str) -> Option<&'static str> {
        jfet::instance_parameter(keyword)
    }
    /// `jfet2.c`'s `IFparm` tables under C's `sgen` walk (#164).
    fn sensitivity(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        Ok(Box::new(Jfet2Sensitivity::new(self, context)))
    }
    /// `JFET2param` then `JFET2temp`, with the instance schema domains.
    fn with_instance_parameter(
        &self,
        parameter: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        let (instance, initial) =
            jfet::swept_instance(&self.name, &self.instance, &self.initial, parameter, value)?;
        let device = Self {
            name: self.name.clone(),
            model_name: self.model_name.clone(),
            terminals: self.terminals.clone(),
            inner: self.inner,
            model: self.model,
            instance,
            derived: TemperatureCache::new(),
            initial,
        };
        device.operating(context)?;
        Ok(Box::new(device))
    }
    /// `JFET2load` (`jfet2load.c`) without bypass, which ngspice disables by
    /// default. As for level 1 the load's own `icheck`/`cghat`/`cdhat` test
    /// is the convergence test (`DEVconvTest = NULL`).
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()> {
        let phase = LoadPhase::new(context.mode, context.integration, &context.states, "JFET2")?;
        let op = self.operating(&context.model_context())?;
        let pol = self.model.pol;
        let ([vgs, vgd], start) = phase.junctions(
            Junctions {
                pol,
                off: self.initial.off,
                vt: op.vt,
                vcrit: op.vcrit,
                vto: self.model.vto,
            },
            context,
            &self.terminals,
            self.inner,
            &self.initial,
        );
        let state1 = |states: &crate::devices::DeviceState<'_>, s: usize, what: &str| {
            states
                .accepted(1, s)
                .ok_or_else(|| self.numerical(&format!("missing accepted {what}")))
        };
        // psmodel.h: TRAN_ANAL is MODETRAN, the `BEFORE` values CKTstate1
        // and STEP `CKTdelta`.
        let memory = if let Some(dt) = phase.step() {
            Some(Memory {
                step: dt,
                vgstrap: state1(&context.states, slot::VGSTRAP, "vgstrap")?,
                vgdtrap: state1(&context.states, slot::VGDTRAP, "vtrap")?,
                powr: state1(&context.states, slot::PAVE, "pave")?,
            })
        } else {
            None
        };
        let point = self.point(&op, vgs, vgd, context.gmin, memory.as_ref())?;
        let vds = vgs - vgd;
        let mut l = Loaded::from_currents([vgs, vgd], point.channel, point.gs, point.gd);
        // Charge storage: C evaluates the charges in transient, `.dc` sweeps,
        // MODEINITSMSIG and the `uic` operating point, and integrates them in
        // transient only. Elsewhere the slots keep their previous values, as
        // C's CKTstate0 does.
        if phase.updates_statz() {
            let capds = op.capds;
            let gates = if phase.step().is_some() {
                lifecycle::incremental_statz(
                    &context.states,
                    [vgs, vgd],
                    lifecycle::StatzCapacitance::Averaged,
                    |what| self.numerical(&format!("missing accepted {what}")),
                    |gs, gd| Ok(self.model.qgg(&op, gs, gd)),
                )?
            } else {
                // PScharge outside transient: total charge in both slots.
                let (q, capgs, capgd) = self.model.qgg(&op, vgs, vgd);
                lifecycle::gate_charges(q, q, capgs, capgd)
            };
            let drain = ChargeBranch::new(slot::QDS, capds * vds, capds);
            if gates
                .iter()
                .any(|branch| !branch.charge.is_finite() || !branch.capacitance.is_finite())
                || !drain.charge.is_finite()
            {
                return Err(self.numerical("nonfinite Statz gate charge"));
            }
            GateUpdate::Parker { gates, drain }.apply(phase, &mut context.states, &mut l)?;
        } else {
            for s in [
                slot::QGS,
                slot::QGS + 1,
                slot::QGD,
                slot::QGD + 1,
                slot::QDS,
                slot::QDS + 1,
            ] {
                carry(&mut context.states, s)?;
            }
        }
        carry(&mut context.states, slot::UNKNOWN)?;
        for (s, value) in [
            (slot::VGSTRAP, point.vgstrap),
            (slot::VGDTRAP, point.vgdtrap),
            (slot::PAVE, point.pave),
        ] {
            context.states.set(s, value)?;
        }
        GateStamp {
            terminals: &self.terminals,
            inner: self.inner,
            polarity: pol,
            multiplicity: self.instance.m,
            series: jfet::series_conductances(self.instance.area, self.model.rd, self.model.rs),
        }
        .finish(context, start, &l)
    }
    /// `jfet2acld.c` after C's `MODEINITSMSIG` load: `PSacload`'s
    /// frequency-dependent transconductance and output conductance at the
    /// stored `gm`, `gds` and `cd`, `CDS` added to the imaginary output
    /// conductance, then the conductances and gate capacitances as for
    /// level 1, every stamp scaled by `m` in C's order and grouping.
    fn assemble_small_signal(
        &self,
        context: &mut LinearContext<'_>,
        bias: &Vector,
    ) -> SpiceResult<()> {
        let op = self.operating(context.model_context)?;
        let unknowns = context.unknowns;
        let voltage = |n: NodeId| unknowns.node_row(n).and_then(|r| bias.get(r)).unwrap_or(0.);
        let [vgs, vgd] = self.normalized(self.inner.map(voltage));
        let point = self.point(&op, vgs, vgd, context.model_context.gmin, None)?;
        let (_, capgs, capgd) = self.model.qgg(&op, vgs, vgd);
        let omega = 2.0 * std::f64::consts::PI * context.model_context.frequency;
        // C passes the stored `JFET2cd` (channel minus gate-drain diode).
        let (cdrain, gm, gds) = point.channel;
        let cd = cdrain - point.gd.0;
        let (gm, xgm, gds, xgds) = self.model.acload(&op, vgs, vgd, cd, omega, gm, gds);
        if [capgs, capgd, gm, xgm, gds, xgds]
            .iter()
            .any(|v| !v.is_finite())
        {
            return Err(self.numerical("nonfinite small-signal load"));
        }
        let m = self.instance.m;
        let [dp, g, sp] = self.inner;
        if self.small_signal_depends_on_frequency() {
            // xgds += qds * CKTomega, with MODEINITSMSIG's qds = CDS * area.
            let xgds = xgds + op.capds * omega;
            for (row, col, terms, scale) in [
                (dp, dp, &[xgds][..], m),
                (sp, sp, &[xgds, xgm][..], m),
                (dp, g, &[xgm][..], m),
                (dp, sp, &[xgds, xgm][..], -m),
                (sp, g, &[xgm][..], -m),
                (sp, dp, &[xgds][..], -m),
            ] {
                context.prescaled_dynamic_sum(row, col, terms, scale)?;
            }
        } else {
            // Without TAUG/TAUD, PSacload's imaginary parts are signed zeros,
            // so `xgds + qds * CKTomega` is `CDS * area * omega` exactly.
            for (row, col, scale) in [(dp, dp, m), (sp, sp, m), (dp, sp, -m), (sp, dp, -m)] {
                context.scaled_dynamic_sum(row, col, &[op.capds], scale)?;
            }
        }
        jfet::linear::ac_gate_capacitances(context, self.inner, m, capgs, capgd)?;
        jfet::linear::ac_conductances(
            context,
            &self.terminals,
            self.inner,
            m,
            jfet::series_conductances(self.instance.area, self.model.rd, self.model.rs),
            [gm, gds, point.gs.1, point.gd.1],
        )
    }
    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise> {
        Err(SpiceError::not_yet_ported(
            format!("noise analysis of JFET2 {}", self.name),
            "src/spicelib/devices/jfet2/jfet2noi.c",
        ))
    }
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Err(SpiceError::not_yet_ported(
            format!("distortion analysis of JFET2 {}", self.name),
            "src/spicelib/devices/jfet2/ (no DEVdisto: C omits the device from .disto)",
        ))
    }
}

// ---------------------------------------------------------------------------
// `.sens` (#164)
// ---------------------------------------------------------------------------

/// `JFET2pTable` entries `set_param()` accepts, in C's table order (`off` is a
/// flag, `ic` a real vector, the `*node` and `vtrap`/`vpave` asks are not
/// settable; `ic-vds`/`ic-vgs` are `IOPAU` and therefore AC-only).
const SENSITIVITY_INSTANCE: &[SensitivityParameter] = &[
    SensitivityParameter::dc("area"),
    SensitivityParameter::dc("m"),
    SensitivityParameter::ac("ic-vds"),
    SensitivityParameter::ac("ic-vgs"),
    SensitivityParameter::dc("temp"),
    SensitivityParameter::dc("dtemp"),
];

/// `JFET2mPTable` in C's order, which follows `jfet2parm.h`: `vto`/`pb` are
/// `IOPR` (redundant), `gd`/`gs` are ask-only, and `cds`/`cgd`/`cgs` are
/// `IOPA`, so those are AC-only.
const SENSITIVITY_MODEL: &[SensitivityParameter] = &[
    SensitivityParameter::dc("acgam"),
    SensitivityParameter::dc("af"),
    SensitivityParameter::dc("beta"),
    SensitivityParameter::ac("cds"),
    SensitivityParameter::ac("cgd"),
    SensitivityParameter::ac("cgs"),
    SensitivityParameter::dc("delta"),
    SensitivityParameter::dc("hfeta"),
    SensitivityParameter::dc("hfe1"),
    SensitivityParameter::dc("hfe2"),
    SensitivityParameter::dc("hfg1"),
    SensitivityParameter::dc("hfg2"),
    SensitivityParameter::dc("mvst"),
    SensitivityParameter::dc("mxi"),
    SensitivityParameter::dc("fc"),
    SensitivityParameter::dc("ibd"),
    SensitivityParameter::dc("is"),
    SensitivityParameter::dc("kf"),
    SensitivityParameter::dc("lambda"),
    SensitivityParameter::dc("lfgam"),
    SensitivityParameter::dc("lfg1"),
    SensitivityParameter::dc("lfg2"),
    SensitivityParameter::dc("n"),
    SensitivityParameter::dc("p"),
    SensitivityParameter::dc("vbi"),
    SensitivityParameter::dc("q"),
    SensitivityParameter::dc("rd"),
    SensitivityParameter::dc("rs"),
    SensitivityParameter::dc("taud"),
    SensitivityParameter::dc("taug"),
    SensitivityParameter::dc("vbd"),
    SensitivityParameter::dc("ver"),
    SensitivityParameter::dc("vst"),
    SensitivityParameter::dc("vt0"),
    SensitivityParameter::dc("xc"),
    SensitivityParameter::dc("xi"),
    SensitivityParameter::dc("z"),
    SensitivityParameter::dc("hfgam"),
    SensitivityParameter::dc("tnom"),
];

/// The record's model fields: the enumerated setters plus `pb` (`vbi`'s C
/// field) and the three fields only the noise routine reads. `tnom` is kelvin.
const SENSITIVITY_MODEL_FIELDS: &[&str] = &[
    "acgam", "af", "beta", "cds", "cgd", "cgs", "delta", "hfeta", "hfe1", "hfe2", "hfg1", "hfg2",
    "hfgam", "mvst", "mxi", "fc", "ibd", "is", "kf", "lambda", "lfgam", "lfg1", "lfg2", "n", "p",
    "vbi", "q", "rd", "rs", "taud", "taug", "vbd", "ver", "vst", "vt0", "xc", "xi", "z", "tnom",
];

/// The record's instance fields (`temp` is kelvin).
const SENSITIVITY_INSTANCE_FIELDS: &[&str] = &["area", "m", "ic-vds", "ic-vgs", "temp", "dtemp"];

/// A JFET2's `.sens` description: `jfet2.c`'s tables, `jfet2par.c`/
/// `jfet2mpar.c` setters, `jfet2set.c`'s defaults and `jfet2temp.c`'s instance
/// temperature. The load is the port's own [`Jfet2`] rebuilt from C's records.
#[derive(Debug)]
pub(super) struct Jfet2Sensitivity<'a> {
    device: &'a Jfet2,
    circuit_kelvin: Real,
    nominal_kelvin: Real,
}

impl<'a> Jfet2Sensitivity<'a> {
    pub(super) fn new(device: &'a Jfet2, context: &ModelContext) -> Self {
        Self {
            device,
            circuit_kelvin: context.temperature + CELSIUS_TO_KELVIN,
            nominal_kelvin: context.nominal_temperature + CELSIUS_TO_KELVIN,
        }
    }
}

impl SensitivityKernel for Jfet2Sensitivity<'_> {
    fn family(&self) -> SensitivityFamily {
        SensitivityFamily::Jfet2
    }

    fn model(&self) -> Option<&str> {
        Some(&self.device.model_name)
    }

    fn model_parameters(&self) -> &'static [SensitivityParameter] {
        SENSITIVITY_MODEL
    }

    fn instance_parameters(&self) -> &'static [SensitivityParameter] {
        SENSITIVITY_INSTANCE
    }

    fn records(&self) -> SpiceResult<(SensitivityRecord, SensitivityRecord)> {
        let (m, i) = (&self.device.model, &self.device.instance);
        let mut model = SensitivityRecord::new(self.family(), ParameterScope::Model);
        for keyword in SENSITIVITY_MODEL_FIELDS {
            let value = match *keyword {
                "acgam" => m.acgam,
                "af" => 1.,
                "beta" => m.beta,
                "cds" => m.cds,
                "cgd" => m.cgd,
                "cgs" => m.cgs,
                "delta" => m.delta,
                "hfeta" => m.hfeta,
                "hfe1" => m.hfe1,
                "hfe2" => m.hfe2,
                "hfg1" => m.hfg1,
                "hfg2" => m.hfg2,
                "hfgam" => m.hfgam,
                "mvst" => m.mvst,
                "mxi" => m.mxi,
                "fc" => m.fc,
                "ibd" => m.ibd,
                "is" => m.is,
                "lambda" => m.lambda,
                "lfgam" => m.lfgam,
                "lfg1" => m.lfg1,
                "lfg2" => m.lfg2,
                "n" => m.n,
                "p" => m.p,
                "vbi" => m.pb,
                "q" => m.q,
                "rd" => m.rd,
                "rs" => m.rs,
                "taud" => m.taud,
                "taug" => m.taug,
                "vbd" => m.vbd,
                "vst" => m.vst,
                "vt0" => m.vto,
                "xc" => m.xc,
                "xi" => m.xi,
                "z" => m.z,
                "tnom" => m
                    .tnom
                    .map_or(self.nominal_kelvin, |c| c + CELSIUS_TO_KELVIN),
                _ => 0.,
            };
            // `hfgam`'s C default is the model's `lfgam`, so its *given* flag
            // is exactly "the card gave it a different value" (`jfet2parm.h`:
            // `PARAM("hfgam", ..., model->JFET2lfgam, ...)`); a card that gives
            // the same value behaves identically either way.
            let given = match *keyword {
                "tnom" => m.tnom.is_some(),
                "hfgam" => m.hfgam != m.lfgam,
                _ => true,
            };
            model = model.with(keyword, value, given)?;
        }
        let mut instance = SensitivityRecord::new(self.family(), ParameterScope::Instance);
        for keyword in SENSITIVITY_INSTANCE_FIELDS {
            let (value, given) = match *keyword {
                "area" => (i.area, true),
                "m" => (i.m, true),
                "temp" => (
                    i.temp
                        .map_or(self.circuit_kelvin + i.dtemp, |c| c + CELSIUS_TO_KELVIN),
                    i.temp.is_some(),
                ),
                "dtemp" => (i.dtemp, true),
                "ic-vds" => (self.device.initial.values[0].unwrap_or(0.), true),
                _ => (self.device.initial.values[1].unwrap_or(0.), true),
            };
            instance = instance.with(keyword, value, given)?;
        }
        self.setup(&mut model, &mut instance)?;
        self.temperature(&mut model, &mut instance)?;
        Ok((model, instance))
    }

    /// `JFET2setup`: `jfet2parm.h`'s defaults for the fields the card left
    /// unset. `hfgam` is the only field whose default reads another one.
    fn setup(
        &self,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        for (keyword, value) in [
            ("acgam", 0.),
            ("af", 1.),
            ("beta", 1e-4),
            ("cds", 0.),
            ("cgd", 0.),
            ("cgs", 0.),
            ("delta", 0.),
            ("hfeta", 0.),
            ("hfe1", 0.),
            ("hfe2", 0.),
            ("hfg1", 0.),
            ("hfg2", 0.),
            ("mvst", 0.),
            ("mxi", 0.),
            ("fc", 0.5),
            ("ibd", 0.),
            ("is", 1e-14),
            ("kf", 0.),
            ("lambda", 0.),
            ("lfgam", 0.),
            ("lfg1", 0.),
            ("lfg2", 0.),
            ("n", 1.),
            ("p", 2.),
            ("vbi", 1.),
            ("q", 2.),
            ("rd", 0.),
            ("rs", 0.),
            ("taud", 0.),
            ("taug", 0.),
            ("vbd", 1.),
            ("ver", 0.),
            ("vst", 0.),
            ("vt0", -2.),
            ("xc", 0.),
            ("xi", 1000.),
            ("z", 1.),
        ] {
            if !model.given(keyword) {
                model.store(keyword, value)?;
            }
        }
        if !model.given("hfgam") {
            let lfgam = model.value("lfgam")?;
            model.store("hfgam", lfgam)?;
        }
        if !instance.given("area") {
            instance.store("area", 1.)?;
        }
        if !instance.given("m") {
            instance.store("m", 1.)?;
        }
        Ok(())
    }

    /// `jfet2temp.c`: `tnom` defaults to `CKTnomTemp`, `dtemp` to 0 and `temp`
    /// to `CKTtemp + dtemp`; every derived quantity is recomputed by the port's
    /// load from these records.
    fn temperature(
        &self,
        model: &mut SensitivityRecord,
        instance: &mut SensitivityRecord,
    ) -> SpiceResult<()> {
        if !model.given("tnom") {
            model.store("tnom", self.nominal_kelvin)?;
        }
        if !instance.given("dtemp") {
            instance.store("dtemp", 0.)?;
        }
        if !instance.given("temp") {
            let dtemp = instance.value("dtemp")?;
            instance.store("temp", self.circuit_kelvin + dtemp)?;
        }
        Ok(())
    }

    fn load(
        &self,
        _setup: RecordPair<'_>,
        current: RecordPair<'_>,
        mode: SensitivityMode,
    ) -> SpiceResult<SensitivityLoad> {
        if mode == SensitivityMode::Ac {
            return Err(SpiceError::circuit(
                "a JFET2's AC sensitivity load is its reset load",
            ));
        }
        Ok(SensitivityLoad::Replacement(Box::new(self.jfet2(current)?)))
    }

    /// `JFET2acLoad` after `cktsens.c`'s reset load: `gm`, `gds`, `ggs`,
    /// `ggd`, `vgs`, `vgd` and `cd` are the `CKTstate0` values `jfet2load.c`
    /// stored at zero junction voltages with the reset records, the charge
    /// slots `qgs`/`qgd`/`qds` are zero, and `PSacload`'s model coefficients,
    /// the series conductances, `area` and `m` come from `current`.
    fn reset_ac_load(
        &self,
        _setup: RecordPair<'_>,
        reset: AcReset<'_>,
        current: RecordPair<'_>,
    ) -> SpiceResult<SensitivityLoad> {
        Ok(SensitivityLoad::Replacement(Box::new(ResetJfet2 {
            state: self.jfet2(reset.state)?,
            current: self.jfet2(current)?,
            omega: reset.omega,
        })))
    }
}

impl Jfet2Sensitivity<'_> {
    /// The port's JFET2 with the records' parameters.
    fn jfet2(&self, current: RecordPair<'_>) -> SpiceResult<Jfet2> {
        let (m, i) = (current.model, current.instance);
        let model = Model {
            pol: self.device.model.pol,
            acgam: m.value("acgam")?,
            beta: m.value("beta")?,
            cds: m.value("cds")?,
            cgd: m.value("cgd")?,
            cgs: m.value("cgs")?,
            delta: m.value("delta")?,
            hfeta: m.value("hfeta")?,
            hfe1: m.value("hfe1")?,
            hfe2: m.value("hfe2")?,
            hfg1: m.value("hfg1")?,
            hfg2: m.value("hfg2")?,
            hfgam: m.value("hfgam")?,
            mvst: m.value("mvst")?,
            mxi: m.value("mxi")?,
            fc: m.value("fc")?,
            ibd: m.value("ibd")?,
            is: m.value("is")?,
            lambda: m.value("lambda")?,
            lfgam: m.value("lfgam")?,
            lfg1: m.value("lfg1")?,
            lfg2: m.value("lfg2")?,
            n: m.value("n")?,
            p: m.value("p")?,
            pb: m.value("vbi")?,
            q: m.value("q")?,
            rd: m.value("rd")?,
            rs: m.value("rs")?,
            taud: m.value("taud")?,
            taug: m.value("taug")?,
            vbd: m.value("vbd")?,
            vst: m.value("vst")?,
            vto: m.value("vt0")?,
            xc: m.value("xc")?,
            xi: m.value("xi")?,
            z: m.value("z")?,
            tnom: m
                .given("tnom")
                .then(|| m.value("tnom"))
                .transpose()?
                .map(|kelvin| kelvin - CELSIUS_TO_KELVIN),
        };
        let instance = Instance {
            area: i.value("area")?,
            m: i.value("m")?,
            temp: i
                .given("temp")
                .then(|| i.value("temp"))
                .transpose()?
                .map(|kelvin| kelvin - CELSIUS_TO_KELVIN),
            dtemp: i.value("dtemp")?,
        };
        Ok(Jfet2 {
            name: self.device.name.clone(),
            model_name: self.device.model_name.clone(),
            terminals: self.device.terminals.clone(),
            inner: self.device.inner,
            model,
            instance,
            derived: TemperatureCache::new(),
            initial: self.device.initial.clone(),
        })
    }
}

/// A JFET2's AC load in `cktsens.c`'s sweep
/// ([`crate::devices::sensitivity::DeviceSensitivity::reset_ac_load`]).
#[derive(Debug)]
struct ResetJfet2 {
    /// Built from the reset load's records: the junction conductances.
    state: Jfet2,
    /// Built from the current records: `PSacload`'s coefficients, series
    /// conductances and `m`.
    current: Jfet2,
    /// `CKTomega`.
    omega: Real,
}

impl Device for ResetJfet2 {
    fn name(&self) -> &str {
        &self.current.name
    }

    fn designator(&self) -> char {
        'j'
    }

    fn terminals(&self) -> &[NodeId] {
        &self.current.terminals
    }

    fn stamp(&self, _context: &mut StampContext<'_>) -> SpiceResult<()> {
        Err(SpiceError::circuit(format!(
            "{}: an AC sensitivity stand-in has no DC load",
            self.current.name
        )))
    }

    /// `jfet2acld.c`'s stamps in its order: `PSacload`'s imaginary parts,
    /// then the real parts (the gate charges' `xgs`/`xgd` are zero).
    fn assemble_linear(&self, context: &mut LinearContext<'_>) -> SpiceResult<()> {
        use crate::devices::sensitivity::{ResetStamp, ResetStamps, StampAt, StampPart};
        let gmin = context.model_context.gmin;
        let state = &self.state;
        let state_op = state.operating(context.model_context)?;
        let [vgs, vgd] = state.normalized([0.; 3]);
        let point = state.point(&state_op, vgs, vgd, gmin, None)?;
        let (cdrain, gm, gds) = point.channel;
        let cd = cdrain - point.gd.0;
        let current = &self.current;
        let op = current.operating(context.model_context)?;
        let omega = self.omega;
        let (gm, xgm, gds, xgds) = current.model.acload(&op, vgs, vgd, cd, omega, gm, gds);
        // `xgds += qds * CKTomega` with the reset's zero charge.
        let xgds = xgds + 0. * omega;
        let m = current.instance.m;
        let [dp, g, sp] = current.inner;
        let imaginary = [
            (dp, dp, m * xgds),
            (sp, sp, m * (xgds + xgm)),
            (dp, g, m * xgm),
            (dp, sp, -(m * (xgds + xgm))),
            (sp, g, -(m * xgm)),
            (sp, dp, -(m * xgds)),
        ];
        let stamps = imaginary
            .into_iter()
            .map(|(a, b, value)| ResetStamp {
                at: StampAt::Nodes(a, b),
                value,
                part: StampPart::Imaginary,
            })
            .collect();
        ResetStamps::new(&current.name, 'j', &current.terminals, stamps)?
            .assemble_linear(context)?;
        jfet::linear::ac_conductances(
            context,
            &current.terminals,
            current.inner,
            m,
            jfet::series_conductances(current.instance.area, current.model.rd, current.model.rs),
            [gm, gds, point.gs.1, point.gd.1],
        )
    }
}

static MODEL_SCHEMA: ScalarSchema<'static> = ScalarSchema::new(MODEL);

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Model {
        Model {
            pol: 1.,
            acgam: 0.1,
            beta: 1.3e-3,
            cds: 0.5e-12,
            cgd: 1e-12,
            cgs: 2e-12,
            delta: 0.3,
            hfeta: 0.02,
            hfe1: 0.01,
            hfe2: 0.015,
            hfg1: 0.012,
            hfg2: 0.008,
            hfgam: 0.04,
            mvst: 0.2,
            mxi: 0.1,
            fc: 0.5,
            ibd: 1e-9,
            is: 1e-14,
            lambda: 0.03,
            lfgam: 0.05,
            lfg1: 0.02,
            lfg2: 0.01,
            n: 1.2,
            p: 2.3,
            pb: 0.9,
            q: 2.1,
            rd: 0.,
            rs: 0.,
            taud: 1e-7,
            taug: 2e-7,
            vbd: 2.,
            vst: 0.06,
            vto: -2.,
            xc: 0.2,
            xi: 8.,
            z: 0.6,
            tnom: None,
        }
    }

    fn device(model: Model) -> (Jfet2, Operating) {
        let device = Jfet2 {
            name: "j1".into(),
            model_name: "jm".into(),
            terminals: vec![],
            inner: [NodeId::GROUND; 3],
            model,
            instance: Instance {
                area: 1.5,
                m: 1.,
                temp: None,
                dtemp: 0.,
            },
            derived: TemperatureCache::new(),
            initial: crate::devices::initial::InstanceInitial::default(),
        };
        let op = device.operating(&ModelContext::default()).unwrap();
        (device, op)
    }

    /// gm = dI/dvgs at fixed vds and gds = dI/dvds at fixed vgs, as well as
    /// the diode conductances, in every region of both modes, with and
    /// without the transient filters.
    #[test]
    fn ids_partials_are_the_current_derivatives() {
        let (device, op) = device(model());
        let memory = Memory {
            step: 3e-8,
            vgstrap: -0.7,
            vgdtrap: -2.4,
            powr: 4e-3,
        };
        for memory in [None, Some(&memory)] {
            for vgs in [-3.2, -2.3, -1.9, -1., 0., 0.4] {
                for vds in [-3., -1.2, -0.4, -0.05, 0.05, 0.4, 1.2, 3.] {
                    let at = |vgs: Real, vds: Real| {
                        device.point(&op, vgs, vgs - vds, 1e-12, memory).unwrap()
                    };
                    let c = at(vgs, vds);
                    let h = 1e-6;
                    let gm = (at(vgs + h, vds).channel.0 - at(vgs - h, vds).channel.0) / (2. * h);
                    let gds = (at(vgs, vds + h).channel.0 - at(vgs, vds - h).channel.0) / (2. * h);
                    let scale = c.channel.1.abs().max(c.channel.2.abs()).max(1e-9);
                    let tag = format!("{vgs} {vds} {}", memory.is_some());
                    assert!(
                        (gm - c.channel.1).abs() <= 1e-5 * scale,
                        "{tag}: gm {gm} {:?}",
                        c.channel
                    );
                    assert!(
                        (gds - c.channel.2).abs() <= 1e-5 * scale,
                        "{tag}: gds {gds} {:?}",
                        c.channel
                    );
                    // d igs / d vgs at fixed vgd = ggs.
                    let ggs = (device
                        .point(&op, vgs + h, vgs - vds, 1e-12, memory)
                        .unwrap()
                        .gs
                        .0
                        - device
                            .point(&op, vgs - h, vgs - vds, 1e-12, memory)
                            .unwrap()
                            .gs
                            .0)
                        / (2. * h);
                    assert!(
                        (ggs - c.gs.1).abs() <= 1e-5 * c.gs.1.abs() + 1e-15,
                        "{tag}: ggs"
                    );
                }
            }
        }
    }

    #[test]
    fn the_channel_is_antisymmetric_without_filters() {
        let (device, op) = device(model());
        for (vgs, vgd) in [(-1., -1.5), (0.2, -2.), (-1.9, -1.95)] {
            let forward = device.point(&op, vgs, vgd, 0., None).unwrap().channel.0;
            let reverse = device.point(&op, vgd, vgs, 0., None).unwrap().channel.0;
            assert!((forward + reverse).abs() <= 1e-15, "{vgs} {vgd}");
        }
    }

    #[test]
    fn statz_charge_partials_are_its_derivatives() {
        let (device, op) = device(model());
        for vgs in [-4., -2.1, -1., 0., 0.3, 0.6, 0.85] {
            for vgd in [-5., -2., -0.5, 0.2, 0.7] {
                let (_, cgs, cgd) = device.model.qgg(&op, vgs, vgd);
                let h = 1e-7;
                let q = |a: Real, b: Real| device.model.qgg(&op, a, b).0;
                let ngs = (q(vgs + h, vgd) - q(vgs - h, vgd)) / (2. * h);
                let ngd = (q(vgs, vgd + h) - q(vgs, vgd - h)) / (2. * h);
                let scale = cgs.abs().max(cgd.abs());
                assert!(
                    (ngs - cgs).abs() <= 1e-6 * scale,
                    "{vgs} {vgd}: cgs {ngs} {cgs}"
                );
                assert!(
                    (ngd - cgd).abs() <= 1e-6 * scale,
                    "{vgs} {vgd}: cgd {ngd} {cgd}"
                );
            }
        }
    }

    #[test]
    fn ac_load_reduces_to_the_dc_conductances_without_time_constants() {
        let mut m = model();
        m.taug = 0.;
        m.taud = 0.;
        let (device, op) = device(m);
        let p = device.point(&op, -0.8, -3., 1e-12, None).unwrap();
        let (gm, xgm, gds, xgds) = device.model.acload(
            &op,
            -0.8,
            -3.,
            p.channel.0 - p.gd.0,
            1e7,
            p.channel.1,
            p.channel.2,
        );
        assert!(
            (gm - p.channel.1).abs() <= 1e-12 * gm.abs(),
            "{gm} {:?}",
            p.channel
        );
        assert!(
            (gds - p.channel.2).abs() <= 1e-12 * gds.abs(),
            "{gds} {:?}",
            p.channel
        );
        assert_eq!((xgm, xgds), (0., 0.));
    }
}
