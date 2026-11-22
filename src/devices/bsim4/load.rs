//! `b4ld.c` (`BSIM4load`): the DC/transient load of one instance, and
//! `BSIM4polyDepletion`.
//!
//! The load follows C statement by statement over C's own struct members and
//! local names, so the arithmetic (operand order, grouping, `exp`/`log`/`pow`
//! forms) is C's. C's labels become structured control flow:
//!
//! - `goto finished` (C-V skipped when `xpart < 0` or no charge is needed)
//!   is the `goto_finished` flag guarding the block before `finished:`;
//! - `goto line850`/`goto line1000` out of the charge block are breaks out of
//!   the labelled `'charges` block with flags, and leaving that block
//!   normally is C's `goto line860`; `line850:` and `line860:` are the two
//!   arms of an `if`, both followed by `line900:`'s stamps, all skipped for
//!   `line1000` (`MODEINITSMSIG`);
//! - C's bypass branch (`goto line755`/`line850` when `CKTbypass` is set) is
//!   not compiled: the port runs with `bypass = 0`, C's default, so `ByPass`
//!   stays 0 (`NOBYPASS` path), and `NEWCONV` is defined as in C.
// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols keep their original distinct names for source-level audit"
)]
#![expect(
    clippy::manual_midpoint,
    reason = "C source midpoint arithmetic retains its original evaluation and rounding order"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]
#![expect(
    non_snake_case,
    reason = "preserves upstream C identifiers for equation-by-equation comparison"
)] // C's member and local names.
#![expect(
    unused_assignments,
    reason = "C state-machine locals retain the reference assignment sequence"
)]
#![expect(
    unused_variables,
    reason = "C-translated intermediate variables preserve the reference layout"
)]
#![expect(
    non_upper_case_globals,
    reason = "constant names intentionally match the reference device equations"
)] // C's `Charge_q`.
#![expect(
    unused_parens,
    reason = "preserves source C expression grouping during numerical port verification"
)] // C's parenthesisation is kept verbatim.
#![expect(
    clippy::assign_op_pattern,
    reason = "retains the reference calculation grouping and floating-point operation order"
)] // `x = x * y` is kept as C writes it.
#![expect(
    clippy::collapsible_if,
    reason = "preserves the reference conditional order and side-effect boundaries"
)] // C's if nesting.
#![expect(
    clippy::float_cmp,
    reason = "exact equality is part of the reference device model branch conditions"
)] // C compares floating point exactly.
#![expect(
    clippy::too_many_lines,
    reason = "keeps the reference numerical routine in the original statement and rounding order"
)] // One C function.
#![expect(
    clippy::cognitive_complexity,
    reason = "single translated reference routine retains its checked evaluation sequence"
)] // One C function.
use super::fields::{Instance, Model, SizeParam};
use super::host::{LoadCkt, mode, state};
use crate::primitives::SpiceResult;

mod bias;
mod channel;
mod junction;
mod leakage;

const MAX_EXPL: f64 = 2.688_117_142e+43;
const MIN_EXPL: f64 = 3.720_075_976e-44;
const EXPL_THRESHOLD: f64 = 100.0;
const MAX_EXP: f64 = 5.834_617_425e14;
const MIN_EXP: f64 = 1.713_908_431e-15;
const EXP_THRESHOLD: f64 = 34.0;
const EPS0: f64 = 8.85418e-12;
const EPSSI: f64 = 1.03594e-10;
const Charge_q: f64 = 1.60219e-19;
const DELTA_1: f64 = 0.02;
const DELTA_3: f64 = 0.02;
const DELTA_4: f64 = 0.02;
/// Smoothing coefficient.
const MM: i32 = 3;
/// `const.h`'s `CHARGE`.
const CHARGE: f64 = 1.602_176_620_8e-19;
/// `CONSTvt0 = CONSTboltz * REFTEMP / CHARGE` (`main.c`), REFTEMP = 27 C.
const CONST_VT0: f64 = 1.380_648_52e-23 * (27.0 + 273.15) / CHARGE;

fn exp(x: f64) -> f64 {
    x.exp()
}
fn log(x: f64) -> f64 {
    x.ln()
}
fn sqrt(x: f64) -> f64 {
    x.sqrt()
}
const fn fabs(x: f64) -> f64 {
    x.abs()
}
/// `b4ld.c`'s `DEXP(A, B, C)` macro: returns `(B, C)`.
fn DEXP(A: f64) -> (f64, f64) {
    if A > EXP_THRESHOLD {
        (MAX_EXP * (1.0 + (A) - EXP_THRESHOLD), MAX_EXP)
    } else if A < -EXP_THRESHOLD {
        (MIN_EXP, 0.0)
    } else {
        let B = exp(A);
        (B, B)
    }
}

/// `BSIM4polyDepletion`: the poly-gate depletion effective gate voltage and
/// its gate derivative, `(Vgs_eff, dVgs_eff_dVg)`.
pub(super) fn BSIM4polyDepletion(
    phi: f64,
    ngate: f64,
    epsgate: f64,
    coxe: f64,
    Vgs: f64,
) -> (f64, f64) {
    // Poly gate Si depletion effect.
    if (ngate > 1.0e18) && (ngate < 1.0e25) && (Vgs > phi) && (epsgate != 0.0) {
        let T1 = 1.0e6 * CHARGE * epsgate * ngate / (coxe * coxe);
        let T8 = Vgs - phi;
        let T4 = sqrt(1.0 + 2.0 * T8 / T1);
        let T2 = 2.0 * T8 / (T4 + 1.0);
        let T3 = 0.5 * T2 * T2 / T1; // Vpoly
        let T7 = 1.12 - T3 - 0.05;
        let T6 = sqrt(T7 * T7 + 0.224);
        let T5 = 1.12 - 0.5 * (T7 + T6);
        (Vgs - T5, 1.0 - (0.5 - 0.5 / T4) * (1.0 + T7 / T6))
    } else {
        (Vgs, 1.0)
    }
}

/// `BSIM4load` for one instance: evaluates the device at the voltages C's
/// mode selects, writes `CKTstate0`, integrates charges and adds the
/// matrix and right-hand-side terms, as C does.
///
/// # Errors
/// `NIintegrate` failures (C returns its error).
pub(super) fn load(
    model: &Model,
    here: &mut Instance,
    pParam: &SizeParam,
    ckt: &mut dyn LoadCkt,
) -> SpiceResult<()> {
    // Remaining charge and stamp scratch; physics results are typed stage inputs.
    let ceqgstot: f64;
    let ceqgdtot: f64;
    let gstot: f64;
    let gstotd: f64;
    let gstotg: f64;
    let gstots: f64;
    let gstotb: f64;
    let gspr: f64;
    let gdtot: f64;
    let gdtotd: f64;
    let gdtotg: f64;
    let gdtots: f64;
    let gdtotb: f64;
    let gdpr: f64;
    let geltd: f64;
    let gcrg: f64;
    let mut gcrgg: f64;
    let gcrgd: f64;
    let gcrgs: f64;
    let gcrgb: f64;
    let mut ceqgcrg: f64;
    let mut gcgmgmb: f64 = 0.0;
    let mut gcgmdb: f64 = 0.0;
    let mut gcgmsb: f64 = 0.0;
    let mut gcdgmb: f64 = 0.0;
    let mut gcsgmb: f64 = 0.0;
    let mut gcgmbb: f64 = 0.0;
    let mut gcbgmb: f64 = 0.0;
    let qgmb: f64;
    let mut qgmid: f64 = 0.0;
    let mut ceqqgmid: f64;
    let ag0: f64;
    let qgb: f64;
    let VgstNVt: f64;
    let mut ExpVgst: f64;
    let mut ceqqb: f64;
    let mut ceqqd: f64;
    let mut ceqqg: f64;
    let mut ceqqjd: f64 = 0.0;
    let mut ceqqjs: f64 = 0.0;
    let mut ceq: f64;
    let mut geq: f64;
    let ceqdrn: f64;
    let ceqbd: f64;
    let ceqbs: f64;
    let ceqjd: f64;
    let ceqjs: f64;
    let gjbd: f64;
    let gjbs: f64;
    let Vfbeff: f64;
    let dVfbeff_dVg: f64;
    let dVfbeff_dVb: f64;
    let mut V3: f64;
    let V4: f64;
    let mut gcbdb: f64 = 0.0;
    let mut gcbgb: f64 = 0.0;
    let mut gcbsb: f64 = 0.0;
    let mut gcddb: f64 = 0.0;
    let mut gcdgb: f64 = 0.0;
    let mut gcdsb: f64 = 0.0;
    let mut gcgdb: f64 = 0.0;
    let mut gcggb: f64 = 0.0;
    let mut gcgsb: f64 = 0.0;
    let mut gcsdb: f64 = 0.0;
    let mut gcgbb: f64 = 0.0;
    let mut gcdbb: f64 = 0.0;
    let mut gcsbb: f64 = 0.0;
    let mut gcbbb: f64 = 0.0;
    let mut gcdbdb: f64 = 0.0;
    let mut gcsbsb: f64 = 0.0;
    let mut gcsgb: f64 = 0.0;
    let mut gcssb: f64 = 0.0;
    let mut qgate: f64 = 0.0;
    let mut qbulk: f64 = 0.0;
    let mut qdrn: f64 = 0.0;
    let mut qsrc: f64;
    let cqgate: f64;
    let cqbody: f64;
    let cqdrn: f64;
    let Gmbs: f64;
    let FwdSum: f64;
    let RevSum: f64;
    let Istoteq: f64;
    let gIstotg: f64;
    let gIstotd: f64;
    let gIstots: f64;
    let gIstotb: f64;
    let Idtoteq: f64;
    let gIdtotg: f64;
    let gIdtotd: f64;
    let gIdtots: f64;
    let gIdtotb: f64;
    let Ibtoteq: f64;
    let gIbtotg: f64;
    let gIbtotd: f64;
    let gIbtots: f64;
    let gIbtotb: f64;
    let Igtoteq: f64;
    let gIgtotg: f64;
    let gIgtotd: f64;
    let gIgtots: f64;
    let gIgtotb: f64;
    let Vfb: f64;
    let Vth: f64;
    let dVgst_dVg: f64;
    let dVgst_dVb: f64;
    let voffcv: f64;
    let noff: f64;
    let dnoff_dVd: f64;
    let dnoff_dVb: f64;
    let mut CoxWLcen: f64;
    let mut QovCox: f64;
    let LINK: f64;
    let DeltaPhi: f64;
    let dDeltaPhi_dVg: f64;
    let VgDP: f64;
    let dVgDP_dVg: f64;
    let Cox: f64;
    let mut Tox: f64;
    let mut Tcen: f64;
    let mut dTcen_dVg: f64;
    let dTcen_dVd: f64;
    let mut dTcen_dVb: f64;
    let mut Ccen: f64;
    let mut Coxeff: f64;
    let dCoxeff_dVd: f64;
    let mut dCoxeff_dVg: f64;
    let mut dCoxeff_dVb: f64;
    let Denomi: f64;
    let Vdsat: f64;
    let dVdsat_dVg: f64;
    let dVdsat_dVb: f64;
    let dAlphaz_dVg: f64;
    let dAlphaz_dVb: f64;
    let VbseffCV: f64;
    let dVbseffCV_dVb: f64;
    let Arg1: f64;
    let One_Third_CoxWL: f64;
    let Two_Third_CoxWL: f64;
    let Alphaz: f64;
    let mut CoxWL: f64;
    let mut T0: f64;
    let mut dT0_dVg: f64;
    let mut dT0_dVb: f64;
    let mut T1: f64;
    let mut dT1_dVg: f64;
    let dT1_dVd: f64;
    let dT1_dVb: f64;
    let mut T2: f64;
    let mut T3: f64;
    let mut T4: f64;
    let mut T5: f64;
    let mut T6: f64;
    let mut T7: f64;
    let mut T8: f64;
    let mut T9: f64;
    let mut dT9_dVg: f64;
    let dT9_dVd: f64;
    let dT9_dVb: f64;
    let mut T10: f64;
    let mut dT10_dVg: f64;
    let dT10_dVb: f64;
    let dT10_dVd: f64;
    let mut T11: f64;
    let mut T12: f64;
    let mut tmp: f64;
    let tmp1: f64;
    let Vgsteff: f64;
    let mut dVgsteff_dVg: f64;
    let mut dVgsteff_dVd: f64;
    let dVgsteff_dVb: f64;
    let mut VdseffCV: f64;
    let mut dVdseffCV_dVg: f64;
    let dVdseffCV_dVd: f64;
    let mut dVdseffCV_dVb: f64;
    let Gm: f64;
    let VdsatCV: f64;
    let AbulkCV: f64;
    let dAbulkCV_dVb: f64;
    let mut qcheq: f64;
    let mut gqdef: f64 = 0.0;
    let mut cqdef: f64 = 0.0;
    let mut cqcheq: f64 = 0.0;
    let mut gcqdb: f64 = 0.0;
    let mut gcqsb: f64 = 0.0;
    let mut gcqgb: f64 = 0.0;
    let mut gcqbb: f64 = 0.0;
    let mut dxpart: f64 = 0.0;
    let mut sxpart: f64 = 0.0;
    let mut ggtg: f64 = 0.0;
    let mut ggtd: f64 = 0.0;
    let mut ggts: f64 = 0.0;
    let mut ggtb: f64 = 0.0;
    let mut ddxpart_dVd: f64 = 0.0;
    let mut ddxpart_dVg: f64 = 0.0;
    let mut ddxpart_dVb: f64 = 0.0;
    let mut ddxpart_dVs: f64 = 0.0;
    let mut dsxpart_dVd: f64 = 0.0;
    let mut dsxpart_dVg: f64 = 0.0;
    let mut dsxpart_dVb: f64 = 0.0;
    let mut dsxpart_dVs: f64 = 0.0;
    let gbspsp: f64;
    let gbbdp: f64;
    let gbbsp: f64;
    let gbspg: f64;
    let gbspb: f64;
    let gbspdp: f64;
    let gbdpdp: f64;
    let gbdpg: f64;
    let gbdpb: f64;
    let gbdpsp: f64;
    let mut qgdo: f64;
    let mut qgso: f64;
    let mut cgdo: f64;
    let mut cgso: f64;
    let Cgg: f64;
    let Cgd: f64;
    let mut Cgb: f64;
    let Cdg: f64;
    let Cdd: f64;
    let Cds: f64;
    let mut Csg: f64;
    let mut Csd: f64;
    let Css: f64;
    let mut Csb: f64;
    let Cbg: f64;
    let Cbd: f64;
    let mut Cbb: f64;
    let mut Cgg1: f64;
    let Cgb1: f64;
    let Cgd1: f64;
    let mut Cbg1: f64;
    let Cbb1: f64;
    let Cbd1: f64;
    let Qac0: f64;
    let Qsub0: f64;
    let dQac0_dVg: f64;
    let dQac0_dVb: f64;
    let dQsub0_dVg: f64;
    let dQsub0_dVd: f64;
    let dQsub0_dVb: f64;
    let ggidld: f64;
    let ggidlg: f64;
    let ggidlb: f64;
    let ggislg: f64;
    let ggislb: f64;
    let ggisls: f64;
    let vgdx: f64;
    let vgsx: f64;
    let m: f64;
    let mult_i: f64;
    let mult_q: f64;
    let ScalingFactor = 1.0e-9;
    let bias = bias::select(model, here, ckt);
    let junction_bias = junction::JunctionBias {
        source: bias.vbs_jct,
        drain: bias.vbd_jct,
    };
    junction::current(model, here, pParam, ckt, junction_bias);
    let electrostatics = channel::electrostatics(model, here, pParam, ckt, &bias);
    let transport = channel::transport(model, here, pParam, &bias, &electrostatics);
    let cdrain = leakage::evaluate(model, here, pParam, &bias, &electrostatics, &transport);
    let &bias::LoadBias {
        ChargeComputationNeeded,
        Check,
        qdef,
        vbd,
        vbd_jct,
        vbs,
        vbs_jct,
        vdbd,
        vdbs,
        vdes,
        vds,
        vgb,
        vgd,
        vges,
        vgmb,
        vgmd,
        vgms,
        vgs,
        vsbs,
        vses,
        ..
    } = &bias;
    let &channel::Electrostatics {
        Phis,
        Vbs,
        Vbseff,
        Vds,
        Vgs_eff,
        mut Vgst,
        Vtm,
        dPhis_dVb,
        dVbseff_dVb,
        dVgs_eff_dVg,
        mut dVth_dVb,
        dVth_dVd,
        dn_dVb,
        dn_dVd,
        dsqrtPhis_dVb,
        epssub,
        n,
        sqrtPhis,
        ..
    } = &electrostatics;
    let &channel::Transport {
        Abulk0_Q,
        dAbulk0_Q_dVb,
        ..
    } = &transport;
    // C `goto finished` (only from the first branch below) skips the
    // block between this if/else chain and the `finished:` label.
    let mut goto_finished = false;
    if (model.xpart < 0.0) || (ChargeComputationNeeded == 0) {
        qbulk = 0.0;
        qsrc = qbulk;
        qdrn = qsrc;
        qgate = qdrn;
        here.cgdb = 0.0;
        here.cgsb = here.cgdb;
        here.cggb = here.cgsb;
        here.cddb = 0.0;
        here.cdsb = here.cddb;
        here.cdgb = here.cdsb;
        here.cbdb = 0.0;
        here.cbsb = here.cbdb;
        here.cbgb = here.cbsb;
        here.csdb = 0.0;
        here.cssb = here.csdb;
        here.csgb = here.cssb;
        here.cbbb = 0.0;
        here.cdbb = here.cbbb;
        here.csbb = here.cdbb;
        here.cgbb = here.csbb;
        here.cqbb = 0.0;
        here.cqgb = here.cqbb;
        here.cqsb = here.cqgb;
        here.cqdb = here.cqsb;
        here.gtau = 0.0;
        goto_finished = true;
    } else if model.capMod == 0 {
        if Vbseff < 0.0 {
            VbseffCV = Vbs;
            dVbseffCV_dVb = 1.0;
        } else {
            VbseffCV = pParam.phi - Phis;
            dVbseffCV_dVb = -dPhis_dVb * dVbseff_dVb;
        }
        Vfb = pParam.vfbcv;
        Vth = Vfb + pParam.phi + pParam.k1ox * sqrtPhis;
        Vgst = Vgs_eff - Vth;
        dVth_dVb = pParam.k1ox * dsqrtPhis_dVb * dVbseff_dVb;
        dVgst_dVb = -dVth_dVb;
        dVgst_dVg = dVgs_eff_dVg;
        CoxWL = model.coxe * pParam.weffCV * pParam.leffCV * here.nf;
        Arg1 = Vgs_eff - VbseffCV - Vfb;
        if Arg1 <= 0.0 {
            qgate = CoxWL * Arg1;
            qbulk = -qgate;
            qdrn = 0.0;
            here.cggb = CoxWL * dVgs_eff_dVg;
            here.cgdb = 0.0;
            here.cgsb = CoxWL * (dVbseffCV_dVb - dVgs_eff_dVg);
            here.cdgb = 0.0;
            here.cddb = 0.0;
            here.cdsb = 0.0;
            here.cbgb = -CoxWL * dVgs_eff_dVg;
            here.cbdb = 0.0;
            here.cbsb = -here.cgsb;
        } else if Vgst <= 0.0 {
            T1 = 0.5 * pParam.k1ox;
            T2 = sqrt(T1 * T1 + Arg1);
            qgate = CoxWL * pParam.k1ox * (T2 - T1);
            qbulk = -qgate;
            qdrn = 0.0;
            T0 = CoxWL * T1 / T2;
            here.cggb = T0 * dVgs_eff_dVg;
            here.cgdb = 0.0;
            here.cgsb = T0 * (dVbseffCV_dVb - dVgs_eff_dVg);
            here.cdgb = 0.0;
            here.cddb = 0.0;
            here.cdsb = 0.0;
            here.cbgb = -here.cggb;
            here.cbdb = 0.0;
            here.cbsb = -here.cgsb;
        } else {
            One_Third_CoxWL = CoxWL / 3.0;
            Two_Third_CoxWL = 2.0 * One_Third_CoxWL;
            AbulkCV = Abulk0_Q * pParam.abulkCVfactor;
            dAbulkCV_dVb = pParam.abulkCVfactor * dAbulk0_Q_dVb * dVbseff_dVb;
            dVdsat_dVg = 1.0 / AbulkCV;
            Vdsat = Vgst * dVdsat_dVg;
            dVdsat_dVb = -(Vdsat * dAbulkCV_dVb + dVth_dVb) * dVdsat_dVg;
            if model.xpart > 0.5 {
                if Vdsat <= Vds {
                    T1 = Vdsat / 3.0;
                    qgate = CoxWL * (Vgs_eff - Vfb - pParam.phi - T1);
                    T2 = -Two_Third_CoxWL * Vgst;
                    qbulk = -(qgate + T2);
                    qdrn = 0.0;
                    here.cggb = One_Third_CoxWL * (3.0 - dVdsat_dVg) * dVgs_eff_dVg;
                    T2 = -One_Third_CoxWL * dVdsat_dVb;
                    here.cgsb = -(here.cggb + T2);
                    here.cgdb = 0.0;
                    here.cdgb = 0.0;
                    here.cddb = 0.0;
                    here.cdsb = 0.0;
                    here.cbgb = -(here.cggb - Two_Third_CoxWL * dVgs_eff_dVg);
                    T3 = -(T2 + Two_Third_CoxWL * dVth_dVb);
                    here.cbsb = -(here.cbgb + T3);
                    here.cbdb = 0.0;
                } else {
                    Alphaz = Vgst / Vdsat;
                    T1 = 2.0 * Vdsat - Vds;
                    T2 = Vds / (3.0 * T1);
                    T3 = T2 * Vds;
                    T9 = 0.25 * CoxWL;
                    T4 = T9 * Alphaz;
                    T7 = 2.0 * Vds - T1 - 3.0 * T3;
                    T8 = T3 - T1 - 2.0 * Vds;
                    qgate = CoxWL * (Vgs_eff - Vfb - pParam.phi - 0.5 * (Vds - T3));
                    T10 = T4 * T8;
                    qdrn = T4 * T7;
                    qbulk = -(qgate + qdrn + T10);
                    T5 = T3 / T1;
                    here.cggb = CoxWL * (1.0 - T5 * dVdsat_dVg) * dVgs_eff_dVg;
                    T11 = -CoxWL * T5 * dVdsat_dVb;
                    here.cgdb = CoxWL * (T2 - 0.5 + 0.5 * T5);
                    here.cgsb = -(here.cggb + T11 + here.cgdb);
                    T6 = 1.0 / Vdsat;
                    dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg);
                    dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb);
                    T7 = T9 * T7;
                    T8 = T9 * T8;
                    T9 = 2.0 * T4 * (1.0 - 3.0 * T5);
                    here.cdgb = (T7 * dAlphaz_dVg - T9 * dVdsat_dVg) * dVgs_eff_dVg;
                    T12 = T7 * dAlphaz_dVb - T9 * dVdsat_dVb;
                    here.cddb = T4 * (3.0 - 6.0 * T2 - 3.0 * T5);
                    here.cdsb = -(here.cdgb + T12 + here.cddb);
                    T9 = 2.0 * T4 * (1.0 + T5);
                    T10 = (T8 * dAlphaz_dVg - T9 * dVdsat_dVg) * dVgs_eff_dVg;
                    T11 = T8 * dAlphaz_dVb - T9 * dVdsat_dVb;
                    T12 = T4 * (2.0 * T2 + T5 - 1.0);
                    T0 = -(T10 + T11 + T12);
                    here.cbgb = -(here.cggb + here.cdgb + T10);
                    here.cbdb = -(here.cgdb + here.cddb + T12);
                    here.cbsb = -(here.cgsb + here.cdsb + T0);
                }
            } else if model.xpart < 0.5 {
                if Vds >= Vdsat {
                    T1 = Vdsat / 3.0;
                    qgate = CoxWL * (Vgs_eff - Vfb - pParam.phi - T1);
                    T2 = -Two_Third_CoxWL * Vgst;
                    qbulk = -(qgate + T2);
                    qdrn = 0.4 * T2;
                    here.cggb = One_Third_CoxWL * (3.0 - dVdsat_dVg) * dVgs_eff_dVg;
                    T2 = -One_Third_CoxWL * dVdsat_dVb;
                    here.cgsb = -(here.cggb + T2);
                    here.cgdb = 0.0;
                    T3 = 0.4 * Two_Third_CoxWL;
                    here.cdgb = -T3 * dVgs_eff_dVg;
                    here.cddb = 0.0;
                    T4 = T3 * dVth_dVb;
                    here.cdsb = -(T4 + here.cdgb);
                    here.cbgb = -(here.cggb - Two_Third_CoxWL * dVgs_eff_dVg);
                    T3 = -(T2 + Two_Third_CoxWL * dVth_dVb);
                    here.cbsb = -(here.cbgb + T3);
                    here.cbdb = 0.0;
                } else {
                    Alphaz = Vgst / Vdsat;
                    T1 = 2.0 * Vdsat - Vds;
                    T2 = Vds / (3.0 * T1);
                    T3 = T2 * Vds;
                    T9 = 0.25 * CoxWL;
                    T4 = T9 * Alphaz;
                    qgate = CoxWL * (Vgs_eff - Vfb - pParam.phi - 0.5 * (Vds - T3));
                    T5 = T3 / T1;
                    here.cggb = CoxWL * (1.0 - T5 * dVdsat_dVg) * dVgs_eff_dVg;
                    tmp = -CoxWL * T5 * dVdsat_dVb;
                    here.cgdb = CoxWL * (T2 - 0.5 + 0.5 * T5);
                    here.cgsb = -(here.cggb + here.cgdb + tmp);
                    T6 = 1.0 / Vdsat;
                    dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg);
                    dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb);
                    T6 = 8.0 * Vdsat * Vdsat - 6.0 * Vdsat * Vds + 1.2 * Vds * Vds;
                    T8 = T2 / T1;
                    T7 = Vds - T1 - T8 * T6;
                    qdrn = T4 * T7;
                    T7 *= T9;
                    tmp = T8 / T1;
                    tmp1 = T4 * (2.0 - 4.0 * tmp * T6 + T8 * (16.0 * Vdsat - 6.0 * Vds));
                    here.cdgb = (T7 * dAlphaz_dVg - tmp1 * dVdsat_dVg) * dVgs_eff_dVg;
                    T10 = T7 * dAlphaz_dVb - tmp1 * dVdsat_dVb;
                    here.cddb = T4
                        * (2.0 - (1.0 / (3.0 * T1 * T1) + 2.0 * tmp) * T6
                            + T8 * (6.0 * Vdsat - 2.4 * Vds));
                    here.cdsb = -(here.cdgb + T10 + here.cddb);
                    T7 = 2.0 * (T1 + T3);
                    qbulk = -(qgate - T4 * T7);
                    T7 *= T9;
                    T0 = 4.0 * T4 * (1.0 - T5);
                    T12 = (-T7 * dAlphaz_dVg - T0 * dVdsat_dVg) * dVgs_eff_dVg - here.cdgb;
                    T11 = -T7 * dAlphaz_dVb - T10 - T0 * dVdsat_dVb;
                    T10 = -4.0 * T4 * (T2 - 0.5 + 0.5 * T5) - here.cddb;
                    tmp = -(T10 + T11 + T12);
                    here.cbgb = -(here.cggb + here.cdgb + T12);
                    here.cbdb = -(here.cgdb + here.cddb + T10);
                    here.cbsb = -(here.cgsb + here.cdsb + tmp);
                }
            } else {
                if Vds >= Vdsat {
                    T1 = Vdsat / 3.0;
                    qgate = CoxWL * (Vgs_eff - Vfb - pParam.phi - T1);
                    T2 = -Two_Third_CoxWL * Vgst;
                    qbulk = -(qgate + T2);
                    qdrn = 0.5 * T2;
                    here.cggb = One_Third_CoxWL * (3.0 - dVdsat_dVg) * dVgs_eff_dVg;
                    T2 = -One_Third_CoxWL * dVdsat_dVb;
                    here.cgsb = -(here.cggb + T2);
                    here.cgdb = 0.0;
                    here.cdgb = -One_Third_CoxWL * dVgs_eff_dVg;
                    here.cddb = 0.0;
                    T4 = One_Third_CoxWL * dVth_dVb;
                    here.cdsb = -(T4 + here.cdgb);
                    here.cbgb = -(here.cggb - Two_Third_CoxWL * dVgs_eff_dVg);
                    T3 = -(T2 + Two_Third_CoxWL * dVth_dVb);
                    here.cbsb = -(here.cbgb + T3);
                    here.cbdb = 0.0;
                } else {
                    Alphaz = Vgst / Vdsat;
                    T1 = 2.0 * Vdsat - Vds;
                    T2 = Vds / (3.0 * T1);
                    T3 = T2 * Vds;
                    T9 = 0.25 * CoxWL;
                    T4 = T9 * Alphaz;
                    qgate = CoxWL * (Vgs_eff - Vfb - pParam.phi - 0.5 * (Vds - T3));
                    T5 = T3 / T1;
                    here.cggb = CoxWL * (1.0 - T5 * dVdsat_dVg) * dVgs_eff_dVg;
                    tmp = -CoxWL * T5 * dVdsat_dVb;
                    here.cgdb = CoxWL * (T2 - 0.5 + 0.5 * T5);
                    here.cgsb = -(here.cggb + here.cgdb + tmp);
                    T6 = 1.0 / Vdsat;
                    dAlphaz_dVg = T6 * (1.0 - Alphaz * dVdsat_dVg);
                    dAlphaz_dVb = -T6 * (dVth_dVb + Alphaz * dVdsat_dVb);
                    T7 = T1 + T3;
                    qdrn = -T4 * T7;
                    qbulk = -(qgate + qdrn + qdrn);
                    T7 *= T9;
                    T0 = T4 * (2.0 * T5 - 2.0);
                    here.cdgb = (T0 * dVdsat_dVg - T7 * dAlphaz_dVg) * dVgs_eff_dVg;
                    T12 = T0 * dVdsat_dVb - T7 * dAlphaz_dVb;
                    here.cddb = T4 * (1.0 - 2.0 * T2 - T5);
                    here.cdsb = -(here.cdgb + T12 + here.cddb);
                    here.cbgb = -(here.cggb + 2.0 * here.cdgb);
                    here.cbdb = -(here.cgdb + 2.0 * here.cddb);
                    here.cbsb = -(here.cgsb + 2.0 * here.cdsb);
                }
            }
        }
    } else {
        if Vbseff < 0.0 {
            VbseffCV = Vbseff;
            dVbseffCV_dVb = 1.0;
        } else {
            VbseffCV = pParam.phi - Phis;
            dVbseffCV_dVb = -dPhis_dVb;
        }
        CoxWL = model.coxe * pParam.weffCV * pParam.leffCV * here.nf;
        if model.cvchargeMod == 0 {
            noff = n * pParam.noff;
            dnoff_dVd = pParam.noff * dn_dVd;
            dnoff_dVb = pParam.noff * dn_dVb;
            T0 = Vtm * noff;
            voffcv = pParam.voffcv;
            VgstNVt = (Vgst - voffcv) / T0;
            if VgstNVt > EXP_THRESHOLD {
                Vgsteff = Vgst - voffcv;
                dVgsteff_dVg = dVgs_eff_dVg;
                dVgsteff_dVd = -dVth_dVd;
                dVgsteff_dVb = -dVth_dVb;
            } else if VgstNVt < -EXP_THRESHOLD {
                Vgsteff = T0 * log(1.0 + MIN_EXP);
                dVgsteff_dVg = 0.0;
                dVgsteff_dVd = Vgsteff / noff;
                dVgsteff_dVb = dVgsteff_dVd * dnoff_dVb;
                dVgsteff_dVd *= dnoff_dVd;
            } else {
                ExpVgst = exp(VgstNVt);
                Vgsteff = T0 * log(1.0 + ExpVgst);
                dVgsteff_dVg = ExpVgst / (1.0 + ExpVgst);
                dVgsteff_dVd = -dVgsteff_dVg * (dVth_dVd + (Vgst - voffcv) / noff * dnoff_dVd)
                    + Vgsteff / noff * dnoff_dVd;
                dVgsteff_dVb = -dVgsteff_dVg * (dVth_dVb + (Vgst - voffcv) / noff * dnoff_dVb)
                    + Vgsteff / noff * dnoff_dVb;
                dVgsteff_dVg *= dVgs_eff_dVg;
            }
        } else {
            T0 = n * Vtm;
            T1 = pParam.mstarcv * Vgst;
            T2 = T1 / T0;
            if T2 > EXP_THRESHOLD {
                T10 = T1;
                dT10_dVg = pParam.mstarcv * dVgs_eff_dVg;
                dT10_dVd = -dVth_dVd * pParam.mstarcv;
                dT10_dVb = -dVth_dVb * pParam.mstarcv;
            } else if T2 < -EXP_THRESHOLD {
                T10 = Vtm * log(1.0 + MIN_EXP);
                dT10_dVg = 0.0;
                dT10_dVd = T10 * dn_dVd;
                dT10_dVb = T10 * dn_dVb;
                T10 *= n;
            } else {
                ExpVgst = exp(T2);
                T3 = Vtm * log(1.0 + ExpVgst);
                T10 = n * T3;
                dT10_dVg = pParam.mstarcv * ExpVgst / (1.0 + ExpVgst);
                dT10_dVb = T3 * dn_dVb - dT10_dVg * (dVth_dVb + Vgst * dn_dVb / n);
                dT10_dVd = T3 * dn_dVd - dT10_dVg * (dVth_dVd + Vgst * dn_dVd / n);
                dT10_dVg *= dVgs_eff_dVg;
            }
            T1 = pParam.voffcbncv - (1.0 - pParam.mstarcv) * Vgst;
            T2 = T1 / T0;
            if T2 < -EXP_THRESHOLD {
                T3 = model.coxe * MIN_EXP / pParam.cdep0;
                T9 = pParam.mstarcv + T3 * n;
                dT9_dVg = 0.0;
                dT9_dVd = dn_dVd * T3;
                dT9_dVb = dn_dVb * T3;
            } else if T2 > EXP_THRESHOLD {
                T3 = model.coxe * MAX_EXP / pParam.cdep0;
                T9 = pParam.mstarcv + T3 * n;
                dT9_dVg = 0.0;
                dT9_dVd = dn_dVd * T3;
                dT9_dVb = dn_dVb * T3;
            } else {
                ExpVgst = exp(T2);
                T3 = model.coxe / pParam.cdep0;
                T4 = T3 * ExpVgst;
                T5 = T1 * T4 / T0;
                T9 = pParam.mstarcv + n * T4;
                dT9_dVg = T3 * (pParam.mstarcv - 1.0) * ExpVgst / Vtm;
                dT9_dVb = T4 * dn_dVb - dT9_dVg * dVth_dVb - T5 * dn_dVb;
                dT9_dVd = T4 * dn_dVd - dT9_dVg * dVth_dVd - T5 * dn_dVd;
                dT9_dVg *= dVgs_eff_dVg;
            }
            Vgsteff = T10 / T9;
            T11 = T9 * T9;
            dVgsteff_dVg = (T9 * dT10_dVg - T10 * dT9_dVg) / T11;
            dVgsteff_dVd = (T9 * dT10_dVd - T10 * dT9_dVd) / T11;
            dVgsteff_dVb = (T9 * dT10_dVb - T10 * dT9_dVb) / T11;
        }
        if model.capMod == 1 {
            Vfb = here.vfbzb;
            V3 = Vfb - Vgs_eff + VbseffCV - DELTA_3;
            if Vfb <= 0.0 {
                T0 = sqrt(V3 * V3 - 4.0 * DELTA_3 * Vfb);
            } else {
                T0 = sqrt(V3 * V3 + 4.0 * DELTA_3 * Vfb);
            }
            T1 = 0.5 * (1.0 + V3 / T0);
            Vfbeff = Vfb - 0.5 * (V3 + T0);
            dVfbeff_dVg = T1 * dVgs_eff_dVg;
            dVfbeff_dVb = -T1 * dVbseffCV_dVb;
            Qac0 = CoxWL * (Vfbeff - Vfb);
            dQac0_dVg = CoxWL * dVfbeff_dVg;
            dQac0_dVb = CoxWL * dVfbeff_dVb;
            T0 = 0.5 * pParam.k1ox;
            T3 = Vgs_eff - Vfbeff - VbseffCV - Vgsteff;
            if pParam.k1ox == 0.0 {
                T1 = 0.0;
                T2 = 0.0;
            } else if T3 < 0.0 {
                T1 = T0 + T3 / pParam.k1ox;
                T2 = CoxWL;
            } else {
                T1 = sqrt(T0 * T0 + T3);
                T2 = CoxWL * T0 / T1;
            }
            Qsub0 = CoxWL * pParam.k1ox * (T1 - T0);
            dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg);
            dQsub0_dVd = -T2 * dVgsteff_dVd;
            dQsub0_dVb = -T2 * (dVfbeff_dVb + dVbseffCV_dVb + dVgsteff_dVb);
            AbulkCV = Abulk0_Q * pParam.abulkCVfactor;
            dAbulkCV_dVb = pParam.abulkCVfactor * dAbulk0_Q_dVb;
            VdsatCV = Vgsteff / AbulkCV;
            T0 = VdsatCV - Vds - DELTA_4;
            dT0_dVg = 1.0 / AbulkCV;
            dT0_dVb = -VdsatCV * dAbulkCV_dVb / AbulkCV;
            T1 = sqrt(T0 * T0 + 4.0 * DELTA_4 * VdsatCV);
            dT1_dVg = (T0 + DELTA_4 + DELTA_4) / T1;
            dT1_dVd = -T0 / T1;
            dT1_dVb = dT1_dVg * dT0_dVb;
            dT1_dVg *= dT0_dVg;
            if T0 >= 0.0 {
                VdseffCV = VdsatCV - 0.5 * (T0 + T1);
                dVdseffCV_dVg = 0.5 * (dT0_dVg - dT1_dVg);
                dVdseffCV_dVd = 0.5 * (1.0 - dT1_dVd);
                dVdseffCV_dVb = 0.5 * (dT0_dVb - dT1_dVb);
            } else {
                let twice_delta = DELTA_4 + DELTA_4;
                T3 = twice_delta / (T1 - T0);
                T4 = 1.0 - T3;
                T5 = VdsatCV * T3 / (T1 - T0);
                VdseffCV = VdsatCV * T4;
                dVdseffCV_dVg = dT0_dVg * T4 + T5 * (dT1_dVg - dT0_dVg);
                dVdseffCV_dVd = T5 * (dT1_dVd + 1.0);
                dVdseffCV_dVb = dT0_dVb * (T4 - T5) + T5 * dT1_dVb;
            }
            if Vds == 0.0 {
                VdseffCV = 0.0;
                dVdseffCV_dVg = 0.0;
                dVdseffCV_dVb = 0.0;
            }
            T0 = AbulkCV * VdseffCV;
            T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1.0e-20);
            T2 = VdseffCV / T1;
            T3 = T0 * T2;
            T4 = (1.0 - 12.0 * T2 * T2 * AbulkCV);
            T5 = (6.0 * T0 * (4.0 * Vgsteff - T0) / (T1 * T1) - 0.5);
            T6 = 12.0 * T2 * T2 * Vgsteff;
            qgate = CoxWL * (Vgsteff - 0.5 * VdseffCV + T3);
            Cgg1 = CoxWL * (T4 + T5 * dVdseffCV_dVg);
            Cgd1 = CoxWL * T5 * dVdseffCV_dVd + Cgg1 * dVgsteff_dVd;
            Cgb1 = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) + Cgg1 * dVgsteff_dVb;
            Cgg1 *= dVgsteff_dVg;
            T7 = 1.0 - AbulkCV;
            qbulk = CoxWL * T7 * (0.5 * VdseffCV - T3);
            T4 = -T7 * (T4 - 1.0);
            T5 = -T7 * T5;
            T6 = -(T7 * T6 + (0.5 * VdseffCV - T3));
            Cbg1 = CoxWL * (T4 + T5 * dVdseffCV_dVg);
            Cbd1 = CoxWL * T5 * dVdseffCV_dVd + Cbg1 * dVgsteff_dVd;
            Cbb1 = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) + Cbg1 * dVgsteff_dVb;
            Cbg1 *= dVgsteff_dVg;
            if model.xpart > 0.5 {
                T1 = T1 + T1;
                qsrc = -CoxWL * (0.5 * Vgsteff + 0.25 * T0 - T0 * T0 / T1);
                T7 = (4.0 * Vgsteff - T0) / (T1 * T1);
                T4 = -(0.5 + 24.0 * T0 * T0 / (T1 * T1));
                T5 = -(0.25 * AbulkCV - 12.0 * AbulkCV * T0 * T7);
                T6 = -(0.25 * VdseffCV - 12.0 * T0 * VdseffCV * T7);
                Csg = CoxWL * (T4 + T5 * dVdseffCV_dVg);
                Csd = CoxWL * T5 * dVdseffCV_dVd + Csg * dVgsteff_dVd;
                Csb = CoxWL * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) + Csg * dVgsteff_dVb;
                Csg *= dVgsteff_dVg;
            } else if model.xpart < 0.5 {
                T1 = T1 / 12.0;
                T2 = 0.5 * CoxWL / (T1 * T1);
                T3 = Vgsteff * (2.0 * T0 * T0 / 3.0 + Vgsteff * (Vgsteff - 4.0 * T0 / 3.0))
                    - 2.0 * T0 * T0 * T0 / 15.0;
                qsrc = -T2 * T3;
                T7 = 4.0 / 3.0 * Vgsteff * (Vgsteff - T0) + 0.4 * T0 * T0;
                T4 = -2.0 * qsrc / T1
                    - T2 * (Vgsteff * (3.0 * Vgsteff - 8.0 * T0 / 3.0) + 2.0 * T0 * T0 / 3.0);
                T5 = (qsrc / T1 + T2 * T7) * AbulkCV;
                T6 = (qsrc / T1 * VdseffCV + T2 * T7 * VdseffCV);
                Csg = (T4 + T5 * dVdseffCV_dVg);
                Csd = T5 * dVdseffCV_dVd + Csg * dVgsteff_dVd;
                Csb = (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb) + Csg * dVgsteff_dVb;
                Csg *= dVgsteff_dVg;
            } else {
                qsrc = -0.5 * (qgate + qbulk);
                Csg = -0.5 * (Cgg1 + Cbg1);
                Csb = -0.5 * (Cgb1 + Cbb1);
                Csd = -0.5 * (Cgd1 + Cbd1);
            }
            qgate += Qac0 + Qsub0;
            qbulk -= (Qac0 + Qsub0);
            qdrn = -(qgate + qbulk + qsrc);
            Cgg = dQac0_dVg + dQsub0_dVg + Cgg1;
            Cgd = dQsub0_dVd + Cgd1;
            Cgb = dQac0_dVb + dQsub0_dVb + Cgb1;
            Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
            Cbd = Cbd1 - dQsub0_dVd;
            Cbb = Cbb1 - dQac0_dVb - dQsub0_dVb;
            Cgb *= dVbseff_dVb;
            Cbb *= dVbseff_dVb;
            Csb *= dVbseff_dVb;
            here.cggb = Cgg;
            here.cgsb = -(Cgg + Cgd + Cgb);
            here.cgdb = Cgd;
            here.cdgb = -(Cgg + Cbg + Csg);
            here.cdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb + Csg + Csd + Csb);
            here.cddb = -(Cgd + Cbd + Csd);
            here.cbgb = Cbg;
            here.cbsb = -(Cbg + Cbd + Cbb);
            here.cbdb = Cbd;
        } else if model.capMod == 2 {
            V3 = here.vfbzb - Vgs_eff + VbseffCV - DELTA_3;
            if here.vfbzb <= 0.0 {
                T0 = sqrt(V3 * V3 - 4.0 * DELTA_3 * here.vfbzb);
            } else {
                T0 = sqrt(V3 * V3 + 4.0 * DELTA_3 * here.vfbzb);
            }
            T1 = 0.5 * (1.0 + V3 / T0);
            Vfbeff = here.vfbzb - 0.5 * (V3 + T0);
            dVfbeff_dVg = T1 * dVgs_eff_dVg;
            dVfbeff_dVb = -T1 * dVbseffCV_dVb;
            Cox = here.coxp;
            Tox = 1.0e8 * here.toxp;
            T0 = (Vgs_eff - VbseffCV - here.vfbzb) / Tox;
            dT0_dVg = dVgs_eff_dVg / Tox;
            dT0_dVb = -dVbseffCV_dVb / Tox;
            tmp = T0 * pParam.acde;
            if (-EXP_THRESHOLD < tmp) && (tmp < EXP_THRESHOLD) {
                Tcen = pParam.ldeb * exp(tmp);
                dTcen_dVg = pParam.acde * Tcen;
                dTcen_dVb = dTcen_dVg * dT0_dVb;
                dTcen_dVg *= dT0_dVg;
            } else if tmp <= -EXP_THRESHOLD {
                Tcen = pParam.ldeb * MIN_EXP;
                dTcen_dVb = 0.0;
                dTcen_dVg = dTcen_dVb;
            } else {
                Tcen = pParam.ldeb * MAX_EXP;
                dTcen_dVb = 0.0;
                dTcen_dVg = dTcen_dVb;
            }
            LINK = 1.0e-3 * here.toxp;
            V3 = pParam.ldeb - Tcen - LINK;
            V4 = sqrt(V3 * V3 + 4.0 * LINK * pParam.ldeb);
            Tcen = pParam.ldeb - 0.5 * (V3 + V4);
            T1 = 0.5 * (1.0 + V3 / V4);
            dTcen_dVg *= T1;
            dTcen_dVb *= T1;
            Ccen = epssub / Tcen;
            T2 = Cox / (Cox + Ccen);
            Coxeff = T2 * Ccen;
            T3 = -Ccen / Tcen;
            dCoxeff_dVg = T2 * T2 * T3;
            dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb;
            dCoxeff_dVg *= dTcen_dVg;
            CoxWLcen = CoxWL * Coxeff / model.coxe;
            Qac0 = CoxWLcen * (Vfbeff - here.vfbzb);
            QovCox = Qac0 / Coxeff;
            dQac0_dVg = CoxWLcen * dVfbeff_dVg + QovCox * dCoxeff_dVg;
            dQac0_dVb = CoxWLcen * dVfbeff_dVb + QovCox * dCoxeff_dVb;
            T0 = 0.5 * pParam.k1ox;
            T3 = Vgs_eff - Vfbeff - VbseffCV - Vgsteff;
            if pParam.k1ox == 0.0 {
                T1 = 0.0;
                T2 = 0.0;
            } else if T3 < 0.0 {
                T1 = T0 + T3 / pParam.k1ox;
                T2 = CoxWLcen;
            } else {
                T1 = sqrt(T0 * T0 + T3);
                T2 = CoxWLcen * T0 / T1;
            }
            Qsub0 = CoxWLcen * pParam.k1ox * (T1 - T0);
            QovCox = Qsub0 / Coxeff;
            dQsub0_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg) + QovCox * dCoxeff_dVg;
            dQsub0_dVd = -T2 * dVgsteff_dVd;
            dQsub0_dVb = -T2 * (dVfbeff_dVb + dVbseffCV_dVb + dVgsteff_dVb) + QovCox * dCoxeff_dVb;
            if pParam.k1ox <= 0.0 {
                Denomi = 0.25 * pParam.moin * Vtm;
                T0 = 0.5 * pParam.sqrtPhi;
            } else {
                Denomi = pParam.moin * Vtm * pParam.k1ox * pParam.k1ox;
                T0 = pParam.k1ox * pParam.sqrtPhi;
            }
            T1 = 2.0 * T0 + Vgsteff;
            DeltaPhi = Vtm * log(1.0 + T1 * Vgsteff / Denomi);
            dDeltaPhi_dVg = 2.0 * Vtm * (T1 - T0) / (Denomi + T1 * Vgsteff);
            T0 = Vgsteff - DeltaPhi - 0.001;
            dT0_dVg = 1.0 - dDeltaPhi_dVg;
            T1 = sqrt(T0 * T0 + Vgsteff * 0.004);
            VgDP = 0.5 * (T0 + T1);
            dVgDP_dVg = 0.5 * (dT0_dVg + (T0 * dT0_dVg + 0.002) / T1);
            Tox += Tox;
            T0 = (Vgsteff + here.vtfbphi2) / Tox;
            tmp = exp(model.bdos * 0.7 * log(T0));
            T1 = 1.0 + tmp;
            T2 = model.bdos * 0.7 * tmp / (T0 * Tox);
            Tcen = model.ados * 1.9e-9 / T1;
            dTcen_dVg = -Tcen * T2 / T1;
            dTcen_dVd = dTcen_dVg * dVgsteff_dVd;
            dTcen_dVb = dTcen_dVg * dVgsteff_dVb;
            dTcen_dVg *= dVgsteff_dVg;
            Ccen = epssub / Tcen;
            T0 = Cox / (Cox + Ccen);
            Coxeff = T0 * Ccen;
            T1 = -Ccen / Tcen;
            dCoxeff_dVg = T0 * T0 * T1;
            dCoxeff_dVd = dCoxeff_dVg * dTcen_dVd;
            dCoxeff_dVb = dCoxeff_dVg * dTcen_dVb;
            dCoxeff_dVg *= dTcen_dVg;
            CoxWLcen = CoxWL * Coxeff / model.coxe;
            AbulkCV = Abulk0_Q * pParam.abulkCVfactor;
            dAbulkCV_dVb = pParam.abulkCVfactor * dAbulk0_Q_dVb;
            VdsatCV = VgDP / AbulkCV;
            T0 = VdsatCV - Vds - DELTA_4;
            dT0_dVg = dVgDP_dVg / AbulkCV;
            dT0_dVb = -VdsatCV * dAbulkCV_dVb / AbulkCV;
            T1 = sqrt(T0 * T0 + 4.0 * DELTA_4 * VdsatCV);
            dT1_dVg = (T0 + DELTA_4 + DELTA_4) / T1;
            dT1_dVd = -T0 / T1;
            dT1_dVb = dT1_dVg * dT0_dVb;
            dT1_dVg *= dT0_dVg;
            if T0 >= 0.0 {
                VdseffCV = VdsatCV - 0.5 * (T0 + T1);
                dVdseffCV_dVg = 0.5 * (dT0_dVg - dT1_dVg);
                dVdseffCV_dVd = 0.5 * (1.0 - dT1_dVd);
                dVdseffCV_dVb = 0.5 * (dT0_dVb - dT1_dVb);
            } else {
                let twice_delta = DELTA_4 + DELTA_4;
                T3 = twice_delta / (T1 - T0);
                T4 = 1.0 - T3;
                T5 = VdsatCV * T3 / (T1 - T0);
                VdseffCV = VdsatCV * T4;
                dVdseffCV_dVg = dT0_dVg * T4 + T5 * (dT1_dVg - dT0_dVg);
                dVdseffCV_dVd = T5 * (dT1_dVd + 1.0);
                dVdseffCV_dVb = dT0_dVb * (T4 - T5) + T5 * dT1_dVb;
            }
            if Vds == 0.0 {
                VdseffCV = 0.0;
                dVdseffCV_dVg = 0.0;
                dVdseffCV_dVb = 0.0;
            }
            T0 = AbulkCV * VdseffCV;
            T1 = VgDP;
            T2 = 12.0 * (T1 - 0.5 * T0 + 1.0e-20);
            T3 = T0 / T2;
            T4 = 1.0 - 12.0 * T3 * T3;
            T5 = AbulkCV * (6.0 * T0 * (4.0 * T1 - T0) / (T2 * T2) - 0.5);
            T6 = T5 * VdseffCV / AbulkCV;
            qgate = CoxWLcen * (T1 - T0 * (0.5 - T3));
            QovCox = qgate / Coxeff;
            Cgg1 = CoxWLcen * (T4 * dVgDP_dVg + T5 * dVdseffCV_dVg);
            Cgd1 = CoxWLcen * T5 * dVdseffCV_dVd + Cgg1 * dVgsteff_dVd + QovCox * dCoxeff_dVd;
            Cgb1 = CoxWLcen * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb)
                + Cgg1 * dVgsteff_dVb
                + QovCox * dCoxeff_dVb;
            Cgg1 = Cgg1 * dVgsteff_dVg + QovCox * dCoxeff_dVg;
            T7 = 1.0 - AbulkCV;
            T8 = T2 * T2;
            T9 = 12.0 * T7 * T0 * T0 / (T8 * AbulkCV);
            T10 = T9 * dVgDP_dVg;
            T11 = -T7 * T5 / AbulkCV;
            T12 = -(T9 * T1 / AbulkCV + VdseffCV * (0.5 - T0 / T2));
            qbulk = CoxWLcen * T7 * (0.5 * VdseffCV - T0 * VdseffCV / T2);
            QovCox = qbulk / Coxeff;
            Cbg1 = CoxWLcen * (T10 + T11 * dVdseffCV_dVg);
            Cbd1 = CoxWLcen * T11 * dVdseffCV_dVd + Cbg1 * dVgsteff_dVd + QovCox * dCoxeff_dVd;
            Cbb1 = CoxWLcen * (T11 * dVdseffCV_dVb + T12 * dAbulkCV_dVb)
                + Cbg1 * dVgsteff_dVb
                + QovCox * dCoxeff_dVb;
            Cbg1 = Cbg1 * dVgsteff_dVg + QovCox * dCoxeff_dVg;
            if model.xpart > 0.5 {
                qsrc = -CoxWLcen * (T1 / 2.0 + T0 / 4.0 - 0.5 * T0 * T0 / T2);
                QovCox = qsrc / Coxeff;
                T2 += T2;
                T3 = T2 * T2;
                T7 = -(0.25 - 12.0 * T0 * (4.0 * T1 - T0) / T3);
                T4 = -(0.5 + 24.0 * T0 * T0 / T3) * dVgDP_dVg;
                T5 = T7 * AbulkCV;
                T6 = T7 * VdseffCV;
                Csg = CoxWLcen * (T4 + T5 * dVdseffCV_dVg);
                Csd = CoxWLcen * T5 * dVdseffCV_dVd + Csg * dVgsteff_dVd + QovCox * dCoxeff_dVd;
                Csb = CoxWLcen * (T5 * dVdseffCV_dVb + T6 * dAbulkCV_dVb)
                    + Csg * dVgsteff_dVb
                    + QovCox * dCoxeff_dVb;
                Csg = Csg * dVgsteff_dVg + QovCox * dCoxeff_dVg;
            } else if model.xpart < 0.5 {
                T2 = T2 / 12.0;
                T3 = 0.5 * CoxWLcen / (T2 * T2);
                T4 = T1 * (2.0 * T0 * T0 / 3.0 + T1 * (T1 - 4.0 * T0 / 3.0))
                    - 2.0 * T0 * T0 * T0 / 15.0;
                qsrc = -T3 * T4;
                QovCox = qsrc / Coxeff;
                T8 = 4.0 / 3.0 * T1 * (T1 - T0) + 0.4 * T0 * T0;
                T5 = -2.0 * qsrc / T2
                    - T3 * (T1 * (3.0 * T1 - 8.0 * T0 / 3.0) + 2.0 * T0 * T0 / 3.0);
                T6 = AbulkCV * (qsrc / T2 + T3 * T8);
                T7 = T6 * VdseffCV / AbulkCV;
                Csg = T5 * dVgDP_dVg + T6 * dVdseffCV_dVg;
                Csd = Csg * dVgsteff_dVd + T6 * dVdseffCV_dVd + QovCox * dCoxeff_dVd;
                Csb = Csg * dVgsteff_dVb
                    + T6 * dVdseffCV_dVb
                    + T7 * dAbulkCV_dVb
                    + QovCox * dCoxeff_dVb;
                Csg = Csg * dVgsteff_dVg + QovCox * dCoxeff_dVg;
            } else {
                qsrc = -0.5 * qgate;
                Csg = -0.5 * Cgg1;
                Csd = -0.5 * Cgd1;
                Csb = -0.5 * Cgb1;
            }
            qgate += Qac0 + Qsub0 - qbulk;
            qbulk -= (Qac0 + Qsub0);
            qdrn = -(qgate + qbulk + qsrc);
            Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
            Cbd = Cbd1 - dQsub0_dVd;
            Cbb = Cbb1 - dQac0_dVb - dQsub0_dVb;
            Cgg = Cgg1 - Cbg;
            Cgd = Cgd1 - Cbd;
            Cgb = Cgb1 - Cbb;
            Cgb *= dVbseff_dVb;
            Cbb *= dVbseff_dVb;
            Csb *= dVbseff_dVb;
            here.cggb = Cgg;
            here.cgsb = -(Cgg + Cgd + Cgb);
            here.cgdb = Cgd;
            here.cdgb = -(Cgg + Cbg + Csg);
            here.cdsb = (Cgg + Cgd + Cgb + Cbg + Cbd + Cbb + Csg + Csd + Csb);
            here.cddb = -(Cgd + Cbd + Csd);
            here.cbgb = Cbg;
            here.cbsb = -(Cbg + Cbd + Cbb);
            here.cbdb = Cbd;
        }
    }
    if !goto_finished {
        here.csgb = -here.cggb - here.cdgb - here.cbgb;
        here.csdb = -here.cgdb - here.cddb - here.cbdb;
        here.cssb = -here.cgsb - here.cdsb - here.cbsb;
        here.cgbb = -here.cgdb - here.cggb - here.cgsb;
        here.cdbb = -here.cddb - here.cdgb - here.cdsb;
        here.cbbb = -here.cbgb - here.cbdb - here.cbsb;
        here.csbb = -here.cgbb - here.cdbb - here.cbbb;
        here.qgate = qgate;
        here.qbulk = qbulk;
        here.qdrn = qdrn;
        here.qsrc = -(qgate + qbulk + qdrn);
        if ((here.trnqsMod) != 0) || ((here.acnqsMod) != 0) {
            qcheq = -(qbulk + qgate);
            here.qchqs = qcheq;
            here.cqgb = -(here.cggb + here.cbgb);
            here.cqdb = -(here.cgdb + here.cbdb);
            here.cqsb = -(here.cgsb + here.cbsb);
            here.cqbb = -(here.cqgb + here.cqdb + here.cqsb);
            CoxWL = model.coxe * pParam.weffCV * here.nf * pParam.leffCV;
            T1 = here.gcrg / CoxWL;
            here.gtau = T1 * ScalingFactor;
            if here.acnqsMod != 0 {
                here.taunet = 1.0 / T1;
            }
            ckt.set_state0(state::QCHEQ, qcheq);
            if (ckt.mode() & mode::MODEINITTRAN) != 0 {
                ckt.set_state1(state::QCHEQ, ckt.state0(state::QCHEQ));
            }
            if here.trnqsMod != 0 {
                (geq, ceq) = ckt.integrate(0.0, state::QCHEQ)?;
            }
        }
    }
    // C label `finished:`.
    if ChargeComputationNeeded != 0 {
        junction::charge(model, here, pParam, ckt, junction_bias);
    }
    if (here.off == 0) || ((ckt.mode() & mode::MODEINITFIX) == 0) {
        if Check == 1 {
            ckt.noncon();
        }
    }
    ckt.set_state0(state::VDS, vds);
    ckt.set_state0(state::VGS, vgs);
    ckt.set_state0(state::VBS, vbs);
    ckt.set_state0(state::VBD, vbd);
    ckt.set_state0(state::VGES, vges);
    ckt.set_state0(state::VGMS, vgms);
    ckt.set_state0(state::VDBS, vdbs);
    ckt.set_state0(state::VDBD, vdbd);
    ckt.set_state0(state::VSBS, vsbs);
    ckt.set_state0(state::VSES, vses);
    ckt.set_state0(state::VDES, vdes);
    ckt.set_state0(state::QDEF, qdef);
    // C's `goto line850` and `goto line1000` out of the charge block are
    // breaks out of 'charges with these flags; leaving the block normally
    // is C's `goto line860`. (C's bypass `goto line755`/`line860` cannot
    // happen: the port never bypasses, CKTbypass = 0.)
    let mut goto_line850 = false;
    let mut goto_line1000 = false;
    'charges: {
        if ChargeComputationNeeded == 0 {
            goto_line850 = true;
            break 'charges;
        }
        if here.rgateMod == 3 {
            vgdx = vgmd;
            vgsx = vgms;
        } else {
            vgdx = vgd;
            vgsx = vgs;
        }
        if model.capMod == 0 {
            cgdo = pParam.cgdo;
            qgdo = pParam.cgdo * vgdx;
            cgso = pParam.cgso;
            qgso = pParam.cgso * vgsx;
        } else {
            T0 = vgdx + DELTA_1;
            T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
            T2 = 0.5 * (T0 - T1);
            T3 = pParam.weffCV * pParam.cgdl;
            T4 = sqrt(1.0 - 4.0 * T2 / pParam.ckappad);
            cgdo = pParam.cgdo + T3 - T3 * (1.0 - 1.0 / T4) * (0.5 - 0.5 * T0 / T1);
            qgdo = (pParam.cgdo + T3) * vgdx - T3 * (T2 + 0.5 * pParam.ckappad * (T4 - 1.0));
            T0 = vgsx + DELTA_1;
            T1 = sqrt(T0 * T0 + 4.0 * DELTA_1);
            T2 = 0.5 * (T0 - T1);
            T3 = pParam.weffCV * pParam.cgsl;
            T4 = sqrt(1.0 - 4.0 * T2 / pParam.ckappas);
            cgso = pParam.cgso + T3 - T3 * (1.0 - 1.0 / T4) * (0.5 - 0.5 * T0 / T1);
            qgso = (pParam.cgso + T3) * vgsx - T3 * (T2 + 0.5 * pParam.ckappas * (T4 - 1.0));
        }
        if here.nf != 1.0 {
            cgdo *= here.nf;
            cgso *= here.nf;
            qgdo *= here.nf;
            qgso *= here.nf;
        }
        here.cgdo = cgdo;
        here.qgdo = qgdo;
        here.cgso = cgso;
        here.qgso = qgso;
        ag0 = ckt.ag(0);
        if here.mode > 0 {
            if here.trnqsMod == 0 {
                qdrn -= qgdo;
                if here.rgateMod == 3 {
                    gcgmgmb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgmdb = -cgdo * ag0;
                    gcgmsb = -cgso * ag0;
                    gcgmbb = -pParam.cgbo * ag0;
                    gcdgmb = gcgmdb;
                    gcsgmb = gcgmsb;
                    gcbgmb = gcgmbb;
                    gcggb = here.cggb * ag0;
                    gcgdb = here.cgdb * ag0;
                    gcgsb = here.cgsb * ag0;
                    gcgbb = -(gcggb + gcgdb + gcgsb);
                    gcdgb = here.cdgb * ag0;
                    gcsgb = -(here.cggb + here.cbgb + here.cdgb) * ag0;
                    gcbgb = here.cbgb * ag0;
                    qgmb = pParam.cgbo * vgmb;
                    qgmid = qgdo + qgso + qgmb;
                    qbulk -= qgmb;
                    qsrc = -(qgate + qgmid + qbulk + qdrn);
                } else {
                    gcggb = (here.cggb + cgdo + cgso + pParam.cgbo) * ag0;
                    gcgdb = (here.cgdb - cgdo) * ag0;
                    gcgsb = (here.cgsb - cgso) * ag0;
                    gcgbb = -(gcggb + gcgdb + gcgsb);
                    gcdgb = (here.cdgb - cgdo) * ag0;
                    gcsgb = -(here.cggb + here.cbgb + here.cdgb + cgso) * ag0;
                    gcbgb = (here.cbgb - pParam.cgbo) * ag0;
                    gcbgmb = 0.0;
                    gcsgmb = gcbgmb;
                    gcdgmb = gcsgmb;
                    qgb = pParam.cgbo * vgb;
                    qgate += qgdo + qgso + qgb;
                    qbulk -= qgb;
                    qsrc = -(qgate + qbulk + qdrn);
                }
                gcddb = (here.cddb + here.capbd + cgdo) * ag0;
                gcdsb = here.cdsb * ag0;
                gcsdb = -(here.cgdb + here.cbdb + here.cddb) * ag0;
                gcssb = (here.capbs + cgso - (here.cgsb + here.cbsb + here.cdsb)) * ag0;
                if here.rbodyMod == 0 {
                    gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb);
                    gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb);
                    gcbdb = (here.cbdb - here.capbd) * ag0;
                    gcbsb = (here.cbsb - here.capbs) * ag0;
                    gcdbdb = 0.0;
                    gcsbsb = 0.0;
                } else {
                    gcdbb = -(here.cddb + here.cdgb + here.cdsb) * ag0;
                    gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb) + here.capbs * ag0;
                    gcbdb = here.cbdb * ag0;
                    gcbsb = here.cbsb * ag0;
                    gcdbdb = -here.capbd * ag0;
                    gcsbsb = -here.capbs * ag0;
                }
                gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb);
                ggts = 0.0;
                ggtb = ggts;
                ggtd = ggtb;
                ggtg = ggtd;
                sxpart = 0.6;
                dxpart = 0.4;
                ddxpart_dVs = 0.0;
                ddxpart_dVb = ddxpart_dVs;
                ddxpart_dVg = ddxpart_dVb;
                ddxpart_dVd = ddxpart_dVg;
                dsxpart_dVs = 0.0;
                dsxpart_dVb = dsxpart_dVs;
                dsxpart_dVg = dsxpart_dVb;
                dsxpart_dVd = dsxpart_dVg;
            } else {
                qcheq = here.qchqs;
                CoxWL = model.coxe * pParam.weffCV * here.nf * pParam.leffCV;
                T0 = qdef * ScalingFactor / CoxWL;
                here.gtg = T0 * here.gcrgg;
                ggtg = here.gtg;
                here.gtd = T0 * here.gcrgd;
                ggtd = here.gtd;
                here.gts = T0 * here.gcrgs;
                ggts = here.gts;
                here.gtb = T0 * here.gcrgb;
                ggtb = here.gtb;
                gqdef = ScalingFactor * ag0;
                gcqgb = here.cqgb * ag0;
                gcqdb = here.cqdb * ag0;
                gcqsb = here.cqsb * ag0;
                gcqbb = here.cqbb * ag0;
                if fabs(qcheq) <= 1.0e-5 * CoxWL {
                    if model.xpart < 0.5 {
                        dxpart = 0.4;
                    } else if model.xpart > 0.5 {
                        dxpart = 0.0;
                    } else {
                        dxpart = 0.5;
                    }
                    ddxpart_dVs = 0.0;
                    ddxpart_dVb = ddxpart_dVs;
                    ddxpart_dVg = ddxpart_dVb;
                    ddxpart_dVd = ddxpart_dVg;
                } else {
                    dxpart = qdrn / qcheq;
                    Cdd = here.cddb;
                    Csd = -(here.cgdb + here.cddb + here.cbdb);
                    ddxpart_dVd = (Cdd - dxpart * (Cdd + Csd)) / qcheq;
                    Cdg = here.cdgb;
                    Csg = -(here.cggb + here.cdgb + here.cbgb);
                    ddxpart_dVg = (Cdg - dxpart * (Cdg + Csg)) / qcheq;
                    Cds = here.cdsb;
                    Css = -(here.cgsb + here.cdsb + here.cbsb);
                    ddxpart_dVs = (Cds - dxpart * (Cds + Css)) / qcheq;
                    ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs);
                }
                sxpart = 1.0 - dxpart;
                dsxpart_dVd = -ddxpart_dVd;
                dsxpart_dVg = -ddxpart_dVg;
                dsxpart_dVs = -ddxpart_dVs;
                dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs);
                if here.rgateMod == 3 {
                    gcgmgmb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgmdb = -cgdo * ag0;
                    gcgmsb = -cgso * ag0;
                    gcgmbb = -pParam.cgbo * ag0;
                    gcdgmb = gcgmdb;
                    gcsgmb = gcgmsb;
                    gcbgmb = gcgmbb;
                    gcbgb = 0.0;
                    gcsgb = gcbgb;
                    gcdgb = gcsgb;
                    gcgbb = 0.0;
                    gcgsb = gcgbb;
                    gcgdb = gcgsb;
                    gcggb = gcgdb;
                    qgmb = pParam.cgbo * vgmb;
                    qgmid = qgdo + qgso + qgmb;
                    qgate = 0.0;
                    qbulk = -qgmb;
                    qdrn = -qgdo;
                    qsrc = -(qgmid + qbulk + qdrn);
                } else {
                    gcggb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgdb = -cgdo * ag0;
                    gcgsb = -cgso * ag0;
                    gcgbb = -pParam.cgbo * ag0;
                    gcdgb = gcgdb;
                    gcsgb = gcgsb;
                    gcbgb = gcgbb;
                    gcbgmb = 0.0;
                    gcsgmb = gcbgmb;
                    gcdgmb = gcsgmb;
                    qgb = pParam.cgbo * vgb;
                    qgate = qgdo + qgso + qgb;
                    qbulk = -qgb;
                    qdrn = -qgdo;
                    qsrc = -(qgate + qbulk + qdrn);
                }
                gcddb = (here.capbd + cgdo) * ag0;
                gcsdb = 0.0;
                gcdsb = gcsdb;
                gcssb = (here.capbs + cgso) * ag0;
                if here.rbodyMod == 0 {
                    gcdbb = -(gcdgb + gcddb + gcdgmb);
                    gcsbb = -(gcsgb + gcssb + gcsgmb);
                    gcbdb = -here.capbd * ag0;
                    gcbsb = -here.capbs * ag0;
                    gcdbdb = 0.0;
                    gcsbsb = 0.0;
                } else {
                    gcbsb = 0.0;
                    gcbdb = gcbsb;
                    gcsbb = gcbdb;
                    gcdbb = gcsbb;
                    gcdbdb = -here.capbd * ag0;
                    gcsbsb = -here.capbs * ag0;
                }
                gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb);
            }
        } else {
            if here.trnqsMod == 0 {
                qsrc = qdrn - qgso;
                if here.rgateMod == 3 {
                    gcgmgmb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgmdb = -cgdo * ag0;
                    gcgmsb = -cgso * ag0;
                    gcgmbb = -pParam.cgbo * ag0;
                    gcdgmb = gcgmdb;
                    gcsgmb = gcgmsb;
                    gcbgmb = gcgmbb;
                    gcggb = here.cggb * ag0;
                    gcgdb = here.cgsb * ag0;
                    gcgsb = here.cgdb * ag0;
                    gcgbb = -(gcggb + gcgdb + gcgsb);
                    gcdgb = -(here.cggb + here.cbgb + here.cdgb) * ag0;
                    gcsgb = here.cdgb * ag0;
                    gcbgb = here.cbgb * ag0;
                    qgmb = pParam.cgbo * vgmb;
                    qgmid = qgdo + qgso + qgmb;
                    qbulk -= qgmb;
                    qdrn = -(qgate + qgmid + qbulk + qsrc);
                } else {
                    gcggb = (here.cggb + cgdo + cgso + pParam.cgbo) * ag0;
                    gcgdb = (here.cgsb - cgdo) * ag0;
                    gcgsb = (here.cgdb - cgso) * ag0;
                    gcgbb = -(gcggb + gcgdb + gcgsb);
                    gcdgb = -(here.cggb + here.cbgb + here.cdgb + cgdo) * ag0;
                    gcsgb = (here.cdgb - cgso) * ag0;
                    gcbgb = (here.cbgb - pParam.cgbo) * ag0;
                    gcbgmb = 0.0;
                    gcsgmb = gcbgmb;
                    gcdgmb = gcsgmb;
                    qgb = pParam.cgbo * vgb;
                    qgate += qgdo + qgso + qgb;
                    qbulk -= qgb;
                    qdrn = -(qgate + qbulk + qsrc);
                }
                gcddb = (here.capbd + cgdo - (here.cgsb + here.cbsb + here.cdsb)) * ag0;
                gcdsb = -(here.cgdb + here.cbdb + here.cddb) * ag0;
                gcsdb = here.cdsb * ag0;
                gcssb = (here.cddb + here.capbs + cgso) * ag0;
                if here.rbodyMod == 0 {
                    gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb);
                    gcsbb = -(gcsgb + gcsdb + gcssb + gcsgmb);
                    gcbdb = (here.cbsb - here.capbd) * ag0;
                    gcbsb = (here.cbdb - here.capbs) * ag0;
                    gcdbdb = 0.0;
                    gcsbsb = 0.0;
                } else {
                    gcdbb = -(gcdgb + gcddb + gcdsb + gcdgmb) + here.capbd * ag0;
                    gcsbb = -(here.cddb + here.cdgb + here.cdsb) * ag0;
                    gcbdb = here.cbsb * ag0;
                    gcbsb = here.cbdb * ag0;
                    gcdbdb = -here.capbd * ag0;
                    gcsbsb = -here.capbs * ag0;
                }
                gcbbb = -(gcbgb + gcbdb + gcbsb + gcbgmb);
                ggts = 0.0;
                ggtb = ggts;
                ggtd = ggtb;
                ggtg = ggtd;
                sxpart = 0.4;
                dxpart = 0.6;
                ddxpart_dVs = 0.0;
                ddxpart_dVb = ddxpart_dVs;
                ddxpart_dVg = ddxpart_dVb;
                ddxpart_dVd = ddxpart_dVg;
                dsxpart_dVs = 0.0;
                dsxpart_dVb = dsxpart_dVs;
                dsxpart_dVg = dsxpart_dVb;
                dsxpart_dVd = dsxpart_dVg;
            } else {
                qcheq = here.qchqs;
                CoxWL = model.coxe * pParam.weffCV * here.nf * pParam.leffCV;
                T0 = qdef * ScalingFactor / CoxWL;
                here.gtg = T0 * here.gcrgg;
                ggtg = here.gtg;
                here.gts = T0 * here.gcrgd;
                ggts = here.gts;
                here.gtd = T0 * here.gcrgs;
                ggtd = here.gtd;
                here.gtb = T0 * here.gcrgb;
                ggtb = here.gtb;
                gqdef = ScalingFactor * ag0;
                gcqgb = here.cqgb * ag0;
                gcqdb = here.cqsb * ag0;
                gcqsb = here.cqdb * ag0;
                gcqbb = here.cqbb * ag0;
                if fabs(qcheq) <= 1.0e-5 * CoxWL {
                    if model.xpart < 0.5 {
                        sxpart = 0.4;
                    } else if model.xpart > 0.5 {
                        sxpart = 0.0;
                    } else {
                        sxpart = 0.5;
                    }
                    dsxpart_dVs = 0.0;
                    dsxpart_dVb = dsxpart_dVs;
                    dsxpart_dVg = dsxpart_dVb;
                    dsxpart_dVd = dsxpart_dVg;
                } else {
                    sxpart = qdrn / qcheq;
                    Css = here.cddb;
                    Cds = -(here.cgdb + here.cddb + here.cbdb);
                    dsxpart_dVs = (Css - sxpart * (Css + Cds)) / qcheq;
                    Csg = here.cdgb;
                    Cdg = -(here.cggb + here.cdgb + here.cbgb);
                    dsxpart_dVg = (Csg - sxpart * (Csg + Cdg)) / qcheq;
                    Csd = here.cdsb;
                    Cdd = -(here.cgsb + here.cdsb + here.cbsb);
                    dsxpart_dVd = (Csd - sxpart * (Csd + Cdd)) / qcheq;
                    dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs);
                }
                dxpart = 1.0 - sxpart;
                ddxpart_dVd = -dsxpart_dVd;
                ddxpart_dVg = -dsxpart_dVg;
                ddxpart_dVs = -dsxpart_dVs;
                ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs);
                if here.rgateMod == 3 {
                    gcgmgmb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgmdb = -cgdo * ag0;
                    gcgmsb = -cgso * ag0;
                    gcgmbb = -pParam.cgbo * ag0;
                    gcdgmb = gcgmdb;
                    gcsgmb = gcgmsb;
                    gcbgmb = gcgmbb;
                    gcbgb = 0.0;
                    gcsgb = gcbgb;
                    gcdgb = gcsgb;
                    gcgbb = 0.0;
                    gcgsb = gcgbb;
                    gcgdb = gcgsb;
                    gcggb = gcgdb;
                    qgmb = pParam.cgbo * vgmb;
                    qgmid = qgdo + qgso + qgmb;
                    qgate = 0.0;
                    qbulk = -qgmb;
                } else {
                    gcggb = (cgdo + cgso + pParam.cgbo) * ag0;
                    gcgdb = -cgdo * ag0;
                    gcgsb = -cgso * ag0;
                    gcgbb = -pParam.cgbo * ag0;
                    gcdgb = gcgdb;
                    gcsgb = gcgsb;
                    gcbgb = gcgbb;
                    gcbgmb = 0.0;
                    gcsgmb = gcbgmb;
                    gcdgmb = gcsgmb;
                    qgb = pParam.cgbo * vgb;
                    qgate = qgdo + qgso + qgb;
                    qbulk = -qgb;
                }
                qdrn = -qgdo;
                qsrc = -qgso;
                gcddb = (here.capbd + cgdo) * ag0;
                gcsdb = 0.0;
                gcdsb = gcsdb;
                gcssb = (here.capbs + cgso) * ag0;
                if here.rbodyMod == 0 {
                    gcdbb = -(gcdgb + gcddb + gcdgmb);
                    gcsbb = -(gcsgb + gcssb + gcsgmb);
                    gcbdb = -here.capbd * ag0;
                    gcbsb = -here.capbs * ag0;
                    gcdbdb = 0.0;
                    gcsbsb = 0.0;
                } else {
                    gcbsb = 0.0;
                    gcbdb = gcbsb;
                    gcsbb = gcbdb;
                    gcdbb = gcsbb;
                    gcdbdb = -here.capbd * ag0;
                    gcsbsb = -here.capbs * ag0;
                }
                gcbbb = -(gcbdb + gcbgb + gcbsb + gcbgmb);
            }
        }
        if here.trnqsMod != 0 {
            ckt.set_state0(state::QCDUMP, qdef * ScalingFactor);
            if (ckt.mode() & mode::MODEINITTRAN) != 0 {
                ckt.set_state1(state::QCDUMP, ckt.state0(state::QCDUMP));
            }
            (geq, ceq) = ckt.integrate(0.0, state::QCDUMP)?;
        }
        ckt.set_state0(state::QG, qgate);
        ckt.set_state0(state::QD, qdrn - ckt.state0(state::QBD));
        ckt.set_state0(state::QS, qsrc - ckt.state0(state::QBS));
        if here.rgateMod == 3 {
            ckt.set_state0(state::QGMID, qgmid);
        }
        if here.rbodyMod == 0 {
            ckt.set_state0(
                state::QB,
                qbulk + ckt.state0(state::QBD) + ckt.state0(state::QBS),
            );
        } else {
            ckt.set_state0(state::QB, qbulk);
        }
        if (ckt.mode() & mode::MODEINITSMSIG) != 0 {
            goto_line1000 = true;
            break 'charges;
        }
        if ChargeComputationNeeded == 0 {
            goto_line850 = true;
            break 'charges;
        }
        if (ckt.mode() & mode::MODEDCTRANCURVE) != 0 {
            goto_line850 = true;
            break 'charges;
        }
        if (ckt.mode() & mode::MODEINITTRAN) != 0 {
            ckt.set_state1(state::QB, ckt.state0(state::QB));
            ckt.set_state1(state::QG, ckt.state0(state::QG));
            ckt.set_state1(state::QD, ckt.state0(state::QD));
            if here.rgateMod == 3 {
                ckt.set_state1(state::QGMID, ckt.state0(state::QGMID));
            }
            if here.rbodyMod != 0 {
                ckt.set_state1(state::QBS, ckt.state0(state::QBS));
                ckt.set_state1(state::QBD, ckt.state0(state::QBD));
            }
        }
        (geq, ceq) = ckt.integrate(0.0, state::QB)?;
        (geq, ceq) = ckt.integrate(0.0, state::QG)?;
        (geq, ceq) = ckt.integrate(0.0, state::QD)?;
        if here.rgateMod == 3 {
            (geq, ceq) = ckt.integrate(0.0, state::QGMID)?;
        }
        if here.rbodyMod != 0 {
            (geq, ceq) = ckt.integrate(0.0, state::QBS)?;
            (geq, ceq) = ckt.integrate(0.0, state::QBD)?;
        }
    }
    if !goto_line1000 {
        if goto_line850 {
            // C label `line850:`.
            ceqqd = 0.0;
            ceqqb = ceqqd;
            ceqqg = ceqqb;
            ceqqjs = 0.0;
            ceqqjd = ceqqjs;
            cqdef = 0.0;
            cqcheq = cqdef;
            gcdbb = 0.0;
            gcdsb = gcdbb;
            gcddb = gcdsb;
            gcdgb = gcddb;
            gcsbb = 0.0;
            gcssb = gcsbb;
            gcsdb = gcssb;
            gcsgb = gcsdb;
            gcgbb = 0.0;
            gcgsb = gcgbb;
            gcgdb = gcgsb;
            gcggb = gcgdb;
            gcbbb = 0.0;
            gcbsb = gcbbb;
            gcbgb = gcbsb;
            gcbdb = gcbgb;
            gcgmbb = 0.0;
            gcgmsb = gcgmbb;
            gcgmdb = gcgmsb;
            gcgmgmb = gcgmdb;
            ceqqgmid = 0.0;
            gcbgmb = ceqqgmid;
            gcsgmb = gcbgmb;
            gcdgmb = gcsgmb;
            gcsbsb = 0.0;
            gcdbdb = gcsbsb;
            gcqbb = 0.0;
            gcqsb = gcqbb;
            gcqdb = gcqsb;
            gcqgb = gcqdb;
            gqdef = gcqgb;
            ggts = 0.0;
            ggtb = ggts;
            ggtd = ggtb;
            ggtg = ggtd;
            dxpart = (if (here.mode > 0) { 0.4 } else { 0.6 });
            sxpart = (1.0 - (dxpart));
            ddxpart_dVs = 0.0;
            ddxpart_dVb = ddxpart_dVs;
            ddxpart_dVg = ddxpart_dVb;
            ddxpart_dVd = ddxpart_dVg;
            dsxpart_dVs = 0.0;
            dsxpart_dVb = dsxpart_dVs;
            dsxpart_dVg = dsxpart_dVb;
            dsxpart_dVd = dsxpart_dVg;
            if here.trnqsMod != 0 {
                CoxWL = model.coxe * pParam.weffCV * here.nf * pParam.leffCV;
                T1 = here.gcrg / CoxWL;
                here.gtau = T1 * ScalingFactor;
            } else {
                here.gtau = 0.0;
            }
        } else {
            // C label `line860:`.
            cqgate = ckt.state0(state::CQG);
            cqbody = ckt.state0(state::CQB);
            cqdrn = ckt.state0(state::CQD);
            ceqqg = cqgate - gcggb * vgb + gcgdb * vbd + gcgsb * vbs;
            ceqqd = cqdrn - gcdgb * vgb - gcdgmb * vgmb + (gcddb + gcdbdb) * vbd - gcdbdb * vbd_jct
                + gcdsb * vbs;
            ceqqb = cqbody - gcbgb * vgb - gcbgmb * vgmb + gcbdb * vbd + gcbsb * vbs;
            if here.rgateMod == 3 {
                ceqqgmid = ckt.state0(state::CQGMID) + gcgmdb * vbd + gcgmsb * vbs - gcgmgmb * vgmb;
            } else {
                ceqqgmid = 0.0;
            }
            if here.rbodyMod != 0 {
                ceqqjs = ckt.state0(state::CQBS) + gcsbsb * vbs_jct;
                ceqqjd = ckt.state0(state::CQBD) + gcdbdb * vbd_jct;
            }
            if here.trnqsMod != 0 {
                T0 = ggtg * vgb - ggtd * vbd - ggts * vbs;
                ceqqg += T0;
                T1 = qdef * here.gtau;
                ceqqd -=
                    dxpart * T0 + T1 * (ddxpart_dVg * vgb - ddxpart_dVd * vbd - ddxpart_dVs * vbs);
                cqdef = ckt.state0(state::CQCDUMP) - gqdef * qdef;
                cqcheq = ckt.state0(state::CQCHEQ) - (gcqgb * vgb - gcqdb * vbd - gcqsb * vbs) + T0;
            }
            if (ckt.mode() & mode::MODEINITTRAN) != 0 {
                ckt.set_state1(state::CQB, ckt.state0(state::CQB));
                ckt.set_state1(state::CQG, ckt.state0(state::CQG));
                ckt.set_state1(state::CQD, ckt.state0(state::CQD));
                if here.rgateMod == 3 {
                    ckt.set_state1(state::CQGMID, ckt.state0(state::CQGMID));
                }
                if here.rbodyMod != 0 {
                    ckt.set_state1(state::CQBS, ckt.state0(state::CQBS));
                    ckt.set_state1(state::CQBD, ckt.state0(state::CQBD));
                }
            }
        }
        // C label `line900:`.
        if here.mode >= 0 {
            Gm = here.gm;
            Gmbs = here.gmbs;
            FwdSum = Gm + Gmbs;
            RevSum = 0.0;
            ceqdrn = f64::from(model.r#type) * (cdrain - here.gds * vds - Gm * vgs - Gmbs * vbs);
            ceqbd = f64::from(model.r#type)
                * (here.csub + here.Igidl
                    - (here.gbds + here.ggidld) * vds
                    - (here.gbgs + here.ggidlg) * vgs
                    - (here.gbbs + here.ggidlb) * vbs);
            ceqbs = f64::from(model.r#type)
                * (here.Igisl + here.ggisls * vds - here.ggislg * vgd - here.ggislb * vbd);
            gbbdp = -(here.gbds);
            gbbsp = here.gbds + here.gbgs + here.gbbs;
            gbdpg = here.gbgs;
            gbdpdp = here.gbds;
            gbdpb = here.gbbs;
            gbdpsp = -(gbdpg + gbdpdp + gbdpb);
            gbspg = 0.0;
            gbspdp = 0.0;
            gbspb = 0.0;
            gbspsp = 0.0;
            if model.igcMod != 0 {
                gIstotg = here.gIgsg + here.gIgcsg;
                gIstotd = here.gIgcsd;
                gIstots = here.gIgss + here.gIgcss;
                gIstotb = here.gIgcsb;
                Istoteq = f64::from(model.r#type)
                    * (here.Igs + here.Igcs
                        - gIstotg * vgs
                        - here.gIgcsd * vds
                        - here.gIgcsb * vbs);
                gIdtotg = here.gIgdg + here.gIgcdg;
                gIdtotd = here.gIgdd + here.gIgcdd;
                gIdtots = here.gIgcds;
                gIdtotb = here.gIgcdb;
                Idtoteq = f64::from(model.r#type)
                    * (here.Igd + here.Igcd
                        - here.gIgdg * vgd
                        - here.gIgcdg * vgs
                        - here.gIgcdd * vds
                        - here.gIgcdb * vbs);
            } else {
                Istoteq = 0.0;
                gIstotb = Istoteq;
                gIstots = gIstotb;
                gIstotd = gIstots;
                gIstotg = gIstotd;
                Idtoteq = 0.0;
                gIdtotb = Idtoteq;
                gIdtots = gIdtotb;
                gIdtotd = gIdtots;
                gIdtotg = gIdtotd;
            }
            if model.igbMod != 0 {
                gIbtotg = here.gIgbg;
                gIbtotd = here.gIgbd;
                gIbtots = here.gIgbs;
                gIbtotb = here.gIgbb;
                Ibtoteq = f64::from(model.r#type)
                    * (here.Igb - here.gIgbg * vgs - here.gIgbd * vds - here.gIgbb * vbs);
            } else {
                Ibtoteq = 0.0;
                gIbtotb = Ibtoteq;
                gIbtots = gIbtotb;
                gIbtotd = gIbtots;
                gIbtotg = gIbtotd;
            }
            if (model.igcMod != 0) || (model.igbMod != 0) {
                gIgtotg = gIstotg + gIdtotg + gIbtotg;
                gIgtotd = gIstotd + gIdtotd + gIbtotd;
                gIgtots = gIstots + gIdtots + gIbtots;
                gIgtotb = gIstotb + gIdtotb + gIbtotb;
                Igtoteq = Istoteq + Idtoteq + Ibtoteq;
            } else {
                Igtoteq = 0.0;
                gIgtotb = Igtoteq;
                gIgtots = gIgtotb;
                gIgtotd = gIgtots;
                gIgtotg = gIgtotd;
            }
            // Setup guarantees rgateMod is 0..=3; only 2/3 need a gate drop.
            if matches!(here.rgateMod, 2 | 3) {
                let T0 = if here.rgateMod == 2 {
                    vges - vgs
                } else {
                    vgms - vgs
                };
                gcrgd = here.gcrgd * T0;
                gcrgg = here.gcrgg * T0;
                gcrgs = here.gcrgs * T0;
                gcrgb = here.gcrgb * T0;
                ceqgcrg = -(gcrgd * vds + gcrgg * vgs + gcrgb * vbs);
                gcrgg -= here.gcrg;
                gcrg = here.gcrg;
            } else {
                gcrgb = 0.0;
                gcrgs = gcrgb;
                gcrgg = gcrgs;
                gcrgd = gcrgg;
                gcrg = gcrgd;
                ceqgcrg = gcrg;
            }
        } else {
            Gm = -here.gm;
            Gmbs = -here.gmbs;
            FwdSum = 0.0;
            RevSum = -(Gm + Gmbs);
            ceqdrn = f64::from(-model.r#type) * (cdrain + here.gds * vds + Gm * vgd + Gmbs * vbd);
            ceqbs = f64::from(model.r#type)
                * (here.csub + here.Igisl + (here.gbds + here.ggisls) * vds
                    - (here.gbgs + here.ggislg) * vgd
                    - (here.gbbs + here.ggislb) * vbd);
            ceqbd = f64::from(model.r#type)
                * (here.Igidl - here.ggidld * vds - here.ggidlg * vgs - here.ggidlb * vbs);
            gbbsp = -(here.gbds);
            gbbdp = here.gbds + here.gbgs + here.gbbs;
            gbdpg = 0.0;
            gbdpsp = 0.0;
            gbdpb = 0.0;
            gbdpdp = 0.0;
            gbspg = here.gbgs;
            gbspsp = here.gbds;
            gbspb = here.gbbs;
            gbspdp = -(gbspg + gbspsp + gbspb);
            if model.igcMod != 0 {
                gIstotg = here.gIgsg + here.gIgcdg;
                gIstotd = here.gIgcds;
                gIstots = here.gIgss + here.gIgcdd;
                gIstotb = here.gIgcdb;
                Istoteq = f64::from(model.r#type)
                    * (here.Igs + here.Igcd - here.gIgsg * vgs - here.gIgcdg * vgd
                        + here.gIgcdd * vds
                        - here.gIgcdb * vbd);
                gIdtotg = here.gIgdg + here.gIgcsg;
                gIdtotd = here.gIgdd + here.gIgcss;
                gIdtots = here.gIgcsd;
                gIdtotb = here.gIgcsb;
                Idtoteq = f64::from(model.r#type)
                    * (here.Igd + here.Igcs - (here.gIgdg + here.gIgcsg) * vgd + here.gIgcsd * vds
                        - here.gIgcsb * vbd);
            } else {
                Istoteq = 0.0;
                gIstotb = Istoteq;
                gIstots = gIstotb;
                gIstotd = gIstots;
                gIstotg = gIstotd;
                Idtoteq = 0.0;
                gIdtotb = Idtoteq;
                gIdtots = gIdtotb;
                gIdtotd = gIdtots;
                gIdtotg = gIdtotd;
            }
            if model.igbMod != 0 {
                gIbtotg = here.gIgbg;
                gIbtotd = here.gIgbs;
                gIbtots = here.gIgbd;
                gIbtotb = here.gIgbb;
                Ibtoteq = f64::from(model.r#type)
                    * (here.Igb - here.gIgbg * vgd + here.gIgbd * vds - here.gIgbb * vbd);
            } else {
                Ibtoteq = 0.0;
                gIbtotb = Ibtoteq;
                gIbtots = gIbtotb;
                gIbtotd = gIbtots;
                gIbtotg = gIbtotd;
            }
            if (model.igcMod != 0) || (model.igbMod != 0) {
                gIgtotg = gIstotg + gIdtotg + gIbtotg;
                gIgtotd = gIstotd + gIdtotd + gIbtotd;
                gIgtots = gIstots + gIdtots + gIbtots;
                gIgtotb = gIstotb + gIdtotb + gIbtotb;
                Igtoteq = Istoteq + Idtoteq + Ibtoteq;
            } else {
                Igtoteq = 0.0;
                gIgtotb = Igtoteq;
                gIgtots = gIgtotb;
                gIgtotd = gIgtots;
                gIgtotg = gIgtotd;
            }
            // Setup guarantees rgateMod is 0..=3; only 2/3 need a gate drop.
            if matches!(here.rgateMod, 2 | 3) {
                let T0 = if here.rgateMod == 2 {
                    vges - vgs
                } else {
                    vgms - vgs
                };
                gcrgd = here.gcrgs * T0;
                gcrgg = here.gcrgg * T0;
                gcrgs = here.gcrgd * T0;
                gcrgb = here.gcrgb * T0;
                ceqgcrg = -(gcrgg * vgd - gcrgs * vds + gcrgb * vbd);
                gcrgg -= here.gcrg;
                gcrg = here.gcrg;
            } else {
                gcrgb = 0.0;
                gcrgs = gcrgb;
                gcrgg = gcrgs;
                gcrgd = gcrgg;
                gcrg = gcrgd;
                ceqgcrg = gcrg;
            }
        }
        if model.rdsMod == 1 {
            ceqgstot = f64::from(model.r#type)
                * (here.gstotd * vds + here.gstotg * vgs + here.gstotb * vbs);
            gstot = here.gstot;
            gstotd = here.gstotd;
            gstotg = here.gstotg;
            gstots = here.gstots - gstot;
            gstotb = here.gstotb;
            ceqgdtot = f64::from(-model.r#type)
                * (here.gdtotd * vds + here.gdtotg * vgs + here.gdtotb * vbs);
            gdtot = here.gdtot;
            gdtotd = here.gdtotd - gdtot;
            gdtotg = here.gdtotg;
            gdtots = here.gdtots;
            gdtotb = here.gdtotb;
        } else {
            ceqgstot = 0.0;
            gstotb = ceqgstot;
            gstots = gstotb;
            gstotg = gstots;
            gstotd = gstotg;
            gstot = gstotd;
            ceqgdtot = 0.0;
            gdtotb = ceqgdtot;
            gdtots = gdtotb;
            gdtotg = gdtots;
            gdtotd = gdtotg;
            gdtot = gdtotd;
        }
        if model.r#type > 0 {
            ceqjs = (here.cbs - here.gbs * vbs_jct);
            ceqjd = (here.cbd - here.gbd * vbd_jct);
        } else {
            ceqjs = -(here.cbs - here.gbs * vbs_jct);
            ceqjd = -(here.cbd - here.gbd * vbd_jct);
            ceqqg = -ceqqg;
            ceqqd = -ceqqd;
            ceqqb = -ceqqb;
            ceqgcrg = -ceqgcrg;
            if here.trnqsMod != 0 {
                cqdef = -cqdef;
                cqcheq = -cqcheq;
            }
            if here.rbodyMod != 0 {
                ceqqjs = -ceqqjs;
                ceqqjd = -ceqqjd;
            }
            if here.rgateMod == 3 {
                ceqqgmid = -ceqqgmid;
            }
        }
        m = here.m;
        mult_i = here.mult_i * here.m;
        mult_q = here.mult_q * here.m;
        ckt.rhs(
            here.dNodePrime,
            (mult_i * (ceqjd - ceqbd + ceqgdtot - ceqdrn + Idtoteq) - mult_q * ceqqd),
        );
        ckt.rhs(
            here.gNodePrime,
            -(mult_q * ceqqg - mult_i * (ceqgcrg - Igtoteq)),
        );
        if here.rgateMod == 2 {
            ckt.rhs(here.gNodeExt, -(mult_i * ceqgcrg));
        } else if here.rgateMod == 3 {
            ckt.rhs(here.gNodeMid, -(mult_q * ceqqgmid + mult_i * ceqgcrg));
        }
        if here.rbodyMod == 0 {
            ckt.rhs(
                here.bNodePrime,
                (mult_i * (ceqbd + ceqbs - ceqjd - ceqjs + Ibtoteq) - mult_q * ceqqb),
            );
            ckt.rhs(
                here.sNodePrime,
                (mult_i * (ceqdrn - ceqbs + ceqjs - ceqgstot + Istoteq)
                    + mult_q * (ceqqg + ceqqb + ceqqd + ceqqgmid)),
            );
        } else {
            ckt.rhs(here.dbNode, -(mult_i * (ceqjd) + mult_q * ceqqjd));
            ckt.rhs(
                here.bNodePrime,
                (mult_i * (ceqbd + ceqbs + Ibtoteq) - mult_q * ceqqb),
            );
            ckt.rhs(here.sbNode, -(mult_i * (ceqjs) + mult_q * ceqqjs));
            ckt.rhs(
                here.sNodePrime,
                (mult_i * (ceqdrn - ceqbs + ceqjs - ceqgstot + Istoteq)
                    + mult_q * (ceqqd + ceqqg + ceqqb + ceqqjd + ceqqjs + ceqqgmid)),
            );
        }
        if model.rdsMod != 0 {
            ckt.rhs(here.dNode, -(mult_i * ceqgdtot));
            ckt.rhs(here.sNode, mult_i * ceqgstot);
        }
        if here.trnqsMod != 0 {
            ckt.rhs(here.qNode, m * (cqcheq - cqdef));
        }
        if here.rbodyMod == 0 {
            gjbd = here.gbd;
            gjbs = here.gbs;
        } else {
            gjbs = 0.0;
            gjbd = gjbs;
        }
        if model.rdsMod == 0 {
            gdpr = here.drainConductance;
            gspr = here.sourceConductance;
        } else {
            gspr = 0.0;
            gdpr = gspr;
        }
        geltd = here.grgeltd;
        T1 = qdef * here.gtau;
        if here.rgateMod == 1 {
            ckt.matrix(here.gNodeExt, here.gNodeExt, mult_i * geltd);
            ckt.matrix(here.gNodePrime, here.gNodeExt, -(mult_i * geltd));
            ckt.matrix(here.gNodeExt, here.gNodePrime, -(mult_i * geltd));
            ckt.matrix(
                here.gNodePrime,
                here.gNodePrime,
                mult_q * (gcggb - ggtg) + mult_i * (geltd + gIgtotg),
            );
            ckt.matrix(
                here.gNodePrime,
                here.dNodePrime,
                mult_q * (gcgdb - ggtd) + mult_i * gIgtotd,
            );
            ckt.matrix(
                here.gNodePrime,
                here.sNodePrime,
                mult_q * (gcgsb - ggts) + mult_i * gIgtots,
            );
            ckt.matrix(
                here.gNodePrime,
                here.bNodePrime,
                mult_q * (gcgbb - ggtb) + mult_i * gIgtotb,
            );
        } else if here.rgateMod == 2 {
            ckt.matrix(here.gNodeExt, here.gNodeExt, mult_i * gcrg);
            ckt.matrix(here.gNodeExt, here.gNodePrime, mult_i * gcrgg);
            ckt.matrix(here.gNodeExt, here.dNodePrime, mult_i * gcrgd);
            ckt.matrix(here.gNodeExt, here.sNodePrime, mult_i * gcrgs);
            ckt.matrix(here.gNodeExt, here.bNodePrime, mult_i * gcrgb);
            ckt.matrix(here.gNodePrime, here.gNodeExt, -(mult_i * gcrg));
            ckt.matrix(
                here.gNodePrime,
                here.gNodePrime,
                mult_q * (gcggb - ggtg) + mult_i * (gIgtotg - gcrgg),
            );
            ckt.matrix(
                here.gNodePrime,
                here.dNodePrime,
                mult_q * (gcgdb - ggtd) + mult_i * (gIgtotd - gcrgd),
            );
            ckt.matrix(
                here.gNodePrime,
                here.sNodePrime,
                mult_q * (gcgsb - ggts) + mult_i * (gIgtots - gcrgs),
            );
            ckt.matrix(
                here.gNodePrime,
                here.bNodePrime,
                mult_q * (gcgbb - ggtb) + mult_i * (gIgtotb - gcrgb),
            );
        } else if here.rgateMod == 3 {
            ckt.matrix(here.gNodeExt, here.gNodeExt, mult_i * geltd);
            ckt.matrix(here.gNodeExt, here.gNodeMid, -(mult_i * geltd));
            ckt.matrix(here.gNodeMid, here.gNodeExt, -(mult_i * geltd));
            ckt.matrix(
                here.gNodeMid,
                here.gNodeMid,
                mult_i * (geltd + gcrg) + mult_q * gcgmgmb,
            );
            ckt.matrix(
                here.gNodeMid,
                here.dNodePrime,
                mult_i * gcrgd + mult_q * gcgmdb,
            );
            ckt.matrix(here.gNodeMid, here.gNodePrime, mult_i * gcrgg);
            ckt.matrix(
                here.gNodeMid,
                here.sNodePrime,
                mult_i * gcrgs + mult_q * gcgmsb,
            );
            ckt.matrix(
                here.gNodeMid,
                here.bNodePrime,
                mult_i * gcrgb + mult_q * gcgmbb,
            );
            ckt.matrix(here.dNodePrime, here.gNodeMid, mult_q * gcdgmb);
            ckt.matrix(here.gNodePrime, here.gNodeMid, -(mult_i * gcrg));
            ckt.matrix(here.sNodePrime, here.gNodeMid, mult_q * gcsgmb);
            ckt.matrix(here.bNodePrime, here.gNodeMid, mult_q * gcbgmb);
            ckt.matrix(
                here.gNodePrime,
                here.gNodePrime,
                mult_q * (gcggb - ggtg) + mult_i * (gIgtotg - gcrgg),
            );
            ckt.matrix(
                here.gNodePrime,
                here.dNodePrime,
                mult_q * (gcgdb - ggtd) + mult_i * (gIgtotd - gcrgd),
            );
            ckt.matrix(
                here.gNodePrime,
                here.sNodePrime,
                mult_q * (gcgsb - ggts) + mult_i * (gIgtots - gcrgs),
            );
            ckt.matrix(
                here.gNodePrime,
                here.bNodePrime,
                mult_q * (gcgbb - ggtb) + mult_i * (gIgtotb - gcrgb),
            );
        } else {
            ckt.matrix(
                here.gNodePrime,
                here.gNodePrime,
                mult_q * (gcggb - ggtg) + mult_i * gIgtotg,
            );
            ckt.matrix(
                here.gNodePrime,
                here.dNodePrime,
                mult_q * (gcgdb - ggtd) + mult_i * gIgtotd,
            );
            ckt.matrix(
                here.gNodePrime,
                here.sNodePrime,
                mult_q * (gcgsb - ggts) + mult_i * gIgtots,
            );
            ckt.matrix(
                here.gNodePrime,
                here.bNodePrime,
                mult_q * (gcgbb - ggtb) + mult_i * gIgtotb,
            );
        }
        if model.rdsMod != 0 {
            ckt.matrix(here.dNode, here.gNodePrime, mult_i * gdtotg);
            ckt.matrix(here.dNode, here.sNodePrime, mult_i * gdtots);
            ckt.matrix(here.dNode, here.bNodePrime, mult_i * gdtotb);
            ckt.matrix(here.sNode, here.dNodePrime, mult_i * gstotd);
            ckt.matrix(here.sNode, here.gNodePrime, mult_i * gstotg);
            ckt.matrix(here.sNode, here.bNodePrime, mult_i * gstotb);
        }
        ckt.matrix(
            here.dNodePrime,
            here.dNodePrime,
            mult_i * (gdpr + here.gds + here.gbd - gdtotd + RevSum + gbdpdp - gIdtotd)
                + mult_q * (T1 * ddxpart_dVd + gcddb + dxpart * ggtd),
        );
        ckt.matrix(here.dNodePrime, here.dNode, -(mult_i * (gdpr + gdtot)));
        ckt.matrix(
            here.dNodePrime,
            here.gNodePrime,
            mult_i * (Gm - gdtotg + gbdpg - gIdtotg)
                + mult_q * (dxpart * ggtg + T1 * ddxpart_dVg + gcdgb),
        );
        ckt.matrix(
            here.dNodePrime,
            here.sNodePrime,
            -(mult_i * (here.gds + gdtots + gIdtots + FwdSum - gbdpsp)
                - mult_q * (dxpart * ggts + T1 * ddxpart_dVs + gcdsb)),
        );
        ckt.matrix(
            here.dNodePrime,
            here.bNodePrime,
            -(mult_i * (gjbd + gdtotb - Gmbs - gbdpb + gIdtotb)
                - mult_q * (dxpart * ggtb + gcdbb + T1 * ddxpart_dVb)),
        );
        ckt.matrix(here.dNode, here.dNodePrime, -(mult_i * (gdpr - gdtotd)));
        ckt.matrix(here.dNode, here.dNode, mult_i * (gdpr + gdtot));
        ckt.matrix(
            here.sNodePrime,
            here.dNodePrime,
            -(mult_i * (here.gds + gstotd + RevSum - gbspdp + gIstotd)
                - mult_q * (T1 * dsxpart_dVd + sxpart * ggtd + gcsdb)),
        );
        ckt.matrix(
            here.sNodePrime,
            here.gNodePrime,
            mult_q * (gcsgb + sxpart * ggtg + T1 * dsxpart_dVg)
                + mult_i * (gbspg - Gm - gstotg - gIstotg),
        );
        ckt.matrix(
            here.sNodePrime,
            here.sNodePrime,
            mult_i * (gspr + here.gds + here.gbs - gIstots - gstots + FwdSum + gbspsp)
                + mult_q * (sxpart * ggts + T1 * dsxpart_dVs + gcssb),
        );
        ckt.matrix(here.sNodePrime, here.sNode, -(mult_i * (gspr + gstot)));
        ckt.matrix(
            here.sNodePrime,
            here.bNodePrime,
            -(mult_i * (gjbs + gstotb + Gmbs - gbspb + gIstotb)
                - mult_q * (gcsbb + sxpart * ggtb + T1 * dsxpart_dVb)),
        );
        ckt.matrix(here.sNode, here.sNodePrime, -(mult_i * (gspr - gstots)));
        ckt.matrix(here.sNode, here.sNode, mult_i * (gspr + gstot));
        ckt.matrix(
            here.bNodePrime,
            here.dNodePrime,
            mult_q * gcbdb - mult_i * (gjbd - gbbdp + gIbtotd),
        );
        ckt.matrix(
            here.bNodePrime,
            here.gNodePrime,
            mult_q * gcbgb - mult_i * (here.gbgs + gIbtotg),
        );
        ckt.matrix(
            here.bNodePrime,
            here.sNodePrime,
            mult_q * gcbsb - mult_i * (gjbs - gbbsp + gIbtots),
        );
        ckt.matrix(
            here.bNodePrime,
            here.bNodePrime,
            mult_i * (gjbd + gjbs - here.gbbs - gIbtotb) + mult_q * gcbbb,
        );
        ggidld = here.ggidld;
        ggidlg = here.ggidlg;
        ggidlb = here.ggidlb;
        ggislg = here.ggislg;
        ggisls = here.ggisls;
        ggislb = here.ggislb;
        ckt.matrix(here.dNodePrime, here.dNodePrime, mult_i * ggidld);
        ckt.matrix(here.dNodePrime, here.gNodePrime, mult_i * ggidlg);
        ckt.matrix(
            here.dNodePrime,
            here.sNodePrime,
            -(mult_i * (ggidlg + ggidld + ggidlb)),
        );
        ckt.matrix(here.dNodePrime, here.bNodePrime, mult_i * ggidlb);
        ckt.matrix(here.bNodePrime, here.dNodePrime, -(mult_i * ggidld));
        ckt.matrix(here.bNodePrime, here.gNodePrime, -(mult_i * ggidlg));
        ckt.matrix(
            here.bNodePrime,
            here.sNodePrime,
            mult_i * (ggidlg + ggidld + ggidlb),
        );
        ckt.matrix(here.bNodePrime, here.bNodePrime, -(mult_i * ggidlb));
        ckt.matrix(
            here.sNodePrime,
            here.dNodePrime,
            -(mult_i * (ggisls + ggislg + ggislb)),
        );
        ckt.matrix(here.sNodePrime, here.gNodePrime, mult_i * ggislg);
        ckt.matrix(here.sNodePrime, here.sNodePrime, mult_i * ggisls);
        ckt.matrix(here.sNodePrime, here.bNodePrime, mult_i * ggislb);
        ckt.matrix(
            here.bNodePrime,
            here.dNodePrime,
            mult_i * (ggislg + ggisls + ggislb),
        );
        ckt.matrix(here.bNodePrime, here.gNodePrime, -(mult_i * ggislg));
        ckt.matrix(here.bNodePrime, here.sNodePrime, -(mult_i * ggisls));
        ckt.matrix(here.bNodePrime, here.bNodePrime, -(mult_i * ggislb));
        if here.rbodyMod != 0 {
            ckt.matrix(
                here.dNodePrime,
                here.dbNode,
                mult_q * gcdbdb - mult_i * here.gbd,
            );
            ckt.matrix(
                here.sNodePrime,
                here.sbNode,
                -(mult_i * here.gbs - mult_q * gcsbsb),
            );
            ckt.matrix(
                here.dbNode,
                here.dNodePrime,
                mult_q * gcdbdb - mult_i * here.gbd,
            );
            ckt.matrix(
                here.dbNode,
                here.dbNode,
                mult_i * (here.gbd + here.grbpd + here.grbdb) - mult_q * gcdbdb,
            );
            ckt.matrix(here.dbNode, here.bNodePrime, -(mult_i * here.grbpd));
            ckt.matrix(here.dbNode, here.bNode, -(mult_i * here.grbdb));
            ckt.matrix(here.bNodePrime, here.dbNode, -(mult_i * here.grbpd));
            ckt.matrix(here.bNodePrime, here.bNode, -(mult_i * here.grbpb));
            ckt.matrix(here.bNodePrime, here.sbNode, -(mult_i * here.grbps));
            ckt.matrix(
                here.bNodePrime,
                here.bNodePrime,
                mult_i * (here.grbpd + here.grbps + here.grbpb),
            );
            ckt.matrix(
                here.sbNode,
                here.sNodePrime,
                mult_q * gcsbsb - mult_i * here.gbs,
            );
            ckt.matrix(here.sbNode, here.bNodePrime, -(mult_i * here.grbps));
            ckt.matrix(here.sbNode, here.bNode, -(mult_i * here.grbsb));
            ckt.matrix(
                here.sbNode,
                here.sbNode,
                mult_i * (here.gbs + here.grbps + here.grbsb) - mult_q * gcsbsb,
            );
            ckt.matrix(here.bNode, here.dbNode, -(mult_i * here.grbdb));
            ckt.matrix(here.bNode, here.bNodePrime, -(mult_i * here.grbpb));
            ckt.matrix(here.bNode, here.sbNode, -(mult_i * here.grbsb));
            ckt.matrix(
                here.bNode,
                here.bNode,
                mult_i * (here.grbsb + here.grbdb + here.grbpb),
            );
        }
        if here.trnqsMod != 0 {
            ckt.matrix(here.qNode, here.qNode, m * (gqdef + here.gtau));
            ckt.matrix(here.qNode, here.gNodePrime, m * (ggtg - gcqgb));
            ckt.matrix(here.qNode, here.dNodePrime, m * (ggtd - gcqdb));
            ckt.matrix(here.qNode, here.sNodePrime, m * (ggts - gcqsb));
            ckt.matrix(here.qNode, here.bNodePrime, m * (ggtb - gcqbb));
            ckt.matrix(here.dNodePrime, here.qNode, mult_q * (dxpart * here.gtau));
            ckt.matrix(here.sNodePrime, here.qNode, mult_q * (sxpart * here.gtau));
            ckt.matrix(here.gNodePrime, here.qNode, -(mult_q * (here.gtau)));
        }
    }
    // C label `line1000:`.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::BSIM4polyDepletion;

    #[test]
    fn poly_depletion_matches_hand_evaluation() {
        // NGATE = 2e19 cm^-3 gives a 58 mV poly drop at Vgs - phi = 0.6 V;
        // the values are b4ld.c's expressions evaluated in IEEE doubles.
        let (vgs_eff, d) = BSIM4polyDepletion(0.9, 2e19, 1.03594e-10, 0.0115, 1.5);
        assert_eq!(vgs_eff, 1.444_195_874_463_227_7);
        assert_eq!(d, 0.830_890_287_421_742_1);
    }

    #[test]
    fn poly_depletion_inactive_outside_its_ranges() {
        // Below the flat-band phi, NGATE outside (1e18, 1e25), or EPSGATE = 0.
        for (phi, ngate, eps, vgs) in [
            (0.9, 2e19, 1.03594e-10, 0.5),
            (0.9, 1e18, 1.03594e-10, 1.5),
            (0.9, 1e25, 1.03594e-10, 1.5),
            (0.9, 2e19, 0.0, 1.5),
        ] {
            assert_eq!(BSIM4polyDepletion(phi, ngate, eps, 0.0115, vgs), (vgs, 1.0));
        }
    }
}

#[cfg(test)]
#[path = "load/tests.rs"]
mod trace_tests;
