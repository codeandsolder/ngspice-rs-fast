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
    clippy::manual_range_contains,
    reason = "comparison spelling matches the reference C conditional sequence"
)] // C's range tests.
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
mod junction;

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
    // C's locals, all declared up front as C declares them; those C leaves
    // uninitialised start at 0.0 (see the module notes on reachability).
    let mut ceqgstot: f64 = 0.0;
    let mut dgstot_dvd: f64 = 0.0;
    let mut dgstot_dvg: f64 = 0.0;
    let mut dgstot_dvs: f64 = 0.0;
    let mut dgstot_dvb: f64 = 0.0;
    let mut ceqgdtot: f64 = 0.0;
    let mut dgdtot_dvd: f64 = 0.0;
    let mut dgdtot_dvg: f64 = 0.0;
    let mut dgdtot_dvs: f64 = 0.0;
    let mut dgdtot_dvb: f64 = 0.0;
    let mut gstot: f64 = 0.0;
    let mut gstotd: f64 = 0.0;
    let mut gstotg: f64 = 0.0;
    let mut gstots: f64 = 0.0;
    let mut gstotb: f64 = 0.0;
    let mut gspr: f64 = 0.0;
    let mut Rs: f64 = 0.0;
    let mut Rd: f64 = 0.0;
    let mut gdtot: f64 = 0.0;
    let mut gdtotd: f64 = 0.0;
    let mut gdtotg: f64 = 0.0;
    let mut gdtots: f64 = 0.0;
    let mut gdtotb: f64 = 0.0;
    let mut gdpr: f64 = 0.0;
    let mut vgs_eff: f64 = 0.0;
    let mut vgd_eff: f64 = 0.0;
    let mut dvgs_eff_dvg: f64 = 0.0;
    let mut dvgd_eff_dvg: f64 = 0.0;
    let mut dRs_dvg: f64 = 0.0;
    let mut dRd_dvg: f64 = 0.0;
    let mut dRs_dvb: f64 = 0.0;
    let mut dRd_dvb: f64 = 0.0;
    let mut dT0_dvg: f64 = 0.0;
    let mut dT1_dvb: f64 = 0.0;
    let mut dT3_dvg: f64 = 0.0;
    let mut dT3_dvb: f64 = 0.0;
    let mut geltd: f64 = 0.0;
    let mut gcrg: f64 = 0.0;
    let mut gcrgg: f64 = 0.0;
    let mut gcrgd: f64 = 0.0;
    let mut gcrgs: f64 = 0.0;
    let mut gcrgb: f64 = 0.0;
    let mut ceqgcrg: f64 = 0.0;
    let mut gcgmgmb: f64 = 0.0;
    let mut gcgmdb: f64 = 0.0;
    let mut gcgmsb: f64 = 0.0;
    let mut gcdgmb: f64 = 0.0;
    let mut gcsgmb: f64 = 0.0;
    let mut gcgmbb: f64 = 0.0;
    let mut gcbgmb: f64 = 0.0;
    let mut qgmb: f64 = 0.0;
    let mut qgmid: f64 = 0.0;
    let mut ceqqgmid: f64 = 0.0;
    let mut ag0: f64 = 0.0;
    let mut qgb: f64 = 0.0;
    let mut VgstNVt: f64 = 0.0;
    let mut ExpVgst: f64 = 0.0;
    let mut ceqqb: f64 = 0.0;
    let mut ceqqd: f64 = 0.0;
    let mut ceqqg: f64 = 0.0;
    let mut ceqqjd: f64 = 0.0;
    let mut ceqqjs: f64 = 0.0;
    let mut ceq: f64 = 0.0;
    let mut geq: f64 = 0.0;
    let mut cdrain: f64 = 0.0;
    let mut ceqdrn: f64 = 0.0;
    let mut ceqbd: f64 = 0.0;
    let mut ceqbs: f64 = 0.0;
    let mut ceqjd: f64 = 0.0;
    let mut ceqjs: f64 = 0.0;
    let mut gjbd: f64 = 0.0;
    let mut gjbs: f64 = 0.0;
    let mut Vfbeff: f64 = 0.0;
    let mut dVfbeff_dVg: f64 = 0.0;
    let mut dVfbeff_dVb: f64 = 0.0;
    let mut V3: f64 = 0.0;
    let mut V4: f64 = 0.0;
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
    let mut qsrc: f64 = 0.0;
    let mut cqgate: f64 = 0.0;
    let mut cqbody: f64 = 0.0;
    let mut cqdrn: f64 = 0.0;
    let mut Vdb: f64 = 0.0;
    let mut Vds: f64 = 0.0;
    let mut Vgs: f64 = 0.0;
    let mut Vbs: f64 = 0.0;
    let mut Gmbs: f64 = 0.0;
    let mut FwdSum: f64 = 0.0;
    let mut RevSum: f64 = 0.0;
    let mut Igidl: f64 = 0.0;
    let mut Ggidld: f64 = 0.0;
    let mut Ggidlg: f64 = 0.0;
    let mut Ggidlb: f64 = 0.0;
    let mut Voxacc: f64 = 0.0;
    let mut dVoxacc_dVg: f64 = 0.0;
    let mut dVoxacc_dVb: f64 = 0.0;
    let mut Voxdepinv: f64 = 0.0;
    let mut dVoxdepinv_dVg: f64 = 0.0;
    let mut dVoxdepinv_dVd: f64 = 0.0;
    let mut dVoxdepinv_dVb: f64 = 0.0;
    let mut VxNVt: f64 = 0.0;
    let mut ExpVxNVt: f64 = 0.0;
    let mut Vaux: f64 = 0.0;
    let mut dVaux_dVg: f64 = 0.0;
    let mut dVaux_dVd: f64 = 0.0;
    let mut dVaux_dVb: f64 = 0.0;
    let mut Igc: f64 = 0.0;
    let mut dIgc_dVg: f64 = 0.0;
    let mut dIgc_dVd: f64 = 0.0;
    let mut dIgc_dVb: f64 = 0.0;
    let mut Igcs: f64 = 0.0;
    let mut dIgcs_dVg: f64 = 0.0;
    let mut dIgcs_dVd: f64 = 0.0;
    let mut dIgcs_dVb: f64 = 0.0;
    let mut Igcd: f64 = 0.0;
    let mut dIgcd_dVg: f64 = 0.0;
    let mut dIgcd_dVd: f64 = 0.0;
    let mut dIgcd_dVb: f64 = 0.0;
    let mut Igs: f64 = 0.0;
    let mut dIgs_dVg: f64 = 0.0;
    let mut dIgs_dVs: f64 = 0.0;
    let mut Igd: f64 = 0.0;
    let mut dIgd_dVg: f64 = 0.0;
    let mut dIgd_dVd: f64 = 0.0;
    let mut Igbacc: f64 = 0.0;
    let mut dIgbacc_dVg: f64 = 0.0;
    let mut dIgbacc_dVb: f64 = 0.0;
    let mut Igbinv: f64 = 0.0;
    let mut dIgbinv_dVg: f64 = 0.0;
    let mut dIgbinv_dVd: f64 = 0.0;
    let mut dIgbinv_dVb: f64 = 0.0;
    let mut Pigcd: f64 = 0.0;
    let mut dPigcd_dVg: f64 = 0.0;
    let mut dPigcd_dVd: f64 = 0.0;
    let mut dPigcd_dVb: f64 = 0.0;
    let mut Istoteq: f64 = 0.0;
    let mut gIstotg: f64 = 0.0;
    let mut gIstotd: f64 = 0.0;
    let mut gIstots: f64 = 0.0;
    let mut gIstotb: f64 = 0.0;
    let mut Idtoteq: f64 = 0.0;
    let mut gIdtotg: f64 = 0.0;
    let mut gIdtotd: f64 = 0.0;
    let mut gIdtots: f64 = 0.0;
    let mut gIdtotb: f64 = 0.0;
    let mut Ibtoteq: f64 = 0.0;
    let mut gIbtotg: f64 = 0.0;
    let mut gIbtotd: f64 = 0.0;
    let mut gIbtots: f64 = 0.0;
    let mut gIbtotb: f64 = 0.0;
    let mut Igtoteq: f64 = 0.0;
    let mut gIgtotg: f64 = 0.0;
    let mut gIgtotd: f64 = 0.0;
    let mut gIgtots: f64 = 0.0;
    let mut gIgtotb: f64 = 0.0;
    let mut Vgs_eff: f64 = 0.0;
    let mut Vfb: f64 = 0.0;
    let mut Vth_NarrowW: f64 = 0.0;
    let mut Phis: f64 = 0.0;
    let mut dPhis_dVb: f64 = 0.0;
    let mut sqrtPhis: f64 = 0.0;
    let mut dsqrtPhis_dVb: f64 = 0.0;
    let mut Vth: f64 = 0.0;
    let mut dVth_dVb: f64 = 0.0;
    let mut dVth_dVd: f64 = 0.0;
    let mut Vgst: f64 = 0.0;
    let mut dVgst_dVg: f64 = 0.0;
    let mut dVgst_dVb: f64 = 0.0;
    let mut dVgs_eff_dVg: f64 = 0.0;
    let mut Vtm: f64 = 0.0;
    let mut Vtm0: f64 = 0.0;
    let mut n: f64 = 0.0;
    let mut dn_dVb: f64 = 0.0;
    let mut dn_dVd: f64 = 0.0;
    let mut voffcv: f64 = 0.0;
    let mut noff: f64 = 0.0;
    let mut dnoff_dVd: f64 = 0.0;
    let mut dnoff_dVb: f64 = 0.0;
    let mut V0: f64 = 0.0;
    let mut CoxWLcen: f64 = 0.0;
    let mut QovCox: f64 = 0.0;
    let mut LINK: f64 = 0.0;
    let mut DeltaPhi: f64 = 0.0;
    let mut dDeltaPhi_dVg: f64 = 0.0;
    let mut VgDP: f64 = 0.0;
    let mut dVgDP_dVg: f64 = 0.0;
    let mut Cox: f64 = 0.0;
    let mut Tox: f64 = 0.0;
    let mut Tcen: f64 = 0.0;
    let mut dTcen_dVg: f64 = 0.0;
    let mut dTcen_dVd: f64 = 0.0;
    let mut dTcen_dVb: f64 = 0.0;
    let mut Ccen: f64 = 0.0;
    let mut Coxeff: f64 = 0.0;
    let mut dCoxeff_dVd: f64 = 0.0;
    let mut dCoxeff_dVg: f64 = 0.0;
    let mut dCoxeff_dVb: f64 = 0.0;
    let mut Denomi: f64 = 0.0;
    let mut dDenomi_dVg: f64 = 0.0;
    let mut dDenomi_dVd: f64 = 0.0;
    let mut dDenomi_dVb: f64 = 0.0;
    let mut ueff: f64 = 0.0;
    let mut dueff_dVg: f64 = 0.0;
    let mut dueff_dVd: f64 = 0.0;
    let mut dueff_dVb: f64 = 0.0;
    let mut Esat: f64 = 0.0;
    let mut Vdsat: f64 = 0.0;
    let mut EsatL: f64 = 0.0;
    let mut dEsatL_dVg: f64 = 0.0;
    let mut dEsatL_dVd: f64 = 0.0;
    let mut dEsatL_dVb: f64 = 0.0;
    let mut dVdsat_dVg: f64 = 0.0;
    let mut dVdsat_dVb: f64 = 0.0;
    let mut dVdsat_dVd: f64 = 0.0;
    let mut Vasat: f64 = 0.0;
    let mut dAlphaz_dVg: f64 = 0.0;
    let mut dAlphaz_dVb: f64 = 0.0;
    let mut dVasat_dVg: f64 = 0.0;
    let mut dVasat_dVb: f64 = 0.0;
    let mut dVasat_dVd: f64 = 0.0;
    let mut Va: f64 = 0.0;
    let mut dVa_dVd: f64 = 0.0;
    let mut dVa_dVg: f64 = 0.0;
    let mut dVa_dVb: f64 = 0.0;
    let mut Vbseff: f64 = 0.0;
    let mut dVbseff_dVb: f64 = 0.0;
    let mut VbseffCV: f64 = 0.0;
    let mut dVbseffCV_dVb: f64 = 0.0;
    let mut VgsteffVth: f64 = 0.0;
    let mut dT11_dVg: f64 = 0.0;
    let mut Arg1: f64 = 0.0;
    let mut One_Third_CoxWL: f64 = 0.0;
    let mut Two_Third_CoxWL: f64 = 0.0;
    let mut Alphaz: f64 = 0.0;
    let mut CoxWL: f64 = 0.0;
    let mut T0: f64;
    let mut dT0_dVg: f64 = 0.0;
    let mut dT0_dVd: f64 = 0.0;
    let mut dT0_dVb: f64;
    let mut T1: f64;
    let mut dT1_dVg: f64 = 0.0;
    let mut dT1_dVd: f64 = 0.0;
    let mut dT1_dVb: f64;
    let mut T2: f64;
    let mut dT2_dVg: f64 = 0.0;
    let mut dT2_dVd: f64 = 0.0;
    let mut dT2_dVb: f64;
    let mut T3: f64;
    let mut dT3_dVg: f64 = 0.0;
    let mut dT3_dVd: f64 = 0.0;
    let mut dT3_dVb: f64;
    let mut T4: f64;
    let mut dT4_dVd: f64 = 0.0;
    let mut T5: f64;
    let mut dT5_dVg: f64 = 0.0;
    let mut dT5_dVd: f64 = 0.0;
    let mut dT5_dVb: f64;
    let mut T6: f64;
    let mut dT6_dVg: f64 = 0.0;
    let mut dT6_dVd: f64 = 0.0;
    let mut dT6_dVb: f64;
    let mut T7: f64 = 0.0;
    let mut dT7_dVg: f64 = 0.0;
    let mut dT7_dVd: f64 = 0.0;
    let mut dT7_dVb: f64 = 0.0;
    let mut T8: f64 = 0.0;
    let mut dT8_dVg: f64 = 0.0;
    let mut dT8_dVd: f64 = 0.0;
    let mut dT8_dVb: f64 = 0.0;
    let mut T9: f64;
    let mut dT9_dVg: f64 = 0.0;
    let mut dT9_dVd: f64 = 0.0;
    let mut dT9_dVb: f64 = 0.0;
    let mut T10: f64;
    let mut dT10_dVg: f64 = 0.0;
    let mut dT10_dVb: f64 = 0.0;
    let mut dT10_dVd: f64 = 0.0;
    let mut T11: f64 = 0.0;
    let mut T12: f64 = 0.0;
    let mut T13: f64 = 0.0;
    let mut T14: f64 = 0.0;
    let mut tmp: f64 = 0.0;
    let mut Abulk: f64 = 0.0;
    let mut dAbulk_dVb: f64 = 0.0;
    let mut Abulk0: f64 = 0.0;
    let mut Abulk0_Q: f64 = 0.0;
    let mut dAbulk0_dVb: f64 = 0.0;
    let mut dAbulk0_Q_dVb: f64 = 0.0;
    let mut Cclm: f64 = 0.0;
    let mut dCclm_dVg: f64 = 0.0;
    let mut dCclm_dVd: f64 = 0.0;
    let mut dCclm_dVb: f64 = 0.0;
    let mut FP: f64 = 0.0;
    let mut dFP_dVg: f64 = 0.0;
    let mut PvagTerm: f64 = 0.0;
    let mut dPvagTerm_dVg: f64 = 0.0;
    let mut dPvagTerm_dVd: f64 = 0.0;
    let mut dPvagTerm_dVb: f64 = 0.0;
    let mut VADITS: f64 = 0.0;
    let mut dVADITS_dVg: f64 = 0.0;
    let mut dVADITS_dVd: f64 = 0.0;
    let mut Lpe_Vb: f64 = 0.0;
    let mut dDITS_Sft_dVb: f64 = 0.0;
    let mut dDITS_Sft_dVd: f64 = 0.0;
    let mut DITS_Sft2: f64 = 0.0;
    let mut dDITS_Sft2_dVd: f64 = 0.0;
    let mut VACLM: f64 = 0.0;
    let mut dVACLM_dVg: f64 = 0.0;
    let mut dVACLM_dVd: f64 = 0.0;
    let mut dVACLM_dVb: f64 = 0.0;
    let mut VADIBL: f64 = 0.0;
    let mut dVADIBL_dVg: f64 = 0.0;
    let mut dVADIBL_dVd: f64 = 0.0;
    let mut dVADIBL_dVb: f64 = 0.0;
    let mut Xdep: f64 = 0.0;
    let mut dXdep_dVb: f64 = 0.0;
    let mut lt1: f64 = 0.0;
    let mut dlt1_dVb: f64 = 0.0;
    let mut ltw: f64 = 0.0;
    let mut dltw_dVb: f64 = 0.0;
    let mut Delt_vth: f64 = 0.0;
    let mut dDelt_vth_dVb: f64 = 0.0;
    let mut Theta0: f64 = 0.0;
    let mut dTheta0_dVb: f64 = 0.0;
    let mut TempRatio: f64 = 0.0;
    let mut tmp1: f64 = 0.0;
    let mut tmp2: f64 = 0.0;
    let mut tmp3: f64 = 0.0;
    let mut tmp4: f64 = 0.0;
    let mut DIBL_Sft: f64 = 0.0;
    let mut dDIBL_Sft_dVd: f64 = 0.0;
    let mut Lambda: f64 = 0.0;
    let mut dLambda_dVg: f64 = 0.0;
    let mut a1: f64 = 0.0;
    let mut ScalingFactor: f64 = 0.0;
    let mut Vgsteff: f64 = 0.0;
    let mut dVgsteff_dVg: f64 = 0.0;
    let mut dVgsteff_dVd: f64 = 0.0;
    let mut dVgsteff_dVb: f64 = 0.0;
    let mut Vdseff: f64 = 0.0;
    let mut dVdseff_dVg: f64 = 0.0;
    let mut dVdseff_dVd: f64 = 0.0;
    let mut dVdseff_dVb: f64 = 0.0;
    let mut VdseffCV: f64 = 0.0;
    let mut dVdseffCV_dVg: f64 = 0.0;
    let mut dVdseffCV_dVd: f64 = 0.0;
    let mut dVdseffCV_dVb: f64 = 0.0;
    let mut diffVds: f64 = 0.0;
    let mut dAbulk_dVg: f64 = 0.0;
    let mut beta: f64 = 0.0;
    let mut dbeta_dVg: f64 = 0.0;
    let mut dbeta_dVd: f64 = 0.0;
    let mut dbeta_dVb: f64 = 0.0;
    let mut gche: f64 = 0.0;
    let mut dgche_dVg: f64 = 0.0;
    let mut dgche_dVd: f64 = 0.0;
    let mut dgche_dVb: f64 = 0.0;
    let mut fgche1: f64 = 0.0;
    let mut dfgche1_dVg: f64 = 0.0;
    let mut dfgche1_dVd: f64 = 0.0;
    let mut dfgche1_dVb: f64 = 0.0;
    let mut fgche2: f64 = 0.0;
    let mut dfgche2_dVg: f64 = 0.0;
    let mut dfgche2_dVd: f64 = 0.0;
    let mut dfgche2_dVb: f64 = 0.0;
    let mut Idl: f64 = 0.0;
    let mut dIdl_dVg: f64 = 0.0;
    let mut dIdl_dVd: f64 = 0.0;
    let mut dIdl_dVb: f64 = 0.0;
    let mut Idsa: f64 = 0.0;
    let mut dIdsa_dVg: f64 = 0.0;
    let mut dIdsa_dVd: f64 = 0.0;
    let mut dIdsa_dVb: f64 = 0.0;
    let mut Ids: f64 = 0.0;
    let mut Gm: f64 = 0.0;
    let mut Gds: f64 = 0.0;
    let mut Gmb: f64 = 0.0;
    let mut Isub: f64 = 0.0;
    let mut Gbd: f64 = 0.0;
    let mut Gbg: f64 = 0.0;
    let mut Gbb: f64 = 0.0;
    let mut VASCBE: f64 = 0.0;
    let mut dVASCBE_dVg: f64 = 0.0;
    let mut dVASCBE_dVd: f64 = 0.0;
    let mut dVASCBE_dVb: f64 = 0.0;
    let mut CoxeffWovL: f64 = 0.0;
    let mut Rds: f64 = 0.0;
    let mut dRds_dVg: f64 = 0.0;
    let mut dRds_dVb: f64 = 0.0;
    let mut WVCox: f64 = 0.0;
    let mut WVCoxRds: f64 = 0.0;
    let mut Vgst2Vtm: f64 = 0.0;
    let mut VdsatCV: f64 = 0.0;
    let mut Leff: f64 = 0.0;
    let mut Weff: f64 = 0.0;
    let mut dWeff_dVg: f64 = 0.0;
    let mut dWeff_dVb: f64 = 0.0;
    let mut AbulkCV: f64 = 0.0;
    let mut dAbulkCV_dVb: f64 = 0.0;
    let mut qcheq: f64 = 0.0;
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
    let mut gbspsp: f64 = 0.0;
    let mut gbbdp: f64 = 0.0;
    let mut gbbsp: f64 = 0.0;
    let mut gbspg: f64 = 0.0;
    let mut gbspb: f64 = 0.0;
    let mut gbspdp: f64 = 0.0;
    let mut gbdpdp: f64 = 0.0;
    let mut gbdpg: f64 = 0.0;
    let mut gbdpb: f64 = 0.0;
    let mut gbdpsp: f64 = 0.0;
    let mut qgdo: f64 = 0.0;
    let mut qgso: f64 = 0.0;
    let mut cgdo: f64 = 0.0;
    let mut cgso: f64 = 0.0;
    let mut Cgg: f64 = 0.0;
    let mut Cgd: f64 = 0.0;
    let mut Cgb: f64 = 0.0;
    let mut Cdg: f64 = 0.0;
    let mut Cdd: f64 = 0.0;
    let mut Cds: f64 = 0.0;
    let mut Csg: f64 = 0.0;
    let mut Csd: f64 = 0.0;
    let mut Css: f64 = 0.0;
    let mut Csb: f64 = 0.0;
    let mut Cbg: f64 = 0.0;
    let mut Cbd: f64 = 0.0;
    let mut Cbb: f64 = 0.0;
    let mut Cgg1: f64 = 0.0;
    let mut Cgb1: f64 = 0.0;
    let mut Cgd1: f64 = 0.0;
    let mut Cbg1: f64 = 0.0;
    let mut Cbb1: f64 = 0.0;
    let mut Cbd1: f64 = 0.0;
    let mut Qac0: f64 = 0.0;
    let mut Qsub0: f64 = 0.0;
    let mut dQac0_dVg: f64 = 0.0;
    let mut dQac0_dVb: f64 = 0.0;
    let mut dQsub0_dVg: f64 = 0.0;
    let mut dQsub0_dVd: f64 = 0.0;
    let mut dQsub0_dVb: f64 = 0.0;
    let mut ggidld: f64 = 0.0;
    let mut ggidlg: f64 = 0.0;
    let mut ggidlb: f64 = 0.0;
    let mut ggislg: f64 = 0.0;
    let mut ggislb: f64 = 0.0;
    let mut ggisls: f64 = 0.0;
    let mut Igisl: f64 = 0.0;
    let mut Ggislg: f64 = 0.0;
    let mut Ggislb: f64 = 0.0;
    let mut Ggisls: f64 = 0.0;
    let mut vs: f64 = 0.0;
    let mut Fsevl: f64 = 0.0;
    let mut dvs_dVg: f64 = 0.0;
    let mut dvs_dVd: f64 = 0.0;
    let mut dvs_dVb: f64 = 0.0;
    let mut dFsevl_dVg: f64 = 0.0;
    let mut dFsevl_dVd: f64 = 0.0;
    let mut dFsevl_dVb: f64 = 0.0;
    let mut vgdx: f64 = 0.0;
    let mut vgsx: f64 = 0.0;
    let mut epssub: f64 = 0.0;
    let mut toxe: f64 = 0.0;
    let mut epsrox: f64 = 0.0;
    let mut m: f64 = 0.0;
    let mut mult_i: f64 = 0.0;
    let mut mult_q: f64 = 0.0;
    ScalingFactor = 1.0e-9;
    let bias::LoadBias {
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
    } = bias::select(model, here, ckt);
    let junction_bias = junction::JunctionBias {
        source: vbs_jct,
        drain: vbd_jct,
    };
    junction::current(model, here, pParam, ckt, junction_bias);
    if vds >= 0.0 {
        here.mode = 1;
        Vds = vds;
        Vgs = vgs;
        Vbs = vbs;
        Vdb = vds - vbs;
    } else {
        here.mode = -1;
        Vds = -vds;
        Vgs = vgd;
        Vbs = vbd;
        Vdb = -vbs;
    }
    if model.mtrlMod != 0 {
        epsrox = 3.9;
        toxe = model.eot;
        epssub = EPS0 * model.epsrsub;
    } else {
        epsrox = model.epsrox;
        toxe = model.toxe;
        epssub = EPSSI;
    }
    T0 = Vbs - here.vbsc - 0.001;
    T1 = sqrt(T0 * T0 - 0.004 * here.vbsc);
    if T0 >= 0.0 {
        Vbseff = here.vbsc + 0.5 * (T0 + T1);
        dVbseff_dVb = 0.5 * (1.0 + T0 / T1);
    } else {
        T2 = -0.002 / (T1 - T0);
        Vbseff = here.vbsc * (1.0 + T2);
        dVbseff_dVb = T2 * here.vbsc / T1;
    }
    T9 = 0.95 * pParam.phi;
    T0 = T9 - Vbseff - 0.001;
    T1 = sqrt(T0 * T0 + 0.004 * T9);
    Vbseff = T9 - 0.5 * (T0 + T1);
    dVbseff_dVb *= 0.5 * (1.0 + T0 / T1);
    Phis = pParam.phi - Vbseff;
    dPhis_dVb = -1.0;
    sqrtPhis = sqrt(Phis);
    dsqrtPhis_dVb = -0.5 / sqrtPhis;
    Xdep = pParam.Xdep0 * sqrtPhis / pParam.sqrtPhi;
    dXdep_dVb = (pParam.Xdep0 / pParam.sqrtPhi) * dsqrtPhis_dVb;
    Leff = pParam.leff;
    Vtm = model.vtm;
    Vtm0 = model.vtm0;
    T3 = sqrt(Xdep);
    V0 = pParam.vbi - pParam.phi;
    T0 = pParam.dvt2 * Vbseff;
    if T0 >= -0.5 {
        T1 = 1.0 + T0;
        T2 = pParam.dvt2;
    } else {
        T4 = 1.0 / (3.0 + 8.0 * T0);
        T1 = (1.0 + 3.0 * T0) * T4;
        T2 = pParam.dvt2 * T4 * T4;
    }
    lt1 = model.factor1 * T3 * T1;
    dlt1_dVb = model.factor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2);
    T0 = pParam.dvt2w * Vbseff;
    if T0 >= -0.5 {
        T1 = 1.0 + T0;
        T2 = pParam.dvt2w;
    } else {
        T4 = 1.0 / (3.0 + 8.0 * T0);
        T1 = (1.0 + 3.0 * T0) * T4;
        T2 = pParam.dvt2w * T4 * T4;
    }
    ltw = model.factor1 * T3 * T1;
    dltw_dVb = model.factor1 * (0.5 / T3 * T1 * dXdep_dVb + T3 * T2);
    T0 = pParam.dvt1 * Leff / lt1;
    if T0 < EXP_THRESHOLD {
        T1 = exp(T0);
        T2 = T1 - 1.0;
        T3 = T2 * T2;
        T4 = T3 + 2.0 * T1 * MIN_EXP;
        Theta0 = T1 / T4;
        dT1_dVb = -T0 * T1 * dlt1_dVb / lt1;
        dTheta0_dVb = dT1_dVb * (T4 - 2.0 * T1 * (T2 + MIN_EXP)) / T4 / T4;
    } else {
        Theta0 = 1.0 / (MAX_EXP - 2.0);
        dTheta0_dVb = 0.0;
    }
    here.thetavth = pParam.dvt0 * Theta0;
    Delt_vth = here.thetavth * V0;
    dDelt_vth_dVb = pParam.dvt0 * dTheta0_dVb * V0;
    T0 = pParam.dvt1w * pParam.weff * Leff / ltw;
    if T0 < EXP_THRESHOLD {
        T1 = exp(T0);
        T2 = T1 - 1.0;
        T3 = T2 * T2;
        T4 = T3 + 2.0 * T1 * MIN_EXP;
        T5 = T1 / T4;
        dT1_dVb = -T0 * T1 * dltw_dVb / ltw;
        dT5_dVb = dT1_dVb * (T4 - 2.0 * T1 * (T2 + MIN_EXP)) / T4 / T4;
    } else {
        T5 = 1.0 / (MAX_EXP - 2.0);
        dT5_dVb = 0.0;
    }
    T0 = pParam.dvt0w * T5;
    T2 = T0 * V0;
    dT2_dVb = pParam.dvt0w * dT5_dVb * V0;
    TempRatio = ckt.temp() / model.tnom - 1.0;
    T0 = sqrt(1.0 + pParam.lpe0 / Leff);
    T1 = pParam.k1ox * (T0 - 1.0) * pParam.sqrtPhi
        + (pParam.kt1 + pParam.kt1l / Leff + pParam.kt2 * Vbseff) * TempRatio;
    Vth_NarrowW = toxe * pParam.phi / (pParam.weff + pParam.w0);
    T3 = here.eta0 + pParam.etab * Vbseff;
    if T3 < 1.0e-4 {
        T9 = 1.0 / (3.0 - 2.0e4 * T3);
        T3 = (2.0e-4 - T3) * T9;
        T4 = T9 * T9;
    } else {
        T4 = 1.0;
    }
    dDIBL_Sft_dVd = T3 * pParam.theta0vb0;
    DIBL_Sft = dDIBL_Sft_dVd * Vds;
    Lpe_Vb = sqrt(1.0 + pParam.lpeb / Leff);
    Vth = f64::from(model.r#type) * here.vth0
        + (pParam.k1ox * sqrtPhis - pParam.k1 * pParam.sqrtPhi) * Lpe_Vb
        - here.k2ox * Vbseff
        - Delt_vth
        - T2
        + (pParam.k3 + pParam.k3b * Vbseff) * Vth_NarrowW
        + T1
        - DIBL_Sft;
    dVth_dVb = Lpe_Vb * pParam.k1ox * dsqrtPhis_dVb - here.k2ox - dDelt_vth_dVb - dT2_dVb
        + pParam.k3b * Vth_NarrowW
        - pParam.etab * Vds * pParam.theta0vb0 * T4
        + pParam.kt2 * TempRatio;
    dVth_dVd = -dDIBL_Sft_dVd;
    tmp1 = epssub / Xdep;
    here.nstar = model.vtm / Charge_q * (model.coxe + tmp1 + pParam.cit);
    tmp2 = pParam.nfactor * tmp1;
    tmp3 = pParam.cdsc + pParam.cdscb * Vbseff + pParam.cdscd * Vds;
    tmp4 = (tmp2 + tmp3 * Theta0 + pParam.cit) / model.coxe;
    if tmp4 >= -0.5 {
        n = 1.0 + tmp4;
        dn_dVb =
            (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb + pParam.cdscb * Theta0) / model.coxe;
        dn_dVd = pParam.cdscd * Theta0 / model.coxe;
    } else {
        T0 = 1.0 / (3.0 + 8.0 * tmp4);
        n = (1.0 + 3.0 * tmp4) * T0;
        T0 *= T0;
        dn_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb + pParam.cdscb * Theta0)
            / model.coxe
            * T0;
        dn_dVd = pParam.cdscd * Theta0 / model.coxe * T0;
    }
    if pParam.dvtp0 > 0.0 {
        T0 = -pParam.dvtp1 * Vds;
        if T0 < -EXP_THRESHOLD {
            T2 = MIN_EXP;
            dT2_dVd = 0.0;
        } else {
            T2 = exp(T0);
            dT2_dVd = -pParam.dvtp1 * T2;
        }
        T3 = Leff + pParam.dvtp0 * (1.0 + T2);
        dT3_dVd = pParam.dvtp0 * dT2_dVd;
        if model.tempMod < 2 {
            T4 = Vtm * log(Leff / T3);
            dT4_dVd = -Vtm * dT3_dVd / T3;
        } else {
            T4 = model.vtm0 * log(Leff / T3);
            dT4_dVd = -model.vtm0 * dT3_dVd / T3;
        }
        dDITS_Sft_dVd = dn_dVd * T4 + n * dT4_dVd;
        dDITS_Sft_dVb = T4 * dn_dVb;
        Vth -= n * T4;
        dVth_dVd -= dDITS_Sft_dVd;
        dVth_dVb -= dDITS_Sft_dVb;
    }
    if (pParam.dvtp4 == 0.0) || (pParam.dvtp2factor == 0.0) {
        T0 = 0.0;
        DITS_Sft2 = 0.0;
    } else {
        T1 = 2.0 * pParam.dvtp4 * Vds;
        (T0, T10) = DEXP(T1);
        DITS_Sft2 = pParam.dvtp2factor * (T0 - 1.0) / (T0 + 1.0);
        dDITS_Sft2_dVd = pParam.dvtp2factor * pParam.dvtp4 * 4.0 * T10 / ((T0 + 1.0) * (T0 + 1.0));
        Vth -= DITS_Sft2;
        dVth_dVd -= dDITS_Sft2_dVd;
    }
    here.von = Vth;
    T0 = here.vfb + pParam.phi;
    if model.mtrlMod == 0 {
        T1 = EPSSI;
    } else {
        T1 = model.epsrgate * EPS0;
    }
    (vgs_eff, dvgs_eff_dvg) = BSIM4polyDepletion(T0, pParam.ngate, T1, model.coxe, vgs);
    (vgd_eff, dvgd_eff_dvg) = BSIM4polyDepletion(T0, pParam.ngate, T1, model.coxe, vgd);
    if here.mode > 0 {
        Vgs_eff = vgs_eff;
        dVgs_eff_dVg = dvgs_eff_dvg;
    } else {
        Vgs_eff = vgd_eff;
        dVgs_eff_dVg = dvgd_eff_dvg;
    }
    here.vgs_eff = vgs_eff;
    here.vgd_eff = vgd_eff;
    here.dvgs_eff_dvg = dvgs_eff_dvg;
    here.dvgd_eff_dvg = dvgd_eff_dvg;
    Vgst = Vgs_eff - Vth;
    T0 = n * Vtm;
    T1 = pParam.mstar * Vgst;
    T2 = T1 / T0;
    if T2 > EXP_THRESHOLD {
        T10 = T1;
        dT10_dVg = pParam.mstar * dVgs_eff_dVg;
        dT10_dVd = -dVth_dVd * pParam.mstar;
        dT10_dVb = -dVth_dVb * pParam.mstar;
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
        dT10_dVg = pParam.mstar * ExpVgst / (1.0 + ExpVgst);
        dT10_dVb = T3 * dn_dVb - dT10_dVg * (dVth_dVb + Vgst * dn_dVb / n);
        dT10_dVd = T3 * dn_dVd - dT10_dVg * (dVth_dVd + Vgst * dn_dVd / n);
        dT10_dVg *= dVgs_eff_dVg;
    }
    T1 = pParam.voffcbn - (1.0 - pParam.mstar) * Vgst;
    T2 = T1 / T0;
    if T2 < -EXP_THRESHOLD {
        T3 = model.coxe * MIN_EXP / pParam.cdep0;
        T9 = pParam.mstar + T3 * n;
        dT9_dVg = 0.0;
        dT9_dVd = dn_dVd * T3;
        dT9_dVb = dn_dVb * T3;
    } else if T2 > EXP_THRESHOLD {
        T3 = model.coxe * MAX_EXP / pParam.cdep0;
        T9 = pParam.mstar + T3 * n;
        dT9_dVg = 0.0;
        dT9_dVd = dn_dVd * T3;
        dT9_dVb = dn_dVb * T3;
    } else {
        ExpVgst = exp(T2);
        T3 = model.coxe / pParam.cdep0;
        T4 = T3 * ExpVgst;
        T5 = T1 * T4 / T0;
        T9 = pParam.mstar + n * T4;
        dT9_dVg = T3 * (pParam.mstar - 1.0) * ExpVgst / Vtm;
        dT9_dVb = T4 * dn_dVb - dT9_dVg * dVth_dVb - T5 * dn_dVb;
        dT9_dVd = T4 * dn_dVd - dT9_dVg * dVth_dVd - T5 * dn_dVd;
        dT9_dVg *= dVgs_eff_dVg;
    }
    Vgsteff = T10 / T9;
    here.Vgsteff = Vgsteff;
    T11 = T9 * T9;
    dVgsteff_dVg = (T9 * dT10_dVg - T10 * dT9_dVg) / T11;
    dVgsteff_dVd = (T9 * dT10_dVd - T10 * dT9_dVd) / T11;
    dVgsteff_dVb = (T9 * dT10_dVb - T10 * dT9_dVb) / T11;
    T9 = sqrtPhis - pParam.sqrtPhi;
    Weff = pParam.weff - 2.0 * (pParam.dwg * Vgsteff + pParam.dwb * T9);
    dWeff_dVg = -2.0 * pParam.dwg;
    dWeff_dVb = -2.0 * pParam.dwb * dsqrtPhis_dVb;
    if Weff < 2.0e-8 {
        T0 = 1.0 / (6.0e-8 - 2.0 * Weff);
        Weff = 2.0e-8 * (4.0e-8 - Weff) * T0;
        let correction = T0 * 4.0e-16;
        T0 *= correction;
        dWeff_dVg *= T0;
        dWeff_dVb *= T0;
    }
    if model.rdsMod == 1 {
        dRds_dVb = 0.0;
        dRds_dVg = dRds_dVb;
        Rds = dRds_dVg;
    } else {
        T0 = 1.0 + pParam.prwg * Vgsteff;
        dT0_dVg = -pParam.prwg / T0 / T0;
        T1 = pParam.prwb * T9;
        dT1_dVb = pParam.prwb * dsqrtPhis_dVb;
        T2 = 1.0 / T0 + T1;
        T3 = T2 + sqrt(T2 * T2 + 0.01);
        dT3_dVg = 1.0 + T2 / (T3 - T2);
        dT3_dVb = dT3_dVg * dT1_dVb;
        dT3_dVg *= dT0_dVg;
        T4 = pParam.rds0 * 0.5;
        Rds = pParam.rdswmin + T3 * T4;
        dRds_dVg = T4 * dT3_dVg;
        dRds_dVb = T4 * dT3_dVb;
        if Rds > 0.0 {
            here.grdsw = 1.0 / Rds * here.nf;
        } else {
            here.grdsw = 0.0;
        }
    }
    T9 = 0.5 * pParam.k1ox * Lpe_Vb / sqrtPhis;
    T1 = T9 + here.k2ox - pParam.k3b * Vth_NarrowW;
    dT1_dVb = -T9 / sqrtPhis * dsqrtPhis_dVb;
    T9 = sqrt(pParam.xj * Xdep);
    tmp1 = Leff + 2.0 * T9;
    T5 = Leff / tmp1;
    tmp2 = pParam.a0 * T5;
    tmp3 = pParam.weff + pParam.b1;
    tmp4 = pParam.b0 / tmp3;
    T2 = tmp2 + tmp4;
    dT2_dVb = -T9 / tmp1 / Xdep * dXdep_dVb;
    T6 = T5 * T5;
    T7 = T5 * T6;
    Abulk0 = 1.0 + T1 * T2;
    dAbulk0_dVb = T1 * tmp2 * dT2_dVb + T2 * dT1_dVb;
    T8 = pParam.ags * pParam.a0 * T7;
    dAbulk_dVg = -T1 * T8;
    Abulk = Abulk0 + dAbulk_dVg * Vgsteff;
    dAbulk_dVb = dAbulk0_dVb - T8 * Vgsteff * (dT1_dVb + 3.0 * T1 * dT2_dVb);
    if Abulk0 < 0.1 {
        T9 = 1.0 / (3.0 - 20.0 * Abulk0);
        Abulk0 = (0.2 - Abulk0) * T9;
        dAbulk0_dVb *= T9 * T9;
    }
    if Abulk < 0.1 {
        T9 = 1.0 / (3.0 - 20.0 * Abulk);
        Abulk = (0.2 - Abulk) * T9;
        T10 = T9 * T9;
        dAbulk_dVb *= T10;
        dAbulk_dVg *= T10;
    }
    here.Abulk = Abulk;
    T2 = pParam.keta * Vbseff;
    if T2 >= -0.9 {
        T0 = 1.0 / (1.0 + T2);
        dT0_dVb = -pParam.keta * T0 * T0;
    } else {
        T1 = 1.0 / (0.8 + T2);
        T0 = (17.0 + 20.0 * T2) * T1;
        dT0_dVb = -pParam.keta * T1 * T1;
    }
    dAbulk_dVg *= T0;
    dAbulk_dVb = dAbulk_dVb * T0 + Abulk * dT0_dVb;
    dAbulk0_Q_dVb = dAbulk0_dVb;
    dAbulk0_dVb = dAbulk0_dVb * T0 + Abulk0 * dT0_dVb;
    Abulk *= T0;
    Abulk0_Q = Abulk0;
    Abulk0 *= T0;
    if pParam.ketac == pParam.keta {
        dAbulk0_Q_dVb = dAbulk0_dVb;
        Abulk0_Q = Abulk0;
    } else {
        T2 = pParam.ketac * Vbseff;
        if T2 >= -0.9 {
            T0 = 1.0 / (1.0 + T2);
            dT0_dVb = -pParam.ketac * T0 * T0;
        } else {
            T1 = 1.0 / (0.8 + T2);
            T0 = (17.0 + 20.0 * T2) * T1;
            dT0_dVb = -pParam.ketac * T1 * T1;
        }
        dAbulk0_Q_dVb = dAbulk0_Q_dVb * T0 + Abulk0_Q * dT0_dVb;
        Abulk0_Q *= T0;
    }
    if (model.mtrlMod != 0) && model.mtrlCompatMod == 0 {
        T14 = 2.0 * f64::from(model.r#type) * (model.phig - model.easub - 0.5 * model.Eg0 + 0.45);
    } else {
        T14 = 0.0;
    }
    if model.mobMod == 0 {
        T0 = Vgsteff + Vth + Vth - T14;
        T2 = pParam.ua + pParam.uc * Vbseff;
        T3 = T0 / toxe;
        T12 = sqrt(Vth * Vth + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * Vth;
        T6 = T8 * Vth;
        T5 = T3 * (T2 + pParam.ub * T3) + T6;
        T7 = -2.0 * T6 * T9;
        T11 = T7 * Vth / T12;
        dDenomi_dVg = (T2 + 2.0 * pParam.ub * T3) / toxe;
        T13 = 2.0 * (dDenomi_dVg + T11 + T8);
        dDenomi_dVd = T13 * dVth_dVd;
        dDenomi_dVb = T13 * dVth_dVb + pParam.uc * T3;
        dDenomi_dVg += T7;
    } else if model.mobMod == 1 {
        T0 = Vgsteff + Vth + Vth - T14;
        T2 = 1.0 + pParam.uc * Vbseff;
        T3 = T0 / toxe;
        T4 = T3 * (pParam.ua + pParam.ub * T3);
        T12 = sqrt(Vth * Vth + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * Vth;
        T6 = T8 * Vth;
        T5 = T4 * T2 + T6;
        T7 = -2.0 * T6 * T9;
        T11 = T7 * Vth / T12;
        dDenomi_dVg = (pParam.ua + 2.0 * pParam.ub * T3) * T2 / toxe;
        T13 = 2.0 * (dDenomi_dVg + T11 + T8);
        dDenomi_dVd = T13 * dVth_dVd;
        dDenomi_dVb = T13 * dVth_dVb + pParam.uc * T4;
        dDenomi_dVg += T7;
    } else if model.mobMod == 2 {
        T0 = (Vgsteff + here.vtfbphi1) / toxe;
        T1 = exp(pParam.eu * log(T0));
        dT1_dVg = T1 * pParam.eu / T0 / toxe;
        T2 = pParam.ua + pParam.uc * Vbseff;
        T12 = sqrt(Vth * Vth + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * Vth;
        T6 = T8 * Vth;
        T5 = T1 * T2 + T6;
        T7 = -2.0 * T6 * T9;
        T11 = T7 * Vth / T12;
        dDenomi_dVg = T2 * dT1_dVg + T7;
        T13 = 2.0 * (T11 + T8);
        dDenomi_dVd = T13 * dVth_dVd;
        dDenomi_dVb = T13 * dVth_dVb + T1 * pParam.uc;
    } else if model.mobMod == 4 {
        T0 = Vgsteff + here.vtfbphi1 - T14;
        T2 = pParam.ua + pParam.uc * Vbseff;
        T3 = T0 / toxe;
        T12 = sqrt(here.vtfbphi1 * here.vtfbphi1 + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * here.vtfbphi1;
        T6 = T8 * here.vtfbphi1;
        T5 = T3 * (T2 + pParam.ub * T3) + T6;
        T7 = -2.0 * T6 * T9;
        dDenomi_dVg = (T2 + 2.0 * pParam.ub * T3) / toxe;
        dDenomi_dVd = 0.0;
        dDenomi_dVb = pParam.uc * T3;
        dDenomi_dVg += T7;
    } else if model.mobMod == 5 {
        T0 = Vgsteff + here.vtfbphi1 - T14;
        T2 = 1.0 + pParam.uc * Vbseff;
        T3 = T0 / toxe;
        T4 = T3 * (pParam.ua + pParam.ub * T3);
        T12 = sqrt(here.vtfbphi1 * here.vtfbphi1 + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * here.vtfbphi1;
        T6 = T8 * here.vtfbphi1;
        T5 = T4 * T2 + T6;
        T7 = -2.0 * T6 * T9;
        dDenomi_dVg = (pParam.ua + 2.0 * pParam.ub * T3) * T2 / toxe;
        dDenomi_dVd = 0.0;
        dDenomi_dVb = pParam.uc * T4;
        dDenomi_dVg += T7;
    } else if model.mobMod == 6 {
        T0 = (Vgsteff + here.vtfbphi1) / toxe;
        T1 = exp(pParam.eu * log(T0));
        dT1_dVg = T1 * pParam.eu / T0 / toxe;
        T2 = pParam.ua + pParam.uc * Vbseff;
        T12 = sqrt(here.vtfbphi1 * here.vtfbphi1 + 0.0001);
        T9 = 1.0 / (Vgsteff + 2.0 * T12);
        T10 = T9 * toxe;
        T8 = pParam.ud * T10 * T10 * here.vtfbphi1;
        T6 = T8 * here.vtfbphi1;
        T5 = T1 * T2 + T6;
        T7 = -2.0 * T6 * T9;
        dDenomi_dVg = T2 * dT1_dVg + T7;
        dDenomi_dVd = 0.0;
        dDenomi_dVb = T1 * pParam.uc;
    } else {
        T0 = (Vgsteff + here.vtfbphi1) * 1.0e-8 / toxe / 6.0;
        T1 = exp(pParam.eu * log(T0));
        dT1_dVg = T1 * pParam.eu * 1.0e-8 / T0 / toxe / 6.0;
        T2 = pParam.ua + pParam.uc * Vbseff;
        VgsteffVth = pParam.VgsteffVth;
        T10 = exp(pParam.ucs * log(0.5 + 0.5 * Vgsteff / VgsteffVth));
        T11 = pParam.ud / T10;
        dT11_dVg = -0.5 * pParam.ucs * T11 / (0.5 + 0.5 * Vgsteff / VgsteffVth) / VgsteffVth;
        dDenomi_dVg = T2 * dT1_dVg + dT11_dVg;
        dDenomi_dVd = 0.0;
        dDenomi_dVb = T1 * pParam.uc;
        T5 = T1 * T2 + T11;
    }
    if T5 >= -0.8 {
        Denomi = 1.0 + T5;
    } else {
        T9 = 1.0 / (7.0 + 10.0 * T5);
        Denomi = (0.6 + T5) * T9;
        T9 *= T9;
        dDenomi_dVg *= T9;
        dDenomi_dVd *= T9;
        dDenomi_dVb *= T9;
    }
    ueff = here.u0temp / Denomi;
    here.ueff = ueff;
    T9 = -ueff / Denomi;
    dueff_dVg = T9 * dDenomi_dVg;
    dueff_dVd = T9 * dDenomi_dVd;
    dueff_dVb = T9 * dDenomi_dVb;
    WVCox = Weff * here.vsattemp * model.coxe;
    WVCoxRds = WVCox * Rds;
    Esat = 2.0 * here.vsattemp / ueff;
    EsatL = Esat * Leff;
    here.EsatL = EsatL;
    T0 = -EsatL / ueff;
    dEsatL_dVg = T0 * dueff_dVg;
    dEsatL_dVd = T0 * dueff_dVd;
    dEsatL_dVb = T0 * dueff_dVb;
    a1 = pParam.a1;
    if a1 == 0.0 {
        Lambda = pParam.a2;
        dLambda_dVg = 0.0;
    } else if a1 > 0.0 {
        T0 = 1.0 - pParam.a2;
        T1 = T0 - pParam.a1 * Vgsteff - 0.0001;
        T2 = sqrt(T1 * T1 + 0.0004 * T0);
        Lambda = pParam.a2 + T0 - 0.5 * (T1 + T2);
        dLambda_dVg = 0.5 * pParam.a1 * (1.0 + T1 / T2);
    } else {
        T1 = pParam.a2 + pParam.a1 * Vgsteff - 0.0001;
        T2 = sqrt(T1 * T1 + 0.0004 * pParam.a2);
        Lambda = 0.5 * (T1 + T2);
        dLambda_dVg = 0.5 * pParam.a1 * (1.0 + T1 / T2);
    }
    Vgst2Vtm = Vgsteff + 2.0 * Vtm;
    if Rds > 0.0 {
        tmp2 = dRds_dVg / Rds + dWeff_dVg / Weff;
        tmp3 = dRds_dVb / Rds + dWeff_dVb / Weff;
    } else {
        tmp2 = dWeff_dVg / Weff;
        tmp3 = dWeff_dVb / Weff;
    }
    if (Rds == 0.0) && (Lambda == 1.0) {
        T0 = 1.0 / (Abulk * EsatL + Vgst2Vtm);
        tmp1 = 0.0;
        T1 = T0 * T0;
        T2 = Vgst2Vtm * T0;
        T3 = EsatL * Vgst2Vtm;
        Vdsat = T3 * T0;
        dT0_dVg = -(Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + 1.0) * T1;
        dT0_dVd = -(Abulk * dEsatL_dVd) * T1;
        dT0_dVb = -(Abulk * dEsatL_dVb + dAbulk_dVb * EsatL) * T1;
        dVdsat_dVg = T3 * dT0_dVg + T2 * dEsatL_dVg + EsatL * T0;
        dVdsat_dVd = T3 * dT0_dVd + T2 * dEsatL_dVd;
        dVdsat_dVb = T3 * dT0_dVb + T2 * dEsatL_dVb;
    } else {
        tmp1 = dLambda_dVg / (Lambda * Lambda);
        T9 = Abulk * WVCoxRds;
        T8 = Abulk * T9;
        T7 = Vgst2Vtm * T9;
        T6 = Vgst2Vtm * WVCoxRds;
        T0 = 2.0 * Abulk * (T9 - 1.0 + 1.0 / Lambda);
        dT0_dVg = 2.0 * (T8 * tmp2 - Abulk * tmp1 + (2.0 * T9 + 1.0 / Lambda - 1.0) * dAbulk_dVg);
        dT0_dVb =
            2.0 * (T8 * (2.0 / Abulk * dAbulk_dVb + tmp3) + (1.0 / Lambda - 1.0) * dAbulk_dVb);
        dT0_dVd = 0.0;
        T1 = Vgst2Vtm * (2.0 / Lambda - 1.0) + Abulk * EsatL + 3.0 * T7;
        dT1_dVg = (2.0 / Lambda - 1.0) - 2.0 * Vgst2Vtm * tmp1
            + Abulk * dEsatL_dVg
            + EsatL * dAbulk_dVg
            + 3.0 * (T9 + T7 * tmp2 + T6 * dAbulk_dVg);
        dT1_dVb = Abulk * dEsatL_dVb + EsatL * dAbulk_dVb + 3.0 * (T6 * dAbulk_dVb + T7 * tmp3);
        dT1_dVd = Abulk * dEsatL_dVd;
        T2 = Vgst2Vtm * (EsatL + 2.0 * T6);
        dT2_dVg = EsatL + Vgst2Vtm * dEsatL_dVg + T6 * (4.0 + 2.0 * Vgst2Vtm * tmp2);
        dT2_dVb = Vgst2Vtm * (dEsatL_dVb + 2.0 * T6 * tmp3);
        dT2_dVd = Vgst2Vtm * dEsatL_dVd;
        T3 = sqrt(T1 * T1 - 2.0 * T0 * T2);
        Vdsat = (T1 - T3) / T0;
        dT3_dVg = (T1 * dT1_dVg - 2.0 * (T0 * dT2_dVg + T2 * dT0_dVg)) / T3;
        dT3_dVd = (T1 * dT1_dVd - 2.0 * (T0 * dT2_dVd + T2 * dT0_dVd)) / T3;
        dT3_dVb = (T1 * dT1_dVb - 2.0 * (T0 * dT2_dVb + T2 * dT0_dVb)) / T3;
        dVdsat_dVg =
            (dT1_dVg - (T1 * dT1_dVg - dT0_dVg * T2 - T0 * dT2_dVg) / T3 - Vdsat * dT0_dVg) / T0;
        dVdsat_dVb =
            (dT1_dVb - (T1 * dT1_dVb - dT0_dVb * T2 - T0 * dT2_dVb) / T3 - Vdsat * dT0_dVb) / T0;
        dVdsat_dVd = (dT1_dVd - (T1 * dT1_dVd - T0 * dT2_dVd) / T3) / T0;
    }
    here.vdsat = Vdsat;
    T1 = Vdsat - Vds - pParam.delta;
    dT1_dVg = dVdsat_dVg;
    dT1_dVd = dVdsat_dVd - 1.0;
    dT1_dVb = dVdsat_dVb;
    T2 = sqrt(T1 * T1 + 4.0 * pParam.delta * Vdsat);
    T0 = T1 / T2;
    T9 = 2.0 * pParam.delta;
    T3 = T9 / T2;
    dT2_dVg = T0 * dT1_dVg + T3 * dVdsat_dVg;
    dT2_dVd = T0 * dT1_dVd + T3 * dVdsat_dVd;
    dT2_dVb = T0 * dT1_dVb + T3 * dVdsat_dVb;
    if T1 >= 0.0 {
        Vdseff = Vdsat - 0.5 * (T1 + T2);
        dVdseff_dVg = dVdsat_dVg - 0.5 * (dT1_dVg + dT2_dVg);
        dVdseff_dVd = dVdsat_dVd - 0.5 * (dT1_dVd + dT2_dVd);
        dVdseff_dVb = dVdsat_dVb - 0.5 * (dT1_dVb + dT2_dVb);
    } else {
        T4 = T9 / (T2 - T1);
        T5 = 1.0 - T4;
        T6 = Vdsat * T4 / (T2 - T1);
        Vdseff = Vdsat * T5;
        dVdseff_dVg = dVdsat_dVg * T5 + T6 * (dT2_dVg - dT1_dVg);
        dVdseff_dVd = dVdsat_dVd * T5 + T6 * (dT2_dVd - dT1_dVd);
        dVdseff_dVb = dVdsat_dVb * T5 + T6 * (dT2_dVb - dT1_dVb);
    }
    if Vds == 0.0 {
        Vdseff = 0.0;
        dVdseff_dVg = 0.0;
        dVdseff_dVb = 0.0;
    }
    if Vdseff > Vds {
        Vdseff = Vds;
    }
    diffVds = Vds - Vdseff;
    here.Vdseff = Vdseff;
    if (model.lambdaGiven) && (model.lambda > 0.0) {
        T1 = Leff * ueff;
        T2 = pParam.lambda / T1;
        T3 = -T2 / T1 * Leff;
        dT2_dVd = T3 * dueff_dVd;
        dT2_dVg = T3 * dueff_dVg;
        dT2_dVb = T3 * dueff_dVb;
        T5 = 1.0 / (Esat * pParam.litl);
        T4 = -T5 / EsatL;
        dT5_dVg = dEsatL_dVg * T4;
        dT5_dVd = dEsatL_dVd * T4;
        dT5_dVb = dEsatL_dVb * T4;
        T6 = 1.0 + diffVds * T5;
        dT6_dVg = dT5_dVg * diffVds - dVdseff_dVg * T5;
        dT6_dVd = dT5_dVd * diffVds + (1.0 - dVdseff_dVd) * T5;
        dT6_dVb = dT5_dVb * diffVds - dVdseff_dVb * T5;
        T7 = 2.0 / (T6 * T6 + 1.0);
        T8 = 1.0 - T7;
        T9 = T6 * T7 * T7;
        dT8_dVg = T9 * dT6_dVg;
        dT8_dVd = T9 * dT6_dVd;
        dT8_dVb = T9 * dT6_dVb;
        T10 = 1.0 + T2 * T8;
        dT10_dVg = dT2_dVg * T8 + T2 * dT8_dVg;
        dT10_dVd = dT2_dVd * T8 + T2 * dT8_dVd;
        dT10_dVb = dT2_dVb * T8 + T2 * dT8_dVb;
        if T10 == 1.0 {
            dT10_dVb = 0.0;
            dT10_dVd = dT10_dVb;
            dT10_dVg = dT10_dVd;
        }
        dEsatL_dVg *= T10;
        dEsatL_dVg += EsatL * dT10_dVg;
        dEsatL_dVd *= T10;
        dEsatL_dVd += EsatL * dT10_dVd;
        dEsatL_dVb *= T10;
        dEsatL_dVb += EsatL * dT10_dVb;
        EsatL *= T10;
        Esat = EsatL / Leff;
        here.EsatL = EsatL;
    }
    tmp4 = 1.0 - 0.5 * Abulk * Vdsat / Vgst2Vtm;
    T9 = WVCoxRds * Vgsteff;
    T8 = T9 / Vgst2Vtm;
    T0 = EsatL + Vdsat + 2.0 * T9 * tmp4;
    T7 = 2.0 * WVCoxRds * tmp4;
    dT0_dVg = dEsatL_dVg + dVdsat_dVg + T7 * (1.0 + tmp2 * Vgsteff)
        - T8 * (Abulk * dVdsat_dVg - Abulk * Vdsat / Vgst2Vtm + Vdsat * dAbulk_dVg);
    dT0_dVb = dEsatL_dVb + dVdsat_dVb + T7 * tmp3 * Vgsteff
        - T8 * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
    dT0_dVd = dEsatL_dVd + dVdsat_dVd - T8 * Abulk * dVdsat_dVd;
    T9 = WVCoxRds * Abulk;
    T1 = 2.0 / Lambda - 1.0 + T9;
    dT1_dVg = -2.0 * tmp1 + WVCoxRds * (Abulk * tmp2 + dAbulk_dVg);
    dT1_dVb = dAbulk_dVb * WVCoxRds + T9 * tmp3;
    Vasat = T0 / T1;
    dVasat_dVg = (dT0_dVg - Vasat * dT1_dVg) / T1;
    dVasat_dVb = (dT0_dVb - Vasat * dT1_dVb) / T1;
    dVasat_dVd = dT0_dVd / T1;
    tmp1 = here.vtfbphi2;
    tmp2 = 2.0e8 * here.toxp;
    dT0_dVg = 1.0 / tmp2;
    T0 = (Vgsteff + tmp1) * dT0_dVg;
    tmp3 = exp(model.bdos * 0.7 * log(T0));
    T1 = 1.0 + tmp3;
    T2 = model.bdos * 0.7 * tmp3 / T0;
    Tcen = model.ados * 1.9e-9 / T1;
    dTcen_dVg = -Tcen * T2 * dT0_dVg / T1;
    Coxeff = epssub * here.coxp / (epssub + here.coxp * Tcen);
    here.Coxeff = Coxeff;
    dCoxeff_dVg = -Coxeff * Coxeff * dTcen_dVg / epssub;
    CoxeffWovL = Coxeff * Weff / Leff;
    beta = ueff * CoxeffWovL;
    T3 = ueff / Leff;
    dbeta_dVg = CoxeffWovL * dueff_dVg + T3 * (Weff * dCoxeff_dVg + Coxeff * dWeff_dVg);
    dbeta_dVd = CoxeffWovL * dueff_dVd;
    dbeta_dVb = CoxeffWovL * dueff_dVb + T3 * Coxeff * dWeff_dVb;
    here.AbovVgst2Vtm = Abulk / Vgst2Vtm;
    T0 = 1.0 - 0.5 * Vdseff * here.AbovVgst2Vtm;
    dT0_dVg =
        -0.5 * (Abulk * dVdseff_dVg - Abulk * Vdseff / Vgst2Vtm + Vdseff * dAbulk_dVg) / Vgst2Vtm;
    dT0_dVd = -0.5 * Abulk * dVdseff_dVd / Vgst2Vtm;
    dT0_dVb = -0.5 * (Abulk * dVdseff_dVb + dAbulk_dVb * Vdseff) / Vgst2Vtm;
    fgche1 = Vgsteff * T0;
    dfgche1_dVg = Vgsteff * dT0_dVg + T0;
    dfgche1_dVd = Vgsteff * dT0_dVd;
    dfgche1_dVb = Vgsteff * dT0_dVb;
    T9 = Vdseff / EsatL;
    fgche2 = 1.0 + T9;
    dfgche2_dVg = (dVdseff_dVg - T9 * dEsatL_dVg) / EsatL;
    dfgche2_dVd = (dVdseff_dVd - T9 * dEsatL_dVd) / EsatL;
    dfgche2_dVb = (dVdseff_dVb - T9 * dEsatL_dVb) / EsatL;
    gche = beta * fgche1 / fgche2;
    dgche_dVg = (beta * dfgche1_dVg + fgche1 * dbeta_dVg - gche * dfgche2_dVg) / fgche2;
    dgche_dVd = (beta * dfgche1_dVd + fgche1 * dbeta_dVd - gche * dfgche2_dVd) / fgche2;
    dgche_dVb = (beta * dfgche1_dVb + fgche1 * dbeta_dVb - gche * dfgche2_dVb) / fgche2;
    T0 = 1.0 + gche * Rds;
    Idl = gche / T0;
    T1 = (1.0 - Idl * Rds) / T0;
    T2 = Idl * Idl;
    dIdl_dVg = T1 * dgche_dVg - T2 * dRds_dVg;
    dIdl_dVd = T1 * dgche_dVd;
    dIdl_dVb = T1 * dgche_dVb - T2 * dRds_dVb;
    if pParam.fprout <= 0.0 {
        FP = 1.0;
        dFP_dVg = 0.0;
    } else {
        T9 = pParam.fprout * sqrt(Leff) / Vgst2Vtm;
        FP = 1.0 / (1.0 + T9);
        dFP_dVg = FP * FP * T9 / Vgst2Vtm;
    }
    T8 = pParam.pvag / EsatL;
    T9 = T8 * Vgsteff;
    if T9 > -0.9 {
        PvagTerm = 1.0 + T9;
        dPvagTerm_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL);
        dPvagTerm_dVb = -T9 * dEsatL_dVb / EsatL;
        dPvagTerm_dVd = -T9 * dEsatL_dVd / EsatL;
    } else {
        T4 = 1.0 / (17.0 + 20.0 * T9);
        PvagTerm = (0.8 + T9) * T4;
        T4 *= T4;
        dPvagTerm_dVg = T8 * (1.0 - Vgsteff * dEsatL_dVg / EsatL) * T4;
        T9 *= T4 / EsatL;
        dPvagTerm_dVb = -T9 * dEsatL_dVb;
        dPvagTerm_dVd = -T9 * dEsatL_dVd;
    }
    if (pParam.pclm > MIN_EXP) && (diffVds > 1.0e-10) {
        T0 = 1.0 + Rds * Idl;
        dT0_dVg = dRds_dVg * Idl + Rds * dIdl_dVg;
        dT0_dVd = Rds * dIdl_dVd;
        dT0_dVb = dRds_dVb * Idl + Rds * dIdl_dVb;
        T2 = Vdsat / Esat;
        T1 = Leff + T2;
        dT1_dVg = (dVdsat_dVg - T2 * dEsatL_dVg / Leff) / Esat;
        dT1_dVd = (dVdsat_dVd - T2 * dEsatL_dVd / Leff) / Esat;
        dT1_dVb = (dVdsat_dVb - T2 * dEsatL_dVb / Leff) / Esat;
        Cclm = FP * PvagTerm * T0 * T1 / (pParam.pclm * pParam.litl);
        dCclm_dVg = Cclm * (dFP_dVg / FP + dPvagTerm_dVg / PvagTerm + dT0_dVg / T0 + dT1_dVg / T1);
        dCclm_dVb = Cclm * (dPvagTerm_dVb / PvagTerm + dT0_dVb / T0 + dT1_dVb / T1);
        dCclm_dVd = Cclm * (dPvagTerm_dVd / PvagTerm + dT0_dVd / T0 + dT1_dVd / T1);
        VACLM = Cclm * diffVds;
        dVACLM_dVg = dCclm_dVg * diffVds - dVdseff_dVg * Cclm;
        dVACLM_dVb = dCclm_dVb * diffVds - dVdseff_dVb * Cclm;
        dVACLM_dVd = dCclm_dVd * diffVds + (1.0 - dVdseff_dVd) * Cclm;
    } else {
        Cclm = MAX_EXP;
        VACLM = Cclm;
        dVACLM_dVb = 0.0;
        dVACLM_dVg = dVACLM_dVb;
        dVACLM_dVd = dVACLM_dVg;
        dCclm_dVb = 0.0;
        dCclm_dVg = dCclm_dVb;
        dCclm_dVd = dCclm_dVg;
    }
    if pParam.thetaRout > MIN_EXP {
        T8 = Abulk * Vdsat;
        T0 = Vgst2Vtm * T8;
        dT0_dVg = Vgst2Vtm * Abulk * dVdsat_dVg + T8 + Vgst2Vtm * Vdsat * dAbulk_dVg;
        dT0_dVb = Vgst2Vtm * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
        dT0_dVd = Vgst2Vtm * Abulk * dVdsat_dVd;
        T1 = Vgst2Vtm + T8;
        dT1_dVg = 1.0 + Abulk * dVdsat_dVg + Vdsat * dAbulk_dVg;
        dT1_dVb = Abulk * dVdsat_dVb + dAbulk_dVb * Vdsat;
        dT1_dVd = Abulk * dVdsat_dVd;
        T9 = T1 * T1;
        T2 = pParam.thetaRout;
        VADIBL = (Vgst2Vtm - T0 / T1) / T2;
        dVADIBL_dVg = (1.0 - dT0_dVg / T1 + T0 * dT1_dVg / T9) / T2;
        dVADIBL_dVb = (-dT0_dVb / T1 + T0 * dT1_dVb / T9) / T2;
        dVADIBL_dVd = (-dT0_dVd / T1 + T0 * dT1_dVd / T9) / T2;
        T7 = pParam.pdiblb * Vbseff;
        if T7 >= -0.9 {
            T3 = 1.0 / (1.0 + T7);
            VADIBL *= T3;
            dVADIBL_dVg *= T3;
            dVADIBL_dVb = (dVADIBL_dVb - VADIBL * pParam.pdiblb) * T3;
            dVADIBL_dVd *= T3;
        } else {
            T4 = 1.0 / (0.8 + T7);
            T3 = (17.0 + 20.0 * T7) * T4;
            dVADIBL_dVg *= T3;
            dVADIBL_dVb = dVADIBL_dVb * T3 - VADIBL * pParam.pdiblb * T4 * T4;
            dVADIBL_dVd *= T3;
            VADIBL *= T3;
        }
        dVADIBL_dVg = dVADIBL_dVg * PvagTerm + VADIBL * dPvagTerm_dVg;
        dVADIBL_dVb = dVADIBL_dVb * PvagTerm + VADIBL * dPvagTerm_dVb;
        dVADIBL_dVd = dVADIBL_dVd * PvagTerm + VADIBL * dPvagTerm_dVd;
        VADIBL *= PvagTerm;
    } else {
        VADIBL = MAX_EXP;
        dVADIBL_dVb = 0.0;
        dVADIBL_dVg = dVADIBL_dVb;
        dVADIBL_dVd = dVADIBL_dVg;
    }
    Va = Vasat + VACLM;
    dVa_dVg = dVasat_dVg + dVACLM_dVg;
    dVa_dVb = dVasat_dVb + dVACLM_dVb;
    dVa_dVd = dVasat_dVd + dVACLM_dVd;
    T0 = pParam.pditsd * Vds;
    if T0 > EXP_THRESHOLD {
        T1 = MAX_EXP;
        dT1_dVd = 0.0;
    } else {
        T1 = exp(T0);
        dT1_dVd = T1 * pParam.pditsd;
    }
    if pParam.pdits > MIN_EXP {
        T2 = 1.0 + model.pditsl * Leff;
        VADITS = (1.0 + T2 * T1) / pParam.pdits;
        dVADITS_dVg = VADITS * dFP_dVg;
        dVADITS_dVd = FP * T2 * dT1_dVd / pParam.pdits;
        VADITS *= FP;
    } else {
        VADITS = MAX_EXP;
        dVADITS_dVd = 0.0;
        dVADITS_dVg = dVADITS_dVd;
    }
    if (pParam.pscbe2 > 0.0) && (pParam.pscbe1 >= 0.0) {
        if diffVds > pParam.pscbe1 * pParam.litl / EXP_THRESHOLD {
            T0 = pParam.pscbe1 * pParam.litl / diffVds;
            VASCBE = Leff * exp(T0) / pParam.pscbe2;
            T1 = T0 * VASCBE / diffVds;
            dVASCBE_dVg = T1 * dVdseff_dVg;
            dVASCBE_dVd = -T1 * (1.0 - dVdseff_dVd);
            dVASCBE_dVb = T1 * dVdseff_dVb;
        } else {
            VASCBE = MAX_EXP * Leff / pParam.pscbe2;
            dVASCBE_dVb = 0.0;
            dVASCBE_dVd = dVASCBE_dVb;
            dVASCBE_dVg = dVASCBE_dVd;
        }
    } else {
        VASCBE = MAX_EXP;
        dVASCBE_dVb = 0.0;
        dVASCBE_dVd = dVASCBE_dVb;
        dVASCBE_dVg = dVASCBE_dVd;
    }
    T9 = diffVds / VADIBL;
    T0 = 1.0 + T9;
    Idsa = Idl * T0;
    dIdsa_dVg = T0 * dIdl_dVg - Idl * (dVdseff_dVg + T9 * dVADIBL_dVg) / VADIBL;
    dIdsa_dVd = T0 * dIdl_dVd + Idl * (1.0 - dVdseff_dVd - T9 * dVADIBL_dVd) / VADIBL;
    dIdsa_dVb = T0 * dIdl_dVb - Idl * (dVdseff_dVb + T9 * dVADIBL_dVb) / VADIBL;
    T9 = diffVds / VADITS;
    T0 = 1.0 + T9;
    dIdsa_dVg = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVADITS_dVg) / VADITS;
    dIdsa_dVd = T0 * dIdsa_dVd + Idsa * (1.0 - dVdseff_dVd - T9 * dVADITS_dVd) / VADITS;
    dIdsa_dVb = T0 * dIdsa_dVb - Idsa * dVdseff_dVb / VADITS;
    Idsa *= T0;
    T0 = log(Va / Vasat);
    dT0_dVg = dVa_dVg / Va - dVasat_dVg / Vasat;
    dT0_dVb = dVa_dVb / Va - dVasat_dVb / Vasat;
    dT0_dVd = dVa_dVd / Va - dVasat_dVd / Vasat;
    T1 = T0 / Cclm;
    T9 = 1.0 + T1;
    dT9_dVg = (dT0_dVg - T1 * dCclm_dVg) / Cclm;
    dT9_dVb = (dT0_dVb - T1 * dCclm_dVb) / Cclm;
    dT9_dVd = (dT0_dVd - T1 * dCclm_dVd) / Cclm;
    dIdsa_dVg = dIdsa_dVg * T9 + Idsa * dT9_dVg;
    dIdsa_dVb = dIdsa_dVb * T9 + Idsa * dT9_dVb;
    dIdsa_dVd = dIdsa_dVd * T9 + Idsa * dT9_dVd;
    Idsa *= T9;
    tmp = pParam.alpha0 + pParam.alpha1 * Leff;
    if (tmp <= 0.0) || (pParam.beta0 <= 0.0) {
        Gbg = 0.0;
        Gbb = Gbg;
        Gbd = Gbb;
        Isub = Gbd;
    } else {
        T2 = tmp / Leff;
        if diffVds > pParam.beta0 / EXP_THRESHOLD {
            T0 = -pParam.beta0 / diffVds;
            T1 = T2 * diffVds * exp(T0);
            T3 = T1 / diffVds * (T0 - 1.0);
            dT1_dVg = T3 * dVdseff_dVg;
            dT1_dVd = T3 * (dVdseff_dVd - 1.0);
            dT1_dVb = T3 * dVdseff_dVb;
        } else {
            T3 = T2 * MIN_EXP;
            T1 = T3 * diffVds;
            dT1_dVg = -T3 * dVdseff_dVg;
            dT1_dVd = T3 * (1.0 - dVdseff_dVd);
            dT1_dVb = -T3 * dVdseff_dVb;
        }
        T4 = Idsa * Vdseff;
        Isub = T1 * T4;
        Gbg = T1 * (dIdsa_dVg * Vdseff + Idsa * dVdseff_dVg) + T4 * dT1_dVg;
        Gbd = T1 * (dIdsa_dVd * Vdseff + Idsa * dVdseff_dVd) + T4 * dT1_dVd;
        Gbb = T1 * (dIdsa_dVb * Vdseff + Idsa * dVdseff_dVb) + T4 * dT1_dVb;
        Gbd += Gbg * dVgsteff_dVd;
        Gbb += Gbg * dVgsteff_dVb;
        Gbg *= dVgsteff_dVg;
        Gbb *= dVbseff_dVb;
    }
    here.csub = Isub;
    here.gbbs = Gbb;
    here.gbgs = Gbg;
    here.gbds = Gbd;
    T9 = diffVds / VASCBE;
    T0 = 1.0 + T9;
    Ids = Idsa * T0;
    Gm = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVASCBE_dVg) / VASCBE;
    Gds = T0 * dIdsa_dVd + Idsa * (1.0 - dVdseff_dVd - T9 * dVASCBE_dVd) / VASCBE;
    Gmb = T0 * dIdsa_dVb - Idsa * (dVdseff_dVb + T9 * dVASCBE_dVb) / VASCBE;
    tmp1 = Gds + Gm * dVgsteff_dVd;
    tmp2 = Gmb + Gm * dVgsteff_dVb;
    tmp3 = Gm;
    Gm = (Ids * dVdseff_dVg + Vdseff * tmp3) * dVgsteff_dVg;
    Gds = Ids * (dVdseff_dVd + dVdseff_dVg * dVgsteff_dVd) + Vdseff * tmp1;
    Gmb = (Ids * (dVdseff_dVb + dVdseff_dVg * dVgsteff_dVb) + Vdseff * tmp2) * dVbseff_dVb;
    cdrain = Ids * Vdseff;
    if (model.vtlGiven) && (model.vtl > 0.0) {
        T12 = 1.0 / Leff / CoxeffWovL;
        T11 = T12 / Vgsteff;
        T10 = -T11 / Vgsteff;
        vs = cdrain * T11;
        dvs_dVg = Gm * T11 + cdrain * T10 * dVgsteff_dVg;
        dvs_dVd = Gds * T11 + cdrain * T10 * dVgsteff_dVd;
        dvs_dVb = Gmb * T11 + cdrain * T10 * dVgsteff_dVb;
        T0 = f64::from(2 * MM);
        T1 = vs / (pParam.vtl * pParam.tfactor);
        if T1 > 0.0 {
            T2 = 1.0 + exp(T0 * log(T1));
            T3 = (T2 - 1.0) * T0 / vs;
            Fsevl = 1.0 / exp(log(T2) / T0);
            dT2_dVg = T3 * dvs_dVg;
            dT2_dVd = T3 * dvs_dVd;
            dT2_dVb = T3 * dvs_dVb;
            T4 = -1.0 / T0 * Fsevl / T2;
            dFsevl_dVg = T4 * dT2_dVg;
            dFsevl_dVd = T4 * dT2_dVd;
            dFsevl_dVb = T4 * dT2_dVb;
        } else {
            Fsevl = 1.0;
            dFsevl_dVg = 0.0;
            dFsevl_dVd = 0.0;
            dFsevl_dVb = 0.0;
        }
        Gm *= Fsevl;
        Gm += cdrain * dFsevl_dVg;
        Gmb *= Fsevl;
        Gmb += cdrain * dFsevl_dVb;
        Gds *= Fsevl;
        Gds += cdrain * dFsevl_dVd;
        cdrain *= Fsevl;
    }
    here.gds = Gds;
    here.gm = Gm;
    here.gmbs = Gmb;
    here.IdovVds = Ids;
    if here.IdovVds <= model.idovvdsc {
        here.IdovVds = model.idovvdsc;
    }
    if (here.rgateMod > 1) || (here.trnqsMod != 0) || (here.acnqsMod != 0) {
        T9 = pParam.xrcrg2 * model.vtm;
        T0 = T9 * beta;
        dT0_dVd = (dbeta_dVd + dbeta_dVg * dVgsteff_dVd) * T9;
        dT0_dVb = (dbeta_dVb + dbeta_dVg * dVgsteff_dVb) * T9;
        dT0_dVg = dbeta_dVg * T9;
        here.gcrg = pParam.xrcrg1 * (T0 + Ids);
        here.gcrgd = pParam.xrcrg1 * (dT0_dVd + tmp1);
        here.gcrgb = pParam.xrcrg1 * (dT0_dVb + tmp2) * dVbseff_dVb;
        here.gcrgg = pParam.xrcrg1 * (dT0_dVg + tmp3) * dVgsteff_dVg;
        if here.nf != 1.0 {
            here.gcrg *= here.nf;
            here.gcrgg *= here.nf;
            here.gcrgd *= here.nf;
            here.gcrgb *= here.nf;
        }
        if here.rgateMod == 2 {
            T10 = here.grgeltd * here.grgeltd;
            T11 = here.grgeltd + here.gcrg;
            here.gcrg = here.grgeltd * here.gcrg / T11;
            T12 = T10 / T11 / T11;
            here.gcrgg *= T12;
            here.gcrgd *= T12;
            here.gcrgb *= T12;
        }
        here.gcrgs = -(here.gcrgg + here.gcrgd + here.gcrgb);
    }
    if model.rdsMod != 0 {
        T0 = vgs - pParam.vfbsd;
        T1 = sqrt(T0 * T0 + 1.0e-4);
        vgs_eff = 0.5 * (T0 + T1);
        dvgs_eff_dvg = vgs_eff / T1;
        T0 = 1.0 + pParam.prwg * vgs_eff;
        dT0_dvg = -pParam.prwg / T0 / T0 * dvgs_eff_dvg;
        T1 = -pParam.prwb * vbs;
        dT1_dvb = -pParam.prwb;
        T2 = 1.0 / T0 + T1;
        T3 = T2 + sqrt(T2 * T2 + 0.01);
        dT3_dvg = T3 / (T3 - T2);
        dT3_dvb = dT3_dvg * dT1_dvb;
        dT3_dvg *= dT0_dvg;
        T4 = pParam.rs0 * 0.5;
        Rs = pParam.rswmin + T3 * T4;
        dRs_dvg = T4 * dT3_dvg;
        dRs_dvb = T4 * dT3_dvb;
        T0 = 1.0 + here.sourceConductance * Rs;
        here.gstot = here.sourceConductance / T0;
        T0 = -here.gstot * here.gstot;
        dgstot_dvd = 0.0;
        dgstot_dvg = T0 * dRs_dvg;
        dgstot_dvb = T0 * dRs_dvb;
        dgstot_dvs = -(dgstot_dvg + dgstot_dvb + dgstot_dvd);
        T0 = vgd - pParam.vfbsd;
        T1 = sqrt(T0 * T0 + 1.0e-4);
        vgd_eff = 0.5 * (T0 + T1);
        dvgd_eff_dvg = vgd_eff / T1;
        T0 = 1.0 + pParam.prwg * vgd_eff;
        dT0_dvg = -pParam.prwg / T0 / T0 * dvgd_eff_dvg;
        T1 = -pParam.prwb * vbd;
        dT1_dvb = -pParam.prwb;
        T2 = 1.0 / T0 + T1;
        T3 = T2 + sqrt(T2 * T2 + 0.01);
        dT3_dvg = T3 / (T3 - T2);
        dT3_dvb = dT3_dvg * dT1_dvb;
        dT3_dvg *= dT0_dvg;
        T4 = pParam.rd0 * 0.5;
        Rd = pParam.rdwmin + T3 * T4;
        dRd_dvg = T4 * dT3_dvg;
        dRd_dvb = T4 * dT3_dvb;
        T0 = 1.0 + here.drainConductance * Rd;
        here.gdtot = here.drainConductance / T0;
        T0 = -here.gdtot * here.gdtot;
        dgdtot_dvs = 0.0;
        dgdtot_dvg = T0 * dRd_dvg;
        dgdtot_dvb = T0 * dRd_dvb;
        dgdtot_dvd = -(dgdtot_dvg + dgdtot_dvb + dgdtot_dvs);
        here.gstotd = vses * dgstot_dvd;
        here.gstotg = vses * dgstot_dvg;
        here.gstots = vses * dgstot_dvs;
        here.gstotb = vses * dgstot_dvb;
        T2 = vdes - vds;
        here.gdtotd = T2 * dgdtot_dvd;
        here.gdtotg = T2 * dgdtot_dvg;
        here.gdtots = T2 * dgdtot_dvs;
        here.gdtotb = T2 * dgdtot_dvb;
    } else {
        here.gstotg = 0.0;
        here.gstotd = here.gstotg;
        here.gstot = here.gstotd;
        here.gstotb = 0.0;
        here.gstots = here.gstotb;
        here.gdtotg = 0.0;
        here.gdtotd = here.gdtotg;
        here.gdtot = here.gdtotd;
        here.gdtotb = 0.0;
        here.gdtots = here.gdtotb;
    }
    if model.mtrlMod == 0 {
        T0 = 3.0 * toxe;
    } else {
        T0 = model.epsrsub * toxe / epsrox;
    }
    vgs_eff = here.vgs_eff;
    dvgs_eff_dvg = here.dvgs_eff_dvg;
    vgd_eff = here.vgd_eff;
    dvgd_eff_dvg = here.dvgd_eff_dvg;
    if model.gidlMod == 0 {
        if model.mtrlMod == 0 {
            T1 = (vds - vgs_eff - pParam.egidl) / T0;
        } else {
            T1 = (vds - vgs_eff - pParam.egidl + pParam.vfbsd) / T0;
        }
        if (pParam.agidl <= 0.0)
            || (pParam.bgidl <= 0.0)
            || (T1 <= 0.0)
            || (pParam.cgidl <= 0.0)
            || (vbd > 0.0)
        {
            Ggidlb = 0.0;
            Ggidlg = Ggidlb;
            Ggidld = Ggidlg;
            Igidl = Ggidld;
        } else {
            dT1_dVd = 1.0 / T0;
            dT1_dVg = -dvgs_eff_dvg * dT1_dVd;
            T2 = pParam.bgidl / T1;
            if T2 < 100.0 {
                Igidl = pParam.agidl * pParam.weffCJ * T1 * exp(-T2);
                T3 = Igidl * (1.0 + T2) / T1;
                Ggidld = T3 * dT1_dVd;
                Ggidlg = T3 * dT1_dVg;
            } else {
                Igidl = pParam.agidl * pParam.weffCJ * 3.720_075_976e-44;
                Ggidld = Igidl * dT1_dVd;
                Ggidlg = Igidl * dT1_dVg;
                Igidl *= T1;
            }
            T4 = vbd * vbd;
            T5 = -vbd * T4;
            T6 = pParam.cgidl + T5;
            T7 = T5 / T6;
            T8 = 3.0 * pParam.cgidl * T4 / T6 / T6;
            Ggidld = Ggidld * T7 + Igidl * T8;
            Ggidlg = Ggidlg * T7;
            Ggidlb = -Igidl * T8;
            Igidl *= T7;
        }
        here.Igidl = Igidl;
        here.ggidld = Ggidld;
        here.ggidlg = Ggidlg;
        here.ggidlb = Ggidlb;
        if model.mtrlMod == 0 {
            T1 = (-vds - vgd_eff - pParam.egisl) / T0;
        } else {
            T1 = (-vds - vgd_eff - pParam.egisl + pParam.vfbsd) / T0;
        }
        if (pParam.agisl <= 0.0)
            || (pParam.bgisl <= 0.0)
            || (T1 <= 0.0)
            || (pParam.cgisl <= 0.0)
            || (vbs > 0.0)
        {
            Ggislb = 0.0;
            Ggislg = Ggislb;
            Ggisls = Ggislg;
            Igisl = Ggisls;
        } else {
            dT1_dVd = 1.0 / T0;
            dT1_dVg = -dvgd_eff_dvg * dT1_dVd;
            T2 = pParam.bgisl / T1;
            if T2 < 100.0 {
                Igisl = pParam.agisl * pParam.weffCJ * T1 * exp(-T2);
                T3 = Igisl * (1.0 + T2) / T1;
                Ggisls = T3 * dT1_dVd;
                Ggislg = T3 * dT1_dVg;
            } else {
                Igisl = pParam.agisl * pParam.weffCJ * 3.720_075_976e-44;
                Ggisls = Igisl * dT1_dVd;
                Ggislg = Igisl * dT1_dVg;
                Igisl *= T1;
            }
            T4 = vbs * vbs;
            T5 = -vbs * T4;
            T6 = pParam.cgisl + T5;
            T7 = T5 / T6;
            T8 = 3.0 * pParam.cgisl * T4 / T6 / T6;
            Ggisls = Ggisls * T7 + Igisl * T8;
            Ggislg = Ggislg * T7;
            Ggislb = -Igisl * T8;
            Igisl *= T7;
        }
        here.Igisl = Igisl;
        here.ggisls = Ggisls;
        here.ggislg = Ggislg;
        here.ggislb = Ggislb;
    } else {
        if model.mtrlMod == 0 {
            T1 = (-vds - pParam.rgisl * vgd_eff - pParam.egisl) / T0;
        } else {
            T1 = (-vds - pParam.rgisl * vgd_eff - pParam.egisl + pParam.vfbsd) / T0;
        }
        if (pParam.agisl <= 0.0) || (pParam.bgisl <= 0.0) || (T1 <= 0.0) || (pParam.cgisl < 0.0) {
            Ggislb = 0.0;
            Ggislg = Ggislb;
            Ggisls = Ggislg;
            Igisl = Ggisls;
        } else {
            dT1_dVd = 1.0 / T0;
            dT1_dVg = -pParam.rgisl * dT1_dVd * dvgd_eff_dvg;
            T2 = pParam.bgisl / T1;
            if T2 < EXPL_THRESHOLD {
                Igisl = pParam.weffCJ * pParam.agisl * T1 * exp(-T2);
                T3 = Igisl / T1 * (T2 + 1.0);
            } else {
                T3 = pParam.weffCJ * pParam.agisl * MIN_EXPL;
                Igisl = T3 * T1;
            }
            Ggisls = T3 * dT1_dVd;
            Ggislg = T3 * dT1_dVg;
            T4 = vbs - pParam.fgisl;
            if T4 > model.gidlclamp {
                T4 = model.gidlclamp;
            }
            if T4 == 0.0 {
                T5 = EXPL_THRESHOLD;
            } else {
                T5 = pParam.kgisl / T4;
            }
            if T5 < EXPL_THRESHOLD {
                T6 = exp(T5);
                Ggislb = -Igisl * T6 * T5 / T4;
            } else {
                T6 = MAX_EXPL;
                Ggislb = 0.0;
            }
            Ggisls *= T6;
            Ggislg *= T6;
            Igisl *= T6;
        }
        here.Igisl = Igisl;
        here.ggisls = Ggisls;
        here.ggislg = Ggislg;
        here.ggislb = Ggislb;
        if model.mtrlMod == 0 {
            T1 = (vds - pParam.rgidl * vgs_eff - pParam.egidl) / T0;
        } else {
            T1 = (vds - pParam.rgidl * vgs_eff - pParam.egidl + pParam.vfbsd) / T0;
        }
        if (pParam.agidl <= 0.0) || (pParam.bgidl <= 0.0) || (T1 <= 0.0) || (pParam.cgidl < 0.0) {
            Ggidlb = 0.0;
            Ggidlg = Ggidlb;
            Ggidld = Ggidlg;
            Igidl = Ggidld;
        } else {
            dT1_dVd = 1.0 / T0;
            dT1_dVg = -pParam.rgidl * dT1_dVd * dvgs_eff_dvg;
            T2 = pParam.bgidl / T1;
            if T2 < EXPL_THRESHOLD {
                Igidl = pParam.weffCJ * pParam.agidl * T1 * exp(-T2);
                T3 = Igidl / T1 * (T2 + 1.0);
            } else {
                T3 = pParam.weffCJ * pParam.agidl * MIN_EXPL;
                Igidl = T3 * T1;
            }
            Ggidld = T3 * dT1_dVd;
            Ggidlg = T3 * dT1_dVg;
            T4 = vbd - pParam.fgidl;
            if T4 > model.gidlclamp {
                T4 = model.gidlclamp;
            }
            if T4 == 0.0 {
                T5 = EXPL_THRESHOLD;
            } else {
                T5 = pParam.kgidl / T4;
            }
            if T5 < EXPL_THRESHOLD {
                T6 = exp(T5);
                Ggidlb = -Igidl * T6 * T5 / T4;
            } else {
                T6 = MAX_EXPL;
                Ggidlb = 0.0;
            }
            Ggidld *= T6;
            Ggidlg *= T6;
            Igidl *= T6;
        }
        here.Igidl = Igidl;
        here.ggidld = Ggidld;
        here.ggidlg = Ggidlg;
        here.ggidlb = Ggidlb;
    }
    if (model.igcMod != 0) || (model.igbMod != 0) {
        Vfb = here.vfbzb;
        V3 = Vfb - Vgs_eff + Vbseff - DELTA_3;
        if Vfb <= 0.0 {
            T0 = sqrt(V3 * V3 - 4.0 * DELTA_3 * Vfb);
        } else {
            T0 = sqrt(V3 * V3 + 4.0 * DELTA_3 * Vfb);
        }
        T1 = 0.5 * (1.0 + V3 / T0);
        Vfbeff = Vfb - 0.5 * (V3 + T0);
        dVfbeff_dVg = T1 * dVgs_eff_dVg;
        dVfbeff_dVb = -T1;
        Voxacc = Vfb - Vfbeff;
        dVoxacc_dVg = -dVfbeff_dVg;
        dVoxacc_dVb = -dVfbeff_dVb;
        if Voxacc < 0.0 {
            dVoxacc_dVb = 0.0;
            dVoxacc_dVg = dVoxacc_dVb;
            Voxacc = dVoxacc_dVg;
        }
        T0 = 0.5 * pParam.k1ox;
        T3 = Vgs_eff - Vfbeff - Vbseff - Vgsteff;
        if pParam.k1ox == 0.0 {
            dVoxdepinv_dVb = 0.0;
            dVoxdepinv_dVd = dVoxdepinv_dVb;
            dVoxdepinv_dVg = dVoxdepinv_dVd;
            Voxdepinv = dVoxdepinv_dVg;
        } else if T3 < 0.0 {
            Voxdepinv = -T3;
            dVoxdepinv_dVg = -dVgs_eff_dVg + dVfbeff_dVg + dVgsteff_dVg;
            dVoxdepinv_dVd = dVgsteff_dVd;
            dVoxdepinv_dVb = dVfbeff_dVb + 1.0 + dVgsteff_dVb;
        } else {
            T1 = sqrt(T0 * T0 + T3);
            T2 = T0 / T1;
            Voxdepinv = pParam.k1ox * (T1 - T0);
            dVoxdepinv_dVg = T2 * (dVgs_eff_dVg - dVfbeff_dVg - dVgsteff_dVg);
            dVoxdepinv_dVd = -T2 * dVgsteff_dVd;
            dVoxdepinv_dVb = -T2 * (dVfbeff_dVb + 1.0 + dVgsteff_dVb);
        }
        Voxdepinv += Vgsteff;
        dVoxdepinv_dVg += dVgsteff_dVg;
        dVoxdepinv_dVd += dVgsteff_dVd;
        dVoxdepinv_dVb += dVgsteff_dVb;
    }
    if model.tempMod < 2 {
        tmp = Vtm;
    } else {
        tmp = Vtm0;
    }
    if model.igcMod != 0 {
        T0 = tmp * pParam.nigc;
        if model.igcMod == 1 {
            VxNVt = (Vgs_eff - f64::from(model.r#type) * here.vth0) / T0;
            if VxNVt > EXP_THRESHOLD {
                Vaux = Vgs_eff - f64::from(model.r#type) * here.vth0;
                dVaux_dVg = dVgs_eff_dVg;
                dVaux_dVd = 0.0;
                dVaux_dVb = 0.0;
            }
        } else if model.igcMod == 2 {
            VxNVt = (Vgs_eff - here.von) / T0;
            if VxNVt > EXP_THRESHOLD {
                Vaux = Vgs_eff - here.von;
                dVaux_dVg = dVgs_eff_dVg;
                dVaux_dVd = -dVth_dVd;
                dVaux_dVb = -dVth_dVb;
            }
        }
        if VxNVt < -EXP_THRESHOLD {
            Vaux = T0 * log(1.0 + MIN_EXP);
            dVaux_dVb = 0.0;
            dVaux_dVd = dVaux_dVb;
            dVaux_dVg = dVaux_dVd;
        } else if (VxNVt >= -EXP_THRESHOLD) && (VxNVt <= EXP_THRESHOLD) {
            ExpVxNVt = exp(VxNVt);
            Vaux = T0 * log(1.0 + ExpVxNVt);
            dVaux_dVg = ExpVxNVt / (1.0 + ExpVxNVt);
            if model.igcMod == 1 {
                dVaux_dVd = 0.0;
                dVaux_dVb = 0.0;
            } else if model.igcMod == 2 {
                dVaux_dVd = -dVaux_dVg * dVth_dVd;
                dVaux_dVb = -dVaux_dVg * dVth_dVb;
            }
            dVaux_dVg *= dVgs_eff_dVg;
        }
        T2 = Vgs * Vaux;
        dT2_dVg = Vaux + Vgs * dVaux_dVg;
        dT2_dVd = Vgs * dVaux_dVd;
        dT2_dVb = Vgs * dVaux_dVb;
        T11 = pParam.Aechvb;
        T12 = pParam.Bechvb;
        T3 = pParam.aigc * pParam.cigc - pParam.bigc;
        T4 = pParam.bigc * pParam.cigc;
        T5 = T12 * (pParam.aigc + T3 * Voxdepinv - T4 * Voxdepinv * Voxdepinv);
        if T5 > EXP_THRESHOLD {
            T6 = MAX_EXP;
            dT6_dVb = 0.0;
            dT6_dVd = dT6_dVb;
            dT6_dVg = dT6_dVd;
        } else if T5 < -EXP_THRESHOLD {
            T6 = MIN_EXP;
            dT6_dVb = 0.0;
            dT6_dVd = dT6_dVb;
            dT6_dVg = dT6_dVd;
        } else {
            T6 = exp(T5);
            dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxdepinv);
            dT6_dVd = dT6_dVg * dVoxdepinv_dVd;
            dT6_dVb = dT6_dVg * dVoxdepinv_dVb;
            dT6_dVg *= dVoxdepinv_dVg;
        }
        Igc = T11 * T2 * T6;
        dIgc_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
        dIgc_dVd = T11 * (T2 * dT6_dVd + T6 * dT2_dVd);
        dIgc_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb);
        if model.pigcdGiven {
            Pigcd = pParam.pigcd;
            dPigcd_dVb = 0.0;
            dPigcd_dVd = dPigcd_dVb;
            dPigcd_dVg = dPigcd_dVd;
        } else {
            T11 = -pParam.Bechvb;
            T12 = Vgsteff + 1.0e-20;
            T13 = T11 / T12 / T12;
            T14 = -T13 / T12;
            Pigcd = T13 * (1.0 - 0.5 * Vdseff / T12);
            dPigcd_dVg = T14 * (2.0 + 0.5 * (dVdseff_dVg - 3.0 * Vdseff / T12));
            dPigcd_dVd = 0.5 * T14 * dVdseff_dVd;
            dPigcd_dVb = 0.5 * T14 * dVdseff_dVb;
        }
        T7 = -Pigcd * Vdseff;
        dT7_dVg = -Vdseff * dPigcd_dVg - Pigcd * dVdseff_dVg;
        dT7_dVd = -Vdseff * dPigcd_dVd - Pigcd * dVdseff_dVd + dT7_dVg * dVgsteff_dVd;
        dT7_dVb = -Vdseff * dPigcd_dVb - Pigcd * dVdseff_dVb + dT7_dVg * dVgsteff_dVb;
        dT7_dVg *= dVgsteff_dVg;
        T8 = T7 * T7 + 2.0e-4;
        dT8_dVg = 2.0 * T7;
        dT8_dVd = dT8_dVg * dT7_dVd;
        dT8_dVb = dT8_dVg * dT7_dVb;
        dT8_dVg *= dT7_dVg;
        if T7 > EXP_THRESHOLD {
            T9 = MAX_EXP;
            dT9_dVb = 0.0;
            dT9_dVd = dT9_dVb;
            dT9_dVg = dT9_dVd;
        } else if T7 < -EXP_THRESHOLD {
            T9 = MIN_EXP;
            dT9_dVb = 0.0;
            dT9_dVd = dT9_dVb;
            dT9_dVg = dT9_dVd;
        } else {
            T9 = exp(T7);
            dT9_dVg = T9 * dT7_dVg;
            dT9_dVd = T9 * dT7_dVd;
            dT9_dVb = T9 * dT7_dVb;
        }
        T0 = T8 * T8;
        T1 = T9 - 1.0 + 1.0e-4;
        T10 = (T1 - T7) / T8;
        dT10_dVg = (dT9_dVg - dT7_dVg - T10 * dT8_dVg) / T8;
        dT10_dVd = (dT9_dVd - dT7_dVd - T10 * dT8_dVd) / T8;
        dT10_dVb = (dT9_dVb - dT7_dVb - T10 * dT8_dVb) / T8;
        Igcs = Igc * T10;
        dIgcs_dVg = dIgc_dVg * T10 + Igc * dT10_dVg;
        dIgcs_dVd = dIgc_dVd * T10 + Igc * dT10_dVd;
        dIgcs_dVb = dIgc_dVb * T10 + Igc * dT10_dVb;
        T1 = T9 - 1.0 - 1.0e-4;
        T10 = (T7 * T9 - T1) / T8;
        dT10_dVg = (dT7_dVg * T9 + (T7 - 1.0) * dT9_dVg - T10 * dT8_dVg) / T8;
        dT10_dVd = (dT7_dVd * T9 + (T7 - 1.0) * dT9_dVd - T10 * dT8_dVd) / T8;
        dT10_dVb = (dT7_dVb * T9 + (T7 - 1.0) * dT9_dVb - T10 * dT8_dVb) / T8;
        Igcd = Igc * T10;
        dIgcd_dVg = dIgc_dVg * T10 + Igc * dT10_dVg;
        dIgcd_dVd = dIgc_dVd * T10 + Igc * dT10_dVd;
        dIgcd_dVb = dIgc_dVb * T10 + Igc * dT10_dVb;
        here.Igcs = Igcs;
        here.gIgcsg = dIgcs_dVg;
        here.gIgcsd = dIgcs_dVd;
        here.gIgcsb = dIgcs_dVb * dVbseff_dVb;
        here.Igcd = Igcd;
        here.gIgcdg = dIgcd_dVg;
        here.gIgcdd = dIgcd_dVd;
        here.gIgcdb = dIgcd_dVb * dVbseff_dVb;
        T0 = vgs - (pParam.vfbsd + pParam.vfbsdoff);
        vgs_eff = sqrt(T0 * T0 + 1.0e-4);
        dvgs_eff_dvg = T0 / vgs_eff;
        T2 = vgs * vgs_eff;
        dT2_dVg = vgs * dvgs_eff_dvg + vgs_eff;
        T11 = pParam.AechvbEdgeS;
        T12 = pParam.BechvbEdge;
        T3 = pParam.aigs * pParam.cigs - pParam.bigs;
        T4 = pParam.bigs * pParam.cigs;
        T5 = T12 * (pParam.aigs + T3 * vgs_eff - T4 * vgs_eff * vgs_eff);
        if T5 > EXP_THRESHOLD {
            T6 = MAX_EXP;
            dT6_dVg = 0.0;
        } else if T5 < -EXP_THRESHOLD {
            T6 = MIN_EXP;
            dT6_dVg = 0.0;
        } else {
            T6 = exp(T5);
            dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * vgs_eff) * dvgs_eff_dvg;
        }
        Igs = T11 * T2 * T6;
        dIgs_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
        dIgs_dVs = -dIgs_dVg;
        T0 = vgd - (pParam.vfbsd + pParam.vfbsdoff);
        vgd_eff = sqrt(T0 * T0 + 1.0e-4);
        dvgd_eff_dvg = T0 / vgd_eff;
        T2 = vgd * vgd_eff;
        dT2_dVg = vgd * dvgd_eff_dvg + vgd_eff;
        T11 = pParam.AechvbEdgeD;
        T3 = pParam.aigd * pParam.cigd - pParam.bigd;
        T4 = pParam.bigd * pParam.cigd;
        T5 = T12 * (pParam.aigd + T3 * vgd_eff - T4 * vgd_eff * vgd_eff);
        if T5 > EXP_THRESHOLD {
            T6 = MAX_EXP;
            dT6_dVg = 0.0;
        } else if T5 < -EXP_THRESHOLD {
            T6 = MIN_EXP;
            dT6_dVg = 0.0;
        } else {
            T6 = exp(T5);
            dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * vgd_eff) * dvgd_eff_dvg;
        }
        Igd = T11 * T2 * T6;
        dIgd_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
        dIgd_dVd = -dIgd_dVg;
        here.Igs = Igs;
        here.gIgsg = dIgs_dVg;
        here.gIgss = dIgs_dVs;
        here.Igd = Igd;
        here.gIgdg = dIgd_dVg;
        here.gIgdd = dIgd_dVd;
    } else {
        here.gIgcsb = 0.0;
        here.gIgcsd = here.gIgcsb;
        here.gIgcsg = here.gIgcsd;
        here.Igcs = here.gIgcsg;
        here.gIgcdb = 0.0;
        here.gIgcdd = here.gIgcdb;
        here.gIgcdg = here.gIgcdd;
        here.Igcd = here.gIgcdg;
        here.gIgss = 0.0;
        here.gIgsg = here.gIgss;
        here.Igs = here.gIgsg;
        here.gIgdd = 0.0;
        here.gIgdg = here.gIgdd;
        here.Igd = here.gIgdg;
    }
    if model.igbMod != 0 {
        T0 = tmp * pParam.nigbacc;
        T1 = -Vgs_eff + Vbseff + Vfb;
        VxNVt = T1 / T0;
        if VxNVt > EXP_THRESHOLD {
            Vaux = T1;
            dVaux_dVg = -dVgs_eff_dVg;
            dVaux_dVb = 1.0;
        } else if VxNVt < -EXP_THRESHOLD {
            Vaux = T0 * log(1.0 + MIN_EXP);
            dVaux_dVb = 0.0;
            dVaux_dVg = dVaux_dVb;
        } else {
            ExpVxNVt = exp(VxNVt);
            Vaux = T0 * log(1.0 + ExpVxNVt);
            dVaux_dVb = ExpVxNVt / (1.0 + ExpVxNVt);
            dVaux_dVg = -dVaux_dVb * dVgs_eff_dVg;
        }
        T2 = (Vgs - Vbs) * Vaux;
        dT2_dVg = Vaux + (Vgs - Vbs) * dVaux_dVg;
        dT2_dVb = -Vaux + (Vgs - Vbs) * dVaux_dVb;
        T11 = 4.97232e-7 * pParam.weff * pParam.leff * pParam.ToxRatio;
        T12 = -7.45669e11 * toxe;
        T3 = pParam.aigbacc * pParam.cigbacc - pParam.bigbacc;
        T4 = pParam.bigbacc * pParam.cigbacc;
        T5 = T12 * (pParam.aigbacc + T3 * Voxacc - T4 * Voxacc * Voxacc);
        if T5 > EXP_THRESHOLD {
            T6 = MAX_EXP;
            dT6_dVb = 0.0;
            dT6_dVg = dT6_dVb;
        } else if T5 < -EXP_THRESHOLD {
            T6 = MIN_EXP;
            dT6_dVb = 0.0;
            dT6_dVg = dT6_dVb;
        } else {
            T6 = exp(T5);
            dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxacc);
            dT6_dVb = dT6_dVg * dVoxacc_dVb;
            dT6_dVg *= dVoxacc_dVg;
        }
        Igbacc = T11 * T2 * T6;
        dIgbacc_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
        dIgbacc_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb);
        T0 = tmp * pParam.nigbinv;
        T1 = Voxdepinv - pParam.eigbinv;
        VxNVt = T1 / T0;
        if VxNVt > EXP_THRESHOLD {
            Vaux = T1;
            dVaux_dVg = dVoxdepinv_dVg;
            dVaux_dVd = dVoxdepinv_dVd;
            dVaux_dVb = dVoxdepinv_dVb;
        } else if VxNVt < -EXP_THRESHOLD {
            Vaux = T0 * log(1.0 + MIN_EXP);
            dVaux_dVb = 0.0;
            dVaux_dVd = dVaux_dVb;
            dVaux_dVg = dVaux_dVd;
        } else {
            ExpVxNVt = exp(VxNVt);
            Vaux = T0 * log(1.0 + ExpVxNVt);
            dVaux_dVg = ExpVxNVt / (1.0 + ExpVxNVt);
            dVaux_dVd = dVaux_dVg * dVoxdepinv_dVd;
            dVaux_dVb = dVaux_dVg * dVoxdepinv_dVb;
            dVaux_dVg *= dVoxdepinv_dVg;
        }
        T2 = (Vgs - Vbs) * Vaux;
        dT2_dVg = Vaux + (Vgs - Vbs) * dVaux_dVg;
        dT2_dVd = (Vgs - Vbs) * dVaux_dVd;
        dT2_dVb = -Vaux + (Vgs - Vbs) * dVaux_dVb;
        T11 *= 0.75610;
        T12 *= 1.31724;
        T3 = pParam.aigbinv * pParam.cigbinv - pParam.bigbinv;
        T4 = pParam.bigbinv * pParam.cigbinv;
        T5 = T12 * (pParam.aigbinv + T3 * Voxdepinv - T4 * Voxdepinv * Voxdepinv);
        if T5 > EXP_THRESHOLD {
            T6 = MAX_EXP;
            dT6_dVb = 0.0;
            dT6_dVd = dT6_dVb;
            dT6_dVg = dT6_dVd;
        } else if T5 < -EXP_THRESHOLD {
            T6 = MIN_EXP;
            dT6_dVb = 0.0;
            dT6_dVd = dT6_dVb;
            dT6_dVg = dT6_dVd;
        } else {
            T6 = exp(T5);
            dT6_dVg = T6 * T12 * (T3 - 2.0 * T4 * Voxdepinv);
            dT6_dVd = dT6_dVg * dVoxdepinv_dVd;
            dT6_dVb = dT6_dVg * dVoxdepinv_dVb;
            dT6_dVg *= dVoxdepinv_dVg;
        }
        Igbinv = T11 * T2 * T6;
        dIgbinv_dVg = T11 * (T2 * dT6_dVg + T6 * dT2_dVg);
        dIgbinv_dVd = T11 * (T2 * dT6_dVd + T6 * dT2_dVd);
        dIgbinv_dVb = T11 * (T2 * dT6_dVb + T6 * dT2_dVb);
        here.Igb = Igbinv + Igbacc;
        here.gIgbg = dIgbinv_dVg + dIgbacc_dVg;
        here.gIgbd = dIgbinv_dVd;
        here.gIgbb = (dIgbinv_dVb + dIgbacc_dVb) * dVbseff_dVb;
    } else {
        here.gIgbb = 0.0;
        here.gIgbs = here.gIgbb;
        here.gIgbd = here.gIgbs;
        here.gIgbg = here.gIgbd;
        here.Igb = here.gIgbg;
    }
    if here.nf != 1.0 {
        cdrain *= here.nf;
        here.gds *= here.nf;
        here.gm *= here.nf;
        here.gmbs *= here.nf;
        here.IdovVds *= here.nf;
        here.gbbs *= here.nf;
        here.gbgs *= here.nf;
        here.gbds *= here.nf;
        here.csub *= here.nf;
        here.Igidl *= here.nf;
        here.ggidld *= here.nf;
        here.ggidlg *= here.nf;
        here.ggidlb *= here.nf;
        here.Igisl *= here.nf;
        here.ggisls *= here.nf;
        here.ggislg *= here.nf;
        here.ggislb *= here.nf;
        here.Igcs *= here.nf;
        here.gIgcsg *= here.nf;
        here.gIgcsd *= here.nf;
        here.gIgcsb *= here.nf;
        here.Igcd *= here.nf;
        here.gIgcdg *= here.nf;
        here.gIgcdd *= here.nf;
        here.gIgcdb *= here.nf;
        here.Igs *= here.nf;
        here.gIgsg *= here.nf;
        here.gIgss *= here.nf;
        here.Igd *= here.nf;
        here.gIgdg *= here.nf;
        here.gIgdd *= here.nf;
        here.Igb *= here.nf;
        here.gIgbg *= here.nf;
        here.gIgbd *= here.nf;
        here.gIgbb *= here.nf;
    }
    here.ggidls = -(here.ggidld + here.ggidlg + here.ggidlb);
    here.ggisld = -(here.ggisls + here.ggislg + here.ggislb);
    here.gIgbs = -(here.gIgbg + here.gIgbd + here.gIgbb);
    here.gIgcss = -(here.gIgcsg + here.gIgcsd + here.gIgcsb);
    here.gIgcds = -(here.gIgcdg + here.gIgcdd + here.gIgcdb);
    here.cd = cdrain;
    if model.tnoiMod == 0 {
        Abulk = Abulk0_Q * pParam.abulkCVfactor;
        Vdsat = Vgsteff / Abulk;
        T0 = Vdsat - Vds - DELTA_4;
        T1 = sqrt(T0 * T0 + 4.0 * DELTA_4 * Vdsat);
        if T0 >= 0.0 {
            Vdseff = Vdsat - 0.5 * (T0 + T1);
        } else {
            let twice_delta = DELTA_4 + DELTA_4;
            T3 = twice_delta / (T1 - T0);
            T4 = 1.0 - T3;
            T5 = Vdsat * T3 / (T1 - T0);
            Vdseff = Vdsat * T4;
        }
        if Vds == 0.0 {
            Vdseff = 0.0;
        }
        T0 = Abulk * Vdseff;
        T1 = 12.0 * (Vgsteff - 0.5 * T0 + 1.0e-20);
        T2 = Vdseff / T1;
        T3 = T0 * T2;
        here.qinv =
            Coxeff * pParam.weffCV * here.nf * pParam.leffCV * (Vgsteff - 0.5 * T0 + Abulk * T3);
    } else if model.tnoiMod == 2 {
        here.noiGd0 = here.nf * beta * Vgsteff / (1.0 + gche * Rds);
    }
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
    let mut T0: f64;
    let mut T1: f64;
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
