//! `b4noi.c` (`BSIM4noise`, `N_CALC`/`N_DENS`): the instance's noise
//! generators at one frequency, from the values the `MODEINITSMSIG` load
//! leaves on the instance.
//!
//! The analysis owns gains, `Nintegrate` and the output columns; this file
//! reproduces what C passes to `NevalSrc`/`NevalSrc2` and the flicker factor
//! it multiplies the `N_GAIN` gain by, in C's order and arithmetic.
// Compact-model noise reductions retain their physical ordering, but use
// fused products and cancellation-free density differences where possible.
#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols keep their original distinct names for source-level audit"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "physical-model equation symbols are kept consistent with the ngspice C reference"
)]
#![expect(
    non_snake_case,
    reason = "preserves upstream C identifiers for equation-by-equation comparison"
)]
// C's member and local names.
// C's own spellings: `x = x * y` and the two-sided `if` limits (`clamp`
// differs on NaN).
#![expect(
    clippy::assign_op_pattern,
    reason = "retains the reference calculation grouping and floating-point operation order"
)]
#![expect(
    clippy::manual_clamp,
    reason = "the explicit reference branch sequence has different NaN handling"
)]

use super::fields::{Instance, Model, SizeParam};
use super::host::state;
use crate::devices::noise::{BOLTZMANN, CHARGE};

/// `N_MINLOG` (`noisedef.h`).
const N_MINLOG: f64 = 1E-38;

/// C's `MAX` macro.
fn max(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

/// One generator, on BSIM4-local nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Source {
    /// `NevalSrc(..., THERMNOISE, a, b, conductance)`.
    Thermal(i32, i32, f64),
    /// `NevalSrc(..., SHOTNOISE, a, b, current)`.
    Shot(i32, i32, f64),
    /// `NevalSrc(..., N_GAIN, a, b, 0)` then `noizDens *= factor`.
    Gain(i32, i32, f64),
    /// `noizDens = 0.0` without evaluating a source.
    Zero,
    /// `NevalSrc2(..., THERMNOISE, a, b, g1, c, d, g2, phase)`.
    Correlated {
        /// C's `node1`, `node2`.
        first: (i32, i32),
        /// C's `param1`.
        g1: f64,
        /// C's `node3`, `node4`.
        second: (i32, i32),
        /// C's `param2`.
        g2: f64,
        /// C's `phi21`.
        phase: f64,
    },
}

/// `BSIM4nNames`, in C's order (the instance total `""` follows).
pub(super) const NAMES: [&str; 14] = [
    ".rd", ".rs", ".rg", ".rbps", ".rbpd", ".rbpb", ".rbsb", ".rbdb", ".id", ".1overf", ".igs",
    ".igd", ".igb", ".corl",
];

/// `Eval1ovFNoise`: the unified flicker-noise density.
fn eval_1ov_f_noise(Vds: f64, m: &Model, h: &Instance, p: &SizeParam, freq: f64, temp: f64) -> f64 {
    let cd = h.cd.abs();
    let Leff = 2.0f64.mul_add(-m.lintnoi, p.leff);
    let Leffsq = Leff * Leff;
    let esat = 2.0 * h.vsattemp / h.ueff;
    let DelClm = if m.em <= 0.0 {
        // Flicker noise modified (JX).
        0.0
    } else {
        let T0 = (((Vds - h.Vdseff) / p.litl) + m.em) / esat;
        let DelClm = p.litl * max(T0, N_MINLOG).ln();
        // Bugfix in C.
        if DelClm < 0.0 { 0.0 } else { DelClm }
    };
    let EffFreq = freq.powf(m.ef);
    let T1 = CHARGE * CHARGE * BOLTZMANN * cd * temp * h.ueff;
    let T2 = 1.0e10 * EffFreq * h.Abulk * m.coxe * Leffsq;
    let N0 = m.coxe * h.Vgsteff / CHARGE;
    let depletion = h.AbovVgst2Vtm * h.Vdseff;
    let Nl = m.coxe * h.Vgsteff * h.AbovVgst2Vtm.mul_add(-h.Vdseff, 1.0) / CHARGE;
    // N0-Nl rounds to zero when the channel depletion is below one ULP
    // of 1-depletion. Recover the physical difference from its *factor*,
    // while preserving the independent difference for large depletions.
    let density_drop = if depletion.is_finite() && depletion.abs() <= 0.5 {
        N0 * depletion
    } else {
        N0 - Nl
    };
    let numerator = N0 + h.nstar;
    let denominator = Nl + h.nstar;
    let ratio = numerator / denominator;
    let log_trap_ratio = if numerator.is_finite()
        && numerator > 0.
        && denominator.is_finite()
        && denominator > 0.
        && ratio >= N_MINLOG
    {
        let relative_drop = density_drop / denominator;
        if relative_drop.is_finite() && relative_drop > -1. {
            relative_drop.ln_1p()
        } else {
            numerator.ln() - denominator.ln()
        }
    } else {
        max(ratio, N_MINLOG).ln()
    };
    let T3 = m.oxideTrapDensityA * log_trap_ratio;
    let T4 = m.oxideTrapDensityB * density_drop;
    // Factored difference of squares prevents N0² and Nl² from
    // independently rounding to the same large value (or overflowing).
    let T5 = (m.oxideTrapDensityC * 0.5) * (density_drop * (N0 + Nl));

    let T6 = BOLTZMANN * temp * cd * cd;
    let T7 = 1.0e10 * EffFreq * Leffsq * p.weff * h.nf;
    let T8 = (m.oxideTrapDensityC * Nl)
        .mul_add(Nl, m.oxideTrapDensityB.mul_add(Nl, m.oxideTrapDensityA));
    let T9 = (Nl + h.nstar) * (Nl + h.nstar);
    (T1 / T2).mul_add(T3 + T4 + T5, T6 / T7 * DelClm * T8 / T9)
}

/// `BSIM4noise`'s `N_DENS` evaluation for one instance at `freq` (Hz):
/// the 14 generators in [`NAMES`] order. `state0` is C's `CKTstates[0]` at
/// the operating point, `temp` is `CKTtemp` (K).
#[expect(
    clippy::too_many_lines,
    reason = "keeps the reference numerical routine in the original statement and rounding order"
)] // One C function, kept whole.
pub(super) fn generators(
    m: &Model,
    h: &Instance,
    p: &SizeParam,
    state0: &dyn Fn(usize) -> f64,
    freq: f64,
    temp: f64,
) -> [Source; 14] {
    let mut out = [Source::Zero; 14];
    let mult_i = h.mult_i * h.m;
    let mult_fn = h.mult_fn * h.m;
    let mut tmp = 0.0;
    let mut npart_theta = 0.0;
    let mut npart_beta = 0.0;
    let gspr;
    let gdpr;
    if m.tnoiMod == 0 {
        if m.rdsMod == 0 {
            gspr = h.sourceConductance;
            gdpr = h.drainConductance;
            // `tmp` is used below.
            tmp = if h.grdsw > 0.0 { 1.0 / h.grdsw } else { 0.0 };
        } else {
            gspr = h.gstot;
            gdpr = h.gdtot;
            tmp = 0.0;
        }
    } else if m.tnoiMod == 1 {
        let mut T5 = h.Vgsteff / h.EsatL;
        T5 *= T5;
        npart_beta = m.rnoia * (T5 * m.tnoia).mul_add(p.leff, 1.0);
        npart_theta = m.rnoib * (T5 * m.tnoib).mul_add(p.leff, 1.0);
        if npart_theta > 0.9 {
            npart_theta = 0.9;
        }
        if npart_theta > 0.9 * npart_beta {
            // 4.6.2
            npart_theta = 0.9 * npart_beta;
        }
        let (mut s, mut d) = if m.rdsMod == 0 {
            (h.sourceConductance, h.drainConductance)
        } else {
            (h.gstot, h.gdtot)
        };
        if state0(state::VDS) >= 0.0 {
            s = s * (1.0 + npart_theta * npart_theta * s / h.IdovVds);
        } else {
            d = d * (1.0 + npart_theta * npart_theta * d / h.IdovVds);
        }
        gspr = s;
        gdpr = d;
    } else {
        // tnoiMod=2 (v4.7).
        if m.rdsMod == 0 {
            gspr = h.sourceConductance;
            gdpr = h.drainConductance;
        } else {
            gspr = h.gstot;
            gdpr = h.gdtot;
        }
    }

    out[0] = Source::Thermal(h.dNodePrime, h.dNode, gdpr * mult_i);
    out[1] = Source::Thermal(h.sNodePrime, h.sNode, gspr * mult_i);

    out[2] = if h.rgateMod == 1 {
        Source::Thermal(h.gNodePrime, h.gNodeExt, h.grgeltd * mult_i)
    } else if h.rgateMod == 2 {
        let T0 = 1.0 + h.grgeltd / h.gcrg;
        let T1 = T0 * T0;
        Source::Thermal(h.gNodePrime, h.gNodeExt, h.grgeltd * mult_i / T1)
    } else if h.rgateMod == 3 {
        Source::Thermal(h.gNodeMid, h.gNodeExt, h.grgeltd * mult_i)
    } else {
        Source::Zero
    };

    let mut bodymode = 5;
    if h.rbodyMod == 2 {
        if !m.rbps0Given || !m.rbpd0Given {
            bodymode = 1;
        } else if (!m.rbsbx0Given && !m.rbsby0Given) || (!m.rbdbx0Given && !m.rbdby0Given) {
            bodymode = 3;
        }
    }
    if h.rbodyMod != 0 {
        if bodymode == 5 {
            out[3] = Source::Thermal(h.bNodePrime, h.sbNode, h.grbps * mult_i);
            out[4] = Source::Thermal(h.bNodePrime, h.dbNode, h.grbpd * mult_i);
            out[5] = Source::Thermal(h.bNodePrime, h.bNode, h.grbpb * mult_i);
            out[6] = Source::Thermal(h.bNode, h.sbNode, h.grbsb * mult_i);
            out[7] = Source::Thermal(h.bNode, h.dbNode, h.grbdb * mult_i);
        }
        if bodymode == 3 {
            out[3] = Source::Thermal(h.bNodePrime, h.sbNode, h.grbps * mult_i);
            out[4] = Source::Thermal(h.bNodePrime, h.dbNode, h.grbpd * mult_i);
            out[5] = Source::Thermal(h.bNodePrime, h.bNode, h.grbpb * mult_i);
        }
        if bodymode == 1 {
            out[5] = Source::Thermal(h.bNodePrime, h.bNode, h.grbpb * mult_i);
        }
    }

    let mut GammaGd0 = 0.0;
    let mut sigrat = 0.0;
    let mut ctnoi = 0.0;
    if m.tnoiMod == 2 {
        let depletion = h.Vdseff * h.AbovVgst2Vtm;
        let eta = h.Vdseff.mul_add(-h.AbovVgst2Vtm, 1.0);
        // 1-(1-depletion) loses sub-ULP depletion and changes correlated
        // channel noise. Preserve the direct physical product when finite.
        let T0 = if depletion.is_finite() {
            depletion
        } else {
            1.0 - eta
        };
        let T1 = 1.0 + eta;
        let T2 = T1 + 2.0 * h.Abulk * m.vtm / h.Vgsteff;
        let Leff = p.leff;
        let Lvsat = Leff * (1.0 + h.Vdseff / h.EsatL);
        let T6 = Leff / Lvsat;
        let mut gamma = T6 * 0.5f64.mul_add(T1, T0 * T0 / (6.0 * T2));
        let T3 = T2 * T2;
        let T4 = T0 * T0;
        let T5 = T3 * T3;
        let mut delta = (T1 / T3 - 5.0f64.mul_add(T1, T2) * T4 / (15.0 * T5)
            + T4 * T4 / (9.0 * T5 * T2))
            / (6.0 * T6 * T6 * T6);
        let T7 = T0 / T2;
        let epsilon = (T7 - T7 * T7 * T7 / 3.0) / (6.0 * T6);
        let mut T8 = h.Vgsteff / h.EsatL;
        T8 *= T8;
        let npart_c = m.rnoic * (T8 * m.tnoic).mul_add(Leff, 1.0);
        if m.v48intVersion <= 480 {
            ctnoi = epsilon / (gamma * delta).sqrt() * (2.5316 * npart_c);
        } else {
            // Limits added for rnoia..tnoic in BSIM4.8.1.
            let T9 = gamma * delta;
            ctnoi = if T9 > 0.0 {
                epsilon / (gamma * delta).sqrt() * (2.5316 * npart_c)
            } else {
                1.0
            };
            if ctnoi > 1.0 {
                ctnoi = 1.0;
            }
            if ctnoi < 0.0 {
                ctnoi = 0.0;
            }
        }

        npart_beta = m.rnoia * (T8 * m.tnoia).mul_add(Leff, 1.0);
        npart_theta = m.rnoib * (T8 * m.tnoib).mul_add(Leff, 1.0);
        gamma = gamma * (3.0 * npart_beta * npart_beta);
        delta = delta * (3.75 * npart_theta * npart_theta);

        GammaGd0 = gamma * h.noiGd0;
        let C0 = h.Coxeff * p.weffCV * h.nf * p.leffCV;
        let T0 = C0 / h.noiGd0;
        // Earlier models did not guard the ratio, including nonfinite inputs.
        sigrat = if m.v48intVersion <= 480 || (gamma > 0.0 && delta > 0.0) {
            T0 * (delta / gamma).sqrt()
        } else {
            0.0
        };
    }
    match m.tnoiMod {
        0 => {
            // Both C version branches load the same `.id` source; the 4.8.0
            // branch leaves `noizDens[BSIM4CORLNOIZ]` unassigned (an
            // indeterminate stack value C adds to the total). The port uses
            // zero there.
            let T0 = h.ueff * h.qinv.abs();
            let leff_squared = p.leff * p.leff;
            let T1 = T0.mul_add(tmp, leff_squared);
            out[8] = Source::Thermal(h.dNodePrime, h.sNodePrime, (T0 / T1) * m.ntnoi * mult_i);
        }
        1 => {
            let mut T0 = h.gm + h.gmbs + h.gds;
            T0 *= T0;
            let igsquare = npart_theta * npart_theta * T0 / h.IdovVds;
            let T1 = f64::mul_add(npart_beta, h.gm + h.gmbs, h.gds);
            let T2 = T1 * T1 / h.IdovVds;
            out[8] = Source::Thermal(h.dNodePrime, h.sNodePrime, (T2 - igsquare) * mult_i);
        }
        2 => {
            let T2 = GammaGd0;
            let T3 = ctnoi * ctnoi;
            let T4 = 1.0 - T3;
            out[8] = Source::Thermal(h.dNodePrime, h.sNodePrime, T2 * T4 * mult_i);

            // Output noise due to two correlated noise sources.
            let omega = 2.0 * std::f64::consts::PI * freq;
            let T5 = omega * sigrat;
            let T6 = T5 * T5;
            let T7 = T6 / (1.0 + T6);
            out[13] = if h.mode >= 0 {
                Source::Correlated {
                    first: (h.dNodePrime, h.sNodePrime),
                    g1: T2 * T3 * mult_i,
                    second: (h.gNodePrime, h.sNodePrime),
                    g2: T2 * T7 * mult_i,
                    phase: 0.5 * std::f64::consts::PI,
                }
            } else {
                Source::Correlated {
                    first: (h.sNodePrime, h.dNodePrime),
                    g1: T2 * T3 * mult_i,
                    second: (h.gNodePrime, h.dNodePrime),
                    g2: T2 * T7 * mult_i,
                    phase: 0.5 * std::f64::consts::PI,
                }
            };
        }
        // C's switch has no default: `.id` keeps its previous density, which
        // the caller rejects (see `Device::noise`).
        _ => {}
    }

    let factor = match m.fnoiMod {
        0 => {
            mult_fn * m.kf * (m.af * max(h.cd.abs(), N_MINLOG).ln()).exp()
                / (freq.powf(m.ef) * p.leff * p.leff * m.coxe)
        }
        1 => {
            let mut Vds = state0(state::VDS);
            if Vds < 0.0 {
                Vds = -Vds;
            }
            let Ssi = eval_1ov_f_noise(Vds, m, h, p, freq, temp);
            let T10 = m.oxideTrapDensityA * BOLTZMANN * temp;
            let T11 = p.weff * h.nf * p.leff * freq.powf(m.ef) * 1.0e10 * h.nstar * h.nstar;
            let Swi = T10 / T11 * h.cd * h.cd;
            let T1 = Swi + Ssi;
            if T1 > 0.0 {
                mult_fn * (Ssi * Swi) / T1
            } else {
                0.0
            }
        }
        // No C case: the gain stays unmultiplied.
        _ => 1.0,
    };
    out[9] = Source::Gain(h.dNodePrime, h.sNodePrime, factor);

    if h.mode >= 0 {
        // Bugfix in C.
        out[10] = Source::Shot(h.gNodePrime, h.sNodePrime, mult_i * (h.Igs + h.Igcs));
        out[11] = Source::Shot(h.gNodePrime, h.dNodePrime, mult_i * (h.Igd + h.Igcd));
    } else {
        out[10] = Source::Shot(h.gNodePrime, h.sNodePrime, mult_i * (h.Igs + h.Igcd));
        out[11] = Source::Shot(h.gNodePrime, h.dNodePrime, mult_i * (h.Igd + h.Igcs));
    }
    out[12] = Source::Shot(h.gNodePrime, h.bNodePrime, mult_i * h.Igb);
    out
}

#[cfg(test)]
mod tests {
    use super::{BOLTZMANN, CHARGE, Instance, Model, SizeParam, eval_1ov_f_noise};

    /// In strong inversion, a representable sub-ULP depletion still changes
    /// trap density. Both N0-Nl and N0²-Nl² in the old form round to zero,
    /// incorrectly erasing both the B/C and logarithmic A flicker terms.
    #[test]
    fn flicker_trap_noise_retains_sub_ulp_charge_depletion() {
        let nominal = 4e18;
        let fraction = 2_f64.powi(-55);
        let mut model = Model {
            coxe: 1.,
            em: 0.,
            ef: 1.,
            lintnoi: 0.,
            oxideTrapDensityC: 1.,
            ..Model::default()
        };
        let device = Instance {
            Vgsteff: nominal * CHARGE,
            Vdseff: 1.,
            AbovVgst2Vtm: fraction,
            cd: 1.,
            ueff: 0.1,
            Abulk: 1.,
            nf: 1.,
            nstar: 1.,
            ..Instance::default()
        };
        let size = SizeParam {
            leff: 1e-6,
            weff: 1e-6,
            ..SizeParam::default()
        };
        let n0 = model.coxe * device.Vgsteff / CHARGE;
        let nl = model.coxe * device.Vgsteff * (1. - fraction) / CHARGE;
        assert_eq!(n0.to_bits(), nl.to_bits());
        let scaling = CHARGE * CHARGE * BOLTZMANN * device.cd * 300. * device.ueff
            / (1e10 * device.Abulk * model.coxe * size.leff * size.leff);
        let recovered = n0 * fraction;
        let wants = [
            0.5 * recovered * (n0 + nl),
            (recovered / (nl + device.nstar)).ln_1p(),
        ];
        for (case, expected_trap) in wants.into_iter().enumerate() {
            model.oxideTrapDensityA = if case == 1 { 1. } else { 0. };
            model.oxideTrapDensityC = if case == 0 { 1. } else { 0. };
            let expected = scaling * expected_trap;
            let got = eval_1ov_f_noise(1., &model, &device, &size, 1., 300.);
            assert!(
                got.is_finite() && got > 0. && (got - expected).abs() <= expected * 3e-14,
                "trap case {case}: expected {expected:.17e}, got {got:.17e}"
            );
        }
    }
}
