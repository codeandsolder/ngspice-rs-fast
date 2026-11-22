//! `.disto` nonlinearities of the Gummel-Poon BJT: `bjtdset.c` (Taylor
//! coefficients at the bias point) and `bjtdisto.c` (which kernels drive
//! which branch). C's distortion model is its own simplified Gummel-Poon,
//! reproduced here as written rather than derived from `bjtload.c`:
//!
//! * junction currents of the ideal exponentials with C's old reverse law
//!   (`gbe = -IS/vbe` below `-5 NF Vt`), `ISE`/`ISC` leakage, Early effect
//!   and high injection through `qb = q1 (1 + sqrt(1 + 4 q2)) / 2` (**no
//!   `NKF`**), evaluated at `vbe = v(b') - v(e')` and `vbc = v(b) - v(c')`
//!   (the **external** base, as `bjtdset.c` reads it);
//! * the base resistance `RB`/`RBM`/`IRB` with `bjtdset.c`'s power-series
//!   inversion (whose `vbb = rbb(ib) * ib` linearizes around `ib = 0`);
//! * the diffusion charge `TF (1 + XTF exp(vbc/(1.44 VTF)) (cbe/(cbe +
//!   ITF))^2) cbe / qb` and the B-E, B-C (split by `XCJC`) and substrate
//!   depletion charges, the substrate one graded with the model's
//!   **unadjusted** `VJS`/`MJS` and always tied to the internal collector;
//! * `M` scaling folded into the parameters, with `bjtdset.c`'s extra `AREA`
//!   factor on the B-E depletion capacitance (`BJTtBEcap` already includes
//!   it);
//! * second-order coefficients signed by the polarity, third-order ones not;
//!   the depletion coefficients are never signed.
//!
//! Without a base resistance (`RB = RBM = 0`), `bjtdisto.c` forms the
//! `B-C'` (`qbx`) control from the stale excess-phase `vbe` kernel plus
//! `vbc`; that is reproduced too (it matters only with `XCJC < 1`). With a
//! base resistance, `bjtdisto.c`'s `D_F1MF2` case reads the `vb - vb'`
//! and substrate (`vs - vc'`) kernels at `f2` without conjugating them
//! (`i1hm2z` lacks its minus sign), which the base-resistance, `B-C'` and
//! substrate terms inherit in the `f1 - f2` product
//! ([`Control::with_unconjugated_in_f1_minus_f2`]). Excess
//! phase itself (`PTF`) is rejected at instantiation, so the delayed `vbe`
//! control equals `vbe`.

// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
#![expect(
    clippy::similar_names,
    reason = "C-translated equation symbols keep their original distinct names for source-level audit"
)]
#![expect(
    clippy::float_cmp,
    reason = "C device branches compare floating-point values exactly, including nonfinite cases"
)]
#![expect(
    clippy::suboptimal_flops,
    reason = "SPICE numerical reference parity requires the original non-fused C operation order"
)]

use crate::devices::distortion::deriv::Dderivs;
use crate::devices::distortion::{
    Control, DeviceDistortion, DistortionContext, DistortionTerm, Response, Taylor,
};
use crate::primitives::{Real, SpiceError, SpiceResult};

use super::{
    Bjt, Thermal,
    physics::{base_current, early_denominator},
};

/// The `dloadfns.c` coefficients `bjtdset.c` stores from a derivative
/// structure in `p`, `q`, `r` (C's `x`, `y`, `z`/`w`): `0.5*type*d2_p2`,
/// `type*d2_pq`, `d3_p3/6.`, `0.5*d3_p2q`, `d3_pqr`.
fn taylor(d: &Dderivs, polarity: Real) -> Taylor {
    Taylor {
        xx: 0.5 * polarity * d.d2_p2,
        yy: 0.5 * polarity * d.d2_q2,
        zz: 0.5 * polarity * d.d2_r2,
        xy: polarity * d.d2_pq,
        yz: polarity * d.d2_qr,
        xz: polarity * d.d2_pr,
        xxx: d.d3_p3 / 6.,
        yyy: d.d3_q3 / 6.,
        zzz: d.d3_r3 / 6.,
        xxy: 0.5 * d.d3_p2q,
        xxz: 0.5 * d.d3_p2r,
        xyy: 0.5 * d.d3_pq2,
        yyz: 0.5 * d.d3_q2r,
        xzz: 0.5 * d.d3_pr2,
        yzz: 0.5 * d.d3_qr2,
        xyz: d.d3_pqr,
    }
}

/// C's `MAX(a, b)`.
fn c_max(a: Real, b: Real) -> Real {
    if a > b { a } else { b }
}

/// The second- and third-order coefficients of a depletion charge with zero-
/// bias capacitance `cz`, potential `pot` and grading `m`, below the forward
/// threshold (`bjtdset.c`'s `lcap*2`/`lcap*3`).
fn depletion(cz: Real, pot: Real, m: Real, v: Real) -> (Real, Real) {
    let arg = 1. - v / pot;
    let sarg = (-m * arg.ln()).exp();
    (
        0.5 * cz * m * sarg / (arg * pot),
        cz * m * (m + 1.) * sarg / (arg * arg * pot * pot * 6.),
    )
}

/// Value and first three derivatives along a single junction voltage.
/// Taylor conversion is deliberately deferred.
#[derive(Clone, Copy)]
struct JunctionDerivatives {
    current: Real,
    first: Real,
    second: Real,
    third: Real,
}

#[derive(Clone, Copy)]
struct DistortionJunction {
    main: JunctionDerivatives,
    leakage: JunctionDerivatives,
}

struct DistortionJunctions {
    be: DistortionJunction,
    bc: DistortionJunction,
}

/// The bjtdset.c policy is not bjtload.c's cubic continuation: the
/// main-junction threshold controls both branches, and gmin is added to
/// the main junction only. Currents already include multiplicity.
fn distortion_junction(
    voltage: Real,
    vtn: Real,
    saturation: Real,
    vleak: Real,
    leakage: Real,
    gmin: Real,
) -> DistortionJunction {
    if voltage > -5. * vtn {
        let ev = (voltage / vtn).exp();
        let current = saturation * (ev - 1.) + gmin * voltage;
        let first = saturation * ev / vtn + gmin;
        let second = saturation * ev / vtn / vtn;
        let third = second / vtn;
        let main = JunctionDerivatives {
            current,
            first,
            second,
            third,
        };
        let leakage = if leakage == 0. {
            JunctionDerivatives {
                current: 0.,
                first: 0.,
                second: 0.,
                third: 0.,
            }
        } else {
            let ev = (voltage / vleak).exp();
            let current = leakage * (ev - 1.);
            let first = leakage * ev / vleak;
            let second = first / vleak;
            let third = second / vleak;
            JunctionDerivatives {
                current,
                first,
                second,
                third,
            }
        };
        DistortionJunction { main, leakage }
    } else {
        let first = -saturation / voltage + gmin;
        let main = JunctionDerivatives {
            current: first * voltage,
            first,
            second: 0.,
            third: 0.,
        };
        let first = -leakage / voltage;
        let leakage = JunctionDerivatives {
            current: first * voltage,
            first,
            second: 0.,
            third: 0.,
        };
        DistortionJunction { main, leakage }
    }
}

#[derive(Clone, Copy)]
struct DistortionBias {
    vbe: Real,
    vbc: Real,
    vsc: Real,
    vbb: Real,
}

fn distortion_junctions(
    t: &Thermal,
    bias: DistortionBias,
    m: Real,
    gmin: Real,
) -> DistortionJunctions {
    let vt = t.vt;
    DistortionJunctions {
        be: distortion_junction(bias.vbe, vt * t.nf, t.is_be * m, t.ne * vt, t.ise * m, gmin),
        bc: distortion_junction(bias.vbc, vt * t.nr, t.is_bc * m, t.nc * vt, t.isc * m, gmin),
    }
}

/// Base-charge derivatives: bjtdset.c uses the sqrt-only policy,
/// independent of the fitted NKF for the ordinary load.
fn distortion_base_charge(
    t: &Thermal,
    bias: DistortionBias,
    m: Real,
    junctions: &DistortionJunctions,
) -> Dderivs {
    let DistortionBias { vbe, vbc, .. } = bias;
    let oik = t.inv_ikf / m;
    let oikr = t.inv_ikr / m;
    let d_p = Dderivs::variable(0);
    let JunctionDerivatives {
        current: cbe,
        first: gbe,
        second: gbe2,
        third: gbe3,
    } = junctions.be.main;
    let JunctionDerivatives {
        current: cbc,
        first: gbc,
        second: gbc2,
        third: gbc3,
    } = junctions.bc.main;
    // Base charge qb(p, q).
    let mut d_dummy = d_p;
    d_dummy.value = early_denominator(t, vbe, vbc);
    d_dummy.d1_p = -t.inv_var;
    d_dummy.d1_q = -t.inv_vaf;
    let d_q1 = d_dummy.inv();
    if oik == 0. && oikr == 0. {
        d_q1
    } else {
        let q2 = oik * cbe + oikr * cbc;
        let mut d_q2 = d_p;
        d_q2.value = q2;
        d_q2.d1_p = oik * gbe;
        d_q2.d1_q = oikr * gbc;
        d_q2.d2_p2 = oik * gbe2;
        d_q2.d2_q2 = oikr * gbc2;
        d_q2.d3_p3 = oik * gbe3;
        d_q2.d3_q3 = oikr * gbc3;
        let arg = c_max(0., 1. + 4. * q2);
        let d_arg = if arg == 0. {
            let mut d = d_p;
            d.d1_p = 0.;
            d
        } else {
            let mut d = d_q2.times(4.0);
            d.value += 1.;
            d
        };
        let mut d_sqarg = d_p;
        d_sqarg.value = 1.0;
        d_sqarg.d1_p = 0.0;
        if arg != 0. {
            d_sqarg = d_arg.sqrt();
        }
        let mut d_dummy = d_sqarg;
        d_dummy.value += 1.0;
        d_q1.mul(&d_dummy).times(0.5)
    }
}

/// Collector transport: r is delayed VBE, q is VBC.
fn distortion_collector_current(
    t: &Thermal,
    junctions: &DistortionJunctions,
    d_qb: &Dderivs,
) -> Dderivs {
    let d_p = Dderivs::variable(0);
    let JunctionDerivatives {
        current: cbe,
        first: gbe,
        second: gbe2,
        third: gbe3,
    } = junctions.be.main;
    let JunctionDerivatives {
        current: cbc,
        first: gbc,
        second: gbc2,
        third: gbc3,
    } = junctions.bc.main;
    let JunctionDerivatives {
        current: cbcn,
        first: gbcn,
        second: gbcn2,
        third: gbcn3,
    } = junctions.bc.leakage;
    // Collector current ic(p, q, r).
    let mut d_dummy = d_p;
    d_dummy.d1_p = 0.0;
    d_dummy.value = cbe - cbc;
    d_dummy.d1_r = gbe;
    d_dummy.d2_r2 = gbe2;
    d_dummy.d3_r3 = gbe3;
    d_dummy.d1_q = -gbc;
    d_dummy.d2_q2 = -gbc2;
    d_dummy.d3_q3 = -gbc3;
    let mut d_ic = d_dummy.div(d_qb);
    d_ic.value -= cbc / t.br + cbcn;
    d_ic.d1_q -= gbc / t.br + gbcn;
    d_ic.d2_q2 -= gbc2 / t.br + gbcn2;
    d_ic.d3_q3 -= gbc3 / t.br + gbcn3;
    d_ic
}

/// C's base-resistance power-series inversion about ib = 0.
fn distortion_base_resistance(
    t: &Thermal,
    vbb: Real,
    m: Real,
    junctions: &DistortionJunctions,
    d_qb: &Dderivs,
    name: &str,
) -> SpiceResult<Dderivs> {
    let rbpr = t.rbm / m;
    let rbpi = t.rb / m - rbpr;
    let xjrb = t.irb * m;
    let d_p = Dderivs::variable(0);
    let d_r = Dderivs::variable(2);
    Ok(if rbpr == 0.0 && rbpi == 0.0 {
        let mut d = d_p;
        d.d1_p = 0.0;
        d
    } else {
        let cb = base_current(
            t,
            junctions.be.main.current,
            junctions.be.leakage.current,
            junctions.bc.main.current,
            junctions.bc.leakage.current,
        );
        if cb == 0.0 {
            let mut d = d_r;
            if rbpr != 0.0 {
                d.d1_r = 1. / rbpr;
            }
            d
        } else if xjrb != 0.0 && rbpi != 0.0 {
            // p stands for the base current here.
            let mut d_dummy = d_p;
            d_dummy.value = c_max(cb / xjrb, 1e-9);
            d_dummy.d1_p = 1. / xjrb;
            let d_dummy = d_dummy.sqrt().times(2.4317);
            let mut d_dummy2 = d_p;
            d_dummy2.value = 1. + 14.59025 * c_max(cb / xjrb, 1e-9);
            d_dummy2.d1_p = 14.59025 / xjrb;
            let mut d_dummy2 = d_dummy2.sqrt();
            d_dummy2.value -= 1.0;
            let d_z = d_dummy2.div(&d_dummy);
            let d_tanz = d_z.tan();
            let d_dummy = d_z.times(-1.0).plus(&d_tanz);
            let d_dummy2 = d_tanz.mul(&d_tanz).mul(&d_z);
            let mut d_rbb = d_dummy.div(&d_dummy2).times(3.0 * rbpi);
            d_rbb.value += rbpr;
            let d_vbb = d_rbb.mul(&d_p);
            if d_vbb.d1_p == 0. {
                return Err(SpiceError::Numerical {
                    context: format!("distortion of {name}"),
                    message: "zero base-resistance derivative (bjtdset.c d_vbb.d1_p = 0)".into(),
                });
            }
            // Power series inversion to the conductance derivatives.
            let gbb1 = 1. / d_vbb.d1_p;
            let gbb2 = -(d_vbb.d2_p2 * 0.5) * gbb1 * gbb1;
            let gbb3 = gbb1
                * gbb1
                * gbb1
                * gbb1
                * (-(d_vbb.d3_p3 / 6.0) + 2. * (d_vbb.d2_p2 * 0.5) * (d_vbb.d2_p2 * 0.5) * gbb1);
            let mut d = d_r;
            d.value = cb;
            d.d1_r = gbb1;
            d.d2_r2 = 2. * gbb2;
            d.d3_r3 = 6.0 * gbb3;
            d
        } else {
            let mut d_rbb = d_p;
            d_rbb.d1_p = 0.0;
            if rbpi != 0.0 {
                d_rbb = d_qb.inv().times(rbpi);
            }
            d_rbb.value += rbpr;
            let mut d_ibb = d_r;
            d_ibb.value = vbb;
            d_ibb.div(&d_rbb)
        }
    })
}

/// Derivative tensor of base current. C does not consume its value.
fn distortion_base_current(t: &Thermal, junctions: &DistortionJunctions) -> Dderivs {
    let d_p = Dderivs::variable(0);
    let JunctionDerivatives {
        first: gbe,
        second: gbe2,
        third: gbe3,
        ..
    } = junctions.be.main;
    let JunctionDerivatives {
        first: gben,
        second: gben2,
        third: gben3,
        ..
    } = junctions.be.leakage;
    let JunctionDerivatives {
        first: gbc,
        second: gbc2,
        third: gbc3,
        ..
    } = junctions.bc.main;
    let JunctionDerivatives {
        first: gbcn,
        second: gbcn2,
        third: gbcn3,
        ..
    } = junctions.bc.leakage;
    // Base current ib(p, q).
    let mut d_ib = d_p;
    d_ib.d1_p = gbe / t.bf + gben;
    d_ib.d2_p2 = gbe2 / t.bf + gben2;
    d_ib.d3_p3 = gbe3 / t.bf + gben3;
    d_ib.d1_q = gbc / t.br + gbcn;
    d_ib.d2_q2 = gbc2 / t.br + gbcn2;
    d_ib.d3_q3 = gbc3 / t.br + gbcn3;
    d_ib
}

/// Forward diffusion: unlike bjtload.c, reverse bias gives the zero
/// derivative tensor, and the forward quotient precedes multiplication.
fn distortion_forward_diffusion(
    t: &Thermal,
    bias: DistortionBias,
    m: Real,
    junctions: &DistortionJunctions,
    d_qb: &Dderivs,
) -> Dderivs {
    let DistortionBias { vbe, vbc, .. } = bias;
    let d_p = Dderivs::variable(0);
    let d_q = Dderivs::variable(1);
    let JunctionDerivatives {
        current: cbe,
        first: gbe,
        second: gbe2,
        third: gbe3,
    } = junctions.be.main;
    // B-E charge qbe(p, q): diffusion plus depletion.
    let tf = t.tf;
    let xtf = t.xtf;
    let ovtf = t.vtf_factor;
    let xjtf = t.itf * m;
    if tf != 0. && vbe > 0. {
        let mut d_cbe = d_p;
        d_cbe.value = cbe;
        d_cbe.d1_p = gbe;
        d_cbe.d2_p2 = gbe2;
        d_cbe.d3_p3 = gbe3;
        let d_tff = if xtf == 0. {
            Dderivs {
                value: tf,
                ..Dderivs::default()
            }
        } else {
            let d_dummy = if ovtf == 0. {
                Dderivs {
                    value: 1.,
                    ..Dderivs::default()
                }
            } else {
                let mut d = d_q;
                d.value = vbc * ovtf;
                d.d1_q = ovtf;
                d.exp()
            };
            let d_dummy2 = if xjtf == 0. {
                Dderivs {
                    value: 1.,
                    ..Dderivs::default()
                }
            } else {
                let mut d = d_cbe;
                d.value += xjtf;
                let d = d_cbe.div(&d);
                d.mul(&d)
            };
            let mut d = d_dummy.mul(&d_dummy2).times(tf * xtf);
            d.value += tf;
            d
        };
        d_tff.div(d_qb).mul(&d_cbe)
    } else {
        let mut d = d_p;
        d.value = 0.0;
        d.d1_p = 0.0;
        d
    }
}

fn distortion_be_charge(
    t: &Thermal,
    vbe: Real,
    area: Real,
    m: Real,
    mut d_qbe: Dderivs,
) -> Dderivs {
    let czbe = t.cje * area * m;
    let (pe, xme) = (t.be.potential, t.be.grading);
    let (capbe2, capbe3) = if vbe < t.be.threshold {
        depletion(czbe, pe, xme, vbe)
    } else {
        (0.5 * xme * (czbe / t.be.f2) / pe, 0.)
    };
    d_qbe.d2_p2 += capbe2 * 2.;
    d_qbe.d3_p3 += capbe3 * 6.;
    d_qbe
}

/// Unsigned Taylor coefficients of the remaining depletion junctions.
struct DepletionTerms {
    bc: Taylor,
    bx: Taylor,
    substrate: Taylor,
}

fn distortion_depletion(
    t: &Thermal,
    bias: DistortionBias,
    m: Real,
    ps: Real,
    xms: Real,
) -> DepletionTerms {
    let DistortionBias { vbc, vsc, .. } = bias;
    // B-C, B-X and substrate depletion charges (unsigned coefficients).
    let ctot = t.cjc * m;
    let czbc = ctot * t.xcjc;
    let czbx = ctot - czbc;
    let (pc, xmc) = (t.bc.potential, t.bc.grading);
    let collector = |cz: Real, v: Real| {
        if v < t.bc.threshold {
            depletion(cz, pc, xmc, v)
        } else {
            (0.5 * xmc * (cz / t.bc.f2) / pc, 0.)
        }
    };
    let (capbc2, capbc3) = collector(czbc, vbc);
    let (capbx2, capbx3) = collector(czbx, vbc);
    let czcs = t.cjs * m;

    let (capsc2, capsc3) = if vsc < 0. {
        depletion(czcs, ps, xms, vsc)
    } else {
        (czcs * 0.5 * xms / ps, 0.)
    };
    DepletionTerms {
        bc: Taylor::single(capbc2, capbc3),
        bx: Taylor::single(capbx2, capbx3),
        substrate: Taylor::single(capsc2, capsc3),
    }
}

impl Bjt {
    /// The `bjtdset.c` terms at the operating point (see the module
    /// documentation).
    pub(super) fn distortion_terms(
        &self,
        context: &DistortionContext<'_>,
    ) -> SpiceResult<DeviceDistortion> {
        let t = self.thermal(context.model_context)?;
        let m = self.instance.multiplier;
        let pol = self.pol;
        let gmin = context.model_context.gmin;
        let n = self.nodes;
        let v = |node| context.voltage(node);
        let bias = DistortionBias {
            vbe: pol * (v(n.bp) - v(n.ep)),
            vbc: pol * (v(n.b) - v(n.cp)),
            vsc: pol * (v(n.s) - v(n.cp)),
            vbb: pol * (v(n.b) - v(n.bp)),
        };
        let junctions = distortion_junctions(&t, bias, m, gmin);
        let d_qb = distortion_base_charge(&t, bias, m, &junctions);
        let d_ic = distortion_collector_current(&t, &junctions, &d_qb);
        let has_base_resistance = !(self.model.rbm == 0. && self.model.rb == self.model.rbm);
        let d_ibb = distortion_base_resistance(&t, bias.vbb, m, &junctions, &d_qb, &self.name)?;
        let d_ib = distortion_base_current(&t, &junctions);
        let diffusion = distortion_forward_diffusion(&t, bias, m, &junctions, &d_qb);
        let d_qbe = distortion_be_charge(&t, bias.vbe, self.instance.area, m, diffusion);
        let depletion = distortion_depletion(&t, bias, m, self.model.vjs, self.model.mjs);

        let vbe_control = || Control::between(n.bp, n.ep);
        let vbc_control = || Control::between(n.bp, n.cp);
        let mut terms = vec![
            DistortionTerm::new(
                Response::Current,
                [n.cp, n.ep],
                vec![vbe_control(), vbc_control(), vbe_control()],
                taylor(&d_ic, pol),
            ),
            DistortionTerm::new(
                Response::Current,
                [n.bp, n.ep],
                vec![vbe_control(), vbc_control()],
                taylor(&d_ib, pol),
            ),
        ];
        if has_base_resistance {
            terms.push(DistortionTerm::new(
                Response::Current,
                [n.b, n.bp],
                vec![
                    vbe_control(),
                    vbc_control(),
                    Control::sum(Vec::new()).with_unconjugated_in_f1_minus_f2(vec![[n.b, n.bp]]),
                ],
                taylor(&d_ibb, pol),
            ));
        }
        terms.push(DistortionTerm::new(
            Response::Charge,
            [n.bp, n.ep],
            vec![vbe_control(), vbc_control()],
            taylor(&d_qbe, pol),
        ));
        // bjtdisto.c: `z = vbx = vb - vc'` is formed as the previous `z` plus
        // `vbc`, the previous `z` being `vb - vb'` after the base-resistance
        // term, and the (undelayed) `vbe` kernel without one.
        let vbx_control = if has_base_resistance {
            Control::between(n.bp, n.cp).with_unconjugated_in_f1_minus_f2(vec![[n.b, n.bp]])
        } else {
            Control::sum(vec![[n.bp, n.ep], [n.bp, n.cp]])
        };
        terms.push(DistortionTerm::new(
            Response::Charge,
            [n.b, n.cp],
            vec![vbx_control],
            depletion.bx,
        ));
        terms.push(DistortionTerm::new(
            Response::Charge,
            [n.bp, n.cp],
            vec![vbc_control()],
            depletion.bc,
        ));
        terms.push(DistortionTerm::new(
            Response::Charge,
            [n.s, n.cp],
            vec![Control::sum(Vec::new()).with_unconjugated_in_f1_minus_f2(vec![[n.s, n.cp]])],
            depletion.substrate,
        ));
        Ok(DeviceDistortion::Terms(terms))
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::thermal;
    use super::*;

    #[test]
    fn distortion_reverse_boundary_and_main_gmin_are_independent_of_load() {
        let vtn = 0.025;
        let threshold = -5. * vtn;
        let saturation = 1e-15;
        let gmin = 1e-12;
        let leakage = 2e-14;
        let reverse = distortion_junction(threshold, vtn, saturation, 0.05, leakage, gmin);
        let main_first = -saturation / threshold + gmin;
        let leak_first = -leakage / threshold;
        assert_eq!(reverse.main.first.to_bits(), main_first.to_bits());
        assert_eq!(
            reverse.main.current.to_bits(),
            (main_first * threshold).to_bits()
        );
        assert_eq!(reverse.leakage.first.to_bits(), leak_first.to_bits());
        assert_eq!(
            reverse.leakage.current.to_bits(),
            (leak_first * threshold).to_bits()
        );
        assert_eq!(
            [
                reverse.main.second,
                reverse.main.third,
                reverse.leakage.second,
                reverse.leakage.third
            ],
            [0.; 4]
        );
        let forward = distortion_junction(threshold + 1e-12, vtn, saturation, 0.05, leakage, gmin);
        assert!(forward.main.second > 0. && forward.main.third > 0.);
        assert!(forward.leakage.second > 0. && forward.leakage.third > 0.);
    }

    #[test]
    fn distortion_coefficients_match_independent_exponential_derivatives() {
        // No Early effect or injection: qb=1, so the reference coefficient
        // of delayed vbe cubed is IS*exp(vbe/vtn)/(6*vtn^3). The base
        // derivative includes the leakage exponent and beta, never gmin.
        let t = thermal("is=1e-15 bf=80 br=2 ise=2e-14 ne=1.5 tf=0.4n", 85., true);
        let m = 2.3;
        let bias = DistortionBias {
            vbe: 0.72,
            vbc: -0.03,
            vsc: 0.,
            vbb: 0.,
        };
        let junctions = distortion_junctions(&t, bias, m, 1e-12);
        let qb = distortion_base_charge(&t, bias, m, &junctions);
        let collector = taylor(&distortion_collector_current(&t, &junctions, &qb), 1.);
        let base = taylor(&distortion_base_current(&t, &junctions), -1.);
        let diffusion = taylor(
            &distortion_forward_diffusion(&t, bias, m, &junctions, &qb),
            1.,
        );
        let vtn = t.vt * t.nf;
        let vte = t.ne * t.vt;
        let main = t.is_be * m * (bias.vbe / vtn).exp();
        let leakage = t.ise * m * (bias.vbe / vte).exp();
        let close = |actual: Real, expected: Real| {
            assert!(
                (actual - expected).abs() <= expected.abs() * 1e-14,
                "{actual:e} != {expected:e}"
            );
        };
        close(collector.zzz, main / (6. * vtn.powi(3)));
        close(
            base.xx,
            -(main / t.bf / vtn.powi(2) + leakage / vte.powi(2)) / 2.,
        );
        close(
            base.xxx,
            (main / t.bf / vtn.powi(3) + leakage / vte.powi(3)) / 6.,
        );
        close(diffusion.xx, t.tf * main / (2. * vtn.powi(2)));
        close(diffusion.xxx, t.tf * main / (6. * vtn.powi(3)));
        let resistance = distortion_base_resistance(&t, bias.vbb, m, &junctions, &qb, "q").unwrap();
        assert_eq!(resistance, Dderivs::default());
    }

    #[test]
    fn distortion_base_charge_does_not_inherit_nkf() {
        let ordinary = thermal("is=1e-15 vaf=40 ikf=3m ikr=0.5m", 27., true);
        let nkf = thermal("is=1e-15 vaf=40 ikf=3m ikr=0.5m nkf=0.7", 27., true);
        let bias = DistortionBias {
            vbe: 0.77,
            vbc: -0.2,
            vsc: 0.,
            vbb: 0.,
        };
        let first = distortion_base_charge(
            &ordinary,
            bias,
            2.,
            &distortion_junctions(&ordinary, bias, 2., 1e-12),
        );
        let second =
            distortion_base_charge(&nkf, bias, 2., &distortion_junctions(&nkf, bias, 2., 1e-12));
        assert_eq!(first, second);
    }
}
