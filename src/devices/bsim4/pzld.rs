//! `b4pzld.c` (`BSIM4pzLoad`): the pole-zero load, followed statement by
//! statement over C's struct members.
//!
//! C adds `x * s` as `*(ptr) += x * s->real; *(ptr + 1) += x * s->imag`;
//! the port passes `x` to [`SmallSignalCkt::cap`], the coefficient of `s`
//! in the pencil `A + s E`. Every `b4pzld.c` term is affine in `s`. One
//! grouping differs from C: the AC NQS charge-node term
//! `m * s->real * ScalingFactor` is the coefficient `m * ScalingFactor`
//! times `s`, which rounds differently from C's left-to-right product.
// PZ conductance and charge-partition products use FMA where the physical
// derivative requires cancellation-sensitive arithmetic. Branch thresholds
// and the input model's dimensionless charge-partition limits remain distinct.
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
use super::fields::{Instance, Model, SizeParam};
use super::host::{SmallSignalCkt, state};
use super::stamp_common::{
    self, ChannelCurrent, ChannelGain, GateCurrent, GateResistance, GateStamp, MidGateCharge,
    Pencil, SeriesResistance, SubstrateCurrent,
};

/// `BSIM4pzLoad` for one instance, from the values the `MODEINITSMSIG`
/// load left on `h`.
///
/// Like C's always-successful kernel, this cannot fail itself; the caller
/// checks the host's recorded stamping error.
#[expect(
    clippy::too_many_lines,
    reason = "keeps the reference numerical routine in the original statement and rounding order"
)] // One C function, kept in C's shape.
pub(super) fn pz_load(m: &Model, h: &Instance, p: &SizeParam, ckt: &mut dyn SmallSignalCkt) {
    let ScalingFactor = 1.0e-9;
    let cgso = h.cgso;
    let cgdo = h.cgdo;

    let mut xcdbdb = 0.0;
    let mid = MidGateCharge::new::<Pencil>(h, p, 0.);
    let MidGateCharge {
        drain: xcdgmb,
        source: xcsgmb,
        bulk: xcbgmb,
        ..
    } = mid;
    let mut xcsbsb = 0.0;
    let mut xcqgb = 0.0;
    let mut xcqdb = 0.0;
    let mut xcqsb = 0.0;
    let mut xcqbb = 0.0;

    let real_gain = ChannelGain::new(h.mode, h.gm, h.gmbs);
    let substrate = SubstrateCurrent::new(h);
    let gate_current = GateCurrent::new(m, h);
    let GateCurrent {
        gIgtotg,
        gIgtotd,
        gIgtots,
        gIgtotb,
        ..
    } = gate_current;
    let resistance = GateResistance::new(h, ckt);
    let GateResistance {
        gcrgd,
        gcrgg,
        gcrgs,
        gcrgb,
        ..
    } = resistance;
    let (xcggb, xcgdb, xcgsb, xcgbb, xcbgb, xcbdb, xcbsb, xcbbb);
    let (xcdgb, xcddb, xcdsb, xcdbb, xcsgb, xcsdb, xcssb, xcsbb);
    let (xgtg, xgtd, xgts, xgtb, sxpart, dxpart);
    let (ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs);
    let (dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs);

    if h.mode >= 0 {
        if h.acnqsMod == 0 {
            if h.rgateMod == 3 {
                xcggb = h.cggb;
                xcgdb = h.cgdb;
                xcgsb = h.cgsb;
                xcgbb = -(xcggb + xcgdb + xcgsb);

                xcdgb = h.cdgb;
                xcsgb = -(h.cggb + h.cbgb + h.cdgb);
                xcbgb = h.cbgb;
            } else {
                xcggb = h.cggb + cgdo + cgso + p.cgbo;
                xcgdb = h.cgdb - cgdo;
                xcgsb = h.cgsb - cgso;
                xcgbb = -(xcggb + xcgdb + xcgsb);

                xcdgb = h.cdgb - cgdo;
                xcsgb = -(h.cggb + h.cbgb + h.cdgb + cgso);
                xcbgb = h.cbgb - p.cgbo;
            }
            xcddb = h.cddb + h.capbd + cgdo;
            xcdsb = h.cdsb;

            xcsdb = -(h.cgdb + h.cbdb + h.cddb);
            xcssb = h.capbs + cgso - (h.cgsb + h.cbsb + h.cdsb);

            if h.rbodyMod == 0 {
                xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                xcbdb = h.cbdb - h.capbd;
                xcbsb = h.cbsb - h.capbs;
                xcdbdb = 0.0;
            } else {
                xcdbb = -(h.cddb + h.cdgb + h.cdsb);
                xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb) + h.capbs;
                xcbdb = h.cbdb;
                xcbsb = h.cbsb;

                xcdbdb = -h.capbd;
                xcsbsb = -h.capbs;
            }
            xcbbb = -(xcbdb + xcbgb + xcbsb + xcbgmb);

            (xgtg, xgtd, xgts, xgtb) = (0.0, 0.0, 0.0, 0.0);
            sxpart = 0.6;
            dxpart = 0.4;
            (ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs) = (0.0, 0.0, 0.0, 0.0);
            (dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs) = (0.0, 0.0, 0.0, 0.0);
        } else {
            (xcggb, xcgdb, xcgsb, xcgbb) = (0.0, 0.0, 0.0, 0.0);
            (xcbgb, xcbdb, xcbsb, xcbbb) = (0.0, 0.0, 0.0, 0.0);
            (xcdgb, xcddb, xcdsb, xcdbb) = (0.0, 0.0, 0.0, 0.0);
            (xcsgb, xcsdb, xcssb, xcsbb) = (0.0, 0.0, 0.0, 0.0);

            xgtg = h.gtg;
            xgtd = h.gtd;
            xgts = h.gts;
            xgtb = h.gtb;

            xcqgb = h.cqgb;
            xcqdb = h.cqdb;
            xcqsb = h.cqsb;
            xcqbb = h.cqbb;

            let CoxWL = m.coxe * p.weffCV * h.nf * p.leffCV;
            let qcheq = -(h.qgate + h.qbulk);
            if qcheq.abs() <= 1.0e-5 * CoxWL {
                dxpart = if m.xpart < 0.5 {
                    0.4
                } else if m.xpart > 0.5 {
                    0.0
                } else {
                    0.5
                };
                (ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs) = (0.0, 0.0, 0.0, 0.0);
            } else {
                dxpart = h.qdrn / qcheq;
                let Cdd = h.cddb;
                let Csd = -(h.cgdb + h.cddb + h.cbdb);
                ddxpart_dVd = f64::mul_add(dxpart, -(Cdd + Csd), Cdd) / qcheq;
                let Cdg = h.cdgb;
                let Csg = -(h.cggb + h.cdgb + h.cbgb);
                ddxpart_dVg = f64::mul_add(dxpart, -(Cdg + Csg), Cdg) / qcheq;

                let Cds = h.cdsb;
                let Css = -(h.cgsb + h.cdsb + h.cbsb);
                ddxpart_dVs = f64::mul_add(dxpart, -(Cds + Css), Cds) / qcheq;

                ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs);
            }
            sxpart = 1.0 - dxpart;
            dsxpart_dVd = -ddxpart_dVd;
            dsxpart_dVg = -ddxpart_dVg;
            dsxpart_dVs = -ddxpart_dVs;
            dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs);
        }
    } else {
        if h.acnqsMod == 0 {
            if h.rgateMod == 3 {
                xcggb = h.cggb;
                xcgdb = h.cgsb;
                xcgsb = h.cgdb;
                xcgbb = -(xcggb + xcgdb + xcgsb);

                xcdgb = -(h.cggb + h.cbgb + h.cdgb);
                xcsgb = h.cdgb;
                xcbgb = h.cbgb;
            } else {
                xcggb = h.cggb + cgdo + cgso + p.cgbo;
                xcgdb = h.cgsb - cgdo;
                xcgsb = h.cgdb - cgso;
                xcgbb = -(xcggb + xcgdb + xcgsb);

                xcdgb = -(h.cggb + h.cbgb + h.cdgb + cgdo);
                xcsgb = h.cdgb - cgso;
                xcbgb = h.cbgb - p.cgbo;
            }
            xcddb = h.capbd + cgdo - (h.cgsb + h.cbsb + h.cdsb);
            xcdsb = -(h.cgdb + h.cbdb + h.cddb);

            xcsdb = h.cdsb;
            xcssb = h.cddb + h.capbs + cgso;

            if h.rbodyMod == 0 {
                xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb);
                xcsbb = -(xcsgb + xcsdb + xcssb + xcsgmb);
                xcbdb = h.cbsb - h.capbd;
                xcbsb = h.cbdb - h.capbs;
                xcdbdb = 0.0;
            } else {
                xcdbb = -(xcdgb + xcddb + xcdsb + xcdgmb) + h.capbd;
                xcsbb = -(h.cddb + h.cdgb + h.cdsb);
                xcbdb = h.cbsb;
                xcbsb = h.cbdb;
                xcdbdb = -h.capbd;
                xcsbsb = -h.capbs;
            }
            xcbbb = -(xcbgb + xcbdb + xcbsb + xcbgmb);

            (xgtg, xgtd, xgts, xgtb) = (0.0, 0.0, 0.0, 0.0);
            sxpart = 0.4;
            dxpart = 0.6;
            (ddxpart_dVd, ddxpart_dVg, ddxpart_dVb, ddxpart_dVs) = (0.0, 0.0, 0.0, 0.0);
            (dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs) = (0.0, 0.0, 0.0, 0.0);
        } else {
            (xcggb, xcgdb, xcgsb, xcgbb) = (0.0, 0.0, 0.0, 0.0);
            (xcbgb, xcbdb, xcbsb, xcbbb) = (0.0, 0.0, 0.0, 0.0);
            (xcdgb, xcddb, xcdsb, xcdbb) = (0.0, 0.0, 0.0, 0.0);
            (xcsgb, xcsdb, xcssb, xcsbb) = (0.0, 0.0, 0.0, 0.0);

            xgtg = h.gtg;
            xgtd = h.gts;
            xgts = h.gtd;
            xgtb = h.gtb;

            xcqgb = h.cqgb;
            xcqdb = h.cqsb;
            xcqsb = h.cqdb;
            xcqbb = h.cqbb;

            let CoxWL = m.coxe * p.weffCV * h.nf * p.leffCV;
            let qcheq = -(h.qgate + h.qbulk);
            if qcheq.abs() <= 1.0e-5 * CoxWL {
                sxpart = if m.xpart < 0.5 {
                    0.4
                } else if m.xpart > 0.5 {
                    0.0
                } else {
                    0.5
                };
                (dsxpart_dVd, dsxpart_dVg, dsxpart_dVb, dsxpart_dVs) = (0.0, 0.0, 0.0, 0.0);
            } else {
                sxpart = h.qdrn / qcheq;
                let Css = h.cddb;
                let Cds = -(h.cgdb + h.cddb + h.cbdb);
                dsxpart_dVs = f64::mul_add(sxpart, -(Css + Cds), Css) / qcheq;
                let Csg = h.cdgb;
                let Cdg = -(h.cggb + h.cdgb + h.cbgb);
                dsxpart_dVg = f64::mul_add(sxpart, -(Csg + Cdg), Csg) / qcheq;

                let Csd = h.cdsb;
                let Cdd = -(h.cgsb + h.cdsb + h.cbsb);
                dsxpart_dVd = f64::mul_add(sxpart, -(Csd + Cdd), Csd) / qcheq;

                dsxpart_dVb = -(dsxpart_dVd + dsxpart_dVg + dsxpart_dVs);
            }
            dxpart = 1.0 - sxpart;
            ddxpart_dVd = -dsxpart_dVd;
            ddxpart_dVg = -dsxpart_dVg;
            ddxpart_dVs = -dsxpart_dVs;
            ddxpart_dVb = -(ddxpart_dVd + ddxpart_dVg + ddxpart_dVs);
        }
    }

    let series = SeriesResistance::new(m, h);
    let SeriesResistance {
        gstot,
        gstots,
        gdtot,
        gdtotd,
        gdpr,
        gspr,
        ..
    } = series;

    let T1 = ckt.state0(state::QDEF) * h.gtau;
    let gds = h.gds;

    // Loading PZ matrix
    let mm = h.m;
    let mult_i = h.mult_i * h.m;
    let mult_q = h.mult_q * h.m;

    let geltd = h.grgeltd;

    let (dp, gp, sp, bp) = (h.dNodePrime, h.gNodePrime, h.sNodePrime, h.bNodePrime);
    let (d, s, q) = (h.dNode, h.sNode, h.qNode);

    let prime_real = match h.rgateMod {
        1 => [
            mult_q.mul_add(-xgtg, mult_i * (geltd + gIgtotg)),
            -mult_i.mul_add(-gIgtotd, mult_q * xgtd),
            -mult_i.mul_add(-gIgtots, mult_q * xgts),
            -mult_i.mul_add(-gIgtotb, mult_q * xgtb),
        ],
        2 | 3 => [
            -mult_q.mul_add(xgtg, mult_i * (gcrgg - gIgtotg)),
            -mult_q.mul_add(xgtd, mult_i * (gcrgd - gIgtotd)),
            -mult_q.mul_add(xgts, mult_i * (gcrgs - gIgtots)),
            -mult_q.mul_add(xgtb, mult_i * (gcrgb - gIgtotb)),
        ],
        _ => [
            -mult_i.mul_add(-gIgtotg, mult_q * xgtg),
            -mult_i.mul_add(-gIgtotd, mult_q * xgtd),
            -mult_i.mul_add(-gIgtots, mult_q * xgts),
            -mult_i.mul_add(-gIgtotb, mult_q * xgtb),
        ],
    };
    GateStamp {
        real: prime_real,
        charge: [xcggb, xcgdb, xcgsb, xcgbb],
        mid,
    }
    .emit::<Pencil>(h, &resistance, ckt);

    series.emit_external(m, h, ckt);
    let current =
        ChannelCurrent::new::<Pencil>(h, real_gain, gds, &substrate, &gate_current, &series);

    ckt.cap(dp, dp, mult_q * xcddb);
    ckt.real(
        dp,
        dp,
        mult_q.mul_add(
            T1.mul_add(ddxpart_dVd, dxpart * xgtd),
            mult_i * current.drain[0],
        ),
    );
    ckt.real(dp, d, -(mult_i * (gdpr + gdtot)));
    ckt.cap(dp, gp, mult_q * xcdgb);
    ckt.real(
        dp,
        gp,
        mult_q.mul_add(
            f64::mul_add(dxpart, xgtg, T1 * ddxpart_dVg),
            mult_i * current.drain[1],
        ),
    );
    ckt.cap(dp, sp, mult_q * xcdsb);
    ckt.real(
        dp,
        sp,
        -mult_q.mul_add(
            -f64::mul_add(dxpart, xgts, T1 * ddxpart_dVs),
            mult_i * current.drain[2],
        ),
    );
    ckt.cap(dp, bp, mult_q * xcdbb);
    ckt.real(
        dp,
        bp,
        -(mult_q * T1).mul_add(
            -f64::mul_add(dxpart, xgtb, ddxpart_dVb),
            mult_i * current.drain[3],
        ),
    );

    ckt.real(d, dp, -(mult_i * (gdpr - gdtotd)));
    ckt.real(d, d, mult_i * (gdpr + gdtot));

    ckt.cap(sp, dp, mult_q * xcsdb);
    ckt.real(
        sp,
        dp,
        -mult_q.mul_add(
            -f64::mul_add(sxpart, xgtd, T1 * dsxpart_dVd),
            mult_i * current.source[0],
        ),
    );
    ckt.cap(sp, gp, mult_q * xcsgb);
    ckt.real(
        sp,
        gp,
        -mult_q.mul_add(
            -f64::mul_add(sxpart, xgtg, T1 * dsxpart_dVg),
            mult_i * current.source[1],
        ),
    );
    ckt.cap(sp, sp, mult_q * xcssb);
    ckt.real(
        sp,
        sp,
        mult_q.mul_add(
            T1.mul_add(dsxpart_dVs, sxpart * xgts),
            mult_i * current.source[2],
        ),
    );
    ckt.real(sp, s, -(mult_i * (gspr + gstot)));
    ckt.cap(sp, bp, mult_q * xcsbb);
    ckt.real(
        sp,
        bp,
        -mult_q.mul_add(
            -f64::mul_add(sxpart, xgtb, T1 * dsxpart_dVb),
            mult_i * current.source[3],
        ),
    );

    ckt.real(s, sp, -(mult_i * (gspr - gstots)));
    ckt.real(s, s, mult_i * (gspr + gstot));

    stamp_common::bulk::<Pencil>(
        h,
        &substrate,
        &gate_current,
        [xcbdb, xcbgb, xcbsb, xcbbb],
        ckt,
    );
    stamp_common::body_network::<Pencil>(h, [xcdbdb, xcsbsb], ckt);

    if h.acnqsMod != 0 {
        // C: `*(QqPtr) += m * s->real * ScalingFactor` (see the module note).
        ckt.cap(q, q, mm * ScalingFactor);
        ckt.cap(q, gp, -(mm * xcqgb));
        ckt.cap(q, dp, -(mm * xcqdb));
        ckt.cap(q, bp, -(mm * xcqbb));
        ckt.cap(q, sp, -(mm * xcqsb));

        ckt.real(gp, q, -(mm * h.gtau));
        ckt.real(dp, q, mm * dxpart * h.gtau);
        ckt.real(sp, q, mm * sxpart * h.gtau);

        ckt.real(q, q, mm * h.gtau);
        ckt.real(q, gp, mm * xgtg);
        ckt.real(q, dp, mm * xgtd);
        ckt.real(q, bp, mm * xgtb);
        ckt.real(q, sp, mm * xgts);
    }
}

#[cfg(test)]
mod tests {
    use super::super::acld::tests::check_shapes;
    use super::SizeParam;

    /// The exact derivative is 2^-55: its separate product rounds to 1/2,
    /// so the historical C-shaped subtraction returned zero. Check the
    /// derivative through the actual PZ stamp and its source companion.
    #[test]
    fn pz_nqs_partition_derivative_retains_sub_ulp_cancellation() {
        use super::super::acld::tests::{Recorder, shaped};
        use super::super::host::state;
        use super::{Instance, Model};
        let mut h: Instance = shaped(0, 0, 1);
        h.acnqsMod = 1;
        h.qgate = -1.;
        h.qdrn = 0.5 + 2_f64.powi(-28);
        h.cddb = 0.5;
        h.cgdb = -(1. - 2_f64.powi(-27));
        h.gtau = 1.;
        let mut ckt = Recorder {
            state: vec![0.; state::COUNT],
            ..Recorder::default()
        };
        ckt.state[state::QDEF] = 1.;
        super::pz_load(&Model::default(), &h, &SizeParam::default(), &mut ckt);
        let stamp = |row, col| {
            ckt.real
                .iter()
                .filter(|&&(r, c, _)| r == row && c == col)
                .map(|entry| entry.2)
                .sum::<f64>()
        };
        let retained = stamp(h.dNodePrime, h.dNodePrime);
        assert_eq!(retained, 2_f64.powi(-55));
        let companion = stamp(h.sNodePrime, h.dNodePrime);
        assert_eq!(companion, -retained);
        assert!((retained + companion).abs() <= f64::MIN_POSITIVE);
    }

    #[test]
    fn pz_stamps_land_on_allocated_elements() {
        check_shapes(
            |m, h, ckt| super::pz_load(m, h, &SizeParam::default(), ckt),
            true,
            false,
        );
    }
}
