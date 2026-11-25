//! `b4acld.c` (`BSIM4acLoad`): the small-signal AC load, followed statement
//! by statement over C's struct members.
// C-compatible numerical expressions are kept in their reference order:
// Clippy's fused multiply-add suggestions change IEEE-754 rounding.
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
use super::host::SmallSignalCkt;
use super::stamp_common::{
    self, Ac, ChannelCurrent, ChannelGain, GateCurrent, GateResistance, GateStamp, MidGateCharge,
    SeriesResistance, SubstrateCurrent,
};

/// Real and quadrature gains of the AC charge-relaxation kernel
/// `1 / (1 + j omega tau)`.
///
/// For `|omega tau| > 1`, work with its reciprocal instead of squaring
/// the product. The original quadratic overflows at ~1e154, silently
/// discarding a *representable* quadrature tail. When two finite inputs
/// overflow their product, evaluate the reciprocal in separate factors.
/// Nonfinite input physics still propagates IEEE nonfinites.
fn nqs_relaxation(omega: f64, tau: f64) -> (f64, f64) {
    let x = omega * tau;
    if x.is_finite() && x.abs() > 1. {
        let reciprocal = 1. / x;
        let denominator = reciprocal.mul_add(reciprocal, 1.);
        return (
            reciprocal * reciprocal / denominator,
            reciprocal / denominator,
        );
    }
    if x.is_infinite() && omega.is_finite() && tau.is_finite() {
        return (0., (1. / omega) / tau);
    }
    let real = 1. / x.mul_add(x, 1.);
    (real, x * real)
}

/// `BSIM4acLoad` for one instance, from the values the `MODEINITSMSIG`
/// load left on `h`.
#[expect(
    clippy::too_many_lines,
    reason = "keeps the reference numerical routine in the original statement and rounding order"
)] // One C function, kept in C's shape.
pub(super) fn ac_load(m: &Model, h: &Instance, p: &SizeParam, ckt: &mut dyn SmallSignalCkt) {
    let omega = ckt.omega();
    let cgso = h.cgso;
    let cgdo = h.cgdo;

    let Csd = -(h.cddb + h.cgdb + h.cbdb);
    let Csg = -(h.cdgb + h.cggb + h.cbgb);
    let Css = -(h.cdsb + h.cgsb + h.cbsb);

    let (gmr, gmbsr, gdsr, gmi, gmbsi, gdsi);
    let (Cddr, Cdgr, Cdsr, Cdbr, Cddi, Cdgi, Cdsi, Cdbi);
    let (Csdr, Csgr, Cssr, Csbr, Csdi, Csgi, Cssi, Csbi);
    let (Cgdr, Cggr, Cgsr, Cgdi, Cggi, Cgsi, Cgbi);
    if h.acnqsMod != 0 {
        let (T2, T3) = nqs_relaxation(omega, h.taunet);
        let nqs_cap_conductance = T3 * omega;

        gmr = h.gm * T2;
        gmbsr = h.gmbs * T2;
        gdsr = h.gds * T2;

        gmi = -h.gm * T3;
        gmbsi = -h.gmbs * T3;
        gdsi = -h.gds * T3;

        Cddr = h.cddb * T2;
        Cdgr = h.cdgb * T2;
        Cdsr = h.cdsb * T2;
        Cdbr = -(Cddr + Cdgr + Cdsr);

        // WDLiu: Cxyi multiplied by jomega below, and actually to be of
        // conductance.
        Cddi = h.cddb * nqs_cap_conductance;
        Cdgi = h.cdgb * nqs_cap_conductance;
        Cdsi = h.cdsb * nqs_cap_conductance;
        Cdbi = -(Cddi + Cdgi + Cdsi);

        Csdr = Csd * T2;
        Csgr = Csg * T2;
        Cssr = Css * T2;
        Csbr = -(Csdr + Csgr + Cssr);

        Csdi = Csd * nqs_cap_conductance;
        Csgi = Csg * nqs_cap_conductance;
        Cssi = Css * nqs_cap_conductance;
        Csbi = -(Csdi + Csgi + Cssi);

        Cgdr = -(Cddr + Csdr + h.cbdb);
        Cggr = -(Cdgr + Csgr + h.cbgb);
        Cgsr = -(Cdsr + Cssr + h.cbsb);
        // C's Cgbr = -(Cgdr + Cggr + Cgsr) is never read.

        Cgdi = -(Cddi + Csdi);
        Cggi = -(Cdgi + Csgi);
        Cgsi = -(Cdsi + Cssi);
        Cgbi = -(Cgdi + Cggi + Cgsi);
    } else {
        // QS
        gmr = h.gm;
        gmbsr = h.gmbs;
        gdsr = h.gds;
        gmi = 0.0;
        gmbsi = 0.0;
        gdsi = 0.0;

        Cddr = h.cddb;
        Cdgr = h.cdgb;
        Cdsr = h.cdsb;
        Cdbr = -(Cddr + Cdgr + Cdsr);
        Cddi = 0.0;
        Cdgi = 0.0;
        Cdsi = 0.0;
        Cdbi = 0.0;

        Csdr = Csd;
        Csgr = Csg;
        Cssr = Css;
        Csbr = -(Csdr + Csgr + Cssr);
        Csdi = 0.0;
        Csgi = 0.0;
        Cssi = 0.0;
        Csbi = 0.0;

        Cgdr = h.cgdb;
        Cggr = h.cggb;
        Cgsr = h.cgsb;
        Cgdi = 0.0;
        Cggi = 0.0;
        Cgsi = 0.0;
        Cgbi = 0.0;
    }
    let _ = Csbr;

    let real_gain = ChannelGain::new(h.mode, gmr, gmbsr);
    let ChannelGain {
        gm: Gmi,
        gmbs: Gmbsi,
        forward: FwdSumi,
        reverse: RevSumi,
    } = ChannelGain::new(h.mode, gmi, gmbsi);
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
    let mid = MidGateCharge::new::<Ac>(h, p, omega);
    let MidGateCharge {
        drain: xcdgmb,
        source: xcsgmb,
        bulk: xcbgmb,
        ..
    } = mid;
    let mut xcsbsb = 0.0;
    let (xcggbr, xcgdbr, xcgsbr, xcgbbr, xcdgbr, xcsgbr, xcbgb);
    let (xcddbr, xcdsbr, xcsdbr, xcssbr, xcdbbr, xcsbbr, xcbdb, xcbsb, xcdbdb, xcbbb);
    let (xcdgbi, xcsgbi, xcddbi, xcdsbi, xcsdbi, xcssbi, xcdbbi, xcsbbi);
    let (xcggbi, xcgdbi, xcgsbi);
    if h.mode >= 0 {
        if h.rgateMod == 3 {
            xcggbr = Cggr * omega;
            xcgdbr = Cgdr * omega;
            xcgsbr = Cgsr * omega;
            xcgbbr = -(xcggbr + xcgdbr + xcgsbr);

            xcdgbr = Cdgr * omega;
            xcsgbr = Csgr * omega;
            xcbgb = h.cbgb * omega;
        } else {
            xcggbr = (Cggr + cgdo + cgso + p.cgbo) * omega;
            xcgdbr = (Cgdr - cgdo) * omega;
            xcgsbr = (Cgsr - cgso) * omega;
            xcgbbr = -(xcggbr + xcgdbr + xcgsbr);

            xcdgbr = (Cdgr - cgdo) * omega;
            xcsgbr = (Csgr - cgso) * omega;
            xcbgb = (h.cbgb - p.cgbo) * omega;
        }
        xcddbr = (Cddr + h.capbd + cgdo) * omega;
        xcdsbr = Cdsr * omega;
        xcsdbr = Csdr * omega;
        xcssbr = (h.capbs + cgso + Cssr) * omega;

        if h.rbodyMod == 0 {
            xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
            xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

            xcbdb = (h.cbdb - h.capbd) * omega;
            xcbsb = (h.cbsb - h.capbs) * omega;
            xcdbdb = 0.0;
        } else {
            xcdbbr = Cdbr * omega;
            xcsbbr = h.capbs.mul_add(omega, -(xcsgbr + xcsdbr + xcssbr + xcsgmb));

            xcbdb = h.cbdb * omega;
            xcbsb = h.cbsb * omega;

            xcdbdb = -h.capbd * omega;
            xcsbsb = -h.capbs * omega;
        }
        xcbbb = -(xcbdb + xcbgb + xcbsb + xcbgmb);

        xcdgbi = Cdgi;
        xcsgbi = Csgi;
        xcddbi = Cddi;
        xcdsbi = Cdsi;
        xcsdbi = Csdi;
        xcssbi = Cssi;
        xcdbbi = Cdbi;
        xcsbbi = Csbi;
        xcggbi = Cggi;
        xcgdbi = Cgdi;
        xcgsbi = Cgsi;
    } else {
        // Reverse mode
        if h.rgateMod == 3 {
            xcggbr = Cggr * omega;
            xcgdbr = Cgsr * omega;
            xcgsbr = Cgdr * omega;
            xcgbbr = -(xcggbr + xcgdbr + xcgsbr);

            xcdgbr = Csgr * omega;
            xcsgbr = Cdgr * omega;
            xcbgb = h.cbgb * omega;
        } else {
            xcggbr = (Cggr + cgdo + cgso + p.cgbo) * omega;
            xcgdbr = (Cgsr - cgdo) * omega;
            xcgsbr = (Cgdr - cgso) * omega;
            xcgbbr = -(xcggbr + xcgdbr + xcgsbr);

            xcdgbr = (Csgr - cgdo) * omega;
            xcsgbr = (Cdgr - cgso) * omega;
            xcbgb = (h.cbgb - p.cgbo) * omega;
        }
        xcddbr = (h.capbd + cgdo + Cssr) * omega;
        xcdsbr = Csdr * omega;
        xcsdbr = Cdsr * omega;
        xcssbr = (Cddr + h.capbs + cgso) * omega;

        if h.rbodyMod == 0 {
            xcdbbr = -(xcdgbr + xcddbr + xcdsbr + xcdgmb);
            xcsbbr = -(xcsgbr + xcsdbr + xcssbr + xcsgmb);

            xcbdb = (h.cbsb - h.capbd) * omega;
            xcbsb = (h.cbdb - h.capbs) * omega;
            xcdbdb = 0.0;
        } else {
            xcdbbr = h.capbd.mul_add(omega, -(xcdgbr + xcddbr + xcdsbr + xcdgmb));
            xcsbbr = Cdbr * omega;

            xcbdb = h.cbsb * omega;
            xcbsb = h.cbdb * omega;
            xcdbdb = -h.capbd * omega;
            xcsbsb = -h.capbs * omega;
        }
        xcbbb = -(xcbgb + xcbdb + xcbsb + xcbgmb);

        xcdgbi = Csgi;
        xcsgbi = Cdgi;
        xcddbi = Cssi;
        xcdsbi = Csdi;
        xcsdbi = Cdsi;
        xcssbi = Cddi;
        xcdbbi = Csbi;
        xcsbbi = Cdbi;
        xcggbi = Cggi;
        xcgdbi = Cgsi;
        xcgsbi = Cgdi;
    }

    let xcgbbi = Cgbi;

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

    // Loading AC matrix
    let mm = h.m;
    let mult_i = h.mult_i * h.m;
    let mult_q = h.mult_q * h.m;

    let geltd = h.grgeltd;

    let (dp, gp, sp, bp) = (h.dNodePrime, h.gNodePrime, h.sNodePrime, h.bNodePrime);
    let (d, s, q) = (h.dNode, h.sNode, h.qNode);

    let prime_real = match h.rgateMod {
        1 => [
            mult_q.mul_add(xcggbi, mult_i * (geltd + gIgtotg)),
            mult_i.mul_add(gIgtotd, mult_q * xcgdbi),
            mult_i.mul_add(gIgtots, mult_q * xcgsbi),
            mult_i.mul_add(gIgtotb, mult_q * xcgbbi),
        ],
        2 | 3 => [
            -mult_q.mul_add(-xcggbi, mult_i * (gcrgg - gIgtotg)),
            -mult_q.mul_add(-xcgdbi, mult_i * (gcrgd - gIgtotd)),
            -mult_q.mul_add(-xcgsbi, mult_i * (gcrgs - gIgtots)),
            -mult_q.mul_add(-xcgbbi, mult_i * (gcrgb - gIgtotb)),
        ],
        _ => [
            mult_i.mul_add(gIgtotg, mult_q * xcggbi),
            mult_i.mul_add(gIgtotd, mult_q * xcgdbi),
            mult_i.mul_add(gIgtots, mult_q * xcgsbi),
            mult_i.mul_add(gIgtotb, mult_q * xcgbbi),
        ],
    };
    GateStamp {
        real: prime_real,
        charge: [xcggbr, xcgdbr, xcgsbr, xcgbbr],
        mid,
    }
    .emit::<Ac>(h, &resistance, ckt);

    series.emit_external(m, h, ckt);
    let current = ChannelCurrent::new::<Ac>(h, real_gain, gdsr, &substrate, &gate_current, &series);

    ckt.imag(dp, dp, mult_i.mul_add(gdsi + RevSumi, mult_q * xcddbr));
    ckt.real(dp, dp, mult_q.mul_add(xcddbi, mult_i * current.drain[0]));
    ckt.real(dp, d, -(mult_i * (gdpr + gdtot)));
    ckt.imag(dp, gp, mult_i.mul_add(Gmi, mult_q * xcdgbr));
    ckt.real(dp, gp, mult_q.mul_add(xcdgbi, mult_i * current.drain[1]));
    ckt.imag(dp, sp, mult_i.mul_add(-(gdsi + FwdSumi), mult_q * xcdsbr));
    ckt.real(dp, sp, -mult_q.mul_add(-xcdsbi, mult_i * current.drain[2]));
    ckt.imag(dp, bp, mult_i.mul_add(Gmbsi, mult_q * xcdbbr));
    ckt.real(dp, bp, -mult_q.mul_add(-xcdbbi, mult_i * current.drain[3]));

    ckt.real(d, dp, -(mult_i * (gdpr - gdtotd)));
    ckt.real(d, d, mult_i * (gdpr + gdtot));

    ckt.imag(sp, dp, mult_i.mul_add(-(gdsi + RevSumi), mult_q * xcsdbr));
    ckt.real(sp, dp, -mult_q.mul_add(-xcsdbi, mult_i * current.source[0]));
    ckt.imag(sp, gp, mult_i.mul_add(-Gmi, mult_q * xcsgbr));
    ckt.real(sp, gp, -mult_q.mul_add(-xcsgbi, mult_i * current.source[1]));
    ckt.imag(sp, sp, mult_i.mul_add(gdsi + FwdSumi, mult_q * xcssbr));
    ckt.real(sp, sp, mult_q.mul_add(xcssbi, mult_i * current.source[2]));
    ckt.real(sp, s, -(mult_i * (gspr + gstot)));
    ckt.imag(sp, bp, mult_i.mul_add(-Gmbsi, mult_q * xcsbbr));
    ckt.real(sp, bp, -mult_q.mul_add(-xcsbbi, mult_i * current.source[3]));

    ckt.real(s, sp, -(mult_i * (gspr - gstots)));
    ckt.real(s, s, mult_i * (gspr + gstot));

    stamp_common::bulk::<Ac>(
        h,
        &substrate,
        &gate_current,
        [xcbdb, xcbgb, xcbsb, xcbbb],
        ckt,
    );
    stamp_common::body_network::<Ac>(h, [xcdbdb, xcsbsb], ckt);

    // WDLiu: The internal charge node generated for transient NQS is not
    // needed for AC NQS. The following is not doing a real job, but we have
    // to keep it; otherwise a singular AC NQS matrix may occur if the
    // transient NQS is on. The charge node is isolated from the instance.
    if h.trnqsMod != 0 {
        ckt.real(q, q, mm * 1.0);
        ckt.real(q, gp, 0.0);
        ckt.real(q, dp, 0.0);
        ckt.real(q, sp, 0.0);
        ckt.real(q, bp, 0.0);

        ckt.real(dp, q, 0.0);
        ckt.real(sp, q, 0.0);
        ckt.real(gp, q, 0.0);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::host::{SmallSignalCkt, node};
    use super::{Instance, Model, SizeParam};
    use std::collections::BTreeSet;

    /// Records every stamp, by kind.
    #[derive(Default)]
    pub(in crate::devices::bsim4) struct Recorder {
        pub(crate) omega: f64,
        pub(crate) state: Vec<f64>,
        pub(crate) real: Vec<(i32, i32, f64)>,
        pub(crate) imag: Vec<(i32, i32, f64)>,
        pub(crate) cap: Vec<(i32, i32, f64)>,
    }

    impl SmallSignalCkt for Recorder {
        fn omega(&self) -> f64 {
            self.omega
        }
        fn state0(&self, slot: usize) -> f64 {
            self.state.get(slot).copied().unwrap_or(0.0)
        }
        fn real(&mut self, row: i32, col: i32, value: f64) {
            self.real.push((row, col, value));
        }
        fn imag(&mut self, row: i32, col: i32, value: f64) {
            self.imag.push((row, col, value));
        }
        fn cap(&mut self, row: i32, col: i32, value: f64) {
            self.cap.push((row, col, value));
        }
    }

    impl Recorder {
        pub(crate) fn positions(&self) -> BTreeSet<(i32, i32)> {
            self.real
                .iter()
                .chain(&self.imag)
                .chain(&self.cap)
                .map(|&(r, c, _)| (r, c))
                .collect()
        }
    }

    /// An instance with node fields as `b4set.c` leaves them.
    pub(in crate::devices::bsim4) fn shaped(rgate: i32, rbody: i32, trnqs: i32) -> Instance {
        let created = rbody != 0;
        Instance {
            dNode: node::D,
            gNodeExt: node::G_EXT,
            sNode: node::S,
            bNode: node::B,
            dNodePrime: node::DP,
            sNodePrime: node::SP,
            gNodePrime: if rgate > 0 { node::GP } else { node::G_EXT },
            gNodeMid: if rgate == 3 { node::GM } else { node::G_EXT },
            dbNode: if created { node::DB } else { node::B },
            bNodePrime: if created { node::BP } else { node::B },
            sbNode: if created { node::SB } else { node::B },
            qNode: if trnqs != 0 { node::Q } else { node::GROUND },
            rgateMod: rgate,
            rbodyMod: rbody,
            trnqsMod: trnqs,
            m: 1.0,
            mult_i: 1.0,
            mult_q: 1.0,
            ..Instance::default()
        }
    }

    /// Every stamp lands on an element `b4set.c` allocates.
    pub(in crate::devices::bsim4) fn check_shapes(
        run: impl Fn(&Model, &Instance, &mut Recorder),
        expect_cap: bool,
        charge_node: bool,
    ) {
        for rgate in 0..=3 {
            for rbody in [0, 1] {
                for trnqs in [0, 1] {
                    for rds in [0, 1] {
                        let h = shaped(rgate, rbody, trnqs);
                        let m = Model {
                            rdsMod: rds,
                            ..Model::default()
                        };
                        let allowed: BTreeSet<_> = super::super::elements::matrix_elements(&m, &h)
                            .into_iter()
                            .collect();
                        let mut ckt = Recorder {
                            omega: 1e6,
                            ..Recorder::default()
                        };
                        run(&m, &h, &mut ckt);
                        let used = ckt.positions();
                        assert!(
                            used.is_subset(&allowed),
                            "rgate {rgate} rbody {rbody}: {:?}",
                            used.difference(&allowed).collect::<Vec<_>>()
                        );
                        assert_eq!(ckt.cap.is_empty(), !expect_cap);
                        assert_eq!(ckt.imag.is_empty(), expect_cap);
                        assert!(used.contains(&(h.gNodePrime, h.gNodePrime)));
                        assert_eq!(used.contains(&(node::DB, node::DB)), rbody != 0);
                        assert_eq!(
                            used.contains(&(node::Q, node::Q)),
                            charge_node && trnqs != 0
                        );
                        assert_eq!(used.contains(&(node::GM, node::GM)), rgate == 3);
                        assert_eq!(used.contains(&(h.dNode, h.gNodePrime)), rds != 0);
                    }
                }
            }
        }
    }

    #[test]
    fn ac_stamps_land_on_allocated_elements() {
        check_shapes(
            |m, h, ckt| super::ac_load(m, h, &SizeParam::default(), ckt),
            false,
            true,
        );
    }

    #[test]
    fn ac_nqs_relaxation_preserves_quadrature_across_square_overflow() {
        for x in [0., 1e-180, 0.25, 1., 2., 1e100, 1e155, 1e170, 1e250] {
            let (real, quadrature) = super::nqs_relaxation(x, 1.);
            assert!(real.is_finite() && quadrature.is_finite());
            let expected_quadrature = if x > 1e150 { 1. / x } else { x / (1. + x * x) };
            assert!(
                (quadrature - expected_quadrature).abs() <= expected_quadrature.abs() * 2e-15,
                "omega*tau={x}, got {quadrature}, expected {expected_quadrature}"
            );
            if x <= 1. {
                assert_eq!(real, 1. / (1. + x * x));
            }
        }
        let (real, imaginary) = super::nqs_relaxation(1e300, 1e20);
        assert_eq!(real, 0.);
        assert_eq!(imaginary, 1e-320);
        let (_, negative) = super::nqs_relaxation(-1e300, 1e20);
        assert_eq!(negative, -1e-320);
        let (_, invalid) = super::nqs_relaxation(f64::NAN, 1.);
        assert!(invalid.is_nan());
    }

    #[test]
    fn ac_nqs_high_frequency_capacitance_contribution_avoids_underflow() {
        let mut h = shaped(0, 0, 0);
        h.acnqsMod = 1;
        h.taunet = 1e-100;
        h.cddb = 1e-30;
        let mut ckt = Recorder {
            omega: 1e300,
            ..Recorder::default()
        };
        super::ac_load(&Model::default(), &h, &SizeParam::default(), &mut ckt);
        // omega*tau = 1e200: C's `(omega*tau)^2` overflows, but the
        // physically correct real NQS contribution remains ~1e70 S.
        let dp_dp = ckt
            .real
            .iter()
            .filter(|&&(r, c, _)| (r, c) == (h.dNodePrime, h.dNodePrime))
            .map(|entry| entry.2)
            .sum::<f64>();
        let want = 1e70;
        assert!(
            (dp_dp - want).abs() <= want.abs() * 2e-15,
            "high-frequency NQS stamp {dp_dp} vs {want}"
        );
        let sp_dp = ckt
            .real
            .iter()
            .filter(|&&(r, c, _)| (r, c) == (h.sNodePrime, h.dNodePrime))
            .map(|entry| entry.2)
            .sum::<f64>();
        assert!(
            (dp_dp + sp_dp).abs() <= want.abs() * 2e-15,
            "NQS drain/source column violates KCL: {dp_dp} + {sp_dp}"
        );
        assert!(ckt.real.iter().chain(&ckt.imag).all(|v| v.2.is_finite()));
    }

    #[test]
    fn ac_gate_capacitance_is_scaled_by_omega() {
        let mut h = shaped(0, 0, 0);
        h.cggb = 2e-15;
        h.cgdo = 1e-16;
        let mut ckt = Recorder {
            omega: 1e6,
            ..Recorder::default()
        };
        super::ac_load(&Model::default(), &h, &SizeParam::default(), &mut ckt);
        let gg = ckt
            .imag
            .iter()
            .find(|&&(r, c, _)| (r, c) == (h.gNodePrime, h.gNodePrime))
            .unwrap()
            .2;
        assert_eq!(gg, (2e-15 + 1e-16 + 0.0 + 0.0) * 1e6);
    }
    fn nonzero_small_signal_instance(rgate: i32, rbody: i32, trnqs: i32) -> Instance {
        let mut h = shaped(rgate, rbody, trnqs);
        h.gm = -0.3125;
        h.gmbs = -0.015625;
        h.gds = 0.28125;
        h.gbds = -0.0625;
        h.gbgs = 0.234375;
        h.gbbs = -0.109375;
        h.gIgsg = 0.1875;
        h.gIgcsg = -0.15625;
        h.gIgcsd = 0.140625;
        h.gIgss = -0.203125;
        h.gIgcss = 0.09375;
        h.gIgcsb = -0.25;
        h.gIgdg = 0.046875;
        h.gIgcdg = -0.296875;
        h.gIgdd = 0.0;
        h.gIgcdd = 0.296875;
        h.gIgcds = -0.046875;
        h.gIgcdb = 0.25;
        h.gIgbg = -0.09375;
        h.gIgbd = 0.203125;
        h.gIgbs = -0.140625;
        h.gIgbb = 0.15625;
        h.gcrg = -0.1875;
        h.gcrgd = 0.109375;
        h.gcrgg = -0.234375;
        h.gcrgs = 0.0625;
        h.gcrgb = -0.28125;
        h.gstot = 0.015625;
        h.gstotd = 0.3125;
        h.gstotg = -0.03125;
        h.gstots = 0.265625;
        h.gstotb = -0.078125;
        h.gdtot = 0.21875;
        h.gdtotd = -0.125;
        h.gdtotg = 0.171875;
        h.gdtots = -0.171875;
        h.gdtotb = 0.125;
        h.drainConductance = -0.21875;
        h.sourceConductance = 0.078125;
        h.gbd = -0.265625;
        h.gbs = 0.03125;
        h.grgeltd = -0.3125;
        h.grbpd = -0.015625;
        h.grbdb = 0.28125;
        h.grbpb = -0.0625;
        h.grbps = 0.234375;
        h.grbsb = -0.109375;
        h.ggidld = 0.1875;
        h.ggidlg = -0.15625;
        h.ggidlb = 0.140625;
        h.ggislg = -0.203125;
        h.ggisls = 0.09375;
        h.ggislb = -0.25;
        h.cddb = 0.046875;
        h.cdgb = -0.296875;
        h.cdsb = 0.0;
        h.cgdb = 0.296875;
        h.cggb = -0.046875;
        h.cgsb = 0.25;
        h.cbdb = -0.09375;
        h.cbgb = 0.203125;
        h.cbsb = -0.140625;
        h.cgdo = 0.15625;
        h.cgso = -0.1875;
        h.capbd = 0.109375;
        h.capbs = -0.234375;
        h.gtg = 0.0625;
        h.gtd = -0.28125;
        h.gts = 0.015625;
        h.gtb = 0.3125;
        h.cqgb = -0.03125;
        h.cqdb = 0.265625;
        h.cqsb = -0.078125;
        h.cqbb = 0.21875;
        h.gtau = -0.125;
        h.taunet = 0.03125;
        h.qgate = -0.75;
        h.qbulk = -0.125;
        h.qdrn = 0.375;
        h.m = 1.125;
        h.mult_i = 0.8125;
        h.mult_q = 1.375;
        h
    }

    #[test]
    fn small_signal_selector_matrix() {
        use super::super::host::state;
        use super::super::pzld::pz_load;
        use std::collections::BTreeMap;
        let p = SizeParam {
            cgbo: 0.0078125,
            weffCV: 1.,
            leffCV: 1.,
            ..SizeParam::default()
        };
        for rgate in 0..=3 {
            for rbody in 0..=2 {
                for rds in 0..=1 {
                    for igc in 0..=1 {
                        for igb in 0..=1 {
                            for nqs in 0..=1 {
                                for mode in [-1, 1] {
                                    let mut h = nonzero_small_signal_instance(rgate, rbody, nqs);
                                    h.mode = mode;
                                    h.acnqsMod = nqs;
                                    let m = Model {
                                        rdsMod: rds,
                                        igcMod: igc,
                                        igbMod: igb,
                                        coxe: 1.,
                                        xpart: 0.4,
                                        ..Model::default()
                                    };
                                    let mut ac = Recorder {
                                        omega: 13.25,
                                        state: vec![0.; state::COUNT],
                                        ..Recorder::default()
                                    };
                                    ac.state[state::VGES] = 0.3125;
                                    ac.state[state::VGMS] = 0.21875;
                                    ac.state[state::VGS] = 0.125;
                                    ac.state[state::QDEF] = 0.03125;
                                    let mut pz = Recorder {
                                        state: ac.state.clone(),
                                        ..Recorder::default()
                                    };
                                    super::ac_load(&m, &h, &p, &mut ac);
                                    pz_load(&m, &h, &p, &mut pz);
                                    // QS AC and the PZ pencil describe the same physical matrix.
                                    // Their reference arithmetic groups some terms differently.
                                    if nqs == 0 {
                                        let aggregate = |stamps: &[(i32, i32, f64)], scale: f64| {
                                            let mut result = BTreeMap::new();
                                            for &(r, c, v) in stamps {
                                                *result.entry((r, c)).or_insert(0.) += v * scale;
                                            }
                                            result
                                        };
                                        for (left, right) in [
                                            (aggregate(&ac.real, 1.), aggregate(&pz.real, 1.)),
                                            (aggregate(&ac.imag, 1.), aggregate(&pz.cap, ac.omega)),
                                        ] {
                                            for key in left.keys().chain(right.keys()) {
                                                let a = left.get(key).copied().unwrap_or(0.);
                                                let b = right.get(key).copied().unwrap_or(0.);
                                                assert!(
                                                    (a - b).abs()
                                                        <= 128.
                                                            * f64::EPSILON
                                                            * (1. + a.abs().max(b.abs())),
                                                    "gate {rgate} body {rbody} rds {rds} igc {igc} igb {igb} mode {mode} {key:?}: {a} {b}"
                                                );
                                            }
                                        }
                                    }
                                    // Ordered per-component fingerprints for the before/after comparison.
                                    for (kind, stamps) in [
                                        ("ac-real", &ac.real),
                                        ("ac-imag", &ac.imag),
                                        ("pz-real", &pz.real),
                                        ("pz-cap", &pz.cap),
                                    ] {
                                        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
                                        for &(row, col, value) in stamps {
                                            for word in [
                                                u64::from(u32::try_from(row).unwrap()),
                                                u64::from(u32::try_from(col).unwrap()),
                                                value.to_bits(),
                                            ] {
                                                hash = (hash ^ word).wrapping_mul(0x100_0000_01b3);
                                            }
                                        }
                                        println!(
                                            "stamp-matrix {rgate} {rbody} {rds} {igc} {igb} {nqs} {mode} {kind} {hash:016x}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
