//! Shared BSIM4 small-signal conductance and topology stamps, plus nonlinear leakage.
//! Keep additions in reference order even when local nodes alias.
#![expect(
    non_snake_case,
    reason = "preserve upstream C derivative identifiers for equation audit"
)]
#![expect(
    clippy::similar_names,
    reason = "retain distinct C drain/source/gate derivative names"
)]
use super::fields::{Instance, Model};
use super::host::{SmallSignalCkt, state};

/// GIDL followed by GISL; the sum and multiplication order matches C.
pub(super) fn leakage(h: &Instance, mult_i: f64, mut add: impl FnMut(i32, i32, f64)) {
    let (dp, gp, sp, bp) = (h.dNodePrime, h.gNodePrime, h.sNodePrime, h.bNodePrime);
    let ggidld = h.ggidld;
    let ggidlg = h.ggidlg;
    let ggidlb = h.ggidlb;
    let ggislg = h.ggislg;
    let ggisls = h.ggisls;
    let ggislb = h.ggislb;

    // stamp gidl
    add(dp, dp, mult_i * ggidld);
    add(dp, gp, mult_i * ggidlg);
    add(dp, sp, -(mult_i * ((ggidlg + ggidld) + ggidlb)));
    add(dp, bp, mult_i * ggidlb);
    add(bp, dp, -(mult_i * ggidld));
    add(bp, gp, -(mult_i * ggidlg));
    add(bp, sp, mult_i * ((ggidlg + ggidld) + ggidlb));
    add(bp, bp, -(mult_i * ggidlb));
    // stamp gisl
    add(sp, dp, -(mult_i * ((ggisls + ggislg) + ggislb)));
    add(sp, gp, mult_i * ggislg);
    add(sp, sp, mult_i * ggisls);
    add(sp, bp, mult_i * ggislb);
    add(bp, dp, mult_i * ((ggislg + ggisls) + ggislb));
    add(bp, gp, -(mult_i * ggislg));
    add(bp, sp, -(mult_i * ggisls));
    add(bp, bp, -(mult_i * ggislb));
}

#[derive(Clone, Copy)]
pub(super) struct SubstrateCurrent {
    pub(super) gbbdp: f64,
    pub(super) gbbsp: f64,
    pub(super) gbdpg: f64,
    pub(super) gbdpdp: f64,
    pub(super) gbdpb: f64,
    pub(super) gbdpsp: f64,
    pub(super) gbspdp: f64,
    pub(super) gbspg: f64,
    pub(super) gbspb: f64,
    pub(super) gbspsp: f64,
}

impl SubstrateCurrent {
    #[inline]
    pub(super) const fn new(h: &Instance) -> Self {
        let (gbbdp, gbbsp, gbdpg, gbdpdp, gbdpb, gbdpsp, gbspdp, gbspg, gbspb, gbspsp);
        if h.mode >= 0 {
            gbbdp = -(h.gbds);
            gbbsp = h.gbds + h.gbgs + h.gbbs;
            gbdpg = h.gbgs;
            gbdpdp = h.gbds;
            gbdpb = h.gbbs;
            gbdpsp = -(gbdpg + gbdpdp + gbdpb);

            gbspdp = 0.0;
            gbspg = 0.0;
            gbspb = 0.0;
            gbspsp = 0.0;
        } else {
            gbbsp = -(h.gbds);
            gbbdp = h.gbds + h.gbgs + h.gbbs;

            gbdpg = 0.0;
            gbdpsp = 0.0;
            gbdpb = 0.0;
            gbdpdp = 0.0;

            gbspg = h.gbgs;
            gbspsp = h.gbds;
            gbspb = h.gbbs;
            gbspdp = -(gbspg + gbspsp + gbspb);
        }
        Self {
            gbbdp,
            gbbsp,
            gbdpg,
            gbdpdp,
            gbdpb,
            gbdpsp,
            gbspdp,
            gbspg,
            gbspb,
            gbspsp,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct GateCurrent {
    pub(super) gIstotg: f64,
    pub(super) gIstotd: f64,
    pub(super) gIstots: f64,
    pub(super) gIstotb: f64,
    pub(super) gIdtotg: f64,
    pub(super) gIdtotd: f64,
    pub(super) gIdtots: f64,
    pub(super) gIdtotb: f64,
    pub(super) gIbtotg: f64,
    pub(super) gIbtotd: f64,
    pub(super) gIbtots: f64,
    pub(super) gIbtotb: f64,
    pub(super) gIgtotg: f64,
    pub(super) gIgtotd: f64,
    pub(super) gIgtots: f64,
    pub(super) gIgtotb: f64,
}

impl GateCurrent {
    #[inline]
    pub(super) const fn new(m: &Model, h: &Instance) -> Self {
        let (
            gIstotg,
            gIstotd,
            gIstots,
            gIstotb,
            gIdtotg,
            gIdtotd,
            gIdtots,
            gIdtotb,
            gIbtotg,
            gIbtotd,
            gIbtots,
            gIbtotb,
            gIgtotg,
            gIgtotd,
            gIgtots,
            gIgtotb,
        );
        if m.igcMod == 0 {
            (gIstotg, gIstotd, gIstots, gIstotb) = (0., 0., 0., 0.);
            (gIdtotg, gIdtotd, gIdtots, gIdtotb) = (0., 0., 0., 0.);
        } else if h.mode >= 0 {
            gIstotg = h.gIgsg + h.gIgcsg;
            gIstotd = h.gIgcsd;
            gIstots = h.gIgss + h.gIgcss;
            gIstotb = h.gIgcsb;
            gIdtotg = h.gIgdg + h.gIgcdg;
            gIdtotd = h.gIgdd + h.gIgcdd;
            gIdtots = h.gIgcds;
            gIdtotb = h.gIgcdb;
        } else {
            gIstotg = h.gIgsg + h.gIgcdg;
            gIstotd = h.gIgcds;
            gIstots = h.gIgss + h.gIgcdd;
            gIstotb = h.gIgcdb;
            gIdtotg = h.gIgdg + h.gIgcsg;
            gIdtotd = h.gIgdd + h.gIgcss;
            gIdtots = h.gIgcsd;
            gIdtotb = h.gIgcsb;
        }
        if m.igbMod != 0 {
            gIbtotg = h.gIgbg;
            (gIbtotd, gIbtots) = if h.mode >= 0 {
                (h.gIgbd, h.gIgbs)
            } else {
                (h.gIgbs, h.gIgbd)
            };
            gIbtotb = h.gIgbb;
        } else {
            (gIbtotg, gIbtotd, gIbtots, gIbtotb) = (0., 0., 0., 0.);
        }
        if m.igcMod != 0 || m.igbMod != 0 {
            gIgtotg = gIstotg + gIdtotg + gIbtotg;
            gIgtotd = gIstotd + gIdtotd + gIbtotd;
            gIgtots = gIstots + gIdtots + gIbtots;
            gIgtotb = gIstotb + gIdtotb + gIbtotb;
        } else {
            (gIgtotg, gIgtotd, gIgtots, gIgtotb) = (0., 0., 0., 0.);
        }
        Self {
            gIstotg,
            gIstotd,
            gIstots,
            gIstotb,
            gIdtotg,
            gIdtotd,
            gIdtots,
            gIdtotb,
            gIbtotg,
            gIbtotd,
            gIbtots,
            gIbtotb,
            gIgtotg,
            gIgtotd,
            gIgtots,
            gIgtotb,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct GateResistance {
    pub(super) gcrg: f64,
    pub(super) gcrgd: f64,
    pub(super) gcrgg: f64,
    pub(super) gcrgs: f64,
    pub(super) gcrgb: f64,
}

impl GateResistance {
    #[inline]
    pub(super) fn new(h: &Instance, ckt: &dyn SmallSignalCkt) -> Self {
        let voltage = match h.rgateMod {
            2 => ckt.state0(state::VGES) - ckt.state0(state::VGS),
            3 => ckt.state0(state::VGMS) - ckt.state0(state::VGS),
            _ => 0.,
        };
        if h.rgateMod <= 1 {
            return Self {
                gcrg: 0.,
                gcrgd: 0.,
                gcrgg: 0.,
                gcrgs: 0.,
                gcrgb: 0.,
            };
        }
        let (drain, source) = if h.mode >= 0 {
            (h.gcrgd, h.gcrgs)
        } else {
            (h.gcrgs, h.gcrgd)
        };
        Self {
            gcrg: h.gcrg,
            gcrgd: drain * voltage,
            gcrgg: h.gcrgg.mul_add(voltage, -h.gcrg),
            gcrgs: source * voltage,
            gcrgb: h.gcrgb * voltage,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct SeriesResistance {
    pub(super) gstot: f64,
    pub(super) gstotd: f64,
    pub(super) gstotg: f64,
    pub(super) gstots: f64,
    pub(super) gstotb: f64,
    pub(super) gdtot: f64,
    pub(super) gdtotd: f64,
    pub(super) gdtotg: f64,
    pub(super) gdtots: f64,
    pub(super) gdtotb: f64,
    pub(super) gdpr: f64,
    pub(super) gspr: f64,
}

impl SeriesResistance {
    #[inline]
    pub(super) const fn new(m: &Model, h: &Instance) -> Self {
        let (gstot, gstotd, gstotg, gstots, gstotb);
        let (gdtot, gdtotd, gdtotg, gdtots, gdtotb);
        if m.rdsMod == 1 {
            gstot = h.gstot;
            gstotd = h.gstotd;
            gstotg = h.gstotg;
            gstots = h.gstots - gstot;
            gstotb = h.gstotb;

            gdtot = h.gdtot;
            gdtotd = h.gdtotd - gdtot;
            gdtotg = h.gdtotg;
            gdtots = h.gdtots;
            gdtotb = h.gdtotb;
        } else {
            (gstot, gstotd, gstotg, gstots, gstotb) = (0.0, 0.0, 0.0, 0.0, 0.0);
            (gdtot, gdtotd, gdtotg, gdtots, gdtotb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let (gdpr, gspr) = if m.rdsMod == 0 {
            (h.drainConductance, h.sourceConductance)
        } else {
            (0.0, 0.0)
        };

        Self {
            gstot,
            gstotd,
            gstotg,
            gstots,
            gstotb,
            gdtot,
            gdtotd,
            gdtotg,
            gdtots,
            gdtotb,
            gdpr,
            gspr,
        }
    }
}

/// AC receives already omega-scaled charges, whereas PZ receives the
/// coefficient of s. Static dispatch eliminates a branch per stamp.
pub(super) trait ChargeProjection {
    const PENCIL: bool;
    fn stamp(ckt: &mut dyn SmallSignalCkt, row: i32, col: i32, value: f64);
    fn prescale(value: f64, omega: f64) -> f64;
}

pub(super) struct Ac;
impl ChargeProjection for Ac {
    const PENCIL: bool = false;
    #[inline]
    fn stamp(ckt: &mut dyn SmallSignalCkt, row: i32, col: i32, value: f64) {
        ckt.imag(row, col, value);
    }
    #[inline]
    fn prescale(value: f64, omega: f64) -> f64 {
        value * omega
    }
}

pub(super) struct Pencil;
impl ChargeProjection for Pencil {
    const PENCIL: bool = true;
    #[inline]
    fn stamp(ckt: &mut dyn SmallSignalCkt, row: i32, col: i32, value: f64) {
        ckt.cap(row, col, value);
    }
    #[inline]
    fn prescale(value: f64, _omega: f64) -> f64 {
        value
    }
}

/// Symmetric middle-gate overlap charge. The PZ NQS state has a
/// different charge model and does not use these coefficients.
#[derive(Clone, Copy, Default)]
pub(super) struct MidGateCharge {
    pub(super) diagonal: f64,
    pub(super) drain: f64,
    pub(super) source: f64,
    pub(super) bulk: f64,
}

impl MidGateCharge {
    #[inline]
    pub(super) fn new<P: ChargeProjection>(
        h: &Instance,
        p: &super::fields::SizeParam,
        omega: f64,
    ) -> Self {
        if h.rgateMod != 3 || (P::PENCIL && h.acnqsMod != 0) {
            return Self::default();
        }
        Self {
            diagonal: P::prescale(h.cgdo + h.cgso + p.cgbo, omega),
            drain: P::prescale(-h.cgdo, omega),
            source: P::prescale(-h.cgso, omega),
            bulk: P::prescale(-p.cgbo, omega),
        }
    }
}

/// The gate charge row and the optional middle-gate overlap network.
/// Prime terminal order is gate, drain, source, bulk.
pub(super) struct GateStamp {
    pub(super) real: [f64; 4],
    pub(super) charge: [f64; 4],
    pub(super) mid: MidGateCharge,
}

impl GateStamp {
    #[inline]
    pub(super) fn emit<P: ChargeProjection>(
        self,
        h: &Instance,
        resistance: &GateResistance,
        ckt: &mut dyn SmallSignalCkt,
    ) {
        let (dp, gp, gm, sp, bp) = (
            h.dNodePrime,
            h.gNodePrime,
            h.gNodeMid,
            h.sNodePrime,
            h.bNodePrime,
        );
        let ge = h.gNodeExt;
        let mult_i = h.mult_i * h.m;
        let mult_q = h.mult_q * h.m;
        let geltd = h.grgeltd;
        let GateResistance {
            gcrg,
            gcrgd,
            gcrgg,
            gcrgs,
            gcrgb,
        } = *resistance;
        match h.rgateMod {
            1 => {
                ckt.real(ge, ge, mult_i * geltd);
                ckt.real(gp, ge, -(mult_i * geltd));
                ckt.real(ge, gp, -(mult_i * geltd));
            }
            2 => {
                ckt.real(ge, ge, mult_i * gcrg);
                ckt.real(ge, gp, mult_i * gcrgg);
                ckt.real(ge, dp, mult_i * gcrgd);
                ckt.real(ge, sp, mult_i * gcrgs);
                ckt.real(ge, bp, mult_i * gcrgb);
                ckt.real(gp, ge, -(mult_i * gcrg));
            }
            3 => {
                ckt.real(ge, ge, mult_i * geltd);
                ckt.real(ge, gm, -(mult_i * geltd));
                ckt.real(gm, ge, -(mult_i * geltd));
                ckt.real(gm, gm, mult_i * (geltd + gcrg));
                P::stamp(ckt, gm, gm, mult_q * self.mid.diagonal);

                ckt.real(gm, dp, mult_i * gcrgd);
                P::stamp(ckt, gm, dp, mult_q * self.mid.drain);
                ckt.real(gm, gp, mult_i * gcrgg);
                ckt.real(gm, sp, mult_i * gcrgs);
                P::stamp(ckt, gm, sp, mult_q * self.mid.source);
                ckt.real(gm, bp, mult_i * gcrgb);
                P::stamp(ckt, gm, bp, mult_q * self.mid.bulk);

                P::stamp(ckt, dp, gm, mult_q * self.mid.drain);
                ckt.real(gp, gm, -(mult_i * gcrg));
                P::stamp(ckt, sp, gm, mult_q * self.mid.source);
                P::stamp(ckt, bp, gm, mult_q * self.mid.bulk);
            }
            _ => {}
        }
        // C PZ emits the drain column before the gate column without a
        // gate-resistance network. The middle-gate mode emits real first.
        if h.rgateMod == 3 {
            for (index, col) in [gp, dp, sp, bp].into_iter().enumerate() {
                ckt.real(gp, col, self.real[index]);
                P::stamp(ckt, gp, col, mult_q * self.charge[index]);
            }
        } else {
            let mut emit = |index: usize, col| {
                P::stamp(ckt, gp, col, mult_q * self.charge[index]);
                ckt.real(gp, col, self.real[index]);
            };
            if P::PENCIL && !matches!(h.rgateMod, 1..=3) {
                emit(1, dp);
                emit(0, gp);
            } else {
                emit(0, gp);
                emit(1, dp);
            }
            emit(2, sp);
            emit(3, bp);
        }
    }
}

impl SeriesResistance {
    #[inline]
    pub(super) fn emit_external(&self, m: &Model, h: &Instance, ckt: &mut dyn SmallSignalCkt) {
        let (d, s, gp, dp, sp, bp) = (
            h.dNode,
            h.sNode,
            h.gNodePrime,
            h.dNodePrime,
            h.sNodePrime,
            h.bNodePrime,
        );
        let mult_i = h.mult_i * h.m;
        let Self {
            gdtotg,
            gdtots,
            gdtotb,
            gstotd,
            gstotg,
            gstotb,
            ..
        } = *self;
        if m.rdsMod != 0 {
            ckt.real(d, gp, mult_i * gdtotg);
            ckt.real(d, sp, mult_i * gdtots);
            ckt.real(d, bp, mult_i * gdtotb);
            ckt.real(s, dp, mult_i * gstotd);
            ckt.real(s, gp, mult_i * gstotg);
            ckt.real(s, bp, mult_i * gstotb);
        }
    }
}

/// Bulk terminal contributions, then the shared GIDL/GISL rows.
#[inline]
pub(super) fn bulk<P: ChargeProjection>(
    h: &Instance,
    substrate: &SubstrateCurrent,
    gate: &GateCurrent,
    charge: [f64; 4],
    ckt: &mut dyn SmallSignalCkt,
) {
    let (bp, dp, gp, sp) = (h.bNodePrime, h.dNodePrime, h.gNodePrime, h.sNodePrime);
    let mult_i = h.mult_i * h.m;
    let mult_q = h.mult_q * h.m;
    let SubstrateCurrent { gbbdp, gbbsp, .. } = *substrate;
    let GateCurrent {
        gIbtotd,
        gIbtotg,
        gIbtots,
        gIbtotb,
        ..
    } = *gate;
    let (gjbd, gjbs) = if h.rbodyMod == 0 {
        (h.gbd, h.gbs)
    } else {
        (0., 0.)
    };
    let [xcbdb, xcbgb, xcbsb, xcbbb] = charge;
    P::stamp(ckt, bp, dp, mult_q * xcbdb);
    ckt.real(bp, dp, -(mult_i * (gjbd - gbbdp + gIbtotd)));
    P::stamp(ckt, bp, gp, mult_q * xcbgb);
    ckt.real(bp, gp, -(mult_i * (h.gbgs + gIbtotg)));
    P::stamp(ckt, bp, sp, mult_q * xcbsb);
    ckt.real(bp, sp, -(mult_i * (gjbs - gbbsp + gIbtots)));
    P::stamp(ckt, bp, bp, mult_q * xcbbb);
    ckt.real(bp, bp, mult_i * (gjbd + gjbs - h.gbbs - gIbtotb));
    leakage(h, mult_i, |row, col, value| ckt.real(row, col, value));
}

/// Distributed body resistances and the two junction charge branches.
#[inline]
pub(super) fn body_network<P: ChargeProjection>(
    h: &Instance,
    charge: [f64; 2],
    ckt: &mut dyn SmallSignalCkt,
) {
    let [xcdbdb, xcsbsb] = charge;
    let (dp, sp, bp, db, sb, b) = (
        h.dNodePrime,
        h.sNodePrime,
        h.bNodePrime,
        h.dbNode,
        h.sbNode,
        h.bNode,
    );
    let mult_i = h.mult_i * h.m;
    let mult_q = h.mult_q * h.m;
    if h.rbodyMod != 0 {
        P::stamp(ckt, dp, db, mult_q * xcdbdb);
        ckt.real(dp, db, -(mult_i * h.gbd));
        P::stamp(ckt, sp, sb, mult_q * xcsbsb);
        ckt.real(sp, sb, -(mult_i * h.gbs));

        P::stamp(ckt, db, dp, mult_q * xcdbdb);
        ckt.real(db, dp, -(mult_i * h.gbd));
        P::stamp(ckt, db, db, -(mult_q * xcdbdb));
        ckt.real(db, db, mult_i * (h.gbd + h.grbpd + h.grbdb));
        ckt.real(db, bp, -(mult_i * h.grbpd));
        ckt.real(db, b, -(mult_i * h.grbdb));

        ckt.real(bp, db, -(mult_i * h.grbpd));
        ckt.real(bp, b, -(mult_i * h.grbpb));
        ckt.real(bp, sb, -(mult_i * h.grbps));
        ckt.real(bp, bp, mult_i * (h.grbpd + h.grbps + h.grbpb));
        // WDLiu: (-here->BSIM4gbbs) already added to BPbpPtr

        P::stamp(ckt, sb, sp, mult_q * xcsbsb);
        ckt.real(sb, sp, -(mult_i * h.gbs));
        ckt.real(sb, bp, -(mult_i * h.grbps));
        ckt.real(sb, b, -(mult_i * h.grbsb));
        P::stamp(ckt, sb, sb, -(mult_q * xcsbsb));
        ckt.real(sb, sb, mult_i * (h.gbs + h.grbps + h.grbsb));

        ckt.real(b, db, -(mult_i * h.grbdb));
        ckt.real(b, bp, -(mult_i * h.grbpb));
        ckt.real(b, sb, -(mult_i * h.grbsb));
        ckt.real(b, b, mult_i * (h.grbsb + h.grbdb + h.grbpb));
    }
}

/// Oriented channel transconductances for either a real or quadrature gain.
#[derive(Clone, Copy)]
pub(super) struct ChannelGain {
    pub(super) gm: f64,
    pub(super) gmbs: f64,
    pub(super) forward: f64,
    pub(super) reverse: f64,
}

impl ChannelGain {
    #[inline]
    pub(super) const fn new(mode: i32, gm: f64, gmbs: f64) -> Self {
        if mode >= 0 {
            Self {
                gm,
                gmbs,
                forward: gm + gmbs,
                reverse: 0.,
            }
        } else {
            let gm = -gm;
            let gmbs = -gmbs;
            Self {
                gm,
                gmbs,
                forward: 0.,
                reverse: -(gm + gmbs),
            }
        }
    }
}

/// Base current terms before the analysis-specific NQS corrections.
/// The sign of each expression follows the enclosing reference FMA;
/// off-diagonal expressions with a leading minus are not negated here.
/// Terminal order within each row: drain, gate, source, bulk.
pub(super) struct ChannelCurrent {
    pub(super) drain: [f64; 4],
    pub(super) source: [f64; 4],
}

impl ChannelCurrent {
    #[inline]
    pub(super) const fn new<P: ChargeProjection>(
        h: &Instance,
        gain: ChannelGain,
        gds: f64,
        substrate: &SubstrateCurrent,
        gate: &GateCurrent,
        series: &SeriesResistance,
    ) -> Self {
        let ChannelGain {
            gm,
            gmbs,
            forward,
            reverse,
        } = gain;
        let SubstrateCurrent {
            gbdpg,
            gbdpdp,
            gbdpb,
            gbdpsp,
            gbspdp,
            gbspg,
            gbspb,
            gbspsp,
            ..
        } = *substrate;
        let GateCurrent {
            gIstotg,
            gIstotd,
            gIstots,
            gIstotb,
            gIdtotg,
            gIdtotd,
            gIdtots,
            gIdtotb,
            ..
        } = *gate;
        let SeriesResistance {
            gstotd,
            gstotg,
            gstots,
            gstotb,
            gdtotd,
            gdtotg,
            gdtots,
            gdtotb,
            gdpr,
            gspr,
            ..
        } = *series;
        let (gjbd, gjbs) = if h.rbodyMod == 0 {
            (h.gbd, h.gbs)
        } else {
            (0., 0.)
        };
        let source_diagonal = if P::PENCIL {
            gspr + gds + h.gbs - gIstots - gstots + forward + gbspsp
        } else {
            gspr + gds + h.gbs - gstots + forward + gbspsp - gIstots
        };
        Self {
            drain: [
                gdpr + gds + h.gbd - gdtotd + reverse + gbdpdp - gIdtotd,
                gm - gdtotg + gbdpg - gIdtotg,
                gds + forward + gdtots - gbdpsp + gIdtots,
                gjbd + gdtotb - gmbs - gbdpb + gIdtotb,
            ],
            source: [
                gds + gstotd + reverse - gbspdp + gIstotd,
                gm + gstotg - gbspg + gIstotg,
                source_diagonal,
                gjbs + gstotb + gmbs - gbspb + gIstotb,
            ],
        }
    }
}
