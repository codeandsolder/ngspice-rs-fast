//! Ordered nonlinear load stamping from b4ld.c, line900.
#![expect(
    non_snake_case,
    reason = "C-translated equation symbols identify their upstream terms"
)]
#![expect(
    clippy::similar_names,
    reason = "C equation names preserve the reference distinctions"
)]
#![expect(
    unused_mut,
    reason = "reference scratch variables may be assigned on different branches"
)]
#![expect(
    clippy::too_many_lines,
    reason = "preserve cohesive C physics equation sequence"
)]
#![expect(
    clippy::cognitive_complexity,
    reason = "retain version-specific reference physics branches"
)]
use super::{Instance, LoadCkt, Model};
use super::{
    bias::LoadBias,
    charge::{CapacitanceStamp, ChargeCurrents, Companion, NqsStamp},
};

/// Ordered finite product sum with recovery for severe cancellation or
/// range loss. The Newton residual is the difference between a physical
/// current and its linearization; losing the small remainder changes the
/// RHS even when the individual branch current and slope are accurate.
#[inline]
fn residual<const N: usize>(terms: [(f64, f64); N]) -> f64 {
    let mut products = terms.into_iter().map(|(a, b)| a * b);
    let Some(first) = products.next() else {
        return 0.;
    };
    let (ordinary, magnitude) = products.fold((first, first.abs()), |(sum, magnitude), term| {
        (sum + term, magnitude + term.abs())
    });
    if ordinary.is_normal() && ordinary.abs() > 0.125 * magnitude {
        return ordinary;
    }
    if magnitude == 0. && terms.iter().all(|&(a, b)| a == 0. || b == 0.) {
        return ordinary;
    }
    recover_residual(terms, ordinary)
}

#[cold]
fn recover_residual<const N: usize>(terms: [(f64, f64); N], ordinary: f64) -> f64 {
    if !terms.iter().all(|&(a, b)| a.is_finite() && b.is_finite()) {
        return ordinary;
    }
    let mut accumulator = crate::primitives::ExactSum::new();
    for (a, b) in terms {
        accumulator.add_product(a, b);
    }
    let result = accumulator.rounded_sum();
    if result == 0. && ordinary == 0. {
        ordinary
    } else {
        result
    }
}

fn sum<const N: usize>(terms: [f64; N]) -> f64 {
    residual(terms.map(|term| (term, 1.)))
}

/// Emit C line900 RHS and matrix contributions without coalescing or reordering.
#[expect(
    clippy::suboptimal_flops,
    reason = "ordered ordinary matrix products retain C event and arithmetic grouping; cancellation-sensitive Newton residuals and RHS reductions use adaptive product sums"
)]
pub(super) fn apply(
    model: &Model,
    here: &Instance,
    ckt: &mut dyn LoadCkt,
    bias: &LoadBias,
    cdrain: f64,
    companion: &Companion,
) {
    let &LoadBias {
        qdef,
        vbd,
        vbd_jct,
        vbs,
        vbs_jct,
        vds,
        vgd,
        vges,
        vgms,
        vgs,
        ..
    } = bias;
    let &Companion {
        currents:
            ChargeCurrents {
                mut ceqqb,
                mut ceqqd,
                mut ceqqg,
                mut ceqqgmid,
                mut ceqqjd,
                mut ceqqjs,
            },
        capacitance:
            CapacitanceStamp {
                gcbbb,
                gcbdb,
                gcbgb,
                gcbgmb,
                gcbsb,
                gcdbb,
                gcdbdb,
                gcddb,
                gcdgb,
                gcdgmb,
                gcdsb,
                gcgbb,
                gcgdb,
                gcggb,
                gcgmbb,
                gcgmdb,
                gcgmgmb,
                gcgmsb,
                gcgsb,
                gcsbb,
                gcsbsb,
                gcsdb,
                gcsgb,
                gcsgmb,
                gcssb,
            },
        nqs:
            NqsStamp {
                mut cqcheq,
                mut cqdef,
                ddxpart_dVb,
                ddxpart_dVd,
                ddxpart_dVg,
                ddxpart_dVs,
                dsxpart_dVb,
                dsxpart_dVd,
                dsxpart_dVg,
                dsxpart_dVs,
                dxpart,
                gcqbb,
                gcqdb,
                gcqgb,
                gcqsb,
                ggtb,
                ggtd,
                ggtg,
                ggts,
                gqdef,
                sxpart,
            },
    } = companion;
    let mut ceqgstot: f64;
    let mut ceqgdtot: f64;
    let mut gstot: f64;
    let mut gstotd: f64;
    let mut gstotg: f64;
    let mut gstots: f64;
    let mut gstotb: f64;
    let mut gspr: f64;
    let mut gdtot: f64;
    let mut gdtotd: f64;
    let mut gdtotg: f64;
    let mut gdtots: f64;
    let mut gdtotb: f64;
    let mut gdpr: f64;
    let mut geltd: f64;
    let mut gcrg: f64;
    let mut gcrgg: f64;
    let mut gcrgd: f64;
    let mut gcrgs: f64;
    let mut gcrgb: f64;
    let mut ceqgcrg: f64;
    let mut ceqdrn: f64;
    let mut ceqbd: f64;
    let mut ceqbs: f64;
    let mut ceqjd: f64;
    let mut ceqjs: f64;
    let mut gjbd: f64;
    let mut gjbs: f64;
    let mut Gmbs: f64;
    let mut FwdSum: f64;
    let mut RevSum: f64;
    let mut Istoteq: f64;
    let mut gIstotg: f64;
    let mut gIstotd: f64;
    let mut gIstots: f64;
    let mut gIstotb: f64;
    let mut Idtoteq: f64;
    let mut gIdtotg: f64;
    let mut gIdtotd: f64;
    let mut gIdtots: f64;
    let mut gIdtotb: f64;
    let mut Ibtoteq: f64;
    let mut gIbtotg: f64;
    let mut gIbtotd: f64;
    let mut gIbtots: f64;
    let mut gIbtotb: f64;
    let mut Igtoteq: f64;
    let mut gIgtotg: f64;
    let mut gIgtotd: f64;
    let mut gIgtots: f64;
    let mut gIgtotb: f64;
    let mut T1: f64;
    let mut Gm: f64;
    let mut gbspsp: f64;
    let mut gbbdp: f64;
    let mut gbbsp: f64;
    let mut gbspg: f64;
    let mut gbspb: f64;
    let mut gbspdp: f64;
    let mut gbdpdp: f64;
    let mut gbdpg: f64;
    let mut gbdpb: f64;
    let mut gbdpsp: f64;
    let mut m: f64;
    let mut mult_i: f64;
    let mut mult_q: f64;
    // C label `line900:`.
    if here.mode >= 0 {
        Gm = here.gm;
        Gmbs = here.gmbs;
        FwdSum = Gm + Gmbs;
        RevSum = 0.0;
        ceqdrn = f64::from(model.r#type)
            * residual([(cdrain, 1.), (-here.gds, vds), (-Gm, vgs), (-Gmbs, vbs)]);
        ceqbd = f64::from(model.r#type)
            * residual([
                (here.csub, 1.),
                (here.Igidl, 1.),
                (-(here.gbds + here.ggidld), vds),
                (-(here.gbgs + here.ggidlg), vgs),
                (-(here.gbbs + here.ggidlb), vbs),
            ]);
        ceqbs = f64::from(model.r#type)
            * residual([
                (here.Igisl, 1.),
                (here.ggisls, vds),
                (-here.ggislg, vgd),
                (-here.ggislb, vbd),
            ]);
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
                * residual([
                    (here.Igs, 1.),
                    (here.Igcs, 1.),
                    (-gIstotg, vgs),
                    (-here.gIgcsd, vds),
                    (-here.gIgcsb, vbs),
                ]);
            gIdtotg = here.gIgdg + here.gIgcdg;
            gIdtotd = here.gIgdd + here.gIgcdd;
            gIdtots = here.gIgcds;
            gIdtotb = here.gIgcdb;
            Idtoteq = f64::from(model.r#type)
                * residual([
                    (here.Igd, 1.),
                    (here.Igcd, 1.),
                    (-here.gIgdg, vgd),
                    (-here.gIgcdg, vgs),
                    (-here.gIgcdd, vds),
                    (-here.gIgcdb, vbs),
                ]);
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
                * residual([
                    (here.Igb, 1.),
                    (-here.gIgbg, vgs),
                    (-here.gIgbd, vds),
                    (-here.gIgbb, vbs),
                ]);
        } else {
            Ibtoteq = 0.0;
            gIbtotb = Ibtoteq;
            gIbtots = gIbtotb;
            gIbtotd = gIbtots;
            gIbtotg = gIbtotd;
        }
        if (model.igcMod != 0) || (model.igbMod != 0) {
            gIgtotg = sum([gIstotg, gIdtotg, gIbtotg]);
            gIgtotd = sum([gIstotd, gIdtotd, gIbtotd]);
            gIgtots = sum([gIstots, gIdtots, gIbtots]);
            gIgtotb = sum([gIstotb, gIdtotb, gIbtotb]);
            Igtoteq = sum([Istoteq, Idtoteq, Ibtoteq]);
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
            ceqgcrg = -residual([(gcrgd, vds), (gcrgg, vgs), (gcrgb, vbs)]);
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
        ceqdrn = f64::from(-model.r#type)
            * residual([(cdrain, 1.), (here.gds, vds), (Gm, vgd), (Gmbs, vbd)]);
        ceqbs = f64::from(model.r#type)
            * residual([
                (here.csub, 1.),
                (here.Igisl, 1.),
                ((here.gbds + here.ggisls), vds),
                (-(here.gbgs + here.ggislg), vgd),
                (-(here.gbbs + here.ggislb), vbd),
            ]);
        ceqbd = f64::from(model.r#type)
            * residual([
                (here.Igidl, 1.),
                (-here.ggidld, vds),
                (-here.ggidlg, vgs),
                (-here.ggidlb, vbs),
            ]);
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
                * residual([
                    (here.Igs, 1.),
                    (here.Igcd, 1.),
                    (-here.gIgsg, vgs),
                    (-here.gIgcdg, vgd),
                    (here.gIgcdd, vds),
                    (-here.gIgcdb, vbd),
                ]);
            gIdtotg = here.gIgdg + here.gIgcsg;
            gIdtotd = here.gIgdd + here.gIgcss;
            gIdtots = here.gIgcsd;
            gIdtotb = here.gIgcsb;
            Idtoteq = f64::from(model.r#type)
                * residual([
                    (here.Igd, 1.),
                    (here.Igcs, 1.),
                    (-(here.gIgdg + here.gIgcsg), vgd),
                    (here.gIgcsd, vds),
                    (-here.gIgcsb, vbd),
                ]);
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
                * residual([
                    (here.Igb, 1.),
                    (-here.gIgbg, vgd),
                    (here.gIgbd, vds),
                    (-here.gIgbb, vbd),
                ]);
        } else {
            Ibtoteq = 0.0;
            gIbtotb = Ibtoteq;
            gIbtots = gIbtotb;
            gIbtotd = gIbtots;
            gIbtotg = gIbtotd;
        }
        if (model.igcMod != 0) || (model.igbMod != 0) {
            gIgtotg = sum([gIstotg, gIdtotg, gIbtotg]);
            gIgtotd = sum([gIstotd, gIdtotd, gIbtotd]);
            gIgtots = sum([gIstots, gIdtots, gIbtots]);
            gIgtotb = sum([gIstotb, gIdtotb, gIbtotb]);
            Igtoteq = sum([Istoteq, Idtoteq, Ibtoteq]);
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
            ceqgcrg = -residual([(gcrgg, vgd), (-gcrgs, vds), (gcrgb, vbd)]);
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
            * residual([(here.gstotd, vds), (here.gstotg, vgs), (here.gstotb, vbs)]);
        gstot = here.gstot;
        gstotd = here.gstotd;
        gstotg = here.gstotg;
        gstots = here.gstots - gstot;
        gstotb = here.gstotb;
        ceqgdtot = f64::from(-model.r#type)
            * residual([(here.gdtotd, vds), (here.gdtotg, vgs), (here.gdtotb, vbs)]);
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
        ceqjs = residual([(here.cbs, 1.), (-here.gbs, vbs_jct)]);
        ceqjd = residual([(here.cbd, 1.), (-here.gbd, vbd_jct)]);
    } else {
        ceqjs = -residual([(here.cbs, 1.), (-here.gbs, vbs_jct)]);
        ceqjd = -residual([(here.cbd, 1.), (-here.gbd, vbd_jct)]);
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
        residual([
            (mult_i, sum([ceqjd, -ceqbd, ceqgdtot, -ceqdrn, Idtoteq])),
            (-mult_q, ceqqd),
        ]),
    );
    ckt.rhs(
        here.gNodePrime,
        residual([(-mult_q, ceqqg), (mult_i, sum([ceqgcrg, -Igtoteq]))]),
    );
    if here.rgateMod == 2 {
        ckt.rhs(here.gNodeExt, -(mult_i * ceqgcrg));
    } else if here.rgateMod == 3 {
        ckt.rhs(
            here.gNodeMid,
            -residual([(mult_q, ceqqgmid), (mult_i, ceqgcrg)]),
        );
    }
    if here.rbodyMod == 0 {
        ckt.rhs(
            here.bNodePrime,
            residual([
                (mult_i, sum([ceqbd, ceqbs, -ceqjd, -ceqjs, Ibtoteq])),
                (-mult_q, ceqqb),
            ]),
        );
        ckt.rhs(
            here.sNodePrime,
            residual([
                (mult_i, sum([ceqdrn, -ceqbs, ceqjs, -ceqgstot, Istoteq])),
                (mult_q, sum([ceqqg, ceqqb, ceqqd, ceqqgmid])),
            ]),
        );
    } else {
        ckt.rhs(here.dbNode, -residual([(mult_i, ceqjd), (mult_q, ceqqjd)]));
        ckt.rhs(
            here.bNodePrime,
            residual([(mult_i, sum([ceqbd, ceqbs, Ibtoteq])), (-mult_q, ceqqb)]),
        );
        ckt.rhs(here.sbNode, -residual([(mult_i, ceqjs), (mult_q, ceqqjs)]));
        ckt.rhs(
            here.sNodePrime,
            residual([
                (mult_i, sum([ceqdrn, -ceqbs, ceqjs, -ceqgstot, Istoteq])),
                (mult_q, sum([ceqqd, ceqqg, ceqqb, ceqqjd, ceqqjs, ceqqgmid])),
            ]),
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
    super::super::stamp_common::leakage(here, mult_i, |row, col, value| {
        ckt.matrix(row, col, value);
    });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonlinear_residual_recovers_cancellation_and_product_rounding() {
        assert_eq!(sum([1e16, 1., -1e16]), 1.);
        let epsilon = f64::EPSILON;
        assert_eq!(
            residual([(-1., 1.), (1. + epsilon, 1. - epsilon)]),
            -epsilon * epsilon
        );
        assert_eq!(residual([(1e308, 2.), (-1e308, 2.), (1., 1.)]), 1.);
        assert_eq!(sum([-0., -0.]).to_bits(), (-0_f64).to_bits());
        assert!(residual([(f64::INFINITY, 0.), (1., 1.)]).is_nan());
        assert_eq!(residual([(f64::INFINITY, 1.), (1., 1.)]), f64::INFINITY);
    }
}
