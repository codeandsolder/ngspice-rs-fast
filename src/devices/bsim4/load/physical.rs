//! Physical quasi-static branch incidence, independent of companion integration.
//! All constitutive laws are shared with the reference DC/transient load.
use super::{Instance, Model, SizeParam, channel, charge, junction, leakage, state};
use crate::devices::bsim_physics::overlap;
use crate::maths::taylor::Constitutive;
use crate::primitives::{Real, SpiceError, SpiceResult};

pub(in crate::devices::bsim4) fn require_quasi_static(here: &Instance) -> SpiceResult<()> {
    if here.trnqsMod != 0 {
        return Err(SpiceError::Unsupported {
            feature: "BSIM4 physical non-quasi-static charge-deficit equation".to_owned(),
            location: None,
        });
    }
    Ok(())
}

/// Independent physical histories are the node charges stored by the
/// reference load, not separate copies of the junction charge on both sides.
/// QD includes the negative drain junction charge even with body resistance.
pub(in crate::devices::bsim4) fn charge_layout(
    here: &Instance,
) -> impl Iterator<Item = (usize, [i32; 2])> {
    let source = here.sNodePrime;
    [
        Some((state::QB, [here.bNodePrime, source])),
        Some((state::QG, [here.gNodePrime, source])),
        Some((state::QD, [here.dNodePrime, source])),
        (here.rgateMod == 3).then_some((state::QGMID, [here.gNodeMid, source])),
        (here.rbodyMod != 0).then_some((state::QBS, [here.sbNode, source])),
        (here.rbodyMod != 0).then_some((state::QBD, [here.dbNode, source])),
    ]
    .into_iter()
    .flatten()
}

pub(in crate::devices::bsim4) struct PhysicalPoint<S> {
    pub(in crate::devices::bsim4) currents: Vec<([i32; 2], S)>,
    pub(in crate::devices::bsim4) charges: Vec<(usize, [i32; 2], S)>,
    pub(in crate::devices::bsim4) auxiliary: Vec<(usize, Real)>,
}

/// Evaluate at polarity-normalized voltages relative to source prime.
/// The caller maps aliases and applies current and charge multiplicities.
pub(in crate::devices::bsim4) fn evaluate<S: Constitutive>(
    model: &Model,
    here: &mut Instance,
    size: &SizeParam,
    temperature: Real,
    gmin: Real,
    voltage: impl Fn(i32) -> S,
) -> SpiceResult<PhysicalPoint<S>> {
    require_quasi_static(here)?;
    let d = here.dNodePrime;
    let g = here.gNodePrime;
    let s = here.sNodePrime;
    let b = here.bNodePrime;
    let de = here.dNode;
    let ge = here.gNodeExt;
    let se = here.sNode;
    let be = here.bNode;
    let gm = here.gNodeMid;
    let db = if here.rbodyMod == 0 { b } else { here.dbNode };
    let sb = if here.rbodyMod == 0 { b } else { here.sbNode };
    let drop = |a, b| voltage(a).add(&voltage(b).scale(-1.));
    let vd = drop(d, s);
    let vg = drop(g, s);
    let vb = drop(b, s);
    let vdes = drop(de, s);
    let vses = drop(se, s);
    let junction_bias = junction::JunctionBias {
        source: drop(sb, s),
        drain: drop(db, d),
    };
    let [ibs, ibd] =
        junction::current_constitutive(model, here, size, gmin, junction_bias.clone())?;
    let electro = channel::electrostatics_constitutive(
        model,
        here,
        size,
        temperature,
        vb.clone(),
        vg.clone(),
        vd.clone(),
    )?;
    let transport = channel::transport_constitutive(
        model,
        here,
        size,
        [
            vb.clone(),
            vg.clone(),
            vd.clone(),
            vdes.clone(),
            vses.clone(),
        ],
        &electro,
    )?;
    let currents = leakage::evaluate_constitutive(
        model,
        here,
        size,
        [vb.clone(), vg.clone(), vd.clone()],
        &electro,
        &transport,
    )?;
    let intrinsic = charge::intrinsic_constitutive(model, here, size, true, &electro, &transport)?;
    let [qbs, qbd] = junction::charge_constitutive(model, here, size, junction_bias, |_, _| {})?;
    let (fd, fs) = if here.mode > 0 { (d, s) } else { (s, d) };
    let mut branches = vec![
        ([fd, fs], currents.channel),
        ([fd, b], currents.substrate),
        ([d, b], currents.gidl),
        ([s, b], currents.gisl),
        ([g, fs], currents.igcs),
        ([g, fd], currents.igcd),
        ([g, s], currents.igs),
        ([g, d], currents.igd),
        ([g, b], currents.igb),
        ([sb, s], ibs),
        ([db, d], ibd),
    ];
    let mut resistor = |a, b, conductance: S| {
        if a != b {
            branches.push(([a, b], drop(a, b).multiply(&conductance)));
        }
    };
    let constant = |value| vd.constant(value);
    if model.rdsMod == 0 {
        resistor(de, d, constant(here.drainConductance));
        resistor(se, s, constant(here.sourceConductance));
    } else {
        resistor(de, d, transport.drain_conductance);
        resistor(se, s, transport.source_conductance);
    }
    match here.rgateMod {
        1 => resistor(ge, g, constant(here.grgeltd)),
        2 => resistor(ge, g, transport.gate_conductance),
        3 => {
            resistor(ge, gm, constant(here.grgeltd));
            resistor(gm, g, transport.gate_conductance);
        }
        _ => {}
    }
    if here.rbodyMod != 0 {
        for (a, b, conductance) in [
            (db, b, here.grbpd),
            (sb, b, here.grbps),
            (b, be, here.grbpb),
            (db, be, here.grbdb),
            (sb, be, here.grbsb),
        ] {
            resistor(a, b, constant(conductance));
        }
    }

    // Intrinsic charges are oriented by the channel, whereas overlap
    // and junction charges are attached to the physical terminal sides.
    let mut qg = intrinsic.qgate;
    let mut qb = intrinsic.qbulk;
    let mut qd = if here.mode > 0 {
        intrinsic.qdrn
    } else {
        qg.clone().add(&qb).add(&intrinsic.qdrn).scale(-1.)
    };
    let overlap_gate = if here.rgateMod == 3 { gm } else { g };
    let ov = |v: S, base, extra, kappa| -> SpiceResult<S> {
        if model.capMod == 0 {
            Ok(v.scale(base * here.nf))
        } else {
            Ok(overlap(v, base, extra, kappa)?.1.scale(here.nf))
        }
    };
    let qgd = ov(
        drop(overlap_gate, d),
        size.cgdo,
        size.weffCV * size.cgdl,
        size.ckappad,
    )?;
    let qgs = ov(
        drop(overlap_gate, s),
        size.cgso,
        size.weffCV * size.cgsl,
        size.ckappas,
    )?;
    let qgb = drop(overlap_gate, b).scale(size.cgbo);
    let overlap_charge = qgd.clone().add(&qgs).add(&qgb);
    let qmid = if here.rgateMod == 3 {
        overlap_charge
    } else {
        qg = qg.add(&overlap_charge);
        constant(0.)
    };
    qb = qb.add(&qgb.scale(-1.));
    qd = qd.add(&qgd.scale(-1.)).add(&qbd.clone().scale(-1.));
    if here.rbodyMod == 0 {
        qb = qb.add(&qbs).add(&qbd);
    }
    let mut qs = qg.clone().add(&qb).add(&qd).add(&qmid).scale(-1.);
    if here.rbodyMod != 0 {
        qs = qs.add(&qbs.clone().scale(-1.)).add(&qbd.clone().scale(-1.));
    }
    let auxiliary = vec![
        (state::VDS, vd.primal()),
        (state::VGS, vg.primal()),
        (state::VBS, vb.primal()),
        (state::VBD, drop(b, d).primal()),
        (state::VGES, drop(ge, s).primal()),
        (state::VGMS, drop(gm, s).primal()),
        (state::VDBS, drop(db, s).primal()),
        (state::VDBD, drop(db, d).primal()),
        (state::VSBS, drop(sb, s).primal()),
        (state::VSES, vses.primal()),
        (state::VDES, vdes.primal()),
        (state::QDEF, 0.),
        (state::QS, qs.primal()),
    ];
    let mut point = PhysicalPoint {
        currents: branches,
        charges: Vec::new(),
        auxiliary,
    };
    if here.rbodyMod == 0 {
        point
            .auxiliary
            .extend([(state::QBS, qbs.primal()), (state::QBD, qbd.primal())]);
    }
    // One incidence declaration for both storage binding and evaluation.
    for (slot, rows) in charge_layout(here) {
        let law = match slot {
            state::QB => &qb,
            state::QG => &qg,
            state::QD => &qd,
            state::QGMID => &qmid,
            state::QBS => &qbs,
            state::QBD => &qbd,
            _ => unreachable!("BSIM4 quasi-static charge layout"),
        };
        point.charges.push((slot, rows, law.clone()));
    }
    Ok(point)
}
