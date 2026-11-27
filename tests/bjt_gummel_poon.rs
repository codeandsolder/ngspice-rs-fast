//! Gummel-Poon BJT (#87): independent equations, Newton/companion Jacobians,
//! small-signal operators, charge conservation and explicit unported physics.
//! C-golden comparisons of the `m7_bjt_*` decks live in `xtask golden verify`.
use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot};
use ngspice_rs::devices::{
    AnalysisMode, Circuit, Forcing, Limit, LoadRequest, ModelContext, StateHistory, TransientTiming,
};
use ngspice_rs::maths::integrator::{Coefficients, DEFAULT_XMU, IntegrationMethod, StepHistory};
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, NodeKind};
use std::path::Path;

use ngspice_conformance::fixtures;

const VT: f64 = (1.38064852e-23 / 1.6021766208e-19) * 300.15;

fn netlist(body: &str) -> ngspice_rs::netlist::ast::Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("gp.cir"),
            &format!("Gummel-Poon\n{body}\n.end\n"),
        ))
        .unwrap()
}
fn circuit(body: &str) -> Circuit {
    Circuit::from_netlist(&netlist(body)).unwrap()
}
fn run(c: &mut Circuit, kind: AnalysisKind, args: &[&str]) -> Plot {
    AnalysisInput::with_arguments(kind, args.iter().copied())
        .resolve()
        .and_then(|plan| plan.run(c, &AnalysisContext::default()))
        .unwrap()
}
fn close(a: f64, b: f64, relative: f64, absolute: f64) {
    assert!(
        (a - b).abs() <= relative * b.abs() + absolute,
        "{a:e} != {b:e}"
    );
}
fn entry(matrix: &SparseMatrix, row: usize, col: usize) -> f64 {
    matrix
        .triplets()
        .iter()
        .filter(|t| t.row == row && t.col == col)
        .map(|t| t.value)
        .sum()
}

#[test]
fn operating_point_obeys_independent_gummel_poon_equations_and_kcl() {
    let model = "is=1e-15 bf=100 br=2 vaf=50 var=10 ikf=10m ikr=1m ise=1e-14 ne=1.5 isc=1e-14 nc=2 rb=100 rbm=10 re=2 rc=5";
    for (family, pol) in [("npn", 1.), ("pnp", -1.)] {
        // SUBS=1 makes both devices vertical so that the PNP mirrors the NPN.
        let mut c = circuit(&format!(
            "vc c 0 {}\nvb b 0 {}\nve e 0 0\nq1 c b e qm\n.model qm {family}({model} subs=1)",
            pol * 3.,
            pol * 0.78
        ));
        // Tight Newton tolerances: ngspice's defaults (reltol 1e-3) stop
        // well short of the 1e-7 agreement checked here.
        let p = run(
            &mut c,
            AnalysisKind::OperatingPoint,
            &["rtol=1e-12", "vntol=1e-12"],
        );
        let v = |name: &str| pol * p.value(&format!("v({name})"), 0).unwrap().re;
        let (vb, vc, ve) = (v("q1#base"), v("q1#collcx"), v("q1#emitter"));
        let (vbe, vbc) = (vb - ve, vb - vc);
        let gmin = 1e-12;
        let junction = |v: f64, n: f64, is: f64| is * ((v / (n * VT)).exp() - 1.);
        let cbe = junction(vbe, 1., 1e-15);
        let cbc = junction(vbc, 1., 1e-15);
        let cben = junction(vbe, 1.5, 1e-14) + gmin * vbe;
        let cbcn = junction(vbc, 2., 1e-14) + gmin * vbc;
        let q1 = 1. / (1. - vbc / 50. - vbe / 10.);
        let qb = q1 * (1. + (1. + 4. * (cbe / 10e-3 + cbc / 1e-3)).sqrt()) / 2.;
        // The vertical substrate junction is gmin from ground to the internal collector.
        let ic = (cbe - cbc) / qb - cbc / 2. - cbcn + gmin * vc;
        let ib = cbe / 100. + cben + cbc / 2. + cbcn;
        close(-pol * p.value("i(vc)", 0).unwrap().re, ic, 1e-7, 1e-15);
        close(-pol * p.value("i(vb)", 0).unwrap().re, ib, 1e-7, 1e-15);
        // Series resistances: RC, RE and the qb-modulated base resistance.
        close(ic, (3. - vc) / 5., 1e-7, 1e-15);
        close(ib + ic - gmin * vc, ve / 2., 1e-7, 1e-15);
        close(ib, (0.78 - vb) / (10. + 90. / qb), 1e-7, 1e-15);
        assert!(qb > 1.05, "high injection and Early effect act: qb = {qb}");
    }
}

#[test]
fn internal_nodes_follow_bjtsetup_and_stay_internal() {
    let c = circuit("vc c 0 1\nq1 c b e qm\nrb b 0 1k\nre e 0 1k\n.model qm npn(rb=10 re=1 rc=2)");
    let names: Vec<_> = c
        .nodes()
        .nodes()
        .iter()
        .filter(|n| n.kind == NodeKind::Internal)
        .map(|n| n.name.to_ascii_lowercase())
        .collect();
    assert_eq!(names, ["q1#collcx", "q1#base", "q1#emitter"]);
    let q = c.devices().iter().find(|d| d.name() == "q1").unwrap();
    assert_eq!(q.terminals().len(), 6);
    // Without series resistances no internal node is created.
    let plain = circuit("vc c 0 1\nq1 c b 0 qm\nrb b 0 1k\n.model qm npn(vaf=50)");
    assert!(
        plain
            .nodes()
            .nodes()
            .iter()
            .all(|n| n.kind != NodeKind::Internal)
    );
}

/// The full Gummel-Poon device: every slice, a substrate terminal and charges.
const FULL: &str = "vc c 0 2.5\nvb b 0 0.8\nve e 0 0.05\nvs s 0 -1\nq1 c b e s qm area=1.5 m=2\n.model qm npn(is=1e-15 bf=120 br=3 vaf=40 var=9 ikf=8m ikr=1m nkf=0.6 ise=1e-14 ne=1.6 isc=1e-14 nc=1.8 rb=200 rbm=20 irb=0.1m re=1.5 rc=12 cje=3p mje=0.4 cjc=2p mjc=0.45 xcjc=0.5 cjs=1.5p mjs=0.3 iss=1e-17 tf=0.3n xtf=5 vtf=2 itf=30m tr=20n fc=0.7)";

fn solved(c: &Circuit, context: &ModelContext) -> (Vector, StateHistory) {
    let solved =
        ngspice_rs::analysis::bias::solve_dc(c, context, &Default::default(), &[], None, None)
            .unwrap();
    let mut history = c.state_history().unwrap();
    history.initialize(solved.trial).unwrap();
    (solved.values, history)
}

fn load(
    c: &Circuit,
    x: &Vector,
    history: &StateHistory,
    context: &ModelContext,
    coefficients: Option<&Coefficients>,
) -> (SparseMatrix, Vector, Vec<f64>) {
    load_on(
        c,
        x,
        history,
        context,
        coefficients,
        SparseMatrix::new(c.unknown_count(), c.unknown_count()),
    )
}

fn load_on(
    c: &Circuit,
    x: &Vector,
    history: &StateHistory,
    context: &ModelContext,
    coefficients: Option<&Coefficients>,
    mut a: SparseMatrix,
) -> (SparseMatrix, Vector, Vec<f64>) {
    let n = c.unknown_count();
    let mut b = Vector::zeros(n);
    let mut trial = history.trial();
    let (mode, forcing) = match coefficients {
        Some(k) => (
            AnalysisMode::Transient {
                time: k.dt(),
                dt: k.dt(),
            },
            Some(Forcing {
                limit: Limit::Right,
                timing: TransientTiming::new(k.dt(), 10. * k.dt()).unwrap(),
            }),
        ),
        None => (AnalysisMode::OperatingPoint, None),
    };
    c.load(
        &LoadRequest {
            mode,
            solution: x,
            model_context: context,
            integration: coefficients,
            history,
            forcing,
        },
        &mut a,
        &mut b,
        &mut trial,
    )
    .unwrap();
    (a, b, trial.values().to_vec())
}

fn residual(a: &SparseMatrix, b: &Vector, x: &Vector) -> Vec<f64> {
    let mut r: Vec<f64> = b.as_slice().iter().map(|v| -v).collect();
    for t in a.triplets() {
        r[t.row] += t.value * x.as_slice()[t.col];
    }
    r
}

/// The Newton matrix of every load (`bjtload.c`'s) is the exact Jacobian of
/// its residual, in DC and with companion charges (including the
/// `qbe(vbc)` cross term), when the base resistance is bias independent:
/// like C, the load stamps `gx` alone and omits `d(gx)/dV`.
#[test]
fn newton_and_companion_jacobians_are_exact() {
    let c = circuit(&FULL.replace("rb=200 rbm=20 irb=0.1m", "rb=200"));
    for temperature in [27., 90.] {
        let context = ModelContext::new(temperature, 27.);
        let (bias, history) = solved(&c, &context);
        // Move off the solution so every branch carries a nonzero residual.
        let mut x = bias.clone();
        for (i, value) in x.as_mut_slice().iter_mut().enumerate() {
            *value += 0.003 * ((i % 5) as f64 - 2.);
        }
        let dt = 1e-9;
        let coefficients = StepHistory::new()
            .trial(IntegrationMethod::Trapezoidal, 1, dt, DEFAULT_XMU)
            .unwrap();
        for k in [None, Some(&coefficients)] {
            let (a, _, _) = load(&c, &x, &history, &context, k);
            let n = c.unknown_count();
            for col in 0..n {
                let h = 1e-6;
                let mut low = x.clone();
                let mut high = x.clone();
                low.as_mut_slice()[col] -= h;
                high.as_mut_slice()[col] += h;
                let (al, bl, _) = load(&c, &low, &history, &context, k);
                let (ah, bh, _) = load(&c, &high, &history, &context, k);
                let rl = residual(&al, &bl, &low);
                let rh = residual(&ah, &bh, &high);
                for row in 0..n {
                    let numerical = (rh[row] - rl[row]) / (2. * h);
                    let analytic = entry(&a, row, col);
                    close(numerical, analytic, 2e-5, 1e-9);
                }
            }
        }
    }
}

/// AC operators: the conductance matrix is the Newton Jacobian (here with a
/// bias-independent RB, where bjtacld.c's gx-only stamp is exact) and the
/// charge matrix is the companion Jacobian divided by `ag0`.
#[test]
fn small_signal_operators_are_the_dc_and_charge_jacobians() {
    let body = FULL.replace("rbm=20 irb=0.1m ", "rbm=200 ");
    let c = circuit(&body);
    let context = ModelContext::new(50., 27.);
    let (bias, history) = solved(&c, &context);
    let system = c.small_signal_system(&context, &bias).unwrap();
    let dt = 1e-9;
    let coefficients = StepHistory::new()
        .trial(IntegrationMethod::Trapezoidal, 1, dt, DEFAULT_XMU)
        .unwrap();
    let (dc, _, _) = load(&c, &bias, &history, &context, None);
    let (tran, _, _) = load(&c, &bias, &history, &context, Some(&coefficients));
    let ag0 = coefficients.ag()[0];
    let n = c.unknown_count();
    // Linear sources/resistors are part of both A operators; the capacitive
    // operator holds only the BJT charges here.
    let mut nonzero = 0;
    for row in 0..n {
        for col in 0..n {
            let g = entry(&dc, row, col);
            close(entry(&system.a, row, col), g, 1e-9, 1e-18);
            let charge = (entry(&tran, row, col) - g) / ag0;
            let e = entry(&system.e, row, col);
            close(e, charge, 1e-6, 1e-24);
            if e != 0. {
                nonzero += 1;
            }
        }
    }
    assert!(nonzero >= 12, "{nonzero}");
}

#[test]
fn charges_are_disposable_and_integrate_actual_nonlinear_q() {
    let c = circuit(FULL);
    let context = ModelContext::default();
    let (bias, history) = solved(&c, &context);
    let before = history.clone();
    let mut x = bias;
    for (node, step) in [("b", 0.01), ("q1#base", 0.01), ("q1#collcx", -0.02)] {
        let row = c.unknowns().node_row(c.nodes().get(node).unwrap()).unwrap();
        x.as_mut_slice()[row] += step;
    }
    let dt = 1e-9;
    let coefficients = StepHistory::new()
        .trial(IntegrationMethod::Trapezoidal, 1, dt, DEFAULT_XMU)
        .unwrap();
    let (_, _, first) = load(&c, &x, &history, &context, Some(&coefficients));
    let (_, _, repeat) = load(&c, &x, &history, &context, Some(&coefficients));
    assert_eq!(first, repeat);
    assert_eq!(history, before);
    let (index, device) = c
        .devices()
        .iter()
        .enumerate()
        .find(|(_, d)| d.name() == "q1")
        .unwrap();
    // bjttrunc.c: qbe, qbc, qsub, and qbx because XCJC < 1.
    assert_eq!(device.truncation_slots(), [0, 2, 4, 6]);
    let base = c.state_rows(index).unwrap().start;
    let mut moved = 0;
    for slot in device.truncation_slots() {
        let q = first[base + slot];
        let old = history.accepted(1).unwrap()[base + slot];
        close(first[base + slot + 1], (q - old) / dt, 1e-12, 1e-15);
        if q != old {
            moved += 1;
        }
    }
    assert!(moved >= 3, "{moved}");
}

#[test]
fn transient_terminal_currents_conserve_charge() {
    let mut c = circuit(
        "vc c 0 pulse(2 0.3 5n 2n 2n 10n 40n)\nvb b 0 pulse(0.6 0.85 2n 1n 1n 15n 40n)\nve e 0 0\nvs s 0 -1\nq1 c b e s qm\n.model qm npn(is=1e-15 bf=120 br=3 vaf=40 ikf=8m ise=1e-14 isc=1e-14 rb=200 rbm=20 irb=0.1m re=1.5 rc=12 cje=3p cjc=2p xcjc=0.5 cjs=1.5p iss=1e-17 tf=0.3n xtf=5 vtf=2 itf=30m tr=20n)",
    );
    let p = run(&mut c, AnalysisKind::Transient, &["0.5n", "40n"]);
    let currents = |row: usize| -> Vec<f64> {
        ["i(vc)", "i(vb)", "i(ve)", "i(vs)"]
            .iter()
            .map(|name| p.value(name, row).unwrap().re)
            .collect()
    };
    let peak = (0..p.point_count())
        .flat_map(currents)
        .fold(0., |m: f64, v| m.max(v.abs()));
    assert!(peak > 1e-3, "{peak}");
    // Every BJT current and charge flows between two of its nodes, so the
    // terminal currents (displacement currents included) sum to zero.
    for row in 0..p.point_count() {
        close(currents(row).iter().sum::<f64>(), 0., 0., 1e-12 * peak);
    }
}

#[test]
fn unported_gummel_poon_physics_is_explicit() {
    for body in [
        "q1 c b 0 qm\n.model qm npn(rco=10)",
        "q1 c b 0 qm\n.model qm npn(gamma=1e-11 vo=5)",
        "q1 c b 0 qm\n.model qm npn(tf=1n ptf=30)",
        "q1 c b 0 qm\n.model qm npn(vbe_max=5)",
    ] {
        let error = Circuit::from_netlist(&netlist(body)).unwrap_err();
        assert!(error.is_not_yet_ported(), "{body}: {error}");
    }
    // OFF and the IC vector are ported (#99, bjtload.c/bjtgetic.c), and so
    // is the flicker noise law (#100, bjtnoise.c).
    for body in [
        "q1 c b 0 qm\n.model qm npn(kf=1e-16 af=1.2)",
        "q1 c b 0 qm off\n.model qm npn",
        "q1 c b 0 qm icvbe=0.6 ic=0.7,2\n.model qm npn",
    ] {
        Circuit::from_netlist(&netlist(body)).unwrap();
    }
    // Excess phase has no effect without TF (bjttemp.c: PTF * TF).
    Circuit::from_netlist(&netlist("q1 c b 0 qm\n.model qm npn(ptf=30)")).unwrap();
    // Aliases resolve to their canonical setter (bjt.c IOPR/IOPAR).
    let mut alias = circuit("vc c 0 2\nvb b 0 0.7\nq1 c b 0 qm\n.model qm npn(va=50 ik=10m)");
    let mut canonical = circuit("vc c 0 2\nvb b 0 0.7\nq1 c b 0 qm\n.model qm npn(vaf=50 ikf=10m)");
    let a = run(&mut alias, AnalysisKind::OperatingPoint, &[]);
    let b = run(&mut canonical, AnalysisKind::OperatingPoint, &[]);
    assert_eq!(a.value("i(vc)", 0), b.value("i(vc)", 0));
    // Unknown and malformed setters remain errors.
    for body in [
        "q1 c b 0 qm\n.model qm npn(bogus=1)",
        "q1 c b 0 qm\n.model qm npn(mje=1)",
        "q1 c b 0 qm\n.model qm npn(tlev=2)",
    ] {
        assert!(Circuit::from_netlist(&netlist(body)).is_err(), "{body}");
    }
}

#[test]
fn m7_fixtures_parse_write_parse_with_a_fixed_point() {
    for name in [
        "m7_bjt_gummel",
        "m7_bjt_output",
        "m7_bjt_temp",
        "m7_bjt_amp_ac",
        "m7_bjt_amp_tran",
    ] {
        let n = Parser::new()
            .parse_file(
                ngspice_conformance::fixtures::workspace_root()
                    .unwrap()
                    .join(format!("conformance/netlists/{name}.cir")),
            )
            .unwrap();
        let text = ngspice_rs::netlist::write_netlist(&n).unwrap();
        let round = Parser::new()
            .parse_deck(&parse_deck_text(Path::new("round.cir"), &text))
            .unwrap();
        assert!(ngspice_rs::netlist::semantic_eq(&n, &round));
        assert_eq!(text, ngspice_rs::netlist::write_netlist(&round).unwrap());
    }
}

/// A lone IBE/IBC is ignored by the initial thermal model but not discarded:
/// restoring the missing partner during `.sens` later activates the pair.
#[test]
fn single_saturation_current_setters_survive_in_sensitivity_records() {
    for (card, be, bc) in [
        ("ibe=1e-13", Some(1e-13), None),
        ("ibc=2e-14", None, Some(2e-14)),
        ("ibe=1e-13 ibc=2e-14", Some(1e-13), Some(2e-14)),
    ] {
        let c = circuit(&format!("q1 c b 0 qm\n.model qm npn {card}"));
        let replay = c
            .device("q1")
            .unwrap()
            .sensitivity(&ModelContext::default())
            .unwrap();
        let (m, _) = replay.records().unwrap();
        assert_eq!(m.given("ibe"), be.is_some(), "{card}");
        assert_eq!(m.given("ibc"), bc.is_some(), "{card}");
        assert_eq!(m.value("ibe").unwrap(), be.unwrap_or(0.));
        assert_eq!(m.value("ibc").unwrap(), bc.unwrap_or(0.));
    }
}

/// Large reference columns previously misdiagnosed as cancellation. The card
/// uses default tolerances, like the measured ngspice-47+ probe. In particular
/// VAF is stepped from zero and exposes the paired saturation-current switch.
#[test]
fn single_sided_saturation_currents_follow_cs_restore_walk() {
    for (card, pins) in [
        (
            "ibc=2e-14",
            [
                ("v(q1:ibe)", 5.433_384_999_027_637e11),
                ("v(q1:ibc)", 4.100_676_337_386_377_5),
                ("v(q1:vaf)", 0.086_011_074_718_727_68),
                ("v(q1:var)", -0.013_988_896_037_050_5),
            ],
        ),
        (
            "ibe=1e-13",
            [
                ("v(q1:ibc)", 54_283.616_822_039_774),
                ("v(q1:vaf)", 2.386_795_986_069_663e11),
                ("v(q1:ikf)", -55_264.635_827_176_47),
                ("v(q1_area)", 0.054_333_850_228_572_1),
            ],
        ),
    ] {
        let plot = run_deck(&format!(
            "vcc 1 0 dc 5\nvids 1 2 dc 0\n\
            vb 4 0 dc 0.7\nrbb 4 3 1k\nq1 2 3 0 qn\n\
            .model qn npn level=1 {card}\n.sens i(vids)"
        ));
        for (name, expected) in pins {
            close(column(&plot, name), expected, 1e-6, 1e-9);
        }
    }
}

/// Runs the first analysis specified by the deck.
fn run_deck(body: &str) -> Plot {
    let netlist = netlist(body);
    let config = ngspice_rs::analysis::RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut c = config.circuit(&netlist).unwrap();
    request.run(&mut c, &config.context()).unwrap()
}
fn column(plot: &Plot, name: &str) -> f64 {
    let index = plot
        .variable_index(name)
        .unwrap_or_else(|| panic!("missing {name}: {:?}", plot.variables()));
    plot.row(0).unwrap()[index].re
}

/// `.sens` over `bjt.c`'s tables (#164): 103 model setters and six instance
/// ones. The `IOPA` capacitance family is `IF_AC` (so `cje`/`tf`/`xcjc` and
/// friends appear only in an AC walk), `ic` is a vector, `off`/`sens_area` are
/// flags, the probe entries are ask-only and the `IOPR` aliases are redundant.
/// `rco` and the unported quasi-saturation/SOA parameters are still enumerated.
#[test]
fn sensitivity_covers_the_bjt_tables() {
    let plot = run_deck(
        "vcc 1 0 dc 5\nq1 1 3 2 qn area=2\nvids 2 0 dc 0\nvb 4 0 dc 0.72\n\
         rb 4 3 1k\n.model qn npn level=1\n.sens i(vids)",
    );
    let names: Vec<&str> = plot
        .variables()
        .iter()
        .map(|variable| variable.name.as_str())
        .filter(|name| name.starts_with("v(q1"))
        .collect();
    assert_eq!(names.len(), 109, "{names:?}");
    for absent in ["v(q1:xcjc)", "v(q1:tf)", "v(q1:vto)", "v(q1:icvbe)"] {
        assert!(!names.contains(&absent), "{absent} in {names:?}");
    }
    for present in ["v(q1:tnom)", "v(q1:nkf)", "v(q1:rth0)", "v(q1_temp)"] {
        assert!(names.contains(&present), "{present} missing from {names:?}");
    }
}

/// `run_deck()` refuses a card with `RC != 0`: `bjtsetup.c` then creates
/// `collCX`, and C's walk marks `VO`/`GAMMA`/`QCO` given, switching the Kull
/// quasi-saturation terms on in the perturbed loads (measured: `v(q1:gamma)`
/// is 0.652 where the port has no such physics).
#[test]
fn sensitivity_refuses_a_card_with_a_collector_resistance() {
    let netlist = netlist(
        "vcc 1 0 dc 5\nq1 1 3 2 qn\nvids 2 0 dc 0\nvb 4 0 dc 0.72\nrb 4 3 1k\n\
         .model qn npn level=1 rc=40\n.sens i(vids)",
    );
    let config = ngspice_rs::analysis::RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut c = config.circuit(&netlist).unwrap();
    let error = request.run(&mut c, &config.context()).unwrap_err();
    assert!(error.is_not_yet_ported(), "{error}");
    assert!(error.to_string().contains("RC"), "{error}");
}

const BJT_SENS_CARD: &str = ".options reltol=1e-10 vntol=1e-13 abstol=1e-16\n\
         vcc 1 0 dc 5\nq1 1 3 2 qn area=2 areab=3 areac=4 dtemp=-10\n\
         vids 2 0 dc 0\nvb 4 0 dc 0.72\nrb 4 3 1k\nvcc2 5 0 dc -5\n\
         q2 5 7 2 qp dtemp=20\nvb2 8 0 dc -0.68\nrb2 8 7 1k\n\
         .model qn npn level=1 ise=1e-3 rb=300 re=20 irb=1e-3 rbm=100 \
         tbf1=0.01 tbf2=1e-5\n\
         .model qp pnp level=1 tnom=50 tbr1=0.02 tbr2=2e-5\n.sens i(vids)";

/// The `sens_bjt` card: an NPN at `dtemp=-10` with `ISE > 1e-4` (rescales to
/// `IS*ISE` in `bjtsetup.c`), `rb`/`re`/`irb`/`rbm`, area factors and a `BF`
/// polynomial, plus a PNP at `dtemp=20` with `tnom=50`. The reference binary
/// prints these columns; `rco` is zero because `sens_setp()` skips it, and
/// `bf`/`vaf` are zero because C's own linearization does not see them here.
/// The ISE step is 1e-25, amplifying reference roundoff enough to
/// reverse the sign between two independent C builds. Require both
/// captures and apply the unchanged tolerance at their nearest endpoint.
#[test]
fn sensitivity_matches_the_reference_on_the_fixture_card() {
    let plot = run_deck(BJT_SENS_CARD);
    let references: Vec<_> = fixtures::numerical_golden_paths(
        &ngspice_conformance::fixtures::workspace_root().unwrap(),
        "sens_bjt",
    )
    .iter()
    .map(|path| ngspice_rs::analysis::RawFile::load(path).unwrap())
    .collect();
    let expected = &references[0].plots[0].plot;
    for raw in &references {
        assert_eq!(raw.plots.len(), 1);
        let reference = &raw.plots[0].plot;
        assert_eq!(plot.variable_count(), reference.variable_count());
        assert_eq!(plot.point_count(), reference.point_count());
        assert_eq!(plot.plotname, reference.plotname);
        assert_eq!(plot.flags(), reference.flags());
        for variable in expected.variables() {
            let index = reference.variable_index(&variable.name).unwrap();
            assert_eq!(reference.variables()[index], *variable);
        }
    }
    let mut mismatches = Vec::new();
    for variable in expected.variables() {
        let actual = column(&plot, &variable.name);
        let reference = fixtures::nearest_reference(
            actual,
            references
                .iter()
                .map(|raw| column(&raw.plots[0].plot, &variable.name)),
        )
        .unwrap();
        if (actual - reference).abs() > 1e-6 * reference.abs() + 1e-9 || !actual.is_finite() {
            mismatches.push(format!("{}: {actual:e} != {reference:e}", variable.name));
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    assert_eq!(column(&plot, "v(q1:rco)"), 0., "sens_setp skips RCO");
}

/// A selected parameter is perturbed before other setters alter its model.
/// Check the actual temperature derivative against independent DC solves,
/// as well as the pinned C reference for the same selected parameter.
#[test]
fn selected_bandgap_sensitivity_matches_physical_dc_derivative() {
    let selected = BJT_SENS_CARD.replace(".sens i(vids)", ".sens i(vids) q1:eg");
    let actual = column(&run_deck(&selected), "v(q1:eg)");
    let dc = |eg: f64| {
        let body = BJT_SENS_CARD
            .replace(".sens i(vids)", ".op")
            .replace("ise=1e-3 rb=300", &format!("ise=1e-3 eg={eg:.17} rb=300"));
        column(&run_deck(&body), "i(vids)")
    };
    let h = 1e-4;
    let derivative = (dc(1.11 + h) - dc(1.11 - h)) / (2. * h);
    close(actual, derivative, 1e-6, 1e-10);
    close(actual, -0.000_143_991_710_094_134_88, 1e-6, 1e-10);
}

#[test]
fn bipolar_and_diode_fixed_stamps_replay_the_unbound_matrix() {
    let full = format!(
        "{FULL}\nd1 b e dm area=1.7 m=2.3\nd2 c 0 dm\n.model dm d(is=1e-14 rs=3 cjo=1.2p tt=0.5n)"
    );
    let aliases = "vc c 0 2.5\nvb b 0 0.63\nq1 c b 0 qm m=1.7\nd1 b 0 dm m=2.3\n\
                   .model qm npn(bf=120 is=1e-15 cje=3p cjc=2p tf=0.3n)\n\
                   .model dm d(is=1e-14 cjo=1.2p tt=0.5n)";
    for body in [&full, aliases] {
        let c = circuit(body);
        let context = ModelContext::default();
        let (bias, history) = solved(&c, &context);
        let coeff = StepHistory::new()
            .trial(IntegrationMethod::Trapezoidal, 1, 0.7e-9, DEFAULT_XMU)
            .unwrap();
        // Repeated loads exercise learning, verification and replay. Using
        // several voltages and alternating DC/transient changes every stamp
        // value without changing device topology.
        for step in 0..6 {
            let mut x = bias.clone();
            for (k, element) in x.as_mut_slice().iter_mut().enumerate() {
                *element += 0.0001 * (f64::from(step) + 1.) * ((k % 3) as f64 - 1.);
            }
            let integration = (step % 2 != 0).then_some(&coeff);
            let (plain, rhs, state) = load(&c, &x, &history, &context, integration);
            let (bound, bound_rhs, bound_state) = load_on(
                &c,
                &x,
                &history,
                &context,
                integration,
                c.new_matrix().unwrap(),
            );
            assert!(bound.bound_values().is_some());
            for r in 0..c.unknown_count() {
                for col in 0..c.unknown_count() {
                    assert_eq!(
                        plain.get(r, col).to_bits(),
                        bound.get(r, col).to_bits(),
                        "step {step}, matrix ({r},{col})"
                    );
                }
            }
            assert_eq!(
                rhs.as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                bound_rhs
                    .as_slice()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                bound_state.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn physical_taylor_reconstructs_multivariate_gummel_poon_equations()
-> Result<(), ngspice_rs::primitives::SpiceError> {
    // Includes varying base-spread resistance (C's ordinary Newton matrix
    // freezes it), bias-modulated diffusion charge, partitioned BC charge,
    // both polarities and both substrate topologies.
    let common = "is=1e-15 bf=120 br=3 vaf=40 var=9 ikf=8m ikr=1m nkf=0.6 ise=1e-14 ne=1.6 isc=1e-14 nc=1.8 rb=200 rbm=20 irb=0.1m re=1.5 rc=12 cje=3p mje=0.4 cjc=2p mjc=0.45 xcjc=0.5 cjs=1.5p mjs=0.3 iss=1e-17 tf=0.3n xtf=5 vtf=2 itf=30m tr=20n fc=0.7";
    for (family, pol) in [("npn", 1.), ("pnp", -1.)] {
        for subs in [-1, 1] {
            let mut c = circuit(&format!(
                "q1 c b e s qm area=1.5 m=2\n.model qm {family}({common} subs={subs})"
            ));
            c.finalize()?;
            let context = ModelContext::default();
            let prepared = c.prepare_run();
            let mut dae =
                prepared.prepare_physical_dae(&context, TransientTiming::new(1e-6, 1e-3)?)?;
            let rows = ["c", "b", "e", "s", "q1#collcx", "q1#base", "q1#emitter"]
                .map(|name| c.unknowns().node_row(c.nodes().get(name).unwrap()).unwrap());
            for (base, collector) in [(0.72, 1.5), (-0.2, 0.5), (0.5, -0.18), (0.81, 0.35)] {
                let mut state = vec![Vector::zeros(c.unknown_count()); 9];
                for (i, (bias, slope)) in [
                    (collector + 0.02, 0.001),
                    (base + 0.01, -0.002),
                    (0.01, 0.0015),
                    (-0.3, 0.0007),
                    (collector, -0.001),
                    (base, 0.0012),
                    (0., -0.0004),
                ]
                .into_iter()
                .enumerate()
                {
                    state[0].set(rows[i], pol * bias)?;
                    state[1].set(rows[i], pol * slope)?;
                }
                let jet = dae.evaluate_taylor(&state, 0., 1e-6, Limit::Right)?;
                for t in [0., 0.25, 0.5] {
                    let mut sample = state[0].clone();
                    for i in 0..sample.len() {
                        sample.set(
                            i,
                            state[1]
                                .get(i)
                                .unwrap()
                                .mul_add(t, state[0].get(i).unwrap()),
                        )?;
                    }
                    let actual = dae.evaluate(&sample, 0., Limit::Right)?;
                    let series = |values: &[Vector], row: usize| {
                        values
                            .iter()
                            .rev()
                            .fold(0_f64, |sum, v| sum.mul_add(t, v.get(row).unwrap()))
                    };
                    for row in 0..sample.len() {
                        close(
                            series(&jet.residual, row),
                            actual.residual.get(row).unwrap(),
                            1e-9,
                            1e-12,
                        );
                        close(
                            series(&jet.charge, row),
                            actual.charge.get(row).unwrap(),
                            1e-9,
                            1e-21,
                        );
                    }
                }
                // Differentiate the independent scalar residual with a
                // fourth-order centered stencil. The ordinary load forms
                // F as G*x-rhs, so cancellation limits the accuracy of
                // small derivatives in a row carrying large substrate
                // current. Bound that roundoff separately from truncation.
                let h = 1e-3;
                for col in 0..c.unknown_count() {
                    let mut samples = Vec::new();
                    for step in [-2_f64, -1., 1., 2.] {
                        let mut sample = state[0].clone();
                        sample.set(col, step.mul_add(h, sample.get(col).unwrap()))?;
                        let point = dae.evaluate(&sample, 0., Limit::Right)?;
                        let scales = (0..sample.len())
                            .map(|row| {
                                point.residual.get(row).unwrap().abs()
                                    + sample
                                        .as_slice()
                                        .iter()
                                        .enumerate()
                                        .map(|(i, v)| (point.conductance.get(row, i) * v).abs())
                                        .sum::<f64>()
                            })
                            .collect::<Vec<_>>();
                        samples.push((point.residual.clone(), point.charge.clone(), scales));
                    }
                    let stencil = |component: usize, row: usize| {
                        let value = |i: usize| {
                            let sample = &samples[i];
                            if component == 0 {
                                sample.0.get(row).unwrap()
                            } else {
                                sample.1.get(row).unwrap()
                            }
                        };
                        (8. * (value(2) - value(1)) - (value(3) - value(0))) / (12. * h)
                    };
                    for row in 0..c.unknown_count() {
                        // 32 eps covers the sparse stamp accumulation,
                        // matrix-vector product and subtraction; the stencil
                        // amplifies point error by at most 1.5/h.
                        let scale = samples
                            .iter()
                            .map(|sample| sample.2[row])
                            .fold(0_f64, f64::max);
                        let roundoff = 48. * f64::EPSILON * scale / h;
                        let exact = jet.conductance[0].get(row, col);
                        let numeric = stencil(0, row);
                        assert!(
                            (exact - numeric).abs() <= 2e-6 * numeric.abs() + 1e-10 + roundoff,
                            "{family} {subs} {base} {collector} dF[{row},{col}]: {exact:e} != {numeric:e} (roundoff {roundoff:e})"
                        );
                        close(
                            jet.capacitance[0].get(row, col),
                            stencil(1, row),
                            2e-6,
                            1e-19,
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn derivative_array_with_bjt_preserves_independent_base_and_collector_charge()
-> Result<(), ngspice_rs::primitives::SpiceError> {
    use ngspice_rs::analysis::RunConfig;
    for (family, pol) in [("npn", 1.), ("pnp", -1.)] {
        for method in ["gear", "trap"] {
            let body = format!(
                "vc c 0 {}\nvb b 0 PWL(0 {} 1m {})\nq1 c b 0 qm m=2\n.model qm {family}(is=1e-15 bf=100 br=2 tf=5n tr=2n cje=20p mje=0 cjc=30p mjc=0 xcjc=0.5 subs=1)\nb1 first 0 v=1m*ddt(v(b))\nb2 out 0 v=1m*ddt(v(first))\nr1 out 0 1k\n.options method={method} reltol=1e-8 vntol=1n\n.tran 1u 20u 0 1u uic",
                0.9 * pol,
                0.6 * pol,
                0.7 * pol
            );
            let deck = netlist(&body);
            let config = RunConfig::from_netlist(&deck)?;
            let mut c = config.circuit(&deck)?;
            c.set_observations(&["@q1[i]".into()])?;
            let plot = config
                .plan_for(&deck.analyses[0])?
                .run(&mut c, &config.context())?;
            let junction = |voltage: f64| {
                if voltage < -3. * VT {
                    let a = (3. * VT / (voltage * std::f64::consts::E)).powi(3);
                    (-1e-15 * (1. + a), 3e-15 * a / voltage)
                } else {
                    (
                        1e-15 * (voltage / VT).exp_m1(),
                        1e-15 / VT * (voltage / VT).exp(),
                    )
                }
            };
            for i in 0..plot.point_count() {
                let v = |name: &str| plot.value(name, i).unwrap().re;
                let normalized_be = 100_f64.mul_add(v("time"), 0.6);
                let normalized_bc = normalized_be - 0.9;
                let (be, gbe) = junction(normalized_be);
                let (bc, gbc) = junction(normalized_bc);
                let base = be / 100. + bc / 2. + 1e-12 * (normalized_be + normalized_bc);
                let collector = be - bc - bc / 2. - 1e-12 * normalized_bc + 0.9e-12;
                let qbe_rate = 100. * (20e-12 + 5e-9 * gbe);
                let qbc_rate = 100. * (30e-12 + 2e-9 * gbc);
                close(v("v(first)"), pol * 0.1, 1e-9, 1e-10);
                close(v("v(out)"), 0., 0., 1e-9);
                close(
                    v("i(vb)"),
                    -2. * pol * (base + qbe_rate + qbc_rate),
                    1e-6,
                    1e-13,
                );
                close(v("i(vc)"), -2. * pol * (collector - qbc_rate), 1e-6, 1e-13);
                close(v("@q1[i]"), -v("i(vc)"), 1e-8, 1e-14);
            }
        }
    }
    Ok(())
}
