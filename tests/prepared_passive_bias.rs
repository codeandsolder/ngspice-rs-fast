//! Physical witnesses for resolved source-prepared initialization equations.
use std::path::Path;

use ngspice_rs::analysis::{Plot, RunConfig};
use ngspice_rs::devices::{
    Circuit, StorageKind,
    passive_bias::{NodeInitialization, PassiveBias, PassiveInitialization},
};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, SourceLoc};

fn deck(body: &str) -> (ngspice_rs::netlist::ast::Netlist, RunConfig, Circuit) {
    let n = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("prepared-bias.cir"),
            &format!("prepared bias\n{body}\n.end\n"),
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&n).unwrap();
    let circuit = config.circuit(&n).unwrap();
    (n, config, circuit)
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} vs {expected}"
    );
}

fn last(plot: &Plot, name: &str) -> f64 {
    plot.value(name, plot.point_count() - 1).unwrap().re
}

fn voltage(voltage: f64, conductance: Option<f64>) -> PassiveBias {
    PassiveBias::Voltage {
        voltage,
        conductance,
    }
}

fn current(current: f64, conductance: f64) -> PassiveBias {
    PassiveBias::Current {
        current,
        conductance,
    }
}

#[test]
fn hard_capacitor_bias_releases_before_ac_and_transient_steps() {
    let (n, config, mut c) = deck(
        "v1 a 0 dc 2 ac 1\nr1 a d 1k\nc1 d 0 1n\n.op\n.ac lin 1 100k 100k\n.tran 10n 3u 0 10n",
    );
    c.bind_passive_initialization(
        "c1",
        PassiveInitialization::new(
            StorageKind::Capacitor,
            Some(1.),
            [
                AnalysisKind::OperatingPoint,
                AnalysisKind::Ac,
                AnalysisKind::Transient,
            ]
            .map(|kind| (kind, voltage(1., None))),
        )
        .unwrap(),
    )
    .unwrap();
    c.finalize().unwrap();
    let plans = config.plans_for(&n.analyses).unwrap();
    let op = plans[0].run(&mut c, &config.context()).unwrap();
    close(last(&op, "v(d)"), 1., 1e-12);
    close(last(&op, "i(c1:bias)"), 1e-3, 1e-12);
    let ac = plans[1].run(&mut c, &config.context()).unwrap();
    let wrc = std::f64::consts::TAU * 100e3 * 1e3 * 1e-9;
    let denominator = 1. + wrc * wrc;
    let h = ac.value("v(d)", 0).unwrap();
    close(h.re, denominator.recip(), 1e-12);
    close(h.im, -wrc / denominator, 1e-12);
    close(last(&ac, "i(c1:bias)"), 0., 1e-12);
    let tran = plans[2].run(&mut c, &config.context()).unwrap();
    close(tran.value("v(d)", 0).unwrap().re, 1., 1e-9);
    close(last(&tran, "v(d)"), 2. - (-3_f64).exp(), 1e-4);
    close(last(&tran, "i(c1:bias)"), 0., 1e-12);
    let again = plans[0].run(&mut c, &config.context()).unwrap();
    close(last(&again, "v(d)"), 1., 1e-12);
}

#[test]
fn soft_capacitor_and_node_forcing_are_physical_norton_sources() {
    let (n, config, mut c) = deck("v1 a 0 2\nr1 a d 1k\nc1 d 0 0\n.op\n.dc v1 2 2 1");
    c.bind_passive_initialization(
        "c1",
        PassiveInitialization::new(
            StorageKind::Capacitor,
            Some(0.),
            [(AnalysisKind::OperatingPoint, voltage(0., Some(1e-3)))],
        )
        .unwrap(),
    )
    .unwrap();
    for (hint, g) in [(1., 1e-3), (9., 1.)] {
        c.bind_node_initialization(
            "d",
            NodeInitialization::new(
                hint,
                SourceLoc::new(Path::new("prepared-bias.cir").to_path_buf(), 1, 1),
                [(AnalysisKind::OperatingPoint, hint, g)],
            )
            .unwrap(),
        )
        .unwrap();
    }
    c.finalize().unwrap();
    c.set_observations(&["i(c1)".into(), "@c1[p]".into()])
        .unwrap();
    let plans = config.plans_for(&n.analyses).unwrap();
    let op = plans[0].run(&mut c, &config.context()).unwrap();
    // Three equal conductances to 2, 0 and the first node hint 1 V.
    close(last(&op, "v(d)"), 1., 1e-12);
    close(last(&op, "i(c1)"), 1e-3, 1e-12);
    close(last(&op, "@c1[p]"), 1e-3, 1e-12);
    let dc = plans[1].run(&mut c, &config.context()).unwrap();
    close(last(&dc, "v(d)"), 2., 1e-12);
    close(last(&dc, "i(c1)"), 0., 1e-12);
    close(last(&dc, "@c1[p]"), 0., 1e-12);
}

#[test]
fn inductor_bias_enforces_total_current_then_restores_flux_dynamics() {
    for conductance in [0., 1e-9] {
        let (n, config, mut c) =
            deck("v1 a 0 1\nr1 a d 100\nl1 d 0 1m\n.op\n.tran 100n 30u 0 100n");
        c.bind_passive_initialization(
            "l1",
            PassiveInitialization::new(
                StorageKind::Inductor,
                Some(2e-3),
                [AnalysisKind::OperatingPoint, AnalysisKind::Transient]
                    .map(|kind| (kind, current(2e-3, conductance))),
            )
            .unwrap(),
        )
        .unwrap();
        c.finalize().unwrap();
        let plans = config.plans_for(&n.analyses).unwrap();
        let initial_v = (1. - 100. * 2e-3) / (1. + 100. * conductance);
        let initial_i = 2e-3 + conductance * initial_v;
        let op = plans[0].run(&mut c, &config.context()).unwrap();
        close(last(&op, "v(d)"), initial_v, 1e-12);
        close(last(&op, "i(l1)"), initial_i, 1e-12);
        let tran = plans[1].run(&mut c, &config.context()).unwrap();
        close(
            last(&tran, "i(l1)"),
            1e-2 + (initial_i - 1e-2) * (-3_f64).exp(),
            1e-6,
        );
    }
}

#[test]
fn redundant_or_conflicting_zero_capacitance_hard_branches_remain_singular() {
    for second in [0., 1.] {
        let (n, config, mut c) = deck("r1 d 0 1k\nc1 d 0 0\nc2 d 0 0\n.op");
        for (name, target) in [("c1", 0.), ("c2", second)] {
            c.bind_passive_initialization(
                name,
                PassiveInitialization::new(
                    StorageKind::Capacitor,
                    None,
                    [(AnalysisKind::OperatingPoint, voltage(target, None))],
                )
                .unwrap(),
            )
            .unwrap();
        }
        c.finalize().unwrap();
        assert!(
            config
                .plan_for(&n.analyses[0])
                .unwrap()
                .run(&mut c, &config.context())
                .is_err()
        );
    }
}

#[test]
fn uic_storage_ic_overrides_differential_node_hint_without_bias_stamps() {
    for explicit in [None, Some(1.)] {
        let (n, config, mut c) = deck("r1 d 0 1k\nc1 d 0 1n\n.tran 10n 1u 0 10n uic");
        c.bind_passive_initialization(
            "c1",
            PassiveInitialization::new(
                StorageKind::Capacitor,
                explicit,
                [(AnalysisKind::Transient, voltage(9., None))],
            )
            .unwrap(),
        )
        .unwrap();
        c.bind_node_initialization(
            "d",
            NodeInitialization::new(
                2.,
                SourceLoc::new(Path::new("prepared-bias.cir").to_path_buf(), 1, 1),
                [(AnalysisKind::Transient, 9., 1e-3)],
            )
            .unwrap(),
        )
        .unwrap();
        c.finalize().unwrap();
        let p = config
            .plan_for(&n.analyses[0])
            .unwrap()
            .run(&mut c, &config.context())
            .unwrap();
        close(
            last(&p, "v(d)"),
            explicit.unwrap_or(2.) * (-1_f64).exp(),
            1e-4,
        );
        close(last(&p, "i(c1:bias)"), 0., 1e-12);
    }
}

#[test]
fn prepared_hard_bias_survives_binned_geometry_candidates() {
    use model_normalize_ngspice::passive_ic::{
        InitialConditions, MissingStorage, PassiveInitialCondition, PassiveKind,
    };
    use model_normalize_ngspice::waveform::{PreparedWaveform, SymbolicPoint, WaveformSpec};
    use ngspice_rs::netlist::prepared_initialization::PreparedInitialConditions;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deck.cir");
    std::fs::write(
        &path,
        "geometry sweep with independent capacitor bias
v1 supply 0 2
r1 supply p 10
c1 p 0 1n
vd d 0 0.1
vg g 0 0.8
m1 d g 0 0 nch l=1u w=1u
.model nch.1 nmos level=8 lmin=0.5u lmax=2u wmin=0.5u wmax=2u vth0=0.3
.model nch.2 nmos level=8 lmin=0.5u lmax=2u wmin=2u wmax=5u vth0=0.5
.dc @m1[w] 1u 3u 1u
.end
",
    )
    .unwrap();
    let mut netlist = Parser::new().parse_file(&path).unwrap();
    PreparedInitialConditions::attach(
        &mut netlist,
        dir.path(),
        InitialConditions {
            devices: vec![PassiveInitialCondition {
                missing_storage: MissingStorage::NodeHints,
                artifact: "deck.cir".to_owned(),
                scope: Vec::new(),
                device: "c1".to_owned(),
                kind: PassiveKind::Capacitor,
                target: Some("3".to_owned()),
                branch: Some("1".to_owned()),
                mode: None,
            }],
            ..InitialConditions::default()
        },
        [(AnalysisKind::DcSweep, false)],
    )
    .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let waveform = PreparedWaveform {
        artifact: "deck.cir".to_owned(),
        scope: Vec::new(),
        source: "v1".to_owned(),
        spec: WaveformSpec::PiecewiseLinear {
            segments: "1".to_owned(),
            points: vec![
                SymbolicPoint {
                    time: "0".to_owned(),
                    value: "1".to_owned(),
                },
                SymbolicPoint {
                    time: "1u".to_owned(),
                    value: "4".to_owned(),
                },
            ],
        },
        dc_override: Some("10".to_owned()),
        idle_in_pop: "0".to_owned(),
    };
    let plan = config.plan_for(&netlist.analyses[0]).unwrap();
    for (declarations, supply) in [(&[][..], 2.), (std::slice::from_ref(&waveform), 10.)] {
        let mut circuit = config
            .prepared_circuit(
                &netlist,
                ngspice_rs::devices::subckt::PreparedDeclarations {
                    waveforms: declarations,
                    ..Default::default()
                },
            )
            .unwrap();
        let plot = plan.run(&mut circuit, &config.context()).unwrap();
        assert_eq!(plot.point_count(), 3);
        for point in 0..plot.point_count() {
            close(plot.value("v(supply)", point).unwrap().re, supply, 1e-10);
            close(plot.value("v(p)", point).unwrap().re, 3., 1e-10);
            close(
                plot.value("i(c1:bias)", point).unwrap().re,
                (supply - 3.) / 10.,
                1e-10,
            );
        }
    }
}

#[test]
fn instantaneous_physical_residual_releases_all_bias_constraints() {
    use ngspice_rs::devices::{Limit, TransientTiming};
    use ngspice_rs::maths::Vector;
    let (netlist, config, mut circuit) =
        deck("v1 a 0 2\nr1 a b 1k\nc1 b 0 1n\nr2 a c 1k\nl1 c 0 1m\n.tran 1u 10u");
    circuit
        .bind_passive_initialization(
            "c1",
            PassiveInitialization::new(
                StorageKind::Capacitor,
                Some(1.),
                [(AnalysisKind::Transient, voltage(1., None))],
            )
            .unwrap(),
        )
        .unwrap();
    circuit
        .bind_passive_initialization(
            "l1",
            PassiveInitialization::new(
                StorageKind::Inductor,
                Some(1e-3),
                [(AnalysisKind::Transient, current(1e-3, 0.))],
            )
            .unwrap(),
        )
        .unwrap();
    circuit
        .bind_node_initialization(
            "b",
            NodeInitialization::new(
                3.,
                netlist.location.clone(),
                [(AnalysisKind::Transient, 3., 0.1)],
            )
            .unwrap(),
        )
        .unwrap();
    circuit.finalize().unwrap();
    let names = circuit.unknown_names();
    let row = |name: &str| {
        names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap()
    };
    let (a, b, c, cb, lb) = (
        row("a"),
        row("b"),
        row("c"),
        row("c1#branch"),
        row("l1#branch"),
    );
    let mut state = Vector::zeros(circuit.unknown_count());
    for (row, value) in [(a, 2.), (b, 0.5), (c, 0.2), (cb, 0.25), (lb, 3e-3)] {
        state.set(row, value).unwrap();
    }
    let mut model = config.context().model_context();
    model.bias_analysis = Some(AnalysisKind::Transient);
    let run = circuit.prepare_run();
    let mut dae = run
        .prepare_physical_dae(&model, TransientTiming::new(1e-6, 10e-6).unwrap())
        .unwrap();
    let point = dae.evaluate(&state, 1e-6, Limit::Right).unwrap();
    close(point.residual.as_slice()[a], 3.3e-3, 1e-12);
    close(point.residual.as_slice()[b], -1.5e-3, 1e-12);
    close(point.residual.as_slice()[c], 1.2e-3, 1e-12);
    close(point.residual.as_slice()[cb], 0.25, 1e-12);
    close(point.residual.as_slice()[lb], 0.2, 1e-12);
    close(point.conductance.get(cb, cb), 1., 1e-12);
    close(point.conductance.get(b, cb), 0., 1e-12);
    close(point.conductance.get(b, b), 1e-3, 1e-12);
    close(point.charge.as_slice()[b], 0.5e-9, 1e-20);
    close(point.charge.as_slice()[lb], -3e-6, 1e-15);
}

#[test]
fn polynomial_bias_reuses_storage_branches_and_releases_constraints() {
    for (cap_g, ind_g) in [(None, 0.), (Some(500.), 1e-3)] {
        let (n, config, mut c) = deck(
            "vc a 0 dc 2 ac 1\nrc a b 10\nc1 b 0 cm 1u\n.model cm pspice_cap(vc1=0.25 vc2=0.1)\nvl d 0 dc 3 ac 1\nrl d e 10\nl1 e 0 lm 2m\n.model lm pspice_ind(il1=0.25 il2=0.1)\n.op\n.ac lin 1 1k 1k\n.dc vc 2 3 1\n.tran 10n 1u 0 10n",
        );
        let kinds = [
            AnalysisKind::OperatingPoint,
            AnalysisKind::Ac,
            AnalysisKind::Transient,
        ];
        c.bind_passive_initialization(
            "c1",
            PassiveInitialization::new(
                StorageKind::Capacitor,
                Some(-1.),
                kinds.map(|kind| (kind, voltage(-1., cap_g))),
            )
            .unwrap(),
        )
        .unwrap();
        c.bind_passive_initialization(
            "l1",
            PassiveInitialization::new(
                StorageKind::Inductor,
                Some(-0.2),
                kinds.map(|kind| (kind, current(-0.2, ind_g))),
            )
            .unwrap(),
        )
        .unwrap();
        c.finalize().unwrap();
        let initial_v = cap_g.map_or(-1., |g| (2. - 10. * g) / (1. + 10. * g));
        let initial_i = (-0.2 + 3. * ind_g) / (1. + 10. * ind_g);
        let plans = config.plans_for(&n.analyses).unwrap();
        let op = plans[0].run(&mut c, &config.context()).unwrap();
        close(last(&op, "v(b)"), initial_v, 1e-10);
        close(last(&op, "i(c1)"), (2. - initial_v) / 10., 1e-10);
        close(last(&op, "v(e)"), 3. - 10. * initial_i, 1e-10);
        close(last(&op, "i(l1)"), initial_i, 1e-10);

        let ac = plans[1].run(&mut c, &config.context()).unwrap();
        let omega = std::f64::consts::TAU * 1000.;
        let cv = 1e-6 * (1. + 0.25 * initial_v + 0.1 * initial_v * initial_v);
        let li = 2e-3 * (1. + 0.25 * initial_i + 0.1 * initial_i * initial_i);
        let rc = omega * 10. * cv;
        let xl = omega * li;
        let cap = ac.value("v(b)", 0).unwrap();
        let ind = ac.value("i(l1)", 0).unwrap();
        close(cap.re, (1. + rc * rc).recip(), 1e-10);
        close(cap.im, -rc / (1. + rc * rc), 1e-10);
        close(ind.re, 10. / (100. + xl * xl), 1e-10);
        close(ind.im, -xl / (100. + xl * xl), 1e-10);

        let dc = plans[2].run(&mut c, &config.context()).unwrap();
        close(last(&dc, "v(b)"), 3., 1e-10);
        close(last(&dc, "i(c1)"), 0., 1e-10);
        close(last(&dc, "v(e)"), 0., 1e-10);
        close(last(&dc, "i(l1)"), 0.3, 1e-10);
        let tran = plans[3].run(&mut c, &config.context()).unwrap();
        close(tran.value("v(b)", 0).unwrap().re, initial_v, 1e-8);
        close(tran.value("i(l1)", 0).unwrap().re, initial_i, 1e-8);
        assert!(last(&tran, "v(b)") > initial_v);
        assert!(last(&tran, "i(l1)") > initial_i);
    }
}

#[test]
fn polynomial_uic_distinguishes_omission_zero_and_signed_coordinates() {
    for explicit in [None, Some(0.), Some(-1.)] {
        let (n, config, mut c) = deck(
            "i1 0 b 1m\nc1 b 0 cm 1u\n.model cm pspice_cap(vc1=0.25)\nv1 e 0 1\nl1 e 0 lm 1m\n.model lm pspice_ind(il1=0.25)\n.ic v(b)=2\n.tran 1u 1m 0 1u uic",
        );
        c.bind_passive_initialization(
            "c1",
            PassiveInitialization::new(
                StorageKind::Capacitor,
                explicit,
                [(AnalysisKind::Transient, voltage(9., None))],
            )
            .unwrap(),
        )
        .unwrap();
        c.bind_passive_initialization(
            "l1",
            PassiveInitialization::new(
                StorageKind::Inductor,
                explicit,
                [(AnalysisKind::Transient, current(9., 0.))],
            )
            .unwrap(),
        )
        .unwrap();
        c.finalize().unwrap();
        let p = config
            .plan_for(&n.analyses[0])
            .unwrap()
            .run(&mut c, &config.context())
            .unwrap();
        for (name, initial) in [
            ("v(b)", explicit.unwrap_or(2.)),
            ("i(l1)", explicit.unwrap_or(0.)),
        ] {
            for index in 0..p.point_count() {
                let t = p.value("time", index).unwrap().re;
                let start = initial + 4.;
                close(
                    p.value(name, index).unwrap().re,
                    (8000_f64.mul_add(t, start * start)).sqrt() - 4.,
                    5e-4,
                );
            }
        }
    }
}

#[test]
fn polynomial_exact_history_is_independent_of_reported_bias() {
    for derivative in [false, true] {
        let body = format!(
            "i1 0 b 1m\nc1 b 0 cm 1u\n.model cm pspice_cap(vc1=0.25)\nv1 e 0 1\nl1 e 0 lm 1m\n.model lm pspice_ind(il1=0.25)\n{}\n.tran 1u 100u 0 1u",
            if derivative {
                "vtime z 0 PWL(0 0 100u 1)\nb1 obs 0 v=100u*ddt(v(z))\nr1 obs 0 1k"
            } else {
                ""
            },
        );
        let (n, config, mut c) = deck(&body);
        for (name, kind, constraint) in [
            ("c1", StorageKind::Capacitor, voltage(-0.5, Some(1e-3))),
            ("l1", StorageKind::Inductor, current(-0.5, 0.1)),
        ] {
            c.bind_passive_initialization(
                name,
                PassiveInitialization::new(
                    kind,
                    Some(-0.5),
                    [(AnalysisKind::Transient, constraint)],
                )
                .unwrap()
                .with_exact_history()
                .unwrap(),
            )
            .unwrap();
        }
        c.finalize().unwrap();
        let p = config
            .plan_for(&n.analyses[0])
            .unwrap()
            .run(&mut c, &config.context())
            .unwrap();
        close(p.value("v(b)", 0).unwrap().re, 0.5, 1e-9);
        close(p.value("i(l1)", 0).unwrap().re, -0.4, 1e-9);
        assert!(p.point_count() > 5);
        for index in 1..p.point_count() {
            let time = p.value("time", index).unwrap().re;
            let expected = 8000_f64.mul_add(time, 12.25).sqrt() - 4.;
            close(p.value("v(b)", index).unwrap().re, expected, 5e-5);
            close(p.value("i(l1)", index).unwrap().re, expected, 5e-5);
        }
    }
}

#[test]
fn parameter_candidates_rebind_the_source_prepared_bias() {
    use model_normalize_ngspice::passive_ic::{
        InitialConditions, PassiveInitialCondition, PassiveKind,
    };
    use ngspice_rs::analysis::optimization::parameters::CircuitParameterPlan;
    use ngspice_rs::netlist::{
        eval::EvalBudget, prepared_initialization::PreparedInitialConditions,
    };

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("deck.cir");
    std::fs::write(
        &path,
        "candidate IC\n.param target=1\nv1 in 0 10\nr1 in out 1k\nc1 out 0 1u\n.op\n.end\n",
    )
    .unwrap();
    let mut source = Parser::new().parse_file(&path).unwrap();
    PreparedInitialConditions::attach(
        &mut source,
        dir.path(),
        InitialConditions {
            devices: vec![PassiveInitialCondition {
                artifact: "deck.cir".into(),
                scope: Vec::new(),
                device: "c1".into(),
                kind: PassiveKind::Capacitor,
                target: Some("{target}".into()),
                branch: Some("1".into()),
                mode: None,
            }],
            ..InitialConditions::default()
        },
        [(AnalysisKind::OperatingPoint, false)],
    )
    .unwrap();
    let plan = CircuitParameterPlan::new(source, &["target"]).unwrap();
    let mut candidate = plan.session().unwrap();
    for target in [1., -2., 0., 1.] {
        candidate
            .update(&[target], &mut EvalBudget::default())
            .unwrap();
        let request = candidate
            .config()
            .plan(ngspice_rs::analysis::AnalysisInput::new(
                AnalysisKind::OperatingPoint,
            ))
            .unwrap();
        let context = candidate.analysis_context();
        let plot = request.run(candidate.circuit_mut(), &context).unwrap();
        close(last(&plot, "v(out)"), target, 1e-10);
        close(last(&plot, "i(c1:bias)"), (10. - target) / 1e3, 1e-10);
    }
}
