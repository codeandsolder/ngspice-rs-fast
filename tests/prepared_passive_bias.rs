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
    let plans = config.plans_for(&n.analyses).unwrap();
    let op = plans[0].run(&mut c, &config.context()).unwrap();
    // Three equal conductances to 2, 0 and the first node hint 1 V.
    close(last(&op, "v(d)"), 1., 1e-12);
    let dc = plans[1].run(&mut c, &config.context()).unwrap();
    close(last(&dc, "v(d)"), 2., 1e-12);
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
