//! Independent candidates: dependency evaluation and validated layout boundaries.
use std::path::Path;

use ngspice_rs::analysis::RunConfig;
use ngspice_rs::analysis::config::parameters::{
    CandidateLayout, CandidatePlan, CandidateSetter, SetterSite,
};
use ngspice_rs::devices::ModelContext;
use ngspice_rs::netlist::{Parser, ast::Netlist, source::parse_deck_text};

fn parse(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("candidate.cir"),
            &format!("candidate\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn output(deck: &Netlist) -> Vec<ngspice_rs::analysis::Plot> {
    let config = RunConfig::from_netlist(deck).unwrap();
    let mut circuit = config.circuit(deck).unwrap();
    config
        .plan_for(&deck.analyses[0])
        .unwrap()
        .run_plots(&mut circuit, &config.context())
        .unwrap()
}

#[test]
fn independent_candidate_vectors_with_scoped_and_model_overrides_match_scalar_decks() {
    fn fixture(r: f64, t: f64, local: f64, model: f64, x: f64) -> Netlist {
        parse(&format!(
            ".param base={r} temp0={t} doubled={{base*2}}\n.option temp={{temp0}}\n\
            v1 in 0 1\nx1 in out cell scale={x}\nrload out 0 rm\n.model rm r(r={model})\n\
            .subckt cell a b params: scale=1\n.param local={local}\n\
            r1 a b {{doubled*scale+local}}\n.ends cell\n.op"
        ))
    }
    let deck = fixture(1000., 27., 3., 2000., 1.);
    let setters = [
        CandidateSetter {
            definition: Some(0),
            site: SetterSite::Parameter {
                card: 0,
                assignment: 0,
            },
        },
        CandidateSetter {
            definition: None,
            site: SetterSite::Model {
                model: 0,
                parameter: 0,
            },
        },
        CandidateSetter {
            definition: None,
            site: SetterSite::Device {
                device: 1,
                parameter: 0,
            },
        },
    ];
    let plan = CandidatePlan::compile(&deck, &["base", "TEMP0"], &setters).unwrap();
    assert_eq!(plan.input_count(), 5);
    let mut session = plan.session().unwrap();
    for vector in [
        [1200., 40., 10., 2400., 2.],
        [1800., 60., 40., 3600., 3.],
        [1200., 40., 10., 2400., 2.],
    ] {
        let mut candidate = session.prepare(&vector).unwrap();
        assert_eq!(candidate.layout, CandidateLayout::Same);
        assert_eq!(candidate.config.context().temperature, vector[1]);
        assert_eq!(candidate.evaluated_definitions, 1);
        let plots = candidate.analyses[0]
            .run_plots(&mut candidate.circuit, &candidate.config.context())
            .unwrap();
        let expected = output(&fixture(
            vector[0], vector[1], vector[2], vector[3], vector[4],
        ));
        assert_eq!(plots, expected);
    }
    let repeat = session.prepare(&[1200., 40., 10., 2400., 2.]).unwrap();
    assert_eq!(repeat.evaluated_definitions, 0);
    assert!(session.prepare(&[f64::NAN, 40., 10., 2400., 2.]).is_err());
    assert!(session.prepare(&[1200., 40., 10., 2400.]).is_err());
    assert!(session.prepare(&[1200., -300., 10., 2400., 2.]).is_err());
    assert!(session.prepare(&[1200., 40., 10., 2400., 2.]).is_ok());
}

#[test]
fn selectors_model_bins_and_private_nodes_require_distinct_layouts() {
    let deck = parse(
        "vd d 0 1\nvg g 0 1\nm1 d g 0 0 nch w=1u l=.7u\n.model nch.1 nmos level=8 lmin=.5u lmax=1u wmin=.5u wmax=5u vth0=.3\n.model nch.2 nmos level=8 lmin=1u lmax=5u wmin=.5u wmax=5u vth0=.5\n.op",
    );
    let length = deck.devices[2]
        .parameters
        .iter()
        .position(|p| p.name == "l")
        .unwrap();
    let plan = CandidatePlan::compile(
        &deck,
        &[],
        &[CandidateSetter {
            definition: None,
            site: SetterSite::Device {
                device: 2,
                parameter: length,
            },
        }],
    )
    .unwrap();
    let mut session = plan.session().unwrap();
    let a = session.prepare(&[0.8e-6]).unwrap();
    let b = session.prepare(&[2e-6]).unwrap();
    assert_eq!(a.layout, CandidateLayout::Same);
    assert_eq!(b.layout, CandidateLayout::Changed);
    assert_eq!(a.circuit.device("m1").unwrap().model_name(), Some("nch.1"));
    assert_eq!(b.circuit.device("m1").unwrap().model_name(), Some("nch.2"));
    assert!(session.prepare(&[6e-6]).is_err());
    assert_eq!(
        session.prepare(&[0.8e-6]).unwrap().layout,
        CandidateLayout::Same
    );
    let deck = parse("v1 in 0 1\nr1 in out 1k\nd1 out 0 dm\n.model dm d(rs=0)\n.op");
    let plan = CandidatePlan::compile(
        &deck,
        &[],
        &[CandidateSetter {
            definition: None,
            site: SetterSite::Model {
                model: 0,
                parameter: 0,
            },
        }],
    )
    .unwrap();
    let mut session = plan.session().unwrap();
    let zero = session.prepare(&[0.]).unwrap();
    let series = session.prepare(&[10.]).unwrap();
    assert_eq!(zero.layout, CandidateLayout::Same);
    assert_eq!(series.layout, CandidateLayout::Changed);
    assert!(series.circuit.unknown_count() > zero.circuit.unknown_count());
}

#[test]
fn bound_device_aliases_reuse_canonical_setter_and_reject_foreign_topology() {
    let deck = parse("v1 in 0 1\nr1 in out 1k\nr2 out 0 1k\n.op");
    let config = RunConfig::from_netlist(&deck).unwrap();
    let mut a = config.circuit(&deck).unwrap();
    let b = config.circuit(&deck).unwrap();
    let target = a.bind_instance_target("R1", "R").unwrap();
    assert_eq!(target, a.bind_resistor_target("r1").unwrap());
    assert_eq!(target.parameter(), "resistance");
    assert!(target.at(f64::INFINITY).is_err());
    for value in [2000., 3000., 1000.] {
        let context = ModelContext::default()
            .with_instance_override(target.at(value).unwrap())
            .unwrap();
        assert!(a.prepare_run().validate_context(&context).is_ok());
        assert!(b.prepare_run().validate_context(&context).is_err());
    }
    a.swap_devices(1, 2).unwrap();
    a.finalize().unwrap();
    let context = ModelContext::default()
        .with_instance_override(target.at(1000.).unwrap())
        .unwrap();
    assert!(a.prepare_run().validate_context(&context).is_err());
}

#[test]
fn duplicate_sites_and_parallel_sessions() {
    let deck = parse(".param r=1000\nv1 in 0 1\nr1 in out {r}\nr2 out 0 1k\n.op");
    let setter = CandidateSetter {
        definition: None,
        site: SetterSite::Device {
            device: 2,
            parameter: 0,
        },
    };
    assert!(CandidatePlan::compile(&deck, &["r", "R"], &[]).is_err());
    assert!(CandidatePlan::compile(&deck, &["r"], &[setter, setter]).is_err());
    let plan = CandidatePlan::compile(&deck, &["r"], &[setter]).unwrap();
    std::thread::scope(|scope| {
        let run = |value| {
            let mut lane = plan.session().unwrap();
            let candidate = lane.prepare(&[value, 1000.]).unwrap();
            candidate.config.params().unwrap().get("r").unwrap()
        };
        let a = scope.spawn(|| run(2000.));
        let b = scope.spawn(|| run(3000.));
        assert_eq!(a.join().unwrap(), 2000.);
        assert_eq!(b.join().unwrap(), 3000.);
    });
}
