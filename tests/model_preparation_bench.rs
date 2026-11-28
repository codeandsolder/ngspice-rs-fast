//! Shared-card setup and lookup workloads. Ignored benchmark cases can be
//! executed independently under DHAT to compare allocation counts.
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use ngspice_rs::devices::{Circuit, ModelResolver, bsim4};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;

const COUNT: usize = 2048;

fn deck(level: u8) -> Netlist {
    let mut text = format!("shared model setup\n.model shared nmos level={level} vto=0.6\n");
    if level == 54 {
        text =
            "shared model setup\n.model shared nmos level=54 vth0=0.6 vtho=0.5 toxe=1e-8\n".into();
    }
    for i in 0..COUNT {
        writeln!(text, "m{i} d g 0 0 shared l=1u w=10u").unwrap();
    }
    text.push_str(".end\n");
    Parser::new()
        .parse_deck(&parse_deck_text(Path::new("preparation-bench.cir"), &text))
        .unwrap()
}

fn build(level: u8) {
    let n = deck(level);
    let start = Instant::now();
    let circuit = Circuit::from_netlist(black_box(&n)).unwrap();
    assert_eq!(circuit.device_count(), COUNT);
    println!("level {level}: {COUNT} instances in {:?}", start.elapsed());
    black_box(circuit);
}

#[test]
#[ignore = "opt-in scalar MOS setup workload"]
fn mos_setup() {
    build(1);
}

#[test]
#[ignore = "opt-in BSIM4 setup workload"]
fn bsim4_setup() {
    build(54);
}

fn input_projection(shared: bool) {
    let n = deck(54);
    let resolver = ModelResolver::new(&n.models).unwrap();
    let start = Instant::now();
    for instance in &n.devices {
        let fresh;
        let r = if shared {
            &resolver
        } else {
            fresh = ModelResolver::new(&n.models).unwrap();
            &fresh
        };
        let selected = r.resolve(instance).unwrap().unwrap();
        let params = bsim4::model_parameters(&selected).unwrap();
        for keyword in ["vth0", "vtho", "toxe", "VTH0", "not_present"] {
            black_box(params.inputs.get(black_box(keyword)));
        }
        black_box(params);
    }
    println!(
        "shared={shared}: {COUNT} model projections in {:?}",
        start.elapsed()
    );
}

#[test]
#[ignore = "opt-in shared declaration projection workload"]
fn shared_projection() {
    input_projection(true);
}

#[test]
#[ignore = "opt-in independent declaration projection workload"]
fn independent_projection() {
    input_projection(false);
}

#[test]
fn ordered_inputs_share_only_within_a_declaration() {
    let n = deck(54);
    let resolver = ModelResolver::new(&n.models).unwrap();
    let first =
        bsim4::model_parameters(&resolver.resolve(&n.devices[0]).unwrap().unwrap()).unwrap();
    for instance in &n.devices[1..] {
        let next = bsim4::model_parameters(&resolver.resolve(instance).unwrap().unwrap()).unwrap();
        assert!(std::ptr::eq(first.inputs.applied(), next.inputs.applied()));
        assert_eq!(next.inputs.get("VTHO"), next.inputs.get("vth0"));
        assert_eq!(next.inputs.get("vth0").unwrap().parameter.name, "vtho");
    }
    let second = ModelResolver::new(&n.models).unwrap();
    let independent =
        bsim4::model_parameters(&second.resolve(&n.devices[0]).unwrap().unwrap()).unwrap();
    assert_eq!(first, independent);
    assert!(!std::ptr::eq(
        first.inputs.applied(),
        independent.inputs.applied()
    ));
}

fn scalar_projection(shared: bool) {
    let n = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("scalar-preparation-bench.cir"),
            "scalar preparation\nr1 d 0 shared l=1u w=2u\n.model shared r(rsh=100 tc1=0.01 tc2=0.0001 defw=2u narrow=0.1u)\n.end\n",
        ))
        .unwrap();
    let resolver = ModelResolver::new(&n.models).unwrap();
    let start = Instant::now();
    for _ in 0..COUNT {
        let fresh;
        let r = if shared {
            &resolver
        } else {
            fresh = ModelResolver::new(&n.models).unwrap();
            &fresh
        };
        let selected = r.resolve(&n.devices[0]).unwrap().unwrap();
        black_box(
            selected
                .passive_parameters(black_box(&n.devices[0]), 1.)
                .unwrap(),
        );
    }
    println!(
        "scalar shared={shared}: {COUNT} projections in {:?}",
        start.elapsed()
    );
}

#[test]
#[ignore = "opt-in shared scalar model projection workload"]
fn shared_scalar_projection() {
    scalar_projection(true);
}

#[test]
#[ignore = "opt-in independent scalar model projection workload"]
fn independent_scalar_projection() {
    scalar_projection(false);
}
