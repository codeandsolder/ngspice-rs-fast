//! C-device instance asks are not aliases for isolated external MNA loads.
//!
//! The committed C ngspice-47 reference below exercises BSIM4 gate tunnelling,
//! body networks and multiple coexisting semiconductor devices. The C model's
//! CD ask is retained independently of whichever current leaves its pin.

use std::path::Path;

use ngspice_rs::analysis::{Plot, RunConfig};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;

fn plot_operating_point(deck: &str, observations: &[&str]) -> Plot {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("intrinsic-asks.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    circuit
        .set_observations(
            &observations
                .iter()
                .map(|s| (*s).to_owned())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    request
        .run_plots(&mut circuit, &config.context())
        .unwrap()
        .into_iter()
        .last()
        .unwrap()
}

fn bsim4_with_intrinsic_and_terminal_asks() -> Plot {
    let original = include_str!("../conformance/netlists/m11_bsim4_op.cir");
    let deck = original.replace(
        ".op\n.end\n",
        ".save @m1[id] @m1[gm] @m1[ibd] @m1[igd] @m1[isub] @m1[i] @m1[current] i(m1)\n.op\n.end\n",
    );
    assert_ne!(
        deck, original,
        "missing committed BSIM4 OP fixture terminator"
    );
    plot_operating_point(
        &deck,
        &[
            "@m1[id]",
            "@m1[gm]",
            "@m1[ibd]",
            "@m1[igd]",
            "@m1[isub]",
            "@m1[i]",
            "@m1[current]",
            "i(m1)",
        ],
    )
}

#[test]
fn bsim4_intrinsic_cd_and_gm_match_independent_c_oracle() {
    let plot = bsim4_with_intrinsic_and_terminal_asks();
    for (name, expected) in [
        ("@m1[id]", 2.743483712211524e-4_f64),
        ("@m1[gm]", 8.464358463227711e-4),
        ("@m1[ibd]", -1.300009999988938e-12),
        ("@m1[igd]", 0.),
        ("@m1[isub]", 0.),
    ] {
        let got = plot
            .value(name, 0)
            .unwrap_or_else(|| panic!("missing {name}"));
        assert!(
            (got.re - expected).abs() <= expected.abs() * 1e-8 + 3e-18,
            "{name}: Rust {:e}, C {expected:e}",
            got.re
        );
        assert_eq!(got.im, 0., "{name}: unexpected imaginary part");
    }
    let channel = plot.value("@m1[id]", 0).unwrap().re;
    let external = plot.value("@m1[i]", 0).unwrap().re;
    assert!(channel.is_finite() && external.is_finite());
    // Explicit external aliases all use the same physical terminal row.
    for name in ["@m1[current]", "i(m1)"] {
        let current = plot.value(name, 0).unwrap().re;
        assert_eq!(current.to_bits(), external.to_bits(), "{name}");
    }
    // C's CD value and the external residual are separate contracts. They
    // may coincide for some bias/topology; equality is never used to route.
}

#[test]
fn jfet_and_mesfet_intrinsic_currents_match_independent_c_reference() {
    // Values from ngspice-47's native *ask.c implementations and their
    // own two-port series-resistor and gate-diode MNA stamps, not from
    // this port's load/observation kernels.
    for (deck, expected) in [
        (
            "JFET native asks with series resistors and gate junction\nvd d 0 4\nvg g 0 -0.5\nvs s 0 0\nj1 d g s jm area=2 m=3\n.model jm njf(vto=-2 beta=1e-3 lambda=0.02 rd=10 rs=5 is=1e-12 pb=0.8)\n.options abstol=1e-15 reltol=1e-8\n.save @j1[id] @j1[ig] @j1[is] @j1[igd] i(vd) i(vg) i(vs)\n.op\n.end\n",
            &[
                ("@j1[id]", 1.433910643688377e-2),
                ("@j1[ig]", -2.696311055610332e-11),
                ("@j1[is]", -1.433910640992066e-2),
                ("@j1[igd]", -1.942830291161683e-11),
                ("i(vd)", -1.433910643688341e-2),
                ("i(vg)", 2.696311055610332e-11),
            ][..],
        ),
        (
            "MESFET native asks with series resistors and gate junction\nvd d 0 4\nvg g 0 -0.5\nvs s 0 0\nz1 d g s zm area=2 m=3\n.model zm nmf(vto=-2 beta=1e-3 lambda=0.02 b=0.3 alpha=2.5 rd=10 rs=5 is=1e-12)\n.options abstol=1e-15 reltol=1e-8\n.save @z1[cd] @z1[cg] @z1[is] @z1[cgd] i(vd) i(vg) i(vs)\n.op\n.end\n",
            &[
                ("@z1[cd]", 9.956747847058347e-3),
                ("@z1[cg]", -2.697404387981658e-11),
                ("@z1[is]", -9.956747820084303e-3),
                ("@z1[cgd]", -1.945021471215920e-11),
                ("i(vd)", -9.956747847058356e-3),
                ("i(vg)", 2.697404387981658e-11),
            ][..],
        ),
    ] {
        let observations: Vec<_> = expected.iter().map(|&(name, _)| name).collect();
        let plot = plot_operating_point(deck, &observations);
        for &(name, reference) in expected {
            let value = plot
                .value(name, 0)
                .unwrap_or_else(|| panic!("missing native C ask {name}"));
            assert!(
                (value.re - reference).abs() <= reference.abs() * 2e-6 + 5e-16,
                "{name}: Rust {:e}, C {reference:e}",
                value.re
            );
            assert_eq!(value.im, 0.);
        }
    }
}

#[test]
fn unimplemented_bjt_c_collector_ask_is_not_replaced_with_a_port_residual() {
    let deck = "BJT intrinsic ask\nvcc c 0 5\nvb b 0 0.7\nq1 c b 0 qn\n.model qn npn(is=1e-14 bf=100)\n.op\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("bjt-ask.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    circuit.set_observations(&["@q1[ic]".to_owned()]).unwrap();
    let error = request
        .run_plots(&mut circuit, &config.context())
        .unwrap_err();
    assert!(
        error.to_string().contains("device observation @q1[ic]"),
        "unexpected BJT error: {error}"
    );
}

#[test]
fn malformed_intrinsic_queries_are_rejected_before_simulation() {
    let source = r"device observations
v1 a 0 1
r1 a 0 1k
.op
.end
";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("intrinsic-invalid.cir"), source))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    for name in ["@r1[]", "@r1[bad]tail", "@r1[wrong[nesting]]", "@"] {
        assert!(
            circuit.set_observations(&[name.to_owned()]).is_err(),
            "{name}"
        );
    }
}
