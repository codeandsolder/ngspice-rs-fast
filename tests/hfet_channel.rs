//! Independent HFET channel oracle for the two model families sharing the
//! intrinsic channel calculation. Reverse bias retains C's normalization
//! convention, including the gate voltage selection after normalizing Vds.

use ngspice_conformance::ngspice::{Ngspice, Scratch, batch_reference_sample};
use ngspice_rs::analysis::{Plot, RawFile, RunConfig};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::Complex;
use std::fmt::Write;
use std::fs;
use std::path::Path;

fn port(deck: &str) -> Plot {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("hfet-channel.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    config
        .plan_for(&netlist.analyses[0])
        .unwrap()
        .run(&mut circuit, &config.context())
        .unwrap()
}

fn compare(reference: &Plot, actual: &Plot, label: &str) {
    assert_eq!(reference.point_count(), actual.point_count(), "{label}");
    for (column, variable) in reference.variables().iter().enumerate() {
        let name = &variable.name;
        let expected = reference.column_at(column).unwrap();
        let got = actual
            .column(name)
            .unwrap_or_else(|| panic!("{label}: no {name}"));
        for (a, b) in got.iter().zip(expected) {
            let [re, im] = batch_reference_sample(column, name, [b.re, b.im]);
            let b = Complex::new(re, im);
            assert!(
                (*a - b).magnitude() <= 1e-6 * b.magnitude() + 1e-15,
                "{label}: {name}: {a} != {b}"
            );
        }
    }
}

/// Sweep both polarities, both density branches, independent instance
/// temperatures, weak inversion and saturation, and both drain orientations.
/// An ideal gate AC source exposes the intrinsic capacitances as currents.
#[test]
#[ignore = "requires independent NGSPICE_BIN"]
fn both_hfet_channels_match_c_at_forward_and_reverse_bias() {
    let binary = Ngspice::pinned().unwrap();
    let scratch = Scratch::new("hfet-channel-oracle").unwrap();
    for level in [5, 6] {
        for (family, polarity) in [("nhfet", 1.), ("phfet", -1.)] {
            for density in ["", "eta2=1.5 d2=0.1u"] {
                for temp in [-25., 27., 85.] {
                    let leakage = if level == 5 {
                        "js1d=0 js2d=0 js1s=0 js2s=0"
                    } else {
                        "js=0 cf=0.02p knmax=1e12"
                    };
                    let mut deck = format!(
                        "HFET channel grid\n.model hm {family} level={level} {density} {leakage} ggr=0 rdi=5 rsi=7\n.options reltol=1e-10 abstol=1e-15 vntol=1e-12\n"
                    );
                    let mut saves = String::new();
                    for (index, (vgs, vds)) in [-0.25, 0., 0.15, 0.3, 0.6]
                        .into_iter()
                        .flat_map(|vgs| [-1., -0.01, 0.01, 1.].map(move |vds| (vgs, vds)))
                        .enumerate()
                    {
                        writeln!(deck, "vd{index} d{index} 0 {}\nvg{index} g{index} 0 dc {} ac 1\nz{index} d{index} g{index} 0 hm l=2u w=20u temp={temp}", polarity * vds, polarity * vgs).unwrap();
                        write!(saves, " i(vd{index}) i(vg{index})").unwrap();
                    }
                    for analysis in [".op", ".ac lin 2 1e6 1e9"] {
                        let deck = format!("{deck}{analysis}\n.save{saves}\n.end\n");
                        let label = format!("{level} {family} {density} {temp} {analysis}");
                        fs::write(
                            scratch.path().join("c.cir"),
                            deck.replace("\n.end\n", "\n.options sparse\n.end\n"),
                        )
                        .unwrap();
                        let output = binary
                            .run(scratch.path(), &["-b", "-r", "c.raw", "c.cir"])
                            .unwrap();
                        output.require_success(&label).unwrap();
                        let raw = RawFile::load(scratch.path().join("c.raw")).unwrap();
                        assert_eq!(raw.plots.len(), 1, "{label}");
                        compare(&raw.plots[0].plot, &port(&deck), &label);
                        fs::remove_file(scratch.path().join("c.raw")).unwrap();
                    }
                }
            }
        }
    }
}
