//! Deck-bound mixed-signal execution, checked against Boolean truth tables
//! and explicitly specified receiver and propagation timing.
use std::path::Path;

use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};

fn circuit(body: &str) -> SpiceResult<Circuit> {
    let deck = Parser::new().parse_deck(&parse_deck_text(
        Path::new("mixed.cir"),
        &format!("native mixed test\n{body}\n.end\n"),
    ))?;
    Circuit::from_netlist(&deck)
}

fn run(circuit: &mut Circuit, kind: AnalysisKind, args: &[&str]) -> SpiceResult<Plot> {
    AnalysisInput::with_arguments(kind, args.iter().copied())
        .resolve()?
        .run(circuit, &AnalysisContext::default())
}

fn output(plot: &Plot, index: usize) -> f64 {
    plot.value("v(out)", index).unwrap().re
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
}

#[test]
fn native_truth_tables_and_initial_latch_state() -> SpiceResult<()> {
    for op in 0..7 {
        for a in [false, true] {
            for b in [false, true] {
                for ic in [false, true] {
                    let expected = match op {
                        0 => a && b,
                        1 => a || b,
                        2 => !(a && b),
                        3 => !(a || b),
                        4 => {
                            if b {
                                false
                            } else {
                                a || ic
                            }
                        }
                        5 => {
                            if a {
                                true
                            } else {
                                !b && ic
                            }
                        }
                        _ => {
                            if a == b {
                                ic
                            } else {
                                a
                            }
                        }
                    };
                    let mut c = circuit(&format!(
                        "va a ref {}\nvb b ref {}\nvref ref 0 7\n\
                         a1 %vd[a ref b ref] logic gate\n\
                         .model gate preprocess_logic_event(operation={op} delay=1u ic={} low=2 high=3)\n\
                         a2 logic %vd(out ref) bridge\n\
                         .model bridge preprocess_logic_voltage(out_low=-2 out_high=5)\n\
                         r1 out ref 1k",
                        if a { 4 } else { 0 },
                        if b { 4 } else { 0 },
                        u8::from(ic)
                    ))?;
                    for _ in 0..2 {
                        let plot = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
                        close(output(&plot, 0), if expected { 12. } else { 5. });
                        assert!(c.last_digital_stats().unwrap().evaluations > 0);
                        assert!(c.nodes().get("logic").is_none());
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn equality_initial_receivers_and_inverted_rails() -> SpiceResult<()> {
    let mut c = circuit(
        "va a 0 3\nvb b 0 2\n\
         a1 %v[a b] logic gate\n\
         .model gate preprocess_logic_event(operation=0 low=2 high=3 inverted=[true false] rx_ic=[1 1])\n\
         a2 logic out bridge\n\
         .model bridge preprocess_logic_voltage(out_low=4 out_high=-1)\n\
         r1 out 0 1k",
    )?;
    close(
        output(&run(&mut c, AnalysisKind::OperatingPoint, &[])?, 0),
        -1.,
    );
    assert!(run(&mut c, AnalysisKind::Ac, &["lin", "1", "1", "1"]).is_err());
    Ok(())
}

#[test]
fn source_transient_distinguishes_inertial_and_transport() -> SpiceResult<()> {
    for inertial in [false, true] {
        let mut c = circuit(&format!(
            "va a 0 pwl(0 0 1u 0 1.1u 4 1.4u 4 1.5u 0 4u 0)\n\
             a1 %v[a] logic gate\n\
             .model gate preprocess_logic_event(operation=0 delay=1u inertial={inertial} low=2 high=3)\n\
             a2 logic out bridge\n\
             .model bridge preprocess_logic_voltage(out_low=0 out_high=5)\n\
             r1 out 0 1k"
        ))?;
        let plot = run(&mut c, AnalysisKind::Transient, &["10n", "4u", "0", "10n"])?;
        let mut high_samples = 0;
        for i in 0..plot.point_count() {
            let time = plot.value("time", i).unwrap().re;
            if (2.12e-6..2.42e-6).contains(&time) {
                close(output(&plot, i), if inertial { 0. } else { 5. });
                high_samples += 1;
            } else if time < 2.06e-6 || time > 2.48e-6 {
                close(output(&plot, i), 0.);
            }
        }
        assert!(high_samples > 10);
    }
    Ok(())
}

#[test]
fn nested_formals_keep_digital_ground_names_distinct() -> SpiceResult<()> {
    let mut c = circuit(
        ".subckt gate in state\n\
         a1 %v[in] state gate_model\n\
         .model gate_model preprocess_logic_event(operation=0)\n\
         .ends\n\
         .subckt outer in state\n\
         x1 in state gate\n\
         .ends\n\
         va a 0 4\nvb b 0 0\n\
         xhi a gnd outer\nxlo b 0 outer\n\
         a2 gnd out bridge\na3 0 other bridge\n\
         .model bridge preprocess_logic_voltage(out_low=0 out_high=5)\n\
         r1 out 0 1k\nr2 other 0 1k",
    )?;
    let plot = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    close(output(&plot, 0), 5.);
    close(plot.value("v(other)", 0).unwrap().re, 0.);
    Ok(())
}
