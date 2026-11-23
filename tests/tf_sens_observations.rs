//! Cross-analysis output-probe binding regression for `.tf` and `.sens`.
//! The two drivers share node/branch validation, not input-source and
//! sensitivity rules; compare physical results after independently binding.

use std::path::Path;

use ngspice_rs::analysis::{Plot, RunConfig};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{Complex, SpiceResult};

const DIVIDER: &str = "v1 in 0 dc 10\nr1 in out 1k\nr2 out 0 2k";
const REORDERED: &str = "r2 out 0 2k\nr1 in out 1k\nv1 in 0 dc 10";

fn run(body: &str, card: &str) -> SpiceResult<Plot> {
    let deck = Parser::new().parse_deck(&parse_deck_text(
        Path::new("tf-sens-observations.cir"),
        &format!("shared output probes\n{body}\n{card}\n.end\n"),
    ))?;
    let config = RunConfig::from_netlist(&deck)?;
    let plans = config.plans_for(&deck.analyses)?;
    let mut circuit = config.circuit(&deck)?;
    plans[0].run(&mut circuit, &config.context())
}

fn result(plot: &Plot, name: &str) -> Complex {
    plot.value(name, 0)
        .unwrap_or_else(|| panic!("missing {name}"))
}

fn transfer(plot: &Plot) -> [f64; 3] {
    assert_eq!(plot.plotname, "Transfer Function");
    assert_eq!(plot.variable_count(), 3);
    let row = plot.row(0).unwrap();
    [row[0].re, row[1].re, row[2].re]
}

fn close(a: f64, b: f64, relative: f64) {
    let scale = a.abs().max(b.abs()).max(1e-12);
    assert!((a - b).abs() <= scale * relative, "{a} != {b}");
}

#[test]
fn voltage_aliases_and_case_preserve_tf_and_sens_physics() {
    let baseline = transfer(&run(DIVIDER, ".tf v(out) v1").unwrap());
    for body in [DIVIDER, REORDERED] {
        for expression in ["v(out)", "V(OUT)", "v(out,0)", "v(Out,GND)"] {
            let card = format!(".tf {expression} V1");
            let actual = transfer(&run(body, &card).unwrap());
            for (got, expected) in actual.into_iter().zip(baseline) {
                close(got, expected, 2e-12);
            }
        }
        let baseline = run(body, ".sens v(out)").unwrap();
        for expression in ["v(out,0)", "V(Out,GnD)"] {
            let card = format!(".sens {expression} r1 r2");
            let filtered = run(body, &card).unwrap();
            for variable in ["v(r1)", "v(r2)"] {
                close(
                    result(&filtered, variable).re,
                    result(&baseline, variable).re,
                    2e-8,
                );
            }
        }
    }
}

#[test]
fn differential_tf_is_the_difference_of_two_voltage_probes() {
    for body in [DIVIDER, REORDERED] {
        let positive = transfer(&run(body, ".tf v(in) v1").unwrap());
        let negative = transfer(&run(body, ".tf v(out) v1").unwrap());
        let differential = transfer(&run(body, ".tf v(IN,OUT) v1").unwrap());
        close(differential[0], positive[0] - negative[0], 1e-12);
        close(differential[1], positive[1], 1e-12);
        let plot = run(body, ".tf v(IN,OUT) v1").unwrap();
        assert_eq!(plot.variables()[2].name, "v(output_impedance_at_V(in,out))");
    }
}

#[test]
fn current_sense_sign_and_input_impedance_remain_analysis_specific() {
    let expected = -1. / 3e3;
    for body in [DIVIDER, REORDERED] {
        let direct = transfer(&run(body, ".tf i(v1) v1").unwrap());
        close(direct[0], expected, 1e-12);
        close(direct[1], 3e3, 1e-12);
        assert_eq!(direct[1], direct[2], "C same-source resistance shortcut");
        let current = run(body, ".sens I(V1) r1 r2").unwrap();
        // i(v1) = -V1/(R1+R2), and sensitivity to either resistance is positive.
        for name in ["v(r1)", "v(r2)"] {
            close(result(&current, name).re, 10. / (3e3_f64.powi(2)), 2e-5);
        }
    }
}

#[test]
fn forbidden_forms_keep_card_specific_error_context_and_order() {
    for (card, fragment) in [
        (".tf i(r1) v1", "no findable branch current"),
        (".sens i(r1)", "no findable branch current"),
        (
            ".tf i(unknown) v1",
            "output source unknown is not in the circuit",
        ),
        (
            ".sens i(unknown)",
            "output source unknown is not in the circuit",
        ),
        (
            ".tf v(missing) v1",
            "output node missing is not in the circuit",
        ),
        (
            ".sens v(missing)",
            "output node missing is not in the circuit",
        ),
        (".tf v(out) r1", "not of proper type"),
        (".tf v(out) v1 extra", "unexpected argument 'extra'"),
        (
            ".sens v(out) ac lin 1 1 10 extra",
            "unexpected argument 'extra'",
        ),
        (".tf v out v1", "expected '('"),
        (".sens v out", "expected '('"),
    ] {
        let error = run(DIVIDER, card).unwrap_err().to_string();
        assert!(error.contains(fragment), "{card}: {error}");
        let has_context = if card.starts_with(".tf") {
            error.contains(".tf") || error.contains("TFanal in tfanal.c")
        } else {
            error.contains(".sens") || error.contains("sens_sens in cktsens.c")
        };
        assert!(has_context, "{card}: {error}");
    }
}
