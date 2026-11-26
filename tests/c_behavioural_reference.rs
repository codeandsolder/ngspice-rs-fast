//! Opt-in live C validation of behavioural-source values and derivatives
//! (#79). `NGSPICE_BIN` must name the external reference binary; run with
//! `-- --ignored`.
//!
//! One deck holds a B source per function, all driven by `v(in)` (plus a
//! second bias `v(b)`). The operating point gives every function's value and
//! a one-point AC analysis with `ac 1` on `vin` gives its derivative with
//! respect to `v(in)` (`ASRCacLoad` stamps the same partials Newton uses), so
//! both C's function values and its derivative rules — including the
//! `PTdifferentiate` quirks that remain valid — are compared at several
//! bias points with `|Rust - C| <= 1e-12 |C| + 1e-15`, tight enough to
//! resolve `inp_modify_exp()`'s 11-digit literal rounding (the last case).
use ngspice_conformance::ngspice::Scratch;
use ngspice_conformance::process::{ProcessLimits, run};
use ngspice_rs::analysis::{Plot, RawFile, RunConfig};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::source::parse_deck_text;
use std::fs;
use std::path::Path;

/// The functions under test, as expressions of `v(in)` and `v(b)`.
const EXPRESSIONS: &[&str] = &[
    "abs(v(in) - 0.1)",
    "acos(0.5*v(in))",
    "acosh(2 + v(in)^2)",
    "asin(0.5*v(in))",
    "asinh(3*v(in))",
    "atan(2*v(in) + v(b))",
    "atanh(0.5*v(in))",
    "cos(4*v(in))",
    "cosh(v(in))",
    "exp(2*v(in))",
    "ln(2 + v(in))",
    "log(2 + v(in))",
    "log10(2 + v(in))",
    "sgn(v(in) - 0.1)",
    "sin(3*v(in))",
    "sinh(v(in))",
    "sqrt(2 + v(in))",
    "tan(0.5*v(in))",
    "tanh(2*v(in))",
    "u(v(in) - 0.1)",
    "uramp(v(in) - 0.1)",
    "ceil(3*v(in))",
    "floor(3*v(in))",
    "nint(3*v(in))",
    "-v(in)*v(b)",
    "u2(0.4*v(in) + 0.3)",
    "pwl(v(in), -1, 2, 0, 0, 0.5, 1, 2, 3)",
    "pwl(v(in), 2, 3, 0.5, 1, 0, 0, -1, 2)",
    "eq0(v(in)) + ne0(v(in)) + gt0(v(in)) + lt0(v(in)) + ge0(v(in)) + le0(v(in))",
    "pow(v(in) + 2, v(b) + 1)",
    "pow(v(b) + 2, v(in))",
    "pow(v(in), 3)",
    "pwr(v(in), 2.5)",
    "pwr(v(in) + 2, v(b))",
    "min(v(in), v(b)) + max(2*v(in), v(b))",
    "ternary_fcn(v(in) > 0, v(in)^2, v(in)^3)",
    "v(in)/(2 + v(b))",
    "(1 + v(b))/(v(in) + 3)",
    "(v(in) + 3)^(v(b) + 1)",
    "v(in)**3",
    "v(in)^2.5",
    "2^v(in)",
    "v(in) > v(b) ? exp(v(in)) : cos(v(in))",
    "(v(in) < 0.5 && v(b) != 0) + (v(in) >= 0 || v(b) <= 0)",
    "temper*v(in) + pi + e",
    "hertz*v(in)^2 + hertz",
    "1.23456789012345*v(in)",
];

/// The deck body without an analysis: `vin` at `bias` with `ac 1`.
fn cards(bias: f64) -> String {
    let mut text = format!(
        "behavioural reference\nvin in 0 dc {bias} ac 1\nrin in 0 1k\nvb b 0 dc 0.35\nrb b 0 1k\n"
    );
    for (index, expression) in EXPRESSIONS.iter().enumerate() {
        text.push_str(&format!(
            "b{index} o{index} 0 v={expression}\nr{index} o{index} 0 1k\n"
        ));
    }
    text
}

/// Runs `cards` in C with `analysis` and returns the written plot.
fn run_c(tag: &str, cards: &str, analysis: &str) -> Plot {
    let binary = std::env::var_os("NGSPICE_BIN").expect("set NGSPICE_BIN");
    let scratch =
        Scratch::new("c-behavioural-reference").unwrap_or_else(|error| panic!("{tag}: {error}"));
    let dir = scratch.path();
    fs::write(
        dir.join("c.cir"),
        format!(
            "{cards}.control\nset filetype=ascii\n{analysis}\nwrite result.raw\nquit\n.endc\n.end\n"
        ),
    )
    .unwrap();
    let result = run(binary, dir, &["-b", "c.cir"], ProcessLimits::ORACLE).unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let raw = RawFile::parse(&fs::read_to_string(dir.join("result.raw")).unwrap()).unwrap();
    raw.plots[0].plot.clone()
}

fn run_rust(cards: &str, analysis: &str) -> Plot {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("c.cir"),
            &format!("{cards}.{analysis}\n.end\n"),
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    request.run(&mut circuit, &config.context()).unwrap()
}

#[test]
#[ignore = "requires NGSPICE_BIN; run with -- --ignored"]
fn every_function_value_and_valid_derivative_matches_live_c() {
    let mut failures = Vec::new();
    for (case, bias) in [-0.45, 0.3, 0.8, 1.7].into_iter().enumerate() {
        let deck = cards(bias);
        for (analysis, part) in [("op", "value"), ("ac lin 1 1k 1k", "derivative")] {
            let c = run_c(&format!("{case}-{part}"), &deck, analysis);
            let rust = run_rust(&deck, analysis);
            for (index, expression) in EXPRESSIONS.iter().enumerate() {
                let name = format!("v(o{index})");
                let ours = rust.value(&name, 0).unwrap();
                let theirs = c.value(&name, 0).unwrap();
                let mut theirs = theirs;
                // These are documented C47 Jacobian defects, not value
                // discrepancies. Compare against independently derived
                // slopes on the negative-base branch.
                if part == "derivative" && bias < 0. {
                    let slope = match *expression {
                        "acosh(2 + v(in)^2)" => {
                            Some(2. * bias / ((2. + bias * bias).powi(2) - 1.).sqrt())
                        }
                        "pow(v(in), 3)"
                        | "v(in)**3"
                        | "ternary_fcn(v(in) > 0, v(in)^2, v(in)^3)" => Some(3. * bias * bias),
                        "pwr(v(in), 2.5)" => Some(2.5 * (-bias).powf(1.5)),
                        "v(in)^2.5" => Some(0.),
                        "hertz*v(in)^2 + hertz" => Some(2000. * bias),
                        _ => None,
                    };
                    if let Some(slope) = slope {
                        theirs.re = slope;
                    }
                }
                // C's descending PWL slope search ignores the actual
                // ordering. The value selects the correct segment.
                if part == "derivative" && *expression == "pwl(v(in), 2, 3, 0.5, 1, 0, 0, -1, 2)" {
                    theirs.re = if bias < 0. {
                        -2.
                    } else if bias < 0.5 {
                        2.
                    } else {
                        4. / 3.
                    };
                }
                for (a, b) in [(ours.re, theirs.re), (ours.im, theirs.im)] {
                    if (a - b).abs() > 1e-12 * b.abs() + 1e-15 {
                        failures.push(format!(
                            "{expression} at v(in) = {bias}: {part} Rust {a:e}, C {b:e}"
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

const PHASE_MACROS: &str = "Native phase-aware macros\n\
    .func choose(c,a,b) {ternary_fcn(c>0.5,a,b)}\n\
    .func shape(x) {choose(x,2*x+3,-x+1)}\n\
    .func compare(a,b,c) {a==b<c}\n\
    .func clip(x) {ternary_fcn(x<-1,-1,ternary_fcn(x>1,1,x))}\n\
    .param scalar={compare(2,3,1)}\n\
    b1 y 0 v=shape(v(x))\n\
    b2 z 0 v=compare(2,3,1)\n\
    b3 q 0 v=clip(v(x))\n\
    v2 ref 0 {scalar}\n\
    r1 y 0 1k\nr2 z 0 1k\nr3 q 0 1k\n";

#[test]
#[ignore = "requires NGSPICE_BIN; run with -- --ignored"]
fn phase_correct_nested_macros_match_native_op_dc_ac_and_transient() {
    let deck = format!("{PHASE_MACROS}v1 x 0 dc 0.75 ac 1\n");
    for analysis in ["op", "dc v1 -1.375 1.375 0.25", "ac lin 1 1k 1k"] {
        let c = run_c(analysis, &deck, analysis);
        let rust = run_rust(&deck, analysis);
        assert_eq!(c.point_count(), rust.point_count(), "{analysis}");
        for point in 0..c.point_count() {
            for name in ["v(y)", "v(z)", "v(q)", "v(ref)"] {
                let ours = rust.value(name, point).unwrap();
                let theirs = c.value(name, point).unwrap();
                assert!(
                    (ours - theirs).magnitude() <= 1e-10,
                    "{analysis}, point {point}, {name}: Rust {ours:?}, C {theirs:?}"
                );
            }
        }
    }

    // Transient time grids need not coincide. The same independent affine
    // input is used for both, on a region with no branch discontinuity.
    let deck = format!("{PHASE_MACROS}v1 x 0 pwl(0 0.6 1m 0.9)\n");
    let analysis = "tran 10u 1m";
    for plot in [run_c(analysis, &deck, analysis), run_rust(&deck, analysis)] {
        assert!(plot.point_count() > 50);
        for point in 0..plot.point_count() {
            let time = plot.value("time", point).unwrap().re;
            let x = 0.6 + 300.0 * time;
            for (name, expected) in [
                ("v(y)", 2.0 * x + 3.0),
                ("v(z)", 0.0),
                ("v(q)", x),
                ("v(ref)", 1.0),
            ] {
                let actual = plot.value(name, point).unwrap().re;
                assert!(
                    (actual - expected).abs() <= 1e-9,
                    "{name} {time}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires NGSPICE_BIN; run with -- --ignored"]
fn hidden_runtime_passive_macros_select_behavioural_precedence() {
    let deck = "Hidden runtime passive macro precedence\n\
        .func law(x) {v(ctrl)+x}\n\
        v1 out 0 1\nv2 ctrl 0 1\n\
        r1 out 0 {1+(2==3<1)+law(0)}\n";
    let c = run_c("hidden-runtime-passive", deck, "op");
    let rust = run_rust(deck, "op");
    let expected = c.value("i(v1)", 0).unwrap().re;
    let actual = rust.value("i(v1)", 0).unwrap().re;
    assert!((expected + 0.5).abs() <= 1e-12);
    assert!(
        (actual - expected).abs() <= 1e-12,
        "Rust {actual}, C {expected}"
    );
}
