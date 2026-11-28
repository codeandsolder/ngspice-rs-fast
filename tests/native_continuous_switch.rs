//! Native continuous XSPICE switch qualification, including a live C DC
//! oracle and independent signed AC / transient differential checks.
use ngspice_rs::analysis::{AnalysisInput, Plot, RunConfig};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, semantic_eq, source::parse_deck_text, write_netlist};
use ngspice_rs::primitives::{AnalysisKind, Real, SpiceResult};
use std::path::Path;

fn parse(cards: &str) -> SpiceResult<ngspice_rs::netlist::ast::Netlist> {
    let text = format!("continuous switch\n{cards}\n.end\n");
    Parser::new().parse_deck(&parse_deck_text(Path::new("continuous.cir"), &text))
}

fn run(cards: &str, kind: AnalysisKind, arguments: &[&str]) -> SpiceResult<Plot> {
    let netlist = parse(cards)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    config
        .plan(AnalysisInput::with_arguments(
            kind,
            arguments.iter().copied(),
        ))?
        .run(&mut circuit, &config.context())
}

fn value(plot: &Plot, name: &str, index: usize) -> Real {
    plot.value(name, index).expect(name).re
}

fn close(label: &str, actual: Real, expected: Real, relative: Real, absolute: Real) {
    assert!(
        actual.is_finite()
            && (actual - expected).abs() <= expected.abs().mul_add(relative, absolute),
        "{label}: {actual:.17e} != {expected:.17e}"
    );
}

const MATRIX: &str = "\
Vctrl c 0 0
Vb1 p1 0 1
Vb2 p2 0 -2
Vb3 p3 0 1
Vb4 p4 0 -2
Vb5 p5 0 1
Vb6 p6 0 -2
Vb7 p7 0 1
Vb8 p8 0 -2
A1 %gd(c 0) %gd(p1 0) m1
A2 %g(c) %gd(p2 0) m2
A3 c 0 p3 0 m3
A4 %gd(c 0) %gd(p4 0) m4
A5 c p5 0 m5
A6 %vd(c 0) %gd(p6 0) m6
A7 %v(c) %gd(p7 0) m7
A8 %v(c) %gd(p8 0) m8
.model m1 pswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=true r_cntl_in=100)
.model m2 pswitch(cntl_off=4 cntl_on=2 r_on=10 r_off=1k log=true r_cntl_in=200)
.model m3 pswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=false r_cntl_in=400)
.model m4 pswitch(cntl_off=4 cntl_on=2 r_on=10 r_off=1k log=false r_cntl_in=800)
.model m5 aswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=true limit=true)
.model m6 aswitch(cntl_off=4 cntl_on=2 r_on=10 r_off=1k log=true limit=true)
.model m7 aswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=false limit=true)
.model m8 aswitch(cntl_off=4 cntl_on=2 r_on=10 r_off=1k log=false limit=true)
";

#[test]
fn continuous_dc_sweep_all_laws_threshold_directions_and_terminal_loading() -> SpiceResult<()> {
    let plot = run(
        MATRIX,
        AnalysisKind::DcSweep,
        &["vctrl", "-0.5", "4.5", ".5"],
    )?;
    assert_eq!(plot.point_count(), 11);
    for index in 0..plot.point_count() {
        let c = value(&plot, "v(c)", index);
        let t = ((c - 2.) / 2.).clamp(0., 1.);
        for (port, fraction, voltage) in [
            (1, t, 1.),
            (2, 1. - t, -2.),
            (3, t, 1.),
            (4, 1. - t, -2.),
            (5, t, 1.),
            (6, 1. - t, -2.),
            (7, t, 1.),
            (8, 1. - t, -2.),
        ] {
            let g = match port {
                1 | 2 => 100_f64.powf(fraction.powi(2) * (3. - 2. * fraction)) / 1000.,
                5 | 6 => 100_f64.powf(fraction) / 1000.,
                _ => (1000. * (1. - fraction) + 10. * fraction).recip(),
            };
            let name = format!("i(vb{port})");
            close(
                &name,
                value(&plot, &name, index),
                -voltage * g,
                1e-10,
                1e-13,
            );
        }
        close(
            "control KCL",
            value(&plot, "i(vctrl)", index),
            -c * (1. / 100. + 1. / 200. + 1. / 400. + 1. / 800.),
            1e-10,
            1e-13,
        );
    }
    Ok(())
}

fn differential_deck(model: &str, c: Real, output: Real) -> String {
    let control = if model.starts_with("pswitch") {
        "%gd(c z)"
    } else {
        "%vd(c z)"
    };
    format!(
        "Vref z 0 5\nVc c z {c:.16e} ac .3\nVb p z {output:.16e} ac .7\nA1 {control} %gd(p z) m\n.model m {model}\n"
    )
}

#[test]
fn ac_is_the_signed_derivative_of_dc_current_including_clamped_regions() -> SpiceResult<()> {
    // Jointly perturb the differential control and output voltage.
    // Independent finite differences of converged DC, not an implementation
    // of the same analytic derivative formula.
    let models = [
        "pswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=true r_cntl_in=100)",
        "pswitch(cntl_off=4 cntl_on=2 r_on=10 r_off=1k log=true r_cntl_in=100)",
        "pswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=false r_cntl_in=100)",
        "pswitch(cntl_off=4 cntl_on=2 r_on=10 r_off=1k log=false r_cntl_in=100)",
        "aswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=true limit=true)",
        "aswitch(cntl_off=4 cntl_on=2 r_on=10 r_off=1k log=true limit=true)",
        "aswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=false limit=true)",
        "aswitch(cntl_off=4 cntl_on=2 r_on=10 r_off=1k log=false limit=true)",
        "aswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=true limit=false)",
        "aswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=false limit=false)",
    ];
    let h = 1e-5;
    for model in models {
        for c in [1.5, 2.25, 3., 3.75, 4.5] {
            let dc = |direction: Real| {
                run(
                    &differential_deck(model, c + direction * h * 0.3, -2. + direction * h * 0.7),
                    AnalysisKind::OperatingPoint,
                    &[],
                )
            };
            let minus = dc(-1.)?;
            let plus = dc(1.)?;
            let ac = run(
                &differential_deck(model, c, -2.),
                AnalysisKind::Ac,
                &["lin", "1", "1k", "1k"],
            )?;
            for name in ["i(vb)", "i(vc)", "i(vref)"] {
                let expected = (value(&plus, name, 0) - value(&minus, name, 0)) / (2. * h);
                let actual = ac.value(name, 0).expect(name);
                close(model, actual.re, expected, 3e-7, 1e-9);
                close(name, actual.im, 0., 0., 1e-12);
            }
        }
    }
    Ok(())
}

const CURRENT_PORTS: &str = "\
Ictrl 0 x 0.4 ac 1
Vsen x y 0
Rsen y 0 1
Vcommon cm 0 5
Idiff cm ip 0.4 ac 1
Idir 0 si 0.4 ac 1
Vb1 p1 0 1
Vb2 p2 0 -2
Vb3 p3 0 1
A1 %vnam(Vsen) %gd(p1 0) model
A2 %id(ip cm) %gd(p2 0) model
A3 %i(si) %gd(p3 0) model
.model model aswitch(cntl_off=0 cntl_on=1 r_on=10 r_off=1k log=true limit=true)
";

#[test]
fn current_control_ports_and_scoped_named_branch_identity() -> SpiceResult<()> {
    let netlist = parse(CURRENT_PORTS)?;
    let written = write_netlist(&netlist)?;
    let reparsed =
        Parser::new().parse_deck(&parse_deck_text(Path::new("roundtrip.cir"), &written))?;
    assert!(semantic_eq(&netlist, &reparsed));
    let circuit = Circuit::from_netlist(&netlist)?;
    assert!(
        circuit.nodes().get("vsen").is_none(),
        "branch reference became a node"
    );
    let op = run(CURRENT_PORTS, AnalysisKind::OperatingPoint, &[])?;
    let ac = run(CURRENT_PORTS, AnalysisKind::Ac, &["lin", "1", "1k", "1k"])?;
    let g = 100_f64.powf(0.4) / 1000.;
    let derivative = g * 100_f64.ln();
    for (name, voltage) in [("i(vb1)", 1.), ("i(vb2)", -2.), ("i(vb3)", 1.)] {
        close(name, value(&op, name, 0), -voltage * g, 1e-10, 1e-13);
        close(
            name,
            value(&ac, name, 0),
            -voltage * derivative,
            1e-10,
            1e-13,
        );
    }
    close("sense current", value(&op, "i(vsen)", 0), 0.4, 1e-12, 1e-14);
    close(
        "differential current input",
        value(&op, "v(ip)", 0),
        5.,
        1e-12,
        1e-14,
    );
    close(
        "single current input",
        value(&op, "v(si)", 0),
        0.,
        0.,
        1e-14,
    );

    let scoped = "\
Vb1 p1 0 1
Vb2 p2 0 1
X1 p1 cell drive=.25
X2 p2 cell drive=.75
.subckt cell p params: drive=.5
Ictrl 0 x {drive} ac 1
Vsen x y 0
Rsen y 0 1
A1 %vnam(vsen) %gd(p 0) m
.model m aswitch(r_on=10 r_off=1k log=true limit=true)
.ends cell
";
    let op = run(scoped, AnalysisKind::OperatingPoint, &[])?;
    let ac = run(scoped, AnalysisKind::Ac, &["lin", "1", "1k", "1k"])?;
    for (name, control) in [("i(vb1)", 0.25), ("i(vb2)", 0.75)] {
        let expected = 100_f64.powf(control) / 1000.;
        close(name, value(&op, name, 0), -expected, 1e-10, 1e-13);
        close(
            name,
            value(&ac, name, 0),
            -expected * 100_f64.ln(),
            1e-10,
            1e-13,
        );
    }
    Ok(())
}

#[test]
fn transient_ramp_drives_the_physical_loaded_resistance() -> SpiceResult<()> {
    let cards = "\
Vc c 0 pwl(0 1.5 1u 1.5 3u 4.5 4u 4.5 6u 1.5)
Vb source 0 1
Rseries source out 100
A1 %gd(c 0) %gd(out 0) m
.model m pswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=true r_cntl_in=200)
.options reltol=1e-8 abstol=1e-14 vntol=1e-10
";
    let plot = run(cards, AnalysisKind::Transient, &["10n", "6u", "0", "10n"])?;
    assert!(plot.point_count() > 500);
    for index in 0..plot.point_count() {
        let control = value(&plot, "v(c)", index);
        let t = ((control - 2.) / 2.).clamp(0., 1.);
        let g = 100_f64.powf(t.powi(2) * (3. - 2. * t)) / 1000.;
        let expected = (1. + 100. * g).recip();
        close(
            "loaded resistance",
            value(&plot, "v(out)", index),
            expected,
            2e-6,
            1e-9,
        );
        close(
            "control input",
            value(&plot, "i(vc)", index),
            -control / 200.,
            1e-10,
            1e-13,
        );
        close(
            "output branch KCL",
            value(&plot, "i(vb)", index),
            -expected * g,
            2e-6,
            1e-11,
        );
    }
    Ok(())
}

#[test]
fn bounds_extrapolation_near_equal_thresholds_and_invalid_parameters() -> SpiceResult<()> {
    for (model, control, expected_r) in [
        ("pswitch(cntl_off=1 cntl_on=1 r_on=10 r_off=1k)", 1., 100.),
        ("pswitch(r_on=1u r_off=1e15)", 2., 1e-3),
        ("pswitch(r_on=1u r_off=1e15)", -1., 1e12),
        ("aswitch(r_on=10 r_off=1k limit=false)", -1., 1e5),
        ("aswitch(r_on=10 r_off=1k limit=false)", 2., 0.1),
        ("aswitch(r_on=10 r_off=1k log=false limit=false)", 2., 1e-9),
        (
            "aswitch(r_on=10 r_off=1k log=false limit=false)",
            -1.,
            1990.,
        ),
        ("aswitch(r_on=1u r_off=1k limit=true)", 2., 1e-3),
    ] {
        let op = run(
            &differential_deck(model, control, 1.),
            AnalysisKind::OperatingPoint,
            &[],
        )?;
        close(model, value(&op, "i(vb)", 0), -1. / expected_r, 1e-9, 1e-13);
    }
    for model in [
        "pswitch(r_cntl_in=0)",
        "pswitch(r_off=0)",
        "aswitch(cntl_off=2 cntl_on=2)",
        "aswitch(r_off=-1)",
        "aswitch(r_cntl_in=10)",
        "pswitch(limit=true)",
    ] {
        assert!(
            run(
                &differential_deck(model, 1., 1.),
                AnalysisKind::OperatingPoint,
                &[]
            )
            .is_err(),
            "{model}"
        );
    }
    for (port, model) in [
        ("%vnam(missing)", "aswitch"),
        ("%vnam(vctrl)", "pswitch"),
        ("%id(c c)", "aswitch"),
        ("%v(c)", "pswitch"),
    ] {
        let cards = format!("Vctrl c 0 1\nVb p 0 1\nA1 {port} %gd(p 0) m\n.model m {model}\n");
        assert!(
            run(&cards, AnalysisKind::OperatingPoint, &[]).is_err(),
            "{port} {model}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; compares live C ngspice DC reference"]
fn native_dc_reference() -> SpiceResult<()> {
    use ngspice_conformance::ngspice::Scratch;
    use ngspice_conformance::process::{ProcessLimits, run as run_process};
    use ngspice_rs::analysis::RawFile;
    use std::fs;

    let binary = std::env::var_os("NGSPICE_BIN").expect("set absolute NGSPICE_BIN");
    assert!(Path::new(&binary).is_absolute());
    for (cards, analysis, kind, args, names) in [
        (
            MATRIX,
            "dc vctrl -0.5 4.5 .5",
            AnalysisKind::DcSweep,
            &["vctrl", "-0.5", "4.5", ".5"][..],
            &[
                "v(c)", "i(vb1)", "i(vb2)", "i(vb3)", "i(vb4)", "i(vb5)", "i(vb6)", "i(vb7)",
                "i(vb8)", "i(vctrl)",
            ][..],
        ),
        (
            CURRENT_PORTS,
            "op",
            AnalysisKind::OperatingPoint,
            &[][..],
            &[
                "i(vb1)", "i(vb2)", "i(vb3)", "i(vsen)", "v(ip)", "v(cm)", "v(si)",
            ][..],
        ),
    ] {
        let scratch = Scratch::new("c-continuous-switch").expect("scratch");
        let dir = scratch.path();
        let deck = format!(
            "c\n{cards}.control\nset filetype=ascii\nset numdgt=16\n{analysis}\nwrite result.raw\nquit\n.endc\n.end\n"
        );
        fs::write(dir.join("c.cir"), deck).expect("C deck");
        let output = run_process(&binary, dir, &["-n", "-b", "c.cir"], ProcessLimits::ORACLE)
            .expect("C run");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let raw = fs::read_to_string(dir.join("result.raw")).expect("C result");
        let reference = RawFile::parse(&raw)?.plots[0].plot.clone();
        let actual = run(cards, kind, args)?;
        assert_eq!(actual.point_count(), reference.point_count());
        for index in 0..actual.point_count() {
            for name in names {
                close(
                    name,
                    value(&actual, name, index),
                    value(&reference, name, index),
                    1e-9,
                    1e-12,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn accepted_pspice_producer_voltage_and_current_switches_execute_without_dialect_fallback()
-> SpiceResult<()> {
    use model_normalize_ngspice::{linked_source::RootKind, pspice};
    let text = br"Prepared continuous switches
Vdrive c 0 0
Rdrive c sense 1
Vsen sense 0 0
Vb1 p1 0 1
Vb2 p2 0 -2
S1 p1 0 c 0 mv
W1 p2 0 Vsen mi
.model mv VSWITCH(VON=4 VOFF=2 RON=10 ROFF=1k)
.model mi ISWITCH(ION=2 IOFF=4 RON=10 ROFF=1k)
.options GMIN=1e-5
.end
";
    let prepared = pspice::normalize_mapped(
        text,
        RootKind::Deck,
        &model_preprocess_command::WorkBudget::default(),
    )
    .map_err(|error| ngspice_rs::primitives::SpiceError::circuit(error.to_string()))?;
    assert_eq!(prepared.renamed.len(), 2);
    let native = String::from_utf8(prepared.mapped.model.bytes)
        .map_err(|error| ngspice_rs::primitives::SpiceError::circuit(error.to_string()))?;
    let netlist = Parser::new().parse_deck(&parse_deck_text(Path::new("prepared.cir"), &native))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    let plot = config
        .plan(AnalysisInput::with_arguments(
            AnalysisKind::DcSweep,
            ["vdrive", "1.5", "4.5", ".25"],
        ))?
        .run(&mut circuit, &config.context())?;
    for index in 0..plot.point_count() {
        let c = value(&plot, "v(c)", index);
        let t = ((c - 2.) / 2.).clamp(0., 1.);
        for (name, fraction, voltage) in [("i(vb1)", t, 1.), ("i(vb2)", 1. - t, -2.)] {
            let g = 100_f64.powf(fraction.powi(2) * (3. - 2. * fraction)) / 1000.;
            close(name, value(&plot, name, index), -voltage * g, 1e-9, 1e-12);
        }
        close(
            "unloaded sensing branch",
            value(&plot, "i(vsen)", index),
            c,
            1e-10,
            1e-12,
        );
        close(
            "PSpice control conductance",
            value(&plot, "i(vdrive)", index),
            -c * (1. + 1e-5),
            1e-10,
            1e-12,
        );
    }
    Ok(())
}
