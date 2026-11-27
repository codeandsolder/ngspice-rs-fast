//! Independent Cartesian AC interpolation and DC port identities for XSPICE xfer.
use std::path::Path;

use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot, RunConfig};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, ast::Netlist, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, Complex, SpiceResult};

fn parse(body: &str) -> SpiceResult<Netlist> {
    Parser::new().parse_deck(&parse_deck_text(
        Path::new("xfer.cir"),
        &format!("xfer test\n{body}\n.end\n"),
    ))
}

fn run(circuit: &mut Circuit, kind: AnalysisKind, args: &[&str]) -> SpiceResult<Plot> {
    AnalysisInput::with_arguments(kind, args.iter().copied())
        .resolve()?
        .run(circuit, &AnalysisContext::default())
}

fn close(actual: Complex, expected: Complex) {
    assert!(
        (actual - expected).magnitude() <= 2e-12 * expected.magnitude().max(1.),
        "{actual:?} != {expected:?}"
    );
}

const RI: &str = "r_i=true table=[100 2 0 300 0 2]";

fn expected(f: f64) -> Complex {
    let t = ((f - 100.) / 200.).clamp(0., 1.);
    Complex::new(2. * (1. - t), 2. * t)
}

#[test]
fn port_matrix_and_encoding_cross_product() -> SpiceResult<()> {
    let encodings = [
        RI,
        "db=false rad=false table=[100 2 0 300 2 90]",
        "db=false rad=true table=[100 2 0 300 2 1.5707963267948966]",
        "table=[100 6.020599913279624 0 300 6.020599913279624 90]",
        "rad=true table=[100 6.020599913279624 0 300 6.020599913279624 1.5707963267948966]",
        "r_i=true span=5 offset=2 table=[100 99 2 0 77 300 99 0 2 77]",
    ];
    for encoding in encodings {
        for (drive, port, output, sign) in [
            ("v1 in 0 dc 1 ac 1", "in", "out", 1.),
            (
                "v1 in ref dc 1 ac 1\nvref ref 0 5",
                "%vd(in ref)",
                "%vd(out oref)",
                1.,
            ),
            (
                "v1 in ref dc 1 ac 1\nvref ref 0 5",
                "%vd(in ref)",
                "%id(out oref)",
                -1.,
            ),
            ("i1 0 in dc 1 ac 1", "%i(in)", "out", 1.),
            (
                "i1 ref in dc 1 ac 1\nvref ref 0 5",
                "%id(in ref)",
                "%id(out oref)",
                -1.,
            ),
        ] {
            let negative = if output.contains("oref") { "oref" } else { "0" };
            let deck = parse(&format!(
                "{drive}\na1 {port} {output} mdl\n.model mdl xfer({encoding})\nvoutref oref 0 7\nr1 out {negative} 1"
            ))?;
            let text = ngspice_rs::netlist::write_netlist(&deck)?;
            let restored =
                Parser::new().parse_deck(&parse_deck_text(Path::new("restored.cir"), &text))?;
            assert!(
                ngspice_rs::netlist::semantic_diff(&deck, &restored).is_none(),
                "{text}"
            );
            let mut c = Circuit::from_netlist(&restored)?;
            let op = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
            let offset = if negative == "oref" { 7. } else { 0. };
            close(
                op.value("v(out)", 0).unwrap(),
                Complex::real(2. * sign + offset),
            );
            let ac = run(&mut c, AnalysisKind::Ac, &["lin", "7", "50", "350"])?;
            for i in 0..ac.point_count() {
                let f = ac.value("frequency", i).unwrap().re;
                close(
                    ac.value("v(out)", i).unwrap(),
                    expected(f) * Complex::real(sign),
                );
            }
        }
    }
    Ok(())
}

#[test]
fn one_row_and_zero_real_dc_gain_are_valid() -> SpiceResult<()> {
    let mut c = Circuit::from_netlist(&parse(
        "v1 in 0 dc 4 ac 1\na1 in out mdl\n.model mdl xfer(r_i=true table=[0 0 -3])\nr1 out 0 1",
    )?)?;
    close(
        run(&mut c, AnalysisKind::OperatingPoint, &[])?
            .value("v(out)", 0)
            .unwrap(),
        Complex::ZERO,
    );
    let ac = run(&mut c, AnalysisKind::Ac, &["lin", "3", "1", "100"])?;
    for v in ac.column("v(out)").unwrap() {
        close(v, Complex::imaginary(-3.));
    }
    Ok(())
}

#[test]
fn invalid_tables_and_unsupported_analyses_are_rejected() -> SpiceResult<()> {
    for model in [
        "r_i=true table=[0 1 0 0 2 0]",
        "r_i=true table=[-1 1 0]",
        "r_i=true table=[2 1 0 1 2 0]",
        "table=[1 0]",
        "table=[1 0 0 2]",
        "span=2 table=[1 2 3 4]",
        "offset=2 table=[1 2 3]",
        "span=4 offset=0 table=[1 2 3 4]",
        "span=3.5 table=[1 2 3]",
        "db=2 table=[1 2 3]",
        "db=false table=[1 -2 0]",
        "table=[1 100000 0]",
        "r_i=true",
    ] {
        let result = parse(&format!(
            "v1 in 0 1\na1 in out mdl\nr1 out 0 1\n.model mdl xfer({model})"
        ))
        .and_then(|deck| Circuit::from_netlist(&deck));
        assert!(result.is_err(), "accepted {model}");
    }
    let mut c = Circuit::from_netlist(&parse(&format!(
        "v1 in 0 dc 1 ac 1\na1 in out mdl\nr1 out 0 1\n.model mdl xfer({RI})"
    ))?)?;
    let error = run(&mut c, AnalysisKind::Transient, &["1u", "10u"]).unwrap_err();
    assert!(error.to_string().contains("xfer"), "{error}");
    assert!(
        run(
            &mut c,
            AnalysisKind::PoleZero,
            &["in", "0", "out", "0", "vol", "pz"]
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn scoped_coefficients_and_candidate_isolation() -> SpiceResult<()> {
    use ngspice_rs::analysis::optimization::parameters::CircuitParameterPlan;
    use ngspice_rs::netlist::eval::EvalBudget;
    let deck = parse(
        ".param g=2 f=100\n.subckt filter in out params: gainv=1\na1 in out mdl\n.model mdl xfer(r_i=true table=[{f} {gainv*g} 0 {3*f} 0 {gainv*g}])\n.ends filter\nv1 in 0 dc 1 ac 1\nx1 in a filter gainv=1\nx2 in b filter gainv=3\nr1 a 0 1\nr2 b 0 1",
    )?;
    let plan = CircuitParameterPlan::new(deck, &["g", "f"])?;
    let mut session = plan.session()?;
    for (g, f) in [(2., 100.), (4., 50.), (2., 100.)] {
        session.update(&[g, f], &mut EvalBudget::default())?;
        let ctx = session.analysis_context();
        let ac = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "1", "150", "150"])
            .resolve()?
            .run(session.circuit_mut(), &ctx)?;
        let t = ((150. - f) / (2. * f)).clamp(0., 1.);
        let e = Complex::new(g * (1. - t), g * t);
        close(ac.value("v(a)", 0).unwrap(), e);
        close(ac.value("v(b)", 0).unwrap(), e * Complex::real(3.));
    }
    assert!(
        session
            .update(&[2., -1.], &mut EvalBudget::default())
            .is_err()
    );
    let ctx = session.analysis_context();
    let ac = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "1", "150", "150"])
        .resolve()?
        .run(session.circuit_mut(), &ctx)?;
    close(ac.value("v(a)", 0).unwrap(), expected(150.));
    Ok(())
}

const PREPARED_SOURCE: &str = "Frequency transfer reference\n.param f0=100 f1=300 gain=2\nV1 in 0 dc 1 ac 1\nV2 ref 0 dc 0.25 ac 0.25\nEout out 0 FREQ {V(in,ref)*2} MAG RAD ({f0},{gain},0) ({f1},{gain},1.5707963267948966)\nRload out 0 1k\nGout gout 0 FREQ {V(in,ref)*2} R_I (f0,gain,0) (f1,0,gain)\nRg gout 0 1\nFprobe probe 0 Eout 1\nRprobe probe 0 1k\n.ac lin 7 50 350\n.end\n";

fn normalized() -> String {
    let normalized = model_normalize_ngspice::normalize_mapped(
        PREPARED_SOURCE.as_bytes(),
        model_normalize_ngspice::linked_source::RootKind::Deck,
    )
    .unwrap();
    String::from_utf8(normalized.model.bytes).unwrap()
}

#[test]
fn prepared_freq_preserves_original_e_g_and_current_control() -> SpiceResult<()> {
    let source = normalized();
    let deck = Parser::new().parse_deck(&parse_deck_text(Path::new("prepared.cir"), &source))?;
    let cfg = RunConfig::from_netlist(&deck)?;
    let mut c = cfg.circuit(&deck)?;
    let ac = cfg
        .plan_for(&deck.analyses[0])?
        .run(&mut c, &cfg.context())?;
    for i in 0..ac.point_count() {
        let f = ac.value("frequency", i).unwrap().re;
        let e = expected(f) * Complex::real(1.5);
        close(ac.value("v(out)", i).unwrap(), e);
        close(ac.value("v(gout)", i).unwrap(), -e);
        close(ac.value("v(probe)", i).unwrap(), e);
        close(ac.value("i(eout)", i).unwrap(), e * Complex::real(-0.001));
    }
    Ok(())
}

#[test]
#[ignore = "requires NGSPICE_BIN with XSPICE xfer"]
fn live_ngspice_47_reference() -> SpiceResult<()> {
    use ngspice_conformance::{
        ngspice::Scratch,
        process::{ProcessLimits, run as process},
    };
    use ngspice_rs::analysis::RawFile;
    use std::fs;
    let dir = Scratch::new("frequency-reference").unwrap();
    let source = normalized();
    let deck = Parser::new().parse_deck(&parse_deck_text(Path::new("prepared.cir"), &source))?;
    let cfg = RunConfig::from_netlist(&deck)?;
    let mut c = cfg.circuit(&deck)?;
    let ac = cfg
        .plan_for(&deck.analyses[0])?
        .run(&mut c, &cfg.context())?;
    let native=source.replace(".end\n",".control\nset filetype=ascii\nac lin 7 50 350\nwrite ac.raw v(out) v(gout) v(probe) i(eout)\nquit\n.endc\n.end\n");
    fs::write(dir.path().join("reference.cir"), native).unwrap();
    let output = process(
        std::env::var_os("NGSPICE_BIN").unwrap(),
        dir.path(),
        &["-b", "-n", "reference.cir"],
        ProcessLimits::ORACLE,
    )
    .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let raw = RawFile::parse(&fs::read_to_string(dir.path().join("ac.raw")).unwrap()).unwrap();
    let reference = &raw.plots[0].plot;
    let mut worst = 0_f64;
    assert_eq!(reference.point_count(), ac.point_count());
    for i in 0..ac.point_count() {
        for name in ["v(out)", "v(gout)", "v(probe)", "i(eout)"] {
            let expected = reference.value(name, i).unwrap();
            let actual = ac.value(name, i).unwrap();
            worst = worst.max((actual - expected).magnitude());
            close(actual, expected);
        }
    }
    println!("live ngspice 47: 28 complex samples; max absolute error {worst:e}");
    Ok(())
}

#[test]
fn dc_first_ordinate_and_transfer_branch_observations() -> SpiceResult<()> {
    for (model, gain) in [
        ("r_i=true table=[100 -2 9 300 3 4]", -2.),
        ("db=false table=[100 4 60 300 3 0]", 2.),
        ("r_i=true table=[100 0 -3 300 4 0]", 0.),
        ("r_i=true table=[100 2 0]", 2.),
    ] {
        let mut c = Circuit::from_netlist(&parse(&format!(
            "v1 in 0 1\na1 in out mdl\n.model mdl xfer({model})\nr1 out 0 1\nbprobe probe 0 v=i(a1)"
        ))?)?;
        let dc = run(&mut c, AnalysisKind::DcSweep, &["v1", "-2", "2", "1"])?;
        assert_eq!(dc.point_count(), 5);
        for (index, input) in [-2., -1., 0., 1., 2.].into_iter().enumerate() {
            close(
                dc.value("v(out)", index).unwrap(),
                Complex::real(input * gain),
            );
            close(
                dc.value("i(a1)", index).unwrap(),
                Complex::real(-input * gain),
            );
            close(
                dc.value("v(probe)", index).unwrap(),
                Complex::real(-input * gain),
            );
        }
    }
    let mut c = Circuit::from_netlist(&parse(&format!(
        "i1 0 in dc 1 ac 1\na1 %i(in) %i(out) mdl\n.model mdl xfer({RI})\nr1 out 0 1\nbprobe probe 0 v=i(a1)"
    ))?)?;
    let ac = run(&mut c, AnalysisKind::Ac, &["lin", "7", "50", "350"])?;
    for i in 0..ac.point_count() {
        let f = ac.value("frequency", i).unwrap().re;
        close(ac.value("i(a1:input)", i).unwrap(), Complex::real(1.));
        close(ac.value("i(a1)", i).unwrap(), expected(f));
        close(ac.value("v(probe)", i).unwrap(), expected(f));
        close(ac.value("v(out)", i).unwrap(), -expected(f));
    }
    Ok(())
}
