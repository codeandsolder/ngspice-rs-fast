//! Independent polynomial and time-domain checks of the native rational
//! transfer realization. The reference response is evaluated by Horner,
//! not by reproducing the state-space realization.
use std::path::Path;

use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot};
use ngspice_rs::devices::Circuit;
use ngspice_rs::devices::code_model::rational::{
    PortLaw, RationalSpec, RationalTransfer, TransferPorts,
};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, Complex, NodeId, SpiceResult};

fn circuit(source: &str, spec: RationalSpec<'_>, law: PortLaw) -> SpiceResult<Circuit> {
    let deck = Parser::new().parse_deck(&parse_deck_text(
        Path::new("transfer.cir"),
        &format!("transfer test\n{source}\nr1 out 0 1\n.end\n"),
    ))?;
    let mut circuit = Circuit::from_netlist(&deck)?;
    let output = circuit.nodes().get("out").unwrap();
    let positive = circuit.nodes().get("in").unwrap();
    let negative = circuit.nodes().get("ref").unwrap_or(NodeId::GROUND);
    circuit.add_device(Box::new(RationalTransfer::new(
        "transfer",
        TransferPorts {
            output: [output, NodeId::GROUND],
            input: [positive, negative],
            output_law: law,
            input_law: PortLaw::Voltage,
        },
        spec,
    )?))?;
    circuit.finalize()?;
    Ok(circuit)
}

fn run(circuit: &mut Circuit, kind: AnalysisKind, args: &[&str]) -> SpiceResult<Plot> {
    let mut request =
        AnalysisInput::with_arguments(kind, args.iter().copied().filter(|arg| *arg != "uic"));
    request.uic = args.contains(&"uic");
    request.resolve()?.run(circuit, &AnalysisContext::default())
}

fn polynomial(coefficients: &[f64], z: Complex) -> Complex {
    coefficients
        .iter()
        .fold(Complex::ZERO, |value, &coefficient| {
            value * z + Complex::real(coefficient)
        })
}

fn close(actual: Complex, expected: Complex, tolerance: f64) {
    assert!(
        (actual - expected).magnitude() <= tolerance * expected.magnitude().max(1.),
        "actual={actual:?}, expected={expected:?}"
    );
}

fn spec<'a>(numerator: &'a [f64], denominator: &'a [f64], scale: f64) -> RationalSpec<'a> {
    RationalSpec {
        numerator,
        denominator,
        frequency_scale: scale,
        gain: 1.,
        input_offset: 0.,
        initial: None,
    }
}

#[test]
fn arbitrary_order_transfer_matches_complex_polynomial() -> SpiceResult<()> {
    // Stable repeated poles, zero, lead/lag and nonzero feedthrough.
    for (num, den, scale) in [
        (&[1., 0.][..], &[1., 2., 1.][..], 1e3),
        (&[2., -3., 1., 0.5][..], &[2., 6., 6., 2.][..], 1e-4),
        (&[3., 2., 1.][..], &[1., 5., 10., 10., 5., 1.][..], 1e8),
        (&[-4.][..], &[2.][..], 1.),
        (&[0., 0.][..], &[1., 2., 1.][..], 1e3),
    ] {
        for law in [PortLaw::Voltage, PortLaw::Current] {
            let mut c = circuit(
                "v1 in ref dc 1 ac 1\nvref ref 0 7",
                spec(num, den, scale),
                law,
            )?;
            let start = format!("{}", scale * 0.01 / std::f64::consts::TAU);
            let stop = format!("{}", scale * 100. / std::f64::consts::TAU);
            let response = run(&mut c, AnalysisKind::Ac, &["dec", "8", &start, &stop])?;
            let sign = if law == PortLaw::Voltage { 1. } else { -1. };
            for index in 0..response.point_count() {
                let z = Complex::imaginary(
                    std::f64::consts::TAU * response.value("frequency", index).unwrap().re / scale,
                );
                let expected = polynomial(num, z) / polynomial(den, z) * Complex::real(sign);
                close(response.value("v(out)", index).unwrap(), expected, 1e-8);
            }
        }
    }
    Ok(())
}

#[test]
fn second_order_step_uses_real_companion_history() -> SpiceResult<()> {
    let tau = 100e-6_f64;
    let ramp = 1e-6_f64;
    let mut c = circuit(
        "v1 in 0 dc 0 pwl(0 0 1u 1 2m 1)",
        spec(&[1.], &[1., 2., 1.], 1. / tau),
        PortLaw::Voltage,
    )?;
    // Ramp is an integral of steps: integrate 1-(1+t/tau)e^(-t/tau).
    let primitive = |t: f64| {
        if t > 0. {
            t + (t + 2. * tau) * (-t / tau).exp() - 2. * tau
        } else {
            0.
        }
    };
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            &mut c,
            AnalysisKind::Transient,
            &["1u", "1m", "0", "1u", method],
        )?;
        let mut checked = 0;
        for index in 0..plot.point_count() {
            let t = plot.value("time", index).unwrap().re;
            if t > 10e-6 {
                let expected = (primitive(t) - primitive(t - ramp)) / ramp;
                close(
                    plot.value("v(out)", index).unwrap(),
                    Complex::real(expected),
                    2e-4,
                );
                checked += 1;
            }
        }
        assert!(checked > 100);
    }
    Ok(())
}

#[test]
fn origin_poles_initialize_and_integrate_with_nonzero_input() -> SpiceResult<()> {
    let mut c = circuit(
        "v1 in 0 dc 2 ac 1",
        spec(&[1.], &[1., 0., 0.], 1000.),
        PortLaw::Voltage,
    )?;
    let op = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    close(op.value("v(out)", 0).unwrap(), Complex::ZERO, 1e-10);
    let ac = run(&mut c, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
    let z = Complex::imaginary(std::f64::consts::TAU);
    close(
        ac.value("v(out)", 0).unwrap(),
        Complex::real(1.) / (z * z),
        1e-10,
    );
    let tran = run(
        &mut c,
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u", "method=gear"],
    )?;
    for index in 0..tran.point_count() {
        let t = tran.value("time", index).unwrap().re;
        close(
            tran.value("v(out)", index).unwrap(),
            Complex::real((1000. * t).powi(2)),
            2e-5,
        );
    }
    Ok(())
}

#[test]
fn feedthrough_offset_and_explicit_initial_coordinates() -> SpiceResult<()> {
    let mut spec = spec(&[2., 3.], &[1., 4.], 1000.);
    spec.gain = 2.;
    spec.input_offset = 1.;
    let mut c = circuit("v1 in 0 dc 3 ac 1", spec, PortLaw::Voltage)?;
    let op = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    close(op.value("v(out)", 0).unwrap(), Complex::real(6.), 1e-10);
    spec.initial = Some(&[0.5]);
    let mut c = circuit("v1 in 0 dc 3", spec, PortLaw::Voltage)?;
    let op = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    // D=4, C=6-4*4=-10; y=16-2.5.
    close(op.value("v(out)", 0).unwrap(), Complex::real(13.5), 1e-10);
    Ok(())
}

#[test]
fn invalid_transfer_specifications_are_rejected() {
    for bad in [
        spec(&[1.], &[0., 0.], 1.),
        spec(&[1., 1.], &[1.], 1.),
        spec(&[f64::NAN], &[1.], 1.),
        spec(&[1.], &[1., 1.], 0.),
        spec(&[1.], &[], 1.),
        RationalSpec {
            initial: Some(&[1., 2.]),
            ..spec(&[1.], &[1., 1.], 1.)
        },
    ] {
        assert!(
            RationalTransfer::new(
                "bad",
                TransferPorts {
                    output: [NodeId::new(1), NodeId::GROUND],
                    input: [NodeId::new(2), NodeId::GROUND],
                    output_law: PortLaw::Voltage,
                    input_law: PortLaw::Voltage
                },
                bad
            )
            .is_err()
        );
    }
}

fn parsed(body: &str) -> SpiceResult<ngspice_rs::netlist::ast::Netlist> {
    Parser::new().parse_deck(&parse_deck_text(
        Path::new("sxfer.cir"),
        &format!("s_xfer test\n{body}\n.end\n"),
    ))
}

#[test]
fn native_s_xfer_voltage_and_current_ports_roundtrip_and_solve() -> SpiceResult<()> {
    for (source, ports, sign) in [
        ("v1 in 0 dc 1 ac 1", "in out", 1.),
        (
            "v1 in ref dc 1 ac 1\nvref ref 0 7",
            "%vd(in ref) %vd(out 0)",
            1.,
        ),
        ("v1 in 0 dc 1 ac 1", "in %i(out)", -1.),
        ("i1 0 in dc 1 ac 1", "%i(in) out", 1.),
        ("i1 0 in dc 1 ac 1", "%id(in 0) %id(out 0)", -1.),
    ] {
        let netlist = parsed(&format!(
            "{source}\na1 {ports} model\nr1 out 0 1\n.model model s_xfer(num_coeff=[1] den_coeff=[1, 1] denormalized_freq=1000)"
        ))?;
        let text = ngspice_rs::netlist::write_netlist(&netlist)?;
        let restored =
            Parser::new().parse_deck(&parse_deck_text(Path::new("restored.cir"), &text))?;
        assert!(
            ngspice_rs::netlist::semantic_diff(&netlist, &restored).is_none(),
            "{text}"
        );
        let mut c = Circuit::from_netlist(&restored)?;
        let op = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
        close(op.value("v(out)", 0).unwrap(), Complex::ZERO, 1e-12);
        let ac = run(&mut c, AnalysisKind::Ac, &["lin", "1", "100", "100"])?;
        close(
            ac.value("v(out)", 0).unwrap(),
            Complex::real(sign)
                / (Complex::real(1.) + Complex::imaginary(std::f64::consts::TAU / 10.)),
            1e-10,
        );
    }
    Ok(())
}

#[test]
fn native_transfer_lexical_coefficients_and_initial_stage_order() -> SpiceResult<()> {
    let netlist = parsed(
        ".subckt filter in out params: pole=1 gainv=1\n.model mdl s_xfer(num_coeff=[{gainv}] den_coeff=[1 {2*pole} {pole*pole}] int_ic=[0 0] denormalized_freq=1k)\na1 in out mdl\n.ends filter\nv1 in 0 dc 1 ac 1\nx1 in out filter pole=2 gainv=3\nr1 out 0 1k",
    )?;
    let text = ngspice_rs::netlist::write_netlist(&netlist)?;
    let restored = Parser::new().parse_deck(&parse_deck_text(Path::new("restored.cir"), &text))?;
    assert!(ngspice_rs::netlist::semantic_diff(&netlist, &restored).is_none());
    let mut c = Circuit::from_netlist(&restored)?;
    let ac = run(&mut c, AnalysisKind::Ac, &["lin", "1", "100", "100"])?;
    let z = Complex::imaginary(std::f64::consts::TAU / 10.);
    close(
        ac.value("v(out)", 0).unwrap(),
        Complex::real(3.) / (z * z + Complex::real(4.) * z + Complex::real(4.)),
        1e-10,
    );

    // XSPICE int_ic values are physical integrators in descending stage
    // order. A zero gain must not erase nonzero free response.
    let mut c = Circuit::from_netlist(&parsed(
        "v1 in 0 dc 0\na1 in out mdl\nr1 out 0 1k\n.model mdl s_xfer(num_coeff=[1 2] den_coeff=[2 4 2] denormalized_freq=1000 gain=0 int_ic=[3000 4])",
    )?)?;
    let op = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    // y = x'(physical)/omega + 2*x = 3 + 8.
    close(op.value("v(out)", 0).unwrap(), Complex::real(11.), 1e-10);
    Ok(())
}

#[test]
fn malformed_native_transfer_vectors_are_positioned() {
    for model in [
        "num_coeff=[1 2 den_coeff=[1 1]",
        "num_coeff=[1,] den_coeff=[1 1]",
        "num_coeff=[,1] den_coeff=[1 1]",
        "num_coeff=[1[2] den_coeff=[1 1]",
        "num_coeff=[1] den_coeff=[0]",
    ] {
        let result = parsed(&format!(
            "v1 in 0 1\na1 in out mdl\nr1 out 0 1k\n.model mdl s_xfer({model})"
        ))
        .and_then(|deck| Circuit::from_netlist(&deck));
        assert!(result.is_err(), "accepted {model}");
    }
}

#[test]
#[ignore = "requires NGSPICE_BIN and the XSPICE s_xfer model"]
fn native_s_xfer_matches_independent_c_ac_and_transient() -> SpiceResult<()> {
    use ngspice_conformance::ngspice::Scratch;
    use ngspice_conformance::process::{ProcessLimits, run as run_process};
    use ngspice_rs::analysis::RawFile;
    use std::fs;

    let binary = std::env::var_os("NGSPICE_BIN").expect("set NGSPICE_BIN");
    let scratch = Scratch::new("rational-reference").expect("scratch");
    let dir = scratch.path();
    let body = "v1 in 0 dc 0 ac 1 PWL(0 0 1u 1 1m 1)\na1 in out mdl\nr1 out 0 1k\n.model mdl s_xfer(num_coeff=[1] den_coeff=[1 2 1] denormalized_freq=10000 int_ic=[0 0])";
    let mut c = Circuit::from_netlist(&parsed(body)?)?;
    let rust_ac = run(&mut c, AnalysisKind::Ac, &["dec", "10", "10", "100000"])?;
    let rust_tran = run(
        &mut c,
        AnalysisKind::Transient,
        &["100n", "1m", "0", "100n"],
    )?;
    fs::write(dir.join("reference.cir"), format!("reference\n{body}\n.control\nset filetype=ascii\nac dec 10 10 100000\nwrite ac.raw v(out)\ntran 100n 1m 0 100n\nwrite tran.raw v(out)\ntran 10n 1m 0 10n\nwrite fine.raw v(out)\nquit\n.endc\n.end\n")).unwrap();
    let reference = run_process(
        binary,
        dir,
        &["-b", "-n", "reference.cir"],
        ProcessLimits::ORACLE,
    )
    .unwrap();
    assert!(
        reference.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&reference.stdout),
        String::from_utf8_lossy(&reference.stderr)
    );
    let ac = RawFile::parse(&fs::read_to_string(dir.join("ac.raw")).unwrap()).unwrap();
    let ac = &ac.plots[0].plot;
    assert_eq!(ac.point_count(), rust_ac.point_count());
    let mut ac_error: f64 = 0.;
    for index in 0..ac.point_count() {
        let expected = ac.value("v(out)", index).unwrap();
        let actual = rust_ac.value("v(out)", index).unwrap();
        ac_error = ac_error.max((actual - expected).magnitude());
        close(actual, expected, 1e-8);
    }
    let tran = RawFile::parse(&fs::read_to_string(dir.join("tran.raw")).unwrap()).unwrap();
    let tran = &tran.plots[0].plot;
    let interpolate = |plot: &Plot, t: f64| {
        let time = plot.column("time").unwrap();
        let values = plot.column("v(out)").unwrap();
        let upper = time.partition_point(|v| v.re < t).min(time.len() - 1);
        let lower = upper.saturating_sub(1);
        if lower == upper || time[upper].re == t {
            values[upper].re
        } else {
            let fraction = (t - time[lower].re) / (time[upper].re - time[lower].re);
            fraction.mul_add(values[upper].re - values[lower].re, values[lower].re)
        }
    };
    let fine = RawFile::parse(&fs::read_to_string(dir.join("fine.raw")).unwrap()).unwrap();
    let fine = &fine.plots[0].plot;
    let primitive = |t: f64| {
        let tau = 100e-6;
        if t > 0. {
            t + (t + 2. * tau) * (-t / tau).exp() - 2. * tau
        } else {
            0.
        }
    };
    let mut errors = [0_f64; 3];
    for index in 0..=1000 {
        let time = f64::from(index) * 1e-6;
        let expected = (primitive(time) - primitive(time - 1e-6)) / 1e-6;
        for (error, plot) in errors.iter_mut().zip([&rust_tran, tran, fine]) {
            *error = error.max((interpolate(plot, time) - expected).abs());
        }
    }
    // Independent analytical response distinguishes C's lagged cascade
    // integration error from ours. Its refined trace must converge rather
    // than merely falling inside an inflated cross-simulator tolerance.
    assert!(errors[0] < 2e-5, "Rust analytic error: {errors:?}");
    assert!(
        errors[2] < 0.2 * errors[1] && errors[2] < 1e-4,
        "C refinement: {errors:?}"
    );
    println!(
        "native C AC max error {ac_error:e}; max ramp analytic errors [Rust, C coarse, C fine] {errors:?}"
    );
    Ok(())
}

#[test]
fn uic_transfer_coordinates_preserve_free_response_and_input_forcing() -> SpiceResult<()> {
    let mut c = Circuit::from_netlist(&parsed(
        "v1 in 0 dc 3\na1 in out mdl\nr1 out 0 1k\n.model mdl s_xfer(num_coeff=[1] den_coeff=[1 1] denormalized_freq=1000 int_ic=[2])",
    )?)?;
    let plot = run(
        &mut c,
        AnalysisKind::Transient,
        &["1u", "2m", "0", "1u", "uic"],
    )?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        close(
            plot.value("v(out)", index).unwrap(),
            Complex::real(3. - (-1000. * t).exp()),
            2e-5,
        );
    }
    Ok(())
}

#[test]
fn uic_transfer_output_constraints_detect_capacitor_conflicts() -> SpiceResult<()> {
    for initial in [2., 1.] {
        let deck = parsed(&format!(
            "v1 in 0 dc 3\na1 in out mdl\nr1 out 0 1k\nc1 out 0 1u ic={initial}\n.model mdl s_xfer(num_coeff=[1] den_coeff=[1 1] denormalized_freq=1000 int_ic=[2])"
        ))?;
        let mut c = Circuit::from_netlist(&deck)?;
        let result = run(
            &mut c,
            AnalysisKind::Transient,
            &["1u", "1m", "0", "1u", "uic"],
        );
        if initial == 2. {
            let plot = result?;
            assert!(plot.point_count() > 100);
        } else {
            let error = result.expect_err("inconsistent ideal output").to_string();
            assert!(error.contains("impulse") && error.contains("a1"), "{error}");
        }
    }
    Ok(())
}

#[test]
fn uic_transfer_multiple_coordinates_and_current_sensor_constraints() -> SpiceResult<()> {
    let deck = parsed(
        "v1 in 0 dc 0\na1 in out mdl\nr1 out 0 1k\n.model mdl s_xfer(num_coeff=[1] den_coeff=[1 2 1] denormalized_freq=1000 gain=0 int_ic=[0 1])",
    )?;
    let mut c = Circuit::from_netlist(&deck)?;
    let plot = run(
        &mut c,
        AnalysisKind::Transient,
        &["1u", "2m", "0", "1u", "uic"],
    )?;
    for index in 0..plot.point_count() {
        let t = 1000. * plot.value("time", index).unwrap().re;
        close(
            plot.value("v(out)", index).unwrap(),
            Complex::real((1. + t) * (-t).exp()),
            2e-5,
        );
    }
    let mut c = Circuit::from_netlist(&parsed(
        "i1 0 a dc 1\na1 %i(a) out mdl\nr1 out 0 1k\nc1 a 0 1u ic=0.1\n.model mdl s_xfer(num_coeff=[1] den_coeff=[1 1])",
    )?)?;
    let error = run(
        &mut c,
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u", "uic"],
    )
    .expect_err("current sensor fixes input voltage")
    .to_string();
    assert!(error.contains("impulse") && error.contains("a1"), "{error}");
    Ok(())
}

#[test]
fn transfer_observations_distinguish_input_output_and_state_coordinates() -> SpiceResult<()> {
    let deck = parsed(
        "i1 0 in dc 2 ac 1\na1 %i(in) out mdl\nr1 out 0 1\n.model mdl s_xfer(num_coeff=[1] den_coeff=[1 2 1] denormalized_freq=1000)",
    )?;
    let mut c = Circuit::from_netlist(&deck)?;
    let plot = run(&mut c, AnalysisKind::Ac, &["lin", "1", "100", "100"])?;
    for name in ["i(a1)", "i(a1:input)", "state(a1,0)", "state(a1,1)"] {
        let matches: Vec<_> = plot.variables().iter().filter(|v| v.name == name).collect();
        assert_eq!(matches.len(), 1, "{name}");
        assert_eq!(matches[0].unit, "current");
    }
    close(
        plot.value("i(a1:input)", 0).unwrap(),
        Complex::real(1.),
        1e-12,
    );
    close(
        plot.value("state(a1,0)", 0).unwrap(),
        plot.value("v(out)", 0).unwrap(),
        1e-12,
    );
    close(
        plot.value("i(a1)", 0).unwrap(),
        -plot.value("v(out)", 0).unwrap(),
        1e-12,
    );
    Ok(())
}

#[test]
fn physical_integrator_initial_values_survive_extreme_frequency_normalization() -> SpiceResult<()> {
    // At order two the frequency power itself is outside binary64, but
    // D_leading * physical / frequency^2 is exactly order unity.
    for (frequency, leading, physical) in
        [("1e200", "1e100", "1e300"), ("1e-200", "1e-100", "1e-300")]
    {
        let deck = parsed(&format!(
            "v1 in 0 0\na1 in out mdl\nr1 out 0 1k\n.model mdl s_xfer(num_coeff=[{leading} 0 0] den_coeff=[{leading} {leading} {leading} {leading}] denormalized_freq={frequency} int_ic=[{physical} 0 0])"
        ))?;
        let mut c = Circuit::from_netlist(&deck)?;
        let plot = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
        close(
            plot.value("state(a1,2)", 0).unwrap(),
            Complex::real(1.),
            1e-12,
        );
        close(plot.value("v(out)", 0).unwrap(), Complex::real(1.), 1e-12);
    }
    Ok(())
}

#[test]
fn model_vector_candidates_invalidate_transfer_realization_transactionally() -> SpiceResult<()> {
    use ngspice_rs::analysis::optimization::parameters::{CandidateTopology, CircuitParameterPlan};
    use ngspice_rs::netlist::eval::EvalBudget;

    let source = parsed(
        ".param g=2 p=1 initial=0.4\nv1 in 0 dc 1 ac 1\na1 in out mdl\nr1 out 0 1k\n.model mdl s_xfer(num_coeff=[{g}] den_coeff=[{p} {p}] denormalized_freq=1000 int_ic=[{initial}])",
    )?;
    let plan = CircuitParameterPlan::new(source, &["g", "p", "initial"])?;
    assert_eq!(plan.sites().len(), 4);
    let mut session = plan.session()?;
    for (gain, pole, initial) in [(2., 1., 0.4), (3., 2., 0.6), (2., 1., 0.4)] {
        session.update(&[gain, pole, initial], &mut EvalBudget::default())?;
        assert_eq!(
            session.topology(),
            if gain == 2. {
                CandidateTopology::Reused
            } else {
                CandidateTopology::Rebuilt
            },
        );
        let context = session.analysis_context();
        let op = AnalysisInput::new(AnalysisKind::OperatingPoint)
            .resolve()?
            .run(session.circuit_mut(), &context)?;
        close(
            op.value("v(out)", 0).unwrap(),
            Complex::real(gain * initial),
            1e-12,
        );
        let ac = AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "1", "100", "100"])
            .resolve()?
            .run(session.circuit_mut(), &context)?;
        close(
            ac.value("v(out)", 0).unwrap(),
            Complex::real(gain / pole) / Complex::new(1., std::f64::consts::TAU * 0.1),
            1e-12,
        );
    }
    assert!(
        session
            .update(&[2., 0., 0.4], &mut EvalBudget::default())
            .is_err()
    );
    let context = session.analysis_context();
    let op = AnalysisInput::new(AnalysisKind::OperatingPoint)
        .resolve()?
        .run(session.circuit_mut(), &context)?;
    close(op.value("v(out)", 0).unwrap(), Complex::real(0.8), 1e-12);
    Ok(())
}

#[test]
fn typed_a_ports_roundtrip_without_flattening_domains() -> SpiceResult<()> {
    use ngspice_rs::netlist::ast::CodePortKind;
    let netlist = parsed(
        "a1 %vd[(a GND) (b 0) %v c d %vd(e f)] %d(~GND) logic\na2 %d(d) %gd(out ref) switch",
    )?;
    let ports = &netlist.devices[0].code_ports;
    assert!(netlist.devices[0].nodes.is_empty());
    assert_eq!(ports.len(), 2);
    assert_eq!(
        ports[0].terminals[0].kind,
        CodePortKind::VoltageDifferential
    );
    assert!(ports[0].vector);
    assert_eq!(ports[0].terminals.len(), 5);
    assert_eq!(ports[0].terminals[2].kind, CodePortKind::Voltage);
    assert_eq!(ports[0].terminals[3].kind, CodePortKind::Voltage);
    assert_eq!(
        ports[0].terminals[4].kind,
        CodePortKind::VoltageDifferential
    );
    assert_eq!(ports[0].terminals[0].negative.as_deref(), Some("0"));
    assert_eq!(ports[1].terminals[0].kind, CodePortKind::Digital);
    assert_eq!(ports[1].terminals[0].positive, "gnd");
    assert!(ports[1].terminals[0].inverted);
    let written = ngspice_rs::netlist::write_netlist(&netlist)?;
    let restored =
        Parser::new().parse_deck(&parse_deck_text(Path::new("roundtrip.cir"), &written))?;
    assert!(ngspice_rs::netlist::semantic_diff(&netlist, &restored).is_none());
    for invalid in ["a1 %d[] logic", "a1 %vd(a) logic", "a1 %v(~a) logic"] {
        assert!(parsed(invalid).is_err(), "{invalid}");
    }
    Ok(())
}
