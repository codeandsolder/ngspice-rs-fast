//! Native passive expression laws through the real parser and solver.
use std::path::Path;

use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, semantic_diff, source::parse_deck_text, write_netlist};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};

fn run(body: &str, kind: AnalysisKind, args: &[&str]) -> SpiceResult<Plot> {
    let deck = parse_deck_text(
        Path::new("passive.cir"),
        &format!("passive\n{body}\n.end\n"),
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let mut circuit = Circuit::from_netlist(&netlist)?;
    let mut request =
        AnalysisInput::with_arguments(kind, args.iter().copied().filter(|arg| *arg != "uic"));
    request.uic = args.contains(&"uic");
    request
        .resolve()?
        .run(&mut circuit, &AnalysisContext::default())
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:e} != {expected:e}"
    );
}

#[test]
fn resistance_uses_control_voltage_and_actual_terminal_current() -> SpiceResult<()> {
    let plot = run(
        "vin a 0 2\nvcontrol ctrl 0 1\nr1 a 0 r={1000*(1+v(ctrl))}",
        AnalysisKind::OperatingPoint,
        &[],
    )?;
    close(plot.value("i(vin)", 0).unwrap().re, -1e-3, 1e-10);
    Ok(())
}

#[test]
fn capacitance_and_charge_have_distinct_ac_derivatives() -> SpiceResult<()> {
    let plot = run(
        "vc c 0 dc 1 ac 1\nvq q 0 dc 1 ac 1\nc1 c 0 c={1u*(1+v(c))}\nc2 q 0 q={1u*(v(q)+v(q)^2)}",
        AnalysisKind::Ac,
        &["lin", "1", "1000", "1000"],
    )?;
    let omega = 2. * std::f64::consts::PI * 1000.;
    close(plot.value("i(vc)", 0).unwrap().im, -omega * 2e-6, 1e-10);
    close(plot.value("i(vq)", 0).unwrap().im, -omega * 3e-6, 1e-10);
    Ok(())
}

#[test]
fn capacitance_and_charge_distinguish_controlled_transient_storage() -> SpiceResult<()> {
    let plot = run(
        "vc c 0 PWL(0 1 1m 2)\nvq q 0 PWL(0 1 1m 2)\nc1 c 0 c={1u*(1+v(c))}\nc2 q 0 q={1u*(v(q)+v(q)^2)}",
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u"],
    )?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        if !(20e-6..=980e-6).contains(&t) {
            continue;
        }
        let voltage = 1. + t * 1000.;
        close(
            plot.value("i(vc)", index).unwrap().re,
            -(1. + voltage) * 1e-3,
            1e-5,
        );
        close(
            plot.value("i(vq)", index).unwrap().re,
            -(1. + 2. * voltage) * 1e-3,
            1e-5,
        );
    }
    Ok(())
}

#[test]
fn variable_inductance_uses_current_derivative_not_flux_product() -> SpiceResult<()> {
    let plot = run(
        "iin 0 a PWL(0 0 1m 1)\nl1 a 0 l={1m*(1+i(l1))}",
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u"],
    )?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        if !(20e-6..=980e-6).contains(&t) {
            continue;
        }
        close(plot.value("v(a)", index).unwrap().re, 1. + t * 1000., 1e-3);
    }
    Ok(())
}

#[test]
fn round_trip_preserves_static_and_runtime_value_categories() -> SpiceResult<()> {
    let parse =
        |text: &str| Parser::new().parse_deck(&parse_deck_text(Path::new("roundtrip.cir"), text));
    let original = parse(
        "title\n.param r=1k\nr1 a 0 {r}\nr2 a 0 {r*(1+v(a))}\nc1 a 0 q={1u*v(a)^2}\nl1 a 0 l={1m*(1+time)}\n.end\n",
    )?;
    let text = write_netlist(&original)?;
    let restored = parse(&text)?;
    assert!(semantic_diff(&original, &restored).is_none());
    Ok(())
}

#[test]
fn ddt_uses_accepted_history_and_exact_ac_derivative() -> SpiceResult<()> {
    let body = "vin x 0 dc 2 ac 1 PWL(0 2 1m 3)\nb1 y 0 v=ddt(v(x)^2)\nr1 y 0 1k";
    let op = run(body, AnalysisKind::OperatingPoint, &[])?;
    close(op.value("v(y)", 0).unwrap().re, 0., 1e-12);
    let ac = run(body, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
    let y = ac.value("v(y)", 0).unwrap();
    close(y.re, 0., 1e-10);
    close(y.im, 4. * std::f64::consts::TAU * 1000., 1e-7);
    for method in ["method=trap", "method=gear"] {
        let mut errors = [0_f64; 2];
        for (index, step) in ["1u", "250n"].into_iter().enumerate() {
            let tran = run(
                body,
                AnalysisKind::Transient,
                &[step, "1m", "0", step, method],
            )?;
            let mut checked = 0;
            for point in 0..tran.point_count() {
                let t = tran.value("time", point).unwrap().re;
                if (20e-6..=980e-6).contains(&t) {
                    let actual = tran.value("v(y)", point).unwrap().re;
                    let expected = 2000. * (2. + 1000. * t);
                    errors[index] = errors[index].max((actual - expected).abs());
                    checked += 1;
                }
            }
            assert!(checked > 100);
        }
        // The initial ramp has a derivative discontinuity at t=0. Startup
        // error is proportional to the step for trapezoid; Gear damps it.
        // Assert both convergence on refinement and a curvature-scaled
        // absolute error, rather than hiding the transient startup effect.
        assert!(
            errors[0] <= 4. && errors[1] <= 1.,
            "{method}: derivative error {errors:?}"
        );
        assert!(
            errors[1] <= 0.5 * errors[0] + 1e-6,
            "{method}: refinement {errors:?}"
        );
    }
    Ok(())
}

#[test]
fn nested_ddt_is_independent_and_composable() -> SpiceResult<()> {
    let body = "vin x 0 dc 1 ac 1\nb1 y 0 v=ddt(ddt(v(x)))+2*ddt(v(x))\nr1 y 0 1k";
    let ac = run(body, AnalysisKind::Ac, &["lin", "1", "10", "10"])?;
    let y = ac.value("v(y)", 0).unwrap();
    let omega = std::f64::consts::TAU * 10.;
    close(y.re, -omega * omega, 1e-8);
    close(y.im, 2. * omega, 1e-9);
    Ok(())
}

#[test]
fn ddt_generated_coordinates_do_not_alias_user_names() -> SpiceResult<()> {
    let body = "vin x 0 dc 1 ac 1\nvother __dynamic_0 0 dc 5\nb__dynamic_1 a 0 v=2\nb1 y 0 v=ddt(v(x))+v(__dynamic_0)+v(a)";
    let op = run(body, AnalysisKind::OperatingPoint, &[])?;
    close(op.value("v(y)", 0).unwrap().re, 7., 1e-12);
    Ok(())
}

#[test]
fn lexical_macros_classify_runtime_and_static_passives() -> SpiceResult<()> {
    for body in [
        ".func v(x) {x*1000}\nvin a 0 2\nr1 a 0 {v(2)}",
        ".func i(x) {x*1000}\nvin a 0 2\nr1 a 0 {i(2)}",
        ".func f(x) {x*(1+v(ctrl))}\n.func g(x) {f(x)}\nvin a 0 2\nvctrl ctrl 0 1\nr1 a 0 {g(1000)}",
        ".func v(x) {x*1000}\nvin a 0 2\nr1 a 0 {v(2)+time}",
        ".func f(x) {x*(1+i(rctrl))}\nvin a 0 2\nvctrl ctrl 0 1\nrctrl ctrl 0 1\nr1 a 0 {f(1000)}",
    ] {
        let plot = run(body, AnalysisKind::OperatingPoint, &[])?;
        close(plot.value("i(vin)", 0).unwrap().re, -1e-3, 1e-10);
    }
    Ok(())
}

#[test]
fn lexical_passive_macros_and_current_probes_are_instance_local() -> SpiceResult<()> {
    let plot = run(
        ".subckt cell a ctrl params: gain=1000\n.func resistance(x) {gain*(1+i(rctrl))}\nrctrl ctrl 0 1\nr1 a 0 {resistance(0)}\n.ends cell\nva a 0 2\nvb b 0 2\nvctrl1 ctrl1 0 1\nvctrl2 ctrl2 0 2\nxa a ctrl1 cell gain=1000\nxb b ctrl2 cell gain=2000",
        AnalysisKind::OperatingPoint,
        &[],
    )?;
    close(plot.value("i(va)", 0).unwrap().re, -1e-3, 1e-10);
    close(plot.value("i(vb)", 0).unwrap().re, -2. / 6000., 1e-10);
    Ok(())
}

#[test]
fn differential_capacitor_uic_preserves_initial_voltage_and_nonlinear_decay() -> SpiceResult<()> {
    for (cards, expected) in [
        ("c1 a 0 c={1u*(1+v(a))} ic=2", 2.),
        ("c1 a 0 c={1u*(1+v(a))}\n.ic v(a)=2", 2.),
        ("c1 a 0 c={1u*(1+v(a))} ic=1.5\n.ic v(a)=2", 1.5),
    ] {
        let body = format!("r1 a 0 1k\n{cards}\n.tran 1u 2m 0 1u uic");
        let deck = Parser::new().parse_deck(&parse_deck_text(
            Path::new("cap-ic.cir"),
            &format!("test\n{body}\n.end\n"),
        ))?;
        let config = ngspice_rs::analysis::RunConfig::from_netlist(&deck)?;
        let plan = config.plan_for(&deck.analyses[0])?;
        let mut circuit = config.circuit(&deck)?;
        let plot = plan.run(&mut circuit, &config.context())?;
        assert!(plot.point_count() > 100);
        for index in 0..plot.point_count() {
            let t = plot.value("time", index).unwrap().re;
            let v = plot.value("v(a)", index).unwrap().re;
            close(v + v.ln(), expected + f64::ln(expected) - t * 1000., 2e-4);
        }
    }
    Ok(())
}

#[test]
fn differential_inductor_uic_releases_rate_equation_not_current_constitutive_law() -> SpiceResult<()>
{
    let plot = run(
        "r1 a 0 10\nl1 a 0 l={10m*(1+i(l1))} ic=0.5",
        AnalysisKind::Transient,
        &["1u", "2m", "0", "1u", "uic"],
    )?;
    assert!(plot.point_count() > 100);
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        let i = -plot.value("v(a)", index).unwrap().re / 10.;
        close(i + i.ln(), 0.5 + 0.5_f64.ln() - t * 1000., 2e-4);
    }
    Ok(())
}

#[test]
fn differential_storage_uic_detects_conflicting_ideal_sources() {
    for body in [
        "v1 a 0 1\nc1 a 0 c={1u*(1+v(a))} ic=2",
        "i1 0 a 1\nl1 a 0 l={1m*(1+i(l1))} ic=0.5",
    ] {
        let error = run(
            body,
            AnalysisKind::Transient,
            &["1u", "1m", "0", "1u", "uic"],
        )
        .expect_err("incompatible initial storage")
        .to_string();
        assert!(error.contains("impulse"), "{body}: {error}");
    }
}

#[test]
fn nonlinear_lc_uic_conserves_charge_balance_and_energy() -> SpiceResult<()> {
    let body = "c1 a 0 c={1u*(1+v(a)^2)} ic=1\nl1 a 0 l={1m*(1+1000*i(l1)^2)} ic=0\nbimon il 0 v=i(l1)\nbcmon ic 0 v=i(c1)";
    let plot = run(
        body,
        AnalysisKind::Transient,
        &["100n", "1m", "0", "100n", "uic"],
    )?;
    assert!(plot.point_count() > 1000);
    let initial_energy = 0.75e-6;
    let mut largest_current: f64 = 0.;
    for index in 0..plot.point_count() {
        let v = plot.value("v(a)", index).unwrap().re;
        let i = plot.value("v(il)", index).unwrap().re;
        let ic = plot.value("v(ic)", index).unwrap().re;
        close(i + ic, 0., 1e-9);
        largest_current = largest_current.max(i.abs());
        let e = 1e-6 * (0.5 * v * v + 0.25 * v.powi(4)) + 1e-3 * (0.5 * i * i + 250. * i.powi(4));
        close(e, initial_energy, 1e-3 * initial_energy);
    }
    assert!(largest_current > 0.02);
    Ok(())
}

#[test]
fn arbitrary_charge_uic_obeys_nonlinear_charge_decay() -> SpiceResult<()> {
    let plot = run(
        "r1 out 0 1k\nc1 out 0 q={1u*(v(out)+v(out)^2)} ic=2",
        AnalysisKind::Transient,
        &["1u", "2m", "0", "1u", "uic"],
    )?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        let v = plot.value("v(out)", index).unwrap().re;
        assert!(v > 0.);
        // dQ/dt = -v/R: ln(v)+2v = ln(v0)+2v0-t/(R*C).
        close(v.ln() + 2. * v, 2_f64.ln() + 4. - t / 1e-3, 3e-3);
    }
    Ok(())
}

#[test]
fn arbitrary_charge_uic_enforces_physical_charge_not_a_voltage_surrogate() -> SpiceResult<()> {
    // Local IC is evaluated with the supplied control hints (zero here).
    // The right-limit control is 1 V, so retaining Q0=1 uC means -1 V
    // across the capacitor, not the 1 V at which the local IC was given.
    let plot = run(
        "vctrl ctrl 0 1\nr1 out 0 1k\nc1 out 0 q={1u*(v(out)+2*v(ctrl))} ic=1",
        AnalysisKind::Transient,
        &["1u", "2m", "0", "1u", "uic"],
    )?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        close(
            plot.value("v(out)", index).unwrap().re,
            -(-t / 1e-3).exp(),
            1e-3,
        );
    }
    Ok(())
}

#[test]
fn redundant_charge_constraints_preserve_algebraic_relations_and_reject_conflicts()
-> SpiceResult<()> {
    let deck =
        |initial: f64| format!("v1 out 0 2\nc1 out 0 q={{1u*v(out)^2}} ic={initial}\nr1 out 0 1k");
    let args = ["1u", "100u", "0", "1u", "uic"];
    let plot = run(&deck(2.), AnalysisKind::Transient, &args)?;
    for index in 0..plot.point_count() {
        close(plot.value("v(out)", index).unwrap().re, 2., 1e-10);
        close(plot.value("i(v1)", index).unwrap().re, -2e-3, 1e-10);
    }
    let error = run(&deck(1.), AnalysisKind::Transient, &args)
        .unwrap_err()
        .to_string();
    assert!(error.contains("initial") && error.contains("c1"), "{error}");
    // Two independent charge histories share a voltage, so their initial
    // currents cannot be independently determined by the instantaneous
    // system. The actual transient equations determine both derivatives.
    let plot = run(
        "r1 out 0 1k\nc1 out 0 q={1u*v(out)^2} ic=1\nc2 out 0 q={2u*v(out)^2} ic=1",
        AnalysisKind::Transient,
        &["1u", "1m", "0", "1u", "uic"],
    )?;
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        close(plot.value("v(out)", index).unwrap().re, 1. - t / 6e-3, 1e-3);
    }
    Ok(())
}

#[test]
fn arbitrary_charge_uic_resolves_its_own_current_dependency() -> SpiceResult<()> {
    let plot = run(
        "r1 out 0 1k\nc1 out 0 q={1u*(v(out)-500*i(c1))} ic=1",
        AnalysisKind::Transient,
        &["1u", "2m", "0", "1u", "uic"],
    )?;
    // At t=0 the seeded current is zero and the local voltage is 1 V,
    // hence Q0=1 uC. With KCL i=-v/1k the actual charge is 1.5 uF*v.
    for index in 0..plot.point_count() {
        let t = plot.value("time", index).unwrap().re;
        close(
            plot.value("v(out)", index).unwrap().re,
            (2. / 3.) * (-t / 1.5e-3).exp(),
            1e-3,
        );
    }
    Ok(())
}
