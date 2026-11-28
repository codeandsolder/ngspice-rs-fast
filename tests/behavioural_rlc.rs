//! Native passive expression laws through the real parser and solver.
use std::path::Path;

use ngspice_rs::analysis::{AnalysisInput, Plot, RunConfig};
use ngspice_rs::netlist::{Parser, semantic_diff, source::parse_deck_text, write_netlist};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};

fn run(body: &str, kind: AnalysisKind, args: &[&str]) -> SpiceResult<Plot> {
    run_observed(body, kind, args, &[])
}

fn run_observed(
    body: &str,
    kind: AnalysisKind,
    args: &[&str],
    observations: &[&str],
) -> SpiceResult<Plot> {
    let deck = parse_deck_text(
        Path::new("passive.cir"),
        &format!("passive\n{body}\n.end\n"),
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    circuit.set_observations(
        &observations
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>(),
    )?;
    let mut request =
        AnalysisInput::with_arguments(kind, args.iter().copied().filter(|arg| *arg != "uic"));
    request.uic = args.contains(&"uic");
    config.plan(request)?.run(&mut circuit, &config.context())
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

#[test]
fn physical_current_and_constitutive_rate_have_distinct_plot_coordinates() -> SpiceResult<()> {
    let body = "v1 a 0 dc 1 ac 1\nc1 a 0 c={1u*(1+v(a))}\nv2 b 0 dc 0 ac 1\nl1 b c l={1m*(1+i(l1))}\nr1 c 0 1k";
    let plot = run(body, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
    for (name, unit) in [
        ("i(c1)", "current"),
        ("dvdt(c1)", "voltage/s"),
        ("i(l1)", "current"),
        ("didt(l1)", "current/s"),
    ] {
        let matches: Vec<_> = plot.variables().iter().filter(|v| v.name == name).collect();
        assert_eq!(matches.len(), 1, "{name}");
        assert_eq!(matches[0].unit, unit);
    }
    let omega = std::f64::consts::TAU * 1000.;
    close(plot.value("dvdt(c1)", 0).unwrap().im, omega, 1e-8);
    close(plot.value("i(c1)", 0).unwrap().im, omega * 2e-6, 1e-12);
    let i = plot.value("i(l1)", 0).unwrap();
    let derivative = plot.value("didt(l1)", 0).unwrap();
    close(derivative.re, -omega * i.im, 1e-8);
    close(derivative.im, omega * i.re, 1e-8);
    Ok(())
}

#[test]
fn native_current_probe_reads_model_backed_inductors() -> SpiceResult<()> {
    let body =
        "v1 in 0 dc 2 ac 1\nr1 in a 1k\nl1 a 0 mdl\n.model mdl l(ind=1m)\nb1 mon 0 v=1000*i(l1)";
    let plot = run(body, AnalysisKind::OperatingPoint, &[])?;
    close(plot.value("v(mon)", 0).unwrap().re, 2., 1e-12);
    let plot = run(body, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
    let current = plot.value("i(l1)", 0).unwrap();
    let mon = plot.value("v(mon)", 0).unwrap();
    close(mon.re, 1000. * current.re, 1e-12);
    close(mon.im, 1000. * current.im, 1e-12);
    Ok(())
}

#[test]
fn uic_derivative_uses_actual_initial_input_instead_of_uninitialized_nodes() -> SpiceResult<()> {
    for method in ["method=trap", "method=gear"] {
        let constant = run(
            "v1 in 0 2\nb1 out 0 v=ddt(v(in)^2)\nr1 out 0 1k",
            AnalysisKind::Transient,
            &["1u", "20u", "0", "1u", "uic", method],
        )?;
        for point in 0..constant.point_count() {
            close(constant.value("v(out)", point).unwrap().re, 0., 1e-9);
        }
        let ramp = run(
            "v1 in 0 PWL(0 2 1m 3)\nb1 out 0 v=ddt(v(in)^2)\nr1 out 0 1k",
            AnalysisKind::Transient,
            &["250n", "50u", "0", "250n", "uic", method],
        )?;
        assert!(ramp.point_count() > 20);
        for point in 0..ramp.point_count() {
            let t = ramp.value("time", point).unwrap().re;
            close(
                ramp.value("v(out)", point).unwrap().re,
                2000. * (2. + 1000. * t),
                1.,
            );
        }
    }
    Ok(())
}

#[test]
fn uic_derivative_can_define_an_independent_feedback_state() -> SpiceResult<()> {
    for method in ["method=trap", "method=gear"] {
        // v = 1 - tau dv/dt, with independent initial v=0.3.
        let plot = run(
            "b1 out 0 v=1-1m*ddt(v(out))\nr1 out 0 1k\n.ic v(out)=0.3",
            AnalysisKind::Transient,
            &["1u", "2m", "0", "1u", "uic", method],
        )?;
        assert!(plot.point_count() > 100);
        for point in 0..plot.point_count() {
            let t = plot.value("time", point).unwrap().re;
            close(
                plot.value("v(out)", point).unwrap().re,
                1. - 0.7 * (-1000. * t).exp(),
                2e-5,
            );
        }
    }
    Ok(())
}

#[test]
fn uic_derivative_respects_physical_storage_and_redundant_inputs() -> SpiceResult<()> {
    // The capacitor is the independent stored voltage; both differentiators
    // observe it. The exact constitutive current determines its decay rate.
    let plot = run(
        "c1 a 0 c={1u*(1+v(a))} ic=1\nr1 a 0 1k\nb1 y 0 v=ddt(v(a))\nb2 z 0 v=ddt(2*v(a))",
        AnalysisKind::Transient,
        &["250n", "200u", "0", "250n", "uic", "method=gear"],
    )?;
    for point in 0..plot.point_count() {
        let v = plot.value("v(a)", point).unwrap().re;
        let y = plot.value("v(y)", point).unwrap().re;
        let z = plot.value("v(z)", point).unwrap().re;
        close(y, -1000. * v / (1. + v), 0.2);
        close(z, 2. * y, 1e-7);
    }
    Ok(())
}

#[test]
fn expression_controlled_inductors_require_an_explicit_coupled_flux_law() {
    let error = run(
        "v1 in 0 dc 0 ac 1\nr1 in p 1\nl1 p 0 l={1m*(1+i(v1)^2)}\nl2 out 0 1m\nr2 out 0 1\nk1 l1 l2 0.5",
        AnalysisKind::Ac,
        &["lin", "1", "1000", "1000"],
    ).expect_err("no constant mutual coupling law is defined").to_string();
    assert!(error.contains("explicit coupled flux law"), "{error}");
}

#[test]
fn signed_differential_coefficients_are_not_clamped() -> SpiceResult<()> {
    let plot = run(
        "vr r 0 dc 1 ac 1\nr1 r 0 r={-1000*(1+v(r))}\nvc c 0 dc 1 ac 1\nc1 c 0 c={-1u*(1+v(c))}\nvl li 0 dc 0 ac 1\nrl li l 1\nl1 l 0 l={-1m*(1+v(r))}",
        AnalysisKind::Ac,
        &["lin", "1", "1000", "1000"],
    )?;
    let omega = std::f64::consts::TAU * 1000.;
    // The R differential slope includes its own voltage-control term.
    close(plot.value("i(vr)", 0).unwrap().re, 0.00025, 1e-12);
    close(plot.value("i(vc)", 0).unwrap().im, omega * 2e-6, 1e-12);
    let expected = -omega * 2e-3 / (1. + (omega * 2e-3).powi(2));
    close(plot.value("i(vl)", 0).unwrap().im, expected, 1e-12);
    Ok(())
}

#[test]
fn controlled_nonlinear_charge_preserves_history_through_stiff_transition() -> SpiceResult<()> {
    let deck = ".options reltol=1e-8 abstol=1e-12 vntol=1e-9\n\
        vctrl ctrl 0 PWL(0 0 0.5m 0 0.500001m 1 2m 1)\n\
        r1 out 0 1k\n\
        c1 out 0 q={1u*(v(out)-v(ctrl)+0.5*(v(out)-v(ctrl))^3)}";
    let start = 0.5e-3;
    let end = 0.500_001e-3;
    // After the ramp, dq/dv = 1 uF * (1 + 1.5*(v-1)^2).
    // Integrate dq/dt = -v/R analytically.
    let primitive = |v: f64| 2.5 * v.ln() - 3. * v + 0.75 * v * v;
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            deck,
            AnalysisKind::Transient,
            &["1u", "2m", "0", "1u", method],
        )?;
        let mut start_seen = false;
        let mut edge = None;
        let mut previous = (0., 0.);
        let mut integral = 0.;
        let mut max_balance: f64 = 0.;
        let mut max_kcl: f64 = 0.;
        let mut max_decay: f64 = 0.;
        for index in 0..plot.point_count() {
            let t = plot.value("time", index).unwrap().re;
            let v = plot.value("v(out)", index).unwrap().re;
            let control = plot.value("v(ctrl)", index).unwrap().re;
            let current = plot.value("i(c1)", index).unwrap().re;
            let u = v - control;
            let charge = 1e-6 * (u + 0.5 * u.powi(3));
            integral += 0.5e-3 * (v + previous.1) * (t - previous.0);
            previous = (t, v);
            max_balance = max_balance.max((charge + integral).abs());
            max_kcl = max_kcl.max((current + v / 1000.).abs());
            start_seen |= (t - start).abs() < 1e-15;
            if (t - end).abs() < 1e-15 {
                close(v, 1., 1.1e-6);
                edge = Some(primitive(v));
            }
            if t > end + 1e-15 {
                let initial = edge.expect("accepted control ramp endpoint");
                max_decay = max_decay.max((primitive(v) - initial + (t - end) / 1e-3).abs());
            }
        }
        assert!(start_seen && edge.is_some());
        eprintln!(
            "{method}: {} points, KCL {max_kcl:e} A, charge balance {max_balance:e} C, decay {max_decay:e}",
            plot.point_count()
        );
        assert!(max_kcl < 2e-11);
        assert!(max_balance < 5e-10);
        assert!(max_decay < 2e-5);
    }
    Ok(())
}

#[test]
fn explicit_charge_coordinate_is_physical_in_dc_and_ac() -> SpiceResult<()> {
    let body = "vdrive a 0 dc 2 ac 1\nvref b 0 dc 1\nc1 a b q={1u*(v(a,b)+v(a,b)^2)}";
    let op = run(body, AnalysisKind::OperatingPoint, &[])?;
    close(op.value("q(c1)", 0).unwrap().re, 2e-6, 1e-15);
    let ac = run(body, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
    let charge = ac.value("q(c1)", 0).unwrap();
    let current = ac.value("i(c1)", 0).unwrap();
    close(charge.re, 3e-6, 1e-15);
    close(charge.im, 0., 1e-15);
    close(
        current.re,
        -std::f64::consts::TAU * 1000. * charge.im,
        1e-12,
    );
    close(current.im, std::f64::consts::TAU * 1000. * charge.re, 1e-12);
    Ok(())
}

#[test]
fn initial_charge_law_uses_charge_not_voltage_tolerance() {
    let error = run(
        "v1 a 0 2\nc1 a 0 q={1p*v(a)^2} ic=1",
        AnalysisKind::Transient,
        &["1u", "2u", "0", "1u", "uic"],
    )
    .expect_err("picocoulomb inconsistency exceeds the charge tolerance")
    .to_string();
    assert!(error.contains("c1") && error.contains("initial"), "{error}");
}

#[test]
fn nested_derivative_uic_starts_at_consistent_right_limit() -> SpiceResult<()> {
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            "v1 in 0 PWL(0 2 1m 3)\nb1 first 0 v=ddt(v(in))\nb2 out 0 v=ddt(v(first))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1e-9",
            AnalysisKind::Transient,
            &["1u", "50u", "0", "1u", "uic", method],
        )?;
        assert!(plot.point_count() > 20);
        for point in 0..plot.point_count() {
            close(plot.value("v(first)", point).unwrap().re, 1000., 1e-6);
            close(plot.value("v(out)", point).unwrap().re, 0., 1e-6);
            close(plot.value("i(b2)", point).unwrap().re, 0., 1e-9);
        }
    }
    Ok(())
}

#[test]
fn non_uic_derivative_input_jump_is_reported_as_a_discontinuity() {
    let result = run(
        "v1 in 0 PWL(0 2 1m 3)\nb1 out 0 v=ddt(ddt(v(in)))\nr1 out 0 1k",
        AnalysisKind::Transient,
        &["1u", "50u", "0", "1u", "method=gear"],
    );
    let error = result
        .expect_err("prehistory slope jump is not a classical second derivative")
        .to_string();
    assert!(error.contains("prehistory discontinuity"), "{error}");
}

#[test]
fn nonlinear_derivative_chain_preserves_analytic_polynomial_and_kcl() -> SpiceResult<()> {
    // With u = t/1ms, x = 2 + u, first = (x^4)' = 4*x^3,
    // second = (first^2)' = 96*x^5, third = (second*x)' = 576*x^5.
    let body = "v1 in 0 PWL(0 2 1m 3)\nb1 first 0 v=1m*ddt(v(in)^4)\nb2 second 0 v=1m*ddt(v(first)^2)\nb3 out 0 v=1m*ddt(v(second)*v(in))\nr1 first 0 1k\nr2 second 0 1k\nr3 out 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["2u", "50u", "0", "2u", "uic", method],
        )?;
        assert!(plot.point_count() > 20);
        for point in 0..plot.point_count() {
            let t = plot.value("time", point).unwrap().re;
            let x = 2. + 1000. * t;
            let expected = [4. * x.powi(3), 96. * x.powi(5), 576. * x.powi(5)];
            for ((node, branch), value) in [
                ("v(first)", "i(b1)"),
                ("v(second)", "i(b2)"),
                ("v(out)", "i(b3)"),
            ]
            .into_iter()
            .zip(expected)
            {
                close(plot.value(node, point).unwrap().re, value, 2e-10 * value);
                close(
                    plot.value(branch, point).unwrap().re,
                    -value * 1e-3,
                    2e-13 * value,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn derivative_array_resolves_current_controls_and_algebraic_coupling() -> SpiceResult<()> {
    let body = "v1 in 0 PWL(0 2 1m 3)\nr1 in 0 1k\ne1 scaled 0 in 0 2\nb1 first 0 v=1m*ddt((1000*i(v1))^3+v(scaled))\nb2 out 0 v=1m*ddt(v(first))\nr2 out 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["2u", "50u", "0", "2u", "uic", method],
        )?;
        for point in 0..plot.point_count() {
            let x = 2. + 1000. * plot.value("time", point).unwrap().re;
            close(
                plot.value("v(first)", point).unwrap().re,
                2. - 3. * x * x,
                1e-9,
            );
            close(plot.value("v(out)", point).unwrap().re, -6. * x, 1e-9);
            close(plot.value("i(v1)", point).unwrap().re, -1e-3 * x, 1e-13);
        }
    }
    Ok(())
}

#[test]
fn nonlinear_rc_derivative_array_preserves_independent_storage() -> SpiceResult<()> {
    let body = "c1 a 0 c={1u*(1+v(a))} ic=1\nr1 a 0 1k\nb1 first 0 v=1m*ddt(v(a))\nb2 second 0 v=1m*ddt(v(first))\nb3 third 0 v=1m*ddt(v(second))\nr2 third 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["1u", "100u", "0", "1u", "uic", method],
        )?;
        assert!(plot.point_count() > 50);
        for point in 0..plot.point_count() {
            let t = plot.value("time", point).unwrap().re;
            let v = plot.value("v(a)", point).unwrap().re;
            // Integrate dv/du = -v/(1+v) exactly: v+log(v)=1-u.
            close(v + v.ln(), 1. - 1000. * t, 2e-6);
            close(
                plot.value("v(first)", point).unwrap().re,
                -v / (1. + v),
                1e-9,
            );
            close(
                plot.value("v(second)", point).unwrap().re,
                v / (1. + v).powi(3),
                1e-9,
            );
            close(
                plot.value("v(third)", point).unwrap().re,
                v * (2. * v - 1.) / (1. + v).powi(5),
                1e-9,
            );
        }
    }
    Ok(())
}

#[test]
fn source_corner_changes_first_derivative_without_creating_a_false_impulse() -> SpiceResult<()> {
    let body = "v1 in 0 PWL(0 2 10u 3 30u 1)\nb1 out 0 v=1u*ddt(v(in))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["1u", "25u", "0", "1u", "uic", method],
        )?;
        let mut corner = false;
        for point in 0..plot.point_count() {
            let t = plot.value("time", point).unwrap().re;
            let at_corner = (t - 10e-6).abs() < 1e-18;
            corner |= at_corner;
            close(
                plot.value("v(out)", point).unwrap().re,
                if t < 10e-6 && !at_corner { 0.1 } else { -0.1 },
                1e-9,
            );
        }
        assert!(corner);
    }
    Ok(())
}

#[test]
fn source_corner_with_discontinuous_derivative_input_is_rejected() {
    let body = "v1 in 0 PWL(0 2 10u 3 30u 1)\nb1 out 0 v=1u*ddt(1u*ddt(v(in)))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let error = run(
            body,
            AnalysisKind::Transient,
            &["1u", "25u", "0", "1u", "uic", method],
        )
        .expect_err("the inner slope jump requires an impulse")
        .to_string();
        assert!(
            error.contains("source event") && error.contains("derivative input jumps"),
            "{error}"
        );
    }
}

#[test]
fn sinusoidal_capacitor_loaded_derivative_matches_physical_current() -> SpiceResult<()> {
    // A smooth right limit with zero initial derivative output, but a
    // nonzero second derivative of the source. Both explicit physical
    // capacitors have an algebraically determined current.
    let body = "v1 in 0 sin(2 0.1 1k 0 0 90)\nb1 y 0 v=1m*ddt(v(in)^2)\nr1 y 0 1k\nrsh1 in 0 1k\nrsh2 y 0 1k\ncsh1 in 0 1n\ncsh2 y 0 1n\n.options reltol=1e-6 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run_observed(
            body,
            AnalysisKind::Transient,
            &["5u", "50u", "0", "5u", method],
            &["@csh1[i]", "@csh2[i]"],
        )?;
        let omega = std::f64::consts::TAU * 1000.;
        for point in 0..plot.point_count() {
            let t = plot.value("time", point).unwrap().re;
            let (sin, cos) = (omega * t).sin_cos();
            let x = 2. + 0.1 * cos;
            let rate = -0.1 * omega * sin;
            let acceleration = -0.1 * omega * omega * cos;
            let y = 2e-3 * x * rate;
            let y_rate = 2e-3 * rate.mul_add(rate, x * acceleration);
            close(
                plot.value("@csh1[i]", point).unwrap().re,
                1e-9 * rate,
                1e-13,
            );
            close(
                plot.value("@csh2[i]", point).unwrap().re,
                1e-9 * y_rate,
                1e-13,
            );
            close(plot.value("v(y)", point).unwrap().re, y, 1e-9);
            close(
                plot.value("i(v1)", point).unwrap().re,
                -1e-3 * x - 1e-9 * rate,
                1e-12,
            );
            close(
                plot.value("i(b1)", point).unwrap().re,
                -2e-3 * y - 1e-9 * y_rate,
                1e-12,
            );
        }
    }
    Ok(())
}

#[test]
fn derivative_initial_storage_tolerance_is_independent_of_common_mode() -> SpiceResult<()> {
    for method in ["method=trap", "method=gear"] {
        for common_mode in [0., 1e9, -1e9] {
            let body = format!(
                "v1 p 0 {common_mode}\nv2 n 0 {common_mode}\nc1 p n 1u ic=1\nb1 out 0 v=1u*ddt(v(p)-v(n))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n"
            );
            let error = run(
                &body,
                AnalysisKind::Transient,
                &["1u", "10u", "0", "1u", "uic", method],
            )
            .expect_err("contradictory differential voltage must not be hidden by common mode")
            .to_string();
            assert!(
                error.contains("c1")
                    && error.contains("initial voltage")
                    && error.contains("impulse"),
                "{error}"
            );
            let consistent = format!(
                "v1 p 0 {}\nv2 n 0 {common_mode}\nc1 p n 1u ic=1\nb1 out 0 v=1u*ddt(v(p)-v(n))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n",
                common_mode + 1.
            );
            let plot = run(
                &consistent,
                AnalysisKind::Transient,
                &["1u", "10u", "0", "1u", "uic", method],
            )?;
            for point in 0..plot.point_count() {
                close(
                    plot.value("v(p)", point).unwrap().re - plot.value("v(n)", point).unwrap().re,
                    1.,
                    1e-9,
                );
                close(plot.value("v(out)", point).unwrap().re, 0., 1e-9);
            }
        }
    }
    Ok(())
}

#[test]
fn derivative_chain_preserves_rational_transfer_initial_coordinates() -> SpiceResult<()> {
    let body = "v1 in 0 dc 3\na1 in out mdl\nr1 out 0 1k\n.model mdl s_xfer(num_coeff=[1] den_coeff=[1 1] denormalized_freq=1000 int_ic=[2])\nb1 first 0 v=1m*ddt(v(out))\nb2 second 0 v=1m*ddt(v(first))\nr2 second 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["1u", "100u", "0", "1u", "uic", method],
        )?;
        for point in 0..plot.point_count() {
            let t = plot.value("time", point).unwrap().re;
            let out = plot.value("v(out)", point).unwrap().re;
            close(out, 3. - (-1000. * t).exp(), 2e-6);
            close(plot.value("v(first)", point).unwrap().re, 3. - out, 1e-9);
            close(plot.value("v(second)", point).unwrap().re, out - 3., 1e-9);
        }
    }
    Ok(())
}

#[test]
fn derivative_chain_preserves_inductor_initial_flux_and_exact_kcl() -> SpiceResult<()> {
    let body = "v1 in 0 0\nr1 in a 1\nl1 a 0 1m ic=1\nb1 first 0 v=1m*ddt(i(l1))\nb2 second 0 v=1m*ddt(v(first))\nr2 second 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["1u", "100u", "0", "1u", "uic", method],
        )?;
        for point in 0..plot.point_count() {
            let t = plot.value("time", point).unwrap().re;
            let current = plot.value("i(l1)", point).unwrap().re;
            close(current, (-1000. * t).exp(), 2e-6);
            close(plot.value("v(a)", point).unwrap().re, -current, 1e-9);
            close(plot.value("v(first)", point).unwrap().re, -current, 1e-9);
            close(plot.value("v(second)", point).unwrap().re, current, 1e-9);
        }
    }
    Ok(())
}

#[test]
fn long_derivative_chain_is_independent_of_print_step_normalization() -> SpiceResult<()> {
    use std::fmt::Write;
    let mut body = String::from("v1 in 0 PWL(0 2 1m 3)\n");
    let mut previous = String::from("in");
    for index in 1..=60 {
        let next = format!("n{index}");
        writeln!(&mut body, "b{index} {next} 0 v=ddt(v({previous}))").unwrap();
        previous = next;
    }
    body.push_str("r1 n60 0 1k\n.options reltol=1e-8 vntol=1n");
    let plot = run(
        &body,
        AnalysisKind::Transient,
        &["1u", "2u", "0", "1u", "uic", "method=gear"],
    )?;
    assert!(plot.point_count() > 10);
    for point in 0..plot.point_count() {
        for index in 1..=60 {
            let expected = if index == 1 { 1000. } else { 0. };
            close(
                plot.value(&format!("v(n{index})"), point).unwrap().re,
                expected,
                1e-9,
            );
        }
        close(plot.value("i(b60)", point).unwrap().re, 0., 1e-12);
    }
    Ok(())
}

#[test]
fn derivative_array_rejects_underdetermined_and_inconsistent_algebraic_loops() {
    // An algebraic identity cannot determine its output. With a nonzero
    // input slope the identity becomes a contradictory physical equation.
    // Neither problem may be resolved by inventing a zero derivative gauge.
    for source in ["0", "PWL(0 0 1m 1)"] {
        let body = format!("v1 in 0 {source}\nb1 out 0 v=v(out)+ddt(v(in))\nr1 out 0 1k");
        let error = run(
            &body,
            AnalysisKind::Transient,
            &["1u", "25u", "0", "1u", "uic", "method=gear"],
        )
        .expect_err("the physical equations do not determine a solution")
        .to_string();
        assert!(
            error.contains("noncausal") || error.contains("underdetermined"),
            "{error}"
        );
    }
}

#[test]
fn derivative_array_distinguishes_a_repeating_jump_and_a_storage_impulse() {
    for method in ["method=trap", "method=gear"] {
        let jump = run(
            "v1 in 0 PWL(0 0 10u 1) r=0\nb1 out 0 v=1u*ddt(v(in))\nr1 out 0 1k",
            AnalysisKind::Transient,
            &["1u", "25u", "0", "1u", "uic", method],
        )
        .expect_err("a repeating PWL voltage jump has no classical derivative")
        .to_string();
        assert!(
            jump.contains("source event") && jump.contains("derivative input jumps"),
            "{jump}"
        );

        let impulse = run(
            "v1 in 0 PWL(0 0 10u 1 20u 0)\nb1 out 0 v=1u*ddt(v(in))\nc1 out 0 1n ic=0.1\nr1 out 0 1k",
            AnalysisKind::Transient,
            &["1u", "25u", "0", "1u", "uic", method],
        )
        .expect_err("the derivative voltage step cannot preserve capacitor voltage")
        .to_string();
        assert!(
            impulse.contains("c1") && impulse.contains("impulse"),
            "{impulse}"
        );
    }
}

#[test]
fn smooth_right_limit_recovers_from_singular_uic_taylor_seed() -> SpiceResult<()> {
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            "v1 in 0 PWL(0 4 1m 5)\nb1 out 0 v=1m*ddt(sqrt(v(in)))\nr1 out 0 1k",
            AnalysisKind::Transient,
            &["1u", "25u", "0", "1u", "uic", method],
        )?;
        for point in 0..plot.point_count() {
            let t = plot.value("time", point).unwrap().re;
            close(
                plot.value("v(out)", point).unwrap().re,
                0.5 / (1000. * t + 4.).sqrt(),
                1e-10,
            );
        }
    }
    Ok(())
}

#[test]
fn prepared_power_laws_preserve_distinct_physical_derivatives_and_kcl() -> SpiceResult<()> {
    let body = "v1 in 0 PWL(0 -2 1m -1)\nv2 exponent 0 PWL(0 2 1m 3)\nv3 negative 0 -2\nb1 absolute 0 v=1m*ddt(simetrix_abs_pow(v(in),3))\nb2 caret 0 v=1m*ddt(simetrix_caret_pow(v(in),3))\nb3 realpart 0 v=1m*ddt(simetrix_pow(v(negative),v(exponent)))\nb4 second 0 v=1m*ddt(v(realpart))\nr1 absolute 0 1k\nr2 caret 0 1k\nr3 realpart 0 1k\nr4 second 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["2u", "80u", "0", "2u", "uic", method],
        )?;
        assert!(plot.point_count() > 20);
        for index in 0..plot.point_count() {
            let u = 1000. * plot.value("time", index).unwrap().re;
            let x = 2. - u;
            let p = 2. + u;
            let pi = std::f64::consts::PI;
            let log = 2_f64.ln();
            let (sin, cos) = (pi * u).sin_cos();
            let scale = 2_f64.powf(p);
            let first = scale * log.mul_add(cos, -pi * sin);
            let second = scale
                * log
                    .mul_add(log, -pi * pi)
                    .mul_add(cos, -2. * pi * log * sin);
            for ((node, branch), expected) in [
                ("v(absolute)", "i(b1)"),
                ("v(caret)", "i(b2)"),
                ("v(realpart)", "i(b3)"),
                ("v(second)", "i(b4)"),
            ]
            .into_iter()
            .zip([-3. * x * x, 3. * x * x, first, second])
            {
                close(plot.value(node, index).unwrap().re, expected, 2e-8);
                close(
                    plot.value(branch, index).unwrap().re,
                    -expected * 1e-3,
                    2e-11,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn delayed_physical_rank_emerges_at_startup_and_interior_events() -> SpiceResult<()> {
    // g(t) is zero through many Taylor degrees, but g(t)*(x-f(t))=0
    // still determines the entire analytic germ. In particular, a
    // temporary rank plateau must not choose arbitrary derived storage.
    for order in [2, 12, 48] {
        for (center, event) in [("0", false), ("0.5u", true)] {
            let body = format!(
                "vtrigger trigger 0 PWL(0 0 0.5u 1 1u 0)\nb1 x 0 i=((time-{center})/1u)^{order}*(v(x)-2-time/1u-(time/1u)^2)\nb2 first 0 v=1u*ddt(v(x))\nb3 second 0 v=1u*ddt(v(first))\nr1 first 0 1k\nr2 second 0 1k\n.options reltol=1e-8 vntol=1n"
            );
            for method in ["method=trap", "method=gear"] {
                let plot = run(
                    &body,
                    AnalysisKind::Transient,
                    &["0.1u", "1u", "0", "0.1u", "uic", method],
                )?;
                assert!(plot.point_count() > 20);
                let mut landed = false;
                for index in 0..plot.point_count() {
                    let time = plot.value("time", index).unwrap().re;
                    landed |= time.to_bits() == 0.5e-6_f64.to_bits();
                    let u = time * 1e6;
                    let first = 2_f64.mul_add(u, 1.);
                    close(
                        plot.value("v(x)", index).unwrap().re,
                        u.mul_add(u, 2. + u),
                        1e-12,
                    );
                    close(plot.value("v(first)", index).unwrap().re, first, 1e-11);
                    close(plot.value("v(second)", index).unwrap().re, 2., 2e-11);
                    close(plot.value("i(b2)", index).unwrap().re, -first * 1e-3, 1e-14);
                    close(plot.value("i(b3)", index).unwrap().re, -2e-3, 2e-14);
                }
                if event {
                    assert!(landed, "the interior rank collapse was not sampled");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn wide_parallel_derivative_constraints_do_not_lose_physical_rank() -> SpiceResult<()> {
    use std::fmt::Write as _;
    // Every output is independently determined by the same source slope.
    // Repeated elimination through the shared input must not manufacture
    // uncertainty and discard a physically independent output.
    for branches in [128, 256] {
        let mut body = String::from("v1 in 0 PWL(0 2 1m 3)\n.options reltol=1e-8 vntol=1n\n");
        for index in 1..=branches {
            writeln!(
                body,
                "b{index} out{index} 0 v=1m*ddt(v(in))\nr{index} out{index} 0 1k"
            )
            .unwrap();
        }
        for method in ["method=trap", "method=gear"] {
            let plot = run(
                &body,
                AnalysisKind::Transient,
                &["1u", "2u", "0", "1u", "uic", method],
            )?;
            for branch in 1..=branches {
                let voltage = format!("v(out{branch})");
                let current = format!("i(b{branch})");
                for point in 0..plot.point_count() {
                    close(plot.value(&voltage, point).unwrap().re, 1., 1e-10);
                    close(plot.value(&current, point).unwrap().re, -1e-3, 1e-13);
                }
            }
        }
    }
    Ok(())
}

#[test]
fn behavioural_step_inside_derivative_requires_impulse() {
    for method in ["method=trap", "method=gear"] {
        for input in [
            "u(time-10u)",
            "floor(time/10u)",
            "sgn(time-10u)",
            "(time<10u ? 0 : 1)",
            "u((time-10.2u)*(time-10.25u))",
            "1u*ddt(abs(time/1u-10))",
        ] {
            let body =
                format!("b1 out 0 v=1u*ddt({input})\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n");
            let error = run(
                &body,
                AnalysisKind::Transient,
                &["1u", "25u", "0", "1u", "uic", method],
            )
            .expect_err("a jump in a derived storage coordinate requires an impulse")
            .to_string();
            assert!(
                error.contains("derivative input jumps"),
                "{input} ({method}): {error}"
            );
        }
    }
}

#[test]
fn compiled_behavioural_corner_uses_right_germ_and_preserves_storage() -> SpiceResult<()> {
    for method in ["method=trap", "method=gear"] {
        let body = "bdrive x 0 v=(time-1u)/1u\nrdrive x 0 1\nbv out 0 v=1u*ddt(abs(v(x)))\nbi other 0 v=1u*ddt(abs(i(bdrive)))\nrout out 0 1k\nrother other 0 1k\nrcap cap 0 1k\nccap cap 0 1n ic=1\nvli lin 0 0\nrli lin li 1\nlli li 0 1u ic=1\nvrational ri 0 dc 3\na1 ri rational mdl\nrrational rational 0 1k\n.model mdl s_xfer(num_coeff=[1] den_coeff=[1 1] denormalized_freq=1e6 int_ic=[2])\n.options reltol=1e-8 vntol=1n chgtol=1e-18 trtol=0.01";
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["1u", "3u", "0", "1u", "uic", method],
        )?;
        let mut seen = false;
        for index in 0..plot.point_count() {
            let t = plot.value("time", index).unwrap().re;
            let at_corner = (t - 1e-6).abs() <= 1e-18;
            seen |= at_corner;
            let expected = if t < 1e-6 && !at_corner { -1. } else { 1. };
            for name in ["v(out)", "v(other)"] {
                close(plot.value(name, index).unwrap().re, expected, 1e-7);
            }
            let decay = (-t / 1e-6).exp();
            close(plot.value("v(cap)", index).unwrap().re, decay, 2e-6);
            close(plot.value("i(lli)", index).unwrap().re, decay, 2e-6);
            close(
                plot.value("v(rational)", index).unwrap().re,
                3. - decay,
                2e-6,
            );
        }
        assert!(seen, "missing localized behavioural event");
    }
    Ok(())
}

#[test]
fn behavioural_lazy_branch_and_boundary_observation_use_physical_value() -> SpiceResult<()> {
    // The isolated scalar equality branch is not part of either
    // one-sided physical germ. Evaluating it would take sqrt(-1).
    let body = "bguard x 0 i=(time==10u ? sqrt(-1) : abs(time/1u-10))\nrguard x 0 1\nb1 out 0 v=1u*ddt(-v(x))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run_observed(
            body,
            AnalysisKind::Transient,
            &["1u", "15u", "0", "1u", "uic", method],
            &["@bguard[i]"],
        )?;
        let mut seen = false;
        for index in 0..plot.point_count() {
            let t = plot.value("time", index).unwrap().re;
            let at_corner = (t - 10e-6).abs() <= 1e-18;
            seen |= at_corner;
            close(
                plot.value("@bguard[i]", index).unwrap().re,
                (t / 1e-6 - 10.).abs(),
                1e-7,
            );
            close(
                plot.value("v(out)", index).unwrap().re,
                if t < 10e-6 && !at_corner { -1. } else { 1. },
                1e-7,
            );
        }
        assert!(seen);
    }
    Ok(())
}

#[test]
fn behavioural_two_corners_within_one_requested_step() -> SpiceResult<()> {
    let body = "b1 out 0 v=1u*ddt(max(0,min((time-10.2u)/1u,(10.3u-time)/1u)))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["1u", "15u", "0", "1u", "uic", method],
        )?;
        let mut seen = [false; 3];
        for index in 0..plot.point_count() {
            let t = plot.value("time", index).unwrap().re;
            let boundaries = [10.2e-6, 10.25e-6, 10.3e-6];
            for (at, flag) in boundaries.into_iter().zip(&mut seen) {
                *flag |= (t - at).abs() <= 1e-18;
            }
            // Ignore a few ulps around knots in this slope comparison;
            // the preceding assertion still demands all three landings.
            if boundaries.into_iter().any(|at| (t - at).abs() <= 1e-18) {
                continue;
            }
            let expected = if (boundaries[0]..boundaries[1]).contains(&t) {
                1.
            } else if (boundaries[1]..boundaries[2]).contains(&t) {
                -1.
            } else {
                0.
            };
            close(plot.value("v(out)", index).unwrap().re, expected, 1e-7);
        }
        assert_eq!(seen, [true; 3]);
    }
    Ok(())
}

#[test]
fn behavioural_adjacent_representable_events_are_not_coalesced() -> SpiceResult<()> {
    let a = 2_f64.powi(-16);
    let b = a.next_up();
    let body = format!(
        "va ta 0 {a:.17e}\nvb tb 0 {b:.17e}\nb1 out 0 v=ddt(uramp(time-v(ta))+uramp(time-v(tb)))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n"
    );
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            &body,
            AnalysisKind::Transient,
            &["1u", "20u", "0", "1u", "uic", method],
        )?;
        let mut seen = [false; 2];
        let mut previous = 0.;
        for index in 0..plot.point_count() {
            let t = plot.value("time", index).unwrap().re;
            assert!(t > previous, "duplicate or reversed event at {t:e}");
            previous = t;
            assert_eq!(
                plot.value("v(ta)", index).unwrap().re.to_bits(),
                a.to_bits()
            );
            assert_eq!(
                plot.value("v(tb)", index).unwrap().re.to_bits(),
                b.to_bits()
            );
            seen[0] |= t.to_bits() == a.to_bits();
            seen[1] |= t.to_bits() == b.to_bits();
            let expected = if t < a {
                0.
            } else if t < b {
                1.
            } else {
                2.
            };
            close(plot.value("v(out)", index).unwrap().re, expected, 1e-9);
        }
        assert_eq!(seen, [true, true], "{method}");
    }
    Ok(())
}

#[test]
fn behavioural_state_dependent_event_follows_physical_decay() -> SpiceResult<()> {
    let body = "rcap cap 0 1k\nccap cap 0 1n ic=1\nb1 out 0 v=1u*ddt(abs(v(cap)-0.5))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n chgtol=1e-18 trtol=0.01";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["0.1u", "2u", "0", "0.1u", "uic", method],
        )?;
        let mut crossing = None;
        let mut previous = -1.;
        for index in 0..plot.point_count() {
            let t = plot.value("time", index).unwrap().re;
            let v = plot.value("v(cap)", index).unwrap().re;
            let rate = plot.value("v(out)", index).unwrap().re;
            close(v, (-t / 1e-6).exp(), 2e-6);
            close(rate.abs(), v, 1e-7);
            if rate > 0. && previous < 0. {
                crossing = Some(t);
                close(v, 0.5, 1e-8);
            }
            previous = rate;
        }
        close(
            crossing.expect("missing state-triggered event"),
            1e-6 * 2_f64.ln(),
            5e-12,
        );
    }
    Ok(())
}

#[test]
fn behavioural_piecewise_and_clipped_derivatives_localize_knots() -> SpiceResult<()> {
    let body = "bup up 0 v=1u*ddt(pwl(time/1u, 0, 0, 1, 1, 2, 3))\nbdown down 0 v=1u*ddt(pwl(time/1u, 2, 3, 1, 1, 0, 0))\nbclip clip 0 v=1u*ddt(u2(time/1u-1))\nrup up 0 1k\nrdown down 0 1k\nrclip clip 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["1u", "3u", "0", "1u", "uic", method],
        )?;
        let mut seen = [false; 2];
        for index in 0..plot.point_count() {
            let t = plot.value("time", index).unwrap().re;
            for (at, flag) in [1e-6, 2e-6].into_iter().zip(&mut seen) {
                *flag |= (t - at).abs() < 1e-18;
            }
            if [1e-6, 2e-6].iter().any(|at| (t - at).abs() < 1e-18) {
                continue;
            }
            let slope = if t < 1e-6 { 1. } else { 2. };
            for name in ["v(up)", "v(down)"] {
                close(plot.value(name, index).unwrap().re, slope, 1e-7);
            }
            close(
                plot.value("v(clip)", index).unwrap().re,
                if (1e-6..2e-6).contains(&t) { 1. } else { 0. },
                1e-7,
            );
        }
        assert_eq!(seen, [true, true]);
    }
    Ok(())
}

#[test]
fn behavioural_smoothed_table_second_derivative_localizes_edges() -> SpiceResult<()> {
    let body = "bdrive in 0 v=time/1u\ne1 x 0 table {v(in)} = (0,0) (1,2) (2,3)\nb1 out 0 v=1u*ddt(1u*ddt(v(x)))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["1u", "3u", "0", "1u", "uic", method],
        )?;
        let mut seen = [false; 5];
        let boundaries = [0.1e-6, 0.9e-6, 1.1e-6, 1.9e-6, 2.1e-6];
        for index in 0..plot.point_count() {
            let t = plot.value("time", index).unwrap().re;
            for (at, flag) in boundaries.into_iter().zip(&mut seen) {
                *flag |= (t - at).abs() < 1e-18;
            }
            if boundaries.iter().any(|at| (t - at).abs() < 1e-18) {
                continue;
            }
            let expected = if t < 0.1e-6 {
                10.
            } else if (0.9e-6..1.1e-6).contains(&t) {
                -5.
            } else if (1.9e-6..2.1e-6).contains(&t) {
                -5.
            } else {
                0.
            };
            close(plot.value("v(out)", index).unwrap().re, expected, 1e-6);
        }
        assert_eq!(seen, [true; 5]);
    }
    Ok(())
}

#[test]
fn singular_provisional_tangent_recovers_regular_nonlinear_root() -> SpiceResult<()> {
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            "b1 x 0 i=v(x)^3-(2+time/1u)^3\nb2 out 0 v=1u*ddt(v(x))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n",
            AnalysisKind::Transient,
            &["0.1u", "1u", "0", "0.1u", "uic", method],
        )?;
        for i in 0..plot.point_count() {
            let u = plot.value("time", i).unwrap().re / 1e-6;
            close(plot.value("v(x)", i).unwrap().re, 2. + u, 1e-10);
            close(plot.value("v(out)", i).unwrap().re, 1., 1e-9);
            close(plot.value("i(b2)", i).unwrap().re, -1e-3, 1e-12);
        }
    }
    Ok(())
}

#[test]
fn singular_nonlinear_germ_recovers_unique_cubic_rate() -> SpiceResult<()> {
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            "b1 x 0 i=v(x)^3-(time/1u)^3\nb2 out 0 v=1u*ddt(v(x))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n",
            AnalysisKind::Transient,
            &["0.1u", "1u", "0", "0.1u", "uic", method],
        )?;
        for i in 0..plot.point_count() {
            let u = plot.value("time", i).unwrap().re / 1e-6;
            close(plot.value("v(x)", i).unwrap().re, u, 1e-10);
            close(plot.value("v(out)", i).unwrap().re, 1., 1e-9);
            close(plot.value("i(b2)", i).unwrap().re, -1e-3, 1e-12);
        }
    }
    Ok(())
}

#[test]
fn singular_quadratic_germ_reports_distinct_branches() {
    for method in ["method=trap", "method=gear"] {
        let error = run(
            "b1 x 0 i=v(x)^2-(time/1u)^2\nb2 out 0 v=1u*ddt(v(x))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n",
            AnalysisKind::Transient,
            &["0.1u", "1u", "0", "0.1u", "uic", method],
        ).expect_err("two distinct right-hand slopes must not be chosen by a numerical gauge");
        assert!(
            error.to_string().contains("ambiguous nonlinear DAE"),
            "{error}"
        );
    }
}

#[test]
fn singular_germ_recovery_uses_coupled_nonpolynomial_equations() -> SpiceResult<()> {
    // The two nonlinear equations have a singular Jacobian at the zero
    // seed, but each real solution of sinh(z)-z=a is unique. The actual
    // variables are recovered through their coupled linear combinations.
    let body = "b1 x 0 i=sinh(v(x)+v(y))-(v(x)+v(y))-(sinh(3+time/1u)-(3+time/1u))\nb2 y 0 i=sinh(v(x)-v(y))-(v(x)-v(y))-(sinh(1+3*time/1u)-(1+3*time/1u))\nb3 dx 0 v=1u*ddt(v(x))\nb4 dy 0 v=1u*ddt(v(y))\nr1 dx 0 1k\nr2 dy 0 2k\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["0.1u", "0.5u", "0", "0.1u", "uic", method],
        )?;
        for i in 0..plot.point_count() {
            let u = plot.value("time", i).unwrap().re / 1e-6;
            close(plot.value("v(x)", i).unwrap().re, 2. + 2. * u, 1e-9);
            close(plot.value("v(y)", i).unwrap().re, 1. - u, 1e-9);
            close(plot.value("v(dx)", i).unwrap().re, 2., 1e-9);
            close(plot.value("v(dy)", i).unwrap().re, -1., 1e-9);
            close(plot.value("i(b3)", i).unwrap().re, -2e-3, 1e-12);
            close(plot.value("i(b4)", i).unwrap().re, 0.5e-3, 1e-12);
        }
    }
    Ok(())
}

#[test]
fn singular_germ_recovery_at_exact_source_event_preserves_storage() -> SpiceResult<()> {
    let body = "vtrigger trigger 0 PWL(0 0 0.5u 1 1u 0)\nb1 x 0 i=v(x)^3-((time-0.5u)/1u)^3\nb2 out 0 v=1u*ddt(v(x))\nr1 out 0 1k\nc1 stored 0 1u ic=2\nr2 stored 0 1k\n.ic v(x)=-0.5\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["0.1u", "1u", "0", "0.1u", "uic", method],
        )?;
        let mut event = false;
        for i in 0..plot.point_count() {
            let t = plot.value("time", i).unwrap().re;
            event |= t.to_bits() == 0.5e-6_f64.to_bits();
            close(plot.value("v(x)", i).unwrap().re, (t - 0.5e-6) / 1e-6, 1e-9);
            close(plot.value("v(out)", i).unwrap().re, 1., 1e-8);
            close(plot.value("i(b2)", i).unwrap().re, -1e-3, 1e-11);
            close(
                plot.value("v(stored)", i).unwrap().re,
                2. * (-t / 1e-3).exp(),
                1e-7,
            );
        }
        assert!(event, "the physical singular event was not sampled");
    }
    Ok(())
}

#[test]
fn singular_moving_repeated_root_recovers_consistent_germ() -> SpiceResult<()> {
    // A repeated root can have a unique moving germ despite a vanishing
    // instantaneous tangent. Verify the actual recovered trajectory.
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            "b1 x 0 i=(v(x)-time/1u)^2\nb2 out 0 v=1u*ddt(v(x))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n",
            AnalysisKind::Transient,
            &["0.1u", "1u", "0", "0.1u", "uic", method],
        )?;
        for i in 0..plot.point_count() {
            let u = plot.value("time", i).unwrap().re / 1e-6;
            close(plot.value("v(x)", i).unwrap().re, u, 1e-10);
            close(plot.value("v(out)", i).unwrap().re, 1., 1e-9);
            close(plot.value("i(b2)", i).unwrap().re, -1e-3, 1e-12);
        }
    }
    Ok(())
}

#[test]
fn singular_homogeneous_repeated_root_is_not_arbitrarily_gauged() {
    let error = run(
        "b1 x 0 i=v(x)^2\nb2 out 0 v=1u*ddt(v(x))\nr1 out 0 1k",
        AnalysisKind::Transient,
        &["0.1u", "1u", "0", "0.1u", "uic", "method=gear"],
    )
    .expect_err("a zero residual and unresolved tangent must not establish rank");
    assert!(
        error.to_string().contains("singular") || error.to_string().contains("underdetermined"),
        "{error}"
    );
}

#[test]
fn singular_germ_recovery_handles_high_order_and_scaled_offsets() -> SpiceResult<()> {
    for (power, offset, h) in [
        (3, 0.001, 1e-6),
        (15, 0.001, 1e-6),
        (15, 0., 1e-9),
        (15, 0., 1e-3),
        (15, 1000., 1e-6),
    ] {
        let body = format!(
            "b1 x 0 i=v(x)^{power}-({offset:e}+time/{h:e})^{power}\nb2 out 0 v={h:e}*ddt(v(x))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n"
        );
        let step = format!("{:e}", h / 10.);
        let stop = format!("{h:e}");
        for method in ["method=trap", "method=gear"] {
            let plot = run(
                &body,
                AnalysisKind::Transient,
                &[&step, &stop, "0", &step, "uic", method],
            )?;
            for i in 0..plot.point_count() {
                let u = plot.value("time", i).unwrap().re / h;
                close(plot.value("v(x)", i).unwrap().re, offset + u, 1e-9);
                close(plot.value("v(out)", i).unwrap().re, 1., 1e-9);
                close(plot.value("i(b2)", i).unwrap().re, -1e-3, 1e-12);
            }
        }
    }
    Ok(())
}

#[test]
fn singular_germ_recovery_respects_explicit_branch_selection() -> SpiceResult<()> {
    for sign in [-1., 1.] {
        let body = format!(
            "b1 x 0 i=v(x)^2-(2+time/1u)^2\nb2 out 0 v=1u*ddt(v(x))\nr1 out 0 1k\n.ic v(x)={}\n.options reltol=1e-8 vntol=1n",
            2. * sign
        );
        for method in ["method=trap", "method=gear"] {
            let plot = run(
                &body,
                AnalysisKind::Transient,
                &["0.1u", "1u", "0", "0.1u", "uic", method],
            )?;
            for i in 0..plot.point_count() {
                let u = plot.value("time", i).unwrap().re / 1e-6;
                close(plot.value("v(x)", i).unwrap().re, sign * (2. + u), 1e-10);
                close(plot.value("v(out)", i).unwrap().re, sign, 1e-9);
                close(plot.value("i(b2)", i).unwrap().re, -sign * 1e-3, 1e-12);
            }
        }
    }
    Ok(())
}

#[test]
fn singular_germ_recovery_rejects_ambiguous_curvature_and_nonanalytic_roots() {
    for (law, diagnostic) in [
        ("v(x)^2-(2+time/1u)^2", "ambiguous nonlinear DAE"),
        ("v(x)^2-(time/1u)^4", "ambiguous nonlinear DAE"),
        ("v(x)^2-time/1u", "could not resolve a singular tangent"),
        ("v(x)^2+(time/1u)^2", "could not resolve a singular tangent"),
    ] {
        let body = format!(
            "b1 x 0 i={law}\nb2 out 0 v=1u*ddt(v(x))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n"
        );
        for method in ["method=trap", "method=gear"] {
            let error = run(
                &body,
                AnalysisKind::Transient,
                &["0.1u", "1u", "0", "0.1u", "uic", method],
            )
            .expect_err("an unresolved or ambiguous germ must not be assigned an arbitrary rate");
            assert!(error.to_string().contains(diagnostic), "{law}: {error}");
        }
    }
}

#[test]
fn singular_germ_predictor_survives_rejected_steps_and_source_event() -> SpiceResult<()> {
    for method in ["trap", "gear"] {
        let body = format!(
            "predictor transaction\nvtrigger trigger 0 PWL(0 0 0.5u 1 1u 0)\nb1 x 0 i=v(x)^3-((time-0.5u)/1u+0.25*((time-0.5u)/1u)^3)^3\nb2 out 0 v=1u*ddt(v(x))\nr1 out 0 1k\nc1 stored 0 1n ic=1\nr2 stored 0 200\n.options method={method} reltol=1e-8 vntol=1n chgtol=1e-18 trtol=0.1\n.tran 0.2u 1u 0 0.5u uic\n.end"
        );
        let netlist = Parser::new().parse_deck(&parse_deck_text(
            Path::new("predictor-transaction.cir"),
            &body,
        ))?;
        let config = RunConfig::from_netlist(&netlist)?;
        let plan = config.plan_for(&netlist.analyses[0])?;
        let mut circuit = config.circuit(&netlist)?;
        let (plot, stats) =
            ngspice_rs::analysis::companion_transient(&mut circuit, &plan, &config.context())?;
        assert!(stats.rejected > 0, "{method}: no rejected trial");
        let mut event = false;
        for i in 0..plot.point_count() {
            let time = plot.value("time", i).unwrap().re;
            event |= time.to_bits() == 0.5e-6_f64.to_bits();
            let u = (time - 0.5e-6) / 1e-6;
            let x = 0.25_f64.mul_add(u.powi(3), u);
            let rate = 0.75_f64.mul_add(u * u, 1.);
            close(plot.value("v(x)", i).unwrap().re, x, 1e-8);
            close(plot.value("v(out)", i).unwrap().re, rate, 1e-7);
            close(plot.value("i(b2)", i).unwrap().re, -rate * 1e-3, 1e-10);
            close(
                plot.value("v(stored)", i).unwrap().re,
                (-time / 0.2e-6).exp(),
                2e-5,
            );
        }
        assert!(event, "{method}: missing exact singular event");
    }
    Ok(())
}

#[test]
fn singular_germ_and_behavioural_corner_share_the_exact_event() -> SpiceResult<()> {
    let body = "b1 x 0 i=v(x)^3-((time-0.5u)/1u)^3\nb2 out 0 v=1u*ddt(abs(v(x)))\nr1 out 0 1k\nc1 stored 0 1u ic=2\nr2 stored 0 1k\n.ic v(x)=-0.5\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let plot = run(
            body,
            AnalysisKind::Transient,
            &["0.1u", "1u", "0", "0.1u", "uic", method],
        )?;
        let mut crossing = None;
        let mut previous = -1.;
        for index in 0..plot.point_count() {
            let time = plot.value("time", index).unwrap().re;
            let x = plot.value("v(x)", index).unwrap().re;
            let rate = plot.value("v(out)", index).unwrap().re;
            close(x, (time - 0.5e-6) / 1e-6, 1e-8);
            close(rate.abs(), 1., 1e-7);
            if (time - 0.5e-6).abs() > 1e-14 {
                close(rate, (time - 0.5e-6).signum(), 1e-7);
            }
            close(plot.value("i(b2)", index).unwrap().re, -rate * 1e-3, 1e-10);
            close(
                plot.value("v(stored)", index).unwrap().re,
                2. * (-time / 1e-3).exp(),
                2e-6,
            );
            if rate > 0. && previous < 0. {
                assert!(crossing.is_none());
                crossing = Some(time);
                close(x, 0., 1e-8);
            }
            previous = rate;
        }
        close(crossing.expect("missing behavioural corner"), 0.5e-6, 1e-14);
    }
    Ok(())
}

#[test]
fn singular_germ_recovery_cannot_accept_discontinuous_derivative_input() {
    let body = "b1 x 0 i=v(x)^3-((time-0.5u)/1u)^3\nb2 out 0 v=1u*ddt(u(v(x)))\nr1 out 0 1k\n.ic v(x)=-0.5\n.options reltol=1e-8 vntol=1n";
    for method in ["method=trap", "method=gear"] {
        let error = run(
            body,
            AnalysisKind::Transient,
            &["0.1u", "1u", "0", "0.1u", "uic", method],
        )
        .expect_err("the recovered singular germ cannot erase a physical impulse")
        .to_string();
        assert!(
            error.contains("derivative input jumps"),
            "{method}: {error}"
        );
    }
}

#[test]
fn behavioural_event_preserves_independent_semiconductor_charge_histories() -> SpiceResult<()> {
    // At the interior corner the right-limit candidate must retain both
    // the selected behavioural value and each diode's charge/rate pair.
    // Two diodes share an equation row, but have different physical laws.
    let body = "vdrive in 0 PWL(0 -0.4 1m 0.6)\nd1 in 0 dm1\nd2 in 0 dm2\n.model dm1 d(is=1e-12 n=10 cjo=20p vj=.8 m=.4 tt=5n)\n.model dm2 d(is=2e-12 n=10 cjo=70p vj=.7 m=.3 tt=9n)\nbguard x 0 i=(time==10u ? sqrt(-1) : abs(time/1u-10))\nrguard x 0 1\nb1 out 0 v=1u*ddt(-v(x))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    let thermal = 10. * (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    for method in ["method=trap", "method=gear"] {
        let plot = run_observed(
            body,
            AnalysisKind::Transient,
            &["1u", "15u", "0", "1u", "uic", method],
            &["@bguard[i]", "@d1[i]", "@d2[i]"],
        )?;
        let mut seen = false;
        for index in 0..plot.point_count() {
            let at = |name: &str| plot.value(name, index).unwrap().re;
            let time = at("time");
            let voltage = 1000_f64.mul_add(time, -0.4);
            let corner = (time - 10e-6).abs() <= 1e-18;
            seen |= corner;
            close(at("@bguard[i]"), (time / 1e-6 - 10.).abs(), 1e-7);
            close(
                at("v(out)"),
                if time < 10e-6 && !corner { -1. } else { 1. },
                1e-7,
            );
            let diode = |sat: f64, cap: f64, potential: f64, grading: f64, transit: f64| {
                let current = sat * (voltage / thermal).exp_m1() + 1e-12 * voltage;
                let conductance = sat / thermal * (voltage / thermal).exp() + 1e-12;
                current
                    + 1000.
                        * (cap * (1. - voltage / potential).powf(-grading) + transit * conductance)
            };
            let first = diode(1e-12, 20e-12, 0.8, 0.4, 5e-9);
            let second = diode(2e-12, 70e-12, 0.7, 0.3, 9e-9);
            close(at("@d1[i]"), first, 1e-15);
            close(at("@d2[i]"), second, 1e-15);
            close(at("i(vdrive)"), -(first + second), 1e-15);
        }
        assert!(seen, "missing semiconductor-loaded branch event");
    }
    Ok(())
}

#[test]
fn singular_germ_recovers_distinct_semiconductor_histories() -> SpiceResult<()> {
    // The nonlinear singular coordinate drives both junctions through a
    // voltage law. The local physical charge constraints must be retained
    // across a coincident behavioural corner, including their separate
    // terminal-current observations and their sum in source KCL.
    let body = "b1 x 0 i=v(x)^3-((time-0.5u)/1u)^3\nb2 out 0 v=1u*ddt(abs(v(x)))\nr1 out 0 1k\nbdrive in 0 v=0.1*v(x)-0.4\nd1 in 0 dm1\nd2 in 0 dm2\n.model dm1 d(is=1e-12 n=10 cjo=20p vj=.8 m=.4 tt=5n)\n.model dm2 d(is=2e-12 n=10 cjo=70p vj=.7 m=.3 tt=9n)\n.ic v(x)=-0.5\n.options reltol=1e-8 vntol=1n";
    let thermal = 10. * (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    for method in ["method=trap", "method=gear"] {
        let plot = run_observed(
            body,
            AnalysisKind::Transient,
            &["0.1u", "1u", "0", "0.1u", "uic", method],
            &["@d1[i]", "@d2[i]"],
        )?;
        let mut previous = -1.;
        let mut event = None;
        for index in 0..plot.point_count() {
            let at = |name: &str| plot.value(name, index).unwrap().re;
            let time = at("time");
            let x = (time - 0.5e-6) / 1e-6;
            let voltage = 0.1_f64.mul_add(x, -0.4);
            let rate = at("v(out)");
            close(at("v(x)"), x, 1e-8);
            close(at("v(in)"), voltage, 1e-9);
            close(rate.abs(), 1., 1e-7);
            if (time - 0.5e-6).abs() > 1e-14 {
                close(rate, x.signum(), 1e-7);
            }
            close(at("i(b2)"), -rate * 1e-3, 1e-10);
            let diode = |sat: f64, cap: f64, potential: f64, grading: f64, transit: f64| {
                let current = sat * (voltage / thermal).exp_m1() + 1e-12 * voltage;
                let conductance = sat / thermal * (voltage / thermal).exp() + 1e-12;
                current
                    + 1e5
                        * (cap * (1. - voltage / potential).powf(-grading) + transit * conductance)
            };
            let first = diode(1e-12, 20e-12, 0.8, 0.4, 5e-9);
            let second = diode(2e-12, 70e-12, 0.7, 0.3, 9e-9);
            close(at("@d1[i]"), first, 1e-13);
            close(at("@d2[i]"), second, 1e-13);
            close(at("i(bdrive)"), -(first + second), 1e-13);
            if rate > 0. && previous < 0. {
                assert!(event.is_none());
                event = Some(time);
            }
            previous = rate;
        }
        close(
            event.expect("missing singular semiconductor event"),
            0.5e-6,
            1e-14,
        );
    }
    Ok(())
}

#[test]
fn diode_constitutive_crossings_land_and_observe_charge_current() -> SpiceResult<()> {
    // All four constitutive crossings are interior to the source segments.
    // The derivative input stays continuous; only the source slope reverses.
    let body = "vdrive in 0 PWL(0 -1 4u 1 8u -1)\nd1 in 0 dm\n.model dm d(is=1e-12 n=10 cjo=20p vj=.8 m=.4 fc=.5 tlevc=1)\nb1 out 0 v=1u*ddt(v(in))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    let thermal = 10. * (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    let reverse = (1. - 3. * thermal) / 5e5;
    let boundaries = [reverse, 2.8e-6, 5.2e-6, 8e-6 - reverse];
    for method in ["method=trap", "method=gear"] {
        let plot = run_observed(
            body,
            AnalysisKind::Transient,
            &["1u", "8u", "0", "1u", "uic", method],
            &["@d1[i]", "@d1[p]"],
        )?;
        let mut seen = [false; 4];
        for index in 0..plot.point_count() {
            let at = |name: &str| plot.value(name, index).unwrap().re;
            let time = at("time");
            for (boundary, flag) in boundaries.into_iter().zip(&mut seen) {
                *flag |= (time - boundary).abs() <= 1e-17;
            }
            let voltage = at("v(in)");
            let current = if voltage < -3. * thermal {
                -1e-12 * (1. + (3. * thermal / (voltage * std::f64::consts::E)).powi(3))
            } else {
                1e-12 * (voltage / thermal).exp_m1()
            };
            let cap = if voltage < 0.4 {
                20e-12 * (1. - voltage / 0.8).powf(-0.4)
            } else {
                20e-12 / 0.5_f64.powf(1.4) * 0.5_f64.mul_add(voltage - 0.4, 0.5)
            };
            let rate = at("v(out)") / 1e-6;
            let expected = rate.mul_add(cap, current + 1e-12 * voltage);
            close(at("@d1[i]"), expected, 1e-13);
            close(at("i(vdrive)"), -expected, 1e-13);
            close(at("@d1[p]"), voltage * expected, 1e-13);
        }
        assert_eq!(seen, [true; 4], "{method}");
    }
    Ok(())
}

#[test]
fn jfet_channel_crossings_land_in_both_directions() -> SpiceResult<()> {
    // VGS=-1.5, VTO=-2. The direction changes at VDS=0 and the
    // normal channel saturates at VDS=0.5. Both are crossed twice.
    let body = "vd d 0 PWL(0 -1 5u 1.5 10u -1)\nvg g 0 -1.5\nj1 d g 0 jm\n.model jm njf(vto=-2 beta=1m lambda=0 b=1 is=1e-15 n=10)\nb1 out 0 v=1u*ddt(v(d))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    let boundaries = [2e-6, 3e-6, 7e-6, 8e-6];
    let thermal = 10. * (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    for method in ["method=trap", "method=gear"] {
        let plot = run_observed(
            body,
            AnalysisKind::Transient,
            &["1u", "10u", "0", "1u", "uic", method],
            &["@j1[i]"],
        )?;
        let mut seen = [false; 4];
        for index in 0..plot.point_count() {
            let at = |name: &str| plot.value(name, index).unwrap().re;
            let time = at("time");
            for (boundary, flag) in boundaries.into_iter().zip(&mut seen) {
                *flag |= (time - boundary).abs() <= 1e-17;
            }
            let drain = at("v(d)");
            let vgd = -1.5 - drain;
            let gate_current = if vgd < -3. * thermal {
                -1e-15 * (1. + (3. * thermal / (vgd * std::f64::consts::E)).powi(3))
            } else {
                1e-15 * (vgd / thermal).exp_m1()
            } + 1e-12 * vgd;
            let overdrive = if drain < 0. { vgd + 2. } else { 0.5 };
            let active = drain.abs().min(overdrive);
            let channel = 1e-3 * drain.signum() * active * (2. * overdrive - active);
            close(at("@j1[i]"), channel - gate_current, 1e-12);
            close(at("i(vd)"), gate_current - channel, 1e-12);
        }
        assert_eq!(seen, [true; 4], "{method}");
    }
    Ok(())
}

#[test]
fn bjt_junction_crossings_land_with_collector_kcl() -> SpiceResult<()> {
    let body = "vc c 0 0.4\nvb b 0 PWL(0 -0.2 4u 0.6 8u -0.2)\nq1 c b 0 qm\n.model qm npn(is=1e-15 bf=100 br=2 cje=20p vje=.8 mje=.4 cjc=30p vjc=.8 mjc=.4 fc=.5 tf=5n tr=2n)\nb1 out 0 v=1u*ddt(v(b))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    let thermal = (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    let be_reverse = (0.2 - 3. * thermal) / 2e5;
    let bc_reverse = (0.6 - 3. * thermal) / 2e5;
    let boundaries = [
        be_reverse,
        bc_reverse,
        3e-6,
        5e-6,
        8e-6 - bc_reverse,
        8e-6 - be_reverse,
    ];
    for method in ["method=trap", "method=gear"] {
        let plot = run_observed(
            body,
            AnalysisKind::Transient,
            &["1u", "8u", "0", "1u", "uic", method],
            &["@q1[i]"],
        )?;
        let mut seen = [false; 6];
        for index in 0..plot.point_count() {
            let at = |name: &str| plot.value(name, index).unwrap().re;
            let time = at("time");
            for (boundary, flag) in boundaries.into_iter().zip(&mut seen) {
                *flag |= (time - boundary).abs() <= 1e-17;
            }
            let be = at("v(b)");
            let bc = be - 0.4;
            let junction = |voltage: f64| {
                if voltage < -3. * thermal {
                    let a = (3. * thermal / (voltage * std::f64::consts::E)).powi(3);
                    (-1e-15 * (1. + a), 3e-15 * a / voltage)
                } else {
                    (
                        1e-15 * (voltage / thermal).exp_m1(),
                        1e-15 / thermal * (voltage / thermal).exp(),
                    )
                }
            };
            let (ibe, _) = junction(be);
            let (ibc, gbc) = junction(bc);
            let rate = at("v(out)") / 1e-6;
            let cap = 30e-12 * (1. - bc / 0.8).powf(-0.4) + 2e-9 * gbc;
            let expected = ibe - 1.5 * ibc - 1e-12 * bc + 0.4e-12 - rate * cap;
            close(at("@q1[i]"), expected, 1e-12);
            close(at("i(vc)"), -expected, 1e-12);
        }
        assert_eq!(seen, [true; 6], "{method}");
    }
    Ok(())
}

#[test]
fn semiconductor_crossing_follows_physical_decay_with_coincident_guard() -> SpiceResult<()> {
    for coincident in [false, true] {
        let derivative = if coincident {
            "abs(v(cap)-0.4)"
        } else {
            "v(cap)"
        };
        let body = format!(
            "rcap cap 0 1k\nccap cap 0 1n ic=1\nbdrive in 0 v=v(cap)\nd1 in 0 dm\n.model dm d(is=1e-12 n=10 cjo=20p vj=.8 m=.4 fc=.5 tlevc=1)\nb1 out 0 v=1u*ddt({derivative})\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n chgtol=1e-18 trtol=0.01"
        );
        let thermal = 10. * (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
        for method in ["method=trap", "method=gear"] {
            let plot = run_observed(
                &body,
                AnalysisKind::Transient,
                &["0.1u", "2u", "0", "0.1u", "uic", method],
                &["@d1[i]"],
            )?;
            let mut crossing = None;
            let mut previous = 1.;
            for index in 0..plot.point_count() {
                let at = |name: &str| plot.value(name, index).unwrap().re;
                let time = at("time");
                let voltage = at("v(cap)");
                close(voltage, (-time / 1e-6).exp(), 2e-6);
                if voltage <= 0.4 && previous > 0.4 {
                    crossing = Some(time);
                    close(voltage, 0.4, 1e-8);
                }
                previous = voltage;
                let current = 1e-12 * (voltage / thermal).exp_m1() + 1e-12 * voltage;
                let cap = if voltage < 0.4 {
                    20e-12 * (1. - voltage / 0.8).powf(-0.4)
                } else {
                    20e-12 / 0.5_f64.powf(1.4) * 0.5_f64.mul_add(voltage - 0.4, 0.5)
                };
                let expected = (-voltage / 1e-6).mul_add(cap, current);
                close(at("@d1[i]"), expected, 1e-12);
                close(at("i(bdrive)"), -expected, 1e-12);
                if coincident {
                    close(at("v(out)").abs(), voltage, 1e-7);
                    if (voltage - 0.4).abs() > 1e-8 {
                        close(at("v(out)"), -voltage * (voltage - 0.4).signum(), 1e-7);
                    }
                } else {
                    close(at("v(out)"), -voltage, 1e-7);
                }
            }
            close(
                crossing.expect("missing state-dependent semiconductor event"),
                1e-6 * 2.5_f64.ln(),
                5e-12,
            );
        }
    }
    Ok(())
}

#[test]
fn diode_breakdown_crossing_retains_selected_current_at_interior_and_stop() -> SpiceResult<()> {
    // IBV is deliberately smaller than IS*BV/Vt: the matching law
    // retains BV exactly, so the cubic and breakdown currents differ
    // at the knot. This distinguishes the selected germ from a scalar ask.
    // The dyadic times and slope give an exactly representable voltage
    // crossing, not a threshold rounded to an earlier adjacent time.
    let body = "vdrive in 0 PWL(0 0 1.9073486328125e-6 -2 3.814697265625e-6 0)\nd1 in 0 dm\n.model dm d(is=1u n=10 bv=1 ibv=1p nbv=10)\nb1 out 0 v=1u*ddt(v(in))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    let thermal = 10. * (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    let down = 9.536_743_164_062_5e-7_f64;
    let up = 3. * down;
    for method in ["method=trap", "method=gear"] {
        for (stop, interior) in [("9.5367431640625e-7", false), ("3.814697265625e-6", true)] {
            let plot = run_observed(
                body,
                AnalysisKind::Transient,
                &["0.4u", stop, "0", "0.4u", "uic", method],
                &["@d1[i]", "@d1[p]"],
            )?;
            let mut seen = [false; 2];
            for index in 0..plot.point_count() {
                let at = |name: &str| plot.value(name, index).unwrap().re;
                let time = at("time");
                let at_down = time.to_bits() == down.to_bits();
                let at_up = time.to_bits() == up.to_bits();
                seen[0] |= at_down;
                seen[1] |= at_up;
                let voltage = at("v(in)");
                let breakdown = if at_down {
                    interior
                } else if at_up {
                    false
                } else {
                    (down..up).contains(&time)
                };
                let current = if breakdown {
                    -1e-6 * (-(1. + voltage) / thermal).exp()
                } else if voltage < -3. * thermal {
                    -1e-6 * (1. + (3. * thermal / (voltage * std::f64::consts::E)).powi(3))
                } else {
                    1e-6 * (voltage / thermal).exp_m1()
                } + 1e-12 * voltage;
                close(at("@d1[i]"), current, 1e-12);
                close(at("i(vdrive)"), -current, 1e-12);
                close(at("@d1[p]"), voltage * current, 1e-12);
            }
            assert_eq!(seen, [true, interior], "{method}, stop={stop}");
        }
    }
    Ok(())
}

#[test]
fn diode_breakdown_charge_discontinuity_requires_an_impulse() {
    // With nonzero transit time the same deliberately unmatched static
    // breakdown law has discontinuous diffusion charge. A prescribed
    // continuous voltage cannot satisfy both one-sided charge values.
    let body = "vdrive in 0 PWL(0 0 1.9073486328125e-6 -2 3.814697265625e-6 0)\nd1 in 0 dm\n.model dm d(is=1u n=10 bv=1 ibv=1p nbv=10 tt=1m)\nb1 out 0 v=1u*ddt(v(in))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n chgtol=1e-18";
    for method in ["method=trap", "method=gear"] {
        let error = run(
            body,
            AnalysisKind::Transient,
            &["0.4u", "3.814697265625e-6", "0", "0.4u", "uic", method],
        )
        .expect_err("a diffusion charge jump requires an impulse")
        .to_string();
        assert!(
            error.contains("source event")
                && error.contains("d1: initial charge constraint")
                && error.contains("impulse"),
            "{method}: {error}"
        );
    }
}

/// Check every accepted physical point and require both interior events,
/// using an independent physical oracle rather than sampled derivatives.
fn qualify_constitutive_crossings(
    body: &str,
    stop: &str,
    observation: &str,
    boundaries: [f64; 2],
    check: impl Fn(&Plot, usize, [bool; 2]),
) -> SpiceResult<()> {
    for method in ["method=trap", "method=gear"] {
        let plot = run_observed(
            body,
            AnalysisKind::Transient,
            &["0.4u", stop, "0", "0.4u", "uic", method],
            &[observation],
        )?;
        let mut seen = [false; 2];
        for index in 0..plot.point_count() {
            let time = plot.value("time", index).unwrap().re;
            let at = boundaries.map(|boundary| (time - boundary).abs() <= 1e-17);
            for (hit, seen) in at.into_iter().zip(&mut seen) {
                *seen |= hit;
            }
            check(&plot, index, at);
        }
        assert_eq!(seen, [true; 2], "{method}");
    }
    Ok(())
}

#[test]
fn diode_high_injection_cutoff_localizes_the_actual_current_threshold() -> SpiceResult<()> {
    // The branch threshold is defined in amperes, not in junction voltage.
    // These intentionally small IS/IKF expose the real discontinuity at
    // 1e-18 A without the ordinary junction gmin obscuring the witness.
    let body = "vdrive in 0 PWL(0 0 2u 0.4 4u 0)\nd1 in 0 dm\n.model dm d(is=1e-18 n=10 ikf=1e-18)\nb1 out 0 v=1u*ddt(v(in))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n abstol=1e-24 gmin=1e-24";
    let thermal = 10. * (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    let crossing = thermal * std::f64::consts::LN_2 / 2e5;
    qualify_constitutive_crossings(
        body,
        "4u",
        "@d1[i]",
        [crossing, 4e-6 - crossing],
        |plot, index, [entering, leaving]| {
            let at = |name: &str| plot.value(name, index).unwrap().re;
            let voltage = at("v(in)");
            let current = 1e-18 * (voltage / thermal).exp_m1();
            let limited = entering || (!leaving && current > 1e-18);
            let expected = if limited {
                current / (1. + (current / 1e-18).sqrt())
            } else {
                current
            } + 1e-24 * voltage;
            close(at("@d1[i]"), expected, 1e-21);
            close(at("i(vdrive)"), -expected, 1e-21);
        },
    )
}

#[test]
fn jfet_cutoff_crossings_follow_the_gate_overdrive() -> SpiceResult<()> {
    let body = "vd d 0 1\nvg g 0 PWL(0 -3 1.9073486328125e-6 -1 3.814697265625e-6 -3)\nj1 d g 0 jm\n.model jm njf(vto=-2 beta=1m lambda=0 b=1 is=1e-15 n=10)\nb1 out 0 v=1u*ddt(v(g))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    let thermal = 10. * (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    let boundary = 9.536_743_164_062_5e-7_f64;
    qualify_constitutive_crossings(
        body,
        "3.814697265625e-6",
        "@j1[i]",
        [boundary, 3. * boundary],
        |plot, index, _| {
            let at = |name: &str| plot.value(name, index).unwrap().re;
            let gate = at("v(g)");
            let vgd = gate - 1.;
            let junction =
                -1e-15 * (1. + (3. * thermal / (vgd * std::f64::consts::E)).powi(3)) + 1e-12 * vgd;
            let expected = 1e-3 * (gate + 2.).max(0.).powi(2) - junction;
            close(at("@j1[i]"), expected, 1e-12);
            close(at("i(vd)"), -expected, 1e-12);
        },
    )
}

// The independent level-one BJT junction characteristic and derivative.
fn crossing_bjt_junction(voltage: f64, thermal: f64, saturation: f64) -> (f64, f64) {
    if voltage < -3. * thermal {
        let a = (3. * thermal / (voltage * std::f64::consts::E)).powi(3);
        (-saturation * (1. + a), 3. * saturation * a / voltage)
    } else {
        (
            saturation * (voltage / thermal).exp_m1(),
            saturation / thermal * (voltage / thermal).exp(),
        )
    }
}

#[test]
fn bjt_forward_diffusion_activation_preserves_charge_and_changes_rate() -> SpiceResult<()> {
    let body = "vc c 0 0.1\nvb b 0 PWL(0 -0.125 1.9073486328125e-6 0.125 3.814697265625e-6 -0.125)\nq1 c b 0 qm\n.model qm npn(is=1e-12 bf=100 br=2 tf=10n xtf=1)\nb1 out 0 v=1u*ddt(v(b))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    let thermal = (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    let boundary = 9.536_743_164_062_5e-7_f64;
    qualify_constitutive_crossings(
        body,
        "3.814697265625e-6",
        "@q1[i]",
        [boundary, 3. * boundary],
        |plot, index, [entering, leaving]| {
            let at = |name: &str| plot.value(name, index).unwrap().re;
            let voltage = at("v(b)");
            let (be, gbe) = crossing_bjt_junction(voltage, thermal, 1e-12);
            let (bc, _) = crossing_bjt_junction(voltage - 0.1, thermal, 1e-12);
            let factor = if entering || (!leaving && voltage > 0.) {
                2.
            } else {
                1.
            };
            let rate = at("v(out)") / 1e-6;
            let base =
                be / 100. + bc / 2. + 1e-12 * (2. * voltage - 0.1) + factor * 1e-8 * gbe * rate;
            let collector = be - 1.5 * bc - 1e-12 * (voltage - 0.1) + 1e-13;
            close(at("i(vb)"), -base, 1e-16);
            close(at("i(vc)"), -collector, 1e-16);
            close(at("@q1[i]"), collector, 1e-16);
        },
    )
}

#[test]
fn bjt_base_charge_clamp_uses_the_selected_one_sided_current() -> SpiceResult<()> {
    // NKF=1 gives analytic germs on both sides of the actual base-charge
    // root clamp. With TF=0 the current jump does not demand an impulse.
    let body = "vc c 0 0.1\nvb b 0 PWL(0 -0.05 2u 0.05 4u -0.05)\nq1 c b 0 qm\n.model qm npn(is=1u bf=100 br=2 ikf=2u nkf=1)\nb1 out 0 v=1u*ddt(v(b))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n";
    let thermal = (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    let crossing = (0.05 - thermal * std::f64::consts::LN_2) / 5e4;
    qualify_constitutive_crossings(
        body,
        "4u",
        "@q1[i]",
        [crossing, 4e-6 - crossing],
        |plot, index, [entering, leaving]| {
            let at = |name: &str| plot.value(name, index).unwrap().re;
            let voltage = at("v(b)");
            let (be, _) = crossing_bjt_junction(voltage, thermal, 1e-6);
            let (bc, _) = crossing_bjt_junction(voltage - 0.1, thermal, 1e-6);
            let argument = 1. + 4. * be / 2e-6;
            let root = if entering || (!leaving && argument > 0.) {
                argument
            } else {
                1.
            };
            let qb = (1. + root) / 2.;
            let expected = (be - bc) / qb - bc / 2. - 1e-12 * (voltage - 0.1) + 1e-13;
            close(at("@q1[i]"), expected, 1e-12);
            close(at("i(vc)"), -expected, 1e-12);
        },
    )
}

#[test]
fn bjt_base_spreading_clamp_tracks_the_intrinsic_base_current() -> SpiceResult<()> {
    let body = "vc c 0 0\nvb b 0 PWL(0 -0.05 2u 0.05 4u -0.05)\nq1 c b 0 qm\n.model qm npn(is=1e-12 bf=1 br=1 rb=1meg rbm=1k irb=1m)\nb1 out 0 v=1u*ddt(v(b))\nr1 out 0 1k\n.options reltol=1e-8 vntol=1n abstol=1e-17";
    let thermal = (1.380_648_52e-23 / 1.602_176_620_8e-19) * 300.15;
    let base = |voltage: f64| 2e-12 * ((voltage / thermal).exp_m1() + voltage);
    // Here the spreading ratio is < 2e-8, so the first two nonconstant
    // terms of 3*(tan(z)-z)/(z*tan(z)^2) suffice at voltage tolerance.
    let resistance = |current: f64| {
        let ratio = (current / 1e-3).max(1e-9);
        let angle =
            (14.59025 / 2.4317) / (ratio.sqrt().recip() + (ratio.recip() + 14.59025).sqrt());
        1000. + 999_000. * (1. - 4. / 15. * angle.powi(2) - 4. / 105. * angle.powi(4))
    };
    let mut lower = 0.;
    let mut upper = 0.05;
    for _ in 0..60 {
        let middle = f64::midpoint(lower, upper);
        if base(middle) < 1e-12 {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let voltage = f64::midpoint(lower, upper) + 1e-12 * resistance(1e-12);
    let crossing = (voltage + 0.05) / 5e4;
    qualify_constitutive_crossings(
        body,
        "4u",
        "@q1[i]",
        [crossing, 4e-6 - crossing],
        |plot, index, hits| {
            let at = |name: &str| plot.value(name, index).unwrap().re;
            let intrinsic = at("v(q1#base)");
            let current = base(intrinsic);
            if hits.into_iter().any(|hit| hit) {
                close(current, 1e-12, 1e-17);
            }
            close(at("i(vb)"), -current, 1e-17);
            close(at("i(vc)"), current / 2., 1e-17);
            close(at("@q1[i]"), -current / 2., 1e-17);
            close(at("v(b)") - intrinsic, current * resistance(current), 1e-12);
        },
    )
}
