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
