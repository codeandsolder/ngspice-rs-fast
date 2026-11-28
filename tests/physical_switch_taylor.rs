//! Exact physical switch germs, independent from the reference AC convention.
use ngspice_rs::{
    analysis::{AnalysisInput, RunConfig},
    devices::{Circuit, Limit, ModelContext, TransientTiming},
    maths::Vector,
    netlist::{Parser, ast::Netlist, source::parse_deck_text},
    primitives::{AnalysisKind, Real, SpiceResult},
};
use std::path::Path;

fn parse(body: &str) -> SpiceResult<Netlist> {
    Parser::new().parse_deck(&parse_deck_text(
        Path::new("physical-switch.cir"),
        &format!("physical switch Taylor\n{body}\n.end\n"),
    ))
}

fn row(c: &Circuit, node: &str) -> usize {
    c.unknowns().node_row(c.nodes().get(node).unwrap()).unwrap()
}

fn branch(c: &Circuit, device: &str) -> usize {
    let index = c.devices().iter().position(|d| d.name() == device).unwrap();
    c.branch_rows(index).unwrap().start
}

fn close(label: &str, a: Real, b: Real, relative: Real, absolute: Real) {
    assert!(
        (a - b).abs() <= relative.mul_add(a.abs().max(b.abs()), absolute),
        "{label}: {a:.17e} != {b:.17e}"
    );
}

fn trajectory(coefficients: &[Vector], u: Real) -> Vector {
    let n = coefficients[0].len();
    Vector::from_slice(
        &(0..n)
            .map(|row| {
                coefficients
                    .iter()
                    .rev()
                    .fold(0., |sum: Real, x| sum.mul_add(u, x.as_slice()[row]))
            })
            .collect::<Vec<_>>(),
    )
}

fn check(c: &Circuit, states: &[Vector], side: Limit) -> SpiceResult<()> {
    let run = c.prepare_run();
    let mut dae =
        run.prepare_physical_dae(&ModelContext::default(), TransientTiming::new(1e-6, 1e-3)?)?;
    let jet = dae.evaluate_taylor(states, 1e-4, 1e-6, side)?;
    let direction = if side == Limit::Right { 1. } else { -1. };
    for u in [0.005, 0.01, 0.025].map(|u| u * direction) {
        let scalar = dae.evaluate(&trajectory(states, u), 1e-4 + u * 1e-6, side)?;
        let n = c.unknown_count();
        for r in 0..n {
            let at = |vectors: &[Vector]| {
                vectors
                    .iter()
                    .rev()
                    .fold(0., |sum: Real, x| sum.mul_add(u, x.as_slice()[r]))
            };
            close(
                "residual",
                at(&jet.residual),
                scalar.residual.as_slice()[r],
                3e-9,
                3e-12,
            );
            close(
                "charge",
                at(&jet.charge),
                scalar.charge.as_slice()[r],
                3e-9,
                3e-20,
            );
            for col in 0..n {
                let at = |matrices: &[ngspice_rs::maths::SparseMatrix]| {
                    matrices
                        .iter()
                        .rev()
                        .fold(0., |sum: Real, x| sum.mul_add(u, x.get(r, col)))
                };
                close(
                    "static Jacobian",
                    at(&jet.conductance),
                    scalar.conductance.get(r, col),
                    5e-8,
                    3e-11,
                );
                close(
                    "charge Jacobian",
                    at(&jet.capacitance),
                    scalar.capacitance.get(r, col),
                    3e-9,
                    3e-20,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn xspice_floating_voltage_controls_all_laws_and_threshold_directions() -> SpiceResult<()> {
    for reversed in [false, true] {
        let (off, on): (Real, Real) = if reversed { (4., 2.) } else { (2., 4.) };
        for (family, log, limited) in [
            ("pswitch", true, true),
            ("pswitch", false, true),
            ("aswitch", true, true),
            ("aswitch", false, true),
            ("aswitch", true, false),
            ("aswitch", false, false),
        ] {
            let extras = if family == "pswitch" {
                "r_cntl_in=100".to_owned()
            } else {
                format!("limit={limited}")
            };
            let c = Circuit::from_netlist(&parse(&format!(
                "Vref z 0 5\nVc c z 0\nVb p z 0\nA1 %gd(c z) %gd(p z) m\n.model m {family}(cntl_off={off} cntl_on={on} r_on=10 r_off=1k log={log} {extras})"
            ))?)?;
            let [p, z, ctrl] = ["p", "z", "c"].map(|node| row(&c, node));
            let (lo, hi) = (off.min(on), off.max(on));
            for control in [lo - 0.2, lo, 2.4, 3., 3.7, hi, hi + 0.2] {
                for output in [-1.4, 0., 0.9] {
                    let mut state = vec![Vector::zeros(c.unknown_count()); 9];
                    for (r, value, slope) in [
                        (p, 5. + output, 0.12),
                        (z, 5., 0.03),
                        (ctrl, 5. + control, 0.09),
                    ] {
                        state[0].set(r, value)?;
                        state[1].set(r, slope)?;
                    }
                    // Avoid sampling across another nearby resistance-floor
                    // boundary; every endpoint is tested from both sides.
                    for side in [Limit::Left, Limit::Right] {
                        check(&c, &state, side)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn xspice_current_controls_keep_ammeter_and_named_branch_equations() -> SpiceResult<()> {
    for (port, control_branch) in [("%id(c z)", "a1"), ("%vnam(vctrl)", "vctrl")] {
        let c = Circuit::from_netlist(&parse(&format!(
            "Vref z 0 5\nVctrl x z 0\nRctrl x z 1k\nRfix c z 1k\nIctrl z c .3\nVb p z 0\nA1 {port} %gd(p z) m\n.model m aswitch(cntl_off=0 cntl_on=1 r_on=10 r_off=1k log=true limit=true)"
        ))?)?;
        let mut state = vec![Vector::zeros(c.unknown_count()); 9];
        for (r, value, slope) in [
            (row(&c, "z"), 5., 0.02),
            (row(&c, "p"), 4., 0.05),
            (branch(&c, control_branch), 0.4, 0.06),
        ] {
            state[0].set(r, value)?;
            state[1].set(r, slope)?;
        }
        check(&c, &state, Limit::Right)?;
        check(&c, &state, Limit::Left)?;
    }
    Ok(())
}

#[test]
fn ltspice_control_current_limits_offset_epsilon_and_series_flux() -> SpiceResult<()> {
    for model in [
        "ron=1 roff=1k vt=0 vh=-1 level=1",
        "ron=1 roff=1k vt=0 vh=-1 level=1 ilimit=.05",
        "ron=1 roff=1k vt=0 vh=-1 level=1 vser=.5 ilimit=.05",
        "ron=1 roff=1k vt=0 vh=-1 level=1 vser=-.5",
        "ron=1 roff=1k vt=0 vh=-1 level=1 lser=2u",
        "ron=1 roff=1k vt=0 vh=-1 level=2 ilimit=.05",
        "ron=1 roff=1k vt=0 vh=-1 level=2 ilimit=.05 oneway",
        "ron=1 roff=1k vt=0 vh=-1 level=2 ilimit=.05 vser=.5 epsilon=.2",
        "ron=.001 roff=.002 vt=0 vh=-1 level=2 ilimit=.05 vser=.5 epsilon=.2",
    ] {
        let c = Circuit::from_netlist(&parse(&format!(
            "Vref z 0 5\nVc c z 0\nVb p z 0\nS1 p z c z sm\n.model sm ltspice_switch({model})"
        ))?)?;
        let [p, z, ctrl] = ["p", "z", "c"].map(|node| row(&c, node));
        let conducting = c
            .nodes()
            .get("s1#lser")
            .and_then(|node| c.unknowns().node_row(node))
            .unwrap_or(p);
        for control in [-1.2, -1., -0.3, 0.4, 1., 1.2] {
            for voltage in [-1.1, -0.7, -0.6, -0.5, -0.4, 0., 0.5, 0.9] {
                let mut state = vec![Vector::zeros(c.unknown_count()); 9];
                for (r, value, slope) in
                    [(p, voltage, 0.035), (z, 0., 0.005), (ctrl, control, 0.027)]
                {
                    state[0].set(r, value)?;
                    state[1].set(r, slope)?;
                }
                if conducting != p {
                    state[0].set(conducting, voltage)?;
                    state[1].set(conducting, 0.035)?;
                    let i = branch(&c, "s1");
                    state[0].set(i, 0.12)?;
                    state[1].set(i, -0.01)?;
                }
                for side in [Limit::Left, Limit::Right] {
                    check(&c, &state, side)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn derivative_array_transient_crosses_actual_switch_joins() -> SpiceResult<()> {
    // A derivative output forces physical derivative-array evolution. The
    // ideal driver and independent load-line equation are exact references.
    for model in [
        "pswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=true r_cntl_in=200)",
        "aswitch(cntl_off=2 cntl_on=4 r_on=10 r_off=1k log=false limit=true)",
    ] {
        let body = format!(
            "Vc c 0 pwl(0 1.5 3u 4.5)\nVb source 0 1\nRseries source out 100\nA1 %gd(c 0) %gd(out 0) m\n.model m {model}\nB1 derivative 0 v=1u*ddt(v(out))\nR1 derivative 0 1k\n.options reltol=1e-8 abstol=1e-13 vntol=1e-10"
        );
        let netlist = parse(&body)?;
        let config = RunConfig::from_netlist(&netlist)?;
        for method in ["method=trap", "method=gear"] {
            let mut c = config.circuit(&netlist)?;
            let plot = config
                .plan(AnalysisInput::with_arguments(
                    AnalysisKind::Transient,
                    ["50n", "3u", "0", "50n", "uic", method],
                ))?
                .run(&mut c, &config.context())?;
            let mut crossed = [false; 2];
            for index in 0..plot.point_count() {
                let get = |name| plot.value(name, index).unwrap().re;
                let time = get("time");
                let control = get("v(c)");
                let t = ((control - 2.) / 2.).clamp(0., 1.);
                let (g, dg) = if model.starts_with("pswitch") {
                    let g = 100_f64.powf(t * t * (3. - 2. * t)) / 1000.;
                    (g, g * 100_f64.ln() * 3. * t * (1. - t))
                } else {
                    let g = (1000. - 990. * t).recip();
                    (
                        g,
                        if control > 2. && control < 4. {
                            495. * g * g
                        } else {
                            0.
                        },
                    )
                };
                let expected = (1. + 100. * g).recip();
                close("load line", get("v(out)"), expected, 2e-6, 2e-9);
                close("source KCL", get("i(vb)"), -expected * g, 2e-6, 2e-11);
                // At a hard linear-resistance join the output derivative
                // is one-sided. The emitted point uses the right-hand germ.
                let at_join = (control - 2.).abs() < 1e-9 || (control - 4.).abs() < 1e-9;
                if !at_join && time > 0. && time < 3e-6 {
                    close(
                        "output derivative",
                        get("v(derivative)"),
                        -100. * dg * expected * expected,
                        2e-6,
                        2e-9,
                    );
                }
                for (slot, join) in [2., 4.].into_iter().enumerate() {
                    crossed[slot] |= (control - join).abs() < 1e-8;
                }
            }
            assert!(
                crossed.into_iter().all(|v| v),
                "missing state-dependent join {model} {method}"
            );
        }
    }
    Ok(())
}
