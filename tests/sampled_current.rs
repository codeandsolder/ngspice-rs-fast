//! Accepted-point behavioural sampling, independently of physical current reporting.
use std::path::Path;

use ngspice_rs::{
    analysis::{AnalysisContext, AnalysisInput, Plot, RunConfig, companion_transient},
    devices::{AnalysisMode, Circuit, LoadRequest, ModelContext},
    maths::{SparseMatrix, Vector},
    netlist::{Parser, source::parse_deck_text},
    primitives::{AnalysisKind, Complex, SpiceResult},
};

fn circuit(text: &str) -> SpiceResult<Circuit> {
    let deck = parse_deck_text(Path::new("sample.cir"), text);
    let netlist = Parser::new().parse_deck(&deck)?;
    RunConfig::from_netlist(&netlist)?.circuit(&netlist)
}

fn run(c: &mut Circuit, kind: AnalysisKind, args: &[&str]) -> SpiceResult<Plot> {
    c.finalize()?;
    AnalysisInput::with_arguments(kind, args.iter().copied())
        .resolve()?
        .run(c, &AnalysisContext::default())
}

fn close(got: Complex, want: Complex) {
    assert!(
        (got - want).magnitude() <= 2e-8 + 1e-8 * want.magnitude(),
        "{got:?} != {want:?}"
    );
}

const NETWORK: &str = "r1 s p 3\nc1 p 0 2u pn_rseries=5 pn_rparallel=20 pn_cparallel=1u pn_control=1 pn_report_dc=1 pn_report_ac=1 pn_report_tran=1\nb1 b 0 v=sampled_current(i(c1))\nb2 live 0 v=i(c1)\nb3 mixed 0 v=sampled_current(i(c1))*v(s)\nr2 b 0 1\n";

#[test]
fn dc_ac_and_instantaneous_probe_are_independent() -> SpiceResult<()> {
    let mut c = circuit(&format!(
        "phase selection\nv1 s 0 dc 2 ac 1\n{NETWORK}.end\n"
    ))?;
    let op = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    close(op.value("v(p)", 0).unwrap(), Complex::real(40. / 23.));
    for name in ["v(b)", "v(live)", "i(c1)", "i(vc1#port)"] {
        close(op.value(name, 0).unwrap(), Complex::real(2. / 23.));
    }
    let dc = run(&mut c, AnalysisKind::DcSweep, &["v1", "1", "2", ".5"])?;
    for (point, v) in [1., 1.5, 2.].into_iter().enumerate() {
        close(dc.value("v(b)", point).unwrap(), Complex::real(v / 23.));
    }
    // Independently measured LTspice 24.0.12 port currents, issue #274.
    for (frequency, physical) in [
        (
            "100",
            Complex::new(0.04349123838977926, 0.001425166557324791),
        ),
        (
            "10000",
            Complex::new(0.1080018567925942, 0.07832699946699417),
        ),
    ] {
        let ac = run(
            &mut c,
            AnalysisKind::Ac,
            &["lin", "1", frequency, frequency],
        )?;
        close(ac.value("v(b)", 0).unwrap(), Complex::real(0.));
        for name in ["v(live)", "i(c1)", "i(vc1#port)"] {
            close(ac.value(name, 0).unwrap(), physical);
        }
        // The sampled current's bias value still multiplies the live voltage tangent.
        close(ac.value("v(mixed)", 0).unwrap(), Complex::real(2. / 23.));
    }
    Ok(())
}

#[test]
fn feedback_reads_preceding_accepted_point_for_both_companion_methods() -> SpiceResult<()> {
    let mut rejected = 0;
    for method in ["method=trap", "method=gear"] {
        let mut c = circuit(&format!(
            "feedback\nv1 drive 0 dc 2 pulse(2 4 10u 100n 100n 30u 80u)\nbdrive s 0 v=v(drive)-.5*sampled_current(i(c1))\n{NETWORK}.end\n"
        ))?;
        let plan = AnalysisInput::with_arguments(
            AnalysisKind::Transient,
            ["2u", "100u", "0", "2u", method, "rtol=1e-6"],
        )
        .resolve()?;
        let (plot, stats) = companion_transient(&mut c, &plan, &AnalysisContext::default())?;
        rejected += stats.rejected;
        assert!(plot.point_count() > 50);
        let mut changed = false;
        for point in 1..plot.point_count() {
            let held = plot.value("v(b)", point).unwrap();
            let live = plot.value("v(live)", point).unwrap();
            let port = plot.value("i(vc1#port)", point).unwrap();
            let previous = plot.value("i(vc1#port)", point - 1).unwrap();
            close(held, previous);
            close(live, port);
            close(
                plot.value("v(s)", point).unwrap(),
                plot.value("v(drive)", point).unwrap() - Complex::real(0.5) * held,
            );
            changed |= (held - live).magnitude() > 1e-5;
        }
        assert!(changed, "{method}: sampled and live current never differed");
    }
    assert!(
        rejected > 0,
        "test did not exercise a rejected adaptive trial"
    );
    Ok(())
}

#[test]
fn acceptance_publishes_solved_current_not_trial_guess() -> SpiceResult<()> {
    let mut c =
        circuit("transaction\nv1 a 0 1\nr1 a 0 1\nb1 b 0 v=sampled_current(i(v1))\n.end\n")?;
    c.finalize()?;
    let sample = c
        .devices()
        .iter()
        .position(|d| !d.accepted_control_samples().is_empty())
        .unwrap();
    let control = c.devices().iter().position(|d| d.name() == "v1").unwrap();
    let slot = c.state_rows(sample).unwrap().start;
    let row = c
        .unknowns()
        .node_row(c.devices()[sample].constitutive_nodes()[0])
        .unwrap();
    let control_row = c.branch_rows(control).unwrap().start;
    let n = c.unknown_count();
    let solution = |current| {
        let mut x = vec![0.; n];
        x[control_row] = current;
        Vector::from_slice(&x)
    };
    let mut history = c.state_history()?;
    let load = |history: &ngspice_rs::devices::StateHistory, mode, current| -> SpiceResult<_> {
        let mut trial = history.trial();
        let mut a = SparseMatrix::new(n, n);
        let mut rhs = Vector::zeros(n);
        c.load(
            &LoadRequest {
                mode,
                solution: &solution(current),
                model_context: &ModelContext::default(),
                integration: None,
                history,
                forcing: None,
            },
            &mut a,
            &mut rhs,
            &mut trial,
        )?;
        Ok((trial, rhs))
    };
    let (trial, _) = load(&history, AnalysisMode::OperatingPoint, 3.)?;
    c.accept_point(&solution(4.), Some(0.), &mut history, trial)?;
    assert_eq!(history.accepted(1).unwrap()[slot], 4.);
    let mode = AnalysisMode::Transient { time: 1., dt: 1. };
    let accepted = history.clone();
    for guess in [9., -11., 2.] {
        let (_, rhs) = load(&history, mode, guess)?;
        assert_eq!(rhs.get(row), Some(4.));
        assert_eq!(history, accepted);
    }
    let (trial, _) = load(&history, mode, 9.)?;
    c.accept_point(&solution(7.), Some(1.), &mut history, trial)?;
    assert_eq!(history.accepted(1).unwrap()[slot], 7.);
    let (_, rhs) = load(&history, AnalysisMode::Transient { time: 2., dt: 1. }, 10.)?;
    assert_eq!(rhs.get(row), Some(7.));
    Ok(())
}

#[test]
fn lexical_macro_expansion_and_removed_networks() -> SpiceResult<()> {
    let mut c = circuit(
        "occurrences\nv1 a 0 2 ac 1\nv2 z 0 1 ac 2\nx1 a b1 live1 cell mval=2\nx2 z b2 live2 cell mval=0\n.subckt cell a b live params: mval=1\nc1 a 0 2u m={mval} pn_rparallel=20 pn_control=1\n.func hold(x) {sampled_current(x)}\nb1 b 0 v=hold(i(c1))\nb2 live 0 v=i(c1)\n.ends\n.end\n",
    )?;
    let op = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    for name in ["v(b1)", "v(live1)"] {
        close(op.value(name, 0).unwrap(), Complex::real(0.2));
    }
    for name in ["v(b2)", "v(live2)"] {
        close(op.value(name, 0).unwrap(), Complex::real(0.));
    }
    let ac = run(&mut c, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
    for name in ["v(b1)", "v(b2)", "v(live2)"] {
        close(ac.value(name, 0).unwrap(), Complex::real(0.));
    }
    close(
        ac.value("v(live1)", 0).unwrap(),
        Complex::new(0.1, 4e-6 * std::f64::consts::TAU * 1000.),
    );
    Ok(())
}
