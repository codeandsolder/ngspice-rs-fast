//! Phase-selected behavioural sampling, independently of physical current reporting.
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
fn feedback_converges_to_present_physical_current_for_both_companion_methods() -> SpiceResult<()> {
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
            assert!(
                (held - port).magnitude() <= 2e-6 * port.magnitude() + 2e-12,
                "{method} point {point}: {held:?} != {port:?}"
            );
            close(live, port);
            close(
                plot.value("v(s)", point).unwrap(),
                plot.value("v(drive)", point).unwrap() - Complex::real(0.5) * held,
            );
            changed |= (held - previous).magnitude() > 1e-5;
        }
        assert!(
            changed,
            "{method}: sampled and preceding-point currents never differed"
        );
    }
    assert!(
        rejected > 0,
        "test did not exercise a rejected adaptive trial"
    );
    Ok(())
}

#[test]
fn feedback_endpoint_matches_independent_ltspice_witness() -> SpiceResult<()> {
    let evidence = include_str!("fixtures/sampled_current/ltspice-24.0.12.tsv");
    for (method, witness) in [
        ("method=trap", "trap-.5-1e-6"),
        ("method=gear", "gear-.5-1e-6"),
    ] {
        let fields: Vec<_> = evidence
            .lines()
            .find(|line| line.split('\t').next() == Some(witness))
            .unwrap()
            .split('\t')
            .collect();
        let mut c = circuit(&format!(
            "feedback witness\nv1 drive 0 dc 2 pulse(2 3 1u 100n 100n 2u 5u)\nbdrive s 0 v=v(drive)-.5*sampled_current(i(c1))\n{NETWORK}.end\n"
        ))?;
        let plan = AnalysisInput::with_arguments(
            AnalysisKind::Transient,
            [
                "50n",
                "5u",
                "0",
                "50n",
                method,
                "rtol=1e-6",
                "vntol=1e-9",
                "abstol=1e-12",
            ],
        )
        .resolve()?;
        let (plot, _) = companion_transient(&mut c, &plan, &AnalysisContext::default())?;
        let last = plot.point_count() - 1;
        for (name, field, absolute) in [
            ("time", 4, 1e-14),
            ("v(s)", 5, 5e-5),
            ("v(p)", 6, 5e-5),
            ("i(c1)", 7, 2e-5),
        ] {
            let expected: f64 = fields[field].parse().unwrap();
            let actual = plot.value(name, last).unwrap();
            assert!(
                (actual - Complex::real(expected)).magnitude() <= absolute,
                "{witness} {name}: {actual:?} != {expected:e}"
            );
        }
    }
    Ok(())
}

#[test]
fn transient_load_samples_only_its_input_iterate_and_has_no_control_tangent() -> SpiceResult<()> {
    let mut c =
        circuit("transaction\nv1 a 0 1\nr1 a 0 1\nb1 b 0 v=sampled_current(i(v1))\n.end\n")?;
    c.finalize()?;
    let sample = c
        .devices()
        .iter()
        .position(|d| d.name().starts_with("b__sampled_current_"))
        .unwrap();
    let control = c.devices().iter().position(|d| d.name() == "v1").unwrap();
    let row = c
        .unknowns()
        .node_row(c.devices()[sample].constitutive_nodes()[0])
        .unwrap();
    let control_row = c.branch_rows(control).unwrap().start;
    let n = c.unknown_count();
    let history = c.state_history()?;
    let accepted = history.clone();
    assert!(c.state_rows(sample).unwrap().is_empty());
    let load = |mode, current| -> SpiceResult<(SparseMatrix, Vector)> {
        let mut x = vec![0.; n];
        x[control_row] = current;
        let mut trial = history.trial();
        let mut a = c.new_matrix()?;
        let mut rhs = Vector::zeros(n);
        c.load_device(
            sample,
            c.devices()[sample].as_ref(),
            &LoadRequest {
                mode,
                solution: &Vector::from_slice(&x),
                model_context: &ModelContext::default(),
                integration: None,
                history: &history,
                forcing: None,
            },
            &mut a,
            &mut rhs,
            &mut trial,
        )?;
        Ok((a, rhs))
    };
    let (a, rhs) = load(AnalysisMode::OperatingPoint, 3.)?;
    assert_eq!(a.get(row, control_row), -1.);
    assert_eq!(rhs.get(row), Some(0.));
    let mode = AnalysisMode::Transient { time: 1., dt: 1. };
    for guess in [9., -11., 2., 9.] {
        let (a, rhs) = load(mode, guess)?;
        assert_eq!(a.get(row, row), 1.);
        assert_eq!(a.get(row, control_row), 0.);
        assert_eq!(rhs.get(row), Some(guess));
        assert_eq!(history, accepted);
    }
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

#[test]
fn malformed_sampling_is_not_silently_converted_to_a_static_expression() {
    for expression in [
        "sampled_current(0)",
        "sampled_current(i(v1)+1)",
        "sampled_current(i(v1),i(v1))",
        "sampled_current(i(missing))",
        "branch_or_sampled_current(0)",
        "branch_or_sampled_current(i(missing))",
    ] {
        let text = format!("invalid sample\nv1 a 0 1\nr1 a 0 1\nb1 b 0 v={expression}\n.end\n");
        assert!(
            circuit(&text).and_then(|mut c| c.finalize()).is_err(),
            "accepted {expression}"
        );
    }
}

#[test]
fn selection_uses_actual_macro_target_before_sensing_aliases() -> SpiceResult<()> {
    // Same independently measured source deck, with only explicit native
    // sampling policy and zero-loss storage spelling substituted.
    let source = include_str!("fixtures/sampled_current/selection.cir");
    let mut deck = source
        .lines()
        .filter(|line| !line.starts_with('.'))
        .collect::<Vec<_>>()
        .join("\n");
    deck = deck
        .replace("C1 drive 0 2u Rpar=0", "C1 drive 0 2u")
        .replace("L1 a 0 3m Rser=0 Rpar=0", "L1 a 0 3m");
    deck.push_str("\n.func read(x) {branch_or_sampled_current(i(x))}\n.func nested(y) {read(y)}\nbslive srclive 0 v=i(v1)\nbrlive rlive 0 v=i(r2)\nbrepeat repeat 0 v=sampled_current(i(r2))\nbforced forced 0 v=sampled_current(i(v1))\n.end\n");
    let mut c = circuit(&deck)?;
    let op = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    let ac = run(&mut c, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
    let evidence = include_str!("fixtures/sampled_current/selection-ltspice-24.0.12.tsv");
    for line in evidence
        .lines()
        .skip(1)
        .filter(|line| line.starts_with("v("))
    {
        let fields: Vec<_> = line.split('\t').collect();
        let value = |index: usize| fields[index].parse::<f64>().unwrap();
        close(op.value(fields[0], 0).unwrap(), Complex::real(value(1)));
        close(
            ac.value(fields[0], 0).unwrap(),
            Complex::new(value(2), value(3)),
        );
    }
    close(
        op.value("v(src)", 0).unwrap(),
        op.value("v(srclive)", 0).unwrap(),
    );
    close(
        op.value("v(forced)", 0).unwrap(),
        op.value("v(src)", 0).unwrap(),
    );
    close(op.value("v(repeat)", 0).unwrap(), Complex::real(0.25));
    close(ac.value("v(rlive)", 0).unwrap(), Complex::real(0.125));
    close(
        ac.value("v(src)", 0).unwrap(),
        ac.value("v(srclive)", 0).unwrap(),
    );
    for name in ["v(repeat)", "v(forced)"] {
        close(ac.value(name, 0).unwrap(), Complex::real(0.));
    }
    // The six observation-only targets, plus a forced sample of v1.
    // Repeated reads of r2 share a stateless sample coordinate.
    assert_eq!(
        c.devices()
            .iter()
            .filter(|d| d.name().starts_with("b__sampled_current_"))
            .count(),
        7
    );
    Ok(())
}

#[test]
fn physical_dae_and_taylor_use_the_exact_residual_not_the_ac_tangent() -> SpiceResult<()> {
    use ngspice_rs::devices::{Limit, TransientTiming};
    let mut c =
        circuit("physical law\nv1 a 0 1\nr1 a 0 1\nb1 b 0 v=sampled_current(i(v1))\n.end\n")?;
    c.finalize()?;
    let index = |name| c.devices().iter().position(|d| d.name() == name).unwrap();
    let sample = c
        .devices()
        .iter()
        .find(|d| d.name().starts_with("b__sampled_current_"))
        .unwrap();
    let row = c
        .unknowns()
        .node_row(sample.constitutive_nodes()[0])
        .unwrap();
    let control = c.branch_rows(index("v1")).unwrap().start;
    let run = c.prepare_run();
    let mut dae =
        run.prepare_physical_dae(&ModelContext::default(), TransientTiming::new(1e-6, 1e-3)?)?;
    let mut state = vec![Vector::zeros(c.unknown_count()); 3];
    for (x, (sample, physical)) in state.iter_mut().zip([(2., 3.), (-3., 1.), (0.5, -0.5)]) {
        x.set(row, sample)?;
        x.set(control, physical)?;
    }
    let exact = dae.evaluate(&state[0], 0., Limit::Right)?;
    assert_eq!(exact.residual.get(row), Some(-1.));
    assert_eq!(exact.conductance.get(row, control), -1.);
    assert_eq!(exact.conductance.get(row, row), 1.);
    assert_eq!(exact.charge.get(row), Some(0.));
    for limit in [Limit::Left, Limit::Right] {
        let taylor = dae.evaluate_taylor(&state, 0., 1e-6, limit)?;
        for (degree, expected) in [-1., -4., 1.].into_iter().enumerate() {
            assert_eq!(taylor.residual[degree].get(row), Some(expected));
            assert_eq!(taylor.charge[degree].get(row), Some(0.));
            assert_eq!(
                taylor.conductance[degree].get(row, control),
                if degree == 0 { -1. } else { 0. }
            );
        }
    }
    Ok(())
}

#[test]
fn sampled_current_does_not_add_noise_or_change_small_signal_gain() -> SpiceResult<()> {
    let mut spectra = Vec::new();
    for expression in ["v(p)", "v(p)+2*sampled_current(i(r2))"] {
        let mut c = circuit(&format!(
            "noise\nv1 s 0 dc 2 ac 1\nr1 s p 3\nr2 p 0 5\nb1 out 0 v={expression}\n.end\n"
        ))?;
        spectra.push(run(
            &mut c,
            AnalysisKind::Noise,
            &["v(out)", "v1", "lin", "3", "100", "10000"],
        )?);
    }
    for name in ["onoise_spectrum", "inoise_spectrum"] {
        for point in 0..3 {
            let baseline = spectra[0].value(name, point).unwrap();
            let sample = spectra[1].value(name, point).unwrap();
            assert!(baseline.magnitude() > 0.);
            assert!(
                (sample - baseline).magnitude() <= 1e-12 * baseline.magnitude(),
                "{name}: {sample:?} != {baseline:?}"
            );
        }
    }
    Ok(())
}
