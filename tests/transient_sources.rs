//! Independent sources stamp their time-`t` forcing in companion transient loads.
use ngspice_rs::devices::{
    AnalysisMode, Circuit, Forcing, IndependentSource, Limit, LoadRequest, ModelContext, Pulse,
    PulseSpec, Resistor, TransientTiming, Waveform,
};
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::primitives::{Complex, NodeId, SpiceResult};

fn circuit(voltage: bool, waveform: Waveform) -> Circuit {
    let mut circuit = Circuit::new();
    let a = circuit.add_node("a").unwrap();
    circuit
        .add_device(Box::new(
            IndependentSource::new(
                "s1",
                [a, NodeId::GROUND],
                voltage,
                7.0,
                Complex::ZERO,
                waveform,
            )
            .unwrap(),
        ))
        .unwrap();
    circuit
        .add_device(Box::new(
            Resistor::new("r1", [a, NodeId::GROUND], 1e3).unwrap(),
        ))
        .unwrap();
    circuit.finalize().unwrap();
    circuit
}

/// RHS of a transient load at `time` (no integration state is involved).
fn rhs(circuit: &Circuit, mode: AnalysisMode, forcing: Option<Forcing>) -> SpiceResult<Vec<f64>> {
    let n = circuit.unknown_count();
    let history = circuit.state_history().unwrap();
    let mut matrix = SparseMatrix::new(n, n);
    let mut rhs = Vector::zeros(n);
    circuit.load(
        &LoadRequest {
            mode,
            solution: &Vector::zeros(n),
            model_context: &ModelContext::default(),
            integration: None,
            history: &history,
            forcing,
        },
        &mut matrix,
        &mut rhs,
        &mut history.trial(),
    )?;
    Ok(rhs.as_slice().to_vec())
}

fn at(time: f64, limit: Limit) -> (AnalysisMode, Option<Forcing>) {
    (
        AnalysisMode::Transient { time, dt: 1e-6 },
        Some(Forcing {
            limit,
            timing: TransientTiming::new(1e-6, 1e-3).unwrap(),
        }),
    )
}

#[test]
fn transient_loads_use_the_waveform_with_the_requested_limit() {
    let step = Waveform::Step {
        before: 1.0,
        after: 3.0,
        time: 1e-4,
    };
    // Voltage source: the branch row (index 1) carries the forcing.
    let voltage = circuit(true, step.clone());
    for (time, limit, want) in [
        (5e-5, Limit::Right, 1.0),
        (1e-4, Limit::Left, 1.0),
        (1e-4, Limit::Right, 3.0),
        (2e-4, Limit::Left, 3.0),
    ] {
        let (mode, forcing) = at(time, limit);
        let b = rhs(&voltage, mode, forcing).unwrap();
        assert_eq!(b, [0.0, want], "t={time} {limit:?}");
    }
    // Current source: positive from the first terminal to the second, so a
    // source from `a` to ground draws current out of node `a`.
    let current = circuit(false, step);
    let (mode, forcing) = at(1e-4, Limit::Right);
    assert_eq!(rhs(&current, mode, forcing).unwrap(), [-3.0]);
    // DC loads still stamp the DC value, not the waveform.
    let b = rhs(&voltage, AnalysisMode::OperatingPoint, None).unwrap();
    assert_eq!(b, [0.0, 7.0]);
}

#[test]
fn unresolved_pulse_defaults_are_bound_by_the_forcing_timing() {
    // PULSE(0 4): TR defaults to the .tran step (1 us); it ramps to 4 V.
    let spec = PulseSpec {
        initial: 0.0,
        pulsed: 4.0,
        delay: None,
        rise: None,
        fall: None,
        width: None,
        period: None,
        count: None,
    };
    let source = circuit(true, Waveform::PulseDefaults(spec));
    let (mode, forcing) = at(5e-7, Limit::Right);
    assert_eq!(rhs(&source, mode, forcing).unwrap(), [0.0, 2.0]);
    let (mode, forcing) = at(2e-6, Limit::Right);
    assert_eq!(rhs(&source, mode, forcing).unwrap(), [0.0, 4.0]);
    // A pre-resolved pulse needs no defaults.
    let pulse = Pulse::new(0.0, 4.0, 0.0, 1e-6, 1e-6, 1e-4, 2e-4).unwrap();
    let source = circuit(true, Waveform::Pulse(pulse));
    let (mode, forcing) = at(5e-7, Limit::Right);
    assert_eq!(rhs(&source, mode, forcing).unwrap(), [0.0, 2.0]);
}

#[test]
fn transient_loads_without_forcing_or_in_ac_fail_explicitly() {
    let source = circuit(true, Waveform::Constant(1.0));
    let (mode, _) = at(1e-6, Limit::Right);
    let error = rhs(&source, mode, None).unwrap_err().to_string();
    assert!(error.contains("without a forcing context"), "{error}");
    let error = rhs(&source, AnalysisMode::Ac { frequency: 1.0 }, None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("AC"), "{error}");
}

#[test]
fn unintegrated_physical_dae_respects_time_and_source_event_side() {
    let circuit = circuit(
        true,
        Waveform::Step {
            before: 1.,
            after: 3.,
            time: 1e-4,
        },
    );
    let run = circuit.prepare_run();
    let mut dae = run
        .prepare_physical_dae(
            &ModelContext::default(),
            TransientTiming::new(1e-6, 1e-3).unwrap(),
        )
        .unwrap();
    let x = Vector::from(vec![2., 0.]);
    for (time, side, expected) in [
        (1e-4, Limit::Left, 1.),
        (1e-4, Limit::Right, -1.),
        (2e-4, Limit::Right, -1.),
        (0., Limit::Right, 1.),
    ] {
        let point = dae.evaluate(&x, time, side).unwrap();
        assert_eq!(point.residual.as_slice(), &[2e-3, expected]);
        assert_eq!(point.charge.as_slice(), &[0., 0.]);
        assert_eq!(point.conductance.get(0, 0), 1e-3);
        assert_eq!(point.conductance.get(1, 0), 1.);
    }
    assert!(dae.evaluate(&x, f64::NAN, Limit::Right).is_err());
    assert_eq!(
        rhs(&circuit, AnalysisMode::OperatingPoint, None).unwrap(),
        [0., 7.]
    );
}

#[test]
fn unintegrated_physical_dae_matches_nonlinear_charge_equations() -> SpiceResult<()> {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;
    let deck = parse_deck_text(
        Path::new("dae.cir"),
        "physical DAE\nv1 a 0 dc 7 PWL(0 1 1m 3)\nb1 b 0 v={v(a)^2+1000*time}\nc1 a 0 2u\nr1 b 0 1k\nc2 b 0 q={1u*(v(b)^2+1000*time)}\n.end\n",
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    circuit.finalize()?;
    let row = |name| {
        circuit
            .unknowns()
            .node_row(circuit.nodes().get(name).unwrap())
            .unwrap()
    };
    let branch = |name| {
        let i = circuit
            .devices()
            .iter()
            .position(|d| d.name() == name)
            .unwrap();
        circuit.branch_rows(i).unwrap().start
    };
    let (a, b, v, bv, q) = (row("a"), row("b"), branch("v1"), branch("b1"), branch("c2"));
    let mut x = Vector::zeros(circuit.unknown_count());
    for (row, value) in [
        (a, 2.),
        (b, 5.),
        (v, -2e-3),
        (bv, -6e-3),
        (q, 1e-3),
        (q + 1, 26e-6),
    ] {
        x.set(row, value)?;
    }
    let run = circuit.prepare_run();
    let mut dae = run.prepare_physical_dae(
        &config.context().model_context(),
        TransientTiming::new(1e-6, 1e-3)?,
    )?;
    let close = |actual: f64, expected: f64| {
        assert!(
            (actual - expected).abs() < 1e-12,
            "{actual:e} != {expected:e}"
        )
    };
    let point = dae.evaluate(&x, 5e-4, Limit::Right)?;
    for (r, expected) in [
        (a, -2e-3),
        (b, 0.),
        (v, 0.),
        (bv, 0.5),
        (q, 0.5e-6),
        (q + 1, 1e-3),
    ] {
        close(point.residual.as_slice()[r], expected);
    }
    close(point.charge.as_slice()[a], 4e-6);
    close(point.charge.as_slice()[q + 1], -26e-6);
    close(point.capacitance.get(a, a), 2e-6);
    close(point.capacitance.get(q + 1, q + 1), -1.);
    close(point.conductance.get(bv, a), -4.);
    close(point.conductance.get(q, b), -10e-6);
    let point = dae.evaluate(&x, 1e-3, Limit::Right)?;
    close(point.residual.as_slice()[v], -1.);
    close(point.residual.as_slice()[bv], 0.);
    close(point.residual.as_slice()[q], 0.);
    Ok(())
}

#[test]
fn exact_source_jets_choose_the_physical_side_at_corners_and_jumps() -> SpiceResult<()> {
    use ngspice_rs::devices::PwlSource;
    let waveform = Waveform::Pwl(PwlSource::new(
        vec![(0., 2.), (1., 3.), (2., 7.)],
        0.,
        Some(1.),
    )?);
    let cases = [
        (0., Limit::Left, [2., 0.]),
        (0., Limit::Right, [2., 1.]),
        (1., Limit::Left, [3., 1.]),
        (1., Limit::Right, [3., 4.]),
        (2., Limit::Left, [7., 4.]),
        (2., Limit::Right, [3., 4.]),
        (3., Limit::Left, [7., 4.]),
        (3., Limit::Right, [3., 4.]),
    ];
    for (t, side, expected) in cases {
        let jet = waveform.taylor_at(t, 1., 6, side)?;
        assert_eq!(&jet[..2], &expected);
        assert_eq!(&jet[2..], &[0.; 5]);
        assert_eq!(waveform.physical_value_at(t, side)?, expected[0]);
    }
    let extreme = Waveform::Pwl(PwlSource::plain(vec![(0., -1e308), (1e308, 1e308)])?);
    let jet = extreme.taylor_at(0., 1e-308, 1, Limit::Right)?;
    assert!((jet[1] / 2e-308 - 1.).abs() < 1e-14);
    let pulse = Waveform::Pulse(Pulse::new(1., 3., 0., 0., 1., 1., 4.)?);
    for (t, side, expected) in [
        (0., Limit::Left, [1., 0.]),
        (0., Limit::Right, [3., 0.]),
        (1., Limit::Left, [3., 0.]),
        (1., Limit::Right, [3., -2.]),
        (2., Limit::Left, [1., -2.]),
        (2., Limit::Right, [1., 0.]),
        (4., Limit::Left, [1., 0.]),
        (4., Limit::Right, [3., 0.]),
    ] {
        let jet = pulse.taylor_at(t, 1., 4, side)?;
        assert_eq!(&jet[..2], &expected);
        assert_eq!(&jet[2..], &[0.; 3]);
    }
    let circuit = circuit(true, pulse);
    let run = circuit.prepare_run();
    let mut dae =
        run.prepare_physical_dae(&ModelContext::default(), TransientTiming::new(0.1, 10.)?)?;
    let x = Vector::from(vec![3., -0.003]);
    let left = dae.evaluate(&x, 0., Limit::Left)?;
    assert_eq!(left.residual.as_slice(), &[0., 2.]);
    let right = dae.evaluate(&x, 0., Limit::Right)?;
    assert_eq!(right.residual.as_slice(), &[0., 0.]);
    Ok(())
}

#[test]
fn analytic_waveform_jets_do_not_differentiate_rounded_samples() -> SpiceResult<()> {
    use ngspice_rs::devices::{AmSpec, ExpSpec, FunctionSpec, SffmSpec, SineSpec};
    use std::f64::consts::TAU;
    let timing = TransientTiming::new(1e-6, 1e-2)?;
    let sine = Waveform::FunctionDefaults(FunctionSpec::Sine(SineSpec {
        offset: 2.,
        amplitude: 3.,
        frequency: Some(1000.),
        delay: None,
        damping: None,
        phase: None,
    }))
    .resolve(&timing)?;
    let scale = 1e-3;
    let slope = TAU * 1000. * scale;
    let actual = sine.taylor_at(0., scale, 6, Limit::Right)?;
    let expected = [
        2.,
        3. * slope,
        0.,
        -3. * slope.powi(3) / 6.,
        0.,
        3. * slope.powi(5) / 120.,
        0.,
    ];
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 1e-12, "{a:e} != {b:e}");
    }
    assert_eq!(
        sine.taylor_at(0., scale, 6, Limit::Left)?,
        vec![2., 0., 0., 0., 0., 0., 0.]
    );
    let exp = Waveform::FunctionDefaults(FunctionSpec::Exp(ExpSpec {
        initial: 1.,
        pulsed: 3.,
        rise_delay: Some(1e-3),
        rise_tau: Some(1e-3),
        fall_delay: Some(4e-3),
        fall_tau: Some(2e-3),
    }))
    .resolve(&timing)?;
    let actual = exp.taylor_at(1e-3, scale, 4, Limit::Right)?;
    for (a, b) in actual.iter().zip([1., 2., -1., 1. / 3., -1. / 12.]) {
        assert!((a - b).abs() < 1e-14);
    }
    let fm = Waveform::FunctionDefaults(FunctionSpec::Sffm(SffmSpec {
        offset: 1.,
        amplitude: 2.,
        carrier: Some(1000.),
        index: Some(1.),
        signal: Some(100.),
        delay: Some(1e-3),
        signal_phase: Some(15.),
        carrier_phase: Some(30.),
    }))
    .resolve(&timing)?;
    let am = Waveform::FunctionDefaults(FunctionSpec::Am(AmSpec {
        offset: 1.,
        carrier_amplitude: 2.,
        modulation_amplitude: Some(0.3),
        modulation_frequency: Some(100.),
        carrier_frequency: Some(1000.),
        delay: Some(1e-3),
        modulation_phase: Some(15.),
        carrier_phase: Some(30.),
    }))
    .resolve(&timing)?;
    let port = Waveform::PortSeries {
        baseline: Box::new(sine.clone()),
        tones: vec![(0.7, TAU * 900.)],
    };
    for waveform in [&sine, &exp, &fm, &am, &port] {
        let time = 2e-3;
        let scale = 1e-5;
        let jet = waveform.taylor_at(time, scale, 8, Limit::Right)?;
        for local in [-0.5_f64, 0., 0.5] {
            let reconstructed = jet
                .iter()
                .rev()
                .fold(0_f64, |sum, &coefficient| sum.mul_add(local, coefficient));
            let expected = waveform.value_at(time + scale * local, Limit::Right)?;
            assert!(
                (reconstructed - expected).abs() < 1e-12,
                "{reconstructed:e} != {expected:e}"
            );
        }
    }
    let circuit = circuit(false, sine);
    // The source's explicit DC value must not enter its derivative coefficients.
    let run = circuit.prepare_run();
    let dae = run.prepare_physical_dae(&ModelContext::default(), timing)?;
    let rhs = dae.forcing_taylor_rhs(0., 1e-3, 6, Limit::Right)?;
    for (actual, expected) in rhs.iter().zip(expected) {
        assert!((actual.as_slice()[0] + expected).abs() < 1e-12);
    }
    Ok(())
}

#[test]
fn physical_taylor_equations_preserve_nonlinear_coefficients_and_gradients() -> SpiceResult<()> {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;
    let deck = parse_deck_text(
        Path::new("jets.cir"),
        "jet DAE\nv1 a 0 PWL(0 2 1m 3)\nb1 b 0 v={v(a)^2+1000*time}\nc1 a 0 2u\nr1 b 0 1k\nc2 b 0 q={1u*(v(b)^2+1000*time)}\n.end\n",
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    circuit.finalize()?;
    let row = |name| {
        circuit
            .unknowns()
            .node_row(circuit.nodes().get(name).unwrap())
            .unwrap()
    };
    let branch = |name| {
        let i = circuit
            .devices()
            .iter()
            .position(|d| d.name() == name)
            .unwrap();
        circuit.branch_rows(i).unwrap().start
    };
    let (a, b, v, bv, q) = (row("a"), row("b"), branch("v1"), branch("b1"), branch("c2"));
    let mut state = vec![Vector::zeros(circuit.unknown_count()); 5];
    for (degree, values) in [
        (
            0,
            [
                (a, 2.),
                (b, 5.),
                (v, -2e-3),
                (bv, -6e-3),
                (q, 1e-3),
                (q + 1, 26e-6),
            ],
        ),
        (
            1,
            [
                (a, 1.),
                (b, 4.),
                (v, -1e-3),
                (bv, -1e-3),
                (q, 2e-3),
                (q + 1, 40e-6),
            ],
        ),
        (
            2,
            [
                (a, 0.5),
                (b, 2.),
                (v, 0.),
                (bv, 0.),
                (q, 0.),
                (q + 1, 36e-6),
            ],
        ),
    ] {
        for (r, value) in values {
            state[degree].set(r, value)?;
        }
    }
    let run = circuit.prepare_run();
    let mut dae = run.prepare_physical_dae(
        &config.context().model_context(),
        TransientTiming::new(1e-6, 1e-3)?,
    )?;
    let time = 0.;
    let scale = 1e-3;
    let taylor = dae.evaluate_taylor(&state, time, scale, Limit::Right)?;
    let close = |a: f64, b: f64| {
        assert!(
            (a - b).abs() <= 2e-11 * a.abs().max(b.abs()).max(1e-9),
            "{a:e} != {b:e}"
        )
    };
    // F_b = v(b)-v(a)^2-1000*time.
    for (degree, expected) in [1., -1., -1., -1., -0.25].into_iter().enumerate() {
        close(taylor.residual[degree].as_slice()[bv], expected);
    }
    // The explicitly retained charge law and its state Jacobian.
    for (degree, expected) in [1e-6, -1e-6, 0., -16e-6, -4e-6].into_iter().enumerate() {
        close(taylor.residual[degree].as_slice()[q], expected);
    }
    for (degree, expected) in [-4., -2., -1., 0., 0.].into_iter().enumerate() {
        close(taylor.conductance[degree].get(bv, a), expected);
    }
    for (degree, expected) in [-10e-6, -8e-6, -4e-6, 0., 0.].into_iter().enumerate() {
        close(taylor.conductance[degree].get(q, b), expected);
    }
    // Reconstruct every physical row and Jacobian at independently sampled
    // interior points. These polynomial laws have no omitted higher terms.
    for tau in [0.125_f64, 0.25, 0.5] {
        let mut x = Vector::zeros(circuit.unknown_count());
        for k in (0..state.len()).rev() {
            for (value, &coefficient) in x.as_mut_slice().iter_mut().zip(state[k].as_slice()) {
                *value = value.mul_add(tau, coefficient);
            }
        }
        let actual = dae.evaluate(&x, time + scale * tau, Limit::Right)?;
        for r in 0..x.len() {
            let reconstruct = |jets: &[Vector]| {
                jets.iter()
                    .rev()
                    .fold(0_f64, |sum, jet| sum.mul_add(tau, jet.as_slice()[r]))
            };
            close(reconstruct(&taylor.residual), actual.residual.as_slice()[r]);
            close(reconstruct(&taylor.charge), actual.charge.as_slice()[r]);
            for c in 0..x.len() {
                let reconstruct = |jets: &[SparseMatrix]| {
                    jets.iter()
                        .rev()
                        .fold(0_f64, |sum, jet| sum.mul_add(tau, jet.get(r, c)))
                };
                close(
                    reconstruct(&taylor.conductance),
                    actual.conductance.get(r, c),
                );
                close(
                    reconstruct(&taylor.capacitance),
                    actual.capacitance.get(r, c),
                );
            }
        }
    }
    Ok(())
}

#[test]
fn physical_taylor_left_and_right_germs_use_same_time_coordinate() -> SpiceResult<()> {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use std::path::Path;
    let deck = parse_deck_text(
        Path::new("germs.cir"),
        "physical germs\nv1 a 0 PWL(0 -1 1m 0 2m 2)\nb1 b 0 v={abs(v(a))+v(a)^3+1000*time}\nr1 b 0 1k\n.end\n",
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    circuit.finalize()?;
    let a = circuit
        .unknowns()
        .node_row(circuit.nodes().get("a").unwrap())
        .unwrap();
    let b = circuit
        .unknowns()
        .node_row(circuit.nodes().get("b").unwrap())
        .unwrap();
    let i = circuit
        .devices()
        .iter()
        .position(|device| device.name() == "b1")
        .unwrap();
    let branch = circuit.branch_rows(i).unwrap().start;
    let run = circuit.prepare_run();
    let dae = run.prepare_physical_dae(
        &config.context().model_context(),
        TransientTiming::new(1e-6, 2e-3)?,
    )?;
    let mut state = vec![Vector::zeros(circuit.unknown_count()); 4];
    state[0].set(b, 1.)?;
    // Use identical state jets on each side; only the one-sided
    // constitutive and independent forcing choices may differ.
    state[1].set(a, 1.)?;
    state[1].set(b, 1.)?;
    for (limit, slope, source_slope) in [(Limit::Left, 1., 1.), (Limit::Right, -1., 2.)] {
        let point = dae.evaluate_taylor(&state, 1e-3, 1e-3, limit)?;
        assert_eq!(point.residual[0].as_slice()[branch], 0.);
        assert_eq!(point.residual[1].as_slice()[branch], slope);
        assert_eq!(point.residual[3].as_slice()[branch], -1.);
        assert_eq!(point.conductance[0].get(branch, a), slope);
        assert_eq!(point.conductance[2].get(branch, a), -3.);
        let forcing = dae.forcing_taylor_rhs(1e-3, 1e-3, 1, limit)?;
        let source = circuit
            .devices()
            .iter()
            .position(|device| device.name() == "v1")
            .unwrap();
        assert_eq!(
            forcing[1].as_slice()[circuit.branch_rows(source).unwrap().start],
            source_slope
        );
    }
    Ok(())
}

#[test]
fn delayed_pulse_germs_agree_with_every_scheduled_corner() -> SpiceResult<()> {
    let delay = 17e-6;
    let rise = 3e-6;
    let width = 7e-6;
    let fall = 11e-6;
    let period = 31e-6;
    let waveform = Waveform::Pulse(Pulse::new(2., 3., delay, rise, fall, width, period)?);
    let offsets = [0., rise, rise + width, rise + width + fall];
    let slopes = [0., 1e-6 / rise, 0., -1e-6 / fall, 0.];
    let levels = [2., 3., 3., 2.];
    let corners = waveform
        .breakpoints_in(0., delay + 8. * period)?
        .take(32)
        .collect::<Vec<_>>();
    assert_eq!(corners.len(), 32);
    for (index, time) in corners.into_iter().enumerate() {
        let edge = index % 4;
        let cycle = f64::from(u32::try_from(index / 4).unwrap());
        assert_eq!(time, delay + (period * cycle + offsets[edge]));
        for (limit, slope) in [
            (Limit::Left, slopes[edge]),
            (Limit::Right, slopes[edge + 1]),
        ] {
            let actual = waveform.taylor_at(time, 1e-6, 2, limit)?;
            assert_eq!(actual[0], levels[edge], "{index} {limit:?}");
            assert!(
                (actual[1] - slope).abs() < 1e-15,
                "{index} {limit:?}: {:?}",
                actual
            );
            assert_eq!(actual[2], 0.);
        }
        // Adjacent representable times must not be snapped to the corner.
        let left = waveform.taylor_at(time.next_down(), 1e-6, 1, Limit::Right)?;
        let right = waveform.taylor_at(time.next_up(), 1e-6, 1, Limit::Left)?;
        assert!((left[1] - slopes[edge]).abs() < 1e-15);
        assert!((right[1] - slopes[edge + 1]).abs() < 1e-15);
    }
    Ok(())
}

#[test]
fn delayed_repeating_pwl_germs_follow_exact_iterator_coordinates() -> SpiceResult<()> {
    use ngspice_rs::devices::PwlSource;
    let delay = 17e-6;
    let first = 5e-6;
    let middle = 15e-6;
    let last = 25e-6;
    let waveform = Waveform::Pwl(PwlSource::new(
        vec![(0., 1.), (first, 2.), (middle, 3.), (last, 7.)],
        delay,
        Some(first),
    )?);
    let corners = waveform
        .breakpoints_in(0., delay + last + 8. * (last - first))?
        .take(20)
        .collect::<Vec<_>>();
    for (index, time) in corners.into_iter().enumerate() {
        let (l, r) = match index {
            0 => ([1., 0.], [1., 1e-6 / first]),
            1 => ([2., 1e-6 / first], [2., 1e-6 / (middle - first)]),
            _ if index % 2 == 0 => ([3., 1e-6 / (middle - first)], [3., 4e-6 / (last - middle)]),
            _ => ([7., 4e-6 / (last - middle)], [2., 1e-6 / (middle - first)]),
        };
        for (limit, expected) in [(Limit::Left, l), (Limit::Right, r)] {
            let actual = waveform.taylor_at(time, 1e-6, 2, limit)?;
            assert_eq!(actual[0], expected[0], "{index} {time:e} {limit:?}");
            assert!(
                (actual[1] - expected[1]).abs() < 1e-15,
                "{index} {limit:?}: {actual:?}"
            );
            assert_eq!(actual[2], 0.);
        }
        for (at, slope, limit) in [
            (time.next_down(), l[1], Limit::Right),
            (time.next_up(), r[1], Limit::Left),
        ] {
            let jet = waveform.taylor_at(at, 1e-6, 1, limit)?;
            assert!(
                (jet[1] - slope).abs() < 1e-15,
                "{index} {at:e} {limit:?}: {jet:?}"
            );
        }
    }
    Ok(())
}

#[test]
fn truncated_pulse_retains_correct_left_limit_at_period_and_count_cut() -> SpiceResult<()> {
    let waveform =
        Waveform::Pulse(Pulse::new(2., 4., 17e-6, 10e-6, 10e-6, 5e-6, 7e-6)?.with_count(2.5)?);
    let corners = waveform.breakpoints_in(0., 100e-6)?.collect::<Vec<_>>();
    for &time in corners.iter().take(3).skip(1) {
        let left = waveform.taylor_at(time, 1e-6, 1, Limit::Left)?;
        let right = waveform.taylor_at(time, 1e-6, 1, Limit::Right)?;
        assert!((left[0] - 3.4).abs() < 1e-14);
        assert_eq!(right[0], 2.);
        assert!((left[1] - 0.2).abs() < 1e-15);
        assert!((right[1] - 0.2).abs() < 1e-15);
    }
    let cut = 17e-6 + 2.5 * 7e-6;
    let left = waveform.taylor_at(cut, 1e-6, 1, Limit::Left)?;
    let right = waveform.taylor_at(cut, 1e-6, 1, Limit::Right)?;
    assert!((left[0] - 2.7).abs() < 1e-14);
    assert!((left[1] - 0.2).abs() < 1e-15);
    assert_eq!(right, [2., 0.]);
    Ok(())
}
