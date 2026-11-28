//! Transactional mixed-signal event tests.

use ngspice_rs::analysis::digital::{Engine, Limits, Network};
use ngspice_rs::devices::code_model::{
    Adc, DFlipFlop, Dac, DelayMode, Delays, Drive, Gate, GateKind, Logic, Strength,
};
use ngspice_rs::primitives::SpiceResult;

#[test]
fn simultaneous_drives_resolve_before_fanout() -> SpiceResult<()> {
    let mut network = Network::default();
    let bus = network.add_net();
    let out = network.add_net();
    let a = network.add_source(bus)?;
    let b = network.add_source(bus)?;
    network.add_model(
        Gate::new(
            GateKind::Inverter,
            1,
            Delays::new(0., 0.)?,
            DelayMode::Transport,
        )?,
        &[bus],
        &[out],
    )?;
    let mut engine = Engine::new(network, Limits::default())?;
    let mut trial = engine.trial();
    trial.schedule(a, 0., Drive::strong(Logic::One))?;
    trial.schedule(b, 0., Drive::strong(Logic::Zero))?;
    trial.advance_to(0.)?;
    assert_eq!(trial.value(bus), Some(Drive::strong(Logic::Unknown)));
    assert_eq!(trial.value(out), Some(Drive::strong(Logic::Unknown)));
    trial.schedule(
        b,
        1.,
        Drive {
            logic: Logic::Zero,
            strength: Strength::RESISTIVE,
        },
    )?;
    trial.advance_to(1.)?;
    assert_eq!(trial.value(bus), Some(Drive::strong(Logic::One)));
    assert_eq!(trial.value(out), Some(Drive::strong(Logic::Zero)));
    assert_eq!(
        trial
            .changes()
            .iter()
            .filter(|change| change.net == out)
            .count(),
        2
    );
    trial.commit()?;
    assert_eq!(engine.stats().evaluations, 2);
    Ok(())
}

fn pulse(mode: DelayMode) -> SpiceResult<Vec<(f64, Logic)>> {
    let mut network = Network::default();
    let input = network.add_net();
    let out = network.add_net();
    let source = network.add_source(input)?;
    network.add_model(
        Gate::new(GateKind::Buffer, 1, Delays::new(2., 2.)?, mode)?,
        &[input],
        &[out],
    )?;
    let mut engine = Engine::new(network, Limits::default())?;
    let mut trial = engine.trial();
    for (time, value) in [(0., Logic::Zero), (4., Logic::One), (5., Logic::Zero)] {
        trial.schedule(source, time, Drive::strong(value))?;
    }
    trial.advance_to(10.)?;
    Ok(trial
        .changes()
        .iter()
        .filter(|change| change.net == out)
        .map(|change| (change.time, change.drive.logic))
        .collect())
}

#[test]
fn inertial_cancels_short_pulses_and_transport_preserves_them() -> SpiceResult<()> {
    assert_eq!(pulse(DelayMode::Inertial)?, vec![(0., Logic::Zero)]);
    assert_eq!(
        pulse(DelayMode::Transport)?,
        vec![(0., Logic::Zero), (6., Logic::One), (7., Logic::Zero)]
    );
    Ok(())
}

#[test]
fn rejected_trial_restores_queue_and_sequential_state() -> SpiceResult<()> {
    let mut network = Network::default();
    let data = network.add_net();
    let clock = network.add_net();
    let q = network.add_net();
    let nq = network.add_net();
    let d = network.add_source(data)?;
    let c = network.add_source(clock)?;
    network.add_model(
        DFlipFlop {
            initial: Logic::Zero,
            delays: Delays::new(0.25, 0.25)?,
            mode: DelayMode::Transport,
        },
        &[data, clock],
        &[q, nq],
    )?;
    let mut engine = Engine::new(network, Limits::default())?;
    let mut init = engine.trial();
    init.schedule(d, 0., Drive::strong(Logic::One))?;
    init.schedule(c, 0., Drive::strong(Logic::Zero))?;
    let edge = init.schedule(c, 1., Drive::strong(Logic::One))?;
    init.advance_to(0.5)?;
    init.commit()?;
    assert_eq!(engine.value(q), Some(Drive::strong(Logic::Zero)));
    let before = engine.stats();
    {
        let mut rejected = engine.trial();
        rejected.advance_to(2.)?;
        assert_eq!(rejected.value(q), Some(Drive::strong(Logic::One)));
        assert_eq!(rejected.value(nq), Some(Drive::strong(Logic::Zero)));
    }
    assert_eq!(engine.stats(), before);
    assert_eq!(engine.next_event_time(), Some(1.));
    let mut retry = engine.trial();
    assert!(retry.cancel(edge));
    retry.advance_to(2.)?;
    assert_eq!(retry.value(q), Some(Drive::strong(Logic::Zero)));
    retry.schedule(c, 3., Drive::strong(Logic::One))?;
    retry.advance_to(3.25)?;
    retry.commit()?;
    assert_eq!(engine.value(q), Some(Drive::strong(Logic::One)));
    Ok(())
}

#[test]
fn zero_delay_oscillation_is_bounded_and_cannot_commit() -> SpiceResult<()> {
    let mut network = Network::default();
    let bus = network.add_net();
    let source = network.add_source(bus)?;
    network.add_model(
        Gate::new(
            GateKind::Inverter,
            1,
            Delays::new(0., 0.)?,
            DelayMode::Transport,
        )?,
        &[bus],
        &[bus],
    )?;
    let mut engine = Engine::new(
        network,
        Limits {
            delta_cycles: 8,
            work: 100,
        },
    )?;
    // A weak initial drive seeds the otherwise unknown self-inverting net.
    let mut trial = engine.trial();
    trial.schedule(
        source,
        0.,
        Drive {
            logic: Logic::Zero,
            strength: Strength::RESISTIVE,
        },
    )?;
    assert!(trial.advance_to(0.).is_err());
    assert!(trial.commit().is_err());
    assert_eq!(engine.stats().events, 0);
    Ok(())
}

#[test]
fn adc_thresholds_and_dac_targets() -> SpiceResult<()> {
    let adc = Adc::new(0.8, 2.)?;
    assert_eq!(adc.sample(1.2, Logic::One)?, Logic::Unknown);
    assert_eq!(adc.crossing((0., 0.), (3., 3.), Logic::Zero)?, Some(0.8));
    assert_eq!(adc.crossing((0., 0.8), (1., 1.2), Logic::Zero)?, Some(0.));
    assert_eq!(adc.crossing((0., 2.), (1., 1.2), Logic::One)?, Some(0.));
    assert_eq!(adc.crossing((0., 0.8), (1., 0.8), Logic::Zero)?, None);
    let schmitt = Adc::new(2., 0.8)?;
    assert_eq!(schmitt.crossing((0., 0.8), (1., 1.2), Logic::Zero)?, None);
    assert_eq!(schmitt.sample(1.2, Logic::One)?, Logic::One);
    assert_eq!(schmitt.sample(1.2, Logic::Zero)?, Logic::Zero);
    assert_eq!(schmitt.sample(2., Logic::Zero)?, Logic::One);
    let dac = Dac {
        low: 0.,
        high: 3.3,
        unknown: 1.65,
    };
    assert_eq!(dac.target(Logic::Unknown)?, 1.65);
    Ok(())
}

#[test]
fn discarded_event_handles_cannot_cancel_replacement_events() -> SpiceResult<()> {
    let mut network = Network::default();
    let net = network.add_net();
    let source = network.add_source(net)?;
    let mut engine = Engine::new(network, Limits::default())?;
    let stale = engine
        .trial()
        .schedule(source, 1., Drive::strong(Logic::Zero))?;
    let mut trial = engine.trial();
    let current = trial.schedule(source, 1., Drive::strong(Logic::One))?;
    assert_ne!(stale, current);
    assert!(!trial.cancel(stale));
    trial.advance_to(1.)?;
    assert_eq!(trial.value(net), Some(Drive::strong(Logic::One)));
    trial.commit()
}

#[test]
fn unequal_transport_delays_supersede_later_output() -> SpiceResult<()> {
    let mut network = Network::default();
    let input = network.add_net();
    let out = network.add_net();
    let source = network.add_source(input)?;
    network.add_model(
        Gate::new(
            GateKind::Buffer,
            1,
            Delays::new(5., 1.)?,
            DelayMode::Transport,
        )?,
        &[input],
        &[out],
    )?;
    let mut engine = Engine::new(network, Limits::default())?;
    let mut trial = engine.trial();
    for (time, value) in [(0., Logic::Zero), (1., Logic::One), (2., Logic::Zero)] {
        trial.schedule(source, time, Drive::strong(value))?;
    }
    trial.advance_to(10.)?;
    assert_eq!(trial.value(out), Some(Drive::strong(Logic::Zero)));
    assert_eq!(trial.changes().iter().filter(|c| c.net == out).count(), 1);
    trial.commit()
}

#[test]
fn analog_ramp_drives_delayed_inverter_and_slewing_dac() -> SpiceResult<()> {
    use ngspice_rs::analysis::digital::{AdcBinding, MixedSignal};
    use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, mixed_signal_transient};
    use ngspice_rs::devices::Circuit;
    use ngspice_rs::devices::code_model::DacConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use ngspice_rs::primitives::{AnalysisKind, NodeId};
    use std::path::Path;

    let deck = parse_deck_text(
        Path::new("mixed.cir"),
        "mixed comparator\nv1 in 0 pwl(0 0 1m 3 2m 0)\nr1 out 0 1k\n.end\n",
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let mut circuit = Circuit::from_netlist(&netlist)?;
    let input = circuit.nodes().get("in").unwrap();
    let output = circuit.nodes().get("out").unwrap();
    let mut network = Network::default();
    let digital_in = network.add_net();
    let digital_out = network.add_net();
    let adc = network.add_source(digital_in)?;
    network.add_model(
        Gate::new(
            GateKind::Inverter,
            1,
            Delays::new(20e-6, 20e-6)?,
            DelayMode::Transport,
        )?,
        &[digital_in],
        &[digital_out],
    )?;
    let mut mixed = MixedSignal::new(Engine::new(network, Limits::default())?);
    mixed.add_adc(AdcBinding {
        terminals: [input, NodeId::GROUND],
        source: adc,
        thresholds: Adc::new(2., 1.)?,
        delays: Delays::new(10e-6, 10e-6)?,
        mode: DelayMode::Transport,
    })?;
    mixed.add_dac(
        &mut circuit,
        "aout",
        [output, NodeId::GROUND],
        digital_out,
        DacConfig::new(
            Dac {
                low: 0.,
                high: 3.,
                unknown: 1.5,
            },
            Delays::new(60e-6, 60e-6)?,
        )?,
    )?;
    let plan = AnalysisInput::with_arguments(AnalysisKind::Transient, ["40u", "2m"]).resolve()?;
    let (plot, stats) =
        mixed_signal_transient(&mut circuit, &plan, &AnalysisContext::default(), &mut mixed)?;
    let changes: Vec<_> = mixed
        .changes()
        .iter()
        .filter(|change| change.net == digital_out)
        .collect();
    assert_eq!(changes.len(), 3, "{changes:?}");
    assert_eq!(changes[0].drive.logic, Logic::One);
    assert_eq!(changes[1].drive.logic, Logic::Zero);
    assert_eq!(changes[2].drive.logic, Logic::One);
    let falling = 2e-3 / 3. + 30e-6;
    let rising = 5e-3 / 3. + 30e-6;
    assert!((changes[1].time - falling).abs() < 1e-9, "{changes:?}");
    assert!((changes[2].time - rising).abs() < 1e-9, "{changes:?}");
    assert!(stats.rejected > 0);
    assert!(mixed.engine().stats().evaluations > 0);
    for i in 0..plot.point_count() {
        let t = plot.value("time", i).unwrap().re;
        let actual = plot.value("v(out)", i).unwrap().re;
        let expected = 3.
            * (1. - ((t - falling) / 60e-6).clamp(0., 1.) + ((t - rising) / 60e-6).clamp(0., 1.));
        assert!(
            (actual - expected).abs() < 1e-4,
            "t={t:e}: {actual} != {expected}"
        );
    }
    Ok(())
}

#[test]
fn analog_code_models_share_companion_history_and_ac_pencil() -> SpiceResult<()> {
    use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, companion_transient};
    use ngspice_rs::devices::code_model::analog::{AnalogCodeModel, Dynamics};
    use ngspice_rs::devices::{Circuit, ModelContext};
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use ngspice_rs::primitives::{AnalysisKind, NodeId};
    use std::path::Path;

    let netlist = Parser::new().parse_deck(&parse_deck_text(Path::new("analog.cir"),
        "native analog models\nv1 in 0 dc 0 pwl(0 0 1u 1 2m 1) ac 1\nr1 amp 0 1k\nr2 filt 0 1k\nr3 integ 0 1k\n.end\n"))?;
    let mut circuit = Circuit::from_netlist(&netlist)?;
    let input = circuit.nodes().get("in").unwrap();
    for (name, node, dynamics, gain, offset) in [
        ("again", "amp", Dynamics::Gain, -2., 0.5),
        (
            "afilter",
            "filt",
            Dynamics::LowPass {
                time_constant: 100e-6,
            },
            2.,
            0.,
        ),
        (
            "aint",
            "integ",
            Dynamics::Integrator {
                initial_output: 0.1,
            },
            1000.,
            0.,
        ),
    ] {
        let output = circuit.nodes().get(node).unwrap();
        circuit.add_device(Box::new(AnalogCodeModel::new(
            name,
            [output, NodeId::GROUND],
            [input, NodeId::GROUND],
            dynamics,
            gain,
            offset,
        )?))?;
    }
    circuit.finalize()?;
    let plan = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m", "0", "1u"])
        .resolve()?;
    let (plot, _) = companion_transient(&mut circuit, &plan, &AnalysisContext::default())?;
    let last = plot.point_count() - 1;
    let t = plot.value("time", last).unwrap().re;
    let v = |node| plot.value(node, last).unwrap().re;
    assert!((v("v(amp)") + 1.5).abs() < 1e-10);
    // For a linear ramp of length r followed by a unit step hold, the
    // first-order response is 2 - 2*tau/r*(1-exp(-r/tau))*exp(-(t-r)/tau).
    let tau: f64 = 100e-6;
    let r: f64 = 1e-6;
    let filtered = 2. + 2. * tau / r * (-r / tau).exp_m1() * (-(t - r) / tau).exp();
    assert!((v("v(filt)") - filtered).abs() < 2e-4);
    assert!(
        (v("v(integ)") - (0.1 + 1000. * (t - r / 2.))).abs() < 2e-5,
        "actual integrator voltage {}",
        v("v(integ)")
    );
    let ac = circuit.linear_system_with_context(&ModelContext::default())?;
    for (name, node, a, e) in [
        ("again", "amp", 1., 0.),
        ("afilter", "filt", 1., tau),
        ("aint", "integ", 0., 1.),
    ] {
        let index = circuit
            .devices()
            .iter()
            .position(|dev| dev.name() == name)
            .unwrap();
        let branch = circuit.branch_rows(index).unwrap().start;
        let row = circuit
            .unknowns()
            .node_row(circuit.nodes().get(node).unwrap())
            .unwrap();
        assert_eq!(ac.a.get(branch, row), a);
        assert_eq!(ac.e.get(branch, row), e);
    }
    let ac_plan =
        AnalysisInput::with_arguments(AnalysisKind::Ac, ["lin", "3", "100", "10000"]).resolve()?;
    let response = ac_plan.run(&mut circuit, &AnalysisContext::default())?;
    for row in 0..response.point_count() {
        let w = std::f64::consts::TAU * response.value("frequency", row).unwrap().re;
        let x = w * tau;
        let gain = response.value("v(amp)", row).unwrap();
        let filter = response.value("v(filt)", row).unwrap();
        let integral = response.value("v(integ)", row).unwrap();
        assert!((gain.re + 2.).abs() < 1e-10 && gain.im.abs() < 1e-10);
        assert!((filter.re - 2. / (1. + x * x)).abs() < 1e-10);
        assert!((filter.im + 2. * x / (1. + x * x)).abs() < 1e-10);
        assert!(integral.re.abs() < 1e-10 && (integral.im + 1000. / w).abs() < 1e-10);
    }
    Ok(())
}

fn run_feedback() -> SpiceResult<(ngspice_rs::analysis::Plot, Vec<f64>)> {
    use ngspice_rs::analysis::digital::{AdcBinding, MixedSignal};
    use ngspice_rs::analysis::{AnalysisContext, RunConfig, mixed_signal_transient};
    use ngspice_rs::devices::code_model::DacConfig;
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use ngspice_rs::primitives::NodeId;
    use std::path::Path;

    let deck = parse_deck_text(
        Path::new("feedback.cir"),
        "mixed RC feedback\nr1 out cap 1k\nc1 cap 0 100n\n.ic v(cap)=0\n.tran 5u 600u 0 500n\n.end\n",
    );
    let netlist = Parser::new().parse_deck(&deck)?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    let plan = config.plan_for(&netlist.analyses[0])?;
    let cap = circuit.nodes().get("cap").unwrap();
    let out = circuit.nodes().get("out").unwrap();
    let mut network = Network::default();
    let din = network.add_net();
    let dout = network.add_net();
    let source = network.add_source(din)?;
    let delay = Delays::new(500e-9, 500e-9)?;
    network.add_model(
        Gate::new(GateKind::Inverter, 1, delay, DelayMode::Transport)?,
        &[din],
        &[dout],
    )?;
    let mut mixed = MixedSignal::new(Engine::new(network, Limits::default())?);
    mixed.add_adc(AdcBinding {
        terminals: [cap, NodeId::GROUND],
        source,
        thresholds: Adc::new(2., 1.)?,
        delays: delay,
        mode: DelayMode::Transport,
    })?;
    mixed.add_dac(
        &mut circuit,
        "adac",
        [out, NodeId::GROUND],
        dout,
        DacConfig::new(
            Dac {
                low: 0.,
                high: 3.,
                unknown: 1.5,
            },
            Delays::new(1e-6, 1e-6)?,
        )?,
    )?;
    let (plot, _) =
        mixed_signal_transient(&mut circuit, &plan, &AnalysisContext::default(), &mut mixed)?;
    let transitions = mixed
        .changes()
        .iter()
        .filter(|change| change.net == dout && change.time > 0.)
        .map(|change| change.time + 500e-9)
        .collect();
    Ok((plot, transitions))
}

#[test]
fn mixed_schmitt_feedback_oscillates_with_physical_delays() -> SpiceResult<()> {
    let (plot, transitions) = run_feedback()?;
    assert_eq!(plot.value("v(cap)", 0).unwrap().re, 0.);
    assert_eq!(plot.value("v(out)", 0).unwrap().re, 3.);
    assert_eq!(transitions.len(), 7, "{transitions:?}");
    // RC rises from zero to 2 V with a 3 V asymptote. Later half-cycles
    // traverse the Schmitt band with finite bridge propagation and slew.
    assert!(
        (transitions[0] - (100e-6 * 3_f64.ln() + 1.5e-6)).abs() < 100e-9,
        "{transitions:?}"
    );
    for pair in transitions.windows(2) {
        assert!(
            (pair[1] - pair[0] - 71.575e-6).abs() < 200e-9,
            "{transitions:?}"
        );
    }
    for point in 0..plot.point_count() {
        let cap = plot.value("v(cap)", point).unwrap().re;
        let out = plot.value("v(out)", point).unwrap().re;
        assert!((-1e-8..=3. + 1e-8).contains(&cap));
        assert!((-1e-8..=3. + 1e-8).contains(&out));
    }
    Ok(())
}

#[test]
fn unrelated_digital_events_do_not_force_analog_steps() -> SpiceResult<()> {
    use ngspice_rs::analysis::digital::MixedSignal;
    use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, mixed_signal_transient};
    use ngspice_rs::devices::{Circuit, code_model::DacConfig};
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use ngspice_rs::primitives::{AnalysisKind, NodeId};
    use std::path::Path;

    let run = |event_count: u32| -> SpiceResult<_> {
        let deck = parse_deck_text(
            Path::new("density.cir"),
            "event density\nr1 out 0 1k\n.end\n",
        );
        let netlist = Parser::new().parse_deck(&deck)?;
        let mut circuit = Circuit::from_netlist(&netlist)?;
        let out = circuit.nodes().get("out").unwrap();
        let mut network = Network::default();
        let dac = network.add_net();
        let unrelated = network.add_net();
        let a = network.add_source(dac)?;
        let b = network.add_source(unrelated)?;
        let mut engine = Engine::new(network, Limits::default())?;
        let mut init = engine.trial();
        init.schedule(a, 0., Drive::strong(Logic::One))?;
        for i in 0..event_count {
            init.schedule(
                b,
                1e-3 * f64::from(i + 1) / f64::from(event_count + 1),
                Drive::strong(if i % 2 == 0 { Logic::Zero } else { Logic::One }),
            )?;
        }
        init.commit()?;
        let mut mixed = MixedSignal::new(engine);
        mixed.add_dac(
            &mut circuit,
            "adac",
            [out, NodeId::GROUND],
            dac,
            DacConfig::new(
                Dac {
                    low: 0.,
                    high: 3.,
                    unknown: 1.5,
                },
                Delays::new(1e-6, 1e-6)?,
            )?,
        )?;
        let plan =
            AnalysisInput::with_arguments(AnalysisKind::Transient, ["20u", "1m"]).resolve()?;
        let (plot, stats) =
            mixed_signal_transient(&mut circuit, &plan, &AnalysisContext::default(), &mut mixed)?;
        assert_eq!(mixed.engine().stats().events, event_count as usize + 1);
        assert!(
            plot.column("v(out)")
                .unwrap()
                .iter()
                .all(|v| (v.re - 3.).abs() < 1e-10)
        );
        Ok((
            plot.column("time")
                .unwrap()
                .iter()
                .map(|v| v.re)
                .collect::<Vec<_>>(),
            stats,
        ))
    };
    let quiet = run(0)?;
    let dense = run(10_000)?;
    assert_eq!(
        quiet, dense,
        "unrelated digital events changed the analog schedule"
    );
    Ok(())
}

fn run_xspice(deck: &str) -> ngspice_rs::analysis::Plot {
    use ngspice_conformance::ngspice::Scratch;
    use ngspice_conformance::process::{ProcessLimits, run};
    use ngspice_rs::analysis::RawFile;
    use std::fs;

    let scratch = Scratch::new("mixed-signal-reference").unwrap();
    let dir = scratch.path();
    fs::write(dir.join("reference.cir"), deck).unwrap();
    let binary = std::env::var_os("NGSPICE_BIN").expect("set NGSPICE_BIN");
    let result = run(binary, dir, &["-b", "reference.cir"], ProcessLimits::ORACLE).unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    let raw = RawFile::parse(&fs::read_to_string(dir.join("result.raw")).unwrap()).unwrap();
    raw.plots[0].plot.clone()
}

/// The C reference uses the same RC feedback circuit, thresholds and bridge
/// delays, but a ten-times-finer maximum analog step.
#[test]
#[ignore = "requires NGSPICE_BIN pointing to XSPICE-enabled ngspice"]
fn schmitt_feedback_matches_xspice() -> SpiceResult<()> {
    let (_, ours) = run_feedback()?;
    let plot = run_xspice(
        "Mixed RC feedback Schmitt oscillator\n\
         a1 [cap] [din] adc\n\
         a2 din dout inv\n\
         a3 [dout] [out] dac\n\
         r1 out cap 1k\n\
         c1 cap 0 100n\n\
         .ic v(cap)=0\n\
         .model adc adc_bridge(in_low=2 in_high=1 rise_delay=500n fall_delay=500n)\n\
         .model inv d_inverter(rise_delay=500n fall_delay=500n)\n\
         .model dac dac_bridge(out_low=0 out_high=3 out_undef=1.5 t_rise=1u t_fall=1u)\n\
         .control\nset filetype=ascii\ntran 5u 600u 0 50n\nwrite result.raw\nquit\n.endc\n.end\n",
    );
    let time = plot.column("time").unwrap();
    let output = plot.column("v(out)").unwrap();
    let mut reference = Vec::new();
    for i in 1..time.len() {
        let a = output[i - 1].re;
        let b = output[i].re;
        if (a < 1.5 && b >= 1.5) || (a > 1.5 && b <= 1.5) {
            let fraction = (1.5 - a) / (b - a);
            reference.push(fraction.mul_add(time[i].re - time[i - 1].re, time[i - 1].re));
        }
    }
    assert_eq!(reference.len(), ours.len(), "{reference:?} vs {ours:?}");
    for (a, b) in ours.iter().zip(&reference) {
        assert!((a - b).abs() < 200e-9, "{ours:?} vs {reference:?}");
    }
    println!("Rust transition midpoints: {ours:?}; XSPICE: {reference:?}");
    Ok(())
}

#[test]
fn dff_retains_on_unknown_clock_and_captures_on_transition_to_one() -> SpiceResult<()> {
    let mut network = Network::default();
    let d = network.add_net();
    let clock = network.add_net();
    let q = network.add_net();
    let nq = network.add_net();
    let data = network.add_source(d)?;
    let clk = network.add_source(clock)?;
    network.add_model(
        DFlipFlop {
            initial: Logic::Zero,
            delays: Delays::new(3e-6, 3e-6)?,
            mode: DelayMode::Transport,
        },
        &[d, clock],
        &[q, nq],
    )?;
    let mut engine = Engine::new(network, Limits::default())?;
    let mut trial = engine.trial();
    for (time, logic) in [
        (0., Logic::One),
        (30e-6, Logic::Unknown),
        (60e-6, Logic::Zero),
    ] {
        trial.schedule(data, time, Drive::strong(logic))?;
    }
    for (time, logic) in [
        (0., Logic::Zero),
        (5e-6, Logic::Unknown),
        (10e-6, Logic::One),
        (20e-6, Logic::Zero),
        (40e-6, Logic::One),
        (50e-6, Logic::Zero),
        (70e-6, Logic::One),
    ] {
        trial.schedule(clk, time, Drive::strong(logic))?;
    }
    trial.advance_to(80e-6)?;
    let actual: Vec<_> = trial
        .changes()
        .iter()
        .filter(|change| change.net == q)
        .map(|change| (change.time, change.drive.logic))
        .collect();
    let expected = [
        (0., Logic::Zero),
        (13e-6, Logic::One),
        (43e-6, Logic::Unknown),
        (73e-6, Logic::Zero),
    ];
    assert_eq!(actual.len(), expected.len());
    for ((t, state), (et, es)) in actual.into_iter().zip(expected) {
        assert!((t - et).abs() < 1e-15);
        assert_eq!(state, es);
    }
    assert_eq!(trial.value(nq), Some(Drive::strong(Logic::One)));
    trial.commit()
}

const DFF_INPUTS: &str = "vdata data 0 pwl(0 1 29.99u 1 30u 0.5 59.99u 0.5 60u 0 80u 0)\n\
    vclk clk 0 pwl(0 0 4.99u 0 5u 0.5 9.99u 0.5 10u 1 19.99u 1 20u 0 39.99u 0 40u 1 49.99u 1 50u 0 69.99u 0 70u 1 80u 1)\n\
    r1 q 0 1k\nr2 nq 0 1k\n";

fn run_mixed_dff() -> SpiceResult<ngspice_rs::analysis::Plot> {
    use ngspice_rs::analysis::digital::{AdcBinding, MixedSignal};
    use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, mixed_signal_transient};
    use ngspice_rs::devices::{Circuit, code_model::DacConfig};
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use ngspice_rs::primitives::{AnalysisKind, NodeId};
    use std::path::Path;

    let deck = format!("mixed sequential logic\n{DFF_INPUTS}.end\n");
    let netlist = Parser::new().parse_deck(&parse_deck_text(Path::new("dff.cir"), &deck))?;
    let mut circuit = Circuit::from_netlist(&netlist)?;
    let mut network = Network::default();
    let [data, clock, q, nq] = std::array::from_fn(|_| network.add_net());
    let d = network.add_source(data)?;
    let c = network.add_source(clock)?;
    network.add_model(
        DFlipFlop {
            initial: Logic::Zero,
            delays: Delays::new(3e-6, 3e-6)?,
            mode: DelayMode::Transport,
        },
        &[data, clock],
        &[q, nq],
    )?;
    let mut mixed = MixedSignal::new(Engine::new(network, Limits::default())?);
    for (node, source) in [("data", d), ("clk", c)] {
        mixed.add_adc(AdcBinding {
            terminals: [circuit.nodes().get(node).unwrap(), NodeId::GROUND],
            source,
            thresholds: Adc::new(0.25, 0.75)?,
            delays: Delays::new(1e-9, 1e-9)?,
            mode: DelayMode::Transport,
        })?;
    }
    let config = DacConfig::new(
        Dac {
            low: 0.,
            high: 1.,
            unknown: 0.5,
        },
        Delays::new(10e-9, 10e-9)?,
    )?;
    for (name, net) in [("q", q), ("nq", nq)] {
        let terminal = circuit.nodes().get(name).unwrap();
        mixed.add_dac(
            &mut circuit,
            format!("a{name}"),
            [terminal, NodeId::GROUND],
            net,
            config,
        )?;
    }
    let plan = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "80u"]).resolve()?;
    let (plot, _) =
        mixed_signal_transient(&mut circuit, &plan, &AnalysisContext::default(), &mut mixed)?;
    let events: Vec<_> = mixed
        .changes()
        .iter()
        .filter(|change| change.net == q)
        .map(|change| (change.time, change.drive.logic))
        .collect();
    assert_eq!(events.len(), 4, "{events:?}");
    // Linear ramp crossing times + 1 ns ADC delay + 3 us clock-to-Q delay.
    let expected = [
        (0., Logic::Zero),
        (12.996e-6, Logic::One),
        (42.9985e-6, Logic::Unknown),
        (72.9985e-6, Logic::Zero),
    ];
    for ((t, state), (expected_t, expected_state)) in events.into_iter().zip(expected) {
        assert_eq!(state, expected_state);
        assert!((t - expected_t).abs() < 1e-10, "{t:e} vs {expected_t:e}");
    }
    Ok(plot)
}

fn output_at(plot: &ngspice_rs::analysis::Plot, node: &str, t: f64) -> f64 {
    let time = plot.column("time").unwrap();
    let output = plot.column(node).unwrap();
    let upper = time.partition_point(|v| v.re < t).min(time.len() - 1);
    let lower = upper.saturating_sub(1);
    if upper == lower {
        return output[upper].re;
    }
    let fraction = (t - time[lower].re) / (time[upper].re - time[lower].re);
    fraction.mul_add(output[upper].re - output[lower].re, output[lower].re)
}

fn check_dff_levels(plot: &ngspice_rs::analysis::Plot) {
    for (time, q) in [
        (0., 0.),
        (7e-6, 0.),
        (15e-6, 1.),
        (35e-6, 1.),
        (45e-6, 0.5),
        (65e-6, 0.5),
        (75e-6, 0.),
    ] {
        assert!((output_at(plot, "v(q)", time) - q).abs() < 1e-8);
        assert!((output_at(plot, "v(nq)", time) - (1. - q)).abs() < 1e-8);
    }
}

#[test]
fn ordinary_adc_unknown_intervals_and_sequential_dac_outputs() -> SpiceResult<()> {
    check_dff_levels(&run_mixed_dff()?);
    Ok(())
}

#[test]
#[ignore = "requires NGSPICE_BIN pointing to XSPICE-enabled ngspice"]
fn sequential_logic_matches_xspice() -> SpiceResult<()> {
    let ours = run_mixed_dff()?;
    let reference = run_xspice(&format!(
        "XSPICE D flip flop\n{DFF_INPUTS}\
        aadc [data clk] [dd cc] adc\n\
        aff dd cc null null dq dnq ff\n\
        adac [dq dnq] [q nq] dac\n\
        .model adc adc_bridge(in_low=0.25 in_high=0.75 rise_delay=1n fall_delay=1n)\n\
        .model ff d_dff(ic=0 clk_delay=1u rise_delay=2u fall_delay=2u)\n\
        .model dac dac_bridge(out_low=0 out_high=1 out_undef=0.5 t_rise=10n t_fall=10n)\n\
        .control\nset filetype=ascii\ntran 1u 80u 0 5n\nwrite result.raw\nquit\n.endc\n.end\n"
    ));
    check_dff_levels(&reference);
    for (midpoint, lo, hi) in [
        (0.5, 10e-6, 20e-6),
        (0.75, 40e-6, 50e-6),
        (0.25, 70e-6, 80e-6),
    ] {
        let crossing = |plot: &ngspice_rs::analysis::Plot| -> f64 {
            let time = plot.column("time").unwrap();
            let output = plot.column("v(q)").unwrap();
            for i in 1..time.len() {
                if time[i].re < lo || time[i].re > hi {
                    continue;
                }
                let (a, b) = (output[i - 1].re, output[i].re);
                if (a < midpoint && b >= midpoint) || (a > midpoint && b <= midpoint) {
                    let f = (midpoint - a) / (b - a);
                    return f.mul_add(time[i].re - time[i - 1].re, time[i - 1].re);
                }
            }
            panic!("no Q transition in {lo:e}..{hi:e}");
        };
        let (a, b) = (crossing(&ours), crossing(&reference));
        assert!((a - b).abs() < 15e-9, "{a:e} vs {b:e}");
    }
    Ok(())
}

#[derive(Debug)]
struct DrivePathEcho(Delays);

impl ngspice_rs::devices::code_model::CodeModel for DrivePathEcho {
    type State = ();

    fn schema(&self) -> ngspice_rs::devices::code_model::ModelSchema {
        ngspice_rs::devices::code_model::ModelSchema::digital(1, 1)
    }

    fn initial_state(&self) -> Self::State {}

    fn evaluate(
        &self,
        _context: ngspice_rs::devices::code_model::EventContext,
        inputs: &[Drive],
        _state: &mut (),
    ) -> SpiceResult<Vec<ngspice_rs::devices::code_model::Output>> {
        Ok(vec![ngspice_rs::devices::code_model::Output {
            drive: inputs[0],
            delays: self.0,
            mode: DelayMode::Transport,
        }])
    }
}

#[test]
fn six_physical_paths_include_strength_and_disconnection() -> SpiceResult<()> {
    let delays = Delays::paths([2., 3., 4., 5., 6., 7.])?;
    assert!(Delays::paths([2., 3., -1., 5., 6., 7.]).is_err());
    assert!(Delays::paths([2., 3., 4., 5., f64::NAN, 7.]).is_err());
    assert!(
        ngspice_rs::devices::code_model::DacConfig::new(
            Dac {
                low: 0.,
                high: 1.,
                unknown: 0.5
            },
            delays
        )
        .is_err()
    );
    let mut network = Network::default();
    let input = network.add_net();
    let output = network.add_net();
    let source = network.add_source(input)?;
    network.add_model(DrivePathEcho(delays), &[input], &[output])?;
    let mut engine = Engine::new(network, Limits::default())?;
    let zero = Drive::strong(Logic::Zero);
    let one = Drive::strong(Logic::One);
    let z = Drive::default();
    let mut trial = engine.trial();
    for (at, drive) in [
        (0., zero),
        (10., one),
        (20., zero),
        (30., z),
        (40., zero),
        (50., one),
        (60., z),
        (70., one),
    ] {
        trial.schedule(source, at, drive)?;
    }
    trial.advance_to(80.)?;
    let actual: Vec<_> = trial
        .changes()
        .iter()
        .filter(|change| change.net == output)
        .map(|change| (change.time, change.drive))
        .collect();
    assert_eq!(
        actual,
        [
            (0., zero),
            (12., one),
            (23., zero),
            (34., z),
            (46., zero),
            (52., one),
            (65., z),
            (77., one)
        ]
    );
    trial.commit()
}

#[test]
fn ordinal_drive_resolution_is_order_independent_and_margin_exact() -> SpiceResult<()> {
    use std::num::NonZeroU8;
    let rank = |logic, level| -> SpiceResult<Drive> {
        Ok(Drive {
            logic,
            strength: Strength::from_rank(level)?,
        })
    };
    let strongest = rank(Logic::One, 35)?;
    let competing = rank(Logic::Zero, 32)?;
    let weaker = rank(Logic::One, 30)?;
    let margin = NonZeroU8::new(4).unwrap();
    let expected = rank(Logic::Unknown, 35)?;
    for drives in [
        [strongest, competing, weaker],
        [strongest, weaker, competing],
        [competing, strongest, weaker],
        [competing, weaker, strongest],
        [weaker, strongest, competing],
        [weaker, competing, strongest],
    ] {
        assert_eq!(Drive::resolve_with_margin(drives, margin), expected);
    }
    assert_eq!(
        Drive::resolve_with_margin([strongest, rank(Logic::Zero, 31)?], margin),
        strongest,
    );
    assert_eq!(
        Drive::resolve_with_margin([strongest, rank(Logic::Unknown, 32)?], margin),
        expected,
    );
    assert_eq!(
        Drive::resolve_with_margin([rank(Logic::Zero, 0)?, rank(Logic::One, 0)?], margin),
        Drive::default(),
    );
    Ok(())
}

#[test]
fn causal_arrival_boundaries_fanout_and_rollback_preserve_correlation() -> SpiceResult<()> {
    use ngspice_rs::analysis::digital::timing::{
        Arrival, Bounds, DelayWindow, Origin, TimingWindow,
    };
    use ngspice_rs::devices::code_model::{WindowDelays, WindowGate};
    let mut network = Network::default();
    let input = network.add_net();
    let data = network.add_net();
    let clock = network.add_net();
    let source = network.add_source(input)?;
    for (net, low, high) in [(data, 10., 30.), (clock, 40., 60.)] {
        let delay = DelayWindow::new(low, high)?;
        network.add_model(
            WindowGate::new(GateKind::Buffer, 1, WindowDelays::binary(delay, delay))?,
            &[input],
            &[net],
        )?;
    }
    let mut engine = Engine::new(network, Limits::default())?;
    let zero = Drive::strong(Logic::Zero);
    let one = Drive::strong(Logic::One);
    let mut init = engine.trial();
    init.schedule(source, 0., zero)?;
    let shared = TimingWindow::exact(0.)?.after(DelayWindow::new(15., 30.)?, Origin::shared(77))?;
    let edge = Arrival::new(zero, one, shared.clone())?;
    let cancelled = init.schedule_arrival(source, edge.clone())?;
    assert!(init.cancel(cancelled));
    let live = init.schedule_arrival(source, edge)?;
    init.advance_to(10.)?;
    init.commit()?;
    {
        let mut rejected = engine.trial();
        rejected.advance_to(26.)?;
        assert_eq!(rejected.value(data), Some(Drive::strong(Logic::Rising)));
        assert_eq!(
            rejected.arrival(data).unwrap().window().bounds(),
            Bounds::new(25., 60.)?
        );
        assert!(!rejected.cancel(live));
    }
    assert_eq!(engine.value(data), Some(zero));
    assert_eq!(engine.next_event_time(), Some(15.));
    let mut trial = engine.trial();
    trial.advance_to(55.)?;
    assert_eq!(trial.value(clock), Some(Drive::strong(Logic::Rising)));
    assert_eq!(
        trial
            .arrival(clock)
            .unwrap()
            .window()
            .relative_to(trial.arrival(data).unwrap().window(), 100)?,
        Bounds::new(10., 50.)?
    );
    trial.advance_to(100.)?;
    assert_eq!(trial.value(data), Some(one));
    assert_eq!(trial.value(clock), Some(one));
    assert_eq!(
        trial
            .arrival(clock)
            .unwrap()
            .window()
            .relative_to(trial.arrival(data).unwrap().window(), 100)?,
        Bounds::new(10., 50.)?
    );
    trial.commit()?;
    assert!(
        engine
            .arrival(input)
            .unwrap()
            .window()
            .same_expression(&shared)
    );
    Ok(())
}

#[test]
fn monotone_causal_gate_joins_previously_masked_arrivals() -> SpiceResult<()> {
    use ngspice_rs::analysis::digital::timing::{
        Arrival, Bounds, DelayWindow, Origin, TimingWindow,
    };
    use ngspice_rs::devices::code_model::{WindowDelays, WindowGate};
    let mut network = Network::default();
    let a = network.add_net();
    let b = network.add_net();
    let output = network.add_net();
    let sa = network.add_source(a)?;
    let sb = network.add_source(b)?;
    let delay = DelayWindow::new(2., 3.)?;
    network.add_model(
        WindowGate::new(GateKind::And, 2, WindowDelays::binary(delay, delay))?,
        &[a, b],
        &[output],
    )?;
    let mut engine = Engine::new(network, Limits::default())?;
    let mut trial = engine.trial();
    let zero = Drive::strong(Logic::Zero);
    let one = Drive::strong(Logic::One);
    trial.schedule(sa, 0., zero)?;
    trial.schedule(sb, 0., zero)?;
    for (source, min, max, id) in [(sa, 5., 8., 1), (sb, 6., 10., 2)] {
        let window =
            TimingWindow::exact(0.)?.after(DelayWindow::new(min, max)?, Origin::shared(id))?;
        trial.schedule_arrival(source, Arrival::new(zero, one, window)?)?;
    }
    trial.advance_to(7.)?;
    assert_eq!(trial.value(output), Some(zero));
    trial.advance_to(8.)?;
    assert_eq!(trial.value(output), Some(Drive::strong(Logic::Rising)));
    assert_eq!(
        trial.arrival(output).unwrap().window().bounds(),
        Bounds::new(8., 13.)?
    );
    trial.advance_to(13.)?;
    assert_eq!(trial.value(output), Some(one));
    trial.commit()
}

#[test]
fn same_causal_expression_cancels_xor_reconvergence() -> SpiceResult<()> {
    use ngspice_rs::analysis::digital::timing::{Arrival, DelayWindow, Origin, TimingWindow};
    use ngspice_rs::devices::code_model::{WindowDelays, WindowGate};
    let mut network = Network::default();
    let input = network.add_net();
    let output = network.add_net();
    let source = network.add_source(input)?;
    let delay = DelayWindow::new(1., 2.)?;
    network.add_model(
        WindowGate::new(GateKind::Xor, 2, WindowDelays::binary(delay, delay))?,
        &[input, input],
        &[output],
    )?;
    let mut engine = Engine::new(network, Limits::default())?;
    let mut trial = engine.trial();
    let zero = Drive::strong(Logic::Zero);
    let one = Drive::strong(Logic::One);
    trial.schedule(source, 0., zero)?;
    trial.schedule_arrival(
        source,
        Arrival::new(
            zero,
            one,
            TimingWindow::exact(0.)?.after(DelayWindow::new(5., 10.)?, Origin::shared(1))?,
        )?,
    )?;
    trial.advance_to(20.)?;
    assert_eq!(trial.value(output), Some(zero));
    assert_eq!(
        trial.changes().iter().filter(|c| c.net == output).count(),
        1
    );
    trial.commit()
}

#[test]
fn cumulative_arrival_overlap_preserves_later_completion_and_rollback() -> SpiceResult<()> {
    use Logic::{Falling, One, Rising, Unknown, Zero};
    use ngspice_rs::analysis::digital::{
        DiagnosticDetail,
        timing::{Bounds, DelayWindow},
    };
    use ngspice_rs::devices::code_model::{WindowDelays, WindowGate};
    // The first case reproduces the overlapping edge windows in the
    // cumulative-ambiguity illustration in the PSpice user guide.
    // The second covers a later edge that completes first.
    for (rise_max, fall_max, trace) in [
        (
            10.,
            10.,
            vec![
                (0., Zero),
                (3., Rising),
                (10., Unknown),
                (12., Falling),
                (19., Zero),
            ],
        ),
        (
            18.,
            3.,
            vec![(0., Zero), (3., Rising), (10., Unknown), (12., Zero)],
        ),
    ] {
        let mut network = Network::default();
        let input = network.add_net();
        let output = network.add_net();
        let inverse = network.add_net();
        let source = network.add_source(input)?;
        network.add_model(
            WindowGate::new(
                GateKind::Buffer,
                1,
                WindowDelays::binary(
                    DelayWindow::new(1., rise_max)?,
                    DelayWindow::new(1., fall_max)?,
                ),
            )?,
            &[input],
            &[output],
        )?;
        let zero = DelayWindow::new(0., 0.)?;
        network.add_model(
            WindowGate::new(GateKind::Inverter, 1, WindowDelays::binary(zero, zero))?,
            &[output],
            &[inverse],
        )?;
        let mut engine = Engine::new(network, Limits::default())?;
        let mut init = engine.trial();
        for (t, logic) in [(0., Zero), (2., One), (9., Zero)] {
            init.schedule(source, t, Drive::strong(logic))?;
        }
        init.advance_to(2.)?;
        init.commit()?;
        let mut rejected = engine.trial();
        rejected.advance_to(11.)?;
        assert_eq!(rejected.value(output), Some(Drive::strong(Unknown)));
        assert_eq!(rejected.diagnostics().len(), 2);
        drop(rejected);
        assert!(engine.diagnostics().is_empty());
        assert_eq!(engine.value(output), Some(Drive::strong(Zero)));
        let mut trial = engine.trial();
        trial.advance_to(30.)?;
        let actual: Vec<_> = trial
            .changes()
            .iter()
            .filter(|c| c.net == output)
            .map(|c| (c.time, c.drive.logic))
            .collect();
        assert_eq!(actual, trace[1..]);
        let actual: Vec<_> = trial
            .changes()
            .iter()
            .filter(|c| c.net == inverse)
            .map(|c| (c.time, c.drive.logic))
            .collect();
        let expected: Vec<_> = trace[1..]
            .iter()
            .map(|&(time, logic)| (time, logic.invert()))
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(trial.value(output), Some(Drive::strong(Zero)));
        assert_eq!(trial.value(inverse), Some(Drive::strong(One)));
        let diagnostics = trial.diagnostics();
        assert_eq!(diagnostics.len(), 2);
        for diagnostic in diagnostics {
            let DiagnosticDetail::CumulativeAmbiguity {
                earlier,
                later,
                separation,
            } = diagnostic.detail
            else {
                panic!("unexpected diagnostic {diagnostic:?}");
            };
            assert_eq!(earlier, Bounds::new(3., 2. + rise_max)?);
            assert_eq!(later, Bounds::new(10., 9. + fall_max)?);
            assert_eq!(separation, Bounds::new(8. - rise_max, 6. + fall_max)?);
        }
        trial.commit()?;
    }
    Ok(())
}

#[test]
fn absolute_overlap_diagnostic_retains_correlated_relative_pulse_width() -> SpiceResult<()> {
    use ngspice_rs::analysis::digital::{
        DiagnosticDetail,
        timing::{Arrival, Bounds, DelayWindow, Origin, TimingWindow},
    };
    let mut network = Network::default();
    let output = network.add_net();
    let source = network.add_source(output)?;
    let mut engine = Engine::new(network, Limits::default())?;
    let low = Drive::strong(Logic::Zero);
    let high = Drive::strong(Logic::One);
    let root = TimingWindow::exact(0.)?.after(DelayWindow::new(5., 15.)?, Origin::shared(77))?;
    let rise = root.after(DelayWindow::new(3., 3.)?, Origin::shared(78))?;
    let fall = root.after(DelayWindow::new(5., 5.)?, Origin::shared(79))?;
    let mut trial = engine.trial();
    trial.schedule(source, 0., low)?;
    trial.schedule_arrival(source, Arrival::new(low, high, rise)?)?;
    trial.schedule_arrival(source, Arrival::new(high, low, fall)?)?;
    trial.advance_to(11.)?;
    assert_eq!(trial.value(output), Some(Drive::strong(Logic::Unknown)));
    let diagnostics = trial.diagnostics();
    assert_eq!(diagnostics.len(), 1);
    let DiagnosticDetail::CumulativeAmbiguity { separation, .. } = diagnostics[0].detail else {
        panic!("missing cumulative ambiguity diagnostic");
    };
    assert_eq!(separation, Bounds::new(2., 2.)?);
    trial.advance_to(30.)?;
    assert_eq!(trial.value(output), Some(low));
    trial.commit()
}

#[test]
fn bounded_transport_cancels_both_boundaries_when_new_arrival_definitely_precedes()
-> SpiceResult<()> {
    use ngspice_rs::analysis::digital::timing::DelayWindow;
    use ngspice_rs::devices::code_model::{WindowDelays, WindowGate};
    let mut network = Network::default();
    let input = network.add_net();
    let output = network.add_net();
    let source = network.add_source(input)?;
    network.add_model(
        WindowGate::new(
            GateKind::Buffer,
            1,
            WindowDelays::binary(DelayWindow::new(10., 20.)?, DelayWindow::new(1., 2.)?),
        )?,
        &[input],
        &[output],
    )?;
    let mut engine = Engine::new(network, Limits::default())?;
    let mut trial = engine.trial();
    for (time, logic) in [(0., Logic::Zero), (5., Logic::One), (6., Logic::Zero)] {
        trial.schedule(source, time, Drive::strong(logic))?;
    }
    trial.advance_to(50.)?;
    assert_eq!(trial.value(output), Some(Drive::strong(Logic::Zero)));
    assert_eq!(
        trial.changes().iter().filter(|c| c.net == output).count(),
        1
    );
    assert!(trial.diagnostics().is_empty());
    assert!(trial.next_event_time().is_none());
    trial.commit()
}
