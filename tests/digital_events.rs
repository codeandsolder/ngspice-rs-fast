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
            strength: Strength::Resistive,
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
            strength: Strength::Resistive,
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
    let schmitt = Adc::new(2., 0.8)?;
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
    let plan = AnalysisInput::with_arguments(AnalysisKind::Transient, ["40u", "2m"].into_iter())
        .resolve()?;
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
