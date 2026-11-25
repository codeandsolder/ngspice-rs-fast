//! Native scheduler and compiled-HDL process integration.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ngspice_rs::analysis::cosim::{ProcessSimulator, ReplayMode, ReplaySimulator};
use ngspice_rs::analysis::digital::{Engine, Limits, Network};
use ngspice_rs::devices::code_model::{Logic, Strength};
use ngspice_rs::devices::digital_bridge::{
    Bit, Bridge, Reply, Request, Schema, Simulator, TimeScale,
};
use ngspice_rs::primitives::{SpiceError, SpiceResult};

const SCHEMA: Schema = Schema {
    inputs: 1,
    outputs: 1,
    snapshot_bytes: 1 << 20,
};

#[derive(Clone, Debug)]
struct CounterState {
    count: u64,
    next: Option<u64>,
}

#[derive(Debug)]
struct Counter;

impl Simulator for Counter {
    type State = CounterState;

    fn evaluate(
        &self,
        _schema: Schema,
        request: Request<'_, CounterState>,
    ) -> SpiceResult<Reply<CounterState>> {
        let mut state = request.snapshot.cloned().unwrap_or(CounterState {
            count: 0,
            next: None,
        });
        if state.next == Some(request.tick) {
            state.count += 1;
        }
        let output = match request.inputs {
            [Bit::One] => {
                state.next = Some(request.tick + 10);
                if state.count.is_multiple_of(2) {
                    Bit::Zero
                } else {
                    Bit::One
                }
            }
            [Bit::Zero] => {
                state.next = None;
                Bit::Zero
            }
            [Bit::X] => {
                state.next = None;
                Bit::X
            }
            [Bit::Z] => {
                state.next = None;
                Bit::Z
            }
            _ => return Err(SpiceError::circuit("counter input width")),
        };
        Ok(Reply {
            reached_tick: request.tick,
            outputs: vec![output],
            next_tick: state.next,
            snapshot: state,
        })
    }
}

#[test]
fn four_states_round_trip_and_time_scale_preserves_causality() -> SpiceResult<()> {
    for bit in [Bit::Zero, Bit::One, Bit::X, Bit::Z] {
        assert_eq!(Bit::from_drive(bit.drive()), bit);
        assert_eq!(Bit::decode(bit as u8)?, bit);
    }
    assert_eq!(Bit::Z.drive().strength, Strength::HighImpedance);
    assert!(Bit::decode(4).is_err());
    for precision in -15..=0 {
        let scale = TimeScale::new(precision)?;
        for tick in [
            0,
            1,
            3,
            10,
            999,
            1_000_000,
            (1_u64 << 51) - 1,
            TimeScale::MAX_TICK,
        ] {
            let physical = scale.physical(tick)?;
            assert_eq!(
                scale.tick(physical)?,
                tick,
                "precision {precision}, tick {tick}"
            );
            if tick > 0 {
                assert!(scale.tick(physical.next_down())? < tick);
            }
        }
    }
    assert!(TimeScale::new(-16).is_err());
    assert!(TimeScale::new(-9)?.tick(f64::NAN).is_err());
    Ok(())
}

#[test]
fn autonomous_wakeup_cancellation_and_model_checkpoint_rollback() -> SpiceResult<()> {
    let scale = TimeScale::new(-9)?;
    let mut network = Network::default();
    let control = network.add_net();
    let output = network.add_net();
    let source = network.add_source(control)?;
    network.add_model(Bridge::new(Counter, SCHEMA, scale)?, &[control], &[output])?;
    let mut engine = Engine::new(network, Limits::default())?;
    let mut initial = engine.trial();
    initial.schedule(source, 0., Bit::One.drive())?;
    initial.advance_to(0.)?;
    initial.commit()?;
    assert_eq!(engine.next_event_time(), Some(scale.physical(10)?));
    {
        let mut discarded = engine.trial();
        discarded.advance_to(scale.physical(31)?)?;
        assert_eq!(discarded.value(output), Some(Bit::One.drive()));
    }
    assert_eq!(engine.next_event_time(), Some(scale.physical(10)?));
    assert_eq!(engine.value(output), Some(Bit::Zero.drive()));
    let mut retry = engine.trial();
    retry.schedule(source, scale.physical(15)?, Bit::Z.drive())?;
    retry.advance_to(scale.physical(100)?)?;
    assert_eq!(retry.value(output), Some(Bit::Z.drive()));
    assert_eq!(retry.next_event_time(), None);
    let times: Vec<_> = retry
        .changes()
        .iter()
        .filter(|c| c.net == output)
        .map(|c| scale.tick(c.time).unwrap())
        .collect();
    assert_eq!(times, [10, 15]);
    retry.commit()
}

#[derive(Debug)]
struct SameDelta;

impl Simulator for SameDelta {
    type State = u32;

    fn evaluate(&self, _schema: Schema, request: Request<'_, u32>) -> SpiceResult<Reply<u32>> {
        let count = request.snapshot.copied().unwrap_or(0) + 1;
        Ok(Reply {
            reached_tick: request.tick,
            outputs: vec![Bit::X],
            next_tick: Some(request.tick),
            snapshot: count,
        })
    }
}

#[test]
fn autonomous_same_tick_loop_is_bounded() -> SpiceResult<()> {
    let mut network = Network::default();
    let input = network.add_net();
    let output = network.add_net();
    network.add_model(
        Bridge::new(SameDelta, SCHEMA, TimeScale::new(-9)?)?,
        &[input],
        &[output],
    )?;
    let mut engine = Engine::new(
        network,
        Limits {
            delta_cycles: 8,
            work: 100,
        },
    )?;
    let mut trial = engine.trial();
    assert!(trial.advance_to(0.).is_err());
    assert!(trial.commit().is_err());
    assert_eq!(engine.stats().events, 0);
    Ok(())
}

fn find_program(name: &str) -> PathBuf {
    let path = std::env::var_os("PATH").expect("PATH");
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|file| file.is_file())
        .unwrap_or_else(|| panic!("required simulator executable {name} is not installed"))
        .canonicalize()
        .unwrap()
}

fn compile_filter(directory: &Path) -> PathBuf {
    use ngspice_conformance::process::{ProcessLimits, run};
    let sources = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cosim");
    let args = [
        "--cc".into(),
        "--exe".into(),
        "--build".into(),
        "--timing".into(),
        "--top-module".into(),
        "clocked_filter".into(),
        "--Mdir".into(),
        directory.as_os_str().to_owned(),
        "-j".into(),
        "2".into(),
        "-CFLAGS".into(),
        "-std=c++20".into(),
        sources.join("clocked_filter.sv").into_os_string(),
        sources.join("clocked_filter.cpp").into_os_string(),
    ];
    let result = run(
        find_program("verilator"),
        directory,
        &args,
        ProcessLimits::ORACLE,
    )
    .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    directory.join("Vclocked_filter")
}

fn replay(program: &Path, directory: &Path) -> SpiceResult<ReplaySimulator> {
    ReplaySimulator::new(
        program.into(),
        vec![],
        directory.into(),
        Duration::from_secs(5),
        100_000,
        ReplayMode::Scheduled,
    )
}

const FILTER: Schema = Schema {
    inputs: 1,
    outputs: 2,
    snapshot_bytes: Schema::MAX_SNAPSHOT,
};

fn compile_ghdl(directory: &Path, entity: &str, source: &Path) -> (PathBuf, Vec<OsString>) {
    use ngspice_conformance::process::{ProcessLimits, run};
    let ghdl = find_program("ghdl");
    let adapter = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cosim/ghdl_bridge.cpp");
    let commands: [Vec<OsString>; 3] = [
        vec!["-a".into(), "--std=08".into(), source.into()],
        vec![
            "--vpi-compile".into(),
            "g++".into(),
            "-std=c++20".into(),
            "-fPIC".into(),
            "-c".into(),
            adapter.into(),
            "-o".into(),
            "bridge.o".into(),
        ],
        vec![
            "--vpi-link".into(),
            "g++".into(),
            "-shared".into(),
            "-o".into(),
            "bridge.so".into(),
            "bridge.o".into(),
        ],
    ];
    for args in commands {
        let result = run(&ghdl, directory, &args, ProcessLimits::ORACLE).unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let args = vec![
        "-r".into(),
        "--std=08".into(),
        entity.into(),
        "--vpi=./bridge.so".into(),
        "--stop-delta=1024".into(),
    ];
    (ghdl, args)
}

fn run_clocked_filter(simulator: ReplaySimulator, precision: i32, per_ns: u64) -> SpiceResult<()> {
    let scale = TimeScale::new(precision)?;
    let mut network = Network::default();
    let input = network.add_net();
    let filtered = network.add_net();
    let clock = network.add_net();
    let source = network.add_source(input)?;
    network.add_model(
        Bridge::new(simulator, FILTER, scale)?,
        &[input],
        &[filtered, clock],
    )?;
    let mut engine = Engine::new(network, Limits::default())?;
    let mut trial = engine.trial();
    trial.schedule(source, 0., Bit::Zero.drive())?;
    trial.schedule(source, scale.physical(12 * per_ns)?, Bit::One.drive())?;
    trial.schedule(source, scale.physical(42 * per_ns)?, Bit::Zero.drive())?;
    trial.advance_to(scale.physical(20 * per_ns)?)?;
    trial.commit()?;
    let mut discarded = engine.trial();
    discarded.advance_to(scale.physical(100 * per_ns)?)?;
    let speculative: Vec<_> = discarded
        .changes()
        .iter()
        .map(|c| (scale.tick(c.time).unwrap() / per_ns, c.net, c.drive))
        .collect();
    drop(discarded);
    assert_eq!(engine.time(), scale.physical(20 * per_ns)?);
    let mut retry = engine.trial();
    retry.advance_to(scale.physical(100 * per_ns)?)?;
    let accepted: Vec<_> = retry
        .changes()
        .iter()
        .map(|c| (scale.tick(c.time).unwrap() / per_ns, c.net, c.drive))
        .collect();
    assert_eq!(accepted, speculative);
    assert_eq!(
        accepted
            .iter()
            .filter(|(_, net, _)| *net == filtered)
            .map(|(t, _, d)| (*t, d.logic))
            .collect::<Vec<_>>(),
        [(35, Logic::One), (45, Logic::Zero)]
    );
    retry.commit()?;
    Ok(())
}

#[test]
#[ignore = "requires Verilator 5.x with timing support and a C++20 compiler"]
fn compiled_verilog_runs_internal_clock_and_recovers_rejected_trials() -> SpiceResult<()> {
    let scratch = ngspice_conformance::ngspice::Scratch::new("verilator-cosim").unwrap();
    let program = compile_filter(scratch.path());
    run_clocked_filter(replay(&program, scratch.path())?, -9, 1)
}

#[test]
#[ignore = "requires GHDL 6.x and a C++20 compiler"]
fn compiled_vhdl_runs_internal_clock_and_recovers_rejected_trials() -> SpiceResult<()> {
    let scratch = ngspice_conformance::ngspice::Scratch::new("ghdl-cosim").unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/cosim/clocked_filter.vhd");
    let (program, args) = compile_ghdl(scratch.path(), "clocked_filter", &source);
    run_clocked_filter(
        ReplaySimulator::new(
            program,
            args,
            scratch.path().into(),
            Duration::from_secs(5),
            100_000,
            ReplayMode::Lookahead,
        )?,
        -15,
        1_000_000,
    )
}

#[test]
#[ignore = "requires GHDL 6.x and a C++20 compiler"]
fn compiled_vhdl_preserves_wide_four_state_vectors() -> SpiceResult<()> {
    let scratch = ngspice_conformance::ngspice::Scratch::new("ghdl-bus").unwrap();
    let source = scratch.path().join("logic_bus.vhd");
    std::fs::write(&source, "library ieee; use ieee.std_logic_1164.all;
entity logic_bus is port (a : in std_logic_vector(64 downto 0); b : out std_logic_vector(64 downto 0)); end;
architecture rtl of logic_bus is begin b <= a; end;").unwrap();
    let (program, args) = compile_ghdl(scratch.path(), "logic_bus", &source);
    let simulator = ReplaySimulator::new(
        program,
        args,
        scratch.path().into(),
        Duration::from_secs(5),
        100_000,
        ReplayMode::Lookahead,
    )?;
    let schema = Schema {
        inputs: 65,
        outputs: 65,
        snapshot_bytes: 1 << 20,
    };
    let pattern: Vec<_> = (0..65)
        .map(|i| [Bit::Zero, Bit::One, Bit::X, Bit::Z][i % 4])
        .collect();
    let reply = simulator.evaluate(
        schema,
        Request {
            tick: 0,
            precision: -15,
            operation: ngspice_rs::devices::digital_bridge::Operation::ApplyInputs,
            delta: 0,
            inputs: &pattern,
            snapshot: None,
        },
    )?;
    assert_eq!(reply.outputs, pattern);
    let reverse: Vec<_> = pattern.iter().rev().copied().collect();
    let reply = simulator.evaluate(
        schema,
        Request {
            tick: 100,
            precision: -15,
            operation: ngspice_rs::devices::digital_bridge::Operation::ApplyInputs,
            delta: 0,
            inputs: &reverse,
            snapshot: Some(&reply.snapshot),
        },
    )?;
    assert_eq!(reply.outputs, reverse);
    Ok(())
}

#[test]
#[ignore = "requires Verilator 5.x with timing support and a C++20 compiler"]
fn compiled_verilog_adc_filter_dac_and_event_density() -> SpiceResult<()> {
    use ngspice_rs::analysis::digital::{AdcBinding, MixedSignal};
    use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, mixed_signal_transient};
    use ngspice_rs::devices::Circuit;
    use ngspice_rs::devices::code_model::{Adc, Dac, DacConfig, DelayMode, Delays};
    use ngspice_rs::netlist::{Parser, source::parse_deck_text};
    use ngspice_rs::primitives::{AnalysisKind, NodeId};

    let scratch = ngspice_conformance::ngspice::Scratch::new("verilator-mixed").unwrap();
    let program = compile_filter(scratch.path());
    let scale = TimeScale::new(-9)?;
    let run = |maxstep: &str| -> SpiceResult<_> {
        let deck = parse_deck_text(
            Path::new("filter.cir"),
            "clocked filter\nv1 in 0 pwl(0 0 11n 0 13n 3 41n 3 43n 0 50u 0)\nr1 out 0 1k\n.end\n",
        );
        let netlist = Parser::new().parse_deck(&deck)?;
        let mut circuit = Circuit::from_netlist(&netlist)?;
        let input_node = circuit.nodes().get("in").unwrap();
        let output_node = circuit.nodes().get("out").unwrap();
        let mut network = Network::default();
        let input = network.add_net();
        let filtered = network.add_net();
        let clock = network.add_net();
        let source = network.add_source(input)?;
        network.add_model(
            Bridge::new(replay(&program, scratch.path())?, FILTER, scale)?,
            &[input],
            &[filtered, clock],
        )?;
        let mut mixed = MixedSignal::new(Engine::new(network, Limits::default())?);
        mixed.add_adc(AdcBinding {
            terminals: [input_node, NodeId::GROUND],
            source,
            thresholds: Adc::new(2., 1.)?,
            delays: Delays::new(0., 0.)?,
            mode: DelayMode::Transport,
        })?;
        mixed.add_dac(
            &mut circuit,
            "aout",
            [output_node, NodeId::GROUND],
            filtered,
            DacConfig::new(
                Dac {
                    low: 0.,
                    high: 3.,
                    unknown: 1.5,
                },
                Delays::new(1e-9, 1e-9)?,
            )?,
        )?;
        let plan =
            AnalysisInput::with_arguments(AnalysisKind::Transient, ["100n", "50u", "0", maxstep])
                .resolve()?;
        let start = Instant::now();
        let (plot, stats) =
            mixed_signal_transient(&mut circuit, &plan, &AnalysisContext::default(), &mut mixed)?;
        let duration = start.elapsed();
        let transitions: Vec<_> = mixed
            .changes()
            .iter()
            .filter(|c| c.net == filtered)
            .map(|c| (scale.tick(c.time).unwrap(), c.drive.logic))
            .collect();
        assert_eq!(
            transitions,
            [(0, Logic::Zero), (35, Logic::One), (45, Logic::Zero)]
        );
        assert_eq!(plot.column("v(out)").unwrap().last().unwrap().re, 0.);
        Ok((stats.accepted, mixed.engine().stats(), duration))
    };
    let adaptive = run("1u")?;
    let brute = run("5n")?;
    println!("adaptive: {adaptive:?}; forced 5 ns analog stepping: {brute:?}");
    assert!(adaptive.0 * 20 < brute.0);
    assert_eq!(adaptive.1, brute.1);
    Ok(())
}

#[test]
fn invalid_process_configuration_fails_explicitly() {
    assert!(
        ProcessSimulator::new(
            "relative".into(),
            vec![],
            "/".into(),
            Duration::from_secs(1)
        )
        .is_err()
    );
    assert!(
        ReplaySimulator::new(
            "/sim".into(),
            vec![],
            "/".into(),
            Duration::ZERO,
            1,
            ReplayMode::Scheduled
        )
        .is_err()
    );
    assert!(
        ReplaySimulator::new(
            "/sim".into(),
            vec![],
            "/".into(),
            Duration::from_secs(1),
            0,
            ReplayMode::Scheduled
        )
        .is_err()
    );
}
