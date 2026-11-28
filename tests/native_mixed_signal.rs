//! Deck-bound mixed-signal execution, checked against Boolean truth tables
//! and explicitly specified receiver and propagation timing.
use std::path::Path;

use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};

fn circuit(body: &str) -> SpiceResult<Circuit> {
    let deck = Parser::new().parse_deck(&parse_deck_text(
        Path::new("mixed.cir"),
        &format!("native mixed test\n{body}\n.end\n"),
    ))?;
    Circuit::from_netlist(&deck)
}

fn run(circuit: &mut Circuit, kind: AnalysisKind, args: &[&str]) -> SpiceResult<Plot> {
    AnalysisInput::with_arguments(kind, args.iter().copied())
        .resolve()?
        .run(circuit, &AnalysisContext::default())
}

fn output(plot: &Plot, index: usize) -> f64 {
    plot.value("v(out)", index).unwrap().re
}

fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
}

#[test]
fn native_truth_tables_and_initial_latch_state() -> SpiceResult<()> {
    for op in 0..7 {
        for a in [false, true] {
            for b in [false, true] {
                for ic in [false, true] {
                    let expected = match op {
                        0 => a && b,
                        1 => a || b,
                        2 => !(a && b),
                        3 => !(a || b),
                        4 => {
                            if b {
                                false
                            } else {
                                a || ic
                            }
                        }
                        5 => {
                            if a {
                                true
                            } else {
                                !b && ic
                            }
                        }
                        _ => {
                            if a == b {
                                ic
                            } else {
                                a
                            }
                        }
                    };
                    let mut c = circuit(&format!(
                        "va a ref {}\nvb b ref {}\nvref ref 0 7\n\
                         a1 %vd[a ref b ref] logic gate\n\
                         .model gate preprocess_logic_event(operation={op} delay=1u ic={} low=2 high=3)\n\
                         a2 logic %vd(out ref) bridge\n\
                         .model bridge preprocess_logic_voltage(out_low=-2 out_high=5)\n\
                         r1 out ref 1k",
                        if a { 4 } else { 0 },
                        if b { 4 } else { 0 },
                        u8::from(ic)
                    ))?;
                    for _ in 0..2 {
                        let plot = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
                        close(output(&plot, 0), if expected { 12. } else { 5. });
                        assert!(c.last_digital_stats().unwrap().evaluations > 0);
                        assert!(c.nodes().get("logic").is_none());
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn equality_initial_receivers_and_inverted_rails() -> SpiceResult<()> {
    let mut c = circuit(
        "va a 0 3\nvb b 0 2\n\
         a1 %v[a b] logic gate\n\
         .model gate preprocess_logic_event(operation=0 low=2 high=3 inverted=[true false] rx_ic=[1 1])\n\
         a2 logic out bridge\n\
         .model bridge preprocess_logic_voltage(out_low=4 out_high=-1)\n\
         r1 out 0 1k",
    )?;
    close(
        output(&run(&mut c, AnalysisKind::OperatingPoint, &[])?, 0),
        -1.,
    );
    assert!(run(&mut c, AnalysisKind::Ac, &["lin", "1", "1", "1"]).is_err());
    Ok(())
}

#[test]
fn source_transient_distinguishes_inertial_and_transport() -> SpiceResult<()> {
    for inertial in [false, true] {
        let mut c = circuit(&format!(
            "va a 0 pwl(0 0 1u 0 1.1u 4 1.4u 4 1.5u 0 4u 0)\n\
             a1 %v[a] logic gate\n\
             .model gate preprocess_logic_event(operation=0 delay=1u inertial={inertial} low=2 high=3)\n\
             a2 logic out bridge\n\
             .model bridge preprocess_logic_voltage(out_low=0 out_high=5)\n\
             r1 out 0 1k"
        ))?;
        let plot = run(&mut c, AnalysisKind::Transient, &["10n", "4u", "0", "10n"])?;
        let mut high_samples = 0;
        for i in 0..plot.point_count() {
            let time = plot.value("time", i).unwrap().re;
            if (2.12e-6..2.42e-6).contains(&time) {
                close(output(&plot, i), if inertial { 0. } else { 5. });
                high_samples += 1;
            } else if time < 2.06e-6 || time > 2.48e-6 {
                close(output(&plot, i), 0.);
            }
        }
        assert!(high_samples > 10);
    }
    Ok(())
}

#[test]
fn nested_formals_keep_digital_ground_names_distinct() -> SpiceResult<()> {
    let mut c = circuit(
        ".subckt gate in state\n\
         a1 %v[in] state gate_model\n\
         .model gate_model preprocess_logic_event(operation=0)\n\
         .ends\n\
         .subckt outer in state\n\
         x1 in state gate\n\
         .ends\n\
         va a 0 4\nvb b 0 0\n\
         xhi a gnd outer\nxlo b 0 outer\n\
         a2 gnd out bridge\na3 0 other bridge\n\
         .model bridge preprocess_logic_voltage(out_low=0 out_high=5)\n\
         r1 out 0 1k\nr2 other 0 1k",
    )?;
    let plot = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    close(output(&plot, 0), 5.);
    close(plot.value("v(other)", 0).unwrap().re, 0.);
    Ok(())
}

#[test]
fn native_logic_matches_independent_reference_matrix() -> SpiceResult<()> {
    let body = include_str!("fixtures/native-logic/matrix.cir");
    let deck = Parser::new().parse_deck(&parse_deck_text(Path::new("matrix.cir"), body))?;
    let mut c = Circuit::from_netlist(&deck)?;
    let plot = run(&mut c, AnalysisKind::Transient, &["1n", "9u", "0", "1n"])?;
    let times: Vec<_> = (0..plot.point_count())
        .map(|i| plot.value("time", i).unwrap().re)
        .collect();
    let mut max_error: f64 = 0.;
    for row in include_str!("fixtures/native-logic/matrix-levels.csv")
        .lines()
        .skip(1)
    {
        let fields: Vec<_> = row.split(',').collect();
        let time: f64 = fields[1].parse().unwrap();
        let expected: f64 = fields[2].parse().unwrap();
        let i = times.partition_point(|t| *t < time).min(times.len() - 1);
        let actual = plot.value(&format!("v({})", fields[0]), i).unwrap().re;
        max_error = max_error.max((actual - expected).abs());
        assert!(
            (actual - expected).abs() < 1e-8,
            "{} at {time}: {actual} != {expected}",
            fields[0]
        );
    }
    let mut max_edge_error: f64 = 0.;
    for op in 0..7 {
        for ic in 0..2 {
            for inv in 0..2 {
                let node = format!("g{op}{ic}{inv}");
                let name = format!("v({node})");
                let expected: Vec<_> = include_str!("fixtures/native-logic/matrix-transitions.csv")
                    .lines()
                    .skip(1)
                    .filter_map(|row| {
                        let fields: Vec<_> = row.split(',').collect();
                        (fields[0] == node).then(|| {
                            (
                                fields[1].parse::<f64>().unwrap(),
                                fields[2].parse::<f64>().unwrap(),
                            )
                        })
                    })
                    .collect();
                let midpoint = 2.113_636_363_636_363_8;
                let actual: Vec<_> = (1..times.len())
                    .filter_map(|i| {
                        let prev = plot.value(&name, i - 1).unwrap().re;
                        let value = plot.value(&name, i).unwrap().re;
                        ((prev > midpoint) != (value > midpoint)).then_some((times[i], value))
                    })
                    .collect();
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "{node}: actual {actual:?}, expected {expected:?}"
                );
                for ((t, v), (reference_t, reference_v)) in actual.into_iter().zip(expected) {
                    close(v, reference_v);
                    max_edge_error = max_edge_error.max((t - reference_t).abs());
                    assert!(
                        (t - reference_t).abs() <= 2e-9,
                        "{node}: {t} != {reference_t}"
                    );
                }
            }
        }
    }
    eprintln!(
        "reference matrix: maximum settled error {max_error:e} V, transition error {max_edge_error:e} s"
    );
    Ok(())
}

#[test]
fn scoped_arrays_and_typed_port_validation() -> SpiceResult<()> {
    let mut c = circuit(
        ".subckt gate in out params: inv=0 rail=5\n\
         .model logic preprocess_logic_event(operation={inv*0} inverted=[ {inv} ] rx_ic=[ {inv} ] inertial={1-inv})\n\
         .model bridge preprocess_logic_voltage(out_low=0 out_high={rail})\n\
         a1 %v[in] logic_net logic\na2 logic_net out bridge\n.ends\n\
         vin in 0 4\nx1 in out gate inv=0 rail=7\nx2 in other gate inv=1 rail=9\n\
         r1 out 0 1k\nr2 other 0 1k",
    )?;
    let plot = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    close(output(&plot, 0), 7.);
    close(plot.value("v(other)", 0).unwrap().re, 0.);
    for params in [
        "operation=0.5",
        "operation=17",
        "operation=4",
        "delay=-1n",
        "delay_paths=[1n 2n]",
        "delay_paths=[1n 2n 3n 4n 5n -6n]",
        "delay=1n delay_paths=[1n 2n 3n 4n 5n 6n]",
        "low=4 high=3",
        "ic=3",
        "strength=0",
        "strength=3",
        "async_policy=[1 1 2]",
        "async_policy=[1 1 3]",
        "inverted=[1 0]",
        "inverted=[2]",
        "rx_ic=[2]",
        "inertial=2",
        "unknown=1",
    ] {
        let result = circuit(&format!(
            "v1 in 0 4\na1 %v[in] net gate\n.model gate preprocess_logic_event({params})\n\
             a2 net out bridge\n.model bridge preprocess_logic_voltage\nr1 out 0 1k"
        ));
        assert!(result.is_err(), "accepted invalid {params}");
    }
    for ports in [
        "%v(in) net",
        "%gd[in 0] net",
        "%v[in] %v(net)",
        "%v[] net",
        "%v[in] net extra",
    ] {
        assert!(
            circuit(&format!(
                "v1 in 0 4\na1 {ports} gate\n.model gate preprocess_logic_event"
            ))
            .is_err(),
            "accepted {ports}"
        );
    }
    assert!(circuit("v1 in 0 4\na1 %v[in] net missing").is_err());
    Ok(())
}

#[test]
fn mixed_resource_limits_and_analog_fast_path() -> SpiceResult<()> {
    use ngspice_rs::devices::{ElaborationLimits, ModelContext, MutualLimits};
    let body = "Mixed limits\nv1 in 0 4\na1 %v[in in in in] net gate\n.model gate preprocess_logic_event\n.end\n";
    let deck = Parser::new().parse_deck(&parse_deck_text(Path::new("limits.cir"), body))?;
    let result = Circuit::from_netlist_with_limits(
        &deck,
        &ModelContext::default(),
        MutualLimits::default(),
        ElaborationLimits {
            nodes: 3,
            ..ElaborationLimits::default()
        },
    );
    assert!(result.is_err());
    let mut analog = circuit("v1 in 0 pulse(0 1 1u 10n 10n 1u 4u)\nr1 in out 1k\nc1 out 0 1n")?;
    let a = run(&mut analog, AnalysisKind::Transient, &["10n", "4u"])?;
    assert!(analog.last_digital_stats().is_none());
    let b = run(&mut analog, AnalysisKind::Transient, &["10n", "4u"])?;
    assert!(analog.last_digital_stats().is_none());
    assert_eq!(a.point_count(), b.point_count());
    for i in 0..a.point_count() {
        assert_eq!(a.value("time", i), b.value("time", i));
        assert_eq!(a.value("v(out)", i), b.value("v(out)", i));
    }
    Ok(())
}

#[test]
fn native_delay_and_rc_feedback_match_independent_reference() -> SpiceResult<()> {
    let body = include_str!("fixtures/native-logic/pulse.cir");
    let deck = Parser::new().parse_deck(&parse_deck_text(Path::new("pulse.cir"), body))?;
    let mut c = Circuit::from_netlist(&deck)?;
    let plot = run(&mut c, AnalysisKind::Transient, &["1n", "8u", "0", "1n"])?;
    let times: Vec<_> = (0..plot.point_count())
        .map(|i| plot.value("time", i).unwrap().re)
        .collect();
    let mut max_cap_error: f64 = 0.;
    for row in include_str!("fixtures/native-logic/pulse-levels.csv")
        .lines()
        .skip(1)
    {
        let fields: Vec<_> = row.split(',').collect();
        let time: f64 = fields[1].parse().unwrap();
        let expected: f64 = fields[2].parse().unwrap();
        let i = times.partition_point(|t| *t < time).min(times.len() - 1);
        let actual = plot.value(&format!("v({})", fields[0]), i).unwrap().re;
        if fields[0] == "cap" {
            max_cap_error = max_cap_error.max((actual - expected).abs());
            // 1 ns sampling / 1 us RC, plus reference integration error.
            assert!(
                (actual - expected).abs() <= 0.01,
                "cap at {time}: {actual} != {expected}"
            );
        } else {
            close(actual, expected);
        }
    }
    let mut max_edge_error: f64 = 0.;
    for node in ["o1", "o2", "fb"] {
        let name = format!("v({node})");
        let expected: Vec<_> = include_str!("fixtures/native-logic/pulse-transitions.csv")
            .lines()
            .skip(1)
            .filter_map(|row| {
                let f: Vec<_> = row.split(',').collect();
                (f[0] == node).then(|| (f[1].parse::<f64>().unwrap(), f[2].parse::<f64>().unwrap()))
            })
            .collect();
        let actual: Vec<_> = (1..times.len())
            .filter_map(|i| {
                let prev = plot.value(&name, i - 1).unwrap().re;
                let value = plot.value(&name, i).unwrap().re;
                ((prev > 2.5) != (value > 2.5)).then_some((times[i], value))
            })
            .collect();
        assert_eq!(
            actual.len(),
            expected.len(),
            "{node}: actual {actual:?}, expected {expected:?}"
        );
        for ((t, v), (reference_t, reference_v)) in actual.into_iter().zip(expected) {
            close(v, reference_v);
            max_edge_error = max_edge_error.max((t - reference_t).abs());
            assert!(
                (t - reference_t).abs() <= 2e-9,
                "{node}: {t} != {reference_t}"
            );
        }
    }
    eprintln!(
        "delay/feedback: max capacitor error {max_cap_error:e} V, transition error {max_edge_error:e} s"
    );
    Ok(())
}

#[test]
fn mixed_vector_type_changes_inherit_within_group() -> SpiceResult<()> {
    let mut c = circuit(
        "v1 a 0 4\nv2 b ref 4\nv3 c ref 4\nvref ref 0 7\nv4 d 0 4\n\
         a1 %v[a %vd(b ref) (c ref) %v d] bus gate\n\
         .model gate preprocess_logic_event(operation=0)\n\
         a2 bus out bridge\n.model bridge preprocess_logic_voltage(out_high=5)\nr1 out 0 1k",
    )?;
    close(
        output(&run(&mut c, AnalysisKind::OperatingPoint, &[])?, 0),
        5.,
    );
    Ok(())
}

#[test]
fn clocked_and_enabled_logic_match_independent_reference() -> SpiceResult<()> {
    let deck = Parser::new().parse_deck(&parse_deck_text(
        Path::new("sequential.cir"),
        include_str!("fixtures/native-logic/sequential.cir"),
    ))?;
    let mut c = Circuit::from_netlist(&deck)?;
    let plot = run(&mut c, AnalysisKind::Transient, &["1n", "7u", "0", "1n"])?;
    let times: Vec<_> = (0..plot.point_count())
        .map(|i| plot.value("time", i).unwrap().re)
        .collect();
    let mut nodes = std::collections::BTreeSet::new();
    for row in include_str!("fixtures/native-logic/sequential-levels.csv")
        .lines()
        .skip(1)
    {
        let fields: Vec<_> = row.split(',').collect();
        nodes.insert(fields[0]);
        let time: f64 = fields[1].parse().unwrap();
        let expected: f64 = fields[2].parse().unwrap();
        let index = times.partition_point(|t| *t < time).min(times.len() - 1);
        let actual = plot.value(&format!("v({})", fields[0]), index).unwrap().re;
        assert!(
            (actual - expected).abs() < 1e-8,
            "{} at {time}: {actual} != {expected}",
            fields[0]
        );
    }
    let mut max_edge_error: f64 = 0.;
    for node in nodes {
        let name = format!("v({node})");
        let expected: Vec<_> = include_str!("fixtures/native-logic/sequential-transitions.csv")
            .lines()
            .skip(1)
            .filter_map(|row| {
                let fields: Vec<_> = row.split(',').collect();
                (fields[0] == node).then(|| {
                    (
                        fields[1].parse::<f64>().unwrap(),
                        fields[2].parse::<f64>().unwrap(),
                    )
                })
            })
            .collect();
        let actual: Vec<_> = (1..times.len())
            .filter_map(|i| {
                let prev = plot.value(&name, i - 1).unwrap().re;
                let value = plot.value(&name, i).unwrap().re;
                ((prev > 2.5) != (value > 2.5)).then_some((times[i], value))
            })
            .collect();
        assert_eq!(
            actual.len(),
            expected.len(),
            "{node}: actual {actual:?}, expected {expected:?}"
        );
        for ((time, voltage), (reference_time, reference_voltage)) in
            actual.into_iter().zip(expected)
        {
            close(voltage, reference_voltage);
            max_edge_error = max_edge_error.max((time - reference_time).abs());
            assert!(
                (time - reference_time).abs() <= 2e-9,
                "{node}: {time} != {reference_time}"
            );
        }
    }
    eprintln!("sequential reference: maximum edge error {max_edge_error:e} s");
    Ok(())
}

#[test]
fn clocked_inertial_capture_survives_intervening_receiver_changes() -> SpiceResult<()> {
    // At 1.015 us, D is high. D falls at 1.050 us and the clock falls
    // before the 1.115 us deadline. Neither is a rising sampling edge.
    let mut c = circuit(
        "vd d 0 pwl(0 4 1.04u 4 1.06u 0 4u 0)\n\
         vc clk 0 pulse(0 4 1u 20n 20n 40n 2u)\n\
         a1 %v[d clk] bit ff\n.model ff preprocess_logic_event(operation=11 delay=100n inertial=true)\n\
         a2 bit out bridge\n.model bridge preprocess_logic_voltage(out_high=5)\nr1 out 0 1k",
    )?;
    let plot = run(&mut c, AnalysisKind::Transient, &["1n", "4u", "0", "1n"])?;
    let mut edges = Vec::new();
    for i in 1..plot.point_count() {
        let before = output(&plot, i - 1);
        let after = output(&plot, i);
        if (before > 2.5) != (after > 2.5) {
            edges.push((plot.value("time", i).unwrap().re, after));
        }
    }
    assert_eq!(edges.len(), 2, "{edges:?}");
    for ((time, voltage), (expected_time, expected_voltage)) in
        edges.into_iter().zip([(1.115e-6, 5.), (3.115e-6, 0.)])
    {
        close(voltage, expected_voltage);
        assert!(
            (time - expected_time).abs() <= 1e-9,
            "{time} != {expected_time}"
        );
    }
    Ok(())
}

#[test]
fn prepared_clocked_hierarchy_preserves_occurrence_initialization_and_complement()
-> anyhow::Result<()> {
    use ngspice_rs::analysis::RunConfig;
    use ngspice_rs_preprocess::{
        Cancellation, CodeModelRuntime, InputFormat, Options, Request, SourceInput, prepare,
    };
    use std::fs;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("clocked.net");
    fs::write(
        &path,
        r"* prepared clocked state
Vd d 0 5
Vc clk 0 PWL(0 0 1u 0 1.02u 5 2u 5)
X1 a ab d clk CELL
X2 b bb d clk CELL
.INIT X1.!D1=1 X2.!D1=0
.subckt CELL q qb d c
!D1 q qb 0 d c M IC=0
.MODEL M CLK_DFF TH=2.5 HYSTWD=1 VOL=0.5 VOH=4.5 RIN=10Meg ROUT=10 LOGIC=POS TRIG_COND=0_TO_1
.ends CELL
R1 a 0 100
R2 ab 0 100
R3 b 0 100
R4 bb 0 100
.end
",
    )?;
    let bundle = prepare(
        &Request {
            format: InputFormat::SimplisDeck,
            input: SourceInput::File(path),
            libraries: Vec::new(),
            search_paths: Vec::new(),
        },
        &Options::default(),
        &Cancellation::default(),
    )?;
    assert_eq!(bundle.required_code_models(), [CodeModelRuntime::Logic]);
    for artifact in bundle.artifacts() {
        fs::write(dir.path().join(&artifact.filename), &artifact.bytes)?;
    }
    let deck = Parser::new().parse_file(dir.path().join(&bundle.primary.filename))?;
    let config = RunConfig::from_netlist(&deck)?;
    let mut c = config.prepared_circuit(
        &deck,
        ngspice_rs::devices::subckt::PreparedDeclarations {
            waveforms: &bundle.waveforms,
            pwl_resistors: &bundle.pwl_resistors,
        },
    )?;
    let op =
        AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
            .resolve()?
            .run(&mut c, &config.context())?;
    for (node, rail) in [("a", 4.5), ("ab", 0.5), ("b", 0.5), ("bb", 4.5)] {
        close(op.value(&format!("v({node})"), 0).unwrap().re, rail / 1.1);
    }
    let tran = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1n", "2u", "0", "1n"])
        .resolve()?
        .run(&mut c, &config.context())?;
    for (node, rail) in [("a", 4.5), ("ab", 0.5), ("b", 4.5), ("bb", 0.5)] {
        close(
            tran.value(&format!("v({node})"), tran.point_count() - 1)
                .unwrap()
                .re,
            rail / 1.1,
        );
    }
    assert!(c.last_digital_stats().unwrap().evaluations > 0);
    Ok(())
}

#[test]
fn source_bound_digital_bus_resolves_strength_and_unknown() -> SpiceResult<()> {
    for data in [0, 4] {
        for enable in [0, 4] {
            for pull in [0, 4] {
                let mut c = circuit(&format!(
                    "vdata d 0 {data}\nvenable e 0 {enable}\nvpull p 0 {pull}\n                    a1 %v[d e] bus tri\n.model tri preprocess_logic_event(operation=21 delay_paths=[1n 2n 3n 4n 5n 6n])\n                    a2 %v[p] bus weak\n.model weak preprocess_logic_event(operation=25 strength=1)\n                    a3 %d[bus] direct buf\n.model buf preprocess_logic_event(operation=25)\n                    a4 direct out bridge\n.model bridge preprocess_logic_voltage(out_high=5)\nr1 out 0 1k"
                ))?;
                let expected = if enable == 4 { data } else { pull };
                close(
                    output(&run(&mut c, AnalysisKind::OperatingPoint, &[])?, 0),
                    if expected == 4 { 5. } else { 0. },
                );
                assert!(c.nodes().get("bus").is_none());
                assert!(c.nodes().get("direct").is_none());
            }
        }
    }
    // A disconnected input is unknown, but AND with a decisive zero is
    // still zero. Invert the second digital terminal after net resolution.
    let mut c = circuit(
        "vzero zero 0 0\na1 %v[zero zero] floating tri\n.model tri preprocess_logic_event(operation=21)\n         a2 %v[zero] known buf\n.model buf preprocess_logic_event(operation=25)\n         a3 %d[floating known] result gate\n.model gate preprocess_logic_event(operation=0)\n         a4 %d[floating ~known] uncertain gate\n         a5 result out bridge\na6 uncertain other bridge\n.model bridge preprocess_logic_voltage(out_high=5)\n         r1 out 0 1k\nr2 other 0 1k",
    )?;
    let plot = run(&mut c, AnalysisKind::OperatingPoint, &[])?;
    close(output(&plot, 0), 0.);
    close(plot.value("v(other)", 0).unwrap().re, 2.5);
    Ok(())
}

#[test]
fn source_bound_asymmetric_tristate_delivery() -> SpiceResult<()> {
    let mut c = circuit(
        "vd data 0 4\nve en 0 PWL(0 0 1u 0 1002n 4 3u 4 3002n 0 5u 0)\n         vp pull 0 0\n         a1 %v[data en] bus tri\n.model tri preprocess_logic_event(operation=21 delay_paths=[10n 20n 30n 40n 50n 60n])\n         a2 %v[pull] bus weak\n.model weak preprocess_logic_event(operation=25 strength=1)\n         a3 %d[bus] result copy\n.model copy preprocess_logic_event(operation=25)\n         a4 result out bridge\n.model bridge preprocess_logic_voltage(out_high=5)\nr1 out 0 1k",
    )?;
    let plot = run(&mut c, AnalysisKind::Transient, &["1n", "5u", "0", "10n"])?;
    let edges: Vec<_> = (1..plot.point_count())
        .filter(|&i| (output(&plot, i) > 2.5) != (output(&plot, i - 1) > 2.5))
        .map(|i| (plot.value("time", i).unwrap().re, output(&plot, i)))
        .collect();
    assert_eq!(edges.len(), 2, "{edges:?}");
    // Qualified ADC strict crossing: rising at 3 V, falling at 2 V.
    // The existing one-sided plot convention shows a new voltage at the
    // first sample after the event (within the 10 ns maximum step).
    for ((time, voltage), (at, expected)) in
        edges.into_iter().zip([(1.0615e-6, 5.), (3.041e-6, 0.)])
    {
        close(voltage, expected);
        assert!((time - at).abs() <= 10e-9, "{time:e} != {at:e}");
    }
    Ok(())
}
