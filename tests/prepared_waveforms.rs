//! Producer-to-solver tests for explicitly sided independent-source forcing.
use std::fs;
use std::path::Path;

use model_normalize_ngspice::waveform::{Forcing, Knot, Repeat, ResolvedWaveform};
use ngspice_rs::analysis::{AnalysisInput, RunConfig};
use ngspice_rs::devices::prepared_waveform::PreparedForcing;
use ngspice_rs::devices::{Limit, TransientTiming, Waveform};
use ngspice_rs::netlist::Parser;
use ngspice_rs::primitives::AnalysisKind;
use ngspice_rs_preprocess::{
    Cancellation, InputFormat, Options, PreparedBundle, Request, SourceInput, prepare,
};

fn knot(time: f64, left: f64, right: f64) -> Knot {
    Knot { time, left, right }
}

fn pwl(knots: Vec<Knot>, repeat: Option<Repeat>, dc: f64) -> PreparedForcing {
    PreparedForcing::new(ResolvedWaveform {
        dc,
        forcing: Forcing::PiecewiseLinear { knots, repeat },
        idle_in_pop: false,
    })
    .unwrap()
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual:.17e} != {expected:.17e} (tolerance {tolerance})"
    );
}

#[test]
fn aperiodic_external_limits_and_adjacent_events_are_exact() {
    let next = 1_f64.next_up();
    let wave = pwl(
        vec![knot(-2., -1., 2.), knot(1., 5., 7.), knot(next, 7., 9.)],
        None,
        4.,
    );
    for (t, side, expected) in [
        (-3., Limit::Right, -1.),
        (-2., Limit::Left, -1.),
        (-2., Limit::Right, 2.),
        (0., Limit::Right, 4.),
        (1., Limit::Left, 5.),
        (1., Limit::Right, 7.),
        (next, Limit::Left, 7.),
        (next, Limit::Right, 9.),
        (3., Limit::Left, 9.),
    ] {
        close(wave.value_at(t, side).unwrap(), expected, 1e-15);
    }
    assert_eq!(
        wave.breakpoints_in(-3., 3.)
            .unwrap()
            .map(|e| e.time.to_bits())
            .collect::<Vec<_>>(),
        [(-2_f64).to_bits(), 1_f64.to_bits(), next.to_bits()]
    );
}

#[test]
fn signed_cycles_hold_and_lazy_repetition_preserve_limits() {
    for origin in [-0.25, 0.25] {
        let wave = pwl(
            vec![knot(0., 6., 2.)],
            Some(Repeat {
                origin,
                period: 1.,
                hold: None,
            }),
            0.,
        );
        for cycle in [-2., -1., 0., 1., 100.] {
            let t = origin + cycle;
            close(wave.value_at(t, Limit::Left).unwrap(), 6., 0.);
            close(wave.value_at(t, Limit::Right).unwrap(), 2., 0.);
            close(wave.value_at(t + 0.5, Limit::Right).unwrap(), 4., 0.);
        }
        let events = wave
            .breakpoints_in(0., 1e9)
            .unwrap()
            .take(3)
            .map(|e| e.time)
            .collect::<Vec<_>>();
        let first = if origin < 0. { origin + 1. } else { origin };
        assert_eq!(events, [first, first + 1., first + 2.]);
    }
    let held = pwl(
        vec![knot(0., 6., 2.), knot(0.5, 2., 6.)],
        Some(Repeat {
            origin: 0.25,
            period: 1.,
            hold: Some(-3.),
        }),
        -3.,
    );
    close(held.value_at(0.25, Limit::Left).unwrap(), -3., 0.);
    close(held.value_at(0.25, Limit::Right).unwrap(), 2., 0.);
    close(held.value_at(1.25, Limit::Left).unwrap(), 6., 0.);
    close(held.value_at(1.25, Limit::Right).unwrap(), 2., 0.);
    assert_eq!(held.breakpoints_in(-100., 0.2).unwrap().count(), 0);
}

#[test]
fn invalid_geometry_is_rejected_and_sine_respects_signed_delay() {
    let invalid = ResolvedWaveform {
        dc: 0.,
        forcing: Forcing::PiecewiseLinear {
            knots: vec![knot(0., 0., 1.), knot(0., 1., 2.)],
            repeat: None,
        },
        idle_in_pop: false,
    };
    assert!(PreparedForcing::new(invalid).is_err());
    let wave = PreparedForcing::new(ResolvedWaveform {
        dc: 0.,
        forcing: Forcing::Sine {
            offset: 1.,
            amplitude: 2.,
            frequency: 1.,
            delay: -0.25,
            damping: 0.4,
            hold_before_delay: false,
        },
        idle_in_pop: true,
    })
    .unwrap();
    close(
        wave.value_at(0., Limit::Right).unwrap(),
        1. + 2. * (-0.1_f64).exp(),
        1e-14,
    );
    assert!(wave.idle_in_pop());
    assert!(!wave.is_piecewise_linear());
}

fn prepare_deck(dir: &Path, source: &str, format: InputFormat) -> PreparedBundle {
    let input = dir.join("source.net");
    fs::write(&input, source).unwrap();
    let bundle = prepare(
        &Request {
            format,
            input: SourceInput::File(input),
            libraries: Vec::new(),
            search_paths: Vec::new(),
        },
        &Options::default(),
        &Cancellation::default(),
    )
    .unwrap();
    for artifact in bundle.artifacts() {
        fs::write(dir.join(&artifact.filename), &artifact.bytes).unwrap();
    }
    bundle
}

#[test]
fn rectangular_current_pulse_delivers_exact_charge_independent_of_step() {
    let dir = tempfile::tempdir().unwrap();
    let bundle = prepare_deck(
        dir.path(),
        r"* charge witness
I1 0 out PUL V1=0 V2=1m PERIOD=1m DRATIO=0.25 DELAY=125u OFF_UNTIL_DELAY=YES
C1 out 0 1u
.end
",
        InputFormat::ExpandedDeck,
    );
    let netlist = Parser::new()
        .parse_file(dir.path().join(&bundle.primary.filename))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    for step in ["200u", "37u", "8u"] {
        let mut circuit = config
            .prepared_circuit(&netlist, &bundle.waveforms)
            .unwrap();
        let timeline = circuit
            .source_timelines(TransientTiming::new(1e-5, 1e-3).unwrap())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(matches!(&timeline[0].waveform, Waveform::Prepared(_)));
        let events = timeline[0]
            .waveform
            .breakpoints_in(0., 1e-3)
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(events.len(), 2);
        let plot =
            AnalysisInput::with_arguments(AnalysisKind::Transient, [step, "1m", "0", step, "uic"])
                .resolve()
                .unwrap()
                .run(&mut circuit, &config.context())
                .unwrap();
        for event in events {
            assert!(
                (0..plot.point_count())
                    .any(|i| plot.value("time", i).unwrap().re.to_bits() == event.to_bits()),
                "missing exact event {event:.17e}"
            );
        }
        close(
            plot.value("v(out)", plot.point_count() - 1).unwrap().re,
            0.25,
            1e-7,
        );
    }
}

#[test]
fn actual_linked_occurrences_bind_independently() {
    let dir = tempfile::tempdir().unwrap();
    let bundle = prepare_deck(
        dir.path(),
        r"* per-occurrence phase and shape
.param lo=2
X1 a 0 cell params: f=1k duty=0.5 edge=0 gate=0
X2 b 0 cell params: f=2k duty=0.25 edge=10u gate=1
.subckt cell p n params: f=1k duty=0.5 edge=0 gate=0
Vraw p n PUL V1={lo} V2=6 FREQ={f} T_RISE={edge} T_FALL=0 PWIDTH={duty/f} DELAY={0.75/f} OFF_UNTIL_DELAY={gate}
.ends
R1 a 0 1k
R2 b 0 1k
.end
",
        InputFormat::SimetrixDeck,
    );
    let netlist = Parser::new()
        .parse_file(dir.path().join(&bundle.primary.filename))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let circuit = config
        .prepared_circuit(&netlist, &bundle.waveforms)
        .unwrap();
    let timelines = circuit
        .source_timelines(TransientTiming::new(1e-5, 1e-3).unwrap())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(timelines.len(), 2);
    for timeline in &timelines {
        assert!(matches!(&timeline.waveform, Waveform::Prepared(_)));
    }
    let mut startup = timelines
        .iter()
        .map(|t| t.waveform.time_zero().unwrap())
        .collect::<Vec<_>>();
    startup.sort_by(f64::total_cmp);
    assert_eq!(startup, [2., 6.]);
    let mut duplicate = bundle.waveforms.clone();
    duplicate.push(duplicate[0].clone());
    assert!(config.prepared_circuit(&netlist, &duplicate).is_err());
    let mut missing = bundle.waveforms.clone();
    missing[0].source = "missing_source".into();
    assert!(config.prepared_circuit(&netlist, &missing).is_err());
}
