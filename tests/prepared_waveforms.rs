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

fn pwl(knots: Vec<Knot>, repeat: Option<Repeat>, dc: f64, startup: f64) -> PreparedForcing {
    PreparedForcing::new(ResolvedWaveform {
        dc,
        startup,
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
                limit: None,
            }),
            0.,
            if origin < 0. { 3. } else { 5. },
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
            limit: None,
        }),
        -3.,
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
        startup: 1.,
        forcing: Forcing::PiecewiseLinear {
            knots: vec![knot(0., 0., 1.), knot(0., 1., 2.)],
            repeat: None,
        },
        idle_in_pop: false,
    };
    assert!(PreparedForcing::new(invalid).is_err());
    let wave = PreparedForcing::new(ResolvedWaveform {
        dc: 0.,
        startup: 1. + 2. * (-0.1_f64).exp(),
        forcing: Forcing::Sine {
            offset: 1.,
            amplitude: 2.,
            frequency: 1.,
            delay: -0.25,
            damping: 0.4,
            hold_before_delay: false,
            phase: 0.,
            stop_after: None,
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
            .prepared_circuit(
                &netlist,
                ngspice_rs::devices::subckt::PreparedDeclarations {
                    waveforms: &bundle.waveforms,
                    pwl_resistors: &bundle.pwl_resistors,
                },
            )
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
        let mut input =
            AnalysisInput::with_arguments(AnalysisKind::Transient, [step, "1m", "0", step]);
        input.uic = true;
        let plot = input
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
        .prepared_circuit(
            &netlist,
            ngspice_rs::devices::subckt::PreparedDeclarations {
                waveforms: &bundle.waveforms,
                pwl_resistors: &bundle.pwl_resistors,
            },
        )
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
    assert!(
        config
            .prepared_circuit(
                &netlist,
                ngspice_rs::devices::subckt::PreparedDeclarations {
                    waveforms: &duplicate,
                    ..Default::default()
                }
            )
            .is_err()
    );
    let mut missing = bundle.waveforms.clone();
    missing[0].source = "missing_source".into();
    assert!(
        config
            .prepared_circuit(
                &netlist,
                ngspice_rs::devices::subckt::PreparedDeclarations {
                    waveforms: &missing,
                    ..Default::default()
                }
            )
            .is_err()
    );
}

#[test]
fn explicit_root_candidate_rebinds_without_changing_source_rows_or_ac() {
    use std::sync::Arc;

    use model_normalize_ngspice::waveform::{
        Period, PreparedWaveform, PulseEdges, PulseSpec, WaveformSpec,
    };
    use ngspice_rs::devices::Circuit;
    use ngspice_rs::netlist::eval::{EvalBudget, ParamBinding, ParamScope};
    use ngspice_rs::netlist::source::parse_deck_text;

    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("deck.cir"),
            "candidate witness\n.param height=1\nV1 a 0 DC 0 AC 2 90\nR1 a 0 1k\n.end\n",
        ))
        .unwrap();
    let mut declarations = [PreparedWaveform {
        artifact: "deck.cir".into(),
        scope: Vec::new(),
        source: "v1".into(),
        spec: WaveformSpec::Pulse(PulseSpec {
            first: "0".into(),
            second: "{height}".into(),
            period: Period::Seconds("1m".into()),
            delay: "0".into(),
            hold_before_delay: "0".into(),
            edges: PulseEdges::Rectangular { duty: "0.5".into() },
        }),
        dc_override: None,
        idle_in_pop: "0".into(),
    }];
    for (height, dc_override) in [(1., None), (3., Some(7.)), (-2., Some(0.))] {
        declarations[0].dc_override = dc_override.map(|value: f64| value.to_string());
        let scope = ParamScope::resolve_instance(
            None,
            &[ParamBinding {
                name: "height".into(),
                value: height,
                location: netlist.location.clone(),
            }],
            &netlist.params,
            &mut EvalBudget::default(),
        )
        .unwrap();
        let mut circuit = Circuit::from_prepared_netlist_with_limits(
            &netlist,
            &Default::default(),
            Default::default(),
            Default::default(),
            Some(Arc::new(scope)),
            ngspice_rs::devices::subckt::PreparedDeclarations {
                waveforms: &declarations,
                ..Default::default()
            },
        )
        .unwrap();
        let linear = circuit.linear_system().unwrap();
        let [source] = linear.sources.as_slice() else {
            panic!("one independent source expected")
        };
        assert_eq!(source.name.as_str(), "v1");
        close(source.dc, dc_override.unwrap_or(height), 0.);
        close(source.waveform.value_at(0., Limit::Left).unwrap(), 0., 0.);
        close(
            source.waveform.value_at(0., Limit::Right).unwrap(),
            height,
            0.,
        );
        close(source.ac.re, 0., 1e-14);
        close(source.ac.im, 2., 1e-14);
        let op =
            AnalysisInput::with_arguments(AnalysisKind::OperatingPoint, std::iter::empty::<&str>())
                .resolve()
                .unwrap()
                .run(&mut circuit, &Default::default())
                .unwrap();
        close(
            op.value("v(a)", 0).unwrap().re,
            dc_override.unwrap_or(height),
            1e-12,
        );
        let plan = AnalysisInput::with_arguments(AnalysisKind::Transient, ["10u", "100u"])
            .resolve()
            .unwrap();
        let plot = plan.run(&mut circuit, &Default::default()).unwrap();
        close(plot.value("v(a)", 0).unwrap().re, height, 1e-12);
    }
}

#[test]
fn rounded_periodic_knots_cannot_silently_collapse() {
    let period = 1.169423730035861e-89;
    let origin = -1.3266700643923667e-90;
    let wave = pwl(
        vec![
            knot(0., 0., 0.),
            knot(9.999089847601106e-90, 0., 1.),
            knot(9.999089847601113e-90, 1., 0.),
        ],
        Some(Repeat {
            origin,
            period,
            hold: None,
            limit: None,
        }),
        0.,
        0.,
    );
    let begin = 2_f64.mul_add(period, origin);
    let end = 3_f64.mul_add(period, origin);
    assert!(wave.breakpoints_in(begin, end).is_err());
    assert!(wave.value_at(begin, Limit::Right).is_err());
}

#[test]
fn prepared_knot_jets_use_exact_incoming_and_outgoing_segments() {
    let wave = Waveform::Prepared(pwl(
        vec![knot(-2., -1., 2.), knot(1., 5., 7.), knot(3., 3., 9.)],
        None,
        4.,
        4.,
    ));
    for (time, limit, value, slope) in [
        (-3., Limit::Right, -1., 0.),
        (-2., Limit::Left, -1., 0.),
        (-2., Limit::Right, 2., 0.5),
        (0., Limit::Right, 4., 0.5),
        (1., Limit::Left, 5., 0.5),
        (1., Limit::Right, 7., -1.),
        (3., Limit::Left, 3., -1.),
        (3., Limit::Right, 9., 0.),
    ] {
        let jet = wave.taylor_at(time, 0.5, 3, limit).unwrap();
        assert_eq!(jet, [value, slope, 0., 0.]);
    }
}

#[test]
fn prepared_repeat_jets_follow_previous_cycle_and_startup_hold() {
    for hold in [None, Some(17.)] {
        let wave = Waveform::Prepared(pwl(
            vec![knot(0., 6., 2.), knot(0.25, 4., 8.)],
            Some(Repeat {
                origin: -0.25,
                period: 1.,
                hold,
                limit: None,
            }),
            0.,
            8.,
        ));
        for cycle in [-2., -1., 0., 1., 100.] {
            if hold.is_some() && cycle < 0. {
                continue;
            }
            let t = -0.25 + cycle;
            let left = wave.taylor_at(t, 0.25, 2, Limit::Left).unwrap();
            if let Some(value) = hold.filter(|_| cycle == 0.) {
                assert_eq!(left, [value, 0., 0.]);
            } else {
                assert_eq!(left, [6., -2. / 3., 0.]);
            }
            assert_eq!(
                wave.taylor_at(t, 0.25, 2, Limit::Right).unwrap(),
                [2., 2., 0.]
            );
            assert_eq!(
                wave.taylor_at(t + 0.25, 0.25, 2, Limit::Left).unwrap(),
                [4., 2., 0.]
            );
            assert_eq!(
                wave.taylor_at(t + 0.25, 0.25, 2, Limit::Right).unwrap(),
                [8., -2. / 3., 0.]
            );
        }
    }
}

#[test]
fn prepared_affine_jets_do_not_overflow_duration_or_round_away_subnormals() {
    let max = f64::MAX;
    let wave = Waveform::Prepared(pwl(
        vec![knot(-max, -max, -max), knot(max, max, max)],
        None,
        0.,
        0.,
    ));
    assert_eq!(
        wave.taylor_at(0., 1., 2, Limit::Right).unwrap(),
        [0., 1., 0.]
    );
    let tiny = f64::from_bits(1);
    let wave = Waveform::Prepared(pwl(
        vec![knot(-max, 0., 0.), knot(max / 2., tiny, tiny)],
        None,
        0.,
        0.,
    ));
    assert_eq!(
        wave.taylor_at(-max, max, 1, Limit::Right).unwrap(),
        [0., tiny]
    );
}

#[test]
fn prepared_sine_jets_respect_delay_hold_and_exact_analytic_derivatives() {
    for hold in [false, true] {
        let wave = Waveform::Prepared(
            PreparedForcing::new(ResolvedWaveform {
                dc: 2.,
                startup: (3. * (-0.5_f64).exp()).mul_add(1., 2.),
                forcing: Forcing::Sine {
                    offset: 2.,
                    amplitude: 3.,
                    frequency: 1.,
                    delay: -0.25,
                    damping: 2.,
                    hold_before_delay: hold,
                    phase: 0.,
                    stop_after: None,
                },
                idle_in_pop: false,
            })
            .unwrap(),
        );
        let phase = std::f64::consts::FRAC_PI_4;
        let damping = 0.25;
        let expected = [
            2.,
            3. * phase,
            -3. * damping * phase,
            3. * (0.5 * damping * damping * phase - phase.powi(3) / 6.),
        ];
        for limit in [Limit::Left, Limit::Right] {
            let actual = wave.taylor_at(-0.25, 0.125, 3, limit).unwrap();
            if hold && limit == Limit::Left {
                assert_eq!(actual, [2., 0., 0., 0.]);
            } else {
                for (a, b) in actual.into_iter().zip(expected) {
                    close(a, b, 2e-15);
                }
            }
        }
    }
}

fn derivative_deck(body: &str) -> (ngspice_rs::netlist::ast::Netlist, RunConfig) {
    let deck =
        format!("prepared derivative witness\n{body}\n.options reltol=1e-8 vntol=1n\n.end\n");
    let netlist = Parser::new()
        .parse_deck(&ngspice_rs::netlist::source::parse_deck_text(
            Path::new("derivative.cir"),
            &deck,
        ))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    (netlist, config)
}

fn derivative_declaration(
    spec: model_normalize_ngspice::waveform::WaveformSpec,
) -> model_normalize_ngspice::waveform::PreparedWaveform {
    model_normalize_ngspice::waveform::PreparedWaveform {
        artifact: "derivative.cir".into(),
        scope: Vec::new(),
        source: "v1".into(),
        spec,
        dc_override: None,
        idle_in_pop: "0".into(),
    }
}

#[test]
fn prepared_pwl_derivative_changes_slope_at_exact_event() {
    use model_normalize_ngspice::waveform::{SymbolicPoint, WaveformSpec};
    let (netlist, config) = derivative_deck("V1 in 0 0\nB1 out 0 v=1u*ddt(v(in))\nR1 out 0 1k");
    let declaration = derivative_declaration(WaveformSpec::PiecewiseLinear {
        segments: "2".into(),
        points: [("0", "0"), ("10u", "1"), ("20u", "0")]
            .map(|(time, value)| SymbolicPoint {
                time: time.into(),
                value: value.into(),
            })
            .to_vec(),
    });
    for method in ["method=trap", "method=gear"] {
        for uic in [false, true] {
            let mut circuit = config
                .prepared_circuit(
                    &netlist,
                    ngspice_rs::devices::subckt::PreparedDeclarations {
                        waveforms: std::slice::from_ref(&declaration),
                        ..Default::default()
                    },
                )
                .unwrap();
            let timelines = circuit
                .source_timelines(TransientTiming::new(1.7e-6, 25e-6).unwrap())
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let events = timelines[0]
                .waveform
                .breakpoints_in(0., 25e-6)
                .unwrap()
                .collect::<Vec<_>>();
            assert_eq!(events.len(), 3);
            let mut input = AnalysisInput::with_arguments(
                AnalysisKind::Transient,
                ["1.7u", "25u", "0", "1.7u", method],
            );
            input.uic = uic;
            let plot = config
                .plan(input)
                .unwrap()
                .run(&mut circuit, &config.context())
                .unwrap();
            for event in &events {
                // UIC suppresses the t=0 output row by design.
                if uic && *event == 0. {
                    continue;
                }
                assert!(
                    (0..plot.point_count())
                        .any(|i| plot.value("time", i).unwrap().re.to_bits() == event.to_bits())
                );
            }
            for index in 0..plot.point_count() {
                let t = plot.value("time", index).unwrap().re;
                let expected = if t < events[1] {
                    0.1
                } else if t < events[2] {
                    -0.1
                } else {
                    0.
                };
                close(plot.value("v(out)", index).unwrap().re, expected, 1e-10);
                close(
                    plot.value("i(b1)", index).unwrap().re,
                    -expected * 1e-3,
                    1e-13,
                );
            }
        }
    }
}

#[test]
fn prepared_signed_sine_drives_nested_derivatives_and_kcl() {
    use model_normalize_ngspice::waveform::{SineDelay, WaveformSpec};
    let (netlist, config) = derivative_deck(
        "V1 in 0 0\nB1 first 0 v=1m*ddt(v(in))\nB2 out 0 v=1m*ddt(v(first))\nR1 out 0 1k",
    );
    let declaration = derivative_declaration(WaveformSpec::Sine {
        offset: "2".into(),
        amplitude: "0.5".into(),
        frequency: "1k".into(),
        delay: SineDelay::Seconds("-125u".into()),
        damping: "500".into(),
        hold_before_delay: "0".into(),
    });
    for method in ["method=trap", "method=gear"] {
        let mut circuit = config
            .prepared_circuit(
                &netlist,
                ngspice_rs::devices::subckt::PreparedDeclarations {
                    waveforms: std::slice::from_ref(&declaration),
                    ..Default::default()
                },
            )
            .unwrap();
        let mut input = AnalysisInput::with_arguments(
            AnalysisKind::Transient,
            ["2u", "80u", "0", "2u", method],
        );
        input.uic = true;
        let plot = config
            .plan(input)
            .unwrap()
            .run(&mut circuit, &config.context())
            .unwrap();
        let omega = std::f64::consts::TAU;
        let decay = 0.5_f64;
        for index in 0..plot.point_count() {
            let elapsed = 1e3 * plot.value("time", index).unwrap().re + 0.125;
            let amplitude = 0.5 * (-decay * elapsed).exp();
            let (sin, cos) = (omega * elapsed).sin_cos();
            let first = amplitude * (omega * cos - decay * sin);
            let second =
                amplitude * ((decay * decay - omega * omega) * sin - 2. * decay * omega * cos);
            close(plot.value("v(first)", index).unwrap().re, first, 1e-8);
            close(plot.value("v(out)", index).unwrap().re, second, 1e-8);
            close(
                plot.value("i(b2)", index).unwrap().re,
                -second * 1e-3,
                1e-11,
            );
        }
    }
}

#[test]
fn prepared_source_startup_bias_and_actual_jump_have_distinct_semantics() {
    use model_normalize_ngspice::waveform::{Period, PulseEdges, PulseSpec, WaveformSpec};
    let (netlist, config) = derivative_deck("V1 in 0 0\nB1 out 0 v=1u*ddt(v(in))\nR1 out 0 1k");
    let declaration = |delay: &str, held: bool| {
        derivative_declaration(WaveformSpec::Pulse(PulseSpec {
            first: "2".into(),
            second: "3".into(),
            period: Period::Seconds("1m".into()),
            delay: delay.into(),
            hold_before_delay: if held { "1" } else { "0" }.into(),
            edges: PulseEdges::Rectangular {
                duty: "0.75".into(),
            },
        }))
    };
    for method in ["method=trap", "method=gear"] {
        let mut circuit = config
            .prepared_circuit(
                &netlist,
                ngspice_rs::devices::subckt::PreparedDeclarations {
                    waveforms: &[declaration("0", false)],
                    ..Default::default()
                },
            )
            .unwrap();
        let input = AnalysisInput::with_arguments(
            AnalysisKind::Transient,
            ["2u", "80u", "0", "2u", method],
        );
        let plot = config
            .plan(input)
            .unwrap()
            .run(&mut circuit, &config.context())
            .unwrap();
        // Prepared bias is the declared right-limit startup, not an invented
        // pulse-low prehistory; there is no derivative impulse at zero.
        for index in 0..plot.point_count() {
            close(plot.value("v(in)", index).unwrap().re, 3., 1e-12);
            close(plot.value("v(out)", index).unwrap().re, 0., 1e-12);
        }
        for uic in [false, true] {
            let mut circuit = config
                .prepared_circuit(
                    &netlist,
                    ngspice_rs::devices::subckt::PreparedDeclarations {
                        waveforms: &[declaration("10u", true)],
                        ..Default::default()
                    },
                )
                .unwrap();
            let mut input = AnalysisInput::with_arguments(
                AnalysisKind::Transient,
                ["2u", "80u", "0", "2u", method],
            );
            input.uic = uic;
            let error = config
                .plan(input)
                .unwrap()
                .run(&mut circuit, &config.context())
                .expect_err("an actual source jump requires an impulse")
                .to_string();
            assert!(
                error.contains("source event") && error.contains("derivative input jumps"),
                "{error}"
            );
        }
    }
}

#[test]
fn prepared_periodic_jets_agree_with_every_generated_interval_at_wide_time_scales() {
    for exponent in [-300, -100, 0, 100, 300] {
        let period = 2_f64.powi(exponent);
        for origin in [-0.37 * period, 0.19 * period] {
            let wave = Waveform::Prepared(pwl(
                vec![
                    knot(0., -3., 5.),
                    knot(0.1 * period, 2., -4.),
                    knot(0.73 * period, 7., 1.),
                ],
                Some(Repeat {
                    origin,
                    period,
                    hold: None,
                    limit: None,
                }),
                0.,
                if origin < 0. {
                    // Fraction of the second interval, measured in period units.
                    -4. + 11. * (0.37 - 0.1) / (0.73 - 0.1)
                } else {
                    // Previous cycle: interpolate its last knot to the first left limit.
                    1. - 4. * (0.81 - 0.73) / (1. - 0.73)
                },
            ));
            for cycle in [-1024_f64, -1., 0., 1., 1024., 1048576.] {
                let begin = cycle.mul_add(period, origin);
                let end = (cycle + 1.).mul_add(period, origin);
                let times = wave.breakpoints_in(begin, end).unwrap().collect::<Vec<_>>();
                assert_eq!(times.len(), 4);
                for interval in times.windows(2) {
                    let (a, b) = (interval[0], interval[1]);
                    let width = b - a;
                    let va = wave.value_at(a, Limit::Right).unwrap();
                    let vb = wave.value_at(b, Limit::Left).unwrap();
                    let slope = vb - va;
                    let right = wave.taylor_at(a, width, 2, Limit::Right).unwrap();
                    let left = wave.taylor_at(b, width, 2, Limit::Left).unwrap();
                    assert_eq!(right[0], va);
                    assert_eq!(left[0], vb);
                    close(right[1], slope, 2e-14);
                    close(left[1], slope, 2e-14);
                    assert_eq!(right[2], 0.);
                    assert_eq!(left[2], 0.);
                }
            }
        }
    }
}

#[test]
fn finite_repetition_preserves_exact_cutoff_and_does_not_scan_after_it() {
    use model_normalize_ngspice::waveform::RepeatLimit;
    for (cutoff, incoming, slope, expected_events) in [
        (1.5, 4., 1., vec![0., 0.5, 1., 1.5]),
        (1.75, 4., -2., vec![0., 0.5, 1., 1.5, 1.75]),
        (2., 2., -2., vec![0., 0.5, 1., 1.5, 2.]),
    ] {
        let wave = pwl(
            vec![knot(0., 2., 2.), knot(0.5, 4., 6.)],
            Some(Repeat {
                origin: 0.,
                period: 1.,
                hold: Some(0.),
                limit: Some(RepeatLimit {
                    time: cutoff,
                    value: -7.,
                }),
            }),
            2.,
            2.,
        );
        assert_eq!(
            wave.breakpoints_in(0., f64::MAX)
                .unwrap()
                .map(|event| event.time)
                .collect::<Vec<_>>(),
            expected_events
        );
        assert_eq!(
            wave.breakpoints_in(cutoff, f64::MAX)
                .unwrap()
                .map(|event| event.time)
                .collect::<Vec<_>>(),
            [cutoff]
        );
        assert_eq!(
            wave.breakpoints_in(cutoff.next_up(), f64::MAX)
                .unwrap()
                .count(),
            0
        );
        close(wave.value_at(cutoff, Limit::Left).unwrap(), incoming, 0.);
        close(wave.value_at(cutoff, Limit::Right).unwrap(), -7., 0.);
        close(wave.value_at(f64::MAX, Limit::Left).unwrap(), -7., 0.);
        let waveform = Waveform::Prepared(wave);
        assert_eq!(
            waveform.taylor_at(cutoff, 0.25, 2, Limit::Left).unwrap(),
            [incoming, slope, 0.]
        );
        assert_eq!(
            waveform.taylor_at(cutoff, 0.25, 2, Limit::Right).unwrap(),
            [-7., 0., 0.]
        );
    }
}

#[test]
fn signed_and_finite_sine_uses_phase_and_exact_hold_boundaries() {
    let delay = 0.125;
    for (frequency, amplitude, damping, phase, cap, hold) in [
        (2., 3., 0.4, 30., Some(0.625), true),
        (-2., -3., -0.4, 30., Some(-0.625), true),
        (-2., 3., 0.4, -30., Some(-0.625), false),
        (0., -3., 0.4, 90., None, true),
        (2., 3., 0.4, 30., None, false),
    ] {
        let phase_rad = f64::to_radians(phase);
        let omega = frequency * std::f64::consts::TAU;
        let at = |elapsed: f64| {
            let a = amplitude * (-damping * elapsed).exp();
            let (sin, cos) = omega.mul_add(elapsed, phase_rad).sin_cos();
            (a.mul_add(sin, 1.), a * omega.mul_add(cos, -damping * sin))
        };
        let elapsed_at_zero = if hold {
            0.
        } else {
            cap.map_or(-delay, |c: f64| c.min(-delay))
        };
        let forcing = PreparedForcing::new(ResolvedWaveform {
            dc: 7.,
            startup: at(elapsed_at_zero).0,
            idle_in_pop: false,
            forcing: Forcing::Sine {
                offset: 1.,
                amplitude,
                frequency,
                delay,
                damping,
                phase,
                hold_before_delay: hold,
                stop_after: cap,
            },
        })
        .unwrap();
        let events = forcing
            .breakpoints_in(-1., 2.)
            .unwrap()
            .map(|event| event.time)
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        if hold {
            expected.push(delay);
        }
        if let Some(cap) = cap
            && (!hold || cap > 0.)
        {
            expected.push(delay + cap);
        }
        expected.sort_by(f64::total_cmp);
        assert_eq!(events, expected);
        close(forcing.dc(), 7., 0.);
        let wave = Waveform::Prepared(forcing);
        let mut samples = vec![-0.75, -0.25, 0., delay, 0.25, 1., 2.];
        samples.extend(expected);
        for time in samples {
            for side in [Limit::Left, Limit::Right] {
                let (elapsed, active) =
                    if hold && (time < delay || (time == delay && side == Limit::Left)) {
                        (0., false)
                    } else if let Some(cap) = cap
                        && (time > delay + cap || (time == delay + cap && side == Limit::Right))
                    {
                        (cap, false)
                    } else {
                        (time - delay, true)
                    };
                let (value, derivative) = at(elapsed);
                let jet = wave.taylor_at(time, 0.125, 2, side).unwrap();
                close(jet[0], value, 2e-13);
                close(jet[1], if active { 0.125 * derivative } else { 0. }, 2e-13);
                if !active {
                    close(jet[2], 0., 0.);
                }
            }
        }
    }
}

#[test]
fn exponential_independent_delays_and_analytic_derivatives() {
    use model_normalize_ngspice::waveform::ExponentialForcing;
    for (rise_delay, fall_delay) in [(-0.5, 0.25), (0.25, -0.5), (0., 0.), (0.25, 0.75)] {
        let exp = ExponentialForcing {
            initial: 1.,
            pulsed: 3.,
            rise_delay,
            fall_delay,
            rise_tau: 0.25,
            fall_tau: 0.5,
        };
        let transition = |time: f64, delay: f64, tau: f64, side| {
            if time < delay || (time == delay && side == Limit::Left) {
                (0., 0., 0.)
            } else {
                let e = (-(time - delay) / tau).exp();
                (-(-(time - delay) / tau).exp_m1(), e / tau, -e / (tau * tau))
            }
        };
        let expected = |time, side| {
            let r = transition(time, rise_delay, exp.rise_tau, side);
            let f = transition(time, fall_delay, exp.fall_tau, side);
            [2_f64.mul_add(r.0 - f.0, 1.), 2. * (r.1 - f.1), r.2 - f.2]
        };
        let forcing = PreparedForcing::new(ResolvedWaveform {
            dc: 7.,
            startup: expected(0., Limit::Right)[0],
            idle_in_pop: false,
            forcing: Forcing::Exponential(exp),
        })
        .unwrap();
        let mut events = vec![rise_delay, fall_delay];
        events.sort_by(f64::total_cmp);
        events.dedup();
        assert_eq!(
            forcing
                .breakpoints_in(-1., 3.)
                .unwrap()
                .map(|event| event.time)
                .collect::<Vec<_>>(),
            events
        );
        let wave = Waveform::Prepared(forcing);
        let mut times = vec![-1., 0., 0.5, 1., 3.];
        times.extend(events);
        for time in times {
            for side in [Limit::Left, Limit::Right] {
                for (value, want) in wave
                    .taylor_at(time, 1., 2, side)
                    .unwrap()
                    .iter()
                    .zip(expected(time, side))
                {
                    close(*value, want, 2e-14);
                }
            }
        }
    }
}

#[test]
fn actual_ltspice_prepared_waveforms_drive_transient_with_exact_cutoffs() {
    let dir = tempfile::tempdir().unwrap();
    let bundle = prepare_deck(
        dir.path(),
        "* linked LTspice waveform witness\nV1 a 0 DC 7 SINE(1 2 1Meg 1u 1e5 30 1.5)\nV2 b 0 EXP(1 3 5u 2u 1u 3u)\nI1 0 c PULSE(0 1 1u .1u .1u 8u 10u 1.5)\nR1 c 0 1\nV3 d 0 SINE(1 2 0 1u 1e5 30)\n.end\n",
        InputFormat::LtspiceDeck,
    );
    assert_eq!(bundle.waveforms.len(), 4);
    let netlist = Parser::new()
        .parse_file(dir.path().join(&bundle.primary.filename))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let mut circuit = config
        .prepared_circuit(
            &netlist,
            ngspice_rs::devices::subckt::PreparedDeclarations {
                waveforms: &bundle.waveforms,
                pwl_resistors: &bundle.pwl_resistors,
            },
        )
        .unwrap();
    let mut input =
        AnalysisInput::with_arguments(AnalysisKind::Transient, ["50n", "20u", "0", "50n"]);
    input.uic = true;
    let plot = config
        .plan(input)
        .unwrap()
        .run(&mut circuit, &config.context())
        .unwrap();
    let cutoff = 1.5_f64.mul_add(10e-6, 1e-6);
    for point in 0..plot.point_count() {
        let time = plot.value("time", point).unwrap().re;
        let elapsed = (time - 1e-6).clamp(0., 1.5e-6);
        let a = (2. * (-1e5 * elapsed).exp()).mul_add(
            (std::f64::consts::TAU * 1e6)
                .mul_add(elapsed, 30_f64.to_radians())
                .sin(),
            1.,
        );
        let rise = -(-(time - 5e-6).max(0.) / 2e-6).exp_m1();
        let fall = -(-(time - 1e-6).max(0.) / 3e-6).exp_m1();
        let b = 2_f64.mul_add(rise - fall, 1.);
        let phase = (time - 1e-6).max(0.) % 10e-6;
        let c = if time >= cutoff {
            0.
        } else {
            (phase / 1e-7)
                .min(1.)
                .min(((8.2e-6 - phase) / 1e-7).max(0.))
        };
        let d = 2_f64.mul_add(
            30_f64.to_radians().sin() * (-1e5 * (time - 1e-6).max(0.)).exp(),
            1.,
        );
        for (name, want) in [("v(a)", a), ("v(b)", b), ("v(c)", c), ("v(d)", d)] {
            close(plot.value(name, point).unwrap().re, want, 2e-8);
        }
    }
    assert!(
        (0..plot.point_count())
            .any(|point| plot.value("time", point).unwrap().re.to_bits() == cutoff.to_bits())
    );
}
