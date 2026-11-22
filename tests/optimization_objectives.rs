//! Objective-only execution shares measurement policies with retained plots.
use ngspice_rs::analysis::optimization::objectives::{
    ObjectiveRequests, Observation, ObservationFailure, ObservationReduction, ObservationValue,
    PlotObjectivePlan, PlotObjectives, ScenarioPlan,
};
use ngspice_rs::analysis::{Plot, PlotFlags, PointSink, RunConfig, Variable, fourier, measure};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::{
    MeasureCard, MeasureEvent, MeasureRequest, MeasureStatistic, MeasureTransition, MeasureWindow,
    RequestedVector, VectorRequest,
};
use ngspice_rs::netlist::eval::ParamScope;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{AnalysisKind, Complex, Real, SourceLoc};
use std::path::Path;

fn parsed(body: &str) -> ngspice_rs::netlist::parser::ParsedDeck {
    Parser::new()
        .parse_deck_with_output(&parse_deck_text(
            Path::new("objective.cir"),
            &format!("Objectives\n{body}\n.end\n"),
        ))
        .unwrap()
}

fn request(name: &str) -> VectorRequest {
    VectorRequest {
        vector: RequestedVector::Named {
            name: name.into(),
            component: None,
        },
        location: SourceLoc::new(Path::new("objective.cir").to_path_buf(), 1, 1),
    }
}

fn observation(name: &str, vector: &str, reduction: ObservationReduction) -> Observation {
    Observation {
        name: name.into(),
        vector: request(vector),
        reduction,
    }
}

fn header(width: usize) -> Plot {
    let mut plot = Plot::builder("tran1", "Transient Analysis", PlotFlags::Real);
    plot.push_variable(Variable::new("time", "time"));
    plot.push_variable(Variable::new("v(out)", "voltage"));
    plot.push_variable(Variable::new("v(ref)", "voltage"));
    for i in 3..width {
        plot.push_variable(Variable::new(format!("v(extra{i})"), "voltage"));
    }
    plot.build()
}

fn plot(rows: &[(Real, Real, Real)], width: usize) -> Plot {
    let mut plot = header(width);
    let mut row = vec![Complex::real(19.); width];
    for &(x, y, z) in rows {
        row[..3].copy_from_slice(&[Complex::real(x), Complex::real(y), Complex::real(z)]);
        plot.push_point(&row).unwrap();
    }
    plot
}

fn finish(plan: &ScenarioPlan, plot: &Plot, scope: &ParamScope) -> PlotObjectives {
    let mut sink = plan.sink(scope);
    sink.begin(&plot.header()).unwrap();
    for row in plot.rows() {
        sink.point(row).unwrap();
    }
    sink.end().unwrap();
    sink.finish().unwrap().remove(0)
}

#[test]
fn static_measurements_are_identical_without_retaining_rows() {
    let parsed = parsed(
        ".param base=2\n.func plus(x) {x+peak+base}\n\
         .meas tran derived param='plus(1)'\n\
         .meas tran peak MAX v(out) FROM=0.25 TO=3.75\n\
         .meas tran low MIN v(out)\n\
         .meas tran lo MIN_AT v(out)\n\
         .meas tran hi MAX_AT v(out)\n\
         .meas tran pp PP v(out)\n\
         .meas tran mean AVG v(out) FROM=0.25 TO=3.75\n\
         .meas tran area INTEG v(out) FROM=0.25 TO=3.75\n\
         .meas tran rms RMS v(out) FROM=0.25 TO=3.75\n\
         .meas tran lookup FIND v(out) AT=2.25\n\
         .meas tran rising WHEN v(out)=1 RISE=2\n\
         .meas tran moving WHEN v(out)=v(ref) CROSS=LAST\n\
         .meas tran distance TRIG v(out) VAL=1 RISE=1 TARG v(out) VAL=1 FALL=LAST",
    );
    let scope = ParamScope::for_netlist(&parsed.netlist).unwrap();
    let full = plot(
        &[
            (0., 0., 0.5),
            (0.5, 2., 0.5),
            (1., 0., 0.5),
            (1.5, 0., 0.5),
            (1.5, 2., 0.5),
            (2., 2., 1.),
            (2.5, 0., 1.),
            (3., 2., 1.),
            (3.5, 2., 1.),
            (4., 0., 1.),
        ],
        65,
    );
    let bound = PlotObjectivePlan::bind(
        &full,
        AnalysisKind::Transient,
        ObjectiveRequests {
            measurements: parsed.measurements.clone(),
            observations: vec![observation(
                "sample",
                "v(out)",
                ObservationReduction::At(2.25),
            )],
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    assert_eq!(bound.required_columns(), [0, 1, 2]);
    assert_eq!(bound.retained_columns(), []);
    let plan = ScenarioPlan::new(vec![bound]);
    let expected =
        measure::resolve_with_scope(&full, AnalysisKind::Transient, &parsed.measurements, &scope)
            .unwrap();
    let got = finish(&plan, &full, &scope);
    assert_eq!(got.measurements.unwrap(), expected);
    assert_eq!(got.retained_cells, 0);
    assert_eq!(got.accepted_points, 10);
    assert!(
        matches!(got.observations[0].value, Ok(ObservationValue::Scalar(point)) if point.value == Complex::real(1.))
    );
    // Reusing the same plan for another candidate does not reuse reductions.
    let second = plot(
        &[
            (0., 2., 1.),
            (1., 0., 1.),
            (2., 2., 1.),
            (3., 0., 1.),
            (4., 2., 1.),
        ],
        65,
    );
    assert_eq!(
        finish(&plan, &second, &scope).measurements,
        measure::resolve_with_scope(
            &second,
            AnalysisKind::Transient,
            &parsed.measurements,
            &scope
        )
    );
}

fn card(request: MeasureRequest) -> MeasureCard {
    let location = request_location();
    MeasureCard {
        analysis: AnalysisKind::Transient,
        analysis_location: location.clone(),
        name: "metric".into(),
        name_location: location.clone(),
        request,
        location,
    }
}

fn request_location() -> SourceLoc {
    SourceLoc::new(Path::new("objective.cir").to_path_buf(), 1, 1)
}

fn same_result(
    got: &ngspice_rs::primitives::SpiceResult<Vec<measure::Measurement>>,
    expected: &ngspice_rs::primitives::SpiceResult<Vec<measure::Measurement>>,
    card: &MeasureCard,
) {
    match (got, expected) {
        (Ok(got), Ok(expected)) => assert_eq!(got, expected, "{card:?}"),
        (Err(got), Err(expected)) => assert_eq!(
            std::mem::discriminant(got),
            std::mem::discriminant(expected),
            "{card:?}: {got} vs {expected}"
        ),
        _ => panic!("{card:?}: streamed {got:?}; retained {expected:?}"),
    }
}

fn window_requests(window: MeasureWindow) -> impl Iterator<Item = MeasureRequest> {
    let statistics = [
        MeasureStatistic::Min,
        MeasureStatistic::Max,
        MeasureStatistic::MinAt,
        MeasureStatistic::MaxAt,
        MeasureStatistic::PeakToPeak,
        MeasureStatistic::Avg,
        MeasureStatistic::Integ,
        MeasureStatistic::Rms,
    ];
    statistics
        .into_iter()
        .map(move |statistic| MeasureRequest::Statistic {
            statistic,
            operand: request("v(out)"),
            window,
        })
        .chain(
            [0., 0.5, 1., 1.5, 2., 4., 5.]
                .into_iter()
                .map(move |at| MeasureRequest::Find {
                    operand: request("v(out)"),
                    at,
                    at_location: request_location(),
                    window,
                }),
        )
        .chain(
            [0., 0.5, 1., 1.5, 2., 4., 5.]
                .into_iter()
                .map(move |at| MeasureRequest::AtEvent {
                    operand: request("v(out)"),
                    event: MeasureEvent::At {
                        at,
                        location: request_location(),
                    },
                    derivative: true,
                    window,
                }),
        )
        .chain(
            [
                MeasureTransition::First,
                MeasureTransition::Rise(2),
                MeasureTransition::Fall(1),
                MeasureTransition::Cross(2),
                MeasureTransition::Last,
            ]
            .into_iter()
            .flat_map(move |transition| {
                [
                    MeasureEvent::Crossing {
                        operand: request("v(out)"),
                        value: 0.,
                        value_location: request_location(),
                        transition,
                    },
                    MeasureEvent::VectorCrossing {
                        operand: request("v(out)"),
                        reference: request("v(ref)"),
                        transition,
                    },
                ]
                .into_iter()
                .flat_map(move |event| {
                    [
                        MeasureRequest::When {
                            event: event.clone(),
                            window,
                        },
                        MeasureRequest::AtEvent {
                            event: event.clone(),
                            operand: request("v(ref)"),
                            derivative: false,
                            window,
                        },
                        MeasureRequest::AtEvent {
                            event,
                            operand: request("v(ref)"),
                            derivative: true,
                            window,
                        },
                    ]
                })
            }),
        )
}

#[test]
fn windows_crossings_and_exact_query_groups_match_retained() {
    let scope = ParamScope::root(&[]).unwrap();
    let samples = [
        vec![
            (0., 0., 1.),
            (1., 2., 0.),
            (1., -1., 0.),
            (2., 1., 1.),
            (3., 1., 0.),
            (4., -2., 1.),
        ],
        vec![(0., 0., 1.), (1., 1., 1.), (1., 1., 1.), (2., 2., 1.)],
        vec![
            (0., 0., 1.),
            (1., Real::NAN, 1.),
            (2., 2., 1.),
            (3., 0., 1.),
        ],
        vec![(1., 0., 1.)],
        vec![],
    ];
    let mut count = 0;
    for rows in &samples {
        let full = plot(rows, 3);
        for from in [
            None,
            Some(-1.),
            Some(0.25),
            Some(1.),
            Some(1.25),
            Some(2.),
            Some(5.),
        ] {
            for to in [
                None,
                Some(0.),
                Some(0.75),
                Some(1.),
                Some(2.5),
                Some(4.),
                Some(6.),
            ] {
                let window = MeasureWindow { from, to };
                let requests = window_requests(window);
                for m in requests {
                    let card = card(m);
                    let bound = PlotObjectivePlan::bind(
                        &full,
                        AnalysisKind::Transient,
                        ObjectiveRequests {
                            measurements: vec![card.clone()],
                            ..ObjectiveRequests::default()
                        },
                    )
                    .unwrap();
                    assert_eq!(bound.retained_columns(), []);
                    let got = finish(&ScenarioPlan::new(vec![bound]), &full, &scope);
                    let expected = measure::resolve(
                        &full,
                        AnalysisKind::Transient,
                        std::slice::from_ref(&card),
                    );
                    same_result(&got.measurements, &expected, &card);
                    count += 1;
                }
            }
        }
    }
    eprintln!("{count} online/retained window, crossing and query cases");
}

#[test]
fn arbitrary_history_retains_only_raw_dependencies_and_shares_scalar_scope() {
    let parsed = parsed(
        ".param base=3\n.func shift(x) {x+peak+base}\n\
         .meas tran first param='shift(1)'\n\
         .meas tran peak MAX v(out)\n\
         .meas tran sample FIND v(out) AT={peak/4}\n\
         .meas tran PEAK FIND v(out) AT=0.75\n\
         .meas tran peak param='PEAK+base'\n\
         .meas tran latest param='shift(0)'\n\
         .meas tran deriv DERIV v(ref) AT=0.5",
    );
    let scope = ParamScope::for_netlist(&parsed.netlist).unwrap();
    let full = plot(&[(0., 0., 0.), (0.5, 1., 0.25), (1., 2., 1.)], 65);
    let bound = PlotObjectivePlan::bind(
        &full,
        AnalysisKind::Transient,
        ObjectiveRequests {
            measurements: parsed.measurements.clone(),
            retained_cell_limit: 6,
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    assert_eq!(bound.retained_columns(), [0, 1]);
    let got = finish(&ScenarioPlan::new(vec![bound]), &full, &scope);
    assert_eq!(got.retained_cells, 6);
    assert_eq!(
        got.measurements.unwrap(),
        measure::resolve_with_scope(&full, AnalysisKind::Transient, &parsed.measurements, &scope)
            .unwrap()
    );

    let limited = PlotObjectivePlan::bind(
        &full,
        AnalysisKind::Transient,
        ObjectiveRequests {
            measurements: parsed.measurements,
            retained_cell_limit: 5,
            observations: vec![observation(
                "unaffected",
                "v(out)",
                ObservationReduction::Last,
            )],
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    let got = finish(&ScenarioPlan::new(vec![limited]), &full, &scope);
    assert!(got.measurements.is_err());
    assert!(got.observations[0].value.is_ok());
    assert_eq!(got.retained_cells, 4);
}

#[test]
fn fourier_retains_narrow_raw_history_and_reuses_existing_transform() {
    let parsed = parsed(".four 1 v(out)");
    let scope = ParamScope::for_netlist(&parsed.netlist).unwrap();
    let mut full = header(65);
    let mut row = vec![Complex::ZERO; 65];
    for i in 0..=200 {
        let t = f64::from(i) / 200.;
        row[0] = Complex::real(t);
        row[1] = Complex::real((t * std::f64::consts::TAU).sin());
        full.push_point(&row).unwrap();
    }
    let bound = PlotObjectivePlan::bind(
        &full,
        AnalysisKind::Transient,
        ObjectiveRequests {
            fourier: parsed.fourier.clone(),
            retained_cell_limit: 402,
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    assert_eq!(bound.retained_columns(), [0, 1]);
    let got = finish(&ScenarioPlan::new(vec![bound]), &full, &scope);
    assert_eq!(
        got.fourier.unwrap(),
        fourier::resolve(&full, AnalysisKind::Transient, &parsed.fourier).unwrap()
    );
    assert_eq!(got.retained_cells, 402);
}

#[test]
fn invalid_observations_and_schema_lifecycle_are_explicit() {
    let scope = ParamScope::root(&[]).unwrap();
    let full = plot(&[(0., 0., 0.), (1., 1., 1.), (1., 2., 2.), (2., 3., 3.)], 3);
    let bound = PlotObjectivePlan::bind(
        &full,
        AnalysisKind::Transient,
        ObjectiveRequests {
            observations: vec![
                observation("outside", "v(out)", ObservationReduction::At(3.)),
                observation("jump", "v(out)", ObservationReduction::At(1.)),
                observation(
                    "trace",
                    "v(out)",
                    ObservationReduction::Trace { max_points: 3 },
                ),
                observation("last", "v(out)", ObservationReduction::Last),
            ],
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    let plan = ScenarioPlan::new(vec![bound]);
    let got = finish(&plan, &full, &scope);
    assert_eq!(
        got.observations[0].value,
        Err(ObservationFailure::OutOfRange {
            at: 3.,
            range: Some((0., 2.))
        })
    );
    assert_eq!(
        got.observations[1].value,
        Err(ObservationFailure::Discontinuous)
    );
    assert_eq!(
        got.observations[2].value,
        Err(ObservationFailure::TraceLimit { max_points: 3 })
    );
    assert!(
        matches!(got.observations[3].value, Ok(ObservationValue::Scalar(point)) if point.value == Complex::real(3.))
    );
    let mut sink = plan.sink(&scope);
    assert!(sink.begin(&header(4)).is_err());
    assert!(sink.finish().is_err());
    let mut sink = plan.sink(&scope);
    sink.begin(&full.header()).unwrap();
    assert!(sink.point(&[Complex::ZERO]).is_err());
    assert!(sink.finish().is_err());
    let mut sink = plan.sink(&scope);
    sink.begin(&full.header()).unwrap();
    assert!(sink.finish().is_err());
}

#[test]
fn prepared_rc_scenario_reuses_scalar_objectives_across_candidates() {
    let parsed = parsed(
        "V1 in 0 dc 1 ac 1\nR1 in out 1k\nC1 out 0 1u\n.op\n.ac lin 3 100 300\n.meas ac gain FIND vm(out) AT=200",
    );
    let scope = ParamScope::for_netlist(&parsed.netlist).unwrap();
    let config = RunConfig::from_netlist(&parsed.netlist).unwrap();
    let analyses = parsed
        .netlist
        .analyses
        .iter()
        .map(|card| config.plan_for(card).unwrap())
        .collect::<Vec<_>>();
    let context = config.context();
    let mut full_circuit = config.circuit(&parsed.netlist).unwrap();
    let mut plans = Vec::new();
    let mut expected = Vec::new();
    for analysis in &analyses {
        for full in analysis.run_plots(&mut full_circuit, &context).unwrap() {
            let kind = analysis.kind();
            let requests = if kind == AnalysisKind::OperatingPoint {
                ObjectiveRequests {
                    observations: vec![observation("bias", "v(out)", ObservationReduction::Last)],
                    ..ObjectiveRequests::default()
                }
            } else {
                expected =
                    measure::resolve_with_scope(&full, kind, &parsed.measurements, &scope).unwrap();
                ObjectiveRequests {
                    measurements: parsed.measurements.clone(),
                    observations: vec![observation(
                        "response",
                        "v(out)",
                        ObservationReduction::At(200.),
                    )],
                    ..ObjectiveRequests::default()
                }
            };
            plans.push(PlotObjectivePlan::bind(&full, kind, requests).unwrap());
        }
    }
    let plan = ScenarioPlan::new(plans);
    for _ in 0..3 {
        let mut circuit = config.circuit(&parsed.netlist).unwrap();
        let got = plan.run(&analyses, &mut circuit, &context, &scope).unwrap();
        assert_eq!(got[1].measurements.as_ref().unwrap(), &expected);
        let ObservationValue::Scalar(point) = got[1].observations[0].value.as_ref().unwrap() else {
            panic!("expected scalar");
        };
        let w = std::f64::consts::TAU * 200. * 1e-3;
        let denominator = w.mul_add(w, 1.);
        let analytical = Complex::new(1. / denominator, -w / denominator);
        assert!((point.value.re - analytical.re).abs() < 1e-12);
        assert!((point.value.im - analytical.im).abs() < 1e-12);
        assert!(got.iter().all(|plot| plot.retained_cells == 0));
    }
}

#[test]
fn streaming_margins_and_frequency_observations_reuse_physical_scale() {
    let scope = ParamScope::root(&[]).unwrap();
    let mut builder = Plot::builder("ac1", "AC Analysis", PlotFlags::Complex);
    builder.push_variable(Variable::new("frequency", "frequency"));
    builder.push_variable(Variable::new("v(out)", "voltage"));
    let mut full = builder.build();
    for (frequency, magnitude, degrees) in [
        (1., 2., -80_f64),
        (10., 1., -150.),
        (100., 0.5, -200.),
        (1000., 0.25, -220.),
    ] {
        let phase = degrees.to_radians();
        full.push_point(&[
            Complex::real(frequency),
            Complex::new(magnitude * phase.cos(), magnitude * phase.sin()),
        ])
        .unwrap();
    }
    let parsed = parsed(".meas ac pm PHASE_MARGIN v(out)\n.meas ac gm GAIN_MARGIN v(out)");
    let bound = PlotObjectivePlan::bind(
        &full,
        AnalysisKind::Ac,
        ObjectiveRequests {
            measurements: parsed.measurements.clone(),
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    assert_eq!(bound.retained_columns(), []);
    let got = finish(&ScenarioPlan::new(vec![bound]), &full, &scope);
    assert_eq!(
        got.measurements.unwrap(),
        measure::resolve(&full, AnalysisKind::Ac, &parsed.measurements).unwrap()
    );
    assert_eq!(got.retained_cells, 0);
    // Noise and sensitivity are valid direct frequency objectives even
    // though the deck measurement grammar does not admit them.
    for kind in [
        AnalysisKind::Noise,
        AnalysisKind::Sensitivity,
        AnalysisKind::Distortion,
    ] {
        let bound = PlotObjectivePlan::bind(
            &full,
            kind,
            ObjectiveRequests {
                observations: vec![observation("at", "v(out)", ObservationReduction::At(10.))],
                ..ObjectiveRequests::default()
            },
        )
        .unwrap();
        let got = finish(&ScenarioPlan::new(vec![bound]), &full, &scope);
        let ObservationValue::Scalar(point) = got.observations[0].value.as_ref().unwrap() else {
            panic!("expected scalar");
        };
        assert_eq!(point.value, full.rows().nth(1).unwrap()[1]);
        assert_eq!(got.retained_cells, 0);
    }
}

fn bench(retained: bool) {
    use std::time::Instant;
    let parsed = parsed(
        ".meas tran peak max v(out)\n.meas tran at_half find v(out) at=0.5\n.meas tran area integ v(out)\n.meas tran rms rms v(out)\n.meas tran cross when v(out)=0.5",
    );
    let scope = ParamScope::for_netlist(&parsed.netlist).unwrap();
    let header = header(65);
    let bound = PlotObjectivePlan::bind(
        &header,
        AnalysisKind::Transient,
        ObjectiveRequests {
            measurements: parsed.measurements.clone(),
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    let plan = ScenarioPlan::new(vec![bound]);
    let mut row = vec![Complex::ZERO; 65];
    let count = 50_001_u32;
    let repeats = 20_u32;
    let start = Instant::now();
    let mut peak = 0;
    let mut checksum = 0.;
    for _ in 0..repeats {
        let mut full = header.header();
        let mut sink = plan.sink(&scope);
        if !retained {
            sink.begin(&header).unwrap();
        }
        for i in 0..count {
            let t = f64::from(i) / f64::from(count - 1);
            row[0] = Complex::real(t);
            row[1] = Complex::real(t);
            if retained {
                full.push_point(&row).unwrap();
            } else {
                sink.point(&row).unwrap();
            }
        }
        let values = if retained {
            peak = peak.max(full.cells().len());
            measure::resolve_with_scope(
                &full,
                AnalysisKind::Transient,
                &parsed.measurements,
                &scope,
            )
            .unwrap()
        } else {
            sink.end().unwrap();
            let result = sink.finish().unwrap().remove(0);
            peak = peak.max(result.retained_cells);
            result.measurements.unwrap()
        };
        checksum += values.iter().map(|value| value.value).sum::<Real>();
    }
    eprintln!(
        "retained={retained}; {repeats} candidates x {count} points x 65 columns; elapsed={:?}; peak cells={peak}; checksum={checksum}",
        start.elapsed()
    );
    assert!(
        f64::from(repeats)
            .mul_add(-(2.5 + 1_f64 / 3_f64.sqrt()), checksum)
            .abs()
            < 1e-8
    );
}

#[test]
#[ignore = "objective-only throughput and allocation benchmark"]
fn objective_only_batch() {
    bench(false);
}

#[test]
#[ignore = "full-plot baseline throughput and allocation benchmark"]
fn full_plot_batch() {
    bench(true);
}

#[test]
fn streaming_extreme_rms_crossing_and_derivative_keep_conditioned_arithmetic() {
    let scope = ParamScope::root(&[]).unwrap();
    let tiny = Real::from_bits(1);
    let offset = 2_f64.powi(50);
    let fixtures = [
        (
            ".meas tran m RMS v(out)",
            vec![(0., 1e308, 0.), (tiny, 1e308, 0.), (tiny, 0., 0.), (1e308, 0., 0.)],
            vec![tiny.sqrt() * 1e154],
        ),
        (
            ".meas tran m RMS v(out)",
            vec![(0., tiny, 0.), (1., tiny, 0.), (2., tiny, 0.), (3., tiny, 0.)],
            vec![tiny],
        ),
        (
            ".meas tran m WHEN v(out)=0\n.meas tran q FIND v(ref) WHEN v(out)=0\n.meas tran f FIND v(out) AT=2",
            vec![(0., -Real::MAX, 0.), (4., Real::MAX, 4.)],
            vec![2., 2., 0.],
        ),
        (
            ".meas tran m DERIV v(out) AT=2",
            vec![(0., offset, 0.), (1., offset + 1., 0.), (3., offset + 9., 0.)],
            vec![4.],
        ),
    ];
    for (cards, rows, expected) in fixtures {
        let parsed = parsed(cards);
        let full = plot(&rows, 3);
        let bound = PlotObjectivePlan::bind(
            &full, AnalysisKind::Transient,
            ObjectiveRequests { measurements: parsed.measurements.clone(), ..ObjectiveRequests::default() },
        ).unwrap();
        assert!(bound.retained_columns().is_empty());
        let actual = finish(&ScenarioPlan::new(vec![bound]), &full, &scope);
        assert_eq!(actual.retained_cells, 0);
        let measurements = actual.measurements.unwrap();
        assert_eq!(measurements, measure::resolve(&full, AnalysisKind::Transient, &parsed.measurements).unwrap());
        assert_eq!(measurements.len(), expected.len());
        for (measured, wanted) in measurements.iter().zip(expected) {
            assert!((measured.value - wanted).abs() <= 8. * Real::EPSILON * wanted.abs(),
                "{cards}: {:?} != {wanted}", measured.value);
        }
    }
}
