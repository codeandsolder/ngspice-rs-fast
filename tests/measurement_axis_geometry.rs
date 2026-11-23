//! Independent geometry regressions for retained and online quadrature.
//! The coordinates are exactly the supplied binary64 values, not intervals
//! obtained by independently rounding their differences.

use std::path::Path;

use ngspice_rs::analysis::optimization::objectives::{
    ObjectiveRequests, PlotObjectivePlan, ScenarioPlan,
};
use ngspice_rs::analysis::{Plot, PlotFlags, PointSink, Variable, measure};
use ngspice_rs::netlist::ast::{
    MeasureCard, MeasureRequest, MeasureStatistic, MeasureWindow, RequestedVector, VectorRequest,
};
use ngspice_rs::netlist::eval::ParamScope;
use ngspice_rs::primitives::{AnalysisKind, Complex, Real, SourceLoc};

fn plot(points: &[(Real, Real)]) -> Plot {
    let mut plot = Plot::new(
        "tran1",
        "Transient Analysis",
        PlotFlags::Real,
        vec![
            Variable::new("time", "time"),
            Variable::new("v(out)", "voltage"),
        ],
    );
    for &(axis, ordinate) in points {
        plot.push_point(&[Complex::real(axis), Complex::real(ordinate)])
            .unwrap();
    }
    plot
}

fn card(statistic: MeasureStatistic, window: MeasureWindow) -> MeasureCard {
    let loc = SourceLoc::new(Path::new("geometry.cir").to_path_buf(), 1, 1);
    MeasureCard {
        analysis: AnalysisKind::Transient,
        analysis_location: loc.clone(),
        name: statistic.name().to_owned(),
        name_location: loc.clone(),
        request: MeasureRequest::Statistic {
            statistic,
            operand: VectorRequest {
                vector: RequestedVector::Named {
                    name: "v(out)".into(),
                    component: None,
                },
                location: loc.clone(),
            },
            window,
        },
        location: loc,
    }
}

fn retained_and_online(
    points: &[(Real, Real)],
    statistics: &[MeasureStatistic],
    window: MeasureWindow,
) -> Vec<Real> {
    let plot = plot(points);
    let cards: Vec<_> = statistics
        .iter()
        .map(|&statistic| card(statistic, window))
        .collect();
    let retained = measure::resolve(&plot, AnalysisKind::Transient, &cards).unwrap();
    let plan = PlotObjectivePlan::bind(
        &plot.header(),
        AnalysisKind::Transient,
        ObjectiveRequests {
            measurements: cards,
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    assert_eq!(plan.retained_columns(), []);
    let scenario = ScenarioPlan::new(vec![plan]);
    let scope = ParamScope::root(&[]).unwrap();
    let mut sink = scenario.sink(&scope);
    sink.begin(&plot.header()).unwrap();
    for row in plot.rows() {
        sink.point(row).unwrap();
    }
    sink.end().unwrap();
    let online = sink.finish().unwrap().remove(0).measurements.unwrap();
    assert_eq!(
        online, retained,
        "streamed and retained geometry must agree"
    );
    retained.into_iter().map(|value| value.value).collect()
}

#[test]
fn constant_max_on_nonuniform_binary64_axes_never_overflows_the_mean() {
    let coordinates = [
        2.655_773_717_743_297,
        4.131_474_704_983_489,
        6.919_626_425_451_377,
        9.015_713_440_187_866,
    ];
    let points: Vec<_> = coordinates.into_iter().map(|x| (x, Real::MAX)).collect();
    let result = retained_and_online(
        &points,
        &[MeasureStatistic::Avg, MeasureStatistic::Rms],
        MeasureWindow::default(),
    );
    assert_eq!(result, [Real::MAX, Real::MAX]);
}

#[test]
fn unrepresentable_axis_width_still_has_a_finite_normalized_mean_and_area() {
    let samples = [(-Real::MAX, 1e-308), (Real::MAX, 1e-308)];
    let result = retained_and_online(
        &samples,
        &[
            MeasureStatistic::Avg,
            MeasureStatistic::Rms,
            MeasureStatistic::Integ,
        ],
        MeasureWindow::default(),
    );
    assert_eq!(result[0].to_bits(), 1e-308_f64.to_bits());
    assert_eq!(result[1].to_bits(), 1e-308_f64.to_bits());
    let area = 2. * (Real::MAX * 1e-308);
    assert!((result[2] - area).abs() < 2e-15, "{result:?}");
}

#[test]
fn clipped_window_and_repeated_axis_jumps_keep_their_actual_coverage() {
    let points = [(0., 0.), (1., 0.), (1., 10.), (2., 10.)];
    let result = retained_and_online(
        &points,
        &[
            MeasureStatistic::Avg,
            MeasureStatistic::Rms,
            MeasureStatistic::Integ,
        ],
        MeasureWindow::default(),
    );
    assert_eq!(result[0], 5.);
    assert_eq!(result[2], 10.);
    assert!((result[1] - (50_f64).sqrt()).abs() <= 4. * Real::EPSILON * result[1]);

    let result = retained_and_online(
        &points,
        &[MeasureStatistic::Avg, MeasureStatistic::Integ],
        MeasureWindow {
            from: Some(0.5),
            to: Some(1.5),
            ..MeasureWindow::default()
        },
    );
    assert_eq!(result, [5., 5.]);
}

#[test]
fn exact_dyadic_trapezoids_agree_with_independent_integer_rationals() {
    // Integer denominators are powers of two, so all binary64 inputs below
    // are exact. An i128 oracle forms the rational trapezoid numerator and
    // total width without using the production weighted-sum implementation.
    let mut seed = 0xded3_74ae_19fc_ba87_u64;
    for _ in 0..256 {
        let mut x = 0_i128;
        let mut previous_y = 0_i128;
        let mut expected_numerator = 0_i128;
        let mut points = Vec::new();
        for index in 0..17 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let y = i128::from((seed % 1001) as i64 - 500);
            if index > 0 {
                let step = i128::from(((seed >> 13) % 31 + 1) as i64);
                x += step;
                expected_numerator += step * (previous_y + y);
            }
            points.push((x as Real / 1024., y as Real / 4096.));
            previous_y = y;
        }
        let rational = expected_numerator as Real / (2. * x as Real * 4096.);
        let result =
            retained_and_online(&points, &[MeasureStatistic::Avg], MeasureWindow::default());
        let tolerance = 2. * Real::EPSILON * rational.abs().max(1.);
        assert!(
            (result[0] - rational).abs() <= tolerance,
            "{points:?} {result:?}"
        );
    }
}
