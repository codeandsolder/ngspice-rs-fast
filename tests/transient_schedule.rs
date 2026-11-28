//! Numerical continuity and exact sampling across independent transient timing partitions.

use std::path::Path;

use ngspice_rs::analysis::{
    AnalysisInput, OutputSampling, RunConfig, TimingSegment, TransientSchedule, companion_transient,
};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::AnalysisKind;

fn segment(until: f64, step: f64) -> TimingSegment {
    TimingSegment { until, step }
}

fn schedule(start: f64) -> TransientSchedule {
    TransientSchedule::new(
        start,
        vec![segment(0.7e-3, 0.1e-3), segment(1.5e-3, 0.25e-3)],
        vec![
            segment(0.4e-3, 10e-6),
            segment(0.93e-3, 2e-6),
            segment(1.5e-3, 5e-6),
        ],
        0.1e-3,
        OutputSampling::Land,
    )
    .unwrap()
}

#[test]
fn continuous_rc_and_rl_state_across_independent_timing_boundaries() {
    for derivative in [false, true] {
        let observer = if derivative {
            "\nb1 obs 0 v=1m*ddt(v(out))\nr2 obs 0 1k"
        } else {
            ""
        };
        let deck = parse_deck_text(
            Path::new("schedule.cir"),
            &format!(
                "schedule\nv1 in 0 1\nr1 in out 1k\nc1 out 0 1u ic=0\nr3 in rl 1k\nl1 rl 0 1 ic=0{observer}\n.options reltol=1e-7 abstol=1e-12 vntol=1e-10\n.end\n"
            ),
        );
        let netlist = Parser::new().parse_deck(&deck).unwrap();
        let config = RunConfig::from_netlist(&netlist).unwrap();
        let mut circuit = config.circuit(&netlist).unwrap();
        let mut request = AnalysisInput::new(AnalysisKind::Transient);
        request.uic = true;
        request.transient_schedule = Some(schedule(0.15e-3));
        let plan = config.plan(request).unwrap();
        let (plot, stats) = companion_transient(&mut circuit, &plan, &config.context()).unwrap();
        let expected = [
            0.15e-3, 0.25e-3, 0.35e-3, 0.45e-3, 0.55e-3, 0.65e-3, 0.7e-3, 0.95e-3, 1.2e-3, 1.45e-3,
            1.5e-3,
        ];
        assert_eq!(plot.point_count(), expected.len());
        assert!(stats.accepted > expected.len());
        assert!(stats.max_step <= 10e-6 * (1. + 1e-12));
        for (i, expected_time) in expected.into_iter().enumerate() {
            let time = plot.value("time", i).unwrap().re;
            assert!((time - expected_time).abs() < 1e-17, "{time}");
            let decay = (-time / 1e-3).exp();
            assert!((plot.value("v(out)", i).unwrap().re - (1. - decay)).abs() < 3e-4);
            assert!((plot.value("i(l1)", i).unwrap().re - (1. - decay) / 1e3).abs() < 3e-7);
            if derivative {
                assert!((plot.value("v(obs)", i).unwrap().re - decay).abs() < 3e-4);
            }
        }
        assert_eq!(
            plot.value("time", expected.len() - 1).unwrap().re.to_bits(),
            1.5e-3_f64.to_bits()
        );
    }
}

#[test]
fn rejects_invalid_schedules_and_conflicting_requests() {
    let output = vec![segment(1., 0.1)];
    assert!(TransientSchedule::new(0., vec![], output.clone(), 0.1, OutputSampling::Land).is_err());
    assert!(
        TransientSchedule::new(
            0.,
            output.clone(),
            vec![segment(2., 0.1)],
            0.1,
            OutputSampling::Land
        )
        .is_err()
    );
    assert!(
        TransientSchedule::new(
            1.,
            output.clone(),
            output.clone(),
            0.1,
            OutputSampling::Land
        )
        .is_err()
    );
    assert!(
        TransientSchedule::new(
            0.,
            vec![segment(1., 0.)],
            output.clone(),
            0.1,
            OutputSampling::Land
        )
        .is_err()
    );
    assert!(
        TransientSchedule::new(
            0.,
            vec![segment(1., 0.1), segment(0.5, 0.1)],
            output,
            0.1,
            OutputSampling::Land
        )
        .is_err()
    );
    let mut request = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"]);
    request.transient_schedule = Some(schedule(0.));
    assert!(request.resolve().is_err());
    let mut request = AnalysisInput::new(AnalysisKind::OperatingPoint);
    request.transient_schedule = Some(schedule(0.));
    assert!(request.resolve().is_err());
}

#[test]
fn interpolated_output_never_changes_the_accepted_trajectory() {
    use ngspice_rs::analysis::{Plot, PointSink, companion_transient_into};
    use ngspice_rs::primitives::{Complex, SpiceResult};
    let text =
        "sample policy\nv1 in 0 1\nr1 in out 1k\nc1 out 0 1u ic=0\n.options reltol=1e-5\n.end\n";
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("interp.cir"), text))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let make_plan = |sampling, start, step, uic| {
        let mut input = AnalysisInput::new(AnalysisKind::Transient);
        input.uic = uic;
        input.transient_schedule = Some(
            TransientSchedule::new(
                start,
                vec![segment(0.7e-3, step), segment(1.5e-3, 0.25e-3)],
                vec![segment(0.4e-3, 15e-6), segment(1.5e-3, 20e-6)],
                10e-6,
                sampling,
            )
            .unwrap(),
        );
        config.plan(input).unwrap()
    };
    for uic in [false, true] {
        let reference_plan = make_plan(OutputSampling::Accepted, 0., 0.1e-3, uic);
        let (reference, stats) = companion_transient(
            &mut config.circuit(&netlist).unwrap(),
            &reference_plan,
            &config.context(),
        )
        .unwrap();
        for (start, step) in [(0.15e-3, 0.1e-3), (0.32e-3, 0.07e-3)] {
            let plan = make_plan(OutputSampling::Interpolate, start, step, uic);
            let (plot, actual_stats) = companion_transient(
                &mut config.circuit(&netlist).unwrap(),
                &plan,
                &config.context(),
            )
            .unwrap();
            assert_eq!(stats, actual_stats);
            let reference_times = reference
                .column("time")
                .unwrap()
                .iter()
                .map(|v| v.re)
                .collect::<Vec<_>>();
            for index in 0..plot.point_count() {
                let t = plot.value("time", index).unwrap().re;
                assert!(t >= start);
                let right = reference_times.partition_point(|value| *value < t);
                assert!(right < reference_times.len());
                let expected = if reference_times[right].to_bits() == t.to_bits() {
                    reference.value("v(out)", right).unwrap().re
                } else {
                    assert!(right > 0);
                    let left = right - 1;
                    let fraction = (t - reference_times[left])
                        / (reference_times[right] - reference_times[left]);
                    (1. - fraction) * reference.value("v(out)", left).unwrap().re
                        + fraction * reference.value("v(out)", right).unwrap().re
                };
                assert!((plot.value("v(out)", index).unwrap().re - expected).abs() < 1e-14);
            }
            assert_eq!(
                plot.value("time", plot.point_count() - 1)
                    .unwrap()
                    .re
                    .to_bits(),
                1.5e-3_f64.to_bits()
            );

            // A selected sink requesting no scale still receives the same
            // interpolated physical column; the wrapper negotiates time
            // internally and strips it before delivery.
            #[derive(Default)]
            struct Selected {
                column: Vec<usize>,
                values: Vec<Complex>,
            }
            impl PointSink for Selected {
                fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
                    self.column.push(header.variable_index("v(out)").unwrap());
                    Ok(())
                }
                fn selected_columns(&self) -> Option<&[usize]> {
                    Some(&self.column)
                }
                fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
                    assert_eq!(values.len(), 1);
                    self.values.push(values[0]);
                    Ok(())
                }
                fn end(&mut self) -> SpiceResult<()> {
                    Ok(())
                }
            }
            let mut selected = Selected::default();
            let selected_stats = companion_transient_into(
                &mut config.circuit(&netlist).unwrap(),
                &plan,
                &config.context(),
                &mut selected,
            )
            .unwrap();
            assert_eq!(stats, selected_stats);
            assert_eq!(
                selected.values,
                plot.column("v(out)").unwrap().iter().collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn interpolated_grid_is_bounded_before_execution() {
    assert!(
        TransientSchedule::new(
            0.,
            vec![segment(1., 1e-12)],
            vec![segment(1., 0.1)],
            0.1,
            OutputSampling::Interpolate
        )
        .is_err()
    );
    // An output increment does not reserve a grid in accepted-point mode.
    assert!(
        TransientSchedule::new(
            0.,
            vec![segment(1., 1e-12)],
            vec![segment(1., 0.1)],
            0.1,
            OutputSampling::Accepted
        )
        .is_ok()
    );
}
