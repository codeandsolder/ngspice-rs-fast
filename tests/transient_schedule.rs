use std::path::Path;

use ngspice_rs::analysis::{
    AnalysisInput, RunConfig, TimingSegment, TransientSchedule, companion_transient,
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
    assert!(TransientSchedule::new(0., vec![], output.clone()).is_err());
    assert!(TransientSchedule::new(0., output.clone(), vec![segment(2., 0.1)]).is_err());
    assert!(TransientSchedule::new(1., output.clone(), output.clone()).is_err());
    assert!(TransientSchedule::new(0., vec![segment(1., 0.)], output.clone()).is_err());
    assert!(TransientSchedule::new(0., vec![segment(1., 0.1), segment(0.5, 0.1)], output).is_err());
    let mut request = AnalysisInput::with_arguments(AnalysisKind::Transient, ["1u", "1m"]);
    request.transient_schedule = Some(schedule(0.));
    assert!(request.resolve().is_err());
    let mut request = AnalysisInput::new(AnalysisKind::OperatingPoint);
    request.transient_schedule = Some(schedule(0.));
    assert!(request.resolve().is_err());
}
