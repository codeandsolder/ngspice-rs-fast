//! Time-domain source randomness with deterministic physical-time sampling.
use ngspice_rs::analysis::{Plot, RunConfig};
use ngspice_rs::devices::Limit;
use ngspice_rs::devices::stochastic::{
    RandomKind, StochasticEventKind, StochasticSpec, StochasticWaveform, TransientNoise,
    TransientRandom,
};
use ngspice_rs::netlist::{Parser, source::parse_deck_text, write_netlist};
use std::path::Path;

fn noise(rms: f64, step: f64) -> StochasticWaveform {
    StochasticWaveform::new(StochasticSpec::Noise(TransientNoise {
        white_rms: rms,
        step,
        alpha: 0.,
        flicker_drive_rms: 0.,
        rts_amplitude: 0.,
        capture_mean: 0.,
        emission_mean: 0.,
    }))
    .unwrap()
}
fn random(kind: RandomKind, step: f64, scale: f64, offset: f64) -> StochasticWaveform {
    StochasticWaveform::new(StochasticSpec::Random(TransientRandom {
        kind,
        step,
        delay: 0.,
        scale,
        offset,
    }))
    .unwrap()
}
fn moments(values: impl Iterator<Item = f64>) -> (usize, f64, f64) {
    let mut n = 0_u64;
    let mut mean = 0.;
    let mut square = 0.;
    for value in values {
        n += 1;
        let delta = value - mean;
        mean += delta / n as f64;
        square += delta * (value - mean);
    }
    (n as usize, mean, square / n as f64)
}
fn plot(deck: &str) -> Plot {
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("trnoise.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let plan = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let plots = plan.run_plots(&mut circuit, &config.context()).unwrap();
    plots.last().unwrap().clone()
}

#[test]
fn white_noise_knots_variance_covariance_and_interpolation() {
    let p = noise(2., 0.001)
        .with_seed(994_651)
        .with_source_name("VNOISE");
    let n = 30_000;
    let (count, average, var) =
        moments((1..=n).map(|k| p.value_at(k as f64 * 0.001, Limit::Right).unwrap()));
    assert_eq!(count, n as usize);
    assert!(average.abs() < 0.045, "knot mean {average}");
    assert!((var - 4.).abs() < 0.15, "knot variance {var}");
    let (count, average_mid, var_mid) =
        moments((1..=n).map(|k| p.value_at((k as f64 + 0.5) * 0.001, Limit::Right).unwrap()));
    assert_eq!(count, n as usize);
    assert!(average_mid.abs() < 0.045);
    assert!((var_mid - 2.).abs() < 0.10, "midpoint variance {var_mid}");
    let q = p.clone();
    for k in [10, 150, 2_000, 20_000] {
        let t = (k as f64 + 0.25) * 0.001;
        assert_eq!(
            p.value_at(t, Limit::Right).unwrap().to_bits(),
            q.value_at(t, Limit::Left).unwrap().to_bits()
        );
    }
}

#[test]
fn random_distributions_match_mean_variance_and_do_not_depend_on_call_order() {
    let count = 30_000;
    for (kind, scale, offset, mean, variance) in [
        (RandomKind::Uniform, 4., -3., -3., 16. / 3.),
        (RandomKind::Gaussian, 2., -3., -3., 4.),
        (RandomKind::Exponential, 1.8, 0.5, 2.3, 1.8 * 1.8),
        (RandomKind::Poisson, 4., -0.5, 3.5, 4.),
        (RandomKind::Poisson, 128., 0., 128., 128.),
    ] {
        let p = random(kind, 0.001, scale, offset).with_seed(0xbad6_0341);
        let data: Vec<_> = (0..count)
            .map(|k| p.value_at(k as f64 * 0.001, Limit::Right).unwrap())
            .collect();
        let (_, got_mean, got_var) = moments(data.iter().copied());
        let variance: f64 = variance;
        let m_tol = 0.055 * variance.sqrt() + 0.02;
        let v_tol = 0.09 * variance + 0.05;
        assert!(
            (got_mean - mean).abs() < m_tol,
            "{kind:?}: mean {got_mean}, expected {mean}"
        );
        assert!(
            (got_var - variance).abs() < v_tol,
            "{kind:?}: variance {got_var}, expected {variance}"
        );
        for k in [0, 50, 200, 28_000] {
            assert_eq!(p.value_at(k as f64 * 0.001, Limit::Right).unwrap(), data[k]);
        }
    }
}

#[test]
fn one_source_random_rewind_and_both_one_sided_limits() {
    let p = random(RandomKind::Uniform, 0.5, 1., 0.25);
    let before = p.value_at(0.5, Limit::Left).unwrap();
    let after = p.value_at(0.5, Limit::Right).unwrap();
    assert_ne!(before, after);
    assert_eq!(before, p.value_at(0.49, Limit::Right).unwrap());
    assert_eq!(p.time_zero(), 0.25);
    assert_eq!(p.clone().with_dc(0.4).unwrap().time_zero(), 0.65);
    let a = p.value_at(15.25, Limit::Right).unwrap();
    let b = p.value_at(300.25, Limit::Right).unwrap();
    let c = p.value_at(15.25, Limit::Right).unwrap();
    assert_eq!(a, c);
    assert_ne!(a, b);
}

#[test]
fn exponential_rts_dwell_statistics_events_and_trial_replay() {
    let mut spec = match noise(0., 0.).spec() {
        StochasticSpec::Noise(n) => n,
        StochasticSpec::Random(_) => unreachable!(),
    };
    spec.rts_amplitude = 4.;
    spec.capture_mean = 0.03;
    spec.emission_mean = 0.05;
    let p = StochasticWaveform::new(StochasticSpec::Noise(spec))
        .unwrap()
        .with_seed(335);
    let mut events = p.events(0., 250.).unwrap();
    let mut prev = 0.;
    let mut capture = Vec::new();
    let mut emission = Vec::new();
    for ordinal in 0..5000 {
        let event = events.next_event().unwrap();
        assert_eq!(event.kind, StochasticEventKind::Telegraph);
        let dwell = event.time - prev;
        if ordinal % 2 == 0 {
            capture.push(dwell)
        } else {
            emission.push(dwell)
        }
        prev = event.time;
        assert!(prev <= 250.);
    }
    let (_, c_mean, _) = moments(capture.into_iter());
    let (_, e_mean, _) = moments(emission.into_iter());
    assert!((c_mean - 0.03).abs() < 0.0018, "capture {c_mean}");
    assert!((e_mean - 0.05).abs() < 0.003, "emission {e_mean}");
    let references: Vec<_> = (0..1000)
        .map(|i| p.value_at(i as f64 * 0.031, Limit::Right).unwrap())
        .collect();
    for i in (0..1000).rev() {
        assert_eq!(
            p.value_at(i as f64 * 0.031, Limit::Right).unwrap(),
            references[i]
        );
    }
}

#[test]
fn netlist_stochastic_sources_parse_write_and_run_without_changing_dc() {
    let deck = "stochastic source deck\n\
vnoise src 0 dc 0.25 trnoise(0.2 1n 0 0)\n\
icurrent src 0 dc -0.01 trrandom(2 2n 0 0.001 0)\n\
r1 src 0 1k\n\
.tran 0.5n 8n\n.end\n";
    let parse = |body: &str| {
        Parser::new()
            .parse_deck(&parse_deck_text(Path::new("trnoise.cir"), body))
            .unwrap()
    };
    let parsed = parse(deck);
    let emitted = write_netlist(&parsed).unwrap();
    let emitted_again = write_netlist(&parse(&emitted)).unwrap();
    assert_eq!(emitted, emitted_again);
    let output = plot(deck);
    let p = noise(0.2, 1e-9)
        .with_source_name("vnoise")
        .with_dc(0.25)
        .unwrap();
    assert!(output.rows().len() > 9, "missing transient grid points");
    for k in 0..output.rows().len() {
        let time = output.value("time", k).unwrap().re;
        let voltage = output.value("v(src)", k).unwrap().re;
        let expected = p.value_at(time, Limit::Right).unwrap();
        assert!(
            (voltage - expected).abs() < 1e-8,
            "at {time:e} source={voltage:e} expected={expected:e}"
        );
    }
}

#[test]
fn seeds_and_source_identity_preserve_aba_under_parallel_evaluation() {
    let a = noise(1., 0.001).with_seed(1).with_source_name("v1");
    let b = noise(1., 0.001).with_seed(2).with_source_name("v1");
    let c = noise(1., 0.001).with_seed(1).with_source_name("i1");
    let base: Vec<_> = (0..2000)
        .map(|i| a.value_at(i as f64 * 0.001, Limit::Right).unwrap())
        .collect();
    let b_first = b.value_at(1.251, Limit::Right).unwrap();
    let c_first = c.value_at(1.251, Limit::Right).unwrap();
    assert_ne!(a.value_at(1.251, Limit::Right).unwrap(), b_first);
    assert_ne!(a.value_at(1.251, Limit::Right).unwrap(), c_first);
    for i in (0..2000).rev() {
        assert_eq!(a.value_at(i as f64 * 0.001, Limit::Right).unwrap(), base[i]);
    }
}

fn average_fourier_power(samples: &[f64], first: usize, last: usize) -> f64 {
    let n = samples.len();
    let mut power = 0.;
    for bin in first..last {
        let theta = std::f64::consts::TAU * bin as f64 / n as f64;
        let (mut re, mut im) = (0., 0.);
        for (index, &sample) in samples.iter().enumerate() {
            let angle = theta * index as f64;
            let (sine, cosine) = angle.sin_cos();
            re += sample * cosine;
            im += sample * sine;
        }
        power += re.mul_add(re, im * im) / n as f64;
    }
    power / (last - first) as f64
}

#[test]
fn finite_band_fractional_noise_has_one_over_f_slope_and_bounded_state() {
    let plan = StochasticSpec::Noise(TransientNoise {
        white_rms: 0.,
        step: 1e-3,
        alpha: 1.,
        flicker_drive_rms: 1.,
        rts_amplitude: 0.,
        capture_mean: 0.,
        emission_mean: 0.,
    });
    let noise = StochasticWaveform::new(plan)
        .unwrap()
        .with_seed(0x2345_1234);
    assert!(std::mem::size_of::<StochasticWaveform>() < 1024);
    let signal: Vec<_> = (0..8192)
        .map(|index| noise.value_at(index as f64 * 1e-3, Limit::Right).unwrap())
        .collect();
    let power_low = average_fourier_power(&signal, 8, 24);
    let power_high = average_fourier_power(&signal, 192, 256);
    assert!(
        power_low > 5. * power_high,
        "colored power did not fall with frequency: low={power_low} high={power_high}"
    );
    assert!(power_low.is_finite() && power_high > 0.);
    // Same fixed seed, physical-time amplitude and events on a cloned process.
    let clone = noise.clone();
    for index in [1, 70, 900, 8100] {
        let t = index as f64 * 1e-3;
        assert_eq!(
            noise.value_at(t, Limit::Right).unwrap().to_bits(),
            clone.value_at(t, Limit::Right).unwrap().to_bits()
        );
    }
}

/// Optimize a noisy physical source across independently labeled candidates.
/// Every candidate owns its physical stream, even when all scalar coordinates
/// happen to match; A/B/A must restore A's exact accepted trajectory.
#[test]
fn optimizer_candidate_identity_is_part_of_stochastic_stream() {
    use ngspice_rs::analysis::optimization::objectives::{
        ObjectiveRequests, Observation, ObservationReduction, ObservationValue, PlotObjectivePlan,
        ScenarioPlan,
    };
    use ngspice_rs::analysis::optimization::parameters::CircuitParameterPlan;
    use ngspice_rs::analysis::optimization::sessions::{
        CandidateJob, CandidateScenario, CandidateWorker, ScenarioAnalyses, ScenarioStart,
        run_parallel,
    };
    use ngspice_rs::netlist::ast::{RequestedVector, VectorRequest};
    use ngspice_rs::netlist::eval::EvalLimits;
    use ngspice_rs::primitives::{AnalysisKind, SourceLoc};
    use std::num::NonZeroUsize;

    let deck = "stochastic candidate sessions\n\
.param load=1000\n\
v1 out 0 dc 0 trnoise(0.2 1n 0 0)\n\
r1 out 0 {load}\n\
.tran 0.5n 10n\n.end\n";
    let source = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("stochastic-sessions.cir"), deck))
        .unwrap();
    let config = RunConfig::from_netlist(&source).unwrap();
    let plan = config.plan_for(&source.analyses[0]).unwrap();
    let mut circuit = config.circuit(&source).unwrap();
    let original = plan.run(&mut circuit, &config.context()).unwrap();
    let out = PlotObjectivePlan::bind(
        &original,
        AnalysisKind::Transient,
        ObjectiveRequests {
            observations: vec![Observation {
                name: "last".into(),
                vector: VectorRequest {
                    vector: RequestedVector::Named {
                        name: "v(out)".into(),
                        component: None,
                    },
                    location: SourceLoc::new(
                        Path::new("stochastic-sessions.cir").to_path_buf(),
                        1,
                        1,
                    ),
                },
                reduction: ObservationReduction::Last,
            }],
            ..ObjectiveRequests::default()
        },
    )
    .unwrap();
    let objectives = ScenarioPlan::new(vec![out]);
    let analyses = [plan];
    let scenarios = [CandidateScenario {
        start: ScenarioStart::Independent,
        analyses: ScenarioAnalyses::Prepared(&analyses),
        objectives: &objectives,
    }];
    let parameters = CircuitParameterPlan::new(source, &["load"]).unwrap();
    let mut worker = CandidateWorker::new(&parameters, EvalLimits::default()).unwrap();
    let job = |id| CandidateJob {
        id,
        values: vec![1000.],
    };
    let mut serial = Vec::new();
    for id in [101, 202, 101, 303, 202, 101] {
        let result = worker.evaluate(&job(id), &scenarios);
        let values = result.outcome.unwrap();
        let ObservationValue::Scalar(obs) = values[0][0].observations[0].value.as_ref().unwrap()
        else {
            panic!("expected final transient voltage");
        };
        serial.push((id, obs.value.re.to_bits()));
    }
    assert_eq!(serial[0].1, serial[2].1);
    assert_eq!(serial[0].1, serial[5].1);
    assert_eq!(serial[1].1, serial[4].1);
    assert_ne!(serial[0].1, serial[1].1);
    assert_ne!(serial[0].1, serial[3].1);

    let mut parallel = Vec::new();
    let stats = run_parallel(
        &parameters,
        &scenarios,
        [101, 202, 101, 303, 202, 101].into_iter().map(job),
        NonZeroUsize::new(3).unwrap(),
        NonZeroUsize::new(6).unwrap(),
        EvalLimits::default(),
        |result| {
            let values = result.outcome.unwrap();
            let ObservationValue::Scalar(obs) =
                values[0][0].observations[0].value.as_ref().unwrap()
            else {
                panic!("expected final transient voltage");
            };
            parallel.push((result.id, obs.value.re.to_bits()));
        },
    )
    .unwrap();
    assert_eq!(stats.succeeded, 6);
    assert_eq!(serial, parallel, "dispatch order changed source stream");
}

#[test]
fn transient_current_source_random_hold_stamps_physical_amperes() {
    let deck = "current stochastic hold\n\
i1 out 0 dc 0.0002 trrandom(1 1n 0 0.0005 0)\n\
r1 out 0 1k\n\
.tran 0.25n 10n\n.end\n";
    let result = plot(deck);
    let p = random(RandomKind::Uniform, 1e-9, 0.0005, 0.)
        .with_source_name("i1")
        .with_dc(0.0002)
        .unwrap();
    let mut distinct_levels = std::collections::BTreeSet::new();
    for k in 0..result.rows().len() {
        let t = result.value("time", k).unwrap().re;
        let v = result.value("v(out)", k).unwrap().re;
        // A saved step endpoint is the left limit at a TRRANDOM update;
        // for interior points both physical one-sided limits agree.
        let expected = -1000. * p.value_at(t, Limit::Left).unwrap();
        assert!(
            (v - expected).abs() < 1e-10,
            "source current mismatch time={t:e}: got {v:e}, expected {expected:e}"
        );
        distinct_levels.insert(v.to_bits());
    }
    assert!(
        distinct_levels.len() >= 4,
        "no held-random sample changes observed"
    );
}

#[test]
fn transient_telegraph_lands_on_seeded_physical_events() {
    let deck = "RTS physical event run\n\
v1 out 0 dc 0 trnoise(0 0 0 0 2 3u 5u)\n\
r1 out 0 1k\n\
.tran 0.2u 40u\n.end\n";
    let result = plot(deck);
    let n = TransientNoise {
        white_rms: 0.,
        step: 0.,
        alpha: 0.,
        flicker_drive_rms: 0.,
        rts_amplitude: 2.,
        capture_mean: 3e-6,
        emission_mean: 5e-6,
    };
    let p = StochasticWaveform::new(StochasticSpec::Noise(n))
        .unwrap()
        .with_source_name("v1");
    let times: Vec<f64> = (0..result.rows().len())
        .map(|i| result.value("time", i).unwrap().re)
        .collect();
    for (i, &t) in times.iter().enumerate() {
        let v = result.value("v(out)", i).unwrap().re;
        let before = p.value_at(t, Limit::Left).unwrap();
        let after = p.value_at(t, Limit::Right).unwrap();
        assert!(
            (v - before).abs() < 1e-12 || (v - after).abs() < 1e-12,
            "telegraph source {v:e} is neither physical one-sided limit at {t:e}"
        );
    }
    let mut events = p.events(0., 40e-6).unwrap();
    let mut matched = 0;
    while let Some(event) = events.next_event() {
        assert_eq!(event.kind, StochasticEventKind::Telegraph);
        let nearest = times
            .iter()
            .copied()
            .fold(f64::INFINITY, |a, t| a.min((t - event.time).abs()));
        assert!(
            nearest < 2e-11,
            "missing physical RTS event at {:e}, nearest {:e}",
            event.time,
            nearest
        );
        matched += 1;
    }
    assert!(matched >= 3, "no meaningful RTS transition coverage");
}

#[test]
fn zero_flicker_exponent_ignores_colored_amplitude_as_upstream_c() {
    let deck = "inactive colored generator\n\
v1 out 0 dc 0 trnoise(0 1n 0 100)\n\
r1 out 0 1k\n\
.tran 0.25n 4n\n.end\n";
    let out = plot(deck);
    assert!(out.rows().len() > 5);
    for row in 0..out.rows().len() {
        assert_eq!(out.value("v(out)", row).unwrap().re, 0.);
    }
}
