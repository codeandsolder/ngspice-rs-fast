//! Native physical timed-switch qualification against the independent
//! ngspice-47 reference, with actual source-deck and MNA execution.
use ngspice_rs::analysis::{AnalysisInput, Plot, RunConfig, companion_transient};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};
use std::path::Path;

fn deck(text: &str) -> SpiceResult<(Circuit, RunConfig)> {
    let source = Parser::new().parse_deck(&parse_deck_text(Path::new("timed.cir"), text))?;
    let config = RunConfig::from_netlist(&source)?;
    let circuit = config.circuit(&source)?;
    Ok((circuit, config))
}

fn plan(
    config: &RunConfig,
    kind: AnalysisKind,
    args: &[&str],
) -> SpiceResult<ngspice_rs::analysis::AnalysisPlan> {
    config.plan(AnalysisInput::with_arguments(kind, args.iter().copied()))
}

fn close(label: &str, actual: f64, expected: f64, relative: f64, absolute: f64) {
    assert!(
        actual.is_finite()
            && (actual - expected).abs() <= expected.abs().mul_add(relative, absolute),
        "{label}: {actual:.17e} != {expected:.17e}"
    );
}

fn value(plot: &Plot, name: &str, index: usize) -> f64 {
    plot.value(name, index).unwrap().re
}

fn times(plot: &Plot) -> Vec<f64> {
    (0..plot.point_count())
        .map(|i| value(plot, "time", i))
        .collect()
}

fn interpolate(plot: &Plot, times: &[f64], name: &str, time: f64) -> f64 {
    let index = times.partition_point(|&t| t < time);
    if index == 0 {
        return value(plot, name, 0);
    }
    if index == times.len() {
        return value(plot, name, index - 1);
    }
    let before = value(plot, name, index - 1);
    let after = value(plot, name, index);
    before + (after - before) * (time - times[index - 1]) / (times[index] - times[index - 1])
}

#[test]
fn bias_and_small_signal_match_independent_reference() -> SpiceResult<()> {
    let (mut c, config) = deck(include_str!("fixtures/native-timed-switch/bias-ac.cir"))?;
    let op = plan(&config, AnalysisKind::OperatingPoint, &[])?.run(&mut c, &config.context())?;
    let ac = plan(&config, AnalysisKind::Ac, &["lin", "1", "1k", "1k"])?
        .run(&mut c, &config.context())?;
    for row in include_str!("fixtures/native-timed-switch/bias-ac.csv")
        .lines()
        .skip(1)
    {
        let f: Vec<_> = row.split(',').collect();
        let branch = format!("i(vb{})", f[0]);
        let control = format!("i(vd{})", f[0]);
        for (name, expected) in [(&branch, f[3]), (&control, f[4])] {
            close(
                name,
                -value(&op, name, 0),
                expected.parse().unwrap(),
                1e-10,
                1e-14,
            );
        }
        for (name, re, im) in [(&branch, f[5], f[6]), (&control, f[7], f[8])] {
            let actual = ac.value(name, 0).unwrap();
            close(name, -actual.re, re.parse().unwrap(), 1e-10, 1e-14);
            close(name, -actual.im, im.parse().unwrap(), 0., 1e-14);
        }
    }
    Ok(())
}

#[test]
fn pulse_matrix_and_device_event_work() -> SpiceResult<()> {
    for step in ["0.25n", "1n"] {
        let (mut c, config) = deck(include_str!(
            "fixtures/native-timed-switch/pulse-matrix.cir"
        ))?;
        let p = plan(&config, AnalysisKind::Transient, &["1n", "6u", "0", step])?;
        let (plot, stats) = companion_transient(&mut c, &p, &config.context())?;
        let t = times(&plot);
        let mut max_relative: f64 = 0.;
        for row in include_str!("fixtures/native-timed-switch/pulse-samples.csv")
            .lines()
            .skip(1)
        {
            let f: Vec<_> = row.split(',').collect();
            let time = f[1].parse().unwrap();
            let expected: f64 = f[2].parse().unwrap();
            let actual = -interpolate(&plot, &t, &format!("i(vb{})", f[0]), time);
            max_relative = max_relative.max((actual - expected).abs() / expected);
            close(
                &format!("{} at {time:e}, step {step}", f[0]),
                actual,
                expected,
                0.002,
                1e-12,
            );
        }
        assert!(stats.device_event_boundary_probes > 0);
        assert!(stats.device_event_trial_probes > 0);
        assert!(stats.device_event_landings > 0);
        eprintln!(
            "pulse matrix {step}: maximum relative current discrepancy {max_relative:e}; {stats:?}"
        );
    }
    Ok(())
}

#[test]
fn closed_loop_reference_and_refinement() -> SpiceResult<()> {
    for step in ["1n", "0.5n"] {
        let (mut c, config) = deck(include_str!("fixtures/native-timed-switch/oscillator.cir"))?;
        let p = plan(&config, AnalysisKind::Transient, &["1n", "10u", "0", step])?;
        let (plot, stats) = companion_transient(&mut c, &p, &config.context())?;
        let sample_times = times(&plot);
        let mut extrema = Vec::new();
        for (i, &time) in sample_times
            .iter()
            .enumerate()
            .take(sample_times.len() - 1)
            .skip(1)
        {
            let before = value(&plot, "v(c)", i - 1);
            let current = value(&plot, "v(c)", i);
            let after = value(&plot, "v(c)", i + 1);
            if time > 1e-7
                && ((current > before && current >= after)
                    || (current < before && current <= after))
            {
                extrema.push((time, current));
            }
        }
        let expected: Vec<_> = include_str!("fixtures/native-timed-switch/oscillator-extrema.csv")
            .lines()
            .skip(1)
            .map(|row| {
                let (t, v) = row.split_once(',').unwrap();
                (t.parse::<f64>().unwrap(), v.parse::<f64>().unwrap())
            })
            .collect();
        assert_eq!(extrema.len(), expected.len(), "{step}: {extrema:?}");
        let mut max_time: f64 = 0.;
        let mut max_voltage: f64 = 0.;
        for ((time, voltage), (et, ev)) in extrema.into_iter().zip(expected) {
            max_time = max_time.max((time - et).abs());
            max_voltage = max_voltage.max((voltage - ev).abs());
            close("oscillator time", time, et, 0., 2e-9);
            close("oscillator voltage", voltage, ev, 0., 1e-4);
        }
        assert!(stats.device_event_retries > 0);
        eprintln!(
            "oscillator {step}: max time {max_time:e} s, voltage {max_voltage:e} V; {stats:?}"
        );
    }
    Ok(())
}

#[test]
fn zero_delay_stiff_load_and_analog_fast_path() -> SpiceResult<()> {
    let (mut c, config) = deck(
        "stiff load\nvdrive a 0 5\nvbias p 0 1\nrload p b 1\na1 %v(a) %gd(b 0) m\n.model m preprocess_timed_switch(ron=1n roff=1T vt=2.5 vh=.5 td_on=0 td_off=0)\n.end\n",
    )?;
    let p = plan(&config, AnalysisKind::OperatingPoint, &[])?;
    close(
        "stiff load",
        value(&p.run(&mut c, &config.context())?, "v(b)", 0),
        1e-9 / (1. + 1e-9),
        1e-8,
        1e-14,
    );
    let (mut c, config) = deck(
        "analog fast path\nv1 in 0 pulse(0 1 1u 10n 10n 1u 4u)\nr1 in out 1k\nc1 out 0 1n\n.end\n",
    )?;
    let p = plan(&config, AnalysisKind::Transient, &["10n", "4u"])?;
    let (_, stats) = companion_transient(&mut c, &p, &config.context())?;
    assert_eq!(stats.device_event_boundary_probes, 0);
    assert_eq!(stats.device_event_trial_probes, 0);
    assert_eq!(stats.device_event_retries, 0);
    assert_eq!(stats.device_event_landings, 0);
    Ok(())
}

#[test]
fn actual_prepared_timed_switch_scopes_and_input_loading() -> anyhow::Result<()> {
    use ngspice_rs_preprocess::{
        Cancellation, CodeModelRuntime, InputFormat, Options, Request, SourceInput, prepare,
    };
    use std::fs;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("timed.net");
    fs::write(
        &path,
        r"* scoped timed switch
Vref rtn 0 0.75
Vdrive a rtn 2.5
Vbias p rtn 1
X1 p rtn a rtn T R=10 INIT=1
X2 p rtn a rtn T R=20 INIT=0
.subckt T p n cp cn params: R=10 OFF=1Meg TOL=100p INIT=1
S1 p n cp cn M IC={INIT}
.model M VSWITCH(RON={max(R,1)}, ROFF={OFF}, VT=2.5, VH=0.5, TIMETOL={max(TOL,1p)}, GIN=0.125)
.ends T
.end
",
    )?;
    let bundle = prepare(
        &Request {
            format: InputFormat::SimetrixDeck,
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
    let source = Parser::new().parse_file(dir.path().join(&bundle.primary.filename))?;
    let config = RunConfig::from_netlist(&source)?;
    let mut c = config.prepared_circuit(
        &source,
        ngspice_rs::devices::subckt::PreparedDeclarations {
            waveforms: &bundle.waveforms,
            pwl_resistors: &bundle.pwl_resistors,
        },
    )?;
    let p = plan(&config, AnalysisKind::OperatingPoint, &[])?;
    let plot = p.run(&mut c, &config.context())?;
    close(
        "branch",
        -value(&plot, "i(vbias)", 0),
        0.100_001,
        1e-10,
        1e-13,
    );
    close("input", -value(&plot, "i(vdrive)", 0), 0.625, 1e-10, 1e-13);
    Ok(())
}

#[test]
fn invalid_scoped_physics_and_ports_are_rejected() {
    for parameter in [
        "ron=0",
        "roff=-1",
        "ron=1e-320",
        "ton=0",
        "toff=-1n",
        "td_on=-1n",
        "td_off=-1n",
        "vh=-1",
        "vt=1e308 vh=1e308",
        "timetol=0",
        "timetol=.1f",
        "ic=2",
        "extra=1",
        "instantaneous=2",
        "instantaneous=1 ton=1n",
        "instantaneous=1 toff=0",
    ] {
        let source = format!(
            "invalid timed model\nvdrive a 0 5\nvbias p 0 1\na1 %v(a) %gd(p 0) m\n.model m preprocess_timed_switch({parameter})\n.end\n"
        );
        assert!(deck(&source).is_err(), "{parameter}");
    }
    for ports in ["%d(a) %gd(p 0)", "%v(a) %v(p)", "%v[a] %gd(p 0)"] {
        let source = format!(
            "invalid timed ports\nvdrive a 0 5\nvbias p 0 1\na1 {ports} m\n.model m preprocess_timed_switch(ron=10)\n.end\n"
        );
        assert!(deck(&source).is_err(), "{ports}");
    }
}

#[test]
fn zero_delay_narrow_pulse_and_analytic_branch_jacobian() -> SpiceResult<()> {
    // At 1.0006 us the rising threshold is crossed, and at 1.0066 us
    // the falling threshold is crossed. Both corners are far closer
    // than the coarse timestep. Current is linear in branch voltage
    // even while the logarithmic conductance ramp is moving.
    let source = "analytic narrow pulse\n\
        vref rtn 0 .75\n\
        vdrive control rtn PWL(0 0 1u 0 1001n 5 1006n 5 1007n 0 2u 0)\n\
        vpos p rtn 2\n\
        vneg n rtn -3\n\
        apos %vd(control rtn) %gd(p rtn) m\n\
        aneg %vd(control rtn) %gd(n rtn) m\n\
        .model m preprocess_timed_switch(ron=10 roff=1Meg vt=2.5 vh=.5 ton=2n toff=4n td_on=0 td_off=0)\n\
        .end\n";
    for max_step in ["100n", "0.1n"] {
        let (mut circuit, config) = deck(source)?;
        let p = plan(
            &config,
            AnalysisKind::Transient,
            &["1n", "2u", "0", max_step],
        )?;
        let (plot, stats) = companion_transient(&mut circuit, &p, &config.context())?;
        let mut reached_on = false;
        for (index, time) in times(&plot).into_iter().enumerate() {
            let phase = ((time - 1.0006e-6) / 2e-9)
                .min((1.0106e-6 - time) / 4e-9)
                .clamp(0., 1.);
            let blend = phase * phase * (-2_f64).mul_add(phase, 3.);
            let expected = 10_f64.powf(5_f64.mul_add(blend, -6.));
            let positive = -value(&plot, "i(vpos)", index);
            let negative = -value(&plot, "i(vneg)", index);
            close("positive branch", positive, 2. * expected, 1e-8, 1e-13);
            close("negative branch", negative, -3. * expected, 1e-8, 1e-13);
            close(
                "branch Jacobian",
                (positive - negative) / 5.,
                expected,
                1e-8,
                1e-13,
            );
            close(
                "unloaded control",
                value(&plot, "i(vdrive)", index),
                0.,
                0.,
                1e-13,
            );
            reached_on |= positive > 0.199;
        }
        assert!(reached_on, "narrow pulse missed at {max_step}: {stats:?}");
        assert!(stats.device_event_landings >= 2, "{stats:?}");
    }
    Ok(())
}

#[test]
fn instantaneous_inertial_pulses_match_independent_scheduler() -> SpiceResult<()> {
    use std::fmt::Write as _;
    // Each analog ramp has a threshold root at the exact binary timestamp
    // of an independently scheduled digital event. Keep the nonzero common
    // mode and physically separate voltage and current control ports.
    let mut wave = String::from("PWL(0 -1");
    let half_ramp = 2_f64.powi(-30);
    let mut old = -1.;
    for row in include_str!("fixtures/native-timed-switch/instant-inertial.txt")
        .lines()
        .skip(1)
    {
        let (time, bit) = row.split_once(' ').unwrap();
        let time: f64 = time.parse().unwrap();
        let next = if bit == "1s" { 1. } else { -1. };
        write!(
            wave,
            " {:.24e} {old} {:.24e} {next}",
            time - half_ramp,
            time + half_ramp
        )
        .unwrap();
        old = next;
    }
    wave.push_str(" 20u -1)");
    let source = format!(
        "instantaneous inertial pulse matrix\n\
        vref rtn 0 .75\n\
        vdrive control rtn {wave}\n\
        idrive rtn sensor {wave}\n\
        vbv p rtn 1\n\
        vbi q rtn -2\n\
        av %vd(control rtn) %gd(p rtn) m\n\
        ai %id(sensor rtn) %gd(q rtn) m\n\
        .model m preprocess_timed_switch(ron=10 roff=1Meg vt=0 vh=0 instantaneous=1 td_on={:.24e} td_off={:.24e})\n\
        .end\n",
        2_f64.powi(-20),
        2_f64.powi(-21)
    );
    let reference: Vec<(f64, bool)> =
        include_str!("fixtures/native-timed-switch/instant-inertial.csv")
            .lines()
            .skip(1)
            .map(|row| {
                let fields: Vec<_> = row.split(',').collect();
                (fields[0].parse().unwrap(), fields[2] == "1")
            })
            .collect();
    for step in ["100n", "0.5n"] {
        let (mut c, config) = deck(&source)?;
        let p = plan(&config, AnalysisKind::Transient, &["1n", "20u", "0", step])?;
        let (plot, stats) = companion_transient(&mut c, &p, &config.context())?;
        let sample_times = times(&plot);
        let mut witnessed = vec![false; reference.len()];
        for (index, time) in sample_times.into_iter().enumerate() {
            let event = reference
                .partition_point(|(at, _)| *at <= time)
                .saturating_sub(1);
            let g_v = -value(&plot, "i(vbv)", index);
            let g_i = value(&plot, "i(vbi)", index) / 2.;
            for g in [g_v, g_i] {
                assert!(
                    (g - 0.1).abs() <= 1e-12 || (g - 1e-6).abs() <= 1e-12,
                    "non-rail conductance {g:e}"
                );
            }
            // No artificial intermediate conductance. Skip only a tiny
            // localization neighborhood of reference event times, not any
            // substantial portion of the pulse (including equal-width).
            if reference.iter().any(|(at, _)| (at - time).abs() < 5e-12) {
                continue;
            }
            let expected = if reference[event].1 { 0.1 } else { 1e-6 };
            close(
                &format!("voltage at {time:e}, step {step}"),
                g_v,
                expected,
                1e-9,
                1e-13,
            );
            close(
                &format!("current at {time:e}, step {step}"),
                g_i,
                expected,
                1e-9,
                1e-13,
            );
            close(
                "ideal sensor voltage",
                value(&plot, "v(sensor)", index) - 0.75,
                0.,
                0.,
                1e-13,
            );
            close(
                "unloaded voltage control",
                value(&plot, "i(vdrive)", index),
                0.,
                0.,
                1e-13,
            );
            witnessed[event] = true;
        }
        for (index, &(_, high)) in reference.iter().enumerate() {
            if high
                && reference
                    .get(index + 1)
                    .is_some_and(|(next, _)| *next - reference[index].0 > 1e-8)
            {
                assert!(
                    witnessed[index],
                    "missing settled high interval {index}, step {step}"
                );
            }
        }
        assert!(stats.device_event_landings > 0);
        assert!(stats.device_event_retries > 0);
    }
    Ok(())
}

#[test]
fn instantaneous_switch_dc_ac_and_thermal_noise() -> SpiceResult<()> {
    use ngspice_rs::devices::noise::BOLTZMANN;
    for (control, terminal) in [
        ("vcontrol cp rtn -1 ac 1", "%vd(cp rtn)"),
        ("icontrol rtn cp -1 ac 1", "%id(cp rtn)"),
    ] {
        for (ic, resistance) in [(0, 1e6), (1, 10.)] {
            let source = format!(
                "instantaneous bias\n\
                vref rtn 0 .75\n\
                {control}\n\
                vin p rtn 1 ac 1\n\
                rload out rtn 1k noisy=0\n\
                a1 {terminal} %gd(p out) m\n\
                .model m preprocess_timed_switch(ron=10 roff=1Meg vt=-1 vh=.5 instantaneous=true td_on=500n td_off=1u ic={ic})\n\
                .noise v(out,rtn) vin lin 2 1k 2k\n.end\n"
            );
            let (mut c, config) = deck(&source)?;
            let gain = 1e3 / (resistance + 1e3);
            let op =
                plan(&config, AnalysisKind::OperatingPoint, &[])?.run(&mut c, &config.context())?;
            close("dc", value(&op, "v(out)", 0), 0.75 + gain, 1e-10, 1e-13);
            let ac = plan(&config, AnalysisKind::Ac, &["lin", "1", "1k", "1k"])?
                .run(&mut c, &config.context())?;
            close("ac", value(&ac, "v(out)", 0), gain, 1e-10, 1e-13);
            let source = Parser::new()
                .parse_deck(&parse_deck_text(Path::new("instant-noise.cir"), &source))?;
            let noise = config
                .plan_for(&source.analyses[0])?
                .run_plots(&mut c, &config.context())?;
            let expected = (4. * BOLTZMANN * 300.15 * resistance).sqrt() * gain;
            for row in 0..noise[0].point_count() {
                close(
                    "thermal voltage density",
                    value(&noise[0], "onoise_spectrum", row),
                    expected,
                    1e-8,
                    1e-20,
                );
            }
        }
    }
    Ok(())
}

#[test]
fn instantaneous_stiff_switch_rc_converges_to_piecewise_physical_solution() -> SpiceResult<()> {
    // 1 V -> switched R -> C, with different physical off/on RC time
    // constants. Before the turn-on event the cap starts at zero; it
    // cannot jump when conductance changes. No hidden conductance floor.
    let source = "instantaneous RC\n\
        vcontrol ctrl 0 PWL(0 -1 1u -1 1002n 1 4u 1 4002n -1 8u -1)\n\
        v1 p 0 PWL(0 0 1n 1 8u 1)\n\
        a1 %v(ctrl) %gd(p out) m\n\
        rload out 0 1k\n\
        c1 out 0 1n\n\
        .model m preprocess_timed_switch(ron=10 roff=1Meg instantaneous=1 vt=0 vh=0 td_on=100n td_off=100n)\n\
        .options reltol=1e-5 abstol=1p vntol=1n\n\
        .end\n";
    let g0 = 1e-6;
    let g1 = 0.1;
    let load = 1e-3;
    let cap = 1e-9;
    let tau0 = cap / (g0 + load);
    let tau1 = cap / (g1 + load);
    let steady0 = g0 / (g0 + load);
    let steady1 = g1 / (g1 + load);
    let rise = 1.101e-6;
    let fall = 4.101e-6;
    // Exact convolution of the initial 1 ns linear source ramp with RC.
    let initial = steady0 * (1. + tau0 / 1e-9 * (-1e-9_f64 / tau0).exp_m1());
    let evolve = |initial: f64, steady: f64, dt: f64, tau: f64| {
        (initial - steady).mul_add((-dt / tau).exp(), steady)
    };
    let before_rise = evolve(initial, steady0, rise - 1e-9, tau0);
    let before_fall = evolve(before_rise, steady1, fall - rise, tau1);
    let expected = |time| {
        if time < rise {
            evolve(initial, steady0, time - 1e-9, tau0)
        } else if time < fall {
            evolve(before_rise, steady1, time - rise, tau1)
        } else {
            evolve(before_fall, steady0, time - fall, tau0)
        }
    };
    for step in ["100n", "1n"] {
        let (mut c, config) = deck(source)?;
        let p = plan(&config, AnalysisKind::Transient, &["1n", "8u", "0", step])?;
        let (plot, stats) = companion_transient(&mut c, &p, &config.context())?;
        let t = times(&plot);
        for at in [
            0.5e-6,
            rise,
            rise + 1e-9,
            rise + 10e-9,
            rise + 50e-9,
            3e-6,
            fall,
            fall + 10e-9,
            fall + 1e-6,
            8e-6,
        ] {
            close(
                &format!("RC at {at:e}, step {step}"),
                interpolate(&plot, &t, "v(out)", at),
                expected(at),
                0.002,
                2e-5,
            );
        }
        assert!(stats.device_event_landings >= 2);
    }
    Ok(())
}
