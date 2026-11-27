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
