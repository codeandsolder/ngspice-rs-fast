//! Independently measured transfer and analytic residual/Jacobian tests for
//! explicitly selected `LTspice` switches. Ordinary ngspice SW is not changed.

use std::path::Path;

use ngspice_rs::analysis::{AnalysisContext, AnalysisInput, Plot, RunConfig};
use ngspice_rs::devices::{AnalysisMode, Circuit, IterationPhase, LoadRequest, ModelContext};
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, SpiceResult};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn circuit(body: &str) -> SpiceResult<Circuit> {
    let netlist = Parser::new().parse_deck(&parse_deck_text(
        Path::new("prepared-switch.cir"),
        &format!("prepared switch\n{body}\n.end\n"),
    ))?;
    Circuit::from_netlist(&netlist)
}

fn run(body: &str, kind: AnalysisKind, args: &[&str]) -> SpiceResult<Plot> {
    AnalysisInput::with_arguments(kind, args.iter().copied())
        .resolve()?
        .run(&mut circuit(body)?, &AnalysisContext::default())
}

fn close(label: &str, actual: f64, expected: f64, rel: f64, absolute: f64) {
    assert!(
        actual.is_finite()
            && (actual - expected).abs() <= rel.mul_add(actual.abs().max(expected.abs()), absolute),
        "{label}: {actual:.17e} != {expected:.17e}"
    );
}

#[test]
fn all_four_control_modes_match_independent_ltspice_24_measurements() -> TestResult {
    check_reference_sweep(
        include_str!("fixtures/ltspice-switch/transfer.cir"),
        include_str!("fixtures/ltspice-switch/transfer.tsv"),
    )
}

fn check_reference_sweep(original: &str, reference: &str) -> TestResult {
    // Only source family selection differs; no equations are rewritten.
    let normalized = original.replace(" SW(", " ltspice_switch(");
    let netlist =
        Parser::new().parse_deck(&parse_deck_text(Path::new("transfer.cir"), &normalized))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let mut circuit = config.circuit(&netlist)?;
    let plan = config.plan_for(&netlist.analyses[0])?;
    let plot = plan.run(&mut circuit, &config.context())?;
    let rows = reference.lines().filter(|line| !line.starts_with('#'));
    assert_eq!(plot.point_count(), rows.clone().count());
    for (index, line) in rows.enumerate() {
        let values = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()?;
        let [control, v1, v2, v3, v4]: [f64; 5] =
            values.try_into().map_err(|_| "reference row width")?;
        close(
            "sweep control",
            plot.value("v(c)", index).ok_or("missing control")?.re,
            control,
            0.,
            1e-14,
        );
        for (name, expected) in [("i(v1)", v1), ("i(v2)", v2), ("i(v3)", v3), ("i(v4)", v4)] {
            close(
                name,
                plot.value(name, index).ok_or("missing supply current")?.re,
                expected,
                5e-10,
                5e-15,
            );
        }
    }
    Ok(())
}

#[test]
fn ordered_level_setters_use_last_value() -> TestResult {
    check_reference_sweep(
        include_str!("fixtures/ltspice-switch/duplicate-level.cir"),
        include_str!("fixtures/ltspice-switch/duplicate-level.tsv"),
    )
}

/// Independent scalar equation, used to verify the actual circuit-level
/// residual and the two derivatives with central finite differences.
fn reference(level: u8, control: f64, voltage: f64, limit: Option<f64>) -> f64 {
    let t = if level == 1 {
        let u = f64::midpoint(control, 1.).clamp(0., 1.);
        u * u * 2_f64.mul_add(-u, 3.)
    } else {
        control.atan() / std::f64::consts::PI + 0.5
    };
    let g = 10_f64.powf(6_f64.mul_add(t, -6.));
    let current = g * voltage;
    let current = if level == 2 {
        let limit = limit.unwrap_or(10.);
        limit * (current / limit).tanh()
    } else if let Some(limit) = limit {
        current.clamp(-limit, limit)
    } else {
        current
    };
    voltage.mul_add(1e-12, current)
}

fn model(level: u8, limit: Option<f64>) -> String {
    let limit = limit.map_or_else(String::new, |value| format!(" ilimit={value}"));
    format!(".model sm ltspice_switch(ron=1 roff=1meg vt=0 vh=-1 level={level}{limit})")
}

fn row(c: &Circuit, name: &str) -> TestResult<usize> {
    Ok(c.unknowns()
        .node_row(c.nodes().get(name).ok_or("missing node")?)
        .ok_or("missing row")?)
}

#[test]
fn smooth_devices_have_no_history_and_stamp_consistent_kcl_and_jacobians() -> TestResult {
    for level in [1, 2] {
        for limit in [None, Some(0.03)] {
            let c = circuit(&format!(
                "vp p 0 0\nvn n 0 0\nvcp cp 0 0\nvcn cn 0 0\ns1 p n cp cn sm\n{}",
                model(level, limit)
            ))?;
            let device = c.device("s1").ok_or("missing switch")?;
            assert_eq!(device.state_count(), 0);
            assert!(!device.has_discrete_state());
            for control in [-2., -0.8, -0.13, 0.29, 0.76, 2.] {
                for voltage in [-2., -0.7, 0., 0.19, 2.] {
                    check_smooth_point(&c, level, limit, control, voltage)?;
                }
            }
        }
    }
    Ok(())
}

fn check_smooth_point(
    circuit: &Circuit,
    level: u8,
    limit: Option<f64>,
    control: f64,
    voltage: f64,
) -> TestResult {
    let step = 1e-5;
    let [p, n, cp, cn] = [
        row(circuit, "p")?,
        row(circuit, "n")?,
        row(circuit, "cp")?,
        row(circuit, "cn")?,
    ];
    let history = circuit.state_history()?;
    let width = circuit.unknown_count();
    let mut bias = Vector::zeros(width);
    bias.as_mut_slice()[p] = voltage + 0.3;
    bias.as_mut_slice()[n] = 0.3;
    bias.as_mut_slice()[cp] = control - 0.41;
    bias.as_mut_slice()[cn] = -0.41;
    let mut matrix = SparseMatrix::new(width, width);
    let mut rhs = Vector::zeros(width);
    let mut trial = history.trial_in(IterationPhase::Predict, None)?;
    circuit.load(
        &LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &bias,
            model_context: &ModelContext::default(),
            integration: None,
            history: &history,
            forcing: None,
        },
        &mut matrix,
        &mut rhs,
        &mut trial,
    )?;
    matrix.fold_duplicates();
    let actual = |row_index: usize| -> TestResult<f64> {
        let residual = (0..width)
            .map(|col| matrix.get(row_index, col) * bias.as_slice()[col])
            .sum::<f64>()
            - rhs.get(row_index).ok_or("missing RHS row")?;
        Ok(residual)
    };
    let expected = reference(level, control, voltage, limit);
    close(
        "positive terminal residual",
        actual(p)?,
        expected,
        5e-11,
        1e-14,
    );
    close(
        "negative terminal residual",
        actual(n)?,
        -expected,
        5e-11,
        1e-14,
    );
    let dv = (reference(level, control, voltage + step, limit)
        - reference(level, control, voltage - step, limit))
        / (2. * step);
    let dc = (reference(level, control + step, voltage, limit)
        - reference(level, control - step, voltage, limit))
        / (2. * step);
    for (row_index, sign) in [(p, 1.), (n, -1.)] {
        for (col, derivative) in [(p, dv), (n, -dv), (cp, dc), (cn, -dc)] {
            close(
                "nodal derivative",
                matrix.get(row_index, col),
                sign * derivative,
                2e-7,
                1e-11,
            );
        }
    }
    check_smooth_small_signal(
        circuit,
        &bias,
        &matrix,
        [p, n, cp, cn],
        SmoothBias {
            level,
            limit,
            control,
            voltage,
        },
    )
}

#[derive(Clone, Copy)]
struct SmoothBias {
    level: u8,
    limit: Option<f64>,
    control: f64,
    voltage: f64,
}

fn check_smooth_small_signal(
    circuit: &Circuit,
    bias: &Vector,
    matrix: &SparseMatrix,
    [p, n, cp, cn]: [usize; 4],
    point: SmoothBias,
) -> TestResult {
    let SmoothBias {
        level,
        limit,
        control,
        voltage,
    } = point;
    let expected = reference(level, control, voltage, limit);
    let linear = circuit.small_signal_system(&ModelContext::default(), bias)?;
    let pencil = circuit.pole_zero_system_at(&ModelContext::default(), bias, None)?;
    for row_index in [p, n] {
        for col in [p, n, cp, cn] {
            let physical = matrix.get(row_index, col);
            let expected_ac = if level == 1
                && limit.is_some_and(|limit| {
                    voltage
                        .mul_add(-1e-12, reference(level, control, voltage, None))
                        .abs()
                        > limit
                })
                && [p, n].contains(&col)
            {
                let chord = expected / voltage;
                if row_index == col { chord } else { -chord }
            } else if level == 1 && [cp, cn].contains(&col) {
                physical * 0.99
            } else {
                physical
            };
            close(
                "AC derivative",
                linear.a.get(row_index, col),
                expected_ac,
                1e-13,
                1e-18,
            );
            close(
                "PZ derivative",
                pencil.a.get(row_index, col),
                expected_ac,
                1e-13,
                1e-18,
            );
            assert_eq!(pencil.e.get(row_index, col).to_bits(), 0_f64.to_bits());
        }
    }
    Ok(())
}

#[test]
fn ac_control_gain_matches_reference_scaling_of_the_dc_derivative() -> TestResult {
    for level in [1, 2] {
        let body = format!(
            "vin in 0 1\nvc c 0 dc .25 ac 1\nres in out 1k\ns1 out 0 c 0 sm\n{}",
            model(level, None)
        );
        let ac = run(&body, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
        let h = 1e-5;
        let get = |control| -> TestResult<f64> {
            let input = body.replace("dc .25 ac 1", &format!("dc {control} ac 1"));
            Ok(run(&input, AnalysisKind::OperatingPoint, &[])?
                .value("v(out)", 0)
                .ok_or("missing output")?
                .re)
        };
        let derivative = (get(0.25 + h)? - get(0.25 - h)?) / (2. * h);
        let value = ac.value("v(out)", 0).ok_or("missing output")?;
        close(
            "control gain",
            value.re,
            derivative * if level == 1 { 0.99 } else { 1. },
            1e-6,
            1e-12,
        );
        close("memoryless gain phase", value.im, 0., 0., 1e-15);
    }
    Ok(())
}

#[test]
fn transient_ramp_preserves_continuous_current_and_reverses_terminal_polarity() -> TestResult {
    for level in [1, 2] {
        let body = format!(
            "vdrive p 0 pwl(0 -1 1m 1 2m -1)\nvctrl c 0 pwl(0 -2 1m 2 2m -2)\ns1 p 0 c 0 sm\n{}",
            model(level, Some(0.03))
        );
        let netlist = Parser::new().parse_deck(&parse_deck_text(Path::new("ramp.cir"),
            &format!("ramp\n{body}\n.options reltol=1e-9 abstol=1e-15 vntol=1e-12\n.tran 1u 2m 0 10u\n.end\n")))?;
        let config = RunConfig::from_netlist(&netlist)?;
        let plot = config
            .plan_for(&netlist.analyses[0])?
            .run(&mut config.circuit(&netlist)?, &config.context())?;
        assert!(plot.point_count() > 100);
        for index in 0..plot.point_count() {
            let v = plot.value("v(p)", index).ok_or("missing voltage")?.re;
            let control = plot.value("v(c)", index).ok_or("missing control")?.re;
            let i = plot.value("i(vdrive)", index).ok_or("missing current")?.re;
            close(
                "transient current",
                -i,
                reference(level, control, v, Some(0.03)),
                2e-7,
                1e-12,
            );
        }
    }
    Ok(())
}

#[test]
fn native_and_prepared_switch_selection_never_aliases_unqualified_modes() -> TestResult {
    for bad in [
        "sw(level=2 vh=-1)",
        "ltspice_switch(level=1.5 vh=-1)",
        "ltspice_switch(level=3 vh=-1)",
        "ltspice_switch(level=2 vh=0)",
        "ltspice_switch(ron=-1 vh=-1)",
        "ltspice_switch(level=2 lser=1u vh=-1)",
        "ltspice_switch(epsilon=.1 vh=-1)",
        "ltspice_switch(level=1 vh=-1 oneway)",
    ] {
        assert!(
            circuit(&format!("s1 a 0 c 0 sm\n.model sm {bad}")).is_err(),
            "{bad}"
        );
    }
    let native = circuit("s1 a 0 c 0 sm\n.model sm sw(vh=-1)")?;
    let smooth = circuit("s1 a 0 c 0 sm\n.model sm ltspice_switch(vh=-1)")?;
    assert_eq!(native.device("s1").ok_or("native switch")?.state_count(), 2);
    assert_eq!(
        smooth.device("s1").ok_or("prepared switch")?.state_count(),
        0
    );
    Ok(())
}

#[test]
fn discrete_ac_uses_converged_bias_instead_of_native_accepted_state() -> TestResult {
    for (control, flag, hysteresis, expected) in [
        ("2", "", "0", 0.1),
        ("-2", "on", "0", 0.001),
        ("2", "off", "1", 0.1),
        ("0", "on", "1", 0.1),
        ("0", "off", "1", 0.001),
    ] {
        let deck = format!(
            "vdrive p 0 dc 1 ac 1\nvc c 0 {control}\ns1 p 0 c 0 sm {flag}\n.model sm ltspice_switch(ron=10 roff=1k vh={hysteresis})"
        );
        let result = run(&deck, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
        let current = result.value("i(vdrive)", 0).ok_or("missing AC current")?;
        close("discrete AC", -current.re, expected + 1e-12, 1e-10, 1e-14);
        close("discrete phase", current.im, 0., 0., 1e-15);
    }
    Ok(())
}

#[test]
fn series_inductance_uses_transient_flux_but_is_shorted_by_reference_ac() -> TestResult {
    for (inductance, time_constant, reference) in [
        (
            "1m",
            100e-6,
            include_str!("fixtures/ltspice-switch/series-transient.tsv"),
        ),
        (
            "-1m",
            -100e-6,
            include_str!("fixtures/ltspice-switch/series-negative.tsv"),
        ),
    ] {
        check_series_inductance(inductance, time_constant, reference)?;
    }
    Ok(())
}

fn series_response(time: f64, time_constant: f64) -> f64 {
    -0.1 * (-(time - 10.0005e-6) / time_constant).exp_m1()
}

fn check_series_inductance(inductance: &str, time_constant: f64, reference: &str) -> TestResult {
    let model = format!(".model sm ltspice_switch(ron=10 roff=1k vt=0 vh=0 lser={inductance})");
    let ac = run(
        &format!("vdrive p 0 dc 1 ac 1\nvc c 0 2\ns1 p 0 c 0 sm\n{model}"),
        AnalysisKind::Ac,
        &["dec", "1", "100", "100000"],
    )?;
    for point in 0..ac.point_count() {
        let value = ac.value("i(vdrive)", point).ok_or("missing AC current")?;
        close("AC current", -value.re, 0.1 + 1e-12, 1e-10, 1e-14);
        close("AC phase", value.im, 0., 0., 1e-15);
    }

    let body = format!("vdrive p 0 pulse(0 1 10u 1n 1n 1m 2m)\nvc c 0 2\ns1 p 0 c 0 sm\n{model}");
    let c = circuit(&body)?;
    let device = c.device("s1").ok_or("missing switch")?;
    assert_eq!(device.external_terminals().len(), 4);
    assert_eq!(device.terminals().len(), 5);
    assert_eq!(device.state_count(), 4);
    assert_eq!(device.branch_currents(), 1);
    assert_eq!(device.truncation_slots(), vec![0]);
    let netlist = Parser::new().parse_deck(&parse_deck_text(
        Path::new("series.cir"),
        &format!("series\n{body}\n.options reltol=1e-9 abstol=1e-14 vntol=1e-12\n.tran 100n 500u 0 100n\n.end\n"),
    ))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let plot = config
        .plan_for(&netlist.analyses[0])?
        .run(&mut config.circuit(&netlist)?, &config.context())?;
    let mut compared = 0;
    for point in 0..plot.point_count() {
        let time = plot.value("time", point).ok_or("missing time")?.re;
        if time < 11e-6 {
            continue;
        }
        // The 1 ns linear ramp is equivalent to a half-rise-time delayed
        // step up to a relative correction below 1e-10 for this RL circuit.
        let expected = series_response(time, time_constant);
        let current = -plot.value("i(vdrive)", point).ok_or("missing current")?.re;
        close("RL response", current, expected, 1e-5, 1e-9);
        compared += 1;
    }
    assert!(compared > 1000);
    // Independent reference samples are on its own adaptive time grid
    // (maximum step 1 us), so compare to the same analytic response.
    for line in reference.lines().filter(|line| !line.starts_with('#')) {
        let values = line
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>()?;
        let [time, current]: [f64; 2] = values.try_into().map_err(|_| "reference row width")?;
        let expected = if time == 0. {
            0.
        } else {
            series_response(time, time_constant)
        };
        close("measured RL response", current, expected, 2e-4, 1e-9);
    }
    Ok(())
}

/// Source `LTspice` 24.0.12 AC measurements. Independent terminal-voltage
/// excitation and control-voltage excitation distinguish chord loads,
/// differential loads, and the high-a epsilon approximation.
#[test]
fn independent_ac_loads_preserve_reference_chords_and_implicit_epsilon_derivative() -> TestResult {
    let cases = [
        ("", 0.486_580_450_469_153_9, 0.316_227_767_016_838_1),
        ("ilimit=.05", 0., 0.166_666_667_666_666_7),
        (
            "level=2 ilimit=.05",
            0.017_948_138_927_685_89,
            0.027_208_908_053_780_99,
        ),
        (
            "level=2 ilimit=.05 epsilon=.2",
            0.036_659_597_975_629_78,
            0.102_905_612_199_792_4,
        ),
        ("vser=.1", 0.324_386_966_979_435_9, 0.316_227_767_016_838_1),
        ("vser=.1 ilimit=.05", 0., 0.250_000_001),
    ];
    for (parameters, gain_control, gain_terminal) in cases {
        for (ac_control, ac_terminal, expected) in [(1, 0, gain_control), (0, 1, -gain_terminal)] {
            let text = format!(
                "AC\n.options gmin=1e-9 reltol=1e-10 abstol=1e-15\nvc c 0 dc 0 ac {ac_control}\nvdrive p 0 dc -.3 ac {ac_terminal}\ns1 p 0 c 0 sm\n.model sm ltspice_switch(ron=.1 roff=100 vh=-1 {parameters})\n.ac lin 1 1k 1k\n.end\n"
            );
            let netlist =
                Parser::new().parse_deck(&parse_deck_text(Path::new("ac-oracle.cir"), &text))?;
            let config = RunConfig::from_netlist(&netlist)?;
            let plot = config
                .plan_for(&netlist.analyses[0])?
                .run(&mut config.circuit(&netlist)?, &config.context())?;
            let i = plot.value("i(vdrive)", 0).ok_or("missing current")?;
            close(parameters, i.re, expected, 1e-9, 1e-13);
            close("AC phase", i.im, 0., 0., 1e-15);
        }
    }
    let high = "vc c 0 dc 2 ac 1\nvdrive p 0 dc -.1\ns1 p 0 c 0 sm\n.model sm ltspice_switch(ron=.1 roff=100 vh=-1 level=2 ilimit=.05 epsilon=.2)";
    let ac = run(high, AnalysisKind::Ac, &["lin", "1", "1000", "1000"])?;
    close(
        "implicit epsilon AC",
        ac.value("i(vdrive)", 0).ok_or("missing current")?.re,
        3.013_854_747_482_482e-5,
        1e-9,
        1e-14,
    );
    Ok(())
}

#[test]
fn level_two_off_resistance_default_is_independent_of_gmin() -> TestResult {
    for (gmin, expected) in [
        ("1e-12", 1.079_159_718_042_647e-6),
        ("1m", 1.001_079_158_718_043e-3),
    ] {
        let text = format!(
            "defaults\n.options gmin={gmin} reltol=1e-10 abstol=1e-15\nvc c 0 -2\nvdrive p 0 1\ns1 p 0 c 0 sm\n.model sm ltspice_switch(ron=1 vh=-1 level=2)\n.op\n.end\n"
        );
        let netlist =
            Parser::new().parse_deck(&parse_deck_text(Path::new("default-oracle.cir"), &text))?;
        let config = RunConfig::from_netlist(&netlist)?;
        let plot = config
            .plan_for(&netlist.analyses[0])?
            .run(&mut config.circuit(&netlist)?, &config.context())?;
        close(
            "level 2 default Roff",
            -plot.value("i(vdrive)", 0).ok_or("missing current")?.re,
            expected,
            1e-10,
            1e-15,
        );
    }
    Ok(())
}

#[test]
fn thermal_noise_matches_independent_measurements_including_gmin_and_hard_limit() -> TestResult {
    let cases = [
        (
            "1e-12",
            "0",
            "0",
            "ron=.1 roff=100 vh=-1 ilimit=.01",
            1.122_213_506_215_76e-10,
        ),
        (
            "1e-12",
            "-1",
            "0",
            "ron=.1 roff=100 vh=-1 ilimit=.05",
            1.254_881_743_208_85e-10,
        ),
        (
            "1e-12",
            "-1",
            "0",
            "ron=.1 roff=100 vh=-1 ilimit=.05 level=2",
            1.287_476_633_019_7e-10,
        ),
        (
            "1e-12",
            "-1",
            "0",
            "ron=.1 roff=100 vh=-1 ilimit=.05 level=2 vser=.5 epsilon=.2",
            1.276_199_331_952_715e-10,
        ),
        ("1m", "0", "-2", "ron=.1 roff=1k", 1.286_195_999_791_727e-10),
        (
            "1m",
            "1",
            "-2",
            "ron=.1 roff=1k level=2 vh=-1 oneway",
            1.286_838_294_962_742e-10,
        ),
    ];
    for (gmin, voltage, control, parameters, expected) in cases {
        let text = format!(
            "noise\n.options gmin={gmin} reltol=1e-10 abstol=1e-15\nVin in 0 dc {voltage} ac 1\nR1 in a 1\nVc c 0 {control}\nS1 a 0 c 0 m1\n.model m1 ltspice_switch({parameters})\n.noise V(a) Vin dec 2 1k 10k\n.end\n"
        );
        let netlist =
            Parser::new().parse_deck(&parse_deck_text(Path::new("noise-oracle.cir"), &text))?;
        let config = RunConfig::from_netlist(&netlist)?;
        let plots = config
            .plan_for(&netlist.analyses[0])?
            .run_plots(&mut config.circuit(&netlist)?, &config.context())?;
        for value in plots[0].column("onoise_spectrum").ok_or("missing noise")? {
            // Vendor uses slightly different physical constants and default
            // operating-point tolerances; compare the measured density.
            close(parameters, value.re, expected, 2e-6, 1e-18);
        }
    }
    Ok(())
}

#[test]
fn descending_hysteresis_and_zero_width_threshold_match_reference() -> TestResult {
    check_reference_sweep(
        include_str!("fixtures/ltspice-switch/hysteresis-down.cir"),
        include_str!("fixtures/ltspice-switch/hysteresis-down.tsv"),
    )?;
    let normalized = include_str!("fixtures/ltspice-switch/threshold-op.cir")
        .replace(" SW(", " ltspice_switch(");
    let netlist =
        Parser::new().parse_deck(&parse_deck_text(Path::new("threshold.cir"), &normalized))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let plot = config
        .plan_for(&netlist.analyses[0])?
        .run(&mut config.circuit(&netlist)?, &config.context())?;
    for (name, expected) in [
        ("i(v1)", -1.000_001e-6),
        ("i(v2)", -1.000_000_000_001),
        ("i(v3)", -1.000_001e-6),
        ("i(v4)", -1.000_001e-6),
        ("i(v5)", -1.000_000_000_001),
        ("i(v6)", -1.000_001e-6),
    ] {
        close(
            name,
            plot.value(name, 0).ok_or("missing supply current")?.re,
            expected,
            5e-10,
            5e-15,
        );
    }
    Ok(())
}

/// Independent direct hyperbolic implementation of the measured ten-step
/// join. The production law uses stable exponential forms and analytic
/// sensitivities; this oracle deliberately does neither.
fn physical_switch_reference(
    level: u8,
    control: f64,
    voltage: f64,
    limit: Option<f64>,
    offset: f64,
    epsilon: Option<f64>,
) -> f64 {
    let fraction = if level == 1 {
        let t = f64::midpoint(control, 1.).clamp(0., 1.);
        t * t * (3. - 2. * t)
    } else {
        control.atan() / std::f64::consts::PI + 0.5
    };
    let mut g = 0.01 * 1000_f64.powf(fraction);
    let shifted = voltage + offset;
    let (active, leakage) = if level == 1 {
        if offset != 0. && shifted != 0. && shifted.is_sign_positive() == offset.is_sign_positive()
        {
            g = 0.01;
        }
        let current = g * shifted;
        (limit.map_or(current, |l| current.clamp(-l, l)), shifted)
    } else {
        let l = limit.unwrap_or(10.);
        let current = if let Some(epsilon) = epsilon {
            let forward = -shifted;
            if forward <= 0. {
                0.
            } else if epsilon == 0. {
                -l * (g * forward / l).tanh()
            } else {
                let a = g * epsilon / l;
                let mut join = a / 2.;
                for _ in 0..10 {
                    join -= ((2. * join).sinh() - a) / (2. * (2. * join).cosh());
                }
                if voltage > -offset - epsilon {
                    -l * join.tanh() * (forward / epsilon).powi(2)
                } else {
                    -l * (g * (-offset - epsilon - voltage) / l + join).tanh()
                }
            }
        } else {
            l * (g * voltage / l).tanh()
        };
        (current, voltage)
    };
    leakage.mul_add(1e-12, active)
}

#[test]
fn continuous_ltspice_physical_taylor_matches_independent_current_and_jacobian() -> TestResult {
    use ngspice_rs::devices::{Limit, TransientTiming};
    for (level, limit, offset, epsilon) in [
        (1, None, 0., None),
        (1, Some(0.05), 0., None),
        (1, Some(0.05), 0.1, None),
        (1, None, -0.1, None),
        (2, None, 0., None),
        (2, Some(0.05), 0., None),
        (2, Some(0.05), 0., Some(0.)),
        (2, Some(0.05), 0.1, Some(0.2)),
    ] {
        let limit_text = limit.map_or_else(String::new, |v| format!("ilimit={v}"));
        let epsilon_text = epsilon.map_or_else(String::new, |e| format!("oneway epsilon={e}"));
        let mut c = circuit(&format!(
            "s1 p n cp cn sm\n.model sm ltspice_switch(ron=.1 roff=100 vt=0 vh=-1 level={level} {limit_text} vser={offset} {epsilon_text})"
        ))?;
        c.finalize()?;
        let [p, n, cp, cn] = ["p", "n", "cp", "cn"].map(|name| row(&c, name).unwrap());
        let prepared = c.prepare_run();
        let dae = prepared
            .prepare_physical_dae(&ModelContext::default(), TransientTiming::new(1e-6, 1e-3)?)?;
        for control in [-1.5, -1., -0.37, 0.2, 1., 1.5, 5.] {
            for voltage in [-0.8, -offset - 0.2, -offset - 0.07, -offset, 0.35] {
                let mut state = vec![Vector::zeros(c.unknown_count()); 9];
                state[0].set(p, voltage)?;
                state[0].set(n, 0.)?;
                state[0].set(cp, control - 0.125)?;
                state[0].set(cn, -0.125)?;
                state[1].set(p, 0.0015)?;
                state[1].set(n, -0.0005)?;
                state[1].set(cp, 0.002)?;
                state[1].set(cn, -0.001)?;
                for (side, tau) in [(Limit::Left, -0.1_f64), (Limit::Right, 0.1_f64)] {
                    let result = dae.evaluate_taylor(&state, 0., 1e-6, side)?;
                    let series = |values: &[Vector], row| {
                        values.iter().rev().fold(0_f64, |sum, value| {
                            sum.mul_add(tau, value.get(row).unwrap())
                        })
                    };
                    let gradient = |r, c| {
                        result
                            .conductance
                            .iter()
                            .rev()
                            .fold(0_f64, |sum, matrix| sum.mul_add(tau, matrix.get(r, c)))
                    };
                    let v = 0.002_f64.mul_add(tau, voltage);
                    let x = 0.003_f64.mul_add(tau, control);
                    let evaluate =
                        |v, x| physical_switch_reference(level, x, v, limit, offset, epsilon);
                    let expected = evaluate(v, x);
                    let h = 1e-6;
                    let stencil = |f: &dyn Fn(f64) -> f64| {
                        (8. * (f(h) - f(-h)) - (f(2. * h) - f(-2. * h))) / (12. * h)
                    };
                    let dv = stencil(&|step| evaluate(v + step, x));
                    let dc = stencil(&|step| evaluate(v, x + step));
                    let label = format!(
                        "level={level} offset={offset} epsilon={epsilon:?} limit={limit:?} v={v} control={x}"
                    );
                    close(&label, series(&result.residual, p), expected, 2e-9, 2e-12);
                    close(&label, series(&result.residual, n), -expected, 2e-9, 2e-12);
                    for (r, c, want) in [
                        (p, p, dv),
                        (p, n, -dv),
                        (p, cp, dc),
                        (p, cn, -dc),
                        (n, p, -dv),
                        (n, cp, -dc),
                    ] {
                        close(&label, gradient(r, c), want, 5e-6, 1e-9);
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn continuous_series_flux_remains_a_physical_branch_coordinate() -> TestResult {
    use ngspice_rs::devices::{Limit, TransientTiming};
    let mut c = circuit("s1 p n cp cn sm\n.model sm ltspice_switch(ron=10 roff=1k vh=-1 lser=2u)")?;
    c.finalize()?;
    let index = c
        .devices()
        .iter()
        .position(|d| d.name() == "s1")
        .ok_or("missing switch")?;
    let device = &c.devices()[index];
    let branch = c.branch_rows(index).ok_or("missing Lser branch")?.start;
    let internal = c
        .unknowns()
        .node_row(device.terminals()[4])
        .ok_or("missing series node")?;
    let p = row(&c, "p")?;
    let mut state = vec![Vector::zeros(c.unknown_count()); 4];
    state[0].set(branch, 0.07)?;
    state[1].set(branch, -0.002)?;
    state[2].set(branch, 0.001)?;
    state[0].set(p, 2.)?;
    state[0].set(internal, 1.)?;
    let prepared = c.prepare_run();
    let dae = prepared
        .prepare_physical_dae(&ModelContext::default(), TransientTiming::new(1e-6, 1e-3)?)?;
    let result = dae.evaluate_taylor(&state, 0., 1e-6, Limit::Right)?;
    for (degree, current) in [0.07, -0.002, 0.001, 0.].into_iter().enumerate() {
        close(
            "series flux",
            result.charge[degree].get(branch).unwrap(),
            -2e-6 * current,
            1e-13,
            1e-20,
        );
        close(
            "terminal current",
            result.residual[degree].get(p).unwrap(),
            current,
            1e-13,
            1e-20,
        );
    }
    close(
        "voltage equation",
        result.residual[0].get(branch).unwrap(),
        1.,
        0.,
        1e-15,
    );
    close(
        "flux derivative",
        result.capacitance[0].get(branch, branch),
        -2e-6,
        1e-13,
        1e-20,
    );
    Ok(())
}

#[test]
fn derivative_arrays_use_physical_switch_tangent_not_ac_chord() -> TestResult {
    let text = "\
vc cp 0 PWL(0 -2 2m 2)
vb p 0 0.5
s1 p 0 cp 0 sw
.model sw ltspice_switch(ron=.1 roff=100 vt=0 vh=-1 level=1 ilimit=.05)
b1 out 0 v=1m*ddt(i(vb))
r1 out 0 1k
.options reltol=1e-8 vntol=1n";
    let p = run(text, AnalysisKind::Transient, &["5u", "2m", "0", "20u"])?;
    let mut limited = 0;
    let mut active = 0;
    for k in 0..p.point_count() {
        let t = p.value("time", k).ok_or("time")?.re;
        let x = 2000_f64.mul_add(t, -2.);
        let z = f64::midpoint(x, 1.).clamp(0., 1.);
        let g = 0.01 * 1000_f64.powf(z * z * 2_f64.mul_add(-z, 3.));
        let i = (0.5 * g).min(0.05);
        close(
            "source current",
            p.value("i(vb)", k).ok_or("source current")?.re,
            -(i + 0.5e-12),
            1e-7,
            1e-9,
        );
        if t == 0. || (t - 2e-3).abs() < 1e-12 || (0.5 * g - 0.05).abs() < 1e-7 {
            continue;
        }
        let dg = if x > -1. && x < 1. && 0.5 * g < 0.05 {
            active += 1;
            -0.5 * g * 1000_f64.ln() * 6. * z * (1. - z)
        } else {
            limited += 1;
            0.
        };
        close(
            "physical derivative",
            p.value("v(out)", k).ok_or("derivative")?.re,
            dg,
            2e-6,
            1e-8,
        );
    }
    assert!(active > 10 && limited > 10);
    Ok(())
}
