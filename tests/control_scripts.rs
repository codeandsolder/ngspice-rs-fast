//! Reusable script session and resource bounds.
use std::path::Path;

use ngspice_rs::analysis::AnalysisInput;
use ngspice_rs::analysis::control::ControlSession;
use ngspice_rs::analysis::optimization::parameters::{CandidateTopology, CircuitParameterPlan};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::eval::EvalLimits;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::AnalysisKind;

fn divider() -> CircuitParameterPlan {
    let source = Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("control.cir"),
            "divider\n.param r=1k guard={1/(r-500)}\nv1 in 0 3\nr1 in out {r}\nr2 out 0 1k\n.end\n",
        ))
        .unwrap();
    CircuitParameterPlan::new(source, &["r"]).unwrap()
}

#[test]
fn repeated_parameter_runs_reuse_topology_and_reject_failed_edits() {
    let plan = divider();
    let mut session = ControlSession::new(&plan, EvalLimits::default()).unwrap();
    for r in [2000., 1000., 2000.] {
        assert_eq!(
            session.alter_parameter("R", r).unwrap(),
            CandidateTopology::Reused
        );
        let plots = session
            .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 100)
            .unwrap();
        let voltage = plots[0].value("v(out)", 0).unwrap().re;
        let analytic = 3000. / (r + 1000.);
        assert!((voltage - analytic).abs() < 1e-12);
    }
    assert!(session.alter_parameter("r", 500.).is_err());
    assert_eq!(session.parameter("r"), Some(2000.));
    let plots = session
        .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 100)
        .unwrap();
    assert!((plots[0].value("v(out)", 0).unwrap().re - 1.).abs() < 1e-12);
}

#[test]
fn plot_budget_failure_does_not_poison_next_run() {
    let plan = divider();
    let mut session = ControlSession::new(&plan, EvalLimits::default()).unwrap();
    assert!(
        session
            .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 0, 100)
            .is_err()
    );
    assert!(
        session
            .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 0)
            .is_err()
    );
    assert!(
        session
            .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 100)
            .is_ok()
    );
}

use ngspice_rs::cli::control::{ScriptLimits, ScriptReport, run_file};

struct ScriptDeck(std::path::PathBuf);

impl ScriptDeck {
    fn new(name: &str, script: &str) -> Self {
        Self::from_text(
            name,
            &format!(
                "divider\n.param r=1k\nv1 in 0 3\nr1 in out {{r}}\nr2 out 0 1k\n.dc v1 0 3 1\n.control\n{script}\n.endc\n.end\n"
            ),
        )
    }

    fn from_text(name: &str, text: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("spice-control-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("deck.cir"), text.trim_start()).unwrap();
        Self(path)
    }

    fn run(&self, limits: ScriptLimits) -> ngspice_rs::SpiceResult<(ScriptReport, String)> {
        let mut bytes = Vec::new();
        let report = run_file(&self.0.join("deck.cir"), true, limits, &mut bytes)?;
        Ok((report, String::from_utf8(bytes).unwrap()))
    }
}

impl Drop for ScriptDeck {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn compiled_control_runs_repeated_sweeps_and_named_measurements() {
    let deck = ScriptDeck::new(
        "sweep",
        "
let total = 0
foreach resistance 1k 2k 1k
  alterparam r = $&resistance
  reset
  run
  meas dc output find v(out) at=2
  meas dc doubled param='output*2'
  let total = total + doubled
  destroy all
end
print total
quit",
    );
    let (report, output) = deck.run(ScriptLimits::default()).unwrap();
    let expected = 2. + 4. / 3. + 2.;
    assert!((report.variables["total"] - expected).abs() < 1e-12);
    assert!(report.plots.is_empty());
    assert!(output.contains("total ="));
}

#[test]
fn nested_while_foreach_and_direct_analysis_use_current_bindings() {
    let deck = ScriptDeck::new(
        "loops",
        "
let i = 0
let total = 0
while i < 3
  foreach gain 1 2
    let total = total + gain
  end
  let i = i + 1
end
dc v1 0 $&i 1
meas dc maximum max v(out)
print maximum",
    );
    let (report, _) = deck.run(ScriptLimits::default()).unwrap();
    assert_eq!(report.variables["i"], 3.);
    assert_eq!(report.variables["total"], 9.);
    assert!((report.variables["maximum"] - 1.5).abs() < 1e-12);
    assert_eq!(report.plots.len(), 1);
    assert_eq!(report.plots[0].plot.point_count(), 4);
}

#[test]
fn alterparam_is_staged_until_reset() {
    let deck = ScriptDeck::new(
        "reset",
        "
alterparam r = 2k
op
reset
op",
    );
    let (report, _) = deck.run(ScriptLimits::default()).unwrap();
    assert_eq!(report.plots.len(), 2);
    for (result, expected) in report.plots.iter().zip([1.5, 1.]) {
        assert!((result.plot.value("v(out)", 0).unwrap().re - expected).abs() < 1e-12);
    }
}

#[test]
fn command_compilation_rejects_malformed_or_unimplemented_code_before_execution() {
    for (name, script) in [
        ("bad-end", "op\nend"),
        ("missing-end", "foreach n 1 2\nop"),
        ("unknown", "op\nprint v(out)\nshell invalid"),
    ] {
        let deck = ScriptDeck::new(name, script);
        let mut output = Vec::new();
        let result = run_file(
            &deck.0.join("deck.cir"),
            true,
            ScriptLimits::default(),
            &mut output,
        );
        assert!(result.is_err(), "{name}");
        assert!(
            output.is_empty(),
            "{name}: partial execution before compilation completed"
        );
    }
}

#[test]
fn control_work_retention_and_output_budgets_are_independent() {
    for (name, script, limits, expected) in [
        (
            "steps",
            "while 1\nend",
            ScriptLimits {
                steps: 8,
                ..ScriptLimits::default()
            },
            "instruction budget",
        ),
        (
            "plots",
            "op\nop",
            ScriptLimits {
                plots: 1,
                ..ScriptLimits::default()
            },
            "plot count budget",
        ),
        (
            "cells",
            "op",
            ScriptLimits {
                cells: 0,
                ..ScriptLimits::default()
            },
            "cell limit",
        ),
        (
            "grid",
            "run\nrun",
            ScriptLimits {
                grid_points: 6,
                ..ScriptLimits::default()
            },
            "grid budget",
        ),
        (
            "variables",
            "let a=1\nlet b=2",
            ScriptLimits {
                variables: 1,
                ..ScriptLimits::default()
            },
            "variable budget",
        ),
        (
            "output",
            "let a=1\nprint a",
            ScriptLimits {
                output_bytes: 1,
                ..ScriptLimits::default()
            },
            "output byte budget",
        ),
    ] {
        let deck = ScriptDeck::new(name, script);
        let error = deck.run(limits).err().expect("budget must reject script");
        assert!(error.to_string().contains(expected), "{name}: {error}");
    }
}

use ngspice_rs::Complex;
use ngspice_rs::analysis::control::vectors::{TimeGrid, TransformBudget, fft, linearize};
use ngspice_rs::analysis::spectrum::{SpectrumLength, SpectrumSettings, SpectrumWindow};
use ngspice_rs::analysis::{Plot, PlotFlags, Variable};

fn waveform(times: impl IntoIterator<Item = f64>, signal: impl Fn(f64) -> f64) -> Plot {
    let mut plot = Plot::new(
        "tran",
        "Transient Analysis",
        PlotFlags::Real,
        vec![
            Variable::new("time", "time"),
            Variable::new("v(out)", "voltage"),
        ],
    );
    for time in times {
        plot.push_point(&[Complex::real(time), Complex::real(signal(time))])
            .unwrap();
    }
    plot
}

fn transform_budget() -> TransformBudget {
    TransformBudget {
        cells: 10_000,
        work: 1_000_000,
    }
}

#[test]
fn linearize_resamples_nonuniform_accepted_points() {
    let plot = waveform([0., 0.1, 0.3, 0.6, 1.], |t| 2. * t + 3.);
    let uniform = linearize(
        &plot,
        TimeGrid {
            start: 0.,
            stop: 1.,
            step: 0.125,
        },
        None,
        &[],
        &mut transform_budget(),
    )
    .unwrap();
    assert_eq!(uniform.point_count(), 9);
    for row in uniform.rows() {
        assert!((row[1].re - (2. * row[0].re + 3.)).abs() < 1e-14);
    }
    assert!(
        fft(
            &plot,
            &[],
            SpectrumSettings {
                window: SpectrumWindow::Rectangular,
                ..SpectrumSettings::default()
            },
            &mut transform_budget()
        )
        .is_err()
    );
    let mut tiny = TransformBudget {
        cells: 10,
        work: 1_000_000,
    };
    assert!(
        linearize(
            &plot,
            TimeGrid {
                start: 0.,
                stop: 1.,
                step: 0.125
            },
            None,
            &[],
            &mut tiny
        )
        .is_err()
    );
}

#[test]
fn spectrum_matches_analytic_sinusoid_dc_and_nyquist() {
    let plot = waveform((0..64).map(|i| f64::from(i) / 64.), |t| {
        1.25 + 2. * (std::f64::consts::TAU * 5. * t).sin()
            + 0.25 * (std::f64::consts::TAU * 32. * t).cos()
    });
    let spectrum = fft(
        &plot,
        &[],
        SpectrumSettings {
            window: SpectrumWindow::Rectangular,
            ..SpectrumSettings::default()
        },
        &mut transform_budget(),
    )
    .unwrap();
    assert_eq!(spectrum.point_count(), 33);
    assert!((spectrum.value("v(out)", 0).unwrap().re - 1.25).abs() < 1e-13);
    let tone = spectrum.value("v(out)", 5).unwrap();
    assert!(tone.re.abs() < 1e-13);
    assert!((tone.im + 2.).abs() < 1e-13);
    assert!((spectrum.value("v(out)", 32).unwrap().re - 0.25).abs() < 1e-13);
    assert!((spectrum.value("frequency", 5).unwrap().re - 5.).abs() < 1e-13);
}

#[test]
fn zero_padding_preserves_actual_sample_count_normalization() {
    let plot = waveform((0..30).map(f64::from), |_| 2.5);
    let spectrum = fft(
        &plot,
        &[],
        SpectrumSettings {
            window: SpectrumWindow::Rectangular,
            length: SpectrumLength::Padded,
            ..SpectrumSettings::default()
        },
        &mut transform_budget(),
    )
    .unwrap();
    assert_eq!(spectrum.point_count(), 17);
    assert!((spectrum.value("v(out)", 0).unwrap().re - 2.5).abs() < 1e-13);
    assert!((spectrum.value("frequency", 1).unwrap().re - 1. / 32.).abs() < 1e-15);
    let hann = fft(
        &plot,
        &[],
        SpectrumSettings {
            window: SpectrumWindow::Hann,
            length: SpectrumLength::Padded,
            ..SpectrumSettings::default()
        },
        &mut transform_budget(),
    )
    .unwrap();
    assert!((hann.value("v(out)", 0).unwrap().re - 2.5).abs() < 1e-13);
}

#[test]
fn control_vector_arithmetic_binds_columns_once_and_preserves_scale() {
    let deck = ScriptDeck::new(
        "vectors",
        "
run
let twice = 2*v(out)
let difference = v(in,out)
let twice = twice + difference
meas dc peak max twice
print dc1.twice",
    );
    let (report, text) = deck.run(ScriptLimits::default()).unwrap();
    assert!((report.variables["peak"] - 4.5).abs() < 1e-12);
    let plot = &report.plots[0].plot;
    assert!((plot.value("twice", 2).unwrap().re - 3.).abs() < 1e-12);
    assert!(text.contains("dc1.twice[2]"));
}

#[test]
fn control_transient_linearization_spectrum_and_plot_selection() {
    let deck = ScriptDeck::new(
        "fft-script",
        "
tran 1m 8m
linearize v(out) np=8
set specwindow=none
fft v(out)
let amplitude = mag(v(out))
print amplitude
setplot tran2
print v(out)",
    );
    let (report, output) = deck.run(ScriptLimits::default()).unwrap();
    assert_eq!(report.plots.len(), 3);
    assert_eq!(report.plots[1].plot.point_count(), 8);
    let spectrum = &report.plots[2].plot;
    assert!((spectrum.value("amplitude", 0).unwrap().re - 1.5).abs() < 1e-12);
    assert!(output.contains("amplitude[0]"));
}

#[test]
fn instance_edits_overlay_parameter_candidates_and_reset_transactionally() {
    let plan = divider();
    let mut session = ControlSession::new(&plan, EvalLimits::default()).unwrap();
    assert_eq!(
        session.alter_parameter("r", 2000.).unwrap(),
        CandidateTopology::Reused
    );
    // The first edit overrides the same physical site as the optimizer
    // coordinate. Three distinct sites exceed the two DC point slots.
    session.alter_instance("r1", None, 3000., 3).unwrap();
    session.alter_instance("r2", None, 2000., 3).unwrap();
    session.alter_instance("v1", Some("dc"), 10., 3).unwrap();
    session.alter_instance("r1", Some("r"), 4000., 3).unwrap();
    assert!(session.alter_instance("r1", None, 0., 3).is_err());
    assert!(session.alter_parameter("r", 500.).is_err());

    let plot = session
        .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 100)
        .unwrap();
    assert!((plot[0].value("v(out)", 0).unwrap().re - 10. / 3.).abs() < 1e-12);
    let sweep = session
        .run(
            AnalysisInput::with_arguments(AnalysisKind::DcSweep, ["v1", "0", "6", "3"]),
            1,
            100,
        )
        .unwrap();
    assert!((sweep[0].value("v(out)", 2).unwrap().re - 2.).abs() < 1e-12);
    let plot = session
        .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 100)
        .unwrap();
    assert!((plot[0].value("v(out)", 0).unwrap().re - 10. / 3.).abs() < 1e-12);
    assert_eq!(
        session
            .alter_parameters(std::iter::empty::<(&str, f64)>())
            .unwrap(),
        CandidateTopology::Reused
    );
    let plot = session
        .run(AnalysisInput::new(AnalysisKind::OperatingPoint), 1, 100)
        .unwrap();
    assert!((plot[0].value("v(out)", 0).unwrap().re - 1.).abs() < 1e-12);
}

#[test]
fn scripted_alter_sweep_and_reset_restore_source_values() {
    let deck = ScriptDeck::new(
        "instance-edits",
        "
alterparam r=2k
reset
alter @r1[r]=3k
alter r2 resistance=2k
alter v1=10
op
dc v1 0 6 3
op
reset
op",
    );
    let (report, _) = deck.run(ScriptLimits::default()).unwrap();
    assert_eq!(report.plots.len(), 4);
    for (index, point, expected) in [(0, 0, 4.), (1, 2, 2.4), (2, 0, 4.), (3, 0, 1.)] {
        let value = report.plots[index].plot.value("v(out)", point).unwrap().re;
        assert!((value - expected).abs() < 1e-12, "{index}: {value}");
    }
    assert!(
        deck.run(ScriptLimits {
            instance_edits: 2,
            ..ScriptLimits::default()
        })
        .err()
        .unwrap()
        .to_string()
        .contains("instance edit budget")
    );
}

#[test]
fn mixed_output_cards_stream_hidden_measurements_and_retain_only_selection() {
    let deck = ScriptDeck::from_text(
        "batch-output",
        "
mixed reports
v1 in 0 3
r1 in out 1k
r2 out 0 1k
.dc v1 0 3 1
.save v(out)
.print dc v(out)
.meas dc hidden max v(in)
.meas dc half param='hidden/2'
.control
run
print hidden half
meas dc observed max v(out)
.endc
.end",
    );
    let (report, text) = deck
        .run(ScriptLimits {
            cells: 8,
            ..ScriptLimits::default()
        })
        .unwrap();
    assert_eq!(report.plots.len(), 1);
    let plot = &report.plots[0].plot;
    assert_eq!(plot.variable_count(), 2);
    assert_eq!(plot.point_count(), 4);
    assert!(plot.value("v(in)", 3).is_none());
    assert!((report.variables["hidden"] - 3.).abs() < 1e-12);
    assert!((report.variables["half"] - 1.5).abs() < 1e-12);
    assert!((report.variables["observed"] - 1.5).abs() < 1e-12);
    assert!(text.contains("hidden"));
    assert!(text.contains("v(out)"));
}

#[test]
fn script_measurements_and_vector_expressions_request_device_observations() {
    let deck = ScriptDeck::new(
        "device-observations",
        "
run
let branch=2*i(r1)
meas dc current find i(r2) at=2
meas dc twice find branch at=2
print @r1[resistance]",
    );
    let (report, output) = deck.run(ScriptLimits::default()).unwrap();
    assert!((report.variables["current"] - 0.001).abs() < 1e-14);
    assert!((report.variables["twice"] - 0.002).abs() < 1e-14);
    assert!(output.contains("@r1[resistance]"));
}

#[test]
fn control_crossing_derivative_extrema_and_selected_plot() {
    let deck = ScriptDeck::new(
        "measure-operators",
        "
run
meas dc crossing when v(out)=.75
meas dc slope deriv v(out) at=1
meas dc range pp v(out)
meas dc peak max_at v(out)
alter r2=3k
run
setplot dc1
meas dc earlier max v(out)
destroy dc2
op
destroy
destroy all
run",
    );
    let (report, _) = deck.run(ScriptLimits::default()).unwrap();
    for (name, expected) in [
        ("crossing", 1.5),
        ("slope", 0.5),
        ("range", 1.5),
        ("peak", 3.),
        ("earlier", 1.5),
    ] {
        assert!((report.variables[name] - expected).abs() < 1e-12, "{name}");
    }
    assert_eq!(report.plots.len(), 1);
    assert_eq!(report.plots[0].plot.name, "dc1");
}

#[test]
fn mixed_fourier_card_uses_hidden_transient_signal() {
    let deck = ScriptDeck::from_text(
        "batch-fourier",
        "
Fourier report
v1 in 0 sin(0 2 1k)
r1 in out 1k
r2 out 0 1k
.tran 10u 2m
.save v(out)
.four 1k v(in)
.control
run
.endc
.end",
    );
    let (report, output) = deck.run(ScriptLimits::default()).unwrap();
    assert_eq!(report.plots.len(), 1);
    assert!(report.plots[0].plot.value("v(in)", 0).is_none());
    assert!(output.contains("Fourier analysis for"));
    assert!(output.contains("v(in)"));
}

#[test]
fn nested_control_flow_handles_transfers_and_counts() {
    let deck = ScriptDeck::new(
        "structured",
        r#"
set loops = 3
let i = 0
let total = 0
repeat $loops
  let i = i + 1
  if (i = 2)
    continue
  else
    let total = total + i
  end
end
dowhile (i < 3)
  let i = i + 1
end
repeat
  repeat
    let total = total + 10
    break 2
  end
end
let vec = 1
if vec = 1
  let total = total + vec
end
echo "total=$&total" i=$i
alter r1 2k
op
"#,
    );
    let (report, output) = deck.run(ScriptLimits::default()).unwrap();
    assert_eq!(report.variables["i"], 4.);
    assert_eq!(report.variables["total"], 15.);
    assert!(output.contains("total=15 i=4"));
    assert!((report.plots[0].plot.value("v(out)", 0).unwrap().re - 1.).abs() < 1e-12);
}

#[test]
fn invalid_flow_is_rejected_and_unbounded_repeat_is_bounded() {
    for script in [
        "break",
        "repeat 1\ncontinue 2\nend",
        "if 1\nelse\nelse\nend",
    ] {
        let deck = ScriptDeck::new("bad-flow", script);
        assert!(deck.run(ScriptLimits::default()).is_err());
    }
    let deck = ScriptDeck::new("repeat-bound", "repeat\nend");
    let limits = ScriptLimits {
        steps: 10,
        ..ScriptLimits::default()
    };
    assert!(
        deck.run(limits)
            .err()
            .unwrap()
            .to_string()
            .contains("instruction budget")
    );
}

#[test]
fn control_only_deck_executes_numeric_loops() {
    let deck = ScriptDeck::from_text(
        "control-only",
        "example while loop\n.control\nlet loopindex = 0\nwhile loopindex < 5\n  echo index is $&loopindex\n  let loopindex = loopindex + 1\nend\n.endc\n.end\n",
    );
    let (report, output) = deck.run(ScriptLimits::default()).unwrap();
    assert_eq!(report.variables["loopindex"], 5.);
    assert!(report.plots.is_empty());
    assert_eq!(
        output,
        "index is 0\nindex is 1\nindex is 2\nindex is 3\nindex is 4\n"
    );
}

#[test]
fn control_spectrum_explicit_mode_and_length() {
    let deck = ScriptDeck::new(
        "spectrum-modes",
        "\ntran 1m 8m\nlinearize v(out) np=7\nset specwindow=none\nfft mode=power length=exact v(out)\nsetplot tran2\nfft mode=psd length=padded v(out)\n",
    );
    let (report, _) = deck.run(ScriptLimits::default()).unwrap();
    let power = &report.plots[2].plot;
    let density = &report.plots[3].plot;
    assert_eq!(power.point_count(), 4);
    assert_eq!(density.point_count(), 5);
    assert_eq!(power.flags(), PlotFlags::Real);
    assert_eq!(density.flags(), PlotFlags::Real);
    let power_sum = power
        .column("v(out)")
        .unwrap()
        .iter()
        .map(|v| v.re)
        .sum::<f64>();
    let density_sum = density
        .column("v(out)")
        .unwrap()
        .iter()
        .map(|v| v.re)
        .sum::<f64>();
    let df = density.value("frequency", 1).unwrap().re;
    assert!((power_sum - 2.25).abs() < 1e-12);
    assert!((density_sum * df - power_sum).abs() < 1e-12);
}

#[test]
fn complex_linearization_preserves_both_components() {
    let mut plot = Plot::new(
        "tran",
        "Transient Analysis",
        PlotFlags::Complex,
        vec![
            Variable::new("time", "time"),
            Variable::complex("v(out)", "voltage"),
        ],
    );
    for time in [0., 0.1, 0.3, 0.6, 1.] {
        plot.push_point(&[Complex::real(time), Complex::new(2. * time + 3., 1. - time)])
            .unwrap();
    }
    let uniform = linearize(
        &plot,
        TimeGrid {
            start: 0.,
            stop: 1.,
            step: 0.125,
        },
        None,
        &[],
        &mut transform_budget(),
    )
    .unwrap();
    assert_eq!(uniform.flags(), PlotFlags::Complex);
    for row in uniform.rows() {
        assert!((row[1].re - (2. * row[0].re + 3.)).abs() < 1e-14);
        assert!((row[1].im - (1. - row[0].re)).abs() < 1e-14);
    }
    let spectrum = fft(
        &uniform,
        &[],
        SpectrumSettings {
            window: SpectrumWindow::Rectangular,
            ..SpectrumSettings::default()
        },
        &mut transform_budget(),
    )
    .unwrap();
    assert_eq!(spectrum.point_count(), 9);
    assert!((spectrum.value("v(out)", 0).unwrap() - Complex::new(4., 0.5)).magnitude() < 1e-12);
    assert!(spectrum.value("frequency", 8).unwrap().re < 0.);
}
