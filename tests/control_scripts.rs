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
        let path =
            std::env::temp_dir().join(format!("spice-control-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(
            path.join("deck.cir"),
            format!("divider\n.param r=1k\nv1 in 0 3\nr1 in out {{r}}\nr2 out 0 1k\n.dc v1 0 3 1\n.control\n{script}\n.endc\n.end\n"),
        ).unwrap();
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
use ngspice_rs::analysis::control::vectors::{
    SpectrumWindow, TimeGrid, TransformBudget, fft, linearize,
};
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
            SpectrumWindow::Rectangular,
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
        SpectrumWindow::Rectangular,
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
        SpectrumWindow::Rectangular,
        &mut transform_budget(),
    )
    .unwrap();
    assert_eq!(spectrum.point_count(), 17);
    assert!((spectrum.value("v(out)", 0).unwrap().re - 2.5).abs() < 1e-13);
    assert!((spectrum.value("frequency", 1).unwrap().re - 1. / 32.).abs() < 1e-15);
    let hann = fft(&plot, &[], SpectrumWindow::Hann, &mut transform_budget()).unwrap();
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
