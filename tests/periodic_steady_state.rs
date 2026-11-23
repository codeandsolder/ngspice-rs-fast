//! Circuit-level periodic shooting regressions, including independent
//! analytic and long-transient references.
use std::path::Path;

use ngspice_rs::analysis::{AnalysisContext, AnalysisPlan, Plot, PointSink, RunConfig};
use ngspice_rs::devices::Circuit;
use ngspice_rs::netlist::{Parser, source::parse_deck_text};
use ngspice_rs::primitives::{AnalysisKind, Complex, SpiceResult};

const RADIAL: &str = "c1 x 0 1u ic=0.3\nc2 y 0 1u ic=0\n\
    b1 x 0 i=1m*v(y)-0.5m*(1-v(x)^2-v(y)^2)*v(x)\n\
    b2 y 0 i=-1m*v(x)-0.5m*(1-v(x)^2-v(y)^2)*v(y)";

const VDP: &str = "c1 x 0 1u ic=0.5\nc2 y 0 1u ic=0\n\
    b1 x 0 i=-1m*v(y)\n\
    b2 y 0 i=1m*v(x)-3m*(1-v(x)^2)*v(y)";

const LC: &str = "c1 x 0 1n ic=0.3\nl1 x 0 1u\n\
    b1 x 0 i=0.001*(v(x)^3-v(x))";

#[derive(Default)]
struct Capture {
    plots: Vec<Plot>,
    active: Option<Plot>,
}
impl PointSink for Capture {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        self.active = Some(header.header());
        Ok(())
    }
    fn point(&mut self, row: &[Complex]) -> SpiceResult<()> {
        self.active.as_mut().unwrap().push_point(row)
    }
    fn end(&mut self) -> SpiceResult<()> {
        self.plots.push(self.active.take().unwrap());
        Ok(())
    }
}

fn prepared(body: &str, analyses: &str) -> (Circuit, Vec<AnalysisPlan>, AnalysisContext) {
    let deck = parse_deck_text(
        Path::new("orbit.cir"),
        &format!(
            "orbit\n{body}\n.options reltol=1e-6 vntol=1e-8 abstol=1e-11 chgtol=1e-13\n{analyses}\n.end\n"
        ),
    );
    let netlist = Parser::new().parse_deck(&deck).unwrap();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let plans = config.plans_for(&netlist.analyses).unwrap();
    let circuit = config.circuit(&netlist).unwrap();
    (circuit, plans, config.context())
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    ((actual - expected) / expected).abs()
}

#[test]
fn radial_orbit_matches_exact_period_amplitude_and_spectrum() {
    let expected_period = std::f64::consts::TAU / 1000.;
    for (body, analysis) in [
        (RADIAL.to_owned(), ".pss 140 2m x 128 4 30 1m uic"),
        (
            RADIAL.replace("ic=0.3", "ic=0.8"),
            ".pss 180 1m y 128 4 30 1m uic",
        ),
    ] {
        let (mut circuit, plans, context) = prepared(&body, analysis);
        let mut output = Capture::default();
        let orbit = plans[0]
            .run_periodic(&mut circuit, &context, None, &mut output)
            .unwrap();
        assert!(
            relative_error(orbit.period(), expected_period) < 0.005,
            "{orbit:?}"
        );
        assert!(
            orbit.residual() <= 1e-3 && orbit.charge_residual() <= 1e-3,
            "{orbit:?}"
        );
        assert!(orbit.transverse_bound() < 1.);
        assert_eq!(output.plots.len(), 2);
        let time = &output.plots[0];
        assert_eq!(time.point_count(), 129);
        for i in 0..=128 {
            let t = time.value("time", i).unwrap().re;
            let x = time.value("v(x)", i).unwrap().re;
            let y = time.value("v(y)", i).unwrap().re;
            assert!((x.hypot(y) - 1.).abs() < 0.005, "radius at {t}: {x}, {y}");
        }
        let spectrum = &output.plots[1];
        assert!(
            relative_error(
                spectrum.value("frequency", 1).unwrap().re,
                orbit.frequency()
            ) < 1e-12
        );
        for node in ["v(x)", "v(y)"] {
            assert!((spectrum.value(node, 1).unwrap().re - 1.).abs() < 0.005);
            assert!(spectrum.value(node, 2).unwrap().re.abs() < 0.005);
        }
        let mut next = Capture::default();
        let warmed = plans[0]
            .run_periodic(&mut circuit, &context, Some(&orbit), &mut next)
            .unwrap();
        assert!(relative_error(warmed.period(), orbit.period()) < 0.005);
        assert!(warmed.accepted_steps() < orbit.accepted_steps());
    }
}

fn transient_period(plot: &Plot, node: &str) -> f64 {
    let time = plot.column("time").unwrap();
    let voltage = plot.column(node).unwrap();
    let mut previous: Option<(f64, f64)> = None;
    let mut last_crossings = Vec::new();
    for (t, v) in time.iter().zip(voltage.iter()) {
        let (t, v) = (t.re, v.re);
        if let Some((old_t, old_v)) = previous
            && old_v < 0.
            && v >= 0.
        {
            let fraction = -old_v / (v - old_v);
            last_crossings.push(fraction.mul_add(t - old_t, old_t));
        }
        previous = Some((t, v));
    }
    let last = last_crossings.len();
    last_crossings[last - 1] - last_crossings[last - 2]
}

#[test]
fn relaxation_and_rf_lc_match_independent_long_transients() {
    for (body, analysis, tolerance) in [
        (
            VDP,
            ".pss 110 20m x 128 0 30 2m uic\n.tran 10u 100m 80m 10u uic",
            0.015,
        ),
        (
            LC,
            ".pss 5meg 3u x 128 0 30 2m uic\n.tran 500p 12u 10u 500p uic",
            0.01,
        ),
    ] {
        let (mut circuit, plans, context) = prepared(body, analysis);
        let mut output = Capture::default();
        let orbit = plans[0]
            .run_periodic(&mut circuit, &context, None, &mut output)
            .unwrap();
        let reference = plans[1].run(&mut circuit, &context).unwrap();
        let expected = transient_period(&reference, "v(x)");
        assert!(
            relative_error(orbit.period(), expected) < tolerance,
            "{orbit:?}, reference {expected:e}"
        );
    }
}

#[test]
fn refuses_non_autonomous_forcing_and_missing_cycles() {
    for (body, expected) in [
        (format!("{RADIAL}\nv1 forced 0 sin(0 1 100)"), "autonomous"),
        (format!("{RADIAL}\nb3 forced 0 v=time"), "explicit time"),
        (
            "r1 x 0 1k\nc1 x 0 1u\nr2 y 0 1k\nc2 y 0 1u".into(),
            "oscillation",
        ),
    ] {
        let (mut circuit, plans, context) = prepared(&body, ".pss 150 1m x 64 0 10 1m uic");
        let error = plans[0].run(&mut circuit, &context).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
    let plan = ngspice_rs::analysis::AnalysisInput::with_arguments(
        AnalysisKind::PeriodicSteadyState,
        ["0", "1m", "x"],
    );
    assert!(plan.resolve().is_err());
}

#[test]
fn period_guess_can_span_multiple_cycles() {
    let (mut circuit, plans, context) = prepared(RADIAL, ".pss 65 30m x 256 0 30 1m uic");
    let orbit = plans[0]
        .run_periodic(&mut circuit, &context, None, &mut Capture::default())
        .unwrap();
    assert!(
        relative_error(orbit.period(), std::f64::consts::TAU / 1000.) < 0.005,
        "{orbit:?}"
    );
}

#[test]
#[ignore = "requires absolute NGSPICE_BIN; runs external C transient references"]
fn relaxation_and_rf_lc_match_c_transients() {
    use ngspice_conformance::ngspice::Scratch;
    use ngspice_conformance::process::{ProcessLimits, run};
    use ngspice_rs::analysis::RawFile;
    let binary = std::env::var_os("NGSPICE_BIN").expect("set absolute NGSPICE_BIN");
    assert!(Path::new(&binary).is_absolute());
    for (body, analysis, transient, tolerance) in [
        (
            VDP,
            ".pss 110 20m x 128 0 30 2m uic",
            "tran 10u 100m 80m 10u uic",
            0.015,
        ),
        (
            LC,
            ".pss 5meg 3u x 128 0 30 2m uic",
            "tran 500p 12u 10u 500p uic",
            0.01,
        ),
    ] {
        let scratch = Scratch::new("c-pss-reference").unwrap();
        let dir = scratch.path();
        std::fs::write(dir.join("orbit.cir"), format!(
            "orbit\n{body}\n.options reltol=1e-6 vntol=1e-8 abstol=1e-11 chgtol=1e-13\n.control\nset filetype=ascii\n{transient}\nwrite reference.raw\nquit\n.endc\n.end\n"
        )).unwrap();
        let result = run(&binary, dir, &["-b", "orbit.cir"], ProcessLimits::ORACLE).unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let raw = RawFile::load(dir.join("reference.raw")).unwrap();
        let expected = transient_period(&raw.plots[0].plot, "v(x)");
        let (mut circuit, plans, context) = prepared(body, analysis);
        let orbit = plans[0]
            .run_periodic(&mut circuit, &context, None, &mut Capture::default())
            .unwrap();
        assert!(
            relative_error(orbit.period(), expected) < tolerance,
            "{orbit:?}, C reference {expected:e}"
        );
        eprintln!(
            "PSS period {:e}; C period {expected:e}; accepted steps {}; C stored points {}",
            orbit.period(),
            orbit.accepted_steps(),
            raw.plots[0].plot.point_count()
        );
    }
}
