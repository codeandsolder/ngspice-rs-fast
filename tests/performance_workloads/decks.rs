//! Full parsed-deck workloads. All dimensions and stimulus sequences are
//! fixed in source, with independently specified physical checks.
use std::fmt::Write as _;
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use ngspice_rs::analysis::bias::solve_dc_with;
use ngspice_rs::analysis::{
    AnalysisPlan, Plot, PointSink, RawFile, RunConfig, companion_transient,
};
use ngspice_rs::devices::{AnalysisMode, Circuit, IterationPhase, LoadRequest};
use ngspice_rs::maths::Vector;
use ngspice_rs::netlist::{Parser, ast::Netlist, source::parse_deck_text};
use ngspice_rs::primitives::{Complex, SpiceResult};

use super::report;

fn setup(name: &str, text: &str) -> (Netlist, RunConfig, AnalysisPlan, Circuit) {
    let start = Instant::now();
    let netlist = Parser::new()
        .parse_deck(&parse_deck_text(Path::new("performance.cir"), text))
        .unwrap();
    report(name, "parse", 1, start.elapsed());
    let start = Instant::now();
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let plan = config.plan_for(&netlist.analyses[0]).unwrap();
    let circuit = config.circuit(&netlist).unwrap();
    report(name, "elaborate_setup", 1, start.elapsed());
    println!(
        "workload={name},devices={},unknowns={},state_slots={},pattern_nnz={}",
        circuit.device_count(),
        circuit.unknown_count(),
        circuit.state_len(),
        circuit.matrix_pattern().map_or(0, |pattern| pattern.len())
    );
    (netlist, config, plan, circuit)
}

#[derive(Default)]
struct Counts {
    rows: usize,
    cells: usize,
    plots: usize,
    width: Option<usize>,
}
impl PointSink for Counts {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        assert!(self.width.is_none());
        self.width = Some(header.variable_count());
        Ok(())
    }
    fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        assert_eq!(Some(values.len()), self.width);
        self.rows += 1;
        self.cells += values.len();
        black_box(values);
        Ok(())
    }
    fn end(&mut self) -> SpiceResult<()> {
        assert!(self.width.take().is_some());
        self.plots += 1;
        Ok(())
    }
}

fn run(name: &str, text: &str, verify: impl Fn(&Plot)) {
    let (netlist, config, plan, mut circuit) = setup(name, text);
    for _ in 0..3 {
        let started = Instant::now();
        let plots = plan.run_plots(&mut circuit, &config.context()).unwrap();
        report(name, "analysis_retained", 1, started.elapsed());
        assert!(!plots.is_empty());
        for plot in &plots {
            assert!(plot.is_finite());
        }
        verify(&plots[0]);
        let rows = plots.iter().map(Plot::point_count).sum::<usize>();
        let cells = plots
            .iter()
            .map(|p| p.point_count() * p.variable_count())
            .sum::<usize>();
        println!(
            "workload={name},plots={},rows={rows},cells={cells}",
            plots.len()
        );

        let mut counts = Counts::default();
        let mut streaming = config.circuit(&netlist).unwrap();
        let start = Instant::now();
        plan.run_into(&mut streaming, &config.context(), &mut counts)
            .unwrap();
        report(name, "analysis_streamed", 1, start.elapsed());
        assert_eq!(
            (counts.plots, counts.rows, counts.cells),
            (plots.len(), rows, cells)
        );
        assert!(counts.width.is_none());

        let mut encode = Duration::ZERO;
        let mut decode = Duration::ZERO;
        for plot in plots {
            let raw = RawFile::single(plot);
            let start = Instant::now();
            let bytes = black_box(raw.to_binary().unwrap());
            encode += start.elapsed();
            let start = Instant::now();
            let read = black_box(RawFile::parse_bytes(&bytes).unwrap());
            decode += start.elapsed();
            // Rawfile stores the display plotname, not the internal name
            // (e.g. tran1). Compare exactly the serialized schema and cells.
            let got = &read.plots[0].plot;
            let want = &raw.plots[0].plot;
            assert_eq!(got.plotname, want.plotname);
            assert_eq!(got.flags(), want.flags());
            assert_eq!(got.variable_count(), want.variable_count());
            // Binary rawfiles encode real/complex at plot granularity.
            // Per-column realness is not serialized in the complex format.
            for (a, b) in got.variables().iter().zip(want.variables()) {
                assert_eq!((&a.name, &a.unit), (&b.name, &b.unit));
            }
            assert_eq!(got.point_count(), want.point_count());
            for variable in want.variables() {
                assert_eq!(
                    want.variables()
                        .iter()
                        .filter(|candidate| candidate.name == variable.name)
                        .count(),
                    1,
                    "benchmark requires unique serialized column identities"
                );
                let want_column = want.column(&variable.name).unwrap();
                let got_column = got.column(&variable.name).unwrap();
                for (row, (a, b)) in got_column.into_iter().zip(want_column).enumerate() {
                    let imaginary = if !want.flags().is_complex() || variable.is_real {
                        0.0
                    } else {
                        b.im
                    };
                    assert_eq!(
                        (a.re.to_bits(), a.im.to_bits()),
                        (b.re.to_bits(), imaginary.to_bits()),
                        "{name}: codec mismatch at {} row {row}",
                        variable.name
                    );
                }
            }
        }
        report(name, "binary_encode", 1, encode);
        report(name, "binary_decode", 1, decode);
        // Every trial starts from a fresh circuit with independent history.
        circuit = config.circuit(&netlist).unwrap();
    }
}

fn load_kernel(name: &str, text: &str, repeats: u32) {
    let (_, config, plan, mut circuit) = setup(name, text);
    const FINALIZE_REPEATS: u32 = 10_000;
    let start = Instant::now();
    for _ in 0..FINALIZE_REPEATS {
        black_box(&mut circuit).finalize().unwrap();
    }
    report(
        name,
        "finalize_unchanged",
        FINALIZE_REPEATS,
        start.elapsed(),
    );
    let model = config.context().model_context();
    let start = Instant::now();
    let dc = solve_dc_with(&circuit, &model, plan.dc_settings(), &[], None, None).unwrap();
    report(name, "bias_initialization", 1, start.elapsed());
    println!(
        "workload={name},dc_iterations={},dc_stages={}",
        dc.report.total_iterations,
        dc.report.stages.len()
    );
    let x = dc.solution.values;
    let history = circuit.state_history().unwrap();
    let request = LoadRequest {
        mode: AnalysisMode::OperatingPoint,
        solution: &x,
        model_context: &model,
        integration: None,
        history: &history,
        forcing: None,
    };
    let mut matrix = circuit.new_matrix().unwrap();
    let mut rhs = Vector::zeros(circuit.unknown_count());
    let mut trial = history
        .trial_in(IterationPhase::Float, Some(&dc.solution.trial))
        .unwrap()
        .with_device_limiting(false);
    let mut stamp = || {
        matrix.clear();
        rhs.clear();
        circuit
            .load(&request, &mut matrix, &mut rhs, &mut trial)
            .unwrap();
    };
    for _ in 0..3 {
        stamp();
    }
    // Isolate repeatable device evaluation + bound stamping. Restamp the
    // same solved bias without accepting a step or timing the linear solve.
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..repeats {
            stamp();
        }
        report(name, "device_load", repeats, start.elapsed());
    }
    // Physical residual of the loaded Jacobian against the converged bias.
    let ax = matrix.mul_vector(&x).unwrap();
    let residual = ax
        .as_slice()
        .iter()
        .zip(rhs.as_slice())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0., f64::max);
    assert!(
        rhs.is_finite() && x.is_finite() && residual.is_finite() && residual < 1e-5,
        "{name}: bad DC residual {residual:e}"
    );
    println!(
        "workload={name},assembled_nnz={},dc_residual_abs={residual:e}",
        matrix.nnz()
    );
}

fn rc_text(n: u32, steps: u32) -> String {
    let mut deck = String::from("RC ladder\nv1 n0 0 pulse(0 1 1u 0.1u 0.1u 100u)\n");
    for i in 1..=n {
        writeln!(deck, "r{i} n{} n{i} 100\nc{i} n{i} 0 10n", i - 1).unwrap();
    }
    writeln!(deck, ".tran 0.1u {}u 0 0.1u\n.end", f64::from(steps) / 10.).unwrap();
    deck
}

fn transient_stats(name: &str, text: &str) {
    let (_, config, plan, mut circuit) = setup(name, text);
    let start = Instant::now();
    let (plot, stats) = companion_transient(&mut circuit, &plan, &config.context()).unwrap();
    report(name, "transient_with_stats", 1, start.elapsed());
    assert!(plot.is_finite() && stats.accepted > 0);
    println!(
        "workload={name},accepted={},rejected={},breakpoints={},min_step={:e},max_step={:e}",
        stats.accepted, stats.rejected, stats.breakpoints, stats.min_step, stats.max_step
    );
    if name == "ltra_history" {
        let work = stats.ltra;
        println!(
            "workload={name},attempts={},newton_loads={},line_loads={},coefficient_preparations={},convolution_evaluations={}",
            work.attempts,
            work.newton_loads,
            work.line_loads,
            work.coefficient_preparations,
            work.convolution_evaluations
        );
        assert_eq!(work.attempts, stats.accepted + stats.rejected);
        assert_eq!(work.coefficient_preparations, work.attempts);
        assert_eq!(work.convolution_evaluations, work.attempts);
        assert_eq!(work.line_loads, work.newton_loads);
        assert!(work.line_loads > work.convolution_evaluations);
    }
}

pub(super) fn rc() {
    let text = rc_text(64, 1000);
    run("rc_transient", &text, |plot| {
        let out = plot.column("v(n1)").unwrap();
        for v in out {
            assert!((-1e-5..=1.00001).contains(&v.re));
        }
        assert!((plot.value("v(n1)", 0).unwrap().re).abs() < 1e-12);
    });
    transient_stats("rc_transient", &text);
}

fn nonlinear_text(copies: u32, sweeps: u32) -> String {
    let mut deck = String::from(
        "heterogeneous inverter/diode array\nvdd supply 0 5\nvin in 0 0\n.model nm nmos level=1 vto=1 kp=100u\n.model pm pmos level=1 vto=-1 kp=50u\n.model diode d is=1e-14 n=1\n",
    );
    for i in 0..copies {
        writeln!(deck,"mn{i} out{i} in 0 0 nm w=10u l=1u\nmp{i} out{i} in supply supply pm w=20u l=1u\nr{i} out{i} clamp{i} 10k\nd{i} clamp{i} 0 diode\nc{i} out{i} 0 10f").unwrap();
    }
    writeln!(deck, ".dc vin 0 5 {}\n.end", 5. / f64::from(sweeps)).unwrap();
    deck
}

fn verify_inverter(plot: &Plot) {
    let first = plot.value("v(out0)", 0).unwrap().re;
    let last = plot.value("v(out0)", plot.point_count() - 1).unwrap().re;
    assert!(
        first > 4. && last < 1.,
        "inverter endpoints {first}, {last}"
    );
}

pub(super) fn nonlinear() {
    let text = nonlinear_text(48, 64);
    run("nonlinear_mixed", &text, verify_inverter);
    // A separate operating-point load fixture with both channels conducting.
    let text = text.replace("vin in 0 0", "vin in 0 2.5");
    load_kernel("nonlinear_device_load", &text, 128);
}

fn compact_text(copies: u32, sweeps: u32) -> String {
    let mut deck = String::from(
        "repeated BSIM3/BSIM4 devices\nvdd supply 0 1.2\nvg g 0 0.7\n.model n3 nmos level=49 version=3.3.0 capmod=3 mobmod=1 tox=4.1n nch=2.3549e17 vth0=0.3694 u0=268.7 rsh=6.8\n.model n4 nmos level=54 version=4.8.3 vth0=0.45 toxe=4n rgatemod=1 rbodymod=1 xrcrg1=12 xrcrg2=5 rshg=0.4 rbpb=5 rbpd=15 rbps=15 rbdb=15 rbsb=15\n",
    );
    for i in 0..copies {
        let model = if i.is_multiple_of(2) { "n3" } else { "n4" };
        writeln!(deck,"r{i} supply d{i} 1k\nm{i} d{i} g 0 0 {model} l=0.18u w=1u ad=1e-13 as=1e-13 pd=1.4u ps=1.4u").unwrap();
    }
    writeln!(deck, ".dc vg 0.4 1.0 {}\n.end", 0.6 / f64::from(sweeps)).unwrap();
    deck
}

fn verify_compact(plot: &Plot) {
    for node in ["v(d0)", "v(d1)"] {
        let first = plot.value(node, 0).unwrap().re;
        let last = plot.value(node, plot.point_count() - 1).unwrap().re;
        assert!(
            first - last > 0.01,
            "{node} did not vary with gate bias: {first}, {last}"
        );
    }
    for i in 0..plot.point_count() {
        for node in ["v(d0)", "v(d1)"] {
            let v = plot.value(node, i).unwrap().re;
            assert!((-1e-6..=1.200001).contains(&v), "{node} at {i}: {v}");
        }
    }
}

pub(super) fn compact() {
    let text = compact_text(96, 32);
    run("compact_model_array", &text, verify_compact);
    load_kernel("compact_model_device_load", &text, 64);
}

fn rf_text(ports: u32, points: u32) -> String {
    let mut deck = String::from("grounded RC multiport\n");
    for i in 1..=ports {
        writeln!(
            deck,
            "v{i} p{i} 0 dc 0 ac 0 portnum {i} z0 50\nr{i} p{i} mid 50\nc{i} p{i} 0 10p"
        )
        .unwrap();
    }
    deck.push_str("rload mid 0 50\n");
    writeln!(deck, ".sp lin {points} 1k 1gig\n.end").unwrap();
    deck
}

fn verify_rf(plot: &Plot) {
    for i in 0..plot.point_count() {
        let a = plot.value("S_1_2", i).unwrap();
        let b = plot.value("S_2_1", i).unwrap();
        assert!((a - b).magnitude() < 1e-9 && a.magnitude() <= 1.000001);
    }
}

pub(super) fn rf() {
    run("rf_multiport", &rf_text(8, 64), verify_rf);
}

fn ltra_text(steps: u32) -> String {
    format!(
        "long lossy convolution\nv1 in 0 pulse(0 1 1n 0.5n 0.5n 8n 20n)\nrs in a 50\no1 a 0 b 0 line\nrl b 0 120\n.model line ltra r=5 l=250n c=100p len=2 mixedinterp\n.tran 0.2n {}n 0 0.2n\n.end",
        f64::from(steps) / 5.
    )
}

fn verify_ltra(plot: &Plot) {
    assert!(plot.value("v(b)", 0).unwrap().re.abs() < 1e-9);
    let out = plot.column("v(b)").unwrap();
    assert!(out.into_iter().any(|v| v.re > 0.1));
}

pub(super) fn ltra() {
    let text = ltra_text(1500);
    run("ltra_history", &text, verify_ltra);
    transient_stats("ltra_history", &text);
}

pub(super) fn parameters() {
    let text = "independent resistor and stimulus ensemble\nv1 in 0 1\nr1 in out 1k\nr2 out 0 1k\n.dc @r1[resistance] 1k 16k 1k v1 1 8 1\n.end\n";
    run("parameter_ensemble", text, |plot| {
        assert_eq!(plot.point_count(), 128);
        for outer in 0..8_u32 {
            for inner in 0..16_u32 {
                let got = plot
                    .value("v(out)", (outer * 16 + inner) as usize)
                    .unwrap()
                    .re;
                let want = f64::from(outer + 1) / f64::from(inner + 2);
                assert!(
                    (got - want).abs() < 1e-9,
                    "parameter sweep {outer},{inner}: {got} vs {want}"
                );
            }
        }
    });
    let (netlist, config, plan, _) = setup("independent_parameters", text);
    for _ in 0..3 {
        let mut preparation = Duration::ZERO;
        let mut solve = Duration::ZERO;
        let mut iterations = 0;
        let mut worst = 0.0_f64;
        for source in 1..=8_u32 {
            for resistor in 1..=16_u32 {
                let start = Instant::now();
                let circuit = config.circuit(&netlist).unwrap();
                let model = config
                    .context()
                    .model_context()
                    .with_resistor_override(
                        circuit
                            .resistor_override("r1", 1000. * f64::from(resistor))
                            .unwrap(),
                    )
                    .unwrap();
                preparation += start.elapsed();
                let start = Instant::now();
                // Fresh circuit and no prior Newton iterate: unlike the DC
                // sweep, independent points are safe to run concurrently.
                let dc = solve_dc_with(
                    &circuit,
                    &model,
                    plan.dc_settings(),
                    &[("v1", f64::from(source))],
                    None,
                    None,
                )
                .unwrap();
                solve += start.elapsed();
                let node = circuit.nodes().get("out").unwrap();
                let row = circuit.unknowns().node_row(node).unwrap();
                let value = dc.solution.values.as_slice()[row];
                let expected = f64::from(source) / f64::from(resistor + 1);
                worst = worst.max((value - expected).abs());
                iterations += dc.report.total_iterations;
            }
        }
        assert!(worst < 1e-9);
        report("independent_parameters", "prepare", 128, preparation);
        report("independent_parameters", "solve", 128, solve);
        println!(
            "workload=independent_parameters,dc_iterations={iterations},forward_abs={worst:e}"
        );
    }
}

pub(super) fn smoke() {
    run("smoke_rc", &rc_text(4, 100), |p| {
        assert!(p.point_count() > 10)
    });
    let mixed = nonlinear_text(2, 4);
    run("smoke_nonlinear", &mixed, verify_inverter);
    load_kernel(
        "smoke_nonlinear_load",
        &mixed.replace("vin in 0 0", "vin in 0 2.5"),
        2,
    );
    let compact = compact_text(2, 4);
    run("smoke_compact", &compact, verify_compact);
    load_kernel("smoke_compact_load", &compact, 2);
    run("smoke_rf", &rf_text(2, 3), verify_rf);
    run("smoke_ltra", &ltra_text(150), verify_ltra);
    parameters();
}
