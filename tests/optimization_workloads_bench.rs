//! Independent repeated simulation baseline; not a continuation sweep or
//! an optimizer implementation. See docs/port/optimization-workloads.md.
use std::hint::black_box;
use std::path::Path;
use std::time::{Duration, Instant};

use ngspice_rs::analysis::bias::solve_dc_with;
use ngspice_rs::analysis::{AnalysisPlan, Plot, PointSink, RunConfig};
use ngspice_rs::netlist::{Parser, ast::Netlist, source::parse_deck_text};
use ngspice_rs::primitives::{Complex, SpiceResult};

#[derive(Clone, Copy, Debug)]
struct Candidate {
    r1: f64,
    r2: f64,
    c1: f64,
    c2: f64,
    saturation: f64,
    ideality: f64,
}

impl Candidate {
    const TARGET: Self = Self {
        r1: 2200.,
        r2: 3300.,
        c1: 22e-9,
        c2: 47e-9,
        saturation: 2e-12,
        ideality: 1.5,
    };

    fn at(index: u32) -> Self {
        // Non-Cartesian, deterministic and reproducible. No optimizer bias.
        let fraction = |multiplier: u32| f64::from(index * multiplier % 127) / 126.;
        Self {
            r1: fraction(17).mul_add(3000., 1000.),
            r2: fraction(31).mul_add(5000., 1500.),
            c1: fraction(43).mul_add(40e-9, 10e-9),
            c2: fraction(53).mul_add(60e-9, 15e-9),
            saturation: 10_f64.powf(fraction(67).mul_add(3., -13.)),
            ideality: fraction(73).mul_add(0.75, 1.25),
        }
    }

    fn valid(self) -> bool {
        [
            self.r1,
            self.r2,
            self.c1,
            self.c2,
            self.saturation,
            self.ideality,
        ]
        .into_iter()
        .all(|v| v.is_finite() && v > 0.)
    }
}

#[derive(Clone, Copy, Debug)]
enum Case {
    Ac { load: f64 },
    Transient,
    Diode,
}

const CASES: [Case; 4] = [
    Case::Ac { load: 10_000. },
    Case::Ac { load: 2200. },
    Case::Transient,
    Case::Diode,
];
const TRANSIENT_TIMES: [f64; 8] = [2e-6, 5e-6, 10e-6, 20e-6, 40e-6, 80e-6, 120e-6, 180e-6];

impl Case {
    const fn plot_name(self) -> &'static str {
        match self {
            Self::Ac { .. } => "AC Analysis",
            Self::Transient => "Transient Analysis",
            Self::Diode => "DC transfer characteristic",
        }
    }

    const fn observation(self) -> &'static str {
        match self {
            Self::Diode => "i(v1)",
            _ => "v(out)",
        }
    }

    fn deck(self, c: Candidate) -> String {
        match self {
            Self::Ac { load } => format!(
                "two-pole filter\n.param ra={:.17e} rb={:.17e} ca={:.17e} cb={:.17e}\nv1 in 0 ac 1\nr1 in mid {{ra}}\nc1 mid 0 {{ca}}\nr2 mid out {{rb}}\nc2 out 0 {{cb}}\nrl out 0 {load:.17e}\n.ac lin 25 100 25000\n.end\n",
                c.r1, c.r2, c.c1, c.c2
            ),
            Self::Transient => format!(
                "RC step objective\n.param rr={:.17e} cc={:.17e}\nv1 in 0 pulse(0 1 0 2u 2u 2m 4m)\nr1 in out {{rr}}\nc1 out 0 {{cc}}\n.tran 2u 200u 0 200n\n.end\n",
                c.r1, c.c1
            ),
            Self::Diode => format!(
                "diode fitting objective\n.param sat={:.17e} ideal={:.17e}\nv1 out 0 0\nd1 out 0 fit\n.model fit d(is={{sat}} n={{ideal}} rs=0)\n.options temp=27 tnom=27\n.dc v1 0.05 0.5 0.05\n.end\n",
                c.saturation, c.ideality
            ),
        }
    }

    // Independent closed-form responses of the physical circuits. They are
    // intentionally not derived from simulator stamps or internal states.
    fn expected(self, c: Candidate, axis: f64) -> Complex {
        match self {
            Self::Ac { load } => {
                let w = std::f64::consts::TAU * axis;
                let g1 = c.r1.recip();
                let g2 = c.r2.recip();
                let a = g1 + g2;
                let b = g2 + load.recip();
                let real = (-(w * c.c1)).mul_add(w * c.c2, a.mul_add(b, -g2 * g2));
                let imag = w * a.mul_add(c.c2, b * c.c1);
                let numerator = g1 * g2;
                let norm = real.mul_add(real, imag * imag);
                Complex::new(numerator * real / norm, -numerator * imag / norm)
            }
            Self::Transient => {
                let tau = c.r1 * c.c1;
                let rise = 2e-6;
                let value = if axis <= rise {
                    tau.mul_add((-axis / tau).exp_m1(), axis) / rise
                } else {
                    let at_end = (tau / rise).mul_add((-rise / tau).exp_m1(), 1.);
                    (at_end - 1.).mul_add((-(axis - rise) / tau).exp(), 1.)
                };
                Complex::real(value)
            }
            Self::Diode => {
                // C ngspice reference physical constants, at nominal 27 C.
                let vt = 300.15 * (1.380_648_52e-23 / 1.602_176_620_8e-19);
                let current = c.saturation * (axis / (c.ideality * vt)).exp_m1();
                Complex::real(-current)
            }
        }
    }

    const fn scale(self) -> f64 {
        match self {
            Self::Diode => 1e-4,
            _ => 1.,
        }
    }

    fn check(self, c: Candidate, axis: f64, value: Complex) {
        let expected = self.expected(c, axis);
        let tolerance = match self {
            Self::Ac { .. } => 2e-10,
            // tau >= 10 us, maximum dt = 200 ns/50 times smaller.
            // Include interpolation error and numerical integration error.
            Self::Transient => 2e-3,
            // Include the default junction gmin and DC convergence error.
            Self::Diode => 1e-9 + 2e-5 * expected.magnitude(),
        };
        assert!(value.re.is_finite() && value.im.is_finite());
        assert!(
            (value - expected).magnitude() <= tolerance,
            "{self:?} at {axis:e}, candidate {c:?}: got {value:?} expected {expected:?} (tol {tolerance:e})"
        );
    }
}

struct Objective {
    case: Case,
    candidate: Candidate,
    active: bool,
    column: usize,
    grid_index: usize,
    previous: Option<(f64, Complex)>,
    sum: f64,
    samples: u32,
    plots: u32,
}

impl Objective {
    const fn new(case: Case, candidate: Candidate) -> Self {
        Self {
            case,
            candidate,
            active: false,
            column: 0,
            grid_index: 0,
            previous: None,
            sum: 0.,
            samples: 0,
            plots: 0,
        }
    }

    fn sample(&mut self, axis: f64, value: Complex) {
        self.case.check(self.candidate, axis, value);
        let error =
            (value - self.case.expected(Candidate::TARGET, axis)).magnitude() / self.case.scale();
        self.sum = error.mul_add(error, self.sum);
        self.samples += 1;
    }

    fn finish(self) -> f64 {
        assert_eq!(self.plots, 1);
        match self.case {
            Case::Ac { .. } => assert_eq!(self.samples, 25),
            Case::Transient => assert_eq!(self.samples, 8),
            Case::Diode => assert_eq!(self.samples, 10),
        }
        let result = self.sum / f64::from(self.samples);
        assert!(result.is_finite());
        result
    }
}

impl PointSink for Objective {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        self.active = header.plotname == self.case.plot_name();
        if self.active {
            self.column = header.variable_index(self.case.observation()).unwrap();
        }
        Ok(())
    }

    fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        if !self.active {
            return Ok(());
        }
        let axis = values[0].re;
        let value = values[self.column];
        if matches!(self.case, Case::Transient) {
            if let Some((last, previous)) = self.previous {
                assert!(axis >= last);
                while let Some(&time) = TRANSIENT_TIMES.get(self.grid_index) {
                    if time > axis {
                        break;
                    }
                    assert!(time >= last);
                    let sample = if time >= axis {
                        value
                    } else if time <= last {
                        previous
                    } else {
                        let fraction = (time - last) / (axis - last);
                        Complex::real(fraction.mul_add(value.re - previous.re, previous.re))
                    };
                    self.sample(time, sample);
                    self.grid_index += 1;
                }
            }
            self.previous = Some((axis, value));
        } else {
            self.sample(axis, value);
        }
        Ok(())
    }

    fn end(&mut self) -> SpiceResult<()> {
        if self.active {
            self.plots += 1;
        }
        self.active = false;
        Ok(())
    }
}

#[derive(Default)]
struct Timing {
    parse: Duration,
    prepare: Duration,
    evaluate: Duration,
}

fn parse(text: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(Path::new("optimization.cir"), text))
        .unwrap()
}

fn prepare(netlist: &Netlist) -> (RunConfig, AnalysisPlan) {
    let config = RunConfig::from_netlist(netlist).unwrap();
    let plan = config.plan_for(&netlist.analyses[0]).unwrap();
    (config, plan)
}

fn evaluate(c: Candidate, retained: bool, timing: &mut Timing) -> Option<[f64; 4]> {
    // Workload-level candidate contract, not a replacement solver validator.
    if !c.valid() {
        return None;
    }
    let mut results = [0.; 4];
    for (result, case) in results.iter_mut().zip(CASES) {
        let start = Instant::now();
        let netlist = parse(&case.deck(c));
        timing.parse += start.elapsed();
        let start = Instant::now();
        let (config, plan) = prepare(&netlist);
        let mut circuit = config.circuit(&netlist).unwrap();
        timing.prepare += start.elapsed();
        let start = Instant::now();
        let mut objective = Objective::new(case, c);
        if retained {
            for plot in plan.run_plots(&mut circuit, &config.context()).unwrap() {
                objective.complete(plot).unwrap();
            }
        } else {
            plan.run_into(&mut circuit, &config.context(), &mut objective)
                .unwrap();
        }
        *result = objective.finish();
        timing.evaluate += start.elapsed();
    }
    Some(results)
}

fn compare(a: [f64; 4], b: [f64; 4]) {
    for (a, b) in a.into_iter().zip(b) {
        assert!(
            (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1.),
            "{a:e} vs {b:e}"
        );
    }
}

fn repeated_dc(count: u32) {
    let netlist = parse("reused topology\nv1 in 0 1\nr1 in out 1k\nr2 out 0 1k\n.op\n.end\n");
    let (config, plan) = prepare(&netlist);
    let circuit = config.circuit(&netlist).unwrap();
    let row = circuit
        .unknowns()
        .node_row(circuit.nodes().get("out").unwrap())
        .unwrap();
    for fresh in [true, false] {
        let start = Instant::now();
        for i in 0..count {
            let c = Candidate::at(i);
            let source = f64::from(i % 17 + 1) / 3.;
            let fresh_circuit = fresh.then(|| config.circuit(&netlist).unwrap());
            let active = fresh_circuit.as_ref().unwrap_or(&circuit);
            let model = config
                .context()
                .model_context()
                .with_resistor_override(active.resistor_override("r1", c.r1).unwrap())
                .unwrap()
                .with_resistor_override(active.resistor_override("r2", c.r2).unwrap())
                .unwrap();
            let dc = solve_dc_with(
                active,
                &model,
                plan.dc_settings(),
                &[("v1", source)],
                None,
                None,
            )
            .unwrap();
            let expected = source * c.r2 / (c.r1 + c.r2);
            assert!((dc.solution.values.as_slice()[row] - expected).abs() < 1e-9);
            black_box(dc);
        }
        let elapsed = start.elapsed();
        println!(
            "workload=repeated_dc,fresh_topology={fresh},candidates={count},elapsed_ns={}",
            elapsed.as_nanos()
        );
    }
}

#[test]
fn independent_objectives_match_and_reject_invalid_candidate() {
    let mut timing = Timing::default();
    let a = Candidate::TARGET;
    let b = Candidate::at(37);
    let first = evaluate(a, false, &mut timing).unwrap();
    compare(first, evaluate(a, true, &mut timing).unwrap());
    compare(
        evaluate(b, false, &mut timing).unwrap(),
        evaluate(b, true, &mut timing).unwrap(),
    );
    let invalid = Candidate { c1: f64::NAN, ..a };
    assert!(evaluate(invalid, false, &mut timing).is_none());
    compare(first, evaluate(a, false, &mut timing).unwrap());
    repeated_dc(12);
}

#[test]
#[ignore = "opt-in repeated design and model-fit workload"]
fn repeated_optimization_ensemble() {
    const COUNT: u32 = 128;
    let mut retained = Timing::default();
    let mut objective = Timing::default();
    let mut reference = Vec::new();
    for i in 0..COUNT {
        reference.push(evaluate(Candidate::at(i), true, &mut retained).unwrap());
    }
    // Reverse order and replay one candidate after other independent runs.
    for (i, expected) in (0..COUNT).zip(reference.iter()).rev() {
        compare(
            *expected,
            evaluate(Candidate::at(i), false, &mut objective).unwrap(),
        );
    }
    for (mode, timing) in [("retained", retained), ("objective_only", objective)] {
        println!(
            "workload=optimization_ensemble,mode={mode},candidates={COUNT},scenarios={},parse_ns={},prepare_ns={},evaluate_ns={}",
            COUNT * 4,
            timing.parse.as_nanos(),
            timing.prepare.as_nanos(),
            timing.evaluate.as_nanos()
        );
    }
    repeated_dc(16384);
}
