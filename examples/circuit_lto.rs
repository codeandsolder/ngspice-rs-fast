//! Whole-simulator circuit specialization experiment.
use ngspice_rs::{analysis::RunConfig, netlist::{Parser, source::parse_deck_text}};
use std::{path::Path, time::Instant, hint::black_box};

const CASES: &[(&str, &str, usize)] = &[
    ("rc_transient", include_str!("../conformance/netlists/rc_transient.cir"), 2000),
    ("m4_diode_tran", include_str!("../conformance/netlists/m4_diode_tran.cir"), 500),
    ("m11_bsim4_tran", include_str!("../conformance/netlists/m11_bsim4_tran.cir"), 250),
    ("m11_bsim4_dc", include_str!("../conformance/netlists/m11_bsim4_dc.cir"), 2000),
    ("diode100_op", include_str!("../conformance/netlists/diode100_op.cir"), 500),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mode = &args[1];
    let case: usize = args[2].parse()?;
    let (name, text, default_count) = CASES[case];
    let path = Path::new(name);
    let netlist = Parser::new().parse_deck(&parse_deck_text(path, text))?;
    let config = RunConfig::from_netlist(&netlist)?;
    let plans = config.plans_for(&netlist.analyses)?;
    if mode == "emit" {
        let circuit = config.circuit(&netlist)?;
        print!("{}", circuit.emit_specialization(case, &config.context().model_context())?);
        return Ok(());
    }
    if mode == "verify" {
        let run = |specialization: Option<bool>| -> Result<_, Box<dyn std::error::Error>> {
            let mut circuit = config.circuit(&netlist)?;
            if let Some(bake) = specialization {
                circuit.enable_specialization(case, bake, &config.context().model_context())?;
            }
            let mut output = Vec::new();
            for plan in &plans {
                output.extend(plan.run_plots(&mut circuit, &config.context())?);
            }
            Ok(output)
        };
        let reference = run(None)?;
        for bake in [false, true] {
            let candidate = run(Some(bake))?;
            assert_eq!(reference.len(), candidate.len());
            for (a,b) in reference.iter().zip(&candidate) {
                assert_eq!(a.variables(), b.variables());
                assert_eq!(a.point_count(), b.point_count());
                for (ar,br) in a.rows().zip(b.rows()) {
                    for (x,y) in ar.iter().zip(br) {
                        assert_eq!(x.re.to_bits(), y.re.to_bits());
                        assert_eq!(x.im.to_bits(), y.im.to_bits());
                    }
                }
            }
        }
        println!("{name}: all plots bitwise identical");
        return Ok(());
    }
    let count = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(default_count);
    let mut digest = 0u64;
    let mut points = 0usize;
    let mut setup = 0u128;
    let mut solve = 0u128;
    let mut result = 0u128;
    for _ in 0..count {
        let start = Instant::now();
        let mut circuit = config.circuit(&netlist)?;
        if mode != "generic" {
            circuit.enable_specialization(case, mode == "baked", &config.context().model_context())?;
        }
        setup += start.elapsed().as_nanos();
        for plan in &plans {
            let start = Instant::now();
            let plots = plan.run_plots(&mut circuit, &config.context())?;
            solve += start.elapsed().as_nanos();
            let start = Instant::now();
            for plot in plots {
                points += plot.point_count();
                for row in plot.rows() {
                    for cell in row {
                        digest = digest.rotate_left(7) ^ cell.re.to_bits();
                        digest = digest.rotate_left(7) ^ cell.im.to_bits();
                    }
                }
                black_box(plot);
            }
            result += start.elapsed().as_nanos();
        }
    }
    println!("{name} {mode} count={count} setup_ns={setup} solve_ns={solve} result_ns={result} points={points} digest={digest:016x}");
    Ok(())
}
