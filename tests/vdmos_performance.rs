//! Opt-in native VDMOS execution cost measurements. Not a CI timing threshold.
//! Compare the homogeneous prepared-kernel batch against scalar calls and
//! full circuit load/commit separately; repository splitting is not assumed
//! to improve execution throughput.
use std::{fmt::Write as _, hint::black_box, path::Path, time::Instant};

use model_kernel_vdmos::{Bias, Instance, Polarity, compile_ngspice, evaluate_batch};
use ngspice_rs::{
    analysis::{RunConfig, bias::solve_dc_with, companion_transient},
    devices::{AnalysisMode, IterationPhase, LoadRequest},
    maths::Vector,
    netlist::{Parser, source::parse_deck_text},
};

#[test]
#[ignore = "run manually: prepared VDMOS scalar/batch and bind cost"]
fn prepared_kernel_batch_throughput() {
    let started = Instant::now();
    let model = compile_ngspice(
        Polarity::N,
        [
            ("vto", 3.0),
            ("kp", 20.0),
            ("rd", 0.03),
            ("rs", 0.01),
            ("rg", 0.5),
            ("rb", 0.04),
            ("cgs", 5e-10),
            ("cgdmax", 7e-10),
            ("cgdmin", 9e-11),
            ("cjo", 2e-10),
            ("is", 1e-12),
            ("tt", 5e-8),
            ("vp", 0.35),
            ("rthjc", 1.2),
            ("rthca", 7.0),
            ("cthj", 0.001),
        ],
    )
    .unwrap();
    println!(
        "kernel_preparation_seconds={:.9}",
        started.elapsed().as_secs_f64()
    );
    let start = Instant::now();
    for _ in 0..100_000 {
        black_box(
            model
                .bind(black_box(Instance {
                    multiplier: 1.0,
                    self_heating: true,
                }))
                .unwrap(),
        );
    }
    println!(
        "kernel_instance_bind_ns={:.3}",
        start.elapsed().as_secs_f64() * 1e9 / 100_000.0
    );
    let device = model
        .bind(Instance {
            multiplier: 1.0,
            self_heating: true,
        })
        .unwrap();
    for tiles in [32_usize, 128, 512, 2048] {
        let bias: Vec<Bias> = (0..tiles)
            .map(|i| {
                let x = i as f64;
                Bias {
                    vgs: 2.4 + 5.0 * ((i * 17 % 127) as f64) / 127.0,
                    vds: 0.1 + 14.0 * ((i * 31 % 131) as f64) / 131.0,
                    vds_external: 0.2 + 13.0 * ((i * 31 % 131) as f64) / 131.0,
                    vdiode: -0.2 + 0.8 * ((i * 13 % 97) as f64) / 97.0,
                    recovery_node: 1e-3 + 1e-4 * ((i * 11 % 19) as f64),
                    temperature: 290.0 + 0.02 * x,
                }
            })
            .collect();
        let seed = device.evaluate(bias[0]).unwrap();
        let mut scalar = vec![seed; tiles];
        let mut batched = vec![seed; tiles];
        let repeats = (262_144 / tiles).max(16);
        let mut scalar_ns = f64::INFINITY;
        let mut batch_ns = f64::INFINITY;
        for _ in 0..4 {
            let start = Instant::now();
            for _ in 0..repeats {
                for (point, slot) in bias.iter().zip(&mut scalar) {
                    *slot = device.evaluate(black_box(*point)).unwrap();
                }
                black_box(&scalar);
            }
            scalar_ns =
                scalar_ns.min(start.elapsed().as_secs_f64() * 1e9 / (repeats * tiles) as f64);
            let start = Instant::now();
            for _ in 0..repeats {
                evaluate_batch(&device, black_box(&bias), black_box(&mut batched)).unwrap();
                black_box(&batched);
            }
            batch_ns = batch_ns.min(start.elapsed().as_secs_f64() * 1e9 / (repeats * tiles) as f64);
        }
        for (index, (a, b)) in scalar.iter().zip(&batched).enumerate() {
            assert_eq!(
                a.channel.value.to_bits(),
                b.channel.value.to_bits(),
                "lane {index} current"
            );
            assert_eq!(
                a.qgd.value.to_bits(),
                b.qgd.value.to_bits(),
                "lane {index} gate charge"
            );
            assert_eq!(
                a.heat.d_temperature.to_bits(),
                b.heat.d_temperature.to_bits(),
                "lane {index} thermal Jacobian"
            );
            assert_eq!(
                a.diode.current.value.to_bits(),
                b.diode.current.value.to_bits(),
                "lane {index} diode"
            );
        }
        println!(
            "kernel_tile={tiles},evaluations={},scalar_ns_per_device={scalar_ns:.3},batch_ns_per_device={batch_ns:.3},ratio_scalar_over_batch={:.4}",
            repeats * tiles,
            scalar_ns / batch_ns
        );
    }
}

fn replicated_devices(n: usize) -> String {
    let mut deck = String::from(
        "VDMOS shared-model MNA benchmark\nVdd supply 0 12\nVgate gate 0 4\n.model dut vdmos(vto=3 kp=20 rd=0.02 rs=0.01 rg=0.5 rb=0.04 cgs=500p cgdmax=700p cgdmin=90p cjo=200p is=1e-12 tt=50n vp=0.35)\n",
    );
    for i in 0..n {
        writeln!(deck, "R{i} supply drain{i} 10\nM{i} drain{i} gate 0 dut").unwrap();
    }
    deck.push_str(".op\n.end\n");
    deck
}

#[test]
#[ignore = "run manually: repeated VDMOS sparse-MNA load versus pure physics"]
fn prepared_model_circuit_stamping_throughput() {
    for instances in [32_usize, 128, 512] {
        let source = replicated_devices(instances);
        let start = Instant::now();
        let parsed = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("vdmos-stamp-bench.cir"),
                &source,
            ))
            .unwrap();
        let config = RunConfig::from_netlist(&parsed).unwrap();
        let plan = config.plan_for(&parsed.analyses[0]).unwrap();
        let circuit = config.circuit(&parsed).unwrap();
        let setup = start.elapsed().as_secs_f64();
        let model_context = config.context().model_context();
        let start = Instant::now();
        let dc = solve_dc_with(
            &circuit,
            &model_context,
            plan.dc_settings(),
            &[],
            None,
            None,
        )
        .unwrap();
        let solve = start.elapsed().as_secs_f64();
        let x = dc.solution.values;
        let history = circuit.state_history();
        let request = LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &x,
            model_context: &model_context,
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
        stamp();
        let repeats = (32_768 / instances).max(8);
        let mut best = f64::INFINITY;
        for _ in 0..4 {
            let start = Instant::now();
            for _ in 0..repeats {
                stamp();
            }
            best = best.min(start.elapsed().as_secs_f64());
        }
        let residual = (matrix
            .mul_vector(&x)
            .unwrap()
            .as_slice()
            .iter()
            .zip(rhs.as_slice())
            .map(|(ax, b)| (ax - b).abs()))
        .fold(0.0_f64, f64::max);
        assert!(
            residual < 1e-5,
            "invalid post-stamp solution residual {residual:e}"
        );
        println!(
            "circuit_instances={instances},devices={},unknowns={},pattern_nnz={},dc_iterations={},setup_s={setup:.6},initial_solve_s={solve:.6},load_us={:.3},load_ns_per_vdmos={:.2},residual={residual:e}",
            circuit.device_count(),
            circuit.unknown_count(),
            matrix.nnz(),
            dc.report.total_iterations,
            best * 1e6 / repeats as f64,
            best * 1e9 / (repeats * instances) as f64
        );
        black_box((matrix, rhs, trial));
    }
}

#[test]
#[ignore = "run manually: complete synchronous VDMOS buck end-to-end cost"]
fn synchronous_buck_adaptive_transient_throughput() {
    let source = include_str!("fixtures/vdmos/synchronous_buck.cir");
    let cards = source.split(".control").next().unwrap();
    let deck = parse_deck_text(
        Path::new("vdmos-buck-bench.cir"),
        &format!("{cards}\n.tran 50n 200u 0 50n maxsteps=100000\n.end\n"),
    );
    let parsed = Parser::new().parse_deck(&deck).unwrap();
    let config = RunConfig::from_netlist(&parsed).unwrap();
    let plan = config.plan_for(&parsed.analyses[0]).unwrap();
    for _ in 0..3 {
        let start = Instant::now();
        let mut circuit = config.circuit(&parsed).unwrap();
        let setup = start.elapsed();
        let start = Instant::now();
        let (plot, stats) = companion_transient(&mut circuit, &plan, &config.context()).unwrap();
        let elapsed = start.elapsed();
        let vout = plot.value("v(out)", plot.point_count() - 1).unwrap().re;
        assert!(
            (vout - 4.93615).abs() < 0.1,
            "buck output did not match C47"
        );
        println!(
            "buck_setup_s={:.6},analysis_s={:.6},accepted={},rejected={},breakpoints={},points={},unknowns={},vout={vout:.8},min_dt={:.3e},max_dt={:.3e}",
            setup.as_secs_f64(),
            elapsed.as_secs_f64(),
            stats.accepted,
            stats.rejected,
            stats.breakpoints,
            plot.point_count(),
            circuit.unknown_count(),
            stats.min_step,
            stats.max_step
        );
        black_box((plot, circuit));
    }
}
