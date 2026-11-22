//! Reproducible long-history coefficient and transient workloads.
use ngspice_rs::{
    analysis::RunConfig,
    devices::ltra::convolution::{Coefficients, rc_coefficients, rlc_coefficients},
    netlist::{Parser, source::parse_deck_text},
};
use std::{collections::VecDeque, hint::black_box, path::Path, time::Instant};

fn time_grid(count: i32, wrapped: bool) -> VecDeque<f64> {
    // Alternating accepted intervals, with the same logical time grid for
    // contiguous and wrapped storage. No uniform-grid assumption.
    let mut tick = 0_i32;
    let capacity = usize::try_from(count).unwrap();
    let mut times = VecDeque::with_capacity(capacity);
    let samples = (0..count).map(|i| {
        tick += 100 + i % 13;
        f64::from(tick) * 1e-13
    });
    if wrapped {
        times.extend((0..count / 3).map(|i| f64::from(i) - f64::from(count)));
        for time in samples {
            if times.len() == capacity {
                times.pop_front();
            }
            times.push_back(time);
        }
    } else {
        times.extend(samples);
    }
    times
}

fn checksum(coefficients: &Coefficients) -> u64 {
    coefficients
        .first
        .into_iter()
        .chain(coefficients.weights.iter().copied().flatten())
        .fold(u64::try_from(coefficients.aux_index).unwrap(), |hash, v| {
            assert!(v.is_finite(), "nonfinite coefficient");
            (hash ^ v.to_bits()).wrapping_mul(0x100_0000_01b3)
        })
}

fn bench_coefficients(kind: &str, count: i32, wrapped: bool, compute: impl Fn() -> Coefficients) {
    let coefficients = compute();
    println!(
        "coefficient_signature,{kind},{count},{wrapped},{:016x}",
        checksum(&coefficients)
    );
    let iterations = (131_072 / count).clamp(4, 4096);
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(compute());
        }
        println!(
            "coefficient_time,{kind},{count},{wrapped},{iterations},{}",
            start.elapsed().as_nanos()
        );
    }
}

#[test]
#[ignore = "optimized coefficient benchmark"]
fn coefficient_workloads() {
    for count in [8, 128, 2048, 16_384] {
        for wrapped in [false, true] {
            let times = time_grid(count, wrapped);
            let now = times.back().unwrap() + 1e-11;
            bench_coefficients("RC", count, wrapped, || {
                rc_coefficients(1e-12, 4e-8, now, black_box(&times), 0.)
            });
            bench_coefficients("RLC", count, wrapped, || {
                rlc_coefficients(5e-9, 2e8, 2e8, now, black_box(&times), 0.)
            });
        }
    }
}

#[test]
#[ignore = "optimized long-transient benchmark"]
fn transient_workloads() {
    for (kind, model) in [
        ("RC", "r=100 c=100p len=2"),
        ("RLC", "r=10 l=250n c=100p len=1"),
    ] {
        let deck = format!(
            "ltra long history\nv1 in 0 sin(0 1 50meg)\nrs in a 50\no1 a 0 b 0 ll\nrl b 0 50\n.model ll ltra {model}\n.tran 0.02n 20n 0 0.02n\n.end\n"
        );
        let netlist = Parser::new()
            .parse_deck(&parse_deck_text(Path::new("ltra_bench.cir"), &deck))
            .unwrap();
        let config = RunConfig::from_netlist(&netlist).unwrap();
        let request = config.plan_for(&netlist.analyses[0]).unwrap();
        for _ in 0..3 {
            let mut circuit = config.circuit(&netlist).unwrap();
            let start = Instant::now();
            let plot = request.run(&mut circuit, &config.context()).unwrap();
            let elapsed = start.elapsed().as_nanos();
            let time = plot.column("time").unwrap();
            let v = plot.column("v(b)").unwrap();
            assert!(time.len() >= 1000 && time.len() == v.len());
            let digest = time.iter().zip(v).fold(0_u64, |h, (t, v)| {
                assert!(v.re.is_finite() && t.re.is_finite());
                (h ^ t.re.to_bits()).wrapping_mul(0x100_0000_01b3) ^ v.re.to_bits()
            });
            println!(
                "transient_time,{kind},{},{digest:016x},{elapsed}",
                time.len()
            );
        }
    }
}

#[test]
fn chopping_matches_the_full_history_in_reverse_order() {
    for count in [2, 4, 8, 9, 13, 128, 2048] {
        let times = time_grid(count, true);
        let now = times.back().unwrap() + 1e-11;
        let rc = |tol| rc_coefficients(1e-12, 4e-8, now, &times, tol);
        for delay in [0., 5e-9, now, now + 1e-9] {
            let rlc = |tol| rlc_coefficients(delay, 2e8, 2e8, now, &times, tol);
            for (inclusive, generate) in
                [(false, &rc as &dyn Fn(f64) -> Coefficients), (true, &rlc)]
            {
                let full = generate(0.);
                for tol in [1e-6, 0.01, 1.] {
                    let chopped = generate(tol);
                    assert_eq!(
                        full.first.map(f64::to_bits),
                        chopped.first.map(f64::to_bits)
                    );
                    assert_eq!(full.aux_index, chopped.aux_index);
                    for k in 0..3 {
                        let threshold = (full.first[k] * tol).abs();
                        let end = if k == 0 {
                            full.weights.len() - 1
                        } else {
                            full.aux_index
                        };
                        let mut active = true;
                        for i in (1..=end).rev() {
                            let expected = if active { full.weights[i][k] } else { 0. };
                            assert_eq!(
                                chopped.weights[i][k].to_bits(),
                                expected.to_bits(),
                                "count={count},delay={delay:e},tol={tol:e},k={k},i={i}"
                            );
                            if if inclusive {
                                expected.abs() <= threshold
                            } else {
                                expected.abs() < threshold
                            } {
                                active = false;
                            }
                        }
                    }
                }
            }
        }
    }
}
