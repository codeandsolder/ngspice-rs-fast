//! Opt-in breakpoint scheduling throughput. Exercise actual BreakTable
//! operations, with no timing in the correctness oracle.
use std::hint::black_box;
use std::time::Instant;

use ngspice_rs::devices::BreakTable;

fn schedule(shuffled: bool) {
    for n in [4096_u32, 16_384, 65_536] {
        let final_time = f64::from(n) + 1.;
        for trial in 0..3 {
            let mut table = BreakTable::new(final_time, 0.).unwrap();
            let start = Instant::now();
            for i in 0..n {
                let event = if shuffled {
                    i.wrapping_mul(40503) & (n - 1)
                } else {
                    i
                };
                table.set(0., black_box(f64::from(event + 1))).unwrap();
            }
            let insert_ns = start.elapsed().as_nanos();
            let start = Instant::now();
            for i in 0..=n {
                assert_eq!(table.first().to_bits(), f64::from(i).to_bits());
                assert_eq!(table.second().to_bits(), f64::from(i + 1).to_bits());
                table.clear();
            }
            let drain_ns = start.elapsed().as_nanos();
            assert_eq!(table.first().to_bits(), final_time.to_bits());
            assert_eq!(table.second().to_bits(), final_time.to_bits());
            println!(
                "breakpoints,shuffled={shuffled},n={n},trial={trial},insert_ns={insert_ns},drain_ns={drain_ns}"
            );
        }
    }
}

#[test]
#[ignore = "opt-in large breakpoint workload"]
fn ascending() {
    schedule(false);
}

#[test]
#[ignore = "opt-in large breakpoint workload"]
fn shuffled() {
    schedule(true);
}
