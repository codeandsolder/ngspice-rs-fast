//! Bounded deterministic performance workloads. See docs/port/PERFORMANCE.md.
use std::time::Duration;

#[path = "performance_workloads/decks.rs"]
mod decks;
#[path = "performance_workloads/matrix.rs"]
mod matrix;

fn report(workload: &str, phase: &str, operations: u32, elapsed: Duration) {
    println!(
        "workload={workload},phase={phase},operations={operations},elapsed_ns={}",
        elapsed.as_nanos()
    );
}

#[test]
fn small_workloads_are_valid() {
    matrix::real("smoke_ladder", 8, 1, 2);
    matrix::real("smoke_mesh", 4, 4, 2);
    matrix::frequency_ensemble(4, 4, 3);
    decks::smoke();
}

#[test]
#[ignore = "opt-in sparse ladder performance workload"]
fn sparse_ladder() {
    matrix::real("sparse_ladder", 1024, 1, 32);
}

#[test]
#[ignore = "opt-in sparse mesh performance workload"]
fn sparse_mesh() {
    matrix::real("sparse_mesh", 32, 32, 24);
}

#[test]
#[ignore = "opt-in independent complex frequency ensemble"]
fn complex_frequency_ensemble() {
    matrix::frequency_ensemble(24, 24, 64);
}

#[test]
#[ignore = "opt-in RC transient workload"]
fn rc_transient() {
    decks::rc();
}

#[test]
#[ignore = "opt-in nonlinear mixed-device workload"]
fn nonlinear_mixed() {
    decks::nonlinear();
}

#[test]
#[ignore = "opt-in compact model array workload"]
fn compact_models() {
    decks::compact();
}

#[test]
#[ignore = "opt-in RF multiport workload"]
fn rf_multiport() {
    decks::rf();
}

#[test]
#[ignore = "opt-in long LTRA convolution workload"]
fn ltra_history() {
    decks::ltra();
}

#[test]
#[ignore = "opt-in independent parameter ensemble"]
fn parameter_ensemble() {
    decks::parameters();
}

/// Compare independent frequency tiles against independently refactored
/// scalar matrices. Assembly is outside the timed region: this measures
/// numerical factorization, not triplet growth or sink overhead.
#[test]
#[ignore = "opt-in independent complex factorization crossover"]
fn independent_complex_factor_crossover() {
    use std::hint::black_box;
    use std::sync::Arc;
    use std::time::Instant;

    use ngspice_rs::maths::complex::{ComplexMatrix, IndependentComplexBatch};
    use ngspice_rs::maths::{Pattern, SparseMatrix};
    use ngspice_rs::primitives::Complex;

    for size in [8_usize, 32, 128] {
        let positions = (0..size).flat_map(|row| {
            [
                row.checked_sub(1),
                Some(row),
                (row + 1 < size).then_some(row + 1),
            ]
            .into_iter()
            .flatten()
            .map(move |col| (row, col))
        });
        let pattern = Arc::new(Pattern::new(size, positions).unwrap());
        let mut g = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        let mut c = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        for row in 0..size {
            g.add(row, row, 3.).unwrap();
            c.add(row, row, 0.01 + (row % 7) as f64 * 0.001).unwrap();
            if row + 1 < size {
                g.add(row, row + 1, -0.25).unwrap();
                g.add(row + 1, row, -0.25).unwrap();
            }
        }
        for count in [4_usize, 16, 64] {
            let matrices: Vec<_> = (0..count)
                .map(|i| {
                    ComplexMatrix::from_operators_on(
                        &pattern,
                        &g,
                        &c,
                        0.01 + (i as f64 + 1.).powi(2) * 0.03125,
                    )
                    .unwrap()
                    .unwrap()
                })
                .collect();
            let start = Instant::now();
            let mut previous = None;
            let scalar: Vec<_> = matrices
                .iter()
                .map(|matrix| {
                    let factor = matrix.factorize_independent(previous.as_ref()).unwrap();
                    previous = Some(factor.clone());
                    black_box(factor)
                })
                .collect();
            let scalar_elapsed = start.elapsed();
            let start = Instant::now();
            let mut batch = IndependentComplexBatch::default();
            let packed: Vec<_> = matrices
                .chunks(4)
                .flat_map(|tile| batch.factorize(tile).into_iter().map(Result::unwrap))
                .map(black_box)
                .collect();
            let packed_elapsed = start.elapsed();
            assert_eq!(scalar.len(), packed.len());
            let rhs: Vec<_> = (0..size)
                .map(|i| Complex::new(1. + i as f64 * 0.01, -0.1))
                .collect();
            let mut worst = 0.0_f64;
            for (reference, candidate) in scalar.iter().zip(&packed) {
                for (a, b) in reference
                    .solve(&rhs)
                    .unwrap()
                    .iter()
                    .zip(candidate.solve(&rhs).unwrap())
                {
                    worst = worst.max((*a - b).magnitude());
                }
                for (a, b) in reference
                    .solve_transposed(&rhs)
                    .unwrap()
                    .iter()
                    .zip(candidate.solve_transposed(&rhs).unwrap())
                {
                    worst = worst.max((*a - b).magnitude());
                }
            }
            assert!(
                worst < 1e-11,
                "independent tiled factor mismatch: {worst:e}"
            );
            println!(
                "workload=independent_factor,size={size},count={count},scalar_ns={},simd_tile_ns={},speedup={:.3},max_solution_diff={worst:e}",
                scalar_elapsed.as_nanos(),
                packed_elapsed.as_nanos(),
                scalar_elapsed.as_secs_f64() / packed_elapsed.as_secs_f64(),
            );
        }
    }
}
