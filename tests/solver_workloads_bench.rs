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

/// Opt-in two-dimensional mesh: substantially more fill-in and Schur work
/// than the one-dimensional ladder crossover above.
#[test]
#[ignore = "opt-in independent complex mesh factorization crossover"]
fn independent_complex_mesh_crossover() {
    use std::hint::black_box;
    use std::sync::Arc;
    use std::time::Instant;

    use ngspice_rs::maths::complex::{ComplexMatrix, IndependentComplexBatch};
    use ngspice_rs::maths::{Pattern, SparseMatrix};
    use ngspice_rs::primitives::Complex;

    for side in [4_usize, 8, 16, 32] {
        let size = side * side;
        let positions = (0..size).flat_map(|row| {
            [
                Some(row),
                (row >= side).then_some(row.saturating_sub(side)),
                (row + side < size).then_some(row + side),
                (row % side != 0).then_some(row.saturating_sub(1)),
                (row % side + 1 < side).then_some(row + 1),
            ]
            .into_iter()
            .flatten()
            .map(move |col| (row, col))
        });
        let pattern = Arc::new(Pattern::new(size, positions).unwrap());
        let mut g = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        let mut c = SparseMatrix::bound(Arc::clone(&pattern)).unwrap();
        for node in 0..size {
            g.add(node, node, 4. + (node % 5) as f64 * 0.125).unwrap();
            c.add(node, node, (node % 13 + 1) as f64 * 0.001).unwrap();
            for next in [
                (node + side < size).then_some(node + side),
                (node % side + 1 < side).then_some(node + 1),
            ]
            .into_iter()
            .flatten()
            {
                g.add(node, next, -0.5).unwrap();
                g.add(next, node, -0.5).unwrap();
            }
        }
        let counts: &[usize] = if side >= 32 { &[8, 32] } else { &[8, 32, 128] };
        for &count in counts {
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
                    let factor = match previous.as_ref() {
                        Some(old) => matrix.factorize_reusing(old).unwrap(),
                        None => matrix.factorize().unwrap(),
                    };
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
                .map(|i| Complex::new(1. + i as f64 * 0.001, -0.1))
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
            }
            assert!(worst < 1e-11, "mesh independent tile mismatch: {worst:e}");
            println!(
                "workload=independent_mesh,side={side},n={size},count={count},cached_scalar_ns={},simd_tile_ns={},speedup={:.3},max_solution_diff={worst:e}",
                scalar_elapsed.as_nanos(),
                packed_elapsed.as_nanos(),
                scalar_elapsed.as_secs_f64() / packed_elapsed.as_secs_f64(),
            );
            // Compare the real AC workload: refactor and immediately solve
            // one RHS, dropping its factors before the next frequency. The
            // packed path avoids scalar factor snapshots entirely.
            drop(scalar);
            drop(packed);
            drop(previous);
            drop(batch);
            let start = Instant::now();
            let reference_solutions: Vec<_> = matrices
                .iter()
                .map(|matrix| {
                    let factor = matrix.factorize().unwrap();
                    black_box(factor.solve(&rhs).unwrap())
                })
                .collect();
            let reference_elapsed = start.elapsed();
            let rhs_lanes = vec![rhs.as_slice(); count];
            let mut direct = IndependentComplexBatch::default();
            let start = Instant::now();
            let direct_solutions: Vec<_> = direct
                .solve(&matrices, &rhs_lanes, false)
                .unwrap()
                .into_iter()
                .map(Result::unwrap)
                .map(black_box)
                .collect();
            let direct_elapsed = start.elapsed();
            let mut worst = 0.0_f64;
            for (reference, candidate) in reference_solutions.iter().zip(&direct_solutions) {
                for (a, b) in reference.iter().zip(candidate) {
                    worst = worst.max((*a - *b).magnitude());
                }
            }
            assert!(worst < 1e-11, "packed solve mismatch: {worst:e}");
            println!(
                "workload=independent_mesh_solve,side={side},n={size},count={count},cached_scalar_ns={},fused_simd_ns={},speedup={:.3},max_solution_diff={worst:e}",
                reference_elapsed.as_nanos(),
                direct_elapsed.as_nanos(),
                reference_elapsed.as_secs_f64() / direct_elapsed.as_secs_f64(),
            );
        }
    }
}

/// Retained-factor real RHS traversal over increasingly filled, nonsymmetric
/// bands. Verify every batched answer against the scalar path before timing.
#[test]
#[ignore = "opt-in real sparse RHS traversal across fill patterns"]
fn real_sparse_rhs_fill_crossover() {
    use std::hint::black_box;
    use std::time::Instant;

    use ngspice_rs::maths::linear::SolveBatchWorkspace;
    use ngspice_rs::maths::{SparseLu, SparseMatrix, Vector};

    let size = 512_usize;
    for (name, offsets) in [
        ("tridiagonal", &[1_usize][..]),
        ("two_band", &[1, 16][..]),
        ("wide_band", &[1, 4, 16, 64][..]),
    ] {
        let mut matrix = SparseMatrix::new(size, size);
        for row in 0..size {
            matrix.add(row, row, 8.).unwrap();
            for &offset in offsets {
                if let Some(col) = row.checked_sub(offset) {
                    matrix.add(row, col, -0.75).unwrap();
                }
                if row + offset < size {
                    matrix.add(row, row + offset, -0.5).unwrap();
                }
            }
        }
        matrix.fold_duplicates();
        let lu = SparseLu::new(&matrix, None).unwrap();
        let rhs: Vec<_> = (0..size * 64)
            .map(|i| f64::from(u32::try_from((i * 13 + 7) % 127).unwrap()) / 64. - 1.)
            .collect();
        let mut work = SolveBatchWorkspace::default();
        let mut scalar_work = Vec::new();
        let mut out = Vec::new();
        println!(
            "workload={name},factor_entries={}",
            lu.factor_storage().factor_entries
        );
        for transposed in [false, true] {
            for count in [1, 2, 4, 8, 16, 32, 64] {
                let rhs = &rhs[..count * size];
                lu.solve_batch_into(rhs, &mut work, &mut out, transposed)
                    .unwrap();
                for (input, output) in rhs.chunks_exact(size).zip(out.chunks_exact(size)) {
                    let input = Vector::from(input.to_vec());
                    let expected = if transposed {
                        lu.solve_transposed_with_workspace(&input, &mut scalar_work)
                    } else {
                        lu.solve_with_workspace(&input, &mut scalar_work)
                    }
                    .unwrap();
                    assert_eq!(output, expected.as_slice());
                }
                let mut times = [Duration::ZERO; 5];
                for elapsed in &mut times {
                    let start = Instant::now();
                    for _ in 0..16 {
                        lu.solve_batch_into(black_box(rhs), &mut work, &mut out, transposed)
                            .unwrap();
                        black_box(&out);
                    }
                    *elapsed = start.elapsed();
                }
                times.sort_unstable();
                println!(
                    "workload={name},transpose={transposed},rhs={count},batch_ns={}",
                    times[2].as_nanos()
                );
            }
        }
    }
}
