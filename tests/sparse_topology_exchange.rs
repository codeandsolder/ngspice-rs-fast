//! Public real/complex solves of sparse structures requiring row/column
//! exchanges, against independently assembled dense matrix products.

use ngspice_rs::maths::complex::ComplexMatrix;
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::primitives::Complex;

// The exact accepted pre-refactor solver (base a0b63a291bbb) reaches
// 4.573790130098836e-10 forward error on the shifted 23x23 complex
// transpose case. Both algorithms produce identical values for every
// configuration tested. Keep this independent-algebra check tighter than
// 1e-9 without asserting a numerical accuracy guarantee the engine does not
// yet provide; structural permutation and handle fidelity are exact above.
const ACCEPTED_SOLVER_FORWARD_ERROR: f64 = 1e-9;

#[test]
fn permuted_sparse_real_and_complex_solves_match_direct_products() {
    for n in [2, 3, 5, 7, 13, 23] {
        for shift in [1, 2, n - 1] {
            let shift = shift % n;
            if shift == 0 {
                continue;
            }
            let mut real = SparseMatrix::new(n, n);
            let mut imaginary = SparseMatrix::new(n, n);
            let mut dense = vec![vec![Complex::ZERO; n]; n];
            for (row, dense_row) in dense.iter_mut().enumerate() {
                for (col, dense_cell) in dense_row.iter_mut().enumerate() {
                    let main = (row + shift) % n == col;
                    let secondary = (row * 11 + col * 7 + shift) % 5 == 0;
                    if main || secondary {
                        let value = if main {
                            10. + n as f64
                        } else {
                            0.125 * f64::from(((row + col) % 3) as u8 + 1)
                        };
                        // Duplicate triplets must sum in their original
                        // insertion order, not change stable source handles.
                        real.add(row, col, value * 0.75).unwrap();
                        real.add(row, col, value * 0.25).unwrap();
                        dense_cell.re += value * 0.75 + value * 0.25;
                    }
                    if main || (row + col) % 4 == 0 {
                        let value = if main { 0.05 } else { 0.007 };
                        imaginary.add(row, col, value).unwrap();
                        dense_cell.im += value;
                    }
                }
            }
            let truth: Vec<_> = (0..n)
                .map(|i| (f64::from((i % 7) as u8) - 2.3) * 0.5)
                .collect();
            let mut b_real = vec![0.; n];
            let mut bt_real = vec![0.; n];
            let mut b_complex = vec![Complex::ZERO; n];
            let mut bt_complex = vec![Complex::ZERO; n];
            for row in 0..n {
                for col in 0..n {
                    let x = truth[col];
                    b_real[row] += dense[row][col].re * x;
                    bt_real[col] += dense[row][col].re * truth[row];
                    b_complex[row] = b_complex[row] + dense[row][col] * Complex::real(x);
                    bt_complex[col] = bt_complex[col] + dense[row][col] * Complex::real(truth[row]);
                }
            }
            let lu_real = real.factorize().unwrap();
            let got = lu_real.solve(&Vector::from_slice(&b_real)).unwrap();
            let transposed = lu_real
                .solve_transposed(&Vector::from_slice(&bt_real))
                .unwrap();
            for (i, expected) in truth.iter().copied().enumerate() {
                assert!(
                    (got.as_slice()[i] - expected).abs() <= ACCEPTED_SOLVER_FORWARD_ERROR,
                    "real n={n} shift={shift} i={i}"
                );
                assert!(
                    (transposed.as_slice()[i] - expected).abs() <= ACCEPTED_SOLVER_FORWARD_ERROR,
                    "transpose n={n} shift={shift} i={i}"
                );
            }
            let lu_complex = ComplexMatrix::from_operators(&real, &imaginary, 1.)
                .unwrap()
                .factorize()
                .unwrap();
            let got = lu_complex.solve(&b_complex).unwrap();
            let transposed = lu_complex.solve_transposed(&bt_complex).unwrap();
            for (i, expected) in truth.iter().copied().enumerate() {
                assert!(
                    (got[i] - Complex::real(expected)).magnitude() <= ACCEPTED_SOLVER_FORWARD_ERROR,
                    "complex n={n} shift={shift} i={i}"
                );
                assert!(
                    (transposed[i] - Complex::real(expected)).magnitude()
                        <= ACCEPTED_SOLVER_FORWARD_ERROR,
                    "complex transpose n={n} shift={shift} i={i}: got={:?} expected={expected} error={}",
                    transposed[i],
                    (transposed[i] - Complex::real(expected)).magnitude()
                );
            }
        }
    }
}

#[test]
fn structural_and_rank_deficiency_remain_explicit_errors() {
    for n in [1, 2, 5, 11] {
        let mut empty = SparseMatrix::new(n, n);
        assert!(empty.factorize().is_err(), "empty {n}x{n}");
        if n == 1 {
            continue;
        }
        for row in 0..n {
            empty.add(row, 0, 1.).unwrap();
        }
        assert!(empty.factorize().is_err(), "rank-deficient {n}x{n}");
        let mut complex = SparseMatrix::new(n, n);
        if n > 1 {
            for row in 0..n {
                complex.add(row, 0, 0.5).unwrap();
            }
        }
        assert!(
            ComplexMatrix::from_operators(&empty, &complex, 1.)
                .unwrap()
                .factorize()
                .is_err(),
            "singular complex matrix {n}x{n}",
        );
    }
}

#[test]
#[ignore = "manual repeated sparse ordering throughput profile"]
fn benchmark_repeat_permuted_factorization() {
    use std::hint::black_box;
    use std::time::Instant;

    let n = 128;
    let mut matrix = SparseMatrix::new(n, n);
    for row in 0..n {
        for col in 0..n {
            if (row + 1) % n == col || (row * 13 + col * 7) % 17 == 0 {
                let value = if (row + 1) % n == col { 12. } else { 0.01 };
                matrix.add(row, col, value).unwrap();
            }
        }
    }
    let start = Instant::now();
    for _ in 0..100 {
        black_box(matrix.factorize().unwrap());
    }
    println!(
        "n={n}, repetitions=100, elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1e3
    );
}
