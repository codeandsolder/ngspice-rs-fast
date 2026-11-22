//! Deterministic real dense kernel workload; run ignored tests serially.
use ngspice_rs::maths::{Matrix, Vector};
use std::hint::black_box;
use std::time::Instant;

fn matrix(rows: usize, cols: usize) -> Matrix {
    let mut matrix = Matrix::zeros(rows, cols).unwrap();
    for (index, slot) in matrix.data_mut().iter_mut().enumerate() {
        *slot = f64::from(u32::try_from(index.wrapping_mul(13) % 127).unwrap()) / 64. - 1.;
    }
    matrix
}

fn vector(len: usize) -> Vector {
    Vector::from(
        (0..len)
            .map(|i| f64::from(u32::try_from(i.wrapping_mul(17) % 101).unwrap()) / 32. - 1.)
            .collect::<Vec<_>>(),
    )
}

// Dense normalized coupling with a known spectrum. Off-diagonal radius is
// larger than one for n >= 8, so plain diagonal dominance cannot certify it.
fn mutual_matrix(n: usize) -> Matrix {
    let mut matrix = Matrix::zeros(n, n).unwrap();
    matrix.fill(0.25);
    for i in 0..n {
        matrix.set(i, i, 1.).unwrap();
    }
    matrix
}

fn timing(name: &str, rows: usize, cols: usize, repeats: usize, mut run: impl FnMut()) {
    for _ in 0..3 {
        run();
    }
    for sample in 0..5 {
        let start = Instant::now();
        for _ in 0..repeats {
            run();
        }
        let seconds = start.elapsed().as_secs_f64();
        println!(
            "kernel={name},rows={rows},cols={cols},sample={sample},repeats={repeats},seconds={seconds:.9}"
        );
    }
}

#[test]
#[ignore = "paired dense-kernel throughput measurement"]
fn dense_matvec() {
    for (rows, cols) in [
        (4, 4),
        (16, 16),
        (64, 64),
        (256, 256),
        (1024, 1024),
        (4096, 7),
        (7, 4096),
        (33, 65),
    ] {
        let a = matrix(rows, cols);
        let x = vector(cols);
        let y = a.mul_vector(&x).unwrap();
        for (row, actual) in a.data().chunks_exact(cols).zip(y.as_slice()) {
            let expected = row
                .iter()
                .zip(x.as_slice())
                .fold(0., |acc, (a, b)| acc + a * b);
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        let repeats = (12_000_000 / (rows * cols)).clamp(12, 100_000);
        timing("matvec", rows, cols, repeats, || {
            black_box(black_box(&a).mul_vector(black_box(&x)).unwrap());
        });
        let mut output = Vector::zeros(rows);
        a.mul_vector_into(&x, &mut output).unwrap();
        assert_eq!(output, y);
        timing("matvec_reused", rows, cols, repeats, || {
            black_box(&a)
                .mul_vector_into(black_box(&x), black_box(&mut output))
                .unwrap();
            black_box(output.as_slice());
        });
    }
}

#[test]
#[ignore = "paired dense-kernel throughput measurement"]
fn dense_dot() {
    for len in [4, 16, 64, 256, 4096, 65_536] {
        let a = vector(len);
        let b = vector(len);
        let expected: f64 = a
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(a, b)| a * b)
            .sum();
        assert_eq!(a.dot(&b).unwrap().to_bits(), expected.to_bits());
        let repeats = (16_000_000 / len).clamp(64, 1_000_000);
        timing("dot", 1, len, repeats, || {
            black_box(black_box(&a).dot(black_box(&b)).unwrap());
        });
    }
}

#[test]
#[ignore = "paired dense-kernel throughput measurement"]
fn dense_transpose() {
    for (rows, cols) in [
        (4, 4),
        (16, 16),
        (64, 64),
        (256, 256),
        (1024, 1024),
        (8192, 64),
        (64, 8192),
        (4096, 7),
        (7, 4096),
        (511, 1025),
    ] {
        let a = matrix(rows, cols);
        let b = a.transpose().unwrap();
        assert_eq!(b.transpose().unwrap(), a);
        let repeats = (12_000_000 / (rows * cols)).clamp(12, 100_000);
        timing("transpose", rows, cols, repeats, || {
            black_box(black_box(&a).transpose().unwrap());
        });
        let mut output = Matrix::zeros(cols, rows).unwrap();
        a.transpose_into(&mut output).unwrap();
        assert_eq!(output, b);
        timing("transpose_reused", rows, cols, repeats, || {
            black_box(&a)
                .transpose_into(black_box(&mut output))
                .unwrap();
            black_box(output.data());
        });
    }
}

#[test]
#[ignore = "paired dense-kernel throughput measurement"]
fn dense_mutual_eigenvalues() {
    for n in [4, 8, 16, 32, 64, 128] {
        let a = mutual_matrix(n);
        let values = a.symmetric_eigenvalues().unwrap();
        let largest = 0.75 + f64::from(u32::try_from(n).unwrap()) * 0.25;
        let error = 16. * f64::EPSILON * largest;
        assert!((values[n - 1] - largest).abs() <= error);
        assert!(values[..n - 1].iter().all(|v| (*v - 0.75).abs() <= error));
        let repeats = (250_000 / (n * n * n)).clamp(4, 2_000);
        timing("mutual_eigenvalues", n, n, repeats, || {
            black_box(black_box(&a).symmetric_eigenvalues().unwrap());
        });
    }
}

#[test]
#[ignore = "bounded DHAT allocation probe"]
fn dense_allocations() {
    let a = matrix(64, 64);
    let x = vector(64);
    let eigen = mutual_matrix(16);
    for _ in 0..128 {
        black_box(black_box(&a).mul_vector(black_box(&x)).unwrap());
        black_box(black_box(&a).transpose().unwrap());
        black_box(black_box(&eigen).symmetric_eigenvalues().unwrap());
    }
}

#[test]
#[ignore = "bounded DHAT reuse allocation probe"]
fn dense_reused_allocations() {
    let a = matrix(64, 64);
    let x = vector(64);
    let mut product = Vector::zeros(64);
    let mut transposed = Matrix::zeros(64, 64).unwrap();
    let eigen = mutual_matrix(16);
    for _ in 0..128 {
        black_box(&a)
            .mul_vector_into(black_box(&x), black_box(&mut product))
            .unwrap();
        black_box(&a)
            .transpose_into(black_box(&mut transposed))
            .unwrap();
        black_box(black_box(&eigen).symmetric_eigenvalues().unwrap());
    }
}

#[test]
fn ordered_matvec_matches_scalar_across_row_groups_and_extreme_values() {
    // Include cancellation, individually overflowing products, signed zero,
    // subnormals, and nonfinite inputs. Preserve the scalar IEEE result
    // (NaN payload selection is not a portable arithmetic contract).
    let small = f64::from_bits(1);
    let cases = [
        (vec![1e300, 1., -1e300, small, -small, 2., -0.], vec![1.; 7]),
        (
            vec![f64::MAX, -f64::MAX, 1., 0., f64::MIN_POSITIVE],
            vec![2., 2., 1., 1., 0.5],
        ),
        (vec![-0., 0., -0., 0.], vec![1., -1., -1., 1.]),
        (vec![f64::INFINITY, 2., f64::NEG_INFINITY], vec![1., 0., 1.]),
        (vec![1., f64::NAN, 2., 3., 4., 5., 6., 7., 8.], vec![1.; 9]),
    ];
    for (coefficients, values) in cases {
        let cols = values.len();
        for rows in [1, 3, 4, 5, 7, 8, 9, 17] {
            let mut a = Matrix::zeros(rows, cols).unwrap();
            for (row_index, row) in a.data_mut().chunks_exact_mut(cols).enumerate() {
                for (col, entry) in row.iter_mut().enumerate() {
                    *entry = coefficients[(row_index + col) % cols];
                }
            }
            let x = Vector::from_slice(&values);
            let actual = a.mul_vector(&x).unwrap();
            let mut reused = Vector::from(vec![f64::NAN; rows]);
            a.mul_vector_into(&x, &mut reused).unwrap();
            for ((row, &allocated), &reused) in a
                .data()
                .chunks_exact(cols)
                .zip(actual.as_slice())
                .zip(reused.as_slice())
            {
                let expected = row.iter().zip(&values).fold(0., |sum, (a, b)| sum + a * b);
                if expected.is_nan() {
                    assert!(allocated.is_nan() && reused.is_nan());
                } else {
                    assert_eq!(allocated.to_bits(), expected.to_bits());
                    assert_eq!(reused.to_bits(), expected.to_bits());
                }
            }
        }
    }
    let empty = Vector::zeros(0);
    let scalar_empty = std::iter::empty::<f64>().sum::<f64>();
    assert_eq!(empty.dot(&empty).unwrap().to_bits(), scalar_empty.to_bits());
}

#[test]
fn reused_output_checks_dimensions_before_mutating() {
    let a = matrix(5, 7);
    let x = vector(7);
    let mut output = Vector::from(vec![42.; 5]);
    assert!(a.mul_vector_into(&vector(6), &mut output).is_err());
    assert_eq!(output.as_slice(), &[42.; 5]);
    let mut wrong_length = Vector::from(vec![42.; 6]);
    assert!(a.mul_vector_into(&x, &mut wrong_length).is_err());
    assert_eq!(wrong_length.as_slice(), &[42.; 6]);
    let mut wrong_shape = matrix(5, 7);
    let before = wrong_shape.clone();
    assert!(a.transpose_into(&mut wrong_shape).is_err());
    assert_eq!(wrong_shape, before);

    let zero_columns = Matrix::zeros(5, 0).unwrap();
    zero_columns
        .mul_vector_into(&Vector::zeros(0), &mut output)
        .unwrap();
    assert_eq!(output, Vector::zeros(5));
    let extreme = Matrix::zeros(usize::MAX, 0).unwrap();
    let mut transposed = Matrix::zeros(0, usize::MAX).unwrap();
    extreme.transpose_into(&mut transposed).unwrap();
    let mut roundtrip = Matrix::zeros(usize::MAX, 0).unwrap();
    transposed.transpose_into(&mut roundtrip).unwrap();
    assert_eq!(roundtrip, extreme);
    assert!(
        extreme
            .mul_vector_into(&Vector::zeros(0), &mut Vector::zeros(0))
            .is_err()
    );
    Matrix::zeros(0, 7)
        .unwrap()
        .mul_vector_into(&x, &mut Vector::zeros(0))
        .unwrap();
}

#[test]
fn transposition_preserves_every_bit_across_tiles() {
    for (rows, cols) in [
        (1, 65),
        (65, 1),
        (3, 5),
        (31, 33),
        (32, 32),
        (33, 65),
        (65, 33),
        (97, 129),
    ] {
        let mut a = Matrix::zeros(rows, cols).unwrap();
        for (i, slot) in a.data_mut().iter_mut().enumerate() {
            *slot = match i % 7 {
                0 => -0.,
                1 => f64::INFINITY,
                2 => f64::NEG_INFINITY,
                3 => f64::from_bits(0x7ff8_0000_0000_0000 | u64::try_from(i).unwrap()),
                _ => f64::from_bits(u64::try_from(i).unwrap()),
            };
        }
        let t = a.transpose().unwrap();
        let mut into = Matrix::zeros(cols, rows).unwrap();
        a.transpose_into(&mut into).unwrap();
        for row in 0..rows {
            for col in 0..cols {
                let expected = a.get(row, col).unwrap().to_bits();
                assert_eq!(t.get(col, row).unwrap().to_bits(), expected);
                assert_eq!(into.get(col, row).unwrap().to_bits(), expected);
            }
        }
    }
}

#[test]
fn jacobi_recovers_nonuniform_known_spectrum_at_extreme_scales() {
    // Hadamard columns are mutually orthogonal, with squared norm n.
    // Construct H diag(lambda) H^T/n directly with binary-exact coefficients.
    // The prescribed spectrum includes both closely spaced values and an
    // indefinite mode; the oracle does not reuse the Jacobi rotation.
    // Larger nonuniform spectra expose the pre-existing convergence defect #148.
    for n in [4_usize, 8, 16, 32] {
        let order = f64::from(u32::try_from(n).unwrap());
        let mut expected: Vec<_> = (0..n)
            .map(|i| 1. + f64::from(u32::try_from((i * 13) % n).unwrap()) / 64.)
            .collect();
        expected[0] = -0.125;
        let mut a = Matrix::zeros(n, n).unwrap();
        for row in 0..n {
            for col in 0..n {
                let value = expected
                    .iter()
                    .enumerate()
                    .map(|(k, &lambda)| {
                        if ((row ^ col) & k).count_ones() % 2 == 0 {
                            lambda
                        } else {
                            -lambda
                        }
                    })
                    .sum::<f64>()
                    / order;
                a.set(row, col, value).unwrap();
            }
        }
        expected.sort_unstable_by(f64::total_cmp);
        let tolerance = 64. * f64::EPSILON * order * expected[n - 1];
        for scale in [2_f64.powi(-600), 1., 2_f64.powi(600)] {
            let mut scaled = a.clone();
            for value in scaled.data_mut() {
                *value *= scale;
            }
            let actual = scaled.symmetric_eigenvalues().unwrap();
            assert!(
                actual
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| ((a / scale) - b).abs() <= tolerance),
                "n={n}, scale={scale:e}"
            );
        }
    }
}
