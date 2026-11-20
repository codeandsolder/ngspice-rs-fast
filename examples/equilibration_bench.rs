//! Reproducible acceptance/overhead probe; failures are counted, never filtered.
//! Run in release mode. Timing includes existing rank diagnostics; no claim
//! that row/column normalization repairs arbitrary near dependence is made.
use ngspice_rs::maths::{
    EquilibratedComplexLu, EquilibratedDenseLu, EquilibratedSparseLu, Matrix, SparseMatrix, Vector,
    complex::ComplexMatrix,
};
use ngspice_rs::primitives::{Complex, SpiceResult};
use std::{hint::black_box, time::Instant};

fn measure<T>(
    case: &str,
    backend: &str,
    repeats: u32,
    expected: &[f64],
    build: impl Fn() -> SpiceResult<T>,
    solve: impl Fn(&T) -> SpiceResult<Vec<f64>>,
) {
    // One untimed attempt for every variant, including rejected cases.
    if let Ok(factor) = build() {
        let _ = black_box(solve(&factor));
    }
    let mut factor_time = 0.0;
    let mut solve_time = 0.0;
    let mut factors_ok = 0;
    let mut solves_ok = 0;
    let mut failed = 0;
    let mut worst_error = None::<f64>;
    let mut first_failure = None;
    for _ in 0..repeats {
        let start = Instant::now();
        let factor = black_box(build());
        factor_time += start.elapsed().as_secs_f64();
        match factor {
            Err(e) => {
                failed += 1;
                first_failure.get_or_insert_with(|| e.to_string());
            }
            Ok(factor) => {
                factors_ok += 1;
                for _ in 0..4 {
                    let start = Instant::now();
                    let x = black_box(solve(&factor));
                    solve_time += start.elapsed().as_secs_f64();
                    match x {
                        Err(e) => {
                            failed += 1;
                            first_failure.get_or_insert_with(|| e.to_string());
                        }
                        Ok(x) => {
                            let error = x.iter().zip(expected).fold(0.0_f64, |m, (&x, &e)| {
                                m.max((x - e).abs() / e.abs().max(1e-24))
                            });
                            worst_error = Some(worst_error.unwrap_or(0.0).max(error));
                            if x.len() == expected.len() && error <= 1e-12 {
                                solves_ok += 1;
                            } else {
                                failed += 1;
                                first_failure.get_or_insert_with(|| {
                                    format!("analytic forward check failed: {error:e}")
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    println!(
        "{case},{backend},repeats={repeats},factor_ok={factors_ok},solve_ok={solves_ok},failures={failed},factor_mean_us={:.3},solve_mean_us={:.3},worst_relative_forward_error={},first_failure={}",
        factor_time * 1e6 / f64::from(repeats),
        if factors_ok == 0 {
            0.0
        } else {
            solve_time * 1e6 / f64::from(4 * factors_ok)
        },
        worst_error.map_or_else(|| "not-measured".into(), |v| format!("{v:.3e}")),
        first_failure.as_deref().unwrap_or("none")
    );
}

fn flatten(x: &[Complex]) -> Vec<f64> {
    x.iter().flat_map(|v| [v.re, v.im]).collect()
}

struct Operators {
    conductance: SparseMatrix,
    dense: Matrix,
    susceptance: SparseMatrix,
    columns: Vec<f64>,
}

impl Operators {
    fn new(size: usize, ill_scaled: bool, failure: Option<&str>) -> SpiceResult<Self> {
        let mut conductance = SparseMatrix::new(size, size);
        let mut dense = Matrix::zeros(size, size)?;
        let mut susceptance = SparseMatrix::new(size, size);
        let columns: Vec<_> = (0..size)
            .map(|column| {
                if ill_scaled {
                    2.0_f64.powi(if column % 2 == 0 { 40 } else { -40 })
                } else {
                    1.0
                }
            })
            .collect();
        for row_index in 0..size {
            let row_scale = if ill_scaled {
                2.0_f64.powi(if row_index % 2 == 0 { 20 } else { -20 })
            } else {
                1.0
            };
            for (column_index, &column_scale) in columns.iter().enumerate() {
                let coefficient = if row_index == column_index {
                    4.0
                } else if column_index + 1 == row_index {
                    -1.0
                } else if row_index + 1 == column_index {
                    0.5
                } else {
                    0.0
                };
                let value = row_scale * coefficient * column_scale;
                conductance.add(row_index, column_index, value)?;
                dense.set(row_index, column_index, value)?;
            }
            susceptance.add(row_index, row_index, row_scale * 0.25 * columns[row_index])?;
        }
        if let Some(kind) = failure {
            conductance.clear();
            dense.clear();
            susceptance.clear();
            let values = match kind {
                "singular" => [1.0, 1.0, 1.0, 1.0],
                "near-dependent" => [1.0, 1.0, 1.0, 1.0 + 1e-15],
                "underflow" => [1e300, 1e-300, 0.0, 1.0],
                _ => [1e308, 0.0, 0.0, 1e308],
            };
            for (index, &value) in values.iter().enumerate() {
                conductance.add(index / 2, index % 2, value)?;
                dense.set(index / 2, index % 2, value)?;
            }
        }
        Ok(Self {
            conductance,
            dense,
            susceptance,
            columns,
        })
    }
}

fn run(
    case: &str,
    size: usize,
    ill_scaled: bool,
    failure: Option<&str>,
    repeats: u32,
) -> SpiceResult<()> {
    let Operators {
        conductance,
        dense,
        susceptance,
        columns,
    } = Operators::new(size, ill_scaled, failure)?;
    let expected = Vector::from_slice(&columns.iter().map(|value| 1.0 / value).collect::<Vec<_>>());
    let real_rhs = conductance.mul_vector(&expected)?;
    measure(
        case,
        "dense-default",
        repeats,
        expected.as_slice(),
        || dense.lu_decompose(),
        |factor| Ok(factor.solve(&real_rhs)?.as_slice().to_vec()),
    );
    measure(
        case,
        "dense-equilibrated",
        repeats,
        expected.as_slice(),
        || EquilibratedDenseLu::new(&dense),
        |factor| Ok(factor.solve(&real_rhs)?.as_slice().to_vec()),
    );
    measure(
        case,
        "sparse-default",
        repeats,
        expected.as_slice(),
        || conductance.factorize(),
        |factor| Ok(factor.solve(&real_rhs)?.as_slice().to_vec()),
    );
    measure(
        case,
        "sparse-equilibrated",
        repeats,
        expected.as_slice(),
        || EquilibratedSparseLu::new(&conductance, None),
        |factor| Ok(factor.solve(&real_rhs)?.as_slice().to_vec()),
    );
    let solution: Vec<_> = expected
        .as_slice()
        .iter()
        .map(|&value| Complex::new(value, -0.5 * value))
        .collect();
    let mut rhs = vec![Complex::ZERO; size];
    for entry in conductance.triplets() {
        rhs[entry.row] = rhs[entry.row] + Complex::real(entry.value) * solution[entry.col];
    }
    for entry in susceptance.triplets() {
        rhs[entry.row] = rhs[entry.row] + Complex::imaginary(entry.value) * solution[entry.col];
    }
    let expected = flatten(&solution);
    measure(
        case,
        "complex-default",
        repeats,
        &expected,
        || ComplexMatrix::from_operators(&conductance, &susceptance, 1.0)?.factorize(),
        |factor| Ok(flatten(&factor.solve(&rhs)?)),
    );
    measure(
        case,
        "complex-equilibrated",
        repeats,
        &expected,
        || EquilibratedComplexLu::from_operators(&conductance, &susceptance, 1.0),
        |factor| Ok(flatten(&factor.solve(&rhs)?)),
    );
    Ok(())
}

fn main() -> SpiceResult<()> {
    println!(
        "One untimed warmup per variant. One row/column pass; factor timing includes assembly/scaling/rank diagnostics. Solve timing includes residual checks and allocation; no timing reported for unattempted solves (0). Four RHS solves per accepted factor."
    );
    println!(
        "Well-scaled real tridiagonal condition_inf <= 2.2 by diagonal dominance. Ill-scaled cases apply row ratio 2^40 and column ratio 2^80 to the same operator: unit imbalance, not arbitrary near dependence. No measured condition-number/forward-accuracy guarantee."
    );
    for (n, repeats) in [(8, 100), (128, 10)] {
        run(&format!("well-n{n}"), n, false, None, repeats)?;
        run(&format!("ill-n{n}"), n, true, None, repeats)?;
    }
    for failure in [
        "singular",
        "near-dependent",
        "underflow",
        "residual-overflow",
    ] {
        run(failure, 2, false, Some(failure), 10)?;
    }
    Ok(())
}
