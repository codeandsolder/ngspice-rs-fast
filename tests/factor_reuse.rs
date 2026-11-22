//! `solve_ni` reuses the factors of a bound matrix whose values, `diag_gmin`
//! and pivot thresholds equal those it last refactored (SPARSE's `spFactor`
//! of identical inputs with the same order recomputes them bit for bit).
//! These tests pin the reuse to C's refactorization and every invalidation:
//! a changed value, a changed `diag_gmin`, a requested reorder and a complex
//! (AC) factorization in the shared frame.
use ngspice_rs::maths::complex::ComplexMatrix;
use ngspice_rs::maths::{NiStep, Pattern, SparseMatrix, Vector, solve_ni};
use ngspice_rs::primitives::Complex;
use std::sync::Arc;

/// A diagonally dominant, nonsymmetric 3x3 pattern (the pivot order stays
/// on the diagonal for every value set below).
const POSITIONS: [(usize, usize); 7] = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)];
const A: [f64; 7] = [4.0, -1.25, -0.5, 3.0, -0.75, -1.5, 5.0];
const B: [f64; 7] = [4.5, -1.25, -0.5, 3.0, -0.75, -1.5, 5.0];

fn pattern() -> Arc<Pattern> {
    Arc::new(Pattern::new(3, POSITIONS).unwrap())
}

fn bound(pattern: &Arc<Pattern>, values: &[f64; 7]) -> SparseMatrix {
    let mut matrix = SparseMatrix::bound(pattern.clone()).unwrap();
    for (&(row, col), &value) in POSITIONS.iter().zip(values) {
        matrix.add(row, col, value).unwrap();
    }
    matrix
}

fn rhs() -> Vector {
    Vector::from_slice(&[1.0, -2.0, 0.5])
}

fn solve(pattern: &Arc<Pattern>, values: &[f64; 7], reorder: bool, gmin: f64) -> Vec<f64> {
    match solve_ni(&bound(pattern, values), &rhs(), &mut None, reorder, gmin).unwrap() {
        NiStep::Solved(x) => x.as_slice().to_vec(),
        NiStep::ZeroPivot => panic!("zero pivot"),
    }
}

/// `(A + gmin I) x - b`, which must vanish for the right factors.
fn residual(values: &[f64; 7], gmin: f64, x: &[f64]) -> f64 {
    let mut r = rhs().as_slice().to_vec();
    for v in &mut r {
        *v = -*v;
    }
    for (&(row, col), &value) in POSITIONS.iter().zip(values) {
        r[row] += value * x[col];
    }
    for (row, value) in r.iter_mut().enumerate() {
        *value += gmin * x[row];
    }
    r.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn identical_inputs_reuse_factors_equal_to_a_refactorization() {
    // Reused: the second solve of A with the factors of the first
    // refactorization of A.
    let reused = pattern();
    solve(&reused, &B, false, 0.0);
    let first = solve(&reused, &A, false, 0.0);
    let second = solve(&reused, &A, false, 0.0);
    // Refactored: the same order, then A refactored from B's frame.
    let refactored = pattern();
    solve(&refactored, &B, false, 0.0);
    let reference = solve(&refactored, &A, false, 0.0);
    assert_eq!(bits(&first), bits(&reference));
    assert_eq!(bits(&second), bits(&reference));
    assert!(residual(&A, 0.0, &second) < 1e-15);
}

#[test]
fn a_changed_value_refactors() {
    let p = pattern();
    solve(&p, &A, false, 0.0);
    solve(&p, &A, false, 0.0);
    let x = solve(&p, &B, false, 0.0);
    assert!(residual(&B, 0.0, &x) < 1e-15, "{x:?}");
    assert!(residual(&A, 0.0, &x) > 1e-3);
}

#[test]
fn a_changed_diagonal_gmin_refactors() {
    let p = pattern();
    solve(&p, &A, false, 0.0);
    solve(&p, &A, false, 0.0);
    let x = solve(&p, &A, false, 0.25);
    assert!(residual(&A, 0.25, &x) < 1e-15, "{x:?}");
    assert!(residual(&A, 0.0, &x) > 1e-3);
}

#[test]
fn a_requested_reorder_factors_again() {
    let p = pattern();
    solve(&p, &A, false, 0.0);
    let before = solve(&p, &A, false, 0.0);
    let reordered = solve(&p, &A, true, 0.0);
    assert!(residual(&A, 0.0, &reordered) < 1e-15);
    // The order is unchanged here, so so is the solution.
    assert_eq!(bits(&before), bits(&reordered));
}

#[test]
fn a_complex_factorization_in_the_shared_frame_invalidates_the_factors() {
    let p = pattern();
    solve(&p, &A, false, 0.0);
    solve(&p, &A, false, 0.0);
    // An AC factorization of A + j omega E on the same pattern (the
    // circuit's one SPARSE frame) overwrites the frame's factors.
    let a = bound(&p, &A);
    let mut e = SparseMatrix::new(3, 3);
    for row in 0..3 {
        e.add(row, row, 1e-3).unwrap();
    }
    let ac = ComplexMatrix::from_operators_on(&p, &a, &e, 2.0e3)
        .unwrap()
        .expect("stamps inside the pattern");
    let lu = ac.factorize().unwrap();
    let z = lu
        .solve(&[Complex::real(1.0), Complex::real(-2.0), Complex::real(0.5)])
        .unwrap();
    assert!(z.iter().any(|v| v.im != 0.0));
    let x = solve(&p, &A, false, 0.0);
    assert!(residual(&A, 0.0, &x) < 1e-15, "{x:?}");
}

#[test]
fn measured_cache_work_distinguishes_order_refactor_exact_hit_and_retained_pivots() {
    use ngspice_rs::maths::{FactorWork, PatternTelemetry, solve_last_factors};

    let p = pattern();
    assert_eq!(p.factor_telemetry(), PatternTelemetry::default());
    let initial = solve(&p, &A, false, 0.);
    let first = p.factor_telemetry();
    assert_eq!(
        first.work.factor,
        FactorWork {
            order_attempts: 1,
            pivot_searches: 1,
            successful_factors: 1,
            ..Default::default()
        }
    );
    let storage = first.storage.unwrap();
    assert_eq!(storage.original_entries, POSITIONS.len());
    assert_eq!(storage.external_order, 3);
    assert_eq!(storage.translated_order, 3);
    assert_eq!(
        storage.numeric_bytes,
        storage.factor_entries * size_of::<f64>()
    );
    assert_eq!(first.work.rhs_solves, 1);

    let changed = solve(&p, &B, false, 0.);
    let reused = solve(&p, &B, false, 0.);
    assert_eq!(bits(&changed), bits(&reused));
    let after_hit = p.factor_telemetry();
    assert_eq!(after_hit.work.factor.fixed_refactors, 1);
    assert_eq!(after_hit.work.factor.successful_factors, 2);
    assert_eq!(after_hit.work.exact_input_hits, 1);
    assert_eq!(after_hit.work.rhs_solves, 3);

    let reordered = solve(&p, &A, true, 0.);
    let after_order = p.factor_telemetry();
    assert_eq!(bits(&reordered), bits(&initial));
    assert_eq!(after_order.work.factor.order_attempts, 2);
    assert_eq!(after_order.work.factor.pivot_searches, 1);
    assert_eq!(after_order.work.factor.successful_factors, 3);
    assert_eq!(after_order.work.rhs_solves, 4);
    assert_eq!(after_order.storage, Some(storage));

    let retained = solve_last_factors(&p, &rhs()).unwrap().unwrap();
    assert_eq!(bits(retained.as_slice()), bits(&reordered));
    assert!(solve_last_factors(&p, &Vector::zeros(1)).is_err());
    let after_retained = p.factor_telemetry();
    assert_eq!(after_retained.work.rhs_solves, 5);
    assert_eq!(after_retained.work.rhs_failures, 1);
    assert_eq!(after_retained.work.factor, after_order.work.factor);
    assert_eq!(
        p.as_ref().clone().factor_telemetry(),
        PatternTelemetry::default()
    );
    assert_eq!(
        p.extended([(0, 2)]).unwrap().factor_telemetry(),
        PatternTelemetry::default()
    );
}

fn two_by_two(p: &Arc<Pattern>, values: [f64; 4]) -> SparseMatrix {
    let mut m = SparseMatrix::bound(Arc::clone(p)).unwrap();
    for ((row, col), value) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().zip(values) {
        m.set(row, col, value).unwrap();
    }
    m
}

#[test]
fn zero_pivot_search_and_failed_factors_are_measured_not_counted_as_success() {
    let p = Arc::new(Pattern::new(2, [(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap());
    let rhs = Vector::from_slice(&[2., 1.]);
    let initial = two_by_two(&p, [4., 1., 1., 5.]);
    let changed = two_by_two(&p, [0., 2., 1., 0.]);
    let mut old = None;
    assert!(matches!(
        solve_ni(&initial, &rhs, &mut old, true, 0.).unwrap(),
        NiStep::Solved(_)
    ));
    assert!(matches!(
        solve_ni(&changed, &rhs, &mut old, false, 0.).unwrap(),
        NiStep::ZeroPivot
    ));
    let failed = p.factor_telemetry();
    assert_eq!(failed.work.factor.successful_factors, 1);
    assert_eq!(failed.work.factor.zero_pivots, 1);
    assert_eq!(failed.work.factor.fixed_refactors, 1);
    assert_eq!(failed.work.rhs_solves, 1);
    assert!(failed.storage.is_none());

    let NiStep::Solved(x) = solve_ni(&changed, &rhs, &mut old, true, 0.).unwrap() else {
        panic!("threshold reorder did not recover the nonsingular matrix");
    };
    assert_eq!(x.as_slice(), [1., 1.]);
    let recovered = p.factor_telemetry();
    assert_eq!(recovered.work.factor.order_attempts, 2);
    assert_eq!(recovered.work.factor.pivot_searches, 2);
    assert_eq!(recovered.work.factor.successful_factors, 2);
    assert_eq!(recovered.storage.unwrap().original_entries, 4);
    assert_eq!(changed.nnz(), 2);

    let singular = two_by_two(&p, [0.; 4]);
    assert!(solve_ni(&singular, &rhs, &mut old, true, 0.).is_err());
    let failed = p.factor_telemetry();
    assert_eq!(failed.work.factor.successful_factors, 2);
    assert_eq!(failed.work.factor.singular_failures, 1);
    assert!(failed.storage.is_none());

    let nonfinite = two_by_two(&p, [f64::NAN, 1., 1., 1.]);
    assert!(solve_ni(&nonfinite, &rhs, &mut old, true, 0.).is_err());
    let rejected = p.factor_telemetry();
    assert_eq!(rejected.work.factor.assembly_rejections, 1);
    assert_eq!(rejected.work.factor.successful_factors, 2);
    assert!(rejected.storage.is_none());

    let tiny = two_by_two(&p, [f64::from_bits(1), 0., 0., 1.]);
    assert!(solve_ni(&tiny, &rhs, &mut old, true, 0.).is_err());
    let nonfinite = p.factor_telemetry();
    assert_eq!(nonfinite.work.factor.nonfinite_failures, 1);
    assert_eq!(nonfinite.work.factor.successful_factors, 2);
    assert!(nonfinite.storage.is_none());
}

#[test]
fn retained_factors_report_their_own_work_and_known_fill() {
    use ngspice_rs::maths::{FactorWork, SparseLu};

    let p = Arc::new(Pattern::new(2, [(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap());
    let original = two_by_two(&p, [4., 1., 1., 5.]);
    let changed = two_by_two(&p, [0., 2., 1., 0.]);
    let a = SparseLu::new(&original, None).unwrap();
    let before = a.solve(&Vector::from_slice(&[5., 6.])).unwrap();
    let b = SparseLu::new(&changed, Some(a.symbolic())).unwrap();
    assert_eq!(
        b.factor_work(),
        FactorWork {
            fixed_refactors: 1,
            order_attempts: 1,
            pivot_searches: 1,
            successful_factors: 1,
            zero_pivots: 1,
            zero_pivot_retries: 1,
            ..Default::default()
        }
    );
    assert_eq!(a.factor_work().order_attempts, 1);
    assert_eq!(a.factor_work().zero_pivots, 0);
    assert_eq!(a.solve(&Vector::from_slice(&[5., 6.])).unwrap(), before);
    assert_eq!(
        b.solve_transposed(&Vector::from_slice(&[1., 2.]))
            .unwrap()
            .as_slice(),
        [1., 1.]
    );
    assert_eq!(p.factor_telemetry().work.factor, FactorWork::default());

    // Elimination of any vertex of the four-cycle adds both directed
    // links between its neighbors. A tree or clique would not test fill.
    let mut cycle = SparseMatrix::new(4, 4);
    for i in 0..4 {
        cycle.add(i, i, 4.).unwrap();
        cycle.add(i, (i + 1) % 4, -1.).unwrap();
        cycle.add((i + 1) % 4, i, -1.).unwrap();
    }
    let lu = SparseLu::new(&cycle, None).unwrap();
    let storage = lu.factor_storage();
    assert_eq!(
        (
            storage.original_entries,
            storage.factor_entries,
            storage.fill_entries()
        ),
        (12, 14, 2)
    );
    assert_eq!(storage.numeric_bytes, 14 * size_of::<f64>());
}

#[test]
fn factor_storage_distinguishes_external_and_translated_order() {
    use ngspice_rs::maths::SparseLu;
    let pattern = Arc::new(Pattern::new(3, [(0, 0), (2, 2)]).unwrap());
    let mut matrix = SparseMatrix::bound(pattern).unwrap();
    matrix.set(0, 0, 2.).unwrap();
    matrix.set(2, 2, 4.).unwrap();
    let lu = SparseLu::new(&matrix, None).unwrap();
    let stats = lu.factor_storage();
    assert_eq!((stats.external_order, stats.translated_order), (3, 2));
    assert_eq!((stats.original_entries, stats.factor_entries), (2, 2));
    assert_eq!(
        lu.solve(&Vector::from_slice(&[2., 7.5, 4.]))
            .unwrap()
            .as_slice(),
        [1., 7.5, 1.]
    );
}

#[test]
fn shared_cache_telemetry_is_transactional_across_concurrent_solves() {
    let p = pattern();
    // Establish a clean refactor and exact-input eligibility before any
    // concurrent requests, so scheduling cannot alter the expected work.
    solve(&p, &A, false, 0.);
    let reference = solve(&p, &A, false, 0.);
    let start = p.factor_telemetry();
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let p = &p;
            let reference = &reference;
            scope.spawn(move || {
                for _ in 0..16 {
                    assert_eq!(bits(&solve(p, &A, false, 0.)), bits(reference));
                }
            });
        }
    });
    let after = p.factor_telemetry();
    assert_eq!(after.work.factor, start.work.factor);
    assert_eq!(
        after.work.exact_input_hits - start.work.exact_input_hits,
        64
    );
    assert_eq!(after.work.rhs_solves - start.work.rhs_solves, 64);
    assert_eq!(after.storage, start.storage);
}
