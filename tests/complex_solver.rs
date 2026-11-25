//! Complex assembly and solver guardrails of the SPARSE 1.3 complex LU.
use ngspice_rs::maths::SparseMatrix;
use ngspice_rs::maths::complex::ComplexMatrix;
use ngspice_rs::primitives::Complex;
#[test]
fn complex_pivoting_snapshot_and_repeated_rhs() {
    let mut a = SparseMatrix::new(2, 2);
    a.add(0, 1, 1.).unwrap();
    a.add(1, 0, 2.).unwrap();
    let mut e = SparseMatrix::new(2, 2);
    e.add(1, 1, 1.).unwrap();
    let lu = ComplexMatrix::from_operators(&a, &e, 1.)
        .unwrap()
        .factorize()
        .unwrap();
    a.clear();
    e.clear();
    for rhs in [
        vec![Complex::real(2.), Complex::new(2., 2.)],
        vec![Complex::ZERO, Complex::real(4.)],
    ] {
        let x = lu.solve(&rhs).unwrap();
        assert!((x[1] - rhs[0]).magnitude() < 1e-14);
        assert!(
            (Complex::real(2.) * x[0] + Complex::imaginary(1.) * x[1] - rhs[1]).magnitude() < 1e-14
        );
    }
    assert!(lu.solve(&[Complex::ZERO]).is_err());
    assert!(lu.solve(&[Complex::real(f64::NAN), Complex::ZERO]).is_err());
}
#[test]
fn complex_assembly_nonfinite_overflow_and_singularity() {
    let mut a = SparseMatrix::new(1, 1);
    let mut e = SparseMatrix::new(1, 1);
    assert!(
        ComplexMatrix::from_operators(&a, &e, 0.)
            .unwrap()
            .factorize()
            .is_err()
    );
    a.add(0, 0, 1.).unwrap();
    e.add(0, 0, f64::MAX).unwrap();
    assert!(ComplexMatrix::from_operators(&a, &e, 2.).is_err());
    assert!(ComplexMatrix::from_operators(&a, &e, f64::NAN).is_err());
    e.clear();
    a.add(0, 0, f64::NAN).unwrap();
    assert!(ComplexMatrix::from_operators(&a, &e, 1.).is_err());
    assert!(
        ComplexMatrix::from_operators(&SparseMatrix::new(0, 0), &SparseMatrix::new(0, 0), 1.)
            .is_err()
    );
    assert!(ComplexMatrix::from_operators(&SparseMatrix::new(1, 2), &e, 1.).is_err());
}
#[test]
fn transposed_solves_use_the_same_factors_without_conjugation() {
    // A = [[1, 2], [0, 3]] + j [[0, 0], [1, 0]]: nonsymmetric and complex,
    // so A^T, A^H and A all differ.
    let mut a = SparseMatrix::new(2, 2);
    a.add(0, 0, 1.).unwrap();
    a.add(0, 1, 2.).unwrap();
    a.add(1, 1, 3.).unwrap();
    let mut e = SparseMatrix::new(2, 2);
    e.add(1, 0, 1.).unwrap();
    let lu = ComplexMatrix::from_operators(&a, &e, 1.)
        .unwrap()
        .factorize()
        .unwrap();
    let rhs = [Complex::real(1.), Complex::real(-1.)];
    let x = lu.solve_transposed(&rhs).unwrap();
    // A^T = [[1, j], [2, 3]].
    let row0 = x[0] + Complex::imaginary(1.) * x[1];
    let row1 = Complex::real(2.) * x[0] + Complex::real(3.) * x[1];
    assert!((row0 - rhs[0]).magnitude() < 1e-14, "{row0}");
    assert!((row1 - rhs[1]).magnitude() < 1e-14, "{row1}");
    // The ordinary solve still solves A x = b with the same factors.
    let y = lu.solve(&rhs).unwrap();
    let row0 = y[0] + Complex::real(2.) * y[1];
    let row1 = Complex::imaginary(1.) * y[0] + Complex::real(3.) * y[1];
    assert!((row0 - rhs[0]).magnitude() < 1e-14);
    assert!((row1 - rhs[1]).magnitude() < 1e-14);
    assert!(lu.solve_transposed(&[Complex::ZERO]).is_err());
    assert!(
        lu.solve_transposed(&[Complex::real(f64::INFINITY), Complex::ZERO])
            .is_err()
    );
}

#[test]
fn exactly_singular_homogeneous_systems_are_rejected() {
    // A duplicate row (rank one) and a zero row are errors with SPARSE
    // semantics; merely ill-conditioned systems are no longer rejected.
    for entries in [
        &[(0, 0, 1.), (0, 1, 1.), (1, 0, 1.), (1, 1, 1.)][..],
        &[(0, 0, 1.), (0, 1, 1.)][..],
    ] {
        let mut a = SparseMatrix::new(2, 2);
        for (r, c, v) in entries {
            a.add(*r, *c, *v).unwrap();
        }
        let result = ComplexMatrix::from_operators(&a, &SparseMatrix::new(2, 2), 0.)
            .unwrap()
            .factorize();
        assert!(result.is_err(), "singular system accepted: {entries:?}");
    }
}

#[test]
fn measured_complex_and_real_cache_transitions_preserve_snapshot_work() {
    use ngspice_rs::maths::{FactorWork, Pattern, SparseLu, Vector, solve_ni};
    use std::sync::Arc;
    let p = Arc::new(Pattern::new(2, [(0, 0), (0, 1), (1, 0), (1, 1)]).unwrap());
    let mut g = SparseMatrix::bound(Arc::clone(&p)).unwrap();
    for ((r, c), value) in [(0, 0), (0, 1), (1, 0), (1, 1)]
        .into_iter()
        .zip([4., 1., 1., 5.])
    {
        g.set(r, c, value).unwrap();
    }
    let mut e = SparseMatrix::new(2, 2);
    e.add(0, 0, 1.).unwrap();
    e.add(1, 1, 2.).unwrap();
    let r = Vector::from_slice(&[5., 6.]);
    solve_ni(
        &g,
        &r,
        &mut None,
        true,
        ngspice_rs::maths::DiagonalGmin::unrestricted(0.),
    )
    .unwrap();
    let ac = |g: &SparseMatrix, e: &SparseMatrix, omega| {
        ComplexMatrix::from_operators_on(&p, g, e, omega)
            .unwrap()
            .unwrap()
            .factorize()
            .unwrap()
    };
    let first = ac(&g, &e, 1.);
    let first_work = first.factor_work();
    let b = [Complex::new(5., 1.), Complex::new(6., 2.)];
    let before = first.solve(&b).unwrap();
    assert_eq!(p.factor_telemetry().work.real_to_complex, 1);
    assert_eq!(first_work.order_attempts, 1);
    assert_eq!(first_work.pivot_searches, 0);
    let second = ac(&g, &e, 2.);
    assert_eq!(
        second.factor_work(),
        FactorWork {
            fixed_refactors: 1,
            successful_factors: 1,
            ..Default::default()
        }
    );
    assert_eq!(first.factor_work(), first_work);
    assert_eq!(first.solve(&b).unwrap(), before);
    let s = first.factor_storage();
    assert_eq!(s.original_entries, 4);
    assert_eq!(
        s.numeric_bytes,
        (s.factor_entries + s.translated_order + 1) * size_of::<Complex>()
    );
    assert_eq!(p.factor_telemetry().storage, Some(second.factor_storage()));

    let empty = SparseMatrix::new(2, 2);
    g.set(0, 0, 0.).unwrap();
    g.set(1, 1, 0.).unwrap();
    let retry = ac(&g, &empty, 0.);
    assert_eq!(retry.factor_work().zero_pivots, 1);
    assert_eq!(retry.factor_work().zero_pivot_retries, 1);
    assert_eq!(retry.factor_work().pivot_searches, 1);
    assert_eq!(retry.factor_work().successful_factors, 1);
    assert_eq!(retry.factor_storage().original_entries, 4);
    assert_eq!(first.solve(&b).unwrap(), before);
    solve_ni(
        &g,
        &r,
        &mut None,
        false,
        ngspice_rs::maths::DiagonalGmin::unrestricted(0.),
    )
    .unwrap();
    let after = p.factor_telemetry();
    assert_eq!(after.work.complex_to_real, 1);
    assert_eq!(
        after.storage.unwrap().numeric_bytes,
        (retry.factor_storage().factor_entries + retry.factor_storage().translated_order + 1)
            * size_of::<f64>()
    );
    assert_eq!(after.work.factor.successful_factors, 5);
    assert_eq!(after.work.rhs_solves, 2);
    // An independent factor built on the same input is not a cache event.
    let independent = SparseLu::new(&g, None).unwrap();
    assert_eq!(independent.factor_work().successful_factors, 1);
    assert_eq!(p.factor_telemetry(), after);
}

#[test]
fn standalone_complex_retry_counts_only_the_producing_operation() {
    use ngspice_rs::maths::FactorWork;

    // Every 2x2 principal minor of the second matrix is zero but the
    // determinant is nonzero. Regardless of the initial diagonal pivot
    // order, a fixed-order refactor must meet a zero second diagonal.
    let matrix = |rows: [[f64; 3]; 3]| {
        let mut g = SparseMatrix::new(3, 3);
        for (r, row) in rows.into_iter().enumerate() {
            for (c, v) in row.into_iter().enumerate() {
                g.add(r, c, v).unwrap();
            }
        }
        ComplexMatrix::from_operators(&g, &SparseMatrix::new(3, 3), 0.).unwrap()
    };
    let first = matrix([[4., 1., 1.], [1., 5., 1.], [1., 1., 6.]])
        .factorize()
        .unwrap();
    let changed = matrix([[1., 1., 2.], [1., 1., 1.], [0.5, 1., 1.]]);
    let retried = changed.factorize_reusing(&first).unwrap();
    assert_eq!(
        retried.factor_work(),
        FactorWork {
            fixed_refactors: 1,
            order_attempts: 1,
            pivot_searches: 1,
            zero_pivots: 1,
            zero_pivot_retries: 1,
            successful_factors: 1,
            ..Default::default()
        }
    );
    assert_eq!(
        first.factor_work(),
        FactorWork {
            order_attempts: 1,
            pivot_searches: 1,
            successful_factors: 1,
            ..Default::default()
        }
    );
    for result in [
        retried.solve(&[4., 3., 2.5].map(Complex::real)).unwrap(),
        retried
            .solve_transposed(&[2.5, 3., 4.].map(Complex::real))
            .unwrap(),
    ] {
        assert!(
            result
                .iter()
                .all(|value| (*value - Complex::real(1.)).magnitude() < 1e-13)
        );
    }
}

#[test]
fn complex_raw_factors_cover_both_binary64_exponent_limits() {
    // A=(1+j)*scale, rhs=A or -j*A. The exact solutions are 1 and -j.
    // Even the reciprocal of the smallest subnormal complex pivot
    // is outside range, without making the matrix or solution invalid.
    for exponent in -1074..=1023 {
        let scale = if exponent < -1022 {
            f64::from_bits(1 << u32::try_from(exponent + 1074).unwrap())
        } else {
            f64::from_bits(u64::try_from(exponent + 1023).unwrap() << 52)
        };
        let mut a = SparseMatrix::new(1, 1);
        a.add(0, 0, scale).unwrap();
        let matrix = ComplexMatrix::from_operators(&a, &a, 1.).unwrap();
        let f = matrix.factorize().unwrap();
        let rhs = [Complex::new(scale, scale)];
        assert_eq!(f.solve(&rhs).unwrap(), [Complex::real(1.)]);
        assert_eq!(f.solve_transposed(&rhs).unwrap(), [Complex::real(1.)]);
        let rhs = [Complex::new(scale, -scale)];
        assert_eq!(f.solve(&rhs).unwrap(), [Complex::imaginary(-1.)]);
        if exponent == -1074 {
            assert!(f.solve(&[Complex::real(1.)]).is_err());
            assert_eq!(f.solve(&rhs).unwrap(), [Complex::imaginary(-1.)]);
        }
    }
}

/// Missing matrix rows must never become a valid cached factor, including
/// repeated AC attempts and transitions from complex to real NI.
#[test]
fn unreferenced_named_unknowns_fail_all_factorization_paths() {
    use ngspice_rs::maths::complex::IndependentComplexBatch;
    use ngspice_rs::maths::{Pattern, SparseLu, Vector, solve_ni};
    use std::sync::Arc;
    let p = Arc::new(Pattern::new(3, [(0, 0)]).unwrap());
    let names: ngspice_rs::maths::UnknownNames = Arc::from([
        "anchored".to_owned(),
        "floating".to_owned(),
        "orphan#branch".to_owned(),
    ]);
    let mut bound = SparseMatrix::bound(Arc::clone(&p)).unwrap();
    bound.set(0, 0, 2.).unwrap();
    bound.set_unknown_names(Some(Arc::clone(&names)));
    let mut triplets = SparseMatrix::new(3, 3);
    triplets.add(0, 0, 2.).unwrap();
    triplets.set_unknown_names(Some(names));
    let empty = SparseMatrix::new(3, 3);
    let diagnostic = |error: ngspice_rs::primitives::SpiceError| {
        let message = error.to_string();
        assert!(
            message.contains("singular matrix")
                && message.contains("floating")
                && message.contains("orphan#branch"),
            "{message}"
        );
    };
    for g in [&bound, &triplets] {
        let matrix = if g.bound_values().is_some() {
            ComplexMatrix::from_operators_on(&p, g, &empty, 1.)
                .unwrap()
                .unwrap()
        } else {
            ComplexMatrix::from_operators(g, &empty, 1.).unwrap()
        };
        for _ in 0..3 {
            diagnostic(matrix.factorize().unwrap_err());
            diagnostic(matrix.factorize_new_matrix().unwrap_err());
            diagnostic(SparseLu::new(g, None).unwrap_err());
            diagnostic(
                solve_ni(
                    g,
                    &Vector::zeros(3),
                    &mut None,
                    false,
                    ngspice_rs::maths::DiagonalGmin::unrestricted(0.),
                )
                .unwrap_err(),
            );
        }
        let mut batch = IndependentComplexBatch::default();
        for result in batch.factorize(&[matrix.clone(), matrix.clone(), matrix]) {
            diagnostic(result.unwrap_err());
        }
    }
    assert_eq!(p.factor_telemetry().work.factor.successful_factors, 0);
}
