//! Rank projection of consistent right-limit linear DAE coefficient arrays.
use ngspice_rs::maths::taylor::dae::{InitialRows, Limits, LinearDae, Solution, Tolerances};
use ngspice_rs::maths::{SparseMatrix, Vector};
use ngspice_rs::primitives::SpiceResult;

fn sparse(rows: usize, cols: usize, entries: &[(usize, usize, f64)]) -> SparseMatrix {
    let mut matrix = SparseMatrix::new(rows, cols);
    for &(row, col, value) in entries {
        matrix.add(row, col, value).unwrap();
    }
    matrix.fold_duplicates();
    matrix
}

fn solve(
    a: &SparseMatrix,
    e: &SparseMatrix,
    forcing: &[Vector],
    scale: f64,
    initial: &InitialRows<'_>,
    degrees: usize,
) -> SpiceResult<Solution> {
    LinearDae {
        static_matrix: a,
        storage_matrix: e,
        forcing,
        time_scale: scale,
    }
    .solve(
        initial,
        degrees,
        &Tolerances {
            relative: 1e-12,
            dae_absolute: &vec![0.; a.rows()],
            initial_absolute: &vec![0.; initial.matrix.rows()],
        },
        Limits::default(),
    )
}

#[test]
fn arbitrary_linear_differentiator_chain_needs_no_zero_derivative_gauge() {
    // x0 = f(t); x1=x0'; x2=x1'; x3=x2'.
    let a = sparse(4, 4, &[(0, 0, 1.), (1, 1, 1.), (2, 2, 1.), (3, 3, 1.)]);
    let e = sparse(4, 4, &[(1, 0, -1.), (2, 1, -1.), (3, 2, -1.)]);
    let forcing = [2., 3., 5., 7., 11.].map(|value| Vector::from_slice(&[value, 0., 0., 0.]));
    let initial = InitialRows {
        matrix: &SparseMatrix::new(0, 4),
        rhs: &Vector::zeros(0),
        optional: &[],
    };
    let shallow = solve(&a, &e, &forcing[..1], 0.5, &initial, 1).unwrap();
    assert_eq!(shallow.coefficients[0], [Some(2.), None, None, None]);
    let full = solve(&a, &e, &forcing, 0.5, &initial, 2).unwrap();
    for (actual, expected) in full
        .coefficients
        .iter()
        .zip([[2., 6., 40., 336.], [3., 20., 168., 2112.]])
    {
        for (value, expected) in actual.iter().zip(expected) {
            assert!((value.unwrap() - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn independent_state_is_retained_but_redundant_derivative_seed_is_not() {
    // x + x' = 1 (differential feedback state); y = x'.
    let a = sparse(2, 2, &[(0, 0, 1.), (1, 1, 1.)]);
    let e = sparse(2, 2, &[(0, 0, 1.), (1, 0, -1.)]);
    let forcing = [
        Vector::from_slice(&[1., 0.]),
        Vector::zeros(2),
        Vector::zeros(2),
    ];
    let constraints = sparse(2, 2, &[(0, 0, 1.), (1, 1, 1.)]);
    let rhs = Vector::from_slice(&[0.3, 0.]);
    let initial = InitialRows {
        matrix: &constraints,
        rhs: &rhs,
        optional: &[0, 1],
    };
    let result = solve(&a, &e, &forcing, 1., &initial, 2).unwrap();
    assert_eq!(result.retained_initial, [true, false]);
    for (actual, expected) in result.coefficients.iter().zip([[0.3, 0.7], [0.7, -0.7]]) {
        for (value, expected) in actual.iter().zip(expected) {
            assert!((value.unwrap() - expected).abs() < 1e-13);
        }
    }
    let contradictory = InitialRows {
        matrix: &constraints,
        rhs: &rhs,
        optional: &[],
    };
    assert!(solve(&a, &e, &forcing, 1., &contradictory, 1).is_err());
}

#[test]
fn algebraic_relations_and_free_gauges_are_distinct() {
    // x+y=3, y-z=0, z=2; an unrelated coordinate is genuinely free.
    let a = sparse(
        3,
        4,
        &[(0, 0, 1.), (0, 1, 1.), (1, 1, 1.), (1, 2, -1.), (2, 2, 1.)],
    );
    let e = SparseMatrix::new(3, 4);
    let forcing = [Vector::from_slice(&[3., 0., 2.])];
    let initial = InitialRows {
        matrix: &SparseMatrix::new(0, 4),
        rhs: &Vector::zeros(0),
        optional: &[],
    };
    let result = solve(&a, &e, &forcing, 1., &initial, 1).unwrap();
    assert_eq!(result.coefficients[0], [Some(1.), Some(2.), Some(2.), None]);
}

#[test]
fn resource_limits_and_inconsistent_zero_rows_are_enforced() {
    let a = sparse(1, 1, &[(0, 0, 1.)]);
    let e = SparseMatrix::new(1, 1);
    let forcing = [Vector::from_slice(&[2.])];
    let initial = InitialRows {
        matrix: &SparseMatrix::new(0, 1),
        rhs: &Vector::zeros(0),
        optional: &[],
    };
    let dae = LinearDae {
        static_matrix: &a,
        storage_matrix: &e,
        forcing: &forcing,
        time_scale: 1.,
    };
    let tolerances = Tolerances {
        relative: 1e-12,
        dae_absolute: &[0.],
        initial_absolute: &[],
    };
    for limits in [
        Limits {
            dimension: 1,
            ..Limits::default()
        },
        Limits {
            stamps: 0,
            ..Limits::default()
        },
        Limits {
            rank_work: 0,
            ..Limits::default()
        },
    ] {
        assert!(dae.solve(&initial, 1, &tolerances, limits).is_err());
    }
    let zero = SparseMatrix::new(1, 1);
    assert!(solve(&zero, &e, &forcing, 1., &initial, 1).is_err());
}

#[test]
fn long_feed_forward_chain_is_solved_with_a_bounded_sparse_array() {
    let count = 32;
    let mut a = SparseMatrix::new(count, count);
    let mut e = SparseMatrix::new(count, count);
    for i in 0..count {
        a.add(i, i, 1.).unwrap();
        if i > 0 {
            e.add(i, i - 1, -1.).unwrap();
        }
    }
    a.fold_duplicates();
    e.fold_duplicates();
    // Taylor expansion of exp(t) at zero, independently generated by
    // the factorial recurrence. All physical derivatives are one.
    let mut coefficient = 1.;
    let mut forcing = Vec::new();
    for order in 0..count {
        if order > 0 {
            coefficient /= f64::from(u32::try_from(order).unwrap());
        }
        let mut rhs = Vector::zeros(count);
        rhs.set(0, coefficient).unwrap();
        forcing.push(rhs);
    }
    let initial = InitialRows {
        matrix: &SparseMatrix::new(0, count),
        rhs: &Vector::zeros(0),
        optional: &[],
    };
    let result = solve(&a, &e, &forcing, 1., &initial, 1).unwrap();
    for value in &result.coefficients[0] {
        assert!((value.unwrap() - 1.).abs() < 1e-12);
    }
}
