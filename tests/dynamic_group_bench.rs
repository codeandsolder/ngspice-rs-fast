//! #133: packed ordered dynamic stamps, ownership and arena lifecycle.
use std::sync::Arc;
use std::time::Instant;

use ngspice_rs::maths::complex::ComplexMatrix;
use ngspice_rs::maths::{DynamicGroups, Pattern, SparseMatrix};
use ngspice_rs::primitives::Complex;

fn with_terms(
    e: &mut SparseMatrix,
    groups: &mut DynamicGroups,
    row: usize,
    col: usize,
    terms: &[f64],
    scale: f64,
    divisor: f64,
    prescaled: bool,
) {
    let start = e.triplets().len();
    for &term in terms {
        e.add(row, col, scale * term / divisor).unwrap();
    }
    let end = e.triplets().len();
    groups
        .push(e, start..end, terms, scale, divisor, prescaled)
        .unwrap();
}

fn group_value(e: &SparseMatrix, groups: &DynamicGroups, omega: f64) -> Complex {
    let mut a = SparseMatrix::new(1, 1);
    a.add(0, 0, 2.).unwrap();
    let pattern = Arc::new(Pattern::new(1, [(0, 0)]).unwrap());
    let matrix = ComplexMatrix::from_operators_grouped_on(&pattern, &a, e, groups, omega)
        .unwrap()
        .unwrap();
    let solution = matrix
        .factorize()
        .unwrap()
        .solve(&[Complex::new(1., 0.)])
        .unwrap();
    solution[0]
}

#[test]
fn prescaled_singleton_divisor_and_grouped_c_rounding_remain_distinct() {
    let mut groups = DynamicGroups::new();
    let mut e = SparseMatrix::new(1, 1);
    with_terms(&mut e, &mut groups, 0, 0, &[4.], 3., 2., true);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups.iter().next().unwrap().terms, &[4.]);
    // Once prescaled, these imaginary contributions are independent of
    // omega, even when C uses a negative AC mixing frequency.
    for omega in [-1e10, 0., 0.5, 7.] {
        let actual = group_value(&e, &groups, omega);
        let expected = Complex::new(1., 0.) / Complex::new(2., 6.);
        assert!((actual.re - expected.re).abs() < 1e-12);
        assert!((actual.im - expected.im).abs() < 1e-12);
    }
    e.clear();
    groups.clear();

    with_terms(
        &mut e,
        &mut groups,
        0,
        0,
        &[1e16, 0., -1e16, 1.],
        3.,
        2.,
        false,
    );
    assert_eq!(e.triplets().len(), 3); // the zero term was not a triplet
    assert_eq!(groups.iter().next().unwrap().terms.len(), 4); // still sums in C order
    for omega in [0., 1., 2., 5.] {
        let source = [1e16, 0., -1e16, 1.];
        let mut scaled = source.iter().map(|&term| omega * term);
        let mut sum = scaled.next().unwrap();
        for term in scaled {
            sum += term;
        }
        let imaginary = 3. * sum / 2.;
        let actual = group_value(&e, &groups, omega);
        let expected = Complex::new(1., 0.) / Complex::new(2., imaginary);
        assert!((actual.re - expected.re).abs() < 1e-12);
        assert!((actual.im - expected.im).abs() < 1e-12);
    }
}

#[test]
fn ordered_group_boundaries_and_invalid_metadata_fail_explicitly() {
    let mut e = SparseMatrix::new(3, 3);
    let mut groups = DynamicGroups::new();
    for (row, col, value) in [(0, 0, 1.), (0, 0, 2.), (1, 1, 3.), (2, 2, 4.)] {
        e.add(row, col, value).unwrap();
    }
    groups.push(&mut e, 0..2, &[1., 2.], 1., 1., false).unwrap();
    groups.push(&mut e, 3..4, &[4.], 1., 1., false).unwrap();
    assert_eq!(
        groups.iter().map(|g| g.range).collect::<Vec<_>>(),
        [0..2, 3..4]
    );
    assert!(groups.push(&mut e, 1..2, &[2.], 1., 1., false).is_err());
    assert!(
        groups
            .push(&mut e, 2..5, &[3., 4., 5.], 1., 1., false)
            .is_err()
    );
    assert!(groups.push(&mut e, 4..4, &[5.], 1., 1., false).is_err());
    let mut mixed = DynamicGroups::new();
    assert!(mixed.push(&mut e, 1..3, &[2., 3.], 1., 1., false).is_err());
    let mut mismatched = DynamicGroups::new();
    assert!(
        mismatched
            .push(&mut e, 0..1, &[27.], 1., 1., false)
            .is_err()
    );
    let mut foreign = SparseMatrix::new(3, 3);
    for stamp in e.triplets() {
        foreign.add(stamp.row, stamp.col, stamp.value).unwrap();
    }
    assert!(groups.validate(&foreign).is_err());
    assert!(ComplexMatrix::from_operators_grouped(&e, &foreign, &groups, 1.).is_err());
    assert!(groups.validate(&e).is_ok());
    e.fold_duplicates();
    assert!(groups.validate(&e).is_err());
}

#[test]
fn cleared_assembly_does_not_accept_stale_groups_and_reuses_arena() {
    const GROUPS: usize = 128;
    let mut groups = DynamicGroups::new();
    let mut e = SparseMatrix::new(GROUPS, GROUPS);
    let terms = [0.5, 0., 3., -0.1, 0.001, 1e-8];
    for repetition in 0..32 {
        e.clear();
        groups.clear();
        if repetition > 0 {
            assert!(groups.is_empty());
        }
        for index in 0..GROUPS {
            with_terms(&mut e, &mut groups, index, index, &terms, 1., 1., false);
        }
        let term_capacity = groups.term_capacity();
        let group_capacity = groups.group_capacity();
        assert_eq!(groups.len(), GROUPS);
        assert!(term_capacity >= GROUPS * terms.len());
        assert!(group_capacity >= GROUPS);
        groups.validate(&e).unwrap();
        e.clear();
        assert!(groups.validate(&e).is_err());
        groups.clear();
        assert_eq!(groups.term_capacity(), term_capacity);
        assert_eq!(groups.group_capacity(), group_capacity);
    }
}

#[test]
fn grouped_and_ungrouped_terms_with_gaps_preserve_frequency_assembly() {
    let mut e = SparseMatrix::new(1, 1);
    let mut groups = DynamicGroups::new();
    // Ungrouped contributions are scaled individually; groups retain C's
    // summed-before-dividing semantics.
    e.add(0, 0, 0.25).unwrap();
    with_terms(&mut e, &mut groups, 0, 0, &[1.5, -0.5], 2., 3., false);
    e.add(0, 0, -0.125).unwrap();
    for omega in [0.125, 1., 2., 32.] {
        let grouped = omega * 1.5 + omega * (-0.5);
        let imag = omega * 0.25 + (2. * grouped / 3.) + omega * (-0.125);
        let actual = group_value(&e, &groups, omega);
        let expected = Complex::new(1., 0.) / Complex::new(2., imag);
        assert!((actual.re - expected.re).abs() < 1e-12);
        assert!((actual.im - expected.im).abs() < 1e-12);
    }
}

#[test]
#[ignore = "measure allocator events externally under Valgrind DHAT"]
fn benchmark_many_grouped_stamps_and_arena_reuse() {
    const GROUPS: usize = 512;
    const RUNS: usize = 300;
    const TERMS: [f64; 8] = [1., -0.25, 0., 0.125, 1e-9, 0.3, -0.4, 0.0001];
    let mut e = SparseMatrix::new(GROUPS, GROUPS);
    let mut groups = DynamicGroups::new();
    let start = Instant::now();
    let mut initial_capacity = None;
    for _ in 0..RUNS {
        e.clear();
        groups.clear();
        for row in 0..GROUPS {
            with_terms(&mut e, &mut groups, row, row, &TERMS, 0.75, 2., false);
        }
        let current = (groups.group_capacity(), groups.term_capacity());
        if let Some(initial) = initial_capacity {
            assert_eq!(current, initial); // no new group or term capacity
        } else {
            initial_capacity = Some(current);
        }
    }
    eprintln!(
        "packed groups: {RUNS} assemblies x {GROUPS} groups x {} source terms: {:?}, capacities {:?}",
        TERMS.len(),
        start.elapsed(),
        initial_capacity
    );
}

#[test]
#[ignore = "test-only historical per-group vector baseline for Valgrind DHAT"]
fn benchmark_original_group_vec_allocations() {
    const GROUPS: usize = 512;
    const RUNS: usize = 300;
    const TERMS: [f64; 8] = [1., -0.25, 0., 0.125, 1e-9, 0.3, -0.4, 0.0001];
    let mut e = SparseMatrix::new(GROUPS, GROUPS);
    let mut groups: Vec<Vec<f64>> = Vec::new();
    let start = Instant::now();
    for _ in 0..RUNS {
        e.clear();
        groups.clear();
        for row in 0..GROUPS {
            for &term in &TERMS {
                e.add(row, row, 0.75 * term / 2.).unwrap();
            }
            groups.push(TERMS.to_vec());
        }
        assert_eq!(groups.len(), GROUPS);
    }
    eprintln!(
        "test-only per-group Vec baseline: {RUNS} assemblies x {GROUPS} groups x {} source terms: {:?}",
        TERMS.len(),
        start.elapsed()
    );
}
