//! Stable symbolic matrix identities and representative construction workloads.

use ngspice_rs::maths::{Pattern, SparseMatrix, Vector};
use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

fn assert_first_seen(pattern: &Pattern, coordinates: impl IntoIterator<Item = (usize, usize)>) {
    let mut expected = BTreeMap::new();
    for position in coordinates {
        let next = expected.len();
        expected.entry(position).or_insert(next);
    }
    assert_eq!(pattern.len(), expected.len());
    for (position, id) in expected {
        let handle = pattern.find(position.0, position.1).unwrap();
        assert_eq!(handle.index(), id);
        assert_eq!(pattern.position(handle), Some(position));
    }
}

fn irregular(n: usize, count: usize) -> Vec<(usize, usize)> {
    let mut state = 0x879a_eda3_7e45_a2b1_u64;
    (0..count)
        .map(|i| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let col = ((state >> 32) % n as u64) as usize;
            let row = if i.is_multiple_of(3) {
                n / 2
            } else {
                ((state >> 16) % n as u64) as usize
            };
            (row, col)
        })
        .collect()
}

#[test]
fn irregular_and_dense_rows_preserve_identities_through_extension() {
    let n = 257;
    let input = irregular(n, 12_000);
    let initial = Pattern::new(n, input.iter().copied()).unwrap();
    assert_first_seen(&initial, input.iter().copied());
    let extra = (0..n)
        .rev()
        .flat_map(|col| [(0, col), (n / 2, col), (0, col)]);
    let extended = initial.extended(extra.clone()).unwrap();
    assert_first_seen(&extended, input.iter().copied().chain(extra));
    assert!(extended.extended([(0, n)]).is_err());
    assert_first_seen(&initial, input);
}

#[test]
fn shuffled_stamps_and_extension_preserve_numeric_factorization() {
    let n = 41;
    let diagonal = (0..n).rev().map(|row| (row, row));
    let links = (1..n).flat_map(|row| [(row, row - 1), (row - 1, row)]);
    let pattern = Pattern::new(n, diagonal.clone()).unwrap();
    let pattern = Arc::new(pattern.extended(links.clone()).unwrap());
    assert_first_seen(&pattern, diagonal.chain(links.clone()));
    let mut matrix = SparseMatrix::bound(pattern).unwrap();
    for row in 0..n {
        matrix.add(row, row, 4.).unwrap();
    }
    for (row, col) in links {
        matrix.add(row, col, -1.).unwrap();
    }
    let rhs = Vector::from_slice(
        &(0..n)
            .map(|row| if row == 0 || row + 1 == n { 3. } else { 2. })
            .collect::<Vec<_>>(),
    );
    let solved = matrix.solve(&rhs).unwrap();
    assert!(
        solved
            .as_slice()
            .iter()
            .all(|value| (value - 1.).abs() < 1e-12)
    );
}

fn ladder(n: usize) -> Vec<(usize, usize)> {
    (0..n)
        .flat_map(|row| {
            [
                (row, row),
                (row, (row + 1) % n),
                ((row + 1) % n, row),
                (row, row),
            ]
        })
        .collect()
}

fn mesh(side: usize) -> Vec<(usize, usize)> {
    let n = side * side;
    (0..n)
        .flat_map(|row| [0, 1, side, n - 1, n - side, 0].map(|offset| (row, (row + offset) % n)))
        .collect()
}

fn construction_samples(name: &str, n: usize, input: &[(usize, usize)], repeats: usize) {
    let extra: Vec<_> = (0..n)
        .step_by(13)
        .flat_map(|row| [(row, (row + 7) % n), (row, row)])
        .collect();
    let mut setup = Vec::new();
    let mut extension = Vec::new();
    for _ in 0..5 {
        let mut setup_seconds = 0.;
        let mut extension_seconds = 0.;
        for _ in 0..repeats {
            let start = Instant::now();
            let pattern = Pattern::new(n, black_box(input.iter().copied())).unwrap();
            setup_seconds += start.elapsed().as_secs_f64();
            let start = Instant::now();
            let extended = pattern.extended(black_box(extra.iter().copied())).unwrap();
            extension_seconds += start.elapsed().as_secs_f64();
            black_box((pattern, extended));
        }
        setup.push(setup_seconds * 1e6 / repeats as f64);
        extension.push(extension_seconds * 1e6 / repeats as f64);
    }
    setup.sort_by(f64::total_cmp);
    extension.sort_by(f64::total_cmp);
    eprintln!(
        "{name}: setup_us={:.3} extension_us={:.3}",
        setup[2], extension[2]
    );
}

#[test]
#[ignore = "manual release construction benchmark"]
fn representative_symbolic_construction() {
    construction_samples("ladder", 32_768, &ladder(32_768), 10);
    construction_samples("mesh", 16_384, &mesh(128), 10);
    construction_samples("irregular_hub", 32_768, &irregular(32_768, 160_000), 10);
    construction_samples("small", 12, &ladder(12), 10_000);
    let sorted: Vec<_> = (0..32_768)
        .flat_map(|row| (0..4).map(move |col| (row, col)))
        .collect();
    construction_samples("row_major", 32_768, &sorted, 10);
}
