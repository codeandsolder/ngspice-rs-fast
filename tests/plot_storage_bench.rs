//! Matched opt-in storage microbenchmarks. Run one ignored test per process
//! using GNU time for peak RSS and Valgrind DHAT for allocation counts.
//! Legacy reproduces the previous Vec<Vec<Complex>> storage and copied
//! column queries; packed uses the actual Plot API. Both visit the same
//! values in the same order. Codec workloads are in rawfile_codec_bench.

use std::hint::black_box;
use std::time::Instant;

use ngspice_rs::analysis::{Plot, PlotFlags, Variable};
use ngspice_rs::primitives::Complex;

fn row(values: &mut Vec<Complex>, index: u32, columns: u32) {
    values.clear();
    values
        .extend((0..columns).map(|column| Complex::new(f64::from(index) + f64::from(column), -0.)));
}

fn packed(rows: u32, columns: u32) {
    let variables = (0..columns)
        .map(|column| Variable::complex(format!("v({column})"), "voltage"))
        .collect();
    let mut plot = Plot::new("bench", "AC Analysis", PlotFlags::Complex, variables);
    let mut values = Vec::with_capacity(usize::try_from(columns).unwrap());
    let start = Instant::now();
    for i in 0..rows {
        row(&mut values, i, columns);
        plot.push_point(&values).unwrap();
    }
    let build = start.elapsed();
    let start = Instant::now();
    let row_sum: f64 = black_box(&plot)
        .rows()
        .flatten()
        .map(|value| value.re)
        .sum();
    let row_scan = start.elapsed();
    let start = Instant::now();
    let mut column_sum = 0.;
    for _ in 0..8 {
        column_sum += black_box(&plot)
            .column_at(usize::try_from(columns / 2).unwrap())
            .unwrap()
            .iter()
            .map(|value| value.re)
            .sum::<f64>();
    }
    let column_scan = start.elapsed();
    let expected = (f64::from(rows) - 1.) * f64::from(rows) / 2.;
    assert_eq!(
        row_sum,
        f64::from(columns) * expected
            + f64::from(rows) * (f64::from(columns) - 1.) * f64::from(columns) / 2.
    );
    assert_eq!(
        column_sum,
        8. * (expected + f64::from(rows) * f64::from(columns / 2))
    );
    println!(
        "packed {rows}x{columns}: build {build:?}, row scan {row_scan:?}, 8 column scans {column_scan:?}; row checksum {row_sum}, column checksum {column_sum}"
    );
    black_box(&plot);
}

fn legacy(rows: u32, columns: u32) {
    let mut points = Vec::new();
    let mut values = Vec::with_capacity(usize::try_from(columns).unwrap());
    let start = Instant::now();
    for i in 0..rows {
        row(&mut values, i, columns);
        // Exactly one allocation per owned row, as the old collector.
        points.push(values.clone());
    }
    let build = start.elapsed();
    let start = Instant::now();
    let row_sum: f64 = black_box(&points)
        .iter()
        .flatten()
        .map(|value| value.re)
        .sum();
    let row_scan = start.elapsed();
    let start = Instant::now();
    let column = usize::try_from(columns / 2).unwrap();
    let mut column_sum = 0.;
    for _ in 0..8 {
        let copied: Vec<_> = black_box(&points)
            .iter()
            .map(|values| values[column])
            .collect();
        column_sum += black_box(&copied).iter().map(|value| value.re).sum::<f64>();
    }
    let column_scan = start.elapsed();
    println!(
        "legacy {rows}x{columns}: build {build:?}, row scan {row_scan:?}, 8 column scans {column_scan:?}; row checksum {row_sum}, column checksum {column_sum}"
    );
    black_box(&points);
}

#[test]
#[ignore = "opt-in matched storage benchmark"]
fn packed_narrow() {
    packed(1_000_000, 2);
}

#[test]
#[ignore = "opt-in matched storage benchmark"]
fn legacy_narrow() {
    legacy(1_000_000, 2);
}

#[test]
#[ignore = "opt-in matched storage benchmark"]
fn packed_wide() {
    packed(50_000, 65);
}

#[test]
#[ignore = "opt-in matched storage benchmark"]
fn legacy_wide() {
    legacy(50_000, 65);
}
