//! Opt-in rawfile allocation, throughput and peak-memory workload.
//!
//! Run individual ignored tests in separate processes with
//! `NGSPICE_RAWFILE_BENCH=/absolute/path/to/fixture.raw`. The encode cases
//! produce the fixture consumed by the read cases. Use `/usr/bin/time -v`
//! for peak RSS and Valgrind DHAT for allocation counts, rather than timing
//! an instrumented process. Each file contains 2,097,152 complex cells,
//! alternating genuinely complex and real-in-complex columns.

use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::Instant;

use ngspice_rs::analysis::{
    BinaryByteOrder, Plot, PlotFlags, RawCollectPolicy, RawFile, RawFileReader, RawFormat,
    RawReadPolicy, Variable,
};
use ngspice_rs::primitives::Complex;

const ROWS: u32 = 65_536;
const COLUMNS: u32 = 32;

fn fixture() -> PathBuf {
    PathBuf::from(std::env::var_os("NGSPICE_RAWFILE_BENCH").expect("set NGSPICE_RAWFILE_BENCH"))
}

fn workload() -> RawFile {
    let mut plot = Plot::builder("bench1", "AC Analysis", PlotFlags::Complex);
    for column in 0..COLUMNS {
        let name = format!("v({column})");
        plot.push_variable(if column % 2 == 0 {
            Variable::new(name, "voltage")
        } else {
            Variable::complex(name, "voltage")
        });
    }
    let mut plot = plot.build();
    for row in 0..ROWS {
        let values = (0..COLUMNS)
            .map(|column| {
                let real = f64::from(row).mul_add(0.125, f64::from(column) + 1.0);
                if column % 2 == 0 {
                    Complex::real(real)
                } else {
                    Complex::new(real, -0.5 * real)
                }
            })
            .collect();
        plot.push_point(values).unwrap();
    }
    RawFile::single(plot)
}

fn report(start: Instant) {
    let elapsed = start.elapsed().as_secs_f64();
    let cells = f64::from(ROWS) * f64::from(COLUMNS);
    println!(
        "{} cells, {elapsed:.3} s, {:.3} M cells/s",
        cells,
        cells / elapsed / 1e6
    );
}

#[test]
#[ignore = "opt-in multi-million-cell codec benchmark"]
fn encode_ascii() {
    let raw = workload();
    let path = fixture();
    let start = Instant::now();
    raw.write(path).unwrap();
    report(start);
}

#[test]
#[ignore = "opt-in multi-million-cell codec benchmark"]
fn encode_binary() {
    let raw = workload();
    let path = fixture();
    let start = Instant::now();
    raw.write_with_format(path, RawFormat::Binary, BinaryByteOrder::LittleEndian)
        .unwrap();
    report(start);
}

#[test]
#[ignore = "opt-in multi-million-cell codec benchmark"]
fn collect() {
    let path = fixture();
    let start = Instant::now();
    let raw = RawFile::load(&path).unwrap();
    let plot = raw.single_plot().unwrap();
    assert_eq!(plot.points().len(), usize::try_from(ROWS).unwrap());
    assert_eq!(plot.variables().len(), usize::try_from(COLUMNS).unwrap());
    black_box(&raw);
    report(start);
}

#[test]
#[ignore = "opt-in multi-million-cell codec benchmark"]
fn iterate() {
    let input = BufReader::new(File::open(fixture()).unwrap());
    let start = Instant::now();
    let mut reader = RawFileReader::new(input).with_policy(RawReadPolicy {
        collect: RawCollectPolicy {
            cells: 0,
            rows: 0,
            header_bytes: 0,
            variables: 0,
        },
        ..RawReadPolicy::DEFAULT
    });
    let mut rows = reader.next_plot().unwrap().unwrap();
    let mut count = 0;
    while let Some(row) = rows.next_row().unwrap() {
        assert_eq!(row.len(), usize::try_from(COLUMNS).unwrap());
        black_box(row);
        count += 1;
    }
    assert_eq!(count, ROWS);
    assert_eq!(
        rows.finish().unwrap().plot.variables().len(),
        usize::try_from(COLUMNS).unwrap()
    );
    assert!(reader.next_plot().unwrap().is_none());
    report(start);
}
