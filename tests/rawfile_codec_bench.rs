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
use std::io::{self, BufRead, BufReader, Read};
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
    workload_for(ROWS, COLUMNS, PlotFlags::Complex)
}

fn workload_for(rows: u32, columns: u32, flags: PlotFlags) -> RawFile {
    let mut plot = Plot::builder("bench1", "AC Analysis", flags);
    for column in 0..columns {
        let name = format!("v({column})");
        plot.push_variable(if !flags.is_complex() || column % 2 == 0 {
            Variable::new(name, "voltage")
        } else {
            Variable::complex(name, "voltage")
        });
    }
    let mut plot = plot.build();
    let mut values = Vec::with_capacity(columns as usize);
    for row in 0..rows {
        values.clear();
        values.extend((0..columns).map(|column| {
            let real = f64::from(row).mul_add(0.125, f64::from(column) + 1.0);
            if !flags.is_complex() || column % 2 == 0 {
                Complex::real(real)
            } else {
                Complex::new(real, -0.5 * real)
            }
        }));
        plot.push_point(&values).unwrap();
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
    assert_eq!(plot.point_count(), usize::try_from(ROWS).unwrap());
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

// Count calls at the codec-facing interface, not the physical File reads
// hidden behind the BufReader. A buffered syscall count alone cannot reveal
// the cost of per-cell copying and dispatch.
struct Counted<R> {
    input: R,
    reads: usize,
    fills: usize,
}

impl<R: Read> Read for Counted<R> {
    fn read(&mut self, target: &mut [u8]) -> io::Result<usize> {
        self.reads += 1;
        self.input.read(target)
    }
}

impl<R: BufRead> BufRead for Counted<R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.fills += 1;
        self.input.fill_buf()
    }

    fn consume(&mut self, count: usize) {
        self.input.consume(count);
    }
}

fn read_matrix(input: impl BufRead, rows: u32, columns: u32) -> (usize, usize) {
    let mut input = Counted {
        input,
        reads: 0,
        fills: 0,
    };
    let mut reader = RawFileReader::new(&mut input);
    let mut plot = reader.next_plot().unwrap().unwrap();
    let mut count = 0;
    while let Some(row) = plot.next_row().unwrap() {
        assert_eq!(row.len(), usize::try_from(columns).unwrap());
        black_box(row);
        count += 1;
    }
    assert_eq!(count, rows);
    plot.finish().unwrap();
    assert!(reader.next_plot().unwrap().is_none());
    (input.reads, input.fills)
}

fn binary_matrix(file: bool) {
    for (rows, columns) in [(262_144, 4), (256, 4097)] {
        for flags in [PlotFlags::Real, PlotFlags::Complex] {
            let raw = workload_for(rows, columns, flags);
            let path = file.then(fixture);
            // Repeat the full I/O phase, with fixture construction outside
            // each timed interval. Memory decoding borrows the output bytes.
            for trial in 0..3 {
                let start = Instant::now();
                let bytes = if let Some(path) = &path {
                    raw.write_with_format(path, RawFormat::Binary, BinaryByteOrder::LittleEndian)
                        .unwrap();
                    Vec::new()
                } else {
                    raw.to_binary().unwrap()
                };
                let encode_ns = start.elapsed().as_nanos();
                let start = Instant::now();
                let (reads, fills) = if let Some(path) = &path {
                    read_matrix(BufReader::new(File::open(path).unwrap()), rows, columns)
                } else {
                    read_matrix(bytes.as_slice(), rows, columns)
                };
                let decode_ns = start.elapsed().as_nanos();
                println!(
                    "matrix,file={file},rows={rows},columns={columns},flags={flags:?},trial={trial},encode_ns={encode_ns},decode_ns={decode_ns},read_calls={reads},fill_calls={fills}"
                );
                black_box(bytes);
            }
        }
    }
}

#[test]
#[ignore = "opt-in binary codec shape and call-count matrix"]
fn binary_matrix_memory() {
    binary_matrix(false);
}

#[test]
#[ignore = "opt-in binary codec shape and call-count matrix"]
fn binary_matrix_file() {
    binary_matrix(true);
}
