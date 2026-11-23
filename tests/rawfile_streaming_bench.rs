//! Opt-in #142 benchmark and C interoperability checks for direct atomic
//! rawfile streaming. The in-memory row buffer is independent of row count.

use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use ngspice_conformance::ngspice::Scratch;
use ngspice_conformance::process::{ProcessLimits, run};
use ngspice_rs::analysis::{
    BinaryByteOrder, Plot, PlotFlags, RawFileReader, RawFileTransaction, RawFormat, RawPlot,
    Variable,
};
use ngspice_rs::primitives::Complex;

fn header(name: &str, plotname: &str, flags: PlotFlags, columns: usize) -> RawPlot {
    let mut plot = Plot::builder(name, plotname, flags);
    for column in 0..columns {
        let name = if column == 0 {
            "time".to_owned()
        } else {
            format!("v({column})")
        };
        plot.push_variable(if flags == PlotFlags::Complex && column != 0 {
            Variable::complex(name, "voltage")
        } else {
            Variable::new(name, "voltage")
        });
    }
    RawPlot {
        title: "direct atomic rawfile benchmark".to_owned(),
        date: "Mon Nov 23 12:00:00 2026".to_owned(),
        command: "ngspice-rs-fast".to_owned(),
        plot: plot.build(),
    }
}

fn short_header(
    name: &str,
    plotname: &str,
    flags: PlotFlags,
    variables: &[(&str, bool)],
) -> RawPlot {
    let mut plot = Plot::builder(name, plotname, flags);
    for &(name, complex) in variables {
        plot.push_variable(if complex {
            Variable::complex(name, "voltage")
        } else {
            Variable::new(name, "voltage")
        });
    }
    RawPlot {
        title: "direct rawfile C reader".to_owned(),
        date: "Mon Nov 23 12:00:00 2026".to_owned(),
        command: "spice-rs".to_owned(),
        plot: plot.build(),
    }
}

fn c_loads_file(ngspice: &std::ffi::OsStr, dir: &Path, filename: &str, checks: &[(&str, &str)]) {
    let file = dir.join(filename);
    let mut control = format!("C rawfile loading\n.control\nload {}\n", file.display());
    for &(plot, signal) in checks {
        control.push_str(&format!("setplot {plot}\nprint {signal}\n"));
    }
    control.push_str("quit\n.endc\n.end\n");
    let script = dir.join("load.cir");
    fs::write(&script, control).unwrap();
    let result = run(ngspice, dir, &["-b", "load.cir"], ProcessLimits::ORACLE).unwrap();
    let stdout = String::from_utf8_lossy(&result.stdout);
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        result.status.success(),
        "C reader {filename}:\n{stdout}\n{stderr}"
    );
    assert!(
        !stderr.to_lowercase().contains("error"),
        "C reader {filename}:\n{stdout}\n{stderr}"
    );
    for &(_, signal) in checks {
        assert!(
            stdout.contains(signal),
            "{filename} missing {signal}:\n{stdout}\n{stderr}"
        );
    }
}

/// Verify C's load path, not merely the Rust rawfile decoder. The little-
/// endian binary payload and ASCII counts both use the same padded decimal
/// header, while a rootless PZ one-row/no-column plot has no payload.
#[test]
#[ignore = "requires NGSPICE_BIN; reads direct ASCII and binary rawfiles in C ngspice"]
fn c_reads_backpatched_counts_in_both_formats() {
    let ngspice = std::env::var_os("NGSPICE_BIN").expect("absolute NGSPICE_BIN");
    assert!(Path::new(&ngspice).is_absolute());
    let scratch = Scratch::new("direct-rawfile-c-read").unwrap();
    let dir = scratch.path();
    for format in [RawFormat::Ascii, RawFormat::Binary] {
        let name = if format == RawFormat::Ascii {
            "direct_ascii.raw"
        } else {
            "direct_binary.raw"
        };
        let mut writer =
            RawFileTransaction::create(dir.join(name), format, BinaryByteOrder::LittleEndian)
                .unwrap();
        writer
            .begin_plot(short_header(
                "op1",
                "Operating Point",
                PlotFlags::Real,
                &[("v(in)", false)],
            ))
            .unwrap();
        writer.push_point(&[Complex::real(1.)]).unwrap();
        writer.finish_plot().unwrap();

        writer
            .begin_plot(short_header(
                "ac1",
                "AC Analysis",
                PlotFlags::Complex,
                &[("frequency", false), ("v(in)", true)],
            ))
            .unwrap();
        writer
            .push_point(&[Complex::real(100.), Complex::new(2., -1.)])
            .unwrap();
        writer.finish_plot().unwrap();
        writer.publish().unwrap();
        c_loads_file(&ngspice, dir, name, &[("op1", "v(in)"), ("ac1", "v(in)")]);

        let bytes = fs::read(dir.join(name)).unwrap();
        let parsed = RawFileReader::new(bytes.as_slice()).read().unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed.plots[0].plot.point_count(), 1);
        assert_eq!(parsed.plots[1].plot.point_count(), 1);
        assert_eq!(
            parsed.plots[1].plot.value("v(in)", 0),
            Some(Complex::new(2., -1.))
        );
    }

    for format in [RawFormat::Ascii, RawFormat::Binary] {
        let name = if format == RawFormat::Ascii {
            "zero_ascii.raw"
        } else {
            "zero_binary.raw"
        };
        let mut writer =
            RawFileTransaction::create(dir.join(name), format, BinaryByteOrder::LittleEndian)
                .unwrap();
        writer
            .begin_plot(short_header(
                "tran1",
                "Transient Analysis",
                PlotFlags::Real,
                &[("time", false)],
            ))
            .unwrap();
        assert_eq!(writer.finish_plot().unwrap().point_count, 0);
        writer.publish().unwrap();
        c_loads_file(&ngspice, dir, name, &[]);
    }

    for format in [RawFormat::Ascii, RawFormat::Binary] {
        let name = if format == RawFormat::Ascii {
            "pz_ascii.raw"
        } else {
            "pz_binary.raw"
        };
        let mut writer =
            RawFileTransaction::create(dir.join(name), format, BinaryByteOrder::LittleEndian)
                .unwrap();
        writer
            .begin_plot(short_header(
                "pz1",
                "Pole-Zero Analysis",
                PlotFlags::Complex,
                &[],
            ))
            .unwrap();
        writer.push_point(&[]).unwrap();
        writer.finish_plot().unwrap();
        writer.publish().unwrap();
        let bytes = fs::read(dir.join(name)).unwrap();
        let parsed = RawFileReader::new(bytes.as_slice()).read().unwrap();
        assert_eq!(parsed.plots[0].plot.point_count(), 1);
        assert_eq!(parsed.plots[0].plot.variable_count(), 0);
        c_loads_file(&ngspice, dir, name, &[]);
    }
}

fn rows_payload() -> (usize, usize, [Complex; 16]) {
    let row = std::array::from_fn(|index| Complex::real(index as f64 + 1.));
    (131_072, row.len(), row)
}

/// Separate opt-in storage workload with no retained plot. Each path writes
/// an equal 16 MiB binary payload; the reference spool implementation writes
/// it to the spool, reads it, then writes it a second time to its destination.
/// Alternating repetitions reduce some filesystem cache-order bias.
#[test]
#[ignore = "opt-in 16 MiB direct-vs-spool storage benchmark; run with --nocapture"]
fn direct_io_work_replaces_two_spool_writes_and_one_payload_read() {
    let scratch = Scratch::new("direct-rawfile-storage-bench").unwrap();
    let dir = scratch.path();
    let (rows, columns, row) = rows_payload();
    let payload_bytes = rows * columns * 8;
    let mut block = [0u8; 16 * 8];
    for (index, value) in row.iter().enumerate() {
        block[index * 8..(index + 1) * 8].copy_from_slice(&value.re.to_le_bytes());
    }
    let mut samples = Vec::new();
    for trial in 0..6 {
        let direct = trial % 2 == 0;
        let start = Instant::now();
        if direct {
            let path = dir.join("direct.raw");
            let mut output =
                RawFileTransaction::create(&path, RawFormat::Binary, BinaryByteOrder::LittleEndian)
                    .unwrap();
            output
                .begin_plot(header(
                    "tran1",
                    "Transient Analysis",
                    PlotFlags::Real,
                    columns,
                ))
                .unwrap();
            for _ in 0..rows {
                output.push_point(&row).unwrap();
            }
            output.finish_plot().unwrap();
            output.publish().unwrap();
            assert!(fs::metadata(path).unwrap().len() >= payload_bytes as u64);
        } else {
            let spool_path = dir.join("legacy.points");
            let file = File::create(&spool_path).unwrap();
            let mut spool = BufWriter::new(file);
            for _ in 0..rows {
                spool.write_all(&block).unwrap();
            }
            spool.flush().unwrap();
            drop(spool);
            let mut source = File::open(&spool_path).unwrap();
            let mut target = BufWriter::new(File::create(dir.join("legacy.raw")).unwrap());
            assert_eq!(
                io::copy(&mut source, &mut target).unwrap(),
                payload_bytes as u64
            );
            target.flush().unwrap();
            fs::remove_file(spool_path).unwrap();
        }
        samples.push((direct, start.elapsed().as_secs_f64() * 1e3));
    }
    for (direct, millis) in samples {
        println!(
            "{}: {millis:.3} ms",
            if direct { "direct" } else { "spool replay" }
        );
    }
    println!(
        "payload: {payload_bytes} bytes; direct: {payload_bytes} payload bytes written, 0 read; \
         legacy: {} payload bytes written, {payload_bytes} read; \
         bounded encoder <=32768 bytes and BufWriter defaults <=8192 bytes",
        payload_bytes * 2
    );
}

/// Empty files must remain untouched by read/parse code; output is exclusively
/// published and its fixed-length reserved header is itself valid rawfile data.
#[test]
fn direct_zero_row_and_zero_width_pz_have_exact_shapes() {
    let scratch = Scratch::new("direct-empty-pz").unwrap();
    let dir = scratch.path();
    for format in [RawFormat::Ascii, RawFormat::Binary] {
        let path = dir.join(format!("direct-{format:?}.raw"));
        let mut writer =
            RawFileTransaction::create(&path, format, BinaryByteOrder::LittleEndian).unwrap();
        writer
            .begin_plot(header("tran1", "Transient Analysis", PlotFlags::Real, 1))
            .unwrap();
        writer.finish_plot().unwrap();
        writer
            .begin_plot(header("pz1", "Pole-Zero Analysis", PlotFlags::Complex, 0))
            .unwrap();
        writer.push_point(&[]).unwrap();
        writer.finish_plot().unwrap();
        writer.publish().unwrap();
        let bytes = fs::read(path).unwrap();
        let parsed = RawFileReader::new(bytes.as_slice()).read().unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed.plots[0].plot.point_count(), 0);
        assert_eq!(parsed.plots[1].plot.point_count(), 1);
        assert_eq!(parsed.plots[1].plot.variable_count(), 0);
    }
}
