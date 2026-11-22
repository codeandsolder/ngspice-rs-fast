//! Streamed output (M10 0.5g): [`ngspice_rs::analysis::AnalysisPlan::run_into`] hands an analysis's
//! points to a [`PointSink`] instead of storing them, and [`RawFileWriter`]
//! writes plots one at a time, spooling a streamed plot's points until its
//! count is known. Both must reproduce the stored path exactly: the same rows,
//! and a rawfile byte-identical to [`RawFile::write_with_format`].

use std::fs;
use std::path::{Path, PathBuf};

use ngspice_rs::analysis::{
    BinaryByteOrder, Plot, PlotFlags, PointSink, RawFile, RawFileReader, RawFileWriter, RawFormat,
    RawPlot, RunConfig, Variable,
};
use ngspice_rs::netlist::Parser;
use ngspice_rs::netlist::ast::Netlist;
use ngspice_rs::netlist::source::parse_deck_text;
use ngspice_rs::primitives::{Complex, SpiceError, SpiceResult};

fn deck(body: &str) -> Netlist {
    Parser::new()
        .parse_deck(&parse_deck_text(
            Path::new("stream.cir"),
            &format!("streamed output\n{body}\n.end\n"),
        ))
        .unwrap()
}

/// A private scratch directory. `name` keeps concurrent tests apart.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("spice-rs-stream-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .expect("readable directory")
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Records what an analysis hands its sink.
#[derive(Default)]
struct Recorder {
    header: Option<Plot>,
    plots: Vec<Plot>,
    points: Vec<Vec<Complex>>,
}

impl PointSink for Recorder {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        assert!(self.header.is_none(), "begin is called once");
        assert!(header.points().is_empty(), "the header holds no points");
        self.header = Some(header.clone());
        Ok(())
    }

    fn point(&mut self, values: Vec<Complex>) -> SpiceResult<()> {
        assert!(self.header.is_some(), "begin precedes the points");
        self.points.push(values);
        Ok(())
    }
    fn end(&mut self) -> SpiceResult<()> {
        self.plots.push(
            self.header
                .take()
                .unwrap()
                .with_points(std::mem::take(&mut self.points))?,
        );
        Ok(())
    }
}

/// Runs the deck's first analysis stored and streamed and checks that the
/// sink saw exactly the stored plot.
fn assert_streams_the_stored_plot(body: &str) {
    let netlist = deck(body);
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();

    let mut circuit = config.circuit(&netlist).unwrap();
    let stored = request.run_plots(&mut circuit, &config.context()).unwrap();
    let final_state = circuit.final_transient_state();

    let mut circuit = config.circuit(&netlist).unwrap();
    let mut recorder = Recorder::default();
    request
        .run_into(&mut circuit, &config.context(), &mut recorder)
        .unwrap();
    assert!(recorder.header.is_none());
    assert_eq!(recorder.plots.len(), stored.len());
    for (streamed, stored) in recorder.plots.iter().zip(stored) {
        assert_eq!(streamed.variables(), stored.variables());
        assert_eq!(streamed.plotname, stored.plotname);
        assert!(!stored.points().is_empty());
        // Bit-exact row values, including signed zero.
        assert_eq!(streamed.point_count(), stored.point_count());
        for (a, b) in streamed
            .points()
            .iter()
            .flatten()
            .zip(stored.points().iter().flatten())
        {
            assert_eq!(a.re.to_bits(), b.re.to_bits());
            assert_eq!(a.im.to_bits(), b.im.to_bits());
        }
    }
    assert_eq!(circuit.final_transient_state(), final_state);
}

#[test]
fn a_streamed_transient_hands_the_sink_every_stored_row() {
    assert_streams_the_stored_plot(
        "v1 in 0 pulse(0 1 1m 1u 1u 1m 2m)\nr1 in out 1k\nc1 out 0 1u\n.tran 0.1m 6m",
    );
    // A delayed start writes only the rows at or after tstart.
    assert_streams_the_stored_plot(
        "v1 in 0 pulse(0 1 1m 1u 1u 1m 2m)\nr1 in out 1k\nc1 out 0 1u\n.tran 0.1m 6m 2m",
    );
}

#[test]
fn a_streamed_mos_transient_carries_the_same_final_load_as_the_stored_path() {
    assert_streams_the_stored_plot(
        "vd d 0 dc 3\nvg g 0 dc 2 ac 1\nvs s 0 0\n\
         vb b 0 dc -0.3 sin(-0.3 0.2 100k)\n\
         m1 d g s b nm l=10u w=20u ad=10p as=20p nrd=2 nrs=3\n\
         .model nm nmos level=1 vto=0.7 rsh=100 cj=1e-4 cgso=1e-10\n\
         .tran 0.2u 1u",
    );
}

#[test]
fn direct_ac_and_dc_streams_and_completed_op_match_collected_rows() {
    assert_streams_the_stored_plot("v1 in 0 dc 5\nr1 in out 1k\nr2 out 0 1k\n.op");
    assert_streams_the_stored_plot("v1 in 0 dc 5 ac 1\nr1 in out 1k\nc1 out 0 1u\n.ac dec 5 1 1k");
}

#[test]
fn multi_output_noise_and_bias_preserve_exact_rows_and_summary_cadence() {
    for (sweep, options) in [
        ("lin 1 1k 1k", ""),
        ("dec 4 1 1k 3", ""),
        (
            "dec 4 1 1k 3",
            ".control\nset sqrnoise\nset ngbehavior=s3\n.endc\n",
        ),
    ] {
        assert_streams_the_stored_plot(&format!(
            "v1 in 0 dc 1 ac 1\nr1 in out 1k\nr2 out 0 3k\n.options keepopinfo\n{options}.noise v(out) v1 {sweep}"
        ));
    }
    assert_streams_the_stored_plot(
        "v1 in 0 dc 1 ac 1\nr1 in out 1k\nc1 out 0 1u\n.options keepopinfo\n.ac lin 8 1 100",
    );
    assert_streams_the_stored_plot(
        "v1 in 0 dc 1\nv2 aux 0 dc 1\nr1 in out 1k\nr2 out aux 1k\n.dc v1 0 4 1 v2 0 2 1",
    );
}

#[test]
fn spectral_and_dc_sink_failures_stop_before_completing_a_plot() {
    struct Abort {
        begun: usize,
        ended: usize,
        points: usize,
    }
    impl PointSink for Abort {
        fn begin(&mut self, _header: &Plot) -> SpiceResult<()> {
            self.begun += 1;
            Ok(())
        }
        fn point(&mut self, _values: Vec<Complex>) -> SpiceResult<()> {
            self.points += 1;
            if self.points == 3 {
                return Err(SpiceError::circuit("stop on third row"));
            }
            Ok(())
        }
        fn end(&mut self) -> SpiceResult<()> {
            self.ended += 1;
            Ok(())
        }
    }
    for (analysis, bias) in [
        (".ac lin 20 1 20", 1),
        (".noise v(out) v1 lin 20 1 20 2", 1),
        (".dc v1 0 20 1", 0),
    ] {
        let netlist = deck(&format!(
            "v1 in 0 dc 1 ac 1\nr1 in out 1k\nr2 out 0 1k\n.options keepopinfo\n{analysis}"
        ));
        let config = RunConfig::from_netlist(&netlist).unwrap();
        let request = config.plan_for(&netlist.analyses[0]).unwrap();
        let mut circuit = config.circuit(&netlist).unwrap();
        let mut sink = Abort {
            begun: 0,
            ended: 0,
            points: 0,
        };
        let error = request
            .run_into(&mut circuit, &config.context(), &mut sink)
            .unwrap_err();
        assert!(error.to_string().contains("stop on third row"), "{error}");
        assert_eq!(
            (sink.begun, sink.ended, sink.points),
            (bias + 1, bias, 3),
            "{analysis}"
        );
    }
}

#[test]
fn a_failed_end_event_does_not_publish_transient_state() {
    struct AbortEnd;
    impl PointSink for AbortEnd {
        fn begin(&mut self, _header: &Plot) -> SpiceResult<()> {
            Ok(())
        }
        fn point(&mut self, _values: Vec<Complex>) -> SpiceResult<()> {
            Ok(())
        }
        fn end(&mut self) -> SpiceResult<()> {
            Err(SpiceError::circuit("end refused"))
        }
    }
    let netlist = deck("v1 in 0 dc 1\nr1 in out 1k\nc1 out 0 1u\n.tran 0.1m 1m");
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    assert!(
        request
            .run_into(&mut circuit, &config.context(), &mut AbortEnd)
            .unwrap_err()
            .to_string()
            .contains("end refused")
    );
    assert!(circuit.final_transient_state().is_none());
}

/// A sink that fails on its third point.
struct Refusing(usize);

impl PointSink for Refusing {
    fn begin(&mut self, _header: &Plot) -> SpiceResult<()> {
        Ok(())
    }

    fn point(&mut self, _values: Vec<Complex>) -> SpiceResult<()> {
        self.0 += 1;
        if self.0 == 3 {
            return Err(SpiceError::Unsupported {
                feature: "sink refused".to_owned(),
                location: None,
            });
        }
        Ok(())
    }
    fn end(&mut self) -> SpiceResult<()> {
        Ok(())
    }
}

#[test]
fn a_sink_error_stops_the_transient_with_that_error() {
    let netlist = deck("v1 in 0 dc 1\nr1 in out 1k\nc1 out 0 1u\n.tran 0.1m 1m");
    let config = RunConfig::from_netlist(&netlist).unwrap();
    let request = config.plan_for(&netlist.analyses[0]).unwrap();
    let mut circuit = config.circuit(&netlist).unwrap();
    let mut sink = Refusing(0);
    let error = request
        .run_into(&mut circuit, &config.context(), &mut sink)
        .unwrap_err();
    assert!(error.to_string().contains("sink refused"), "{error}");
    assert_eq!(sink.0, 3, "no point is offered after the failing one");
    assert!(circuit.final_transient_state().is_none());

    // A later failed run must not publish its partial accepted history over
    // the last successful transient's snapshot, either.
    request.run(&mut circuit, &config.context()).unwrap();
    let before = circuit
        .final_transient_state()
        .expect("successful run retained its load");
    let mut sink = Refusing(0);
    assert!(
        request
            .run_into(&mut circuit, &config.context(), &mut sink)
            .is_err()
    );
    assert_eq!(circuit.final_transient_state(), Some(before));
}

fn sample_plots() -> RawFile {
    let mut real = Plot::builder("tran1", "Transient Analysis", PlotFlags::Real);
    real.push_variable(Variable::new("time", "time"));
    real.push_variable(Variable::new("v(out)", "voltage"));
    let mut real = real.build();
    for index in 0..5 {
        let t = f64::from(index) * 1e-4;
        real.push_point(vec![Complex::real(t), Complex::real((-t / 1e-3).exp())])
            .unwrap();
    }
    let mut complex = Plot::builder("ac1", "AC Analysis", PlotFlags::Complex);
    complex.push_variable(Variable::new("frequency", "frequency"));
    complex.push_variable(Variable::complex("v(out)", "voltage"));
    let mut complex = complex.build();
    for index in 0..3 {
        let f = 10f64.powi(index);
        complex
            .push_point(vec![Complex::real(f), Complex::new(1.0 / f, -0.5 / f)])
            .unwrap();
    }
    let mut empty = Plot::builder("tran2", "Transient Analysis", PlotFlags::Real);
    empty.push_variable(Variable::new("time", "time"));
    let empty = empty.build();
    RawFile {
        plots: [real, complex, empty]
            .into_iter()
            .map(|plot| RawPlot {
                title: "streamed output".to_owned(),
                date: "Thu Jan  1 00:00:00 1970".to_owned(),
                command: "test".to_owned(),
                plot,
            })
            .collect(),
    }
}

#[test]
fn streamed_and_stored_plots_write_the_same_bytes() {
    let dir = scratch("writer");
    let rawfile = sample_plots();
    for (format, order) in [
        (RawFormat::Ascii, BinaryByteOrder::LittleEndian),
        (RawFormat::Binary, BinaryByteOrder::LittleEndian),
        (RawFormat::Binary, BinaryByteOrder::BigEndian),
    ] {
        let stored = dir.join("stored.raw");
        rawfile.write_with_format(&stored, format, order).unwrap();

        // The first plot whole, the rest streamed point by point.
        let streamed = dir.join("streamed.raw");
        let spool = dir.join("points.spool");
        let mut writer = RawFileWriter::create(&streamed, format, order).unwrap();
        writer.write_plot(&rawfile.plots[0]).unwrap();
        for raw_plot in &rawfile.plots[1..] {
            let mut plot = writer.begin_plot(raw_plot.clone(), &spool).unwrap();
            assert!(plot.header().plot.points().is_empty());
            for point in raw_plot.plot.points() {
                plot.push_point(point).unwrap();
            }
            assert_eq!(plot.point_count(), raw_plot.plot.points().len());
            writer.finish_plot(plot).unwrap();
            assert!(!spool.exists(), "finishing removes the spool");
        }
        writer.finish().unwrap();

        let expected_bytes = match format {
            RawFormat::Ascii => rawfile.to_ascii().unwrap().into_bytes(),
            RawFormat::Binary => rawfile.to_binary_with(order).unwrap(),
        };
        let stored_bytes = fs::read(&stored).unwrap();
        let streamed_bytes = fs::read(&streamed).unwrap();
        assert_eq!(stored_bytes, expected_bytes, "{format:?} {order:?}");
        assert_eq!(streamed_bytes, expected_bytes, "{format:?} {order:?}");
        // The binary format does not record its byte order; use the selected order
        // when verifying real and complex rows.
        let parsed = RawFileReader::new(streamed_bytes.as_slice())
            .with_byte_order(order)
            .read()
            .unwrap();
        assert_eq!(
            parsed,
            RawFileReader::new(stored_bytes.as_slice())
                .with_byte_order(order)
                .read()
                .unwrap(),
            "{format:?} {order:?}"
        );
        if format.is_binary() {
            for (actual, expected) in parsed.plots.iter().zip(&rawfile.plots) {
                assert_eq!(actual.plot.points(), expected.plot.points());
            }
        }
    }
    assert_eq!(entries(&dir), ["stored.raw", "streamed.raw"]);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_streamed_plot_rejects_ragged_points_and_cleans_up_when_abandoned() {
    let dir = scratch("abandon");
    let rawfile = sample_plots();
    let spool = dir.join("points.spool");
    let writer = RawFileWriter::create(
        dir.join("out.raw"),
        RawFormat::Binary,
        BinaryByteOrder::LittleEndian,
    )
    .unwrap();
    let mut plot = writer.begin_plot(rawfile.plots[0].clone(), &spool).unwrap();
    plot.push_point(&[Complex::real(0.0), Complex::real(1.0)])
        .unwrap();
    let error = plot.push_point(&[Complex::real(0.0)]).unwrap_err();
    assert!(error.to_string().contains("1 value(s)"), "{error}");
    assert_eq!(plot.point_count(), 1, "a rejected point is not counted");

    // An existing spool is never overwritten.
    let error = writer
        .begin_plot(rawfile.plots[1].clone(), &spool)
        .unwrap_err();
    assert!(matches!(error, SpiceError::Io { .. }), "{error}");

    assert!(spool.exists());
    drop(plot);
    assert!(
        !spool.exists(),
        "dropping an unfinished plot removes its spool"
    );
    drop(writer);
    assert_eq!(entries(&dir), ["out.raw"]);
    fs::remove_dir_all(&dir).unwrap();
}
#[test]
fn a_streamed_binary_plot_cannot_cross_byte_orders() {
    let dir = scratch("byte-order");
    let rawfile = sample_plots();
    let spool = dir.join("points.spool");
    let origin = RawFileWriter::create(
        dir.join("source.raw"),
        RawFormat::Binary,
        BinaryByteOrder::BigEndian,
    )
    .unwrap();
    let mut writer = RawFileWriter::create(
        dir.join("dest.raw"),
        RawFormat::Binary,
        BinaryByteOrder::LittleEndian,
    )
    .unwrap();
    let mut plot = origin.begin_plot(rawfile.plots[0].clone(), &spool).unwrap();
    plot.push_point(&rawfile.plots[0].plot.points()[0]).unwrap();
    assert!(writer.finish_plot(plot).is_err());
    writer.finish().unwrap();
    origin.finish().unwrap();
    assert!(!spool.exists());
    assert_eq!(fs::metadata(dir.join("dest.raw")).unwrap().len(), 0);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn atomic_transaction_matches_stored_and_streamed_encodings() {
    use ngspice_rs::analysis::RawFileTransaction;

    let dir = scratch("transaction-encodings");
    let rawfile = sample_plots();
    let output = dir.join("out.raw");
    for (format, order) in [
        (RawFormat::Ascii, BinaryByteOrder::LittleEndian),
        (RawFormat::Binary, BinaryByteOrder::LittleEndian),
        (RawFormat::Binary, BinaryByteOrder::BigEndian),
    ] {
        fs::write(&output, "PREVIOUS").unwrap();
        let mut transaction = RawFileTransaction::create(&output, format, order).unwrap();
        transaction.write_plot(&rawfile.plots[0]).unwrap();
        for raw in &rawfile.plots[1..] {
            transaction.begin_plot(raw.clone()).unwrap();
            for row in raw.plot.points() {
                transaction.push_point(row).unwrap();
            }
            let completed = transaction.finish_plot().unwrap();
            assert_eq!(completed.point_count, raw.plot.point_count());
            assert!(completed.header.plot.points().is_empty());
        }
        assert_eq!(fs::read_to_string(&output).unwrap(), "PREVIOUS");
        transaction.publish().unwrap();
        let expected = match format {
            RawFormat::Ascii => rawfile.to_ascii().unwrap().into_bytes(),
            RawFormat::Binary => rawfile.to_binary_with(order).unwrap(),
        };
        assert_eq!(fs::read(&output).unwrap(), expected);
        assert_eq!(entries(&dir), ["out.raw"]);
    }
    fs::remove_dir_all(&dir).unwrap();
}
