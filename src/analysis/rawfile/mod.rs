//! ngspice rawfile I/O — the ASCII and binary variants, read and write.
//!
//! Ported from `src/frontend/rawfile.c` (`ft_rawfile()`, `raw_read()`,
//! `raw_write()`). ngspice writes rawfiles in **binary** by default; the ASCII
//! form is selected with `set filetype=ascii`, which is what
//! `cargo xtask golden capture` does so that the golden files are diffable and
//! reviewable.
//!
//! Both forms share one header and differ only in the line that announces the
//! values (`Values:` against `Binary:`) and in how those values are encoded.
//! [`RawFileReader`] reads each header once and selects a row decoder,
//! so binary bytes are never decoded as text. [`RawFormat::detect`] reports the
//! encoding of a valid first header without reading the payload.
//!
//! An ASCII rawfile is a sequence of plots, each looking like:
//!
//! ```text
//! Title: rc divider
//! Date: Mon Oct  5 17:53:58 2026
//! Command: ngspice-47+, Build
//! Plotname: Operating Point
//! Flags: real
//! No. Variables: 3
//! No. Points: 1
//! Variables:
//! 	0	v(in)	voltage
//! 	1	v(out)	voltage
//! 	2	i(v1)	current
//! Values:
//!  0	5.000000000000000e+00
//! 	2.500000000000000e+00
//! 	-2.500000000000000e-03
//! ```
//!
//! Two details of the `Values:` section matter for parsing. The first value of
//! each point carries its point index, written as a space and the digits with no
//! tab; every point is followed by a blank line; and complex values are written
//! as `re,im`. Because every value is written with `%-.15e`, a leading run of
//! digits followed by whitespace can only be a point index — never a value —
//! which is how the internal `split_index_and_value` helper tells them apart.
//!
//! A binary rawfile is the same header followed by `Binary:` and then the raw
//! `double`s `raw_write()` `fwrite()`s: point-major, one `double` per value for a
//! `real` plot and `re, im` for a `complex` one. The complete layout, its
//! provenance in `rawfile.c` and the variants the port rejects are documented in
//! `docs/port/RAWFILES.md` and in the `binary` submodule. [`RawFile::parse`]
//! still refuses binary input; use [`RawFile::parse_bytes`] or
//! [`RawFileReader`].

// The `Values:` section is tab-separated, so the doc example above uses tabs.
// They are intentional: they show the exact byte layout ngspice writes.
#![expect(
    clippy::tabs_in_doc_comments,
    reason = "the ASCII ngspice rawfile documentation demonstrates literal tab separator bytes"
)]

mod transaction;

use transaction::OwnedTemporary;
pub use transaction::RawFileTransaction;
/// One transaction-owned report spool, with byte ranges for consecutive plots.
/// Text is not published until simulation and rawfile publication succeed.
#[derive(Debug)]
pub(crate) struct TextSpool {
    file: BufWriter<File>,
    temporary: OwnedTemporary,
}

use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

use crate::analysis::results::{Plot, PlotFlags, Variable};

mod binary;
mod policy;

pub use policy::{RawCollectPolicy, RawReadPolicy};
use policy::{limit, reserve, spend};

pub use binary::BinaryByteOrder;

fn unsupported(feature: impl Into<String>) -> SpiceError {
    SpiceError::Unsupported {
        feature: feature.into(),
        location: None,
    }
}

fn missing_header(key: &str) -> SpiceError {
    unsupported(format!("rawfile plot without a '{key}:' header"))
}

const BINARY_UNSUPPORTED: &str = concat!(
    "binary rawfile; read it with RawFile::parse_bytes or RawFileReader, ",
    "or write the ASCII form with 'set filetype=ascii'"
);

/// A plot together with the headers ngspice writes around it.
#[derive(Debug, Clone, PartialEq)]
pub struct RawPlot {
    /// The `Title:` header, which is the deck's title line.
    pub title: String,
    /// The `Date:` header.
    pub date: String,
    /// The `Command:` header.
    pub command: String,
    /// The data.
    pub plot: Plot,
}

/// A rawfile: the plots it contains, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct RawFile {
    /// The plots, in file order.
    pub plots: Vec<RawPlot>,
}

impl RawFile {
    /// A rawfile holding a single plot with empty headers.
    #[must_use]
    pub fn single(plot: Plot) -> Self {
        Self {
            plots: vec![RawPlot {
                title: String::new(),
                date: String::new(),
                command: String::new(),
                plot,
            }],
        }
    }

    /// Number of plots.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.plots.len()
    }

    /// True when the rawfile holds no plots.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.plots.is_empty()
    }

    /// The only plot, when the rawfile holds exactly one.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Unsupported`] when there are zero or several plots, because
    /// the caller asked for one and the file is ambiguous.
    pub fn single_plot(&self) -> SpiceResult<&Plot> {
        match self.plots.as_slice() {
            [only] => Ok(&only.plot),
            other => Err(unsupported(format!(
                "expected exactly one plot in the rawfile, found {}",
                other.len()
            ))),
        }
    }

    /// Parses ASCII rawfile text.
    ///
    /// Use [`RawFile::parse_bytes`] for a file that may be binary: this method
    /// only ever sees text.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Unsupported`] for a binary rawfile, a malformed header, a
    /// truncated file, or values that are not numbers.
    pub fn parse(text: &str) -> SpiceResult<Self> {
        RawFileReader::new(text.as_bytes())
            .with_format(RawFormat::Ascii)
            .read()
    }

    /// Reads and parses a rawfile of either encoding.
    ///
    /// A buffered reader decodes headers and rows without holding the input file in memory.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Io`] when the file cannot be read, plus everything
    /// [`RawFile::parse_bytes`] reports.
    pub fn load(path: impl AsRef<Path>) -> SpiceResult<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| SpiceError::io(path, &error))?;
        RawFileReader::new(BufReader::new(file))
            .read()
            .map_err(|error| {
                if let SpiceError::Io { message, .. } = error {
                    SpiceError::Io {
                        path: path.to_path_buf(),
                        message,
                    }
                } else {
                    error
                }
            })
    }

    /// Parses a rawfile from its bytes, detecting which encoding it uses.
    ///
    /// A `real` or `complex` binary payload is read as little-endian, the layout
    /// `raw_write()` produces on the platforms the port supports. Because a
    /// binary rawfile does not record its byte order, a file written on a
    /// big-endian machine needs an explicit
    /// [`RawFileReader::with_byte_order`].
    ///
    /// # Errors
    ///
    /// Everything [`RawFile::parse`] reports for the ASCII form, plus the
    /// failures of the binary codec, which `docs/port/RAWFILES.md` lists.
    pub fn parse_bytes(bytes: &[u8]) -> SpiceResult<Self> {
        RawFileReader::new(bytes).read()
    }

    /// Checks that every plot is internally consistent.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] when a point does not have one value per
    /// variable.
    pub fn validate(&self) -> SpiceResult<()> {
        self.plots
            .iter()
            .try_for_each(|raw_plot| validate_plot(&raw_plot.plot))
    }

    /// Renders the rawfile in ngspice's ASCII format.
    ///
    /// The output round-trips through [`RawFile::parse`] and reproduces
    /// `raw_write()`'s layout byte for byte: the point index is a space and the
    /// digits with no tab, every point is followed by a blank line, a `real`
    /// plot carries only real parts, and a vector flagged real inside a
    /// `complex` plot is written as `re,0.0` (see [`Variable::is_real`]).
    ///
    /// [`RawFile::write`] streams the same bytes to a file without
    /// building them in memory.
    ///
    /// # Errors
    /// [`SpiceError::Numerical`] for a row whose length differs from the schema,
    /// or [`SpiceError::Io`] if rendering fails.
    pub fn to_ascii(&self) -> SpiceResult<String> {
        let bytes = self.encode(RawFormat::Ascii, BinaryByteOrder::LittleEndian)?;
        String::from_utf8(bytes)
            .map_err(|error| unsupported(format!("invalid UTF-8 in encoded rawfile: {error}")))
    }

    /// Writes the rawfile in ASCII format.
    ///
    /// The file is streamed through a buffer, so its contents are never held
    /// in memory as a whole.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] from [`RawFile::validate`], or
    /// [`SpiceError::Io`] when the file cannot be written.
    pub fn write(&self, path: impl AsRef<Path>) -> SpiceResult<()> {
        self.write_with_format(path, RawFormat::Ascii, BinaryByteOrder::LittleEndian)
    }

    /// Renders the rawfile in ngspice's binary format, little-endian.
    ///
    /// The bytes are what `raw_write()` `fwrite()`s after it switches on
    /// `set filetype=binary`: the same header as [`RawFile::to_ascii`], then
    /// `Binary:`, then one `double` per value point-major. See
    /// `docs/port/RAWFILES.md` for the full layout and its provenance.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] from [`RawFile::validate`].
    pub fn to_binary(&self) -> SpiceResult<Vec<u8>> {
        self.to_binary_with(BinaryByteOrder::LittleEndian)
    }

    /// Renders the rawfile in ngspice's binary format with an explicit byte
    /// order.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] from [`RawFile::validate`], or
    /// [`SpiceError::Io`] if the binary renderer fails.
    pub fn to_binary_with(&self, byte_order: BinaryByteOrder) -> SpiceResult<Vec<u8>> {
        self.encode(RawFormat::Binary, byte_order)
    }

    fn encode(&self, format: RawFormat, byte_order: BinaryByteOrder) -> SpiceResult<Vec<u8>> {
        self.validate()?;
        let mut bytes = Vec::new();
        let mut binary = binary::Encoder::new();
        for raw_plot in &self.plots {
            write_plot(&mut bytes, raw_plot, format, byte_order, &mut binary)
                .map_err(|error| SpiceError::io("<rawfile memory buffer>", &error))?;
        }
        Ok(bytes)
    }

    /// Writes the rawfile in the requested encoding.
    ///
    /// [`RawFormat::Ascii`] writes exactly [`RawFile::to_ascii`];
    /// [`RawFormat::Binary`] writes exactly [`RawFile::to_binary_with`] for the selected byte order. Either is
    /// streamed through a buffer rather than rendered in memory first, so
    /// writing a long transient costs no second copy of its data.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] from [`RawFile::validate`], or
    /// [`SpiceError::Io`] when the file cannot be written.
    pub fn write_with_format(
        &self,
        path: impl AsRef<Path>,
        format: RawFormat,
        byte_order: BinaryByteOrder,
    ) -> SpiceResult<()> {
        self.validate()?;
        let mut writer = RawFileWriter::create(path, format, byte_order)?;
        for raw_plot in &self.plots {
            write_plot(
                &mut writer.out,
                raw_plot,
                format,
                byte_order,
                &mut writer.binary,
            )
            .map_err(|error| writer.io_error(&error))?;
        }
        writer.finish()
    }
}

/// A stream's metadata is bounded by the plot schema, never the point count.
/// The fixed-width decimal field is the only header data changed in-place.
#[derive(Debug)]
struct ActivePlot {
    header: RawPlot,
    point_count: usize,
    count_offset: u64,
}

/// Number of decimal digits reserved for an unsigned 64-bit point count.
/// A left-space-padded decimal is accepted by C's numeric rawfile reader and
/// the Rust rawfile reader. Retained (known-count) plots remain byte-exact.
const STREAMED_COUNT_WIDTH: usize = 20;

/// A seekable rawfile writer: completed plots retain their canonical compact
/// header, while unknown-length plots stream into this file directly.
/// No data spool or post-run payload copy is needed.
///
/// An incomplete or failed plot permanently poisons the writer. The atomic
/// output transaction owns the destination and publishes only a completed
/// writer, so provisional headers never become visible at the final path.
#[derive(Debug)]
pub struct RawFileWriter {
    out: BufWriter<File>,
    binary: binary::Encoder,
    path: PathBuf,
    format: RawFormat,
    byte_order: BinaryByteOrder,
    active: Option<ActivePlot>,
    failed: bool,
}

impl RawFileWriter {
    /// Creates a non-atomic seekable rawfile writer, truncating an existing file.
    /// Use RawFileTransaction to publish complete output atomically.
    ///
    /// # Errors
    /// Failure to open or create the output.
    pub fn create(
        path: impl AsRef<Path>,
        format: RawFormat,
        byte_order: BinaryByteOrder,
    ) -> SpiceResult<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path).map_err(|error| SpiceError::io(&path, &error))?;
        Ok(Self::from_file(file, path, format, byte_order))
    }

    fn from_file(
        file: File,
        path: PathBuf,
        format: RawFormat,
        byte_order: BinaryByteOrder,
    ) -> Self {
        Self {
            out: BufWriter::new(file),
            binary: binary::Encoder::new(),
            path,
            format,
            byte_order,
            active: None,
            failed: false,
        }
    }

    /// The encoding this writer uses.
    #[must_use]
    pub const fn format(&self) -> RawFormat {
        self.format
    }

    fn execute<T>(
        &mut self,
        operation: impl FnOnce(&mut Self) -> SpiceResult<T>,
    ) -> SpiceResult<T> {
        if self.failed {
            return Err(unsupported("rawfile writer is poisoned"));
        }
        let result = operation(self);
        if result.is_err() {
            self.failed = true;
        }
        result
    }

    /// Whether a streamed plot still has an unpatched header.
    #[must_use]
    const fn has_active_plot(&self) -> bool {
        self.active.is_some()
    }

    /// Writes a complete plot with the same compact header as ngspice.
    ///
    /// # Errors
    /// Invalid plot, active stream, or I/O failure; errors poison the writer.
    pub fn write_plot(&mut self, raw_plot: &RawPlot) -> SpiceResult<()> {
        self.execute(|this| {
            if this.has_active_plot() {
                return Err(unsupported(
                    "rawfile writer has an unfinished streamed plot",
                ));
            }
            validate_plot(&raw_plot.plot)?;
            write_plot(
                &mut this.out,
                raw_plot,
                this.format,
                this.byte_order,
                &mut this.binary,
            )
            .map_err(|error| this.io_error(&error))
        })
    }

    /// Begin a streamed plot and reserve its count in the final seekable file.
    /// Incoming points are encoded directly behind this provisional header.
    ///
    /// # Errors
    /// Invalid schema, unfinished previous plot, or output write/seek failure.
    pub fn begin_plot(&mut self, mut header: RawPlot) -> SpiceResult<()> {
        self.execute(|this| {
            if this.has_active_plot() {
                return Err(unsupported(
                    "rawfile writer has an unfinished streamed plot",
                ));
            }
            header.plot.clear_points();
            validate_plot(&header.plot)?;
            render_header_before_count(&mut this.out, &header)
                .map_err(|error| this.io_error(&error))?;
            this.out
                .write_all(b"No. Points: ")
                .map_err(|error| this.io_error(&error))?;
            let count_offset = this
                .out
                .stream_position()
                .map_err(|error| this.io_error(&error))?;
            this.out
                .write_all(&point_count_field(0)?)
                .and_then(|()| this.out.write_all(b"\n"))
                .and_then(|()| render_header_after_count(&mut this.out, &header, this.format))
                .map_err(|error| this.io_error(&error))?;
            this.active = Some(ActivePlot {
                header,
                point_count: 0,
                count_offset,
            });
            Ok(())
        })
    }

    /// The active plot's schema, or None between plots.
    #[must_use]
    pub fn active_header(&self) -> Option<&RawPlot> {
        self.active.as_ref().map(|plot| &plot.header)
    }

    /// The number of points written since begin_plot.
    #[must_use]
    pub fn active_point_count(&self) -> Option<usize> {
        self.active.as_ref().map(|plot| plot.point_count)
    }

    /// Encode a row directly into the output, buffering at most one binary
    /// block. A rejected or failed write permanently poisons the writer.
    ///
    /// # Errors
    /// No active plot, row/schema mismatch, count overflow, or I/O failure.
    pub fn push_point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        self.execute(|this| {
            let active = this
                .active
                .as_mut()
                .ok_or_else(|| unsupported("rawfile writer has no active streamed plot"))?;
            active.header.plot.check_point(values)?;
            let next_count = active
                .point_count
                .checked_add(1)
                .ok_or_else(|| unsupported("rawfile point count overflows a usize"))?;
            if active.header.plot.variable_count() == 0
                && !super::results::valid_empty_shape(
                    &active.header.plot.plotname,
                    active.header.plot.flags(),
                    next_count,
                )
            {
                return Err(unsupported(
                    "zero variables with nonzero points outside a rootless pole-zero result",
                ));
            }
            let result = match this.format {
                RawFormat::Ascii => render_ascii_point(
                    &mut this.out,
                    &active.header.plot,
                    active.point_count,
                    values,
                ),
                RawFormat::Binary => this.binary.write_row(
                    &mut this.out,
                    &active.header.plot,
                    values,
                    this.byte_order,
                ),
            };
            result.map_err(|error| SpiceError::io(&this.path, &error))?;
            active.point_count = next_count;
            Ok(())
        })
    }

    /// Flush the payload, overwrite only the reserved count field and restore
    /// the end position for the next plot. No payload bytes are moved.
    ///
    /// # Errors
    /// No stream, count overflow, flush, seek or backpatch I/O failure.
    pub fn finish_plot(&mut self) -> SpiceResult<FinishedPlot> {
        self.execute(|this| {
            let active = this
                .active
                .take()
                .ok_or_else(|| unsupported("rawfile writer has no active streamed plot"))?;
            this.binary
                .flush(&mut this.out)
                .map_err(|error| this.io_error(&error))?;
            let end = this
                .out
                .stream_position()
                .map_err(|error| this.io_error(&error))?;
            let count = point_count_field(active.point_count)?;
            this.out
                .seek(SeekFrom::Start(active.count_offset))
                .and_then(|_| this.out.write_all(&count))
                .and_then(|()| this.out.seek(SeekFrom::Start(end)).map(|_| ()))
                .map_err(|error| this.io_error(&error))?;
            Ok(FinishedPlot {
                header: active.header,
                point_count: active.point_count,
            })
        })
    }

    /// Explicitly flush the last plot and close the writer.
    ///
    /// # Errors
    /// Incomplete/failed plot or output flush failure.
    pub fn finish(mut self) -> SpiceResult<()> {
        if self.failed || self.has_active_plot() {
            return Err(unsupported(
                "cannot finish failed or unfinished rawfile writer",
            ));
        }
        self.out.flush().map_err(|error| self.io_error(&error))
    }

    fn io_error(&self, error: &io::Error) -> SpiceError {
        SpiceError::io(&self.path, error)
    }
}

/// Encode an unsigned count as the fixed-width decimal field reserved in the
/// header. The upper bound is explicit rather than allowing format! to grow.
fn point_count_field(count: usize) -> SpiceResult<[u8; STREAMED_COUNT_WIDTH]> {
    let decimal = count.to_string();
    let begin = STREAMED_COUNT_WIDTH
        .checked_sub(decimal.len())
        .ok_or_else(|| unsupported("rawfile point count exceeds reserved header width"))?;
    let mut field = [b' '; STREAMED_COUNT_WIDTH];
    field[begin..].copy_from_slice(decimal.as_bytes());
    Ok(field)
}

/// Metadata of a completed streamed plot; rows were written without retention.
#[derive(Debug)]
pub struct FinishedPlot {
    /// Plot headers and column schema.
    pub header: RawPlot,
    /// Number of stored points.
    pub point_count: usize,
}

fn validate_plot(plot: &Plot) -> SpiceResult<()> {
    plot.validate_shape()
}

/// The line that ends the header in `format`.
const fn section_line(format: RawFormat) -> &'static [u8] {
    match format {
        RawFormat::Ascii => b"Values:\n",
        RawFormat::Binary => b"Binary:\n",
    }
}

/// Writes one validated plot with an explicit format and byte order.
fn write_plot(
    out: &mut impl Write,
    raw_plot: &RawPlot,
    format: RawFormat,
    byte_order: BinaryByteOrder,
    binary: &mut binary::Encoder,
) -> io::Result<()> {
    let plot = &raw_plot.plot;
    render_header(out, raw_plot, plot.point_count(), format)?;
    match format {
        RawFormat::Ascii => {
            for (index, point) in plot.rows().enumerate() {
                render_ascii_point(out, plot, index, point)?;
            }
        }
        RawFormat::Binary => {
            for point in plot.rows() {
                binary.write_row(out, plot, point, byte_order)?;
            }
            binary.flush(out)?;
        }
    }
    Ok(())
}

fn render_ascii_point(
    out: &mut impl Write,
    plot: &Plot,
    index: usize,
    point: &[Complex],
) -> io::Result<()> {
    if point.is_empty() {
        // C's incremental writer has no value/newline to follow its index.
        return write!(out, "{index}\t");
    }
    write!(out, " {index}")?;
    for (variable, value) in plot.variables().iter().zip(point) {
        let real = SpiceNumber(value.re);
        match plot.flags() {
            PlotFlags::Real => writeln!(out, "\t{real}")?,
            PlotFlags::Complex if variable.is_real => writeln!(out, "\t{real},0.0")?,
            PlotFlags::Complex => writeln!(out, "\t{real},{}", SpiceNumber(value.im))?,
        }
    }
    writeln!(out)
}

/// Common immutable header prefix, ending immediately before the count.
fn render_header_before_count(out: &mut impl Write, raw_plot: &RawPlot) -> io::Result<()> {
    let plot = &raw_plot.plot;
    writeln!(out, "Title: {}", raw_plot.title)?;
    writeln!(out, "Date: {}", raw_plot.date)?;
    writeln!(out, "Command: {}", raw_plot.command)?;
    writeln!(out, "Plotname: {}", plot.plotname)?;
    writeln!(out, "Flags: {}", plot.flags().as_rawfile())?;
    writeln!(out, "No. Variables: {}", plot.variables().len())
}

/// Common section after the count; payload starts at its final newline.
fn render_header_after_count(
    out: &mut impl Write,
    raw_plot: &RawPlot,
    format: RawFormat,
) -> io::Result<()> {
    writeln!(out, "Variables:")?;
    for (index, variable) in raw_plot.plot.variables().iter().enumerate() {
        writeln!(out, "\t{index}\t{}\t{}", variable.name, variable.unit)?;
    }
    out.write_all(section_line(format))
}

/// Retained plots preserve ngspice's unpadded decimal header byte for byte.
fn render_header(
    out: &mut impl Write,
    raw_plot: &RawPlot,
    point_count: usize,
    format: RawFormat,
) -> io::Result<()> {
    render_header_before_count(out, raw_plot)?;
    writeln!(out, "No. Points: {point_count}")?;
    render_header_after_count(out, raw_plot, format)
}

/// Remaining physical bytes and lines in the current header.
struct HeaderBudget {
    bytes: usize,
    lines: usize,
}

/// Byte-aware line reader: only the current line is decoded as text.
struct Cursor<R> {
    input: R,
    line: Vec<u8>,
    policy: RawReadPolicy,
    header: Option<HeaderBudget>,
}

impl<R: BufRead> Cursor<R> {
    const fn new(input: R) -> Self {
        Self {
            input,
            line: Vec::new(),
            policy: RawReadPolicy::DEFAULT,
            header: None,
        }
    }

    const fn begin_header(&mut self) {
        self.header = Some(HeaderBudget {
            bytes: self.policy.header_bytes,
            lines: self.policy.header_lines,
        });
    }

    fn end_header(&mut self) -> usize {
        self.header
            .take()
            .map_or(0, |work| self.policy.header_bytes - work.bytes)
    }

    /// Read no more than the remaining limit plus one inspected byte, without
    /// consuming the excess or reserving capacity proportional to input size.
    fn read_line(&mut self) -> SpiceResult<bool> {
        self.line.clear();
        loop {
            let available = match self.input.fill_buf() {
                Ok(bytes) => bytes,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(input_error(&error)),
            };
            if available.is_empty() {
                return Ok(!self.line.is_empty());
            }
            if self.header.as_ref().is_some_and(|work| work.lines == 0) {
                return Err(limit("header lines per plot"));
            }
            let line_remaining = self.policy.line_bytes - self.line.len();
            let header_remaining = self.header.as_ref().map_or(usize::MAX, |work| work.bytes);
            let remaining = line_remaining.min(header_remaining);
            let inspect = available.len().min(remaining.saturating_add(1));
            let end = available[..inspect].iter().position(|&byte| byte == b'\n');
            let count = end.map_or(inspect, |index| index + 1);
            if count > remaining {
                return Err(limit(if line_remaining <= header_remaining {
                    "text line bytes"
                } else {
                    "header bytes per plot"
                }));
            }
            let needed = self.line.len() + count;
            if needed > self.line.capacity() {
                let capacity = self
                    .line
                    .capacity()
                    .saturating_mul(2)
                    .max(needed)
                    .min(self.policy.line_bytes);
                self.line
                    .try_reserve_exact(capacity - self.line.len())
                    .map_err(|error| {
                        unsupported(format!("rawfile allocation failed for text line: {error}"))
                    })?;
            }
            self.line.extend_from_slice(&available[..count]);
            if let Some(work) = &mut self.header {
                work.bytes -= count;
            }
            self.input.consume(count);
            if end.is_some() {
                return Ok(true);
            }
        }
    }

    fn is_blank(&self) -> SpiceResult<bool> {
        // Most data lines begin with ASCII whitespace followed by a number.
        // A single non-whitespace ASCII byte proves the line is not blank.
        for &byte in &self.line {
            if byte.is_ascii() {
                if !byte.is_ascii_whitespace() && byte != b'\x0b' {
                    return Ok(false);
                }
            } else {
                // Preserve the original Unicode trim semantics.
                return std::str::from_utf8(&self.line)
                    .map(|line| line.trim().is_empty())
                    .map_err(|_| unsupported("rawfile header or ASCII value is not UTF-8 text"));
            }
        }
        Ok(true)
    }

    fn next(&mut self, skip_blank: bool) -> SpiceResult<Option<&str>> {
        let mut blank_remaining = self.policy.blank_bytes;
        loop {
            if !self.read_line()? {
                return Ok(None);
            }
            if let Some(work) = &mut self.header {
                spend(&mut work.lines, 1, "header lines per plot")?;
            }
            if !skip_blank || !self.is_blank()? {
                break;
            }
            spend(
                &mut blank_remaining,
                self.line.len(),
                "consecutive blank bytes",
            )?;
        }
        let line = std::str::from_utf8(&self.line)
            .map_err(|_| unsupported("rawfile header or ASCII value is not UTF-8 text"))?;
        Ok(Some(line.strip_suffix('\n').map_or(line, |line| {
            line.strip_suffix('\r').unwrap_or(line)
        })))
    }
}

fn input_error(error: &io::Error) -> SpiceError {
    SpiceError::io("<rawfile input>", error)
}

fn parse_count(key: &str, value: &str) -> SpiceResult<usize> {
    value
        .trim()
        .parse::<usize>()
        .map_err(|_| unsupported(format!("rawfile header '{key}: {value}' is not a count")))
}

/// Reads ngspice's `padded`/`unpadded` flag, in which the last word wins.
///
/// `raw_write()` appends ` unpadded` when the `nopadding` option is set
/// (rawfile.c:120) and `raw_read()` treats `unpadded` as "the payload is not
/// rectangular" (rawfile.c:411). A padded payload is the default, and the only
/// one the binary codec accepts, because an unpadded payload's length cannot be
/// derived from the header.
fn parse_padding(flags: &str) -> bool {
    let mut padded = true;
    for token in flags.split_whitespace() {
        if token.eq_ignore_ascii_case("unpadded") {
            padded = false;
        } else if token.eq_ignore_ascii_case("padded") {
            padded = true;
        }
    }
    padded
}

/// The encoding a rawfile uses for its values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RawFormat {
    /// Text values under a `Values:` line — `set filetype=ascii`, and what
    /// `cargo xtask golden capture` records.
    Ascii,
    /// Native `double`s under a `Binary:` line — `raw_write()`'s default.
    Binary,
}

impl RawFormat {
    /// Detects the encoding from a valid first plot header without reading its payload.
    ///
    /// # Errors
    /// Invalid or missing rawfile header.
    pub fn detect(bytes: &[u8]) -> SpiceResult<Self> {
        Self::detect_with_policy(bytes, RawReadPolicy::DEFAULT)
    }

    /// Detects format using the specified header limits.
    ///
    /// # Errors
    /// Invalid, missing or over-budget first header.
    pub fn detect_with_policy(bytes: &[u8], policy: RawReadPolicy) -> SpiceResult<Self> {
        let mut cursor = Cursor::new(bytes);
        cursor.policy = policy;
        parse_header(&mut cursor)?
            .map(|(_, format)| format)
            .ok_or_else(|| unsupported("rawfile contains no plot header"))
    }

    /// True for [`RawFormat::Binary`].
    #[must_use]
    pub const fn is_binary(self) -> bool {
        matches!(self, Self::Binary)
    }
}

/// One plot's headers, up to but excluding its data section.
struct Header {
    title: String,
    date: String,
    command: String,
    plotname: String,
    flags: PlotFlags,
    /// False when the `Flags:` line carries ngspice's `unpadded`.
    padded: bool,
    point_count: usize,
    variables: Vec<Variable>,
    text_bytes: usize,
}

/// Parses one plot header, up to and including its `Values:` or `Binary:` line.
///
/// Returns `None` at the end of the file. The returned [`Section`] is the line
/// that ended the header and determines the payload decoder.
fn parse_header(cursor: &mut Cursor<impl BufRead>) -> SpiceResult<Option<(Header, RawFormat)>> {
    cursor.begin_header();
    let Some(first) = cursor.next(true)? else {
        return Ok(None);
    };

    let mut title = String::new();
    let mut date = String::new();
    let mut command = String::new();
    let mut plotname = None;
    let mut flags = None;
    let mut padded = true;
    let mut declared_variables = None;
    let mut declared_points = None;

    // Header, up to and including the `Variables:` line.
    let mut line = first;
    loop {
        let Some((key, value)) = line.split_once(':') else {
            return Err(unsupported(format!(
                "rawfile header line without a colon: '{line}'"
            )));
        };
        let key = key.trim();
        // ngspice writes exactly one space after the colon, so strip one and keep
        // the rest verbatim: `Command: ngspice-47+, Build ` really does end in a
        // space, and reproducing the file byte for byte depends on keeping it.
        let value = value.strip_prefix(' ').unwrap_or(value);
        match key {
            "Title" => value.clone_into(&mut title),
            "Date" => value.clone_into(&mut date),
            "Command" => value.clone_into(&mut command),
            "Plotname" => plotname = Some(value.to_owned()),
            "Flags" => {
                flags = Some(PlotFlags::parse(value));
                padded = parse_padding(value);
            }
            "No. Variables" => declared_variables = Some(parse_count(key, value.trim())?),
            "No. Points" => declared_points = Some(parse_count(key, value.trim())?),
            "Variables" => break,
            // A section line where a header key belongs means this plot carries
            // no header at all. The caller is already reading bytes here, so
            // the `parse_bytes` advice in `BINARY_UNSUPPORTED` would only send
            // it back to the entry point it used.
            "Binary" | "Values" => {
                return Err(unsupported(format!(
                    "rawfile plot has no header: '{key}:' before 'Variables:'"
                )));
            }
            other => {
                return Err(unsupported(format!("rawfile header key '{other}'")));
            }
        }
        let Some(next) = cursor.next(false)? else {
            return Err(unsupported("truncated rawfile: no 'Variables:' section"));
        };
        line = next;
    }

    let variable_count = declared_variables.ok_or_else(|| missing_header("No. Variables"))?;
    let point_count = declared_points.ok_or_else(|| missing_header("No. Points"))?;
    let plotname = plotname.ok_or_else(|| missing_header("Plotname"))?;
    let flags = flags.ok_or_else(|| missing_header("Flags"))?;

    if variable_count > cursor.policy.variables {
        return Err(limit("variables per plot"));
    }
    if variable_count == 0 && !super::results::valid_empty_shape(&plotname, flags, point_count) {
        return Err(unsupported(
            "rawfile declares zero variables with nonzero points",
        ));
    }
    let variables = parse_variables(cursor, variable_count)?;

    let Some(line) = cursor.next(false)? else {
        return Err(unsupported("truncated rawfile: no 'Values:' section"));
    };
    let key = line
        .split_once(':')
        .ok_or_else(|| unsupported("rawfile values section has no colon"))?
        .0
        .trim();
    let format = match key {
        "Values" => RawFormat::Ascii,
        "Binary" => RawFormat::Binary,
        other => {
            return Err(unsupported(format!(
                "rawfile values section starts with '{other}', expected 'Values'"
            )));
        }
    };

    Ok(Some((
        Header {
            title,
            date,
            command,
            plotname,
            flags,
            padded,
            point_count,
            variables,
            text_bytes: cursor.end_header(),
        },
        format,
    )))
}

fn parse_variables(
    cursor: &mut Cursor<impl BufRead>,
    variable_count: usize,
) -> SpiceResult<Vec<Variable>> {
    // A declared count is a header claim, not a promise, so the list grows as
    // the lines are read: a bogus `No. Variables:` cannot ask for a huge
    // allocation before the file has been shown to hold that many lines.
    let mut variables = Vec::new();
    for index in 0..variable_count {
        let Some(line) = cursor.next(false)? else {
            return Err(unsupported(format!(
                "truncated rawfile: variable {index} of {variable_count} is missing"
            )));
        };
        let mut fields = line.split_whitespace();
        let (Some(declared), Some(name)) = (fields.next(), fields.next()) else {
            return Err(unsupported(format!(
                "rawfile variable line with fewer than two fields: '{line}'"
            )));
        };
        let declared: usize = declared.parse().map_err(|_| {
            unsupported(format!(
                "rawfile variable index '{declared}' is not a number"
            ))
        })?;
        if declared != index {
            return Err(unsupported(format!(
                "rawfile variable index {declared} out of order, expected {index}"
            )));
        }
        reserve(&mut variables, 1, "variable definitions")?;
        variables.push(Variable::new(name, fields.next().unwrap_or("")));
    }

    Ok(variables)
}

/// Reads one rawfile sequentially from any buffered byte source.
///
/// The first header selects the format unless explicitly configured. Subsequent
/// plots must use the same format. Binary payloads never pass through UTF-8.
pub struct RawFileReader<R> {
    cursor: Cursor<R>,
    format: Option<RawFormat>,
    byte_order: BinaryByteOrder,
    blocked: bool,
    plot_count: usize,
}

impl<R: BufRead> RawFileReader<R> {
    /// Constructs a reader with automatic format detection and little-endian doubles.
    #[must_use]
    pub const fn new(input: R) -> Self {
        Self {
            cursor: Cursor::new(input),
            format: None,
            byte_order: BinaryByteOrder::LittleEndian,
            blocked: false,
            plot_count: 0,
        }
    }

    /// Set resource limits for the complete input. Configure before reading.
    #[must_use]
    pub const fn with_policy(mut self, policy: RawReadPolicy) -> Self {
        self.cursor.policy = policy;
        self
    }

    /// Requires the specified file encoding.
    #[must_use]
    pub const fn with_format(mut self, format: RawFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Selects the byte order for binary doubles.
    #[must_use]
    pub const fn with_byte_order(mut self, byte_order: BinaryByteOrder) -> Self {
        self.byte_order = byte_order;
        self
    }

    /// Reads the next plot header and lends a bounded row decoder.
    ///
    /// Finish the previous plot before requesting another.
    ///
    /// # Errors
    /// Invalid headers, mixed encodings, unsupported layouts, I/O failures or
    /// an unfinished previous plot.
    pub fn next_plot(&mut self) -> SpiceResult<Option<RawPlotRows<'_, R>>> {
        if self.blocked {
            return Err(unsupported("previous rawfile plot was not finished"));
        }
        self.blocked = true;
        let Some((header, format)) = parse_header(&mut self.cursor)? else {
            self.blocked = false;
            return Ok(None);
        };
        if self.plot_count >= self.cursor.policy.plots {
            return Err(limit("plots per input"));
        }
        self.plot_count += 1;
        if self.format.is_some_and(|expected| expected != format) {
            return Err(unsupported(if format.is_binary() {
                BINARY_UNSUPPORTED
            } else {
                "mixed rawfile encodings: a 'Values:' section cannot appear in a binary rawfile"
            }));
        }
        self.format = Some(format);
        let Header {
            title,
            date,
            command,
            plotname,
            flags,
            padded,
            point_count,
            mut variables,
            text_bytes: header_bytes,
        } = header;
        if format.is_binary() && !padded {
            return Err(unsupported(
                "binary rawfile flagged 'unpadded': its payload length cannot be derived from the header",
            ));
        }
        let columns = variables.len();
        let cells = point_count.checked_mul(columns).ok_or_else(|| {
            unsupported(format!("rawfile declares No. Variables: {columns} and No. Points: {point_count}, whose value count overflows a usize"))
        })?;
        let payload_len = if format.is_binary() {
            if flags.is_complex() {
                for variable in &mut variables {
                    variable.is_real = false;
                }
            }
            cells.checked_mul(binary::value_width(flags)).ok_or_else(|| {
                unsupported(format!("binary rawfile declares No. Variables: {columns} and No. Points: {point_count}, whose payload size overflows a usize"))
            })?
        } else {
            0
        };
        let mut realness = Vec::new();
        reserve(&mut realness, columns, "column realness")?;
        realness.extend(variables.iter().map(|v| v.is_real));
        let plot = Plot::new(plotname.clone(), plotname, flags, variables);
        Ok(Some(RawPlotRows {
            reader: self,
            header: RawPlot {
                title,
                date,
                command,
                plot,
            },
            format,
            point_count,
            header_bytes,
            next_index: 0,
            payload_len,
            consumed: 0,
            row: Vec::new(),
            realness,
            failed: false,
        }))
    }

    /// Collects every plot and its rows.
    ///
    /// # Errors
    /// Invalid or truncated rawfile data, unsupported layouts or I/O failures.
    pub fn read(mut self) -> SpiceResult<RawFile> {
        let mut plots = Vec::new();
        let mut budget = self.cursor.policy.collect;
        while let Some(mut rows) = self.next_plot()? {
            spend(
                &mut budget.header_bytes,
                rows.header_bytes,
                "collected header bytes",
            )?;
            spend(
                &mut budget.variables,
                rows.header.plot.variables().len(),
                "collected variables",
            )?;
            let mut cells = Vec::new();
            let mut point_count = 0;
            while let Some(row) = rows.next_row()? {
                spend(&mut budget.rows, 1, "collected rows")?;
                spend(&mut budget.cells, row.len(), "collected cells")?;
                reserve(&mut cells, row.len(), "collected cells")?;
                cells.extend_from_slice(row);
                point_count += 1;
            }
            let mut raw_plot = rows.finish()?;
            raw_plot.plot = raw_plot.plot.with_cells(point_count, cells)?;
            reserve(&mut plots, 1, "collected plots")?;
            plots.push(raw_plot);
        }
        Ok(RawFile { plots })
    }
}

/// A plot header and reusable row buffer borrowed from a sequential rawfile reader.
///
/// Column realness for ASCII complex plots is inferred as rows arrive. The
/// final metadata is available from [`Self::finish`].
pub struct RawPlotRows<'a, R> {
    reader: &'a mut RawFileReader<R>,
    header: RawPlot,
    format: RawFormat,
    point_count: usize,
    header_bytes: usize,
    next_index: usize,
    payload_len: usize,
    consumed: usize,
    row: Vec<Complex>,
    realness: Vec<bool>,
    failed: bool,
}

impl<R: BufRead> RawPlotRows<'_, R> {
    /// Provisional header; ASCII complex column flags are finalized by finish.
    #[must_use]
    pub const fn header(&self) -> &RawPlot {
        &self.header
    }

    /// Number of rows to decode.
    #[must_use]
    pub const fn point_count(&self) -> usize {
        self.point_count
    }

    /// Decodes the next row into a reusable buffer, without allocating per row.
    ///
    /// # Errors
    /// Malformed, out-of-order or truncated values, or I/O failure.
    pub fn next_row(&mut self) -> SpiceResult<Option<&[Complex]>> {
        if self.failed {
            return Err(unsupported("rawfile row decoder has failed"));
        }
        if self.next_index == self.point_count {
            return Ok(None);
        }
        self.row.clear();
        self.failed = true;
        reserve(&mut self.row, self.realness.len(), "streaming row")?;
        match self.format {
            RawFormat::Ascii => self.decode_ascii_row()?,
            RawFormat::Binary => self.decode_binary_row()?,
        }
        self.failed = false;
        self.next_index += 1;
        Ok(Some(&self.row))
    }

    fn decode_ascii_row(&mut self) -> SpiceResult<()> {
        let columns = self.header.plot.variables().len();
        if columns == 0 {
            return self.decode_empty_pz_row();
        }
        for (column, is_real) in self.realness.iter_mut().enumerate() {
            let Some(line) = self.reader.cursor.next(true)? else {
                return Err(unsupported(format!(
                    "truncated rawfile: expected {} point(s) of {columns} value(s)",
                    self.point_count
                )));
            };
            let (declared_point, value_text) = split_index_and_value(line);
            if column == 0
                && let Some(declared_point) = declared_point
                && declared_point != self.next_index
            {
                return Err(unsupported(format!(
                    "rawfile point index {declared_point} out of order, expected {}",
                    self.next_index
                )));
            }
            let parsed = parse_value(value_text)?;
            if self.header.plot.flags().is_complex() && !parsed.real_short_form {
                *is_real = false;
            }
            self.row.push(parsed.value);
        }
        Ok(())
    }

    fn decode_empty_pz_row(&mut self) -> SpiceResult<()> {
        // C outitf.c writes 0 + TAB, without a newline. A following Title:
        // may immediately follow the TAB, so never read a whole text line.
        // Also accept the older writer's space-prefixed index and newline.
        let cursor = &mut self.reader.cursor;
        let mut remaining = cursor.policy.line_bytes;
        let mut byte = [0];
        loop {
            spend(&mut remaining, 1, "empty pole-zero point index")?;
            cursor.input.read_exact(&mut byte).map_err(|error| {
                if error.kind() == io::ErrorKind::UnexpectedEof {
                    unsupported("truncated empty pole-zero row")
                } else {
                    input_error(&error)
                }
            })?;
            match byte[0] {
                b' ' | b'\t' => {}
                b'0' => break,
                _ => return Err(unsupported("invalid empty pole-zero point index")),
            }
        }
        spend(&mut remaining, 1, "empty pole-zero point index")?;
        cursor
            .input
            .read_exact(&mut byte)
            .map_err(|error| input_error(&error))?;
        if !byte[0].is_ascii_whitespace() {
            return Err(unsupported("invalid empty pole-zero point index separator"));
        }
        Ok(())
    }

    fn decode_binary_row(&mut self) -> SpiceResult<()> {
        let columns = self.header.plot.variables().len();
        if !binary::read_row(
            &mut self.reader.cursor.input,
            columns,
            self.header.plot.flags(),
            self.reader.byte_order,
            &mut self.consumed,
            &mut self.row,
        )
        .map_err(|error| input_error(&error))?
        {
            return Err(unsupported(format!(
                "truncated binary rawfile: No. Variables: {columns} x No. Points: {} needs {} byte(s) of data, but only {} byte(s) remain",
                self.point_count, self.payload_len, self.consumed,
            )));
        }
        Ok(())
    }

    /// Consumes any remaining rows and returns the final metadata with no points.
    ///
    /// # Errors
    /// Any malformed or truncated row or I/O failure.
    pub fn finish(mut self) -> SpiceResult<RawPlot> {
        while self.next_row()?.is_some() {}
        self.reader.blocked = false;
        let plot = &self.header.plot;
        if plot
            .variables()
            .iter()
            .zip(&self.realness)
            .any(|(v, real)| v.is_real != *real)
        {
            let mut variables = plot.variables().to_vec();
            for (v, real) in variables.iter_mut().zip(self.realness) {
                v.is_real = real;
            }
            self.header.plot = Plot::new(
                plot.name.clone(),
                plot.plotname.clone(),
                plot.flags(),
                variables,
            );
        }
        Ok(self.header)
    }
}

/// Splits the optional point index ngspice writes before the first value of each
/// point.
///
/// Values are always written with `%-.15e`, so they contain a `.` or an `e`; a
/// leading run of digits followed by whitespace is therefore an index.
fn split_index_and_value(line: &str) -> (Option<usize>, &str) {
    let trimmed = line.trim_start();
    if let Some(position) = trimmed.find(|character: char| character.is_ascii_whitespace()) {
        let head = &trimmed[..position];
        let rest = trimmed[position..].trim_start();
        if !head.is_empty()
            && !rest.is_empty()
            && head.bytes().all(|byte| byte.is_ascii_digit())
            && let Ok(index) = head.parse::<usize>()
        {
            return (Some(index), rest);
        }
    }
    (None, line.trim())
}

/// An allocation-free display of ngspice's `%-.15e` number spelling.
struct SpiceNumber(Real);

impl fmt::Display for SpiceNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Sign, leading digit, decimal point, 15 digits, 'e', exponent sign
        // and at most three exponent digits for any binary64.
        const MAX_SCIENTIFIC_LEN: usize = 1 + 1 + 1 + 15 + 1 + 1 + 3;
        let mut buffer = [0_u8; MAX_SCIENTIFIC_LEN];
        let mut remaining = buffer.as_mut_slice();
        write!(&mut remaining, "{:.15e}", self.0).map_err(|_| fmt::Error)?;
        let written = MAX_SCIENTIFIC_LEN - remaining.len();
        let text = std::str::from_utf8(&buffer[..written]).map_err(|_| fmt::Error)?;
        let Some((mantissa, exponent)) = text.split_once('e') else {
            return f.write_str(text);
        };
        let exponent = exponent.parse::<i32>().map_err(|_| fmt::Error)?;
        write!(f, "{mantissa}e{exponent:+03}")
    }
}

fn parse_value(text: &str) -> SpiceResult<ParsedValue> {
    let text = text.trim();
    match text.split_once(',') {
        Some((real, imaginary)) => Ok(ParsedValue {
            value: Complex::new(parse_real(real)?, parse_real(imaginary)?),
            real_short_form: imaginary.trim() == "0.0",
        }),
        None => Ok(ParsedValue {
            value: Complex::real(parse_real(text)?),
            real_short_form: false,
        }),
    }
}

/// One parsed value, and whether it used the `re,0.0` short form.
struct ParsedValue {
    value: Complex,
    /// True when the imaginary part was written as the literal `0.0`, which
    /// `raw_write()` only does for a vector flagged real.
    real_short_form: bool,
}

fn parse_real(text: &str) -> SpiceResult<Real> {
    text.trim()
        .parse::<Real>()
        .map_err(|_| unsupported(format!("rawfile value '{}'", text.trim())))
}

#[cfg(test)]
mod tests {
    use super::{RawFile, SpiceNumber, split_index_and_value};
    use crate::analysis::results::PlotFlags;
    use crate::primitives::Complex;

    const OP_RAWFILE: &str = "\
Title: rc divider
Date: Mon Oct  5 17:53:58 2026
Command: ngspice-47+, Build
Plotname: Operating Point
Flags: real
No. Variables: 3
No. Points: 1
Variables:
\t0\tv(in)\tvoltage
\t1\tv(out)\tvoltage
\t2\ti(v1)\tcurrent
Values:
 0\t5.000000000000000e+00
\t2.500000000000000e+00
\t-2.500000000000000e-03
";

    const AC_RAWFILE: &str = "\
Title: rc lowpass
Date: Mon Oct  5 18:00:00 2026
Command: ngspice-47+, Build
Plotname: AC Analysis
Flags: complex
No. Variables: 3
No. Points: 2
Variables:
\t0\tfrequency\tfrequency
\t1\tv(out)\tvoltage
\t2\ti(v1)\tcurrent
Values:
 0\t1.000000000000000e+02,0.000000000000000e+00
\t9.900000000000000e-01,-1.000000000000000e-02
\t-1.000000000000000e-03,1.000000000000000e-05
 1\t1.000000000000000e+03,0.000000000000000e+00
\t9.000000000000000e-01,-1.000000000000000e-01
\t-1.000000000000000e-03,1.000000000000000e-04
";

    #[test]
    fn bounded_line_reads_do_not_consume_or_allocate_the_oversize_suffix() {
        let data = vec![b'x'; 2 * 1024 * 1024];
        let mut cursor = super::Cursor::new(data.as_slice());
        cursor.policy.line_bytes = 64;
        assert!(cursor.next(false).is_err());
        assert_eq!(cursor.input.len(), data.len());
        assert!(cursor.line.capacity() <= 64);

        let mut cursor = super::Cursor::new(std::io::BufReader::with_capacity(3, data.as_slice()));
        cursor.policy.line_bytes = 63;
        assert!(cursor.next(false).is_err());
        assert!(cursor.line.len() <= 63);
        assert!(cursor.line.capacity() <= 63);

        // A line exactly at the limit is accepted, with either LF or CRLF.
        for text in ["1234567\n", "123456\r\n", "12345678"] {
            let mut cursor =
                super::Cursor::new(std::io::BufReader::with_capacity(3, text.as_bytes()));
            cursor.policy.line_bytes = 8;
            assert_eq!(cursor.next(false).unwrap(), Some(text.trim_end()));
            assert_eq!(cursor.next(false).unwrap(), None);
        }
    }

    #[test]
    fn direct_numeric_output_preserves_spice_spelling() {
        let mut bits = 0x12ab_3456_e789_cdef_u64;
        for value in [
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            0.0,
            -0.0,
            1e99,
            1e100,
            1e-99,
            1e-100,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ]
        .into_iter()
        .chain((0..4096).map(|_| {
            bits ^= bits << 13;
            bits ^= bits >> 7;
            bits ^= bits << 17;
            f64::from_bits(bits)
        })) {
            assert_eq!(
                SpiceNumber(value).to_string(),
                crate::primitives::format_spice_number(value)
            );
        }
    }

    #[test]
    fn parses_ngspice_operating_point_output() {
        let rawfile = RawFile::parse(OP_RAWFILE).expect("parses");
        assert_eq!(rawfile.len(), 1);
        let raw_plot = &rawfile.plots[0];
        assert_eq!(raw_plot.title, "rc divider");
        assert_eq!(raw_plot.plot.plotname, "Operating Point");
        assert_eq!(raw_plot.plot.flags(), PlotFlags::Real);
        assert_eq!(raw_plot.plot.variables().len(), 3);
        assert_eq!(raw_plot.plot.variables()[1].name, "v(out)");
        assert_eq!(raw_plot.plot.variables()[1].unit, "voltage");
        assert_eq!(raw_plot.plot.point_count(), 1);
        assert_eq!(raw_plot.plot.value("v(out)", 0), Some(Complex::real(2.5)));
        assert_eq!(raw_plot.plot.value("v(in)", 0), Some(Complex::real(5.0)));
        assert_eq!(
            raw_plot.plot.value("i(v1)", 0),
            Some(Complex::real(-2.5e-3))
        );
    }

    #[test]
    fn parses_complex_values_and_point_indices() {
        let rawfile = RawFile::parse(AC_RAWFILE).expect("parses");
        let plot = &rawfile.single_plot().expect("one plot");
        assert_eq!(plot.flags(), PlotFlags::Complex);
        assert_eq!(plot.point_count(), 2);
        assert_eq!(plot.value("frequency", 1), Some(Complex::new(1e3, 0.0)));
        assert_eq!(plot.value("v(out)", 1), Some(Complex::new(0.9, -0.1)));
        assert_eq!(plot.value("i(v1)", 0), Some(Complex::new(-1e-3, 1e-5)));
    }

    #[test]
    fn values_round_trip_through_the_writer() {
        for text in [OP_RAWFILE, AC_RAWFILE] {
            let parsed = RawFile::parse(text).expect("parses");
            let rendered = parsed.to_ascii().unwrap();
            let reparsed = RawFile::parse(&rendered).expect("reparses");
            assert_eq!(parsed, reparsed);
        }
    }

    #[test]
    fn the_writer_matches_ngspice_layout() {
        let rawfile = RawFile::parse(OP_RAWFILE).expect("parses");
        let rendered = rawfile.to_ascii().unwrap();
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines[0], "Title: rc divider");
        assert_eq!(lines[3], "Plotname: Operating Point");
        assert_eq!(lines[4], "Flags: real");
        assert_eq!(lines[5], "No. Variables: 3");
        assert_eq!(lines[6], "No. Points: 1");
        assert_eq!(lines[7], "Variables:");
        assert_eq!(lines[8], "\t0\tv(in)\tvoltage");
        assert_eq!(lines[11], "Values:");
        assert_eq!(lines[12], " 0\t5.000000000000000e+00");
        assert_eq!(lines[13], "\t2.500000000000000e+00");
        assert_eq!(lines[14], "\t-2.500000000000000e-03");
        // `raw_write()` emits a blank line after every point.
        assert_eq!(lines[15], "");
    }

    #[test]
    fn parses_concatenated_plots() {
        let text = format!("{OP_RAWFILE}\n{AC_RAWFILE}");
        let rawfile = RawFile::parse(&text).expect("parses");
        assert_eq!(rawfile.len(), 2);
        assert_eq!(rawfile.plots[0].plot.plotname, "Operating Point");
        assert_eq!(rawfile.plots[1].plot.plotname, "AC Analysis");
        assert!(rawfile.single_plot().is_err());
    }

    #[test]
    fn rejects_binary_rawfiles() {
        let text = OP_RAWFILE.replace("Values:", "Binary:");
        let error = RawFile::parse(&text).expect_err("binary is unsupported");
        assert!(error.to_string().contains("binary rawfile"));
    }

    #[test]
    fn rejects_truncated_and_malformed_input() {
        assert!(RawFile::parse("Title: x\n").is_err());
        assert!(RawFile::parse("nonsense\n").is_err());
        let mut text = OP_RAWFILE.to_owned();
        text.truncate(text.len() - 30);
        assert!(RawFile::parse(&text).is_err());
    }

    #[test]
    fn declared_ascii_counts_do_not_drive_allocations() {
        let overflow = OP_RAWFILE.replace("No. Points: 1", &format!("No. Points: {}", usize::MAX));
        assert!(
            RawFile::parse(&overflow)
                .unwrap_err()
                .to_string()
                .contains("overflows a usize")
        );
        // This multiplication fits, but reserving a row for every declared
        // point would overflow an allocation before reading the actual data.
        let count = usize::MAX / 3;
        let truncated = OP_RAWFILE.replace("No. Points: 1", &format!("No. Points: {count}"));
        assert!(
            RawFile::parse(&truncated)
                .unwrap_err()
                .to_string()
                .contains("truncated rawfile")
        );
    }

    #[test]
    fn streamed_count_reservation_backpatches_exactly_twenty_decimal_columns() {
        use ngspice_conformance::ngspice::Scratch;
        use std::fs;
        let dir = Scratch::new("rawfile-count-backpatch").unwrap();
        let path = dir.path().join("out.raw");
        let header = RawFile::parse(OP_RAWFILE).unwrap().plots.remove(0);
        let mut writer = super::RawFileWriter::create(
            &path,
            super::RawFormat::Ascii,
            super::BinaryByteOrder::LittleEndian,
        )
        .unwrap();
        writer.begin_plot(header.clone()).unwrap();
        assert_eq!(writer.active_point_count(), Some(0));
        for value in [0.0, 1.0, 2.0] {
            writer
                .push_point(&[
                    Complex::real(value),
                    Complex::real(2.0),
                    Complex::real(-2.0),
                ])
                .unwrap();
        }
        let finished = writer.finish_plot().unwrap();
        assert_eq!(finished.point_count, 3);
        assert!(writer.active_header().is_none());
        writer.finish().unwrap();
        let bytes = fs::read(&path).unwrap();
        let rendered = String::from_utf8(bytes).unwrap();
        let padded = format!("No. Points: {:>20}\n", 3);
        assert!(rendered.contains(&padded), "{rendered}");
        let parsed = RawFile::parse(&rendered).unwrap();
        assert_eq!(parsed.single_plot().unwrap().point_count(), 3);
        assert_eq!(super::point_count_field(usize::MAX).unwrap().len(), 20);
        assert_eq!(super::point_count_field(0).unwrap()[19], b'0');
    }

    #[test]
    fn rejects_misnumbered_variables_and_points() {
        let text = OP_RAWFILE.replace("\t1\tv(out)", "\t7\tv(out)");
        let error = RawFile::parse(&text).expect_err("index out of order");
        assert!(error.to_string().contains("variable index 7"), "{error}");

        let text = AC_RAWFILE.replace("\n 1\t1.000000000000000e+03", "\n 5\t1.000000000000000e+03");
        let error = RawFile::parse(&text).expect_err("point out of order");
        assert!(error.to_string().contains("point index 5"), "{error}");
    }

    #[test]
    fn the_index_heuristic_distinguishes_values_from_indices() {
        assert_eq!(
            split_index_and_value(" 0\t5.000000000000000e+00"),
            (Some(0), "5.000000000000000e+00")
        );
        assert_eq!(
            split_index_and_value("\t2.500000000000000e+00"),
            (None, "2.500000000000000e+00")
        );
        assert_eq!(split_index_and_value("12\t-1.0e-3"), (Some(12), "-1.0e-3"));
        // A bare number with nothing after it is a value, not an index.
        assert_eq!(split_index_and_value("5"), (None, "5"));
        assert_eq!(
            split_index_and_value("\t-2.500000000000000e-03"),
            (None, "-2.500000000000000e-03")
        );
    }

    const MIXED_RAWFILE: &str = "\
Title: noise
Date: Mon Oct  5 18:00:00 2026
Command: ngspice-47+, Build
Plotname: Noise Analysis
Flags: complex
No. Variables: 2
No. Points: 1
Variables:
\t0\tfrequency\tfrequency
\t1\tinoise_spectrum\tnoise-spectral-density
Values:
 0\t1.000000000000000e+03,0.0
\t1.000000000000000e-18,0.000000000000000e+00

";

    #[test]
    fn the_real_short_form_recovers_the_vector_flag() {
        // `raw_write()` writes `re,0.0` for a vector it has flagged real and
        // `re,im` otherwise, so the spelling in the file tells the two apart.
        let rawfile = RawFile::parse(MIXED_RAWFILE).expect("parses");
        let plot = &rawfile.plots[0].plot;
        assert!(
            plot.variables()[0].is_real,
            "frequency was written as re,0.0"
        );
        assert!(!plot.variables()[1].is_real);
        assert_eq!(rawfile.to_ascii().unwrap(), MIXED_RAWFILE);
    }

    #[test]
    fn a_real_plot_flags_every_column_real() {
        let rawfile = RawFile::parse(OP_RAWFILE).expect("parses");
        assert!(rawfile.plots[0].plot.variables().iter().all(|v| v.is_real));
    }

    #[test]
    fn an_ac_plot_flags_no_column_real() {
        // ngspice stores AC results as complex vectors, so even `frequency` is
        // written as `re,0.000000000000000e+00`.
        let rawfile = RawFile::parse(AC_RAWFILE).expect("parses");
        assert!(rawfile.plots[0].plot.variables().iter().all(|v| !v.is_real));
    }

    #[test]
    fn empty_input_parses_to_an_empty_rawfile() {
        let rawfile = RawFile::parse("\n\n").expect("parses");
        assert!(rawfile.is_empty());
        assert!(rawfile.single_plot().is_err());
    }

    #[test]
    fn validate_catches_ragged_plots() {
        let mut rawfile = RawFile::parse(OP_RAWFILE).expect("parses");
        assert!(rawfile.plots[0].plot.push_point(&[]).is_err());
        assert!(rawfile.validate().is_ok());
    }
}
