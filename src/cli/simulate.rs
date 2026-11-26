//! `spice-rs simulate` — run every analysis of a deck and write an ASCII or binary rawfile.
//!
//! C: the batch path of `src/ngspice.c` (`ngspice -b -r`: `ft_dorun()` →
//! `CKTdoJob()`) plus the writer in `src/frontend/rawfile.c` (`raw_write`). The command is deliberately narrow:
//!
//! * the deck is loaded by the ordinary [`crate::netlist::Parser`] and elaborated
//!   by the ordinary production so no device or solver logic is
//!   duplicated here;
//! * **every** analysis card runs, in ngspice batch order (see
//!   [`crate::analysis::batch`]: `.ac`, `.dc`, `.op`, `.tran`, `.noise`,
//!   same-type cards in reverse deck order), each on a freshly elaborated
//!   circuit, and each result becomes one plot of a single multi-plot rawfile
//!   in that order (`.noise` contributes its spectrum and, for a frequency
//!   range, its integrated-noise plot);
//! * a deck's `.save` cards narrow every plot, and its `.print <type>` cards
//!   narrow (and print a table for) the plots of that type only, through
//!   [`crate::analysis::selection`]; the selection is resolved against the
//!   plot's variables (the full plot, or the header a streamed `.tran`
//!   announces), so an unresolvable request fails before anything is published
//!   or printed;
//! * a deck's `.measure`/`.meas` cards are evaluated through
//!   [`crate::analysis::measure`] against the **full** last plot of their own
//!   analysis type, so an operand the output selection excluded is still
//!   measurable and a measurement never changes the written rawfile. A deck with
//!   no `.measure` card prints exactly what it printed before this work
//!   ([`Report`] carries only what was written and measured);
//! * a deck's `.four` cards are evaluated through [`crate::analysis::fourier`]
//!   against the same **full** last `.tran` plot, so a `.save`/`.print`
//!   selection never hides a transformed vector. A deck with no `.four` card
//!   prints exactly what it printed before this work, and a `.four` card never
//!   changes the written rawfile (C registers the named vectors for the
//!   transient plot instead);
//! * each scheduled output has a consumer plan. Transient, AC, DC and noise
//!   spectrum drivers produce rows directly. The sink projects rawfile rows,
//!   spools print tables and evaluates online measurements without retaining
//!   complete results. Only nonlocal objectives and ASCII plots need bounded
//!   selected history. Other drivers transfer completed plots without copying
//!   the rawfile projection when the selection is full;
//!   every successful plot is written in its scheduled order.
//! * publication is atomic: every analysis runs and every output card resolves
//!   before anything is printed or published, so one failing analysis publishes
//!   nothing (C would keep the plots that succeeded);
//! * the rawfile is written through a temporary file in the destination's
//!   directory and renamed into place, so a failed run never truncates,
//!   replaces or removes an existing destination, and never leaves a partial
//!   rawfile behind (the temporary file is removed on every failure path that
//!   can run; only process death or a failing cleanup leaves one);
//! * an unadorned `.tran` runs the companion trap/Gear driver the engine
//!   defaults to; this command never injects a `backend=`.

use std::io::Write;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::analysis::batch::{self, BoundConsumers, ObjectiveReports};
use crate::analysis::fourier::{self, FourierAnalysis};
use crate::analysis::measure::{self, Measurement};
use crate::analysis::rawfile::TextSpool;
use crate::analysis::selection::RowProjection;
use crate::analysis::{
    BinaryByteOrder, Plot, PointSink, RawFileTransaction, RawFormat, RawPlot, RunConfig,
};
use crate::netlist::eval::ParamScope;
use crate::netlist::{ParsedDeck, Parser, SourceAdmission, SourceLimits};
use crate::primitives::{AnalysisKind, Complex, SpiceError, SpiceResult};

use crate::cli::args::kind_name;

/// What one successful run produced, for the report printed afterwards.
#[derive(Debug)]
pub struct Report {
    /// The deck that was simulated.
    pub deck: PathBuf,
    /// The deck's title line, as written to the rawfile.
    pub title: String,
    /// The rawfile that was written.
    pub output: PathBuf,
    /// Encoding of the published rawfile.
    pub format: RawFormat,
    /// One entry per plot, in rawfile (ngspice batch) order.
    pub plots: Vec<PlotReport>,
    spool: Option<TextSpool>,
}

/// What one analysis of a run produced.
#[derive(Debug, Clone, PartialEq)]
pub struct PlotReport {
    /// The name ngspice gives the plot in memory, e.g. `tran1` (not written to
    /// the rawfile; see [`crate::analysis::batch`]).
    pub name: String,
    /// The analysis that ran.
    pub analysis: AnalysisKind,
    /// The rawfile's `Plotname:` header, e.g. `Transient Analysis`.
    pub plotname: String,
    /// The rawfile's variable names, in column order (after `.save`/`.print`
    /// selection, when the deck has any).
    pub variables: Vec<String>,
    /// How many points the plot has.
    pub points: usize,
    // Byte range of this plot's text in the report-owned spool.
    printed: Option<Range<u64>>,
    /// The `.measure` results, when the deck asked for any of this analysis
    /// type. `None` otherwise, which keeps the report unchanged.
    pub measured: Option<String>,
    /// The measurement results themselves, in card order.
    pub measurements: Vec<Measurement>,
    /// The `.four` block, when the deck asked for one and this is the last
    /// `.tran` plot. `None` otherwise, which keeps the report unchanged.
    pub fourier: Option<String>,
    /// The Fourier results themselves, in card order, one per transformed
    /// vector.
    pub fourier_results: Vec<FourierAnalysis>,
}

impl PlotReport {
    /// Shared metadata for materialized and streamed plots, before optional
    /// print/measure/Fourier outputs are attached.
    fn written(name: &str, analysis: AnalysisKind, plot: &Plot, points: usize) -> Self {
        Self {
            name: name.to_owned(),
            analysis,
            plotname: plot.plotname.clone(),
            variables: plot.variables().iter().map(|v| v.name.clone()).collect(),
            points,
            printed: None,
            measured: None,
            measurements: Vec::new(),
            fourier: None,
            fourier_results: Vec::new(),
        }
    }
}

/// Simulates every analysis of the deck and writes them as one ASCII rawfile.
///
/// # Errors
///
/// * [`SpiceError::Io`] when the deck cannot be read or the rawfile cannot be
///   written;
/// * [`SpiceError::Parse`] when the deck cannot be understood, or requests no
///   analysis at all;
/// * [`SpiceError::Unsupported`] when a `.print`, `.measure` or `.four` card
///   names an analysis type the deck does not run, when a `.save`/`.print`
///   request cannot be resolved against a plot (an unknown vector, an AC
///   component of a real plot), when a `.measure` request cannot be evaluated
///   against its plot (an operand the plot lacks, a window that covers no data,
///   a crossing that does not occur, …; see `docs/port/MEASURE.md`), or when a
///   `.four` request cannot be evaluated (a run shorter than one period, a
///   period with too few samples for the requested harmonics, a harmonic count
///   beyond the port's budget; see `docs/port/FOURIER.md`), and
///   [`SpiceError::Numerical`] when a computed vector, measurement or Fourier
///   value is not finite;
/// * whatever the production runner reports for unsupported devices, analyses,
///   options or numerically failed runs ([`SpiceError::Unsupported`],
///   [`SpiceError::NotYetPorted`], [`SpiceError::Numerical`], …), for **any**
///   of the deck's analyses.
///
/// Every analysis request and driver is validated, and every output card is
/// checked against the analysis types the deck runs, before the first analysis
/// starts. Plots are written into a temporary file as their analyses finish;
/// the destination is replaced only after every analysis returned a plot,
/// every selection resolved and every measurement and Fourier card was
/// evaluated, and the `.print` tables and measurement blocks are only returned
/// with the report, so a failure publishes nothing at all.
pub fn run(deck: &Path, output: &Path, auto_gnd: bool) -> SpiceResult<Report> {
    run_with_format(deck, output, auto_gnd, Some(RawFormat::Ascii))
}

/// Simulate with explicit encoding, or C's deck/environment/default precedence.
/// C: `frontend/runcoms.c::dosim`, `misc/ivars.c::init_ivars`.
///
/// # Errors
/// The same parsing, simulation and atomic output errors as [`run`].
pub fn run_with_format(
    deck: &Path,
    output: &Path,
    auto_gnd: bool,
    format: Option<RawFormat>,
) -> SpiceResult<Report> {
    run_with_admission(deck, output, auto_gnd, format, None)
}

/// Simulate a deck admitted through optional explicit source transformation.
///
/// # Errors
/// Source admission, parsing, simulation or atomic output failure.
pub fn run_with_admission(
    deck: &Path,
    output: &Path,
    auto_gnd: bool,
    format: Option<RawFormat>,
    admission: Option<&SourceAdmission>,
) -> SpiceResult<Report> {
    let (parsed, scope) = prepare_deck(deck, auto_gnd, admission)?;
    let netlist = &parsed.netlist;
    if netlist.analyses.is_empty() {
        return Err(SpiceError::parse(
            netlist.location.clone(),
            format!(
                "the deck requests no analysis: 'simulate' needs at least one {} card",
                crate::analysis::driver::DRIVERS
                    .iter()
                    .map(|kind| format!(".{}", kind.as_str()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    // Options are validated before anything runs, exactly as `parse` does:
    // unknown or unsupported settings are errors, never ignored.
    let config = RunConfig::from_netlist(netlist)?;
    let format = format.or_else(|| config.raw_format()).unwrap_or_else(|| {
        raw_format_from_env(&std::env::var("SPICE_ASCIIRAWFILE").unwrap_or_default())
    });
    // The scheduler and the execution loop share the same validated plans.
    let plans = config.plans_for(&netlist.analyses)?;
    let schedule = batch::schedule(&plans);
    let mut first_circuit = Some(config.circuit(netlist)?);
    let mut previous_circuit: Option<crate::devices::Circuit> = None;
    let routes = batch::OutputRoutes::new(
        &schedule,
        &parsed.output,
        &parsed.measurements,
        &parsed.fourier,
    )?;
    let output_plans = schedule
        .iter()
        .map(|entry| routes.plan_outputs(entry))
        .collect::<Vec<_>>();
    let date = now_header();
    // Each plot is written as soon as its analysis has finished, into a
    // temporary file that is renamed into place only after the last one.
    let mut rawfile = RawFileTransaction::create(output, format, BinaryByteOrder::LittleEndian)?;
    let mut plots = Vec::with_capacity(output_plans.iter().map(Vec::len).sum());
    let mut spool = None;
    for (entry, planned) in schedule.iter().zip(&output_plans) {
        let request = &plans[entry.card_index];
        // Each analysis starts from a freshly elaborated circuit; only what C
        // keeps across the analyses of a deck is inherited below.
        // C runs every analysis on the matrix and state vectors set up once
        // (`CKTmatrix` with its pivot order and reorder flags, `CKTstate0`),
        // so the circuit inherits them from the previous analysis.
        let mut circuit = if let Some(circuit) = first_circuit.take() {
            circuit
        } else {
            let mut circuit = config.circuit(netlist)?;
            if let Some(previous) = &previous_circuit {
                circuit.inherit_from(previous);
            }
            circuit
        };
        prepare_observations(&mut circuit, entry.kind, &parsed)?;
        let mut sink = BatchOutput {
            planned,
            next: 0,
            active: None,
            reports: &mut plots,
            title: &netlist.title,
            date: &date,
            writer: &mut rawfile,
            spool: &mut spool,
            output,
            scope: &scope,
            fourier_settings: config.fourier_settings(),
        };
        request.run_into(&mut circuit, &config.context(), &mut sink)?;
        sink.finish()?;
        circuit.note_completed(entry.kind);
        previous_circuit = Some(circuit);
    }

    if let Some(spool) = &mut spool {
        spool.flush()?;
    }
    rawfile.publish()?;
    Ok(Report {
        deck: netlist.path.clone(),
        title: netlist.title.trim().to_owned(),
        output: output.to_path_buf(),
        format,
        plots,
        spool,
    })
}

struct ActiveOutput {
    projection: RowProjection,
    consumers: BoundConsumers,
    print_start: Option<u64>,
}

/// One job's consumer: an ordered output plan, one active plot, and one
/// transaction. The driver cannot silently emit fewer or extra plots.
struct BatchOutput<'a> {
    planned: &'a [batch::PlannedOutput<'a>],
    next: usize,
    active: Option<ActiveOutput>,
    reports: &'a mut Vec<PlotReport>,
    title: &'a str,
    date: &'a str,
    writer: &'a mut RawFileTransaction,
    spool: &'a mut Option<TextSpool>,
    output: &'a Path,
    scope: &'a ParamScope,
    fourier_settings: Option<fourier::FourierSettings>,
}

fn output_error(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: "batch analysis output".into(),
        message: message.into(),
    }
}

impl BatchOutput<'_> {
    fn current(&self) -> SpiceResult<&batch::PlannedOutput<'_>> {
        self.planned
            .get(self.next)
            .ok_or_else(|| output_error("the driver produced more plots than scheduled"))
    }

    fn finish(self) -> SpiceResult<()> {
        if self.active.is_some() || self.next != self.planned.len() {
            return Err(output_error(
                "the driver did not complete the planned number of plots",
            ));
        }
        Ok(())
    }

    fn begin_text(&mut self, consumers: &BoundConsumers) -> SpiceResult<Option<u64>> {
        if !consumers.has_text() {
            return Ok(None);
        }
        if self.spool.is_none() {
            *self.spool = Some(TextSpool::create(self.output)?);
        }
        let spool = self
            .spool
            .as_mut()
            .ok_or_else(|| output_error("missing report spool"))?;
        let start = spool.position()?;
        consumers.begin_text(|text| spool.append(text))?;
        Ok(Some(start))
    }

    fn finish_consumers(
        &mut self,
        consumers: BoundConsumers,
        start: Option<u64>,
    ) -> SpiceResult<(ObjectiveReports, Option<Range<u64>>)> {
        let reports = consumers.finish(self.scope, self.fourier_settings, |text| {
            append_text(self.spool, text)
        })?;
        let printed = start
            .map(|start| -> SpiceResult<_> {
                let spool = self
                    .spool
                    .as_mut()
                    .ok_or_else(|| output_error("missing report spool"))?;
                Ok(start..spool.position()?)
            })
            .transpose()?;
        Ok((reports, printed))
    }

    fn record(
        &mut self,
        plot: &Plot,
        points: usize,
        objectives: ObjectiveReports,
        printed: Option<Range<u64>>,
    ) -> SpiceResult<()> {
        let planned = self.current()?;
        let mut report = PlotReport::written(planned.name, planned.kind, plot, points);
        report.printed = printed;
        report.measured = (!objectives.measurements.is_empty())
            .then(|| measure::to_text(&objectives.measurements));
        report.measurements = objectives.measurements;
        report.fourier =
            (!objectives.fourier.is_empty()).then(|| fourier::to_text(&objectives.fourier));
        report.fourier_results = objectives.fourier;
        self.reports.push(report);
        self.next += 1;
        Ok(())
    }
}

fn append_text(spool: &mut Option<TextSpool>, text: &str) -> SpiceResult<()> {
    spool
        .as_mut()
        .ok_or_else(|| output_error("missing report spool"))?
        .append(text)
}

impl PointSink for BatchOutput<'_> {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        if self.active.is_some() {
            return Err(output_error(
                "the driver began a plot before ending the previous one",
            ));
        }
        let planned = self.current()?;
        let selection = planned.selection(header)?;
        let raw_plot = raw_plot_for(self.title, selection.header(header)?, self.date);
        let projection = selection.bind(header)?;
        let consumers = planned.bind_consumers(header)?;
        let print_start = self.begin_text(&consumers)?;
        self.writer.begin_plot(raw_plot)?;
        self.active = Some(ActiveOutput {
            projection,
            consumers,
            print_start,
        });
        Ok(())
    }

    fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        let active = self
            .active
            .as_mut()
            .ok_or_else(|| output_error("point before plot header"))?;
        if !active.consumers.is_empty() {
            active
                .consumers
                .point(values, |text| append_text(self.spool, text))?;
        }
        self.writer.push_point(active.projection.project(values)?)
    }

    fn end(&mut self) -> SpiceResult<()> {
        let active = self
            .active
            .take()
            .ok_or_else(|| output_error("end without an active plot"))?;
        let (objectives, printed) = self.finish_consumers(active.consumers, active.print_start)?;
        let completed = self.writer.finish_plot()?;
        self.record(
            &completed.header.plot,
            completed.point_count,
            objectives,
            printed,
        )
    }

    fn complete(&mut self, plot: Plot) -> SpiceResult<()> {
        if self.active.is_some() {
            return Err(output_error(
                "completed a plot before ending the previous one",
            ));
        }
        let planned = self.current()?;
        let selection = planned.selection(&plot)?;
        let mut consumers = planned.bind_consumers(&plot.header())?;
        let start = self.begin_text(&consumers)?;
        if !consumers.is_empty() {
            for point in plot.rows() {
                consumers.point(point, |text| append_text(self.spool, text))?;
            }
        }
        let (objectives, printed) = self.finish_consumers(consumers, start)?;
        let raw_plot = raw_plot_for(self.title, selection.into_plot(plot)?, self.date);
        self.writer.write_plot(&raw_plot)?;
        self.record(
            &raw_plot.plot,
            raw_plot.plot.point_count(),
            objectives,
            printed,
        )
    }
}

fn prepare_observations(
    circuit: &mut crate::devices::Circuit,
    kind: AnalysisKind,
    parsed: &ParsedDeck,
) -> SpiceResult<()> {
    let observations = crate::analysis::observations::requests(
        kind,
        &parsed.output,
        &parsed.measurements,
        &parsed.fourier,
    );
    circuit.set_observations(&observations)?;
    if !circuit.observations().is_empty()
        && !matches!(
            kind,
            crate::primitives::AnalysisKind::OperatingPoint
                | crate::primitives::AnalysisKind::DcSweep
                | crate::primitives::AnalysisKind::Ac
                | crate::primitives::AnalysisKind::Transient
                | crate::primitives::AnalysisKind::PeriodicSteadyState
        )
    {
        return Err(SpiceError::not_yet_ported(
            "device observations for this analysis",
            "src/frontend/outitf.c (OUTpData)",
        ));
    }
    Ok(())
}

/// Prepare frontend requests and the top-level scope once. Expanding
/// subcircuit output cards already requires a resolved scope, so reuse it
/// when evaluating the root Fourier cards and measurement outputs.
fn prepare_deck(
    deck: &Path,
    auto_gnd: bool,
    admission: Option<&SourceAdmission>,
) -> SpiceResult<(ParsedDeck, Arc<ParamScope>)> {
    let mut parsed = Parser::with_auto_gnd(auto_gnd).parse_file_with_admission(
        deck,
        SourceLimits::default(),
        admission,
    )?;
    let scope = if parsed.netlist.subcircuits.iter().any(|sub| {
        !sub.output.is_empty() || !sub.measurements.is_empty() || !sub.fourier.is_empty()
    }) {
        let literal = crate::netlist::elaborate::literalize(&parsed.netlist)?;
        let expanded = crate::devices::subckt::expand_with_output(
            &literal.netlist,
            &literal.scope,
            crate::devices::subckt::SubcircuitLimits::default(),
            &parsed.output,
            &parsed.measurements,
            &parsed.fourier,
        )?;
        parsed.output = expanded.output;
        parsed.measurements = expanded.measurements;
        parsed.fourier = expanded.fourier;
        literal.scope
    } else {
        Arc::new(ParamScope::for_netlist(&parsed.netlist)?)
    };
    parsed.fourier = fourier::resolve_cards(&parsed.fourier, &scope)?;
    Ok((parsed, scope))
}

/// C uses `atoi` on `SPICE_ASCIIRAWFILE`; preserve the existing signed prefix
/// and overflow policy without allocating a second string.
fn raw_format_from_env(value: &str) -> RawFormat {
    let value = value.trim_start();
    let sign = usize::from(value.starts_with(['+', '-']));
    let digits = value[sign..].bytes().take_while(u8::is_ascii_digit).count();
    if value[..sign + digits].parse::<i64>().unwrap_or(0) != 0 {
        RawFormat::Ascii
    } else {
        RawFormat::Binary
    }
}

/// One rawfile plot: ngspice's three headers, then the data.
///
/// C treats arbitrary `Command:` headers as executable commands when loading.
/// The `ngspice-compatible` prefix makes this descriptive metadata under C's
/// simulator-prefix check (`rawfile.c::raw_read`), while naming the actual writer.
/// `title` is the deck's title line and `command` names the writer, as
/// `raw_write()` does for ngspice itself; `date` is the write time in UTC (see
/// [`now_header`]). Every plot of one run carries the same three headers.
fn raw_plot_for(title: &str, plot: crate::analysis::Plot, date: &str) -> RawPlot {
    RawPlot {
        title: title.trim().to_owned(),
        date: date.to_owned(),
        command: format!(
            "ngspice-compatible spice-rs {} (Rust port), Build",
            env!("CARGO_PKG_VERSION")
        ),
        plot,
    }
}

/// The `Date:` header for a rawfile written now: standard `ctime` spelling
/// (`Mon Oct  5 18:07:26 2026`) in **UTC**.
///
/// This is not byte-identical to ngspice's `datestring()`, which writes local
/// time and leaves an extra pre-year space (`Mon Oct  5 18:06:31  2026`); the
/// committed comparators ignore `Date:` entirely.
///
/// The port has no clock or timezone crate (and none may be added for this), so
/// the calendar fields are derived from the Unix epoch directly; a clock before
/// the epoch is reported as `unknown` instead of panicking.
fn now_header() -> String {
    format_ctime(SystemTime::now())
}

/// Renders a `ctime`-style UTC timestamp, e.g. `Thu Jan  1 00:00:00 1970`.
fn format_ctime(now: SystemTime) -> String {
    const WEEKDAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let Ok(since_epoch) = now.duration_since(UNIX_EPOCH) else {
        return "unknown".to_owned();
    };
    let seconds = since_epoch.as_secs();
    let days = seconds / 86_400;
    let time_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let Some(month_name) = usize::try_from(month - 1)
        .ok()
        .and_then(|index| MONTHS.get(index))
    else {
        return "unknown".to_owned();
    };
    // 1970-01-01 was a Thursday, index 4 in `WEEKDAYS`.
    let weekday = WEEKDAYS[((days + 4) % 7) as usize];
    format!(
        "{weekday} {month_name} {day:2} {:02}:{:02}:{:02} {year}",
        time_of_day / 3600,
        time_of_day % 3600 / 60,
        time_of_day % 60,
    )
}

/// The civil date (`year`, 1-based `month`, 1-based `day`) of a day count since
/// 1970-01-01. Howard Hinnant's `civil_from_days`, for nonnegative days.
///
/// Split the epoch offset into complete Gregorian eras and its remainder so
/// even a `u64::MAX` day count cannot overflow or wrap through a signed cast.
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    const DAYS_PER_ERA: u64 = 146_097;
    const EPOCH_OFFSET: u64 = 719_468;
    let shifted = days % DAYS_PER_ERA + EPOCH_OFFSET;
    let era = days / DAYS_PER_ERA + shifted / DAYS_PER_ERA;
    let day_of_era = shifted % DAYS_PER_ERA;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = if month_prime < 10 {
        month_prime + 3
    } else {
        month_prime - 9
    };
    let year = year_of_era + era * 400 + u64::from(month <= 2);
    (year, month, day)
}

/// Renders the report printed after a successful run.
///
/// A single-analysis deck prints exactly the report it printed before
/// multi-analysis support: the analysis, plot and variable lines, the output
/// line, then the `.print` table, the `.measure` block and the `.four` block.
/// A multi-analysis deck prints the deck, title and output lines, the plot
/// names in rawfile order, and then one section per plot (headed by its C plot
/// name, e.g. `[tran1]`) with that plot's lines and blocks.
///
/// # Errors
/// An error reading the completed report spool or writing to the destination.
pub fn write_report(report: &mut Report, to: &mut impl Write) -> SpiceResult<()> {
    use std::fmt::Write as _;

    let mut out = String::new();
    let title = if report.title.trim().is_empty() {
        "<empty title line>"
    } else {
        report.title.trim()
    };
    let _ = writeln!(out, "deck:      {}", report.deck.display());
    let _ = writeln!(out, "title:     {title}");
    if let [plot] = report.plots.as_slice() {
        plot_lines(&mut out, plot);
        let _ = writeln!(
            out,
            "output:    {} (ngspice {} rawfile)",
            report.output.display(),
            if report.format == RawFormat::Ascii {
                "ASCII"
            } else {
                "binary"
            }
        );
        report_write(to, &out)?;
        plot_blocks(to, plot, &mut report.spool)?;
    } else {
        let _ = writeln!(
            out,
            "output:    {} (ngspice {} rawfile with {} plots)",
            report.output.display(),
            if report.format == RawFormat::Ascii {
                "ASCII"
            } else {
                "binary"
            },
            report.plots.len()
        );
        let names: Vec<&str> = report.plots.iter().map(|plot| plot.name.as_str()).collect();
        let _ = writeln!(out, "plots:     {} (ngspice batch order)", names.join(" "));
        report_write(to, &out)?;
        for plot in &report.plots {
            out.clear();
            let _ = writeln!(out, "[{}]", plot.name);
            plot_lines(&mut out, plot);
            report_write(to, &out)?;
            plot_blocks(to, plot, &mut report.spool)?;
        }
    }
    to.flush()
        .map_err(|error| SpiceError::io("<report>", &error))
}

fn report_write(to: &mut impl Write, text: &str) -> SpiceResult<()> {
    to.write_all(text.as_bytes())
        .map_err(|error| SpiceError::io("<report>", &error))
}

/// The analysis, plot and variable lines of one plot.
fn plot_lines(out: &mut String, plot: &PlotReport) {
    use std::fmt::Write as _;

    let _ = writeln!(
        out,
        "analysis:  .{:<4} {}",
        plot.analysis.as_str(),
        kind_name(plot.analysis)
    );
    let _ = writeln!(
        out,
        "plot:      {} ({} variable(s), {} point(s))",
        plot.plotname,
        plot.variables.len(),
        plot.points
    );
    let _ = writeln!(out, "variables: {}", plot.variables.join(" "));
}

/// The `.print`, `.measure` and `.four` blocks of one plot, in that order.
///
/// Only a plot with an applicable `.print` card has a table, only one with
/// `.measure` results has a measurement block, and only one with `.four`
/// results has a Fourier block; a deck without any of them keeps the report
/// unchanged.
fn plot_blocks(
    to: &mut impl Write,
    plot: &PlotReport,
    spool: &mut Option<TextSpool>,
) -> SpiceResult<()> {
    if let Some(range) = &plot.printed {
        spool
            .as_mut()
            .ok_or_else(|| output_error("missing report spool"))?
            .copy_range(range.clone(), to)?;
    }
    for block in [&plot.measured, &plot.fourier].into_iter().flatten() {
        report_write(to, block)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        PlotReport, RawFormat, Report, civil_from_days, format_ctime, raw_format_from_env,
        write_report,
    };
    use crate::primitives::AnalysisKind;
    use std::path::PathBuf;
    use std::time::{Duration, UNIX_EPOCH};

    fn plot(
        analysis: AnalysisKind,
        plotname: String,
        variables: Vec<String>,
        points: usize,
        printed: Option<std::ops::Range<u64>>,
        measured: Option<String>,
    ) -> PlotReport {
        PlotReport {
            name: format!("{}1", crate::analysis::batch::plot_abbreviation(analysis)),
            analysis,
            plotname,
            variables,
            points,
            printed,
            measured,
            measurements: Vec::new(),
            fourier: None,
            fourier_results: Vec::new(),
        }
    }

    fn report_text(mut report: Report) -> String {
        let mut out = Vec::new();
        write_report(&mut report, &mut out).unwrap();
        String::from_utf8(out).unwrap()
    }

    fn attach_print(report: &mut Report, plot: usize, text: &str) {
        let path = std::env::temp_dir().join("report-test.raw");
        let mut spool = super::TextSpool::create(&path).unwrap();
        spool.append(text).unwrap();
        report.plots[plot].printed = Some(0..spool.position().unwrap());
        report.spool = Some(spool);
    }

    #[test]
    fn ascii_rawfile_environment_uses_signed_decimal_prefix() {
        for (text, expected) in [
            ("", RawFormat::Binary),
            ("0", RawFormat::Binary),
            ("+0foo", RawFormat::Binary),
            ("  -0", RawFormat::Binary),
            ("+", RawFormat::Binary),
            ("x1", RawFormat::Binary),
            ("01", RawFormat::Ascii),
            (" -12trailing", RawFormat::Ascii),
            ("+1e0", RawFormat::Ascii),
            ("1é", RawFormat::Ascii),
            ("9223372036854775808", RawFormat::Binary),
        ] {
            assert_eq!(raw_format_from_env(text), expected, "{text}");
        }
    }

    #[test]
    fn the_timestamp_is_a_ctime_style_utc_string() {
        assert_eq!(
            format_ctime(UNIX_EPOCH),
            "Thu Jan  1 00:00:00 1970",
            "the epoch is a Thursday, with a space-padded day"
        );
        assert_eq!(
            format_ctime(UNIX_EPOCH + Duration::from_secs(1_791_223_646)),
            "Mon Oct  5 18:07:26 2026"
        );
        // A clock before the epoch is reported, not panicked on.
        assert_eq!(format_ctime(UNIX_EPOCH - Duration::from_secs(1)), "unknown");
    }

    #[test]
    fn the_civil_date_conversion_handles_leap_years_and_millennia() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        assert_eq!(civil_from_days(20_677), (2026, 8, 12));
        assert_eq!(civil_from_days(365), (1971, 1, 1));
        // Each 400-year era is exactly 146097 days long, including leap
        // years. The split-epoch calculation remains valid at u64::MAX.
        let (year, month, day) = civil_from_days(u64::MAX - 146_097);
        assert_eq!(civil_from_days(u64::MAX), (year + 400, month, day));
        assert!((1..=12).contains(&month));
        assert!((1..=31).contains(&day));
    }

    #[test]
    fn the_report_names_the_deck_analysis_plot_and_output() {
        let report = Report {
            deck: PathBuf::from("rc.cir"),
            title: "RC divider".to_owned(),
            output: PathBuf::from("out.raw"),
            format: crate::analysis::RawFormat::Ascii,
            spool: None,
            plots: vec![plot(
                AnalysisKind::OperatingPoint,
                "Operating Point".to_owned(),
                vec!["v(in)".to_owned(), "v(out)".to_owned()],
                1,
                None,
                None,
            )],
        };
        let text = report_text(report);
        assert!(text.contains("deck:      rc.cir"), "{text}");
        assert!(text.contains("title:     RC divider"), "{text}");
        assert!(
            text.contains("analysis:  .op   DC operating point"),
            "{text}"
        );
        assert!(
            text.contains("plot:      Operating Point (2 variable(s), 1 point(s))"),
            "{text}"
        );
        assert!(text.contains("variables: v(in) v(out)"), "{text}");
        assert!(text.contains("output:    out.raw"), "{text}");
        assert!(
            !text.contains("print:"),
            "a deck without .print has no table: {text}"
        );
        assert!(
            !text.contains("measure:"),
            "a deck without .measure has no measurement block: {text}"
        );
    }

    #[test]
    fn a_print_table_is_appended_after_the_report() {
        let mut report = Report {
            deck: PathBuf::from("rc.cir"),
            title: "RC divider".to_owned(),
            output: PathBuf::from("out.raw"),
            format: crate::analysis::RawFormat::Ascii,
            spool: None,
            plots: vec![plot(
                AnalysisKind::OperatingPoint,
                "Operating Point".to_owned(),
                vec!["v(out)".to_owned()],
                1,
                None,
                Some("measure: 1 result(s)\n".to_owned()),
            )],
        };
        attach_print(&mut report, 0, "print: 1 vector(s): v(out)\nvalues: real\n");
        let text = report_text(report);
        let report_end = text.find("output:    out.raw").expect("the report");
        let table = text.find("print: 1 vector(s): v(out)").expect("the table");
        let measured = text.find("measure: 1 result(s)").expect("the block");
        assert!(report_end < table, "the table follows the report: {text}");
        assert!(table < measured, "the block follows the table: {text}");
    }

    #[test]
    fn an_empty_title_is_named_like_the_summary_does() {
        let report = Report {
            deck: PathBuf::from("rc.cir"),
            title: String::new(),
            output: PathBuf::from("out.raw"),
            format: crate::analysis::RawFormat::Ascii,
            spool: None,
            plots: vec![plot(
                AnalysisKind::Transient,
                "Transient Analysis".to_owned(),
                Vec::new(),
                0,
                None,
                None,
            )],
        };
        assert!(report_text(report).contains("title:     <empty title line>"));
    }

    #[test]
    fn a_multi_plot_report_has_one_section_per_plot_in_rawfile_order() {
        let mut tran = plot(
            AnalysisKind::Transient,
            "Transient Analysis".to_owned(),
            vec!["time".to_owned(), "v(out)".to_owned()],
            70,
            None,
            Some("measure: 1 result(s)\n".to_owned()),
        );
        tran.name = "tran2".to_owned();
        let mut ac = plot(
            AnalysisKind::Ac,
            "AC Analysis".to_owned(),
            vec!["frequency".to_owned(), "v(out)".to_owned()],
            3,
            None,
            None,
        );
        ac.name = "ac1".to_owned();
        let mut report = Report {
            deck: PathBuf::from("rc.cir"),
            title: "RC".to_owned(),
            output: PathBuf::from("out.raw"),
            format: crate::analysis::RawFormat::Ascii,
            spool: None,
            plots: vec![ac, tran],
        };
        attach_print(&mut report, 0, "print: 1 vector(s): vm(out)\n");
        let text = report_text(report);
        assert!(
            text.contains("output:    out.raw (ngspice ASCII rawfile with 2 plots"),
            "{text}"
        );
        assert!(
            text.contains("plots:     ac1 tran2 (ngspice batch order)"),
            "{text}"
        );
        let ac = text.find("[ac1]\nanalysis:  .ac ").expect("the ac section");
        let table = text.find("print: 1 vector(s): vm(out)").expect("the table");
        let tran = text
            .find("[tran2]\nanalysis:  .tran")
            .expect("the tran section");
        let measured = text.find("measure: 1 result(s)").expect("the block");
        assert!(ac < table && table < tran && tran < measured, "{text}");
    }
}
