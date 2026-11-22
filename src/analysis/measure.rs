//! `.measure`/`.meas` evaluation over an accepted plot (GitHub #43).
//!
//! `.measure` is post-processing: after a run, the cards are evaluated against
//! the **full** plot the driver produced, before (and independently of) the
//! `.save`/`.print` selection that narrows what is written. An operand the
//! output selection dropped is still measurable, and a deck with a `.measure`
//! card writes exactly the rawfile it would have written without one.
//!
//! C: `get_measure2()` and its helpers in `src/frontend/com_measure2.c`,
//! driven by `do_measure()` (`src/frontend/measure.c`). The bounded grammar is
//! parsed by `netlist` (`crates/netlist/src/parser/measure.rs`); the
//! supported subset, the axis/interpolation/crossing/window rules and every
//! divergence from C are documented in `docs/port/MEASURE.md`.
//!
//! The layer is a pure function of a [`Plot`] plus the deck's
//! [`MeasureCard`]s, so it is unit-tested over hand-built plots with analytic
//! results.
//!
//! # Failure policy
//!
//! A measurement that cannot be computed is an error, never a fabricated
//! `NaN`/`0`:
//!
//! * [`SpiceError::Unsupported`], positioned at the card, when the request
//!   cannot be honoured by this plot (an operand the plot lacks, a window that
//!   covers no data or has no width, an `AT=` outside the axis, a crossing that
//!   does not occur, a discontinuity at an exact query point, a descending or
//!   nested axis);
//! * [`SpiceError::Numerical`] when the plot's own values do not permit the
//!   arithmetic (a non-finite operand value at a point the measurement uses, or
//!   a non-finite result).

use crate::netlist::ast::{
    MeasureCard, MeasureEvent, MeasureRequest, MeasureStatistic, MeasureTransition, MeasureWindow,
    VectorRequest,
};
use crate::primitives::{AnalysisKind, Real, SpiceError, SpiceResult};

use super::ordered_samples::{AxisError, OrderedAxis, SampleError, SampleWindow, column_names};
use crate::analysis::results::Plot;
use crate::analysis::selection::{self, Column};

/// One named measurement result.
#[derive(Debug, Clone, PartialEq)]
pub struct Measurement {
    /// The result name from the card, as written.
    pub name: String,
    /// The measured value.
    pub value: Real,
    /// The value's unit: the operand's unit for `FIND`/`MIN`/`MAX`/`AVG`/`RMS`
    /// (`voltage`, `current`, `phase`, `db`), `<operand>*<axis>` for `INTEG`,
    /// and the axis' unit (`time`, `frequency`) for a `TRIG`/`TARG` distance.
    pub unit: String,
    /// The axis value a single-position result sits at: the `AT=` query
    /// position for `FIND`, and the position of the extremum for `MIN`/`MAX`
    /// (C prints both as `at=`).
    pub at: Option<Real>,
    /// The axis window a whole-window statistic covered, as C echoes with
    /// `from=`/`to=`. The covered window is the requested window clipped to the
    /// axis, including the interpolated boundaries (see [`MeasureSpan`]).
    pub window: Option<MeasureSpan>,
    /// The trig and target axis positions of a `TRIG`/`TARG` measurement.
    pub events: Option<MeasureEvents>,
}

/// The axis window a whole-window statistic covered, in ascending order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureSpan {
    /// The window's lower bound on the axis.
    pub from: Real,
    /// The window's upper bound on the axis.
    pub to: Real,
}

/// The two axis positions of a `TRIG`/`TARG` measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasureEvents {
    /// The trigger event's axis position.
    pub trig: Real,
    /// The target event's axis position.
    pub targ: Real,
}

/// Evaluates every card against `plot`, in card order.
///
/// Returns an empty list for a deck without a `.measure` card, so a caller that
/// prints only when the list is non-empty leaves today's output unchanged.
///
/// # Errors
///
/// * [`SpiceError::Unsupported`], positioned at the card, for a card naming
///   another analysis than `kind`, for a run whose plot has no measurement axis
///   (`op` and the analysis kinds without a driver), and for every measurement
///   that cannot be computed (see the module docs);
/// * [`SpiceError::Numerical`] for a non-finite operand value or result;
/// * whatever [`selection::resolve_request`] reports for an operand the plot
///   cannot resolve.
pub fn resolve(
    plot: &Plot,
    kind: AnalysisKind,
    cards: &[MeasureCard],
) -> SpiceResult<Vec<Measurement>> {
    resolve_with_scope(
        plot,
        kind,
        cards,
        &crate::netlist::eval::ParamScope::root(&[])?,
    )
}

/// Evaluates vector measurements first, then PARAM/EXPR cards in source order.
/// C: `measure.c::do_measure` uses two passes and publishes scalar results to numparam.
///
/// # Errors
/// As [`resolve`], plus undefined or invalid scalar expressions.
pub fn resolve_with_scope(
    plot: &Plot,
    kind: AnalysisKind,
    cards: &[MeasureCard],
    scope: &crate::netlist::eval::ParamScope,
) -> SpiceResult<Vec<Measurement>> {
    resolve_refs_with_scope(plot, kind, cards.iter(), scope)
}

pub(super) fn resolve_refs_with_scope<'a>(
    plot: &Plot,
    kind: AnalysisKind,
    cards: impl Iterator<Item = &'a MeasureCard> + Clone,
    scope: &crate::netlist::eval::ParamScope,
) -> SpiceResult<Vec<Measurement>> {
    let count = bounded_count(cards.clone())?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let axis = Axis::of(plot, kind)?;
    resolve_ordered(kind, cards, scope, |_, card| evaluate(plot, &axis, card))
}

/// Limit the number of output records before binding or evaluating operands.
pub(super) fn bounded_count<'a>(
    cards: impl Iterator<Item = &'a MeasureCard>,
) -> SpiceResult<usize> {
    let limit = crate::netlist::eval::EvalBudget::default()
        .limits()
        .definitions;
    let mut scan = cards;
    let count = scan.by_ref().take(limit).count();
    if let Some(card) = scan.next() {
        return Err(unsupported(
            card,
            format!("measurement record limit of {limit} exceeded"),
        ));
    }
    Ok(count)
}

/// Shared two-pass sequential binding environment for retained and streamed
/// measurements. Vector results are supplied at their original card index.
/// Deferred numeric setters are resolved only after earlier results exist.
pub(super) fn resolve_ordered<'a>(
    kind: AnalysisKind,
    cards: impl Iterator<Item = &'a MeasureCard> + Clone,
    scope: &crate::netlist::eval::ParamScope,
    mut vector: impl FnMut(usize, &MeasureCard) -> SpiceResult<Measurement>,
) -> SpiceResult<Vec<Measurement>> {
    let count = bounded_count(cards.clone())?;
    let mut budget = crate::netlist::eval::EvalBudget::default();
    // Evaluation order is two-pass, but each result occupies its source-order
    // slot immediately. No result sort or chain of lexical scopes is needed.
    let mut results = Vec::new();
    results
        .try_reserve_exact(count)
        .map_err(|error| SpiceError::circuit(format!("allocating measurement records: {error}")))?;
    results.resize_with(count, || None);
    let mut scope = crate::netlist::eval::SequentialBindings::new(scope, count);
    for (index, card) in cards
        .clone()
        .enumerate()
        .filter(|(_, card)| !matches!(card.request, MeasureRequest::Parameter(_)))
        .chain(
            cards
                .enumerate()
                .filter(|(_, card)| matches!(card.request, MeasureRequest::Parameter(_))),
        )
    {
        if card.analysis != kind {
            return Err(unsupported(
                card,
                format!(
                    ".measure {} {}: the card names a .{} measurement; this plot is .{}, \
                     so the card can never be honoured against it",
                    card.analysis.as_str(),
                    card.name,
                    card.analysis.as_str(),
                    kind.as_str()
                ),
            ));
        }
        // Only deferred numeric setter expressions need a newly parsed card.
        // All other cards are already resolved and can be borrowed as-is.
        let resolved;
        let card = if matches!(card.request, MeasureRequest::Deferred { .. }) {
            resolved = crate::netlist::parser::resolve_measure_card(card, &scope, &mut budget)?;
            &resolved
        } else {
            card
        };
        let result = if let MeasureRequest::Parameter(expression) = &card.request {
            Measurement {
                name: card.name.clone(),
                value: scope.evaluate(expression, &mut budget)?,
                unit: "scalar".into(),
                at: None,
                window: None,
                events: None,
            }
        } else {
            vector(index, card)?
        };
        scope.publish(&card.name, result.value, &card.name_location)?;
        results[index] = Some(result);
    }
    results
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| SpiceError::circuit("measurement result missing after evaluation"))
}

/// Renders results as the bounded text block the CLI appends to its report.
///
/// The shape follows C's `do_measure()` printout (`%-20s=  %.*e` plus the
/// operation's own fields), with the port's 15-fractional-digit spelling. The
/// `unit` of a [`Measurement`] is part of the API, not of this block, exactly
/// as in C.
#[must_use]
pub fn to_text(results: &[Measurement]) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "measure: {} result(s), evaluated on the full plot before any .save/.print selection",
        results.len()
    );
    for result in results {
        let _ = write!(
            out,
            "{:<20}=  {}",
            result.name,
            crate::primitives::format_spice_number(result.value)
        );
        if let Some(events) = &result.events {
            let _ = write!(
                out,
                " targ=  {} trig=  {}",
                crate::primitives::format_spice_number(events.targ),
                crate::primitives::format_spice_number(events.trig)
            );
        }
        if let Some(at) = result.at {
            let _ = write!(out, " at=  {}", crate::primitives::format_spice_number(at));
        }
        if let Some(window) = &result.window {
            let _ = write!(
                out,
                " from=  {} to=  {}",
                crate::primitives::format_spice_number(window.from),
                crate::primitives::format_spice_number(window.to)
            );
        }
        let _ = writeln!(out);
    }
    out
}

/// A positioned failure for a request this plot cannot answer.
fn unsupported(card: &MeasureCard, feature: String) -> SpiceError {
    SpiceError::Unsupported {
        feature,
        location: Some(card.location.clone()),
    }
}

/// A non-finite value the measurement would have to use.
fn nonfinite(card: &MeasureCard, context: &str, message: String) -> SpiceError {
    SpiceError::Numerical {
        context: format!("measure {}: {context}", card.name),
        message,
    }
}

/// The scale vectors a DC sweep plot may carry, in C's lookup order.
const DC_SCALES: [&str; 5] = [
    "v(v-sweep)",
    "i(i-sweep)",
    "temp-sweep",
    "res-sweep",
    "v(param-sweep)",
];

pub(super) mod online;

/// The ordered physical axis of a plot: the driver's scale vector.
///
/// The companion transient driver keeps every accepted time point and lands
/// exactly on every source breakpoint, so consecutive samples never straddle a
/// source discontinuity (`src/analysis/companion.rs`, "Output"). This module therefore interpolates only between consecutive
/// samples, and treats two samples that share an axis value as a jump (see
/// [`value_at`] and [`crossing`]).
pub(super) struct Axis {
    /// The axis' column name, e.g. `time`.
    name: &'static str,
    /// One physical value per plot point.
    values: OrderedAxis<'static>,
}

/// Physical measurement scale identity, independent of retained rows.
pub(super) fn axis_name(plot: &Plot, kind: AnalysisKind) -> SpiceResult<&'static str> {
    let name = match kind {
        AnalysisKind::Transient => "time",
        AnalysisKind::Ac | AnalysisKind::SParameter => "frequency",
        // com_measure2.c tries v-sweep, i-sweep, temp-sweep, res-sweep in
        // that order. C stops there, so a `.dc @dev[param]` plot has no
        // measurable scale; the port also accepts its param-sweep.
        AnalysisKind::DcSweep => DC_SCALES
            .into_iter()
            .find(|name| plot.variable_index(name).is_some())
            .unwrap_or(DC_SCALES[0]),
        other => {
            return Err(SpiceError::Unsupported {
                feature: format!(
                    "a .measure result needs a .tran, .dc or .ac plot with an ordered axis; \
                         this run is .{} (docs/port/MEASURE.md)",
                    other.as_str()
                ),
                location: None,
            });
        }
    };
    Ok(name)
}

impl Axis {
    pub(super) fn of(plot: &Plot, kind: AnalysisKind) -> SpiceResult<Self> {
        let name = axis_name(plot, kind)?;
        let values = OrderedAxis::from_plot(plot, name).map_err(|error| match error {
            AxisError::MissingColumn => SpiceError::Unsupported {
                feature: format!("the {} plot carries no '{name}' axis vector; it carries {}", plot.plotname, column_names(plot)),
                location: None,
            },
            AxisError::Nonfinite(index) => SpiceError::Numerical {
                context: format!("measure axis {name}"), message: format!("point {index} is not finite"),
            },
            AxisError::Descending(index) => SpiceError::Unsupported {
                feature: format!("the {name} axis is not non-decreasing (it descends at point {index}); a measurement needs an ordered axis, so a descending or nested .dc sweep is not measurable"),
                location: None,
            },
        })?;
        Ok(Self { name, values })
    }

    fn first(&self) -> Real {
        self.values.as_slice().first().copied().unwrap_or(0.0)
    }

    fn last(&self) -> Real {
        self.values.as_slice().last().copied().unwrap_or(0.0)
    }
}

/// The requested window, clipped to the axis and validated.
#[derive(Debug, Clone, Copy)]
struct Window {
    from: Real,
    to: Real,
}

impl Window {
    fn resolve(window: &MeasureWindow, axis: &Axis, card: &MeasureCard) -> SpiceResult<Self> {
        let range = axis
            .values
            .as_slice()
            .first()
            .copied()
            .zip(axis.values.as_slice().last().copied());
        Self::from_range(window, range, axis.name, card)
    }

    fn from_range(
        window: &MeasureWindow,
        range: Option<(Real, Real)>,
        axis: &str,
        card: &MeasureCard,
    ) -> SpiceResult<Self> {
        let Some((first, last)) = range else {
            return Err(unsupported(
                card,
                format!(".measure {}: the plot has no points", card.name),
            ));
        };
        let from = window.from.unwrap_or(first);
        let to = window.to.unwrap_or(last);
        if !from.is_finite() || !to.is_finite() {
            return Err(unsupported(
                card,
                "measurement window bounds must be finite".into(),
            ));
        }
        if from > to {
            return Err(unsupported(
                card,
                format!(
                    ".measure {}: FROM= is above TO= (a {axis} window must ascend; C swaps an \
                     inverted window for a .dc measurement)",
                    card.name
                ),
            ));
        }
        if to < first || from > last {
            return Err(unsupported(
                card,
                format!(
                    ".measure {}: the window [{}, {}] lies outside the {axis} range [{}, {}]",
                    card.name,
                    crate::primitives::format_spice_number(from),
                    crate::primitives::format_spice_number(to),
                    axis,
                    crate::primitives::format_spice_number(first),
                    crate::primitives::format_spice_number(last)
                ),
            ));
        }
        Ok(Self {
            from: from.max(first),
            to: to.min(last),
        })
    }

    /// The index range of the samples inside the window, inclusive.
    fn samples<'a>(&self, axis: &'a Axis, card: &MeasureCard) -> SpiceResult<SampleWindow<'a>> {
        axis.values
            .sample_window(self.from, self.to)
            .ok_or_else(|| {
                unsupported(
                    card,
                    format!(
                        ".measure {}: the window [{}, {}] covers no {} sample",
                        card.name,
                        crate::primitives::format_spice_number(self.from),
                        crate::primitives::format_spice_number(self.to),
                        axis.name
                    ),
                )
            })
    }

    /// The window's samples, with both boundaries interpolated exactly.
    ///
    /// A boundary strictly inside a bracket is interpolated, as C's
    /// Enhancement-302 does, so `AVG` over `[from, to]` equals
    /// `INTEG/(to - from)` exactly. The bracket always has positive width here:
    /// a zero-width bracket is a jump and its value *is* the boundary.
    fn clipped<'a>(
        &self,
        axis: &'a Axis,
        operand: &'a Operand<'_>,
        card: &MeasureCard,
    ) -> SpiceResult<impl Iterator<Item = SpiceResult<(Real, Real)>> + 'a> {
        Ok(self
            .samples(axis, card)?
            .clipped_samples(move |index| operand.value(index)))
    }
}

/// One measurement operand: the column a `.measure` vector resolves to, plus the
/// result name, so that a value the measurement cannot use is reported as a
/// measurement failure rather than as an output-selection one.
struct Operand<'a> {
    plot: &'a Plot,
    name: &'a str,
    column: Column,
}

impl<'a> Operand<'a> {
    fn resolve(plot: &'a Plot, request: &VectorRequest, name: &'a str) -> SpiceResult<Self> {
        Ok(Self {
            plot,
            name,
            column: selection::resolve_request(plot, request)?,
        })
    }

    /// The operand's unit, e.g. `voltage`.
    fn unit(&self) -> &str {
        self.column.unit()
    }

    /// The operand's value at one plot point.
    fn value(&self, index: usize) -> SpiceResult<Real> {
        let Some(point) = self.plot.row(index) else {
            return Err(SpiceError::Numerical {
                context: format!("measure {}", self.name),
                message: format!("the plot has no point {index}"),
            });
        };
        self.column
            .value(point)
            .map_err(|error| match error {
                SpiceError::Numerical { message, .. } => SpiceError::Numerical {
                    context: format!("measure {}: {}", self.name, self.column.name()),
                    message,
                },
                other => other,
            })
            .map(|value| value.re)
    }
}

pub(super) fn evaluate(plot: &Plot, axis: &Axis, card: &MeasureCard) -> SpiceResult<Measurement> {
    match &card.request {
        MeasureRequest::Deferred { .. } | MeasureRequest::Parameter(_) => Err(unsupported(
            card,
            "scalar measurement requires the two-pass resolver".into(),
        )),
        MeasureRequest::AtEvent {
            operand,
            event,
            derivative,
            window,
        } => {
            let window = Window::resolve(window, axis, card)?;
            let at = event_at(plot, axis, event, &window, card)?;
            let operand = Operand::resolve(plot, operand, &card.name)?;
            let value = if *derivative {
                derivative_at(axis, &operand, at, card)?
            } else {
                value_at(axis, &operand, at, card)?
            };
            Ok(Measurement {
                name: card.name.clone(),
                value,
                unit: if *derivative {
                    format!("{}/{}", operand.unit(), axis.name)
                } else {
                    operand.unit().to_owned()
                },
                at: Some(at),
                window: None,
                events: None,
            })
        }
        MeasureRequest::When { event, window } => {
            let window = Window::resolve(window, axis, card)?;
            let value = event_at(plot, axis, event, &window, card)?;
            Ok(Measurement {
                name: card.name.clone(),
                value,
                unit: axis.name.to_owned(),
                at: None,
                window: None,
                events: None,
            })
        }
        MeasureRequest::Find {
            operand,
            at,
            window,
            ..
        } => evaluate_find(plot, axis, card, operand, *at, window),
        MeasureRequest::Statistic {
            statistic,
            operand,
            window,
        } => evaluate_statistic(plot, axis, card, *statistic, operand, window),
        MeasureRequest::TrigTarg { trig, targ, window } => {
            let window = Window::resolve(window, axis, card)?;
            let trig = event_at(plot, axis, trig, &window, card)?;
            let targ = event_at(plot, axis, targ, &window, card)?;
            let value = targ - trig;
            if !value.is_finite() {
                return Err(nonfinite(
                    card,
                    "TRIG/TARG",
                    "the computed distance is not finite".to_owned(),
                ));
            }
            Ok(Measurement {
                name: card.name.clone(),
                value,
                unit: axis.name.to_owned(),
                at: None,
                window: None,
                events: Some(MeasureEvents { trig, targ }),
            })
        }
    }
}

fn evaluate_statistic(
    plot: &Plot,
    axis: &Axis,
    card: &MeasureCard,
    statistic: MeasureStatistic,
    operand: &VectorRequest,
    window: &MeasureWindow,
) -> SpiceResult<Measurement> {
    let operand = Operand::resolve(plot, operand, &card.name)?;
    let window = Window::resolve(window, axis, card)?;
    let (value, at, span) = match statistic {
        MeasureStatistic::PhaseMargin | MeasureStatistic::GainMargin => {
            let (value, at) = margin(
                axis,
                &operand,
                &window,
                statistic == MeasureStatistic::PhaseMargin,
                card,
            )?;
            (value, Some(at), None)
        }
        MeasureStatistic::MinAt
        | MeasureStatistic::MaxAt
        | MeasureStatistic::PeakToPeak
        | MeasureStatistic::Min
        | MeasureStatistic::Max => {
            let (value, at) = extrema(axis, &operand, &window, card)?.result(statistic)?;
            (value, at, None)
        }
        MeasureStatistic::Avg | MeasureStatistic::Rms | MeasureStatistic::Integ => {
            let span = MeasureSpan {
                from: window.from,
                to: window.to,
            };
            // Every one of these three is an integral over the covered
            // window; a window without width has no mean and is a
            // failure rather than a division by zero.
            let width = span_width(span, card)?;
            let points = window.clipped(axis, &operand, card)?;
            let squared = statistic == MeasureStatistic::Rms;
            let area = integrate(points, statistic, card)?;
            let mean = if statistic == MeasureStatistic::Integ {
                area
            } else {
                area / width
            };
            let value = if squared { mean.sqrt() } else { mean };
            (value, None, Some(span))
        }
    };
    if !value.is_finite() {
        return Err(nonfinite(
            card,
            statistic.name(),
            "the computed result is not finite".to_owned(),
        ));
    }
    Ok(Measurement {
        name: card.name.clone(),
        value,
        unit: statistic_unit(statistic, operand.unit(), axis.name),
        at,
        window: span,
        events: None,
    })
}

fn evaluate_find(
    plot: &Plot,
    axis: &Axis,
    card: &MeasureCard,
    operand: &VectorRequest,
    at: Real,
    window: &MeasureWindow,
) -> SpiceResult<Measurement> {
    let operand = Operand::resolve(plot, operand, &card.name)?;
    let window = Window::resolve(window, axis, card)?;
    check_range(axis, at, card, "AT=")?;
    if at < window.from || at > window.to {
        return Err(unsupported(
            card,
            format!(
                ".measure {}: AT={} is outside the measurement window [{}, {}]",
                card.name,
                crate::primitives::format_spice_number(at),
                crate::primitives::format_spice_number(window.from),
                crate::primitives::format_spice_number(window.to)
            ),
        ));
    }
    Ok(Measurement {
        name: card.name.clone(),
        value: value_at(axis, &operand, at, card)?,
        unit: operand.unit().to_owned(),
        at: Some(at),
        window: None,
        events: None,
    })
}

/// The operand's value at one physical axis value.
///
/// Interpolation is linear between the samples bracketing `x`; the bracket has
/// positive width, so it never spans a source discontinuity (the drivers keep a
/// sample on every breakpoint). An axis value carried by two samples is a jump:
/// asking for its value is ambiguous, so it is an explicit failure rather than
/// a silent choice between the left and right limits (C may choose a limit or
/// report "out of interval" depending on the sample layout).
fn value_at(axis: &Axis, operand: &Operand<'_>, x: Real, card: &MeasureCard) -> SpiceResult<Real> {
    check_range(axis, x, card, "AT=")?;
    axis.values.linear_sample(x, |i| operand.value(i)).map_err(|error| match error {
        SampleError::Discontinuous => unsupported(card, format!(
            ".measure {}: the operand is discontinuous at {} = {}: the plot has multiple values at that {} value, so the value there is not defined",
            card.name, axis.name, crate::primitives::format_spice_number(x), axis.name)),
        SampleError::OutOfRange => unsupported(card, "AT= is outside the measurement axis".into()),
        SampleError::Nonfinite => nonfinite(card, "AT=", "nonfinite interpolated value".into()),
        SampleError::Operand(error) => error,
    })
}

/// Rejects an axis value outside the plot's own range.
fn check_range(axis: &Axis, x: Real, card: &MeasureCard, what: &str) -> SpiceResult<()> {
    if axis.values.is_empty() {
        return Err(unsupported(
            card,
            format!(".measure {}: the plot has no points", card.name),
        ));
    }
    if !x.is_finite() || x < axis.first() || x > axis.last() {
        return Err(unsupported(
            card,
            format!(
                ".measure {}: {what}{} is outside the {} range [{}, {}] (out of interval)",
                card.name,
                crate::primitives::format_spice_number(x),
                axis.name,
                crate::primitives::format_spice_number(axis.first()),
                crate::primitives::format_spice_number(axis.last())
            ),
        ));
    }
    Ok(())
}

/// Extremal sample values and their axis locations. A peak-to-peak
/// measurement gets both extrema in one traversal.
struct Extrema {
    min: (Real, Real),
    max: (Real, Real),
}

impl Extrema {
    const fn new(value: Real, location: Real) -> Self {
        Self {
            min: (value, location),
            max: (value, location),
        }
    }

    const fn include(&mut self, value: Real, location: Real) {
        if value <= self.min.0 {
            self.min = (value, location);
        }
        if value >= self.max.0 {
            self.max = (value, location);
        }
    }

    fn result(&self, statistic: MeasureStatistic) -> SpiceResult<(Real, Option<Real>)> {
        Ok(match statistic {
            MeasureStatistic::MinAt => (self.min.1, None),
            MeasureStatistic::MaxAt => (self.max.1, None),
            MeasureStatistic::PeakToPeak => (self.max.0 - self.min.0, None),
            MeasureStatistic::Min => (self.min.0, Some(self.min.1)),
            MeasureStatistic::Max => (self.max.0, Some(self.max.1)),
            _ => return Err(SpiceError::circuit("invalid extrema statistic")),
        })
    }
}

fn extrema(
    axis: &Axis,
    operand: &Operand<'_>,
    window: &Window,
    card: &MeasureCard,
) -> SpiceResult<Extrema> {
    let samples = window.samples(axis, card)?;
    let (first, location) = samples.first_sample();
    let mut extrema = Extrema::new(operand.value(first)?, location);
    for (index, location) in samples.raw_samples().skip(1) {
        extrema.include(operand.value(index)?, location);
    }
    Ok(extrema)
}

/// Shared incremental area reducer. AVG requires one interval; the greedy
/// Simpson policies require at most four points and one overlapping endpoint.
struct Integral {
    panel: [(Real, Real); 4],
    count: usize,
    area: Real,
    statistic: MeasureStatistic,
}

impl Integral {
    const fn new(statistic: MeasureStatistic) -> Self {
        Self {
            panel: [(0., 0.); 4],
            count: 0,
            area: 0.,
            statistic,
        }
    }

    fn push(&mut self, point: (Real, Real)) {
        self.panel[self.count] = point;
        self.count += 1;
        if (self.statistic == MeasureStatistic::Avg && self.count == 2) || self.count == 4 {
            self.consume();
        }
    }

    #[expect(
        clippy::suboptimal_flops,
        reason = "preserve C Simpson weighting and independently rounded accumulation"
    )]
    #[expect(
        clippy::manual_midpoint,
        reason = "C explicitly sums sample values before scaling"
    )]
    fn consume(&mut self) {
        let y = |index: usize| {
            let value = self.panel[index].1;
            if self.statistic == MeasureStatistic::Rms {
                value * value
            } else {
                value
            }
        };
        let width = |index: usize| self.panel[index + 1].0 - self.panel[index].0;
        let equal = |a: Real, b: Real| a > 0. && b > 0. && a.to_bits().abs_diff(b.to_bits()) <= 100;
        let h = width(0);
        let consumed = if self.statistic == MeasureStatistic::Avg {
            self.area += 0.5 * (y(0) + y(1)) * h;
            1
        } else if self.count == 4 && equal(h, width(1)) && equal(h, width(2)) {
            self.area += 3. * h * (y(0) + 3. * (y(1) + y(2)) + y(3)) / 8.;
            3
        } else if self.count >= 3 && equal(h, width(1)) {
            self.area += h * (y(0) + 4. * y(1) + y(2)) / 3.;
            2
        } else {
            self.area += h * (y(0) + y(1)) / 2.;
            1
        };
        self.panel.copy_within(consumed..self.count, 0);
        self.count -= consumed;
    }

    fn area(mut self, card: &MeasureCard) -> SpiceResult<Real> {
        while self.count >= 2 {
            self.consume();
        }
        if !self.area.is_finite() {
            return Err(nonfinite(
                card,
                if self.statistic == MeasureStatistic::Avg {
                    "INTEG"
                } else {
                    "INTEG/RMS"
                },
                "the integrated value is not finite".into(),
            ));
        }
        Ok(self.area)
    }
}

fn integrate(
    points: impl Iterator<Item = SpiceResult<(Real, Real)>>,
    statistic: MeasureStatistic,
    card: &MeasureCard,
) -> SpiceResult<Real> {
    let mut integral = Integral::new(statistic);
    for point in points {
        integral.push(point?);
    }
    integral.area(card)
}

/// C `measure_margin`: unwrap phase, interpolate dB/degrees and log frequency.
#[expect(
    clippy::suboptimal_flops,
    reason = "retain reference interpolation rounding for the selected crossover and its margin"
)]
fn margin(
    axis: &Axis,
    operand: &Operand<'_>,
    window: &Window,
    phase_margin: bool,
    card: &MeasureCard,
) -> SpiceResult<(Real, Real)> {
    if axis.name != "frequency" || !operand.plot.flags().is_complex() {
        return Err(unsupported(
            card,
            "margin measurements require complex AC/SP data".into(),
        ));
    }
    let mut previous: Option<(Real, Real, Real)> = None;
    let mut raw_previous = 0.;
    let mut offset = 0.;
    for (i, point) in operand.plot.rows().enumerate() {
        let value = operand.column.value(point)?;
        let db = 20. * value.re.hypot(value.im).log10();
        let raw = value.im.atan2(value.re).to_degrees();
        if i > 0 {
            if raw - raw_previous > 180. {
                offset -= 360.;
            } else if raw - raw_previous < -180. {
                offset += 360.;
            }
        }
        raw_previous = raw;
        let phase = raw + offset;
        let frequency = axis.values.as_slice()[i];
        if !db.is_finite() {
            return Err(nonfinite(card, "margin", "zero or nonfinite gain".into()));
        }
        if let Some((f0, db0, p0)) = previous {
            let (a, b, target) = if phase_margin {
                (db0, db, 0.)
            } else {
                (p0, phase, -180.)
            };
            if (a >= target && b < target) || (a < target && b >= target) {
                let fraction = (target - a) / (b - a);
                let at = if f0 > 0. && frequency > 0. {
                    (f0.ln() + fraction * (frequency.ln() - f0.ln())).exp()
                } else {
                    f0 + fraction * (frequency - f0)
                };
                if at >= window.from && at <= window.to {
                    let value = if phase_margin {
                        180. + p0 + fraction * (phase - p0)
                    } else {
                        -(db0 + fraction * (db - db0))
                    };
                    return Ok((value, at));
                }
            }
        }
        previous = Some((frequency, db, phase));
    }
    Err(unsupported(
        card,
        "no margin crossover in the measurement window".into(),
    ))
}

/// Nonuniform quadratic derivative, C `measure_deriv_at`; two samples use a
/// safe secant instead of the C implementation's out-of-range stencil.
#[expect(
    clippy::float_cmp,
    reason = "exact coordinate identity selects discontinuities and zero-width stencil intervals; a tolerance would merge distinct samples"
)]
fn derivative_at(
    axis: &Axis,
    operand: &Operand<'_>,
    at: Real,
    card: &MeasureCard,
) -> SpiceResult<Real> {
    check_range(axis, at, card, "AT=")?;
    let (bracket, exact) = axis
        .values
        .derivative_bracket(at)
        .ok_or_else(|| unsupported(card, "DERIV needs a bracketing interval".into()))?;
    let j = bracket.right();
    if exact {
        // Check every duplicate at the exact point before choosing a stencil.
        value_at(axis, operand, at, card)?;
    }
    let secant = || -> SpiceResult<Real> {
        let width = bracket.x1 - bracket.x0;
        if width == 0. {
            return Err(unsupported(card, "DERIV at a repeated axis value".into()));
        }
        Ok((operand.value(j)? - operand.value(j - 1)?) / width)
    };
    let value = if axis.values.len() == 2 {
        secant()?
    } else {
        let middle = (j - 1).clamp(1, axis.values.len() - 2);
        let t = &axis.values.as_slice()[middle - 1..=middle + 1];
        if t[0] == t[1] || t[1] == t[2] {
            secant()?
        } else {
            let mut sum = 0.;
            for k in 0..3 {
                let a = (k + 1) % 3;
                let b = (k + 2) % 3;
                sum += operand.value(middle - 1 + k)? * ((at - t[a]) + (at - t[b]))
                    / ((t[k] - t[a]) * (t[k] - t[b]));
            }
            sum
        }
    };
    if !value.is_finite() {
        return Err(nonfinite(card, "DERIV", "nonfinite derivative".into()));
    }
    Ok(value)
}

/// The width of a covered window, which a mean needs to be defined.
fn span_width(span: MeasureSpan, card: &MeasureCard) -> SpiceResult<Real> {
    let width = span.to - span.from;
    if width <= 0.0 || !width.is_finite() {
        return Err(unsupported(
            card,
            format!(
                ".measure {}: the window [{}, {}] has no width, so the mean is undefined",
                card.name,
                crate::primitives::format_spice_number(span.from),
                crate::primitives::format_spice_number(span.to)
            ),
        ));
    }
    Ok(width)
}

/// The axis position of one `TRIG`/`TARG` event.
fn event_at(
    plot: &Plot,
    axis: &Axis,
    event: &MeasureEvent,
    window: &Window,
    card: &MeasureCard,
) -> SpiceResult<Real> {
    match event {
        MeasureEvent::Delayed { event, td } => {
            if !td.is_finite() {
                return Err(unsupported(card, "TD must be finite".into()));
            }
            let window = Window {
                from: window.from.max(*td),
                to: window.to,
            };
            if window.from > window.to {
                return Err(unsupported(
                    card,
                    "TD lies after the measurement window".into(),
                ));
            }
            event_at(plot, axis, event, &window, card)
        }
        MeasureEvent::At { at, .. } => {
            check_range(axis, *at, card, "the event AT=")?;
            if *at < window.from || *at > window.to {
                return Err(unsupported(
                    card,
                    format!(
                        ".measure {}: the event AT={} is outside the measurement window [{}, {}]",
                        card.name,
                        crate::primitives::format_spice_number(*at),
                        crate::primitives::format_spice_number(window.from),
                        crate::primitives::format_spice_number(window.to)
                    ),
                ));
            }
            Ok(*at)
        }
        MeasureEvent::VectorCrossing {
            operand,
            reference,
            transition,
        } => {
            let lhs = Operand::resolve(plot, operand, &card.name)?;
            let rhs = Operand::resolve(plot, reference, &card.name)?;
            crossing_samples(
                axis,
                |index| {
                    let difference = lhs.value(index)? - rhs.value(index)?;
                    if !difference.is_finite() {
                        return Err(nonfinite(card, "WHEN", "vector difference overflow".into()));
                    }
                    Ok(difference)
                },
                0.0,
                *transition,
                window,
                card,
                &format!("{} = {}", operand.vector.name(), reference.vector.name()),
            )
        }
        MeasureEvent::Crossing {
            operand,
            value,
            transition,
            ..
        } => crossing(plot, axis, operand, *value, *transition, window, card),
    }
}

/// The axis position where an operand crosses a threshold.
///
/// The scan starts at the last sample at or before the window's lower bound, so
/// the side the operand is on when it enters the window is known, and it ends at
/// the last sample inside the window. The side is `value >= threshold` (the high
/// side), as C's `com_measure_when()` section test; a transition from one side
/// to the other is one crossing, in the direction the operand moved. The
/// crossing's axis position is interpolated between the pair, or, when the pair
/// shares an axis value (a jump), that axis value exactly — never a division by
/// a zero span.
fn crossing(
    plot: &Plot,
    axis: &Axis,
    request: &VectorRequest,
    value: Real,
    transition: MeasureTransition,
    window: &Window,
    card: &MeasureCard,
) -> SpiceResult<Real> {
    let operand = Operand::resolve(plot, request, &card.name)?;
    crossing_samples(
        axis,
        |index| operand.value(index),
        value,
        transition,
        window,
        card,
        &request.vector.name(),
    )
}

struct CrossingSelector {
    transition: MeasureTransition,
    direction: Option<bool>,
    remaining: u32,
    selected: Option<Real>,
}

impl CrossingSelector {
    fn new(transition: MeasureTransition, card: &MeasureCard) -> SpiceResult<Self> {
        let (direction, ordinal) = match transition {
            MeasureTransition::First | MeasureTransition::Last => (None, 1),
            MeasureTransition::Cross(n) => (None, n),
            MeasureTransition::Rise(n) => (Some(true), n),
            MeasureTransition::Fall(n) => (Some(false), n),
        };
        if ordinal == 0 {
            return Err(unsupported(card, "crossing count must be positive".into()));
        }
        Ok(Self {
            transition,
            direction,
            remaining: ordinal,
            selected: None,
        })
    }

    fn include(
        &mut self,
        (x0, before): (Real, Real),
        (x1, after): (Real, Real),
        value: Real,
        window: &Window,
        card: &MeasureCard,
    ) -> SpiceResult<()> {
        let before_high = before >= value;
        let after_high = after >= value;
        if before_high == after_high {
            return Ok(());
        }
        let x = if super::ordered_samples::same_coordinate(x0, x1) {
            x1
        } else {
            interpolate_axis(x0, before, x1, after, value)
        };
        if !x.is_finite() {
            return Err(nonfinite(
                card,
                "WHEN/TRIG/TARG",
                "nonfinite crossing position".into(),
            ));
        }
        if x >= window.from
            && x <= window.to
            && self.direction.is_none_or(|rising| rising == after_high)
        {
            if self.transition == MeasureTransition::Last {
                self.selected = Some(x);
            } else if self.remaining != 0 {
                self.remaining -= 1;
                if self.remaining == 0 {
                    self.selected = Some(x);
                }
            }
        }
        Ok(())
    }
}

fn crossing_samples(
    axis: &Axis,
    sample: impl Fn(usize) -> SpiceResult<Real>,
    value: Real,
    transition: MeasureTransition,
    window: &Window,
    card: &MeasureCard,
    description: &str,
) -> SpiceResult<Real> {
    if !value.is_finite() {
        return Err(unsupported(
            card,
            "crossing threshold must be finite".into(),
        ));
    }
    let mut selector = CrossingSelector::new(transition, card)?;
    let mut segments = window.samples(axis, card)?.event_segments().peekable();
    if let Some(first) = segments.peek() {
        let mut before = sample(first.left)?;
        for segment in segments {
            let after = sample(segment.right())?;
            selector.include(
                (segment.x0, before),
                (segment.x1, after),
                value,
                window,
                card,
            )?;
            before = after;
        }
    }
    // Keep scanning after selecting a crossing: a later invalid sample still
    // invalidates the entire requested scan, just as the old collected list did.
    selector.selected.ok_or_else(|| {
        unsupported(
            card,
            format!(
                ".measure {}: no {} crossing of {description} through {} found from the window's \
                 lower bound [{}, {}]",
                card.name,
                transition.name(),
                crate::primitives::format_spice_number(value),
                crate::primitives::format_spice_number(window.from),
                crate::primitives::format_spice_number(window.to)
            ),
        )
    })
}

/// The unit of a statistic's result.
fn statistic_unit(statistic: MeasureStatistic, unit: &str, axis: &str) -> String {
    match statistic {
        MeasureStatistic::PhaseMargin => "degrees".into(),
        MeasureStatistic::GainMargin => "db".into(),
        MeasureStatistic::MinAt | MeasureStatistic::MaxAt => axis.to_owned(),
        MeasureStatistic::Integ => format!("{unit}*{axis}"),
        _ => unit.to_owned(),
    }
}

/// The axis value at which the line through `(x0, y0)` and `(x1, y1)` reaches
/// `y`, i.e. the crossing position.
///
/// The caller guarantees `y0 != y1` (the two samples are on opposite sides of
/// the threshold) and a positive bracket width, so there is never a division by
/// a zero span.
fn interpolate_axis(x0: Real, y0: Real, x1: Real, y1: Real, y: Real) -> Real {
    x0 + (x1 - x0) * (y - y0) / (y1 - y0)
}

#[cfg(test)]
mod tests {
    use super::{MeasureSpan, Measurement, resolve, to_text};
    use crate::analysis::results::{Plot, PlotFlags, Variable};
    use crate::netlist::ast::{
        MeasureCard, MeasureEvent, MeasureRequest, MeasureStatistic, MeasureTransition,
        MeasureWindow, RequestedVector, VectorRequest,
    };
    use crate::primitives::{AnalysisKind, Complex, Real, SourceLoc};
    use std::path::PathBuf;

    fn loc() -> SourceLoc {
        SourceLoc::new(PathBuf::from("deck.cir"), 2, 1)
    }

    /// A real sweep plot: `time` then `v(out)`, from `(time, value)` rows.
    fn tran(rows: &[(Real, Real)]) -> Plot {
        let mut plot = Plot::builder("tran1", "Transient Analysis", PlotFlags::Real);
        plot.push_variable(Variable::new("time", "time"));
        plot.push_variable(Variable::new("v(out)", "voltage"));
        let mut plot = plot.build();
        for (time, value) in rows {
            plot.push_point(&[Complex::real(*time), Complex::real(*value)])
                .unwrap();
        }
        plot
    }

    /// `(time, value)` rows of an evenly spaced sweep.
    fn sweep(
        from: Real,
        to: Real,
        steps: usize,
        value: impl Fn(Real) -> Real,
    ) -> Vec<(Real, Real)> {
        (0..=steps)
            .map(|step| {
                let time = from + (to - from) * (step as Real) / (steps as Real);
                (time, value(time))
            })
            .collect()
    }

    fn request(
        statistic: MeasureStatistic,
        from: Option<Real>,
        to: Option<Real>,
    ) -> MeasureRequest {
        MeasureRequest::Statistic {
            statistic,
            operand: out(),
            window: MeasureWindow { from, to },
        }
    }

    fn out() -> VectorRequest {
        VectorRequest {
            vector: RequestedVector::Voltage {
                positive: "out".to_owned(),
                negative: None,
            },
            location: loc(),
        }
    }

    fn card(name: &str, request: MeasureRequest) -> MeasureCard {
        MeasureCard {
            analysis: AnalysisKind::Transient,
            analysis_location: loc(),
            name: name.to_owned(),
            name_location: loc(),
            request,
            location: loc(),
        }
    }

    fn measure(plot: &Plot, name: &str, request: MeasureRequest) -> Measurement {
        let mut results = resolve(plot, AnalysisKind::Transient, &[card(name, request)])
            .expect("the measurement resolves");
        assert_eq!(results.len(), 1);
        results.remove(0)
    }

    fn failure(plot: &Plot, name: &str, request: MeasureRequest) -> crate::primitives::SpiceError {
        resolve(plot, AnalysisKind::Transient, &[card(name, request)])
            .expect_err("the measurement fails")
    }

    fn crossing(value: Real, transition: MeasureTransition) -> MeasureEvent {
        MeasureEvent::Crossing {
            operand: out(),
            value,
            value_location: loc(),
            transition,
        }
    }

    #[test]
    fn distinct_adjacent_coordinates_are_not_discontinuities() {
        let left = 1.0_f64;
        let middle = left.next_up();
        let right = middle.next_up();
        let plot = tran(&[(left, 0.0), (right, 2.0)]);
        let result = measure(
            &plot,
            "crossing",
            MeasureRequest::When {
                event: crossing(1.0, MeasureTransition::Rise(1)),
                window: MeasureWindow::default(),
            },
        );
        assert_eq!(result.value.to_bits(), middle.to_bits());
        let result = measure(
            &plot,
            "slope",
            MeasureRequest::AtEvent {
                operand: out(),
                event: MeasureEvent::At {
                    at: middle,
                    location: loc(),
                },
                derivative: true,
                window: MeasureWindow::default(),
            },
        );
        assert_eq!(result.value.to_bits(), (2.0 / (right - left)).to_bits());
    }

    #[test]
    fn trapezoid_keeps_reference_sum_before_scaling() {
        let card = card("m", request(MeasureStatistic::Avg, None, None));
        let values = [(0.0, Real::MAX), (0.5, Real::MAX)];
        // C's intermediate sample sum overflows, even though an alternative
        // midpoint implementation could keep the final integral finite.
        assert!(
            super::integrate(values.into_iter().map(Ok), MeasureStatistic::Avg, &card).is_err()
        );
    }

    #[test]
    fn streaming_simpson_panels_keep_greedy_selection_and_rounding() {
        let points: [(Real, Real); 9] = [
            (0., 0.1),
            (1., -0.2),
            (2., 1.3),
            (3., 0.4),
            (4., 0.5),
            (5., -0.6),
            (7., 2.7),
            (7., -0.8),
            (8., 0.9),
        ];
        let card = card("m", request(MeasureStatistic::Integ, None, None));
        for square in [false, true] {
            let y = |i: usize| {
                if square {
                    points[i].1 * points[i].1
                } else {
                    points[i].1
                }
            };
            let h = |i: usize| points[i + 1].0 - points[i].0;
            let mut expected = 3. * h(0) * (y(0) + 3. * (y(1) + y(2)) + y(3)) / 8.;
            expected += h(3) * (y(3) + 4. * y(4) + y(5)) / 3.;
            for i in 5..8 {
                expected += h(i) * (y(i) + y(i + 1)) / 2.;
            }
            let got = super::integrate(
                points.into_iter().map(Ok),
                if square {
                    MeasureStatistic::Rms
                } else {
                    MeasureStatistic::Integ
                },
                &card,
            )
            .unwrap();
            assert_eq!(got.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn a_ramp_has_analytic_statistics_and_boundaries_are_interpolated() {
        // v(t) = 2t on [0, 1], sampled every 1 ms. AVG/INTEG are exact (the
        // trapezoid rule integrates linear data exactly); RMS integrates v^2,
        // which is quadratic, so the trapezoid rule leaves its known O(h^2)
        // error of (h^2/12) * 8 = 6.7e-7.
        let plot = tran(&sweep(0.0, 1.0, 1000, |time| 2.0 * time));
        let whole = |statistic| measure(&plot, "m", request(statistic, None, None));
        assert_eq!(whole(MeasureStatistic::Min).value, 0.0);
        assert_eq!(whole(MeasureStatistic::Max).value, 2.0);
        assert!((whole(MeasureStatistic::Avg).value - 1.0).abs() < 1e-15);
        assert!((whole(MeasureStatistic::Integ).value - 1.0).abs() < 1e-15);
        assert!((whole(MeasureStatistic::Rms).value - (4.0f64 / 3.0).sqrt()).abs() < 1e-6);

        // [0.25, 0.75] is bounded by samples here, so MIN/MAX see them whole.
        let window = |statistic| measure(&plot, "m", request(statistic, Some(0.25), Some(0.75)));
        assert!((window(MeasureStatistic::Avg).value - 1.0).abs() < 1e-15);
        assert!((window(MeasureStatistic::Integ).value - 0.5).abs() < 1e-15);
        let mean_square = 8.0 * (0.75f64.powi(3) - 0.25f64.powi(3)) / 3.0;
        assert!((window(MeasureStatistic::Rms).value - mean_square.sqrt()).abs() < 1e-6);
        assert_eq!(window(MeasureStatistic::Min).value, 0.5);
        assert_eq!(window(MeasureStatistic::Max).value, 1.5);
        assert_eq!(
            window(MeasureStatistic::Avg).window,
            Some(MeasureSpan {
                from: 0.25,
                to: 0.75
            })
        );
        // MIN/MAX report where they found the extremum (C's `at=`).
        assert_eq!(window(MeasureStatistic::Min).at, Some(0.25));
        assert_eq!(window(MeasureStatistic::Max).at, Some(0.75));
        assert_eq!(window(MeasureStatistic::Min).unit, "voltage");
    }

    #[test]
    fn a_window_boundary_off_the_samples_is_interpolated_for_the_means() {
        // v(t) = 2t sampled every 0.1, window [0.25, 0.75]: neither boundary is a
        // sample, so AVG/INTEG/RMS are only exact because the boundaries are
        // interpolated.
        let plot = tran(&sweep(0.0, 1.0, 10, |time| 2.0 * time));
        let window = |statistic| measure(&plot, "m", request(statistic, Some(0.25), Some(0.75)));
        assert!((window(MeasureStatistic::Avg).value - 1.0).abs() < 1e-15);
        assert!((window(MeasureStatistic::Integ).value - 0.5).abs() < 1e-15);
        // The trapezoid rule's own error on v^2 (h = 0.1) is ~7e-3, so the exact
        // mean square is only approached to that order.
        let mean_square = 8.0 * (0.75f64.powi(3) - 0.25f64.powi(3)) / 3.0;
        assert!((window(MeasureStatistic::Rms).value - mean_square.sqrt()).abs() < 0.01);
        // MIN/MAX keep whole samples (C's semantics), so they see 0.3 .. 0.7.
        assert_eq!(window(MeasureStatistic::Min).value, 0.6);
        assert_eq!(window(MeasureStatistic::Max).value, 1.4);
        assert_eq!(window(MeasureStatistic::Min).at, Some(0.3));
    }

    #[test]
    fn a_constant_is_its_own_mean_rms_and_area() {
        let plot = tran(&sweep(0.0, 1.0, 4, |_| 3.0));
        for statistic in [MeasureStatistic::Min, MeasureStatistic::Max] {
            let result = measure(&plot, "m", request(statistic, None, None));
            assert_eq!(result.value, 3.0);
        }
        assert_eq!(
            measure(&plot, "m", request(MeasureStatistic::Avg, None, None)).value,
            3.0
        );
        assert_eq!(
            measure(&plot, "m", request(MeasureStatistic::Rms, None, None)).value,
            3.0
        );
        assert_eq!(
            measure(&plot, "m", request(MeasureStatistic::Integ, None, None)).value,
            3.0
        );
        // An integral's unit is the operand's times the axis'.
        let integral = measure(&plot, "m", request(MeasureStatistic::Integ, None, None));
        assert_eq!(integral.unit, "voltage*time");
        assert_eq!(integral.window.map(|span| span.to), Some(1.0));
    }

    #[test]
    fn a_min_max_tie_keeps_the_last_sample_like_c() {
        // A flat top: C's `value <= mValue` / `value >= mValue` update on ties.
        let plot = tran(&[(0.0, 0.0), (1.0, 1.0), (2.0, 1.0), (3.0, 0.0)]);
        assert_eq!(
            measure(&plot, "m", request(MeasureStatistic::Max, None, None)).at,
            Some(2.0)
        );
        assert_eq!(
            measure(&plot, "m", request(MeasureStatistic::Min, None, None)).at,
            Some(3.0)
        );
    }

    #[test]
    fn a_sinusoid_has_analytic_crossings_and_rms() {
        let plot = tran(&sweep(0.0, 2.0, 2000, |time| {
            (2.0 * std::f64::consts::PI * time).sin()
        }));
        // Crossing -0.5: rising at 7/12, falling at 11/12.
        let event = |transition| MeasureRequest::TrigTarg {
            trig: crossing(-0.5, transition),
            targ: crossing(-0.5, MeasureTransition::Rise(2)),
            window: MeasureWindow::default(),
        };
        // -0.5 is crossed falling at 7/12 and rising at 11/12, one period apart.
        let result = measure(&plot, "delay", event(MeasureTransition::Fall(1)));
        assert!((result.events.unwrap().trig - 7.0 / 12.0).abs() < 1e-6);
        let result = measure(&plot, "delay", event(MeasureTransition::Rise(1)));
        let events = result.events.unwrap();
        assert!((events.trig - 11.0 / 12.0).abs() < 1e-6);
        assert!((events.targ - (11.0 / 12.0 + 1.0)).abs() < 1e-6);
        assert!((result.value - 1.0).abs() < 1e-6);
        assert_eq!(result.unit, "time");

        // sin >= 0 at t = 0, so the first transition is the fall through 0 at
        // 0.5 and the first rise is at 1.0, exactly as C's section test reads it.
        let first = MeasureRequest::TrigTarg {
            trig: crossing(0.0, MeasureTransition::First),
            targ: crossing(0.0, MeasureTransition::Rise(1)),
            window: MeasureWindow::default(),
        };
        let result = measure(&plot, "delay", first);
        let events = result.events.unwrap();
        assert!((events.trig - 0.5).abs() < 1e-6);
        assert!((events.targ - 1.0).abs() < 1e-6);

        // RMS of a full period of sin is 1/sqrt(2); the trapezoid rule leaves an
        // O(dt^2) error.
        let rms = measure(&plot, "rms", request(MeasureStatistic::Rms, None, None));
        assert!((rms.value - 0.5f64.sqrt()).abs() < 1e-6, "{}", rms.value);
        // ... and the mean over a period is zero.
        let avg = measure(
            &plot,
            "avg",
            request(MeasureStatistic::Avg, Some(0.0), Some(1.0)),
        );
        assert!(avg.value.abs() < 1e-6, "{}", avg.value);
    }

    #[test]
    fn last_takes_the_last_crossing_and_counts_select_the_nth() {
        let plot = tran(&sweep(0.0, 4.0, 400, |time| {
            (std::f64::consts::PI * time).sin()
        }));
        // sin(pi t) is sampled over [0, 4]: its transitions are the fall at
        // t ~ 1, the rise at t ~ 2 and the fall at t ~ 3 (at t = 1, 2, 3 the
        // sample is an ulp off zero and belongs to the side it approaches).
        let event = |transition| MeasureRequest::TrigTarg {
            trig: crossing(0.0, transition),
            targ: MeasureEvent::At {
                at: 0.0,
                location: loc(),
            },
            window: MeasureWindow::default(),
        };
        let trig_of = |transition| {
            let result = measure(&plot, "delay", event(transition));
            result.events.unwrap().trig
        };
        assert!((trig_of(MeasureTransition::Cross(1)) - 1.0).abs() < 1e-2);
        assert!((trig_of(MeasureTransition::Cross(2)) - 2.0).abs() < 1e-2);
        assert!((trig_of(MeasureTransition::Cross(3)) - 3.0).abs() < 1e-2);
        assert!((trig_of(MeasureTransition::Last) - 3.0).abs() < 1e-2);
        assert!((trig_of(MeasureTransition::Rise(1)) - 2.0).abs() < 1e-2);
        assert!((trig_of(MeasureTransition::Fall(2)) - 3.0).abs() < 1e-2);
        // There is no second rising crossing, so asking for one fails explicitly.
        let error = failure(&plot, "delay", event(MeasureTransition::Rise(2)));
        assert!(error.to_string().contains("no RISE=2 crossing"), "{error}");
    }

    #[test]
    fn find_at_interpolates_the_bracket_at_and_inside_a_window() {
        let plot = tran(&sweep(0.0, 1.0, 4, |time| 2.0 * time));
        let find = |at, from, to| MeasureRequest::Find {
            operand: out(),
            at,
            at_location: loc(),
            window: MeasureWindow { from, to },
        };
        let result = measure(&plot, "vat", find(0.35, None, None));
        assert!((result.value - 0.7).abs() < 1e-15);
        assert_eq!(result.at, Some(0.35));
        assert_eq!(result.unit, "voltage");
        // A sample position is read exactly.
        assert_eq!(measure(&plot, "vat", find(0.5, None, None)).value, 1.0);
        // A query inside the window works; one outside it does not.
        assert_eq!(
            measure(&plot, "vat", find(0.5, Some(0.25), Some(0.75))).value,
            1.0
        );
        let error = failure(&plot, "vat", find(0.8, Some(0.25), Some(0.75)));
        assert!(
            error.to_string().contains("outside the measurement window"),
            "{error}"
        );
    }

    #[test]
    fn queries_outside_the_axis_and_windows_without_data_are_explicit_failures() {
        let plot = tran(&sweep(0.0, 1.0, 4, |time| 2.0 * time));
        let find = |at| MeasureRequest::Find {
            operand: out(),
            at,
            at_location: loc(),
            window: MeasureWindow::default(),
        };
        for at in [-1.0, 1.5] {
            let error = failure(&plot, "vat", find(at));
            assert!(!error.is_not_yet_ported(), "{error}");
            assert!(error.to_string().contains("out of interval"), "{error}");
            assert!(error.to_string().contains("time range"), "{error}");
            assert!(error.to_string().contains("deck.cir:2:1"), "{error}");
        }
        // A window wholly outside the axis, and an inverted one.
        let error = failure(
            &plot,
            "m",
            request(MeasureStatistic::Avg, Some(5.0), Some(6.0)),
        );
        assert!(error.to_string().contains("lies outside"), "{error}");
        let error = failure(
            &plot,
            "m",
            request(MeasureStatistic::Avg, Some(0.8), Some(0.2)),
        );
        assert!(error.to_string().contains("above TO="), "{error}");
        // A window that only reaches past the axis is clipped, not rejected: a
        // transient's stop time is a sample, so this is the requested window
        // intersected with the data.
        let clipped = measure(
            &plot,
            "m",
            request(MeasureStatistic::Integ, Some(0.0), Some(5.0)),
        );
        assert_eq!(clipped.value, 1.0);
        assert_eq!(clipped.window.unwrap().to, 1.0);
    }

    #[test]
    fn an_empty_or_zero_width_window_is_never_a_fabricated_mean() {
        let plot = tran(&sweep(0.0, 1.0, 4, |time| 2.0 * time));
        for statistic in [
            MeasureStatistic::Avg,
            MeasureStatistic::Rms,
            MeasureStatistic::Integ,
        ] {
            let error = failure(&plot, "m", request(statistic, Some(0.5), Some(0.5)));
            assert!(error.to_string().contains("has no width"), "{error}");
        }
        // A plot with no points cannot be measured at all.
        let empty = tran(&[]);
        let error = failure(&empty, "m", request(MeasureStatistic::Avg, None, None));
        assert!(error.to_string().contains("no points"), "{error}");
    }

    #[test]
    fn a_duplicate_time_is_a_jump_with_a_defined_crossing_but_no_defined_value() {
        // t = 1 carries both the left limit 1 and the right limit 10: the plot
        // records a source jump the way the drivers never do.
        let plot = tran(&[(0.0, 0.0), (1.0, 1.0), (1.0, 10.0), (2.0, 12.0)]);
        // The jump crosses 5 exactly at t = 1, with no interpolation across it.
        let crossing = MeasureEvent::Crossing {
            operand: out(),
            value: 5.0,
            value_location: loc(),
            transition: MeasureTransition::Rise(1),
        };
        let result = measure(
            &plot,
            "delay",
            MeasureRequest::TrigTarg {
                trig: crossing,
                targ: MeasureEvent::At {
                    at: 0.0,
                    location: loc(),
                },
                window: MeasureWindow::default(),
            },
        );
        assert_eq!(result.events.unwrap().trig, 1.0);
        // The operand's value at the jump is not defined...
        let error = failure(
            &plot,
            "vat",
            MeasureRequest::Find {
                operand: out(),
                at: 1.0,
                at_location: loc(),
                window: MeasureWindow::default(),
            },
        );
        assert!(error.to_string().contains("discontinuous"), "{error}");
        // ... while a duplicated time with the same value is harmless.
        let smooth = tran(&[(0.0, 0.0), (1.0, 1.0), (1.0, 1.0), (2.0, 2.0)]);
        let value = measure(
            &smooth,
            "vat",
            MeasureRequest::Find {
                operand: out(),
                at: 1.0,
                at_location: loc(),
                window: MeasureWindow::default(),
            },
        );
        assert_eq!(value.value, 1.0);
        // Interpolation still works inside the continuous brackets around it.
        assert_eq!(
            measure(
                &plot,
                "vat",
                MeasureRequest::Find {
                    operand: out(),
                    at: 0.5,
                    at_location: loc(),
                    window: MeasureWindow::default(),
                },
            )
            .value,
            0.5
        );
    }

    #[test]
    fn exact_queries_check_the_entire_duplicate_sample_group() {
        let plot = tran(&[(0., 0.), (1., 1.), (1., 1.), (1., 3.), (2., 4.)]);
        let find = |at| MeasureRequest::Find {
            operand: out(),
            at,
            at_location: loc(),
            window: MeasureWindow::default(),
        };
        let error = failure(&plot, "m", find(1.));
        assert!(error.to_string().contains("discontinuous"));
        let deriv = MeasureRequest::AtEvent {
            operand: out(),
            event: MeasureEvent::At {
                at: 1.,
                location: loc(),
            },
            derivative: true,
            window: MeasureWindow::default(),
        };
        assert!(
            failure(&plot, "m", deriv)
                .to_string()
                .contains("discontinuous")
        );
        assert_eq!(measure(&plot, "m", find(0.5)).value, 0.5);
        assert_eq!(measure(&plot, "m", find(1.5)).value, 3.5);
    }

    #[test]
    fn clipped_window_policies_remain_distinct() {
        let plot = tran(&[(0., 0.), (1., 1.), (2., 2.), (3., 3.)]);
        let window = |statistic| request(statistic, Some(0.25), Some(1.75));
        assert_eq!(
            measure(&plot, "min", window(MeasureStatistic::Min)).value,
            1.
        );
        assert_eq!(
            measure(&plot, "max", window(MeasureStatistic::Max)).value,
            1.
        );
        assert_eq!(
            measure(&plot, "avg", window(MeasureStatistic::Avg)).value,
            1.
        );
        assert_eq!(
            measure(&plot, "integ", window(MeasureStatistic::Integ)).value,
            1.5
        );
        // The previously documented no-stored-sample behavior is intentional:
        // pinned C returns zero and can report NaN interval metadata.
        for statistic in [
            MeasureStatistic::Min,
            MeasureStatistic::Max,
            MeasureStatistic::Avg,
            MeasureStatistic::Rms,
            MeasureStatistic::Integ,
        ] {
            let error = failure(&plot, "m", request(statistic, Some(0.25), Some(0.75)));
            assert!(
                error.to_string().contains("covers no time sample"),
                "{error}"
            );
        }
        let event = |threshold| MeasureRequest::When {
            event: crossing(threshold, MeasureTransition::Rise(1)),
            window: MeasureWindow {
                from: Some(0.25),
                to: Some(1.75),
            },
        };
        // Retain the pre-window sample for the left crossing, but do not
        // synthesize a crossing in the right bracket beyond the last sample.
        assert_eq!(measure(&plot, "left", event(0.5)).value, 0.5);
        assert!(
            failure(&plot, "right", event(1.5))
                .to_string()
                .contains("no RISE=1 crossing")
        );
    }

    #[test]
    fn signed_zero_jump_and_coincident_smooth_samples_preserve_policy() {
        let plot = tran(&[
            (-1., -1.),
            (-0., 1.),
            (0., 10.),
            (1., 11.),
            (1., 11.),
            (2., 12.),
        ]);
        let find = |at| MeasureRequest::Find {
            operand: out(),
            at,
            at_location: loc(),
            window: MeasureWindow::default(),
        };
        assert!(
            failure(&plot, "jump", find(0.))
                .to_string()
                .contains("discontinuous")
        );
        assert_eq!(measure(&plot, "smooth", find(1.)).value, 11.);
        let when = MeasureRequest::When {
            event: crossing(5., MeasureTransition::Rise(1)),
            window: MeasureWindow::default(),
        };
        assert_eq!(
            measure(&plot, "jump", when).value.to_bits(),
            0_f64.to_bits()
        );
        assert_eq!(
            measure(
                &plot,
                "avg",
                request(MeasureStatistic::Avg, Some(0.), Some(2.))
            )
            .value,
            11.
        );
        assert_eq!(
            measure(
                &plot,
                "min",
                request(MeasureStatistic::Min, Some(1.), Some(1.))
            )
            .value,
            11.
        );
    }

    #[test]
    fn invalid_hand_built_queries_return_errors_instead_of_panicking_or_clamping() {
        let plot = tran(&[(0., 0.), (1., 1.), (2., 2.)]);
        let find = |at| MeasureRequest::Find {
            operand: out(),
            at,
            at_location: loc(),
            window: MeasureWindow::default(),
        };
        assert!(
            failure(&plot, "m", find(Real::NAN))
                .to_string()
                .contains("outside")
        );
        let window = request(MeasureStatistic::Avg, Some(Real::NAN), None);
        assert!(failure(&plot, "m", window).to_string().contains("finite"));
        let cross = |value, transition| MeasureRequest::When {
            event: MeasureEvent::Crossing {
                operand: out(),
                value,
                value_location: loc(),
                transition,
            },
            window: MeasureWindow::default(),
        };
        assert!(
            failure(&plot, "m", cross(0.5, MeasureTransition::Rise(0)))
                .to_string()
                .contains("positive")
        );
        assert!(
            failure(&plot, "m", cross(Real::INFINITY, MeasureTransition::First))
                .to_string()
                .contains("finite")
        );
    }

    #[test]
    fn crossing_selection_still_checks_later_samples_after_finding_a_match() {
        let plot = tran(&[(0., 0.), (1., 1.), (2., Real::NAN)]);
        let crossing = MeasureEvent::Crossing {
            operand: out(),
            value: 0.5,
            value_location: loc(),
            transition: MeasureTransition::First,
        };
        let error = failure(
            &plot,
            "m",
            MeasureRequest::When {
                event: crossing,
                window: MeasureWindow::default(),
            },
        );
        assert!(error.to_string().contains("not finite"));
    }

    #[test]
    fn a_nonfinite_operand_or_result_is_numerical_not_a_written_value() {
        let plot = tran(&[(0.0, 0.0), (1.0, Real::NAN), (2.0, 2.0)]);
        let error = failure(&plot, "m", request(MeasureStatistic::Max, None, None));
        assert!(
            matches!(error, crate::primitives::SpiceError::Numerical { .. }),
            "{error}"
        );
        assert!(error.to_string().contains("measure m: v(out)"), "{error}");
        assert!(error.to_string().contains("not finite"), "{error}");

        // An operand that is finite everywhere still cannot be integrated when
        // squaring overflows.
        let huge = tran(&[(0.0, 0.0), (1.0, 1e200)]);
        let error = failure(&huge, "m", request(MeasureStatistic::Rms, None, None));
        assert!(error.to_string().contains("not finite"), "{error}");
    }

    #[test]
    fn a_missing_axis_operand_or_analysis_is_an_explicit_failure() {
        let plot = tran(&sweep(0.0, 1.0, 2, |time| 2.0 * time));
        // An operand the plot does not carry.
        let missing = MeasureRequest::Statistic {
            statistic: MeasureStatistic::Max,
            operand: VectorRequest {
                vector: RequestedVector::Voltage {
                    positive: "nosuch".to_owned(),
                    negative: None,
                },
                location: loc(),
            },
            window: MeasureWindow::default(),
        };
        let error = failure(&plot, "m", missing);
        assert!(error.to_string().contains("v(nosuch)"), "{error}");

        // An axis the plot does not carry.
        let no_time = Plot::new("op1", "Operating Point", PlotFlags::Real, Vec::new());
        let error = resolve(
            &plot,
            AnalysisKind::Ac,
            &[card("m", request(MeasureStatistic::Max, None, None))],
        )
        .expect_err("wrong axis");
        assert!(error.to_string().contains("frequency"), "{error}");
        let error = resolve(
            &no_time,
            AnalysisKind::OperatingPoint,
            &[card("m", request(MeasureStatistic::Max, None, None))],
        )
        .expect_err("no axis");
        assert!(error.to_string().contains("ordered axis"), "{error}");

        // A card naming another analysis than the run.
        let mut other = card("m", request(MeasureStatistic::Max, None, None));
        other.analysis = AnalysisKind::Ac;
        let error = resolve(&plot, AnalysisKind::Transient, &[other]).expect_err("wrong analysis");
        assert!(error.to_string().contains("names a .ac"), "{error}");

        // A crossing that does not occur.
        let error = failure(
            &plot,
            "delay",
            MeasureRequest::TrigTarg {
                trig: crossing(5.0, MeasureTransition::Rise(1)),
                targ: crossing(5.0, MeasureTransition::Fall(1)),
                window: MeasureWindow::default(),
            },
        );
        assert!(error.to_string().contains("no RISE=1 crossing"), "{error}");
        assert!(error.to_string().contains("v(out)"), "{error}");
    }

    #[test]
    fn the_axis_must_ascend_and_a_complex_plot_measures_its_real_part() {
        let descending = tran(&[(1.0, 0.0), (0.5, 1.0), (0.0, 2.0)]);
        let error = failure(&descending, "m", request(MeasureStatistic::Max, None, None));
        assert!(error.to_string().contains("not non-decreasing"), "{error}");
        assert!(error.to_string().contains("descends at point 1"), "{error}");

        // An AC plot: the frequency axis and the real part of a complex vector,
        // exactly as C's `get_value()` defaults.
        let mut ac = Plot::builder("ac1", "AC Analysis", PlotFlags::Complex);
        ac.push_variable(Variable::complex("frequency", "frequency"));
        ac.push_variable(Variable::complex("v(out)", "voltage"));
        let mut ac = ac.build();
        for frequency in [100.0, 200.0, 400.0] {
            ac.push_point(&[
                Complex::real(frequency),
                Complex::new(frequency / 100.0, -1.0),
            ])
            .unwrap();
        }
        let mut ac_card = card("vm", request(MeasureStatistic::Max, None, None));
        ac_card.analysis = AnalysisKind::Ac;
        let result =
            resolve(&ac, AnalysisKind::Ac, &[ac_card]).expect("the AC measurement resolves");
        assert_eq!(result[0].value, 4.0);
        assert_eq!(result[0].unit, "voltage");
    }

    #[test]
    fn a_deck_without_measure_cards_produces_no_results_and_no_failure() {
        // The CLI relies on this: a plot with no measurement axis (`op`) and no
        // `.measure` card must not fail.
        let op = Plot::new("op1", "Operating Point", PlotFlags::Real, Vec::new());
        assert_eq!(
            resolve(&op, AnalysisKind::OperatingPoint, &[]).unwrap(),
            Vec::new()
        );
        assert_eq!(
            to_text(&[]),
            "measure: 0 result(s), evaluated on the full plot before any .save/.print selection\n"
        );
    }

    #[test]
    fn measurement_record_limit_counts_duplicates_before_evaluating() {
        let scope = crate::netlist::eval::ParamScope::root(&[]).unwrap();
        let limit = crate::netlist::eval::EvalLimits::default().definitions;
        let plot = tran(&[(0., 1.)]);
        let repeated = card("same", request(MeasureStatistic::Max, None, None));
        let error = super::resolve_refs_with_scope(
            &plot,
            AnalysisKind::Transient,
            std::iter::repeat_n(&repeated, limit + 1),
            &scope,
        )
        .unwrap_err();
        assert!(
            matches!(&error, crate::primitives::SpiceError::Unsupported {
            location: Some(location), ..
        } if location == &repeated.location)
        );
        assert!(error.to_string().contains("measurement record limit"));
        let results = super::resolve_refs_with_scope(
            &plot,
            AnalysisKind::Transient,
            std::iter::repeat_n(&repeated, limit),
            &scope,
        )
        .unwrap();
        assert_eq!(results.len(), limit);
        assert!(results.iter().all(|result| result.value == 1.));
        assert!(scope.get("same").is_none());
    }

    #[test]
    fn duplicate_result_names_are_kept_in_card_order() {
        let plot = tran(&sweep(0.0, 1.0, 4, |time| 2.0 * time));
        let results = resolve(
            &plot,
            AnalysisKind::Transient,
            &[
                card("m", request(MeasureStatistic::Min, None, None)),
                card("m", request(MeasureStatistic::Max, None, None)),
            ],
        )
        .expect("both cards resolve");
        assert_eq!(
            results
                .iter()
                .map(|result| (result.name.as_str(), result.value))
                .collect::<Vec<_>>(),
            [("m", 0.0), ("m", 2.0)],
            "a repeated name is two results, not one overwritten result"
        );
    }

    #[test]
    fn the_text_block_follows_c_with_the_ports_spelling() {
        let plot = tran(&sweep(0.0, 1.0, 4, |time| 2.0 * time));
        let results = resolve(
            &plot,
            AnalysisKind::Transient,
            &[
                card("vmax", request(MeasureStatistic::Max, None, None)),
                card("vavg", request(MeasureStatistic::Avg, None, None)),
                card(
                    "delay",
                    MeasureRequest::TrigTarg {
                        trig: crossing(1.0, MeasureTransition::Rise(1)),
                        targ: crossing(2.0, MeasureTransition::Rise(1)),
                        window: MeasureWindow::default(),
                    },
                ),
            ],
        )
        .expect("the measurements resolve");
        let text = to_text(&results);
        let expected = concat!(
            "measure: 3 result(s), evaluated on the full plot before any .save/.print selection\n",
            "vmax                =  2.000000000000000e+00 at=  1.000000000000000e+00\n",
            "vavg                =  1.000000000000000e+00 from=  0.000000000000000e+00 to=  1.000000000000000e+00\n",
            "delay               =  5.000000000000000e-01 targ=  1.000000000000000e+00 trig=  5.000000000000000e-01\n",
        );
        assert_eq!(text, expected);
    }
}
