//! Bounded reducers over the accepted-point stream. The retained evaluator
//! and this execution path share interpolation, extrema, crossing selection
//! and quadrature policies; only the traversal differs.

use super::{
    CrossingSelector, Extrema, Integral, MarginScan, MeasureEvents, MeasureSpan, Measurement,
    Window, axis_name, descending_axis, missing_axis, nonfinite, nonfinite_axis, span_width,
    unsupported,
};
use crate::analysis::ordered_samples::same_coordinate;
use crate::analysis::results::Plot;
use crate::analysis::selection::{Column, resolve_request};
use crate::netlist::ast::{
    MeasureCard, MeasureEvent, MeasureRequest, MeasureStatistic, MeasureTransition, MeasureWindow,
};
use crate::primitives::{AnalysisKind, Complex, Real, SpiceError, SpiceResult};

mod local_query;
use local_query::LocalHistory;

type Sample = (Real, SpiceResult<Complex>);

/// Shared coordinate validation. Deck measurements require the retained
/// evaluator's nondecreasing axis; direct point objectives may instead use
/// a nonincreasing axis. Finiteness has precedence over order violations.
pub(in crate::analysis) struct AxisTracker {
    pub(super) name: &'static str,
    column: usize,
    first: Option<Real>,
    last: Option<Real>,
    count: usize,
    nonfinite: Option<usize>,
    descending: Option<usize>,
    ascending: Option<usize>,
}

impl AxisTracker {
    pub(in crate::analysis) fn bind(plot: &Plot, kind: AnalysisKind) -> SpiceResult<Self> {
        Self::bind_named(plot, axis_name(plot, kind)?)
    }

    pub(in crate::analysis) fn bind_named(plot: &Plot, name: &'static str) -> SpiceResult<Self> {
        let column = plot
            .variable_index(name)
            .ok_or_else(|| missing_axis(plot, name))?;
        Ok(Self {
            name,
            column,
            first: None,
            last: None,
            count: 0,
            nonfinite: None,
            descending: None,
            ascending: None,
        })
    }

    pub(in crate::analysis) const fn column(&self) -> usize {
        self.column
    }

    pub(in crate::analysis) const fn fresh(&self) -> Self {
        Self {
            name: self.name,
            column: self.column,
            first: None,
            last: None,
            count: 0,
            nonfinite: None,
            descending: None,
            ascending: None,
        }
    }

    /// Row width is checked by the enclosing sink before any binding is used.
    pub(in crate::analysis) fn observe(&mut self, row: &[Complex]) -> Option<Real> {
        let coordinate = row[self.column];
        let index = self.count;
        self.count += 1;
        if !coordinate.is_finite() {
            self.nonfinite.get_or_insert(index);
            return None;
        }
        let x = coordinate.re;
        if let Some(last) = self.last {
            if x < last {
                self.descending.get_or_insert(index);
            } else if x > last {
                self.ascending.get_or_insert(index);
            }
        }
        self.first.get_or_insert(x);
        self.last = Some(x);
        (self.nonfinite.is_none() && !(self.ascending.is_some() && self.descending.is_some()))
            .then_some(x)
    }

    pub(in crate::analysis) const fn is_nondecreasing(&self) -> bool {
        self.descending.is_none()
    }

    fn finite(&self) -> SpiceResult<()> {
        if let Some(index) = self.nonfinite {
            return Err(nonfinite_axis(self.name, index));
        }
        Ok(())
    }

    pub(in crate::analysis) fn range(&self) -> SpiceResult<Option<(Real, Real)>> {
        self.finite()?;
        if let Some(index) = self.descending {
            return Err(descending_axis(self.name, index));
        }
        Ok(self.first.zip(self.last))
    }

    /// The physical coordinate interval, in ascending order regardless of
    /// traversal direction. Repeated inner sweeps have no unique interval
    /// interpolation and are rejected rather than choosing one arbitrarily.
    pub(in crate::analysis) fn query_range(&self) -> SpiceResult<Option<(Real, Real)>> {
        self.finite()?;
        if let (Some(up), Some(down)) = (self.ascending, self.descending) {
            return Err(SpiceError::Unsupported {
                feature: format!(
                    "the {} axis changes direction at point {}; objective AT requires a monotonic scale",
                    self.name,
                    up.max(down)
                ),
                location: None,
            });
        }
        Ok(self
            .first
            .zip(self.last)
            .map(|(first, last)| (first.min(last), first.max(last))))
    }
}

/// Reuse `OrderedAxis` interpolation, including its endpoint and signed-zero
/// rules, without allocating or constructing a synthetic plot.
fn interpolate(x0: Real, y0: Real, x1: Real, y1: Real, x: Real) -> Real {
    // This function is only called for an interior clipped boundary or query,
    // for which the two finite coordinates are already ordered and distinct.
    crate::analysis::ordered_samples::interpolate(x0, y0, x1, y1, x)
}

fn read(column: &Column, row: &[Complex], card: &MeasureCard) -> SpiceResult<Real> {
    column
        .value(row)
        .map(|value| value.re)
        .map_err(|error| match error {
            SpiceError::Numerical { message, .. } => SpiceError::Numerical {
                context: format!("measure {}: {}", card.name, column.name()),
                message,
            },
            other => other,
        })
}

/// Scalar or complex point query. Exact coordinate groups compare every
/// duplicate, and only the adjacent left/right limits are used otherwise.
/// Errors on unrelated samples never invalidate the query.
pub(in crate::analysis) struct PointQuery {
    at: Real,
    real_only: bool,
    previous: Option<Sample>,
    result: Option<Result<Complex, QueryFailure>>,
    exact: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(in crate::analysis) enum QueryFailure {
    Outside,
    Discontinuous,
    Invalid(SpiceError),
}

impl PointQuery {
    pub(in crate::analysis) const fn new(at: Real, real_only: bool) -> Self {
        Self {
            at,
            real_only,
            previous: None,
            result: None,
            exact: false,
        }
    }

    pub(in crate::analysis) fn observe(
        &mut self,
        x: Real,
        sample: impl FnOnce() -> SpiceResult<Complex>,
    ) {
        if same_coordinate(x, self.at) {
            let current = sample().map_err(QueryFailure::Invalid).map(|value| {
                if self.real_only {
                    Complex::real(value.re)
                } else {
                    value
                }
            });
            if !self.exact {
                self.exact = true;
                self.previous = None;
                self.result = Some(current);
            } else if let Some(Ok(first)) = self.result {
                self.result = Some(current.and_then(|next| {
                    if same_coordinate(first.re, next.re) && same_coordinate(first.im, next.im) {
                        Ok(first)
                    } else {
                        Err(QueryFailure::Discontinuous)
                    }
                }));
            }
            return;
        }
        if self.result.is_some() {
            return;
        }
        if let Some((x0, before)) = self.previous.take()
            && ((x0 < self.at) != (x < self.at))
        {
            self.result = Some(
                before
                    .and_then(|first| {
                        let next = sample()?;
                        // Numerical interpolation always receives its
                        // coordinates in increasing order. The chosen
                        // samples are adjacent in actual traversal order.
                        let ((lo, left), (hi, right)) = if x0 < x {
                            ((x0, first), (x, next))
                        } else {
                            ((x, next), (x0, first))
                        };
                        let value = Complex::new(
                            interpolate(lo, left.re, hi, right.re, self.at),
                            if self.real_only {
                                0.
                            } else {
                                interpolate(lo, left.im, hi, right.im, self.at)
                            },
                        );
                        if value.is_finite() {
                            Ok(value)
                        } else {
                            Err(SpiceError::Numerical {
                                context: "point query".into(),
                                message: "nonfinite interpolated value".into(),
                            })
                        }
                    })
                    .map_err(QueryFailure::Invalid),
            );
        } else {
            self.previous = Some((x, sample()));
        }
    }

    pub(in crate::analysis) fn finish(self) -> Result<Complex, QueryFailure> {
        self.result.unwrap_or(Err(QueryFailure::Outside))
    }
}

/// Compiled crossing operands and lower/upper window. Delays are flattened
/// into a lower bound once; dynamic bound expressions stay with the retained
/// fallback because prior results are not known while the rows arrive.
pub(in crate::analysis) enum EventPlan {
    At {
        at: Real,
        window: MeasureWindow,
    },
    Cross {
        left: Column,
        right: Option<Column>,
        threshold: Real,
        transition: MeasureTransition,
        window: MeasureWindow,
    },
}

impl EventPlan {
    fn bind(
        plot: &Plot,
        event: &MeasureEvent,
        mut window: MeasureWindow,
        card: &MeasureCard,
    ) -> SpiceResult<Self> {
        let mut event = event;
        while let MeasureEvent::Delayed { event: inner, td } = event {
            if !td.is_finite() {
                return Err(unsupported(card, "TD must be finite".into()));
            }
            window.from = Some(window.from.map_or(*td, |from| from.max(*td)));
            event = inner;
        }
        match event {
            MeasureEvent::At { at, .. } => Ok(Self::At { at: *at, window }),
            MeasureEvent::Crossing {
                operand,
                value,
                transition,
                ..
            } => Ok(Self::Cross {
                left: resolve_request(plot, operand)?,
                right: None,
                threshold: *value,
                transition: *transition,
                window,
            }),
            MeasureEvent::VectorCrossing {
                operand,
                reference,
                transition,
            } => Ok(Self::Cross {
                left: resolve_request(plot, operand)?,
                right: Some(resolve_request(plot, reference)?),
                threshold: 0.,
                transition: *transition,
                window,
            }),
            MeasureEvent::Delayed { .. } => Err(unsupported(card, "invalid delayed event".into())),
        }
    }

    fn inputs(&self, out: &mut Vec<usize>) {
        if let Self::Cross { left, right, .. } = self {
            out.extend(left.input_columns());
            if let Some(right) = right {
                out.extend(right.input_columns());
            }
        }
    }
}

enum EventState<'a> {
    At {
        at: Real,
        window: MeasureWindow,
    },
    Cross {
        plan: &'a EventPlan,
        selector: CrossingSelector,
        previous: Option<(Real, SpiceResult<Real>)>,
        error: Option<SpiceError>,
        in_window: bool,
    },
}

impl<'a> EventState<'a> {
    fn new(plan: &'a EventPlan, card: &MeasureCard) -> SpiceResult<Self> {
        match plan {
            EventPlan::At { at, window } => Ok(Self::At {
                at: *at,
                window: *window,
            }),
            EventPlan::Cross {
                transition,
                threshold,
                ..
            } => {
                if !threshold.is_finite() {
                    return Err(unsupported(
                        card,
                        "crossing threshold must be finite".into(),
                    ));
                }
                Ok(Self::Cross {
                    plan,
                    selector: CrossingSelector::new(*transition, card)?,
                    previous: None,
                    error: None,
                    in_window: false,
                })
            }
        }
    }

    fn observe(&mut self, x: Real, row: &[Complex], card: &MeasureCard) {
        let Self::Cross {
            plan:
                EventPlan::Cross {
                    left,
                    right,
                    threshold,
                    window,
                    ..
                },
            selector,
            previous,
            error,
            in_window,
        } = self
        else {
            return;
        };
        let bounds = raw_window(window);
        if error.is_some() || x > bounds.to {
            return;
        }
        let value = read(left, row, card).and_then(|a| {
            let value = a - right
                .as_ref()
                .map_or(Ok(0.), |column| read(column, row, card))?;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(nonfinite(card, "WHEN", "vector difference overflow".into()))
            }
        });
        if x >= bounds.from {
            *in_window = true;
            if let Some((x0, before)) = previous.as_ref() {
                let step = before.as_ref().map_err(Clone::clone).and_then(|before| {
                    let after = value.as_ref().map_err(Clone::clone)?;
                    selector.include((*x0, *before), (x, *after), *threshold, &bounds, card)
                });
                if let Err(failure) = step {
                    *error = Some(failure);
                }
            }
        }
        *previous = Some((x, value));
    }

    const fn selected(&self) -> Option<Real> {
        match self {
            Self::At { at, .. } => Some(*at),
            Self::Cross { selector, .. } => selector.selected,
        }
    }

    fn finish(self, axis: &AxisTracker, card: &MeasureCard) -> SpiceResult<Real> {
        let range = axis.range()?;
        match self {
            Self::At { at, window } => {
                let window = Window::from_range(&window, range, axis.name, card)?;
                if !at.is_finite() || at < window.from || at > window.to {
                    return Err(unsupported(
                        card,
                        "the event AT= is outside the measurement window (out of interval)".into(),
                    ));
                }
                Ok(at)
            }
            Self::Cross {
                plan:
                    EventPlan::Cross {
                        window, transition, ..
                    },
                selector,
                error,
                in_window,
                ..
            } => {
                let bounds = Window::from_range(window, range, axis.name, card)?;
                if !in_window {
                    return Err(no_samples(card));
                }
                if let Some(error) = error {
                    return Err(error);
                }
                selector.selected.ok_or_else(|| unsupported(card, format!(".measure {}: no {} crossing found from the window's lower bound [{}, {}]", card.name, transition.name(), bounds.from, bounds.to)))
            }
            Self::Cross { .. } => Err(unsupported(card, "invalid bound event".into())),
        }
    }
}

fn raw_window(window: &MeasureWindow) -> Window {
    Window {
        from: window.from.unwrap_or(Real::NEG_INFINITY),
        to: window.to.unwrap_or(Real::INFINITY),
    }
}

fn no_samples(card: &MeasureCard) -> SpiceError {
    unsupported(
        card,
        format!(".measure {}: window covers no stored sample", card.name),
    )
}

/// Collect only the immediately preceding sample, overlapping Simpson panel
/// and extrema. A window wholly within a bracket is still rejected; an
/// interpolation is emitted only after a real in-window sample is seen.
struct StatisticState {
    extrema: Option<Extrema>,
    integral: Option<Integral>,
    previous: Option<(Real, SpiceResult<Real>)>,
    seen: bool,
    done: bool,
}

impl StatisticState {
    const fn new(statistic: MeasureStatistic) -> Self {
        Self {
            extrema: None,
            integral: if matches!(
                statistic,
                MeasureStatistic::Avg | MeasureStatistic::Rms | MeasureStatistic::Integ
            ) {
                Some(Integral::new(statistic))
            } else {
                None
            },
            previous: None,
            seen: false,
            done: false,
        }
    }

    fn observe(
        &mut self,
        x: Real,
        sample: impl FnOnce() -> SpiceResult<Real>,
        window: &MeasureWindow,
    ) -> SpiceResult<()> {
        let bounds = raw_window(window);
        if self.done {
            return Ok(());
        }
        if let Some(integral) = &mut self.integral {
            if x < bounds.from {
                self.previous = Some((x, sample()));
            } else if x <= bounds.to {
                if !self.seen
                    && x > bounds.from
                    && let Some((x0, before)) = self.previous.take()
                {
                    let before = before?;
                    let value = sample()?;
                    integral.push((bounds.from, interpolate(x0, before, x, value, bounds.from)));
                    integral.push((x, value));
                    self.previous = Some((x, Ok(value)));
                } else {
                    let value = sample()?;
                    integral.push((x, value));
                    self.previous = Some((x, Ok(value)));
                }
                self.seen = true;
            } else {
                if self.seen
                    && let Some((x0, before)) = self.previous.take()
                    && x0 < bounds.to
                {
                    integral.push((bounds.to, interpolate(x0, before?, x, sample()?, bounds.to)));
                }
                self.done = true;
            }
        } else if x >= bounds.from && x <= bounds.to {
            let value = sample()?;
            if let Some(extrema) = &mut self.extrema {
                extrema.include(value, x);
            } else {
                self.extrema = Some(Extrema::new(value, x));
            }
            self.seen = true;
        }
        Ok(())
    }
    fn finish(
        self,
        statistic: MeasureStatistic,
        window: Window,
        card: &MeasureCard,
        error: Option<SpiceError>,
    ) -> SpiceResult<(Real, Option<Real>, Option<MeasureSpan>)> {
        let span = MeasureSpan {
            from: window.from,
            to: window.to,
        };
        let width = if self.integral.is_some() {
            Some(span_width(span, card)?)
        } else {
            None
        };
        if let Some(error) = error {
            return Err(error);
        }
        if !self.seen {
            return Err(no_samples(card));
        }
        if let Some(integral) = self.integral {
            let value = integral.finish(
                width.ok_or_else(|| SpiceError::circuit("missing integration width"))?,
                card,
            )?;
            Ok((value, None, Some(span)))
        } else if let Some(extrema) = self.extrema {
            let (value, at) = extrema.result(statistic)?;
            Ok((value, at, None))
        } else {
            Err(no_samples(card))
        }
    }
}

/// Immutable bound operation, reusable for every candidate with this schema.
pub(in crate::analysis) enum BoundMeasure {
    Find {
        column: Column,
        at: Real,
        window: MeasureWindow,
    },
    Statistic {
        column: Column,
        statistic: MeasureStatistic,
        window: MeasureWindow,
    },
    AtEvent {
        column: Column,
        event: Box<EventPlan>,
        derivative: bool,
        window: MeasureWindow,
    },
    When {
        event: EventPlan,
        window: MeasureWindow,
    },
    Delay {
        trig: EventPlan,
        targ: Box<EventPlan>,
        window: MeasureWindow,
    },
}

impl BoundMeasure {
    pub(in crate::analysis) fn bind(plot: &Plot, card: &MeasureCard) -> SpiceResult<Option<Self>> {
        Ok(Some(match &card.request {
            MeasureRequest::Find {
                operand,
                at,
                window,
                ..
            } => Self::Find {
                column: resolve_request(plot, operand)?,
                at: *at,
                window: *window,
            },
            MeasureRequest::AtEvent {
                operand,
                event,
                derivative,
                window,
            } => Self::AtEvent {
                column: resolve_request(plot, operand)?,
                event: Box::new(EventPlan::bind(plot, event, *window, card)?),
                derivative: *derivative,
                window: *window,
            },
            MeasureRequest::Statistic {
                statistic,
                operand,
                window,
            } => {
                if matches!(
                    statistic,
                    MeasureStatistic::PhaseMargin | MeasureStatistic::GainMargin
                ) && (axis_name(plot, card.analysis)? != "frequency"
                    || !plot.flags().is_complex())
                {
                    return Err(unsupported(
                        card,
                        "margin measurements require complex AC/SP data".into(),
                    ));
                }
                Self::Statistic {
                    column: resolve_request(plot, operand)?,
                    statistic: *statistic,
                    window: *window,
                }
            }
            MeasureRequest::When { event, window } => Self::When {
                event: EventPlan::bind(plot, event, *window, card)?,
                window: *window,
            },
            MeasureRequest::TrigTarg { trig, targ, window } => Self::Delay {
                trig: EventPlan::bind(plot, trig, *window, card)?,
                targ: Box::new(EventPlan::bind(plot, targ, *window, card)?),
                window: *window,
            },
            _ => return Ok(None),
        }))
    }

    pub(in crate::analysis) fn inputs(&self, out: &mut Vec<usize>) {
        match self {
            Self::Find { column, .. } | Self::Statistic { column, .. } => {
                out.extend(column.input_columns());
            }
            Self::AtEvent { column, event, .. } => {
                out.extend(column.input_columns());
                event.inputs(out);
            }
            Self::When { event, .. } => event.inputs(out),
            Self::Delay { trig, targ, .. } => {
                trig.inputs(out);
                targ.inputs(out);
            }
        }
    }

    pub(in crate::analysis) fn start<'a>(
        &'a self,
        card: &'a MeasureCard,
    ) -> SpiceResult<MeasureState<'a>> {
        let state = match self {
            Self::Find { at, .. } => Reduction::Find(PointQuery::new(*at, true)),
            Self::Statistic { statistic, .. }
                if matches!(
                    statistic,
                    MeasureStatistic::PhaseMargin | MeasureStatistic::GainMargin
                ) =>
            {
                Reduction::Margin(MarginScan::new(*statistic == MeasureStatistic::PhaseMargin))
            }
            Self::Statistic { statistic, .. } => {
                Reduction::Statistic(StatisticState::new(*statistic))
            }
            Self::AtEvent { event, .. } => Reduction::AtEvent(Box::new(EventQueryState {
                event: EventState::new(event, card)?,
                history: LocalHistory::new(),
            })),
            Self::When { event, .. } => Reduction::When(EventState::new(event, card)?),
            Self::Delay { trig, targ, .. } => {
                Reduction::Delay(EventState::new(trig, card)?, EventState::new(targ, card)?)
            }
        };
        Ok(MeasureState {
            plan: self,
            card,
            state,
            error: None,
        })
    }
}

struct EventQueryState<'a> {
    event: EventState<'a>,
    history: LocalHistory,
}

enum Reduction<'a> {
    AtEvent(Box<EventQueryState<'a>>),
    Find(PointQuery),
    Statistic(StatisticState),
    Margin(MarginScan),
    When(EventState<'a>),
    Delay(EventState<'a>, EventState<'a>),
}

pub(in crate::analysis) struct MeasureState<'a> {
    plan: &'a BoundMeasure,
    card: &'a MeasureCard,
    state: Reduction<'a>,
    error: Option<SpiceError>,
}

impl MeasureState<'_> {
    pub(in crate::analysis) fn observe(&mut self, x: Real, row: &[Complex]) {
        if self.error.is_some() {
            return;
        }
        match (&mut self.state, self.plan) {
            (Reduction::Find(query), BoundMeasure::Find { column, .. }) => {
                query.observe(x, || read(column, row, self.card).map(Complex::real));
            }
            (Reduction::Margin(scan), BoundMeasure::Statistic { column, window, .. }) => {
                if scan.selected.is_none() {
                    self.error = column
                        .value(row)
                        .and_then(|value| scan.observe(x, value, &raw_window(window), self.card))
                        .err();
                }
            }
            (Reduction::Statistic(state), BoundMeasure::Statistic { column, window, .. }) => {
                self.error = state
                    .observe(x, || read(column, row, self.card), window)
                    .err();
            }
            (Reduction::AtEvent(state), BoundMeasure::AtEvent { column, .. }) => {
                state.event.observe(x, row, self.card);
                state.history.observe(x, read(column, row, self.card));
                if let Some(at) = state.event.selected() {
                    state.history.select(at);
                }
            }
            (Reduction::When(event), _) => event.observe(x, row, self.card),
            (Reduction::Delay(trig, targ), _) => {
                trig.observe(x, row, self.card);
                targ.observe(x, row, self.card);
            }
            _ => self.error = Some(SpiceError::circuit("measurement reduction mismatch")),
        }
    }

    pub(in crate::analysis) fn finish(self, axis: &AxisTracker) -> SpiceResult<Measurement> {
        let card = self.card;
        let mut result = Measurement {
            name: card.name.clone(),
            value: 0.,
            unit: axis.name.into(),
            at: None,
            window: None,
            events: None,
        };
        let range = axis.range()?;
        match (self.state, self.plan) {
            (Reduction::Find(query), BoundMeasure::Find { column, at, window }) => {
                let window = Window::from_range(window, range, axis.name, card)?;
                if !at.is_finite() || *at < window.from || *at > window.to {
                    return Err(unsupported(
                        card,
                        "AT= is outside the measurement window (out of interval)".into(),
                    ));
                }
                result.value = query_value(query, card)?;
                result.at = Some(*at);
                result.unit = column.unit().into();
            }
            (
                Reduction::Margin(scan),
                BoundMeasure::Statistic {
                    column,
                    statistic,
                    window,
                },
            ) => {
                Window::from_range(window, range, axis.name, card)?;
                if let Some(error) = self.error {
                    return Err(error);
                }
                let (value, at) = scan.finish(card)?;
                result.value = value;
                result.at = Some(at);
                result.unit = super::statistic_unit(*statistic, column.unit(), axis.name);
            }
            (
                Reduction::Statistic(state),
                BoundMeasure::Statistic {
                    column,
                    statistic,
                    window,
                },
            ) => {
                let window = Window::from_range(window, range, axis.name, card)?;
                let (value, at, span) = state.finish(*statistic, window, card, self.error)?;
                result.value = value;
                result.at = at;
                result.window = span;
                result.unit = super::statistic_unit(*statistic, column.unit(), axis.name);
            }
            (
                Reduction::AtEvent(state),
                BoundMeasure::AtEvent {
                    column,
                    derivative,
                    window,
                    ..
                },
            ) => {
                Window::from_range(window, range, axis.name, card)?;
                let at = state.event.finish(axis, card)?;
                result.value = state.history.finish(at, *derivative, card)?;
                result.at = Some(at);
                result.unit = if *derivative {
                    format!("{}/{}", column.unit(), axis.name)
                } else {
                    column.unit().into()
                };
            }
            (Reduction::When(event), BoundMeasure::When { window, .. }) => {
                Window::from_range(window, range, axis.name, card)?;
                result.value = event.finish(axis, card)?;
            }
            (Reduction::Delay(trig, targ), BoundMeasure::Delay { window, .. }) => {
                Window::from_range(window, range, axis.name, card)?;
                let trig = trig.finish(axis, card)?;
                let targ = targ.finish(axis, card)?;
                result.value = targ - trig;
                result.events = Some(MeasureEvents { trig, targ });
            }
            _ => return Err(SpiceError::circuit("measurement reduction mismatch")),
        }
        if !result.value.is_finite() {
            return Err(nonfinite(
                card,
                "stream",
                "the computed result is not finite".into(),
            ));
        }
        Ok(result)
    }
}

fn query_value(query: PointQuery, card: &MeasureCard) -> SpiceResult<Real> {
    query.finish().map(|v| v.re).map_err(|error| match error {
        QueryFailure::Outside => unsupported(
            card,
            "AT= is outside the measurement axis (out of interval)".into(),
        ),
        QueryFailure::Discontinuous => {
            unsupported(card, "the operand is discontinuous at AT=".into())
        }
        QueryFailure::Invalid(error) => error,
    })
}
