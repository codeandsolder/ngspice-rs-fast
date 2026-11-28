//! Streaming linear interpolation of accepted output rows. No interpolated
//! value can feed back into Newton iteration or device/integration history.

use super::schedule::ScheduledGrid;
use super::{TransientSchedule, failure};
use crate::analysis::{Plot, PointSink};
use crate::primitives::{Complex, Real, SpiceResult};

pub(super) struct InterpolatedSink<'a> {
    sink: &'a mut dyn PointSink,
    grid: ScheduledGrid,
    stop: Real,
    omit_zero: bool,
    selection: Option<Vec<usize>>,
    strip_scale: bool,
    width: usize,
    previous: Vec<Complex>,
    row: Vec<Complex>,
    complete: bool,
}

impl<'a> InterpolatedSink<'a> {
    pub(super) fn new(
        schedule: &TransientSchedule,
        omit_zero: bool,
        sink: &'a mut dyn PointSink,
    ) -> SpiceResult<Self> {
        Ok(Self {
            sink,
            grid: schedule.grid()?,
            stop: schedule.bounds()[1],
            omit_zero,
            selection: None,
            strip_scale: false,
            width: 0,
            previous: Vec::new(),
            row: Vec::new(),
            complete: false,
        })
    }

    fn emit(&mut self, values: &[Complex], time: Real) -> SpiceResult<()> {
        if self.omit_zero && time == 0. {
            return Ok(());
        }
        let offset = usize::from(self.strip_scale);
        self.sink.point(&values[offset..])
    }

    fn interpolate(&mut self, values: &[Complex], time: Real, at: Real) -> SpiceResult<()> {
        let previous = self
            .previous
            .first()
            .ok_or_else(|| failure("output grid precedes the first solved point"))?
            .re;
        if at < previous || time <= previous {
            return Err(failure("invalid interpolation bracket"));
        }
        let fraction = (at - previous) / (time - previous);
        self.row.clear();
        self.row.push(Complex::real(at));
        // Convex weighting cannot overflow on opposite-signed finite endpoints.
        let blend = |a: Real, b: Real| (1. - fraction).mul_add(a, fraction * b);
        self.row.extend(
            self.previous[1..]
                .iter()
                .zip(&values[1..])
                .map(|(a, b)| Complex::new(blend(a.re, b.re), blend(a.im, b.im))),
        );
        let offset = usize::from(self.strip_scale);
        if !self.omit_zero || at != 0. {
            self.sink.point(&self.row[offset..])?;
        }
        Ok(())
    }
}

impl PointSink for InterpolatedSink<'_> {
    fn begin(&mut self, header: &Plot) -> SpiceResult<()> {
        if !header.variables().first().is_some_and(|v| v.name == "time") {
            return Err(failure("interpolated output requires a time scale"));
        }
        self.sink.begin(header)?;
        self.width = header.variable_count();
        if let Some(columns) = self.sink.selected_columns() {
            if columns.iter().any(|&i| i >= self.width) || columns.windows(2).any(|p| p[0] >= p[1])
            {
                return Err(failure("invalid interpolated output column selection"));
            }
            let mut selected = columns.to_vec();
            self.strip_scale = selected.first() != Some(&0);
            if self.strip_scale {
                selected.insert(0, 0);
            }
            self.width = selected.len();
            self.selection = Some(selected);
        }
        Ok(())
    }

    fn selected_columns(&self) -> Option<&[usize]> {
        self.selection.as_deref()
    }

    fn point(&mut self, values: &[Complex]) -> SpiceResult<()> {
        if values.len() != self.width || values.is_empty() || !values[0].is_real() {
            return Err(failure("invalid interpolated output row"));
        }
        let time = values[0].re;
        if !time.is_finite() || time < 0. || self.previous.first().is_some_and(|p| p.re >= time) {
            return Err(failure("nonincreasing accepted output time"));
        }
        while !self.complete && self.grid.time() <= time {
            let at = self.grid.time();
            if at.to_bits() == time.to_bits() {
                self.emit(values, at)?;
            } else {
                self.interpolate(values, time, at)?;
            }
            if at.to_bits() == self.stop.to_bits() {
                self.complete = true;
            } else {
                self.grid.advance(true)?;
            }
        }
        self.previous.clear();
        self.previous.extend_from_slice(values);
        Ok(())
    }

    fn end(&mut self) -> SpiceResult<()> {
        if !self.complete {
            return Err(failure("accepted output ended before interpolation grid"));
        }
        self.sink.end()
    }
}
