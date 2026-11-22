//! Event-aware transient comparison on a shared physical time grid.
//!
//! C (adaptive trap/Gear) and Rust (the companion trap/Gear driver) choose
//! different internal timepoints, so sample sequences are never compared. Both plots are evaluated at common output times instead:
//!
//! * **Smooth interval**: a value at `t` is the linear interpolation of the two
//!   neighbouring samples of that plot. Interpolation is refused (an error, not
//!   a pass) if a source breakpoint lies strictly between those two samples.
//! * **Breakpoint**: a discontinuity or slope change of a source (PWL corner,
//!   PULSE edge, from the elaborated circuit via [`breakpoints`], never guessed from
//!   data). The left and right limits are compared separately. A plot carries
//!   two samples at the instant for a true jump (first = left, second = right);
//!   a single sample is taken as both limits, which is exact for the continuous
//!   quantities (capacitor voltage, inductor current) and fails loudly for a
//!   discontinuous one that the other plot resolves into two distinct limits.
//!   A plot with no sample at a breakpoint cannot supply a limit and is
//!   rejected: ngspice lands a step on every breakpoint (`CKTbreak`,
//!   `dctran.c`) and the Rust driver must as well.
//!
//! Tolerances are in [`compare::TRAN`]. Interpolating the denser plot adds
//! `O(h^2 v'')` error, so the comparison is only meaningful when the sample
//! spacing is small against the circuit time constants; that is the caller's
//! fixture design responsibility and is not hidden by a looser tolerance.

use std::collections::BTreeMap;

use ngspice_rs::analysis::{Plot, PlotFlags};
use ngspice_rs::devices::{Circuit, TransientTiming};
use ngspice_rs::primitives::{Complex, exact_nonnegative_integer};

use super::compare::{self, TranTolerance};

/// Time coincidence is judged within `TIME_EPS_REL * stop`: far below any
/// physical timestep, far above the 17-digit rawfile rounding of `t` and of
/// `td + n*per + corner` sums.
const TIME_EPS_REL: f64 = 1e-9;
/// Upper bound on comparison instants and generated breakpoints.
const MAX_POINTS: u32 = 2_000_000;

/// The shared output grid: `start, step, 2*step, ... , stop` (each time is
/// `i*step`, not an accumulated sum). Breakpoints inside `(start, stop)` are
/// inserted by [`transient`] and compared as left/right limit pairs.
///
/// `start` is 0 for an ordinary run. With `uic` C writes no `t = 0` row (its
/// first row is the first accepted step); the caller then passes that first
/// time, and **both** plots must begin exactly there: the instant is compared as
/// an ordinary sample, and nothing before it is interpolated or extrapolated.
#[derive(Debug, Clone, Copy)]
pub(super) struct Grid {
    pub(super) start: f64,
    pub(super) stop: f64,
    pub(super) step: f64,
}

/// What a successful comparison covered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Summary {
    /// Interior (smooth) or end-point instants compared.
    pub(super) instants: usize,
    /// One-sided breakpoint limits compared (two per breakpoint).
    pub(super) limits: usize,
    /// Largest `|error| / bound` seen over every compared value (<= 1).
    pub(super) worst_ratio: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Side {
    Interior,
    Left,
    Right,
}

/// Bound rows and weights for one instant. Every signal uses the same plan,
/// and sampling two plots needs only two plans live at a time.
#[derive(Debug)]
enum SamplePlan<'a> {
    Exact(&'a [Complex]),
    Linear {
        lower: &'a [Complex],
        upper: &'a [Complex],
        lower_weight: f64,
        upper_weight: f64,
    },
}

impl SamplePlan<'_> {
    fn value(&self, column: usize) -> f64 {
        match self {
            Self::Exact(row) => row[column].re,
            Self::Linear {
                lower,
                upper,
                lower_weight,
                upper_weight,
            } => {
                // Preserve the original arithmetic order and rounding.
                let left = lower_weight * lower[column].re;
                let right = upper_weight * upper[column].re;
                left + right
            }
        }
    }
}

type EventRows = Option<(usize, usize)>;

/// Index the first and last original row matching each event. Iterate rows
/// in storage order: small time reversals are legal at corners, so a binary
/// search of the time axis cannot establish the one-sided limits. A row may
/// belong to two nearby event windows; retain both rather than only the
/// first matching event. The event timeline is strictly ordered.
fn index_event_rows(times: &[f64], events: &[f64], eps: f64) -> Vec<EventRows> {
    let mut rows: Vec<EventRows> = vec![None; events.len()];
    if !events.is_empty() {
        for (row, &time) in times.iter().enumerate() {
            let start = first_possible_breakpoint(events, time, eps);
            for (&event, endpoints) in events[start..].iter().zip(&mut rows[start..]) {
                if (event - time).abs() > eps {
                    break;
                }
                *endpoints = Some((endpoints.map_or(row, |(first, _)| first), row));
            }
        }
    }
    rows
}

/// One validated transient plot.
struct Series<'a> {
    label: &'static str,
    plot: &'a Plot,
    columns: BTreeMap<String, usize>,
    times: Vec<f64>,
    breakpoints: &'a [f64],
    event_rows: Vec<EventRows>,
    eps: f64,
}

impl<'a> Series<'a> {
    fn new(
        label: &'static str,
        plot: &'a Plot,
        breakpoints: &'a [f64],
        grid: &Grid,
        eps: f64,
    ) -> Result<Self, String> {
        let columns = compare::validate(plot, label)?;
        if plot.flags() != PlotFlags::Real {
            return Err(format!("{label}: transient data must be a real plot"));
        }
        let &time = columns
            .get("time")
            .ok_or_else(|| format!("{label}: missing axis 'time'"))?;
        if plot.variables()[time].unit != "time" {
            return Err(format!("{label}: axis 'time' must have unit 'time'"));
        }
        let times: Vec<f64> = plot.points().iter().map(|row| row[time].re).collect();
        if (times[0] - grid.start).abs() > eps {
            return Err(format!(
                "{label}: transient starts at {:e}, not {:e}",
                times[0], grid.start
            ));
        }
        let end = times[times.len() - 1];
        if (end - grid.stop).abs() > eps {
            return Err(format!(
                "{label}: end time {end:e} differs from requested stop {:e}",
                grid.stop
            ));
        }
        let mut series = Self {
            label,
            plot,
            columns,
            times,
            breakpoints,
            event_rows: Vec::new(),
            eps,
        };
        series.check_times()?;
        series.event_rows = index_event_rows(&series.times, breakpoints, eps);
        Ok(series)
    }

    /// Times must not decrease; two samples may share an instant only at a
    /// declared breakpoint (left/right limits), and never three.
    fn check_times(&self) -> Result<(), String> {
        for (index, pair) in self.times.windows(2).enumerate() {
            let step = pair[1] - pair[0];
            if step < -self.eps {
                return Err(format!(
                    "{}: time decreases at point {}: {:e} -> {:e}",
                    self.label,
                    index + 1,
                    pair[0],
                    pair[1]
                ));
            }
            if step.abs() <= self.eps {
                if self.breakpoint_near(pair[0]).is_none() {
                    return Err(format!(
                        "{}: repeated time {:e} at point {} is not a declared breakpoint",
                        self.label,
                        pair[0],
                        index + 1
                    ));
                }
                if index > 0 && (self.times[index - 1] - pair[0]).abs() <= self.eps {
                    return Err(format!(
                        "{}: more than two samples at breakpoint {:e}",
                        self.label, pair[0]
                    ));
                }
            }
        }
        Ok(())
    }

    fn breakpoint_near(&self, t: f64) -> Option<f64> {
        nearby_breakpoint(self.breakpoints, t, self.eps)
    }

    /// Bind the sample once, independent of the requested signal.
    fn sample(&self, t: f64, side: Side) -> Result<SamplePlan<'a>, String> {
        let row = |index: usize| self.plot.points()[index].as_slice();
        if let Some(index) = nearby_breakpoint_index(self.breakpoints, t, self.eps) {
            let b = self.breakpoints[index];
            let Some((first, last)) = self.event_rows[index] else {
                return Err(format!(
                    "{}: no sample at breakpoint {b:e}; cannot form a one-sided limit",
                    self.label
                ));
            };
            return match side {
                Side::Left => Ok(SamplePlan::Exact(row(first))),
                Side::Right => Ok(SamplePlan::Exact(row(last))),
                Side::Interior => Err(format!(
                    "{}: interior evaluation requested at breakpoint {b:e}",
                    self.label
                )),
            };
        }
        // Keep the original interior bracket-selection rule, including its
        // behavior for the tiny reversals allowed by check_times. Only the
        // event lookup above required a complete scan of matching rows.
        let upper = self.times.partition_point(|&x| x < t - self.eps);
        if upper == self.times.len() {
            return Err(format!("{}: time {t:e} is beyond the samples", self.label));
        }
        if (self.times[upper] - t).abs() <= self.eps {
            return Ok(SamplePlan::Exact(row(upper)));
        }
        if upper == 0 {
            return Err(format!("{}: time {t:e} precedes the samples", self.label));
        }
        let (t0, t1) = (self.times[upper - 1], self.times[upper]);
        let next = self.breakpoints.partition_point(|&b| b <= t0 + self.eps);
        if let Some(b) = self.breakpoints.get(next).filter(|&&b| b < t1 - self.eps) {
            return Err(format!(
                "{}: samples [{t0:e}, {t1:e}] straddle breakpoint {b:e}; refusing to interpolate across it",
                self.label
            ));
        }
        let fraction = (t - t0) / (t1 - t0);
        Ok(SamplePlan::Linear {
            lower: row(upper - 1),
            upper: row(upper),
            lower_weight: 1.0 - fraction,
            upper_weight: fraction,
        })
    }
}

/// Find the first matching corner in an already validated sorted timeline.
/// Use the original subtraction predicate rather than rounding a new bound.
fn first_possible_breakpoint(breakpoints: &[f64], time: f64, eps: f64) -> usize {
    breakpoints.partition_point(|&event| event < time && time - event > eps)
}

fn nearby_breakpoint_index(breakpoints: &[f64], time: f64, eps: f64) -> Option<usize> {
    let index = first_possible_breakpoint(breakpoints, time, eps);
    breakpoints
        .get(index)
        .filter(|&&event| (event - time).abs() <= eps)
        .map(|_| index)
}

fn nearby_breakpoint(breakpoints: &[f64], time: f64, eps: f64) -> Option<f64> {
    nearby_breakpoint_index(breakpoints, time, eps).map(|index| breakpoints[index])
}

fn comparison_instants(
    grid: &Grid,
    breakpoints: &[f64],
    eps: f64,
) -> Result<Vec<(f64, Side)>, String> {
    if !(grid.stop.is_finite() && grid.stop > 0.0 && grid.step.is_finite() && grid.step > 0.0) {
        return Err("grid stop and step must be finite and positive".into());
    }
    if !(grid.start.is_finite() && grid.start >= 0.0 && grid.start < grid.stop - eps) {
        return Err("grid start must be finite and lie in [0, stop)".into());
    }
    let count = (grid.stop / grid.step + 1e-9).floor();
    if count >= f64::from(MAX_POINTS) {
        return Err(format!("grid needs more than {MAX_POINTS} points"));
    }
    let count = exact_nonnegative_integer::<u32>(count)
        .ok_or_else(|| "invalid grid point count".to_owned())?;
    let mut previous = grid.start;
    for &b in breakpoints {
        if !(b.is_finite() && b > eps && b < grid.stop - eps && b > previous + eps) {
            return Err(format!(
                "breakpoint {b:e} must be finite, strictly inside (start, stop) and strictly increasing"
            ));
        }
        previous = b;
    }
    let limit_count = breakpoints
        .len()
        .checked_mul(2)
        .ok_or("too many breakpoint limits")?;
    check_event_capacity(2, limit_count)?;
    let mandatory = limit_count + 2;
    let mut instants = Vec::with_capacity(mandatory);
    instants.push((grid.start, Side::Interior));
    let mut interiors = 0;
    let mut events = breakpoints.iter().copied().peekable();
    // Merge monotonically increasing grid times with the event timeline.
    // A grid time near a breakpoint is omitted, using exactly the original
    // subtraction test (not a rounded t +/- eps bound).
    for index in 0..=count {
        let t = f64::from(index) * grid.step;
        if !(t > grid.start + eps && t < grid.stop - eps) {
            continue;
        }
        while events.peek().is_some_and(|&b| b < t && t - b > eps) {
            if let Some(b) = events.next() {
                instants.extend([(b, Side::Left), (b, Side::Right)]);
            }
        }
        if events.peek().is_none_or(|&b| (b - t).abs() > eps) {
            check_event_capacity(mandatory, interiors + 1)?;
            interiors += 1;
            instants.push((t, Side::Interior));
        }
    }
    for b in events {
        instants.extend([(b, Side::Left), (b, Side::Right)]);
    }
    instants.push((grid.stop, Side::Interior));
    Ok(instants)
}

/// Compare Rust `got` with C `want` on `grid`, treating `breakpoints` (sorted,
/// strictly inside `(0, stop)`; see [`breakpoints`]) as events. Variables match
/// by case-insensitive name; every `want` signal other than `time` must have a
/// `voltage` or `current` unit and satisfy
/// `|got - want| <= relative*|want| + absolute(kind)` at every instant and at
/// both limits of every breakpoint. Both plots must end at `grid.stop`.
pub(super) fn transient(
    got: &Plot,
    want: &Plot,
    tolerance: TranTolerance,
    grid: &Grid,
    breakpoints: &[f64],
) -> Result<Summary, String> {
    let eps = TIME_EPS_REL * grid.stop;
    let instants = comparison_instants(grid, breakpoints, eps)?;
    let rust = Series::new("Rust", got, breakpoints, grid, eps)?;
    let c = Series::new("C", want, breakpoints, grid, eps)?;
    compare::check_structure(got, want, &rust.columns, &c.columns)?;
    let mut signals = Vec::new();
    for (name, &column) in &c.columns {
        if name == "time" {
            continue;
        }
        let absolute = match want.variables()[column].unit.as_str() {
            "voltage" => tolerance.voltage_absolute,
            "current" => tolerance.current_absolute,
            other => return Err(format!("signal '{name}' has unsupported unit '{other}'")),
        };
        let peak = (0..want.point_count())
            .map(|row| want.points()[row][column].re.abs())
            .fold(0.0, f64::max);
        let peak_allowance = tolerance.peak_relative * peak;
        let absolute = absolute + peak_allowance;
        signals.push((name, rust.columns[name], column, absolute));
    }
    let mut summary = Summary {
        instants: 0,
        limits: 0,
        worst_ratio: 0.0,
    };
    let mut first = None;
    let mut worst = None;
    let mut mismatches = 0usize;
    for &(t, side) in &instants {
        if side == Side::Interior {
            summary.instants += 1;
        } else {
            summary.limits += 1;
        }
        // With no compared signals the old comparator never bound samples.
        if signals.is_empty() {
            continue;
        }
        let got_sample = rust.sample(t, side)?;
        let want_sample = c.sample(t, side)?;
        for &(name, got_column, want_column, absolute) in &signals {
            let actual = got_sample.value(got_column);
            let expected = want_sample.value(want_column);
            let relative_allowance = tolerance.relative * expected.abs();
            let limit = relative_allowance + absolute;
            let error = (actual - expected).abs();
            let ratio = error / limit;
            summary.worst_ratio = summary.worst_ratio.max(ratio);
            if error > limit {
                mismatches += 1;
                let report = format!(
                    "t={t:.9e} ({side:?}), {name}: Rust {actual:.17e}, C {expected:.17e}, |error| {error:.6e} > bound {limit:.6e}"
                );
                if first.is_none() {
                    first = Some(report.clone());
                }
                if worst.as_ref().is_none_or(|(r, _)| ratio > *r) {
                    worst = Some((ratio, report));
                }
            }
        }
    }
    match (first, worst) {
        (Some(first), Some((ratio, worst))) => Err(format!(
            "{mismatches} transient mismatch(es)\nfirst: {first}\nworst ({ratio:.6e}x bound): {worst}"
        )),
        _ => Ok(summary),
    }
}

/// Source breakpoints required to have samples in both transient plots.
/// Uses the same validated, elaborated forcing as the driver, including
/// nested parameterized sources and the effective last waveform setter.
/// Analytic function corners and trailing scheduling requests are not
/// required; an actual pulse-count cut is a discontinuity and must not be
/// interpolated across, even if C did not schedule a landing there.
pub(super) fn breakpoints(circuit: &Circuit, timing: &TransientTiming) -> Result<Vec<f64>, String> {
    let stop_time = timing.final_time();
    let eps = TIME_EPS_REL * stop_time;
    let mut times = Vec::new();
    let mut generated = 0;
    for source in circuit.source_timelines(*timing) {
        let source = source.map_err(|error| error.to_string())?;
        let mut events = source
            .waveform
            .breakpoints_in(0., stop_time)
            .map_err(|error| format!("{}: {error}", source.name))?;
        while let Some(event) = events.next_event() {
            // Count all generated events before filtering or deduplication.
            check_event_capacity(generated, 1)?;
            generated += 1;
            if event.kind.requires_landing() && event.time > eps && event.time < stop_time - eps {
                times.push(event.time);
            }
        }
    }
    times.sort_by(f64::total_cmp);
    times.dedup_by(|b, a| (*b - *a).abs() <= eps);
    Ok(times)
}

/// `breakpoints` plus `b + offset` for every breakpoint `b` and offset (the
/// echoes of a transmission-line deck, see `verify::Gate::Echoes`), inside
/// `(0, stop_time)`, sorted and deduplicated like [`breakpoints`].
pub(super) fn echoes(
    breakpoints: &[f64],
    offsets: &[f64],
    stop_time: f64,
) -> Result<Vec<f64>, String> {
    if offsets.iter().any(|o| !(o.is_finite() && *o > 0.0)) {
        return Err("echo offsets must be finite and positive".into());
    }
    let eps = TIME_EPS_REL * stop_time;
    let additional = breakpoints
        .len()
        .checked_mul(offsets.len())
        .ok_or_else(|| "too many echo breakpoints".to_owned())?;
    check_event_capacity(breakpoints.len(), additional)?;
    let mut times = breakpoints.to_vec();
    for b in breakpoints {
        times.extend(offsets.iter().map(|o| b + o));
    }
    times.retain(|&t| t > eps && t < stop_time - eps);
    times.sort_by(f64::total_cmp);
    times.dedup_by(|b, a| (*b - *a).abs() <= eps);
    Ok(times)
}

/// Bound the total event count before extending any waveform's output.
fn check_event_capacity(existing: usize, additional: usize) -> Result<(), String> {
    let count = existing
        .checked_add(additional)
        .and_then(|n| u32::try_from(n).ok());
    if count.is_some_and(|n| n <= MAX_POINTS) {
        Ok(())
    } else {
        Err("too many generated breakpoints".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ngspice_rs::analysis::Variable;
    use ngspice_rs::netlist::Parser;
    use ngspice_rs::netlist::source::parse_deck_text;
    use ngspice_rs::primitives::Complex;
    use std::path::Path;

    const STOP: f64 = 10.0;

    fn plot(rows: &[(f64, f64)]) -> Plot {
        Plot::new(
            "tran1",
            "Transient Analysis",
            PlotFlags::Real,
            vec![
                Variable::new("time", "time"),
                Variable::new("v(out)", "voltage"),
            ],
        )
        .with_points(
            rows.iter()
                .map(|&(t, v)| vec![Complex::real(t), Complex::real(v)])
                .collect(),
        )
        .unwrap()
    }
    fn edit(plot: &Plot, change: impl FnOnce(&mut PlotFlags, &mut [Variable])) -> Plot {
        let mut flags = plot.flags();
        let mut variables = plot.variables().to_vec();
        change(&mut flags, &mut variables);
        Plot::new(plot.name.clone(), plot.plotname.clone(), flags, variables)
            .with_points(plot.points().to_vec())
            .unwrap()
    }
    /// Samples of `f` at the given times.
    fn sampled(times: &[f64], f: impl Fn(f64) -> f64) -> Plot {
        plot(&times.iter().map(|&t| (t, f(t))).collect::<Vec<_>>())
    }
    fn times(step: f64) -> Vec<f64> {
        (0..=(STOP / step).round() as usize)
            .map(|i| i as f64 * step)
            .collect()
    }
    fn grid() -> Grid {
        Grid {
            start: 0.0,
            stop: STOP,
            step: 1.0,
        }
    }
    fn run(got: &Plot, want: &Plot, breakpoints: &[f64]) -> Result<Summary, String> {
        transient(got, want, compare::TRAN, &grid(), breakpoints)
    }
    /// 0 for t<5, 3 + t after: a true jump of 3 V plus a ramp, at t = 5.
    fn jump(t: f64, right: bool) -> f64 {
        if t < 5.0 || (t == 5.0 && !right) {
            0.0
        } else {
            3.0 + t
        }
    }
    fn jump_plot(step: f64) -> Plot {
        let mut rows = Vec::new();
        for t in times(step) {
            if t == 5.0 {
                rows.push((t, jump(t, false)));
                rows.push((t, jump(t, true)));
            } else {
                rows.push((t, jump(t, true)));
            }
        }
        plot(&rows)
    }

    // Pre-refactor oracle. Keep the original row scan, arithmetic and
    // per-call bracket lookup independent of the indexed implementation.
    fn legacy_value(series: &Series<'_>, column: usize, t: f64, side: Side) -> Result<f64, String> {
        let at = |index: usize| series.plot.points()[index][column].re;
        if let Some(b) = series.breakpoint_near(t) {
            let near: Vec<usize> = (0..series.times.len())
                .filter(|&i| (series.times[i] - b).abs() <= series.eps)
                .collect();
            let (Some(&first), Some(&last)) = (near.first(), near.last()) else {
                return Err(format!(
                    "{}: no sample at breakpoint {b:e}; cannot form a one-sided limit",
                    series.label
                ));
            };
            return match side {
                Side::Left => Ok(at(first)),
                Side::Right => Ok(at(last)),
                Side::Interior => Err(format!(
                    "{}: interior evaluation requested at breakpoint {b:e}",
                    series.label
                )),
            };
        }
        let upper = series.times.partition_point(|&x| x < t - series.eps);
        if upper == series.times.len() {
            return Err(format!(
                "{}: time {t:e} is beyond the samples",
                series.label
            ));
        }
        if (series.times[upper] - t).abs() <= series.eps {
            return Ok(at(upper));
        }
        if upper == 0 {
            return Err(format!("{}: time {t:e} precedes the samples", series.label));
        }
        let (t0, t1) = (series.times[upper - 1], series.times[upper]);
        let next = series
            .breakpoints
            .partition_point(|&b| b <= t0 + series.eps);
        if let Some(b) = series
            .breakpoints
            .get(next)
            .filter(|&&b| b < t1 - series.eps)
        {
            return Err(format!(
                "{}: samples [{t0:e}, {t1:e}] straddle breakpoint {b:e}; refusing to interpolate across it",
                series.label
            ));
        }
        let fraction = (t - t0) / (t1 - t0);
        let left = (1.0 - fraction) * at(upper - 1);
        let right = fraction * at(upper);
        Ok(left + right)
    }

    fn legacy_instants(
        grid: &Grid,
        breakpoints: &[f64],
        eps: f64,
    ) -> Result<Vec<(f64, Side)>, String> {
        if !(grid.stop.is_finite() && grid.stop > 0.0 && grid.step.is_finite() && grid.step > 0.0) {
            return Err("grid stop and step must be finite and positive".into());
        }
        if !(grid.start.is_finite() && grid.start >= 0.0 && grid.start < grid.stop - eps) {
            return Err("grid start must be finite and lie in [0, stop)".into());
        }
        let count = (grid.stop / grid.step + 1e-9).floor();
        if count >= f64::from(MAX_POINTS) {
            return Err(format!("grid needs more than {MAX_POINTS} points"));
        }
        let count = exact_nonnegative_integer::<u32>(count)
            .ok_or_else(|| "invalid grid point count".to_owned())?;
        let mut previous = grid.start;
        for &b in breakpoints {
            if !(b.is_finite() && b > eps && b < grid.stop - eps && b > previous + eps) {
                return Err(format!(
                    "breakpoint {b:e} must be finite, strictly inside (start, stop) and strictly increasing"
                ));
            }
            previous = b;
        }
        let mut instants: Vec<(f64, Side)> = Vec::new();
        // The first instant is `start` itself (0 for an ordinary run); grid times
        // before it do not exist in either plot.
        instants.push((grid.start, Side::Interior));
        for index in 0..=count {
            let t = f64::from(index) * grid.step;
            if t > grid.start + eps
                && t < grid.stop - eps
                && nearby_breakpoint(breakpoints, t, eps).is_none()
            {
                check_event_capacity(instants.len(), 1)?;
                instants.push((t, Side::Interior));
            }
        }
        check_event_capacity(instants.len(), 1)?;
        instants.push((grid.stop, Side::Interior));
        let limit_count = breakpoints
            .len()
            .checked_mul(2)
            .ok_or("too many breakpoint limits")?;
        check_event_capacity(instants.len(), limit_count)?;
        for &b in breakpoints {
            instants.push((b, Side::Left));
            instants.push((b, Side::Right));
        }
        // Stable sort keeps Left before Right at equal times.
        instants.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ok(instants)
    }

    #[test]
    fn indexed_event_rows_match_full_scans_at_rounding_boundaries() {
        let eps = 0.125;
        let events: [f64; 5] = [0.5, 1.0, 1.1875, 2.0, 4.0];
        let mut times = vec![-0.0, 0.0, 3.0];
        for event in events {
            for boundary in [event - eps, event, event + eps] {
                for time in [boundary.next_down(), boundary, boundary.next_up()] {
                    times.push(time);
                }
            }
        }
        // This deliberately permutes storage order, to exercise the
        // first/last index invariant independently of time validation.
        times.reverse();
        let indexed = index_event_rows(&times, &events, eps);
        for (&event, actual) in events.iter().zip(indexed) {
            let matching: Vec<_> = times
                .iter()
                .enumerate()
                .filter(|(_, time)| (**time - event).abs() <= eps)
                .map(|(index, _)| index)
                .collect();
            assert_eq!(
                actual,
                matching.first().zip(matching.last()).map(|(&a, &b)| (a, b))
            );
        }
    }

    #[test]
    fn planned_samples_preserve_legacy_results_including_reversed_limits() {
        let eps = TIME_EPS_REL * STOP;
        let b = 5.0;
        let cases = [
            (
                vec![b],
                vec![
                    (-0.0, -0.0),
                    (3.0, 1.0),
                    (b, 2.0),
                    (b, 3.0),
                    (7.0, 4.0),
                    (STOP, 5.0),
                ],
            ),
            (
                vec![b],
                vec![
                    (-0.0, -0.0),
                    (3.0, 1.0),
                    (b + eps * 0.25, 2.0),
                    (b - eps * 0.25, 3.0),
                    (7.0, 4.0),
                    (STOP, 5.0),
                ],
            ),
            (
                vec![b, b + eps * 1.5],
                vec![
                    (-0.0, -0.0),
                    (3.0, 1.0),
                    (b + eps * 0.75, 2.0),
                    (7.0, 4.0),
                    (STOP, 5.0),
                ],
            ),
            (
                vec![b],
                vec![(-0.0, -0.0), (3.0, 1.0), (7.0, 4.0), (STOP, 5.0)],
            ),
        ];
        for (events, rows) in cases {
            let p = plot(&rows);
            let series = Series::new("test", &p, &events, &grid(), eps).unwrap();
            let mut queries = vec![-0.0, 0.0, 1.5, 4.0, 6.0, 8.0, STOP];
            for event in &events {
                for boundary in [event - eps, *event, event + eps] {
                    queries.extend([boundary.next_down(), boundary, boundary.next_up()]);
                }
            }
            for t in queries {
                for side in [Side::Interior, Side::Left, Side::Right] {
                    for column in 0..2 {
                        let expected = legacy_value(&series, column, t, side).map(f64::to_bits);
                        let actual = series
                            .sample(t, side)
                            .map(|plan| plan.value(column).to_bits());
                        assert_eq!(actual, expected, "{events:?}, {t:e}, {side:?}, {column}");
                    }
                }
            }
        }
    }

    #[test]
    fn merged_instants_match_legacy_sort_and_coincidence_rules() {
        for start in [-0.0, 0.0, 0.4] {
            for step in [0.125, 0.3, 0.5, 1.0, 20.0] {
                let grid = Grid {
                    start,
                    stop: STOP,
                    step,
                };
                let eps = TIME_EPS_REL * grid.stop;
                let mut events = vec![1.0_f64, 3.5, 8.0];
                for delta in [-eps, eps] {
                    for edge in [
                        (5.0 + delta).next_down(),
                        5.0 + delta,
                        (5.0 + delta).next_up(),
                    ] {
                        events[1] = edge;
                        let expected = legacy_instants(&grid, &events, eps).unwrap();
                        let actual = comparison_instants(&grid, &events, eps).unwrap();
                        let bits = |v: Vec<(f64, Side)>| {
                            v.into_iter()
                                .map(|(t, side)| (t.to_bits(), side))
                                .collect::<Vec<_>>()
                        };
                        assert_eq!(bits(actual), bits(expected));
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "timing benchmark; run explicitly"]
    fn indexed_dense_transient_comparison_benchmark() {
        use std::{hint::black_box, time::Instant};
        let grid = Grid {
            start: 0.0,
            stop: STOP,
            step: STOP / 20_000.0,
        };
        let eps = TIME_EPS_REL * grid.stop;
        let events: Vec<_> = (1..500).map(|i| f64::from(i * 40) * grid.step).collect();
        let mut variables = vec![Variable::new("time", "time")];
        variables.extend((0..32).map(|i| Variable::new(format!("v({i})"), "voltage")));
        let points = (0..=20_000)
            .map(|i| {
                let t = f64::from(i) * grid.step;
                std::iter::once(Complex::real(t))
                    .chain((0..32).map(|j| Complex::real(t * f64::from(j + 1))))
                    .collect()
            })
            .collect();
        let p = Plot::new("tran1", "Transient Analysis", PlotFlags::Real, variables)
            .with_points(points)
            .unwrap();
        let baseline = Series::new("test", &p, &events, &grid, eps).unwrap();
        let start = Instant::now();
        let queries = legacy_instants(&grid, &events, eps).unwrap();
        let mut old_sum = 0.0;
        for &(t, side) in &queries {
            for col in 1..33 {
                old_sum += black_box(legacy_value(black_box(&baseline), col, t, side).unwrap());
            }
        }
        let old_time = start.elapsed();
        let start = Instant::now();
        let queries = comparison_instants(&grid, &events, eps).unwrap();
        let indexed = Series::new("test", &p, &events, &grid, eps).unwrap();
        let mut new_sum = 0.0;
        for &(t, side) in &queries {
            let sample = indexed.sample(t, side).unwrap();
            for col in 1..33 {
                new_sum += black_box(sample.value(col));
            }
        }
        let new_time = start.elapsed();
        assert_eq!(old_sum.to_bits(), new_sum.to_bits());
        eprintln!(
            "20,001 rows, 499 events, 32 signals; old {old_time:?}, indexed including validation {new_time:?}, {:.2}x",
            old_time.as_secs_f64() / new_time.as_secs_f64()
        );
    }

    #[test]
    fn breakpoint_search_preserves_first_match_and_rounding() {
        let events: [f64; 5] = [0.5, 1.0, 1.125, 2.0, 4.0];
        for eps in [0.0, 0.125, 1e-9] {
            for event in events {
                for boundary in [event - eps, event, event + eps] {
                    for time in [boundary.next_down(), boundary, boundary.next_up()] {
                        let expected = events.iter().copied().find(|b| (b - time).abs() <= eps);
                        assert_eq!(nearby_breakpoint(&events, time, eps), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn event_capacity_is_global_and_checked_before_growth() {
        let limit = usize::try_from(MAX_POINTS).unwrap();
        assert!(check_event_capacity(limit - 1, 1).is_ok());
        assert!(check_event_capacity(limit, 1).is_err());
        assert!(check_event_capacity(usize::MAX, 1).is_err());
        // The Cartesian product exceeds the event budget, without needing
        // to allocate the oversized product to discover that fact.
        let points = vec![1.0; 2_000];
        let offsets = vec![1.0; 2_000];
        assert!(echoes(&points, &offsets, 10.0).is_err());
    }

    #[test]
    fn echoes_extend_breakpoints_inside_the_run() {
        let echoed = echoes(&[1.0, 2.5], &[1.0, 1.5], 4.0).unwrap();
        assert_eq!(echoed, vec![1.0, 2.0, 2.5, 3.5]);
        assert!(echoes(&[1.0], &[0.0], 4.0).is_err());
        assert!(echoes(&[1.0], &[f64::NAN], 4.0).is_err());
    }

    #[test]
    fn different_timesteps_agree_on_the_common_grid() {
        // Linear data is reproduced exactly by linear interpolation, so two
        // unrelated, non-nested sample sequences must pass.
        let line = |t: f64| 2.0 * t + 1.0;
        let got = sampled(&times(0.25), line);
        let mut odd: Vec<f64> = (0..=40).map(|i| (i as f64 * 0.2437).min(STOP)).collect();
        odd.dedup();
        odd.push(STOP);
        odd.dedup();
        let want = sampled(&odd, line);
        let summary = run(&got, &want, &[]).unwrap();
        assert_eq!(summary.instants, 11);
        assert_eq!(summary.limits, 0);
        assert!(summary.worst_ratio < 1e-9);
    }

    #[test]
    fn interpolation_stays_within_a_smooth_segment_and_is_checked() {
        // 1 V at t=0 to 3 V at t=10: grid point 1.0 is not a sample of `want`.
        let got = plot(&[(0.0, 1.0), (1.0, 1.2), (10.0, 3.0)]);
        let want = plot(&[(0.0, 1.0), (10.0, 3.0)]);
        run(&got, &want, &[]).unwrap();
        let mut wrong = want.clone();
        wrong.point_mut(1).unwrap()[1] = Complex::real(3.1);
        let error = run(&got, &wrong, &[]).unwrap_err();
        assert!(
            error.contains("first: t=") && error.contains("worst"),
            "{error}"
        );
    }

    #[test]
    fn interpolation_across_a_breakpoint_is_refused() {
        // Samples 3 and 6 bracket grid instant 4 but the event at 5 lies
        // between them: refused, even though a straight line would "fit".
        let coarse = plot(&[(0.0, 0.0), (3.0, 0.0), (6.0, 9.0), (10.0, 13.0)]);
        let fine = jump_plot(0.5);
        for (got, want) in [(&coarse, &fine), (&fine, &coarse)] {
            let error = run(got, want, &[5.0]).unwrap_err();
            assert!(error.contains("refusing to interpolate"), "{error}");
        }
        // A plot with the bracket split by a sample at 5 but no jump pair is a
        // single-sample (continuous) limit, so it is usable for kinks.
        let kink = |t: f64| if t < 5.0 { t } else { 5.0 + 2.0 * (t - 5.0) };
        let rows = |step| sampled(&times(step), kink);
        assert_eq!(run(&rows(0.5), &rows(0.25), &[5.0]).unwrap().limits, 2);
        // Without a sample at the event no limit exists.
        let skipped = plot(&[(0.0, 0.0), (4.0, 4.0), (6.0, 7.0), (10.0, 15.0)]);
        let error = run(&skipped, &rows(1.0), &[5.0]);
        assert!(error.is_err());
    }

    #[test]
    fn interior_bracket_containing_a_breakpoint_is_rejected_directly() {
        // Series-level check of the refusal path: bracket [3, 6] contains 5.
        let p = plot(&[(0.0, 0.0), (3.0, 0.0), (6.0, 9.0), (10.0, 9.0)]);
        let bps = [5.0];
        let series = Series::new("Rust", &p, &bps, &grid(), 1e-8).unwrap();
        let error = series.sample(4.0, Side::Interior).unwrap_err();
        assert!(error.contains("refusing to interpolate"), "{error}");
        assert_eq!(series.sample(1.5, Side::Interior).unwrap().value(1), 0.0);
        assert_eq!(series.sample(8.0, Side::Interior).unwrap().value(1), 9.0);
    }

    #[test]
    fn left_and_right_limits_are_compared_separately() {
        let got = jump_plot(0.5);
        let want = jump_plot(0.25);
        let summary = run(&got, &want, &[5.0]).unwrap();
        assert_eq!(summary.limits, 2);
        // Interpolating across the jump would pass 4.5..5.5 values wrongly; the
        // plots above only agree because each side is handled on its own.
        let mut bad_right = jump_plot(0.5);
        let last_at_five = bad_right
            .points()
            .iter()
            .rposition(|row| row[0].re == 5.0)
            .unwrap();
        bad_right.point_mut(last_at_five).unwrap()[1] = Complex::real(8.5);
        let error = run(&bad_right, &want, &[5.0]).unwrap_err();
        assert!(error.contains("(Right)"), "{error}");
        let mut bad_left = jump_plot(0.5);
        let first_at_five = bad_left
            .points()
            .iter()
            .position(|row| row[0].re == 5.0)
            .unwrap();
        bad_left.point_mut(first_at_five).unwrap()[1] = Complex::real(0.5);
        let error = run(&bad_left, &want, &[5.0]).unwrap_err();
        assert!(error.contains("(Left)"), "{error}");
    }

    #[test]
    fn a_jump_cannot_be_hidden_by_interpolating_across_it() {
        // Same continuous-looking trace without the repeated instant: a single
        // sample at 5 is both limits, so the C jump (0 vs 8) must be caught.
        let want = jump_plot(0.5);
        let mut smeared = jump_plot(0.5);
        let first = smeared
            .points()
            .iter()
            .position(|r| r[0].re == 5.0)
            .unwrap();
        smeared.remove_point(first).unwrap();
        let error = run(&smeared, &want, &[5.0]).unwrap_err();
        assert!(error.contains("(Left)"), "{error}");
        // And with no declared event the repeated instant is invalid input.
        let error = run(&want, &want, &[]).unwrap_err();
        assert!(error.contains("not a declared breakpoint"), "{error}");
    }

    #[test]
    fn a_continuous_single_sample_serves_both_limits() {
        let line = |t: f64| t;
        let got = sampled(&times(0.5), line);
        let want = sampled(&times(0.25), line);
        assert_eq!(run(&got, &want, &[5.0]).unwrap().limits, 2);
    }

    #[test]
    fn end_time_and_start_time_must_match() {
        let full = sampled(&times(1.0), |t| t);
        let short = sampled(&times(1.0)[..10], |t| t);
        for (got, want) in [(&short, &full), (&full, &short)] {
            assert!(run(got, want, &[]).unwrap_err().contains("end time"));
        }
        let late = plot(&[(0.5, 0.0), (10.0, 0.0)]);
        assert!(run(&late, &full, &[]).unwrap_err().contains("not 0"));
    }

    #[test]
    fn missing_extra_and_unsupported_signals_fail() {
        let want = sampled(&times(1.0), |t| t);
        let missing_time = edit(&want, |_, v| v[0].name = "t".into());
        assert!(run(&missing_time, &want, &[]).unwrap_err().contains("time"));
        let missing = edit(&want, |_, v| v[1].name = "v(other)".into());
        let error = run(&missing, &want, &[]).unwrap_err();
        assert!(
            error.contains("missing") && error.contains("v(out)"),
            "{error}"
        );
        assert!(run(&want, &missing, &[]).unwrap_err().contains("extra"));
        let odd_unit = edit(&want, |_, v| v[1].unit = "charge".into());
        assert!(
            run(&odd_unit, &odd_unit, &[])
                .unwrap_err()
                .contains("unsupported unit")
        );
        let complex = edit(&want, |f, _| *f = PlotFlags::Complex);
        assert!(run(&complex, &complex, &[]).is_err());
    }

    #[test]
    fn nonfinite_and_malformed_data_fail_on_either_side() {
        let good = sampled(&times(1.0), |t| t);
        for value in [f64::NAN, f64::INFINITY] {
            for column in [0, 1] {
                let mut bad = good.clone();
                bad.point_mut(3).unwrap()[column] = Complex::real(value);
                assert!(run(&bad, &good, &[]).is_err());
                assert!(run(&good, &bad, &[]).is_err());
            }
        }
        let mut backwards = good.clone();
        backwards.point_mut(4).unwrap()[0] = Complex::real(1.0);
        assert!(
            run(&backwards, &good, &[])
                .unwrap_err()
                .contains("decreases")
        );
        let mut empty = good.clone();
        empty.clear_points();
        assert!(run(&empty, &good, &[]).is_err());
    }

    #[test]
    fn current_uses_its_own_absolute_floor_and_relative_scales_with_value() {
        let want = edit(&sampled(&times(1.0), |_| 0.0), |_, v| {
            v[1] = Variable::new("i(v1)", "current")
        });
        let mut got = want.clone();
        for row in got.rows_mut() {
            row[1] = Complex::real(0.5 * compare::TRAN.current_absolute);
        }
        run(&got, &want, &[]).unwrap();
        for row in got.rows_mut() {
            row[1] = Complex::real(2.0 * compare::TRAN.current_absolute);
        }
        assert!(run(&got, &want, &[]).is_err());
        // Voltage uses relative 1e-3 of a 1 V value: 0.5 mV passes, 2 mV fails.
        let mut volts = sampled(&times(1.0), |_| 1.0);
        let reference = volts.clone();
        for row in volts.rows_mut() {
            row[1] = Complex::real(1.0 + 0.5e-3);
        }
        run(&volts, &reference, &[]).unwrap();
        for row in volts.rows_mut() {
            row[1] = Complex::real(1.0 + 2e-3);
        }
        assert!(run(&volts, &reference, &[]).is_err());
    }

    #[test]
    fn invalid_grids_and_breakpoints_are_rejected() {
        let p = sampled(&times(1.0), |t| t);
        for g in [
            Grid {
                start: 0.0,
                stop: STOP,
                step: 0.0,
            },
            Grid {
                start: 0.0,
                stop: STOP,
                step: f64::NAN,
            },
            Grid {
                start: 0.0,
                stop: 0.0,
                step: 1.0,
            },
            Grid {
                start: 0.0,
                stop: STOP,
                step: 1e-12,
            },
        ] {
            assert!(transient(&p, &p, compare::TRAN, &g, &[]).is_err());
        }
        for bps in [
            vec![0.0],
            vec![STOP],
            vec![f64::NAN],
            vec![6.0, 5.0],
            vec![5.0, 5.0],
        ] {
            assert!(run(&p, &p, &bps).is_err(), "{bps:?}");
        }
    }

    use ngspice_rs::netlist::ast::Netlist;

    fn deck_breakpoints(netlist: &Netlist, step: f64, stop: f64) -> Result<Vec<f64>, String> {
        let config =
            ngspice_rs::analysis::RunConfig::from_netlist(netlist).map_err(|e| e.to_string())?;
        let circuit = config.circuit(netlist).map_err(|e| e.to_string())?;
        super::breakpoints(
            &circuit,
            &TransientTiming::new(step, stop).map_err(|e| e.to_string())?,
        )
    }

    fn deck(body: &str) -> Netlist {
        Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("t.cir"),
                &format!("title\n{body}\nr1 in 0 1k\n.end\n"),
            ))
            .unwrap()
    }

    #[test]
    fn pwl_corners_become_breakpoints() {
        let netlist = deck("v1 in 0 pwl(0 0 1m 0 1.01m 1 6m 2 9m 0)");
        let found = deck_breakpoints(&netlist, 1e-5, 8e-3).unwrap();
        assert_eq!(found, vec![1e-3, 1.01e-3, 6e-3]);
        assert!(deck_breakpoints(&netlist, 0.0, 8e-3).is_err());
    }

    #[test]
    fn nested_and_parameterized_waveforms_use_expanded_sources() {
        // A source-free macromodel (the M6 gate's op-amp) adds no breakpoints.
        let netlist = deck(
            "v1 in 0 pulse(0 1 1m 1u 1u 1m)\n.subckt amp a b\ne1 b 0 a 0 2\n.ends\nx1 in out amp",
        );
        assert_eq!(
            deck_breakpoints(&netlist, 1e-5, 4e-3).unwrap(),
            vec![1e-3, 1.001e-3, 2.001e-3, 2.002e-3]
        );
        // Nested instances resolve their own formal parameters and names.
        let netlist = deck(
            ".subckt inner b params: td=1m\nv1 b 0 pwl(0 0 1u 1 1.001m 1 1.003m 0) td={td}\n.ends\n.subckt outer a params: d=1m\nx1 a inner td={d}\n.ends\nx1 n outer d=0.5m\nx2 m outer d=1m",
        );
        let found = deck_breakpoints(&netlist, 1e-5, 4e-3).unwrap();
        assert_eq!(found.len(), 8);
        for (actual, expected) in found.iter().zip([
            0.5e-3, 0.501e-3, 1e-3, 1.001e-3, 1.501e-3, 1.503e-3, 2.001e-3, 2.003e-3,
        ]) {
            assert!((actual - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn pulse_edges_follow_c_defaults_and_period() {
        // td=1, tr=1, tf=2, pw=3, per=10 over 0..25.
        let netlist = deck("v1 in 0 pulse(0 5 1 1 2 3 10)");
        let found = deck_breakpoints(&netlist, 0.1, 25.0).unwrap();
        assert_eq!(
            found,
            vec![1.0, 2.0, 5.0, 7.0, 11.0, 12.0, 15.0, 17.0, 21.0, 22.0]
        );
        // Omitted tr/tf -> tstep; omitted pw with five values -> 0; per -> tstop.
        let netlist = deck("i1 in 0 pulse(0 1 0 0 0)");
        assert_eq!(
            deck_breakpoints(&netlist, 0.5, 4.0).unwrap(),
            vec![0.5, 1.0]
        );
        let netlist = deck("i1 in 0 pulse(0 1)");
        assert_eq!(deck_breakpoints(&netlist, 0.5, 4.0).unwrap(), vec![0.5]);
    }

    #[test]
    fn m6_source_breakpoints_follow_c_vsrcaccept() {
        // PULSE count: corners at or before td + np*per only.
        let netlist = deck("v1 in 0 pulse(0 5 1 1 2 3 10 2)");
        assert_eq!(
            deck_breakpoints(&netlist, 0.1, 40.0).unwrap(),
            vec![1.0, 2.0, 5.0, 7.0, 11.0, 12.0, 15.0, 17.0, 21.0]
        );
        // PWL td=/r=: delayed knots, then the knots after r recur.
        let netlist = deck("v1 in 0 pwl(0 0 1 1 2 0) r=1 td=0.5");
        assert_eq!(
            deck_breakpoints(&netlist, 0.1, 5.0).unwrap(),
            vec![0.5, 1.5, 2.5, 3.5, 4.5]
        );
        let netlist = deck("v1 in 0 pwl(0 0 1 1 2 0) r=-1 td=0.5");
        assert_eq!(
            deck_breakpoints(&netlist, 0.1, 5.0).unwrap(),
            vec![0.5, 1.5, 2.5]
        );
        assert!(deck_breakpoints(&deck("v1 in 0 pwl(0 0 1 1) r=0.5"), 0.1, 5.0).is_err());
        // C sets no breakpoints for SIN/EXP/SFFM/AM; only the last setter counts.
        for card in [
            "v1 in 0 sin(0 1 1 0.5)",
            "v1 in 0 exp(0 1 1 1 2 1)",
            "i1 in 0 sffm(0 1 1 1 1 0.5)",
            "v1 in 0 am(0 1 1 1 1 0.5)",
            "v1 in 0 pwl(0 0 1 1) sin(0 1)",
        ] {
            assert!(
                deck_breakpoints(&deck(card), 0.1, 5.0).unwrap().is_empty(),
                "{card}"
            );
        }
    }

    #[test]
    fn fractional_count_cut_is_required_but_trailing_request_is_not() {
        let netlist = deck("v1 in 0 pulse(0 5 2 1 2 3 10 1.25)");
        let found = deck_breakpoints(&netlist, 0.1, 30.0).unwrap();
        assert_eq!(found, vec![2.0, 3.0, 6.0, 8.0, 12.0, 13.0, 14.5]);
    }

    #[test]
    fn non_source_devices_and_out_of_range_corners_are_ignored() {
        let netlist = deck("v1 in 0 dc 1");
        assert!(deck_breakpoints(&netlist, 1.0, 5.0).unwrap().is_empty());
        let netlist = deck("v1 in 0 pwl(0 0 9 1)");
        assert!(deck_breakpoints(&netlist, 1.0, 5.0).unwrap().is_empty());
    }

    #[test]
    fn breakpoints_feed_the_comparator_end_to_end() {
        let netlist = deck("v1 in 0 pwl(0 0 5 3 10 13)");
        let bps = deck_breakpoints(&netlist, 0.1, STOP).unwrap();
        assert_eq!(bps, vec![5.0]);
        run(&jump_plot(0.5), &jump_plot(0.2), &bps).unwrap();
    }

    #[test]
    fn a_uic_style_start_after_zero_is_compared_from_that_instant_only() {
        // C with `uic` has no t = 0 row: both plots begin at the first accepted
        // step (here 0.4). The grid starts there and nothing earlier exists.
        let line = |t: f64| 2.0 * t + 1.0;
        let times_from = |step: f64| -> Vec<f64> {
            let mut times = vec![0.4];
            times.extend(
                (1..=(STOP / step).round() as usize)
                    .map(|i| i as f64 * step)
                    .filter(|&t| t > 0.4),
            );
            times
        };
        let grid = Grid {
            start: 0.4,
            ..grid()
        };
        let got = sampled(&times_from(0.25), line);
        let want = sampled(&times_from(0.5), line);
        let summary = transient(&got, &want, compare::TRAN, &grid, &[]).unwrap();
        // 0.4, then 1..=9 on the unit grid, and the stop time.
        assert_eq!(summary.instants, 11);
        // A plot starting elsewhere (including the ordinary t = 0) is refused,
        // as is a nonsensical start.
        let from_zero = sampled(&times(0.5), line);
        let error = transient(&from_zero, &want, compare::TRAN, &grid, &[]).unwrap_err();
        assert!(error.contains("starts at"), "{error}");
        for bad in [-1.0, f64::NAN, STOP] {
            let grid = Grid { start: bad, ..grid };
            assert!(transient(&got, &want, compare::TRAN, &grid, &[]).is_err());
        }
        // A breakpoint at or before the start is rejected.
        assert!(transient(&got, &want, compare::TRAN, &grid, &[0.4]).is_err());
    }

    #[test]
    fn peak_scaled_policy_widens_only_by_the_signal_peak() {
        // C data: a 1 V ramp (peak 1). Rust is offset by 3e-4 V at mid-scale and
        // by the same amount near zero.
        let want = sampled(&times(0.5), |t| t / STOP);
        let near_zero = sampled(&times(0.5), |t| t / STOP + 3e-4 * (-t).exp());
        // Pointwise bound at t = 0 is only 1e-6 V: TRAN rejects, the peak-scaled
        // bound (1e-3 * 1 V) accepts.
        assert!(run(&near_zero, &want, &[]).is_err());
        let scaled = compare::TranTolerance {
            peak_relative: 1e-3,
            ..compare::TRAN
        };
        assert!(transient(&near_zero, &want, scaled, &grid(), &[]).is_ok());
        // An error above reltol * peak is still rejected, and the bound follows
        // the peak of C's data rather than a fixed constant.
        let off = sampled(&times(0.5), |t| t / STOP + 2e-3);
        assert!(transient(&off, &want, scaled, &grid(), &[]).is_err());
        let small = sampled(&times(0.5), |t| 1e-3 * t / STOP);
        let small_off = sampled(&times(0.5), |t| 1e-3 * t / STOP + 3e-6);
        assert!(transient(&small_off, &small, scaled, &grid(), &[]).is_err());
        assert_eq!(compare::TRAN.peak_relative, 0.0);
    }
}
