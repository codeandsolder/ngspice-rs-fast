//! Internal Fourier plans: validated sampling, bounded projection batches and normalization.

use super::{FourierAnalysis, FourierSettings, FourierWindow, Harmonic};
use crate::analysis::ordered_samples::{
    AxisError, Location, OrderedAxis, SampleError, column_names,
};
use crate::analysis::results::Plot;
use crate::analysis::selection::{self, Column};
use crate::netlist::ast::{FourierCard, MAX_HARMONICS, VectorRequest};
use crate::primitives::{AnalysisKind, Real, SpiceError, SpiceResult};

/// The grid's harmonic-count floor: `4 * 50 = 200` subintervals per period,
/// matching C's default `fourgridsize`. Higher counts retain four subintervals
/// per requested harmonic, with the same maximum of 400 subintervals.
const GRID_FLOOR: u32 = 50;

/// How many ULPs of the quadrature's own sums a fundamental amplitude must
/// exceed before a total harmonic distortion can be formed from it.
///
/// The sums accumulate at most `4 * MAX_HARMONICS + 1 = 401` terms, so a trace
/// with no fundamental content leaves an amplitude of a few `f64::EPSILON`
/// times the trace's own magnitude there; a THD formed from that residue would
/// be pure noise (C prints whatever it computes). This is the arithmetic's
/// resolution, not a physical tolerance.
const NOISE_ULPS: Real = 16.0;

/// A positioned failure for a request this plot cannot answer.
fn unsupported(card: &FourierCard, feature: String) -> SpiceError {
    SpiceError::Unsupported {
        feature,
        location: Some(card.location.clone()),
    }
}

/// A value the transform would have to use but cannot.
fn nonfinite(context: &str, message: String) -> SpiceError {
    SpiceError::Numerical {
        context: format!(".four {context}"),
        message,
    }
}

/// The port's bounded harmonic count, enforced where the grid is built as well
/// as in the grammar, so a hand-built card cannot ask for unbounded work.
pub(super) fn check_budget(card: &FourierCard) -> SpiceResult<()> {
    if card.fundamental_expression.is_some() {
        return Err(unsupported(card, "unresolved .four fundamental expression; call resolve_cards with the deck parameter scope".into()));
    }
    if !(card.fundamental.is_finite() && card.fundamental > 0.0) {
        return Err(unsupported(
            card,
            format!(
                ".four: the fundamental frequency must be a finite value greater than zero (found \
                 {})",
                crate::primitives::format_spice_number(card.fundamental)
            ),
        ));
    }
    if card.harmonics == 0 || card.harmonics > MAX_HARMONICS {
        let harmonics = card.harmonics;
        let max_divisions = 4 * MAX_HARMONICS;
        return Err(unsupported(
            card,
            format!(
                ".four: HARMONICS={harmonics} is outside this port's bounded resampling budget of 1..={MAX_HARMONICS} \
                 harmonics ({max_divisions} grid subintervals per vector); C has no such bound \
                 (docs/port/FOURIER.md)"
            ),
        ));
    }
    Ok(())
}

/// The ordered physical time axis of a plot: the driver's scale vector.
///
/// The plot holds every accepted time point and lands exactly on every source
/// breakpoint, so consecutive samples never straddle a source discontinuity
/// (`crates/analysis/src/companion.rs`, "Output"). This module therefore
/// interpolates only between consecutive samples, and treats two samples that
/// share an axis value as a jump (see [`sample`]).
fn time_axis(plot: &Plot, card: &FourierCard) -> SpiceResult<OrderedAxis<'static>> {
    let values = OrderedAxis::from_plot(plot, "time").map_err(|error| match error {
            AxisError::MissingColumn => unsupported(card, format!(
                ".four: the {} plot carries no 'time' axis vector; it carries {}", plot.plotname, column_names(plot))),
            AxisError::MissingSample(index) => nonfinite("time", format!("point {index} has no time value")),
            AxisError::Nonfinite(index) => nonfinite("time", format!("point {index} is not finite")),
            AxisError::Descending(index) => unsupported(card, format!(
                ".four: the time axis is not non-decreasing (it descends at point {index}); a Fourier transform needs an ordered axis")),
        })?;
    if values.len() < 2 {
        return Err(unsupported(
            card,
            format!(
                ".four: a Fourier transform needs at least 2 data points (the plot has {}); C reports the same refusal",
                values.len()
            ),
        ));
    }
    Ok(values)
}

/// One transformed vector: the column a `.four` request resolves to, plus the
/// plot it reads, so a value the transform cannot use is reported as a Fourier
/// failure rather than as an output-selection one.
struct Trace<'a> {
    plot: &'a Plot,
    column: Column,
}

impl<'a> Trace<'a> {
    fn resolve(plot: &'a Plot, request: &VectorRequest) -> SpiceResult<Self> {
        Ok(Self {
            plot,
            column: selection::resolve_request(plot, request)?,
        })
    }

    /// The vector's canonical name, e.g. `v(out)`.
    fn name(&self) -> &str {
        self.column.name()
    }

    /// The vector's unit, e.g. `voltage`.
    fn unit(&self) -> &str {
        self.column.unit()
    }

    /// The vector's value at one plot point, as a real.
    fn value(&self, index: usize) -> SpiceResult<Real> {
        let Some(point) = self.plot.points().get(index) else {
            return Err(nonfinite(
                self.name(),
                format!("the plot has no point {index}"),
            ));
        };
        self.column
            .value(point)
            .map_err(|error| match error {
                SpiceError::Numerical { message, .. } => {
                    nonfinite(&format!("{}: {}", self.name(), self.column.name()), message)
                }
                other => other,
            })
            .map(|value| value.re)
    }
}

/// Closed-grid locations are independent of the signal being sampled.
struct ClosedPlan {
    window: FourierWindow,
    divisions: u32,
    samples: usize,
    locations: Vec<(Real, Location)>,
}

impl ClosedPlan {
    fn new(axis: &OrderedAxis<'_>, card: &FourierCard) -> SpiceResult<Self> {
        let first = axis.as_slice()[0];
        let to = axis.as_slice()[axis.len() - 1];
        let period = 1.0 / card.fundamental;
        let from = to - period;
        if from < first {
            return Err(unsupported(
                card,
                format!(
                    ".four: the run spans {} s (from {} to {}), shorter than one period ({} s) of the {} Hz fundamental; C reports the same refusal",
                    crate::primitives::format_spice_number(to - first),
                    crate::primitives::format_spice_number(first),
                    crate::primitives::format_spice_number(to),
                    crate::primitives::format_spice_number(period),
                    crate::primitives::format_spice_number(card.fundamental)
                ),
            ));
        }
        check_window(card, from, to)?;
        let harmonics = card.harmonics as usize;
        let samples = axis.len() - axis.lower_bound(from);
        let needed = 2 * harmonics + 1;
        if samples < needed {
            return Err(unsupported(
                card,
                format!(
                    ".four: the final period [{}, {}] covers only {samples} sample(s); {harmonics} harmonics need at least {needed} (two per cycle of the highest harmonic, plus the endpoint) to be well posed",
                    crate::primitives::format_spice_number(from),
                    crate::primitives::format_spice_number(to)
                ),
            ));
        }
        let divisions = 4 * card.harmonics.max(GRID_FLOOR);
        let step = period / Real::from(divisions);
        let mut locations = Vec::with_capacity(divisions as usize + 1);
        let mut previous = None;
        for i in 0..=divisions {
            // Preserve the original arithmetic and exact closing coordinate.
            let x = if i == divisions {
                to
            } else {
                let offset = Real::from(i) * step;
                from + offset
            };
            check_grid_point(card, previous, x)?;
            previous = Some(x);
            let location = axis.locate(x).ok_or_else(|| {
                unsupported(
                    card,
                    ".four: resampling coordinate is outside the time axis".into(),
                )
            })?;
            locations.push((x, location));
        }
        Ok(Self {
            window: FourierWindow { from, to },
            divisions,
            samples,
            locations,
        })
    }

    fn samples(&self, card: &FourierCard, trace: &Trace<'_>) -> SpiceResult<Vec<Real>> {
        self.locations
            .iter()
            .map(|(x, location)| sample(location, trace, *x, card))
            .collect()
    }

    fn finish(
        &self,
        card: &FourierCard,
        prepared: PreparedTrace<'_>,
    ) -> SpiceResult<FourierAnalysis> {
        let trace = &prepared.trace;
        let (dc, harmonics, thd) = closed_grid_spectrum(
            &prepared.values,
            self.divisions,
            card,
            trace,
            &prepared.sums,
        )?;
        Ok(FourierAnalysis {
            vector: trace.name().to_owned(),
            unit: trace.unit().to_owned(),
            fundamental: card.fundamental,
            dc,
            harmonics,
            thd,
            window: self.window,
            divisions: self.divisions as usize,
            samples: self.samples,
        })
    }
}

fn check_window(card: &FourierCard, from: Real, to: Real) -> SpiceResult<()> {
    if !from.is_finite() || !to.is_finite() || from >= to {
        return Err(unsupported(
            card,
            "Fourier window is not representable on the time axis".into(),
        ));
    }
    Ok(())
}

fn check_grid_point(card: &FourierCard, previous: Option<Real>, x: Real) -> SpiceResult<()> {
    if !x.is_finite() || previous.is_some_and(|p| p >= x) {
        return Err(unsupported(
            card,
            "Fourier grid coordinates are not strictly increasing representable times".into(),
        ));
    }
    Ok(())
}

/// Preserve the distinct phase expressions: rearranging multiplication or
/// replacing the separate sin/cos calls changes reference rounding.
#[derive(Clone, Copy)]
enum ProjectionGrid {
    Closed(u32),
    HalfOpen { samples: u32, periods: u32 },
}

impl ProjectionGrid {
    const fn samples(self) -> usize {
        match self {
            Self::Closed(n) => n as usize + 1,
            Self::HalfOpen { samples, .. } => samples as usize,
        }
    }

    fn coefficient(self, k: u32, i: u32) -> [Real; 2] {
        let angle = match self {
            Self::Closed(divisions) => {
                let phase = Real::from(i) / Real::from(divisions);
                std::f64::consts::TAU * Real::from(k) * phase
            }
            Self::HalfOpen { samples, periods } => {
                std::f64::consts::TAU * Real::from(k) * Real::from(periods) * Real::from(i)
                    / Real::from(samples)
            }
        };
        [angle.sin(), angle.cos()]
    }
}

/// Buffered samples and accumulators for a bounded group of traces. The
/// coefficients of one harmonic are reused across the entire group.
const TRACE_BATCH_BYTES: usize = 4 * 1024 * 1024;

struct PreparedTrace<'a> {
    trace: Trace<'a>,
    values: Vec<Real>,
    sums: Vec<[Real; 2]>,
}

impl ProjectionGrid {
    fn batch_size(self, harmonics: u32) -> usize {
        let per_trace = self.samples() * size_of::<Real>()
            + (harmonics as usize + 1) * size_of::<[Real; 2]>()
            + size_of::<PreparedTrace<'_>>();
        (TRACE_BATCH_BYTES / per_trace).max(1)
    }

    fn project(self, batch: &mut [PreparedTrace<'_>], harmonics: u32) -> SpiceResult<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let mut basis =
            crate::primitives::filled_vec(self.samples(), [0.; 2], "Fourier projection row")?;
        let first = u32::from(matches!(self, Self::HalfOpen { .. }));
        for k in first..=harmonics {
            for (i, pair) in (0..).zip(&mut basis) {
                *pair = self.coefficient(k, i);
            }
            for prepared in batch.iter_mut() {
                let mut sine = 0.;
                let mut cosine = 0.;
                for (i, (y, [sin, cos])) in prepared.values.iter().zip(&basis).enumerate() {
                    let weighted = match self {
                        Self::Closed(divisions) => {
                            let weight = if i == 0 || i == divisions as usize {
                                0.5
                            } else {
                                1.
                            };
                            weight * y
                        }
                        Self::HalfOpen { .. } => *y,
                    };
                    let sine_term = weighted * sin;
                    let cosine_term = weighted * cos;
                    sine += sine_term;
                    cosine += cosine_term;
                }
                prepared.sums[k as usize] = [sine, cosine];
            }
        }
        Ok(())
    }
}

/// Normalize the independently accumulated legacy closed-grid sums.
fn closed_grid_spectrum(
    values: &[Real],
    divisions: u32,
    card: &FourierCard,
    trace: &Trace<'_>,
    sums: &[[Real; 2]],
) -> SpiceResult<(Real, Vec<Harmonic>, Real)> {
    let harmonics = card.harmonics as usize;
    let divisor = Real::from(divisions);
    let scale = 2.0 / divisor;
    let dc = sums[0][1] / divisor;
    // The trace's own magnitude, from the same sums: the noise floor the
    // fundamental amplitude is compared against below.
    let magnitude = (0..=divisions).zip(values).fold(0.0, |sum, (i, y)| {
        let weight = if i == 0 || i == divisions { 0.5 } else { 1.0 };
        let term = weight * y.abs();
        sum + term
    }) / divisor;
    let mut tabulated = Vec::with_capacity(harmonics);
    for (k, [sine, cosine]) in (1..=card.harmonics).zip(sums.iter().skip(1)) {
        let amplitude = (scale * sine).hypot(scale * cosine);
        let phase = (scale * cosine).atan2(scale * sine);
        if !amplitude.is_finite() || !phase.is_finite() {
            return Err(nonfinite(
                trace.name(),
                format!("harmonic {k} is not finite"),
            ));
        }
        tabulated.push(Harmonic {
            order: k,
            frequency: Real::from(k) * card.fundamental,
            amplitude,
            phase,
        });
    }
    let fundamental_amplitude = tabulated.first().map_or(0.0, |harmonic| harmonic.amplitude);
    // Amplitudes were checked finite above. At or below the quadrature's
    // rounding noise, THD has no denominator. An overflowed magnitude also
    // refuses the transform instead of producing a spurious denominator.
    if fundamental_amplitude <= NOISE_ULPS * f64::EPSILON * magnitude {
        return Err(nonfinite(
            trace.name(),
            format!(
                "the fundamental amplitude ({}) is at or below the quadrature's own rounding \
                 noise, so the total harmonic distortion has no denominator",
                crate::primitives::format_spice_number(fundamental_amplitude)
            ),
        ));
    }
    let thd = tabulated
        .iter()
        .skip(1)
        .map(|harmonic| (harmonic.amplitude / fundamental_amplitude).powi(2))
        .sum::<Real>()
        .sqrt();
    if !dc.is_finite() || !thd.is_finite() {
        return Err(nonfinite(
            trace.name(),
            "the DC component or the total harmonic distortion is not finite".to_owned(),
        ));
    }
    Ok((dc, tabulated, thd))
}

/// The vector's value at one physical time, with `.measure`'s sample model.
///
/// Interpolation is linear between the samples bracketing `x`, never across a
/// source discontinuity (the drivers keep a sample on every breakpoint). A time
/// carried by two samples with different values is a jump: its value is
/// ambiguous, so it is an explicit failure rather than a silent choice between
/// the left and right limits. A window boundary that falls strictly inside a
/// bracket is interpolated from the bracketing samples, which may lie just
/// outside the window.
fn sample(
    location: &Location,
    trace: &Trace<'_>,
    x: Real,
    card: &FourierCard,
) -> SpiceResult<Real> {
    location.sample(x, |i| trace.value(i)).map_err(|error| match error {
        SampleError::Discontinuous => unsupported(card, format!(
            ".four {}: the vector is discontinuous at time = {}: the plot has multiple values at that time, so the resampling grid cannot read it there",
            trace.name(), crate::primitives::format_spice_number(x))),
        SampleError::OutOfRange => unsupported(card, format!(".four {}: resampling coordinate is outside the time axis", trace.name())),
        SampleError::Nonfinite => nonfinite(trace.name(), "nonfinite interpolated value".into()),
        SampleError::Operand(error) => error,
    })
}

/// Validated sampling and spectrum geometry, shared by all vectors of a card.
struct ConfiguredPlan {
    window: FourierWindow,
    harmonics: u32,
    grid: Option<Vec<Real>>,
    degree: usize,
}

impl ConfiguredPlan {
    fn new(
        axis: &OrderedAxis<'_>,
        card: &FourierCard,
        settings: FourierSettings,
        count: u32,
    ) -> SpiceResult<Self> {
        let to = *axis
            .as_slice()
            .last()
            .ok_or_else(|| unsupported(card, "empty Fourier axis".into()))?;
        let from = if settings.polydegree == 0 {
            axis.as_slice()[0]
        } else {
            to - Real::from(settings.nperiods) / card.fundamental
        };
        if from < axis.as_slice()[0] {
            return Err(unsupported(
                card,
                "Fourier window is longer than the run".into(),
            ));
        }
        if settings.polydegree != 0 {
            check_window(card, from, to)?;
        }
        let harmonics = configured_harmonics(card, settings);
        if harmonics == 0 {
            return Err(unsupported(
                card,
                "nfreqs must include a fundamental for THD".into(),
            ));
        }
        let n = if settings.polydegree == 0 {
            axis.len()
        } else {
            count as usize
        };
        if n > 100_000 {
            return Err(unsupported(card, "Fourier sample budget exceeded".into()));
        }
        // The kth harmonic occupies DFT bin k * nperiods, not bin k.
        // A real signal's highest requested bin must be strictly below Nyquist.
        if n <= (2 * harmonics * settings.nperiods) as usize {
            return Err(unsupported(
                card,
                "Fourier grid undersamples requested harmonics".into(),
            ));
        }
        let grid = if settings.polydegree == 0 {
            None
        } else {
            let step = (to - from) / Real::from(count);
            let mut grid = Vec::with_capacity(count as usize);
            for i in 0..count {
                let offset = Real::from(i) * step;
                let x = from + offset;
                check_grid_point(card, grid.last().copied(), x)?;
                if x >= to {
                    return Err(unsupported(
                        card,
                        "Fourier half-open grid reached its closing coordinate".into(),
                    ));
                }
                grid.push(x);
            }
            Some(grid)
        };
        Ok(Self {
            window: FourierWindow { from, to },
            harmonics,
            grid,
            degree: settings.polydegree as usize,
        })
    }

    fn samples(
        &self,
        axis: &OrderedAxis<'_>,
        trace: &Trace<'_>,
        stable: Option<&StablePlan>,
    ) -> SpiceResult<Vec<Real>> {
        let samples = (0..axis.len())
            .map(|i| trace.value(i))
            .collect::<SpiceResult<Vec<_>>>()?;
        let Some(grid) = &self.grid else {
            return Ok(samples);
        };
        stable.map_or_else(
            || crate::analysis::fourier_resample::resample(axis, &samples, grid, self.degree),
            |plan| plan.sample(&samples, trace.name()),
        )
    }

    fn transform(
        &self,
        card: &FourierCard,
        trace: &Trace<'_>,
        values: &[Real],
        samples: usize,
        sums: &[[Real; 2]],
    ) -> SpiceResult<FourierAnalysis> {
        let n = u32::try_from(values.len())
            .map_err(|_| unsupported(card, "Fourier sample count exceeds u32".into()))?;
        let divisor = Real::from(n);
        let dc = values.iter().sum::<Real>() / divisor;
        let mut tabulated = Vec::with_capacity(self.harmonics as usize);
        for (k, &[sine, cosine]) in (1..=self.harmonics).zip(sums.iter().skip(1)) {
            tabulated.push(Harmonic {
                order: k,
                frequency: Real::from(k) * card.fundamental,
                amplitude: 2. * sine.hypot(cosine) / divisor,
                phase: cosine.atan2(sine),
            });
        }
        let fundamental = tabulated[0].amplitude;
        let magnitude = values.iter().map(|v| v.abs()).sum::<Real>() / divisor;
        if fundamental <= NOISE_ULPS * f64::EPSILON * magnitude {
            return Err(nonfinite(
                trace.name(),
                "fundamental at rounding noise; undefined THD".into(),
            ));
        }
        let thd = tabulated
            .iter()
            .skip(1)
            .map(|h| (h.amplitude / fundamental).powi(2))
            .sum::<Real>()
            .sqrt();
        if !dc.is_finite()
            || !thd.is_finite()
            || tabulated.iter().any(|h| {
                !h.frequency.is_finite() || !h.amplitude.is_finite() || !h.phase.is_finite()
            })
        {
            return Err(nonfinite(trace.name(), "nonfinite Fourier result".into()));
        }
        Ok(FourierAnalysis {
            vector: trace.name().into(),
            unit: trace.unit().into(),
            fundamental: card.fundamental,
            dc,
            harmonics: tabulated,
            thd,
            window: self.window,
            divisions: n as usize,
            samples,
        })
    }
}

/// Stable local polynomial geometry. Coefficients depend only on times,
/// not the vector; keep the exact old multiplication and evaluation order.
struct StablePlan {
    starts: Vec<usize>,
    weights: Vec<Real>,
    order: usize,
}
impl StablePlan {
    fn new(
        axis: &OrderedAxis<'_>,
        grid: &[Real],
        degree: usize,
        card: &FourierCard,
    ) -> SpiceResult<Self> {
        if axis.len() <= degree || axis.has_duplicates() {
            return Err(unsupported(
                card,
                "stable polynomial interpolation needs distinct samples".into(),
            ));
        }
        let order = degree + 1;
        let mut starts = Vec::with_capacity(grid.len());
        let mut weights =
            crate::primitives::filled_vec(grid.len() * order, 1., "Fourier interpolation weights")?;
        let times = axis.as_slice();
        for (&x, row) in grid.iter().zip(weights.chunks_exact_mut(order)) {
            let start = axis
                .lower_bound(x)
                .saturating_sub(degree.div_ceil(2))
                .min(axis.len() - order);
            starts.push(start);
            for (k, weight) in row.iter_mut().enumerate() {
                for j in 0..order {
                    if j != k {
                        *weight *= (x - times[start + j]) / (times[start + k] - times[start + j]);
                    }
                }
            }
        }
        Ok(Self {
            starts,
            weights,
            order,
        })
    }

    fn sample(&self, samples: &[Real], name: &str) -> SpiceResult<Vec<Real>> {
        let mut values = Vec::with_capacity(self.starts.len());
        for (&start, row) in self
            .starts
            .iter()
            .zip(self.weights.chunks_exact(self.order))
        {
            let mut value = 0.;
            for (&weight, &sample) in row.iter().zip(&samples[start..start + self.order]) {
                let term = weight * sample;
                value += term;
            }
            if !value.is_finite() {
                return Err(nonfinite(
                    name,
                    "nonfinite stable polynomial interpolation".into(),
                ));
            }
            values.push(value);
        }
        Ok(values)
    }
}

const fn configured_harmonics(card: &FourierCard, settings: FourierSettings) -> u32 {
    if card.harmonics_location.is_some() {
        card.harmonics
    } else {
        settings.nfreqs.saturating_sub(1)
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Closed,
    Configured {
        settings: FourierSettings,
        count: u32,
    },
}

impl Mode {
    fn select(
        card: &FourierCard,
        kind: AnalysisKind,
        settings: Option<FourierSettings>,
    ) -> SpiceResult<Self> {
        let settings = settings.or_else(|| card.frontend_command.then(FourierSettings::default));
        if let Some(settings) = settings {
            if kind != AnalysisKind::Transient
                || !(1..=101).contains(&settings.nfreqs)
                || !(1..=100).contains(&settings.nperiods)
                || settings.polydegree > 16
                || !(1..=100_000).contains(&settings.gridsize)
            {
                return Err(unsupported(
                    card,
                    "invalid Fourier settings or analysis".into(),
                ));
            }
            let count = settings
                .gridsize
                .checked_mul(settings.nperiods)
                .filter(|n| *n <= 100_000)
                .ok_or_else(|| unsupported(card, "Fourier grid exceeds 100000 samples".into()))?;
            Ok(Self::Configured { settings, count })
        } else {
            if kind != AnalysisKind::Transient {
                return Err(unsupported(
                    card,
                    format!(
                        ".four: the card transforms a .tran result (C selects the tran plot); this plot is .{}, so the card can never be honoured against it",
                        kind.as_str()
                    ),
                ));
            }
            Ok(Self::Closed)
        }
    }

    fn harmonics(self, card: &FourierCard) -> u32 {
        match self {
            Self::Closed => card.harmonics,
            Self::Configured { settings, .. } => configured_harmonics(card, settings),
        }
    }

    fn geometry(self, card: &FourierCard, rows: usize) -> SpiceResult<ProjectionGrid> {
        match self {
            Self::Closed => Ok(ProjectionGrid::Closed(4 * card.harmonics.max(GRID_FLOOR))),
            Self::Configured { settings, count } => {
                let samples = if settings.polydegree == 0 {
                    u32::try_from(rows)
                        .ok()
                        .filter(|n| *n <= 100_000)
                        .ok_or_else(|| unsupported(card, "Fourier sample budget exceeded".into()))?
                } else {
                    count
                };
                Ok(ProjectionGrid::HalfOpen {
                    samples,
                    periods: settings.nperiods,
                })
            }
        }
    }
}

/// A conservative aggregate resource policy. All arithmetic saturates above
/// the limits, so an overflowing estimate is necessarily rejected.
const MAX_WORK: usize = 500_000_000;
const MAX_MEMORY: usize = 128 * 1024 * 1024;

struct WorkBudget {
    work: usize,
    retained: usize,
    scratch: usize,
    vectors: usize,
}

impl WorkBudget {
    const fn new(rows: usize) -> Self {
        Self {
            work: rows,
            retained: rows.saturating_mul(size_of::<Real>()),
            scratch: 0,
            vectors: 0,
        }
    }

    fn charge(
        &mut self,
        rows: usize,
        count: usize,
        harmonics: usize,
        vectors: usize,
        mode: Mode,
        compatible: bool,
    ) {
        let projection = count.saturating_mul(harmonics + 1);
        let (sampling, scratch) = match mode {
            // Distinct grid coordinates visit each coincident group at most
            // once, plus at most two bracketing reads per grid coordinate.
            Mode::Closed => (
                rows.saturating_add(2 * count),
                count * (size_of::<(Real, Location)>() + size_of::<Real>()),
            ),
            Mode::Configured { settings, .. } => {
                let order = settings.polydegree as usize + 1;
                // C can retry every polynomial degree for each moving fit.
                // A fit advances at least one grid sample, except for at most
                // order tail steps; the traversal itself costs rows * order.
                let fits = rows.min(count + order);
                let fit_cost = (1..=order).map(|n| n.pow(3) + n * n).sum::<usize>();
                let interpolation = if order == 1 {
                    0
                } else if compatible {
                    rows.saturating_mul(order)
                        .saturating_add(fits.saturating_mul(fit_cost))
                } else {
                    count.saturating_mul(order * order)
                };
                let scratch = rows
                    .saturating_mul(size_of::<Real>())
                    .saturating_add(count * size_of::<Real>() * 2)
                    .saturating_add(if compatible || order == 1 {
                        0
                    } else {
                        count * (size_of::<usize>() + order * size_of::<Real>())
                    });
                (rows.saturating_add(interpolation), scratch)
            }
        };
        self.work = self
            .work
            .saturating_add(vectors.saturating_mul(sampling.saturating_add(projection)));
        self.vectors = self.vectors.saturating_add(vectors);
        self.retained = self.retained.saturating_add(
            vectors
                .saturating_mul(size_of::<FourierAnalysis>() + harmonics * size_of::<Harmonic>()),
        );
        let per_trace = count * size_of::<Real>()
            + (harmonics + 1) * size_of::<[Real; 2]>()
            + size_of::<PreparedTrace<'_>>();
        let batched = vectors
            .saturating_mul(per_trace)
            .min(TRACE_BATCH_BYTES.max(per_trace));
        let basis = count * size_of::<[Real; 2]>();
        self.scratch = self
            .scratch
            .max(scratch.saturating_add(batched).saturating_add(basis));
    }

    fn check(&self, card: &FourierCard) -> SpiceResult<()> {
        if self.work > MAX_WORK || self.retained.saturating_add(self.scratch) > MAX_MEMORY {
            return Err(unsupported(
                card,
                format!(
                    "Fourier aggregate work or memory budget exceeded ({} work units, {} bytes; limits {MAX_WORK} and {MAX_MEMORY})",
                    self.work,
                    self.retained.saturating_add(self.scratch)
                ),
            ));
        }
        Ok(())
    }
}

fn preflight(
    rows: usize,
    kind: AnalysisKind,
    cards: &[FourierCard],
    settings: Option<FourierSettings>,
    compatible: bool,
) -> SpiceResult<WorkBudget> {
    let mut budget = WorkBudget::new(rows);
    for card in cards {
        let mode = Mode::select(card, kind, settings)?;
        check_budget(card)?;
        let harmonics = mode.harmonics(card);
        if harmonics == 0 {
            return Err(unsupported(
                card,
                "nfreqs must include a fundamental for THD".into(),
            ));
        }
        if !(Real::from(harmonics) * card.fundamental).is_finite() {
            return Err(unsupported(
                card,
                "Fourier harmonic frequency is not finite".into(),
            ));
        }
        let geometry = mode.geometry(card, rows)?;
        budget.charge(
            rows,
            geometry.samples(),
            harmonics as usize,
            card.vectors.len(),
            mode,
            compatible,
        );
        budget.check(card)?;
    }
    Ok(budget)
}

pub(super) fn resolve_plans(
    plot: &Plot,
    kind: AnalysisKind,
    cards: &[FourierCard],
    settings: Option<FourierSettings>,
    compatible: bool,
) -> SpiceResult<Vec<FourierAnalysis>> {
    let Some(first) = cards.first() else {
        return Ok(Vec::new());
    };
    let budget = preflight(plot.point_count(), kind, cards, settings, compatible)?;
    let axis = time_axis(plot, first)?;
    let mut results = Vec::new();
    results.try_reserve_exact(budget.vectors).map_err(|error| {
        nonfinite(
            "planning",
            format!("cannot allocate Fourier result entries: {error}"),
        )
    })?;
    for card in cards {
        let mode = Mode::select(card, kind, settings)?;
        let execution = CardPlan::new(&axis, card, mode, compatible)?;
        execution.run(
            plot,
            &axis,
            card,
            mode.geometry(card, axis.len())?,
            &mut results,
        )?;
    }
    Ok(results)
}

enum CardPlan {
    Closed(ClosedPlan),
    Configured {
        plan: ConfiguredPlan,
        stable: Option<StablePlan>,
    },
}

impl CardPlan {
    fn new(
        axis: &OrderedAxis<'_>,
        card: &FourierCard,
        mode: Mode,
        compatible: bool,
    ) -> SpiceResult<Self> {
        match mode {
            Mode::Closed => Ok(Self::Closed(ClosedPlan::new(axis, card)?)),
            Mode::Configured { settings, count } => {
                let plan = ConfiguredPlan::new(axis, card, settings, count)?;
                let stable = if compatible {
                    None
                } else {
                    plan.grid
                        .as_ref()
                        .map(|grid| StablePlan::new(axis, grid, plan.degree, card))
                        .transpose()?
                };
                Ok(Self::Configured { plan, stable })
            }
        }
    }

    fn prepare<'a>(
        &self,
        plot: &'a Plot,
        axis: &OrderedAxis<'_>,
        card: &FourierCard,
        request: &VectorRequest,
        harmonics: u32,
    ) -> SpiceResult<PreparedTrace<'a>> {
        let trace = Trace::resolve(plot, request)?;
        let values = match self {
            Self::Closed(plan) => plan.samples(card, &trace)?,
            Self::Configured { plan, stable } => plan.samples(axis, &trace, stable.as_ref())?,
        };
        Ok(PreparedTrace {
            trace,
            values,
            sums: vec![[0.; 2]; harmonics as usize + 1],
        })
    }

    fn finish(
        &self,
        axis: &OrderedAxis<'_>,
        card: &FourierCard,
        prepared: PreparedTrace<'_>,
    ) -> SpiceResult<FourierAnalysis> {
        match self {
            Self::Closed(plan) => plan.finish(card, prepared),
            Self::Configured { plan, .. } => plan.transform(
                card,
                &prepared.trace,
                &prepared.values,
                axis.len(),
                &prepared.sums,
            ),
        }
    }

    fn run(
        &self,
        plot: &Plot,
        axis: &OrderedAxis<'_>,
        card: &FourierCard,
        geometry: ProjectionGrid,
        results: &mut Vec<FourierAnalysis>,
    ) -> SpiceResult<()> {
        let harmonics = match self {
            Self::Closed(_) => card.harmonics,
            Self::Configured { plan, .. } => plan.harmonics,
        };
        let batch_size = geometry.batch_size(harmonics);
        let mut batch = Vec::with_capacity(batch_size.min(card.vectors.len()));
        for requests in card.vectors.chunks(batch_size) {
            let mut failure = None;
            for request in requests {
                match self.prepare(plot, axis, card, request, harmonics) {
                    Ok(prepared) => batch.push(prepared),
                    Err(error) => {
                        failure = Some(error);
                        break;
                    }
                }
            }
            geometry.project(&mut batch, harmonics)?;
            // Finish prior successful samples before returning a later sampling
            // failure: retain the historical vector-wise diagnostic ordering.
            for prepared in batch.drain(..) {
                results.push(self.finish(axis, card, prepared)?);
            }
            if let Some(error) = failure {
                return Err(error);
            }
        }
        Ok(())
    }
}
