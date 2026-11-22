//! Internal Fourier plans: validated sampling, bounded projection batches and normalization.

use super::{FourierAnalysis, FourierSettings, FourierWindow, Harmonic};
use crate::analysis::fourier_resample::PolynomialPlan;
use crate::analysis::ordered_samples::{
    AxisError, Location, OrderedAxis, SampleError, column_names,
};
use crate::analysis::reduction::CompensatedSum;
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
        let Some(point) = self.plot.row(index) else {
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
        prepared: &PreparedTrace<'_>,
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

/// Each grid uses its own integer bin index; reduce it modulo the grid
/// length before converting to radians to avoid large-angle phase error.
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
        let (length, periods) = match self {
            Self::Closed(divisions) => (divisions, 1),
            Self::HalfOpen { samples, periods } => (samples, periods),
        };
        // Bounds: <=100 harmonics, <=100 periods and <=100000 grid points.
        let residue = (k * periods * i) % length;
        let phase = Real::from(residue) / Real::from(length);
        let angle = std::f64::consts::TAU * phase;
        let (sin, cos) = angle.sin_cos();
        [sin, cos]
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
                let mut sine = CompensatedSum::default();
                let mut cosine = CompensatedSum::default();
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
                    sine.add(sine_term);
                    cosine.add(cosine_term);
                }
                prepared.sums[k as usize] = [sine.total(), cosine.total()];
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
        polynomial: Option<&PolynomialPlan>,
        input: &mut Vec<Real>,
    ) -> SpiceResult<Vec<Real>> {
        let Some(grid) = &self.grid else {
            return (0..axis.len()).map(|i| trace.value(i)).collect();
        };
        input.clear();
        input.try_reserve_exact(axis.len()).map_err(|error| {
            nonfinite(
                trace.name(),
                format!("cannot allocate Fourier input samples: {error}"),
            )
        })?;
        for i in 0..axis.len() {
            input.push(trace.value(i)?);
        }
        polynomial
            .ok_or_else(|| nonfinite(trace.name(), "missing polynomial sampling geometry".into()))?
            .sample(grid, input)
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
        let mut dc = CompensatedSum::default();
        for &value in values {
            dc.add(value / divisor);
        }
        let dc = dc.total();
        let mut tabulated = Vec::with_capacity(self.harmonics as usize);
        for (k, &[sine, cosine]) in (1..=self.harmonics).zip(sums.iter().skip(1)) {
            tabulated.push(Harmonic {
                order: k,
                frequency: Real::from(k) * card.fundamental,
                amplitude: ((2. / divisor) * sine).hypot((2. / divisor) * cosine),
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

    const fn harmonics(self, card: &FourierCard) -> u32 {
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
            work: rows.saturating_mul(3),
            retained: rows.saturating_mul(size_of::<Real>()),
            scratch: 0,
            vectors: 0,
        }
    }

    fn charge(&mut self, rows: usize, count: usize, harmonics: usize, vectors: usize, mode: Mode) {
        let projection = count.saturating_mul(harmonics + 1);
        let (sampling, setup, scratch) = match mode {
            // For distinct source coordinates, at most two reads per grid
            // point. A second preflight charges any excess coincident-group
            // reads after axis validation, before visiting any vector.
            Mode::Closed => (
                2 * count,
                count.saturating_mul(rows.checked_ilog2().unwrap_or(0) as usize + 1),
                count * size_of::<(Real, Location)>(),
            ),
            Mode::Configured { settings, .. } => {
                let order = settings.polydegree as usize + 1;
                let (setup, plan_bytes) = if order == 1 {
                    (0, 0)
                } else {
                    PolynomialPlan::budget(rows, count, order)
                };
                let sampling = rows.saturating_add(count.saturating_mul(order));
                let scratch = rows
                    .saturating_mul(size_of::<Real>())
                    .saturating_add(count * size_of::<Real>())
                    .saturating_add(plan_bytes);
                (sampling, setup, scratch)
            }
        };
        self.work = self
            .work
            .saturating_add(setup)
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

    fn charge_duplicate_reads(
        &mut self,
        axis: &OrderedAxis<'_>,
        card: &FourierCard,
    ) -> SpiceResult<()> {
        let plan = ClosedPlan::new(axis, card)?;
        let reads = plan
            .locations
            .iter()
            .map(|(_, location)| location.reads())
            .sum::<usize>();
        let excess = reads.saturating_sub(2 * plan.locations.len());
        self.work = self
            .work
            .saturating_add(excess.saturating_mul(card.vectors.len()));
        self.check(card)
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
) -> SpiceResult<Vec<FourierAnalysis>> {
    let Some(first) = cards.first() else {
        return Ok(Vec::new());
    };
    let mut budget = preflight(plot.point_count(), kind, cards, settings)?;
    let axis = time_axis(plot, first)?;
    if axis.has_duplicates() {
        for card in cards {
            if !card.vectors.is_empty()
                && matches!(Mode::select(card, kind, settings)?, Mode::Closed)
            {
                budget.charge_duplicate_reads(&axis, card)?;
            }
        }
    }
    let mut results = Vec::new();
    results.try_reserve_exact(budget.vectors).map_err(|error| {
        nonfinite(
            "planning",
            format!("cannot allocate Fourier result entries: {error}"),
        )
    })?;
    for card in cards {
        let mode = Mode::select(card, kind, settings)?;
        if card.vectors.is_empty() {
            continue;
        }
        let execution = CardPlan::new(&axis, card, mode)?;
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
        polynomial: Option<PolynomialPlan>,
    },
}

impl CardPlan {
    fn new(axis: &OrderedAxis<'_>, card: &FourierCard, mode: Mode) -> SpiceResult<Self> {
        match mode {
            Mode::Closed => Ok(Self::Closed(ClosedPlan::new(axis, card)?)),
            Mode::Configured { settings, count } => {
                let plan = ConfiguredPlan::new(axis, card, settings, count)?;
                let polynomial = plan
                    .grid
                    .as_ref()
                    .map(|grid| PolynomialPlan::new(axis, grid, plan.degree))
                    .transpose()?;
                Ok(Self::Configured { plan, polynomial })
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
        input: &mut Vec<Real>,
    ) -> SpiceResult<PreparedTrace<'a>> {
        let trace = Trace::resolve(plot, request)?;
        let values = match self {
            Self::Closed(plan) => plan.samples(card, &trace)?,
            Self::Configured { plan, polynomial } => {
                plan.samples(axis, &trace, polynomial.as_ref(), input)?
            }
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
        prepared: &PreparedTrace<'_>,
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
        let mut input = Vec::new();
        for requests in card.vectors.chunks(batch_size) {
            let mut failure = None;
            for request in requests {
                match self.prepare(plot, axis, card, request, harmonics, &mut input) {
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
            for prepared in &batch {
                results.push(self.finish(axis, card, prepared)?);
            }
            batch.clear();
            if let Some(error) = failure {
                return Err(error);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::FourierSettings;
    use super::{Mode, ProjectionGrid, WorkBudget, preflight};
    use crate::netlist::ast::{FourierCard, RequestedVector, VectorRequest};
    use crate::primitives::{AnalysisKind, Real, SourceLoc};
    use std::path::PathBuf;

    fn card(vectors: usize, harmonics: u32) -> FourierCard {
        let loc = SourceLoc::new(PathBuf::from("budget.cir"), 2, 1);
        let request = VectorRequest {
            vector: RequestedVector::Voltage {
                positive: "out".into(),
                negative: None,
            },
            location: loc.clone(),
        };
        FourierCard {
            frontend_command: false,
            fundamental_expression: None,
            fundamental: 1e3,
            fundamental_location: loc.clone(),
            harmonics,
            harmonics_location: Some(loc.clone()),
            vectors: vec![request; vectors],
            location: loc,
        }
    }

    #[test]
    fn aggregate_budget_preflights_work_and_memory_without_building_a_plot() {
        let settings = FourierSettings {
            gridsize: 100_000,
            polydegree: 1,
            ..Default::default()
        };
        let one = card(1, 100);
        assert!(
            preflight(
                1_000,
                AnalysisKind::Transient,
                std::slice::from_ref(&one),
                Some(settings)
            )
            .is_ok()
        );
        assert!(
            preflight(
                1_000,
                AnalysisKind::Transient,
                &vec![one.clone(); 50],
                Some(settings)
            )
            .is_err_and(|error| error
                .to_string()
                .contains("aggregate work or memory budget"))
        );
        assert!(
            preflight(usize::MAX, AnalysisKind::Transient, &[one], Some(settings)).is_err_and(
                |error| error
                    .to_string()
                    .contains("aggregate work or memory budget")
            )
        );
        // Saturation is a fail-closed arithmetic policy, rather than a wrapped count.
        let mut budget = WorkBudget::new(1);
        budget.charge(1, 401, 100, usize::MAX, Mode::Closed);
        assert!(budget.check(&card(1, 1)).is_err());
    }

    #[test]
    fn reduced_coefficient_rows_agree_with_unreduced_phase_within_roundoff() {
        for geometry in [
            ProjectionGrid::Closed(204),
            ProjectionGrid::HalfOpen {
                samples: 1_117,
                periods: 3,
            },
        ] {
            for k in [0, 1, 3, 100] {
                for i in [0, 1, 99, 203] {
                    let expected_angle = match geometry {
                        ProjectionGrid::Closed(n) => {
                            let phase = Real::from(i) / Real::from(n);
                            std::f64::consts::TAU * Real::from(k) * phase
                        }
                        ProjectionGrid::HalfOpen { samples, periods } => {
                            std::f64::consts::TAU
                                * Real::from(k)
                                * Real::from(periods)
                                * Real::from(i)
                                / Real::from(samples)
                        }
                    };
                    let actual = geometry.coefficient(k, i);
                    let bound = 8. * Real::EPSILON * expected_angle.abs().max(1.);
                    assert!((actual[0] - expected_angle.sin()).abs() <= bound);
                    assert!((actual[1] - expected_angle.cos()).abs() <= bound);
                }
            }
        }
    }
    #[test]
    fn large_distinct_axis_is_charged_once_and_coincident_reads_are_charged_per_vector() {
        use super::OrderedAxis;
        assert!(preflight(1_000_000, AnalysisKind::Transient, &[card(1000, 9)], None).is_ok());
        let mut times = vec![0.; 1000];
        times.extend([0.5e-3, 1e-3]);
        let axis = OrderedAxis::borrowed(&times).unwrap();
        let mut budget = WorkBudget::new(times.len());
        let initial = budget.work;
        let card = card(10, 1);
        budget.charge_duplicate_reads(&axis, &card).unwrap();
        assert!(budget.work > initial + 9000);
    }

    #[test]
    fn high_bin_phase_reduction_is_periodic_and_preserves_endpoints() {
        let grid = ProjectionGrid::HalfOpen {
            samples: 100_000,
            periods: 100,
        };
        assert_eq!(grid.coefficient(100, 0), [0., 1.]);
        assert_eq!(grid.coefficient(100, 99_990), [0., 1.]);
        assert_eq!(grid.coefficient(100, 99_991), grid.coefficient(100, 1));
        let closed = ProjectionGrid::Closed(400);
        assert_eq!(closed.coefficient(100, 400), [0., 1.]);
    }

    #[test]
    fn batched_projection_agrees_with_independent_scalar_quadrature() {
        use super::{PreparedTrace, Trace};
        use crate::analysis::results::{Plot, PlotFlags, Variable};
        let mut builder = Plot::builder("test", "Transient Analysis", PlotFlags::Real);
        builder.push_variable(Variable::new("v(out)", "voltage"));
        let plot = builder.build();
        let c = card(1, 8);
        for geometry in [
            ProjectionGrid::Closed(204),
            ProjectionGrid::HalfOpen {
                samples: 1117,
                periods: 3,
            },
        ] {
            let mut batch: Vec<_> = (0..3)
                .map(|trace_index| {
                    let trace = Trace::resolve(&plot, &c.vectors[0]).unwrap();
                    let values = (0..geometry.samples())
                        .map(|i| {
                            let x = f64::from(u32::try_from(i).unwrap());
                            match (i + trace_index) % 7 {
                                0 => -0.,
                                1 => Real::from_bits(1),
                                _ => (x * 0.37).sin() * (1. + x * 0.007),
                            }
                        })
                        .collect();
                    PreparedTrace {
                        trace,
                        values,
                        sums: vec![[0.; 2]; 9],
                    }
                })
                .collect();
            geometry.project(&mut batch, 8).unwrap();
            for prepared in batch {
                let mut wanted = [[0.; 2]; 9];
                match geometry {
                    ProjectionGrid::Closed(divisions) => {
                        for (i, &y) in (0..=divisions).zip(&prepared.values) {
                            let weight = if i == 0 || i == divisions { 0.5 } else { 1. };
                            let weighted = weight * y;
                            let phase = Real::from(i) / Real::from(divisions);
                            for (k, [sin, cos]) in (0..=8).zip(&mut wanted) {
                                let angle = std::f64::consts::TAU * Real::from(k) * phase;
                                let sine_term = weighted * angle.sin();
                                let cosine_term = weighted * angle.cos();
                                *sin += sine_term;
                                *cos += cosine_term;
                            }
                        }
                    }
                    ProjectionGrid::HalfOpen { samples, periods } => {
                        for (k, [sin, cos]) in (1..=8).zip(wanted.iter_mut().skip(1)) {
                            for (i, &y) in (0..samples).zip(&prepared.values) {
                                let angle = std::f64::consts::TAU
                                    * Real::from(k)
                                    * Real::from(periods)
                                    * Real::from(i)
                                    / Real::from(samples);
                                let sine_term = y * angle.sin();
                                let cosine_term = y * angle.cos();
                                *sin += sine_term;
                                *cos += cosine_term;
                            }
                        }
                    }
                }
                let magnitude = prepared.values.iter().map(|v| v.abs()).sum::<Real>();
                // Naive summation has O(n*epsilon) relative-to-L1 error.
                // The unreduced phase also has O(k*periods*TAU*epsilon) error.
                let bound = (geometry.samples() as Real + 8. * 3. * std::f64::consts::TAU)
                    * Real::EPSILON
                    * magnitude;
                for (obtained, expected) in prepared.sums.iter().zip(wanted) {
                    for (&actual, reference) in obtained.iter().zip(expected) {
                        assert!((actual - reference).abs() <= bound);
                    }
                }
            }
        }
    }
}
