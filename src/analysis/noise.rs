//! `.noise` — small-signal noise analysis (adjoint method).
//!
//! C references (behaviour only, reimplemented): `noisean.c` (`NOISEan`: the
//! card, the frequency loop and the two plots), `cktnoise.c` (`CKTnoise`:
//! naming and summing), `nevalsrc.c` (`NevalSrc`: generator gains and
//! densities), `ninteg.c` (`Nintegrate`: the power-law integration),
//! `src/maths/ni/niniter.c` (`NInzIter`: the adjoint solve), and the device
//! routines summarised in [`crate::devices::noise`].
//!
//! ```text
//! .noise v(out[,ref]) src {dec|oct|lin} pts fstart fstop [pts_per_summary]
//! ```
//!
//! At every frequency the complex small-signal matrix `A + j omega E` of the
//! `.ac` analysis is factored once and solved twice:
//!
//! * forward, with a unit excitation of the input source only (C's
//!   `MODEACNOISE` zeroes every other AC source and drives the input with
//!   `1 + 0j` whatever its `ac` value), for the gain `H` from the input to the
//!   output; the input-referred noise is the output noise divided by
//!   `max(abs(H)^2, 1e-20)`;
//! * transposed (`A^T y = e_out`, a unit current between the output nodes),
//!   whose solution `y` holds every generator's transfer impedance: a
//!   generator between `n1` and `n2` reaches the output with the squared gain
//!   `abs(y[n1] - y[n2])^2`.
//!
//! Each generator's output density is integrated over frequency with C's
//! piecewise power law (`Nintegrate`) into the integrated output and input
//! noise. Results, exactly as C names and orders them:
//!
//! * plot 1, `Noise Spectral Density Curves` (real, `frequency` scale): with
//!   `pts_per_summary` the per-generator densities `onoise_<inst><suffix>` and
//!   per-instance totals `onoise_<inst>`, then `onoise_spectrum` and
//!   `inoise_spectrum`; with a summary, only every `pts_per_summary`-th
//!   frequency (counting from the first) is written;
//! * plot 2, `Integrated Noise` (real, one point, no scale), only when the
//!   start and stop frequencies differ: `v(onoise_total_<inst><suffix>)` /
//!   `inoise_total_...` pairs with a summary, then `v(onoise_total)` and the
//!   input total. The `v(...)`/`i(...)` spelling and the `voltage`/`current`
//!   units are those C's rawfile writer gives these vectors.
//!
//! Every value is the square root of the noise power (C without
//! `set sqrnoise`, supported through bounded pre-run settings): V/sqrt(Hz)
//! or A/sqrt(Hz) in plot 1, V or A in plot 2. Input-referred values are
//! voltages for a V input and currents for an I input.
//!
//! Divergences, all explicit errors where C would carry on: an output node
//! the circuit does not have (C creates it), identical output nodes, a
//! nonpositive or decreasing frequency range, an input that is not an
//! independent source with an `ac` value, and devices whose noise is not
//! ported. `.option keepopinfo`'s extra operating-point plot is not produced
//! (the option is not accepted by the port).

use crate::analysis::frequency::{Advance, FrequencyPlan, MAX_POINTS};
use crate::analysis::linear::{number, unsupported};
use crate::analysis::results::{Plot, PlotFlags, Variable};
use crate::analysis::{AnalysisContext, AnalysisPlan};
use crate::devices::noise::{BoundNoiseSource, NoisePlan};
use crate::devices::{Circuit, SourceKind};
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult, parse_spice_number};

mod integration;
mod plan;
use integration::Interval;
use plan::NoiseSolve;

/// C `N_MINGAIN`: the smallest squared input-output gain.
const N_MINGAIN: Real = 1e-20;
/// The spectrum plot's title (`noisean.c`, without `sqrnoise`).
pub const SPECTRUM_PLOTNAME: &str = "Noise Spectral Density Curves";
/// The integrated plot's title (`noisean.c`, without `sqrnoise`).
pub const INTEGRATED_PLOTNAME: &str = "Integrated Noise";

fn noise_failure(message: &str) -> SpiceError {
    SpiceError::Numerical {
        context: ".noise".into(),
        message: message.into(),
    }
}

/// The frequency stepping of a `.noise` card (`NOISEAN.NstpType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stepping {
    Decade,
    Octave,
    Linear,
}

impl Stepping {
    const fn parse(text: &str) -> Option<Self> {
        if text.eq_ignore_ascii_case("dec") {
            Some(Self::Decade)
        } else if text.eq_ignore_ascii_case("oct") {
            Some(Self::Octave)
        } else if text.eq_ignore_ascii_case("lin") {
            Some(Self::Linear)
        } else {
            None
        }
    }

    /// noisean.c's asymmetric singleton handling. For lin 1, C only
    /// replaces the stop when it is NOT almost equal. An unequal stop
    /// within three ULPs survives, producing a zero integrated-noise plot
    /// even though the frequency loop executes just once.
    fn normalize(self, steps: usize, start: Real, stop: Real) -> (usize, Real) {
        let equal = almost_equal_ulps(start, stop, 3);
        if self == Self::Linear && steps == 1 {
            (steps, if equal { stop } else { start })
        } else if equal {
            (1, start)
        } else {
            (steps, stop)
        }
    }
}

/// A parsed `.noise` card, after C's single-frequency adjustments.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct NoiseCard {
    positive: String,
    negative: Option<String>,
    input: String,
    stepping: Stepping,
    steps: u32,
    start: Real,
    stop: Real,
    summary: usize,
}

impl NoiseCard {
    pub(super) const fn plot_count(&self) -> usize {
        if self.start.to_bits() == self.stop.to_bits() {
            1
        } else {
            2
        }
    }

    fn output(&self, circuit: &Circuit) -> SpiceResult<[NodeId; 2]> {
        let node = |name: &str| {
            circuit.nodes().get(name).ok_or_else(|| {
                unsupported(format!(
                    ".noise output node '{name}' is not in the circuit (C would create a floating node)"
                ))
            })
        };
        let positive = node(&self.positive)?;
        let negative = self
            .negative
            .as_deref()
            .map(node)
            .transpose()?
            .unwrap_or(NodeId::GROUND);
        if positive == negative {
            return Err(unsupported(
                ".noise output nodes must differ (the output voltage is identically zero)",
            ));
        }
        Ok([positive, negative])
    }
}

/// C's `INPgetValue(IF_INTEGER)`: `floor(value + 0.5)`.
fn integer(text: Option<&str>, what: &str) -> SpiceResult<i64> {
    let value = text
        .and_then(parse_spice_number)
        .filter(|v| v.is_finite())
        .ok_or_else(|| unsupported(format!(".noise expects an integer {what}")))?;
    crate::primitives::c_integer(value)
        .ok_or_else(|| unsupported(format!(".noise {what} is out of range")))
}

use crate::primitives::almost_equal_ulps;

/// The positional tokens of a `.noise` card: `v ( out [, ref] ) src type pts
/// fstart fstop [pts_per_summary]`, the output spelled as one or several
/// tokens.
struct Positional {
    output: Vec<String>,
    rest: Vec<String>,
}

fn split_output(arguments: &[String]) -> SpiceResult<Positional> {
    let syntax = || {
        unsupported(
            ".noise needs v(out[,ref]) src {dec|oct|lin} pts fstart fstop [pts_per_summary]",
        )
    };
    let joined_end = arguments
        .iter()
        .position(|token| token.contains(')'))
        .ok_or_else(syntax)?;
    let text: String = arguments[..=joined_end].concat();
    let lower = text.to_ascii_lowercase();
    let inner = lower
        .strip_prefix("v(")
        .and_then(|rest| rest.strip_suffix(')'))
        .ok_or_else(|| {
            unsupported(format!(
                ".noise output must be v(out) or v(out,ref), not '{text}' (noisean.c measures \
                 a node voltage)"
            ))
        })?;
    let output: Vec<String> = inner.split(',').map(|s| s.trim().to_owned()).collect();
    if output.is_empty() || output.len() > 2 || output.iter().any(String::is_empty) {
        return Err(syntax());
    }
    Ok(Positional {
        output,
        rest: arguments[joined_end + 1..].to_vec(),
    })
}

/// Determine the number of plots emitted by a noise-analysis card.
///
pub(super) fn parse(positional: &[String]) -> SpiceResult<NoiseCard> {
    let Positional { output, rest } = split_output(positional)?;
    if !(5..=6).contains(&rest.len()) {
        return Err(unsupported(
            ".noise needs v(out[,ref]) src {dec|oct|lin} pts fstart fstop [pts_per_summary]",
        ));
    }
    let mut output = output.into_iter();
    let positive = output
        .next()
        .ok_or_else(|| unsupported(".noise output node"))?;
    let negative = output.next();
    let stepping = Stepping::parse(&rest[1]).ok_or_else(|| {
        unsupported(format!(
            ".noise sweep must be dec, oct or lin, not '{}'",
            rest[1]
        ))
    })?;
    let steps = integer(Some(&rest[2]), "point count")?;
    if steps < 1 {
        return Err(unsupported(format!(
            ".noise needs at least one step, got {steps} (noisean.c)"
        )));
    }
    let start = number(Some(&rest[3]), "start frequency")?;
    let stop = number(Some(&rest[4]), "stop frequency")?;
    let summary = match rest.get(5) {
        Some(text) => {
            let value = integer(Some(text), "pts_per_summary")?;
            usize::try_from(value).map_err(|_| {
                unsupported(format!(
                    ".noise pts_per_summary must not be negative, got {value}"
                ))
            })?
        }
        None => 0,
    };
    if start <= 0. || stop < start {
        return Err(unsupported(format!(
            ".noise needs 0 < fstart <= fstop, got {start} .. {stop}"
        )));
    }
    let steps = usize::try_from(steps).map_err(|_| unsupported(".noise point count"))?;
    let (steps, stop) = stepping.normalize(steps, start, stop);
    if steps > MAX_POINTS {
        return Err(unsupported(format!(
            ".noise point count is limited to {MAX_POINTS}"
        )));
    }
    let steps = u32::try_from(steps).map_err(|_| unsupported(".noise point count"))?;
    Ok(NoiseCard {
        positive,
        negative,
        input: rest[0].clone(),
        stepping,
        steps,
        start,
        stop,
        summary,
    })
}

/// Front-end representation of the same underlying noise powers. In C
/// `CKTnoise` writes raw powers and only applies sqrt when sqrnoise is off.
#[derive(Clone, Copy)]
enum Representation {
    Amplitude,
    Power,
}

impl Representation {
    const fn from_squared(squared: bool) -> Self {
        if squared {
            Self::Power
        } else {
            Self::Amplitude
        }
    }

    fn value(self, power: Real) -> Complex {
        Complex::real(match self {
            Self::Amplitude => power.sqrt(),
            Self::Power => power,
        })
    }

    const fn plotname(self, density: bool) -> &'static str {
        match (self, density) {
            (Self::Amplitude, true) => SPECTRUM_PLOTNAME,
            (Self::Amplitude, false) => INTEGRATED_PLOTNAME,
            (Self::Power, true) => "Noise Spectral Density Curves - (V^2 or A^2)/Hz",
            (Self::Power, false) => "Integrated Noise - V^2 or A^2",
        }
    }

    fn variable(self, name: String, kind: SourceKind, density: bool) -> Variable {
        let unit = match (self, kind, density) {
            (Self::Amplitude, SourceKind::Voltage, true) => "voltage-density",
            (Self::Amplitude, SourceKind::Current, true) => "current-density",
            (Self::Amplitude, SourceKind::Voltage, false) => "voltage",
            (Self::Amplitude, SourceKind::Current, false) => "current",
            (Self::Power, SourceKind::Voltage, true) => "voltage^2-density",
            (Self::Power, SourceKind::Current, true) => "current^2-density",
            (Self::Power, SourceKind::Voltage, false) => "voltage^2",
            (Self::Power, SourceKind::Current, false) => "current^2",
        };
        let name = if matches!(self, Self::Amplitude) && !density {
            let prefix = match kind {
                SourceKind::Voltage => "v",
                SourceKind::Current => "i",
            };
            format!("{prefix}({name})")
        } else {
            name
        };
        Variable::new(name, unit)
    }
}

/// An output row, separate from the integration histories. No amplitude
/// conversion ever feeds back into the integrator.
struct NoiseRow {
    representation: Representation,
    values: Vec<Complex>,
}

impl NoiseRow {
    fn new(representation: Representation, capacity: usize) -> Self {
        Self {
            representation,
            values: Vec::with_capacity(capacity),
        }
    }

    fn push_power(&mut self, power: Real) {
        self.values.push(self.representation.value(power));
    }
}

/// Integrated output- and input-referred noise power, in C's summation order.
#[derive(Debug, Clone, Copy, Default)]
struct Power {
    output: Real,
    input: Real,
}

impl Power {
    fn add(&mut self, contribution: Self) -> SpiceResult<()> {
        let sum = Self {
            output: self.output + contribution.output,
            input: self.input + contribution.input,
        };
        sum.check()?;
        *self = sum;
        Ok(())
    }

    fn check(self) -> SpiceResult<()> {
        if self.output.is_finite() && self.input.is_finite() {
            Ok(())
        } else {
            Err(noise_failure("nonfinite integrated noise"))
        }
    }

    fn append(self, row: &mut NoiseRow) {
        row.push_power(self.output);
        row.push_power(self.input);
    }
}

/// Frequency-local data shared by every generator.
struct NoiseSample {
    frequency: Real,
    interval: Interval,
    gain_sq_inv: Real,
    reset: bool,
    summary: bool,
}

impl NoiseSample {
    fn new(frequency: Real, previous: Real, gain_sq_inv: Real, start: Real, summary: bool) -> Self {
        Self {
            frequency,
            interval: Interval::new(previous, frequency),
            gain_sq_inv,
            reset: frequency.to_bits() == start.to_bits(),
            summary,
        }
    }
}

/// Per-generator integration history (C `<dev>nVar`). The optional instance
/// total only accumulates power: it never enters the logarithmic integrator.
#[derive(Debug, Clone, Copy, Default)]
struct History {
    previous_density: Real,
    integrated: Power,
}

impl History {
    fn advance(&mut self, density: Real, sample: &NoiseSample) -> SpiceResult<Power> {
        if sample.interval.is_empty() {
            self.previous_density = density;
            if sample.reset {
                self.integrated = Power::default();
            }
            return Ok(Power::default());
        }
        let area = sample.interval.integrate(self.previous_density, density);
        let contribution = Power {
            output: area.scaled(1.),
            input: area.scaled(sample.gain_sq_inv),
        };
        contribution.check()?;
        self.previous_density = density;
        if sample.summary {
            self.integrated.add(contribution)?;
        }
        Ok(contribution)
    }
}

/// One instance's fixed ordered output slots and their integration history.
struct Generators {
    history: Vec<History>,
    total: Option<Power>,
}

impl Generators {
    fn new(source_count: usize, total: bool) -> Self {
        Self {
            history: vec![History::default(); source_count],
            total: total.then(Power::default),
        }
    }

    /// Evaluate and integrate each source once, without allocating a density
    /// vector. Accumulate global totals per source, never per instance, to
    /// preserve the reference addition order.
    fn accumulate(
        &mut self,
        sources: &[BoundNoiseSource],
        sample: &NoiseSample,
        adjoint: &[Complex],
        integrated: &mut Power,
        mut summary_row: Option<&mut NoiseRow>,
    ) -> SpiceResult<Real> {
        let mut density_sum = 0.;
        if sample.reset
            && sample.interval.is_empty()
            && let Some(total) = &mut self.total
        {
            *total = Power::default();
        }
        for (source, history) in sources.iter().zip(&mut self.history) {
            let density = source.output_density(adjoint, sample.frequency)?;
            density_sum += density;
            let contribution = history.advance(density, sample)?;
            if !sample.interval.is_empty() {
                integrated.add(contribution)?;
                if sample.summary
                    && let Some(total) = &mut self.total
                {
                    total.add(contribution)?;
                }
            }
            if let Some(row) = summary_row.as_deref_mut() {
                row.push_power(density);
            }
        }
        if !density_sum.is_finite() {
            return Err(noise_failure("nonfinite instance noise density"));
        }
        if self.total.is_some()
            && let Some(row) = summary_row
        {
            row.push_power(density_sum);
        }
        Ok(density_sum)
    }

    fn append_integrated(&self, row: &mut NoiseRow) {
        for history in &self.history {
            history.integrated.append(row);
        }
        if let Some(total) = self.total {
            total.append(row);
        }
    }
}

/// The spectrum and integrated-plot columns (`CKTnoise` `N_OPEN` and the
/// devices' `NOISE_ADD_OUTVAR`); per-instance columns only with a summary.
fn plot_names(
    plan: &NoisePlan<'_>,
    input_kind: SourceKind,
    summary: bool,
    representation: Representation,
) -> (Vec<Variable>, Vec<Variable>) {
    let mut spectrum = vec![Variable::new("frequency", "frequency")];
    let mut integrated = Vec::new();
    let output =
        |name: String, density| representation.variable(name, SourceKind::Voltage, density);
    let input = |name: String, density| representation.variable(name, input_kind, density);
    for (instance, _) in plan.instances().filter(|_| summary) {
        let suffixes = instance
            .suffixes
            .iter()
            .map(std::borrow::Cow::as_ref)
            .chain(instance.total.then_some(""));
        for suffix in suffixes {
            let name = format!("{}{suffix}", instance.name);
            let separator = match instance.family {
                crate::devices::noise::NoiseFamily::Bsim3
                | crate::devices::noise::NoiseFamily::Bsim4 => '.',
                _ => '_',
            };
            spectrum.push(output(format!("onoise{separator}{name}"), true));
            integrated.push(output(format!("onoise_total{separator}{name}"), false));
            integrated.push(input(format!("inoise_total{separator}{name}"), false));
        }
    }
    spectrum.push(output("onoise_spectrum".into(), true));
    spectrum.push(input("inoise_spectrum".into(), true));
    integrated.push(output("onoise_total".into(), false));
    integrated.push(input("inoise_total".into(), false));
    (spectrum, integrated)
}

/// Resolve the independent input and validate its AC excitation.
fn input_kind(circuit: &Circuit, name: &str) -> SpiceResult<SourceKind> {
    let input = circuit
        .device(name)
        .ok_or_else(|| unsupported(format!(".noise input source {name} is not in the circuit")))?
        .input_source()
        .ok_or_else(|| {
            unsupported(format!(
                ".noise input {name} is not an independent voltage or current source"
            ))
        })?;
    if !input.ac_given {
        return Err(unsupported(format!(
            ".noise input source {name} has no AC value (noisean.c E_NOACINPUT)"
        )));
    }
    Ok(input.kind)
}

/// C's frequency grid, with exactly representable bounded point counts.
pub(super) fn frequency_grid(card: &NoiseCard, reltol: Real) -> SpiceResult<FrequencyPlan> {
    let delta = match card.stepping {
        Stepping::Decade => (10f64.ln() / Real::from(card.steps)).exp(),
        Stepping::Octave => (2f64.ln() / Real::from(card.steps)).exp(),
        Stepping::Linear if card.steps == 1 => 0.,
        Stepping::Linear => (card.stop - card.start) / Real::from(card.steps - 1),
    };
    let tolerance = match card.stepping {
        Stepping::Decade | Stepping::Octave => delta * card.stop * reltol,
        Stepping::Linear => delta * reltol,
    };
    let advance = match card.stepping {
        Stepping::Decade | Stepping::Octave => Advance::Multiply(delta),
        Stepping::Linear if card.steps == 1 => Advance::Single,
        Stepping::Linear => Advance::Add(delta),
    };
    FrequencyPlan::new(card.start, card.stop, tolerance, advance, ".noise")
}

fn integrated_plot(
    representation: Representation,
    integrated_variables: Vec<Variable>,
    generators: &[Generators],
    summary: bool,
    integrated_power: Power,
) -> SpiceResult<Plot> {
    let mut values = NoiseRow::new(representation, integrated_variables.len());
    let mut integrated = Plot::new(
        "noise2",
        representation.plotname(false),
        PlotFlags::Real,
        integrated_variables,
    );
    if summary {
        for g in generators {
            g.append_integrated(&mut values);
        }
    }
    integrated_power.append(&mut values);
    integrated.push_point(&values.values)?;
    if !integrated.is_finite() {
        return Err(SpiceError::Numerical {
            context: ".noise".into(),
            message: "nonfinite integrated noise".into(),
        });
    }
    Ok(integrated)
}

fn spectrum_plot(representation: Representation, spectrum_variables: Vec<Variable>) -> Plot {
    Plot::new(
        "noise1",
        representation.plotname(true),
        PlotFlags::Real,
        spectrum_variables,
    )
}

/// Emits the optional bias, streamed spectrum and, unless the sweep is a
/// single frequency, its integrated noise in that order.
pub(super) fn run(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    card: &NoiseCard,
    frequencies: &FrequencyPlan,
    include_bias: bool,
    sink: &mut dyn crate::analysis::PointSink,
) -> SpiceResult<()> {
    let options = request
        .noise_options()
        .ok_or_else(|| unsupported("noise driver requires a noise plan"))?;
    let settings = request.dc_settings().clone();
    circuit.finalize()?;
    let output = card.output(circuit)?;
    let input_kind = input_kind(circuit, &card.input)?;
    // The operating point and small-signal system exactly as `.ac` builds
    // them (`noisean.c`: `CKTop`, then a `MODEINITSMSIG` load; a
    // `hertz`-dependent circuit is re-solved at every frequency).
    let mut small = crate::analysis::ac::SmallSignal::prepare(circuit, request, context, settings)?;
    for bias in small.retained_bias(circuit, request, context, include_bias)? {
        sink.complete(bias)?;
    }
    let mut model = context.model_context();
    model.spice3_noise = options.spice3;
    // Generators that depend on the frequency are described at the first
    // frequency; they are re-described at every frequency below.
    let noise_varies = circuit
        .devices()
        .iter()
        .any(|device| device.noise_depends_on_frequency());
    let first = if small.reassembles() || noise_varies {
        model.with_frequency(card.start)
    } else {
        model
    };
    let mut noise_plan =
        NoisePlan::bind(small.prepared(), &first, small.bias(), Some(small.state()))?;
    let mut generators: Vec<_> = noise_plan
        .instances()
        .map(|(instance, sources)| Generators::new(sources.len(), instance.total))
        .collect();

    let representation = Representation::from_squared(options.squared);
    let (spectrum_variables, integrated_variables) =
        plot_names(&noise_plan, input_kind, card.summary != 0, representation);
    let spectrum = spectrum_plot(representation, spectrum_variables);
    let mut solve = NoiseSolve::bind(circuit, small.system(), &card.input, output)?;
    sink.begin(&spectrum)?;

    let mut integrated_power = Power::default();
    let mut last = card.start;
    let mut row = NoiseRow::new(representation, spectrum.variable_count());
    for (step, freq) in frequencies.values().enumerate() {
        let gain_sq_inv = small.at_frequency(context, freq, |system| {
            solve.validate(system)?;
            let matrix =
                crate::analysis::ac::ac_matrix(circuit, system, 2. * std::f64::consts::PI * freq)?;
            solve.evaluate(&matrix.factorize()?)
        })?;
        // The laws can change with bias/frequency but the named history
        // slots cannot. Refresh transactionally before consuming the point.
        if small.varies() || small.reassembles() || noise_varies {
            noise_plan.refresh(
                small.prepared(),
                &model.with_frequency(freq),
                small.bias(),
                Some(small.state()),
            )?;
        }
        let sample = NoiseSample::new(freq, last, gain_sq_inv, card.start, card.summary != 0);
        let print_summary = card.summary != 0 && step.is_multiple_of(card.summary);
        let mut out_density = 0.;
        let emit = card.summary == 0 || print_summary;
        row.values.clear();
        if emit {
            row.values.push(Complex::real(freq));
        }
        for (g, (_, sources)) in generators.iter_mut().zip(noise_plan.instances()) {
            out_density += g.accumulate(
                sources,
                &sample,
                solve.adjoint(),
                &mut integrated_power,
                print_summary.then_some(&mut row),
            )?;
        }
        let in_density = out_density * gain_sq_inv;
        if !out_density.is_finite() || !in_density.is_finite() {
            return Err(noise_failure("nonfinite noise density"));
        }
        if emit {
            row.push_power(out_density);
            row.push_power(in_density);
            spectrum.check_point(&row.values)?;
            if !row.values.iter().all(|value| value.is_finite()) {
                return Err(SpiceError::Numerical {
                    context: ".noise".into(),
                    message: "nonfinite noise density".into(),
                });
            }
            sink.point(&row.values)?;
        }
        last = freq;
    }
    sink.end()?;
    if card.start.to_bits() != card.stop.to_bits() {
        sink.complete(integrated_plot(
            representation,
            integrated_variables,
            &generators,
            card.summary != 0,
            integrated_power,
        )?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Generators, NoiseRow, NoiseSample, Power, Representation, almost_equal_ulps};
    use crate::devices::MnaUnknowns;
    use crate::devices::noise::{BoundNoiseSource, NoiseKind, NoiseSource};
    use crate::primitives::NodeTable;
    use crate::primitives::{Complex, NodeId};

    fn args(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    fn source(suffix: &'static str, node: u32, factor: f64) -> BoundNoiseSource {
        let mut nodes = NodeTable::new();
        nodes.intern("a").unwrap();
        nodes.intern("b").unwrap();
        let mut unknowns = MnaUnknowns::new();
        unknowns.rebuild(&nodes);
        BoundNoiseSource::bind(
            &NoiseSource::new(
                suffix,
                [NodeId::new(node), NodeId::GROUND],
                NoiseKind::Gain { factor },
            ),
            &unknowns,
            0..0,
            None,
        )
        .unwrap()
    }

    fn sources() -> [BoundNoiseSource; 2] {
        [source("_a", 1, 2.), source("_b", 1, 3.)]
    }

    #[test]
    fn noise_history_is_bound_to_named_slots_not_frequency_local_values() {
        let mut generators = Generators::new(2, true);
        let mut integrated = Power::default();
        let mut row = NoiseRow::new(Representation::Amplitude, 6);
        let transfer = [Complex::real(1.), Complex::real(0.5)];
        let first = NoiseSample::new(10., 10., 0.25, 10., true);
        assert_eq!(
            generators
                .accumulate(
                    &sources(),
                    &first,
                    &transfer,
                    &mut integrated,
                    Some(&mut row)
                )
                .unwrap(),
            5.
        );
        assert_eq!(
            row.values,
            [2_f64.sqrt(), 3_f64.sqrt(), 5_f64.sqrt()].map(Complex::real)
        );
        let mut fresh = sources();
        // Change both the law and the transfer pair while leaving the
        // resulting output power unchanged: 8 * (0.5)^2 == 2.
        fresh[0] = source("_a", 2, 8.);
        let second = NoiseSample::new(20., 10., 0.25, 10., true);
        // Integrate even when this frequency has no printed summary row.
        assert_eq!(
            generators
                .accumulate(&fresh, &second, &transfer, &mut integrated, None)
                .unwrap(),
            5.
        );
        assert_eq!(integrated.output, 50.);
        assert_eq!(integrated.input, 12.5);
        row.values.clear();
        generators.append_integrated(&mut row);
        assert_eq!(
            row.values,
            [
                20_f64.sqrt(),
                5_f64.sqrt(),
                30_f64.sqrt(),
                7.5_f64.sqrt(),
                50_f64.sqrt(),
                12.5_f64.sqrt()
            ]
            .map(Complex::real)
        );
    }

    #[test]
    fn invalid_generator_is_rejected_even_without_a_summary_row() {
        let mut generators = Generators::new(2, true);
        let mut integrated = Power::default();
        let sample = NoiseSample::new(10., 10., 0.25, 10., true);
        let invalid_transfer = [Complex::real(f64::NAN), Complex::ZERO];
        assert!(
            generators
                .accumulate(
                    &sources(),
                    &sample,
                    &invalid_transfer,
                    &mut integrated,
                    None
                )
                .is_err()
        );
    }

    #[test]
    fn squared_noise_preserves_original_powers_in_every_slot() {
        let mut generators = Generators::new(2, true);
        let mut integrated = Power::default();
        let mut row = NoiseRow::new(Representation::Power, 6);
        let transfer = [Complex::real(1.), Complex::real(0.5)];
        let first = NoiseSample::new(10., 10., 0.25, 10., true);
        generators
            .accumulate(
                &sources(),
                &first,
                &transfer,
                &mut integrated,
                Some(&mut row),
            )
            .unwrap();
        // sqrt(2)^2 != 2 in binary64: the former rendering lost this bit.
        assert_ne!(2_f64.sqrt().powi(2), 2.);
        assert_eq!(row.values, [2., 3., 5.].map(Complex::real));
        let second = NoiseSample::new(20., 10., 0.25, 10., true);
        generators
            .accumulate(&sources(), &second, &transfer, &mut integrated, None)
            .unwrap();
        row.values.clear();
        generators.append_integrated(&mut row);
        integrated.append(&mut row);
        assert_eq!(
            row.values,
            [20., 5., 30., 7.5, 50., 12.5, 50., 12.5].map(Complex::real)
        );
    }

    #[test]
    fn plot_count_follows_the_single_frequency_rules() {
        use crate::analysis::AnalysisInput;
        use crate::primitives::AnalysisKind;
        let plot_count = |arguments: &[String]| {
            AnalysisInput::with_arguments(AnalysisKind::Noise, arguments.iter().cloned())
                .resolve()
                .unwrap()
                .plot_count()
        };
        assert_eq!(plot_count(&args("v ( out ) v1 dec 10 1 1k")), 2);
        assert_eq!(plot_count(&args("v ( out ) v1 dec 10 1k 1k")), 1);
        assert_eq!(plot_count(&args("v ( out , ref ) v1 lin 1 1 1k 2")), 1);
        assert_eq!(plot_count(&args("v ( out ) v1 lin 2 1 1k rtol=1e-3")), 2);
        assert!(
            AnalysisInput::with_arguments(AnalysisKind::Noise, ["garbage"])
                .resolve()
                .is_err()
        );
    }

    #[test]
    fn almost_equal_linear_singleton_retains_integrated_plot_like_c() {
        use crate::analysis::{AnalysisContext, AnalysisInput};
        use crate::devices::Circuit;
        use crate::netlist::{Parser, source::parse_deck_text};
        use crate::primitives::AnalysisKind;
        use std::path::Path;
        let netlist = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("noise-single.cir"),
                "nearby noise bounds\nv1 a 0 dc 0 ac 1\nr1 a 0 1k\n.end\n",
            ))
            .unwrap();
        let start = 1_f64;
        for distance in 0..=5 {
            let stop = f64::from_bits(start.to_bits() + distance);
            for mode in ["lin", "dec", "oct"] {
                let arguments = args(&format!("v ( a ) v1 {mode} 1 {start:.17} {stop:.17}"));
                let request = AnalysisInput::with_arguments(AnalysisKind::Noise, arguments.clone());
                let parsed_start = crate::primitives::parse_spice_number(&arguments[7]).unwrap();
                let parsed_stop = crate::primitives::parse_spice_number(&arguments[8]).unwrap();
                let equal = almost_equal_ulps(parsed_start, parsed_stop, 3);
                let integrated = if mode == "lin" {
                    equal && parsed_start != parsed_stop
                } else {
                    !equal
                };
                let expected = 1 + usize::from(integrated);
                let mut circuit = Circuit::from_netlist(&netlist).unwrap();
                let plan = request.resolve().unwrap();
                assert_eq!(plan.plot_count(), expected);
                let plots = plan
                    .run_plots(&mut circuit, &AnalysisContext::default())
                    .unwrap();
                assert_eq!(plots.len(), expected, "{mode} {distance}");
                assert_eq!(plots[0].point_count(), 1);
                if let Some(integrated) = plots.get(1) {
                    assert_eq!(integrated.row(0).unwrap(), [Complex::ZERO; 2]);
                }
            }
        }
    }

    #[test]
    fn ulps_comparison_matches_c() {
        assert!(almost_equal_ulps(1.0, 1.0, 3));
        assert!(almost_equal_ulps(1.0, 1.0 + 2.0 * f64::EPSILON, 3));
        assert!(!almost_equal_ulps(1.0, 1.0 + 8.0 * f64::EPSILON, 3));
        assert!(!almost_equal_ulps(1.0, -1.0, 3));
    }

    #[test]
    fn nonfinite_noise_is_rejected_even_between_summary_rows() {
        let mut generators = Generators::new(2, true);
        let mut integrated = Power::default();
        let fresh = [source("_a", 1, f64::MAX), source("_b", 1, 0.)];
        let transfer = [Complex::real(1.), Complex::real(0.5)];
        let first = NoiseSample::new(1., 1., 1., 1., true);
        generators
            .accumulate(&fresh, &first, &transfer, &mut integrated, None)
            .unwrap();
        let second = NoiseSample::new(3., 1., 1., 1., true);
        assert!(
            generators
                .accumulate(&fresh, &second, &transfer, &mut integrated, None)
                .is_err()
        );
    }
}
