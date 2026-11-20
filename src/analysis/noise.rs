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

use crate::analysis::frequency::{Advance, FrequencyGrid, MAX_POINTS};
use crate::analysis::linear::{number, unsupported};
use crate::analysis::results::{Plot, PlotFlags, Variable};
use crate::analysis::{AnalysisContext, AnalysisRequest};
use crate::devices::noise::{InstanceNoise, circuit_noise};
use crate::devices::{Circuit, LinearSystem, SourceKind};
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult, parse_spice_number};

/// C `N_MINLOG`: the smallest number whose logarithm is taken.
const N_MINLOG: Real = 1e-38;
/// C `N_MINGAIN`: the smallest squared input-output gain.
const N_MINGAIN: Real = 1e-20;
/// C `N_INTFTHRESH`: the largest log-log slope still integrated as flat.
const N_INTFTHRESH: Real = 1e-10;
/// C `N_INTUSELOG`: the region around a slope of -1 integrated as `ln f`.
const N_INTUSELOG: Real = 1e-10;
/// C's default `CKTreltol`, the frequency-loop tolerance without `.option
/// reltol`.
const DEFAULT_RELTOL: Real = 1e-3;
/// The spectrum plot's title (`noisean.c`, without `sqrnoise`).
pub const SPECTRUM_PLOTNAME: &str = "Noise Spectral Density Curves";
/// The integrated plot's title (`noisean.c`, without `sqrnoise`).
pub const INTEGRATED_PLOTNAME: &str = "Integrated Noise";

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
struct NoiseCard {
    positive: NodeId,
    negative: NodeId,
    input: String,
    stepping: Stepping,
    steps: u32,
    start: Real,
    stop: Real,
    summary: usize,
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
/// How many plots a `.noise` card produces: two (the spectrum and the
/// integrated noise), or one when C normalizes the stop to the start.
/// A one-point linear sweep with distinct, nearly equal bounds is the
/// exception: the integrated plot still exists. Used by the batch
/// scheduler to name the plots before anything runs; a card the driver would
/// reject (or one with braced, not yet evaluated, arguments) counts two.
#[must_use]
pub fn plot_count(arguments: &[String]) -> usize {
    let positional: Vec<String> = arguments
        .iter()
        .filter(|argument| !argument.contains('='))
        .cloned()
        .collect();
    let Ok(Positional { rest, .. }) = split_output(&positional) else {
        return 2;
    };
    let (Some(kind), Ok(steps), Ok(start), Ok(stop)) = (
        rest.get(1),
        integer(rest.get(2).map(String::as_str), "point count"),
        number(rest.get(3).map(String::as_str), "start frequency"),
        number(rest.get(4).map(String::as_str), "stop frequency"),
    ) else {
        return 2;
    };
    let Some(stepping) = Stepping::parse(kind) else {
        return 2;
    };
    let Ok(steps) = usize::try_from(steps) else {
        return 2;
    };
    if steps == 0 || start <= 0. || stop < start {
        return 2;
    }
    let (_, stop) = stepping.normalize(steps, start, stop);
    if start.to_bits() == stop.to_bits() {
        1
    } else {
        2
    }
}

fn parse(circuit: &Circuit, request: &AnalysisRequest) -> SpiceResult<NoiseCard> {
    let positional: Vec<String> = request
        .arguments
        .iter()
        .filter(|argument| !argument.contains('='))
        .cloned()
        .collect();
    let Positional { output, rest } = split_output(&positional)?;
    if !(5..=6).contains(&rest.len()) {
        return Err(unsupported(
            ".noise needs v(out[,ref]) src {dec|oct|lin} pts fstart fstop [pts_per_summary]",
        ));
    }
    let node = |name: &str| {
        circuit.nodes().get(name).ok_or_else(|| {
            unsupported(format!(
                ".noise output node '{name}' is not in the circuit (C would create a floating \
                 node)"
            ))
        })
    };
    let positive = node(&output[0])?;
    let negative = match output.get(1) {
        Some(name) => node(name)?,
        None => NodeId::GROUND,
    };
    if positive == negative {
        return Err(unsupported(
            ".noise output nodes must differ (the output voltage is identically zero)",
        ));
    }
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

/// `ninteg.c`'s per-step frequency data.
struct Step {
    width: Real,
    log_end: Real,
    log_start: Real,
    log_width: Real,
}

/// C's `limexp`.
fn limexp(x: Real) -> Real {
    if x > 700. {
        700f64.exp() * (1. + x - 700.)
    } else {
        x.exp()
    }
}

/// `Nintegrate`: the integral of `a f^k` between the previous and the present
/// frequency, given the density and its logarithm at both ends.
fn nintegrate(density: Real, ln_density: Real, ln_last_density: Real, step: &Step) -> Real {
    let mut exponent = (ln_density - ln_last_density) / step.log_width;
    if exponent.abs() < N_INTFTHRESH {
        density * step.width
    } else {
        let log_frequency_power = exponent * step.log_end;
        let a = limexp(ln_density - log_frequency_power);
        exponent += 1.;
        if exponent.abs() < N_INTUSELOG {
            a * (step.log_end - step.log_start)
        } else {
            a * ((limexp(exponent * step.log_end) - limexp(exponent * step.log_start)) / exponent)
        }
    }
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
    fn add(&mut self, contribution: Self) {
        self.output += contribution.output;
        self.input += contribution.input;
    }

    fn append(self, row: &mut NoiseRow) {
        row.push_power(self.output);
        row.push_power(self.input);
    }
}

/// Frequency-local data shared by every generator.
struct NoiseSample {
    frequency: Real,
    step: Step,
    gain_sq_inv: Real,
    ln_gain_inv: Real,
    reset: bool,
    summary: bool,
}

impl NoiseSample {
    fn new(frequency: Real, previous: Real, gain_sq_inv: Real, start: Real, summary: bool) -> Self {
        let log_end = frequency.max(N_MINLOG).ln();
        let log_start = previous.max(N_MINLOG).ln();
        Self {
            frequency,
            step: Step {
                width: frequency - previous,
                log_end,
                log_start,
                log_width: log_end - log_start,
            },
            gain_sq_inv,
            ln_gain_inv: gain_sq_inv.ln(),
            reset: frequency.to_bits() == start.to_bits(),
            summary,
        }
    }
}

/// Per-generator integration history (C `<dev>nVar`). The optional instance
/// total only accumulates power: it never enters the logarithmic integrator.
#[derive(Debug, Clone, Copy, Default)]
struct History {
    ln_last: Real,
    integrated: Power,
}

impl History {
    fn advance(&mut self, density: Real, sample: &NoiseSample) -> Power {
        let ln_density = density.max(N_MINLOG).ln();
        if sample.step.width == 0. {
            self.ln_last = ln_density;
            if sample.reset {
                self.integrated = Power::default();
            }
            return Power::default();
        }
        let contribution = Power {
            output: nintegrate(density, ln_density, self.ln_last, &sample.step),
            input: nintegrate(
                density * sample.gain_sq_inv,
                ln_density + sample.ln_gain_inv,
                self.ln_last + sample.ln_gain_inv,
                &sample.step,
            ),
        };
        self.ln_last = ln_density;
        if sample.summary {
            self.integrated.add(contribution);
        }
        contribution
    }
}

/// One instance's fixed ordered output slots and their integration history.
struct Generators {
    instance: InstanceNoise,
    history: Vec<History>,
    total: Option<Power>,
}

impl Generators {
    fn new(instance: InstanceNoise) -> Self {
        let history = vec![History::default(); instance.sources.len()];
        let total = instance.total.then(Power::default);
        Self {
            instance,
            history,
            total,
        }
    }

    /// Frequency-dependent devices may change their laws and transfer nodes,
    /// but cannot change the named slots that own integration history and columns.
    fn refresh(&mut self, instance: InstanceNoise) -> SpiceResult<()> {
        let old = &self.instance;
        if old.device != instance.device
            || old.name != instance.name
            || old.total != instance.total
            || old.sources.len() != instance.sources.len()
            || old
                .sources
                .iter()
                .zip(&instance.sources)
                .any(|(a, b)| a.suffix != b.suffix)
        {
            return Err(SpiceError::circuit(format!(
                "noise generator layout changed between frequencies for {}",
                old.name
            )));
        }
        self.instance = instance;
        Ok(())
    }

    /// Evaluate and integrate each source once, without allocating a density
    /// vector. Accumulate global totals per source, never per instance, to
    /// preserve the reference addition order.
    fn accumulate(
        &mut self,
        sample: &NoiseSample,
        difference: impl Fn([NodeId; 2]) -> Complex,
        integrated: &mut Power,
        mut summary_row: Option<&mut NoiseRow>,
    ) -> Real {
        let mut density_sum = 0.;
        if sample.reset
            && sample.step.width == 0.
            && let Some(total) = &mut self.total
        {
            *total = Power::default();
        }
        for (source, history) in self.instance.sources.iter().zip(&mut self.history) {
            let density = source.output_density(&difference, sample.frequency);
            density_sum += density;
            let contribution = history.advance(density, sample);
            if sample.step.width != 0. {
                integrated.add(contribution);
                if sample.summary
                    && let Some(total) = &mut self.total
                {
                    total.add(contribution);
                }
            }
            if let Some(row) = summary_row.as_deref_mut() {
                row.push_power(density);
            }
        }
        if self.total.is_some()
            && let Some(row) = summary_row
        {
            row.push_power(density_sum);
        }
        density_sum
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
    generators: &[Generators],
    input_kind: SourceKind,
    summary: bool,
    representation: Representation,
) -> (Vec<Variable>, Vec<Variable>) {
    let mut spectrum = vec![Variable::new("frequency", "frequency")];
    let mut integrated = Vec::new();
    let output =
        |name: String, density| representation.variable(name, SourceKind::Voltage, density);
    let input = |name: String, density| representation.variable(name, input_kind, density);
    for g in generators.iter().filter(|_| summary) {
        let suffixes = g
            .instance
            .sources
            .iter()
            .map(|source| source.suffix)
            .chain(g.instance.total.then_some(""));
        for suffix in suffixes {
            let name = format!("{}{suffix}", g.instance.name);
            spectrum.push(output(format!("onoise_{name}"), true));
            integrated.push(output(format!("onoise_total_{name}"), false));
            integrated.push(input(format!("inoise_total_{name}"), false));
        }
    }
    spectrum.push(output("onoise_spectrum".into(), true));
    spectrum.push(input("inoise_spectrum".into(), true));
    integrated.push(output("onoise_total".into(), false));
    integrated.push(input("inoise_total".into(), false));
    (spectrum, integrated)
}

/// The squared magnitude of `y[n1] - y[n2]`, ground being zero.
fn gain(circuit: &Circuit, y: &[Complex], nodes: [NodeId; 2]) -> Real {
    let difference = difference(circuit, y, nodes);
    let real_power = difference.re * difference.re;
    let imaginary_power = difference.im * difference.im;
    real_power + imaginary_power
}

/// `y[n1] - y[n2]`, ground reading zero (C's `CKTrhs`/`CKTirhs`
/// differences in `nevalsrc.c`).
fn difference(circuit: &Circuit, y: &[Complex], nodes: [NodeId; 2]) -> Complex {
    let at = |node: NodeId| {
        circuit
            .unknowns()
            .node_row(node)
            .and_then(|row| y.get(row).copied())
            .unwrap_or(Complex::ZERO)
    };
    at(nodes[0]) - at(nodes[1])
}

/// The unit forcing of the input source alone (C `MODEACNOISE`).
fn input_rhs(system: &LinearSystem, input: &str) -> SpiceResult<Vec<Complex>> {
    let source = system
        .sources
        .iter()
        .find(|source| source.name.eq_ignore_ascii_case(input))
        .ok_or_else(|| SpiceError::circuit(format!("noise input source {input} has no AC row")))?;
    let mut rhs = vec![Complex::ZERO; system.a.rows()];
    for (row, sign) in &source.rows {
        rhs[*row] = rhs[*row] + Complex::real(*sign);
    }
    Ok(rhs)
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
fn frequency_grid(card: &NoiseCard, reltol: Real) -> SpiceResult<FrequencyGrid> {
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
    FrequencyGrid::new(card.start, card.stop, tolerance, advance, ".noise")
}

fn integrated_plot(
    representation: Representation,
    integrated_variables: Vec<Variable>,
    generators: &[Generators],
    summary: bool,
    integrated_power: Power,
) -> SpiceResult<Plot> {
    let mut integrated = Plot::new("noise2", representation.plotname(false), PlotFlags::Real);
    let mut values = NoiseRow::new(representation, integrated_variables.len());
    for variable in integrated_variables {
        integrated.push_variable(variable);
    }
    if summary {
        for g in generators {
            g.append_integrated(&mut values);
        }
    }
    integrated_power.append(&mut values);
    integrated.push_point(values.values)?;
    if !integrated.is_finite() {
        return Err(SpiceError::Numerical {
            context: ".noise".into(),
            message: "nonfinite integrated noise".into(),
        });
    }
    Ok(integrated)
}

/// Runs `.noise` and returns its plots: the spectrum and, unless the sweep is
/// a single frequency, the integrated noise.
pub(crate) fn run(
    circuit: &mut Circuit,
    request: &AnalysisRequest,
    context: &AnalysisContext,
    include_bias: bool,
) -> SpiceResult<Vec<Plot>> {
    let settings = crate::analysis::bias::DcSettings::from_request(request)?;
    let reltol = match request.named("rtol") {
        Some(text) => number(Some(text), "rtol")?,
        None => DEFAULT_RELTOL,
    };
    circuit.finalize()?;
    let card = parse(circuit, request)?;
    let input_kind = input_kind(circuit, &card.input)?;
    // The operating point and small-signal system exactly as `.ac` builds
    // them (`noisean.c`: `CKTop`, then a `MODEINITSMSIG` load; a
    // `hertz`-dependent circuit is re-solved at every frequency).
    let mut small = crate::analysis::ac::SmallSignal::prepare(circuit, request, context, settings)?;
    let mut plots = small.retained_bias(circuit, request, context, include_bias)?;
    let mut model = context.model_context();
    model.spice3_noise = request.spice3_noise;
    // Generators that depend on the frequency are described at the first
    // frequency; they are re-described at every frequency below.
    let first = if small.reassembles() {
        model.with_frequency(card.start)
    } else {
        model
    };
    let mut generators: Vec<Generators> =
        circuit_noise(circuit, &first, small.bias(), Some(small.state()))?
            .into_iter()
            .map(Generators::new)
            .collect();

    let representation = Representation::from_squared(request.squared_noise);
    let (spectrum_variables, integrated_variables) =
        plot_names(&generators, input_kind, card.summary != 0, representation);
    let mut spectrum = Plot::new("noise1", representation.plotname(true), PlotFlags::Real);
    for variable in spectrum_variables {
        spectrum.push_variable(variable);
    }
    let output = [card.positive, card.negative];
    let mut adjoint_rhs = vec![Complex::ZERO; circuit.unknown_count()];
    if let Some(row) = circuit.unknowns().node_row(card.positive) {
        adjoint_rhs[row] = Complex::real(1.);
    }
    if let Some(row) = circuit.unknowns().node_row(card.negative) {
        adjoint_rhs[row] = Complex::real(-1.);
    }

    let frequencies = frequency_grid(&card, reltol)?;
    let mut integrated_power = Power::default();
    let mut last = card.start;
    for (step, frequency) in frequencies.enumerate() {
        let freq = frequency?;
        let input_name = card.input.as_str();
        let (forward, adjoint) = small.at_frequency(circuit, context, freq, |system| {
            let matrix =
                crate::analysis::ac::ac_matrix(circuit, system, 2. * std::f64::consts::PI * freq)?;
            let lu = matrix.factorize()?;
            Ok((
                lu.solve(&input_rhs(system, input_name)?)?,
                lu.solve_transposed(&adjoint_rhs)?,
            ))
        })?;
        // A device whose small-signal load depends on the frequency also
        // describes its generators at that frequency (`b4noi.c` evaluates
        // its flicker factor and correlated gate noise at `data->freq`).
        if small.varies() || small.reassembles() {
            let local = model.with_frequency(freq);
            let fresh = circuit_noise(circuit, &local, small.bias(), Some(small.state()))?;
            if fresh.len() != generators.len() {
                return Err(SpiceError::circuit(
                    "noise generators changed between frequencies",
                ));
            }
            for (g, instance) in generators.iter_mut().zip(fresh) {
                g.refresh(instance)?;
            }
        }
        let gain_sq_inv = 1. / gain(circuit, &forward, output).max(N_MINGAIN);
        let sample = NoiseSample::new(freq, last, gain_sq_inv, card.start, card.summary != 0);
        let print_summary = card.summary != 0 && step.is_multiple_of(card.summary);
        let mut out_density = 0.;
        let emit = card.summary == 0 || print_summary;
        let mut row = NoiseRow::new(
            representation,
            if emit { spectrum.variables.len() } else { 0 },
        );
        if emit {
            row.values.push(Complex::real(freq));
        }
        for g in &mut generators {
            out_density += g.accumulate(
                &sample,
                |nodes| difference(circuit, &adjoint, nodes),
                &mut integrated_power,
                print_summary.then_some(&mut row),
            );
        }
        if emit {
            row.push_power(out_density);
            row.push_power(out_density * gain_sq_inv);
            spectrum.push_point(row.values)?;
        }
        last = freq;
    }
    if !spectrum.is_finite() {
        return Err(SpiceError::Numerical {
            context: ".noise".into(),
            message: "nonfinite noise density".into(),
        });
    }
    plots.push(spectrum);
    if card.start.to_bits() != card.stop.to_bits() {
        plots.push(integrated_plot(
            representation,
            integrated_variables,
            &generators,
            card.summary != 0,
            integrated_power,
        )?);
    }
    Ok(plots)
}

#[cfg(test)]
mod tests {
    use super::{
        Generators, NoiseRow, NoiseSample, Power, Representation, Step, almost_equal_ulps,
        nintegrate, plot_count,
    };
    use crate::devices::noise::{InstanceNoise, NoiseKind, NoiseSource};
    use crate::primitives::{Complex, NodeId};

    fn args(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    fn instance() -> InstanceNoise {
        InstanceNoise {
            device: 3,
            name: "test".into(),
            sources: vec![
                NoiseSource::new(
                    "_a",
                    [NodeId::new(1), NodeId::GROUND],
                    NoiseKind::Gain { factor: 2. },
                ),
                NoiseSource::new(
                    "_b",
                    [NodeId::new(1), NodeId::GROUND],
                    NoiseKind::Gain { factor: 3. },
                ),
            ],
            total: true,
        }
    }

    #[test]
    fn noise_history_is_bound_to_named_slots_not_frequency_local_values() {
        let mut generators = Generators::new(instance());
        let mut integrated = Power::default();
        let mut row = NoiseRow::new(Representation::Amplitude, 6);
        let transfer = |_| Complex::real(1.);
        let first = NoiseSample::new(10., 10., 0.25, 10., true);
        assert_eq!(
            generators.accumulate(&first, transfer, &mut integrated, Some(&mut row)),
            5.
        );
        assert_eq!(
            row.values,
            [2_f64.sqrt(), 3_f64.sqrt(), 5_f64.sqrt()].map(Complex::real)
        );
        let mut fresh = instance();
        fresh.sources[0].kind = NoiseKind::Gain { factor: 2. };
        fresh.sources[0].nodes = [NodeId::new(2), NodeId::GROUND];
        generators.refresh(fresh).unwrap();
        let second = NoiseSample::new(20., 10., 0.25, 10., true);
        // Integrate even when this frequency has no printed summary row.
        assert_eq!(
            generators.accumulate(&second, transfer, &mut integrated, None),
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
    fn squared_noise_preserves_original_powers_in_every_slot() {
        let mut generators = Generators::new(instance());
        let mut integrated = Power::default();
        let mut row = NoiseRow::new(Representation::Power, 6);
        let transfer = |_| Complex::real(1.);
        let first = NoiseSample::new(10., 10., 0.25, 10., true);
        generators.accumulate(&first, transfer, &mut integrated, Some(&mut row));
        // sqrt(2)^2 != 2 in binary64: the former rendering lost this bit.
        assert_ne!(2_f64.sqrt().powi(2), 2.);
        assert_eq!(row.values, [2., 3., 5.].map(Complex::real));
        let second = NoiseSample::new(20., 10., 0.25, 10., true);
        generators.accumulate(&second, transfer, &mut integrated, None);
        row.values.clear();
        generators.append_integrated(&mut row);
        integrated.append(&mut row);
        assert_eq!(
            row.values,
            [20., 5., 30., 7.5, 50., 12.5, 50., 12.5].map(Complex::real)
        );
    }

    #[test]
    fn refreshed_noise_slots_reject_changed_identity_order_and_shape() {
        let mut generators = Generators::new(instance());
        let mut changed = instance();
        changed.sources.swap(0, 1);
        assert!(generators.refresh(changed).is_err());
        let mut changed = instance();
        changed.sources.pop();
        assert!(generators.refresh(changed).is_err());
        let mut changed = instance();
        changed.total = false;
        assert!(generators.refresh(changed).is_err());
        let mut changed = instance();
        changed.device = 4;
        assert!(generators.refresh(changed).is_err());
        let mut changed = instance();
        changed.name = "other".into();
        assert!(generators.refresh(changed).is_err());
        assert!(generators.refresh(instance()).is_ok());
    }

    #[test]
    fn plot_count_follows_the_single_frequency_rules() {
        assert_eq!(plot_count(&args("v ( out ) v1 dec 10 1 1k")), 2);
        assert_eq!(plot_count(&args("v ( out ) v1 dec 10 1k 1k")), 1);
        assert_eq!(plot_count(&args("v ( out , ref ) v1 lin 1 1 1k 2")), 1);
        assert_eq!(plot_count(&args("v ( out ) v1 lin 2 1 1k rtol=1e-3")), 2);
        assert_eq!(plot_count(&args("garbage")), 2);
    }

    #[test]
    fn almost_equal_linear_singleton_retains_integrated_plot_like_c() {
        use crate::analysis::{AnalysisContext, AnalysisRequest, runner};
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
                let request =
                    AnalysisRequest::with_arguments(AnalysisKind::Noise, arguments.clone());
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
                let plots = runner(AnalysisKind::Noise)
                    .unwrap()
                    .run_plots(&mut circuit, &request, &AnalysisContext::default())
                    .unwrap();
                assert_eq!(plots.len(), expected, "{mode} {distance}");
                assert_eq!(plot_count(&arguments), expected);
                assert_eq!(plots[0].point_count(), 1);
                if let Some(integrated) = plots.get(1) {
                    assert_eq!(integrated.points[0], [Complex::ZERO; 2]);
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
    fn nintegrate_is_exact_for_power_laws() {
        // A flat density integrates to density * df.
        let (f0, f1) = (10.0_f64, 100.0_f64);
        let step = Step {
            width: f1 - f0,
            log_end: f1.ln(),
            log_start: f0.ln(),
            log_width: f1.ln() - f0.ln(),
        };
        let flat = nintegrate(2.0, 2f64.ln(), 2f64.ln(), &step);
        assert!((flat - 180.0).abs() < 1e-12);
        // 1/f integrates to ln(f1/f0).
        let pink = nintegrate(1.0 / f1, (1.0 / f1).ln(), (1.0 / f0).ln(), &step);
        assert!((pink - 10f64.ln()).abs() < 1e-12);
        // 1/f^2 integrates to 1/f0 - 1/f1.
        let brown = nintegrate(
            1.0 / (f1 * f1),
            (1.0 / (f1 * f1)).ln(),
            (1.0 / (f0 * f0)).ln(),
            &step,
        );
        assert!((brown - (1.0 / f0 - 1.0 / f1)).abs() < 1e-12);
    }
}
