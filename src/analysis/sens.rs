//! `.sens` — DC and AC sensitivity of one output to every device parameter.
//!
//! C (behaviour only, reimplemented): `sens_sens()` in
//! `src/spicelib/analysis/cktsens.c`, the parameter generator of `cktsgen.c`,
//! the card grammar of `dot_sens()` in `src/spicelib/parser/inp2dot.c` and the
//! job parameters of `senssetp.c`. The reference binary registers `.sens` as
//! this finite-difference analysis (`SENSinfo`); the SPICE2 adjoint
//! `*sload.c`/`*sset.c` code belongs to the separate `WANT_SENSE2` `.sens2`
//! analysis, which the reference build does not compile.
//!
//! ```text
//! .sens v(n[,m])|i(vsrc) [filter ...] [dc | ac dec|oct|lin pts fstart fstop]
//! ```
//!
//! 1. The DC operating point is solved as for `.op`.
//! 2. **DC**: `Y` is the matrix the operating point's last Newton load left
//!    factored (C's `CKTmatrix` after `CKTop`, as `.tf` uses it); the devices
//!    are reloaded at that load's iterate (`CKTrhsOld`, with C's `DEVload`
//!    matrix, [`crate::devices::TrialState::with_c_jacobian`]) and `x` is the
//!    solve of that load (`CKTrhs`). **AC**: at every frequency the circuit
//!    is set up, temperature-updated and assembled again from its (by then
//!    perturbed and restored) records into a new matrix, `Y = A + j omega E`
//!    is ordered from scratch (`CKTsetup` + `NIacIter`) and `x` solved. The
//!    fresh state and the DC load at all-zero node voltages before `NIacIter`
//!    linearize every device at that reset state, not at the operating point
//!    ([`DeviceSensitivity::reset_ac_load`]); each K stamps its own
//!    `MUTacLoad` from its record ([`DeviceSensitivity::coupled_inductors`]).
//! 3. For every parameter, in C's `sgen` order, the one device is loaded with
//!    its current records, the result negated, the parameter set to
//!    `p + delta` with `delta = 1e-6 p` (`1e-6` when `p = 0`), the device
//!    loaded again on top, and the parameter set back to `p`: element by
//!    element `delta_Y = -dY0 + dY1` and `delta_I = -dI0 + dI1` in original
//!    stamp order with compensated accumulation. The sensitivity is `Y^-1 (delta_I - delta_Y x)`
//!    read at the output and divided by `delta`, with `delta_Y x` summed with compensation per
//!    row in the internal column order of C's `delta_Y` (see
//!    [`DeltaNumbering`]).
//!
//! The device side, including C's side effects of the in-place perturbation,
//! lives in [`crate::devices::sensitivity`]; see `docs/port/SENSITIVITY.md`.

use super::frequency::FrequencyPlan;
use super::output_probe::{self, Card, Output, Probe, parse_output};
use super::reduction::CompensatedSum;
use crate::analysis::linear::{number, unsupported};
use crate::analysis::results::{PlotFlags, Variable};
use crate::analysis::{AnalysisContext, AnalysisPlan, Plot};
use crate::devices::linear::LinearSystem;
use crate::devices::sensitivity::{
    AcReset, DeviceSensitivity, ParameterScope, RecordPair, SensitivityFamily, SensitivityLoad,
    SensitivityMode, SensitivityRecord,
};
use crate::devices::{AnalysisMode, Circuit, Device, IterationPhase, LoadRequest, TrialState};
use crate::maths::complex::ComplexMatrix;
use crate::maths::linear::solve_last_factors_transposed;
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Complex, NodeId, Real, SpiceError, SpiceResult, parse_spice_number};

/// `Sens_Delta` of `cktsens.c`: the relative perturbation.
pub(super) const RELATIVE_DELTA: Real = 1e-6;
/// `Sens_Abs_Delta` of `cktsens.c`: the perturbation of a zero parameter.
pub(super) const ABSOLUTE_DELTA: Real = 1e-6;
/// The plot title (`dot_sens()` names the job "Sensitivity Analysis").
pub(super) const PLOTNAME: &str = "Sensitivity Analysis";
/// `M_LOG2E`, which `count_steps()` divides by for octave sweeps.
const LOG2E: Real = std::f64::consts::LOG2_E;
/// The port's bound on the number of frequencies.
const MAX_POINTS: u32 = 100_000;

/// A differential estimate for a smooth physical scalar parameter.
///
/// Sampling returns the device's *projected residual difference* relative
/// to the unchanged baseline, not an independently Newton-converged output.
/// The latter would mix solver error into every local derivative.
#[derive(Debug, Clone, Copy)]
struct PhysicalSlope {
    derivative: Complex,
    /// Actual representable perturbation span, including binary64 rounding.
    span: Real,
    symmetric: bool,
    /// Separation of the two one-sided slopes, to distinguish a cusp.
    asymmetry: Real,
}

fn physical_slope(
    parameter: Real,
    trial: Real,
    sample: &mut impl FnMut(Real) -> SpiceResult<Option<Complex>>,
) -> SpiceResult<Option<PhysicalSlope>> {
    let at = |offset: Real,
              sample: &mut dyn FnMut(Real) -> SpiceResult<Option<Complex>>|
     -> SpiceResult<Option<(Real, Complex)>> {
        let point = parameter + offset;
        if !point.is_finite() || point.to_bits() == parameter.to_bits() {
            return Ok(None);
        }
        Ok(sample(point)?.map(|residual| (point - parameter, residual)))
    };
    let plus = at(trial, sample)?;
    let minus = at(-trial, sample)?;
    let (derivative, span, symmetric, asymmetry) = match (plus, minus) {
        (Some((h_plus, pos)), Some((h_minus, neg))) => (
            Complex::new(
                (pos.re - neg.re) / (h_plus - h_minus),
                (pos.im - neg.im) / (h_plus - h_minus),
            ),
            h_plus - h_minus,
            true,
            (pos.re / h_plus - neg.re / h_minus)
                .abs()
                .max((pos.im / h_plus - neg.im / h_minus).abs()),
        ),
        (Some((h, value)), _) | (_, Some((h, value))) => {
            (Complex::new(value.re / h, value.im / h), h.abs(), false, 0.)
        }
        _ => return Ok(None),
    };
    Ok(
        (derivative.re.is_finite() && derivative.im.is_finite()).then_some(PhysicalSlope {
            derivative,
            span,
            symmetric,
            asymmetry,
        }),
    )
}

/// Choose a representable step from candidate scales and compare two
/// independently sampled slopes. Symmetric second-order differences use
/// Richardson factor 4, while valid one-sided first-order differences use 2.
/// Return `None` when no physical neighborhood could be sampled, rather
/// than manufacturing a derivative at a nonsmooth/undefined boundary.
pub(in crate::analysis) fn conditioned_derivative(
    parameter: Real,
    mut sample: impl FnMut(Real) -> SpiceResult<Option<Complex>>,
) -> SpiceResult<Option<(Complex, Real)>> {
    let scale = if parameter == 0. { 1. } else { parameter.abs() };
    let mut best: Option<(Complex, Real, Real)> = None;
    for factor in [0.04, 0.02, 0.01, 0.005, 0.0025] {
        let h = scale * factor;
        if !h.is_finite() || h == 0. {
            continue;
        }
        let (Some(coarse), Some(fine)) = (
            physical_slope(parameter, h, &mut sample)?,
            physical_slope(parameter, h / 2., &mut sample)?,
        ) else {
            continue;
        };
        if coarse.symmetric != fine.symmetric {
            continue;
        }
        // Smooth curvature causes one-sided slopes to coalesce as h shrinks.
        // A cusp keeps their separation; a parameter jump increases it.
        // Do not infer a cusp from relative differences below the binary64
        // finite-difference resolution scale (sqrt(epsilon)): subtracting
        // two rounded residuals can make those differences grow as h shrinks.
        // This is a significance test, not a physical error guarantee.
        let slope_scale = coarse
            .derivative
            .re
            .abs()
            .max(coarse.derivative.im.abs())
            .max(fine.derivative.re.abs())
            .max(fine.derivative.im.abs());
        let resolution = Real::EPSILON.sqrt() * slope_scale;
        if fine.symmetric && fine.asymmetry > resolution && fine.asymmetry > coarse.asymmetry * 0.75
        {
            continue;
        }
        let factor = if fine.symmetric { 3. } else { 1. };
        let value = Complex::new(
            (fine.derivative.re - coarse.derivative.re).mul_add(1. / factor, fine.derivative.re),
            (fine.derivative.im - coarse.derivative.im).mul_add(1. / factor, fine.derivative.im),
        );
        let difference = (fine.derivative.re - coarse.derivative.re)
            .abs()
            .max((fine.derivative.im - coarse.derivative.im).abs());
        if !value.is_finite() {
            continue;
        }
        if best.as_ref().is_none_or(|(_, _, prev)| difference < *prev) {
            best = Some((value, fine.span, difference));
        }
    }
    Ok(best.and_then(|(value, step, discrepancy)| {
        let magnitude = value.re.abs().max(value.im.abs());
        // An inconsistent pair is ill-conditioned even when it is the best
        // of the available samples. Do not silently report a false gradient.
        (discrepancy <= 0.01 * magnitude).then_some((value, step))
    }))
}

/// The AC stepping of `senssetp.c` (`SENS_DECADE`, `SENS_OCTAVE`,
/// `SENS_LINEAR`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stepping {
    Decade,
    Octave,
    Linear,
}

/// An AC sweep as written: `ac kind pts fstart fstop`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Sweep {
    stepping: Stepping,
    steps: i32,
    start: Real,
    stop: Real,
}

/// A parsed `.sens` card.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct SensCard {
    output: Output,
    /// Name filters (`Sens_filter`), lowercased; empty for every parameter.
    filters: Vec<String>,
    /// `None` for DC.
    sweep: Option<Sweep>,
    /// Opt-in physically conditioned local scalar partials, not the C record
    /// replay. Historical C output and names remain unchanged by default.
    physical: bool,
}

impl SensCard {
    pub(super) fn grid(&self) -> SpiceResult<Option<FrequencyPlan>> {
        self.sweep.as_ref().map(frequencies).transpose()
    }
}

fn syntax(message: impl std::fmt::Display) -> SpiceError {
    unsupported(format!(
        ".sens {message}; expected `.sens v(n[,m])|i(vsrc) [filter ...] [dc | ac dec|oct|lin \
         pts fstart fstop]` (C: dot_sens in inp2dot.c)"
    ))
}

/// C's `INPgetValue(IF_INTEGER)`: `floor(value + 0.5)`.
fn integer(text: Option<&str>) -> SpiceResult<i32> {
    let value = text
        .and_then(parse_spice_number)
        .filter(|v| v.is_finite())
        .ok_or_else(|| syntax("expects an integer point count"))?;
    crate::primitives::c_integer(value)
        .and_then(|count| i32::try_from(count).ok())
        .ok_or_else(|| syntax("point count is out of range"))
}

/// Parses the arguments of a `.sens` card (`dot_sens()`).
pub(super) fn parse(arguments: &[String]) -> SpiceResult<SensCard> {
    let mut tokens = arguments.iter().map(String::as_str).peekable();
    let output = parse_output(&mut tokens, syntax)?;
    let mut filters = Vec::new();
    let mut sweep = None;
    let mut physical = false;
    while let Some(token) = tokens.next() {
        let lower = token.to_ascii_lowercase();
        match lower.as_str() {
            "physical" => physical = true,
            "dc" => break,
            "ac" => {
                let stepping = match tokens.next().map(str::to_ascii_lowercase).as_deref() {
                    Some("dec") => Stepping::Decade,
                    Some("oct") => Stepping::Octave,
                    Some("lin") => Stepping::Linear,
                    Some(other) => {
                        return Err(syntax(format!(
                            "AC stepping must be dec, oct or lin, not '{other}'"
                        )));
                    }
                    None => return Err(syntax("is missing the AC stepping")),
                };
                let steps = integer(tokens.next())?;
                let start = number(tokens.next(), ".sens start frequency")?;
                let stop = number(tokens.next(), ".sens stop frequency")?;
                sweep = Some(Sweep {
                    stepping,
                    steps,
                    start,
                    stop,
                });
                break;
            }
            "(" | ")" | "," => {
                return Err(syntax(format!(
                    "has an unexpected '{token}' among the filters"
                )));
            }
            _ if token.contains('=') => {
                return Err(syntax(format!("has an unexpected assignment '{token}'")));
            }
            _ => filters.push(lower),
        }
    }
    if let Some(extra) = tokens.next() {
        return Err(syntax(format!(
            "has an unexpected argument '{extra}' (C ignores it)"
        )));
    }
    Ok(SensCard {
        output,
        filters,
        sweep,
        physical,
    })
}

/// A direct differentiable observation; no plot-column name guessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::analysis) enum PhysicalProbe {
    /// Voltage between two explicit nodes (or a node and ground).
    Voltage { pos: String, neg: Option<String> },
    /// Current through a source with a findable matrix branch.
    Current { source: String },
}

/// Exactly one instance parameter from the existing typed sensitivity walk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::analysis) enum PhysicalParameter {
    /// One instance-owned device parameter.
    Instance { instance: String, keyword: String },
    /// One model parameter shared by every matching instance; the adjoint
    /// contributions from all model users are accumulated in C source order.
    Model { model: String, keyword: String },
}

impl PhysicalParameter {
    fn matches(
        &self,
        scope: ParameterScope,
        instance: &str,
        model: Option<&str>,
        keyword: &str,
    ) -> bool {
        match self {
            Self::Instance {
                instance: wanted,
                keyword: field,
            } => {
                scope == ParameterScope::Instance
                    && wanted.eq_ignore_ascii_case(instance)
                    && field.eq_ignore_ascii_case(keyword)
            }
            Self::Model {
                model: wanted,
                keyword: field,
            } => {
                scope == ParameterScope::Model
                    && model.is_some_and(|model| wanted.eq_ignore_ascii_case(model))
                    && field.eq_ignore_ascii_case(keyword)
            }
        }
    }

    /// Error provenance without guessing names from rawfile columns.
    pub(in crate::analysis) fn describe(&self) -> String {
        match self {
            Self::Instance { instance, keyword } => format!("{instance}:{keyword}"),
            Self::Model { model, keyword } => format!("model {model}:{keyword}"),
        }
    }
}

/// A partial from the existing adjoint sensitivity walk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(in crate::analysis) enum PhysicalPartial {
    Valid(Complex),
    NonSmooth,
    Missing,
}

/// Evaluate selected physical device partials at one operating point or AC
/// frequency. The existing driver reuses one primal factor and adjoint,
/// without a circuit simulation for every parameter. Selection is typed.
///
/// Missing and non-smooth physical parameters retain separate typed statuses.
/// Operational solver failures retain their original error.
///
/// # Errors
/// Underlying operating-point, AC, sensitivity or allocation failures.
pub(in crate::analysis) fn physical_partials(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    probe: &PhysicalProbe,
    frequency: Option<Real>,
    selected: &[PhysicalParameter],
) -> SpiceResult<Vec<PhysicalPartial>> {
    if selected.is_empty() {
        return Ok(Vec::new());
    }
    if frequency.is_some_and(|value| !(value.is_finite() && value > 0.)) {
        return Err(SpiceError::circuit(
            "physical AC derivative requires a positive finite frequency",
        ));
    }
    let model = context.model_context();
    let (listed, _) = members(circuit, &model)?;
    let catalogue = entries(&listed, frequency.is_some(), &[]);
    let mut filters = Vec::new();
    for entry in &catalogue {
        let member = &listed[entry.member];
        if selected.iter().any(|wanted| {
            wanted.matches(
                entry.scope,
                &member.name,
                member.sensitivity.model(),
                entry.keyword,
            )
        }) {
            let label = entry
                .name
                .as_ref()
                .ok_or_else(|| SpiceError::circuit("missing typed sensitivity label"))?;
            filters.push(label.clone());
        }
    }
    let result = vec![PhysicalPartial::Missing; selected.len()];
    if filters.is_empty() {
        return Ok(result);
    }
    // The original sensitivity filter grammar supports glob characters.
    // Build the actual filtered column order using the *same* typed entry
    // iterator the driver uses; a wildcard-bearing device label must never
    // shift the mapping of the selected derivatives.
    let ordered: Vec<_> = entries(&listed, frequency.is_some(), &filters)
        .into_iter()
        .filter(|entry| entry.name.is_some())
        .map(|entry| {
            (
                listed[entry.member].name.clone(),
                listed[entry.member].sensitivity.model().map(str::to_owned),
                entry.scope,
                entry.keyword.to_owned(),
            )
        })
        .collect();
    drop(listed);
    let card = SensCard {
        output: match probe {
            PhysicalProbe::Voltage { pos, neg } => Output::Voltage {
                pos: pos.clone(),
                neg: neg.clone(),
            },
            PhysicalProbe::Current { source } => Output::Current {
                source: source.clone(),
            },
        },
        filters,
        sweep: None,
        physical: true,
    };
    // Exactly one conditioned small-signal point, using the same admitted
    // frequency plan as the ordinary spectral analyses. No second sweep API.
    let point = frequency
        .map(|value| crate::analysis::frequency::FrequencyPlan::counted(value, 1., 1))
        .transpose()?;
    let (plot, _) = run(circuit, request, context, &card, point.as_ref(), false)?;
    let row = plot
        .rows()
        .next()
        .ok_or_else(|| SpiceError::circuit("sensitivity produced no physical derivative row"))?;
    let offset = usize::from(frequency.is_some());
    sum_physical_columns(row, &ordered, selected, offset)
}

/// Accumulate typed model/instance contributions using the original column
/// order and compensated sums, preserving non-smooth and missing statuses.
fn sum_physical_columns(
    row: &[Complex],
    ordered: &[(String, Option<String>, ParameterScope, String)],
    selected: &[PhysicalParameter],
    offset: usize,
) -> SpiceResult<Vec<PhysicalPartial>> {
    let mut result = vec![PhysicalPartial::Missing; selected.len()];
    let mut real = vec![CompensatedSum::default(); selected.len()];
    let mut imaginary = vec![CompensatedSum::default(); selected.len()];
    for (column, (instance, model, scope, keyword)) in ordered.iter().enumerate() {
        let value = row.get(column + offset).copied().ok_or_else(|| {
            SpiceError::circuit("typed sensitivity derivative column count changed")
        })?;
        for (slot, wanted) in selected.iter().enumerate() {
            if wanted.matches(*scope, instance, model.as_deref(), keyword) {
                if value.is_finite() {
                    if result[slot] != PhysicalPartial::NonSmooth {
                        real[slot].add(value.re);
                        imaginary[slot].add(value.im);
                        result[slot] = PhysicalPartial::Valid(Complex::ZERO);
                    }
                } else {
                    result[slot] = PhysicalPartial::NonSmooth;
                }
            }
        }
    }
    for (slot, partial) in result.iter_mut().enumerate() {
        if matches!(partial, PhysicalPartial::Valid(_)) {
            let value = Complex::new(real[slot].total(), imaginary[slot].total());
            *partial = if value.is_finite() {
                PhysicalPartial::Valid(value)
            } else {
                PhysicalPartial::NonSmooth
            };
        }
    }
    Ok(result)
}

/// The frequencies of an AC sweep, exactly as `count_steps()` counts them and
/// `inc_freq()` steps them in `cktsens.c`. Both carry C defects the port
/// reproduces: `inc_freq()` compares the step type with the noise/distortion
/// `LINEAR` constant (3) instead of `SENS_LINEAR`, so a `lin` sweep
/// *multiplies* by its step `(fstop - fstart)/pts`; and the octave count
/// divides `ln(fstop/fstart)` by `M_LOG2E` instead of multiplying.
#[expect(
    clippy::suboptimal_flops,
    reason = "count_steps truncates a discrete sweep length: 3 * log10(4.6060996915966586) + 1.01 is 3 separately rounded but 2.9999999999999996 fused"
)]
fn frequencies(sweep: &Sweep) -> SpiceResult<FrequencyPlan> {
    let steps = Real::from(sweep.steps.max(1));
    let (mut low, mut high) = (sweep.start, sweep.stop);
    let (count, factor) = match sweep.stepping {
        Stepping::Linear => (steps, (high - low) / steps),
        Stepping::Decade => {
            if low <= 0. {
                low = 1e-3;
            }
            if high <= low {
                high = 10. * low;
            }
            (
                (steps * (high / low).log10() + 1.01).trunc(),
                10f64.powf(1. / steps),
            )
        }
        Stepping::Octave => {
            if low <= 0. {
                low = 1e-3;
            }
            if high <= low {
                high = 2. * low;
            }
            (
                (steps * (high / low).ln() / LOG2E + 1.01).trunc(),
                2f64.powf(1. / steps),
            )
        }
    };
    if !count.is_finite() || count > Real::from(MAX_POINTS) {
        return Err(unsupported(format!(
            ".sens AC sweep has more than {MAX_POINTS} frequencies"
        )));
    }
    // Linear counts come from a validated integer; logarithmic counts are
    // truncated above. Avoid a saturating float-to-usize cast if those
    // invariants change while preserving C's minimum of one frequency.
    let count = if count <= 0. {
        1
    } else {
        crate::primitives::exact_nonnegative_integer::<usize>(count)
            .ok_or_else(|| unsupported(".sens AC sweep point count is not a valid integer"))?
    };
    FrequencyPlan::counted(sweep.start, factor, count)
}

/// `scan()` of `cktsens.c`: `*` matches any run, `?` any one character.
fn scan(filter: &[u8], name: &[u8]) -> bool {
    let (mut f, mut n) = (0, 0);
    while f < filter.len() && n < name.len() {
        if filter[f] == b'*' {
            if f + 1 == filter.len() {
                return true;
            }
            while n < name.len() && !scan(&filter[f + 1..], &name[n..]) {
                n += 1;
            }
            return n < name.len();
        }
        if filter[f] == name[n] || filter[f] == b'?' {
            f += 1;
            n += 1;
        } else {
            return false;
        }
    }
    n == name.len() && (f == filter.len() || (filter[f] == b'*' && f + 1 == filter.len()))
}

impl Probe {
    /// The unknowns the output reads.
    fn rows(&self) -> impl Iterator<Item = usize> {
        match *self {
            Self::Voltage { pos, neg } => [pos, neg],
            Self::Current { row } => [Some(row), None],
        }
        .into_iter()
        .flatten()
    }

    /// The exact linear functional `g` for the measured output `g^T x`.
    /// This is *not* a complex conjugate: sensitivity is a derivative with
    /// respect to a real parameter in an analytic complex linear system.
    fn dual(self, n: usize) -> Vector {
        let mut result = Vector::zeros(n);
        match self {
            Self::Voltage { pos, neg } => {
                if let Some(row) = pos {
                    result.as_mut_slice()[row] += 1.;
                }
                if let Some(row) = neg {
                    result.as_mut_slice()[row] -= 1.;
                }
            }
            Self::Current { row } => result.as_mut_slice()[row] = 1.,
        }
        result
    }

    /// Output-scale estimate before subtracting the two voltage terminals.
    fn resolution_scale(self, values: &[Complex]) -> Real {
        self.rows()
            .map(|row| {
                let value = values.get(row).copied().unwrap_or(Complex::ZERO);
                value.re.abs().max(value.im.abs())
            })
            .sum()
    }

    const fn absolute_resolution(self, options: &crate::analysis::newton::NewtonOptions) -> Real {
        match self {
            Self::Voltage { .. } => options.vntol,
            Self::Current { .. } => options.abstol,
        }
    }
}

/// One device in `sgen` order.
struct Member<'a> {
    index: usize,
    name: String,
    sensitivity: Box<dyn DeviceSensitivity + 'a>,
    /// The key of its model in [`Records::models`].
    model: usize,
}

/// One perturbable parameter, in output order.
#[derive(Debug, Clone)]
struct Entry {
    member: usize,
    scope: ParameterScope,
    keyword: &'static str,
    /// The output vector name, `None` when a filter excludes it (C skips the
    /// parameter entirely: it is neither perturbed nor restored).
    name: Option<String>,
}

/// The live records, which C mutates as the analysis proceeds.
#[derive(Clone)]
struct Records {
    models: Vec<SensitivityRecord>,
    instances: Vec<SensitivityRecord>,
}

impl Records {
    fn pair(
        &mut self,
        member: &Member<'_>,
        slot: usize,
    ) -> (&mut SensitivityRecord, &mut SensitivityRecord) {
        (&mut self.models[member.model], &mut self.instances[slot])
    }

    fn view(&self, member: &Member<'_>, slot: usize) -> RecordPair<'_> {
        RecordPair {
            model: &self.models[member.model],
            instance: &self.instances[slot],
        }
    }
}

/// A K coupling's inductors: their member slots and branch rows.
#[derive(Debug, Clone, Copy)]
struct Coupling {
    inductors: [usize; 2],
    rows: [usize; 2],
}

/// The coupling of every member (`None` for anything but K), resolved by
/// inductor name as `MUTsetup` does with `CKTfndDev`.
fn couplings(
    circuit: &Circuit,
    members: &[Member<'_>],
    ac: bool,
) -> SpiceResult<Vec<Option<Coupling>>> {
    let mut couplings = Vec::with_capacity(members.len());
    for member in members {
        if member.sensitivity.family() != SensitivityFamily::Mutual {
            couplings.push(None);
            continue;
        }
        let Some(names) = member.sensitivity.coupled_inductors() else {
            if !ac {
                couplings.push(None);
                continue;
            }
            return Err(unsupported(format!(
                ".sens ac with K coupling {} of more than two inductors: C's front end splits \
                 the card into one K instance per pair (inp_compat()), whose names and \
                 parameters the port does not reproduce",
                member.name
            )));
        };
        let mut inductors = [0; 2];
        let mut rows = [0; 2];
        for (k, name) in names.iter().enumerate() {
            let slot = members
                .iter()
                .position(|other| {
                    other.sensitivity.family() == SensitivityFamily::Inductor
                        && other.name.eq_ignore_ascii_case(name)
                })
                .ok_or_else(|| {
                    SpiceError::circuit(format!(
                        "{}: coupled inductor {name} has no sensitivity record",
                        member.name
                    ))
                })?;
            inductors[k] = slot;
            rows[k] = circuit
                .branch_rows(members[slot].index)
                .filter(|range| range.len() == 1)
                .ok_or_else(|| {
                    SpiceError::circuit(format!("{}: inductor {name} has no branch", member.name))
                })?
                .start;
        }
        couplings.push(Some(Coupling { inductors, rows }));
    }
    Ok(couplings)
}

/// The analyses before which C's instance fields hold small-signal or
/// transient values: `.op` (`dcop.c`), `.ac`, `.pz`, `.disto` and `.noise`
/// load with `MODEINITSMSIG`, `.tran` with `MODETRAN`.
const SMALL_SIGNAL_ANALYSES: &[crate::primitives::AnalysisKind] = &[
    crate::primitives::AnalysisKind::OperatingPoint,
    crate::primitives::AnalysisKind::Ac,
    crate::primitives::AnalysisKind::Transient,
    crate::primitives::AnalysisKind::PoleZero,
    crate::primitives::AnalysisKind::Distortion,
    crate::primitives::AnalysisKind::Noise,
];

/// Instance fields that escape `cktsens.c`'s reset. The last analysis that
/// writes them decides: a small-signal load's operating point or the actual
/// final transient load. A subsequent DC solve must not overwrite the latter.
#[derive(Default)]
struct EarlierAcState {
    small_signal: bool,
    transient: Option<Vec<Real>>,
}

impl EarlierAcState {
    fn reset<'a>(
        &'a self,
        circuit: &Circuit,
        member: &Member<'_>,
        state: RecordPair<'a>,
        omega: Real,
        bias: &'a Vector,
    ) -> SpiceResult<AcReset<'a>> {
        let transient = self
            .transient
            .as_ref()
            .map(|values| {
                circuit
                    .state_rows(member.index)
                    .and_then(|rows| values.get(rows))
                    .ok_or_else(|| {
                        SpiceError::circuit("AC sensitivity: stale transient state layout")
                    })
            })
            .transpose()?;
        Ok(AcReset {
            state,
            omega,
            small_signal: self.small_signal.then_some(bias),
            transient,
        })
    }
}

fn earlier_ac_state(circuit: &Circuit, members: &[Member<'_>]) -> SpiceResult<EarlierAcState> {
    let Some(last) = circuit
        .completed_analyses()
        .iter()
        .rev()
        .find(|kind| SMALL_SIGNAL_ANALYSES.contains(kind))
    else {
        return Ok(EarlierAcState::default());
    };
    if *last != crate::primitives::AnalysisKind::Transient {
        return Ok(EarlierAcState {
            small_signal: true,
            transient: None,
        });
    }
    let Some((member, acload)) = members
        .iter()
        .find_map(|member| Some((member, member.sensitivity.stale_ac_fields()?)))
    else {
        return Ok(EarlierAcState::default());
    };
    let transient = circuit
        .final_transient_state()
        .filter(|state| state.len() == circuit.state_len())
        .ok_or_else(|| {
            SpiceError::not_yet_ported(
                format!(
                    ".sens ac after .tran with {}: the final transient load's junction \
                 capacitances are missing; run the transient and inherit its circuit, \
                 not just its completed-analysis marker",
                    member.name
                ),
                format!("src/spicelib/analysis/cktsens.c and {acload}"),
            )
        })?;
    Ok(EarlierAcState {
        small_signal: false,
        transient: Some(transient),
    })
}

/// Mirrors a K member's inductors into its record, as C's `MUTind1`/`MUTind2`
/// pointers let `MUTtemp` read their current `INDinduct`.
fn mirror(couplings: &[Option<Coupling>], records: &mut Records, slot: usize) -> SpiceResult<()> {
    let Some(coupling) = couplings[slot] else {
        return Ok(());
    };
    let induct = [
        records.instances[coupling.inductors[0]].value("induct")?,
        records.instances[coupling.inductors[1]].value("induct")?,
    ];
    let instance = &mut records.instances[slot];
    instance.store("induct1", induct[0])?;
    instance.store("induct2", induct[1])?;
    instance.store_index("row1", coupling.rows[0])?;
    instance.store_index("row2", coupling.rows[1])
}

/// Every device's sensitivity description in C's `sgen` order: by device type
/// in `DEVices[]` order, then by model in reverse order of model creation (a
/// model is created when the first instance naming it is parsed; instances
/// without a model share the type's default model), then by instance in
/// reverse deck order.
fn members<'a>(
    circuit: &'a Circuit,
    context: &crate::devices::ModelContext,
) -> SpiceResult<(Vec<Member<'a>>, Records)> {
    let mut keys: Vec<(SensitivityFamily, Option<String>)> = Vec::new();
    let mut layouts = std::collections::BTreeMap::<
        SensitivityFamily,
        (SensitivityRecord, SensitivityRecord),
    >::new();
    let mut models: Vec<SensitivityRecord> = Vec::new();
    let mut ordered: Vec<(
        SensitivityFamily,
        usize,
        usize,
        Member<'a>,
        SensitivityRecord,
    )> = Vec::new();
    for (index, device) in circuit.devices().iter().enumerate() {
        let sensitivity = device.sensitivity(context)?;
        let (mut model_record, mut instance_record) = sensitivity.records()?;
        let family = sensitivity.family();
        match layouts.entry(family) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((model_record.clone(), instance_record.clone()));
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                let (model, instance) = entry.get();
                model_record.share_layout(model)?;
                instance_record.share_layout(instance)?;
            }
        }
        let key = (
            sensitivity.family(),
            sensitivity.model().map(str::to_ascii_lowercase),
        );
        let rank = keys
            .iter()
            .position(|known| *known == key)
            .unwrap_or_else(|| {
                keys.push(key.clone());
                models.push(model_record);
                keys.len() - 1
            });
        ordered.push((
            key.0,
            rank,
            index,
            Member {
                index,
                name: device.name().to_ascii_lowercase(),
                sensitivity,
                model: rank,
            },
            instance_record,
        ));
    }
    ordered.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then(b.2.cmp(&a.2)));
    let mut members = Vec::with_capacity(ordered.len());
    let mut instances = Vec::with_capacity(ordered.len());
    for (_, _, _, member, record) in ordered {
        members.push(member);
        instances.push(record);
    }
    Ok((members, Records { models, instances }))
}

/// The parameters C perturbs, with their output names (`sens_sens()`):
/// `<inst>:<kw>` for a model parameter, `<inst>` for the first principal
/// instance parameter, `<inst>_<kw>` otherwise.
fn entries(members: &[Member<'_>], ac: bool, filters: &[String]) -> Vec<Entry> {
    let mut entries = Vec::new();
    for (slot, member) in members.iter().enumerate() {
        let tables = [
            (ParameterScope::Model, member.sensitivity.model_parameters()),
            (
                ParameterScope::Instance,
                member.sensitivity.instance_parameters(),
            ),
        ];
        let mut principals = 0;
        for (scope, table) in tables {
            for parameter in table.iter().filter(|parameter| ac || !parameter.ac) {
                let name = match scope {
                    ParameterScope::Model => format!("{}:{}", member.name, parameter.keyword),
                    ParameterScope::Instance => {
                        if parameter.principal {
                            principals += 1;
                        }
                        if parameter.principal && principals == 1 {
                            member.name.clone()
                        } else {
                            format!("{}_{}", member.name, parameter.keyword)
                        }
                    }
                };
                let kept = filters.is_empty()
                    || filters
                        .iter()
                        .any(|filter| scan(filter.as_bytes(), name.as_bytes()));
                entries.push(Entry {
                    member: slot,
                    scope,
                    keyword: parameter.keyword,
                    name: kept.then_some(name),
                });
            }
        }
    }
    entries
}

/// Checked external matrix coordinates and C SPARSE first-touch identities
/// belong to different domains, even when their numeric values coincide.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExternalIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct InternalIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Position {
    row: ExternalIndex,
    col: ExternalIndex,
}

#[derive(Clone, Copy)]
struct NumberedStamp {
    position: Position,
    internal_col: InternalIndex,
    value: Complex,
    next: Option<usize>,
}

/// Unvalidated contribution at the assembly boundary.
#[derive(Clone, Copy)]
struct Stamp {
    row: usize,
    col: usize,
    value: Complex,
}

/// C's persistent `TRANSLATE` numbering of `delta_Y` (`spbuild.c`). Setup stamps
/// precede load stamps; a first-touch row is numbered before its column.
struct DeltaNumbering {
    internal: Vec<Option<InternalIndex>>,
    next: usize,
}

impl DeltaNumbering {
    fn new(n: usize) -> Self {
        Self {
            internal: vec![None; n],
            next: 0,
        }
    }

    fn translate(&mut self, row: usize, col: usize) -> SpiceResult<(Position, InternalIndex)> {
        if row >= self.internal.len() || col >= self.internal.len() {
            return Err(SpiceError::Numerical {
                context: ".sens".into(),
                message: format!("out-of-range perturbation coordinate ({row}, {col})"),
            });
        }
        let position = Position {
            row: ExternalIndex(row),
            col: ExternalIndex(col),
        };
        self.touch(position.row);
        let internal_col = self.touch(position.col);
        Ok((position, internal_col))
    }

    fn touch(&mut self, external: ExternalIndex) -> InternalIndex {
        *self.internal[external.0].get_or_insert_with(|| {
            let internal = InternalIndex(self.next);
            self.next += 1;
            internal
        })
    }
}

/// An append-only chain of contributions to one external row.
#[derive(Clone, Copy, Default)]
struct RowChain {
    first: Option<usize>,
    last: Option<usize>,
}

/// One folded column of the row currently being assembled.
struct RowEntry {
    col: ExternalIndex,
    internal_col: InternalIndex,
    real: CompensatedSum,
    imaginary: CompensatedSum,
}

#[cfg(test)]
impl RowEntry {
    const fn value(&self) -> Complex {
        Complex::new(self.real.total(), self.imaginary.total())
    }
}

/// Reusable sparse perturbation assembly.
///
/// Row chains retain chronological
/// stamp order. A direct-address column table folds each row without a map,
/// sorting only its distinct columns before multiplication. Storage is
/// linear in the matrix dimension and the largest single-device load.
struct DeltaMatrix {
    numbering: DeltaNumbering,
    rows: Vec<RowChain>,
    stamps: Vec<NumberedStamp>,
    column_slots: Vec<Option<usize>>,
    row_entries: Vec<RowEntry>,
}

impl DeltaMatrix {
    fn new(n: usize) -> Self {
        Self {
            numbering: DeltaNumbering::new(n),
            rows: vec![RowChain::default(); n],
            stamps: Vec::new(),
            column_slots: vec![None; n],
            row_entries: Vec::new(),
        }
    }

    fn begin(
        &mut self,
        setup: impl IntoIterator<Item = (usize, usize)>,
    ) -> SpiceResult<BaseAssembly<'_>> {
        // Only populated rows need resetting, including on an abandoned load.
        for stamp in &self.stamps {
            self.rows[stamp.position.row.0] = RowChain::default();
        }
        self.stamps.clear();
        for (row, col) in setup {
            self.numbering.translate(row, col)?;
        }
        Ok(BaseAssembly(self))
    }

    fn add(&mut self, stamps: impl IntoIterator<Item = Stamp>) -> SpiceResult<()> {
        for stamp in stamps {
            let (position, internal_col) = self.numbering.translate(stamp.row, stamp.col)?;
            let i = self.stamps.len();
            let row = &mut self.rows[position.row.0];
            if let Some(last) = row.last {
                self.stamps[last].next = Some(i);
            } else {
                row.first = Some(i);
            }
            row.last = Some(i);
            self.stamps.push(NumberedStamp {
                position,
                internal_col,
                value: stamp.value,
                next: None,
            });
        }
        Ok(())
    }

    fn fold_stamp(&mut self, stamp: NumberedStamp, zero: Complex) {
        let col = stamp.position.col;
        let i = *self.column_slots[col.0].get_or_insert_with(|| {
            let i = self.row_entries.len();
            self.row_entries.push(RowEntry {
                col,
                internal_col: stamp.internal_col,
                real: CompensatedSum::with_initial(zero.re),
                imaginary: CompensatedSum::with_initial(zero.im),
            });
            i
        });
        let entry = &mut self.row_entries[i];
        entry.real.add(stamp.value.re);
        entry.imaginary.add(stamp.value.im);
    }

    /// Fold the base load, negate it, then add the perturbed load. The
    /// boundary is the original stamp-stream index, not the sorted position.
    fn fold_row(&mut self, row: usize, base_count: usize) -> &[RowEntry] {
        self.row_entries.clear();
        let mut next = self.rows[row].first;
        while let Some(i) = next.filter(|i| *i < base_count) {
            let stamp = self.stamps[i];
            self.fold_stamp(stamp, Complex::ZERO);
            next = stamp.next;
        }
        for entry in &mut self.row_entries {
            // Negate both components of the accumulator. Rounding the base
            // sum before subtracting would lose a small derivative entirely.
            entry.real.negate();
            entry.imaginary.negate();
        }
        while let Some(i) = next {
            let stamp = self.stamps[i];
            self.fold_stamp(stamp, Complex::new(-0., -0.));
            next = stamp.next;
        }
        // Drop the direct-address mappings before sorting the distinct
        // columns. No clearing of the circuit-sized index is necessary.
        for entry in &self.row_entries {
            self.column_slots[entry.col.0] = None;
        }
        self.row_entries
            .sort_unstable_by_key(|entry| entry.internal_col);
        &self.row_entries
    }
}

/// The base contribution phase, after `DEVsetup` has numbered its elements.
struct BaseAssembly<'a>(&'a mut DeltaMatrix);

impl<'a> BaseAssembly<'a> {
    fn add(&mut self, stamps: impl IntoIterator<Item = Stamp>) -> SpiceResult<()> {
        self.0.add(stamps)
    }

    const fn negate(self) -> PerturbedAssembly<'a> {
        let base_count = self.0.stamps.len();
        PerturbedAssembly {
            matrix: self.0,
            base_count,
        }
    }
}

/// The perturbed contribution phase, after `SMPconstMult(delta_Y, -1)`.
struct PerturbedAssembly<'a> {
    matrix: &'a mut DeltaMatrix,
    base_count: usize,
}

impl PerturbedAssembly<'_> {
    fn add(&mut self, stamps: impl IntoIterator<Item = Stamp>) -> SpiceResult<()> {
        self.matrix.add(stamps)
    }

    /// `SMPmultiply`: visit every row, including empty ones, in external row
    /// order, with terms ordered by their persistent internal column number.
    fn visit_products(
        self,
        x: &[Complex],
        mut visit: impl FnMut(usize, [CompensatedSum; 2]),
    ) -> SpiceResult<()> {
        if x.len() != self.matrix.numbering.internal.len() {
            return Err(SpiceError::Numerical {
                context: ".sens".into(),
                message: "perturbation and solution dimensions disagree".into(),
            });
        }
        for row in 0..x.len() {
            let mut real = CompensatedSum::default();
            let mut imaginary = CompensatedSum::default();
            // Preserve the sparse visitation and perturbation semantics,
            // but compensate each component of the cancellation-prone dot.
            if self.matrix.rows[row].first.is_some() {
                for entry in self.matrix.fold_row(row, self.base_count) {
                    let value = x[entry.col.0];
                    entry.real.multiply_into(value.re, &mut real);
                    entry.imaginary.multiply_into(-value.im, &mut real);
                    entry.real.multiply_into(value.im, &mut imaginary);
                    entry.imaginary.multiply_into(value.re, &mut imaginary);
                }
            }
            visit(row, [real, imaginary]);
        }
        Ok(())
    }
}

fn real_stamps(matrix: &SparseMatrix) -> impl Iterator<Item = Stamp> + '_ {
    matrix.triplets().iter().map(|t| Stamp {
        row: t.row,
        col: t.col,
        value: Complex::real(t.value),
    })
}

/// The device a sensitivity load stamps.
fn stand_in<'d>(circuit: &'d Circuit, index: usize, load: &'d SensitivityLoad) -> &'d dyn Device {
    match load {
        SensitivityLoad::Original => &*circuit.devices()[index],
        SensitivityLoad::Replacement(device) => &**device,
    }
}

/// The connected block of every unknown, union-find over checked entries.
fn blocks(n: usize, entries: impl IntoIterator<Item = (usize, usize)>) -> SpiceResult<Vec<usize>> {
    const fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut parent: Vec<usize> = (0..n).collect();
    for (row, col) in entries {
        if row >= n || col >= n {
            return Err(SpiceError::Numerical {
                context: ".sens".into(),
                message: format!("out-of-range block coordinate ({row}, {col})"),
            });
        }
        let (a, b) = (root(&mut parent, row), root(&mut parent, col));
        if a != b {
            parent[a] = b;
        }
    }
    for x in 0..n {
        parent[x] = root(&mut parent, x);
    }
    Ok(parent)
}

/// `NIacIter` at one frequency: the reset system factored from scratch and
/// solved.
fn ac_state<'a>(
    circuit: &'a Circuit,
    system: &LinearSystem,
    omega: Real,
    model: crate::devices::ModelContext,
    zeros: &'a Vector,
    state: &'a crate::devices::StateSnapshot,
) -> SpiceResult<AcState<'a>> {
    // CKTunsetup/CKTsetup make a new circuit matrix (elements in setup
    // order) and NIacIter reorders it from scratch.
    let matrix = match circuit.matrix_pattern() {
        Some(pattern) => ComplexMatrix::from_operators_grouped_on(
            &pattern,
            &system.a,
            &system.e,
            &system.e_groups,
            omega,
        )?,
        None => None,
    };
    let matrix = match matrix {
        Some(matrix) => matrix,
        None => {
            ComplexMatrix::from_operators_grouped(&system.a, &system.e, &system.e_groups, omega)?
        }
    };
    let factors = matrix.factorize_new_matrix()?;
    let solution = factors.solve(&system.ac_rhs())?;
    Ok(AcState {
        omega,
        factors,
        solution,
        model,
        bias: zeros,
        state,
        circuit,
    })
}

/// Unknowns in a block containing a nonfinite AC entry. In the normal
/// finite case no block structure or dynamic stamp vector is allocated.
fn poisoned_rows(
    system: &LinearSystem,
    omega: Real,
) -> SpiceResult<std::collections::BTreeSet<usize>> {
    let n = system.a.rows();
    let entries = || {
        let dynamic =
            crate::maths::complex::iter_dynamic_terms(&system.e, &system.e_groups, omega)?;
        Ok::<_, SpiceError>(
            system
                .a
                .triplets()
                .iter()
                .map(|t| (t.row, t.col, t.value))
                .chain(dynamic),
        )
    };
    let mut bad = Vec::new();
    for (row, col, value) in entries()? {
        if row >= n || col >= n {
            return Err(SpiceError::Numerical {
                context: ".sens".into(),
                message: format!("out-of-range AC block coordinate ({row}, {col})"),
            });
        }
        if !value.is_finite() {
            bad.push(row);
        }
    }
    if bad.is_empty() {
        return Ok(std::collections::BTreeSet::new());
    }
    let blocks = blocks(n, entries()?.map(|(row, col, _)| (row, col)))?;
    let tainted: std::collections::BTreeSet<usize> = bad.iter().map(|&row| blocks[row]).collect();
    Ok((0..n)
        .filter(|&row| tainted.contains(&blocks[row]))
        .collect())
}

/// The DC state of the analysis: C's operating-point factors and vectors.
struct DcState<'a> {
    /// The circuit pattern whose last factors (`CKTop`'s final `NIiter`
    /// load, at `load_point`) are C's `Y`.
    pattern: std::sync::Arc<crate::maths::Pattern>,
    /// The connected block of every unknown in `Y`'s coupling.
    blocks: Vec<usize>,
    /// C's `CKTrhsOld`: the iterate of the last load, where the devices are
    /// loaded again.
    load_point: Vector,
    /// C's `CKTrhs` (`E`): the solve of that load, which `delta_Y`
    /// multiplies.
    solution: Vec<Complex>,
    model: crate::devices::ModelContext,
    history: crate::devices::StateHistory,
    trial: TrialState,
    circuit: &'a Circuit,
}

fn perturbation_error(label: &str, error: &SpiceError) -> SpiceError {
    SpiceError::Numerical {
        context: format!(".sens {label}: {error}"),
        message: "the perturbed load could not be evaluated".to_owned(),
    }
}

/// Reusable residual work for one parameter at one fixed operating point.
/// The objective's adjoint is solved once, rather than one primal system per
/// parameter. Retain only block masks for nonfinite local physics.
struct DcAssembly {
    matrix: SparseMatrix,
    rhs: Vector,
    tainted: Vec<bool>,
}

impl DcAssembly {
    fn new(n: usize) -> Self {
        Self {
            matrix: SparseMatrix::new(n, n),
            rhs: Vector::zeros(n),
            tainted: Vec::new(),
        }
    }
}

/// Compensated scalar product, including the low part of every finite
/// multiplication. The residual stays undivided until the final projection.
fn projected_real(adjoint: &[Real], residual: &[Real]) -> Real {
    let mut sum = CompensatedSum::default();
    for (&weight, &value) in adjoint.iter().zip(residual) {
        CompensatedSum::with_initial(weight).multiply_into(value, &mut sum);
    }
    sum.total()
}

/// Complex algebraic transpose projection, without conjugation.
fn projected_complex(adjoint: &[Complex], residual: &[Complex]) -> Complex {
    let mut real = CompensatedSum::default();
    let mut imaginary = CompensatedSum::default();
    for (&weight, &value) in adjoint.iter().zip(residual) {
        CompensatedSum::with_initial(weight.re).multiply_into(value.re, &mut real);
        CompensatedSum::with_initial(-weight.im).multiply_into(value.im, &mut real);
        CompensatedSum::with_initial(weight.re).multiply_into(value.im, &mut imaginary);
        CompensatedSum::with_initial(weight.im).multiply_into(value.re, &mut imaginary);
    }
    Complex::new(real.total(), imaginary.total())
}

/// Sanitizes the factored solve's RHS while retaining exactly the blocks that
/// C would poison. Preserve finite and signed-zero entries.
fn mask_nonfinite(rhs: &mut Vector, blocks: &[usize], tainted: &mut Vec<bool>) {
    tainted.clear();
    let n = rhs.len();
    for (row, value) in rhs.as_mut_slice().iter_mut().enumerate() {
        if !value.is_finite() {
            if tainted.is_empty() {
                tainted.resize(n, false);
            }
            tainted[blocks[row]] = true;
            *value = 0.;
        }
    }
}

fn probe_tainted(probe: Probe, blocks: &[usize], tainted: &[bool]) -> bool {
    !tainted.is_empty() && probe.rows().any(|row| tainted[blocks[row]])
}

impl DcState<'_> {
    /// One `DEVload` of the device into `matrix`/`rhs`.
    fn load(
        &self,
        index: usize,
        load: &SensitivityLoad,
        matrix: &mut SparseMatrix,
        rhs: &mut Vector,
    ) -> SpiceResult<()> {
        let mut trial = self.trial.clone();
        self.circuit.load_device(
            index,
            stand_in(self.circuit, index, load),
            &LoadRequest {
                mode: AnalysisMode::OperatingPoint,
                solution: &self.load_point,
                model_context: &self.model,
                integration: None,
                history: &self.history,
                forcing: None,
            },
            matrix,
            rhs,
            &mut trial,
        )
    }

    /// `Y^-1 (delta_I - delta_Y E)` of a base and a perturbed load, with
    /// `delta_I` and `delta_Y` accumulated as `sens_sens()` does.
    fn residual(
        &self,
        mut assembly: BaseAssembly<'_>,
        scratch: &mut DcAssembly,
        index: usize,
        base: &SensitivityLoad,
        perturbed: &SensitivityLoad,
    ) -> SpiceResult<()> {
        scratch.matrix.clear();
        scratch.rhs.clear();
        self.load(index, base, &mut scratch.matrix, &mut scratch.rhs)?;
        assembly.add(real_stamps(&scratch.matrix))?;
        let mut assembly = assembly.negate();
        // delta_I *= -1, then the perturbed load adds to it in place.
        for value in scratch.rhs.as_mut_slice() {
            *value *= -1.;
        }
        scratch.matrix.clear();
        self.load(index, perturbed, &mut scratch.matrix, &mut scratch.rhs)?;
        assembly.add(real_stamps(&scratch.matrix))?;
        assembly.visit_products(&self.solution, |row, [mut real, _]| {
            // Round delta_I - delta_Y E only after the residual is assembled,
            // not after the potentially cancelling matrix-vector product.
            let value = &mut scratch.rhs.as_mut_slice()[row];
            real.negate();
            real.add(*value);
            *value = real.total();
        })?;
        // Nonfinite loads poison the entire connected block, as in C.
        // The mask remains unallocated on the normal path and reuses capacity.
        mask_nonfinite(&mut scratch.rhs, &self.blocks, &mut scratch.tainted);
        Ok(())
    }

    fn project(&self, scratch: &DcAssembly, probe: Probe, adjoint: &[Real]) -> Real {
        if probe_tainted(probe, &self.blocks, &scratch.tainted) {
            return Real::NAN;
        }
        projected_real(adjoint, scratch.rhs.as_slice())
    }
}

/// The AC state at one frequency.
struct AcState<'a> {
    omega: Real,
    factors: crate::maths::complex::ComplexLu,
    solution: Vec<Complex>,
    model: crate::devices::ModelContext,
    bias: &'a Vector,
    state: &'a crate::devices::StateSnapshot,
    circuit: &'a Circuit,
}

struct AcAssembly {
    system: LinearSystem,
    rhs: Vec<Complex>,
    perturbed_rhs: Vec<Complex>,
}

impl AcAssembly {
    const fn new(n: usize) -> Self {
        Self {
            system: LinearSystem::new(n),
            rhs: Vec::new(),
            perturbed_rhs: Vec::new(),
        }
    }
}

impl AcState<'_> {
    fn assemble(
        &self,
        index: usize,
        load: &SensitivityLoad,
        system: &mut LinearSystem,
    ) -> SpiceResult<()> {
        system.clear();
        self.circuit.assemble_small_signal_device(
            index,
            stand_in(self.circuit, index, load),
            &self.model,
            self.bias,
            Some(self.state),
            system,
        )
    }

    /// Real contributions, then frequency-scaled dynamic contributions,
    /// with each group's original arithmetic and stamp order.
    fn stamps<'s>(
        &self,
        system: &'s LinearSystem,
    ) -> SpiceResult<impl Iterator<Item = Stamp> + use<'s>> {
        Ok(real_stamps(&system.a).chain(
            crate::maths::complex::iter_dynamic_terms(&system.e, &system.e_groups, self.omega)?
                .map(|(row, col, value)| Stamp {
                    row,
                    col,
                    value: Complex::new(0., value),
                }),
        ))
    }

    fn residual(
        &self,
        mut assembly: BaseAssembly<'_>,
        scratch: &mut AcAssembly,
        index: usize,
        base: &SensitivityLoad,
        perturbed: &SensitivityLoad,
    ) -> SpiceResult<()> {
        self.assemble(index, base, &mut scratch.system)?;
        scratch.system.ac_rhs_into(&mut scratch.rhs);
        assembly.add(self.stamps(&scratch.system)?)?;
        let mut assembly = assembly.negate();
        self.assemble(index, perturbed, &mut scratch.system)?;
        scratch.system.ac_rhs_into(&mut scratch.perturbed_rhs);
        // -(base) + perturbed, part by part.
        for (base, perturbed) in scratch.rhs.iter_mut().zip(&scratch.perturbed_rhs) {
            *base = Complex::new(-base.re + perturbed.re, -base.im + perturbed.im);
        }
        assembly.add(self.stamps(&scratch.system)?)?;
        assembly.visit_products(&self.solution, |row, [mut real, mut imaginary]| {
            let value = &mut scratch.rhs[row];
            real.negate();
            imaginary.negate();
            real.add(value.re);
            imaginary.add(value.im);
            *value = Complex::new(real.total(), imaginary.total());
        })?;
        if scratch.rhs.iter().any(|value| !value.is_finite()) {
            return Err(SpiceError::Numerical {
                context: "complex solve".to_owned(),
                message: "invalid RHS".to_owned(),
            });
        }
        Ok(())
    }

    fn project(scratch: &AcAssembly, adjoint: &[Complex]) -> Complex {
        projected_complex(adjoint, &scratch.rhs)
    }
}

/// An estimated output-resolution floor divided by the actual finite-
/// difference denominator. This is a conformance diagnostic, not a bound on
/// the true derivative or a change to the sensitivity algorithm.
struct Resolution {
    plot: Plot,
    absolute: Real,
}

impl Resolution {
    fn row(&self, probe: Probe, solution: &[Complex]) -> (Real, Vec<Complex>) {
        // Allow several rounding operations, including independent reference
        // arithmetic. No claim of a certified forward-error bound.
        let roundoff = 32. * Real::EPSILON * probe.resolution_scale(solution);
        (
            self.absolute.max(roundoff),
            Vec::with_capacity(self.plot.variable_count()),
        )
    }
}

/// Runs `.sens` on `circuit`; see the module documentation.
///
/// # Errors
/// Malformed cards, unknown outputs, devices whose sensitivity is not ported,
/// operating-point failures and singular or nonfinite solves.
pub(super) fn run(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    card: &SensCard,
    grid: Option<&FrequencyPlan>,
    estimate_resolution: bool,
) -> SpiceResult<(Plot, Option<Plot>)> {
    let mut settings = request.dc_settings().clone();
    let ac = grid.is_some();
    settings.newton.retain_linearization = !ac;
    circuit.finalize()?;
    let circuit: &Circuit = circuit;
    let probe = output_probe::bind(circuit, &card.output, Card::Sens)?.probe;
    let model = context.model_context();

    let mut workspace = ParameterWalk::new(circuit, &model, ac, &card.filters, card.physical)?;
    let earlier = if ac {
        earlier_ac_state(circuit, &workspace.members)?
    } else {
        EarlierAcState::default()
    };
    // The operating point, exactly as `.op` solves it (`CKTop`).
    let solved = crate::analysis::bias::OperatingPointStart::resolve(circuit, request)?.solve(
        &circuit.prepare_run(),
        &model,
        &settings,
    )?;

    let mut plot = workspace.plot(ac);
    if estimate_resolution {
        workspace.resolution = Some(Resolution {
            plot: workspace.plot(ac),
            // AC is a linear small-signal solve, not a Newton convergence test.
            absolute: if ac || !circuit.devices().iter().any(|device| device.is_nonlinear()) {
                0.
            } else {
                probe.absolute_resolution(&settings.newton)
            },
        });
    }
    if let Some(grid) = grid {
        run_ac(
            &mut workspace,
            &earlier,
            &solved.values,
            model,
            probe,
            grid.values(),
            &mut plot,
        )?;
    } else {
        run_dc(
            &mut workspace,
            solved,
            model,
            settings.newton.limiting.is_device(),
            probe,
            &mut plot,
        )?;
    }
    Ok((plot, workspace.resolution.map(|resolution| resolution.plot)))
}

/// One analysis's parameter sequence, owning all live mutable C records and
/// the persistent perturbation-matrix numbering. AC resets and individual
/// perturbations share this state; DC and AC never duplicate the walk.
struct ParameterWalk<'a> {
    circuit: &'a Circuit,
    members: Vec<Member<'a>>,
    records: Records,
    entries: Vec<Entry>,
    couplings: Vec<Option<Coupling>>,
    delta: DeltaMatrix,
    resolution: Option<Resolution>,
    physical: bool,
}

impl<'a> ParameterWalk<'a> {
    fn new(
        circuit: &'a Circuit,
        model: &crate::devices::ModelContext,
        ac: bool,
        filters: &[String],
        physical: bool,
    ) -> SpiceResult<Self> {
        let (members, records) = members(circuit, model)?;
        let couplings = couplings(circuit, &members, ac)?;
        let entries = entries(&members, ac, filters);
        if entries.iter().all(|entry| entry.name.is_none()) {
            return Err(unsupported(
                ".sens perturbs no parameter (every name is filtered out, or the circuit has no perturbable parameter); C writes no plot in that case",
            ));
        }
        Ok(Self {
            circuit,
            members,
            records,
            entries,
            couplings,
            delta: DeltaMatrix::new(circuit.unknown_count()),
            resolution: None,
            physical,
        })
    }

    fn plot(&self, ac: bool) -> Plot {
        let mut plot = Plot::builder(
            "sens1",
            PLOTNAME,
            if ac {
                PlotFlags::Complex
            } else {
                PlotFlags::Real
            },
        );
        if ac {
            plot.push_variable(Variable::complex("frequency", "frequency"));
        }
        for name in self.entries.iter().filter_map(|entry| entry.name.as_ref()) {
            let name = format!("v({name})");
            plot.push_variable(if ac {
                Variable::complex(name, "voltage")
            } else {
                Variable::new(name, "voltage")
            });
        }
        plot.build()
    }

    /// `CKTunsetup`, then `CKTsetup` and `CKTtemp` in sgen order. The reset-load
    /// snapshot survives the following in-place parameter perturbations.
    fn reset_ac(&mut self) -> SpiceResult<Records> {
        for (slot, member) in self.members.iter().enumerate() {
            let (model, instance) = self.records.pair(member, slot);
            member.sensitivity.unsetup(model, instance)?;
        }
        for (slot, member) in self.members.iter().enumerate() {
            let (model, instance) = self.records.pair(member, slot);
            member.sensitivity.setup(model, instance)?;
            mirror(&self.couplings, &mut self.records, slot)?;
            let (model, instance) = self.records.pair(member, slot);
            member.sensitivity.temperature(model, instance)?;
        }
        Ok(self.records.clone())
    }
}

/// The state reset for an AC frequency: the load point and state vectors are
/// zero, while the preceding analysis's surviving fields belong to `earlier`.
struct AcSweep<'a> {
    earlier: &'a EarlierAcState,
    bias: &'a Vector,
    model: crate::devices::ModelContext,
    zeros: Vector,
    state: crate::devices::StateSnapshot,
}

impl AcSweep<'_> {
    fn load(
        &self,
        circuit: &Circuit,
        member: &Member<'_>,
        reset: RecordPair<'_>,
        setup: RecordPair<'_>,
        current: RecordPair<'_>,
        omega: Real,
    ) -> SpiceResult<SensitivityLoad> {
        member.sensitivity.reset_ac_load(
            setup,
            self.earlier
                .reset(circuit, member, reset, omega, self.bias)?,
            current,
        )
    }

    fn assemble(
        &self,
        walk: &ParameterWalk<'_>,
        reset: &Records,
        omega: Real,
        system: &mut LinearSystem,
    ) -> SpiceResult<()> {
        system.clear();
        system.a.set_pivot_tolerances(self.model.pivot);
        for (slot, member) in walk.members.iter().enumerate() {
            let pair = reset.view(member, slot);
            let load = self.load(walk.circuit, member, pair, pair, pair, omega)?;
            walk.circuit.assemble_small_signal_device(
                member.index,
                stand_in(walk.circuit, member.index, &load),
                &self.model,
                &self.zeros,
                Some(&self.state),
                system,
            )?;
        }
        Ok(())
    }
}

/// One AC frequency's already factored physical and historical parameter
/// projections. The frequency loop only owns setup and plot lifecycle.
struct AcRow<'a, 's, 'c> {
    sweep: &'a AcSweep<'s>,
    reset: &'a Records,
    at: Option<&'a AcState<'c>>,
    adjoint: Option<&'a [Complex]>,
    omega: Real,
}

impl AcRow<'_, '_, '_> {
    fn evaluate(
        &self,
        walk: &mut ParameterWalk<'_>,
        scratch: &mut AcAssembly,
        point: &mut Vec<Complex>,
        resolution: &mut Option<(Real, Vec<Complex>)>,
    ) -> SpiceResult<()> {
        let Self {
            sweep,
            reset,
            at,
            adjoint,
            omega,
        } = *self;
        let circuit = walk.circuit;
        if walk.physical {
            let responses = walk.sweep_physical(
                |slot, member, setup, current| {
                    sweep.load(
                        circuit,
                        member,
                        reset.view(member, slot),
                        setup,
                        current,
                        omega,
                    )
                },
                |assembly, index, base, perturbed, _, label| {
                    let Some(at) = &at else {
                        return Ok(Complex::new(Real::NAN, Real::NAN));
                    };
                    at.residual(assembly, scratch, index, base, perturbed)
                        .map_err(|error| perturbation_error(label, &error))?;
                    Ok(AcState::project(
                        scratch,
                        adjoint
                            .ok_or_else(|| SpiceError::circuit("missing AC sensitivity adjoint"))?,
                    ))
                },
            )?;
            for (derivative, span) in responses {
                point.push(derivative);
                if let Some((floor, row)) = resolution.as_mut() {
                    row.push(Complex::real(*floor / span.abs()));
                }
            }
        } else {
            let walked = walk.sweep(
                |slot, member, setup, current| {
                    sweep.load(
                        circuit,
                        member,
                        reset.view(member, slot),
                        setup,
                        current,
                        omega,
                    )
                },
                |assembly, index, base, perturbed, delta, label| {
                    if let Some((floor, row)) = resolution.as_mut() {
                        row.push(Complex::real(*floor / delta.abs()));
                    }
                    let Some(at) = &at else {
                        point.push(Complex::new(Real::NAN, Real::NAN));
                        return Ok(());
                    };
                    at.residual(assembly, scratch, index, base, perturbed)
                        .map_err(|error| perturbation_error(label, &error))?;
                    let value = AcState::project(
                        scratch,
                        adjoint
                            .ok_or_else(|| SpiceError::circuit("missing AC sensitivity adjoint"))?,
                    );
                    point.push(Complex::new(value.re / delta, value.im / delta));
                    Ok(())
                },
            );
            walked?;
        }
        Ok(())
    }
}

fn run_ac(
    walk: &mut ParameterWalk<'_>,
    earlier: &EarlierAcState,
    bias: &Vector,
    model: crate::devices::ModelContext,
    probe: Probe,
    grid: impl Iterator<Item = Real>,
    plot: &mut Plot,
) -> SpiceResult<()> {
    let circuit = walk.circuit;
    let sweep = AcSweep {
        earlier,
        bias,
        model,
        zeros: Vector::zeros(circuit.unknown_count()),
        state: circuit.bind_state(vec![0.; circuit.state_len()])?,
    };
    let mut scratch = AcAssembly::new(circuit.unknown_count());
    let mut system = LinearSystem::new(circuit.unknown_count());
    for frequency in grid {
        let omega = 2. * std::f64::consts::PI * frequency;
        let reset = walk.reset_ac()?;
        sweep.assemble(walk, &reset, omega, &mut system)?;
        // A device whose present records C loads into a non-finite
        // matrix (BSIM4's TNOM drifting by 273.15 K per frequency, see
        // `cktsens.c` and `b4mask.c`) poisons NIacIter's solve: every
        // unknown of a block holding such an entry is NaN, so is every
        // column read from it, and the walk still perturbs and restores
        // each parameter. A poisoned block the output does not read
        // would leave C's other blocks finite; that is refused.
        let poisoned = poisoned_rows(&system, omega)?;
        let probe_poisoned = probe.rows().any(|row| poisoned.contains(&row));
        if !poisoned.is_empty() && !probe_poisoned {
            return Err(SpiceError::not_yet_ported(
                format!(
                    ".sens ac at {frequency:e} Hz: a non-finite device load in a block \
                     the output does not read"
                ),
                "src/spicelib/analysis/cktsens.c (NIacIter on a partly non-finite matrix)",
            ));
        }
        let at = if probe_poisoned {
            None
        } else {
            Some(ac_state(
                circuit,
                &system,
                omega,
                model,
                &sweep.zeros,
                &sweep.state,
            )?)
        };
        let adjoint = at
            .as_ref()
            .map(|at| {
                let rhs: Vec<_> = probe
                    .dual(circuit.unknown_count())
                    .as_slice()
                    .iter()
                    .map(|&value| Complex::real(value))
                    .collect();
                at.factors.solve_transposed(&rhs)
            })
            .transpose()?;
        let mut point = Vec::with_capacity(plot.variables().len());
        point.push(Complex::real(frequency));
        let mut resolution = walk.resolution.as_ref().map(|resolution| {
            let (floor, mut row) = resolution.row(
                probe,
                at.as_ref()
                    .map_or(&[][..], |state| state.solution.as_slice()),
            );
            row.push(Complex::ZERO); // frequency is not a derivative
            (floor, row)
        });
        AcRow {
            sweep: &sweep,
            reset: &reset,
            at: at.as_ref(),
            adjoint: adjoint.as_deref(),
            omega,
        }
        .evaluate(walk, &mut scratch, &mut point, &mut resolution)?;
        plot.push_point(&point)?;
        if let (Some(resolution), Some((_, row))) = (&mut walk.resolution, resolution) {
            resolution.plot.push_point(&row)?;
        }
    }
    Ok(())
}

fn run_dc(
    walk: &mut ParameterWalk<'_>,
    solved: crate::analysis::newton::NewtonSolution<TrialState>,
    model: crate::devices::ModelContext,
    device_limiting: bool,
    probe: Probe,
    plot: &mut Plot,
) -> SpiceResult<()> {
    let circuit = walk.circuit;
    let n = circuit.unknown_count();
    // `Y` is the matrix CKTop's last NIiter load left factored (its
    // load at CKTrhsOld), as for `.tf`; the devices are reloaded at that
    // iterate and `delta_Y` multiplies its solve, CKTrhs.
    let history = circuit.state_history()?;
    let trial = history
        .trial_in(IterationPhase::Float, Some(&solved.trial))?
        .with_device_limiting(device_limiting)
        .with_c_jacobian(true);
    let mut jacobian = SparseMatrix::new(n, n);
    circuit.load(
        &LoadRequest {
            mode: AnalysisMode::OperatingPoint,
            solution: &solved.values,
            model_context: &model,
            integration: None,
            history: &history,
            forcing: None,
        },
        &mut jacobian,
        &mut Vector::zeros(n),
        &mut trial.clone(),
    )?;
    let pattern = circuit
        .matrix_pattern()
        .ok_or_else(|| SpiceError::Numerical {
            context: ".sens".to_owned(),
            message: "the operating point left no factored circuit matrix".to_owned(),
        })?;
    let at = DcState {
        pattern,
        blocks: blocks(n, jacobian.triplets().iter().map(|t| (t.row, t.col)))?,
        load_point: solved.values,
        solution: solved
            .next
            .as_slice()
            .iter()
            .map(|&v| Complex::real(v))
            .collect(),
        model,
        history,
        trial,
        circuit,
    };
    let adjoint = solve_last_factors_transposed(&at.pattern, &probe.dual(n))?.ok_or_else(|| {
        SpiceError::Numerical {
            context: ".sens".to_owned(),
            message: "the operating point left no factored circuit matrix".to_owned(),
        }
    })?;
    let mut point = Vec::with_capacity(plot.variables().len());
    let mut scratch = DcAssembly::new(n);
    let mut resolution = walk
        .resolution
        .as_ref()
        .map(|resolution| resolution.row(probe, &at.solution));
    if walk.physical {
        let responses = walk.sweep_physical(
            |_, member, setup, current| {
                member.sensitivity.load(setup, current, SensitivityMode::Dc)
            },
            |assembly, index, base, perturbed, _, label| {
                at.residual(assembly, &mut scratch, index, base, perturbed)
                    .map_err(|error| perturbation_error(label, &error))?;
                Ok(Complex::real(at.project(
                    &scratch,
                    probe,
                    adjoint.as_slice(),
                )))
            },
        )?;
        for (derivative, span) in responses {
            point.push(derivative);
            if let Some((floor, row)) = &mut resolution {
                row.push(Complex::real(*floor / span.abs()));
            }
        }
    } else {
        let walked = walk.sweep(
            |_, member, setup, current| {
                member.sensitivity.load(setup, current, SensitivityMode::Dc)
            },
            |assembly, index, base, perturbed, delta, label| {
                if let Some((floor, row)) = &mut resolution {
                    row.push(Complex::real(*floor / delta.abs()));
                }
                at.residual(assembly, &mut scratch, index, base, perturbed)
                    .map_err(|error| perturbation_error(label, &error))?;
                point.push(Complex::real(
                    at.project(&scratch, probe, adjoint.as_slice()) / delta,
                ));
                Ok(())
            },
        );
        walked?;
    }
    plot.push_point(&point)?;
    if let (Some(resolution), Some((_, row))) = (&mut walk.resolution, resolution) {
        resolution.plot.push_point(&row)?;
    }
    Ok(())
}

/// The elements a member's `DEVsetup` makes on `delta_Y`: its own list for
/// the present records ([`DeviceSensitivity::setup_elements`]) or the
/// circuit's setup elements.
fn setup_elements(
    circuit: &Circuit,
    index: usize,
    elements: Option<Vec<(NodeId, NodeId)>>,
) -> SpiceResult<Vec<(usize, usize)>> {
    elements.map_or_else(
        || circuit.setup_elements(index),
        |pairs| {
            let row_of = |node| {
                if node == NodeId::GROUND {
                    Ok(None)
                } else {
                    circuit.unknowns().node_row(node).map(Some).ok_or_else(|| {
                        SpiceError::Numerical {
                            context: ".sens".into(),
                            message: format!("unmapped setup node {node}"),
                        }
                    })
                }
            };
            let mut mapped = Vec::with_capacity(pairs.len());
            for (row, col) in pairs {
                if let (Some(row), Some(col)) = (row_of(row)?, row_of(col)?) {
                    mapped.push((row, col));
                }
            }
            Ok(mapped)
        },
    )
}

impl ParameterWalk<'_> {
    /// Physical smooth-parameter partials at one fixed operating point.
    ///
    /// Each trial has independent record storage, so the model's C-style
    /// `given` transitions and previous perturbations cannot leak into a
    /// physical derivative. The same physical load, residual assembler and
    /// adjoint projection are used as the historical walk; only the numeric
    /// parameter policy changes. Unrepresentable/nonsmooth local variations
    /// emit NaN without silently switching to C's finite record transition.
    fn sweep_physical(
        &mut self,
        load: impl Fn(
            usize,
            &Member<'_>,
            RecordPair<'_>,
            RecordPair<'_>,
        ) -> SpiceResult<SensitivityLoad>,
        mut measure: impl FnMut(
            BaseAssembly<'_>,
            usize,
            &SensitivityLoad,
            &SensitivityLoad,
            Real,
            &str,
        ) -> SpiceResult<Complex>,
    ) -> SpiceResult<Vec<(Complex, Real)>> {
        let mut response = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            let Some(label) = entry.name.as_deref() else {
                continue;
            };
            let member = &self.members[entry.member];
            let device = &member.sensitivity;
            mirror(&self.couplings, &mut self.records, entry.member)?;
            let (model, instance) = self.records.pair(member, entry.member);
            let parameter =
                device.ask(entry.scope, entry.keyword, RecordPair { model, instance })?;
            // The candidate is isolated from the historical mutable record
            // sequence. A new candidate never inherits a previous setter.
            let (mut model, mut instance) = (model.clone(), instance.clone());
            device.setup(&mut model, &mut instance)?;
            device.temperature(&mut model, &mut instance)?;
            let setup = RecordPair {
                model: &model,
                instance: &instance,
            };
            let base_elements = device.setup_elements(setup)?;
            let elements = setup_elements(self.circuit, member.index, base_elements.clone())?;
            let base = load(entry.member, member, setup, setup)?;
            let result = conditioned_derivative(parameter, |point| {
                let (mut candidate_model, mut candidate_instance) =
                    (model.clone(), instance.clone());
                if device
                    .set(
                        entry.scope,
                        entry.keyword,
                        point,
                        &mut candidate_model,
                        &mut candidate_instance,
                    )
                    .is_err()
                {
                    return Ok(None);
                }
                if device
                    .setup(&mut candidate_model, &mut candidate_instance)
                    .is_err()
                    || device
                        .temperature(&mut candidate_model, &mut candidate_instance)
                        .is_err()
                {
                    return Ok(None);
                }
                let candidate = RecordPair {
                    model: &candidate_model,
                    instance: &candidate_instance,
                };
                // A changed setup element pattern is a topology transition,
                // not a differentiable scalar device contribution.
                if device.setup_elements(candidate)? != base_elements {
                    return Ok(None);
                }
                let perturbed = load(entry.member, member, setup, candidate)?;
                let assembly = self.delta.begin(elements.iter().copied())?;
                measure(
                    assembly,
                    member.index,
                    &base,
                    &perturbed,
                    point - parameter,
                    label,
                )
                .map(Some)
            })?;
            response.push(result.unwrap_or((Complex::new(Real::NAN, Real::NAN), Real::NAN)));
        }
        Ok(response)
    }
}

impl ParameterWalk<'_> {
    /// C `sens_sens` parameter lifecycle. The callback sees an already numbered
    /// perturbation and the two device loads, independent of DC/AC assembly.
    fn sweep(
        &mut self,
        load: impl Fn(
            usize,
            &Member<'_>,
            RecordPair<'_>,
            RecordPair<'_>,
        ) -> SpiceResult<SensitivityLoad>,
        mut measure: impl FnMut(
            BaseAssembly<'_>,
            usize,
            &SensitivityLoad,
            &SensitivityLoad,
            Real,
            &str,
        ) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        // A device that refuses `.sens` outright keeps its own error class; only a
        // numerical failure is labelled with the parameter that caused it.
        fn labelled(error: SpiceError, label: &str, what: &str) -> SpiceError {
            match error {
                error @ (SpiceError::NotYetPorted { .. } | SpiceError::Unsupported { .. }) => error,
                error => SpiceError::Numerical {
                    context: format!(".sens {label}: {error}"),
                    message: what.to_owned(),
                },
            }
        }
        for entry in &self.entries {
            let Some(label) = entry.name.as_deref() else {
                continue;
            };
            let member = &self.members[entry.member];
            let device = &member.sensitivity;
            mirror(&self.couplings, &mut self.records, entry.member)?;
            let (model, instance) = self.records.pair(member, entry.member);
            // `sgen_next()` asks the value before this parameter's DEVsetup.
            let value = device.ask(entry.scope, entry.keyword, RecordPair { model, instance })?;
            device.setup(model, instance)?;
            device.temperature(model, instance)?;
            let snapshot = (model.clone(), instance.clone());
            let setup = RecordPair {
                model: &snapshot.0,
                instance: &snapshot.1,
            };
            let elements = device.setup_elements(setup)?;
            // A device whose perturbed state C loads without complaint but the
            // port refuses (a strict range check) has to name the parameter:
            // `.sens` is C's walk over dozens of them.
            let base = load(entry.member, member, setup, setup)
                .map_err(|error| labelled(error, label, "the base load failed"))?;
            let delta = if value == 0. {
                ABSOLUTE_DELTA
            } else {
                value * RELATIVE_DELTA
            };
            device.set(entry.scope, entry.keyword, value + delta, model, instance)?;
            device.temperature(model, instance)?;
            let perturbed = load(
                entry.member,
                member,
                setup,
                RecordPair {
                    model: &*model,
                    instance: &*instance,
                },
            )
            .map_err(|error| labelled(error, label, "the perturbed load failed"))?;
            let elements = setup_elements(self.circuit, member.index, elements)
                .map_err(|error| perturbation_error(label, &error))?;
            let assembly = self
                .delta
                .begin(elements)
                .map_err(|error| perturbation_error(label, &error))?;
            measure(assembly, member.index, &base, &perturbed, delta, label)?;
            device.set(entry.scope, entry.keyword, value, model, instance)?;
            device.temperature(model, instance)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Output, Stepping, frequencies, parse, scan};

    fn tokens(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    const fn stamp(row: usize, col: usize, value: crate::primitives::Complex) -> super::Stamp {
        super::Stamp { row, col, value }
    }

    #[test]
    fn conditioned_steps_are_representable_and_refine_smooth_limits() {
        use super::{Complex, conditioned_derivative};
        let reference = |x: f64| x.mul_add(x, 2. * x);
        let p = 0.;
        let (value, span) = conditioned_derivative(p, |point| {
            Ok(Some(Complex::real(reference(point) - reference(p))))
        })
        .unwrap()
        .unwrap();
        assert!((value.re - 2.).abs() < 1e-11);
        assert!(span > 0. && span.is_finite());

        // Negative values are outside the domain; extrapolate actual
        // representable one-sided slopes instead of stepping across zero.
        let (value, _) = conditioned_derivative(p, |point| {
            Ok((point >= 0.).then(|| Complex::real(reference(point) - reference(p))))
        })
        .unwrap()
        .unwrap();
        assert!((value.re - 2.).abs() < 1e-11);

        // A large, non-power-of-two coordinate gets actual binary64 spans,
        // not its rounded nominal perturbation as the denominator.
        let p = 1_000_000_000_000.125;
        let (value, _) =
            conditioned_derivative(p, |point| Ok(Some(Complex::real(3. * (point - p)))))
                .unwrap()
                .unwrap();
        assert!((value.re - 3.).abs() < 1e-12);
        // A one-ulp baseline error is amplified as the step shrinks, even
        // for an affine residual. Its insignificant asymmetry is not a cusp.
        let p = 1.2e-14;
        let gain: f64 = -9.585_764_503_17e11;
        let baseline = (gain * p).next_up();
        let (value, _) =
            conditioned_derivative(p, |point| Ok(Some(Complex::real(gain * point - baseline))))
                .unwrap()
                .unwrap();
        assert!(((value.re - gain) / gain).abs() < 1e-10);

        // A nonzero mean slope must not hide a genuine cusp.
        assert!(
            conditioned_derivative(0., |point| {
                Ok(Some(Complex::real(3. * point + point.abs())))
            })
            .unwrap()
            .is_none()
        );
        // A central secant across a cusp would be a fake zero derivative.
        assert!(
            conditioned_derivative(0., |point| { Ok(Some(Complex::real(point.abs()))) })
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn resolution_is_measured_in_output_units_and_scaled_by_perturbation() {
        use super::{Complex, Plot, PlotFlags, Probe, Real, Resolution};
        let plot = Plot::new("test", "test", PlotFlags::Real, vec![]);
        let probe = Probe::Current { row: 0 };
        let options = crate::analysis::newton::NewtonOptions {
            abstol: 1e-16,
            vntol: 1e-13,
            ..crate::analysis::newton::NewtonOptions::default()
        };
        let resolution = Resolution {
            plot,
            absolute: probe.absolute_resolution(&options),
        };
        let (floor, _) = resolution.row(probe, &[Complex::real(1e-4)]);
        let tox_denominator = 20e-9 * super::RELATIVE_DELTA;
        assert!((floor / tox_denominator - 0.005).abs() < 8. * Real::EPSILON);
        let voltage = Probe::Voltage {
            pos: Some(0),
            neg: Some(1),
        };
        let scale = voltage.resolution_scale(&[Complex::real(2.), Complex::real(2.)]);
        assert_eq!(scale.to_bits(), 4_f64.to_bits());
        assert_eq!(
            voltage.absolute_resolution(&options).to_bits(),
            options.vntol.to_bits()
        );
    }

    #[test]
    fn perturbation_products_follow_internal_column_order_and_clear_workspace() {
        use super::DeltaMatrix;
        use crate::primitives::Complex;

        let mut matrix = DeltaMatrix::new(3);
        let base = matrix.begin([(2, 2), (0, 0), (1, 1)]).unwrap();
        let mut perturbed = base.negate();
        perturbed
            .add([
                stamp(0, 0, Complex::real(-1e16)),
                stamp(0, 1, Complex::real(1.)),
                stamp(0, 2, Complex::real(1e16)),
            ])
            .unwrap();
        let x = [Complex::real(1.); 3];
        let mut result = [Complex::ZERO; 3];
        perturbed
            .visit_products(&x, |row, [real, imaginary]| {
                result[row] = Complex::new(real.total(), imaginary.total())
            })
            .unwrap();
        assert_eq!(result, [Complex::real(1.), Complex::ZERO, Complex::ZERO]);
        let stamp_capacity = matrix.stamps.capacity();
        let row_capacity = matrix.row_entries.capacity();
        matrix
            .begin([(1, 0)])
            .unwrap()
            .negate()
            .visit_products(&x, |row, [real, imaginary]| {
                result[row] = Complex::new(real.total(), imaginary.total())
            })
            .unwrap();
        assert_eq!(result, [Complex::ZERO; 3]);
        assert_eq!(matrix.stamps.capacity(), stamp_capacity);
        assert_eq!(matrix.row_entries.capacity(), row_capacity);
        assert_eq!(
            matrix.numbering.internal,
            [
                Some(super::InternalIndex(1)),
                Some(super::InternalIndex(2)),
                Some(super::InternalIndex(0))
            ]
        );
    }

    #[test]
    fn perturbation_residual_retains_product_roundoff_across_rhs_cancellation() {
        use super::DeltaMatrix;
        use crate::primitives::Complex;

        let small = 2_f64.powi(-27);
        let mut matrix = DeltaMatrix::new(1);
        let mut perturbed = matrix.begin([]).unwrap().negate();
        perturbed
            .add([stamp(0, 0, Complex::real(1. + small))])
            .unwrap();
        perturbed
            .visit_products(&[Complex::real(1. - small)], |_, [mut real, _]| {
                real.negate();
                real.add(1.);
                assert_eq!(real.total(), small * small);
            })
            .unwrap();
    }

    #[test]
    fn perturbation_rejects_invalid_setup_and_load_coordinates() {
        use super::DeltaMatrix;
        use crate::primitives::Complex;
        let mut matrix = DeltaMatrix::new(2);
        for (row, col) in [(0, 2), (2, 0)] {
            assert!(matrix.begin([(row, col)]).is_err());
            assert!(
                matrix
                    .begin([])
                    .unwrap()
                    .add([stamp(row, col, Complex::real(1.))])
                    .is_err()
            );
            assert!(
                matrix
                    .begin([])
                    .unwrap()
                    .negate()
                    .add([stamp(row, col, Complex::real(1.))])
                    .is_err()
            );
        }
        assert!(
            matrix
                .begin([])
                .unwrap()
                .negate()
                .visit_products(&[Complex::ZERO], |_, _| {})
                .is_err()
        );
        assert_eq!(matrix.numbering.next, 0);
    }

    #[test]
    fn separate_base_and_perturbed_reductions_retain_small_differences_and_signed_zeros() {
        use super::DeltaMatrix;
        use crate::primitives::Complex;
        let mut matrix = DeltaMatrix::new(4);
        let mut base = matrix.begin([(1, 0)]).unwrap();
        base.add([
            stamp(0, 0, Complex::new(1e16, 0.)),
            stamp(0, 0, Complex::new(1., 0.)),
            stamp(0, 0, Complex::new(-1e16, 0.)),
            stamp(1, 1, Complex::real(0.)),
        ])
        .unwrap();
        let mut perturbed = base.negate();
        perturbed
            .add([
                stamp(0, 0, Complex::real(2.)),
                stamp(2, 2, Complex::new(-0., -0.)),
            ])
            .unwrap();
        let cutoff = perturbed.base_count;
        assert_eq!(
            perturbed.matrix.fold_row(0, cutoff)[0].value().re.to_bits(),
            1_f64.to_bits()
        );
        for row in [1, 2] {
            let entry = &perturbed.matrix.fold_row(row, cutoff)[0];
            assert_eq!(entry.value().re.to_bits(), (-0_f64).to_bits());
            assert_eq!(entry.value().im.to_bits(), (-0_f64).to_bits());
        }
        let mut seen = Vec::new();
        perturbed
            .visit_products(&[Complex::real(1.); 4], |row, [real, imaginary]| {
                seen.push((row, Complex::new(real.total(), imaginary.total())))
            })
            .unwrap();
        assert_eq!(
            seen,
            [
                (0, Complex::real(1.)),
                (1, Complex::ZERO),
                (2, Complex::ZERO),
                (3, Complex::ZERO)
            ]
        );
    }

    #[test]
    fn perturbation_does_not_round_the_base_sum_before_subtraction() {
        use super::DeltaMatrix;
        use crate::primitives::Complex;
        let mut matrix = DeltaMatrix::new(1);
        let mut base = matrix.begin([(0, 0)]).unwrap();
        base.add([
            stamp(0, 0, Complex::new(1e16, -1e16)),
            stamp(0, 0, Complex::new(1., -1.)),
        ])
        .unwrap();
        let mut perturbed = base.negate();
        perturbed
            .add([
                stamp(0, 0, Complex::new(1e16, -1e16)),
                stamp(0, 0, Complex::new(2., -2.)),
            ])
            .unwrap();
        let cutoff = perturbed.base_count;
        assert_eq!(
            perturbed.matrix.fold_row(0, cutoff)[0].value(),
            Complex::new(1., -1.)
        );
    }

    #[test]
    fn dc_nonfinite_mask_is_block_local_and_reset_between_parameters() {
        use super::{Probe, blocks, mask_nonfinite, probe_tainted};
        use crate::maths::Vector;

        let blocks = blocks(5, [(0, 2), (2, 1), (3, 4)]).unwrap();
        let mut rhs = Vector::from_slice(&[1., f64::INFINITY, -0., 3., 4.]);
        let mut tainted = Vec::new();
        mask_nonfinite(&mut rhs, &blocks, &mut tainted);
        assert_eq!(rhs.as_slice(), &[1., 0., -0., 3., 4.]);
        assert_eq!(rhs.as_slice()[2].to_bits(), (-0_f64).to_bits());
        assert!(probe_tainted(Probe::Current { row: 0 }, &blocks, &tainted));
        assert!(!probe_tainted(Probe::Current { row: 3 }, &blocks, &tainted));
        assert!(probe_tainted(
            Probe::Voltage {
                pos: Some(4),
                neg: Some(2)
            },
            &blocks,
            &tainted
        ));
        let capacity = tainted.capacity();
        mask_nonfinite(&mut rhs, &blocks, &mut tainted);
        assert!(tainted.is_empty());
        assert_eq!(tainted.capacity(), capacity);
        assert!(!probe_tainted(Probe::Current { row: 0 }, &blocks, &tainted));
        assert!(rhs.is_finite());
    }

    #[test]
    fn ac_nonfinite_structure_spans_dynamic_and_static_couplings() {
        use super::{blocks, poisoned_rows};
        use crate::devices::linear::LinearSystem;
        let mut system = LinearSystem::new(5);
        system.a.add(0, 1, 2.).unwrap();
        system.a.add(3, 4, 3.).unwrap();
        system.e.add(1, 2, f64::MAX).unwrap();
        assert!(poisoned_rows(&system, 1.).unwrap().is_empty());
        assert_eq!(
            poisoned_rows(&system, 2.)
                .unwrap()
                .into_iter()
                .collect::<Vec<_>>(),
            [0, 1, 2],
        );
        assert!(blocks(2, [(0, 2)]).is_err());
        system.clear();
        assert!(poisoned_rows(&system, 2.).unwrap().is_empty());
    }

    #[test]
    fn cards_parse_outputs_filters_and_sweeps() {
        let card = parse(&tokens("V ( Out ) R* d1:is ac DEC 10 1 1k")).unwrap();
        assert_eq!(
            card.output,
            Output::Voltage {
                pos: "out".into(),
                neg: None
            }
        );
        assert_eq!(card.filters, ["r*", "d1:is"]);
        let sweep = card.sweep.unwrap();
        assert_eq!(sweep.stepping, Stepping::Decade);
        assert_eq!(sweep.steps, 10);
        let card = parse(&tokens("v ( a , b ) dc")).unwrap();
        assert!(card.sweep.is_none() && card.filters.is_empty());
        let card = parse(&tokens("i ( vm )")).unwrap();
        assert_eq!(
            card.output,
            Output::Current {
                source: "vm".into()
            }
        );
        for (text, message) in [
            ("", "no output variable"),
            ("x ( a )", "neither a voltage"),
            ("v ( a", "is missing"),
            ("v ( a ) ac foo 1 1 2", "dec, oct or lin"),
            ("v ( a ) ac dec 1 1", "stop frequency"),
            ("v ( a ) dc extra", "unexpected argument"),
            ("v ( a ) ( b", "unexpected '('"),
        ] {
            let error = parse(&tokens(text)).expect_err(text);
            assert!(error.to_string().contains(message), "{text}: {error}");
        }
    }

    #[test]
    fn filters_match_like_cktsens_scan() {
        assert!(scan(b"r*", b"r1_m"));
        assert!(scan(b"*", b"anything"));
        assert!(scan(b"r?", b"r1"));
        assert!(!scan(b"r?", b"r12"));
        assert!(scan(b"d1:*s", b"d1:is"));
        assert!(!scan(b"v1", b"v1_z0"));
        assert!(scan(b"v1", b"v1"));
        assert!(scan(b"*_m", b"r2_m"));
        assert!(!scan(b"*_m", b"r2_ms"));
    }

    #[test]
    fn frequency_count_rounding_is_a_discrete_grid_policy() {
        // Exact binary64 log10(4.6060996915966586) on the reference platform.
        let log = f64::from_bits(0x3fe5_3a06_d3a0_6d39);
        let scaled = 3. * log;
        assert_eq!((scaled + 1.01).trunc(), 3.);
        assert_eq!(3_f64.mul_add(log, 1.01).trunc(), 2.);
        let sweep = super::Sweep {
            stepping: super::Stepping::Decade,
            steps: 3,
            start: 1.,
            stop: 4.6060996915966586,
        };
        // On platforms whose log10 differs, still check the grid policy
        // against the separately rounded product on that platform.
        let scaled = 3. * sweep.stop.log10();
        let want = (scaled + 1.01).trunc();
        assert_eq!(super::frequencies(&sweep).unwrap().len() as f64, want);
    }

    #[test]
    fn sweeps_step_like_count_steps_and_inc_freq() {
        let sweep = |text: &str| {
            parse(&tokens(&format!("v ( a ) ac {text}")))
                .unwrap()
                .sweep
                .unwrap()
        };
        let f: Vec<_> = frequencies(&sweep("dec 1 1k 100k"))
            .unwrap()
            .values()
            .collect();
        assert_eq!(f.len(), 3);
        assert_eq!(f[0], 1e3);
        assert!((f[2] / 1e5 - 1.).abs() < 1e-12);
        // `lin` multiplies by (fstop - fstart)/pts (C's inc_freq defect).
        assert_eq!(
            frequencies(&sweep("lin 2 1k 2k"))
                .unwrap()
                .values()
                .collect::<Vec<_>>(),
            [1e3, 5e5]
        );
        // `oct` divides by M_LOG2E: 1 per octave over 1..16 gives 2 points.
        assert_eq!(frequencies(&sweep("oct 1 1 16")).unwrap().len(), 2);
        assert!(frequencies(&sweep("lin 2 1k 1k")).is_err());
        assert!(frequencies(&sweep("dec 1 0 1k")).is_err());
    }
}
