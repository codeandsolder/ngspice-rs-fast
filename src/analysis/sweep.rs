//! Typed bounded DC sweeps, including one outer sweep (`dctrcurv.c`).
//!
//! Targets are independent V/I sources, resistors (literal or model-backed,
//! identified by physical device metadata), circuit temperature and the
//! settable instance parameters `@inst[param]` that both C and the port
//! support. Model parameters (C rejects them too) and more than two axes (C
//! has two nesting levels) are explicitly unsupported.
//! Nothing is mutated: source DC values (loaded as `dctrcurv.c` sets
//! `dcValue`/`dcGiven`), temperature, resistor and instance-parameter values
//! travel in an immutable per-point `ModelContext`, and every point runs
//! `NIiter`, linear circuits included. See `docs/port/DC_SWEEPS.md`.
use crate::analysis::linear::{number, plot, unsupported};
use crate::analysis::{AnalysisContext, AnalysisPlan};
use crate::devices::{Circuit, InstanceTarget, ModelContext};
use crate::maths::Vector;
use crate::primitives::{Complex, Real, SpiceError, SpiceResult};

/// Largest number of grid points, per axis and as a Cartesian product.
pub const MAX_SWEEP_POINTS: usize = MAX_SWEEP_POINTS_U32 as usize;
const MAX_SWEEP_POINTS_U32: u32 = 100_000;
/// Relative slack, in steps, within which a grid is taken to reach its stop
/// (`0..0.3 by 0.1` is `2.9999999999999996` steps in binary floating point).
const ENDPOINT_TOLERANCE: Real = 32. * Real::EPSILON;
/// C `CONSTCtoK`: `dctrcurv.c` accumulates a temperature sweep in kelvin.
const CELSIUS_TO_KELVIN: Real = 273.15;

/// A resolved sweep target, not an arbitrary parameter-name string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SweepTarget {
    /// Independent voltage-source DC value (V).
    VoltageSource(String),
    /// Independent current-source DC value (A).
    CurrentSource(String),
    /// Supplied scalar resistance (ohms) of a literal or model-backed resistor.
    /// Temperature, TC, scale and multiplicity still apply to model-backed
    /// resistors (`crate::devices::sweep`).
    Resistor(String),
    /// Circuit temperature (Celsius); nominal temperature is unchanged.
    Temperature,
    /// A settable real instance parameter, `.dc @instance[parameter]` (C
    /// `dctrcurv.c` `PARAM_CODE`, `DCTfindInstParam`/`DCTsetInstParam`).
    InstanceParameter {
        /// The target as C names it, `@instance[parameter]`, with the
        /// instance's own spelling and the canonical parameter keyword.
        name: String,
        /// The instance name, in the device's own spelling.
        instance: String,
        /// The canonical (lowercase, alias-folded) parameter keyword.
        parameter: String,
        /// How the value reaches the equations.
        route: ParameterRoute,
    },
}
/// How an [`SweepTarget::InstanceParameter`] value reaches the equations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterRoute {
    /// `@v1[dc]`: the voltage-source DC value, as [`SweepTarget::VoltageSource`].
    VoltageSource,
    /// `@i1[dc]` / `@i1[c]`: the current-source DC value, as
    /// [`SweepTarget::CurrentSource`].
    CurrentSource,
    /// `@r1[r]` / `@r1[resistance]`: the supplied resistance, as
    /// [`SweepTarget::Resistor`] (C `RESparam` then `REStemp`).
    Resistor,
    /// Any other supported parameter: a per-point device replacement
    /// ([`crate::devices::Device::with_instance_parameter`]).
    Device,
}
impl SweepTarget {
    /// The scale vector of a sweep whose first (inner) axis is this target,
    /// as C names and types it in the plot: `dctrcurv.c` calls the scale
    /// `v-sweep`, `i-sweep`, `temp-sweep`, `res-sweep` or `param-sweep`, and
    /// the front end types it by name (`guess_type`, `outitf.c`), which
    /// leaves `v-sweep` and `param-sweep` as voltages. Like the port's other
    /// vectors, voltage and current names carry their `v(...)`/`i(...)`
    /// wrapping, as C's rawfile writes them.
    #[must_use = "the returned name and quantity identify the rawfile sweep axis"]
    pub const fn scale(&self) -> (&'static str, &'static str) {
        match self {
            Self::VoltageSource(_) => ("v(v-sweep)", "voltage"),
            Self::CurrentSource(_) => ("i(i-sweep)", "current"),
            Self::Resistor(_) => ("res-sweep", "res-sweep"),
            Self::Temperature => ("temp-sweep", "temp-sweep"),
            Self::InstanceParameter { .. } => ("v(param-sweep)", "voltage"),
        }
    }
    /// The instance name, `temp`, or `@instance[parameter]`.
    #[must_use = "the returned identifier selects the swept source, device parameter or temperature"]
    pub fn name(&self) -> &str {
        match self {
            Self::VoltageSource(n) | Self::CurrentSource(n) | Self::Resistor(n) => n,
            Self::Temperature => "temp",
            Self::InstanceParameter { name, .. } => name,
        }
    }
    /// True when the swept value is a supplied resistance.
    const fn is_resistance(&self) -> bool {
        matches!(
            self,
            Self::Resistor(_)
                | Self::InstanceParameter {
                    route: ParameterRoute::Resistor,
                    ..
                }
        )
    }
}
/// A finite directional sweep with an inclusive reachable stop and bounded work.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepSpec {
    /// Typed source, resistor or circuit-temperature target.
    pub target: SweepTarget,
    /// First value in the target's physical unit.
    pub start: Real,
    /// Last bound (included only if reached on the grid).
    pub stop: Real,
    /// Nonzero step with the correct direction.
    pub step: Real,
}
impl SweepSpec {
    /// Build a bounded grid `start, start+step, ...`, in the sign of `step`.
    ///
    /// The stop is included when the grid reaches it within rounding (the
    /// last value is then exactly `stop`) and is never overshot. A single
    /// point (`start == stop`) is allowed. Rejects nonfinite inputs, a zero or
    /// wrong-direction step, overflow, more than [`MAX_SWEEP_POINTS`], steps too
    /// small to make progress, temperatures at/below absolute zero and
    /// resistances that are zero or have a nonfinite conductance.
    /// # Errors
    /// [`SpiceError::Unsupported`] for any of the above.
    pub fn grid(&self) -> SpiceResult<Vec<Real>> {
        Ok(self.exact_values()?.collect())
    }

    const fn range(&self) -> SweepRange {
        SweepRange {
            start: self.start,
            stop: self.stop,
            step: self.step,
        }
    }

    fn exact_values(&self) -> SpiceResult<impl ExactSizeIterator<Item = Real> + Clone + use<>> {
        let values = self.range().exact_grid()?.values();
        for value in values.clone() {
            check_domain(
                value,
                self.target == SweepTarget::Temperature,
                self.target.is_resistance(),
            )?;
        }
        Ok(values)
    }

    /// C's repeated-addition grid, with the absolute stop test and kelvin
    /// accumulation for temperature axes.
    ///
    /// # Errors
    /// As [`Self::grid`], including bounds on accumulated progress and work.
    pub fn accumulated_grid(&self) -> SpiceResult<Vec<Real>> {
        let _ = self.exact_values()?;
        let temperature = self.target == SweepTarget::Temperature;
        let values = self.range().accumulate(temperature)?;
        for value in values.values() {
            check_domain(value, temperature, self.target.is_resistance())?;
        }
        Ok(values.values().collect())
    }
}

fn check_domain(value: Real, temperature: bool, resistance: bool) -> SpiceResult<()> {
    if temperature && value <= -273.15 {
        return Err(unsupported("DC temperature below absolute zero"));
    }
    if resistance && (value == 0. || !(1. / value).is_finite()) {
        return Err(unsupported(
            "DC resistance must be nonzero with finite conductance",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SweepRange {
    start: Real,
    stop: Real,
    step: Real,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ExactGrid {
    range: SweepRange,
    intervals: u32,
    snap_stop: bool,
}

impl ExactGrid {
    fn values(self) -> impl ExactSizeIterator<Item = Real> + Clone {
        let SweepRange { start, stop, step } = self.range;
        let point_count = self.intervals + 1;
        (0..point_count).map(move |i| {
            if self.snap_stop && i == self.intervals {
                stop
            } else {
                let offset = Real::from(i) * step;
                start + offset
            }
        })
    }
}

/// Compact, validated C accumulated traversal, including Kelvin offset.
#[derive(Debug, Clone, Copy, PartialEq)]
struct AccumulatedGrid {
    range: SweepRange,
    temperature: bool,
    count: usize,
}

impl AccumulatedGrid {
    fn values(self) -> GridValues {
        let offset = if self.temperature {
            CELSIUS_TO_KELVIN
        } else {
            0.
        };
        GridValues {
            stored: self.range.start + offset,
            offset,
            step: self.range.step,
            remaining: self.count,
        }
    }
}

#[derive(Debug, Clone)]
struct GridValues {
    stored: Real,
    offset: Real,
    step: Real,
    remaining: usize,
}

impl GridValues {
    const fn singleton(value: Real) -> Self {
        Self {
            stored: value,
            offset: 0.,
            step: 0.,
            remaining: 1,
        }
    }
}

impl Iterator for GridValues {
    type Item = Real;

    fn next(&mut self) -> Option<Real> {
        if self.remaining == 0 {
            return None;
        }
        let value = self.stored - self.offset;
        self.remaining -= 1;
        if self.remaining != 0 {
            self.stored += self.step;
        }
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl ExactSizeIterator for GridValues {}

impl SweepRange {
    fn exact_grid(self) -> SpiceResult<ExactGrid> {
        let (start, stop, increment) = (self.start, self.stop, self.step);
        if ![start, stop, increment].iter().all(|v| v.is_finite()) || increment == 0. {
            return Err(unsupported(
                "DC sweep needs finite start/stop and a nonzero step",
            ));
        }
        let ratio = (stop - start) / increment;
        if !ratio.is_finite() {
            return Err(unsupported("DC sweep span overflows"));
        }
        if ratio < 0. {
            return Err(unsupported("DC sweep step points away from its stop"));
        }
        let slack = ENDPOINT_TOLERANCE * ratio.max(1.);
        let interval_count = (ratio + slack).floor();
        if interval_count + 1. > Real::from(MAX_SWEEP_POINTS_U32) {
            return Err(unsupported("DC sweep point limit exceeded"));
        }
        let intervals = crate::primitives::exact_nonnegative_integer::<u32>(interval_count)
            .ok_or_else(|| unsupported("DC sweep point limit exceeded"))?;
        let snap_stop = intervals > 0 && (ratio - Real::from(intervals)).abs() <= slack;
        let grid = ExactGrid {
            range: self,
            intervals,
            snap_stop,
        };
        let mut previous = None;
        for value in grid.values() {
            self.check_progress(value, previous)?;
            previous = Some(value);
        }
        Ok(grid)
    }

    fn accumulate(self, temperature: bool) -> SpiceResult<AccumulatedGrid> {
        let offset = if temperature { CELSIUS_TO_KELVIN } else { 0. };
        let tolerance = 1e3 * Real::EPSILON;
        let direction = self.step.signum();
        let mut stored = self.start + offset;
        let mut previous = None;
        let mut count = 0;
        loop {
            let value = stored - offset;
            if direction * (value - self.stop) > tolerance {
                break;
            }
            if count >= MAX_SWEEP_POINTS {
                return Err(unsupported("DC sweep point limit exceeded"));
            }
            self.check_progress(value, previous)?;
            count += 1;
            previous = Some(value);
            stored += self.step;
        }
        Ok(AccumulatedGrid {
            range: self,
            temperature,
            count,
        })
    }

    fn check_progress(self, value: Real, previous: Option<Real>) -> SpiceResult<()> {
        if !value.is_finite() {
            return Err(unsupported("DC sweep overflows"));
        }
        if previous.is_some_and(|previous| (value - previous) * self.step.signum() <= 0.) {
            return Err(unsupported("DC sweep makes no progress"));
        }
        Ok(())
    }
}
/// Split a C instance-parameter target `@instance[parameter]` (`dctrcurv.c`
/// `DCTfindInstParam`) into its nonempty instance and parameter. C ignores
/// anything after the closing bracket; the port rejects it.
fn parameter_target(text: &str) -> SpiceResult<Option<(&str, &str)>> {
    let Some(rest) = text.strip_prefix('@') else {
        return Ok(None);
    };
    let malformed = || {
        unsupported(format!(
            "malformed DC sweep target {text}; expected @instance[parameter]"
        ))
    };
    let (instance, rest) = rest.split_once('[').ok_or_else(malformed)?;
    let parameter = rest.strip_suffix(']').ok_or_else(malformed)?;
    if instance.is_empty() || parameter.is_empty() || parameter.contains(['[', ']']) {
        return Err(malformed());
    }
    Ok(Some((instance, parameter)))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AxisTarget {
    Temperature,
    Name(String),
    Parameter { instance: String, parameter: String },
}

impl AxisTarget {
    fn parse(text: &str) -> SpiceResult<Self> {
        if text.eq_ignore_ascii_case("temp") {
            return Ok(Self::Temperature);
        }
        Ok(match parameter_target(text)? {
            Some((instance, parameter)) => Self::Parameter {
                instance: instance.to_owned(),
                parameter: parameter.to_owned(),
            },
            None => Self::Name(text.to_owned()),
        })
    }
}

/// A DC sweep with checked numeric grids, before circuit identity binding.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DcSweepPlan {
    axes: Vec<AxisPlan>,
    grids: Vec<AccumulatedGrid>,
}

#[derive(Debug, Clone, PartialEq)]
struct AxisPlan {
    target: AxisTarget,
    exact: ExactGrid,
}

pub(super) fn parse_axes(args: &[String]) -> SpiceResult<DcSweepPlan> {
    if !matches!(args.len(), 4 | 8) {
        return Err(unsupported(
            ".dc requires one or two target/start/stop/step axes",
        ));
    }
    let mut axes = Vec::with_capacity(args.len() / 4);
    let mut grids = Vec::with_capacity(axes.capacity());
    let mut exact_work = 1usize;
    let mut accumulated_work = 1usize;
    for axis in args.as_chunks::<4>().0 {
        let target = AxisTarget::parse(&axis[0])?;
        let range = SweepRange {
            start: number(Some(&axis[1]), "sweep start")?,
            stop: number(Some(&axis[2]), "sweep stop")?,
            step: number(Some(&axis[3]), "sweep step")?,
        };
        let exact = range.exact_grid()?;
        let expected = exact.values().len();
        exact_work = bounded_product(exact_work, expected)?;
        let temperature = target == AxisTarget::Temperature;
        let values = range.accumulate(temperature)?;
        accumulated_work = bounded_product(accumulated_work, values.count)?;
        if temperature {
            for value in exact.values().chain(values.values()) {
                check_domain(value, true, false)?;
            }
        }
        axes.push(AxisPlan { target, exact });
        grids.push(values);
    }
    Ok(DcSweepPlan { axes, grids })
}

impl DcSweepPlan {
    pub(super) fn point_count(&self) -> usize {
        self.grids.iter().map(|grid| grid.count).product()
    }
}

fn bounded_product(work: usize, points: usize) -> SpiceResult<usize> {
    work.checked_mul(points)
        .filter(|n| *n <= MAX_SWEEP_POINTS)
        .ok_or_else(|| unsupported("nested DC sweep point limit exceeded"))
}

/// What one axis written as `target` names before the source table exists:
/// the quantity it sweeps (for duplicate detection) and how it is applied.
enum Written<'a> {
    Temperature,
    /// A plain name: an independent source or a resistor.
    Name(&'a str),
    /// `@instance[r|resistance]` of a resistor.
    Resistance(&'a str, InstanceTarget),
    /// `@instance[parameter]` that the device replaces itself.
    Device(&'a str, InstanceTarget, &'a str),
    /// `@instance[parameter]` left for the source table (`dc`, current `c`)
    /// or for an explicit rejection.
    Other(&'a str, &'a str),
}
impl<'a> Written<'a> {
    fn classify(circuit: &'a Circuit, target: &'a AxisTarget) -> Self {
        let (instance, parameter) = match target {
            AxisTarget::Temperature => return Self::Temperature,
            AxisTarget::Name(name) => return Self::Name(name),
            AxisTarget::Parameter {
                instance,
                parameter,
            } => (instance.as_str(), parameter.as_str()),
        };
        if (parameter.eq_ignore_ascii_case("r") || parameter.eq_ignore_ascii_case("resistance"))
            && let Ok(target) = circuit.bind_resistor_target(instance)
        {
            return Self::Resistance(instance, target);
        }
        if let Ok(target) = circuit.bind_instance_target(instance, parameter) {
            let name = circuit.devices()[target.device()]
                .numeric_parameter_name(target.parameter())
                .unwrap_or(parameter);
            return Self::Device(instance, target, name);
        }
        Self::Other(instance, parameter)
    }
    /// The swept quantity: `r1` and `@r1[r]`, `v1` and `@v1[dc]` are one.
    fn key(&self) -> String {
        match self {
            Self::Temperature => "temp".into(),
            Self::Name(name) | Self::Resistance(name, _) => name.to_ascii_lowercase(),
            Self::Device(name, _, parameter) => {
                format!("@{}[{parameter}]", name.to_ascii_lowercase())
            }
            Self::Other(name, parameter) => match parameter.to_ascii_lowercase().as_str() {
                "dc" | "c" => name.to_ascii_lowercase(),
                other => format!("@{}[{other}]", name.to_ascii_lowercase()),
            },
        }
    }
}

fn resolve_target(
    circuit: &Circuit,
    written: &Written<'_>,
) -> SpiceResult<(SweepTarget, AxisBinding)> {
    let source = |name: &str| {
        circuit.device(name).and_then(|device| {
            device
                .input_source()
                .map(|source| (device.name(), source.kind))
        })
    };
    let instance_name = |index: usize| circuit.devices()[index].name().to_owned();
    let parameter = |instance: String, parameter: &str, route| SweepTarget::InstanceParameter {
        name: format!("@{instance}[{parameter}]"),
        instance,
        parameter: parameter.to_owned(),
        route,
    };
    Ok(match *written {
        Written::Temperature => (SweepTarget::Temperature, AxisBinding::Temperature),
        Written::Name(name) => {
            if let Some((name, kind)) = source(name) {
                let target = match kind {
                    crate::devices::SourceKind::Voltage => {
                        SweepTarget::VoltageSource(name.to_owned())
                    }
                    crate::devices::SourceKind::Current => {
                        SweepTarget::CurrentSource(name.to_owned())
                    }
                };
                (
                    target,
                    AxisBinding::Source(circuit.bind_source_target(name)?),
                )
            } else if let Ok(target) = circuit.bind_resistor_target(name) {
                (
                    SweepTarget::Resistor(instance_name(target.device())),
                    AxisBinding::Resistor(target),
                )
            } else {
                return Err(unsupported(format!(
                    "DC sweep supports independent V/I sources, resistors, temp and \
                         @instance[parameter] only; unsupported target {name}"
                )));
            }
        }
        Written::Resistance(_, target) => (
            parameter(
                instance_name(target.device()),
                "r",
                ParameterRoute::Resistor,
            ),
            AxisBinding::Resistor(target),
        ),
        Written::Device(_, target, keyword) => (
            parameter(
                instance_name(target.device()),
                keyword,
                ParameterRoute::Device,
            ),
            AxisBinding::Device(target),
        ),
        Written::Other(name, keyword) => {
            let keyword = keyword.to_ascii_lowercase();
            match source(name) {
                // vsrc.c/isrc.c: `dc` (isrc alias `c`) is the DC value
                // that `.dc v1` sweeps.
                Some((name, kind))
                    if keyword == "dc"
                        || (keyword == "c" && kind == crate::devices::SourceKind::Current) =>
                {
                    let route = match kind {
                        crate::devices::SourceKind::Voltage => ParameterRoute::VoltageSource,
                        crate::devices::SourceKind::Current => ParameterRoute::CurrentSource,
                    };
                    (
                        parameter(name.to_owned(), "dc", route),
                        AxisBinding::Source(circuit.bind_source_target(name)?),
                    )
                }
                // Explicit NotYetPorted/Unsupported with the C reference.
                _ => {
                    circuit.instance_parameter(name, &keyword)?;
                    return Err(SpiceError::circuit(format!(
                        "DC sweep target @{name}[{keyword}] could not be resolved"
                    )));
                }
            }
        }
    })
}

/// A topology-bound setter, independent of any particular grid value.
/// Resistors retain the supplied-scalar route, including its slot ordering.
#[derive(Debug, Clone, Copy)]
enum AxisBinding {
    Temperature,
    Resistor(InstanceTarget),
    Device(InstanceTarget),
    Source(InstanceTarget),
}

impl AxisBinding {
    const fn is_source(self) -> bool {
        matches!(self, Self::Source(_))
    }

    const fn changes_physics(self) -> bool {
        matches!(self, Self::Temperature | Self::Device(_))
    }
}

#[derive(Debug)]
struct BoundAxis {
    spec: SweepSpec,
    binding: AxisBinding,
}

/// Resolve one or two DC sweep axes against circuit devices and parameters.
///
/// Parse and resolve one/two axes against actual independent sources,
/// resistors (by physical metadata, never by instance-name prefix), circuit
/// temperature and `@instance[parameter]` targets.
/// Positional form: `target start stop step [outer start stop step]`; C's
/// `dot_dc` (`inp2dot.c`) reads exactly these two levels (`TRCVNESTLEVEL` is 2)
/// and silently ignores anything after them, which the port rejects.
/// # Errors
/// Missing/unsupported/duplicate targets, invalid grid or Cartesian work
/// budget; [`SpiceError::NotYetPorted`] for an instance parameter C sweeps but
/// the port does not.
pub fn resolve(
    circuit: &Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
) -> SpiceResult<Vec<SweepSpec>> {
    let crate::analysis::plan::PlanKind::Dc { axes, .. } = request.analysis() else {
        return Err(unsupported("typed sweep requires a DC request"));
    };
    let prepared = circuit.prepare_run();
    let axes = bind_axes(circuit, axes)?;
    validate_start(&prepared, &axes, context.model_context())?;
    Ok(axes.into_iter().map(|axis| axis.spec).collect())
}

fn bind_axes(circuit: &Circuit, plan: &DcSweepPlan) -> SpiceResult<Vec<BoundAxis>> {
    // Targets are physical metadata, not an AC matrix. Do not require an
    // otherwise DC-capable device to implement a small-signal load merely to
    // discover which name denotes an independent source.
    let mut names = std::collections::BTreeSet::new();
    let mut written = Vec::new();
    for axis in &plan.axes {
        let target = Written::classify(circuit, &axis.target);
        if !names.insert(target.key()) {
            return Err(unsupported("duplicate nested DC target"));
        }
        written.push(target);
    }
    let mut axes = Vec::with_capacity(plan.axes.len());
    for ((axis, written), grid) in plan.axes.iter().zip(&written).zip(&plan.grids) {
        let (target, binding) = resolve_target(circuit, written)?;
        if target.is_resistance() {
            for value in axis.exact.values().chain(grid.values()) {
                check_domain(value, false, true)?;
            }
        }
        let SweepRange { start, stop, step } = axis.exact.range;
        axes.push(BoundAxis {
            spec: SweepSpec {
                target,
                start,
                stop,
                step,
            },
            binding,
        });
    }
    Ok(axes)
}

fn validate_start(
    prepared: &crate::devices::PreparedRun<'_>,
    axes: &[BoundAxis],
    base: ModelContext,
) -> SpiceResult<()> {
    // The public resolver checks the first written value, not every point.
    let start = [
        axes[0].spec.start,
        axes.get(1).map_or(0., |axis| axis.spec.start),
    ];
    let point = point_context(prepared.circuit(), axes, start, base)?;
    prepared.validate_context(&point.model)
}

/// Independent source values fit in C's two-axis sweep layout. No heap
/// allocation is required per point.
struct PointContext<'a> {
    model: ModelContext,
    sources: [(&'a str, Real); 2],
    source_count: usize,
}

impl PointContext<'_> {
    fn source_values(&self) -> &[(&str, Real)] {
        &self.sources[..self.source_count]
    }
}

/// The per-point model context and source values of one Cartesian point:
/// `values` pairs with `axes` (inner first).
fn point_context<'a>(
    circuit: &'a Circuit,
    axes: &[BoundAxis],
    values: [Real; 2],
    base: ModelContext,
) -> SpiceResult<PointContext<'a>> {
    let mut model = base;
    let mut sources = [("", 0.); 2];
    let mut source_count = 0;
    for (axis, value) in axes.iter().zip(values) {
        match axis.binding {
            AxisBinding::Temperature => model.temperature = value,
            AxisBinding::Resistor(target) => {
                model = model.with_resistor_override(target.at_resistance(value)?)?;
            }
            AxisBinding::Source(_) | AxisBinding::Device(_) => {}
        }
    }
    // Preserve the temperature-first and source-last setter ordering. The
    // topology ordinal, canonical keyword and route never change per point.
    for (axis, value) in axes.iter().zip(values) {
        if let AxisBinding::Device(target) = axis.binding {
            model = model.with_instance_override(target.at(value)?)?;
        }
    }
    for (axis, value) in axes.iter().zip(values) {
        if let AxisBinding::Source(target) = axis.binding {
            model = model.with_instance_override(target.at(value)?)?;
            sources[source_count] = (circuit.devices()[target.device()].name(), value);
            source_count += 1;
        }
    }
    Ok(PointContext {
        model,
        sources,
        source_count,
    })
}

/// The same Cartesian traversal serves validation and execution. Independent
/// DC source values do not change thermal derivations; their largest absolute
/// value additionally validates the worst-case linear forcing (including a
/// concurrently swept current-source multiplier).
struct GridPair {
    inner: GridValues,
    outer: GridValues,
}

impl GridPair {
    fn new(grids: &[AccumulatedGrid]) -> Self {
        Self {
            inner: grids[0].values(),
            outer: grids
                .get(1)
                .map_or(GridValues::singleton(0.), |grid| grid.values()),
        }
    }

    fn physical_contexts(mut self, sources: [bool; 2]) -> Self {
        let unique = |grid: GridValues, is_source| {
            if is_source {
                match (grid.clone().next(), grid.clone().last()) {
                    (Some(first), Some(last)) => {
                        GridValues::singleton(if last.abs() > first.abs() {
                            last
                        } else {
                            first
                        })
                    }
                    _ => grid,
                }
            } else {
                grid
            }
        };
        self.inner = unique(self.inner, sources[0]);
        self.outer = unique(self.outer, sources[1]);
        self
    }

    fn points(&self) -> impl Iterator<Item = [Real; 2]> + '_ {
        self.outer
            .clone()
            .flat_map(|outer| self.inner.clone().map(move |inner| [inner, outer]))
    }
}

/// Reject every point the run would later fail on for *input* reasons: a swept
/// resistance whose effective value is invalid at some swept temperature, or a
/// circuit whose contextual device inputs are invalid at some swept temperature. Convergence
/// failures remain run-time errors.
fn preflight(
    prepared: &crate::devices::PreparedRun<'_>,
    axes: &[BoundAxis],
    grids: &[AccumulatedGrid],
    context: &AnalysisContext,
) -> SpiceResult<()> {
    let circuit = prepared.circuit();
    let base = context.model_context();
    let sources = [
        axes[0].binding.is_source(),
        axes.get(1).is_some_and(|axis| axis.binding.is_source()),
    ];
    let contexts = GridPair::new(grids).physical_contexts(sources);
    let (Some(inner), Some(outer)) = (contexts.inner.clone().next(), contexts.outer.clone().next())
    else {
        return Ok(());
    };
    let start = [inner, outer];
    // Check every coupled physical combination, but unchanged devices only
    // need validation once at each scalar temperature context. The grids
    // are strictly monotone, so the first value of the other axis identifies
    // the first visit to each temperature in either C traversal order.
    // Rebuild replacements every visit; no live device or trial state is
    // reused for a different ModelContext.
    if axes.iter().any(|axis| axis.binding.changes_physics()) {
        let temperature = axes
            .iter()
            .position(|axis| matches!(axis.binding, AxisBinding::Temperature));
        let coupling_may_change = axes
            .iter()
            .any(|axis| matches!(axis.binding, AxisBinding::Device(_)));
        for values in contexts.points() {
            let point = point_context(circuit, axes, values, base)?;
            let first_scalar_context = temperature.map_or_else(
                || values.map(Real::to_bits) == start.map(Real::to_bits),
                |axis| values[1 - axis].to_bits() == start[1 - axis].to_bits(),
            );
            if first_scalar_context {
                prepared.validate_dc_inputs(&point.model)?;
            } else {
                prepared.validate_dc_replacements(&point.model, coupling_may_change)?;
            }
        }
        return Ok(());
    }

    // Only independent resistances and source values (or both) vary. Their
    // physical recipes are independent at fixed temperature. Validate the
    // whole effective circuit once, then each resistor's remaining values
    // through the actual ordered setter/replacement path, not an approximate
    // resistance formula. This also validates C's clamping and any recipe
    // metadata. Avoid a full Cartesian product of unrelated resistors.
    let point = point_context(circuit, axes, start, base)?;
    prepared.validate_dc_inputs(&point.model)?;
    for (index, (axis, grid)) in axes.iter().zip(grids).enumerate() {
        if !matches!(axis.binding, AxisBinding::Resistor(_)) {
            continue;
        }
        for value in grid.values().skip(1) {
            let mut values = start;
            values[index] = value;
            let point = point_context(circuit, axes, values, base)?;
            prepared.validate_dc_replacements(&point.model, false)?;
        }
    }
    Ok(())
}

fn warm_start_settings(
    point_iterations: Option<usize>,
    settings: &crate::analysis::bias::DcSettings,
) -> Option<crate::analysis::bias::DcSettings> {
    // dctrcurv.c always warm-starts every point after the first with
    // `NIiter(CKTdcTrcvMaxIter)` (C's effective default 100) before falling
    // back to a fresh `CKTop`; the port's legacy ladder schedule instead runs
    // one full solve per point unless `itl2` is given.
    let point_iterations = point_iterations.or(match settings.continuation.schedule {
        crate::analysis::bias::ContinuationSchedule::Ngspice(_) => {
            Some(crate::analysis::config::C_DEFAULT_ITL2_EFFECTIVE)
        }
        crate::analysis::bias::ContinuationSchedule::Ladder { .. } => None,
    });
    // The direct warm-start policy is invariant across sweep points. Build it
    // once; only the previous accepted solution and model overrides change.
    point_iterations.map(|limit| crate::analysis::bias::DcSettings {
        newton: crate::analysis::newton::NewtonOptions {
            max_iterations: limit,
            ..settings.newton
        },
        continuation: crate::analysis::bias::ContinuationPolicy::disabled(),
    })
}

/// Policy and circuit inputs shared by all points in a DC sweep.
struct SweepSolver<'a> {
    prepared: crate::devices::PreparedRun<'a>,
    settings: &'a crate::analysis::bias::DcSettings,
    warm_settings: Option<&'a crate::analysis::bias::DcSettings>,
    nodes: &'a crate::analysis::bias::NodeForcing,
}

impl SweepSolver<'_> {
    fn solve(
        &self,
        point: &PointContext<'_>,
        previous: &Vector,
        history: &crate::devices::StateHistory,
        first: bool,
    ) -> SpiceResult<crate::analysis::newton::NewtonSolution<crate::devices::TrialState>> {
        use crate::analysis::bias::{DcOutcome, solve_dc_from};
        use crate::analysis::newton::PhasePolicy;
        // After the first point, C tries a bounded warm Newton solve before CKTop.
        if let Some(direct) = self.warm_settings.filter(|_| !first) {
            match solve_dc_from(
                &self.prepared,
                &point.model,
                direct,
                &[],
                Some(previous),
                history,
                PhasePolicy::Predicted,
                self.nodes,
            ) {
                Ok(solved) => return Ok(solved.solution),
                Err(failure)
                    if matches!(
                        failure.report.outcome,
                        DcOutcome::Exhausted | DcOutcome::BudgetExhausted
                    ) => {}
                Err(failure) => return Err(failure.error),
            }
        }
        // Without a warm limit the full solve uses the predicted point.
        // Otherwise retry with a fresh operating-point phase as C does.
        let phase = if first || self.warm_settings.is_some() {
            PhasePolicy::OperatingPoint
        } else {
            PhasePolicy::Predicted
        };
        Ok(solve_dc_from(
            &self.prepared,
            &point.model,
            self.settings,
            &[],
            Some(previous),
            history,
            phase,
            self.nodes,
        )?
        .solution)
    }
}

pub(super) fn run(
    circuit: &mut Circuit,
    request: &AnalysisPlan,
    context: &AnalysisContext,
    sweep: &DcSweepPlan,
    point_iterations: Option<usize>,
    sink: &mut dyn crate::analysis::PointSink,
) -> SpiceResult<()> {
    circuit.finalize()?;
    let circuit: &Circuit = circuit;
    // Everything that can be rejected is rejected before the first sample.
    let hints = crate::analysis::initial::resolve(circuit, request)?;
    // cktload.c forces .nodeset rows in the MODEINITJCT/MODEINITFIX loads
    // (the first point of each inner sweep and every CKTop restart).
    let nodes = crate::analysis::bias::NodeForcing {
        initial: Vec::new(),
        nodesets: crate::analysis::initial::forced_nodesets(&hints.nodesets),
    };
    let prepared = circuit.prepare_run();
    let axes = bind_axes(circuit, sweep)?;
    let settings = request.dc_settings();
    let warm_settings = warm_start_settings(point_iterations, settings);
    let solver = SweepSolver {
        prepared,
        settings,
        warm_settings: warm_settings.as_ref(),
        nodes: &nodes,
    };
    let grids = &sweep.grids;
    preflight(&solver.prepared, &axes, grids, context).map_err(|error| match error {
        SpiceError::Numerical { context, message } => {
            SpiceError::circuit(format!("DC sweep input validation ({context}): {message}"))
        }
        other => other,
    })?;
    // C order: the first axis is the inner (fast) loop, the second the outer.
    let GridPair { inner, outer } = GridPair::new(grids);
    let result = plot(
        circuit,
        "dc1",
        "DC transfer characteristic",
        // C writes only the inner axis' scale; an outer axis has no vector.
        Some(axes[0].spec.target.scale()),
        false,
    )?;
    sink.begin(&result)?;
    let mut selected = crate::analysis::results::SelectedOutput::bind(
        sink,
        &result,
        circuit.unknown_count(),
        true,
    )?;
    let mut previous = Vector::zeros(circuit.unknown_count());
    // dctrcurv.c rotates the state vectors before every point, so a point's
    // CKTstate1 is the previous point's state (the first point's own state is
    // copied in after it). Only devices with discrete state (switches) read it.
    let mut history = circuit.state_history()?;
    for hint in hints.nodesets {
        previous.as_mut_slice()[hint.row] = hint.value;
    }
    let mut point = Vec::with_capacity(result.variable_count());
    for outer_value in outer {
        // dctrcurv.c: when the inner sweep wraps to its start for the next
        // outer value, `firstTime` is set again and the mode is MODEINITJCT,
        // so the first point of every inner sweep is decided from the
        // instance flags like the very first point, and its converged state
        // then becomes the accepted history (C's `firstTime` memcpy).
        let mut first = true;
        for inner_value in inner.clone() {
            let point_context = point_context(
                circuit,
                &axes,
                [inner_value, outer_value],
                context.model_context(),
            )?;
            let model = point_context.model;
            let overrides = point_context.source_values();
            // dctrcurv.c runs NIiter at every point, linear circuits included.
            let solution = solver.solve(&point_context, &previous, &history, first)?;
            let x = solution.values;
            let trial = solution.trial;
            let first_point = std::mem::replace(&mut first, false);
            // Only a solved point is accepted; a failure above returns before
            // any accept hook runs for it. A nonlinear point's state becomes the
            // accepted history of the next one.
            let load = crate::devices::LoadRequest {
                mode: crate::devices::AnalysisMode::DcSweep,
                solution: &x,
                model_context: &model,
                integration: None,
                history: &history,
                forcing: None,
            };
            let observed = if let Some(output) = selected.as_ref() {
                solver.prepared.observe_real_selected(
                    &load,
                    overrides,
                    None,
                    Some(&trial),
                    output.observations(),
                )?
            } else {
                solver
                    .prepared
                    .observe_real(&load, overrides, None, Some(&trial))?
            };
            // dctrcurv.c `firstTime`: after the first point of an inner
            // sweep C copies CKTstate0 into CKTstate1, so after the next
            // rotation CKTstate2 equals CKTstate1 (DEVpred extrapolates
            // flat) instead of holding the previous outer value's last point.
            circuit.accept_point(&x, None, &mut history, trial)?;
            if first_point {
                history.retain_latest();
            }
            if let Some(output) = selected.as_mut() {
                output.real(sink, Some(inner_value), x.as_slice(), &observed)?;
            } else {
                point.clear();
                point.push(Complex::real(inner_value));
                point.extend(x.as_slice().iter().map(|v| Complex::real(*v)));
                point.extend(observed);
                result.check_point(&point)?;
                sink.point(&point)?;
            }
            previous = x;
        }
    }
    sink.end()
}

#[cfg(test)]
mod grid_tests {
    use super::*;

    #[test]
    fn exact_and_accumulated_grids_keep_their_distinct_endpoint_arithmetic() {
        let spec = SweepSpec {
            target: SweepTarget::VoltageSource("v1".into()),
            start: 0.3,
            stop: 0.,
            step: -0.1,
        };
        assert_eq!(spec.exact_values().unwrap().len(), 4);
        let exact = spec.grid().unwrap();
        let accumulated = spec.accumulated_grid().unwrap();
        assert_eq!(exact.len(), accumulated.len());
        assert_eq!(exact[3].to_bits(), 0_f64.to_bits());
        let expected = ((spec.start + spec.step) + spec.step) + spec.step;
        assert_eq!(accumulated[3].to_bits(), expected.to_bits());
        assert_ne!(accumulated[3].to_bits(), exact[3].to_bits());
    }

    #[test]
    fn preflight_projects_only_source_axes_out_of_the_physical_grid() {
        let grids = [
            SweepRange {
                start: 1.,
                stop: 3.,
                step: 1.,
            }
            .accumulate(false)
            .unwrap(),
            SweepRange {
                start: 10.,
                stop: 20.,
                step: 10.,
            }
            .accumulate(false)
            .unwrap(),
        ];
        assert_eq!(GridPair::new(&grids).points().count(), 6);
        assert_eq!(
            GridPair::new(&grids)
                .physical_contexts([true, false])
                .points()
                .collect::<Vec<_>>(),
            [[3., 10.], [3., 20.]]
        );
        assert_eq!(
            GridPair::new(&grids)
                .physical_contexts([false, true])
                .points()
                .collect::<Vec<_>>(),
            [[1., 20.], [2., 20.], [3., 20.]]
        );
        assert_eq!(
            GridPair::new(&grids)
                .physical_contexts([false, false])
                .points()
                .count(),
            6
        );
    }

    fn fixture() -> Circuit {
        let deck = crate::netlist::source::parse_deck_text(
            std::path::Path::new("binding.cir"),
            "binding\nv1 a 0 1\nv2 b 0 1\nr1 a 0 rm 1000\nr2 b 0 2000\nd1 0 a dm\n.model rm r(tc1=0.001)\n.model dm d\n.end\n",
        );
        let netlist = crate::netlist::Parser::new().parse_deck(&deck).unwrap();
        Circuit::from_netlist(&netlist).unwrap()
    }

    #[test]
    fn actual_sweep_binding_and_preflight_work_follows_axes_not_points() {
        use crate::devices::circuit::preparation_metrics::take;
        use crate::primitives::AnalysisKind;
        // (request, target bindings, physical validation contexts, replacement-only checks)
        for (args, bindings, physical, resistances) in [
            ("v1 0 2 1", 1, 1, 0),
            ("r1 1000 1002 1", 1, 1, 2),
            ("temp 27 29 1", 0, 3, 0),
            ("@d1[area] 1 3 1", 1, 1, 2),
            ("v1 0 3 1 temp 27 29 1", 1, 3, 0),
            ("r1 1000 1002 1 temp 27 29 1", 1, 3, 6),
            ("@d1[area] 1 3 1 v1 0 3 1", 2, 1, 2),
            ("temp 27 29 1 r1 1000 1002 1", 1, 3, 6),
            ("temp 27 29 1 @d1[area] 1 3 1", 1, 3, 6),
            ("@d1[area] 1 3 1 temp 27 29 1", 1, 3, 6),
            ("@d1[area] 1 3 1 @d1[m] 1 3 1", 2, 1, 8),
            ("@v1[dc] 0 3 1 v2 0 2 1", 2, 1, 0),
            ("r1 1000 1002 1 @r2[resistance] 2000 2002 1", 2, 1, 4),
        ] {
            let mut circuit = fixture();
            let context = AnalysisContext::default();
            let plan = parse_axes(
                &args
                    .split_whitespace()
                    .map(str::to_owned)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let prepared = circuit.prepare_run();
            take();
            let axes = bind_axes(&circuit, &plan).unwrap();
            let binding = take();
            assert_eq!(binding.bindings, bindings, "{args}: {binding:?}");
            assert!(binding.name_lookups <= 3 * bindings, "{args}: {binding:?}");
            preflight(&prepared, &axes, &plan.grids, &context).unwrap();
            let pre = take();
            assert_eq!(pre.name_lookups, 0, "{args}: {pre:?}");
            assert_eq!(pre.bindings, 0, "{args}: {pre:?}");
            assert_eq!(pre.physical_validations, physical, "{args}: {pre:?}");
            assert_eq!(pre.replacement_validations, resistances, "{args}: {pre:?}");
            assert_eq!(pre.small_signal_assemblies, 0, "{args}: {pre:?}");
            for values in GridPair::new(&plan.grids).points() {
                let point =
                    point_context(&circuit, &axes, values, context.model_context()).unwrap();
                prepared.validate_context(&point.model).unwrap();
            }
            let points = take();
            assert_eq!(points.bindings, 0, "{args}: {points:?}");
            assert_eq!(points.name_lookups, 0, "{args}: {points:?}");
            assert_eq!(points.small_signal_assemblies, 0, "{args}: {points:?}");
            // Verify the same actual solver path, not only synthetic point construction.
            drop(prepared);
            let request = crate::analysis::AnalysisInput::with_arguments(
                AnalysisKind::DcSweep,
                args.split_whitespace(),
            )
            .resolve()
            .unwrap();
            let plot = request.run(&mut circuit, &context).unwrap();
            let run = take();
            assert_eq!(run.bindings, bindings, "{args}: {run:?}");
            assert_eq!(run.name_lookups, binding.name_lookups, "{args}: {run:?}");
            assert_eq!(run.replacement_validations, resistances, "{args}: {run:?}");
            assert_eq!(run.small_signal_assemblies, 0, "{args}: {run:?}");
            assert_eq!(run.physical_validations, physical, "{args}: {run:?}");
            assert_eq!(
                plot.point_count(),
                GridPair::new(&plan.grids).points().count()
            );
            println!(
                "{args}: bind={binding:?}; preflight={pre:?}; point_preparation={points:?}; run={run:?}"
            );
        }
    }

    #[test]
    fn bound_handles_replay_values_and_reject_foreign_topologies() {
        let mut a = fixture();
        let b = fixture();
        let resistance = a.bind_resistor_target("R1").unwrap();
        let area = a.bind_instance_target("D1", "AREA").unwrap();
        let voltage = a.bind_source_target("V1").unwrap();
        let base = ModelContext::default();
        let run = a.prepare_run();
        for value in [1., 2., 1.] {
            let context = base
                .with_resistor_override(resistance.at_resistance(1000. * value).unwrap())
                .unwrap()
                .with_instance_override(area.at(value).unwrap())
                .unwrap()
                .with_instance_override(voltage.at(value).unwrap())
                .unwrap();
            run.validate_dc_inputs(&context).unwrap();
            assert!(b.prepare_run().validate_dc_inputs(&context).is_err());
            assert_eq!(a.resistor("r1").unwrap().1.supplied, 1000.);
        }
        drop(run);
        let node = a.nodes().get("a").unwrap();
        a.add_device(Box::new(
            crate::devices::Resistor::new("r_stale", [node, crate::primitives::NodeId::GROUND], 1.)
                .unwrap(),
        ))
        .unwrap();
        a.finalize().unwrap();
        let stale = base
            .with_instance_override(voltage.at(1.).unwrap())
            .unwrap();
        assert!(a.prepare_run().validate_context(&stale).is_err());
        assert!(resistance.at_resistance(0.).is_err());
        assert!(area.at(Real::NAN).is_err());
        assert!(voltage.at_resistance(1.).is_err());
    }

    #[test]
    fn kelvin_accumulation_rejects_sub_ulp_temperature_steps() {
        let spec = SweepSpec {
            target: SweepTarget::Temperature,
            start: 0.,
            stop: 1e-12,
            step: 1e-15,
        };
        assert_eq!(spec.grid().unwrap().len(), 1001);
        let error = spec.accumulated_grid().unwrap_err();
        assert!(error.to_string().contains("makes no progress"));
    }
}
