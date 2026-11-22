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
//!    element `delta_Y = -dY0 + dY1` and `delta_I = -dI0 + dI1` in C's
//!    accumulation order. The sensitivity is `Y^-1 (delta_I - delta_Y x)`
//!    read at the output and divided by `delta`, with `delta_Y x` summed per
//!    row in the internal column order of C's `delta_Y` (see
//!    [`DeltaNumbering`]).
//!
//! The device side, including C's side effects of the in-place perturbation,
//! lives in [`crate::devices::sensitivity`]; see `docs/port/SENSITIVITY.md`.

use std::collections::BTreeMap;

use crate::analysis::linear::{number, unsupported};
use crate::analysis::results::{PlotFlags, Variable};
use crate::analysis::{AnalysisContext, AnalysisRequest, Plot};
use crate::devices::linear::LinearSystem;
use crate::devices::sensitivity::{
    AcReset, DeviceSensitivity, ParameterScope, RecordPair, SensitivityFamily, SensitivityLoad,
    SensitivityMode, SensitivityRecord,
};
use crate::devices::{AnalysisMode, Circuit, Device, IterationPhase, LoadRequest, TrialState};
use crate::maths::complex::ComplexMatrix;
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{
    Complex, NodeId, NodeKind, NodeTable, Real, SpiceError, SpiceResult, parse_spice_number,
};

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

/// The `.sens` output variable.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Output {
    /// `v(pos)` or `v(pos,neg)`.
    Voltage { pos: String, neg: Option<String> },
    /// `i(source)`.
    Current { source: String },
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
struct SensCard {
    output: Output,
    /// Name filters (`Sens_filter`), lowercased; empty for every parameter.
    filters: Vec<String>,
    /// `None` for DC.
    sweep: Option<Sweep>,
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
fn parse(arguments: &[String]) -> SpiceResult<SensCard> {
    let mut tokens = arguments.iter().map(String::as_str).peekable();
    let expect = |wanted: &str, token: Option<&str>| match token {
        Some(text) if text == wanted => Ok(()),
        Some(text) => Err(syntax(format!("expected '{wanted}', found '{text}'"))),
        None => Err(syntax(format!("is missing '{wanted}'"))),
    };
    let name = |token: Option<&str>, what: &str| -> SpiceResult<String> {
        match token {
            Some(text) if !matches!(text, "(" | ")" | ",") => Ok(text.to_ascii_lowercase()),
            Some(text) => Err(syntax(format!("expected {what}, found '{text}'"))),
            None => Err(syntax(format!("is missing {what}"))),
        }
    };
    let kind = tokens
        .next()
        .ok_or_else(|| syntax("has no output variable"))?
        .to_ascii_lowercase();
    let output = match kind.as_str() {
        "v" => {
            expect("(", tokens.next())?;
            let pos = name(tokens.next(), "an output node")?;
            if tokens.peek().is_some_and(|token| *token == ",") {
                tokens.next();
            }
            let neg = if tokens.peek().is_some_and(|token| *token == ")") {
                None
            } else {
                Some(name(tokens.next(), "the negative output node")?)
            };
            expect(")", tokens.next())?;
            Output::Voltage { pos, neg }
        }
        "i" => {
            expect("(", tokens.next())?;
            let source = name(tokens.next(), "an output source")?;
            expect(")", tokens.next())?;
            Output::Current { source }
        }
        other => {
            return Err(syntax(format!(
                "output '{other}' is neither a voltage v(...) nor a current i(...)"
            )));
        }
    };
    let mut filters = Vec::new();
    let mut sweep = None;
    while let Some(token) = tokens.next() {
        let lower = token.to_ascii_lowercase();
        match lower.as_str() {
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
    })
}

/// The frequencies of an AC sweep, exactly as `count_steps()` counts them and
/// `inc_freq()` steps them in `cktsens.c`. Both carry C defects the port
/// reproduces: `inc_freq()` compares the step type with the noise/distortion
/// `LINEAR` constant (3) instead of `SENS_LINEAR`, so a `lin` sweep
/// *multiplies* by its step `(fstop - fstart)/pts`; and the octave count
/// divides `ln(fstop/fstart)` by `M_LOG2E` instead of multiplying.
#[expect(
    clippy::suboptimal_flops,
    reason = "preserve C count_steps and inc_freq rounding, including pow and truncation"
)]
fn frequencies(sweep: &Sweep) -> SpiceResult<Vec<Real>> {
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
    let mut frequencies = Vec::with_capacity(count);
    let mut frequency = sweep.start;
    for _ in 0..count {
        if !(frequency.is_finite() && frequency > 0.) {
            return Err(unsupported(format!(
                ".sens AC frequency {frequency} is not positive and finite (C would \
                 silently use the DC operating-point matrix for a zero frequency)"
            )));
        }
        frequencies.push(frequency);
        frequency *= factor;
    }
    Ok(frequencies)
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

/// Where the output is read.
#[derive(Debug, Clone, Copy)]
enum Probe {
    Voltage {
        pos: Option<usize>,
        neg: Option<usize>,
    },
    Current {
        row: usize,
    },
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

    fn read<T: Copy + std::ops::Sub<Output = T>>(&self, x: &[T], zero: T) -> T {
        let at = |row: Option<usize>| row.and_then(|row| x.get(row).copied()).unwrap_or(zero);
        match *self {
            Self::Voltage { pos, neg } => at(pos) - at(neg),
            Self::Current { row } => at(Some(row)),
        }
    }
}

fn not_found(message: impl std::fmt::Display) -> SpiceError {
    SpiceError::Circuit {
        message: format!("{message} (C: sens_sens in cktsens.c)"),
    }
}

fn node_row(circuit: &Circuit, name: &str) -> SpiceResult<Option<usize>> {
    let canonical = NodeTable::canonical_name(name, circuit.nodes().auto_gnd());
    let id = circuit.nodes().get(&canonical).ok_or_else(|| {
        not_found(format_args!(
            ".sens output node {name} is not in the circuit"
        ))
    })?;
    let row = circuit.unknowns().node_row(id);
    let ground = circuit
        .nodes()
        .node(id)
        .is_some_and(|node| node.kind == NodeKind::Ground);
    if row.is_none() && !ground {
        return Err(not_found(format_args!(
            ".sens output node {name} has no matrix row"
        )));
    }
    Ok(row)
}

fn resolve_output(circuit: &Circuit, output: &Output) -> SpiceResult<Probe> {
    match output {
        Output::Voltage { pos, neg } => Ok(Probe::Voltage {
            pos: node_row(circuit, pos)?,
            neg: match neg {
                Some(neg) => node_row(circuit, neg)?,
                None => None,
            },
        }),
        Output::Current { source } => {
            let device = circuit
                .devices()
                .iter()
                .position(|device| device.name().eq_ignore_ascii_case(source))
                .ok_or_else(|| {
                    not_found(format_args!(
                        ".sens output source {source} is not in the circuit"
                    ))
                })?;
            let branch = circuit.devices()[device].findable_branch().ok_or_else(|| {
                unsupported(format!(
                    ".sens i({source}): the device has no findable branch current (only \
                     independent voltage sources, E/H sources and voltage B sources can be \
                     sensed, as by CKTfndBranch; C would silently read ground)"
                ))
            })?;
            let row = circuit
                .branch_rows(device)
                .and_then(|mut rows| rows.nth(branch))
                .ok_or_else(|| not_found(format_args!("{source}: missing branch row")))?;
            Ok(Probe::Current { row })
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
    #[expect(
        clippy::cast_precision_loss,
        reason = "in-memory matrix row indices are stored in the C real-valued sensitivity record"
    )]
    {
        instance.store("row1", coupling.rows[0] as Real)?;
        instance.store("row2", coupling.rows[1] as Real)
    }
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
        let (model_record, instance_record) = sensitivity.records()?;
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

/// The numbering of C's perturbation matrix `delta_Y` (`sens_sens()` in
/// `cktsens.c`). SPARSE creates it once with `TRANSLATE`: an external row or
/// column gets the next internal number the first time an element names it
/// (`Translate()` in `spbuild.c`, row before column), and the first names
/// come from the `DEVsetup` that `sens_sens()` runs on `delta_Y` for each
/// device in `sgen` order. `SMPmultiply()` (`spMultiply()`) sums each row of
/// `delta_Y * E` in that internal column order.
struct DeltaNumbering {
    internal: Vec<Option<usize>>,
    next: usize,
    ordered: Vec<(usize, usize, usize, Complex)>,
}

impl DeltaNumbering {
    fn new(n: usize) -> Self {
        Self {
            internal: vec![None; n],
            next: 0,
            ordered: Vec::new(),
        }
    }

    fn translate_one(&mut self, external: usize) {
        if let Some(slot) = self.internal.get_mut(external)
            && slot.is_none()
        {
            *slot = Some(self.next);
            self.next += 1;
        }
    }

    /// `Translate()` of each element in order.
    fn translate(&mut self, elements: impl IntoIterator<Item = (usize, usize)>) {
        for (row, col) in elements {
            self.translate_one(row);
            self.translate_one(col);
        }
    }

    /// `SMPmultiply`'s row products in internal SPARSE column order. Reuse
    /// one entry buffer across the parameter walk, visiting even empty rows
    /// so that subtraction from the RHS preserves signed-zero arithmetic.
    fn visit_products(
        &mut self,
        elements: BTreeMap<(usize, usize), Complex>,
        x: &[Complex],
        mut visit: impl FnMut(usize, Complex),
    ) -> SpiceResult<()> {
        self.ordered.clear();
        for ((row, col), value) in elements {
            let internal = self
                .internal
                .get(col)
                .copied()
                .flatten()
                .filter(|_| row < x.len() && col < x.len())
                .ok_or_else(|| SpiceError::Numerical {
                    context: ".sens".into(),
                    message: format!("invalid or untranslated perturbation entry ({row}, {col})"),
                })?;
            self.ordered.push((row, internal, col, value));
        }
        self.ordered
            .sort_unstable_by_key(|&(row, internal, _, _)| (row, internal));
        let mut entries = self.ordered.iter().copied().peekable();
        for row in 0..x.len() {
            let mut sum = Complex::ZERO;
            while let Some((_, _, col, a)) = entries.next_if(|entry| entry.0 == row) {
                // Complex multiplication and addition retain C's separate
                // products, subtraction/addition, and row accumulation.
                sum = sum + a * x[col];
            }
            visit(row, sum);
        }
        Ok(())
    }
}

/// One `delta_Y` element as C accumulates it: the base load's contributions
/// summed in stamp order from zero, the sum negated (`SMPconstMult(delta_Y,
/// -1)`), then the perturbed load's contributions added in stamp order. The
/// real and imaginary parts accumulate separately, as the loads stamp them.
fn accumulate(
    base: impl IntoIterator<Item = (usize, usize, Complex)>,
    perturbed: impl IntoIterator<Item = (usize, usize, Complex)>,
) -> BTreeMap<(usize, usize), Complex> {
    let mut sums: BTreeMap<(usize, usize), Complex> = BTreeMap::new();
    for (row, col, value) in base {
        let entry = sums.entry((row, col)).or_insert(Complex::ZERO);
        entry.re += value.re;
        entry.im += value.im;
    }
    for value in sums.values_mut() {
        value.re *= -1.;
        value.im *= -1.;
    }
    for (row, col, value) in perturbed {
        // A fresh element was cleared to zero and negated with the rest.
        let entry = sums.entry((row, col)).or_insert(Complex::new(-0., -0.));
        entry.re += value.re;
        entry.im += value.im;
    }
    sums
}

fn real_stamps(matrix: &SparseMatrix) -> impl Iterator<Item = (usize, usize, Complex)> + '_ {
    matrix
        .triplets()
        .iter()
        .map(|t| (t.row, t.col, Complex::real(t.value)))
}

/// The device a sensitivity load stamps.
fn stand_in<'d>(circuit: &'d Circuit, index: usize, load: &'d SensitivityLoad) -> &'d dyn Device {
    match load {
        SensitivityLoad::Original => &*circuit.devices()[index],
        SensitivityLoad::Replacement(device) => &**device,
    }
}

/// The connected block of every unknown of `matrix` (union-find over its
/// entries).
fn blocks(matrix: &SparseMatrix) -> Vec<usize> {
    const fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut parent: Vec<usize> = (0..matrix.rows()).collect();
    for triplet in matrix.triplets() {
        let (a, b) = (
            root(&mut parent, triplet.row),
            root(&mut parent, triplet.col),
        );
        if a != b {
            parent[a] = b;
        }
    }
    (0..parent.len()).map(|x| root(&mut parent, x)).collect()
}

/// `NIacIter` at one frequency: the reset system factored from scratch and
/// solved.
fn ac_state<'a>(
    circuit: &'a Circuit,
    system: &LinearSystem,
    omega: Real,
    model: crate::devices::ModelContext,
    zeros: &'a Vector,
    state: &'a [Real],
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

/// The unknowns in a coupling block of `system` (at `omega`) that holds a
/// non-finite entry; empty when every entry is finite.
fn poisoned_rows(system: &LinearSystem, omega: Real) -> std::collections::BTreeSet<usize> {
    let n = system.a.rows();
    let mut structure = SparseMatrix::new(n, n);
    let mut bad = Vec::new();
    let dynamic = crate::maths::complex::dynamic_terms(&system.e, &system.e_groups, omega);
    let entries = system
        .a
        .triplets()
        .iter()
        .map(|t| (t.row, t.col, t.value))
        .chain(dynamic);
    for (row, col, value) in entries {
        if structure.add(row, col, 1.).is_err() {
            continue;
        }
        if !value.is_finite() {
            bad.push(row);
        }
    }
    if bad.is_empty() {
        return std::collections::BTreeSet::new();
    }
    let blocks = blocks(&structure);
    let tainted: std::collections::BTreeSet<usize> = bad.iter().map(|&row| blocks[row]).collect();
    (0..n)
        .filter(|&row| tainted.contains(&blocks[row]))
        .collect()
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
    fn delta(
        &self,
        numbering: &mut DeltaNumbering,
        index: usize,
        base: &SensitivityLoad,
        perturbed: &SensitivityLoad,
    ) -> SpiceResult<Vector> {
        let n = self.circuit.unknown_count();
        let (mut y0, mut y1) = (SparseMatrix::new(n, n), SparseMatrix::new(n, n));
        let mut rhs = Vector::zeros(n);
        self.load(index, base, &mut y0, &mut rhs)?;
        // delta_I *= -1, then the perturbed load adds to it in place.
        for value in rhs.as_mut_slice() {
            *value *= -1.;
        }
        self.load(index, perturbed, &mut y1, &mut rhs)?;
        numbering.translate(
            y0.triplets()
                .iter()
                .chain(y1.triplets())
                .map(|t| (t.row, t.col)),
        );
        let elements = accumulate(real_stamps(&y0), real_stamps(&y1));
        numbering.visit_products(elements, &self.solution, |row, product| {
            rhs.as_mut_slice()[row] -= product.re;
        })?;
        // Nonfinite loads poison the entire connected block, as in C.
        // Allocate the block mask only when a nonfinite entry actually occurs.
        let mut tainted = None;
        for (row, value) in rhs.as_mut_slice().iter_mut().enumerate() {
            if !value.is_finite() {
                tainted.get_or_insert_with(|| vec![false; n])[self.blocks[row]] = true;
                *value = 0.;
            }
        }
        let mut solved =
            crate::maths::solve_last_factors(&self.pattern, &rhs)?.ok_or_else(|| {
                SpiceError::Numerical {
                    context: ".sens".to_owned(),
                    message: "the operating point left no factored circuit matrix".to_owned(),
                }
            })?;
        if let Some(tainted) = tainted {
            for (row, value) in solved.as_mut_slice().iter_mut().enumerate() {
                if tainted[self.blocks[row]] {
                    *value = Real::NAN;
                }
            }
        }
        Ok(solved)
    }
}

/// The AC state at one frequency.
struct AcState<'a> {
    omega: Real,
    factors: crate::maths::complex::ComplexLu,
    solution: Vec<Complex>,
    model: crate::devices::ModelContext,
    bias: &'a Vector,
    state: &'a [Real],
    circuit: &'a Circuit,
}

impl AcState<'_> {
    fn assemble(&self, index: usize, load: &SensitivityLoad) -> SpiceResult<LinearSystem> {
        let mut system = LinearSystem::new(self.circuit.unknown_count());
        self.circuit.assemble_small_signal_device(
            index,
            stand_in(self.circuit, index, load),
            &self.model,
            self.bias,
            Some(self.state),
            &mut system,
        )?;
        Ok(system)
    }

    /// The complex stamps of one `DEVacLoad`: the real contributions and
    /// `omega` times the dynamic ones, each in stamp order.
    fn stamps(&self, system: &LinearSystem) -> Vec<(usize, usize, Complex)> {
        real_stamps(&system.a)
            .chain(
                crate::maths::complex::dynamic_terms(&system.e, &system.e_groups, self.omega)
                    .into_iter()
                    .map(|(row, col, value)| (row, col, Complex::new(0., value))),
            )
            .collect()
    }

    fn delta(
        &self,
        numbering: &mut DeltaNumbering,
        index: usize,
        base: &SensitivityLoad,
        perturbed: &SensitivityLoad,
    ) -> SpiceResult<Vec<Complex>> {
        let s0 = self.assemble(index, base)?;
        let s1 = self.assemble(index, perturbed)?;
        // -(base) + perturbed, part by part (one source stamp per row).
        let mut rhs = s0.ac_rhs();
        for (base, perturbed) in rhs.iter_mut().zip(s1.ac_rhs()) {
            *base = Complex::new(-base.re + perturbed.re, -base.im + perturbed.im);
        }
        let (t0, t1) = (self.stamps(&s0), self.stamps(&s1));
        numbering.translate(t0.iter().chain(&t1).map(|&(row, col, _)| (row, col)));
        let elements = accumulate(t0, t1);
        numbering.visit_products(elements, &self.solution, |row, product| {
            rhs[row].re -= product.re;
            rhs[row].im -= product.im;
        })?;
        self.factors.solve(&rhs)
    }
}

/// Runs `.sens` on `circuit`; see the module documentation.
///
/// # Errors
/// Malformed cards, unknown outputs, devices whose sensitivity is not ported,
/// operating-point failures and singular or nonfinite solves.
pub(super) fn run(
    circuit: &mut Circuit,
    request: &AnalysisRequest,
    context: &AnalysisContext,
) -> SpiceResult<Plot> {
    // Named `key=value` arguments are the operating point's Newton and
    // continuation settings (the deck's `.options reltol=...`, which C's
    // CKTop for `.sens` honours like any other analysis); DcSettings rejects
    // any it does not know, so the card itself is the positional rest.
    let settings = crate::analysis::bias::DcSettings::from_request(request)?;
    let positional: Vec<String> = request
        .arguments
        .iter()
        .filter(|argument| !argument.contains('='))
        .cloned()
        .collect();
    let card = parse(&positional)?;
    let grid = card.sweep.as_ref().map(frequencies).transpose()?;
    let ac = grid.is_some();
    circuit.finalize()?;
    let circuit: &Circuit = circuit;
    let probe = resolve_output(circuit, &card.output)?;
    let model = context.model_context();

    let mut workspace = ParameterWalk::new(circuit, &model, ac, &card.filters)?;
    let earlier = if ac {
        earlier_ac_state(circuit, &workspace.members)?
    } else {
        EarlierAcState::default()
    };
    // The operating point, exactly as `.op` solves it (`CKTop`).
    let solved = crate::analysis::bias::OperatingPointStart::resolve(circuit, request)?
        .solve(circuit, &model, &settings)?;

    let mut plot = workspace.plot(ac);
    if let Some(grid) = grid {
        run_ac(
            &mut workspace,
            &earlier,
            &solved.values,
            model,
            probe,
            grid,
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
    Ok(plot)
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
    numbering: DeltaNumbering,
}

impl<'a> ParameterWalk<'a> {
    fn new(
        circuit: &'a Circuit,
        model: &crate::devices::ModelContext,
        ac: bool,
        filters: &[String],
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
            numbering: DeltaNumbering::new(circuit.unknown_count()),
        })
    }

    fn plot(&self, ac: bool) -> Plot {
        let mut plot = Plot::new(
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
        plot
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
    state: Vec<Real>,
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
    ) -> SpiceResult<LinearSystem> {
        let mut system = LinearSystem::new(walk.circuit.unknown_count());
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
                &mut system,
            )?;
        }
        Ok(system)
    }
}

fn run_ac(
    walk: &mut ParameterWalk<'_>,
    earlier: &EarlierAcState,
    bias: &Vector,
    model: crate::devices::ModelContext,
    probe: Probe,
    grid: Vec<Real>,
    plot: &mut Plot,
) -> SpiceResult<()> {
    let circuit = walk.circuit;
    let sweep = AcSweep {
        earlier,
        bias,
        model,
        zeros: Vector::zeros(circuit.unknown_count()),
        state: vec![0.; circuit.state_history().len()],
    };
    for frequency in grid {
        let omega = 2. * std::f64::consts::PI * frequency;
        let reset = walk.reset_ac()?;
        let system = sweep.assemble(walk, &reset, omega)?;
        // A device whose present records C loads into a non-finite
        // matrix (BSIM4's TNOM drifting by 273.15 K per frequency, see
        // `cktsens.c` and `b4mask.c`) poisons NIacIter's solve: every
        // unknown of a block holding such an entry is NaN, so is every
        // column read from it, and the walk still perturbs and restores
        // each parameter. A poisoned block the output does not read
        // would leave C's other blocks finite; that is refused.
        let poisoned = poisoned_rows(&system, omega);
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
        let mut point = Vec::with_capacity(plot.variables.len());
        point.push(Complex::real(frequency));
        walk.sweep(
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
            |numbering, index, base, perturbed, delta| {
                let Some(at) = &at else {
                    point.push(Complex::new(Real::NAN, Real::NAN));
                    return Ok(());
                };
                let dx = at.delta(numbering, index, base, perturbed)?;
                let value = probe.read(&dx, Complex::ZERO);
                point.push(Complex::new(value.re / delta, value.im / delta));
                Ok(())
            },
        )?;
        plot.push_point(point)?;
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
    let history = circuit.state_history();
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
        blocks: blocks(&jacobian),
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
    let mut point = Vec::with_capacity(plot.variables.len());
    walk.sweep(
        |_, member, setup, current| member.sensitivity.load(setup, current, SensitivityMode::Dc),
        |numbering, index, base, perturbed, delta| {
            let dx = at.delta(numbering, index, base, perturbed)?;
            point.push(Complex::real(probe.read(dx.as_slice(), 0.) / delta));
            Ok(())
        },
    )?;
    plot.push_point(point)?;
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
            Ok(pairs
                .into_iter()
                .filter_map(|(row, col)| {
                    Some((
                        circuit.unknowns().node_row(row)?,
                        circuit.unknowns().node_row(col)?,
                    ))
                })
                .collect())
        },
    )
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
            &mut DeltaNumbering,
            usize,
            &SensitivityLoad,
            &SensitivityLoad,
            Real,
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
            let elements = device.setup_elements(setup);
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
            let measured = (|| {
                self.numbering
                    .translate(setup_elements(self.circuit, member.index, elements)?);
                measure(&mut self.numbering, member.index, &base, &perturbed, delta)
            })();
            measured.map_err(|error| SpiceError::Numerical {
                context: format!(".sens {label}: {error}"),
                message: "the perturbed load could not be evaluated".to_owned(),
            })?;
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

    #[test]
    fn perturbation_products_follow_internal_column_order_and_clear_workspace() {
        use super::{DeltaNumbering, accumulate};
        use crate::primitives::Complex;

        let mut numbering = DeltaNumbering::new(3);
        numbering.translate([(2, 2), (0, 0), (1, 1)]);
        let elements = accumulate(
            [],
            [
                (0, 0, Complex::real(-1e16)),
                (0, 1, Complex::real(1.)),
                (0, 2, Complex::real(1e16)),
            ],
        );
        let x = [Complex::real(1.); 3];
        let mut result = [Complex::ZERO; 3];
        numbering
            .visit_products(elements, &x, |row, sum| result[row] = sum)
            .unwrap();
        assert_eq!(result, [Complex::real(1.), Complex::ZERO, Complex::ZERO]);
        numbering
            .visit_products(accumulate([], []), &x, |row, sum| result[row] = sum)
            .unwrap();
        assert_eq!(result, [Complex::ZERO; 3]);
    }

    #[test]
    fn perturbation_products_reject_untranslated_or_invalid_entries() {
        use super::{DeltaNumbering, accumulate};
        use crate::primitives::Complex;
        let mut numbering = DeltaNumbering::new(2);
        numbering.translate([(0, 0)]);
        let x = [Complex::real(1.); 2];
        for entry in [(0, 1), (0, 2), (2, 0)] {
            let elements = accumulate([], [(entry.0, entry.1, Complex::real(1.))]);
            assert!(numbering.visit_products(elements, &x, |_, _| {}).is_err());
        }
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
    fn sweeps_step_like_count_steps_and_inc_freq() {
        let sweep = |text: &str| {
            parse(&tokens(&format!("v ( a ) ac {text}")))
                .unwrap()
                .sweep
                .unwrap()
        };
        let f = frequencies(&sweep("dec 1 1k 100k")).unwrap();
        assert_eq!(f.len(), 3);
        assert_eq!(f[0], 1e3);
        assert!((f[2] / 1e5 - 1.).abs() < 1e-12);
        // `lin` multiplies by (fstop - fstart)/pts (C's inc_freq defect).
        assert_eq!(frequencies(&sweep("lin 2 1k 2k")).unwrap(), [1e3, 5e5]);
        // `oct` divides by M_LOG2E: 1 per octave over 1..16 gives 2 points.
        assert_eq!(frequencies(&sweep("oct 1 1 16")).unwrap().len(), 2);
        assert!(frequencies(&sweep("lin 2 1k 1k")).is_err());
        assert!(frequencies(&sweep("dec 1 0 1k")).is_err());
    }
}
