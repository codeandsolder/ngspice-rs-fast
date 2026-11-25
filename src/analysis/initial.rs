//! `.ic`, `.nodeset`, instance `ic=` and `uic` initialization (GitHub #27).
//!
//! C behaviour, all read from the reference sources and re-checked against the
//! reference binary (see `docs/port/TRANSIENT.md` for the table):
//!
//! * `cktsetnp.c`/`inppas3.c` store one `ic` and one `nodeset` per node; every
//!   entry overwrites the previous one, so the **last duplicate wins**. An
//!   unknown node only produces a warning there; the port reports it as an
//!   error ([`resolve`]).
//! * `cktload.c` enforces `.ic` only while `MODETRANOP` and not `MODEUIC`, i.e.
//!   during the initial operating point of an ordinary `.tran`. Each constrained
//!   node row is zeroed in place (`ZeroNoncurRow`) and replaced by `v = ic`, or,
//!   when the row also holds a branch-current entry (a node on a voltage source,
//!   inductor or E/H output), by a `1e10` conductance. The port loads exactly
//!   these rows ([`crate::analysis::bias::NodeForcing`]);
//!   [`irredundant_constraints`] only rejects entries that contradict ideal
//!   sources, where C would return a meaningless compromise.
//! * A node held by an E/H output (directly or through other ideal relations)
//!   keeps the controlled-source relation in C (the `1e10` conductance only
//!   loads the ideal output). [`check_implied`] compares such `.ic` entries
//!   with the solved bias, and under `uic` a capacitor across such an output
//!   is checked the same way.
//! * `.nodeset` is stamped by `cktload.c` only in the first (`MODEINITJCT`,
//!   `MODEINITFIX`) iterations of a DC iteration and then released; it only
//!   steers convergence (in a linear circuit it changes rounding, as in C).
//!   Every solve forces it through [`crate::analysis::bias::NodeForcing`]
//!   ([`forced_nodesets`]).
//! * Nonlinear `.ic` without `uic` uses the same row replacement in every
//!   Newton load of the transient operating point (C `CKTop` in
//!   `MODETRANOP`), after [`irredundant_constraints`], and
//!   instance `ic=`/`off` of D/Q/M devices act through their loads
//!   (`crate::devices::limiting`). Under `uic` the one load C performs is the
//!   device-flagged initial load of [`uic_start`]'s vector.
//! * With `uic`, `NIiter()` returns right after one `CKTload` (no solve at all),
//!   `CKTic()` first copies `.nodeset` and then `.ic` values into the node
//!   vector, `CAPgetic()` derives an unset capacitor `ic` from those node values
//!   and `indload.c` takes the inductor current from `ic=` (default 0). Charges
//!   and fluxes therefore come only from the capacitor/inductor initial values;
//!   the first timepoint is solved normally and no `t = 0` row is written.
//!
//! The analyses of this module never touch a circuit: they return values.

use std::collections::BTreeMap;

use crate::devices::{Circuit, IterationPhase, LoadRequest, ModelContext, PreparedRun};
use crate::maths::{SparseMatrix, Vector};
use crate::primitives::{Real, SourceLoc, SpiceError, SpiceResult};

use crate::analysis::{AnalysisPlan, NodeCondition};

mod constraints;
mod system;
use constraints::{AbsoluteRow, ConstraintGraph, Join, Origin, VoltageVertex};
use system::{InstantaneousSolution, InstantaneousSystem};

fn failure(message: impl Into<String>) -> SpiceError {
    SpiceError::Numerical {
        context: "initial conditions".to_owned(),
        message: message.into(),
    }
}

/// One validated hint on an MNA node row.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RowHint {
    pub row: usize,
    pub node: String,
    pub value: Real,
    pub location: SourceLoc,
}

/// The request's `.ic`/`.nodeset` entries bound to matrix rows, one per node
/// (last duplicate wins), ordered by row.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Hints {
    pub initial: Vec<RowHint>,
    pub nodesets: Vec<RowHint>,
}

/// Binds the request's hints to `circuit`.
///
/// # Errors
/// A node that is not part of the circuit (C: warning "IC on non-existent node
/// ... ignored") or that is ground.
pub(super) fn resolve(circuit: &Circuit, request: &AnalysisPlan) -> SpiceResult<Hints> {
    Ok(Hints {
        initial: bind(circuit, request.initial_conditions(), ".ic")?,
        nodesets: bind(circuit, request.nodesets(), ".nodeset")?,
    })
}

fn bind(circuit: &Circuit, entries: &[NodeCondition], card: &str) -> SpiceResult<Vec<RowHint>> {
    let mut by_row = BTreeMap::new();
    for entry in entries {
        let row = circuit
            .nodes()
            .get(&entry.node)
            .ok_or_else(|| {
                SpiceError::parse(
                    entry.location.clone(),
                    format!(
                        "{card} V({}) names a node that does not exist in the circuit \
                         (ngspice only warns and ignores it; this port rejects it)",
                        entry.node
                    ),
                )
            })
            .and_then(|id| {
                circuit.unknowns().checked_node_row(id)?.ok_or_else(|| {
                    SpiceError::parse(
                        entry.location.clone(),
                        format!("{card} V({}) names the ground node", entry.node),
                    )
                })
            })?;
        by_row.insert(
            row,
            RowHint {
                row,
                node: entry.node.clone(),
                value: entry.value,
                location: entry.location.clone(),
            },
        );
    }
    Ok(by_row.into_values().collect())
}

/// Relative and absolute voltage tolerances (`reltol`, `vntol`).
#[derive(Debug, Clone, Copy)]
pub(super) struct VoltageTolerance {
    pub reltol: Real,
    pub vntol: Real,
}

impl VoltageTolerance {
    fn agree(self, a: Real, b: Real) -> bool {
        let relative_limit = self.reltol * a.abs().max(b.abs());
        (a - b).abs() <= self.vntol + relative_limit
    }
}

/// The `.ic` constraints of the initial bias, split by how they are handled.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct Constraints {
    /// Entries no ideal relation fixes.
    pub imposed: Vec<RowHint>,
    /// Nodes already fixed by an E/H output relation (whose value is only
    /// known after the solve): C keeps the controlled source's relation (its
    /// `1e10` conductance just loads the ideal output), so [`check_implied`]
    /// compares the entry with the solution.
    implied: Vec<ImpliedHint>,
}

#[derive(Debug, Clone, PartialEq)]
struct ImpliedHint {
    hint: RowHint,
    provenance: String,
}

/// Checks each implied `.ic` entry against the bias solution `x`.
///
/// # Errors
/// An entry that contradicts the value the controlled sources force.
pub(super) fn check_implied(
    x: &Vector,
    constraints: &Constraints,
    tolerance: VoltageTolerance,
) -> SpiceResult<()> {
    for ImpliedHint { hint, provenance } in &constraints.implied {
        let value = x.as_slice()[hint.row];
        if !tolerance.agree(value, hint.value) {
            return Err(failure(format!(
                ".ic V({})={} at {} contradicts the {value} V that an ideal controlled \
                 source (E/H) output forces on this node, directly or through other ideal \
                 relations (path: {provenance}); ngspice would return a meaningless 1e10-conductance compromise",
                hint.node, hint.value, hint.location
            )));
        }
    }
    Ok(())
}

/// Classifies the `.ic` constraints of the operating point `A x = rhs`, in
/// entry order. Every entry is still loaded, as C loads it.
///
/// A node whose voltage is already fixed by ideal voltage sources and
/// inductors (shorts at DC) chained to ground, or to another constrained node,
/// is determined by the circuit: the entry is left out of both lists when it
/// agrees with that value within `tolerance` and is an error when it does not.
/// Every other entry is [`Constraints::imposed`]. `rhs` holds the source values
/// at the `t = 0` left limit.
///
/// A node tied to ground (or to a constrained node) through a chain that
/// includes an E/H output is determined by the circuit too, but the value is
/// only known after the solve: such entries are returned as
/// [`Constraints::implied`] and checked by [`check_implied`].
///
/// # Errors
/// An `.ic` value contradicting an ideal source (or two `.ic` values on nodes
/// rigidly tied by sources). C produces a `1e10`-weighted compromise there.
pub(super) fn irredundant_constraints(
    circuit: &Circuit,
    rhs: &Vector,
    hints: &[RowHint],
    tolerance: VoltageTolerance,
) -> SpiceResult<Constraints> {
    if hints.is_empty() {
        return Ok(Constraints::default());
    }
    let mut graph = ConstraintGraph::for_circuit(circuit, rhs, true)?;
    let mut kept = Constraints::default();
    for hint in hints {
        let vertex = VoltageVertex::Node(AbsoluteRow(hint.row));
        if graph.controlled_implies(vertex, VoltageVertex::Ground) {
            kept.implied.push(ImpliedHint {
                hint: hint.clone(),
                provenance: graph.controlled_path(vertex, VoltageVertex::Ground)?,
            });
            continue;
        }
        match graph.join(
            vertex,
            VoltageVertex::Ground,
            hint.value,
            Origin::Hint(hint),
        )? {
            Join::Tree => kept.imposed.push(hint.clone()),
            Join::Cycle { expected } if tolerance.agree(expected, hint.value) => {}
            Join::Cycle { expected } => {
                return Err(failure(format!(
                    ".ic V({})={} at {} contradicts the {expected} V that ideal voltage sources \
                     and inductors (DC shorts) force on this node, directly or through other \
                     .ic entries (path: {}); ngspice would return a meaningless 1e10-conductance \
                     compromise",
                    hint.node,
                    hint.value,
                    hint.location,
                    graph.path_description()
                )));
            }
        }
    }
    Ok(kept)
}

/// The `.nodeset` rows a DC solve forces in its `MODEINITJCT`/`MODEINITFIX`
/// loads ([`crate::analysis::bias::NodeForcing::nodesets`]): every entry, as
/// `cktload.c` loads them. A node a source or an E/H output also ties gets
/// C's `1e10` row ([`crate::analysis::bias::NodeForcing`]); an `.ic` on the
/// same node overrides the nodeset there, as C's later `.ic` loop does.
pub(super) fn forced_nodesets(nodesets: &[RowHint]) -> Vec<(usize, Real)> {
    nodesets.iter().map(|hint| (hint.row, hint.value)).collect()
}

/// The `uic` starting point: node voltages for the first load and the charge
/// slots whose value comes from an instance `ic=`.
#[derive(Debug)]
pub(super) struct UicStart {
    /// Node rows from `.nodeset` then `.ic` (zero elsewhere); inductor branch
    /// rows hold their initial current or another prescribed storage coordinate.
    pub x: Vector,
    /// `(absolute state slot, signed storage coordinate)` for voltage storage.
    pub charges: Vec<(usize, Real)>,
}

/// Builds the `uic` start exactly as `CKTic()`/`CAPgetic()`/`indload.c` do.
///
/// # Errors
/// Missing state or branch bindings.
pub(super) fn uic_start(
    circuit: &Circuit,
    hints: &Hints,
    context: &ModelContext,
) -> SpiceResult<UicStart> {
    let mut x = Vector::zeros(circuit.unknown_count());
    for hint in hints.nodesets.iter().chain(&hints.initial) {
        x.as_mut_slice()[hint.row] = hint.value;
    }
    let mut charges = Vec::new();
    for (index, _) in circuit.devices().iter().enumerate() {
        for (row, coordinate) in bound_initial_coordinates(circuit, index, context)? {
            x.as_mut_slice()[row.0] = coordinate.value;
        }
    }
    for index in 0..circuit.device_count() {
        if let Some(voltage) = bound_initial_voltage(circuit, index, &x, context)? {
            charges.push((voltage.slot, voltage.scale * voltage.voltage));
        }
    }
    Ok(UicStart { x, charges })
}

/// Tolerances for the `uic` consistency check.
#[derive(Debug, Clone, Copy)]
pub(super) struct Tolerances {
    pub reltol: Real,
    pub vntol: Real,
    pub abstol: Real,
}

/// Bind both sides of the local coordinate/evolution-equation relation.
fn bound_initial_coordinates(
    circuit: &Circuit,
    index: usize,
    context: &ModelContext,
) -> SpiceResult<Vec<(AbsoluteRow, FixedCoordinate)>> {
    let device = &circuit.devices()[index];
    let coordinates = device.initial_coordinates(context)?;
    if coordinates.is_empty() {
        return Ok(Vec::new());
    }
    let rows = circuit
        .branch_rows(index)
        .ok_or_else(|| failure("missing initial coordinate rows"))?;
    let mut bound = Vec::with_capacity(coordinates.len());
    for coordinate in coordinates {
        if coordinate.coordinate >= rows.len()
            || coordinate.equation >= rows.len()
            || !coordinate.value.is_finite()
        {
            return Err(failure(format!(
                "{}: invalid initial coordinate binding",
                device.name()
            )));
        }
        bound.push((
            AbsoluteRow(rows.start + coordinate.coordinate),
            FixedCoordinate {
                name: device.name().to_owned(),
                value: coordinate.value,
                equation: AbsoluteRow(rows.start + coordinate.equation),
            },
        ));
    }
    Ok(bound)
}

fn bound_initial_voltage(
    circuit: &Circuit,
    index: usize,
    initial: &Vector,
    context: &ModelContext,
) -> SpiceResult<Option<CapacitorInitial>> {
    let device = &circuit.devices()[index];
    let Some(storage) = device.initial_voltage_storage(context)? else {
        return Ok(None);
    };
    let [positive, negative, ..] = device.terminals() else {
        return Err(failure("voltage storage needs two terminals"));
    };
    let positive = VoltageVertex::bind(circuit, *positive)?;
    let negative = VoltageVertex::bind(circuit, *negative)?;
    let at = |vertex| match vertex {
        VoltageVertex::Ground => 0.,
        VoltageVertex::Node(AbsoluteRow(row)) => initial.as_slice()[row],
    };
    let voltage = storage
        .voltage
        .unwrap_or_else(|| at(positive) - at(negative));
    let states = circuit
        .state_rows(index)
        .ok_or_else(|| failure("missing voltage storage range"))?;
    if storage.slot >= states.len() || !storage.scale.is_finite() || !voltage.is_finite() {
        return Err(failure("invalid initial voltage storage binding"));
    }
    let equation = storage
        .equation
        .map(|equation| {
            let rows = circuit
                .branch_rows(index)
                .ok_or_else(|| failure("missing voltage evolution row"))?;
            if equation >= rows.len() {
                return Err(failure("voltage evolution row outside device"));
            }
            Ok(AbsoluteRow(rows.start + equation))
        })
        .transpose()?;
    Ok(Some(CapacitorInitial {
        slot: states.start + storage.slot,
        scale: storage.scale,
        name: device.name().to_owned(),
        positive,
        negative,
        voltage,
        equation,
    }))
}

/// Storage constraints for the instantaneous circuit, in device order.
struct ImpulseStorage {
    fixed_coordinates: BTreeMap<AbsoluteRow, FixedCoordinate>,
    capacitors: Vec<CapacitorInitial>,
}

struct FixedCoordinate {
    name: String,
    value: Real,
    equation: AbsoluteRow,
}

struct CapacitorInitial {
    name: String,
    positive: VoltageVertex,
    negative: VoltageVertex,
    voltage: Real,
    equation: Option<AbsoluteRow>,
    slot: usize,
    scale: Real,
}

impl ImpulseStorage {
    fn collect(circuit: &Circuit, initial: &Vector, context: &ModelContext) -> SpiceResult<Self> {
        if initial.len() != circuit.unknown_count() {
            return Err(failure("initial vector does not match finalized circuit"));
        }
        let mut storage = Self {
            fixed_coordinates: BTreeMap::new(),
            capacitors: Vec::new(),
        };
        for index in 0..circuit.device_count() {
            for (row, coordinate) in bound_initial_coordinates(circuit, index, context)? {
                if storage.fixed_coordinates.insert(row, coordinate).is_some() {
                    return Err(failure("duplicate fixed initial coordinate"));
                }
            }
            if let Some(voltage) = bound_initial_voltage(circuit, index, initial, context)? {
                storage.capacitors.push(voltage);
            }
        }
        Ok(storage)
    }
}

/// Independent capacitor voltage relations and relations implied by E/H outputs.
struct CapacitorRelations<'a> {
    kept: Vec<&'a CapacitorInitial>,
    implied: Vec<ImpliedCapacitor<'a>>,
    // Known redundant voltage relations; snap their accepted storage
    // to the actual constrained voltage, within the user's IC tolerance.
    canonical: Vec<(usize, Real)>,
}

struct ImpliedCapacitor<'a> {
    capacitor: &'a CapacitorInitial,
    provenance: String,
}

impl<'a> CapacitorRelations<'a> {
    fn classify(
        circuit: &Circuit,
        rhs: &Vector,
        capacitors: &'a [CapacitorInitial],
        tolerance: VoltageTolerance,
    ) -> SpiceResult<Self> {
        let mut graph = ConstraintGraph::for_circuit(circuit, rhs, false)?;
        let mut relations = Self {
            kept: Vec::new(),
            implied: Vec::new(),
            canonical: Vec::new(),
        };
        for capacitor in capacitors {
            let CapacitorInitial {
                name,
                positive,
                negative,
                voltage,
                ..
            } = capacitor;
            if graph.controlled_implies(*positive, *negative) {
                relations.implied.push(ImpliedCapacitor {
                    capacitor,
                    provenance: graph.controlled_path(*positive, *negative)?,
                });
                continue;
            }
            match graph.join(*positive, *negative, *voltage, Origin::Device(name))? {
                Join::Tree => relations.kept.push(capacitor),
                Join::Cycle { expected } if tolerance.agree(expected, *voltage) => {
                    relations
                        .canonical
                        .push((capacitor.slot, capacitor.scale * expected));
                }
                Join::Cycle { expected } => {
                    return Err(failure(format!(
                        "uic initial conditions are inconsistent with the circuit at t = 0+: \
                         capacitor {name} starts at {voltage} V but ideal sources and other \
                         initial conditions force {expected} V across it (path: {}), an impulse; make the \
                         initial condition agree or add series impedance",
                        graph.path_description()
                    )));
                }
            }
        }
        Ok(relations)
    }

    fn check_implied(
        &mut self,
        solution: &InstantaneousSolution,
        tolerance: VoltageTolerance,
    ) -> SpiceResult<()> {
        for ImpliedCapacitor {
            capacitor,
            provenance,
        } in &self.implied
        {
            let CapacitorInitial {
                name,
                positive,
                negative,
                voltage,
                ..
            } = capacitor;
            let (Some(vp), Some(vq)) =
                (solution.potential(*positive), solution.potential(*negative))
            else {
                return Err(failure(format!(
                    "uic initial conditions cannot be shown consistent: capacitor {name} is \
                     across the output of controlled sources ({provenance}) but its voltage is not \
                     determined by the instantaneous system"
                )));
            };
            let expected = vp - vq;
            if !tolerance.agree(expected, *voltage) {
                return Err(failure(format!(
                    "uic initial conditions are inconsistent with the circuit at t = 0+: \
                     capacitor {name} starts at {voltage} V but ideal controlled sources ({provenance}) \
                     force {expected} V across it, an impulse; make the initial condition agree \
                     or add series impedance"
                )));
            }
            self.canonical
                .push((capacitor.slot, capacitor.scale * expected));
        }
        Ok(())
    }
}

/// Disposable exact static Newton loads for the reduced instantaneous
/// circuit. The initial history remains untouched; the ordinary UIC
/// device-flagged load still owns semiconductor initial charge semantics.
fn solve_nonlinear_instantaneous(
    prepared: &PreparedRun<'_>,
    request: &LoadRequest<'_>,
    storage: &ImpulseStorage,
    kept: &[&CapacitorInitial],
    tolerances: Tolerances,
    options: &crate::analysis::newton::NewtonOptions,
) -> SpiceResult<Option<InstantaneousSolution>> {
    let circuit = prepared.circuit();
    let mut x = request.solution.clone();
    let mut matrix = circuit.new_matrix()?;
    let mut rhs = Vector::zeros(x.len());
    let limited = crate::analysis::bias::limited_rows(circuit).unwrap_or_else(|| {
        crate::analysis::bias::branch_rows(circuit)
            .into_iter()
            .map(|branch| !branch)
            .collect()
    });
    let mut previous = None;
    let mut best: Option<(Real, InstantaneousSolution)> = None;
    let mut trial = None;
    for _ in 0..options.max_iterations {
        matrix.clear();
        rhs.clear();
        let mut next_trial = request
            .history
            .trial_in(IterationPhase::Float, trial.as_ref())?
            .with_device_limiting(false);
        prepared.load(
            &LoadRequest {
                solution: &x,
                ..*request
            },
            &mut matrix,
            &mut rhs,
            &mut next_trial,
        )?;
        let nonconvergent = next_trial.is_nonconvergent();
        trial = Some(next_trial);
        let system = InstantaneousSystem::assemble(circuit, &matrix, &rhs, storage, kept)?;
        if let Some(solution) = previous.take()
            && !nonconvergent
        {
            let (ratio, roundoff) = system.residual_ratio(&solution, tolerances);
            if ratio <= 1. {
                // Refine a consistent algebraic point to rounding accuracy
                // when possible. Otherwise retain the best verified iterate
                // once the residual stops improving. A tiny voltage error
                // can become a large artificial derivative on the first step.
                if roundoff {
                    return Ok(Some(solution));
                }
                if best.as_ref().is_some_and(|(score, _)| ratio >= *score) {
                    return Ok(best.map(|(_, solution)| solution));
                }
                best = Some((ratio, solution));
            }
        }
        let Some(solution) = system.solve(circuit, storage, tolerances)? else {
            return Ok(None);
        };
        let scale = solution.step_scale(&x, options.voltage_step, &limited);
        x = solution.update(&x, scale);
        previous = (scale == 1.).then_some(solution);
    }
    best.map(|(_, solution)| Some(solution))
        .ok_or_else(|| failure("uic nonlinear instantaneous constraints did not converge"))
}

/// Consistent algebraic guess plus canonical storage values for redundant
/// initial voltage constraints.
pub(super) struct CheckedInitial {
    pub solution: Option<Vector>,
    pub canonical: Vec<(usize, Real)>,
}

/// Checks that the `uic` state can start without an impulse at `t = 0+`.
///
/// C does not look: it integrates a first step from whatever charges it was
/// given, so a capacitor whose `ic` differs from an ideal source across it
/// just produces a huge first-step current. Here the instantaneous (`h -> 0`)
/// problem is formed exactly: every capacitor becomes a voltage constraint with
/// a free current, every inductor a fixed current (its voltage is free), every
/// other linear element and the sources at their `t = 0` right limit (`rhs`)
/// stay as they are. Native differential devices replace their evolution
/// equations, preserving their constitutive rows. Fixed internal transfer
/// coordinates eliminate their own differential equations. For nonlinear
/// devices the reduced system is reloaded at each Newton iterate and
/// checked against the true residual before accepting consistency.
///
/// Redundant-but-consistent relations (a capacitor whose `ic` equals the
/// voltage that ideal sources already force, an inductor in series with a
/// current source carrying its current) are recognised structurally and
/// dropped; a contradicting one is an error naming the element. Whatever else
/// leaves the system singular or non-square is reported as ill-posed.
///
/// # Errors
/// An inconsistent (impulsive) or ill-posed initial state.
pub(super) fn check_impulse_free(
    prepared: &PreparedRun<'_>,
    a: &SparseMatrix,
    rhs: &Vector,
    request: &LoadRequest<'_>,
    tolerances: Tolerances,
    options: &crate::analysis::newton::NewtonOptions,
) -> SpiceResult<CheckedInitial> {
    let circuit = prepared.circuit();
    let storage = ImpulseStorage::collect(circuit, request.solution, request.model_context)?;
    if storage.capacitors.is_empty() && storage.fixed_coordinates.is_empty() {
        return Ok(CheckedInitial {
            solution: None,
            canonical: Vec::new(),
        });
    }
    let voltage_tolerance = VoltageTolerance {
        reltol: tolerances.reltol,
        vntol: tolerances.vntol,
    };
    let mut relations =
        CapacitorRelations::classify(circuit, rhs, &storage.capacitors, voltage_tolerance)?;
    let nonlinear = circuit.devices().iter().any(|device| device.is_nonlinear());
    let solution = if nonlinear {
        solve_nonlinear_instantaneous(
            prepared,
            request,
            &storage,
            &relations.kept,
            tolerances,
            options,
        )?
    } else {
        InstantaneousSystem::assemble(circuit, a, rhs, &storage, &relations.kept)?
            .solve(circuit, &storage, tolerances)?
    };
    match solution {
        Some(solution) => {
            relations.check_implied(&solution, voltage_tolerance)?;
            Ok(CheckedInitial {
                solution: nonlinear.then(|| solution.update(request.solution, 1.)),
                canonical: relations.canonical,
            })
        }
        None if relations.implied.is_empty() => Ok(CheckedInitial {
            solution: None,
            canonical: relations.canonical,
        }),
        None => Err(failure(format!(
            "uic initial conditions cannot be shown consistent: capacitors across the \
             output of controlled sources {} are left without an instantaneous system",
            relations
                .implied
                .iter()
                .map(|item| item.provenance.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        ))),
    }
}
