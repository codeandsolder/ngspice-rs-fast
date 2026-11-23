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

use crate::devices::{Circuit, ModelContext, StorageKind};
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
    /// rows hold their initial current.
    pub x: Vector,
    /// `(absolute state slot, charge)` for capacitors with an instance `ic=`.
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
    for (index, device) in circuit.devices().iter().enumerate() {
        let (Some(element), Some(slot)) =
            (device.storage_element(context), device.truncation_slot())
        else {
            continue;
        };
        let element = element?;
        match element.kind {
            StorageKind::Capacitor => {
                if let Some(voltage) = element.initial {
                    let base = circuit
                        .state_rows(index)
                        .ok_or_else(|| failure("missing state range"))?
                        .start;
                    charges.push((base + slot, element.value * voltage));
                }
            }
            StorageKind::Inductor => {
                let row = circuit
                    .branch_rows(index)
                    .filter(|rows| rows.len() == 1)
                    .ok_or_else(|| {
                        failure(format!("inductor {} has no branch row", device.name()))
                    })?
                    .start;
                x.as_mut_slice()[row] = element.initial.unwrap_or(0.);
            }
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

/// Storage constraints for the instantaneous circuit, in device order.
struct ImpulseStorage {
    fixed_currents: BTreeMap<AbsoluteRow, FixedCurrent>,
    capacitors: Vec<CapacitorInitial>,
}

struct FixedCurrent {
    name: String,
    value: Real,
}

struct CapacitorInitial {
    name: String,
    positive: VoltageVertex,
    negative: VoltageVertex,
    voltage: Real,
}

impl ImpulseStorage {
    fn collect(circuit: &Circuit, initial: &Vector, context: &ModelContext) -> SpiceResult<Self> {
        if initial.len() != circuit.unknown_count() {
            return Err(failure("initial vector does not match finalized circuit"));
        }
        let mut storage = Self {
            fixed_currents: BTreeMap::new(),
            capacitors: Vec::new(),
        };
        for (index, device) in circuit.devices().iter().enumerate() {
            let Some(element) = device.storage_element(context).transpose()? else {
                continue;
            };
            let [positive, negative] = device.terminals() else {
                continue;
            };
            match element.kind {
                StorageKind::Inductor => {
                    let row = circuit
                        .branch_rows(index)
                        .filter(|rows| rows.len() == 1)
                        .ok_or_else(|| {
                            failure(format!("inductor {} has no branch row", device.name()))
                        })?
                        .start;
                    let value = *initial
                        .as_slice()
                        .get(row)
                        .ok_or_else(|| failure("inductor branch row outside finalized circuit"))?;
                    storage.fixed_currents.insert(
                        AbsoluteRow(row),
                        FixedCurrent {
                            name: device.name().to_owned(),
                            value,
                        },
                    );
                }
                StorageKind::Capacitor => {
                    let positive = VoltageVertex::bind(circuit, *positive)?;
                    let negative = VoltageVertex::bind(circuit, *negative)?;
                    let at = |vertex| match vertex {
                        VoltageVertex::Ground => 0.,
                        VoltageVertex::Node(AbsoluteRow(row)) => initial.as_slice()[row],
                    };
                    let voltage = element
                        .initial
                        .unwrap_or_else(|| at(positive) - at(negative));
                    storage.capacitors.push(CapacitorInitial {
                        name: device.name().to_owned(),
                        positive,
                        negative,
                        voltage,
                    });
                }
            }
        }
        Ok(storage)
    }
}

/// Independent capacitor voltage relations and relations implied by E/H outputs.
struct CapacitorRelations<'a> {
    kept: Vec<&'a CapacitorInitial>,
    implied: Vec<ImpliedCapacitor<'a>>,
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
        };
        for capacitor in capacitors {
            let CapacitorInitial {
                name,
                positive,
                negative,
                voltage,
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
                Join::Cycle { expected } if tolerance.agree(expected, *voltage) => {}
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
        &self,
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
        }
        Ok(())
    }
}

/// Checks that the `uic` state can start without an impulse at `t = 0+`.
///
/// C does not look: it integrates a first step from whatever charges it was
/// given, so a capacitor whose `ic` differs from an ideal source across it
/// just produces a huge first-step current. Here the instantaneous (`h -> 0`)
/// problem is formed exactly: every capacitor becomes a voltage constraint with
/// a free current, every inductor a fixed current (its voltage is free), every
/// other element and the sources at their `t = 0` right limit (`rhs`) stay as
/// they are. A state is impulse-free exactly when that system has a solution.
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
    circuit: &Circuit,
    a: &SparseMatrix,
    rhs: &Vector,
    x: &Vector,
    tolerances: Tolerances,
    context: &ModelContext,
) -> SpiceResult<()> {
    let storage = ImpulseStorage::collect(circuit, x, context)?;
    if storage.capacitors.is_empty() && storage.fixed_currents.is_empty() {
        return Ok(());
    }
    let voltage_tolerance = VoltageTolerance {
        reltol: tolerances.reltol,
        vntol: tolerances.vntol,
    };
    let relations =
        CapacitorRelations::classify(circuit, rhs, &storage.capacitors, voltage_tolerance)?;
    let system = InstantaneousSystem::assemble(circuit, a, rhs, &storage, &relations.kept)?;
    match system.solve(circuit, &storage, tolerances)? {
        Some(solution) => relations.check_implied(&solution, voltage_tolerance),
        None if relations.implied.is_empty() => Ok(()),
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
