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

use crate::devices::{
    Circuit, InitialCoordinateHistory, InitialStoragePolicy, InitialStorageRow, IterationPhase,
    LoadRequest, ModelContext, PreparedRun,
};
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
    pub chgtol: Real,
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
                history: coordinate.history,
                device: index,
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
    charges: BTreeMap<AbsoluteRow, ChargeInitial>,
}

struct FixedCoordinate {
    name: String,
    value: Real,
    equation: AbsoluteRow,
    history: InitialCoordinateHistory,
    device: usize,
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

struct ChargeInitial {
    name: String,
    device: usize,
    slot: usize,
    gauge: AbsoluteRow,
    target: Real,
    voltage: Option<Real>,
    policy: InitialStoragePolicy,
}

impl ImpulseStorage {
    fn collect(
        circuit: &Circuit,
        initial: &Vector,
        context: &ModelContext,
        values: &[Real],
    ) -> SpiceResult<Self> {
        if initial.len() != circuit.unknown_count() {
            return Err(failure("initial vector does not match finalized circuit"));
        }
        let mut storage = Self {
            fixed_coordinates: BTreeMap::new(),
            capacitors: Vec::new(),
            charges: BTreeMap::new(),
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
            if let Some((row, item)) = bound_expression_storage(circuit, index, values)?
                && storage.charges.insert(row, item).is_some()
            {
                return Err(failure("duplicate initial expression equation"));
            }
        }
        Ok(storage)
    }
}

/// Bind local storage coordinates to the finalized MNA layout. Signal
/// rows are exclusively declared by generated differentiator devices.
fn bound_expression_storage(
    circuit: &Circuit,
    index: usize,
    values: &[Real],
) -> SpiceResult<Option<(AbsoluteRow, ChargeInitial)>> {
    let device = &circuit.devices()[index];
    let Some(storage) = device.initial_expression_storage() else {
        return Ok(None);
    };
    let rows = circuit
        .branch_rows(index)
        .ok_or_else(|| failure("missing initial expression branch binding"))?;
    let mut states = circuit
        .state_rows(index)
        .ok_or_else(|| failure("missing initial expression state binding"))?;
    let bind = |row| {
        match row {
            InitialStorageRow::Branch(local) => rows.clone().nth(local),
            InitialStorageRow::Signal(node) => {
                if storage.policy == InitialStoragePolicy::Derived
                    && device.terminals().first() == Some(&node)
                {
                    circuit.unknowns().node_row(node)
                } else {
                    None
                }
            }
        }
        .map(AbsoluteRow)
        .ok_or_else(|| failure(format!("{}: invalid initial expression row", device.name())))
    };
    let slot = states.nth(storage.slot).ok_or_else(|| {
        failure(format!(
            "{}: invalid initial expression slot",
            device.name()
        ))
    })?;
    let target = values[slot];
    if !target.is_finite() {
        return Err(failure(format!(
            "{}: nonfinite initial stored expression",
            device.name()
        )));
    }
    Ok(Some((
        bind(storage.equation)?,
        ChargeInitial {
            name: device.name().to_owned(),
            device: index,
            slot,
            gauge: bind(storage.gauge)?,
            target,
            voltage: storage.voltage,
            policy: storage.policy,
        },
    )))
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

/// Replaces only the selected charge evolution rows with their actual
/// signed physical storage constraint; no extra integration timepoint or
/// artificial capacitor is introduced.
struct ChargeConstraints {
    value: Vector,
    jacobian: SparseMatrix,
}

impl ChargeConstraints {
    fn new(size: usize) -> Self {
        Self {
            value: Vector::zeros(size),
            jacobian: SparseMatrix::new(size, size),
        }
    }

    fn load(
        &mut self,
        prepared: &PreparedRun<'_>,
        request: &LoadRequest<'_>,
        storage: &ImpulseStorage,
        matrix: &mut SparseMatrix,
        rhs: &mut Vector,
    ) -> SpiceResult<()> {
        self.value.clear();
        self.jacobian.clear();
        prepared.assemble_charge_for(
            request.model_context,
            request.solution,
            0.,
            &mut self.value,
            &mut self.jacobian,
            storage.charges.values().map(|charge| charge.device),
        )?;
        for (&AbsoluteRow(row), charge) in &storage.charges {
            matrix.zero_row(row, |_| false);
            rhs.set(row, charge.target - self.value.as_slice()[row])?;
        }
        // The replacement RHS is J_Q*x - Q(x) + Q0. Preserve the
        // physical signed quantity separately for the acceptance residual.
        for term in self.jacobian.triplets() {
            if !storage.charges.contains_key(&AbsoluteRow(term.row)) {
                return Err(failure(
                    "initial charge kernel wrote an undeclared equation",
                ));
            }
            matrix.add(term.row, term.col, term.value)?;
            let value = term.value.mul_add(
                request.solution.as_slice()[term.col],
                rhs.as_slice()[term.row],
            );
            rhs.set(term.row, value)?;
        }
        Ok(())
    }

    fn residual(
        &self,
        storage: &ImpulseStorage,
        solution: &InstantaneousSolution,
        tolerance: Tolerances,
    ) -> (Real, bool) {
        let mut ratio: Real = 0.;
        let mut roundoff = true;
        for (&AbsoluteRow(row), charge) in &storage.charges {
            let actual = self.value.as_slice()[row];
            if !actual.is_finite() {
                return (Real::INFINITY, false);
            }
            if charge.policy == InitialStoragePolicy::Derived
                && !solution.retained_derived(AbsoluteRow(row))
            {
                continue;
            }
            let absolute = match charge.policy {
                InitialStoragePolicy::Physical => tolerance.chgtol,
                InitialStoragePolicy::Derived => tolerance.vntol,
            };
            let scale = actual.abs().max(charge.target.abs());
            let residual = (actual - charge.target).abs();
            ratio = ratio.max(residual / tolerance.reltol.mul_add(scale, absolute));
            roundoff &= residual <= 4. * Real::EPSILON * scale;
        }
        (ratio, roundoff)
    }

    fn canonical(&self, storage: &ImpulseStorage) -> Vec<(usize, Real)> {
        storage
            .charges
            .iter()
            .map(|(&AbsoluteRow(row), charge)| (charge.slot, self.value.as_slice()[row]))
            .collect()
    }
}

struct NonlinearInitial {
    solution: InstantaneousSolution,
    canonical: Vec<(usize, Real)>,
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
) -> SpiceResult<Option<NonlinearInitial>> {
    let circuit = prepared.circuit();
    let mut x = request.solution.clone();
    // Explicit instance voltage is only a Newton seed here. Its physical
    // effect is the signed charge value fixed by the original UIC load.
    for charge in storage.charges.values() {
        if let Some(voltage) = charge.voltage {
            let [positive, negative, ..] = circuit.devices()[charge.device].terminals() else {
                return Err(failure("initial charge requires two terminals"));
            };
            circuit
                .unknowns()
                .seed_terminal_voltage(&mut x, [*positive, *negative], voltage)?;
        }
    }
    let mut charge_constraints =
        (!storage.charges.is_empty()).then(|| ChargeConstraints::new(x.len()));
    let mut matrix = circuit.new_matrix()?;
    let mut rhs = Vector::zeros(x.len());
    let limited = crate::analysis::bias::limited_rows(circuit).unwrap_or_else(|| {
        crate::analysis::bias::branch_rows(circuit)
            .into_iter()
            .map(|branch| !branch)
            .collect()
    });
    let mut previous = None;
    let mut best: Option<(Real, NonlinearInitial)> = None;
    let mut trial = None;
    for _ in 0..options.max_iterations {
        matrix.clear();
        rhs.clear();
        let mut next_trial = request
            .history
            .trial_in(
                if trial.is_some() {
                    IterationPhase::Float
                } else {
                    IterationPhase::Junction
                },
                trial.as_ref(),
            )?
            .with_device_limiting(false);
        let point = LoadRequest {
            solution: &x,
            ..*request
        };
        prepared.load(&point, &mut matrix, &mut rhs, &mut next_trial)?;
        let nonconvergent = next_trial.is_nonconvergent();
        trial = Some(next_trial);
        if let Some(constraints) = &mut charge_constraints {
            constraints.load(prepared, &point, storage, &mut matrix, &mut rhs)?;
        }
        let system = InstantaneousSystem::assemble(circuit, &matrix, &rhs, storage, kept)?;
        if let Some(solution) = previous.take()
            && !nonconvergent
        {
            let (mut ratio, mut roundoff) = system.residual_ratio(&solution, &x, tolerances);
            if let Some(constraints) = &charge_constraints {
                let (physical, exact) = constraints.residual(storage, &solution, tolerances);
                ratio = ratio.max(physical);
                roundoff &= exact;
            }
            if ratio <= 1. {
                let solution = NonlinearInitial {
                    solution,
                    canonical: charge_constraints
                        .as_ref()
                        .map_or_else(Vec::new, |constraints| constraints.canonical(storage)),
                };
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
            if charge_constraints.is_some() {
                return Err(failure(
                    "uic initial charge constraints leave no instantaneous solution",
                ));
            }
            return Ok(None);
        };
        let scale = solution.step_scale(&x, options.voltage_step, &limited);
        x = solution.update(&x, scale);
        previous = (scale >= 1.).then_some(solution);
    }
    best.map(|(_, solution)| Some(solution)).ok_or_else(|| {
        let names = storage
            .charges
            .values()
            .map(|charge| charge.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        failure(format!(
            "uic nonlinear instantaneous constraints did not converge (charge storage: {names})"
        ))
    })
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
    initial_trial: &crate::devices::TrialState,
    tolerances: Tolerances,
    options: &crate::analysis::newton::NewtonOptions,
) -> SpiceResult<CheckedInitial> {
    let circuit = prepared.circuit();
    let storage = ImpulseStorage::collect(
        circuit,
        request.solution,
        request.model_context,
        initial_trial.values(),
    )?;
    if storage.capacitors.is_empty()
        && storage.fixed_coordinates.is_empty()
        && storage.charges.is_empty()
    {
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
            .map(|solution| NonlinearInitial {
                solution,
                canonical: Vec::new(),
            })
    };
    match solution {
        Some(NonlinearInitial {
            solution,
            canonical,
        }) => {
            relations.check_implied(&solution, voltage_tolerance)?;
            relations.canonical.extend(canonical);
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

/// Whether this circuit has explicitly derived, rather than physical,
/// initial storage. Ordinary RLC and charge-law circuits keep their established
/// operating-point and UIC initialization path.
pub(super) fn has_derived_storage(circuit: &Circuit) -> bool {
    circuit.devices().iter().any(|device| {
        device
            .initial_expression_storage()
            .is_some_and(|storage| storage.policy == InitialStoragePolicy::Derived)
    })
}

impl ImpulseStorage {
    fn retain_operating_point(&mut self, state: &Vector) {
        for (&AbsoluteRow(row), coordinate) in &mut self.fixed_coordinates {
            coordinate.value = state.as_slice()[row];
        }
        for capacitor in &mut self.capacitors {
            capacitor.voltage =
                potential(state, capacitor.positive) - potential(state, capacitor.negative);
        }
    }

    fn consistent_history(
        &self,
        circuit: &Circuit,
        right: &system::RightLimit,
        continuity: Option<Real>,
        tolerances: Tolerances,
    ) -> SpiceResult<Vec<(usize, Real)>> {
        let mut slots = Vec::new();
        for (&AbsoluteRow(row), charge) in &self.charges {
            let actual = right.charge.as_slice()[row];
            if let Some(time) = continuity
                && charge.policy == InitialStoragePolicy::Derived
            {
                let tolerance = VoltageTolerance {
                    reltol: tolerances.reltol,
                    vntol: tolerances.vntol,
                };
                if !tolerance.agree(actual, charge.target) {
                    return Err(failure(format!(
                        "{}: the derivative input jumps from {} to {} at t = {time}; a classical derivative does not exist across this prehistory discontinuity or source event",
                        charge.name, -charge.target, -actual
                    )));
                }
            }
            slots.extend([
                (charge.slot, actual),
                (charge.slot + 1, right.charge_rate.as_slice()[row]),
            ]);
        }
        for capacitor in &self.capacitors {
            let voltage = potential(&right.state, capacitor.positive)
                - potential(&right.state, capacitor.negative);
            let slope = potential(&right.rate, capacitor.positive)
                - potential(&right.rate, capacitor.negative);
            slots.extend([
                (capacitor.slot, capacitor.scale * voltage),
                (capacitor.slot + 1, capacitor.scale * slope),
            ]);
        }
        for (&AbsoluteRow(row), coordinate) in &self.fixed_coordinates {
            let states = circuit
                .state_rows(coordinate.device)
                .ok_or_else(|| failure("missing initial coordinate history"))?;
            let (slot, value, rate) = match coordinate.history {
                InitialCoordinateHistory::EquationCharge { slot, scale } => {
                    let AbsoluteRow(equation) = coordinate.equation;
                    (
                        slot,
                        scale * right.charge.as_slice()[equation],
                        scale * right.charge_rate.as_slice()[equation],
                    )
                }
                InitialCoordinateHistory::Coordinate { slot } => (
                    slot,
                    right.state.as_slice()[row],
                    right.rate.as_slice()[row],
                ),
            };
            if slot.saturating_add(1) >= states.len() {
                return Err(failure("invalid initial coordinate history binding"));
            }
            slots.extend([
                (states.start + slot, value),
                (states.start + slot + 1, rate),
            ]);
        }
        Ok(slots)
    }
}

fn potential(state: &Vector, vertex: VoltageVertex) -> Real {
    match vertex {
        VoltageVertex::Ground => 0.,
        VoltageVertex::Node(AbsoluteRow(row)) => state.as_slice()[row],
    }
}

/// A completely determined physical derivative point. The order hint is
/// disposable computational metadata, not an accepted integration state.
pub(super) struct DerivativePoint {
    pub solution: Vector,
    pub canonical: Vec<(usize, Real)>,
    pub order_hint: usize,
}

/// Calculate the right-limit state and all declared initial storage rates.
/// The caller publishes a fresh exact device trial only after this succeeds.
pub(super) fn consistent_derivative_start(
    prepared: &PreparedRun<'_>,
    request: &LoadRequest<'_>,
    trial: &crate::devices::TrialState,
    uic: bool,
    tolerances: Tolerances,
    options: &crate::analysis::newton::NewtonOptions,
) -> SpiceResult<Option<DerivativePoint>> {
    let circuit = prepared.circuit();
    if !has_derived_storage(circuit) {
        return Ok(None);
    }
    let forcing = request
        .forcing
        .ok_or_else(|| failure("missing derivative initialization timing"))?;
    let mut storage = ImpulseStorage::collect(
        circuit,
        request.solution,
        request.model_context,
        trial.values(),
    )?;
    if !uic {
        storage.retain_operating_point(request.solution);
    }
    let result = (|| {
        let dae = prepared.prepare_physical_dae(request.model_context, forcing.timing)?;
        let right = system::solve_right_limit(
            &dae,
            request.solution,
            &storage,
            forcing.timing.step(),
            tolerances,
            options.max_iterations,
        )?;
        let canonical =
            storage.consistent_history(circuit, &right, (!uic).then_some(0.), tolerances)?;
        Ok(DerivativePoint {
            solution: right.state,
            order_hint: right.order_hint,
            canonical,
        })
    })();
    match result {
        // Existing ordinary transient loads remain available for device
        // families that have not yet acquired a high-order physical kernel.
        Err(SpiceError::Unsupported { .. }) => Ok(None),
        other => other.map(Some),
    }
}

/// Solve the physical derivative array with independent numerical storage
/// constraints; dependent companion equations are intentionally projected out.
pub(super) fn consistent_derivative_step(
    prepared: &PreparedRun<'_>,
    request: &LoadRequest<'_>,
    tolerances: Tolerances,
    iterations: usize,
    order_hint: usize,
) -> SpiceResult<DerivativePoint> {
    let circuit = prepared.circuit();
    let forcing = request
        .forcing
        .ok_or_else(|| failure("missing derivative step forcing"))?;
    let values = request
        .history
        .accepted(1)
        .ok_or_else(|| failure("missing derivative step history"))?;
    let storage =
        ImpulseStorage::collect(circuit, request.solution, request.model_context, values)?;
    let dae = prepared.prepare_physical_dae(request.model_context, forcing.timing)?;
    let right = system::solve_step(
        &dae, circuit, request, &storage, tolerances, iterations, order_hint,
    )?;
    let canonical = storage.consistent_history(circuit, &right, None, tolerances)?;
    Ok(DerivativePoint {
        solution: right.state,
        order_hint: right.order_hint,
        canonical,
    })
}

/// Cross an event without changing physical or derivative-input storage.
/// The left point is a disposable, already validated integration candidate.
pub(super) fn consistent_derivative_event(
    prepared: &PreparedRun<'_>,
    request: &LoadRequest<'_>,
    left: &crate::devices::TrialState,
    tolerances: Tolerances,
    iterations: usize,
    order_hint: usize,
) -> SpiceResult<DerivativePoint> {
    let circuit = prepared.circuit();
    let forcing = request
        .forcing
        .ok_or_else(|| failure("missing derivative event timing"))?;
    let crate::devices::AnalysisMode::Instantaneous { time } = request.mode else {
        return Err(failure("invalid derivative event mode"));
    };
    let mut storage = ImpulseStorage::collect(
        circuit,
        request.solution,
        request.model_context,
        left.values(),
    )?;
    storage.retain_operating_point(request.solution);
    let dae = prepared.prepare_physical_dae(request.model_context, forcing.timing)?;
    let right = system::solve_event(&dae, request, &storage, tolerances, iterations, order_hint)?;
    let canonical = storage.consistent_history(circuit, &right, Some(time), tolerances)?;
    Ok(DerivativePoint {
        solution: right.state,
        order_hint: right.order_hint,
        canonical,
    })
}
