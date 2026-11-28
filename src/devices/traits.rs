//! The [`Device`] trait and the context devices stamp into.
//!
//! # How stamping works in ngspice
//!
//! Every analysis builds a modified nodal analysis (MNA) system `A x = b` where
//! the unknowns are the node voltages (plus a current for each voltage source and
//! inductor). Each device contributes to `A` and `b` through a `<dev>load()`
//! function — `resload.c`, `capload.c`, `bsim4v7load.c`, … The structure of those
//! functions is uniform: look up the matrix positions of the device's terminals,
//! then add conductances and current sources.
//!
//! [`StampContext`] models exactly that: it owns the matrix and right-hand side
//! being assembled, and offers [`StampContext::stamp`], which resolves
//! [`NodeId`]s to matrix rows and silently drops contributions that involve
//! ground — the row/column for ground is eliminated from the system, which is
//! what `CKTground` accomplishes in the C code.
//!
//! # Trial and accepted state
//!
//! [`Device::stamp`] takes `&self`: a load is a *trial* that may read the
//! accepted history and write only its [`crate::devices::state::DeviceState`] trial
//! slots. Nothing a trial does survives unless the analysis accepts the point
//! through [`crate::devices::Circuit::accept_point`], which first runs every
//! [`Device::accept`] hook and only then commits the trial state. See
//! [`crate::devices::state`].

use std::fmt;
use std::ops::Range;

use crate::maths::{Coefficients, SparseMatrix, Vector};
use crate::primitives::{Node, NodeId, NodeTable, Real, SourceLoc, SpiceError, SpiceResult};

use crate::devices::linear::Forcing;
use crate::devices::state::DeviceState;

/// Device-local physical charge history and its electrical incidence.
///
/// The stored rate is multiplied by `scale` and enters the positive node,
/// leaving the negative node. Internal state equations are not terminals.
#[derive(Clone, Copy, Debug)]
pub struct PhysicalChargeHistory {
    /// Local charge slot, followed immediately by its physical time derivative.
    pub slot: usize,
    /// Positive and negative electrical nodes of the charge contribution.
    pub nodes: [NodeId; 2],
    /// Conversion from device-local charge to signed electrical charge.
    pub scale: Real,
}

/// Physical nonlinear charge and its exact Jacobian in MNA coordinates.
/// This is independent of the time integration formula and AC's historical
/// small-signal linearization conventions.
pub struct ChargeContext<'a> {
    /// Effective model parameters and temperatures.
    pub model: &'a crate::devices::ModelContext,
    /// Physical node and branch numbering.
    pub unknowns: &'a MnaUnknowns,
    /// This device's first branch row, when allocated.
    pub branch: Option<usize>,
    /// Bound rows of branch-current controls, in expression order.
    pub controls: &'a [usize],
    /// State at which to evaluate the constitutive law.
    pub state: &'a Vector,
    /// Absolute time of the constitutive law evaluation.
    pub time: Real,
    /// Signed charge and flux residual by equation row.
    pub charge: &'a mut Vector,
    /// Exact derivative of that residual.
    pub jacobian: &'a mut SparseMatrix,
}

impl ChargeContext<'_> {
    /// Read the physical voltage of a node; ground is exactly zero.
    #[must_use = "the node potential is needed to evaluate the device at the current state"]
    pub fn voltage(&self, node: NodeId) -> Real {
        self.unknowns
            .node_row(node)
            .and_then(|row| self.state.get(row))
            .unwrap_or(0.)
    }

    /// Add an off-diagonal derivative of a charge branch with respect to
    /// a distinct control voltage. The signed four-terminal stamp preserves
    /// both charge conservation and invariance under a common-mode shift.
    ///
    /// # Errors
    /// Nonfinite derivative or invalid row/column.
    pub fn coupling(
        &mut self,
        output: [NodeId; 2],
        control: [NodeId; 2],
        derivative: Real,
    ) -> SpiceResult<()> {
        if !derivative.is_finite() {
            return Err(SpiceError::circuit("nonfinite nonlinear charge coupling"));
        }
        for (node, sign) in [(output[0], 1.), (output[1], -1.)] {
            if let Some(row) = self.unknowns.node_row(node) {
                for (node, weight) in [(control[0], 1.), (control[1], -1.)] {
                    if let Some(col) = self.unknowns.node_row(node) {
                        self.jacobian.add(row, col, sign * weight * derivative)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Publish a two-terminal charge law, positive from the first node to
    /// the second, with its derivative with respect to their voltage.
    ///
    /// # Errors
    /// Nonfinite constitutive law or invalid matrix dimensions.
    pub fn branch(
        &mut self,
        terminals: [NodeId; 2],
        charge: Real,
        capacitance: Real,
    ) -> SpiceResult<()> {
        if !charge.is_finite() || !capacitance.is_finite() {
            return Err(SpiceError::circuit("nonfinite nonlinear charge law"));
        }
        for (node, signed) in [(terminals[0], charge), (terminals[1], -charge)] {
            if let Some(row) = self.unknowns.node_row(node) {
                self.charge.add_to(row, signed)?;
            }
        }
        crate::devices::linear::nodal_stamp(self.jacobian, self.unknowns, terminals, capacitance)
    }
}

/// Physical convergence tolerance of an MNA coordinate. Auxiliary
/// constitutive coordinates are checked through their device's physical
/// residual, not an arbitrary tolerance in amperes or volts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceKind {
    /// Voltage-like coordinate.
    Voltage,
    /// Current-like coordinate.
    Current,
    /// Auxiliary coordinate governed by the device convergence test.
    Constitutive,
}

/// Observable physical meaning of a device-local MNA coordinate.
/// Auxiliary states are not terminal currents merely because their rows
/// are allocated in the branch part of the MNA layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchQuantity {
    /// Oriented terminal current.
    Current,
    /// Differential capacitor voltage rate.
    VoltageRate,
    /// Differential inductor current rate.
    CurrentRate,
    /// Explicit physical charge coordinate.
    Charge,
    /// Stored input of a differentiator (arbitrary expression units).
    ExpressionState,
    /// Normalized transfer state driven by a voltage input.
    VoltageState(usize),
    /// Normalized transfer state driven by a current input.
    CurrentState(usize),
    /// Oriented current at the transfer's sensing input.
    InputCurrent,
    /// Auxiliary capacitor voltage-constraint branch, zero outside bias.
    BiasCurrent,
}

impl BranchQuantity {
    /// Plot vector identity and unit for this coordinate.
    #[must_use = "the returned identity and unit define the requested branch observation"]
    pub fn observation(self, name: &str) -> (String, &'static str) {
        match self {
            Self::Current => (format!("i({name})"), "current"),
            Self::VoltageRate => (format!("dvdt({name})"), "voltage/s"),
            Self::CurrentRate => (format!("didt({name})"), "current/s"),
            Self::Charge => (format!("q({name})"), "charge"),
            Self::ExpressionState => (format!("state({name},0)"), "unknown"),
            Self::VoltageState(index) => (format!("state({name},{index})"), "voltage"),
            Self::CurrentState(index) => (format!("state({name},{index})"), "current"),
            Self::InputCurrent => (format!("i({name}:input)"), "current"),
            Self::BiasCurrent => (format!("i({name}:bias)"), "current"),
        }
    }
}

/// Which stored quantity a [`StorageElement`] integrates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKind {
    /// Capacitor: charge `q = C v`, `v` across the terminals (first minus second).
    Capacitor,
    /// Inductor: flux `phi = L i`, `i` positive from the first terminal to the second.
    Inductor,
}

/// A charge/flux-storing element and its instance `ic=`, seen by analyses that
/// apply initial conditions. The quantity lives in the state slot
/// [`Device::truncation_slot`]; the derivative in the next slot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StorageElement {
    /// Capacitor or inductor.
    pub kind: StorageKind,
    /// Capacitance in farads or inductance in henries.
    pub value: Real,
    /// The instance `ic=` (volts for a capacitor, amperes for an inductor),
    /// if given.
    pub initial: Option<Real>,
}

/// How an initial dynamic coordinate is represented in adjacent history slots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitialCoordinateHistory {
    /// The signed physical charge/flux in the associated evolution equation.
    EquationCharge {
        /// Local state slot; its successor holds the time derivative.
        slot: usize,
        /// Conversion from signed equation charge to stored device quantity.
        scale: Real,
    },
    /// The state coordinate itself is integrated by the device.
    Coordinate {
        /// Local state slot; its successor holds the time derivative.
        slot: usize,
    },
}

/// A prescribed internal state coordinate for UIC.
///
/// The coordinate and the equation that normally evolves it are distinct: e.g. an implicit
/// differential inductor stores its current in branch 0 and evolves it
/// through branch equation 1. Indices are device-local branch indices.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialCoordinate {
    /// Fixed unknown.
    pub coordinate: usize,
    /// Differential equation removed in the instantaneous constraint solve.
    pub equation: usize,
    /// Value of the coordinate at the initial point.
    pub value: Real,
    /// Mapping to accepted history.
    pub history: InitialCoordinateHistory,
}

/// Physical history associated with a prescribed initial voltage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoltageStorageHistory {
    /// A signed linear storage coordinate.
    Linear(Real),
    /// Integrated nonlinear charge with a signed MNA evolution equation.
    Polynomial {
        /// Prepared physical charge law.
        law: model_normalize_ngspice::passive_storage::PolynomialStorage,
        /// Multiplier from the evolution equation residual to physical charge.
        equation_scale: Real,
    },
}

impl VoltageStorageHistory {
    /// Evaluate the stored quantity and its derivative at a voltage.
    /// # Errors
    /// Nonfinite constitutive evaluation.
    pub fn at(self, voltage: Real) -> SpiceResult<(Real, Real)> {
        match self {
            Self::Linear(scale) => Ok((scale * voltage, scale)),
            Self::Polynomial { law, .. } => law
                .evaluate(voltage)
                .map(|point| (point.stored, point.differential))
                .map_err(|error| SpiceError::circuit(error.to_string())),
        }
    }
}

/// A voltage-controlled storage coordinate at UIC. Scalar capacitors
/// need a new free-current equation; implicit capacitors already have
/// that current and specify the local evolution equation to replace.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialVoltageStorage {
    /// Optional instance voltage, otherwise read the seeded terminal voltages.
    pub voltage: Option<Real>,
    /// Local history slot containing the signed integrated quantity.
    pub slot: usize,
    /// Physical stored quantity as a function of voltage.
    pub history: VoltageStorageHistory,
    /// Existing evolution row, or None to append a free-current constraint.
    pub equation: Option<usize>,
}

/// A device-local coordinate or a generated internal signal node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialStorageRow {
    /// Device-local branch row.
    Branch(usize),
    /// Generated internal signal coordinate (not an arbitrary external node).
    Signal(NodeId),
}

/// Initial signed storage expression constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialStoragePolicy {
    /// Physical stored charge: its initial value is mandatory.
    Physical,
    /// Derivative input: the seeded value is an initial condition only
    /// when independent of the circuit's mandatory constraints.
    Derived,
}

/// Binding of a nonlinear stored expression for instantaneous initialization.
/// The physical charge assembly supplies its value and Jacobian.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialExpressionStorage {
    /// Local history slot containing the signed stored expression.
    pub slot: usize,
    /// Private storage coordinate and its two-term linear evolution row.
    /// The only other coordinate in that row is the derivative output.
    pub equation: InitialStorageRow,
    /// Free current or derivative-output coordinate, whose own row holds
    /// the constitutive storage law. A rank completion may select zero
    /// only when the retained constraints do not determine it.
    pub gauge: InitialStorageRow,
    /// Optional initial terminal voltage used to evaluate physical charge.
    pub voltage: Option<Real>,
    /// Whether the seeded value is a mandatory or a derived constraint.
    pub policy: InitialStoragePolicy,
}

/// An ideal voltage relation used by the initial-condition constraint graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialVoltageRelation {
    /// Positive and negative terminals.
    pub terminals: [NodeId; 2],
    /// How the differential voltage is determined.
    pub law: InitialVoltageLaw,
}

/// Ideal voltage relation in the initial instantaneous system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitialVoltageLaw {
    /// Voltage determined by the other equations of the circuit.
    Controlled,
    /// Fixed voltage, including a zero-voltage sensing branch.
    Fixed(Real),
    /// Voltage from the forcing vector at a device-local branch row.
    Source(usize),
}

/// A device whose branch current another device senses, named as in the deck
/// (after subcircuit renaming): the controlling voltage source of an F/H card.
///
/// [`crate::devices::Circuit`] resolves it to the named device's branch row when it
/// numbers the unknowns (C: `CCCSsetup`/`CCVSsetup` call `CKTfndBranch`,
/// `src/spicelib/analysis/cktfbran.c`). Only devices whose
/// [`Device::findable_branch`] is `Some` can be named.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlReference {
    /// Instance name, lowercased.
    pub name: String,
    /// Where the reference was written, for diagnostics.
    pub location: Option<SourceLoc>,
}

/// One mutual-inductance term of an inductor's branch equation: the branch
/// row of a coupled inductor and the mutual inductance `M` in henries.
///
/// [`crate::devices::Circuit`] derives these from the K devices that name the
/// inductor (C `MUTtemp`: `M = k sqrt(|L1 L2|)`) and passes them to the
/// inductor in [`StampContext::mutual`] and [`crate::devices::LinearContext::mutual`].
/// The inductor's flux is then `L i + sum(M i_other)` (`indload.c` adds the
/// mutual flux to `INDflux`), so its companion model, its truncation error
/// and its `uic` initial flux all see the coupled flux. Duplicate couplings
/// of one pair are summed, as C's loads sum them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MutualTerm {
    /// Branch-current row of the coupled inductor.
    pub row: usize,
    /// Mutual inductance in henries (may be negative).
    pub inductance: Real,
    /// The coupled inductor's `ic=` (C `INDinitCond`, zero when absent),
    /// which `INDload` uses for the mutual flux of a `uic` first step.
    pub initial: Real,
}

/// The inductance of an inductor, as mutual coupling sees it
/// ([`Device::inductance`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InductanceValue {
    /// The self inductance the inductor stamps (C `INDinduct / INDm`).
    pub effective: Real,
    /// The inductance `MUTtemp` uses in `M = k sqrt(|L1 L2|)` (C
    /// `INDinduct`, after temperature and scale but before dividing by the
    /// multiplicity `m`).
    pub coupling_base: Real,
    /// The `ic=` initial current (C `INDinitCond`, zero when absent).
    pub initial: Real,
}

/// What a K (mutual inductance) device couples: every pair of the named
/// inductors with the same coefficient ([`Device::mutual_coupling`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MutualCoupling<'a> {
    /// The coupled inductors' instance names (after subcircuit renaming).
    pub inductors: &'a [ControlReference],
    /// The coupling coefficient `k`.
    pub coefficient: Real,
    /// Where the K card was written, for diagnostics.
    pub location: Option<&'a SourceLoc>,
}

/// Which analysis is currently loading the matrix.
///
/// Devices branch on this the way the C `CKTmode` bitmask does: a capacitor
/// stamps an open circuit at DC and a companion model in transient; a diode
/// linearises at DC and adds junction capacitances in AC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalysisMode {
    /// `.op` and the bias point of every other analysis.
    OperatingPoint,
    /// A step of a `.dc` sweep.
    DcSweep,
    /// `.ac` and `.noise`, at a single frequency.
    Ac {
        /// Frequency in Hz.
        frequency: Real,
    },
    /// Static physical equations at a transient time, without integration.
    /// Used with separately assembled charge/flux in a memoryless DAE.
    Instantaneous {
        /// Absolute time in seconds.
        time: Real,
    },
    /// A step of a `.tran` analysis.
    Transient {
        /// Absolute time in seconds.
        time: Real,
        /// Current timestep in seconds.
        dt: Real,
    },
}

impl AnalysisMode {
    /// True for the AC and noise analyses, where everything is complex and the
    /// system is loaded at `omega = 2 pi f`.
    #[must_use = "AC and real-valued loads require different stamping paths"]
    pub const fn is_ac(self) -> bool {
        matches!(self, Self::Ac { .. })
    }

    /// True for transient analysis.
    #[must_use = "transient loads require history and integration terms"]
    pub const fn is_transient(self) -> bool {
        matches!(self, Self::Transient { .. })
    }

    /// True only for equilibrium bias loads, not unintegrated transient
    /// physical residuals.
    #[must_use = "bias constraints must be released during transient physical loads"]
    pub const fn is_bias(self) -> bool {
        matches!(self, Self::OperatingPoint | Self::DcSweep)
    }

    /// True for static loads, including the unintegrated physical DAE.
    #[must_use = "static loads omit time-dependent companion terms"]
    pub const fn is_dc(self) -> bool {
        self.is_bias() || matches!(self, Self::Instantaneous { .. })
    }
}

/// Maps circuit unknowns to MNA matrix rows.
///
/// Ground is not an unknown: it is the reference and its row is eliminated.
/// Branch currents are unknowns too, and get rows after the nodes; the circuit
/// numbers them, and device-internal nodes, in C's setup order
/// ([`crate::devices::Circuit::finalize`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MnaUnknowns {
    node_rows: Vec<Option<usize>>,
    next_row: usize,
}

impl MnaUnknowns {
    /// An empty mapping.
    #[must_use = "the new unknown allocator must be retained to assign matrix rows"]
    pub const fn new() -> Self {
        Self {
            node_rows: Vec::new(),
            next_row: 0,
        }
    }

    /// Assigns a row to every unknown node of `nodes`, in node-id order.
    ///
    /// Ground and any node whose kind is not an unknown are left without a row.
    /// The whole mapping is rebuilt, so call this after the circuit's nodes are
    /// all known.
    pub fn rebuild(&mut self, nodes: &NodeTable) {
        self.rebuild_deferring(nodes, |_| false);
    }

    /// Like [`Self::rebuild`], but leaves the nodes `deferred` selects without
    /// a row, for [`Self::assign_deferred`] to number later. C numbers the
    /// nodes a device creates in `DEVsetup` (`CKTmkVolt`) after every parsed
    /// node, interleaved with branch currents (`CKTmkCur`).
    pub fn rebuild_deferring(&mut self, nodes: &NodeTable, deferred: impl Fn(NodeId) -> bool) {
        self.node_rows.clear();
        self.next_row = 0;
        for node in nodes.nodes() {
            if deferred(node.id) {
                self.node_rows.push(None);
            } else {
                self.push(node);
            }
        }
    }

    /// Numbers a node [`Self::rebuild_deferring`] deferred with the next free
    /// row and returns its row. A node that already has a row keeps it.
    pub fn assign_deferred(&mut self, node: NodeId) -> Option<usize> {
        let slot = self.node_rows.get_mut(node.index())?;
        if slot.is_none() {
            *slot = Some(self.next_row);
            self.next_row += 1;
        }
        *slot
    }

    fn push(&mut self, node: &Node) -> Option<usize> {
        let row = if node.kind.is_unknown() {
            let row = self.next_row;
            self.next_row += 1;
            Some(row)
        } else {
            None
        };
        debug_assert_eq!(node.id.index(), self.node_rows.len());
        self.node_rows.push(row);
        row
    }

    /// Assigns a row to a node that was added after the last rebuild.
    ///
    /// Returns `None` for ground.
    pub fn assign(&mut self, node: &Node) -> Option<usize> {
        while self.node_rows.len() < node.id.index() {
            self.node_rows.push(None);
        }
        if node.id.index() < self.node_rows.len() {
            return self.node_rows[node.id.index()];
        }
        self.push(node)
    }

    /// The matrix row for a node, or `None` for ground.
    #[must_use = "the row identifies the matrix coordinate or a reference node"]
    #[inline]
    pub fn node_row(&self, node: NodeId) -> Option<usize> {
        self.node_rows.get(node.index()).copied().flatten()
    }

    /// Resolve a node at an analysis boundary. Only ground has no row;
    /// a non-ground node absent from the finalized mapping is an error.
    ///
    /// # Errors
    /// The node is unknown or has no validated MNA row.
    pub fn checked_node_row(&self, node: NodeId) -> SpiceResult<Option<usize>> {
        if node == NodeId::GROUND {
            return Ok(None);
        }
        self.node_row(node)
            .map(Some)
            .ok_or_else(|| SpiceError::circuit(format!("unmapped MNA node {node}")))
    }

    /// Seed a two-terminal voltage while retaining the negative terminal's
    /// common-mode potential when both terminals are non-ground. This is
    /// a guess or a local instance-IC evaluation, not a voltage source.
    pub(crate) fn seed_terminal_voltage(
        &self,
        state: &mut Vector,
        terminals: [NodeId; 2],
        voltage: Real,
    ) -> SpiceResult<()> {
        let [positive, negative] = terminals;
        let p = self.checked_node_row(positive)?;
        let n = self.checked_node_row(negative)?;
        if !voltage.is_finite() || state.len() != self.len() {
            return Err(SpiceError::circuit("invalid initial voltage seed"));
        }
        if p == n {
            if voltage != 0. {
                return Err(SpiceError::circuit(
                    "nonzero initial voltage across identical terminals",
                ));
            }
        } else if let Some(p) = p {
            let common = n.map_or(0., |n| state.as_slice()[n]);
            let value = common + voltage;
            if !value.is_finite() {
                return Err(SpiceError::circuit("nonfinite initial voltage seed"));
            }
            state.set(p, value)?;
        } else if let Some(n) = n {
            state.set(n, -voltage)?;
        }
        Ok(())
    }

    /// Reserves `count` rows after every node row, for branch currents.
    ///
    /// Returns the first reserved row, or `None` when `count` is zero.
    ///
    /// # Errors
    /// MNA row count overflow.
    pub fn add_rows(&mut self, count: usize) -> SpiceResult<Option<usize>> {
        if count == 0 {
            return Ok(None);
        }
        let first = self.next_row;
        self.next_row = first
            .checked_add(count)
            .ok_or_else(|| SpiceError::circuit("MNA row count overflow"))?;
        Ok(Some(first))
    }

    /// Number of unknowns, i.e. the size of the MNA system.
    #[must_use = "the unknown count determines the MNA system dimension"]
    pub const fn len(&self) -> usize {
        self.next_row
    }

    /// True when the system has no unknowns, which means the circuit is empty.
    #[must_use = "the empty-system result determines whether matrix rows exist"]
    pub const fn is_empty(&self) -> bool {
        self.next_row == 0
    }
}

impl Default for MnaUnknowns {
    fn default() -> Self {
        Self::new()
    }
}

/// What a device's convergence test ([`Device::convergence_test`]) sees.
#[derive(Debug)]
pub struct ConvergenceContext<'a> {
    /// Where the unknowns live.
    pub unknowns: &'a MnaUnknowns,
    /// The new iterate (C `CKTrhs`).
    pub solution: &'a Vector,
    /// The iterate the last load used (C `CKTrhsOld`).
    pub previous: &'a Vector,
    /// Branch rows of [`Device::controlling_sources`], in order.
    pub controls: &'a [usize],
    /// This device's first allocated branch, if any.
    pub branch: Option<usize>,
    /// Time-integration formula of the load, absent in DC.
    pub integration: Option<&'a Coefficients>,
    /// This device's slots of the last load's trial (C `CKTstate0`).
    pub states: &'a [Real],
    /// Accepted state revision corresponding to the trial.
    pub history: &'a crate::devices::StateHistory,
    /// Range of this device's state in an accepted vector.
    pub state_range: Range<usize>,
    /// The Newton phase of the last load (C's `MODEINITF` bits when
    /// `NIconvTest` runs, which `NIiter` does in `MODEINITFIX` as well as
    /// `MODEINITFLOAT`).
    pub phase: crate::devices::IterationPhase,
    /// Temperatures, `gmin` and frequency of the solve.
    pub model_context: &'a crate::devices::ModelContext,
    /// The time of a transient solve, zero otherwise (C `CKTtime`).
    pub time: Real,
    /// C `CKTreltol`.
    pub reltol: Real,
    /// C `CKTabstol`.
    pub abstol: Real,
    /// C `CKTvoltTol`.
    pub vntol: Real,
}

impl ConvergenceContext<'_> {
    /// A device-local accepted state value, age one being the latest.
    #[must_use = "the returned value is the accepted state used for device convergence checks"]
    pub fn accepted(&self, age: usize, slot: usize) -> Option<Real> {
        self.history
            .accepted(age)?
            .get(self.state_range.clone())?
            .get(slot)
            .copied()
    }

    /// The new iterate's value at a node, zero at ground.
    #[must_use = "the solved node potential is needed to evaluate the requested device quantity"]
    pub fn voltage(&self, node: NodeId) -> Real {
        self.unknowns
            .node_row(node)
            .and_then(|row| self.solution.get(row))
            .unwrap_or(0.0)
    }
}

/// RHS sink used by full and isolated device loads. Isolated stamps retain
/// only the rows actually written, rather than clearing a circuit-sized vector.
#[derive(Debug)]
pub struct StampRhs<'a>(RhsTarget<'a>);

#[derive(Debug)]
enum RhsTarget<'a> {
    Dense(&'a mut Vector),
    Sparse(&'a mut IsolatedRhs),
}

impl<'a> StampRhs<'a> {
    /// Stamp into the complete solver RHS.
    #[must_use = "the returned writer is required to stamp the dense right-hand side"]
    pub const fn dense(rhs: &'a mut Vector) -> Self {
        Self(RhsTarget::Dense(rhs))
    }

    pub(super) const fn isolated(rhs: &'a mut IsolatedRhs) -> Self {
        Self(RhsTarget::Sparse(rhs))
    }

    /// Add a contribution at a matrix row, preserving stamp order.
    ///
    /// # Errors
    /// The row is outside the finalized MNA dimensions.
    #[inline]
    pub fn add_to(&mut self, row: usize, value: Real) -> SpiceResult<()> {
        match &mut self.0 {
            RhsTarget::Dense(rhs) => rhs.add_to(row, value),
            RhsTarget::Sparse(rhs) => rhs.write(row, |old| old + value),
        }
    }
}

/// Sparse scratch for a single isolated load. Capacity is retained across
/// devices and points; rows are unique and sorted.
#[derive(Debug)]
pub(super) struct IsolatedRhs {
    rows: usize,
    values: Vec<(usize, Real)>,
}

impl IsolatedRhs {
    pub(super) const fn new(rows: usize) -> Self {
        Self {
            rows,
            values: Vec::new(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.values.clear();
    }

    fn write(&mut self, row: usize, update: impl FnOnce(Real) -> Real) -> SpiceResult<()> {
        if row >= self.rows {
            return Err(SpiceError::Numerical {
                context: "isolated RHS".to_owned(),
                message: format!("row {row} is out of range for length {}", self.rows),
            });
        }
        match self.values.binary_search_by_key(&row, |&(row, _)| row) {
            Ok(index) => self.values[index].1 = update(self.values[index].1),
            Err(index) => self.values.insert(index, (row, update(0.))),
        }
        Ok(())
    }

    pub(super) fn set(&mut self, row: usize, value: Real) -> SpiceResult<()> {
        self.write(row, |_| value)
    }

    pub(super) fn get(&self, row: usize) -> Real {
        self.values
            .binary_search_by_key(&row, |&(row, _)| row)
            .map_or(0., |index| self.values[index].1)
    }
}

/// Everything a device needs in order to stamp itself.
#[derive(Debug)]
pub struct StampContext<'a> {
    /// The MNA matrix being assembled.
    pub matrix: &'a mut SparseMatrix,
    /// The right-hand side being assembled.
    pub rhs: StampRhs<'a>,
    /// Where the unknowns live.
    pub unknowns: &'a MnaUnknowns,
    /// The node table, for diagnostics.
    pub nodes: &'a NodeTable,
    /// The present solution, linearised around it for nonlinear devices.
    pub solution: &'a Vector,
    /// Circuit temperature in degrees Celsius; model instances may override it.
    pub temperature: Real,
    /// Default model nominal temperature in degrees Celsius.
    pub nominal_temperature: Real,
    /// Junction minimum conductance (S), [`crate::devices::ModelContext::gmin`].
    pub gmin: Real,
    /// Configured physical damping conductance basis, not temporary junction gmin.
    pub configured_gmin: Real,
    /// Resolved circuit relative tolerance, not a Newton-local override.
    pub reltol: Real,
    /// The `hertz` frequency (Hz), [`crate::devices::ModelContext::frequency`].
    pub frequency: Real,
    /// Which analysis is loading the matrix.
    pub mode: AnalysisMode,
    /// Enclosing analysis for phase-dependent physical coefficients.
    pub analysis_kind: Option<crate::primitives::AnalysisKind>,
    /// Enclosing analysis for source-prepared passive bias constraints.
    pub bias_analysis: Option<crate::primitives::AnalysisKind>,
    /// Branch-current rows allocated to this device, in order (empty if none).
    pub branches: Range<usize>,
    /// Branch rows of the devices named by [`Device::controlling_sources`], in
    /// that order (empty for devices that sense no branch current).
    pub controls: &'a [usize],
    /// Mutual-inductance terms of this device's branch equation (empty except
    /// for inductors named by a K device); see [`MutualTerm`].
    pub mutual: &'a [MutualTerm],
    /// Companion integration coefficients for this trial step; `None` outside
    /// companion transient loads.
    pub integration: Option<&'a Coefficients>,
    /// This device's trial and accepted state slots.
    pub states: DeviceState<'a>,
    /// Source-forcing context of a companion transient load; `None` otherwise.
    pub forcing: Option<Forcing>,
    /// C's `CKTsrcFact`, [`crate::devices::ModelContext::source_factor`].
    pub source_factor: Real,
    /// C's running `value` in `VSRCload`: the value the previous voltage
    /// source in load order loaded (0 before the first). Each independent
    /// voltage source writes its own; a PORT adds its cosine to it.
    pub voltage_source_value: &'a mut Real,
}

impl StampContext<'_> {
    /// The temperatures and junction `gmin` of this load as a [`crate::devices::ModelContext`]
    /// (without resistor or instance-parameter overrides, which
    /// [`crate::devices::Circuit`] has already applied).
    #[must_use = "the model context carries the analysis temperature and numerical settings"]
    pub const fn model_context(&self) -> crate::devices::ModelContext {
        let mut context =
            crate::devices::ModelContext::new(self.temperature, self.nominal_temperature)
                .with_gmin(self.configured_gmin)
                .with_junction_gmin(self.gmin)
                .with_reltol(self.reltol)
                .with_frequency(self.frequency);
        context.analysis_kind = self.analysis_kind;
        context.bias_analysis = self.bias_analysis;
        context
    }

    /// The `index`-th branch row of this device.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Circuit`] when the device has no such branch.
    pub fn branch(&self, index: usize) -> SpiceResult<usize> {
        self.branches
            .clone()
            .nth(index)
            .ok_or_else(|| SpiceError::circuit("missing branch-row binding"))
    }

    /// The present solution value of a matrix row.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Numerical`] for a row outside the solution.
    pub fn row_value(&self, row: usize) -> SpiceResult<Real> {
        self.solution.get(row).ok_or_else(|| SpiceError::Numerical {
            context: "stamp".to_owned(),
            message: format!("solution row {row} out of range"),
        })
    }

    /// Adds `value` to `A[row][col]`, ignoring contributions that involve ground.
    ///
    /// # Errors
    ///
    /// [`crate::primitives::SpiceError::Numerical`] if a row is out of range, which
    /// would be a bug in the caller rather than bad input.
    #[inline]
    pub fn stamp(&mut self, row: NodeId, col: NodeId, value: Real) -> SpiceResult<()> {
        let (Some(row), Some(col)) = (self.unknowns.node_row(row), self.unknowns.node_row(col))
        else {
            return Ok(());
        };
        self.matrix.add(row, col, value)
    }

    /// [`Self::stamp`] for each entry in order, for a device whose element
    /// positions are the same on every load (C stamps them through the
    /// pointers `DEVsetup` took with `TSTALLOC`). Once a load has verified
    /// the batch against the matrix's recorded stamp sequence, later loads
    /// add the values by element ([`SparseMatrix::add_fixed`]); the matrix
    /// receives the same additions in the same order either way.
    ///
    /// # Errors
    ///
    /// As [`Self::stamp`].
    #[inline]
    pub fn stamp_fixed<const N: usize>(
        &mut self,
        entries: [(NodeId, NodeId, Real); N],
    ) -> SpiceResult<()> {
        if self.matrix.add_fixed(&entries.map(|(_, _, value)| value)) {
            return Ok(());
        }
        let start = self.matrix.fixed_start();
        let mut mask = 0;
        for (k, (row, col, value)) in entries.into_iter().enumerate() {
            if let (Some(row), Some(col)) =
                (self.unknowns.node_row(row), self.unknowns.node_row(col))
            {
                // Batches longer than a mask holds are never verified.
                mask |= u32::try_from(k)
                    .ok()
                    .and_then(|k| 1u64.checked_shl(k))
                    .unwrap_or(0);
                self.matrix.add(row, col, value)?;
            }
        }
        self.matrix.verify_fixed(start, N, mask);
        Ok(())
    }

    /// Adds `value` to `b[row]`, ignoring ground.
    ///
    /// # Errors
    ///
    /// [`crate::primitives::SpiceError::Numerical`] if the row is out of range.
    #[inline]
    pub fn stamp_rhs(&mut self, row: NodeId, value: Real) -> SpiceResult<()> {
        let Some(row) = self.unknowns.node_row(row) else {
            return Ok(());
        };
        self.rhs.add_to(row, value)
    }

    /// The present solution value at a node, or zero at ground.
    #[must_use = "the node potential determines the voltage-dependent load contribution"]
    #[inline]
    pub fn node_voltage(&self, node: NodeId) -> Real {
        self.unknowns
            .node_row(node)
            .and_then(|row| self.solution.get(row))
            .unwrap_or(0.0)
    }

    /// The name of a node, for diagnostics.
    #[must_use = "the returned name identifies the node in load diagnostics"]
    pub fn node_name(&self, node: NodeId) -> &str {
        self.nodes
            .node(node)
            .map_or("<unknown node>", |node| node.name.as_str())
    }
}

/// A device instance in a circuit.
///
/// The C equivalent is the `CKTdevice` vtable plus its per-device `*load`,
/// `*accept`, `*ask` and `*delete` functions.
///
/// [`fmt::Debug`] is a supertrait so that a `Box<dyn Device>` can be printed in
/// diagnostics and compared in tests.
pub trait Device: fmt::Debug + std::any::Any {
    /// Scalar instance ask (`*ask.c`), evaluated at the point's model context.
    /// Unavailable keywords return `None`; callers report an explicit error.
    /// # Errors
    /// Invalid contextual temperature or derived value.
    fn observation_parameter(
        &self,
        _keyword: &str,
        _context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<Real>> {
        Ok(None)
    }
    /// Independent current-source forcing for observations, including multiplicity.
    /// # Errors
    /// Invalid or unresolved source waveform.
    fn observation_source_current(
        &self,
        _time: Option<(Real, crate::devices::Forcing)>,
    ) -> SpiceResult<Option<Real>> {
        Ok(None)
    }
    /// Current-source parallel multiplier, used for swept forcing observations.
    fn observation_source_multiplier(&self) -> Real {
        1.
    }
    /// Bias-dependent instance ask (`*ask.c` quantities C reads from
    /// `CKTstate0`, e.g. a FET's `gm`) at the converged solution, whose node
    /// voltages `voltage` returns; `transient` is set for transient points.
    /// Unavailable keywords return `None` (the default); callers report an
    /// explicit error.
    /// # Errors
    /// Invalid contextual temperature, bias physics, or a quantity whose C
    /// value includes state this hook cannot reproduce.
    fn observation_operating(
        &self,
        _keyword: &str,
        _context: &crate::devices::ModelContext,
        _voltage: &dyn Fn(NodeId) -> Real,
        _transient: bool,
    ) -> SpiceResult<Option<Real>> {
        Ok(None)
    }
    /// Physical two-terminal current retained by a converged derivative
    /// array. Returning a value avoids scalar evaluation at a branch event.
    /// The default leaves the existing isolated observation load in use.
    fn physical_terminal_current(&self, _states: &[Real], _branches: &[Real]) -> Option<Real> {
        None
    }

    /// Factor applied to this device's observed terminal currents: `-1` for
    /// devices whose C ask reports polarity-normalized currents for the
    /// p-type variant (`jfetask.c`), `1` (the default) otherwise.
    fn observation_current_sign(&self) -> Real {
        1.
    }

    /// The `.model` card this instance uses, when the device is model-backed.
    /// Diagnostics and model-binning tests read it; `None` otherwise.
    fn model_name(&self) -> Option<&str> {
        None
    }

    /// Whether finalized topology must prepare per-candidate random streams.
    /// False by default; deterministic sources add no candidate dispatch work.
    fn has_stochastic_source(&self) -> bool {
        false
    }

    /// Rebind only transient stochastic source identity for an independent
    /// optimizer candidate, leaving terminal and state layout unchanged.
    /// Other devices return None; this does not alter deterministic sources.
    fn with_stochastic_candidate(&self, _identity: u64) -> Option<Box<dyn Device>> {
        None
    }

    /// The instance name, e.g. `r1`.
    fn name(&self) -> &str;

    /// The designator letter, lowercased.
    fn designator(&self) -> char;

    /// The nodes this device is connected to, in numerical binding order.
    fn terminals(&self) -> &[NodeId];

    /// Public ports in source order, excluding setup-private
    /// coordinates. Used for external current and power observations.
    fn external_terminals(&self) -> &[NodeId] {
        self.terminals()
    }

    /// Named internal node coordinates whose rows are constitutive equations,
    /// not electrical KCL. They do not receive physical voltage shunts or
    /// voltage tolerances; the owner checks their physical residuals.
    fn constitutive_nodes(&self) -> &[NodeId] {
        &[]
    }

    /// Install an occurrence-bound prepared passive startup policy before setup.
    ///
    /// # Errors
    /// This device is not the matching physical storage family.
    fn bind_passive_initialization(
        &mut self,
        _policy: crate::devices::passive_bias::PassiveInitialization,
    ) -> SpiceResult<()> {
        Err(SpiceError::circuit(
            "prepared initialization requires a physical capacitor or inductor",
        ))
    }

    /// True when bias equations need physical nodal continuation rather
    /// than perturbing a numerical pivot diagonal.
    fn has_passive_bias(&self) -> bool {
        false
    }

    /// Whether an explicit physical initialization equation applies in this bias analysis.
    fn has_active_passive_bias(&self, _analysis: crate::primitives::AnalysisKind) -> bool {
        false
    }

    /// Whether real loads require the enclosing analysis selected by the
    /// analysis plan, independently of numerical regularization.
    fn requires_analysis_context(&self) -> bool {
        self.has_passive_bias()
    }

    /// How many device-local coordinates the device adds to the MNA
    /// system, including branch currents and auxiliary constitutive states.
    fn branch_currents(&self) -> usize {
        0
    }

    /// Observable meaning of a local coordinate. Ordinary branch-current
    /// devices keep their reference representation; stateful native devices
    /// identify their additional coordinates explicitly.
    fn branch_quantity(&self, _branch: usize) -> BranchQuantity {
        BranchQuantity::Current
    }

    /// Convergence classification of a device-local branch coordinate.
    /// A constitutive coordinate must be checked by `convergence_test`.
    fn branch_convergence(&self, _branch: usize) -> ConvergenceKind {
        ConvergenceKind::Current
    }

    /// Devices whose branch current this device senses (F/H controlling
    /// sources), resolved to rows passed as `controls` in [`StampContext`] and
    /// [`crate::devices::LinearContext`]. Empty (the default) for everything else.
    fn controlling_sources(&self) -> &[ControlReference] {
        &[]
    }

    /// Which of this device's branch currents another device may sense by
    /// name, as an index into its branch rows (C `DEVfindBranch`: `VSRCfindBr`,
    /// `VCVSfindBr`, `CCVSfindBr`). `None` (the default) for devices C's
    /// `CKTfndBranch` cannot find, including inductors.
    fn findable_branch(&self) -> Option<usize> {
        None
    }

    /// Physical branch for a resolved expression's current probe.
    /// Unlike voltage-source-only controls, this can read a storage current.
    fn expression_current_branch(&self) -> Option<usize> {
        self.findable_branch()
    }

    /// Which current coordinate this device reads from a control target.
    /// Traditional F/H/W cards retain their defined voltage-source semantics.
    fn control_branch(&self, target: &dyn Device) -> Option<usize> {
        target.findable_branch()
    }

    /// Binds the previous voltage-source value in C `VSRCload`'s reverse-deck
    /// traversal. Non-voltage devices return None; PORT adds to that value.
    /// # Errors
    /// Invalid accumulated RF excitation.
    fn bind_voltage_predecessor(
        &mut self,
        _waveform: &crate::devices::Waveform,
        _dc: Real,
    ) -> SpiceResult<Option<(crate::devices::Waveform, Real)>> {
        Ok(None)
    }

    /// The RFSPICE port data, or None for every other device.
    fn rf_port(&self) -> Option<&crate::devices::sources::RfPort> {
        None
    }

    /// True when the device's contribution depends on the present solution, so
    /// the analysis has to iterate.
    fn is_nonlinear(&self) -> bool {
        false
    }

    /// Whether Newton's global voltage-step damping (the port's stand-in for
    /// C's junction limiting) should watch this device's terminals. Only
    /// meaningful for [`Self::is_nonlinear`] devices; `true` by default.
    /// Behavioural sources return `false`: `asrcload.c` applies no limiting,
    /// and their outputs may legitimately move by kilovolts in one exact
    /// step.
    fn limits_voltage_steps(&self) -> bool {
        true
    }

    /// True when the instance carries operating-point start settings that
    /// only C's phased, device-limited Newton loads apply: an `off` flag
    /// (`MODEINITJCT`/`MODEINITFIX`) or MOS1 `ic=` start voltages
    /// (`MODEINITJCT`). Drivers running the port's legacy global damping
    /// (`limiting=global`, exact loads) must reject such circuits instead of
    /// ignoring the settings. `false` by default.
    fn has_start_settings(&self) -> bool {
        false
    }

    /// True when the device's DC equations read the analysis frequency
    /// (a behavioural source using `hertz`): AC then re-solves the operating
    /// point at every frequency, as `acan.c` does when `CKTvarHertz` is set.
    fn depends_on_frequency(&self) -> bool {
        false
    }

    /// Whether a device's equations depend explicitly on transient time.
    /// Independent source waveforms are checked separately at analysis binding.
    fn depends_on_time(&self) -> bool {
        false
    }

    /// True when the device's small-signal (AC) stamp itself depends on the
    /// analysis frequency in a way the affine `A + j omega E` pencil cannot
    /// express, e.g. a transmission line's `exp(-j omega TD)` (`traacld.c`).
    /// Small-signal drivers then reassemble the system at every frequency
    /// with [`crate::devices::ModelContext::frequency`] set, without
    /// re-solving the operating point (unlike [`Self::depends_on_frequency`]).
    /// `false` by default.
    fn small_signal_depends_on_frequency(&self) -> bool {
        false
    }

    /// Identity of a homogeneous prepared artifact, if this device supports
    /// tile evaluation. Only exact artifact identity may share a tile.
    fn prepared_batch_key(&self) -> Option<crate::devices::prepared_model::batch::Key> {
        None
    }

    /// Exact preparation inputs for detecting unchanged tile lanes.
    fn prepared_recipe(&self) -> Option<crate::devices::prepared_model::batch::Recipe<'_>> {
        None
    }

    /// Prepare a bounded homogeneous tile at a candidate-context boundary.
    /// # Errors
    /// Incompatible instances, invalid parameters or unsupported tile.
    fn prepare_batch(
        &self,
        _instances: &[&dyn Device],
        _unknowns: &MnaUnknowns,
    ) -> SpiceResult<Box<dyn crate::devices::prepared_model::batch::Tile>> {
        Err(SpiceError::circuit(
            "device has no prepared batch implementation",
        ))
    }

    /// Whether a prepared lane uses solver-owned trial linearization state.
    fn needs_prepared_linearization(&self) -> bool {
        false
    }

    /// Apply solver-owned trial linearization to one prepared lane before
    /// constitutive evaluation. Exact loads retain the original coordinates.
    /// # Errors
    /// Invalid coordinate or history state.
    fn linearize_prepared(
        &self,
        _states: &mut DeviceState<'_>,
        _coordinates: &mut [Real],
    ) -> SpiceResult<()> {
        Ok(())
    }

    /// Stamp a previously evaluated tile lane into a trial.
    /// # Errors
    /// Invalid numerical or history state.
    fn stamp_prepared(
        &self,
        _context: &mut StampContext<'_>,
        _point: &model_kernel_api::Point<'_>,
        _coordinates: &[Real],
    ) -> SpiceResult<()> {
        Err(SpiceError::circuit("device has no prepared batch stamp"))
    }

    /// Assemble an already evaluated tile lane into the affine AC pencil.
    /// # Errors
    /// Missing charge equations or invalid row binding.
    fn assemble_prepared(
        &self,
        _context: &mut crate::devices::linear::LinearContext<'_>,
        _point: &model_kernel_api::Point<'_>,
    ) -> SpiceResult<()> {
        Err(SpiceError::circuit("device has no prepared batch AC stamp"))
    }

    /// Whether the noise generator densities or injection weights depend
    /// on the requested spectral frequency, independently of the AC pencil.
    fn noise_depends_on_frequency(&self) -> bool {
        false
    }

    /// The device's accepted-waveform delay history behaviour (transmission
    /// lines), or `None` (the default). See [`crate::devices::delay`]: such a
    /// device reads its history while stamping
    /// ([`crate::devices::DeviceState::delay_history`]), and the companion
    /// transient driver records accepted samples, device breakpoints and step
    /// bounds through it. Any future backend that cannot integrate a delay
    /// must refuse circuits containing one.
    fn delay_line(&self) -> Option<&dyn crate::devices::delay::DelayLine> {
        None
    }

    /// True when the device holds a discrete state (a switch position) that
    /// a last-bit change of its control can flip, so drivers must reproduce
    /// C's exact control values (e.g. `dctrcurv.c`'s accumulated sweep values).
    fn has_discrete_state(&self) -> bool {
        false
    }

    /// How many state slots (C `CKTnumStates`) the device owns. Slots are
    /// allocated after the branch rows, in device order.
    fn state_count(&self) -> usize {
        0
    }

    /// The state slot holding the integrated charge or flux that takes part in
    /// local-truncation-error control; the *next* slot holds its derivative.
    /// `None` (the default) for devices without a `DEVtrunc` equivalent
    /// (`captrunc.c`, `indtrunc.c`, `CKTterr`).
    fn truncation_slot(&self) -> Option<usize> {
        None
    }

    /// All charge/flux pairs participating in truncation control. Nonlinear
    /// multi-junction devices override this; each listed slot is followed by
    /// its time derivative. Existing single-storage devices retain their API.
    fn truncation_slots(&self) -> Vec<usize> {
        self.truncation_slot().into_iter().collect()
    }

    /// The coupling a K (mutual inductance) device describes; `None` (the
    /// default) for every other device. [`crate::devices::Circuit`] resolves the names
    /// to inductors and hands each one its [`MutualTerm`]s.
    fn mutual_coupling(&self) -> Option<MutualCoupling<'_>> {
        None
    }

    /// The inductance under `context` when this device is an inductor that a
    /// K device may couple (C `CKTfndDev` on an inductor instance); `None`
    /// (the default) for everything else.
    fn inductance(
        &self,
        _context: &crate::devices::models::ModelContext,
    ) -> Option<SpiceResult<InductanceValue>> {
        None
    }

    /// The charge/flux-storage description used to seed initial conditions
    /// (`CAPgetic`/`INDgetic`-style `ic=` handling, see `capload.c`/`indload.c`).
    /// `None` (the default) for devices that store no charge or flux in the
    /// state slots named by [`Self::truncation_slot`]. The value is evaluated
    /// under `context` (model-backed C/L depend on temperature, TC, scale and
    /// `m`).
    fn storage_element(
        &self,
        _context: &crate::devices::models::ModelContext,
    ) -> Option<SpiceResult<StorageElement>> {
        None
    }

    /// Ideal voltage relations visible to initial-condition consistency
    /// checking. A multiport device may publish more than one relation.
    /// Only DC bias regards inductors as shorts; under UIC their stored
    /// current is constrained instead.
    fn initial_voltage_relations(&self, inductors_short: bool) -> Vec<InitialVoltageRelation> {
        let law = match self.designator() {
            'e' | 'h' => InitialVoltageLaw::Controlled,
            'v' if self.branch_currents() == 1 => InitialVoltageLaw::Source(0),
            'l' if inductors_short => InitialVoltageLaw::Fixed(0.),
            _ => return Vec::new(),
        };
        let terminals = match (self.terminals(), self.rf_port()) {
            ([_, negative, _], Some(port)) => [port.internal, *negative],
            ([positive, negative, ..], None) => [*positive, *negative],
            _ => return Vec::new(),
        };
        vec![InitialVoltageRelation { terminals, law }]
    }

    /// Explicit internal storage coordinates for UIC, independent of
    /// the physical current/voltage storage representation of scalar RLC.
    /// The device seeds its history from these coordinates during the
    /// initial load; the analysis binds and checks the instantaneous system.
    /// # Errors
    /// Invalid effective storage inputs.
    fn initial_coordinates(
        &self,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Vec<InitialCoordinate>> {
        Ok(self
            .storage_element(context)
            .transpose()?
            .filter(|storage| storage.kind == StorageKind::Inductor)
            .map(|storage| InitialCoordinate {
                coordinate: 0,
                equation: 0,
                value: storage.initial.unwrap_or(0.),
                history: InitialCoordinateHistory::EquationCharge {
                    slot: 0,
                    scale: -1.,
                },
            })
            .into_iter()
            .collect())
    }

    /// Initial differential-voltage storage and the corresponding equation
    /// binding, independent of the physical charge scale.
    /// # Errors
    /// Invalid effective storage inputs or missing state slot.
    fn initial_voltage_storage(
        &self,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<InitialVoltageStorage>> {
        let Some(storage) = self
            .storage_element(context)
            .transpose()?
            .filter(|storage| storage.kind == StorageKind::Capacitor)
        else {
            return Ok(None);
        };
        let slot = self
            .truncation_slot()
            .ok_or_else(|| SpiceError::circuit("missing capacitor storage slot"))?;
        Ok(Some(InitialVoltageStorage {
            voltage: storage.initial,
            slot,
            history: VoltageStorageHistory::Linear(storage.value),
            equation: None,
        }))
    }

    /// Nonlinear storage expression constraints used during UIC. Devices
    /// with explicit differential voltage/current or transfer coordinates
    /// need not also declare an expression constraint.
    fn initial_expression_storage(&self) -> Option<InitialExpressionStorage> {
        None
    }

    /// Validate bias-independent device inputs at a DC point, without an
    /// AC load, Newton trial, or accepted-state mutation. Context-independent
    /// devices need no extra checks beyond construction and replacement.
    /// Context-dependent devices must check their thermal/effective parameters
    /// here using the same derivation as their DC stamp. The recipe already
    /// carries its own setters; the context contains common scalar inputs,
    /// not per-device override tables. Cross-device mutual constraints are
    /// validated by the circuit. Solution-dependent errors belong to the
    /// actual solve, not to a fabricated zero bias.
    ///
    /// # Errors
    /// Invalid contextual parameters or nonfinite derived device inputs.
    fn validate_dc_inputs(&self, _context: &crate::devices::ModelContext) -> SpiceResult<()> {
        Ok(())
    }

    /// Loads the device's contribution into the MNA system for one trial.
    ///
    /// Implementations may read accepted history and write their trial state
    /// slots, but cannot change the device: a rejected or repeated trial has
    /// no lasting effect.
    ///
    /// # Errors
    ///
    /// Device-specific failures, and [`crate::primitives::SpiceError::NotYetPorted`] for
    /// devices that have not been ported.
    fn stamp(&self, context: &mut StampContext<'_>) -> SpiceResult<()>;

    /// C's `DEVconvTest`, run by `NIconvTest` (`CKTconvTest`) after the node
    /// voltages and branch currents of a Newton pass converged: `true` when
    /// this device has not converged. Only devices whose C type has a
    /// convergence test override it; for the diode and BJT C's test passes
    /// whenever the load did not already report nonconvergence (it measures
    /// limiting at the load point), so they rely on the load's flag.
    ///
    /// # Errors
    /// An evaluation failure (C `E_BADPARM`).
    fn convergence_test(&self, _context: &ConvergenceContext<'_>) -> SpiceResult<bool> {
        Ok(false)
    }

    /// The C device type whose `DEVsetup`/`DEVload` handles this instance,
    /// which places it in [`crate::devices::setup`]'s traversal. The default
    /// maps the designator; devices whose type depends on the model (MOS
    /// levels, the lowered behavioural forms) override it.
    fn c_device_type(&self) -> crate::devices::setup::CDeviceType {
        crate::devices::setup::CDeviceType::of_designator(self.designator())
            .unwrap_or(crate::devices::setup::CDeviceType::CodeModel)
    }

    /// Allocates this device's matrix elements in its C `DEVsetup`
    /// `TSTALLOC` order, including elements whose value may be zero in some
    /// analyses. Returns `false` when the device does not declare them, in
    /// which case its elements are learned from its first load and appended.
    ///
    /// # Errors
    /// Missing branch or control bindings.
    fn setup_matrix(
        &self,
        _setup: &mut crate::devices::setup::MatrixSetup<'_>,
    ) -> SpiceResult<bool> {
        Ok(false)
    }

    /// Assembles immutable, state-independent linear equations.
    ///
    /// # Errors
    /// Unsupported devices must not silently contribute zero.
    fn assemble_linear(
        &self,
        _context: &mut crate::devices::linear::LinearContext<'_>,
    ) -> SpiceResult<()> {
        Err(crate::primitives::SpiceError::Unsupported {
            feature: format!("linear equation assembly for {}", self.name()),
            location: None,
        })
    }

    /// Evaluate a nonlinear physical charge law separately from the
    /// integration companion and small-signal AC conventions.
    ///
    /// # Errors
    /// Unsupported device charge physics or a failed model evaluation.
    fn assemble_nonlinear_charge(&self, _context: &mut ChargeContext<'_>) -> SpiceResult<()> {
        Err(SpiceError::Unsupported {
            feature: format!("nonlinear frequency-domain charge for {}", self.name()),
            location: None,
        })
    }

    /// Visit device-local charge/rate pairs and their electrical incidence.
    /// The same declaration binds physical histories and reconstructs their
    /// displacement currents in instantaneous terminal observations.
    /// # Errors
    /// Propagates an error from the visitor.
    fn physical_charge_histories(
        &self,
        _visit: &mut dyn FnMut(PhysicalChargeHistory) -> SpiceResult<()>,
    ) -> SpiceResult<()> {
        Ok(())
    }

    /// Assemble nonlinear physical residual/charge Taylor coefficients and
    /// the corresponding state-Jacobian jets.
    ///
    /// # Errors
    /// No analytic Taylor kernel for this device or invalid local physics.
    fn assemble_nonlinear_taylor(
        &self,
        _context: &mut crate::devices::physical_dae::TaylorContext<'_>,
    ) -> SpiceResult<()> {
        Err(SpiceError::Unsupported {
            feature: format!("physical Taylor equations for {}", self.name()),
            location: None,
        })
    }

    /// Assemble the bias-linearized conductance and charge Jacobians for AC.
    /// Linear devices reuse their state-independent operators; nonlinear devices
    /// must explicitly implement this and must not provide a DC equivalent RHS.
    /// # Errors
    /// Invalid bias or unsupported small-signal physics.
    fn assemble_small_signal(
        &self,
        context: &mut crate::devices::linear::LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        self.assemble_linear(context)
    }

    /// Assemble the device's pole-zero load at the bias point: its part of
    /// the s-domain pencil `A + s E` (C `DEVpzLoad`, called by `CKTpzLoad`
    /// in `cktpzld.c`). Every C pole-zero load is affine in `s` and equals
    /// the AC load with `j omega` replaced by `s`, except an independent
    /// voltage source with an AC value, which `vsrcpzld.c` removes from the
    /// circuit (its current is forced to zero) because the pole-zero drive
    /// takes its place.
    ///
    /// Opt-in: the default is an explicit error, so a device whose C model
    /// has no `DEVpzLoad` (or whose pole-zero load the port has not
    /// reviewed) can never contribute nothing silently. Devices whose C
    /// pole-zero load equals their small-signal load delegate to
    /// [`Self::assemble_small_signal`].
    ///
    /// # Errors
    /// Unsupported devices, or the small-signal assembly's errors.
    fn assemble_pole_zero(
        &self,
        _context: &mut crate::devices::linear::LinearContext<'_>,
        _bias: &Vector,
    ) -> SpiceResult<()> {
        Err(SpiceError::Unsupported {
            feature: format!(
                "pole-zero analysis of {} (no pole-zero load; C devices without DEVpzLoad, \
                 e.g. transmission lines and XSPICE code models, are not supported)",
                self.name()
            ),
            location: None,
        })
    }

    /// Validate a prospective accepted point without observable side effects.
    /// All devices must approve before any accept notification is delivered.
    ///
    /// # Errors
    /// Device-specific failures.
    fn validate_accept(&self, _context: &AcceptContext<'_>) -> SpiceResult<()> {
        Ok(())
    }

    /// Infallible notification of an accepted point, after all acceptance
    /// checks succeeded and immediately before the histories are published.
    /// Never called for Newton trials or rejected steps.
    fn accept(&self, _context: &AcceptContext<'_>) {}

    /// Opt in to exact transient event scheduling, using ordinary accepted
    /// and tentative device state. No event queue is needed for devices
    /// whose next event follows directly from their accepted state.
    fn transient_events(&self) -> Option<&dyn DeviceTransientEvents> {
        None
    }

    /// An upper bound on the next transient step from this device's discrete
    /// state (C `DEVtrunc` of devices without charge storage, e.g.
    /// `swtrunc.c`), or `None` (the default) for no bound. Called for an
    /// accepted-candidate trial alongside the charge/flux truncation estimate;
    /// a bound at or below `0.9 dt` rejects the trial.
    ///
    /// # Errors
    ///
    /// Device-specific failures.
    fn timestep_limit(&self, _context: &TruncationContext<'_>) -> SpiceResult<Option<Real>> {
        Ok(None)
    }

    /// Physical metadata when this device is a two-terminal resistor a typed
    /// `.dc` sweep may target; `None` (the default) for everything else. This is
    /// the only way a sweep identifies a resistor: never the instance-name prefix.
    fn resistor_metadata(&self) -> Option<crate::devices::sweep::ResistorMetadata> {
        None
    }

    /// The effective resistance in ohms this resistor would stamp if its supplied
    /// scalar were `supplied`, under `context`'s temperatures. Immutable: nothing
    /// about the device changes. See [`crate::devices::sweep`] for the semantics.
    ///
    /// # Errors
    /// Not a resistor, or `supplied` / the derived value is invalid (nonfinite,
    /// zero, nonfinite conductance, nonpositive temperature factor).
    fn resistor_effective(
        &self,
        _supplied: Real,
        _context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<Real> {
        Err(SpiceError::circuit(format!(
            "{} is not a resistor",
            self.name()
        )))
    }

    /// The device's `.noise` generators at the operating point
    /// ([`crate::devices::noise`], C `DEVnoise`). Devices C gives no noise
    /// routine return [`crate::devices::noise::DeviceNoise::Noiseless`]
    /// explicitly; the default is an error, so a device whose noise is not
    /// ported can never be silently omitted from the spectrum.
    /// Append generators in reference order to the caller-provided empty
    /// scratch buffer; a noiseless device leaves it empty. Model identity
    /// borrows from the effective device.
    ///
    /// # Errors
    /// [`SpiceError::NotYetPorted`] by default; invalid bias physics.
    fn noise(
        &self,
        _context: &crate::devices::noise::NoiseContext<'_>,
        _sources: &mut Vec<crate::devices::noise::NoiseSource>,
    ) -> SpiceResult<crate::devices::noise::DeviceNoise<'_>> {
        Err(SpiceError::not_yet_ported(
            format!(
                "noise analysis of device {} (designator '{}')",
                self.name(),
                self.designator()
            ),
            "src/spicelib/devices/<dev>/<dev>noi*.c (DEVnoise)",
        ))
    }

    /// The effective independent-source waveform after circuit finalization
    /// (including RF port predecessor accumulation). Other devices have none.
    fn independent_waveform(
        &self,
    ) -> Option<(crate::devices::SourceKind, &crate::devices::Waveform)> {
        None
    }

    /// When this device is an independent V/I source: its kind and whether an
    /// `ac` value was given, as the `.noise` input reference needs them.
    /// `None` (the default) for every other device.
    fn input_source(&self) -> Option<crate::devices::noise::InputSource> {
        None
    }

    /// Whether small-signal assembly requires the converged operating-point
    /// trial state rather than the native reference's accepted history.
    fn small_signal_uses_trial_state(&self) -> bool {
        false
    }

    /// The state slots C's `MODEINITSMSIG` load writes at the operating
    /// point that later analyses of the deck read back (C keeps one
    /// `CKTstate0` for the whole deck): `slots` is this device's window of
    /// the carried state vector ([`crate::devices::Circuit::record_small_signal_state`]).
    /// The default writes nothing.
    ///
    /// # Errors
    /// Invalid bias physics.
    fn small_signal_state(
        &self,
        context: &crate::devices::distortion::DistortionContext<'_>,
        slots: &mut [Real],
    ) -> SpiceResult<()> {
        let _ = (context, slots);
        Ok(())
    }

    /// The device's `.disto` nonlinearities at the operating point
    /// ([`crate::devices::distortion`], C `DEVdisto`). Devices C gives no
    /// distortion routine but simulates through their AC load return
    /// [`crate::devices::distortion::DeviceDistortion::Linear`] explicitly;
    /// the default is an error, so a device whose distortion is not ported
    /// can never be silently treated as linear.
    ///
    /// # Errors
    /// [`SpiceError::NotYetPorted`] by default; invalid bias physics.
    fn distortion(
        &self,
        _context: &crate::devices::distortion::DistortionContext<'_>,
    ) -> SpiceResult<crate::devices::distortion::DeviceDistortion> {
        Err(SpiceError::not_yet_ported(
            format!(
                "distortion analysis of device {} (designator '{}')",
                self.name(),
                self.designator()
            ),
            "src/spicelib/devices/<dev>/<dev>disto.c (DEVdisto)",
        ))
    }
    /// Bind a numeric parameter to the device's immutable setter table.
    fn bind_numeric_parameter(&self, name: &str) -> Option<crate::devices::sweep::ParameterKey> {
        self.instance_parameter(name)
            .map(crate::devices::sweep::ParameterKey::Native)
    }

    /// Name of a bound numeric setter for diagnostics and output labels.
    fn numeric_parameter_name(&self, key: crate::devices::sweep::ParameterKey) -> Option<&str> {
        match key {
            crate::devices::sweep::ParameterKey::Native(name) => Some(name),
            crate::devices::sweep::ParameterKey::Prepared(_) => None,
        }
    }

    /// Dimensional transformation of a raw source setter. Effective device
    /// replacements always consume physical values, never apply scale again.
    fn parameter_geometry_scale(
        &self,
        _key: crate::devices::sweep::ParameterKey,
    ) -> crate::devices::sweep::GeometryScale {
        crate::devices::sweep::GeometryScale::Identity
    }

    /// Create an independent effective device with a previously bound setter.
    /// # Errors
    /// Foreign parameter or invalid physical input.
    fn with_bound_parameter(
        &self,
        key: crate::devices::sweep::ParameterKey,
        value: Real,
        context: &crate::devices::ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        match key {
            crate::devices::sweep::ParameterKey::Native(name) => {
                self.with_instance_parameter(name, value, context)
            }
            crate::devices::sweep::ParameterKey::Prepared(_) => {
                Err(SpiceError::circuit("foreign prepared parameter"))
            }
        }
    }

    /// Whether typed instance setters preserve the selected model declaration.
    /// Model-backed devices opt in only when no setter can change model
    /// selection (including binning). The candidate validator still checks
    /// each replacement's physical validity, topology and state layout.
    fn model_candidate_setters_are_stable(&self) -> bool {
        false
    }

    /// Whether this device needs context preparation, independently of batching.
    fn needs_context_preparation(&self) -> bool {
        self.prepared_batch_key().is_some()
    }

    /// Reprepare immutable context-dependent data at a context transition.
    /// Ordinary native devices need no replacement.
    /// # Errors
    /// Invalid effective physical parameters.
    fn reprepare_context(
        &self,
        _context: &crate::devices::ModelContext,
    ) -> SpiceResult<Option<Box<dyn Device>>> {
        Ok(None)
    }

    /// The canonical (lowercase, alias-folded) keyword of a settable real
    /// instance parameter that a `.dc @inst[param]` sweep may replace on this
    /// device (C `dctrcurv.c` `DCTfindInstParam`), or `None` (the default)
    /// when the port cannot sweep `keyword` on it. Independent-source `dc`
    /// and resistor `r` are swept by the source and resistor targets and are
    /// not reported here.
    fn instance_parameter(&self, _keyword: &str) -> Option<&'static str> {
        None
    }

    /// A disposable copy of this device with `parameter` (a keyword returned
    /// by [`Self::instance_parameter`]) given the value `value`, exactly as C's
    /// `DEVparam` setter followed by `DEVtemperature` would leave the
    /// instance. The copy has the same name, terminals, branch rows and state
    /// layout; `context` is the point at which it will be evaluated and is
    /// used for validation only. Immutable: this device does not change.
    ///
    /// # Errors
    /// A parameter this device does not sweep, or an invalid value.
    fn with_instance_parameter(
        &self,
        parameter: &str,
        _value: Real,
        _context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<Box<dyn Device>> {
        Err(SpiceError::circuit(format!(
            "{}: instance parameter {parameter} cannot be swept",
            self.name()
        )))
    }

    /// The device as `.sens` perturbs it ([`crate::devices::sensitivity`],
    /// C `cktsens.c`/`cktsgen.c`): C's parameter tables and records and the
    /// setter/temperature/load routines replayed on them, evaluated under
    /// `context`'s temperatures. The default is an explicit error, so a device
    /// whose C parameters the port cannot reproduce is refused rather than
    /// silently missing from the output.
    ///
    /// # Errors
    /// [`SpiceError::NotYetPorted`] by default; device data the records cannot
    /// represent.
    fn sensitivity(
        &self,
        _context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<Box<dyn crate::devices::sensitivity::DeviceSensitivity + '_>> {
        Err(crate::devices::sensitivity::not_ported(
            self.name(),
            self.designator(),
            "",
        ))
    }
}

/// Converged trial data for a device-local, rollback-safe transient event.
#[derive(Debug)]
pub struct TransientEventContext<'a> {
    /// Device-local state at the preceding accepted point.
    pub accepted: &'a [Real],
    /// Absolute candidate time.
    pub time: Real,
    /// The candidate solution.
    pub solution: &'a Vector,
    /// The circuit's ground-eliminated node numbering.
    pub unknowns: &'a MnaUnknowns,
}

impl TransientEventContext<'_> {
    /// Candidate node potential, zero at ground.
    #[must_use = "the candidate node potential is required to evaluate the event law"]
    pub fn voltage(&self, node: NodeId) -> Real {
        self.unknowns
            .node_row(node)
            .and_then(|row| self.solution.get(row))
            .unwrap_or(0.)
    }
}

/// A device whose finite history directly determines future exact boundaries.
/// Accepted state is read-only; probes cannot commit, queue or cancel events.
pub trait DeviceTransientEvents {
    /// The next boundary after the supplied accepted state, if any.
    ///
    /// # Errors
    /// Invalid state or unrepresentable event time.
    fn next_boundary(&self, accepted: &[Real]) -> SpiceResult<Option<Real>>;

    /// Refine receiver state from the converged solution and optionally
    /// request a retry at an interior crossing. Only candidate slots may
    /// change; the Newton stamp must already satisfy its convergence test.
    ///
    /// # Errors
    /// Invalid candidate physics, time or state.
    fn refine_trial(
        &self,
        context: &TransientEventContext<'_>,
        trial: &mut [Real],
    ) -> SpiceResult<Option<Real>>;
}

/// What [`Device::timestep_limit`] sees for one converged trial step.
#[derive(Debug, Clone, Copy)]
pub struct TruncationContext<'a> {
    /// This device's slots of the converged trial (C `CKTstate0`).
    pub trial: &'a [Real],
    /// This device's slots of the latest accepted point (C `CKTstate1`).
    pub accepted: Option<&'a [Real]>,
    /// The step just taken (C `CKTdeltaOld[0]`).
    pub dt: Real,
}

/// What [`Device::accept`] sees for one accepted point.
#[derive(Debug, Clone, Copy)]
pub struct AcceptContext<'a> {
    /// The accepted solution.
    pub solution: &'a Vector,
    /// The accepted time, `None` for DC points.
    pub time: Option<Real>,
    /// This device's branch-current rows.
    pub branches: &'a Range<usize>,
    /// This device's slots of the state about to be committed, when the
    /// analysis tracks state.
    pub states: Option<&'a [Real]>,
}

#[cfg(test)]
mod tests {
    use super::{AnalysisMode, MnaUnknowns};
    use crate::primitives::{NodeId, NodeKind, NodeTable};

    #[test]
    fn ground_gets_no_row() {
        let mut nodes = NodeTable::new();
        let a = nodes.intern("a").unwrap();
        let b = nodes.intern("b").unwrap();
        let mut unknowns = MnaUnknowns::new();
        unknowns.rebuild(&nodes);
        assert_eq!(unknowns.node_row(NodeId::GROUND), None);
        assert_eq!(unknowns.node_row(a), Some(0));
        assert_eq!(unknowns.node_row(b), Some(1));
        assert_eq!(unknowns.len(), 2);
    }

    #[test]
    fn internal_nodes_are_unknowns_too() {
        let mut nodes = NodeTable::new();
        let terminal = nodes.intern("out").unwrap();
        let internal = nodes.intern("x1#1").unwrap();
        nodes.set_kind(internal, NodeKind::Internal);
        let mut unknowns = MnaUnknowns::new();
        unknowns.rebuild(&nodes);
        assert_eq!(unknowns.node_row(terminal), Some(0));
        assert_eq!(unknowns.node_row(internal), Some(1));
    }

    #[test]
    fn rows_can_be_assigned_incrementally() {
        let mut nodes = NodeTable::new();
        nodes.intern("a").unwrap();
        let mut unknowns = MnaUnknowns::new();
        unknowns.rebuild(&nodes);
        assert_eq!(unknowns.len(), 1);
        let late = nodes.intern("late").unwrap();
        let row = unknowns.assign(nodes.node(late).expect("node exists"));
        assert_eq!(row, Some(1));
        assert_eq!(unknowns.len(), 2);
        // Idempotent.
        assert_eq!(
            unknowns.assign(nodes.node(late).expect("node exists")),
            Some(1)
        );
    }

    #[test]
    fn rows_can_be_reserved_for_branch_currents() {
        let mut unknowns = MnaUnknowns::new();
        assert_eq!(unknowns.add_rows(0).unwrap(), None);
        assert_eq!(unknowns.add_rows(2).unwrap(), Some(0));
        assert_eq!(unknowns.add_rows(1).unwrap(), Some(2));
        assert_eq!(unknowns.len(), 3);
    }

    #[test]
    fn unsupported_modes_are_classified() {
        assert!(AnalysisMode::OperatingPoint.is_dc());
        assert!(AnalysisMode::DcSweep.is_dc());
        assert!(AnalysisMode::Ac { frequency: 1e3 }.is_ac());
        assert!(!AnalysisMode::Ac { frequency: 1e3 }.is_dc());
        assert!(
            AnalysisMode::Transient {
                time: 0.0,
                dt: 1e-6
            }
            .is_transient()
        );
    }
}
