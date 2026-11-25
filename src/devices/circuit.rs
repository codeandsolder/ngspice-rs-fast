//! The circuit: nodes, unknowns and device instances.
//!
//! The C equivalent is `CKTcircuit` in `src/include/ngspice/cktdefs.h`, built by
//! `CKTcrtElt`/`CKTbindNode` and friends in `src/spicelib/devices/ckt*.c`. The
//! port keeps the same two-step shape:
//!
//! 1. nodes and devices are *added*, in deck order;
//! 2. [`Circuit::finalize`] numbers the MNA unknowns, which must happen
//!    after every node and device is known, because device-made unknowns
//!    (internal nodes, branch currents) get rows after the parsed nodes, in
//!    C's setup order.
//!
//! Step 2 is easy to forget, so the analyses call
//! [`Circuit::finalize`], which validates the graph and rebuilds the numbering
//! in one go.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use crate::maths::{Coefficients, SparseMatrix, Vector};
use crate::primitives::{NodeId, NodeKind, NodeTable, Real, SpiceError, SpiceResult};
use petgraph::graph::UnGraph;

use crate::devices::elaboration::{
    ElaborationLimits, ElaborationSize, ExpansionKind, PlannedBatch, reserve, resource_error,
};
use crate::devices::linear::Forcing;
use crate::devices::ltra::{LtraWork, LtraWorkspace};
use crate::devices::models::ModelContext;
use crate::devices::mutual::validation::ValidationBudget;
use crate::devices::mutual::{CouplingSite, MutualLimits, pair_count};
use crate::devices::state::{
    DeviceState, HistoryPublication, StateHistory, StateSnapshot, TrialState,
};
use crate::devices::sweep::{
    InstanceOverride, InstanceTarget, ParameterKey, ResistorMetadata, ResistorOverride,
};
use crate::devices::traits::{
    AcceptContext, AnalysisMode, Device, InductanceValue, MnaUnknowns, MutualTerm, StampContext,
    StampRhs,
};

/// A vertex in the circuit's bipartite incidence graph.
///
/// Node IDs and device ordinals are separate namespaces. Neither is an MNA row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitVertex {
    /// A node in the circuit's node table, including ground.
    Node(NodeId),
    /// A device's position in [`Circuit::devices`], in insertion order.
    Device(usize),
}

/// Petgraph circuit incidence graph: node/device vertices and one edge per port.
///
/// Each edge's weight is the zero-based terminal ordinal. Parallel edges retain
/// repeated terminals (e.g. a device whose two ports both connect to ground).
/// Graph indices are snapshot-local handles, not SPICE node IDs or MNA rows.
/// Connectivity is structural, not proof of a conductive DC path or solvability.
pub type CircuitGraph = UnGraph<CircuitVertex, usize, usize>;

// Test-only counters at the actual shared preparation operations, not a
// synthetic estimate from point counts. Thread-local for parallel unit tests.
#[cfg(test)]
pub(crate) mod preparation_metrics {
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct Counts {
        pub(crate) name_lookups: usize,
        pub(crate) bindings: usize,
        pub(crate) contexts: usize,
        pub(crate) setters: usize,
        pub(crate) replacement_validations: usize,
        pub(crate) physical_validations: usize,
        pub(crate) small_signal_assemblies: usize,
    }

    thread_local! {
        static COUNTS: std::cell::Cell<Counts> = const { std::cell::Cell::new(Counts {
            name_lookups: 0, bindings: 0, contexts: 0, setters: 0,
            replacement_validations: 0, physical_validations: 0, small_signal_assemblies: 0,
        }) };
    }

    pub(super) fn record(update: impl FnOnce(&mut Counts)) {
        COUNTS.with(|cell| {
            let mut counts = cell.get();
            update(&mut counts);
            cell.set(counts);
        });
    }

    pub(crate) fn take() -> Counts {
        COUNTS.with(|cell| cell.replace(Counts::default()))
    }
}

/// Name-to-ordinal index built during setup and retained only for the
/// lifetime of the finalized topology.
#[derive(Default)]
struct DeviceNames(BTreeMap<String, usize>);

impl DeviceNames {
    fn insert(&mut self, name: &str, index: usize) {
        self.0.entry(name.to_ascii_lowercase()).or_insert(index);
    }

    fn get(&self, name: &str) -> Option<usize> {
        let name = if name.bytes().any(|b| b.is_ascii_uppercase()) {
            Cow::Owned(name.to_ascii_lowercase())
        } else {
            Cow::Borrowed(name)
        };
        self.0.get(name.as_ref()).copied()
    }
}

/// Elaborated provenance of a device, kept as one record rather than three
/// independently extended lists. Its ordinal indexes the device slice.
#[derive(Debug, Default, PartialEq, Eq)]
struct DeviceOrigin {
    /// The parsed model name (lowercase), or C's default model.
    model: Option<String>,
    /// Nodes this device creates in its `DEVsetup`, in creation order.
    internal_nodes: Vec<NodeId>,
    /// The URC setup which generated this device and its model.
    generated_by: Option<usize>,
}

/// One elaborated device together with its setup provenance.
#[derive(Debug)]
pub struct ElaboratedDevice {
    device: Box<dyn Device>,
    origin: DeviceOrigin,
}

impl std::ops::Deref for ElaboratedDevice {
    type Target = dyn Device;
    fn deref(&self) -> &Self::Target {
        self.device.as_ref()
    }
}

impl AsRef<dyn Device> for ElaboratedDevice {
    fn as_ref(&self) -> &(dyn Device + 'static) {
        self.device.as_ref()
    }
}

/// Elaborated devices and finalized bindings have separate lifecycles.
///
/// Bindings are inaccessible until finalization publishes a layout.
#[derive(Default)]
pub struct Circuit {
    nodes: NodeTable,
    devices: Vec<ElaboratedDevice>,
    bindings: TopologyBindings,
    mutual_limits: MutualLimits,
    elaboration_limits: ElaborationLimits,
    pub(super) runtime: CircuitRuntime,
}

/// Setup-only numbering and topology.
///
/// The identity is published only after every validation and allocation succeeds.
#[derive(Default)]
struct TopologyBindings {
    unknowns: MnaUnknowns,
    names: DeviceNames,
    order: Vec<usize>,
    branch_rows: Vec<std::ops::Range<usize>>,
    convergence_kinds: Vec<crate::devices::ConvergenceKind>,
    control_rows: Vec<Vec<usize>>,
    state_rows: Arc<[std::ops::Range<usize>]>,
    state_len: usize,
    stochastic_sources: Vec<usize>,
    prepared_devices: Vec<usize>,
    mutual: MutualBinding,
    identity: Option<TopologyId>,
}

/// Independently validated numeric candidate sites, in netlist setter order.
/// A worker owns these values without changing the `Copy` `ModelContext` or the
/// original immutable device recipes.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CandidateNumericOverride {
    Resistor(ResistorOverride),
    Instance(InstanceOverride),
}

impl CandidateNumericOverride {
    const fn device(self) -> usize {
        match self {
            Self::Resistor(target) => target.device(),
            Self::Instance(target) => target.device(),
        }
    }
}

/// Analysis state and caches, never a source of topology truth.
#[derive(Default)]
pub(super) struct CircuitRuntime {
    pub(super) observations: crate::devices::observe::ObservationPlan,
    initial_revision: std::cell::RefCell<Arc<()>>,
    pattern: std::cell::RefCell<Option<Arc<crate::maths::Pattern>>>,
    carried_state: std::cell::RefCell<Option<StateSnapshot>>,
    completed: Vec<crate::primitives::AnalysisKind>,
    final_transient: std::cell::RefCell<Option<StateSnapshot>>,
    unknown_names: std::cell::RefCell<Option<crate::maths::UnknownNames>>,
    fixed_stamps: std::cell::RefCell<crate::maths::FixedStamps>,
    mutual_evaluation: std::cell::RefCell<Option<MutualEvaluation>>,
    candidate_overrides: Vec<CandidateNumericOverride>,
    control_overrides: Vec<CandidateNumericOverride>,
}

/// A process-local, never recycled identity for a validated topology.
///
/// It fits in `Copy` analysis requests; exhaustion is an explicit error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TopologyId(u64);

impl TopologyId {
    fn allocate() -> SpiceResult<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let mut id = NEXT.load(Ordering::Relaxed);
        loop {
            let next = id
                .checked_add(1)
                .ok_or_else(|| SpiceError::circuit("topology identity space exhausted"))?;
            match NEXT.compare_exchange_weak(id, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Ok(Self(id)),
                Err(observed) => id = observed,
            }
        }
    }
}

/// One inductor pair coupled by a K device, by device ordinal.
#[derive(Debug, Clone, Copy, PartialEq)]
struct MutualPair {
    /// The K device.
    coupling: usize,
    first: usize,
    second: usize,
}

/// The resolved K couplings of a circuit.
#[derive(Debug, Clone, Default)]
struct MutualBinding {
    pairs: Vec<MutualPair>,
    groups: Vec<CoupledGroup>,
    degrees: BTreeMap<usize, usize>,
    setup_rows: BTreeMap<usize, Vec<(usize, usize)>>,
}

/// The actual physical input values used for a verified coupling load.
/// Cache by bit pattern, not by analysis/temperature guesses, so any
/// contextual device or setter can invalidate it without special rules.
#[derive(Debug, Clone)]
struct MutualEvaluation {
    values: BTreeMap<usize, InductanceValue>,
    mutuals: Vec<Real>,
    terms: PreparedMutualTerms,
}

impl MutualEvaluation {
    fn matches(&self, values: &BTreeMap<usize, InductanceValue>, mutuals: &[Real]) -> bool {
        const fn bits(value: &InductanceValue) -> [u64; 3] {
            [
                value.effective.to_bits(),
                value.coupling_base.to_bits(),
                value.initial.to_bits(),
            ]
        }
        self.values.len() == values.len()
            && self
                .values
                .iter()
                .zip(values)
                .all(|((a, x), (b, y))| a == b && bits(x) == bits(y))
            && self.mutuals.len() == mutuals.len()
            && self
                .mutuals
                .iter()
                .zip(mutuals)
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }
}

/// No allocation for the ordinary uncoupled circuit. A checked coupled
/// circuit shares its complete stamp input across successive loads.
#[derive(Debug, Clone, Default)]
pub(super) struct PreparedMutualTerms(Option<Arc<[Vec<MutualTerm>]>>);

impl std::ops::Deref for PreparedMutualTerms {
    type Target = [Vec<MutualTerm>];

    fn deref(&self) -> &Self::Target {
        self.0.as_deref().unwrap_or(&[])
    }
}

/// Analysis-owned effective device preparation.
///
/// The circuit is borrowed immutably for the run, so no topology edit can invalidate a live view.
/// Only the most recent exact model context is retained. Observation scratch
/// retains capacity, not accepted or previous trial state. Transient attempts
/// borrow their accepted history and reuse this run's LTRA scratch allocation;
/// the prepared inputs themselves remain local to a single attempt.
pub struct PreparedRun<'a> {
    circuit: &'a Circuit,
    point: std::cell::RefCell<Option<PreparedPoint>>,
    ltra: std::cell::RefCell<LtraWorkspace>,
    pub(super) observation_work:
        std::cell::RefCell<Option<crate::devices::observe::ObservationWorkspace>>,
}

/// One transient proposal, binding exact time, physical context and immutable
/// accepted history for repeated Newton loads and isolated observations.
/// Exclusive scratch ownership prevents another proposal from reusing it.
pub(crate) struct PreparedTransient<'a> {
    pub(super) run: &'a PreparedRun<'a>,
    history: &'a StateHistory,
    context: &'a ModelContext,
    coefficients: &'a Coefficients,
    time: Real,
    forcing: Forcing,
    pub(super) workspace: std::cell::RefMut<'a, LtraWorkspace>,
}

impl Drop for PreparedTransient<'_> {
    fn drop(&mut self) {
        self.workspace.finish();
    }
}

impl<'a> PreparedTransient<'a> {
    pub(crate) const fn time(&self) -> Real {
        self.time
    }

    pub(crate) const fn run(&self) -> &'a PreparedRun<'a> {
        self.run
    }

    pub(crate) const fn coefficients(&self) -> &'a Coefficients {
        self.coefficients
    }

    pub(crate) const fn history(&self) -> &'a StateHistory {
        self.history
    }

    pub(crate) const fn context(&self) -> &'a ModelContext {
        self.context
    }

    /// A request can vary the Newton solution but never the accepted history
    /// or model/time parameters that determine a prepared delay input.
    pub(super) const fn request<'s>(&self, solution: &'s Vector) -> LoadRequest<'s>
    where
        'a: 's,
    {
        LoadRequest {
            mode: AnalysisMode::Transient {
                time: self.time,
                dt: self.coefficients.dt(),
            },
            solution,
            model_context: self.context,
            integration: Some(self.coefficients),
            history: self.history,
            forcing: Some(self.forcing),
        }
    }

    pub(crate) fn load(
        &mut self,
        solution: &Vector,
        matrix: &mut SparseMatrix,
        rhs: &mut Vector,
        trial: &mut TrialState,
    ) -> SpiceResult<()> {
        let request = self.request(solution);
        self.workspace.work.newton_loads += 1;
        self.run.circuit.load_prepared(
            self.run,
            &request,
            matrix,
            rhs,
            trial,
            Some(&mut self.workspace),
        )
    }
}

pub(super) struct PreparedPoint {
    context: ModelContext,
    replacements: Vec<(usize, Box<dyn Device>)>,
    mutual: std::cell::OnceCell<PreparedMutualTerms>,
    prepared_batches:
        std::cell::OnceCell<std::cell::RefCell<crate::devices::prepared_model::batch::Plan>>,
}

fn effective_device<'a>(
    circuit: &'a Circuit,
    replacements: &'a [(usize, Box<dyn Device>)],
    ordinal: usize,
) -> &'a dyn Device {
    replacements
        .binary_search_by_key(&ordinal, |(index, _)| *index)
        .map_or_else(
            |_| circuit.devices[ordinal].as_ref(),
            |index| replacements[index].1.as_ref(),
        )
}

impl PreparedPoint {
    pub(super) fn device<'a>(&'a self, circuit: &'a Circuit, ordinal: usize) -> &'a dyn Device {
        effective_device(circuit, &self.replacements, ordinal)
    }

    fn batches(
        &self,
        circuit: &Circuit,
    ) -> SpiceResult<std::cell::RefMut<'_, crate::devices::prepared_model::batch::Plan>> {
        if self.prepared_batches.get().is_none() {
            let plan = crate::devices::prepared_model::batch::Plan::new(
                circuit.devices.len(),
                &circuit.bindings.unknowns,
                circuit
                    .bindings
                    .prepared_devices
                    .iter()
                    .copied()
                    .map(|index| (index, self.device(circuit, index))),
            )?;
            let _ = self.prepared_batches.set(std::cell::RefCell::new(plan));
        }
        self.prepared_batches
            .get()
            .ok_or_else(|| SpiceError::circuit("missing prepared batch plan"))?
            .try_borrow_mut()
            .map_err(|_| SpiceError::circuit("prepared batch plan already active"))
    }

    pub(super) fn mutual(&self, circuit: &Circuit) -> SpiceResult<PreparedMutualTerms> {
        if let Some(terms) = self.mutual.get() {
            return Ok(terms.clone());
        }
        let terms = circuit.mutual_terms_with_replacements(&self.context, &self.replacements)?;
        Ok(self.mutual.get_or_init(|| terms).clone())
    }
}

impl<'a> PreparedRun<'a> {
    /// The finalized circuit borrowed by this analysis.
    #[must_use = "the borrowed circuit provides the topology associated with this prepared run"]
    pub const fn circuit(&self) -> &'a Circuit {
        self.circuit
    }

    pub(super) fn point(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<std::cell::Ref<'_, PreparedPoint>> {
        if let Ok(point) = std::cell::Ref::filter_map(self.point.borrow(), |current| {
            current
                .as_ref()
                .filter(|point| point.context.same_preparation(context))
        }) {
            return Ok(point);
        }
        // Validate once on a context transition. Do not discard a valid
        // preparation on error (including a foreign override).
        #[cfg(test)]
        preparation_metrics::record(|counts| counts.contexts += 1);
        context.validate(&MODEL_CONTEXT_LOCATION)?;
        let mut replacements = self.circuit.replacements(context)?;
        for &index in &self.circuit.bindings.prepared_devices {
            match replacements.binary_search_by_key(&index, |(ordinal, _)| *ordinal) {
                Ok(slot) => {
                    if let Some(device) = replacements[slot].1.reprepare_context(context)? {
                        replacements[slot].1 = device;
                    }
                }
                Err(slot) => {
                    if let Some(device) = self.circuit.devices[index].reprepare_context(context)? {
                        replacements.insert(slot, (index, device));
                    }
                }
            }
        }
        let prepared_batches = std::cell::OnceCell::new();
        {
            let previous = self.point.borrow();
            if let Some(plan) = previous
                .as_ref()
                .and_then(|point| point.prepared_batches.get())
            {
                let mut plan = plan
                    .try_borrow_mut()
                    .map_err(|_| SpiceError::circuit("prepared batch plan already active"))?;
                plan.rebind(
                    |index| effective_device(self.circuit, &replacements, index),
                    &self.circuit.bindings.unknowns,
                )?;
                let _ = prepared_batches.set(std::cell::RefCell::new(std::mem::take(&mut *plan)));
            }
        }
        self.point.replace(Some(PreparedPoint {
            context: *context,
            replacements,
            mutual: std::cell::OnceCell::new(),
            prepared_batches,
        }));
        std::cell::Ref::filter_map(self.point.borrow(), Option::as_ref)
            .map_err(|_| SpiceError::circuit("missing prepared model context"))
    }

    /// Begin a proposal without eagerly evaluating devices: preserve load-order
    /// errors and avoid allocating convolution weights for non-LTRA devices.
    pub(crate) fn transient_attempt<'s>(
        &'s self,
        history: &'s StateHistory,
        context: &'s ModelContext,
        time: Real,
        coefficients: &'s Coefficients,
        forcing: Forcing,
    ) -> SpiceResult<PreparedTransient<'s>> {
        let mut workspace = self
            .ltra
            .try_borrow_mut()
            .map_err(|_| SpiceError::circuit("a transient proposal is already active"))?;
        workspace.begin();
        Ok(PreparedTransient {
            run: self,
            history,
            context,
            coefficients,
            time,
            forcing,
            workspace,
        })
    }

    pub(crate) fn ltra_work(&self) -> LtraWork {
        self.ltra.borrow().work
    }

    /// Validate the ordered overrides at a physical point without stamping.
    /// # Errors
    /// Invalid model context, foreign targets, conflicting setters, or device physics.
    pub fn validate_context(&self, context: &ModelContext) -> SpiceResult<()> {
        self.point(context).map(|_| ())
    }

    /// Validate effective device inputs for DC, independently of any bias.
    /// The prepared replacement view and mutual-term validation are exactly
    /// those used by the real load; AC-only laws are not evaluated.
    /// # Errors
    /// Invalid context, replacement, contextual device input or mutual coupling.
    pub fn validate_dc_inputs(&self, context: &ModelContext) -> SpiceResult<()> {
        #[cfg(test)]
        preparation_metrics::record(|counts| counts.physical_validations += 1);
        let point = self.point(context)?;
        point.mutual(self.circuit)?;
        let inputs = Self::dc_device_context(context);
        for index in self.circuit.load_order() {
            point
                .device(self.circuit, index)
                .validate_dc_inputs(&inputs)?;
        }
        Ok(())
    }

    /// Validate replaced devices after confirming the unchanged recipes at
    /// the same scalar context. General setters can change mutual coupling;
    /// independent resistance/source replacements cannot.
    pub(crate) fn validate_dc_replacements(
        &self,
        context: &ModelContext,
        coupling_may_change: bool,
    ) -> SpiceResult<()> {
        #[cfg(test)]
        preparation_metrics::record(|counts| counts.replacement_validations += 1);
        let point = self.point(context)?;
        if coupling_may_change {
            point.mutual(self.circuit)?;
        }
        let inputs = Self::dc_device_context(context);
        for (_, device) in &point.replacements {
            device.validate_dc_inputs(&inputs)?;
        }
        Ok(())
    }

    // Each effective recipe already incorporates its own setters. A device's
    // pure input check deliberately sees shared scalar inputs, not the other
    // devices' override tables, just as its real DC load does.
    const fn dc_device_context(context: &ModelContext) -> ModelContext {
        ModelContext {
            resistor_overrides: [None; crate::devices::MAX_RESISTOR_OVERRIDES],
            instance_overrides: [None; crate::devices::MAX_INSTANCE_OVERRIDES],
            ..*context
        }
    }

    /// Load a disposable trial using this run's prepared effective devices.
    /// # Errors
    /// As [`Circuit::load`]; a changed context is revalidated before use.
    pub fn load(
        &self,
        request: &LoadRequest<'_>,
        matrix: &mut SparseMatrix,
        rhs: &mut Vector,
        trial: &mut TrialState,
    ) -> SpiceResult<()> {
        self.circuit
            .load_prepared(self, request, matrix, rhs, trial, None)
    }

    /// Check convergence using the same effective devices as the load.
    /// # Errors
    /// As [`Circuit::convergence_test`].
    pub fn convergence_test(&self, request: &ConvergenceRequest<'_>) -> SpiceResult<bool> {
        self.circuit.convergence_prepared(self, request)
    }

    /// Record persistent MODEINITSMSIG slots using the effective devices.
    /// # Errors
    /// As [`Circuit::record_small_signal_state`].
    pub fn record_small_signal_state(
        &self,
        context: &ModelContext,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.circuit
            .record_small_signal_state_prepared(self, context, bias)
    }

    /// Build independent physical small-signal state at a sampled orbit
    /// point, without publishing any carried analysis state.
    pub(crate) fn independent_small_signal_snapshot(
        &self,
        context: &ModelContext,
        bias: &Vector,
        values: &[Real],
    ) -> SpiceResult<StateSnapshot> {
        let mut state = values.to_vec();
        self.circuit
            .small_signal_slots(self, context, bias, &mut state)?;
        self.circuit.bind_state(state)
    }

    /// Assemble the small-signal system with this run's effective devices.
    /// # Errors
    /// As [`Circuit::small_signal_system_at`].
    pub fn small_signal_system_at(
        &self,
        context: &ModelContext,
        bias: &Vector,
        state: Option<&StateSnapshot>,
    ) -> SpiceResult<crate::devices::linear::LinearSystem> {
        self.circuit
            .bias_linearized_system(self, context, bias, state, 0, None)
    }

    /// Transient source timing and fixed linear equations, without
    /// evaluating nonlinear devices at an artificial zero bias. The actual
    /// nonlinear static and transient equations are loaded by Newton.
    pub(crate) fn transient_forcing_system(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<crate::devices::linear::LinearSystem> {
        self.circuit.assemble_linear_devices(self, context, true)
    }

    /// The constant linear operators and source bindings of a physical DAE.
    /// Nonlinear devices are handled by the ordinary exact load and their
    /// separate physical charge laws. Reject history-dependent equations
    /// before any frequency-domain Newton iteration.
    pub(crate) fn dae_linear_part(
        &self,
        context: &ModelContext,
    ) -> SpiceResult<crate::devices::linear::LinearSystem> {
        let point = self.point(context)?;
        for index in self.circuit.load_order() {
            let device = point.device(self.circuit, index);
            if device.depends_on_time()
                || device.depends_on_frequency()
                || device.small_signal_depends_on_frequency()
                || device.delay_line().is_some()
                || device.has_discrete_state()
            {
                return Err(SpiceError::Unsupported {
                    feature: format!("frequency-domain physical DAE for {}", device.name()),
                    location: None,
                });
            }
        }
        drop(point);
        self.circuit.assemble_linear_devices(self, context, true)
    }

    /// Add exact nonlinear device charge and its derivative to a linear
    /// charge assembly. The same effective recipes are used as in `load()`.
    pub(crate) fn dae_nonlinear_charge(
        &self,
        context: &ModelContext,
        state: &Vector,
        charge: &mut Vector,
        jacobian: &mut SparseMatrix,
    ) -> SpiceResult<()> {
        self.assemble_charge_for(context, state, charge, jacobian, self.circuit.load_order())
    }

    /// Assemble only selected physical charge equations for a constrained
    /// instantaneous load. Uses the same prepared model and physical kernel
    /// as the frequency-domain DAE; does not evaluate unrelated devices.
    pub(crate) fn assemble_charge_for(
        &self,
        context: &ModelContext,
        state: &Vector,
        charge: &mut Vector,
        jacobian: &mut SparseMatrix,
        indices: impl Iterator<Item = usize>,
    ) -> SpiceResult<()> {
        let point = self.point(context)?;
        let mut input = crate::devices::traits::ChargeContext {
            model: context,
            unknowns: &self.circuit.bindings.unknowns,
            branch: None,
            controls: &[],
            state,
            charge,
            jacobian,
        };
        for index in indices {
            let device = point.device(self.circuit, index);
            if device.is_nonlinear() {
                input.branch = self.circuit.bindings.branch_rows[index].clone().next();
                input.controls = self.circuit.controls(index)?;
                device.assemble_nonlinear_charge(&mut input)?;
            }
        }
        Ok(())
    }
}

static MODEL_CONTEXT_LOCATION: std::sync::LazyLock<crate::primitives::SourceLoc> =
    std::sync::LazyLock::new(|| {
        crate::primitives::SourceLoc::new(std::path::PathBuf::from("<model-context>"), 1, 1)
    });

/// A K card resolved without expanding its complete graph.
struct ResolvedCard {
    coupling: usize,
    inductors: Vec<usize>,
    pairs: usize,
}

/// A fixed connected component. Local rows are increasing device ordinals.
/// Edges are grouped by unordered matrix cell and retain the original pair
/// order within a cell, including repeated and reversed K cards.
#[derive(Debug, Clone)]
struct CoupledGroup {
    inductors: Vec<usize>,
    edges: Vec<GroupEdge>,
    first_coupling: usize,
}

#[derive(Debug, Clone, Copy)]
struct GroupEdge {
    pair: usize,
    first_row: usize,
    second_row: usize,
}

struct GroupPlan {
    first_coupling: usize,
    nodes: usize,
    pairs: usize,
}

/// Count the mutual-specific payload before any pair expansion. The plan
/// charges both orientations of stamp terms, the current and cached evaluation,
/// matrix position declarations, per-reference indices and group work arrays.
/// Actual allocator/tree overhead and unrelated circuit storage are separate.
fn mutual_storage_bytes(
    pairs: usize,
    references: usize,
    cards: usize,
    devices: usize,
) -> Option<usize> {
    let pair_bytes = size_of::<MutualPair>()
        + size_of::<GroupEdge>()
        + 3 * size_of::<(usize, usize)>()
        + 2 * size_of::<Real>()
        + 4 * size_of::<MutualTerm>();
    let reference_bytes =
        12 * size_of::<usize>() + 2 * size_of::<InductanceValue>() + 2 * size_of::<Real>();
    let card_bytes = size_of::<ResolvedCard>()
        + size_of::<GroupPlan>()
        + size_of::<CoupledGroup>()
        + size_of::<Vec<(usize, usize)>>();
    pairs
        .checked_mul(pair_bytes)?
        .checked_add(references.checked_mul(reference_bytes)?)?
        .checked_add(cards.checked_mul(card_bytes)?)?
        .checked_add(devices.checked_mul(3 * size_of::<Vec<MutualTerm>>())?)
}

/// Linear-reference connectivity and group sizes, checked before pair expansion.
struct CouplingTopology {
    degrees: BTreeMap<usize, usize>,
    positions: BTreeMap<usize, usize>,
    labels: Vec<usize>,
    plans: BTreeMap<usize, GroupPlan>,
}

impl CouplingTopology {
    fn plan(
        cards: &[ResolvedCard],
        devices: &[ElaboratedDevice],
        limits: MutualLimits,
        first_site: CouplingSite<'_>,
    ) -> SpiceResult<Self> {
        let site = |coupling: usize| CouplingSite::of(devices[coupling].as_ref());
        // The connectivity of a K clique is exactly the connectivity of
        // its star. Do this before any quadratic expansion.
        let mut degrees = BTreeMap::<usize, usize>::new();
        for card in cards {
            for &inductor in &card.inductors {
                *degrees.entry(inductor).or_default() += card.inductors.len() - 1;
            }
        }
        let positions: BTreeMap<_, _> = degrees
            .keys()
            .copied()
            .enumerate()
            .map(|(position, device)| (device, position))
            .collect();
        let mut systems = petgraph::unionfind::UnionFind::<usize>::new(positions.len());
        for card in cards {
            let first = positions[&card.inductors[0]];
            for second in &card.inductors[1..] {
                systems.union(first, positions[second]);
            }
        }
        let labels = systems.into_labeling();
        let mut plans = BTreeMap::<usize, GroupPlan>::new();
        for card in cards {
            let label = labels[positions[&card.inductors[0]]];
            let plan = plans.entry(label).or_insert(GroupPlan {
                first_coupling: card.coupling,
                nodes: 0,
                pairs: 0,
            });
            // Already covered by the checked aggregate pair count.
            plan.pairs += card.pairs;
        }
        for &label in &labels {
            plans
                .get_mut(&label)
                .ok_or_else(|| first_site.resource("missing coupled group"))?
                .nodes += 1;
        }

        // Charge the minimum validation work, even if every group will be
        // certified by a sparse bound. Dense fallback is charged separately
        // before its matrix allocation, once actual coefficients are known.
        let mut minimum_work = ValidationBudget::new(limits);
        for plan in plans.values() {
            let group_site = site(plan.first_coupling);
            if plan.nodes > limits.group_inductors || u32::try_from(plan.nodes).is_err() {
                return Err(group_site.resource(format_args!(
                    "mutual connected-group size limit exceeded ({} inductors, limit {})",
                    plan.nodes, limits.group_inductors
                )));
            }
            minimum_work.charge_sparse(plan.nodes, plan.pairs, group_site)?;
        }

        Ok(Self {
            degrees,
            positions,
            labels,
            plans,
        })
    }
}

impl MutualBinding {
    fn from_cards(
        cards: Vec<ResolvedCard>,
        total_pairs: usize,
        branch_rows: &[std::ops::Range<usize>],
        devices: &[ElaboratedDevice],
        limits: MutualLimits,
    ) -> SpiceResult<Self> {
        let Some(first_card) = cards.first() else {
            return Ok(Self::default());
        };
        let site = |coupling: usize| CouplingSite::of(devices[coupling].as_ref());
        let first_site = site(first_card.coupling);

        let CouplingTopology {
            degrees,
            positions,
            labels,
            plans,
        } = CouplingTopology::plan(&cards, devices, limits, first_site)?;

        let mut groups = BTreeMap::new();
        for (label, plan) in plans {
            let group_site = site(plan.first_coupling);
            groups.insert(
                label,
                CoupledGroup {
                    inductors: group_site.reserve(plan.nodes)?,
                    edges: group_site.reserve(plan.pairs)?,
                    first_coupling: plan.first_coupling,
                },
            );
        }
        let mut local = first_site.reserve(positions.len())?;
        for (&device, &label) in positions.keys().zip(&labels) {
            let group = groups
                .get_mut(&label)
                .ok_or_else(|| first_site.resource("missing coupled group"))?;
            local.push(group.inductors.len());
            group.inductors.push(device);
        }

        let mut pairs = first_site.reserve(total_pairs)?;
        let mut setup_rows = BTreeMap::new();
        for card in cards {
            let group_site = site(card.coupling);
            let label = labels[positions[&card.inductors[0]]];
            let group = groups
                .get_mut(&label)
                .ok_or_else(|| group_site.resource("missing coupled group"))?;
            let mut rows = group_site.reserve(card.pairs)?;
            for (position, &first) in card.inductors.iter().enumerate() {
                for &second in &card.inductors[position + 1..] {
                    let a = local[positions[&first]];
                    let b = local[positions[&second]];
                    group.edges.push(GroupEdge {
                        pair: pairs.len(),
                        first_row: a.min(b),
                        second_row: a.max(b),
                    });
                    pairs.push(MutualPair {
                        coupling: card.coupling,
                        first,
                        second,
                    });
                    rows.push((branch_rows[first].start, branch_rows[second].start));
                }
            }
            // C's instance list visits each K card's original pairs in reverse.
            rows.reverse();
            setup_rows.insert(card.coupling, rows);
        }
        for group in groups.values_mut() {
            group
                .edges
                .sort_unstable_by_key(|edge| (edge.first_row, edge.second_row, edge.pair));
        }
        Ok(Self {
            pairs,
            groups: groups.into_values().collect(),
            degrees,
            setup_rows,
        })
    }
}

/// One device convergence test pass ([`Circuit::convergence_test`]).
#[derive(Debug, Clone, Copy)]
pub struct ConvergenceRequest<'a> {
    /// The new iterate (C `CKTrhs`).
    pub solution: &'a Vector,
    /// The iterate the last load used (C `CKTrhsOld`).
    pub previous: &'a Vector,
    /// Accepted revision used by the last load.
    pub history: &'a StateHistory,
    /// The last load's trial (C `CKTstate0`).
    pub trial: &'a TrialState,
    /// Temperatures and `gmin` of the solve.
    pub model_context: &'a ModelContext,
    /// Transient time, zero otherwise.
    pub time: Real,
    /// Companion formula for the current transient step.
    pub integration: Option<&'a Coefficients>,
    /// C `CKTreltol`.
    pub reltol: Real,
    /// C `CKTabstol`.
    pub abstol: Real,
    /// C `CKTvoltTol`.
    pub vntol: Real,
}

/// The driver quantities [`Circuit::accept_transient_point`] and
/// [`Circuit::delay_timestep_limit`] pass to delay lines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DelayAcceptance {
    /// C `CKTdeltaOld[0..3]` ([`crate::devices::delay::DeltaOld`]).
    pub steps: crate::devices::delay::DeltaOld,
    /// C `CKTminBreak` as delay lines see it: the smallest sample spacing
    /// recorded.
    pub min_break: Real,
    /// The `t = 0` point that initializes the delay histories.
    pub start: bool,
    /// The run's relative tolerance (C `CKTreltol`).
    pub reltol: Real,
    /// The run's absolute current tolerance (C `CKTabstol`).
    pub abstol: Real,
}

/// One trial load of every device (C `CKTload`).
#[derive(Debug, Clone, Copy)]
pub struct LoadRequest<'a> {
    /// Which analysis is loading.
    pub mode: AnalysisMode,
    /// The present solution the trial linearizes around.
    pub solution: &'a Vector,
    /// Circuit and nominal temperatures.
    pub model_context: &'a ModelContext,
    /// Companion integration coefficients, for companion transient loads.
    pub integration: Option<&'a Coefficients>,
    /// The accepted state history this trial reads.
    pub history: &'a StateHistory,
    /// Source-forcing context, required by companion transient loads (sources
    /// evaluate their waveform at the mode's time) and `None` otherwise.
    pub forcing: Option<Forcing>,
}

impl fmt::Debug for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Circuit")
            .field("nodes", &self.nodes.len())
            .field("unknowns", &self.bindings.unknowns.len())
            .field(
                "devices",
                &self
                    .devices
                    .iter()
                    .map(|device| device.name())
                    .collect::<Vec<_>>(),
            )
            .finish_non_exhaustive()
    }
}

fn assemble_numeric_targets(
    context: &ModelContext,
    overrides: &[CandidateNumericOverride],
    count: usize,
) -> SpiceResult<Vec<CandidateNumericOverride>> {
    let mut targets = Vec::new();
    targets.try_reserve(count).map_err(|error| {
        SpiceError::circuit(format!("cannot allocate candidate device sites: {error}"))
    })?;
    targets.extend(
        context
            .resistor_overrides
            .iter()
            .flatten()
            .copied()
            .map(CandidateNumericOverride::Resistor),
    );
    targets.extend(
        context
            .instance_overrides
            .iter()
            .flatten()
            .copied()
            .map(CandidateNumericOverride::Instance),
    );
    targets.extend_from_slice(overrides);
    Ok(targets)
}

// Script edits supersede candidate sites, and a DC point supersedes both.
// Normalize resistance to its canonical instance setter, so a resistance
// edit and a different parameter of the same resistor can coexist. Retain
// setter order for independent sites and the last occurrence of each site.
fn ordered_control_targets(
    targets: impl IntoIterator<Item = CandidateNumericOverride>,
) -> SpiceResult<Vec<CandidateNumericOverride>> {
    let mut unique = BTreeMap::new();
    for (order, target) in targets.into_iter().enumerate() {
        let value = match target {
            CandidateNumericOverride::Resistor(value) => InstanceTarget::new(
                value.topology(),
                value.device(),
                ParameterKey::Native("resistance"),
            )
            .at(value.supplied())?,
            CandidateNumericOverride::Instance(value) => value,
        };
        unique.insert((value.device(), value.parameter()), (order, value));
    }
    let mut ordered: Vec<_> = unique.into_values().collect();
    ordered.sort_unstable_by_key(|(order, _)| *order);
    Ok(ordered
        .into_iter()
        .map(|(_, value)| CandidateNumericOverride::Instance(value))
        .collect())
}

impl Circuit {
    /// An empty circuit holding only ground.
    #[must_use = "the new circuit must be retained to elaborate and simulate devices"]
    pub fn new() -> Self {
        Self::default()
    }

    /// Start an analysis-owned preparation scope. It lazily validates the
    /// first context, then reuses its effective devices until that context
    /// changes. Separate scopes cannot share disposable trial state.
    #[must_use = "the prepared run retains the per-analysis point and load workspace"]
    pub const fn prepare_run(&self) -> PreparedRun<'_> {
        PreparedRun {
            circuit: self,
            point: std::cell::RefCell::new(None),
            ltra: std::cell::RefCell::new(LtraWorkspace::new()),
            observation_work: std::cell::RefCell::new(None),
        }
    }

    /// Construct an empty circuit with explicit mutual-inductor budgets.
    #[must_use = "discarding the new circuit loses the specified mutual coupling limits"]
    pub fn with_mutual_limits(limits: MutualLimits) -> Self {
        Self {
            mutual_limits: limits,
            ..Self::default()
        }
    }

    /// Construct an empty circuit with explicit aggregate elaboration limits.
    #[must_use = "discarding the new circuit loses the specified elaboration limits"]
    pub fn with_elaboration_limits(limits: ElaborationLimits) -> Self {
        Self {
            elaboration_limits: limits,
            ..Self::default()
        }
    }

    fn declared_state_slots(
        devices: &[ElaboratedDevice],
        at: Option<&crate::primitives::SourceLoc>,
    ) -> SpiceResult<usize> {
        devices.iter().try_fold(0_usize, |total, device| {
            total
                .checked_add(device.state_count())
                .ok_or_else(|| resource_error(at, "aggregate state slots overflow"))
        })
    }

    fn elaboration_size(&self) -> SpiceResult<ElaborationSize> {
        Ok(ElaborationSize::new(
            self.devices.len(),
            self.nodes.len() - 1,
            Self::declared_state_slots(&self.devices, None)?,
        ))
    }

    /// The node table.
    #[must_use = "the node table resolves circuit node identities"]
    pub const fn nodes(&self) -> &NodeTable {
        &self.nodes
    }

    /// The MNA unknown numbering. Only valid after [`Circuit::finalize`].
    #[must_use = "the unknown layout maps circuit coordinates to matrix rows"]
    pub const fn unknowns(&self) -> &MnaUnknowns {
        &self.bindings.unknowns
    }

    /// The device instances, in insertion order.
    #[must_use = "the elaborated devices define the circuit topology"]
    pub fn devices(&self) -> &[ElaboratedDevice] {
        &self.devices
    }

    /// Resolve the effective source timelines in a finalized circuit, without
    /// constructing a linear system or re-elaborating a deck. Expanded
    /// subcircuit names and the source setter/PORT ordering are retained.
    /// Each yielded result can fail when analysis-specific defaults are invalid.
    pub fn source_timelines(
        &self,
        timing: crate::devices::TransientTiming,
    ) -> impl Iterator<Item = SpiceResult<crate::devices::SourceTimeline<'_>>> + '_ {
        let error = self.check_numbering().err();
        let valid = error.is_none();
        error
            .into_iter()
            .map(Err)
            .chain(self.devices.iter().filter_map(move |device| {
                if !valid {
                    return None;
                }
                let (kind, waveform) = device.independent_waveform()?;
                Some(
                    waveform
                        .resolve(&timing)
                        .map(|waveform| crate::devices::SourceTimeline {
                            name: device.name(),
                            kind,
                            waveform,
                        }),
                )
            }))
    }

    /// Replace one elaborated device.
    ///
    /// Invalidate all bindings and analysis state before the edit;
    /// finalize again before analysis.
    ///
    /// Private nodes and generated device groups are owned by their original
    /// elaboration recipe. Replace such a group by elaborating a new deck.
    ///
    /// # Errors
    /// Unknown ordinal, or replacement of a generated/private-node device.
    pub fn replace_device(&mut self, index: usize, device: Box<dyn Device>) -> SpiceResult<()> {
        let origin = &self
            .devices
            .get(index)
            .ok_or_else(|| SpiceError::circuit("device ordinal out of bounds"))?
            .origin;
        if !origin.internal_nodes.is_empty()
            || origin.generated_by.is_some()
            || self
                .devices
                .iter()
                .any(|entry| entry.origin.generated_by == Some(index))
        {
            return Err(SpiceError::circuit(
                "replace generated or private-node devices by re-elaboration",
            ));
        }
        let mut size = self.elaboration_size()?;
        size.states = size
            .states
            .checked_sub(self.devices[index].state_count())
            .and_then(|n| n.checked_add(device.state_count()))
            .ok_or_else(|| resource_error(None, "replacement state slots overflow"))?;
        self.elaboration_limits.check(size, None)?;
        self.invalidate_topology();
        self.devices[index] = ElaboratedDevice {
            device,
            origin: DeviceOrigin::default(),
        };
        Ok(())
    }

    /// Reorder two devices and their provenance.
    ///
    /// Update generated-owner references; no positional binding survives the edit.
    ///
    /// # Errors
    /// An ordinal is outside the elaborated device list.
    pub fn swap_devices(&mut self, first: usize, second: usize) -> SpiceResult<()> {
        if first >= self.devices.len() || second >= self.devices.len() {
            return Err(SpiceError::circuit("device ordinal out of bounds"));
        }
        if first != second {
            self.invalidate_topology();
            self.devices.swap(first, second);
            for entry in &mut self.devices {
                let origin = &mut entry.origin;
                origin.generated_by = origin.generated_by.map(|index| {
                    if index == first {
                        second
                    } else if index == second {
                        first
                    } else {
                        index
                    }
                });
            }
        }
        Ok(())
    }

    fn invalidate_topology(&mut self) {
        if self.bindings.identity.is_some() {
            self.bindings = TopologyBindings::default();
            self.runtime = CircuitRuntime::default();
        }
    }

    /// Number of device instances.
    #[must_use = "the device count determines the size of per-instance workspaces"]
    pub const fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Size of the MNA system.
    #[must_use = "the unknown count determines the linear system dimensions"]
    pub const fn unknown_count(&self) -> usize {
        self.bindings.unknowns.len()
    }

    /// Adds a node, or returns the existing one.
    ///
    /// Matrix rows change as nodes are added, so call
    /// [`Circuit::finalize`] once the circuit is complete.
    /// # Errors
    /// Node identity or allocation capacity is exhausted.
    pub fn add_node(&mut self, name: &str) -> SpiceResult<NodeId> {
        let before = self.nodes.len();
        let limit = self.elaboration_limits.nodes;
        let node = self.nodes.transaction(|nodes| {
            let node = nodes.intern(name)?;
            if nodes.len() - 1 > limit {
                return Err(resource_error(
                    None,
                    format!("aggregate non-ground nodes exceed {limit}"),
                ));
            }
            Ok(node)
        })?;
        if self.nodes.len() != before {
            self.invalidate_topology();
        }
        Ok(node)
    }

    fn rebuild_unknowns_in_order(
        &mut self,
        order: &[usize],
        names: &DeviceNames,
    ) -> SpiceResult<()> {
        use crate::devices::setup::CDeviceType;

        self.runtime.unknown_names.get_mut().take();
        self.runtime.fixed_stamps.get_mut().forget();
        let mut owned = vec![false; self.nodes.len()];
        for node in self
            .devices
            .iter()
            .flat_map(|entry| &entry.origin.internal_nodes)
        {
            if let Some(slot) = owned.get_mut(node.index()) {
                *slot = true;
            }
        }
        self.bindings
            .unknowns
            .rebuild_deferring(&self.nodes, |node| owned[node.index()]);
        self.bindings.branch_rows = vec![0..0; self.devices.len()];
        // `CKTmkCur` runs once per device: in its own DEVsetup, or earlier
        // when a CCCS/CCVS/CSW/ASRC setup asks `CKTfndBranch` for it
        // (`VSRCfindBr`, `VCVSfindBr`, `CCVSfindBr` and `ASRCfindBr` create the
        // branch on demand), so a controlling source's branch is numbered
        // where its first user is set up.
        let mut allocated = vec![false; self.devices.len()];
        let mut allocate_branch = |unknowns: &mut MnaUnknowns, index: usize| -> SpiceResult<()> {
            if allocated[index] {
                return Ok(());
            }
            allocated[index] = true;
            let count = self.devices[index].branch_currents();
            let start = unknowns.len();
            unknowns.add_rows(count)?;
            self.bindings.branch_rows[index] = start..unknowns.len();
            Ok(())
        };
        for &index in order {
            let kind = self.devices[index].c_device_type();
            // vsrcset.c, ccvsset.c and asrcset.c create their own branch before
            // anything else; the other setups create internal nodes first.
            let branches_first = matches!(kind, CDeviceType::Vsrc | CDeviceType::Ccvs)
                || (kind == CDeviceType::Asrc && self.devices[index].branch_currents() > 0);
            if branches_first {
                allocate_branch(&mut self.bindings.unknowns, index)?;
            }
            // cccsset.c, ccvsset.c, cswsetup.c and asrcset.c call
            // CKTfndBranch for each controlling source, in order.
            for control in self.devices[index].controlling_sources() {
                if let Some(target) = names.get(&control.name)
                    && self.devices[index]
                        .control_branch(self.devices[target].as_ref())
                        .is_some()
                {
                    allocate_branch(&mut self.bindings.unknowns, target)?;
                }
            }
            for &node in &self.devices[index].origin.internal_nodes {
                self.bindings.unknowns.assign_deferred(node);
            }
            allocate_branch(&mut self.bindings.unknowns, index)?;
        }
        self.bindings.convergence_kinds =
            vec![crate::devices::ConvergenceKind::Voltage; self.bindings.unknowns.len()];
        for (device, rows) in self.devices.iter().zip(&self.bindings.branch_rows) {
            for (local, row) in rows.clone().enumerate() {
                self.bindings.convergence_kinds[row] = device.branch_convergence(local);
            }
        }
        let mut state_rows = Vec::with_capacity(self.devices.len());
        self.bindings.state_len = 0;
        for device in &self.devices {
            let states = device.state_count();
            let end = self
                .bindings
                .state_len
                .checked_add(states)
                .ok_or_else(|| SpiceError::circuit("device state slot count overflow"))?;
            state_rows.push(self.bindings.state_len..end);
            self.bindings.state_len = end;
        }
        self.bindings.state_rows = state_rows.into();
        self.bindings.control_rows = self
            .devices
            .iter()
            .map(|device| self.resolve_controls(device.as_ref(), names))
            .collect::<SpiceResult<_>>()?;
        self.bindings.mutual = self.resolve_mutual(names)?;
        Ok(())
    }

    /// Preflight every K card in linear input work before expanding any
    /// pairs, then resolve its names in original card order.
    fn resolve_mutual(&self, names: &DeviceNames) -> SpiceResult<MutualBinding> {
        let limits = self.mutual_limits;
        let mut total_pairs = 0_usize;
        let mut references = 0_usize;
        let mut card_count = 0_usize;
        for device in &self.devices {
            let Some(description) = device.mutual_coupling() else {
                continue;
            };
            let site = CouplingSite::of(device.as_ref());
            let count = description.inductors.len();
            if count < 2 {
                return Err(site.resource("a K card couples at least two inductors"));
            }
            let pairs =
                pair_count(count).ok_or_else(|| site.resource("mutual pair count overflows"))?;
            total_pairs = total_pairs
                .checked_add(pairs)
                .ok_or_else(|| site.resource("aggregate mutual pair count overflows"))?;
            if total_pairs > limits.pairs {
                return Err(site.resource(format_args!(
                    "mutual pair limit exceeded (aggregate {total_pairs} pairs, limit {})",
                    limits.pairs
                )));
            }
            references = references
                .checked_add(count)
                .ok_or_else(|| site.resource("mutual reference count overflows"))?;
            card_count = card_count
                .checked_add(1)
                .ok_or_else(|| site.resource("mutual card count overflows"))?;
            let bytes =
                mutual_storage_bytes(total_pairs, references, card_count, self.devices.len())
                    .ok_or_else(|| site.resource("mutual storage byte count overflows"))?;
            if bytes > limits.storage_bytes {
                return Err(site.resource(format_args!(
                    "mutual storage limit exceeded (planned {bytes} bytes, limit {})",
                    limits.storage_bytes
                )));
            }
        }
        if card_count == 0 {
            return Ok(MutualBinding::default());
        }

        // Every card passed aggregate resource preflight. Only resolved
        // references, not pairs, are materialized until group bounds pass.
        let mut cards = Vec::new();
        for (coupling, device) in self.devices.iter().enumerate() {
            let Some(description) = device.mutual_coupling() else {
                continue;
            };
            let site = CouplingSite::of(device.as_ref());
            if cards.is_empty() {
                cards = site.reserve(card_count)?;
            }
            let mut inductors = site.reserve(description.inductors.len())?;
            let mut seen = BTreeSet::new();
            for reference in description.inductors {
                let failure = |message: String| match &reference.location {
                    Some(location) => SpiceError::parse(location.clone(), message),
                    None => SpiceError::circuit(message),
                };
                let Some(index) = names.get(&reference.name) else {
                    return Err(failure(format!(
                        "{}: coupling to non-existent inductor {}",
                        device.name(),
                        reference.name
                    )));
                };
                let target = &self.devices[index];
                if target.inductance(&ModelContext::default()).is_none()
                    || self.bindings.branch_rows[index].len() != 1
                {
                    return Err(failure(format!(
                        "{}: {} is not an inductor (a K card couples inductors only)",
                        device.name(),
                        target.name()
                    )));
                }
                if !seen.insert(index) {
                    return Err(failure(format!(
                        "{}: couples inductor {} to itself",
                        device.name(),
                        target.name()
                    )));
                }
                inductors.push(index);
            }
            let pairs = pair_count(inductors.len())
                .ok_or_else(|| site.resource("mutual pair count overflows"))?;
            cards.push(ResolvedCard {
                coupling,
                inductors,
                pairs,
            });
        }
        MutualBinding::from_cards(
            cards,
            total_pairs,
            &self.bindings.branch_rows,
            &self.devices,
            limits,
        )
    }

    /// The mutual-inductance terms of every device's branch equation under
    /// `context`, by device ordinal (empty when the circuit has no K device).
    ///
    /// `M = k sqrt(|L1 L2|)` from the inductors' coupling inductances (C
    /// `MUTtemp`). Every inductive system (a connected group of coupled
    /// inductors, found with petgraph) is checked: its inductance matrix must
    /// be positive semidefinite. C only warns ("is not positive definite",
    /// `muttemp.c`) and then simulates a system that stores negative energy;
    /// the port rejects it. See [`crate::devices::mutual`].
    ///
    /// # Errors
    /// Stale numbering, unresolved K references, invalid inductances or a
    /// system that is not positive semidefinite.
    pub fn mutual_terms(&self, context: &ModelContext) -> SpiceResult<Vec<Vec<MutualTerm>>> {
        Ok(self.prepared_mutual_terms(context)?.to_vec())
    }

    fn prepared_mutual_terms(&self, context: &ModelContext) -> SpiceResult<PreparedMutualTerms> {
        self.check_numbering()?;
        let replacements = if self.bindings.mutual.pairs.is_empty() {
            Vec::new()
        } else {
            self.replacements(context)?
        };
        self.mutual_terms_with_replacements(context, &replacements)
    }

    /// The same load's already prepared overrides are also the source of
    /// coupling values; do not rebuild their contextual devices a second time.
    pub(super) fn mutual_terms_with_replacements(
        &self,
        context: &ModelContext,
        replacements: &[(usize, Box<dyn Device>)],
    ) -> SpiceResult<PreparedMutualTerms> {
        if self.bindings.mutual.pairs.is_empty() {
            return Ok(PreparedMutualTerms::default());
        }
        let device_at =
            |index: usize| -> &dyn Device { effective_device(self, replacements, index) };
        let mut values = BTreeMap::new();
        for &index in self.bindings.mutual.degrees.keys() {
            let device = device_at(index);
            let value = device.inductance(context).ok_or_else(|| {
                SpiceError::circuit(format!("{} is not an inductor", device.name()))
            })??;
            values.insert(index, value);
        }
        let site = CouplingSite::of(self.devices[self.bindings.mutual.pairs[0].coupling].as_ref());
        let mut mutuals = site.reserve(self.bindings.mutual.pairs.len())?;
        // INDload adds each K's flux in the MUT instance list's order, the
        // last card first (instances are prepended).
        for pair in self.bindings.mutual.pairs.iter().rev() {
            let (a, b) = (values[&pair.first], values[&pair.second]);
            let coefficient = device_at(pair.coupling)
                .mutual_coupling()
                .ok_or_else(|| SpiceError::circuit("stale coupling override"))?
                .coefficient;
            let inductance = coefficient * (a.coupling_base * b.coupling_base).abs().sqrt();
            if !inductance.is_finite() {
                return Err(SpiceError::circuit(format!(
                    "{}: nonfinite mutual inductance",
                    self.devices[pair.coupling].name()
                )));
            }
            mutuals.push(inductance);
        }
        // PSD additions use pair order while the INDload flux sum uses
        // reverse pair order. Both remain unchanged.
        mutuals.reverse();
        if let Some(cached) = &*self.runtime.mutual_evaluation.borrow()
            && cached.matches(&values, &mutuals)
        {
            return Ok(cached.terms.clone());
        }
        self.check_inductive_systems(&values, &mutuals)?;
        let mut terms = site.reserve(self.devices.len())?;
        terms.resize_with(self.devices.len(), Vec::new);
        for (&device, &degree) in &self.bindings.mutual.degrees {
            terms[device] = site.reserve(degree)?;
        }
        for (pair, &inductance) in self.bindings.mutual.pairs.iter().zip(&mutuals).rev() {
            let (a, b) = (values[&pair.first], values[&pair.second]);
            terms[pair.first].push(MutualTerm {
                row: self.bindings.branch_rows[pair.second].start,
                inductance,
                initial: b.initial,
            });
            terms[pair.second].push(MutualTerm {
                row: self.bindings.branch_rows[pair.first].start,
                inductance,
                initial: a.initial,
            });
        }
        let terms = PreparedMutualTerms(Some(terms.into()));
        self.runtime
            .mutual_evaluation
            .replace(Some(MutualEvaluation {
                values,
                mutuals,
                terms: terms.clone(),
            }));
        Ok(terms)
    }

    /// `MUTtemp`'s inductive-system check, as a rejection: every connected
    /// group of coupled inductors must have a positive semidefinite
    /// inductance matrix. The test runs on the matrix actually stamped
    /// (self-inductances `INDinduct / m`, mutuals from `INDinduct`) normalized
    /// to a unit diagonal (`M_ij / sqrt(L_i L_j)`, a congruence that preserves
    /// definiteness), whose eigenvalues are compared with a rounding
    /// tolerance of `64 n eps` times the largest magnitude. C's own check puts
    /// `INDinduct` on the diagonal; when only the stamped matrix fails (an
    /// `m != 1`), the message says that C stays silent.
    fn check_inductive_systems(
        &self,
        values: &std::collections::BTreeMap<usize, InductanceValue>,
        mutuals: &[Real],
    ) -> SpiceResult<()> {
        let mut budget = ValidationBudget::new(self.mutual_limits);
        for group in &self.bindings.mutual.groups {
            let site = CouplingSite::of(self.devices[group.first_coupling].as_ref());
            let mut smallest =
                |diagonal: fn(&InductanceValue) -> Real| -> SpiceResult<Option<Real>> {
                    // Coalesce only for PSD validation, not for the stamped or
                    // stored mutual terms. Each cell sees the same additions in
                    // the same order as the original dense construction.
                    let cells = group
                        .edges
                        .chunk_by(|a, b| (a.first_row, a.second_row) == (b.first_row, b.second_row))
                        .map(|edges| {
                            let first = &edges[0];
                            let value = edges.iter().fold(0.0, |sum, edge| {
                                let pair = &self.bindings.mutual.pairs[edge.pair];
                                let scale = (diagonal(&values[&pair.first])
                                    * diagonal(&values[&pair.second]))
                                .sqrt();
                                // A zero self inductance leaves a unit diagonal and
                                // zero normalized coupling, exactly as before.
                                let normalized = if scale == 0.0 {
                                    0.0
                                } else {
                                    mutuals[edge.pair] / scale
                                };
                                sum + normalized
                            });
                            (first.first_row, first.second_row, value)
                        });
                    budget.check(group.inductors.len(), group.edges.len(), cells, site)
                };
            // Compare the actually stamped matrix first, and the undivided
            // C diagonal only when needed for diagnostic classification.
            if let Some(stamped) = smallest(|value| value.effective)? {
                let c_indefinite = smallest(|value| value.coupling_base.abs())?.is_some();
                let mut couplings: Vec<_> = group
                    .edges
                    .iter()
                    .map(|edge| self.bindings.mutual.pairs[edge.pair].coupling)
                    .collect();
                couplings.sort_unstable();
                couplings.dedup();
                let names = |indices: &mut dyn Iterator<Item = usize>| {
                    indices
                        .map(|index| self.devices[index].name().to_owned())
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                let inductor_names = names(&mut group.inductors.iter().copied());
                let coupling_names = names(&mut couplings.iter().copied());
                let location = couplings.first().and_then(|index| {
                    self.devices[*index]
                        .mutual_coupling()
                        .and_then(|coupling| coupling.location.cloned())
                });
                let cause = if c_indefinite {
                    "a coupling with |k| > 1 or an inconsistent set of couplings stores \
                     negative energy; C only warns, see muttemp.c"
                } else {
                    "an inductor multiplicity m divides the self-inductance but not \
                     M = k sqrt(L1 L2), which C computes from INDinduct before /m; C checks \
                     that undivided matrix in muttemp.c, so it neither warns nor rejects \
                     here, and simulates the indefinite system"
                };
                return Err(SpiceError::Unsupported {
                    feature: format!(
                        "the inductive system {inductor_names} coupled by {coupling_names} is not \
                         positive semidefinite as stamped (smallest normalized eigenvalue \
                         {stamped:.6e}; {cause})"
                    ),
                    location,
                });
            }
        }
        Ok(())
    }

    /// Branch rows of the devices `device` senses (C `CKTfndBranch`, called
    /// from `CCCSsetup`/`CCVSsetup`): looked up by instance name after every
    /// device is known, so a controlling source may follow its user in the
    /// deck. The consumer selects the physical branch through
    /// [`Device::control_branch`]; expression probes can also use native
    /// storage currents.
    fn resolve_controls(
        &self,
        device: &dyn Device,
        names: &DeviceNames,
    ) -> Result<Vec<usize>, SpiceError> {
        device
            .controlling_sources()
            .iter()
            .map(|control| {
                let failure = |message: String| match &control.location {
                    Some(location) => SpiceError::parse(location.clone(), message),
                    None => SpiceError::circuit(message),
                };
                let Some(index) = names.get(&control.name) else {
                    return Err(failure(format!(
                        "{}: unknown controlling source {}",
                        device.name(),
                        control.name
                    )));
                };
                let target = &self.devices[index];
                let Some(branch) = device.control_branch(target.as_ref()) else {
                    return Err(failure(format!(
                        "{}: controlling source {} has no findable branch current for this control",
                        device.name(),
                        target.name()
                    )));
                };
                self.bindings.branch_rows[index]
                    .clone()
                    .nth(branch)
                    .ok_or_else(|| failure(format!("{}: missing branch row", target.name())))
            })
            .collect()
    }

    /// The resolved controlling-branch rows of device `index`.
    pub(super) fn controls(&self, index: usize) -> SpiceResult<&[usize]> {
        self.bindings
            .control_rows
            .get(index)
            .map(Vec::as_slice)
            .ok_or_else(|| {
                SpiceError::circuit("circuit numbering is stale; finalize after adding devices")
            })
    }

    /// Controlling-branch rows by device ordinal, after finalization (empty for
    /// devices that sense no branch current).
    ///
    /// # Errors
    /// Stale numbering, or the device's controlling source is unknown or has no
    /// findable branch current.
    pub fn control_rows(&self, index: usize) -> SpiceResult<Vec<usize>> {
        self.check_numbering()?;
        self.controls(index).map(<[usize]>::to_vec)
    }

    /// Validates the graph and renumbers the unknowns.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Circuit`] if a device refers to a node that is not in the
    /// table, or if two devices share an instance name — ngspice rejects the
    /// latter too, in `INP2dot`/`CKTcrtElt`. A controlling source (F/H) that
    /// names no device, or a device without a findable branch current, is a
    /// [`SpiceError::Parse`] at the reference (C: "unknown controlling source").
    pub fn finalize(&mut self) -> SpiceResult<()> {
        if self.bindings.identity.is_some() {
            return Ok(());
        }
        self.runtime = CircuitRuntime::default();
        let result = self.prepare_topology();
        if result.is_err() {
            self.bindings = TopologyBindings::default();
            self.runtime = CircuitRuntime::default();
        }
        result
    }

    fn prepare_topology(&mut self) -> SpiceResult<()> {
        // Validate graph before any predecessor binding or numbering mutation.
        self.elaboration_limits
            .check(self.elaboration_size()?, None)?;
        let names = self.validate_topology()?;
        let mut previous = crate::devices::Waveform::Constant(0.);
        let mut previous_dc = 0.;
        for entry in self.devices.iter_mut().rev() {
            if let Some((waveform, dc)) = entry
                .device
                .bind_voltage_predecessor(&previous, previous_dc)?
            {
                previous = waveform;
                previous_dc = dc;
            }
        }
        let order = self.setup_order();
        self.rebuild_unknowns_in_order(&order, &names)?;
        for node in self.nodes.nodes() {
            self.bindings.unknowns.checked_node_row(node.id)?;
        }
        self.check_port_numbering()?;
        self.allocate_matrix(order)?;
        self.bindings.stochastic_sources = self
            .devices
            .iter()
            .enumerate()
            .filter_map(|(ordinal, device)| device.has_stochastic_source().then_some(ordinal))
            .collect();
        self.bindings.prepared_devices = self
            .bindings
            .order
            .iter()
            .copied()
            .filter(|&index| self.devices[index].prepared_batch_key().is_some())
            .collect();
        self.bindings.names = names;
        self.bindings.identity = Some(TopologyId::allocate()?);
        Ok(())
    }

    /// Device ordinals in C's `CKTsetup`/`CKTload` order.
    fn setup_order(&self) -> Vec<usize> {
        let keys: Vec<crate::devices::setup::SetupKey<'_>> = self
            .devices
            .iter()
            .enumerate()
            .map(|(index, device)| crate::devices::setup::SetupKey {
                device_type: device.c_device_type(),
                model: self.devices[index].origin.model.as_deref(),
                generated_by: self.devices[index].origin.generated_by,
            })
            .collect();
        crate::devices::setup::setup_order(&keys)
    }

    /// The matrix elements device `index` allocates in its `DEVsetup`
    /// (`TSTALLOC` order; with `pole_zero` its `DEVpzSetup`), appended to
    /// `elements`.
    fn declare_elements(
        &self,
        index: usize,
        pole_zero_setups: u32,
        elements: &mut Vec<(usize, usize)>,
    ) -> SpiceResult<bool> {
        let coupled = self
            .bindings
            .mutual
            .setup_rows
            .get(&index)
            .map_or(&[][..], Vec::as_slice);
        let mut setup = crate::devices::setup::MatrixSetup {
            unknowns: &self.bindings.unknowns,
            branches: self.bindings.branch_rows[index].clone(),
            controls: self.controls(index)?,
            coupled,
            pole_zero: pole_zero_setups > 0,
            pole_zero_setups,
            elements,
        };
        self.devices[index].setup_matrix(&mut setup)
    }

    /// The `(row, col)` matrix elements device `index` allocates in its
    /// `DEVsetup`, in `TSTALLOC` order (`.sens` replays them into its
    /// perturbation matrix, `cktsens.c`).
    ///
    /// # Errors
    /// Stale numbering or an ordinal out of range.
    pub fn setup_elements(&self, index: usize) -> SpiceResult<Vec<(usize, usize)>> {
        self.check_numbering()?;
        if index >= self.devices.len() {
            return Err(SpiceError::circuit("device ordinal out of range"));
        }
        let mut elements = Vec::new();
        self.declare_elements(index, 0, &mut elements)?;
        Ok(elements)
    }

    /// C's `CKTsetup`: orders the devices and allocates the matrix elements
    /// they declare, in that order. A pattern that already starts with the
    /// same elements is kept, so its pivot order survives re-finalization
    /// (C sets a circuit up once).
    fn allocate_matrix(&mut self, order: Vec<usize>) -> SpiceResult<()> {
        let elements = self.ordered_elements(&order, 0)?;
        self.bindings.order = order;
        let n = self.unknown_count();
        let mut slot = self.runtime.pattern.borrow_mut();
        if elements.is_empty() || n == 0 {
            return Ok(());
        }
        let allocated = crate::maths::Pattern::new(n, elements)?;
        let same = slot.as_ref().is_some_and(|current| {
            current.order() == n
                && current.len() >= allocated.len()
                && current
                    .positions()
                    .take(allocated.len())
                    .eq(allocated.positions())
        });
        if !same {
            *slot = Some(std::sync::Arc::new(allocated));
        }
        Ok(())
    }

    /// The matrix elements the devices declare ([`Device::setup_matrix`]),
    /// visited in `order`: C's `CKTsetup`, or with `pole_zero` the
    /// `DEVpzSetup` pass of `CKTpzSetup` (`cktpzset.c`) on a fresh matrix.
    fn ordered_elements(
        &self,
        order: &[usize],
        pole_zero_setups: u32,
    ) -> SpiceResult<Vec<(usize, usize)>> {
        let mut elements = Vec::new();
        for &index in order {
            self.declare_elements(index, pole_zero_setups, &mut elements)?;
        }
        Ok(elements)
    }

    /// The elements `CKTpzSetup` (`cktpzset.c`) allocates on its fresh
    /// matrix before the drive, in allocation order: every device's
    /// `DEVpzSetup` (its `DEVsetup`, except that `VSRCpzSetup` adds the
    /// branch diagonal), then any element the circuit pattern learned from
    /// a load of a device that declares none. `setups` counts the
    /// `CKTpzSetup` runs, this one included (see
    /// [`crate::devices::setup::MatrixSetup::pole_zero_setups`]).
    ///
    /// The internal nodes of a device that declares its `DEVpzSetup`
    /// elements are its own: only its declaration places them, so a learned
    /// position on one of them (from the circuit's `DEVsetup` pattern) is not
    /// added. A `DEVpzSetup` that drops an internal node (`b4set.c` re-run)
    /// therefore leaves no row for it, as SPARSE creates rows only for
    /// referenced nodes.
    /// # Errors
    /// Mismatched numbering or missing bindings.
    pub fn pole_zero_elements(&self, setups: u32) -> SpiceResult<Vec<(usize, usize)>> {
        self.check_numbering()?;
        let setups = setups.max(1);
        let mut elements = Vec::new();
        let mut private = BTreeSet::new();
        for index in self.load_order() {
            if self.declare_elements(index, setups, &mut elements)? {
                private.extend(
                    self.devices[index]
                        .origin
                        .internal_nodes
                        .iter()
                        .filter_map(|&node| self.bindings.unknowns.node_row(node)),
                );
            }
        }
        if let Some(pattern) = self.matrix_pattern() {
            let known: BTreeSet<(usize, usize)> = elements.iter().copied().collect();
            for position in pattern.positions() {
                if !known.contains(&position)
                    && !private.contains(&position.0)
                    && !private.contains(&position.1)
                {
                    elements.push(position);
                }
            }
        }
        Ok(elements)
    }

    /// C `CKTconvTest`: every device's convergence test, in `CKTload` order,
    /// after a Newton pass whose iterate converged; `true` at the first
    /// device that has not converged.
    ///
    /// # Errors
    /// Mismatched numbering or a device evaluation failure.
    pub fn convergence_test(&self, request: &ConvergenceRequest<'_>) -> SpiceResult<bool> {
        self.prepare_run().convergence_test(request)
    }

    fn convergence_prepared(
        &self,
        run: &PreparedRun<'_>,
        request: &ConvergenceRequest<'_>,
    ) -> SpiceResult<bool> {
        self.validate_history(request.history)?;
        request.history.check_trial(request.trial)?;
        let n = self.unknown_count();
        if request.solution.len() != n
            || request.previous.len() != n
            || request.trial.values().len() != self.bindings.state_len
        {
            return Err(SpiceError::circuit(
                "convergence test does not match the circuit numbering",
            ));
        }
        let point = run.point(request.model_context)?;
        for index in self.load_order() {
            let context = crate::devices::traits::ConvergenceContext {
                unknowns: &self.bindings.unknowns,
                solution: request.solution,
                previous: request.previous,
                controls: self.controls(index)?,
                branch: self.bindings.branch_rows[index].clone().next(),
                integration: request.integration,
                states: &request.trial.values()[self.bindings.state_rows[index].clone()],
                phase: request.trial.phase(),
                model_context: request.model_context,
                time: request.time,
                reltol: request.reltol,
                abstol: request.abstol,
                vntol: request.vntol,
            };
            if point.device(self, index).convergence_test(&context)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Device ordinals in finalized `CKTload` order.
    pub(crate) fn load_order(&self) -> impl Iterator<Item = usize> + '_ {
        self.bindings.order.iter().copied()
    }

    /// `vsrctemp.c` (`VSRCtemp`, run before every analysis): the RF port
    /// numbers must be exactly `1..=N` for `N` ports. A number above `N` is
    /// "incorrect port ordering" (a gap or an out-of-range index), a repeated
    /// one a "duplicate port Index"; both are fatal in C.
    fn check_port_numbering(&self) -> SpiceResult<()> {
        let ports: Vec<_> = self
            .devices
            .iter()
            .filter_map(|device| device.rf_port().map(|port| (device.name(), port.number)))
            .collect();
        let mut seen = BTreeSet::new();
        for (name, number) in &ports {
            if *number > ports.len() {
                return Err(SpiceError::circuit(format!(
                    "{name}: incorrect port ordering (portnum {number} with {} port(s); \
                     ports must be numbered 1..=N)",
                    ports.len()
                )));
            }
            if !seen.insert(*number) {
                return Err(SpiceError::circuit(format!(
                    "{name}: duplicate port Index {number}"
                )));
            }
        }
        Ok(())
    }

    /// Builds a validated petgraph incidence snapshot of the current circuit.
    ///
    /// Includes every node (even unused ones), every device, and one edge for
    /// each terminal in `Device::terminals()` order. This follows the binding in
    /// `src/spicelib/devices/cktbindnode.c::CKTbindNode`; C numbers ports from 1,
    /// while edge weights here are zero-based. Device order remains deck order.
    /// Use petgraph's connectivity/traversal algorithms on the returned graph.
    ///
    /// The snapshot is rebuilt on demand: an explicit device edit can change topology,
    /// so a cached graph could silently become stale. A structural path through
    /// a capacitor or multiport device need not be a DC conductive path. This
    /// method never rejects a circuit for disconnectedness or claims a matrix
    /// is nonsingular; analysis-specific checks belong to the eventual engine.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Circuit`] for dangling terminals or duplicate names, as in
    /// [`Circuit::finalize`]. No circuit state is changed on failure.
    pub fn topology(&self) -> SpiceResult<CircuitGraph> {
        self.validate_topology()?;
        let mut graph = CircuitGraph::default();
        let node_vertices: Vec<_> = self
            .nodes
            .nodes()
            .iter()
            .map(|node| graph.add_node(CircuitVertex::Node(node.id)))
            .collect();
        for (index, device) in self.devices.iter().enumerate() {
            let vertex = graph.add_node(CircuitVertex::Device(index));
            for (port, terminal) in device.terminals().iter().enumerate() {
                graph.add_edge(vertex, node_vertices[terminal.index()], port);
            }
        }
        Ok(graph)
    }

    /// Validate exactly the incidence constraints enforced by `topology()`,
    /// without constructing a graph just to discard it. Build the lookup index
    /// in the same pass. Keep the existing Unicode duplicate-name rule as
    /// well as ASCII case-insensitive reference resolution.
    fn validate_topology(&self) -> SpiceResult<DeviceNames> {
        let mut names = DeviceNames::default();
        let mut unicode = self
            .devices
            .iter()
            .any(|d| !d.name().is_ascii())
            .then(BTreeSet::new);
        for (index, device) in self.devices.iter().enumerate() {
            for terminal in device.terminals() {
                if self.nodes.node(*terminal).is_none() {
                    return Err(SpiceError::circuit(format!(
                        "device {} refers to {terminal}, which is not in the node table",
                        device.name()
                    )));
                }
            }
            let duplicate = unicode.as_mut().map_or_else(
                || names.get(device.name()).is_some(),
                |seen| !seen.insert(device.name().to_lowercase()),
            );
            if duplicate {
                return Err(SpiceError::circuit(format!(
                    "duplicate instance name '{}'",
                    device.name()
                )));
            }
            names.insert(device.name(), index);
        }
        Ok(names)
    }

    /// Adds a device, checking that its terminals are already in the node table
    /// and that its name is not taken.
    ///
    /// # Errors
    ///
    /// [`SpiceError::Circuit`] on a dangling terminal or a duplicate name.
    pub fn add_device(&mut self, device: Box<dyn Device>) -> SpiceResult<()> {
        for terminal in device.terminals() {
            if self.nodes.node(*terminal).is_none() {
                return Err(SpiceError::circuit(format!(
                    "device {} refers to {terminal}, which is not in the node table",
                    device.name()
                )));
            }
        }
        if self.device(device.name()).is_some() {
            return Err(SpiceError::circuit(format!(
                "duplicate instance name '{}'",
                device.name()
            )));
        }
        let size = self
            .elaboration_size()?
            .checked_add(ElaborationSize::new(1, 0, device.state_count()), None)?;
        self.elaboration_limits.check(size, None)?;
        self.devices
            .try_reserve(1)
            .map_err(|error| resource_error(None, format!("cannot reserve device: {error}")))?;
        let internal = device
            .terminals()
            .iter()
            .copied()
            .filter(|&node| {
                self.nodes
                    .node(node)
                    .is_some_and(|node| node.kind == NodeKind::Internal)
            })
            .collect();
        self.invalidate_topology();
        self.devices.push(ElaboratedDevice {
            device,
            origin: DeviceOrigin {
                internal_nodes: internal,
                ..DeviceOrigin::default()
            },
        });
        Ok(())
    }

    /// Append an already budgeted device graph without synthesizing a netlist.
    /// Node ownership is assigned once, to the first device in the graph.
    pub(super) fn add_prepared_devices(
        &mut self,
        size: ElaborationSize,
        build: impl FnOnce(&mut crate::primitives::NodeAppend<'_>) -> SpiceResult<Vec<Box<dyn Device>>>,
    ) -> SpiceResult<()> {
        let existing = self.elaboration_size()?;
        self.elaboration_limits
            .check(existing.checked_add(size, None)?, None)?;
        let mut seen: BTreeSet<_> = self
            .devices
            .iter()
            .map(|d| d.name().to_lowercase())
            .collect();
        let devices = &mut self.devices;
        let staged = self.nodes.transaction(|nodes| {
            let mark = nodes.len();
            let generated = build(nodes)?;
            let mut staged = Vec::new();
            reserve(&mut staged, generated.len(), None)?;
            for device in generated {
                if !seen.insert(device.name().to_lowercase()) {
                    return Err(SpiceError::circuit(format!(
                        "duplicate instance name '{}'",
                        device.name()
                    )));
                }
                if device
                    .terminals()
                    .iter()
                    .any(|&id| nodes.node(id).is_none())
                {
                    return Err(SpiceError::circuit(
                        "prepared device refers to an unknown node",
                    ));
                }
                staged.push(ElaboratedDevice {
                    device,
                    origin: DeviceOrigin::default(),
                });
            }
            let actual = ElaborationSize::new(
                staged.len(),
                nodes.len() - mark,
                Self::declared_state_slots(&staged, None)?,
            );
            if actual != size {
                return Err(SpiceError::circuit(
                    "prepared graph differs from its elaboration budget",
                ));
            }
            if let Some(first) = staged.first_mut() {
                first.origin.internal_nodes = nodes.nodes()[mark..]
                    .iter()
                    .filter(|node| node.kind == NodeKind::Internal)
                    .map(|node| node.id)
                    .collect();
            }
            reserve(devices, staged.len(), None)?;
            Ok(staged)
        })?;
        self.invalidate_topology();
        self.devices.extend(staged);
        Ok(())
    }

    /// Branch rows by device ordinal, after finalization.
    pub fn branch_rows(&self, index: usize) -> Option<std::ops::Range<usize>> {
        self.bindings.identity.as_ref()?;
        self.bindings.branch_rows.get(index).cloned()
    }

    pub(crate) fn convergence_kinds(&self) -> &[crate::devices::ConvergenceKind] {
        &self.bindings.convergence_kinds
    }

    /// The name of every unknown by matrix row, as ngspice names them
    /// (`CKTnodName`): the node name, or `<device>#branch` for a branch
    /// current (`CKTmkCur`; a device's further branches add their ordinal).
    /// Solver diagnostics use them (`singular matrix:  check node X`).
    pub fn unknown_names(&self) -> crate::maths::UnknownNames {
        let n = self.unknown_count();
        if let Some(names) = &*self.runtime.unknown_names.borrow()
            && names.len() == n
        {
            return std::sync::Arc::clone(names);
        }
        let mut names: Vec<String> = (0..n).map(|row| format!("row {row}")).collect();
        for node in self.nodes.nodes() {
            if let Some(row) = self.bindings.unknowns.node_row(node.id)
                && let Some(name) = names.get_mut(row)
            {
                name.clone_from(&node.name);
            }
        }
        for (device, range) in self.devices.iter().zip(&self.bindings.branch_rows) {
            for (k, row) in range.clone().enumerate() {
                if let Some(name) = names.get_mut(row) {
                    *name = if k == 0 {
                        format!("{}#branch", device.name())
                    } else {
                        format!("{}#branch{}", device.name(), k + 1)
                    };
                }
            }
        }
        let names: crate::maths::UnknownNames = names.into();
        *self.runtime.unknown_names.borrow_mut() = Some(std::sync::Arc::clone(&names));
        names
    }

    /// State slots by device ordinal, after finalization. Slots are not
    /// matrix rows.
    pub fn state_rows(&self, index: usize) -> Option<std::ops::Range<usize>> {
        self.bindings.identity.as_ref()?;
        self.bindings.state_rows.get(index).cloned()
    }

    /// Total number of state slots, after finalization.
    #[must_use = "the state length determines the required device history storage"]
    pub const fn state_len(&self) -> usize {
        self.bindings.state_len
    }

    /// Create an empty history bound to the finalized circuit.
    ///
    /// Identical fresh histories share the immutable initial revision.
    ///
    /// # Errors
    /// The circuit has no validated topology.
    pub fn state_history(&self) -> SpiceResult<StateHistory> {
        self.make_state_history(true)
    }

    /// A physical trial history without analysis-to-analysis carry state.
    pub(crate) fn independent_state_history(&self) -> SpiceResult<StateHistory> {
        self.make_state_history(false)
    }

    fn make_state_history(&self, carry: bool) -> SpiceResult<StateHistory> {
        Ok(StateHistory::bound(
            self.topology_id()?,
            Arc::clone(&self.bindings.state_rows),
            self.bindings.state_len,
            Arc::clone(&self.runtime.initial_revision.borrow()),
            self.runtime
                .carried_state
                .borrow()
                .as_ref()
                .filter(|snapshot| carry && self.snapshot_matches(snapshot))
                .map(|snapshot| Arc::clone(&snapshot.values)),
            self.devices
                .iter()
                .enumerate()
                .filter_map(|(index, device)| {
                    device.delay_line().map(|line| (index, line.new_history()))
                }),
        ))
    }

    pub(super) fn validate_history(&self, history: &StateHistory) -> SpiceResult<()> {
        if !history.matches_topology(self.topology_id()?) {
            return Err(SpiceError::circuit(
                "state history belongs to a different finalized topology",
            ));
        }
        Ok(())
    }

    /// Whether any device has a delay history (a transmission line).
    #[must_use = "the presence of delay lines determines whether delay history must be maintained"]
    pub fn has_delay_lines(&self) -> bool {
        self.devices
            .iter()
            .any(|device| device.delay_line().is_some())
    }

    /// C's `MODEINITSMSIG` load at the operating point `bias` (`acan.c`,
    /// `distoan.c`, `noisean.c`, `pzan.c`) as far as it outlives the
    /// analysis: the state slots devices write there and later analyses read
    /// ([`Device::small_signal_state`]), recorded into the state vector this
    /// circuit carries forward.
    ///
    /// # Errors
    /// Stale numbering, a bias of the wrong size, or device physics errors.
    pub fn record_small_signal_state(
        &self,
        context: &ModelContext,
        bias: &crate::maths::Vector,
    ) -> SpiceResult<()> {
        self.prepare_run().record_small_signal_state(context, bias)
    }

    fn record_small_signal_state_prepared(
        &self,
        run: &PreparedRun<'_>,
        context: &ModelContext,
        bias: &Vector,
    ) -> SpiceResult<()> {
        self.check_numbering()?;
        if bias.len() != self.unknown_count() {
            return Err(SpiceError::circuit("small-signal state bias dimensions"));
        }
        let mut state = self
            .runtime
            .carried_state
            .borrow()
            .as_ref()
            .filter(|snapshot| self.snapshot_matches(snapshot))
            .map_or_else(
                || vec![0.; self.bindings.state_len],
                |snapshot| snapshot.values.to_vec(),
            );
        self.small_signal_slots(run, context, bias, &mut state)?;
        let snapshot = self.bind_state(state)?;
        *self.runtime.carried_state.borrow_mut() = Some(snapshot);
        self.runtime.initial_revision.replace(Arc::new(()));
        Ok(())
    }

    fn small_signal_slots(
        &self,
        run: &PreparedRun<'_>,
        context: &ModelContext,
        bias: &Vector,
        state: &mut [Real],
    ) -> SpiceResult<()> {
        let operating = crate::devices::distortion::DistortionContext::new(
            context,
            &self.bindings.unknowns,
            bias,
        )?;
        let point = run.point(context)?;
        for index in 0..self.devices.len() {
            let rows = self.bindings.state_rows[index].clone();
            if let Some(slots) = state.get_mut(rows) {
                point
                    .device(self, index)
                    .small_signal_state(&operating, slots)?;
            }
        }
        Ok(())
    }

    /// Keeps the final accepted transient load's device state for later
    /// analyses (`dctran.c`: instance fields outlive the transient). Called
    /// only after the whole transient succeeds, never on a rejected trial.
    ///
    /// # Errors
    /// Stale numbering, a mismatched history, or no accepted point.
    pub(crate) fn record_final_transient_state(&self, history: &StateHistory) -> SpiceResult<()> {
        self.check_numbering()?;
        self.validate_history(history)?;
        let state = history.accepted_snapshot(1).ok_or_else(|| {
            SpiceError::circuit("a completed transient has no accepted device state")
        })?;
        let snapshot = self.bind_state(state)?;
        *self.runtime.final_transient.borrow_mut() = Some(snapshot);
        Ok(())
    }

    /// Explicitly bind supplied finite state values to this finalized circuit.
    ///
    /// # Errors
    /// Stale topology, incorrect length or nonfinite values.
    pub fn bind_state(&self, values: impl Into<Arc<[Real]>>) -> SpiceResult<StateSnapshot> {
        let topology = self.topology_id()?;
        let values = values.into();
        if values.len() != self.bindings.state_len || values.iter().any(|value| !value.is_finite())
        {
            return Err(SpiceError::circuit(
                "invalid state snapshot dimensions/values",
            ));
        }
        Ok(StateSnapshot { values, topology })
    }

    fn snapshot_matches(&self, snapshot: &StateSnapshot) -> bool {
        self.bindings.identity == Some(snapshot.topology)
    }

    pub(super) fn snapshot_values<'a>(
        &self,
        state: Option<&'a StateSnapshot>,
    ) -> SpiceResult<Option<&'a [Real]>> {
        self.check_numbering()?;
        if state.is_some_and(|snapshot| !self.snapshot_matches(snapshot)) {
            return Err(SpiceError::circuit(
                "state snapshot belongs to a different finalized topology",
            ));
        }
        Ok(state.map(StateSnapshot::values))
    }

    /// A snapshot of the last successful transient's final load state, in
    /// [`Self::state_rows`] order. Unlike the final solution, this retains
    /// the actual load's limited voltages and derived instance quantities.
    /// [`Self::inherit_from`] carries it to the next analysis. A changed
    /// device/state layout invalidates it; re-finalizing the same circuit
    /// preserves it. It is not an integration history.
    #[must_use = "the returned transient state is required to inspect or reuse the final solution"]
    pub fn final_transient_state(&self) -> Option<Vec<Real>> {
        self.runtime
            .final_transient
            .borrow()
            .as_ref()
            .filter(|state| self.snapshot_matches(state))
            .map(|state| state.values.to_vec())
    }

    pub(crate) fn topology_id(&self) -> SpiceResult<TopologyId> {
        self.bindings
            .identity
            .ok_or_else(|| SpiceError::circuit("circuit topology is stale or not finalized"))
    }

    pub(super) fn observation_device_index(&self, name: &str) -> Option<usize> {
        self.bindings.names.get(name)
    }

    pub(super) fn check_numbering(&self) -> SpiceResult<()> {
        self.topology_id().map(drop)
    }

    /// The resistor named `name` (case-insensitive) and its device ordinal,
    /// identified by [`Device::resistor_metadata`], never by the name's first
    /// letter.
    #[must_use = "the returned index and metadata identify the resistor and its parameters"]
    pub fn resistor(&self, name: &str) -> Option<(usize, ResistorMetadata)> {
        let index = self.device_index(name)?;
        self.devices[index]
            .resistor_metadata()
            .map(|metadata| (index, metadata))
    }

    /// An immutable per-point override of resistor `name`'s supplied scalar.
    /// Carry it in a [`ModelContext`] ([`ModelContext::with_resistor_override`]);
    /// no device changes. See [`crate::devices::sweep`] for supplied-versus-effective
    /// semantics.
    ///
    /// # Errors
    /// [`SpiceError::Circuit`] if `name` is not a resistor, or `supplied` is
    /// nonfinite, zero, or has a nonfinite conductance.
    pub fn resistor_override(&self, name: &str, supplied: Real) -> SpiceResult<ResistorOverride> {
        self.bind_resistor_target(name)?.at_resistance(supplied)
    }

    /// The effective resistance (ohms) `target` stamps under `context`'s
    /// temperatures: the supplied scalar with the device's own temperature, TC,
    /// scale and multiplicity laws applied. Nothing is mutated.
    ///
    /// # Errors
    /// A stale override, an invalid context, or an invalid derived value.
    pub fn effective_resistance(
        &self,
        target: &ResistorOverride,
        context: &ModelContext,
    ) -> SpiceResult<Real> {
        if target.topology() != self.topology_id()? {
            return Err(SpiceError::circuit("foreign or stale resistor override"));
        }
        let device = self
            .devices
            .get(target.device())
            .ok_or_else(|| SpiceError::circuit("stale resistor override"))?;
        context.validate(&MODEL_CONTEXT_LOCATION)?;
        device.resistor_effective(target.supplied(), context)
    }

    /// The device named `name` (case-insensitive) and its ordinal, with the
    /// canonical keyword of its swept instance parameter `keyword`
    /// ([`Device::instance_parameter`]), the target of `.dc @name[keyword]`.
    ///
    /// # Errors
    /// [`SpiceError::Unsupported`] when no device is named `name` or C has no
    /// settable real instance parameter `keyword` on it (C `dctrcurv.c`
    /// rejects both); [`SpiceError::NotYetPorted`] when C sweeps it but this
    /// port does not yet.
    pub fn instance_parameter(
        &self,
        name: &str,
        keyword: &str,
    ) -> SpiceResult<(usize, ParameterKey)> {
        let Some(index) = self.device_index(name) else {
            return Err(SpiceError::Unsupported {
                feature: format!("DC sweep parameter target: no instance named {name}"),
                location: None,
            });
        };
        let device = &self.devices[index];
        if let Some(parameter) = device.bind_numeric_parameter(keyword) {
            return Ok((index, parameter));
        }
        if crate::devices::sweep::c_instance_parameter_known(device.designator(), keyword) {
            return Err(SpiceError::not_yet_ported(
                format!(
                    "DC sweep of instance parameter {keyword} of {}",
                    device.name()
                ),
                "src/spicelib/analysis/dctrcurv.c (DCTfindInstParam, DCTsetInstParam)",
            ));
        }
        Err(SpiceError::Unsupported {
            feature: format!(
                "DC sweep parameter target: {} has no settable real instance parameter {keyword}",
                device.name()
            ),
            location: None,
        })
    }

    /// An immutable per-point override giving instance `name`'s parameter
    /// `keyword` the value `value` (`.dc @name[keyword]`). Carry it in a
    /// [`ModelContext`] ([`ModelContext::with_instance_override`]); no device
    /// changes. The value is validated by building the replacement once at
    /// `context`.
    ///
    /// # Errors
    /// As [`Self::instance_parameter`], or an invalid value.
    pub fn instance_override(
        &self,
        name: &str,
        keyword: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<InstanceOverride> {
        let target = self.bind_instance_override(name, keyword, value)?;
        self.devices[target.device()].with_bound_parameter(target.parameter(), value, context)?;
        Ok(target)
    }

    /// Bind the device's canonical numeric setter to this finalized topology.
    /// No trial value or throwaway device construction is needed.
    ///
    /// # Errors
    /// Unfinalized topology, unknown device or unsupported parameter.
    pub fn bind_instance_target(&self, name: &str, keyword: &str) -> SpiceResult<InstanceTarget> {
        let (index, parameter) = self.instance_parameter(name, keyword)?;
        #[cfg(test)]
        preparation_metrics::record(|counts| counts.bindings += 1);
        Ok(InstanceTarget::new(self.topology_id()?, index, parameter))
    }

    /// Bind a resistor's supplied (not effective) resistance via its metadata.
    ///
    /// # Errors
    /// Unfinalized topology or target is not a resistor.
    pub fn bind_resistor_target(&self, name: &str) -> SpiceResult<InstanceTarget> {
        let (index, _) = self
            .resistor(name)
            .ok_or_else(|| SpiceError::circuit(format!("{name} is not a resistor")))?;
        #[cfg(test)]
        preparation_metrics::record(|counts| counts.bindings += 1);
        Ok(InstanceTarget::new(
            self.topology_id()?,
            index,
            ParameterKey::Native("resistance"),
        ))
    }

    pub(crate) fn bind_instance_override(
        &self,
        name: &str,
        keyword: &str,
        value: Real,
    ) -> SpiceResult<InstanceOverride> {
        self.bind_instance_target(name, keyword)?.at(value)
    }

    /// A DC value override of the independent V or I source `name`, as
    /// `dctrcurv.c` applies it to the swept source (`dcValue`, `dcGiven`).
    ///
    /// # Errors
    /// An unknown name, a device that is not an independent source, or a
    /// nonfinite value.
    pub fn source_override(
        &self,
        name: &str,
        value: Real,
        context: &ModelContext,
    ) -> SpiceResult<InstanceOverride> {
        let target = self.bind_source_override(name, value)?;
        self.devices[target.device()].with_instance_parameter("dc", value, context)?;
        Ok(target)
    }

    /// Bind an independent V/I source's DC setter without a trial value.
    ///
    /// # Errors
    /// Unfinalized topology, unknown name or non-source device.
    pub fn bind_source_target(&self, name: &str) -> SpiceResult<InstanceTarget> {
        let index = self
            .device_index(name)
            .filter(|&i| matches!(self.devices[i].designator().to_ascii_lowercase(), 'v' | 'i'))
            .ok_or_else(|| SpiceError::circuit("invalid or unknown DC source override"))?;
        #[cfg(test)]
        preparation_metrics::record(|counts| counts.bindings += 1);
        Ok(InstanceTarget::new(
            self.topology_id()?,
            index,
            ParameterKey::Native("dc"),
        ))
    }

    pub(crate) fn bind_source_override(
        &self,
        name: &str,
        value: Real,
    ) -> SpiceResult<InstanceOverride> {
        self.bind_source_target(name)?.at(value)
    }

    /// The Newton load of the single device at ordinal `index`, with `device`
    /// (the circuit's own or a stand-in with the same terminals, branch rows
    /// and state layout) in its place: C's `DEVload` of one isolated instance,
    /// as `.sens` performs it (`cktsens.c`). Contributions are added to
    /// `matrix` and `rhs`; mutual-inductance terms are those of the circuit's
    /// own devices.
    ///
    /// # Errors
    /// Stale numbering, an ordinal out of range, mismatched dimensions or a
    /// device failure.
    pub fn load_device(
        &self,
        index: usize,
        device: &dyn Device,
        request: &LoadRequest<'_>,
        matrix: &mut SparseMatrix,
        rhs: &mut Vector,
        trial: &mut TrialState,
    ) -> SpiceResult<()> {
        self.validate_history(request.history)?;
        request.history.check_trial(trial)?;
        self.check_matrix_pattern(matrix)?;
        let n = self.unknown_count();
        if index >= self.devices.len()
            || matrix.rows() != n
            || matrix.cols() != n
            || rhs.len() != n
            || request.solution.len() != n
            || request.history.len() != self.bindings.state_len
            || trial.values().len() != self.bindings.state_len
        {
            return Err(SpiceError::circuit(
                "single-device load does not match the circuit numbering",
            ));
        }
        let mutual = self.prepared_mutual_terms(request.model_context)?;
        self.stamp_isolated(
            index,
            device,
            request,
            matrix,
            StampRhs::dense(rhs),
            request.history.device_of(trial, index)?,
            &mutual,
            None,
        )
    }

    /// A checked isolated load's shared stamping path. Observations reuse the
    /// point's already prepared coupling terms for every isolated device.
    #[expect(
        clippy::too_many_arguments,
        reason = "explicit isolated load and its prepared coupling inputs"
    )]
    pub(super) fn stamp_isolated(
        &self,
        index: usize,
        device: &dyn Device,
        request: &LoadRequest<'_>,
        matrix: &mut SparseMatrix,
        rhs: StampRhs<'_>,
        states: DeviceState<'_>,
        mutual: &[Vec<MutualTerm>],
        ltra: Option<&mut LtraWorkspace>,
    ) -> SpiceResult<()> {
        let states = states.with_ltra(ltra, request.history, index);
        device.stamp(&mut StampContext {
            matrix,
            rhs,
            unknowns: &self.bindings.unknowns,
            nodes: &self.nodes,
            solution: request.solution,
            temperature: request.model_context.temperature,
            nominal_temperature: request.model_context.nominal_temperature,
            gmin: request.model_context.gmin,
            reltol: request.model_context.reltol,
            frequency: request.model_context.frequency,
            mode: request.mode,
            branches: self.bindings.branch_rows[index].clone(),
            controls: self.controls(index)?,
            mutual: mutual.get(index).map_or(&[], Vec::as_slice),
            integration: request.integration,
            states,
            forcing: request.forcing,
            source_factor: request.model_context.source_factor,
            // An isolated load has no preceding voltage source.
            voltage_source_value: &mut 0.,
        })
    }

    /// The small-signal assembly of the single device at ordinal `index`, with
    /// `device` in its place, added to `system` (C's `DEVacLoad` of one
    /// isolated instance, as `.sens` performs it). An inductor stamps no
    /// mutual-inductance terms: `.sens` loads each K's `MUTacLoad` as a device
    /// of its own. `bias` and `state` are as for
    /// [`Self::small_signal_system_at`].
    ///
    /// # Errors
    /// Stale numbering, an ordinal out of range, mismatched dimensions or
    /// unsupported device physics.
    pub fn assemble_small_signal_device(
        &self,
        index: usize,
        device: &dyn Device,
        context: &ModelContext,
        bias: &Vector,
        state: Option<&StateSnapshot>,
        system: &mut crate::devices::linear::LinearSystem,
    ) -> SpiceResult<()> {
        let state = self.snapshot_values(state)?;
        let n = self.unknown_count();
        if index >= self.devices.len()
            || bias.len() != n
            || system.a.rows() != n
            || system.e.rows() != n
            || state.is_some_and(|state| state.len() != self.bindings.state_len)
        {
            return Err(SpiceError::circuit(
                "single-device small-signal assembly does not match the circuit numbering",
            ));
        }
        let range = &self.bindings.branch_rows[index];
        let mut linear = crate::devices::linear::LinearContext {
            model_context: context,
            system,
            unknowns: &self.bindings.unknowns,
            branch: (!range.is_empty()).then_some(range.start),
            controls: self.controls(index)?,
            mutual: &[],
            states: match state {
                Some(state) => Some(
                    state
                        .get(self.bindings.state_rows[index].clone())
                        .ok_or_else(|| SpiceError::circuit("bias state is too short"))?,
                ),
                None => None,
            },
            pole_zero_statements: None,
            pole_zero_setups: 0,
        };
        device.assemble_small_signal(&mut linear, bias)
    }

    /// Disposable devices carrying the context's resistor and instance-parameter
    /// overrides, as `(device ordinal, replacement)`. The circuit's own devices
    /// are untouched; callers stamp the replacement instead of the original for
    /// that ordinal. Several instance overrides of one device are applied in
    /// slot order to the same replacement.
    /// Rebind stochastic source streams to the caller's candidate identity.
    /// The replacement preserves terminals, branch rows and device state
    /// layout so finalized topology and symbolic matrices remain valid.
    pub(crate) fn reseed_stochastic_sources(&mut self, identity: u64) {
        for &ordinal in &self.bindings.stochastic_sources {
            if let Some(source) = self.devices.get_mut(ordinal)
                && let Some(replacement) = source.device.with_stochastic_candidate(identity)
            {
                source.device = replacement;
            }
        }
    }

    /// Clear analysis-scoped accepted/trial state when an optimizer session
    /// changes its candidate. Preserve finalized bindings, sparse pattern and
    /// prepared names; only independent simulation state is invalidated.
    pub(crate) fn clear_candidate_analysis_state(&mut self) {
        self.runtime.completed.clear();
        self.runtime.carried_state.get_mut().take();
        self.runtime.final_transient.get_mut().take();
        self.runtime.fixed_stamps.get_mut().forget();
        self.runtime.mutual_evaluation.get_mut().take();
        *self.runtime.initial_revision.get_mut() = Arc::new(());
    }

    /// Validate an arbitrary sparse numeric candidate overlay transactionally.
    /// This uses the same ordered replacement construction, physical setters
    /// and mutual-coupling checks as an actual prepared simulation load, but
    /// does not publish any value on failure. The caller owns the resulting
    /// overlay and commits it to its exclusive circuit after validation.
    pub(crate) fn validate_candidate_overrides(
        &self,
        context: &ModelContext,
        overrides: &[CandidateNumericOverride],
    ) -> SpiceResult<()> {
        #[cfg(test)]
        preparation_metrics::record(|counts| counts.replacement_validations += 1);
        context.validate(&MODEL_CONTEXT_LOCATION)?;
        let replacements = self.replacements_with_overrides(context, overrides)?;
        self.mutual_terms_with_replacements(context, &replacements)?;
        let inputs = PreparedRun::dc_device_context(context);
        for (_, device) in &replacements {
            device.validate_dc_inputs(&inputs)?;
        }
        Ok(())
    }

    /// Commit already-validated worker-owned numeric candidates. An exclusive
    /// circuit borrow rules out stale prepared analysis scopes during update.
    pub(crate) fn set_candidate_overrides(&mut self, overrides: Vec<CandidateNumericOverride>) {
        self.runtime.candidate_overrides = overrides;
        self.runtime.control_overrides.clear();
    }

    /// Validate and publish persistent script edits over the current candidate.
    /// A script edit does not consume the fixed DC sweep slots. The original
    /// candidate and its immutable device recipes remain available for reset.
    pub(crate) fn set_control_overrides(
        &mut self,
        context: &ModelContext,
        edits: &[InstanceOverride],
    ) -> SpiceResult<()> {
        if edits.is_empty() {
            self.runtime.control_overrides.clear();
            self.clear_candidate_analysis_state();
            return Ok(());
        }
        let base = assemble_numeric_targets(
            context,
            &self.runtime.candidate_overrides,
            self.runtime.candidate_overrides.len()
                + context.resistor_overrides.len()
                + context.instance_overrides.len(),
        )?;
        let combined = ordered_control_targets(
            base.into_iter().chain(
                edits
                    .iter()
                    .copied()
                    .map(CandidateNumericOverride::Instance),
            ),
        )?;
        let mut context = *context;
        context.resistor_overrides.fill(None);
        context.instance_overrides.fill(None);
        self.validate_candidate_overrides(&context, &combined)?;
        self.runtime.control_overrides = combined;
        self.clear_candidate_analysis_state();
        Ok(())
    }

    fn replacements(&self, context: &ModelContext) -> SpiceResult<Vec<(usize, Box<dyn Device>)>> {
        if self.runtime.control_overrides.is_empty() {
            return self.replacements_with_overrides(context, &self.runtime.candidate_overrides);
        }
        if context.resistor_overrides.iter().all(Option::is_none)
            && context.instance_overrides.iter().all(Option::is_none)
        {
            return self.replacements_with_overrides(context, &self.runtime.control_overrides);
        }
        // The control list already contains the underlying candidate's
        // sites. Context slots now belong only to the particular analysis
        // point, not the candidate. Overlay them last.
        let point = assemble_numeric_targets(
            context,
            &[],
            context.resistor_overrides.len() + context.instance_overrides.len(),
        )?;
        let combined =
            ordered_control_targets(self.runtime.control_overrides.iter().copied().chain(point))?;
        let mut context = *context;
        context.resistor_overrides.fill(None);
        context.instance_overrides.fill(None);
        self.replacements_with_overrides(&context, &combined)
    }

    fn single_candidate_replacement(
        &self,
        context: &ModelContext,
        overrides: &[CandidateNumericOverride],
    ) -> SpiceResult<Vec<(usize, Box<dyn Device>)>> {
        // A single setter is the common DC/optimizer case: construct its
        // validated device directly, without maps or staging allocations.
        let target = context
            .resistor_overrides
            .iter()
            .flatten()
            .copied()
            .map(CandidateNumericOverride::Resistor)
            .chain(
                context
                    .instance_overrides
                    .iter()
                    .flatten()
                    .copied()
                    .map(CandidateNumericOverride::Instance),
            )
            .chain(overrides.iter().copied())
            .next()
            .ok_or_else(|| SpiceError::circuit("missing candidate override"))?;
        let topology = self.topology_id()?;
        let ordinal = target.device();
        let original = self
            .devices
            .get(ordinal)
            .ok_or_else(|| SpiceError::circuit("stale candidate device override"))?;
        let device = match target {
            CandidateNumericOverride::Resistor(value) => {
                if value.topology() != topology {
                    return Err(SpiceError::circuit("foreign or stale resistor override"));
                }
                if original.terminals().len() != 2 || original.resistor_metadata().is_none() {
                    return Err(SpiceError::circuit(format!(
                        "resistor override targets {}, which is not a two-terminal resistor",
                        original.name()
                    )));
                }
                original.with_instance_parameter("resistance", value.supplied(), context)?
            }
            CandidateNumericOverride::Instance(value) => {
                if value.topology() != topology {
                    return Err(SpiceError::circuit("foreign or stale instance override"));
                }
                original.with_bound_parameter(value.parameter(), value.value(), context)?
            }
        };
        #[cfg(test)]
        preparation_metrics::record(|counts| counts.setters += 1);
        let replacements = vec![(ordinal, device)];
        self.validate_replacements(&replacements)?;
        Ok(replacements)
    }

    fn replacements_with_overrides(
        &self,
        context: &ModelContext,
        overrides: &[CandidateNumericOverride],
    ) -> SpiceResult<Vec<(usize, Box<dyn Device>)>> {
        let count = context.resistor_overrides.iter().flatten().count()
            + context.instance_overrides.iter().flatten().count()
            + overrides.len();
        if count == 0 {
            return Ok(Vec::new());
        }
        if count == 1 {
            return self.single_candidate_replacement(context, overrides);
        }

        let targets = assemble_numeric_targets(context, overrides, count)?;
        let topology = self.topology_id()?;
        let mut replacements: Vec<(usize, Box<dyn Device>)> = Vec::new();
        let mut device_slots = BTreeMap::<usize, usize>::new();
        let mut resistor_sites = BTreeSet::new();
        let mut instance_sites = BTreeSet::new();

        for target in targets {
            let ordinal = target.device();
            let original = self
                .devices
                .get(ordinal)
                .ok_or_else(|| SpiceError::circuit("stale candidate device override"))?;
            match target {
                CandidateNumericOverride::Resistor(value) => {
                    if value.topology() != topology {
                        return Err(SpiceError::circuit("foreign or stale resistor override"));
                    }
                    if original.terminals().len() != 2 || original.resistor_metadata().is_none() {
                        return Err(SpiceError::circuit(format!(
                            "resistor override targets {}, which is not a two-terminal resistor",
                            original.name()
                        )));
                    }
                    if !resistor_sites.insert(ordinal) {
                        return Err(SpiceError::circuit("duplicate resistor override"));
                    }
                    if device_slots.contains_key(&ordinal) {
                        return Err(SpiceError::circuit(format!(
                            "{} has both a resistor and an instance parameter override",
                            original.name()
                        )));
                    }
                    #[cfg(test)]
                    preparation_metrics::record(|counts| counts.setters += 1);
                    device_slots.insert(ordinal, replacements.len());
                    replacements.push((
                        ordinal,
                        original.with_instance_parameter(
                            "resistance",
                            value.supplied(),
                            context,
                        )?,
                    ));
                }
                CandidateNumericOverride::Instance(value) => {
                    if value.topology() != topology {
                        return Err(SpiceError::circuit("foreign or stale instance override"));
                    }
                    if resistor_sites.contains(&ordinal) {
                        return Err(SpiceError::circuit(format!(
                            "{} has both a resistor and an instance parameter override",
                            original.name()
                        )));
                    }
                    if !instance_sites.insert((ordinal, value.parameter())) {
                        return Err(SpiceError::circuit("duplicate instance parameter override"));
                    }
                    #[cfg(test)]
                    preparation_metrics::record(|counts| counts.setters += 1);
                    if let Some(index) = device_slots.get(&ordinal).copied() {
                        let previous = replacements[index].1.as_ref();
                        replacements[index].1 = previous.with_bound_parameter(
                            value.parameter(),
                            value.value(),
                            context,
                        )?;
                    } else {
                        device_slots.insert(ordinal, replacements.len());
                        replacements.push((
                            ordinal,
                            original.with_bound_parameter(
                                value.parameter(),
                                value.value(),
                                context,
                            )?,
                        ));
                    }
                }
            }
        }
        self.validate_replacements(&replacements)?;
        replacements.sort_unstable_by_key(|(index, _)| *index);
        Ok(replacements)
    }

    fn validate_replacements(&self, replacements: &[(usize, Box<dyn Device>)]) -> SpiceResult<()> {
        for (index, replacement) in replacements {
            let original = &self.devices[*index];
            if original.name() != replacement.name()
                || original.c_device_type() != replacement.c_device_type()
                || original.designator() != replacement.designator()
                || original.terminals() != replacement.terminals()
                || original.branch_currents() != replacement.branch_currents()
                || (0..original.branch_currents()).any(|row| {
                    original.branch_convergence(row) != replacement.branch_convergence(row)
                })
                || original.state_count() != replacement.state_count()
                || original.findable_branch() != replacement.findable_branch()
                || original.expression_current_branch() != replacement.expression_current_branch()
                || !original
                    .controlling_sources()
                    .iter()
                    .map(|c| &c.name)
                    .eq(replacement.controlling_sources().iter().map(|c| &c.name))
                || original.rf_port().map(|p| (p.number, p.internal))
                    != replacement.rf_port().map(|p| (p.number, p.internal))
                || original.mutual_coupling().map(|c| c.inductors)
                    != replacement.mutual_coupling().map(|c| c.inductors)
                || original.delay_line().is_some() != replacement.delay_line().is_some()
            {
                return Err(SpiceError::circuit(format!(
                    "{}: parameter override changed finalized topology",
                    original.name()
                )));
            }
        }
        Ok(())
    }

    fn check_matrix_pattern(&self, matrix: &SparseMatrix) -> SpiceResult<()> {
        if let Some((bound, _)) = matrix.overflow()
            && !self
                .runtime
                .pattern
                .borrow()
                .as_ref()
                .is_some_and(|pattern| pattern.same_namespace(bound))
        {
            return Err(SpiceError::circuit(
                "matrix belongs to a different finalized pattern",
            ));
        }
        Ok(())
    }

    /// Loads every device for one trial into `matrix`, `rhs` and `trial`.
    ///
    /// The accepted history is read-only here. Resistors and instance
    /// parameters named by the model context's overrides stamp their per-point
    /// replacement; no device is modified. On error, `matrix`, `rhs` and
    /// `trial` hold a partial load and must be discarded; nothing the circuit
    /// or history owns has changed.
    ///
    /// # Errors
    ///
    /// Stale numbering, mismatched dimensions/nonfinite solution, device
    /// failures, or an invalid resistor or instance-parameter override.
    pub fn load(
        &self,
        request: &LoadRequest<'_>,
        matrix: &mut SparseMatrix,
        rhs: &mut Vector,
        trial: &mut TrialState,
    ) -> SpiceResult<()> {
        self.prepare_run().load(request, matrix, rhs, trial)
    }

    fn load_prepared(
        &self,
        run: &PreparedRun<'_>,
        request: &LoadRequest<'_>,
        matrix: &mut SparseMatrix,
        rhs: &mut Vector,
        trial: &mut TrialState,
        ltra: Option<&mut LtraWorkspace>,
    ) -> SpiceResult<()> {
        self.validate_history(request.history)?;
        request.history.check_trial(trial)?;
        self.check_matrix_pattern(matrix)?;
        let n = self.unknown_count();
        if matrix.rows() != n
            || matrix.cols() != n
            || rhs.len() != n
            || request.solution.len() != n
            || !request.solution.is_finite()
            || request.history.len() != self.bindings.state_len
            || trial.values().len() != self.bindings.state_len
        {
            return Err(SpiceError::circuit(
                "load dimensions do not match the circuit numbering",
            ));
        }
        let point = run.point(request.model_context)?;
        matrix.set_pivot_tolerances(request.model_context.pivot);
        matrix.set_unknown_names(Some(self.unknown_names()));
        let mutual = point.mutual(self)?;
        let mut voltage_source_value = 0.;
        let unlent = matrix.lend_fixed(self.runtime.fixed_stamps.take(), self.devices.len())?;
        let stamped = self.stamp_devices(
            request,
            matrix,
            rhs,
            trial,
            &point,
            &mutual,
            &mut voltage_source_value,
            ltra,
        );
        self.runtime
            .fixed_stamps
            .replace(unlent.unwrap_or_else(|| matrix.return_fixed()));
        stamped?;
        self.learn_pattern(matrix)?;
        matrix.record_trace()?;
        Ok(())
    }

    /// [`Circuit::load`]'s device loop: each device stamps in load order.
    #[expect(
        clippy::too_many_arguments,
        reason = "keeps the positional interface explicit at the reference solver or device boundary"
    )]
    fn stamp_devices(
        &self,
        request: &LoadRequest<'_>,
        matrix: &mut SparseMatrix,
        rhs: &mut Vector,
        trial: &mut TrialState,
        point: &PreparedPoint,
        mutual: &[Vec<MutualTerm>],
        voltage_source_value: &mut Real,
        mut ltra: Option<&mut LtraWorkspace>,
    ) -> SpiceResult<()> {
        let mut batches = point.batches(self)?;
        batches.begin();
        for (position, index) in self.load_order().enumerate() {
            let device = point.device(self, index);
            let states = request.history.device_of(trial, index)?.with_ltra(
                ltra.as_deref_mut(),
                request.history,
                index,
            );
            matrix.begin_device(position)?;
            let mut context = StampContext {
                matrix: &mut *matrix,
                rhs: StampRhs::dense(rhs),
                unknowns: &self.bindings.unknowns,
                nodes: &self.nodes,
                solution: request.solution,
                temperature: request.model_context.temperature,
                nominal_temperature: request.model_context.nominal_temperature,
                gmin: request.model_context.gmin,
                reltol: request.model_context.reltol,
                frequency: request.model_context.frequency,
                mode: request.mode,
                branches: self.bindings.branch_rows[index].clone(),
                controls: self.controls(index)?,
                mutual: mutual.get(index).map_or(&[], Vec::as_slice),
                integration: request.integration,
                states,
                forcing: request.forcing,
                source_factor: request.model_context.source_factor,
                voltage_source_value: &mut *voltage_source_value,
            };
            if let Some(output) = batches.point(index, request.solution)? {
                device.stamp_prepared(&mut context, &output)?;
            } else {
                device.stamp(&mut context)?;
            }
        }
        Ok(())
    }

    /// A zeroed MNA matrix for [`Circuit::load`]: bound to the learned element
    /// pattern (stamps accumulate by element, no triplet list or sort), or a
    /// plain triplet matrix before the first load.
    ///
    /// # Errors
    /// Bound coefficient storage cannot be allocated.
    pub fn new_matrix(&self) -> SpiceResult<SparseMatrix> {
        self.check_numbering()?;
        let n = self.unknown_count();
        match &*self.runtime.pattern.borrow() {
            Some(pattern) if pattern.order() == n => SparseMatrix::bound(pattern.clone()),
            _ => Ok(SparseMatrix::new(n, n)),
        }
    }

    // Only explicit cross-analysis inheritance compares structure. Finalization
    // and loads never duplicate or rescan a layout to establish its identity.
    fn compatible_topology(&self, previous: &Self) -> bool {
        let a = &self.bindings;
        let b = &previous.bindings;
        if a.identity.is_none() || b.identity.is_none() {
            return false;
        }
        a.identity == b.identity
            || (a.unknowns == b.unknowns
                && a.order == b.order
                && a.branch_rows == b.branch_rows
                && a.control_rows == b.control_rows
                && a.state_rows == b.state_rows
                && self.nodes.nodes() == previous.nodes.nodes()
                && self.devices.len() == previous.devices.len()
                && self
                    .devices
                    .iter()
                    .zip(&previous.devices)
                    .all(|(left, right)| {
                        left.origin == right.origin
                            && left.name() == right.name()
                            && left.as_ref().type_id() == right.as_ref().type_id()
                            && left.c_device_type() == right.c_device_type()
                            && left.designator() == right.designator()
                            && left.terminals() == right.terminals()
                            && left.findable_branch() == right.findable_branch()
                            && left.rf_port().map(|port| (port.number, port.internal))
                                == right.rf_port().map(|port| (port.number, port.internal))
                            && left.mutual_coupling().map(|coupling| coupling.inductors)
                                == right.mutual_coupling().map(|coupling| coupling.inductors)
                            && left.delay_line().is_some() == right.delay_line().is_some()
                    }))
    }

    /// Takes over `previous`'s matrix (C's one `CKTmatrix`, with its pivot
    /// order and reorder flags), carried state vector (C's `CKTstate0`) and
    /// final transient load snapshot (instance fields such as MOS junction
    /// capacitances) for a freshly elaborated copy of the same circuit, as C runs every analysis
    /// of a deck on the matrix and state vectors set up once.
    /// The matrix is taken only when it starts with exactly the elements
    /// this circuit has allocated (the rule `finalize` applies on
    /// re-finalization); otherwise this circuit keeps its own.
    pub fn inherit_from(&mut self, previous: &Self) {
        self.runtime.fixed_stamps.get_mut().forget();
        self.runtime.completed.clear();
        self.runtime.final_transient.get_mut().take();
        self.runtime.carried_state.get_mut().take();
        *self.runtime.initial_revision.get_mut() = Arc::new(());
        let Some(topology) = self.bindings.identity else {
            return;
        };
        if !self.compatible_topology(previous) {
            return;
        }
        self.runtime
            .completed
            .clone_from(&previous.runtime.completed);
        let inherit = |snapshot: &StateSnapshot| StateSnapshot {
            values: Arc::clone(&snapshot.values),
            topology,
        };
        *self.runtime.final_transient.get_mut() = previous
            .runtime
            .final_transient
            .borrow()
            .as_ref()
            .filter(|state| previous.snapshot_matches(state))
            .map(inherit);
        *self.runtime.carried_state.get_mut() = previous
            .runtime
            .carried_state
            .borrow()
            .as_ref()
            .filter(|state| previous.snapshot_matches(state))
            .map(inherit);
        let Some(inherited) = previous.runtime.pattern.borrow().clone() else {
            return;
        };
        let mut slot = self.runtime.pattern.borrow_mut();
        let compatible = slot.as_ref().is_none_or(|current| {
            inherited.order() == current.order()
                && inherited.len() >= current.len()
                && inherited
                    .positions()
                    .take(current.len())
                    .eq(current.positions())
        });
        if compatible {
            *slot = Some(inherited);
        }
    }

    /// Records that `kind` has run on this circuit; [`Self::inherit_from`]
    /// carries the record to the next analysis's circuit.
    pub fn note_completed(&mut self, kind: crate::primitives::AnalysisKind) {
        self.runtime.completed.push(kind);
    }

    /// The analyses already run on this circuit (and the circuits it
    /// inherited from), in order.
    #[must_use = "the completed analysis sequence identifies which results are available"]
    pub fn completed_analyses(&self) -> &[crate::primitives::AnalysisKind] {
        &self.runtime.completed
    }

    /// The circuit matrix's element pattern (C's one `CKTmatrix`), once
    /// `finalize` or a load has created it for the present numbering.
    #[must_use = "the returned pattern determines the reusable sparse matrix structure"]
    pub fn matrix_pattern(&self) -> Option<std::sync::Arc<crate::maths::Pattern>> {
        self.bindings.identity.as_ref()?;
        self.runtime
            .pattern
            .borrow()
            .as_ref()
            .filter(|pattern| pattern.order() == self.unknown_count())
            .cloned()
    }

    /// Records the positions a load stamped: the first load defines the
    /// pattern in stamp order; positions outside it are appended (which, as a
    /// new SPARSE element does, forces the next factorization to reorder).
    fn learn_pattern(&self, matrix: &SparseMatrix) -> SpiceResult<()> {
        let n = self.unknown_count();
        let mut slot = self.runtime.pattern.borrow_mut();
        let current = slot.as_ref().filter(|pattern| pattern.order() == n);
        let next = if let Some((bound, mut extra)) = matrix.overflow() {
            let mut extra = extra.by_ref().peekable();
            if extra.peek().is_none() {
                return Ok(());
            }
            match current {
                Some(pattern) if std::sync::Arc::ptr_eq(pattern, bound) => {
                    pattern.extended(extra)?
                }
                _ => return Ok(()),
            }
        } else {
            let positions = matrix.triplets().iter().map(|t| (t.row, t.col));
            match current {
                None => crate::maths::Pattern::new(n, positions)?,
                Some(pattern) => {
                    let mut missing = positions
                        .filter(|&(row, col)| pattern.find(row, col).is_none())
                        .peekable();
                    if missing.peek().is_none() {
                        return Ok(());
                    }
                    pattern.extended(missing)?
                }
            }
        };
        *slot = Some(std::sync::Arc::new(next));
        Ok(())
    }

    fn validate_accept_point(&self, solution: &Vector, time: Option<Real>) -> SpiceResult<()> {
        self.check_numbering()?;
        if solution.len() != self.unknown_count() || !solution.is_finite() {
            return Err(SpiceError::circuit(
                "accepted solution does not match the circuit numbering or is nonfinite",
            ));
        }
        if time.is_some_and(|t| !t.is_finite()) {
            return Err(SpiceError::circuit("accepted time is nonfinite"));
        }
        Ok(())
    }

    fn run_accept_hooks(
        &self,
        solution: &Vector,
        time: Option<Real>,
        trial: Option<&TrialState>,
    ) -> SpiceResult<()> {
        // All fallible device-specific checks precede any notifications.
        // The caller has validated the trial against the finalized state layout.
        let context = |index: usize| AcceptContext {
            solution,
            time,
            branches: &self.bindings.branch_rows[index],
            states: trial.map(|trial| &trial.values()[self.bindings.state_rows[index].clone()]),
        };
        for (index, device) in self.devices.iter().enumerate() {
            device.validate_accept(&context(index))?;
        }
        for (index, device) in self.devices.iter().enumerate() {
            device.accept(&context(index));
        }
        Ok(())
    }

    /// Accepts a point whose trial state is committed into `history`.
    ///
    /// Validation and every [`Device::accept`] hook run first; the history
    /// changes only if all of them succeed, so a failure is atomic.
    ///
    /// # Errors
    ///
    /// Invalid solution/time, an uncommittable trial, or a hook failure.
    pub fn accept_point(
        &self,
        solution: &Vector,
        time: Option<Real>,
        history: &mut StateHistory,
        trial: TrialState,
    ) -> SpiceResult<()> {
        self.validate_history(history)?;
        self.validate_accept_point(solution, time)?;
        history
            .accept_transaction(
                trial,
                HistoryPublication::Rotate,
                |_, _| Ok(Vec::new()),
                |trial| self.run_accept_hooks(solution, time, Some(trial)),
            )
            .map(drop)
    }

    /// Accepts a companion-transient point: [`Self::accept_point`] plus the
    /// delay histories of transmission lines ([`crate::devices::delay`]).
    ///
    /// With `delay.start` (the `t = 0` point), the initial state fills the
    /// integration history and every delay line initializes
    /// its history ([`crate::devices::delay::DelayLine::start`]); otherwise it
    /// records the accepted sample ([`crate::devices::delay::DelayLine::accept`]).
    /// Every update is computed and validated first, then the hooks run, the
    /// fixed state is committed and the delay updates are applied, so a
    /// failure leaves the history unchanged. Returns the breakpoints the
    /// devices request (C `CKTsetBreak` from `DEVaccept`), in C's load order;
    /// a circuit without delay lines behaves exactly as
    /// [`Self::accept_point`] and returns none.
    ///
    /// # Errors
    ///
    /// As [`Self::accept_point`], and delay-line failures or invalid updates.
    pub fn accept_transient_point(
        &self,
        solution: &Vector,
        time: Real,
        history: &mut StateHistory,
        trial: TrialState,
        delay: &DelayAcceptance,
    ) -> SpiceResult<Vec<Real>> {
        self.validate_history(history)?;
        self.validate_accept_point(solution, Some(time))?;
        let publication = if delay.start {
            if time != 0. {
                return Err(SpiceError::circuit(
                    "transient initial point must be at t = 0",
                ));
            }
            HistoryPublication::Initialize
        } else {
            if history.accepted(1).is_none() {
                return Err(SpiceError::circuit(
                    "transient step requires initial accepted history",
                ));
            }
            HistoryPublication::Rotate
        };
        history.accept_transaction(
            trial,
            publication,
            |history, trial| self.delay_updates(solution, time, history, trial, delay),
            |trial| self.run_accept_hooks(solution, Some(time), Some(trial)),
        )
    }

    /// Compute delay updates in C's load order without modifying history.
    fn delay_updates(
        &self,
        solution: &Vector,
        time: Real,
        history: &StateHistory,
        trial: &TrialState,
        delay: &DelayAcceptance,
    ) -> SpiceResult<Vec<(usize, crate::devices::delay::DelayUpdate)>> {
        let mut updates = Vec::new();
        // TRAaccept walks C's model/instance lists: the load order.
        for index in self.load_order() {
            let device = &self.devices[index];
            let Some(line) = device.delay_line() else {
                continue;
            };
            let samples = history.delay(index).ok_or_else(|| {
                SpiceError::circuit(format!(
                    "{}: the state history has no delay history (use Circuit::state_history)",
                    device.name()
                ))
            })?;
            let context = crate::devices::delay::DelayContext {
                solution,
                unknowns: &self.bindings.unknowns,
                time,
                steps: delay.steps,
                min_break: delay.min_break,
                reltol: delay.reltol,
                abstol: delay.abstol,
                initial_conditions: trial.initial_conditions(),
                branches: self.branch_rows(index).unwrap_or(0..0),
            };
            let update = if delay.start {
                let update = line.start(&context)?;
                if !update.is_reset() {
                    return Err(SpiceError::circuit(format!(
                        "{}: a delay-line start must reset its history",
                        device.name()
                    )));
                }
                update
            } else {
                if samples.is_empty() {
                    return Err(SpiceError::circuit(format!(
                        "{}: delay history used before the transient start",
                        device.name()
                    )));
                }
                line.accept(samples, &context)?
            };
            updates.push((index, update));
        }
        Ok(updates)
    }

    /// The tightest step bound of every delay line for a converged transient
    /// trial at `time` (C `DEVtrunc` of `TRA`), or `None` without delay lines.
    ///
    /// # Errors
    ///
    /// A missing delay history or a device failure.
    pub fn delay_timestep_limit(
        &self,
        history: &StateHistory,
        solution: &Vector,
        time: Real,
        delay: &DelayAcceptance,
    ) -> SpiceResult<Option<Real>> {
        self.validate_history(history)?;
        let mut limit: Option<Real> = None;
        for (index, device) in self.devices.iter().enumerate() {
            let Some(line) = device.delay_line() else {
                continue;
            };
            let samples = history
                .delay(index)
                .filter(|h| !h.is_empty())
                .ok_or_else(|| {
                    SpiceError::circuit(format!(
                        "{}: delay history used before the transient start",
                        device.name()
                    ))
                })?;
            let context = crate::devices::delay::DelayContext {
                solution,
                unknowns: &self.bindings.unknowns,
                time,
                steps: delay.steps,
                min_break: delay.min_break,
                reltol: delay.reltol,
                abstol: delay.abstol,
                initial_conditions: false,
                branches: self.branch_rows(index).unwrap_or(0..0),
            };
            if let Some(bound) = line.timestep_limit(samples, &context)? {
                if bound.is_nan() {
                    return Err(SpiceError::circuit(format!(
                        "{}: NaN timestep limit",
                        device.name()
                    )));
                }
                limit = Some(limit.map_or(bound, |l| l.min(bound)));
            }
        }
        Ok(limit)
    }

    /// Accepts a point of an analysis that tracks no device state (linear
    /// DC/AC). Hooks see `states: None`.
    ///
    /// # Errors
    ///
    /// Invalid solution/time or a hook failure.
    pub fn accept_solution(&self, solution: &Vector, time: Option<Real>) -> SpiceResult<()> {
        self.validate_accept_point(solution, time)?;
        self.run_accept_hooks(solution, time, None)
    }

    /// Assemble at the default circuit/nominal temperature (27 Celsius).
    /// # Errors
    /// Invalid topology, model derivation or unsupported equations.
    pub fn linear_system(&mut self) -> SpiceResult<crate::devices::linear::LinearSystem> {
        self.linear_system_with_context(&crate::devices::models::ModelContext::default())
    }

    /// Assembles immutable equations at explicit circuit/nominal temperatures.
    /// Model recipes are reevaluated without cumulative value/state changes.
    /// # Errors
    /// Invalid context/topology, model arithmetic or unsupported equations.
    pub fn linear_system_with_context(
        &mut self,
        context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<crate::devices::linear::LinearSystem> {
        context.validate(&MODEL_CONTEXT_LOCATION)?;
        self.finalize()?;
        let run = self.prepare_run();
        self.assemble_linear_devices(&run, context, false)
    }

    fn assemble_linear_devices(
        &self,
        run: &PreparedRun<'_>,
        context: &ModelContext,
        skip_nonlinear: bool,
    ) -> SpiceResult<crate::devices::linear::LinearSystem> {
        let point = run.point(context)?;
        let mutual = point.mutual(self)?;
        let mut system = crate::devices::linear::LinearSystem::new(self.unknown_count());
        // Source collection size comes from finalized source identities.
        // Pre-size once per assembly, preserving the device load order.
        let source_count = self
            .devices
            .iter()
            .filter(|device| matches!(device.designator(), 'v' | 'i'))
            .count();
        system.sources.try_reserve(source_count).map_err(|error| {
            SpiceError::circuit(format!("independent source collection allocation: {error}"))
        })?;
        system.a.set_pivot_tolerances(context.pivot);
        system.a.set_unknown_names(Some(self.unknown_names()));
        for index in self.load_order() {
            let device = point.device(self, index);
            if skip_nonlinear && device.is_nonlinear() {
                continue;
            }
            let range = &self.bindings.branch_rows[index];
            device.assemble_linear(&mut crate::devices::linear::LinearContext {
                model_context: context,
                system: &mut system,
                unknowns: &self.bindings.unknowns,
                branch: (!range.is_empty()).then_some(range.start),
                controls: self.controls(index)?,
                mutual: mutual.get(index).map_or(&[], Vec::as_slice),
                states: None,
                pole_zero_statements: None,
                pole_zero_setups: 0,
            })?;
        }
        system.a.fold_duplicates();
        // E stays unfolded: `CKTacLoad` scales every device term by
        // `omega` before adding it (`xc = cap * omega`), so the complex
        // assembly must scale each contribution, not their sum.
        Ok(system)
    }

    /// Assemble `G(bias)` and `dQ/dx(bias)` for nonlinear small-signal analysis.
    /// This is deliberately separate from immutable linear/BDF assembly.
    /// # Errors
    /// Invalid bias/context, stale numbering, or unsupported device physics.
    pub fn small_signal_system(
        &self,
        context: &ModelContext,
        bias: &Vector,
    ) -> SpiceResult<crate::devices::linear::LinearSystem> {
        self.small_signal_system_at(context, bias, None)
    }

    /// [`Self::small_signal_system`] with a full state vector (C's
    /// `CKTstate0` at `MODEINITSMSIG`) that devices with discrete state
    /// (switches) read through [`crate::devices::LinearContext::states`].
    /// # Errors
    /// As [`Self::small_signal_system`], or a state of the wrong length.
    pub fn small_signal_system_at(
        &self,
        context: &ModelContext,
        bias: &Vector,
        state: Option<&StateSnapshot>,
    ) -> SpiceResult<crate::devices::linear::LinearSystem> {
        self.prepare_run()
            .small_signal_system_at(context, bias, state)
    }

    /// The pole-zero pencil `A + s E` at a solved bias point: every device's
    /// [`Device::assemble_pole_zero`] (C `CKTpzLoad` without the drive and
    /// column operations, which belong to the analysis). The same bias and
    /// state rules as [`Self::small_signal_system_at`] apply; no source
    /// forcing is assembled.
    /// # Errors
    /// As [`Self::small_signal_system_at`], and an explicit error for any
    /// device without a pole-zero load.
    pub fn pole_zero_system_at(
        &self,
        context: &ModelContext,
        bias: &Vector,
        state: Option<&StateSnapshot>,
    ) -> SpiceResult<crate::devices::linear::LinearSystem> {
        self.bias_linearized_system(&self.prepare_run(), context, bias, state, 1, None)
    }

    /// The pole-zero load as `CKTpzLoad` adds it: every device's
    /// [`Device::assemble_pole_zero`] contributions in load order, one
    /// [`crate::maths::pencil::PencilTerm`] per device and element. A
    /// device's static stamps at one element are summed into its `g` and
    /// its dynamic stamps into its `c` (one [`crate::maths::DynamicGroup`]
    /// as `scale * (x1 + x2 + ...)`), as the C loads add `g + c * s`
    /// (`diopzld.c` adds `geq + gspr + xceq * s->real`).
    /// # Errors
    /// As [`Self::pole_zero_system_at`].
    pub fn pole_zero_terms_at(
        &self,
        context: &ModelContext,
        bias: &Vector,
        state: Option<&StateSnapshot>,
        setups: u32,
    ) -> SpiceResult<Vec<crate::maths::pencil::PencilTerm>> {
        let mut ranges = Vec::new();
        let system = self.bias_linearized_system(
            &self.prepare_run(),
            context,
            bias,
            state,
            setups.max(1),
            Some(&mut ranges),
        )?;
        let conductance_stamps = system.a.triplets();
        system.e_groups.validate(&system.e)?;
        let capacitance_stamps = system.e.triplets();
        let mut ordered_groups = system.e_groups.iter().peekable();
        let mut terms = Vec::new();
        for (conductance_range, capacitance_range, statements) in ranges {
            if !statements.is_empty() {
                // The device recorded its C statements in order.
                terms.extend(statements);
                continue;
            }
            let first = terms.len();
            let slot = |terms: &mut Vec<crate::maths::pencil::PencilTerm>, row, col| {
                let existing =
                    terms[first..]
                        .iter()
                        .position(|stamp: &crate::maths::pencil::PencilTerm| {
                            stamp.row == row && stamp.col == col
                        });
                existing.map_or_else(
                    || {
                        terms.push(crate::maths::pencil::PencilTerm {
                            row,
                            col,
                            conductance: None,
                            capacitance: None,
                        });
                        terms.len() - 1
                    },
                    |slot_index| first + slot_index,
                )
            };
            for stamp in &conductance_stamps[conductance_range] {
                let slot_index = slot(&mut terms, stamp.row, stamp.col);
                let conductance = &mut terms[slot_index].conductance;
                *conductance =
                    Some(conductance.map_or(stamp.value, |conductance| conductance + stamp.value));
            }
            while ordered_groups
                .peek()
                .is_some_and(|group| group.range.end <= capacitance_range.start)
            {
                ordered_groups.next();
            }
            let mut index = capacitance_range.start;
            while index < capacitance_range.end {
                let stamp = &capacitance_stamps[index];
                let (value, next) = match ordered_groups.peek() {
                    Some(group) if group.range.start == index => {
                        if group.range.end > capacitance_range.end {
                            return Err(SpiceError::circuit(
                                "dynamic group crosses pole-zero device boundary",
                            ));
                        }
                        if group.prescaled {
                            return Err(SpiceError::circuit(
                                "prescaled AC dynamic group is not an s coefficient",
                            ));
                        }
                        let mut sum = group.terms.first().copied().unwrap_or(0.);
                        for term in group.terms.iter().skip(1) {
                            sum += term;
                        }
                        let result = (group.scale * sum / group.divisor, group.range.end);
                        ordered_groups.next();
                        result
                    }
                    Some(group) if group.range.start < index => {
                        return Err(SpiceError::circuit(
                            "dynamic group crosses pole-zero stamp boundary",
                        ));
                    }
                    _ => (stamp.value, index + 1),
                };
                let slot_index = slot(&mut terms, stamp.row, stamp.col);
                let capacitance = &mut terms[slot_index].capacitance;
                *capacitance = Some(capacitance.map_or(value, |capacitance| capacitance + value));
                index = next;
            }
        }
        Ok(terms)
    }

    fn bias_linearized_system(
        &self,
        run: &PreparedRun<'_>,
        context: &ModelContext,
        bias: &Vector,
        state: Option<&StateSnapshot>,
        pole_zero_setups: u32,
        mut ranges: Option<&mut Vec<PoleZeroRange>>,
    ) -> SpiceResult<crate::devices::linear::LinearSystem> {
        #[cfg(test)]
        preparation_metrics::record(|counts| counts.small_signal_assemblies += 1);
        let state = self.snapshot_values(state)?;
        context.validate(&MODEL_CONTEXT_LOCATION)?;
        if bias.len() != self.unknown_count() || !bias.is_finite() {
            return Err(SpiceError::circuit(
                "invalid small-signal bias dimensions/values",
            ));
        }
        let point = run.point(context)?;
        let mutual = point.mutual(self)?;
        let mut system = crate::devices::linear::LinearSystem::new(self.unknown_count());
        // Source collection size comes from finalized source identities.
        // Pre-size once per assembly, preserving the device load order.
        let source_count = self
            .devices
            .iter()
            .filter(|device| matches!(device.designator(), 'v' | 'i'))
            .count();
        system.sources.try_reserve(source_count).map_err(|error| {
            SpiceError::circuit(format!("independent source collection allocation: {error}"))
        })?;
        system.a.set_pivot_tolerances(context.pivot);
        system.a.set_unknown_names(Some(self.unknown_names()));
        let mut batches = point.batches(self)?;
        batches.begin();
        for index in self.load_order() {
            let device = point.device(self, index);
            let range = &self.bindings.branch_rows[index];
            let mut statements = Vec::new();
            let collecting = ranges.is_some();
            let mut linear = crate::devices::linear::LinearContext {
                model_context: context,
                system: &mut system,
                unknowns: &self.bindings.unknowns,
                branch: (!range.is_empty()).then_some(range.start),
                controls: self.controls(index)?,
                mutual: mutual.get(index).map_or(&[], Vec::as_slice),
                states: match state {
                    Some(state) => Some(
                        state
                            .get(self.bindings.state_rows[index].clone())
                            .ok_or_else(|| SpiceError::circuit("bias state is too short"))?,
                    ),
                    None => None,
                },
                pole_zero_statements: collecting.then_some(&mut statements),
                pole_zero_setups,
            };
            let start = (
                linear.system.a.triplets().len(),
                linear.system.e.triplets().len(),
            );
            if let Some(output) = batches.point(index, bias)? {
                device.assemble_prepared(&mut linear, &output)?;
            } else if pole_zero_setups > 0 {
                device.assemble_pole_zero(&mut linear, bias)?;
            } else {
                device.assemble_small_signal(&mut linear, bias)?;
            }
            if let Some(ranges) = ranges.as_deref_mut() {
                ranges.push((
                    start.0..system.a.triplets().len(),
                    start.1..system.e.triplets().len(),
                    statements,
                ));
            }
        }
        if ranges.is_some() {
            // The caller reads the unfolded stamps.
            return Ok(system);
        }
        system.a.fold_duplicates();
        // E stays unfolded: `CKTacLoad` scales every device term by
        // `omega` before adding it (`xc = cap * omega`), so the complex
        // assembly must scale each contribution, not their sum.
        Ok(system)
    }

    /// Elaborate one AST instance atomically with explicit model context.
    /// Model lookup/schema validation happens before a factory; unavailable
    /// model-backed/nonlinear factories remain explicit errors. As with
    /// [`Circuit::add_device`], finalize after successful additions to renumber.
    ///
    /// # Errors
    /// Duplicate names, missing/wrong-family models, invalid setters/selectors
    /// or unavailable factories. Nodes, devices and branch rows are unchanged
    /// on failure, including construction/validation failures.
    pub fn add_instance(
        &mut self,
        instance: &crate::netlist::ast::DeviceInstance,
        models: &crate::devices::models::ModelResolver<'_>,
        context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<()> {
        self.add_instances(std::slice::from_ref(instance), models, context)
    }

    /// Elaborate several AST instances atomically, in order.
    ///
    /// Builtin factories extend the node table under an append-only
    /// transaction. Devices and their origins are staged as pairs; the new
    /// node suffix and all devices are published only if the entire batch
    /// succeeds. A failed batch does not change existing nodes, devices or
    /// numbering. Finalize after adding instances.
    ///
    /// # Errors
    /// Aggregate resource limits, unavailable storage, a duplicate instance
    /// name (against the existing devices and within the batch), or whatever
    /// [`crate::devices::factory::instantiate_with_models`] reports.
    pub fn add_instances(
        &mut self,
        instances: &[crate::netlist::ast::DeviceInstance],
        models: &crate::devices::models::ModelResolver<'_>,
        context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<()> {
        let existing = self.elaboration_size()?;
        let limits = self.elaboration_limits;
        let instances = crate::devices::behavioural::dynamic::expand(
            instances,
            &self.nodes,
            models,
            self.devices.iter().map(|device| device.name()),
            limits.devices.saturating_sub(existing.devices),
        )?;
        let instances = instances.as_ref();
        // All generated plans are parsed and budgeted before any of them
        // create a ladder. Ordinary factories keep their existing order.
        let plan =
            PlannedBatch::prepare(instances, models, context, &self.nodes, existing, limits)?;
        let seen: BTreeSet<String> = self
            .devices
            .iter()
            .map(|device| device.name().to_lowercase())
            .collect();
        let devices = &mut self.devices;
        let staged = self.nodes.transaction(|nodes| {
            let staged =
                Self::elaborate_batch(plan, models, context, nodes, existing, limits, seen)?;
            // The node transaction commits only after the final device and
            // provenance vector can hold the entire staged suffix.
            devices.try_reserve(staged.len()).map_err(|error| {
                resource_error(
                    instances.last().map(|i| &i.location),
                    format!("cannot reserve final device storage: {error}"),
                )
            })?;
            Ok(staged)
        })?;
        if !staged.is_empty() {
            self.invalidate_topology();
        }
        self.devices.extend(staged);
        Ok(())
    }

    /// Build each prepared group and its provenance in deck order. All node
    /// mutations are append-only, as required by the surrounding transaction.
    fn elaborate_batch(
        plan: PlannedBatch<'_>,
        models: &crate::devices::models::ModelResolver<'_>,
        context: &ModelContext,
        nodes: &mut crate::primitives::NodeAppend<'_>,
        mut usage: ElaborationSize,
        limits: ElaborationLimits,
        mut seen: BTreeSet<String>,
    ) -> SpiceResult<Vec<ElaboratedDevice>> {
        let mark = nodes.len();
        let base = usage.devices;
        let mut staged = Vec::new();
        reserve(
            &mut staged,
            plan.devices,
            plan.instances.first().map(|i| &i.location),
        )?;
        let mut expansions = plan.expansions.into_iter().peekable();
        for (ordinal, instance) in plan.instances.iter().enumerate() {
            let expansion = expansions.next_if(|prepared| prepared.ordinal == ordinal);
            let at = Some(&instance.location);
            if !seen.insert(instance.name.to_lowercase()) {
                return Err(SpiceError::circuit(format!(
                    "duplicate instance name '{}'",
                    instance.name
                )));
            }
            let first_node = nodes.len();
            let first_device = staged.len();
            let exact = expansion.is_some();
            let expected = expansion
                .as_ref()
                .map_or(Ok(ElaborationSize::new(1, 0, 0)), |prepared| {
                    prepared.size_at_bind(instance, nodes)
                })?;
            limits.check(usage.checked_add(expected, at)?, at)?;
            match expansion.map(|prepared| prepared.kind) {
                Some(ExpansionKind::Txl(crate::devices::txl::TxlPlan::Fallback(plan_txl))) => {
                    let expanded = plan_txl.bind(instance, nodes, &plan.deck_nodes)?;
                    for device in expanded {
                        if !seen.insert(device.name().to_ascii_lowercase()) {
                            return Err(SpiceError::circuit(format!(
                                "duplicate TXL generated instance {}",
                                device.name()
                            )));
                        }
                        staged.push(ElaboratedDevice {
                            device,
                            origin: DeviceOrigin::default(),
                        });
                    }
                }
                Some(ExpansionKind::Urc(plan_urc)) => {
                    let expanded = plan_urc.bind(instance, nodes, context, &plan.deck_nodes)?;
                    let line = base + staged.len();
                    Self::stage_urc(
                        instance,
                        nodes,
                        expanded,
                        first_node,
                        line,
                        &mut seen,
                        &mut staged,
                    )?;
                }
                None => {
                    let device = crate::devices::factory::instantiate_with_models(
                        instance, nodes, models, context,
                    )?;
                    Self::stage_single(instance, nodes, device, first_node, &mut staged)?;
                }
                Some(ExpansionKind::Txl(crate::devices::txl::TxlPlan::Native(native))) => {
                    let device = native.bind(instance, nodes)?;
                    Self::stage_single(instance, nodes, device, first_node, &mut staged)?;
                }
            }
            let actual = ElaborationSize::new(
                staged.len() - first_device,
                nodes.len() - first_node,
                Self::declared_state_slots(&staged[first_device..], at)?,
            );
            if exact && actual != expected {
                return Err(SpiceError::parse(
                    instance.location.clone(),
                    format!(
                        "{}: prepared expansion size {expected:?} differs from materialized {actual:?}",
                        instance.name
                    ),
                ));
            }
            usage = usage.checked_add(actual, at)?;
            limits.check(usage, at)?;
        }
        // Factories already reject existing names. Check private nodes which
        // were created before a later parsed card could introduce its name.
        if let Some(collision) = nodes.nodes()[mark..]
            .iter()
            .find(|node| node.kind == NodeKind::Internal && plan.deck_nodes.contains(&node.name))
        {
            return Err(SpiceError::circuit(format!(
                "internal-node name collision: {}",
                collision.name
            )));
        }
        Ok(staged)
    }

    fn stage_single(
        instance: &crate::netlist::ast::DeviceInstance,
        nodes: &NodeTable,
        device: Box<dyn Device>,
        first: usize,
        staged: &mut Vec<ElaboratedDevice>,
    ) -> SpiceResult<()> {
        let mut internal_nodes = Vec::new();
        let internal = nodes.nodes()[first..]
            .iter()
            .filter(|node| node.kind == NodeKind::Internal);
        reserve(
            &mut internal_nodes,
            internal.clone().count(),
            Some(&instance.location),
        )?;
        internal_nodes.extend(internal.map(|node| node.id));
        staged.push(ElaboratedDevice {
            device,
            origin: DeviceOrigin {
                internal_nodes,
                model: instance.model.as_ref().map(|m| m.to_ascii_lowercase()),
                generated_by: None,
            },
        });
        Ok(())
    }

    // Preserve ownership and insertion order of a URC's generated devices.
    fn stage_urc(
        instance: &crate::netlist::ast::DeviceInstance,
        nodes: &NodeTable,
        expanded: Vec<Box<dyn Device>>,
        first: usize,
        line: usize,
        seen: &mut BTreeSet<String>,
        staged: &mut Vec<ElaboratedDevice>,
    ) -> SpiceResult<()> {
        let parsed_terminals: BTreeSet<_> = instance
            .nodes
            .iter()
            .filter_map(|name| nodes.get(name))
            .collect();
        let lower = instance.name.to_lowercase();
        for device in expanded {
            let is_line = device.name().eq_ignore_ascii_case(&instance.name);
            if !is_line && !seen.insert(device.name().to_lowercase()) {
                return Err(SpiceError::circuit(format!(
                    "duplicate instance name '{}' generated by URC {}",
                    device.name(),
                    instance.name
                )));
            }
            let origin = if is_line {
                // URCsetup's own ladder nodes: defer until the URC's setup.
                DeviceOrigin {
                    internal_nodes: {
                        let internal = nodes.nodes()[first..].iter().filter(|node| {
                            node.kind != NodeKind::Internal && !parsed_terminals.contains(&node.id)
                        });
                        let mut ids = Vec::new();
                        reserve(&mut ids, internal.clone().count(), Some(&instance.location))?;
                        ids.extend(internal.map(|node| node.id));
                        ids
                    },
                    model: instance.model.as_ref().map(|m| m.to_ascii_lowercase()),
                    generated_by: None,
                }
            } else {
                // The generated diode's prime node belongs to diosetup.
                let internal_nodes = device
                    .terminals()
                    .iter()
                    .copied()
                    .filter(|id| {
                        id.index() >= first
                            && !parsed_terminals.contains(id)
                            && nodes
                                .node(*id)
                                .is_some_and(|node| node.kind == NodeKind::Internal)
                    })
                    .collect();
                let suffix = match device.c_device_type() {
                    crate::devices::setup::CDeviceType::Res => "resmod",
                    crate::devices::setup::CDeviceType::Cap => "capmod",
                    _ => "diodemod",
                };
                DeviceOrigin {
                    internal_nodes,
                    model: Some(format!("{lower}#{suffix}")),
                    generated_by: Some(line),
                }
            };
            staged.push(ElaboratedDevice { device, origin });
        }
        Ok(())
    }

    /// Elaborate literal (or top-level `.param`/`{expr}`-literalized) R/C/L/V/I and
    /// bounded model-backed R/C/L at 27 Celsius.
    ///
    /// A deck with `.option` cards is rejected here: this entry point would
    /// silently apply default temperatures. Resolve the options with
    /// `crate::analysis::RunConfig` and elaborate with
    /// [`Self::from_netlist_with_context`] (or `RunConfig::circuit`).
    /// # Errors
    /// Unsupported elaboration constructs, `.option` cards, models or invalid
    /// parameters.
    pub fn from_netlist(netlist: &crate::netlist::ast::Netlist) -> SpiceResult<Self> {
        if !netlist.options.is_empty()
            || netlist
                .subcircuits
                .iter()
                .any(|sub| !sub.options.is_empty())
        {
            return Err(SpiceError::Unsupported {
                feature: ".option cards require RunConfig::from_netlist to resolve them; \
                          Circuit::from_netlist would apply default temperatures"
                    .into(),
                location: netlist.options.first().map(|card| card.location.clone()),
            });
        }
        Self::from_netlist_with_context(netlist, &crate::devices::models::ModelContext::default())
    }

    /// Elaborate with explicit validation temperatures; the immutable model recipe
    /// is retained and later assemblies use their own explicit context.
    /// `.option` cards are the caller's responsibility: the supplied context must
    /// come from resolving them (`crate::analysis::RunConfig`). Top-level
    /// `.global` cards name nodes that stay global through subcircuit expansion
    /// (see [`crate::devices::subckt`]); everything else keeps the deck's names.
    /// # Errors
    /// Invalid context, unsupported constructs/models, invalid parameters or a
    /// failing subcircuit expansion ([`crate::devices::subckt::expand_subcircuits`]).
    /// Nothing is added to the returned circuit; a failed deck yields `Err`.
    pub fn from_netlist_with_context(
        netlist: &crate::netlist::ast::Netlist,
        context: &crate::devices::models::ModelContext,
    ) -> SpiceResult<Self> {
        Self::from_netlist_with_mutual_limits(netlist, context, MutualLimits::default())
    }

    /// Elaborate with explicit K-card construction and validation limits.
    ///
    /// # Errors
    /// The same errors as [`Self::from_netlist_with_context`], including any
    /// exceeded mutual-inductor budget before allocating an oversized system.
    pub fn from_netlist_with_mutual_limits(
        netlist: &crate::netlist::ast::Netlist,
        context: &ModelContext,
        mutual_limits: MutualLimits,
    ) -> SpiceResult<Self> {
        Self::from_netlist_with_limits(
            netlist,
            context,
            mutual_limits,
            ElaborationLimits::default(),
        )
    }

    /// Elaborate with both mutual-inductor and aggregate materialized-device
    /// resource limits. The latter also bounds input subcircuit expansion.
    /// # Errors
    /// Invalid deck/context, failed factory or expansion, or exceeded resource limits.
    pub fn from_netlist_with_limits(
        netlist: &crate::netlist::ast::Netlist,
        context: &ModelContext,
        mutual_limits: MutualLimits,
        elaboration_limits: ElaborationLimits,
    ) -> SpiceResult<Self> {
        Self::from_netlist_with_scoped_limits(
            netlist,
            context,
            mutual_limits,
            elaboration_limits,
            None,
        )
    }

    /// Instantiate candidate parameters from the existing evaluated scope.
    /// The same validated lowering, expansion and factory path as ordinary
    /// elaboration is used, including model-bin and topology-dependent logic.
    pub(crate) fn from_netlist_with_scoped_limits(
        netlist: &crate::netlist::ast::Netlist,
        context: &ModelContext,
        mutual_limits: MutualLimits,
        elaboration_limits: ElaborationLimits,
        scope: Option<std::sync::Arc<crate::netlist::eval::ParamScope>>,
    ) -> SpiceResult<Self> {
        context.validate(&netlist.location)?;
        if !netlist.includes.is_empty() {
            return Err(SpiceError::Unsupported {
                feature: "includes in linear elaboration".into(),
                location: None,
            });
        }
        // The nonlinear E/G/F/H forms become B sources and generated XSPICE
        // instances first, as inpcom.c rewrites them before numparam and
        // subcircuit expansion (#79).
        let lowered = crate::netlist::behavioural::lower_nonlinear_sources(netlist)?;
        // Top-level `.param` values and `{expr}` sites are evaluated into a
        // literal copy before any factory sees them (#15).
        let elaborated = if let Some(scope) = scope {
            crate::netlist::elaborate::literalize_with(
                &lowered,
                scope,
                &mut crate::netlist::eval::EvalBudget::default(),
            )?
        } else {
            crate::netlist::elaborate::literalize(&lowered)?
        };
        let netlist = &elaborated.netlist;
        // `X` instances are expanded into a fresh device/model list before any
        // device is built, so a deck that fails to elaborate never leaves a
        // partial circuit behind (#18).
        let expanded = crate::devices::subckt::expand_subcircuits(
            netlist,
            &elaborated.scope,
            crate::devices::subckt::SubcircuitLimits {
                max_devices: elaboration_limits.devices,
                ..Default::default()
            },
        )?;
        let models = crate::devices::models::ModelResolver::new(&expanded.models)?;
        let mut circuit = Self {
            mutual_limits,
            elaboration_limits,
            ..Self::default()
        };
        // `b4set.c` reads the task's analysis list (`noiseAnalGiven`).
        let mut setup = *context;
        setup.noise_analysis = netlist
            .analyses
            .iter()
            .any(|card| card.kind == crate::primitives::AnalysisKind::Noise);
        circuit.add_instances(&expanded.devices, &models, &setup)?;
        let referenced: BTreeSet<_> = expanded
            .devices
            .iter()
            .filter_map(|instance| instance.model.as_ref())
            // A binned reference uses its whole `<name>.<n>` set (C's
            // `mark_all_binned`), not only the selected bin.
            .flat_map(|name| models.declarations_for(name))
            .map(|card| card.name.to_ascii_lowercase())
            .collect();
        if let Some(model) = netlist
            .models
            .iter()
            .find(|model| !referenced.contains(&model.name.to_ascii_lowercase()))
        {
            // First-declaration duplicate policy remains explicit; unused
            // declarations cannot silently discard unsupported physics. A root
            // model shadowed only inside a subcircuit body is unused by this
            // rule and is reported, not dropped silently.
            return Err(SpiceError::Unsupported {
                feature: "unused model declarations in scalar linear elaboration".into(),
                location: Some(model.location.clone()),
            });
        }
        circuit.finalize()?;
        // Reject an invalid inductive system (K coupling) before any analysis.
        circuit.prepared_mutual_terms(context)?;
        Ok(circuit)
    }

    /// Looks up a device by instance name, case-insensitively.
    #[must_use = "the returned device is needed to inspect the named instance"]
    pub fn device(&self, name: &str) -> Option<&dyn Device> {
        self.device_index(name)
            .map(|index| self.devices[index].as_ref())
    }

    // The finalized names are already indexed. During construction, callers
    // such as add_device still need to inspect the unfinalized device list.
    fn device_index(&self, name: &str) -> Option<usize> {
        #[cfg(test)]
        preparation_metrics::record(|counts| counts.name_lookups += 1);
        if self.bindings.identity.is_some() {
            self.bindings.names.get(name)
        } else {
            self.devices
                .iter()
                .position(|device| device.name().eq_ignore_ascii_case(name))
        }
    }
}

/// One device's stamps in a pole-zero term collection: its A and E triplet
/// ranges and any C statements it recorded in order
/// ([`crate::devices::linear::LinearContext::pole_zero_statements`]).
type PoleZeroRange = (
    std::ops::Range<usize>,
    std::ops::Range<usize>,
    Vec<crate::maths::pencil::PencilTerm>,
);

#[cfg(test)]
mod tests {
    use super::{Circuit, CircuitGraph, CircuitVertex};
    use crate::devices::traits::Device;
    use crate::primitives::{NodeId, SpiceError, SpiceResult};
    use petgraph::algo::{connected_components, has_path_connecting};
    use petgraph::graph::NodeIndex;
    use petgraph::visit::EdgeRef;

    // Compare the actual production load and observation paths, not just a
    // synthetic cache lookup. Synthetic accepted histories intentionally
    // include equal-length timelines with different interior knots.
    #[test]
    fn ltra_attempts_preserve_stamps_and_separate_histories() {
        use super::{LoadRequest, ModelContext};
        use crate::devices::{
            Forcing, Limit, StateHistory, TransientTiming,
            delay::{DelayUpdate, WaveformUpdate},
            state::HistoryPublication,
        };
        use crate::maths::{IntegrationMethod, SparseMatrix, StepHistory, Vector};
        use crate::netlist::{Parser, source::parse_deck_text};
        use crate::primitives::Real;
        use std::path::Path;

        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("attempts.cir"),
                "LTRA preparations
o1 a 0 b 0 rc
o2 c 0 d 0 rc
o3 e 0 f 0 rc
o4 g 0 h 0 rc2
o5 i 0 j 0 rlc
o6 k 0 l 0 lc
o7 m 0 n 0 rg
.model rc ltra r=1 c=1 len=1
.model rc2 ltra r=1 c=1 len=1
.model rlc ltra r=0.1 l=1 c=1 len=1
.model lc ltra l=1 c=1 len=1
.model rg ltra r=1 g=1 len=1
.end
",
            ))
            .unwrap();
        let mut circuit = Circuit::from_netlist(&deck).unwrap();
        circuit.finalize().unwrap();
        circuit
            .set_observations(&["i(o1)".into(), "i(o2)".into()])
            .unwrap();
        let initialize = |history: &mut StateHistory, scale: Real| {
            let updates = circuit
                .devices()
                .iter()
                .enumerate()
                .map(|(ordinal, device)| {
                    let middle = if device.name() == "o3" { 0.75 } else { 1. };
                    let factor = scale * Real::from(u32::try_from(ordinal + 1).unwrap());
                    (
                        ordinal,
                        DelayUpdate::Waveform(WaveformUpdate {
                            reset: Some(
                                [0., middle, 2.]
                                    .into_iter()
                                    .map(|time| {
                                        (
                                            time,
                                            vec![
                                                factor * time,
                                                0.5 * factor * time,
                                                -factor * time * time,
                                                -0.25 * factor * time,
                                            ],
                                        )
                                    })
                                    .collect(),
                            ),
                            ..WaveformUpdate::default()
                        }),
                    )
                })
                .collect::<Vec<_>>();
            let trial = history.trial();
            history
                .accept_transaction(
                    trial,
                    HistoryPublication::Initialize,
                    |_, _| Ok(updates),
                    |_| Ok(()),
                )
                .unwrap();
        };
        let mut history = circuit.state_history().unwrap();
        initialize(&mut history, 1.);
        let original = history.clone();
        let run = circuit.prepare_run();
        let n = circuit.unknown_count();
        let forcing = Forcing {
            limit: Limit::Right,
            timing: TransientTiming::new(0.25, 4.).unwrap(),
        };
        let coefficients = StepHistory::with_fill(0.25)
            .trial(IntegrationMethod::Trapezoidal, 1, 0.25, 0.5)
            .unwrap();
        let model = ModelContext::default();
        let changed_gmin = model.with_gmin(1e-6);

        let compare = |history: &StateHistory, time: Real, context: &ModelContext| {
            let before = run.ltra_work();
            let mut attempt = run
                .transient_attempt(history, context, time, &coefficients, forcing)
                .unwrap();
            assert!(
                run.transient_attempt(history, context, time, &coefficients, forcing)
                    .is_err()
            );
            let mut result = Vec::new();
            for iteration in 0..3_u32 {
                let x = Vector::from_slice(
                    &(0..n)
                        .map(|row| {
                            0.01 * Real::from(u32::try_from(row).unwrap()) + Real::from(iteration)
                        })
                        .collect::<Vec<_>>(),
                );
                let request: LoadRequest<'_> = attempt.request(&x);
                let mut expected = SparseMatrix::new(n, n);
                let mut actual = SparseMatrix::new(n, n);
                let mut rhs_expected = Vector::zeros(n);
                let mut rhs_actual = Vector::zeros(n);
                let mut trial_expected = history.trial();
                let mut trial_actual = history.trial();
                run.load(
                    &request,
                    &mut expected,
                    &mut rhs_expected,
                    &mut trial_expected,
                )
                .unwrap();
                attempt
                    .load(&x, &mut actual, &mut rhs_actual, &mut trial_actual)
                    .unwrap();
                let signatures = |matrix: &SparseMatrix| {
                    matrix
                        .triplets()
                        .iter()
                        .map(|t| (t.row, t.col, t.value.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(signatures(&actual), signatures(&expected));
                let bits = |rhs: &Vector| {
                    rhs.as_slice()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&rhs_actual), bits(&rhs_expected));
                assert_eq!(trial_actual, trial_expected);
                let expected_obs = run
                    .observe_real(&request, &[], Some((time, forcing)), Some(&trial_expected))
                    .unwrap();
                let actual_obs = attempt.observe(&x, &trial_actual).unwrap();
                let obs_bits = |values: &[crate::primitives::Complex]| {
                    values
                        .iter()
                        .map(|value| (value.re.to_bits(), value.im.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(obs_bits(&actual_obs), obs_bits(&expected_obs));
                result = bits(&rhs_actual);
            }
            drop(attempt);
            let after = run.ltra_work();
            assert_eq!(after.attempts - before.attempts, 1);
            assert_eq!(after.newton_loads - before.newton_loads, 3);
            assert_eq!(after.line_loads - before.line_loads, 24); // 6 transient lines × 3, plus 2 isolated observations × 3
            assert_eq!(
                after.convolution_evaluations - before.convolution_evaluations,
                6
            );
            assert_eq!(
                after.coefficient_preparations - before.coefficient_preparations,
                4
            );
            result
        };

        let a = compare(&history, 2.25, &model);
        let b = compare(&history, 2.5, &model);
        assert_ne!(a, b);
        assert_eq!(a, compare(&history, 2.25, &model));
        assert_eq!(history, original);
        // Same times and same length, different accepted waveform contents.
        // A failed/rejected trial has not modified the original/fork.
        let mut fork = history.clone();
        fork.clear();
        initialize(&mut fork, 3.);
        assert_ne!(a, compare(&fork, 2.25, &model));
        assert_eq!(a, compare(&history, 2.25, &model));
        // Context changes must not retain a stale numeric stamp. The LTRA
        // inputs are unchanged by gmin, which is read by the stamp itself.
        compare(&history, 2.25, &changed_gmin);
        history.clear();
        initialize(&mut history, 2.);
        assert_ne!(a, compare(&history, 2.25, &model));
        assert_eq!(a, compare(&original, 2.25, &model));

        // A successful acceptance rotates the revision and extends the
        // waveform; the next proposal must evaluate the extended history.
        let mut advanced = original.clone();
        let updates = circuit
            .devices()
            .iter()
            .enumerate()
            .map(|(ordinal, _)| {
                (
                    ordinal,
                    DelayUpdate::Waveform(WaveformUpdate {
                        append: Some((2.125, vec![3.5, 1.5, -4.5, -2.5])),
                        ..WaveformUpdate::default()
                    }),
                )
            })
            .collect::<Vec<_>>();
        let trial = advanced.trial();
        advanced
            .accept_transaction(
                trial,
                HistoryPublication::Rotate,
                |_, _| Ok(updates),
                |_| Ok(()),
            )
            .unwrap();
        assert_ne!(a, compare(&advanced, 2.25, &model));

        // Merely opening an attempt performs no device work. Empty histories
        // still fail when stamping, just as with a direct one-off load.
        let empty = circuit.state_history().unwrap();
        let before = run.ltra_work();
        let mut proposal = run
            .transient_attempt(&empty, &model, 2.25, &coefficients, forcing)
            .unwrap();
        let x = Vector::zeros(n);
        let mut matrix = SparseMatrix::new(n, n);
        let mut rhs = Vector::zeros(n);
        let mut trial = empty.trial();
        let expected = run
            .load(&proposal.request(&x), &mut matrix, &mut rhs, &mut trial)
            .unwrap_err();
        let actual = proposal
            .load(&x, &mut matrix, &mut rhs, &mut trial)
            .unwrap_err();
        assert_eq!(actual.to_string(), expected.to_string());
        drop(proposal);
        let after = run.ltra_work();
        assert_eq!(
            after.coefficient_preparations,
            before.coefficient_preparations
        );
        assert_eq!(
            after.convolution_evaluations,
            before.convolution_evaluations
        );
        assert_eq!(a, compare(&original, 2.25, &model));
    }

    #[test]
    fn mutual_pair_and_payload_arithmetic_are_checked() {
        assert_eq!(super::pair_count(0), Some(0));
        assert_eq!(super::pair_count(1), Some(0));
        assert_eq!(super::pair_count(10_000), Some(49_995_000));
        assert_eq!(super::pair_count(usize::MAX), None);
        assert_eq!(super::mutual_storage_bytes(usize::MAX, 0, 0, 0), None);
        assert_eq!(super::mutual_storage_bytes(0, usize::MAX, 0, 0), None);
    }

    #[test]
    #[ignore = "manual coupling setup scaling measurement"]
    fn mutual_setup_scaling() {
        use crate::devices::{ModelContext, MutualLimits};
        use crate::netlist::{Parser, source::parse_deck_text};
        use std::{fmt::Write, path::Path, time::Instant};

        for (clique, nodes) in [
            (false, 1_000),
            (false, 10_000),
            (true, 100),
            (true, 300),
            (true, 1_000),
        ] {
            let mut deck = String::from("coupling benchmark\n");
            for i in 0..nodes {
                writeln!(deck, "l{i} n{i} 0 1").unwrap();
            }
            let (pairs, references, cards) = if clique {
                deck.push_str("k1");
                for i in 0..nodes {
                    write!(deck, " l{i}").unwrap();
                }
                deck.push_str(" 0.0005\n");
                (super::pair_count(nodes).unwrap(), nodes, 1)
            } else {
                for i in 1..nodes {
                    writeln!(deck, "k{i} l{} l{i} 0.5", i - 1).unwrap();
                }
                (nodes - 1, 2 * (nodes - 1), nodes - 1)
            };
            deck.push_str(".end\n");
            let netlist = Parser::new()
                .parse_deck(&parse_deck_text(Path::new("bench.cir"), &deck))
                .unwrap();
            let start = Instant::now();
            let circuit = Circuit::from_netlist_with_mutual_limits(
                &netlist,
                &ModelContext::default(),
                MutualLimits {
                    dense_bytes: 0,
                    ..MutualLimits::default()
                },
            )
            .unwrap();
            let elapsed = start.elapsed();
            assert_eq!(circuit.bindings.mutual.pairs.len(), pairs);
            let accounted =
                super::mutual_storage_bytes(pairs, references, cards, nodes + cards).unwrap();
            let dense_bytes = crate::maths::dense::symmetric_eigenvalue_cost(nodes)
                .unwrap()
                .0;
            eprintln!(
                "{} n={nodes} pairs={pairs} elapsed={elapsed:?} accounted_payload={accounted} dense_payload_avoided={dense_bytes}",
                if clique { "clique" } else { "chain" }
            );
        }
    }

    /// A test double with arbitrary terminals that never stamps.
    #[derive(Debug)]
    struct Stub {
        name: String,
        terminals: Vec<NodeId>,
        branch_currents: usize,
    }

    impl Device for Stub {
        fn name(&self) -> &str {
            &self.name
        }

        fn designator(&self) -> char {
            'r'
        }

        fn terminals(&self) -> &[NodeId] {
            &self.terminals
        }

        fn branch_currents(&self) -> usize {
            self.branch_currents
        }

        fn stamp(
            &self,
            _context: &mut crate::devices::traits::StampContext<'_>,
        ) -> SpiceResult<()> {
            Err(SpiceError::not_yet_ported("test stub", "tests"))
        }
    }

    fn stub(circuit: &mut Circuit, name: &str, a: &str, b: &str, branch_currents: usize) -> Stub {
        let first = circuit.add_node(a).unwrap();
        let second = circuit.add_node(b).unwrap();
        Stub {
            name: name.to_owned(),
            terminals: vec![first, second],
            branch_currents,
        }
    }

    #[test]
    fn coupling_cache_reuses_only_bitwise_identical_validated_values() {
        use crate::devices::ModelContext;
        use crate::netlist::{Parser, source::parse_deck_text};
        use std::{path::Path, sync::Arc};
        let deck = Parser::new()
            .parse_deck(&parse_deck_text(
                Path::new("coupling.cir"),
                "coupling\nl1 a 0 1m ic=0\nl2 b 0 4m\nk1 l1 l2 0.5\n.end\n",
            ))
            .unwrap();
        let context = ModelContext::default();
        let mut c = Circuit::from_netlist(&deck).unwrap();
        let first = c.prepared_mutual_terms(&context).unwrap();
        let same = c
            .prepared_mutual_terms(&ModelContext::new(57., 27.))
            .unwrap();
        assert!(Arc::ptr_eq(
            first.0.as_ref().unwrap(),
            same.0.as_ref().unwrap()
        ));
        let override_k = |value| {
            context
                .with_instance_override(c.instance_override("k1", "k", value, &context).unwrap())
                .unwrap()
        };
        let invalid = override_k(1.5);
        assert!(c.prepared_mutual_terms(&invalid).is_err());
        let retained = c.prepared_mutual_terms(&context).unwrap();
        assert!(Arc::ptr_eq(
            first.0.as_ref().unwrap(),
            retained.0.as_ref().unwrap()
        ));
        let changed = c.prepared_mutual_terms(&override_k(0.25)).unwrap();
        assert!(!Arc::ptr_eq(
            first.0.as_ref().unwrap(),
            changed.0.as_ref().unwrap()
        ));
        assert_ne!(&*first, &*changed);
        let zero = c.prepared_mutual_terms(&override_k(0.)).unwrap();
        let negative_zero = c.prepared_mutual_terms(&override_k(-0.)).unwrap();
        assert!(!Arc::ptr_eq(
            zero.0.as_ref().unwrap(),
            negative_zero.0.as_ref().unwrap()
        ));
        let repeated = c.prepared_mutual_terms(&context).unwrap();
        assert_eq!(&*repeated, &*first);
        c.finalize().unwrap();
        let rebuilt = c.prepared_mutual_terms(&context).unwrap();
        assert!(Arc::ptr_eq(
            repeated.0.as_ref().unwrap(),
            rebuilt.0.as_ref().unwrap()
        ));
    }

    #[test]
    fn inherited_state_checks_node_identity_and_shares_compatible_snapshots() {
        use crate::devices::ModelContext;
        use crate::maths::Vector;
        use std::sync::Arc;
        // A real capacitor supplies two state slots. A programmatic caller
        // can have an equal-size layout whose node IDs mean other nodes.
        let build = |name: &str| {
            let mut circuit = Circuit::new();
            let node = circuit.add_node(name).unwrap();
            circuit
                .add_device(Box::new(
                    crate::devices::Capacitor::new("c1", [node, NodeId::GROUND], 1., None).unwrap(),
                ))
                .unwrap();
            circuit.finalize().unwrap();
            circuit
        };
        let original = build("a");
        original
            .record_small_signal_state(&ModelContext::default(), &Vector::zeros(1))
            .unwrap();
        let mut history = original.state_history().unwrap();
        let mut trial = history.trial();
        let mut state = history.device_of(&mut trial, 0).unwrap();
        state.set(0, 1.).unwrap();
        state.set(1, 2.).unwrap();
        history.commit(trial).unwrap();
        original.record_final_transient_state(&history).unwrap();
        let snapshot = original.runtime.final_transient.borrow();
        let snapshot = snapshot.as_ref().unwrap();
        assert!(Arc::ptr_eq(
            &snapshot.values,
            &history.accepted_snapshot(1).unwrap()
        ));
        assert_eq!(
            snapshot.topology,
            original
                .runtime
                .carried_state
                .borrow()
                .as_ref()
                .unwrap()
                .topology
        );

        let mut compatible = build("a");
        compatible.inherit_from(&original);
        assert_eq!(compatible.final_transient_state(), Some(vec![1., 2.]));
        assert!(Arc::ptr_eq(
            &compatible
                .runtime
                .carried_state
                .borrow()
                .as_ref()
                .unwrap()
                .values,
            &original
                .runtime
                .carried_state
                .borrow()
                .as_ref()
                .unwrap()
                .values
        ));
        let mut foreign = build("different");
        foreign.inherit_from(&original);
        assert!(foreign.final_transient_state().is_none());
        assert!(foreign.runtime.carried_state.borrow().is_none());
        // Device mutation can invalidate a snapshot before re-finalization.
        compatible
            .replace_device(
                0,
                Box::new(
                    crate::devices::Capacitor::new(
                        "other",
                        [NodeId::new(1), NodeId::GROUND],
                        1.,
                        None,
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
        assert!(compatible.final_transient_state().is_none());
        assert!(compatible.state_history().is_err());
        compatible.finalize().unwrap();
        let history = compatible.state_history().unwrap();
        let mut trial = history.trial();
        assert_eq!(history.device_of(&mut trial, 0).unwrap().carried(0), None);
    }

    #[test]
    fn unknowns_are_renumbered_after_every_device_is_added() {
        let mut circuit = Circuit::new();
        let device = stub(&mut circuit, "r1", "in", "out", 0);
        circuit.add_device(Box::new(device)).unwrap();
        let source = stub(&mut circuit, "v1", "in", "0", 1);
        circuit.add_device(Box::new(source)).unwrap();

        circuit.finalize().unwrap();
        // Two nodes plus one branch current.
        assert_eq!(circuit.unknown_count(), 3);
        assert_eq!(circuit.device_count(), 2);
        assert_eq!(circuit.device("R1").map(Device::name), Some("r1"));
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut circuit = Circuit::new();
        let first = stub(&mut circuit, "r1", "a", "b", 0);
        circuit.add_device(Box::new(first)).unwrap();
        let second = stub(&mut circuit, "R1", "a", "b", 0);
        let error = circuit.add_device(Box::new(second)).unwrap_err();
        assert!(error.to_string().contains("duplicate instance name"));
    }

    #[test]
    fn dangling_terminals_are_rejected() {
        let mut circuit = Circuit::new();
        let known = circuit.add_node("a").unwrap();
        let dangling = NodeId::new(99);
        let device = Stub {
            name: "r1".to_owned(),
            terminals: vec![known, dangling],
            branch_currents: 0,
        };
        let error = circuit.add_device(Box::new(device)).unwrap_err();
        assert!(error.to_string().contains("not in the node table"));
    }

    #[test]
    fn finalize_catches_devices_added_behind_its_back() {
        let mut circuit = Circuit::new();
        let device = stub(&mut circuit, "r1", "a", "b", 0);
        circuit.add_device(Box::new(device)).unwrap();
        circuit.finalize().unwrap();
        assert_eq!(circuit.unknown_count(), 2);

        // A node added afterwards shifts nothing until the next rebuild.
        circuit.add_node("late").unwrap();
        circuit.finalize().unwrap();
        assert_eq!(circuit.unknown_count(), 3);
    }

    #[test]
    fn debug_output_lists_device_names() {
        let mut circuit = Circuit::new();
        let device = stub(&mut circuit, "r1", "a", "b", 0);
        circuit.add_device(Box::new(device)).unwrap();
        let text = format!("{circuit:?}");
        assert!(text.contains("r1"), "{text}");
        assert!(text.contains("nodes: 3"), "{text}");
    }

    fn vertex(graph: &CircuitGraph, weight: CircuitVertex) -> NodeIndex<usize> {
        graph
            .node_indices()
            .find(|index| graph[*index] == weight)
            .unwrap()
    }

    fn ports(graph: &CircuitGraph, device: usize) -> Vec<(usize, CircuitVertex)> {
        let index = vertex(graph, CircuitVertex::Device(device));
        let mut ports: Vec<_> = graph
            .edges(index)
            .map(|edge| (*edge.weight(), graph[edge.target()]))
            .collect();
        ports.sort_by_key(|(port, _)| *port);
        ports
    }

    #[test]
    fn topology_keeps_ground_and_unused_nodes() {
        let mut circuit = Circuit::new();
        let empty = circuit.topology().unwrap();
        assert_eq!(empty.node_count(), 1);
        assert_eq!(empty.edge_count(), 0);
        assert_eq!(
            empty.node_weights().copied().collect::<Vec<_>>(),
            vec![CircuitVertex::Node(NodeId::GROUND)]
        );
        let unused = circuit.add_node("unused").unwrap();
        let graph = circuit.topology().unwrap();
        assert_eq!(graph.node_count(), 2);
        assert_eq!(connected_components(&graph), 2);
        assert_eq!(
            graph[vertex(&graph, CircuitVertex::Node(unused))],
            CircuitVertex::Node(unused)
        );
        // Connectivity alone is not grounds for rejecting a circuit here.
        circuit.finalize().unwrap();
        assert_eq!(circuit.unknown_count(), 1);
    }

    #[test]
    fn topology_distinguishes_node_ids_device_ordinals_and_mna_rows() {
        let mut circuit = Circuit::new();
        let resistor = stub(&mut circuit, "r1", "in", "out", 0);
        let [input, output] = [resistor.terminals[0], resistor.terminals[1]];
        circuit.add_device(Box::new(resistor)).unwrap();
        let source = stub(&mut circuit, "v1", "in", "GND", 1);
        circuit.add_device(Box::new(source)).unwrap();
        circuit.finalize().unwrap();
        let graph = circuit.topology().unwrap();
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4);
        assert_eq!(connected_components(&graph), 1);
        assert_eq!(
            ports(&graph, 0),
            vec![
                (0, CircuitVertex::Node(input)),
                (1, CircuitVertex::Node(output))
            ]
        );
        assert_eq!(
            ports(&graph, 1),
            vec![
                (0, CircuitVertex::Node(input)),
                (1, CircuitVertex::Node(NodeId::GROUND))
            ]
        );
        assert_ne!(
            vertex(&graph, CircuitVertex::Device(0)),
            vertex(&graph, CircuitVertex::Node(NodeId::GROUND))
        );
        assert_eq!(circuit.unknowns().node_row(input), Some(0));
        assert_eq!(circuit.unknowns().node_row(NodeId::GROUND), None);
        assert_eq!(circuit.unknown_count(), 3);
    }

    #[test]
    fn multiport_and_repeated_terminals_keep_parallel_edges_and_ordinals() {
        let mut circuit = Circuit::new();
        let output = circuit.add_node("out").unwrap();
        let ground = circuit.add_node("gnd").unwrap();
        circuit
            .add_device(Box::new(Stub {
                name: "m1".to_owned(),
                terminals: vec![ground, output, output, ground],
                branch_currents: 0,
            }))
            .unwrap();
        let graph = circuit.topology().unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 4);
        assert_eq!(
            ports(&graph, 0),
            vec![
                (0, CircuitVertex::Node(ground)),
                (1, CircuitVertex::Node(output)),
                (2, CircuitVertex::Node(output)),
                (3, CircuitVertex::Node(ground)),
            ]
        );
    }

    #[test]
    fn petgraph_finds_structurally_separate_circuit_blocks() {
        let mut circuit = Circuit::new();
        let grounded = stub(&mut circuit, "c1", "out", "0", 0);
        let output = grounded.terminals[0];
        circuit.add_device(Box::new(grounded)).unwrap();
        let isolated = stub(&mut circuit, "r2", "a", "b", 0);
        let island = isolated.terminals[0];
        circuit.add_device(Box::new(isolated)).unwrap();
        let graph = circuit.topology().unwrap();
        let ground = vertex(&graph, CircuitVertex::Node(NodeId::GROUND));
        assert_eq!(connected_components(&graph), 2);
        assert!(has_path_connecting(
            &graph,
            ground,
            vertex(&graph, CircuitVertex::Node(output)),
            None
        ));
        assert!(!has_path_connecting(
            &graph,
            ground,
            vertex(&graph, CircuitVertex::Node(island)),
            None
        ));
        circuit.finalize().unwrap(); // structural islands are not numerical diagnostics
    }

    #[test]
    fn zero_port_devices_are_kept_without_inventing_matrix_row_vertices() {
        let mut circuit = Circuit::new();
        circuit
            .add_device(Box::new(Stub {
                name: "internal".to_owned(),
                terminals: vec![],
                branch_currents: 2,
            }))
            .unwrap();
        circuit.finalize().unwrap();
        let graph = circuit.topology().unwrap();
        assert_eq!(graph.node_count(), 2); // ground and the device, not its rows
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(connected_components(&graph), 2);
        assert_eq!(circuit.unknown_count(), 2);
        assert_eq!(
            graph[vertex(&graph, CircuitVertex::Device(0))],
            CircuitVertex::Device(0)
        );
    }

    #[test]
    fn topology_is_rebuilt_after_node_and_device_mutation() {
        let mut circuit = Circuit::new();
        let device = stub(&mut circuit, "r1", "a", "0", 0);
        let a = device.terminals[0];
        circuit.add_device(Box::new(device)).unwrap();
        let before = circuit.topology().unwrap();
        circuit.add_node("late").unwrap();
        circuit
            .replace_device(
                0,
                Box::new(Stub {
                    name: "r1".to_owned(),
                    terminals: vec![a, a],
                    branch_currents: 0,
                }),
            )
            .unwrap();
        let after = circuit.topology().unwrap();
        assert_eq!(connected_components(&before), 1);
        assert_eq!(connected_components(&after), 3);
        assert_eq!(before.node_count(), 3);
        assert_eq!(after.node_count(), 4);
        assert_ne!(
            vertex(&before, CircuitVertex::Device(0)),
            vertex(&after, CircuitVertex::Device(0))
        );
        assert_eq!(
            ports(&after, 0),
            vec![(0, CircuitVertex::Node(a)), (1, CircuitVertex::Node(a))]
        );
    }

    #[test]
    fn mutated_duplicate_names_fail_without_changing_numbering() {
        let mut circuit = Circuit::new();
        let first = stub(&mut circuit, "r1", "a", "0", 1);
        circuit.add_device(Box::new(first)).unwrap();
        let second = stub(&mut circuit, "r2", "b", "0", 1);
        let b = second.terminals[0];
        circuit.add_device(Box::new(second)).unwrap();
        circuit.finalize().unwrap();
        circuit.add_node("late").unwrap();
        circuit
            .replace_device(
                1,
                Box::new(Stub {
                    name: "R1".to_owned(),
                    terminals: vec![b, NodeId::GROUND],
                    branch_currents: 5,
                }),
            )
            .unwrap();
        assert!(
            circuit
                .topology()
                .unwrap_err()
                .to_string()
                .contains("duplicate instance name")
        );
        assert!(
            circuit
                .finalize()
                .unwrap_err()
                .to_string()
                .contains("duplicate instance name")
        );
        assert!(circuit.new_matrix().is_err());
        assert!(circuit.state_history().is_err());
        assert_eq!(circuit.unknown_count(), 0);
    }

    #[test]
    fn mutated_dangling_terminals_fail_without_changing_numbering() {
        let mut circuit = Circuit::new();
        let device = stub(&mut circuit, "r1", "a", "0", 0);
        let a = device.terminals[0];
        circuit.add_device(Box::new(device)).unwrap();
        circuit.finalize().unwrap();
        circuit
            .replace_device(
                0,
                Box::new(Stub {
                    name: "bad".to_owned(),
                    terminals: vec![a, NodeId::new(99)],
                    branch_currents: 5,
                }),
            )
            .unwrap();
        let error = circuit.topology().unwrap_err();
        assert!(matches!(error, SpiceError::Circuit { .. }));
        assert!(error.to_string().contains("node 99"));
        assert!(
            circuit
                .finalize()
                .unwrap_err()
                .to_string()
                .contains("not in the node table")
        );
        assert!(circuit.new_matrix().is_err());
        assert!(circuit.state_history().is_err());
        assert_eq!(circuit.unknown_count(), 0);
    }

    #[test]
    fn topology_preserves_deck_order_and_is_deterministic() {
        let mut circuit = Circuit::new();
        for name in ["z1", "a1"] {
            let device = stub(&mut circuit, name, "a", "b", 0);
            circuit.add_device(Box::new(device)).unwrap();
        }
        let a = circuit.nodes().get("a").unwrap();
        let b = circuit.nodes().get("b").unwrap();
        let first = circuit.topology().unwrap();
        let second = circuit.topology().unwrap();
        let expected = vec![
            CircuitVertex::Node(NodeId::GROUND),
            CircuitVertex::Node(a),
            CircuitVertex::Node(b),
            CircuitVertex::Device(0),
            CircuitVertex::Device(1),
        ];
        assert_eq!(first.node_weights().copied().collect::<Vec<_>>(), expected);
        assert_eq!(second.node_weights().copied().collect::<Vec<_>>(), expected);
        for device in 0..2 {
            assert_eq!(ports(&first, device), ports(&second, device));
        }
        assert_eq!(circuit.devices()[0].name(), "z1");
        assert_eq!(circuit.devices()[1].name(), "a1");
    }
}
